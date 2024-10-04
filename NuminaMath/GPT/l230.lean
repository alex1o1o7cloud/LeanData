import Mathlib

namespace math_problem_l230_230355

theorem math_problem (a b n r : ℕ) (h₁ : 1853 ≡ 53 [MOD 600]) (h₂ : 2101 ≡ 101 [MOD 600]) :
  (1853 * 2101) ≡ 553 [MOD 600] := by
  sorry

end math_problem_l230_230355


namespace card_probability_l230_230128

-- Define the total number of cards
def total_cards : ℕ := 52

-- Define the number of Kings in the deck
def kings_in_deck : ℕ := 4

-- Define the number of Aces in the deck
def aces_in_deck : ℕ := 4

-- Define the probability of the top card being a King
def prob_top_king : ℚ := kings_in_deck / total_cards

-- Define the probability of the second card being an Ace given the first card is a King
def prob_second_ace_given_king : ℚ := aces_in_deck / (total_cards - 1)

-- Define the combined probability of both events happening in sequence
def combined_probability : ℚ := prob_top_king * prob_second_ace_given_king

-- Theorem statement that the combined probability is equal to 4/663
theorem card_probability : combined_probability = 4 / 663 := by
  -- Proof to be filled in
  sorry

end card_probability_l230_230128


namespace total_bricks_in_wall_l230_230705

theorem total_bricks_in_wall :
  let bottom_row_bricks := 18
  let rows := [bottom_row_bricks, bottom_row_bricks - 1, bottom_row_bricks - 2, bottom_row_bricks - 3, bottom_row_bricks - 4]
  (rows.sum = 80) := 
by
  let bottom_row_bricks := 18
  let rows := [bottom_row_bricks, bottom_row_bricks - 1, bottom_row_bricks - 2, bottom_row_bricks - 3, bottom_row_bricks - 4]
  sorry

end total_bricks_in_wall_l230_230705


namespace cows_grazed_by_C_l230_230784

-- Define the initial conditions as constants
def cows_grazed_A : ℕ := 24
def months_grazed_A : ℕ := 3
def cows_grazed_B : ℕ := 10
def months_grazed_B : ℕ := 5
def cows_grazed_D : ℕ := 21
def months_grazed_D : ℕ := 3
def share_rent_A : ℕ := 1440
def total_rent : ℕ := 6500

-- Define the cow-months calculation for A, B, D
def cow_months_A : ℕ := cows_grazed_A * months_grazed_A
def cow_months_B : ℕ := cows_grazed_B * months_grazed_B
def cow_months_D : ℕ := cows_grazed_D * months_grazed_D

-- Let x be the number of cows grazed by C
variable (x : ℕ)

-- Define the cow-months calculation for C
def cow_months_C : ℕ := x * 4

-- Define rent per cow-month
def rent_per_cow_month : ℕ := share_rent_A / cow_months_A

-- Proof problem statement
theorem cows_grazed_by_C : 
  (6500 = (cow_months_A + cow_months_B + cow_months_C x + cow_months_D) * rent_per_cow_month) →
  x = 35 := by
  sorry

end cows_grazed_by_C_l230_230784


namespace committee_size_l230_230387

theorem committee_size (n : ℕ)
  (h : ((n - 2 : ℕ) : ℚ) / ((n - 1) * (n - 2) / 2 : ℚ) = 0.4) :
  n = 6 :=
by
  sorry

end committee_size_l230_230387


namespace number_of_pieces_l230_230984

def length_piece : ℝ := 0.40
def total_length : ℝ := 47.5

theorem number_of_pieces : ⌊total_length / length_piece⌋ = 118 := by
  sorry

end number_of_pieces_l230_230984


namespace archer_hits_less_than_8_l230_230130

variables (P10 P9 P8 : ℝ)

-- Conditions
def hitting10_ring := P10 = 0.3
def hitting9_ring := P9 = 0.3
def hitting8_ring := P8 = 0.2

-- Statement to prove
theorem archer_hits_less_than_8 (P10 P9 P8 : ℝ)
  (h10 : hitting10_ring P10)
  (h9 : hitting9_ring P9)
  (h8 : hitting8_ring P8)
  (mutually_exclusive: P10 + P9 + P8 <= 1):
  1 - (P10 + P9 + P8) = 0.2 :=
by
  -- Here goes the proof 
  sorry

end archer_hits_less_than_8_l230_230130


namespace geometric_sequence_value_l230_230999

theorem geometric_sequence_value (a : ℕ → ℝ) (h : ∀ n, a n > 0)
  (h_geometric : ∀ n, a (n+2) = a (n+1) * (a (n+1) / a n)) :
  a 3 * a 5 = 4 → a 4 = 2 :=
by
  sorry

end geometric_sequence_value_l230_230999


namespace negation_of_existential_proposition_l230_230164

theorem negation_of_existential_proposition :
  (¬ (∃ x : ℝ, x > Real.sin x)) ↔ (∀ x : ℝ, x ≤ Real.sin x) :=
by 
  sorry

end negation_of_existential_proposition_l230_230164


namespace problem_I_solution_set_problem_II_range_a_l230_230664

-- Problem (I)
-- Given f(x) = |x-1|, g(x) = 2|x+1|, and a=1, prove that the inequality f(x) - g(x) > 1 has the solution set (-1, -1/3)
theorem problem_I_solution_set (x: ℝ) : abs (x - 1) - 2 * abs (x + 1) > 1 ↔ -1 < x ∧ x < -1 / 3 := 
by sorry

-- Problem (II)
-- Given f(x) = |x-1|, g(x) = 2|x+a|, prove that if 2f(x) + g(x) ≤ (a + 1)^2 has a solution for x,
-- then a ∈ (-∞, -3] ∪ [1, ∞)
theorem problem_II_range_a (a x: ℝ) (h : ∃ x, 2 * abs (x - 1) + 2 * abs (x + a) ≤ (a + 1) ^ 2) : 
  a ≤ -3 ∨ a ≥ 1 := 
by sorry

end problem_I_solution_set_problem_II_range_a_l230_230664


namespace gcd_256_180_600_l230_230630

theorem gcd_256_180_600 : Nat.gcd (Nat.gcd 256 180) 600 = 4 :=
by
  -- This proof is marked as 'sorry' to indicate it is a place holder
  sorry

end gcd_256_180_600_l230_230630


namespace value_of_M_l230_230060

theorem value_of_M :
  let row_seq := [25, 25 + (8 - 25) / 3, 25 + 2 * (8 - 25) / 3, 8, 8 + (8 - 25) / 3, 8 + 2 * (8 - 25) / 3, -9]
  let col_seq1 := [25, 25 - 4, 25 - 8]
  let col_seq2 := [16, 20, 20 + 4]
  let col_seq3 := [-9, -9 - 11/4, -9 - 2 * 11/4, -20]
  let M := -9 - (-11/4)
  M = -6.25 :=
by
  sorry

end value_of_M_l230_230060


namespace bus_passengers_total_l230_230886

theorem bus_passengers_total (children_percent : ℝ) (adults_number : ℝ) (H1 : children_percent = 0.25) (H2 : adults_number = 45) :
  ∃ T : ℝ, T = 60 :=
by
  sorry

end bus_passengers_total_l230_230886


namespace combined_yearly_return_percentage_l230_230642

-- Given conditions
def investment1 : ℝ := 500
def return_rate1 : ℝ := 0.07
def investment2 : ℝ := 1500
def return_rate2 : ℝ := 0.15

-- Question to prove
theorem combined_yearly_return_percentage :
  let yearly_return1 := investment1 * return_rate1
  let yearly_return2 := investment2 * return_rate2
  let total_yearly_return := yearly_return1 + yearly_return2
  let total_investment := investment1 + investment2
  ((total_yearly_return / total_investment) * 100) = 13 :=
by
  -- skipping the proof
  sorry

end combined_yearly_return_percentage_l230_230642


namespace cistern_width_l230_230122

theorem cistern_width (l d A : ℝ) (h_l: l = 5) (h_d: d = 1.25) (h_A: A = 42.5) :
  ∃ w : ℝ, 5 * w + 2 * (1.25 * 5) + 2 * (1.25 * w) = 42.5 ∧ w = 4 :=
by
  use 4
  sorry

end cistern_width_l230_230122


namespace zoe_spent_amount_l230_230797

def flower_price : ℕ := 3
def roses_bought : ℕ := 8
def daisies_bought : ℕ := 2

theorem zoe_spent_amount :
  roses_bought + daisies_bought = 10 ∧
  flower_price = 3 →
  (roses_bought + daisies_bought) * flower_price = 30 :=
by
  sorry

end zoe_spent_amount_l230_230797


namespace simplify_expression_l230_230594

theorem simplify_expression (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h : a^4 + b^4 = a^2 + b^2) :
  (a / b + b / a - 1 / (a * b)) = 3 :=
  sorry

end simplify_expression_l230_230594


namespace lance_more_pebbles_l230_230662

-- Given conditions
def candy_pebbles : ℕ := 4
def lance_pebbles : ℕ := 3 * candy_pebbles

-- Proof statement
theorem lance_more_pebbles : lance_pebbles - candy_pebbles = 8 :=
by
  sorry

end lance_more_pebbles_l230_230662


namespace find_z_l230_230031

theorem find_z (x y k : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hk : k ≠ 0) (h : 1/x + 1/y = k) :
  ∃ z : ℝ, 1/z = k ∧ z = xy/(x + y) :=
by {
  sorry
}

end find_z_l230_230031


namespace least_whole_number_l230_230496

theorem least_whole_number (n : ℕ) 
  (h1 : n % 2 = 1)
  (h2 : n % 3 = 1)
  (h3 : n % 4 = 1)
  (h4 : n % 5 = 1)
  (h5 : n % 6 = 1)
  (h6 : 7 ∣ n) : 
  n = 301 := 
sorry

end least_whole_number_l230_230496


namespace probability_of_sum_leq_9_on_two_dice_is_5_over_6_l230_230623

def probability_sum_leq_9 (n : ℕ) (m : ℕ) : ℚ :=
  if n ∈ {1, 2, 3, 4, 5, 6} ∧ m ∈ {1, 2, 3, 4, 5, 6}
  then (36 - 6) / 36 else 0 -- considering the total favorable outcomes (30 out of 36)

theorem probability_of_sum_leq_9_on_two_dice_is_5_over_6 :
  probability_sum_leq_9 6 6 = 5 / 6 :=
by 
  sorry

end probability_of_sum_leq_9_on_two_dice_is_5_over_6_l230_230623


namespace gcd_256_180_600_l230_230628

theorem gcd_256_180_600 : Nat.gcd (Nat.gcd 256 180) 600 = 4 :=
by
  -- This proof is marked as 'sorry' to indicate it is a place holder
  sorry

end gcd_256_180_600_l230_230628


namespace total_games_is_seven_l230_230888

def total_football_games (games_missed : ℕ) (games_attended : ℕ) : ℕ :=
  games_missed + games_attended

theorem total_games_is_seven : total_football_games 4 3 = 7 := 
by
  sorry

end total_games_is_seven_l230_230888


namespace find_y_l230_230764

theorem find_y : ∃ y : ℝ, (7 / 3) * y = 42 ∧ y = 18 :=
by
  use 18
  split
  · norm_num
  · norm_num

end find_y_l230_230764


namespace right_triangle_inequality_l230_230191

theorem right_triangle_inequality {a b c : ℝ} (h₁ : a^2 + b^2 = c^2) : a + b ≤ c * Real.sqrt 2 := by
  sorry

end right_triangle_inequality_l230_230191


namespace total_meals_per_week_l230_230559

-- Definitions of the conditions
def meals_per_day_r1 : ℕ := 20
def meals_per_day_r2 : ℕ := 40
def meals_per_day_r3 : ℕ := 50
def days_per_week : ℕ := 7

-- The proof goal
theorem total_meals_per_week : 
  (meals_per_day_r1 * days_per_week) + 
  (meals_per_day_r2 * days_per_week) + 
  (meals_per_day_r3 * days_per_week) = 770 :=
by
  sorry

end total_meals_per_week_l230_230559


namespace average_visitors_on_sundays_l230_230378

theorem average_visitors_on_sundays 
  (avg_other_days : ℕ) (avg_per_day : ℕ) (days_in_month : ℕ) (sundays : ℕ) (S : ℕ)
  (h_avg_other_days : avg_other_days = 240)
  (h_avg_per_day : avg_per_day = 310)
  (h_days_in_month : days_in_month = 30)
  (h_sundays : sundays = 5) :
  (sundays * S + (days_in_month - sundays) * avg_other_days = avg_per_day * days_in_month) → 
  S = 660 :=
by
  intros h
  rw [h_avg_other_days, h_avg_per_day, h_days_in_month, h_sundays] at h
  sorry

end average_visitors_on_sundays_l230_230378


namespace sasha_remainder_is_20_l230_230080

theorem sasha_remainder_is_20 (n a b c d : ℤ) (h1 : n = 102 * a + b) (h2 : n = 103 * c + d) (h3 : d = 20 - a) : b = 20 :=
by
  sorry

end sasha_remainder_is_20_l230_230080


namespace AMHSE_1988_l230_230427

theorem AMHSE_1988 (x y : ℝ) (h1 : |x| + x + y = 10) (h2 : x + |y| - y = 12) : x + y = 18 / 5 :=
sorry

end AMHSE_1988_l230_230427


namespace convert_degrees_to_radians_l230_230138

theorem convert_degrees_to_radians (θ : ℝ) (h : θ = -630) : θ * (Real.pi / 180) = -7 * Real.pi / 2 := by
  sorry

end convert_degrees_to_radians_l230_230138


namespace problem_solution_l230_230972

open Function

-- Definitions of the points
structure Point where
  x : ℝ
  y : ℝ

def A : Point := ⟨-3, 2⟩
def B : Point := ⟨1, 0⟩
def C : Point := ⟨4, 1⟩
def D : Point := ⟨-2, 4⟩

-- Definitions of vectors
def vec (P Q : Point) : Point := ⟨Q.x - P.x, Q.y - P.y⟩

-- Definitions of conditions
def AB := vec A B
def AD := vec A D
def DC := vec D C

-- Definitions of dot product to check orthogonality
def dot (v w : Point) : ℝ := v.x * w.x + v.y * w.y

-- Lean statement to prove the conditions
theorem problem_solution :
  AB ≠ ⟨-4, 2⟩ ∧
  dot AB AD = 0 ∧
  AB.y * DC.x = AB.x * DC.y ∧
  ((AB.y * DC.x = AB.x * DC.y) ∧ (dot AB AD = 0) → 
  (∃ a b : ℝ, a ≠ b ∧ (a = 0 ∨ b = 0) ∧ AB = ⟨a, -a⟩  ∧ DC = ⟨3 * a, -3 * a⟩)) :=
by
  -- Proof omitted
  sorry

end problem_solution_l230_230972


namespace sasha_remainder_l230_230067

theorem sasha_remainder (n a b c d : ℕ) (h1 : n = 102 * a + b) (h2 : n = 103 * c + d) 
  (h3 : a + d = 20) (h_b_range : 0 ≤ b ∧ b ≤ 101) (h_d_range : 0 ≤ d ∧ d ≤ 102) : b = 20 := 
sorry

end sasha_remainder_l230_230067


namespace find_p_l230_230808

-- Conditions: Consider the quadratic equation 2x^2 + px + q = 0 where p and q are integers.
-- Roots of the equation differ by 2.
-- q = 4

theorem find_p (p : ℤ) (q : ℤ) (h1 : q = 4) (h2 : ∃ x₁ x₂ : ℝ, 2 * x₁^2 + p * x₁ + q = 0 ∧ 2 * x₂^2 + p * x₂ + q = 0 ∧ |x₁ - x₂| = 2) :
  p = 7 ∨ p = -7 :=
by
  sorry

end find_p_l230_230808


namespace gcd_min_value_l230_230304

-- Definitions of the conditions
def is_positive_integer (x : ℕ) := x > 0

def gcd_cond (m n : ℕ) := Nat.gcd m n = 18

-- The main theorem statement
theorem gcd_min_value (m n : ℕ) (hm : is_positive_integer m) (hn : is_positive_integer n) (hgcd : gcd_cond m n) : 
  Nat.gcd (12 * m) (20 * n) = 72 :=
sorry

end gcd_min_value_l230_230304


namespace identify_heaviest_and_lightest_13_weighings_l230_230100

theorem identify_heaviest_and_lightest_13_weighings (coins : Fin 10 → ℝ) (h_distinct : Function.Injective coins) :
  ∃ f : (Fin 13 → ((Fin 10) × (Fin 10) × ℝ)), true :=
by
  sorry

end identify_heaviest_and_lightest_13_weighings_l230_230100


namespace todd_saved_44_dollars_l230_230349

-- Definitions of the conditions.
def full_price : ℝ := 125
def sale_discount : ℝ := 0.20
def coupon_discount : ℝ := 10
def credit_card_discount : ℝ := 0.10

-- The statement we want to prove: Todd saved $44 on the original price of the jeans.
theorem todd_saved_44_dollars :
  let sale_amount := full_price * sale_discount,
      price_after_sale := full_price - sale_amount,
      price_after_coupon := price_after_sale - coupon_discount,
      credit_card_amount := price_after_coupon * credit_card_discount,
      final_price := price_after_coupon - credit_card_amount,
      savings := full_price - final_price
  in savings = 44 :=
by
  sorry

end todd_saved_44_dollars_l230_230349


namespace cannot_pay_exactly_500_can_pay_exactly_600_l230_230204

-- Defining the costs and relevant equations
def price_of_bun : ℕ := 15
def price_of_croissant : ℕ := 12

-- Proving the non-existence for the 500 Ft case
theorem cannot_pay_exactly_500 : ¬ ∃ (x y : ℕ), price_of_croissant * x + price_of_bun * y = 500 :=
sorry

-- Proving the existence for the 600 Ft case
theorem can_pay_exactly_600 : ∃ (x y : ℕ), price_of_croissant * x + price_of_bun * y = 600 :=
sorry

end cannot_pay_exactly_500_can_pay_exactly_600_l230_230204


namespace all_three_selected_l230_230347

-- Define the probabilities
def P_R : ℚ := 6 / 7
def P_Rv : ℚ := 1 / 5
def P_Rs : ℚ := 2 / 3
def P_Rv_given_R : ℚ := 2 / 5
def P_Rs_given_Rv : ℚ := 1 / 2

-- The probability that all three are selected
def P_all : ℚ := P_R * P_Rv_given_R * P_Rs_given_Rv

-- Prove that the calculated probability is equal to the given answer
theorem all_three_selected : P_all = 6 / 35 :=
by
  sorry

end all_three_selected_l230_230347


namespace samantha_eggs_left_l230_230039

variables (initial_eggs : ℕ) (total_cost price_per_egg : ℝ)

-- Conditions
def samantha_initial_eggs : initial_eggs = 30 := sorry
def samantha_total_cost : total_cost = 5 := sorry
def samantha_price_per_egg : price_per_egg = 0.20 := sorry

-- Theorem to prove:
theorem samantha_eggs_left : 
  initial_eggs - (total_cost / price_per_egg) = 5 := 
  by
  rw [samantha_initial_eggs, samantha_total_cost, samantha_price_per_egg]
  -- Completing the arithmetic proof
  rw [Nat.cast_sub (by norm_num), Nat.cast_div (by norm_num), Nat.cast_mul (by norm_num)]
  norm_num
  sorry

end samantha_eggs_left_l230_230039


namespace average_headcount_is_11033_l230_230141

def average_headcount (count1 count2 count3 : ℕ) : ℕ :=
  (count1 + count2 + count3) / 3

theorem average_headcount_is_11033 :
  average_headcount 10900 11500 10700 = 11033 :=
by
  sorry

end average_headcount_is_11033_l230_230141


namespace charity_ticket_sales_l230_230643

theorem charity_ticket_sales
  (x y p : ℕ)
  (h1 : x + y = 200)
  (h2 : x * p + y * (p / 2) = 3501)
  (h3 : x = 3 * y) :
  150 * 20 = 3000 :=
by
  sorry

end charity_ticket_sales_l230_230643


namespace chord_length_l230_230109

theorem chord_length (R a b : ℝ) (hR : a + b = R) (hab : a * b = 10) 
    (h_nonneg : 0 ≤ R ∧ 0 ≤ a ∧ 0 ≤ b) : ∃ L : ℝ, L = 2 * Real.sqrt 10 :=
by
  sorry

end chord_length_l230_230109


namespace Berry_Temperature_Friday_l230_230661

theorem Berry_Temperature_Friday (temps : Fin 6 → ℝ) (avg_temp : ℝ) (total_days : ℕ) (friday_temp : ℝ) :
  temps 0 = 99.1 → 
  temps 1 = 98.2 →
  temps 2 = 98.7 →
  temps 3 = 99.3 →
  temps 4 = 99.8 →
  temps 5 = 98.9 →
  avg_temp = 99 →
  total_days = 7 →
  friday_temp = (avg_temp * total_days) - (temps 0 + temps 1 + temps 2 + temps 3 + temps 4 + temps 5) →
  friday_temp = 99 :=
by 
  intros h0 h1 h2 h3 h4 h5 h_avg h_days h_friday
  sorry

end Berry_Temperature_Friday_l230_230661


namespace distance_between_trees_l230_230018

theorem distance_between_trees 
  (rows columns : ℕ)
  (boundary_distance garden_length d : ℝ)
  (h_rows : rows = 10)
  (h_columns : columns = 12)
  (h_boundary_distance : boundary_distance = 5)
  (h_garden_length : garden_length = 32) :
  (9 * d + 2 * boundary_distance = garden_length) → 
  d = 22 / 9 := 
by 
  intros h_eq
  sorry

end distance_between_trees_l230_230018


namespace find_group_2018_l230_230961

theorem find_group_2018 :
  ∃ n : ℕ, 2 ≤ n ∧ 2018 ≤ 2 * n * (n + 1) ∧ 2018 > 2 * (n - 1) * n :=
by
  sorry

end find_group_2018_l230_230961


namespace nonoverlapping_area_difference_l230_230236

theorem nonoverlapping_area_difference :
  let radius := 3
  let side := 2
  let circle_area := Real.pi * radius^2
  let square_area := side^2
  ∃ (x : ℝ), (circle_area - x) - (square_area - x) = 9 * Real.pi - 4 :=
by
  sorry

end nonoverlapping_area_difference_l230_230236


namespace mike_max_marks_l230_230777

theorem mike_max_marks
  (M : ℝ)
  (h1 : 0.30 * M = 234)
  (h2 : 234 = 212 + 22) : M = 780 := 
sorry

end mike_max_marks_l230_230777


namespace inequality_bounds_of_xyz_l230_230974

theorem inequality_bounds_of_xyz
  (x y z : ℝ)
  (h1 : x < y) (h2 : y < z)
  (h3 : x + y + z = 6)
  (h4 : x * y + y * z + z * x = 9) :
  0 < x ∧ x < 1 ∧ 1 < y ∧ y < 3 ∧ 3 < z ∧ z < 4 := 
sorry

end inequality_bounds_of_xyz_l230_230974


namespace value_of_p_l230_230776

theorem value_of_p (m n p : ℝ) (h1 : m = 6 * n + 5) (h2 : m + 2 = 6 * (n + p) + 5) : p = 1 / 3 :=
by
  sorry

end value_of_p_l230_230776


namespace largest_n_satisfying_inequality_l230_230906

theorem largest_n_satisfying_inequality :
  ∃ n : ℕ, (1/4 + n/6 < 3/2) ∧ (∀ m : ℕ, m > n → 1/4 + m/6 ≥ 3/2) :=
by
  sorry

end largest_n_satisfying_inequality_l230_230906


namespace evaluate_expression_l230_230216

def a : ℕ := 3^1
def b : ℕ := 3^2
def c : ℕ := 3^3
def d : ℕ := 3^4
def e : ℕ := 3^10
def S : ℕ := a + b + c + d

theorem evaluate_expression : e - S = 58929 := 
by
  sorry

end evaluate_expression_l230_230216


namespace stream_speed_l230_230241

-- Define the conditions
def still_water_speed : ℝ := 15
def upstream_time_factor : ℕ := 2

-- Define the theorem
theorem stream_speed (t v : ℝ) (h : (still_water_speed + v) * t = (still_water_speed - v) * (upstream_time_factor * t)) : v = 5 :=
by
  sorry

end stream_speed_l230_230241


namespace round_robin_tournament_points_l230_230873

theorem round_robin_tournament_points :
  ∀ (teams : Finset ℕ), teams.card = 6 →
  ∀ (matches_played : ℕ), matches_played = 12 →
  ∀ (total_points : ℤ), total_points = 32 →
  ∀ (third_highest_points : ℤ), third_highest_points = 7 →
  ∀ (draws : ℕ), draws = 4 →
  ∃ (fifth_highest_points_min fifth_highest_points_max : ℤ),
    fifth_highest_points_min = 1 ∧
    fifth_highest_points_max = 3 :=
by
  sorry

end round_robin_tournament_points_l230_230873


namespace find_a_l230_230988

theorem find_a (a b c : ℤ) (h1 : a + b = c) (h2 : b + c = 8) (h3 : c = 4) : a = 0 :=
by
  sorry

end find_a_l230_230988


namespace unique_common_root_m_value_l230_230996

theorem unique_common_root_m_value (m : ℝ) (h : m > 5) :
  (∃ x : ℝ, x^2 - 5 * x + 6 = 0 ∧ x^2 + 2 * x - 2 * m + 1 = 0) →
  m = 8 :=
by
  sorry

end unique_common_root_m_value_l230_230996


namespace sasha_remainder_is_20_l230_230081

theorem sasha_remainder_is_20 (n a b c d : ℤ) (h1 : n = 102 * a + b) (h2 : n = 103 * c + d) (h3 : d = 20 - a) : b = 20 :=
by
  sorry

end sasha_remainder_is_20_l230_230081


namespace sasha_remainder_l230_230063

statement:
  theorem sasha_remainder (n a b c d : ℕ) (h1 : n = 102 * a + b) (h2 : n = 103 * c + d) (h3 : a + d = 20) (h4: 0 ≤ b ∧ b ≤ 101) : 
  b = 20 :=
by
  sorry

end sasha_remainder_l230_230063


namespace annual_interest_rate_l230_230446

theorem annual_interest_rate (principal total_paid: ℝ) (h_principal : principal = 150) (h_total_paid : total_paid = 162) : 
  ((total_paid - principal) / principal) * 100 = 8 :=
by
  sorry

end annual_interest_rate_l230_230446


namespace painting_frame_ratio_proof_l230_230124

def framed_painting_ratio (x : ℝ) : Prop :=
  let width := 20
  let height := 20
  let side_border := x
  let top_bottom_border := 3 * x
  let framed_width := width + 2 * side_border
  let framed_height := height + 2 * top_bottom_border
  let painting_area := width * height
  let frame_area := painting_area
  let total_area := framed_width * framed_height - painting_area
  total_area = frame_area ∧ (width + 2 * side_border) ≤ (height + 2 * top_bottom_border) → 
  framed_width / framed_height = 4 / 7

theorem painting_frame_ratio_proof (x : ℝ) (h : framed_painting_ratio x) : (20 + 2 * x) / (20 + 6 * x) = 4 / 7 :=
  sorry

end painting_frame_ratio_proof_l230_230124


namespace sum_of_four_smallest_divisors_eq_11_l230_230670

noncomputable def common_divisors_sum : ℤ :=
  let common_divisors := [1, 2, 3, 5, 6, 10, 15, 30]
  let smallest_four := common_divisors.take 4
  smallest_four.sum

theorem sum_of_four_smallest_divisors_eq_11 :
  common_divisors_sum = 11 := by
  sorry

end sum_of_four_smallest_divisors_eq_11_l230_230670


namespace team_A_days_additional_people_l230_230144

theorem team_A_days (x : ℕ) (y : ℕ)
  (h1 : 2700 / x = 2 * (1800 / y))
  (h2 : y = x + 1)
  : x = 3 ∧ y = 4 :=
by
  sorry

theorem additional_people (m : ℕ)
  (h1 : (200 : ℝ) * 10 * 3 + 150 * 8 * 4 = 10800)
  (h2 : (170 : ℝ) * (10 + m) * 3 + 150 * 8 * 4 = 1.20 * 10800)
  : m = 6 :=
by
  sorry

end team_A_days_additional_people_l230_230144


namespace water_left_after_four_hours_l230_230565

def initial_water : ℕ := 40
def water_loss_per_hour : ℕ := 2
def added_water_hour3 : ℕ := 1
def added_water_hour4 : ℕ := 3
def total_hours : ℕ := 4

theorem water_left_after_four_hours :
  initial_water - (water_loss_per_hour * total_hours) + (added_water_hour3 + added_water_hour4) = 36 := by
  sorry

end water_left_after_four_hours_l230_230565


namespace rod_volume_proof_l230_230238

-- Definitions based on given conditions
def original_length : ℝ := 2
def increase_in_surface_area : ℝ := 0.6
def rod_volume : ℝ := 0.3

-- Problem statement
theorem rod_volume_proof
  (len : ℝ)
  (inc_surface_area : ℝ)
  (vol : ℝ)
  (h_len : len = original_length)
  (h_inc_surface_area : inc_surface_area = increase_in_surface_area) :
  vol = rod_volume :=
sorry

end rod_volume_proof_l230_230238


namespace evaluate_f_at_1_l230_230684

def f (x : ℝ) : ℝ := x^2 + 2*x + 3

theorem evaluate_f_at_1 : f 1 = 6 := 
  sorry

end evaluate_f_at_1_l230_230684


namespace find_n_l230_230826

open Nat

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Given condition for the proof
def condition (n : ℕ) : Prop := binom (n + 1) 7 - binom n 7 = binom n 8

-- The statement to prove
theorem find_n (n : ℕ) (h : condition n) : n = 14 :=
sorry

end find_n_l230_230826


namespace exists_polynomial_P_l230_230140

open Int Nat

/-- Define a predicate for a value is a perfect square --/
def is_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

/-- Define the polynomial P(x, y, z) --/
noncomputable def P (x y z : ℕ) : ℤ := 
  (1 - 2013 * (z - 1) * (z - 2)) * 
  ((x + y - 1) * (x + y - 1) + 2 * y - 2 + z)

/-- The main theorem to prove --/
theorem exists_polynomial_P :
  ∃ (P : ℕ → ℕ → ℕ → ℤ), 
  (∀ n : ℕ, (¬ is_square n) ↔ ∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ P x y z = n) := 
sorry

end exists_polynomial_P_l230_230140


namespace percent_of_b_l230_230429

theorem percent_of_b (a b c : ℝ) (h1 : c = 0.25 * a) (h2 : b = 2.5 * a) : c = 0.1 * b := 
by
  sorry

end percent_of_b_l230_230429


namespace stephanie_speed_l230_230611

noncomputable def distance : ℝ := 15
noncomputable def time : ℝ := 3

theorem stephanie_speed :
  distance / time = 5 := 
sorry

end stephanie_speed_l230_230611


namespace mean_and_mode_of_data_l230_230019

open BigOperators

def data : List ℕ := [7, 5, 6, 8, 7, 9]

lemma mean_of_data : (data.sum / data.length) = 7 := 
by {
  -- sum of the data is (7 + 5 + 6 + 8 + 7 + 9) = 42
  -- length of the data is 6
  -- mean is 42 / 6 = 7
  sorry
}

lemma mode_of_data : (∃ n, n ∈ data ∧ (data.count n = data.maximum.data.count n)) ∧ (data.count 7 = 2) := 
by {
  -- 7 appears twice which is the most frequent 
  sorry
}

# Theorem combining both the mean and mode lemmas
theorem mean_and_mode_of_data : (data.sum / data.length) = 7 ∧ (∃ n, n ∈ data ∧ (data.count n = data.maximum.data.count n)) ∧ (data.count 7 = 2) := 
by {
  split,
  exact mean_of_data,
  exact mode_of_data
}

end mean_and_mode_of_data_l230_230019


namespace cos_a2_plus_a8_eq_neg_half_l230_230550

noncomputable def a_n (n : ℕ) (a₁ d : ℝ) : ℝ :=
  a₁ + (n - 1) * d

theorem cos_a2_plus_a8_eq_neg_half 
  (a₁ d : ℝ) 
  (h : a₁ + a_n 5 a₁ d + a_n 9 a₁ d = 5 * Real.pi)
  : Real.cos (a_n 2 a₁ d + a_n 8 a₁ d) = -1 / 2 :=
by
  sorry

end cos_a2_plus_a8_eq_neg_half_l230_230550


namespace Sonja_oil_used_l230_230743

theorem Sonja_oil_used :
  ∀ (oil peanuts total_weight : ℕ),
  (2 * oil + 8 * peanuts = 10) → (total_weight = 20) →
  ((20 / 10) * 2 = 4) :=
by 
  sorry

end Sonja_oil_used_l230_230743


namespace caterpillar_count_l230_230883

theorem caterpillar_count 
    (initial_count : ℕ)
    (hatched : ℕ)
    (left : ℕ)
    (h_initial : initial_count = 14)
    (h_hatched : hatched = 4)
    (h_left : left = 8) :
    initial_count + hatched - left = 10 :=
by
    sorry

end caterpillar_count_l230_230883


namespace total_meals_per_week_l230_230561

-- Definitions for the conditions
def first_restaurant_meals := 20
def second_restaurant_meals := 40
def third_restaurant_meals := 50
def days_in_week := 7

-- The theorem for the total meals per week
theorem total_meals_per_week : 
  (first_restaurant_meals + second_restaurant_meals + third_restaurant_meals) * days_in_week = 770 := 
by
  sorry

end total_meals_per_week_l230_230561


namespace smallest_positive_integer_solution_l230_230356

theorem smallest_positive_integer_solution :
  ∃ x : ℕ, 0 < x ∧ 5 * x ≡ 17 [MOD 34] ∧ (∀ y : ℕ, 0 < y ∧ 5 * y ≡ 17 [MOD 34] → x ≤ y) :=
sorry

end smallest_positive_integer_solution_l230_230356


namespace max_min_values_of_f_l230_230053

noncomputable def f (x : ℝ) : ℝ := 3 * x - x ^ 3

theorem max_min_values_of_f :
  (∀ x : ℝ, f x ≤ 2) ∧ (∃ x : ℝ, f x = 2) ∧
  (∀ x : ℝ, -2 ≤ f x) ∧ (∃ x : ℝ, f x = -2) :=
by 
  sorry

end max_min_values_of_f_l230_230053


namespace CD_is_b_minus_a_minus_c_l230_230855

variables (V : Type) [AddCommGroup V] [Module ℝ V]
variables (A B C D : V) (a b c : V)

def AB : V := a
def AD : V := b
def BC : V := c

theorem CD_is_b_minus_a_minus_c (h1 : A + a = B) (h2 : A + b = D) (h3 : B + c = C) :
  D - C = b - a - c :=
by sorry

end CD_is_b_minus_a_minus_c_l230_230855


namespace geometric_sequence_sum_l230_230178

theorem geometric_sequence_sum (S : ℕ → ℚ) (a : ℕ → ℚ)
  (h1 : S 4 = 1)
  (h2 : S 8 = 3)
  (h3 : ∀ n, S (n + 4) - S n = a (n + 1) + a (n + 2) + a (n + 3) + a (n + 4)) :
  a 17 + a 18 + a 19 + a 20 = 16 :=
by
  -- Insert your proof here.
  sorry

end geometric_sequence_sum_l230_230178


namespace find_a_l230_230986

theorem find_a {a b c : ℕ} (h₁ : a + b = c) (h₂ : b + c = 8) (h₃ : c = 4) : a = 0 := by
  sorry

end find_a_l230_230986


namespace sum_of_sampled_types_l230_230120

-- Define the types of books in each category
def Chinese_types := 20
def Mathematics_types := 10
def Liberal_Arts_Comprehensive_types := 40
def English_types := 30

-- Define the total types of books
def total_types := Chinese_types + Mathematics_types + Liberal_Arts_Comprehensive_types + English_types

-- Define the sample size and stratified sampling ratio
def sample_size := 20
def sampling_ratio := sample_size / total_types

-- Define the number of types sampled from each category
def Mathematics_sampled := Mathematics_types * sampling_ratio
def Liberal_Arts_Comprehensive_sampled := Liberal_Arts_Comprehensive_types * sampling_ratio

-- Define the proof statement
theorem sum_of_sampled_types : Mathematics_sampled + Liberal_Arts_Comprehensive_sampled = 10 :=
by
  -- Your proof here
  sorry

end sum_of_sampled_types_l230_230120


namespace total_peaches_l230_230207

theorem total_peaches (num_baskets num_red num_green : ℕ)
    (h1 : num_baskets = 11)
    (h2 : num_red = 10)
    (h3 : num_green = 18) : (num_red + num_green) * num_baskets = 308 := by
  sorry

end total_peaches_l230_230207


namespace girls_attending_ball_l230_230458

theorem girls_attending_ball (g b : ℕ) 
    (h1 : g + b = 1500) 
    (h2 : 3 * g / 4 + 2 * b / 3 = 900) : 
    g = 1200 ∧ 3 * 1200 / 4 = 900 := 
by
  sorry

end girls_attending_ball_l230_230458


namespace yoki_cans_correct_l230_230533

def total_cans := 85
def ladonna_cans := 25
def prikya_cans := 2 * ladonna_cans
def yoki_cans := total_cans - ladonna_cans - prikya_cans

theorem yoki_cans_correct : yoki_cans = 10 :=
by
  sorry

end yoki_cans_correct_l230_230533


namespace min_sum_of_arithmetic_sequence_terms_l230_230290

open Real

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∀ n m : ℕ, ∃ d : ℝ, a m = a n + d * (m - n)

theorem min_sum_of_arithmetic_sequence_terms (a : ℕ → ℝ) 
  (hpos : ∀ n, a n > 0) 
  (harith : arithmetic_sequence a) 
  (hprod : a 1 * a 20 = 100) : 
  a 7 + a 14 ≥ 20 := sorry

end min_sum_of_arithmetic_sequence_terms_l230_230290


namespace find_interest_rate_l230_230819

noncomputable def amount : ℝ := 896
noncomputable def principal : ℝ := 799.9999999999999
noncomputable def time : ℝ := 2 + 2 / 5
noncomputable def interest : ℝ := amount - principal
noncomputable def rate : ℝ := interest / (principal * time)

theorem find_interest_rate :
  rate * 100 = 5 := by
  sorry

end find_interest_rate_l230_230819


namespace and_implies_or_or_does_not_imply_and_and_is_sufficient_but_not_necessary_for_or_l230_230228

theorem and_implies_or (p q : Prop) (hpq : p ∧ q) : p ∨ q :=
by {
  sorry
}

theorem or_does_not_imply_and (p q : Prop) (hp_or_q : p ∨ q) : ¬ (p ∧ q) :=
by {
  sorry
}

theorem and_is_sufficient_but_not_necessary_for_or (p q : Prop) : (p ∧ q → p ∨ q) ∧ ¬ (p ∨ q → p ∧ q) :=
by {
  exact ⟨and_implies_or p q, or_does_not_imply_and p q⟩,
}

end and_implies_or_or_does_not_imply_and_and_is_sufficient_but_not_necessary_for_or_l230_230228


namespace pascal_triangle_probability_l230_230946

theorem pascal_triangle_probability :
  let total_elements := 210
  let total_ones := 39
  let probability := (total_ones : ℚ) / total_elements
  probability = (13 : ℚ) / 70 := by
  -- Here we provide the necessary definitions and the statement of the theorem
  -- To keep the focus on proving the relationship, the steps and proof are abstracted out
  sorry

end pascal_triangle_probability_l230_230946


namespace petya_coloring_l230_230226

theorem petya_coloring (n : ℕ) (h₁ : n = 100) (h₂ : ∀ i j : ℕ, 1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n → 1 ≤ (number i j) ∧ (number i j) ≤ n * n) :
  ∃ k, k = 1 ∧ ∀ (initial_coloring : fin (n * n) → bool) (next_colorable : (fin (n * n) → bool) → (fin (n * n) → bool)),
    (∀ (table : fin n × fin n → fin (n * n)), next_colorable (λ a, initial_coloring a)
    (λ a, initial_coloring a) a) :=
begin
  sorry,
end

end petya_coloring_l230_230226


namespace total_distance_hopped_l230_230206

def distance_hopped (rate: ℕ) (time: ℕ) : ℕ := rate * time

def spotted_rabbit_distance (time: ℕ) : ℕ :=
  let pattern := [8, 11, 16, 20, 9]
  let full_cycles := time / pattern.length
  let remaining_minutes := time % pattern.length
  let full_cycle_distance := full_cycles * pattern.sum
  let remaining_distance := (List.take remaining_minutes pattern).sum
  full_cycle_distance + remaining_distance

theorem total_distance_hopped :
  distance_hopped 15 12 + distance_hopped 12 12 + distance_hopped 18 12 + distance_hopped 10 12 + spotted_rabbit_distance 12 = 807 :=
by
  sorry

end total_distance_hopped_l230_230206


namespace pastries_left_l230_230672

def pastries_baked : ℕ := 4 + 29
def pastries_sold : ℕ := 9

theorem pastries_left : pastries_baked - pastries_sold = 24 :=
by
  -- assume pastries_baked = 33
  -- assume pastries_sold = 9
  -- prove 33 - 9 = 24
  sorry

end pastries_left_l230_230672


namespace watch_cost_price_l230_230222

noncomputable def cost_price : ℝ := 1166.67

theorem watch_cost_price (CP : ℝ) (loss_percent gain_percent : ℝ) (delta : ℝ) 
  (h1 : loss_percent = 0.10) 
  (h2 : gain_percent = 0.02) 
  (h3 : delta = 140) 
  (h4 : (1 - loss_percent) * CP + delta = (1 + gain_percent) * CP) : 
  CP = cost_price := 
by 
  sorry

end watch_cost_price_l230_230222


namespace sasha_remainder_l230_230086

theorem sasha_remainder (n a b c d : ℕ) 
  (h1 : n = 102 * a + b) 
  (h2 : n = 103 * c + d) 
  (h3 : a + d = 20)
  (hb : 0 ≤ b ∧ b ≤ 101) : 
  b = 20 :=
sorry

end sasha_remainder_l230_230086


namespace simplify_expression_l230_230546

theorem simplify_expression (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a ≠ 0) (h3 : b ≠ 0) (h4 : c ≠ 0) :
    a * (1 / b + 1 / c) + b * (1 / a + 1 / c) + c * (1 / a + 1 / b) = -3 :=
by
  sorry

end simplify_expression_l230_230546


namespace kristin_runs_n_times_faster_l230_230028

theorem kristin_runs_n_times_faster (D K S : ℝ) (n : ℝ) 
  (h1 : K = n * S) 
  (h2 : 12 * D / K = 4 * D / S) : 
  n = 3 :=
by
  sorry

end kristin_runs_n_times_faster_l230_230028


namespace factorize_expression_l230_230538

theorem factorize_expression (x : ℝ) : 2 * x ^ 3 - 4 * x ^ 2 - 6 * x = 2 * x * (x - 3) * (x + 1) :=
by
  sorry

end factorize_expression_l230_230538


namespace no_rational_roots_l230_230441

theorem no_rational_roots (p q : ℤ) (h1 : p % 3 = 2) (h2 : q % 3 = 2) :
  ¬ ∃ (a b : ℤ), b ≠ 0 ∧ a * a = b * b * (p^2 - 4 * q) :=
by
  sorry

end no_rational_roots_l230_230441


namespace linear_function_result_l230_230325

variable {R : Type*} [LinearOrderedField R]

noncomputable def linear_function (g : R → R) : Prop :=
  ∃ (a b : R), ∀ x, g x = a * x + b

theorem linear_function_result (g : R → R) (h_lin : linear_function g) (h : g 5 - g 1 = 16) : g 13 - g 1 = 48 :=
  by
  sorry

end linear_function_result_l230_230325


namespace Tom_age_ratio_l230_230493

variable (T N : ℕ)
variable (a : ℕ)
variable (c3 c4 : ℕ)

-- conditions
def condition1 : Prop := T = 4 * a + 5
def condition2 : Prop := T - N = 3 * (4 * a + 5 - 4 * N)

theorem Tom_age_ratio (h1 : condition1 T a) (h2 : condition2 T N a) : (T = 6 * N) :=
by sorry

end Tom_age_ratio_l230_230493


namespace palindrome_count_l230_230929

theorem palindrome_count :
  let A_choices := 9
  let B_choices := 10
  let C_choices := 10
  (A_choices * B_choices * C_choices) = 900 :=
by
  let A_choices := 9
  let B_choices := 10
  let C_choices := 10
  show (A_choices * B_choices * C_choices) = 900
  sorry

end palindrome_count_l230_230929


namespace total_crayons_l230_230867

-- Definitions for conditions
def boxes : Nat := 7
def crayons_per_box : Nat := 5

-- Statement that needs to be proved
theorem total_crayons : boxes * crayons_per_box = 35 := by
  sorry

end total_crayons_l230_230867


namespace combination_x_l230_230168
noncomputable def C (n k : ℕ) : ℕ := Nat.choose n k

theorem combination_x (x : ℕ) (H : C 25 (2 * x) = C 25 (x + 4)) : x = 4 ∨ x = 7 :=
by sorry

end combination_x_l230_230168


namespace peter_and_susan_dollars_l230_230190

theorem peter_and_susan_dollars :
  (2 / 5 : Real) + (1 / 4 : Real) = 0.65 := 
by
  sorry

end peter_and_susan_dollars_l230_230190


namespace angle_opposite_c_exceeds_l230_230702

theorem angle_opposite_c_exceeds (a b : ℝ) (c : ℝ) (C : ℝ) (h_a : a = 2) (h_b : b = 2) (h_c : c >= 4) : 
  C >= 120 := 
sorry

end angle_opposite_c_exceeds_l230_230702


namespace arithmetic_progression_condition_l230_230401

theorem arithmetic_progression_condition
  (a b c : ℝ) : ∃ (A B : ℤ), A ≠ 0 ∧ B ≠ 0 ∧ (b - a) * B = (c - b) * A := 
by {
  sorry
}

end arithmetic_progression_condition_l230_230401


namespace boat_distance_along_stream_l230_230853

theorem boat_distance_along_stream
  (distance_against_stream : ℝ)
  (speed_still_water : ℝ)
  (time : ℝ)
  (v_s : ℝ)
  (H1 : distance_against_stream = 5)
  (H2 : speed_still_water = 6)
  (H3 : time = 1)
  (H4 : speed_still_water - v_s = distance_against_stream / time) :
  (speed_still_water + v_s) * time = 7 :=
by
  -- Sorry to skip proof
  sorry

end boat_distance_along_stream_l230_230853


namespace correct_operation_l230_230916

noncomputable def check_operations : Prop :=
    ∀ (a : ℝ), ( a^6 / a^3 = a^3 ) ∧ 
               ¬( 3 * a^5 + a^5 = 4 * a^10 ) ∧
               ¬( (2 * a)^3 = 2 * a^3 ) ∧
               ¬( (a^2)^4 = a^6 )

theorem correct_operation : check_operations :=
by
  intro a
  have h1 : a^6 / a^3 = a^3 := by
    sorry
  have h2 : ¬(3 * a^5 + a^5 = 4 * a^10) := by
    sorry
  have h3 : ¬((2 * a)^3 = 2 * a^3) := by
    sorry
  have h4 : ¬((a^2)^4 = a^6) := by
    sorry
  exact ⟨h1, h2, h3, h4⟩

end correct_operation_l230_230916


namespace added_number_is_four_l230_230897

theorem added_number_is_four :
  ∃ x y, 2 * x < 3 * x ∧ (3 * x - 2 * x = 8) ∧ 
         ((2 * x + y) * 7 = 5 * (3 * x + y)) ∧ y = 4 :=
  sorry

end added_number_is_four_l230_230897


namespace gcd_1995_228_eval_f_at_2_l230_230115

-- Euclidean Algorithm Problem
theorem gcd_1995_228 : Nat.gcd 1995 228 = 57 :=
by
  sorry

-- Horner's Method Problem
def f (x : ℝ) : ℝ := 3 * x ^ 5 + 2 * x ^ 3 - 8 * x + 5

theorem eval_f_at_2 : f 2 = 101 :=
by
  sorry

end gcd_1995_228_eval_f_at_2_l230_230115


namespace simplify_expression_to_fraction_l230_230740

theorem simplify_expression_to_fraction : 
  (1 / (1 / (1/2)^2 + 1 / (1/2)^3 + 1 / (1/2)^4 + 1 / (1/2)^5)) = 1/60 :=
by 
  have h1 : 1 / (1/2)^2 = 4 := by sorry
  have h2 : 1 / (1/2)^3 = 8 := by sorry
  have h3 : 1 / (1/2)^4 = 16 := by sorry
  have h4 : 1 / (1/2)^5 = 32 := by sorry
  have h5 : 4 + 8 + 16 + 32 = 60 := by sorry
  have h6 : 1 / 60 = 1/60 := by sorry
  sorry

end simplify_expression_to_fraction_l230_230740


namespace combine_fraction_l230_230918

variable (d : ℤ)

theorem combine_fraction : (3 + 4 * d) / 5 + 3 = (18 + 4 * d) / 5 := by
  sorry

end combine_fraction_l230_230918


namespace maximize_probability_sum_8_l230_230899

def L : List Int := [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12]

theorem maximize_probability_sum_8 :
  (∀ x ∈ L, x ≠ 4 → (∃ y ∈ (List.erase L x), y = 8 - x)) ∧ 
  (∀ y ∈ List.erase L 4, ¬(∃ x ∈ List.erase L 4, x + y = 8)) :=
sorry

end maximize_probability_sum_8_l230_230899


namespace arctan_sum_of_roots_eq_pi_div_4_l230_230185

theorem arctan_sum_of_roots_eq_pi_div_4 (x₁ x₂ x₃ : ℝ) 
  (h₁ : Polynomial.eval x₁ (Polynomial.C 11 - Polynomial.C 10 * Polynomial.X + Polynomial.X ^ 3) = 0)
  (h₂ : Polynomial.eval x₂ (Polynomial.C 11 - Polynomial.C 10 * Polynomial.X + Polynomial.X ^ 3) = 0)
  (h₃ : Polynomial.eval x₃ (Polynomial.C 11 - Polynomial.C 10 * Polynomial.X + Polynomial.X ^ 3) = 0)
  (h_intv : -5 < x₁ ∧ x₁ < 5 ∧ -5 < x₂ ∧ x₂ < 5 ∧ -5 < x₃ ∧ x₃ < 5) :
  Real.arctan x₁ + Real.arctan x₂ + Real.arctan x₃ = Real.pi / 4 :=
sorry

end arctan_sum_of_roots_eq_pi_div_4_l230_230185


namespace factorization_example_l230_230219

theorem factorization_example (x: ℝ) : x^3 - x = x * (x + 1) * (x - 1) :=
sorry

end factorization_example_l230_230219


namespace trigonometric_operation_l230_230981

theorem trigonometric_operation :
  let m := Real.cos (Real.pi / 6)
  let n := Real.sin (Real.pi / 6)
  let op (m n : ℝ) := m^2 - m * n - n^2
  op m n = (1 / 2 : ℝ) - (Real.sqrt 3 / 4) :=
by
  sorry

end trigonometric_operation_l230_230981


namespace bisection_method_third_interval_l230_230365

theorem bisection_method_third_interval 
  (f : ℝ → ℝ) (a b : ℝ) (H1 : a = -2) (H2 : b = 4) 
  (H3 : f a * f b ≤ 0) : 
  ∃ c d : ℝ, c = -1/2 ∧ d = 1 ∧ f c * f d ≤ 0 :=
by 
  sorry

end bisection_method_third_interval_l230_230365


namespace literature_club_students_neither_english_nor_french_l230_230034

theorem literature_club_students_neither_english_nor_french
  (total_students english_students french_students both_students : ℕ)
  (h1 : total_students = 120)
  (h2 : english_students = 72)
  (h3 : french_students = 52)
  (h4 : both_students = 12) :
  (total_students - ((english_students - both_students) + (french_students - both_students) + both_students) = 8) :=
by
  sorry

end literature_club_students_neither_english_nor_french_l230_230034


namespace swimming_pool_width_l230_230517

theorem swimming_pool_width (L D1 D2 V : ℝ) (W : ℝ) (h : L = 12) (h1 : D1 = 1) (h2 : D2 = 4) (hV : V = 270) : W = 9 :=
  by
    -- We begin by stating the formula for the volume of 
    -- a trapezoidal prism: Volume = (1/2) * (D1 + D2) * L * W
    
    -- According to the problem, we have the following conditions:
    have hVolume : V = (1/2) * (D1 + D2) * L * W :=
      by sorry

    -- Substitute the provided values into the volume equation:
    -- 270 = (1/2) * (1 + 4) * 12 * W
    
    -- Simplify and solve for W
    simp at hVolume
    exact sorry

end swimming_pool_width_l230_230517


namespace onions_total_l230_230335

theorem onions_total (Sara : ℕ) (Sally : ℕ) (Fred : ℕ)
  (hSara : Sara = 4) (hSally : Sally = 5) (hFred : Fred = 9) :
  Sara + Sally + Fred = 18 :=
by
  sorry

end onions_total_l230_230335


namespace det_value_l230_230536

open Matrix

noncomputable def det_example (α β : ℝ) : ℝ :=
  det ![
    ![0, Real.cos α, Real.sin α],
    ![Real.sin α, 0, Real.cos β],
    ![-Real.cos α, -Real.sin β, 0]
  ]

theorem det_value (α β : ℝ) : 
  det_example α β = -(Real.cos β * Real.cos α ^ 2 + Real.sin β * Real.sin α ^ 2) :=
by
  sorry

end det_value_l230_230536


namespace minimum_value_expr_l230_230586

theorem minimum_value_expr (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : m + n = 1) : 
  (1 + (1 / m)) * (1 + (1 / n)) = 9 :=
sorry

end minimum_value_expr_l230_230586


namespace count_non_congruent_rectangles_l230_230381

-- Definitions based on conditions given in the problem
def is_rectangle (w h : ℕ) : Prop := 2 * (w + h) = 40 ∧ w % 2 = 0

-- Theorem that we need to prove based on the problem statement
theorem count_non_congruent_rectangles : 
  ∃ n : ℕ, n = 9 ∧ 
  (∀ p : ℕ × ℕ, p ∈ { p | is_rectangle p.1 p.2 } → ∀ q : ℕ × ℕ, q ∈ { q | is_rectangle q.1 q.2 } → p = q ∨ p ≠ q) := 
sorry

end count_non_congruent_rectangles_l230_230381


namespace proof_range_of_a_l230_230299

/-- p is the proposition that for all x in [1,2], x^2 - a ≥ 0 --/
def p (a : ℝ) : Prop := ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0

/-- q is the proposition that there exists an x0 in ℝ such that x0^2 + (a-1)x0 + 1 < 0 --/
def q (a : ℝ) : Prop := ∃ x0 : ℝ, x0^2 + (a-1)*x0 + 1 < 0

theorem proof_range_of_a (a : ℝ) : (p a ∨ q a) ∧ (¬p a ∧ ¬q a) → (a ≥ -1 ∧ a ≤ 1) ∨ a > 3 :=
by
  sorry -- proof will be filled out here

end proof_range_of_a_l230_230299


namespace ethanol_combustion_heat_l230_230915

theorem ethanol_combustion_heat (Q : Real) :
  (∃ (m : Real), m = 0.1 ∧ (∀ (n : Real), n = 1 → Q * n / m = 10 * Q)) :=
by
  sorry

end ethanol_combustion_heat_l230_230915


namespace correct_statements_l230_230863

-- Define the function and the given conditions
def f : ℝ → ℝ := sorry

lemma not_constant (h: ∃ x y: ℝ, x ≠ y ∧ f x ≠ f y) : true := sorry
lemma periodic (x : ℝ) : f (x - 1) = f (x + 1) := sorry
lemma symmetric (x : ℝ) : f (2 - x) = f x := sorry

-- The statements we want to prove
theorem correct_statements : 
  (∀ x, f (-x) = f x) ∧ 
  (∀ x, f (1 - x) = f (1 + x)) ∧ 
  (∃ T > 0, ∀ x, f (x + T) = f x)
:= by
  sorry

end correct_statements_l230_230863


namespace sufficient_and_necessary_condition_l230_230936

theorem sufficient_and_necessary_condition (a : ℝ) :
  (∀ x : ℝ, a * x^2 - 2 * a * x + 1 > 0) ↔ (0 < a ∧ a < 1) :=
by 
  sorry

end sufficient_and_necessary_condition_l230_230936


namespace tangent_lines_through_origin_l230_230418

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 3 * x^2 + 1

variable (a : ℝ)

theorem tangent_lines_through_origin 
  (h1 : ∃ m1 m2 : ℝ, m1 ≠ m2 ∧ (f a (-m1) + f a (m1 + 2)) / 2 = f a 1) :
  ∃ t1 t2 : ℝ, t1 ≠ t2 ∧ (f a t1 * (1 / t1) = f a 0) ∧ (f a t2 * (1 / t2) = f a 0) := 
sorry

end tangent_lines_through_origin_l230_230418


namespace Alex_age_l230_230766

theorem Alex_age : ∃ (x : ℕ), (∃ (y : ℕ), x - 2 = y^2) ∧ (∃ (z : ℕ), x + 2 = z^3) ∧ x = 6 := by
  sorry

end Alex_age_l230_230766


namespace C_share_l230_230789

-- Definitions based on conditions
def total_sum : ℝ := 164
def ratio_B : ℝ := 0.65
def ratio_C : ℝ := 0.40

-- Statement of the proof problem
theorem C_share : (ratio_C * (total_sum / (1 + ratio_B + ratio_C))) = 32 :=
by
  sorry

end C_share_l230_230789


namespace find_A_for_diamond_eq_85_l230_230844

def diamond (A B : ℝ) : ℝ := 4 * A + B^2 + 7

theorem find_A_for_diamond_eq_85 :
  ∃ (A : ℝ), diamond A 3 = 85 ∧ A = 17.25 :=
by
  sorry

end find_A_for_diamond_eq_85_l230_230844


namespace ten_integers_disjoint_subsets_same_sum_l230_230102

theorem ten_integers_disjoint_subsets_same_sum (S : Finset ℕ) (h : S.card = 10) (h_range : ∀ x ∈ S, 10 ≤ x ∧ x ≤ 99) :
  ∃ A B : Finset ℕ, A ≠ B ∧ A ⊆ S ∧ B ⊆ S ∧ A ∩ B = ∅ ∧ A.sum id = B.sum id :=
by sorry

end ten_integers_disjoint_subsets_same_sum_l230_230102


namespace b_is_dk_squared_l230_230183

theorem b_is_dk_squared (a b : ℤ) (h : ∃ r1 r2 r3 : ℤ, (r1 * r2 * r3 = b) ∧ (r1 + r2 + r3 = a) ∧ (r1 * r2 + r1 * r3 + r2 * r3 = 0))
  : ∃ d k : ℤ, (b = d * k^2) ∧ (d ∣ a) := 
sorry

end b_is_dk_squared_l230_230183


namespace gcd_of_256_180_600_l230_230627

-- Define the numbers
def n1 := 256
def n2 := 180
def n3 := 600

-- Statement to prove the GCD of the three numbers is 12
theorem gcd_of_256_180_600 : Nat.gcd (Nat.gcd n1 n2) n3 = 12 := 
by {
  -- Prime factorizations given as part of the conditions:
  have pf1 : n1 = 2^8 := by norm_num,
  have pf2 : n2 = 2^2 * 3^2 * 5 := by norm_num,
  have pf3 : n3 = 2^3 * 3 * 5^2 := by norm_num,
  
  -- Calculate GCD step by step should be done here, 
  -- But replaced by sorry as per the instructions.
  sorry
}

end gcd_of_256_180_600_l230_230627


namespace problem_omega_pow_l230_230029

noncomputable def omega : ℂ := Complex.I -- Define a non-real root for x² = 1; an example choice could be i, the imaginary unit.

theorem problem_omega_pow :
  omega^2 = 1 → 
  (1 - omega + omega^2)^6 + (1 + omega - omega^2)^6 = 730 := 
by
  intro h1
  -- proof steps omitted
  sorry

end problem_omega_pow_l230_230029


namespace root_expression_value_l230_230010

theorem root_expression_value (a : ℝ) (h : a^2 + a - 1 = 0) : 2021 - 2 * a^2 - 2 * a = 2019 := 
by sorry

end root_expression_value_l230_230010


namespace implicit_derivative_l230_230542

noncomputable section

open Real

section ImplicitDifferentiation

variable {x : ℝ} {y : ℝ → ℝ}

def f (x y : ℝ) : ℝ := y^2 + x^2 - 1

theorem implicit_derivative (h : f x (y x) = 0) :
  deriv y x = -x / y x :=
  sorry

end ImplicitDifferentiation

end implicit_derivative_l230_230542


namespace smallest_positive_real_number_l230_230271

noncomputable def smallest_x : ℝ :=
  Inf {x : ℝ | x > 0 ∧ (floor (x^2) - x * floor x = 8)}

theorem smallest_positive_real_number :
  smallest_x = 89 / 9 :=
by 
  sorry

end smallest_positive_real_number_l230_230271


namespace inequality_solution_l230_230818

theorem inequality_solution (x : ℝ) : 
  (0 < x ∧ x ≤ 3) ∨ (4 ≤ x) ↔ (3 * (x - 3) * (x - 4)) / x ≥ 0 := 
sorry

end inequality_solution_l230_230818


namespace bread_slices_remaining_l230_230462

-- Conditions
def total_slices : ℕ := 12
def fraction_eaten_for_breakfast : ℕ := total_slices / 3
def slices_used_for_lunch : ℕ := 2

-- Mathematically Equivalent Proof Problem
theorem bread_slices_remaining : total_slices - fraction_eaten_for_breakfast - slices_used_for_lunch = 6 :=
by
  sorry

end bread_slices_remaining_l230_230462


namespace man_l230_230795

theorem man's_salary (S : ℝ)
  (h1 : S * (1/5 + 1/10 + 3/5) = 9/10 * S)
  (h2 : S - 9/10 * S = 14000) :
  S = 140000 :=
by
  sorry

end man_l230_230795


namespace angle_in_third_quadrant_l230_230619

/-- 
Given that the terminal side of angle α is in the third quadrant,
prove that the terminal side of α/3 cannot be in the second quadrant.
-/
theorem angle_in_third_quadrant (α : ℝ) (k : ℤ)
  (h : π + 2 * k * π < α ∧ α < 3 / 2 * π + 2 * k * π) :
  ¬ (π / 2 < α / 3 ∧ α / 3 < π) :=
sorry

end angle_in_third_quadrant_l230_230619


namespace calc_fraction_power_l230_230803

theorem calc_fraction_power (n m : ℤ) (h_n : n = 2023) (h_m : m = 2022) :
  (- (2 / 3 : ℚ))^n * ((3 / 2 : ℚ))^m = - (2 / 3) := by
  sorry

end calc_fraction_power_l230_230803


namespace find_constants_l230_230824

noncomputable def find_a {a : ℝ} (expansion_constant : ℝ) := 
  ∃ (n : ℕ) (k : ℕ), n = 8 ∧ k = 4 ∧ ((binom n k * (-a)^k): ℝ) = expansion_constant

noncomputable def sum_of_coefficients {a : ℝ} (s1 s2 : ℝ) :=
  let p (x : ℝ) := (x - a/x : ℝ)^8 in
  p 1 = s1 ∨ p 1 = s2

theorem find_constants (a s1 s2 : ℝ) (expansion_constant : ℝ) :
  find_a expansion_constant → (a = 2 ∨ a = -2) ∧
  sum_of_coefficients s1 s2 :=
by
  sorry

end find_constants_l230_230824


namespace manager_final_price_l230_230933

noncomputable def wholesale_cost : ℝ := 200
noncomputable def retail_price : ℝ := wholesale_cost + 0.2 * wholesale_cost
noncomputable def manager_discount : ℝ := 0.1 * retail_price
noncomputable def price_after_manager_discount : ℝ := retail_price - manager_discount
noncomputable def weekend_sale_discount : ℝ := 0.1 * price_after_manager_discount
noncomputable def price_after_weekend_sale : ℝ := price_after_manager_discount - weekend_sale_discount
noncomputable def sales_tax : ℝ := 0.08 * price_after_weekend_sale
noncomputable def total_price : ℝ := price_after_weekend_sale + sales_tax

theorem manager_final_price : total_price = 209.95 := by
  sorry

end manager_final_price_l230_230933


namespace fabric_ratio_wednesday_tuesday_l230_230139

theorem fabric_ratio_wednesday_tuesday :
  let fabric_monday := 20
  let fabric_tuesday := 2 * fabric_monday
  let cost_per_yard := 2
  let total_earnings := 140
  let earnings_monday := fabric_monday * cost_per_yard
  let earnings_tuesday := fabric_tuesday * cost_per_yard
  let earnings_wednesday := total_earnings - (earnings_monday + earnings_tuesday)
  let fabric_wednesday := earnings_wednesday / cost_per_yard
  (fabric_wednesday / fabric_tuesday = 1 / 4) :=
by
  sorry

end fabric_ratio_wednesday_tuesday_l230_230139


namespace union_sets_l230_230556

open Set

def A : Set ℕ := {2, 5, 6}
def B : Set ℕ := {3, 5}

theorem union_sets : A ∪ B = {2, 3, 5, 6} := by
  sorry

end union_sets_l230_230556


namespace g_nested_result_l230_230719

def g (n : ℕ) : ℕ :=
if n < 5 then
  n^2 + 1
else
  2 * n + 3

theorem g_nested_result : g (g (g 3)) = 49 := by
sorry

end g_nested_result_l230_230719


namespace postal_code_permutations_l230_230892

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

def multiplicity_permutations (n : Nat) (repetitions : List Nat) : Nat :=
  factorial n / List.foldl (λ acc k => acc * factorial k) 1 repetitions

theorem postal_code_permutations : multiplicity_permutations 4 [2, 1, 1] = 12 :=
by
  unfold multiplicity_permutations
  unfold factorial
  sorry

end postal_code_permutations_l230_230892


namespace find_length_DC_l230_230023

noncomputable def length_DC (AB BC AD : ℕ) (BD : ℕ) (h1 : AB = 52) (h2 : BC = 21) (h3 : AD = 48) (h4 : AB^2 = AD^2 + BD^2) (h5 : BD^2 = 20^2) : ℕ :=
  let DC := 29
  DC

theorem find_length_DC (AB BC AD : ℕ) (BD : ℕ) (h1 : AB = 52) (h2 : BC = 21) (h3 : AD = 48) (h4 : AB^2 = AD^2 + BD^2) (h5 : BD^2 = 20^2) (h6 : 20^2 + BC^2 = DC^2) : length_DC AB BC AD BD h1 h2 h3 h4 h5 = 29 :=
  by
  sorry

end find_length_DC_l230_230023


namespace num_integer_solutions_eq_3_l230_230683

theorem num_integer_solutions_eq_3 :
  ∃ (S : Finset (ℤ × ℤ)), 
  (∀ (x y : ℤ), ((2 * x^2) + (x * y) + (y^2) - x + 2 * y + 1 = 0 ↔ (x, y) ∈ S)) ∧ 
  S.card = 3 :=
sorry

end num_integer_solutions_eq_3_l230_230683


namespace repeating_decimal_as_fraction_lowest_terms_l230_230812

theorem repeating_decimal_as_fraction_lowest_terms :
  ∃ (x : ℝ), x = 0.36 ∧ x = 4 / 11 :=
begin
  let x := 0.363636363636...,
  have h1 : x = 0.36, sorry, -- Represent the repeating decimal
  have h2 : 100 * x = 36.363636..., sorry, -- Multiply by 100 and represent the repeating decimal again
  have h3 : 100 * x - x = 36.363636... - 0.36, sorry, -- Subtraction step and represent the repeating decimal again
  have h4 : 99 * x = 36, sorry, -- Simplify the equation
  have h5 : x = 36 / 99, sorry, -- Solve for x
  have h6 : (36 / 99) = (4 / 11), sorry, -- Simplify the fraction
  use 0.36,
  split,
  { exact h1 },
  { exact h6.symm }
end

end repeating_decimal_as_fraction_lowest_terms_l230_230812


namespace cuboid_area_correct_l230_230368

def cuboid_surface_area (length breadth height : ℕ) :=
  2 * (length * height) + 2 * (breadth * height) + 2 * (length * breadth)

theorem cuboid_area_correct : cuboid_surface_area 4 6 5 = 148 := by
  sorry

end cuboid_area_correct_l230_230368


namespace part_a_part_b_l230_230450

def n_good (n m : ℕ) : Prop :=
  ∃ S : Finset ℕ, S.card ≤ 2 * n ∧ ∀ p ∈ S, p.prime ∧ p^2 ∣ m

theorem part_a (n a b : ℕ) (co : Nat.gcd a b = 1) :
  ∃ x y : ℕ, n_good n (a * x^n + b * y^n) :=
sorry

theorem part_b (n k : ℕ) (a : Fin k → ℕ) (co : Nat.gcd (Finset.univ.card) a = 1) :
  ∃ x : Fin k → ℕ, n_good n (Finset.univ.sum (λ i, a i * x i ^ n)) :=
sorry

end part_a_part_b_l230_230450


namespace number_of_grade12_students_selected_l230_230377

def total_students : ℕ := 1500
def grade10_students : ℕ := 550
def grade11_students : ℕ := 450
def total_sample_size : ℕ := 300
def grade12_students : ℕ := total_students - grade10_students - grade11_students

theorem number_of_grade12_students_selected :
    (total_sample_size * grade12_students / total_students) = 100 := by
  sorry

end number_of_grade12_students_selected_l230_230377


namespace fraction_value_l230_230200

theorem fraction_value (a : ℕ) (h : a > 0) (h_eq : (a:ℝ) / (a + 35) = 0.7) : a = 82 :=
by
  -- Steps to prove the theorem here
  sorry

end fraction_value_l230_230200


namespace sum_possible_values_of_k_l230_230708

theorem sum_possible_values_of_k (j k : ℕ) (h : (1 / (j : ℚ) + 1 / (k : ℚ) = 1 / 4)) (hj : 0 < j) (hk : 0 < k) :
  {x : ℕ | (1 / (j : ℚ) + 1 / (x : ℚ) = 1 / 4) ∧ 0 < x}.sum id = 51 :=
sorry

end sum_possible_values_of_k_l230_230708


namespace least_n_exceeds_product_l230_230163

def product_exceeds (n : ℕ) : Prop :=
  10^(n * (n + 1) / 18) > 10^6

theorem least_n_exceeds_product (n : ℕ) (h : n = 12) : product_exceeds n :=
by
  rw [h]
  sorry

end least_n_exceeds_product_l230_230163


namespace arithmetic_sequence_product_l230_230155

noncomputable def a_n (n : ℕ) (a1 d : ℤ) : ℤ := a1 + (n - 1) * d

theorem arithmetic_sequence_product (a_1 d : ℤ) :
  (a_n 4 a_1 d) + (a_n 7 a_1 d) = 2 →
  (a_n 5 a_1 d) * (a_n 6 a_1 d) = -3 →
  a_1 * (a_n 10 a_1 d) = -323 :=
by
  sorry

end arithmetic_sequence_product_l230_230155


namespace fraction_evaluation_l230_230925

def number_of_primes_between_10_and_30 : ℕ := 6

theorem fraction_evaluation : (number_of_primes_between_10_and_30^2 - 4) / (number_of_primes_between_10_and_30 + 2) = 4 := by
  sorry

end fraction_evaluation_l230_230925


namespace pat_donut_selections_l230_230735

theorem pat_donut_selections : ∃ (n : ℕ), n = 10 :=
  let g' := 0
  let c' := 0
  let p' := 0
  let s' := 0
  have h : g' + c' + p' + s' = 2 := by sorry
  have binomial_calc := (5.choose 3) = 10 := by sorry
  ⟨10, binomial_calc⟩

end pat_donut_selections_l230_230735


namespace chemical_x_percentage_l230_230012

-- Define the initial volume of the mixture
def initial_volume : ℕ := 80

-- Define the percentage of chemical x in the initial mixture
def percentage_x_initial : ℚ := 0.30

-- Define the volume of chemical x added to the mixture
def added_volume_x : ℕ := 20

-- Define the calculation of the amount of chemical x in the initial mixture
def initial_amount_x : ℚ := percentage_x_initial * initial_volume

-- Define the calculation of the total amount of chemical x after adding more
def total_amount_x : ℚ := initial_amount_x + added_volume_x

-- Define the calculation of the total volume after adding 20 liters of chemical x
def total_volume : ℚ := initial_volume + added_volume_x

-- Define the percentage of chemical x in the final mixture
def percentage_x_final : ℚ := (total_amount_x / total_volume) * 100

-- The proof goal
theorem chemical_x_percentage : percentage_x_final = 44 := 
by
  sorry

end chemical_x_percentage_l230_230012


namespace minimum_value_of_quadratic_expression_l230_230357

theorem minimum_value_of_quadratic_expression : ∃ x ∈ ℝ, ∀ y ∈ ℝ, x^2 + 10 * x ≤ y^2 + 10 * y := by
  sorry

end minimum_value_of_quadratic_expression_l230_230357


namespace expression_is_integer_l230_230333

theorem expression_is_integer (n : ℕ) : 
  (3 ^ (2 * n) / 112 - 4 ^ (2 * n) / 63 + 5 ^ (2 * n) / 144) = (k : ℤ) :=
sorry

end expression_is_integer_l230_230333


namespace total_sheets_of_paper_l230_230727

theorem total_sheets_of_paper (num_classes : ℕ) (students_per_class : ℕ) (sheets_per_student : ℕ)
  (h1 : num_classes = 4) (h2 : students_per_class = 20) (h3 : sheets_per_student = 5) :
  (num_classes * students_per_class * sheets_per_student) = 400 :=
by
  sorry

end total_sheets_of_paper_l230_230727


namespace animal_costs_l230_230052

theorem animal_costs :
  ∃ (C G S P : ℕ),
      C + G + S + P = 1325 ∧
      G + S + P = 425 ∧
      C + S + P = 1225 ∧
      G + P = 275 ∧
      C = 900 ∧
      G = 100 ∧
      S = 150 ∧
      P = 175 :=
by
  sorry

end animal_costs_l230_230052


namespace max_height_l230_230373

def h (t : ℝ) : ℝ := -20 * t ^ 2 + 80 * t + 50

theorem max_height : ∃ t : ℝ, ∀ t' : ℝ, h t' ≤ h t ∧ h t = 130 :=
by
  sorry

end max_height_l230_230373


namespace scheme_choice_l230_230788

variable (x y₁ y₂ : ℕ)

def cost_scheme_1 (x : ℕ) : ℕ := 12 * x + 40

def cost_scheme_2 (x : ℕ) : ℕ := 16 * x

theorem scheme_choice :
  ∀ (x : ℕ), 5 ≤ x → x ≤ 20 →
  (if x < 10 then cost_scheme_2 x < cost_scheme_1 x else
   if x = 10 then cost_scheme_2 x = cost_scheme_1 x else
   cost_scheme_1 x < cost_scheme_2 x) :=
by
  sorry

end scheme_choice_l230_230788


namespace number_is_18_l230_230765

theorem number_is_18 (x : ℝ) (h : (7 / 3) * x = 42) : x = 18 :=
sorry

end number_is_18_l230_230765


namespace cubic_eq_roots_l230_230699

theorem cubic_eq_roots (x1 x2 x3 : ℕ) (P : ℕ) 
  (h1 : x1 + x2 + x3 = 10) 
  (h2 : x1 * x2 * x3 = 30) 
  (h3 : x1 * x2 + x2 * x3 + x3 * x1 = P) : 
  P = 31 := by
  sorry

end cubic_eq_roots_l230_230699


namespace domain_of_f_of_f_l230_230159

noncomputable def f (x : ℝ) : ℝ := (2 * x - 1) / (3 + x)

theorem domain_of_f_of_f :
  {x : ℝ | x ≠ -3 ∧ x ≠ -8 / 5} =
  {x : ℝ | ∃ y : ℝ, f x = y ∧ y ≠ -3 ∧ x ≠ -3} :=
by
  sorry

end domain_of_f_of_f_l230_230159


namespace no_such_function_exists_l230_230527

noncomputable def func_a (a : ℕ → ℕ) : Prop :=
  a 0 = 0 ∧ ∀ n : ℕ, a n = n - a (a n)

theorem no_such_function_exists : ¬ ∃ a : ℕ → ℕ, func_a a :=
by
  sorry

end no_such_function_exists_l230_230527


namespace gcd_7488_12467_eq_39_l230_230402

noncomputable def gcd_7488_12467 : ℕ := Nat.gcd 7488 12467

theorem gcd_7488_12467_eq_39 : gcd_7488_12467 = 39 :=
sorry

end gcd_7488_12467_eq_39_l230_230402


namespace sin_alpha_beta_l230_230976

theorem sin_alpha_beta (a b c α β : Real) (h₁ : a * Real.cos α + b * Real.sin α + c = 0)
  (h₂ : a * Real.cos β + b * Real.sin β + c = 0) (h₃ : 0 < α) (h₄ : α < β) (h₅ : β < π) :
  Real.sin (α + β) = (2 * a * b) / (a^2 + b^2) :=
by 
  sorry

end sin_alpha_beta_l230_230976


namespace twelve_row_triangle_pieces_l230_230239

theorem twelve_row_triangle_pieces :
  let S_n_arithmetic_sum (a d n : ℕ) := n * (2 * a + (n - 1) * d) / 2
  let total_rods := S_n_arithmetic_sum 3 3 12
  let total_connectors := S_n_arithmetic_sum 1 1 13
  total_rods + total_connectors = 325 :=
by
  sorry

end twelve_row_triangle_pieces_l230_230239


namespace solution_to_quadratic_inequality_l230_230666

def quadratic_inequality (x : ℝ) : Prop := 3 * x^2 - 5 * x > 9

theorem solution_to_quadratic_inequality (x : ℝ) : quadratic_inequality x ↔ x < -1 ∨ x > 3 :=
by
  sorry

end solution_to_quadratic_inequality_l230_230666


namespace sum_of_remainders_l230_230658

theorem sum_of_remainders (n : ℤ) (h : n % 18 = 11) : 
  (n % 2) + (n % 3) + (n % 6) + (n % 9) = 10 := 
by 
  sorry

end sum_of_remainders_l230_230658


namespace security_deposit_amount_l230_230182

-- Definitions of the given conditions
def daily_rate : ℝ := 125.00
def pet_fee : ℝ := 100.00
def service_cleaning_fee_percentage : ℝ := 0.20
def duration_in_days : ℝ := 14 -- 2 weeks
def security_deposit_percentage : ℝ := 0.50

-- Summarize the problem into a theorem
theorem security_deposit_amount :
  let total_rent := duration_in_days * daily_rate in
  let total_with_pet_fee := total_rent + pet_fee in
  let service_cleaning_fee := service_cleaning_fee_percentage * total_with_pet_fee in
  let total_with_service_fee := total_with_pet_fee + service_cleaning_fee in
  let security_deposit := security_deposit_percentage * total_with_service_fee in
  security_deposit = 1110.00 :=
by
  sorry

end security_deposit_amount_l230_230182


namespace test_score_range_l230_230579

theorem test_score_range
  (mark_score : ℕ) (least_score : ℕ) (highest_score : ℕ)
  (twice_least_score : mark_score = 2 * least_score)
  (mark_fixed : mark_score = 46)
  (highest_fixed : highest_score = 98) :
  (highest_score - least_score) = 75 :=
by
  sorry

end test_score_range_l230_230579


namespace percentage_of_useful_items_l230_230866

theorem percentage_of_useful_items
  (junk_percentage : ℚ)
  (useful_items junk_items total_items : ℕ)
  (h1 : junk_percentage = 0.70)
  (h2 : useful_items = 8)
  (h3 : junk_items = 28)
  (h4 : junk_percentage * total_items = junk_items) :
  (useful_items : ℚ) / (total_items : ℚ) * 100 = 20 :=
sorry

end percentage_of_useful_items_l230_230866


namespace fencing_rate_l230_230261

noncomputable def rate_per_meter (d : ℝ) (total_cost : ℝ) : ℝ :=
  let circumference := Real.pi * d
  total_cost / circumference

theorem fencing_rate (diameter cost : ℝ) (h₀ : diameter = 34) (h₁ : cost = 213.63) :
  rate_per_meter diameter cost = 2 := by
  sorry

end fencing_rate_l230_230261


namespace smallest_positive_multiple_l230_230910

theorem smallest_positive_multiple (n : ℕ) (h1 : n > 0) (h2 : n % 45 = 0) (h3 : n % 75 = 0) (h4 : n % 20 ≠ 0) :
  n = 225 :=
by
  sorry

end smallest_positive_multiple_l230_230910


namespace initial_percentage_rise_l230_230647

-- Definition of the conditions
def final_price_gain (P : ℝ) (x : ℝ) : Prop :=
  P * (1 + x / 100) * 0.9 * 0.85 = P * 1.03275

-- The statement to be proven
theorem initial_percentage_rise (P : ℝ) (x : ℝ) : final_price_gain P x → x = 35.03 :=
by
  sorry -- Proof to be filled in

end initial_percentage_rise_l230_230647


namespace initial_time_between_maintenance_checks_l230_230506

theorem initial_time_between_maintenance_checks (x : ℝ) (h1 : 1.20 * x = 30) : x = 25 := by
  sorry

end initial_time_between_maintenance_checks_l230_230506


namespace angle_reduction_l230_230751

theorem angle_reduction (θ : ℝ) : θ = 1303 → ∃ k : ℤ, θ = 360 * k - 137 := 
by  
  intro h 
  use 4 
  simp [h] 
  sorry

end angle_reduction_l230_230751


namespace caterpillar_count_proof_l230_230884

def number_of_caterpillars_after_events (initial : ℕ) (hatched : ℕ) (left : ℕ) : ℕ :=
  initial + hatched - left

theorem caterpillar_count_proof :
  number_of_caterpillars_after_events 14 4 8 = 10 :=
by
  simp [number_of_caterpillars_after_events]
  sorry

end caterpillar_count_proof_l230_230884


namespace factorize_expression_l230_230816

-- Defining the variables x and y as real numbers.
variable (x y : ℝ)

-- Statement of the proof problem.
theorem factorize_expression : 
  x * y^2 - x = x * (y + 1) * (y - 1) :=
sorry

end factorize_expression_l230_230816


namespace finite_steps_iff_power_of_2_l230_230656

-- Define the conditions of the problem
def S (k n : ℕ) : ℕ := (k * (k + 1) / 2) % n

-- Define the predicate to check if the game finishes in finite number of steps
def game_completes (n : ℕ) : Prop :=
  ∃ k : ℕ, ∀ i : ℕ, i < n → S (k + i) n ≠ S k n

-- The main statement to prove
theorem finite_steps_iff_power_of_2 (n : ℕ) : game_completes n ↔ ∃ t : ℕ, n = 2^t :=
sorry  -- Placeholder for the proof

end finite_steps_iff_power_of_2_l230_230656


namespace poly_expansion_sum_l230_230008

theorem poly_expansion_sum (A B C D E : ℤ) (x : ℤ):
  (x + 3) * (4 * x^3 - 2 * x^2 + 3 * x - 1) = A * x^4 + B * x^3 + C * x^2 + D * x + E → 
  A + B + C + D + E = 16 :=
by
  sorry

end poly_expansion_sum_l230_230008


namespace decodeMINT_l230_230484

def charToDigit (c : Char) : Option Nat :=
  match c with
  | 'G' => some 0
  | 'R' => some 1
  | 'E' => some 2
  | 'A' => some 3
  | 'T' => some 4
  | 'M' => some 5
  | 'I' => some 6
  | 'N' => some 7
  | 'D' => some 8
  | 'S' => some 9
  | _   => none

def decodeWord (word : String) : Option Nat :=
  let digitsOption := word.toList.map charToDigit
  if digitsOption.all Option.isSome then
    let digits := digitsOption.map Option.get!
    some (digits.foldl (λ acc d => 10 * acc + d) 0)
  else
    none

theorem decodeMINT : decodeWord "MINT" = some 5674 := by
  sorry

end decodeMINT_l230_230484


namespace complement_union_l230_230166

noncomputable def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 4}
def complement_U_A : Set ℕ := U \ A

theorem complement_union (U A B : Set ℕ) (hU : U = {0, 1, 2, 3, 4}) (hA : A = {1, 2, 3}) (hB : B = {2, 4}) :
  (complement_U_A ∪ B) = {0, 2, 4} := by
  sorry

end complement_union_l230_230166


namespace smallest_solution_l230_230363

def polynomial (x : ℝ) := x^4 - 34 * x^2 + 225 = 0

theorem smallest_solution : ∃ x : ℝ, polynomial x ∧ ∀ y : ℝ, polynomial y → x ≤ y := 
sorry

end smallest_solution_l230_230363


namespace jennifer_money_left_l230_230112

theorem jennifer_money_left (initial_amount : ℕ) (sandwich_fraction museum_ticket_fraction book_fraction : ℚ) 
  (h_initial : initial_amount = 90) 
  (h_sandwich : sandwich_fraction = 1/5)
  (h_museum_ticket : museum_ticket_fraction = 1/6)
  (h_book : book_fraction = 1/2) : 
  initial_amount - (initial_amount * sandwich_fraction + initial_amount * museum_ticket_fraction + initial_amount * book_fraction) = 12 :=
by
  sorry

end jennifer_money_left_l230_230112


namespace find_age_l230_230796

variable (x : ℤ)

def age_4_years_hence := x + 4
def age_4_years_ago := x - 4
def brothers_age := x - 6

theorem find_age (hx : x = 4 * (x + 4) - 4 * (x - 4) + 1/2 * (x - 6)) : x = 58 :=
sorry

end find_age_l230_230796


namespace average_last_4_matches_l230_230778

theorem average_last_4_matches 
  (avg_10 : ℝ) (avg_6 : ℝ) (result : ℝ)
  (h1 : avg_10 = 38.9)
  (h2 : avg_6 = 42)
  (h3 : result = 34.25) :
  let total_runs_10 := avg_10 * 10
  let total_runs_6 := avg_6 * 6
  let total_runs_4 := total_runs_10 - total_runs_6
  let avg_4 := total_runs_4 / 4
  avg_4 = result :=
  sorry

end average_last_4_matches_l230_230778


namespace cost_keyboard_l230_230242

def num_keyboards : ℕ := 15
def num_printers : ℕ := 25
def total_cost : ℝ := 2050
def cost_printer : ℝ := 70
def total_cost_printers : ℝ := num_printers * cost_printer
def total_cost_keyboards : ℝ := total_cost - total_cost_printers

theorem cost_keyboard : total_cost_keyboards / num_keyboards = 20 := by
  sorry

end cost_keyboard_l230_230242


namespace sasha_remainder_l230_230064

statement:
  theorem sasha_remainder (n a b c d : ℕ) (h1 : n = 102 * a + b) (h2 : n = 103 * c + d) (h3 : a + d = 20) (h4: 0 ≤ b ∧ b ≤ 101) : 
  b = 20 :=
by
  sorry

end sasha_remainder_l230_230064


namespace total_items_on_shelf_l230_230320

-- Given conditions
def initial_action_figures : Nat := 4
def initial_books : Nat := 22
def initial_video_games : Nat := 10

def added_action_figures : Nat := 6
def added_video_games : Nat := 3
def removed_books : Nat := 5

-- Definitions based on conditions
def final_action_figures : Nat := initial_action_figures + added_action_figures
def final_books : Nat := initial_books - removed_books
def final_video_games : Nat := initial_video_games + added_video_games

-- Claim to prove
theorem total_items_on_shelf : final_action_figures + final_books + final_video_games = 40 := by
  sorry

end total_items_on_shelf_l230_230320


namespace hair_ratio_l230_230132

theorem hair_ratio (washed : ℕ) (grow_back : ℕ) (brushed : ℕ) (n : ℕ)
  (hwashed : washed = 32)
  (hgrow_back : grow_back = 49)
  (heq : washed + brushed + 1 = grow_back) :
  (brushed : ℚ) / washed = 1 / 2 := 
by 
  sorry

end hair_ratio_l230_230132


namespace division_problem_l230_230912

theorem division_problem : (5 * 8) / 10 = 4 := by
  sorry

end division_problem_l230_230912


namespace neg_of_forall_sin_ge_neg_one_l230_230291

open Real

theorem neg_of_forall_sin_ge_neg_one :
  (¬ (∀ x : ℝ, sin x ≥ -1)) ↔ (∃ x0 : ℝ, sin x0 < -1) := by
  sorry

end neg_of_forall_sin_ge_neg_one_l230_230291


namespace average_speed_of_tiger_exists_l230_230920

-- Conditions
def head_start_distance (v_t : ℝ) : ℝ := 5 * v_t
def zebra_distance : ℝ := 6 * 55
def tiger_distance (v_t : ℝ) : ℝ := 6 * v_t

-- Problem statement
theorem average_speed_of_tiger_exists (v_t : ℝ) (h : zebra_distance = head_start_distance v_t + tiger_distance v_t) : v_t = 30 :=
by
  sorry

end average_speed_of_tiger_exists_l230_230920


namespace number_of_ways_to_choose_two_groups_l230_230854

theorem number_of_ways_to_choose_two_groups (Mathematics ComputerScience ModelAviation : Type) :
  (finset.card (finset.powerset (finset.insert Mathematics (finset.insert ComputerScience (finset.singleton ModelAviation))).filter (λ s, finset.card s = 2)) = 3) :=
by sorry

end number_of_ways_to_choose_two_groups_l230_230854


namespace abs_inequality_solution_l230_230959

theorem abs_inequality_solution (x : ℝ) :
  3 ≤ |x + 2| ∧ |x + 2| ≤ 7 ↔ (1 ≤ x ∧ x ≤ 5) ∨ (-9 ≤ x ∧ x ≤ -5) :=
by
  sorry

end abs_inequality_solution_l230_230959


namespace part1_part2_part3_l230_230161

-- Part 1: Prove that if the tangent line condition holds, then a = -2
theorem part1 (a : ℝ) (h : ∀ (x : ℝ), 6 * x - 2 * (1 / 2 * x ^ 2 - a * Real.log x) - 5 = 0) : 
  a = -2 := sorry

-- Part 2: Prove the range for a under the given conditions
theorem part2 (a : ℝ) (h : ∀ (x₁ x₂ : ℝ), x₁ ≠ x₂ → (1 / 2 * x₁ ^ 2 + a * Real.log x₁ - (1 / 2 * x₂ ^ 2 + a * Real.log x₂)) / (x₁ - x₂) > 2) : 
  1 ≤ a := sorry

-- Part 3: Prove the range for a given an interval condition
theorem part3 (a : ℝ) (h : ∃ x_0 ∈ Icc 1 Real.exp 1, (1 * x_0 - 1 / 2 * x_0 ^ 2 + a * Real.log x_0 + a / x_0 - 2) < 0) : 
  a ∈ Set.Iio (-2) ∪ Set.Ioi ((Real.exp 1 ^ 2 + 1) / (Real.exp 1 - 1)) := sorry

end part1_part2_part3_l230_230161


namespace negation_of_proposition_l230_230617

theorem negation_of_proposition :
  (¬ ∃ x_0 : ℝ, x_0^2 + x_0 - 1 > 0) ↔ (∀ x : ℝ, x^2 + x - 1 ≤ 0) :=
by
  sorry

end negation_of_proposition_l230_230617


namespace sequence_an_form_sum_cn_terms_l230_230289

theorem sequence_an_form (a_n S_n : ℕ → ℕ) (b_n : ℕ → ℕ) (h : ∀ n : ℕ, a_n n = 3/4 * S_n n + 2) :
  ∀ n : ℕ, b_n n = 2 * n + 1 :=
sorry 

theorem sum_cn_terms (a_n S_n : ℕ → ℕ) (b_n : ℕ → ℕ) (c_n : ℕ → ℕ) (T_n : ℕ → ℕ)
    (h : ∀ n : ℕ, a_n n = 3/4 * S_n n + 2)
    (hb : ∀ n : ℕ, b_n n = 2 * n + 1)
    (hc : ∀ n : ℕ, c_n n = 1 / (b_n n * b_n (n + 1))) :
  ∀ n : ℕ, T_n n = n / (3 * (2 * n + 3)) :=
sorry

end sequence_an_form_sum_cn_terms_l230_230289


namespace cost_price_6500_l230_230922

variable (CP SP : ℝ)

-- Condition 1: The selling price is 30% more than the cost price.
def selling_price (CP : ℝ) : ℝ := CP * 1.3

-- Condition 2: The selling price is Rs. 8450.
axiom selling_price_8450 : selling_price CP = 8450

-- Prove that the cost price of the computer table is Rs. 6500.
theorem cost_price_6500 : CP = 6500 :=
by
  sorry

end cost_price_6500_l230_230922


namespace g_f_4_eq_l230_230338

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom f_g_diff'ble (x : ℝ) : Differentiable ℝ f ∧ Differentiable ℝ g
axiom f_eq (x : ℝ) : x * g (f x) * (deriv f) (g x) * (deriv g) x = f (g x) * (deriv g) (f x) * (deriv f) x
axiom f_nonneg (x : ℝ) : 0 ≤ f x
axiom g_pos (x : ℝ) : 0 < g x
axiom f_g_integral (a : ℝ) : ∫ x in 0 .. a, f (g x) = 1 - (real.exp (-2 * a)) / 2
axiom g_f_zero : g (f 0) = 1

theorem g_f_4_eq : g (f 4) = real.exp (-16) := sorry

end g_f_4_eq_l230_230338


namespace neg_eight_degrees_celsius_meaning_l230_230301

-- Define the temperature in degrees Celsius
def temp_in_degrees_celsius (t : Int) : String :=
  if t >= 0 then toString t ++ "°C above zero"
  else toString (abs t) ++ "°C below zero"

-- Define the proof statement
theorem neg_eight_degrees_celsius_meaning :
  temp_in_degrees_celsius (-8) = "8°C below zero" :=
sorry

end neg_eight_degrees_celsius_meaning_l230_230301


namespace slope_of_tangent_line_l230_230414

theorem slope_of_tangent_line : ∃ k : ℝ, (∀ f : ℝ → ℝ, (∀ x, f x = Real.exp x) → 
  ∃ x0 : ℝ, k = Real.exp x0 ∧ f x0 = x0 * k ∧ (0, 0) ∈ {(x0, Real.exp x0)} ∧ k = Real.exp x0) ∧ k = Real.exp 1 :=
begin
  sorry
end

end slope_of_tangent_line_l230_230414


namespace dice_composite_probability_l230_230713

theorem dice_composite_probability (m n : ℕ) (h : Nat.gcd m n = 1) :
  (∃ m n : ℕ, (m * 36 = 29 * n) ∧ Nat.gcd m n = 1) → m + n = 65 :=
by {
  sorry
}

end dice_composite_probability_l230_230713


namespace gcd_of_256_180_600_l230_230632

theorem gcd_of_256_180_600 : Nat.gcd (Nat.gcd 256 180) 600 = 12 :=
by
  -- The proof would be placed here
  sorry

end gcd_of_256_180_600_l230_230632


namespace ordering_l230_230674

noncomputable def a : ℝ := 1 / (Real.exp 0.6)
noncomputable def b : ℝ := 0.4
noncomputable def c : ℝ := Real.log 1.4 / 1.4

theorem ordering : a > b ∧ b > c :=
by
  have ha : a = 1 / (Real.exp 0.6) := rfl
  have hb : b = 0.4 := rfl
  have hc : c = Real.log 1.4 / 1.4 := rfl
  sorry

end ordering_l230_230674


namespace lucille_total_revenue_l230_230459

theorem lucille_total_revenue (salary_ratio stock_ratio : ℕ) (salary_amount : ℝ) (h_ratio : salary_ratio / stock_ratio = 4 / 11) (h_salary : salary_amount = 800) : 
  ∃ total_revenue : ℝ, total_revenue = 3000 :=
by
  sorry

end lucille_total_revenue_l230_230459


namespace solve_equation_l230_230205

theorem solve_equation (x : ℝ) (h : x * (x - 3) = 10) : x = 5 ∨ x = -2 :=
by sorry

end solve_equation_l230_230205


namespace circle_passes_first_and_second_quadrants_l230_230582

theorem circle_passes_first_and_second_quadrants :
  ∀ (x y : ℝ), (x - 1)^2 + (y - 3)^2 = 4 → ((x ≥ 0 ∧ y ≥ 0) ∨ (x ≤ 0 ∧ y ≥ 0)) :=
by
  sorry

end circle_passes_first_and_second_quadrants_l230_230582


namespace work_days_l230_230505

theorem work_days (x : ℕ) (hx : 0 < x) :
  (1 / (x : ℚ) + 1 / 20) = 1 / 15 → x = 60 := by
sorry

end work_days_l230_230505


namespace mary_starting_weight_l230_230187

def initial_weight (final_weight lost_1 gained_2 lost_3 gained_4 : ℕ) : ℕ :=
  final_weight + (lost_3 - gained_4) + (gained_2 - lost_1) + lost_1

theorem mary_starting_weight :
  ∀ (final_weight lost_1 gained_2 lost_3 gained_4 : ℕ),
  final_weight = 81 →
  lost_1 = 12 →
  gained_2 = 2 * lost_1 →
  lost_3 = 3 * lost_1 →
  gained_4 = lost_1 / 2 →
  initial_weight final_weight lost_1 gained_2 lost_3 gained_4 = 99 :=
by
  intros final_weight lost_1 gained_2 lost_3 gained_4 h_final_weight h_lost_1 h_gained_2 h_lost_3 h_gained_4
  rw [h_final_weight, h_lost_1] at *
  rw [h_gained_2, h_lost_3, h_gained_4]
  unfold initial_weight
  sorry

end mary_starting_weight_l230_230187


namespace find_cos_alpha_l230_230969

theorem find_cos_alpha (α : ℝ) (h : (1 - Real.cos α) / Real.sin α = 3) : Real.cos α = -4/5 :=
by
  sorry

end find_cos_alpha_l230_230969


namespace smallest_positive_real_number_l230_230273

noncomputable def smallest_x : ℝ :=
  let x := 89 / 9 in x

theorem smallest_positive_real_number :
  ∀ x : ℝ, (x > 0 ∧ ⌊x^2⌋ - x * ⌊x⌋ = 8) → x ≥ smallest_x  :=
by
  sorry

end smallest_positive_real_number_l230_230273


namespace water_left_after_four_hours_l230_230564

def initial_water : ℕ := 40
def water_loss_per_hour : ℕ := 2
def added_water_hour3 : ℕ := 1
def added_water_hour4 : ℕ := 3
def total_hours : ℕ := 4

theorem water_left_after_four_hours :
  initial_water - (water_loss_per_hour * total_hours) + (added_water_hour3 + added_water_hour4) = 36 := by
  sorry

end water_left_after_four_hours_l230_230564


namespace problem1_problem2_l230_230150

-- Define the universe U
def U : Set ℝ := Set.univ

-- Define the sets A and B
def A : Set ℝ := { x | -4 < x ∧ x < 4 }
def B : Set ℝ := { x | x ≤ 1 ∨ x ≥ 3 }

-- Statement of the first proof problem: Prove A ∩ B is equal to the given set
theorem problem1 : A ∩ B = { x | -4 < x ∧ x ≤ 1 ∨ 4 > x ∧ x ≥ 3 } :=
by
  sorry

-- Statement of the second proof problem: Prove the complement of (A ∪ B) in the universe U is ∅
theorem problem2 : Set.compl (A ∪ B) = ∅ :=
by
  sorry

end problem1_problem2_l230_230150


namespace required_number_l230_230771

-- Define the main variables and conditions
variables {i : ℂ} (z : ℂ)
axiom i_squared : i^2 = -1

-- State the theorem that needs to be proved
theorem required_number (h : z + (4 - 8 * i) = 1 + 10 * i) : z = -3 + 18 * i :=
by {
  -- the exact steps for the proof will follow here
  sorry
}

end required_number_l230_230771


namespace proposition_3_correct_l230_230152

open Real

def is_obtuse (A B C : ℝ) : Prop :=
  A + B + C = π ∧ (A > π / 2 ∨ B > π / 2 ∨ C > π / 2)

theorem proposition_3_correct (A B C : ℝ) (h₀ : 0 < A) (h₁ : 0 < B) (h₂ : 0 < C) (h₃ : A + B + C = π)
  (h : sin A ^ 2 + sin B ^ 2 + cos C ^ 2 < 1) : is_obtuse A B C :=
by
  sorry

end proposition_3_correct_l230_230152


namespace solve_for_x_l230_230114

theorem solve_for_x (x : ℝ) (h : 3 * x = 16 - x + 4) : x = 5 := 
by
  sorry

end solve_for_x_l230_230114


namespace area_enclosed_by_3x2_l230_230048

theorem area_enclosed_by_3x2 (a b : ℝ) (h₀ : a = 0) (h₁ : b = 1) :
  ∫ (x : ℝ) in a..b, 3 * x^2 = 1 :=
by 
  rw [h₀, h₁]
  sorry

end area_enclosed_by_3x2_l230_230048


namespace part_1_part_2_part_3_l230_230956

def whiteHorseNumber (a b c : ℚ) : ℚ :=
  min (a - b) (min ((a - c) / 2) ((b - c) / 3))

theorem part_1 : 
  whiteHorseNumber (-2) (-4) 1 = -5/3 :=
by sorry

theorem part_2 : 
  max (whiteHorseNumber (-2) (-4) 1) (max (whiteHorseNumber (-2) 1 (-4)) 
  (max (whiteHorseNumber (-4) (-2) 1) (max (whiteHorseNumber (-4) 1 (-2)) 
  (max (whiteHorseNumber 1 (-4) (-2)) (whiteHorseNumber 1 (-2) (-4)) )))) = 2/3 :=
by sorry

theorem part_3 (x : ℚ) (h : ∃a b c : ℚ, a = -1 ∧ b = 6 ∧ c = x ∧ whiteHorseNumber a b c = 2) : 
  x = -7 ∨ x = 8 :=
by sorry

end part_1_part_2_part_3_l230_230956


namespace probability_of_drawing_1_boy_1_girl_l230_230342

theorem probability_of_drawing_1_boy_1_girl 
  (total_boys : ℕ) (total_girls : ℕ) (choose_two : ℕ) 
  (choose_boy_girl : ℕ) :
  total_boys = 3 → total_girls = 2 → choose_two = Nat.choose 5 2 → choose_boy_girl = Nat.choose 3 1 * Nat.choose 2 1 →
  (choose_boy_girl : ℚ) / choose_two = (3 / 5 : ℚ) :=
begin
  intros h1 h2 h3 h4,
  rw [h1, h2, h3, h4],
  simp,
  norm_num,
end

end probability_of_drawing_1_boy_1_girl_l230_230342


namespace student_avg_greater_actual_avg_l230_230650

theorem student_avg_greater_actual_avg
  (x y z : ℝ)
  (hxy : x < y)
  (hyz : y < z) :
  (x + y + 2 * z) / 4 > (x + y + z) / 3 := by
  sorry

end student_avg_greater_actual_avg_l230_230650


namespace movies_in_series_l230_230620

theorem movies_in_series :
  -- conditions 
  let number_books := 10
  let books_read := 14
  let book_read_vs_movies_extra := 5
  (∀ number_movies : ℕ, 
  (books_read = number_movies + book_read_vs_movies_extra) →
  -- question
  number_movies = 9) := sorry

end movies_in_series_l230_230620


namespace ball_placement_problem_l230_230607

noncomputable def num_ways_to_place_balls : ℕ :=
(choose 8 5) * (choose 3 2) * (choose 1 1) * (factorial 3) +
(choose 8 4) * (choose 4 3) * (choose 1 1) * (factorial 3)

theorem ball_placement_problem : num_ways_to_place_balls = 2688 :=
by sorry

end ball_placement_problem_l230_230607


namespace William_won_10_rounds_l230_230917

theorem William_won_10_rounds (H : ℕ) (total_rounds : H + (H + 5) = 15) : H + 5 = 10 := by
  sorry

end William_won_10_rounds_l230_230917


namespace k_less_than_two_l230_230691

theorem k_less_than_two
    (x : ℝ)
    (k : ℝ)
    (y : ℝ)
    (h : y = (2 - k) / x)
    (h1 : (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0)) : k < 2 :=
by
  sorry

end k_less_than_two_l230_230691


namespace inverse_proportion_first_third_quadrant_l230_230690

theorem inverse_proportion_first_third_quadrant (k : ℝ) :
  (∀ (x : ℝ), x ≠ 0 → ((x > 0 → (2 - k) / x > 0) ∧ (x < 0 → (2 - k) / x < 0))) → k < 2 :=
by
  sorry

end inverse_proportion_first_third_quadrant_l230_230690


namespace range_of_m_l230_230693

def A := {x : ℝ | x^2 - 3*x + 2 = 0}
def B (m : ℝ) := {x : ℝ | x^2 - 2*x + m = 0}

theorem range_of_m (m : ℝ) : (A ∪ B m = A) ↔ m ∈ Set.Ici 1 :=
by
  sorry

end range_of_m_l230_230693


namespace MrMartinBought2Cups_l230_230189

theorem MrMartinBought2Cups (c b : ℝ) (x : ℝ) (h1 : 3 * c + 2 * b = 12.75)
                             (h2 : x * c + 5 * b = 14.00)
                             (hb : b = 1.5) :
  x = 2 :=
sorry

end MrMartinBought2Cups_l230_230189


namespace cubes_in_fig_6_surface_area_fig_10_l230_230949

-- Define the function to calculate the number of unit cubes in Fig. n
def cubes_in_fig (n : ℕ) : ℕ := (n * (n + 1) * (2 * n + 1)) / 6

-- Define the function to calculate the surface area of the solid figure for Fig. n
def surface_area_fig (n : ℕ) : ℕ := 6 * n * n

-- Theorem statements
theorem cubes_in_fig_6 : cubes_in_fig 6 = 91 :=
by sorry

theorem surface_area_fig_10 : surface_area_fig 10 = 600 :=
by sorry

end cubes_in_fig_6_surface_area_fig_10_l230_230949


namespace combined_area_is_correct_l230_230006

def tract1_length := 300
def tract1_width  := 500
def tract2_length := 250
def tract2_width  := 630
def tract3_length := 350
def tract3_width  := 450
def tract4_length := 275
def tract4_width  := 600
def tract5_length := 325
def tract5_width  := 520

def area (length width : ℕ) : ℕ := length * width

theorem combined_area_is_correct :
  area tract1_length tract1_width +
  area tract2_length tract2_width +
  area tract3_length tract3_width +
  area tract4_length tract4_width +
  area tract5_length tract5_width = 799000 :=
by
  sorry

end combined_area_is_correct_l230_230006


namespace right_triangle_medians_right_triangle_l230_230902

theorem right_triangle_medians_right_triangle (a b c s_a s_b s_c : ℝ)
  (hyp_a_lt_b : a < b) (hyp_b_lt_c : b < c)
  (h_c_hypotenuse : c = Real.sqrt (a^2 + b^2))
  (h_sa : s_a^2 = b^2 + (a / 2)^2)
  (h_sb : s_b^2 = a^2 + (b / 2)^2)
  (h_sc : s_c^2 = (a^2 + b^2) / 4) :
  b = a * Real.sqrt 2 :=
by
  sorry

end right_triangle_medians_right_triangle_l230_230902


namespace three_digit_number_l230_230919

theorem three_digit_number (a b c : ℕ) (h1 : a + b + c = 10) (h2 : b = a + c) (h3 : 100 * c + 10 * b + a = 100 * a + 10 * b + c + 99) : (100 * a + 10 * b + c) = 253 := 
by
  sorry

end three_digit_number_l230_230919


namespace remainder_of_x_plus_2_pow_2022_l230_230769

theorem remainder_of_x_plus_2_pow_2022 (x : ℂ) :
  ∃ r : ℂ, ∃ q : ℂ, (x + 2)^2022 = q * (x^2 - x + 1) + r ∧ (r = x) :=
by
  sorry

end remainder_of_x_plus_2_pow_2022_l230_230769


namespace part_a_part_b_l230_230227

theorem part_a (a : ℤ) (k : ℤ) (h : a + 1 = 3 * k) : ∃ m : ℤ, 4 + 7 * a = 3 * m := by
  sorry

theorem part_b (a b : ℤ) (m n : ℤ) (h1 : 2 + a = 11 * m) (h2 : 35 - b = 11 * n) : ∃ p : ℤ, a + b = 11 * p := by
  sorry

end part_a_part_b_l230_230227


namespace expenditures_ratio_l230_230880

open Real

variables (I1 I2 E1 E2 : ℝ)
variables (x : ℝ)

theorem expenditures_ratio 
  (h1 : I1 = 4500)
  (h2 : I1 / I2 = 5 / 4)
  (h3 : I1 - E1 = 1800)
  (h4 : I2 - E2 = 1800) : 
  E1 / E2 = 3 / 2 :=
by
  have h5 : I1 / 5 = x := by sorry
  have h6 : I2 = 4 * x := by sorry
  have h7 : I2 = 3600 := by sorry
  have h8 : E1 = 2700 := by sorry
  have h9 : E2 = 1800 := by sorry
  exact sorry 

end expenditures_ratio_l230_230880


namespace how_many_engineers_l230_230578

theorem how_many_engineers (n : ℕ) (h₁ : 3 ≤ 8) (h₂ : 5 + 3 = 8) (h₃ : n > 0) 
  (h₄ : (Nat.choose 8 n) - (Nat.choose 5 n) = 46) : n = 3 :=
by 
  sorry

end how_many_engineers_l230_230578


namespace max_marks_l230_230176

theorem max_marks (marks_secured : ℝ) (percentage : ℝ) (max_marks : ℝ) 
  (h1 : marks_secured = 332) 
  (h2 : percentage = 83) 
  (h3 : percentage = (marks_secured / max_marks) * 100) 
  : max_marks = 400 :=
by
  sorry

end max_marks_l230_230176


namespace sochi_apartment_price_decrease_l230_230782

theorem sochi_apartment_price_decrease (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) :
  let moscow_rub_decrease := 0.2
  let moscow_eur_decrease := 0.4
  let sochi_rub_decrease := 0.1
  let new_moscow_rub := (1 - moscow_rub_decrease) * a
  let new_moscow_eur := (1 - moscow_eur_decrease) * b
  let ruble_to_euro := new_moscow_rub / new_moscow_eur
  let new_sochi_rub := (1 - sochi_rub_decrease) * a
  let new_sochi_eur := new_sochi_rub / ruble_to_euro
  let decrease_percentage := (b - new_sochi_eur) / b * 100
  decrease_percentage = 32.5 :=
by
  sorry

end sochi_apartment_price_decrease_l230_230782


namespace inequality_proof_l230_230858

variable (a b c d : ℝ)

theorem inequality_proof
  (h_pos: 0 < a ∧ a < 1 ∧ 0 < b ∧ b < 1 ∧ 0 < c ∧ c < 1 ∧ 0 < d ∧ d < 1)
  (h_product: a * b * c * d = (1 - a) * (1 - b) * (1 - c) * (1 - d)) : 
  (a + b + c + d) - (a + c) * (b + d) ≥ 1 :=
by
  sorry

end inequality_proof_l230_230858


namespace valid_triangle_side_l230_230306

theorem valid_triangle_side (x : ℝ) (h1 : 2 + x > 6) (h2 : 2 + 6 > x) (h3 : x + 6 > 2) : x = 6 :=
by
  sorry

end valid_triangle_side_l230_230306


namespace money_left_after_purchases_l230_230587

variable (initial_money : ℝ) (fraction_for_cupcakes : ℝ) (money_spent_on_milkshake : ℝ)

theorem money_left_after_purchases (h_initial : initial_money = 10)
  (h_fraction : fraction_for_cupcakes = 1/5)
  (h_milkshake : money_spent_on_milkshake = 5) :
  initial_money - (initial_money * fraction_for_cupcakes) - money_spent_on_milkshake = 3 := 
by
  sorry

end money_left_after_purchases_l230_230587


namespace polynomial_coefficient_product_identity_l230_230292

theorem polynomial_coefficient_product_identity (a_0 a_1 a_2 a_3 a_4 a_5 : ℤ)
  (h1 : a_0 + a_1 + a_2 + a_3 + a_4 + a_5 = 0)
  (h2 : a_0 - a_1 + a_2 - a_3 + a_4 - a_5 = 32) :
  (a_0 + a_2 + a_4) * (a_1 + a_3 + a_5) = -256 := 
by {
  sorry
}

end polynomial_coefficient_product_identity_l230_230292


namespace height_of_spherical_cap_case1_height_of_spherical_cap_case2_l230_230127

variable (R : ℝ) (c : ℝ)
variable (h_c_gt_1 : c > 1)

-- Case 1: Not including the circular cap in the surface area
theorem height_of_spherical_cap_case1 : ∃ m : ℝ, m = (2 * R * (c - 1)) / c :=
by
  sorry

-- Case 2: Including the circular cap in the surface area
theorem height_of_spherical_cap_case2 : ∃ m : ℝ, m = (2 * R * (c - 2)) / (c - 1) :=
by
  sorry

end height_of_spherical_cap_case1_height_of_spherical_cap_case2_l230_230127


namespace least_possible_value_of_z_minus_x_l230_230015

theorem least_possible_value_of_z_minus_x 
  (x y z : ℤ) 
  (hx : Even x) 
  (hy : Odd y) 
  (hz : Odd z) 
  (h1 : x < y) 
  (h2 : y < z) 
  (h3 : y - x > 5) : 
  z - x = 9 :=
sorry

end least_possible_value_of_z_minus_x_l230_230015


namespace jerry_bought_one_pound_of_pasta_sauce_l230_230024

-- Definitions of the given conditions
def cost_mustard_oil_per_liter : ℕ := 13
def liters_mustard_oil : ℕ := 2
def cost_pasta_per_pound : ℕ := 4
def pounds_pasta : ℕ := 3
def cost_pasta_sauce_per_pound : ℕ := 5
def leftover_amount : ℕ := 7
def initial_amount : ℕ := 50

-- The goal to prove
theorem jerry_bought_one_pound_of_pasta_sauce :
  (initial_amount - leftover_amount - liters_mustard_oil * cost_mustard_oil_per_liter 
  - pounds_pasta * cost_pasta_per_pound) / cost_pasta_sauce_per_pound = 1 :=
by
  sorry

end jerry_bought_one_pound_of_pasta_sauce_l230_230024


namespace compare_f_values_l230_230457

noncomputable def f (x : ℝ) : ℝ := 1 / (1 + x^2)

theorem compare_f_values (a : ℝ) (h_pos : 0 < a) :
  (a > 2 * Real.sqrt 2 → f a > f (a / 2) * f (a / 2)) ∧
  (a = 2 * Real.sqrt 2 → f a = f (a / 2) * f (a / 2)) ∧
  (0 < a ∧ a < 2 * Real.sqrt 2 → f a < f (a / 2) * f (a / 2)) :=
by
  sorry

end compare_f_values_l230_230457


namespace evaluate_nav_expression_l230_230345
noncomputable def nav (k m : ℕ) := k * (k - m)

theorem evaluate_nav_expression : (nav 5 1) + (nav 4 1) = 32 :=
by
  -- Skipping the proof as instructed
  sorry

end evaluate_nav_expression_l230_230345


namespace gain_percent_l230_230370

variable (MP CP SP : ℝ)

def costPrice (CP : ℝ) (MP : ℝ) := CP = 0.64 * MP

def sellingPrice (SP : ℝ) (MP : ℝ) := SP = MP * 0.88

theorem gain_percent (h1 : costPrice CP MP) (h2 : sellingPrice SP MP) : 
  ((SP - CP) / CP) * 100 = 37.5 :=
by
  sorry

end gain_percent_l230_230370


namespace num_employees_excluding_manager_l230_230612

/-- 
If the average monthly salary of employees is Rs. 1500, 
and adding a manager with salary Rs. 14100 increases 
the average salary by Rs. 600, prove that the number 
of employees (excluding the manager) is 20.
-/
theorem num_employees_excluding_manager 
  (avg_salary : ℕ) 
  (manager_salary : ℕ) 
  (new_avg_increase : ℕ) : 
  (∃ n : ℕ, 
    avg_salary = 1500 ∧ 
    manager_salary = 14100 ∧ 
    new_avg_increase = 600 ∧ 
    n = 20) := 
sorry

end num_employees_excluding_manager_l230_230612


namespace yellow_more_than_green_l230_230248

-- Given conditions
def G : ℕ := 90               -- Number of green buttons
def B : ℕ := 85               -- Number of blue buttons
def T : ℕ := 275              -- Total number of buttons
def Y : ℕ := 100              -- Number of yellow buttons (derived from conditions)

-- Mathematically equivalent proof problem
theorem yellow_more_than_green : (90 + 100 + 85 = 275) → (100 - 90 = 10) :=
by sorry

end yellow_more_than_green_l230_230248


namespace digit_H_value_l230_230616

theorem digit_H_value (E F G H : ℕ) (h1 : E < 10) (h2 : F < 10) (h3 : G < 10) (h4 : H < 10)
  (cond1 : 10 * E + F + 10 * G + E = 10 * H + E)
  (cond2 : 10 * E + F - (10 * G + E) = E)
  (cond3 : E + G = H + 1) : H = 8 :=
sorry

end digit_H_value_l230_230616


namespace square_perimeter_ratio_l230_230752

theorem square_perimeter_ratio (x y : ℝ)
(h : (x / y) ^ 2 = 16 / 25) : (4 * x) / (4 * y) = 4 / 5 :=
by sorry

end square_perimeter_ratio_l230_230752


namespace remaining_liquid_weight_l230_230194

theorem remaining_liquid_weight 
  (liqX_content : ℝ := 0.20)
  (water_content : ℝ := 0.80)
  (initial_solution : ℝ := 8)
  (evaporated_water : ℝ := 2)
  (added_solution : ℝ := 2)
  (new_solution_fraction : ℝ := 0.25) :
  ∃ (remaining_liquid : ℝ), remaining_liquid = 6 := 
by
  -- Skip the proof to ensure the statement is built successfully
  sorry

end remaining_liquid_weight_l230_230194


namespace street_sweeper_routes_l230_230649

def num_routes (A B C : Type) :=
  -- Conditions: Starts from point A, 
  -- travels through all streets exactly once, 
  -- and returns to point A.
  -- Correct Answer: Total routes = 12
  2 * 6 = 12

theorem street_sweeper_routes (A B C : Type) : num_routes A B C := by
  -- The proof is omitted as per instructions
  sorry

end street_sweeper_routes_l230_230649


namespace fifth_group_pythagorean_triples_l230_230011

theorem fifth_group_pythagorean_triples :
  ∃ (a b c : ℕ), (a, b, c) = (11, 60, 61) ∧ a^2 + b^2 = c^2 :=
by
  use 11, 60, 61
  sorry

end fifth_group_pythagorean_triples_l230_230011


namespace prob_xi_leq_neg2_l230_230552

variable {σ : ℝ} (ξ : ℝ → ℝ)

axiom h1 : ∀ x, ξ x ~ Normal 1 σ^2
axiom h2 : Pξ (ξ ≤ 4) = 0.84

theorem prob_xi_leq_neg2 : Pξ (ξ ≤ -2) = 0.16 := by
  sorry

end prob_xi_leq_neg2_l230_230552


namespace cells_at_day_10_l230_230509

-- Define a function to compute the number of cells given initial cells, tripling rate, intervals, and total time.
def number_of_cells (initial_cells : ℕ) (ratio : ℕ) (interval : ℕ) (total_time : ℕ) : ℕ :=
  let n := total_time / interval + 1
  initial_cells * ratio^(n-1)

-- State the main theorem
theorem cells_at_day_10 :
  number_of_cells 5 3 2 10 = 1215 := by
  sorry

end cells_at_day_10_l230_230509


namespace county_population_percentage_l230_230753

theorem county_population_percentage 
    (percent_less_than_20000 : ℝ)
    (percent_20000_to_49999 : ℝ) 
    (h1 : percent_less_than_20000 = 35) 
    (h2 : percent_20000_to_49999 = 40) : 
    percent_less_than_20000 + percent_20000_to_49999 = 75 := 
by
  sorry

end county_population_percentage_l230_230753


namespace natural_number_40_times_smaller_l230_230965

-- Define the sum of the first (n-1) natural numbers
def sum_natural_numbers (n : ℕ) := (n * (n - 1)) / 2

-- Define the proof statement
theorem natural_number_40_times_smaller (n : ℕ) (h : sum_natural_numbers n = 40 * n) : n = 81 :=
by {
  -- The proof is left as an exercise
  sorry
}

end natural_number_40_times_smaller_l230_230965


namespace inequality_min_value_l230_230614

theorem inequality_min_value (a : ℝ) : 
  (∀ x : ℝ, abs (x - 1) + abs (x + 2) ≥ a) → (a ≤ 3) := 
by
  sorry

end inequality_min_value_l230_230614


namespace necessary_condition_real_roots_l230_230406

theorem necessary_condition_real_roots (a : ℝ) :
  (a >= 1 ∨ a <= -2) → (∃ x : ℝ, x^2 - a * x + 1 = 0) :=
by
  sorry

end necessary_condition_real_roots_l230_230406


namespace parents_present_l230_230210

theorem parents_present (pupils teachers total_people parents : ℕ)
  (h_pupils : pupils = 724)
  (h_teachers : teachers = 744)
  (h_total_people : total_people = 1541) :
  parents = total_people - (pupils + teachers) :=
sorry

end parents_present_l230_230210


namespace decimal_to_vulgar_fraction_l230_230809

theorem decimal_to_vulgar_fraction (d : ℚ) (h : d = 0.36) : d = 9 / 25 :=
by {
  sorry
}

end decimal_to_vulgar_fraction_l230_230809


namespace arrange_desc_l230_230827

noncomputable def a : ℝ := Real.sin (33 * Real.pi / 180)
noncomputable def b : ℝ := Real.sin (35 * Real.pi / 180)
noncomputable def c : ℝ := Real.tan (35 * Real.pi / 180)
noncomputable def d : ℝ := Real.log 5

theorem arrange_desc : d > c ∧ c > b ∧ b > a := by
  sorry

end arrange_desc_l230_230827


namespace total_missed_questions_l230_230606

-- Definitions
def missed_by_you : ℕ := 36
def missed_by_friend : ℕ := 7
def missed_by_you_friends : ℕ := missed_by_you + missed_by_friend

-- Theorem
theorem total_missed_questions (h1 : missed_by_you = 5 * missed_by_friend) :
  missed_by_you_friends = 43 :=
by
  sorry

end total_missed_questions_l230_230606


namespace sum_even_odd_functions_l230_230295

theorem sum_even_odd_functions (f g : ℝ → ℝ) (h₁ : ∀ x, f (-x) = f x) (h₂ : ∀ x, g (-x) = -g x) (h₃ : ∀ x, f x - g x = x^3 + x^2 + 1) : 
  f 1 + g 1 = 1 := 
by 
  sorry

end sum_even_odd_functions_l230_230295


namespace total_vegetarian_is_33_l230_230577

-- Definitions of the quantities involved
def only_vegetarian : Nat := 19
def both_vegetarian_non_vegetarian : Nat := 12
def vegan_strictly_vegetarian : Nat := 3
def vegan_non_vegetarian : Nat := 2

-- The total number of people consuming vegetarian dishes
def total_vegetarian_consumers : Nat := only_vegetarian + both_vegetarian_non_vegetarian + vegan_non_vegetarian

-- Prove the number of people consuming vegetarian dishes
theorem total_vegetarian_is_33 :
  total_vegetarian_consumers = 33 :=
sorry

end total_vegetarian_is_33_l230_230577


namespace snow_first_day_eq_six_l230_230522

variable (snow_first_day snow_second_day snow_fourth_day snow_fifth_day : ℤ)

theorem snow_first_day_eq_six
  (h1 : snow_second_day = snow_first_day + 8)
  (h2 : snow_fourth_day = snow_second_day - 2)
  (h3 : snow_fifth_day = snow_fourth_day + 2 * snow_first_day)
  (h4 : snow_fifth_day = 24) :
  snow_first_day = 6 := by
  sorry

end snow_first_day_eq_six_l230_230522


namespace daily_evaporation_l230_230787

theorem daily_evaporation :
  ∀ (initial_amount : ℝ) (percentage_evaporated : ℝ) (days : ℕ),
  initial_amount = 10 →
  percentage_evaporated = 6 →
  days = 50 →
  (initial_amount * (percentage_evaporated / 100)) / days = 0.012 :=
by
  intros initial_amount percentage_evaporated days
  intros h_initial h_percentage h_days
  rw [h_initial, h_percentage, h_days]
  sorry

end daily_evaporation_l230_230787


namespace total_students_l230_230473

theorem total_students (teams students_per_team : ℕ) (h1 : teams = 9) (h2 : students_per_team = 18) :
  teams * students_per_team = 162 := by
  sorry

end total_students_l230_230473


namespace cos_product_l230_230389

theorem cos_product : 
  (1 + Real.cos (Real.pi / 12)) * (1 + Real.cos (5 * Real.pi / 12)) * (1 + Real.cos (7 * Real.pi / 12)) * (1 + Real.cos (11 * Real.pi / 12)) = 1 / 8 := 
by
  sorry

end cos_product_l230_230389


namespace range_of_m_l230_230279

open Set

theorem range_of_m (m : ℝ) :
  (∃ f : ℤ → Prop, (∀ x, f x ↔ x + 5 > 0 ∧ x - m ≤ 1) ∧ (∃ a b c : ℤ, f a ∧ f b ∧ f c))
  → (-3 ≤ m ∧ m < -2) := 
sorry

end range_of_m_l230_230279


namespace sandbox_perimeter_l230_230712

def sandbox_width : ℝ := 5
def sandbox_length := 2 * sandbox_width
def perimeter (length width : ℝ) := 2 * (length + width)

theorem sandbox_perimeter : perimeter sandbox_length sandbox_width = 30 := 
by
  sorry

end sandbox_perimeter_l230_230712


namespace smallest_positive_multiple_l230_230909

theorem smallest_positive_multiple (n : ℕ) (h1 : n > 0) (h2 : n % 45 = 0) (h3 : n % 75 = 0) (h4 : n % 20 ≠ 0) :
  n = 225 :=
by
  sorry

end smallest_positive_multiple_l230_230909


namespace inequality_holds_l230_230937

theorem inequality_holds (a : ℝ) : (∀ x : ℝ, a * x ^ 2 - 2 * a * x + 1 > 0) ↔ (0 < a ∧ a < 1) :=
sorry

end inequality_holds_l230_230937


namespace arithmetic_sequence_sum_l230_230856

theorem arithmetic_sequence_sum 
    (a : ℕ → ℤ)
    (h1 : ∀ n, a (n + 1) - a n = a 1 - a 0) -- Arithmetic sequence condition
    (h2 : a 5 = 3)
    (h3 : a 6 = -2) :
    (a 3 + a 4 + a 5 + a 6 + a 7 + a 8 = 3) :=
sorry

end arithmetic_sequence_sum_l230_230856


namespace chord_length_perpendicular_l230_230107

theorem chord_length_perpendicular 
  (R a b : ℝ)  
  (h1 : a + b = R)
  (h2 : (1 / 2) * Real.pi * R^2 - (1 / 2) * Real.pi * (a^2 + b^2) = 10 * Real.pi) :
  2 * Real.sqrt 10 = 6.32 :=
by 
  sorry

end chord_length_perpendicular_l230_230107


namespace quadratic_inequality_k_range_l230_230171

variable (k : ℝ)

theorem quadratic_inequality_k_range (h : ∀ x : ℝ, k * x^2 + 2 * k * x - (k + 2) < 0) :
  -1 < k ∧ k < 0 := by
sorry

end quadratic_inequality_k_range_l230_230171


namespace average_score_first_2_matches_l230_230198

theorem average_score_first_2_matches (A : ℝ) 
  (h1 : 3 * 40 = 120) 
  (h2 : 5 * 36 = 180) 
  (h3 : 2 * A + 120 = 180) : 
  A = 30 := 
by 
  have hA : 2 * A = 60 := by linarith [h3]
  have hA2 : A = 30 := by linarith [hA]
  exact hA2

end average_score_first_2_matches_l230_230198


namespace wings_area_l230_230608

-- Define the areas of the two cut triangles
def A1 : ℕ := 4
def A2 : ℕ := 9

-- Define the area of the wings (remaining two triangles)
def W : ℕ := 12

-- The proof goal
theorem wings_area (A1 A2 : ℕ) (W : ℕ) : A1 = 4 → A2 = 9 → W = 12 → A1 + A2 = 13 → W = 12 :=
by
  intros hA1 hA2 hW hTotal
  -- Sorry is used as a placeholder for the proof steps
  sorry

end wings_area_l230_230608


namespace hyperbola_a_value_l230_230162

theorem hyperbola_a_value (a : ℝ) :
  (∀ x y : ℝ, (x^2 / (a + 3) - y^2 / 3 = 1)) ∧ 
  (∀ e : ℝ, e = 2) → 
  a = -2 :=
by sorry

end hyperbola_a_value_l230_230162


namespace deliver_all_cargo_l230_230136

theorem deliver_all_cargo (containers : ℕ) (cargo_mass : ℝ) (ships : ℕ) (ship_capacity : ℝ)
  (h1 : containers ≥ 35)
  (h2 : cargo_mass = 18)
  (h3 : ships = 7)
  (h4 : ship_capacity = 3)
  (h5 : ∀ t, (0 < t) → (t ≤ containers) → (t = 35)) :
  (ships * ship_capacity) ≥ cargo_mass :=
by
  sorry

end deliver_all_cargo_l230_230136


namespace probability_heads_before_tails_l230_230456

noncomputable def solve_prob : ℚ := 
  let p := λ n : ℕ, if n = 4 then 1 else if n = 3 then (1 / 2 + 1 / 2 * t 1) else
                        if n = 2 then (1 / 2 * (1/2 + 1/2 * t 1) + 1 / 2 * t 1) else
                        if n = 1 then (1 / 2 * (1 / 4 + 3 / 4 * t 1) + 1 / 2 * t 1) else
                                   (1 / 2 * (1 / 8 + 7 / 8 * t 1) + 1 / 2 * t 1)
  and t := λ n : ℕ, if n = 2 then 0 else
                        if n = 1 then 1 / 2 * (t 1 + 1 / 2) else
                                   1 / 2 * (t 1 + t n)
  in p 0

theorem probability_heads_before_tails : solve_prob = 15/23 :=
sorry

end probability_heads_before_tails_l230_230456


namespace satisfying_integers_l230_230528

theorem satisfying_integers (a b : ℤ) :
  a^4 + (a + b)^4 + b^4 = x^2 → a = 0 ∧ b = 0 :=
by
  -- Proof is required to be filled in here.
  sorry

end satisfying_integers_l230_230528


namespace infinitely_many_lovely_no_lovely_square_gt_1_l230_230901

def lovely (n : ℕ) : Prop :=
  ∃ (k : ℕ) (d : Fin k → ℕ),
    n = (List.ofFn d).prod ∧
    ∀ i, (d i)^2 ∣ n + (d i)

theorem infinitely_many_lovely : ∀ N : ℕ, ∃ n > N, lovely n :=
  sorry

theorem no_lovely_square_gt_1 : ∀ n : ℕ, n > 1 → lovely n → ¬∃ m, n = m^2 :=
  sorry

end infinitely_many_lovely_no_lovely_square_gt_1_l230_230901


namespace product_sum_divisibility_l230_230618

theorem product_sum_divisibility (m n : ℕ) (h : (m + n) ∣ (m * n)) (hm : 0 < m) (hn : 0 < n) : m + n ≤ n^2 :=
sorry

end product_sum_divisibility_l230_230618


namespace star_value_l230_230700

-- Define the operation &
def and_operation (a b : ℕ) : ℕ := (a + b) * (a - b)

-- Define the operation star
def star_operation (c d : ℕ) : ℕ := and_operation c d + 2 * (c + d)

-- The proof problem
theorem star_value : star_operation 8 4 = 72 :=
by
  sorry

end star_value_l230_230700


namespace old_toilet_water_per_flush_correct_l230_230026

noncomputable def old_toilet_water_per_flush (water_saved : ℕ) (flushes_per_day : ℕ) (days_in_june : ℕ) (reduction_percentage : ℚ) : ℚ :=
  let total_flushes := flushes_per_day * days_in_june
  let water_saved_per_flush := water_saved / total_flushes
  let reduction_factor := reduction_percentage
  let original_water_per_flush := water_saved_per_flush / (1 - reduction_factor)
  original_water_per_flush

theorem old_toilet_water_per_flush_correct :
  old_toilet_water_per_flush 1800 15 30 (80 / 100) = 5 := by
  sorry

end old_toilet_water_per_flush_correct_l230_230026


namespace range_of_m_l230_230004

theorem range_of_m (m : ℝ) (h : ∀ x : ℝ, (m + 1) * x^2 ≥ 0) : m > -1 :=
by
  sorry

end range_of_m_l230_230004


namespace polygon_sides_from_diagonals_l230_230698

theorem polygon_sides_from_diagonals (D : ℕ) (hD : D = 16) : 
  ∃ n : ℕ, 2 * D = n * (n - 3) ∧ n = 7 :=
by
  use 7
  simp [hD]
  norm_num
  sorry

end polygon_sides_from_diagonals_l230_230698


namespace age_sum_proof_l230_230221

theorem age_sum_proof (a b c : ℕ) (h1 : a = b + 2) (h2 : b = 2 * c) (h3 : b = 20) : a + b + c = 52 :=
by
  sorry

end age_sum_proof_l230_230221


namespace most_people_can_attend_on_most_days_l230_230532

-- Define the days of the week as a type
inductive Day
| Mon | Tues | Wed | Thurs | Fri

open Day

-- Define the availability of each person
def is_available (person : String) (day : Day) : Prop :=
  match person, day with
  | "Anna", Mon => False
  | "Anna", Wed => False
  | "Anna", Fri => False
  | "Bill", Tues => False
  | "Bill", Thurs => False
  | "Bill", Fri => False
  | "Carl", Mon => False
  | "Carl", Tues => False
  | "Carl", Thurs => False
  | "Diana", Wed => False
  | "Diana", Fri => False
  | _, _ => True

-- Prove the result
theorem most_people_can_attend_on_most_days :
  {d : Day | d ∈ [Mon, Tues, Wed]} = {d : Day | ∀p : String, is_available p d → p ∈ ["Bill", "Carl", "Diana"] ∨ p ∉ ["Anna", "Bill"]} :=
sorry

end most_people_can_attend_on_most_days_l230_230532


namespace remaining_paint_fraction_l230_230644

theorem remaining_paint_fraction :
  ∀ (initial_paint : ℝ) (half_usage : ℕ → ℝ → ℝ),
    initial_paint = 2 →
    half_usage 0 (2 : ℝ) = 1 →
    half_usage 1 (1 : ℝ) = 0.5 →
    half_usage 2 (0.5 : ℝ) = 0.25 →
    half_usage 3 (0.25 : ℝ) = (0.25 / initial_paint) := by
  sorry

end remaining_paint_fraction_l230_230644


namespace right_triangle_area_l230_230478

theorem right_triangle_area (base hypotenuse : ℕ) (h_base : base = 8) (h_hypotenuse : hypotenuse = 10) :
  ∃ height : ℕ, height^2 = hypotenuse^2 - base^2 ∧ (base * height) / 2 = 24 :=
by
  sorry

end right_triangle_area_l230_230478


namespace total_sheets_of_paper_l230_230730

theorem total_sheets_of_paper (classes : ℕ) (students_per_class : ℕ) (sheets_per_student : ℕ) 
  (h1 : classes = 4) (h2 : students_per_class = 20) (h3 : sheets_per_student = 5) : 
  (classes * students_per_class * sheets_per_student) = 400 := 
by {
  sorry
}

end total_sheets_of_paper_l230_230730


namespace maximum_a3_S10_l230_230439

-- Given definitions of the problem
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

def conditions (a : ℕ → ℝ) : Prop :=
  is_arithmetic_sequence a ∧ (∀ n, a n > 0) ∧ (a 1 + a 3 + a 8 = a 4 ^ 2)

-- The problem statement
theorem maximum_a3_S10 (a : ℕ → ℝ) (h : conditions a) : 
  (∃ S : ℝ, S = a 3 * ((10 / 2) * (a 1 + a 10)) ∧ S ≤ 375 / 4) :=
sorry

end maximum_a3_S10_l230_230439


namespace mod_product_l230_230196

theorem mod_product (n : ℕ) (h1 : 0 ≤ n) (h2 : n < 50) : 
  173 * 927 % 50 = n := 
  by
    sorry

end mod_product_l230_230196


namespace locus_of_C_l230_230555

open EuclideanSpace

variables {E : Type*} [normed_add_comm_group E] [inner_product_space ℝ E]
variables (A B C : E)
variables [ordered_ring ℝ]

def midpoint (A B : E) : E := (A + B) / 2

-- Assuming the segment AB has midpoint D
let D := midpoint A B

-- E is the perpendicular projection of C onto AB
def projection (C A B : E) : E := 
  A + (((C - A).dot_product (B - A)) / ((B - A).dot_product (B - A))) • (B - A)

let E := projection C A B

-- The internal angle bisector of ∠ACB bisects the segment DE
def angle_bisector_bisects (A B C : E) : Prop :=
  let F := midpoint (angle_bisector_point C A B) D in
  let DE := dist D E in
  dist D F = DE / 2

def is_perpendicular_bisector (C : E) (A B : E) : Prop :=
  let m := (A + B) / 2 in
  dist C m = sqrt (dist A B) / 2 

def is_ellipse (C : E) (A B : E) : Prop :=
  (norm (C - A) + norm (C - B)) = sqrt (2) * norm (B - A)

theorem locus_of_C (h : angle_bisector_bisects A B C) : 
  is_perpendicular_bisector C A B ∨ is_ellipse C A B :=
sorry

end locus_of_C_l230_230555


namespace solve_other_endpoint_l230_230481

structure Point where
  x : ℤ
  y : ℤ

def midpoint : Point := { x := 3, y := 1 }
def known_endpoint : Point := { x := 7, y := -3 }

def calculate_other_endpoint (m k : Point) : Point :=
  let x2 := 2 * m.x - k.x;
  let y2 := 2 * m.y - k.y;
  { x := x2, y := y2 }

theorem solve_other_endpoint : calculate_other_endpoint midpoint known_endpoint = { x := -1, y := 5 } :=
  sorry

end solve_other_endpoint_l230_230481


namespace total_number_of_plugs_l230_230422

variables (pairs_mittens pairs_plugs : ℕ)

-- Conditions
def initial_pairs_mittens : ℕ := 150
def initial_pairs_plugs : ℕ := initial_pairs_mittens + 20
def added_pairs_plugs : ℕ := 30
def total_pairs_plugs : ℕ := initial_pairs_plugs + added_pairs_plugs

-- The proposition we're going to prove:
theorem total_number_of_plugs : initial_pairs_mittens = 150 ∧ initial_pairs_plugs = initial_pairs_mittens + 20 ∧ added_pairs_plugs = 30 → 
  total_pairs_plugs * 2 = 400 := sorry

end total_number_of_plugs_l230_230422


namespace range_of_a_l230_230982

open Set

noncomputable def setA (a : ℝ) : Set ℝ := {x : ℝ | x^2 - 2*x + a ≥ 0}

theorem range_of_a (a : ℝ) : (1 ∉ setA a) → a < 1 :=
sorry

end range_of_a_l230_230982


namespace machine_B_fewer_bottles_l230_230490

-- Definitions and the main theorem statement
def MachineA_caps_per_minute : ℕ := 12
def MachineC_additional_capacity : ℕ := 5
def total_bottles_in_10_minutes : ℕ := 370

theorem machine_B_fewer_bottles (B : ℕ) 
  (h1 : MachineA_caps_per_minute * 10 + 10 * B + 10 * (B + MachineC_additional_capacity) = total_bottles_in_10_minutes) :
  MachineA_caps_per_minute - B = 2 :=
by
  sorry

end machine_B_fewer_bottles_l230_230490


namespace smallest_positive_x_l230_230267

theorem smallest_positive_x (x : ℝ) (h : ⌊x^2⌋ - x * ⌊x⌋ = 8) : x = 89 / 9 :=
sorry

end smallest_positive_x_l230_230267


namespace probability_not_touch_outer_edge_l230_230871

def checkerboard : ℕ := 10

def total_squares : ℕ := checkerboard * checkerboard

def perimeter_squares : ℕ := 4 * checkerboard - 4

def inner_squares : ℕ := total_squares - perimeter_squares

def probability : ℚ := inner_squares / total_squares

theorem probability_not_touch_outer_edge : probability = 16 / 25 :=
by
  sorry

end probability_not_touch_outer_edge_l230_230871


namespace union_complements_l230_230330

open Set

variable (U : Set ℕ) (A B : Set ℕ)

-- Define the conditions
def condition_U : U = {1, 2, 3, 4, 5} := by
  sorry

def condition_A : A = {1, 2, 3} := by
  sorry

def condition_B : B = {2, 3, 4} := by
  sorry

-- Prove that (complement_U A) ∪ (complement_U B) = {1, 4, 5}
theorem union_complements :
  (U \ A) ∪ (U \ B) = {1, 4, 5} := by
  sorry

end union_complements_l230_230330


namespace serves_probability_l230_230710

variable (p : ℝ) (hpos : 0 < p) (hneq0 : p ≠ 0)

def ExpectedServes (p : ℝ) : ℝ :=
  p + 2 * p * (1 - p) + 3 * (1 - p) ^ 2

theorem serves_probability (h : ExpectedServes p > 1.75) : 0 < p ∧ p < 1 / 2 :=
  sorry

end serves_probability_l230_230710


namespace remainder_of_power_mod_l230_230637

theorem remainder_of_power_mod 
  (n : ℕ)
  (h₁ : 7 ≡ 1 [MOD 6]) : 7^51 ≡ 1 [MOD 6] := 
sorry

end remainder_of_power_mod_l230_230637


namespace odd_numbers_divisibility_l230_230736

theorem odd_numbers_divisibility 
  (a b c : ℤ) 
  (h_a_odd : a % 2 = 1) 
  (h_b_odd : b % 2 = 1) 
  (h_c_odd : c % 2 = 1) 
  : (ab - 1) % 4 = 0 ∨ (bc - 1) % 4 = 0 ∨ (ca - 1) % 4 = 0 := 
sorry

end odd_numbers_divisibility_l230_230736


namespace solid_circles_2006_l230_230651

noncomputable def circlePattern : Nat → Nat
| n => (2 + n * (n + 3)) / 2

theorem solid_circles_2006 :
  ∃ n, circlePattern n < 2006 ∧ circlePattern (n + 1) > 2006 ∧ n = 61 :=
by
  sorry

end solid_circles_2006_l230_230651


namespace tycho_jogging_schedule_count_l230_230213

-- Definition of the conditions
def non_consecutive_shot_schedule (days : Finset ℕ) : Prop :=
  ∀ day ∈ days, ∀ next_day ∈ days, day < next_day → next_day - day > 1

-- Definition stating there are exactly seven valid schedules
theorem tycho_jogging_schedule_count :
  ∃ (S : Finset (Finset ℕ)), (∀ s ∈ S, s.card = 3 ∧ non_consecutive_shot_schedule s) ∧ S.card = 7 := 
sorry

end tycho_jogging_schedule_count_l230_230213


namespace total_number_of_trees_l230_230209

theorem total_number_of_trees (D P : ℕ) (cost_D cost_P total_cost : ℕ)
  (hD : D = 350)
  (h_cost_D : cost_D = 300)
  (h_cost_P : cost_P = 225)
  (h_total_cost : total_cost = 217500)
  (h_cost_equation : cost_D * D + cost_P * P = total_cost) :
  D + P = 850 :=
by
  rw [hD, h_cost_D, h_cost_P, h_total_cost] at h_cost_equation
  sorry

end total_number_of_trees_l230_230209


namespace max_x_y3_z4_l230_230599

noncomputable def max_value_expression (x y z : ℝ) : ℝ :=
  x + y^3 + z^4

theorem max_x_y3_z4 (x y z : ℝ) (h₀ : 0 ≤ x) (h₁ : 0 ≤ y) (h₂ : 0 ≤ z) (h₃ : x + y + z = 1) :
  max_value_expression x y z ≤ 1 :=
sorry

end max_x_y3_z4_l230_230599


namespace lap_length_l230_230494

theorem lap_length (I P : ℝ) (K : ℝ) 
  (h1 : 2 * I - 2 * P = 3 * K) 
  (h2 : 3 * I + 10 - 3 * P = 7 * K) : 
  K = 4 :=
by 
  -- Proof goes here
  sorry

end lap_length_l230_230494


namespace my_car_mpg_l230_230868

-- Definitions from the conditions.
def total_miles := 100
def total_gallons := 5

-- The statement we need to prove.
theorem my_car_mpg : (total_miles / total_gallons : ℕ) = 20 :=
by
  sorry

end my_car_mpg_l230_230868


namespace gcd_of_256_180_600_l230_230625

-- Define the numbers
def n1 := 256
def n2 := 180
def n3 := 600

-- Statement to prove the GCD of the three numbers is 12
theorem gcd_of_256_180_600 : Nat.gcd (Nat.gcd n1 n2) n3 = 12 := 
by {
  -- Prime factorizations given as part of the conditions:
  have pf1 : n1 = 2^8 := by norm_num,
  have pf2 : n2 = 2^2 * 3^2 * 5 := by norm_num,
  have pf3 : n3 = 2^3 * 3 * 5^2 := by norm_num,
  
  -- Calculate GCD step by step should be done here, 
  -- But replaced by sorry as per the instructions.
  sorry
}

end gcd_of_256_180_600_l230_230625


namespace number_of_yellow_crayons_l230_230093

theorem number_of_yellow_crayons : 
  ∃ (R B Y : ℕ), 
  R = 14 ∧ 
  B = R + 5 ∧ 
  Y = 2 * B - 6 ∧ 
  Y = 32 :=
by
  sorry

end number_of_yellow_crayons_l230_230093


namespace math_problem_l230_230001

theorem math_problem
  (x y z : ℤ)
  (hz : z ≠ 0)
  (eq1 : 2 * x - 3 * y - z = 0)
  (eq2 : x + 3 * y - 14 * z = 0) :
  (x^2 - x * y) / (y^2 + 2 * z^2) = 10 / 11 := 
by 
  sorry

end math_problem_l230_230001


namespace bobby_initial_candy_l230_230254

theorem bobby_initial_candy (initial_candy : ℕ) (remaining_candy : ℕ) (extra_candy : ℕ) (total_eaten : ℕ)
  (h_candy_initial : initial_candy = 36)
  (h_candy_remaining : remaining_candy = 4)
  (h_candy_extra : extra_candy = 15)
  (h_candy_total_eaten : total_eaten = initial_candy - remaining_candy) :
  total_eaten - extra_candy = 17 :=
by
  sorry

end bobby_initial_candy_l230_230254


namespace sasha_remainder_20_l230_230075

theorem sasha_remainder_20
  (n a b c d : ℕ)
  (h1 : n = 102 * a + b)
  (h2 : 0 ≤ b ∧ b ≤ 101)
  (h3 : n = 103 * c + d)
  (h4 : d = 20 - a) :
  b = 20 :=
by
  sorry

end sasha_remainder_20_l230_230075


namespace john_score_l230_230447

theorem john_score (s1 s2 s3 s4 s5 s6 : ℕ) (h1 : s1 = 85) (h2 : s2 = 88) (h3 : s3 = 90) (h4 : s4 = 92) (h5 : s5 = 83) (h6 : s6 = 102) :
  (s1 + s2 + s3 + s4 + s5 + s6) / 6 = 90 :=
by
  sorry

end john_score_l230_230447


namespace bottle_caps_found_l230_230654

theorem bottle_caps_found
  (caps_current : ℕ) 
  (caps_earlier : ℕ) 
  (h_current : caps_current = 32) 
  (h_earlier : caps_earlier = 25) :
  caps_current - caps_earlier = 7 :=
by 
  sorry

end bottle_caps_found_l230_230654


namespace max_gold_coins_l230_230773

theorem max_gold_coins (n : ℤ) (h₁ : ∃ k : ℤ, n = 13 * k + 3) (h₂ : n < 150) : n ≤ 146 :=
by {
  sorry -- Proof not required as per instructions
}

end max_gold_coins_l230_230773


namespace ratio_KL_eq_3_over_5_l230_230665

theorem ratio_KL_eq_3_over_5
  (K L : ℤ)
  (h : ∀ (x : ℝ), x ≠ -3 ∧ x ≠ 0 ∧ x ≠ 3 → 
    (K : ℝ) / (x + 3) + (L : ℝ) / (x^2 - 3 * x) = (x^2 - x + 5) / (x^3 + x^2 - 9 * x)):
  (K : ℝ) / (L : ℝ) = 3 / 5 :=
by
  sorry

end ratio_KL_eq_3_over_5_l230_230665


namespace union_is_correct_l230_230432

def A : Set ℝ := { x | 3 ≤ x ∧ x < 7 }
def B : Set ℝ := { x | 2 < x ∧ x < 10 }

theorem union_is_correct : A ∪ B = { x : ℝ | 2 < x ∧ x < 10 } :=
by
  sorry

end union_is_correct_l230_230432


namespace cyclist_A_speed_l230_230894

theorem cyclist_A_speed (a b : ℝ) (h1 : b = a + 5)
    (h2 : 80 / a = 120 / b) : a = 10 :=
by
  sorry

end cyclist_A_speed_l230_230894


namespace alice_monthly_salary_l230_230655

-- Definitions for given conditions
def commission_rate : ℝ := 0.02
def sales : ℝ := 2500
def savings : ℝ := 29
def savings_rate : ℝ := 0.10

-- Intermediate calculations from the problem
def commission : ℝ := commission_rate * sales
def total_earnings : ℝ := 10 * savings

-- The problem to be proven
theorem alice_monthly_salary : ∃ S : ℝ, total_earnings = S + commission ∧ S = 240 :=
by
  have commission_calc : commission = 50 := by
    unfold commission
    norm_num
  have total_earnings_calc : total_earnings = 290 := by
    unfold total_earnings
    norm_num
  use (total_earnings - commission)
  split
  case left => 
    norm_num at total_earnings_calc commission_calc
    rw [total_earnings_calc, commission_calc]
    ring
  case right => 
    norm_num

end alice_monthly_salary_l230_230655


namespace forty_percent_jacqueline_candy_l230_230673

def fred_candy : ℕ := 12
def uncle_bob_candy : ℕ := fred_candy + 6
def total_fred_uncle_bob_candy : ℕ := fred_candy + uncle_bob_candy
def jacqueline_candy : ℕ := 10 * total_fred_uncle_bob_candy

theorem forty_percent_jacqueline_candy : (40 * jacqueline_candy) / 100 = 120 := by
  sorry

end forty_percent_jacqueline_candy_l230_230673


namespace fifth_inequality_proof_l230_230463

theorem fifth_inequality_proof :
  (1 + 1 / (2^2 : ℝ) + 1 / (3^2 : ℝ) + 1 / (4^2 : ℝ) + 1 / (5^2 : ℝ) + 1 / (6^2 : ℝ) < 11 / 6) 
  := 
sorry

end fifth_inequality_proof_l230_230463


namespace sum_of_possible_values_of_G_F_l230_230428

theorem sum_of_possible_values_of_G_F (G F : ℕ) (hG : 0 ≤ G ∧ G ≤ 9) (hF : 0 ≤ F ∧ F ≤ 9)
  (hdiv : (G + 2 + 4 + 3 + F + 1 + 6) % 9 = 0) : G + F = 2 ∨ G + F = 11 → 2 + 11 = 13 :=
by { sorry }

end sum_of_possible_values_of_G_F_l230_230428


namespace cube_volume_is_27_l230_230237

noncomputable def original_cube_edge (a : ℝ) : ℝ := a

noncomputable def original_cube_volume (a : ℝ) : ℝ := a^3

noncomputable def new_rectangular_solid_volume (a : ℝ) : ℝ := (a-2) * a * (a+2)

theorem cube_volume_is_27 (a : ℝ) (h : original_cube_volume a - new_rectangular_solid_volume a = 14) : original_cube_volume a = 27 :=
by
  sorry

end cube_volume_is_27_l230_230237


namespace Miss_Adamson_paper_usage_l230_230723

-- Definitions from the conditions
def classes : ℕ := 4
def students_per_class : ℕ := 20
def sheets_per_student : ℕ := 5

-- Total number of students
def total_students : ℕ := classes * students_per_class

-- Total number of sheets of paper
def total_sheets : ℕ := total_students * sheets_per_student

-- The proof problem
theorem Miss_Adamson_paper_usage : total_sheets = 400 :=
by
  -- Proof to be filled in
  sorry

end Miss_Adamson_paper_usage_l230_230723


namespace betty_gave_stuart_percentage_l230_230944

theorem betty_gave_stuart_percentage (P : ℝ) 
  (betty_marbles : ℝ := 60) 
  (stuart_initial_marbles : ℝ := 56) 
  (stuart_final_marbles : ℝ := 80)
  (increase_in_stuart_marbles : ℝ := stuart_final_marbles - stuart_initial_marbles)
  (betty_to_stuart : ℝ := (P / 100) * betty_marbles) :
  56 + ((P / 100) * betty_marbles) = 80 → P = 40 :=
by
  intros h
  -- Sorry is used since the proof steps are not required
  sorry

end betty_gave_stuart_percentage_l230_230944


namespace convert_2e_15pi_i4_to_rectangular_form_l230_230256

noncomputable def convert_to_rectangular_form (z : ℂ) : ℂ :=
  let θ := (15 * Real.pi) / 4
  let θ' := θ - 2 * Real.pi
  2 * Complex.exp (θ' * Complex.I)

theorem convert_2e_15pi_i4_to_rectangular_form :
  convert_to_rectangular_form (2 * Complex.exp ((15 * Real.pi) / 4 * Complex.I)) = (Real.sqrt 2 - Complex.I * Real.sqrt 2) :=
  sorry

end convert_2e_15pi_i4_to_rectangular_form_l230_230256


namespace rate_of_second_batch_of_wheat_l230_230385

theorem rate_of_second_batch_of_wheat (total_cost1 cost_per_kg1 weight1 weight2 total_weight total_cost selling_price_per_kg profit_rate cost_per_kg2 : ℝ)
  (H1 : total_cost1 = cost_per_kg1 * weight1)
  (H2 : total_weight = weight1 + weight2)
  (H3 : total_cost = total_cost1 + cost_per_kg2 * weight2)
  (H4 : selling_price_per_kg = (1 + profit_rate) * total_cost / total_weight)
  (H5 : profit_rate = 0.30)
  (H6 : cost_per_kg1 = 11.50)
  (H7 : weight1 = 30)
  (H8 : weight2 = 20)
  (H9 : selling_price_per_kg = 16.38) :
  cost_per_kg2 = 14.25 :=
by
  sorry

end rate_of_second_batch_of_wheat_l230_230385


namespace slope_of_tangent_line_l230_230413

theorem slope_of_tangent_line (f : ℝ → ℝ) (f_deriv : ∀ x, deriv f x = f x) (h_tangent : ∃ x₀, f x₀ = x₀ * deriv f x₀ ∧ (0 < f x₀)) :
  ∃ k, k = Real.exp 1 :=
by
  sorry

end slope_of_tangent_line_l230_230413


namespace range_of_a_plus_b_at_least_one_nonnegative_l230_230829

-- Conditions
variable (x : ℝ) (a := x^2 - 1) (b := 2 * x + 2)

-- Proof Problem 1: Prove that the range of a + b is [0, +∞)
theorem range_of_a_plus_b : (a + b) ≥ 0 :=
by sorry

-- Proof Problem 2: Prove by contradiction that at least one of a or b is greater than or equal to 0
theorem at_least_one_nonnegative : ¬(a < 0 ∧ b < 0) :=
by sorry

end range_of_a_plus_b_at_least_one_nonnegative_l230_230829


namespace range_of_a_l230_230847

theorem range_of_a (a : ℝ) :
  ¬ ∃ x : ℝ, x^2 + (a - 1) * x + 1 ≤ 0 ↔ a ∈ Set.Ioo (-1 : ℝ) (3 : ℝ) :=
by
  sorry

end range_of_a_l230_230847


namespace barrel_capacity_l230_230786

theorem barrel_capacity (x y : ℝ) (h1 : y = 45 / (3/5)) (h2 : 0.6*x = y*3/5) (h3 : 0.4*x = 18) : 
  y = 75 :=
by
  sorry

end barrel_capacity_l230_230786


namespace LindaCandiesLeft_l230_230601

variable (initialCandies : ℝ)
variable (candiesGiven : ℝ)

theorem LindaCandiesLeft (h1 : initialCandies = 34.0) (h2 : candiesGiven = 28.0) : initialCandies - candiesGiven = 6.0 := by
  sorry

end LindaCandiesLeft_l230_230601


namespace sasha_remainder_l230_230091

theorem sasha_remainder (n a b c d : ℕ) (h1 : n = 102 * a + b) (h2 : n = 103 * c + d)
  (h3 : d = 20 - a) (h4 : 0 ≤ b ∧ b ≤ 101) : b = 20 :=
by
  sorry

end sasha_remainder_l230_230091


namespace sasha_remainder_l230_230087

theorem sasha_remainder (n a b c d : ℕ) (h1 : n = 102 * a + b) (h2 : n = 103 * c + d)
  (h3 : d = 20 - a) (h4 : 0 ≤ b ∧ b ≤ 101) : b = 20 :=
by
  sorry

end sasha_remainder_l230_230087


namespace eggs_left_after_capital_recovered_l230_230041

-- Conditions as definitions
def eggs_in_crate := 30
def crate_cost_dollars := 5
def price_per_egg_cents := 20

-- The amount of cents in a dollar
def cents_per_dollar := 100

-- Total cost in cents
def crate_cost_cents := crate_cost_dollars * cents_per_dollar

-- The number of eggs needed to recover the capital
def eggs_to_recover_capital := crate_cost_cents / price_per_egg_cents

-- The number of eggs left
def eggs_left := eggs_in_crate - eggs_to_recover_capital

-- The theorem stating the problem
theorem eggs_left_after_capital_recovered : eggs_left = 5 :=
by
  sorry

end eggs_left_after_capital_recovered_l230_230041


namespace inequality_solution_absolute_inequality_l230_230116

-- Statement for Inequality Solution Problem
theorem inequality_solution (x : ℝ) : |x - 1| + |2 * x + 1| > 3 ↔ (x < -1 ∨ x > 1) := sorry

-- Statement for Absolute Inequality Problem with Bounds
theorem absolute_inequality (a b : ℝ) (ha : -1 ≤ a) (hb : a ≤ 1) (hc : -1 ≤ b) (hd : b ≤ 1) : 
  |1 + (a * b) / 4| > |(a + b) / 2| := sorry

end inequality_solution_absolute_inequality_l230_230116


namespace fisher_needed_score_l230_230492

-- Condition 1: To have an average of at least 85% over all four quarters
def average_score_threshold := 85
def total_score := 4 * average_score_threshold

-- Condition 2: Fisher's scores for the first three quarters
def first_three_scores := [82, 77, 75]
def current_total_score := first_three_scores.sum

-- Define the Lean statement to prove
theorem fisher_needed_score : ∃ x, current_total_score + x = total_score ∧ x = 106 := by
  sorry

end fisher_needed_score_l230_230492


namespace fraction_subtraction_simplify_l230_230255

noncomputable def fraction_subtraction : ℚ :=
  (12 / 25) - (3 / 75)

theorem fraction_subtraction_simplify : fraction_subtraction = (11 / 25) :=
  by
    -- Proof goes here
    sorry

end fraction_subtraction_simplify_l230_230255


namespace derivative_of_sin_squared_is_sin_2x_l230_230951

open Real

noncomputable def f (x : ℝ) : ℝ := sin x ^ 2

theorem derivative_of_sin_squared_is_sin_2x : 
  ∀ x : ℝ, deriv f x = sin (2 * x) :=
by
  sorry

end derivative_of_sin_squared_is_sin_2x_l230_230951


namespace smallest_multiple_of_45_and_75_not_20_l230_230907

-- Definitions of the conditions
def isMultipleOf (a b : ℕ) : Prop := ∃ k : ℕ, a = k * b
def notMultipleOf (a b : ℕ) : Prop := ¬ (isMultipleOf a b)

-- The proof statement
theorem smallest_multiple_of_45_and_75_not_20 :
  ∃ N : ℕ, isMultipleOf N 45 ∧ isMultipleOf N 75 ∧ notMultipleOf N 20 ∧ N = 225 :=
by
  -- sorry is used to indicate that the proof needs to be filled here
  sorry

end smallest_multiple_of_45_and_75_not_20_l230_230907


namespace simplify_sqrt_square_l230_230044

theorem simplify_sqrt_square (h : Real.sqrt 7 < 3) : Real.sqrt ((Real.sqrt 7 - 3)^2) = 3 - Real.sqrt 7 :=
by
  sorry

end simplify_sqrt_square_l230_230044


namespace snowman_volume_l230_230870

noncomputable def volume_snowman (r₁ r₂ r₃ r_c h_c : ℝ) : ℝ :=
  (4 / 3 * Real.pi * r₁^3) + (4 / 3 * Real.pi * r₂^3) + (4 / 3 * Real.pi * r₃^3) + (Real.pi * r_c^2 * h_c)

theorem snowman_volume 
  : volume_snowman 4 6 8 3 5 = 1101 * Real.pi := 
by 
  sorry

end snowman_volume_l230_230870


namespace annika_total_kilometers_east_l230_230921

def annika_constant_rate : ℝ := 10 -- 10 minutes per kilometer
def distance_hiked_initially : ℝ := 2.5 -- 2.5 kilometers
def total_time_to_return : ℝ := 35 -- 35 minutes

theorem annika_total_kilometers_east :
  (total_time_to_return - (distance_hiked_initially * annika_constant_rate)) / annika_constant_rate + distance_hiked_initially = 3.5 := by
  sorry

end annika_total_kilometers_east_l230_230921


namespace trigonometric_bound_l230_230371

open Real

theorem trigonometric_bound (x y : ℝ) : 
  -1/2 ≤ (x + y) * (1 - x * y) / ((1 + x^2) * (1 + y^2)) ∧ 
  (x + y) * (1 - x * y) / ((1 + x^2) * (1 + y^2)) ≤ 1/2 :=
by 
  sorry

end trigonometric_bound_l230_230371


namespace minutes_after_2017_is_0554_l230_230104

theorem minutes_after_2017_is_0554 :
  let initial_time := (20, 17) -- time in hours and minutes
  let total_minutes := 2017
  let hours_passed := total_minutes / 60
  let minutes_passed := total_minutes % 60
  let days_passed := hours_passed / 24
  let remaining_hours := hours_passed % 24
  let resulting_hours := (initial_time.fst + remaining_hours) % 24
  let resulting_minutes := initial_time.snd + minutes_passed
  let final_hours := if resulting_minutes >= 60 then resulting_hours + 1 else resulting_hours
  let final_minutes := if resulting_minutes >= 60 then resulting_minutes - 60 else resulting_minutes
  final_hours % 24 = 5 ∧ final_minutes = 54 := by
  sorry

end minutes_after_2017_is_0554_l230_230104


namespace joe_used_fraction_paint_in_first_week_l230_230445

variable (x : ℝ) -- Define the fraction x as a real number

-- Given conditions
def given_conditions : Prop := 
  let total_paint := 360
  let paint_first_week := x * total_paint
  let remaining_paint := (1 - x) * total_paint
  let paint_second_week := (1 / 2) * remaining_paint
  paint_first_week + paint_second_week = 225

-- The theorem to prove
theorem joe_used_fraction_paint_in_first_week (h : given_conditions x) : x = 1 / 4 :=
sorry

end joe_used_fraction_paint_in_first_week_l230_230445


namespace part1_part2_l230_230454

section
variable (x a : ℝ)
def p (x a : ℝ) : Prop := (x - 3 * a) * (x - a) < 0
def q (x : ℝ) : Prop := (x - 3) / (x - 2) ≤ 0

theorem part1 (h : a = 1) (hq : q x) (hp : p x a) : 2 < x ∧ x < 3 := by
  sorry

theorem part2 (h : ∀ x, q x → p x a) : 1 ≤ a ∧ a ≤ 2 := by
  sorry
end

end part1_part2_l230_230454


namespace sasha_remainder_l230_230068

theorem sasha_remainder (n a b c d : ℕ) (h1 : n = 102 * a + b) (h2 : n = 103 * c + d) 
  (h3 : a + d = 20) (h_b_range : 0 ≤ b ∧ b ≤ 101) (h_d_range : 0 ≤ d ∧ d ≤ 102) : b = 20 := 
sorry

end sasha_remainder_l230_230068


namespace min_value_of_quadratic_l230_230361

theorem min_value_of_quadratic (x : ℝ) : ∃ m : ℝ, (∀ x, x^2 + 10 * x ≥ m) ∧ m = -25 := by
  sorry

end min_value_of_quadratic_l230_230361


namespace intersection_M_N_l230_230865

def M : Set ℝ := { x | x^2 + x - 2 < 0 }
def N : Set ℝ := { x | 0 < x ∧ x ≤ 2 }

theorem intersection_M_N : M ∩ N = { x | 0 < x ∧ x < 1 } :=
by
  sorry

end intersection_M_N_l230_230865


namespace sasha_remainder_l230_230090

theorem sasha_remainder (n a b c d : ℕ) (h1 : n = 102 * a + b) (h2 : n = 103 * c + d)
  (h3 : d = 20 - a) (h4 : 0 ≤ b ∧ b ≤ 101) : b = 20 :=
by
  sorry

end sasha_remainder_l230_230090


namespace minimum_embrasure_length_l230_230520

theorem minimum_embrasure_length : ∀ (s : ℝ), 
  (∀ t : ℝ, (∃ k : ℤ, t = k / 2 ∧ k % 2 = 0) ∨ (∃ k : ℤ, t = (k + 1) / 2 ∧ k % 2 = 1)) → 
  (∃ z : ℝ, z = 2 / 3) := 
sorry

end minimum_embrasure_length_l230_230520


namespace coin_pile_problem_l230_230621

theorem coin_pile_problem (x y z : ℕ) (h1 : 2 * (x - y) = 16) (h2 : 2 * y - z = 16) (h3 : 2 * z - x + y = 16) :
  x = 22 ∧ y = 14 ∧ z = 12 :=
by
  sorry

end coin_pile_problem_l230_230621


namespace LengthRatiosSimultaneous_LengthRatiosNonSimultaneous_l230_230896

noncomputable section

-- Problem 1: Prove length ratios for simultaneous ignition
def LengthRatioSimultaneous (t : ℝ) : Prop :=
  let LA := 1 - t / 3
  let LB := 1 - t / 5
  (LA / LB = 1 / 2 ∨ LA / LB = 1 / 3 ∨ LA / LB = 1 / 4)

theorem LengthRatiosSimultaneous (t : ℝ) : LengthRatioSimultaneous t := sorry

-- Problem 2: Prove length ratios when one candle is lit 30 minutes earlier
def LengthRatioNonSimultaneous (t : ℝ) : Prop :=
  let LA := 5 / 6 - t / 3
  let LB := 1 - t / 5
  (LA / LB = 1 / 2 ∨ LA / LB = 1 / 3 ∨ LA / LB = 1 / 4)

theorem LengthRatiosNonSimultaneous (t : ℝ) : LengthRatioNonSimultaneous t := sorry

end LengthRatiosSimultaneous_LengthRatiosNonSimultaneous_l230_230896


namespace circle_is_axisymmetric_and_centrally_symmetric_l230_230521

structure Shape where
  isAxisymmetric : Prop
  isCentrallySymmetric : Prop

theorem circle_is_axisymmetric_and_centrally_symmetric :
  ∃ (s : Shape), s.isAxisymmetric ∧ s.isCentrallySymmetric :=
by
  sorry

end circle_is_axisymmetric_and_centrally_symmetric_l230_230521


namespace complement_of_P_subset_Q_l230_230230

-- Definitions based on conditions
def P : Set ℝ := {x | x < 1}
def Q : Set ℝ := {x | x > -1}

-- Theorem statement to prove the correct option C
theorem complement_of_P_subset_Q : {x | ¬ (x < 1)} ⊆ {x | x > -1} :=
by {
  sorry
}

end complement_of_P_subset_Q_l230_230230


namespace peanuts_in_box_l230_230775

theorem peanuts_in_box (original_peanuts added_peanuts total_peanuts : ℕ) (h1 : original_peanuts = 10) (h2 : added_peanuts = 8) (h3 : total_peanuts = original_peanuts + added_peanuts) : total_peanuts = 18 := 
by {
  sorry
}

end peanuts_in_box_l230_230775


namespace unique_function_l230_230146

-- Define the function in the Lean environment
def f (n : ℕ) : ℕ := n

-- State the theorem with the given conditions and expected answer
theorem unique_function (f : ℕ → ℕ) : 
  (∀ x y : ℕ, 0 < x → 0 < y → f x + y * f (f x) < x * (1 + f y) + 2021) → (∀ x : ℕ, f x = x) :=
by
  intros h x
  -- Placeholder for the proof
  sorry

end unique_function_l230_230146


namespace sum_of_consecutive_even_integers_l230_230911

theorem sum_of_consecutive_even_integers (n : ℕ) (h1 : (n - 2) + (n + 2) = 162) (h2 : ∃ k : ℕ, n = k^2) :
  (n - 2) + n + (n + 2) = 243 :=
by
  -- no proof required
  sorry

end sum_of_consecutive_even_integers_l230_230911


namespace sufficient_and_necessary_condition_l230_230935

theorem sufficient_and_necessary_condition (a : ℝ) :
  (∀ x : ℝ, a * x^2 - 2 * a * x + 1 > 0) ↔ (0 < a ∧ a < 1) :=
by 
  sorry

end sufficient_and_necessary_condition_l230_230935


namespace parallelogram_area_l230_230646

variable (base height : ℝ) (tripled_area_factor original_area new_area : ℝ)

theorem parallelogram_area (h_base : base = 6) (h_height : height = 20)
    (h_tripled_area_factor : tripled_area_factor = 9)
    (h_original_area_calc : original_area = base * height)
    (h_new_area_calc : new_area = original_area * tripled_area_factor) :
    original_area = 120 ∧ tripled_area_factor = 9 ∧ new_area = 1080 := by
  sorry

end parallelogram_area_l230_230646


namespace quotient_of_division_l230_230733

theorem quotient_of_division (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) 
  (h1 : dividend = 181) (h2 : divisor = 20) (h3 : remainder = 1) 
  (h4 : dividend = (divisor * quotient) + remainder) : quotient = 9 :=
by
  sorry -- proof goes here

end quotient_of_division_l230_230733


namespace walking_distance_l230_230573

theorem walking_distance (D : ℕ) (h : D / 15 = (D + 60) / 30) : D = 60 :=
by
  sorry

end walking_distance_l230_230573


namespace least_integer_solution_l230_230354

theorem least_integer_solution (x : ℤ) (h : x^2 = 2 * x + 98) : x = -7 :=
by {
  sorry
}

end least_integer_solution_l230_230354


namespace sphere_in_cube_volume_unreachable_l230_230382

noncomputable def volume_unreachable_space (cube_side : ℝ) (sphere_radius : ℝ) : ℝ :=
  let corner_volume := 64 - (32/3) * Real.pi
  let edge_volume := 288 - 72 * Real.pi
  corner_volume + edge_volume

theorem sphere_in_cube_volume_unreachable : 
  (volume_unreachable_space 6 1 = 352 - (248 * Real.pi / 3)) :=
by
  sorry

end sphere_in_cube_volume_unreachable_l230_230382


namespace Ivan_increases_share_more_than_six_times_l230_230465

theorem Ivan_increases_share_more_than_six_times
  (p v s i : ℝ)
  (hp : p / (v + s + i) = 3 / 7)
  (hv : v / (p + s + i) = 1 / 3)
  (hs : s / (p + v + i) = 1 / 3) :
  ∃ k : ℝ, k > 6 ∧ i * k > 0.6 * (p + v + s + i * k) :=
by
  sorry

end Ivan_increases_share_more_than_six_times_l230_230465


namespace necessary_but_not_sufficient_l230_230170

theorem necessary_but_not_sufficient (a : ℝ) (ha : a > 1) : a^2 > a :=
sorry

end necessary_but_not_sufficient_l230_230170


namespace normal_distribution_properties_l230_230134

open ProbabilityTheory

noncomputable def standard_normal_CDF (x : ℝ) : ℝ := P(λ (ξ : ℝ), ξ < x)

theorem normal_distribution_properties :
  (∀ x : ℝ, standard_normal_CDF x = P(λ (ξ : ℝ), ξ < x)) → 
  (standard_normal_CDF 0 = 0.5) ∧
  (∀ x : ℝ, standard_normal_CDF x = 1 - standard_normal_CDF (-x)) ∧
  (P(λ (ξ : ℝ), |ξ| < 2) = 2 * standard_normal_CDF 2 - 1) :=
by sorry

end normal_distribution_properties_l230_230134


namespace quadratic_vertex_l230_230199

theorem quadratic_vertex (x y : ℝ) (h : y = -3 * x^2 + 2) : (x, y) = (0, 2) :=
sorry

end quadratic_vertex_l230_230199


namespace min_expression_n_12_l230_230488

theorem min_expression_n_12 : ∃ n : ℕ, (n > 0) ∧ (∀ m : ℕ, m > 0 → (n = 12 → (n / 3 + 50 / n ≤ 
                        m / 3 + 50 / m))) :=
by
  sorry

end min_expression_n_12_l230_230488


namespace find_k_value_l230_230592

theorem find_k_value (a : ℕ → ℕ) (k : ℕ) (S : ℕ → ℕ) 
  (h₁ : a 1 = 1) 
  (h₂ : a 3 = 5) 
  (h₃ : S (k + 2) - S k = 36) : 
  k = 8 := 
by 
  sorry

end find_k_value_l230_230592


namespace find_a_l230_230557

-- Define sets A and B
def A : Set ℕ := {1, 2, 5}
def B (a : ℕ) : Set ℕ := {2, a}

-- Given condition: A ∪ B = {1, 2, 3, 5}
def union_condition (a : ℕ) : Prop := A ∪ B a = {1, 2, 3, 5}

-- Theorem we want to prove
theorem find_a (a : ℕ) : union_condition a → a = 3 :=
by
  intro h
  sorry

end find_a_l230_230557


namespace fraction_division_l230_230768

theorem fraction_division :
  (1/4) / 2 = 1/8 :=
by
  sorry

end fraction_division_l230_230768


namespace eighth_term_of_arithmetic_sequence_l230_230351

noncomputable def arithmetic_sequence (n : ℕ) (a1 an : ℚ) (k : ℕ) : ℚ :=
  a1 + (k - 1) * ((an - a1) / (n - 1))

theorem eighth_term_of_arithmetic_sequence :
  ∀ (a1 a30 : ℚ), a1 = 5 → a30 = 86 → 
  arithmetic_sequence 30 a1 a30 8 = 592 / 29 :=
by
  intros a1 a30 h_a1 h_a30
  rw [h_a1, h_a30]
  dsimp [arithmetic_sequence]
  sorry

end eighth_term_of_arithmetic_sequence_l230_230351


namespace set_intersection_l230_230412

-- Define set A
def A := {x : ℝ | x^2 - 4 * x < 0}

-- Define set B
def B := {x : ℤ | -2 < x ∧ x ≤ 2}

-- Define the intersection of A and B in ℝ
def A_inter_B := {x : ℝ | (x ∈ A) ∧ (∃ (z : ℤ), (x = z) ∧ (z ∈ B))}

-- Proof statement
theorem set_intersection : A_inter_B = {1, 2} :=
by sorry

end set_intersection_l230_230412


namespace compare_rental_fees_l230_230487

namespace HanfuRental

def store_A_rent_price : ℝ := 120
def store_B_rent_price : ℝ := 160
def store_A_discount : ℝ := 0.20
def store_B_discount_limit : ℕ := 6
def store_B_excess_rate : ℝ := 0.50
def x : ℕ := 40 -- number of Hanfu costumes

def y₁ (x : ℕ) : ℝ := (store_A_rent_price * (1 - store_A_discount)) * x

def y₂ (x : ℕ) : ℝ :=
  if x ≤ store_B_discount_limit then store_B_rent_price * x
  else store_B_rent_price * store_B_discount_limit + store_B_excess_rate * store_B_rent_price * (x - store_B_discount_limit)

theorem compare_rental_fees (x : ℕ) (hx : x = 40) :
  y₂ x ≤ y₁ x :=
sorry

end HanfuRental

end compare_rental_fees_l230_230487


namespace restaurant_cost_l230_230926

section Restaurant
variable (total_people kids adults : ℕ) 
variable (meal_cost : ℕ)
variable (total_cost : ℕ)

def calculate_adults (total_people kids : ℕ) : ℕ := 
  total_people - kids

def calculate_total_cost (adults meal_cost : ℕ) : ℕ :=
  adults * meal_cost

theorem restaurant_cost (total_people kids meal_cost : ℕ) :
  total_people = 13 →
  kids = 9 →
  meal_cost = 7 →
  calculate_adults total_people kids = 4 →
  calculate_total_cost 4 meal_cost = 28 :=
by
  intros
  simp [calculate_adults, calculate_total_cost]
  sorry -- Proof would be added here
end Restaurant

end restaurant_cost_l230_230926


namespace sasha_remainder_l230_230083

theorem sasha_remainder (n a b c d : ℕ) 
  (h1 : n = 102 * a + b) 
  (h2 : n = 103 * c + d) 
  (h3 : a + d = 20)
  (hb : 0 ≤ b ∧ b ≤ 101) : 
  b = 20 :=
sorry

end sasha_remainder_l230_230083


namespace find_angle_A_and_triangle_perimeter_l230_230997

-- Declare the main theorem using the provided conditions and the desired results
theorem find_angle_A_and_triangle_perimeter
  (a b c : ℝ) (A B : ℝ)
  (h1 : 0 < A ∧ A < Real.pi)
  (h2 : (Real.sqrt 3) * b * c * (Real.cos A) = a * (Real.sin B))
  (h3 : a = Real.sqrt 2)
  (h4 : (c / a) = (Real.sin A / Real.sin B)) :
  (A = Real.pi / 3) ∧ (a + b + c = 3 * Real.sqrt 2) :=
  sorry -- Proof is left as an exercise

end find_angle_A_and_triangle_perimeter_l230_230997


namespace find_B_share_l230_230384

theorem find_B_share (x : ℕ) (x_pos : 0 < x) (C_share_difference : 5 * x = 4 * x + 1000) (B_share_eq : 3 * x = B) : B = 3000 :=
by
  sorry

end find_B_share_l230_230384


namespace range_of_m_l230_230837

theorem range_of_m (x1 x2 y1 y2 m : ℝ) (h1 : x1 > x2) (h2 : y1 > y2) (h3 : y1 = (m-2)*x1) (h4 : y2 = (m-2)*x2) : m > 2 :=
by sorry

end range_of_m_l230_230837


namespace set_swept_by_all_lines_l230_230890

theorem set_swept_by_all_lines
  (a c x y : ℝ)
  (h1 : 0 < a)
  (h2 : 0 < c)
  (h3 : c < a)
  (h4 : x^2 + y^2 ≤ a^2) : 
  (c^2 - a^2) * x^2 - a^2 * y^2 ≤ (c^2 - a^2) * c^2 :=
sorry

end set_swept_by_all_lines_l230_230890


namespace initial_ratio_milk_water_l230_230379

-- Define the initial conditions
variables (M W : ℕ) (h_volume : M + W = 115) (h_ratio : M / (W + 46) = 3 / 4)

-- State the theorem to prove the initial ratio of milk to water
theorem initial_ratio_milk_water (h_volume : M + W = 115) (h_ratio : M / (W + 46) = 3 / 4) :
  (M * 2 = W * 3) :=
by
  sorry

end initial_ratio_milk_water_l230_230379


namespace total_handshakes_l230_230135

theorem total_handshakes (team1 team2 refs : ℕ) (players_per_team : ℕ) :
  team1 = 11 → team2 = 11 → refs = 3 → players_per_team = 11 →
  (players_per_team * players_per_team + (players_per_team * 2 * refs) = 187) :=
by
  intros h_team1 h_team2 h_refs h_players_per_team
  -- Now we want to prove that
  -- 11 * 11 + (11 * 2 * 3) = 187
  -- However, we can just add sorry here as the purpose is to write the statement
  sorry

end total_handshakes_l230_230135


namespace find_x_minus_y_l230_230327

open Real

theorem find_x_minus_y (x y : ℝ) (h : (sin x ^ 2 - cos x ^ 2 + cos x ^ 2 * cos y ^ 2 - sin x ^ 2 * sin y ^ 2) / sin (x + y) = 1) :
  ∃ k : ℤ, x - y = π / 2 + 2 * k * π :=
by
  sorry

end find_x_minus_y_l230_230327


namespace remainder_of_M_mod_1000_l230_230591

def M : ℕ := Nat.choose 9 8

theorem remainder_of_M_mod_1000 : M % 1000 = 9 := by
  sorry

end remainder_of_M_mod_1000_l230_230591


namespace log_sum_range_l230_230679

theorem log_sum_range {x y : ℝ} (hx : 0 < x) (hy : 0 < y)
  (h : Real.log (x + y) / Real.log 2 = Real.log x / Real.log 2 + Real.log y / Real.log 2) :
  4 ≤ x + y :=
by
  sorry

end log_sum_range_l230_230679


namespace gcd_256_180_600_l230_230631

theorem gcd_256_180_600 : Nat.gcd (Nat.gcd 256 180) 600 = 4 :=
by
  -- This proof is marked as 'sorry' to indicate it is a place holder
  sorry

end gcd_256_180_600_l230_230631


namespace original_number_of_men_l230_230500

theorem original_number_of_men 
  (x : ℕ) 
  (H1 : x * 15 = (x - 8) * 18) : 
  x = 48 := 
sorry

end original_number_of_men_l230_230500


namespace scheduled_conference_games_total_l230_230750

def number_of_teams_in_A := 7
def number_of_teams_in_B := 5
def games_within_division (n : Nat) : Nat := n * (n - 1)
def interdivision_games := 7 * 5
def rivalry_games := 7

theorem scheduled_conference_games_total : 
  let games_A := games_within_division number_of_teams_in_A
  let games_B := games_within_division number_of_teams_in_B
  let total_games := games_A + games_B + interdivision_games + rivalry_games
  total_games = 104 :=
by
  sorry

end scheduled_conference_games_total_l230_230750


namespace regular_polygon_sides_l230_230126

theorem regular_polygon_sides (P s : ℕ) (hP : P = 108) (hs : s = 12) : 
  ∃ n : ℕ, P = n * s ∧ n = 9 :=
by {
  use 9,
  split,
  { rw [hP, hs], norm_num },
  refl
}

end regular_polygon_sides_l230_230126


namespace solve_problem_l230_230296

-- Define the given conditions
def condition1 (a : ℝ) : Prop := real.cbrt (5 * a + 2) = 3
def condition2 (a b : ℝ) : Prop := real.sqrt (3 * a + b - 1) = 4
def condition3 (c : ℝ) : Prop := c = real.floor (real.sqrt 13)

-- Define the target values and resulting statement
def values_a_b_c (a b c : ℝ) : Prop := 
  a = 5 ∧ b = 2 ∧ c = 3

def final_sqrt (a b c : ℝ) : Prop :=
  real.sqrt (3 * a - b + c) = 4 ∨ real.sqrt (3 * a - b + c) = -4

-- Complete the statement bringing everything together
theorem solve_problem (a b c : ℝ) :
  condition1 a →
  condition2 a b →
  condition3 c →
  values_a_b_c a b c ∧ final_sqrt a b c :=
begin
  sorry
end

end solve_problem_l230_230296


namespace how_many_raisins_did_bryce_receive_l230_230563

def raisins_problem : Prop :=
  ∃ (B C : ℕ), B = C + 8 ∧ C = B / 3 ∧ B + C = 44 ∧ B = 33

theorem how_many_raisins_did_bryce_receive : raisins_problem :=
sorry

end how_many_raisins_did_bryce_receive_l230_230563


namespace difference_of_squares_evaluation_l230_230537

theorem difference_of_squares_evaluation :
  49^2 - 16^2 = 2145 :=
by sorry

end difference_of_squares_evaluation_l230_230537


namespace a_minus_b_ge_one_l230_230864

def a : ℕ := 19^91
def b : ℕ := (999991)^19

theorem a_minus_b_ge_one : a - b ≥ 1 :=
by
  sorry

end a_minus_b_ge_one_l230_230864


namespace find_a_l230_230987

theorem find_a (a b c : ℤ) (h1 : a + b = c) (h2 : b + c = 8) (h3 : c = 4) : a = 0 :=
by
  sorry

end find_a_l230_230987


namespace coefficient_of_determination_indicates_better_fit_l230_230438

theorem coefficient_of_determination_indicates_better_fit (R_squared : ℝ) (h1 : 0 ≤ R_squared) (h2 : R_squared ≤ 1) :
  R_squared = 1 → better_fitting_effect_of_regression_model :=
by
  sorry

end coefficient_of_determination_indicates_better_fit_l230_230438


namespace find_smallest_x_l230_230276

def smallest_x_satisfying_condition : Prop :=
  ∃ x : ℝ, x > 0 ∧ (⌊x^2⌋ - x * ⌊x⌋ = 8) ∧ x = 89 / 9

theorem find_smallest_x : smallest_x_satisfying_condition :=
begin
  -- proof goes here (not required for this task)
  sorry
end

end find_smallest_x_l230_230276


namespace sales_tax_difference_l230_230952

theorem sales_tax_difference (price : ℝ) (tax_rate1 tax_rate2 : ℝ) :
  price = 50 → tax_rate1 = 0.075 → tax_rate2 = 0.065 →
  (price * tax_rate1 - price * tax_rate2 = 0.5) :=
by
  intros
  sorry

end sales_tax_difference_l230_230952


namespace total_logs_combined_l230_230247

theorem total_logs_combined 
  (a1 l1 a2 l2 : ℕ) 
  (n1 n2 : ℕ) 
  (S1 S2 : ℕ) 
  (h1 : a1 = 15) 
  (h2 : l1 = 10) 
  (h3 : n1 = 6) 
  (h4 : S1 = n1 * (a1 + l1) / 2) 
  (h5 : a2 = 9) 
  (h6 : l2 = 5) 
  (h7 : n2 = 5) 
  (h8 : S2 = n2 * (a2 + l2) / 2) : 
  S1 + S2 = 110 :=
by {
  sorry
}

end total_logs_combined_l230_230247


namespace angle_between_diagonals_l230_230483

variables (α β : ℝ) 

theorem angle_between_diagonals (α β : ℝ) :
  ∃ γ : ℝ, γ = Real.arccos (Real.sin α * Real.sin β) :=
by
  sorry

end angle_between_diagonals_l230_230483


namespace binomial_eight_three_l230_230806

noncomputable def factorial : ℕ → ℕ 
| 0       := 1
| (n + 1) := (n + 1) * factorial n

def binomial (n k : ℕ) : ℕ :=
factorial n / (factorial k * factorial (n - k))

theorem binomial_eight_three : binomial 8 3 = 56 := by
  sorry

end binomial_eight_three_l230_230806


namespace sum_mod_9_l230_230404

theorem sum_mod_9 (h1 : 34125 % 9 = 1) (h2 : 34126 % 9 = 2) (h3 : 34127 % 9 = 3)
                  (h4 : 34128 % 9 = 4) (h5 : 34129 % 9 = 5) (h6 : 34130 % 9 = 6)
                  (h7 : 34131 % 9 = 7) :
  (34125 + 34126 + 34127 + 34128 + 34129 + 34130 + 34131) % 9 = 1 :=
by
  sorry

end sum_mod_9_l230_230404


namespace abc_plus_2_gt_a_plus_b_plus_c_l230_230151

theorem abc_plus_2_gt_a_plus_b_plus_c (a b c : ℝ) (ha : -1 < a) (ha' : a < 1) (hb : -1 < b) (hb' : b < 1) (hc : -1 < c) (hc' : c < 1) :
  a * b * c + 2 > a + b + c :=
sorry

end abc_plus_2_gt_a_plus_b_plus_c_l230_230151


namespace area_of_region_l230_230137

theorem area_of_region :
  (∫ x, ∫ y in {y : ℝ | x^4 + y^4 = |x|^3 + |y|^3}, (1 : ℝ)) = 4 :=
sorry

end area_of_region_l230_230137


namespace sum_of_all_N_l230_230393

-- Define the machine's processing rules
def process (N : ℕ) : ℕ :=
  if N % 2 = 1 then 4 * N + 2 else N / 2

-- Define the 6-step process starting from N
def six_steps (N : ℕ) : ℕ :=
  process (process (process (process (process (process N)))))

-- Definition for the main theorem
theorem sum_of_all_N (N : ℕ) : six_steps N = 10 → N = 640 :=
by 
  sorry

end sum_of_all_N_l230_230393


namespace mutually_exclusive_but_not_complementary_l230_230407

open ProbabilityTheory

-- Define the event of drawing different colored balls
def Bag := {red := 3, black := 3}

def draw_two_balls (bag : Bag) : Set (Set String) :=
  {s | s.card = 2 ∧ ∀ b ∈ s, b = "red" ∨ b = "black"}

-- Event: Exactly one black ball in the draw
def exactly_one_black (event : Set String) : Prop :=
  event.count "black" = 1

-- Event: Exactly two red balls in the draw
def exactly_two_red (event : Set String) : Prop :=
  event.count "red" = 2

theorem mutually_exclusive_but_not_complementary :
  (∀ e ∈ (draw_two_balls Bag), exactly_one_black e → ¬ exactly_two_red e) ∧
  (∃ e ∈ (draw_two_balls Bag), ¬ exactly_one_black e ∧ ¬ exactly_two_red e) :=
by sorry

end mutually_exclusive_but_not_complementary_l230_230407


namespace solve_for_exponent_l230_230913

theorem solve_for_exponent (K : ℕ) (h1 : 32 = 2 ^ 5) (h2 : 64 = 2 ^ 6) 
    (h3 : 32 ^ 5 * 64 ^ 2 = 2 ^ K) : K = 37 := 
by 
    sorry

end solve_for_exponent_l230_230913


namespace total_meals_sold_l230_230252

-- Definitions based on the conditions
def ratio_kids_adult := 2 / 1
def kids_meals := 8

-- The proof problem statement
theorem total_meals_sold : (∃ adults_meals : ℕ, 2 * adults_meals = kids_meals) → (kids_meals + 4 = 12) := 
by 
  sorry

end total_meals_sold_l230_230252


namespace max_n_intersection_non_empty_l230_230839

-- Define the set An
def An (n : ℕ) : Set ℝ := {x : ℝ | n < x^n ∧ x^n < n + 1}

-- State the theorem
theorem max_n_intersection_non_empty : 
  ∃ x, (∀ n, n ≤ 4 → x ∈ An n) ∧ (∀ n, n > 4 → x ∉ An n) :=
by
  sorry

end max_n_intersection_non_empty_l230_230839


namespace total_sheets_of_paper_l230_230725

theorem total_sheets_of_paper (num_classes : ℕ) (students_per_class : ℕ) (sheets_per_student : ℕ)
  (h1 : num_classes = 4) (h2 : students_per_class = 20) (h3 : sheets_per_student = 5) :
  (num_classes * students_per_class * sheets_per_student) = 400 :=
by
  sorry

end total_sheets_of_paper_l230_230725


namespace sasha_remainder_l230_230082

theorem sasha_remainder (n a b c d : ℕ) 
  (h1 : n = 102 * a + b) 
  (h2 : n = 103 * c + d) 
  (h3 : a + d = 20)
  (hb : 0 ≤ b ∧ b ≤ 101) : 
  b = 20 :=
sorry

end sasha_remainder_l230_230082


namespace gcd_of_256_180_600_l230_230634

theorem gcd_of_256_180_600 : Nat.gcd (Nat.gcd 256 180) 600 = 12 :=
by
  -- The proof would be placed here
  sorry

end gcd_of_256_180_600_l230_230634


namespace symmetrical_point_l230_230049

structure Point :=
  (x : ℝ)
  (y : ℝ)

def reflect_x_axis (p : Point) : Point :=
  { x := p.x, y := -p.y }

theorem symmetrical_point (M : Point) (hM : M = {x := 3, y := -4}) : reflect_x_axis M = {x := 3, y := 4} :=
  by
  sorry

end symmetrical_point_l230_230049


namespace yoki_cans_correct_l230_230534

def total_cans := 85
def ladonna_cans := 25
def prikya_cans := 2 * ladonna_cans
def yoki_cans := total_cans - ladonna_cans - prikya_cans

theorem yoki_cans_correct : yoki_cans = 10 :=
by
  sorry

end yoki_cans_correct_l230_230534


namespace simplify_fraction_l230_230609

theorem simplify_fraction (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) : 
  (10 * x * y^2) / (5 * x * y) = 2 * y := 
by
  sorry

end simplify_fraction_l230_230609


namespace hexagonal_tessellation_color_count_l230_230663

noncomputable def hexagonal_tessellation : SimpleGraph ℕ :=
{ adj := λ n m, (m = n + 1 ∨ m = n - 1 ∨ 
                 m = n + 2 ∨ m = n - 2 ∨
                 m = n + 3 ∨ m = n - 3),
  sym := by {
    intros a b h,
    cases h;
    simp [h],
  },
  loopless := by {
    intro a,
    simp,
  }
}

theorem hexagonal_tessellation_color_count :
  ∃ k, k = 4 ∧ ∀ f : ℕ → Fin k, (∀ {x y}, hexagonal_tessellation.adj x y → f x ≠ f y) := 
begin
  use 4,
  split,
  { refl },
  { intros f h,
    sorry,
  }
end

end hexagonal_tessellation_color_count_l230_230663


namespace minimum_marbles_l230_230250

theorem minimum_marbles
  (r w b g y n : ℕ)
  (h_y : y = 4)
  (h_n : n = r + w + b + g + y)
  (h_1 : r * (r - 1) * (r - 2) * (r - 3) * (r - 4) / 120 = w * r * (r - 1) * (r - 2) * (r - 3) / 24)
  (h_2 : r * (r - 1) * (r - 2) * (r - 3) * (r - 4) / 120 = w * b * r * (r - 1) * (r - 2) / 6)
  (h_3 : w * b * g * r * (r - 1) / 2 = w * b * g * r):
  n = 27 :=
by
  sorry

end minimum_marbles_l230_230250


namespace division_of_decimals_l230_230103

theorem division_of_decimals : (0.5 : ℝ) / (0.025 : ℝ) = 20 := 
sorry

end division_of_decimals_l230_230103


namespace num_solutions_abs_x_plus_abs_y_lt_100_l230_230843

theorem num_solutions_abs_x_plus_abs_y_lt_100 :
  (∃ n : ℕ, n = 338350 ∧ ∀ (x y : ℤ), (|x| + |y| < 100) → True) :=
sorry

end num_solutions_abs_x_plus_abs_y_lt_100_l230_230843


namespace arithmetic_sequence_term_l230_230857

variable (a : ℕ → ℕ)
variable (d : ℕ)

-- Conditions
def common_difference := d = 2
def value_a_2007 := a 2007 = 2007

-- Question to be proved
theorem arithmetic_sequence_term :
  common_difference d →
  value_a_2007 a →
  a 2009 = 2011 :=
by
  sorry

end arithmetic_sequence_term_l230_230857


namespace problem_statement_l230_230158

noncomputable def f (x : ℝ) := 3 * Real.sin (2 * x + Real.pi / 3)

def h (x m : ℝ) := 2 * f x + 1 - m

theorem problem_statement (m : ℝ) :
  (∀ x, f x ≤ 3) →
  (∀ x, f x ≥ -3) →
  (∀ x, x = Real.pi / 12 → f x = 3) →
  (∀ x, x = 7 * Real.pi / 12 → f x = -3) →
  (∀ x, x ∈ Icc (-Real.pi / 3) (Real.pi / 6) → h x m = 0) →
  m ∈ Icc (3 * Real.sqrt 3 + 1) 7 :=
sorry

end problem_statement_l230_230158


namespace proof_of_problem_l230_230502

noncomputable def problem : Prop :=
  (1 + Real.cos (20 * Real.pi / 180)) / (2 * Real.sin (20 * Real.pi / 180)) -
  (Real.sin (10 * Real.pi / 180) * 
  (1 / Real.tan (5 * Real.pi / 180) - Real.tan (5 * Real.pi / 180))) =
  (Real.sqrt 3) / 2

theorem proof_of_problem : problem :=
by
  sorry

end proof_of_problem_l230_230502


namespace inscribed_square_product_l230_230934

theorem inscribed_square_product (a b : ℝ)
  (h1 : a + b = 2 * Real.sqrt 5)
  (h2 : Real.sqrt (a^2 + b^2) = 4 * Real.sqrt 2) :
  a * b = -6 := 
by
  sorry

end inscribed_square_product_l230_230934


namespace geometric_sequence_second_term_l230_230344

theorem geometric_sequence_second_term (a r : ℝ) (h1 : a * r ^ 2 = 5) (h2 : a * r ^ 4 = 45) :
  a * r = 5 / 3 :=
by
  sorry

end geometric_sequence_second_term_l230_230344


namespace smaller_acute_angle_l230_230581

theorem smaller_acute_angle (x : ℝ) (h : 5 * x + 4 * x = 90) : 4 * x = 40 :=
by 
  -- proof steps can be added here, but are omitted as per the instructions
  sorry

end smaller_acute_angle_l230_230581


namespace first_shipment_weight_l230_230503

variable (first_shipment : ℕ)
variable (total_dishes_made : ℕ := 13)
variable (couscous_per_dish : ℕ := 5)
variable (second_shipment : ℕ := 45)
variable (same_day_shipment : ℕ := 13)

theorem first_shipment_weight :
  13 * 5 = 65 → second_shipment ≠ first_shipment → 
  first_shipment + same_day_shipment = 65 →
  first_shipment = 65 :=
by
  sorry

end first_shipment_weight_l230_230503


namespace solve_inequality_l230_230817

theorem solve_inequality (y : ℚ) :
  (3 / 40 : ℚ) + |y - (17 / 80 : ℚ)| < (1 / 8 : ℚ) ↔ (13 / 80 : ℚ) < y ∧ y < (21 / 80 : ℚ) := 
by
  sorry

end solve_inequality_l230_230817


namespace exists_zero_in_interval_l230_230615

theorem exists_zero_in_interval :
  ∃ x ∈ (2 : ℝ, 3), (λ x : ℝ, Real.log x - 1) x = 0 :=
by
  -- Define the function f(x)
  let f := λ x : ℝ, Real.log x - 1
  -- Monotonicity condition (not necessary to state explicitly, but can be mentioned)
  have monotonic : ∀ x y : ℝ, x < y → f x < f y := 
    λ x y hx, Real.log_lt_log hx
  -- Conditions at the interval ends
  have f2 : f 2 < 0 := by norm_num; exact Real.log_lt_one_of_lt (by norm_num)
  have f3 : f 3 > 0 := by norm_num; exact Real.one_lt_log_of_lt (by norm_num)
  -- Conclude that there is a zero in the interval (2, 3)
  have exists_zero := IntermediateValueTheorem exists_zero_in_interval_monotonic f 2 3 f2 f3 monotonic sorry
  exact exists_zero

end exists_zero_in_interval_l230_230615


namespace constant_for_odd_m_l230_230967

theorem constant_for_odd_m (constant : ℝ) (f : ℕ → ℝ)
  (h1 : ∀ m : ℕ, m > 0 → (∃ k : ℕ, m = 2 * k + 1) → f m = constant * m)
  (h2 : ∀ m : ℕ, m > 0 → (∃ k : ℕ, m = 2 * k) → f m = (1/2 : ℝ) * m)
  (h3 : f 5 * f 6 = 15) : constant = 1 :=
by
  sorry

end constant_for_odd_m_l230_230967


namespace regular_train_passes_by_in_4_seconds_l230_230932

theorem regular_train_passes_by_in_4_seconds
    (l_high_speed : ℕ)
    (l_regular : ℕ)
    (t_observed : ℕ)
    (v_relative : ℕ)
    (h_length_high_speed : l_high_speed = 80)
    (h_length_regular : l_regular = 100)
    (h_time_observed : t_observed = 5)
    (h_velocity : v_relative = l_regular / t_observed) :
    v_relative * 4 = l_high_speed :=
by
  sorry

end regular_train_passes_by_in_4_seconds_l230_230932


namespace prob_dist_of_ξ_expected_value_of_ξ_prob_ξ_leq_1_l230_230545

open ProbabilityMassFunction

-- define the context
def students : Finset ℕ := {0, 1, 2, 3, 4, 5} -- 0-3 are males, 4-5 are females
def males : Finset ℕ := {0, 1, 2, 3}
def females : Finset ℕ := {4, 5}
def choose3 : Finset (Finset ℕ) := students.powerset.filter (λ s, s.card = 3)

-- define the random variable ξ (xi)
def ξ (s : Finset ℕ) : ℕ := s.card ∩ females.card

-- Total ways to choose 3 out of 6 students
def total_ways := choose3.card

-- Define pmf for ξ
def pmf_ξ (k : ℕ) : ℚ := 
  (choose3.filter (λ s, ξ s = k)).card / total_ways

-- Proof problem statements
theorem prob_dist_of_ξ : 
  pmf_ξ 0 = 1/5 ∧ pmf_ξ 1 = 3/5 ∧ pmf_ξ 2 = 1/5 :=
by sorry

theorem expected_value_of_ξ : 
  (0 * pmf_ξ 0 + 1 * pmf_ξ 1 + 2 * pmf_ξ 2) = 1 :=
by sorry

theorem prob_ξ_leq_1 : 
  (pmf_ξ 0 + pmf_ξ 1) = 4/5 :=
by sorry

end prob_dist_of_ξ_expected_value_of_ξ_prob_ξ_leq_1_l230_230545


namespace min_shift_sine_l230_230879

theorem min_shift_sine (φ : ℝ) (hφ : φ > 0) :
    (∃ k : ℤ, 2 * φ + π / 3 = 2 * k * π) → φ = 5 * π / 6 :=
sorry

end min_shift_sine_l230_230879


namespace tim_initial_soda_l230_230762

-- Define the problem
def initial_cans (x : ℕ) : Prop :=
  let after_jeff_takes := x - 6
  let after_buying_more := after_jeff_takes + after_jeff_takes / 2
  after_buying_more = 24

-- Theorem stating the problem in Lean 4
theorem tim_initial_soda (x : ℕ) (h: initial_cans x) : x = 22 :=
by
  sorry

end tim_initial_soda_l230_230762


namespace diagonal_angle_with_plane_l230_230055

theorem diagonal_angle_with_plane (α : ℝ) {a : ℝ} 
  (h_square: a > 0)
  (θ : ℝ := Real.arcsin ((Real.sin α) / Real.sqrt 2)): 
  ∃ (β : ℝ), β = θ :=
sorry

end diagonal_angle_with_plane_l230_230055


namespace parallel_lines_condition_l230_230846

theorem parallel_lines_condition (m : ℝ) :
  (∀ x y : ℝ, 2 * m * x + y + 6 = 0 → (m - 3) * x - y + 7 = 0) → m = 1 :=
by
  sorry

end parallel_lines_condition_l230_230846


namespace total_handshakes_l230_230799

-- Define the conditions
def number_of_players_per_team : Nat := 11
def number_of_referees : Nat := 3
def total_number_of_players : Nat := number_of_players_per_team * 2

-- Prove the total number of handshakes
theorem total_handshakes : 
  (number_of_players_per_team * number_of_players_per_team) + (total_number_of_players * number_of_referees) = 187 := 
by {
  sorry
}

end total_handshakes_l230_230799


namespace largest_n_satisfying_inequality_l230_230905

theorem largest_n_satisfying_inequality :
  ∃ n : ℕ, (1/4 + n/6 < 3/2) ∧ (∀ m : ℕ, m > n → 1/4 + m/6 ≥ 3/2) :=
by
  sorry

end largest_n_satisfying_inequality_l230_230905


namespace triangle_inequality_sqrt_equality_condition_l230_230007

theorem triangle_inequality_sqrt 
  {a b c : ℝ} 
  (h1 : a + b > c) 
  (h2 : b + c > a) 
  (h3 : c + a > b) :
  (Real.sqrt (a + b - c) + Real.sqrt (c + a - b) + Real.sqrt (b + c - a) 
  ≤ Real.sqrt a + Real.sqrt b + Real.sqrt c) := 
sorry

theorem equality_condition 
  {a b c : ℝ} 
  (h1 : a + b > c) 
  (h2 : b + c > a) 
  (h3 : c + a > b) :
  (Real.sqrt (a + b - c) + Real.sqrt (c + a - b) + Real.sqrt (b + c - a) 
  = Real.sqrt a + Real.sqrt b + Real.sqrt c) → 
  (a = b ∧ b = c) := 
sorry

end triangle_inequality_sqrt_equality_condition_l230_230007


namespace oil_used_l230_230745

theorem oil_used (total_weight : ℕ) (ratio_oil_peanuts : ℕ) (ratio_total_parts : ℕ) 
  (ratio_peanuts : ℕ) (ratio_parts : ℕ) (peanuts_weight : ℕ) : 
  ratio_oil_peanuts = 2 → 
  ratio_peanuts = 8 → 
  ratio_total_parts = 10 → 
  ratio_parts = 20 →
  peanuts_weight = total_weight / ratio_total_parts →
  total_weight = 20 → 
  2 * peanuts_weight = 4 :=
by sorry

end oil_used_l230_230745


namespace part1_part2_l230_230600

noncomputable def A : Set ℝ := {x | x^2 + 4 * x = 0 }

noncomputable def B (a : ℝ) : Set ℝ := {x | x^2 + 2 * (a + 1) * x + a^2 - 1 = 0 }

-- Part (1): Prove a = 1 given A ∪ B = B
theorem part1 (a : ℝ) (h : A ∪ B a = B a) : a = 1 :=
sorry

-- Part (2): Prove the set C composed of the values of a given A ∩ B = B
def C : Set ℝ := {a | a ≤ -1 ∨ a = 1}

theorem part2 (h : ∀ a, A ∩ B a = B a ↔ a ∈ C) : forall a, A ∩ B a = B a ↔ a ∈ C :=
sorry

end part1_part2_l230_230600


namespace potato_slice_length_l230_230940

theorem potato_slice_length (x : ℕ) (h1 : 600 = x + (x + 50)) : x + 50 = 325 :=
by
  sorry

end potato_slice_length_l230_230940


namespace tetrahedron_paintings_l230_230900

theorem tetrahedron_paintings (n : ℕ) (h : n ≥ 4) : 
  let term1 := (n - 1) * (n - 2) * (n - 3) / 12
  let term2 := (n - 1) * (n - 2) / 3
  let term3 := n - 1
  let term4 := 1
  2 * (term1 + term2 + term3) + n = 
  n * (term1 + term2 + term3 + term4) := by
{
  sorry
}

end tetrahedron_paintings_l230_230900


namespace convert_deg_to_rad_l230_230954

theorem convert_deg_to_rad (deg : ℝ) (h : deg = -630) : deg * (Real.pi / 180) = -7 * Real.pi / 2 :=
by
  rw [h]
  simp
  sorry

end convert_deg_to_rad_l230_230954


namespace min_value_of_square_sum_l230_230409

theorem min_value_of_square_sum (x y : ℝ) (h : (x-1)^2 + y^2 = 16) : ∃ (a : ℝ), a = x^2 + y^2 ∧ a = 9 :=
by 
  sorry

end min_value_of_square_sum_l230_230409


namespace total_capsules_in_july_l230_230585

theorem total_capsules_in_july : 
  let mondays := 4
  let tuesdays := 5
  let wednesdays := 5
  let thursdays := 4
  let fridays := 4
  let saturdays := 4
  let sundays := 5

  let capsules_monday := mondays * 2
  let capsules_tuesday := tuesdays * 3
  let capsules_wednesday := wednesdays * 2
  let capsules_thursday := thursdays * 3
  let capsules_friday := fridays * 2
  let capsules_saturday := saturdays * 4
  let capsules_sunday := sundays * 4

  let total_capsules := capsules_monday + capsules_tuesday + capsules_wednesday + capsules_thursday + capsules_friday + capsules_saturday + capsules_sunday

  let missed_capsules_tuesday := 3
  let missed_capsules_sunday := 4

  let total_missed_capsules := missed_capsules_tuesday + missed_capsules_sunday

  let total_consumed_capsules := total_capsules - total_missed_capsules
  total_consumed_capsules = 82 := 
by
  -- Details omitted, proof goes here
  sorry

end total_capsules_in_july_l230_230585


namespace min_value_seq_l230_230165

theorem min_value_seq (a : ℕ → ℕ) (n : ℕ) (h₁ : a 1 = 26) (h₂ : ∀ n, a (n + 1) - a n = 2 * n + 1) :
  ∃ m, (m > 0) ∧ (∀ k, k > 0 → (a k / k : ℚ) ≥ 10) ∧ (a m / m : ℚ) = 10 :=
by
  sorry

end min_value_seq_l230_230165


namespace find_numbers_with_conditions_l230_230398

theorem find_numbers_with_conditions (n : ℕ) (hn1 : n % 100 = 0) (hn2 : (n.divisors).card = 12) : 
  n = 200 ∨ n = 500 :=
by
  sorry

end find_numbers_with_conditions_l230_230398


namespace petya_coloring_l230_230225

theorem petya_coloring (k : ℕ) : k = 1 :=
  sorry

end petya_coloring_l230_230225


namespace intersection_M_N_l230_230845

def M := {x : ℝ | -2 ≤ x ∧ x ≤ 2}
def N := {x : ℝ | x^2 - 3*x ≤ 0}

theorem intersection_M_N : M ∩ N = {x : ℝ | 0 ≤ x ∧ x ≤ 2} :=
by
  sorry

end intersection_M_N_l230_230845


namespace geometric_sequence_first_term_l230_230486

theorem geometric_sequence_first_term (a : ℕ) (r : ℕ)
    (h1 : a * r^2 = 27) 
    (h2 : a * r^3 = 81) : 
    a = 3 :=
by
  sorry

end geometric_sequence_first_term_l230_230486


namespace Tom_needs_11_25_hours_per_week_l230_230622

theorem Tom_needs_11_25_hours_per_week
  (summer_weeks: ℕ) (summer_weeks_val: summer_weeks = 8)
  (summer_hours_per_week: ℕ) (summer_hours_per_week_val: summer_hours_per_week = 45)
  (summer_earnings: ℝ) (summer_earnings_val: summer_earnings = 3600)
  (rest_weeks: ℕ) (rest_weeks_val: rest_weeks = 40)
  (rest_earnings_goal: ℝ) (rest_earnings_goal_val: rest_earnings_goal = 4500) :
  (rest_earnings_goal / (summer_earnings / (summer_hours_per_week * summer_weeks))) / rest_weeks = 11.25 :=
by
  simp [summer_earnings_val, rest_earnings_goal_val, summer_hours_per_week_val, summer_weeks_val]
  sorry

end Tom_needs_11_25_hours_per_week_l230_230622


namespace payment_to_C_l230_230374

/-- 
If A can complete a work in 6 days, B can complete the same work in 8 days, 
they signed to do the work for Rs. 2400 and completed the work in 3 days with 
the help of C, then the payment to C should be Rs. 300.
-/
theorem payment_to_C (total_payment : ℝ) (days_A : ℝ) (days_B : ℝ) (days_worked : ℝ) (portion_C : ℝ) :
   total_payment = 2400 ∧ days_A = 6 ∧ days_B = 8 ∧ days_worked = 3 ∧ portion_C = 1 / 8 →
   (portion_C * total_payment) = 300 := 
by 
  intros h
  cases h
  sorry

end payment_to_C_l230_230374


namespace roots_equation_l230_230294

theorem roots_equation (α β : ℝ) (h1 : α^2 - 4 * α - 1 = 0) (h2 : β^2 - 4 * β - 1 = 0) :
  3 * α^3 + 4 * β^2 = 80 + 35 * α :=
by
  sorry

end roots_equation_l230_230294


namespace projection_correct_l230_230841

theorem projection_correct :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (-1, 3)
  -- Definition of dot product for 2D vectors
  let dot (x y : ℝ × ℝ) : ℝ := x.1 * y.1 + x.2 * y.2
  -- Definition of projection of a onto b
  let proj := (dot a b / (b.1^2 + b.2^2)) • b
  proj = (-1 / 2, 3 / 2) :=
by
  sorry

end projection_correct_l230_230841


namespace time_via_route_B_l230_230589

-- Given conditions
def time_via_route_A : ℕ := 5
def time_saved_round_trip : ℕ := 6

-- Defining the proof problem
theorem time_via_route_B : time_via_route_A - (time_saved_round_trip / 2) = 2 :=
by
  -- Expected proof here
  sorry

end time_via_route_B_l230_230589


namespace largest_power_of_two_divides_a2013_l230_230475

noncomputable def a_2013 (n : ℕ) : ℤ := -1007 * 2013 * Nat.factorial 2013

noncomputable def largest_power_of_two_dividing (n : ℕ) : ℕ :=
  Nat.factorial.find_greatest_pow 2 n

theorem largest_power_of_two_divides_a2013 :
  largest_power_of_two_dividing 2013 = 2004 :=
by
  sorry

end largest_power_of_two_divides_a2013_l230_230475


namespace problem_1_problem_2_l230_230286

open Real

def vec_parallel (u v : ℝ × ℝ) : Prop :=
  u.1 * v.2 = u.2 * v.1

def vec_perpendicular (u v : ℝ × ℝ) : Prop :=
  u.1 * v.1 + u.2 * v.2 = 0

theorem problem_1 (k : ℝ) : 
  let a := (3, 4)
  let b := (2, k)
  (vec_parallel (a.1 + 2 * b.1, a.2 + 2 * b.2) (a.1 - b.1, a.2 - b.2)) →
  k = 8 / 3 := sorry

theorem problem_2 (k : ℝ) : 
  let a := (3, 4)
  let b := (2, k)
  (vec_perpendicular (a.1 + b.1, a.2 + b.2) (a.1 - b.1, a.2 - b.2)) →
  k = sqrt 21 ∨ k = - sqrt 21 := sorry

end problem_1_problem_2_l230_230286


namespace find_common_difference_l230_230415

-- Definitions based on conditions in a)
def common_difference_4_10 (a₁ d : ℝ) : Prop :=
  (a₁ + 3 * d) + (a₁ + 9 * d) = 0

def sum_relation (a₁ d : ℝ) : Prop :=
  2 * (12 * a₁ + 66 * d) = (2 * a₁ + d + 10)

-- Math proof problem statement
theorem find_common_difference (a₁ d : ℝ) 
  (h₁ : common_difference_4_10 a₁ d) 
  (h₂ : sum_relation a₁ d) : 
  d = -10 :=
sorry

end find_common_difference_l230_230415


namespace sasha_remainder_l230_230062

statement:
  theorem sasha_remainder (n a b c d : ℕ) (h1 : n = 102 * a + b) (h2 : n = 103 * c + d) (h3 : a + d = 20) (h4: 0 ≤ b ∧ b ≤ 101) : 
  b = 20 :=
by
  sorry

end sasha_remainder_l230_230062


namespace no_same_distribution_of_silver_as_gold_l230_230047

theorem no_same_distribution_of_silver_as_gold (n m : ℕ) 
  (hn : n ≡ 5 [MOD 10]) 
  (hm : m = 2 * n) 
  : ∀ (f : Fin 10 → ℕ), (∀ i j : Fin 10, i ≠ j → ¬ (f i - f j ≡ 0 [MOD 10])) 
  → ∀ (g : Fin 10 → ℕ), ¬ (∀ i j : Fin 10, i ≠ j → ¬ (g i - g j ≡ 0 [MOD 10])) :=
sorry

end no_same_distribution_of_silver_as_gold_l230_230047


namespace arithmetic_sequence_sum_l230_230059

theorem arithmetic_sequence_sum : 
  ∃ x y, (∃ d, 
  d = 12 - 5 ∧ 
  19 + d = x ∧ 
  x + d = y ∧ 
  y + d = 40 ∧ 
  x + y = 59) :=
by {
  sorry
}

end arithmetic_sequence_sum_l230_230059


namespace draw_balls_ways_l230_230576

theorem draw_balls_ways :
  let balls := {balls | (count b in balls = 2) ∧ (count w in balls = 6) ∧ (length balls = 8)},
      draw_two, -- function that handles drawing two balls without replacement
      draw := (draw_two balls) 
  in
  count_ways draw (2 black balls) = 10 :=
by sorry

end draw_balls_ways_l230_230576


namespace john_behind_steve_l230_230180

theorem john_behind_steve
  (vJ : ℝ) (vS : ℝ) (ahead : ℝ) (t : ℝ) (d : ℝ)
  (hJ : vJ = 4.2) (hS : vS = 3.8) (hA : ahead = 2) (hT : t = 42.5)
  (h1 : vJ * t = d + ahead)
  (h2 : vS * t + ahead = vJ * t - ahead) :
  d = 15 :=
by
  -- Proof omitted
  sorry

end john_behind_steve_l230_230180


namespace sum_abcd_l230_230595

variable {a b c d : ℚ}

theorem sum_abcd 
  (h : a + 2 = b + 3 ∧ b + 3 = c + 4 ∧ c + 4 = d + 5 ∧ d + 5 = a + b + c + d + 4) : 
  a + b + c + d = -2/3 :=
by sorry

end sum_abcd_l230_230595


namespace solve_inequality_l230_230881

theorem solve_inequality (x : ℝ) : 2 - x < 1 → x > 1 := 
by
  sorry

end solve_inequality_l230_230881


namespace number_of_technicians_l230_230308

theorem number_of_technicians
  (total_workers : ℕ)
  (avg_salary_all : ℝ)
  (avg_salary_techs : ℝ)
  (avg_salary_rest : ℝ)
  (num_techs num_rest : ℕ)
  (h_total_workers : total_workers = 56)
  (h_avg_salary_all : avg_salary_all = 6750)
  (h_avg_salary_techs : avg_salary_techs = 12000)
  (h_avg_salary_rest : avg_salary_rest = 6000)
  (h_eq_workers : num_techs + num_rest = total_workers)
  (h_eq_salaries : (num_techs * avg_salary_techs + num_rest * avg_salary_rest) = total_workers * avg_salary_all) :
  num_techs = 7 := sorry

end number_of_technicians_l230_230308


namespace borrowing_methods_l230_230783

theorem borrowing_methods (A_has_3_books : True) (B_borrows_at_least_one_book : True) :
  (∃ (methods : ℕ), methods = 7) :=
by
  existsi 7
  sorry

end borrowing_methods_l230_230783


namespace sarah_total_height_in_cm_l230_230336

def sarah_height_in_inches : ℝ := 54
def book_thickness_in_inches : ℝ := 2
def conversion_factor : ℝ := 2.54

def total_height_in_inches : ℝ := sarah_height_in_inches + book_thickness_in_inches
def total_height_in_cm : ℝ := total_height_in_inches * conversion_factor

theorem sarah_total_height_in_cm : total_height_in_cm = 142.2 :=
by
  -- Skip the proof for now
  sorry

end sarah_total_height_in_cm_l230_230336


namespace total_commute_time_l230_230188

theorem total_commute_time 
  (first_bus : ℕ) (delay1 : ℕ) (wait1 : ℕ) 
  (second_bus : ℕ) (delay2 : ℕ) (wait2 : ℕ) 
  (third_bus : ℕ) (delay3 : ℕ) 
  (arrival_time : ℕ) :
  first_bus = 40 →
  delay1 = 10 →
  wait1 = 10 →
  second_bus = 50 →
  delay2 = 5 →
  wait2 = 15 →
  third_bus = 95 →
  delay3 = 15 →
  arrival_time = 540 →
  first_bus + delay1 + wait1 + second_bus + delay2 + wait2 + third_bus + delay3 = 240 :=
by
  intros
  sorry

end total_commute_time_l230_230188


namespace sasha_remainder_is_20_l230_230079

theorem sasha_remainder_is_20 (n a b c d : ℤ) (h1 : n = 102 * a + b) (h2 : n = 103 * c + d) (h3 : d = 20 - a) : b = 20 :=
by
  sorry

end sasha_remainder_is_20_l230_230079


namespace initial_distances_l230_230495

theorem initial_distances (x y : ℝ) 
  (h1: x^2 + y^2 = 400)
  (h2: (x - 6)^2 + (y - 8)^2 = 100) : 
  x = 12 ∧ y = 16 := 
by 
  sorry

end initial_distances_l230_230495


namespace find_a10_l230_230833

variable {a : ℕ → ℝ}
variable (h1 : ∀ n m, a (n + 1) = a n + a m)
variable (h2 : a 6 + a 8 = 16)
variable (h3 : a 4 = 1)

theorem find_a10 : a 10 = 15 := by
  sorry

end find_a10_l230_230833


namespace camera_filter_kit_savings_l230_230927

variable (kit_price : ℝ) (single_prices : List ℝ)
variable (correct_saving_amount : ℝ)

theorem camera_filter_kit_savings
    (h1 : kit_price = 145.75)
    (h2 : single_prices = [3 * 9.50, 2 * 15.30, 1 * 20.75, 2 * 25.80])
    (h3 : correct_saving_amount = -14.30) :
    (single_prices.sum - kit_price = correct_saving_amount) :=
by
  sorry

end camera_filter_kit_savings_l230_230927


namespace students_left_zoo_l230_230928

theorem students_left_zoo
  (students_first_class students_second_class : ℕ)
  (chaperones teachers : ℕ)
  (initial_individuals remaining_individuals : ℕ)
  (chaperones_left remaining_individuals_after_chaperones_left : ℕ)
  (remaining_students initial_students : ℕ)
  (H1 : students_first_class = 10)
  (H2 : students_second_class = 10)
  (H3 : chaperones = 5)
  (H4 : teachers = 2)
  (H5 : initial_individuals = students_first_class + students_second_class + chaperones + teachers) 
  (H6 : initial_individuals = 27)
  (H7 : remaining_individuals = 15)
  (H8 : chaperones_left = 2)
  (H9 : remaining_individuals_after_chaperones_left = remaining_individuals - chaperones_left)
  (H10 : remaining_individuals_after_chaperones_left = 13)
  (H11 : remaining_students = remaining_individuals_after_chaperones_left - teachers)
  (H12 : remaining_students = 11)
  (H13 : initial_students = students_first_class + students_second_class)
  (H14 : initial_students = 20) :
  20 - 11 = 9 :=
by sorry

end students_left_zoo_l230_230928


namespace inequality_solution_has_3_integer_solutions_l230_230284

theorem inequality_solution_has_3_integer_solutions (m : ℝ) :
  (∃ x ∈ set.Icc (-4) (-2), x ∈ ℤ ∧ (x + 5 > 0) ∧ (x - m ≤ 1)) →
  (-3 ≤ m ∧ m < -2) :=
by sorry

end inequality_solution_has_3_integer_solutions_l230_230284


namespace rajeev_share_of_profit_l230_230466

open Nat

theorem rajeev_share_of_profit (profit : ℕ) (ramesh_xyz_ratio1 ramesh_xyz_ratio2 xyz_rajeev_ratio1 xyz_rajeev_ratio2 : ℕ) (rajeev_ratio_part : ℕ) (total_parts : ℕ) (individual_part_value : ℕ) :
  profit = 36000 →
  ramesh_xyz_ratio1 = 5 →
  ramesh_xyz_ratio2 = 4 →
  xyz_rajeev_ratio1 = 8 →
  xyz_rajeev_ratio2 = 9 →
  rajeev_ratio_part = 9 →
  total_parts = ramesh_xyz_ratio1 * (xyz_rajeev_ratio1 / ramesh_xyz_ratio2) + xyz_rajeev_ratio1 + xyz_rajeev_ratio2 →
  individual_part_value = profit / total_parts →
  rajeev_ratio_part * individual_part_value = 12000 := 
sorry

end rajeev_share_of_profit_l230_230466


namespace jose_initial_caps_l230_230861

-- Definition of conditions and the problem
def jose_starting_caps : ℤ :=
  let final_caps := 9
  let caps_from_rebecca := 2
  final_caps - caps_from_rebecca

-- Lean theorem to state the required proof
theorem jose_initial_caps : jose_starting_caps = 7 := by
  -- skip proof
  sorry

end jose_initial_caps_l230_230861


namespace bread_slices_remaining_l230_230461

theorem bread_slices_remaining 
  (total_slices : ℕ)
  (third_eaten: ℕ)
  (slices_eaten_breakfast : total_slices / 3 = third_eaten)
  (slices_after_breakfast : total_slices - third_eaten = 8)
  (slices_used_lunch : 2)
  (slices_remaining : 8 - slices_used_lunch = 6) : 
  total_slices = 12 → third_eaten = 4 → slices_remaining = 6 := by 
  sorry

end bread_slices_remaining_l230_230461


namespace trivia_team_absentees_l230_230518

theorem trivia_team_absentees (total_members : ℕ) (total_points : ℕ) (points_per_member : ℕ) 
  (h1 : total_members = 5) 
  (h2 : total_points = 6) 
  (h3 : points_per_member = 2) : 
  total_members - (total_points / points_per_member) = 2 := 
by 
  sorry

end trivia_team_absentees_l230_230518


namespace false_proposition_p_and_q_l230_230558

open Classical

-- Define the propositions
def p (a b c : ℝ) : Prop := b * b = a * c
def q (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- We provide the conditions specified in the problem
variable (a b c : ℝ)
variable (f : ℝ → ℝ)
axiom hq : ∀ x, f x = f (-x)
axiom hp : ¬ (∀ a b c, p a b c ↔ (b ≠ 0 ∧ c ≠ 0 ∧ b * b = a * c))

-- The false proposition among the given options is "p and q"
theorem false_proposition_p_and_q : ¬ (∀ a b c (f : ℝ → ℝ), p a b c ∧ q f) :=
by
  -- This is where the proof would go, but is marked as a placeholder
  sorry

end false_proposition_p_and_q_l230_230558


namespace sugar_for_recipe_l230_230504

theorem sugar_for_recipe (sugar_frosting sugar_cake : ℝ) (h1 : sugar_frosting = 0.6) (h2 : sugar_cake = 0.2) :
  sugar_frosting + sugar_cake = 0.8 :=
by
  sorry

end sugar_for_recipe_l230_230504


namespace joe_initial_paint_l230_230321
-- Use necessary imports

-- Define the hypothesis
def initial_paint_gallons (g : ℝ) :=
  (1 / 4) * g + (1 / 7) * (3 / 4) * g = 128.57

-- Define the theorem
theorem joe_initial_paint (P : ℝ) (h : initial_paint_gallons P) : P = 360 :=
  sorry

end joe_initial_paint_l230_230321


namespace smallest_positive_real_x_l230_230277

theorem smallest_positive_real_x :
  ∃ (x : ℝ), x > 0 ∧ ⌊x^2⌋ - x * ⌊x⌋ = 8 ∧ x = 89 / 9 :=
by
  sorry

end smallest_positive_real_x_l230_230277


namespace ratio_fraction_l230_230174

variable (X Y Z : ℝ)
variable (k : ℝ) (hk : k > 0)

-- Given conditions
def ratio_condition := (3 * Y = 2 * X) ∧ (6 * Y = 2 * Z)

-- Statement
theorem ratio_fraction (h : ratio_condition X Y Z) : 
  (2 * X + 3 * Y) / (5 * Z - 2 * X) = 1 / 2 := by
  sorry

end ratio_fraction_l230_230174


namespace rhyme_around_3_7_l230_230895

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def rhymes_around (p q m : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ p ≠ q ∧ ((p < m ∧ q > m ∧ q - m = m - p) ∨ (p > m ∧ q < m ∧ p - m = m - q))

theorem rhyme_around_3_7 : ∃ m : ℕ, rhymes_around 3 7 m ∧ m = 5 :=
by
  sorry

end rhyme_around_3_7_l230_230895


namespace range_of_m_inequality_system_l230_230281

theorem range_of_m_inequality_system (m : ℝ) :
  (∀ x : ℤ, (-5 < x ∧ x ≤ m + 1) ↔ (x = -4 ∨ x = -3 ∨ x = -2)) →
  -3 ≤ m ∧ m < -2 :=
by
  sorry

end range_of_m_inequality_system_l230_230281


namespace total_weight_of_lifts_l230_230706

theorem total_weight_of_lifts
  (F S : ℕ)
  (h1 : F = 600)
  (h2 : 2 * F = S + 300) :
  F + S = 1500 := by
  sorry

end total_weight_of_lifts_l230_230706


namespace agreed_period_of_service_l230_230123

theorem agreed_period_of_service (x : ℕ) (rs800 : ℕ) (rs400 : ℕ) (servant_period : ℕ) (received_amount : ℕ) (uniform : ℕ) (half_period : ℕ) :
  rs800 = 800 ∧ rs400 = 400 ∧ servant_period = 9 ∧ received_amount = 400 ∧ half_period = x / 2 ∧ servant_period = half_period → x = 18 :=
by sorry

end agreed_period_of_service_l230_230123


namespace expression_value_l230_230497

theorem expression_value (x y z : ℤ) (hx : x = 25) (hy : y = 30) (hz : z = 10) :
  (x - (y - z)) - ((x - y) - z) = 20 :=
by
  rw [hx, hy, hz]
  -- After substituting the values, we will need to simplify the expression to reach 20.
  sorry

end expression_value_l230_230497


namespace find_hyperbola_m_l230_230687

theorem find_hyperbola_m (m : ℝ) :
  (∃ (a b : ℝ), a^2 = 16 ∧ b^2 = m ∧ (sqrt (1 + m / 16) = 5 / 4)) → m = 9 :=
by
  intro h
  sorry

end find_hyperbola_m_l230_230687


namespace find_angle_A_max_perimeter_triangle_l230_230314

-- Part 1: Prove the value of angle A
theorem find_angle_A (A B C : ℝ) (h : sin A ^ 2 - sin B ^ 2 - sin C ^ 2 = sin B * sin C) :
  A = 2 * real.pi / 3 := sorry

-- Part 2: Prove the maximum perimeter for BC = 3
theorem max_perimeter_triangle (B C : ℝ) (BC : ℝ) (hBC : BC = 3) (A : ℝ) (hA : A = 2 * real.pi / 3) :
  let b := 2 * real.sqrt 3 * sin B,
      c := 2 * real.sqrt 3 * sin C,
      perimeter := 3 + b + c
  in ∀ d : ℝ, -real.pi / 6 < d ∧ d < real.pi / 6 → 
     B + C = real.pi / 3 → 
     perimeter ≤ (3 + 2 * real.sqrt 3) := sorry

end find_angle_A_max_perimeter_triangle_l230_230314


namespace correct_option_l230_230638

def inverse_proportion (x y : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ y = k / x

theorem correct_option :
  inverse_proportion x y → 
  (y = x + 3 ∨ y = x / 3 ∨ y = 3 / (x ^ 2) ∨ y = 3 / x) → 
  y = 3 / x :=
by
  sorry

end correct_option_l230_230638


namespace find_G_14_l230_230717

noncomputable def G (x : ℝ) : ℝ := sorry

lemma G_at_7 : G 7 = 20 := sorry

lemma functional_equation (x : ℝ) (hx: x ^ 2 + 8 * x + 16 ≠ 0) : 
  G (4 * x) / G (x + 4) = 16 - (96 * x + 128) / (x^2 + 8 * x + 16) := sorry

theorem find_G_14 : G 14 = 96 := sorry

end find_G_14_l230_230717


namespace part1_part2_l230_230317

-- Part (1): Prove that A = 2π/3 given the trigonometric condition.
theorem part1 (A B C : ℝ) (h_condition : sin A ^ 2 - sin B ^ 2 - sin C ^ 2 = sin B * sin C) : 
  A = 2 * π / 3 :=
sorry

-- Part (2): Prove that the maximum perimeter is 3 + 2√3 given BC = 3 and A = 2π/3.
theorem part2 (B C : angle) (h_BC : 3 = 3) (h_A : A = 2 * π / 3) :
  (3 + (sqrt 3) * 2 ≤ perimeter (triangle.mk 3 B C)) :=
sorry

end part1_part2_l230_230317


namespace total_fertilizer_usage_l230_230790

theorem total_fertilizer_usage :
  let daily_A : ℝ := 3 / 12
  let daily_B : ℝ := 4 / 10
  let daily_C : ℝ := 5 / 8
  let final_A : ℝ := daily_A + 6
  let final_B : ℝ := daily_B + 5
  let final_C : ℝ := daily_C + 7
  (final_A + final_B + final_C) = 19.275 := by
  sorry

end total_fertilizer_usage_l230_230790


namespace monotonic_intervals_increasing_monotonic_intervals_decreasing_maximum_value_minimum_value_l230_230553

noncomputable def f (x : ℝ) : ℝ := 3 * (Real.sin (2 * x + (Real.pi / 4)))

theorem monotonic_intervals_increasing (k : ℤ) :
  ∀ x y : ℝ, (k * Real.pi - 3 * Real.pi / 8 ≤ x ∧ x ≤ y ∧ y ≤ k * Real.pi + Real.pi / 8) → f x ≤ f y :=
sorry

theorem monotonic_intervals_decreasing (k : ℤ) :
  ∀ x y : ℝ, (k * Real.pi + Real.pi / 8 ≤ x ∧ x ≤ y ∧ y ≤ k * Real.pi + 5 * Real.pi / 8) → f x ≥ f y :=
sorry

theorem maximum_value (k : ℤ) :
  f (k * Real.pi + Real.pi / 8) = 3 :=
sorry

theorem minimum_value (k : ℤ) :
  f (k * Real.pi - 3 * Real.pi / 8) = -3 :=
sorry

end monotonic_intervals_increasing_monotonic_intervals_decreasing_maximum_value_minimum_value_l230_230553


namespace speed_of_river_l230_230240

theorem speed_of_river (speed_still_water : ℝ) (total_time : ℝ) (total_distance : ℝ) 
  (h_still_water: speed_still_water = 6) 
  (h_total_time: total_time = 1) 
  (h_total_distance: total_distance = 16/3) : 
  ∃ (speed_river : ℝ), speed_river = 2 :=
by 
  -- sorry is used to skip the proof
  sorry

end speed_of_river_l230_230240


namespace sam_final_investment_l230_230038

-- Definitions based on conditions
def initial_investment : ℝ := 10000
def first_interest_rate : ℝ := 0.20
def years_first_period : ℕ := 3
def triple_amount : ℕ := 3
def second_interest_rate : ℝ := 0.15
def years_second_period : ℕ := 1

-- Lean function to accumulate investment with compound interest
def compound_interest (P r: ℝ) (n: ℕ) : ℝ := P * (1 + r) ^ n

-- Sam's investment calculations
def amount_after_3_years : ℝ := compound_interest initial_investment first_interest_rate years_first_period
def new_investment : ℝ := triple_amount * amount_after_3_years
def final_amount : ℝ := compound_interest new_investment second_interest_rate years_second_period

-- Proof goal (statement with the proof skipped)
theorem sam_final_investment : final_amount = 59616 := by
  sorry

end sam_final_investment_l230_230038


namespace range_of_m_l230_230280

open Set

theorem range_of_m (m : ℝ) :
  (∃ f : ℤ → Prop, (∀ x, f x ↔ x + 5 > 0 ∧ x - m ≤ 1) ∧ (∃ a b c : ℤ, f a ∧ f b ∧ f c))
  → (-3 ≤ m ∧ m < -2) := 
sorry

end range_of_m_l230_230280


namespace josette_additional_cost_l230_230027

def small_bottle_cost_eur : ℝ := 1.50
def large_bottle_cost_eur : ℝ := 2.40
def exchange_rate : ℝ := 1.20
def discount_10_percent : ℝ := 0.10
def discount_15_percent : ℝ := 0.15

def initial_small_bottles : ℕ := 3
def initial_large_bottles : ℕ := 2

def initial_total_cost_eur : ℝ :=
  (small_bottle_cost_eur * initial_small_bottles) +
  (large_bottle_cost_eur * initial_large_bottles)

def discounted_cost_eur_10 : ℝ :=
  initial_total_cost_eur * (1 - discount_10_percent)

def additional_bottle_cost_eur : ℝ := small_bottle_cost_eur

def new_total_cost_eur : ℝ :=
  initial_total_cost_eur + additional_bottle_cost_eur

def discounted_cost_eur_15 : ℝ :=
  new_total_cost_eur * (1 - discount_15_percent)

def cost_usd (eur_amount : ℝ) : ℝ :=
  eur_amount * exchange_rate

def discounted_cost_usd_10 : ℝ := cost_usd discounted_cost_eur_10
def discounted_cost_usd_15 : ℝ := cost_usd discounted_cost_eur_15

def additional_cost_usd : ℝ :=
  discounted_cost_usd_15 - discounted_cost_usd_10

theorem josette_additional_cost :
  additional_cost_usd = 0.972 :=
by 
  sorry

end josette_additional_cost_l230_230027


namespace area_C_greater_than_sum_area_A_B_by_159_point_2_percent_l230_230501

-- Define the side lengths of squares A, B, and C
def side_length_A (s : ℝ) : ℝ := s
def side_length_B (s : ℝ) : ℝ := 2 * s
def side_length_C (s : ℝ) : ℝ := 3.6 * s

-- Define the areas of squares A, B, and C
def area_A (s : ℝ) : ℝ := (side_length_A s) ^ 2
def area_B (s : ℝ) : ℝ := (side_length_B s) ^ 2
def area_C (s : ℝ) : ℝ := (side_length_C s) ^ 2

-- Define the sum of areas of squares A and B
def sum_area_A_B (s : ℝ) : ℝ := area_A s + area_B s

-- Prove that the area of square C is 159.2% greater than the sum of areas of squares A and B
theorem area_C_greater_than_sum_area_A_B_by_159_point_2_percent (s : ℝ) : 
  ((area_C s - sum_area_A_B s) / (sum_area_A_B s)) * 100 = 159.2 := 
sorry

end area_C_greater_than_sum_area_A_B_by_159_point_2_percent_l230_230501


namespace brothers_work_rate_l230_230211

theorem brothers_work_rate (A B C : ℝ) :
  (1 / A + 1 / B = 1 / 8) ∧ (1 / A + 1 / C = 1 / 9) ∧ (1 / B + 1 / C = 1 / 10) →
  A = 160 / 19 ∧ B = 160 / 9 ∧ C = 32 / 3 :=
by
  sorry

end brothers_work_rate_l230_230211


namespace tangent_line_at_0_2_is_correct_l230_230877

noncomputable def curve (x : ℝ) : ℝ := Real.exp (-2 * x) + 1

def tangent_line_at_0_2 (x : ℝ) : ℝ := -2 * x + 2

theorem tangent_line_at_0_2_is_correct :
  tangent_line_at_0_2 = fun x => -2 * x + 2 :=
by {
  sorry
}

end tangent_line_at_0_2_is_correct_l230_230877


namespace part1_part2_l230_230288

variables (x y z : ℝ)

-- Conditions
def conditions := (x >= 0) ∧ (y >= 0) ∧ (z >= 0) ∧ (x + y + z = 1)

-- Part 1: Prove 2(x^2 + y^2 + z^2) + 9xyz >= 1
theorem part1 (h : conditions x y z) : 2 * (x^2 + y^2 + z^2) + 9 * x * y * z ≥ 1 :=
sorry

-- Part 2: Prove xy + yz + zx - 3xyz ≤ 1/4
theorem part2 (h : conditions x y z) : x * y + y * z + z * x - 3 * x * y * z ≤ 1 / 4 :=
sorry

end part1_part2_l230_230288


namespace abc_solution_l230_230992

theorem abc_solution (a b c : ℕ) (h1 : a + b = c - 1) (h2 : a^3 + b^3 = c^2 - 1) : 
  (a = 2 ∧ b = 3 ∧ c = 6) ∨ (a = 3 ∧ b = 2 ∧ c = 6) :=
sorry

end abc_solution_l230_230992


namespace find_a_l230_230978

-- Given function and its condition
def f (a x : ℝ) := a * x ^ 3 + 3 * x ^ 2 + 2
def f' (a x : ℝ) := 3 * a * x ^ 2 + 6 * x

-- Condition and proof that a = -2 given the condition f'(-1) = -12
theorem find_a 
  (a : ℝ)
  (h : f' a (-1) = -12) : 
  a = -2 := 
by 
  sorry

end find_a_l230_230978


namespace lottery_sample_representativeness_l230_230878

theorem lottery_sample_representativeness (A B C D : Prop) :
  B :=
by
  sorry

end lottery_sample_representativeness_l230_230878


namespace initial_investment_C_l230_230941

def total_investment : ℝ := 425
def increase_A (a : ℝ) : ℝ := 0.05 * a
def increase_B (b : ℝ) : ℝ := 0.08 * b
def increase_C (c : ℝ) : ℝ := 0.10 * c

theorem initial_investment_C (a b c : ℝ) (h1 : a + b + c = total_investment)
  (h2 : increase_A a = increase_B b) (h3 : increase_B b = increase_C c) : c = 100 := by
  sorry

end initial_investment_C_l230_230941


namespace red_team_score_l230_230583

theorem red_team_score (C R : ℕ) (h1 : C = 95) (h2 : C - R = 19) : R = 76 :=
by
  sorry

end red_team_score_l230_230583


namespace message_channels_encryption_l230_230489

theorem message_channels_encryption :
  ∃ (assign_key : Fin 105 → Fin 105 → Fin 100),
  ∀ (u v w x : Fin 105), 
  u ≠ v → u ≠ w → u ≠ x → v ≠ w → v ≠ x → w ≠ x →
  (assign_key u v = assign_key u w ∧ assign_key u v = assign_key u x ∧ 
   assign_key u v = assign_key v w ∧ assign_key u v = assign_key v x ∧ 
   assign_key u v = assign_key w x) → False :=
by
  sorry

end message_channels_encryption_l230_230489


namespace sqrt_pow_simplification_l230_230523

theorem sqrt_pow_simplification :
  (Real.sqrt ((Real.sqrt 5) ^ 5)) ^ 6 = 125 * (5 ^ (3 / 4)) :=
by
  sorry

end sqrt_pow_simplification_l230_230523


namespace average_new_data_set_is_5_l230_230548

variable {x1 x2 x3 x4 : ℝ}
variable (h1 : x1 > 0) (h2 : x2 > 0) (h3 : x3 > 0) (h4 : x4 > 0)
variable (var_sqr : ℝ) (h_var : var_sqr = (1 / 4) * (x1 ^ 2 + x2 ^ 2 + x3 ^ 2 + x4 ^ 2 - 16))

theorem average_new_data_set_is_5 (h_var : var_sqr = (1 / 4) * (x1 ^ 2 + x2 ^ 2 + x3 ^ 2 + x4 ^ 2 - 16)) : 
  (x1 + 3 + x2 + 3 + x3 + 3 + x4 + 3) / 4 = 5 := 
by 
  sorry

end average_new_data_set_is_5_l230_230548


namespace find_x_l230_230394

theorem find_x (x : ℕ) (h : 27^3 + 27^3 + 27^3 + 27^3 = 3^x) : x = 11 :=
sorry

end find_x_l230_230394


namespace increased_numerator_value_l230_230234

theorem increased_numerator_value (x y a : ℝ) (h1 : x / y = 2 / 5) (h2 : (x + a) / (2 * y) = 1 / 3) (h3 : x + y = 5.25) : a = 1 :=
by
  -- skipped proof: sorry
  sorry

end increased_numerator_value_l230_230234


namespace team_selection_count_l230_230869

-- The problem's known conditions
def boys : ℕ := 10
def girls : ℕ := 12
def team_size : ℕ := 8

-- The number of ways to select a team of 8 members with at least 2 boys and no more than 4 boys
noncomputable def count_ways : ℕ :=
  (Nat.choose boys 2) * (Nat.choose girls 6) +
  (Nat.choose boys 3) * (Nat.choose girls 5) +
  (Nat.choose boys 4) * (Nat.choose girls 4)

-- The main statement to prove
theorem team_selection_count : count_ways = 238570 := by
  sorry

end team_selection_count_l230_230869


namespace second_and_third_finish_job_together_in_8_days_l230_230760

theorem second_and_third_finish_job_together_in_8_days
  (x y : ℕ)
  (h1 : 1/24 + 1/x + 1/y = 1/6) :
  1/x + 1/y = 1/8 :=
by sorry

end second_and_third_finish_job_together_in_8_days_l230_230760


namespace find_mystery_number_l230_230575

theorem find_mystery_number (x : ℕ) (h : x + 45 = 92) : x = 47 :=
sorry

end find_mystery_number_l230_230575


namespace proof_problem_l230_230957

variable (x : Int) (y : Int) (m : Real)

theorem proof_problem :
  ((x = -6 ∧ y = 1 ∧ m = 7.5) ∨ (x = -1 ∧ y = 2 ∧ m = 4)) ↔
  (-2 * x + 3 * y = 2 * m ∧ x - 5 * y = -11 ∧ x < 0 ∧ y > 0)
:= sorry

end proof_problem_l230_230957


namespace not_both_267_and_269_non_standard_l230_230201

def G : ℤ → ℤ := sorry

def exists_x_ne_c (G : ℤ → ℤ) : Prop :=
  ∀ c : ℤ, ∃ x : ℤ, G x ≠ c

def non_standard (G : ℤ → ℤ) (a : ℤ) : Prop :=
  ∀ x : ℤ, G x = G (a - x)

theorem not_both_267_and_269_non_standard (G : ℤ → ℤ)
  (h1 : exists_x_ne_c G) :
  ¬ (non_standard G 267 ∧ non_standard G 269) :=
sorry

end not_both_267_and_269_non_standard_l230_230201


namespace smallest_n_for_multiples_of_2015_l230_230293

theorem smallest_n_for_multiples_of_2015 (n : ℕ) (hn : 0 < n)
  (h5 : (2^n - 1) % 5 = 0)
  (h13 : (2^n - 1) % 13 = 0)
  (h31 : (2^n - 1) % 31 = 0) : n = 60 := by
  sorry

end smallest_n_for_multiples_of_2015_l230_230293


namespace find_7c_plus_7d_l230_230955

noncomputable def f (c d x : ℝ) : ℝ := c * x + d
noncomputable def h (x : ℝ) : ℝ := 7 * x - 6
noncomputable def f_inv (x : ℝ) : ℝ := 7 * x - 1

theorem find_7c_plus_7d (c d : ℝ) (h_def : ∀ x, h x = f_inv x - 5) (f_def : ∀ x, f c d x = c * x + d) (f_inv_def : ∀ x, f_inv x = 7 * x - 1) : 7 * c + 7 * d = 2 := by
  sorry

end find_7c_plus_7d_l230_230955


namespace measure_angle_R_l230_230998

theorem measure_angle_R (P Q R : ℝ) (h1 : P + Q = 60) : R = 120 :=
by
  have sum_of_angles_in_triangle : P + Q + R = 180 := sorry
  rw [h1] at sum_of_angles_in_triangle
  linarith

end measure_angle_R_l230_230998


namespace find_diagonal_length_l230_230260

theorem find_diagonal_length (d : ℝ) (offset1 offset2 : ℝ) (area : ℝ)
  (h1 : offset1 = 9)
  (h2 : offset2 = 6)
  (h3 : area = 300) :
  (1/2) * d * (offset1 + offset2) = area → d = 40 :=
by
  -- placeholder for proof
  sorry

end find_diagonal_length_l230_230260


namespace ab_value_l230_230754

theorem ab_value (a b : ℝ) 
  (h1 : ∃ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1 ∧ (∀ y : ℝ, (x = 0 ∧ (y = 5 ∨ y = -5)))))
  (h2 : ∃ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1 ∧ (∀ x : ℝ, (y = 0 ∧ (x = 8 ∨ x = -8))))) :
  |a * b| = Real.sqrt 867.75 :=
by
  sorry

end ab_value_l230_230754


namespace cookies_left_l230_230603

theorem cookies_left (total_cookies : ℕ) (fraction_given : ℚ) (given_cookies : ℕ) (remaining_cookies : ℕ) 
  (h1 : total_cookies = 20)
  (h2 : fraction_given = 2/5)
  (h3 : given_cookies = fraction_given * total_cookies)
  (h4 : remaining_cookies = total_cookies - given_cookies) :
  remaining_cookies = 12 :=
by
  sorry

end cookies_left_l230_230603


namespace total_donation_correct_l230_230738

-- Define the donations to each orphanage
def first_orphanage_donation : ℝ := 175.00
def second_orphanage_donation : ℝ := 225.00
def third_orphanage_donation : ℝ := 250.00

-- State the total donation
def total_donation : ℝ := 650.00

-- The theorem statement to be proved
theorem total_donation_correct :
  first_orphanage_donation + second_orphanage_donation + third_orphanage_donation = total_donation :=
by
  sorry

end total_donation_correct_l230_230738


namespace always_composite_for_x64_l230_230030

def is_composite (n : ℕ) : Prop :=
  ∃ a b : ℕ, 1 < a ∧ 1 < b ∧ a * b = n

theorem always_composite_for_x64 (n : ℕ) : is_composite (n^4 + 64) :=
by
  sorry

end always_composite_for_x64_l230_230030


namespace expansion_contains_x4_l230_230156

noncomputable def binomial_coeff (n k : ℕ) : ℕ :=
  Nat.choose n k

noncomputable def expansion_term (x : ℂ) (i : ℂ) : ℂ :=
  binomial_coeff 6 2 * x^4 * i^2

theorem expansion_contains_x4 (x i : ℂ) (hi : i = Complex.I) : 
  expansion_term x i = -15 * x^4 := by
  sorry

end expansion_contains_x4_l230_230156


namespace growth_comparison_l230_230218

theorem growth_comparison (x : ℝ) (h : ℝ) (hx : x > 0) : 
  (0 < x ∧ x < 1 / 2 → (x + h) - x > ((x + h)^2 - x^2)) ∧
  (x > 1 / 2 → ((x + h)^2 - x^2) > (x + h) - x) :=
by
  sorry

end growth_comparison_l230_230218


namespace small_circle_to_large_circle_ratio_l230_230440

theorem small_circle_to_large_circle_ratio (a b : ℝ) (h : π * b^2 - π * a^2 = 3 * π * a^2) :
  a / b = 1 / 2 :=
sorry

end small_circle_to_large_circle_ratio_l230_230440


namespace sasha_remainder_20_l230_230074

theorem sasha_remainder_20
  (n a b c d : ℕ)
  (h1 : n = 102 * a + b)
  (h2 : 0 ≤ b ∧ b ≤ 101)
  (h3 : n = 103 * c + d)
  (h4 : d = 20 - a) :
  b = 20 :=
by
  sorry

end sasha_remainder_20_l230_230074


namespace sasha_remainder_l230_230088

theorem sasha_remainder (n a b c d : ℕ) (h1 : n = 102 * a + b) (h2 : n = 103 * c + d)
  (h3 : d = 20 - a) (h4 : 0 ≤ b ∧ b ≤ 101) : b = 20 :=
by
  sorry

end sasha_remainder_l230_230088


namespace tan_195_l230_230973

theorem tan_195 (a : ℝ) (h : Real.cos 165 = a) : Real.tan 195 = - (Real.sqrt (1 - a^2)) / a := 
sorry

end tan_195_l230_230973


namespace susan_typing_time_l230_230323

theorem susan_typing_time :
  let Jonathan_rate := 1 -- page per minute
  let Jack_rate := 5 / 3 -- pages per minute
  let combined_rate := 4 -- pages per minute
  ∃ S : ℝ, (1 + 1/S + 5/3 = 4) → S = 30 :=
by
  sorry

end susan_typing_time_l230_230323


namespace smallest_positive_x_l230_230268

theorem smallest_positive_x (x : ℝ) (h : ⌊x^2⌋ - x * ⌊x⌋ = 8) : x = 89 / 9 :=
sorry

end smallest_positive_x_l230_230268


namespace grasshopper_total_distance_l230_230376

theorem grasshopper_total_distance :
  let initial := 2
  let first_jump := -3
  let second_jump := 8
  let final_jump := -1
  abs (first_jump - initial) + abs (second_jump - first_jump) + abs (final_jump - second_jump) = 25 :=
by
  sorry

end grasshopper_total_distance_l230_230376


namespace team_e_speed_l230_230212

-- Definitions and conditions
variables (v t : ℝ)
def distance_team_e := 300 = v * t
def distance_team_a := 300 = (v + 5) * (t - 3)

-- The theorem statement: Prove that given the conditions, Team E's speed is 20 mph
theorem team_e_speed (h1 : distance_team_e v t) (h2 : distance_team_a v t) : v = 20 :=
by
  sorry -- proof steps are omitted as requested

end team_e_speed_l230_230212


namespace girls_select_same_colored_marble_l230_230232

def probability_same_color (total_white total_black girls boys : ℕ) : ℚ :=
  let prob_white := (total_white * (total_white - 1)) / ((total_white + total_black) * (total_white + total_black - 1))
  let prob_black := (total_black * (total_black - 1)) / ((total_white + total_black) * (total_white + total_black - 1))
  prob_white + prob_black

theorem girls_select_same_colored_marble :
  probability_same_color 2 2 2 2 = 1 / 3 :=
by
  sorry

end girls_select_same_colored_marble_l230_230232


namespace orthogonal_lines_solution_l230_230887

theorem orthogonal_lines_solution (a b c d : ℝ)
  (h1 : b - a = 0)
  (h2 : c - a = 2)
  (h3 : 12 * d - a = 1)
  : d = 3 / 11 :=
by {
  sorry
}

end orthogonal_lines_solution_l230_230887


namespace maximal_probability_C_n_l230_230328

open Classical
open Set

variable (A : Set ℕ) (hA : A = {1, 2})
variable (B : Set ℕ) (hB : B = {1, 2, 3})

def event_C_n (n : ℕ) : Prop := ∃ (a ∈ A) (b ∈ B), a + b = n

theorem maximal_probability_C_n :
  ∃ (n : ℕ), (2 ≤ n ∧ n ≤ 5) ∧ 
  (∀ m, (2 ≤ m ∧ m ≤ 5) → 
    probability (event_C_n A B m) ≤ probability (event_C_n A B n)) 
  ∧ (n = 3 ∨ n = 4) :=
sorry

end maximal_probability_C_n_l230_230328


namespace chord_length_l230_230108

theorem chord_length (R a b : ℝ) (hR : a + b = R) (hab : a * b = 10) 
    (h_nonneg : 0 ≤ R ∧ 0 ≤ a ∧ 0 ≤ b) : ∃ L : ℝ, L = 2 * Real.sqrt 10 :=
by
  sorry

end chord_length_l230_230108


namespace measure_of_angle_BYZ_l230_230525

-- Definitions based on conditions
def angle_A : ℝ := 50
def angle_B : ℝ := 70
def angle_C : ℝ := 60

-- Theorem statement based on the proof problem
theorem measure_of_angle_BYZ 
 (h1 : ∃ γ : Type, incircle (triangle ABC) γ ∧ circumcircle (triangle XYZ) γ)
 (h2 : point_on_line_segment X BC)
 (h3 : point_on_line_segment Y AB)
 (h4 : point_on_line_segment Z AC)
 (h5 : ∠A = angle_A)
 (h6 : ∠B = angle_B)
 (h7 : ∠C = angle_C) : 
 ∠BYZ = 60 :=
sorry

end measure_of_angle_BYZ_l230_230525


namespace product_of_fractions_l230_230953

open BigOperators

theorem product_of_fractions :
  (∏ n in Finset.range 9, (n + 2)^3 - 1) / (∏ n in Finset.range 9, (n + 2)^3 + 1) = 74 / 55 :=
by
  sorry

end product_of_fractions_l230_230953


namespace gcd_256_180_600_l230_230629

theorem gcd_256_180_600 : Nat.gcd (Nat.gcd 256 180) 600 = 4 :=
by
  -- This proof is marked as 'sorry' to indicate it is a place holder
  sorry

end gcd_256_180_600_l230_230629


namespace stratified_sampling_household_l230_230605

/-
  Given:
  - Total valid questionnaires: 500,000.
  - Number of people who purchased:
    - clothing, shoes, and hats: 198,000,
    - household goods: 94,000,
    - cosmetics: 116,000,
    - home appliances: 92,000.
  - Number of questionnaires selected from the "cosmetics" category: 116.
  
  Prove:
  - The number of questionnaires that should be selected from the "household goods" category is 94.
-/

theorem stratified_sampling_household (total_valid: ℕ)
  (clothing_shoes_hats: ℕ)
  (household_goods: ℕ)
  (cosmetics: ℕ)
  (home_appliances: ℕ)
  (sample_cosmetics: ℕ) :
  total_valid = 500000 →
  clothing_shoes_hats = 198000 →
  household_goods = 94000 →
  cosmetics = 116000 →
  home_appliances = 92000 →
  sample_cosmetics = 116 →
  (116 * household_goods = sample_cosmetics * cosmetics) →
  116 * 94000 = 116 * 116000 →
  94000 = 116000 →
  94 = 94 := by
  intros
  sorry

end stratified_sampling_household_l230_230605


namespace original_number_of_men_l230_230791

theorem original_number_of_men (x : ℕ) (h1 : 40 * x = 60 * (x - 5)) : x = 15 :=
by
  sorry

end original_number_of_men_l230_230791


namespace sufficient_but_not_necessary_condition_l230_230352

theorem sufficient_but_not_necessary_condition (x y m : ℝ) (h: x^2 + y^2 - 4 * x + 2 * y + m = 0):
  (m = 0) → (5 > m) ∧ ((5 > m) → (m ≠ 0)) :=
by
  sorry

end sufficient_but_not_necessary_condition_l230_230352


namespace chocolate_bars_produced_per_minute_l230_230101

theorem chocolate_bars_produced_per_minute
  (sugar_per_bar : ℝ)
  (total_sugar : ℝ)
  (time_in_minutes : ℝ) 
  (bars_per_min : ℝ) :
  sugar_per_bar = 1.5 →
  total_sugar = 108 →
  time_in_minutes = 2 →
  bars_per_min = 36 :=
sorry

end chocolate_bars_produced_per_minute_l230_230101


namespace cos_2theta_plus_sin_2theta_l230_230424

theorem cos_2theta_plus_sin_2theta (θ : ℝ) (h : 3 * Real.sin θ = Real.cos θ) : 
  Real.cos (2 * θ) + Real.sin (2 * θ) = 7 / 5 :=
by
  sorry

end cos_2theta_plus_sin_2theta_l230_230424


namespace mark_second_part_playtime_l230_230017

theorem mark_second_part_playtime (total_time initial_time sideline_time : ℕ) 
  (h1 : total_time = 90) (h2 : initial_time = 20) (h3 : sideline_time = 35) :
  total_time - initial_time - sideline_time = 35 :=
sorry

end mark_second_part_playtime_l230_230017


namespace cost_per_topping_is_2_l230_230842

theorem cost_per_topping_is_2 : 
  ∃ (x : ℝ), 
    let large_pizza_cost := 14 
    let num_large_pizzas := 2 
    let num_toppings_per_pizza := 3 
    let tip_rate := 0.25 
    let total_cost := 50 
    let cost_pizzas := num_large_pizzas * large_pizza_cost 
    let num_toppings := num_large_pizzas * num_toppings_per_pizza 
    let cost_toppings := num_toppings * x 
    let before_tip_cost := cost_pizzas + cost_toppings 
    let tip := tip_rate * before_tip_cost 
    let final_cost := before_tip_cost + tip 
    final_cost = total_cost ∧ x = 2 := 
by
  simp
  sorry

end cost_per_topping_is_2_l230_230842


namespace sasha_remainder_l230_230066

statement:
  theorem sasha_remainder (n a b c d : ℕ) (h1 : n = 102 * a + b) (h2 : n = 103 * c + d) (h3 : a + d = 20) (h4: 0 ≤ b ∧ b ≤ 101) : 
  b = 20 :=
by
  sorry

end sasha_remainder_l230_230066


namespace number_of_managers_in_sample_l230_230519

def totalStaff : ℕ := 160
def salespeople : ℕ := 104
def managers : ℕ := 32
def logisticsPersonnel : ℕ := 24
def sampleSize : ℕ := 20

theorem number_of_managers_in_sample : 
  (managers * (sampleSize / totalStaff) = 4) := by
  sorry

end number_of_managers_in_sample_l230_230519


namespace largest_integer_solution_l230_230636

theorem largest_integer_solution : ∃ x : ℤ, (x ≤ 10) ∧ (∀ y : ℤ, (y > 10 → (y / 4 + 5 / 6 < 7 / 2) = false)) :=
sorry

end largest_integer_solution_l230_230636


namespace find_k_l230_230020

-- Define the conditions
def parabola (k : ℝ) (x : ℝ) : ℝ := x^2 + 2 * x + k

-- Theorem statement
theorem find_k (k : ℝ) : (∀ x : ℝ, parabola k x = 0 → x = -1) → k = 1 :=
by
  sorry

end find_k_l230_230020


namespace number_of_yellow_crayons_l230_230092

theorem number_of_yellow_crayons : 
  ∃ (R B Y : ℕ), 
  R = 14 ∧ 
  B = R + 5 ∧ 
  Y = 2 * B - 6 ∧ 
  Y = 32 :=
by
  sorry

end number_of_yellow_crayons_l230_230092


namespace fewer_people_correct_l230_230388

def pop_Springfield : ℕ := 482653
def pop_total : ℕ := 845640
def pop_new_city : ℕ := pop_total - pop_Springfield
def fewer_people : ℕ := pop_Springfield - pop_new_city

theorem fewer_people_correct : fewer_people = 119666 :=
by
  unfold fewer_people
  unfold pop_new_city
  unfold pop_total
  unfold pop_Springfield
  sorry

end fewer_people_correct_l230_230388


namespace scientific_notation_of_coronavirus_diameter_l230_230051

theorem scientific_notation_of_coronavirus_diameter : 
  (0.00000011 : ℝ) = 1.1 * 10^(-7) :=
  sorry

end scientific_notation_of_coronavirus_diameter_l230_230051


namespace smallest_m_inequality_l230_230148

theorem smallest_m_inequality (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_sum : a + b + c = 1) : 27 * (a^3 + b^3 + c^3) ≥ 6 * (a^2 + b^2 + c^2) + 1 :=
sorry

end smallest_m_inequality_l230_230148


namespace no_such_function_exists_l230_230530

-- Let's define the assumptions as conditions
def condition1 (f : ℝ → ℝ) := ∀ x : ℝ, f (x^2) - (f x)^2 ≥ 1 / 4
def distinct_values (f : ℝ → ℝ) := ∀ x y : ℝ, x ≠ y → f x ≠ f y

-- Now we state the main theorem
theorem no_such_function_exists : ¬ ∃ f : ℝ → ℝ, condition1 f ∧ distinct_values f :=
sorry

end no_such_function_exists_l230_230530


namespace f_2021_l230_230153

noncomputable def f : ℝ → ℝ := sorry
axiom odd_f : ∀ x : ℝ, f (-x) = -f (x)
axiom period_f : ∀ x : ℝ, f (x) = f (2 - x)
axiom f_neg1 : f (-1) = 1

theorem f_2021 : f (2021) = -1 :=
by
  sorry

end f_2021_l230_230153


namespace second_number_less_than_first_by_16_percent_l230_230781

variable (X : ℝ)

theorem second_number_less_than_first_by_16_percent
  (h1 : X > 0)
  (first_num : ℝ := 0.75 * X)
  (second_num : ℝ := 0.63 * X) :
  (first_num - second_num) / first_num * 100 = 16 := by
  sorry

end second_number_less_than_first_by_16_percent_l230_230781


namespace sasha_remainder_l230_230085

theorem sasha_remainder (n a b c d : ℕ) 
  (h1 : n = 102 * a + b) 
  (h2 : n = 103 * c + d) 
  (h3 : a + d = 20)
  (hb : 0 ≤ b ∧ b ≤ 101) : 
  b = 20 :=
sorry

end sasha_remainder_l230_230085


namespace sasha_remainder_is_20_l230_230078

theorem sasha_remainder_is_20 (n a b c d : ℤ) (h1 : n = 102 * a + b) (h2 : n = 103 * c + d) (h3 : d = 20 - a) : b = 20 :=
by
  sorry

end sasha_remainder_is_20_l230_230078


namespace vectors_parallel_implies_fraction_l230_230694

theorem vectors_parallel_implies_fraction (α : ℝ) :
  let a := (Real.sin α, 3)
  let b := (Real.cos α, 1)
  (a.1 / b.1 = 3) → (Real.sin (2 * α) / (Real.cos α) ^ 2 = 6) :=
by
  sorry

end vectors_parallel_implies_fraction_l230_230694


namespace sasha_remainder_20_l230_230073

theorem sasha_remainder_20
  (n a b c d : ℕ)
  (h1 : n = 102 * a + b)
  (h2 : 0 ≤ b ∧ b ≤ 101)
  (h3 : n = 103 * c + d)
  (h4 : d = 20 - a) :
  b = 20 :=
by
  sorry

end sasha_remainder_20_l230_230073


namespace find_p_l230_230678

theorem find_p (p : ℕ) (hp : Nat.Prime p) (hp2 : Nat.Prime (5 * p^2 - 2)) : p = 3 :=
sorry

end find_p_l230_230678


namespace tan_alpha_eq_neg2_l230_230485

theorem tan_alpha_eq_neg2 {α : ℝ} {x y : ℝ} (hx : x = -2) (hy : y = 4) (hM : (x, y) = (-2, 4)) :
  Real.tan α = -2 :=
by
  sorry

end tan_alpha_eq_neg2_l230_230485


namespace cubes_with_even_red_faces_count_l230_230943

def block_dimensions : ℕ × ℕ × ℕ := (6, 4, 2)
def is_painted_red : Prop := true
def total_cubes : ℕ := 48
def cubes_with_even_red_faces : ℕ := 24

theorem cubes_with_even_red_faces_count :
  ∀ (dimensions : ℕ × ℕ × ℕ) (painted_red : Prop) (cubes_count : ℕ), 
  dimensions = block_dimensions → painted_red = is_painted_red → cubes_count = total_cubes → 
  (cubes_with_even_red_faces = 24) :=
by intros dimensions painted_red cubes_count h1 h2 h3; exact sorry

end cubes_with_even_red_faces_count_l230_230943


namespace Helen_taller_than_Amy_l230_230131

-- Definitions from conditions
def Angela_height : ℕ := 157
def Amy_height : ℕ := 150
def Helen_height := Angela_height - 4

-- Question as a theorem
theorem Helen_taller_than_Amy : Helen_height - Amy_height = 3 := by
  sorry

end Helen_taller_than_Amy_l230_230131


namespace correct_weights_l230_230035

def weight (item : String) : Nat :=
  match item with
  | "Banana" => 140
  | "Pear" => 120
  | "Melon" => 1500
  | "Tomato" => 150
  | "Apple" => 170
  | _ => 0

theorem correct_weights :
  weight "Banana" = 140 ∧
  weight "Pear" = 120 ∧
  weight "Melon" = 1500 ∧
  weight "Tomato" = 150 ∧
  weight "Apple" = 170 ∧
  (weight "Melon" > weight "Pear") ∧
  (weight "Melon" < weight "Tomato") :=
by
  sorry

end correct_weights_l230_230035


namespace exam_total_boys_l230_230346

theorem exam_total_boys (T F : ℕ) (avg_total avg_passed avg_failed : ℕ) 
    (H1 : avg_total = 40) (H2 : avg_passed = 39) (H3 : avg_failed = 15) (H4 : 125 > 0) (H5 : 125 * avg_passed + (T - 125) * avg_failed = T * avg_total) : T = 120 :=
by
  sorry

end exam_total_boys_l230_230346


namespace electric_blankets_sold_l230_230898

theorem electric_blankets_sold (T H E : ℕ)
  (h1 : 2 * T + 6 * H + 10 * E = 1800)
  (h2 : T = 7 * H)
  (h3 : H = 2 * E) : 
  E = 36 :=
by {
  sorry
}

end electric_blankets_sold_l230_230898


namespace find_divisor_l230_230433

theorem find_divisor (x y : ℕ) (h1 : (x - 5) / 7 = 7) (h2 : (x - 14) / y = 4) : y = 10 :=
sorry

end find_divisor_l230_230433


namespace gwen_remaining_money_l230_230820

theorem gwen_remaining_money:
  ∀ (Gwen_received Gwen_spent Gwen_remaining: ℕ),
    Gwen_received = 5 →
    Gwen_spent = 3 →
    Gwen_remaining = Gwen_received - Gwen_spent →
    Gwen_remaining = 2 :=
by
  intros Gwen_received Gwen_spent Gwen_remaining h₁ h₂ h₃
  rw [h₁, h₂] at h₃
  exact h₃

end gwen_remaining_money_l230_230820


namespace bread_problem_l230_230367

variable (x : ℝ)

theorem bread_problem (h1 : x > 0) :
  (15 / x) - 1 = 14 / (x + 2) :=
sorry

end bread_problem_l230_230367


namespace stamps_problem_l230_230931

theorem stamps_problem (x y : ℕ) : 
  2 * x + 6 * x + 5 * y / 2 = 60 → x = 5 ∧ y = 8 ∧ 6 * x = 30 :=
by 
  sorry

end stamps_problem_l230_230931


namespace find_total_pupils_l230_230761

-- Define the conditions for the problem
def diff1 : ℕ := 85 - 45
def diff2 : ℕ := 79 - 49
def diff3 : ℕ := 64 - 34
def total_diff : ℕ := diff1 + diff2 + diff3
def avg_increase : ℕ := 3

-- Assert that the number of pupils n satisfies the given conditions
theorem find_total_pupils (n : ℕ) (h_diff : total_diff = 100) (h_avg_inc : avg_increase * n = total_diff) : n = 33 :=
by
  sorry

end find_total_pupils_l230_230761


namespace average_customers_per_table_l230_230942

-- Definitions for conditions
def tables : ℝ := 9.0
def women : ℝ := 7.0
def men : ℝ := 3.0

-- Proof problem statement
theorem average_customers_per_table : (women + men) / tables = 10.0 / 9.0 :=
by
  sorry

end average_customers_per_table_l230_230942


namespace part1_part2_l230_230311

theorem part1 (A B C : ℝ) (h : sin A ^ 2 - sin B ^ 2 - sin C ^ 2 = sin B * sin C) : A = 2 * π / 3 :=
sorry

theorem part2 (b c : ℝ) (h1 : sin (2 * π / 3) ^ 2 - sin b ^ 2 - sin c ^ 2 = sin b * sin c)
  (h2 : b + c = 2 * sqrt 3) : 3 * 2 + b + c = 3 + 2 * sqrt 3 :=
sorry

end part1_part2_l230_230311


namespace spring_festival_scientific_notation_l230_230660

noncomputable def scientific_notation := (260000000: ℝ) = (2.6 * 10^8)

theorem spring_festival_scientific_notation : scientific_notation :=
by
  -- proof logic goes here
  sorry

end spring_festival_scientific_notation_l230_230660


namespace range_of_t_l230_230677

theorem range_of_t (a b : ℝ) (h : a^2 + a*b + b^2 = 1) :
  ∃ t : ℝ, (t = a^2 - a*b + b^2) ∧ (1/3 ≤ t ∧ t ≤ 3) :=
sorry

end range_of_t_l230_230677


namespace smallest_positive_real_number_l230_230274

noncomputable def smallest_x : ℝ :=
  let x := 89 / 9 in x

theorem smallest_positive_real_number :
  ∀ x : ℝ, (x > 0 ∧ ⌊x^2⌋ - x * ⌊x⌋ = 8) → x ≥ smallest_x  :=
by
  sorry

end smallest_positive_real_number_l230_230274


namespace cos_add_pi_over_4_l230_230970

theorem cos_add_pi_over_4 (α : ℝ) (h : Real.sin (α - π/4) = 1/3) : Real.cos (π/4 + α) = -1/3 := 
  sorry

end cos_add_pi_over_4_l230_230970


namespace khali_shovels_snow_l230_230715

theorem khali_shovels_snow :
  let section1_length := 30
  let section1_width := 3
  let section1_depth := 1
  let section2_length := 15
  let section2_width := 2
  let section2_depth := 0.5
  let volume1 := section1_length * section1_width * section1_depth
  let volume2 := section2_length * section2_width * section2_depth
  volume1 + volume2 = 105 :=
by 
  sorry

end khali_shovels_snow_l230_230715


namespace solve_for_x_l230_230337

theorem solve_for_x (x : ℝ) (h : (3 / 4) + (1 / x) = 7 / 8) : x = 8 :=
sorry

end solve_for_x_l230_230337


namespace Miss_Adamson_paper_usage_l230_230724

-- Definitions from the conditions
def classes : ℕ := 4
def students_per_class : ℕ := 20
def sheets_per_student : ℕ := 5

-- Total number of students
def total_students : ℕ := classes * students_per_class

-- Total number of sheets of paper
def total_sheets : ℕ := total_students * sheets_per_student

-- The proof problem
theorem Miss_Adamson_paper_usage : total_sheets = 400 :=
by
  -- Proof to be filled in
  sorry

end Miss_Adamson_paper_usage_l230_230724


namespace max_min_condition_monotonic_condition_l230_230372

-- (1) Proving necessary and sufficient condition for f(x) to have both a maximum and minimum value
theorem max_min_condition (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧ -2*x₁ + a - (1/x₁) = 0 ∧ -2*x₂ + a - (1/x₂) = 0) ↔ a > Real.sqrt 8 :=
sorry

-- (2) Proving the range of values for a such that f(x) is monotonic on [1, 2]
theorem monotonic_condition (a : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → (-2 * x + a - (1 / x)) ≥ 0) ∨
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → (-2 * x + a - (1 / x)) ≤ 0) ↔ a ≤ 3 ∨ a ≥ 4.5 :=
sorry

end max_min_condition_monotonic_condition_l230_230372


namespace complement_intersection_l230_230167

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {2, 4}
def N : Set ℕ := {3, 5}

theorem complement_intersection (hU: U = {1, 2, 3, 4, 5}) (hM: M = {2, 4}) (hN: N = {3, 5}) : 
  (U \ M) ∩ N = {3, 5} := 
by 
  sorry

end complement_intersection_l230_230167


namespace ascending_order_l230_230948

theorem ascending_order (a b c d : ℝ) (h1 : a = -6) (h2 : b = 0) (h3 : c = Real.sqrt 5) (h4 : d = Real.pi) :
  a < b ∧ b < c ∧ c < d :=
by
  sorry

end ascending_order_l230_230948


namespace trigonometric_ratio_l230_230834

theorem trigonometric_ratio (θ : ℝ) (h : Real.sin θ + 2 * Real.cos θ = 1) :
  (Real.sin θ - Real.cos θ) / (Real.sin θ + Real.cos θ) = -7 ∨
  (Real.sin θ - Real.cos θ) / (Real.sin θ + Real.cos θ) = 1 :=
sorry

end trigonometric_ratio_l230_230834


namespace bridge_length_correct_l230_230652

noncomputable def train_length : ℝ := 120
noncomputable def train_speed_kmph : ℝ := 45
noncomputable def crossing_time_seconds : ℝ := 30

noncomputable def train_speed_mps : ℝ := (train_speed_kmph * 1000) / 3600
noncomputable def total_distance : ℝ := train_speed_mps * crossing_time_seconds
noncomputable def bridge_length : ℝ := total_distance - train_length

theorem bridge_length_correct : bridge_length = 255 := by
  sorry

end bridge_length_correct_l230_230652


namespace jose_internet_speed_l230_230860

-- Define the given conditions
def file_size : ℕ := 160
def upload_time : ℕ := 20

-- Define the statement we need to prove
theorem jose_internet_speed : file_size / upload_time = 8 :=
by
  -- Proof should be provided here
  sorry

end jose_internet_speed_l230_230860


namespace factorize_expression_l230_230813

theorem factorize_expression (x y : ℂ) : (x * y^2 - x = x * (y + 1) * (y - 1)) :=
sorry

end factorize_expression_l230_230813


namespace find_n_l230_230302

-- Define the variables d, Q, r, m, and n
variables (d Q r m n : ℝ)

-- Define the conditions Q = d / ((1 + r)^n - m) and m < (1 + r)^n
def conditions (d Q r m n : ℝ) : Prop :=
  Q = d / ((1 + r)^n - m) ∧ m < (1 + r)^n

theorem find_n (d Q r m : ℝ) (h : conditions d Q r m n) : 
  n = (Real.log (d / Q + m)) / (Real.log (1 + r)) :=
sorry

end find_n_l230_230302


namespace f_leq_zero_l230_230828

noncomputable def f (x a : ℝ) := x * Real.log x - a * x^2 + (2 * a - 1) * x

theorem f_leq_zero (a x : ℝ) (h1 : 1/2 < a) (h2 : a ≤ 1) (hx : 0 < x) :
  f x a ≤ 0 :=
sorry

end f_leq_zero_l230_230828


namespace cats_left_in_store_l230_230380

theorem cats_left_in_store 
  (initial_siamese : ℕ := 25)
  (initial_persian : ℕ := 18)
  (initial_house : ℕ := 12)
  (initial_maine_coon : ℕ := 10)
  (sold_siamese : ℕ := 6)
  (sold_persian : ℕ := 4)
  (sold_maine_coon : ℕ := 3)
  (sold_house : ℕ := 0)
  (remaining_siamese : ℕ := 19)
  (remaining_persian : ℕ := 14)
  (remaining_house : ℕ := 12)
  (remaining_maine_coon : ℕ := 7) : 
  initial_siamese - sold_siamese = remaining_siamese ∧
  initial_persian - sold_persian = remaining_persian ∧
  initial_house - sold_house = remaining_house ∧
  initial_maine_coon - sold_maine_coon = remaining_maine_coon :=
by sorry

end cats_left_in_store_l230_230380


namespace four_digit_numbers_greater_3999_with_middle_product_exceeding_12_l230_230568

-- Number of four-digit numbers greater than 3999 such that the product of the middle two digits > 12 is 4260
theorem four_digit_numbers_greater_3999_with_middle_product_exceeding_12
  {d1 d2 d3 d4 : ℕ}
  (h1 : 4 ≤ d1 ∧ d1 ≤ 9)
  (h2 : 0 ≤ d4 ∧ d4 ≤ 9)
  (h3 : 1 ≤ d2 ∧ d2 ≤ 9)
  (h4 : 1 ≤ d3 ∧ d3 ≤ 9)
  (h5 : d2 * d3 > 12) :
  (6 * 71 * 10 = 4260) :=
by
  sorry

end four_digit_numbers_greater_3999_with_middle_product_exceeding_12_l230_230568


namespace equation_solution_l230_230471

theorem equation_solution (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 0) :
  (2 * x / (x - 2) - 2 = 1 / (x * (x - 2))) ↔ x = 1 / 4 :=
by
  sorry

end equation_solution_l230_230471


namespace tims_initial_cans_l230_230763
noncomputable theory

-- Definitions extracted from conditions
def initial_cans (x : ℕ) : ℕ := x
def after_jeff (x : ℕ) : ℕ := x - 6
def after_buying_more (x : ℕ) : ℕ := after_jeff x + (after_jeff x / 2)

-- Statement of the problem
theorem tims_initial_cans (x : ℕ) (h : after_buying_more x = 24) : x = 22 :=
by
  sorry

end tims_initial_cans_l230_230763


namespace expression_undefined_count_l230_230544

theorem expression_undefined_count : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ ∀ x : ℝ,
  ((x = x1 ∨ x = x2) ↔ (x^2 - 2*x - 3 = 0 ∨ x - 3 = 0)) ∧ 
  ((x^2 - 2*x - 3) * (x - 3) = 0 → (x = x1 ∨ x = x2)) :=
by
  sorry

end expression_undefined_count_l230_230544


namespace bamboo_capacity_l230_230177

theorem bamboo_capacity :
  ∃ (a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 d : ℚ),
    a_1 + a_2 + a_3 = 4 ∧
    a_6 + a_7 + a_8 + a_9 = 3 ∧
    a_2 = a_1 + d ∧
    a_3 = a_1 + 2*d ∧
    a_4 = a_1 + 3*d ∧
    a_5 = a_1 + 4*d ∧
    a_7 = a_1 + 5*d ∧
    a_8 = a_1 + 6*d ∧
    a_9 = a_1 + 7*d ∧
    a_4 = 1 + 8/66 ∧
    a_5 = 1 + 1/66 :=
sorry

end bamboo_capacity_l230_230177


namespace total_gulbis_l230_230208

theorem total_gulbis (dureums fish_per_dureum : ℕ) (h1 : dureums = 156) (h2 : fish_per_dureum = 20) : dureums * fish_per_dureum = 3120 :=
by
  sorry

end total_gulbis_l230_230208


namespace mrs_franklin_needs_more_valentines_l230_230731

theorem mrs_franklin_needs_more_valentines (valentines_have : ℝ) (students : ℝ) : valentines_have = 58 ∧ students = 74 → students - valentines_have = 16 :=
by
  sorry

end mrs_franklin_needs_more_valentines_l230_230731


namespace eggs_collected_l230_230535

def total_eggs_collected (b1 e1 b2 e2 : ℕ) : ℕ :=
  b1 * e1 + b2 * e2

theorem eggs_collected :
  total_eggs_collected 450 36 405 42 = 33210 :=
by
  sorry

end eggs_collected_l230_230535


namespace athlete_more_stable_l230_230513

theorem athlete_more_stable (var_A var_B : ℝ) 
                                (h1 : var_A = 0.024) 
                                (h2 : var_B = 0.008) 
                                (h3 : var_A > var_B) : 
  var_B < var_A :=
by
  exact h3

end athlete_more_stable_l230_230513


namespace inequality_solution_has_3_integer_solutions_l230_230283

theorem inequality_solution_has_3_integer_solutions (m : ℝ) :
  (∃ x ∈ set.Icc (-4) (-2), x ∈ ℤ ∧ (x + 5 > 0) ∧ (x - m ≤ 1)) →
  (-3 ≤ m ∧ m < -2) :=
by sorry

end inequality_solution_has_3_integer_solutions_l230_230283


namespace present_population_l230_230203

theorem present_population (P : ℕ) (h1 : P * 11 / 10 = 264) : P = 240 :=
by sorry

end present_population_l230_230203


namespace balls_per_bag_l230_230793

theorem balls_per_bag (total_balls : ℕ) (total_bags : ℕ) (h1 : total_balls = 36) (h2 : total_bags = 9) : total_balls / total_bags = 4 :=
by
  sorry

end balls_per_bag_l230_230793


namespace sasha_remainder_l230_230071

theorem sasha_remainder (n a b c d : ℕ) (h1 : n = 102 * a + b) (h2 : n = 103 * c + d) 
  (h3 : a + d = 20) (h_b_range : 0 ≤ b ∧ b ≤ 101) (h_d_range : 0 ≤ d ∧ d ≤ 102) : b = 20 := 
sorry

end sasha_remainder_l230_230071


namespace total_sheets_of_paper_l230_230728

theorem total_sheets_of_paper (classes : ℕ) (students_per_class : ℕ) (sheets_per_student : ℕ) 
  (h1 : classes = 4) (h2 : students_per_class = 20) (h3 : sheets_per_student = 5) : 
  (classes * students_per_class * sheets_per_student) = 400 := 
by {
  sorry
}

end total_sheets_of_paper_l230_230728


namespace quadruple_perimeter_l230_230340

variable (s : ℝ) -- side length of the original square
variable (x : ℝ) -- perimeter of the original square
variable (P_new : ℝ) -- new perimeter after side length is quadrupled

theorem quadruple_perimeter (h1 : x = 4 * s) (h2 : P_new = 4 * (4 * s)) : P_new = 4 * x := 
by sorry

end quadruple_perimeter_l230_230340


namespace prism_diagonals_not_valid_l230_230499

theorem prism_diagonals_not_valid
  (a b c : ℕ)
  (h3 : a^2 + b^2 = 3^2 ∨ b^2 + c^2 = 3^2 ∨ a^2 + c^2 = 3^2)
  (h4 : a^2 + b^2 = 4^2 ∨ b^2 + c^2 = 4^2 ∨ a^2 + c^2 = 4^2)
  (h6 : a^2 + b^2 = 6^2 ∨ b^2 + c^2 = 6^2 ∨ a^2 + c^2 = 6^2) :
  False := 
sorry

end prism_diagonals_not_valid_l230_230499


namespace equation_pattern_l230_230848
open Nat

theorem equation_pattern (n : ℕ) (h_pos : 0 < n) : n * (n + 2) + 1 = (n + 1) ^ 2 := by
  sorry

end equation_pattern_l230_230848


namespace grid_segments_divisible_by_4_l230_230515

-- Definition: square grid where each cell has a side length of 1
structure SquareGrid (n : ℕ) :=
  (segments : ℕ)

-- Condition: Function to calculate the total length of segments in the grid
def total_length {n : ℕ} (Q : SquareGrid n) : ℕ := Q.segments

-- Lean 4 statement: Prove that for any grid, the total length is divisible by 4
theorem grid_segments_divisible_by_4 {n : ℕ} (Q : SquareGrid n) :
  total_length Q % 4 = 0 :=
sorry

end grid_segments_divisible_by_4_l230_230515


namespace solve_for_M_l230_230695

theorem solve_for_M (a b M : ℝ) (h : (a + 2 * b) ^ 2 = (a - 2 * b) ^ 2 + M) : M = 8 * a * b :=
by sorry

end solve_for_M_l230_230695


namespace find_A_max_perimeter_of_triangle_l230_230319

-- Definition of the given problem conditions
def triangle_condition (A B C : ℝ) : Prop :=
  sin(A)^2 - sin(B)^2 - sin(C)^2 = sin(B) * sin(C)

-- (1) Proving the value of A given the condition
theorem find_A (A B C : ℝ) (h : triangle_condition A B C) : A = 2 * π / 3 :=
by sorry

-- (2) Proving the maximum perimeter given BC = 3 and A = 2π/3
theorem max_perimeter_of_triangle (B C : ℝ) (BC : ℝ) (hBC : BC = 3) (hA : 2 * π / 3 = 2 * π / 3) : 
  ∃ (P : ℝ), P = 3 + 2 * sqrt 3 :=
by sorry

end find_A_max_perimeter_of_triangle_l230_230319


namespace rectangle_segments_sum_l230_230872

theorem rectangle_segments_sum :
  let EF := 6
  let FG := 8
  let n := 210
  let diagonal_length := Real.sqrt (EF^2 + FG^2)
  let segment_length (k : ℕ) : ℝ := diagonal_length * (n - k) / n
  let sum_segments := 2 * (Finset.sum (Finset.range 210) segment_length) - diagonal_length
  sum_segments = 2080 := by
  sorry

end rectangle_segments_sum_l230_230872


namespace parabola_focus_distance_l230_230003

-- defining the problem in Lean
theorem parabola_focus_distance
  (A : ℝ × ℝ)
  (hA : A.2^2 = 4 * A.1)
  (h_distance : |A.1| = 3)
  (F : ℝ × ℝ)
  (hF : F = (1, 0)) :
  |(A.1 - F.1)^2 + (A.2 - F.2)^2| = 4 := 
sorry

end parabola_focus_distance_l230_230003


namespace lines_intersect_and_not_perpendicular_l230_230757

theorem lines_intersect_and_not_perpendicular (a : ℝ) :
  (∃ (x y : ℝ), 3 * x + 3 * y + a = 0 ∧ 3 * x - 2 * y + 1 = 0) ∧ 
  ¬ (∃ k1 k2 : ℝ, k1 = -1 ∧ k2 = 3 / 2 ∧ k1 ≠ k2 ∧ k1 * k2 = -1) :=
by
  sorry

end lines_intersect_and_not_perpendicular_l230_230757


namespace find_x_l230_230117

theorem find_x :
  ∃ x : ℝ, 12.1212 + x - 9.1103 = 20.011399999999995 ∧ x = 18.000499999999995 :=
sorry

end find_x_l230_230117


namespace negation_of_p_l230_230420

variable (x : ℝ)

def proposition_p : Prop := ∀ x : ℝ, x^2 + 1 ≥ 1

theorem negation_of_p : ¬ (∀ x : ℝ, x^2 + 1 ≥ 1) ↔ (∃ x : ℝ, x^2 + 1 < 1) :=
by sorry

end negation_of_p_l230_230420


namespace sum_first_four_terms_geo_seq_l230_230971

theorem sum_first_four_terms_geo_seq (q : ℝ) (a_1 : ℝ)
  (h1 : q ≠ 1) 
  (h2 : a_1 * (a_1 * q) * (a_1 * q^2) = -1/8)
  (h3 : 2 * (a_1 * q^3) = (a_1 * q) + (a_1 * q^2)) :
  (a_1 + (a_1 * q) + (a_1 * q^2) + (a_1 * q^3)) = 5 / 8 :=
  sorry

end sum_first_four_terms_geo_seq_l230_230971


namespace sasha_remainder_20_l230_230076

theorem sasha_remainder_20
  (n a b c d : ℕ)
  (h1 : n = 102 * a + b)
  (h2 : 0 ≤ b ∧ b ≤ 101)
  (h3 : n = 103 * c + d)
  (h4 : d = 20 - a) :
  b = 20 :=
by
  sorry

end sasha_remainder_20_l230_230076


namespace find_n_l230_230758

theorem find_n (a b n : ℕ) (k l m : ℤ) 
  (ha : a % n = 2) 
  (hb : b % n = 3) 
  (h_ab : a > b) 
  (h_ab_mod : (a - b) % n = 5) : 
  n = 6 := 
sorry

end find_n_l230_230758


namespace find_quadrant_372_degrees_l230_230810

theorem find_quadrant_372_degrees : 
  ∃ q : ℕ, q = 1 ↔ (372 % 360 = 12 ∧ (0 ≤ 12 ∧ 12 < 90)) :=
by
  sorry

end find_quadrant_372_degrees_l230_230810


namespace divisibility_by_11_l230_230449

theorem divisibility_by_11
  (n : ℕ) (hn : n ≥ 2)
  (h : (n^2 + (4^n) + (7^n)) % n = 0) :
  (n^2 + 4^n + 7^n) % 11 = 0 := 
by
  sorry

end divisibility_by_11_l230_230449


namespace julia_gold_watch_percentage_l230_230714

def silver_watches : ℕ := 20
def bronze_watches : ℕ := 3 * silver_watches
def total_watches_before_gold : ℕ := silver_watches + bronze_watches
def total_watches_after_gold : ℕ := 88
def gold_watches : ℕ := total_watches_after_gold - total_watches_before_gold
def percentage_gold_watches : ℚ := (gold_watches : ℚ) / (total_watches_after_gold : ℚ) * 100

theorem julia_gold_watch_percentage :
  percentage_gold_watches = 9.09 := by
  sorry

end julia_gold_watch_percentage_l230_230714


namespace k_less_than_two_l230_230692

theorem k_less_than_two
    (x : ℝ)
    (k : ℝ)
    (y : ℝ)
    (h : y = (2 - k) / x)
    (h1 : (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0)) : k < 2 :=
by
  sorry

end k_less_than_two_l230_230692


namespace no_simultaneous_inequalities_l230_230032

theorem no_simultaneous_inequalities (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  ¬ ((a + b < c + d) ∧ ((a + b) * (c + d) < a * b + c * d) ∧ ((a + b) * c * d < a * b * (c + d))) :=
by
  sorry

end no_simultaneous_inequalities_l230_230032


namespace polar_line_equation_l230_230231

theorem polar_line_equation (r θ: ℝ) (p : r = 3 ∧ θ = 0) : r = 3 := 
by 
  sorry

end polar_line_equation_l230_230231


namespace mean_score_all_students_l230_230331

theorem mean_score_all_students
  (M A E : ℝ) (m a e : ℝ)
  (hM : M = 78)
  (hA : A = 68)
  (hE : E = 82)
  (h_ratio_ma : m / a = 4 / 5)
  (h_ratio_mae : (m + a) / e = 9 / 2)
  : (M * m + A * a + E * e) / (m + a + e) = 74.4 := by
  sorry

end mean_score_all_students_l230_230331


namespace inequality_relations_l230_230287

noncomputable def a : ℝ := Real.sqrt 3
noncomputable def b : ℝ := Real.sqrt 125 ^ (1 / 6)
noncomputable def c : ℝ := Real.log 7 / Real.log (1 / 6)

theorem inequality_relations :
  c < a ∧ a < b := 
by 
  sorry

end inequality_relations_l230_230287


namespace row_time_to_100_yards_l230_230467

theorem row_time_to_100_yards :
  let init_width_yd := 50
  let final_width_yd := 100
  let increase_width_yd_per_10m := 2
  let rowing_speed_mps := 5
  let current_speed_mps := 1
  let yard_to_meter := 0.9144
  let init_width_m := init_width_yd * yard_to_meter
  let final_width_m := final_width_yd * yard_to_meter
  let width_increase_m_per_10m := increase_width_yd_per_10m * yard_to_meter
  let total_width_increase := (final_width_m - init_width_m)
  let num_segments := total_width_increase / width_increase_m_per_10m
  let total_distance := num_segments * 10
  let effective_speed := rowing_speed_mps + current_speed_mps
  let time := total_distance / effective_speed
  time = 41.67 := by
  sorry

end row_time_to_100_yards_l230_230467


namespace percentage_of_volume_is_P_l230_230874

noncomputable def volumeOfSolutionP {P Q : ℝ} (h : 0.80 * P + 0.55 * Q = 0.675 * (P + Q)) : ℝ := 
(P / (P + Q)) * 100

theorem percentage_of_volume_is_P {P Q: ℝ} (h : 0.80 * P + 0.55 * Q = 0.675 * (P + Q)) : 
  volumeOfSolutionP h = 50 :=
sorry

end percentage_of_volume_is_P_l230_230874


namespace find_quadratic_polynomial_with_conditions_l230_230264

noncomputable def quadratic_polynomial : polynomial ℝ :=
  3 * (X - C (2 + 2 * I)) * (X - C (2 - 2 * I))

theorem find_quadratic_polynomial_with_conditions :
  (quadratic_polynomial = 3 * X^2 - 12 * X + 24) :=
by
  sorry

end find_quadratic_polynomial_with_conditions_l230_230264


namespace tree_planting_activity_l230_230668

variables (trees_first_group trees_second_group people_first_group people_second_group : ℕ)
variable (average_trees_per_person_first_group average_trees_per_person_second_group : ℕ)

theorem tree_planting_activity :
  trees_first_group = 12 →
  trees_second_group = 36 →
  people_second_group = people_first_group + 6 →
  average_trees_per_person_first_group = trees_first_group / people_first_group →
  average_trees_per_person_second_group = trees_second_group / people_second_group →
  average_trees_per_person_first_group = average_trees_per_person_second_group →
  people_first_group = 3 := 
by 
  intros h1 h2 h3 h4 h5 h6
  sorry

end tree_planting_activity_l230_230668


namespace minimize_z_l230_230435

theorem minimize_z (x y : ℝ) (h1 : 2 * x - y ≥ 0) (h2 : y ≥ x) (h3 : y ≥ -x + 2) :
  ∃ (x y : ℝ), (z = 2 * x + y) ∧ z = 8 / 3 :=
by
  sorry

end minimize_z_l230_230435


namespace monotonicity_F_range_k_l230_230298

noncomputable def f (x : ℝ) : ℝ := Real.log (1 + x) - Real.log (1 - x)
noncomputable def F (x : ℝ) (a : ℝ) : ℝ := f x + a * x
noncomputable def g (x : ℝ) (k : ℝ) : ℝ := f x - k * (x^3 - 3 * x)

theorem monotonicity_F (a : ℝ) (ha : a ≠ 0) :
(∀ x : ℝ, (-1 < x ∧ x < 1) → 
    (if (-2 ≤ a ∧ a < 0) ∨ (a > 0) then 0 ≤ (a - a * x^2 + 2) / (1 - x^2)
     else if a < -2 then 
        ((-1 < x ∧ x < -Real.sqrt ((a + 2) / a)) ∨ (Real.sqrt ((a + 2) / a) < x ∧ x < 1)) → 0 ≤ (a - a * x^2 + 2) / (1 - x^2) ∧ 
        (-Real.sqrt ((a + 2) / a) < x ∧ x < Real.sqrt ((a + 2) / a)) → 0 > (a - a * x^2 + 2) / (1 - x^2)
    else false)) :=
sorry

theorem range_k (k : ℝ) (hk : ∀ x : ℝ, (0 < x ∧ x < 1) → f x > k * (x^3 - 3 * x)) :
k ≥ -2 / 3 :=
sorry

end monotonicity_F_range_k_l230_230298


namespace binom_8_3_eq_56_l230_230804

def binom (n k : ℕ) : ℕ :=
(n.factorial) / (k.factorial * (n - k).factorial)

theorem binom_8_3_eq_56 : binom 8 3 = 56 := by
  sorry

end binom_8_3_eq_56_l230_230804


namespace divisibility_of_powers_l230_230551

theorem divisibility_of_powers (a b c d m : ℤ) (h_odd : m % 2 = 1)
  (h_sum_div : m ∣ (a + b + c + d))
  (h_sum_squares_div : m ∣ (a^2 + b^2 + c^2 + d^2)) : 
  m ∣ (a^4 + b^4 + c^4 + d^4 + 4 * a * b * c * d) :=
sorry

end divisibility_of_powers_l230_230551


namespace yellow_crayons_count_l230_230095

def red_crayons := 14
def blue_crayons := red_crayons + 5
def yellow_crayons := 2 * blue_crayons - 6

theorem yellow_crayons_count : yellow_crayons = 32 := by
  sorry

end yellow_crayons_count_l230_230095


namespace oranges_in_bin_after_changes_l230_230129

-- Define the initial number of oranges
def initial_oranges : ℕ := 34

-- Define the number of oranges thrown away
def oranges_thrown_away : ℕ := 20

-- Define the number of new oranges added
def new_oranges_added : ℕ := 13

-- Theorem statement to prove the final number of oranges in the bin
theorem oranges_in_bin_after_changes :
  initial_oranges - oranges_thrown_away + new_oranges_added = 27 := by
  sorry

end oranges_in_bin_after_changes_l230_230129


namespace not_unique_equilateral_by_one_angle_and_opposite_side_l230_230105

-- Definitions related to triangles
structure Triangle :=
  (a b c : ℝ) -- sides
  (alpha beta gamma : ℝ) -- angles

-- Definition of triangle types
def isIsosceles (t : Triangle) : Prop :=
  (t.a = t.b ∨ t.b = t.c ∨ t.a = t.c)

def isRight (t : Triangle) : Prop :=
  (t.alpha = 90 ∨ t.beta = 90 ∨ t.gamma = 90)

def isEquilateral (t : Triangle) : Prop :=
  (t.a = t.b ∧ t.b = t.c ∧ t.alpha = 60 ∧ t.beta = 60 ∧ t.gamma = 60)

def isScalene (t : Triangle) : Prop :=
  (t.a ≠ t.b ∧ t.b ≠ t.c ∧ t.a ≠ t.c)

-- Proof that having one angle and the side opposite it does not determine an equilateral triangle.
theorem not_unique_equilateral_by_one_angle_and_opposite_side :
  ¬ ∀ (t1 t2 : Triangle), (isEquilateral t1 ∧ isEquilateral t2 →
    t1.alpha = t2.alpha ∧ t1.a = t2.a →
    t1 = t2) := sorry

end not_unique_equilateral_by_one_angle_and_opposite_side_l230_230105


namespace total_animal_sightings_l230_230704

def A_Jan := 26
def A_Feb := 3 * A_Jan
def A_Mar := A_Feb / 2

theorem total_animal_sightings : A_Jan + A_Feb + A_Mar = 143 := by
  sorry

end total_animal_sightings_l230_230704


namespace find_m_n_l230_230962

theorem find_m_n (m n : ℕ) (h1 : m ≥ 0) (h2 : n ≥ 0) (h3 : 3^m - 7^n = 2) : m = 2 ∧ n = 1 := 
sorry

end find_m_n_l230_230962


namespace eggs_in_box_l230_230096

theorem eggs_in_box (initial_count : ℝ) (added_count : ℝ) (total_count : ℝ) 
  (h_initial : initial_count = 47.0) 
  (h_added : added_count = 5.0) : total_count = 52.0 :=
by 
  sorry

end eggs_in_box_l230_230096


namespace remaining_cubes_l230_230220

-- The configuration of the initial cube and the properties of a layer
def initial_cube : ℕ := 10
def total_cubes : ℕ := 1000
def layer_cubes : ℕ := (initial_cube * initial_cube)

-- The proof problem: Prove that the remaining number of cubes is 900 after removing one layer
theorem remaining_cubes : total_cubes - layer_cubes = 900 := 
by 
  sorry

end remaining_cubes_l230_230220


namespace avg_salary_rest_of_workers_l230_230477

theorem avg_salary_rest_of_workers (avg_all : ℝ) (avg_technicians : ℝ) (total_workers : ℕ) (n_technicians : ℕ) (avg_rest : ℝ) :
  avg_all = 8000 ∧ avg_technicians = 20000 ∧ total_workers = 49 ∧ n_technicians = 7 →
  avg_rest = 6000 :=
by
  sorry

end avg_salary_rest_of_workers_l230_230477


namespace show_revenue_and_vacancies_l230_230850

theorem show_revenue_and_vacancies:
  let total_seats := 600
  let vip_seats := 50
  let general_seats := 400
  let balcony_seats := 150
  let vip_price := 40
  let general_price := 25
  let balcony_price := 15
  let vip_filled_rate := 0.80
  let general_filled_rate := 0.70
  let balcony_filled_rate := 0.50
  let vip_filled := vip_filled_rate * vip_seats
  let general_filled := general_filled_rate * general_seats
  let balcony_filled := balcony_filled_rate * balcony_seats
  let vip_revenue := vip_filled * vip_price
  let general_revenue := general_filled * general_price
  let balcony_revenue := balcony_filled * balcony_price
  let overall_revenue := vip_revenue + general_revenue + balcony_revenue
  let vip_vacant := vip_seats - vip_filled
  let general_vacant := general_seats - general_filled
  let balcony_vacant := balcony_seats - balcony_filled
  vip_revenue = 1600 ∧
  general_revenue = 7000 ∧
  balcony_revenue = 1125 ∧
  overall_revenue = 9725 ∧
  vip_vacant = 10 ∧
  general_vacant = 120 ∧
  balcony_vacant = 75 :=
by
  sorry

end show_revenue_and_vacancies_l230_230850


namespace rotated_parabola_eq_l230_230807

theorem rotated_parabola_eq :
  ∀ x y : ℝ, y = x^2 → ∃ y' x' : ℝ, (y' = (-x':ℝ)^2) := sorry

end rotated_parabola_eq_l230_230807


namespace geometric_sequence_sum_l230_230709

theorem geometric_sequence_sum (a : ℕ → ℝ) (r : ℝ) 
    (h : ∀ n : ℕ, a (n + 1) = a n * r) 
    (h1 : a 1 + a 2 = 40) 
    (h2 : a 3 + a 4 = 60) : 
    a 7 + a 8 = 135 :=
sorry

end geometric_sequence_sum_l230_230709


namespace function_value_l230_230540

theorem function_value (f : ℝ → ℝ) (h : ∀ x, x + 17 = 60 * f x) : f 3 = 1 / 3 :=
by
  sorry

end function_value_l230_230540


namespace smallest_a_divisible_by_1984_l230_230529

theorem smallest_a_divisible_by_1984 :
  ∃ a : ℕ, (∀ n : ℕ, n % 2 = 1 → 1984 ∣ (47^n + a * 15^n)) ∧ a = 1055 := 
by 
  sorry

end smallest_a_divisible_by_1984_l230_230529


namespace simplify_expression_l230_230193

variable (a b : ℤ)

theorem simplify_expression :
  (15 * a + 45 * b) + (20 * a + 35 * b) - (25 * a + 55 * b) + (30 * a - 5 * b) = 
  40 * a + 20 * b :=
by
  sorry

end simplify_expression_l230_230193


namespace line_parallel_to_parallel_set_l230_230430

variables {Point Line Plane : Type} 
variables (a : Line) (α : Plane)
variables (parallel : Line → Plane → Prop) (parallel_set : Line → Plane → Prop)

-- Definition for line parallel to plane
axiom line_parallel_to_plane : parallel a α

-- Goal: line a is parallel to a set of parallel lines within plane α
theorem line_parallel_to_parallel_set (h : parallel a α) : parallel_set a α := 
sorry

end line_parallel_to_parallel_set_l230_230430


namespace magic_8_ball_probability_l230_230891

theorem magic_8_ball_probability :
  let p := 3 / 8
  let q := 5 / 8
  let n := 7
  let k := 3
  (Nat.choose 7 3) * (p^3) * (q^4) = 590625 / 2097152 :=
by
  let p := 3 / 8
  let q := 5 / 8
  let n := 7
  let k := 3
  sorry

end magic_8_ball_probability_l230_230891


namespace range_of_a_l230_230297

open Complex Real

theorem range_of_a (a : ℝ) (h : abs (1 + a * Complex.I) ≤ 2) : a ∈ Set.Icc (-Real.sqrt 3) (Real.sqrt 3) :=
sorry

end range_of_a_l230_230297


namespace roots_square_sum_eq_l230_230172

theorem roots_square_sum_eq (r s t p q : ℝ) 
  (h1 : r + s + t = p) 
  (h2 : r * s + r * t + s * t = q) 
  (h3 : r * s * t = r) :
  r^2 + s^2 + t^2 = p^2 - 2 * q :=
by
  sorry

end roots_square_sum_eq_l230_230172


namespace expression_value_l230_230914

theorem expression_value : (5^2 - 5) * (6^2 - 6) - (7^2 - 7) = 558 := by
  sorry

end expression_value_l230_230914


namespace slope_angle_range_l230_230014

-- Given conditions expressed as Lean definitions
def circle_eq (x y : ℝ) : Prop := (x - 2)^2 + (y - 2)^2 = 18

def line_eq (a b x y : ℝ) : Prop := a * x + b * y = 0

def distance_point_to_line (a b x y d : ℝ) : Prop := 
  abs (a * x + b * y) / sqrt (a^2 + b^2) = d

-- Given conditions specified in Lean
constant a b : ℝ
constant α : ℝ
constant slope_ranges : ∀ m : ℝ, 2 - sqrt 3 ≤ m ∧ m ≤ 2 + sqrt 3 → 
                                  ∀ α : ℝ, tan α = m → 
                                  π / 12 ≤ α ∧ α ≤ 5 * π / 12

-- Main theorem to prove
theorem slope_angle_range (h_circle : ∀ x y, circle_eq x y) 
                          (h_distance : ∃ x y, circle_eq x y ∧ distance_point_to_line a b x y (2 * sqrt 2)) 
                          (h_inequality : (a / b)^2 + 4 * (a / b) + 1 ≤ 0) : 
                          π / 12 ≤ α ∧ α ≤ 5 * π / 12 := 
begin
  sorry
end

end slope_angle_range_l230_230014


namespace gcd_power_of_two_sub_one_l230_230353

def a : ℤ := 2^1100 - 1
def b : ℤ := 2^1122 - 1
def c : ℤ := 2^22 - 1

theorem gcd_power_of_two_sub_one :
  Int.gcd (2^1100 - 1) (2^1122 - 1) = 2^22 - 1 := by
  sorry

end gcd_power_of_two_sub_one_l230_230353


namespace equation_solution_l230_230472

theorem equation_solution (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 0) :
  (2 * x / (x - 2) - 2 = 1 / (x * (x - 2))) ↔ x = 1 / 4 :=
by
  sorry

end equation_solution_l230_230472


namespace gcd_of_256_180_600_l230_230624

-- Define the numbers
def n1 := 256
def n2 := 180
def n3 := 600

-- Statement to prove the GCD of the three numbers is 12
theorem gcd_of_256_180_600 : Nat.gcd (Nat.gcd n1 n2) n3 = 12 := 
by {
  -- Prime factorizations given as part of the conditions:
  have pf1 : n1 = 2^8 := by norm_num,
  have pf2 : n2 = 2^2 * 3^2 * 5 := by norm_num,
  have pf3 : n3 = 2^3 * 3 * 5^2 := by norm_num,
  
  -- Calculate GCD step by step should be done here, 
  -- But replaced by sorry as per the instructions.
  sorry
}

end gcd_of_256_180_600_l230_230624


namespace first_number_is_48_l230_230110

-- Definitions of the conditions
def ratio (A B : ℕ) := 8 * B = 9 * A
def lcm (A B : ℕ) := Nat.lcm A B = 432

-- The statement to prove
theorem first_number_is_48 (A B : ℕ) (h_ratio : ratio A B) (h_lcm : lcm A B) : A = 48 :=
by
  sorry

end first_number_is_48_l230_230110


namespace triangle_inequality_l230_230597

-- Let α, β, γ be the angles of a triangle opposite to its sides with lengths a, b, and c, respectively.
variables (α β γ a b c : ℝ)

-- Assume that α, β, γ are positive.
axiom positive_angles : α > 0 ∧ β > 0 ∧ γ > 0
-- Assume that a, b, c are the sides opposite to angles α, β, γ respectively.
axiom positive_sides : a > 0 ∧ b > 0 ∧ c > 0

theorem triangle_inequality :
  a * (1 / β + 1 / γ) + b * (1 / γ + 1 / α) + c * (1 / α + 1 / β) ≥ 
  2 * (a / α + b / β + c / γ) :=
sorry

end triangle_inequality_l230_230597


namespace mike_marbles_l230_230721

theorem mike_marbles (original : ℕ) (given : ℕ) (final : ℕ) 
  (h1 : original = 8) 
  (h2 : given = 4)
  (h3 : final = original - given) : 
  final = 4 :=
by sorry

end mike_marbles_l230_230721


namespace total_meals_per_week_l230_230562

-- Definitions for the conditions
def first_restaurant_meals := 20
def second_restaurant_meals := 40
def third_restaurant_meals := 50
def days_in_week := 7

-- The theorem for the total meals per week
theorem total_meals_per_week : 
  (first_restaurant_meals + second_restaurant_meals + third_restaurant_meals) * days_in_week = 770 := 
by
  sorry

end total_meals_per_week_l230_230562


namespace identify_heaviest_and_lightest_coin_l230_230098

theorem identify_heaviest_and_lightest_coin :
  ∀ (coins : Fin 10 → ℕ), 
  (∀ i j, i ≠ j → coins i ≠ coins j) → 
  ∃ (seq : List (Fin 10 × Fin 10)), 
  seq.length = 13 ∧ 
  (∀ (i j : Fin 10), (i, j) ∈ seq → 
    (coins i < coins j ∨ coins i > coins j)) ∧ 
  (∃ (heaviest lightest : Fin 10),
    (∀ coin, coins coin ≤ coins heaviest) ∧ (∀ coin, coins coin ≥ coins lightest)) :=
by
  intros coins h_coins
  exists [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), -- initial pairs
          (0, 2), (2, 4), (4, 6), (6, 8),         -- heaviest coin comparisons
          (1, 3), (3, 5), (5, 7), (7, 9)]         -- lightest coin comparisons
  constructor
  . -- length check
    rfl
  . constructor
    . -- all comparisons
      intros i j h_pair
      cases h_pair; simp; solve_by_elim
    . -- finding heaviest and lightest coins
      exists 8, 9
      constructor
      . -- all coins are less than or equal to the heaviest
        sorry
      . -- all coins are greater than or equal to the lightest
        sorry

end identify_heaviest_and_lightest_coin_l230_230098


namespace solve_for_q_l230_230045

theorem solve_for_q (k l q : ℕ) (h1 : (2 : ℚ) / 3 = k / 45) (h2 : (2 : ℚ) / 3 = (k + l) / 75) (h3 : (2 : ℚ) / 3 = (q - l) / 105) : q = 90 :=
sorry

end solve_for_q_l230_230045


namespace number_of_books_l230_230133

-- Define the conditions
def ratio_books : ℕ := 7
def ratio_pens : ℕ := 3
def ratio_notebooks : ℕ := 2
def total_items : ℕ := 600

-- Define the theorem and the goal to prove
theorem number_of_books (sets : ℕ) (ratio_books : ℕ := 7) (total_items : ℕ := 600) : 
  sets = total_items / (7 + 3 + 2) → 
  sets * ratio_books = 350 :=
by
  sorry

end number_of_books_l230_230133


namespace intersection_of_sets_l230_230574

def setA (x : ℝ) : Prop := 2 * x + 1 > 0
def setB (x : ℝ) : Prop := abs (x - 1) < 2

theorem intersection_of_sets :
  {x : ℝ | setA x} ∩ {x : ℝ | setB x} = {x : ℝ | -1/2 < x ∧ x < 3} :=
by 
  sorry  -- Placeholder for the proof

end intersection_of_sets_l230_230574


namespace karlsson_weight_l230_230640

variable {F K M : ℕ}

theorem karlsson_weight (h1 : F + K = M + 120) (h2 : K + M = F + 60) : K = 90 := by
  sorry

end karlsson_weight_l230_230640


namespace boys_of_other_communities_l230_230851

theorem boys_of_other_communities (total_boys : ℕ) (percentage_muslims percentage_hindus percentage_sikhs : ℝ) 
  (h_tm : total_boys = 1500)
  (h_pm : percentage_muslims = 37.5)
  (h_ph : percentage_hindus = 25.6)
  (h_ps : percentage_sikhs = 8.4) : 
  ∃ (boys_other_communities : ℕ), boys_other_communities = 428 :=
by
  sorry

end boys_of_other_communities_l230_230851


namespace sequence_a4_l230_230832

theorem sequence_a4 :
  (∀ n : ℕ, n > 0 → ∀ (a : ℕ → ℝ),
    (a 1 = 1) →
    (∀ n > 0, a (n + 1) = (1 / 2) * a n + 1 / (2 ^ n)) →
    a 4 = 1 / 2) :=
by
  sorry

end sequence_a4_l230_230832


namespace neils_cookies_l230_230604

theorem neils_cookies (total_cookies : ℕ) (fraction_given_to_friend : ℚ) (remaining_cookies : ℕ) :
  total_cookies = 20 →
  fraction_given_to_friend = 2 / 5 →
  remaining_cookies = total_cookies - (total_cookies * (fraction_given_to_friend.num : ℕ) / fraction_given_to_friend.denom) →
  remaining_cookies = 12 :=
by
  intros h_total h_fraction h_remaining
  rw [h_total, h_fraction, h_remaining]
  norm_num

end neils_cookies_l230_230604


namespace all_selected_prob_l230_230033

def probability_of_selection (P_ram P_ravi P_raj : ℚ) : ℚ :=
  P_ram * P_ravi * P_raj

theorem all_selected_prob :
  let P_ram := 2/7
  let P_ravi := 1/5
  let P_raj := 3/8
  probability_of_selection P_ram P_ravi P_raj = 3/140 := by
  sorry

end all_selected_prob_l230_230033


namespace rachel_minutes_before_bed_l230_230334

-- Define the conditions in the Lean Lean.
def minutes_spent_solving_before_bed (m : ℕ) : Prop :=
  let problems_solved_before_bed := 5 * m
  let problems_finished_at_lunch := 16
  let total_problems_solved := 76
  problems_solved_before_bed + problems_finished_at_lunch = total_problems_solved

-- The statement we want to prove
theorem rachel_minutes_before_bed : ∃ m : ℕ, minutes_spent_solving_before_bed m ∧ m = 12 :=
sorry

end rachel_minutes_before_bed_l230_230334


namespace deposit_amount_l230_230118

theorem deposit_amount (P : ℝ) (h₀ : 0.1 * P + 720 = P) : 0.1 * P = 80 :=
by
  sorry

end deposit_amount_l230_230118


namespace quadratic_real_roots_condition_l230_230991

theorem quadratic_real_roots_condition (a : ℝ) :
  (∃ x : ℝ, a * x^2 + 4 * x - 1 = 0) ↔ (a ≥ -4 ∧ a ≠ 0) :=
sorry

end quadratic_real_roots_condition_l230_230991


namespace roots_polynomial_sum_l230_230596

theorem roots_polynomial_sum :
  ∀ (p q r : ℂ), (p^3 - 3*p^2 - p + 3 = 0) ∧ (q^3 - 3*q^2 - q + 3 = 0) ∧ (r^3 - 3*r^2 - r + 3 = 0) →
  (1 / (p - 2) + 1 / (q - 2) + 1 / (r - 2) = 1) :=
by
  intros p q r h
  sorry

end roots_polynomial_sum_l230_230596


namespace find_fourth_speed_l230_230243

theorem find_fourth_speed 
  (avg_speed : ℝ)
  (speed1 speed2 speed3 fourth_speed : ℝ)
  (h_avg_speed : avg_speed = 11.52)
  (h_speed1 : speed1 = 6.0)
  (h_speed2 : speed2 = 12.0)
  (h_speed3 : speed3 = 18.0)
  (expected_avg_speed_eq : avg_speed = 4 / ((1 / speed1) + (1 / speed2) + (1 / speed3) + (1 / fourth_speed))) :
  fourth_speed = 2.095 :=
by 
  sorry

end find_fourth_speed_l230_230243


namespace john_money_left_l230_230322

-- Given definitions
def drink_cost (q : ℝ) := q
def small_pizza_cost (q : ℝ) := q
def large_pizza_cost (q : ℝ) := 4 * q
def initial_amount := 50

-- Problem statement
theorem john_money_left (q : ℝ) : initial_amount - (4 * drink_cost q + 2 * small_pizza_cost q + large_pizza_cost q) = 50 - 10 * q :=
by
  sorry

end john_money_left_l230_230322


namespace xiaohui_pe_score_l230_230436

-- Define the conditions
def morning_score : ℝ := 95
def midterm_score : ℝ := 90
def final_score : ℝ := 85

def morning_weight : ℝ := 0.2
def midterm_weight : ℝ := 0.3
def final_weight : ℝ := 0.5

-- The problem is to prove that Xiaohui's physical education score for the semester is 88.5 points.
theorem xiaohui_pe_score :
  morning_score * morning_weight +
  midterm_score * midterm_weight +
  final_score * final_weight = 88.5 :=
by
  sorry

end xiaohui_pe_score_l230_230436


namespace min_value_x_squared_plus_10x_l230_230360

theorem min_value_x_squared_plus_10x : ∃ x : ℝ, (x^2 + 10 * x) = -25 :=
by {
  sorry
}

end min_value_x_squared_plus_10x_l230_230360


namespace Gina_gave_fraction_to_mom_l230_230823

variable (M : ℝ)

theorem Gina_gave_fraction_to_mom :
  (∃ M, M + (1/8 : ℝ) * 400 + (1/5 : ℝ) * 400 + 170 = 400) →
  M / 400 = 1/4 :=
by
  intro h
  sorry

end Gina_gave_fraction_to_mom_l230_230823


namespace water_left_after_four_hours_l230_230566

def initial_water : ℕ := 40
def water_loss_per_hour : ℕ := 2
def water_added_hour3 : ℕ := 1
def water_added_hour4 : ℕ := 3

theorem water_left_after_four_hours :
    initial_water - 2 * 2 - 2 + water_added_hour3 - 2 + water_added_hour4 - 2 = 36 := 
by
    sorry

end water_left_after_four_hours_l230_230566


namespace correct_calculation_l230_230772

theorem correct_calculation (a b : ℝ) : 
  ¬(3 * a + b = 3 * a * b) ∧ 
  ¬(a^2 + a^2 = a^4) ∧ 
  ¬((a - b)^2 = a^2 - b^2) ∧ 
  ((-3 * a)^2 = 9 * a^2) :=
by
  sorry

end correct_calculation_l230_230772


namespace joan_seashells_left_l230_230025

theorem joan_seashells_left (original_seashells : ℕ) (given_seashells : ℕ) (seashells_left : ℕ)
  (h1 : original_seashells = 70) (h2 : given_seashells = 43) : seashells_left = 27 :=
by
  sorry

end joan_seashells_left_l230_230025


namespace comb_8_3_eq_56_l230_230805

theorem comb_8_3_eq_56 : nat.choose 8 3 = 56 := sorry

end comb_8_3_eq_56_l230_230805


namespace value_of_y_at_x_8_l230_230569

theorem value_of_y_at_x_8 (k : ℝ) (x y : ℝ) 
  (hx1 : y = k * x^(1/3)) 
  (hx2 : y = 4 * Real.sqrt 3) 
  (hx3 : x = 64) 
  (hx4 : 8^(1/3) = 2) : 
  (y = 2 * Real.sqrt 3) := 
by 
  sorry

end value_of_y_at_x_8_l230_230569


namespace greatest_value_of_a_l230_230403

theorem greatest_value_of_a (a : ℝ) : a^2 - 12 * a + 32 ≤ 0 → a ≤ 8 :=
by
  sorry

end greatest_value_of_a_l230_230403


namespace value_of_a_minus_b_l230_230451

theorem value_of_a_minus_b (a b : ℝ) 
  (h₁ : (a-4)*(a+4) = 28*a - 112) 
  (h₂ : (b-4)*(b+4) = 28*b - 112) 
  (h₃ : a ≠ b)
  (h₄ : a > b) :
  a - b = 20 :=
sorry

end value_of_a_minus_b_l230_230451


namespace solve_for_x_l230_230770

theorem solve_for_x (x : ℝ) (h : 3034 - 1002 / x = 3029) : x = 200.4 :=
by
  sorry

end solve_for_x_l230_230770


namespace coronavirus_diameter_scientific_notation_l230_230050

theorem coronavirus_diameter_scientific_notation : 
  ∃ (a : ℝ) (n : ℤ), a = 1.1 ∧ n = -7 ∧ 0.00000011 = a * 10^n := by
sorry

end coronavirus_diameter_scientific_notation_l230_230050


namespace negative_half_power_zero_l230_230802

theorem negative_half_power_zero : (- (1 / 2)) ^ 0 = 1 :=
by
  sorry

end negative_half_power_zero_l230_230802


namespace consecutive_page_sum_l230_230057

theorem consecutive_page_sum (n : ℕ) (h : n * (n + 1) = 2156) : n + (n + 1) = 93 :=
sorry

end consecutive_page_sum_l230_230057


namespace range_of_a_l230_230160

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (Real.exp x / x) - a * x

theorem range_of_a (a : ℝ) :
  (∀ (x1 x2 : ℝ), 0 < x1 ∧ 0 < x2 ∧ x2 > x1 → (f x1 a / x2 - f x2 a / x1 < 0)) ↔ a ≤ Real.exp 1 / 2 := sorry

end range_of_a_l230_230160


namespace smallest_positive_real_x_l230_230278

theorem smallest_positive_real_x :
  ∃ (x : ℝ), x > 0 ∧ ⌊x^2⌋ - x * ⌊x⌋ = 8 ∧ x = 89 / 9 :=
by
  sorry

end smallest_positive_real_x_l230_230278


namespace find_a_plus_b_l230_230554

-- Conditions for the lines
def line_l0 (x y : ℝ) : Prop := x - y + 1 = 0
def line_l1 (a : ℝ) (x y : ℝ) : Prop := a * x - 2 * y + 1 = 0
def line_l2 (b : ℝ) (x y : ℝ) : Prop := x + b * y + 3 = 0

-- Perpendicularity condition for l1 to l0
def perpendicular (a : ℝ) : Prop := 1 * a + (-1) * (-2) = 0

-- Parallel condition for l2 to l0
def parallel (b : ℝ) : Prop := 1 * b = (-1) * 1

-- Prove the value of a + b given the conditions
theorem find_a_plus_b (a b : ℝ) 
  (h1 : perpendicular a)
  (h2 : parallel b) : a + b = -3 :=
sorry

end find_a_plus_b_l230_230554


namespace sector_area_is_2_l230_230831

-- Definition of the sector's properties
def sector_perimeter (r : ℝ) (θ : ℝ) : ℝ := r * θ + 2 * r

def sector_area (r : ℝ) (θ : ℝ) : ℝ := 0.5 * r^2 * θ

-- Theorem stating that the area of the sector is 2 cm² given the conditions
theorem sector_area_is_2 (r θ : ℝ) (h1 : sector_perimeter r θ = 6) (h2 : θ = 1) : sector_area r θ = 2 :=
by
  sorry

end sector_area_is_2_l230_230831


namespace sasha_remainder_is_20_l230_230077

theorem sasha_remainder_is_20 (n a b c d : ℤ) (h1 : n = 102 * a + b) (h2 : n = 103 * c + d) (h3 : d = 20 - a) : b = 20 :=
by
  sorry

end sasha_remainder_is_20_l230_230077


namespace lana_extra_flowers_l230_230716

theorem lana_extra_flowers (tulips roses used total extra : ℕ) 
  (h1 : tulips = 36) 
  (h2 : roses = 37) 
  (h3 : used = 70) 
  (h4 : total = tulips + roses) 
  (h5 : extra = total - used) : 
  extra = 3 := 
sorry

end lana_extra_flowers_l230_230716


namespace find_m_n_sum_l230_230455

noncomputable def q : ℚ := 2 / 11

theorem find_m_n_sum {m n : ℕ} (hq : q = m / n) (coprime_mn : Nat.gcd m n = 1) : m + n = 13 := by
  sorry

end find_m_n_sum_l230_230455


namespace find_angle_A_max_perimeter_l230_230318

noncomputable def sin_sq_minus (A B C : ℝ) : ℝ :=
  (Real.sin A) * (Real.sin A) - (Real.sin B) * (Real.sin B) - (Real.sin C) * (Real.sin C)

noncomputable def sin_prod (B C : ℝ) : ℝ :=
  (Real.sin B) * (Real.sin C)

theorem find_angle_A (A B C : ℝ) (h : sin_sq_minus A B C = sin_prod B C) :
  A = 2 * Real.pi / 3 :=
by
  sorry

noncomputable def perimeter (a b c : ℝ) : ℝ :=
  a + b + c

theorem max_perimeter (B C : ℝ) (a : ℝ) (h1 : a = 3) (h2 : B + C = Real.pi / 3) :
  ∃ (b c : ℝ), perimeter a b c = 3 + 2 * Real.sqrt 3 :=
by
  sorry

end find_angle_A_max_perimeter_l230_230318


namespace find_x_eq_7714285714285714_l230_230262

theorem find_x_eq_7714285714285714 (x : ℝ) (hx_pos : 0 < x) (h : floor x * x = 54) : x = 54 / 7 :=
by
  sorry

end find_x_eq_7714285714285714_l230_230262


namespace part1_solution_part2_solution_l230_230184

noncomputable def f (x : ℝ) : ℝ := Real.exp x

theorem part1_solution :
  (∃ x : ℝ, (f x)^2 = f x + 2) ↔ x = Real.log 2 := 
by sorry

theorem part2_solution :
  (∀ x : ℝ, x + b ≤ f x) ↔ b ≤ 1 := 
by sorry

end part1_solution_part2_solution_l230_230184


namespace james_louise_age_sum_l230_230179

variables (J L : ℝ)

theorem james_louise_age_sum
  (h₁ : J = L + 9)
  (h₂ : J + 5 = 3 * (L - 3)) :
  J + L = 32 :=
by
  /- Proof goes here -/
  sorry

end james_louise_age_sum_l230_230179


namespace jackies_free_time_l230_230443

-- Define the conditions
def hours_working : ℕ := 8
def hours_sleeping : ℕ := 8
def hours_exercising : ℕ := 3
def total_hours_in_day : ℕ := 24

-- The statement to be proven
theorem jackies_free_time : total_hours_in_day - (hours_working + hours_sleeping + hours_exercising) = 5 :=
by 
  rw [total_hours_in_day, hours_working, hours_sleeping, hours_exercising]
  -- 24 - (8 + 8 + 3) = 5
  sorry

end jackies_free_time_l230_230443


namespace original_selling_price_l230_230800

theorem original_selling_price (P : ℝ) (h1 : ∀ P, 1.17 * P = 1.10 * P + 42) :
    1.10 * P = 660 := by
  sorry

end original_selling_price_l230_230800


namespace sequence_an_value_l230_230593

theorem sequence_an_value (a : ℕ → ℕ) (S : ℕ → ℕ)
  (hS : ∀ n, 4 * S n = (a n - 1) * (a n + 3))
  (h_pos : ∀ n, 0 < a n)
  (n_nondec : ∀ n, a (n + 1) - a n = 2) :
  a 1005 = 2011 := 
sorry

end sequence_an_value_l230_230593


namespace smallest_number_of_ones_l230_230266

-- Definitions inferred from the problem conditions
def N := (10^100 - 1) / 3
def M_k (k : ℕ) := (10^k - 1) / 9

theorem smallest_number_of_ones (k : ℕ) : M_k k % N = 0 → k = 300 :=
by {
  sorry
}

end smallest_number_of_ones_l230_230266


namespace add_congruence_l230_230426

variable (a b c d m : ℤ)

theorem add_congruence (h₁ : a ≡ b [ZMOD m]) (h₂ : c ≡ d [ZMOD m]) : (a + c) ≡ (b + d) [ZMOD m] :=
sorry

end add_congruence_l230_230426


namespace sequence_ab_sum_l230_230257

theorem sequence_ab_sum (s a b : ℝ) (h1 : 16 * s = 4) (h2 : 1024 * s = a) (h3 : a * s = b) : a + b = 320 := by
  sorry

end sequence_ab_sum_l230_230257


namespace ratio_of_abc_l230_230923

theorem ratio_of_abc {a b c : ℕ} (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0) 
                     (h_ratio : ∃ x : ℕ, a = 2 * x ∧ b = 3 * x ∧ c = 4 * x)
                     (h_mean : (a + b + c) / 3 = 42) : 
  a = 28 := 
sorry

end ratio_of_abc_l230_230923


namespace insects_legs_l230_230310

theorem insects_legs (L N : ℕ) (hL : L = 54) (hN : N = 9) : (L / N = 6) :=
by sorry

end insects_legs_l230_230310


namespace calculate_selling_price_l230_230042

-- Define the conditions
def purchase_price : ℝ := 900
def repair_cost : ℝ := 300
def gain_percentage : ℝ := 0.10

-- Define the total cost
def total_cost : ℝ := purchase_price + repair_cost

-- Define the gain
def gain : ℝ := gain_percentage * total_cost

-- Define the selling price
def selling_price : ℝ := total_cost + gain

-- The theorem to prove
theorem calculate_selling_price : selling_price = 1320 := by
  sorry

end calculate_selling_price_l230_230042


namespace no_positive_integer_solutions_l230_230300

theorem no_positive_integer_solutions (x : ℕ) : ¬(15 < 3 - 2 * x) := by
  sorry

end no_positive_integer_solutions_l230_230300


namespace problems_per_hour_l230_230821

theorem problems_per_hour :
  ∀ (mathProblems spellingProblems totalHours problemsPerHour : ℕ), 
    mathProblems = 36 →
    spellingProblems = 28 →
    totalHours = 8 →
    (mathProblems + spellingProblems) / totalHours = problemsPerHour →
    problemsPerHour = 8 :=
by
  intros
  subst_vars
  sorry

end problems_per_hour_l230_230821


namespace greatest_gcd_f_l230_230968

def f (n : ℕ) : ℕ := 70 + n^2

def g (n : ℕ) : ℕ := Nat.gcd (f n) (f (n + 1))

theorem greatest_gcd_f (n : ℕ) (h : 0 < n) : g n = 281 :=
  sorry

end greatest_gcd_f_l230_230968


namespace simplify_expression_l230_230043

-- Define the algebraic expression
def algebraic_expr (x : ℚ) : ℚ := (3 / (x - 1) - x - 1) * (x - 1) / (x^2 - 4 * x + 4)

theorem simplify_expression : algebraic_expr 0 = 1 :=
by
  -- The proof is skipped using sorry
  sorry

end simplify_expression_l230_230043


namespace cubic_roots_l230_230741

open Real

theorem cubic_roots (x1 x2 x3 : ℝ) (h1 : x1 * x2 = 1)
  (h2 : 3 * x1^3 + 2 * sqrt 3 * x1^2 - 21 * x1 + 6 * sqrt 3 = 0)
  (h3 : 3 * x2^3 + 2 * sqrt 3 * x2^2 - 21 * x2 + 6 * sqrt 3 = 0)
  (h4 : 3 * x3^3 + 2 * sqrt 3 * x3^2 - 21 * x3 + 6 * sqrt 3 = 0) :
  (x1 = sqrt 3 ∧ x2 = sqrt 3 / 3 ∧ x3 = -2 * sqrt 3) ∨
  (x2 = sqrt 3 ∧ x1 = sqrt 3 / 3 ∧ x3 = -2 * sqrt 3) ∨
  (x1 = sqrt 3 ∧ x3 = sqrt 3 / 3 ∧ x2 = -2 * sqrt 3) ∨
  (x3 = sqrt 3 ∧ x1 = sqrt 3 / 3 ∧ x2 = -2 * sqrt 3) ∨
  (x2 = sqrt 3 ∧ x3 = sqrt 3 / 3 ∧ x1 = -2 * sqrt 3) ∨
  (x3 = sqrt 3 ∧ x2 = sqrt 3 / 3 ∧ x1 = -2 * sqrt 3) := 
sorry

end cubic_roots_l230_230741


namespace polynomial_irreducible_l230_230448

theorem polynomial_irreducible (n : ℕ) (hn : n > 1) :
  ¬ ∃ g h : Polynomial ℤ,
    (g.degree ≥ 1 ∧ h.degree ≥ 1 ∧ f = g * h) :=
by
  let f := Polynomial.C 3 + Polynomial.C 5 * Polynomial.X^(n-1) + Polynomial.X^n
  sorry

end polynomial_irreducible_l230_230448


namespace identify_heaviest_and_lightest_l230_230099

theorem identify_heaviest_and_lightest (coins : Fin 10 → ℝ) (h_distinct : Function.Injective coins) :
  ∃ weighings : Fin 13 → (Fin 10 × Fin 10),
  (let outcomes := fun w ℕ => ite (coins (weighings w).fst > coins (weighings w).snd) (weighings w).fst (weighings w).snd,
  max_coin := nat.rec_on 12 (outcomes 0) (λ n max_n, if coins (outcomes (succ n)) > coins max_n then outcomes (succ n) else max_n),
  min_coin := nat.rec_on 12 (outcomes 0) (λ n min_n, if coins (outcomes (succ n)) < coins min_n then outcomes (succ n) else min_n))
  (∃ max_c : Fin 10, ∃ min_c : Fin 10, max_c ≠ min_c ∧ max_c = Some max_coin ∧ min_c = Some min_coin) :=
sorry

end identify_heaviest_and_lightest_l230_230099


namespace total_meals_per_week_l230_230560

-- Definitions of the conditions
def meals_per_day_r1 : ℕ := 20
def meals_per_day_r2 : ℕ := 40
def meals_per_day_r3 : ℕ := 50
def days_per_week : ℕ := 7

-- The proof goal
theorem total_meals_per_week : 
  (meals_per_day_r1 * days_per_week) + 
  (meals_per_day_r2 * days_per_week) + 
  (meals_per_day_r3 * days_per_week) = 770 :=
by
  sorry

end total_meals_per_week_l230_230560


namespace weight_of_a_l230_230779

variables (a b c d e : ℝ)

theorem weight_of_a (h1 : (a + b + c) / 3 = 80)
                    (h2 : (a + b + c + d) / 4 = 82)
                    (h3 : e = d + 3)
                    (h4 : (b + c + d + e) / 4 = 81) :
  a = 95 :=
by
  sorry

end weight_of_a_l230_230779


namespace turtle_ran_while_rabbit_sleeping_l230_230249

-- Define the constants and variables used in the problem
def total_distance : ℕ := 1000
def rabbit_speed_multiple : ℕ := 5
def rabbit_behind_distance : ℕ := 10

-- Define a function that represents the turtle's distance run while the rabbit is sleeping
def turtle_distance_while_rabbit_sleeping (total_distance : ℕ) (rabbit_speed_multiple : ℕ) (rabbit_behind_distance : ℕ) : ℕ :=
  total_distance - total_distance / (rabbit_speed_multiple + 1)

-- Prove that the turtle ran 802 meters while the rabbit was sleeping
theorem turtle_ran_while_rabbit_sleeping :
  turtle_distance_while_rabbit_sleeping total_distance rabbit_speed_multiple rabbit_behind_distance = 802 :=
by
  -- We reserve the proof and focus only on the statement
  sorry

end turtle_ran_while_rabbit_sleeping_l230_230249


namespace general_term_formula_l230_230479

theorem general_term_formula (n : ℕ) (a : ℕ → ℚ) :
  (∀ n, a n = (-1)^n * (n^2)/(2 * n - 1)) :=
sorry

end general_term_formula_l230_230479


namespace total_children_with_cats_l230_230639

variable (D C B : ℕ)
variable (h1 : D = 18)
variable (h2 : B = 6)
variable (h3 : D + C + B = 30)

theorem total_children_with_cats : C + B = 12 := by
  sorry

end total_children_with_cats_l230_230639


namespace sqrt_11_custom_op_l230_230111

noncomputable def sqrt := Real.sqrt

def custom_op (x y : Real) := (x + y) ^ 2 - (x - y) ^ 2

theorem sqrt_11_custom_op : custom_op (sqrt 11) (sqrt 11) = 44 :=
by
  sorry

end sqrt_11_custom_op_l230_230111


namespace james_after_paying_debt_l230_230186

variables (L J A : Real)

-- Define the initial conditions
def total_money : Real := 300
def debt : Real := 25
def total_with_debt : Real := total_money + debt

axiom h1 : J = A + 40
axiom h2 : J + A = total_with_debt

-- Prove that James owns $170 after paying off half of Lucas' debt
theorem james_after_paying_debt (h1 : J = A + 40) (h2 : J + A = total_with_debt) :
  (J - (debt / 2)) = 170 :=
  sorry

end james_after_paying_debt_l230_230186


namespace eugene_total_pencils_l230_230669

-- Define the initial number of pencils Eugene has
def initial_pencils : ℕ := 51

-- Define the number of pencils Joyce gives to Eugene
def pencils_from_joyce : ℕ := 6

-- Define the expected total number of pencils
def expected_total_pencils : ℕ := 57

-- Theorem to prove the total number of pencils Eugene has
theorem eugene_total_pencils : initial_pencils + pencils_from_joyce = expected_total_pencils := 
by sorry

end eugene_total_pencils_l230_230669


namespace smallest_sum_of_4_numbers_l230_230244

noncomputable def relatively_prime (a b : ℕ) : Prop :=
  Nat.gcd a b = 1

noncomputable def not_relatively_prime (a b : ℕ) : Prop :=
  ¬ relatively_prime a b

noncomputable def problem_statement : Prop :=
  ∃ (V1 V2 V3 V4 : ℕ), 
  relatively_prime V1 V3 ∧ 
  relatively_prime V2 V4 ∧ 
  not_relatively_prime V1 V2 ∧ 
  not_relatively_prime V1 V4 ∧ 
  not_relatively_prime V2 V3 ∧ 
  not_relatively_prime V3 V4 ∧ 
  V1 + V2 + V3 + V4 = 60

theorem smallest_sum_of_4_numbers : problem_statement := sorry

end smallest_sum_of_4_numbers_l230_230244


namespace simplify_expression_l230_230192

def expression (x y : ℤ) : ℤ := 
  ((2 * x + y) * (2 * x - y) - (2 * x - 3 * y)^2) / (-2 * y)

theorem simplify_expression {x y : ℤ} (hx : x = 1) (hy : y = -2) :
  expression x y = -16 :=
by 
  -- This proof will involve algebraic manipulation and substitution.
  sorry

end simplify_expression_l230_230192


namespace union_of_M_and_N_l230_230421

noncomputable def U : Set ℝ := {x | -3 ≤ x ∧ x < 2}
noncomputable def M : Set ℝ := {x | -1 < x ∧ x < 1}
noncomputable def compl_U_N : Set ℝ := {x | 0 < x ∧ x < 2}
noncomputable def N : Set ℝ := {x | -3 ≤ x ∧ x ≤ 0}

theorem union_of_M_and_N :
  M ∪ N = {x | -3 ≤ x ∧ x < 1} :=
sorry

end union_of_M_and_N_l230_230421


namespace frank_money_made_l230_230822

theorem frank_money_made
  (spent_on_blades : ℕ)
  (number_of_games : ℕ)
  (cost_per_game : ℕ)
  (total_cost_games := number_of_games * cost_per_game)
  (total_money_made := spent_on_blades + total_cost_games)
  (H1 : spent_on_blades = 11)
  (H2 : number_of_games = 4)
  (H3 : cost_per_game = 2) :
  total_money_made = 19 :=
by
  sorry

end frank_money_made_l230_230822


namespace trig_expression_value_l230_230825

theorem trig_expression_value (θ : ℝ) (h1 : Real.tan (2 * θ) = -2 * Real.sqrt 2)
  (h2 : 2 * θ > Real.pi / 2 ∧ 2 * θ < Real.pi) : 
  (2 * Real.cos θ / 2 ^ 2 - Real.sin θ - 1) / (Real.sqrt 2 * Real.sin (θ + Real.pi / 4)) = 2 * Real.sqrt 2 - 3 :=
by
  sorry

end trig_expression_value_l230_230825


namespace initial_dragon_fruits_remaining_kiwis_l230_230924

variable (h d k : ℕ)    -- h: initial number of cantaloupes, d: initial number of dragon fruits, k: initial number of kiwis
variable (d_rem : ℕ)    -- d_rem: remaining number of dragon fruits after all cantaloupes are used up
variable (k_rem : ℕ)    -- k_rem: remaining number of kiwis after all cantaloupes are used up

axiom condition1 : d = 3 * h + 10
axiom condition2 : k = 2 * d
axiom condition3 : d_rem = 130
axiom condition4 : (d - d_rem) = 2 * h
axiom condition5 : k_rem = k - 10 * h

theorem initial_dragon_fruits (h : ℕ) (d : ℕ) (k : ℕ) (d_rem : ℕ) : 
  3 * h + 10 = d → 
  2 * d = k → 
  d_rem = 130 →
  2 * h + d_rem = d → 
  h = 120 → 
  d = 370 :=
by 
  intros
  sorry

theorem remaining_kiwis (h : ℕ) (d : ℕ) (k : ℕ) (k_rem : ℕ) : 
  3 * h + 10 = d → 
  2 * d = k → 
  h = 120 →
  k_rem = k - 10 * h → 
  k_rem = 140 :=
by 
  intros
  sorry

end initial_dragon_fruits_remaining_kiwis_l230_230924


namespace john_speed_first_part_l230_230588

theorem john_speed_first_part (S : ℝ) (h1 : 2 * S + 3 * 55 = 255) : S = 45 :=
by
  sorry

end john_speed_first_part_l230_230588


namespace a_18_value_l230_230675

variable (a : ℕ → ℚ)

axiom a1 : a 1 = 1
axiom a2 : a 2 = 2
axiom a_rec (n : ℕ) (hn : 2 ≤ n) : 2 * n * a n = (n - 1) * a (n - 1) + (n + 1) * a (n + 1)

theorem a_18_value : a 18 = 26 / 9 :=
sorry

end a_18_value_l230_230675


namespace find_smallest_x_l230_230269

noncomputable def smallest_x : ℝ :=
  min { x : ℝ | 0 < x ∧ (⌊x^2⌋ : ℤ) - (⌊x⌋ : ℤ) * x = 8 }

theorem find_smallest_x :
  smallest_x = 89 / 9 :=
by
  sorry

end find_smallest_x_l230_230269


namespace sasha_remainder_l230_230065

statement:
  theorem sasha_remainder (n a b c d : ℕ) (h1 : n = 102 * a + b) (h2 : n = 103 * c + d) (h3 : a + d = 20) (h4: 0 ≤ b ∧ b ≤ 101) : 
  b = 20 :=
by
  sorry

end sasha_remainder_l230_230065


namespace slices_per_large_pizza_l230_230995

theorem slices_per_large_pizza (total_pizzas : ℕ) (slices_eaten : ℕ) (slices_remaining : ℕ) 
  (H1 : total_pizzas = 2) (H2 : slices_eaten = 7) (H3 : slices_remaining = 9) : 
  (slices_remaining + slices_eaten) / total_pizzas = 8 := 
by
  sorry

end slices_per_large_pizza_l230_230995


namespace probability_of_A_l230_230056

theorem probability_of_A (P : Set α → ℝ) (A B : Set α) :
  P B = 0.40 →
  P (A ∩ B) = 0.15 →
  P Aᶜ ∩ Bᶜ = 0.50 →
  P A = 0.25 :=
by
  intros h1 h2 h3
  sorry

end probability_of_A_l230_230056


namespace monotonicity_of_f_range_of_a_if_f_lt_x_squared_l230_230836

open Real

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := log x - a / x

theorem monotonicity_of_f (a : ℝ) (h : a > 0) :
  ∀ x y : ℝ, 0 < x → x < y → f x a < f y a := by
  sorry

theorem range_of_a_if_f_lt_x_squared (a : ℝ) :
  (∀ x : ℝ, 1 < x → f x a < x^2) → a ≥ -1 := by
  sorry

end monotonicity_of_f_range_of_a_if_f_lt_x_squared_l230_230836


namespace equation_transformation_correct_l230_230498

theorem equation_transformation_correct :
  ∀ (x : ℝ), 
  6 * ((x - 1) / 2 - 1) = 6 * ((3 * x + 1) / 3) → 
  (3 * (x - 1) - 6 = 2 * (3 * x + 1)) :=
by
  intro x
  intro h
  sorry

end equation_transformation_correct_l230_230498


namespace sum_of_valid_primes_eq_222_l230_230405

open Nat

def satisfies_conditions (p : ℕ) : Prop :=
  p % 5 = 1 ∧ p % 7 = 6 ∧ p ≤ 200

def all_valid_primes : List ℕ :=
  (List.range 201).filter (λ p, Prime p ∧ satisfies_conditions p)

theorem sum_of_valid_primes_eq_222 : (all_valid_primes.sum) = 222 :=
by 
  -- We would prove this by evaluating all_valid_primes, checking primality, and computing the sum.
  sorry

end sum_of_valid_primes_eq_222_l230_230405


namespace part1_part2_l230_230408

-- Definition of p: x² + 2x - 8 < 0
def p (x : ℝ) : Prop := x^2 + 2 * x - 8 < 0

-- Definition of q: (x - 1 + m)(x - 1 - m) ≤ 0
def q (x : ℝ) (m : ℝ) : Prop := (x - 1 + m) * (x - 1 - m) ≤ 0

-- Define A as the set of real numbers that satisfy p
def A : Set ℝ := { x | p x }

-- Define B as the set of real numbers that satisfy q when m = 2
def B (m : ℝ) : Set ℝ := { x | q x m }

theorem part1 : A ∩ B 2 = { x | -1 ≤ x ∧ x < 2 } :=
sorry

-- Prove that m ≥ 5 is the range for which p is a sufficient but not necessary condition for q
theorem part2 : ∀ m : ℝ, (∀ x: ℝ, p x → q x m) ∧ (∃ x: ℝ, q x m ∧ ¬p x) ↔ m ≥ 5 :=
sorry

end part1_part2_l230_230408


namespace closest_clock_to_16_is_C_l230_230251

noncomputable def closestTo16InMirror (clock : Char) : Bool :=
  clock = 'C'

theorem closest_clock_to_16_is_C : 
  (closestTo16InMirror 'A' = False) ∧ 
  (closestTo16InMirror 'B' = False) ∧ 
  (closestTo16InMirror 'C' = True) ∧ 
  (closestTo16InMirror 'D' = False) := 
by
  sorry

end closest_clock_to_16_is_C_l230_230251


namespace proof_C_l230_230452

variable {a b c : Type} [LinearOrder a] [LinearOrder b] [LinearOrder c]
variable {y : Type}

-- Definitions for parallel and perpendicular relationships
def parallel (x1 x2 : Type) : Prop := sorry
def perp (x1 x2 : Type) : Prop := sorry

theorem proof_C (a b c : Type) [LinearOrder a] [LinearOrder b] [LinearOrder c] (y : Type):
  (parallel a b ∧ parallel b c → parallel a c) ∧
  (perp a y ∧ perp b y → parallel a b) :=
by
  sorry

end proof_C_l230_230452


namespace problem_3_at_7_hash_4_l230_230009

def oper_at (a b : ℕ) : ℚ := (a * b) / (a + b)
def oper_hash (c d : ℚ) : ℚ := c + d

theorem problem_3_at_7_hash_4 :
  oper_hash (oper_at 3 7) 4 = 61 / 10 := by
  sorry

end problem_3_at_7_hash_4_l230_230009


namespace min_value_of_quadratic_l230_230362

theorem min_value_of_quadratic (x : ℝ) : ∃ m : ℝ, (∀ x, x^2 + 10 * x ≥ m) ∧ m = -25 := by
  sorry

end min_value_of_quadratic_l230_230362


namespace marie_packs_construction_paper_l230_230945

theorem marie_packs_construction_paper (marie_glue_sticks : ℕ) (allison_glue_sticks : ℕ) (total_allison_items : ℕ)
    (glue_sticks_difference : allison_glue_sticks = marie_glue_sticks + 8)
    (marie_glue_sticks_count : marie_glue_sticks = 15)
    (total_items_allison : total_allison_items = 28)
    (marie_construction_paper_multiplier : ℕ)
    (construction_paper_ratio : marie_construction_paper_multiplier = 6) : 
    ∃ (marie_construction_paper_packs : ℕ), marie_construction_paper_packs = 30 := 
by
  sorry

end marie_packs_construction_paper_l230_230945


namespace geometric_sequence_a_5_l230_230410

noncomputable def a_n : ℕ → ℝ := sorry

theorem geometric_sequence_a_5 :
  (∀ n : ℕ, ∃ r : ℝ, a_n (n + 1) = r * a_n n) →  -- geometric sequence property
  (∃ x₁ x₂ : ℝ, x₁ + x₂ = -7 ∧ x₁ * x₂ = 9 ∧ a_n 3 = x₁ ∧ a_n 7 = x₂) →  -- roots of the quadratic equation and their assignments
  a_n 5 = -3 := sorry

end geometric_sequence_a_5_l230_230410


namespace total_sheets_of_paper_l230_230726

theorem total_sheets_of_paper (num_classes : ℕ) (students_per_class : ℕ) (sheets_per_student : ℕ)
  (h1 : num_classes = 4) (h2 : students_per_class = 20) (h3 : sheets_per_student = 5) :
  (num_classes * students_per_class * sheets_per_student) = 400 :=
by
  sorry

end total_sheets_of_paper_l230_230726


namespace conveyor_belt_sampling_l230_230253

noncomputable def sampling_method (interval : ℕ) (total_items : ℕ) : String :=
  if interval = 5 ∧ total_items > 0 then "systematic sampling" else "unknown"

theorem conveyor_belt_sampling :
  ∀ (interval : ℕ) (total_items : ℕ),
  interval = 5 ∧ total_items > 0 →
  sampling_method interval total_items = "systematic sampling" :=
sorry

end conveyor_belt_sampling_l230_230253


namespace find_common_difference_l230_230022

variable {a : ℕ → ℤ}  -- Define a sequence indexed by natural numbers, returning integers
variable (d : ℤ)  -- Define the common difference as an integer

-- The conditions: sequence is arithmetic, a_2 = 14, a_5 = 5
axiom arithmetic_sequence (n : ℕ) : a n = a 0 + n * d
axiom a_2_eq_14 : a 2 = 14
axiom a_5_eq_5 : a 5 = 5

-- The proof statement
theorem find_common_difference : d = -3 :=
by sorry

end find_common_difference_l230_230022


namespace islander_C_response_l230_230332

-- Define the types and assumptions
variables {Person : Type} (is_knight : Person → Prop) (is_liar : Person → Prop)
variables (A B C : Person)

-- Conditions from the problem
axiom A_statement : (is_liar A) ↔ (is_knight B = false ∧ is_knight C = false)
axiom B_statement : (is_knight B) ↔ (is_knight A ↔ ¬ is_knight C)

-- Conclusion we want to prove
theorem islander_C_response : is_knight C → (is_knight A ↔ ¬ is_knight C) := sorry

end islander_C_response_l230_230332


namespace alcohol_by_volume_l230_230195

/-- Solution x is 10% alcohol by volume and is 50 ml.
    Solution y is 30% alcohol by volume and is 150 ml.
    We must prove the final solution is 25% alcohol by volume. -/
theorem alcohol_by_volume (vol_x vol_y : ℕ) (conc_x conc_y : ℕ) (vol_mix : ℕ) (conc_mix : ℕ) :
  vol_x = 50 →
  conc_x = 10 →
  vol_y = 150 →
  conc_y = 30 →
  vol_mix = vol_x + vol_y →
  conc_mix = 100 * (vol_x * conc_x + vol_y * conc_y) / vol_mix →
  conc_mix = 25 :=
by
  intros h1 h2 h3 h4 h5 h_cons
  sorry

end alcohol_by_volume_l230_230195


namespace minimum_value_of_quadratic_expression_l230_230358

theorem minimum_value_of_quadratic_expression : ∃ x ∈ ℝ, ∀ y ∈ ℝ, x^2 + 10 * x ≤ y^2 + 10 * y := by
  sorry

end minimum_value_of_quadratic_expression_l230_230358


namespace part_1_part_2_l230_230312

-- Define the triangle and the given condition
variables {α β γ : ℝ}
axiom triangle_ABC : 0 < α ∧ 0 < β ∧ 0 < γ ∧ α + β + γ = π
axiom sin_identity : sin(α)^2 - sin(β)^2 - sin(γ)^2 = sin(β) * sin(γ)
axiom BC_length : (sin(γ) / sin(α)) * BC = 3

-- State the main theorem parts separately
theorem part_1 (h_triangle: 0 < α ∧ 0 < β ∧ 0 < γ ∧ α + β + γ = π) (h_sin_identity: sin(α)^2 - sin(β)^2 - sin(γ)^2 = sin(β) * sin(γ)) :
  α = 2 * π / 3 :=
sorry

theorem part_2 (h_triangle: 0 < α ∧ 0 < β ∧ 0 < γ ∧ α + β + γ = π) (h_α: α = 2 * π / 3) (h_BC_length: BC = 3) :
  let b := (2 * sqrt(3) * sin(π/6 - β)),
      c := (2 * sqrt(3) * sin(π/6 + β)) in
  (3 + 2 * sqrt(3)) :=
sorry

end part_1_part_2_l230_230312


namespace miles_remaining_l230_230113

theorem miles_remaining (total_miles driven_miles : ℕ) (h1 : total_miles = 1200) (h2 : driven_miles = 768) :
    total_miles - driven_miles = 432 := by
  sorry

end miles_remaining_l230_230113


namespace sum_possible_values_for_k_l230_230707

theorem sum_possible_values_for_k :
  ∃ (k_vals : Finset ℕ), (∀ j k : ℕ, 0 < j → 0 < k → (1 / j + 1 / k = 1 / 4) → k ∈ k_vals) ∧ 
    k_vals.sum id = 51 :=
by 
  sorry

end sum_possible_values_for_k_l230_230707


namespace inequality_range_l230_230980

theorem inequality_range (a : ℝ) : (∀ x : ℝ, |x + 2| + |x - 3| > a) → a < 5 :=
  sorry

end inequality_range_l230_230980


namespace calc_product_l230_230950

def x : ℝ := 150.15
def y : ℝ := 12.01
def z : ℝ := 1500.15
def w : ℝ := 12

theorem calc_product :
  x * y * z * w = 32467532.8227 :=
by
  sorry

end calc_product_l230_230950


namespace initial_sale_price_percent_l230_230514

theorem initial_sale_price_percent (P S : ℝ) (h1 : S * 0.90 = 0.63 * P) :
  S = 0.70 * P :=
by
  sorry

end initial_sale_price_percent_l230_230514


namespace no_solution_for_b_a_divides_a_b_minus_1_l230_230259

theorem no_solution_for_b_a_divides_a_b_minus_1 :
  ¬ (∃ a b : ℕ, 1 ≤ a ∧ 1 ≤ b ∧ b^a ∣ a^b - 1) :=
by
  sorry

end no_solution_for_b_a_divides_a_b_minus_1_l230_230259


namespace find_other_number_l230_230774

theorem find_other_number
  (B : ℕ)
  (hcf_condition : Nat.gcd 24 B = 12)
  (lcm_condition : Nat.lcm 24 B = 396) :
  B = 198 :=
by
  sorry

end find_other_number_l230_230774


namespace units_digit_of_k_squared_plus_2_to_the_k_l230_230326

def k : ℕ := 2021^2 + 2^2021 + 3

theorem units_digit_of_k_squared_plus_2_to_the_k :
    (k^2 + 2^k) % 10 = 0 :=
by
    sorry

end units_digit_of_k_squared_plus_2_to_the_k_l230_230326


namespace percentage_increase_painting_l230_230602

/-
Problem:
Given:
1. The original cost of jewelry is $30 each.
2. The original cost of paintings is $100 each.
3. The new cost of jewelry is $40 each.
4. The new cost of paintings is $100 + ($100 * P / 100).
5. A buyer purchased 2 pieces of jewelry and 5 paintings for $680.

Prove:
The percentage increase in the cost of each painting (P) is 20%.
-/

theorem percentage_increase_painting (P : ℝ) :
  let jewelry_price := 30
  let painting_price := 100
  let new_jewelry_price := 40
  let new_painting_price := 100 * (1 + P / 100)
  let total_cost := 2 * new_jewelry_price + 5 * new_painting_price
  total_cost = 680 → P = 20 := by
sorry

end percentage_increase_painting_l230_230602


namespace smallest_positive_real_number_l230_230272

noncomputable def smallest_x : ℝ :=
  Inf {x : ℝ | x > 0 ∧ (floor (x^2) - x * floor x = 8)}

theorem smallest_positive_real_number :
  smallest_x = 89 / 9 :=
by 
  sorry

end smallest_positive_real_number_l230_230272


namespace smallest_multiple_of_45_and_75_not_20_l230_230908

-- Definitions of the conditions
def isMultipleOf (a b : ℕ) : Prop := ∃ k : ℕ, a = k * b
def notMultipleOf (a b : ℕ) : Prop := ¬ (isMultipleOf a b)

-- The proof statement
theorem smallest_multiple_of_45_and_75_not_20 :
  ∃ N : ℕ, isMultipleOf N 45 ∧ isMultipleOf N 75 ∧ notMultipleOf N 20 ∧ N = 225 :=
by
  -- sorry is used to indicate that the proof needs to be filled here
  sorry

end smallest_multiple_of_45_and_75_not_20_l230_230908


namespace triangle_properties_l230_230653

-- Definitions of sides of the triangle
def a : ℕ := 15
def b : ℕ := 11
def c : ℕ := 18

-- Definition of the triangle inequality theorem in the context
def triangle_inequality (x y z : ℕ) : Prop :=
  x + y > z ∧ x + z > y ∧ y + z > x

-- Perimeter calculation
def perimeter (x y z : ℕ) : ℕ :=
  x + y + z

-- Stating the proof problem
theorem triangle_properties : triangle_inequality a b c ∧ perimeter a b c = 44 :=
by
  -- Start the process for the actual proof that will be filled out
  sorry

end triangle_properties_l230_230653


namespace value_of_y_at_x_8_l230_230570

theorem value_of_y_at_x_8 (k : ℝ) (x y : ℝ) 
  (hx1 : y = k * x^(1/3)) 
  (hx2 : y = 4 * Real.sqrt 3) 
  (hx3 : x = 64) 
  (hx4 : 8^(1/3) = 2) : 
  (y = 2 * Real.sqrt 3) := 
by 
  sorry

end value_of_y_at_x_8_l230_230570


namespace range_of_a_l230_230000

theorem range_of_a (a : ℝ) : (∃ x : ℝ, |x + 2| + |x - 3| ≤ a) → a ≥ 5 :=
sorry

end range_of_a_l230_230000


namespace inequality_holds_l230_230938

theorem inequality_holds (a : ℝ) : (∀ x : ℝ, a * x ^ 2 - 2 * a * x + 1 > 0) ↔ (0 < a ∧ a < 1) :=
sorry

end inequality_holds_l230_230938


namespace identify_heaviest_and_lightest_in_13_weighings_l230_230097

-- Definitions based on the conditions
def coins := Finset ℕ
def weighs_with_balance_scale (c1 c2: coins) : Prop := true  -- Placeholder for weighing functionality

/-- There are 10 coins, each with a distinct weight. -/
def ten_distinct_coins (coin_set : coins) : Prop :=
  coin_set.card = 10 ∧ (∀ c1 c2 ∈ coin_set, c1 ≠ c2 → weighs_with_balance_scale c1 c2)

-- Theorem statement
theorem identify_heaviest_and_lightest_in_13_weighings 
  (coin_set : coins)
  (hc: ten_distinct_coins coin_set):
  ∃ (heaviest lightest : coins), 
    weighs_with_balance_scale heaviest coin_set ∧ weighs_with_balance_scale coin_set lightest ∧ 
    -- Assuming weighs_with_balance_scale keeps track of number of weighings
    weights_used coin_set = 13 :=
sorry

end identify_heaviest_and_lightest_in_13_weighings_l230_230097


namespace balls_into_boxes_all_ways_balls_into_boxes_one_empty_l230_230885

/-- There are 4 different balls and 4 different boxes. -/
def balls : ℕ := 4
def boxes : ℕ := 4

/-- The number of ways to put 4 different balls into 4 different boxes is 256. -/
theorem balls_into_boxes_all_ways : (balls ^ boxes) = 256 := by
  sorry

/-- The number of ways to put 4 different balls into 4 different boxes such that exactly one box remains empty is 144. -/
theorem balls_into_boxes_one_empty : (boxes.choose 1 * (balls ^ (boxes - 1))) = 144 := by
  sorry

end balls_into_boxes_all_ways_balls_into_boxes_one_empty_l230_230885


namespace max_value_in_interval_l230_230964

theorem max_value_in_interval :
  ∀ (x : ℝ), -2 ≤ x ∧ x ≤ 2 → x^4 - 2 * x^2 + 5 ≤ 13 :=
by
  sorry

end max_value_in_interval_l230_230964


namespace part1_coordinates_of_P_if_AB_perp_PB_part2_coordinates_of_P_area_ABP_10_l230_230021

-- Part (Ⅰ)
theorem part1_coordinates_of_P_if_AB_perp_PB :
  ∃ P : ℝ × ℝ, P.2 = 0 ∧ (P = (7, 0)) :=
by
  sorry

-- Part (Ⅱ)
theorem part2_coordinates_of_P_area_ABP_10 :
  ∃ P : ℝ × ℝ, P.2 = 0 ∧ (P = (9, 0) ∨ P = (-11, 0)) :=
by
  sorry

end part1_coordinates_of_P_if_AB_perp_PB_part2_coordinates_of_P_area_ABP_10_l230_230021


namespace least_perimeter_of_triangle_l230_230756

theorem least_perimeter_of_triangle (a b : ℕ) (a_eq : a = 33) (b_eq : b = 42) (c : ℕ) (h1 : c + a > b) (h2 : c + b > a) (h3 : a + b > c) : a + b + c = 85 :=
sorry

end least_perimeter_of_triangle_l230_230756


namespace h_at_2_l230_230697

noncomputable def h (x : ℝ) : ℝ := 
(x + 2) * (x - 1) * (x + 4) * (x - 3) - x^2

theorem h_at_2 : 
  h (-2) = -4 ∧ h (1) = -1 ∧ h (-4) = -16 ∧ h (3) = -9 → h (2) = -28 := 
by
  intro H
  sorry

end h_at_2_l230_230697


namespace find_added_value_l230_230701

theorem find_added_value (N : ℕ) (V : ℕ) (H : N = 1280) :
  ((N + V) / 125 = 7392 / 462) → V = 720 :=
by 
  sorry

end find_added_value_l230_230701


namespace gcd_of_256_180_600_l230_230633

theorem gcd_of_256_180_600 : Nat.gcd (Nat.gcd 256 180) 600 = 12 :=
by
  -- The proof would be placed here
  sorry

end gcd_of_256_180_600_l230_230633


namespace sum_of_corners_9x9_grid_l230_230748

theorem sum_of_corners_9x9_grid : 
  let topLeft := 1
  let topRight := 9
  let bottomLeft := 73
  let bottomRight := 81
  topLeft + topRight + bottomLeft + bottomRight = 164 :=
by {
  let topLeft := 1
  let topRight := 9
  let bottomLeft := 73
  let bottomRight := 81
  show topLeft + topRight + bottomLeft + bottomRight = 164
  sorry
}

end sum_of_corners_9x9_grid_l230_230748


namespace line_through_point_p_intersects_ellipse_l230_230682

open Real

theorem line_through_point_p_intersects_ellipse
    (a b : ℝ)
    (h1 : 0 < a^2 + b^2)
    (h2 : a^2 + b^2 < 3) :
    ∃ (m c : ℝ), {p : ℝ × ℝ | p.1 = m * p.2 + c} ∩ {p : ℝ × ℝ | (p.1)^2 / 4 + (p.2)^2 / 3 = 1} = 
    {p1, p2 : ℝ × ℝ | {p1, p2} ≠ ∅ ∧ p1 ≠ p2} := 
by
  sorry

end line_through_point_p_intersects_ellipse_l230_230682


namespace greater_savings_on_hat_l230_230648

theorem greater_savings_on_hat (savings_shoes spent_shoes savings_hat sale_price_hat : ℝ) 
  (h1 : savings_shoes = 3.75)
  (h2 : spent_shoes = 42.25)
  (h3 : savings_hat = 1.80)
  (h4 : sale_price_hat = 18.20) :
  ((savings_hat / (sale_price_hat + savings_hat)) * 100) > ((savings_shoes / (spent_shoes + savings_shoes)) * 100) :=
by
  sorry

end greater_savings_on_hat_l230_230648


namespace factor_difference_of_squares_l230_230258

theorem factor_difference_of_squares (x : ℝ) : x^2 - 81 = (x - 9) * (x + 9) := 
by
  sorry

end factor_difference_of_squares_l230_230258


namespace sheep_ratio_l230_230460

theorem sheep_ratio (s : ℕ) (h1 : s = 400) (h2 : s / 4 + 150 = s - s / 4) : (s / 4 * 3 - 150) / 150 = 1 :=
by {
  sorry
}

end sheep_ratio_l230_230460


namespace goat_cow_difference_l230_230423

-- Given the number of pigs (P), cows (C), and goats (G) on a farm
variables (P C G : ℕ)

-- Conditions:
def pig_count := P = 10
def cow_count_relationship := C = 2 * P - 3
def total_animals := P + C + G = 50

-- Theorem: The difference between the number of goats and cows
theorem goat_cow_difference (h1 : pig_count P)
                           (h2 : cow_count_relationship P C)
                           (h3 : total_animals P C G) :
  G - C = 6 := 
  sorry

end goat_cow_difference_l230_230423


namespace find_other_endpoint_l230_230480

theorem find_other_endpoint 
    (Mx My : ℝ) (x1 y1 : ℝ) 
    (hx_Mx : Mx = 3) (hy_My : My = 1)
    (hx1 : x1 = 7) (hy1 : y1 = -3) : 
    ∃ (x2 y2 : ℝ), Mx = (x1 + x2) / 2 ∧ My = (y1 + y2) / 2 ∧ x2 = -1 ∧ y2 = 5 :=
by
    sorry

end find_other_endpoint_l230_230480


namespace total_amount_proof_l230_230939

noncomputable def total_amount (x_share y_share z_share : ℝ) : ℝ :=
  x_share + y_share + z_share

theorem total_amount_proof (x_ratio y_ratio z_ratio : ℝ) (y_share : ℝ) 
  (h1 : y_ratio = 0.45) (h2 : z_ratio = 0.50) (h3 : y_share = 54) 
  : total_amount (y_share / y_ratio) y_share (z_ratio * (y_share / y_ratio)) = 234 :=
by
  sorry

end total_amount_proof_l230_230939


namespace largest_whole_number_for_inequality_l230_230903

theorem largest_whole_number_for_inequality :
  ∀ n : ℕ, (1 : ℝ) / 4 + (n : ℝ) / 6 < 3 / 2 → n ≤ 7 :=
by
  admit  -- skip the proof

end largest_whole_number_for_inequality_l230_230903


namespace solution_is_unique_zero_l230_230350

theorem solution_is_unique_zero : ∀ (x y z : ℤ), x^3 + 2 * y^3 = 4 * z^3 → x = 0 ∧ y = 0 ∧ z = 0 :=
by
  intros x y z h
  sorry

end solution_is_unique_zero_l230_230350


namespace manuscript_page_count_l230_230058

-- Define the main statement
theorem manuscript_page_count
  (P : ℕ)
  (cost_per_page : ℕ := 10)
  (rev1_pages : ℕ := 30)
  (rev2_pages : ℕ := 20)
  (total_cost : ℕ := 1350)
  (cost_rev1 : ℕ := 15)
  (cost_rev2 : ℕ := 20) 
  (remaining_pages_cost : ℕ := 10 * (P - (rev1_pages + rev2_pages))) :
  (remaining_pages_cost + rev1_pages * cost_rev1 + rev2_pages * cost_rev2 = total_cost)
  → P = 100 :=
by
  sorry

end manuscript_page_count_l230_230058


namespace jackie_free_time_correct_l230_230442

noncomputable def jackie_free_time : ℕ :=
  let total_hours_in_a_day := 24
  let hours_working := 8
  let hours_exercising := 3
  let hours_sleeping := 8
  let total_activity_hours := hours_working + hours_exercising + hours_sleeping
  total_hours_in_a_day - total_activity_hours

theorem jackie_free_time_correct : jackie_free_time = 5 := by
  sorry

end jackie_free_time_correct_l230_230442


namespace P_Q_sum_l230_230425

noncomputable def find_P_Q_sum (P Q : ℚ) : Prop :=
  ∀ x : ℚ, (x^2 + 3 * x + 7) * (x^2 + (51/7) * x - 2) = x^4 + P * x^3 + Q * x^2 + 45 * x - 14

theorem P_Q_sum :
  ∃ P Q : ℚ, find_P_Q_sum P Q ∧ (P + Q = 260 / 7) :=
by
  sorry

end P_Q_sum_l230_230425


namespace sasha_remainder_l230_230069

theorem sasha_remainder (n a b c d : ℕ) (h1 : n = 102 * a + b) (h2 : n = 103 * c + d) 
  (h3 : a + d = 20) (h_b_range : 0 ≤ b ∧ b ≤ 101) (h_d_range : 0 ≤ d ∧ d ≤ 102) : b = 20 := 
sorry

end sasha_remainder_l230_230069


namespace range_a_minus_b_l230_230416

theorem range_a_minus_b (a b : ℝ) (h_pos : a > 0) (h_roots : ∃ x₁ x₂ : ℝ, (ax² + bx - 1 = 0) ∧ x₁ ≠ x₂) (h_interval : ∃ x : ℝ, x ∈ (1, 2) ∧ (ax² + bx - 1 = 0)) :
  -1 < a - b ∧ a - b < 1 := sorry

end range_a_minus_b_l230_230416


namespace increasing_interval_sin_cos_l230_230366

theorem increasing_interval_sin_cos : 
  ∀ x, x ∈ Icc (-π/2) 0 → 0 < real.sin' x ∧ 0 < real.cos' x := 
by 
  sorry

end increasing_interval_sin_cos_l230_230366


namespace quadratic_has_real_roots_range_l230_230434

-- Lean 4 statement

theorem quadratic_has_real_roots_range (m : ℝ) :
  (∀ x : ℝ, (m - 3) * x^2 + 4 * x + 1 = 0) → m ≤ 7 ∧ m ≠ 3 :=
by
  sorry

end quadratic_has_real_roots_range_l230_230434


namespace arithmetic_sequence_sum_l230_230309

variable (a : ℕ → ℝ)

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (d : ℝ), ∀ (n : ℕ), a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h_arith : is_arithmetic_sequence a) 
  (h_a3 : a 3 = 4) (h_a101 : a 101 = 36) : 
  a 9 + a 52 + a 95 = 60 :=
sorry

end arithmetic_sequence_sum_l230_230309


namespace evaluate_expression_l230_230395

lemma pow_mod_four_cycle (n : ℕ) : (n % 4) = 1 → (i : ℂ)^n = i :=
by sorry

lemma pow_mod_four_cycle2 (n : ℕ) : (n % 4) = 2 → (i : ℂ)^n = -1 :=
by sorry

lemma pow_mod_four_cycle3 (n : ℕ) : (n % 4) = 3 → (i : ℂ)^n = -i :=
by sorry

lemma pow_mod_four_cycle4 (n : ℕ) : (n % 4) = 0 → (i : ℂ)^n = 1 :=
by sorry

theorem evaluate_expression : 
  (i : ℂ)^(2021) + (i : ℂ)^(2022) + (i : ℂ)^(2023) + (i : ℂ)^(2024) = 0 :=
by sorry

end evaluate_expression_l230_230395


namespace original_square_perimeter_l230_230246

theorem original_square_perimeter (x : ℝ) 
  (h1 : ∀ r, r = x ∨ r = 4 * x) 
  (h2 : 28 * x = 56) : 
  4 * (4 * x) = 32 :=
by
  -- We don't need to consider the proof as per instructions.
  sorry

end original_square_perimeter_l230_230246


namespace inequality_and_equality_l230_230739

variables {x y z : ℝ}

theorem inequality_and_equality (x y z : ℝ) :
  (x^2 + y^4 + z^6 >= x * y^2 + y^2 * z^3 + x * z^3) ∧ (x^2 + y^4 + z^6 = x * y^2 + y^2 * z^3 + x * z^3 ↔ x = y^2 ∧ y^2 = z^3) :=
by sorry

end inequality_and_equality_l230_230739


namespace samanta_s_eggs_left_l230_230040

def total_eggs : ℕ := 30
def cost_per_crate_dollars : ℕ := 5
def cost_per_crate_cents : ℕ := cost_per_crate_dollars * 100
def sell_price_per_egg_cents : ℕ := 20

theorem samanta_s_eggs_left
  (total_eggs : ℕ) (cost_per_crate_dollars : ℕ) (sell_price_per_egg_cents : ℕ) 
  (cost_per_crate_cents = cost_per_crate_dollars * 100) : 
  total_eggs - (cost_per_crate_cents / sell_price_per_egg_cents) = 5 :=
by sorry

end samanta_s_eggs_left_l230_230040


namespace Jacqueline_gave_Jane_l230_230711

def total_fruits (plums guavas apples : ℕ) : ℕ :=
  plums + guavas + apples

def fruits_given_to_Jane (initial left : ℕ) : ℕ :=
  initial - left

theorem Jacqueline_gave_Jane :
  let plums := 16
  let guavas := 18
  let apples := 21
  let left := 15
  let initial := total_fruits plums guavas apples
  fruits_given_to_Jane initial left = 40 :=
by
  sorry

end Jacqueline_gave_Jane_l230_230711


namespace find_root_of_quadratic_equation_l230_230966

theorem find_root_of_quadratic_equation
  (a b c : ℝ)
  (h1 : 3 * a * (2 * b - 3 * c) ≠ 0)
  (h2 : 2 * b * (3 * c - 2 * a) ≠ 0)
  (h3 : 5 * c * (2 * a - 3 * b) ≠ 0)
  (r : ℝ)
  (h_roots : (r = -2 * b * (3 * c - 2 * a) / (9 * a * (2 * b - 3 * c))) ∨ (r = (-2 * b * (3 * c - 2 * a) / (9 * a * (2 * b - 3 * c))) * 2)) :
  r = -2 * b * (3 * c - 2 * a) / (9 * a * (2 * b - 3 * c)) :=
by
  sorry

end find_root_of_quadratic_equation_l230_230966


namespace area_of_ABC_l230_230392

def point : Type := ℝ × ℝ

def area_of_triangle (A B C : point) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem area_of_ABC : area_of_triangle (0, 0) (1, 0) (0, 1) = 0.5 :=
by
  sorry

end area_of_ABC_l230_230392


namespace solvable_consecutive_integers_solvable_consecutive_even_integers_not_solvable_consecutive_odd_integers_l230_230960

-- Definitions only directly appearing in the conditions problem
def consecutive_integers (x y z : ℤ) : Prop := x = y - 1 ∧ z = y + 1
def consecutive_even_integers (x y z : ℤ) : Prop := x = y - 2 ∧ z = y + 2 ∧ y % 2 = 0
def consecutive_odd_integers (x y z : ℤ) : Prop := x = y - 2 ∧ z = y + 2 ∧ y % 2 = 1

-- Problem Statements
theorem solvable_consecutive_integers : ∃ (x y z : ℤ), x + y + z = 72 ∧ consecutive_integers x y z :=
sorry

theorem solvable_consecutive_even_integers : ∃ (x y z : ℤ), x + y + z = 72 ∧ consecutive_even_integers x y z :=
sorry

theorem not_solvable_consecutive_odd_integers : ¬ ∃ (x y z : ℤ), x + y + z = 72 ∧ consecutive_odd_integers x y z :=
sorry

end solvable_consecutive_integers_solvable_consecutive_even_integers_not_solvable_consecutive_odd_integers_l230_230960


namespace least_positive_integer_l230_230963
  
theorem least_positive_integer 
  (x : ℕ) (d n : ℕ) (p : ℕ) 
  (h_eq : x = 10^p * d + n) 
  (h_ratio : n = x / 17) 
  (h_cond1 : 1 ≤ d) 
  (h_cond2 : d ≤ 9)
  (h_nonzero : n > 0) : 
  x = 10625 :=
by
  sorry

end least_positive_integer_l230_230963


namespace sasha_remainder_l230_230084

theorem sasha_remainder (n a b c d : ℕ) 
  (h1 : n = 102 * a + b) 
  (h2 : n = 103 * c + d) 
  (h3 : a + d = 20)
  (hb : 0 ≤ b ∧ b ≤ 101) : 
  b = 20 :=
sorry

end sasha_remainder_l230_230084


namespace find_number_l230_230989

theorem find_number (N : ℝ) (h1 : (3 / 10) * N = 64.8) : N = 216 ∧ (1 / 3) * (1 / 4) * N = 18 := 
by 
  sorry

end find_number_l230_230989


namespace polynomial_root_conditions_l230_230046

theorem polynomial_root_conditions (a b : ℤ) (h₁ : a ≠ 0) (h₂ : b ≠ 0)
  (h₃ : ∃ r s : ℤ, (x^3 + a * x^2 + b * x + 16 * a) = (x - r)^2 * (x - s) ∧ r * r * s = -16 * a ∧ (r^2 + 2 * r * s) = b) : 
  |a * b| = 5832 :=
sorry

end polynomial_root_conditions_l230_230046


namespace smallest_positive_k_l230_230215

theorem smallest_positive_k (k a n : ℕ) (h_pos : k > 0) (h_cond : 3^3 + 4^3 + 5^3 = 216) (h_eq : k * 216 = a^n) (h_n : n > 1) : k = 1 :=
by {
    sorry
}

end smallest_positive_k_l230_230215


namespace quadrilateral_probability_l230_230657

def total_shapes : ℕ := 6
def quadrilateral_shapes : ℕ := 3

theorem quadrilateral_probability : (quadrilateral_shapes : ℚ) / (total_shapes : ℚ) = 1 / 2 :=
by
  sorry

end quadrilateral_probability_l230_230657


namespace find_interest_rate_l230_230671

-- Defining the conditions
def P : ℝ := 5000
def A : ℝ := 5302.98
def t : ℝ := 1.5
def n : ℕ := 2

-- Statement of the problem in Lean 4
theorem find_interest_rate (P A t : ℝ) (n : ℕ) (hP : P = 5000) (hA : A = 5302.98) (ht : t = 1.5) (hn : n = 2) : 
  ∃ r : ℝ, r * 100 = 3.96 :=
sorry

end find_interest_rate_l230_230671


namespace value_of_M_l230_230061

theorem value_of_M : 
  ∃ (M : ℚ), 
  let a₁₁ := 25,
      a₄₁ := 16, a₅₁ := 20,
      a₄₂ := -20 in
  ∀ a b c d : ℚ,
  (20 - 16 = a) ∧
  (16 - a = b) ∧
  (b - a = c) ∧
  (a₁₁ + -5 * (-17 / 3) = d) ∧
  ([-20 - d] / 4 = -115 / 6) ∧
  (d - 115 / 6 = M) 
  → M = 37.5 := 
by
  sorry

end value_of_M_l230_230061


namespace range_of_a_l230_230305

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, 1 < x ∧ x < 4 ∧ 2 * x^2 - 8 * x - 4 - a > 0) → a < -4 :=
by
  sorry

end range_of_a_l230_230305


namespace equivalence_of_expressions_l230_230524

-- Define the expression on the left-hand side
def lhs := (real.sqrt ((real.sqrt 5) ^ 5)) ^ 6

-- Define the expression on the right-hand side
noncomputable def rhs := 78125 * real.sqrt 5

-- The theorem to prove
theorem equivalence_of_expressions : lhs = rhs :=
by
  sorry

end equivalence_of_expressions_l230_230524


namespace sin_neg_three_halves_pi_l230_230539

theorem sin_neg_three_halves_pi : Real.sin (-3 * Real.pi / 2) = 1 := sorry

end sin_neg_three_halves_pi_l230_230539


namespace height_after_16_minutes_l230_230785

noncomputable def ferris_wheel_height (t : ℝ) : ℝ :=
  8 * Real.sin ((Real.pi / 6) * t - Real.pi / 2) + 10

theorem height_after_16_minutes : ferris_wheel_height 16 = 6 := by
  sorry

end height_after_16_minutes_l230_230785


namespace set_intersection_example_l230_230411

theorem set_intersection_example (A : Set ℕ) (B : Set ℕ) (hA : A = {1, 3, 5}) (hB : B = {3, 4}) :
  A ∩ B = {3} :=
by
  sorry

end set_intersection_example_l230_230411


namespace sarah_amount_l230_230801

theorem sarah_amount:
  ∀ (X : ℕ), (X + (X + 50) = 300) → X = 125 := by
  sorry

end sarah_amount_l230_230801


namespace donna_babysitting_hours_l230_230667

theorem donna_babysitting_hours 
  (total_earnings : ℝ)
  (dog_walking_hours : ℝ)
  (dog_walking_rate : ℝ)
  (dog_walking_days : ℝ)
  (card_shop_hours : ℝ)
  (card_shop_rate : ℝ)
  (card_shop_days : ℝ)
  (babysitting_rate : ℝ)
  (days : ℝ)
  (total_dog_walking_earnings : ℝ := dog_walking_hours * dog_walking_rate * dog_walking_days)
  (total_card_shop_earnings : ℝ := card_shop_hours * card_shop_rate * card_shop_days)
  (total_earnings_dog_card : ℝ := total_dog_walking_earnings + total_card_shop_earnings)
  (babysitting_hours : ℝ := (total_earnings - total_earnings_dog_card) / babysitting_rate) :
  total_earnings = 305 → dog_walking_hours = 2 → dog_walking_rate = 10 → dog_walking_days = 5 →
  card_shop_hours = 2 → card_shop_rate = 12.5 → card_shop_days = 5 →
  babysitting_rate = 10 → babysitting_hours = 8 :=
by
  intros
  sorry

end donna_babysitting_hours_l230_230667


namespace average_height_of_students_l230_230882

theorem average_height_of_students (x : ℕ) (female_height male_height : ℕ) 
  (female_height_eq : female_height = 170) (male_height_eq : male_height = 185) 
  (ratio : 2 * x = x * 2) : 
  ((2 * x * male_height + x * female_height) / (2 * x + x) = 180) := 
by
  sorry

end average_height_of_students_l230_230882


namespace sasha_remainder_l230_230089

theorem sasha_remainder (n a b c d : ℕ) (h1 : n = 102 * a + b) (h2 : n = 103 * c + d)
  (h3 : d = 20 - a) (h4 : 0 ≤ b ∧ b ≤ 101) : b = 20 :=
by
  sorry

end sasha_remainder_l230_230089


namespace problem_1_and_2_l230_230313

-- Definition of the problem in Lean 4
theorem problem_1_and_2 (A B C : ℝ) (a b c : ℝ)
  (h1 : sin (A)^2 - sin (B)^2 - sin (C)^2 = sin (B) * sin (C))
  (h2 : BC = 3)
  (h3 : triangle ABC)
  (h4 : a = side_length (opposite A))
  (h5 : b = side_length (opposite B))
  (h6 : c = side_length (opposite C)) :
  A = 2 * real.pi / 3 ∧
  (a + b + c ≤ 3 + 2 * real.sqrt 3) :=
sorry

end problem_1_and_2_l230_230313


namespace pool_width_l230_230476

-- Define the given conditions
def hose_rate : ℝ := 60 -- cubic feet per minute
def drain_time : ℝ := 2000 -- minutes
def pool_length : ℝ := 150 -- feet
def pool_depth : ℝ := 10 -- feet

-- Calculate the total volume drained
def total_volume := hose_rate * drain_time -- cubic feet

-- Define a variable for the pool width
variable (W : ℝ)

-- The statement to prove
theorem pool_width :
  (total_volume = pool_length * W * pool_depth) → W = 80 :=
by
  sorry

end pool_width_l230_230476


namespace difference_of_M_and_m_l230_230386

-- Define the variables and conditions
def total_students : ℕ := 2500
def min_G : ℕ := 1750
def max_G : ℕ := 1875
def min_R : ℕ := 1000
def max_R : ℕ := 1125

-- The statement to prove
theorem difference_of_M_and_m : 
  ∃ G R m M, 
  (G = total_students - R + m) ∧ 
  (min_G ≤ G ∧ G ≤ max_G) ∧
  (min_R ≤ R ∧ R ≤ max_R) ∧
  (m = min_G + min_R - total_students) ∧
  (M = max_G + max_R - total_students) ∧
  (M - m = 250) :=
sorry

end difference_of_M_and_m_l230_230386


namespace find_a_no_solution_l230_230400

noncomputable def no_solution_eq (a : ℝ) : Prop :=
  ∀ x : ℝ, ¬ (8 * |x - 4 * a| + |x - a^2| + 7 * x - 2 * a = 0)

theorem find_a_no_solution :
  ∀ a : ℝ, no_solution_eq a ↔ (a < -22 ∨ a > 0) :=
by
  intro a
  sorry

end find_a_no_solution_l230_230400


namespace Sonja_oil_used_l230_230744

theorem Sonja_oil_used :
  ∀ (oil peanuts total_weight : ℕ),
  (2 * oil + 8 * peanuts = 10) → (total_weight = 20) →
  ((20 / 10) * 2 = 4) :=
by 
  sorry

end Sonja_oil_used_l230_230744


namespace unique_zero_iff_a_in_range_l230_230686

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 3 * x^2 + 1

theorem unique_zero_iff_a_in_range (a : ℝ) :
  (∃ x0 : ℝ, f a x0 = 0 ∧ (∀ x1 : ℝ, f a x1 = 0 → x1 = x0) ∧ x0 > 0) ↔ a < -2 :=
by sorry

end unique_zero_iff_a_in_range_l230_230686


namespace tan_double_angle_l230_230303

theorem tan_double_angle (x : ℝ) (h : (Real.sqrt 3) * Real.cos x - Real.sin x = 0) : Real.tan (2 * x) = - (Real.sqrt 3) :=
by
  sorry

end tan_double_angle_l230_230303


namespace sarah_probability_l230_230470

noncomputable def probability_odd_product_less_than_20 : ℚ :=
  let total_possibilities := 36
  let favorable_pairs := [(1, 1), (1, 3), (1, 5), (3, 1), (3, 3), (3, 5), (5, 1), (5, 3)]
  let favorable_count := favorable_pairs.length
  let probability := favorable_count / total_possibilities
  probability

theorem sarah_probability : probability_odd_product_less_than_20 = 2 / 9 :=
by
  sorry

end sarah_probability_l230_230470


namespace arman_sister_age_l230_230798

-- Define the conditions
variables (S : ℝ) -- Arman's sister's age four years ago
variable (A : ℝ) -- Arman's age four years ago

-- Given conditions as hypotheses
axiom h1 : A = 6 * S -- Arman is six times older than his sister
axiom h2 : A + 8 = 40 -- In 4 years, Arman's age will be 40 (hence, A in 4 years should be A + 8)

-- Main theorem to prove
theorem arman_sister_age (h1 : A = 6 * S) (h2 : A + 8 = 40) : S = 16 / 3 :=
by
  sorry

end arman_sister_age_l230_230798


namespace floor_x_mul_x_eq_54_l230_230263

def positive_real (x : ℝ) : Prop := x > 0

theorem floor_x_mul_x_eq_54 (x : ℝ) (h_pos : positive_real x) : ⌊x⌋ * x = 54 ↔ x = 54 / 7 :=
by
  sorry

end floor_x_mul_x_eq_54_l230_230263


namespace total_peanuts_in_box_l230_230994

def initial_peanuts := 4
def peanuts_taken_out := 3
def peanuts_added := 12

theorem total_peanuts_in_box : initial_peanuts - peanuts_taken_out + peanuts_added = 13 :=
by
sorry

end total_peanuts_in_box_l230_230994


namespace Dan_team_lost_games_l230_230526

/-- Dan's high school played eighteen baseball games this year.
Two were at night and they won 15 games. Prove that they lost 3 games. -/
theorem Dan_team_lost_games (total_games won_games : ℕ) (h_total : total_games = 18) (h_won : won_games = 15) :
  total_games - won_games = 3 :=
by {
  sorry
}

end Dan_team_lost_games_l230_230526


namespace candy_distribution_l230_230875

theorem candy_distribution (A B C : ℕ) (x y : ℕ)
  (h1 : A > 2 * B)
  (h2 : B > 3 * C)
  (h3 : A + B + C = 200) :
  (A = 121) ∧ (C = 19) :=
  sorry

end candy_distribution_l230_230875


namespace min_value_x_squared_plus_10x_l230_230359

theorem min_value_x_squared_plus_10x : ∃ x : ℝ, (x^2 + 10 * x) = -25 :=
by {
  sorry
}

end min_value_x_squared_plus_10x_l230_230359


namespace tray_height_l230_230383

-- Declare the main theorem with necessary given conditions.
theorem tray_height (a b c : ℝ) (side_length : ℝ) (cut_distance : ℝ) (angle : ℝ) : 
  (side_length = 150) →
  (cut_distance = Real.sqrt 50) →
  (angle = 45) →
  a^2 + b^2 = c^2 → -- Condition from Pythagorean theorem
  a = side_length * Real.sqrt 2 / 2 - cut_distance → -- Calculation for half diagonal minus cut distance
  b = (side_length * Real.sqrt 2 / 2 - cut_distance) / 2 → -- Perpendicular from R to the side
  side_length = 150 → -- Ensure consistency of side length
  b^2 + c^2 = side_length^2 → -- Ensure we use another Pythagorean relation
  c = Real.sqrt 7350 → -- Derived c value
  c = Real.sqrt 1470 := -- Simplified form of c.
  sorry

end tray_height_l230_230383


namespace necessary_and_sufficient_condition_for_negative_root_l230_230224

theorem necessary_and_sufficient_condition_for_negative_root (a : ℝ) :
  (∃ x : ℝ, x < 0 ∧ a * x^2 + 1 = 0) ↔ (a < 0) :=
by
  sorry

end necessary_and_sufficient_condition_for_negative_root_l230_230224


namespace juniors_in_program_l230_230437

theorem juniors_in_program (J S x y : ℕ) (h1 : J + S = 40) 
                           (h2 : x = y) 
                           (h3 : J / 5 = x) 
                           (h4 : S / 10 = y) : J = 12 :=
by
  sorry

end juniors_in_program_l230_230437


namespace probability_all_girls_l230_230508

theorem probability_all_girls (total_members boys girls : ℕ) (total_members = 15) (boys = 6) (girls = 9)
    (choose_3_total : ℕ := (Nat.choose total_members 3)) (choose_3_girls : ℕ := (Nat.choose girls 3)) :
    (choose_3_total = 455) →
    (choose_3_girls = 84) →
    (choose_3_girls.toRat / choose_3_total.toRat = (12 / 65) : ℚ) :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end probability_all_girls_l230_230508


namespace monotonic_decreasing_interval_l230_230341

noncomputable def xlnx (x : ℝ) : ℝ := x * Real.log x

theorem monotonic_decreasing_interval : 
  ∀ x, (0 < x) ∧ (x < 5) → (Real.log x + 1 < 0) ↔ (0 < x) ∧ (x < 1 / Real.exp 1) := 
by
  sorry

end monotonic_decreasing_interval_l230_230341


namespace find_speed_of_second_car_l230_230893

noncomputable def problem : Prop := 
  let s1 := 1600 -- meters
  let s2 := 800 -- meters
  let v1 := 72 / 3.6 -- converting to meters per second for convenience; 72 km/h = 20 m/s
  let s := 200 -- meters
  let t1 := s1 / v1 -- time taken by the first car to reach the intersection
  let l1 := s2 - s -- scenario 1: second car travels 600 meters
  let l2 := s2 + s -- scenario 2: second car travels 1000 meters
  let v2_1 := l1 / t1 -- speed calculation for scenario 1
  let v2_2 := l2 / t1 -- speed calculation for scenario 2
  v2_1 = 7.5 ∧ v2_2 = 12.5 -- expected speeds in both scenarios

theorem find_speed_of_second_car : problem := sorry

end find_speed_of_second_car_l230_230893


namespace min_value_l230_230975

theorem min_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 3 * x + y + 1 / x + 2 / y = 13 / 2) :
  x - 1 / y ≥ - 1 / 2 :=
sorry

end min_value_l230_230975


namespace todd_savings_l230_230348

-- Define the initial conditions
def original_price : ℝ := 125
def sale_discount : ℝ := 0.20
def coupon : ℝ := 10
def card_discount : ℝ := 0.10

-- Define the resulting values after applying discounts
def sale_price := original_price * (1 - sale_discount)
def after_coupon := sale_price - coupon
def final_price := after_coupon * (1 - card_discount)

-- Define the total savings
def savings := original_price - final_price

-- The proof statement
theorem todd_savings : savings = 44 := by
  sorry

end todd_savings_l230_230348


namespace sum_q_t_at_12_l230_230324

def T := {t : Fin 12 → Bool // true}

def q_t (t : T) : Polynomial ℝ :=
  Polynomial.ofFinList (λ n, if h : n < 12 then t.val ⟨n, h⟩ else 0)

noncomputable def q (x : ℕ) : ℝ :=
  ∑ t in Finset.univ.image subtype.val, q_t ⟨t, sorry⟩.eval x

theorem sum_q_t_at_12 : q 12 = 2048 := sorry

end sum_q_t_at_12_l230_230324


namespace rent_percentage_l230_230590

noncomputable def condition1 (E : ℝ) : ℝ := 0.25 * E
noncomputable def condition2 (E : ℝ) : ℝ := 1.35 * E
noncomputable def condition3 (E' : ℝ) : ℝ := 0.40 * E'

theorem rent_percentage (E R R' : ℝ) (hR : R = condition1 E) (hE' : E = condition2 E) (hR' : R' = condition3 E) :
  (R' / R) * 100 = 216 :=
sorry

end rent_percentage_l230_230590


namespace winning_cards_at_least_one_l230_230811

def cyclicIndex (n : ℕ) (i : ℕ) : ℕ := (i % n + n) % n

theorem winning_cards_at_least_one (a : ℕ → ℕ) (h : ∀ i, (a (cyclicIndex 8 (i - 1)) + a i + a (cyclicIndex 8 (i + 1))) % 2 = 1) :
  ∀ i, 1 ≤ a i :=
by
  sorry

end winning_cards_at_least_one_l230_230811


namespace water_left_after_four_hours_l230_230567

def initial_water : ℕ := 40
def water_loss_per_hour : ℕ := 2
def water_added_hour3 : ℕ := 1
def water_added_hour4 : ℕ := 3

theorem water_left_after_four_hours :
    initial_water - 2 * 2 - 2 + water_added_hour3 - 2 + water_added_hour4 - 2 = 36 := 
by
    sorry

end water_left_after_four_hours_l230_230567


namespace max_value_of_a_l230_230173

theorem max_value_of_a {a : ℝ} (h : ∀ x ≥ 1, -3 * x^2 + a ≤ 0) : a ≤ 3 :=
sorry

end max_value_of_a_l230_230173


namespace marble_weight_l230_230732

-- Define the conditions
def condition1 (m k : ℝ) : Prop := 9 * m = 5 * k
def condition2 (k : ℝ) : Prop := 4 * k = 120

-- Define the main goal, i.e., proving m = 50/3 given the conditions
theorem marble_weight (m k : ℝ) 
  (h1 : condition1 m k) 
  (h2 : condition2 k) : 
  m = 50 / 3 := by 
  sorry

end marble_weight_l230_230732


namespace typist_speeds_l230_230142

noncomputable def num_pages : ℕ := 72
noncomputable def ratio : ℚ := 6 / 5
noncomputable def time_difference : ℚ := 1.5

theorem typist_speeds :
  ∃ (x y : ℚ), (x = 9.6 ∧ y = 8) ∧ 
                (num_pages / x - num_pages / y = time_difference) ∧
                (x / y = ratio) :=
by
  -- Let's skip the proof for now
  sorry

end typist_speeds_l230_230142


namespace donna_received_total_interest_l230_230531

-- Donna's investment conditions
def totalInvestment : ℝ := 33000
def investmentAt4Percent : ℝ := 13000
def investmentAt225Percent : ℝ := totalInvestment - investmentAt4Percent
def rate4Percent : ℝ := 0.04
def rate225Percent : ℝ := 0.0225

-- The interest calculation
def interestFrom4PercentInvestment : ℝ := investmentAt4Percent * rate4Percent
def interestFrom225PercentInvestment : ℝ := investmentAt225Percent * rate225Percent
def totalInterest : ℝ := interestFrom4PercentInvestment + interestFrom225PercentInvestment

-- The proof statement
theorem donna_received_total_interest :
  totalInterest = 970 := by
sorry

end donna_received_total_interest_l230_230531


namespace range_of_m_inequality_system_l230_230282

theorem range_of_m_inequality_system (m : ℝ) :
  (∀ x : ℤ, (-5 < x ∧ x ≤ m + 1) ↔ (x = -4 ∨ x = -3 ∨ x = -2)) →
  -3 ≤ m ∧ m < -2 :=
by
  sorry

end range_of_m_inequality_system_l230_230282


namespace min_S_in_grid_l230_230016

def valid_grid (grid : Fin 10 × Fin 10 → Fin 100) (S : ℕ) : Prop :=
  ∀ i j, 
    (i < 9 → grid (i, j) + grid (i + 1, j) ≤ S) ∧
    (j < 9 → grid (i, j) + grid (i, j + 1) ≤ S)

theorem min_S_in_grid : ∃ grid : Fin 10 × Fin 10 → Fin 100, ∃ S : ℕ, valid_grid grid S ∧ 
  (∀ (other_S : ℕ), valid_grid grid other_S → S ≤ other_S) ∧ S = 106 :=
sorry

end min_S_in_grid_l230_230016


namespace factorize_expression_l230_230814

theorem factorize_expression (x y : ℂ) : (x * y^2 - x = x * (y + 1) * (y - 1)) :=
sorry

end factorize_expression_l230_230814


namespace correct_transformation_l230_230285

variable (a b : ℝ)
variable (h₀ : a ≠ 0)
variable (h₁ : b ≠ 0)
variable (h₂ : a / 2 = b / 3)

theorem correct_transformation : 3 / b = 2 / a :=
by
  sorry

end correct_transformation_l230_230285


namespace breadth_of_hall_l230_230792

theorem breadth_of_hall (length_hall : ℝ) (stone_length_dm : ℝ) (stone_breadth_dm : ℝ)
    (num_stones : ℕ) (area_stone_m2 : ℝ) (total_area_m2 : ℝ) (breadth_hall : ℝ):
    length_hall = 36 → 
    stone_length_dm = 8 → 
    stone_breadth_dm = 5 → 
    num_stones = 1350 → 
    area_stone_m2 = (stone_length_dm * stone_breadth_dm) / 100 → 
    total_area_m2 = num_stones * area_stone_m2 → 
    breadth_hall = total_area_m2 / length_hall → 
    breadth_hall = 15 :=
by 
  intros h1 h2 h3 h4 h5 h6 h7
  rw [h1, h2, h3, h4] at *
  simp [h5, h6, h7]
  sorry

end breadth_of_hall_l230_230792


namespace triangle_A_value_and_max_perimeter_l230_230315

theorem triangle_A_value_and_max_perimeter (A B C a b c : ℝ) 
  (h1 : sin A ^ 2 - sin B ^ 2 - sin C ^ 2 = sin B * sin C)
  (h2 : a = 3) :
  (A = 2 * Real.pi / 3) ∧ (a + b + c ≤ 3 + 2 * Real.sqrt 3) :=
by
  sorry

end triangle_A_value_and_max_perimeter_l230_230315


namespace oldest_child_age_l230_230339

open Nat

def avg_age (a b c d : ℕ) := (a + b + c + d) / 4

theorem oldest_child_age 
  (h_avg : avg_age 5 8 11 x = 9) : x = 12 :=
by
  sorry

end oldest_child_age_l230_230339


namespace number_of_sides_l230_230125

theorem number_of_sides (P s : ℝ) (hP : P = 108) (hs : s = 12) : P / s = 9 :=
by sorry

end number_of_sides_l230_230125


namespace find_quadratic_polynomial_l230_230265

noncomputable def quadratic_polynomial : Polynomial ℝ :=
  3 * (X - C (2 + 2*I)) * (X - C (2 - 2*I))

theorem find_quadratic_polynomial :
  quadratic_polynomial = 3 * X^2 - 12 * X + 24 :=
by
  sorry

end find_quadratic_polynomial_l230_230265


namespace rebecca_charge_for_dye_job_l230_230737

def charges_for_services (haircuts per perms per dye_jobs hair_dye_per_dye_job tips : ℕ) : ℕ := 
  4 * 30 + 1 * 40 + 2 * (dye_jobs - hair_dye_per_dye_job) + tips

theorem rebecca_charge_for_dye_job 
  (haircuts: ℕ) (perms: ℕ) (hair_dye_per_dye_job: ℕ) (tips: ℕ) (end_of_day_amount: ℕ) : 
  haircuts = 4 → perms = 1 → hair_dye_per_dye_job = 10 → tips = 50 → 
  end_of_day_amount = 310 → 
  ∃ D: ℕ, D = 60 := 
by
  sorry

end rebecca_charge_for_dye_job_l230_230737


namespace range_of_k_l230_230680

open Real

noncomputable def f (x : ℝ) : ℝ := x^2 + x - 1

noncomputable def g (x : ℝ) : ℝ := x^2 - 1

noncomputable def h (x : ℝ) : ℝ := x

theorem range_of_k (k : ℝ) :
  (∀ x : ℝ, x ≠ 0 → g (k * x + k / x) < g (x^2 + 1 / x^2 + 1)) ↔ (-3 / 2 < k ∧ k < 3 / 2) :=
by
  sorry

end range_of_k_l230_230680


namespace angie_carlos_probability_l230_230659

theorem angie_carlos_probability :
  (∃ (positions : list string), 
    positions.length = 5 ∧ 
    {a, b, c, d, e} = {Angie, Bridget, Carlos, Diego, Eliza} ∧ 
    position Angie positions = n ∧ 
    (position Carlos positions = (n + 2) % 5 ∨ position Carlos positions = (n - 2) % 5) 
    /
    (5! = 5 permutations)) = 1/2 := by
  sorry

end angie_carlos_probability_l230_230659


namespace radius_increase_l230_230364

theorem radius_increase (C1 C2 : ℝ) (h1 : C1 = 30) (h2 : C2 = 40) : 
  let r1 := C1 / (2 * Real.pi)
  let r2 := C2 / (2 * Real.pi)
  let Δr := r2 - r1
  Δr = 5 / Real.pi := by
sorry

end radius_increase_l230_230364


namespace geometric_sequence_new_product_l230_230547

theorem geometric_sequence_new_product 
  (a r : ℝ) (n : ℕ) (h_even : n % 2 = 0)
  (P S S' : ℝ)
  (hP : P = a^n * r^(n * (n-1) / 2))
  (hS : S = a * (1 - r^n) / (1 - r))
  (hS' : S' = (1 - r^n) / (a * (1 - r))) :
  (2^n * a^n * r^(n * (n-1) / 2)) = (S * S')^(n / 2) :=
sorry

end geometric_sequence_new_product_l230_230547


namespace find_number_l230_230990

theorem find_number 
  (x y n : ℝ)
  (h1 : n * x = 0.04 * y)
  (h2 : (y - x) / (y + x) = 0.948051948051948) :
  n = 37.5 :=
sorry  -- proof omitted

end find_number_l230_230990


namespace shekar_math_marks_l230_230469

variable (science socialStudies english biology average : ℕ)

theorem shekar_math_marks 
  (h1 : science = 65)
  (h2 : socialStudies = 82)
  (h3 : english = 67)
  (h4 : biology = 95)
  (h5 : average = 77) :
  ∃ M, average = (science + socialStudies + english + biology + M) / 5 ∧ M = 76 :=
by
  sorry

end shekar_math_marks_l230_230469


namespace abc_value_l230_230169

theorem abc_value 
  (a b c : ℝ)
  (a_pos : 0 < a) 
  (b_pos : 0 < b) 
  (c_pos : 0 < c) 
  (hab : a * b = 24) 
  (hac : a * c = 40) 
  (hbc : b * c = 60) : 
  a * b * c = 240 := 
by sorry

end abc_value_l230_230169


namespace pyramid_pattern_l230_230507

theorem pyramid_pattern
  (R : ℕ → ℕ)  -- a function representing the number of blocks in each row
  (R₁ : R 1 = 9)  -- the first row has 9 blocks
  (sum_eq : R 1 + R 2 + R 3 + R 4 + R 5 = 25)  -- the total number of blocks is 25
  (pattern : ∀ n, 1 ≤ n ∧ n < 5 → R (n + 1) = R n - 2) : ∃ d, d = 2 :=
by
  have pattern_valid : R 1 = 9 ∧ R 2 = 7 ∧ R 3 = 5 ∧ R 4 = 3 ∧ R 5 = 1 :=
    sorry  -- Proof omitted
  exact ⟨2, rfl⟩

end pyramid_pattern_l230_230507


namespace unique_positive_integer_solutions_l230_230399

theorem unique_positive_integer_solutions : 
  ∀ (m n : ℕ), 0 < m ∧ 0 < n ∧ 7 ^ m - 3 * 2 ^ n = 1 ↔ (m = 1 ∧ n = 1) ∨ (m = 2 ∧ n = 4) :=
by sorry

end unique_positive_integer_solutions_l230_230399


namespace student_failed_by_l230_230516

theorem student_failed_by :
  ∀ (total_marks obtained_marks passing_percentage : ℕ),
  total_marks = 700 →
  obtained_marks = 175 →
  passing_percentage = 33 →
  (passing_percentage * total_marks) / 100 - obtained_marks = 56 :=
by
  intros total_marks obtained_marks passing_percentage h1 h2 h3
  sorry

end student_failed_by_l230_230516


namespace product_of_consecutive_multiples_of_4_divisible_by_192_l230_230223

theorem product_of_consecutive_multiples_of_4_divisible_by_192 :
  ∀ (n : ℤ), 192 ∣ (4 * n) * (4 * (n + 1)) * (4 * (n + 2)) :=
by
  intro n
  sorry

end product_of_consecutive_multiples_of_4_divisible_by_192_l230_230223


namespace find_b_value_l230_230889

variable (a p q b : ℝ)
variable (h1 : p * 0 + q * (3 * a) + b * 1 = 1)
variable (h2 : p * (9 * a) + q * (-1) + b * 2 = 1)
variable (h3 : p * 0 + q * (3 * a) + b * 0 = 1)

theorem find_b_value : b = 0 :=
by
  sorry

end find_b_value_l230_230889


namespace min_value_of_abc_l230_230157

variables {a b c : ℝ}

noncomputable def satisfies_condition (a b c : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ (b + c) / a + (a + c) / b = (a + b) / c + 1

theorem min_value_of_abc (a b c : ℝ) (h : satisfies_condition a b c) : (a + b) / c ≥ 5 / 2 :=
sorry

end min_value_of_abc_l230_230157


namespace find_prime_p_l230_230720

noncomputable def concatenate (q r : ℕ) : ℕ :=
q * 10 ^ (r.digits 10).length + r

theorem find_prime_p (q r p : ℕ) (hq : Nat.Prime q) (hr : Nat.Prime r) (hp : Nat.Prime p)
  (h : concatenate q r + 3 = p^2) : p = 5 :=
sorry

end find_prime_p_l230_230720


namespace total_animal_sightings_l230_230703

def A_Jan := 26
def A_Feb := 3 * A_Jan
def A_Mar := A_Feb / 2

theorem total_animal_sightings : A_Jan + A_Feb + A_Mar = 143 := by
  sorry

end total_animal_sightings_l230_230703


namespace angle_AQB_obtuse_probability_correct_l230_230036

-- Define the vertices of the pentagon
def A : ℝ × ℝ := (-1, 3)
def B : ℝ × ℝ := (3, 1)
def C : ℝ × ℝ := (5, 1)
def D : ℝ × ℝ := (5, 6)
def E : ℝ × ℝ := (-1, 6)

-- Define the probability calculation
noncomputable def probability_obtuse_AQB : ℝ :=
  let area_pentagon := 23
  let radius_semi := Real.sqrt 5
  let area_semi := (radius_semi^2 * Real.pi) / 2
  (area_semi / area_pentagon: ℝ)

-- The theorem statement
theorem angle_AQB_obtuse_probability_correct :
  probability_obtuse_AQB = 5 * Real.pi / 46 :=
sorry

end angle_AQB_obtuse_probability_correct_l230_230036


namespace parabola_vertex_point_l230_230613

theorem parabola_vertex_point (a b c : ℝ) 
    (h_vertex : ∃ a b c : ℝ, ∀ x : ℝ, y = a * x^2 + b * x + c) 
    (h_vertex_coord : ∃ (h k : ℝ), h = 3 ∧ k = -5) 
    (h_pass : ∃ (x y : ℝ), x = 0 ∧ y = -2) :
    c = -2 := by
  sorry

end parabola_vertex_point_l230_230613


namespace hyperbola_eccentricity_l230_230688

theorem hyperbola_eccentricity (m : ℝ) : 
  (∃ e : ℝ, e = 5 / 4 ∧ (∀ x y : ℝ, (x^2 / 16) - (y^2 / m) = 1)) → m = 9 :=
by
  intro h
  sorry

end hyperbola_eccentricity_l230_230688


namespace solution_inequality_l230_230993

theorem solution_inequality (m : ℝ) :
  (∀ x : ℝ, x^2 - (m+3)*x + 3*m < 0 ↔ m ∈ Set.Icc 3 (-1) ∪ Set.Icc 6 7) →
  m = -1/2 ∨ m = 13/2 :=
sorry

end solution_inequality_l230_230993


namespace geometric_sequence_common_ratio_l230_230375

theorem geometric_sequence_common_ratio :
  ∃ r : ℚ, 
  ∃ (a1 a2 a3 a4 : ℚ), 
  a1 = 16 ∧ a2 = -24 ∧ a3 = 36 ∧ a4 = -54 ∧ 
  r = a2 / a1 ∧ r = -3 / 2 :=
begin
  use -3 / 2,
  use 16, use -24, use 36, use -54,
  split, refl,
  split, refl,
  split, refl,
  split, refl,
  split,
  { norm_num },
  { norm_num }
end

end geometric_sequence_common_ratio_l230_230375


namespace find_a_l230_230002

noncomputable def angle := 30 * Real.pi / 180 -- In radians

noncomputable def tan_angle : ℝ := Real.tan angle

theorem find_a (a : ℝ) (h1 : tan_angle = 1 / Real.sqrt 3) : 
  x - a * y + 3 = 0 → a = Real.sqrt 3 :=
by
  sorry

end find_a_l230_230002


namespace probability_of_at_least_2_girls_equals_specified_value_l230_230749

def num_combinations (n k : ℕ) : ℕ :=
  (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

noncomputable def probability_at_least_2_girls : ℚ :=
  let total_committees := num_combinations 24 5
  let all_boys := num_combinations 14 5
  let one_girl_four_boys := num_combinations 10 1 * num_combinations 14 4
  let at_least_2_girls := total_committees - (all_boys + one_girl_four_boys)
  at_least_2_girls / total_committees

theorem probability_of_at_least_2_girls_equals_specified_value :
  probability_at_least_2_girls = 2541 / 3542 := 
sorry

end probability_of_at_least_2_girls_equals_specified_value_l230_230749


namespace Olivia_money_left_l230_230464

theorem Olivia_money_left (initial_amount spend_amount : ℕ) (h1 : initial_amount = 128) 
  (h2 : spend_amount = 38) : initial_amount - spend_amount = 90 := by
  sorry

end Olivia_money_left_l230_230464


namespace find_m_positive_root_l230_230417

theorem find_m_positive_root :
  (∃ x > 0, (x - 4) / (x - 3) - m - 4 = m / (3 - x)) → m = 1 :=
by
  sorry

end find_m_positive_root_l230_230417


namespace election_1002nd_k_election_1001st_k_l230_230584

variable (k : ℕ)

noncomputable def election_in_1002nd_round_max_k : Prop :=
  ∀ (n : ℕ), (n = 2002) → (k ≤ n - 1) → k = 2001 → -- The conditions include the number of candidates 'n', and specifying that 'k' being the maximum initially means k ≤ 2001.
  true

noncomputable def election_in_1001st_round_max_k : Prop :=
  ∀ (n : ℕ), (n = 2002) → (k ≤ n - 1) → k = 1 → -- Similarly, these conditions specify the initial maximum placement as 1 when elected in 1001st round.
  true

-- Definitions specifying the problem to identify max k for given rounds
theorem election_1002nd_k : election_in_1002nd_round_max_k k := sorry

theorem election_1001st_k : election_in_1001st_round_max_k k := sorry

end election_1002nd_k_election_1001st_k_l230_230584


namespace abs_value_difference_l230_230571

theorem abs_value_difference (x y : ℤ) (h1 : |x| = 7) (h2 : |y| = 9) (h3 : |x + y| = -(x + y)) :
  x - y = 16 ∨ x - y = -16 :=
sorry

end abs_value_difference_l230_230571


namespace remainder_zero_l230_230491

theorem remainder_zero (x : ℕ) (h1 : x = 1680) :
  (x % 5 = 0) ∧ (x % 6 = 0) ∧ (x % 7 = 0) ∧ (x % 8 = 0) :=
by
  sorry

end remainder_zero_l230_230491


namespace mean_of_xyz_l230_230197

theorem mean_of_xyz (x y z : ℚ) (eleven_mean : ℚ)
  (eleven_sum : eleven_mean = 32)
  (fourteen_sum : 14 * 45 = 630)
  (new_mean : 14 * 45 = 630) :
  (x + y + z) / 3 = 278 / 3 :=
by
  sorry

end mean_of_xyz_l230_230197


namespace ratio_brother_to_joanna_l230_230444

/-- Definitions for the conditions -/
def joanna_money : ℝ := 8
def sister_money : ℝ := 4 -- since it's half of Joanna's money
def total_money : ℝ := 36

/-- Stating the theorem -/
theorem ratio_brother_to_joanna (x : ℝ) (h : joanna_money + 8*x + sister_money = total_money) :
  x = 3 :=
by 
  -- The ratio of brother's money to Joanna's money is 3:1
  sorry

end ratio_brother_to_joanna_l230_230444


namespace sphere_cone_radius_ratio_l230_230512

-- Define the problem using given conditions and expected outcome.
theorem sphere_cone_radius_ratio (r R h : ℝ)
  (h1 : h = 2 * r)
  (h2 : (1/3) * π * R^2 * h = 3 * (4/3) * π * r^3) :
  r / R = 1 / Real.sqrt 6 :=
by
  sorry

end sphere_cone_radius_ratio_l230_230512


namespace geometric_sequence_general_term_and_sum_l230_230840

variable {a : ℕ → ℕ}

-- We define the initial condition and the recurrence relation
def a_1 := 5
def recurrence (n : ℕ) := a (n + 1) = 2 * a n + 1

-- Problem 1: Prove that the sequence {a_n + 1} is a geometric sequence
theorem geometric_sequence (n : ℕ) (h₁ : a 0 = a_1) (h₂ : ∀ n : ℕ, recurrence n) : 
  ∃ r : ℕ, ∀ n : ℕ, (a (n + 1) + 1) = r * (a n + 1) := sorry

-- Problem 2: Find the general term formula for {a_n} and the sum of the first n terms S_n
theorem general_term_and_sum (n : ℕ) (h₁ : a 0 = a_1) (h₂ : ∀ n : ℕ, recurrence n) :
  (a n = 6 * 2^(n - 1) - 1) ∧ (∑ i in Finset.range n, a i = 6 * 2^n - n - 6) := sorry

end geometric_sequence_general_term_and_sum_l230_230840


namespace xy_equation_result_l230_230037

theorem xy_equation_result (x y : ℝ) (h1 : x + y = 6) (h2 : x * y = -5) :
  x + (x^4 / y^3) + (y^4 / x^3) + y = -10.528 :=
by
  sorry

end xy_equation_result_l230_230037


namespace geometric_sequence_properties_l230_230838

-- Given conditions as definitions
def seq (a : ℕ → ℝ) : Prop :=
  a 1 * a 3 = a 4 ∧ a 3 = 8

-- Prove the common ratio and the sum of the first n terms
theorem geometric_sequence_properties (a : ℕ → ℝ)
  (h : seq a) :
  (∃ q, ∀ n, a n = a 1 * q ^ (n - 1) ∧ q = 2) ∧
  (∀ S_n, S_n = (1 - (2 : ℝ) ^ S_n) / (1 - 2) ∧ S_n = 2 ^ S_n - 1) :=
by
  sorry

end geometric_sequence_properties_l230_230838


namespace find_a_l230_230985

theorem find_a {a b c : ℕ} (h₁ : a + b = c) (h₂ : b + c = 8) (h₃ : c = 4) : a = 0 := by
  sorry

end find_a_l230_230985


namespace largest_whole_number_for_inequality_l230_230904

theorem largest_whole_number_for_inequality :
  ∀ n : ℕ, (1 : ℝ) / 4 + (n : ℝ) / 6 < 3 / 2 → n ≤ 7 :=
by
  admit  -- skip the proof

end largest_whole_number_for_inequality_l230_230904


namespace probability_sum_5_l230_230510

theorem probability_sum_5 :
  let total_outcomes := 36
  let favorable_outcomes := 4
  (favorable_outcomes / total_outcomes : ℚ) = 1 / 9 :=
by
  -- proof omitted
  sorry

end probability_sum_5_l230_230510


namespace evaluate_expression_l230_230143

theorem evaluate_expression : 6 - 8 * (9 - 4^2) * 5 = 286 := by
  sorry

end evaluate_expression_l230_230143


namespace solve_trig_inequality_l230_230541

noncomputable def sin_triple_angle_identity (x : ℝ) : ℝ :=
  3 * (Real.sin x) - 4 * (Real.sin x) ^ 3

theorem solve_trig_inequality (x : ℝ) (h1 : 0 < x) (h2 : x < Real.pi) :
  (8 / (3 * Real.sin x - sin_triple_angle_identity x) + 3 * (Real.sin x) ^ 2) ≤ 5 ↔
  x = Real.pi / 2 :=
by
  sorry

end solve_trig_inequality_l230_230541


namespace line_through_point_with_equal_intercepts_l230_230147

theorem line_through_point_with_equal_intercepts :
  ∃ (m b : ℝ), ∀ (x y : ℝ), 
    ((y = m * x + b ∧ ((x = 0 ∨ y = 0) → (x = y))) ∧ 
    (1 = m * 1 + b ∧ 1 + 1 = b)) → 
    (m = 1 ∧ b = 0) ∨ (m = -1 ∧ b = 2) :=
by
  sorry

end line_through_point_with_equal_intercepts_l230_230147


namespace incorrect_conclusion_l230_230549

variable {a b c : ℝ}

theorem incorrect_conclusion
  (h1 : a^2 + a * b = c)
  (h2 : a * b + b^2 = c + 5) :
  ¬(2 * c + 5 < 0) ∧ ¬(∃ k, a^2 - b^2 ≠ k) ∧ ¬(a = b ∨ a = -b) ∧ ¬(b / a > 1) :=
by sorry

end incorrect_conclusion_l230_230549


namespace factorize_expression_l230_230815

-- Defining the variables x and y as real numbers.
variable (x y : ℝ)

-- Statement of the proof problem.
theorem factorize_expression : 
  x * y^2 - x = x * (y + 1) * (y - 1) :=
sorry

end factorize_expression_l230_230815


namespace find_smallest_x_l230_230270

noncomputable def smallest_x : ℝ :=
  min { x : ℝ | 0 < x ∧ (⌊x^2⌋ : ℤ) - (⌊x⌋ : ℤ) * x = 8 }

theorem find_smallest_x :
  smallest_x = 89 / 9 :=
by
  sorry

end find_smallest_x_l230_230270


namespace number_is_seven_l230_230734

theorem number_is_seven (x : ℝ) (h : x^2 + 120 = (x - 20)^2) : x = 7 := 
by
  sorry

end number_is_seven_l230_230734


namespace negation_all_dogs_playful_l230_230202

variable {α : Type} (dog playful : α → Prop)

theorem negation_all_dogs_playful :
  (¬ ∀ x, dog x → playful x) ↔ (∃ x, dog x ∧ ¬ playful x) :=
by sorry

end negation_all_dogs_playful_l230_230202


namespace largest_fraction_addition_l230_230930

-- Definitions for the problem conditions
def proper_fraction (a b : ℕ) : Prop :=
  a < b

def denom_less_than (d : ℕ) (bound : ℕ) : Prop :=
  d < bound

-- Main statement of the problem
theorem largest_fraction_addition :
  ∃ (a b : ℕ), (b > 0) ∧ proper_fraction (b + 7 * a) (7 * b) ∧ denom_less_than b 5 ∧ (a / b : ℚ) <= 3/4 := 
sorry

end largest_fraction_addition_l230_230930


namespace journey_time_l230_230755

-- Conditions
def initial_speed : ℝ := 80  -- miles per hour
def initial_time : ℝ := 5    -- hours
def new_speed : ℝ := 50      -- miles per hour
def distance : ℝ := initial_speed * initial_time

-- Statement
theorem journey_time :
  distance / new_speed = 8.00 :=
by
  sorry

end journey_time_l230_230755


namespace largest_possible_A_l230_230217

theorem largest_possible_A : ∃ A B : ℕ, 13 = 4 * A + B ∧ B < A ∧ A = 3 := by
  sorry

end largest_possible_A_l230_230217


namespace gcd_of_256_180_600_l230_230626

-- Define the numbers
def n1 := 256
def n2 := 180
def n3 := 600

-- Statement to prove the GCD of the three numbers is 12
theorem gcd_of_256_180_600 : Nat.gcd (Nat.gcd n1 n2) n3 = 12 := 
by {
  -- Prime factorizations given as part of the conditions:
  have pf1 : n1 = 2^8 := by norm_num,
  have pf2 : n2 = 2^2 * 3^2 * 5 := by norm_num,
  have pf3 : n3 = 2^3 * 3 * 5^2 := by norm_num,
  
  -- Calculate GCD step by step should be done here, 
  -- But replaced by sorry as per the instructions.
  sorry
}

end gcd_of_256_180_600_l230_230626


namespace probability_more_than_70_l230_230580

-- Definitions based on problem conditions
def P_A : ℝ := 0.15
def P_B : ℝ := 0.45
def P_C : ℝ := 0.25

-- Theorem to state that the probability of scoring more than 70 points is 0.85
theorem probability_more_than_70 (hA : P_A = 0.15) (hB : P_B = 0.45) (hC : P_C = 0.25):
  P_A + P_B + P_C = 0.85 :=
by
  rw [hA, hB, hC]
  sorry

end probability_more_than_70_l230_230580


namespace fraction_of_total_l230_230369

def total_amount : ℝ := 5000
def r_amount : ℝ := 2000.0000000000002

theorem fraction_of_total
  (h1 : r_amount = 2000.0000000000002)
  (h2 : total_amount = 5000) :
  r_amount / total_amount = 0.40000000000000004 :=
by
  -- The proof is skipped
  sorry

end fraction_of_total_l230_230369


namespace find_sample_size_l230_230175

theorem find_sample_size : ∃ n : ℕ, n ∣ 36 ∧ (n + 1) ∣ 35 ∧ n = 6 := by
  use 6
  simp
  sorry

end find_sample_size_l230_230175


namespace find_number_l230_230696

variable (x : ℝ)

theorem find_number (h : 2 * x - 6 = (1/4) * x + 8) : x = 8 :=
sorry

end find_number_l230_230696


namespace find_m_values_l230_230849

theorem find_m_values (m : ℝ) : 
  (∃ A B : ℝ × ℝ, A = (2, 2) ∧ B = (m, 0) ∧ 
   ∃ r R : ℝ, r = 1 ∧ R = 3 ∧ 
   ∃ d : ℝ, d = abs (dist A B) ∧ d = (R + r)) →
  (m = 2 - 2 * Real.sqrt 3 ∨ m = 2 + 2 * Real.sqrt 3) := 
sorry

end find_m_values_l230_230849


namespace case_D_has_two_solutions_l230_230610

-- Definitions for the conditions of each case
structure CaseA :=
(b : ℝ) (A : ℝ) (B : ℝ)

structure CaseB :=
(a : ℝ) (c : ℝ) (B : ℝ)

structure CaseC :=
(a : ℝ) (b : ℝ) (A : ℝ)

structure CaseD :=
(a : ℝ) (b : ℝ) (A : ℝ)

-- Setting the values based on the given conditions
def caseA := CaseA.mk 10 45 70
def caseB := CaseB.mk 60 48 100
def caseC := CaseC.mk 14 16 45
def caseD := CaseD.mk 7 5 80

-- Define a function that checks if a case has two solutions
def has_two_solutions (a b c : ℝ) (A B : ℝ) : Prop := sorry

-- The theorem to prove that out of the given cases, only Case D has two solutions
theorem case_D_has_two_solutions :
  has_two_solutions caseA.b caseB.B caseC.a caseC.b caseC.A = false →
  has_two_solutions caseB.a caseB.c caseB.B caseC.b caseC.A = false →
  has_two_solutions caseC.a caseC.b caseC.A caseD.a caseD.b = false →
  has_two_solutions caseD.a caseD.b caseD.A caseA.b caseA.A = true :=
sorry

end case_D_has_two_solutions_l230_230610


namespace part1_part2_l230_230121

theorem part1 (x y : ℝ) (h1 : y = x + 30) (h2 : 2 * x + 3 * y = 340) : x = 50 ∧ y = 80 :=
by {
  -- Later, we can place the steps to prove x = 50 and y = 80 here.
  sorry
}

theorem part2 (m : ℕ) (h3 : 0 ≤ m ∧ m ≤ 50)
               (h4 : 54 * (50 - m) + 72 * m = 3060) : m = 20 :=
by {
  -- Later, we can place the steps to prove m = 20 here.
  sorry
}

end part1_part2_l230_230121


namespace qt_q_t_neq_2_l230_230431

theorem qt_q_t_neq_2 (q t : ℕ) (hq : 0 < q) (ht : 0 < t) : q * t + q + t ≠ 2 :=
  sorry

end qt_q_t_neq_2_l230_230431


namespace cherries_left_l230_230235

def initial_cherries : ℕ := 77
def cherries_used : ℕ := 60

theorem cherries_left : initial_cherries - cherries_used = 17 := by
  sorry

end cherries_left_l230_230235


namespace omitted_digits_correct_l230_230214

theorem omitted_digits_correct :
  (287 * 23 = 6601) := by
  sorry

end omitted_digits_correct_l230_230214


namespace transistors_in_2005_l230_230307

theorem transistors_in_2005
  (initial_count : ℕ)
  (doubles_every : ℕ)
  (triples_every : ℕ)
  (years : ℕ) :
  initial_count = 500000 ∧ doubles_every = 2 ∧ triples_every = 6 ∧ years = 15 →
  (initial_count * 2^(years/doubles_every) + initial_count * 3^(years/triples_every)) = 68500000 :=
by
  sorry

end transistors_in_2005_l230_230307


namespace evaluate_expression_l230_230396

theorem evaluate_expression :
  -25 + 7 * ((8 / 4) ^ 2) = 3 :=
by
  sorry

end evaluate_expression_l230_230396


namespace inequality_always_holds_l230_230977

theorem inequality_always_holds
  (f : ℝ → ℝ)
  (a b : ℝ)
  (h_def : ∀ x, f x = (1 - 2^x) / (1 + 2^x))
  (h_odd : ∀ x, f (-x) = -f x)
  (h_decreasing : ∀ x y, x < y → f x > f y)
  (h_ineq : f (2 * a + b) + f (4 - 3 * b) > 0)
  : b - a > 2 :=
sorry

end inequality_always_holds_l230_230977


namespace shadow_length_to_time_l230_230145

theorem shadow_length_to_time (shadow_length_inches : ℕ) (stretch_rate_feet_per_hour : ℕ) (inches_per_foot : ℕ) 
                              (shadow_start_time : ℕ) :
  shadow_length_inches = 360 → stretch_rate_feet_per_hour = 5 → inches_per_foot = 12 → shadow_start_time = 0 →
  (shadow_length_inches / inches_per_foot) / stretch_rate_feet_per_hour = 6 := by
  intros h1 h2 h3 h4
  sorry

end shadow_length_to_time_l230_230145


namespace union_A_B_equals_C_l230_230005

-- Define Set A
def A : Set ℝ := {x : ℝ | 3 - 2 * x > 0}

-- Define Set B
def B : Set ℝ := {x : ℝ | x^2 ≤ 4}

-- Define the target set C which is supposed to be A ∪ B
def C : Set ℝ := {x : ℝ | x ≤ 2}

theorem union_A_B_equals_C : A ∪ B = C := by 
  -- Proof is omitted here
  sorry

end union_A_B_equals_C_l230_230005


namespace find_c_plus_d_l230_230572

theorem find_c_plus_d (a b c d : ℤ) (h1 : a + b = 14) (h2 : b + c = 9) (h3 : a + d = 8) : c + d = 3 := 
by
  sorry

end find_c_plus_d_l230_230572


namespace relatively_prime_days_in_june_l230_230511

def relatively_prime_to (m : ℕ) (n : ℕ) : Prop :=
  Nat.gcd m n = 1

theorem relatively_prime_days_in_june : 
  (finset.filter (λ d : ℕ, relatively_prime_to 6 d) (finset.range 31)).card = 10 :=
by sorry

end relatively_prime_days_in_june_l230_230511


namespace middle_number_between_52_and_certain_number_l230_230759

theorem middle_number_between_52_and_certain_number :
  ∃ n, n > 52 ∧ (∀ k, 52 ≤ k ∧ k ≤ n → ∃ l, k = 52 + l) ∧ (n = 52 + 16) :=
sorry

end middle_number_between_52_and_certain_number_l230_230759


namespace total_sheets_of_paper_l230_230729

theorem total_sheets_of_paper (classes : ℕ) (students_per_class : ℕ) (sheets_per_student : ℕ) 
  (h1 : classes = 4) (h2 : students_per_class = 20) (h3 : sheets_per_student = 5) : 
  (classes * students_per_class * sheets_per_student) = 400 := 
by {
  sorry
}

end total_sheets_of_paper_l230_230729


namespace trajectory_equation_line_slope_is_constant_l230_230149

/-- Definitions for points A, B, and the moving point P -/ 
def pointA : ℝ × ℝ := (-2, 0)
def pointB : ℝ × ℝ := (2, 0)

/-- The condition that the product of the slopes is -3/4 -/
def slope_condition (P : ℝ × ℝ) : Prop :=
  let k_PA := P.2 / (P.1 + 2)
  let k_PB := P.2 / (P.1 - 2)
  k_PA * k_PB = -3 / 4

/-- The trajectory equation as a theorem to be proved -/
theorem trajectory_equation (P : ℝ × ℝ) (h : slope_condition P) : 
  P.2 ≠ 0 ∧ (P.1^2 / 4 + P.2^2 / 3 = 1) := 
sorry

/-- Additional conditions for the line l and points M, N -/ 
def line_l (k m : ℝ) (x : ℝ) : ℝ := k * x + m
def intersect_conditions (P M N : ℝ × ℝ) (k m : ℝ) : Prop :=
  (M.2 = line_l k m M.1) ∧ (N.2 = line_l k m N.1) ∧ 
  (P ≠ M ∧ P ≠ N) ∧ ((P.1 = 1) ∧ (P.2 = 3 / 2)) ∧ 
  (let k_PM := (M.2 - P.2) / (M.1 - P.1)
  let k_PN := (N.2 - P.2) / (N.1 - P.1)
  k_PM + k_PN = 0)

/-- The theorem to prove that the slope of line l is 1/2 -/
theorem line_slope_is_constant (P M N : ℝ × ℝ) (k m : ℝ) 
  (h1 : slope_condition P) 
  (h2 : intersect_conditions P M N k m) : 
  k = 1 / 2 := 
sorry

end trajectory_equation_line_slope_is_constant_l230_230149


namespace yellow_crayons_count_l230_230094

def red_crayons := 14
def blue_crayons := red_crayons + 5
def yellow_crayons := 2 * blue_crayons - 6

theorem yellow_crayons_count : yellow_crayons = 32 := by
  sorry

end yellow_crayons_count_l230_230094


namespace sasha_remainder_l230_230070

theorem sasha_remainder (n a b c d : ℕ) (h1 : n = 102 * a + b) (h2 : n = 103 * c + d) 
  (h3 : a + d = 20) (h_b_range : 0 ≤ b ∧ b ≤ 101) (h_d_range : 0 ≤ d ∧ d ≤ 102) : b = 20 := 
sorry

end sasha_remainder_l230_230070


namespace triangle_theorem_l230_230316

theorem triangle_theorem 
  (A B C : ℝ) 
  (h1 : ∀ A B C : ℝ, ∃ (a b c : ℝ), 
      a^2 - b^2 - c^2 = b * c ∧ sin^2 A - sin^2 B - sin^2 C = sin B * sin C) 
  (h2 : 0 < A ∧ A < π) : 
  (A = 2 * π / 3) ∧ 
  (∀ (BC : ℝ), BC = 3 → ∃ (a b c : ℝ), 
      a + b + c ≤ 3 + 2 * sqrt 3) :=
by
  sorry

end triangle_theorem_l230_230316


namespace p_sufficient_not_necessary_for_q_l230_230154

-- Define the conditions p and q
def p (x : ℝ) : Prop := |x - 1| < 2
def q (x : ℝ) : Prop := x^2 - 5*x - 6 < 0

-- State the theorem that p is a sufficient but not necessary condition for q
theorem p_sufficient_not_necessary_for_q (x : ℝ) :
  (∀ x, p x → q x) ∧ (∃ x, q x ∧ ¬p x) :=
by
  sorry

end p_sufficient_not_necessary_for_q_l230_230154


namespace find_m_l230_230947

variables (a : ℕ → ℝ) (r : ℝ) (m : ℕ)

-- Define the conditions of the problem
def exponential_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * r

def condition_1 (a : ℕ → ℝ) (r : ℝ) : Prop :=
  a 5 * a 6 + a 4 * a 7 = 18

def condition_2 (a : ℕ → ℝ) (m : ℕ) : Prop :=
  a 1 * a m = 9

-- The theorem to prove based on the given conditions
theorem find_m
  (h_exp : exponential_sequence a r)
  (h_r_ne_1 : r ≠ 1)
  (h_cond1 : condition_1 a r)
  (h_cond2 : condition_2 a m) :
  m = 10 :=
sorry

end find_m_l230_230947


namespace cost_of_history_book_l230_230767

theorem cost_of_history_book (total_books : ℕ) (cost_math_book : ℕ) (total_price : ℕ) (num_math_books : ℕ) (num_history_books : ℕ) (cost_history_book : ℕ) 
    (h_books_total : total_books = 90)
    (h_cost_math : cost_math_book = 4)
    (h_total_price : total_price = 396)
    (h_num_math_books : num_math_books = 54)
    (h_num_total_books : num_math_books + num_history_books = total_books)
    (h_total_cost : num_math_books * cost_math_book + num_history_books * cost_history_book = total_price) : cost_history_book = 5 := by 
  sorry

end cost_of_history_book_l230_230767


namespace unit_A_saplings_l230_230876

theorem unit_A_saplings 
  (Y B D J : ℕ)
  (h1 : J = 2 * Y + 20)
  (h2 : J = 3 * B + 24)
  (h3 : J = 5 * D - 45)
  (h4 : J + Y + B + D = 2126) :
  J = 1050 :=
by sorry

end unit_A_saplings_l230_230876


namespace inverse_proportion_first_third_quadrant_l230_230689

theorem inverse_proportion_first_third_quadrant (k : ℝ) :
  (∀ (x : ℝ), x ≠ 0 → ((x > 0 → (2 - k) / x > 0) ∧ (x < 0 → (2 - k) / x < 0))) → k < 2 :=
by
  sorry

end inverse_proportion_first_third_quadrant_l230_230689


namespace alpha_div_beta_is_rational_l230_230862

noncomputable def alpha_is_multiple (α : ℝ) (k : ℕ) : Prop :=
  ∃ k : ℕ, α = k * (2 * Real.pi / 1996)

noncomputable def beta_is_multiple (β : ℝ) (m : ℕ) : Prop :=
  β ≠ 0 ∧ ∃ m : ℕ, β = m * (2 * Real.pi / 1996)

theorem alpha_div_beta_is_rational (α β : ℝ) (k m : ℕ)
  (hα : alpha_is_multiple α k) (hβ : beta_is_multiple β m) :
  ∃ r : ℚ, α / β = r := by
    sorry

end alpha_div_beta_is_rational_l230_230862


namespace find_smallest_x_l230_230275

def smallest_x_satisfying_condition : Prop :=
  ∃ x : ℝ, x > 0 ∧ (⌊x^2⌋ - x * ⌊x⌋ = 8) ∧ x = 89 / 9

theorem find_smallest_x : smallest_x_satisfying_condition :=
begin
  -- proof goes here (not required for this task)
  sorry
end

end find_smallest_x_l230_230275


namespace sasha_remainder_20_l230_230072

theorem sasha_remainder_20
  (n a b c d : ℕ)
  (h1 : n = 102 * a + b)
  (h2 : 0 ≤ b ∧ b ≤ 101)
  (h3 : n = 103 * c + d)
  (h4 : d = 20 - a) :
  b = 20 :=
by
  sorry

end sasha_remainder_20_l230_230072


namespace oil_used_l230_230746

theorem oil_used (total_weight : ℕ) (ratio_oil_peanuts : ℕ) (ratio_total_parts : ℕ) 
  (ratio_peanuts : ℕ) (ratio_parts : ℕ) (peanuts_weight : ℕ) : 
  ratio_oil_peanuts = 2 → 
  ratio_peanuts = 8 → 
  ratio_total_parts = 10 → 
  ratio_parts = 20 →
  peanuts_weight = total_weight / ratio_total_parts →
  total_weight = 20 → 
  2 * peanuts_weight = 4 :=
by sorry

end oil_used_l230_230746


namespace pages_in_second_chapter_l230_230119

theorem pages_in_second_chapter
  (total_pages : ℕ)
  (first_chapter_pages : ℕ)
  (second_chapter_pages : ℕ)
  (h1 : total_pages = 93)
  (h2 : first_chapter_pages = 60)
  (h3: second_chapter_pages = total_pages - first_chapter_pages) :
  second_chapter_pages = 33 :=
by
  sorry

end pages_in_second_chapter_l230_230119


namespace part_a_part_b_l230_230676

-- Definitions and conditions
variable {A B C D E F K L M A₁ : Type}
variable (hABC : AcuteTriangle A B C)
variable (hIncircle : IncircleTouchesTriangle A B C D E F)
variable (hAngleBisector : AngleBisector A D K E L F)
variable (hAltitude : IsAltitude A A₁ B C)
variable (hMidpoint : IsMidpoint M B C)

-- Question 1: Prove that BK and CL are perpendicular to the angle bisector of ∠BAC.
theorem part_a
  (hAcute : AcuteTriangle A B C)
  (hIncircle : IncircleTouchesTriangle A B C D E F)
  (hAngleBisector : AngleBisector A D K E L F)
  (hAltitude : IsAltitude A A₁ B C)
  (hMidpoint : IsMidpoint M B C) :
  Perpendicular (Line B K) (AngleBisector A D K E L F) ∧ Perpendicular (Line C L) (AngleBisector A D K E L F) :=
sorry

-- Question 2: Prove that A₁KML is a cyclic quadrilateral.
theorem part_b
  (hAcute : AcuteTriangle A B C)
  (hIncircle : IncircleTouchesTriangle A B C D E F)
  (hAngleBisector : AngleBisector A D K E L F)
  (hAltitude : IsAltitude A A₁ B C)
  (hMidpoint : IsMidpoint M B C) :
  CyclicQuadrilateral A₁ K M L :=
sorry

end part_a_part_b_l230_230676


namespace range_of_a_l230_230343

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, ¬ (x^2 - 2 * x + 3 ≤ a^2 - 2 * a - 1)) ↔ (-1 < a ∧ a < 3) :=
by
  sorry

end range_of_a_l230_230343


namespace arithmetic_expression_evaluation_l230_230390

theorem arithmetic_expression_evaluation :
  (-12 * 6) - (-4 * -8) + (-15 * -3) - (36 / (-2)) = -77 :=
by
  sorry

end arithmetic_expression_evaluation_l230_230390


namespace product_d_e_l230_230747

-- Define the problem: roots of the polynomial x^2 + x - 2
def roots_of_quadratic : Prop :=
  ∃ α β: ℚ, (α^2 + α - 2 = 0) ∧ (β^2 + β - 2 = 0)

-- Define the condition that both roots are also roots of another polynomial
def roots_of_higher_poly (α β : ℚ) : Prop :=
  (α^7 - 7 * α^3 - 10 = 0 ) ∧ (β^7 - 7 * β^3 - 10 = 0)

-- The final proposition to prove
theorem product_d_e :
  ∀ α β: ℚ, (α^2 + α - 2 = 0) ∧ (β^2 + β - 2 = 0) → (α^7 - 7 * α^3 - 10 = 0) ∧ (β^7 - 7 * β^3 - 10 = 0) → 7 * 10 = 70 := 
by sorry

end product_d_e_l230_230747


namespace team_b_can_serve_on_submarine_l230_230245

   def can_serve_on_submarine (height : ℝ) : Prop := height ≤ 168

   def average_height_condition (avg_height : ℝ) : Prop := avg_height = 166

   def median_height_condition (median_height : ℝ) : Prop := median_height = 167

   def tallest_height_condition (max_height : ℝ) : Prop := max_height = 169

   def mode_height_condition (mode_height : ℝ) : Prop := mode_height = 167

   theorem team_b_can_serve_on_submarine (H : median_height_condition 167) :
     ∀ (h : ℝ), can_serve_on_submarine h :=
   sorry
   
end team_b_can_serve_on_submarine_l230_230245


namespace find_original_height_l230_230233

noncomputable def original_height : ℝ := by
  let H := 102.19
  sorry

lemma ball_rebound (H : ℝ) : 
  (H + 2 * 0.8 * H + 2 * 0.56 * H + 2 * 0.336 * H + 2 * 0.168 * H + 2 * 0.0672 * H + 2 * 0.02016 * H = 500) :=
by
  sorry

theorem find_original_height : original_height = 102.19 :=
by
  have h := ball_rebound original_height
  sorry

end find_original_height_l230_230233


namespace exists_balanced_sequence_set_l230_230543

theorem exists_balanced_sequence_set (n : ℕ) (h_pos : 0 < n) :
  ∃ S : Finset (Fin (2 * n) → Fin 2), 
    S.card ≤ Nat.choose (2 * n) n / (n + 1) ∧
    ∀ a : Fin (2 * n) → Fin 2, 
      (is_balanced a n) → 
      (a ∈ S ∨ ∃ b : Fin (2 * n) → Fin 2, (b ∈ S ∧ is_neighbor a b)) :=
sorry

-- Auxiliary definitions:
def is_balanced (a : Fin (2 * n) → Fin 2) (n : ℕ) : Prop :=
  a.to_multiset.count 0 = n ∧ a.to_multiset.count 1 = n

def is_neighbor (a b : Fin (2 * n) → Fin 2) : Prop :=
  ∃ i j : Fin (2 * n), a j = b i ∧ 
                        ∀ k : Fin (2 * n), k ≠ i → 
                          (a k = b k ∧ (i = j ∨ a i = b j))

end exists_balanced_sequence_set_l230_230543


namespace johns_overall_loss_l230_230859

noncomputable def johns_loss_percentage : ℝ :=
  let cost_A := 1000 * 2
  let cost_B := 1500 * 3
  let cost_C := 2000 * 4
  let discount_A := 0.1
  let discount_B := 0.15
  let discount_C := 0.2
  let cost_A_after_discount := cost_A * (1 - discount_A)
  let cost_B_after_discount := cost_B * (1 - discount_B)
  let cost_C_after_discount := cost_C * (1 - discount_C)
  let total_cost_after_discount := cost_A_after_discount + cost_B_after_discount + cost_C_after_discount
  let import_tax_rate := 0.08
  let import_tax := total_cost_after_discount * import_tax_rate
  let total_cost_incl_tax := total_cost_after_discount + import_tax
  let cost_increase_rate_C := 0.04
  let new_cost_C := 2000 * (4 + 4 * cost_increase_rate_C)
  let adjusted_total_cost := cost_A_after_discount + cost_B_after_discount + new_cost_C
  let total_selling_price := (800 * 3) + (70 * 3 + 1400 * 3.5 + 900 * 5) + (130 * 2.5 + 130 * 3 + 130 * 5)
  let gain_or_loss := total_selling_price - adjusted_total_cost
  let loss_percentage := (gain_or_loss / adjusted_total_cost) * 100
  loss_percentage

theorem johns_overall_loss : abs (johns_loss_percentage + 4.09) < 0.01 := sorry

end johns_overall_loss_l230_230859


namespace necessary_but_not_sufficient_condition_l230_230482

theorem necessary_but_not_sufficient_condition (a : ℝ) :
  (∀ x : ℝ, 1 < x → x < 2 → x^2 - a > 0) → (a < 2) :=
sorry

end necessary_but_not_sufficient_condition_l230_230482


namespace Miss_Adamson_paper_usage_l230_230722

-- Definitions from the conditions
def classes : ℕ := 4
def students_per_class : ℕ := 20
def sheets_per_student : ℕ := 5

-- Total number of students
def total_students : ℕ := classes * students_per_class

-- Total number of sheets of paper
def total_sheets : ℕ := total_students * sheets_per_student

-- The proof problem
theorem Miss_Adamson_paper_usage : total_sheets = 400 :=
by
  -- Proof to be filled in
  sorry

end Miss_Adamson_paper_usage_l230_230722


namespace range_of_a_l230_230979

theorem range_of_a (x a : ℝ) :
  (∀ x, x^2 - 2 * x + 5 ≥ a^2 - 3 * a) ↔ -1 ≤ a ∧ a ≤ 4 :=
sorry

end range_of_a_l230_230979


namespace sufficient_not_necessary_l230_230229

theorem sufficient_not_necessary (p q : Prop) (h : p ∧ q) : (p ∧ q → p ∨ q) ∧ ¬(p ∨ q → p ∧ q) :=
by
  sorry

end sufficient_not_necessary_l230_230229


namespace compute_expression_l230_230391

theorem compute_expression : 7 * (1 / 21) * 42 = 14 :=
by
  sorry

end compute_expression_l230_230391


namespace hyperbola_smaller_focus_l230_230958

noncomputable def smaller_focus_coordinates : ℝ × ℝ :=
  let h := 5
  let k := 20
  let a := 3
  let b := 7
  let c := Real.sqrt (a^2 + b^2)
  (h - c, k)

theorem hyperbola_smaller_focus :
  (smaller_focus_coordinates = (Real.sqrt 58 - 2.62, 20)) :=
by
  sorry

end hyperbola_smaller_focus_l230_230958


namespace equation_of_tangent_line_l230_230685

noncomputable def f (x : ℝ) : ℝ := 2 * x - 2 / x - 2 * Real.log x

theorem equation_of_tangent_line :
  let x0 := 1
      p0 := (1 : ℝ, f 1)
      m := (deriv f 1)
  in m = 2 ∧ p0 = (1, 0) ∧ ∀ x, f 1 + m * (x - 1) = 2 * x - 2 :=
by
  let x0 := 1
  let p0 := (1 : ℝ, f 1)
  let m := (deriv f 1)
  show m = 2 ∧ p0 = (1, 0) ∧ ∀ x, f 1 + m * (x - 1) = 2 * x - 2
  sorry

end equation_of_tangent_line_l230_230685


namespace security_deposit_amount_correct_l230_230181

noncomputable def daily_rate : ℝ := 125.00
noncomputable def pet_fee : ℝ := 100.00
noncomputable def service_cleaning_fee_rate : ℝ := 0.20
noncomputable def security_deposit_rate : ℝ := 0.50
noncomputable def weeks : ℝ := 2
noncomputable def days_per_week : ℝ := 7

noncomputable def number_of_days : ℝ := weeks * days_per_week
noncomputable def total_rental_fee : ℝ := number_of_days * daily_rate
noncomputable def total_rental_fee_with_pet : ℝ := total_rental_fee + pet_fee
noncomputable def service_cleaning_fee : ℝ := service_cleaning_fee_rate * total_rental_fee_with_pet
noncomputable def total_cost : ℝ := total_rental_fee_with_pet + service_cleaning_fee

theorem security_deposit_amount_correct : 
    security_deposit_rate * total_cost = 1110.00 := 
by 
  sorry

end security_deposit_amount_correct_l230_230181


namespace vendor_pepsi_volume_l230_230645

theorem vendor_pepsi_volume 
    (liters_maaza : ℕ)
    (liters_sprite : ℕ)
    (num_cans : ℕ)
    (h1 : liters_maaza = 40)
    (h2 : liters_sprite = 368)
    (h3 : num_cans = 69)
    (volume_pepsi : ℕ)
    (total_volume : ℕ)
    (h4 : total_volume = liters_maaza + liters_sprite + volume_pepsi)
    (h5 : total_volume = num_cans * n)
    (h6 : 408 % num_cans = 0) :
  volume_pepsi = 75 :=
sorry

end vendor_pepsi_volume_l230_230645


namespace intersect_in_third_quadrant_l230_230419

theorem intersect_in_third_quadrant (b : ℝ) : (¬ (∃ x y : ℝ, y = 2*x + 1 ∧ y = 3*x + b ∧ x < 0 ∧ y < 0)) ↔ b > 3 / 2 := sorry

end intersect_in_third_quadrant_l230_230419


namespace gcd_of_256_180_600_l230_230635

theorem gcd_of_256_180_600 : Nat.gcd (Nat.gcd 256 180) 600 = 12 :=
by
  -- The proof would be placed here
  sorry

end gcd_of_256_180_600_l230_230635


namespace expand_polynomial_l230_230397

theorem expand_polynomial :
  (3 * x^2 + 2 * x + 1) * (2 * x^2 + 3 * x + 4) = 6 * x^4 + 13 * x^3 + 20 * x^2 + 11 * x + 4 :=
by
  sorry

end expand_polynomial_l230_230397


namespace find_f_2023_l230_230681

def is_odd_function (g : ℝ → ℝ) := ∀ x, g x = -g (-x)

def condition1 (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, (f (x + 1)) - 2 = -((f (1 - x)) - 2)

def condition2 (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (1 - x) = f (3 + x)

theorem find_f_2023 (f : ℝ → ℝ) (h1 : ∀ x : ℝ, (f (x + 1)) - 2 = -((f (1 - x)) - 2)) 
  (h2 : ∀ x : ℝ, f (1 - x) = f (3 + x)) : 
  f 2023 = 2 :=
sorry

end find_f_2023_l230_230681


namespace area_ratio_l230_230054

theorem area_ratio
  (a b c : ℕ)
  (h1 : 2 * (a + c) = 2 * 2 * (b + c))
  (h2 : a = 2 * b)
  (h3 : c = c) :
  (a * c) = 2 * (b * c) :=
by
  sorry

end area_ratio_l230_230054


namespace find_l_l230_230718

variables (a b c l : ℤ)
def g (x : ℤ) : ℤ := a * x^2 + b * x + c

theorem find_l :
  g a b c 2 = 0 →
  60 < g a b c 6 ∧ g a b c 6 < 70 →
  80 < g a b c 9 ∧ g a b c 9 < 90 →
  6000 * l < g a b c 100 ∧ g a b c 100 < 6000 * (l + 1) →
  l = 5 :=
sorry

end find_l_l230_230718


namespace polynomial_satisfies_conditions_l230_230598

noncomputable def f (x y z : ℝ) : ℝ :=
  (x^2 - y^3) * (y^3 - z^6) * (z^6 - x^2)

theorem polynomial_satisfies_conditions :
  (f x (z^2) y + f x (y^2) z = 0) ∧ (f (z^3) y x + f (x^3) y z = 0) :=
by
  sorry

end polynomial_satisfies_conditions_l230_230598


namespace abs_tan_45_eq_sqrt3_factor_4x2_36_l230_230641

theorem abs_tan_45_eq_sqrt3 : abs (1 - Real.sqrt 3) + Real.tan (Real.pi / 4) = Real.sqrt 3 := 
by 
  sorry

theorem factor_4x2_36 (x : ℝ) : 4 * x ^ 2 - 36 = 4 * (x + 3) * (x - 3) := 
by 
  sorry

end abs_tan_45_eq_sqrt3_factor_4x2_36_l230_230641


namespace ruel_usable_stamps_l230_230468

def totalStamps (books10 books15 books25 books30 : ℕ) (stamps10 stamps15 stamps25 stamps30 : ℕ) : ℕ :=
  books10 * stamps10 + books15 * stamps15 + books25 * stamps25 + books30 * stamps30

def damagedStamps (damaged25 damaged30 : ℕ) : ℕ :=
  damaged25 + damaged30

def usableStamps (books10 books15 books25 books30 stamps10 stamps15 stamps25 stamps30 damaged25 damaged30 : ℕ) : ℕ :=
  totalStamps books10 books15 books25 books30 stamps10 stamps15 stamps25 stamps30 - damagedStamps damaged25 damaged30

theorem ruel_usable_stamps :
  usableStamps 4 6 3 2 10 15 25 30 5 3 = 257 := by
  sorry

end ruel_usable_stamps_l230_230468


namespace chord_length_perpendicular_l230_230106

theorem chord_length_perpendicular 
  (R a b : ℝ)  
  (h1 : a + b = R)
  (h2 : (1 / 2) * Real.pi * R^2 - (1 / 2) * Real.pi * (a^2 + b^2) = 10 * Real.pi) :
  2 * Real.sqrt 10 = 6.32 :=
by 
  sorry

end chord_length_perpendicular_l230_230106


namespace solve_quadratic_1_solve_quadratic_2_solve_quadratic_3_solve_quadratic_4_l230_230742

-- Define the first theorem
theorem solve_quadratic_1 (x : ℝ) : x^2 - 2*x = 0 ↔ x = 0 ∨ x = 2 := 
by {
  -- We assume this proof is provided
  sorry
}

-- Define the second theorem
theorem solve_quadratic_2 (x : ℝ) : 25*x^2 - 36 = 0 ↔ x = 6/5 ∨ x = -6/5 := 
by {
  -- We assume this proof is provided
  sorry
}

-- Define the third theorem
theorem solve_quadratic_3 (x : ℝ) : x^2 + 10*x + 21 = 0 ↔ x = -3 ∨ x = -7 := 
by {
  -- We assume this proof is provided
  sorry
}

-- Define the fourth theorem
theorem solve_quadratic_4 (x : ℝ) : (x-3)^2 + 2*x*(x-3) = 0 ↔ x = 3 ∨ x = 1 := 
by {
  -- We assume this proof is provided
  sorry
}

end solve_quadratic_1_solve_quadratic_2_solve_quadratic_3_solve_quadratic_4_l230_230742


namespace geometric_sequence_a_eq_2_l230_230329

theorem geometric_sequence_a_eq_2 (a : ℝ) (h1 : ¬ a = 0) (h2 : (2 * a) ^ 2 = 8 * a) : a = 2 :=
by {
  sorry -- Proof not required, only the statement.
}

end geometric_sequence_a_eq_2_l230_230329


namespace find_a3_plus_a5_l230_230830

variable (a : ℕ → ℝ)
variable (positive_arith_geom_seq : ∀ n : ℕ, 0 < a n)
variable (h1 : a 1 * a 5 + 2 * a 3 * a 5 + a 3 * a 7 = 25)

theorem find_a3_plus_a5 (positive_arith_geom_seq : ∀ n : ℕ, 0 < a n) (h1 : a 1 * a 5 + 2 * a 3 * a 5 + a 3 * a 7 = 25) :
  a 3 + a 5 = 5 :=
by
  sorry

end find_a3_plus_a5_l230_230830


namespace aaron_earnings_l230_230983

def time_worked_monday := 75 -- in minutes
def time_worked_tuesday := 50 -- in minutes
def time_worked_wednesday := 145 -- in minutes
def time_worked_friday := 30 -- in minutes
def hourly_rate := 3 -- dollars per hour

def total_minutes_worked := 
  time_worked_monday + time_worked_tuesday + 
  time_worked_wednesday + time_worked_friday

def total_hours_worked := total_minutes_worked / 60

def total_earnings := total_hours_worked * hourly_rate

theorem aaron_earnings :
  total_earnings = 15 := by
  sorry

end aaron_earnings_l230_230983


namespace third_recipe_soy_sauce_l230_230474

theorem third_recipe_soy_sauce :
  let bottle_ounces := 16
  let cup_ounces := 8
  let first_recipe_cups := 2
  let second_recipe_cups := 1
  let total_bottles := 3
  (total_bottles * bottle_ounces) / cup_ounces - (first_recipe_cups + second_recipe_cups) = 3 :=
by
  sorry

end third_recipe_soy_sauce_l230_230474


namespace prime_power_minus_l230_230453

theorem prime_power_minus (p : ℕ) (hp : Nat.Prime p) (hps : Nat.Prime (p + 3)) : p ^ 11 - 52 = 1996 := by
  -- this is where the proof would go
  sorry

end prime_power_minus_l230_230453


namespace krishan_money_l230_230780

theorem krishan_money (R G K : ℕ) (h₁ : 7 * G = 17 * R) (h₂ : 7 * K = 17 * G) (h₃ : R = 686) : K = 4046 :=
  by sorry

end krishan_money_l230_230780


namespace find_difference_condition_l230_230013

variable (a b c : ℝ)

theorem find_difference_condition (h1 : (a + b) / 2 = 40) (h2 : (b + c) / 2 = 60) : c - a = 40 := by
  sorry

end find_difference_condition_l230_230013


namespace journey_speed_l230_230794

theorem journey_speed 
  (total_time : ℝ)
  (total_distance : ℝ)
  (second_half_speed : ℝ)
  (first_half_speed : ℝ) :
  total_time = 30 ∧ total_distance = 400 ∧ second_half_speed = 10 ∧
  2 * (total_distance / 2 / second_half_speed) + total_distance / 2 / first_half_speed = total_time →
  first_half_speed = 20 :=
by
  intros hyp
  sorry

end journey_speed_l230_230794


namespace find_m_l230_230835

theorem find_m (m : ℕ) (h1 : Nat.Pos m) (h2 : Nat.lcm 36 m = 180) (h3 : Nat.lcm m 50 = 300) : m = 60 :=
by {
  sorry
}

end find_m_l230_230835


namespace mutually_exclusive_BC_dependent_AC_probability_C_conditional_probability_CA_l230_230852

open ProbabilityTheory

-- Definitions for the problem
variables (BoxA_red BoxA_white BoxB_red BoxB_white : ℕ)
variables (A B C : Event)

-- Conditions
def BoxA := 3 -- 3 red balls in Box A
def BoxA_white := 2 -- 2 white balls in Box A
def BoxB := 2 -- 2 red balls in Box B
def BoxB_white := 3 -- 3 white balls in Box B
def P_A : ℚ := 3 / (3 + 2) -- Probability of drawing a red ball from Box A
def P_B : ℚ := 2 / (3 + 2) -- Probability of drawing a white ball from Box A

def move_ball_event_A := sorry   -- Assuming moving event A is given as an event
def move_ball_event_B := sorry   -- Assuming moving event B is given as an event

-- Event Definitions
def P_C_given_A : ℚ := 3 / (3 + 3) -- P(C|A), probability of drawing red ball from Box B if red ball added
def P_C_given_B : ℚ := 2 / (2 + 3) -- P(C|B), probability of drawing red ball from Box B if white ball added

-- P(C) combining both scenarios
def P_C : ℚ := (P_A * P_C_given_A) + (P_B * P_C_given_B)

-- Conditional Probability P(C|A)
def P_CA : ℚ := (P_A * P_C_given_A) / P_A

-- Rephrasing problem into proofs
theorem mutually_exclusive_BC : ¬(B ∧ C) :=
sorry

theorem dependent_AC : ¬(independent A C) :=
sorry

theorem probability_C : P_C = 13 / 30 :=
sorry

theorem conditional_probability_CA : P_CA = 1 / 2 :=
sorry

end mutually_exclusive_BC_dependent_AC_probability_C_conditional_probability_CA_l230_230852
