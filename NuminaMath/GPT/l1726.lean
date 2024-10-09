import Mathlib

namespace number_of_boxes_of_nectarines_l1726_172613

namespace ProofProblem

/-- Define the given conditions: -/
def crates : Nat := 12
def oranges_per_crate : Nat := 150
def nectarines_per_box : Nat := 30
def total_fruit : Nat := 2280

/-- Define the number of oranges: -/
def total_oranges : Nat := crates * oranges_per_crate

/-- Calculate the number of nectarines: -/
def total_nectarines : Nat := total_fruit - total_oranges

/-- Calculate the number of boxes of nectarines: -/
def boxes_of_nectarines : Nat := total_nectarines / nectarines_per_box

-- Theorem stating that given the conditions, the number of boxes of nectarines is 16.
theorem number_of_boxes_of_nectarines :
  boxes_of_nectarines = 16 := by
  sorry

end ProofProblem

end number_of_boxes_of_nectarines_l1726_172613


namespace line_tangent_through_A_l1726_172606

theorem line_tangent_through_A {A : ℝ × ℝ} (hA : A = (1, 2)) : 
  ∃ m b : ℝ, (b = 2) ∧ (∀ x : ℝ, y = m * x + b) ∧ (∀ y x : ℝ, y^2 = 4*x → y = 2) :=
by
  sorry

end line_tangent_through_A_l1726_172606


namespace sum_last_two_digits_l1726_172670

theorem sum_last_two_digits (h1 : 9 ^ 23 ≡ a [MOD 100]) (h2 : 11 ^ 23 ≡ b [MOD 100]) :
  (a + b) % 100 = 60 := 
  sorry

end sum_last_two_digits_l1726_172670


namespace no_perfect_square_in_seq_l1726_172671

noncomputable def seq : ℕ → ℕ
| 0       => 2
| 1       => 7
| (n + 2) => 4 * seq (n + 1) - seq n

theorem no_perfect_square_in_seq :
  ¬ ∃ (n : ℕ), ∃ (k : ℕ), (seq n) = k * k :=
sorry

end no_perfect_square_in_seq_l1726_172671


namespace prove_correct_operation_l1726_172615

def correct_operation (a b : ℕ) : Prop :=
  (a^3 * a^2 ≠ a^6) ∧
  ((a * b^2)^2 = a^2 * b^4) ∧
  (a^10 / a^5 ≠ a^2) ∧
  (a^2 + a ≠ a^3)

theorem prove_correct_operation (a b : ℕ) : correct_operation a b :=
by {
  sorry
}

end prove_correct_operation_l1726_172615


namespace volume_of_prism_l1726_172687

theorem volume_of_prism :
  ∃ (a b c : ℝ), ab * bc * ac = 762 ∧ (ab = 56) ∧ (bc = 63) ∧ (ac = 72) ∧ (b = 2 * a) :=
sorry

end volume_of_prism_l1726_172687


namespace students_got_on_second_stop_l1726_172668

-- Given conditions translated into definitions and hypotheses
def students_after_first_stop := 39
def students_after_second_stop := 68

-- The proof statement we aim to prove
theorem students_got_on_second_stop : (students_after_second_stop - students_after_first_stop) = 29 := by
  -- Proof goes here
  sorry

end students_got_on_second_stop_l1726_172668


namespace find_m_value_l1726_172678

theorem find_m_value
  (m : ℝ)
  (h1 : 10 - m > 0)
  (h2 : m - 2 > 0)
  (h3 : 2 * Real.sqrt (10 - m - (m - 2)) = 4) :
  m = 4 := by
sorry

end find_m_value_l1726_172678


namespace altitude_length_of_right_triangle_l1726_172685

theorem altitude_length_of_right_triangle 
    (a b c : ℝ) 
    (h1 : a = 8) 
    (h2 : b = 15) 
    (h3 : c = 17) 
    (h4 : a^2 + b^2 = c^2) 
    : (2 * (1/2 * a * b))/c = 120/17 := 
by {
  sorry
}

end altitude_length_of_right_triangle_l1726_172685


namespace symmetric_point_condition_l1726_172620

theorem symmetric_point_condition (a b : ℝ) (l : ℝ → ℝ → Prop) 
  (H_line: ∀ x y, l x y ↔ x + y + 1 = 0)
  (H_symmetric: l a b ∧ l (2*(-a-1) + a) (2*(-b-1) + b))
  : a + b = -1 :=
by 
  sorry

end symmetric_point_condition_l1726_172620


namespace freda_flag_dimensions_l1726_172665

/--  
Given the area of the dove is 192 cm², and the perimeter of the dove consists of quarter-circles or straight lines,
prove that the dimensions of the flag are 24 cm by 16 cm.
-/
theorem freda_flag_dimensions (area_dove : ℝ) (h1 : area_dove = 192) : 
∃ (length width : ℝ), length = 24 ∧ width = 16 := 
sorry

end freda_flag_dimensions_l1726_172665


namespace cos_240_eq_neg_half_l1726_172645

theorem cos_240_eq_neg_half : Real.cos (4 * Real.pi / 3) = -1/2 :=
by
  sorry

end cos_240_eq_neg_half_l1726_172645


namespace evaluate_expression_l1726_172698

theorem evaluate_expression (a x : ℤ) (h : x = a + 5) : 2 * x - a + 4 = a + 14 :=
by
  sorry

end evaluate_expression_l1726_172698


namespace sin_identity_l1726_172608

theorem sin_identity (α : ℝ) (h : Real.sin (π / 6 - α) = 1 / 4) :
  Real.sin (2 * α + π / 6) = 7 / 8 := 
by
  sorry

end sin_identity_l1726_172608


namespace original_profit_margin_l1726_172697

theorem original_profit_margin (x : ℝ) (h1 : x - 0.9 / 0.9 = 12 / 100) : (x - 1) / 1 * 100 = 8 :=
by
  sorry

end original_profit_margin_l1726_172697


namespace gcd_computation_l1726_172654

theorem gcd_computation (a b : ℕ) (h₁ : a = 7260) (h₂ : b = 540) : 
  Nat.gcd a b - 12 + 5 = 53 :=
by
  rw [h₁, h₂]
  sorry

end gcd_computation_l1726_172654


namespace probability_red_in_both_jars_l1726_172635

def original_red_buttons : ℕ := 6
def original_blue_buttons : ℕ := 10
def total_original_buttons : ℕ := original_red_buttons + original_blue_buttons
def remaining_buttons : ℕ := (2 * total_original_buttons) / 3
def moved_buttons : ℕ := total_original_buttons - remaining_buttons
def moved_red_buttons : ℕ := 2
def moved_blue_buttons : ℕ := 3

theorem probability_red_in_both_jars :
  moved_red_buttons = moved_blue_buttons →
  remaining_buttons = 11 →
  (∃ m n : ℚ, m / remaining_buttons = 4 / 11 ∧ n / (moved_red_buttons + moved_blue_buttons) = 2 / 5 ∧ (m / remaining_buttons) * (n / (moved_red_buttons + moved_blue_buttons)) = 8 / 55) :=
by sorry

end probability_red_in_both_jars_l1726_172635


namespace larger_number_is_1891_l1726_172614

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem larger_number_is_1891 :
  ∃ L S : ℕ, (L - S = 1355) ∧ (L = 6 * S + 15) ∧ is_prime (sum_of_digits L) ∧ sum_of_digits L ≠ 12
  :=
sorry

end larger_number_is_1891_l1726_172614


namespace no_ratio_p_squared_l1726_172637

theorem no_ratio_p_squared {p : ℕ} (hp : Nat.Prime p) :
  ∀ l n m : ℕ, 1 ≤ l → (∃ k : ℕ, k = p^l) → ((2 * (n*(n+1)) = (m*(m+1))*p^(2*l)) → false) := 
sorry

end no_ratio_p_squared_l1726_172637


namespace factorize_xy_l1726_172661

theorem factorize_xy (x y : ℕ): xy - x + y - 1 = (x + 1) * (y - 1) :=
by
  sorry

end factorize_xy_l1726_172661


namespace ratio_of_width_to_perimeter_l1726_172639

-- Condition definitions
def length := 22
def width := 13
def perimeter := 2 * (length + width)

-- Statement of the problem in Lean 4
theorem ratio_of_width_to_perimeter : width = 13 ∧ length = 22 → width * 70 = 13 * perimeter :=
by
  sorry

end ratio_of_width_to_perimeter_l1726_172639


namespace find_N_l1726_172673

theorem find_N :
  ∃ N : ℕ, N > 1 ∧
    (1743 % N = 2019 % N) ∧ (2019 % N = 3008 % N) ∧ N = 23 :=
by
  sorry

end find_N_l1726_172673


namespace find_n_and_d_l1726_172667

theorem find_n_and_d (n d : ℕ) (hn_pos : 0 < n) (hd_digit : d < 10)
    (h1 : 3 * n^2 + 2 * n + d = 263)
    (h2 : 3 * n^2 + 2 * n + 4 = 1 * 8^3 + 1 * 8^2 + d * 8 + 1) :
    n + d = 12 := 
sorry

end find_n_and_d_l1726_172667


namespace find_first_term_of_geometric_series_l1726_172643

theorem find_first_term_of_geometric_series 
  (r : ℚ) (S : ℚ) (a : ℚ) 
  (hr : r = -1/3) (hS : S = 9)
  (h_sum_formula : S = a / (1 - r)) : 
  a = 12 := 
by
  sorry

end find_first_term_of_geometric_series_l1726_172643


namespace original_digit_sum_six_and_product_is_1008_l1726_172617

theorem original_digit_sum_six_and_product_is_1008 (x : ℕ) :
  (2 ∣ x / 10) → (4 ∣ x / 10) → 
  (x % 10 + (x / 10) = 6) →
  ((x % 10) * 10 + (x / 10)) * ((x / 10) * 10 + (x % 10)) = 1008 →
  x = 42 ∨ x = 24 :=
by
  intro h1 h2 h3 h4
  sorry


end original_digit_sum_six_and_product_is_1008_l1726_172617


namespace floor_equality_iff_l1726_172601

variable (x : ℝ)

theorem floor_equality_iff :
  (⌊3 * x + 4⌋ = ⌊5 * x - 1⌋) ↔
  (11 / 5 ≤ x ∧ x < 7 / 3) ∨
  (12 / 5 ≤ x ∧ x < 13 / 5) ∨
  (17 / 5 ≤ x ∧ x < 18 / 5) := by
  sorry

end floor_equality_iff_l1726_172601


namespace probability_journalist_A_to_group_A_l1726_172681

open Nat

theorem probability_journalist_A_to_group_A :
  let group_A := 0
  let group_B := 1
  let group_C := 2
  let journalists := [0, 1, 2, 3]  -- four journalists

  -- total number of ways to distribute 4 journalists into 3 groups such that each group has at least one journalist
  let total_ways := 36

  -- number of ways to assign journalist 0 to group A specifically
  let favorable_ways := 12

  -- probability calculation
  ∃ (prob : ℚ), prob = favorable_ways / total_ways ∧ prob = 1 / 3 :=
sorry

end probability_journalist_A_to_group_A_l1726_172681


namespace range_of_y_coordinate_of_C_l1726_172641

-- Define the given parabola equation
def on_parabola (x y : ℝ) : Prop := y^2 = x + 4

-- Define the coordinates for point A
def A : (ℝ × ℝ) := (0, 2)

-- Determine if points B and C lies on the parabola
def point_on_parabola (B C : ℝ × ℝ) : Prop :=
  on_parabola B.1 B.2 ∧ on_parabola C.1 C.2

-- Determine if lines AB and BC are perpendicular
def perpendicular_slopes (B C : ℝ × ℝ) : Prop :=
  let k_AB := (B.2 - A.2) / (B.1 - A.1)
  let k_BC := (C.2 - B.2) / (C.1 - B.1)
  k_AB * k_BC = -1

-- Prove the range for y-coordinate of C
theorem range_of_y_coordinate_of_C (B C : ℝ × ℝ) (h1 : point_on_parabola B C) (h2 : perpendicular_slopes B C) :
  C.2 ≤ 0 ∨ C.2 ≥ 4 := sorry

end range_of_y_coordinate_of_C_l1726_172641


namespace treasure_distribution_l1726_172602

noncomputable def calculate_share (investment total_investment total_value : ℝ) : ℝ :=
  (investment / total_investment) * total_value

theorem treasure_distribution 
  (investment_fonzie investment_aunt_bee investment_lapis investment_skylar investment_orion total_treasure : ℝ)
  (total_investment : ℝ)
  (h : total_investment = investment_fonzie + investment_aunt_bee + investment_lapis + investment_skylar + investment_orion) :
  calculate_share investment_fonzie total_investment total_treasure = 210000 ∧
  calculate_share investment_aunt_bee total_investment total_treasure = 255000 ∧
  calculate_share investment_lapis total_investment total_treasure = 270000 ∧
  calculate_share investment_skylar total_investment total_treasure = 225000 ∧
  calculate_share investment_orion total_investment total_treasure = 240000 :=
by
  sorry

end treasure_distribution_l1726_172602


namespace sweeties_remainder_l1726_172624

theorem sweeties_remainder (m k : ℤ) (h : m = 12 * k + 11) :
  (4 * m) % 12 = 8 :=
by
  -- The proof steps will go here
  sorry

end sweeties_remainder_l1726_172624


namespace john_umbrella_in_car_l1726_172632

variable (UmbrellasInHouse : Nat)
variable (CostPerUmbrella : Nat)
variable (TotalAmountPaid : Nat)

theorem john_umbrella_in_car
  (h1 : UmbrellasInHouse = 2)
  (h2 : CostPerUmbrella = 8)
  (h3 : TotalAmountPaid = 24) :
  (TotalAmountPaid / CostPerUmbrella) - UmbrellasInHouse = 1 := by
  sorry

end john_umbrella_in_car_l1726_172632


namespace regular_polygon_sides_l1726_172662

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, 0 < i → i < n → ∃ θ, θ = 18) :  n = 20 :=
by
    sorry

end regular_polygon_sides_l1726_172662


namespace fireflies_win_l1726_172633

theorem fireflies_win 
  (initial_hornets : ℕ) (initial_fireflies : ℕ) 
  (hornets_scored : ℕ) (fireflies_scored : ℕ) 
  (three_point_baskets : ℕ) (two_point_baskets : ℕ)
  (h1 : initial_hornets = 86)
  (h2 : initial_fireflies = 74)
  (h3 : three_point_baskets = 7)
  (h4 : two_point_baskets = 2)
  (h5 : fireflies_scored = three_point_baskets * 3)
  (h6 : hornets_scored = two_point_baskets * 2)
  : initial_fireflies + fireflies_scored - (initial_hornets + hornets_scored) = 5 := 
sorry

end fireflies_win_l1726_172633


namespace solution_set_inequality_l1726_172646

theorem solution_set_inequality (x : ℝ) :
  ((x^2 - 4) * (x - 6)^2 ≤ 0) ↔ (-2 ≤ x ∧ x ≤ 2 ∨ x = 6) :=
  sorry

end solution_set_inequality_l1726_172646


namespace tom_sara_age_problem_l1726_172625

-- Define the given conditions as hypotheses and variables
variables (t s : ℝ)
variables (h1 : t - 3 = 2 * (s - 3))
variables (h2 : t - 8 = 3 * (s - 8))

-- Lean statement of the problem
theorem tom_sara_age_problem :
  ∃ x : ℝ, (t + x) / (s + x) = 3 / 2 ∧ x = 7 :=
by
  sorry

end tom_sara_age_problem_l1726_172625


namespace juice_drinks_costs_2_l1726_172664

-- Define the conditions and the proof problem
theorem juice_drinks_costs_2 (given_amount : ℕ) (amount_returned : ℕ) 
                            (pizza_cost : ℕ) (number_of_pizzas : ℕ) 
                            (number_of_juice_packs : ℕ) 
                            (total_spent_on_juice : ℕ) (cost_per_pack : ℕ) 
                            (h1 : given_amount = 50) (h2 : amount_returned = 22)
                            (h3 : pizza_cost = 12) (h4 : number_of_pizzas = 2)
                            (h5 : number_of_juice_packs = 2) 
                            (h6 : given_amount - amount_returned - number_of_pizzas * pizza_cost = total_spent_on_juice) 
                            (h7 : total_spent_on_juice / number_of_juice_packs = cost_per_pack) : 
                            cost_per_pack = 2 := by
  sorry

end juice_drinks_costs_2_l1726_172664


namespace kira_memory_space_is_140_l1726_172655

def kira_songs_memory_space 
  (n_m : ℕ) -- number of songs downloaded in the morning
  (n_d : ℕ) -- number of songs downloaded later that day
  (n_n : ℕ) -- number of songs downloaded at night
  (s : ℕ) -- size of each song in MB
  : ℕ := (n_m + n_d + n_n) * s

theorem kira_memory_space_is_140 :
  kira_songs_memory_space 10 15 3 5 = 140 := 
by
  sorry

end kira_memory_space_is_140_l1726_172655


namespace sum_of_roots_l1726_172676

-- Define the polynomial equation
def poly (x : ℝ) := (3 * x + 4) * (x - 5) + (3 * x + 4) * (x - 7)

-- The theorem claiming the sum of the roots
theorem sum_of_roots : 
  (∀ x : ℝ, poly x = 0 → (x = -4/3 ∨ x = 6)) → 
  (∀ s : ℝ, s = -4 / 3 + 6) → s = 14 / 3 :=
by
  sorry

end sum_of_roots_l1726_172676


namespace cost_of_each_skirt_l1726_172682

-- Problem definitions based on conditions
def cost_of_art_supplies : ℕ := 20
def total_expenditure : ℕ := 50
def number_of_skirts : ℕ := 2

-- Proving the cost of each skirt
theorem cost_of_each_skirt (cost_of_each_skirt : ℕ) : 
  number_of_skirts * cost_of_each_skirt + cost_of_art_supplies = total_expenditure → 
  cost_of_each_skirt = 15 := 
by 
  sorry

end cost_of_each_skirt_l1726_172682


namespace geom_sixth_term_is_31104_l1726_172628

theorem geom_sixth_term_is_31104 :
  ∃ (r : ℝ), 4 * r^8 = 39366 ∧ 4 * r^(6-1) = 31104 :=
by
  sorry

end geom_sixth_term_is_31104_l1726_172628


namespace find_interest_rate_l1726_172600

noncomputable def interest_rate_solution : ℝ :=
  let P := 800
  let A := 1760
  let t := 4
  let n := 1
  (A / P) ^ (1 / (n * t)) - 1

theorem find_interest_rate : interest_rate_solution = 0.1892 := 
by
  sorry

end find_interest_rate_l1726_172600


namespace problem_statement_l1726_172689

noncomputable def inequality_not_necessarily_true (a b c : ℝ) :=
  c < b ∧ b < a ∧ a * c < 0

theorem problem_statement (a b c : ℝ) (h : inequality_not_necessarily_true a b c) : ¬ (∃ a b c : ℝ, c < b ∧ b < a ∧ a * c < 0 ∧ ¬ (b^2/c > a^2/c)) :=
by sorry

end problem_statement_l1726_172689


namespace mod_add_l1726_172699

theorem mod_add (n : ℕ) (h : n % 5 = 3) : (n + 2025) % 5 = 3 := by
  sorry

end mod_add_l1726_172699


namespace negative_root_no_positive_l1726_172627

theorem negative_root_no_positive (a : ℝ) : 
  (∃ x : ℝ, x < 0 ∧ |x| = ax + 1) ∧ (¬ ∃ x : ℝ, x > 0 ∧ |x| = ax + 1) → a > -1 :=
by
  sorry

end negative_root_no_positive_l1726_172627


namespace solve_f_inv_zero_l1726_172636

noncomputable def f (a b x : ℝ) : ℝ := 1 / (a * x + b)
noncomputable def f_inv (a b x : ℝ) : ℝ := sorry -- this is where the inverse function definition would go

theorem solve_f_inv_zero (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : f_inv a b 0 = (1 / b) :=
by sorry

end solve_f_inv_zero_l1726_172636


namespace calculate_expression_l1726_172623

theorem calculate_expression : 61 + 5 * 12 / (180 / 3) = 62 := by
  sorry

end calculate_expression_l1726_172623


namespace fallen_tree_trunk_length_l1726_172652

noncomputable def tiger_speed (tiger_length : ℕ) (time_pass_grass : ℕ) : ℕ := tiger_length / time_pass_grass

theorem fallen_tree_trunk_length
  (tiger_length : ℕ)
  (time_pass_grass : ℕ)
  (time_pass_tree : ℕ)
  (speed := tiger_speed tiger_length time_pass_grass) :
  tiger_length = 5 →
  time_pass_grass = 1 →
  time_pass_tree = 5 →
  (speed * time_pass_tree) = 25 :=
by
  intros h_tiger_length h_time_pass_grass h_time_pass_tree
  sorry

end fallen_tree_trunk_length_l1726_172652


namespace smallest_prime_after_six_nonprimes_l1726_172648

-- Define the set of natural numbers and prime numbers
def is_natural (n : ℕ) : Prop := n ≥ 1
def is_prime (n : ℕ) : Prop := 1 < n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n
def is_nonprime (n : ℕ) : Prop := ¬ is_prime n

-- The condition of six consecutive nonprime numbers
def six_consecutive_nonprime (n : ℕ) : Prop := 
  is_nonprime n ∧ 
  is_nonprime (n + 1) ∧ 
  is_nonprime (n + 2) ∧ 
  is_nonprime (n + 3) ∧ 
  is_nonprime (n + 4) ∧ 
  is_nonprime (n + 5)

-- The main theorem stating that 37 is the smallest prime following six consecutive nonprime numbers
theorem smallest_prime_after_six_nonprimes : 
  ∃ (n : ℕ), six_consecutive_nonprime n ∧ is_prime (n + 6) ∧ (∀ m, m < (n + 6) → ¬ is_prime m) :=
sorry

end smallest_prime_after_six_nonprimes_l1726_172648


namespace number_of_teams_l1726_172630

-- Define the statement representing the problem and conditions
theorem number_of_teams (n : ℕ) (h : 2 * n * (n - 1) = 9800) : n = 50 :=
sorry

end number_of_teams_l1726_172630


namespace negation_proposition_equivalence_l1726_172686

theorem negation_proposition_equivalence : 
    (¬ ∃ x_0 : ℝ, (x_0^2 + 1 > 0) ∨ (x_0 > Real.sin x_0)) ↔ 
    (∀ x : ℝ, (x^2 + 1 ≤ 0) ∧ (x ≤ Real.sin x)) :=
by 
    sorry

end negation_proposition_equivalence_l1726_172686


namespace number_of_zeros_l1726_172680

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f' : ℝ → ℝ := sorry
noncomputable def f'' : ℝ → ℝ := sorry

def odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

def conditions (f : ℝ → ℝ) (f'' : ℝ → ℝ) :=
  odd_function f ∧ ∀ x : ℝ, x < 0 → (2 * f x + x * f'' x < x * f x)

theorem number_of_zeros (f : ℝ → ℝ) (f'' : ℝ → ℝ) (h : conditions f f'') :
  ∃! x : ℝ, f x = 0 :=
sorry

end number_of_zeros_l1726_172680


namespace find_perpendicular_line_through_intersection_l1726_172669

theorem find_perpendicular_line_through_intersection : 
  (∃ (M : ℚ × ℚ), 
    (M.1 - 2 * M.2 + 3 = 0) ∧ 
    (2 * M.1 + 3 * M.2 - 8 = 0) ∧ 
    (∃ (c : ℚ), M.1 + 3 * M.2 + c = 0 ∧ 3 * M.1 - M.2 + 1 = 0)) → 
  ∃ (c : ℚ), x + 3 * y + c = 0 :=
sorry

end find_perpendicular_line_through_intersection_l1726_172669


namespace distinct_elements_triangle_not_isosceles_l1726_172638

theorem distinct_elements_triangle_not_isosceles
  {a b c : ℝ} (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c)
  (h4 : a + b > c) (h5 : a + c > b) (h6 : b + c > a) :
  ¬(a = b ∨ b = c ∨ a = c) := by
  sorry

end distinct_elements_triangle_not_isosceles_l1726_172638


namespace max_subset_size_l1726_172692

theorem max_subset_size :
  ∃ S : Finset ℕ, (∀ (x y : ℕ), x ∈ S → y ∈ S → y ≠ 2 * x) →
  S.card = 1335 :=
sorry

end max_subset_size_l1726_172692


namespace total_amount_paid_l1726_172612

-- Definitions based on the conditions.
def cost_per_pizza : ℝ := 12
def delivery_charge : ℝ := 2
def distance_threshold : ℝ := 1000 -- distance in meters
def park_distance : ℝ := 100
def building_distance : ℝ := 2000

def pizzas_at_park : ℕ := 3
def pizzas_at_building : ℕ := 2

-- The proof problem stating the total amount paid to Jimmy.
theorem total_amount_paid :
  let total_pizzas := pizzas_at_park + pizzas_at_building
  let cost_without_delivery := total_pizzas * cost_per_pizza
  let park_charge := if park_distance > distance_threshold then pizzas_at_park * delivery_charge else 0
  let building_charge := if building_distance > distance_threshold then pizzas_at_building * delivery_charge else 0
  let total_cost := cost_without_delivery + park_charge + building_charge
  total_cost = 64 :=
by
  sorry

end total_amount_paid_l1726_172612


namespace sarah_proof_l1726_172695

-- Defining cards and conditions
inductive Card
| P : Card
| A : Card
| C5 : Card
| C4 : Card
| C7 : Card

-- Definition of vowel
def is_vowel : Card → Prop
| Card.P => false
| Card.A => true
| _ => false

-- Definition of prime numbers for the sides
def is_prime : Card → Prop
| Card.C5 => true
| Card.C4 => false
| Card.C7 => true
| _ => false

-- Tom's statement
def toms_statement (c : Card) : Prop :=
is_vowel c → is_prime c

-- Sarah shows Tom was wrong by turning over one card
theorem sarah_proof : ∃ c, toms_statement c = false ∧ c = Card.A :=
sorry

end sarah_proof_l1726_172695


namespace max_sum_of_distances_l1726_172679

theorem max_sum_of_distances (x1 x2 y1 y2 : ℝ)
  (h1 : x1^2 + y1^2 = 1)
  (h2 : x2^2 + y2^2 = 1)
  (h3 : x1 * x2 + y1 * y2 = 1 / 2) :
  (|x1 + y1 - 1| / Real.sqrt 2 + |x2 + y2 - 1| / Real.sqrt 2) ≤ Real.sqrt 2 + Real.sqrt 3 :=
sorry

end max_sum_of_distances_l1726_172679


namespace solve_abs_eq_zero_l1726_172651

theorem solve_abs_eq_zero : ∃ x : ℝ, |5 * x - 3| = 0 ↔ x = 3 / 5 :=
by
  sorry

end solve_abs_eq_zero_l1726_172651


namespace limit_log_div_x_alpha_l1726_172656

open Real

theorem limit_log_div_x_alpha (α : ℝ) (hα : α > 0) :
  (Filter.Tendsto (fun x => (log x) / (x^α)) Filter.atTop (nhds 0)) :=
by
  sorry

end limit_log_div_x_alpha_l1726_172656


namespace MaxCandy_l1726_172693

theorem MaxCandy (frankieCandy : ℕ) (extraCandy : ℕ) (maxCandy : ℕ) 
  (h1 : frankieCandy = 74) (h2 : extraCandy = 18) (h3 : maxCandy = frankieCandy + extraCandy) :
  maxCandy = 92 := 
by
  sorry

end MaxCandy_l1726_172693


namespace how_many_oranges_put_back_l1726_172605

variables (A O x : ℕ)

-- Conditions: prices and initial selection.
def price_apple (A : ℕ) : ℕ := 40 * A
def price_orange (O : ℕ) : ℕ := 60 * O
def total_fruit := 20
def average_price_initial : ℕ := 56 -- Average price in cents

-- Conditions: equation from initial average price.
def total_initial_cost := total_fruit * average_price_initial
axiom initial_cost_eq : price_apple A + price_orange O = total_initial_cost
axiom total_fruit_eq : A + O = total_fruit

-- New conditions: desired average price and number of fruits
def average_price_new : ℕ := 52 -- Average price in cents
axiom new_cost_eq : price_apple A + price_orange (O - x) = (total_fruit - x) * average_price_new

-- The statement to be proven
theorem how_many_oranges_put_back : 40 * A + 60 * (O - 10) = (total_fruit - 10) * 52 → x = 10 :=
sorry

end how_many_oranges_put_back_l1726_172605


namespace weaving_problem_solution_l1726_172696

noncomputable def daily_increase :=
  let a1 := 5
  let n := 30
  let sum_total := 390
  let d := (sum_total - a1 * n) * 2 / (n * (n - 1))
  d

theorem weaving_problem_solution :
  daily_increase = 16 / 29 :=
by
  sorry

end weaving_problem_solution_l1726_172696


namespace countDistinguishedDigitsTheorem_l1726_172657

-- Define a function to count numbers with four distinct digits where leading zeros are allowed
def countDistinguishedDigits : Nat :=
  10 * 9 * 8 * 7

-- State the theorem we need to prove
theorem countDistinguishedDigitsTheorem :
  countDistinguishedDigits = 5040 := 
by
  sorry

end countDistinguishedDigitsTheorem_l1726_172657


namespace Carly_injured_week_miles_l1726_172666

def week1_miles : ℕ := 2
def week2_miles : ℕ := week1_miles * 2 + 3
def week3_miles : ℕ := week2_miles * 9 / 7
def week4_miles : ℕ := week3_miles - 5

theorem Carly_injured_week_miles : week4_miles = 4 :=
  by
    sorry

end Carly_injured_week_miles_l1726_172666


namespace number_of_possible_flags_l1726_172663

-- Define the number of colors available
def num_colors : ℕ := 3

-- Define the number of stripes on the flag
def num_stripes : ℕ := 3

-- Define the total number of possible flags
def total_flags : ℕ := num_colors ^ num_stripes

-- The statement we need to prove
theorem number_of_possible_flags : total_flags = 27 := by
  sorry

end number_of_possible_flags_l1726_172663


namespace ellipse_constants_sum_l1726_172603

/-- Given the center of the ellipse at (h, k) = (3, -5),
    the semi-major axis a = 7,
    and the semi-minor axis b = 4,
    prove that h + k + a + b = 9. -/
theorem ellipse_constants_sum :
  let h := 3
  let k := -5
  let a := 7
  let b := 4
  h + k + a + b = 9 :=
by
  let h := 3
  let k := -5
  let a := 7
  let b := 4
  sorry

end ellipse_constants_sum_l1726_172603


namespace value_of_x_l1726_172610

theorem value_of_x (x y : ℕ) (h1 : x / y = 7 / 3) (h2 : y = 21) : x = 49 := sorry

end value_of_x_l1726_172610


namespace pieces_after_cuts_l1726_172616

theorem pieces_after_cuts (n : ℕ) (h : n = 10) : (n + 1) = 11 := by
  sorry

end pieces_after_cuts_l1726_172616


namespace bus_waiting_probability_l1726_172683

-- Definitions
def arrival_time_range := (0, 90)  -- minutes from 1:00 to 2:30
def bus_wait_time := 20             -- bus waits for 20 minutes

noncomputable def probability_bus_there_when_Laura_arrives : ℚ :=
  let total_area := 90 * 90
  let trapezoid_area := 1400
  let triangle_area := 200
  (trapezoid_area + triangle_area) / total_area

-- Theorem statement
theorem bus_waiting_probability : probability_bus_there_when_Laura_arrives = 16 / 81 := by
  sorry

end bus_waiting_probability_l1726_172683


namespace farm_distance_l1726_172677

theorem farm_distance (a x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (triangle_ineq1 : x + z = 85)
  (triangle_ineq2 : x + y = 4 * z)
  (triangle_ineq3 : z + y = x + a) :
  0 < a ∧ a < 85 ∧
  x = (340 - a) / 6 ∧
  y = (2 * a + 85) / 3 ∧
  z = (170 + a) / 6 :=
sorry

end farm_distance_l1726_172677


namespace gnomes_in_fifth_house_l1726_172653

-- Defining the problem conditions
def num_houses : Nat := 5
def gnomes_per_house : Nat := 3
def total_gnomes : Nat := 20

-- Defining the condition for the first four houses
def gnomes_in_first_four_houses : Nat := 4 * gnomes_per_house

-- Statement of the problem
theorem gnomes_in_fifth_house : 20 - (4 * 3) = 8 := by
  sorry

end gnomes_in_fifth_house_l1726_172653


namespace bucket_weight_one_third_l1726_172644

theorem bucket_weight_one_third 
    (x y c b : ℝ) 
    (h1 : x + 3/4 * y = c)
    (h2 : x + 1/2 * y = b) :
    x + 1/3 * y = 5/3 * b - 2/3 * c :=
by
  sorry

end bucket_weight_one_third_l1726_172644


namespace central_cell_value_l1726_172688

def table (a b c d e f g h i : ℝ) : Prop :=
  (a * b * c = 10) ∧ (d * e * f = 10) ∧ (g * h * i = 10) ∧
  (a * d * g = 10) ∧ (b * e * h = 10) ∧ (c * f * i = 10) ∧
  (a * b * d * e = 3) ∧ (b * c * e * f = 3) ∧ (d * e * g * h = 3) ∧ (e * f * h * i = 3)

theorem central_cell_value (a b c d f g h i e : ℝ) (h_table : table a b c d e f g h i) : 
  e = 0.00081 :=
by sorry

end central_cell_value_l1726_172688


namespace inequality_proof_l1726_172629

variables (a b c d : ℝ)

theorem inequality_proof 
  (h1 : a + b > abs (c - d)) 
  (h2 : c + d > abs (a - b)) : 
  a + c > abs (b - d) := 
sorry

end inequality_proof_l1726_172629


namespace sphere_center_ratio_l1726_172609

/-
Let O be the origin and let (a, b, c) be a fixed point.
A plane with the equation x + 2y + 3z = 6 passes through (a, b, c)
and intersects the x-axis, y-axis, and z-axis at A, B, and C, respectively, all distinct from O.
Let (p, q, r) be the center of the sphere passing through A, B, C, and O.
Prove: a / p + b / q + c / r = 2
-/
theorem sphere_center_ratio (a b c : ℝ) (p q r : ℝ)
  (h_plane : a + 2 * b + 3 * c = 6) 
  (h_p : p = 3)
  (h_q : q = 1.5)
  (h_r : r = 1) :
  a / p + b / q + c / r = 2 :=
by
  sorry

end sphere_center_ratio_l1726_172609


namespace coffee_shrinkage_l1726_172690

theorem coffee_shrinkage :
  let initial_volume_per_cup := 8
  let shrink_factor := 0.5
  let number_of_cups := 5
  let final_volume_per_cup := initial_volume_per_cup * shrink_factor
  let total_remaining_coffee := final_volume_per_cup * number_of_cups
  total_remaining_coffee = 20 :=
by
  -- This is where the steps of the solution would go.
  -- We'll put a sorry here to indicate omission of proof.
  sorry

end coffee_shrinkage_l1726_172690


namespace chocolates_sold_in_second_week_l1726_172634

theorem chocolates_sold_in_second_week
  (c₁ c₂ c₃ c₄ c₅ : ℕ)
  (h₁ : c₁ = 75)
  (h₃ : c₃ = 75)
  (h₄ : c₄ = 70)
  (h₅ : c₅ = 68)
  (h_mean : (c₁ + c₂ + c₃ + c₄ + c₅) / 5 = 71) :
  c₂ = 67 := 
sorry

end chocolates_sold_in_second_week_l1726_172634


namespace second_metal_gold_percentage_l1726_172691

theorem second_metal_gold_percentage (w_final : ℝ) (p_final : ℝ) (w_part : ℝ) (p_part1 : ℝ) (w_part1 : ℝ) (w_part2 : ℝ)
  (h_w_final : w_final = 12.4) (h_p_final : p_final = 0.5) (h_w_part : w_part = 6.2) (h_p_part1 : p_part1 = 0.6)
  (h_w_part1 : w_part1 = 6.2) (h_w_part2 : w_part2 = 6.2) :
  ∃ p_part2 : ℝ, p_part2 = 0.4 :=
by sorry

end second_metal_gold_percentage_l1726_172691


namespace polynomial_root_triples_l1726_172604

theorem polynomial_root_triples (a b c : ℝ) :
  (∀ x : ℝ, x > 0 → (x^4 + a * x^3 + b * x^2 + c * x + b = 0)) ↔ (a, b, c) = (-21, 112, -204) ∨ (a, b, c) = (-12, 48, -80) :=
by
  sorry

end polynomial_root_triples_l1726_172604


namespace price_without_and_with_coupon_l1726_172607

theorem price_without_and_with_coupon
  (commission_rate sale_tax_rate discount_rate : ℝ)
  (cost producer_price shipping_fee: ℝ)
  (S: ℝ)
  (h_commission: commission_rate = 0.20)
  (h_sale_tax: sale_tax_rate = 0.08)
  (h_discount: discount_rate = 0.10)
  (h_producer_price: producer_price = 20)
  (h_shipping_fee: shipping_fee = 5)
  (h_total_cost: cost = producer_price + shipping_fee)
  (h_profit: 0.20 * cost = 5)
  (h_total_earn: cost + sale_tax_rate * S + 5 = 0.80 * S)
  (h_S: S = 41.67):
  S = 41.67 ∧ 0.90 * S = 37.50 :=
by
  sorry

end price_without_and_with_coupon_l1726_172607


namespace average_last_two_numbers_l1726_172684

theorem average_last_two_numbers (a b c d e f g : ℝ) 
  (h1 : (a + b + c + d + e + f + g) / 7 = 63) 
  (h2 : (a + b + c) / 3 = 58) 
  (h3 : (d + e) / 2 = 70) :
  ((f + g) / 2) = 63.5 := 
sorry

end average_last_two_numbers_l1726_172684


namespace perpendicular_lines_l1726_172674

theorem perpendicular_lines :
  ∃ m₁ m₄, (m₁ : ℚ) * (m₄ : ℚ) = -1 ∧
  (∀ x y : ℚ, 4 * y - 3 * x = 16 → y = m₁ * x + 4) ∧
  (∀ x y : ℚ, 3 * y + 4 * x = 15 → y = m₄ * x + 5) :=
by sorry

end perpendicular_lines_l1726_172674


namespace inequality_ab_leq_a_b_l1726_172694

theorem inequality_ab_leq_a_b (a b : ℝ) (x : ℝ) (h_a : 0 < a) (h_b : 0 < b) :
  a * b ≤ (a * (Real.sin x) ^ 2 + b * (Real.cos x) ^ 2) * (a * (Real.cos x) ^ 2 + b * (Real.sin x) ^ 2)
  ∧ (a * (Real.sin x) ^ 2 + b * (Real.cos x) ^ 2) * (a * (Real.cos x) ^ 2 + b * (Real.sin x) ^ 2) ≤ (a + b)^2 / 4 := 
sorry

end inequality_ab_leq_a_b_l1726_172694


namespace original_price_of_wand_l1726_172622

theorem original_price_of_wand (x : ℝ) (h : x / 8 = 12) : x = 96 :=
by
  sorry

end original_price_of_wand_l1726_172622


namespace min_acute_triangles_for_isosceles_l1726_172626

noncomputable def isosceles_triangle_acute_division : ℕ :=
  sorry

theorem min_acute_triangles_for_isosceles {α : ℝ} (hα : α = 108) (isosceles : ∀ β γ : ℝ, β = γ) :
  isosceles_triangle_acute_division = 7 :=
sorry

end min_acute_triangles_for_isosceles_l1726_172626


namespace cube_sum_identity_l1726_172618

theorem cube_sum_identity (r : ℝ) (h : (r + 1/r)^2 = 5) : r^3 + 1/r^3 = 2 * Real.sqrt 5 ∨ r^3 + 1/r^3 = -2 * Real.sqrt 5 := by
  sorry

end cube_sum_identity_l1726_172618


namespace stock_yield_percentage_l1726_172675

noncomputable def FaceValue : ℝ := 100
noncomputable def AnnualYield : ℝ := 0.20 * FaceValue
noncomputable def MarketPrice : ℝ := 166.66666666666669
noncomputable def ExpectedYieldPercentage : ℝ := 12

theorem stock_yield_percentage :
  (AnnualYield / MarketPrice) * 100 = ExpectedYieldPercentage :=
by
  -- given conditions directly from the problem
  have h1 : FaceValue = 100 := rfl
  have h2 : AnnualYield = 0.20 * FaceValue := rfl
  have h3 : MarketPrice = 166.66666666666669 := rfl
  
  -- we are proving that the yield percentage is 12%
  sorry

end stock_yield_percentage_l1726_172675


namespace find_x2_plus_y2_l1726_172619

theorem find_x2_plus_y2 
  (x y : ℕ) (hx : 0 < x) (hy : 0 < y)
  (h1 : x * y + x + y = 83)
  (h2 : x^2 * y + x * y^2 = 1056) :
  x^2 + y^2 = 458 :=
by
  sorry

end find_x2_plus_y2_l1726_172619


namespace problem_l1726_172649

-- Definition for condition 1
def condition1 (uniform_band : Prop) (appropriate_model : Prop) := 
  uniform_band → appropriate_model

-- Definition for condition 2
def condition2 (smaller_residual : Prop) (better_fit : Prop) :=
  smaller_residual → better_fit

-- Formal statement of the problem
theorem problem (uniform_band appropriate_model smaller_residual better_fit : Prop)
  (h1 : condition1 uniform_band appropriate_model)
  (h2 : condition2 smaller_residual better_fit)
  (h3 : uniform_band ∧ smaller_residual) :
  appropriate_model ∧ better_fit :=
  sorry

end problem_l1726_172649


namespace no_perfect_square_l1726_172660

theorem no_perfect_square (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) 
  (h : ∃ (a : ℕ), p + q^2 = a^2) : ∀ (n : ℕ), n > 0 → ¬ (∃ (b : ℕ), p^2 + q^n = b^2) := 
by
  sorry

end no_perfect_square_l1726_172660


namespace boxes_sold_l1726_172659

def case_size : ℕ := 12
def remaining_boxes : ℕ := 7

theorem boxes_sold (sold_boxes : ℕ) : ∃ n : ℕ, sold_boxes = n * case_size + remaining_boxes :=
sorry

end boxes_sold_l1726_172659


namespace boat_speed_in_still_water_l1726_172621

theorem boat_speed_in_still_water (V_b : ℝ) : 
  (∀ t : ℝ, t = 26 / (V_b + 6) → t = 14 / (V_b - 6)) → V_b = 20 :=
by
  sorry

end boat_speed_in_still_water_l1726_172621


namespace trigonometric_identity_l1726_172640

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 3) : 
    (2 * Real.sin α - Real.cos α) / (Real.sin α + 3 * Real.cos α) = 5 / 6 :=
by
  sorry

end trigonometric_identity_l1726_172640


namespace each_piece_of_paper_weight_l1726_172658

noncomputable def paper_weight : ℚ :=
 sorry

theorem each_piece_of_paper_weight (w : ℚ) (n : ℚ) (envelope_weight : ℚ) (stamps_needed : ℚ) (paper_pieces : ℚ) :
  paper_pieces = 8 →
  envelope_weight = 2/5 →
  stamps_needed = 2 →
  n = paper_pieces * w + envelope_weight →
  n ≤ stamps_needed →
  w = 1/5 :=
by sorry

end each_piece_of_paper_weight_l1726_172658


namespace cube_of_product_of_ab_l1726_172642

theorem cube_of_product_of_ab (a b c : ℕ) (h1 : a * b * c = 180) (h2 : 0 < a) (h3 : 0 < b) (h4 : 0 < c) : (a * b) ^ 3 = 216 := 
sorry

end cube_of_product_of_ab_l1726_172642


namespace find_minimum_value_l1726_172672

theorem find_minimum_value (c : ℝ) : 
  (∀ c : ℝ, (c = -12) ↔ (∀ d : ℝ, (1 / 3) * d^2 + 8 * d - 7 ≥ (1 / 3) * (-12)^2 + 8 * (-12) - 7)) :=
sorry

end find_minimum_value_l1726_172672


namespace zero_integers_in_range_such_that_expr_is_perfect_square_l1726_172650

theorem zero_integers_in_range_such_that_expr_is_perfect_square :
  (∃ n : ℕ, 5 ≤ n ∧ n ≤ 15 ∧ ∃ m : ℕ, 2 * n ^ 2 + n + 2 = m ^ 2) → False :=
by sorry

end zero_integers_in_range_such_that_expr_is_perfect_square_l1726_172650


namespace subtraction_of_largest_three_digit_from_smallest_five_digit_l1726_172647

def largest_three_digit_number : ℕ := 999
def smallest_five_digit_number : ℕ := 10000

theorem subtraction_of_largest_three_digit_from_smallest_five_digit :
  smallest_five_digit_number - largest_three_digit_number = 9001 :=
by
  sorry

end subtraction_of_largest_three_digit_from_smallest_five_digit_l1726_172647


namespace pastries_calculation_l1726_172611

theorem pastries_calculation 
    (G : ℕ) (C : ℕ) (P : ℕ) (F : ℕ)
    (hG : G = 30) 
    (hC : C = G - 5)
    (hP : P = G - 5)
    (htotal : C + P + F + G = 97) :
    C - F = 8 ∧ P - F = 8 :=
by
  sorry

end pastries_calculation_l1726_172611


namespace initial_investment_l1726_172631

theorem initial_investment
  (P r : ℝ)
  (h1 : P + (P * r * 2) / 100 = 600)
  (h2 : P + (P * r * 7) / 100 = 850) :
  P = 500 :=
sorry

end initial_investment_l1726_172631
