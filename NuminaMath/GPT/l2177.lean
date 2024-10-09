import Mathlib

namespace grace_can_reach_target_sum_l2177_217773

theorem grace_can_reach_target_sum :
  ∃ (half_dollars dimes pennies : ℕ),
    half_dollars ≤ 5 ∧ dimes ≤ 20 ∧ pennies ≤ 25 ∧
    (5 * 50 + 13 * 10 + 5) = 385 :=
sorry

end grace_can_reach_target_sum_l2177_217773


namespace smallest_multiple_1_through_10_l2177_217787

theorem smallest_multiple_1_through_10 : ∃ n : ℕ, (∀ x : ℕ, 1 ≤ x ∧ x ≤ 10 → x ∣ n) ∧ (∀ m : ℕ, (∀ x : ℕ, 1 ≤ x ∧ x ≤ 10 → x ∣ m) → n ≤ m) ∧ n = 27720 :=
by sorry

end smallest_multiple_1_through_10_l2177_217787


namespace cubic_identity_l2177_217705

theorem cubic_identity (x y z : ℝ) 
  (h1 : x + y + z = 12) 
  (h2 : xy + xz + yz = 30) : 
  x^3 + y^3 + z^3 - 3 * x * y * z = 648 :=
sorry

end cubic_identity_l2177_217705


namespace squirrel_nuts_l2177_217760

theorem squirrel_nuts :
  ∃ (a b c d : ℕ), 103 ≤ a ∧ 103 ≤ b ∧ 103 ≤ c ∧ 103 ≤ d ∧
                   a ≥ b ∧ a ≥ c ∧ a ≥ d ∧
                   a + b + c + d = 2020 ∧
                   b + c = 1277 ∧
                   a = 640 :=
by {
  -- proof goes here
  sorry
}

end squirrel_nuts_l2177_217760


namespace percent_parrots_among_non_pelicans_l2177_217715

theorem percent_parrots_among_non_pelicans 
  (parrots_percent pelicans_percent owls_percent sparrows_percent : ℝ) 
  (H1 : parrots_percent = 40) 
  (H2 : pelicans_percent = 20) 
  (H3 : owls_percent = 15) 
  (H4 : sparrows_percent = 100 - parrots_percent - pelicans_percent - owls_percent)
  (H5 : pelicans_percent / 100 < 1) :
  parrots_percent / (100 - pelicans_percent) * 100 = 50 :=
by sorry

end percent_parrots_among_non_pelicans_l2177_217715


namespace bathroom_square_footage_l2177_217742

theorem bathroom_square_footage 
  (tiles_width : ℕ) (tiles_length : ℕ) (tile_size_inch : ℕ)
  (inch_to_foot : ℕ) 
  (h_width : tiles_width = 10) 
  (h_length : tiles_length = 20)
  (h_tile_size : tile_size_inch = 6)
  (h_inch_to_foot : inch_to_foot = 12) :
  let tile_size_foot : ℚ := tile_size_inch / inch_to_foot
  let width_foot : ℚ := tiles_width * tile_size_foot
  let length_foot : ℚ := tiles_length * tile_size_foot
  let area : ℚ := width_foot * length_foot
  area = 50 := 
by
  sorry

end bathroom_square_footage_l2177_217742


namespace remainder_8_times_10_pow_18_plus_1_pow_18_div_9_l2177_217724

theorem remainder_8_times_10_pow_18_plus_1_pow_18_div_9 :
  (8 * 10^18 + 1^18) % 9 = 0 := 
by 
  sorry

end remainder_8_times_10_pow_18_plus_1_pow_18_div_9_l2177_217724


namespace range_of_k_l2177_217756

def quadratic_discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

theorem range_of_k (k : ℝ) :
  (∀ x : ℝ, (k - 1) * x^2 + (k - 1) * x + 2 > 0) ↔ 1 ≤ k ∧ k < 9 :=
by
  sorry

end range_of_k_l2177_217756


namespace vertices_after_cut_off_four_corners_l2177_217793

-- Definitions for the conditions
def regular_tetrahedron.num_vertices : ℕ := 4

def new_vertices_per_cut : ℕ := 3

def total_vertices_after_cut : ℕ := 
  regular_tetrahedron.num_vertices + regular_tetrahedron.num_vertices * new_vertices_per_cut

-- The theorem to prove the question
theorem vertices_after_cut_off_four_corners :
  total_vertices_after_cut = 12 :=
by
  -- sorry is used to skip the proof steps, as per instructions
  sorry

end vertices_after_cut_off_four_corners_l2177_217793


namespace initial_birds_correct_l2177_217739

def flown_away : ℝ := 8.0
def left_on_fence : ℝ := 4.0
def initial_birds : ℝ := flown_away + left_on_fence

theorem initial_birds_correct : initial_birds = 12.0 := by
  sorry

end initial_birds_correct_l2177_217739


namespace find_number_l2177_217728

-- Define the number x and state the condition 55 + x = 88
def x := 33

-- State the theorem to be proven: if 55 + x = 88, then x = 33
theorem find_number (h : 55 + x = 88) : x = 33 :=
by
  sorry

end find_number_l2177_217728


namespace isabel_pictures_l2177_217704

theorem isabel_pictures
  (phone_pics : ℕ)
  (camera_pics : ℕ)
  (total_albums : ℕ)
  (h_phone_pics : phone_pics = 2)
  (h_camera_pics : camera_pics = 4)
  (h_total_albums : total_albums = 3) :
  (phone_pics + camera_pics) / total_albums = 2 :=
by
  sorry

end isabel_pictures_l2177_217704


namespace johns_groceries_cost_l2177_217752

noncomputable def calculate_total_cost : ℝ := 
  let bananas_cost := 6 * 2
  let bread_cost := 2 * 3
  let butter_cost := 3 * 5
  let cereal_cost := 4 * (6 - 0.25 * 6)
  let subtotal := bananas_cost + bread_cost + butter_cost + cereal_cost
  if subtotal >= 50 then
    subtotal - 10
  else
    subtotal

-- The statement to prove
theorem johns_groceries_cost : calculate_total_cost = 41 := by
  sorry

end johns_groceries_cost_l2177_217752


namespace positive_integer_fraction_l2177_217763

theorem positive_integer_fraction (p : ℕ) (h1 : p > 0) (h2 : (3 * p + 25) / (2 * p - 5) > 0) :
  3 ≤ p ∧ p ≤ 35 :=
by
  sorry

end positive_integer_fraction_l2177_217763


namespace tan_x_eq_sqrt3_l2177_217790

theorem tan_x_eq_sqrt3 (x : Real) (h : Real.sin (x + 20 * Real.pi / 180) = Real.cos (x + 10 * Real.pi / 180) + Real.cos (x - 10 * Real.pi / 180)) : Real.tan x = Real.sqrt 3 := 
by
  sorry

end tan_x_eq_sqrt3_l2177_217790


namespace John_l2177_217713

/-- Assume Grant scored 10 points higher on his math test than John.
John received a certain ratio of points as Hunter who scored 45 points on his math test.
Grant's test score was 100. -/
theorem John's_points_to_Hunter's_points_ratio 
  (Grant John Hunter : ℕ) 
  (h1 : Grant = John + 10)
  (h2 : Hunter = 45)
  (h_grant_score : Grant = 100) : 
  (John : ℚ) / (Hunter : ℚ) = 2 / 1 :=
sorry

end John_l2177_217713


namespace charlene_gave_18_necklaces_l2177_217734

theorem charlene_gave_18_necklaces
  (initial_necklaces : ℕ) (sold_necklaces : ℕ) (left_necklaces : ℕ)
  (h1 : initial_necklaces = 60)
  (h2 : sold_necklaces = 16)
  (h3 : left_necklaces = 26) :
  initial_necklaces - sold_necklaces - left_necklaces = 18 :=
by
  sorry

end charlene_gave_18_necklaces_l2177_217734


namespace solve_floor_equation_l2177_217761

noncomputable def x_solution_set : Set ℚ := 
  {x | x = 1 ∨ ∃ k : ℕ, 16 ≤ k ∧ k ≤ 22 ∧ x = (k : ℚ)/23 }

theorem solve_floor_equation (x : ℚ) (hx : x ∈ x_solution_set) : 
  (⌊20*x + 23⌋ : ℚ) = 20 + 23*x :=
sorry

end solve_floor_equation_l2177_217761


namespace find_num_apples_l2177_217774

def num_apples (A P : ℕ) : Prop :=
  P = (3 * A) / 5 ∧ A + P = 240

theorem find_num_apples (A : ℕ) (P : ℕ) :
  num_apples A P → A = 150 :=
by
  intros h
  -- sorry for proof
  sorry

end find_num_apples_l2177_217774


namespace wrapping_paper_cost_l2177_217718
noncomputable def cost_per_roll (shirt_boxes XL_boxes: ℕ) (cost_total: ℝ) : ℝ :=
  let rolls_for_shirts := shirt_boxes / 5
  let rolls_for_xls := XL_boxes / 3
  let total_rolls := rolls_for_shirts + rolls_for_xls
  cost_total / total_rolls

theorem wrapping_paper_cost : cost_per_roll 20 12 32 = 4 :=
by
  sorry

end wrapping_paper_cost_l2177_217718


namespace union_of_sets_l2177_217767

-- Definitions based on conditions
def A : Set ℕ := {2, 3}
def B (a : ℕ) : Set ℕ := {1, a}
def condition (a : ℕ) : Prop := A ∩ (B a) = {2}

-- Main theorem to be proven
theorem union_of_sets (a : ℕ) (h : condition a) : A ∪ (B a) = {1, 2, 3} :=
sorry

end union_of_sets_l2177_217767


namespace a8_equals_two_or_minus_two_l2177_217732

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n m : ℕ, ∃ r : ℝ, a (n + m) = a n * a m / a 0

theorem a8_equals_two_or_minus_two (a : ℕ → ℝ) 
    (h_geom : geometric_sequence a)
    (h_roots : ∃ x y : ℝ, x^2 - 8 * x + 4 = 0 ∧ y^2 - 8 * y + 4 = 0 ∧ a 6 = x ∧ a 10 = y) :
  a 8 = 2 ∨ a 8 = -2 :=
by
  sorry

end a8_equals_two_or_minus_two_l2177_217732


namespace six_to_2049_not_square_l2177_217722

theorem six_to_2049_not_square
  (h1: ∃ x: ℝ, 1^2048 = x^2)
  (h2: ∃ x: ℝ, 2^2050 = x^2)
  (h3: ¬∃ x: ℝ, 6^2049 = x^2)
  (h4: ∃ x: ℝ, 4^2051 = x^2)
  (h5: ∃ x: ℝ, 5^2052 = x^2):
  ¬∃ y: ℝ, y^2 = 6^2049 := 
by sorry

end six_to_2049_not_square_l2177_217722


namespace average_rst_l2177_217762

theorem average_rst (r s t : ℝ) (h : (5 / 4) * (r + s + t - 2) = 15) : (r + s + t) / 3 = 14 / 3 :=
by
  sorry

end average_rst_l2177_217762


namespace unique_non_congruent_rectangle_with_conditions_l2177_217757

theorem unique_non_congruent_rectangle_with_conditions :
  ∃! (w h : ℕ), 2 * (w + h) = 80 ∧ w * h = 400 :=
by
  sorry

end unique_non_congruent_rectangle_with_conditions_l2177_217757


namespace most_stable_athlete_l2177_217740

theorem most_stable_athlete (s2_A s2_B s2_C s2_D : ℝ) 
  (hA : s2_A = 0.5) 
  (hB : s2_B = 0.5) 
  (hC : s2_C = 0.6) 
  (hD : s2_D = 0.4) :
  s2_D < s2_A ∧ s2_D < s2_B ∧ s2_D < s2_C :=
by
  sorry

end most_stable_athlete_l2177_217740


namespace min_value_of_fraction_l2177_217759

theorem min_value_of_fraction (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 2 * x + 3 * y = 3) : 
  (3 / x + 2 / y) = 8 :=
sorry

end min_value_of_fraction_l2177_217759


namespace triangles_formed_l2177_217748

-- Define the combinatorial function for binomial coefficients.
def binom (n k : ℕ) : ℕ :=
  if h : k ≤ n then Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k)) else 0

-- Given conditions
def points_on_first_line := 6
def points_on_second_line := 8

-- Number of triangles calculation
def total_triangles :=
  binom points_on_first_line 2 * binom points_on_second_line 1 +
  binom points_on_first_line 1 * binom points_on_second_line 2

-- The final theorem to prove
theorem triangles_formed : total_triangles = 288 :=
by
  sorry

end triangles_formed_l2177_217748


namespace prime_factorial_division_l2177_217794

theorem prime_factorial_division (p k n : ℕ) (hp : Prime p) (h : p^k ∣ n!) : (p!)^k ∣ n! :=
sorry

end prime_factorial_division_l2177_217794


namespace average_score_for_girls_at_both_schools_combined_l2177_217700

/-
  The following conditions are given:
  - Average score for boys at Lincoln HS = 75
  - Average score for boys at Monroe HS = 85
  - Average score for boys at both schools combined = 82
  - Average score for girls at Lincoln HS = 78
  - Average score for girls at Monroe HS = 92
  - Average score for boys and girls combined at Lincoln HS = 76
  - Average score for boys and girls combined at Monroe HS = 88

  The goal is to prove that the average score for the girls at both schools combined is 89.
-/
theorem average_score_for_girls_at_both_schools_combined 
  (L l M m : ℕ)
  (h1 : (75 * L + 78 * l) / (L + l) = 76)
  (h2 : (85 * M + 92 * m) / (M + m) = 88)
  (h3 : (75 * L + 85 * M) / (L + M) = 82)
  : (78 * l + 92 * m) / (l + m) = 89 := 
sorry

end average_score_for_girls_at_both_schools_combined_l2177_217700


namespace fraction_eaten_correct_l2177_217744

def initial_nuts : Nat := 30
def nuts_left : Nat := 5
def eaten_nuts : Nat := initial_nuts - nuts_left
def fraction_eaten : Rat := eaten_nuts / initial_nuts

theorem fraction_eaten_correct : fraction_eaten = 5 / 6 := by
  sorry

end fraction_eaten_correct_l2177_217744


namespace vector_sum_l2177_217738

def v1 : ℤ × ℤ := (5, -3)
def v2 : ℤ × ℤ := (-2, 4)
def scalar : ℤ := 3

theorem vector_sum : 
  (v1.1 + scalar * v2.1, v1.2 + scalar * v2.2) = (-1, 9) := 
by 
  sorry

end vector_sum_l2177_217738


namespace even_function_value_l2177_217779

-- Define the function condition
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

-- Define the main problem with given conditions
theorem even_function_value (f : ℝ → ℝ) (h1 : is_even_function f) (h2 : ∀ x : ℝ, x < 0 → f x = x * (x + 1)) 
  (x : ℝ) (hx : x > 0) : f x = x * (x - 1) :=
  sorry

end even_function_value_l2177_217779


namespace price_of_stock_l2177_217751

-- Defining the conditions
def income : ℚ := 650
def dividend_rate : ℚ := 10
def investment : ℚ := 6240

-- Defining the face value calculation from income and dividend rate
def face_value (i : ℚ) (d_rate : ℚ) : ℚ := (i * 100) / d_rate

-- Calculating the price of the stock
def stock_price (inv : ℚ) (fv : ℚ) : ℚ := (inv / fv) * 100

-- Main theorem to be proved
theorem price_of_stock : stock_price investment (face_value income dividend_rate) = 96 := by
  sorry

end price_of_stock_l2177_217751


namespace inequality_false_l2177_217710

variable {x y w : ℝ}

theorem inequality_false (hx : x > y) (hy : y > 0) (hw : w ≠ 0) : ¬(x^2 * w > y^2 * w) :=
by {
  sorry -- You could replace this "sorry" with a proper proof.
}

end inequality_false_l2177_217710


namespace geom_seq_frac_l2177_217709

noncomputable def geom_seq_sum (a1 : ℕ) (q : ℕ) (n : ℕ) : ℕ :=
  a1 * (1 - q ^ n) / (1 - q)

theorem geom_seq_frac (a1 q : ℕ) (hq : q > 1) (h_sum : a1 * (q ^ 3 + q ^ 6 + 1 + q + q ^ 2 + q ^ 5) = 20)
  (h_prod : a1 ^ 7 * q ^ (3 + 6) = 64) :
  geom_seq_sum a1 q 6 / geom_seq_sum a1 q 9 = 5 / 21 :=
by
  sorry

end geom_seq_frac_l2177_217709


namespace inequality_example_l2177_217736

theorem inequality_example (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / Real.sqrt (a ^ 2 + 8 * b * c)) + (b / Real.sqrt (b ^ 2 + 8 * c * a)) + (c / Real.sqrt (c ^ 2 + 8 * a * b)) ≥ 1 :=
by
  sorry

end inequality_example_l2177_217736


namespace student_solved_correctly_l2177_217792

theorem student_solved_correctly (x : ℕ) :
  (x + 2 * x = 36) → x = 12 :=
by
  intro h
  sorry

end student_solved_correctly_l2177_217792


namespace periodic_modulo_h_l2177_217777

open Nat

-- Defining the binomial coefficient
def binom (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Defining the sequence as per the problem
def x_seq (n : ℕ) : ℕ :=
  binom (2 * n) n

-- The main theorem stating the required condition
theorem periodic_modulo_h (h : ℕ) (h_gt_one : h > 1) :
  (∃ N, ∀ n ≥ N, x_seq n % h = x_seq (n + 1) % h) ↔ h = 2 :=
by
  sorry

end periodic_modulo_h_l2177_217777


namespace molecular_weight_of_moles_l2177_217721

-- Approximate atomic weights
def atomic_weight_N := 14.01
def atomic_weight_O := 16.00

-- Molecular weight of N2O3
def molecular_weight_N2O3 := (2 * atomic_weight_N) + (3 * atomic_weight_O)

-- Given the total molecular weight of some moles of N2O3
def total_molecular_weight : ℝ := 228

-- We aim to prove that the total molecular weight of some moles of N2O3 equals 228 g
theorem molecular_weight_of_moles (h: molecular_weight_N2O3 ≠ 0) :
  total_molecular_weight = 228 := by
  sorry

end molecular_weight_of_moles_l2177_217721


namespace roberto_current_salary_l2177_217780

theorem roberto_current_salary (starting_salary current_salary : ℝ) (h₀ : starting_salary = 80000)
(h₁ : current_salary = (starting_salary * 1.4) * 1.2) : 
current_salary = 134400 := by
  sorry

end roberto_current_salary_l2177_217780


namespace smallest_positive_integer_y_l2177_217765

theorem smallest_positive_integer_y
  (y : ℕ)
  (h_pos : 0 < y)
  (h_ineq : y^3 > 80) :
  y = 5 :=
sorry

end smallest_positive_integer_y_l2177_217765


namespace complex_multiplication_l2177_217743

def i := Complex.I

theorem complex_multiplication (i := Complex.I) : (-1 + i) * (2 - i) = -1 + 3 * i := 
by 
    -- The actual proof steps would go here.
    sorry

end complex_multiplication_l2177_217743


namespace domain_of_function_l2177_217789

theorem domain_of_function (x : ℝ) : (|x - 2| + |x + 2| ≠ 0) := 
sorry

end domain_of_function_l2177_217789


namespace correct_option_l2177_217745

-- Define the operations as functions to be used in the Lean statement.
def optA : ℕ := 3 + 5 * 7 + 9
def optB : ℕ := 3 + 5 + 7 * 9
def optC : ℕ := 3 * 5 * 7 - 9
def optD : ℕ := 3 * 5 * 7 + 9
def optE : ℕ := 3 * 5 + 7 * 9

-- The theorem to prove that the correct option is (E).
theorem correct_option : optE = 78 ∧ optA ≠ 78 ∧ optB ≠ 78 ∧ optC ≠ 78 ∧ optD ≠ 78 := by {
  sorry
}

end correct_option_l2177_217745


namespace total_cost_of_long_distance_bill_l2177_217727

theorem total_cost_of_long_distance_bill
  (monthly_fee : ℝ := 5)
  (cost_per_minute : ℝ := 0.25)
  (minutes_billed : ℝ := 28.08) :
  monthly_fee + cost_per_minute * minutes_billed = 12.02 := by
  sorry

end total_cost_of_long_distance_bill_l2177_217727


namespace solve_logarithmic_system_l2177_217747

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem solve_logarithmic_system :
  ∃ x y : ℝ, log_base 2 x + log_base 4 y = 4 ∧ log_base 4 x + log_base 2 y = 5 ∧ x = 4 ∧ y = 16 :=
by
  sorry

end solve_logarithmic_system_l2177_217747


namespace geometric_sequence_common_ratio_l2177_217753

theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ)
  (q : ℝ)
  (h1 : ∀ n, a n > 0)
  (h2 : ∀ n, a (n + 1) = a n * q)
  (h3 : 3 * a 0 + 2 * a 1 = a 2 / 0.5) :
  q = 3 :=
  sorry

end geometric_sequence_common_ratio_l2177_217753


namespace no_solutions_l2177_217770

theorem no_solutions (x y : ℕ) (hx : x ≥ 1) (hy : y ≥ 1) : ¬ (x^5 = y^2 + 4) :=
by sorry

end no_solutions_l2177_217770


namespace find_a1_in_arithmetic_sequence_l2177_217706

noncomputable def arithmetic_sequence_sum (a₁ d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem find_a1_in_arithmetic_sequence :
  ∀ (a₁ d : ℤ), d = -2 →
  (arithmetic_sequence_sum a₁ d 11 = arithmetic_sequence_sum a₁ d 10) →
  a₁ = 20 :=
by
  intro a₁ d hd hs
  sorry

end find_a1_in_arithmetic_sequence_l2177_217706


namespace weights_equal_weights_equal_ints_weights_equal_rationals_l2177_217749

theorem weights_equal (w : Fin 13 → ℝ) (swap_n_weighs_balance : ∀ (s : Finset (Fin 13)), s.card = 12 → 
  ∃ (t u : Finset (Fin 13)), t.card = 6 ∧ u.card = 6 ∧ t ∪ u = s ∧ t ∩ u = ∅ ∧ Finset.sum t w = Finset.sum u w) :
  ∃ (m : ℝ), ∀ (i : Fin 13), w i = m :=
by
  sorry

theorem weights_equal_ints (w : Fin 13 → ℤ) (swap_n_weighs_balance_ints : ∀ (s : Finset (Fin 13)), s.card = 12 → 
  ∃ (t u : Finset (Fin 13)), t.card = 6 ∧ u.card = 6 ∧ t ∪ u = s ∧ t ∩ u = ∅ ∧ Finset.sum t w = Finset.sum u w) :
  ∃ (m : ℤ), ∀ (i : Fin 13), w i = m :=
by
  sorry

theorem weights_equal_rationals (w : Fin 13 → ℚ) (swap_n_weighs_balance_rationals : ∀ (s : Finset (Fin 13)), s.card = 12 → 
  ∃ (t u : Finset (Fin 13)), t.card = 6 ∧ u.card = 6 ∧ t ∪ u = s ∧ t ∩ u = ∅ ∧ Finset.sum t w = Finset.sum u w) :
  ∃ (m : ℚ), ∀ (i : Fin 13), w i = m :=
by
  sorry

end weights_equal_weights_equal_ints_weights_equal_rationals_l2177_217749


namespace population_at_seven_years_l2177_217735

theorem population_at_seven_years (a x : ℕ) (y: ℝ) (h₀: a = 100) (h₁: x = 7) (h₂: y = a * Real.logb 2 (x + 1)):
  y = 300 :=
by
  -- We include the conditions in the theorem statement
  sorry

end population_at_seven_years_l2177_217735


namespace problem_statement_l2177_217788

-- Universal set U is the set of all real numbers
def U : Set ℝ := Set.univ

-- Definition of set M
def M : Set ℝ := { y | ∃ x : ℝ, y = 2 ^ (Real.sqrt (2 * x - x ^ 2 + 3)) }

-- Complement of M in U
def C_U_M : Set ℝ := { y | y < 1 ∨ y > 4 }

-- Definition of set N
def N : Set ℝ := { x | -3 < x ∧ x < 2 }

-- Theorem stating (C_U_M) ∩ N = (-3, 1)
theorem problem_statement : (C_U_M ∩ N) = { x | -3 < x ∧ x < 1 } :=
sorry

end problem_statement_l2177_217788


namespace simplify_frac_l2177_217776

theorem simplify_frac (b : ℤ) (hb : b = 2) : (15 * b^4) / (45 * b^3) = 2 / 3 :=
by {
  sorry
}

end simplify_frac_l2177_217776


namespace gas_volumes_correct_l2177_217764

noncomputable def west_gas_vol_per_capita : ℝ := 21428
noncomputable def non_west_gas_vol : ℝ := 185255
noncomputable def non_west_population : ℝ := 6.9
noncomputable def non_west_gas_vol_per_capita : ℝ := non_west_gas_vol / non_west_population

noncomputable def russia_gas_vol_68_percent : ℝ := 30266.9
noncomputable def russia_gas_vol : ℝ := russia_gas_vol_68_percent * 100 / 68
noncomputable def russia_population : ℝ := 0.147
noncomputable def russia_gas_vol_per_capita : ℝ := russia_gas_vol / russia_population

theorem gas_volumes_correct :
  west_gas_vol_per_capita = 21428 ∧
  non_west_gas_vol_per_capita = 26848.55 ∧
  russia_gas_vol_per_capita = 302790.13 := by
    sorry

end gas_volumes_correct_l2177_217764


namespace race_position_problem_l2177_217758

theorem race_position_problem 
  (Cara Bruno Emily David Fiona Alan: ℕ)
  (participants : Finset ℕ)
  (participants_card : participants.card = 12)
  (hCara_Bruno : Cara = Bruno - 3)
  (hEmily_David : Emily = David + 1)
  (hAlan_Bruno : Alan = Bruno + 4)
  (hDavid_Fiona : David = Fiona + 3)
  (hFiona_Cara : Fiona = Cara - 2)
  (hBruno : Bruno = 9)
  (Cara_in_participants : Cara ∈ participants)
  (Bruno_in_participants : Bruno ∈ participants)
  (Emily_in_participants : Emily ∈ participants)
  (David_in_participants : David ∈ participants)
  (Fiona_in_participants : Fiona ∈ participants)
  (Alan_in_participants : Alan ∈ participants)
  : David = 7 := 
sorry

end race_position_problem_l2177_217758


namespace sum_m_n_zero_l2177_217720

theorem sum_m_n_zero
  (m n p : ℝ)
  (h1 : mn + p^2 + 4 = 0)
  (h2 : m - n = 4) :
  m + n = 0 :=
sorry

end sum_m_n_zero_l2177_217720


namespace number_of_people_who_bought_1_balloon_l2177_217729

-- Define the variables and the main theorem statement
variables (x1 x2 x3 x4 : ℕ)

theorem number_of_people_who_bought_1_balloon : 
  (x1 + x2 + x3 + x4 = 101) → 
  (x1 + 2 * x2 + 3 * x3 + 4 * x4 = 212) →
  (x4 = x2 + 13) → 
  x1 = 52 :=
by
  intros h1 h2 h3
  sorry

end number_of_people_who_bought_1_balloon_l2177_217729


namespace place_value_accuracy_l2177_217731

theorem place_value_accuracy (x : ℝ) (h : x = 3.20 * 10000) :
  ∃ p : ℕ, p = 100 ∧ (∃ k : ℤ, x / p = k) := by
  sorry

end place_value_accuracy_l2177_217731


namespace minimum_value_condition_l2177_217711

theorem minimum_value_condition (x a : ℝ) (h1 : x > a) (h2 : ∀ y, y > a → x + 4 / (y - a) > 9) : a = 6 :=
sorry

end minimum_value_condition_l2177_217711


namespace pencils_multiple_of_10_l2177_217717

theorem pencils_multiple_of_10 (pens : ℕ) (students : ℕ) (pencils : ℕ) 
  (h_pens : pens = 1230) 
  (h_students : students = 10) 
  (h_max_distribute : ∀ s, s ≤ students → (∃ pens_per_student, pens = pens_per_student * s ∧ ∃ pencils_per_student, pencils = pencils_per_student * s)) :
  ∃ n, pencils = 10 * n :=
by
  sorry

end pencils_multiple_of_10_l2177_217717


namespace base_b_cube_l2177_217796

theorem base_b_cube (b : ℕ) : (b > 4) → (∃ n : ℕ, (b^2 + 4 * b + 4 = n^3)) ↔ (b = 5 ∨ b = 6) :=
by
  sorry

end base_b_cube_l2177_217796


namespace tan_3theta_eq_9_13_l2177_217775

open Real

noncomputable def tan3theta (θ : ℝ) (h : tan θ = 3) : Prop :=
  tan (3 * θ) = (9 / 13)

theorem tan_3theta_eq_9_13 (θ : ℝ) (h : tan θ = 3) : tan3theta θ h :=
by
  sorry

end tan_3theta_eq_9_13_l2177_217775


namespace value_of_frac_l2177_217730

theorem value_of_frac (a b c d : ℕ) (h1 : a = 4 * b) (h2 : b = 3 * c) (h3 : c = 5 * d) :
  (a * c) / (b * d) = 20 :=
by
  sorry

end value_of_frac_l2177_217730


namespace sum_of_exponents_l2177_217726

-- Definition of Like Terms
def like_terms (m n : ℕ) : Prop :=
  m = 3 ∧ n = 2

-- Theorem statement
theorem sum_of_exponents (m n : ℕ) (h : like_terms m n) : m + n = 5 :=
sorry

end sum_of_exponents_l2177_217726


namespace sum_m_n_zero_l2177_217798

theorem sum_m_n_zero (m n p : ℝ) (h1 : mn + p^2 + 4 = 0) (h2 : m - n = 4) : m + n = 0 :=
sorry

end sum_m_n_zero_l2177_217798


namespace pure_imaginary_complex_l2177_217799

theorem pure_imaginary_complex (a : ℝ) (i : ℂ) (h : i * i = -1) (p : (1 + a * i) / (1 - i) = (0 : ℂ) + b * i) :
  a = 1 := 
sorry

end pure_imaginary_complex_l2177_217799


namespace overlap_percentage_l2177_217768

noncomputable def square_side_length : ℝ := 10
noncomputable def rectangle_length : ℝ := 18
noncomputable def rectangle_width : ℝ := square_side_length
noncomputable def overlap_length : ℝ := 2
noncomputable def overlap_width : ℝ := rectangle_width

noncomputable def rectangle_area : ℝ :=
  rectangle_length * rectangle_width

noncomputable def overlap_area : ℝ :=
  overlap_length * overlap_width

noncomputable def percentage_shaded : ℝ :=
  (overlap_area / rectangle_area) * 100

theorem overlap_percentage :
  percentage_shaded = 100 * (1 / 9) :=
sorry

end overlap_percentage_l2177_217768


namespace nine_fact_div_four_fact_eq_15120_l2177_217725

theorem nine_fact_div_four_fact_eq_15120 :
  (362880 / 24) = 15120 :=
by
  sorry

end nine_fact_div_four_fact_eq_15120_l2177_217725


namespace minimum_value_of_k_l2177_217703

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x + b
noncomputable def g (a c : ℝ) (x : ℝ) : ℝ := a * x + c
noncomputable def h (a b c : ℝ) (x : ℝ) : ℝ := (f a b x)^2 + 8 * (g a c x)
noncomputable def k (a b c : ℝ) (x : ℝ) : ℝ := (g a c x)^2 + 8 * (f a b x)

theorem minimum_value_of_k:
  ∃ a b c : ℝ, a ≠ 0 ∧ (∀ x : ℝ, h a b c x ≥ -29) → (∃ x : ℝ, k a b c x = -3) := sorry

end minimum_value_of_k_l2177_217703


namespace circle_radius_l2177_217719

theorem circle_radius (r : ℝ) (x y : ℝ) (h₁ : x = π * r ^ 2) (h₂ : y = 2 * π * r - 6) (h₃ : x + y = 94 * π) : 
  r = 10 :=
sorry

end circle_radius_l2177_217719


namespace packs_sold_to_uncle_is_correct_l2177_217716

-- Define the conditions and constants
def total_packs_needed := 50
def packs_sold_to_grandmother := 12
def packs_sold_to_neighbor := 5
def packs_left_to_sell := 26

-- Calculate total packs sold so far
def total_packs_sold := total_packs_needed - packs_left_to_sell

-- Calculate total packs sold to grandmother and neighbor
def packs_sold_to_grandmother_and_neighbor := packs_sold_to_grandmother + packs_sold_to_neighbor

-- The pack sold to uncle
def packs_sold_to_uncle := total_packs_sold - packs_sold_to_grandmother_and_neighbor

-- Prove the packs sold to uncle
theorem packs_sold_to_uncle_is_correct : packs_sold_to_uncle = 7 := by
  -- The proof steps are omitted
  sorry

end packs_sold_to_uncle_is_correct_l2177_217716


namespace no_prime_solutions_for_x2_plus_y3_eq_z4_l2177_217707

theorem no_prime_solutions_for_x2_plus_y3_eq_z4 :
  ¬ ∃ (x y z : ℕ), Prime x ∧ Prime y ∧ Prime z ∧ x^2 + y^3 = z^4 := sorry

end no_prime_solutions_for_x2_plus_y3_eq_z4_l2177_217707


namespace min_value_of_inverse_proportional_function_l2177_217784

theorem min_value_of_inverse_proportional_function 
  (x y : ℝ) (k : ℝ) 
  (h1 : y = k / x) 
  (h2 : ∀ x, -2 ≤ x ∧ x ≤ -1 → y ≤ 4) :
  (∀ x, x ≥ 8 → y = -1 / 2) :=
by
  sorry

end min_value_of_inverse_proportional_function_l2177_217784


namespace kristen_turtles_l2177_217785

variable (K : ℕ)
variable (T : ℕ)
variable (R : ℕ)

-- Conditions
def kris_turtles (K : ℕ) : ℕ := K / 4
def trey_turtles (R : ℕ) : ℕ := 7 * R
def trey_more_than_kristen (T K : ℕ) : Prop := T = K + 9

-- Theorem to prove 
theorem kristen_turtles (K : ℕ) (R : ℕ) (T : ℕ) (h1 : R = kris_turtles K) (h2 : T = trey_turtles R) (h3 : trey_more_than_kristen T K) : K = 12 :=
by
  sorry

end kristen_turtles_l2177_217785


namespace prove_relationship_l2177_217791

noncomputable def relationship_x_y_z (x y z : ℝ) (t : ℝ) : Prop :=
  (x / Real.sin t) = (y / Real.sin (2 * t)) ∧ (x / Real.sin t) = (z / Real.sin (3 * t))

theorem prove_relationship (x y z t : ℝ) (h : relationship_x_y_z x y z t) : x^2 - y^2 + x * z = 0 :=
by
  sorry

end prove_relationship_l2177_217791


namespace unique_point_graph_eq_l2177_217754

theorem unique_point_graph_eq (c : ℝ) : 
  (∀ x y : ℝ, 3 * x^2 + y^2 + 6 * x - 12 * y + c = 0 → x = -1 ∧ y = 6) ↔ c = 39 :=
sorry

end unique_point_graph_eq_l2177_217754


namespace sum_of_squares_l2177_217755

-- Define conditions
def condition1 (a b : ℝ) : Prop := a - b = 6
def condition2 (a b : ℝ) : Prop := a * b = 7

-- Define what we want to prove
def target (a b : ℝ) : Prop := a^2 + b^2 = 50

-- Main theorem stating the required proof
theorem sum_of_squares (a b : ℝ) (h1 : condition1 a b) (h2 : condition2 a b) : target a b :=
by sorry

end sum_of_squares_l2177_217755


namespace product_between_21st_and_24th_multiple_of_3_l2177_217772

theorem product_between_21st_and_24th_multiple_of_3 : 
  (66 * 69 = 4554) :=
by
  sorry

end product_between_21st_and_24th_multiple_of_3_l2177_217772


namespace sequence_problem_l2177_217786

noncomputable def b_n (n : ℕ) : ℝ := 5 * (5/3)^(n-2)

theorem sequence_problem 
  (a_n : ℕ → ℝ) 
  (d : ℝ) 
  (h1 : ∀ n, a_n (n + 1) = a_n n + d)
  (h2 : d ≠ 0)
  (h3 : a_n 8 = a_n 5 + 3 * d)
  (h4 : a_n 13 = a_n 8 + 5 * d)
  (b_2 : ℝ)
  (hb2 : b_2 = 5)
  (h5 : ∀ n, b_n n = (match n with | 2 => b_2 | _ => sorry))
  (conseq_terms : ∀ (n : ℕ), (a_n 5 + 3 * d)^2 = a_n 5 * (a_n 5 + 8 * d)) 
  : ∀ n, b_n n = b_n 2 * (5/3)^(n-2) := 
by 
  sorry

end sequence_problem_l2177_217786


namespace fifth_month_sale_correct_l2177_217782

noncomputable def fifth_month_sale
  (sales : Fin 4 → ℕ)
  (sixth_month_sale : ℕ)
  (average_sale : ℕ) : ℕ :=
  let total_sales := average_sale * 6
  let known_sales := sales 0 + sales 1 + sales 2 + sales 3 + sixth_month_sale
  total_sales - known_sales

theorem fifth_month_sale_correct :
  ∀ (sales : Fin 4 → ℕ) (sixth_month_sale : ℕ) (average_sale : ℕ),
    sales 0 = 6435 →
    sales 1 = 6927 →
    sales 2 = 6855 →
    sales 3 = 7230 →
    sixth_month_sale = 5591 →
    average_sale = 6600 →
    fifth_month_sale sales sixth_month_sale average_sale = 13562 :=
by
  intros sales sixth_month_sale average_sale h0 h1 h2 h3 h4 h5
  unfold fifth_month_sale
  sorry

end fifth_month_sale_correct_l2177_217782


namespace simplify_and_evaluate_l2177_217778

theorem simplify_and_evaluate (m n : ℤ) (h1 : m = 1) (h2 : n = -2) :
  -2 * (m * n - 3 * m^2) - (2 * m * n - 5 * (m * n - m^2)) = -1 :=
by
  sorry

end simplify_and_evaluate_l2177_217778


namespace problem1_problem2_l2177_217783

variables (a b : ℝ)

-- Problem 1: Prove that 3a^2 - 6a^2 - a^2 = -4a^2
theorem problem1 : (3 * a^2 - 6 * a^2 - a^2 = -4 * a^2) :=
by sorry

-- Problem 2: Prove that (5a - 3b) - 3(a^2 - 2b) = -3a^2 + 5a + 3b
theorem problem2 : ((5 * a - 3 * b) - 3 * (a^2 - 2 * b) = -3 * a^2 + 5 * a + 3 * b) :=
by sorry

end problem1_problem2_l2177_217783


namespace distance_between_trees_correct_l2177_217737

-- Define the given conditions
def yard_length : ℕ := 300
def tree_count : ℕ := 26
def interval_count : ℕ := tree_count - 1

-- Define the target distance between two consecutive trees
def target_distance : ℕ := 12

-- Prove that the distance between two consecutive trees is correct
theorem distance_between_trees_correct :
  yard_length / interval_count = target_distance := 
by
  sorry

end distance_between_trees_correct_l2177_217737


namespace fraction_value_l2177_217766

theorem fraction_value (x : ℝ) (h₀ : x^2 - 3 * x - 1 = 0) (h₁ : x ≠ 0) : 
  x^2 / (x^4 + x^2 + 1) = 1 / 12 := 
by
  sorry

end fraction_value_l2177_217766


namespace minimum_guests_at_banquet_l2177_217781

theorem minimum_guests_at_banquet (total_food : ℝ) (max_food_per_guest : ℝ) (min_guests : ℕ) 
  (h1 : total_food = 411) (h2 : max_food_per_guest = 2.5) : min_guests = 165 :=
by
  -- Proof omitted
  sorry

end minimum_guests_at_banquet_l2177_217781


namespace brians_gas_usage_l2177_217714

theorem brians_gas_usage (miles_per_gallon : ℕ) (miles_traveled : ℕ) (gallons_used : ℕ) 
  (h1 : miles_per_gallon = 20) 
  (h2 : miles_traveled = 60) 
  (h3 : gallons_used = miles_traveled / miles_per_gallon) : 
  gallons_used = 3 := 
by 
  rw [h1, h2] at h3 
  exact h3

end brians_gas_usage_l2177_217714


namespace total_students_l2177_217733

theorem total_students (N : ℕ)
    (h1 : (15 * 75) + (10 * 90) = N * 81) :
    N = 25 :=
by
  sorry

end total_students_l2177_217733


namespace probability_correct_l2177_217708

-- Define the total number of bulbs, good quality bulbs, and inferior quality bulbs
def total_bulbs : ℕ := 6
def good_bulbs : ℕ := 4
def inferior_bulbs : ℕ := 2

-- Define the probability of drawing one good bulb and one inferior bulb with replacement
def probability_one_good_one_inferior : ℚ := (good_bulbs * inferior_bulbs * 2) / (total_bulbs ^ 2)

-- Theorem stating that the probability of drawing one good bulb and one inferior bulb is 4/9
theorem probability_correct : probability_one_good_one_inferior = 4 / 9 := 
by
  -- Proof is skipped here
  sorry

end probability_correct_l2177_217708


namespace equivalent_form_l2177_217769

theorem equivalent_form (p q : ℝ) (hp₁ : p ≠ 0) (hp₂ : p ≠ 5) (hq₁ : q ≠ 0) (hq₂ : q ≠ 7) :
  (3/p + 4/q = 1/3) ↔ (p = 9*q/(q - 12)) :=
by
  sorry

end equivalent_form_l2177_217769


namespace joe_first_lift_weight_l2177_217741

variable (x y : ℕ)

def conditions : Prop :=
  (x + y = 1800) ∧ (2 * x = y + 300)

theorem joe_first_lift_weight (h : conditions x y) : x = 700 := by
  sorry

end joe_first_lift_weight_l2177_217741


namespace find_x_l2177_217750

theorem find_x (x : ℝ) : (0.75 / x = 10 / 8) → (x = 0.6) := by
  sorry

end find_x_l2177_217750


namespace final_number_not_perfect_square_l2177_217771

theorem final_number_not_perfect_square :
  (∃ final_number : ℕ, 
    ∀ a b : ℕ, a ∈ Finset.range 101 ∧ b ∈ Finset.range 101 ∧ a ≠ b → 
    gcd (a^2 + b^2 + 2) (a^2 * b^2 + 3) = final_number) →
  ∀ final_number : ℕ, ¬ ∃ k : ℕ, final_number = k ^ 2 :=
sorry

end final_number_not_perfect_square_l2177_217771


namespace logan_list_count_l2177_217702

theorem logan_list_count : 
    let smallest_square_multiple := 900
    let smallest_cube_multiple := 27000
    ∃ n, n = 871 ∧ 
        ∀ k, (k * 30 ≥ smallest_square_multiple ∧ k * 30 ≤ smallest_cube_multiple) ↔ (30 ≤ k ∧ k ≤ 900) :=
by
    let smallest_square_multiple := 900
    let smallest_cube_multiple := 27000
    use 871
    sorry

end logan_list_count_l2177_217702


namespace min_value_abs_function_l2177_217746

theorem min_value_abs_function : ∀ x : ℝ, 4 ≤ x ∧ x ≤ 6 → (|x - 4| + |x - 6| = 2) :=
by
  sorry


end min_value_abs_function_l2177_217746


namespace intercept_sum_modulo_l2177_217795

theorem intercept_sum_modulo (x_0 y_0 : ℤ) (h1 : 0 ≤ x_0) (h2 : x_0 < 17) (h3 : 0 ≤ y_0) (h4 : y_0 < 17)
                       (hx : 5 * x_0 ≡ 2 [ZMOD 17])
                       (hy : 3 * y_0 ≡ 15 [ZMOD 17]) :
    x_0 + y_0 = 19 := 
by
  sorry

end intercept_sum_modulo_l2177_217795


namespace compare_expressions_l2177_217712

-- Define the theorem statement
theorem compare_expressions (x : ℝ) : (x - 2) * (x + 3) > x^2 + x - 7 := by
  sorry -- The proof is omitted.

end compare_expressions_l2177_217712


namespace divide_inequality_by_negative_l2177_217723

theorem divide_inequality_by_negative {x : ℝ} (h : -6 * x > 2) : x < -1 / 3 :=
by sorry

end divide_inequality_by_negative_l2177_217723


namespace exponentiation_81_5_4_eq_243_l2177_217797

theorem exponentiation_81_5_4_eq_243 : 81^(5/4) = 243 := by
  sorry

end exponentiation_81_5_4_eq_243_l2177_217797


namespace Brian_watch_animal_videos_l2177_217701

theorem Brian_watch_animal_videos :
  let cat_video := 4
  let dog_video := 2 * cat_video
  let gorilla_video := 2 * (cat_video + dog_video)
  let elephant_video := cat_video + dog_video + gorilla_video
  let dolphin_video := cat_video + dog_video + gorilla_video + elephant_video
  let total_time := cat_video + dog_video + gorilla_video + elephant_video + dolphin_video
  total_time = 144 := by
{
  let cat_video := 4
  let dog_video := 2 * cat_video
  let gorilla_video := 2 * (cat_video + dog_video)
  let elephant_video := cat_video + dog_video + gorilla_video
  let dolphin_video := cat_video + dog_video + gorilla_video + elephant_video
  let total_time := cat_video + dog_video + gorilla_video + elephant_video + dolphin_video
  have h1 : total_time = (4 + 8 + 24 + 36 + 72) := sorry
  exact h1
}

end Brian_watch_animal_videos_l2177_217701
