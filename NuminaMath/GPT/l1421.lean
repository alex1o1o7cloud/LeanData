import Mathlib

namespace age_difference_l1421_142153

theorem age_difference (A B : ℕ) (h1 : B = 39) (h2 : A + 10 = 2 * (B - 10)) : A - B = 9 := by
  sorry

end age_difference_l1421_142153


namespace original_price_vase_l1421_142124

-- Definitions based on the conditions and problem elements
def original_price (P : ℝ) : Prop :=
  0.825 * P = 165

-- Statement to prove equivalence
theorem original_price_vase : ∃ P : ℝ, original_price P ∧ P = 200 :=
  by
    sorry

end original_price_vase_l1421_142124


namespace smallest_number_of_cubes_l1421_142111

theorem smallest_number_of_cubes (l w d : ℕ) (hl : l = 36) (hw : w = 45) (hd : d = 18) : 
  ∃ n : ℕ, n = 40 ∧ (∃ s : ℕ, l % s = 0 ∧ w % s = 0 ∧ d % s = 0 ∧ (l / s) * (w / s) * (d / s) = n) := 
by
  sorry

end smallest_number_of_cubes_l1421_142111


namespace participants_count_l1421_142130

theorem participants_count (F M : ℕ)
  (hF2 : F / 2 = 110)
  (hM4 : M / 4 = 330 - F - M / 3)
  (hFm : (F + M) / 3 = F / 2 + M / 4) :
  F + M = 330 :=
sorry

end participants_count_l1421_142130


namespace coefficient_of_x_100_l1421_142131

-- Define the polynomial P
noncomputable def P : Polynomial ℤ :=
  (Polynomial.C (-1) + Polynomial.X) *
  (Polynomial.C (-2) + Polynomial.X^2) *
  (Polynomial.C (-3) + Polynomial.X^3) *
  (Polynomial.C (-4) + Polynomial.X^4) *
  (Polynomial.C (-5) + Polynomial.X^5) *
  (Polynomial.C (-6) + Polynomial.X^6) *
  (Polynomial.C (-7) + Polynomial.X^7) *
  (Polynomial.C (-8) + Polynomial.X^8) *
  (Polynomial.C (-9) + Polynomial.X^9) *
  (Polynomial.C (-10) + Polynomial.X^10) *
  (Polynomial.C (-11) + Polynomial.X^11) *
  (Polynomial.C (-12) + Polynomial.X^12) *
  (Polynomial.C (-13) + Polynomial.X^13) *
  (Polynomial.C (-14) + Polynomial.X^14) *
  (Polynomial.C (-15) + Polynomial.X^15)

-- State the theorem
theorem coefficient_of_x_100 : P.coeff 100 = 445 :=
  by sorry

end coefficient_of_x_100_l1421_142131


namespace evaluate_fraction_l1421_142107

noncomputable def evaluate_expression : ℚ := 
  1 / (2 - (1 / (2 - (1 / (2 - (1 / 3))))))
  
theorem evaluate_fraction :
  evaluate_expression = 5 / 7 :=
sorry

end evaluate_fraction_l1421_142107


namespace withdraw_representation_l1421_142105

-- Define the concept of depositing and withdrawing money.
def deposit (amount : ℕ) : ℤ := amount
def withdraw (amount : ℕ) : ℤ := - amount

-- Define the given condition: depositing $30,000 is represented as $+30,000.
def deposit_condition : deposit 30000 = 30000 := by rfl

-- The statement to be proved: withdrawing $40,000 is represented as $-40,000
theorem withdraw_representation (deposit_condition : deposit 30000 = 30000) : withdraw 40000 = -40000 :=
by
  sorry

end withdraw_representation_l1421_142105


namespace num_students_l1421_142143

theorem num_students (x : ℕ) (h1 : ∃ z : ℕ, z = 10 * x + 6) (h2 : ∃ z : ℕ, z = 12 * x - 6) : x = 6 :=
by
  sorry

end num_students_l1421_142143


namespace boys_on_soccer_team_l1421_142135

theorem boys_on_soccer_team (B G : ℕ) (h1 : B + G = 30) (h2 : 1 / 3 * G + B = 20) : B = 15 :=
sorry

end boys_on_soccer_team_l1421_142135


namespace download_time_l1421_142190

theorem download_time (speed : ℕ) (file1 file2 file3 : ℕ) (total_time : ℕ) (hours : ℕ) :
  speed = 2 ∧ file1 = 80 ∧ file2 = 90 ∧ file3 = 70 ∧ total_time = file1 / speed + file2 / speed + file3 / speed ∧
  hours = total_time / 60 → hours = 2 := 
by
  sorry

end download_time_l1421_142190


namespace solution_set_inequality_l1421_142117

open Real

theorem solution_set_inequality (k : ℤ) (x : ℝ) :
  (x ∈ Set.Ioo (-π/4 + k * π) (k * π)) ↔ cos (4 * x) - 2 * sin (2 * x) - sin (4 * x) - 1 > 0 :=
by
  sorry

end solution_set_inequality_l1421_142117


namespace winner_for_2023_winner_for_2024_l1421_142102

-- Definitions for the game conditions
def barbara_moves : List ℕ := [3, 5]
def jenna_moves : List ℕ := [1, 4, 5]

-- Lean theorem statement proving the required answers
theorem winner_for_2023 (coins : ℕ) (barbara_moves : List ℕ) (jenna_moves : List ℕ) :
  coins = 2023 →  -- Specifying that the game starts with 2023 coins
  (∀n, n ∈ barbara_moves → n ≤ 2023) ∧ (∀n, n ∈ jenna_moves → n ≤ 2023) → 
  -- Specifying valid moves for both players
  ∃ winner : String, winner = "Barbara" := 
sorry

theorem winner_for_2024 (coins : ℕ) (barbara_moves : List ℕ) (jenna_moves : List ℕ) :
  coins = 2024 →  -- Specifying that the game starts with 2024 coins
  (∀n, n ∈ barbara_moves → n ≤ 2024) ∧ (∀n, n ∈ jenna_moves → n ≤ 2024) → 
  -- Specifying valid moves for both players
  ∃ winner : String, winner = "Whoever starts" :=
sorry

end winner_for_2023_winner_for_2024_l1421_142102


namespace problem_statement_l1421_142165

theorem problem_statement (h1 : Real.cos (Real.pi / 6) = (Real.sqrt 3) / 2) :
  (Real.pi / (Real.sqrt 3 - 1))^0 - (Real.cos (Real.pi / 6))^2 = 1 / 4 := by
  sorry

end problem_statement_l1421_142165


namespace min_total_cost_minimize_cost_l1421_142109

theorem min_total_cost (x : ℝ) (h₀ : x > 0) :
  (900 / x * 3 + 3 * x) ≥ 180 :=
by sorry

theorem minimize_cost (x : ℝ) (h₀ : x > 0) :
  x = 30 ↔ (900 / x * 3 + 3 * x) = 180 :=
by sorry

end min_total_cost_minimize_cost_l1421_142109


namespace triangle_shape_l1421_142163

theorem triangle_shape (a b c : ℝ) (A B C : ℝ) (h1 : a * Real.cos A = b * Real.cos B) :
  (a = b ∨ c = a ∨ c = b ∨ A = Real.pi / 2 ∨ B = Real.pi / 2 ∨ C = Real.pi / 2) :=
sorry

end triangle_shape_l1421_142163


namespace find_orig_denominator_l1421_142134

-- Definitions as per the conditions
def orig_numer : ℕ := 2
def mod_numer : ℕ := orig_numer + 3

-- The modified fraction yields 1/3
def new_fraction (d : ℕ) : Prop :=
  (mod_numer : ℚ) / (d + 4) = 1 / 3

-- Proof Problem Statement
theorem find_orig_denominator (d : ℕ) : new_fraction d → d = 11 :=
  sorry

end find_orig_denominator_l1421_142134


namespace june_earnings_l1421_142119

theorem june_earnings
  (total_clovers : ℕ)
  (clover_3_petals_percentage : ℝ)
  (clover_2_petals_percentage : ℝ)
  (clover_4_petals_percentage : ℝ)
  (earnings_per_clover : ℝ) :
  total_clovers = 200 →
  clover_3_petals_percentage = 0.75 →
  clover_2_petals_percentage = 0.24 →
  clover_4_petals_percentage = 0.01 →
  earnings_per_clover = 1 →
  (total_clovers * earnings_per_clover) = 200 := by
  sorry

end june_earnings_l1421_142119


namespace find_a5_geometric_sequence_l1421_142113

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop := 
  ∃ r > 0, ∀ n ≥ 1, a (n + 1) = r * a n

theorem find_a5_geometric_sequence :
  ∀ (a : ℕ → ℝ),
  geometric_sequence a ∧ 
  (∀ n, a n > 0) ∧ 
  (a 3 * a 11 = 16) 
  → a 5 = 1 :=
by
  sorry

end find_a5_geometric_sequence_l1421_142113


namespace balls_to_boxes_l1421_142140

theorem balls_to_boxes (balls boxes : ℕ) (h1 : balls = 5) (h2 : boxes = 3) :
  ∃ ways : ℕ, ways = 150 := by
  sorry

end balls_to_boxes_l1421_142140


namespace prob1_prob2_l1421_142192

-- Define the polynomial function
def polynomial (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Problem 1: Prove |b| ≤ 1, given conditions
theorem prob1 (a b c : ℝ) (h : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → |polynomial a b c x| ≤ 1) : |b| ≤ 1 :=
sorry

-- Problem 2: Find a = 2, given conditions
theorem prob2 (a b c : ℝ) 
  (h1 : polynomial a b c 0 = -1) 
  (h2 : polynomial a b c 1 = 1) 
  (h3 : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → |polynomial a b c x| ≤ 1) : 
  a = 2 :=
sorry

end prob1_prob2_l1421_142192


namespace tamara_diff_3kim_height_l1421_142144

variables (K T X : ℕ) -- Kim's height, Tamara's height, and the difference inches respectively

-- Conditions
axiom ht_Tamara : T = 68
axiom combined_ht : T + K = 92
axiom diff_eqn : T = 3 * K - X

theorem tamara_diff_3kim_height (h₁ : T = 68) (h₂ : T + K = 92) (h₃ : T = 3 * K - X) : X = 4 :=
by
  sorry

end tamara_diff_3kim_height_l1421_142144


namespace smallest_k_square_divisible_l1421_142146

theorem smallest_k_square_divisible (k : ℤ) (n : ℤ) (h1 : k = 60)
    (h2 : ∀ m : ℤ, m < k → ∃ d : ℤ, d ∣ (k^2) → m = d ) : n = 3600 :=
sorry

end smallest_k_square_divisible_l1421_142146


namespace Gyeongyeon_cookies_l1421_142103

def initial_cookies : ℕ := 20
def cookies_given : ℕ := 7
def cookies_received : ℕ := 5

def final_cookies (initial : ℕ) (given : ℕ) (received : ℕ) : ℕ :=
  initial - given + received

theorem Gyeongyeon_cookies :
  final_cookies initial_cookies cookies_given cookies_received = 18 :=
by
  sorry

end Gyeongyeon_cookies_l1421_142103


namespace capacity_ratio_proof_l1421_142168

noncomputable def capacity_ratio :=
  ∀ (C_X C_Y : ℝ), 
    (1 / 2) * C_X + (2 / 5) * C_Y = (65 / 100) * C_Y →
    (C_Y / C_X) = (1 / 2)

-- includes a statement without proof
theorem capacity_ratio_proof (C_X C_Y : ℝ) (h : (1 / 2) * C_X + (2 / 5) * C_Y = (65 / 100) * C_Y) : 
  (C_Y / C_X) = (1 / 2) :=
  by
    sorry

end capacity_ratio_proof_l1421_142168


namespace initial_passengers_l1421_142141

theorem initial_passengers (P : ℝ) :
  (1/2 * (2/3 * P + 280) + 12 = 242) → P = 270 :=
by
  sorry

end initial_passengers_l1421_142141


namespace binomial_expansion_b_value_l1421_142185

theorem binomial_expansion_b_value (a b x : ℝ) (h : (1 + a * x) ^ 5 = 1 + 10 * x + b * x ^ 2 + a^5 * x ^ 5) : b = 40 := 
sorry

end binomial_expansion_b_value_l1421_142185


namespace caleb_caught_trouts_l1421_142186

theorem caleb_caught_trouts (C : ℕ) (h1 : 3 * C = C + 4) : C = 2 :=
by {
  sorry
}

end caleb_caught_trouts_l1421_142186


namespace weights_identical_l1421_142120

theorem weights_identical (w : Fin 13 → ℤ) 
  (h : ∀ i, ∃ (A B : Finset (Fin 13)), A.card = 6 ∧ B.card = 6 ∧ A ∪ B = Finset.univ.erase i ∧ (A.sum w) = (B.sum w)) :
  ∀ i j, w i = w j :=
by
  sorry

end weights_identical_l1421_142120


namespace handshakes_correct_l1421_142164

-- Definitions based on conditions
def num_gremlins : ℕ := 25
def num_imps : ℕ := 20
def num_imps_shaking_hands_among_themselves : ℕ := num_imps / 2
def comb (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

-- Function to calculate the total handshakes
def total_handshakes : ℕ :=
  (comb num_gremlins 2) + -- Handshakes among gremlins
  (comb num_imps_shaking_hands_among_themselves 2) + -- Handshakes among half the imps
  (num_gremlins * num_imps) -- Handshakes between all gremlins and all imps

-- The theorem to be proved
theorem handshakes_correct : total_handshakes = 845 := by
  sorry

end handshakes_correct_l1421_142164


namespace open_door_within_time_l1421_142170

-- Define the initial conditions
def device := ℕ → ℕ

-- Constraint: Each device has 5 toggle switches ("0" or "1") and a three-digit display.
def valid_configuration (d : device) (k : ℕ) : Prop :=
  d k < 32 ∧ d k <= 999

def system_configuration (A B : device) (k : ℕ) : Prop :=
  A k = B k

-- Constraint: The devices can be synchronized to display the same number simultaneously to open the door.
def open_door (A B : device) : Prop :=
  ∃ k, system_configuration A B k

-- The main theorem: Devices A and B can be synchronized within the given time constraints to open the door.
theorem open_door_within_time (A B : device) (notebook : ℕ) : 
  (∀ k, valid_configuration A k ∧ valid_configuration B k) →
  open_door A B :=
by sorry

end open_door_within_time_l1421_142170


namespace diophantine_solution_l1421_142148

theorem diophantine_solution (a b : ℕ) (h_coprime : Nat.gcd a b = 1) (n : ℕ) (h_n : n > a * b) :
  ∃ x y : ℕ, n = a * x + b * y :=
by
  sorry

end diophantine_solution_l1421_142148


namespace mary_money_left_l1421_142137

variable (p : ℝ)

theorem mary_money_left :
  have cost_drinks := 3 * p
  have cost_medium_pizza := 2 * p
  have cost_large_pizza := 3 * p
  let total_cost := cost_drinks + cost_medium_pizza + cost_large_pizza
  30 - total_cost = 30 - 8 * p := by {
    sorry
  }

end mary_money_left_l1421_142137


namespace cube_root_sum_is_integer_l1421_142126

theorem cube_root_sum_is_integer :
  let a := (2 + (10 / 9) * Real.sqrt 3)^(1/3)
  let b := (2 - (10 / 9) * Real.sqrt 3)^(1/3)
  a + b = 2 := by
  sorry

end cube_root_sum_is_integer_l1421_142126


namespace store_sales_correct_l1421_142196

def price_eraser_pencil : ℝ := 0.8
def price_regular_pencil : ℝ := 0.5
def price_short_pencil : ℝ := 0.4
def price_mechanical_pencil : ℝ := 1.2
def price_novelty_pencil : ℝ := 1.5

def quantity_eraser_pencil : ℕ := 200
def quantity_regular_pencil : ℕ := 40
def quantity_short_pencil : ℕ := 35
def quantity_mechanical_pencil : ℕ := 25
def quantity_novelty_pencil : ℕ := 15

def total_sales : ℝ :=
  (quantity_eraser_pencil * price_eraser_pencil) +
  (quantity_regular_pencil * price_regular_pencil) +
  (quantity_short_pencil * price_short_pencil) +
  (quantity_mechanical_pencil * price_mechanical_pencil) +
  (quantity_novelty_pencil * price_novelty_pencil)

theorem store_sales_correct : total_sales = 246.5 :=
by sorry

end store_sales_correct_l1421_142196


namespace correct_option_l1421_142125

def inverse_proportion (x y : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ y = k / x

theorem correct_option :
  inverse_proportion x y → 
  (y = x + 3 ∨ y = x / 3 ∨ y = 3 / (x ^ 2) ∨ y = 3 / x) → 
  y = 3 / x :=
by
  sorry

end correct_option_l1421_142125


namespace no_square_number_divisible_by_six_in_range_l1421_142158

theorem no_square_number_divisible_by_six_in_range :
  ¬ ∃ (x : ℕ), (x ^ 2) % 6 = 0 ∧ 39 < x ^ 2 ∧ x ^ 2 < 120 :=
by
  sorry

end no_square_number_divisible_by_six_in_range_l1421_142158


namespace angle_ABC_measure_l1421_142177

theorem angle_ABC_measure
  (CBD : ℝ)
  (ABC ABD : ℝ)
  (h1 : CBD = 90)
  (h2 : ABC + ABD + CBD = 270)
  (h3 : ABD = 100) : 
  ABC = 80 :=
by
  -- Given:
  -- CBD = 90
  -- ABC + ABD + CBD = 270
  -- ABD = 100
  sorry

end angle_ABC_measure_l1421_142177


namespace smallest_value_of_M_l1421_142149

theorem smallest_value_of_M :
  ∀ (a b c d e f g M : ℕ), a > 0 → b > 0 → c > 0 → d > 0 → e > 0 → f > 0 → g > 0 →
  a + b + c + d + e + f + g = 2024 →
  M = max (a + b) (max (b + c) (max (c + d) (max (d + e) (max (e + f) (f + g))))) →
  M = 338 :=
by
  intro a b c d e f g M ha hb hc hd he hf hg hsum hmax
  sorry

end smallest_value_of_M_l1421_142149


namespace sandra_fathers_contribution_ratio_l1421_142121

theorem sandra_fathers_contribution_ratio :
  let saved := 10
  let mother := 4
  let candy_cost := 0.5
  let jellybean_cost := 0.2
  let candies := 14
  let jellybeans := 20
  let remaining := 11
  let total_cost := candies * candy_cost + jellybeans * jellybean_cost
  let total_amount := total_cost + remaining
  let amount_without_father := saved + mother
  let father := total_amount - amount_without_father
  (father / mother) = 2 := by 
  sorry

end sandra_fathers_contribution_ratio_l1421_142121


namespace seating_arrangement_l1421_142194

theorem seating_arrangement (x y : ℕ) (h : x + y ≤ 8) (h1 : 9 * x + 6 * y = 57) : x = 5 := 
by
  sorry

end seating_arrangement_l1421_142194


namespace eating_time_l1421_142162

-- Defining the terms based on the conditions provided
def rate_mr_swift := 1 / 15 -- Mr. Swift eats 1 pound in 15 minutes
def rate_mr_slow := 1 / 45  -- Mr. Slow eats 1 pound in 45 minutes

-- Combined eating rate of Mr. Swift and Mr. Slow
def combined_rate := rate_mr_swift + rate_mr_slow

-- Total amount of cereal to be consumed
def total_cereal := 4 -- pounds

-- Proving the total time to eat the cereal
theorem eating_time :
  (total_cereal / combined_rate) = 45 :=
by
  sorry

end eating_time_l1421_142162


namespace goods_train_length_is_470_l1421_142116

noncomputable section

def speed_kmph := 72
def platform_length := 250
def crossing_time := 36

def speed_mps := speed_kmph * 5 / 18
def distance_covered := speed_mps * crossing_time

def length_of_train := distance_covered - platform_length

theorem goods_train_length_is_470 :
  length_of_train = 470 :=
by
  sorry

end goods_train_length_is_470_l1421_142116


namespace problem_l1421_142161

theorem problem 
  (a b A B : ℝ)
  (h : ∀ x : ℝ, 1 - a * Real.cos x - b * Real.sin x - A * Real.cos (2 * x) - B * Real.sin (2 * x) ≥ 0) :
  a^2 + b^2 ≤ 2 ∧ A^2 + B^2 ≤ 1 :=
by sorry

end problem_l1421_142161


namespace cost_per_bundle_l1421_142157

-- Condition: each rose costs 500 won
def rose_price := 500

-- Condition: total number of roses
def total_roses := 200

-- Condition: number of bundles
def bundles := 25

-- Question: Prove the cost per bundle
theorem cost_per_bundle (rp : ℕ) (tr : ℕ) (b : ℕ) : rp = 500 → tr = 200 → b = 25 → (rp * tr) / b = 4000 :=
by
  intros h0 h1 h2
  sorry

end cost_per_bundle_l1421_142157


namespace minimum_value_of_a_l1421_142187

theorem minimum_value_of_a (x : ℝ) (a : ℝ) (hx : 0 ≤ x) (hx2 : x ≤ 20) (ha : 0 < a) (h : (20 - x) / 4 + a / 2 * Real.sqrt x ≥ 5) : 
  a ≥ Real.sqrt 5 := 
sorry

end minimum_value_of_a_l1421_142187


namespace initial_men_in_garrison_l1421_142172

variable (x : ℕ)

theorem initial_men_in_garrison (h1 : x * 65 = x * 50 + (x + 3000) * 20) : x = 2000 :=
  sorry

end initial_men_in_garrison_l1421_142172


namespace cost_of_eight_CDs_l1421_142128

theorem cost_of_eight_CDs (cost_of_two_CDs : ℕ) (h : cost_of_two_CDs = 36) : 8 * (cost_of_two_CDs / 2) = 144 := by
  sorry

end cost_of_eight_CDs_l1421_142128


namespace max_girls_with_five_boys_l1421_142197

theorem max_girls_with_five_boys : 
  ∃ n : ℕ, n = 20 ∧ ∀ (boys : Fin 5 → ℝ × ℝ), 
  (∃ (girls : Fin n → ℝ × ℝ),
  (∀ i : Fin n, ∃ j k : Fin 5, j ≠ k ∧ dist (girls i) (boys j) = 5 ∧ dist (girls i) (boys k) = 5)) :=
sorry

end max_girls_with_five_boys_l1421_142197


namespace chords_even_arcs_even_l1421_142180

theorem chords_even_arcs_even (N : ℕ) 
  (h1 : ∀ k : ℕ, k ≤ N → ¬ ((k : ℤ) % 2 = 1)) : 
  N % 2 = 0 := 
sorry

end chords_even_arcs_even_l1421_142180


namespace solve_for_x_l1421_142199

theorem solve_for_x (x : ℝ) (h : 0.009 / x = 0.1) : x = 0.09 :=
sorry

end solve_for_x_l1421_142199


namespace simplify_fraction_l1421_142156

theorem simplify_fraction : 
  (1 - (1 / 4)) / (1 - (1 / 3)) = 9 / 8 :=
by
  sorry

end simplify_fraction_l1421_142156


namespace factor_64_minus_16y_squared_l1421_142159

theorem factor_64_minus_16y_squared (y : ℝ) : 
  64 - 16 * y^2 = 16 * (2 - y) * (2 + y) :=
by
  -- skipping the actual proof steps
  sorry

end factor_64_minus_16y_squared_l1421_142159


namespace rectangle_area_l1421_142104

theorem rectangle_area (a w l : ℝ) (h_square_area : a = 36) 
    (h_rect_width : w * w = a) 
    (h_rect_length : l = 3 * w) : w * l = 108 := 
sorry

end rectangle_area_l1421_142104


namespace other_root_of_quadratic_l1421_142123

theorem other_root_of_quadratic (m : ℝ) (h : ∀ x : ℝ, x^2 + m*x - 20 = 0 → (x = -4)) 
: ∃ t : ℝ, t = 5 := 
by
  existsi 5
  sorry

end other_root_of_quadratic_l1421_142123


namespace find_x_l1421_142108

def vec (x y : ℝ) := (x, y)

def a := vec 1 (-4)
def b (x : ℝ) := vec (-1) x
def c (x : ℝ) := (a.1 + 3 * (b x).1, a.2 + 3 * (b x).2)

theorem find_x (x : ℝ) : a.1 * (c x).2 = (c x).1 * a.2 → x = 4 :=
by
  sorry

end find_x_l1421_142108


namespace domain_of_log_function_l1421_142122

open Real

noncomputable def domain_of_function : Set ℝ :=
  {x | x > 2 ∨ x < -1}

theorem domain_of_log_function :
  ∀ x : ℝ, (x^2 - x - 2 > 0) ↔ (x > 2 ∨ x < -1) :=
by
  intro x
  exact sorry

end domain_of_log_function_l1421_142122


namespace solve_for_x_add_y_l1421_142115

theorem solve_for_x_add_y (x y : ℤ) 
  (h1 : y = 245) 
  (h2 : x - y = 200) : 
  x + y = 690 :=
by {
  -- Here we would provide the proof if needed
  sorry
}

end solve_for_x_add_y_l1421_142115


namespace no_triangle_formed_l1421_142188

def line1 (x y : ℝ) := 2 * x - 3 * y + 1 = 0
def line2 (x y : ℝ) := 4 * x + 3 * y + 5 = 0
def line3 (m : ℝ) (x y : ℝ) := m * x - y - 1 = 0

theorem no_triangle_formed (m : ℝ) :
  (∀ x y, line1 x y → line3 m x y) ∨
  (∀ x y, line2 x y → line3 m x y) ∨
  (∃ x y, line1 x y ∧ line2 x y ∧ line3 m x y) ↔
  (m = -4/3 ∨ m = 2/3 ∨ m = 4/3) :=
sorry -- Proof to be provided

end no_triangle_formed_l1421_142188


namespace evaluate_expression_l1421_142106

-- Define the condition b = 2
def b : ℕ := 2

-- Theorem statement
theorem evaluate_expression : (b^3 * b^4 = 128) := 
by
  sorry

end evaluate_expression_l1421_142106


namespace blue_marbles_difference_l1421_142145

theorem blue_marbles_difference  (a b : ℚ) 
  (h1 : 3 * a + 2 * b = 80)
  (h2 : 2 * a = b) :
  (7 * a - 3 * b) = 80 / 7 := by
  sorry

end blue_marbles_difference_l1421_142145


namespace geometric_sequence_sum_l1421_142139

theorem geometric_sequence_sum (q : ℝ) (a : ℕ → ℝ)
  (h_pos : ∀ n, 0 < a n)
  (h_geom : ∀ n, a (n + 1) = q * a n)
  (h_a1 : a 1 = 1)
  (h_sum : a 1 + a 3 + a 5 = 21) :
  a 2 + a 4 + a 6 = 42 :=
sorry

end geometric_sequence_sum_l1421_142139


namespace find_nsatisfy_l1421_142184

-- Define the function S(n) that denotes the sum of the digits of n
def S (n : ℕ) : ℕ := n.digits 10 |>.sum

-- State the main theorem
theorem find_nsatisfy {n : ℕ} : n = 2 * (S n)^2 → n = 50 ∨ n = 162 ∨ n = 392 ∨ n = 648 := 
sorry

end find_nsatisfy_l1421_142184


namespace no_positive_integer_solution_exists_l1421_142189

theorem no_positive_integer_solution_exists :
  ¬ ∃ (x y : ℕ), 0 < x ∧ 0 < y ∧ 3 * x^2 + 2 * x + 2 = y^2 :=
by
  -- The proof steps will go here.
  sorry

end no_positive_integer_solution_exists_l1421_142189


namespace max_catch_up_distance_l1421_142179

/-- 
Given:
  - The total length of the race is 5000 feet.
  - Alex and Max are even for the first 200 feet, so the initial distance between them is 0 feet.
  - On the uphill slope, Alex gets ahead by 300 feet.
  - On the downhill slope, Max gains a lead of 170 feet over Alex, reducing Alex's lead.
  - On the flat section, Alex pulls ahead by 440 feet.

Prove:
  - The distance left for Max to catch up to Alex is 4430 feet.
--/
theorem max_catch_up_distance :
  let total_distance := 5000
  let initial_distance := 0
  let alex_uphill_lead := 300
  let max_downhill_gain := 170
  let alex_flat_gain := 440
  let final_distance := initial_distance + alex_uphill_lead - max_downhill_gain + alex_flat_gain
  total_distance - final_distance = 4430 :=
by
  let total_distance := 5000
  let initial_distance := 0
  let alex_uphill_lead := 300
  let max_downhill_gain := 170
  let alex_flat_gain := 440
  let final_distance := initial_distance + alex_uphill_lead - max_downhill_gain + alex_flat_gain
  have final_distance_calc : final_distance = 570
  sorry
  show total_distance - final_distance = 4430
  sorry

end max_catch_up_distance_l1421_142179


namespace find_m_solve_inequality_l1421_142183

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := m - |x - 2|

theorem find_m (m : ℝ) : (∀ x : ℝ, m - |x| ≥ 0 ↔ x ∈ [-1, 1]) → m = 1 :=
by
  sorry

theorem solve_inequality (x : ℝ) : |x + 1| + |x - 2| > 4 * 1 ↔ x < -3 / 2 ∨ x > 5 / 2 :=
by
  sorry

end find_m_solve_inequality_l1421_142183


namespace complement_union_l1421_142114

def U : Set ℕ := {0, 1, 2, 3, 4}
def M : Set ℕ := {1, 2, 4}
def N : Set ℕ := {2, 3}

theorem complement_union (U : Set ℕ) (M : Set ℕ) (N : Set ℕ) (hU : U = {0, 1, 2, 3, 4}) (hM : M = {1, 2, 4}) (hN : N = {2, 3}) :
  (U \ M) ∪ N = {0, 2, 3} :=
by
  rw [hU, hM, hN] -- Substitute U, M, N definitions
  sorry -- Proof omitted

end complement_union_l1421_142114


namespace parabola_coeff_sum_l1421_142133

def parabola_vertex_form (a b c : ℚ) : Prop :=
  (∀ y : ℚ, y = 2 → (-3) = a * (y - 2)^2 + b * (y - 2) + c) ∧
  (∀ x y : ℚ, x = 1 ∧ y = -1 → x = a * y^2 + b * y + c) ∧
  (a < 0)  -- Since the parabola opens to the left, implying the coefficient 'a' is positive.

theorem parabola_coeff_sum (a b c : ℚ) :
  parabola_vertex_form a b c → a + b + c = -23 / 9 :=
by
  sorry

end parabola_coeff_sum_l1421_142133


namespace Miss_Darlington_total_blueberries_l1421_142171

-- Conditions
def initial_basket := 20
def additional_baskets := 9

-- Definition and statement to be proved
theorem Miss_Darlington_total_blueberries :
  initial_basket + additional_baskets * initial_basket = 200 :=
by
  sorry

end Miss_Darlington_total_blueberries_l1421_142171


namespace intersection_of_A_and_B_l1421_142127

-- Define the sets A and B
def A := {x : ℝ | x ≥ 1}
def B := {x : ℝ | -1 < x ∧ x < 2}

-- Define the expected intersection
def expected_intersection := {x : ℝ | 1 ≤ x ∧ x < 2}

-- The proof problem statement
theorem intersection_of_A_and_B :
  A ∩ B = expected_intersection := by
  sorry

end intersection_of_A_and_B_l1421_142127


namespace yogurt_combinations_l1421_142154

theorem yogurt_combinations : (4 * Nat.choose 8 3) = 224 := by
  sorry

end yogurt_combinations_l1421_142154


namespace divisibility_condition_l1421_142147

theorem divisibility_condition (M C D U A q1 q2 q3 r1 r2 r3 : ℕ)
  (h1 : 10 = A * q1 + r1)
  (h2 : 10 * r1 = A * q2 + r2)
  (h3 : 10 * r2 = A * q3 + r3) :
  (U + D * r1 + C * r2 + M * r3) % A = 0 ↔ (1000 * M + 100 * C + 10 * D + U) % A = 0 :=
sorry

end divisibility_condition_l1421_142147


namespace ratio_of_doctors_to_lawyers_l1421_142178

variable (d l : ℕ) -- number of doctors and lawyers
variable (h1 : (40 * d + 55 * l) / (d + l) = 45) -- overall average age condition

theorem ratio_of_doctors_to_lawyers : d = 2 * l :=
by
  sorry

end ratio_of_doctors_to_lawyers_l1421_142178


namespace find_c_l1421_142195

open Function

noncomputable def g (x : ℝ) : ℝ :=
  (x - 4) * (x - 2) * x * (x + 2) * (x + 4) / 255 - 5

theorem find_c (c : ℤ) :
  (∃ x₁ x₂ x₃ x₄ : ℝ, g x₁ = c ∧ g x₂ = c ∧ g x₃ = c ∧ g x₄ = c ∧
    x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄) →
  ∀ k : ℤ, k < c → ¬ ∃ x₁ x₂ x₃ x₄ : ℝ, g x₁ = k ∧ g x₂ = k ∧ g x₃ = k ∧ g x₄ = k ∧
    x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ :=
sorry

end find_c_l1421_142195


namespace waiters_dropped_out_l1421_142198

theorem waiters_dropped_out (initial_chefs initial_waiters chefs_dropped remaining_staff : ℕ)
  (h1 : initial_chefs = 16) 
  (h2 : initial_waiters = 16) 
  (h3 : chefs_dropped = 6) 
  (h4 : remaining_staff = 23) : 
  initial_waiters - (remaining_staff - (initial_chefs - chefs_dropped)) = 3 := 
by 
  sorry

end waiters_dropped_out_l1421_142198


namespace max_value_inequality_l1421_142175

theorem max_value_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 5 * x + 3 * y < 90) : 
  x * y * (90 - 5 * x - 3 * y) ≤ 1800 := 
sorry

end max_value_inequality_l1421_142175


namespace necklace_wire_length_l1421_142138

theorem necklace_wire_length
  (spools : ℕ)
  (feet_per_spool : ℕ)
  (total_necklaces : ℕ)
  (h1 : spools = 3)
  (h2 : feet_per_spool = 20)
  (h3 : total_necklaces = 15) :
  (spools * feet_per_spool) / total_necklaces = 4 := by
  sorry

end necklace_wire_length_l1421_142138


namespace smart_charging_piles_growth_l1421_142181

-- Define the conditions
variables {x : ℝ}

-- First month charging piles
def first_month_piles : ℝ := 301

-- Third month charging piles
def third_month_piles : ℝ := 500

-- The theorem stating the relationship between the first and third month
theorem smart_charging_piles_growth : 
  first_month_piles * (1 + x) ^ 2 = third_month_piles :=
by
  sorry

end smart_charging_piles_growth_l1421_142181


namespace total_tiles_l1421_142169

theorem total_tiles (n : ℕ) (h : 2 * n - 1 = 133) : n^2 = 4489 :=
by
  sorry

end total_tiles_l1421_142169


namespace cone_base_circumference_l1421_142167

-- Definitions of the problem
def radius : ℝ := 5
def angle_sector_degree : ℝ := 120
def full_circle_degree : ℝ := 360

-- Proof statement
theorem cone_base_circumference 
  (r : ℝ) (angle_sector : ℝ) (full_angle : ℝ) 
  (h1 : r = radius) 
  (h2 : angle_sector = angle_sector_degree) 
  (h3 : full_angle = full_circle_degree) : 
  (angle_sector / full_angle) * (2 * π * r) = (10 * π) / 3 := 
by sorry

end cone_base_circumference_l1421_142167


namespace min_value_frac_inverse_l1421_142174

theorem min_value_frac_inverse (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 2) : 
  (1 / a + 1 / b) >= 2 :=
by
  sorry

end min_value_frac_inverse_l1421_142174


namespace price_reduction_relationship_l1421_142151

variable (a : ℝ) -- original price a in yuan
variable (b : ℝ) -- final price b in yuan

-- condition: price decreased by 10% first
def priceAfterFirstReduction := a * (1 - 0.10)

-- condition: price decreased by 20% on the result of the first reduction
def finalPrice := priceAfterFirstReduction a * (1 - 0.20)

-- theorem: relationship between original price a and final price b
theorem price_reduction_relationship (h : b = finalPrice a) : 
  b = a * (1 - 0.10) * (1 - 0.20) :=
by
  -- proof would go here
  sorry

end price_reduction_relationship_l1421_142151


namespace horse_problem_l1421_142110

theorem horse_problem (x : ℕ) :
  150 * (x + 12) = 240 * x :=
sorry

end horse_problem_l1421_142110


namespace simplify_fraction_l1421_142155

theorem simplify_fraction : (5^5 + 5^3) / (5^4 - 5^2) = 65 / 12 := by
  sorry

end simplify_fraction_l1421_142155


namespace symmetric_circle_eqn_l1421_142136

theorem symmetric_circle_eqn :
  ∀ (x y : ℝ),
  ((x + 1)^2 + (y - 1)^2 = 1) ∧ (x - y - 1 = 0) →
  (∀ (x' y' : ℝ), (x' = y + 1) ∧ (y' = x - 1) → (x' + 1)^2 + (y' - 1)^2 = 1) →
  (x - 2)^2 + (y + 2)^2 = 1 :=
by
  intros x y h h_sym
  sorry

end symmetric_circle_eqn_l1421_142136


namespace pats_and_mats_numbers_l1421_142129

theorem pats_and_mats_numbers (x y : ℕ) (hxy : x ≠ y) (hx_gt_hy : x > y) 
    (h_sum : (x + y) + (x - y) + x * y + (x / y) = 98) : x = 12 ∧ y = 6 :=
by
  sorry

end pats_and_mats_numbers_l1421_142129


namespace probability_at_least_two_students_succeeding_l1421_142160

-- The probabilities of each student succeeding
def p1 : ℚ := 1 / 2
def p2 : ℚ := 1 / 4
def p3 : ℚ := 1 / 5

/-- Calculation of the total probability that at least two out of the three students succeed -/
theorem probability_at_least_two_students_succeeding : 
  (p1 * p2 * (1 - p3)) + (p1 * (1 - p2) * p3) + ((1 - p1) * p2 * p3) + (p1 * p2 * p3) = 9 / 40 :=
  sorry

end probability_at_least_two_students_succeeding_l1421_142160


namespace total_nails_needed_l1421_142193

-- Given conditions
def nails_per_plank : ℕ := 2
def number_of_planks : ℕ := 16

-- Prove the total number of nails required
theorem total_nails_needed : nails_per_plank * number_of_planks = 32 :=
by
  sorry

end total_nails_needed_l1421_142193


namespace tangent_intersection_locus_l1421_142173

theorem tangent_intersection_locus :
  ∀ (l : ℝ → ℝ) (C : ℝ → ℝ), 
  (∀ x > 0, C x = x + 1/x) →
  (∃ k : ℝ, ∀ x, l x = k * x + 1) →
  ∃ (P : ℝ × ℝ), (P = (2, 2)) ∨ (P = (2, 5/2)) :=
by sorry

end tangent_intersection_locus_l1421_142173


namespace reciprocal_inequality_reciprocal_inequality_opposite_l1421_142100

theorem reciprocal_inequality (a b : ℝ) (h1 : a > b) (h2 : ab > 0) : (1 / a < 1 / b) := 
sorry

theorem reciprocal_inequality_opposite (a b : ℝ) (h1 : a > b) (h2 : ab < 0) : (1 / a > 1 / b) := 
sorry

end reciprocal_inequality_reciprocal_inequality_opposite_l1421_142100


namespace budget_equality_year_l1421_142132

theorem budget_equality_year :
  let budget_q_1990 := 540000
  let budget_v_1990 := 780000
  let annual_increase_q := 30000
  let annual_decrease_v := 10000

  let budget_q (n : ℕ) := budget_q_1990 + n * annual_increase_q
  let budget_v (n : ℕ) := budget_v_1990 - n * annual_decrease_v

  (∃ n : ℕ, budget_q n = budget_v n ∧ 1990 + n = 1996) :=
by
  sorry

end budget_equality_year_l1421_142132


namespace range_of_k_for_positivity_l1421_142101

theorem range_of_k_for_positivity (k x : ℝ) (h1 : -1 ≤ x) (h2 : x ≤ 2) :
  ((k - 2) * x + 2 * |k| - 1 > 0) → (k > 5 / 4) :=
sorry

end range_of_k_for_positivity_l1421_142101


namespace speed_increase_71_6_percent_l1421_142118

theorem speed_increase_71_6_percent (S : ℝ) (hS : 0 < S) : 
    let S₁ := S * 1.30
    let S₂ := S₁ * 1.10
    let S₃ := S₂ * 1.20
    (S₃ - S) / S * 100 = 71.6 :=
by
  let S₁ := S * 1.30
  let S₂ := S₁ * 1.10
  let S₃ := S₂ * 1.20
  sorry

end speed_increase_71_6_percent_l1421_142118


namespace marriage_year_proof_l1421_142150

-- Definitions based on conditions
def marriage_year : ℕ := sorry
def child1_birth_year : ℕ := 1982
def child2_birth_year : ℕ := 1984
def reference_year : ℕ := 1986

-- Age calculations based on reference year
def age_in_1986 (birth_year : ℕ) : ℕ := reference_year - birth_year

-- Combined ages in the reference year
def combined_ages_in_1986 : ℕ := age_in_1986 child1_birth_year + age_in_1986 child2_birth_year

-- The main theorem to prove
theorem marriage_year_proof :
  combined_ages_in_1986 = reference_year - marriage_year →
  marriage_year = 1980 := by
  sorry

end marriage_year_proof_l1421_142150


namespace largest_cube_edge_length_l1421_142191

theorem largest_cube_edge_length (a : ℕ) : 
  (6 * a ^ 2 ≤ 1500) ∧
  (a * 15 ≤ 60) ∧
  (a * 15 ≤ 25) →
  a ≤ 15 :=
by
  sorry

end largest_cube_edge_length_l1421_142191


namespace probability_two_hearts_is_one_seventeenth_l1421_142112

-- Define the problem parameters
def totalCards : ℕ := 52
def hearts : ℕ := 13
def drawCount : ℕ := 2

-- Define function to calculate combinations
def combination (n k : ℕ) : ℕ :=
  n.choose k

-- Define the probability calculation
def probability_drawing_two_hearts : ℚ :=
  (combination hearts drawCount) / (combination totalCards drawCount)

-- State the theorem to be proved
theorem probability_two_hearts_is_one_seventeenth :
  probability_drawing_two_hearts = 1 / 17 :=
by
  -- Proof not required, so provide sorry
  sorry

end probability_two_hearts_is_one_seventeenth_l1421_142112


namespace intersection_of_A_and_B_eq_C_l1421_142142

noncomputable def A (x : ℝ) : Prop := x^2 - 4*x + 3 < 0
noncomputable def B (x : ℝ) : Prop := 2 - x > 0
noncomputable def A_inter_B (x : ℝ) : Prop := A x ∧ B x

theorem intersection_of_A_and_B_eq_C :
  {x : ℝ | A_inter_B x} = {x : ℝ | 1 < x ∧ x < 2} :=
by sorry

end intersection_of_A_and_B_eq_C_l1421_142142


namespace students_walk_fraction_l1421_142182

theorem students_walk_fraction :
  (1 - (1/3 + 1/5 + 1/10 + 1/15)) = 3/10 :=
by sorry

end students_walk_fraction_l1421_142182


namespace power_modulo_l1421_142176

theorem power_modulo (a b c n : ℕ) (h1 : a = 17) (h2 : b = 1999) (h3 : c = 29) (h4 : n = a^b % c) : 
  n = 17 := 
by
  -- Note: Additional assumptions and intermediate calculations could be provided as needed
  sorry

end power_modulo_l1421_142176


namespace total_baseball_fans_l1421_142152

variable (Y M R : ℕ)

open Nat

theorem total_baseball_fans (h1 : 3 * M = 2 * Y) 
    (h2 : 4 * R = 5 * M) 
    (h3 : M = 96) : Y + M + R = 360 := by
  sorry

end total_baseball_fans_l1421_142152


namespace hyperbola_representation_l1421_142166

variable (x y : ℝ)

/--
Given the equation (x - y)^2 = 3(x^2 - y^2), we prove that
the resulting graph represents a hyperbola.
-/
theorem hyperbola_representation :
  (x - y)^2 = 3 * (x^2 - y^2) →
  ∃ A B C : ℝ, A ≠ 0 ∧ (x^2 + x * y - 2 * y^2 = 0) ∧ (A = 1) ∧ (B = 1) ∧ (C = -2) ∧ (B^2 - 4*A*C > 0) :=
by
  sorry

end hyperbola_representation_l1421_142166
