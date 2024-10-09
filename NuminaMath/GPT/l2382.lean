import Mathlib

namespace sum_of_interior_angles_of_pentagon_l2382_238289

theorem sum_of_interior_angles_of_pentagon : (5 - 2) * 180 = 540 := 
by
  sorry

end sum_of_interior_angles_of_pentagon_l2382_238289


namespace diminished_value_160_l2382_238208

theorem diminished_value_160 (x : ℕ) (n : ℕ) : 
  (∀ m, m > 200 ∧ (∀ k, m = k * 180) → n = m) →
  (200 + x = n) →
  x = 160 :=
by
  sorry

end diminished_value_160_l2382_238208


namespace focus_of_parabola_l2382_238239

theorem focus_of_parabola (x y : ℝ) (h : x^2 = 16 * y) : (0, 4) = (0, 4) :=
by {
  sorry
}

end focus_of_parabola_l2382_238239


namespace price_per_pie_l2382_238211

-- Define the relevant variables and conditions
def cost_pumpkin_pie : ℕ := 3
def num_pumpkin_pies : ℕ := 10
def cost_cherry_pie : ℕ := 5
def num_cherry_pies : ℕ := 12
def desired_profit : ℕ := 20

-- Total production and profit calculation
def total_cost : ℕ := (cost_pumpkin_pie * num_pumpkin_pies) + (cost_cherry_pie * num_cherry_pies)
def total_earnings_needed : ℕ := total_cost + desired_profit
def total_pies : ℕ := num_pumpkin_pies + num_cherry_pies

-- Proposition to prove that the price per pie should be $5
theorem price_per_pie : (total_earnings_needed / total_pies) = 5 := by
  sorry

end price_per_pie_l2382_238211


namespace find_k_l2382_238284

theorem find_k (x k : ℝ) (h1 : (x^2 - k) * (x + k) = x^3 + k * (x^2 - x - 2)) (h2 : k ≠ 0) :
  k = 2 :=
sorry

end find_k_l2382_238284


namespace wendys_brother_pieces_l2382_238273

-- Definitions based on conditions
def number_of_boxes : ℕ := 2
def pieces_per_box : ℕ := 3
def total_pieces : ℕ := 12

-- Summarization of Wendy's pieces of candy
def wendys_pieces : ℕ := number_of_boxes * pieces_per_box

-- Lean statement: Prove the number of pieces Wendy's brother had
theorem wendys_brother_pieces : total_pieces - wendys_pieces = 6 :=
by
  sorry

end wendys_brother_pieces_l2382_238273


namespace fraction_equiv_l2382_238292

theorem fraction_equiv (x y : ℚ) (h : (5/6) * 192 = (x/y) * 192 + 100) : x/y = 5/16 :=
sorry

end fraction_equiv_l2382_238292


namespace michael_eggs_count_l2382_238261

def initial_crates : List ℕ := [24, 28, 32, 36, 40, 44]
def wednesday_given : List ℕ := [28, 32, 40]
def thursday_purchases : List ℕ := [50, 45, 55, 60]
def friday_sold : List ℕ := [60, 55]

theorem michael_eggs_count :
  let total_tuesday := initial_crates.sum
  let total_given_wednesday := wednesday_given.sum
  let remaining_wednesday := total_tuesday - total_given_wednesday
  let total_thursday := thursday_purchases.sum
  let total_after_thursday := remaining_wednesday + total_thursday
  let total_sold_friday := friday_sold.sum
  total_after_thursday - total_sold_friday = 199 :=
by
  sorry

end michael_eggs_count_l2382_238261


namespace sum_of_integers_l2382_238205

theorem sum_of_integers (x y : ℕ) (h1 : x - y = 8) (h2 : x * y = 288) : x + y = 35 :=
sorry

end sum_of_integers_l2382_238205


namespace sum_of_reciprocals_roots_transformed_eq_neg11_div_4_l2382_238209

theorem sum_of_reciprocals_roots_transformed_eq_neg11_div_4 :
  (∃ a b c : ℝ, (a^3 - a - 2 = 0) ∧ (b^3 - b - 2 = 0) ∧ (c^3 - c - 2 = 0)) → 
  ( ∃ a b c : ℝ, a^3 - a - 2 = 0 ∧ b^3 - b - 2 = 0 ∧ c^3 - c - 2 = 0 ∧ 
  (1 / (a - 2) + 1 / (b - 2) + 1 / (c - 2) = - 11 / 4)) :=
by
  sorry

end sum_of_reciprocals_roots_transformed_eq_neg11_div_4_l2382_238209


namespace chess_team_boys_l2382_238279

variable (B G : ℕ)

theorem chess_team_boys (h1 : B + G = 30) (h2 : (1 / 3 : ℝ) * G + B = 20) : B = 15 := by
  sorry

end chess_team_boys_l2382_238279


namespace first_discount_is_10_l2382_238245

def list_price : ℝ := 70
def final_price : ℝ := 59.85
def second_discount : ℝ := 0.05

theorem first_discount_is_10 :
  ∃ (x : ℝ), list_price * (1 - x/100) * (1 - second_discount) = final_price ∧ x = 10 :=
by
  sorry

end first_discount_is_10_l2382_238245


namespace total_questions_on_test_l2382_238233

/-- A teacher grades students' tests by subtracting twice the number of incorrect responses
    from the number of correct responses. Given that a student received a score of 64
    and answered 88 questions correctly, prove that the total number of questions on the test is 100. -/
theorem total_questions_on_test (score correct_responses : ℕ) (grading_system : ℕ → ℕ → ℕ)
  (h1 : score = grading_system correct_responses (88 - 2 * 12))
  (h2 : correct_responses = 88)
  (h3 : score = 64) : correct_responses + (88 - 2 * 12) = 100 :=
by
  sorry

end total_questions_on_test_l2382_238233


namespace calc_result_l2382_238254

theorem calc_result : (-3)^2 - (-2)^3 = 17 := 
by
  sorry

end calc_result_l2382_238254


namespace average_monthly_balance_l2382_238294

theorem average_monthly_balance :
  let balances := [100, 200, 250, 50, 300, 300]
  (balances.sum / balances.length : ℕ) = 200 :=
by
  sorry

end average_monthly_balance_l2382_238294


namespace lcm_28_72_l2382_238291

theorem lcm_28_72 : Nat.lcm 28 72 = 504 := by
  sorry

end lcm_28_72_l2382_238291


namespace journey_time_l2382_238280

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

end journey_time_l2382_238280


namespace part1_part2_l2382_238212

variable {f : ℝ → ℝ}

-- Condition 1: f is an odd function
def isOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Condition 2: ∀ a b ∈ ℝ, (a + b ≠ 0) → (f(a) + f(b))/(a + b) > 0
def positiveQuotient (f : ℝ → ℝ) : Prop :=
  ∀ a b, a + b ≠ 0 → (f a + f b) / (a + b) > 0

-- Sub-problem (1): For any a, b ∈ ℝ, a > b ⟹ f(a) > f(b)
theorem part1 (h_odd : isOddFunction f) (h_posQuot : positiveQuotient f) (a b : ℝ) (h : a > b) : f a > f b :=
  sorry

-- Sub-problem (2): If f(9^x - 2 * 3^x) + f(2 * 9^x - k) > 0 for any x ∈ [0, ∞), then k < 1
theorem part2 (h_odd : isOddFunction f) (h_posQuot : positiveQuotient f) :
  (∀ x : ℝ, 0 ≤ x → f (9^x - 2 * 3^x) + f (2 * 9^x - k) > 0) → k < 1 :=
  sorry

end part1_part2_l2382_238212


namespace wall_area_l2382_238229

-- Define the conditions
variables (R J D : ℕ) (L W : ℝ)
variable (area_regular_tiles : ℝ)
variables (ratio_regular : ℕ) (ratio_jumbo : ℕ) (ratio_diamond : ℕ)
variables (length_ratio_jumbo : ℝ) (width_ratio_jumbo : ℝ)
variables (length_ratio_diamond : ℝ) (width_ratio_diamond : ℝ)
variable (total_area : ℝ)

-- Assign values to the conditions
axiom ratio : ratio_regular = 4 ∧ ratio_jumbo = 2 ∧ ratio_diamond = 1
axiom size_regular : area_regular_tiles = 80
axiom jumbo_tile_ratio : length_ratio_jumbo = 3 ∧ width_ratio_jumbo = 3
axiom diamond_tile_ratio : length_ratio_diamond = 2 ∧ width_ratio_diamond = 0.5

-- Define the statement
theorem wall_area (ratio : ratio_regular = 4 ∧ ratio_jumbo = 2 ∧ ratio_diamond = 1)
    (size_regular : area_regular_tiles = 80)
    (jumbo_tile_ratio : length_ratio_jumbo = 3 ∧ width_ratio_jumbo = 3)
    (diamond_tile_ratio : length_ratio_diamond = 2 ∧ width_ratio_diamond = 0.5):
    total_area = 140 := 
sorry

end wall_area_l2382_238229


namespace inequality_holds_l2382_238237

variable (f : ℝ → ℝ)
variable (a : ℝ)

-- Conditions
def even_function : Prop := ∀ x : ℝ, f x = f (-x)
def decreasing_on_pos : Prop := ∀ x y : ℝ, 0 < x → x < y → f y ≤ f x

-- Proof goal
theorem inequality_holds (h_even : even_function f) (h_decreasing : decreasing_on_pos f) : 
  f (-3/4) ≥ f (a^2 - a + 1) := 
by
  sorry

end inequality_holds_l2382_238237


namespace initialNumberMembers_l2382_238264

-- Define the initial number of members in the group
def initialMembers (n : ℕ) : Prop :=
  let W := n * 48 -- Initial total weight
  let newWeight := W + 78 + 93 -- New total weight after two members join
  let newAverageWeight := (n + 2) * 51 -- New total weight based on the new average weight
  newWeight = newAverageWeight -- The condition that the new total weights are equal

-- Theorem stating that the initial number of members is 23
theorem initialNumberMembers : initialMembers 23 :=
by
  -- Placeholder for proof steps
  sorry

end initialNumberMembers_l2382_238264


namespace probability_of_non_defective_product_l2382_238203

-- Define the probability of producing a grade B product
def P_B : ℝ := 0.03

-- Define the probability of producing a grade C product
def P_C : ℝ := 0.01

-- Define the probability of producing a non-defective product (grade A)
def P_A : ℝ := 1 - P_B - P_C

-- The theorem to prove: The probability of producing a non-defective product is 0.96
theorem probability_of_non_defective_product : P_A = 0.96 := by
  -- Insert proof here
  sorry

end probability_of_non_defective_product_l2382_238203


namespace father_three_times_marika_in_year_l2382_238250

-- Define the given conditions as constants.
def marika_age_2004 : ℕ := 8
def father_age_2004 : ℕ := 32

-- Define the proof goal.
theorem father_three_times_marika_in_year :
  ∃ (x : ℕ), father_age_2004 + x = 3 * (marika_age_2004 + x) → 2004 + x = 2008 := 
by {
  sorry
}

end father_three_times_marika_in_year_l2382_238250


namespace sum_of_squares_of_consecutive_even_numbers_l2382_238252

theorem sum_of_squares_of_consecutive_even_numbers :
  ∃ (x : ℤ), x + (x + 2) + (x + 4) + (x + 6) = 36 → (x ^ 2 + (x + 2) ^ 2 + (x + 4) ^ 2 + (x + 6) ^ 2 = 344) :=
by
  sorry

end sum_of_squares_of_consecutive_even_numbers_l2382_238252


namespace smallest_element_in_M_l2382_238221

def f : ℝ → ℝ := sorry
axiom f1 (x y : ℝ) (h1 : x ≥ 1) (h2 : y = 3 * x) : f y = 3 * f x
axiom f2 (x : ℝ) (h : 1 ≤ x ∧ x ≤ 3) : f x = 1 - abs (x - 2)
axiom f99_value : f 99 = 18

theorem smallest_element_in_M : ∃ x : ℝ, x = 45 ∧ f x = 18 := by
  -- proof will be provided later
  sorry

end smallest_element_in_M_l2382_238221


namespace smallest_sum_arith_geo_seq_l2382_238246

theorem smallest_sum_arith_geo_seq (A B C D : ℕ) 
  (h1 : A + B + C + D > 0)
  (h2 : 2 * B = A + C)
  (h3 : 16 * C = 7 * B)
  (h4 : 16 * D = 49 * B) :
  A + B + C + D = 97 :=
sorry

end smallest_sum_arith_geo_seq_l2382_238246


namespace roman_numeral_sketching_l2382_238231

/-- Roman numeral sketching problem. -/
theorem roman_numeral_sketching (n : ℕ) (k : ℕ) (students : ℕ) 
  (h1 : ∀ i : ℕ, 1 ≤ i ∧ i ≤ n ∧ i / 1 = i) 
  (h2 : ∀ i : ℕ, i > n → i = n - (i - n)) 
  (h3 : k = 7) 
  (h4 : ∀ r : ℕ, r = (k * n)) : students = 350 :=
by
  sorry

end roman_numeral_sketching_l2382_238231


namespace sin_four_arcsin_eq_l2382_238268

theorem sin_four_arcsin_eq (x : ℝ) (h : -1 ≤ x ∧ x ≤ 1) : 
  Real.sin (4 * Real.arcsin x) = 4 * x * (1 - 2 * x^2) * Real.sqrt (1 - x^2) :=
by
  sorry

end sin_four_arcsin_eq_l2382_238268


namespace target_hit_probability_l2382_238218

-- Defining the probabilities for A, B, and C hitting the target.
def P_A_hit := 1 / 2
def P_B_hit := 1 / 3
def P_C_hit := 1 / 4

-- Defining the probability that A, B, and C miss the target.
def P_A_miss := 1 - P_A_hit
def P_B_miss := 1 - P_B_hit
def P_C_miss := 1 - P_C_hit

-- Calculating the combined probability that none of them hit the target.
def P_none_hit := P_A_miss * P_B_miss * P_C_miss

-- Now, calculating the probability that at least one of them hits the target.
def P_hit := 1 - P_none_hit

-- Statement of the theorem.
theorem target_hit_probability : P_hit = 3 / 4 := by
  sorry

end target_hit_probability_l2382_238218


namespace polynomial_simplification_l2382_238285

noncomputable def given_polynomial (x : ℝ) : ℝ :=
  3 - 5 * x - 7 * x^2 + 9 + 11 * x - 13 * x^2 + 15 - 17 * x + 19 * x^2 + 2 * x^3

theorem polynomial_simplification (x : ℝ) :
  given_polynomial x = 2 * x^3 - x^2 - 11 * x + 27 :=
by
  -- The proof is skipped
  sorry

end polynomial_simplification_l2382_238285


namespace problem_expression_value_l2382_238259

theorem problem_expression_value :
  (1 * 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9 : ℤ) / (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 : ℤ) = 6608 :=
by
  sorry

end problem_expression_value_l2382_238259


namespace find_coefficient_of_x_in_expansion_l2382_238255

noncomputable def coefficient_of_x_in_expansion (x : ℤ) : ℤ :=
  (1 / 2 * x - 1) * (2 * x - 1 / x) ^ 6

theorem find_coefficient_of_x_in_expansion :
  coefficient_of_x_in_expansion x = -80 :=
by {
  sorry
}

end find_coefficient_of_x_in_expansion_l2382_238255


namespace average_of_three_l2382_238286

theorem average_of_three (y : ℝ) (h : (15 + 24 + y) / 3 = 20) : y = 21 :=
by
  sorry

end average_of_three_l2382_238286


namespace train_crosses_bridge_in_time_l2382_238226

noncomputable def length_of_train : ℝ := 125
noncomputable def length_of_bridge : ℝ := 250.03
noncomputable def speed_of_train_kmh : ℝ := 45

noncomputable def speed_of_train_ms : ℝ := (speed_of_train_kmh * 1000) / 3600
noncomputable def total_distance : ℝ := length_of_train + length_of_bridge
noncomputable def time_to_cross_bridge : ℝ := total_distance / speed_of_train_ms

theorem train_crosses_bridge_in_time :
  time_to_cross_bridge = 30.0024 :=
  sorry

end train_crosses_bridge_in_time_l2382_238226


namespace power_multiplication_l2382_238240

theorem power_multiplication :
  2^4 * 5^4 = 10000 := 
by
  sorry

end power_multiplication_l2382_238240


namespace normal_distribution_test_l2382_238243

noncomputable def normal_distribution_at_least_90 : Prop :=
  let μ := 78
  let σ := 4
  -- Given reference data
  let p_within_3_sigma := 0.9974
  -- Calculate P(X >= 90)
  let p_at_least_90 := (1 - p_within_3_sigma) / 2
  -- The expected answer 0.13% ⇒ 0.0013
  p_at_least_90 = 0.0013

theorem normal_distribution_test :
  normal_distribution_at_least_90 :=
by
  sorry

end normal_distribution_test_l2382_238243


namespace find_factor_l2382_238244

theorem find_factor (x f : ℕ) (h1 : x = 15) (h2 : (2 * x + 5) * f = 105) : f = 3 :=
sorry

end find_factor_l2382_238244


namespace smallest_positive_x_l2382_238206

theorem smallest_positive_x (x : ℝ) (h : ⌊x^2⌋ - x * ⌊x⌋ = 8) : x = 89 / 9 :=
sorry

end smallest_positive_x_l2382_238206


namespace gcd_654327_543216_is_1_l2382_238241

-- Define the gcd function and relevant numbers
def gcd_problem : Prop :=
  gcd 654327 543216 = 1

-- The statement of the theorem, with a placeholder for the proof
theorem gcd_654327_543216_is_1 : gcd_problem :=
by {
  -- actual proof will go here
  sorry
}

end gcd_654327_543216_is_1_l2382_238241


namespace largest_integer_divides_product_l2382_238265

theorem largest_integer_divides_product (n : ℕ) : 
  ∃ m, ∀ k : ℕ, k = (2*n-1)*(2*n)*(2*n+2) → m ≥ 1 ∧ m = 8 ∧ m ∣ k :=
by
  sorry

end largest_integer_divides_product_l2382_238265


namespace snow_leopards_arrangement_l2382_238207

theorem snow_leopards_arrangement : 
  ∃ (perm : Fin 9 → Fin 9), 
    (∀ i, perm i ≠ perm j → i ≠ j) ∧ 
    (perm 0 < perm 1 ∧ perm 8 < perm 1 ∧ perm 0 < perm 8) ∧ 
    (∃ count_ways, count_ways = 4320) :=
sorry

end snow_leopards_arrangement_l2382_238207


namespace find_b_value_l2382_238281

-- Let's define the given conditions as hypotheses in Lean

theorem find_b_value 
  (x1 y1 x2 y2 : ℤ) 
  (h1 : (x1, y1) = (2, 2)) 
  (h2 : (x2, y2) = (8, 14)) 
  (midpoint : ∃ (m1 m2 : ℤ), m1 = (x1 + x2) / 2 ∧ m2 = (y1 + y2) / 2 ∧ (m1, m2) = (5, 8))
  (perpendicular_bisector : ∀ (x y : ℤ), x + y = b → (x, y) = (5, 8)) :
  b = 13 := 
by {
  sorry
}

end find_b_value_l2382_238281


namespace greg_age_is_16_l2382_238290

-- Definitions based on given conditions
def cindy_age : ℕ := 5
def jan_age : ℕ := cindy_age + 2
def marcia_age : ℕ := 2 * jan_age
def greg_age : ℕ := marcia_age + 2

-- Theorem stating that Greg's age is 16 years given the above conditions
theorem greg_age_is_16 : greg_age = 16 := by
  sorry

end greg_age_is_16_l2382_238290


namespace probability_equals_two_thirds_l2382_238297

-- Definitions for total arrangements and favorable arrangements
def total_arrangements : ℕ := Nat.choose 6 2
def favorable_arrangements : ℕ := Nat.choose 5 2

-- Probability that 2 zeros are not adjacent
def probability_not_adjacent : ℚ := favorable_arrangements / total_arrangements

theorem probability_equals_two_thirds : probability_not_adjacent = 2 / 3 := 
by 
  let total_arrangements := 15
  let favorable_arrangements := 10
  have h1 : probability_not_adjacent = (10 : ℚ) / (15 : ℚ) := rfl
  have h2 : (10 : ℚ) / (15 : ℚ) = 2 / 3 := by norm_num
  exact Eq.trans h1 h2 

end probability_equals_two_thirds_l2382_238297


namespace smallest_positive_solution_eq_sqrt_29_l2382_238296

theorem smallest_positive_solution_eq_sqrt_29 :
  ∃ x : ℝ, 0 < x ∧ x^4 - 58 * x^2 + 841 = 0 ∧ x = Real.sqrt 29 :=
by
  sorry

end smallest_positive_solution_eq_sqrt_29_l2382_238296


namespace rectangle_area_l2382_238266

theorem rectangle_area (r : ℝ) (w l : ℝ) (h_radius : r = 7) 
  (h_ratio : l = 3 * w) (h_width : w = 2 * r) : l * w = 588 :=
by
  sorry

end rectangle_area_l2382_238266


namespace number_of_cows_l2382_238293

theorem number_of_cows (H : ℕ) (C : ℕ) (h1 : H = 6) (h2 : C / H = 7 / 2) : C = 21 :=
by
  sorry

end number_of_cows_l2382_238293


namespace complement_intersection_eq_l2382_238287

variable (U P Q : Set ℕ)
variable (hU : U = {1, 2, 3})
variable (hP : P = {1, 2})
variable (hQ : Q = {2, 3})

theorem complement_intersection_eq : 
  (U \ (P ∩ Q)) = {1, 3} := by
  sorry

end complement_intersection_eq_l2382_238287


namespace prime_count_at_least_two_l2382_238248

theorem prime_count_at_least_two :
  ∃ (n1 n2 : ℕ), n1 ≥ 2 ∧ n2 ≥ 2 ∧ (n1 ≠ n2) ∧ Prime (n1^3 + n1^2 + 1) ∧ Prime (n2^3 + n2^2 + 1) := 
by
  sorry

end prime_count_at_least_two_l2382_238248


namespace max_obtuse_angles_in_quadrilateral_l2382_238299

theorem max_obtuse_angles_in_quadrilateral (a b c d : ℝ) 
  (h₁ : a + b + c + d = 360)
  (h₂ : 90 < a)
  (h₃ : 90 < b)
  (h₄ : 90 < c) :
  90 > d :=
sorry

end max_obtuse_angles_in_quadrilateral_l2382_238299


namespace kaleb_clothing_problem_l2382_238213

theorem kaleb_clothing_problem 
  (initial_clothing : ℕ) 
  (one_load : ℕ) 
  (remaining_loads : ℕ) : 
  initial_clothing = 39 → one_load = 19 → remaining_loads = 5 → (initial_clothing - one_load) / remaining_loads = 4 :=
sorry

end kaleb_clothing_problem_l2382_238213


namespace wax_current_eq_l2382_238271

-- Define the constants for the wax required and additional wax needed
def w_required : ℕ := 166
def w_more : ℕ := 146

-- Define the term to represent the current wax he has
def w_current : ℕ := w_required - w_more

-- Theorem statement to prove the current wax quantity
theorem wax_current_eq : w_current = 20 := by
  -- Proof outline would go here, but per instructions, we skip with sorry
  sorry

end wax_current_eq_l2382_238271


namespace find_bc_div_a_l2382_238298

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin x + 2 * Real.cos x + 1

variable (a b c : ℝ)

def satisfied (x : ℝ) : Prop := a * f x + b * f (x - c) = 1

theorem find_bc_div_a (ha : ∀ x, satisfied a b c x) : (b * Real.cos c / a) = -1 := 
by sorry

end find_bc_div_a_l2382_238298


namespace find_n_in_range_l2382_238234

theorem find_n_in_range :
  ∃ n : ℕ, n > 1 ∧ 
           n % 3 = 2 ∧ 
           n % 5 = 2 ∧ 
           n % 7 = 2 ∧ 
           101 ≤ n ∧ n ≤ 134 :=
by sorry

end find_n_in_range_l2382_238234


namespace goods_train_speed_l2382_238215

theorem goods_train_speed 
  (length_train : ℕ)
  (length_platform : ℕ)
  (time_to_cross : ℕ)
  (h_train : length_train = 270)
  (h_platform : length_platform = 250)
  (h_time : time_to_cross = 26) : 
  (length_train + length_platform) / time_to_cross = 20 := 
by
  sorry

end goods_train_speed_l2382_238215


namespace no_three_distinct_integers_solving_polynomial_l2382_238256

theorem no_three_distinct_integers_solving_polynomial (p : ℤ → ℤ) (hp : ∀ x, ∃ k : ℕ, p x = k • x + p 0) :
  ∀ a b c : ℤ, a ≠ b → b ≠ c → c ≠ a → p a = b → p b = c → p c = a → false :=
by
  intros a b c hab hbc hca hpa_hp pb_pc_pc
  sorry

end no_three_distinct_integers_solving_polynomial_l2382_238256


namespace log_relationships_l2382_238201

theorem log_relationships (c d y : ℝ) (hc : c > 0) (hd : d > 0) (hy : y > 0) :
  9 * (Real.log y / Real.log c)^2 + 5 * (Real.log y / Real.log d)^2 = 18 * (Real.log y)^2 / (Real.log c * Real.log d) →
  d = c^(1 / Real.sqrt 3) ∨ d = c^(Real.sqrt 3) ∨ d = c^(1 / Real.sqrt (6 / 10)) ∨ d = c^(Real.sqrt (6 / 10)) :=
sorry

end log_relationships_l2382_238201


namespace difference_between_possible_x_values_l2382_238277

theorem difference_between_possible_x_values :
  ∀ (x : ℝ), (x + 3) ^ 2 / (2 * x + 15) = 3 → (x = 6 ∨ x = -6) →
  (abs (6 - (-6)) = 12) :=
by
  intro x h1 h2
  sorry

end difference_between_possible_x_values_l2382_238277


namespace playgroup_count_l2382_238202

-- Definitions based on the conditions
def total_people (girls boys parents : ℕ) := girls + boys + parents
def playgroups (total size_per_group : ℕ) := total / size_per_group

-- Statement of the problem
theorem playgroup_count (girls boys parents size_per_group : ℕ)
  (h_girls : girls = 14)
  (h_boys : boys = 11)
  (h_parents : parents = 50)
  (h_size_per_group : size_per_group = 25) :
  playgroups (total_people girls boys parents) size_per_group = 3 :=
by {
  -- This is just the statement, the proof is skipped with sorry
  sorry
}

end playgroup_count_l2382_238202


namespace hyperbola_line_intersection_unique_l2382_238257

theorem hyperbola_line_intersection_unique :
  ∀ (x y : ℝ), (x^2 / 9 - y^2 = 1) ∧ (y = 1/3 * (x + 1)) → ∃! p : ℝ × ℝ, p.1 = x ∧ p.2 = y :=
by
  sorry

end hyperbola_line_intersection_unique_l2382_238257


namespace find_pairs_l2382_238247

theorem find_pairs (x y : ℤ) (h : 19 / x + 96 / y = (19 * 96) / (x * y)) :
  ∃ m : ℤ, x = 19 * m ∧ y = 96 - 96 * m :=
by
  sorry

end find_pairs_l2382_238247


namespace basketball_game_score_l2382_238232

theorem basketball_game_score 
  (a r b d : ℕ)
  (H1 : a = b)
  (H2 : a + a * r + a * r^2 = b + (b + d) + (b + 2 * d))
  (H3 : a * (1 + r + r^2 + r^3) = 4 * b + 6 * d + 3)
  (H4 : r = 3)
  (H5 : a = 3)
  (H6 : d = 10)
  (H7 : a * (1 + r) = 12)
  (H8 : b * (1 + 3 + (b + d)) = 16) :
  a + a * r + b + (b + d) = 28 :=
by simp [H4, H5, H6, H7, H8]; linarith

end basketball_game_score_l2382_238232


namespace expression_value_l2382_238214

noncomputable def expression (x b : ℝ) : ℝ :=
  (x / (x + b) + b / (x - b)) / (b / (x + b) - x / (x - b))

theorem expression_value (b x : ℝ) (hb : b ≠ 0) (hx : x ≠ b ∧ x ≠ -b) :
  expression x b = -1 := 
by
  sorry

end expression_value_l2382_238214


namespace y_intercept_of_line_l2382_238275

theorem y_intercept_of_line (x y : ℝ) (h : 4 * x + 7 * y = 28) (hx : x = 0) : y = 4 :=
by
  -- The proof goes here
  sorry

end y_intercept_of_line_l2382_238275


namespace books_left_in_library_l2382_238276

theorem books_left_in_library (initial_books : ℕ) (borrowed_books : ℕ) (left_books : ℕ) 
  (h1 : initial_books = 75) (h2 : borrowed_books = 18) : left_books = 57 :=
by
  sorry

end books_left_in_library_l2382_238276


namespace negation_of_existential_statement_l2382_238269

theorem negation_of_existential_statement {f : ℝ → ℝ} :
  (¬ ∃ x₀ : ℝ, f x₀ < 0) ↔ (∀ x : ℝ, f x ≥ 0) :=
by
  sorry

end negation_of_existential_statement_l2382_238269


namespace percentage_of_men_35_l2382_238249

theorem percentage_of_men_35 (M W : ℝ) (hm1 : M + W = 100) 
  (hm2 : 0.6 * M + 0.2923 * W = 40)
  (hw : W = 100 - M) : 
  M = 35 :=
by
  sorry

end percentage_of_men_35_l2382_238249


namespace find_p_a_l2382_238225

variables (p : ℕ → ℝ) (a b : ℕ)

-- Given conditions
axiom p_b : p b = 0.5
axiom p_b_given_a : p b / p a = 0.2 
axiom p_a_inter_b : p a * p b = 0.36

-- Problem statement
theorem find_p_a : p a = 1.8 :=
by
  sorry

end find_p_a_l2382_238225


namespace Jordana_current_age_is_80_l2382_238251

-- Given conditions
def current_age_Jennifer := 20  -- since Jennifer will be 30 in ten years
def current_age_Jordana := 80  -- since the problem states we need to verify this

-- Prove that Jordana's current age is 80 years old given the conditions
theorem Jordana_current_age_is_80:
  (current_age_Jennifer + 10 = 30) →
  (current_age_Jordana + 10 = 3 * 30) →
  current_age_Jordana = 80 :=
by 
  intros h1 h2
  sorry

end Jordana_current_age_is_80_l2382_238251


namespace triangle_side_length_l2382_238242

theorem triangle_side_length 
  (A : ℝ) (a m n : ℝ) 
  (hA : A = 60) 
  (h1 : m + n = 7) 
  (h2 : m * n = 11) : a = 4 :=
by
  sorry

end triangle_side_length_l2382_238242


namespace tangent_line_at_pi_one_l2382_238263

noncomputable def function (x : ℝ) : ℝ := Real.exp x * Real.sin x + 1
noncomputable def tangent_line (x : ℝ) (y : ℝ) : ℝ := x * Real.exp Real.pi + y - 1 - Real.pi * Real.exp Real.pi

theorem tangent_line_at_pi_one :
  tangent_line x y = 0 ↔ y = function x → x = Real.pi ∧ y = 1 :=
by
  sorry

end tangent_line_at_pi_one_l2382_238263


namespace balls_in_boxes_l2382_238288

-- Define the conditions
def num_balls : ℕ := 3
def num_boxes : ℕ := 4

-- Define the problem
theorem balls_in_boxes : (num_boxes ^ num_balls) = 64 :=
by
  -- We acknowledge that we are skipping the proof details here
  sorry

end balls_in_boxes_l2382_238288


namespace probability_all_quitters_same_tribe_l2382_238236

-- Definitions of the problem conditions
def total_contestants : ℕ := 20
def tribe_size : ℕ := 10
def quitters : ℕ := 3

-- Definition of the binomial coefficient
def choose (n k : ℕ) : ℕ := Nat.choose n k

-- Theorem statement
theorem probability_all_quitters_same_tribe :
  (choose tribe_size quitters + choose tribe_size quitters) * 
  (total_contestants.choose quitters) = 240 
  ∧ ((choose tribe_size quitters + choose tribe_size quitters) / (total_contestants.choose quitters)) = 20 / 95 :=
by
  sorry

end probability_all_quitters_same_tribe_l2382_238236


namespace total_pennies_l2382_238200

theorem total_pennies (rachelle_pennies : ℕ) (gretchen_pennies : ℕ) (rocky_pennies : ℕ)
  (h1 : rachelle_pennies = 180)
  (h2 : gretchen_pennies = rachelle_pennies / 2)
  (h3 : rocky_pennies = gretchen_pennies / 3) :
  rachelle_pennies + gretchen_pennies + rocky_pennies = 300 :=
by
  sorry

end total_pennies_l2382_238200


namespace songs_per_album_l2382_238238

theorem songs_per_album (C P : ℕ) (h1 : 4 * C + 5 * P = 72) (h2 : C = P) : C = 8 :=
by
  sorry

end songs_per_album_l2382_238238


namespace max_ab_value_l2382_238224

theorem max_ab_value (a b : ℝ) (h : ∀ x : ℝ, x^2 - 2 * a * x - b^2 + 12 ≤ 0 → x = a) : ab = 6 := by
  sorry

end max_ab_value_l2382_238224


namespace percentage_by_which_x_more_than_y_l2382_238217

theorem percentage_by_which_x_more_than_y
    (x y z : ℝ)
    (h1 : y = 1.20 * z)
    (h2 : z = 150)
    (h3 : x + y + z = 555) :
    ((x - y) / y) * 100 = 25 :=
by
  sorry

end percentage_by_which_x_more_than_y_l2382_238217


namespace coordinates_of_point_B_l2382_238235

theorem coordinates_of_point_B (A B : ℝ × ℝ) (AB : ℝ) :
  A = (-1, 2) ∧ B.1 = -1 ∧ AB = 3 ∧ (B.2 = 5 ∨ B.2 = -1) :=
by
  sorry

end coordinates_of_point_B_l2382_238235


namespace range_of_x_squared_plus_y_squared_l2382_238295

def increasing (f : ℝ → ℝ) := ∀ x y, x < y → f x < f y
def symmetric_about_origin (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

theorem range_of_x_squared_plus_y_squared 
  (f : ℝ → ℝ) 
  (h_incr : increasing f) 
  (h_symm : symmetric_about_origin f) 
  (h_ineq : ∀ x y, f (x^2 - 6 * x) + f (y^2 - 8 * y + 24) < 0) : 
  ∀ x y, 16 < x^2 + y^2 ∧ x^2 + y^2 < 36 := 
sorry

end range_of_x_squared_plus_y_squared_l2382_238295


namespace find_c_l2382_238258

-- Definitions for the conditions
def is_solution (x c : ℝ) : Prop := x^2 + c * x - 36 = 0

theorem find_c (c : ℝ) (h : is_solution (-9) c) : c = 5 :=
sorry

end find_c_l2382_238258


namespace redistribute_oil_l2382_238228

def total_boxes (trucks1 trucks2 boxes1 boxes2 : Nat) :=
  (trucks1 * boxes1) + (trucks2 * boxes2)

def total_containers (boxes containers_per_box : Nat) :=
  boxes * containers_per_box

def containers_per_truck (total_containers trucks : Nat) :=
  total_containers / trucks

theorem redistribute_oil :
  ∀ (trucks1 trucks2 boxes1 boxes2 containers_per_box total_trucks : Nat),
  trucks1 = 7 →
  trucks2 = 5 →
  boxes1 = 20 →
  boxes2 = 12 →
  containers_per_box = 8 →
  total_trucks = 10 →
  containers_per_truck (total_containers (total_boxes trucks1 trucks2 boxes1 boxes2) containers_per_box) total_trucks = 160 :=
by
  intros trucks1 trucks2 boxes1 boxes2 containers_per_box total_trucks
  intros h_trucks1 h_trucks2 h_boxes1 h_boxes2 h_containers_per_box h_total_trucks
  sorry

end redistribute_oil_l2382_238228


namespace find_p_q_sum_l2382_238253

variable (P Q x : ℝ)

theorem find_p_q_sum (h : (P / (x - 3)) +  Q * (x + 2) = (-5 * x^2 + 18 * x + 40) / (x - 3)) : P + Q = 20 :=
sorry

end find_p_q_sum_l2382_238253


namespace cory_fruits_arrangement_l2382_238283

-- Conditions
def apples : ℕ := 4
def oranges : ℕ := 2
def lemon : ℕ := 1
def total_fruits : ℕ := apples + oranges + lemon

-- Formula to calculate the number of distinct ways
def factorial (n : ℕ) : ℕ := 
  if n = 0 then 1 else n * factorial (n - 1)

def arrangement_count : ℕ :=
  factorial total_fruits / (factorial apples * factorial oranges * factorial lemon)

theorem cory_fruits_arrangement : arrangement_count = 105 := by
  -- Sorry is placed here to skip the actual proof
  sorry

end cory_fruits_arrangement_l2382_238283


namespace number_of_vegetarians_l2382_238282

-- Define the conditions
def only_veg : ℕ := 11
def only_nonveg : ℕ := 6
def both_veg_and_nonveg : ℕ := 9

-- Define the total number of vegetarians
def total_veg : ℕ := only_veg + both_veg_and_nonveg

-- The statement to be proved
theorem number_of_vegetarians : total_veg = 20 := 
by
  sorry

end number_of_vegetarians_l2382_238282


namespace arithmetic_sequence_problem_l2382_238262

variables (a_n b_n : ℕ → ℚ)
variables (S_n T_n : ℕ → ℚ)
variable (n : ℕ)

axiom sum_a_terms : ∀ n : ℕ, S_n n = n / 2 * (a_n 1 + a_n n)
axiom sum_b_terms : ∀ n : ℕ, T_n n = n / 2 * (b_n 1 + b_n n)
axiom given_fraction : ∀ n : ℕ, n > 0 → S_n n / T_n n = (2 * n + 1) / (4 * n - 2)

theorem arithmetic_sequence_problem : 
  (a_n 10) / (b_n 3 + b_n 18) + (a_n 11) / (b_n 6 + b_n 15) = 41 / 78 :=
sorry

end arithmetic_sequence_problem_l2382_238262


namespace negation_of_universal_l2382_238227

theorem negation_of_universal (P : ∀ x : ℤ, x^3 < 1) : ¬ (∀ x : ℤ, x^3 < 1) ↔ ∃ x : ℤ, x^3 ≥ 1 :=
by
  sorry

end negation_of_universal_l2382_238227


namespace rectangle_in_triangle_area_l2382_238274

theorem rectangle_in_triangle_area
  (PR : ℝ) (h_PR : PR = 15)
  (Q_altitude : ℝ) (h_Q_altitude : Q_altitude = 9)
  (x : ℝ)
  (AD : ℝ) (h_AD : AD = x)
  (AB : ℝ) (h_AB : AB = x / 3) :
  (AB * AD = 675 / 64) :=
by
  sorry

end rectangle_in_triangle_area_l2382_238274


namespace conic_sections_hyperbola_and_ellipse_l2382_238204

theorem conic_sections_hyperbola_and_ellipse
  (x y : ℝ) (h : y^4 - 9 * x^4 = 3 * y^2 - 3) :
  (∃ a b c : ℝ, a * y^2 - b * x^2 = c ∧ a = b ∧ c ≠ 0) ∨ (∃ a b c : ℝ, a * y^2 + b * x^2 = c ∧ a ≠ b ∧ c ≠ 0) :=
by
  sorry

end conic_sections_hyperbola_and_ellipse_l2382_238204


namespace flooring_area_already_installed_l2382_238272

variable (living_room_length : ℕ) (living_room_width : ℕ) 
variable (flooring_sqft_per_box : ℕ)
variable (remaining_boxes_needed : ℕ)
variable (already_installed : ℕ)

theorem flooring_area_already_installed 
  (h1 : living_room_length = 16)
  (h2 : living_room_width = 20)
  (h3 : flooring_sqft_per_box = 10)
  (h4 : remaining_boxes_needed = 7)
  (h5 : living_room_length * living_room_width = 320)
  (h6 : already_installed = 320 - remaining_boxes_needed * flooring_sqft_per_box) : 
  already_installed = 250 :=
by
  sorry

end flooring_area_already_installed_l2382_238272


namespace inversely_proportional_find_p_l2382_238220

theorem inversely_proportional_find_p (p q : ℕ) (h1 : p * 8 = 160) (h2 : q = 10) : p * q = 160 → p = 16 :=
by
  intro h
  sorry

end inversely_proportional_find_p_l2382_238220


namespace students_in_same_month_l2382_238267

theorem students_in_same_month (students : ℕ) (months : ℕ) 
  (h : students = 50) (h_months : months = 12) : 
  ∃ k ≥ 5, ∃ i, i < months ∧ ∃ f : ℕ → ℕ, (∀ j < students, f j < months) 
  ∧ ∃ n ≥ 5, ∃ j < students, f j = i :=
by 
  sorry

end students_in_same_month_l2382_238267


namespace reciprocal_proof_l2382_238223

theorem reciprocal_proof :
  (-2) * (-(1 / 2)) = 1 := 
by 
  sorry

end reciprocal_proof_l2382_238223


namespace area_of_inscribed_rectangle_l2382_238216

variable (b h x : ℝ)

def is_isosceles_triangle (b h : ℝ) : Prop :=
  b > 0 ∧ h > 0

def is_inscribed_rectangle (b h x : ℝ) : Prop :=
  x > 0 ∧ x < h 

theorem area_of_inscribed_rectangle (h_pos : is_isosceles_triangle b h) 
                                    (rect_pos : is_inscribed_rectangle b h x) : 
                                    ∃ A : ℝ, A = (b / (2 * h)) * x ^ 2 :=
by
  sorry

end area_of_inscribed_rectangle_l2382_238216


namespace value_of_f_2017_l2382_238278

def f (x : ℕ) : ℕ := x^2 - x * (0 : ℕ) - 1

theorem value_of_f_2017 : f 2017 = 2016 * 2018 := by
  sorry

end value_of_f_2017_l2382_238278


namespace right_triangle_similarity_l2382_238222

theorem right_triangle_similarity (y : ℝ) (h : 12 / y = 9 / 7) : y = 9.33 := 
by 
  sorry

end right_triangle_similarity_l2382_238222


namespace max_marked_cells_100x100_board_l2382_238230

theorem max_marked_cells_100x100_board : 
  ∃ n, (3 * n + 1 = 100) ∧ (2 * n + 1) * (n + 1) = 2278 :=
by
  sorry

end max_marked_cells_100x100_board_l2382_238230


namespace arithmetic_sequence_product_l2382_238270

theorem arithmetic_sequence_product (b : ℕ → ℤ) (h1 : ∀ n, b (n + 1) = b n + d) 
  (h2 : b 5 * b 6 = 35) : b 4 * b 7 = 27 :=
sorry

end arithmetic_sequence_product_l2382_238270


namespace find_the_number_l2382_238219

theorem find_the_number (x k : ℕ) (h1 : x / k = 4) (h2 : k = 8) : x = 32 := by
  sorry

end find_the_number_l2382_238219


namespace marciaHairLengthProof_l2382_238210

noncomputable def marciaHairLengthAtEndOfSchoolYear : Float :=
  let L0 := 24.0                           -- initial length
  let L1 := L0 - 0.3 * L0                  -- length after September cut
  let L2 := L1 + 3.0 * 1.5                 -- length after three months of growth (Sept - Dec)
  let L3 := L2 - 0.2 * L2                  -- length after January cut
  let L4 := L3 + 5.0 * 1.8                 -- length after five months of growth (Jan - May)
  let L5 := L4 - 4.0                       -- length after June cut
  L5

theorem marciaHairLengthProof : marciaHairLengthAtEndOfSchoolYear = 22.04 :=
by
  sorry

end marciaHairLengthProof_l2382_238210


namespace parabola_focus_coordinates_l2382_238260

open Real

theorem parabola_focus_coordinates (x y : ℝ) (h : y^2 = 6 * x) : (x, y) = (3 / 2, 0) :=
  sorry

end parabola_focus_coordinates_l2382_238260
