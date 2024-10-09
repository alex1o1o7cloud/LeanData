import Mathlib

namespace meaningful_range_fraction_l1139_113923

theorem meaningful_range_fraction (x : ℝ) : 
  ¬ (x = 3) ↔ (∃ y, y = x / (x - 3)) :=
sorry

end meaningful_range_fraction_l1139_113923


namespace positive_integers_square_of_sum_of_digits_l1139_113973

-- Define the sum of the digits function
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Define the main theorem
theorem positive_integers_square_of_sum_of_digits :
  ∀ (n : ℕ), (n > 0) → (n = sum_of_digits n ^ 2) → (n = 1 ∨ n = 81) :=
by
  sorry

end positive_integers_square_of_sum_of_digits_l1139_113973


namespace train_length_l1139_113996

theorem train_length (speed_kmh : ℕ) (time_s : ℕ) (length_m : ℕ) 
  (h1 : speed_kmh = 180)
  (h2 : time_s = 18)
  (h3 : 1 = 1000 / 3600) :
  length_m = (speed_kmh * 1000 / 3600) * time_s :=
by
  sorry

end train_length_l1139_113996


namespace dice_impossible_divisible_by_10_l1139_113950

theorem dice_impossible_divisible_by_10 :
  ¬ ∃ n ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ), n % 10 = 0 :=
by
  sorry

end dice_impossible_divisible_by_10_l1139_113950


namespace reflect_parabola_y_axis_l1139_113945

theorem reflect_parabola_y_axis (x y : ℝ) :
  (y = 2 * (x - 1)^2 - 4) → (y = 2 * (-x - 1)^2 - 4) :=
sorry

end reflect_parabola_y_axis_l1139_113945


namespace production_cost_per_performance_l1139_113929

def overhead_cost := 81000
def income_per_performance := 16000
def performances_needed := 9

theorem production_cost_per_performance :
  ∃ P, 9 * income_per_performance = overhead_cost + 9 * P ∧ P = 7000 :=
by
  sorry

end production_cost_per_performance_l1139_113929


namespace tan_sin_cos_l1139_113932

theorem tan_sin_cos (θ : ℝ) (h : Real.tan θ = 1 / 2) : 
  Real.sin (2 * θ) - 2 * Real.cos θ ^ 2 = - 4 / 5 := by 
  sorry

end tan_sin_cos_l1139_113932


namespace percent_of_percent_l1139_113955

theorem percent_of_percent (y : ℝ) : 0.3 * (0.6 * y) = 0.18 * y :=
sorry

end percent_of_percent_l1139_113955


namespace vendor_throws_away_8_percent_l1139_113938

theorem vendor_throws_away_8_percent (total_apples: ℕ) (h₁ : total_apples > 0) :
    let apples_after_first_day := total_apples * 40 / 100
    let thrown_away_first_day := apples_after_first_day * 10 / 100
    let apples_after_second_day := (apples_after_first_day - thrown_away_first_day) * 30 / 100
    let thrown_away_second_day := apples_after_second_day * 20 / 100
    let apples_after_third_day := (apples_after_second_day - thrown_away_second_day) * 60 / 100
    let thrown_away_third_day := apples_after_third_day * 30 / 100
    total_apples > 0 → (8 : ℕ) * total_apples = (thrown_away_first_day + thrown_away_second_day + thrown_away_third_day) * 100 := 
by
    -- Placeholder proof
    sorry

end vendor_throws_away_8_percent_l1139_113938


namespace correct_calculation_l1139_113905

-- Definitions for each condition
def conditionA (a b : ℝ) : Prop := (a - b) * (-a - b) = a^2 - b^2
def conditionB (a : ℝ) : Prop := 2 * a^3 + 3 * a^3 = 5 * a^6
def conditionC (x y : ℝ) : Prop := 6 * x^3 * y^2 / (3 * x) = 2 * x^2 * y^2
def conditionD (x : ℝ) : Prop := (-2 * x^2)^3 = -6 * x^6

-- The proof problem
theorem correct_calculation (a b x y : ℝ) :
  ¬ conditionA a b ∧ ¬ conditionB a ∧ conditionC x y ∧ ¬ conditionD x := 
sorry

end correct_calculation_l1139_113905


namespace triplet_solution_l1139_113910

theorem triplet_solution (a b c : ℝ)
  (h1 : a^2 + b = c^2)
  (h2 : b^2 + c = a^2)
  (h3 : c^2 + a = b^2) :
  (a = 0 ∧ b = 0 ∧ c = 0) ∨
  (a = 0 ∧ b = 1 ∧ c = -1) ∨
  (a = -1 ∧ b = 0 ∧ c = 1) ∨
  (a = 1 ∧ b = -1 ∧ c = 0) :=
sorry

end triplet_solution_l1139_113910


namespace minimum_value_l1139_113940

-- Given conditions
variables (a b c d : ℝ)
variables (h_a : a > 0) (h_b : b = 0) (h_a_eq : a = 1)

-- Define the function
def f (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

-- The statement to prove
theorem minimum_value (h_c : c = 0) : ∃ x : ℝ, f a b c d x = d :=
by
  -- Given the conditions a=1, b=0, and c=0, we need to show that the minimum value is d
  sorry

end minimum_value_l1139_113940


namespace odd_number_divisibility_l1139_113993

theorem odd_number_divisibility (a : ℤ) (h : a % 2 = 1) : ∃ (k : ℤ), a^4 + 9 * (9 - 2 * a^2) = 16 * k :=
by
  sorry

end odd_number_divisibility_l1139_113993


namespace faster_cow_days_to_eat_one_bag_l1139_113931

-- Conditions as assumptions
def num_cows : ℕ := 60
def num_husks : ℕ := 150
def num_days : ℕ := 80
def faster_cows : ℕ := 20
def normal_cows : ℕ := num_cows - faster_cows
def faster_rate : ℝ := 1.3

-- The question translated to Lean 4 statement
theorem faster_cow_days_to_eat_one_bag :
  (faster_cows * faster_rate + normal_cows) / num_cows * (num_husks / num_days) = 1 / 27.08 :=
sorry

end faster_cow_days_to_eat_one_bag_l1139_113931


namespace least_m_plus_n_l1139_113904

theorem least_m_plus_n (m n : ℕ) (h1 : Nat.gcd (m + n) 231 = 1) 
                                  (h2 : m^m ∣ n^n) 
                                  (h3 : ¬ m ∣ n)
                                  : m + n = 75 :=
sorry

end least_m_plus_n_l1139_113904


namespace exists_integers_m_n_l1139_113941

theorem exists_integers_m_n (a b c p q r : ℝ) (h_a : a ≠ 0) (h_p : p ≠ 0) :
  ∃ (m n : ℤ), ∀ (x : ℝ), (a * x^2 + b * x + c = m * (p * x^2 + q * x + r) + n) := sorry

end exists_integers_m_n_l1139_113941


namespace c_share_l1139_113911

theorem c_share (a b c d e : ℝ) (k : ℝ)
  (h1 : a + b + c + d + e = 1010)
  (h2 : a - 25 = 4 * k)
  (h3 : b - 10 = 3 * k)
  (h4 : c - 15 = 6 * k)
  (h5 : d - 20 = 2 * k)
  (h6 : e - 30 = 5 * k) :
  c = 288 :=
by
  -- proof with necessary steps
  sorry

end c_share_l1139_113911


namespace correct_operation_is_multiplication_by_3_l1139_113998

theorem correct_operation_is_multiplication_by_3
  (x : ℝ)
  (percentage_error : ℝ)
  (correct_result : ℝ := 3 * x)
  (incorrect_result : ℝ := x / 5)
  (error_percentage : ℝ := (correct_result - incorrect_result) / correct_result * 100) :
  percentage_error = 93.33333333333333 → correct_result / x = 3 :=
by
  intro h
  sorry

end correct_operation_is_multiplication_by_3_l1139_113998


namespace volleyball_team_math_count_l1139_113943

theorem volleyball_team_math_count (total_players taking_physics taking_both : ℕ) 
  (h1 : total_players = 30) 
  (h2 : taking_physics = 15) 
  (h3 : taking_both = 6) 
  (h4 : total_players = 30 ∧ total_players = (taking_physics + (total_players - taking_physics - taking_both))) 
  : (total_players - (taking_physics - taking_both) + taking_both) = 21 := 
by
  sorry

end volleyball_team_math_count_l1139_113943


namespace leftover_grass_seed_coverage_l1139_113927

/-
Question: How many extra square feet could the leftover grass seed cover after Drew reseeds his lawn?

Conditions:
- One bag of grass seed covers 420 square feet of lawn.
- The lawn consists of a rectangular section and a triangular section.
- Rectangular section:
    - Length: 32 feet
    - Width: 45 feet
- Triangular section:
    - Base: 25 feet
    - Height: 20 feet
- Triangular section requires 1.5 times the standard coverage rate.
- Drew bought seven bags of seed.

Answer: The leftover grass seed coverage is 1125 square feet.
-/

theorem leftover_grass_seed_coverage
  (bag_coverage : ℕ := 420)
  (rect_length : ℕ := 32)
  (rect_width : ℕ := 45)
  (tri_base : ℕ := 25)
  (tri_height : ℕ := 20)
  (coverage_multiplier : ℕ := 15)  -- Using 15 instead of 1.5 for integer math
  (bags_bought : ℕ := 7) :
  (bags_bought * bag_coverage - 
    (rect_length * rect_width + tri_base * tri_height * coverage_multiplier / 20) = 1125) :=
  by {
    -- Placeholder for proof steps
    sorry
  }

end leftover_grass_seed_coverage_l1139_113927


namespace tan_subtraction_l1139_113937

variable {α β : ℝ}

theorem tan_subtraction (h1 : Real.tan α = 2) (h2 : Real.tan β = -3) : Real.tan (α - β) = -1 := by
  sorry

end tan_subtraction_l1139_113937


namespace probability_correct_l1139_113986

def elenaNameLength : Nat := 5
def markNameLength : Nat := 4
def juliaNameLength : Nat := 5
def totalCards : Nat := elenaNameLength + markNameLength + juliaNameLength

-- Without replacement, drawing three cards from 14 cards randomly
def probabilityThreeDifferentSources : ℚ := 
  (elenaNameLength / totalCards) * (markNameLength / (totalCards - 1)) * (juliaNameLength / (totalCards - 2))

def totalPermutations : Nat := 6  -- EMJ, EJM, MEJ, MJE, JEM, JME

def requiredProbability : ℚ := totalPermutations * probabilityThreeDifferentSources

theorem probability_correct :
  requiredProbability = 25 / 91 := by
  sorry

end probability_correct_l1139_113986


namespace at_least_two_of_three_equations_have_solutions_l1139_113944

theorem at_least_two_of_three_equations_have_solutions
  (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a) :
  ∃ x : ℝ, (x - a) * (x - b) = x - c ∨ (x - b) * (x - c) = x - a ∨ (x - c) * (x - a) = x - b := 
sorry

end at_least_two_of_three_equations_have_solutions_l1139_113944


namespace quadratic_inequality_solution_l1139_113999

theorem quadratic_inequality_solution (k : ℝ) :
  (-1 < k ∧ k < 7) ↔ ∀ x : ℝ, x^2 - (k - 5) * x - k + 8 > 0 :=
by
  sorry

end quadratic_inequality_solution_l1139_113999


namespace initial_blue_balls_l1139_113906

theorem initial_blue_balls (B : ℕ) (h1 : 25 - 5 = 20) (h2 : (B - 5) / 20 = 1 / 5) : B = 9 :=
by
  sorry

end initial_blue_balls_l1139_113906


namespace middle_tree_distance_l1139_113903

theorem middle_tree_distance (d : ℕ) (b : ℕ) (c : ℕ) 
  (h_b : b = 84) (h_c : c = 91) 
  (h_right_triangle : d^2 + b^2 = c^2) : 
  d = 35 :=
by
  sorry

end middle_tree_distance_l1139_113903


namespace total_rainfall_in_2004_l1139_113992

noncomputable def average_monthly_rainfall_2003 : ℝ := 35.0
noncomputable def average_monthly_rainfall_2004 : ℝ := average_monthly_rainfall_2003 + 4.0
noncomputable def total_rainfall_2004 : ℝ := 
  let regular_months := 11 * average_monthly_rainfall_2004
  let daily_rainfall_feb := average_monthly_rainfall_2004 / 30
  let feb_rain := daily_rainfall_feb * 29 
  regular_months + feb_rain

theorem total_rainfall_in_2004 : total_rainfall_2004 = 466.7 := by
  sorry

end total_rainfall_in_2004_l1139_113992


namespace problem_A_problem_B_problem_C_problem_D_problem_E_l1139_113926

-- Definitions and assumptions based on the problem statement
def eqI (x y z : ℕ) := x + y + z = 45
def eqII (x y z w : ℕ) := x + y + z + w = 50
def consecutive_odd_integers (x y z : ℕ) := y = x + 2 ∧ z = x + 4
def multiples_of_five (x y z w : ℕ) := (∃ a b c d : ℕ, x = 5 * a ∧ y = 5 * b ∧ z = 5 * c ∧ w = 5 * d)
def consecutive_integers (x y z w : ℕ) := y = x + 1 ∧ z = x + 2 ∧ w = x + 3
def prime_integers (x y z : ℕ) := Prime x ∧ Prime y ∧ Prime z

-- Lean theorem statements
theorem problem_A : ∃ x y z : ℕ, eqI x y z ∧ consecutive_odd_integers x y z := 
sorry

theorem problem_B : ¬ (∃ x y z : ℕ, eqI x y z ∧ prime_integers x y z) := 
sorry

theorem problem_C : ¬ (∃ x y z w : ℕ, eqII x y z w ∧ consecutive_odd_integers x y z) :=
sorry

theorem problem_D : ∃ x y z w : ℕ, eqII x y z w ∧ multiples_of_five x y z w := 
sorry

theorem problem_E : ∃ x y z w : ℕ, eqII x y z w ∧ consecutive_integers x y z w := 
sorry

end problem_A_problem_B_problem_C_problem_D_problem_E_l1139_113926


namespace max_value_of_expression_max_value_achieved_l1139_113960

theorem max_value_of_expression (x y z : ℝ) (h : 9 * x^2 + 4 * y^2 + 25 * z^2 = 1) :
    8 * x + 3 * y + 10 * z ≤ Real.sqrt 173 :=
sorry

theorem max_value_achieved (x y z : ℝ) (h : 9 * x^2 + 4 * y^2 + 25 * z^2 = 1)
    (hx : x = Real.sqrt 173 / 30)
    (hy : y = Real.sqrt 173 / 20)
    (hz : z = Real.sqrt 173 / 50) :
    8 * x + 3 * y + 10 * z = Real.sqrt 173 :=
sorry

end max_value_of_expression_max_value_achieved_l1139_113960


namespace find_k_l1139_113962

theorem find_k (a b k : ℝ) (h1 : 2^a = k) (h2 : 3^b = k) (hk : k ≠ 1) (h3 : 2 * a + b = a * b) : 
  k = 18 :=
sorry

end find_k_l1139_113962


namespace minimum_discount_percentage_l1139_113935

theorem minimum_discount_percentage (cost_price marked_price : ℝ) (profit_margin : ℝ) (discount : ℝ) :
  cost_price = 400 ∧ marked_price = 600 ∧ profit_margin = 0.05 ∧ 
  (marked_price * (1 - discount / 100) - cost_price) / cost_price ≥ profit_margin → discount ≤ 30 := 
by
  intros h
  rcases h with ⟨hc, hm, hp, hineq⟩
  sorry

end minimum_discount_percentage_l1139_113935


namespace josh_money_left_l1139_113915

def initial_amount : ℝ := 20
def cost_hat : ℝ := 10
def cost_pencil : ℝ := 2
def number_of_cookies : ℝ := 4
def cost_per_cookie : ℝ := 1.25

theorem josh_money_left : initial_amount - cost_hat - cost_pencil - (number_of_cookies * cost_per_cookie) = 3 := by
  sorry

end josh_money_left_l1139_113915


namespace green_duck_percentage_l1139_113922

noncomputable def smaller_pond_ducks : ℕ := 45
noncomputable def larger_pond_ducks : ℕ := 55
noncomputable def green_percentage_small_pond : ℝ := 0.20
noncomputable def green_percentage_large_pond : ℝ := 0.40

theorem green_duck_percentage :
  let total_ducks := smaller_pond_ducks + larger_pond_ducks
  let green_ducks_smaller := green_percentage_small_pond * (smaller_pond_ducks : ℝ)
  let green_ducks_larger := green_percentage_large_pond * (larger_pond_ducks : ℝ)
  let total_green_ducks := green_ducks_smaller + green_ducks_larger
  (total_green_ducks / total_ducks) * 100 = 31 :=
by {
  -- The proof is omitted.
  sorry
}

end green_duck_percentage_l1139_113922


namespace maximum_cards_without_equal_pair_sums_l1139_113985

def max_cards_no_equal_sum_pairs : ℕ :=
  let card_points := {x : ℕ | 1 ≤ x ∧ x ≤ 13}
  6

theorem maximum_cards_without_equal_pair_sums (deck : Finset ℕ) (h_deck : deck = {x : ℕ | 1 ≤ x ∧ x ≤ 13}) :
  ∃ S ⊆ deck, S.card = 6 ∧ ∀ {a b c d : ℕ}, a ∈ S → b ∈ S → c ∈ S → d ∈ S → a + b = c + d → a = c ∧ b = d ∨ a = d ∧ b = c := 
sorry

end maximum_cards_without_equal_pair_sums_l1139_113985


namespace number_of_boys_l1139_113918

theorem number_of_boys (x : ℕ) (y : ℕ) (h1 : x + y = 8) (h2 : y > x) : x = 1 ∨ x = 2 ∨ x = 3 :=
by
  sorry

end number_of_boys_l1139_113918


namespace large_doll_cost_is_8_l1139_113968

-- Define the cost of the large monkey doll
def cost_large_doll : ℝ := 8

-- Define the total amount spent
def total_spent : ℝ := 320

-- Define the price difference between large and small dolls
def price_difference : ℝ := 4

-- Define the count difference between buying small dolls and large dolls
def count_difference : ℝ := 40

theorem large_doll_cost_is_8 
    (h1 : total_spent = 320)
    (h2 : ∀ L, L - price_difference = 4)
    (h3 : ∀ L, (total_spent / (L - 4)) = (total_spent / L) + count_difference) :
    cost_large_doll = 8 := 
by 
  sorry

end large_doll_cost_is_8_l1139_113968


namespace quadratic_inequality_solution_l1139_113957

noncomputable def solve_inequality (a b : ℝ) : Prop :=
  (∀ x : ℝ, (x > -1/2 ∧ x < 1/3) → (a * x^2 + b * x + 2 > 0)) →
  (a = -12) ∧ (b = -2)

theorem quadratic_inequality_solution :
   solve_inequality (-12) (-2) :=
by
  intro h
  sorry

end quadratic_inequality_solution_l1139_113957


namespace no_cube_sum_of_three_consecutive_squares_l1139_113983

theorem no_cube_sum_of_three_consecutive_squares :
  ¬∃ x y : ℤ, x^3 = (y-1)^2 + y^2 + (y+1)^2 :=
by
  sorry

end no_cube_sum_of_three_consecutive_squares_l1139_113983


namespace hypotenuse_longer_side_difference_l1139_113953

theorem hypotenuse_longer_side_difference
  (x : ℝ)
  (h1 : 17^2 = x^2 + (x - 7)^2)
  (h2 : x = 15)
  : 17 - x = 2 := by
  sorry

end hypotenuse_longer_side_difference_l1139_113953


namespace integer_solutions_l1139_113939

-- Define the polynomial equation as a predicate
def polynomial (n : ℤ) : Prop := n^5 - 2 * n^4 - 7 * n^2 - 7 * n + 3 = 0

-- The theorem statement
theorem integer_solutions :
  {n : ℤ | polynomial n} = {-1, 3} :=
by 
  sorry

end integer_solutions_l1139_113939


namespace bag_contains_n_black_balls_l1139_113908

theorem bag_contains_n_black_balls (n : ℕ) : (5 / (n + 5) = 1 / 3) → n = 10 := by
  sorry

end bag_contains_n_black_balls_l1139_113908


namespace Kishore_education_expense_l1139_113971

theorem Kishore_education_expense
  (rent milk groceries petrol misc saved : ℝ) -- expenses
  (total_saved_salary : ℝ) -- percentage of saved salary
  (saving_amount : ℝ) -- actual saving
  (total_salary total_expense_children_education : ℝ) -- total salary and expense on children's education
  (H1 : rent = 5000)
  (H2 : milk = 1500)
  (H3 : groceries = 4500)
  (H4 : petrol = 2000)
  (H5 : misc = 3940)
  (H6 : total_saved_salary = 0.10)
  (H7 : saving_amount = 2160)
  (H8 : total_salary = saving_amount / total_saved_salary)
  (H9 : total_expense_children_education = total_salary - (rent + milk + groceries + petrol + misc) - saving_amount) :
  total_expense_children_education = 2600 :=
by 
  simp only [H1, H2, H3, H4, H5, H6, H7] at *
  norm_num at *
  sorry

end Kishore_education_expense_l1139_113971


namespace painting_cost_in_cny_l1139_113975

theorem painting_cost_in_cny (usd_to_nad : ℝ) (usd_to_cny : ℝ) (painting_cost_nad : ℝ) :
  usd_to_nad = 8 → usd_to_cny = 7 → painting_cost_nad = 160 →
  painting_cost_nad / usd_to_nad * usd_to_cny = 140 :=
by
  intros
  sorry

end painting_cost_in_cny_l1139_113975


namespace index_card_area_l1139_113913

theorem index_card_area (a b : ℕ) (new_area : ℕ) (reduce_length reduce_width : ℕ)
  (original_length : a = 3) (original_width : b = 7)
  (reduced_area_condition : a * (b - reduce_width) = new_area)
  (reduce_width_2 : reduce_width = 2) 
  (new_area_correct : new_area = 15) :
  (a - reduce_length) * b = 7 := by
  sorry

end index_card_area_l1139_113913


namespace sequence_sum_S6_l1139_113954

theorem sequence_sum_S6 (a_n : ℕ → ℕ) (S_n : ℕ → ℕ) (h : ∀ n, S_n n = 2 * a_n n - 3) :
  S_n 6 = 189 :=
by
  sorry

end sequence_sum_S6_l1139_113954


namespace four_digit_sum_10_divisible_by_9_is_0_l1139_113972

theorem four_digit_sum_10_divisible_by_9_is_0 : 
  ∀ (N : ℕ), (1000 * ((N / 1000) % 10) + 100 * ((N / 100) % 10) + 10 * ((N / 10) % 10) + (N % 10) = 10) ∧ (N % 9 = 0) → false :=
by
  sorry

end four_digit_sum_10_divisible_by_9_is_0_l1139_113972


namespace product_of_real_roots_eq_one_l1139_113934

theorem product_of_real_roots_eq_one:
  ∀ x : ℝ, (x ^ (Real.log x / Real.log 5) = 25) → (∀ x1 x2 : ℝ, (x1 ^ (Real.log x1 / Real.log 5) = 25) → (x2 ^ (Real.log x2 / Real.log 5) = 25) → x1 * x2 = 1) :=
by
  sorry

end product_of_real_roots_eq_one_l1139_113934


namespace determine_values_l1139_113902

theorem determine_values (x y : ℝ) (h1 : x - y = 25) (h2 : x * y = 36) : (x^2 + y^2 = 697) ∧ (x + y = Real.sqrt 769) :=
by
  -- Proof goes here
  sorry

end determine_values_l1139_113902


namespace negation_of_existence_l1139_113930

theorem negation_of_existence :
  (¬ ∃ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ (∀ x : ℝ, x^3 - x^2 + 1 > 0) :=
by
  sorry

end negation_of_existence_l1139_113930


namespace red_balls_removal_l1139_113942

theorem red_balls_removal (total_balls : ℕ) (red_balls : ℕ) (blue_balls : ℕ) (x : ℕ) :
  total_balls = 600 →
  red_balls = 420 →
  blue_balls = 180 →
  (red_balls - x) / (total_balls - x : ℚ) = 3 / 5 ↔ x = 150 :=
by 
  intros;
  sorry

end red_balls_removal_l1139_113942


namespace visitors_on_monday_l1139_113969

theorem visitors_on_monday (M : ℕ) (h : M + 2 * M + 100 = 250) : M = 50 :=
by
  sorry

end visitors_on_monday_l1139_113969


namespace trigonometric_identity_l1139_113989

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 4) :
  (2 * Real.sin α + Real.cos α) / (Real.sin α - 3 * Real.cos α) = 9 := 
sorry

end trigonometric_identity_l1139_113989


namespace find_OP_l1139_113979

variable (a b c d e f : ℝ)
variable (P : ℝ)

-- Given conditions
axiom AP_PD_ratio : (a - P) / (P - d) = 2 / 3
axiom BP_PC_ratio : (b - P) / (P - c) = 3 / 4

-- Conclusion to prove
theorem find_OP : P = (3 * a + 2 * d) / 5 :=
by
  sorry

end find_OP_l1139_113979


namespace airport_exchange_rate_frac_l1139_113978

variable (euros_received : ℕ) (euros : ℕ) (official_exchange_rate : ℕ) (dollars_received : ℕ)

theorem airport_exchange_rate_frac (h1 : euros = 70) (h2 : official_exchange_rate = 5) (h3 : dollars_received = 10) :
  (euros_received * dollars_received) = (euros * official_exchange_rate) →
  euros_received = 5 / 7 :=
  sorry

end airport_exchange_rate_frac_l1139_113978


namespace yellow_sweets_l1139_113925

-- Definitions
def green_sweets : Nat := 212
def blue_sweets : Nat := 310
def sweets_per_person : Nat := 256
def people : Nat := 4

-- Proof problem statement
theorem yellow_sweets : green_sweets + blue_sweets + x = sweets_per_person * people → x = 502 := by
  sorry

end yellow_sweets_l1139_113925


namespace sue_necklace_total_beads_l1139_113958

theorem sue_necklace_total_beads :
  ∀ (purple blue green : ℕ),
  purple = 7 →
  blue = 2 * purple →
  green = blue + 11 →
  (purple + blue + green = 46) :=
by
  intros purple blue green h1 h2 h3
  rw [h1, h2, h3]
  sorry

end sue_necklace_total_beads_l1139_113958


namespace no_intersection_points_l1139_113974

theorem no_intersection_points :
  ∀ x y : ℝ, y = abs (3 * x + 6) ∧ y = -2 * abs (2 * x - 1) → false :=
by
  intros x y h
  cases h
  sorry

end no_intersection_points_l1139_113974


namespace range_of_a_l1139_113988

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (x^2 - x - 2 ≥ 0) ↔ (x ≤ -1 ∨ x ≥ 2)) ∧
  (∀ x : ℝ, (2 * a - 1 ≤ x ∧ x ≤ a + 3)) →
  (-1 ≤ a ∧ a ≤ 0) :=
by
  -- Prove the theorem
  sorry

end range_of_a_l1139_113988


namespace common_difference_zero_l1139_113994

theorem common_difference_zero (a b c : ℕ) 
  (h_seq : ∃ d : ℕ, a = b + d ∧ b = c + d)
  (h_eq : (c - b) / a + (a - c) / b + (b - a) / c = 0) : 
  ∀ d : ℕ, d = 0 :=
by sorry

end common_difference_zero_l1139_113994


namespace measure_of_angle_F_l1139_113900

theorem measure_of_angle_F (angle_D angle_E angle_F : ℝ) (h1 : angle_D = 80)
  (h2 : angle_E = 4 * angle_F + 10)
  (h3 : angle_D + angle_E + angle_F = 180) : angle_F = 18 := 
by
  sorry

end measure_of_angle_F_l1139_113900


namespace anthony_solve_l1139_113920

def completing_square (a b c : ℤ) : ℤ :=
  let d := Int.sqrt a
  let e := b / (2 * d)
  let f := (d * e * e - c)
  d + e + f

theorem anthony_solve (d e f : ℤ) (h_d_pos : d > 0)
  (h_eqn : 25 * d * d + 30 * d * e - 72 = 0)
  (h_form : (d * x + e)^2 = f) : 
  d + e + f = 89 :=
by
  have d : ℤ := 5
  have e : ℤ := 3
  have f : ℤ := 81
  sorry

end anthony_solve_l1139_113920


namespace percentage_l_75_m_l1139_113901

theorem percentage_l_75_m
  (j k l m : ℝ)
  (x : ℝ)
  (h1 : 1.25 * j = 0.25 * k)
  (h2 : 1.5 * k = 0.5 * l)
  (h3 : (x / 100) * l = 0.75 * m)
  (h4 : 0.2 * m = 7 * j) :
  x = 175 :=
by
  sorry

end percentage_l_75_m_l1139_113901


namespace convert_to_scientific_notation_l1139_113917

theorem convert_to_scientific_notation :
  (26.62 * 10^9) = 2.662 * 10^9 :=
by
  sorry

end convert_to_scientific_notation_l1139_113917


namespace repeating_fraction_equality_l1139_113965

theorem repeating_fraction_equality : (0.5656565656 : ℚ) = 56 / 99 :=
by
  sorry

end repeating_fraction_equality_l1139_113965


namespace sequence_term_1000_l1139_113980

theorem sequence_term_1000 (a : ℕ → ℤ) 
  (h1 : a 1 = 2010) 
  (h2 : a 2 = 2011) 
  (h3 : ∀ n, 1 ≤ n → a n + a (n + 1) + a (n + 2) = 2 * n) : 
  a 1000 = 2676 :=
sorry

end sequence_term_1000_l1139_113980


namespace percentage_orange_juice_in_blend_l1139_113963

theorem percentage_orange_juice_in_blend :
  let pear_juice_per_pear := 10 / 2
  let orange_juice_per_orange := 8 / 2
  let pear_juice := 2 * pear_juice_per_pear
  let orange_juice := 3 * orange_juice_per_orange
  let total_juice := pear_juice + orange_juice
  (orange_juice / total_juice) = (6 / 11) := 
by
  sorry

end percentage_orange_juice_in_blend_l1139_113963


namespace find_a_l1139_113977

theorem find_a (a : ℝ) :
  (∃ b : ℝ, 4 * b + 3 = 7 ∧ 5 * (-b) - 1 = 2 * (-b) + a) → a = -4 :=
by
  sorry

end find_a_l1139_113977


namespace find_eccentricity_l1139_113909

noncomputable def ellipse_eccentricity (m : ℝ) (c : ℝ) (a : ℝ) : ℝ :=
  c / a

theorem find_eccentricity
  (m : ℝ) (c := Real.sqrt 2) (a := 3 * Real.sqrt 2 / 2)
  (h1 : 2 * m^2 - (m + 1) = 2)
  (h2 : m > 0) :
  ellipse_eccentricity m c a = 2 / 3 :=
by sorry

end find_eccentricity_l1139_113909


namespace max_log_value_l1139_113961

noncomputable def max_log_product (a b : ℝ) : ℝ :=
  if h : a > 0 ∧ b > 0 ∧ a * b = 8 then (Real.logb 2 a) * (Real.logb 2 (2 * b)) else 0

theorem max_log_value (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a * b = 8) :
  max_log_product a b ≤ 4 :=
sorry

end max_log_value_l1139_113961


namespace trapezoid_diagonals_l1139_113916

theorem trapezoid_diagonals (a c b d e f : ℝ) (h1 : a ≠ c):
  e^2 = a * c + (a * d^2 - c * b^2) / (a - c) ∧ 
  f^2 = a * c + (a * b^2 - c * d^2) / (a - c) := 
by
  sorry

end trapezoid_diagonals_l1139_113916


namespace reflection_of_C_over_y_eq_x_l1139_113967

def point_reflection_over_yx := ∀ (A B C : (ℝ × ℝ)), 
  A = (6, 2) → 
  B = (2, 5) → 
  C = (2, 2) → 
  (reflect_y_eq_x C) = (2, 2)
where reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

theorem reflection_of_C_over_y_eq_x :
  point_reflection_over_yx :=
by 
  sorry

end reflection_of_C_over_y_eq_x_l1139_113967


namespace ab_sum_l1139_113997

theorem ab_sum (A B C D : Nat) (h_digits: A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) 
  (h_mult : A * (10 * C + D) = 1001 + 100 * A + 10 * B + A) : A + B = 1 := 
  sorry

end ab_sum_l1139_113997


namespace veronica_initial_marbles_l1139_113984

variable {D M P V : ℕ}

theorem veronica_initial_marbles (hD : D = 14) (hM : M = 20) (hP : P = 19)
  (h_total : D + M + P + V = 60) : V = 7 :=
by
  sorry

end veronica_initial_marbles_l1139_113984


namespace not_always_possible_repaint_all_white_l1139_113976

-- Define the conditions and the problem
def equilateral_triangle_division (n: ℕ) : Prop := 
  ∀ m, m > 1 → m = n^2

def line_parallel_repaint (triangles : List ℕ) : Prop :=
  -- Definition of how the repaint operation affects the triangle colors
  sorry

theorem not_always_possible_repaint_all_white (n : ℕ) (h: equilateral_triangle_division n) :
  ¬∀ triangles, line_parallel_repaint triangles → (∀ t ∈ triangles, t = 0) := 
sorry

end not_always_possible_repaint_all_white_l1139_113976


namespace smallest_x_satisfying_equation_l1139_113933

theorem smallest_x_satisfying_equation :
  ∀ x : ℝ, (2 * x ^ 2 + 24 * x - 60 = x * (x + 13)) → x = -15 ∨ x = 4 ∧ ∃ y : ℝ, y = -15 ∨ y = 4 ∧ y ≤ x :=
by
  sorry

end smallest_x_satisfying_equation_l1139_113933


namespace platform_length_l1139_113928

theorem platform_length
    (train_length : ℕ)
    (time_to_cross_tree : ℕ)
    (speed : ℕ)
    (time_to_pass_platform : ℕ)
    (platform_length : ℕ) :
    train_length = 1200 →
    time_to_cross_tree = 120 →
    speed = train_length / time_to_cross_tree →
    time_to_pass_platform = 150 →
    speed * time_to_pass_platform = train_length + platform_length →
    platform_length = 300 :=
by
  intros h_train_length h_time_to_cross_tree h_speed h_time_to_pass_platform h_pass_platform_eq
  sorry

end platform_length_l1139_113928


namespace ratio_of_side_lengths_l1139_113924

theorem ratio_of_side_lengths (w1 w2 : ℝ) (s1 s2 : ℝ)
  (h1 : w1 = 8) (h2 : w2 = 64)
  (v1 : w1 = s1 ^ 3)
  (v2 : w2 = s2 ^ 3) : 
  s2 / s1 = 2 := by
  sorry

end ratio_of_side_lengths_l1139_113924


namespace positive_difference_of_two_numbers_l1139_113947

variable {x y : ℝ}

theorem positive_difference_of_two_numbers (h₁ : x + y = 8) (h₂ : x^2 - y^2 = 24) : |x - y| = 3 :=
by
  sorry

end positive_difference_of_two_numbers_l1139_113947


namespace square_units_digit_l1139_113981

theorem square_units_digit (n : ℕ) (h : (n^2 / 10) % 10 = 7) : n^2 % 10 = 6 := 
sorry

end square_units_digit_l1139_113981


namespace min_people_liking_both_l1139_113990

theorem min_people_liking_both (total : ℕ) (Beethoven : ℕ) (Chopin : ℕ) 
    (total_eq : total = 150) (Beethoven_eq : Beethoven = 120) (Chopin_eq : Chopin = 95) : 
    ∃ (both : ℕ), both = 65 := 
by 
  have H := Beethoven + Chopin - total
  sorry

end min_people_liking_both_l1139_113990


namespace find_number_l1139_113951

theorem find_number (x : ℝ) : 
  0.05 * x = 0.20 * 650 + 190 → x = 6400 :=
by
  intro h
  sorry

end find_number_l1139_113951


namespace white_area_of_sign_l1139_113987

theorem white_area_of_sign : 
  let total_area := 6 * 18
  let F_area := 2 * (4 * 1) + 6 * 1
  let O_area := 2 * (6 * 1) + 2 * (4 * 1)
  let D_area := 6 * 1 + 4 * 1 + 4 * 1
  let total_black_area := F_area + O_area + O_area + D_area
  total_area - total_black_area = 40 :=
by
  sorry

end white_area_of_sign_l1139_113987


namespace find_value_of_X_l1139_113956

theorem find_value_of_X :
  let X_initial := 5
  let S_initial := 0
  let X_increment := 3
  let target_sum := 15000
  let X := X_initial + X_increment * 56
  2 * target_sum ≥ 3 * 57 * 57 + 7 * 57 →
  X = 173 :=
by
  sorry

end find_value_of_X_l1139_113956


namespace abs_inequality_solution_set_l1139_113949

-- Define the main problem as a Lean theorem statement
theorem abs_inequality_solution_set (x : ℝ) : 
  (|x - 5| + |x + 3| ≥ 10 ↔ (x ≤ -4 ∨ x ≥ 6)) :=
by {
  sorry
}

end abs_inequality_solution_set_l1139_113949


namespace max_area_rectangle_l1139_113948

-- Define the conditions using Lean
def is_rectangle (length width : ℕ) : Prop :=
  2 * (length + width) = 34

-- Define the problem as a theorem in Lean
theorem max_area_rectangle : ∃ (length width : ℕ), is_rectangle length width ∧ length * width = 72 :=
by
  sorry

end max_area_rectangle_l1139_113948


namespace find_unknown_number_l1139_113982

theorem find_unknown_number (x : ℕ) :
  (x + 30 + 50) / 3 = ((20 + 40 + 6) / 3 + 8) → x = 10 := by
    sorry

end find_unknown_number_l1139_113982


namespace loss_percentage_l1139_113970

theorem loss_percentage (cost_price selling_price : ℝ) (h_cost : cost_price = 1500) (h_sell : selling_price = 1260) : 
  (cost_price - selling_price) / cost_price * 100 = 16 := 
by
  sorry

end loss_percentage_l1139_113970


namespace painting_colors_area_l1139_113946

theorem painting_colors_area
  (B G Y : ℕ)
  (h_total_blue : B + (1 / 3 : ℝ) * G = 38)
  (h_total_yellow : Y + (2 / 3 : ℝ) * G = 38)
  (h_grass_sky_relation : G = B + 6) :
  B = 27 ∧ G = 33 ∧ Y = 16 :=
by
  sorry

end painting_colors_area_l1139_113946


namespace value_of_f_10_l1139_113964

def f (n : ℕ) : ℕ := n^2 - n + 17

theorem value_of_f_10 : f 10 = 107 := by
  sorry

end value_of_f_10_l1139_113964


namespace limit_expression_l1139_113995

theorem limit_expression :
  (∀ (n : ℕ), ∃ l : ℝ, 
    ∀ ε > 0, ∃ N : ℕ, n > N → 
      abs (( (↑(n) + 1)^3 - (↑(n) - 1)^3) / ((↑(n) + 1)^2 + (↑(n) - 1)^2) - l) < ε) 
  → l = 3 :=
sorry

end limit_expression_l1139_113995


namespace inequality_holds_l1139_113936

theorem inequality_holds (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  (x^3 / (x^3 + 2 * y^2 * z) + y^3 / (y^3 + 2 * z^2 * x) + z^3 / (z^3 + 2 * x^2 * y)) ≥ 1 :=
by
  sorry

end inequality_holds_l1139_113936


namespace simplify_fraction_l1139_113966

theorem simplify_fraction : (81000 ^ 3) / (27000 ^ 3) = 27 := by
  sorry

end simplify_fraction_l1139_113966


namespace max_non_overlapping_triangles_l1139_113914

variable (L : ℝ) (n : ℕ)
def equilateral_triangle (L : ℝ) := true   -- Placeholder definition for equilateral triangle 
def non_overlapping_interior := true        -- Placeholder definition for non-overlapping condition
def unit_triangle_orientation_shift := true -- Placeholder for orientation condition

theorem max_non_overlapping_triangles (L_pos : 0 < L)
                                    (h1 : equilateral_triangle L)
                                    (h2 : ∀ i, i < n → non_overlapping_interior)
                                    (h3 : ∀ i, i < n → unit_triangle_orientation_shift) :
                                    n ≤ (2 : ℝ) / 3 * L^2 := 
by 
  sorry

end max_non_overlapping_triangles_l1139_113914


namespace sin_330_degree_l1139_113921

theorem sin_330_degree : Real.sin (330 * Real.pi / 180) = -1 / 2 :=
by
  sorry

end sin_330_degree_l1139_113921


namespace find_n_eq_5_l1139_113959

variable {a_n b_n : ℕ → ℤ}

def a (n : ℕ) : ℤ := 2 + 3 * (n - 1)
def b (n : ℕ) : ℤ := -2 + 4 * (n - 1)

theorem find_n_eq_5 :
  ∃ n : ℕ, a n = b n ∧ n = 5 :=
by
  sorry

end find_n_eq_5_l1139_113959


namespace determine_a_l1139_113991

theorem determine_a
  (a b : ℝ)
  (P1 P2 : ℝ × ℝ)
  (direction_vector : ℝ × ℝ)
  (h1 : P1 = (-3, 4))
  (h2 : P2 = (4, -1))
  (h3 : direction_vector = (4 - (-3), -1 - 4))
  (h4 : b = a / 2)
  (h5 : direction_vector = (7, -5)) :
  a = -10 :=
sorry

end determine_a_l1139_113991


namespace least_xy_value_l1139_113907

theorem least_xy_value (x y : ℕ) (hx : 0 < x) (hy : 0 < y) 
  (h : 1 / x + 1 / (3 * y) = 1 / 6) : x * y = 96 :=
by sorry

end least_xy_value_l1139_113907


namespace oliver_final_amount_l1139_113919

def initial_amount : ℤ := 33
def spent : ℤ := 4
def received : ℤ := 32

def final_amount (initial_amount spent received : ℤ) : ℤ :=
  initial_amount - spent + received

theorem oliver_final_amount : final_amount initial_amount spent received = 61 := 
by sorry

end oliver_final_amount_l1139_113919


namespace positive_divisors_multiple_of_5_l1139_113952

theorem positive_divisors_multiple_of_5 (a b c : ℕ) (h_a : 0 ≤ a ∧ a ≤ 2) (h_b : 0 ≤ b ∧ b ≤ 3) (h_c : 1 ≤ c ∧ c ≤ 2) :
  (a * b * c = 3 * 4 * 2) :=
sorry

end positive_divisors_multiple_of_5_l1139_113952


namespace num_factors_x_l1139_113912

theorem num_factors_x (x : ℕ) (h : 2011^(2011^2012) = x^x) : ∃ n : ℕ, n = 2012 ∧  ∀ d : ℕ, d ∣ x -> d ≤ n :=
sorry

end num_factors_x_l1139_113912
