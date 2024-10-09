import Mathlib

namespace find_n_l2343_234354

theorem find_n (n : ℕ) :
  (∀ k : ℕ, k > 0 → k^2 + (n / k^2) ≥ 1991) ∧ (∃ k : ℕ, k > 0 ∧ k^2 + (n / k^2) < 1992) ↔ 967 * 1024 ≤ n ∧ n < 968 * 1024 :=
by
  sorry

end find_n_l2343_234354


namespace sequence_converges_l2343_234300

open Real

theorem sequence_converges (x : ℕ → ℝ) (h₀ : ∀ n, x (n + 1) = 1 + x n - 0.5 * (x n) ^ 2) (h₁ : 1 < x 1 ∧ x 1 < 2) :
  ∀ n ≥ 3, |x n - sqrt 2| < 2 ^ (-n : ℝ) :=
by
  sorry

end sequence_converges_l2343_234300


namespace find_x_l2343_234378

theorem find_x (x : ℝ) :
  (x^2 - 7 * x + 12) / (x^2 - 9 * x + 20) = (x^2 - 4 * x - 21) / (x^2 - 5 * x - 24) -> x = 11 :=
by
  sorry

end find_x_l2343_234378


namespace xyz_value_l2343_234330

-- We define the constants from the problem
variables {x y z : ℂ}

-- Here's the theorem statement in Lean 4.
theorem xyz_value :
  (x * y + 5 * y = -20) →
  (y * z + 5 * z = -20) →
  (z * x + 5 * x = -20) →
  x * y * z = 100 :=
by
  intros h1 h2 h3
  sorry

end xyz_value_l2343_234330


namespace min_elements_of_B_l2343_234352

def A (k : ℝ) : Set ℝ :=
if k < 0 then {x | (k / 4 + 9 / (4 * k) + 3) < x ∧ x < 11 / 2}
else if k = 0 then {x | x < 11 / 2}
else if 0 < k ∧ k < 1 ∨ k > 9 then {x | x < 11 / 2 ∨ x > k / 4 + 9 / (4 * k) + 3}
else if 1 ≤ k ∧ k ≤ 9 then {x | x < k / 4 + 9 / (4 * k) + 3 ∨ x > 11 / 2}
else ∅

def B (k : ℝ) : Set ℤ := {x : ℤ | ↑x ∈ A k}

theorem min_elements_of_B (k : ℝ) (hk : k < 0) : 
  B k = {2, 3, 4, 5} :=
sorry

end min_elements_of_B_l2343_234352


namespace cubic_sum_l2343_234357

theorem cubic_sum (a b c : ℤ) (h1 : a + b + c = 7) (h2 : a * b + a * c + b * c = 11) (h3 : a * b * c = -6) :
  a^3 + b^3 + c^3 = 223 :=
by
  sorry

end cubic_sum_l2343_234357


namespace original_number_l2343_234327

theorem original_number (x : ℤ) (h : 5 * x - 9 = 51) : x = 12 :=
sorry

end original_number_l2343_234327


namespace no_valid_six_digit_palindrome_years_l2343_234329

noncomputable def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

noncomputable def is_six_digit_palindrome (n : ℕ) : Prop :=
  100000 ≤ n ∧ n ≤ 999999 ∧ is_palindrome n

noncomputable def is_four_digit_prime_palindrome (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧ is_palindrome n ∧ is_prime n

noncomputable def is_two_digit_prime_palindrome (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧ is_palindrome n ∧ is_prime n

theorem no_valid_six_digit_palindrome_years :
  ∀ N : ℕ, is_six_digit_palindrome N →
  ¬ ∃ (p q : ℕ), is_four_digit_prime_palindrome p ∧ is_two_digit_prime_palindrome q ∧ N = p * q := 
sorry

end no_valid_six_digit_palindrome_years_l2343_234329


namespace solution_of_inequality_answer_A_incorrect_answer_B_incorrect_answer_C_incorrect_D_is_correct_l2343_234388

theorem solution_of_inequality (a b x : ℝ) :
    (b - a * x > 0) ↔
    (a > 0 ∧ x < b / a ∨ 
     a < 0 ∧ x > b / a ∨ 
     a = 0 ∧ false) :=
by sorry

-- Additional theorems to rule out incorrect answers
theorem answer_A_incorrect (a b : ℝ) :
    (∀ x : ℝ, b - a * x > 0 → x > |b| / |a|) → false :=
by sorry

theorem answer_B_incorrect (a b : ℝ) :
    (∀ x : ℝ, b - a * x > 0 → x < |b| / |a|) → false :=
by sorry

theorem answer_C_incorrect (a b : ℝ) :
    (∀ x : ℝ, b - a * x > 0 → x > -|b| / |a|) → false :=
by sorry

theorem D_is_correct (a b : ℝ) :
    (∀ x : ℝ, b - a * x > 0 → x > |b| / |a| ∨ x < |b| / |a| ∨ x > -|b| / |a|) → false :=
by sorry

end solution_of_inequality_answer_A_incorrect_answer_B_incorrect_answer_C_incorrect_D_is_correct_l2343_234388


namespace number_of_female_workers_l2343_234345

theorem number_of_female_workers (M F : ℕ) (M_no F_no : ℝ) 
  (hM : M = 112)
  (h1 : M_no = 0.40 * M)
  (h2 : F_no = 0.25 * F)
  (h3 : M_no / (M_no + F_no) = 0.30)
  (h4 : F_no / (M_no + F_no) = 0.70)
  : F = 420 := 
by 
  sorry

end number_of_female_workers_l2343_234345


namespace trigonometric_identity_l2343_234356

open Real

theorem trigonometric_identity (θ : ℝ) (h : π / 4 < θ ∧ θ < π / 2) :
  2 * cos θ + sqrt (1 - 2 * sin (π - θ) * cos θ) = sin θ + cos θ :=
sorry

end trigonometric_identity_l2343_234356


namespace quadratic_inequality_solution_l2343_234309

theorem quadratic_inequality_solution
  (a : ℝ) :
  (∀ x : ℝ, (a-2)*x^2 + 2*(a-2)*x - 4 ≤ 0) ↔ -2 ≤ a ∧ a ≤ 2 :=
by
  sorry

end quadratic_inequality_solution_l2343_234309


namespace domain_of_function_l2343_234360

theorem domain_of_function:
  {x : ℝ | x^2 - 5*x + 6 > 0 ∧ x ≠ 3} = {x : ℝ | x < 2 ∨ x > 3} :=
by
  sorry

end domain_of_function_l2343_234360


namespace prob_A_wins_match_expected_games_won_variance_games_won_l2343_234301

-- Definitions of probabilities
def prob_A_win := 0.6
def prob_B_win := 0.4

-- Prove that the probability of A winning the match is 0.648
theorem prob_A_wins_match : 
  prob_A_win * prob_A_win + 2 * prob_B_win * prob_A_win * prob_A_win = 0.648 :=
  sorry

-- Define the expected number of games won by A
noncomputable def expected_games_won_by_A := 
  0 * (prob_B_win * prob_B_win) + 1 * (2 * prob_A_win * prob_B_win * prob_B_win) + 
  2 * (prob_A_win * prob_A_win + 2 * prob_B_win * prob_A_win * prob_A_win)

-- Prove the expected number of games won by A is 1.5
theorem expected_games_won : 
  expected_games_won_by_A = 1.5 :=
  sorry

-- Define the variance of the number of games won by A
noncomputable def variance_games_won_by_A := 
  (prob_B_win * prob_B_win) * (0 - 1.5)^2 + 
  (2 * prob_A_win * prob_B_win * prob_B_win) * (1 - 1.5)^2 + 
  (prob_A_win * prob_A_win + 2 * prob_B_win * prob_A_win * prob_A_win) * (2 - 1.5)^2

-- Prove the variance of the number of games won by A is 0.57
theorem variance_games_won : 
  variance_games_won_by_A = 0.57 :=
  sorry

end prob_A_wins_match_expected_games_won_variance_games_won_l2343_234301


namespace pounds_of_oranges_l2343_234321

noncomputable def price_of_pounds_oranges (E O : ℝ) (P : ℕ) : Prop :=
  let current_total_price := E
  let increased_total_price := 1.09 * E + 1.06 * (O * P)
  (increased_total_price - current_total_price) = 15

theorem pounds_of_oranges (E O : ℝ) (P : ℕ): 
  E = O * P ∧ 
  (price_of_pounds_oranges E O P) → 
  P = 100 := 
by
  sorry

end pounds_of_oranges_l2343_234321


namespace find_D_l2343_234320

noncomputable def Point : Type := ℝ × ℝ

-- Given points A, B, and C
def A : Point := (-2, 0)
def B : Point := (6, 8)
def C : Point := (8, 6)

-- Condition: AB parallel to DC and AD parallel to BC, which means it is a parallelogram
def is_parallelogram (A B C D : Point) : Prop :=
  ((B.1 - A.1, B.2 - A.2) = (D.1 - C.1, D.2 - C.2)) ∧
  ((C.1 - B.1, C.2 - B.2) = (D.1 - A.1, D.2 - A.2))

-- Proves that with given A, B, and C, D should be (0, -2)
theorem find_D : ∃ D : Point, is_parallelogram A B C D ∧ D = (0, -2) :=
  by sorry

end find_D_l2343_234320


namespace belts_count_l2343_234373

-- Definitions based on conditions
variable (shoes belts hats : ℕ)

-- Conditions from the problem
axiom shoes_eq_14 : shoes = 14
axiom hat_count : hats = 5
axiom shoes_double_of_belts : shoes = 2 * belts

-- Definition of the theorem to prove the number of belts
theorem belts_count : belts = 7 :=
by
  sorry

end belts_count_l2343_234373


namespace point_reflection_x_axis_l2343_234366

-- Definition of the original point P
def P : ℝ × ℝ := (-2, 5)

-- Function to reflect a point across the x-axis
def reflect_x_axis (point : ℝ × ℝ) : ℝ × ℝ :=
  (point.1, -point.2)

-- Our theorem
theorem point_reflection_x_axis :
  reflect_x_axis P = (-2, -5) := by
  sorry

end point_reflection_x_axis_l2343_234366


namespace remainder_b96_div_50_l2343_234323

theorem remainder_b96_div_50 (b : ℕ → ℕ) (h : ∀ n, b n = 7^n + 9^n) : b 96 % 50 = 2 :=
by
  -- The proof is omitted.
  sorry

end remainder_b96_div_50_l2343_234323


namespace original_money_in_wallet_l2343_234332

-- Definitions based on the problem's conditions
def grandmother_gift : ℕ := 20
def aunt_gift : ℕ := 25
def uncle_gift : ℕ := 30
def cost_per_game : ℕ := 35
def number_of_games : ℕ := 3
def money_left : ℕ := 20

-- Calculations as specified in the solution
def birthday_money := grandmother_gift + aunt_gift + uncle_gift
def total_game_cost := cost_per_game * number_of_games
def total_money_before_purchase := total_game_cost + money_left

-- Proof that the original amount of money in Geoffrey's wallet
-- was €50 before he got the birthday money and made the purchase.
theorem original_money_in_wallet : total_money_before_purchase - birthday_money = 50 := by
  sorry

end original_money_in_wallet_l2343_234332


namespace third_restaurant_meals_per_day_l2343_234390

-- Define the daily meals served by the first two restaurants
def meals_first_restaurant_per_day : ℕ := 20
def meals_second_restaurant_per_day : ℕ := 40

-- Define the total meals served by all three restaurants per week
def total_meals_per_week : ℕ := 770

-- Define the weekly meals served by the first two restaurants
def meals_first_restaurant_per_week : ℕ := meals_first_restaurant_per_day * 7
def meals_second_restaurant_per_week : ℕ := meals_second_restaurant_per_day * 7

-- Total weekly meals served by the first two restaurants
def total_meals_first_two_restaurants_per_week : ℕ := meals_first_restaurant_per_week + meals_second_restaurant_per_week

-- Weekly meals served by the third restaurant
def meals_third_restaurant_per_week : ℕ := total_meals_per_week - total_meals_first_two_restaurants_per_week

-- Convert weekly meals served by the third restaurant to daily meals
def meals_third_restaurant_per_day : ℕ := meals_third_restaurant_per_week / 7

-- Goal: Prove the third restaurant serves 50 meals per day
theorem third_restaurant_meals_per_day : meals_third_restaurant_per_day = 50 := by
  -- proof skipped
  sorry

end third_restaurant_meals_per_day_l2343_234390


namespace water_overflowed_calculation_l2343_234328

/-- The water supply rate is 200 kilograms per hour. -/
def water_supply_rate : ℕ := 200

/-- The water tank capacity is 4000 kilograms. -/
def tank_capacity : ℕ := 4000

/-- The water runs for 24 hours. -/
def running_time : ℕ := 24

/-- Calculation for the kilograms of water that overflowed. -/
theorem water_overflowed_calculation :
  water_supply_rate * running_time - tank_capacity = 800 :=
by
  -- calculation skipped
  sorry

end water_overflowed_calculation_l2343_234328


namespace men_per_table_l2343_234333

theorem men_per_table 
  (num_tables : ℕ) 
  (women_per_table : ℕ) 
  (total_customers : ℕ) 
  (h1 : num_tables = 9) 
  (h2 : women_per_table = 7) 
  (h3 : total_customers = 90)
  : (total_customers - num_tables * women_per_table) / num_tables = 3 :=
by
  sorry

end men_per_table_l2343_234333


namespace initial_ratio_men_to_women_l2343_234325

theorem initial_ratio_men_to_women (M W : ℕ) (h1 : (W - 3) * 2 = 24) (h2 : 14 - 2 = M) : M / gcd M W = 4 ∧ W / gcd M W = 5 := by 
  sorry

end initial_ratio_men_to_women_l2343_234325


namespace find_k_l2343_234326

def f (x : ℝ) : ℝ := 3 * x ^ 2 - 2 * x + 8
def g (x : ℝ) (k : ℝ) : ℝ := x ^ 2 - k * x + 3

theorem find_k : 
  (f 5 - g 5 k = 12) → k = -53 / 5 :=
by
  intro hyp
  sorry

end find_k_l2343_234326


namespace find_sin_E_floor_l2343_234370

variable {EF GH EH FG : ℝ}
variable (E G : ℝ)

-- Conditions from the problem
def is_convex_quadrilateral (EF GH EH FG : ℝ) : Prop := true
def angles_congruent (E G : ℝ) : Prop := E = G
def sides_equal (EF GH : ℝ) : Prop := EF = GH ∧ EF = 200
def sides_not_equal (EH FG : ℝ) : Prop := EH ≠ FG
def perimeter (EF GH EH FG : ℝ) : Prop := EF + GH + EH + FG = 800

-- The theorem to be proved
theorem find_sin_E_floor (h_convex : is_convex_quadrilateral EF GH EH FG)
                         (h_angles : angles_congruent E G)
                         (h_sides : sides_equal EF GH)
                         (h_sides_ne : sides_not_equal EH FG)
                         (h_perimeter : perimeter EF GH EH FG) :
  ⌊ 1000 * Real.sin E ⌋ = 0 := by
  sorry

end find_sin_E_floor_l2343_234370


namespace molecular_weight_CaCO3_is_100_09_l2343_234362

-- Declare the atomic weights
def atomic_weight_Ca : ℝ := 40.08
def atomic_weight_C : ℝ := 12.01
def atomic_weight_O : ℝ := 16.00

-- Define the molecular weight constant for calcium carbonate
def molecular_weight_CaCO3 : ℝ :=
  (1 * atomic_weight_Ca) + (1 * atomic_weight_C) + (3 * atomic_weight_O)

-- Prove that the molecular weight of calcium carbonate is 100.09 g/mol
theorem molecular_weight_CaCO3_is_100_09 :
  molecular_weight_CaCO3 = 100.09 :=
by
  -- Proof goes here, placeholder for now
  sorry

end molecular_weight_CaCO3_is_100_09_l2343_234362


namespace range_of_a_if_exists_x_l2343_234386

variable {a x : ℝ}

theorem range_of_a_if_exists_x :
  (∃ x : ℝ, -1 < x ∧ x < 1 ∧ (a * x^2 - 1 ≥ 0)) → (a > 1) :=
by
  sorry

end range_of_a_if_exists_x_l2343_234386


namespace id_tags_divided_by_10_l2343_234305

def uniqueIDTags (chars : List Char) (counts : Char → Nat) : Nat :=
  let permsWithoutRepetition := 
    Nat.factorial 7 / Nat.factorial (7 - 5)
  let repeatedCharTagCount := 10 * 10 * 6
  permsWithoutRepetition + repeatedCharTagCount

theorem id_tags_divided_by_10 :
  uniqueIDTags ['M', 'A', 'T', 'H', '2', '0', '3'] (fun c =>
    if c = 'M' then 1 else
    if c = 'A' then 1 else
    if c = 'T' then 1 else
    if c = 'H' then 1 else
    if c = '2' then 2 else
    if c = '0' then 1 else
    if c = '3' then 1 else 0) / 10 = 312 :=
by
  sorry

end id_tags_divided_by_10_l2343_234305


namespace num_balls_picked_l2343_234312

-- Definitions based on the conditions
def numRedBalls : ℕ := 4
def numBlueBalls : ℕ := 3
def numGreenBalls : ℕ := 2
def totalBalls : ℕ := numRedBalls + numBlueBalls + numGreenBalls
def probFirstRed : ℚ := numRedBalls / totalBalls
def probSecondRed : ℚ := (numRedBalls - 1) / (totalBalls - 1)

-- Theorem stating the problem
theorem num_balls_picked :
  probFirstRed * probSecondRed = 1 / 6 → 
  (∃ (n : ℕ), n = 2) :=
by 
  sorry

end num_balls_picked_l2343_234312


namespace find_varphi_intervals_of_increase_l2343_234336

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := Real.sin (2 * x + φ)

theorem find_varphi (φ : ℝ) (h1 : -Real.pi < φ) (h2 : φ < 0)
  (h3 : ∃ k : ℤ, 2 * (Real.pi / 8) + φ = (Real.pi / 2) + k * Real.pi) :
  φ = -3 * Real.pi / 4 :=
sorry

theorem intervals_of_increase (m : ℤ) :
  ∀ x : ℝ, (π / 8 + m * π ≤ x ∧ x ≤ 5 * π / 8 + m * π) ↔
  Real.sin (2 * x - 3 * π / 4) > 0 :=
sorry

end find_varphi_intervals_of_increase_l2343_234336


namespace solution_is_thirteen_over_nine_l2343_234344

noncomputable def check_solution (x : ℝ) : Prop :=
  (3 * x^2 / (x - 2) - (3 * x + 9) / 4 + (6 - 9 * x) / (x - 2) + 2 = 0) ∧
  (x^3 ≠ 3 * x + 1)

theorem solution_is_thirteen_over_nine :
  check_solution (13 / 9) :=
by
  sorry

end solution_is_thirteen_over_nine_l2343_234344


namespace inequality_solution_l2343_234364

theorem inequality_solution (a x : ℝ) : 
  (ax^2 + (2 - a) * x - 2 < 0) → 
  ((a = 0) → x < 1) ∧ 
  ((a > 0) → (-2/a < x ∧ x < 1)) ∧ 
  ((a < 0) → 
    ((-2 < a ∧ a < 0) → (x < 1 ∨ x > -2/a)) ∧
    (a = -2 → (x ≠ 1)) ∧
    (a < -2 → (x < -2/a ∨ x > 1)))
:=
sorry

end inequality_solution_l2343_234364


namespace tablecloth_overhang_l2343_234317

theorem tablecloth_overhang (d r l overhang1 overhang2 : ℝ) (h1 : d = 0.6) (h2 : r = d / 2) (h3 : l = 1) 
  (h4 : overhang1 = 0.5) (h5 : overhang2 = 0.3) :
  ∃ overhang3 overhang4 : ℝ, overhang3 = 0.33 ∧ overhang4 = 0.52 := 
sorry

end tablecloth_overhang_l2343_234317


namespace ratio_length_to_breadth_l2343_234306

theorem ratio_length_to_breadth (l b : ℕ) (h1 : b = 14) (h2 : l * b = 588) : l / b = 3 :=
by
  sorry

end ratio_length_to_breadth_l2343_234306


namespace kristin_runs_around_l2343_234302

-- Definitions of the conditions.
def kristin_runs_faster (v_k v_s : ℝ) : Prop := v_k = 3 * v_s
def sarith_runs_times (S : ℕ) : Prop := S = 8
def field_length (c_field a_field : ℝ) : Prop := c_field = a_field / 2

-- The question is to prove Kristin runs around the field 12 times.
def kristin_runs_times (K : ℕ) : Prop := K = 12

-- The main theorem statement combining conditions to prove the question.
theorem kristin_runs_around :
  ∀ (v_k v_s c_field a_field : ℝ) (S K : ℕ),
    kristin_runs_faster v_k v_s →
    sarith_runs_times S →
    field_length c_field a_field →
    K = (S : ℝ) * (3 / 2) →
    kristin_runs_times K :=
by sorry

end kristin_runs_around_l2343_234302


namespace union_of_sets_l2343_234387

open Set

variable (a : ℤ)

def setA : Set ℤ := {1, 3}
def setB (a : ℤ) : Set ℤ := {a + 2, 5}

theorem union_of_sets (h : {3} = setA ∩ setB a) : setA ∪ setB a = {1, 3, 5} :=
by
  sorry

end union_of_sets_l2343_234387


namespace convert_mps_to_kmph_l2343_234334

theorem convert_mps_to_kmph (v_mps : ℝ) (conversion_factor : ℝ) : v_mps = 22 → conversion_factor = 3.6 → v_mps * conversion_factor = 79.2 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num

end convert_mps_to_kmph_l2343_234334


namespace range_of_a_l2343_234351

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 - a * x - 1 else a / x

def is_increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ s → y ∈ s → x < y → f x ≤ f y

def func_increasing_on_R (a : ℝ) : Prop :=
  is_increasing_on (f a) Set.univ

theorem range_of_a (a : ℝ) : func_increasing_on_R a ↔ a < -2 :=
sorry

end range_of_a_l2343_234351


namespace assistant_professor_pencils_l2343_234389

theorem assistant_professor_pencils :
  ∀ (A B P : ℕ), 
    A + B = 7 →
    2 * A + P * B = 10 →
    A + 2 * B = 11 →
    P = 1 :=
by 
  sorry

end assistant_professor_pencils_l2343_234389


namespace base_conversion_sum_l2343_234314

-- Definition of conversion from base 13 to base 10
def base13_to_base10 (n : ℕ) : ℕ :=
  3 * (13^2) + 4 * (13^1) + 5 * (13^0)

-- Definition of conversion from base 14 to base 10 where C = 12 and D = 13
def base14_to_base10 (m : ℕ) : ℕ :=
  4 * (14^2) + 12 * (14^1) + 13 * (14^0)

theorem base_conversion_sum :
  base13_to_base10 345 + base14_to_base10 (4 * 14^2 + 12 * 14 + 13) = 1529 := 
by
  sorry -- proof to be provided

end base_conversion_sum_l2343_234314


namespace valid_integer_lattice_points_count_l2343_234380

def point := (ℤ × ℤ)
def A : point := (-4, 3)
def B : point := (4, -3)

def manhattan_distance (p1 p2 : point) : ℤ :=
  abs (p2.1 - p1.1) + abs (p2.2 - p1.2)

def valid_path_length (p1 p2 : point) : Prop :=
  manhattan_distance p1 p2 ≤ 18

def does_not_cross_y_eq_x (p1 p2 : point) : Prop :=
  ∀ x y, (x, y) ∈ [(p1, p2)] → y ≠ x

def integer_lattice_points_on_path (p1 p2 : point) : ℕ := sorry

theorem valid_integer_lattice_points_count :
  integer_lattice_points_on_path A B = 112 :=
sorry

end valid_integer_lattice_points_count_l2343_234380


namespace slope_range_l2343_234307

open Real

theorem slope_range (k : ℝ) :
  (∃ b : ℝ, 
    ∃ x1 x2 x3 : ℝ,
      (x1 + x2 + x3 = 0) ∧
      (x1 ≥ 0) ∧ (x2 ≥ 0) ∧ (x3 < 0) ∧
      ((kx1 + b) = ((x1 + 1) / (|x1| + 1))) ∧
      ((kx2 + b) = ((x2 + 1) / (|x2| + 1))) ∧
      ((kx3 + b) = ((x3 + 1) / (|x3| + 1)))) →
  (0 < k ∧ k < (2 / 9)) :=
sorry

end slope_range_l2343_234307


namespace unique_positive_integer_n_l2343_234308

theorem unique_positive_integer_n (n x : ℕ) (hx : x > 0) (hn : n = 2 ^ (2 * x - 1) - 5 * x - 3 ∧ n = (2 ^ (x-1) - 1) * (2 ^ x + 1)) : n = 2015 := by
  sorry

end unique_positive_integer_n_l2343_234308


namespace product_of_tangents_is_constant_l2343_234383

theorem product_of_tangents_is_constant (a b : ℝ) (h_ab : a > b) (P : ℝ × ℝ)
  (hP_on_ellipse : P.1^2 / a^2 + P.2^2 / b^2 = 1)
  (A1 A2 : ℝ × ℝ)
  (hA1 : A1 = (-a, 0))
  (hA2 : A2 = (a, 0)) :
  ∃ (Q1 Q2 : ℝ × ℝ),
  (A1.1 - Q1.1, A2.1 - Q2.1) = (b^2, b^2) :=
sorry

end product_of_tangents_is_constant_l2343_234383


namespace box_contains_1_8_grams_child_ingests_0_1_grams_l2343_234303

-- Define the conditions
def packet_weight : ℝ := 0.2
def packets_in_box : ℕ := 9
def half_a_packet : ℝ := 0.5

-- Prove that a box contains 1.8 grams of "acetaminophen"
theorem box_contains_1_8_grams : packets_in_box * packet_weight = 1.8 :=
by
  sorry

-- Prove that a child will ingest 0.1 grams of "acetaminophen" if they take half a packet
theorem child_ingests_0_1_grams : half_a_packet * packet_weight = 0.1 :=
by
  sorry

end box_contains_1_8_grams_child_ingests_0_1_grams_l2343_234303


namespace range_of_k_l2343_234384

open Real

noncomputable def f (x : ℝ) : ℝ := x^2 + x - 1

noncomputable def g (x : ℝ) : ℝ := x^2 - 1

noncomputable def h (x : ℝ) : ℝ := x

theorem range_of_k (k : ℝ) :
  (∀ x : ℝ, x ≠ 0 → g (k * x + k / x) < g (x^2 + 1 / x^2 + 1)) ↔ (-3 / 2 < k ∧ k < 3 / 2) :=
by
  sorry

end range_of_k_l2343_234384


namespace root_relationship_l2343_234395

theorem root_relationship (m n a b : ℝ) 
  (h_eq : ∀ x, 3 - (x - m) * (x - n) = 0 ↔ x = a ∨ x = b) : a < m ∧ m < n ∧ n < b :=
by
  sorry

end root_relationship_l2343_234395


namespace number_of_glass_bottles_l2343_234368

theorem number_of_glass_bottles (total_litter : ℕ) (aluminum_cans : ℕ) (glass_bottles : ℕ) : 
  total_litter = 18 → aluminum_cans = 8 → glass_bottles = total_litter - aluminum_cans → glass_bottles = 10 :=
by
  intros h_total h_aluminum h_glass
  rw [h_total, h_aluminum] at h_glass
  exact h_glass.trans rfl


end number_of_glass_bottles_l2343_234368


namespace number_of_articles_l2343_234365

theorem number_of_articles (C S : ℝ) (h_gain : S = 1.4285714285714286 * C) (h_cost : ∃ X : ℝ, X * C = 35 * S) : ∃ X : ℝ, X = 50 :=
by
  -- Define the specific existence and equality proof here
  sorry

end number_of_articles_l2343_234365


namespace find_coefficients_l2343_234339

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem find_coefficients 
  (A B Q C P : V) 
  (hQ : Q = (5 / 7 : ℝ) • A + (2 / 7 : ℝ) • B)
  (hC : C = A + 2 • B)
  (hP : P = Q + C) : 
  ∃ s v : ℝ, P = s • A + v • B ∧ s = 12 / 7 ∧ v = 16 / 7 :=
by
  sorry

end find_coefficients_l2343_234339


namespace intersection_correct_l2343_234316

open Set

def M := {x : ℝ | x^2 + x - 6 < 0}
def N := {x : ℝ | 1 ≤ x ∧ x ≤ 3}
def intersection := (M ∩ N) = {x : ℝ | 1 ≤ x ∧ x < 2}

theorem intersection_correct : intersection := by
  sorry

end intersection_correct_l2343_234316


namespace ratio_of_ages_l2343_234369

variable (D R : ℕ)

theorem ratio_of_ages : (D = 9) → (R + 6 = 18) → (R / D = 4 / 3) :=
by
  intros hD hR
  -- proof goes here
  sorry

end ratio_of_ages_l2343_234369


namespace determine_all_functions_l2343_234313

-- Define the natural numbers (ℕ) as positive integers
def is_perfect_square (x : ℕ) : Prop :=
  ∃ k : ℕ, k * k = x

theorem determine_all_functions (g : ℕ → ℕ) :
  (∀ m n : ℕ, is_perfect_square ((g m + n) * (m + g n))) →
  ∃ c : ℕ, ∀ n : ℕ, g n = n + c :=
by
  sorry

end determine_all_functions_l2343_234313


namespace odd_operations_l2343_234331

theorem odd_operations (a b : ℤ) (ha : ∃ k : ℤ, a = 2 * k + 1) (hb : ∃ j : ℤ, b = 2 * j + 1) :
  (∃ k : ℤ, (a * b) = 2 * k + 1) ∧ (∃ m : ℤ, a^2 = 2 * m + 1) :=
by {
  sorry
}

end odd_operations_l2343_234331


namespace tank_empty_time_when_inlet_open_l2343_234340

-- Define the conditions
def leak_empty_time : ℕ := 6
def tank_capacity : ℕ := 4320
def inlet_rate_per_minute : ℕ := 6

-- Calculate rates from conditions
def leak_rate_per_hour : ℕ := tank_capacity / leak_empty_time
def inlet_rate_per_hour : ℕ := inlet_rate_per_minute * 60

-- Proof Problem: Prove the time for the tank to empty when both leak and inlet are open
theorem tank_empty_time_when_inlet_open :
  tank_capacity / (leak_rate_per_hour - inlet_rate_per_hour) = 12 :=
by
  sorry

end tank_empty_time_when_inlet_open_l2343_234340


namespace fixed_point_min_value_l2343_234381

theorem fixed_point_min_value {a m n : ℝ} (ha_pos : 0 < a) (ha_ne_one : a ≠ 1) (hm_pos : 0 < m) (hn_pos : 0 < n)
  (h : 3 * m + n = 1) : (1 / m + 3 / n) = 12 := sorry

end fixed_point_min_value_l2343_234381


namespace probability_same_color_is_27_over_100_l2343_234342

def num_sides_die1 := 20
def num_sides_die2 := 20

def maroon_die1 := 5
def teal_die1 := 6
def cyan_die1 := 7
def sparkly_die1 := 1
def silver_die1 := 1

def maroon_die2 := 4
def teal_die2 := 6
def cyan_die2 := 7
def sparkly_die2 := 1
def silver_die2 := 2

noncomputable def probability_same_color : ℚ :=
  (maroon_die1 * maroon_die2 + teal_die1 * teal_die2 + cyan_die1 * cyan_die2 + sparkly_die1 * sparkly_die2 + silver_die1 * silver_die2) /
  (num_sides_die1 * num_sides_die2)

theorem probability_same_color_is_27_over_100 :
  probability_same_color = 27 / 100 := 
sorry

end probability_same_color_is_27_over_100_l2343_234342


namespace circle_radius_l2343_234396

-- Define the main geometric scenario in Lean 4
theorem circle_radius 
  (O P A B : Type) 
  (r OP PA PB : ℝ)
  (h1 : PA * PB = 24)
  (h2 : OP = 5)
  : r = 7 
:= sorry

end circle_radius_l2343_234396


namespace spending_percentage_A_l2343_234393

def combined_salary (S_A S_B : ℝ) : Prop := S_A + S_B = 7000
def A_salary (S_A : ℝ) : Prop := S_A = 5250
def B_salary (S_B : ℝ) : Prop := S_B = 1750
def B_spending (P_B : ℝ) : Prop := P_B = 0.85
def same_savings (S_A S_B P_A P_B : ℝ) : Prop := S_A * (1 - P_A) = S_B * (1 - P_B)
def A_spending (P_A : ℝ) : Prop := P_A = 0.95

theorem spending_percentage_A (S_A S_B P_A P_B : ℝ) 
  (h1: combined_salary S_A S_B) 
  (h2: A_salary S_A) 
  (h3: B_salary S_B) 
  (h4: B_spending P_B) 
  (h5: same_savings S_A S_B P_A P_B) : A_spending P_A :=
sorry

end spending_percentage_A_l2343_234393


namespace brenda_peaches_remaining_l2343_234392

theorem brenda_peaches_remaining (total_peaches : ℕ) (percent_fresh : ℚ) (thrown_away : ℕ) (fresh_peaches : ℕ) (remaining_peaches : ℕ) :
    total_peaches = 250 → 
    percent_fresh = 0.60 → 
    thrown_away = 15 → 
    fresh_peaches = total_peaches * percent_fresh → 
    remaining_peaches = fresh_peaches - thrown_away → 
    remaining_peaches = 135 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end brenda_peaches_remaining_l2343_234392


namespace find_x_l2343_234318

theorem find_x (x y : ℕ) (h1 : x / y = 12 / 5) (h2 : y = 25) : x = 60 :=
sorry

end find_x_l2343_234318


namespace function_translation_l2343_234382

def translateLeft (f : ℝ → ℝ) (a : ℝ) : ℝ → ℝ := λ x => f (x + a)
def translateUp (f : ℝ → ℝ) (b : ℝ) : ℝ → ℝ := λ x => (f x) + b

theorem function_translation :
  (translateUp (translateLeft (λ x => 2 * x^2) 1) 3) = λ x => 2 * (x + 1)^2 + 3 :=
by
  sorry

end function_translation_l2343_234382


namespace find_k_l2343_234304

theorem find_k (x y k : ℝ) (h1 : x + y = 5 * k) (h2 : x - 2 * y = -k) (h3 : 2 * x - y = 8) : k = 2 :=
by
  sorry

end find_k_l2343_234304


namespace tournament_total_games_l2343_234361

def total_number_of_games (num_teams : ℕ) (group_size : ℕ) (num_groups : ℕ) (teams_for_knockout : ℕ) : ℕ :=
  let games_per_group := (group_size * (group_size - 1)) / 2
  let group_stage_games := num_groups * games_per_group
  let knockout_teams := num_groups * teams_for_knockout
  let knockout_games := knockout_teams - 1
  group_stage_games + knockout_games

theorem tournament_total_games : total_number_of_games 32 4 8 2 = 63 := by
  sorry

end tournament_total_games_l2343_234361


namespace geometric_sequence_sum_ratio_l2343_234391

noncomputable def geometric_sum (a q : ℝ) (n : ℕ) : ℝ := a * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum_ratio (a q : ℝ) (h : a * q^2 = 8 * a * q^5) :
  (geometric_sum a q 4) / (geometric_sum a q 2) = 5 / 4 :=
by
  -- The proof will go here.
  sorry

end geometric_sequence_sum_ratio_l2343_234391


namespace inequality_proof_l2343_234358

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ((a^2 + a + 1) * (b^2 + b + 1) * (c^2 + c + 1)) / (a * b * c) ≥ 27 :=
by
  sorry

end inequality_proof_l2343_234358


namespace simplify_expression_l2343_234375

theorem simplify_expression (r : ℝ) (h1 : r^2 ≠ 0) (h2 : r^4 > 16) :
  ( ( ( (r^2 + 4) ^ (3 : ℝ) ) ^ (1 / 3 : ℝ) * (1 + 4 / r^2) ^ (1 / 2 : ℝ) ^ (1 / 3 : ℝ)
    - ( (r^2 - 4) ^ (3 : ℝ) ) ^ (1 / 3 : ℝ) * (1 - 4 / r^2) ^ (1 / 2 : ℝ) ^ (1 / 3 : ℝ) ) ^ 2 )
  / ( r^2 - (r^4 - 16) ^ (1 / 2 : ℝ) )
  = 2 * r ^ (-(2 / 3 : ℝ)) := by
  sorry

end simplify_expression_l2343_234375


namespace min_abs_expr1_min_abs_expr2_l2343_234335

theorem min_abs_expr1 (x : ℝ) : |x - 4| + |x + 2| ≥ 6 := sorry

theorem min_abs_expr2 (x : ℝ) : |(5 / 6) * x - 1| + |(1 / 2) * x - 1| + |(2 / 3) * x - 1| ≥ 1 / 2 := sorry

end min_abs_expr1_min_abs_expr2_l2343_234335


namespace quadratic_sum_l2343_234399

theorem quadratic_sum (x : ℝ) (h : x^2 = 16*x - 9) : x = 8 ∨ x = 9 := sorry

end quadratic_sum_l2343_234399


namespace complement_intersection_l2343_234363

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)

theorem complement_intersection : 
  U = {1, 2, 3, 4, 5} → 
  A = {1, 2, 3} → 
  B = {2, 3, 4} → 
  (U \ (A ∩ B) = {1, 4, 5}) := 
by
  sorry

end complement_intersection_l2343_234363


namespace complement_unions_subset_condition_l2343_234376

open Set

-- Condition Definitions
def A : Set ℝ := {x | 1 ≤ x ∧ x < 6}
def B : Set ℝ := {x | 3 < x ∧ x < 9}
def C (a : ℝ) : Set ℝ := {x | x < a + 1}

-- Questions Translated to Lean Statements
theorem complement_unions (U : Set ℝ)
  (hU : U = univ) : (compl A ∪ compl B) = compl (A ∩ B) := by sorry

theorem subset_condition (a : ℝ)
  (h : B ⊆ C a) : a ≥ 8 := by sorry

end complement_unions_subset_condition_l2343_234376


namespace max_value_of_expression_l2343_234359

noncomputable def maximum_value (x y z : ℝ) := 8 * x + 3 * y + 10 * z

theorem max_value_of_expression :
  ∀ (x y z : ℝ), 9 * x^2 + 4 * y^2 + 25 * z^2 = 1 → maximum_value x y z ≤ (Real.sqrt 481) / 6 :=
by
  sorry

end max_value_of_expression_l2343_234359


namespace run_time_difference_l2343_234398

variables (distance duration_injured : ℝ) (initial_speed : ℝ)

theorem run_time_difference (H1 : distance = 20) 
                            (H2 : duration_injured = 22) 
                            (H3 : initial_speed = distance * 2 / duration_injured) :
                            duration_injured - (distance / initial_speed) = 11 :=
by
  sorry

end run_time_difference_l2343_234398


namespace area_of_pentagon_m_n_l2343_234350

noncomputable def m : ℤ := 12
noncomputable def n : ℤ := 11

theorem area_of_pentagon_m_n :
  let pentagon_area := (Real.sqrt m) + (Real.sqrt n)
  m + n = 23 :=
by
  have m_pos : m > 0 := by sorry
  have n_pos : n > 0 := by sorry
  sorry

end area_of_pentagon_m_n_l2343_234350


namespace white_area_correct_l2343_234310

def total_sign_area : ℕ := 8 * 20
def black_area_C : ℕ := 8 * 1 + 2 * (1 * 3)
def black_area_A : ℕ := 2 * (8 * 1) + 2 * (1 * 2)
def black_area_F : ℕ := 8 * 1 + 2 * (1 * 4)
def black_area_E : ℕ := 3 * (1 * 4)

def total_black_area : ℕ := black_area_C + black_area_A + black_area_F + black_area_E
def white_area : ℕ := total_sign_area - total_black_area

theorem white_area_correct : white_area = 98 :=
  by 
    sorry -- State the theorem without providing the proof.

end white_area_correct_l2343_234310


namespace tan_neg405_deg_l2343_234374

theorem tan_neg405_deg : Real.tan (-405 * Real.pi / 180) = -1 := by
  -- This is a placeholder for the actual proof
  sorry

end tan_neg405_deg_l2343_234374


namespace exchange_rate_lire_l2343_234343

theorem exchange_rate_lire (x : ℕ) (h : 2500 / 2 = x / 5) : x = 6250 :=
by
  sorry

end exchange_rate_lire_l2343_234343


namespace f_1982_eq_660_l2343_234341

def f : ℕ → ℕ := sorry

axiom h1 : ∀ m n : ℕ, f (m + n) - f m - f n = 0 ∨ f (m + n) - f m - f n = 1
axiom h2 : f 2 = 0
axiom h3 : f 3 > 0
axiom h4 : f 9999 = 3333

theorem f_1982_eq_660 : f 1982 = 660 := sorry

end f_1982_eq_660_l2343_234341


namespace part_one_retail_wholesale_l2343_234315

theorem part_one_retail_wholesale (x : ℕ) (wholesale : ℕ) : 
  70 * x + 40 * wholesale = 4600 ∧ x + wholesale = 100 → x = 20 ∧ wholesale = 80 :=
by
  sorry

end part_one_retail_wholesale_l2343_234315


namespace part_a_part_b_part_c_part_d_l2343_234371

-- (a)
theorem part_a : ∃ x y : ℤ, x > 0 ∧ y > 0 ∧ x ≤ 5 ∧ x^2 - 2 * y^2 = 1 :=
by
  -- proof here
  sorry

-- (b)
theorem part_b : ∃ u v : ℤ, (3 + 2 * Real.sqrt 2)^2 = u + v * Real.sqrt 2 ∧ u^2 - 2 * v^2 = 1 :=
by
  -- proof here
  sorry

-- (c)
theorem part_c : ∀ a b c d : ℤ, a^2 - 2 * b^2 = 1 → (a + b * Real.sqrt 2) * (3 + 2 * Real.sqrt 2) = c + d * Real.sqrt 2
                  → c^2 - 2 * d^2 = 1 :=
by
  -- proof here
  sorry

-- (d)
theorem part_d : ∃ x y : ℤ, y > 100 ∧ x^2 - 2 * y^2 = 1 :=
by
  -- proof here
  sorry

end part_a_part_b_part_c_part_d_l2343_234371


namespace sequence_form_l2343_234372

theorem sequence_form (c : ℕ) (a : ℕ → ℕ) :
  (∀ n : ℕ, 0 < n →
    (∃! i : ℕ, 0 < i ∧ a i ≤ a (n + 1) + c)) ↔
  (∀ n : ℕ, 0 < n → a n = n + (c + 1)) :=
by
  sorry

end sequence_form_l2343_234372


namespace z_in_second_quadrant_l2343_234385

def is_second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

theorem z_in_second_quadrant (z : ℂ) (i : ℂ) (hi : i^2 = -1) (h : z * (1 + i^3) = i) : 
  is_second_quadrant z := by
  sorry

end z_in_second_quadrant_l2343_234385


namespace other_root_of_quadratic_l2343_234397

theorem other_root_of_quadratic (k : ℝ) :
  (∃ x : ℝ, 3 * x^2 + k * x - 5 = 0 ∧ x = 3) →
  ∃ r : ℝ, 3 * r * 3 = -5 / 3 ∧ r = -5 / 9 :=
by
  sorry

end other_root_of_quadratic_l2343_234397


namespace jerry_pick_up_trays_l2343_234324

theorem jerry_pick_up_trays : 
  ∀ (trays_per_trip trips trays_from_second total),
  trays_per_trip = 8 →
  trips = 2 →
  trays_from_second = 7 →
  total = (trays_per_trip * trips) →
  (total - trays_from_second) = 9 :=
by
  intros trays_per_trip trips trays_from_second total
  intro h1 h2 h3 h4
  sorry

end jerry_pick_up_trays_l2343_234324


namespace min_value_on_top_layer_l2343_234367

-- Definitions reflecting conditions
def bottom_layer : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def block_value (layer : List ℕ) (i : ℕ) : ℕ :=
  layer.getD (i-1) 0 -- assuming 1-based indexing

def second_layer_values : List ℕ :=
  [block_value bottom_layer 1 + block_value bottom_layer 2 + block_value bottom_layer 3,
   block_value bottom_layer 2 + block_value bottom_layer 3 + block_value bottom_layer 4,
   block_value bottom_layer 4 + block_value bottom_layer 5 + block_value bottom_layer 6,
   block_value bottom_layer 5 + block_value bottom_layer 6 + block_value bottom_layer 7,
   block_value bottom_layer 7 + block_value bottom_layer 8 + block_value bottom_layer 9,
   block_value bottom_layer 8 + block_value bottom_layer 9 + block_value bottom_layer 10]

def third_layer_values : List ℕ :=
  [second_layer_values.getD 0 0 + second_layer_values.getD 1 0 + second_layer_values.getD 2 0,
   second_layer_values.getD 1 0 + second_layer_values.getD 2 0 + second_layer_values.getD 3 0,
   second_layer_values.getD 3 0 + second_layer_values.getD 4 0 + second_layer_values.getD 5 0]

def top_layer_value : ℕ :=
  third_layer_values.getD 0 0 + third_layer_values.getD 1 0 + third_layer_values.getD 2 0

theorem min_value_on_top_layer : top_layer_value = 114 :=
by
  have h0 := block_value bottom_layer 1 -- intentionally leaving this incomplete as we're skipping the actual proof
  sorry

end min_value_on_top_layer_l2343_234367


namespace solve_rational_eq_l2343_234353

theorem solve_rational_eq (x : ℝ) :
  (1 / (x^2 + 14*x - 36)) + (1 / (x^2 + 5*x - 14)) + (1 / (x^2 - 16*x - 36)) = 0 ↔ 
  x = 9 ∨ x = -4 ∨ x = 12 ∨ x = 3 :=
sorry

end solve_rational_eq_l2343_234353


namespace condition_sufficiency_l2343_234338

theorem condition_sufficiency (x : ℝ) :
  (2 ≤ x ∧ x ≤ 3) → (x < -3 ∨ x ≥ 1) ∧ (∃ x : ℝ, (x < -3 ∨ x ≥ 1) ∧ ¬(2 ≤ x ∧ x ≤ 3)) :=
by
  sorry

end condition_sufficiency_l2343_234338


namespace find_diameter_l2343_234355

noncomputable def cost_per_meter : ℝ := 2
noncomputable def total_cost : ℝ := 188.49555921538757
noncomputable def circumference (c : ℝ) (p : ℝ) : ℝ := c / p
noncomputable def diameter (c : ℝ) : ℝ := c / Real.pi

theorem find_diameter :
  diameter (circumference total_cost cost_per_meter) = 30 := by
  sorry

end find_diameter_l2343_234355


namespace number_less_than_neg_two_l2343_234347

theorem number_less_than_neg_two : ∃ x : Int, x = -2 - 1 := 
by
  use -3
  sorry

end number_less_than_neg_two_l2343_234347


namespace archer_expected_hits_l2343_234311

noncomputable def binomial_expected_value (n : ℕ) (p : ℝ) : ℝ :=
  n * p

theorem archer_expected_hits :
  binomial_expected_value 10 0.9 = 9 :=
by
  sorry

end archer_expected_hits_l2343_234311


namespace solution_set_of_inequality_l2343_234346

variable (a b x : ℝ)
variable (h1 : a < 0)

theorem solution_set_of_inequality (h : a * x + b < 0) : x > -b / a :=
sorry

end solution_set_of_inequality_l2343_234346


namespace simplest_square_root_l2343_234394

theorem simplest_square_root :
  let a := Real.sqrt (1 / 2)
  let b := Real.sqrt 11
  let c := Real.sqrt 27
  let d := Real.sqrt 0.3
  (b < a ∧ b < c ∧ b < d) :=
sorry

end simplest_square_root_l2343_234394


namespace range_of_a_l2343_234337

noncomputable def y (x : ℝ) (a : ℝ) : ℝ := Real.log x + a * x ^ 2 - 2 * x

noncomputable def y' (x : ℝ) (a : ℝ) : ℝ := 1 / x + 2 * a * x - 2

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 < x → y' x a ≥ 0) ↔ a ≥ 1 / 2 :=
by
  sorry

end range_of_a_l2343_234337


namespace find_x_converges_to_l2343_234377

noncomputable def series_sum (x : ℝ) : ℝ := ∑' n : ℕ, (4 * (n + 1) - 2) * x^n

theorem find_x_converges_to (x : ℝ) (h : |x| < 1) :
  series_sum x = 60 → x = 29 / 30 :=
by
  sorry

end find_x_converges_to_l2343_234377


namespace find_s_when_t_is_64_l2343_234319

theorem find_s_when_t_is_64 (s : ℝ) (t : ℝ) (h1 : t = 8 * s^3) (h2 : t = 64) : s = 2 :=
by
  -- Proof will be written here
  sorry

end find_s_when_t_is_64_l2343_234319


namespace number_of_possible_measures_l2343_234379

theorem number_of_possible_measures (A B : ℕ) (h1 : A > 0) (h2 : B > 0) (h3 : A + B = 180) (h4 : ∃ k : ℕ, k ≥ 1 ∧ A = k * B) : 
  ∃ n : ℕ, n = 17 :=
sorry

end number_of_possible_measures_l2343_234379


namespace sum_of_consecutive_numbers_with_lcm_168_l2343_234322

theorem sum_of_consecutive_numbers_with_lcm_168 (a b c : ℕ) (h1 : b = a + 1) (h2 : c = b + 1) (h3 : Nat.lcm a (Nat.lcm b c) = 168) : a + b + c = 21 :=
sorry

end sum_of_consecutive_numbers_with_lcm_168_l2343_234322


namespace obtuse_triangle_k_values_l2343_234349

theorem obtuse_triangle_k_values (k : ℕ) (h : k > 0) :
  (∃ k, (5 < k ∧ k ≤ 12) ∨ (21 ≤ k ∧ k < 29)) → ∃ n : ℕ, n = 15 :=
by
  sorry

end obtuse_triangle_k_values_l2343_234349


namespace value_of_expression_l2343_234348

theorem value_of_expression (m n : ℤ) (hm : |m| = 3) (hn : |n| = 2) (hmn : m < n) :
  m^2 + m * n + n^2 = 7 ∨ m^2 + m * n + n^2 = 19 := by
  sorry

end value_of_expression_l2343_234348
