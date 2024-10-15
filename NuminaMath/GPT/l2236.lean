import Mathlib

namespace NUMINAMATH_GPT_multiples_of_7_between_20_and_150_l2236_223692

def number_of_multiples_of_7_between (a b : ℕ) : ℕ :=
  (b / 7) - (a / 7) + (if a % 7 = 0 then 1 else 0)

theorem multiples_of_7_between_20_and_150 : number_of_multiples_of_7_between 21 147 = 19 := by
  sorry

end NUMINAMATH_GPT_multiples_of_7_between_20_and_150_l2236_223692


namespace NUMINAMATH_GPT_students_absent_afternoon_l2236_223624

theorem students_absent_afternoon
  (morning_registered afternoon_registered total_students morning_absent : ℕ)
  (h_morning_registered : morning_registered = 25)
  (h_morning_absent : morning_absent = 3)
  (h_afternoon_registered : afternoon_registered = 24)
  (h_total_students : total_students = 42) :
  (afternoon_registered - (total_students - (morning_registered - morning_absent))) = 4 :=
by
  sorry

end NUMINAMATH_GPT_students_absent_afternoon_l2236_223624


namespace NUMINAMATH_GPT_find_m_l2236_223655

-- Definitions
variable {A B C O H : Type}
variable {O_is_circumcenter : is_circumcenter O A B C}
variable {H_is_altitude_intersection : is_altitude_intersection H A B C}
variable (AH BH CH OA OB OC : ℝ)

-- Problem Statement
theorem find_m (h : AH * BH * CH = m * (OA * OB * OC)) : m = 1 :=
sorry

end NUMINAMATH_GPT_find_m_l2236_223655


namespace NUMINAMATH_GPT_r_daily_earning_l2236_223644

-- Definitions from conditions in the problem
def earnings_of_all (P Q R : ℕ) : Prop := 9 * (P + Q + R) = 1620
def earnings_p_and_r (P R : ℕ) : Prop := 5 * (P + R) = 600
def earnings_q_and_r (Q R : ℕ) : Prop := 7 * (Q + R) = 910

-- Theorem to prove the daily earnings of r
theorem r_daily_earning (P Q R : ℕ) 
    (h1 : earnings_of_all P Q R)
    (h2 : earnings_p_and_r P R)
    (h3 : earnings_q_and_r Q R) : 
    R = 70 := 
by 
  sorry

end NUMINAMATH_GPT_r_daily_earning_l2236_223644


namespace NUMINAMATH_GPT_little_john_gave_to_each_friend_l2236_223651

noncomputable def little_john_total : ℝ := 10.50
noncomputable def sweets : ℝ := 2.25
noncomputable def remaining : ℝ := 3.85

theorem little_john_gave_to_each_friend :
  (little_john_total - sweets - remaining) / 2 = 2.20 :=
by
  sorry

end NUMINAMATH_GPT_little_john_gave_to_each_friend_l2236_223651


namespace NUMINAMATH_GPT_find_f2_l2236_223613

noncomputable def f : ℝ → ℝ := sorry

axiom function_property : ∀ (x : ℝ), f (2^x) + x * f (2^(-x)) = 1

theorem find_f2 : f 2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_find_f2_l2236_223613


namespace NUMINAMATH_GPT_maximum_xyz_l2236_223693

theorem maximum_xyz (x y z : ℝ) (hx : x > 1) (hy : y > 1) (hz : z > 1) 
  (h: x ^ (Real.log x / Real.log y) * y ^ (Real.log y / Real.log z) * z ^ (Real.log z / Real.log x) = 10) : 
  x * y * z ≤ 10 := 
sorry

end NUMINAMATH_GPT_maximum_xyz_l2236_223693


namespace NUMINAMATH_GPT_sum_of_powers_of_2_and_mersenne_primes_is_sum_of_squares_l2236_223694

theorem sum_of_powers_of_2_and_mersenne_primes_is_sum_of_squares 
  (n : ℕ)
  (a b c d : ℕ) 
  (h1 : n = 2^a + 2^b) 
  (h2 : a ≠ b) 
  (h3 : n = (2^c - 1) + (2^d - 1)) 
  (h4 : c ≠ d)
  (h5 : Nat.Prime (2^c - 1)) 
  (h6 : Nat.Prime (2^d - 1)) : 
  ∃ x y : ℕ, x ≠ y ∧ n = x^2 + y^2 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_powers_of_2_and_mersenne_primes_is_sum_of_squares_l2236_223694


namespace NUMINAMATH_GPT_lemonade_price_fraction_l2236_223687

theorem lemonade_price_fraction :
  (2 / 5) * (L / S) = 0.35714285714285715 → L / S = 0.8928571428571429 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_lemonade_price_fraction_l2236_223687


namespace NUMINAMATH_GPT_car_travel_distance_l2236_223640

-- Definitions based on the problem
def arith_seq_sum (a1 d : ℤ) (n : ℕ) : ℤ :=
  n * a1 + d * (n * (n - 1)) / 2

-- Main statement to prove
theorem car_travel_distance : arith_seq_sum 40 (-12) 5 = 88 :=
by sorry

end NUMINAMATH_GPT_car_travel_distance_l2236_223640


namespace NUMINAMATH_GPT_no_natural_n_divisible_by_2019_l2236_223607

theorem no_natural_n_divisible_by_2019 :
  ∀ n : ℕ, ¬ 2019 ∣ (n^2 + n + 2) :=
by sorry

end NUMINAMATH_GPT_no_natural_n_divisible_by_2019_l2236_223607


namespace NUMINAMATH_GPT_prove_incorrect_statement_l2236_223615

-- Definitions based on given conditions
def isIrrational (x : ℝ) : Prop := ¬ ∃ a b : ℚ, x = a / b ∧ b ≠ 0
def isSquareRoot (x y : ℝ) : Prop := x * x = y
def hasSquareRoot (x : ℝ) : Prop := ∃ y : ℝ, isSquareRoot y x

-- Options translated into Lean
def optionA : Prop := ∀ x : ℝ, isIrrational x → ¬ hasSquareRoot x
def optionB (x : ℝ) : Prop := 0 < x → ∃ y : ℝ, y * y = x ∧ (-y) * (-y) = x
def optionC : Prop := isSquareRoot 0 0
def optionD (a : ℝ) : Prop := ∀ x : ℝ, x = -a → (x ^ 3 = - (a ^ 3))

-- The incorrect statement according to the solution
def incorrectStatement : Prop := optionA

-- The theorem to be proven
theorem prove_incorrect_statement : incorrectStatement :=
by
  -- Replace with the actual proof, currently a placeholder using sorry
  sorry

end NUMINAMATH_GPT_prove_incorrect_statement_l2236_223615


namespace NUMINAMATH_GPT_sum_of_x_y_l2236_223690

theorem sum_of_x_y (x y : ℝ) (h : (x + y + 2) * (x + y - 1) = 0) : x + y = -2 ∨ x + y = 1 :=
by sorry

end NUMINAMATH_GPT_sum_of_x_y_l2236_223690


namespace NUMINAMATH_GPT_exists_sum_of_three_l2236_223652

theorem exists_sum_of_three {a b c d : ℕ} 
  (h1 : Nat.Coprime a b) 
  (h2 : Nat.Coprime a c) 
  (h3 : Nat.Coprime a d)
  (h4 : Nat.Coprime b c) 
  (h5 : Nat.Coprime b d) 
  (h6 : Nat.Coprime c d) 
  (h7 : a * b + c * d = a * c - 10 * b * d) :
  ∃ x y z, (x = a ∨ x = b ∨ x = c ∨ x = d) ∧ 
           (y = a ∨ y = b ∨ y = c ∨ y = d) ∧ 
           (z = a ∨ z = b ∨ z = c ∨ z = d) ∧ 
           x ≠ y ∧ x ≠ z ∧ y ≠ z ∧ 
           (x = y + z ∨ y = x + z ∨ z = x + y) :=
by
  sorry

end NUMINAMATH_GPT_exists_sum_of_three_l2236_223652


namespace NUMINAMATH_GPT_remainder_of_polynomial_division_l2236_223662

theorem remainder_of_polynomial_division
  (x : ℝ)
  (h : 2 * x - 4 = 0) :
  (8 * x^4 - 18 * x^3 + 6 * x^2 - 4 * x + 30) % (2 * x - 4) = 30 := by
  sorry

end NUMINAMATH_GPT_remainder_of_polynomial_division_l2236_223662


namespace NUMINAMATH_GPT_B_join_months_after_A_l2236_223660

-- Definitions based on conditions
def capitalA (monthsA : ℕ) : ℕ := 3500 * monthsA
def capitalB (monthsB : ℕ) : ℕ := 9000 * monthsB

-- The condition that profit is in ratio 2:3 implies the ratio of their capitals should equal 2:3
def ratio_condition (x : ℕ) : Prop := 2 * (capitalB (12 - x)) = 3 * (capitalA 12)

-- Main theorem stating that B joined the business 5 months after A started
theorem B_join_months_after_A : ∃ x, ratio_condition x ∧ x = 5 :=
by
  use 5
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_B_join_months_after_A_l2236_223660


namespace NUMINAMATH_GPT_volleyball_tournament_l2236_223679

theorem volleyball_tournament (n m : ℕ) (h : n = m) :
  n = m := 
by
  sorry

end NUMINAMATH_GPT_volleyball_tournament_l2236_223679


namespace NUMINAMATH_GPT_rhombus_shorter_diagonal_l2236_223635

theorem rhombus_shorter_diagonal (perimeter : ℝ) (angle_ratio : ℝ) (side_length diagonal_length : ℝ)
  (h₁ : perimeter = 9.6) 
  (h₂ : angle_ratio = 1 / 2) 
  (h₃ : side_length = perimeter / 4) 
  (h₄ : diagonal_length = side_length) :
  diagonal_length = 2.4 := 
sorry

end NUMINAMATH_GPT_rhombus_shorter_diagonal_l2236_223635


namespace NUMINAMATH_GPT_simplify_A_plus_2B_value_A_plus_2B_at_a1_bneg1_l2236_223699

variable (a b : ℤ)

def A : ℤ := 3 * a^2 - 6 * a * b + b^2
def B : ℤ := -2 * a^2 + 3 * a * b - 5 * b^2

theorem simplify_A_plus_2B : 
  A a b + 2 * B a b = -a^2 - 9 * b^2 := by
  sorry

theorem value_A_plus_2B_at_a1_bneg1 : 
  let a := 1
  let b := -1
  A a b + 2 * B a b = -10 := by
  sorry

end NUMINAMATH_GPT_simplify_A_plus_2B_value_A_plus_2B_at_a1_bneg1_l2236_223699


namespace NUMINAMATH_GPT_equilateral_triangle_area_increase_l2236_223666

theorem equilateral_triangle_area_increase (A : ℝ) (k : ℝ) (s : ℝ) (s' : ℝ) (A' : ℝ) (ΔA : ℝ) :
  A = 36 * Real.sqrt 3 →
  A = (Real.sqrt 3 / 4) * s^2 →
  s' = s + 3 →
  A' = (Real.sqrt 3 / 4) * s'^2 →
  ΔA = A' - A →
  ΔA = 20.25 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_equilateral_triangle_area_increase_l2236_223666


namespace NUMINAMATH_GPT_correct_regression_equation_l2236_223643

-- Problem Statement
def negatively_correlated (x y : ℝ) : Prop := sorry -- Define negative correlation for x, y
def sample_mean_x : ℝ := 3
def sample_mean_y : ℝ := 3.5
def regression_equation (b0 b1 : ℝ) (x : ℝ) : ℝ := b0 + b1 * x

theorem correct_regression_equation 
    (H_neg_corr : negatively_correlated x y) :
    regression_equation 9.5 (-2) sample_mean_x = sample_mean_y :=
by
    -- The proof will go here, skipping with sorry
    sorry

end NUMINAMATH_GPT_correct_regression_equation_l2236_223643


namespace NUMINAMATH_GPT_fraction_B_compared_to_A_and_C_l2236_223610

theorem fraction_B_compared_to_A_and_C
    (A B C : ℕ) 
    (h1 : A = (B + C) / 3) 
    (h2 : A = B + 35) 
    (h3 : A + B + C = 1260) : 
    (∃ x : ℚ, B = x * (A + C) ∧ x = 2 / 7) :=
by
  sorry

end NUMINAMATH_GPT_fraction_B_compared_to_A_and_C_l2236_223610


namespace NUMINAMATH_GPT_solve_for_x_l2236_223669

theorem solve_for_x (x : ℝ) (y : ℝ) (z : ℝ) (h1 : y = 1) (h2 : z = 3) (h3 : x^2 * y * z - x * y * z^2 = 6) :
  x = -2 / 3 ∨ x = 3 :=
by sorry

end NUMINAMATH_GPT_solve_for_x_l2236_223669


namespace NUMINAMATH_GPT_lyndee_friends_count_l2236_223618

-- Definitions
variables (total_chicken total_garlic_bread : ℕ)
variables (lyndee_chicken lyndee_garlic_bread : ℕ)
variables (friends_large_chicken_count : ℕ)
variables (friends_large_chicken : ℕ)
variables (friend_garlic_bread_per_friend : ℕ)

def remaining_chicken (total_chicken lyndee_chicken friends_large_chicken_count friends_large_chicken : ℕ) : ℕ :=
  total_chicken - (lyndee_chicken + friends_large_chicken_count * friends_large_chicken)

def remaining_garlic_bread (total_garlic_bread lyndee_garlic_bread : ℕ) : ℕ :=
  total_garlic_bread - lyndee_garlic_bread

def total_friends (friends_large_chicken_count remaining_chicken remaining_garlic_bread friend_garlic_bread_per_friend : ℕ) : ℕ :=
  friends_large_chicken_count + remaining_chicken + remaining_garlic_bread / friend_garlic_bread_per_friend

-- Theorem statement
theorem lyndee_friends_count : 
  total_chicken = 11 → 
  total_garlic_bread = 15 →
  lyndee_chicken = 1 →
  lyndee_garlic_bread = 1 →
  friends_large_chicken_count = 3 →
  friends_large_chicken = 2 →
  friend_garlic_bread_per_friend = 3 →
  total_friends friends_large_chicken_count 
                (remaining_chicken total_chicken lyndee_chicken friends_large_chicken_count friends_large_chicken)
                (remaining_garlic_bread total_garlic_bread lyndee_garlic_bread)
                friend_garlic_bread_per_friend = 7 := 
by 
  intros h1 h2 h3 h4 h5 h6 h7
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_lyndee_friends_count_l2236_223618


namespace NUMINAMATH_GPT_product_of_dodecagon_l2236_223621

open Complex

theorem product_of_dodecagon (Q : Fin 12 → ℂ) (h₁ : Q 0 = 2) (h₇ : Q 6 = 8) :
  (Q 0) * (Q 1) * (Q 2) * (Q 3) * (Q 4) * (Q 5) * (Q 6) * (Q 7) * (Q 8) * (Q 9) * (Q 10) * (Q 11) = 244140624 :=
sorry

end NUMINAMATH_GPT_product_of_dodecagon_l2236_223621


namespace NUMINAMATH_GPT_polynomial_value_l2236_223691

theorem polynomial_value (x : ℝ) (h : 3 * x^2 - x = 1) : 6 * x^3 + 7 * x^2 - 5 * x + 2008 = 2011 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_value_l2236_223691


namespace NUMINAMATH_GPT_qatar_location_is_accurate_l2236_223682

def qatar_geo_location :=
  "The most accurate representation of Qatar's geographical location is latitude 25 degrees North, longitude 51 degrees East."

theorem qatar_location_is_accurate :
  qatar_geo_location = "The most accurate representation of Qatar's geographical location is latitude 25 degrees North, longitude 51 degrees East." :=
sorry

end NUMINAMATH_GPT_qatar_location_is_accurate_l2236_223682


namespace NUMINAMATH_GPT_toby_breakfast_calories_l2236_223611

noncomputable def calories_bread := 100
noncomputable def calories_peanut_butter_per_serving := 200
noncomputable def servings_peanut_butter := 2

theorem toby_breakfast_calories :
  1 * calories_bread + servings_peanut_butter * calories_peanut_butter_per_serving = 500 :=
by
  sorry

end NUMINAMATH_GPT_toby_breakfast_calories_l2236_223611


namespace NUMINAMATH_GPT_time_to_finish_task_l2236_223681

-- Define the conditions
def printerA_rate (total_pages : ℕ) (time_A_alone : ℕ) : ℚ := total_pages / time_A_alone
def printerB_rate (rate_A : ℚ) : ℚ := rate_A + 10

-- Define the combined rate of printers working together
def combined_rate (rate_A : ℚ) (rate_B : ℚ) : ℚ := rate_A + rate_B

-- Define the time taken to finish the task together
def time_to_finish (total_pages : ℕ) (combined_rate : ℚ) : ℚ := total_pages / combined_rate

-- Given conditions
def total_pages : ℕ := 35
def time_A_alone : ℕ := 60

-- Definitions derived from given conditions
def rate_A : ℚ := printerA_rate total_pages time_A_alone
def rate_B : ℚ := printerB_rate rate_A

-- Combined rate when both printers work together
def combined_rate_AB : ℚ := combined_rate rate_A rate_B

-- Lean theorem statement to prove time taken by both printers
theorem time_to_finish_task : time_to_finish total_pages combined_rate_AB = 210 / 67 := 
by
  sorry

end NUMINAMATH_GPT_time_to_finish_task_l2236_223681


namespace NUMINAMATH_GPT_numWaysToChoosePairs_is_15_l2236_223604

def numWaysToChoosePairs : ℕ :=
  let white := Nat.choose 5 2
  let brown := Nat.choose 3 2
  let blue := Nat.choose 2 2
  let black := Nat.choose 2 2
  white + brown + blue + black

theorem numWaysToChoosePairs_is_15 : numWaysToChoosePairs = 15 := by
  -- We will prove this theorem in actual proof
  sorry

end NUMINAMATH_GPT_numWaysToChoosePairs_is_15_l2236_223604


namespace NUMINAMATH_GPT_inequality_with_equality_condition_l2236_223676

variables {a b c d : ℝ}

theorem inequality_with_equality_condition (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) 
    (habcd : a + b + c + d = 1) : 
    (a^2 / (a + b) + b^2 / (b + c) + c^2 / (c + d) + d^2 / (d + a) >= 1 / 2) ∧ 
    (a^2 / (a + b) + b^2 / (b + c) + c^2 / (c + d) + d^2 / (d + a) = 1 / 2 ↔ a = b ∧ b = c ∧ c = d) := 
sorry

end NUMINAMATH_GPT_inequality_with_equality_condition_l2236_223676


namespace NUMINAMATH_GPT_chimney_height_theorem_l2236_223627

noncomputable def chimney_height :=
  let BCD := 75 * Real.pi / 180
  let BDC := 60 * Real.pi / 180
  let CBD := 45 * Real.pi / 180
  let CD := 40
  let BC := CD * Real.sin BDC / Real.sin CBD
  let CE := 1
  let elevation := 30 * Real.pi / 180
  let AB := CE + (Real.tan elevation * BC)
  AB

theorem chimney_height_theorem : chimney_height = 1 + 20 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_chimney_height_theorem_l2236_223627


namespace NUMINAMATH_GPT_problem_a_b_c_l2236_223661

theorem problem_a_b_c (a b c : ℝ) (h1 : a < b) (h2 : b < c) (h3 : ab + bc + ac = 0) (h4 : abc = 1) : |a + b| > |c| := 
by sorry

end NUMINAMATH_GPT_problem_a_b_c_l2236_223661


namespace NUMINAMATH_GPT_percentage_increase_visitors_l2236_223634

theorem percentage_increase_visitors 
  (V_Oct : ℕ)
  (V_Nov V_Dec : ℕ)
  (h1 : V_Oct = 100)
  (h2 : V_Dec = V_Nov + 15)
  (h3 : V_Oct + V_Nov + V_Dec = 345) : 
  (V_Nov - V_Oct) * 100 / V_Oct = 15 := 
by 
  sorry

end NUMINAMATH_GPT_percentage_increase_visitors_l2236_223634


namespace NUMINAMATH_GPT_probability_incorrect_pairs_l2236_223670

theorem probability_incorrect_pairs 
  (k : ℕ) (h_k : k < 6)
  : let m := 7
    let n := 72
    m + n = 79 :=
by
  sorry

end NUMINAMATH_GPT_probability_incorrect_pairs_l2236_223670


namespace NUMINAMATH_GPT_maximize_profit_l2236_223602

noncomputable def R (x : ℝ) : ℝ := 
  if x ≤ 40 then
    40 * x - (1 / 2) * x^2
  else
    1500 - 25000 / x

noncomputable def cost (x : ℝ) : ℝ := 2 + 0.1 * x

noncomputable def f (x : ℝ) : ℝ := R x - cost x

theorem maximize_profit :
  ∃ x : ℝ, x = 50 ∧ f 50 = 300 := by
  sorry

end NUMINAMATH_GPT_maximize_profit_l2236_223602


namespace NUMINAMATH_GPT_MissyTotalTVTime_l2236_223650

theorem MissyTotalTVTime :
  let reality_shows := [28, 35, 42, 39, 29]
  let cartoons := [10, 10]
  let ad_breaks := [8, 6, 12]
  let total_time := reality_shows.sum + cartoons.sum + ad_breaks.sum
  total_time = 219 := by
{
  -- Lean proof logic goes here (proof not requested)
  sorry
}

end NUMINAMATH_GPT_MissyTotalTVTime_l2236_223650


namespace NUMINAMATH_GPT_root_in_interval_l2236_223642

noncomputable def f (x : ℝ) : ℝ := x^3 + 2 * x - 1

theorem root_in_interval (k : ℤ) (h : ∃ x : ℝ, k < x ∧ x < k + 1 ∧ f x = 0) : k = 0 :=
by
  sorry

end NUMINAMATH_GPT_root_in_interval_l2236_223642


namespace NUMINAMATH_GPT_asian_games_tourists_scientific_notation_l2236_223619

theorem asian_games_tourists_scientific_notation : 
  ∀ (n : ℕ), n = 18480000 → 1.848 * (10:ℝ) ^ 7 = (n : ℝ) :=
by
  intro n
  sorry

end NUMINAMATH_GPT_asian_games_tourists_scientific_notation_l2236_223619


namespace NUMINAMATH_GPT_andy_starting_problem_l2236_223680

theorem andy_starting_problem (end_num problems_solved : ℕ) 
  (h_end : end_num = 125) (h_solved : problems_solved = 46) : 
  end_num - problems_solved + 1 = 80 := 
by
  sorry

end NUMINAMATH_GPT_andy_starting_problem_l2236_223680


namespace NUMINAMATH_GPT_beaver_hid_36_carrots_l2236_223631

variable (x y : ℕ)

-- Conditions
def beaverCarrots := 4 * x
def bunnyCarrots := 6 * y

-- Given that both animals hid the same total number of carrots
def totalCarrotsEqual := beaverCarrots x = bunnyCarrots y

-- Bunny used 3 fewer burrows than the beaver
def bunnyBurrows := y = x - 3

-- The goal is to show the beaver hid 36 carrots
theorem beaver_hid_36_carrots (H1 : totalCarrotsEqual x y) (H2 : bunnyBurrows x y) : beaverCarrots x = 36 := by
  sorry

end NUMINAMATH_GPT_beaver_hid_36_carrots_l2236_223631


namespace NUMINAMATH_GPT_last_fish_in_swamp_l2236_223686

noncomputable def final_fish (perches pikes sudaks : ℕ) : String :=
  let p := perches
  let pi := pikes
  let s := sudaks
  if p = 6 ∧ pi = 7 ∧ s = 8 then "Sudak" else "Unknown"

theorem last_fish_in_swamp : final_fish 6 7 8 = "Sudak" := by
  sorry

end NUMINAMATH_GPT_last_fish_in_swamp_l2236_223686


namespace NUMINAMATH_GPT_reading_speed_increase_factor_l2236_223606

-- Define Tom's normal reading speed as a constant rate
def tom_normal_speed := 12 -- pages per hour

-- Define the time period
def hours := 2 -- hours

-- Define the number of pages read in the given time period
def pages_read := 72 -- pages

-- Calculate the expected pages read at normal speed in the given time
def expected_pages := tom_normal_speed * hours -- should be 24 pages

-- Define the calculated factor by which the reading speed has increased
def expected_factor := pages_read / expected_pages -- should be 3

-- Prove that the factor is indeed 3
theorem reading_speed_increase_factor :
  expected_factor = 3 := by
  sorry

end NUMINAMATH_GPT_reading_speed_increase_factor_l2236_223606


namespace NUMINAMATH_GPT_youtube_dislikes_calculation_l2236_223639

theorem youtube_dislikes_calculation :
  ∀ (l d_initial d_final : ℕ),
    l = 3000 →
    d_initial = (l / 2) + 100 →
    d_final = d_initial + 1000 →
    d_final = 2600 :=
by
  intros l d_initial d_final h_l h_d_initial h_d_final
  sorry

end NUMINAMATH_GPT_youtube_dislikes_calculation_l2236_223639


namespace NUMINAMATH_GPT_notebook_cost_l2236_223658

theorem notebook_cost {s n c : ℕ}
  (h1 : s > 18)
  (h2 : c > n)
  (h3 : s * n * c = 2275) :
  c = 13 :=
sorry

end NUMINAMATH_GPT_notebook_cost_l2236_223658


namespace NUMINAMATH_GPT_solve_for_a_l2236_223668

theorem solve_for_a (a x : ℝ) (h : x = 1 ∧ 2 * a * x - 2 = a + 3) : a = 5 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_a_l2236_223668


namespace NUMINAMATH_GPT_equivalent_math_problem_l2236_223626

noncomputable def P : ℝ := Real.sqrt 1011 + Real.sqrt 1012
noncomputable def Q : ℝ := - (Real.sqrt 1011 + Real.sqrt 1012)
noncomputable def R : ℝ := Real.sqrt 1011 - Real.sqrt 1012
noncomputable def S : ℝ := Real.sqrt 1012 - Real.sqrt 1011

theorem equivalent_math_problem :
  (P * Q)^2 * R * S = 8136957 :=
by
  sorry

end NUMINAMATH_GPT_equivalent_math_problem_l2236_223626


namespace NUMINAMATH_GPT_ice_cream_total_volume_l2236_223678

/-- 
  The interior of a right, circular cone is 12 inches tall with a 3-inch radius at the opening.
  The interior of the cone is filled with ice cream.
  The cone has a hemisphere of ice cream exactly covering the opening of the cone.
  On top of this hemisphere, there is a cylindrical layer of ice cream of height 2 inches 
  and the same radius as the hemisphere (3 inches).
  Prove that the total volume of ice cream is 72π cubic inches.
-/
theorem ice_cream_total_volume :
  let r := 3
  let h_cone := 12
  let h_cylinder := 2
  let V_cone := 1/3 * Real.pi * r^2 * h_cone
  let V_hemisphere := 2/3 * Real.pi * r^3
  let V_cylinder := Real.pi * r^2 * h_cylinder
  V_cone + V_hemisphere + V_cylinder = 72 * Real.pi :=
by {
  let r := 3
  let h_cone := 12
  let h_cylinder := 2
  let V_cone := 1/3 * Real.pi * r^2 * h_cone
  let V_hemisphere := 2/3 * Real.pi * r^3
  let V_cylinder := Real.pi * r^2 * h_cylinder
  sorry
}

end NUMINAMATH_GPT_ice_cream_total_volume_l2236_223678


namespace NUMINAMATH_GPT_pizza_slices_all_toppings_l2236_223685

theorem pizza_slices_all_toppings (x : ℕ) :
  (16 = (8 - x) + (12 - x) + (6 - x) + x) → x = 5 := by
  sorry

end NUMINAMATH_GPT_pizza_slices_all_toppings_l2236_223685


namespace NUMINAMATH_GPT_larger_page_number_l2236_223698

theorem larger_page_number (x : ℕ) (h1 : (x + (x + 1) = 125)) : (x + 1 = 63) :=
by
  sorry

end NUMINAMATH_GPT_larger_page_number_l2236_223698


namespace NUMINAMATH_GPT_gcd_of_repeated_three_digit_numbers_l2236_223608

theorem gcd_of_repeated_three_digit_numbers :
  ∀ n : ℕ, 100 ≤ n ∧ n ≤ 999 → Int.gcd 1001001 n = 1001001 :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_gcd_of_repeated_three_digit_numbers_l2236_223608


namespace NUMINAMATH_GPT_simplify_expression_l2236_223653

theorem simplify_expression : (-5) - (-4) + (-7) - (2) = -5 + 4 - 7 - 2 := 
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l2236_223653


namespace NUMINAMATH_GPT_pairs_sold_l2236_223638

-- Define the given conditions
def initial_large_pairs : ℕ := 22
def initial_medium_pairs : ℕ := 50
def initial_small_pairs : ℕ := 24
def pairs_left : ℕ := 13

-- Translate to the equivalent proof problem
theorem pairs_sold : (initial_large_pairs + initial_medium_pairs + initial_small_pairs) - pairs_left = 83 := by
  sorry

end NUMINAMATH_GPT_pairs_sold_l2236_223638


namespace NUMINAMATH_GPT_hyperbola_range_m_l2236_223609

theorem hyperbola_range_m (m : ℝ) : 
  (∃ x y : ℝ, (x^2 / (16 - m)) + (y^2 / (9 - m)) = 1) → 9 < m ∧ m < 16 :=
by 
  sorry

end NUMINAMATH_GPT_hyperbola_range_m_l2236_223609


namespace NUMINAMATH_GPT_circle_radius_l2236_223633

theorem circle_radius {C : ℝ → ℝ → Prop} (h1 : C 4 0) (h2 : C (-4) 0) : ∃ r : ℝ, r = 4 :=
by
  -- sorry for brevity
  sorry

end NUMINAMATH_GPT_circle_radius_l2236_223633


namespace NUMINAMATH_GPT_hyperbola_eccentricity_range_l2236_223683

theorem hyperbola_eccentricity_range (a b e : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_upper : b / a < 2) :
  e = Real.sqrt (1 + (b / a) ^ 2) → 1 < e ∧ e < Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_range_l2236_223683


namespace NUMINAMATH_GPT_angle_triple_complement_l2236_223617

theorem angle_triple_complement (x : ℝ) (h1 : x + (180 - x) = 180) (h2 : x = 3 * (180 - x)) : x = 135 := 
by
  sorry

end NUMINAMATH_GPT_angle_triple_complement_l2236_223617


namespace NUMINAMATH_GPT_second_largest_consecutive_odd_195_l2236_223657

theorem second_largest_consecutive_odd_195 :
  ∃ x : Int, (x - 4) + (x - 2) + x + (x + 2) + (x + 4) = 195 ∧ (x + 2) = 41 := by
  sorry

end NUMINAMATH_GPT_second_largest_consecutive_odd_195_l2236_223657


namespace NUMINAMATH_GPT_margin_in_terms_of_selling_price_l2236_223649

variable (C S n M : ℝ)

theorem margin_in_terms_of_selling_price (h : M = (2 * C) / n) : M = (2 * S) / (n + 2) :=
sorry

end NUMINAMATH_GPT_margin_in_terms_of_selling_price_l2236_223649


namespace NUMINAMATH_GPT_jake_third_test_score_l2236_223664

theorem jake_third_test_score
  (avg_score_eq_75 : (80 + 90 + third_score + third_score) / 4 = 75)
  (second_score : ℕ := 80 + 10) :
  third_score = 65 :=
by
  sorry

end NUMINAMATH_GPT_jake_third_test_score_l2236_223664


namespace NUMINAMATH_GPT_domain_f1_correct_f2_correct_f2_at_3_l2236_223614

noncomputable def f1 (x : ℝ) : ℝ := Real.sqrt (4 - 2 * x) + 1 + 1 / (x + 1)

noncomputable def domain_f1 : Set ℝ := {x | 4 - 2 * x ≥ 0} \ (insert 1 (insert (-1) {}))

theorem domain_f1_correct : domain_f1 = { x | x ≤ 2 ∧ x ≠ 1 ∧ x ≠ -1 } :=
by
  sorry

noncomputable def f2 (x : ℝ) : ℝ := x^2 - 4 * x + 3

theorem f2_correct : ∀ x, f2 (x + 1) = x^2 - 2 * x :=
by
  sorry

theorem f2_at_3 : f2 3 = 0 :=
by
  sorry

end NUMINAMATH_GPT_domain_f1_correct_f2_correct_f2_at_3_l2236_223614


namespace NUMINAMATH_GPT_opposite_of_neg_abs_opposite_of_neg_abs_correct_l2236_223689

theorem opposite_of_neg_abs (x : ℚ) (hx : |x| = 2 / 5) : -|x| = - (2 / 5) := sorry

theorem opposite_of_neg_abs_correct (x : ℚ) (hx : |x| = 2 / 5) : - -|x| = 2 / 5 := by
  rw [opposite_of_neg_abs x hx]
  simp

end NUMINAMATH_GPT_opposite_of_neg_abs_opposite_of_neg_abs_correct_l2236_223689


namespace NUMINAMATH_GPT_problem_1_problem_2_l2236_223672

theorem problem_1 :
  83 * 87 = 100 * 8 * (8 + 1) + 21 :=
by sorry

theorem problem_2 (n : ℕ) :
  (10 * n + 3) * (10 * n + 7) = 100 * n * (n + 1) + 21 :=
by sorry

end NUMINAMATH_GPT_problem_1_problem_2_l2236_223672


namespace NUMINAMATH_GPT_combinations_of_three_toppings_l2236_223616

def number_of_combinations (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem combinations_of_three_toppings : number_of_combinations 10 3 = 120 := by
  sorry

end NUMINAMATH_GPT_combinations_of_three_toppings_l2236_223616


namespace NUMINAMATH_GPT_opposite_of_neg_2_is_2_l2236_223623

theorem opposite_of_neg_2_is_2 : -(-2) = 2 := 
by
  sorry

end NUMINAMATH_GPT_opposite_of_neg_2_is_2_l2236_223623


namespace NUMINAMATH_GPT_remainder_of_67_pow_67_plus_67_mod_68_l2236_223695

theorem remainder_of_67_pow_67_plus_67_mod_68 : (67^67 + 67) % 68 = 66 := by
  sorry

end NUMINAMATH_GPT_remainder_of_67_pow_67_plus_67_mod_68_l2236_223695


namespace NUMINAMATH_GPT_prove_f_cos_eq_l2236_223675

variable (f : ℝ → ℝ)

theorem prove_f_cos_eq :
  (∀ x : ℝ, f (Real.sin x) = 3 - Real.cos (2 * x)) →
  (∀ x : ℝ, f (Real.cos x) = 3 + Real.cos (2 * x)) :=
by
  sorry

end NUMINAMATH_GPT_prove_f_cos_eq_l2236_223675


namespace NUMINAMATH_GPT_plane_equation_correct_l2236_223612

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def vectorBC (B C : Point3D) : Point3D :=
  { x := C.x - B.x, y := C.y - B.y, z := C.z - B.z }

def planeEquation (n : Point3D) (A : Point3D) (P : Point3D) : ℝ :=
  n.x * (P.x - A.x) + n.y * (P.y - A.y) + n.z * (P.z - A.z)

theorem plane_equation_correct :
  let A := ⟨3, -3, -6⟩
  let B := ⟨1, 9, -5⟩
  let C := ⟨6, 6, -4⟩
  let n := vectorBC B C
  ∀ P, planeEquation n A P = 0 ↔ 5 * (P.x - A.x) - 3 * (P.y - A.y) + 1 * (P.z - A.z) - 18 = 0 :=
by
  sorry

end NUMINAMATH_GPT_plane_equation_correct_l2236_223612


namespace NUMINAMATH_GPT_problem1_problem2_problem3_l2236_223688

-- Problem 1
theorem problem1 (x : ℝ) : (3 * (x - 1)^2 = 12) ↔ (x = 3 ∨ x = -1) :=
by
  sorry

-- Problem 2
theorem problem2 (x : ℝ) : (3 * x^2 - 6 * x - 2 = 0) ↔ (x = (3 + Real.sqrt 15) / 3 ∨ x = (3 - Real.sqrt 15) / 3) :=
by
  sorry

-- Problem 3
theorem problem3 (x : ℝ) : (3 * x * (2 * x + 1) = 4 * x + 2) ↔ (x = -1 / 2 ∨ x = 2 / 3) :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_l2236_223688


namespace NUMINAMATH_GPT_general_term_formula_l2236_223605

-- Define the sequence as given in the conditions
def seq (n : ℕ) : ℚ := 
  match n with 
  | 0       => 1
  | 1       => 2 / 3
  | 2       => 1 / 2
  | 3       => 2 / 5
  | (n + 1) => sorry   -- This is just a placeholder, to be proved

-- State the theorem
theorem general_term_formula (n : ℕ) : seq n = 2 / (n + 1) := 
by {
  -- Proof will be provided here
  sorry
}

end NUMINAMATH_GPT_general_term_formula_l2236_223605


namespace NUMINAMATH_GPT_min_value_func_l2236_223654

noncomputable def func (x : ℝ) : ℝ := Real.sin x + Real.sqrt 3 * Real.cos x

theorem min_value_func : ∃ x : ℝ, func x = -2 :=
by
  existsi (Real.pi / 2 + Real.pi / 3)
  sorry

end NUMINAMATH_GPT_min_value_func_l2236_223654


namespace NUMINAMATH_GPT_primes_product_less_than_20_l2236_223620

-- Define the primes less than 20
def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

-- Define the product of a list of natural numbers
def product (l : List ℕ) : ℕ := l.foldr (· * ·) 1

theorem primes_product_less_than_20 :
  product primes_less_than_20 = 9699690 :=
by
  sorry

end NUMINAMATH_GPT_primes_product_less_than_20_l2236_223620


namespace NUMINAMATH_GPT_marcy_votes_correct_l2236_223663

-- Definition of variables based on the conditions
def joey_votes : ℕ := 8
def barry_votes : ℕ := 2 * (joey_votes + 3)
def marcy_votes : ℕ := 3 * barry_votes

-- The main statement to prove
theorem marcy_votes_correct : marcy_votes = 66 := 
by 
  sorry

end NUMINAMATH_GPT_marcy_votes_correct_l2236_223663


namespace NUMINAMATH_GPT_exists_distinct_ij_l2236_223696

theorem exists_distinct_ij (n : ℕ) (a : Fin n → ℤ) (h_distinct : Function.Injective a) (h_n_ge_3 : 3 ≤ n) :
  ∃ (i j : Fin n), i ≠ j ∧ (∀ k, (a i + a j) ∣ 3 * a k → False) :=
by
  sorry

end NUMINAMATH_GPT_exists_distinct_ij_l2236_223696


namespace NUMINAMATH_GPT_min_a_plus_b_l2236_223646

variable (a b : ℝ)
variable (ha_pos : a > 0)
variable (hb_pos : b > 0)
variable (h1 : a^2 - 12 * b ≥ 0)
variable (h2 : 9 * b^2 - 4 * a ≥ 0)

theorem min_a_plus_b (a b : ℝ) (ha_pos : a > 0) (hb_pos : b > 0)
  (h1 : a^2 - 12 * b ≥ 0) (h2 : 9 * b^2 - 4 * a ≥ 0) :
  a + b = 3.3442 := 
sorry

end NUMINAMATH_GPT_min_a_plus_b_l2236_223646


namespace NUMINAMATH_GPT_population_ratio_l2236_223677

-- Definitions
def population_z (Z : ℕ) : ℕ := Z
def population_y (Z : ℕ) : ℕ := 2 * population_z Z
def population_x (Z : ℕ) : ℕ := 6 * population_y Z

-- Theorem stating the ratio
theorem population_ratio (Z : ℕ) : (population_x Z) / (population_z Z) = 12 :=
  by 
  unfold population_x population_y population_z
  sorry

end NUMINAMATH_GPT_population_ratio_l2236_223677


namespace NUMINAMATH_GPT_min_x_plus_y_l2236_223684

theorem min_x_plus_y (x y : ℕ) (hxy : x ≠ y) (h : (1/x : ℝ) + 1/y = 1/24) : x + y = 98 :=
sorry

end NUMINAMATH_GPT_min_x_plus_y_l2236_223684


namespace NUMINAMATH_GPT_sculpture_and_base_total_height_l2236_223629

noncomputable def sculpture_height_ft : Nat := 2
noncomputable def sculpture_height_in : Nat := 10
noncomputable def base_height_in : Nat := 4
noncomputable def inches_per_foot : Nat := 12

theorem sculpture_and_base_total_height :
  (sculpture_height_ft * inches_per_foot + sculpture_height_in + base_height_in = 38) :=
by
  sorry

end NUMINAMATH_GPT_sculpture_and_base_total_height_l2236_223629


namespace NUMINAMATH_GPT_remainder_of_875_div_by_170_l2236_223637

theorem remainder_of_875_div_by_170 :
  ∃ r, (∀ x, x ∣ 680 ∧ x ∣ (875 - r) → x ≤ 170) ∧ 170 ∣ (875 - r) ∧ r = 25 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_875_div_by_170_l2236_223637


namespace NUMINAMATH_GPT_jihyae_initial_money_l2236_223671

variables {M : ℕ}

def spent_on_supplies (M : ℕ) := M / 2 + 200
def left_after_buying (M : ℕ) := M - spent_on_supplies M
def saved (M : ℕ) := left_after_buying M / 2 + 300
def final_leftover (M : ℕ) := left_after_buying M - saved M

theorem jihyae_initial_money : final_leftover M = 350 → M = 3000 :=
by
  sorry

end NUMINAMATH_GPT_jihyae_initial_money_l2236_223671


namespace NUMINAMATH_GPT_basketball_team_wins_l2236_223622

theorem basketball_team_wins (f : ℚ) (h1 : 40 + 40 * f + (40 + 40 * f) = 130) : f = 5 / 8 :=
by
  sorry

end NUMINAMATH_GPT_basketball_team_wins_l2236_223622


namespace NUMINAMATH_GPT_find_a_l2236_223636

theorem find_a (x : ℝ) (hx1 : 0 < x)
  (hx2 : x + 1/x ≥ 2)
  (hx3 : x + 4/x^2 ≥ 3)
  (hx4 : x + 27/x^3 ≥ 4) :
  (x + a/x^4 ≥ 5) → a = 4^4 :=
sorry

end NUMINAMATH_GPT_find_a_l2236_223636


namespace NUMINAMATH_GPT_evaluate_expression_l2236_223601

theorem evaluate_expression : (∃ (x : Real), 6 < x ∧ x < 7 ∧ x = Real.sqrt 45) → (Int.floor (Real.sqrt 45))^2 + 2*Int.floor (Real.sqrt 45) + 1 = 49 := 
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2236_223601


namespace NUMINAMATH_GPT_average_infections_l2236_223667

theorem average_infections (x : ℝ) (h : 1 + x + x^2 = 121) : x = 10 :=
sorry

end NUMINAMATH_GPT_average_infections_l2236_223667


namespace NUMINAMATH_GPT_sum_of_distances_l2236_223697

theorem sum_of_distances (AB A'B' AD A'D' x y : ℝ) 
  (h1 : AB = 8)
  (h2 : A'B' = 6)
  (h3 : AD = 3)
  (h4 : A'D' = 1)
  (h5 : x = 2)
  (h6 : x / y = 3 / 2) : 
  x + y = 10 / 3 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_distances_l2236_223697


namespace NUMINAMATH_GPT_option_A_correct_option_B_correct_option_C_correct_option_D_incorrect_l2236_223659

variable (a b : ℝ)
variable (h : a < b)

theorem option_A_correct : a + 2 < b + 2 := by
  sorry

theorem option_B_correct : 3 * a < 3 * b := by
  sorry

theorem option_C_correct : (1 / 2) * a < (1 / 2) * b := by
  sorry

theorem option_D_incorrect : ¬(-2 * a < -2 * b) := by
  sorry

end NUMINAMATH_GPT_option_A_correct_option_B_correct_option_C_correct_option_D_incorrect_l2236_223659


namespace NUMINAMATH_GPT_value_of_x_l2236_223665

theorem value_of_x (x : ℤ) : (3000 + x) ^ 2 = x ^ 2 → x = -1500 := 
by
  sorry

end NUMINAMATH_GPT_value_of_x_l2236_223665


namespace NUMINAMATH_GPT_business_transaction_loss_l2236_223625

theorem business_transaction_loss (cost_price : ℝ) (final_price : ℝ) (markup_percent : ℝ) (reduction_percent : ℝ) : 
  (final_price = 96) ∧ (markup_percent = 0.2) ∧ (reduction_percent = 0.2) ∧ (cost_price * (1 + markup_percent) * (1 - reduction_percent) = final_price) → 
  (cost_price - final_price = -4) :=
by
sorry

end NUMINAMATH_GPT_business_transaction_loss_l2236_223625


namespace NUMINAMATH_GPT_positive_difference_1010_1000_l2236_223600

-- Define the arithmetic sequence
def arithmetic_sequence (a d n : ℕ) : ℕ :=
  a + (n - 1) * d

-- Define the specific terms
def a_1000 := arithmetic_sequence 5 7 1000
def a_1010 := arithmetic_sequence 5 7 1010

-- Proof statement
theorem positive_difference_1010_1000 : a_1010 - a_1000 = 70 :=
by
  sorry

end NUMINAMATH_GPT_positive_difference_1010_1000_l2236_223600


namespace NUMINAMATH_GPT_average_chem_math_l2236_223630

theorem average_chem_math (P C M : ℕ) (h : P + C + M = P + 180) : (C + M) / 2 = 90 :=
  sorry

end NUMINAMATH_GPT_average_chem_math_l2236_223630


namespace NUMINAMATH_GPT_smallest_angle_convex_15_polygon_l2236_223674

theorem smallest_angle_convex_15_polygon :
  ∃ (a : ℕ) (d : ℕ), (∀ n : ℕ, n ∈ Finset.range 15 → (a + n * d < 180)) ∧
  15 * (a + 7 * d) = 2340 ∧ 15 * d <= 24 -> a = 135 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_smallest_angle_convex_15_polygon_l2236_223674


namespace NUMINAMATH_GPT_divisible_by_42_l2236_223647

theorem divisible_by_42 (a : ℤ) : ∃ k : ℤ, a^7 - a = 42 * k := 
sorry

end NUMINAMATH_GPT_divisible_by_42_l2236_223647


namespace NUMINAMATH_GPT_sum_of_faces_l2236_223641

variable (a d b c e f : ℕ)
variable (pos_a : a > 0) (pos_d : d > 0) (pos_b : b > 0) (pos_c : c > 0) 
variable (pos_e : e > 0) (pos_f : f > 0)
variable (h : a * b * e + a * b * f + a * c * e + a * c * f + d * b * e + d * b * f + d * c * e + d * c * f = 1176)

theorem sum_of_faces : a + d + b + c + e + f = 33 := by
  sorry

end NUMINAMATH_GPT_sum_of_faces_l2236_223641


namespace NUMINAMATH_GPT_time_to_cover_length_l2236_223673

def speed_escalator : ℝ := 10
def speed_person : ℝ := 4
def length_escalator : ℝ := 112

theorem time_to_cover_length :
  (length_escalator / (speed_escalator + speed_person) = 8) :=
by
  sorry

end NUMINAMATH_GPT_time_to_cover_length_l2236_223673


namespace NUMINAMATH_GPT_Moscow_1975_p_q_r_equal_primes_l2236_223632

theorem Moscow_1975_p_q_r_equal_primes (a b c : ℕ) (p q r : ℕ) 
  (hp : p = b^c + a) 
  (hq : q = a^b + c) 
  (hr : r = c^a + b) 
  (prime_p : Prime p) 
  (prime_q : Prime q) 
  (prime_r : Prime r) : 
  q = r :=
sorry

end NUMINAMATH_GPT_Moscow_1975_p_q_r_equal_primes_l2236_223632


namespace NUMINAMATH_GPT_sum_three_consecutive_integers_divisible_by_three_l2236_223656

theorem sum_three_consecutive_integers_divisible_by_three (a : ℕ) (h : 1 < a) :
  (a - 1) + a + (a + 1) % 3 = 0 :=
by
  sorry

end NUMINAMATH_GPT_sum_three_consecutive_integers_divisible_by_three_l2236_223656


namespace NUMINAMATH_GPT_fractional_eq_solution_range_l2236_223648

theorem fractional_eq_solution_range (x m : ℝ) (h : (2 * x - m) / (x + 1) = 1) (hx : x < 0) : 
  m < -1 ∧ m ≠ -2 := 
by 
  sorry

end NUMINAMATH_GPT_fractional_eq_solution_range_l2236_223648


namespace NUMINAMATH_GPT_isosceles_triangles_with_perimeter_21_l2236_223603

theorem isosceles_triangles_with_perimeter_21 : 
  ∃ n : ℕ, n = 5 ∧ (∀ (a b c : ℕ), a ≤ b ∧ b = c ∧ a + 2*b = 21 → 1 ≤ a ∧ a ≤ 10) :=
sorry

end NUMINAMATH_GPT_isosceles_triangles_with_perimeter_21_l2236_223603


namespace NUMINAMATH_GPT_gcd_of_459_and_357_l2236_223628

theorem gcd_of_459_and_357 : Nat.gcd 459 357 = 51 := by
  sorry

end NUMINAMATH_GPT_gcd_of_459_and_357_l2236_223628


namespace NUMINAMATH_GPT_solve_triangle_l2236_223645

noncomputable def triangle_side_lengths (a b c : ℝ) : Prop :=
  a = 10 ∧ b = 9 ∧ c = 17

theorem solve_triangle (a b c : ℝ) :
  (a ^ 2 - b ^ 2 = 19) ∧ 
  (126 + 52 / 60 + 12 / 3600 = 126.87) ∧ -- Converting the angle into degrees for simplicity
  (21.25 = 21.25)  -- Diameter given directly
  → triangle_side_lengths a b c :=
sorry

end NUMINAMATH_GPT_solve_triangle_l2236_223645
