import Mathlib

namespace sum_of_remainders_and_parity_l950_95031

theorem sum_of_remainders_and_parity 
  (n : ℤ) 
  (h₀ : n % 20 = 13) : 
  (n % 4 + n % 5 = 4) ∧ (n % 2 = 1) :=
by
  sorry

end sum_of_remainders_and_parity_l950_95031


namespace logarithm_identity_l950_95054

theorem logarithm_identity :
  1 / (Real.log 3 / Real.log 8 + 1) + 
  1 / (Real.log 2 / Real.log 12 + 1) + 
  1 / (Real.log 4 / Real.log 9 + 1) = 3 := 
by
  sorry

end logarithm_identity_l950_95054


namespace tile_calc_proof_l950_95039

noncomputable def total_tiles (length width : ℕ) : ℕ :=
  let border_tiles_length := (2 * (length - 4)) * 2
  let border_tiles_width := (2 * (width - 4)) * 2
  let total_border_tiles := (border_tiles_length + border_tiles_width) * 2 - 8
  let inner_length := (length - 4)
  let inner_width := (width - 4)
  let inner_area := inner_length * inner_width
  let inner_tiles := inner_area / 4
  total_border_tiles + inner_tiles

theorem tile_calc_proof :
  total_tiles 15 20 = 144 :=
by
  sorry

end tile_calc_proof_l950_95039


namespace jade_transactions_correct_l950_95095

-- Definitions for the conditions
def mabel_transactions : ℕ := 90
def anthony_transactions : ℕ := mabel_transactions + (mabel_transactions * 10 / 100)
def cal_transactions : ℕ := (2 * anthony_transactions) / 3
def jade_transactions : ℕ := cal_transactions + 16

-- The theorem stating what we want to prove
theorem jade_transactions_correct : jade_transactions = 82 := by
  sorry

end jade_transactions_correct_l950_95095


namespace correct_conclusion_l950_95090

noncomputable def proof_problem (a x : ℝ) (x1 x2 : ℝ) :=
  (a * (x - 1) * (x - 3) + 2 > 0 ∧ x1 < x2 ∧ 
   (∀ x, a * (x - 1) * (x - 3) + 2 > 0 ↔ x < x1 ∨ x > x2)) →
  (x1 + x2 = 4 ∧ 3 < x1 * x2 ∧ x1 * x2 < 4 ∧ 
   (∀ x, ((3 * a + 2) * x^2 - 4 * a * x + a < 0) ↔ (1 / x2 < x ∧ x < 1 / x1)))

theorem correct_conclusion (a x x1 x2 : ℝ) : 
proof_problem a x x1 x2 :=
by 
  unfold proof_problem 
  sorry

end correct_conclusion_l950_95090


namespace div_by_seven_iff_multiple_of_three_l950_95083

theorem div_by_seven_iff_multiple_of_three (n : ℕ) (hn : 0 < n) : 
  (7 ∣ (2^n - 1)) ↔ (3 ∣ n) := 
sorry

end div_by_seven_iff_multiple_of_three_l950_95083


namespace jina_teddies_l950_95063

variable (T : ℕ)

def initial_teddies (bunnies : ℕ) (koala : ℕ) (add_teddies : ℕ) (total : ℕ) :=
  T + bunnies + add_teddies + koala

theorem jina_teddies (bunnies : ℕ) (koala : ℕ) (add_teddies : ℕ) (total : ℕ) :
  bunnies = 3 * T ∧ koala = 1 ∧ add_teddies = 2 * bunnies ∧ total = 51 → T = 5 :=
by
  sorry

end jina_teddies_l950_95063


namespace edge_length_approx_17_1_l950_95025

-- Define the base dimensions of the rectangular vessel
def length_base : ℝ := 20
def width_base : ℝ := 15

-- Define the rise in water level
def rise_water_level : ℝ := 16.376666666666665

-- Calculate the area of the base
def area_base : ℝ := length_base * width_base

-- Calculate the volume of the cube (which is equal to the volume of water displaced)
def volume_cube : ℝ := area_base * rise_water_level

-- Calculate the edge length of the cube
def edge_length_cube : ℝ := volume_cube^(1/3)

-- Statement: The edge length of the cube is approximately 17.1 cm
theorem edge_length_approx_17_1 : abs (edge_length_cube - 17.1) < 0.1 :=
by sorry

end edge_length_approx_17_1_l950_95025


namespace minimum_value_expr_eq_neg6680_25_l950_95030

noncomputable def expr (x : ℝ) : ℝ := (15 - x) * (8 - x) * (15 + x) * (8 + x) - 200

theorem minimum_value_expr_eq_neg6680_25 : ∃ x : ℝ, (∀ y : ℝ, expr y ≥ expr x) ∧ expr x = -6680.25 :=
sorry

end minimum_value_expr_eq_neg6680_25_l950_95030


namespace series_converges_to_one_l950_95013

noncomputable def series_sum : ℝ :=
  ∑' n : ℕ, if n > 0 then (3 * (n : ℝ)^2 - 2 * (n : ℝ) + 1) / ((n : ℝ)^4 - (n : ℝ)^3 + (n : ℝ)^2 - (n : ℝ) + 1) else 0

theorem series_converges_to_one : series_sum = 1 := 
  sorry

end series_converges_to_one_l950_95013


namespace table_tennis_probability_l950_95011

-- Define the given conditions
def prob_A_wins_set : ℚ := 2 / 3
def prob_B_wins_set : ℚ := 1 / 3
def best_of_five_sets := 5
def needed_wins_for_A := 3
def needed_losses_for_A := 2

-- Define the problem to prove
theorem table_tennis_probability :
  ((prob_A_wins_set ^ 2) * prob_B_wins_set * prob_A_wins_set) = 8 / 27 :=
by
  sorry

end table_tennis_probability_l950_95011


namespace problem_statement_l950_95007

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else List.prod (List.range (n+1))

theorem problem_statement : ∃ r : ℕ, r < 13 ∧ (factorial 10) % 13 = r :=
by
  sorry

end problem_statement_l950_95007


namespace math_problem_l950_95056

theorem math_problem
  (p q r : ℝ)
  (h₁ : p + q + r = 5)
  (h₂ : (1 / (p + q)) + (1 / (q + r)) + (1 / (p + r)) = 9) :
  (r / (p + q)) + (p / (q + r)) + (q / (p + r)) = 42 := 
by
  sorry

end math_problem_l950_95056


namespace brother_combined_age_l950_95028

-- Define the ages of the brothers as integers
variable (x y : ℕ)

-- Define the condition given in the problem
def combined_age_six_years_ago : Prop := (x - 6) + (y - 6) = 100

-- State the theorem to prove the current combined age
theorem brother_combined_age (h : combined_age_six_years_ago x y): x + y = 112 :=
  sorry

end brother_combined_age_l950_95028


namespace probability_black_balls_l950_95076

variable {m1 m2 k1 k2 : ℕ}

/-- Given conditions:
  1. The total number of balls in both urns is 25.
  2. The probability of drawing one white ball from each urn is 0.54.
To prove: The probability of both drawn balls being black is 0.04.
-/
theorem probability_black_balls : 
  m1 + m2 = 25 → 
  (k1 * k2) * 50 = 27 * m1 * m2 → 
  ((m1 - k1) * (m2 - k2) : ℚ) / (m1 * m2) = 0.04 :=
by
  intros h1 h2
  sorry

end probability_black_balls_l950_95076


namespace time_to_fill_pool_l950_95055

-- Define the conditions given in the problem
def pool_volume_gallons : ℕ := 30000
def num_hoses : ℕ := 5
def hose_flow_rate_gpm : ℕ := 3

-- Define the total flow rate per minute
def total_flow_rate_gpm : ℕ := num_hoses * hose_flow_rate_gpm

-- Define the total flow rate per hour
def total_flow_rate_gph : ℕ := total_flow_rate_gpm * 60

-- Prove that the time to fill the pool is equal to 34 hours
theorem time_to_fill_pool : pool_volume_gallons / total_flow_rate_gph = 34 :=
by {
  -- Insert detailed proof steps here.
  sorry
}

end time_to_fill_pool_l950_95055


namespace total_blue_marbles_correct_l950_95022

def total_blue_marbles (j t e : ℕ) : ℕ :=
  j + t + e

theorem total_blue_marbles_correct :
  total_blue_marbles 44 24 36 = 104 :=
by
  sorry

end total_blue_marbles_correct_l950_95022


namespace hens_not_laying_eggs_l950_95026

def chickens_on_farm := 440
def number_of_roosters := 39
def total_eggs := 1158
def eggs_per_hen := 3

theorem hens_not_laying_eggs :
  (chickens_on_farm - number_of_roosters) - (total_eggs / eggs_per_hen) = 15 :=
by
  sorry

end hens_not_laying_eggs_l950_95026


namespace pentagon_rectangle_ratio_l950_95091

theorem pentagon_rectangle_ratio (p w l : ℝ) (h₁ : 5 * p = 20) (h₂ : l = 2 * w) (h₃ : 2 * l + 2 * w = 20) : p / w = 6 / 5 :=
by
  sorry

end pentagon_rectangle_ratio_l950_95091


namespace a_and_b_together_time_eq_4_over_3_l950_95081

noncomputable def work_together_time (a b c h : ℝ) :=
  (1 / a) + (1 / b) + (1 / c) = (1 / (a - 6)) ∧
  (1 / a) + (1 / b) = 1 / h ∧
  (1 / (a - 6)) = (1 / (b - 1)) ∧
  (1 / (a - 6)) = 2 / c

theorem a_and_b_together_time_eq_4_over_3 (a b c h : ℝ) (h_wt : work_together_time a b c h) : 
  h = 4 / 3 :=
  sorry

end a_and_b_together_time_eq_4_over_3_l950_95081


namespace ben_eggs_left_l950_95065

def initial_eggs : ℕ := 50
def day1_morning : ℕ := 5
def day1_afternoon : ℕ := 4
def day2_morning : ℕ := 8
def day2_evening : ℕ := 3
def day3_afternoon : ℕ := 6
def day3_night : ℕ := 2

theorem ben_eggs_left : initial_eggs - (day1_morning + day1_afternoon + day2_morning + day2_evening + day3_afternoon + day3_night) = 22 := 
by
  sorry

end ben_eggs_left_l950_95065


namespace geometric_sequence_min_value_l950_95023

theorem geometric_sequence_min_value (r : ℝ) (a1 a2 a3 : ℝ) 
  (h1 : a1 = 1) 
  (h2 : a2 = a1 * r) 
  (h3 : a3 = a2 * r) :
  4 * a2 + 5 * a3 ≥ -(4 / 5) :=
by
  sorry

end geometric_sequence_min_value_l950_95023


namespace dogs_legs_l950_95037

theorem dogs_legs (num_dogs : ℕ) (legs_per_dog : ℕ) (h1 : num_dogs = 109) (h2 : legs_per_dog = 4) : num_dogs * legs_per_dog = 436 :=
by {
  -- The proof is omitted as it's indicated that it should contain "sorry"
  sorry
}

end dogs_legs_l950_95037


namespace first_tribe_term_is_longer_l950_95060

def years_to_days_first_tribe (years : ℕ) : ℕ := 
  years * 12 * 30

def months_to_days_first_tribe (months : ℕ) : ℕ :=
  months * 30

def total_days_first_tribe (years months days : ℕ) : ℕ :=
  (years_to_days_first_tribe years) + (months_to_days_first_tribe months) + days

def years_to_days_second_tribe (years : ℕ) : ℕ := 
  years * 13 * 4 * 7

def moons_to_days_second_tribe (moons : ℕ) : ℕ :=
  moons * 4 * 7

def weeks_to_days_second_tribe (weeks : ℕ) : ℕ :=
  weeks * 7

def total_days_second_tribe (years moons weeks days : ℕ) : ℕ :=
  (years_to_days_second_tribe years) + (moons_to_days_second_tribe moons) + (weeks_to_days_second_tribe weeks) + days

theorem first_tribe_term_is_longer :
  total_days_first_tribe 7 1 18 > total_days_second_tribe 6 12 1 3 :=
by
  sorry

end first_tribe_term_is_longer_l950_95060


namespace woman_lawyer_probability_l950_95092

noncomputable def probability_of_woman_lawyer : ℚ :=
  let total_members : ℚ := 100
  let women_percentage : ℚ := 0.80
  let lawyer_percentage_women : ℚ := 0.40
  let women_members := women_percentage * total_members
  let women_lawyers := lawyer_percentage_women * women_members
  let probability := women_lawyers / total_members
  probability

theorem woman_lawyer_probability :
  probability_of_woman_lawyer = 0.32 := by
  sorry

end woman_lawyer_probability_l950_95092


namespace odd_times_even_is_even_l950_95098

def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1
def is_even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

theorem odd_times_even_is_even (a b : ℤ) (h₁ : is_odd a) (h₂ : is_even b) : is_even (a * b) :=
by sorry

end odd_times_even_is_even_l950_95098


namespace average_visitors_on_Sundays_l950_95005

theorem average_visitors_on_Sundays (S : ℕ) (h1 : 30 = 5 + 25) (h2 : 25 * 240 + 5 * S = 30 * 285) :
  S = 510 := sorry

end average_visitors_on_Sundays_l950_95005


namespace hyejin_math_score_l950_95066

theorem hyejin_math_score :
  let ethics := 82
  let korean_language := 90
  let science := 88
  let social_studies := 84
  let avg_score := 88
  let total_subjects := 5
  ∃ (M : ℕ), (ethics + korean_language + science + social_studies + M) / total_subjects = avg_score := by
    sorry

end hyejin_math_score_l950_95066


namespace unable_to_determine_questions_answered_l950_95048

variable (total_questions : ℕ) (total_time : ℕ) (used_time : ℕ) (remaining_time : ℕ)

theorem unable_to_determine_questions_answered (total_questions_eq : total_questions = 80)
  (total_time_eq : total_time = 60)
  (used_time_eq : used_time = 12)
  (remaining_time_eq : remaining_time = 0) :
  ∀ (answered_rate : ℕ → ℕ), ¬ ∃ questions_answered, answered_rate used_time = questions_answered :=
by sorry

end unable_to_determine_questions_answered_l950_95048


namespace gcd_24_36_54_l950_95049

-- Define the numbers and the gcd function
def num1 : ℕ := 24
def num2 : ℕ := 36
def num3 : ℕ := 54

-- The Lean statement to prove that the gcd of num1, num2, and num3 is 6
theorem gcd_24_36_54 : Nat.gcd (Nat.gcd num1 num2) num3 = 6 := by
  sorry

end gcd_24_36_54_l950_95049


namespace no_real_x_for_sqrt_l950_95046

theorem no_real_x_for_sqrt :
  ¬ ∃ x : ℝ, - (x^2 + 2 * x + 5) ≥ 0 :=
sorry

end no_real_x_for_sqrt_l950_95046


namespace percent_of_x_is_y_l950_95080

variable (x y : ℝ)

theorem percent_of_x_is_y (h : 0.20 * (x - y) = 0.15 * (x + y)) : (y / x) * 100 = 100 / 7 :=
by
  sorry

end percent_of_x_is_y_l950_95080


namespace trigonometric_identity_l950_95059

theorem trigonometric_identity (α : ℝ) (h : Real.cos α + Real.sin α = 2 / 3) :
  (Real.sqrt 2 * Real.sin (2 * α - Real.pi / 4) + 1) / (1 + Real.tan α) = - 5 / 9 :=
sorry

end trigonometric_identity_l950_95059


namespace exists_infinite_B_with_property_l950_95047

-- Definition of the sequence A
def seqA (n : ℕ) : ℤ := 5 * n - 2

-- Definition of the sequence B with its general form
def seqB (k : ℕ) (d : ℤ) : ℤ := k * d + 7 - d

-- The proof problem statement
theorem exists_infinite_B_with_property :
  ∃ (B : ℕ → ℤ) (d : ℤ), B 1 = 7 ∧ 
  (∀ k, k > 1 → B k = B (k - 1) + d) ∧
  (∀ n : ℕ, ∃ (k : ℕ), seqB k d = seqA n) :=
sorry

end exists_infinite_B_with_property_l950_95047


namespace phase_shift_3cos_4x_minus_pi_over_4_l950_95087

theorem phase_shift_3cos_4x_minus_pi_over_4 :
    ∃ (φ : ℝ), y = 3 * Real.cos (4 * x - φ) ∧ φ = π / 16 :=
sorry

end phase_shift_3cos_4x_minus_pi_over_4_l950_95087


namespace original_profit_percentage_l950_95015

noncomputable def originalCost : ℝ := 80
noncomputable def P := 30
noncomputable def profitPercentage : ℝ := ((100 - originalCost) / originalCost) * 100

theorem original_profit_percentage:
  ∀ (S C : ℝ),
  C = originalCost →
  ( ∀ (newCost : ℝ),
    newCost = 0.8 * C →
    ∀ (newSell : ℝ),
    newSell = S - 16.8 →
    newSell = 1.3 * newCost → P = 30 ) →
  profitPercentage = 25 := sorry

end original_profit_percentage_l950_95015


namespace rectangle_area_is_200000_l950_95053

structure Point :=
  (x : ℝ)
  (y : ℝ)

def isRectangle (P Q R S : Point) : Prop :=
  (P.x - Q.x) * (P.x - Q.x) + (P.y - Q.y) * (P.y - Q.y) = 
  (R.x - S.x) * (R.x - S.x) + (R.y - S.y) * (R.y - S.y) ∧
  (P.x - S.x) * (P.x - S.x) + (P.y - S.y) * (P.y - S.y) = 
  (Q.x - R.x) * (Q.x - R.x) + (Q.y - R.y) * (Q.y - R.y) ∧
  (P.x - Q.x) * (P.x - S.x) + (P.y - Q.y) * (P.y - S.y) = 0

theorem rectangle_area_is_200000:
  ∀ (P Q R S : Point),
  P = ⟨-15, 30⟩ →
  Q = ⟨985, 230⟩ →
  R.x = 985 → 
  S.x = -13 →
  R.y = S.y → 
  isRectangle P Q R S →
  ( ( (Q.x - P.x)^2 + (Q.y - P.y)^2 ).sqrt *
    ( (S.x - P.x)^2 + (S.y - P.y)^2 ).sqrt ) = 200000 :=
by
  intros P Q R S hP hQ hxR hxS hyR hRect
  sorry

end rectangle_area_is_200000_l950_95053


namespace b_investment_calculation_l950_95042

noncomputable def total_profit : ℝ := 9600
noncomputable def A_investment : ℝ := 2000
noncomputable def A_management_fee : ℝ := 0.10 * total_profit
noncomputable def remaining_profit : ℝ := total_profit - A_management_fee
noncomputable def A_total_received : ℝ := 4416
noncomputable def B_investment : ℝ := 1000

theorem b_investment_calculation (B: ℝ) 
  (h_total_profit: total_profit = 9600)
  (h_A_investment: A_investment = 2000)
  (h_A_management_fee: A_management_fee = 0.10 * total_profit)
  (h_remaining_profit: remaining_profit = total_profit - A_management_fee)
  (h_A_total_received: A_total_received = 4416)
  (h_A_total_formula : A_total_received = A_management_fee + (A_investment / (A_investment + B)) * remaining_profit) :
  B = 1000 :=
by
  have h1 : total_profit = 9600 := h_total_profit
  have h2 : A_investment = 2000 := h_A_investment
  have h3 : A_management_fee = 0.10 * total_profit := h_A_management_fee
  have h4 : remaining_profit = total_profit - A_management_fee := h_remaining_profit
  have h5 : A_total_received = 4416 := h_A_total_received
  have h6 : A_total_received = A_management_fee + (A_investment / (A_investment + B)) * remaining_profit := h_A_total_formula
  
  sorry

end b_investment_calculation_l950_95042


namespace pencil_length_after_sharpening_l950_95075

-- Definition of the initial length of the pencil
def initial_length : ℕ := 22

-- Definition of the amount sharpened each day
def sharpened_each_day : ℕ := 2

-- Final length of the pencil after sharpening on Monday and Tuesday
def final_length (initial_length : ℕ) (sharpened_each_day : ℕ) : ℕ :=
  initial_length - sharpened_each_day * 2

-- Theorem stating that the final length is 18 inches
theorem pencil_length_after_sharpening : final_length initial_length sharpened_each_day = 18 := by
  sorry

end pencil_length_after_sharpening_l950_95075


namespace smallest_positive_angle_l950_95002

def coterminal_angle (θ : ℤ) : ℤ := θ % 360

theorem smallest_positive_angle (θ : ℤ) (hθ : θ % 360 ≠ 0) : 
  0 < coterminal_angle θ ∧ coterminal_angle θ = 158 :=
by
  sorry

end smallest_positive_angle_l950_95002


namespace total_ages_l950_95093

theorem total_ages (bride_age groom_age : ℕ) (h1 : bride_age = 102) (h2 : groom_age = bride_age - 19) : bride_age + groom_age = 185 :=
by
  sorry

end total_ages_l950_95093


namespace circle_equation_l950_95004

theorem circle_equation (x y : ℝ) :
  (∃ a < 0, (x - a)^2 + y^2 = 4 ∧ (0 - a)^2 + 0^2 = 4) ↔ (x + 2)^2 + y^2 = 4 := 
sorry

end circle_equation_l950_95004


namespace minimum_value_of_expression_l950_95079

noncomputable def min_value_expression (x y z : ℝ) : ℝ :=
  1 / x + 4 / y + 9 / z

theorem minimum_value_of_expression (x y z : ℝ) (h_pos : x > 0 ∧ y > 0 ∧ z > 0) (h_sum : x + y + z = 1) :
  min_value_expression x y z ≥ 36 :=
sorry

end minimum_value_of_expression_l950_95079


namespace differentiable_function_zero_l950_95078

theorem differentiable_function_zero
    (f : ℝ → ℝ)
    (h_diff : Differentiable ℝ f)
    (h_zero : f 0 = 0)
    (h_ineq : ∀ x : ℝ, 0 < |f x| ∧ |f x| < 1/2 → |deriv f x| ≤ |f x * Real.log (|f x|)|) :
    ∀ x : ℝ, f x = 0 :=
by
  sorry

end differentiable_function_zero_l950_95078


namespace task_completion_time_l950_95069

noncomputable def work_time (A B C : ℝ) : ℝ := 1 / (A + B + C)

theorem task_completion_time (x y z : ℝ) (h1 : 8 * (x + y) = 1) (h2 : 6 * (x + z) = 1) (h3 : 4.8 * (y + z) = 1) :
    work_time x y z = 4 :=
by
  sorry

end task_completion_time_l950_95069


namespace sin_double_pi_minus_theta_eq_l950_95012

variable {θ : ℝ}
variable {k : ℤ}
variable (h1 : 3 * (Real.cos θ) ^ 2 = Real.tan θ + 3)
variable (h2 : θ ≠ k * Real.pi)

theorem sin_double_pi_minus_theta_eq :
  Real.sin (2 * (Real.pi - θ)) = 2 / 3 :=
sorry

end sin_double_pi_minus_theta_eq_l950_95012


namespace problem_1_problem_2_l950_95096

theorem problem_1 (h : Real.tan (α / 2) = 2) : Real.tan (α + Real.arctan 1) = -1/7 :=
by
  sorry

theorem problem_2 (h : Real.tan (α / 2) = 2) : (6 * Real.sin α + Real.cos α) / (3 * Real.sin α - 2 * Real.cos α) = 7 / 6 :=
by
  sorry

end problem_1_problem_2_l950_95096


namespace math_problem_l950_95086

theorem math_problem (x y : ℤ) (h1 : x = 12) (h2 : y = 18) : (x - y) * ((x + y) ^ 2) = -5400 := by
  sorry

end math_problem_l950_95086


namespace shaded_region_area_l950_95073

noncomputable def radius_large : ℝ := 10
noncomputable def radius_small : ℝ := 4

theorem shaded_region_area :
  let area_large := Real.pi * radius_large^2 
  let area_small := Real.pi * radius_small^2 
  (area_large - 2 * area_small) = 68 * Real.pi :=
by
  sorry

end shaded_region_area_l950_95073


namespace elena_hike_total_miles_l950_95084

theorem elena_hike_total_miles (x1 x2 x3 x4 x5 : ℕ)
  (h1 : x1 + x2 = 36)
  (h2 : x2 + x3 = 40)
  (h3 : x3 + x4 + x5 = 45)
  (h4 : x1 + x4 = 38) : 
  x1 + x2 + x3 + x4 + x5 = 81 := 
sorry

end elena_hike_total_miles_l950_95084


namespace distinct_divisors_sum_factorial_l950_95052

theorem distinct_divisors_sum_factorial (n : ℕ) (h : n ≥ 3) :
  ∃ (d : Fin n → ℕ), (∀ i j, i ≠ j → d i ≠ d j) ∧ (∀ i, d i ∣ n!) ∧ (n! = (Finset.univ.sum d)) :=
sorry

end distinct_divisors_sum_factorial_l950_95052


namespace eating_time_175_seconds_l950_95064

variable (Ponchik_time Neznaika_time : ℝ)
variable (Ponchik_rate Neznaika_rate : ℝ)

theorem eating_time_175_seconds
    (hP_rate : Ponchik_rate = 1 / Ponchik_time)
    (hP_time : Ponchik_time = 5)
    (hN_rate : Neznaika_rate = 1 / Neznaika_time)
    (hN_time : Neznaika_time = 7)
    (combined_rate := Ponchik_rate + Neznaika_rate)
    (total_minutes := 1 / combined_rate)
    (total_seconds := total_minutes * 60):
    total_seconds = 175 := by
  sorry

end eating_time_175_seconds_l950_95064


namespace athlete_speed_200m_in_24s_is_30kmh_l950_95071

noncomputable def speed_in_kmh (distance_meters : ℝ) (time_seconds : ℝ) : ℝ :=
  (distance_meters / 1000) / (time_seconds / 3600)

theorem athlete_speed_200m_in_24s_is_30kmh :
  speed_in_kmh 200 24 = 30 := by
  sorry

end athlete_speed_200m_in_24s_is_30kmh_l950_95071


namespace eq1_solution_eq2_solution_l950_95009


-- Theorem for the first equation (4(x + 1)^2 - 25 = 0)
theorem eq1_solution (x : ℝ) : (4 * (x + 1)^2 - 25 = 0) ↔ (x = 3 / 2 ∨ x = -7 / 2) :=
by
  sorry

-- Theorem for the second equation ((x + 10)^3 = -125)
theorem eq2_solution (x : ℝ) : ((x + 10)^3 = -125) ↔ (x = -15) :=
by
  sorry

end eq1_solution_eq2_solution_l950_95009


namespace simplify_and_evaluate_l950_95062

def expr (x : ℤ) : ℤ := (x + 2) * (x - 2) - (x - 1) ^ 2

theorem simplify_and_evaluate : expr (-1) = -7 := by
  sorry

end simplify_and_evaluate_l950_95062


namespace proof_multiple_l950_95072

def is_multiple (a b : ℕ) : Prop := ∃ k : ℕ, a = k * b

theorem proof_multiple (a b : ℕ) 
  (h₁ : is_multiple a 5) 
  (h₂ : is_multiple b 10) : 
  is_multiple b 5 ∧ 
  is_multiple (a + b) 5 ∧ 
  is_multiple (a + b) 2 :=
by
  sorry

end proof_multiple_l950_95072


namespace second_tree_ring_groups_l950_95033

-- Definition of the problem conditions
def group_rings (fat thin : Nat) : Nat := fat + thin

-- Conditions
def FirstTreeRingGroups : Nat := 70
def RingsPerGroup : Nat := group_rings 2 4
def FirstTreeRings : Nat := FirstTreeRingGroups * RingsPerGroup
def AgeDifference : Nat := 180

-- Calculate the total number of rings in the second tree
def SecondTreeRings : Nat := FirstTreeRings - AgeDifference

-- Prove the number of ring groups in the second tree
theorem second_tree_ring_groups : SecondTreeRings / RingsPerGroup = 40 :=
by
  sorry

end second_tree_ring_groups_l950_95033


namespace sum_of_digits_in_product_is_fourteen_l950_95019

def first_number : ℕ := -- Define the 101-digit number 141,414,141,...,414,141
  141 * 10^98 + 141 * 10^95 + 141 * 10^92 -- continue this pattern...

def second_number : ℕ := -- Define the 101-digit number 707,070,707,...,070,707
  707 * 10^98 + 707 * 10^95 + 707 * 10^92 -- continue this pattern...

def units_digit (n : ℕ) : ℕ := n % 10
def ten_thousands_digit (n : ℕ) : ℕ := (n / 10000) % 10

theorem sum_of_digits_in_product_is_fourteen :
  units_digit (first_number * second_number) + ten_thousands_digit (first_number * second_number) = 14 :=
sorry

end sum_of_digits_in_product_is_fourteen_l950_95019


namespace min_fence_dimensions_l950_95043

theorem min_fence_dimensions (A : ℝ) (hA : A ≥ 800) (x : ℝ) (hx : 2 * x * x = A) : x = 20 ∧ 2 * x = 40 := by
  sorry

end min_fence_dimensions_l950_95043


namespace biff_break_even_time_l950_95036

noncomputable def total_cost_excluding_wifi : ℝ :=
  11 + 3 + 16 + 8 + 10 + 35 + 0.1 * 35

noncomputable def total_cost_including_wifi_connection : ℝ :=
  total_cost_excluding_wifi + 5

noncomputable def effective_hourly_earning : ℝ := 12 - 1

noncomputable def hours_to_break_even : ℝ :=
  total_cost_including_wifi_connection / effective_hourly_earning

theorem biff_break_even_time : hours_to_break_even ≤ 9 := by
  sorry

end biff_break_even_time_l950_95036


namespace graph_is_pair_of_straight_lines_l950_95034

theorem graph_is_pair_of_straight_lines : ∀ (x y : ℝ), 9 * x^2 - y^2 - 6 * x = 0 → ∃ a b c : ℝ, (y = 3 * x - 2 ∨ y = 2 - 3 * x) :=
by
  intro x y h
  sorry

end graph_is_pair_of_straight_lines_l950_95034


namespace inequality_solution_l950_95021

theorem inequality_solution (x : ℝ) : 9 - x^2 < 0 ↔ x < -3 ∨ x > 3 := by
  sorry

end inequality_solution_l950_95021


namespace complex_square_l950_95040

-- Define z and the condition on i
def z := 5 + (6 * Complex.I)
axiom i_squared : Complex.I ^ 2 = -1

-- State the theorem to prove z^2 = -11 + 60i
theorem complex_square : z ^ 2 = -11 + (60 * Complex.I) := by {
  sorry
}

end complex_square_l950_95040


namespace find_i_when_x_is_0_point3_l950_95014

noncomputable def find_i (x : ℝ) (i : ℝ) : Prop :=
  (10 * x + 2) / 4 - (3 * x - 6) / 18 = (2 * x + 4) / i

theorem find_i_when_x_is_0_point3 : find_i 0.3 2.9993 :=
by
  sorry

end find_i_when_x_is_0_point3_l950_95014


namespace minimum_deposits_needed_l950_95024

noncomputable def annual_salary_expense : ℝ := 100000
noncomputable def annual_fixed_expense : ℝ := 170000
noncomputable def interest_rate_paid : ℝ := 0.0225
noncomputable def interest_rate_earned : ℝ := 0.0405

theorem minimum_deposits_needed :
  ∃ (x : ℝ), 
    (interest_rate_earned * x = annual_salary_expense + annual_fixed_expense + interest_rate_paid * x) →
    x = 1500 :=
by
  sorry

end minimum_deposits_needed_l950_95024


namespace minimum_restoration_time_l950_95041

structure Handicraft :=
  (shaping: ℕ)
  (painting: ℕ)

def handicraft_A : Handicraft := ⟨9, 15⟩
def handicraft_B : Handicraft := ⟨16, 8⟩
def handicraft_C : Handicraft := ⟨10, 14⟩

def total_restoration_time (order: List Handicraft) : ℕ :=
  let rec aux (remaining: List Handicraft) (A_time: ℕ) (B_time: ℕ) (acc: ℕ) : ℕ :=
    match remaining with
    | [] => acc
    | h :: t =>
      let A_next := A_time + h.shaping
      let B_next := max A_next B_time + h.painting
      aux t A_next B_next B_next
  aux order 0 0 0

theorem minimum_restoration_time :
  total_restoration_time [handicraft_A, handicraft_C, handicraft_B] = 46 :=
by
  simp [total_restoration_time, handicraft_A, handicraft_B, handicraft_C]
  sorry

end minimum_restoration_time_l950_95041


namespace sum_mod_eleven_l950_95001

variable (x y z : ℕ)

theorem sum_mod_eleven (h1 : (x * y * z) % 11 = 3)
                       (h2 : (7 * z) % 11 = 4)
                       (h3 : (9 * y) % 11 = (5 + y) % 11) :
                       (x + y + z) % 11 = 5 :=
sorry

end sum_mod_eleven_l950_95001


namespace num_machines_first_scenario_l950_95074

theorem num_machines_first_scenario (r : ℝ) (n : ℕ) :
  (∀ r, (2 : ℝ) * r * 24 = 1) →
  (∀ r, (n : ℝ) * r * 6 = 1) →
  n = 8 :=
by
  intros h1 h2
  sorry

end num_machines_first_scenario_l950_95074


namespace people_speak_neither_l950_95020

-- Define the total number of people
def total_people : ℕ := 25

-- Define the number of people who can speak Latin
def speak_latin : ℕ := 13

-- Define the number of people who can speak French
def speak_french : ℕ := 15

-- Define the number of people who can speak both Latin and French
def speak_both : ℕ := 9

-- Prove that the number of people who don't speak either Latin or French is 6
theorem people_speak_neither : (total_people - (speak_latin + speak_french - speak_both)) = 6 := by
  sorry

end people_speak_neither_l950_95020


namespace problem_l950_95008

noncomputable def B : Matrix (Fin 2) (Fin 2) ℝ := sorry
def v : Fin 2 → ℝ := ![7, -3]
def result : Fin 2 → ℝ := ![-14, 6]
def expected : Fin 2 → ℝ := ![112, -48]

theorem problem :
    B.vecMul v = result →
    B.vecMul (B.vecMul (B.vecMul (B.vecMul v))) = expected := 
by
  intro h
  sorry

end problem_l950_95008


namespace andy_coats_l950_95032

theorem andy_coats 
  (initial_minks : ℕ)
  (offspring_4_minks count_4_offspring : ℕ)
  (offspring_6_minks count_6_offspring : ℕ)
  (offspring_8_minks count_8_offspring : ℕ)
  (freed_percentage coat_requirement total_minks offspring_minks freed_minks remaining_minks coats : ℕ) :
  initial_minks = 30 ∧
  offspring_4_minks = 10 ∧ count_4_offspring = 4 ∧
  offspring_6_minks = 15 ∧ count_6_offspring = 6 ∧
  offspring_8_minks = 5 ∧ count_8_offspring = 8 ∧
  freed_percentage = 60 ∧ coat_requirement = 15 ∧
  total_minks = initial_minks + offspring_minks ∧
  offspring_minks = offspring_4_minks * count_4_offspring + offspring_6_minks * count_6_offspring + offspring_8_minks * count_8_offspring ∧
  freed_minks = total_minks * freed_percentage / 100 ∧
  remaining_minks = total_minks - freed_minks ∧
  coats = remaining_minks / coat_requirement →
  coats = 5 :=
sorry

end andy_coats_l950_95032


namespace num_real_solutions_abs_eq_l950_95085

theorem num_real_solutions_abs_eq :
  (∃ x y : ℝ, x ≠ y ∧ |x-1| = |x-2| + |x-3| + |x-4| 
    ∧ |y-1| = |y-2| + |y-3| + |y-4| 
    ∧ ∀ z : ℝ, |z-1| = |z-2| + |z-3| + |z-4| → (z = x ∨ z = y)) := sorry

end num_real_solutions_abs_eq_l950_95085


namespace find_smaller_number_l950_95057

theorem find_smaller_number (n m : ℕ) (h1 : n - m = 58)
  (h2 : n^2 % 100 = m^2 % 100) : m = 21 :=
by
  sorry

end find_smaller_number_l950_95057


namespace solve_r_l950_95088

theorem solve_r (r : ℚ) :
  (r^2 - 5*r + 4) / (r^2 - 8*r + 7) = (r^2 - 2*r - 15) / (r^2 - r - 20) →
  r = -5/4 :=
by
  -- Proof would go here
  sorry

end solve_r_l950_95088


namespace man_speed_in_still_water_l950_95051

noncomputable def speedInStillWater 
  (upstreamSpeedWithCurrentAndWind : ℝ)
  (downstreamSpeedWithCurrentAndWind : ℝ)
  (waterCurrentSpeed : ℝ)
  (windSpeedUpstream : ℝ) : ℝ :=
  (upstreamSpeedWithCurrentAndWind + waterCurrentSpeed + windSpeedUpstream + downstreamSpeedWithCurrentAndWind - waterCurrentSpeed + windSpeedUpstream) / 2
  
theorem man_speed_in_still_water :
  speedInStillWater 20 60 5 2.5 = 42.5 :=
  sorry

end man_speed_in_still_water_l950_95051


namespace rod_length_l950_95067

theorem rod_length (pieces : ℕ) (length_per_piece_cm : ℕ) (total_length_m : ℝ) :
  pieces = 35 → length_per_piece_cm = 85 → total_length_m = 29.75 :=
by
  intros h1 h2
  sorry

end rod_length_l950_95067


namespace power_mod_remainder_l950_95097

theorem power_mod_remainder :
  (7 ^ 2023) % 17 = 16 :=
sorry

end power_mod_remainder_l950_95097


namespace average_speed_of_car_l950_95045

/-- The car's average speed given it travels 65 km in the first hour and 45 km in the second hour. -/
theorem average_speed_of_car (d1 d2 : ℕ) (t : ℕ) (h1 : d1 = 65) (h2 : d2 = 45) (h3 : t = 2) :
  (d1 + d2) / t = 55 :=
by
  sorry

end average_speed_of_car_l950_95045


namespace uncle_gave_13_l950_95017

-- Define all the given constants based on the conditions.
def J := 7    -- cost of the jump rope
def B := 12   -- cost of the board game
def P := 4    -- cost of the playground ball
def S := 6    -- savings from Dalton's allowance
def N := 4    -- additional amount needed

-- Derived quantities
def total_cost := J + B + P

-- Statement: to prove Dalton's uncle gave him $13.
theorem uncle_gave_13 : (total_cost - N) - S = 13 := by
  sorry

end uncle_gave_13_l950_95017


namespace find_M_plus_N_l950_95006

theorem find_M_plus_N (M N : ℕ)
  (h1 : 4 * 63 = 7 * M)
  (h2 : 4 * N = 7 * 84) :
  M + N = 183 :=
by sorry

end find_M_plus_N_l950_95006


namespace first_person_days_l950_95016

-- Define the condition that Tanya is 25% more efficient than the first person and that Tanya takes 12 days to do the work.
def tanya_more_efficient (x : ℕ) : Prop :=
  -- Efficiency relationship: tanya (12 days) = 3 days less than the first person
  12 = x - (x / 4)

-- Define the theorem that the first person takes 15 days to do the work
theorem first_person_days : ∃ x : ℕ, tanya_more_efficient x ∧ x = 15 := 
by
  sorry -- proof is not required

end first_person_days_l950_95016


namespace min_value_fraction_l950_95070

theorem min_value_fraction {a : ℕ → ℕ} (h1 : a 1 = 10)
    (h2 : ∀ n : ℕ, a (n + 1) - a n = 2 * n) :
    ∃ n : ℕ, (n > 0) ∧ (n - 1 + 10 / n = 16 / 3) :=
by {
  sorry
}

end min_value_fraction_l950_95070


namespace central_angle_eq_one_l950_95077

noncomputable def radian_measure_of_sector (α r : ℝ) : Prop :=
  α * r = 2 ∧ (1 / 2) * α * r^2 = 2

-- Theorem stating the radian measure of the central angle is 1
theorem central_angle_eq_one (α r : ℝ) (h : radian_measure_of_sector α r) : α = 1 :=
by
  -- provide proof steps here
  sorry

end central_angle_eq_one_l950_95077


namespace product_remainder_mod_7_l950_95000

theorem product_remainder_mod_7 (a b c : ℕ) 
  (ha : a % 7 = 2) 
  (hb : b % 7 = 3) 
  (hc : c % 7 = 4) : 
  (a * b * c) % 7 = 3 :=
by 
  sorry

end product_remainder_mod_7_l950_95000


namespace part_a_constant_part_b_inequality_l950_95068

open Real

noncomputable def cubic_root (x : ℝ) : ℝ := x ^ (1 / 3)

theorem part_a_constant (x1 x2 x3 : ℝ) (h : x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3) :
  (cubic_root (x1 * x2 / x3^2) + cubic_root (x2 * x3 / x1^2) + cubic_root (x3 * x1 / x2^2)) = 
  const_value := sorry

theorem part_b_inequality (x1 x2 x3 : ℝ) (h : x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3) :
  (cubic_root (x1^2 / (x2 * x3)) + cubic_root (x2^2 / (x3 * x1)) + cubic_root (x3^2 / (x1 * x2))) < (-15 / 4) := sorry

end part_a_constant_part_b_inequality_l950_95068


namespace max_value_of_sequence_l950_95089

theorem max_value_of_sequence : 
  ∃ n : ℕ, n > 0 ∧ ∀ m : ℕ, m > 0 → (∃ (a : ℝ), a = (m / (m^2 + 6 : ℝ)) ∧ a ≤ (n / (n^2 + 6 : ℝ))) :=
sorry

end max_value_of_sequence_l950_95089


namespace rachel_total_homework_pages_l950_95050

-- Define the conditions
def math_homework_pages : Nat := 10
def additional_reading_pages : Nat := 3

-- Define the proof goal
def total_homework_pages (math_pages reading_extra : Nat) : Nat :=
  math_pages + (math_pages + reading_extra)

-- The final statement with the expected result
theorem rachel_total_homework_pages : total_homework_pages math_homework_pages additional_reading_pages = 23 :=
by
  sorry

end rachel_total_homework_pages_l950_95050


namespace perfect_square_base9_last_digit_l950_95029

-- We define the problem conditions
variable {b d f : ℕ} -- all variables are natural numbers
-- Condition 1: Base 9 representation of a perfect square
variable (n : ℕ) -- n is the perfect square number
variable (sqrt_n : ℕ) -- sqrt_n is the square root of n (so, n = sqrt_n^2)
variable (h1 : n = b * 9^3 + d * 9^2 + 4 * 9 + f)
variable (h2 : b ≠ 0)
-- The question becomes that the possible values of f are 0, 1, or 4
theorem perfect_square_base9_last_digit (h3 : n = sqrt_n^2) (hb : b ≠ 0) : 
  (f = 0) ∨ (f = 1) ∨ (f = 4) :=
by
  sorry

end perfect_square_base9_last_digit_l950_95029


namespace symmetric_point_with_respect_to_y_eq_x_l950_95038

variables (P : ℝ × ℝ) (line : ℝ → ℝ)

theorem symmetric_point_with_respect_to_y_eq_x (P : ℝ × ℝ) (hP : P = (1, 3)) (hline : ∀ x, line x = x) :
  (∃ Q : ℝ × ℝ, Q = (3, 1) ∧ Q = (P.snd, P.fst)) :=
by
  sorry

end symmetric_point_with_respect_to_y_eq_x_l950_95038


namespace probability_of_equal_numbers_when_throwing_two_fair_dice_l950_95027

theorem probability_of_equal_numbers_when_throwing_two_fair_dice :
  let total_outcomes := 36
  let favorable_outcomes := 6
  favorable_outcomes / total_outcomes = 1 / 6 :=
by
  sorry

end probability_of_equal_numbers_when_throwing_two_fair_dice_l950_95027


namespace composite_divides_factorial_l950_95003

-- Define the factorial of a number
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Statement of the problem
theorem composite_divides_factorial (m : ℕ) (hm : m ≠ 4) (hcomposite : ∃ a b : ℕ, 1 < a ∧ 1 < b ∧ a * b = m) :
  m ∣ factorial (m - 1) :=
by
  sorry

end composite_divides_factorial_l950_95003


namespace min_races_needed_l950_95082

noncomputable def minimum_races (total_horses : ℕ) (max_race_horses : ℕ) : ℕ :=
  if total_horses ≤ max_race_horses then 1 else
  if total_horses % max_race_horses = 0 then total_horses / max_race_horses else total_horses / max_race_horses + 1

/-- We need to show that the minimum number of races required to find the top 3 fastest horses
    among 35 horses, where a maximum of 4 horses can race together at a time, is 10. -/
theorem min_races_needed : minimum_races 35 4 = 10 :=
  sorry

end min_races_needed_l950_95082


namespace girls_percentage_l950_95061

theorem girls_percentage (total_students girls boys : ℕ) 
    (total_eq : total_students = 42)
    (ratio : 3 * girls = 4 * boys)
    (total_students_eq : total_students = girls + boys) : 
    (girls * 100 / total_students : ℚ) = 57.14 := 
by 
  sorry

end girls_percentage_l950_95061


namespace ceil_floor_difference_l950_95058

open Int

theorem ceil_floor_difference : 
  (Int.ceil (15 / 8 * (-34 / 4)) - Int.floor (15 / 8 * Int.floor (-34 / 4))) = 2 := 
by
  sorry

end ceil_floor_difference_l950_95058


namespace younger_brother_silver_fraction_l950_95035

def frac_silver (x y : ℕ) : ℚ := (100 - x / 7 ) / y

theorem younger_brother_silver_fraction {x y : ℕ} 
    (cond1 : x / 5 + y / 7 = 100) 
    (cond2 : x / 7 + (100 - x / 7) = 100) : 
    frac_silver x y = 5 / 14 := 
sorry

end younger_brother_silver_fraction_l950_95035


namespace number_of_orange_ribbons_l950_95099

/-- Define the total number of ribbons -/
def total_ribbons (yellow purple orange black total : ℕ) : Prop :=
  yellow + purple + orange + black = total

/-- Define the fractions -/
def fractions (total_ribbons yellow purple orange black : ℕ) : Prop :=
  yellow = total_ribbons / 4 ∧ purple = total_ribbons / 3 ∧ orange = total_ribbons / 12 ∧ black = 40

/-- Define the black ribbons fraction -/
def black_fraction (total_ribbons : ℕ) : Prop :=
  40 = total_ribbons / 3

theorem number_of_orange_ribbons :
  ∃ (total : ℕ), total_ribbons (total / 4) (total / 3) (total / 12) 40 total ∧ black_fraction total ∧ (total / 12 = 10) :=
by
  sorry

end number_of_orange_ribbons_l950_95099


namespace probability_of_heads_on_999th_toss_l950_95018

theorem probability_of_heads_on_999th_toss (fair_coin : Bool → ℝ) :
  (∀ (i : ℕ), fair_coin true = 1 / 2 ∧ fair_coin false = 1 / 2) →
  fair_coin true = 1 / 2 :=
by
  sorry

end probability_of_heads_on_999th_toss_l950_95018


namespace wraps_add_more_l950_95010

/-- Let John's raw squat be 600 pounds. Let sleeves add 30 pounds to his lift. Let wraps add 25% 
to his squat. We aim to prove that wraps add 120 pounds more to John's squat than sleeves. -/
theorem wraps_add_more (raw_squat : ℝ) (sleeves_bonus : ℝ) (wraps_percentage : ℝ) : 
  raw_squat = 600 → sleeves_bonus = 30 → wraps_percentage = 0.25 → 
  (raw_squat * wraps_percentage) - sleeves_bonus = 120 :=
by
  intros h1 h2 h3
  sorry

end wraps_add_more_l950_95010


namespace sufficient_condition_perpendicular_l950_95044

-- Definitions of perpendicularity and lines/planes intersections
variables {Plane : Type} {Line : Type}

variable (α β γ : Plane)
variable (m n l : Line)

-- Axioms representing the given conditions
axiom perp_planes (p₁ p₂ : Plane) : Prop -- p₁ is perpendicular to p₂
axiom perp_line_plane (line : Line) (plane : Plane) : Prop -- line is perpendicular to plane

-- Given conditions for the problem.
axiom n_perp_α : perp_line_plane n α
axiom n_perp_β : perp_line_plane n β
axiom m_perp_α : perp_line_plane m α

-- The proposition to be proved.
theorem sufficient_condition_perpendicular (h₁ : perp_line_plane n α)
                                           (h₂ : perp_line_plane n β)
                                           (h₃ : perp_line_plane m α) :
  perp_line_plane m β := sorry

end sufficient_condition_perpendicular_l950_95044


namespace weight_of_one_apple_l950_95094

-- Conditions
def total_weight_of_bag_with_apples : ℝ := 1.82
def weight_of_empty_bag : ℝ := 0.5
def number_of_apples : ℕ := 6

-- The proposition to prove: the weight of one apple
theorem weight_of_one_apple : (total_weight_of_bag_with_apples - weight_of_empty_bag) / number_of_apples = 0.22 := 
by
  sorry

end weight_of_one_apple_l950_95094
