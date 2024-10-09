import Mathlib

namespace negative_integers_abs_le_4_l1053_105315

theorem negative_integers_abs_le_4 :
  ∀ x : ℤ, x < 0 ∧ |x| ≤ 4 ↔ (x = -1 ∨ x = -2 ∨ x = -3 ∨ x = -4) :=
by
  sorry

end negative_integers_abs_le_4_l1053_105315


namespace hendricks_payment_l1053_105371

variable (Hendricks Gerald : ℝ)
variable (less_percent : ℝ) (amount_paid : ℝ)

theorem hendricks_payment (h g : ℝ) (h_less_g : h = g * (1 - less_percent)) (g_val : g = amount_paid) (less_percent_val : less_percent = 0.2) (amount_paid_val: amount_paid = 250) :
h = 200 :=
by
  sorry

end hendricks_payment_l1053_105371


namespace gcd_polynomial_multiple_l1053_105331

theorem gcd_polynomial_multiple (b : ℕ) (hb : 620 ∣ b) : gcd (4 * b^3 + 2 * b^2 + 5 * b + 93) b = 93 := by
  sorry

end gcd_polynomial_multiple_l1053_105331


namespace find_b_l1053_105345

noncomputable def geom_seq_term (a b c : ℝ) : Prop :=
∃ r : ℝ, r > 0 ∧ b = a * r ∧ c = b * r

theorem find_b (b : ℝ) (h_geom : geom_seq_term 160 b (108 / 64)) (h_pos : b > 0) :
  b = 15 * Real.sqrt 6 :=
by
  sorry

end find_b_l1053_105345


namespace fish_worth_bags_of_rice_l1053_105366

variable (f l a r : ℝ)

theorem fish_worth_bags_of_rice
    (h1 : 5 * f = 3 * l)
    (h2 : l = 6 * a)
    (h3 : 2 * a = r) :
    1 / f = 9 / (5 * r) :=
by
  sorry

end fish_worth_bags_of_rice_l1053_105366


namespace abs_sqrt2_sub_2_l1053_105387

theorem abs_sqrt2_sub_2 (h : 1 < Real.sqrt 2 ∧ Real.sqrt 2 < 2) : |Real.sqrt 2 - 2| = 2 - Real.sqrt 2 :=
by
  sorry

end abs_sqrt2_sub_2_l1053_105387


namespace employees_salaries_l1053_105370

theorem employees_salaries (M N P : ℝ)
  (hM : M = 1.20 * N)
  (hN_median : N = N) -- Indicates N is the median
  (hP : P = 0.65 * M)
  (h_total : N + M + P = 3200) :
  M = 1288.58 ∧ N = 1073.82 ∧ P = 837.38 :=
by
  sorry

end employees_salaries_l1053_105370


namespace journey_speed_first_half_l1053_105330

theorem journey_speed_first_half (total_distance : ℕ) (total_time : ℕ) (second_half_distance : ℕ) (second_half_speed : ℕ)
  (distance_first_half_eq_half_total : second_half_distance = total_distance / 2)
  (time_for_journey_eq : total_time = 20)
  (journey_distance_eq : total_distance = 240)
  (second_half_speed_eq : second_half_speed = 15) :
  let v := second_half_distance / (total_time - (second_half_distance / second_half_speed))
  v = 10 := 
by
  sorry

end journey_speed_first_half_l1053_105330


namespace john_buys_1000_balloons_l1053_105306

-- Define conditions
def balloon_volume : ℕ := 10
def tank_volume : ℕ := 500
def num_tanks : ℕ := 20

-- Define the total volume of gas
def total_gas_volume : ℕ := num_tanks * tank_volume

-- Define the number of balloons
def num_balloons : ℕ := total_gas_volume / balloon_volume

-- Prove that the number of balloons is 1,000
theorem john_buys_1000_balloons : num_balloons = 1000 := by
  sorry

end john_buys_1000_balloons_l1053_105306


namespace book_length_l1053_105396

variable (length width perimeter : ℕ)

theorem book_length
  (h1 : perimeter = 100)
  (h2 : width = 20)
  (h3 : perimeter = 2 * (length + width)) :
  length = 30 :=
by sorry

end book_length_l1053_105396


namespace sequence_term_expression_l1053_105338

theorem sequence_term_expression (a : ℕ → ℝ) (S : ℕ → ℝ) (C : ℝ)
  (h1 : a 1 = 1)
  (h2 : ∀ n, S n + n * a n = C)
  (h3 : ∀ n ≥ 2, (n + 1) * a n = (n - 1) * a (n - 1)) :
  ∀ n, a n = 2 / (n * (n + 1)) :=
by
  sorry

end sequence_term_expression_l1053_105338


namespace maddie_spent_in_all_l1053_105373

-- Define the given conditions
def white_packs : ℕ := 2
def blue_packs : ℕ := 4
def t_shirts_per_white_pack : ℕ := 5
def t_shirts_per_blue_pack : ℕ := 3
def cost_per_t_shirt : ℕ := 3

-- Define the question as a theorem to be proved
theorem maddie_spent_in_all :
  (white_packs * t_shirts_per_white_pack + blue_packs * t_shirts_per_blue_pack) * cost_per_t_shirt = 66 :=
by 
  -- The proof goes here
  sorry

end maddie_spent_in_all_l1053_105373


namespace balloons_remaining_l1053_105316
-- Importing the necessary libraries

-- Defining the conditions
def originalBalloons : Nat := 709
def givenBalloons : Nat := 221

-- Stating the theorem
theorem balloons_remaining : originalBalloons - givenBalloons = 488 := by
  sorry

end balloons_remaining_l1053_105316


namespace calculate_x_l1053_105376

theorem calculate_x (a b x : ℕ) (h1 : b = 9) (h2 : b - a = 5) (h3 : a * b = 2 * (a + b) + x) : x = 10 :=
by
  sorry

end calculate_x_l1053_105376


namespace math_problem_l1053_105351

theorem math_problem
  (a b c d : ℚ)
  (h₁ : a = 1 / 3)
  (h₂ : b = 1 / 6)
  (h₃ : c = 1 / 9)
  (h₄ : d = 1 / 18) :
  9 * (a + b + c + d)⁻¹ = 27 / 2 := 
sorry

end math_problem_l1053_105351


namespace min_value_fraction_sum_l1053_105347

theorem min_value_fraction_sum (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 4) :
  (∃ x : ℝ, x = (1 / a + 4 / b) ∧ x = 9 / 4) :=
by
  sorry

end min_value_fraction_sum_l1053_105347


namespace uncovered_side_length_l1053_105336

theorem uncovered_side_length
  (A : ℝ) (F : ℝ)
  (h1 : A = 600)
  (h2 : F = 130) :
  ∃ L : ℝ, L = 120 :=
by {
  sorry
}

end uncovered_side_length_l1053_105336


namespace math_quiz_scores_stability_l1053_105393

theorem math_quiz_scores_stability :
  let avgA := (90 + 82 + 88 + 96 + 94) / 5
  let avgB := (94 + 86 + 88 + 90 + 92) / 5
  let varA := ((90 - avgA) ^ 2 + (82 - avgA) ^ 2 + (88 - avgA) ^ 2 + (96 - avgA) ^ 2 + (94 - avgA) ^ 2) / 5
  let varB := ((94 - avgB) ^ 2 + (86 - avgB) ^ 2 + (88 - avgB) ^ 2 + (90 - avgB) ^ 2 + (92 - avgB) ^ 2) / 5
  avgA = avgB ∧ varB < varA :=
by
  sorry

end math_quiz_scores_stability_l1053_105393


namespace jerry_age_l1053_105335

variable (M J : ℕ) -- Declare Mickey's and Jerry's ages as natural numbers

-- Define the conditions as hypotheses
def condition1 := M = 2 * J - 6
def condition2 := M = 18

-- Theorem statement where we need to prove J = 12 given the conditions
theorem jerry_age
  (h1 : condition1 M J)
  (h2 : condition2 M) :
  J = 12 :=
sorry

end jerry_age_l1053_105335


namespace new_girl_weight_l1053_105329

theorem new_girl_weight (W : ℝ) (h : (W + 24) / 8 = W / 8 + 3) :
  (W + 24) - (W - 70) = 94 :=
by
  sorry

end new_girl_weight_l1053_105329


namespace find_number_l1053_105311

theorem find_number (x q : ℕ) (h1 : x = 7 * q) (h2 : q + x + 7 = 175) : x = 147 := 
by
  sorry

end find_number_l1053_105311


namespace calc_expression_l1053_105314

theorem calc_expression : 3 * 3^4 - 9^32 / 9^30 = 162 := by
  -- We would provide the proof here, but skipping with sorry
  sorry

end calc_expression_l1053_105314


namespace midpoint_chord_hyperbola_l1053_105327

-- Definitions to use in our statement
variables (a b x y : ℝ)
def ellipse : Prop := (x^2)/(a^2) + (y^2)/(b^2) = 1
def line_ellipse : Prop := x / (a^2) + y / (b^2) = 0
def hyperbola : Prop := (x^2)/(a^2) - (y^2)/(b^2) = 1
def line_hyperbola : Prop := x / (a^2) - y / (b^2) = 0

-- The theorem to prove
theorem midpoint_chord_hyperbola (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) (x y : ℝ) 
    (h_ellipse : ellipse a b x y)
    (h_line_ellipse : line_ellipse a b x y)
    (h_hyperbola : hyperbola a b x y) :
    line_hyperbola a b x y :=
sorry

end midpoint_chord_hyperbola_l1053_105327


namespace inequality_proof_l1053_105388

theorem inequality_proof (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b * c = 1) : 
  1 / (a^3 * (b + c)) + 1 / (b^3 * (c + a)) + 1 / (c^3 * (a + b)) ≥ 3 / 2 :=
by
  sorry

end inequality_proof_l1053_105388


namespace victory_points_value_l1053_105326

theorem victory_points_value (V : ℕ) (H : ∀ (v d t : ℕ), 
    v + d + t = 20 ∧ v * V + d ≥ 40 ∧ v ≥ 6 ∧ (t = 20 - 5)) : 
    V = 3 := 
sorry

end victory_points_value_l1053_105326


namespace simplify_expression1_simplify_expression2_l1053_105365

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (A B C D F : V)

-- Problem 1:
theorem simplify_expression1 : 
  (D - C) + (C - B) + (B - A) = D - A := 
sorry

-- Problem 2:
theorem simplify_expression2 : 
  (B - A) + (F - D) + (D - C) + (C - B) + (A - F) = 0 := 
sorry

end simplify_expression1_simplify_expression2_l1053_105365


namespace city_population_l1053_105349

theorem city_population (P: ℝ) (h: 0.85 * P = 85000) : P = 100000 := 
by
  sorry

end city_population_l1053_105349


namespace tan_square_B_eq_tan_A_tan_C_range_l1053_105359

theorem tan_square_B_eq_tan_A_tan_C_range (A B C : ℝ) (h_triangle : 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2 ∧ A + B + C = π) 
  (h_tan : Real.tan B * Real.tan B = Real.tan A * Real.tan C) : (π / 3) ≤ B ∧ B < (π / 2) :=
by
  sorry

end tan_square_B_eq_tan_A_tan_C_range_l1053_105359


namespace simplify_expr_l1053_105386

theorem simplify_expr (x y : ℝ) : 
  (3 * x - 2 * y - 4) * (x + y + 5) - (x + 2 * y + 5) * (3 * x - y - 1) = -4 * x * y - 3 * x - 7 * y - 15 := 
by 
  sorry

end simplify_expr_l1053_105386


namespace sum_of_consecutive_integers_of_sqrt3_l1053_105389

theorem sum_of_consecutive_integers_of_sqrt3 {a b : ℤ} (h1 : a + 1 = b) (h2 : (a : ℝ) < Real.sqrt 3) (h3 : Real.sqrt 3 < (b : ℝ)) :
  a + b = 3 := by
  sorry

end sum_of_consecutive_integers_of_sqrt3_l1053_105389


namespace sixDigitIntegersCount_l1053_105303

-- Define the digits to use.
def digits : List ℕ := [1, 2, 2, 5, 9, 9]

-- Define the factorial function as it might not be pre-defined in Mathlib.
def factorial : ℕ → ℕ
| 0 => 1
| (n + 1) => (n + 1) * factorial n

-- Calculate the number of unique permutations accounting for repeated digits.
def numberOfUniquePermutations : ℕ :=
  factorial 6 / (factorial 2 * factorial 2)

-- State the theorem proving that we can form exactly 180 unique six-digit integers.
theorem sixDigitIntegersCount : numberOfUniquePermutations = 180 :=
  sorry

end sixDigitIntegersCount_l1053_105303


namespace fraction_of_married_men_l1053_105381

/-- At a social gathering, there are only single women and married men with their wives.
     The probability that a randomly selected woman is single is 3/7.
     The fraction of the people in the gathering that are married men is 4/11. -/
theorem fraction_of_married_men (women : ℕ) (single_women : ℕ) (married_men : ℕ) (total_people : ℕ) 
  (h_women_total : women = 7)
  (h_single_women_probability : single_women = women * 3 / 7)
  (h_married_women : women - single_women = married_men)
  (h_total_people : total_people = women + married_men) :
  married_men / total_people = 4 / 11 := 
by sorry

end fraction_of_married_men_l1053_105381


namespace pool_balls_pyramid_arrangement_l1053_105340

/-- In how many distinguishable ways can 10 distinct pool balls be arranged in a pyramid
    (6 on the bottom, 3 in the middle, 1 on the top), assuming that all rotations of the pyramid are indistinguishable? -/
def pyramid_pool_balls_distinguishable_arrangements : Nat :=
  let total_arrangements := Nat.factorial 10
  let indistinguishable_rotations := 9
  total_arrangements / indistinguishable_rotations

theorem pool_balls_pyramid_arrangement :
  pyramid_pool_balls_distinguishable_arrangements = 403200 :=
by
  -- Proof will be added here
  sorry

end pool_balls_pyramid_arrangement_l1053_105340


namespace problem_statement_l1053_105308

def horse_lap_times : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

noncomputable def LCM (a b : ℕ) : ℕ := Nat.lcm a b

-- Least common multiple of a set of numbers
noncomputable def LCM_set (s : List ℕ) : ℕ :=
s.foldl LCM 1

-- Calculate the sum of the digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
n.digits 10 |>.sum

theorem problem_statement :
  let T := LCM_set [2, 3, 5, 7, 11, 13]
  sum_of_digits T = 6 := by
  sorry

end problem_statement_l1053_105308


namespace acute_angle_slope_neg_product_l1053_105334

   theorem acute_angle_slope_neg_product (a b c : ℝ) (h : a ≠ 0 ∧ b ≠ 0) (acute_inclination : ∃ (k : ℝ), k > 0 ∧ y = -a/b): (a * b < 0) :=
   by
     sorry
   
end acute_angle_slope_neg_product_l1053_105334


namespace parabola_directrix_l1053_105319

theorem parabola_directrix (a : ℝ) (h1 : ∀ x : ℝ, - (1 / (4 * a)) = 2):
  a = -(1 / 8) :=
sorry

end parabola_directrix_l1053_105319


namespace determine_m_l1053_105309

def f (x : ℝ) := 5 * x^2 + 3 * x + 7
def g (x : ℝ) (m : ℝ) := 2 * x^2 - m * x + 1

theorem determine_m (m : ℝ) : f 5 - g 5 m = 55 → m = -7 :=
by
  unfold f
  unfold g
  sorry

end determine_m_l1053_105309


namespace natural_solutions_3x_4y_eq_12_l1053_105332

theorem natural_solutions_3x_4y_eq_12 :
  ∃ x y : ℕ, (3 * x + 4 * y = 12) ∧ ((x = 4 ∧ y = 0) ∨ (x = 0 ∧ y = 3)) := 
sorry

end natural_solutions_3x_4y_eq_12_l1053_105332


namespace student_average_always_greater_l1053_105305

theorem student_average_always_greater (x y z : ℝ) (h1 : x < z) (h2 : z < y) :
  (B = (x + z + 2 * y) / 4) > (A = (x + y + z) / 3) := by
  sorry

end student_average_always_greater_l1053_105305


namespace largest_prime_mersenne_below_500_l1053_105361

def is_mersenne (m : ℕ) (n : ℕ) := m = 2^n - 1
def is_power_of_2 (n : ℕ) := ∃ (k : ℕ), n = 2^k

theorem largest_prime_mersenne_below_500 : ∀ (m : ℕ), 
  m < 500 →
  (∃ n, is_power_of_2 n ∧ is_mersenne m n ∧ Nat.Prime m) →
  m ≤ 3 := 
by
  sorry

end largest_prime_mersenne_below_500_l1053_105361


namespace cos_4_3pi_add_alpha_l1053_105385

theorem cos_4_3pi_add_alpha (α : ℝ) (h : Real.sin (Real.pi / 6 - α) = 1 / 3) :
    Real.cos (4 * Real.pi / 3 + α) = -1 / 3 := 
by sorry

end cos_4_3pi_add_alpha_l1053_105385


namespace coin_toss_probability_l1053_105399

-- Definition of the conditions
def total_outcomes : ℕ := 2 ^ 8
def favorable_outcomes : ℕ := Nat.choose 8 3
def probability : ℚ := favorable_outcomes / total_outcomes

-- The theorem to be proved
theorem coin_toss_probability : probability = 7 / 32 := by
  sorry

end coin_toss_probability_l1053_105399


namespace number_of_students_run_red_light_l1053_105328

theorem number_of_students_run_red_light :
  let total_students := 300
  let yes_responses := 90
  let odd_id_students := 75
  let coin_probability := 1/2
  -- Calculate using the conditions:
  total_students / 2 - odd_id_students / 2 * coin_probability + total_students / 2 * coin_probability = 30 :=
by
  sorry

end number_of_students_run_red_light_l1053_105328


namespace total_bananas_bought_l1053_105392

-- Define the conditions
def went_to_store_times : ℕ := 2
def bananas_per_trip : ℕ := 10

-- State the theorem/question and provide the answer
theorem total_bananas_bought : (went_to_store_times * bananas_per_trip) = 20 :=
by
  -- Proof here
  sorry

end total_bananas_bought_l1053_105392


namespace volume_of_given_solid_l1053_105322

noncomputable def volume_of_solid (s : ℝ) (h : ℝ) : ℝ :=
  (h / 3) * (s^2 + (s * (3 / 2))^2 + (s * (3 / 2)) * s)

theorem volume_of_given_solid : volume_of_solid 8 10 = 3040 / 3 :=
by
  sorry

end volume_of_given_solid_l1053_105322


namespace sequence_is_arithmetic_l1053_105321

-- Define a_n as a sequence in terms of n, where the formula is given.
def a_n (n : ℕ) : ℕ := 2 * n + 1

-- Theorem stating that the sequence is arithmetic with a common difference of 2.
theorem sequence_is_arithmetic : ∀ (n : ℕ), n > 0 → (a_n n) - (a_n (n - 1)) = 2 :=
by
  sorry

end sequence_is_arithmetic_l1053_105321


namespace sum_of_x_and_y_l1053_105367

theorem sum_of_x_and_y (x y : ℤ) (h : 2 * x * y + x + y = 83) : x + y = 83 ∨ x + y = -85 :=
sorry

end sum_of_x_and_y_l1053_105367


namespace abs_val_inequality_solution_l1053_105337

theorem abs_val_inequality_solution (x : ℝ) : |x - 2| + |x + 3| ≥ 4 ↔ x ≤ - (5 / 2) :=
by
  sorry

end abs_val_inequality_solution_l1053_105337


namespace value_of_work_clothes_l1053_105323

theorem value_of_work_clothes (x y : ℝ) (h1 : x + 70 = 30 * y) (h2 : x + 20 = 20 * y) : x = 80 :=
by
  sorry

end value_of_work_clothes_l1053_105323


namespace cos_neg_three_pi_over_two_eq_zero_l1053_105364

noncomputable def cos_neg_three_pi_over_two : ℝ :=
  Real.cos (-3 * Real.pi / 2)

theorem cos_neg_three_pi_over_two_eq_zero :
  cos_neg_three_pi_over_two = 0 :=
by
  -- Using trigonometric identities and periodicity of cosine function
  sorry

end cos_neg_three_pi_over_two_eq_zero_l1053_105364


namespace find_XY_square_l1053_105374

noncomputable def triangleABC := Type

variables (A B C T X Y : triangleABC)
variables (ω : Type) (BT CT BC TX TY XY : ℝ)

axiom acute_scalene_triangle (ABC : triangleABC) : Prop
axiom circumcircle (ABC: triangleABC) (ω: Type) : Prop
axiom tangents_intersect (ω: Type) (B C T: triangleABC) (BT CT : ℝ) : Prop
axiom projections (T: triangleABC) (X: triangleABC) (AB: triangleABC) (Y: triangleABC) (AC: triangleABC) : Prop

axiom BT_value : BT = 18
axiom CT_value : CT = 18
axiom BC_value : BC = 24
axiom TX_TY_XY_relation : TX^2 + TY^2 + XY^2 = 1450

theorem find_XY_square : XY^2 = 841 :=
by { sorry }

end find_XY_square_l1053_105374


namespace original_number_is_144_l1053_105325

theorem original_number_is_144 :
  ∃ (A B C : ℕ), A ≠ 0 ∧
  (100 * A + 11 * B = 144) ∧
  (A * B^2 = 10 * A + C) ∧
  (A * C = C) ∧
  A = 1 ∧ B = 4 ∧ C = 6 :=
by
  sorry

end original_number_is_144_l1053_105325


namespace alpha_more_economical_l1053_105318

theorem alpha_more_economical (n : ℕ) : n ≥ 12 → 80 + 12 * n < 10 + 18 * n := 
by
  sorry

end alpha_more_economical_l1053_105318


namespace fx_leq_one_l1053_105356

noncomputable def f (x : ℝ) : ℝ := (x + 1) / Real.exp x

theorem fx_leq_one : ∀ x : ℝ, f x ≤ 1 := by
  sorry

end fx_leq_one_l1053_105356


namespace roots_cubic_eq_l1053_105369

theorem roots_cubic_eq (r s p q : ℝ) (h1 : r + s = p) (h2 : r * s = q) :
    r^3 + s^3 = p^3 - 3 * q * p :=
by
    -- Placeholder for proof
    sorry

end roots_cubic_eq_l1053_105369


namespace count_three_digit_numbers_divisible_by_seventeen_l1053_105383

def is_three_digit_number (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def is_divisible_by_seventeen (n : ℕ) : Prop := n % 17 = 0

theorem count_three_digit_numbers_divisible_by_seventeen : 
  ∃ (count : ℕ), count = 53 ∧ 
    (∀ (n : ℕ), is_three_digit_number n → is_divisible_by_seventeen n → response) := 
sorry

end count_three_digit_numbers_divisible_by_seventeen_l1053_105383


namespace total_trees_correct_l1053_105398

def apricot_trees : ℕ := 58
def peach_trees : ℕ := 3 * apricot_trees
def total_trees : ℕ := apricot_trees + peach_trees

theorem total_trees_correct : total_trees = 232 :=
by
  sorry

end total_trees_correct_l1053_105398


namespace bulgarian_inequality_l1053_105379

theorem bulgarian_inequality (a b c d : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) (d_pos : 0 < d) :
    (a^4 / (a^3 + a^2 * b + a * b^2 + b^3) + 
     b^4 / (b^3 + b^2 * c + b * c^2 + c^3) + 
     c^4 / (c^3 + c^2 * d + c * d^2 + d^3) + 
     d^4 / (d^3 + d^2 * a + d * a^2 + a^3)) 
    ≥ (a + b + c + d) / 4 :=
sorry

end bulgarian_inequality_l1053_105379


namespace find_parallel_lines_a_l1053_105300

/--
Given two lines \(l_1\): \(x + 2y - 3 = 0\) and \(l_2\): \(2x - ay + 3 = 0\),
prove that if the lines are parallel, then \(a = -4\).
-/
theorem find_parallel_lines_a (a : ℝ) :
  (∀ (x y : ℝ), x + 2*y - 3 = 0) 
  → (∀ (x y : ℝ), 2*x - a*y + 3 = 0)
  → (-1 / 2 = 2 / -a) 
  → a = -4 :=
by
  intros
  sorry

end find_parallel_lines_a_l1053_105300


namespace exists_square_in_interval_l1053_105378

def x_k (k : ℕ) : ℕ := k * (k + 1) / 2

noncomputable def sum_x (n : ℕ) : ℕ := (List.range n).map x_k |>.sum

theorem exists_square_in_interval (n : ℕ) (hn : n ≥ 10) :
  ∃ m, (sum_x n - x_k n ≤ m^2 ∧ m^2 ≤ sum_x n) :=
by sorry

end exists_square_in_interval_l1053_105378


namespace ratio_of_numbers_l1053_105360

theorem ratio_of_numbers (a b : ℝ) (h1 : 0 < b) (h2 : b < a) 
  (h3 : 2 * ((a + b) / 2) = Real.sqrt (10 * a * b)) : abs (a / b - 8) < 1 :=
by
  sorry

end ratio_of_numbers_l1053_105360


namespace find_diameters_l1053_105343

theorem find_diameters (x y z : ℕ) (hx : x ≠ y) (hy : y ≠ z) (hz : x ≠ z) :
  x + y + z = 26 ∧ x^2 + y^2 + z^2 = 338 :=
  sorry

end find_diameters_l1053_105343


namespace mira_weekly_distance_l1053_105372

noncomputable def total_distance_jogging : ℝ :=
  let monday_distance := 4 * 2
  let thursday_distance := 5 * 1.5
  monday_distance + thursday_distance

noncomputable def total_distance_swimming : ℝ :=
  2 * 1

noncomputable def total_distance_cycling : ℝ :=
  12 * 1

noncomputable def total_distance : ℝ :=
  total_distance_jogging + total_distance_swimming + total_distance_cycling

theorem mira_weekly_distance : total_distance = 29.5 := by
  unfold total_distance
  unfold total_distance_jogging
  unfold total_distance_swimming
  unfold total_distance_cycling
  sorry

end mira_weekly_distance_l1053_105372


namespace find_n_l1053_105344

theorem find_n (n : ℕ) (h1 : Nat.lcm n 16 = 48) (h2 : Nat.gcd n 16 = 8): n = 24 := by
  sorry

end find_n_l1053_105344


namespace crow_eating_time_l1053_105346

/-- 
We are given that a crow eats a fifth of the total number of nuts in 6 hours.
We are to prove that it will take the crow 7.5 hours to finish a quarter of the nuts.
-/
theorem crow_eating_time (h : (1/5:ℚ) * t = 6) : (1/4) * t = 7.5 := 
by 
  -- Skipping the proof
  sorry

end crow_eating_time_l1053_105346


namespace sourball_candies_division_l1053_105310

theorem sourball_candies_division (N J L : ℕ) (total_candies : ℕ) (remaining_candies : ℕ) :
  N = 12 →
  J = N / 2 →
  L = J - 3 →
  total_candies = 30 →
  remaining_candies = total_candies - (N + J + L) →
  (remaining_candies / 3) = 3 :=
by 
  sorry

end sourball_candies_division_l1053_105310


namespace reciprocal_neg_3_div_4_l1053_105362

theorem reciprocal_neg_3_div_4 : (- (3 / 4 : ℚ))⁻¹ = -(4 / 3 : ℚ) :=
by
  sorry

end reciprocal_neg_3_div_4_l1053_105362


namespace find_k_max_product_l1053_105375

theorem find_k_max_product : 
  (∃ k : ℝ, (3 : ℝ) * (x ^ 2) - 4 * x + k = 0 ∧ 16 - 12 * k ≥ 0 ∧ (∀ x1 x2 : ℝ, x1 * x2 = k / 3 → x1 + x2 = 4 / 3 → x1 * x2 ≤ (2 / 3) ^ 2)) →
  k = 4 / 3 :=
by 
  sorry

end find_k_max_product_l1053_105375


namespace dasha_flags_proof_l1053_105380

variable (Tata_flags_right Yasha_flags_right Vera_flags_right Maxim_flags_right : ℕ)
variable (Total_flags : ℕ)

theorem dasha_flags_proof 
  (hTata: Tata_flags_right = 14)
  (hYasha: Yasha_flags_right = 32)
  (hVera: Vera_flags_right = 20)
  (hMaxim: Maxim_flags_right = 8)
  (hTotal: Total_flags = 37) :
  ∃ (Dasha_flags : ℕ), Dasha_flags = 8 :=
by
  sorry

end dasha_flags_proof_l1053_105380


namespace sharks_at_newport_l1053_105324

theorem sharks_at_newport :
  ∃ (x : ℕ), (∃ (y : ℕ), y = 4 * x ∧ x + y = 110) ∧ x = 22 :=
by {
  sorry
}

end sharks_at_newport_l1053_105324


namespace johns_climb_height_correct_l1053_105317

noncomputable def johns_total_height : ℝ :=
  let stair1_height := 4 * 15
  let stair2_height := 5 * 12.5
  let total_stair_height := stair1_height + stair2_height
  let rope1_height := (2 / 3) * stair1_height
  let rope2_height := (3 / 5) * stair2_height
  let total_rope_height := rope1_height + rope2_height
  let rope1_height_m := rope1_height / 3.281
  let rope2_height_m := rope2_height / 3.281
  let total_rope_height_m := rope1_height_m + rope2_height_m
  let ladder_height := 1.5 * total_rope_height_m * 3.281
  let rock_wall_height := (2 / 3) * ladder_height
  let total_pre_tree := total_stair_height + total_rope_height + ladder_height + rock_wall_height
  let tree_height := (3 / 4) * total_pre_tree - 10
  total_stair_height + total_rope_height + ladder_height + rock_wall_height + tree_height

theorem johns_climb_height_correct : johns_total_height = 679.115 := by
  sorry

end johns_climb_height_correct_l1053_105317


namespace points_per_touchdown_l1053_105350

theorem points_per_touchdown (number_of_touchdowns : ℕ) (total_points : ℕ) (h1 : number_of_touchdowns = 3) (h2 : total_points = 21) : (total_points / number_of_touchdowns) = 7 :=
by
  sorry

end points_per_touchdown_l1053_105350


namespace a2_a8_sum_l1053_105394

variable {a : ℕ → ℝ}  -- Define the arithmetic sequence a

-- Conditions:
axiom arithmetic_sequence (n : ℕ) : a (n + 1) - a n = a 1 - a 0
axiom a1_a9_sum : a 1 + a 9 = 8

-- Theorem stating the question and the answer
theorem a2_a8_sum : a 2 + a 8 = 8 :=
by
  sorry

end a2_a8_sum_l1053_105394


namespace find_denominator_l1053_105352

noncomputable def original_denominator (d : ℝ) : Prop :=
  (7 / (d + 3)) = 2 / 3

theorem find_denominator : ∃ d : ℝ, original_denominator d ∧ d = 7.5 :=
by
  use 7.5
  unfold original_denominator
  sorry

end find_denominator_l1053_105352


namespace value_of_expression_l1053_105333

theorem value_of_expression (x : ℤ) (h : x ^ 2 = 2209) : (x + 2) * (x - 2) = 2205 := 
by
  -- the proof goes here
  sorry

end value_of_expression_l1053_105333


namespace prove_ordered_triple_l1053_105391

theorem prove_ordered_triple (x y z : ℝ) (h1 : x > 2) (h2 : y > 2) (h3 : z > 2)
  (h4 : (x + 3)^2 / (y + z - 3) + (y + 5)^2 / (z + x - 5) + (z + 7)^2 / (x + y - 7) = 45) : 
  (x, y, z) = (13, 11, 6) :=
sorry

end prove_ordered_triple_l1053_105391


namespace waiting_period_l1053_105348

-- Variable declarations
variables (P : ℕ) (H : ℕ) (W : ℕ) (A : ℕ) (T : ℕ)
-- Condition declarations
variables (hp : P = 3) (hh : H = 5 * P) (ha : A = 3 * 7) (ht : T = 39)
-- Total time equation
variables (h_total : P + H + W + A = T)

-- Statement to prove
theorem waiting_period (hp : P = 3) (hh : H = 5 * P) (ha : A = 3 * 7) (ht : T = 39) (h_total : P + H + W + A = T) : 
  W = 3 :=
sorry

end waiting_period_l1053_105348


namespace measure_of_B_l1053_105368

theorem measure_of_B (A B C : ℝ) (h1 : B = A + 20) (h2 : C = 50) (h3 : A + B + C = 180) : B = 75 := by
  sorry

end measure_of_B_l1053_105368


namespace kolya_pays_90_rubles_l1053_105382

theorem kolya_pays_90_rubles {x y : ℝ} 
  (h1 : x + 3 * y = 78) 
  (h2 : x + 8 * y = 108) :
  x + 5 * y = 90 :=
by sorry

end kolya_pays_90_rubles_l1053_105382


namespace negate_universal_proposition_l1053_105353

theorem negate_universal_proposition : 
  (¬ (∀ x : ℝ, x^2 - 2 * x + 1 > 0)) ↔ (∃ x : ℝ, x^2 - 2 * x + 1 ≤ 0) :=
by sorry

end negate_universal_proposition_l1053_105353


namespace add_to_frac_eq_l1053_105312

theorem add_to_frac_eq {n : ℚ} (h : (4 + n) / (7 + n) = 7 / 9) : n = 13 / 2 :=
by 
  sorry

end add_to_frac_eq_l1053_105312


namespace abs_eq_two_implies_l1053_105377

theorem abs_eq_two_implies (x : ℝ) (h : |x - 3| = 2) : x = 5 ∨ x = 1 := 
sorry

end abs_eq_two_implies_l1053_105377


namespace total_surface_area_correct_l1053_105363

def surface_area_calculation (height_e height_f height_g : ℚ) : ℚ :=
  let top_bottom_area := 4
  let side_area := (height_e + height_f + height_g) * 2
  let front_back_area := 4
  top_bottom_area + side_area + front_back_area

theorem total_surface_area_correct :
  surface_area_calculation (5 / 8) (1 / 4) (9 / 8) = 12 := 
by
  sorry

end total_surface_area_correct_l1053_105363


namespace right_triangle_with_a_as_hypotenuse_l1053_105354

theorem right_triangle_with_a_as_hypotenuse
  (a b c : ℝ)
  (A B C : ℝ)
  (h1 : a = (b^2 + c^2 - a^2) / (2 * b * c))
  (h2 : b = (a^2 + c^2 - b^2) / (2 * a * c))
  (h3 : c = (a^2 + b^2 - c^2) / (2 * a * b))
  (h4 : a * ((b^2 + c^2 - a^2) / (2 * b * c)) + b * ((a^2 + c^2 - b^2) / (2 * a * c)) = c * ((a^2 + b^2 - c^2) / (2 * a * b))) :
  a^2 = b^2 + c^2 :=
by
  sorry

end right_triangle_with_a_as_hypotenuse_l1053_105354


namespace polygon_sides_l1053_105358

theorem polygon_sides (n : ℕ) (h : 44 = n * (n - 3) / 2) : n = 11 :=
sorry

end polygon_sides_l1053_105358


namespace major_axis_length_l1053_105397

theorem major_axis_length (x y : ℝ) (h : 16 * x^2 + 9 * y^2 = 144) : 8 = 8 :=
by
  sorry

end major_axis_length_l1053_105397


namespace smallest_possible_gcd_l1053_105320

noncomputable def smallestGCD (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : Nat.gcd a b = 9) : ℕ :=
  Nat.gcd (12 * a) (18 * b)

theorem smallest_possible_gcd (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : Nat.gcd a b = 9) : 
  smallestGCD a b h1 h2 h3 = 54 :=
sorry

end smallest_possible_gcd_l1053_105320


namespace remainder_when_divided_by_5_l1053_105302

-- Definitions of the conditions
def condition1 (N : ℤ) : Prop := ∃ R1 : ℤ, N = 5 * 2 + R1
def condition2 (N : ℤ) : Prop := ∃ Q2 : ℤ, N = 4 * Q2 + 2

-- Statement to prove
theorem remainder_when_divided_by_5 (N : ℤ) (R1 : ℤ) (Q2 : ℤ) :
  (N = 5 * 2 + R1) ∧ (N = 4 * Q2 + 2) → (R1 = 4) :=
by
  sorry

end remainder_when_divided_by_5_l1053_105302


namespace right_triangle_count_l1053_105390

theorem right_triangle_count :
  ∃! (a b : ℕ), (a^2 + b^2 = (b + 3)^2) ∧ (b < 50) :=
by
  sorry

end right_triangle_count_l1053_105390


namespace problem_statement_l1053_105357

variables {x y z w p q : Prop}

theorem problem_statement (h1 : x = y → z ≠ w) (h2 : z = w → p ≠ q) : x ≠ y → p ≠ q :=
by
  sorry

end problem_statement_l1053_105357


namespace charity_distribution_l1053_105301

theorem charity_distribution
    (amount_raised : ℝ)
    (donation_percentage : ℝ)
    (num_organizations : ℕ)
    (h_amount_raised : amount_raised = 2500)
    (h_donation_percentage : donation_percentage = 0.80)
    (h_num_organizations : num_organizations = 8) :
    (amount_raised * donation_percentage) / num_organizations = 250 := by
  sorry

end charity_distribution_l1053_105301


namespace matt_current_age_is_65_l1053_105384

variable (matt_age james_age : ℕ)

def james_current_age := 30
def james_age_in_5_years := james_current_age + 5
def matt_age_in_5_years := 2 * james_age_in_5_years
def matt_current_age := matt_age_in_5_years - 5

theorem matt_current_age_is_65 : matt_current_age = 65 := 
by
  -- sorry is here to skip the proof.
  sorry

end matt_current_age_is_65_l1053_105384


namespace simplify_fraction_subtraction_l1053_105313

theorem simplify_fraction_subtraction : (1 / 210) - (17 / 35) = -101 / 210 := by
  sorry

end simplify_fraction_subtraction_l1053_105313


namespace min_abs_sum_l1053_105395

theorem min_abs_sum (x : ℝ) : ∃ x : ℝ, (∀ y, abs (y + 3) + abs (y - 2) ≥ abs (x + 3) + abs (x - 2)) ∧ (abs (x + 3) + abs (x - 2) = 5) := sorry

end min_abs_sum_l1053_105395


namespace shuai_fen_ratio_l1053_105355

theorem shuai_fen_ratio 
  (C : ℕ) (B_and_D : ℕ) (a : ℕ) (x : ℚ) 
  (hC : C = 36) (hB_and_D : B_and_D = 75) :
  (x = 0.25) ∧ (a = 175) := 
by {
  -- This is where the proof steps would go
  sorry
}

end shuai_fen_ratio_l1053_105355


namespace Tina_profit_correct_l1053_105341

theorem Tina_profit_correct :
  ∀ (price_per_book cost_per_book books_per_customer total_customers : ℕ),
  price_per_book = 20 →
  cost_per_book = 5 →
  books_per_customer = 2 →
  total_customers = 4 →
  (price_per_book * (books_per_customer * total_customers) - 
   cost_per_book * (books_per_customer * total_customers) = 120) :=
by
  intros price_per_book cost_per_book books_per_customer total_customers
  sorry

end Tina_profit_correct_l1053_105341


namespace equation_solution_l1053_105304

theorem equation_solution (x : ℤ) (h : x + 1 = 2) : x = 1 :=
sorry

end equation_solution_l1053_105304


namespace cerulean_survey_l1053_105307

theorem cerulean_survey :
  let total_people := 120
  let kind_of_blue := 80
  let kind_and_green := 35
  let neither := 20
  total_people = kind_of_blue + (total_people - kind_of_blue - neither)
  → (kind_and_green + (total_people - kind_of_blue - kind_and_green - neither) + neither) = total_people
  → 55 = (kind_and_green + (total_people - kind_of_blue - kind_and_green - neither)) :=
by
  sorry

end cerulean_survey_l1053_105307


namespace b_remainder_l1053_105342

theorem b_remainder (n : ℕ) (hn : n > 0) : ∃ b : ℕ, b % 11 = 5 :=
by
  sorry

end b_remainder_l1053_105342


namespace cookies_per_bag_l1053_105339

-- Definitions based on given conditions
def total_cookies : ℕ := 75
def number_of_bags : ℕ := 25

-- The statement of the problem
theorem cookies_per_bag : total_cookies / number_of_bags = 3 := by
  sorry

end cookies_per_bag_l1053_105339
