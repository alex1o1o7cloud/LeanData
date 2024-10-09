import Mathlib

namespace find_y_intercept_l2098_209833

theorem find_y_intercept (m : ℝ) (x_intercept: ℝ × ℝ) : (x_intercept.snd = 0) → (x_intercept = (-4, 0)) → m = 3 → (0, m * 4 - m * (-4)) = (0, 12) :=
by
  sorry

end find_y_intercept_l2098_209833


namespace no_real_solutions_l2098_209861

theorem no_real_solutions :
  ∀ y : ℝ, ( (-2 * y + 7)^2 + 2 = -2 * |y| ) → false := by
  sorry

end no_real_solutions_l2098_209861


namespace units_digit_of_expression_l2098_209802

theorem units_digit_of_expression :
  (9 * 19 * 1989 - 9 ^ 3) % 10 = 0 :=
by
  sorry

end units_digit_of_expression_l2098_209802


namespace sum_radical_conjugates_l2098_209825

theorem sum_radical_conjugates : (5 - Real.sqrt 500) + (5 + Real.sqrt 500) = 10 :=
by
  sorry

end sum_radical_conjugates_l2098_209825


namespace real_solutions_count_l2098_209852

theorem real_solutions_count :
  ∃ n : ℕ, n = 2 ∧ ∀ x : ℝ, |x + 1| = |x - 3| + |x - 4| → x = 2 ∨ x = 8 :=
by
  sorry

end real_solutions_count_l2098_209852


namespace find_n_l2098_209864

theorem find_n (n : ℤ) (h1 : -90 ≤ n) (h2 : n ≤ 90) (h3 : Real.sin (n * Real.pi / 180) = Real.sin (782 * Real.pi / 180)) :
  n = 62 ∨ n = -62 := 
sorry

end find_n_l2098_209864


namespace percentage_decrease_is_24_l2098_209893

-- Define the given constants Rs. 820 and Rs. 1078.95
def current_price : ℝ := 820
def original_price : ℝ := 1078.95

-- Define the percentage decrease P
def percentage_decrease (P : ℝ) : Prop :=
  original_price - (P / 100) * original_price = current_price

-- Prove that percentage decrease P is approximately 24
theorem percentage_decrease_is_24 : percentage_decrease 24 :=
by
  unfold percentage_decrease
  sorry

end percentage_decrease_is_24_l2098_209893


namespace archibald_percentage_wins_l2098_209847

def archibald_wins : ℕ := 12
def brother_wins : ℕ := 18
def total_games_played : ℕ := archibald_wins + brother_wins

def percentage_archibald_wins : ℚ := (archibald_wins : ℚ) / (total_games_played : ℚ) * 100

theorem archibald_percentage_wins : percentage_archibald_wins = 40 := by
  sorry

end archibald_percentage_wins_l2098_209847


namespace steve_halfway_longer_than_danny_l2098_209812

theorem steve_halfway_longer_than_danny :
  let T_d : Float := 31
  let T_s : Float := 2 * T_d
  (T_s / 2) - (T_d / 2) = 15.5 :=
by
  let T_d : Float := 31
  let T_s : Float := 2 * T_d
  show (T_s / 2) - (T_d / 2) = 15.5
  sorry

end steve_halfway_longer_than_danny_l2098_209812


namespace value_of_k_l2098_209836

theorem value_of_k :
  ∀ (x k : ℝ), (x + 6) * (x - 5) = x^2 + k * x - 30 → k = 1 :=
by
  intros x k h
  sorry

end value_of_k_l2098_209836


namespace star_calculation_l2098_209814

-- Define the operation '*' via the given table
def star_table : Matrix (Fin 5) (Fin 5) (Fin 5) :=
  ![
    ![0, 1, 2, 3, 4],
    ![1, 0, 4, 2, 3],
    ![2, 3, 1, 4, 0],
    ![3, 4, 0, 1, 2],
    ![4, 2, 3, 0, 1]
  ]

def star (a b : Fin 5) : Fin 5 := star_table a b

-- Prove (3 * 5) * (2 * 4) = 3
theorem star_calculation : star (star 2 4) (star 4 1) = 2 := by
  sorry

end star_calculation_l2098_209814


namespace max_matching_pairs_l2098_209823

theorem max_matching_pairs (total_pairs : ℕ) (lost_individual : ℕ) (left_pair : ℕ) : 
  total_pairs = 25 ∧ lost_individual = 9 → left_pair = 20 :=
by
  sorry

end max_matching_pairs_l2098_209823


namespace find_number_l2098_209834

theorem find_number :
  ∃ x : ℕ, (x / 5 = 80 + x / 6) ∧ x = 2400 := 
by 
  sorry

end find_number_l2098_209834


namespace Mikaela_initially_planned_walls_l2098_209810

/-- 
Mikaela bought 16 containers of paint to cover a certain number of equally-sized walls in her bathroom.
At the last minute, she decided to put tile on one wall and paint flowers on the ceiling with one 
container of paint instead. She had 3 containers of paint left over. 
Prove she initially planned to paint 13 walls.
-/
theorem Mikaela_initially_planned_walls
  (PaintContainers : ℕ)
  (CeilingPaint : ℕ)
  (LeftOverPaint : ℕ)
  (TiledWalls : ℕ) : PaintContainers = 16 → CeilingPaint = 1 → LeftOverPaint = 3 → TiledWalls = 1 → 
    (PaintContainers - CeilingPaint - LeftOverPaint + TiledWalls = 13) :=
by
  -- Given conditions:
  intros h1 h2 h3 h4
  -- Proof goes here.
  sorry

end Mikaela_initially_planned_walls_l2098_209810


namespace papi_calot_additional_plants_l2098_209896

def initial_plants := 7 * 18

def total_plants := 141

def additional_plants := total_plants - initial_plants

theorem papi_calot_additional_plants : additional_plants = 15 :=
by
  sorry

end papi_calot_additional_plants_l2098_209896


namespace remainder_when_divided_by_13_l2098_209888

theorem remainder_when_divided_by_13 (N : ℤ) (k : ℤ) (h : N = 39 * k + 17) : 
  N % 13 = 4 :=
by
  sorry

end remainder_when_divided_by_13_l2098_209888


namespace three_digit_odd_nums_using_1_2_3_4_5_without_repetition_l2098_209878

def three_digit_odd_nums (digits : Finset ℕ) : ℕ :=
  let odd_digits := digits.filter (λ n => n % 2 = 1)
  let num_choices_for_units_place := odd_digits.card
  let remaining_digits := digits \ odd_digits
  let num_choices_for_hundreds_tens_places := remaining_digits.card * (remaining_digits.card - 1)
  num_choices_for_units_place * num_choices_for_hundreds_tens_places

theorem three_digit_odd_nums_using_1_2_3_4_5_without_repetition :
  three_digit_odd_nums {1, 2, 3, 4, 5} = 36 :=
by
  -- Proof is skipped
  sorry

end three_digit_odd_nums_using_1_2_3_4_5_without_repetition_l2098_209878


namespace Elizabeth_More_Revenue_Than_Banks_l2098_209865

theorem Elizabeth_More_Revenue_Than_Banks : 
  let banks_investments := 8
  let banks_revenue_per_investment := 500
  let elizabeth_investments := 5
  let elizabeth_revenue_per_investment := 900
  let banks_total_revenue := banks_investments * banks_revenue_per_investment
  let elizabeth_total_revenue := elizabeth_investments * elizabeth_revenue_per_investment
  elizabeth_total_revenue - banks_total_revenue = 500 :=
by
  sorry

end Elizabeth_More_Revenue_Than_Banks_l2098_209865


namespace average_student_headcount_l2098_209815

theorem average_student_headcount : 
  let headcount_03_04 := 11500
  let headcount_04_05 := 11600
  let headcount_05_06 := 11300
  (headcount_03_04 + headcount_04_05 + headcount_05_06) / 3 = 11467 :=
by
  sorry

end average_student_headcount_l2098_209815


namespace semicircle_arc_length_l2098_209885

theorem semicircle_arc_length (a b : ℝ) (hypotenuse_sum : a + b = 70) (a_eq_30 : a = 30) (b_eq_40 : b = 40) :
  ∃ (R : ℝ), (R = 24) ∧ (π * R = 12 * π) :=
by
  sorry

end semicircle_arc_length_l2098_209885


namespace xu_jun_age_l2098_209883

variable (x y : ℕ)

def condition1 : Prop := y - 2 = 3 * (x - 2)
def condition2 : Prop := y + 8 = 2 * (x + 8)

theorem xu_jun_age (h1 : condition1 x y) (h2 : condition2 x y) : x = 12 :=
by 
sorry

end xu_jun_age_l2098_209883


namespace solve_equation_1_solve_equation_2_l2098_209853

theorem solve_equation_1 :
  ∀ x : ℝ, 3 * x - 5 = 6 * x - 8 → x = 1 :=
by
  intro x
  intro h
  sorry

theorem solve_equation_2 :
  ∀ x : ℝ, (x + 1) / 2 - (2 * x - 1) / 3 = 1 → x = -1 :=
by
  intro x
  intro h
  sorry

end solve_equation_1_solve_equation_2_l2098_209853


namespace solution_set_of_inequality_l2098_209811

theorem solution_set_of_inequality (x : ℝ) : (x - 1) * (2 - x) > 0 ↔ 1 < x ∧ x < 2 := by
  sorry

end solution_set_of_inequality_l2098_209811


namespace find_two_digit_number_l2098_209846

theorem find_two_digit_number
  (X : ℕ)
  (h1 : 57 + (10 * X + 6) = 123)
  (h2 : two_digit_number = 10 * X + 9) :
  two_digit_number = 69 :=
by
  sorry

end find_two_digit_number_l2098_209846


namespace sally_balance_fraction_l2098_209832

variable (G : ℝ) (x : ℝ)
-- spending limit on gold card is G
-- spending limit on platinum card is 2G
-- Balance on platinum card is G/2
-- After transfer, 0.5833333333333334 portion of platinum card remains unspent

theorem sally_balance_fraction
  (h1 : (5/12) * 2 * G = G / 2 + x * G) : x = 1 / 3 :=
by
  sorry

end sally_balance_fraction_l2098_209832


namespace last_three_digits_of_7_pow_123_l2098_209850

theorem last_three_digits_of_7_pow_123 : 7^123 % 1000 = 773 := 
by sorry

end last_three_digits_of_7_pow_123_l2098_209850


namespace systematic_sampling_interval_l2098_209830

-- Definitions for the given conditions
def total_students : ℕ := 1203
def sample_size : ℕ := 40

-- Theorem statement to be proven
theorem systematic_sampling_interval (N n : ℕ) (hN : N = total_students) (hn : n = sample_size) : 
  N % n ≠ 0 → ∃ k : ℕ, k = 30 :=
by
  sorry

end systematic_sampling_interval_l2098_209830


namespace number_of_blocks_needed_to_form_cube_l2098_209856

-- Define the dimensions of the rectangular block
def block_length : ℕ := 5
def block_width : ℕ := 4
def block_height : ℕ := 3

-- Define the side length of the cube
def cube_side_length : ℕ := 60

-- The expected number of rectangular blocks needed
def expected_number_of_blocks : ℕ := 3600

-- Statement to prove the number of rectangular blocks needed to form the cube
theorem number_of_blocks_needed_to_form_cube
  (l : ℕ) (w : ℕ) (h : ℕ) (cube_side : ℕ) (expected_count : ℕ)
  (h_l : l = block_length)
  (h_w : w = block_width)
  (h_h : h = block_height)
  (h_cube_side : cube_side = cube_side_length)
  (h_expected : expected_count = expected_number_of_blocks) :
  (cube_side ^ 3) / (l * w * h) = expected_count :=
sorry

end number_of_blocks_needed_to_form_cube_l2098_209856


namespace add_fractions_11_12_7_15_l2098_209820

/-- A theorem stating that the sum of 11/12 and 7/15 is 83/60. -/
theorem add_fractions_11_12_7_15 : (11 / 12) + (7 / 15) = (83 / 60) := 
by
  sorry

end add_fractions_11_12_7_15_l2098_209820


namespace house_painting_l2098_209863

theorem house_painting (n : ℕ) (h1 : n = 1000)
  (occupants : Fin n → Fin n) (perm : ∀ i, occupants i ≠ i) :
  ∃ (coloring : Fin n → Fin 3), ∀ i, coloring i ≠ coloring (occupants i) :=
by
  sorry

end house_painting_l2098_209863


namespace lines_are_parallel_l2098_209838

theorem lines_are_parallel : 
  ∀ (x y : ℝ), (2 * x - y = 7) → (2 * x - y - 1 = 0) → False :=
by
  sorry  -- Proof will be filled in later

end lines_are_parallel_l2098_209838


namespace remaining_distance_proof_l2098_209804

/-
In a bicycle course with a total length of 10.5 kilometers (km), if Yoongi goes 1.5 kilometers (km) and then goes another 3730 meters (m), prove that the remaining distance of the course is 5270 meters.
-/

def km_to_m (km : ℝ) : ℝ := km * 1000

def total_course_length_km : ℝ := 10.5
def total_course_length_m : ℝ := km_to_m total_course_length_km

def yoongi_initial_distance_km : ℝ := 1.5
def yoongi_initial_distance_m : ℝ := km_to_m yoongi_initial_distance_km

def yoongi_additional_distance_m : ℝ := 3730

def yoongi_total_distance_m : ℝ := yoongi_initial_distance_m + yoongi_additional_distance_m

def remaining_distance_m (total_course_length_m yoongi_total_distance_m : ℝ) : ℝ :=
  total_course_length_m - yoongi_total_distance_m

theorem remaining_distance_proof : remaining_distance_m total_course_length_m yoongi_total_distance_m = 5270 := 
  sorry

end remaining_distance_proof_l2098_209804


namespace fencing_required_l2098_209835

theorem fencing_required (L W : ℕ) (area : ℕ) (hL : L = 20) (hA : area = 120) (hW : area = L * W) :
  2 * W + L = 32 :=
by
  -- Steps and proof logic to be provided here
  sorry

end fencing_required_l2098_209835


namespace find_positive_n_for_quadratic_l2098_209828

-- Define the discriminant of a quadratic equation
def discriminant (a b c : ℝ) : ℝ := b * b - 4 * a * c

-- Define the condition: the quadratic equation has exactly one real root if its discriminant is zero
def has_one_real_root (a b c : ℝ) : Prop := discriminant a b c = 0

-- The specific quadratic equation y^2 + 6ny + 9n
def my_quadratic (n : ℝ) : Prop := has_one_real_root 1 (6 * n) (9 * n)

-- The statement to be proven: for the quadratic equation y^2 + 6ny + 9n to have one real root, n must be 1
theorem find_positive_n_for_quadratic : ∃ (n : ℝ), my_quadratic n ∧ n > 0 ∧ n = 1 := 
by
  sorry

end find_positive_n_for_quadratic_l2098_209828


namespace sum_of_numbers_l2098_209859

theorem sum_of_numbers (a b c : ℝ) (h1 : a^2 + b^2 + c^2 = 222) (h2 : a * b + b * c + c * a = 131) : a + b + c = 22 :=
by
  sorry

end sum_of_numbers_l2098_209859


namespace minimum_distance_l2098_209898

noncomputable def point_on_curve (x : ℝ) : ℝ := -x^2 + 3 * Real.log x

noncomputable def point_on_line (x : ℝ) : ℝ := x + 2

theorem minimum_distance 
  (a b c d : ℝ) 
  (hP : b = point_on_curve a) 
  (hQ : d = point_on_line c) 
  : (a - c)^2 + (b - d)^2 = 8 :=
by
  sorry

end minimum_distance_l2098_209898


namespace intersection_M_N_l2098_209800

noncomputable def M : Set ℝ := { x | x^2 - x ≤ 0 }
noncomputable def N : Set ℝ := { x | 1 - abs x > 0 }
noncomputable def intersection : Set ℝ := { x | x ≥ 0 ∧ x < 1 }

theorem intersection_M_N : M ∩ N = intersection :=
by
  sorry

end intersection_M_N_l2098_209800


namespace employed_females_percentage_l2098_209871

def P_total : ℝ := 0.64
def P_males : ℝ := 0.46

theorem employed_females_percentage : 
  ((P_total - P_males) / P_total) * 100 = 28.125 :=
by
  sorry

end employed_females_percentage_l2098_209871


namespace find_x_from_conditions_l2098_209803

theorem find_x_from_conditions 
  (x y : ℕ) 
  (h1 : 1 ≤ x)
  (h2 : x ≤ 100)
  (h3 : 1 ≤ y)
  (h4 : y ≤ 100)
  (h5 : y > x)
  (h6 : (21 + 45 + 77 + 2 * x + y) / 6 = 2 * x) 
  : x = 16 := 
sorry

end find_x_from_conditions_l2098_209803


namespace minimize_theta_l2098_209816

theorem minimize_theta (K : ℤ) : ∃ θ : ℝ, -495 = K * 360 + θ ∧ |θ| ≤ 180 ∧ θ = -135 :=
by
  sorry

end minimize_theta_l2098_209816


namespace positive_integer_solution_l2098_209881

theorem positive_integer_solution (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (h : x^4 = y^2 + 71) :
  x = 6 ∧ y = 35 :=
by
  sorry

end positive_integer_solution_l2098_209881


namespace production_today_l2098_209806

def average_production (P : ℕ) (n : ℕ) := P / n

theorem production_today :
  ∀ (T P n : ℕ), n = 9 → average_production P n = 50 → average_production (P + T) (n + 1) = 54 → T = 90 :=
by
  intros T P n h1 h2 h3
  sorry

end production_today_l2098_209806


namespace min_value_of_function_l2098_209805

-- Define the function f
def f (x : ℝ) := 3 * x^2 - 6 * x + 9

-- State the theorem about the minimum value of the function.
theorem min_value_of_function : ∀ x : ℝ, f x ≥ 6 := by
  sorry

end min_value_of_function_l2098_209805


namespace lucky_ticket_N123456_l2098_209882

def digits : List ℕ := [1, 2, 3, 4, 5, 6]

def is_lucky (digits : List ℕ) : Prop :=
  ∃ f : ℕ → ℕ → ℕ, (f 1 (f (f 2 3) 4) * f 5 6) = 100

theorem lucky_ticket_N123456 : is_lucky digits :=
  sorry

end lucky_ticket_N123456_l2098_209882


namespace min_value_frac_sq_l2098_209875

theorem min_value_frac_sq (x : ℝ) (h : x > 12) : (x^2 / (x - 12)) >= 48 :=
by
  sorry

end min_value_frac_sq_l2098_209875


namespace wheel_moves_in_one_hour_l2098_209869

theorem wheel_moves_in_one_hour
  (rotations_per_minute : ℕ)
  (distance_per_rotation_cm : ℕ)
  (minutes_in_hour : ℕ) :
  rotations_per_minute = 20 →
  distance_per_rotation_cm = 35 →
  minutes_in_hour = 60 →
  let distance_per_rotation_m : ℚ := distance_per_rotation_cm / 100
  let total_rotations_per_hour : ℕ := rotations_per_minute * minutes_in_hour
  let total_distance_in_hour : ℚ := distance_per_rotation_m * total_rotations_per_hour
  total_distance_in_hour = 420 := by
  intros
  sorry

end wheel_moves_in_one_hour_l2098_209869


namespace total_votes_l2098_209843

theorem total_votes (V : ℝ) (h1 : 0.32 * V = 0.32 * V) (h2 : 0.32 * V + 1908 = 0.68 * V) : V = 5300 :=
by
  sorry

end total_votes_l2098_209843


namespace donny_total_cost_eq_45_l2098_209809

-- Definitions for prices of each type of apple
def price_small : ℝ := 1.5
def price_medium : ℝ := 2
def price_big : ℝ := 3

-- Quantities purchased by Donny
def count_small : ℕ := 6
def count_medium : ℕ := 6
def count_big : ℕ := 8

-- Total cost calculation
def total_cost (count_small count_medium count_big : ℕ) : ℝ := 
  (count_small * price_small) + (count_medium * price_medium) + (count_big * price_big)

-- Theorem stating the total cost
theorem donny_total_cost_eq_45 : total_cost count_small count_medium count_big = 45 := by
  sorry

end donny_total_cost_eq_45_l2098_209809


namespace number_exceeds_part_l2098_209841

theorem number_exceeds_part (x : ℝ) (h : x = (5 / 9) * x + 150) : x = 337.5 := sorry

end number_exceeds_part_l2098_209841


namespace points_scored_by_others_l2098_209862

-- Define the conditions as hypothesis
variables (P_total P_Jessie : ℕ)
  (H1 : P_total = 311)
  (H2 : P_Jessie = 41)
  (H3 : ∀ P_Lisa P_Devin: ℕ, P_Lisa = P_Jessie ∧ P_Devin = P_Jessie)

-- Define what we need to prove
theorem points_scored_by_others (P_others : ℕ) :
  P_total = 311 → P_Jessie = 41 → 
  (∀ P_Lisa P_Devin: ℕ, P_Lisa = P_Jessie ∧ P_Devin = P_Jessie) → 
  P_others = 188 :=
by
  sorry

end points_scored_by_others_l2098_209862


namespace highest_average_speed_interval_l2098_209895

theorem highest_average_speed_interval
  (d : ℕ → ℕ)
  (h0 : d 0 = 45)        -- Distance from 0 to 30 minutes
  (h1 : d 1 = 135)       -- Distance from 30 to 60 minutes
  (h2 : d 2 = 255)       -- Distance from 60 to 90 minutes
  (h3 : d 3 = 325) :     -- Distance from 90 to 120 minutes
  (1 / 2) * ((d 2 - d 1 : ℕ) : ℝ) > 
  max ((1 / 2) * ((d 1 - d 0 : ℕ) : ℝ)) 
      (max ((1 / 2) * ((d 3 - d 2 : ℕ) : ℝ))
          ((1 / 2) * ((d 3 - d 1 : ℕ) : ℝ))) :=
by
  sorry

end highest_average_speed_interval_l2098_209895


namespace number_of_maple_trees_planted_l2098_209876

def before := 53
def after := 64
def planted := after - before

theorem number_of_maple_trees_planted : planted = 11 := by
  sorry

end number_of_maple_trees_planted_l2098_209876


namespace sum_cos_4x_4y_4z_l2098_209844

theorem sum_cos_4x_4y_4z (x y z : ℝ)
  (h1 : Real.cos x + Real.cos y + Real.cos z = 0)
  (h2 : Real.sin x + Real.sin y + Real.sin z = 0) :
  Real.cos (4 * x) + Real.cos (4 * y) + Real.cos (4 * z) = 0 :=
by
  sorry

end sum_cos_4x_4y_4z_l2098_209844


namespace sum_is_correct_l2098_209880

-- Define the variables and conditions
variables (a b c d : ℝ)
variable (x : ℝ)

-- Define the condition
def condition : Prop :=
  a + 1 = x ∧
  b + 2 = x ∧
  c + 3 = x ∧
  d + 4 = x ∧
  a + b + c + d + 5 = x

-- The theorem we need to prove
theorem sum_is_correct (h : condition a b c d x) : a + b + c + d = -10 / 3 :=
  sorry

end sum_is_correct_l2098_209880


namespace compute_expression_l2098_209824

theorem compute_expression :
  (5 + 7)^2 + 5^2 + 7^2 = 218 :=
by
  sorry

end compute_expression_l2098_209824


namespace seq_identity_l2098_209821

-- Define the sequence (a_n)
def seq (a : ℕ → ℕ) : Prop :=
  a 0 = 0 ∧ a 1 = 0 ∧ a 2 = 1 ∧ ∀ n, a (n + 3) = a (n + 1) + 1998 * a n

theorem seq_identity (a : ℕ → ℕ) (h : seq a) (n : ℕ) (hn : 0 < n) :
  a (2 * n - 1) = 2 * a n * a (n + 1) + 1998 * (a (n - 1))^2 :=
sorry

end seq_identity_l2098_209821


namespace train_crossing_time_l2098_209808

-- Defining basic conditions
def train_length : ℕ := 150
def platform_length : ℕ := 100
def time_to_cross_post : ℕ := 15

-- The time it takes for the train to cross the platform
theorem train_crossing_time :
  (train_length + platform_length) / (train_length / time_to_cross_post) = 25 := 
sorry

end train_crossing_time_l2098_209808


namespace students_interested_in_both_l2098_209854

theorem students_interested_in_both (A B C Total : ℕ) (hA : A = 35) (hB : B = 45) (hC : C = 4) (hTotal : Total = 55) :
  A + B - 29 + C = Total :=
by
  -- Assuming the correct answer directly while skipping the proof.
  sorry

end students_interested_in_both_l2098_209854


namespace problem_l2098_209890

def op (x y : ℝ) : ℝ := x^2 + y^3

theorem problem (k : ℝ) : op k (op k k) = k^2 + k^6 + 6*k^7 + k^9 :=
by
  sorry

end problem_l2098_209890


namespace no_solutions_for_sin_cos_eq_sqrt3_l2098_209889

theorem no_solutions_for_sin_cos_eq_sqrt3 (x : ℝ) (hx : 0 ≤ x ∧ x < 2 * Real.pi) :
  ¬ (Real.sin x + Real.cos x = Real.sqrt 3) :=
by
  sorry

end no_solutions_for_sin_cos_eq_sqrt3_l2098_209889


namespace correctly_calculated_expression_l2098_209842

theorem correctly_calculated_expression (x : ℝ) :
  ¬ (x^3 + x^2 = x^5) ∧ 
  ¬ (x^3 * x^2 = x^6) ∧ 
  (x^3 / x^2 = x) ∧ 
  ¬ ((x^3)^2 = x^9) := by
sorry

end correctly_calculated_expression_l2098_209842


namespace vector_calc_l2098_209818

def vec1 : ℝ × ℝ := (5, -8)
def vec2 : ℝ × ℝ := (2, 6)
def vec3 : ℝ × ℝ := (-1, 4)
def scalar : ℝ := 5

theorem vector_calc :
  (vec1.1 - scalar * vec2.1 + vec3.1, vec1.2 - scalar * vec2.2 + vec3.2) = (-6, -34) :=
sorry

end vector_calc_l2098_209818


namespace range_of_c_l2098_209827

variable {a b c : ℝ} -- Declare the variables

-- Define the conditions
def triangle_condition (a b : ℝ) : Prop :=
|a + b - 4| + (a - b + 2)^2 = 0

-- Define the proof problem
theorem range_of_c {a b c : ℝ} (h : triangle_condition a b) : 2 < c ∧ c < 4 :=
sorry -- Proof to be completed

end range_of_c_l2098_209827


namespace polygon_sides_l2098_209855

-- Define the conditions
def sum_interior_angles (x : ℕ) : ℝ := 180 * (x - 2)
def sum_given_angles (x : ℕ) : ℝ := 160 + 112 * (x - 1)

-- State the theorem
theorem polygon_sides (x : ℕ) (h : sum_interior_angles x = sum_given_angles x) : x = 6 := by
  sorry

end polygon_sides_l2098_209855


namespace miguel_socks_probability_l2098_209867

theorem miguel_socks_probability :
  let total_socks := 10
  let socks_per_color := 2
  let colors := 5
  let draw_socks := 5
  let total_combinations := Nat.choose total_socks draw_socks
  let desired_combinations :=
    (Nat.choose colors 2) * (Nat.choose (colors - 2 + 1) 1) * socks_per_color
  let probability := desired_combinations / total_combinations
  probability = 5 / 21 :=
by
  let total_socks := 10
  let socks_per_color := 2
  let colors := 5
  let draw_socks := 5
  let total_combinations := Nat.choose total_socks draw_socks
  let desired_combinations :=
    (Nat.choose colors 2) * (Nat.choose (colors - 2 + 1) 1) * socks_per_color
  let probability := desired_combinations / total_combinations
  sorry

end miguel_socks_probability_l2098_209867


namespace transformed_parabola_eq_l2098_209831

noncomputable def initial_parabola (x : ℝ) : ℝ := 2 * (x - 1)^2 + 3
def shift_left (h : ℝ) (c : ℝ): ℝ := h - c
def shift_down (k : ℝ) (d : ℝ): ℝ := k - d

theorem transformed_parabola_eq :
  ∃ (x : ℝ), (initial_parabola (shift_left x 2) - 1 = 2 * (x + 1)^2 + 2) :=
sorry

end transformed_parabola_eq_l2098_209831


namespace second_integer_is_64_l2098_209891

theorem second_integer_is_64
  (n : ℤ)
  (h1 : (n - 2) + (n + 2) = 128) :
  n = 64 := 
  sorry

end second_integer_is_64_l2098_209891


namespace find_digits_l2098_209872

theorem find_digits (A B D E C : ℕ) 
  (hC : C = 9) 
  (hA : 2 < A ∧ A < 4)
  (hB : B = 5)
  (hE : E = 6)
  (hD : D = 0)
  (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ C ≠ D ∧ C ≠ E ∧ D ≠ E) :
  (A, B, D, E) = (3, 5, 0, 6) := by
  sorry

end find_digits_l2098_209872


namespace complex_identity_l2098_209874

variable (i : ℂ)
axiom i_squared : i^2 = -1

theorem complex_identity : 1 + i + i^2 = i :=
by sorry

end complex_identity_l2098_209874


namespace total_amount_paid_l2098_209887

def p1 := 20
def p2 := p1 + 2
def p3 := p2 + 3
def p4 := p3 + 4

theorem total_amount_paid : p1 + p2 + p3 + p4 = 96 :=
by
  sorry

end total_amount_paid_l2098_209887


namespace parallel_lines_condition_l2098_209879

theorem parallel_lines_condition (a : ℝ) : 
  (∀ x y : ℝ, ax + y + 1 = 0 ↔ x + ay - 1 = 0) ↔ (a = 1) :=
sorry

end parallel_lines_condition_l2098_209879


namespace lcm_gcf_ratio_120_504_l2098_209840

theorem lcm_gcf_ratio_120_504 : 
  let a := 120
  let b := 504
  (Int.lcm a b) / (Int.gcd a b) = 105 := by
  sorry

end lcm_gcf_ratio_120_504_l2098_209840


namespace count_distinct_reals_a_with_integer_roots_l2098_209822

-- Define the quadratic equation with its roots and conditions
theorem count_distinct_reals_a_with_integer_roots :
  ∃ (a_vals : Finset ℝ), a_vals.card = 6 ∧
    (∀ a ∈ a_vals, ∃ r s : ℤ, 
      (r + s : ℝ) = -a ∧ (r * s : ℝ) = 9 * a) :=
by
  sorry

end count_distinct_reals_a_with_integer_roots_l2098_209822


namespace room_width_to_perimeter_ratio_l2098_209868

theorem room_width_to_perimeter_ratio (L W : ℕ) (hL : L = 25) (hW : W = 15) :
  let P := 2 * (L + W)
  let ratio := W / P
  ratio = 3 / 16 :=
by
  sorry

end room_width_to_perimeter_ratio_l2098_209868


namespace ellen_smoothie_ingredients_l2098_209848

theorem ellen_smoothie_ingredients :
  let strawberries := 0.2
  let yogurt := 0.1
  let orange_juice := 0.2
  strawberries + yogurt + orange_juice = 0.5 :=
by
  sorry

end ellen_smoothie_ingredients_l2098_209848


namespace range_of_a_l2098_209817

noncomputable def in_range (a : ℝ) : Prop :=
  (0 < a ∧ a < 1) ∨ (a ≥ 1)

theorem range_of_a (a : ℝ) (p q : Prop) (h1 : p ↔ (0 < a ∧ a < 1)) (h2 : q ↔ (a ≥ 1 / 2)) (h3 : p ∨ q) (h4 : ¬ (p ∧ q)) :
  in_range a :=
by
  sorry

end range_of_a_l2098_209817


namespace largest_integer_not_greater_than_expr_l2098_209819

theorem largest_integer_not_greater_than_expr (x : ℝ) (hx : 20 * Real.sin x = 22 * Real.cos x) :
    ⌊(1 / (Real.sin x * Real.cos x) - 1)^7⌋ = 1 := 
sorry

end largest_integer_not_greater_than_expr_l2098_209819


namespace units_digit_29_pow_8_pow_7_l2098_209807

/-- The units digit of 29 raised to an arbitrary power follows a cyclical pattern. 
    For the purposes of this proof, we use that 29^k for even k ends in 1.
    Since 8^7 is even, we prove the units digit of 29^(8^7) is 1. -/
theorem units_digit_29_pow_8_pow_7 : (29^(8^7)) % 10 = 1 :=
by
  have even_power_cycle : ∀ k, k % 2 = 0 → (29^k) % 10 = 1 := sorry
  have eight_power_seven_even : (8^7) % 2 = 0 := by norm_num
  exact even_power_cycle (8^7) eight_power_seven_even

end units_digit_29_pow_8_pow_7_l2098_209807


namespace negation_of_proposition_l2098_209839

theorem negation_of_proposition :
  (¬ ∃ x_0 : ℝ, x_0^2 + x_0 - 1 > 0) ↔ (∀ x : ℝ, x^2 + x - 1 ≤ 0) :=
by
  sorry

end negation_of_proposition_l2098_209839


namespace math_problems_not_a_set_l2098_209866

-- Define the conditions in Lean
def is_well_defined (α : Type) : Prop := sorry

-- Type definitions for the groups of objects
def table_tennis_players : Type := sorry
def positive_integers_less_than_5 : Type := sorry
def irrational_numbers : Type := sorry
def math_problems_2023_college_exam : Type := sorry

-- Defining specific properties of each group
def well_defined_table_tennis_players : is_well_defined table_tennis_players := sorry
def well_defined_positive_integers_less_than_5 : is_well_defined positive_integers_less_than_5 := sorry
def well_defined_irrational_numbers : is_well_defined irrational_numbers := sorry

-- The key property that math problems from 2023 college entrance examination cannot form a set.
theorem math_problems_not_a_set : ¬ is_well_defined math_problems_2023_college_exam := sorry

end math_problems_not_a_set_l2098_209866


namespace paths_for_content_l2098_209860

def grid := [
  [none, none, none, none, none, none, some 'C', none, none, none, none, none, none, none],
  [none, none, none, none, none, some 'C', some 'O', some 'C', none, none, none, none, none, none],
  [none, none, none, none, some 'C', some 'O', some 'N', some 'O', some 'C', none, none, none, none, none],
  [none, none, none, some 'C', some 'O', some 'N', some 'T', some 'N', some 'O', some 'C', none, none, none, none],
  [none, none, some 'C', some 'O', some 'N', some 'T', some 'E', some 'T', some 'N', some 'O', some 'C', none, none, none],
  [none, some 'C', some 'O', some 'N', some 'T', some 'E', some 'N', some 'E', some 'T', some 'N', some 'O', some 'C', none, none],
  [some 'C', some 'O', some 'N', some 'T', some 'E', some 'N', some 'T', some 'N', some 'E', some 'T', some 'N', some 'O', some 'C']
]

def spelling_paths : Nat :=
  -- Skipping the actual calculation and providing the given total for now
  127

theorem paths_for_content : spelling_paths = 127 := sorry

end paths_for_content_l2098_209860


namespace floor_factorial_even_l2098_209897

theorem floor_factorial_even (n : ℕ) (hn : n > 0) : 
  Nat.floor ((Nat.factorial (n - 1) : ℝ) / (n * (n + 1))) % 2 = 0 := 
sorry

end floor_factorial_even_l2098_209897


namespace min_value_of_f_l2098_209877

noncomputable def f (x : ℝ) : ℝ := 3 * x + 12 / x ^ 2

theorem min_value_of_f : ∀ x > 0, f x ≥ 9 ∧ (f x = 9 ↔ x = 2) :=
by
  sorry

end min_value_of_f_l2098_209877


namespace ratio_girls_to_boys_l2098_209899

-- Definitions of the conditions
def numGirls : ℕ := 10
def numBoys : ℕ := 20

-- Statement of the proof problem
theorem ratio_girls_to_boys : (numGirls / Nat.gcd numGirls numBoys) = 1 ∧ (numBoys / Nat.gcd numGirls numBoys) = 2 :=
by
  sorry

end ratio_girls_to_boys_l2098_209899


namespace smallest_possible_number_of_apples_l2098_209845

theorem smallest_possible_number_of_apples :
  ∃ (M : ℕ), M > 2 ∧ M % 9 = 2 ∧ M % 10 = 2 ∧ M % 11 = 2 ∧ M = 200 :=
by
  sorry

end smallest_possible_number_of_apples_l2098_209845


namespace lazy_worker_days_worked_l2098_209894

theorem lazy_worker_days_worked :
  ∃ x : ℕ, 24 * x - 6 * (30 - x) = 0 ∧ x = 6 :=
by
  existsi 6
  sorry

end lazy_worker_days_worked_l2098_209894


namespace probability_exactly_three_heads_in_seven_tosses_l2098_209813

def combinations (n k : ℕ) : ℕ := Nat.choose n k

def binomial_probability (n k : ℕ) : ℚ :=
  (combinations n k) / (2^n : ℚ)

theorem probability_exactly_three_heads_in_seven_tosses :
  binomial_probability 7 3 = 35 / 128 := 
by 
  sorry

end probability_exactly_three_heads_in_seven_tosses_l2098_209813


namespace marcus_batches_l2098_209851

theorem marcus_batches (B : ℕ) : (5 * B = 35) ∧ (35 - 8 = 27) → B = 7 :=
by {
  sorry
}

end marcus_batches_l2098_209851


namespace antonov_packs_remaining_l2098_209892

theorem antonov_packs_remaining (total_candies : ℕ) (pack_size : ℕ) (packs_given : ℕ) (candies_remaining : ℕ) (packs_remaining : ℕ) :
  total_candies = 60 →
  pack_size = 20 →
  packs_given = 1 →
  candies_remaining = total_candies - pack_size * packs_given →
  packs_remaining = candies_remaining / pack_size →
  packs_remaining = 2 := by
  sorry

end antonov_packs_remaining_l2098_209892


namespace ratio_doubled_to_original_l2098_209873

theorem ratio_doubled_to_original (x y : ℕ) (h1 : y = 2 * x + 9) (h2 : 3 * y = 57) : 2 * x = 2 * (x / 1) := 
by sorry

end ratio_doubled_to_original_l2098_209873


namespace expression_evaluation_l2098_209826

noncomputable def x := Real.sqrt 5 + 1
noncomputable def y := Real.sqrt 5 - 1

theorem expression_evaluation : 
  ( ( (5 * x + 3 * y) / (x^2 - y^2) + (2 * x) / (y^2 - x^2) ) / (1 / (x^2 * y - x * y^2)) ) = 12 := 
by 
  -- Provide a proof here
  sorry

end expression_evaluation_l2098_209826


namespace largest_possible_n_l2098_209870

open Nat

-- Define arithmetic sequences a_n and b_n with given initial conditions
def arithmetic_seq (a_n : ℕ → ℕ) (b_n : ℕ → ℕ) :=
  a_n 1 = 1 ∧ b_n 1 = 1 ∧ 
  a_n 2 ≤ b_n 2 ∧
  (∃n : ℕ, a_n n * b_n n = 1764)

-- Given the arithmetic sequences defined above, prove that the largest possible value of n is 44
theorem largest_possible_n : 
  ∀ (a_n b_n : ℕ → ℕ), arithmetic_seq a_n b_n →
  ∀ (n : ℕ), (a_n n * b_n n = 1764) → n ≤ 44 :=
sorry

end largest_possible_n_l2098_209870


namespace part1_part2_l2098_209837

noncomputable def f (a x : ℝ) : ℝ := (a / 2) * x * x - (a - 2) * x - 2 * x * Real.log x
noncomputable def f' (a x : ℝ) : ℝ := a * x - a - 2 * Real.log x

theorem part1 (a : ℝ) : (∀ x > 0, f' a x ≥ 0) ↔ a = 2 :=
sorry

theorem part2 (a x1 x2 : ℝ) (h1 : 0 < a) (h2 : a < 2) (h3 : f' a x1 = 0) (h4 : f' a x2 = 0) (h5 : x1 < x2) : 
  x2 - x1 > 4 / a - 2 :=
sorry

end part1_part2_l2098_209837


namespace women_in_business_class_l2098_209849

theorem women_in_business_class 
  (total_passengers : ℕ) 
  (percent_women : ℝ) 
  (percent_women_in_business : ℝ) 
  (H1 : total_passengers = 300)
  (H2 : percent_women = 0.70)
  (H3 : percent_women_in_business = 0.08) : 
  ∃ (num_women_business_class : ℕ), num_women_business_class = 16 := 
by
  sorry

end women_in_business_class_l2098_209849


namespace restore_original_price_l2098_209886

def price_after_increases (p : ℝ) : ℝ :=
  let p1 := p * 1.10
  let p2 := p1 * 1.10
  let p3 := p2 * 1.05
  p3

theorem restore_original_price (p : ℝ) (h : p = 1) : 
  ∃ x : ℝ, x = 22 ∧ (price_after_increases p) * (1 - x / 100) = 1 := 
by 
  sorry

end restore_original_price_l2098_209886


namespace transformed_curve_l2098_209884

theorem transformed_curve :
  (∀ x y : ℝ, 3*x = x' ∧ 4*y = y' → x^2 + y^2 = 1) ↔ (x'^2 / 9 + y'^2 / 16 = 1) :=
by
  sorry

end transformed_curve_l2098_209884


namespace sqrt_of_9_eq_3_l2098_209858

theorem sqrt_of_9_eq_3 : Real.sqrt 9 = 3 := 
by 
  sorry

end sqrt_of_9_eq_3_l2098_209858


namespace find_natural_number_l2098_209857

-- Define the problem statement
def satisfies_condition (n : ℕ) : Prop :=
  ∃ k : ℕ, k > 0 ∧ (2 * n^2 - 2) = k * (n^3 - n)

-- The main theorem
theorem find_natural_number (n : ℕ) : satisfies_condition n ↔ n = 2 :=
sorry

end find_natural_number_l2098_209857


namespace smallest_cube_ends_in_584_l2098_209829

theorem smallest_cube_ends_in_584 (n : ℕ) : n^3 ≡ 584 [MOD 1000] → n = 34 := by
  sorry

end smallest_cube_ends_in_584_l2098_209829


namespace sum_of_sequences_l2098_209801

theorem sum_of_sequences :
  (1 + 11 + 21 + 31 + 41) + (9 + 19 + 29 + 39 + 49) = 250 := 
by 
  sorry

end sum_of_sequences_l2098_209801
