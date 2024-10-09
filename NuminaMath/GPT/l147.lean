import Mathlib

namespace running_time_l147_14721

variable (t : ℝ)
variable (v_j v_p d : ℝ)

-- Given conditions
variable (v_j : ℝ := 0.133333333333)  -- Joe's speed
variable (v_p : ℝ := 0.0666666666665) -- Pete's speed
variable (d : ℝ := 16)                -- Distance between them after t minutes

theorem running_time (h : v_j + v_p = 0.2 * t) : t = 80 :=
by
  -- Distance covered by Joe and Pete running in opposite directions
  have h1 : v_j * t + v_p * t = d := by sorry
  -- Given combined speeds
  have h2 : v_j + v_p = 0.2 := by sorry
  -- Using the equation to solve for time t
  exact sorry

end running_time_l147_14721


namespace circle_radius_l147_14715

-- Given conditions
def central_angle : ℝ := 225
def perimeter : ℝ := 83
noncomputable def pi_val : ℝ := Real.pi

-- Formula for the radius
noncomputable def radius : ℝ := 332 / (5 * pi_val + 8)

-- Prove that the radius is correct given the conditions
theorem circle_radius (theta : ℝ) (P : ℝ) (r : ℝ) (h_theta : theta = central_angle) (h_P : P = perimeter) :
  r = radius :=
sorry

end circle_radius_l147_14715


namespace remainder_when_dividing_150_l147_14781

theorem remainder_when_dividing_150 (k : ℕ) (hk1 : k > 0) (hk2 : 80 % k^2 = 8) : 150 % k = 6 :=
by
  sorry

end remainder_when_dividing_150_l147_14781


namespace find_side_DF_in_triangle_DEF_l147_14714

theorem find_side_DF_in_triangle_DEF
  (DE EF DM : ℝ)
  (h_DE : DE = 7)
  (h_EF : EF = 10)
  (h_DM : DM = 5) :
  ∃ DF : ℝ, DF = Real.sqrt 51 :=
by
  sorry

end find_side_DF_in_triangle_DEF_l147_14714


namespace tim_balloon_count_l147_14776

theorem tim_balloon_count (Dan_balloons : ℕ) (h1 : Dan_balloons = 59) (Tim_balloons : ℕ) (h2 : Tim_balloons = 11 * Dan_balloons) : Tim_balloons = 649 :=
sorry

end tim_balloon_count_l147_14776


namespace new_student_weight_l147_14729

theorem new_student_weight 
  (w_avg : ℝ)
  (w_new : ℝ)
  (condition : (5 * w_avg - 72 = 5 * (w_avg - 12) + w_new)) 
  : w_new = 12 := 
  by 
  sorry

end new_student_weight_l147_14729


namespace express_repeating_decimal_as_fraction_l147_14742

noncomputable def repeating_decimal_to_fraction : ℚ :=
  3 + 7 / 9  -- Representation of 3.\overline{7} as a Rational number representation

theorem express_repeating_decimal_as_fraction :
  (3 + 7 / 9 : ℚ) = 34 / 9 :=
by
  -- Placeholder for proof steps
  sorry

end express_repeating_decimal_as_fraction_l147_14742


namespace correct_total_score_l147_14711

theorem correct_total_score (total_score1 total_score2 : ℤ) : 
  (total_score1 = 5734 ∨ total_score2 = 5734) → (total_score1 = 5735 ∨ total_score2 = 5735) → 
  (total_score1 % 2 = 0 ∨ total_score2 % 2 = 0) → 
  (total_score1 ≠ total_score2) → 
  5734 % 2 = 0 :=
by
  sorry

end correct_total_score_l147_14711


namespace units_digit_of_quotient_l147_14720

theorem units_digit_of_quotient (n : ℕ) (h1 : n = 1987) : 
  (((4^n + 6^n) / 5) % 10) = 0 :=
by
  have pattern_4 : ∀ (k : ℕ), (4^k) % 10 = if k % 2 = 0 then 6 else 4 := sorry
  have pattern_6 : ∀ (k : ℕ), (6^k) % 10 = 6 := sorry
  have units_sum : (4^1987 % 10 + 6^1987 % 10) % 10 = 0 := sorry
  have multiple_of_5 : (4^1987 + 6^1987) % 5 = 0 := sorry
  sorry

end units_digit_of_quotient_l147_14720


namespace trajectory_of_M_l147_14707

noncomputable def P : ℝ × ℝ := (2, 2)
noncomputable def circleC (x y : ℝ) : Prop := x^2 + y^2 - 8 * y = 0
noncomputable def isMidpoint (A B M : ℝ × ℝ) : Prop := M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
noncomputable def isIntersectionPoint (l : ℝ × ℝ → Prop) (A B : ℝ × ℝ) : Prop :=
  ∃ x y : ℝ, circleC x y ∧ l (x, y) ∧ ((A = (x, y)) ∨ (B = (x, y))) 

theorem trajectory_of_M (M : ℝ × ℝ) : 
  (∃ A B : ℝ × ℝ, isIntersectionPoint (fun p => ∃ k : ℝ, p = (k, k)) A B ∧ isMidpoint A B M) →
  (M.1 - 1)^2 + (M.2 - 3)^2 = 2 := 
sorry

end trajectory_of_M_l147_14707


namespace dot_product_ABC_l147_14791

open Real

noncomputable def a : ℝ := 5
noncomputable def b : ℝ := 6
noncomputable def angleC : ℝ := π / 6  -- 30 degrees in radians

theorem dot_product_ABC :
  let CB := a
  let CA := b
  let angle_between := π - angleC  -- 150 degrees in radians
  let cos_angle := - (sqrt 3) / 2  -- cos(150 degrees)
  ∃ (dot_product : ℝ), dot_product = CB * CA * cos_angle :=
by
  have CB := a
  have CA := b
  have angle_between := π - angleC
  have cos_angle := - (sqrt 3) / 2
  use CB * CA * cos_angle
  sorry

end dot_product_ABC_l147_14791


namespace correct_option_l147_14710

def condition_A : Prop := abs ((-5 : ℤ)^2) = -5
def condition_B : Prop := abs (9 : ℤ) = 3 ∨ abs (9 : ℤ) = -3
def condition_C : Prop := abs (3 : ℤ) / abs (((-2)^3 : ℤ)) = -2
def condition_D : Prop := (2 * abs (3 : ℤ))^2 = 6 

theorem correct_option : ¬condition_A ∧ ¬condition_B ∧ condition_C ∧ ¬condition_D :=
by
  sorry

end correct_option_l147_14710


namespace factorize_expr1_factorize_expr2_l147_14735

-- Problem (1) Statement
theorem factorize_expr1 (x y : ℝ) : 
  -x^5 * y^3 + x^3 * y^5 = -x^3 * y^3 * (x + y) * (x - y) :=
sorry

-- Problem (2) Statement
theorem factorize_expr2 (a : ℝ) : 
  (a^2 + 1)^2 - 4 * a^2 = (a + 1)^2 * (a - 1)^2 :=
sorry

end factorize_expr1_factorize_expr2_l147_14735


namespace pony_wait_time_l147_14796

-- Definitions of the conditions
def cycle_time_monster_A : ℕ := 2 + 1 -- hours (2 awake, 1 rest)
def cycle_time_monster_B : ℕ := 3 + 2 -- hours (3 awake, 2 rest)

-- The theorem to prove the correct answer
theorem pony_wait_time :
  Nat.lcm cycle_time_monster_A cycle_time_monster_B = 15 :=
by
  -- Skip the proof
  sorry

end pony_wait_time_l147_14796


namespace compute_k_l147_14794

noncomputable def tan_inverse (k : ℝ) : ℝ := Real.arctan k

theorem compute_k (x k : ℝ) (hx1 : Real.tan x = 2 / 3) (hx2 : Real.tan (3 * x) = 3 / 5) : k = 2 / 3 := sorry

end compute_k_l147_14794


namespace prob_correct_l147_14704

noncomputable def r : ℝ := (4.5 : ℝ)  -- derived from solving area and line equations
noncomputable def s : ℝ := (7.5 : ℝ)  -- derived from solving area and line equations

theorem prob_correct (P Q T : ℝ × ℝ)
  (hP : P = (9, 0))
  (hQ : Q = (0, 15))
  (hT : T = (r, s))
  (hline : s = -5/3 * r + 15)
  (harea : 2 * (1/2 * 9 * 15) = (1/2 * 9 * s) * 4) :
  r + s = 12 := by
  sorry

end prob_correct_l147_14704


namespace equal_volume_rect_parallelepipeds_decomposable_equal_volume_prisms_decomposable_l147_14731

-- Definition of volumes for rectangular parallelepipeds
def volume_rect_parallelepiped (a b c: ℝ) : ℝ := a * b * c

-- Definition of volumes for prisms
def volume_prism (base_area height: ℝ) : ℝ := base_area * height

-- Definition of decomposability of rectangular parallelepipeds
def decomposable_rect_parallelepipeds (a1 b1 c1 a2 b2 c2: ℝ) : Prop :=
  (volume_rect_parallelepiped a1 b1 c1) = (volume_rect_parallelepiped a2 b2 c2)

-- Lean statement for part (a)
theorem equal_volume_rect_parallelepipeds_decomposable (a1 b1 c1 a2 b2 c2: ℝ) (h: decomposable_rect_parallelepipeds a1 b1 c1 a2 b2 c2) :
  True := sorry

-- Definition of decomposability of prisms
def decomposable_prisms (base_area1 height1 base_area2 height2: ℝ) : Prop :=
  (volume_prism base_area1 height1) = (volume_prism base_area2 height2)

-- Lean statement for part (b)
theorem equal_volume_prisms_decomposable (base_area1 height1 base_area2 height2: ℝ) (h: decomposable_prisms base_area1 height1 base_area2 height2) :
  True := sorry

end equal_volume_rect_parallelepipeds_decomposable_equal_volume_prisms_decomposable_l147_14731


namespace custom_op_evaluation_l147_14734

def custom_op (x y : ℤ) : ℤ := x * y - 3 * x

theorem custom_op_evaluation : custom_op 6 4 - custom_op 4 6 = -6 :=
by
  sorry

end custom_op_evaluation_l147_14734


namespace european_customer_savings_l147_14766

noncomputable def popcorn_cost : ℝ := 8 - 3
noncomputable def drink_cost : ℝ := popcorn_cost + 1
noncomputable def candy_cost : ℝ := drink_cost / 2

noncomputable def discounted_popcorn_cost : ℝ := popcorn_cost * (1 - 0.15)
noncomputable def discounted_candy_cost : ℝ := candy_cost * (1 - 0.1)

noncomputable def total_normal_cost : ℝ := 8 + discounted_popcorn_cost + drink_cost + discounted_candy_cost
noncomputable def deal_price : ℝ := 20
noncomputable def savings_in_dollars : ℝ := total_normal_cost - deal_price

noncomputable def exchange_rate : ℝ := 0.85
noncomputable def savings_in_euros : ℝ := savings_in_dollars * exchange_rate

theorem european_customer_savings : savings_in_euros = 0.81 := by
  sorry

end european_customer_savings_l147_14766


namespace vector_AD_length_l147_14785

open Real EuclideanSpace

noncomputable def problem_statement
  (m n : ℝ) (angle_mn : ℝ) (norm_m : ℝ) (norm_n : ℝ) (AB AC : ℝ) (AD : ℝ) : Prop :=
  angle_mn = π / 6 ∧ 
  norm_m = sqrt 3 ∧ 
  norm_n = 2 ∧ 
  AB = 2 * m + 2 * n ∧ 
  AC = 2 * m - 6 * n ∧ 
  AD = 2 * m - 2 * n ∧
  sqrt ((AD) * (AD)) = 2

theorem vector_AD_length 
  (m n : ℝ) (angle_mn : ℝ) (norm_m : ℝ) (norm_n : ℝ) (AB AC AD : ℝ) :
  problem_statement m n angle_mn norm_m norm_n AB AC AD :=
by
  unfold problem_statement
  sorry

end vector_AD_length_l147_14785


namespace james_eats_three_l147_14780

variables {p : ℕ} {f : ℕ} {j : ℕ}

-- The initial number of pizza slices
def initial_slices : ℕ := 8

-- The number of slices his friend eats
def friend_slices : ℕ := 2

-- The number of slices left after his friend eats
def remaining_slices : ℕ := initial_slices - friend_slices

-- The number of slices James eats
def james_slices : ℕ := remaining_slices / 2

-- The theorem to prove James eats 3 slices
theorem james_eats_three : james_slices = 3 :=
by
  sorry

end james_eats_three_l147_14780


namespace part1_part2_part3_l147_14739

-- Define the sequences a_n and b_n as described in the problem
def X_sequence (a : ℕ → ℝ) : Prop :=
  (a 1 = 1) ∧ (∀ n : ℕ, n > 0 → (a n = 0 ∨ a n = 1))

def accompanying_sequence (a b : ℕ → ℝ) : Prop :=
  (b 1 = 1) ∧ (∀ n : ℕ, n > 0 → b (n + 1) = abs (a n - (a (n + 1) / 2)) * b n)

-- 1. Prove the values of b_2, b_3, and b_4
theorem part1 (a b : ℕ → ℝ) (h_a : X_sequence a) (h_b : accompanying_sequence a b) :
  a 2 = 1 → a 3 = 0 → a 4 = 1 →
  b 2 = 1 / 2 ∧ b 3 = 1 / 2 ∧ b 4 = 1 / 4 := 
sorry

-- 2. Prove the equivalence for geometric sequence and constant sequence
theorem part2 (a b : ℕ → ℝ) (h_a : X_sequence a) (h_b : accompanying_sequence a b) :
  (∀ n : ℕ, n > 0 → a n = 1) ↔ (∃ r : ℝ, ∀ n : ℕ, n > 0 → b (n + 1) = r * b n) := 
sorry

-- 3. Prove the maximum value of b_2019
theorem part3 (a b : ℕ → ℝ) (h_a : X_sequence a) (h_b : accompanying_sequence a b) :
  b 2019 ≤ 1 / 2^1009 := 
sorry

end part1_part2_part3_l147_14739


namespace Hari_joined_after_5_months_l147_14782

noncomputable def Praveen_investment_per_year : ℝ := 3360 * 12
noncomputable def Hari_investment_for_given_months (x : ℝ) : ℝ := 8640 * (12 - x)

theorem Hari_joined_after_5_months (x : ℝ) (h : Praveen_investment_per_year / Hari_investment_for_given_months x = 2 / 3) : x = 5 :=
by
  sorry

end Hari_joined_after_5_months_l147_14782


namespace sum_of_234_and_142_in_base_4_l147_14700

theorem sum_of_234_and_142_in_base_4 :
  (234 + 142) = 376 ∧ (376 + 0) = 256 * 1 + 64 * 1 + 16 * 3 + 4 * 2 + 1 * 0 :=
by sorry

end sum_of_234_and_142_in_base_4_l147_14700


namespace paper_cost_l147_14752
noncomputable section

variables (P C : ℝ)

theorem paper_cost (h : 100 * P + 200 * C = 6.00) : 
  20 * P + 40 * C = 1.20 :=
sorry

end paper_cost_l147_14752


namespace four_played_games_l147_14706

theorem four_played_games
  (A B C D E : Prop)
  (A_answer : ¬A)
  (B_answer : A ∧ ¬B)
  (C_answer : B ∧ ¬C)
  (D_answer : C ∧ ¬D)
  (E_answer : D ∧ ¬E)
  (truth_condition : (¬A ∧ ¬B) ∨ (¬B ∧ ¬C) ∨ (¬C ∧ ¬D) ∨ (¬D ∧ ¬E)) :
  A ∨ B ∨ C ∨ D ∧ E := sorry

end four_played_games_l147_14706


namespace abs_inequality_solution_l147_14792

theorem abs_inequality_solution {x : ℝ} (h : |x + 1| < 5) : -6 < x ∧ x < 4 :=
by
  sorry

end abs_inequality_solution_l147_14792


namespace relationship_between_A_and_B_l147_14765

noncomputable def f (x : ℝ) : ℝ := x^2

def A : Set ℝ := {x | f x = x}

def B : Set ℝ := {x | f (f x) = x}

theorem relationship_between_A_and_B : A ∩ B = A :=
by sorry

end relationship_between_A_and_B_l147_14765


namespace solve_triangle_l147_14755

theorem solve_triangle :
  (a = 6 ∧ b = 6 * Real.sqrt 3 ∧ A = 30) →
  ((B = 60 ∧ C = 90 ∧ c = 12) ∨ (B = 120 ∧ C = 30 ∧ c = 6)) :=
by
  intros h
  sorry

end solve_triangle_l147_14755


namespace simplify_expression_l147_14744

theorem simplify_expression : (2468 * 2468) / (2468 + 2468) = 1234 :=
by
  sorry

end simplify_expression_l147_14744


namespace houses_in_lawrence_county_l147_14784

theorem houses_in_lawrence_county 
  (houses_before_boom : ℕ := 1426) 
  (houses_built_during_boom : ℕ := 574) 
  : houses_before_boom + houses_built_during_boom = 2000 := 
by 
  sorry

end houses_in_lawrence_county_l147_14784


namespace tan_theta_expr_l147_14795

theorem tan_theta_expr (θ : ℝ) (h : Real.tan θ = 4) : 
  (Real.sin θ + Real.cos θ) / (17 * Real.sin θ) + (Real.sin θ ^ 2) / 4 = 21 / 68 := 
by sorry

end tan_theta_expr_l147_14795


namespace power_mod_zero_problem_solution_l147_14767

theorem power_mod_zero (n : ℕ) (h : n ≥ 2) : 2 ^ n % 4 = 0 :=
  sorry

theorem problem_solution : 2 ^ 300 % 4 = 0 :=
  power_mod_zero 300 (by norm_num)

end power_mod_zero_problem_solution_l147_14767


namespace length_60_more_than_breadth_l147_14769

noncomputable def length_more_than_breadth (cost_per_meter : ℝ) (total_cost : ℝ) (length : ℝ) : Prop :=
  ∃ (breadth : ℝ) (x : ℝ), 
    length = breadth + x ∧
    2 * length + 2 * breadth = total_cost / cost_per_meter ∧
    x = length - breadth ∧
    x = 60

theorem length_60_more_than_breadth : length_more_than_breadth 26.5 5300 80 :=
by
  sorry

end length_60_more_than_breadth_l147_14769


namespace sum_of_x_and_y_l147_14746

theorem sum_of_x_and_y (x y : ℕ) (h_pos_x: 0 < x) (h_pos_y: 0 < y) (h_gt: x > y) (h_eq: x + x * y = 391) : x + y = 39 :=
by
  sorry

end sum_of_x_and_y_l147_14746


namespace adoption_cost_l147_14702

theorem adoption_cost :
  let cost_cat := 50
  let cost_adult_dog := 100
  let cost_puppy := 150
  let num_cats := 2
  let num_adult_dogs := 3
  let num_puppies := 2
  (num_cats * cost_cat + num_adult_dogs * cost_adult_dog + num_puppies * cost_puppy) = 700 :=
by
  sorry

end adoption_cost_l147_14702


namespace log_comparisons_l147_14772

noncomputable def a := Real.log 3 / Real.log 2
noncomputable def b := Real.log 3 / (2 * Real.log 2)
noncomputable def c := 1 / 2

theorem log_comparisons : c < b ∧ b < a := 
by
  sorry

end log_comparisons_l147_14772


namespace construct_length_one_l147_14726

theorem construct_length_one
    (a : ℝ) 
    (h_a : a = Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 5) : 
    ∃ (b : ℝ), b = 1 :=
by
    sorry

end construct_length_one_l147_14726


namespace angie_pretzels_dave_pretzels_l147_14737

theorem angie_pretzels (B S A : ℕ) (hB : B = 12) (hS : S = B / 2) (hA : A = 3 * S) : A = 18 := by
  -- We state the problem using variables B, S, and A for Barry, Shelly, and Angie respectively
  sorry

theorem dave_pretzels (A S D : ℕ) (hA : A = 18) (hS : S = 12 / 2) (hD : D = 25 * (A + S) / 100) : D = 6 := by
  -- We use variables A and S from the first theorem, and introduce D for Dave
  sorry

end angie_pretzels_dave_pretzels_l147_14737


namespace average_speed_palindrome_trip_l147_14703

theorem average_speed_palindrome_trip :
  ∀ (initial final : ℕ) (time : ℝ),
    initial = 13431 → final = 13531 → time = 3 →
    (final - initial) / time = 33 :=
by
  intros initial final time h_initial h_final h_time
  rw [h_initial, h_final, h_time]
  norm_num
  sorry

end average_speed_palindrome_trip_l147_14703


namespace find_numbers_in_progressions_l147_14736

theorem find_numbers_in_progressions (a b c d : ℝ) :
    (a + b + c = 114) ∧ -- Sum condition
    (b^2 = a * c) ∧ -- Geometric progression condition
    (b = a + 3 * d) ∧ -- Arithmetic progression first condition
    (c = a + 24 * d) -- Arithmetic progression second condition
    ↔ (a = 38 ∧ b = 38 ∧ c = 38) ∨ (a = 2 ∧ b = 14 ∧ c = 98) := by
  sorry

end find_numbers_in_progressions_l147_14736


namespace find_second_number_l147_14740

theorem find_second_number (n : ℕ) 
  (h1 : Nat.lcm 24 (Nat.lcm n 42) = 504)
  (h2 : 504 = 2^3 * 3^2 * 7) 
  (h3 : Nat.lcm 24 42 = 168) : n = 3 := 
by 
  sorry

end find_second_number_l147_14740


namespace y_share_per_rupee_of_x_l147_14750

theorem y_share_per_rupee_of_x (share_y : ℝ) (total_amount : ℝ) (z_per_x : ℝ) (y_per_x : ℝ) 
  (h1 : share_y = 54) 
  (h2 : total_amount = 210) 
  (h3 : z_per_x = 0.30) 
  (h4 : share_y = y_per_x * (total_amount / (1 + y_per_x + z_per_x))) : 
  y_per_x = 0.45 :=
sorry

end y_share_per_rupee_of_x_l147_14750


namespace number_of_pencils_bought_l147_14708

-- Define the conditions
def cost_of_glue : ℕ := 270
def cost_per_pencil : ℕ := 210
def amount_paid : ℕ := 1000
def change_received : ℕ := 100

-- Define the statement to prove
theorem number_of_pencils_bought : 
  ∃ (n : ℕ), cost_of_glue + (cost_per_pencil * n) = amount_paid - change_received :=
by {
  sorry 
}

end number_of_pencils_bought_l147_14708


namespace johns_age_l147_14741

variable (J : ℕ)

theorem johns_age :
  J - 5 = (1 / 2) * (J + 8) → J = 18 := by
    sorry

end johns_age_l147_14741


namespace puppies_sold_l147_14732

theorem puppies_sold (total_puppies sold_puppies puppies_per_cage total_cages : ℕ)
  (h1 : total_puppies = 102)
  (h2 : puppies_per_cage = 9)
  (h3 : total_cages = 9)
  (h4 : total_puppies - sold_puppies = puppies_per_cage * total_cages) :
  sold_puppies = 21 :=
by {
  -- Proof details would go here
  sorry
}

end puppies_sold_l147_14732


namespace augmented_matrix_determinant_l147_14713

theorem augmented_matrix_determinant (m : ℝ) 
  (h : (1 - 2 * m) / (3 - 2) = 5) : 
  m = -2 :=
  sorry

end augmented_matrix_determinant_l147_14713


namespace find_f_at_six_l147_14777

theorem find_f_at_six (f : ℝ → ℝ) (h : ∀ x : ℝ, f (4 * x - 2) = x^2 - x + 2) : f 6 = 3.75 :=
by
  sorry

end find_f_at_six_l147_14777


namespace inequality_relationship_l147_14768

noncomputable def a : ℝ := Real.sin (4 / 5)
noncomputable def b : ℝ := Real.cos (4 / 5)
noncomputable def c : ℝ := Real.tan (4 / 5)

theorem inequality_relationship : c > a ∧ a > b := sorry

end inequality_relationship_l147_14768


namespace original_selling_price_l147_14789

-- Definitions based on the conditions
def original_price : ℝ := 933.33

-- Given conditions
def discount_rate : ℝ := 0.40
def price_after_discount : ℝ := 560.0

-- Lean theorem statement to prove that original selling price (x) is equal to 933.33
theorem original_selling_price (x : ℝ) 
  (h1 : x * (1 - discount_rate) = price_after_discount) : 
  x = original_price :=
  sorry

end original_selling_price_l147_14789


namespace solution_set_of_inequality_l147_14733

variable {f : ℝ → ℝ}

theorem solution_set_of_inequality (h₁ : ∀ x > 0, deriv f x + 2 * f x > 0) :
  {x : ℝ | x + 2018 > 0 ∧ x + 2018 < 5} = {x : ℝ | -2018 < x ∧ x < -2013} := 
by
  sorry

end solution_set_of_inequality_l147_14733


namespace sum_cubes_of_roots_l147_14728

noncomputable def cube_root_sum_cubes (α β γ : ℝ) : ℝ :=
  α^3 + β^3 + γ^3
  
theorem sum_cubes_of_roots : 
  (cube_root_sum_cubes (Real.rpow 27 (1/3)) (Real.rpow 64 (1/3)) (Real.rpow 125 (1/3))) - 3 * ((Real.rpow 27 (1/3)) * (Real.rpow 64 (1/3)) * (Real.rpow 125 (1/3)) + 4/3) = 36 
  ∧
  ((Real.rpow 27 (1/3) + Real.rpow 64 (1/3) + Real.rpow 125 (1/3)) * ((Real.rpow 27 (1/3) + Real.rpow 64 (1/3) + Real.rpow 125 (1/3))^2 - 3 * ((Real.rpow 27 (1/3)) * (Real.rpow 64 (1/3)) + (Real.rpow 64 (1/3)) * (Real.rpow 125 (1/3)) + (Real.rpow 125 (1/3)) * (Real.rpow 27 (1/3)))) = 36) 
  → 
  cube_root_sum_cubes (Real.rpow 27 (1/3)) (Real.rpow 64 (1/3)) (Real.rpow 125 (1/3)) = 220 := 
sorry

end sum_cubes_of_roots_l147_14728


namespace find_a_plus_b_l147_14779

-- Given conditions
variable (a b : ℝ)

-- The imaginary unit i
noncomputable def i : ℂ := Complex.I

-- Condition equation
def equation := (a + i) * i = b - 2 * i

-- Define the lean statement
theorem find_a_plus_b (h : equation a b) : a + b = -3 :=
by sorry

end find_a_plus_b_l147_14779


namespace math_problem_l147_14757

theorem math_problem (a b : ℕ) (x y : ℚ) (h1 : a = 10) (h2 : b = 11) (h3 : x = 1.11) (h4 : y = 1.01) :
  ∃ k : ℕ, k * y = 2.02 ∧ (a * x + b * y - k * y = 20.19) :=
by {
  sorry
}

end math_problem_l147_14757


namespace nicky_pace_l147_14709

theorem nicky_pace :
  ∃ v : ℝ, v = 3 ∧ (
    ∀ (head_start : ℝ) (cristina_pace : ℝ) (time : ℝ) (distance_encounter : ℝ), 
      head_start = 36 ∧ cristina_pace = 4 ∧ time = 36 ∧ distance_encounter = cristina_pace * time - head_start →
      distance_encounter / time = v
  ) :=
sorry

end nicky_pace_l147_14709


namespace value_of_expression_l147_14727

theorem value_of_expression (x y z : ℤ) (h1 : x = 2) (h2 : y = 3) (h3 : z = 4) :
  (4 * x^2 - 6 * y^3 + z^2) / (5 * x + 7 * z - 3 * y^2) = -130 / 11 :=
by
  sorry

end value_of_expression_l147_14727


namespace product_equivalence_l147_14799

theorem product_equivalence 
  (a b c d e f : ℝ) 
  (h1 : a + b + c + d + e + f = 0) 
  (h2 : a^3 + b^3 + c^3 + d^3 + e^3 + f^3 = 0) : 
  (a + c) * (a + d) * (a + e) * (a + f) = (b + c) * (b + d) * (b + e) * (b + f) :=
by
  sorry

end product_equivalence_l147_14799


namespace willy_episodes_per_day_l147_14743

def total_episodes (seasons : ℕ) (episodes_per_season : ℕ) : ℕ :=
  seasons * episodes_per_season

def episodes_per_day (total_episodes : ℕ) (days : ℕ) : ℕ :=
  total_episodes / days

theorem willy_episodes_per_day :
  episodes_per_day (total_episodes 3 20) 30 = 2 :=
by
  sorry

end willy_episodes_per_day_l147_14743


namespace log_abs_monotone_decreasing_l147_14763

open Real

theorem log_abs_monotone_decreasing {a : ℝ} (h : ∀ x y, 0 < x ∧ x < y ∧ y ≤ a → |log x| ≥ |log y|) : 0 < a ∧ a ≤ 1 :=
by
  sorry

end log_abs_monotone_decreasing_l147_14763


namespace initial_crayons_count_l147_14759

variable (x : ℕ) -- x represents the initial number of crayons

theorem initial_crayons_count (h1 : x + 3 = 12) : x = 9 := 
by sorry

end initial_crayons_count_l147_14759


namespace initial_days_planned_l147_14771

-- We define the variables and conditions given in the problem.
variables (men_original men_absent men_remaining days_remaining days_initial : ℕ)
variable (work_equivalence : men_original * days_initial = men_remaining * days_remaining)

-- Conditions from the problem
axiom men_original_cond : men_original = 48
axiom men_absent_cond : men_absent = 8
axiom men_remaining_cond : men_remaining = men_original - men_absent
axiom days_remaining_cond : days_remaining = 18

-- Theorem to be proved
theorem initial_days_planned : days_initial = 15 :=
by
  -- Insert proof steps here
  sorry

end initial_days_planned_l147_14771


namespace perpendicular_lines_condition_l147_14786

theorem perpendicular_lines_condition (a : ℝ) : 
  (∀ x y : ℝ, x + y = 0 ∧ x - ay = 0 → x = 0) ↔ (a = 1) := 
sorry

end perpendicular_lines_condition_l147_14786


namespace johns_initial_money_l147_14756

/-- John's initial money given that he gives 3/8 to his mother and 3/10 to his father,
and he has $65 left after giving away the money. Prove that he initially had $200. -/
theorem johns_initial_money 
  (M : ℕ)
  (h_left : (M : ℚ) - (3 / 8) * M - (3 / 10) * M = 65) :
  M = 200 :=
sorry

end johns_initial_money_l147_14756


namespace smallest_integer_b_l147_14749

theorem smallest_integer_b (b : ℕ) : 27 ^ b > 3 ^ 9 ↔ b = 4 := by
  sorry

end smallest_integer_b_l147_14749


namespace amount_lent_to_B_l147_14705

theorem amount_lent_to_B
  (rate_of_interest_per_annum : ℝ)
  (P_C : ℝ)
  (years_C : ℝ)
  (total_interest : ℝ)
  (years_B : ℝ)
  (IB : ℝ)
  (IC : ℝ)
  (P_B : ℝ):
  (rate_of_interest_per_annum = 10) →
  (P_C = 3000) →
  (years_C = 4) →
  (total_interest = 2200) →
  (years_B = 2) →
  (IC = (P_C * rate_of_interest_per_annum * years_C) / 100) →
  (IB = (P_B * rate_of_interest_per_annum * years_B) / 100) →
  (total_interest = IB + IC) →
  P_B = 5000 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end amount_lent_to_B_l147_14705


namespace find_x_perpendicular_l147_14797

/-- Given vectors a = ⟨-1, 2⟩ and b = ⟨1, x⟩, if a is perpendicular to (a + 2 * b),
    then x = -3/4. -/
theorem find_x_perpendicular
  (x : ℝ)
  (a : ℝ × ℝ := (-1, 2))
  (b : ℝ × ℝ := (1, x))
  (h : (a.1 * (a.1 + 2 * b.1) + a.2 * (a.2 + 2 * b.2) = 0)) :
  x = -3 / 4 :=
sorry

end find_x_perpendicular_l147_14797


namespace polygon_sides_in_arithmetic_progression_l147_14738

theorem polygon_sides_in_arithmetic_progression 
  (a : ℕ → ℝ) (n : ℕ) (h1: ∀ i, 1 ≤ i ∧ i ≤ n → a i = a 1 + (i - 1) * 10) 
  (h2 : a n = 150) : n = 12 :=
sorry

end polygon_sides_in_arithmetic_progression_l147_14738


namespace hyperbola_foci_coordinates_l147_14745

theorem hyperbola_foci_coordinates :
  let a : ℝ := Real.sqrt 7
  let b : ℝ := Real.sqrt 3
  let c : ℝ := Real.sqrt (a^2 + b^2)
  (c = Real.sqrt 10 ∧
  ∀ x y, (x^2 / 7 - y^2 / 3 = 1) → ((x, y) = (c, 0) ∨ (x, y) = (-c, 0))) :=
by
  let a := Real.sqrt 7
  let b := Real.sqrt 3
  let c := Real.sqrt (a^2 + b^2)
  have hc : c = Real.sqrt 10 := sorry
  have h_foci : ∀ x y, (x^2 / 7 - y^2 / 3 = 1) → ((x, y) = (c, 0) ∨ (x, y) = (-c, 0)) := sorry
  exact ⟨hc, h_foci⟩

end hyperbola_foci_coordinates_l147_14745


namespace league_games_and_weeks_l147_14716

/--
There are 15 teams in a league, and each team plays each of the other teams exactly once.
Due to scheduling limitations, each team can only play one game per week.
Prove that the total number of games played is 105 and the minimum number of weeks needed to complete all the games is 15.
-/
theorem league_games_and_weeks :
  let teams := 15
  let total_games := teams * (teams - 1) / 2
  let games_per_week := Nat.div teams 2
  total_games = 105 ∧ total_games / games_per_week = 15 :=
by
  sorry

end league_games_and_weeks_l147_14716


namespace perimeter_of_smaller_polygon_l147_14712

/-- The ratio of the areas of two similar polygons is 1:16, and the difference in their perimeters is 9.
Find the perimeter of the smaller polygon. -/
theorem perimeter_of_smaller_polygon (a b : ℝ) (h1 : a / b = 1 / 16) (h2 : b - a = 9) : a = 3 :=
by
  sorry

end perimeter_of_smaller_polygon_l147_14712


namespace find_solution_l147_14758

theorem find_solution (x y z : ℝ) :
  (x * (y^2 + z) = z * (z + x * y)) ∧ 
  (y * (z^2 + x) = x * (x + y * z)) ∧ 
  (z * (x^2 + y) = y * (y + x * z)) → 
  (x = 1 ∧ y = 1 ∧ z = 1) ∨ (x = 0 ∧ y = 0 ∧ z = 0) :=
by
  sorry

end find_solution_l147_14758


namespace car_overtakes_truck_l147_14718

theorem car_overtakes_truck 
  (car_speed : ℝ)
  (truck_speed : ℝ)
  (car_arrival_time : ℝ)
  (truck_arrival_time : ℝ)
  (route_same : Prop)
  (time_difference : ℝ)
  (car_speed_km_min : car_speed = 66 / 60)
  (truck_speed_km_min : truck_speed = 42 / 60)
  (arrival_time_difference : truck_arrival_time - car_arrival_time = 18 / 60) :
  ∃ d : ℝ, d = 34.65 := 
by {
  sorry
}

end car_overtakes_truck_l147_14718


namespace value_of_expression_l147_14747

theorem value_of_expression (m : ℝ) (h : m^2 + m - 1 = 0) : m^3 + 2 * m^2 - 2005 = -2004 :=
by
  sorry

end value_of_expression_l147_14747


namespace find_y_l147_14761

variable {R : Type} [Field R] (y : R)

-- The condition: y = (1/y) * (-y) + 3
def condition (y : R) : Prop :=
  y = (1 / y) * (-y) + 3

-- The theorem to prove: under the condition, y = 2
theorem find_y (y : R) (h : condition y) : y = 2 := 
sorry

end find_y_l147_14761


namespace John_traded_in_car_money_back_l147_14762

-- First define the conditions provided in the problem.
def UberEarnings : ℝ := 30000
def CarCost : ℝ := 18000
def UberProfit : ℝ := 18000

-- We need to prove that John got $6000 back when trading in the car.
theorem John_traded_in_car_money_back : 
  UberEarnings - UberProfit = CarCost - 6000 := 
by
  -- provide the detailed steps inside the proof block if needed
  sorry

end John_traded_in_car_money_back_l147_14762


namespace magnitude_of_complex_l147_14788

noncomputable def z : ℂ := (2 / 3 : ℝ) - (4 / 5 : ℝ) * Complex.I

theorem magnitude_of_complex :
  Complex.abs z = (2 * Real.sqrt 61) / 15 :=
by
  sorry

end magnitude_of_complex_l147_14788


namespace distance_between_adjacent_parallel_lines_l147_14783

noncomputable def distance_between_lines (r d : ℝ) : ℝ :=
  (49 * r^2 - 49 * 600.25 - (49 / 4) * d^2) / (1 - 49 / 4)

theorem distance_between_adjacent_parallel_lines :
  ∃ d : ℝ, ∀ (r : ℝ), 
    (r^2 = 506.25 + (1 / 4) * d^2 ∧ r^2 = 600.25 + (49 / 4) * d^2) →
    d = 2.8 :=
sorry

end distance_between_adjacent_parallel_lines_l147_14783


namespace inequality_example_l147_14723

theorem inequality_example (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  x^2 + y^4 + z^6 ≥ x * y^2 + y^2 * z^3 + x * z^3 :=
sorry

end inequality_example_l147_14723


namespace relation_between_x_and_y_l147_14793

noncomputable def x : ℝ := 2 + Real.sqrt 3
noncomputable def y : ℝ := 1 / (2 - Real.sqrt 3)

theorem relation_between_x_and_y : x = y := sorry

end relation_between_x_and_y_l147_14793


namespace katy_brownies_l147_14778

theorem katy_brownies :
  ∃ (n : ℤ), (n = 5 + 2 * 5) :=
by
  sorry

end katy_brownies_l147_14778


namespace cities_with_highest_increase_l147_14725

-- Define population changes for each city
def cityF_initial := 30000
def cityF_final := 45000
def cityG_initial := 55000
def cityG_final := 77000
def cityH_initial := 40000
def cityH_final := 60000
def cityI_initial := 70000
def cityI_final := 98000
def cityJ_initial := 25000
def cityJ_final := 37500

-- Function to calculate percentage increase
def percentage_increase (initial final : ℕ) : ℚ :=
  ((final - initial) : ℚ) / (initial : ℚ) * 100

-- Theorem stating cities F, H, and J had the highest percentage increase
theorem cities_with_highest_increase :
  percentage_increase cityF_initial cityF_final = 50 ∧
  percentage_increase cityH_initial cityH_final = 50 ∧
  percentage_increase cityJ_initial cityJ_final = 50 ∧
  percentage_increase cityG_initial cityG_final < 50 ∧
  percentage_increase cityI_initial cityI_final < 50 :=
by
-- Proof omitted
sorry

end cities_with_highest_increase_l147_14725


namespace translation_line_segment_l147_14754

theorem translation_line_segment (a b : ℝ) :
  (∃ A B A1 B1: ℝ × ℝ,
    A = (1,0) ∧ B = (3,2) ∧ A1 = (a, 1) ∧ B1 = (4,b) ∧
    ∃ t : ℝ × ℝ, A + t = A1 ∧ B + t = B1) →
  a = 2 ∧ b = 3 :=
by
  sorry

end translation_line_segment_l147_14754


namespace calculate_brick_quantity_l147_14751

noncomputable def brick_quantity (brick_length brick_width brick_height wall_length wall_height wall_width : ℝ) : ℝ :=
  let brick_volume := brick_length * brick_width * brick_height
  let wall_volume := wall_length * wall_height * wall_width
  wall_volume / brick_volume

theorem calculate_brick_quantity :
  brick_quantity 20 10 8 1000 800 2450 = 1225000 := 
by 
  -- Volume calculations are shown but proof is omitted
  sorry

end calculate_brick_quantity_l147_14751


namespace evaluate_powers_of_i_l147_14798

theorem evaluate_powers_of_i :
  (Complex.I ^ 50) + (Complex.I ^ 105) = -1 + Complex.I :=
by 
  sorry

end evaluate_powers_of_i_l147_14798


namespace factorize_expression_l147_14790

theorem factorize_expression : ∀ x : ℝ, 2 * x^2 - 4 * x = 2 * x * (x - 2) :=
by
  intro x
  sorry

end factorize_expression_l147_14790


namespace find_a_l147_14760

theorem find_a (a : ℝ) (b : ℝ) :
  (9 * x^2 - 27 * x + a = (3 * x + b)^2) → b = -4.5 → a = 20.25 := 
by sorry

end find_a_l147_14760


namespace problem_statement_l147_14764

-- Definitions of the operations △ and ⊗
def triangle (a b : ℤ) : ℤ := a + b + a * b - 1
def otimes (a b : ℤ) : ℤ := a * a - a * b + b * b

-- The theorem statement
theorem problem_statement : triangle 3 (otimes 2 4) = 50 := by
  sorry

end problem_statement_l147_14764


namespace fraction_defined_iff_l147_14773

theorem fraction_defined_iff (x : ℝ) : (∃ y : ℝ, y = 1 / (|x| - 6)) ↔ (x ≠ 6 ∧ x ≠ -6) :=
by 
  sorry

end fraction_defined_iff_l147_14773


namespace h_at_2_l147_14770

noncomputable def f (x : ℝ) : ℝ := 3 * x - 4
noncomputable def g (x : ℝ) : ℝ := Real.sqrt (3 * f x) - 3
noncomputable def h (x : ℝ) : ℝ := f (g x)

theorem h_at_2 : h 2 = 3 * Real.sqrt 6 - 13 := 
by 
  sorry -- We skip the proof steps.

end h_at_2_l147_14770


namespace phoenix_hike_distance_l147_14753

variable (a b c d : ℕ)

theorem phoenix_hike_distance
  (h1 : a + b = 24)
  (h2 : b + c = 30)
  (h3 : c + d = 32)
  (h4 : a + c = 28) :
  a + b + c + d = 56 :=
by
  sorry

end phoenix_hike_distance_l147_14753


namespace min_stamps_value_l147_14774

theorem min_stamps_value (x y : ℕ) (hx : 5 * x + 7 * y = 74) : x + y = 12 :=
by
  sorry

end min_stamps_value_l147_14774


namespace prop_p_iff_prop_q_iff_not_or_p_q_l147_14748

theorem prop_p_iff (m : ℝ) :
  (∃ x₀ : ℝ, x₀^2 + 2 * m * x₀ + (2 + m) = 0) ↔ (m ≤ -1 ∨ m ≥ 2) :=
sorry

theorem prop_q_iff (m : ℝ) :
  (∃ x y : ℝ, (x^2)/(1 - 2*m) + (y^2)/(m + 2) = 1) ↔ (m < -2 ∨ m > 1/2) :=
sorry

theorem not_or_p_q (m : ℝ) :
  ¬(∃ x₀ : ℝ, x₀^2 + 2 * m * x₀ + (2 + m) = 0) ∧
  ¬(∃ x y : ℝ, (x^2)/(1 - 2*m) + (y^2)/(m + 2) = 1) ↔
  (-1 < m ∧ m ≤ 1/2) :=
sorry

end prop_p_iff_prop_q_iff_not_or_p_q_l147_14748


namespace mean_of_first_set_is_67_l147_14775

theorem mean_of_first_set_is_67 (x : ℝ) 
  (h : (50 + 62 + 97 + 124 + x) / 5 = 75.6) : 
  (28 + x + 70 + 88 + 104) / 5 = 67 := 
by
  sorry

end mean_of_first_set_is_67_l147_14775


namespace max_value_of_quadratic_l147_14717

theorem max_value_of_quadratic:
  ∀ (x : ℝ), (∃ y : ℝ, y = -3 * x ^ 2 + 9) → (∃ max_y : ℝ, max_y = 9 ∧ ∀ x : ℝ, -3 * x ^ 2 + 9 ≤ max_y) :=
by
  sorry

end max_value_of_quadratic_l147_14717


namespace number_of_members_greater_than_median_l147_14787

theorem number_of_members_greater_than_median (n : ℕ) (median : ℕ) (avg_age : ℕ) (youngest : ℕ) (oldest : ℕ) :
  n = 100 ∧ avg_age = 21 ∧ youngest = 1 ∧ oldest = 70 →
  ∃ k, k = 50 :=
by
  sorry

end number_of_members_greater_than_median_l147_14787


namespace coffee_consumption_l147_14724

-- Defining the necessary variables and conditions
variable (Ivory_cons Brayan_cons : ℕ)
variable (hr : ℕ := 1)
variable (hrs : ℕ := 5)

-- Condition: Brayan drinks twice as much coffee as Ivory
def condition1 := Brayan_cons = 2 * Ivory_cons

-- Condition: Brayan drinks 4 cups of coffee in an hour
def condition2 := Brayan_cons = 4

-- The proof problem
theorem coffee_consumption : ∀ (Ivory_cons Brayan_cons : ℕ), (Brayan_cons = 2 * Ivory_cons) → 
  (Brayan_cons = 4) → 
  ((Brayan_cons * hrs) + (Ivory_cons * hrs) = 30) :=
by
  intro hBrayan hIvory hr
  sorry

end coffee_consumption_l147_14724


namespace sum_of_distinct_prime_factors_315_l147_14730

theorem sum_of_distinct_prime_factors_315 : 
  ∃ factors : List ℕ, factors = [3, 5, 7] ∧ 315 = 3 * 3 * 5 * 7 ∧ factors.sum = 15 :=
by
  sorry

end sum_of_distinct_prime_factors_315_l147_14730


namespace base10_equivalent_of_43210_7_l147_14701

def base7ToDecimal (num : Nat) : Nat :=
  let digits := [4, 3, 2, 1, 0]
  digits[0] * 7^4 + digits[1] * 7^3 + digits[2] * 7^2 + digits[3] * 7^1 + digits[4] * 7^0

theorem base10_equivalent_of_43210_7 :
  base7ToDecimal 43210 = 10738 :=
by
  sorry

end base10_equivalent_of_43210_7_l147_14701


namespace previous_year_profit_percentage_l147_14722

theorem previous_year_profit_percentage (R : ℝ) (P : ℝ) :
  (0.16 * 0.70 * R = 1.1200000000000001 * (P / 100 * R)) → P = 10 :=
by {
  sorry
}

end previous_year_profit_percentage_l147_14722


namespace arithmetic_sequence_solution_l147_14719

theorem arithmetic_sequence_solution (x : ℝ) (h : 2 * (x + 1) = 2 * x + (x + 2)) : x = 0 :=
by {
  -- To avoid actual proof steps, we add sorry.
  sorry 
}

end arithmetic_sequence_solution_l147_14719
