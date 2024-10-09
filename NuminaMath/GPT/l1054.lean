import Mathlib

namespace units_digit_multiplication_l1054_105450

-- Define a function to find the units digit of a number
def units_digit (n : ℕ) : ℕ := n % 10

-- Statement of the problem: Given the product 27 * 36, prove that the units digit is 2.
theorem units_digit_multiplication (a b : ℕ) (h1 : units_digit 27 = 7) (h2 : units_digit 36 = 6) :
  units_digit (27 * 36) = 2 :=
by
  have h3 : units_digit (7 * 6) = 2 := by sorry
  exact h3

end units_digit_multiplication_l1054_105450


namespace range_of_a_l1054_105483

def p (a : ℝ) := ∀ x : ℝ, (1 ≤ x ∧ x ≤ 2) → x^2 - a ≥ 0
def q (a : ℝ) := ∃ x₀ : ℝ, x₀^2 + 2 * a * x₀ + 2 - a = 0

theorem range_of_a (a : ℝ) (hp : p a) (hq : q a) : a ≤ -2 ∨ a = 1 :=
by
  sorry

end range_of_a_l1054_105483


namespace two_trucks_carry_2_tons_l1054_105403

theorem two_trucks_carry_2_tons :
  ∀ (truck_capacity : ℕ), truck_capacity = 999 →
  (truck_capacity * 2) / 1000 = 2 :=
by
  intros truck_capacity h_capacity
  rw [h_capacity]
  exact sorry

end two_trucks_carry_2_tons_l1054_105403


namespace ben_heads_probability_l1054_105425

def coin_flip_probability : ℚ :=
  let total_ways := 2^10
  let ways_exactly_five_heads := Nat.choose 10 5
  let probability_exactly_five_heads := ways_exactly_five_heads / total_ways
  let remaining_probability := 1 - probability_exactly_five_heads
  let probability_more_heads := remaining_probability / 2
  probability_more_heads

theorem ben_heads_probability :
  coin_flip_probability = 193 / 512 := by
  sorry

end ben_heads_probability_l1054_105425


namespace profit_diff_is_560_l1054_105488

-- Define the initial conditions
def capital_A : ℕ := 8000
def capital_B : ℕ := 10000
def capital_C : ℕ := 12000
def profit_share_B : ℕ := 1400

-- Define the ratio parts
def ratio_A : ℕ := 4
def ratio_B : ℕ := 5
def ratio_C : ℕ := 6

-- Define the value of one part based on B's profit share and ratio part
def value_per_part : ℕ := profit_share_B / ratio_B

-- Define the profit shares of A and C
def profit_share_A : ℕ := ratio_A * value_per_part
def profit_share_C : ℕ := ratio_C * value_per_part

-- Define the difference between the profit shares of A and C
def profit_difference : ℕ := profit_share_C - profit_share_A

-- The theorem to prove
theorem profit_diff_is_560 : profit_difference = 560 := 
by sorry

end profit_diff_is_560_l1054_105488


namespace mao_li_total_cards_l1054_105460

theorem mao_li_total_cards : (23 : ℕ) + (20 : ℕ) = 43 := by
  sorry

end mao_li_total_cards_l1054_105460


namespace expected_value_is_150_l1054_105449

noncomputable def expected_value_of_winnings : ℝ :=
  let p := (1:ℝ)/8
  let winnings := [0, 2, 3, 5, 7]
  let losses := [4, 6]
  let extra := 5
  let win_sum := (winnings.sum : ℝ)
  let loss_sum := (losses.sum : ℝ)
  let E := p * 0 + p * win_sum - p * loss_sum + p * extra
  E

theorem expected_value_is_150 : expected_value_of_winnings = 1.5 := 
by sorry

end expected_value_is_150_l1054_105449


namespace birthday_pizza_problem_l1054_105433

theorem birthday_pizza_problem (m : ℕ) (h1 : m > 11) (h2 : 55 % m = 0) : 10 + 55 / m = 13 := by
  sorry

end birthday_pizza_problem_l1054_105433


namespace polynomial_roots_l1054_105424

theorem polynomial_roots :
  ∀ x, (3 * x^4 + 16 * x^3 - 36 * x^2 + 8 * x = 0) ↔ 
       (x = 0 ∨ x = 1 / 3 ∨ x = -3 + 2 * Real.sqrt 17 ∨ x = -3 - 2 * Real.sqrt 17) :=
by
  sorry

end polynomial_roots_l1054_105424


namespace horner_first_calculation_at_3_l1054_105435

def f (x : ℝ) : ℝ :=
  0.5 * x ^ 6 + 4 * x ^ 5 - x ^ 4 + 3 * x ^ 3 - 5 * x

def horner_first_step (x : ℝ) : ℝ :=
  0.5 * x + 4

theorem horner_first_calculation_at_3 :
  horner_first_step 3 = 5.5 := by
  sorry

end horner_first_calculation_at_3_l1054_105435


namespace toothpicks_15_l1054_105407

noncomputable def toothpicks : ℕ → ℕ
| 0       => 0  -- since the stage count n >= 1, stage 0 is not required, default 0.
| 1       => 5
| (n + 1) => 2 * toothpicks n + 2

theorem toothpicks_15 : toothpicks 15 = 32766 := by
  sorry

end toothpicks_15_l1054_105407


namespace jasmine_average_pace_l1054_105405

-- Define the conditions given in the problem
def totalDistance : ℝ := 45
def totalTime : ℝ := 9

-- Define the assertion that needs to be proved
theorem jasmine_average_pace : totalDistance / totalTime = 5 :=
by sorry

end jasmine_average_pace_l1054_105405


namespace exists_even_in_sequence_l1054_105426

theorem exists_even_in_sequence 
  (a : ℕ → ℕ)
  (h₀ : ∀ n : ℕ, a (n+1) = a n + (a n % 10)) :
  ∃ n : ℕ, a n % 2 = 0 :=
sorry

end exists_even_in_sequence_l1054_105426


namespace min_sum_of_dimensions_l1054_105404

theorem min_sum_of_dimensions (a b c : ℕ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_vol : a * b * c = 3003) : 
  a + b + c ≥ 57 := sorry

end min_sum_of_dimensions_l1054_105404


namespace complement_intersection_l1054_105470

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)

#check (Set.compl B) ∩ A = {1}

theorem complement_intersection (hU : U = {1, 2, 3, 4, 5}) (hA : A = {1, 5}) (hB : B = {2, 3, 5}) :
   (U \ B) ∩ A = {1} :=
by
  sorry

end complement_intersection_l1054_105470


namespace smallest_x_for_palindrome_l1054_105499

-- Define the condition for a number to be a palindrome
def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

-- Mathematically equivalent proof problem statement
theorem smallest_x_for_palindrome : ∃ (x : ℕ), x > 0 ∧ is_palindrome (x + 2345) ∧ x = 97 :=
by sorry

end smallest_x_for_palindrome_l1054_105499


namespace compound_interest_rate_l1054_105429

theorem compound_interest_rate (P r : ℝ) (h1 : 17640 = P * (1 + r / 100)^8)
                                (h2 : 21168 = P * (1 + r / 100)^12) :
  4 * (r / 100) = 18.6 :=
by
  sorry

end compound_interest_rate_l1054_105429


namespace forty_percent_of_number_l1054_105416

theorem forty_percent_of_number (N : ℝ) 
  (h : (1 / 4) * (1 / 3) * (2 / 5) * N = 17) : 
  0.40 * N = 204 :=
sorry

end forty_percent_of_number_l1054_105416


namespace ms_hatcher_students_l1054_105459

-- Define the number of third-graders
def third_graders : ℕ := 20

-- Condition: The number of fourth-graders is twice the number of third-graders
def fourth_graders : ℕ := 2 * third_graders

-- Condition: The number of fifth-graders is half the number of third-graders
def fifth_graders : ℕ := third_graders / 2

-- The total number of students Ms. Hatcher teaches in a day
def total_students : ℕ := third_graders + fourth_graders + fifth_graders

-- The statement to prove
theorem ms_hatcher_students : total_students = 70 := by
  sorry

end ms_hatcher_students_l1054_105459


namespace parabola_equation_l1054_105431

theorem parabola_equation (P : ℝ × ℝ) (hp : P = (4, -2)) : 
  ∃ m : ℝ, (∀ x y : ℝ, (y^2 = m * x) → (x, y) = P) ∧ (m = 1) :=
by
  have m_val : 1 = 1 := rfl
  sorry

end parabola_equation_l1054_105431


namespace possible_roots_l1054_105419

theorem possible_roots (a b p q : ℤ)
  (h1 : a ≠ 0)
  (h2 : b ≠ 0)
  (h3 : a ≠ b)
  (h4 : p = -(a + b))
  (h5 : q = ab)
  (h6 : (a + p) % (q - 2 * b) = 0) :
  a = 1 ∨ a = 3 :=
  sorry

end possible_roots_l1054_105419


namespace alice_has_ball_after_three_turns_l1054_105414

def probability_Alice_has_ball (turns: ℕ) : ℚ :=
  match turns with
  | 0 => 1 -- Alice starts with the ball
  | _ => sorry -- We would typically calculate this by recursion or another approach.

theorem alice_has_ball_after_three_turns :
  probability_Alice_has_ball 3 = 11 / 27 :=
by
  sorry

end alice_has_ball_after_three_turns_l1054_105414


namespace part_a_part_b_l1054_105412

-- Define the functions K_m and K_4
def K (m : ℕ) (x y z : ℝ) : ℝ :=
  x * (x - y)^m * (x - z)^m + y * (y - x)^m * (y - z)^m + z * (z - x)^m * (z - y)^m

-- Define M
def M (x y z : ℝ) : ℝ :=
  (x - y)^2 * (y - z)^2 * (z - x)^2

-- The proof goals:
-- 1. Prove K_m >= 0 for odd positive integer m
theorem part_a (m : ℕ) (hm : m % 2 = 1) (x y z : ℝ) : 
  0 ≤ K m x y z := 
sorry

-- 2. Prove K_7 + M^2 * K_1 >= M * K_4
theorem part_b (x y z : ℝ) : 
  K 7 x y z + (M x y z)^2 * K 1 x y z ≥ M x y z * K 4 x y z := 
sorry

end part_a_part_b_l1054_105412


namespace max_profit_l1054_105477

noncomputable def C (x : ℝ) : ℝ :=
  if x < 80 then (1/2) * x^2 + 40 * x
  else 101 * x + 8100 / x - 2180

noncomputable def profit (x : ℝ) : ℝ :=
  if x < 80 then 100 * x - C x - 500
  else 100 * x - C x - 500

theorem max_profit :
  (∀ x, (0 < x ∧ x < 80) → profit x = - (1/2) * x^2 + 60 * x - 500) ∧
  (∀ x, (80 ≤ x) → profit x = 1680 - (x + 8100 / x)) ∧
  (∃ x, x = 90 ∧ profit x = 1500) :=
by {
  -- Proof here
  sorry
}

end max_profit_l1054_105477


namespace largest_4digit_congruent_17_mod_28_l1054_105445

theorem largest_4digit_congruent_17_mod_28 :
  ∃ n, n < 10000 ∧ n % 28 = 17 ∧ ∀ m, m < 10000 ∧ m % 28 = 17 → m ≤ 9982 :=
by
  sorry

end largest_4digit_congruent_17_mod_28_l1054_105445


namespace proportion_in_triangle_l1054_105420

-- Definitions of the variables and conditions
variables {P Q R E : Point}
variables {p q r m n : ℝ}

-- Conditions
def angle_bisector_theorem (h : p = 2 * q) (h1 : m = q + q) (h2 : n = 2 * q) : Prop :=
  ∀ (p q r m n : ℝ), 
  (m / r) = (n / q) ∧ 
  (m + n = p) ∧
  (p = 2 * q)

-- The theorem to be proved
theorem proportion_in_triangle (h : p = 2 * q) (h1 : m / r = n / q) (h2 : m + n = p) : 
  (n / q = 2 * q / (r + q)) :=
by
  sorry

end proportion_in_triangle_l1054_105420


namespace direct_proportion_increases_inverse_proportion_increases_l1054_105458

-- Question 1: Prove y=2x increases as x increases.
theorem direct_proportion_increases (x1 x2 : ℝ) (h : x1 < x2) : 
  2 * x1 < 2 * x2 := by sorry

-- Question 2: Prove y=-2/x increases as x increases when x > 0.
theorem inverse_proportion_increases (x1 x2 : ℝ) (h1 : 0 < x1) (h2 : x1 < x2) : 
  - (2 / x1) < - (2 / x2) := by sorry

end direct_proportion_increases_inverse_proportion_increases_l1054_105458


namespace original_rectangle_area_at_least_90_l1054_105415

variable (a b c x y z : ℝ)
variable (hx1 : a * x = 1)
variable (hx2 : c * x = 3)
variable (hy : b * y = 10)
variable (hz : a * z = 9)
variable (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
variable (hx : 0 < x) (hy' : 0 < y) (hz' : 0 < z)

theorem original_rectangle_area_at_least_90 : ∀ {a b c x y z : ℝ},
  (a * x = 1) →
  (c * x = 3) →
  (b * y = 10) →
  (a * z = 9) →
  (0 < a) →
  (0 < b) →
  (0 < c) →
  (0 < x) →
  (0 < y) →
  (0 < z) →
  (a + b + c) * (x + y + z) ≥ 90 :=
sorry

end original_rectangle_area_at_least_90_l1054_105415


namespace circle_points_l1054_105406

noncomputable def proof_problem (x1 y1 x2 y2: ℝ) : Prop :=
  (x1^2 + y1^2 = 4) ∧ (x2^2 + y2^2 = 4) ∧ ((x1 - x2)^2 + (y1 - y2)^2 = 12) →
    (x1 * x2 + y1 * y2 = -2)

theorem circle_points (x1 y1 x2 y2 : ℝ) : proof_problem x1 y1 x2 y2 := 
by
  sorry

end circle_points_l1054_105406


namespace max_distance_between_circle_and_ellipse_l1054_105498

noncomputable def max_distance_PQ : ℝ :=
  1 + (3 * Real.sqrt 6) / 2

theorem max_distance_between_circle_and_ellipse :
  ∀ (P Q : ℝ × ℝ), (P.1^2 + (P.2 - 2)^2 = 1) → 
                   (Q.1^2 / 9 + Q.2^2 = 1) →
                   dist P Q ≤ max_distance_PQ :=
by
  intros P Q hP hQ
  sorry

end max_distance_between_circle_and_ellipse_l1054_105498


namespace candy_groups_l1054_105497

theorem candy_groups (total_candies group_size : Nat) (h1 : total_candies = 30) (h2 : group_size = 3) : total_candies / group_size = 10 := by
  sorry

end candy_groups_l1054_105497


namespace length_of_path_along_arrows_l1054_105423

theorem length_of_path_along_arrows (s : List ℝ) (h : s.sum = 73) :
  (3 * s.sum = 219) :=
by
  sorry

end length_of_path_along_arrows_l1054_105423


namespace find_a_conditions_l1054_105434

theorem find_a_conditions (a : ℝ) : 
    (∃ m : ℤ, a = m + 1/2) ∨ (∃ m : ℤ, a = m + 1/3) ∨ (∃ m : ℤ, a = m - 1/3) ↔ 
    (∃ n : ℤ, a = n + 1/2 ∨ a = n + 1/3 ∨ a = n - 1/3) :=
by
  sorry

end find_a_conditions_l1054_105434


namespace water_added_to_mixture_is_11_l1054_105451

noncomputable def initial_mixture_volume : ℕ := 45
noncomputable def initial_milk_ratio : ℚ := 4
noncomputable def initial_water_ratio : ℚ := 1
noncomputable def final_milk_ratio : ℚ := 9
noncomputable def final_water_ratio : ℚ := 5

theorem water_added_to_mixture_is_11 :
  ∃ x : ℚ, (initial_milk_ratio * initial_mixture_volume / 
            (initial_water_ratio * initial_mixture_volume + x)) = (final_milk_ratio / final_water_ratio)
  ∧ x = 11 :=
by
  -- Proof here
  sorry

end water_added_to_mixture_is_11_l1054_105451


namespace prob_defective_first_draw_prob_defective_both_draws_prob_defective_second_given_first_l1054_105427

-- Definitions
def total_products := 20
def defective_products := 5

-- Probability of drawing a defective product on the first draw
theorem prob_defective_first_draw : (defective_products / total_products : ℚ) = 1 / 4 :=
sorry

-- Probability of drawing defective products on both the first and the second draws
theorem prob_defective_both_draws : (defective_products / total_products * (defective_products - 1) / (total_products - 1) : ℚ) = 1 / 19 :=
sorry

-- Probability of drawing a defective product on the second draw given that the first was defective
theorem prob_defective_second_given_first : ((defective_products - 1) / (total_products - 1) / (defective_products / total_products) : ℚ) = 4 / 19 :=
sorry

end prob_defective_first_draw_prob_defective_both_draws_prob_defective_second_given_first_l1054_105427


namespace system_of_equations_solutions_l1054_105452

theorem system_of_equations_solutions (x1 x2 x3 : ℝ) :
  (2 * x1^2 / (1 + x1^2) = x2) ∧ (2 * x2^2 / (1 + x2^2) = x3) ∧ (2 * x3^2 / (1 + x3^2) = x1)
  → (x1 = 0 ∧ x2 = 0 ∧ x3 = 0) ∨ (x1 = 1 ∧ x2 = 1 ∧ x3 = 1) :=
by
  sorry

end system_of_equations_solutions_l1054_105452


namespace principal_amount_unique_l1054_105438

theorem principal_amount_unique (SI R T : ℝ) (P : ℝ) : 
  SI = 4016.25 → R = 14 → T = 5 → SI = (P * R * T) / 100 → P = 5737.5 :=
by
  intro h₁ h₂ h₃ h₄
  rw [h₁, h₂, h₃] at h₄
  sorry

end principal_amount_unique_l1054_105438


namespace prism_diagonal_correct_l1054_105421

open Real

noncomputable def prism_diagonal_1 := 2 * sqrt 6
noncomputable def prism_diagonal_2 := sqrt 66

theorem prism_diagonal_correct (length width : ℝ) (h1 : length = 8) (h2 : width = 4) :
  (prism_diagonal_1 = 2 * sqrt 6 ∧ prism_diagonal_2 = sqrt 66) :=
by
  sorry

end prism_diagonal_correct_l1054_105421


namespace common_difference_of_arithmetic_sequence_l1054_105448

variable (a1 d : ℤ)
def S : ℕ → ℤ
| 0     => 0
| (n+1) => S n + (a1 + n * d)

theorem common_difference_of_arithmetic_sequence
  (h : 2 * S a1 d 3 = 3 * S a1 d 2 + 6) :
  d = 2 :=
  sorry

end common_difference_of_arithmetic_sequence_l1054_105448


namespace MMobile_cheaper_l1054_105462

-- Define the given conditions
def TMobile_base_cost : ℕ := 50
def TMobile_additional_cost : ℕ := 16
def MMobile_base_cost : ℕ := 45
def MMobile_additional_cost : ℕ := 14
def additional_lines : ℕ := 3

-- Define functions to calculate total costs
def TMobile_total_cost : ℕ := TMobile_base_cost + TMobile_additional_cost * additional_lines
def MMobile_total_cost : ℕ := MMobile_base_cost + MMobile_additional_cost * additional_lines

-- Statement to be proved
theorem MMobile_cheaper : TMobile_total_cost - MMobile_total_cost = 11 := by
  sorry

end MMobile_cheaper_l1054_105462


namespace smallest_three_digit_multiple_of_three_with_odd_hundreds_l1054_105476

theorem smallest_three_digit_multiple_of_three_with_odd_hundreds :
  ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ (∃ a b c : ℕ, n = 100 * a + 10 * b + c ∧ a % 2 = 1 ∧ n % 3 = 0 ∧ n = 102) :=
by
  sorry

end smallest_three_digit_multiple_of_three_with_odd_hundreds_l1054_105476


namespace travel_time_to_Virgo_island_l1054_105457

theorem travel_time_to_Virgo_island (boat_time : ℝ) (plane_time : ℝ) (total_time : ℝ) 
  (h1 : boat_time ≤ 2) (h2 : plane_time = 4 * boat_time) (h3 : total_time = plane_time + boat_time) : 
  total_time = 10 :=
by
  sorry

end travel_time_to_Virgo_island_l1054_105457


namespace first_discount_percentage_l1054_105468

theorem first_discount_percentage :
  ∃ x : ℝ, (9649.12 * (1 - x / 100) * 0.9 * 0.95 = 6600) ∧ (19.64 ≤ x ∧ x ≤ 19.66) :=
sorry

end first_discount_percentage_l1054_105468


namespace num_vec_a_exists_l1054_105417

-- Define the vectors and the conditions
def vec_a (x y : ℝ) : (ℝ × ℝ) := (x, y)
def vec_b (x y : ℝ) : (ℝ × ℝ) := (x^2, y^2)
def vec_c : (ℝ × ℝ) := (1, 1)

-- Define the dot product
def dot_prod (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Define the conditions
def cond_1 (x y : ℝ) : Prop := (x + y = 1)
def cond_2 (x y : ℝ) : Prop := (x^2 / 4 + (1 - x)^2 / 9 = 1)

-- The proof problem statement
theorem num_vec_a_exists : ∃! (x y : ℝ), cond_1 x y ∧ cond_2 x y := by
  sorry

end num_vec_a_exists_l1054_105417


namespace ratio_of_money_earned_l1054_105496

variable (L T J : ℕ) 

theorem ratio_of_money_earned 
  (total_earned : L + T + J = 60)
  (lisa_earning : L = 30)
  (lisa_tommy_diff : L = T + 15) : 
  T / L = 1 / 2 := 
by
  sorry

end ratio_of_money_earned_l1054_105496


namespace donation_to_treetown_and_forest_reserve_l1054_105494

noncomputable def donation_problem (x : ℕ) :=
  x + (x + 140) = 1000

theorem donation_to_treetown_and_forest_reserve :
  ∃ x : ℕ, donation_problem x ∧ (x + 140 = 570) := 
by
  sorry

end donation_to_treetown_and_forest_reserve_l1054_105494


namespace variance_of_dataSet_l1054_105409

-- Define the given data set
def dataSet : List ℤ := [-2, -1, 0, 1, 2]

-- Define the function to calculate mean
def mean (data : List ℤ) : ℚ :=
  (data.sum : ℚ) / data.length

-- Define the function to calculate variance
def variance (data : List ℤ) : ℚ :=
  let μ := mean data
  (data.map (λ x => (x - μ) ^ 2)).sum / data.length

-- State the theorem: The variance of the given data set is 2
theorem variance_of_dataSet : variance dataSet = 2 := by
  sorry

end variance_of_dataSet_l1054_105409


namespace numbers_at_distance_1_from_neg2_l1054_105465

theorem numbers_at_distance_1_from_neg2 : 
  ∃ x : ℤ, (|x + 2| = 1) ∧ (x = -1 ∨ x = -3) :=
by
  sorry

end numbers_at_distance_1_from_neg2_l1054_105465


namespace surveyor_problem_l1054_105456

theorem surveyor_problem
  (GF : ℝ) (G4 : ℝ)
  (hGF : GF = 70)
  (hG4 : G4 = 60) :
  (1/2) * GF * G4 = 2100 := 
  by
  sorry

end surveyor_problem_l1054_105456


namespace arithmetic_problem_l1054_105473

theorem arithmetic_problem :
  12.1212 + 17.0005 - 9.1103 = 20.0114 :=
sorry

end arithmetic_problem_l1054_105473


namespace tangent_line_at_x0_minimum_value_on_interval_minimum_value_on_interval_high_minimum_value_on_interval_mid_l1054_105471

noncomputable def f (x a : ℝ) : ℝ := (x - a) * Real.exp x

theorem tangent_line_at_x0 (a : ℝ) (h : a = 2) : 
    (∃ m b : ℝ, (∀ x : ℝ, f x a = m * x + b) ∧ m = -1 ∧ b = -2) :=
by 
    sorry

theorem minimum_value_on_interval (a : ℝ) :
    (1 ≤ a) → (a ≤ 2) → f 1 a = (1 - a) * Real.exp 1 :=
by 
    sorry

theorem minimum_value_on_interval_high (a : ℝ) :
    (a ≥ 3) → f 2 a = (2 - a) * Real.exp 2 :=
by 
    sorry

theorem minimum_value_on_interval_mid (a : ℝ) :
    (2 < a) → (a < 3) → f (a - 1) a = -(Real.exp (a - 1)) :=
by 
    sorry

end tangent_line_at_x0_minimum_value_on_interval_minimum_value_on_interval_high_minimum_value_on_interval_mid_l1054_105471


namespace todd_initial_money_l1054_105495

-- Definitions of the conditions
def cost_per_candy_bar : ℕ := 2
def number_of_candy_bars : ℕ := 4
def money_left : ℕ := 12
def total_money_spent := number_of_candy_bars * cost_per_candy_bar

-- The statement proving the initial amount of money Todd had
theorem todd_initial_money : 
  (total_money_spent + money_left) = 20 :=
by
  sorry

end todd_initial_money_l1054_105495


namespace find_angle_l1054_105491

theorem find_angle (θ : ℝ) (h : 180 - θ = 3 * (90 - θ)) : θ = 45 :=
by
  sorry

end find_angle_l1054_105491


namespace number_of_samples_from_retired_l1054_105428

def ratio_of_forms (retired current students : ℕ) : Prop :=
retired = 3 ∧ current = 7 ∧ students = 40

def total_sampled_forms := 300

theorem number_of_samples_from_retired :
  ∃ (xr : ℕ), ratio_of_forms 3 7 40 → xr = (300 / (3 + 7 + 40)) * 3 :=
sorry

end number_of_samples_from_retired_l1054_105428


namespace minimum_framing_needed_l1054_105408

-- Definitions given the conditions
def original_width := 5
def original_height := 7
def enlargement_factor := 4
def border_width := 3
def inches_per_foot := 12

-- Conditions translated to definitions
def enlarged_width := original_width * enlargement_factor
def enlarged_height := original_height * enlargement_factor
def bordered_width := enlarged_width + 2 * border_width
def bordered_height := enlarged_height + 2 * border_width
def perimeter := 2 * (bordered_width + bordered_height)
def perimeter_in_feet := perimeter / inches_per_foot

-- Prove that the minimum number of linear feet of framing required is 10 feet
theorem minimum_framing_needed : perimeter_in_feet = 10 := 
by 
  sorry

end minimum_framing_needed_l1054_105408


namespace third_side_length_is_six_l1054_105447

theorem third_side_length_is_six
  (a b : ℝ) (c : ℤ)
  (h1 : a = 6.31) 
  (h2 : b = 0.82) 
  (h3 : (a + b > c) ∧ ((b : ℝ) + (c : ℝ) > a) ∧ (c + a > b)) 
  (h4 : 5.49 < (c : ℝ)) 
  (h5 : (c : ℝ) < 7.13) : 
  c = 6 :=
by
  -- Proof goes here
  sorry

end third_side_length_is_six_l1054_105447


namespace range_of_a_l1054_105469

noncomputable def f (x : ℝ) : ℝ := 
  if x ≥ 0 then x^2 else -x^2

theorem range_of_a (a : ℝ) : (∀ x : ℝ, a ≤ x ∧ x ≤ a + 2 → f (x + a) ≥ 2 * f x) → a ≥ Real.sqrt 2 :=
by
  -- provided condition
  intros h
  sorry

end range_of_a_l1054_105469


namespace roots_g_eq_zero_l1054_105487

noncomputable def g : ℝ → ℝ := sorry

theorem roots_g_eq_zero :
  (∀ x : ℝ, g (3 + x) = g (3 - x)) →
  (∀ x : ℝ, g (8 + x) = g (8 - x)) →
  (∀ x : ℝ, g (12 + x) = g (12 - x)) →
  g 0 = 0 →
  ∃ L : ℕ, 
  (∀ k, 0 ≤ k ∧ k ≤ L → g (k * 48) = 0) ∧ 
  (∀ k : ℤ, -1000 ≤ k ∧ k ≤ 1000 → (∃ n : ℕ, k = n * 48)) ∧ 
  L + 1 = 42 := 
by sorry

end roots_g_eq_zero_l1054_105487


namespace bike_ride_distance_l1054_105485

theorem bike_ride_distance (D : ℝ) (h : D / 10 = D / 15 + 0.5) : D = 15 :=
  sorry

end bike_ride_distance_l1054_105485


namespace condition1_condition2_condition3_l1054_105474

noncomputable def Z (m : ℝ) : ℂ := (m^2 - 4 * m) + (m^2 - m - 6) * Complex.I

-- Condition 1: Point Z is in the third quadrant
theorem condition1 (m : ℝ) (h_quad3 : (m^2 - 4 * m) < 0 ∧ (m^2 - m - 6) < 0) : 0 < m ∧ m < 3 :=
sorry

-- Condition 2: Point Z is on the imaginary axis
theorem condition2 (m : ℝ) (h_imaginary : (m^2 - 4 * m) = 0 ∧ (m^2 - m - 6) ≠ 0) : m = 0 ∨ m = 4 :=
sorry

-- Condition 3: Point Z is on the line x - y + 3 = 0
theorem condition3 (m : ℝ) (h_line : (m^2 - 4 * m) - (m^2 - m - 6) + 3 = 0) : m = 3 :=
sorry

end condition1_condition2_condition3_l1054_105474


namespace product_of_xy_l1054_105443

-- Define the problem conditions
variables (x y : ℝ)
-- Define the condition that |x-3| and |y+1| are opposite numbers
def opposite_abs_values := |x - 3| = - |y + 1|

-- State the theorem
theorem product_of_xy (h : opposite_abs_values x y) : x * y = -3 :=
sorry -- Proof is omitted

end product_of_xy_l1054_105443


namespace legs_sum_of_right_triangle_with_hypotenuse_41_l1054_105466

noncomputable def right_triangle_legs_sum (x : ℕ) : ℕ := x + (x + 1)

theorem legs_sum_of_right_triangle_with_hypotenuse_41 :
  ∃ x : ℕ, (x * x + (x + 1) * (x + 1) = 41 * 41) ∧ right_triangle_legs_sum x = 57 := by
sorry

end legs_sum_of_right_triangle_with_hypotenuse_41_l1054_105466


namespace distinct_solutions_abs_eq_l1054_105478

theorem distinct_solutions_abs_eq (x : ℝ) : (|x - 5| = |x + 3|) → x = 1 :=
by
  -- The proof is omitted
  sorry

end distinct_solutions_abs_eq_l1054_105478


namespace destiny_cookies_divisible_l1054_105413

theorem destiny_cookies_divisible (C : ℕ) (h : C % 6 = 0) : ∃ k : ℕ, C = 6 * k :=
by {
  sorry
}

end destiny_cookies_divisible_l1054_105413


namespace find_number_l1054_105490

theorem find_number (x : ℕ) (h : x * 48 = 173 * 240) : x = 865 :=
sorry

end find_number_l1054_105490


namespace graph_is_empty_l1054_105489

/-- The given equation 3x² + 4y² - 12x - 16y + 36 = 0 has no real solutions. -/
theorem graph_is_empty : ∀ (x y : ℝ), 3 * x^2 + 4 * y^2 - 12 * x - 16 * y + 36 ≠ 0 :=
by
  intro x y
  sorry

end graph_is_empty_l1054_105489


namespace directrix_of_parabola_l1054_105486

theorem directrix_of_parabola (a : ℝ) (h : a = -4) : ∃ k : ℝ, k = 1/16 ∧ ∀ x : ℕ, y = ax ^ 2 → y = k := 
by 
  sorry

end directrix_of_parabola_l1054_105486


namespace john_initial_pens_l1054_105484

theorem john_initial_pens (P S C : ℝ) (n : ℕ) 
  (h1 : 20 * S = P) 
  (h2 : C = (2 / 3) * S) 
  (h3 : n * C = P)
  (h4 : P > 0) 
  (h5 : S > 0) 
  (h6 : C > 0)
  : n = 30 :=
by
  sorry

end john_initial_pens_l1054_105484


namespace point_on_x_axis_coordinates_l1054_105437

theorem point_on_x_axis_coordinates (a : ℝ) (P : ℝ × ℝ) (h : P = (a - 1, a + 2)) (hx : P.2 = 0) : P = (-3, 0) :=
by
  -- Replace this with the full proof
  sorry

end point_on_x_axis_coordinates_l1054_105437


namespace painting_time_equation_l1054_105442

theorem painting_time_equation (t : ℝ) :
  (1/6 + 1/8) * (t - 2) = 1 :=
sorry

end painting_time_equation_l1054_105442


namespace prove_divisibility_l1054_105475

-- Given the conditions:
variables (a b r s : ℕ)
variables (pos_a : a > 0) (pos_b : b > 0) (pos_r : r > 0) (pos_s : s > 0)
variables (a_le_two : a ≤ 2)
variables (no_common_prime_factor : (gcd a b) = 1)
variables (divisibility_condition : (a ^ s + b ^ s) ∣ (a ^ r + b ^ r))

-- We aim to prove that:
theorem prove_divisibility : s ∣ r := 
sorry

end prove_divisibility_l1054_105475


namespace diophantine_infinite_solutions_l1054_105493

theorem diophantine_infinite_solutions :
  ∃ (a b c x y : ℤ), (a + b + c = x + y) ∧ (a^3 + b^3 + c^3 = x^3 + y^3) ∧ 
  ∃ (d : ℤ), (a = b - d) ∧ (c = b + d) :=
sorry

end diophantine_infinite_solutions_l1054_105493


namespace identify_parrots_l1054_105455

-- Definitions of parrots
inductive Parrot
| gosha : Parrot
| kesha : Parrot
| roma : Parrot

open Parrot

-- Properties of each parrot
def always_honest (p : Parrot) : Prop :=
  p = gosha

def always_liar (p : Parrot) : Prop :=
  p = kesha

def sometimes_honest (p : Parrot) : Prop :=
  p = roma

-- Statements given by each parrot
def Gosha_statement : Prop :=
  always_liar kesha

def Kesha_statement : Prop :=
  sometimes_honest kesha

def Roma_statement : Prop :=
  always_honest kesha

-- Final statement to prove the identities
theorem identify_parrots (p : Parrot) :
  Gosha_statement ∧ Kesha_statement ∧ Roma_statement → (always_liar Parrot.kesha ∧ sometimes_honest Parrot.roma) :=
by
  intro h
  exact sorry

end identify_parrots_l1054_105455


namespace range_of_f_l1054_105440

-- Define the function f(x) = 4 sin^3(x) + sin^2(x) - 4 sin(x) + 8
noncomputable def f (x : ℝ) : ℝ :=
  4 * (Real.sin x) ^ 3 + (Real.sin x) ^ 2 - 4 * (Real.sin x) + 8

-- Statement to prove the range of f(x)
theorem range_of_f :
  ∀ x : ℝ, 6 + 3 / 4 ≤ f x ∧ f x ≤ 9 + 25 / 27 :=
sorry

end range_of_f_l1054_105440


namespace Grant_spending_is_200_l1054_105430

def Juanita_daily_spending (day: String) : Float :=
  if day = "Sunday" then 2.0 else 0.5

def Juanita_weekly_spending : Float :=
  6 * Juanita_daily_spending "weekday" + Juanita_daily_spending "Sunday"

def Juanita_yearly_spending : Float :=
  52 * Juanita_weekly_spending

def Grant_yearly_spending := Juanita_yearly_spending - 60

theorem Grant_spending_is_200 : Grant_yearly_spending = 200 := by
  sorry

end Grant_spending_is_200_l1054_105430


namespace line_equation_l1054_105482

theorem line_equation
  (P : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ) (hP : P = (-4, 6))
  (hxA : A.2 = 0) (hyB : B.1 = 0)
  (hMidpoint : P = ((A.1 + B.1)/2, (A.2 + B.2)/2)):
  3 * A.1 - 2 * B.2 + 24 = 0 :=
by
  -- Define point P
  let P := (-4, 6)
  -- Define points A and B, knowing P is the midpoint of AB and using conditions from the problem
  let A := (-8, 0)
  let B := (0, 12)
  sorry

end line_equation_l1054_105482


namespace half_angle_quadrant_l1054_105472

variables {α : ℝ} {k : ℤ} {n : ℤ}

theorem half_angle_quadrant (h : ∃ k : ℤ, k * 360 + 180 < α ∧ α < k * 360 + 270) :
  ∃ (n : ℤ), (n * 360 + 90 < α / 2 ∧ α / 2 < n * 360 + 135) ∨ 
      (n * 360 + 270 < α / 2 ∧ α / 2 < n * 360 + 315) :=
by sorry

end half_angle_quadrant_l1054_105472


namespace find_circle_equation_l1054_105441

-- Define the intersection point of the lines x + y + 1 = 0 and x - y - 1 = 0
def center : ℝ × ℝ := (0, -1)

-- Define the chord length AB
def chord_length : ℝ := 6

-- Line equation that intersects the circle
def line_eq (x y : ℝ) : Prop := 3 * x + 4 * y - 11 = 0

-- Circle equation to be proven
def circle_eq (x y : ℝ) : Prop := x^2 + (y + 1)^2 = 18

-- Main theorem: Prove that the given circle equation is correct under the conditions
theorem find_circle_equation (x y : ℝ) (hc : x + y + 1 = 0) (hc' : x - y - 1 = 0) 
  (hl : line_eq x y) : circle_eq x y :=
sorry

end find_circle_equation_l1054_105441


namespace arccos_sin_three_pi_over_two_eq_pi_l1054_105467

theorem arccos_sin_three_pi_over_two_eq_pi : 
  Real.arccos (Real.sin (3 * Real.pi / 2)) = Real.pi :=
by
  sorry

end arccos_sin_three_pi_over_two_eq_pi_l1054_105467


namespace hall_area_l1054_105453

theorem hall_area {L W : ℝ} (h₁ : W = 0.5 * L) (h₂ : L - W = 20) : L * W = 800 := by
  sorry

end hall_area_l1054_105453


namespace gcd_ab_eq_one_l1054_105418

def a : ℕ := 97^10 + 1
def b : ℕ := 97^10 + 97^3 + 1

theorem gcd_ab_eq_one : Nat.gcd a b = 1 :=
by
  sorry

end gcd_ab_eq_one_l1054_105418


namespace sum_of_roots_l1054_105432

theorem sum_of_roots (r s t : ℝ) (hroots : 3 * (r^3 + s^3 + t^3) + 9 * (r^2 + s^2 + t^2) - 36 * (r + s + t) + 12 = 0) :
  r + s + t = -3 :=
sorry

end sum_of_roots_l1054_105432


namespace sara_total_score_l1054_105401

-- Definitions based on the conditions
def correct_points (correct_answers : Nat) : Int := correct_answers * 2
def incorrect_points (incorrect_answers : Nat) : Int := incorrect_answers * (-1)
def unanswered_points (unanswered_questions : Nat) : Int := unanswered_questions * 0

def total_score (correct_answers incorrect_answers unanswered_questions : Nat) : Int :=
  correct_points correct_answers + incorrect_points incorrect_answers + unanswered_points unanswered_questions

-- The main theorem stating the problem requirement
theorem sara_total_score :
  total_score 18 10 2 = 26 :=
by
  sorry

end sara_total_score_l1054_105401


namespace no_exp_function_satisfies_P_d_decreasing_nonnegative_exists_c_infinitely_many_l1054_105481

-- Define the context for real numbers and the main property P.
def property_P (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f x + f (x + 2) ≤ 2 * f (x + 1)

-- For part (1)
theorem no_exp_function_satisfies_P (a : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1) :
  ¬ ∃ (f : ℝ → ℝ), (∀ x : ℝ, f x = a^x) ∧ property_P f :=
sorry

-- Define the context for natural numbers, d(x), and main properties related to P.
def d (f : ℕ → ℕ) (x : ℕ) : ℕ := f (x + 1) - f x

-- For part (2)(i)
theorem d_decreasing_nonnegative (f : ℕ → ℕ) (P : ∀ x : ℕ, f x + f (x + 2) ≤ 2 * f (x + 1)) :
  ∀ x : ℕ, d f (x + 1) ≤ d f x ∧ d f x ≥ 0 :=
sorry

-- For part (2)(ii)
theorem exists_c_infinitely_many (f : ℕ → ℕ) (P : ∀ x : ℕ, f x + f (x + 2) ≤ 2 * f (x + 1)) :
  ∃ c : ℕ, 0 ≤ c ∧ c ≤ d f 1 ∧ ∀ N : ℕ, ∃ n : ℕ, n > N ∧ d f n = c :=
sorry

end no_exp_function_satisfies_P_d_decreasing_nonnegative_exists_c_infinitely_many_l1054_105481


namespace boys_passed_l1054_105444

theorem boys_passed (total_boys : ℕ) (avg_marks : ℕ) (avg_passed : ℕ) (avg_failed : ℕ) (P : ℕ) 
    (h1 : total_boys = 120) (h2 : avg_marks = 36) (h3 : avg_passed = 39) (h4 : avg_failed = 15)
    (h5 : P + (total_boys - P) = 120) 
    (h6 : P * avg_passed + (total_boys - P) * avg_failed = total_boys * avg_marks) :
    P = 105 := 
sorry

end boys_passed_l1054_105444


namespace multiple_of_area_l1054_105411

-- Define the given conditions
def perimeter (s : ℝ) : ℝ := 4 * s
def area (s : ℝ) : ℝ := s * s

theorem multiple_of_area (m s a p : ℝ) 
  (h1 : p = perimeter s)
  (h2 : a = area s)
  (h3 : m * a = 10 * p + 45)
  (h4 : p = 36) : m = 5 :=
by 
  sorry

end multiple_of_area_l1054_105411


namespace milk_distribution_l1054_105436

theorem milk_distribution 
  (x y z : ℕ)
  (h_total : x + y + z = 780)
  (h_equiv : 3 * x / 4 = 4 * y / 5 ∧ 3 * x / 4 = 4 * z / 7) :
  x = 240 ∧ y = 225 ∧ z = 315 := 
sorry

end milk_distribution_l1054_105436


namespace proportional_parts_l1054_105461

theorem proportional_parts (A B C D : ℕ) (number : ℕ) (h1 : A = 5 * x) (h2 : B = 7 * x) (h3 : C = 4 * x) (h4 : D = 8 * x) (h5 : C = 60) : number = 360 := by
  sorry

end proportional_parts_l1054_105461


namespace balls_in_boxes_l1054_105480

theorem balls_in_boxes :
  let n := 7
  let k := 3
  (Nat.choose (n + k - 1) (k - 1)) = 36 :=
by
  let n := 7
  let k := 3
  sorry

end balls_in_boxes_l1054_105480


namespace y_days_worked_l1054_105454

theorem y_days_worked 
  ( W : ℝ )
  ( x_rate : ℝ := W / 21 )
  ( y_rate : ℝ := W / 15 )
  ( d : ℝ )
  ( y_work_done : ℝ := d * y_rate )
  ( x_work_done_after_y_leaves : ℝ := 14 * x_rate )
  ( total_work_done : y_work_done + x_work_done_after_y_leaves = W ) :
  d = 5 := 
sorry

end y_days_worked_l1054_105454


namespace tan_theta_sub_9pi_l1054_105439

theorem tan_theta_sub_9pi (θ : ℝ) (h : Real.cos (Real.pi + θ) = -1 / 2) : 
  Real.tan (θ - 9 * Real.pi) = Real.sqrt 3 :=
by
  sorry

end tan_theta_sub_9pi_l1054_105439


namespace households_neither_car_nor_bike_l1054_105479

-- Define the given conditions
def total_households : ℕ := 90
def car_and_bike : ℕ := 18
def households_with_car : ℕ := 44
def bike_only : ℕ := 35

-- Prove the number of households with neither car nor bike
theorem households_neither_car_nor_bike :
  (total_households - ((households_with_car + bike_only) - car_and_bike)) = 11 :=
by
  sorry

end households_neither_car_nor_bike_l1054_105479


namespace tan_difference_l1054_105463

theorem tan_difference (α β : ℝ) (hα : Real.tan α = 5) (hβ : Real.tan β = 3) : 
    Real.tan (α - β) = 1 / 8 := 
by 
  sorry

end tan_difference_l1054_105463


namespace bead_necklaces_sold_l1054_105422

def cost_per_necklace : ℕ := 7
def total_earnings : ℕ := 70
def gemstone_necklaces_sold : ℕ := 7

theorem bead_necklaces_sold (B : ℕ) 
  (h1 : total_earnings = cost_per_necklace * (B + gemstone_necklaces_sold))  :
  B = 3 :=
by {
  sorry
}

end bead_necklaces_sold_l1054_105422


namespace num_undefined_values_l1054_105410

theorem num_undefined_values :
  ∃! x : Finset ℝ, (∀ y ∈ x, (y + 5 = 0) ∨ (y - 1 = 0) ∨ (y - 4 = 0)) ∧ (x.card = 3) := sorry

end num_undefined_values_l1054_105410


namespace car_win_probability_l1054_105400

-- Definitions from conditions
def total_cars : ℕ := 12
def p_X : ℚ := 1 / 6
def p_Y : ℚ := 1 / 10
def p_Z : ℚ := 1 / 8

-- Proof statement: The probability that one of the cars X, Y, or Z will win is 47/120
theorem car_win_probability : p_X + p_Y + p_Z = 47 / 120 := by
  sorry

end car_win_probability_l1054_105400


namespace exp_mul_l1054_105464

variable {a : ℝ}

-- Define a theorem stating the problem: proof that a^2 * a^3 = a^5
theorem exp_mul (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end exp_mul_l1054_105464


namespace probability_of_winning_noughts_l1054_105492

def ticTacToeProbability : ℚ :=
  let totalWays := Nat.choose 9 3
  let winningWays := 8
  winningWays / totalWays

theorem probability_of_winning_noughts :
  ticTacToeProbability = 2 / 21 := by
  sorry

end probability_of_winning_noughts_l1054_105492


namespace not_possible_coloring_possible_coloring_l1054_105402

-- Problem (a): For n = 2001 and k = 4001, prove that such coloring is not possible.
theorem not_possible_coloring (n : ℕ) (k : ℕ) (h_n : n = 2001) (h_k : k = 4001) :
  ¬ ∃ (color : ℕ × ℕ → ℕ), (∀ i j : ℕ, (1 ≤ i ∧ i ≤ n) ∧ (1 ≤ j ∧ j ≤ n) → 1 ≤ color (i, j) ∧ color (i, j) ≤ k)
  ∧ (∀ i, 1 ≤ i ∧ i ≤ n → ∀ j1 j2, (1 ≤ j1 ∧ j1 ≤ n) ∧ (1 ≤ j2 ∧ j2 ≤ n) → j1 ≠ j2 → color (i, j1) ≠ color (i, j2))
  ∧ (∀ j, 1 ≤ j ∧ j ≤ n → ∀ i1 i2, (1 ≤ i1 ∧ i1 ≤ n) ∧ (1 ≤ i2 ∧ i2 ≤ n) → i1 ≠ i2 → color (i1, j) ≠ color (i2, j)) := 
sorry

-- Problem (b): For n = 2^m - 1 and k = 2^(m+1) - 1, prove that such coloring is possible.
theorem possible_coloring (m : ℕ) (n k : ℕ) (h_n : n = 2^m - 1) (h_k : k = 2^(m+1) - 1) :
  ∃ (color : ℕ × ℕ → ℕ), (∀ i j : ℕ, (1 ≤ i ∧ i ≤ n) ∧ (1 ≤ j ∧ j ≤ n) → 1 ≤ color (i, j) ∧ color (i, j) ≤ k)
  ∧ (∀ i, 1 ≤ i ∧ i ≤ n → ∀ j1 j2, (1 ≤ j1 ∧ j1 ≤ n) ∧ (1 ≤ j2 ∧ j2 ≤ n) → j1 ≠ j2 → color (i, j1) ≠ color (i, j2))
  ∧ (∀ j, 1 ≤ j ∧ j ≤ n → ∀ i1 i2, (1 ≤ i1 ∧ i1 ≤ n) ∧ (1 ≤ i2 ∧ i2 ≤ n) → i1 ≠ i2 → color (i1, j) ≠ color (i2, j)) := 
sorry

end not_possible_coloring_possible_coloring_l1054_105402


namespace food_last_after_join_l1054_105446

-- Define the conditions
def initial_men := 760
def additional_men := 2280
def initial_days := 22
def days_before_join := 2
def initial_food := initial_men * initial_days
def remaining_food := initial_food - (initial_men * days_before_join)
def total_men := initial_men + additional_men

-- Define the goal to prove
theorem food_last_after_join :
  (remaining_food / total_men) = 5 :=
by
  sorry

end food_last_after_join_l1054_105446
