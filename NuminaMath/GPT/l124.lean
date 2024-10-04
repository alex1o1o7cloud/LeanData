import Mathlib

namespace no_valid_pair_for_tangential_quadrilateral_l124_124554

theorem no_valid_pair_for_tangential_quadrilateral (a d : ℝ) (h : d > 0) :
  ¬((∃ a d, a + (a + 2 * d) = (a + d) + (a + 3 * d))) :=
by
  sorry

end no_valid_pair_for_tangential_quadrilateral_l124_124554


namespace find_n_l124_124666

theorem find_n (n : ℕ) (h1 : Nat.lcm n 16 = 48) (h2 : Nat.gcd n 16 = 8): n = 24 := by
  sorry

end find_n_l124_124666


namespace nadia_flower_shop_l124_124328

theorem nadia_flower_shop (roses lilies cost_per_rose cost_per_lily cost_roses cost_lilies total_cost : ℕ)
  (h1 : roses = 20)
  (h2 : lilies = 3 * roses / 4)
  (h3 : cost_per_rose = 5)
  (h4 : cost_per_lily = 2 * cost_per_rose)
  (h5 : cost_roses = roses * cost_per_rose)
  (h6 : cost_lilies = lilies * cost_per_lily)
  (h7 : total_cost = cost_roses + cost_lilies) :
  total_cost = 250 :=
by
  sorry

end nadia_flower_shop_l124_124328


namespace find_y_when_x_is_1_l124_124934

theorem find_y_when_x_is_1 (t : ℝ) (x : ℝ) (y : ℝ) 
  (h1 : x = 3 - 2 * t) 
  (h2 : y = 5 * t + 6) 
  (h3 : x = 1) : 
  y = 11 :=
by
  sorry

end find_y_when_x_is_1_l124_124934


namespace calculate_shot_cost_l124_124261

theorem calculate_shot_cost :
  let num_pregnant_dogs := 3
  let puppies_per_dog := 4
  let shots_per_puppy := 2
  let cost_per_shot := 5
  let total_puppies := num_pregnant_dogs * puppies_per_dog
  let total_shots := total_puppies * shots_per_puppy
  let total_cost := total_shots * cost_per_shot
  total_cost = 120 :=
by
  sorry

end calculate_shot_cost_l124_124261


namespace square_root_of_4_is_pm2_l124_124946

theorem square_root_of_4_is_pm2 : ∃ (x : ℤ), x * x = 4 ∧ (x = 2 ∨ x = -2) := by
  sorry

end square_root_of_4_is_pm2_l124_124946


namespace problem1_problem2_problem3_l124_124275

-- Definitions and conditions
def digits : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}
def even_digits : Finset ℕ := digits.filter (λ x => x % 2 = 0)
def odd_digits : Finset ℕ := digits.filter (λ x => x % 2 = 1)

-- Lean statements equivalent to the given problems
theorem problem1 : 
  let num_even_chosen := Nat.choose 4 3,
      num_odd_chosen := Nat.choose 5 4,
      arrangements := Nat.factorial 7 in
  num_even_chosen * num_odd_chosen * arrangements = 100800 := by sorry

theorem problem2 : 
  let num_even_chosen := Nat.choose 4 3,
      num_odd_chosen := Nat.choose 5 4,
      grouped_arrangement := Nat.factorial 5,
      arrangements_of_group := Nat.factorial 3 in
  num_even_chosen * num_odd_chosen * grouped_arrangement * arrangements_of_group = 14400 := by sorry

theorem problem3 : 
  let num_odd_chosen := Nat.choose 5 4,
      positions_choosen := Nat.choose 5 3,
      odd_arrangements := Nat.factorial 4,
      even_arrangements := Nat.factorial 3 in
  num_odd_chosen * positions_choosen * odd_arrangements * even_arrangements = 28800 := by sorry

end problem1_problem2_problem3_l124_124275


namespace find_functions_l124_124834

variable (f : ℝ → ℝ)

def isFunctionPositiveReal := ∀ x : ℝ, x > 0 → f x > 0

axiom functional_eq (x y : ℝ) (hx : x > 0) (hy : y > 0) : f (x ^ y) = f x ^ f y

theorem find_functions (hf : isFunctionPositiveReal f) :
  (∀ x : ℝ, x > 0 → f x = 1) ∨ (∀ x : ℝ, x > 0 → f x = x) := sorry

end find_functions_l124_124834


namespace simplify_sqrt_72_plus_sqrt_32_l124_124040

theorem simplify_sqrt_72_plus_sqrt_32 : 
  sqrt 72 + sqrt 32 = 10 * sqrt 2 :=
by
  -- Define the intermediate results based on the conditions
  let sqrt72 := sqrt (2^3 * 3^2)
  let sqrt32 := sqrt (2^5)
  -- Specific simplifications from steps are not used directly, but they guide the statement
  show sqrt72 + sqrt32 = 10 * sqrt 2
  sorry

end simplify_sqrt_72_plus_sqrt_32_l124_124040


namespace minimize_expression_l124_124211

theorem minimize_expression (n : ℕ) (h : 0 < n) : 
  (n = 10) ↔ (∀ m : ℕ, 0 < m → ((n / 2) + (50 / n) ≤ (m / 2) + (50 / m))) :=
sorry

end minimize_expression_l124_124211


namespace trapezoid_shorter_base_l124_124691

theorem trapezoid_shorter_base (a b : ℕ) (mid_segment : ℕ) (longer_base : ℕ) 
    (h1 : mid_segment = 5) (h2 : longer_base = 105) 
    (h3 : mid_segment = (longer_base - a) / 2) : 
  a = 95 := 
by
  sorry

end trapezoid_shorter_base_l124_124691


namespace simplify_sqrt_sum_l124_124043

theorem simplify_sqrt_sum : sqrt 72 + sqrt 32 = 10 * sqrt 2 := sorry

end simplify_sqrt_sum_l124_124043


namespace part1_daily_sales_profit_final_max_daily_sales_profit_l124_124199

-- Conditions from part (a)
def original_selling_price : ℚ := 30
def cost_price : ℚ := 15
def original_sales_volume : ℚ := 60
def sales_increase_per_yuan : ℚ := 10

-- Part (1): Daily sales profit if the price is reduced by 2 yuan
def new_selling_price1 : ℚ := original_selling_price - 2
def new_sales_volume1 : ℚ := original_sales_volume + (2 * sales_increase_per_yuan)
def profit_per_kilogram1 : ℚ := new_selling_price1 - cost_price
def daily_sales_profit1 : ℚ := profit_per_kilogram1 * new_sales_volume1

theorem part1_daily_sales_profit : daily_sales_profit1 = 1040 := by
  sorry

-- Part (2): Maximum daily sales profit and corresponding selling price
def selling_price_at_max_profit : ℚ := 51 / 2

def daily_profit (x : ℚ) : ℚ :=
  (x - cost_price) * (original_sales_volume + (original_selling_price - x) * sales_increase_per_yuan)

theorem final_max_daily_sales_profit :
  (∀ x : ℚ, daily_profit x ≤ daily_profit selling_price_at_max_profit) ∧ daily_profit selling_price_at_max_profit = 1102.5 := by
  sorry

end part1_daily_sales_profit_final_max_daily_sales_profit_l124_124199


namespace students_taking_neither_l124_124978

theorem students_taking_neither (total_students music_students art_students dance_students music_art music_dance art_dance music_art_dance : ℕ) :
  total_students = 2500 →
  music_students = 200 →
  art_students = 150 →
  dance_students = 100 →
  music_art = 75 →
  art_dance = 50 →
  music_dance = 40 →
  music_art_dance = 25 →
  total_students - ((music_students + art_students + dance_students) - (music_art + art_dance + music_dance) + music_art_dance) = 2190 :=
by
  intros
  sorry

end students_taking_neither_l124_124978


namespace negation_of_both_even_l124_124482

-- Definitions
def even (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

-- Main statement
theorem negation_of_both_even (a b : ℕ) : ¬ (even a ∧ even b) ↔ (¬even a ∨ ¬even b) :=
by sorry

end negation_of_both_even_l124_124482


namespace tangents_to_discriminant_parabola_l124_124408

variable (a : ℝ) (p q : ℝ)

theorem tangents_to_discriminant_parabola :
  (a^2 + a * p + q = 0) ↔ (p^2 - 4 * q = 0) :=
sorry

end tangents_to_discriminant_parabola_l124_124408


namespace find_blue_beads_per_row_l124_124871

-- Given the conditions of the problem:
def number_of_purple_beads : ℕ := 50 * 20
def number_of_gold_beads : ℕ := 80
def total_cost : ℕ := 180

-- Define the main theorem to solve for the number of blue beads per row.
theorem find_blue_beads_per_row (x : ℕ) :
  (number_of_purple_beads + 40 * x + number_of_gold_beads = total_cost) → x = (total_cost - (number_of_purple_beads + number_of_gold_beads)) / 40 := 
by {
  -- Proof steps would go here
  sorry
}

end find_blue_beads_per_row_l124_124871


namespace terminal_side_second_quadrant_l124_124282

theorem terminal_side_second_quadrant (α : ℝ) (h1 : Real.tan α < 0) (h2 : Real.cos α < 0) : 
  (π / 2 < α ∧ α < π) := 
sorry

end terminal_side_second_quadrant_l124_124282


namespace remaining_pencils_l124_124075

/-
Given the initial number of pencils in the drawer and the number of pencils Sally took out,
prove that the number of pencils remaining in the drawer is 5.
-/
def pencils_in_drawer (initial_pencils : ℕ) (pencils_taken : ℕ) : ℕ :=
  initial_pencils - pencils_taken

theorem remaining_pencils : pencils_in_drawer 9 4 = 5 := by
  sorry

end remaining_pencils_l124_124075


namespace sqrt_sum_l124_124046

theorem sqrt_sum (a b : ℕ) (ha : a = 72) (hb : b = 32) : 
  Real.sqrt a + Real.sqrt b = 10 * Real.sqrt 2 := 
by 
  rw [ha, hb] 
  -- Insert any further required simplifications as a formal proof or leave it abstracted.
  exact sorry -- skipping the proof to satisfy this step.

end sqrt_sum_l124_124046


namespace evaluate_expression_c_eq_4_l124_124712

theorem evaluate_expression_c_eq_4 :
  (4^4 - 4 * (4-1)^(4-1))^(4-1) = 3241792 :=
by
  sorry

end evaluate_expression_c_eq_4_l124_124712


namespace tom_total_payment_l124_124641

theorem tom_total_payment :
  let apples_cost := 8 * 70
  let mangoes_cost := 9 * 55
  let oranges_cost := 5 * 40
  let bananas_cost := 12 * 30
  let grapes_cost := 7 * 45
  let cherries_cost := 4 * 80
  apples_cost + mangoes_cost + oranges_cost + bananas_cost + grapes_cost + cherries_cost = 2250 :=
by
  sorry

end tom_total_payment_l124_124641


namespace hour_hand_rotations_l124_124630

theorem hour_hand_rotations (degrees_per_hour : ℕ) (hours_per_day : ℕ) (days : ℕ) (rotations_per_day : ℕ) :
  degrees_per_hour = 30 →
  hours_per_day = 24 →
  rotations_per_day = (degrees_per_hour * hours_per_day) / 360 →
  days = 6 →
  rotations_per_day * days = 12 :=
by
  intros
  sorry

end hour_hand_rotations_l124_124630


namespace circle_area_l124_124529

/-
Circle A has a diameter equal to the radius of circle B.
The area of circle A is 16π square units.
Prove the area of circle B is 64π square units.
-/

theorem circle_area (rA dA rB : ℝ) (h1 : dA = 2 * rA) (h2 : rB = dA) (h3 : π * rA ^ 2 = 16 * π) : π * rB ^ 2 = 64 * π :=
by
  sorry

end circle_area_l124_124529


namespace no_such_x_exists_l124_124880

theorem no_such_x_exists : ¬ ∃ x : ℝ, 
  (∃ x1 : ℤ, x - 1/x = x1) ∧ 
  (∃ x2 : ℤ, 1/x - 1/(x^2 + 1) = x2) ∧ 
  (∃ x3 : ℤ, 1/(x^2 + 1) - 2*x = x3) :=
by
  sorry

end no_such_x_exists_l124_124880


namespace complement_union_l124_124452

open Set

-- Definitions and conditions
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 3, 5}
def B : Set ℕ := {2, 3}
noncomputable def C_UA : Set ℕ := U \ A

-- Statement to prove
theorem complement_union (U A B C_UA : Set ℕ) 
  (hU : U = {1, 2, 3, 4, 5})
  (hA : A = {1, 3, 5})
  (hB : B = {2, 3}) 
  (hCUA : C_UA = U \ A) : 
  (C_UA ∪ B) = {2, 3, 4} := 
sorry

end complement_union_l124_124452


namespace probability_A_and_B_same_county_l124_124986

/-
We have four experts and three counties. We need to assign the experts to the counties such 
that each county has at least one expert. We need to prove that the probability of experts 
A and B being dispatched to the same county is 1/6.
-/

def num_experts : Nat := 4
def num_counties : Nat := 3

def total_possible_events : Nat := 36
def favorable_events : Nat := 6

theorem probability_A_and_B_same_county :
  (favorable_events : ℚ) / total_possible_events = 1 / 6 := by sorry

end probability_A_and_B_same_county_l124_124986


namespace greatest_three_digit_number_condition_l124_124649

theorem greatest_three_digit_number_condition :
  ∃ n : ℕ, (100 ≤ n) ∧ (n ≤ 999) ∧ (n % 7 = 2) ∧ (n % 6 = 4) ∧ (n = 982) := 
by
  sorry

end greatest_three_digit_number_condition_l124_124649


namespace S_eq_Z_l124_124451

noncomputable def set_satisfies_conditions (S : Set ℤ) (a : Fin n → ℤ) :=
  (∀ i : Fin n, a i ∈ S) ∧
  (∀ i j : Fin n, (a i - a j) ∈ S) ∧
  (∀ x y : ℤ, x ∈ S → y ∈ S → x + y ∈ S → x - y ∈ S) ∧
  (Nat.gcd (List.foldr Nat.gcd 0 (Fin.val <$> List.finRange n)) = 1)

theorem S_eq_Z (S : Set ℤ) (a : Fin n → ℤ) (h_cond : set_satisfies_conditions S a) : S = Set.univ :=
  sorry

end S_eq_Z_l124_124451


namespace ratio_of_gilled_to_spotted_l124_124843

theorem ratio_of_gilled_to_spotted (total_mushrooms gilled_mushrooms spotted_mushrooms : ℕ) 
  (h1 : total_mushrooms = 30) 
  (h2 : gilled_mushrooms = 3) 
  (h3 : spotted_mushrooms = total_mushrooms - gilled_mushrooms) :
  gilled_mushrooms / gcd gilled_mushrooms spotted_mushrooms = 1 ∧ 
  spotted_mushrooms / gcd gilled_mushrooms spotted_mushrooms = 9 := 
by
  sorry

end ratio_of_gilled_to_spotted_l124_124843


namespace greatest_mean_weight_l124_124098

variable (X Y Z : Type) [Group X] [Group Y] [Group Z]

theorem greatest_mean_weight 
  (mean_X : ℝ) (mean_Y : ℝ) (mean_XY : ℝ) (mean_XZ : ℝ)
  (hX : mean_X = 30)
  (hY : mean_Y = 70)
  (hXY : mean_XY = 50)
  (hXZ : mean_XZ = 40) :
  ∃ k : ℝ, k = 70 :=
by {
  sorry
}

end greatest_mean_weight_l124_124098


namespace point_not_on_line_l124_124158

theorem point_not_on_line (m b : ℝ) (h1 : m > 2) (h2 : m * b > 0) : ¬ (b = -2023) :=
by
  sorry

end point_not_on_line_l124_124158


namespace total_students_l124_124312

-- Define n as total number of students
variable (n : ℕ)

-- Define conditions
variable (h1 : 550 ≤ n)
variable (h2 : (n / 10) + 10 ≤ n)

-- Define the proof statement
theorem total_students (h : (550 * 10 + 5) = n ∧ 
                        550 * 10 / n + 10 = 45 + n) : 
                        n = 1000 := by
  sorry

end total_students_l124_124312


namespace sums_of_integers_have_same_remainder_l124_124384

theorem sums_of_integers_have_same_remainder (n : ℕ) (n_pos : 0 < n) : 
  ∃ (i j : ℕ), (1 ≤ i ∧ i ≤ 2 * n) ∧ (1 ≤ j ∧ j ≤ 2 * n) ∧ i ≠ j ∧ ((i + i) % (2 * n) = (j + j) % (2 * n)) :=
by
  sorry

end sums_of_integers_have_same_remainder_l124_124384


namespace arithmetic_sequence_smallest_value_l124_124128

theorem arithmetic_sequence_smallest_value:
  ∃ a : ℕ, (7 * a + 63) % 11 = 0 ∧ (a - 9) % 11 = 4 := sorry

end arithmetic_sequence_smallest_value_l124_124128


namespace roots_quadratic_relation_l124_124090

theorem roots_quadratic_relation (a b c d A B : ℝ)
  (h1 : a^2 + A * a + 1 = 0)
  (h2 : b^2 + A * b + 1 = 0)
  (h3 : c^2 + B * c + 1 = 0)
  (h4 : d^2 + B * d + 1 = 0) :
  (a - c) * (b - c) * (a + d) * (b + d) = B^2 - A^2 :=
sorry

end roots_quadratic_relation_l124_124090


namespace verify_a_l124_124534

def g (x : ℝ) : ℝ := 5 * x - 7

theorem verify_a (a : ℝ) : g a = 0 ↔ a = 7 / 5 := by
  sorry

end verify_a_l124_124534


namespace find_n_from_ratio_l124_124423

theorem find_n_from_ratio (a b n : ℕ) (h : (a + 3 * b) ^ n = 4 ^ n)
  (h_ratio : 4 ^ n / 2 ^ n = 64) : 
  n = 6 := 
by
  sorry

end find_n_from_ratio_l124_124423


namespace find_velocity_l124_124485

variable (k V : ℝ)
variable (P A : ℕ)

theorem find_velocity (k_eq : k = 1 / 200) 
  (initial_cond : P = 4 ∧ A = 2 ∧ V = 20) 
  (new_cond : P = 16 ∧ A = 4) : 
  V = 20 * Real.sqrt 2 :=
by
  sorry

end find_velocity_l124_124485


namespace greatest_three_digit_number_l124_124651

theorem greatest_three_digit_number 
  (n : ℕ)
  (h1 : n % 7 = 2)
  (h2 : n % 6 = 4)
  (h3 : n ≥ 100)
  (h4 : n < 1000) :
  n = 994 :=
sorry

end greatest_three_digit_number_l124_124651


namespace domain_lg_sqrt_l124_124542

def domain_of_function (x : ℝ) : Prop :=
  1 - x > 0 ∧ x + 2 > 0

theorem domain_lg_sqrt (x : ℝ) : 
  domain_of_function x ↔ -2 < x ∧ x < 1 :=
sorry

end domain_lg_sqrt_l124_124542


namespace box_surface_area_l124_124676

theorem box_surface_area (a b c : ℕ) (h1 : a * b * c = 280) (h2 : a < 10) (h3 : b < 10) (h4 : c < 10) : 
  2 * (a * b + b * c + c * a) = 262 :=
sorry

end box_surface_area_l124_124676


namespace area_of_square_KLMN_is_25_l124_124604

-- Given a square ABCD with area 25
def ABCD_area_is_25 : Prop :=
  ∃ s : ℝ, (s * s = 25)

-- Given points K, L, M, and N forming isosceles right triangles with the sides of the square
def isosceles_right_triangles_at_vertices (A B C D K L M N : ℝ) : Prop :=
  ∃ (a b c d : ℝ),
    (a = b) ∧ (c = d) ∧
    (K - A)^2 + (B - K)^2 = (A - B)^2 ∧  -- AKB
    (L - B)^2 + (C - L)^2 = (B - C)^2 ∧  -- BLC
    (M - C)^2 + (D - M)^2 = (C - D)^2 ∧  -- CMD
    (N - D)^2 + (A - N)^2 = (D - A)^2    -- DNA

-- Given that KLMN is a square
def KLMN_is_square (K L M N : ℝ) : Prop :=
  (K - L)^2 + (L - M)^2 = (M - N)^2 + (N - K)^2

-- Proving that the area of square KLMN is 25 given the conditions
theorem area_of_square_KLMN_is_25 (A B C D K L M N : ℝ) :
  ABCD_area_is_25 → isosceles_right_triangles_at_vertices A B C D K L M N → KLMN_is_square K L M N → ∃s, s * s = 25 :=
by
  intro h1 h2 h3
  sorry

end area_of_square_KLMN_is_25_l124_124604


namespace john_new_weekly_earnings_l124_124172

theorem john_new_weekly_earnings :
  ∀ (original_earnings : ℤ) (percentage_increase : ℝ),
  original_earnings = 60 →
  percentage_increase = 66.67 →
  (original_earnings + (percentage_increase / 100 * original_earnings)) = 100 := 
by
  intros original_earnings percentage_increase h_earnings h_percentage
  rw [h_earnings, h_percentage]
  norm_num
  sorry

end john_new_weekly_earnings_l124_124172


namespace determine_a_l124_124937

theorem determine_a (a b c : ℤ)
  (vertex_condition : ∀ x : ℝ, x = 2 → ∀ y : ℝ, y = -3 → y = a * (x - 2) ^ 2 - 3)
  (point_condition : ∀ x : ℝ, x = 1 → ∀ y : ℝ, y = -2 → y = a * (x - 2) ^ 2 - 3) :
  a = 1 :=
by
  sorry

end determine_a_l124_124937


namespace max_sum_of_digits_l124_124984

theorem max_sum_of_digits (a b c : ℕ) (x : ℕ) (N : ℕ) :
  N = 100 * a + 10 * b + c →
  100 <= N →
  N < 1000 →
  a ≠ 0 →
  (100 * a + 10 * b + c) + (100 * a + 10 * c + b) = 1730 + x →
  a + b + c = 20 :=
by
  intros hN hN_ge_100 hN_lt_1000 ha_ne_0 hsum
  sorry

end max_sum_of_digits_l124_124984


namespace driving_scenario_l124_124660

theorem driving_scenario (x : ℝ) (h1 : x > 0) :
  (240 / x) - (240 / (1.5 * x)) = 1 :=
by
  sorry

end driving_scenario_l124_124660


namespace smaller_angle_in_parallelogram_l124_124368

theorem smaller_angle_in_parallelogram (a b : ℝ) (h1 : a + b = 180)
  (h2 : b = a + 70) : a = 55 :=
by sorry

end smaller_angle_in_parallelogram_l124_124368


namespace min_sum_xy_l124_124136

theorem min_sum_xy (x y : ℕ) (hx : x ≠ y) (hcond : ↑(1 / x) + ↑(1 / y) = 1 / 15) : x + y = 64 :=
sorry

end min_sum_xy_l124_124136


namespace min_friend_pairs_l124_124836

-- Define conditions
def n : ℕ := 2000
def invitations_per_person : ℕ := 1000
def total_invitations : ℕ := n * invitations_per_person

-- Mathematical problem statement
theorem min_friend_pairs : (total_invitations / 2) = 1000000 := 
by sorry

end min_friend_pairs_l124_124836


namespace largest_number_l124_124658

theorem largest_number (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h₁ : x₁ = 0.9791) 
  (h₂ : x₂ = 0.97019)
  (h₃ : x₃ = 0.97909)
  (h₄ : x₄ = 0.971)
  (h₅ : x₅ = 0.97109)
  : max x₁ (max x₂ (max x₃ (max x₄ x₅))) = 0.9791 :=
  sorry

end largest_number_l124_124658


namespace decimal_to_base_five_l124_124996

theorem decimal_to_base_five : 
  (2 * 5^3 + 1 * 5^1 + 0 * 5^2 + 0 * 5^0 = 255) := 
by
  sorry

end decimal_to_base_five_l124_124996


namespace minimum_common_perimeter_l124_124493

theorem minimum_common_perimeter :
  ∃ (a b c : ℕ), 
  let p := 2 * a + 10 * c in
  (a > b) ∧ 
  (b + 4c = a + 5c) ∧
  (5 * (a^2 - (5 * c)^2).sqrt = 4 * (b^2 - (4 * c)^2).sqrt) ∧
  p = 1180 :=
sorry

end minimum_common_perimeter_l124_124493


namespace martin_discounted_tickets_l124_124664

-- Definitions of the problem conditions
def total_tickets (F D : ℕ) := F + D = 10
def total_cost (F D : ℕ) := 2 * F + (16/10) * D = 184/10

-- Statement of the proof
theorem martin_discounted_tickets (F D : ℕ) (h1 : total_tickets F D) (h2 : total_cost F D) :
  D = 4 :=
sorry

end martin_discounted_tickets_l124_124664


namespace debby_candy_problem_l124_124125

theorem debby_candy_problem (D : ℕ) (sister_candy : ℕ) (eaten : ℕ) (remaining : ℕ) 
  (h1 : sister_candy = 42) (h2 : eaten = 35) (h3 : remaining = 39) :
  D + sister_candy - eaten = remaining ↔ D = 32 :=
by
  sorry

end debby_candy_problem_l124_124125


namespace deepak_present_age_l124_124207

-- Define the variables R and D
variables (R D : ℕ)

-- The conditions:
-- 1. After 4 years, Rahul's age will be 32 years.
-- 2. The ratio between Rahul and Deepak's ages is 4:3.
def rahul_age_after_4 : Prop := R + 4 = 32
def age_ratio : Prop := R / D = 4 / 3

-- The statement we want to prove:
theorem deepak_present_age (h1 : rahul_age_after_4 R) (h2 : age_ratio R D) : D = 21 :=
by sorry

end deepak_present_age_l124_124207


namespace incorrect_operation_l124_124506

noncomputable def a : ℤ := -2

def operation_A (a : ℤ) : ℤ := abs a
def operation_B (a : ℤ) : ℤ := abs (a - 2) + abs (a + 1)
def operation_C (a : ℤ) : ℤ := -a ^ 3 + a + (-a) ^ 2
def operation_D (a : ℤ) : ℤ := abs a ^ 2

theorem incorrect_operation :
  operation_D a ≠ abs 4 :=
by
  sorry

end incorrect_operation_l124_124506


namespace fraction_addition_l124_124963

theorem fraction_addition :
  let a := (1 : ℚ) / 6
  let b := (5 : ℚ) / 12
  a + b = 7 / 12 :=
by
  let a := (1 : ℚ) / 6
  let b := (5 : ℚ) / 12
  have : a + b = 7 / 12 := sorry
  exact this

end fraction_addition_l124_124963


namespace travel_agency_choice_l124_124116

noncomputable def y₁ (x : ℝ) : ℝ := 350 * x + 1000

noncomputable def y₂ (x : ℝ) : ℝ := 400 * x + 800

theorem travel_agency_choice (x : ℝ) (h : 0 < x) :
  (x < 4 → y₁ x > y₂ x) ∧ 
  (x = 4 → y₁ x = y₂ x) ∧ 
  (x > 4 → y₁ x < y₂ x) :=
by {
  sorry
}

end travel_agency_choice_l124_124116


namespace cost_of_360_songs_in_2005_l124_124581

theorem cost_of_360_songs_in_2005 :
  ∀ (c : ℕ), (200 * (c + 32) = 360 * c) → 360 * c / 100 = 144 :=
by
  assume c : ℕ
  assume h : 200 * (c + 32) = 360 * c
  sorry

end cost_of_360_songs_in_2005_l124_124581


namespace simplify_sqrt_sum_l124_124055

theorem simplify_sqrt_sum : sqrt 72 + sqrt 32 = 10 * sqrt 2 :=
by
  sorry

end simplify_sqrt_sum_l124_124055


namespace travel_time_reduction_l124_124799

theorem travel_time_reduction
  (original_speed : ℝ)
  (new_speed : ℝ)
  (time : ℝ)
  (distance : ℝ)
  (new_time : ℝ)
  (h1 : original_speed = 80)
  (h2 : new_speed = 50)
  (h3 : time = 3)
  (h4 : distance = original_speed * time)
  (h5 : new_time = distance / new_speed) :
  new_time = 4.8 := 
sorry

end travel_time_reduction_l124_124799


namespace greatest_divisor_of_arithmetic_sequence_l124_124819

theorem greatest_divisor_of_arithmetic_sequence (x c : ℕ) : ∃ d, d = 15 ∧ ∀ S, S = 15 * x + 105 * c → d ∣ S :=
by
  sorry

end greatest_divisor_of_arithmetic_sequence_l124_124819


namespace range_of_values_for_a_l124_124577

noncomputable def problem_statement (a : ℝ) : Prop :=
  ∀ x : ℝ, ¬ (|x - 2| + |x + 3| < a)

theorem range_of_values_for_a (a : ℝ) :
  problem_statement a → a ≤ 5 :=
  sorry

end range_of_values_for_a_l124_124577


namespace largest_common_divisor_l124_124820

theorem largest_common_divisor (a b : ℕ) (h1 : a = 360) (h2 : b = 315) : 
  ∃ d : ℕ, d ∣ a ∧ d ∣ b ∧ ∀ e : ℕ, (e ∣ a ∧ e ∣ b) → e ≤ d ∧ d = 45 :=
by
  sorry

end largest_common_divisor_l124_124820


namespace geom_seq_sum_eqn_l124_124003

theorem geom_seq_sum_eqn (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : a 2 + 2 * a 3 = a 1)
  (h2 : a 1 * a 4 = a 6)
  (h3 : ∀ n, a (n + 1) = a 1 * (1 / 2) ^ n)
  (h4 : ∀ n, S n = 2 * ((1 - (1 / 2) ^ n) / (1 - (1 / 2)))) :
  a n + S n = 4 :=
sorry

end geom_seq_sum_eqn_l124_124003


namespace find_a_l124_124735

noncomputable def f (a x : ℝ) : ℝ :=
  (1 / 2 : ℝ) * a * x^3 - (3 / 2 : ℝ) * x^2 + (3 / 2 : ℝ) * a^2 * x

theorem find_a (a : ℝ) (h_max : ∀ x : ℝ, f a x ≤ f a 1) : a = -2 :=
sorry

end find_a_l124_124735


namespace find_N_l124_124255

theorem find_N : ∃ N : ℕ, 36^2 * 72^2 = 12^2 * N^2 ∧ N = 216 :=
by
  sorry

end find_N_l124_124255


namespace product_xyz_l124_124431

noncomputable def xyz_value (x y z : ℝ) :=
  x * y * z

theorem product_xyz (x y z : ℝ) (h1 : x + 1 / y = 3) (h2 : y + 1 / z = 3) :
  xyz_value x y z = -1 :=
by
  sorry

end product_xyz_l124_124431


namespace problem_c_l124_124177

noncomputable def M (a b : ℝ) := (a^4 + b^4) * (a^2 + b^2)
noncomputable def N (a b : ℝ) := (a^3 + b^3) ^ 2

theorem problem_c (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h_neq : a ≠ b) : M a b > N a b := 
by
  -- Proof goes here
  sorry

end problem_c_l124_124177


namespace solution_to_exponential_equation_l124_124088

theorem solution_to_exponential_equation :
  ∃ x : ℕ, (8^12 + 8^12 + 8^12 = 2^x) ∧ x = 38 :=
by
  sorry

end solution_to_exponential_equation_l124_124088


namespace min_value_expression_l124_124600

variable (p q r : ℝ)
variable (hp : 0 < p) (hq : 0 < q) (hr : 0 < r)

theorem min_value_expression :
  (9 * r / (3 * p + 2 * q) + 9 * p / (2 * q + 3 * r) + 2 * q / (p + r)) ≥ 2 :=
sorry

end min_value_expression_l124_124600


namespace simplify_sqrt_sum_l124_124036

theorem simplify_sqrt_sum : (Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2) :=
by
  sorry

end simplify_sqrt_sum_l124_124036


namespace sum_of_first_n_natural_numbers_l124_124524

theorem sum_of_first_n_natural_numbers (n : ℕ) (h : n * (n + 1) / 2 = 190) : n = 19 :=
sorry

end sum_of_first_n_natural_numbers_l124_124524


namespace add_fractions_l124_124960

theorem add_fractions : (1 / 6 : ℚ) + (5 / 12) = 7 / 12 := 
by
  sorry

end add_fractions_l124_124960


namespace find_x_l124_124877

def bin_op (p1 p2 : ℤ × ℤ) : ℤ × ℤ :=
  (p1.1 - 2 * p2.1, p1.2 + 2 * p2.2)

theorem find_x :
  ∃ x y : ℤ, 
  bin_op (2, -4) (1, -3) = bin_op (x, y) (2, 1) ∧ x = 4 :=
by
  sorry

end find_x_l124_124877


namespace find_slope_of_q_l124_124323

theorem find_slope_of_q (j : ℝ) : 
  (∀ x y : ℝ, y = 2 * x + 3 → y = j * x + 1 → x = 1 → y = 5) → j = 4 := 
by
  intro h
  sorry

end find_slope_of_q_l124_124323


namespace smallest_sum_l124_124145

theorem smallest_sum (x y : ℕ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hxy_neq : x ≠ y) 
  (h_fraction : (1 : ℚ) / x + (1 : ℚ) / y = 1 / 15) : x + y = 64 :=
sorry

end smallest_sum_l124_124145


namespace factorization_correctness_l124_124362

theorem factorization_correctness :
  ∀ x : ℝ, x^2 - 2*x + 1 = (x - 1)^2 :=
by
  -- Proof omitted
  sorry

end factorization_correctness_l124_124362


namespace solve_laundry_problem_l124_124253

def laundry_problem : Prop :=
  let total_weight := 20
  let clothes_weight := 5
  let detergent_per_scoop := 0.02
  let initial_detergent := 2 * detergent_per_scoop
  let optimal_ratio := 0.004
  let additional_detergent := 0.02
  let additional_water := 14.94
  let total_detergent := initial_detergent + additional_detergent
  let final_amount := clothes_weight + initial_detergent + additional_detergent + additional_water
  final_amount = total_weight ∧ total_detergent / (total_weight - clothes_weight) = optimal_ratio

theorem solve_laundry_problem : laundry_problem :=
by 
  -- the proof would go here
  sorry

end solve_laundry_problem_l124_124253


namespace range_of_m_l124_124736

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x)

theorem range_of_m (m : ℝ) (h : ∀ x > 0, f x > m * x) : m ≤ 2 := sorry

end range_of_m_l124_124736


namespace initial_deadline_is_75_days_l124_124454

-- Define constants for the problem
def initial_men : ℕ := 100
def initial_hours_per_day : ℕ := 8
def days_worked_initial : ℕ := 25
def fraction_work_completed : ℚ := 1 / 3
def additional_men : ℕ := 60
def new_hours_per_day : ℕ := 10
def total_man_hours : ℕ := 60000

-- Prove that the initial deadline for the project is 75 days
theorem initial_deadline_is_75_days : 
  ∃ (D : ℕ), (D * initial_men * initial_hours_per_day = total_man_hours) ∧ D = 75 := 
by {
  sorry
}

end initial_deadline_is_75_days_l124_124454


namespace simplify_sqrt_sum_l124_124056

theorem simplify_sqrt_sum : sqrt 72 + sqrt 32 = 10 * sqrt 2 :=
by
  sorry

end simplify_sqrt_sum_l124_124056


namespace Jake_weight_loss_l124_124299

variable (J S: ℕ) (x : ℕ)

theorem Jake_weight_loss:
  J = 93 -> J + S = 132 -> J - x = 2 * S -> x = 15 :=
by
  intros hJ hJS hCondition
  sorry

end Jake_weight_loss_l124_124299


namespace tan_neg_225_is_neg_1_l124_124349

def tan_neg_225_eq_neg_1 : Prop :=
  Real.tan (-225 * Real.pi / 180) = -1

theorem tan_neg_225_is_neg_1 : tan_neg_225_eq_neg_1 :=
  by
    sorry

end tan_neg_225_is_neg_1_l124_124349


namespace continuous_piecewise_function_l124_124184

theorem continuous_piecewise_function (a c : ℝ) (h1 : 2 * a * 2 + 6 = 3 * 2 - 2) (h2 : 4 * (-2) + 2 * c = 3 * (-2) - 2) : 
  a + c = -1/2 := 
sorry

end continuous_piecewise_function_l124_124184


namespace cannot_determine_congruency_l124_124858

-- Define the congruency criteria for triangles
def SSS (a1 b1 c1 a2 b2 c2 : ℝ) : Prop := a1 = a2 ∧ b1 = b2 ∧ c1 = c2
def SAS (a1 b1 angle1 a2 b2 angle2 : ℝ) : Prop := a1 = a2 ∧ b1 = b2 ∧ angle1 = angle2
def ASA (angle1 b1 angle2 angle3 b2 angle4 : ℝ) : Prop := angle1 = angle2 ∧ b1 = b2 ∧ angle3 = angle4
def AAS (angle1 angle2 b1 angle3 angle4 b2 : ℝ) : Prop := angle1 = angle2 ∧ angle3 = angle4 ∧ b1 = b2
def HL (hyp1 leg1 hyp2 leg2 : ℝ) : Prop := hyp1 = hyp2 ∧ leg1 = leg2

-- Define the condition D, which states the equality of two corresponding sides and a non-included angle
def conditionD (a1 b1 angle1 a2 b2 angle2 : ℝ) : Prop := a1 = a2 ∧ b1 = b2 ∧ angle1 = angle2

-- The theorem to be proven
theorem cannot_determine_congruency (a1 b1 angle1 a2 b2 angle2 : ℝ) :
  conditionD a1 b1 angle1 a2 b2 angle2 → ¬(SSS a1 b1 0 a2 b2 0 ∨ SAS a1 b1 0 a2 b2 0 ∨ ASA 0 b1 0 0 b2 0 ∨ AAS 0 0 b1 0 0 b2 ∨ HL 0 0 0 0) :=
by
  sorry

end cannot_determine_congruency_l124_124858


namespace average_non_prime_squares_approx_l124_124274

-- Define a function to check if a number is prime
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the list of non-prime numbers between 50 and 100
def non_prime_numbers : List ℕ :=
  [51, 52, 54, 55, 56, 57, 58, 60, 62, 63, 64, 65, 66, 68, 69, 70,
   72, 74, 75, 76, 77, 78, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91,
   92, 93, 94, 95, 96, 98, 99]

-- Define the sum of squares of the elements in a list
def sum_of_squares (l : List ℕ) : ℕ :=
  l.foldr (λ x acc => x * x + acc) 0

-- Define the count of non-prime numbers
def count_non_prime : ℕ :=
  non_prime_numbers.length

-- Calculate the average
def average_non_prime_squares : ℚ :=
  sum_of_squares non_prime_numbers / count_non_prime

-- Theorem to state that the average of the sum of squares of non-prime numbers
-- between 50 and 100 is approximately 6417.67
theorem average_non_prime_squares_approx :
  abs ((average_non_prime_squares : ℝ) - 6417.67) < 0.01 := 
  sorry

end average_non_prime_squares_approx_l124_124274


namespace find_n_l124_124557

-- Define the first term a₁, the common ratio q, and the sum Sₙ
def a₁ : ℕ := 2
def q : ℕ := 2
def Sₙ (n : ℕ) : ℕ := 2^(n + 1) - 2

-- The sum of the first n terms is given as 126
def given_sum : ℕ := 126

-- The theorem to be proven
theorem find_n (n : ℕ) (h : Sₙ n = given_sum) : n = 6 :=
by
  sorry

end find_n_l124_124557


namespace find_d_l124_124757

theorem find_d (d : ℝ) (h1 : ∃ (x y : ℝ), y = x + d ∧ x = -y + d ∧ x = d-1 ∧ y = d) : d = 1 :=
sorry

end find_d_l124_124757


namespace probability_exactly_one_male_one_female_same_topic_l124_124488

theorem probability_exactly_one_male_one_female_same_topic :
  let numOutcomes := 8
  let desiredOutcomes := 4
  let probability := (desiredOutcomes : ℝ) / numOutcomes
  probability = 1 / 2 := by
  sorry

end probability_exactly_one_male_one_female_same_topic_l124_124488


namespace minimum_value_attained_at_n_10_l124_124210

theorem minimum_value_attained_at_n_10 : ∀ (n : ℕ) (h : 0 < n), 
  (n = 10) → (n / 2 + 50 / n = 10) :=
begin
  intros n h hn,
  have h1 : n / 2 = 50 / n, by sorry,
  have h2 : 2 * sqrt (n / 2 * 50 / n) = 10, by sorry,
  exact eq.trans (add_eq_of_eq_of_eq h1 h2) hn,
end

end minimum_value_attained_at_n_10_l124_124210


namespace p_sufficient_but_not_necessary_for_q_l124_124294

-- Definitions
variable {p q : Prop}

-- The condition: ¬p is a necessary but not sufficient condition for ¬q
def necessary_but_not_sufficient (p q : Prop) : Prop :=
  (∀ q, ¬q → ¬p) ∧ (∃ q, ¬q ∧ p)

-- The theorem stating the problem
theorem p_sufficient_but_not_necessary_for_q 
  (h : necessary_but_not_sufficient (¬p) (¬q)) : 
  (∀ p, p → q) ∧ (∃ p, p ∧ ¬q) :=
sorry

end p_sufficient_but_not_necessary_for_q_l124_124294


namespace quadratic_real_roots_discriminant_quadratic_real_roots_sum_of_squares_l124_124133

theorem quadratic_real_roots_discriminant (m : ℝ) :
  (2 * (m + 1))^2 - 4 * m * (m - 1) > 0 ↔ (m > -1/2 ∧ m ≠ 0) := 
sorry

theorem quadratic_real_roots_sum_of_squares (m x1 x2 : ℝ) 
  (h1 : m > -1/2 ∧ m ≠ 0)
  (h2 : x1 + x2 = -2 * (m + 1) / m)
  (h3 : x1 * x2 = (m - 1) / m)
  (h4 : x1^2 + x2^2 = 8) : 
  m = (6 + 2 * Real.sqrt 33) / 8 := 
sorry

end quadratic_real_roots_discriminant_quadratic_real_roots_sum_of_squares_l124_124133


namespace greatest_three_digit_number_condition_l124_124648

theorem greatest_three_digit_number_condition :
  ∃ n : ℕ, (100 ≤ n) ∧ (n ≤ 999) ∧ (n % 7 = 2) ∧ (n % 6 = 4) ∧ (n = 982) := 
by
  sorry

end greatest_three_digit_number_condition_l124_124648


namespace common_point_sufficient_condition_l124_124811

theorem common_point_sufficient_condition (k : ℝ) : 
  (∃ x y : ℝ, x^2 + y^2 = 1 ∧ y = k * x - 3) → k ≤ -2 * Real.sqrt 2 :=
by
  -- Proof will go here
  sorry

end common_point_sufficient_condition_l124_124811


namespace part1_daily_sales_profit_part2_maximum_daily_profit_l124_124200

-- Definitions of initial conditions
def original_price : ℝ := 30
def original_sales_volume : ℝ := 60
def cost_price : ℝ := 15
def price_reduction_effect : ℝ := 10

-- Part 1: Prove the daily sales profit if the price is reduced by 2 yuan
def new_price_after_reduction (reduction : ℝ) : ℝ := original_price - reduction
def new_sales_volume (reduction : ℝ) : ℝ := original_sales_volume + reduction * price_reduction_effect
def profit_per_kg (selling_price : ℝ) : ℝ := selling_price - cost_price
def daily_sales_profit (reduction : ℝ) : ℝ := profit_per_kg (new_price_after_reduction reduction) * new_sales_volume reduction

theorem part1_daily_sales_profit : daily_sales_profit 2 = 1040 := by sorry

-- Part 2: Prove the selling price for maximum profit and the maximum profit
def profit_function (x : ℝ) : ℝ := (x - cost_price) * (original_sales_volume + (original_price - x) * price_reduction_effect)

theorem part2_maximum_daily_profit : 
  ∃ x, profit_function x = 1102.5 ∧ x = 51 / 2 := by sorry

end part1_daily_sales_profit_part2_maximum_daily_profit_l124_124200


namespace channel_depth_l124_124065

theorem channel_depth
  (top_width bottom_width area : ℝ)
  (h : ℝ)
  (trapezium_area_formula : area = (1 / 2) * (top_width + bottom_width) * h)
  (top_width_val : top_width = 14)
  (bottom_width_val : bottom_width = 8)
  (area_val : area = 770) :
  h = 70 := 
by
  sorry

end channel_depth_l124_124065


namespace max_intersection_points_three_circles_two_lines_l124_124355

theorem max_intersection_points_three_circles_two_lines : 
  ∀ (C1 C2 C3 L1 L2 : set ℝ × ℝ) (hC1 : is_circle C1) (hC2 : is_circle C2) (hC3 : is_circle C3) (hL1 : is_line L1) (hL2 : is_line L2),
  ∃ P : ℕ, P = 19 ∧
  (∀ (P : ℝ × ℝ), P ∈ C1 ∧ P ∈ C2 ∨ P ∈ C1 ∧ P ∈ C3 ∨ P ∈ C2 ∧ P ∈ C3 ∨ P ∈ C1 ∧ P ∈ L1 ∨ P ∈ C2 ∧ P ∈ L1 ∨ P ∈ C3 ∧ P ∈ L1 ∨ P ∈ C1 ∧ P ∈ L2 ∨ P ∈ C2 ∧ P ∈ L2 ∨ P ∈ C3 ∧ P ∈ L2 ∨ P ∈ L1 ∧ P ∈ L2) ↔ P = 19 :=
sorry

end max_intersection_points_three_circles_two_lines_l124_124355


namespace length_of_chord_l124_124290

theorem length_of_chord {x1 x2 : ℝ} (h1 : ∃ (y : ℝ), y^2 = 8 * x1)
                                   (h2 : ∃ (y : ℝ), y^2 = 8 * x2)
                                   (h_midpoint : (x1 + x2) / 2 = 3) :
  x1 + x2 + 4 = 10 :=
sorry

end length_of_chord_l124_124290


namespace percentage_k_equal_125_percent_j_l124_124298

theorem percentage_k_equal_125_percent_j
  (j k l m : ℝ)
  (h1 : 1.25 * j = (x / 100) * k)
  (h2 : 1.5 * k = 0.5 * l)
  (h3 : 1.75 * l = 0.75 * m)
  (h4 : 0.2 * m = 7 * j) :
  x = 25 := 
sorry

end percentage_k_equal_125_percent_j_l124_124298


namespace decreasing_function_range_l124_124562

theorem decreasing_function_range {f : ℝ → ℝ} (h_decreasing : ∀ x y : ℝ, x < y → f x > f y) :
  {x : ℝ | f (x^2 - 3 * x - 3) < f 1} = {x : ℝ | x < -1 ∨ x > 4} :=
by
  sorry

end decreasing_function_range_l124_124562


namespace find_f_neg2_l124_124289

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then 3^x - 1 else -3^(-x) + 1

theorem find_f_neg2 : f (-2) = -8 := by
  sorry

end find_f_neg2_l124_124289


namespace arithmetic_sequence_n_sum_arithmetic_sequence_S17_arithmetic_sequence_S13_l124_124441

-- Question 1
theorem arithmetic_sequence_n (a1 a4 a10 : ℤ) (d : ℤ) (n : ℤ) (Sn : ℤ) 
  (h1 : a1 + 3 * d = a4) 
  (h2 : a1 + 9 * d = a10)
  (h3 : Sn = n * (2 * a1 + (n - 1) * d) / 2)
  (h4 : a4 = 10)
  (h5 : a10 = -2)
  (h6 : Sn = 60)
  : n = 5 ∨ n = 6 := 
sorry

-- Question 2
theorem sum_arithmetic_sequence_S17 (a1 : ℤ) (d : ℤ) (a_n1 : ℤ → ℤ) (S17 : ℤ)
  (h1 : a1 = -7)
  (h2 : ∀ n, a_n1 (n + 1) = a_n1 n + d)
  (h3 : S17 = 17 * (2 * a1 + 16 * d) / 2)
  : S17 = 153 := 
sorry

-- Question 3
theorem arithmetic_sequence_S13 (a_2 a_7 a_12 : ℤ) (S13 : ℤ)
  (h1 : a_2 + a_7 + a_12 = 24)
  (h2 : S13 = a_7 * 13)
  : S13 = 104 := 
sorry

end arithmetic_sequence_n_sum_arithmetic_sequence_S17_arithmetic_sequence_S13_l124_124441


namespace smallest_integer_in_set_l124_124761

open Real

theorem smallest_integer_in_set : 
  ∃ (n : ℤ), (λ n, let avg := (n + (n+1) + (n+2) + (n+3) + (n+4) + (n+5) + (n+6)) / 7 
               in n + 6 > 2 * avg) n ∧ 
            ∀ k : ℤ, (λ k, let avg_k := (k + (k+1) + (k+2) + (k+3) + (k+4) + (k+5) + (k+6)) / 7 
               in k + 6 > 2 * avg_k) k → n ≤ k := 
by
  sorry

end smallest_integer_in_set_l124_124761


namespace integrate_diff_eq_l124_124311

noncomputable def particular_solution (x y : ℝ) : Prop :=
  (y^2 - x^2) / 2 + Real.exp y - Real.log ((x + Real.sqrt (1 + x^2)) / (2 + Real.sqrt 5)) = Real.exp 1 - 3 / 2

theorem integrate_diff_eq (x y : ℝ) :
  (∀ x y : ℝ, y' = (x * Real.sqrt (1 + x^2) + 1) / (Real.sqrt (1 + x^2) * (y + Real.exp y))) → 
  (∃ x0 y0 : ℝ, x0 = 2 ∧ y0 = 1) → 
  particular_solution x y :=
sorry

end integrate_diff_eq_l124_124311


namespace project_completion_l124_124975

theorem project_completion (x : ℕ) :
  (21 - x) * (1 / 12 : ℚ) + x * (1 / 30 : ℚ) = 1 → x = 15 :=
by
  sorry

end project_completion_l124_124975


namespace fraction_of_robs_doubles_is_one_third_l124_124464

theorem fraction_of_robs_doubles_is_one_third 
  (total_robs_cards : ℕ) (total_jess_doubles : ℕ) 
  (times_jess_doubles_robs : ℕ)
  (robs_doubles : ℕ) :
  total_robs_cards = 24 →
  total_jess_doubles = 40 →
  times_jess_doubles_robs = 5 →
  total_jess_doubles = times_jess_doubles_robs * robs_doubles →
  (robs_doubles : ℚ) / total_robs_cards = 1 / 3 := 
by 
  intros h1 h2 h3 h4
  sorry

end fraction_of_robs_doubles_is_one_third_l124_124464


namespace simplify_sqrt72_add_sqrt32_l124_124058

theorem simplify_sqrt72_add_sqrt32 : (sqrt 72) + (sqrt 32) = 10 * (sqrt 2) :=
by sorry

end simplify_sqrt72_add_sqrt32_l124_124058


namespace expected_value_of_winning_is_2550_l124_124235

-- Definitions based on the conditions
def outcomes : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8]
def probability (n : ℕ) : ℚ := 1 / 8
def winnings (n : ℕ) : ℕ := n^2

-- Expected value calculation based on the conditions
noncomputable def expected_value : ℚ :=
  (outcomes.map (λ n => probability n * winnings n)).sum

-- Proposition stating that the expected value is 25.50
theorem expected_value_of_winning_is_2550 : expected_value = 25.50 :=
by
  sorry

end expected_value_of_winning_is_2550_l124_124235


namespace tangent_fraction_15_degrees_l124_124258

theorem tangent_fraction_15_degrees : (1 + Real.tan (Real.pi / 12 )) / (1 - Real.tan (Real.pi / 12)) = Real.sqrt 3 :=
by
  sorry

end tangent_fraction_15_degrees_l124_124258


namespace sufficient_but_not_necessary_condition_l124_124005

theorem sufficient_but_not_necessary_condition 
  (a : ℝ) 
  (h1 : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x ^ 2 - a ≤ 0) : 
  a ≥ 5 :=
sorry

end sufficient_but_not_necessary_condition_l124_124005


namespace ball_drawing_prob_exp_l124_124582

-- Define the variables and conditions of the problem
def numBalls := 4
def numRedBalls := 1
def numGreenBalls := 1
def numYellowBalls := 2
def probDrawFirstRed := 1 / 4
def probDrawFirstGreenThenRed := (1 / 4) * (1 / 3)
def p_xi_0 := probDrawFirstRed + probDrawFirstGreenThenRed

def exp_xi := 0 * (1 / 3) + 1 * (1 / 3) + 2 * (1 / 3)

-- The theorem statement
theorem ball_drawing_prob_exp :
  p_xi_0 = 1 / 3 ∧ exp_xi = 1 := by
  sorry

end ball_drawing_prob_exp_l124_124582


namespace lipstick_cost_is_correct_l124_124915

noncomputable def cost_of_lipstick (palette_cost : ℝ) (num_palettes : ℝ) (hair_color_cost : ℝ) (num_hair_colors : ℝ) (total_paid : ℝ) (num_lipsticks : ℝ) : ℝ :=
  let total_palette_cost := num_palettes * palette_cost
  let total_hair_color_cost := num_hair_colors * hair_color_cost
  let remaining_amount := total_paid - (total_palette_cost + total_hair_color_cost)
  remaining_amount / num_lipsticks

theorem lipstick_cost_is_correct :
  cost_of_lipstick 15 3 4 3 67 4 = 2.5 :=
by
  sorry

end lipstick_cost_is_correct_l124_124915


namespace count_six_letter_strings_l124_124382

open Nat

def vowels_count : List ℕ := [6, 6, 6, 6, 3]

noncomputable def count_strings (n : ℕ) : ℕ :=
  (Finset.range 4).sum (λ k => choose n k * 6^(n - k))

theorem count_six_letter_strings : count_strings 6 = 117072 := by
  sorry

end count_six_letter_strings_l124_124382


namespace tower_height_count_l124_124918

theorem tower_height_count (bricks : ℕ) (height1 height2 height3 : ℕ) :
  height1 = 3 → height2 = 11 → height3 = 18 → bricks = 100 →
  (∃ (h : ℕ),  h = 1404) :=
by
  sorry

end tower_height_count_l124_124918


namespace find_k_of_inverse_proportion_l124_124567

theorem find_k_of_inverse_proportion (k x y : ℝ) (h : y = k / x) (hx : x = 2) (hy : y = 6) : k = 12 :=
by
  sorry

end find_k_of_inverse_proportion_l124_124567


namespace find_constant_term_l124_124123

theorem find_constant_term (c : ℤ) (y : ℤ) (h1 : y = 2) (h2 : 5 * y^2 - 8 * y + c = 59) : c = 55 :=
by
  sorry

end find_constant_term_l124_124123


namespace find_eccentricity_l124_124899

variable (a b : ℝ) (ha : a > 0) (hb : b > 0)
variable (asymp_cond : b / a = 1 / 2)

theorem find_eccentricity : ∃ e : ℝ, e = Real.sqrt 5 / 2 :=
by
  let c := Real.sqrt ((a^2 + b^2) / 4)
  let e := c / a
  use e
  sorry

end find_eccentricity_l124_124899


namespace garden_width_l124_124173

theorem garden_width (w : ℕ) (h_area : w * (w + 10) ≥ 150) : w = 10 :=
sorry

end garden_width_l124_124173


namespace max_ab_ac_bc_l124_124775

theorem max_ab_ac_bc (a b c : ℝ) (h : a + 3 * b + c = 6) : 
    ab + ac + bc <= 8 :=
sorry

end max_ab_ac_bc_l124_124775


namespace description_of_T_l124_124448

-- Define the conditions
def T := { p : ℝ × ℝ | (∃ (c : ℝ), ((c = 5 ∨ c = p.1 + 3 ∨ c = p.2 - 6) ∧ (5 ≥ p.1 + 3) ∧ (5 ≥ p.2 - 6))) }

-- The main theorem
theorem description_of_T : 
  ∃ p : ℝ × ℝ, 
    (p = (2, 11)) ∧ 
    ∀ q ∈ T, 
      (q.fst = 2 ∧ q.snd ≤ 11) ∨ 
      (q.snd = 11 ∧ q.fst ≤ 2) ∨ 
      (q.snd = q.fst + 9 ∧ q.fst ≤ 2) :=
sorry

end description_of_T_l124_124448


namespace compute_difference_l124_124320

noncomputable def f (n : ℝ) : ℝ := (1 / 4) * n * (n + 1) * (n + 2) * (n + 3)

theorem compute_difference (r : ℝ) : f r - f (r - 1) = r * (r + 1) * (r + 2) := by
  sorry

end compute_difference_l124_124320


namespace gamma_max_success_ratio_l124_124308

theorem gamma_max_success_ratio (x y z w : ℕ) (h_yw : y + w = 500)
    (h_gamma_first_day : 0 < x ∧ x < 170 * y / 280)
    (h_gamma_second_day : 0 < z ∧ z < 150 * w / 220)
    (h_less_than_500 : (28 * x + 22 * z) / 17 < 500) :
    (x + z) ≤ 170 := 
sorry

end gamma_max_success_ratio_l124_124308


namespace fourth_vertex_l124_124590

-- Define the given vertices
def vertex1 := (2, 1)
def vertex2 := (4, 1)
def vertex3 := (2, 5)

-- Define what it means to be a rectangle in this context
def is_vertical_segment (p1 p2 : ℕ × ℕ) : Prop :=
  p1.1 = p2.1

def is_horizontal_segment (p1 p2 : ℕ × ℕ) : Prop :=
  p1.2 = p2.2

def is_rectangle (v1 v2 v3 v4: (ℕ × ℕ)) : Prop :=
  is_vertical_segment v1 v3 ∧
  is_horizontal_segment v1 v2 ∧
  is_vertical_segment v2 v4 ∧
  is_horizontal_segment v3 v4 ∧
  is_vertical_segment v1 v4 ∧ -- additional condition to ensure opposite sides are equal
  is_horizontal_segment v2 v3

-- Prove the coordinates of the fourth vertex of the rectangle
theorem fourth_vertex (v4 : ℕ × ℕ) : 
  is_rectangle vertex1 vertex2 vertex3 v4 → v4 = (4, 5) := 
by
  intro h_rect
  sorry

end fourth_vertex_l124_124590


namespace simplify_sqrt_72_plus_sqrt_32_l124_124039

theorem simplify_sqrt_72_plus_sqrt_32 : 
  sqrt 72 + sqrt 32 = 10 * sqrt 2 :=
by
  -- Define the intermediate results based on the conditions
  let sqrt72 := sqrt (2^3 * 3^2)
  let sqrt32 := sqrt (2^5)
  -- Specific simplifications from steps are not used directly, but they guide the statement
  show sqrt72 + sqrt32 = 10 * sqrt 2
  sorry

end simplify_sqrt_72_plus_sqrt_32_l124_124039


namespace find_angle_A_l124_124163

theorem find_angle_A (a b : ℝ) (sin_B : ℝ) (ha : a = 3) (hb : b = 4) (hsinB : sin_B = 2/3) :
  ∃ A : ℝ, A = π / 6 :=
by
  sorry

end find_angle_A_l124_124163


namespace find_functions_l124_124120

variable (f : ℝ → ℝ)

theorem find_functions (h : ∀ x y : ℝ, f (x + f y) = f x + f y ^ 2 + 2 * x * f y) :
  ∃ c : ℝ, (∀ x, f x = x ^ 2 + c) ∨ (∀ x, f x = 0) :=
by
  sorry

end find_functions_l124_124120


namespace solve_cos_theta_l124_124979

def cos_theta_proof (v1 v2 : ℝ × ℝ) (θ : ℝ) : Prop :=
  let dot_product := (v1.1 * v2.1 + v1.2 * v2.2)
  let norm_v1 := Real.sqrt (v1.1 ^ 2 + v1.2 ^ 2)
  let norm_v2 := Real.sqrt (v2.1 ^ 2 + v2.2 ^ 2)
  let cos_theta := dot_product / (norm_v1 * norm_v2)
  cos_theta = 43 / Real.sqrt 2173

theorem solve_cos_theta :
  cos_theta_proof (4, 5) (2, 7) (43 / Real.sqrt 2173) :=
by
  sorry

end solve_cos_theta_l124_124979


namespace other_number_of_given_conditions_l124_124808

theorem other_number_of_given_conditions 
  (a b : ℕ) 
  (h_lcm : Nat.lcm a b = 4620) 
  (h_gcd : Nat.gcd a b = 21) 
  (h_a : a = 210) : 
  b = 462 := 
sorry

end other_number_of_given_conditions_l124_124808


namespace fish_minimum_catch_l124_124713

theorem fish_minimum_catch (a1 a2 a3 a4 a5 : ℕ) (h_sum : a1 + a2 + a3 + a4 + a5 = 100)
  (h_non_increasing : a1 ≥ a2 ∧ a2 ≥ a3 ∧ a3 ≥ a4 ∧ a4 ≥ a5) : 
  a1 + a3 + a5 ≥ 50 :=
sorry

end fish_minimum_catch_l124_124713


namespace lara_cookies_l124_124447

theorem lara_cookies (total_cookies trays rows_per_row : ℕ)
  (h_total : total_cookies = 120)
  (h_trays : trays = 4)
  (h_rows_per_row : rows_per_row = 6) :
  total_cookies / rows_per_row / trays = 5 :=
by
  sorry

end lara_cookies_l124_124447


namespace not_solvable_equations_l124_124544

theorem not_solvable_equations :
  ¬(∃ x : ℝ, (x - 5) ^ 2 = -1) ∧ ¬(∃ x : ℝ, |2 * x| + 3 = 0) :=
by
  sorry

end not_solvable_equations_l124_124544


namespace smallest_positive_period_max_min_value_interval_l124_124564

noncomputable def f (x : ℝ) : ℝ :=
  2 * (Real.sin (x + Real.pi / 3))^2 - (Real.cos x)^2 + (Real.sin x)^2

theorem smallest_positive_period : (∀ x : ℝ, f (x + Real.pi) = f x) :=
by sorry

theorem max_min_value_interval :
  (∀ x ∈ Set.Icc (-Real.pi / 3) (Real.pi / 6), 
    f x ≤ 3 / 2 ∧ f x ≥ 0 ∧ 
    (f (-Real.pi / 6) = 0) ∧ 
    (f (Real.pi / 6) = 3 / 2)) :=
by sorry

end smallest_positive_period_max_min_value_interval_l124_124564


namespace value_of_a_l124_124538

theorem value_of_a :
  ∀ (g : ℝ → ℝ), (∀ x, g x = 5*x - 7) → ∃ a, g a = 0 ∧ a = 7 / 5 :=
by
  sorry

end value_of_a_l124_124538


namespace bahs_equivalent_to_1500_yahs_l124_124751

-- Definitions from conditions
def bahs := ℕ
def rahs := ℕ
def yahs := ℕ

-- Conversion ratios given in conditions
def ratio_bah_rah : ℚ := 10 / 16
def ratio_rah_yah : ℚ := 9 / 15

-- Given the conditions
def condition1 (b r : ℚ) : Prop := b / r = ratio_bah_rah
def condition2 (r y : ℚ) : Prop := r / y = ratio_rah_yah

-- Goal: proving the question
theorem bahs_equivalent_to_1500_yahs (b : ℚ) (r : ℚ) (y : ℚ)
  (h1 : condition1 b r) (h2 : condition2 r y) : b * (1500 / y) = 562.5
:=
sorry

end bahs_equivalent_to_1500_yahs_l124_124751


namespace algebra_expression_eq_l124_124297

theorem algebra_expression_eq (x : ℝ) (h : x = Real.sqrt 2 + 1) : x^2 - 2 * x + 2 = 3 := by
  sorry

end algebra_expression_eq_l124_124297


namespace machines_produce_x_units_l124_124967

variable (x : ℕ) (d : ℕ)

-- Define the conditions
def four_machines_produce_in_d_days (x : ℕ) (d : ℕ) : Prop := 
  4 * (x / d) = x / d

def twelve_machines_produce_three_x_in_d_days (x : ℕ) (d : ℕ) : Prop := 
  12 * (x / d) = 3 * (x / d)

-- Given the conditions, prove the number of days for 4 machines to produce x units
theorem machines_produce_x_units (x : ℕ) (d : ℕ) 
  (H1 : four_machines_produce_in_d_days x d)
  (H2 : twelve_machines_produce_three_x_in_d_days x d) : 
  x / d = x / d := 
by 
  sorry

end machines_produce_x_units_l124_124967


namespace rectangle_perimeter_l124_124835
-- Refined definitions and setup
variables (AB BC AE BE CF : ℝ)
-- Conditions provided in the problem
def conditions := AB = 2 * BC ∧ AE = 10 ∧ BE = 26 ∧ CF = 5
-- Perimeter calculation based on the conditions
def perimeter (AB BC : ℝ) : ℝ := 2 * (AB + BC)
-- Main theorem stating the conditions and required result
theorem rectangle_perimeter {m n : ℕ} (h: conditions AB BC AE BE CF) :
  m + n = 105 ∧ Int.gcd m n = 1 ∧ perimeter AB BC = m / n := sorry

end rectangle_perimeter_l124_124835


namespace number_of_children_l124_124474

theorem number_of_children (C A : ℕ) (h1 : C = 2 * A) (h2 : C + A = 120) : C = 80 :=
by
  sorry

end number_of_children_l124_124474


namespace tom_purchases_l124_124490

def total_cost_before_discount (price_per_box : ℝ) (num_boxes : ℕ) : ℝ :=
  price_per_box * num_boxes

def discount (total_cost : ℝ) (discount_rate : ℝ) : ℝ :=
  total_cost * discount_rate

def total_cost_after_discount (total_cost : ℝ) (discount_amount : ℝ) : ℝ :=
  total_cost - discount_amount

def remaining_boxes (total_boxes : ℕ) (given_boxes : ℕ) : ℕ :=
  total_boxes - given_boxes

def total_pieces (num_boxes : ℕ) (pieces_per_box : ℕ) : ℕ :=
  num_boxes * pieces_per_box

theorem tom_purchases
  (price_per_box : ℝ) (num_boxes : ℕ) (discount_rate : ℝ) (given_boxes : ℕ) (pieces_per_box : ℕ) :
  (price_per_box = 4) →
  (num_boxes = 12) →
  (discount_rate = 0.15) →
  (given_boxes = 7) →
  (pieces_per_box = 6) →
  total_cost_after_discount (total_cost_before_discount price_per_box num_boxes) 
                             (discount (total_cost_before_discount price_per_box num_boxes) discount_rate)
  = 40.80 ∧
  total_pieces (remaining_boxes num_boxes given_boxes) pieces_per_box
  = 30 :=
by
  intros
  sorry

end tom_purchases_l124_124490


namespace fin_solutions_l124_124182

theorem fin_solutions (u : ℕ) (hu : u > 0) :
  ∃ N : ℕ, ∀ n a b : ℕ, n > N → ¬ (n! = u^a - u^b) :=
sorry

end fin_solutions_l124_124182


namespace other_number_of_given_conditions_l124_124806

theorem other_number_of_given_conditions 
  (a b : ℕ) 
  (h_lcm : Nat.lcm a b = 4620) 
  (h_gcd : Nat.gcd a b = 21) 
  (h_a : a = 210) : 
  b = 462 := 
sorry

end other_number_of_given_conditions_l124_124806


namespace total_amount_returned_l124_124195

noncomputable def continuous_compounding_interest : ℝ :=
  let P : ℝ := 325 / (Real.exp 0.12 - 1)
  let A1 : ℝ := P * Real.exp 0.04
  let A2 : ℝ := A1 * Real.exp 0.05
  let A3 : ℝ := A2 * Real.exp 0.03
  let total_interest : ℝ := 325
  let total_amount : ℝ := P + total_interest
  total_amount

theorem total_amount_returned :
  continuous_compounding_interest = 2874.02 :=
by
  sorry

end total_amount_returned_l124_124195


namespace tony_gas_expense_in_4_weeks_l124_124643

theorem tony_gas_expense_in_4_weeks :
  let miles_per_gallon := 25
  let miles_per_round_trip_per_day := 50
  let travel_days_per_week := 5
  let tank_capacity_in_gallons := 10
  let cost_per_gallon := 2
  let weeks := 4
  let total_miles_per_week := miles_per_round_trip_per_day * travel_days_per_week
  let total_miles := total_miles_per_week * weeks
  let miles_per_tank := miles_per_gallon * tank_capacity_in_gallons
  let fill_ups_needed := total_miles / miles_per_tank
  let total_gallons_needed := fill_ups_needed * tank_capacity_in_gallons
  let total_cost := total_gallons_needed * cost_per_gallon
  total_cost = 80 :=
by
  sorry

end tony_gas_expense_in_4_weeks_l124_124643


namespace base7_addition_l124_124520

theorem base7_addition : (26:ℕ) + (245:ℕ) = 304 :=
  sorry

end base7_addition_l124_124520


namespace complementary_event_l124_124029

-- Definitions based on the conditions
def EventA (products : List Bool) : Prop := 
  (products.filter (λ x => x = true)).length ≥ 2

def complementEventA (products : List Bool) : Prop := 
  (products.filter (λ x => x = true)).length ≤ 1

-- Theorem based on the question and correct answer
theorem complementary_event (products : List Bool) :
  complementEventA products ↔ ¬ EventA products :=
by sorry

end complementary_event_l124_124029


namespace lcm_fractions_l124_124221

theorem lcm_fractions (x : ℕ) (hx : x > 0) :
  lcm (1 / (2 * x)) (lcm (1 / (4 * x)) (lcm (1 / (6 * x)) (1 / (12 * x)))) = 1 / (12 * x) :=
sorry

end lcm_fractions_l124_124221


namespace symmetric_about_y_axis_l124_124732

theorem symmetric_about_y_axis (m n : ℝ) (A B : ℝ × ℝ)
  (hA : A = (-3, 2 * m - 1))
  (hB : B = (n + 1, 4))
  (symmetry : A.1 = -B.1)
  : m = 2.5 ∧ n = 2 :=
by
  sorry

end symmetric_about_y_axis_l124_124732


namespace parallelogram_area_l124_124795

theorem parallelogram_area (base height : ℕ) (h_base : base = 5) (h_height : height = 3) :
  base * height = 15 :=
by
  -- Here would be the proof, but it is omitted per instructions
  sorry

end parallelogram_area_l124_124795


namespace square_root_of_4_is_pm2_l124_124945

theorem square_root_of_4_is_pm2 : ∃ (x : ℤ), x * x = 4 ∧ (x = 2 ∨ x = -2) := by
  sorry

end square_root_of_4_is_pm2_l124_124945


namespace range_of_x_l124_124758

theorem range_of_x (x : ℝ) (h : 4 * x - 12 ≥ 0) : x ≥ 3 := 
sorry

end range_of_x_l124_124758


namespace point_on_graph_l124_124823

variable (x y : ℝ)

-- Define the condition for a point to be on the graph of the function y = 6/x
def is_on_graph (x y : ℝ) : Prop :=
  x * y = 6

-- State the theorem to be proved
theorem point_on_graph : is_on_graph (-2) (-3) :=
  by
  sorry

end point_on_graph_l124_124823


namespace julia_change_l124_124585

-- Definitions based on the problem conditions
def price_of_snickers : ℝ := 1.5
def price_of_mms : ℝ := 2 * price_of_snickers
def total_cost_of_snickers (num_snickers : ℕ) : ℝ := num_snickers * price_of_snickers
def total_cost_of_mms (num_mms : ℕ) : ℝ := num_mms * price_of_mms
def total_purchase (num_snickers num_mms : ℕ) : ℝ := total_cost_of_snickers num_snickers + total_cost_of_mms num_mms
def amount_given : ℝ := 2 * 10

-- Prove the change is $8
theorem julia_change : total_purchase 2 3 = 12 ∧ (amount_given - total_purchase 2 3) = 8 :=
by
  sorry

end julia_change_l124_124585


namespace exists_disjoint_subsets_for_prime_products_l124_124545

theorem exists_disjoint_subsets_for_prime_products :
  ∃ (A : Fin 100 → Set ℕ), (∀ i j, i ≠ j → Disjoint (A i) (A j)) ∧
    (∀ S : Set ℕ, Infinite S → (∃ m : ℕ, ∃ (a : Fin 100 → ℕ),
      (∀ i, a i ∈ A i) ∧ (∀ i, ∃ p : Fin m → ℕ, (∀ k, p k ∈ S) ∧ a i = (List.prod (List.ofFn p))))) :=
sorry

end exists_disjoint_subsets_for_prime_products_l124_124545


namespace bobby_candy_total_l124_124389

-- Definitions for the conditions
def initial_candy : Nat := 20
def first_candy_eaten : Nat := 34
def second_candy_eaten : Nat := 18

-- Theorem to prove the total pieces of candy Bobby ate
theorem bobby_candy_total : first_candy_eaten + second_candy_eaten = 52 := by
  sorry

end bobby_candy_total_l124_124389


namespace max_ab_min_3x_4y_max_f_l124_124099

-- Proof Problem 1
theorem max_ab (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 4 * a + b = 1) : ab <= 1/16 :=
  sorry

-- Proof Problem 2
theorem min_3x_4y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 3 * y = 5 * x * y) : 3 * x + 4 * y >= 5 :=
  sorry

-- Proof Problem 3
theorem max_f (x : ℝ) (h1 : x < 5/4) : 4 * x - 2 + 1 / (4 * x - 5) <= 1 :=
  sorry

end max_ab_min_3x_4y_max_f_l124_124099


namespace probability_at_least_partly_green_no_red_l124_124788
  
theorem probability_at_least_partly_green_no_red :
  let colors := {red, green, blue, yellow, purple},
      excludeRed := colors.erase red,
      numSingleColor := excludeRed.card,
      numDoubleColors := excludeRed.subsets.card - numSingleColor + 1, -- numDoubleColors = (card choose 2) + (card choose 1)
      totalNoRed := numSingleColor + numDoubleColors,
      greenOnly := excludeRed.erase green,
      partlyGreen := greenOnly.card in
  totalNoRed = 10 → partlyGreen = 4 → (partlyGreen / totalNoRed : Real) = 2 / 5 :=
by
  intros colors excludeRed numSingleColor numDoubleColors totalNoRed greenOnly partlyGreen
  intros h_totalNoRed h_partlyGreen
  sorry

end probability_at_least_partly_green_no_red_l124_124788


namespace find_second_number_l124_124160

variable (n : ℕ)

theorem find_second_number (h : 8000 * n = 480 * 10^5) : n = 6000 :=
by
  sorry

end find_second_number_l124_124160


namespace total_shots_cost_l124_124262

def numDogs : ℕ := 3
def puppiesPerDog : ℕ := 4
def shotsPerPuppy : ℕ := 2
def costPerShot : ℕ := 5

theorem total_shots_cost : (numDogs * puppiesPerDog * shotsPerPuppy * costPerShot) = 120 := by
  sorry

end total_shots_cost_l124_124262


namespace probability_units_digit_4_l124_124124

open Finset

theorem probability_units_digit_4 :
  let S := (range 100).map (λ n, n + 1)
  let count := (card (filter (λ (a : ℕ), (2^a % 10 + 5^5 % 10) % 10 = 4) S))
  --> The probability calculation
  (count : ℚ) / (100 * 100) = 1 / 4 :=
by
  let S := (range 100).map (λ n, n + 1)
  let count := (card (filter (λ (a : ℕ), (2^a % 10 + 5 % 10) % 10 = 4) S))
  have : count = 25,
  {
    -- Explanation here why the count is 25
    sorry
  }
  have total_outcomes : 100 * 100 = 10000,
  {
    sorry
  }
  have probability : (count : ℚ) / 10000 = 1 / 4,
  {
    rw [this, total_outcomes],
    norm_num
  }
  exact probability

end probability_units_digit_4_l124_124124


namespace average_speed_uphill_l124_124465

theorem average_speed_uphill (d : ℝ) (v : ℝ) :
  (2 * d) / ((d / v) + (d / 100)) = 9.523809523809524 → v = 5 :=
by
  intro h1
  sorry

end average_speed_uphill_l124_124465


namespace sqrt_sum_simplify_l124_124050

theorem sqrt_sum_simplify :
  Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 := 
by
  sorry

end sqrt_sum_simplify_l124_124050


namespace range_of_x_l124_124185

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := |x - 4| + |x - a|

theorem range_of_x (a : ℝ) (h1 : a > 1) (h2 : ∀ x : ℝ, f x a ≥ |a - 4|) (h3 : |a - 4| = 3) :
  { x : ℝ | f x a ≤ 5 } = { x : ℝ | 3 ≤ x ∧ x ≤ 8 } := 
sorry

end range_of_x_l124_124185


namespace value_of_3W5_l124_124162

def W (a b : ℕ) : ℕ := b + 7 * a - a ^ 2

theorem value_of_3W5 : W 3 5 = 17 := by 
  sorry

end value_of_3W5_l124_124162


namespace simplify_sqrt_sum_l124_124034

theorem simplify_sqrt_sum : (Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2) :=
by
  sorry

end simplify_sqrt_sum_l124_124034


namespace min_value_four_over_a_plus_nine_over_b_l124_124570

theorem min_value_four_over_a_plus_nine_over_b :
  ∀ (a b : ℝ), a > 0 → b > 0 → a + b = 1 → (∀ x y, x > 0 → y > 0 → x + y ≥ 2 * Real.sqrt (x * y)) →
  (∃ (min_val : ℝ), min_val = (4 / a + 9 / b) ∧ min_val = 25) :=
by
  intros a b ha hb hab am_gm
  sorry

end min_value_four_over_a_plus_nine_over_b_l124_124570


namespace box_surface_area_l124_124677

theorem box_surface_area
  (a b c : ℕ)
  (h1 : a < 10)
  (h2 : b < 10)
  (h3 : c < 10)
  (h4 : a * b * c = 280) : 2 * (a * b + b * c + c * a) = 262 := 
sorry

end box_surface_area_l124_124677


namespace monotonic_decreasing_range_of_a_l124_124903

-- Define the given function
def f (a x : ℝ) := a * x^2 - 3 * x + 4

-- State the proof problem
theorem monotonic_decreasing_range_of_a (a : ℝ) : (∀ x : ℝ, x < 6 → deriv (f a) x ≤ 0) ↔ 0 ≤ a ∧ a ≤ 1/4 :=
sorry

end monotonic_decreasing_range_of_a_l124_124903


namespace age_problem_l124_124913

theorem age_problem (a b c d : ℕ) 
  (h1 : a = b + 2)
  (h2 : b = 2 * c)
  (h3 : b = 3 * d)
  (h4 : a + b + c + d = 87) : 
  b = 30 :=
by sorry

end age_problem_l124_124913


namespace sum_binomial_coeff_l124_124923

open BigOperators

theorem sum_binomial_coeff (m n : ℕ) (h1 : 0 < m) (h2 : 0 < n) (h3 : n ≥ m) :
  ∑ k in Finset.range (m + 1), (-1 : ℤ)^k * (n.choose k) = (-1 : ℤ)^m * (n - 1).choose m :=
sorry

end sum_binomial_coeff_l124_124923


namespace candy_total_l124_124395

theorem candy_total (chocolate_boxes caramel_boxes mint_boxes berry_boxes : ℕ)
  (chocolate_pieces caramel_pieces mint_pieces berry_pieces : ℕ)
  (h_chocolate : chocolate_boxes = 7)
  (h_caramel : caramel_boxes = 3)
  (h_mint : mint_boxes = 5)
  (h_berry : berry_boxes = 4)
  (p_chocolate : chocolate_pieces = 8)
  (p_caramel : caramel_pieces = 8)
  (p_mint : mint_pieces = 10)
  (p_berry : berry_pieces = 12) :
  (chocolate_boxes * chocolate_pieces + caramel_boxes * caramel_pieces + mint_boxes * mint_pieces + berry_boxes * berry_pieces) = 178 := by
  sorry

end candy_total_l124_124395


namespace gcd_of_powers_of_two_l124_124818

def m : ℕ := 2^2100 - 1
def n : ℕ := 2^2000 - 1

theorem gcd_of_powers_of_two :
  Nat.gcd m n = 2^100 - 1 := sorry

end gcd_of_powers_of_two_l124_124818


namespace number_of_diagonals_dodecagon_sum_of_interior_angles_dodecagon_l124_124234

-- Definitions for the problem
def n : Nat := 12

-- Statement 1: Number of diagonals in a dodecagon
theorem number_of_diagonals_dodecagon (n : Nat) (h : n = 12) : (n * (n - 3)) / 2 = 54 := by
  sorry

-- Statement 2: Sum of interior angles in a dodecagon
theorem sum_of_interior_angles_dodecagon (n : Nat) (h : n = 12) : 180 * (n - 2) = 1800 := by
  sorry

end number_of_diagonals_dodecagon_sum_of_interior_angles_dodecagon_l124_124234


namespace sum_of_areas_of_triangles_l124_124386

theorem sum_of_areas_of_triangles 
  (AB BG GE DE : ℕ) 
  (A₁ A₂ : ℕ)
  (H1 : AB = 2) 
  (H2 : BG = 3) 
  (H3 : GE = 4) 
  (H4 : DE = 5) 
  (H5 : 3 * A₁ + 4 * A₂ = 48)
  (H6 : 9 * A₁ + 5 * A₂ = 102) : 
  1 * AB * A₁ / 2 + 1 * DE * A₂ / 2 = 23 :=
by
  sorry

end sum_of_areas_of_triangles_l124_124386


namespace find_p_l124_124421

open Real

variable (A : ℝ × ℝ)
variable (p : ℝ) (hp : p > 0)

-- Conditions
def on_parabola (A : ℝ × ℝ) (p : ℝ) : Prop := A.snd^2 = 2 * p * A.fst
def dist_focus (A : ℝ × ℝ) (p : ℝ) : Prop := sqrt ((A.fst - p / 2)^2 + A.snd^2) = 12
def dist_y_axis (A : ℝ × ℝ) : Prop := abs (A.fst) = 9

-- Theorem to prove
theorem find_p (h1 : on_parabola A p) (h2 : dist_focus A p) (h3 : dist_y_axis A) : p = 6 :=
sorry

end find_p_l124_124421


namespace value_of_5a_l124_124572

variable (a : ℕ)

theorem value_of_5a (h : 5 * (a - 3) = 25) : 5 * a = 40 :=
sorry

end value_of_5a_l124_124572


namespace swimming_pool_width_l124_124852

theorem swimming_pool_width 
  (V : ℝ) (L : ℝ) (B1 : ℝ) (B2 : ℝ) (h : ℝ)
  (h_volume : V = (h / 2) * (B1 + B2) * L) 
  (h_V : V = 270) 
  (h_L : L = 12) 
  (h_B1 : B1 = 1) 
  (h_B2 : B2 = 4) : 
  h = 9 :=
  sorry

end swimming_pool_width_l124_124852


namespace range_of_x_l124_124886

variable {p : ℝ} {x : ℝ}

theorem range_of_x (h : 0 ≤ p ∧ p ≤ 4) : x^2 + p * x > 4 * x + p - 3 ↔ (x ≤ -1 ∨ x ≥ 3) :=
sorry

end range_of_x_l124_124886


namespace stuffed_animal_ratio_l124_124022

theorem stuffed_animal_ratio
  (K : ℕ)
  (h1 : 34 + K + (K + 5) = 175) :
  K / 34 = 2 :=
by sorry

end stuffed_animal_ratio_l124_124022


namespace additional_men_required_l124_124697

variables (W_r : ℚ) (W : ℚ) (D : ℚ) (M : ℚ) (E : ℚ)

-- Given variables
def initial_work_rate := (2.5 : ℚ) / (50 * 100)
def remaining_work_length := (12.5 : ℚ)
def remaining_days := (200 : ℚ)
def initial_men := (50 : ℚ)
def additional_men_needed := (75 : ℚ)

-- Calculating the additional men required
theorem additional_men_required
  (calc_wr : W_r = initial_work_rate)
  (calc_wr_remain : W = remaining_work_length)
  (calc_days_remain : D = remaining_days)
  (calc_initial_men : M = initial_men)
  (calc_additional_men : M + E = (125 : ℚ)) :
  E = additional_men_needed :=
sorry

end additional_men_required_l124_124697


namespace average_number_of_glasses_per_box_l124_124387

-- Definitions and conditions
variables (S L : ℕ) -- S is the number of smaller boxes, L is the number of larger boxes

-- Condition 1: One box contains 12 glasses, and the other contains 16 glasses.
-- (This is implicitly understood in the equation for total glasses)

-- Condition 3: There are 16 more larger boxes than smaller smaller boxes
def condition_3 := L = S + 16

-- Condition 4: The total number of glasses is 480.
def condition_4 := 12 * S + 16 * L = 480

-- Proving the average number of glasses per box is 15
theorem average_number_of_glasses_per_box (h1 : condition_3 S L) (h2 : condition_4 S L) :
  (480 : ℝ) / (S + L) = 15 :=
by 
  -- Assuming S and L are natural numbers 
  sorry

end average_number_of_glasses_per_box_l124_124387


namespace circumcenter_rational_l124_124624

theorem circumcenter_rational {a1 b1 a2 b2 a3 b3 : ℚ} 
  (h1 : a1 ≠ a2 ∨ b1 ≠ b2) 
  (h2 : a1 ≠ a3 ∨ b1 ≠ b3) 
  (h3 : a2 ≠ a3 ∨ b2 ≠ b3) :
  ∃ (x y : ℚ), 
    (x - a1)^2 + (y - b1)^2 = (x - a2)^2 + (y - b2)^2 ∧
    (x - a1)^2 + (y - b1)^2 = (x - a3)^2 + (y - b3)^2 :=
sorry

end circumcenter_rational_l124_124624


namespace remaining_minutes_proof_l124_124521

def total_series_minutes : ℕ := 360

def first_session_end : ℕ := 17 * 60 + 44  -- in minutes
def first_session_start : ℕ := 15 * 60 + 20  -- in minutes
def second_session_end : ℕ := 20 * 60 + 40  -- in minutes
def second_session_start : ℕ := 19 * 60 + 15  -- in minutes
def third_session_end : ℕ := 22 * 60 + 30  -- in minutes
def third_session_start : ℕ := 21 * 60 + 35  -- in minutes

def first_session_duration : ℕ := first_session_end - first_session_start
def second_session_duration : ℕ := second_session_end - second_session_start
def third_session_duration : ℕ := third_session_end - third_session_start

def total_watched : ℕ := first_session_duration + second_session_duration + third_session_duration

def remaining_time : ℕ := total_series_minutes - total_watched

theorem remaining_minutes_proof : remaining_time = 76 := 
by 
  sorry  -- Proof goes here

end remaining_minutes_proof_l124_124521


namespace distance_corresponds_to_additional_charge_l124_124015

-- Define the initial fee
def initial_fee : ℝ := 2.5

-- Define the charge per part of a mile
def charge_per_part_of_mile : ℝ := 0.35

-- Define the total charge for a 3.6 miles trip
def total_charge : ℝ := 5.65

-- Define the correct distance corresponding to the additional charge
def correct_distance : ℝ := 0.9

-- The theorem to prove
theorem distance_corresponds_to_additional_charge :
  (total_charge - initial_fee) / charge_per_part_of_mile * (0.1) = correct_distance :=
by
  sorry

end distance_corresponds_to_additional_charge_l124_124015


namespace solution_set_of_inequality_system_l124_124944

theorem solution_set_of_inequality_system (x : ℝ) : 
  (x + 5 < 4) ∧ (3 * x + 1 ≥ 2 * (2 * x - 1)) ↔ (x < -1) :=
  by
  sorry

end solution_set_of_inequality_system_l124_124944


namespace divisor_of_99_l124_124238

def reverse_digits (n : ℕ) : ℕ :=
  -- We assume a placeholder definition for reversing the digits of a number
  sorry

theorem divisor_of_99 (k : ℕ) (h : ∀ n : ℕ, k ∣ n → k ∣ reverse_digits n) : k ∣ 99 :=
  sorry

end divisor_of_99_l124_124238


namespace add_fractions_l124_124962

theorem add_fractions : (1 / 6 : ℚ) + (5 / 12) = 7 / 12 := 
by
  sorry

end add_fractions_l124_124962


namespace arithmetic_seq_sum_l124_124560

theorem arithmetic_seq_sum (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_123 : a 0 + a 1 + a 2 = -3)
  (h_456 : a 3 + a 4 + a 5 = 6) :
  ∀ n, S n = n * (-2) + n * (n - 1) / 2 :=
by
  sorry

end arithmetic_seq_sum_l124_124560


namespace min_sum_xy_l124_124137

theorem min_sum_xy (x y : ℕ) (hx : x ≠ y) (hcond : ↑(1 / x) + ↑(1 / y) = 1 / 15) : x + y = 64 :=
sorry

end min_sum_xy_l124_124137


namespace sum_of_integers_with_product_5_pow_4_l124_124942

theorem sum_of_integers_with_product_5_pow_4 :
  ∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  a * b * c * d = 5^4 ∧
  a + b + c + d = 156 :=
by sorry

end sum_of_integers_with_product_5_pow_4_l124_124942


namespace find_k_l124_124571

theorem find_k (k : ℕ) : (1 / 2) ^ 18 * (1 / 81) ^ k = 1 / 18 ^ 18 → k = 0 := by
  intro h
  sorry

end find_k_l124_124571


namespace box_surface_area_l124_124678

theorem box_surface_area
  (a b c : ℕ)
  (h1 : a < 10)
  (h2 : b < 10)
  (h3 : c < 10)
  (h4 : a * b * c = 280) : 2 * (a * b + b * c + c * a) = 262 := 
sorry

end box_surface_area_l124_124678


namespace find_x_y_l124_124793

theorem find_x_y (x y : ℕ) (h1 : 1 ≤ x) (h2 : 1 ≤ y) (h3 : y ≥ x) (h4 : x + y ≤ 20) 
  (h5 : ¬(∃ s, (x * y = s) → x + y = s ∧ ∃ a b : ℕ, a * b = s ∧ a ≠ x ∧ b ≠ y))
  (h6 : ∃ s_t, (x + y = s_t) → x * y = s_t):
  x = 2 ∧ y = 11 :=
by {
  sorry
}

end find_x_y_l124_124793


namespace james_paid_with_l124_124170

variable (candy_packs : ℕ) (cost_per_pack : ℕ) (change_received : ℕ)

theorem james_paid_with (h1 : candy_packs = 3) (h2 : cost_per_pack = 3) (h3 : change_received = 11) :
  let total_cost := candy_packs * cost_per_pack
  let amount_paid := total_cost + change_received
  amount_paid = 20 :=
by
  sorry

end james_paid_with_l124_124170


namespace probability_rain_once_l124_124486

theorem probability_rain_once (p : ℚ) 
  (h₁ : p = 1 / 2) 
  (h₂ : 1 - p = 1 / 2) 
  (h₃ : (1 - p) ^ 4 = 1 / 16) 
  : 1 - (1 - p) ^ 4 = 15 / 16 :=
by
  sorry

end probability_rain_once_l124_124486


namespace sqrt_defined_value_l124_124908

theorem sqrt_defined_value (x : ℝ) (h : x ≥ 4) : x = 5 → true := 
by 
  intro hx
  sorry

end sqrt_defined_value_l124_124908


namespace roger_steps_to_minutes_l124_124927

theorem roger_steps_to_minutes (h1 : ∃ t: ℕ, t = 30 ∧ ∃ s: ℕ, s = 2000)
                               (h2 : ∃ g: ℕ, g = 10000) :
  ∃ m: ℕ, m = 150 :=
by 
  sorry

end roger_steps_to_minutes_l124_124927


namespace sum_of_first_ten_terms_l124_124949

theorem sum_of_first_ten_terms (a1 d : ℝ) (h1 : 3 * (a1 + d) = 15) 
  (h2 : (a1 + d - 1) ^ 2 = (a1 - 1) * (a1 + 2 * d + 1)) : 
  (10 / 2) * (2 * a1 + (10 - 1) * d) = 120 := 
by 
  sorry

end sum_of_first_ten_terms_l124_124949


namespace transformation_correct_l124_124825

theorem transformation_correct (a b c : ℝ) (h : a / c = b / c) (hc : c ≠ 0) : a = b :=
sorry

end transformation_correct_l124_124825


namespace find_f_l124_124119

theorem find_f (f : ℤ → ℤ) (h : ∀ n : ℤ, n^2 + 4 * (f n) = (f (f n))^2) :
  (∀ x : ℤ, f x = 1 + x) ∨
  (∃ a : ℤ, (∀ x ≤ a, f x = 1 - x) ∧ (∀ x > a, f x = 1 + x)) ∨
  (f 0 = 0 ∧ (∀ x < 0, f x = 1 - x) ∧ (∀ x > 0, f x = 1 + x)) :=
sorry

end find_f_l124_124119


namespace repeating_decimal_sum_l124_124112

theorem repeating_decimal_sum :
  let a := (2 : ℚ) / 3
  let b := (2 : ℚ) / 9
  let c := (4 : ℚ) / 9
  a + b - c = (4 : ℚ) / 9 :=
by
  sorry

end repeating_decimal_sum_l124_124112


namespace no_solutions_system_of_inequalities_l124_124511

open Set

theorem no_solutions_system_of_inequalities :
  ∀ (x y : ℝ),
    ¬(11 * x^2 - 10 * x * y + 3 * y^2 ≤ 3 ∧ 5 * x + y ≤ -10) :=
by
  intro x y
  rw not_and
  intro h1 h2
  let y' := -10 - 5 * x
  have h3 : y = y' := eq_of_le_of_le h2 (le_of_eq rfl)
  sorry

end no_solutions_system_of_inequalities_l124_124511


namespace high_school_sampling_problem_l124_124842

theorem high_school_sampling_problem :
  let first_year_classes := 20
  let first_year_students_per_class := 50
  let first_year_total_students := first_year_classes * first_year_students_per_class
  let second_year_classes := 24
  let second_year_students_per_class := 45
  let second_year_total_students := second_year_classes * second_year_students_per_class
  let total_students := first_year_total_students + second_year_total_students
  let survey_students := 208
  let first_year_sample := (first_year_total_students * survey_students) / total_students
  let second_year_sample := (second_year_total_students * survey_students) / total_students
  let A_selected_probability := first_year_sample / first_year_total_students
  let B_selected_probability := second_year_sample / second_year_total_students
  (survey_students = 208) →
  (first_year_sample = 100) →
  (second_year_sample = 108) →
  (A_selected_probability = 1 / 10) →
  (B_selected_probability = 1 / 10) →
  (A_selected_probability = B_selected_probability) →
  (student_A_in_first_year : true) →
  (student_B_in_second_year : true) →
  true :=
  by sorry

end high_school_sampling_problem_l124_124842


namespace cos_evaluation_l124_124765

open Real

noncomputable def a (n : ℕ) : ℝ := sorry  -- since it's an arithmetic sequence

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m k : ℕ, a n + a k = 2 * a ((n + k) / 2)

def satisfies_condition (a : ℕ → ℝ) : Prop :=
  a 3 + a 6 + a 9 = 3 * a 6 ∧ a 6 = π / 4

theorem cos_evaluation :
  is_arithmetic_sequence a →
  satisfies_condition a →
  cos (a 2 + a 10 + π / 4) = - (sqrt 2 / 2) :=
by
  intros
  sorry

end cos_evaluation_l124_124765


namespace solution_set_of_inequality_l124_124071

theorem solution_set_of_inequality (x : ℝ) : 
  (3*x^2 - 4*x + 7 > 0) → (1 - 2*x) / (3*x^2 - 4*x + 7) ≥ 0 ↔ x ≤ 1 / 2 :=
by
  intros
  sorry

end solution_set_of_inequality_l124_124071


namespace water_remaining_l124_124314

variable (initial_amount : ℝ) (leaked_amount : ℝ)

theorem water_remaining (h1 : initial_amount = 0.75)
                       (h2 : leaked_amount = 0.25) :
  initial_amount - leaked_amount = 0.50 :=
by
  sorry

end water_remaining_l124_124314


namespace song_distribution_l124_124522

-- Let us define the necessary conditions and the result as a Lean statement.

theorem song_distribution :
    ∃ (AB BC CA A B C N : Finset ℕ),
    -- Six different songs.
    (AB ∪ BC ∪ CA ∪ A ∪ B ∪ C ∪ N) = {1, 2, 3, 4, 5, 6} ∧
    -- No song is liked by all three.
    (∀ song, ¬(song ∈ AB ∩ BC ∩ CA)) ∧
    -- Each girl dislikes at least one song.
    (N ≠ ∅) ∧
    -- For each pair of girls, at least one song liked by those two but disliked by the third.
    (AB ≠ ∅ ∧ BC ≠ ∅ ∧ CA ≠ ∅) ∧
    -- The total number of ways this can be done is 735.
    True := sorry

end song_distribution_l124_124522


namespace value_of_x_add_y_not_integer_l124_124175

theorem value_of_x_add_y_not_integer (x y: ℝ) (h1: y = 3 * ⌊x⌋ + 4) (h2: y = 2 * ⌊x - 3⌋ + 7) (h3: ¬ ∃ n: ℤ, x = n): -8 < x + y ∧ x + y < -7 := 
sorry

end value_of_x_add_y_not_integer_l124_124175


namespace all_palindromes_divisible_by_11_probability_palindrome_divisible_by_11_l124_124686

theorem all_palindromes_divisible_by_11 : 
  (∀ a b : ℕ, 1 <= a ∧ a <= 9 ∧ 0 <= b ∧ b <= 9 →
    (1001 * a + 110 * b) % 11 = 0 ) := sorry

theorem probability_palindrome_divisible_by_11 : 
  (∀ (palindromes : ℕ → Prop), 
  (∀ n, palindromes n ↔ ∃ (a b : ℕ), 
  1 <= a ∧ a <= 9 ∧ 0 <= b ∧ b <= 9 ∧ 
  n = 1001 * a + 110 * b) → 
  (∀ n, palindromes n → n % 11 = 0) →
  ∃ p : ℝ, p = 1) := sorry

end all_palindromes_divisible_by_11_probability_palindrome_divisible_by_11_l124_124686


namespace bridge_length_l124_124344

theorem bridge_length 
  (train_length : ℝ) 
  (train_speed_kmph : ℝ) 
  (cross_time_seconds : ℝ)
  (train_length_eq : train_length = 150)
  (train_speed_kmph_eq : train_speed_kmph = 45)
  (cross_time_seconds_eq : cross_time_seconds = 30) : 
  ∃ (bridge_length : ℝ), bridge_length = 225 := 
  by
  sorry

end bridge_length_l124_124344


namespace rectangle_dimensions_l124_124327

-- Define the known shapes and their dimensions
def square (s : ℝ) : ℝ := s^2
def rectangle1 : ℝ := 10 * 24
def rectangle2 (a b : ℝ) : ℝ := a * b

-- The total area must match the area of a square of side length 24 cm
def total_area (s a b : ℝ) : ℝ := (2 * square s) + rectangle1 + rectangle2 a b

-- The problem statement
theorem rectangle_dimensions
  (s a b : ℝ)
  (h0 : a ∈ [2, 19, 34, 34, 14, 14, 24])
  (h1 : b ∈ [24, 17.68, 10, 44, 24, 17, 38])
  : (total_area s a b = 24^2) :=
by
  sorry

end rectangle_dimensions_l124_124327


namespace unique_prime_pair_l124_124085

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem unique_prime_pair :
  ∀ p : ℕ, is_prime p ∧ is_prime (p + 1) → p = 2 := by
  sorry

end unique_prime_pair_l124_124085


namespace num_subsets_containing_6_l124_124740

open Finset

-- Define the set S
def S : Finset ℕ := {1, 2, 3, 4, 5, 6}

-- Define the subset containing number 6
def subsets_with_6 (s : Finset ℕ) : Finset (Finset ℕ) :=
  s.powerset.filter (λ x => 6 ∈ x)

-- Theorem: The number of subsets of {1, 2, 3, 4, 5, 6} containing the number 6 is 32
theorem num_subsets_containing_6 : (subsets_with_6 S).card = 32 := by
  sorry

end num_subsets_containing_6_l124_124740


namespace cube_root_of_64_l124_124868

theorem cube_root_of_64 : ∃ x : ℝ, x^3 = 64 ∧ x = 4 :=
by
  sorry

end cube_root_of_64_l124_124868


namespace compute_product_l124_124638

variable (x1 y1 x2 y2 x3 y3 : ℝ)

def condition1 (x y : ℝ) : Prop := x^3 - 3 * x * y^2 = 2010
def condition2 (x y : ℝ) : Prop := y^3 - 3 * x^2 * y = 2000

theorem compute_product (h1 : condition1 x1 y1) (h2 : condition2 x1 y1)
    (h3 : condition1 x2 y2) (h4 : condition2 x2 y2)
    (h5 : condition1 x3 y3) (h6 : condition2 x3 y3) :
    (1 - x1 / y1) * (1 - x2 / y2) * (1 - x3 / y3) = 1 / 100 := 
    sorry

end compute_product_l124_124638


namespace simplify_sqrt_72_plus_sqrt_32_l124_124038

theorem simplify_sqrt_72_plus_sqrt_32 : 
  sqrt 72 + sqrt 32 = 10 * sqrt 2 :=
by
  -- Define the intermediate results based on the conditions
  let sqrt72 := sqrt (2^3 * 3^2)
  let sqrt32 := sqrt (2^5)
  -- Specific simplifications from steps are not used directly, but they guide the statement
  show sqrt72 + sqrt32 = 10 * sqrt 2
  sorry

end simplify_sqrt_72_plus_sqrt_32_l124_124038


namespace regular_bike_wheels_eq_two_l124_124988

-- Conditions
def regular_bikes : ℕ := 7
def childrens_bikes : ℕ := 11
def wheels_per_childrens_bike : ℕ := 4
def total_wheels_seen : ℕ := 58

-- Define the problem
theorem regular_bike_wheels_eq_two 
  (w : ℕ)
  (h1 : total_wheels_seen = regular_bikes * w + childrens_bikes * wheels_per_childrens_bike) :
  w = 2 :=
by
  -- Proof steps would go here
  sorry

end regular_bike_wheels_eq_two_l124_124988


namespace suitable_sampling_method_l124_124684

noncomputable def is_stratified_sampling_suitable (mountainous hilly flat low_lying sample_size : ℕ) (yield_dependent_on_land_type : Bool) : Bool :=
  if yield_dependent_on_land_type && mountainous + hilly + flat + low_lying > 0 then true else false

theorem suitable_sampling_method :
  is_stratified_sampling_suitable 8000 12000 24000 4000 480 true = true :=
by
  sorry

end suitable_sampling_method_l124_124684


namespace set_intersection_complement_l124_124729

open Set

variable (A B U : Set ℕ)

theorem set_intersection_complement (A B : Set ℕ) (U : Set ℕ) (hU : U = {1, 2, 3, 4})
  (h1 : compl (A ∪ B) = {4}) (h2 : B = {1, 2}) :
  A ∩ compl B = {3} :=
by
  sorry

end set_intersection_complement_l124_124729


namespace find_remainder_l124_124329

def dividend : ℝ := 17698
def divisor : ℝ := 198.69662921348313
def quotient : ℝ := 89
def remainder : ℝ := 14

theorem find_remainder :
  dividend = (divisor * quotient) + remainder :=
by 
  -- Placeholder proof
  sorry

end find_remainder_l124_124329


namespace rob_total_cards_l124_124335

variables (r r_d j_d : ℕ)

-- Definitions of conditions
def condition1 : Prop := r_d = r / 3
def condition2 : Prop := j_d = 5 * r_d
def condition3 : Prop := j_d = 40

-- Problem Statement
theorem rob_total_cards (h1 : condition1 r r_d)
                        (h2 : condition2 r_d j_d)
                        (h3 : condition3 j_d) :
  r = 24 :=
by
  sorry

end rob_total_cards_l124_124335


namespace num_subsets_containing_6_l124_124746

theorem num_subsets_containing_6 : 
  (∃ (subset : set (fin 6)), 6 ∈ subset ∧ fintype.card {s : set (fin 6) | s ∈ subset}) = 32 :=
sorry

end num_subsets_containing_6_l124_124746


namespace triangle_inequality_l124_124443

theorem triangle_inequality 
  (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  2 * (a + b + c) * (a * b + b * c + c * a) ≤ (a + b + c) * (a^2 + b^2 + c^2) + 9 * a * b * c :=
by
  sorry

end triangle_inequality_l124_124443


namespace chandler_saves_for_laptop_l124_124392

theorem chandler_saves_for_laptop :
  ∃ x : ℕ, 140 + 20 * x = 800 ↔ x = 33 :=
by
  use 33
  sorry

end chandler_saves_for_laptop_l124_124392


namespace expression_simplification_l124_124656

theorem expression_simplification (x y : ℝ) :
  20 * (x + y) - 19 * (x + y) = x + y :=
by
  sorry

end expression_simplification_l124_124656


namespace rational_coordinates_of_circumcenter_l124_124618

open Classical

noncomputable theory

theorem rational_coordinates_of_circumcenter
  {a1 b1 a2 b2 a3 b3 : ℚ}
  (h1 : ∃ (x y : ℚ), (x - a1)^2 + (y - b1)^2 = (x - a2)^2 + (y - b2)^2 ∧
                      (x - a1)^2 + (y - b1)^2 = (x - a3)^2 + (y - b3)^2) :
  ∃ (x y : ℚ),
    (x - a1)^2 + (y - b1)^2 = (x - a2)^2 + (y - b2)^2 ∧
    (x - a1)^2 + (y - b1)^2 = (x - a3)^2 + (y - b3)^2 := 
begin
  obtain ⟨x, y, hx⟩ := h1,
  use [x, y],
  exact hx,
end

end rational_coordinates_of_circumcenter_l124_124618


namespace mul_binom_expansion_l124_124870

variable (a : ℝ)

theorem mul_binom_expansion : (a + 1) * (a - 1) = a^2 - 1 :=
by
  sorry

end mul_binom_expansion_l124_124870


namespace inequality_for_positive_real_numbers_l124_124331

theorem inequality_for_positive_real_numbers (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
    x^2 + 2*y^2 + 3*z^2 > x*y + 3*y*z + z*x := 
by 
  sorry

end inequality_for_positive_real_numbers_l124_124331


namespace probability_listens_to_second_class_l124_124850

theorem probability_listens_to_second_class (interval_start interval_end class_start class_end arrive_start arrive_end : ℝ)
  (h1 : class_start = 8 + 5 / 6)
  (h2 : class_end = 9 + 1 / 2)
  (h3 : arrive_start = 9 + 1 / 6)
  (h4 : arrive_end = 10)
  (h5 : interval_start = 9 + 1 / 6)
  (h6 : interval_end = 9 + 1 / 3) :
  (interval_end - interval_start) / (arrive_end - arrive_start) = 1 / 5 := 
  sorry

end probability_listens_to_second_class_l124_124850


namespace three_pow_2010_mod_eight_l124_124503

theorem three_pow_2010_mod_eight : (3^2010) % 8 = 1 :=
  sorry

end three_pow_2010_mod_eight_l124_124503


namespace work_duration_l124_124833

theorem work_duration (X_full_days : ℕ) (Y_full_days : ℕ) (Y_worked_days : ℕ) (R : ℚ) :
  X_full_days = 18 ∧ Y_full_days = 15 ∧ Y_worked_days = 5 ∧ R = (2 / 3) →
  (R / (1 / X_full_days)) = 12 :=
by
  intros h
  sorry

end work_duration_l124_124833


namespace circle_range_of_a_l124_124733

theorem circle_range_of_a (a : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 + 2*x - 4*y + a = 0) → a < 5 := by
  sorry

end circle_range_of_a_l124_124733


namespace evaluate_expression_l124_124183

theorem evaluate_expression :
  let a := Real.sqrt 2 ^ 2 + Real.sqrt 3 + Real.sqrt 5
  let b := - Real.sqrt 2 ^ 2 + Real.sqrt 3 + Real.sqrt 5
  let c := Real.sqrt 2 ^ 2 - Real.sqrt 3 + Real.sqrt 5
  let d := - Real.sqrt 2 ^ 2 - Real.sqrt 3 + Real.sqrt 5
  (1/a + 1/b + 1/c + 1/d)^2 = 5 :=
by
  sorry

end evaluate_expression_l124_124183


namespace factorization_correctness_l124_124363

theorem factorization_correctness :
  ∀ x : ℝ, x^2 - 2*x + 1 = (x - 1)^2 :=
by
  -- Proof omitted
  sorry

end factorization_correctness_l124_124363


namespace sqrt_sum_simplify_l124_124052

theorem sqrt_sum_simplify :
  Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 := 
by
  sorry

end sqrt_sum_simplify_l124_124052


namespace solve_determinant_l124_124385

-- Definitions based on the conditions
def determinant (a b c d : ℤ) : ℤ := a * d - b * c

-- The problem translated to Lean 4:
theorem solve_determinant (x : ℤ) 
  (h : determinant (x + 1) x (2 * x - 6) (2 * (x - 1)) = 10) :
  x = 2 :=
sorry -- Proof is skipped

end solve_determinant_l124_124385


namespace screen_width_l124_124855

theorem screen_width
  (A : ℝ) -- Area of the screen
  (h : ℝ) -- Height of the screen
  (w : ℝ) -- Width of the screen
  (area_eq : A = 21) -- Condition 1: Area is 21 sq ft
  (height_eq : h = 7) -- Condition 2: Height is 7 ft
  (area_formula : A = w * h) -- Condition 3: Area formula
  : w = 3 := -- Conclusion: Width is 3 ft
sorry

end screen_width_l124_124855


namespace red_beads_cost_l124_124924

theorem red_beads_cost (R : ℝ) (H : 4 * R + 4 * 2 = 10 * 1.72) : R = 2.30 :=
by
  sorry

end red_beads_cost_l124_124924


namespace find_fraction_l124_124599

variables (a b c : ℝ)
variables (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
variable (h : a + b + c = 1)

theorem find_fraction :
  (a^3 + b^3 + c^3) / (a * b * c) = (1 + 3 * (a - b)^2) / (a * b * (1 - a - b)) :=
by
  sorry

end find_fraction_l124_124599


namespace find_all_possible_f_l124_124118

-- Noncomputability is needed here since we cannot construct a function 
-- like f deterministically via computation due to the nature of the problem.
noncomputable def functional_equation_solution (f : ℕ → ℕ) := 
  (∀ a b : ℕ, f a + f b ∣ 2 * (a + b - 1)) → 
  (∀ x : ℕ, f x = 1) ∨ (∀ x : ℕ, f x = 2 * x - 1)

-- Statement of the mathematically equivalent proof problem.
theorem find_all_possible_f (f : ℕ → ℕ) : functional_equation_solution f := 
sorry

end find_all_possible_f_l124_124118


namespace kimberly_peanuts_per_visit_l124_124315

theorem kimberly_peanuts_per_visit 
  (trips : ℕ) (total_peanuts : ℕ) 
  (h1 : trips = 3) 
  (h2 : total_peanuts = 21) : 
  total_peanuts / trips = 7 :=
by
  sorry

end kimberly_peanuts_per_visit_l124_124315


namespace inscribed_circle_radius_l124_124586

theorem inscribed_circle_radius (a b c : ℝ) (R : ℝ) (r : ℝ) :
  a = 20 → b = 20 → d = 25 → r = 6 := 
by
  -- conditions of the problem
  sorry

end inscribed_circle_radius_l124_124586


namespace remainder_of_899830_divided_by_16_is_6_l124_124093

theorem remainder_of_899830_divided_by_16_is_6 :
  ∃ k : ℕ, 899830 = 16 * k + 6 :=
by
  sorry

end remainder_of_899830_divided_by_16_is_6_l124_124093


namespace quadratic_inequality_solution_l124_124209

theorem quadratic_inequality_solution (x : ℝ) :
  (x^2 - 2 * x - 3 < 0) ↔ (-1 < x ∧ x < 3) :=
sorry

end quadratic_inequality_solution_l124_124209


namespace middle_number_consecutive_even_l124_124072

theorem middle_number_consecutive_even (a b c : ℤ) 
  (h1 : a = b - 2) 
  (h2 : c = b + 2) 
  (h3 : a + b = 18) 
  (h4 : a + c = 22) 
  (h5 : b + c = 28) : 
  b = 11 :=
by sorry

end middle_number_consecutive_even_l124_124072


namespace number_of_solutions_l124_124887

theorem number_of_solutions : ∃ (s : Finset ℕ), (∀ x ∈ s, 100 ≤ x^2 ∧ x^2 ≤ 200) ∧ s.card = 5 :=
by
  sorry

end number_of_solutions_l124_124887


namespace rainy_days_l124_124603

theorem rainy_days (n R NR : ℕ): (n * R + 3 * NR = 20) ∧ (3 * NR = n * R + 10) ∧ (R + NR = 7) → R = 2 :=
by
  intro h
  obtain ⟨h1, h2, h3⟩ := h
  sorry

end rainy_days_l124_124603


namespace speed_of_train_l124_124247

-- Conditions
def length_of_train : ℝ := 100
def time_to_cross : ℝ := 12

-- Question and answer
theorem speed_of_train : length_of_train / time_to_cross = 8.33 := 
by 
  sorry

end speed_of_train_l124_124247


namespace initial_pen_count_is_30_l124_124225

def pen_count (initial_pens : ℕ) : ℕ :=
  let after_mike := initial_pens + 20
  let after_cindy := 2 * after_mike
  let after_sharon := after_cindy - 10
  after_sharon

theorem initial_pen_count_is_30 : pen_count 30 = 30 :=
by
  sorry

end initial_pen_count_is_30_l124_124225


namespace simplify_sqrt_72_plus_sqrt_32_l124_124037

theorem simplify_sqrt_72_plus_sqrt_32 : 
  sqrt 72 + sqrt 32 = 10 * sqrt 2 :=
by
  -- Define the intermediate results based on the conditions
  let sqrt72 := sqrt (2^3 * 3^2)
  let sqrt32 := sqrt (2^5)
  -- Specific simplifications from steps are not used directly, but they guide the statement
  show sqrt72 + sqrt32 = 10 * sqrt 2
  sorry

end simplify_sqrt_72_plus_sqrt_32_l124_124037


namespace suff_not_necess_cond_perpendicular_l124_124559

theorem suff_not_necess_cond_perpendicular (m : ℝ) :
  (m = 1 → ∀ x y : ℝ, x - y = 0 ∧ x + y = 0) ∧
  (m ≠ 1 → ∃ (x y : ℝ), ¬ (x - y = 0 ∧ x + y = 0)) :=
sorry

end suff_not_necess_cond_perpendicular_l124_124559


namespace money_sum_l124_124692

theorem money_sum (A B C : ℕ) (h1 : A + C = 300) (h2 : B + C = 600) (h3 : C = 200) : A + B + C = 700 :=
by
  sorry

end money_sum_l124_124692


namespace inequality_proof_l124_124889

theorem inequality_proof (x y : ℝ) (h : x^8 + y^8 ≤ 1) : x^12 - y^12 + 2 * x^6 * y^6 ≤ (Real.pi / 2) := 
by 
  sorry

end inequality_proof_l124_124889


namespace find_n_l124_124550

theorem find_n (n : ℤ) (h1 : -90 ≤ n) (h2 : n ≤ 90) (h3 : Real.sin (n * Real.pi / 180) = Real.sin (782 * Real.pi / 180)) :
  n = 62 ∨ n = -62 := 
sorry

end find_n_l124_124550


namespace jack_salt_evaporation_l124_124169

/-- Calculate the volume of salt in milliliters from a given volume of seawater and salt concentration. --/
def volume_of_salt_in_ml (seawater_volume : ℝ) (salt_percentage : ℝ) : ℝ :=
  seawater_volume * salt_percentage * 1000

theorem jack_salt_evaporation :
  volume_of_salt_in_ml 2 0.20 = 400 :=
by 
sory

end jack_salt_evaporation_l124_124169


namespace find_x_value_l124_124276

open Real

theorem find_x_value (a : ℝ) (x : ℝ) (h : a > 0) (h_eq : 10^x = log (10 * a) + log (a⁻¹)) : x = 0 :=
by
  sorry

end find_x_value_l124_124276


namespace root_of_quadratic_expression_l124_124422

theorem root_of_quadratic_expression (n : ℝ) (h : n^2 - 5 * n + 4 = 0) : n^2 - 5 * n = -4 :=
by
  sorry

end root_of_quadratic_expression_l124_124422


namespace circumcenter_rational_l124_124614

theorem circumcenter_rational {a1 b1 a2 b2 a3 b3 : ℚ} :
  ∃ (x y : ℚ), 
    ((x - a1)^2 + (y - b1)^2 = (x - a2)^2 + (y - b2)^2) ∧
    ((x - a1)^2 + (y - b1)^2 = (x - a3)^2 + (y - b3)^2) :=
sorry

end circumcenter_rational_l124_124614


namespace weight_of_3_moles_of_BaF2_is_correct_l124_124653

-- Definitions for the conditions
def atomic_weight_Ba : ℝ := 137.33 -- g/mol
def atomic_weight_F : ℝ := 19.00 -- g/mol

-- Definition of the molecular weight of BaF2
def molecular_weight_BaF2 : ℝ := (1 * atomic_weight_Ba) + (2 * atomic_weight_F)

-- The statement to prove
theorem weight_of_3_moles_of_BaF2_is_correct : (3 * molecular_weight_BaF2) = 525.99 :=
by
  -- Proof omitted
  sorry

end weight_of_3_moles_of_BaF2_is_correct_l124_124653


namespace roots_polynomial_value_l124_124786

theorem roots_polynomial_value (a b c : ℝ) 
  (h1 : a + b + c = 15)
  (h2 : a * b + b * c + c * a = 25)
  (h3 : a * b * c = 12) :
  (2 + a) * (2 + b) * (2 + c) = 130 := 
by
  sorry

end roots_polynomial_value_l124_124786


namespace count_total_legs_l124_124351

theorem count_total_legs :
  let tables4 := 4 * 4
  let sofa := 1 * 4
  let chairs4 := 2 * 4
  let tables3 := 3 * 3
  let table1 := 1 * 1
  let rocking_chair := 1 * 2
  let total_legs := tables4 + sofa + chairs4 + tables3 + table1 + rocking_chair
  total_legs = 40 :=
by
  sorry

end count_total_legs_l124_124351


namespace ages_total_l124_124831

theorem ages_total (a b c : ℕ) (h1 : b = 8) (h2 : a = b + 2) (h3 : b = 2 * c) : a + b + c = 22 := by
  sorry

end ages_total_l124_124831


namespace problem_statement_l124_124824

theorem problem_statement :
  ¬(∀ n : ℤ, n ≥ 0 → n = 0) ∧
  ¬(∀ q : ℚ, q ≠ 0 → q > 0 ∨ q < 0) ∧
  ¬(∀ a b : ℝ, abs a = abs b → a = b) ∧
  (∀ a : ℝ, abs a = abs (-a)) :=
by
  sorry

end problem_statement_l124_124824


namespace necessary_but_not_sufficient_l124_124671

theorem necessary_but_not_sufficient (x : ℝ) :
  (x < 1 ∨ x > 4) → (x^2 - 3 * x + 2 > 0) ∧ ¬((x^2 - 3 * x + 2 > 0) → (x < 1 ∨ x > 4)) :=
by
  sorry

end necessary_but_not_sufficient_l124_124671


namespace lcm_of_numbers_l124_124435

theorem lcm_of_numbers (a b lcm hcf : ℕ) (h_prod : a * b = 45276) (h_hcf : hcf = 22) (h_relation : a * b = hcf * lcm) : lcm = 2058 :=
by sorry

end lcm_of_numbers_l124_124435


namespace distinct_nonzero_digits_sum_l124_124574

theorem distinct_nonzero_digits_sum
  (x y z w : Nat)
  (hx : x ≠ 0)
  (hy : y ≠ 0)
  (hz : z ≠ 0)
  (hw : w ≠ 0)
  (hxy : x ≠ y)
  (hxz : x ≠ z)
  (hxw : x ≠ w)
  (hyz : y ≠ z)
  (hyw : y ≠ w)
  (hzw : z ≠ w)
  (h1 : w + x = 10)
  (h2 : y + w = 9)
  (h3 : z + x = 9) :
  x + y + z + w = 18 :=
sorry

end distinct_nonzero_digits_sum_l124_124574


namespace plane_equation_proof_l124_124240

-- Define the parametric representation of the plane
def plane_parametric (s t : ℝ) : ℝ × ℝ × ℝ :=
  (2 + 2 * s - t, 1 + 2 * s, 4 - s + 3 * t)

-- Define the plane equation form
def plane_equation (x y z : ℝ) (A B C D : ℤ) : Prop :=
  (A : ℝ) * x + (B : ℝ) * y + (C : ℝ) * z + (D : ℝ) = 0

-- Define the normal vector derived from the cross product
def normal_vector : ℝ × ℝ × ℝ := (6, -5, 2)

-- Define the initial point used to calculate D
def initial_point : ℝ × ℝ × ℝ := (2, 1, 4)

-- Proposition to prove the equation of the plane
theorem plane_equation_proof :
  ∃ (A B C D : ℤ), A = 6 ∧ B = -5 ∧ C = 2 ∧ D = -15 ∧
    ∀ x y z : ℝ, plane_equation x y z A B C D ↔
      ∃ s t : ℝ, plane_parametric s t = (x, y, z) :=
by
  sorry

end plane_equation_proof_l124_124240


namespace x_minus_y_eq_neg_200_l124_124555

theorem x_minus_y_eq_neg_200 (x y : ℤ) (h1 : x + y = 290) (h2 : y = 245) : x - y = -200 := by
  sorry

end x_minus_y_eq_neg_200_l124_124555


namespace geometric_sequence_a9_l124_124306

theorem geometric_sequence_a9 (a : ℕ → ℝ) (q : ℝ) 
  (h1 : a 3 = 2) 
  (h2 : a 4 = 8 * a 7) 
  (h3 : ∀ n, a (n + 1) = a n * q) 
  (hq : q > 0) 
  : a 9 = 1 / 32 := 
by sorry

end geometric_sequence_a9_l124_124306


namespace gcd_884_1071_l124_124814

theorem gcd_884_1071 : Nat.gcd 884 1071 = 17 := by
  sorry

end gcd_884_1071_l124_124814


namespace problem1_problem2_l124_124771

-- Step 1
theorem problem1 (a b c A B C : ℝ) (h1 : Real.sin C * Real.sin (A - B) = Real.sin B * Real.sin (C - A)) :
  2 * a^2 = b^2 + c^2 := sorry

-- Step 2
theorem problem2 (a b c : ℝ) (h_a : a = 5) (h_cosA : Real.cos A = 25 / 31) 
  (h_conditions : 2 * a^2 = b^2 + c^2 ∧ 2 * b * c = a^2 / Real.cos A) :
  a + b + c = 14 := sorry

end problem1_problem2_l124_124771


namespace perfect_square_adjacent_smaller_l124_124159

noncomputable def is_perfect_square (n : ℕ) : Prop := 
    ∃ k : ℕ, k * k = n

theorem perfect_square_adjacent_smaller (m : ℕ) (hm : is_perfect_square m) : 
    ∃ k : ℕ, (k * k = m ∧ (k - 1) * (k - 1) = m - 2 * k + 1) := 
by 
  sorry

end perfect_square_adjacent_smaller_l124_124159


namespace tangent_slopes_product_eq_one_l124_124281

-- Given Point P(2, 2) and Circle C: x^2 + y^2 = 1
def P : ℝ × ℝ := (2, 2)
def C : ℝ × ℝ → Prop := λ q, q.1^2 + q.2^2 = 1

-- Defines the slopes of the tangents from point P to circle C
noncomputable def slopes_of_tangents (P : ℝ × ℝ) (C : ℝ × ℝ → Prop) : set ℝ := 
  { k | ∃ (x : ℝ), (x - 2)^2 + (k * (x - 2) - 2)^2 = 1 }

noncomputable def k1k2 (P : ℝ × ℝ) (C : ℝ × ℝ → Prop) : ℝ :=
  let s := slopes_of_tangents P C in 
  if h : s = {k | ∃ x, 3 * k^2 - 8 * k + 3 = 0}
  then let ⟨k1, k2⟩ := classical.some h in k1 * k2 else 0

-- Prove the value of k1 * k2 is 1
theorem tangent_slopes_product_eq_one :
  ∀ P C, k1k2 P C = 1 :=
sorry

end tangent_slopes_product_eq_one_l124_124281


namespace no_such_number_l124_124982

def isPerfectSquare (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def productOfDigitsIsPerfectSquare (n : ℕ) : Prop :=
  ∃ (d1 d2 : ℕ), n = 10 * d1 + d2 ∧ isPerfectSquare (d1 * d2)

theorem no_such_number :
  ¬ ∃ (N : ℕ),
    (N > 9) ∧ (N < 100) ∧ -- N is a two-digit number
    (N % 2 = 0) ∧        -- N is even
    (N % 13 = 0) ∧       -- N is a multiple of 13
    productOfDigitsIsPerfectSquare N := -- The product of digits of N is a perfect square
by
  sorry

end no_such_number_l124_124982


namespace number_of_marks_for_passing_l124_124680

theorem number_of_marks_for_passing (T P : ℝ) 
  (h1 : 0.40 * T = P - 40) 
  (h2 : 0.60 * T = P + 20) 
  (h3 : 0.45 * T = P - 10) :
  P = 160 :=
by
  sorry

end number_of_marks_for_passing_l124_124680


namespace border_area_correct_l124_124518

noncomputable def area_of_border (poster_height poster_width border_width : ℕ) : ℕ :=
  let framed_height := poster_height + 2 * border_width
  let framed_width := poster_width + 2 * border_width
  (framed_height * framed_width) - (poster_height * poster_width)

theorem border_area_correct :
  area_of_border 12 16 4 = 288 :=
by
  rfl

end border_area_correct_l124_124518


namespace hcf_of_two_numbers_l124_124352

theorem hcf_of_two_numbers (A B H L : ℕ) (h1 : A * B = 1800) (h2 : L = 200) (h3 : A * B = H * L) : H = 9 :=
by
  sorry

end hcf_of_two_numbers_l124_124352


namespace fraction_power_seven_l124_124816

theorem fraction_power_seven : (5 / 3 : ℚ) ^ 7 = 78125 / 2187 := 
by
  sorry

end fraction_power_seven_l124_124816


namespace f_value_at_3_l124_124724

theorem f_value_at_3 (a b : ℝ) (h : (a * (-3)^3 - b * (-3) + 2 = -1)) : a * (3)^3 - b * 3 + 2 = 5 :=
sorry

end f_value_at_3_l124_124724


namespace candy_bar_cost_correct_l124_124997

def initial_amount : ℕ := 4
def remaining_amount : ℕ := 3
def candy_bar_cost : ℕ := initial_amount - remaining_amount

theorem candy_bar_cost_correct : candy_bar_cost = 1 := by
  unfold candy_bar_cost
  sorry

end candy_bar_cost_correct_l124_124997


namespace problem1_problem2_l124_124470

theorem problem1 (x : ℝ) : (x + 4) ^ 2 - 5 * (x + 4) = 0 → x = -4 ∨ x = 1 :=
by
  sorry

theorem problem2 (x : ℝ) : x ^ 2 - 2 * x - 15 = 0 → x = -3 ∨ x = 5 :=
by
  sorry

end problem1_problem2_l124_124470


namespace length_of_garden_l124_124662

variables (w l : ℕ)

-- Definitions based on the problem conditions
def length_twice_width := l = 2 * w
def perimeter_eq_900 := 2 * l + 2 * w = 900

-- The statement to be proved
theorem length_of_garden (h1 : length_twice_width w l) (h2 : perimeter_eq_900 w l) : l = 300 :=
sorry

end length_of_garden_l124_124662


namespace calculate_value_l124_124700

theorem calculate_value : (2200 - 2090)^2 / (144 + 25) = 64 := 
by
  sorry

end calculate_value_l124_124700


namespace arithmetic_sequence_sum_a3_a4_a5_l124_124575

variable {a : ℕ → ℝ}
variable {d : ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum_a3_a4_a5
  (ha : is_arithmetic_sequence a d)
  (h : a 2 + a 3 + a 4 = 12) : 
  (7 * (a 0 + a 6)) / 2 = 28 := 
sorry

end arithmetic_sequence_sum_a3_a4_a5_l124_124575


namespace find_other_number_l124_124801

open Nat

def gcd (a b : ℕ) : ℕ := if a = 0 then b else gcd (b % a) a
noncomputable def lcm (a b : ℕ) : ℕ := a * b / gcd a b

def a : ℕ := 210
def lcm_ab : ℕ := 4620
def gcd_ab : ℕ := 21

theorem find_other_number (b : ℕ) (h_lcm : lcm a b = lcm_ab) (h_gcd : gcd a b = gcd_ab) :
  b = 462 := by
  sorry

end find_other_number_l124_124801


namespace greatest_three_digit_number_l124_124650

theorem greatest_three_digit_number 
  (n : ℕ)
  (h1 : n % 7 = 2)
  (h2 : n % 6 = 4)
  (h3 : n ≥ 100)
  (h4 : n < 1000) :
  n = 994 :=
sorry

end greatest_three_digit_number_l124_124650


namespace distance_center_is_12_l124_124627

-- Define the side length of the square and the radius of the circle
def side_length_square : ℝ := 5
def radius_circle : ℝ := 1

-- The center path forms a smaller square inside the original square
-- with side length 3 units
def side_length_smaller_square : ℝ := side_length_square - 2 * radius_circle

-- The perimeter of the smaller square, which is the path length that
-- the center of the circle travels
def distance_center_travel : ℝ := 4 * side_length_smaller_square

-- Prove that the distance traveled by the center of the circle is 12 units
theorem distance_center_is_12 : distance_center_travel = 12 := by
  -- the proof is skipped
  sorry

end distance_center_is_12_l124_124627


namespace smallest_sum_l124_124144

theorem smallest_sum (x y : ℕ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hxy_neq : x ≠ y) 
  (h_fraction : (1 : ℚ) / x + (1 : ℚ) / y = 1 / 15) : x + y = 64 :=
sorry

end smallest_sum_l124_124144


namespace inheritance_amount_l124_124596

-- Definitions of conditions
def inheritance (y : ℝ) : Prop :=
  let federalTaxes := 0.25 * y
  let remainingAfterFederal := 0.75 * y
  let stateTaxes := 0.1125 * y
  let totalTaxes := federalTaxes + stateTaxes
  totalTaxes = 12000

-- Theorem statement
theorem inheritance_amount (y : ℝ) (h : inheritance y) : y = 33103 :=
sorry

end inheritance_amount_l124_124596


namespace B_150_eq_I_l124_124174

open Matrix

def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![
     #[0, 1, 0],
     #[0, 0, 1],
     #[1, 0, 0]
   ]

theorem B_150_eq_I : B ^ 150 = (1 : Matrix (Fin 3) (Fin 3) ℝ) :=
by sorry

end B_150_eq_I_l124_124174


namespace max_profit_l124_124845

variables (x y : ℝ)

def profit (x y : ℝ) : ℝ := 50000 * x + 30000 * y

theorem max_profit :
  (3 * x + y ≤ 13) ∧ (2 * x + 3 * y ≤ 18) ∧ (x ≥ 0) ∧ (y ≥ 0) →
  (∃ x y, profit x y = 390000) :=
by
  sorry

end max_profit_l124_124845


namespace subsets_with_six_l124_124742

open Finset

theorem subsets_with_six (S : Finset ℕ) (hS : S = {1, 2, 3, 4, 5, 6}) :
  (S.filter (λ x, x = 6)).powerset.card = 32 :=
by
  rw [hS]
  have T : {1, 2, 3, 4, 5}.powerset = T.powerset := rfl
  sorry

end subsets_with_six_l124_124742


namespace radian_measure_of_central_angle_l124_124475

-- Given conditions
variables (l r : ℝ)
variables (h1 : (1 / 2) * l * r = 1)
variables (h2 : 2 * r + l = 4)

-- The theorem to prove
theorem radian_measure_of_central_angle (l r : ℝ) (h1 : (1 / 2) * l * r = 1) (h2 : 2 * r + l = 4) : 
  l / r = 2 :=
by 
  -- Proof steps are not provided as per the requirement
  sorry

end radian_measure_of_central_angle_l124_124475


namespace comparison_of_logs_l124_124284

noncomputable def a : ℝ := Real.logb 4 6
noncomputable def b : ℝ := Real.logb 4 0.2
noncomputable def c : ℝ := Real.logb 2 3

theorem comparison_of_logs : c > a ∧ a > b := by
  sorry

end comparison_of_logs_l124_124284


namespace ratio_of_albums_l124_124326

variable (M K B A : ℕ)
variable (s : ℕ)

-- Conditions
def adele_albums := (A = 30)
def bridget_albums := (B = A - 15)
def katrina_albums := (K = 6 * B)
def miriam_albums := (M = s * K)
def total_albums := (M + K + B + A = 585)

-- Proof statement
theorem ratio_of_albums (h1 : adele_albums A) (h2 : bridget_albums B A) (h3 : katrina_albums K B) 
(h4 : miriam_albums M s K) (h5 : total_albums M K B A) :
  s = 5 :=
by
  sorry

end ratio_of_albums_l124_124326


namespace initial_number_of_men_l124_124931

theorem initial_number_of_men (P : ℝ) (M : ℝ) (h1 : P = 15 * M * (P / (15 * M))) (h2 : P = 12.5 * (M + 200) * (P / (12.5 * (M + 200)))) : M = 1000 :=
by
  sorry

end initial_number_of_men_l124_124931


namespace seashells_ratio_l124_124024

theorem seashells_ratio (s_1 s_2 S t s3 : ℕ) (hs1 : s_1 = 5) (hs2 : s_2 = 7) (hS : S = 36)
  (ht : t = s_1 + s_2) (hs3 : s3 = S - t) :
  s3 / t = 2 :=
by
  rw [hs1, hs2] at ht
  simp at ht
  rw [hS, ht] at hs3
  simp at hs3
  sorry

end seashells_ratio_l124_124024


namespace circumcenter_rational_l124_124612

theorem circumcenter_rational {a1 b1 a2 b2 a3 b3 : ℚ} :
  ∃ (x y : ℚ), 
    ((x - a1)^2 + (y - b1)^2 = (x - a2)^2 + (y - b2)^2) ∧
    ((x - a1)^2 + (y - b1)^2 = (x - a3)^2 + (y - b3)^2) :=
sorry

end circumcenter_rational_l124_124612


namespace value_of_a_minus_b_l124_124753

theorem value_of_a_minus_b (a b : ℚ) (h1 : 3015 * a + 3021 * b = 3025) (h2 : 3017 * a + 3023 * b = 3027) : 
  a - b = - (7 / 3) :=
by
  sorry

end value_of_a_minus_b_l124_124753


namespace ratio_of_arithmetic_sequence_sums_l124_124867

theorem ratio_of_arithmetic_sequence_sums :
  let a1 := 2
  let d1 := 3
  let l1 := 41
  let n1 := (l1 - a1) / d1 + 1
  let sum1 := n1 / 2 * (a1 + l1)

  let a2 := 4
  let d2 := 4
  let l2 := 60
  let n2 := (l2 - a2) / d2 + 1
  let sum2 := n2 / 2 * (a2 + l2)
  sum1 / sum2 = 301 / 480 :=
by
  sorry

end ratio_of_arithmetic_sequence_sums_l124_124867


namespace eccentricity_of_hyperbola_l124_124279

theorem eccentricity_of_hyperbola
  (a b c e : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : c = a * e)
  (h4 : c^2 = a^2 + b^2)
  (h5 : ∀ B : ℝ × ℝ, B = (0, b))
  (h6 : ∀ F : ℝ × ℝ, F = (c, 0))
  (h7 : ∀ m_FB m_asymptote : ℝ, m_FB * m_asymptote = -1 → (m_FB = -b / c) ∧ (m_asymptote = b / a)) :
  e = (1 + Real.sqrt 5) / 2 :=
sorry

end eccentricity_of_hyperbola_l124_124279


namespace min_sum_xy_l124_124135

theorem min_sum_xy (x y : ℕ) (hx : x ≠ y) (hcond : ↑(1 / x) + ↑(1 / y) = 1 / 15) : x + y = 64 :=
sorry

end min_sum_xy_l124_124135


namespace range_of_a_l124_124021

def f (a x : ℝ) : ℝ := x^2 - a*x + a + 3
def g (a x : ℝ) : ℝ := x - a

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, ¬(f a x < 0 ∧ g a x < 0)) ↔ a ∈ Set.Icc (-3 : ℝ) 6 :=
sorry

end range_of_a_l124_124021


namespace annual_interest_income_l124_124246

variables (totalInvestment firstBondPrincipal secondBondPrincipal firstRate secondRate : ℝ)
           (firstInterest secondInterest totalInterest : ℝ)

def investment_conditions : Prop :=
  totalInvestment = 32000 ∧
  firstRate = 0.0575 ∧
  secondRate = 0.0625 ∧
  firstBondPrincipal = 20000 ∧
  secondBondPrincipal = totalInvestment - firstBondPrincipal

def calculate_interest (principal rate : ℝ) : ℝ := principal * rate

def total_annual_interest (firstInterest secondInterest : ℝ) : ℝ :=
  firstInterest + secondInterest

theorem annual_interest_income
  (hc : investment_conditions totalInvestment firstBondPrincipal secondBondPrincipal firstRate secondRate) :
  total_annual_interest (calculate_interest firstBondPrincipal firstRate)
    (calculate_interest secondBondPrincipal secondRate) = 1900 :=
by {
  sorry
}

end annual_interest_income_l124_124246


namespace points_earned_l124_124972

def each_enemy_points : ℕ := 3
def total_enemies : ℕ := 6
def defeated_enemies : ℕ := total_enemies - 2

theorem points_earned : defeated_enemies * each_enemy_points = 12 :=
by
  -- proof goes here
  sorry

end points_earned_l124_124972


namespace maximize_sector_area_l124_124300

noncomputable def max_area_sector_angle (r : ℝ) (l := 36 - 2 * r) (α := l / r) : ℝ :=
  α

theorem maximize_sector_area (h : ∀ r : ℝ, 2 * r + 36 - 2 * r = 36) :
  max_area_sector_angle 9 = 2 :=
by
  sorry

end maximize_sector_area_l124_124300


namespace olya_candies_l124_124919

theorem olya_candies (P M T O : ℕ) (h1 : P + M + T + O = 88) (h2 : 1 ≤ P) (h3 : 1 ≤ M) (h4 : 1 ≤ T) (h5 : 1 ≤ O) (h6 : M + T = 57) (h7 : P > M) (h8 : P > T) (h9 : P > O) : O = 1 :=
by
  sorry

end olya_candies_l124_124919


namespace smallest_pos_int_y_satisfies_congruence_l124_124553

theorem smallest_pos_int_y_satisfies_congruence :
  ∃ y : ℕ, (y > 0) ∧ (26 * y + 8) % 16 = 4 ∧ ∀ z : ℕ, (z > 0) ∧ (26 * z + 8) % 16 = 4 → y ≤ z :=
sorry

end smallest_pos_int_y_satisfies_congruence_l124_124553


namespace sampled_individual_l124_124644

theorem sampled_individual {population_size sample_size : ℕ} (population_size_cond : population_size = 1000)
  (sample_size_cond : sample_size = 20) (sampled_number : ℕ) (sampled_number_cond : sampled_number = 15) :
  (∃ n : ℕ, sampled_number + n * (population_size / sample_size) = 65) :=
by 
  sorry

end sampled_individual_l124_124644


namespace quadratic_equal_real_roots_l124_124576

theorem quadratic_equal_real_roots :
  ∃ k : ℝ, (∀ x : ℝ, x^2 - 4 * x + k = 0) ∧ k = 4 := by
  sorry

end quadratic_equal_real_roots_l124_124576


namespace sum_of_roots_l124_124782

theorem sum_of_roots :
  ∀ (x1 x2 : ℝ), (x1*x2 = 2 ∧ x1 + x2 = 3 ∧ x1 ≠ x2) ↔ (x1*x2 + 3*x1*x2 = 2 * x1 * x2 * x1:     by sorry

end sum_of_roots_l124_124782


namespace gift_items_l124_124219

theorem gift_items (x y z : ℕ) : 
  x + y + z = 20 ∧ 60 * x + 50 * y + 10 * z = 720 ↔ 
  ((x = 4 ∧ y = 8 ∧ z = 8) ∨ (x = 8 ∧ y = 3 ∧ z = 9)) :=
by sorry

end gift_items_l124_124219


namespace second_hand_distance_l124_124634

theorem second_hand_distance (r : ℝ) (t : ℝ) (π : ℝ) (hand_length_6cm : r = 6) (time_15_min : t = 15) : 
  ∃ d : ℝ, d = 180 * π :=
by
  sorry

end second_hand_distance_l124_124634


namespace probability_calculation_l124_124583

def prob_correct : ℝ := 0.8
def prob_incorrect : ℝ := 0.2

def probability_four_questions_before_advancing : ℝ := 0.128

theorem probability_calculation :
  (∃ scenario1 scenario2 : ℝ,
    scenario1 = prob_correct * prob_correct * prob_correct * prob_incorrect ∧
    scenario2 = prob_incorrect * prob_correct * prob_correct * prob_correct ∧
    scenario1 + scenario2 = probability_four_questions_before_advancing) :=
by {
  sorry
}

end probability_calculation_l124_124583


namespace vector_combination_l124_124309

noncomputable def m : ℝ := (0.5 * Real.sqrt 3 * Real.sqrt 5 + 9) / (2 * Real.sqrt 3 * Real.sqrt 5 + 12)
noncomputable def n : ℝ := (-9 * (3 / 5) + 13.5 * Real.sqrt 3) / 35

def vec_length {α : Type*} [inner_product_space ℝ α] (v : α) : ℝ :=
  real.sqrt (inner_product_space.inner v v)

variables {V : Type*} [inner_product_space ℝ V]

theorem vector_combination (OA OB OC : V)
  (hOA_len : vec_length OA = 2)
  (hOB_len : vec_length OB = 3)
  (hOC_len : vec_length OC = 3)
  (hTanAOC : real.tan (real.angle_of_vectors OA OC) = 4)
  (hAngleBOC : real.angle_of_vectors OB OC = real.pi / 3) :
  OC = m • OA + n • OB :=
sorry

end vector_combination_l124_124309


namespace max_happy_family_intervals_l124_124316

theorem max_happy_family_intervals {n : ℕ} (h : 0 < n) :
  ∃ (wp : set (ℕ × ℕ)), (∀ (i j : ℕ), (0 ≤ i) → (i < j) → (j ≤ n) → ((i, j) ∈ wp)) ∧
    (∀ (I₁ I₂ : (ℕ × ℕ)), (I₁ ∈ wp) → (I₂ ∈ wp) → ((fst I₁ ≠ fst I₂ ∨ snd I₁ ≠ snd I₂) → ((fst I₁ + snd I₁) ≠ (fst I₂ + snd I₂))) → 
    (I₁ ⊆ I₂ → (fst I₁ = fst I₂ ∨ snd I₁ = snd I₂))) →
  wp.card = Catalan n :=
sorry

end max_happy_family_intervals_l124_124316


namespace grid_values_constant_l124_124866

open Int

theorem grid_values_constant (f : ℤ × ℤ → ℕ)
  (h : ∀ x y, f (x, y) = (f (x + 1, y) + f (x - 1, y) + f (x, y + 1) + f (x, y - 1)) / 4) :
  ∃ c : ℕ, ∀ x y, f (x, y) = c :=
begin
  -- proof goes here
  sorry
end

end grid_values_constant_l124_124866


namespace girls_in_wind_band_not_string_band_l124_124106

def M_G : ℕ := 100
def F_G : ℕ := 80
def M_O : ℕ := 80
def F_O : ℕ := 100
def total_students : ℕ := 230
def boys_in_both : ℕ := 60

theorem girls_in_wind_band_not_string_band : (F_G - (total_students - (M_G + F_G + M_O + F_O - boys_in_both - boys_in_both))) = 10 :=
by
  sorry

end girls_in_wind_band_not_string_band_l124_124106


namespace problem_statement_l124_124293

variable (n : ℕ)
variable (op : ℕ → ℕ → ℕ)
variable (h1 : op 1 1 = 1)
variable (h2 : ∀ n, op (n+1) 1 = 3 * op n 1)

theorem problem_statement : op 5 1 - op 2 1 = 78 := by
  sorry

end problem_statement_l124_124293


namespace number_description_l124_124427

theorem number_description :
  4 * 10000 + 3 * 1000 + 7 * 100 + 5 * 10 + 2 + 8 / 10 + 4 / 100 = 43752.84 :=
by
  sorry

end number_description_l124_124427


namespace simplify_expression_l124_124469

theorem simplify_expression : |(-5^2 - 6 * 2)| = 37 := by
  sorry

end simplify_expression_l124_124469


namespace fred_found_28_more_seashells_l124_124956

theorem fred_found_28_more_seashells (tom_seashells : ℕ) (fred_seashells : ℕ) (h_tom : tom_seashells = 15) (h_fred : fred_seashells = 43) : 
  fred_seashells - tom_seashells = 28 := 
by 
  sorry

end fred_found_28_more_seashells_l124_124956


namespace subsets_containing_six_l124_124747

theorem subsets_containing_six :
  ∃ s : Finset (Fin 6), s = {1, 2, 3, 4, 5, 6} ∧ (∃ n : ℕ, n = 32 ∧ n = 2 ^ 5) := by
  sorry

end subsets_containing_six_l124_124747


namespace matrix_eq_l124_124016

open Matrix

def matA : Matrix (Fin 2) (Fin 2) ℤ := ![![1, 3], ![4, 2]]
def matI : Matrix (Fin 2) (Fin 2) ℤ := 1

theorem matrix_eq (A : Matrix (Fin 2) (Fin 2) ℤ)
  (hA : A = ![![1, 3], ![4, 2]]) :
  A ^ 7 = 9936 * A ^ 2 + 12400 * 1 :=
  by
    sorry

end matrix_eq_l124_124016


namespace pool_one_quarter_capacity_at_6_l124_124214

-- Variables and parameters
variables (volume : ℕ → ℝ) (T : ℕ)

-- Conditions
def doubles_every_hour : Prop :=
  ∀ t, volume (t + 1) = 2 * volume t

def full_capacity_at_8 : Prop :=
  volume 8 = T

def one_quarter_capacity (t : ℕ) : Prop :=
  volume t = T / 4

-- Theorem to prove
theorem pool_one_quarter_capacity_at_6 (h1 : doubles_every_hour volume) (h2 : full_capacity_at_8 volume T) : one_quarter_capacity volume T 6 :=
sorry

end pool_one_quarter_capacity_at_6_l124_124214


namespace round_trip_time_l124_124636

variable (boat_speed standing_water_speed stream_speed distance : ℕ)

theorem round_trip_time (boat_speed := 9) (stream_speed := 6) (distance := 170) : 
  (distance / (boat_speed - stream_speed) + distance / (boat_speed + stream_speed)) = 68 := by 
  sorry

end round_trip_time_l124_124636


namespace max_and_min_W_l124_124726

noncomputable def W (x y z : ℝ) : ℝ := 2 * x + 6 * y + 4 * z

theorem max_and_min_W {x y z : ℝ} (h1 : x + y + z = 1) (h2 : 3 * y + z ≥ 2) (h3 : 0 ≤ x ∧ x ≤ 1) (h4 : 0 ≤ y ∧ y ≤ 2) :
  ∃ (W_max W_min : ℝ), W_max = 6 ∧ W_min = 4 :=
by
  sorry

end max_and_min_W_l124_124726


namespace G_at_8_l124_124317

noncomputable def G (x : ℝ) : ℝ := sorry

theorem G_at_8 :
  (G 4 = 8) →
  (∀ x : ℝ, (x^2 + 3 * x + 2 ≠ 0) →
    G (2 * x) / G (x + 2) = 4 - (16 * x + 8) / (x^2 + 3 * x + 2)) →
  G 8 = 112 / 3 :=
by
  intros h1 h2
  sorry

end G_at_8_l124_124317


namespace G_five_times_of_2_l124_124472

def G (x : ℝ) : ℝ := (x - 2) ^ 2 - 1

theorem G_five_times_of_2 : G (G (G (G (G 2)))) = 1179395 := 
by 
  rw [G, G, G, G, G]; 
  sorry

end G_five_times_of_2_l124_124472


namespace smallest_x_y_sum_l124_124139

theorem smallest_x_y_sum :
  ∃ x y : ℕ,
    0 < x ∧ 0 < y ∧ x ≠ y ∧ (1 / (x : ℝ) + 1 / (y : ℝ) = 1 / 15) ∧ (x + y = 64) := 
by
  sorry

end smallest_x_y_sum_l124_124139


namespace simplify_sqrt_sum_l124_124033

theorem simplify_sqrt_sum : (Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2) :=
by
  sorry

end simplify_sqrt_sum_l124_124033


namespace sqrt_sum_l124_124045

theorem sqrt_sum (a b : ℕ) (ha : a = 72) (hb : b = 32) : 
  Real.sqrt a + Real.sqrt b = 10 * Real.sqrt 2 := 
by 
  rw [ha, hb] 
  -- Insert any further required simplifications as a formal proof or leave it abstracted.
  exact sorry -- skipping the proof to satisfy this step.

end sqrt_sum_l124_124045


namespace evaluate_expression_l124_124270

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem evaluate_expression :
  (4 / log_base 5 (2500^3) + 2 / log_base 2 (2500^3) = 1 / 3) := by
  sorry

end evaluate_expression_l124_124270


namespace frac_abs_div_a_plus_one_l124_124750

theorem frac_abs_div_a_plus_one (a : ℝ) (h : a ≠ 0) : abs a / a + 1 = 0 ∨ abs a / a + 1 = 2 :=
by sorry

end frac_abs_div_a_plus_one_l124_124750


namespace cost_of_downloading_360_songs_in_2005_is_144_dollars_l124_124580

theorem cost_of_downloading_360_songs_in_2005_is_144_dollars :
  (∀ (c_2004 c_2005 : ℕ), (∀ c : ℕ, c_2005 = c ∧ c_2004 = c + 32) →
  200 * c_2004 = 360 * c_2005 → 360 * c_2005 / 100 = 144) :=
  by sorry

end cost_of_downloading_360_songs_in_2005_is_144_dollars_l124_124580


namespace tile_floor_covering_l124_124826

theorem tile_floor_covering (n : ℕ) (h1 : 10 < n) (h2 : n < 20) (h3 : ∃ x, 9 * x = n^2) : n = 12 ∨ n = 15 ∨ n = 18 := by
  sorry

end tile_floor_covering_l124_124826


namespace find_a_l124_124532

def g (x : ℝ) : ℝ := 5 * x - 7

theorem find_a (a : ℝ) : g a = 0 → a = 7 / 5 := by
  intro h
  sorry

end find_a_l124_124532


namespace find_other_number_l124_124800

open Nat

def gcd (a b : ℕ) : ℕ := if a = 0 then b else gcd (b % a) a
noncomputable def lcm (a b : ℕ) : ℕ := a * b / gcd a b

def a : ℕ := 210
def lcm_ab : ℕ := 4620
def gcd_ab : ℕ := 21

theorem find_other_number (b : ℕ) (h_lcm : lcm a b = lcm_ab) (h_gcd : gcd a b = gcd_ab) :
  b = 462 := by
  sorry

end find_other_number_l124_124800


namespace part1_part2_l124_124707

-- Definition of the operation '※'
def operation (a b : ℝ) : ℝ := a^2 - b^2

-- Part 1: Proving 2※(-4) = -12
theorem part1 : operation 2 (-4) = -12 := 
by
  sorry

-- Part 2: Proving the solutions to the equation (x + 5)※3 = 0 are x = -8 and x = -2
theorem part2 : (∃ x : ℝ, operation (x + 5) 3 = 0) ↔ (x = -8 ∨ x = -2) := 
by
  sorry

end part1_part2_l124_124707


namespace inclination_angle_tan_60_perpendicular_l124_124481

/-
The inclination angle of the line given by x = tan(60 degrees) is 90 degrees.
-/
theorem inclination_angle_tan_60_perpendicular : 
  ∀ (x : ℝ), x = Real.tan (60 *Real.pi / 180) → 
  ∃ θ : ℝ, θ = 90 :=
sorry

end inclination_angle_tan_60_perpendicular_l124_124481


namespace find_speed_of_man_in_still_water_l124_124100

def speed_of_man_in_still_water (t1 t2 d1 d2: ℝ) (v_m v_s: ℝ) : Prop :=
  d1 / t1 = v_m + v_s ∧ d2 / t2 = v_m - v_s

theorem find_speed_of_man_in_still_water :
  ∃ v_m : ℝ, ∃ v_s : ℝ, speed_of_man_in_still_water 2 2 16 10 v_m v_s ∧ v_m = 6.5 :=
by
  sorry

end find_speed_of_man_in_still_water_l124_124100


namespace jessica_exam_time_l124_124313

theorem jessica_exam_time (total_questions : ℕ) (answered_questions : ℕ) (used_minutes : ℕ)
    (total_time : ℕ) (remaining_time : ℕ) (rate : ℚ) :
    total_questions = 80 ∧ answered_questions = 16 ∧ used_minutes = 12 ∧ total_time = 60 ∧ rate = (answered_questions : ℚ) / used_minutes →
    remaining_time = total_time - used_minutes →
    remaining_time = 48 :=
by
  -- Proof will be filled in here
  sorry

end jessica_exam_time_l124_124313


namespace functions_not_linearly_independent_l124_124450

noncomputable def linear_independent (s : set (ℝ → ℝ)) : Prop :=
∀ (l : ℝ → ℝ), (∀ x ∈ s, l x = 0) → (∀ x ∈ s, 0 = l x)

theorem functions_not_linearly_independent
  (n : ℕ)
  (x : fin n → ℝ → ℝ)
  (a : fin n → fin n → ℝ)
  (h_diff : ∀ i, differentiable ℝ (x i))
  (h_de : ∀ i t, derivative_at ℝ ℝ (x i) t = finset.sum finset.univ (λ j, a i j * x j t))
  (h_coeff_nonneg: ∀ i j, 0 ≤ a i j)
  (h_lim : ∀ i, filter.tendsto (x i) filter.at_top (filter.principal {0})) :
  ¬linear_independent {x i | i : fin n} := 
sorry

end functions_not_linearly_independent_l124_124450


namespace number_of_digits_is_nine_l124_124403

noncomputable def expression : ℝ := (8^4 * 4^12) / 2^8

theorem number_of_digits_is_nine (x : ℝ) (h : x = expression) : ⌊real.log10 x⌋ + 1 = 9 :=
sorry

end number_of_digits_is_nine_l124_124403


namespace part1_daily_sales_profit_part2_maximum_daily_profit_l124_124201

-- Definitions of initial conditions
def original_price : ℝ := 30
def original_sales_volume : ℝ := 60
def cost_price : ℝ := 15
def price_reduction_effect : ℝ := 10

-- Part 1: Prove the daily sales profit if the price is reduced by 2 yuan
def new_price_after_reduction (reduction : ℝ) : ℝ := original_price - reduction
def new_sales_volume (reduction : ℝ) : ℝ := original_sales_volume + reduction * price_reduction_effect
def profit_per_kg (selling_price : ℝ) : ℝ := selling_price - cost_price
def daily_sales_profit (reduction : ℝ) : ℝ := profit_per_kg (new_price_after_reduction reduction) * new_sales_volume reduction

theorem part1_daily_sales_profit : daily_sales_profit 2 = 1040 := by sorry

-- Part 2: Prove the selling price for maximum profit and the maximum profit
def profit_function (x : ℝ) : ℝ := (x - cost_price) * (original_sales_volume + (original_price - x) * price_reduction_effect)

theorem part2_maximum_daily_profit : 
  ∃ x, profit_function x = 1102.5 ∧ x = 51 / 2 := by sorry

end part1_daily_sales_profit_part2_maximum_daily_profit_l124_124201


namespace mod_remainder_1287_1499_l124_124223

theorem mod_remainder_1287_1499 : (1287 * 1499) % 300 = 213 := 
by 
  sorry

end mod_remainder_1287_1499_l124_124223


namespace spell_casting_contest_orders_l124_124737

-- Definition for factorial
def factorial : ℕ → ℕ
  | 0       => 1
  | (n + 1) => (n + 1) * factorial n

-- Theorem statement: number of ways to order 4 contestants is 4!
theorem spell_casting_contest_orders : factorial 4 = 24 := by
  sorry

end spell_casting_contest_orders_l124_124737


namespace box_length_is_24_l124_124513

theorem box_length_is_24 (L : ℕ) (h1 : ∀ s : ℕ, (L * 40 * 16 = 30 * s^3) → s ∣ 40 ∧ s ∣ 16) (h2 : ∃ s : ℕ, s ∣ 40 ∧ s ∣ 16) : L = 24 :=
by
  sorry

end box_length_is_24_l124_124513


namespace simplify_sqrt_sum_l124_124042

theorem simplify_sqrt_sum : sqrt 72 + sqrt 32 = 10 * sqrt 2 := sorry

end simplify_sqrt_sum_l124_124042


namespace percentage_of_alcohol_in_new_mixture_l124_124673

def original_solution_volume : ℕ := 11
def added_water_volume : ℕ := 3
def alcohol_percentage_original : ℝ := 0.42

def total_volume : ℕ := original_solution_volume + added_water_volume
def amount_of_alcohol : ℝ := alcohol_percentage_original * original_solution_volume

theorem percentage_of_alcohol_in_new_mixture :
  (amount_of_alcohol / total_volume) * 100 = 33 := by
  sorry

end percentage_of_alcohol_in_new_mixture_l124_124673


namespace skittles_per_friend_l124_124909

theorem skittles_per_friend (ts : ℕ) (nf : ℕ) (h1 : ts = 200) (h2 : nf = 5) : (ts / nf = 40) :=
by sorry

end skittles_per_friend_l124_124909


namespace circle_equation_tangent_y_axis_center_on_line_chord_length_condition_l124_124131

theorem circle_equation_tangent_y_axis_center_on_line_chord_length_condition :
  ∃ (x₀ y₀ r : ℝ), 
  (x₀ - 3 * y₀ = 0) ∧ 
  (r = |3 * y₀|) ∧ 
  ((x₀ + 3)^2 + (y₀ - 1)^2 = r^2 ∨ (x₀ - 3)^2 + (y₀ + 1)^2 = r^2) :=
sorry

end circle_equation_tangent_y_axis_center_on_line_chord_length_condition_l124_124131


namespace min_value_expr_l124_124884

noncomputable def expr (θ : Real) : Real :=
  3 * (Real.cos θ) + 2 / (Real.sin θ) + 2 * Real.sqrt 2 * (Real.tan θ)

theorem min_value_expr :
  ∃ (θ : Real), 0 < θ ∧ θ < Real.pi / 2 ∧ expr θ = (7 * Real.sqrt 2) / 2 := 
by
  sorry

end min_value_expr_l124_124884


namespace area_bounded_by_curves_l124_124668

noncomputable def area_of_figure_bounded_by_curves : ℝ :=
  let f_x (t : ℝ) := 6 * Real.cos t
  let f_y (t : ℝ) := 4 * Real.sin t
  let g_y : ℝ := 2 * Real.sqrt 3
  let t1 := Real.arcsin (g_y / 4)
  let t2 := π - t1
  let bounded_area := ∫ t in t2..t1, f_y t * (f_x' t) in
  let rectangle_area := g_y * (f_x t1 - f_x t2)
  bounded_area - rectangle_area

theorem area_bounded_by_curves :
  area_of_figure_bounded_by_curves = 4 * π - 12 * Real.sqrt 3 := by
  sorry

end area_bounded_by_curves_l124_124668


namespace rob_baseball_cards_l124_124333

theorem rob_baseball_cards
  (r j r_d : ℕ)
  (hj : j = 40)
  (h_double : r_d = j / 5)
  (h_cards : r = 3 * r_d) :
  r = 24 :=
by
  sorry

end rob_baseball_cards_l124_124333


namespace find_third_number_x_l124_124864

variable {a b : ℝ}

theorem find_third_number_x (h : a < b) :
  (∃ x : ℝ, x = a * b / (2 * b - a) ∧ x < a) ∨ 
  (∃ x : ℝ, x = 2 * a * b / (a + b) ∧ a < x ∧ x < b) ∨ 
  (∃ x : ℝ, x = a * b / (2 * a - b) ∧ a < b ∧ b < x) :=
sorry

end find_third_number_x_l124_124864


namespace find_x_l124_124640

-- define initial quantities of apples and oranges
def initial_apples (x : ℕ) : ℕ := 3 * x + 1
def initial_oranges (x : ℕ) : ℕ := 4 * x + 12

-- define the condition that the number of oranges is twice the number of apples
def condition (x : ℕ) : Prop := initial_oranges x = 2 * initial_apples x

-- define the final state
def final_apples : ℕ := 1
def final_oranges : ℕ := 12

-- theorem to prove that the number of times is 5
theorem find_x : ∃ x : ℕ, condition x ∧ final_apples = 1 ∧ final_oranges = 12 :=
by
  use 5
  sorry

end find_x_l124_124640


namespace find_divisor_l124_124970

theorem find_divisor (D : ℕ) : 
  let dividend := 109
  let quotient := 9
  let remainder := 1
  (dividend = D * quotient + remainder) → D = 12 :=
by
  sorry

end find_divisor_l124_124970


namespace internet_usage_minutes_l124_124933

-- Define the given conditions
variables (M P E : ℕ)

-- Problem statement
theorem internet_usage_minutes (h : P ≠ 0) : 
  (∀ M P E : ℕ, ∃ y : ℕ, y = (100 * E * M) / P) :=
by {
  sorry
}

end internet_usage_minutes_l124_124933


namespace smallest_x_y_sum_l124_124141

theorem smallest_x_y_sum :
  ∃ x y : ℕ,
    0 < x ∧ 0 < y ∧ x ≠ y ∧ (1 / (x : ℝ) + 1 / (y : ℝ) = 1 / 15) ∧ (x + y = 64) := 
by
  sorry

end smallest_x_y_sum_l124_124141


namespace max_distance_circle_to_point_A_l124_124121

noncomputable def circle_eq (x y : ℝ) : Prop := (x - 2) ^ 2 + y ^ 2 = 2

noncomputable def point_A : ℝ × ℝ := (-1, 3)

noncomputable def max_distance (d : ℝ) : Prop :=
  ∃ x y, circle_eq x y ∧ d = Real.sqrt ((2 + 1)^2 + (0 - 3)^2) + Real.sqrt 2 

theorem max_distance_circle_to_point_A : max_distance (4 * Real.sqrt 2) :=
sorry

end max_distance_circle_to_point_A_l124_124121


namespace sum_of_three_integers_l124_124069

theorem sum_of_three_integers :
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ a * b * c = 125 ∧ a + b + c = 31 :=
by
  sorry

end sum_of_three_integers_l124_124069


namespace ages_total_l124_124830

theorem ages_total (a b c : ℕ) (h1 : b = 8) (h2 : a = b + 2) (h3 : b = 2 * c) : a + b + c = 22 := by
  sorry

end ages_total_l124_124830


namespace binom_product_l124_124992

open Nat

-- Define binomial coefficient
def binom (n k : ℕ) : ℕ := n.choose k

theorem binom_product :
  binom 10 3 * binom 8 3 = 6720 := by
  sorry

end binom_product_l124_124992


namespace smallest_integer_in_set_l124_124760

theorem smallest_integer_in_set :
  ∀ (n : ℤ), (n + 6 > 2 * ((n + (n+1) + (n+2) + (n+3) + (n+4) + (n+5) + (n+6)) / 7)) → n = -1 :=
by
  intros n h
  sorry

end smallest_integer_in_set_l124_124760


namespace alice_flips_heads_probability_l124_124383

def prob_heads (n k : ℕ) (p q : ℚ) : ℚ :=
  (Nat.choose n k) * (p ^ k) * (q ^ (n - k))

theorem alice_flips_heads_probability :
  prob_heads 8 3 (1/3 : ℚ) (2/3 : ℚ) = 1792 / 6561 :=
by
  sorry

end alice_flips_heads_probability_l124_124383


namespace parabola_vertex_in_other_l124_124495

theorem parabola_vertex_in_other (p q a : ℝ) (h₁ : a ≠ 0) 
  (h₂ : ∀ (x : ℝ),  x = a → pa^2 = p * x^2) 
  (h₃ : ∀ (x : ℝ),  x = 0 → 0 = q * (x - a)^2 + pa^2) : 
  p + q = 0 := 
sorry

end parabola_vertex_in_other_l124_124495


namespace functional_equation_solution_l124_124715

theorem functional_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x ^ 2 - y ^ 2) = (x - y) * (f x + f y)) :
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x :=
sorry

end functional_equation_solution_l124_124715


namespace ellipse_with_given_foci_and_point_l124_124067

noncomputable def areFociEqual (a b c₁ c₂ : ℝ) : Prop :=
  c₁ = Real.sqrt (a^2 - b^2) ∧ c₂ = Real.sqrt (a^2 - b^2)

noncomputable def isPointOnEllipse (x₀ y₀ a₂ b₂ : ℝ) : Prop :=
  (x₀^2 / a₂) + (y₀^2 / b₂) = 1

theorem ellipse_with_given_foci_and_point :
  ∃a b : ℝ, 
    areFociEqual 8 3 a b ∧
    a = Real.sqrt 5 ∧ b = Real.sqrt 5 ∧
    isPointOnEllipse 3 (-2) 15 10  :=
sorry

end ellipse_with_given_foci_and_point_l124_124067


namespace functional_equation_l124_124714

noncomputable def f : ℝ → ℝ := sorry

theorem functional_equation (f : ℝ → ℝ) (h : ∀ x y : ℝ, f (x^2 - y^2) = (x - y) * (f x + f y)) :
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x :=
begin
  sorry
end

end functional_equation_l124_124714


namespace benny_number_of_kids_l124_124254

-- Define the conditions
def benny_has_dollars (d: ℕ): Prop := d = 360
def cost_per_apple (c: ℕ): Prop := c = 4
def apples_shared (num_kids num_apples: ℕ): Prop := num_apples = 5 * num_kids

-- State the main theorem
theorem benny_number_of_kids : 
  ∀ (d c k a : ℕ), benny_has_dollars d → cost_per_apple c → apples_shared k a → k = 18 :=
by
  intros d c k a hd hc ha
  -- The goal is to prove k = 18; use the provided conditions
  sorry

end benny_number_of_kids_l124_124254


namespace Doug_age_l124_124084

theorem Doug_age
  (B : ℕ) (D : ℕ) (N : ℕ)
  (h1 : 2 * B = N)
  (h2 : B + D = 90)
  (h3 : 20 * N = 2000) : 
  D = 40 := sorry

end Doug_age_l124_124084


namespace boolean_logic_problem_l124_124973

theorem boolean_logic_problem (p q : Prop) (h₁ : ¬(p ∧ q)) (h₂ : ¬(¬p)) : ¬q :=
by {
  sorry
}

end boolean_logic_problem_l124_124973


namespace quadratic_inequality_solution_l124_124411

theorem quadratic_inequality_solution (x : ℝ) : x^2 - 36 * x + 318 ≤ 0 ↔ 18 - Real.sqrt 6 ≤ x ∧ x ≤ 18 + Real.sqrt 6 := by
  sorry

end quadratic_inequality_solution_l124_124411


namespace least_possible_value_of_smallest_integer_l124_124370

theorem least_possible_value_of_smallest_integer :
  ∀ (A B C D : ℕ), 
    A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D →
    (A + B + C + D) / 4 = 68 →
    D = 90 →
    A = 5 :=
by
  intros A B C D h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end least_possible_value_of_smallest_integer_l124_124370


namespace original_number_is_10_l124_124752

theorem original_number_is_10 (x : ℤ) (h : 2 * x + 3 = 23) : x = 10 :=
sorry

end original_number_is_10_l124_124752


namespace domain_of_h_l124_124817

noncomputable def h (x : ℝ) : ℝ := (3 * x + 1) / (x ^ 2 - 9)

theorem domain_of_h :
  ∀ x : ℝ, (∃ y, h x = y) ↔ x ∈ ((set.Ioo (real.is_real_neg_infty) (-3)) ∪ (set.Ioo (-3) 3) ∪ (set.Ioo 3 (real.is_real_pos_infty))) :=
by 
  sorry

end domain_of_h_l124_124817


namespace analytical_expression_of_f_l124_124150

theorem analytical_expression_of_f (f : ℝ → ℝ) :
  (∀ x : ℝ, x ≠ 0 → f (x + 1 / x) = x^2 + 1 / x^2) →
  (∀ y : ℝ, (y ≥ 2 ∨ y ≤ -2) → f y = y^2 - 2) :=
by
  intro h1 y hy
  sorry

end analytical_expression_of_f_l124_124150


namespace sum_ge_six_l124_124921

theorem sum_ge_six (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b + b * c + c * a ≥ 12) : a + b + c ≥ 6 :=
by
  sorry

end sum_ge_six_l124_124921


namespace three_digit_number_divisibility_four_digit_number_divisibility_l124_124432

-- Definition of three-digit number
def is_three_digit_number (a : ℕ) : Prop := 100 ≤ a ∧ a ≤ 999

-- Definition of four-digit number
def is_four_digit_number (b : ℕ) : Prop := 1000 ≤ b ∧ b ≤ 9999

-- First proof problem
theorem three_digit_number_divisibility (a : ℕ) (h : is_three_digit_number a) : 
  (1001 * a) % 7 = 0 ∧ (1001 * a) % 11 = 0 ∧ (1001 * a) % 13 = 0 := 
sorry

-- Second proof problem
theorem four_digit_number_divisibility (b : ℕ) (h : is_four_digit_number b) : 
  (10001 * b) % 73 = 0 ∧ (10001 * b) % 137 = 0 := 
sorry

end three_digit_number_divisibility_four_digit_number_divisibility_l124_124432


namespace three_digit_divisible_by_11_l124_124790

theorem three_digit_divisible_by_11 {x y z : ℕ} 
  (h1 : 0 ≤ x ∧ x < 10) 
  (h2 : 0 ≤ y ∧ y < 10) 
  (h3 : 0 ≤ z ∧ z < 10) 
  (h4 : x + z = y) : 
  (100 * x + 10 * y + z) % 11 = 0 := 
by 
  sorry

end three_digit_divisible_by_11_l124_124790


namespace number_of_students_taking_statistics_l124_124001

theorem number_of_students_taking_statistics
  (total_students : ℕ)
  (history_students : ℕ)
  (history_or_statistics : ℕ)
  (history_only : ℕ)
  (history_and_statistics : ℕ := history_students - history_only)
  (statistics_only : ℕ := history_or_statistics - history_and_statistics - history_only)
  (statistics_students : ℕ := history_and_statistics + statistics_only) :
  total_students = 90 → history_students = 36 → history_or_statistics = 59 → history_only = 29 →
    statistics_students = 30 :=
by
  intros
  -- Proof goes here but is omitted.
  sorry

end number_of_students_taking_statistics_l124_124001


namespace time_to_fill_remaining_l124_124371

-- Define the rates at which pipes P and Q fill the cistern
def rate_P := 1 / 12
def rate_Q := 1 / 15

-- Define the time both pipes are open together
def time_both_open := 4

-- Calculate the combined rate when both pipes are open
def combined_rate := rate_P + rate_Q

-- Calculate the amount of the cistern filled in the time both pipes are open
def filled_amount_both_open := time_both_open * combined_rate

-- Calculate the remaining amount to fill after Pipe P is turned off
def remaining_amount := 1 - filled_amount_both_open

-- Calculate the time it will take for Pipe Q alone to fill the remaining amount
def time_Q_to_fill_remaining := remaining_amount / rate_Q

-- The final theorem
theorem time_to_fill_remaining : time_Q_to_fill_remaining = 6 := by
  sorry

end time_to_fill_remaining_l124_124371


namespace element_in_set_l124_124914

-- Definitions
def U : Set ℕ := {1, 2, 3, 4, 5}
def complement_U_M : Set ℕ := {1, 2}

-- The main statement to prove
theorem element_in_set (M : Set ℕ) (h1 : U = {1, 2, 3, 4, 5}) (h2 : U \ M = complement_U_M) : 3 ∈ M := 
sorry

end element_in_set_l124_124914


namespace odd_function_sum_zero_l124_124433

def odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g x

theorem odd_function_sum_zero (g : ℝ → ℝ) (a : ℝ) (h_odd : odd_function g) : 
  g a + g (-a) = 0 :=
by 
  sorry

end odd_function_sum_zero_l124_124433


namespace min_colors_needed_l124_124661

theorem min_colors_needed (n : ℕ) (h : n + n.choose 2 ≥ 12) : n = 5 :=
sorry

end min_colors_needed_l124_124661


namespace cream_cheese_cost_l124_124126

theorem cream_cheese_cost:
  ∃ (B C : ℝ), (2 * B + 3 * C = 12) ∧ (4 * B + 2 * C = 14) ∧ (C = 2.5) :=
by
  sorry

end cream_cheese_cost_l124_124126


namespace actual_distance_traveled_l124_124755

theorem actual_distance_traveled (D T : ℝ)
  (h1 : D = 10 * T)
  (h2 : D + 20 = 20 * T) : D = 20 :=
by
  sorry

end actual_distance_traveled_l124_124755


namespace yasmine_chocolate_beverage_l124_124827

theorem yasmine_chocolate_beverage :
  ∃ (m s : ℕ), (∀ k : ℕ, k > 0 → (∃ n : ℕ, 4 * n = 7 * k) → (m, s) = (7 * k, 4 * k)) ∧
  (2 * 7 * 1 + 1.4 * 4 * 1) = 19.6 := by
sorry

end yasmine_chocolate_beverage_l124_124827


namespace max_intersection_points_l124_124356

theorem max_intersection_points (C L : Type) [fintype C] [fintype L] (hC : fintype.card C = 3) (hL : fintype.card L = 2) :
  (∑ c1 in C, ∑ c2 in (C \ {c1}), 2) + (∑ l in L, ∑ c in C, 2) + 1 = 19 :=
by
  sorry

end max_intersection_points_l124_124356


namespace target_runs_l124_124907

theorem target_runs (r1 r2 : ℝ) (o1 o2 : ℕ) (target : ℝ) :
  r1 = 3.6 ∧ o1 = 10 ∧ r2 = 6.15 ∧ o2 = 40 → target = (r1 * o1) + (r2 * o2) := by
  sorry

end target_runs_l124_124907


namespace max_a_for_three_solutions_l124_124406

-- Define the equation as a Lean function
def equation (x a : ℝ) : ℝ :=
  (|x-2| + 2 * a)^2 - 3 * (|x-2| + 2 * a) + 4 * a * (3 - 4 * a)

-- Statement of the proof problem
theorem max_a_for_three_solutions :
  (∃ (a : ℝ), (∀ x : ℝ, equation x a = 0) ∧
  (∀ (b : ℝ), (∀ x : ℝ, equation x b = 0) → b ≤ 0.5)) :=
sorry

end max_a_for_three_solutions_l124_124406


namespace part_a_part_b_part_c_l124_124004

noncomputable def rect := {x : ℝ × ℝ // (0 ≤ x.1 ∧ x.1 ≤ 1 ) ∧ (0 ≤ x.2 ∧ x.2 ≤ 1)}
noncomputable def shapes : Type := rect → Prop

variable (shape1 shape2 shape3 shape4 shape5 : shapes)
variable (h1 : ∀ x, shape1 x → x ∈ rect)
variable (h2 : ∀ x, shape2 x → x ∈ rect)
variable (h3 : ∀ x, shape3 x → x ∈ rect)
variable (h4 : ∀ x, shape4 x → x ∈ rect)
variable (h5 : ∀ x, shape5 x → x ∈ rect)
variable (A1 : ∀ x, shape1 x → (1 / 2 : ℝ))
variable (A2 : ∀ x, shape2 x → (1 / 2 : ℝ))
variable (A3 : ∀ x, shape3 x → (1 / 2 : ℝ))
variable (A4 : ∀ x, shape4 x → (1 / 2 : ℝ))
variable (A5 : ∀ x, shape5 x → (1 / 2 : ℝ))

theorem part_a : ∃ (S1 S2 : shapes), S1 ≠ S2 ∧ (∀ x, S1 x ∧ S2 x → (3 / 20 : ℝ)) :=
sorry

theorem part_b : ∃ (S1 S2 : shapes), S1 ≠ S2 ∧ (∀ x, S1 x ∧ S2 x → (1 / 5 : ℝ)) :=
sorry

theorem part_c : ∃ (S1 S2 S3 : shapes), S1 ≠ S2 ∧ S2 ≠ S3 ∧ S1 ≠ S3 ∧ (∀ x, S1 x ∧ S2 x ∧ S3 x → (1 / 20 : ℝ)) :=
sorry

end part_a_part_b_part_c_l124_124004


namespace average_of_first_21_multiples_of_7_l124_124095

theorem average_of_first_21_multiples_of_7 :
  let a1 := 7
  let d := 7
  let n := 21
  let an := a1 + (n - 1) * d
  let Sn := n / 2 * (a1 + an)
  Sn / n = 77 :=
by
  let a1 := 7
  let d := 7
  let n := 21
  let an := a1 + (n - 1) * d
  let Sn := n / 2 * (a1 + an)
  have h1 : an = 147 := by
    sorry
  have h2 : Sn = 1617 := by
    sorry
  have h3 : Sn / n = 77 := by
    sorry
  exact h3

end average_of_first_21_multiples_of_7_l124_124095


namespace circumcenter_is_rational_l124_124620

theorem circumcenter_is_rational (a1 a2 a3 b1 b2 b3 : ℚ)
  (h1 : (a2 - a1) ≠ 0 ∨ (b2 - b1) ≠ 0)
  (h2 : (a3 - a1) ≠ 0 ∨ (b3 - b1) ≠ 0) :
  ∃ x y : ℚ,
    ((a2 - a1) * x + (b2 - b1) * y = (a2^2 - a1^2 + b2^2 - b1^2) / 2) ∧
    ((a3 - a1) * x + (b3 - b1) * y = (a3^2 - a1^2 + b3^2 - b1^2) / 2) :=
begin
  -- proof goes here
  sorry,
end

end circumcenter_is_rational_l124_124620


namespace given_system_solution_l124_124397

noncomputable def solve_system : Prop :=
  ∃ x y z : ℝ, 
  x + y + z = 1 ∧ 
  x^2 + y^2 + z^2 = 1 ∧ 
  x^3 + y^3 + z^3 = 89 / 125 ∧ 
  (x = 2 / 5 ∧ y = (3 + Real.sqrt 33) / 10 ∧ z = (3 - Real.sqrt 33) / 10 ∨ 
   x = 2 / 5 ∧ y = (3 - Real.sqrt 33) / 10 ∧ z = (3 + Real.sqrt 33) / 10 ∨ 
   x = (3 + Real.sqrt 33) / 10 ∧ y = 2 / 5 ∧ z = (3 - Real.sqrt 33) / 10 ∨ 
   x = (3 - Real.sqrt 33) / 10 ∧ y = 2 / 5 ∧ z = (3 + Real.sqrt 33) / 10 ∨ 
   x = (3 + Real.sqrt 33) / 10 ∧ y = (3 - Real.sqrt 33) / 10 ∧ z = 2 / 5 ∨ 
   x = (3 - Real.sqrt 33) / 10 ∧ y = (3 + Real.sqrt 33) / 10 ∧ z = 2 / 5)

theorem given_system_solution : solve_system :=
sorry

end given_system_solution_l124_124397


namespace part1_part2_l124_124602

noncomputable def A : Set ℝ := {x | x^2 - 3 * x + 2 = 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | x^2 + 2 * (a + 1) * x + (a^2 - 5) = 0}

theorem part1 (a : ℝ) (h : A ∩ B a = {2}) : a = -1 ∨ a = -3 := by
  sorry

theorem part2 (a : ℝ) (h : A ∪ B a = A) : a ≤ -3 := by
  sorry

end part1_part2_l124_124602


namespace length_of_PR_l124_124591

theorem length_of_PR (x y : ℝ) (h₁ : x^2 + y^2 = 250) : 
  ∃ PR : ℝ, PR = 10 * Real.sqrt 5 :=
by
  use Real.sqrt (2 * (x^2 + y^2))
  sorry

end length_of_PR_l124_124591


namespace rhombus_area_8_cm2_l124_124439

open Real

noncomputable def rhombus_area (side : ℝ) (angle : ℝ) : ℝ :=
  (side * side * sin angle) / 2 * 2

theorem rhombus_area_8_cm2 (side : ℝ) (angle : ℝ) (h1 : side = 4) (h2 : angle = π / 4) : rhombus_area side angle = 8 :=
by
  -- Definitions and calculations are omitted and replaced with 'sorry'
  sorry

end rhombus_area_8_cm2_l124_124439


namespace calculate_division_of_powers_l124_124869

theorem calculate_division_of_powers (a : ℝ) : a^8 / a^2 = a^6 :=
by sorry

end calculate_division_of_powers_l124_124869


namespace intersection_eq_l124_124426

def M : Set ℝ := { x | -2 ≤ x ∧ x ≤ 0 }
def N : Set ℝ := { -1, 0, 1 }

theorem intersection_eq : M ∩ N = { -1, 0 } := by
  sorry

end intersection_eq_l124_124426


namespace triangle_perimeter_l124_124273

theorem triangle_perimeter (a b c : ℕ) (ha : a = 7) (hb : b = 10) (hc : c = 15) :
  a + b + c = 32 :=
by
  -- Given the lengths of the sides
  have H1 : a = 7 := ha
  have H2 : b = 10 := hb
  have H3 : c = 15 := hc
  
  -- Therefore, we need to prove the sum
  sorry

end triangle_perimeter_l124_124273


namespace deepak_present_age_l124_124369

/-- Let Rahul and Deepak's current ages be 4x and 3x respectively
  Given that:
  1. The ratio between Rahul and Deepak's ages is 4:3
  2. After 6 years, Rahul's age will be 26 years
  Prove that Deepak's present age is 15 years.
-/
theorem deepak_present_age (x : ℕ) (hx : 4 * x + 6 = 26) : 3 * x = 15 :=
by
  sorry

end deepak_present_age_l124_124369


namespace f_13_eq_223_l124_124429

def f (n : ℕ) : ℕ := n^2 + n + 41

theorem f_13_eq_223 : f 13 = 223 :=
by
  sorry

end f_13_eq_223_l124_124429


namespace find_solution_l124_124271

-- Definitions for the problem
def is_solution (x y z t : ℕ) : Prop := (x > 0 ∧ y > 0 ∧ z > 0 ∧ t > 0 ∧ (2^y + 2^z * 5^t - 5^x = 1))

-- Statement of the theorem
theorem find_solution : ∀ x y z t : ℕ, is_solution x y z t → (x, y, z, t) = (2, 4, 1, 1) := by
  sorry

end find_solution_l124_124271


namespace single_train_car_passenger_count_l124_124243

theorem single_train_car_passenger_count (P : ℕ) 
  (h1 : ∀ (plane_capacity train_capacity : ℕ), plane_capacity = 366 →
    train_capacity = 16 * P →
      (train_capacity = (2 * plane_capacity) + 228)) : 
  P = 60 :=
by
  sorry

end single_train_car_passenger_count_l124_124243


namespace prob_exactly_four_blue_pens_l124_124904

-- Define the basic probabilities
def prob_blue : ℚ := 5 / 9
def prob_red : ℚ := 4 / 9

-- Define the probability of any specific sequence of 4 blue and 3 red pens
def prob_specific_sequence : ℚ := (prob_blue ^ 4) * (prob_red ^ 3)

-- Define the combinatorial number of ways to pick 4 blue pens out of 7
def num_ways : ℕ := Nat.choose 7 4

-- Total probability calculation
def total_prob : ℚ := num_ways * prob_specific_sequence

-- The final result should be approximately 0.294
theorem prob_exactly_four_blue_pens :
  total_prob ≈ 0.294 := sorry

end prob_exactly_four_blue_pens_l124_124904


namespace tomatoes_eaten_l124_124738

theorem tomatoes_eaten 
  (initial_tomatoes : ℕ) 
  (final_tomatoes : ℕ) 
  (half_given : ℕ) 
  (B : ℕ) 
  (h_initial : initial_tomatoes = 127) 
  (h_final : final_tomatoes = 54) 
  (h_half : half_given = final_tomatoes * 2) 
  (h_remaining : initial_tomatoes - half_given = B)
  : B = 19 := 
by
  sorry

end tomatoes_eaten_l124_124738


namespace solve_inequality_l124_124062

theorem solve_inequality (a x : ℝ) (ha : a ≠ 0) :
  (a > 0 → (x^2 - 5 * a * x + 6 * a^2 > 0 ↔ (x < 2 * a ∨ x > 3 * a))) ∧
  (a < 0 → (x^2 - 5 * a * x + 6 * a^2 > 0 ↔ (x < 3 * a ∨ x > 2 * a))) :=
by
  sorry

end solve_inequality_l124_124062


namespace array_element_count_l124_124110

theorem array_element_count (A : Finset ℕ) 
  (h1 : ∀ n ∈ A, n ≠ 1 → (∃ a ∈ [2, 3, 5], a ∣ n)) 
  (h2 : ∀ n ∈ A, (2 * n ∈ A ∨ 3 * n ∈ A ∨ 5 * n ∈ A) ↔ (n ∈ A ∧ 2 * n ∈ A ∧ 3 * n ∈ A ∧ 5 * n ∈ A)) 
  (card_A_range : 300 ≤ A.card ∧ A.card ≤ 400) : 
  A.card = 364 := 
sorry

end array_element_count_l124_124110


namespace problem1_problem2_problem3_l124_124372

-- Problem 1 Statement
theorem problem1 : (π - 3.14)^0 + (1 / 2)^(-1) + (-1)^(2023) = 2 :=
by {
  -- use tactic mode to assist the proof
  sorry
}

-- Problem 2 Statement
theorem problem2 (b : ℝ) : (-b)^2 * b + 6 * b^4 / (2 * b) + (-2 * b)^3 = -4 * b^3 :=
by {
  -- use tactic mode to assist the proof
  sorry
}

-- Problem 3 Statement
theorem problem3 (x : ℝ) : (x - 1)^2 - x * (x + 2) = -4 * x + 1 :=
by {
  -- use tactic mode to assist the proof
  sorry
}

end problem1_problem2_problem3_l124_124372


namespace cubic_root_sum_l124_124017

-- Assume we have three roots a, b, and c of the polynomial x^3 - 3x - 2 = 0
variables {a b c : ℝ}

-- Using Vieta's formulas for the polynomial x^3 - 3x - 2 = 0
axiom Vieta1 : a + b + c = 0
axiom Vieta2 : a * b + a * c + b * c = -3
axiom Vieta3 : a * b * c = -2

-- The proof that the given expression evaluates to 9
theorem cubic_root_sum:
  a^2 * (b - c)^2 + b^2 * (c - a)^2 + c^2 * (a - b)^2 = 9 :=
by
  sorry

end cubic_root_sum_l124_124017


namespace distributor_B_lower_avg_price_l124_124350

theorem distributor_B_lower_avg_price (p_1 p_2 : ℝ) (h : p_1 < p_2) :
  (p_1 + p_2) / 2 > (2 * p_1 * p_2) / (p_1 + p_2) :=
by {
  sorry
}

end distributor_B_lower_avg_price_l124_124350


namespace max_value_ab_ac_bc_l124_124776

open Real

theorem max_value_ab_ac_bc {a b c : ℝ} (h : a + 3 * b + c = 6) : 
  ab + ac + bc ≤ 4 :=
sorry

end max_value_ab_ac_bc_l124_124776


namespace smallest_perimeter_of_triangle_PQR_l124_124081

noncomputable def triangle_PQR_perimeter (PQ PR QR : ℕ) (QJ : ℝ) 
  (h1 : PQ = PR) (h2 : QJ = 10) : ℕ :=
2 * (PQ + QR)

theorem smallest_perimeter_of_triangle_PQR (PQ PR QR : ℕ) (QJ : ℝ) :
  PQ = PR → QJ = 10 → 
  ∃ p, p = triangle_PQR_perimeter PQ PR QR QJ (by assumption) (by assumption) ∧ p = 78 :=
sorry

end smallest_perimeter_of_triangle_PQR_l124_124081


namespace other_number_of_given_conditions_l124_124807

theorem other_number_of_given_conditions 
  (a b : ℕ) 
  (h_lcm : Nat.lcm a b = 4620) 
  (h_gcd : Nat.gcd a b = 21) 
  (h_a : a = 210) : 
  b = 462 := 
sorry

end other_number_of_given_conditions_l124_124807


namespace find_n_l124_124418

-- Definitions for conditions given in the problem
def a₂ (a : ℕ → ℕ) : Prop := a 2 = 3
def consecutive_sum (S : ℕ → ℕ) (n : ℕ) : Prop := ∀ n > 3, S n - S (n - 3) = 51
def total_sum (S : ℕ → ℕ) (n : ℕ) : Prop := S n = 100

-- The main proof problem
theorem find_n (a : ℕ → ℕ) (S : ℕ → ℕ) (n : ℕ) 
  (h₁ : a₂ a) (h₂ : consecutive_sum S n) (h₃ : total_sum S n) : n = 10 :=
sorry

end find_n_l124_124418


namespace range_of_k_l124_124068

theorem range_of_k 
  (k : ℝ) 
  (line_intersects_hyperbola : ∃ x y : ℝ, y = k * x + 2 ∧ x^2 - y^2 = 6) : 
  -Real.sqrt (15) / 3 < k ∧ k < Real.sqrt (15) / 3 := 
by
  sorry

end range_of_k_l124_124068


namespace driving_scenario_l124_124659

theorem driving_scenario (x : ℝ) (h1 : x > 0) :
  (240 / x) - (240 / (1.5 * x)) = 1 :=
by
  sorry

end driving_scenario_l124_124659


namespace percent_increase_decrease_l124_124107

theorem percent_increase_decrease (P y : ℝ) (h : (P * (1 + y / 100) * (1 - y / 100) = 0.90 * P)) :
    y = 31.6 :=
by
  sorry

end percent_increase_decrease_l124_124107


namespace other_number_eq_462_l124_124805

theorem other_number_eq_462 (a b : ℕ) 
  (lcm_ab : Nat.lcm a b = 4620) 
  (gcd_ab : Nat.gcd a b = 21) 
  (a_eq : a = 210) : b = 462 := 
by
  sorry

end other_number_eq_462_l124_124805


namespace fuel_used_l124_124453

theorem fuel_used (x : ℝ) (h1 : x + 0.8 * x = 27) : x = 15 :=
sorry

end fuel_used_l124_124453


namespace arithmetic_seq_sum_l124_124588

theorem arithmetic_seq_sum (a : ℕ → ℝ) (h_arith : ∀ n, a (n + 1) - a n = a 1 - a 0)
    (h_sum : a 0 + a 1 + a 2 + a 3 = 30) : a 1 + a 2 = 15 :=
by
  sorry

end arithmetic_seq_sum_l124_124588


namespace function_passing_point_l124_124896

variable (f : ℝ → ℝ)

theorem function_passing_point (h : f 1 = 0) : f (0 + 1) + 1 = 1 := by
  calc f (0 + 1) + 1 = f 1 + 1 := by rfl
                  ... = 0 + 1 := by rw [h]
                  ... = 1 := by rfl

#check function_passing_point

end function_passing_point_l124_124896


namespace geometric_sequence_find_a_n_l124_124322

variable {n m p : ℕ}
variable {a : ℕ → ℕ}
variable {S : ℕ → ℕ}

-- Given conditions
axiom h1 : ∀ n, 2 * S (n + 1) - 3 * S n = 2 * a 1
axiom h2 : a 1 ≠ 0
axiom h3 : ∀ n, S (n + 1) = S n + a (n + 1)

-- Part (1)
theorem geometric_sequence : ∃ r, ∀ n, a (n + 1) = r * a n :=
sorry

-- Part (2)
axiom p_geq_3 : 3 ≤ p
axiom a1_pos : 0 < a 1
axiom a_p_pos : 0 < a p
axiom constraint1 : a 1 ≥ m ^ (p - 1)
axiom constraint2 : a p ≤ (m + 1) ^ (p - 1)

theorem find_a_n : ∀ n, a n = 2 ^ (p - 1) * (3 / 2) ^ (n - 1) :=
sorry

end geometric_sequence_find_a_n_l124_124322


namespace brocard_angle_property_l124_124995

-- Define the triangle ABC with angles α, β, and γ.
variables {α β γ ω : ℝ}

-- Define the condition that ω is the Brocard angle such that angle KAB = angle KBC = angle KCA.
def brocard_angle (α β γ ω : ℝ) : Prop :=
  ∃ (K : EuclideanGeometry.Point),
    (EuclideanGeometry.angle K (EuclideanGeometry.Point.mk 0 0) (EuclideanGeometry.Point.mk 1 0) = ω) ∧
    (EuclideanGeometry.angle K (EuclideanGeometry.Point.mk 1 0) (EuclideanGeometry.Point.mk 0 1) = ω) ∧
    (EuclideanGeometry.angle K (EuclideanGeometry.Point.mk 0 1) (EuclideanGeometry.Point.mk 0 0) = ω)

-- Prove the desired property of the Brocard angle.
theorem brocard_angle_property (α β γ : ℝ) (hαβγ : α + β + γ = π) :
  brocard_angle α β γ ω → real.cot ω = real.cot α + real.cot β + real.cot γ :=
by
  sorry

end brocard_angle_property_l124_124995


namespace max_value_ab_ac_bc_l124_124777

open Real

theorem max_value_ab_ac_bc {a b c : ℝ} (h : a + 3 * b + c = 6) : 
  ab + ac + bc ≤ 4 :=
sorry

end max_value_ab_ac_bc_l124_124777


namespace symmetric_point_x_axis_l124_124610

def symmetric_point (M : ℝ × ℝ) : ℝ × ℝ := (M.1, -M.2)

theorem symmetric_point_x_axis :
  ∀ (M : ℝ × ℝ), M = (3, -4) → symmetric_point M = (3, 4) :=
by
  intros M h
  rw [h]
  dsimp [symmetric_point]
  congr
  sorry

end symmetric_point_x_axis_l124_124610


namespace math_problem_l124_124639
noncomputable def sum_of_terms (a b c d : ℕ) : ℕ := a + b + c + d

theorem math_problem
  (x y : ℝ)
  (h₁ : x + y = 5)
  (h₂ : 5 * x * y = 7) :
  ∃ a b c d : ℕ, 
  x = (a + b * Real.sqrt c) / d ∧
  a = 25 ∧ b = 1 ∧ c = 485 ∧ d = 10 ∧ sum_of_terms a b c d = 521 := by
sorry

end math_problem_l124_124639


namespace second_to_last_digit_of_special_number_l124_124239

theorem second_to_last_digit_of_special_number :
  ∀ (N : ℕ), (N % 10 = 0) ∧ (∃ k : ℕ, k > 0 ∧ N = 2 * 5^k) →
  (N / 10) % 10 = 5 :=
by
  sorry

end second_to_last_digit_of_special_number_l124_124239


namespace molecular_weight_boric_acid_l124_124822

theorem molecular_weight_boric_acid :
  let H := 1.008  -- atomic weight of Hydrogen in g/mol
  let B := 10.81  -- atomic weight of Boron in g/mol
  let O := 16.00  -- atomic weight of Oxygen in g/mol
  let H3BO3 := 3 * H + B + 3 * O  -- molecular weight of H3BO3
  H3BO3 = 61.834 :=  -- correct molecular weight of H3BO3
by
  sorry

end molecular_weight_boric_acid_l124_124822


namespace rectangle_perimeter_l124_124030

-- Defining the given conditions
def rectangleArea := 4032
noncomputable def ellipseArea := 4032 * Real.pi
noncomputable def b := Real.sqrt 2016
noncomputable def a := 2 * Real.sqrt 2016

-- Problem statement: the perimeter of the rectangle
theorem rectangle_perimeter (x y : ℝ) (h1 : x * y = rectangleArea)
  (h2 : x + y = 2 * a) : 2 * (x + y) = 8 * Real.sqrt 2016 :=
by
  sorry

end rectangle_perimeter_l124_124030


namespace john_learns_vowels_in_fifteen_days_l124_124720

def days_to_learn_vowels (days_per_vowel : ℕ) (num_vowels : ℕ) : ℕ :=
  days_per_vowel * num_vowels

theorem john_learns_vowels_in_fifteen_days :
  days_to_learn_vowels 3 5 = 15 :=
by
  -- Proof goes here
  sorry

end john_learns_vowels_in_fifteen_days_l124_124720


namespace smallest_sum_l124_124142

theorem smallest_sum (x y : ℕ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hxy_neq : x ≠ y) 
  (h_fraction : (1 : ℚ) / x + (1 : ℚ) / y = 1 / 15) : x + y = 64 :=
sorry

end smallest_sum_l124_124142


namespace snake_count_l124_124187

def neighborhood : Type := {n : ℕ // n = 200}

def percentage (total : ℕ) (percent : ℕ) : ℕ := total * percent / 100

def owns_only_dogs (total : ℕ) : ℕ := percentage total 13
def owns_only_cats (total : ℕ) : ℕ := percentage total 10
def owns_only_snakes (total : ℕ) : ℕ := percentage total 5
def owns_only_rabbits (total : ℕ) : ℕ := percentage total 7
def owns_only_birds (total : ℕ) : ℕ := percentage total 3
def owns_only_exotic (total : ℕ) : ℕ := percentage total 6
def owns_dogs_and_cats (total : ℕ) : ℕ := percentage total 8
def owns_dogs_cats_exotic (total : ℕ) : ℕ := percentage total 9
def owns_cats_and_snakes (total : ℕ) : ℕ := percentage total 4
def owns_cats_and_birds (total : ℕ) : ℕ := percentage total 2
def owns_snakes_and_rabbits (total : ℕ) : ℕ := percentage total 5
def owns_snakes_and_birds (total : ℕ) : ℕ := percentage total 3
def owns_rabbits_and_birds (total : ℕ) : ℕ := percentage total 1
def owns_all_except_snakes (total : ℕ) : ℕ := percentage total 2
def owns_all_except_birds (total : ℕ) : ℕ := percentage total 1
def owns_three_with_exotic (total : ℕ) : ℕ := percentage total 11
def owns_only_chameleons (total : ℕ) : ℕ := percentage total 3
def owns_only_hedgehogs (total : ℕ) : ℕ := percentage total 2

def exotic_pet_owners (total : ℕ) : ℕ :=
  owns_only_exotic total + owns_dogs_cats_exotic total + owns_all_except_snakes total +
  owns_all_except_birds total + owns_three_with_exotic total + owns_only_chameleons total +
  owns_only_hedgehogs total

def exotic_pet_owners_with_snakes (total : ℕ) : ℕ :=
  percentage (exotic_pet_owners total) 25

def total_snake_owners (total : ℕ) : ℕ :=
  owns_only_snakes total + owns_cats_and_snakes total +
  owns_snakes_and_rabbits total + owns_snakes_and_birds total +
  exotic_pet_owners_with_snakes total

theorem snake_count (nh : neighborhood) : total_snake_owners (nh.val) = 51 :=
by
  sorry

end snake_count_l124_124187


namespace quadratic_function_value_l124_124517

theorem quadratic_function_value (a b c : ℝ) 
  (h1 : a + b + c = 5) 
  (h2 : a - b + c = 9) :
  a + 3 * b + c = 1 := 
by 
  sorry

end quadratic_function_value_l124_124517


namespace triangle_XYZ_r_s_max_sum_l124_124957

theorem triangle_XYZ_r_s_max_sum
  (r s : ℝ)
  (h_area : 1/2 * abs (r * (15 - 18) + 10 * (18 - s) + 20 * (s - 15)) = 90)
  (h_slope : s = -3 * r + 61.5) :
  r + s ≤ 42.91 :=
sorry

end triangle_XYZ_r_s_max_sum_l124_124957


namespace ferry_captives_successfully_l124_124012

-- Definition of conditions
def valid_trip_conditions (trips: ℕ) (captives: ℕ) : Prop :=
  captives = 43 ∧
  (∀ k < trips, k % 2 = 0 ∨ k % 2 = 1) ∧     -- Trips done in pairs or singles
  (∀ k < captives, k > 40)                    -- At least 40 other captives known as werewolves

-- Theorem statement to be proved
theorem ferry_captives_successfully (trips : ℕ) (captives : ℕ) (result : Prop) : 
  valid_trip_conditions trips captives → result = true := by sorry

end ferry_captives_successfully_l124_124012


namespace alan_total_spending_l124_124250

-- Define the conditions
def eggs_bought : ℕ := 20
def price_per_egg : ℕ := 2
def chickens_bought : ℕ := 6
def price_per_chicken : ℕ := 8

-- Total cost calculation
def cost_eggs : ℕ := eggs_bought * price_per_egg
def cost_chickens : ℕ := chickens_bought * price_per_chicken
def total_amount_spent : ℕ := cost_eggs + cost_chickens

-- Prove the total amount spent
theorem alan_total_spending : total_amount_spent = 88 := by
  show cost_eggs + cost_chickens = 88
  sorry

end alan_total_spending_l124_124250


namespace transform_to_100_l124_124343

theorem transform_to_100 (a b c : ℤ) (h : Int.gcd (Int.gcd a b) c = 1) :
  ∃ f : (ℤ × ℤ × ℤ → ℤ × ℤ × ℤ), (∀ p : ℤ × ℤ × ℤ,
    ∃ q : ℕ, q ≤ 5 ∧ f^[q] p = (1, 0, 0)) :=
sorry

end transform_to_100_l124_124343


namespace find_actual_balance_l124_124092

-- Define the given conditions
def current_balance : ℝ := 90000
def rate : ℝ := 0.10

-- Define the target
def actual_balance_before_deduction (X : ℝ) : Prop :=
  (X * (1 - rate) = current_balance)

-- Statement of the theorem
theorem find_actual_balance : ∃ X : ℝ, actual_balance_before_deduction X :=
  sorry

end find_actual_balance_l124_124092


namespace simplify_sqrt_sum_l124_124054

theorem simplify_sqrt_sum : sqrt 72 + sqrt 32 = 10 * sqrt 2 :=
by
  sorry

end simplify_sqrt_sum_l124_124054


namespace max_points_of_intersection_l124_124358

theorem max_points_of_intersection (n m : ℕ) : n = 3 → m = 2 → 
  (let circle_intersections := (n * (n - 1)) / 2 * 2;
       line_circle_intersections := m * n * 2;
       line_intersections := (m * (m - 1)) / 2;
       total_intersections := circle_intersections + line_circle_intersections + line_intersections
   in total_intersections = 19) :=
by
  intros hn hm
  let circle_intersections := (n * (n - 1)) / 2 * 2
  let line_circle_intersections := m * n * 2
  let line_intersections := (m * (m - 1)) / 2
  let total_intersections := circle_intersections + line_circle_intersections + line_intersections
  rw [hn, hm] at *
  calc total_intersections
      = ((3 * 2) * 2) + (2 * 3 * 2) + 1 : by sorry
      ... = 6 + 12 + 1  : by sorry
      ... = 19         : by sorry

end max_points_of_intersection_l124_124358


namespace iterate_F_l124_124019

def F (x : ℝ) : ℝ := x^3 + 3*x^2 + 3*x

theorem iterate_F (x : ℝ) : (Nat.iterate F 2017 x) = (x + 1)^(3^2017) - 1 :=
by
  sorry

end iterate_F_l124_124019


namespace arithmetic_expression_l124_124815

theorem arithmetic_expression : 5 + 12 / 3 - 3 ^ 2 + 1 = 1 := by
  sorry

end arithmetic_expression_l124_124815


namespace min_total_bananas_l124_124813

noncomputable def total_bananas_condition (b1 b2 b3 : ℕ) : Prop :=
  let m1 := (5/8 : ℚ) * b1 + (5/16 : ℚ) * b2 + (23/48 : ℚ) * b3
  let m2 := (3/16 : ℚ) * b1 + (3/8 : ℚ) * b2 + (23/48 : ℚ) * b3
  let m3 := (3/16 : ℚ) * b1 + (5/16 : ℚ) * b2 + (1/24 : ℚ) * b3
  (((m1 : ℚ) * 4) = ((m2 : ℚ) * 3)) ∧ (((m1 : ℚ) * 4) = ((m3 : ℚ) * 2))

theorem min_total_bananas : ∃ (b1 b2 b3 : ℕ), b1 + b2 + b3 = 192 ∧ total_bananas_condition b1 b2 b3 :=
sorry

end min_total_bananas_l124_124813


namespace cos_neg_30_eq_sqrt_3_div_2_l124_124212

theorem cos_neg_30_eq_sqrt_3_div_2 : 
  Real.cos (-30 * Real.pi / 180) = Real.sqrt 3 / 2 := by
  sorry

end cos_neg_30_eq_sqrt_3_div_2_l124_124212


namespace expression_equals_8_l124_124256

-- Define the expression we are interested in.
def expression : ℚ :=
  (1 + 1 / 2) * (1 + 1 / 3) * (1 + 1 / 4) * (1 + 1 / 5) * (1 + 1 / 6) * (1 + 1 / 7)

-- Statement we need to prove
theorem expression_equals_8 : expression = 8 := by
  sorry

end expression_equals_8_l124_124256


namespace graph_inverse_point_sum_l124_124197

theorem graph_inverse_point_sum 
  (f : ℝ → ℝ) (f_inv : ℝ → ℝ) 
  (h1 : ∀ x, f_inv (f x) = x) 
  (h2 : ∀ x, f (f_inv x) = x) 
  (h3 : f 2 = 6) 
  (h4 : (2, 3) ∈ {p : ℝ × ℝ | p.snd = f p.fst / 2}) :
  (6, 1) ∈ {p : ℝ × ℝ | p.snd = f_inv p.fst / 2} ∧ (6 + 1 = 7) :=
by
  sorry

end graph_inverse_point_sum_l124_124197


namespace probability_neither_red_nor_purple_l124_124373

section Probability

def total_balls : ℕ := 60
def red_balls : ℕ := 15
def purple_balls : ℕ := 3
def total_red_or_purple_balls : ℕ := red_balls + purple_balls
def non_red_or_purple_balls : ℕ := total_balls - total_red_or_purple_balls

theorem probability_neither_red_nor_purple :
  (non_red_or_purple_balls : ℚ) / (total_balls : ℚ) = 7 / 10 :=
by
  sorry

end Probability

end probability_neither_red_nor_purple_l124_124373


namespace sum_units_digits_3a_l124_124319

theorem sum_units_digits_3a (a : ℕ) (h_pos : 0 < a) (h_units : (2 * a) % 10 = 4) : 
  ((3 * (a % 10) = (6 : ℕ) ∨ (3 * (a % 10) = (21 : ℕ))) → 6 + 1 = 7) := 
by
  sorry

end sum_units_digits_3a_l124_124319


namespace age_difference_proof_l124_124251

def AlexAge : ℝ := 16.9996700066
def AlexFatherAge (A : ℝ) (F : ℝ) : Prop := F = 2 * A + 4.9996700066
def FatherAgeSixYearsAgo (A : ℝ) (F : ℝ) : Prop := A - 6 = 1 / 3 * (F - 6)

theorem age_difference_proof :
  ∃ (A F : ℝ), A = 16.9996700066 ∧
  (AlexFatherAge A F) ∧
  (FatherAgeSixYearsAgo A F) :=
by
  sorry

end age_difference_proof_l124_124251


namespace complement_union_result_l124_124749

open Set

variable (U : Set ℕ)
variable (A : Set ℕ)
variable (B : Set ℕ)

theorem complement_union_result :
    U = { x | x < 6 } →
    A = {1, 2, 3} → 
    B = {2, 4, 5} → 
    (U \ A) ∪ (U \ B) = {0, 1, 3, 4, 5} :=
by
    intros hU hA hB
    sorry

end complement_union_result_l124_124749


namespace rect_area_162_l124_124104

def rectangle_field_area (w l : ℝ) (A : ℝ) : Prop :=
  w = (1/2) * l ∧ 2 * (w + l) = 54 ∧ A = w * l

theorem rect_area_162 {w l A : ℝ} :
  rectangle_field_area w l A → A = 162 :=
by
  intro h
  sorry

end rect_area_162_l124_124104


namespace angle_intersecting_lines_l124_124767

/-- 
Given three lines intersecting at a point forming six equal angles 
around the point, each angle equals 60 degrees.
-/
theorem angle_intersecting_lines (x : ℝ) (h : 6 * x = 360) : x = 60 := by
  sorry

end angle_intersecting_lines_l124_124767


namespace pencil_packing_l124_124460

theorem pencil_packing (a : ℕ) : 
  (200 ≤ a ∧ a ≤ 300) →
  (a % 10 = 7) →
  (a % 12 = 9) →
  (a = 237 ∨ a = 297) :=
by {
  assume h_range h_red_boxes h_blue_boxes,
  sorry
}

end pencil_packing_l124_124460


namespace Mark_paid_total_cost_l124_124325

def length_of_deck : ℝ := 30
def width_of_deck : ℝ := 40
def cost_per_sq_ft_without_sealant : ℝ := 3
def additional_cost_per_sq_ft_sealant : ℝ := 1

def area (length width : ℝ) : ℝ := length * width
def total_cost (area cost_without_sealant cost_sealant : ℝ) : ℝ := 
  area * cost_without_sealant + area * cost_sealant

theorem Mark_paid_total_cost :
  total_cost (area length_of_deck width_of_deck) cost_per_sq_ft_without_sealant additional_cost_per_sq_ft_sealant = 4800 := 
by
  -- Placeholder for proof
  sorry

end Mark_paid_total_cost_l124_124325


namespace tape_needed_for_large_box_l124_124876

-- Definition of the problem conditions
def tape_per_large_box (L : ℕ) : Prop :=
  -- Each large box takes L feet of packing tape to seal
  -- Each medium box takes 2 feet of packing tape to seal
  -- Each small box takes 1 foot of packing tape to seal
  -- Each box also takes 1 foot of packing tape to stick the address label on
  -- Debbie packed two large boxes this afternoon
  -- Debbie packed eight medium boxes this afternoon
  -- Debbie packed five small boxes this afternoon
  -- Debbie used 44 feet of tape in total
  2 * L + 2 + 24 + 10 = 44

theorem tape_needed_for_large_box : ∃ L : ℕ, tape_per_large_box L ∧ L = 4 :=
by {
  -- Proof goes here
  sorry
}

end tape_needed_for_large_box_l124_124876


namespace add_base6_l124_124856

def base6_to_base10 (n : Nat) : Nat :=
  let d0 := n % 10
  let n1 := n / 10
  let d1 := n1 % 10
  6 * d1 + d0

theorem add_base6 (a b : Nat) (ha : base6_to_base10 a = 23) (hb : base6_to_base10 b = 10) : 
  base6_to_base10 (53 : Nat) = 33 :=
by
  sorry

end add_base6_l124_124856


namespace square_of_binomial_l124_124296

theorem square_of_binomial (c : ℝ) (h : ∃ a : ℝ, x^2 + 50 * x + c = (x + a)^2) : c = 625 :=
by
  sorry

end square_of_binomial_l124_124296


namespace sqrt_of_4_l124_124947

theorem sqrt_of_4 : ∃ y : ℝ, y^2 = 4 ∧ (y = 2 ∨ y = -2) :=
by
  sorry

end sqrt_of_4_l124_124947


namespace log_comparison_l124_124277

theorem log_comparison (a b : ℝ) (h1 : 0 < a) (h2 : a < e) (h3 : 0 < b) (h4 : b < e) (h5 : a < b) :
  a * Real.log b > b * Real.log a := sorry

end log_comparison_l124_124277


namespace probability_of_hitting_target_at_least_once_l124_124242

noncomputable def prob_hit_target_once : ℚ := 2/3

noncomputable def prob_miss_target_once : ℚ := 1 - prob_hit_target_once

noncomputable def prob_miss_target_three_times : ℚ := prob_miss_target_once ^ 3

noncomputable def prob_hit_target_at_least_once : ℚ := 1 - prob_miss_target_three_times

theorem probability_of_hitting_target_at_least_once :
  prob_hit_target_at_least_once = 26 / 27 := 
sorry

end probability_of_hitting_target_at_least_once_l124_124242


namespace min_distinct_integers_for_ap_and_gp_l124_124652

theorem min_distinct_integers_for_ap_and_gp (n : ℕ) :
  (∀ (b q a d : ℤ), b ≠ 0 ∧ q ≠ 0 ∧ q ≠ 1 ∧ q ≠ -1 →
    (∃ (i : ℕ), i < 5 → b * (q ^ i) = a + i * d) ∧ 
    (∃ (j : ℕ), j < 5 → b * (q ^ j) ≠ a + j * d) ↔ n ≥ 6) :=
by {
  sorry
}

end min_distinct_integers_for_ap_and_gp_l124_124652


namespace boys_meet_time_is_correct_l124_124217

structure TrackMeetProblem where
  (track_length : ℕ) -- Track length in meters
  (speed_first_boy_kmh : ℚ) -- Speed of the first boy in km/hr
  (speed_second_boy_kmh : ℚ) -- Speed of the second boy in km/hr

noncomputable def time_to_meet (p : TrackMeetProblem) : ℚ :=
  let speed_first_boy_ms := (p.speed_first_boy_kmh * 1000) / 3600
  let speed_second_boy_ms := (p.speed_second_boy_kmh * 1000) / 3600
  let relative_speed := speed_first_boy_ms + speed_second_boy_ms
  (p.track_length : ℚ) / relative_speed

theorem boys_meet_time_is_correct (p : TrackMeetProblem) : 
  p.track_length = 4800 → 
  p.speed_first_boy_kmh = 61.3 → 
  p.speed_second_boy_kmh = 97.5 → 
  time_to_meet p = 108.8 := by
  intros
  sorry  

end boys_meet_time_is_correct_l124_124217


namespace jill_water_stored_l124_124014

theorem jill_water_stored (n : ℕ) (h : n = 24) : 
  8 * (1 / 4 : ℝ) + 8 * (1 / 2 : ℝ) + 8 * 1 = 14 :=
by
  sorry

end jill_water_stored_l124_124014


namespace fraction_addition_l124_124966

theorem fraction_addition :
  let a := (1 : ℚ) / 6
  let b := (5 : ℚ) / 12
  a + b = 7 / 12 :=
by
  let a := (1 : ℚ) / 6
  let b := (5 : ℚ) / 12
  have : a + b = 7 / 12 := sorry
  exact this

end fraction_addition_l124_124966


namespace pencil_packing_l124_124461

theorem pencil_packing (a : ℕ) : 
  (200 ≤ a ∧ a ≤ 300) →
  (a % 10 = 7) →
  (a % 12 = 9) →
  (a = 237 ∨ a = 297) :=
by {
  assume h_range h_red_boxes h_blue_boxes,
  sorry
}

end pencil_packing_l124_124461


namespace B_days_to_complete_work_l124_124232

theorem B_days_to_complete_work (B : ℕ) (hB : B ≠ 0)
  (A_work_days : ℕ := 9) (combined_days : ℕ := 6)
  (work_rate_A : ℚ := 1 / A_work_days) (work_rate_combined : ℚ := 1 / combined_days):
  (1 / B : ℚ) = work_rate_combined - work_rate_A → B = 18 :=
by
  intro h
  sorry

end B_days_to_complete_work_l124_124232


namespace original_number_multiple_of_8_l124_124376

theorem original_number_multiple_of_8 (x y : ℤ) (h : 14 * x = 112 * y) : ∃ k : ℤ, x = 8 * k :=
by
  use y
  have h' : 112 = 14 * 8 := rfl
  rw [h', mul_assoc, mul_comm 8 y, ← mul_assoc, mul_comm 14 x] at h
  rw [← mul_assoc, mul_eq_mul_left_iff] at h
  cases h; { use h; rfl }
  sorry

end original_number_multiple_of_8_l124_124376


namespace car_arrives_before_bus_l124_124679

theorem car_arrives_before_bus
  (d : ℝ) (s_bus : ℝ) (s_car : ℝ) (v : ℝ)
  (h1 : d = 240)
  (h2 : s_bus = 40)
  (h3 : s_car = v)
  : 56 < v ∧ v < 120 := 
sorry

end car_arrives_before_bus_l124_124679


namespace domain_of_function_l124_124479

theorem domain_of_function : 
  {x : ℝ | 0 < x ∧ 4 - x^2 > 0} = {x : ℝ | 0 < x ∧ x < 2} :=
sorry

end domain_of_function_l124_124479


namespace unanswered_questions_equal_nine_l124_124769

theorem unanswered_questions_equal_nine
  (x y z : ℕ)
  (h1 : 5 * x + 2 * z = 93)
  (h2 : 4 * x - y = 54)
  (h3 : x + y + z = 30) : 
  z = 9 := by
  sorry

end unanswered_questions_equal_nine_l124_124769


namespace find_b_minus_a_l124_124484

noncomputable def rotate_90_counterclockwise (x y xc yc : ℝ) : ℝ × ℝ :=
  (xc + (-(y - yc)), yc + (x - xc))

noncomputable def reflect_about_y_eq_x (x y : ℝ) : ℝ × ℝ :=
  (y, x)

theorem find_b_minus_a (a b : ℝ) :
  let xc := 2
  let yc := 3
  let P := (a, b)
  let P_rotated := rotate_90_counterclockwise a b xc yc
  let P_reflected := reflect_about_y_eq_x P_rotated.1 P_rotated.2
  P_reflected = (4, 1) →
  b - a = 1 :=
by
  intros
  sorry

end find_b_minus_a_l124_124484


namespace find_a_given_coefficient_l124_124563

theorem find_a_given_coefficient (a : ℝ) :
  (∀ x : ℝ, a ≠ 0 → x ≠ 0 → a^4 * x^4 + 4 * a^3 * x^2 * (1/x) + 6 * a^2 * (1/x)^2 * x^4 + 4 * a * (1/x)^3 * x^6 + (1/x)^4 * x^8 = (ax + 1/x)^4) → (4 * a^3 = 32) → a = 2 :=
by
  intros H1 H2
  sorry

end find_a_given_coefficient_l124_124563


namespace min_y_value_l124_124566

noncomputable def y (a x : ℝ) : ℝ := (Real.exp x - a)^2 + (Real.exp (-x) - a)^2

theorem min_y_value (a : ℝ) (h : a ≠ 0) : 
  (a ≥ 2 → ∃ x, y a x = a^2 - 2) ∧ (a < 2 → ∃ x, y a x = 2*(a-1)^2) :=
sorry

end min_y_value_l124_124566


namespace sqrt_sum_simplify_l124_124049

theorem sqrt_sum_simplify :
  Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 := 
by
  sorry

end sqrt_sum_simplify_l124_124049


namespace problem_statement_l124_124196

def avg2 (a b : ℚ) : ℚ := (a + b) / 2
def avg3 (a b c : ℚ) : ℚ := (a + b + c) / 3

theorem problem_statement : avg3 (avg3 (-1) 2 3) (avg2 2 3) 1 = 29 / 18 := 
by 
  sorry

end problem_statement_l124_124196


namespace angle_F_calculation_l124_124011

theorem angle_F_calculation (D E F : ℝ) :
  D = 80 ∧ E = 2 * F + 30 ∧ D + E + F = 180 → F = 70 / 3 :=
by
  intro h
  cases' h with hD h_remaining
  cases' h_remaining with hE h_sum
  sorry

end angle_F_calculation_l124_124011


namespace system_of_equations_solution_l124_124471

theorem system_of_equations_solution (x y z : ℝ) :
  (x = 6 + Real.sqrt 29 ∧ y = (5 - 2 * (6 + Real.sqrt 29)) / 3 ∧ z = (4 - (6 + Real.sqrt 29)) / 3 ∧
   x + y + z = 3 ∧ x + 2 * y - z = 2 ∧ x + y * z + z * x = 3) ∨
  (x = 6 - Real.sqrt 29 ∧ y = (5 - 2 * (6 - Real.sqrt 29)) / 3 ∧ z = (4 - (6 - Real.sqrt 29)) / 3 ∧
   x + y + z = 3 ∧ x + 2 * y - z = 2 ∧ x + y * z + z * x = 3) :=
sorry

end system_of_equations_solution_l124_124471


namespace rob_baseball_cards_l124_124334

theorem rob_baseball_cards
  (r j r_d : ℕ)
  (hj : j = 40)
  (h_double : r_d = j / 5)
  (h_cards : r = 3 * r_d) :
  r = 24 :=
by
  sorry

end rob_baseball_cards_l124_124334


namespace solid_has_identical_views_is_sphere_or_cube_l124_124244

-- Define the conditions for orthographic projections being identical
def identical_views_in_orthographic_projections (solid : Type) : Prop :=
  sorry -- Assume the logic for checking identical orthographic projections is defined

-- Define the types for sphere and cube
structure Sphere : Type := 
  (radius : ℝ)

structure Cube : Type := 
  (side_length : ℝ)

-- The main statement to prove
theorem solid_has_identical_views_is_sphere_or_cube (solid : Type) 
  (h : identical_views_in_orthographic_projections solid) : 
  solid = Sphere ∨ solid = Cube :=
by 
  sorry -- The detailed proof is omitted

end solid_has_identical_views_is_sphere_or_cube_l124_124244


namespace prob_two_black_balls_l124_124674

-- Definitions based on conditions
def totalBalls : ℕ := 7 + 8
def initialBlackBalls : ℕ := 8
def ballsLeftAfterOneDraw : ℕ := 14
def blackBallsLeftAfterOneDraw : ℕ := 7

-- Probability of success
def probabilityFirstBlack : ℚ := initialBlackBalls / totalBalls
def probabilitySecondBlack : ℚ := blackBallsLeftAfterOneDraw / ballsLeftAfterOneDraw

-- Combined probability for two black balls without replacement
def combinedProbability : ℚ := probabilityFirstBlack * probabilitySecondBlack

theorem prob_two_black_balls : combinedProbability = 4 / 15 :=
by
  sorry

end prob_two_black_balls_l124_124674


namespace find_ages_l124_124709

-- Definitions of the conditions
def cond1 (D S : ℕ) : Prop := D = 3 * S
def cond2 (D S : ℕ) : Prop := D + 5 = 2 * (S + 5)

-- Theorem statement
theorem find_ages (D S : ℕ) 
  (h1 : cond1 D S) 
  (h2 : cond2 D S) : 
  D = 15 ∧ S = 5 :=
by 
  sorry

end find_ages_l124_124709


namespace geometric_sequence_collinear_vectors_l124_124730

theorem geometric_sequence_collinear_vectors (a : ℕ → ℝ) (q : ℝ)
  (h_geometric : ∀ n, a (n + 1) = q * a n)
  (a2 a3 : ℝ)
  (h_a2 : a 2 = a2)
  (h_a3 : a 3 = a3)
  (h_parallel : 3 * a2 = 2 * a3) :
  (a2 + a 4) / (a3 + a 5) = 2 / 3 := 
by
  sorry

end geometric_sequence_collinear_vectors_l124_124730


namespace tens_digit_of_6_pow_19_l124_124509

def tens_digit (n : ℕ) : ℕ :=
  (n / 10) % 10

theorem tens_digit_of_6_pow_19 : tens_digit (6 ^ 19) = 9 := 
by 
  sorry

end tens_digit_of_6_pow_19_l124_124509


namespace ab_equals_one_l124_124430

theorem ab_equals_one (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (f : ℝ → ℝ) (h3 : f = abs ∘ log) (h4 : f a = f b) : a * b = 1 :=
by
  sorry

end ab_equals_one_l124_124430


namespace not_possible_coloring_possible_coloring_l124_124647

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

end not_possible_coloring_possible_coloring_l124_124647


namespace circumcenter_rational_l124_124623

theorem circumcenter_rational {a1 b1 a2 b2 a3 b3 : ℚ} 
  (h1 : a1 ≠ a2 ∨ b1 ≠ b2) 
  (h2 : a1 ≠ a3 ∨ b1 ≠ b3) 
  (h3 : a2 ≠ a3 ∨ b2 ≠ b3) :
  ∃ (x y : ℚ), 
    (x - a1)^2 + (y - b1)^2 = (x - a2)^2 + (y - b2)^2 ∧
    (x - a1)^2 + (y - b1)^2 = (x - a3)^2 + (y - b3)^2 :=
sorry

end circumcenter_rational_l124_124623


namespace possible_winning_scores_count_l124_124905

def total_runners := 15
def total_score := (total_runners * (total_runners + 1)) / 2

def min_score := 15
def max_potential_score := 39

def is_valid_winning_score (score : ℕ) : Prop :=
  min_score ≤ score ∧ score ≤ max_potential_score

theorem possible_winning_scores_count : 
  ∃ scores : Finset ℕ, ∀ score ∈ scores, is_valid_winning_score score ∧ Finset.card scores = 25 := 
sorry

end possible_winning_scores_count_l124_124905


namespace median_on_hypotenuse_length_l124_124558

theorem median_on_hypotenuse_length
  (a b : ℝ) (h₁ : a = 6) (h₂ : b = 8) (right_triangle : (a ^ 2 + b ^ 2) = c ^ 2) :
  (1 / 2) * c = 5 :=
  sorry

end median_on_hypotenuse_length_l124_124558


namespace students_who_wanted_fruit_l124_124070

theorem students_who_wanted_fruit (red_apples green_apples extra_apples ordered_apples served_apples students_wanted_fruit : ℕ)
    (h1 : red_apples = 43)
    (h2 : green_apples = 32)
    (h3 : extra_apples = 73)
    (h4 : ordered_apples = red_apples + green_apples)
    (h5 : served_apples = ordered_apples + extra_apples)
    (h6 : students_wanted_fruit = served_apples - ordered_apples) :
    students_wanted_fruit = 73 := 
by
    sorry

end students_who_wanted_fruit_l124_124070


namespace simplify_sqrt_sum_l124_124035

theorem simplify_sqrt_sum : (Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2) :=
by
  sorry

end simplify_sqrt_sum_l124_124035


namespace vertical_asymptote_unique_d_values_l124_124541

theorem vertical_asymptote_unique_d_values (d : ℝ) :
  (∃! x : ℝ, ∃ c : ℝ, x ≠ 2 ∧ x ≠ 3 ∧ (x^2 - 2*x + d) = 0) ↔ (d = 0 ∨ d = -3) := 
sorry

end vertical_asymptote_unique_d_values_l124_124541


namespace actual_estate_area_l124_124917

theorem actual_estate_area (map_scale : ℝ) (length_inches : ℝ) (width_inches : ℝ) 
  (actual_length : ℝ) (actual_width : ℝ) (area_square_miles : ℝ) 
  (h_scale : map_scale = 300)
  (h_length : length_inches = 4)
  (h_width : width_inches = 3)
  (h_actual_length : actual_length = length_inches * map_scale)
  (h_actual_width : actual_width = width_inches * map_scale)
  (h_area : area_square_miles = actual_length * actual_width) :
  area_square_miles = 1080000 :=
sorry

end actual_estate_area_l124_124917


namespace ken_situps_l124_124770

variable (K : ℕ)

theorem ken_situps (h1 : Nathan = 2 * K)
                   (h2 : Bob = 3 * K / 2)
                   (h3 : Bob = K + 10) : 
                   K = 20 := 
by
  sorry

end ken_situps_l124_124770


namespace game_terminates_if_n_lt_1994_game_does_not_terminate_if_n_eq_1994_l124_124097

-- Definitions and conditions for the problem
def num_girls : ℕ := 1994
def tokens (n : ℕ) := n

-- Main theorem statements
theorem game_terminates_if_n_lt_1994 (n : ℕ) (h : n < num_girls) :
  ∃ (S : ℕ) (invariant : ℕ) (steps : ℕ), (∀ j : ℕ, 1 ≤ j ∧ j ≤ num_girls → (tokens n % num_girls) ≤ 1) :=
by
  sorry

theorem game_does_not_terminate_if_n_eq_1994 :
  ∃ (S : ℕ) (invariant : ℕ) (steps : ℕ), (tokens 1994 % num_girls = 0) :=
by
  sorry

end game_terminates_if_n_lt_1994_game_does_not_terminate_if_n_eq_1994_l124_124097


namespace fraction_decomposition_l124_124204

noncomputable def p (n : ℕ) : ℚ :=
  (n + 1) / 2

noncomputable def q (n : ℕ) : ℚ :=
  n * p n

theorem fraction_decomposition (n : ℕ) (h : ∃ k : ℕ, n = 5 + 2*k) :
  (2 / n : ℚ) = (1 / p n) + (1 / q n) :=
by
  sorry

end fraction_decomposition_l124_124204


namespace measure_angle_F_correct_l124_124008

noncomputable def measure_angle_D : ℝ := 80
noncomputable def measure_angle_F : ℝ := 70 / 3
noncomputable def measure_angle_E (angle_F : ℝ) : ℝ := 2 * angle_F + 30
noncomputable def angle_sum_property (angle_D angle_E angle_F : ℝ) : Prop :=
  angle_D + angle_E + angle_F = 180

theorem measure_angle_F_correct : measure_angle_F = 70 / 3 :=
by
  let angle_D := measure_angle_D
  let angle_F := measure_angle_F
  have h1 : measure_angle_E angle_F = 2 * angle_F + 30 := rfl
  have h2 : angle_sum_property angle_D (measure_angle_E angle_F) angle_F := sorry
  sorry

end measure_angle_F_correct_l124_124008


namespace even_n_has_parallel_pair_odd_n_cannot_have_exactly_one_parallel_pair_l124_124932

-- Definitions for the conditions in Lean 4
def regular_n_gon (n : ℕ) := true -- Dummy definition; actual geometric properties not needed for statement

def connected_path_visits_each_vertex_once (n : ℕ) := true -- Dummy definition; actual path properties not needed for statement

def parallel_pair (i j p q : ℕ) (n : ℕ) : Prop := (i + j) % n = (p + q) % n

-- Statements for part (a) and (b)

theorem even_n_has_parallel_pair (n : ℕ) (h_even : n % 2 = 0) 
  (h_path : connected_path_visits_each_vertex_once n) : 
  ∃ (i j p q : ℕ), i ≠ p ∧ j ≠ q ∧ parallel_pair i j p q n := 
sorry

theorem odd_n_cannot_have_exactly_one_parallel_pair (n : ℕ) (h_odd : n % 2 = 1) 
  (h_path : connected_path_visits_each_vertex_once n) : 
  ¬∃ (i j p q : ℕ), i ≠ p ∧ j ≠ q ∧ parallel_pair i j p q n ∧ 
  (∀ (i' j' p' q' : ℕ), (i' ≠ p' ∨ j' ≠ q') → ¬parallel_pair i' j' p' q' n) := 
sorry

end even_n_has_parallel_pair_odd_n_cannot_have_exactly_one_parallel_pair_l124_124932


namespace find_fraction_l124_124157

def f (x : ℤ) : ℤ := 3 * x + 4
def g (x : ℤ) : ℤ := 4 * x - 3

theorem find_fraction :
  (f (g (f 2)):ℚ) / (g (f (g 2)):ℚ) = 115 / 73 := by
  sorry

end find_fraction_l124_124157


namespace part1_daily_sales_profit_final_max_daily_sales_profit_l124_124198

-- Conditions from part (a)
def original_selling_price : ℚ := 30
def cost_price : ℚ := 15
def original_sales_volume : ℚ := 60
def sales_increase_per_yuan : ℚ := 10

-- Part (1): Daily sales profit if the price is reduced by 2 yuan
def new_selling_price1 : ℚ := original_selling_price - 2
def new_sales_volume1 : ℚ := original_sales_volume + (2 * sales_increase_per_yuan)
def profit_per_kilogram1 : ℚ := new_selling_price1 - cost_price
def daily_sales_profit1 : ℚ := profit_per_kilogram1 * new_sales_volume1

theorem part1_daily_sales_profit : daily_sales_profit1 = 1040 := by
  sorry

-- Part (2): Maximum daily sales profit and corresponding selling price
def selling_price_at_max_profit : ℚ := 51 / 2

def daily_profit (x : ℚ) : ℚ :=
  (x - cost_price) * (original_sales_volume + (original_selling_price - x) * sales_increase_per_yuan)

theorem final_max_daily_sales_profit :
  (∀ x : ℚ, daily_profit x ≤ daily_profit selling_price_at_max_profit) ∧ daily_profit selling_price_at_max_profit = 1102.5 := by
  sorry

end part1_daily_sales_profit_final_max_daily_sales_profit_l124_124198


namespace number_condition_l124_124556

theorem number_condition (x : ℝ) (h : 45 - 3 * x^2 = 12) : x = Real.sqrt 11 ∨ x = -Real.sqrt 11 :=
sorry

end number_condition_l124_124556


namespace investment_ratio_correct_l124_124943

variable (P Q : ℝ)
variable (investment_ratio: ℝ := 7 / 5)
variable (profit_ratio: ℝ := 7 / 10)
variable (time_p: ℝ := 7)
variable (time_q: ℝ := 14)

theorem investment_ratio_correct :
  (P * time_p) / (Q * time_q) = profit_ratio → (P / Q) = investment_ratio := 
by
  sorry

end investment_ratio_correct_l124_124943


namespace smallest_sum_l124_124146

theorem smallest_sum (x y : ℕ) (hx : x ≠ y) (hxy : (1/x:ℚ) + (1/y:ℚ) = 1/15) : x + y = 64 :=
sorry

end smallest_sum_l124_124146


namespace total_goats_l124_124646

theorem total_goats (W: ℕ) (H_W: W = 180) (H_P: W + 70 = 250) : W + (W + 70) = 430 :=
by
  -- proof goes here
  sorry

end total_goats_l124_124646


namespace simplify_expression_l124_124785

variable (a b c x : ℝ)

def distinct (a b c : ℝ) : Prop := a ≠ b ∧ a ≠ c ∧ b ≠ c

noncomputable def p (x a b c : ℝ) : ℝ :=
  (x - a)^3/(a - b)*(a - c) + a*x +
  (x - b)^3/(b - a)*(b - c) + b*x +
  (x - c)^3/(c - a)*(c - b) + c*x

theorem simplify_expression (h : distinct a b c) :
  p x a b c = a + b + c + 3*x + 1 := by
  sorry

end simplify_expression_l124_124785


namespace smallest_n_l124_124190

theorem smallest_n (n : ℕ) (h1 : n % 6 = 4) (h2 : n % 7 = 3) (h3 : 15 < n) : n = 52 :=
by
  sorry

end smallest_n_l124_124190


namespace total_students_in_class_l124_124476

theorem total_students_in_class 
  (avg_age_all : ℝ)
  (num_students1 : ℕ) (avg_age1 : ℝ)
  (num_students2 : ℕ) (avg_age2 : ℝ)
  (age_student17 : ℕ)
  (total_students : ℕ) :
  avg_age_all = 17 →
  num_students1 = 5 →
  avg_age1 = 14 →
  num_students2 = 9 →
  avg_age2 = 16 →
  age_student17 = 75 →
  total_students = num_students1 + num_students2 + 1 →
  total_students = 17 :=
by
  intro h_avg_all h_num1 h_avg1 h_num2 h_avg2 h_age17 h_total
  -- Additional proof steps would go here
  sorry

end total_students_in_class_l124_124476


namespace sum_even_integers_12_to_40_l124_124504

theorem sum_even_integers_12_to_40 : 
  ∑ k in finset.filter (λ k, even k) (finset.range 41), k = 390 := by
  sorry

end sum_even_integers_12_to_40_l124_124504


namespace negation_of_universal_proposition_l124_124941

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 0 → x^2 + x ≥ 0) ↔ ∃ x : ℝ, x > 0 ∧ x^2 + x < 0 :=
by sorry

end negation_of_universal_proposition_l124_124941


namespace blue_hat_cost_l124_124218

variable (B : ℕ)
variable (totalHats : ℕ := 85)
variable (greenHatCost : ℕ := 7)
variable (greenHatsBought : ℕ := 38)
variable (totalCost : ℕ := 548)

theorem blue_hat_cost 
(h1 : greenHatsBought = 38) 
(h2 : totalHats = 85) 
(h3 : greenHatCost = 7)
(h4 : totalCost = 548) :
  let totalGreenHatCost := greenHatCost * greenHatsBought
  let totalBlueHatCost := totalCost - totalGreenHatCost
  let totalBlueHatsBought := totalHats - greenHatsBought
  B = totalBlueHatCost / totalBlueHatsBought := by
  sorry

end blue_hat_cost_l124_124218


namespace verify_a_l124_124536

def g (x : ℝ) : ℝ := 5 * x - 7

theorem verify_a (a : ℝ) : g a = 0 ↔ a = 7 / 5 := by
  sorry

end verify_a_l124_124536


namespace find_all_a_l124_124272

def digit_sum_base_4038 (n : ℕ) : ℕ :=
  n.digits 4038 |>.sum

def is_good (n : ℕ) : Prop :=
  2019 ∣ digit_sum_base_4038 n

def is_bad (n : ℕ) : Prop :=
  ¬ is_good n

def satisfies_condition (seq : ℕ → ℕ) (a : ℝ) : Prop :=
  (∀ n, seq n ≤ a * n) ∧ ∀ n, seq n = seq (n + 1) + 1

theorem find_all_a (a : ℝ) (h1 : 1 ≤ a) :
  (∀ seq, (∀ n m, n ≠ m → seq n ≠ seq m) → satisfies_condition seq a →
    ∃ n_infinitely, is_bad (seq n_infinitely)) ↔ a < 2019 := sorry

end find_all_a_l124_124272


namespace exists_abcd_for_n_gt_one_l124_124188

theorem exists_abcd_for_n_gt_one (n : Nat) (h : n > 1) :
  ∃ a b c d : Nat, a + b = 4 * n ∧ c + d = 4 * n ∧ a * b - c * d = 4 * n := 
by
  sorry

end exists_abcd_for_n_gt_one_l124_124188


namespace keira_guarantees_capture_l124_124601

theorem keira_guarantees_capture (k : ℕ) (n : ℕ) (h_k_pos : 0 < k) (h_n_cond : n > k / 2023) :
    k ≥ 1012 :=
sorry

end keira_guarantees_capture_l124_124601


namespace f_2020_eq_neg_1_l124_124895

noncomputable def f: ℝ → ℝ :=
sorry

axiom f_2_x_eq_neg_f_x : ∀ x: ℝ, f (2 - x) = -f x
axiom f_x_minus_2_eq_f_neg_x : ∀ x: ℝ, f (x - 2) = f (-x)
axiom f_specific : ∀ x : ℝ, -1 < x ∧ x < 1 -> f x = x^2 + 1

theorem f_2020_eq_neg_1 : f 2020 = -1 :=
sorry

end f_2020_eq_neg_1_l124_124895


namespace k_divides_99_l124_124237

-- Define what it means for a number to reverse its digits
def reverse_digits (n : ℕ) : ℕ :=
  n.digits.reverse.foldl (λ acc d, acc * 10 + d) 0

-- The main statement to prove:
theorem k_divides_99 (k : ℕ) (h : ∀ n : ℕ, k ∣ n → k ∣ reverse_digits n) : k ∣ 99 :=
by
  sorry

end k_divides_99_l124_124237


namespace selection_methods_with_both_genders_l124_124412

noncomputable def selection_methods_count : ℕ :=
  (finset.card (finset.Ico (0 : ℕ) 10 : finset ℕ)).choose 5 -
  (finset.card (finset.Ico (0 : ℕ) 7 : finset ℕ)).choose 5

theorem selection_methods_with_both_genders :
  selection_methods_count = 120 :=
  by sorry

end selection_methods_with_both_genders_l124_124412


namespace solution_to_system_of_eqns_l124_124548

theorem solution_to_system_of_eqns (x y z : ℝ) :
  (x = (2 * z ^ 2) / (1 + z ^ 2) ∧ y = (2 * x ^ 2) / (1 + x ^ 2) ∧ z = (2 * y ^ 2) / (1 + y ^ 2)) →
  (x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1 ∧ y = 1 ∧ z = 1) :=
by sorry

end solution_to_system_of_eqns_l124_124548


namespace blue_string_length_l124_124345

def length_red := 8
def length_white := 5 * length_red
def length_blue := length_white / 8

theorem blue_string_length : length_blue = 5 := by
  sorry

end blue_string_length_l124_124345


namespace amount_of_rice_distributed_in_first_5_days_l124_124969

-- Definitions from conditions
def workers_day (d : ℕ) : ℕ := if d = 1 then 64 else 64 + 7 * (d - 1)

-- The amount of rice each worker receives per day
def rice_per_worker : ℕ := 3

-- Total workers dispatched in the first 5 days
def total_workers_first_5_days : ℕ := (workers_day 1 + workers_day 2 + workers_day 3 + workers_day 4 + workers_day 5)

-- Given these definitions, we now state the theorem to prove
theorem amount_of_rice_distributed_in_first_5_days : total_workers_first_5_days * rice_per_worker = 1170 :=
by
  sorry

end amount_of_rice_distributed_in_first_5_days_l124_124969


namespace maximize_sector_area_l124_124301

theorem maximize_sector_area :
  (∀ (r l : ℝ), 2 * r + l = 36 ∧ S = 1 / 2 * l * r ∧ α = l / r → α = 2) :=
by
  intros r l h
  cases h with h1 h_rest
  cases h_rest with h2 h3
  sorry

end maximize_sector_area_l124_124301


namespace skate_time_correct_l124_124245

noncomputable def skate_time (path_length miles_length : ℝ) (skating_speed : ℝ) : ℝ :=
  let time_taken := (1.58 * Real.pi) / skating_speed
  time_taken

theorem skate_time_correct :
  skate_time 1 1 4 = 1.58 * Real.pi / 4 :=
by
  sorry

end skate_time_correct_l124_124245


namespace sqrt_sum_simplify_l124_124051

theorem sqrt_sum_simplify :
  Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 := 
by
  sorry

end sqrt_sum_simplify_l124_124051


namespace remainder_17_pow_49_mod_5_l124_124654

theorem remainder_17_pow_49_mod_5 : (17^49) % 5 = 2 :=
by
  sorry

end remainder_17_pow_49_mod_5_l124_124654


namespace calculate_ab_l124_124073

theorem calculate_ab {a b c : ℝ} (hc : c ≠ 0) (h1 : (a * b) / c = 4) (h2 : a * (b / c) = 12) : a * b = 12 :=
by
  sorry

end calculate_ab_l124_124073


namespace highest_qualification_number_possible_l124_124794

theorem highest_qualification_number_possible (n : ℕ) (qualifies : ℕ → ℕ → Prop)
    (h512 : n = 512)
    (hqualifies : ∀ a b, qualifies a b ↔ (a < b ∧ b - a ≤ 2)): 
    ∃ k, k = 18 ∧ (∀ m, qualifies m k → m < k) :=
by
  sorry

end highest_qualification_number_possible_l124_124794


namespace predict_height_at_age_10_l124_124101

def regression_line := fun (x : ℝ) => 7.19 * x + 73.93

theorem predict_height_at_age_10 :
  regression_line 10 = 145.83 :=
by
  sorry

end predict_height_at_age_10_l124_124101


namespace prob_first_question_correct_is_4_5_distribution_of_X_l124_124233

-- Assume probabilities for member A and member B answering correctly.
def prob_A_correct : ℚ := 2 / 5
def prob_B_correct : ℚ := 2 / 3

def prob_A_incorrect : ℚ := 1 - prob_A_correct
def prob_B_incorrect : ℚ := 1 - prob_B_correct

-- Given that A answers first, followed by B.
-- Calculate the probability that the first team answers the first question correctly.
def prob_first_question_correct : ℚ :=
  prob_A_correct + (prob_A_incorrect * prob_B_correct)

-- Assert that the calculated probability is equal to 4/5
theorem prob_first_question_correct_is_4_5 :
  prob_first_question_correct = 4 / 5 := by
  sorry

-- Define the possible scores and their probabilities
def prob_X_eq_0 : ℚ := prob_A_incorrect * prob_B_incorrect
def prob_X_eq_10 : ℚ := (prob_A_correct + prob_A_incorrect * prob_B_correct) * prob_A_incorrect * prob_B_incorrect
def prob_X_eq_20 : ℚ := (prob_A_correct + prob_A_incorrect * prob_B_correct) ^ 2 * prob_A_incorrect * prob_B_incorrect
def prob_X_eq_30 : ℚ := (prob_A_correct + prob_A_incorrect * prob_B_correct) ^ 3

-- Assert the distribution probabilities for the random variable X
theorem distribution_of_X :
  prob_X_eq_0 = 1 / 5 ∧
  prob_X_eq_10 = 4 / 25 ∧
  prob_X_eq_20 = 16 / 125 ∧
  prob_X_eq_30 = 64 / 125 := by
  sorry

end prob_first_question_correct_is_4_5_distribution_of_X_l124_124233


namespace minimum_common_perimeter_exists_l124_124491

noncomputable def find_minimum_perimeter
  (a b x : ℕ) 
  (is_int_sided_triangle_1 : 2 * a + 20 * x = 2 * b + 25 * x)
  (is_int_sided_triangle_2 : 20 * (sqrt (a^2 - 100 * x^2)) = 25 * (sqrt (b^2 - 156.25 * x^2))) 
  (base_ratio : 20 * 2 * (a - b) = 25 * 2 * (a - b)): ℕ :=
2 * a + 20 * (2 * (a - b))

-- The final goal should prove the minimum perimeter under the given conditions.
theorem minimum_common_perimeter_exists :
∃ (minimum_perimeter : ℕ), 
  (∀ (a b x : ℕ), 
    2 * a + 20 * x = 2 * b + 25 * x → 
    20 * (sqrt (a^2 - 100 * x^2)) = 25 * (sqrt (b^2 - 156.25 * x^2)) → 
    20 * 2 * (a - b) = 25 * 2 * (a - b) → 
    minimum_perimeter = 2 * a + 20 * x) :=
sorry

end minimum_common_perimeter_exists_l124_124491


namespace finite_perfect_squares_l124_124772

noncomputable def finite_squares (a b : ℕ) : Prop :=
  ∃ (f : Finset ℕ), ∀ n, n ∈ f ↔ 
    ∃ (x y : ℕ), a * n ^ 2 + b = x ^ 2 ∧ a * (n + 1) ^ 2 + b = y ^ 2

theorem finite_perfect_squares (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  finite_squares a b :=
sorry

end finite_perfect_squares_l124_124772


namespace no_real_roots_l124_124998

noncomputable def polynomial (p : ℝ) (x : ℝ) : ℝ :=
  x^4 + 4 * p * x^3 + 6 * x^2 + 4 * p * x + 1

theorem no_real_roots (p : ℝ) :
  (p > -Real.sqrt 5 / 2) ∧ (p < Real.sqrt 5 / 2) ↔ ¬(∃ x : ℝ, polynomial p x = 0) := by
  sorry

end no_real_roots_l124_124998


namespace triangle_right_triangle_l124_124074

-- Defining the sides of the triangle
variables (a b c : ℝ)

-- Theorem statement
theorem triangle_right_triangle (h : (a + b)^2 = c^2 + 2 * a * b) : a^2 + b^2 = c^2 :=
by {
  sorry
}

end triangle_right_triangle_l124_124074


namespace inequality_S_l124_124597

def S (n m : ℕ) : ℕ := sorry

theorem inequality_S (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  S (2015 * n) n * S (2015 * m) m ≥ S (2015 * n) m * S (2015 * m) n :=
sorry

end inequality_S_l124_124597


namespace smallest_x_y_sum_l124_124140

theorem smallest_x_y_sum :
  ∃ x y : ℕ,
    0 < x ∧ 0 < y ∧ x ≠ y ∧ (1 / (x : ℝ) + 1 / (y : ℝ) = 1 / 15) ∧ (x + y = 64) := 
by
  sorry

end smallest_x_y_sum_l124_124140


namespace radical_conjugate_sum_l124_124396

theorem radical_conjugate_sum:
  let a := 15 - Real.sqrt 500
  let b := 15 + Real.sqrt 500
  3 * (a + b) = 90 :=
by
  sorry

end radical_conjugate_sum_l124_124396


namespace smallest_positive_integer_l124_124359

theorem smallest_positive_integer (n : ℕ) (h : 629 * n ≡ 1181 * n [MOD 35]) : n = 35 :=
sorry

end smallest_positive_integer_l124_124359


namespace add_fractions_l124_124959

theorem add_fractions : (1 / 6 : ℚ) + (5 / 12) = 7 / 12 := 
by
  sorry

end add_fractions_l124_124959


namespace find_p_q_l124_124721

noncomputable def f (p q : ℝ) (x : ℝ) : ℝ :=
if x < -1 then p * x + q else 5 * x - 10

theorem find_p_q (p q : ℝ) (h : ∀ x, f p q (f p q x) = x) : p + q = 11 :=
sorry

end find_p_q_l124_124721


namespace possible_values_a_l124_124756

def A : Set ℝ := {-1, 2}
def B (a : ℝ) : Set ℝ := {x | a * x^2 = 2 ∧ a ≥ 0}

def whale_swallowing (S T : Set ℝ) : Prop :=
S ⊆ T ∨ T ⊆ S

def moth_eating (S T : Set ℝ) : Prop :=
(∃ x, x ∈ S ∧ x ∈ T) ∧ ¬(S ⊆ T) ∧ ¬(T ⊆ S)

def valid_a (a : ℝ) : Prop :=
whale_swallowing A (B a) ∨ moth_eating A (B a)

theorem possible_values_a :
  {a : ℝ | valid_a a} = {0, 1/2, 2} :=
sorry

end possible_values_a_l124_124756


namespace wrapping_paper_needs_l124_124873

theorem wrapping_paper_needs :
  let first_present := 2
  let second_present := (3 / 4) * first_present
  let third_present := first_present + second_present
  first_present + second_present + third_present = 7 := by
  let first_present := 2
  let second_present := (3 / 4) * first_present
  let third_present := first_present + second_present
  sorry

end wrapping_paper_needs_l124_124873


namespace sports_club_total_members_l124_124440

theorem sports_club_total_members :
  ∀ (B T Both Neither Total : ℕ),
    B = 17 → T = 19 → Both = 10 → Neither = 2 → Total = B + T - Both + Neither → Total = 28 :=
by
  intros B T Both Neither Total hB hT hBoth hNeither hTotal
  rw [hB, hT, hBoth, hNeither] at hTotal
  exact hTotal

end sports_club_total_members_l124_124440


namespace closest_point_on_parabola_to_line_is_l124_124206

-- Definitions of the parabola and the line
def parabola (x : ℝ) : ℝ := 4 * x^2
def line (x : ℝ) : ℝ := 4 * x - 5

-- Prove that the point on the parabola that is closest to the line is (1/2, 1)
theorem closest_point_on_parabola_to_line_is (x y : ℝ) :
  parabola x = y ∧ (∀ (x' y' : ℝ), parabola x' = y' -> (line x - y)^2 >= (line x' - y')^2) ->
  (x, y) = (1/2, 1) :=
by
  sorry

end closest_point_on_parabola_to_line_is_l124_124206


namespace distinct_real_roots_a1_l124_124409

theorem distinct_real_roots_a1 {x : ℝ} :
  ∀ a : ℝ, a = 1 →
    ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (a * x1^2 + (1 - a) * x1 - 1 = 0) ∧ (a * x2^2 + (1 - a) * x2 - 1 = 0) :=
by sorry

end distinct_real_roots_a1_l124_124409


namespace no_prime_sum_seventeen_l124_124292

def is_prime (n : ℕ) : Prop := n ≥ 2 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem no_prime_sum_seventeen :
  ¬ ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p + q = 17 := by
  sorry

end no_prime_sum_seventeen_l124_124292


namespace total_time_correct_l124_124681

variable (b n : ℕ)

def total_travel_time (b n : ℕ) : ℚ := (3*b + 4*n + 2*b) / 150

theorem total_time_correct :
  total_travel_time b n = (5 * b + 4 * n) / 150 :=
by sorry

end total_time_correct_l124_124681


namespace line_passes_through_center_l124_124288

-- Define the equation of the circle as given in the problem.
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 6*y + 8 = 0

-- Define the center of the circle.
def center_of_circle (x y : ℝ) : Prop := x = 1 ∧ y = -3

-- Define the equation of the line.
def line_equation (x y : ℝ) : Prop := 2*x + y + 1 = 0

-- The theorem to prove.
theorem line_passes_through_center :
  (∃ x y, circle_equation x y ∧ center_of_circle x y) →
  (∃ x y, center_of_circle x y ∧ line_equation x y) :=
by
  sorry

end line_passes_through_center_l124_124288


namespace exist_non_negative_product_l124_124028

theorem exist_non_negative_product (a1 a2 a3 a4 a5 a6 a7 a8 : ℝ) :
  0 ≤ a1 * a3 + a2 * a4 ∨
  0 ≤ a1 * a5 + a2 * a6 ∨
  0 ≤ a1 * a7 + a2 * a8 ∨
  0 ≤ a3 * a5 + a4 * a6 ∨
  0 ≤ a3 * a7 + a4 * a8 ∨
  0 ≤ a5 * a7 + a6 * a8 :=
sorry

end exist_non_negative_product_l124_124028


namespace simplify_sqrt_sum_l124_124053

theorem simplify_sqrt_sum : sqrt 72 + sqrt 32 = 10 * sqrt 2 :=
by
  sorry

end simplify_sqrt_sum_l124_124053


namespace domain_f_monotonicity_f_inequality_solution_l124_124152

noncomputable def f (x: ℝ) := Real.log ((1 - x) / (1 + x))

variable {x : ℝ}

theorem domain_f : ∀ x, x ∈ Set.Ioo (-1 : ℝ) 1 -> Set.Ioo (-1 : ℝ) 1 := sorry

theorem monotonicity_f : ∀ x ∈ Set.Ioo (-1 : ℝ) 1, ∀ y ∈ Set.Ioo (-1 : ℝ) 1, x < y → f y < f x := sorry

theorem inequality_solution :
  {x : ℝ | f (2 * x - 1) < 0} = {x | x > 1 / 2 ∧ x < 1} := sorry

end domain_f_monotonicity_f_inequality_solution_l124_124152


namespace infinity_non_almost_square_l124_124241

def is_almost_square (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ a ≤ b ∧ b ≤ 101 * a / 100 ∧ n = a * b

theorem infinity_non_almost_square :
  ∃ᶠ m in at_top, ∀ k : ℕ, k ≤ 198 → ¬ is_almost_square (m + k) :=
by sorry

end infinity_non_almost_square_l124_124241


namespace P_at_2018_l124_124181

open Polynomial

noncomputable def P : Polynomial ℤ :=
-- This represents a monic polynomial of degree 2017 satisfying the given conditions.
sorry

theorem P_at_2018 :
  ∀ (P : Polynomial ℤ), 
  (∀ n, 1 ≤ n ∧ n ≤ 2017 → P.eval n = n) ∧
  (P.degree = 2017) → 
  P.eval 2018 = nat.factorial 2017 + 2018 :=
begin
  -- Since the proof is complex, place a placeholder for the actual proof here.
  sorry
end

end P_at_2018_l124_124181


namespace probability_two_points_one_unit_apart_l124_124216

def twelve_points_probability : ℚ := 2 / 11

/-- Twelve points are spaced around at intervals of one unit around a \(3 \times 3\) square.
    Two of the 12 points are chosen at random.
    Prove that the probability that the two points are one unit apart is \(\frac{2}{11}\). -/
theorem probability_two_points_one_unit_apart :
  let total_points := 12
  let total_combinations := (total_points * (total_points - 1)) / 2
  let favorable_pairs := 12
  (favorable_pairs : ℚ) / total_combinations = twelve_points_probability := by
  sorry

end probability_two_points_one_unit_apart_l124_124216


namespace bisection_method_second_step_l124_124353

noncomputable def f (x : ℝ) : ℝ := x^3 + 2 * x - 1

theorem bisection_method_second_step :
  (f 0 < 0) ∧ (f 0.5 > 0) ∧ (continuous_on f (set.Icc 0 0.5)) →
  (∃ x ∈ (set.Icc 0 0.5), f x = 0) ∧ (f 0.25 ∈ f '' (set.Icc 0 0.5)) :=
by
  sorry

end bisection_method_second_step_l124_124353


namespace graph_transformation_point_l124_124897

theorem graph_transformation_point {f : ℝ → ℝ} (h : f 1 = 0) : f (0 + 1) + 1 = 1 :=
by
  sorry

end graph_transformation_point_l124_124897


namespace weight_of_substance_l124_124950

variable (k W1 W2 : ℝ)

theorem weight_of_substance (h1 : ∃ (k : ℝ), ∀ (V W : ℝ), V = k * W)
  (h2 : 48 = k * W1) (h3 : 36 = k * 84) : 
  (∃ (W2 : ℝ), 48 = (36 / 84) * W2) → W2 = 112 := 
by
  sorry

end weight_of_substance_l124_124950


namespace joe_used_225_gallons_l124_124094

def initial_paint : ℕ := 360

def paint_first_week (initial : ℕ) : ℕ := initial / 4

def remaining_paint_after_first_week (initial : ℕ) : ℕ :=
  initial - paint_first_week initial

def paint_second_week (remaining : ℕ) : ℕ := remaining / 2

def total_paint_used (initial : ℕ) : ℕ :=
  paint_first_week initial + paint_second_week (remaining_paint_after_first_week initial)

theorem joe_used_225_gallons :
  total_paint_used initial_paint = 225 :=
by
  sorry

end joe_used_225_gallons_l124_124094


namespace largest_angle_between_a_and_c_l124_124176

variables (a b c : ℝ^3)

-- Defining the conditions given in the problem
def norm_a : ∥a∥ = 2 := sorry
def norm_b : ∥b∥ = 2 := sorry
def norm_c : ∥c∥ = 3 := sorry
def vector_triple_product (a b c : ℝ^3) : a × (b × c) + 2 • b = 0 := sorry

-- Statement of the proof
theorem largest_angle_between_a_and_c (a b c : ℝ^3)
  (h1 : ∥a∥ = 2)
  (h2 : ∥b∥ = 2)
  (h3 : ∥c∥ = 3)
  (h4 : a × (b × c) + 2 • b = 0) : 
  ∃ θ : ℝ, real.arccos (-1/3) = θ ∧ θ ≈ 109.47 := 
by sorry

end largest_angle_between_a_and_c_l124_124176


namespace train_crossing_time_l124_124690

theorem train_crossing_time
  (length_train : ℝ) (length_bridge : ℝ) (speed_kmh : ℝ)
  (train_length_eq : length_train = 720)
  (bridge_length_eq : length_bridge = 320)
  (speed_eq : speed_kmh = 90) :
  (length_train + length_bridge) / (speed_kmh * (1000 / 3600)) = 41.6 := by
  sorry

end train_crossing_time_l124_124690


namespace rational_coordinates_of_circumcenter_l124_124617

open Classical

noncomputable theory

theorem rational_coordinates_of_circumcenter
  {a1 b1 a2 b2 a3 b3 : ℚ}
  (h1 : ∃ (x y : ℚ), (x - a1)^2 + (y - b1)^2 = (x - a2)^2 + (y - b2)^2 ∧
                      (x - a1)^2 + (y - b1)^2 = (x - a3)^2 + (y - b3)^2) :
  ∃ (x y : ℚ),
    (x - a1)^2 + (y - b1)^2 = (x - a2)^2 + (y - b2)^2 ∧
    (x - a1)^2 + (y - b1)^2 = (x - a3)^2 + (y - b3)^2 := 
begin
  obtain ⟨x, y, hx⟩ := h1,
  use [x, y],
  exact hx,
end

end rational_coordinates_of_circumcenter_l124_124617


namespace alice_bob_sum_is_42_l124_124930

theorem alice_bob_sum_is_42 :
  ∃ (A B : ℕ), 
    (1 ≤ A ∧ A ≤ 60) ∧ 
    (1 ≤ B ∧ B ≤ 60) ∧ 
    Nat.Prime B ∧ B > 10 ∧ 
    (∀ n : ℕ, n < 5 → (A + B) % n ≠ 0) ∧ 
    ∃ k : ℕ, 150 * B + A = k * k ∧ 
    A + B = 42 :=
by 
  sorry

end alice_bob_sum_is_42_l124_124930


namespace Sharmila_hourly_wage_l124_124031

def Sharmila_hours_per_day (day : String) : ℕ :=
  if day = "Monday" ∨ day = "Wednesday" ∨ day = "Friday" then 10
  else if day = "Tuesday" ∨ day = "Thursday" then 8
  else 0

def weekly_total_hours : ℕ :=
  Sharmila_hours_per_day "Monday" + Sharmila_hours_per_day "Tuesday" +
  Sharmila_hours_per_day "Wednesday" + Sharmila_hours_per_day "Thursday" +
  Sharmila_hours_per_day "Friday"

def weekly_earnings : ℤ := 460

def hourly_wage : ℚ :=
  weekly_earnings / weekly_total_hours

theorem Sharmila_hourly_wage :
  hourly_wage = (10 : ℚ) :=
by
  -- proof skipped
  sorry

end Sharmila_hourly_wage_l124_124031


namespace cos_C_in_triangle_l124_124007

theorem cos_C_in_triangle (A B C : ℝ)
  (hA : 0 < A ∧ A < π)
  (hB : 0 < B ∧ B < π)
  (hC : 0 < C ∧ C < π)
  (h_sum : A + B + C = π)
  (h_cos_A : Real.cos A = 3/5)
  (h_sin_B : Real.sin B = 12/13) :
  Real.cos C = 63/65 ∨ Real.cos C = 33/65 :=
sorry

end cos_C_in_triangle_l124_124007


namespace measure_angle_F_correct_l124_124009

noncomputable def measure_angle_D : ℝ := 80
noncomputable def measure_angle_F : ℝ := 70 / 3
noncomputable def measure_angle_E (angle_F : ℝ) : ℝ := 2 * angle_F + 30
noncomputable def angle_sum_property (angle_D angle_E angle_F : ℝ) : Prop :=
  angle_D + angle_E + angle_F = 180

theorem measure_angle_F_correct : measure_angle_F = 70 / 3 :=
by
  let angle_D := measure_angle_D
  let angle_F := measure_angle_F
  have h1 : measure_angle_E angle_F = 2 * angle_F + 30 := rfl
  have h2 : angle_sum_property angle_D (measure_angle_E angle_F) angle_F := sorry
  sorry

end measure_angle_F_correct_l124_124009


namespace box_surface_area_l124_124675

theorem box_surface_area (a b c : ℕ) (h1 : a * b * c = 280) (h2 : a < 10) (h3 : b < 10) (h4 : c < 10) : 
  2 * (a * b + b * c + c * a) = 262 :=
sorry

end box_surface_area_l124_124675


namespace sins_prayers_l124_124846

structure Sins :=
  (pride : Nat)
  (slander : Nat)
  (laziness : Nat)
  (adultery : Nat)
  (gluttony : Nat)
  (self_love : Nat)
  (jealousy : Nat)
  (malicious_gossip : Nat)

def prayer_requirements (s : Sins) : Nat × Nat × Nat :=
  ( s.pride + 2 * s.laziness + 10 * s.adultery + s.gluttony,
    2 * s.pride + 2 * s.slander + 10 * s.adultery + 3 * s.self_love + 3 * s.jealousy + 7 * s.malicious_gossip,
    7 * s.slander + 10 * s.adultery + s.self_love + 2 * s.malicious_gossip )

theorem sins_prayers (sins : Sins) :
  sins.pride = 0 ∧
  sins.slander = 1 ∧
  sins.laziness = 0 ∧
  sins.adultery = 0 ∧
  sins.gluttony = 9 ∧
  sins.self_love = 1 ∧
  sins.jealousy = 0 ∧
  sins.malicious_gossip = 2 ∧
  (sins.pride + sins.slander + sins.laziness + sins.adultery + sins.gluttony + sins.self_love + sins.jealousy + sins.malicious_gossip = 12) ∧
  prayer_requirements sins = (9, 12, 10) :=
  by
  sorry

end sins_prayers_l124_124846


namespace closed_path_has_even_length_l124_124456

   theorem closed_path_has_even_length 
     (u d r l : ℤ) 
     (hu : u = d) 
     (hr : r = l) : 
     ∃ k : ℤ, 2 * (u + r) = 2 * k :=
   by
     sorry
   
end closed_path_has_even_length_l124_124456


namespace polar_eq_circle_l124_124632

-- Definition of the problem condition in polar coordinates
def polar_eq (ρ : ℝ) : Prop := ρ = 1

-- Definition of the assertion we want to prove: that it represents a circle
def represents_circle (ρ : ℝ) (θ : ℝ) : Prop := (ρ = 1) → ∃ (x y : ℝ), (ρ = 1) ∧ (x^2 + y^2 = 1)

theorem polar_eq_circle : ∀ (ρ θ : ℝ), polar_eq ρ → represents_circle ρ θ :=
by
  intros ρ θ hρ hs
  sorry

end polar_eq_circle_l124_124632


namespace simplify_fraction_multiplication_l124_124338

theorem simplify_fraction_multiplication :
  8 * (15 / 4) * (-40 / 45) = -64 / 9 :=
by
  sorry

end simplify_fraction_multiplication_l124_124338


namespace percentage_y_less_than_x_l124_124847

variable (x y : ℝ)

-- given condition
axiom hyp : x = 11 * y

-- proof problem: Prove that the percentage y is less than x is (10/11) * 100
theorem percentage_y_less_than_x (x y : ℝ) (hyp : x = 11 * y) : 
  (x - y) / x * 100 = (10 / 11) * 100 :=
by
  sorry

end percentage_y_less_than_x_l124_124847


namespace weight_of_replaced_person_is_correct_l124_124759

-- Define a constant representing the number of persons in the group.
def num_people : ℕ := 10
-- Define a constant representing the weight of the new person.
def new_person_weight : ℝ := 110
-- Define a constant representing the increase in average weight when the new person joins.
def avg_weight_increase : ℝ := 5
-- Define the weight of the person who was replaced.
noncomputable def replaced_person_weight : ℝ :=
  new_person_weight - num_people * avg_weight_increase

-- Prove that the weight of the replaced person is 60 kg.
theorem weight_of_replaced_person_is_correct : replaced_person_weight = 60 :=
by
  -- Skip the detailed proof steps.
  sorry

end weight_of_replaced_person_is_correct_l124_124759


namespace g_self_inverse_if_one_l124_124530

variables (f : ℝ → ℝ) (symm_about : ∀ x, f (f x) = x - 1)

def g (b : ℝ) (x : ℝ) : ℝ := f (x + b)

theorem g_self_inverse_if_one (b : ℝ) :
  (∀ x, g f b (g f b x) = x) ↔ b = 1 := 
by
  sorry

end g_self_inverse_if_one_l124_124530


namespace trigonometric_identity_cos_58_cos_13_plus_sin_58_sin_13_l124_124193

theorem trigonometric_identity_cos_58_cos_13_plus_sin_58_sin_13 :
  (Real.cos (58 * Real.pi / 180) * Real.cos (13 * Real.pi / 180) +
   Real.sin (58 * Real.pi / 180) * Real.sin (13 * Real.pi / 180) =
   Real.cos (45 * Real.pi / 180)) :=
sorry

end trigonometric_identity_cos_58_cos_13_plus_sin_58_sin_13_l124_124193


namespace parents_gave_money_l124_124874

def money_before_birthday : ℕ := 159
def money_from_grandmother : ℕ := 25
def money_from_aunt_uncle : ℕ := 20
def total_money_after_birthday : ℕ := 279

theorem parents_gave_money :
  total_money_after_birthday = money_before_birthday + money_from_grandmother + money_from_aunt_uncle + 75 :=
by
  sorry

end parents_gave_money_l124_124874


namespace subsets_containing_six_l124_124743

theorem subsets_containing_six : 
  let S := {1, 2, 3, 4, 5, 6}
  in set.count (λ s, 6 ∈ s ∧ s ⊆ S) = 32 :=
sorry

end subsets_containing_six_l124_124743


namespace toby_money_share_l124_124078

theorem toby_money_share (initial_money : ℕ) (fraction : ℚ) (brothers : ℕ) (money_per_brother : ℚ)
  (total_shared : ℕ) (remaining_money : ℕ) :
  initial_money = 343 →
  fraction = 1/7 →
  brothers = 2 →
  money_per_brother = fraction * initial_money →
  total_shared = brothers * money_per_brother →
  remaining_money = initial_money - total_shared →
  remaining_money = 245 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end toby_money_share_l124_124078


namespace friend_reading_time_l124_124023

def my_reading_time : ℕ := 120  -- It takes me 120 minutes to read the novella

def speed_ratio : ℕ := 3  -- My friend reads three times as fast as I do

theorem friend_reading_time : my_reading_time / speed_ratio = 40 := by
  -- Proof
  sorry

end friend_reading_time_l124_124023


namespace unique_five_digit_integers_l124_124156

-- Define the problem conditions
def digits := [2, 2, 3, 9, 9]
def total_spots := 5
def factorial (n : Nat) : Nat := if n = 0 then 1 else n * factorial (n - 1)

-- Compute the number of five-digit integers that can be formed
noncomputable def num_unique_permutations : Nat :=
  factorial total_spots / (factorial 2 * factorial 1 * factorial 2)

-- Proof statement
theorem unique_five_digit_integers : num_unique_permutations = 30 := by
  sorry

end unique_five_digit_integers_l124_124156


namespace smallest_sum_l124_124148

theorem smallest_sum (x y : ℕ) (hx : x ≠ y) (hxy : (1/x:ℚ) + (1/y:ℚ) = 1/15) : x + y = 64 :=
sorry

end smallest_sum_l124_124148


namespace chess_tournament_proof_l124_124587

-- Define the conditions
variables (i g n I G : ℕ)
variables (VI VG VD : ℕ)

-- Condition 1: The number of GMs is ten times the number of IMs
def condition1 : Prop := g = 10 * i
  
-- Condition 2: The sum of the points of all GMs is 4.5 times the sum of the points of all IMs
def condition2 : Prop := G = 5 * I + I / 2

-- Condition 3: The total number of players is the sum of IMs and GMs
def condition3 : Prop := n = i + g

-- Condition 4: Each player played only once against all other opponents
def condition4 : Prop := n * (n - 1) = 2 * (VI + VG + VD)

-- Condition 5: The sum of the points of all games is 5.5 times the sum of the points of all IMs
def condition5 : Prop := I + G = 11 * I / 2

-- Condition 6: Total games played
def total_games (n : ℕ) : ℕ := n * (n - 1) / 2

-- The questions to be proven given the conditions
theorem chess_tournament_proof:
  condition1 i g →
  condition2 I G →
  condition3 i g n →
  condition4 n VI VG VD →
  condition5 I G →
  i = 1 ∧ g = 10 ∧ total_games n = 55 :=
by
  -- The proof is left as an exercise
  sorry

end chess_tournament_proof_l124_124587


namespace equation_of_circle_C_equation_of_line_l_l124_124894

-- Condition: The center of the circle lies on the line y = x + 1.
def center_on_line (a b : ℝ) : Prop :=
  b = a + 1

-- Condition: The circle is tangent to the x-axis.
def tangent_to_x_axis (a b r : ℝ) : Prop :=
  r = b

-- Condition: Point P(-5, -2) lies on the circle.
def point_on_circle (a b r x y : ℝ) : Prop :=
  (x - a)^2 + (y - b)^2 = r^2

-- Condition: Point Q(-4, -5) lies outside the circle.
def point_outside_circle (a b r x y : ℝ) : Prop :=
  (x - a)^2 + (y - b)^2 > r^2

-- Proof (1): Find the equation of the circle.
theorem equation_of_circle_C :
  ∃ (a b r : ℝ), center_on_line a b ∧ tangent_to_x_axis a b r ∧ point_on_circle a b r (-5) (-2) ∧ point_outside_circle a b r (-4) (-5) ∧ (∀ x y, (x - a)^2 + (y - b)^2 = r^2 ↔ (x + 3)^2 + (y + 2)^2 = 4) :=
sorry

-- Proof (2): Find the equation of the line l.
theorem equation_of_line_l (a b r : ℝ) (ha : center_on_line a b) (hb : tangent_to_x_axis a b r) (hc : point_on_circle a b r (-5) (-2)) (hd : point_outside_circle a b r (-4) (-5)) :
  ∃ (k : ℝ), ∀ x y, ((k = 0 ∧ x = -2) ∨ (k ≠ 0 ∧ y + 4 = -3/4 * (x + 2))) ↔ ((x = -2) ∨ (3 * x + 4 * y + 22 = 0)) :=
sorry

end equation_of_circle_C_equation_of_line_l_l124_124894


namespace rotation_90_deg_l124_124838

theorem rotation_90_deg (z : ℂ) (r : ℂ → ℂ) (h : ∀ (x y : ℝ), r (x + y*I) = -y + x*I) :
  r (8 - 5*I) = 5 + 8*I :=
by sorry

end rotation_90_deg_l124_124838


namespace math_problem_solution_l124_124706

theorem math_problem_solution : ∃ (x y : ℕ), 0 < x ∧ 0 < y ∧ x^y - 1 = y^x ∧ 2*x^y = y^x + 5 ∧ x = 2 ∧ y = 2 :=
by {
  sorry
}

end math_problem_solution_l124_124706


namespace quadratic_completing_the_square_l124_124592

theorem quadratic_completing_the_square :
  ∀ x : ℝ, x^2 - 4 * x - 2 = 0 → (x - 2)^2 = 6 :=
by sorry

end quadratic_completing_the_square_l124_124592


namespace gari_fare_probability_l124_124413

theorem gari_fare_probability :
  let total_coins := 1 + 2 + 6,
      total_ways := Nat.choose total_coins 4,
      unfavorable_outcomes := Nat.choose 6 4 in
  total_coins = 9 ∧
  total_ways = 126 ∧
  unfavorable_outcomes = 15 →
  (total_ways - unfavorable_outcomes).toRat / total_ways.toRat = 37 / 42 :=
by
  intros total_coins total_ways unfavorable_outcomes h
  sorry

end gari_fare_probability_l124_124413


namespace find_f_28_l124_124608

theorem find_f_28 (f : ℕ → ℚ) (h1 : ∀ n : ℕ, f (n + 1) = (3 * f n + n) / 3) (h2 : f 1 = 1) :
  f 28 = 127 := by
sorry

end find_f_28_l124_124608


namespace red_ball_higher_probability_l124_124473

noncomputable def bins := {1, 2, 3, 4}
noncomputable def probability_of_bin (k : bins) : ℝ := (16 / 15) * 2^(-k)

noncomputable def same_bin_probability : ℝ :=
  ∑ k in bins, (probability_of_bin k) ^ 2

noncomputable def different_bin_probability : ℝ :=
  1 - same_bin_probability

noncomputable def higher_numbered_bin_probability : ℝ :=
  different_bin_probability / 2

theorem red_ball_higher_probability : 
  higher_numbered_bin_probability = 0.3533 :=
  by
    sorry

end red_ball_higher_probability_l124_124473


namespace circumcircle_area_l124_124593

theorem circumcircle_area (a b c A B C : ℝ) (h : a * Real.cos B + b * Real.cos A = 4 * Real.sin C) :
    π * (2 : ℝ) ^ 2 = 4 * π :=
by
  sorry

end circumcircle_area_l124_124593


namespace distinct_digit_S_problem_l124_124166

theorem distinct_digit_S_problem :
  ∃! (S : ℕ), S < 10 ∧ 
  ∃ (P Q R : ℕ), P ≠ Q ∧ Q ≠ R ∧ P ≠ R ∧ P ≠ S ∧ Q ≠ S ∧ R ≠ S ∧ 
  P < 10 ∧ Q < 10 ∧ R < 10 ∧
  ((P + Q = S) ∨ (P + Q = S + 10)) ∧
  (R = 0) :=
sorry

end distinct_digit_S_problem_l124_124166


namespace sin_sides_of_triangle_l124_124020

theorem sin_sides_of_triangle {a b c : ℝ} 
  (habc: a + b > c) (hbac: a + c > b) (hcbc: b + c > a) (h_sum: a + b + c ≤ 2 * Real.pi) :
  a > 0 ∧ a < Real.pi ∧ b > 0 ∧ b < Real.pi ∧ c > 0 ∧ c < Real.pi ∧ 
  (Real.sin a + Real.sin b > Real.sin c) ∧ 
  (Real.sin a + Real.sin c > Real.sin b) ∧ 
  (Real.sin b + Real.sin c > Real.sin a) :=
by
  sorry

end sin_sides_of_triangle_l124_124020


namespace calculate_expression_l124_124480

theorem calculate_expression :
  6 * 1000 + 5 * 100 + 6 * 1 = 6506 :=
by
  sorry

end calculate_expression_l124_124480


namespace red_black_ball_ratio_l124_124303

theorem red_black_ball_ratio (R B x : ℕ) (h1 : 3 * R = B + x) (h2 : 2 * R + x = B) :
  R / B = 2 / 5 := by
  sorry

end red_black_ball_ratio_l124_124303


namespace bowl_capacity_l124_124916

theorem bowl_capacity (C : ℝ) (h1 : (2/3) * C * 5 + (1/3) * C * 4 = 700) : C = 150 := 
by
  sorry

end bowl_capacity_l124_124916


namespace arrangement_count_l124_124974

theorem arrangement_count (students : Fin 6) (teacher : Bool) :
  (teacher = true) ∧
  ∀ (A B : Fin 6), 
    A ≠ 0 ∧ B ≠ 5 →
    A ≠ B →
    (Sorry) = 960 := sorry

end arrangement_count_l124_124974


namespace connie_tickets_l124_124993

variable (T : ℕ)

theorem connie_tickets (h : T = T / 2 + 10 + 15) : T = 50 :=
by 
sorry

end connie_tickets_l124_124993


namespace shots_cost_l124_124264

theorem shots_cost (n_dogs : ℕ) (puppies_per_dog : ℕ) (shots_per_puppy : ℕ) (cost_per_shot : ℕ) :
  n_dogs = 3 →
  puppies_per_dog = 4 →
  shots_per_puppy = 2 →
  cost_per_shot = 5 →
  n_dogs * puppies_per_dog * shots_per_puppy * cost_per_shot = 120 := by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  calc
    3 * 4 * 2 * 5 = 12 * 2 * 5 : by rfl
    ... = 24 * 5 : by rfl
    ... = 120 : by rfl

end shots_cost_l124_124264


namespace find_other_number_l124_124802

open Nat

def gcd (a b : ℕ) : ℕ := if a = 0 then b else gcd (b % a) a
noncomputable def lcm (a b : ℕ) : ℕ := a * b / gcd a b

def a : ℕ := 210
def lcm_ab : ℕ := 4620
def gcd_ab : ℕ := 21

theorem find_other_number (b : ℕ) (h_lcm : lcm a b = lcm_ab) (h_gcd : gcd a b = gcd_ab) :
  b = 462 := by
  sorry

end find_other_number_l124_124802


namespace shortest_chord_l124_124568

noncomputable def line_eq (m : ℝ) (x y : ℝ) : Prop := 2 * m * x - y - 8 * m - 3 = 0
noncomputable def circle_eq (x y : ℝ) : Prop := (x - 3)^2 + (y + 6)^2 = 25

theorem shortest_chord (m : ℝ) :
  (∃ x y, line_eq m x y ∧ circle_eq x y) →
  m = 1 / 6 :=
by sorry

end shortest_chord_l124_124568


namespace sum_of_squared_distances_range_l124_124900

theorem sum_of_squared_distances_range
  (φ : ℝ)
  (x : ℝ := 2 * Real.cos φ)
  (y : ℝ := 3 * Real.sin φ)
  (A : ℝ × ℝ := (1, Real.sqrt 3))
  (B : ℝ × ℝ := (-Real.sqrt 3, 1))
  (C : ℝ × ℝ := (-1, -Real.sqrt 3))
  (D : ℝ × ℝ := (Real.sqrt 3, -1))
  (PA := (x - A.1)^2 + (y - A.2)^2)
  (PB := (x - B.1)^2 + (y - B.2)^2)
  (PC := (x - C.1)^2 + (y - C.2)^2)
  (PD := (x - D.1)^2 + (y - D.2)^2) :
  32 ≤ PA + PB + PC + PD ∧ PA + PB + PC + PD ≤ 52 :=
  by sorry

end sum_of_squared_distances_range_l124_124900


namespace excluded_twins_lineup_l124_124840

/-- 
  Prove that the number of ways to choose 5 starters from 15 players,
  such that both Alice and Bob (twins) are not included together in the lineup, is 2717.
-/
theorem excluded_twins_lineup (n : ℕ) (k : ℕ) (t : ℕ) (u : ℕ) (h_n : n = 15) (h_k : k = 5) (h_t : t = 2) (h_u : u = 3) :
  ((n.choose k) - ((n - t).choose u)) = 2717 :=
by {
  sorry
}

end excluded_twins_lineup_l124_124840


namespace sqrt_of_4_l124_124948

theorem sqrt_of_4 : ∃ y : ℝ, y^2 = 4 ∧ (y = 2 ∨ y = -2) :=
by
  sorry

end sqrt_of_4_l124_124948


namespace num_subsets_containing_6_l124_124745

theorem num_subsets_containing_6 : 
  (∃ (subset : set (fin 6)), 6 ∈ subset ∧ fintype.card {s : set (fin 6) | s ∈ subset}) = 32 :=
sorry

end num_subsets_containing_6_l124_124745


namespace sandy_jacket_price_l124_124192

noncomputable def discounted_shirt_price (initial_shirt_price discount_percentage : ℝ) : ℝ :=
  initial_shirt_price - (initial_shirt_price * discount_percentage / 100)

noncomputable def money_left (initial_money additional_money discounted_price : ℝ) : ℝ :=
  initial_money + additional_money - discounted_price

noncomputable def jacket_price_before_tax (remaining_money tax_percentage : ℝ) : ℝ :=
  remaining_money / (1 + tax_percentage / 100)

theorem sandy_jacket_price :
  let initial_money := 13.99
  let initial_shirt_price := 12.14
  let discount_percentage := 5.0
  let additional_money := 7.43
  let tax_percentage := 10.0
  
  let discounted_price := discounted_shirt_price initial_shirt_price discount_percentage
  let remaining_money := money_left initial_money additional_money discounted_price
  
  jacket_price_before_tax remaining_money tax_percentage = 8.99 := sorry

end sandy_jacket_price_l124_124192


namespace arithmetic_sequence_a2_a8_l124_124006

variable {a : ℕ → ℝ}

-- given condition
axiom h1 : a 4 + a 5 + a 6 = 450

-- problem statement
theorem arithmetic_sequence_a2_a8 : a 2 + a 8 = 300 :=
by
  sorry

end arithmetic_sequence_a2_a8_l124_124006


namespace train_distance_l124_124514

theorem train_distance (t : ℕ) (d : ℕ) (rate : d / t = 1 / 2) (total_time : ℕ) (h : total_time = 90) : ∃ distance : ℕ, distance = 45 := by
  sorry

end train_distance_l124_124514


namespace john_candies_on_fourth_day_l124_124171

theorem john_candies_on_fourth_day (c : ℕ) (h1 : 5 * c + 80 = 150) : c + 24 = 38 :=
by 
  -- Placeholder for proof
  sorry

end john_candies_on_fourth_day_l124_124171


namespace problem_statement_l124_124910

open Set

-- Definition of the set S'
def S' : Set (ℤ × ℤ × ℤ) := 
  { p | (0 ≤ p.1 ∧ p.1 ≤ 3) ∧ (0 ≤ p.2.1 ∧ p.2.1 ≤ 4) ∧ (0 ≤ p.2.2 ∧ p.2.2 ≤ 5) }

-- The Lean statement for the problem condition and required proof
theorem problem_statement :
  let valid_midpoint (p1 p2 : ℤ × ℤ × ℤ) := 
    ((p1.1 + p2.1) / 2, (p1.2.1 + p2.2.1) / 2, (p1.2.2 + p2.2.2) / 2) ∈ S' ∧ 
    (∃ p, p ∈ S' ∧ p ≠ p1 ∧ p ≠ p2 ∧ (even p1.1 ∨ even p1.2.1 ∨ even p1.2.2) ∧ (even p2.1 ∨ even p2.2.1 ∨ even p2.2.2)) in
  let m := 13 in
  let n := 25 in
  valid_midpoint ⟶ (m + n = 38) := sorry

end problem_statement_l124_124910


namespace speed_on_local_roads_l124_124841

theorem speed_on_local_roads (v : ℝ) (h1 : 60 + 120 = 180) (h2 : (60 + 120) / (60 / v + 120 / 60) = 36) : v = 20 :=
by
  sorry

end speed_on_local_roads_l124_124841


namespace dalton_needs_more_money_l124_124398

-- Definitions based on the conditions
def jumpRopeCost : ℕ := 7
def boardGameCost : ℕ := 12
def ballCost : ℕ := 4
def savedAllowance : ℕ := 6
def moneyFromUncle : ℕ := 13

-- Computation of how much more money is needed
theorem dalton_needs_more_money : 
  let totalCost := jumpRopeCost + boardGameCost + ballCost
  let totalMoney := savedAllowance + moneyFromUncle
  totalCost - totalMoney = 4 := 
by 
  let totalCost := jumpRopeCost + boardGameCost + ballCost
  let totalMoney := savedAllowance + moneyFromUncle
  have h1 : totalCost = 23 := by rfl
  have h2 : totalMoney = 19 := by rfl
  calc
    totalCost - totalMoney = 23 - 19 := by rw [h1, h2]
    _ = 4 := by rfl

end dalton_needs_more_money_l124_124398


namespace sufficient_but_not_necessary_l124_124096

theorem sufficient_but_not_necessary (x : ℝ) : (x < -1 → x^2 > 1) ∧ ¬(x^2 > 1 → x < -1) :=
by
  sorry

end sufficient_but_not_necessary_l124_124096


namespace simplify_sqrt72_add_sqrt32_l124_124059

theorem simplify_sqrt72_add_sqrt32 : (sqrt 72) + (sqrt 32) = 10 * (sqrt 2) :=
by sorry

end simplify_sqrt72_add_sqrt32_l124_124059


namespace other_number_eq_462_l124_124804

theorem other_number_eq_462 (a b : ℕ) 
  (lcm_ab : Nat.lcm a b = 4620) 
  (gcd_ab : Nat.gcd a b = 21) 
  (a_eq : a = 210) : b = 462 := 
by
  sorry

end other_number_eq_462_l124_124804


namespace sequence_product_l124_124310

theorem sequence_product {n : ℕ} (h : 1 < n) (a : ℕ → ℕ) (h₀ : ∀ n, a n = 2^n) : 
  a (n-1) * a (n+1) = 4^n :=
by sorry

end sequence_product_l124_124310


namespace problem_statement_l124_124278

theorem problem_statement (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) 
  (h4 : x^3 + (1 / (y + 2016)) = y^3 + (1 / (z + 2016))) 
  (h5 : y^3 + (1 / (z + 2016)) = z^3 + (1 / (x + 2016))) : 
  (x^3 + y^3 + z^3) / (x * y * z) = 3 :=
by
  sorry

end problem_statement_l124_124278


namespace xiaoming_problem_l124_124939

theorem xiaoming_problem :
  (- 1 / 24) / (1 / 3 - 1 / 6 + 3 / 8) = - 1 / 13 :=
by
  sorry

end xiaoming_problem_l124_124939


namespace intersection_A_B_eq_B_l124_124414

variable (a : ℝ) (A : Set ℝ) (B : Set ℝ)

def satisfies_quadratic (a : ℝ) (x : ℝ) : Prop := x^2 - a*x + 1 = 0

def set_A : Set ℝ := {1, 2, 3}

def set_B (a : ℝ) : Set ℝ := {x | satisfies_quadratic a x}

theorem intersection_A_B_eq_B (a : ℝ) (h : a ∈ set_A) : 
  (∀ x, x ∈ set_B a → x ∈ set_A) → (∃ x, x ∈ set_A ∧ satisfies_quadratic a x) →
  a = 2 :=
sorry

end intersection_A_B_eq_B_l124_124414


namespace intersection_of_A_and_B_l124_124291

-- Define the sets A and B
def A : Set ℤ := {-1, 0, 1}
def B : Set ℤ := {x | 1 ≤ x ∧ x < 4}

-- The theorem stating the problem
theorem intersection_of_A_and_B : A ∩ B = {1} :=
by
  sorry

end intersection_of_A_and_B_l124_124291


namespace min_value_expr_l124_124122

theorem min_value_expr : ∀ x : ℝ, (x^2 + 8) / Real.sqrt (x^2 + 4) ≥ 4 ∧ ((x = 0) → (x^2 + 8)/Real.sqrt (x^2 + 4) = 4) := by
  intro x
  split
  sorry
  sorry

end min_value_expr_l124_124122


namespace arithmetic_sequence_and_sum_l124_124487

noncomputable def a_n (n : ℕ) : ℤ := 2 * n + 10

def S_n (n : ℕ) : ℤ := n * (12 + 2 * n + 10) / 2

theorem arithmetic_sequence_and_sum :
    (a_n 10 = 30) ∧ 
    (a_n 20 = 50) ∧ 
    (∀ n, S_n n = 11 * n + n^2) ∧ 
    (S_n 3 = 42) :=
by {
    -- a_n 10 = 2 * 10 + 10 = 30
    -- a_n 20 = 2 * 20 + 10 = 50
    -- S_n n = n * (2n + 22) / 2 = 11n + n^2
    -- S_n 3 = 3 * 14 = 42
    sorry
}

end arithmetic_sequence_and_sum_l124_124487


namespace train_crossing_time_l124_124380

def train_length := 140
def train_speed_kmph := 45
def bridge_length := 235
def speed_to_mps (kmph : ℕ) : ℕ := (kmph * 1000) / 3600
def total_distance := train_length + bridge_length
def train_speed := speed_to_mps train_speed_kmph
def time_to_cross := total_distance / train_speed

theorem train_crossing_time : time_to_cross = 30 := by
  sorry

end train_crossing_time_l124_124380


namespace complex_number_quadrant_l124_124589

theorem complex_number_quadrant :
  let z := (2 - (1 * Complex.I)) / (1 + (1 * Complex.I))
  (z.re > 0 ∧ z.im < 0) :=
by
  sorry

end complex_number_quadrant_l124_124589


namespace no_two_digit_number_with_properties_l124_124981

theorem no_two_digit_number_with_properties :
  ¬ (∃ (N : ℕ), (10 ≤ N ∧ N < 100) ∧ 2 ∣ N ∧ 13 ∣ N ∧ (∃ (a b : ℕ), 
    (N = 10 * a + b) ∧ (a * b) ^ (1/2) ∈ ℕ)) :=
by {
  sorry
}

end no_two_digit_number_with_properties_l124_124981


namespace participation_schemes_correct_l124_124109

noncomputable def total_participation_schemes : ℕ :=
  let choose := Nat.choose
  let perm := Nat.factorial
  in (choose 4 2) * (perm 3) - perm 3

theorem participation_schemes_correct :
  total_participation_schemes = 30 :=
by
  sorry

end participation_schemes_correct_l124_124109


namespace rob_total_cards_l124_124336

variables (r r_d j_d : ℕ)

-- Definitions of conditions
def condition1 : Prop := r_d = r / 3
def condition2 : Prop := j_d = 5 * r_d
def condition3 : Prop := j_d = 40

-- Problem Statement
theorem rob_total_cards (h1 : condition1 r r_d)
                        (h2 : condition2 r_d j_d)
                        (h3 : condition3 j_d) :
  r = 24 :=
by
  sorry

end rob_total_cards_l124_124336


namespace find_C_l124_124507

theorem find_C (A B C : ℕ) (h1 : A + B + C = 900) (h2 : A + C = 400) (h3 : B + C = 750) : C = 250 :=
by
  sorry

end find_C_l124_124507


namespace problem1_problem2_l124_124259

-- First Problem
theorem problem1 : 
  Real.cos (Real.pi / 3) + Real.sin (Real.pi / 4) - Real.tan (Real.pi / 4) = (-1 + Real.sqrt 2) / 2 :=
by
  sorry

-- Second Problem
theorem problem2 : 
  6 * (Real.tan (Real.pi / 6))^2 - Real.sqrt 3 * Real.sin (Real.pi / 3) - 2 * Real.cos (Real.pi / 4) = 1 / 2 - Real.sqrt 2 :=
by
  sorry

end problem1_problem2_l124_124259


namespace sum_of_digits_of_N_l124_124248

-- The total number of coins
def total_coins : ℕ := 3081

-- Setting up the equation N^2 = 3081
def N : ℕ := 55 -- Since 55^2 is closest to 3081 and sqrt(3081) ≈ 55

-- Proving the sum of the digits of N is 10
theorem sum_of_digits_of_N : (5 + 5) = 10 :=
by
  sorry

end sum_of_digits_of_N_l124_124248


namespace smallest_area_of_ellipse_l124_124863

theorem smallest_area_of_ellipse 
    (a b : ℝ)
    (h1 : ∀ x y, (x - 2)^2 + y^2 < 4 → (x / a)^2 + (y / b)^2 < 1)
    (h2 : ∀ x y, (x + 2)^2 + y^2 < 4 → (x / a)^2 + (y / b)^2 < 1) :
    π * a * b = π :=
sorry

end smallest_area_of_ellipse_l124_124863


namespace f_of_72_l124_124892

theorem f_of_72 (f : ℕ → ℝ) (p q : ℝ) (h1 : ∀ a b : ℕ, f (a * b) = f a + f b)
  (h2 : f 2 = p) (h3 : f 3 = q) : f 72 = 3 * p + 2 * q := 
sorry

end f_of_72_l124_124892


namespace inequality_am_gm_l124_124890

theorem inequality_am_gm (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : 1/a + 1/b + 1/c ≥ a + b + c) : a + b + c ≥ 3 * a * b * c :=
sorry

end inequality_am_gm_l124_124890


namespace find_f2_l124_124731

-- Definitions based on the given conditions
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

variable (f g : ℝ → ℝ)
variable (h_odd : is_odd_function f)
variable (h_g_def : ∀ x, g x = f x + 9)
variable (h_g_val : g (-2) = 3)

-- Prove the required goal
theorem find_f2 : f 2 = 6 :=
by
  sorry

end find_f2_l124_124731


namespace distance_between_points_A_B_l124_124728

theorem distance_between_points_A_B :
  let A := (8, -5)
  let B := (0, 10)
  Real.sqrt ((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2) = 17 :=
by
  let A := (8, -5)
  let B := (0, 10)
  sorry

end distance_between_points_A_B_l124_124728


namespace area_of_triangle_AEF_l124_124727

open Real

def regular_triangular_prism :=
  { base_edge_length : ℝ,
    lateral_edge_length : ℝ }

variables (D A B C E F : ℝ → ℝ) (prism : regular_triangular_prism)

axiom base_edge_length_is_one : prism.base_edge_length = 1
axiom lateral_edge_length_is_two : prism.lateral_edge_length = 2
axiom plane_intersects_BD_at_E : plane_passes_through A ∧ intersects BD E
axiom plane_intersects_CD_at_F : plane_passes_through A ∧ intersects CD F
axiom perimeter_minimized : minimized_perimeter A E F

theorem area_of_triangle_AEF : area_of_triangle A E F = 3 * sqrt 55 / 64 := sorry

end area_of_triangle_AEF_l124_124727


namespace initial_rows_of_chairs_l124_124885

theorem initial_rows_of_chairs (x : ℕ) (h1 : 12 * x + 11 = 95) : x = 7 := 
by
  sorry

end initial_rows_of_chairs_l124_124885


namespace slower_time_l124_124324

-- Definitions for the problem conditions
def num_stories : ℕ := 50
def lola_time_per_story : ℕ := 12
def tara_time_per_story : ℕ := 10
def tara_stop_time : ℕ := 4
def tara_num_stops : ℕ := num_stories - 2 -- Stops on each floor except the first and last

-- Calculations based on the conditions
def lola_total_time : ℕ := num_stories * lola_time_per_story
def tara_total_time : ℕ := num_stories * tara_time_per_story + tara_num_stops * tara_stop_time

-- Target statement to be proven
theorem slower_time : tara_total_time = 692 := by
  sorry  -- Proof goes here (excluded as per instructions)

end slower_time_l124_124324


namespace tara_dad_second_year_attendance_l124_124710

theorem tara_dad_second_year_attendance :
  let games_played_per_year := 20
  let attendance_rate := 0.90
  let first_year_games_attended := attendance_rate * games_played_per_year
  let second_year_games_difference := 4
  first_year_games_attended - second_year_games_difference = 14 :=
by
  -- We skip the proof here
  sorry

end tara_dad_second_year_attendance_l124_124710


namespace perimeter_triangle_pqr_l124_124082

theorem perimeter_triangle_pqr (PQ PR QR QJ : ℕ) (h1 : PQ = PR) (h2 : QJ = 10) :
  ∃ PQR', PQR' = 198 ∧ triangle PQR PQ PR QR := sorry

end perimeter_triangle_pqr_l124_124082


namespace Morse_code_distinct_symbols_count_l124_124165

theorem Morse_code_distinct_symbols_count :
  let count (n : ℕ) := 2 ^ n
  count 1 + count 2 + count 3 + count 4 + count 5 = 62 :=
by
  sorry

end Morse_code_distinct_symbols_count_l124_124165


namespace sum_series_eq_three_l124_124882

theorem sum_series_eq_three : 
  ∑' (k : ℕ), (k^2 : ℝ) / (2^k : ℝ) = 3 := sorry

end sum_series_eq_three_l124_124882


namespace student_passes_test_probability_l124_124762

noncomputable def probability_passes_test : ℝ :=
  let p := 0.6 in
  let n := 3 in
  let k := 2 in
  (Nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k)) + (Nat.choose n (k + 1)) * (p ^ (k + 1)) * ((1 - p) ^ (n - (k + 1)))

theorem student_passes_test_probability :
  probability_passes_test = 0.648 :=
by
  sorry

end student_passes_test_probability_l124_124762


namespace sphere_surface_area_l124_124920

-- Let A, B, C, D be distinct points on the same sphere
variables (A B C D : ℝ)

-- Defining edges AB, AC, AD and their lengths
variables (AB AC AD : ℝ)
variable (is_perpendicular : AB * AC = 0 ∧ AB * AD = 0 ∧ AC * AD = 0)

-- Setting specific edge lengths
variables (AB_length : AB = 1) (AC_length : AC = 2) (AD_length : AD = 3)

-- The proof problem: Prove that the surface area of the sphere is 14π
theorem sphere_surface_area : 4 * Real.pi * ((1 + 4 + 9) / 4) = 14 * Real.pi :=
by
  sorry

end sphere_surface_area_l124_124920


namespace money_left_is_correct_l124_124540

-- Define initial amount of money Dan has
def initial_amount : ℕ := 3

-- Define the cost of the candy bar
def candy_cost : ℕ := 1

-- Define the money left after the purchase
def money_left : ℕ := initial_amount - candy_cost

-- The theorem stating that the money left is 2
theorem money_left_is_correct : money_left = 2 := by
  sorry

end money_left_is_correct_l124_124540


namespace max_min_y_l124_124551

noncomputable def y (x : ℝ) : ℝ :=
  7 - 4 * (Real.sin x) * (Real.cos x) + 4 * (Real.cos x)^2 - 4 * (Real.cos x)^4

theorem max_min_y :
  (∃ x : ℝ, y x = 10) ∧ (∃ x : ℝ, y x = 6) := by
  sorry

end max_min_y_l124_124551


namespace total_weight_on_scale_l124_124191

def weight_blue_ball : ℝ := 6
def weight_brown_ball : ℝ := 3.12

theorem total_weight_on_scale :
  weight_blue_ball + weight_brown_ball = 9.12 :=
by sorry

end total_weight_on_scale_l124_124191


namespace volume_rectangular_solid_l124_124213

theorem volume_rectangular_solid
  (a b c : ℝ) 
  (h1 : a * b = 12)
  (h2 : b * c = 8)
  (h3 : a * c = 6) :
  a * b * c = 24 :=
sorry

end volume_rectangular_solid_l124_124213


namespace chuck_bicycle_trip_l124_124393

theorem chuck_bicycle_trip (D : ℝ) (h1 : D / 16 + D / 24 = 3) : D = 28.80 :=
by
  sorry

end chuck_bicycle_trip_l124_124393


namespace sum_is_correct_l124_124991

noncomputable def calculate_sum : ℚ :=
  (4 / 3) + (13 / 9) + (40 / 27) + (121 / 81) - (8 / 3)

theorem sum_is_correct : calculate_sum = 171 / 81 := 
by {
  sorry
}

end sum_is_correct_l124_124991


namespace smallest_x_y_sum_l124_124138

theorem smallest_x_y_sum :
  ∃ x y : ℕ,
    0 < x ∧ 0 < y ∧ x ≠ y ∧ (1 / (x : ℝ) + 1 / (y : ℝ) = 1 / 15) ∧ (x + y = 64) := 
by
  sorry

end smallest_x_y_sum_l124_124138


namespace log_product_eq_one_sixth_log_y_x_l124_124708

variable (x y : ℝ) (hx : 0 < x) (hy : 0 < y)

theorem log_product_eq_one_sixth_log_y_x :
  (Real.log x ^ 2 / Real.log (y ^ 5)) * 
  (Real.log (y ^ 3) / Real.log (x ^ 4)) *
  (Real.log (x ^ 4) / Real.log (y ^ 3)) *
  (Real.log (y ^ 5) / Real.log (x ^ 3)) *
  (Real.log (x ^ 3) / Real.log (y ^ 4)) = 
  (1 / 6) * (Real.log x / Real.log y) := 
sorry

end log_product_eq_one_sixth_log_y_x_l124_124708


namespace seeds_germinated_percentage_l124_124127

theorem seeds_germinated_percentage 
  (n1 n2 : ℕ) 
  (p1 p2 : ℝ) 
  (h1 : n1 = 300)
  (h2 : n2 = 200)
  (h3 : p1 = 0.15)
  (h4 : p2 = 0.35) : 
  ( ( p1 * n1 + p2 * n2 ) / ( n1 + n2 ) ) * 100 = 23 :=
by
  -- Mathematical proof goes here.
  sorry

end seeds_germinated_percentage_l124_124127


namespace positive_integer_solution_l124_124717

theorem positive_integer_solution (x y z t : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (ht : 0 < t) :
  (1 / (x * x : ℝ) + 1 / (y * y : ℝ) + 1 / (z * z : ℝ) + 1 / (t * t : ℝ) = 1) ↔ (x = 2 ∧ y = 2 ∧ z = 2 ∧ t = 2) :=
by
  sorry

end positive_integer_solution_l124_124717


namespace rational_coordinates_of_circumcenter_l124_124616

open Classical

noncomputable theory

theorem rational_coordinates_of_circumcenter
  {a1 b1 a2 b2 a3 b3 : ℚ}
  (h1 : ∃ (x y : ℚ), (x - a1)^2 + (y - b1)^2 = (x - a2)^2 + (y - b2)^2 ∧
                      (x - a1)^2 + (y - b1)^2 = (x - a3)^2 + (y - b3)^2) :
  ∃ (x y : ℚ),
    (x - a1)^2 + (y - b1)^2 = (x - a2)^2 + (y - b2)^2 ∧
    (x - a1)^2 + (y - b1)^2 = (x - a3)^2 + (y - b3)^2 := 
begin
  obtain ⟨x, y, hx⟩ := h1,
  use [x, y],
  exact hx,
end

end rational_coordinates_of_circumcenter_l124_124616


namespace find_k_l124_124784

-- Definition of the vertices and conditions
variables {t k : ℝ}
def A : (ℝ × ℝ) := (0, 3)
def B : (ℝ × ℝ) := (0, k)
def C : (ℝ × ℝ) := (t, 10)
def D : (ℝ × ℝ) := (t, 0)

-- Condition that the area of the quadrilateral is 50 square units
def area_cond (height base1 base2 : ℝ) : Prop :=
  50 = (1 / 2) * height * (base1 + base2)

-- Stating the problem in Lean
theorem find_k
  (ht : t = 5)
  (hk : k > 3) 
  (t_pos : t > 0)
  (area : area_cond t (k - 3) 10) :
  k = 13 :=
  sorry

end find_k_l124_124784


namespace avg_remaining_two_l124_124665

variables {A B C D E : ℝ}

-- Conditions
def avg_five (A B C D E : ℝ) : Prop := (A + B + C + D + E) / 5 = 10
def avg_three (A B C : ℝ) : Prop := (A + B + C) / 3 = 4

-- Theorem to prove
theorem avg_remaining_two (A B C D E : ℝ) (h1 : avg_five A B C D E) (h2 : avg_three A B C) : ((D + E) / 2) = 19 := 
sorry

end avg_remaining_two_l124_124665


namespace total_age_l124_124828

theorem total_age (a b c : ℕ) (h1 : a = b + 2) (h2 : b = 2 * c) (h3 : b = 8) : a + b + c = 22 :=
by
  sorry

end total_age_l124_124828


namespace circumcenter_is_rational_l124_124621

theorem circumcenter_is_rational (a1 a2 a3 b1 b2 b3 : ℚ)
  (h1 : (a2 - a1) ≠ 0 ∨ (b2 - b1) ≠ 0)
  (h2 : (a3 - a1) ≠ 0 ∨ (b3 - b1) ≠ 0) :
  ∃ x y : ℚ,
    ((a2 - a1) * x + (b2 - b1) * y = (a2^2 - a1^2 + b2^2 - b1^2) / 2) ∧
    ((a3 - a1) * x + (b3 - b1) * y = (a3^2 - a1^2 + b3^2 - b1^2) / 2) :=
begin
  -- proof goes here
  sorry,
end

end circumcenter_is_rational_l124_124621


namespace tidal_power_station_location_l124_124527

-- Define the conditions
def tidal_power_plants : ℕ := 9
def first_bidirectional_plant := 1980
def significant_bidirectional_plant_location : String := "Jiangxia"
def largest_bidirectional_plant : Prop := true

-- Assumptions based on conditions
axiom china_has_9_tidal_power_plants : tidal_power_plants = 9
axiom first_bidirectional_in_1980 : (first_bidirectional_plant = 1980) -> significant_bidirectional_plant_location = "Jiangxia"
axiom largest_bidirectional_in_world : largest_bidirectional_plant

-- Definition of the problem
theorem tidal_power_station_location : significant_bidirectional_plant_location = "Jiangxia" :=
by
  sorry

end tidal_power_station_location_l124_124527


namespace no_integer_solutions_l124_124922

theorem no_integer_solutions (x y : ℤ) : ¬ (x^2 + 4 * x - 11 = 8 * y) := 
by
  sorry

end no_integer_solutions_l124_124922


namespace largest_part_of_proportional_division_l124_124428

theorem largest_part_of_proportional_division (sum : ℚ) (a b c largest : ℚ) 
  (prop1 prop2 prop3 : ℚ) 
  (h1 : sum = 156)
  (h2 : prop1 = 2)
  (h3 : prop2 = 1 / 2)
  (h4 : prop3 = 1 / 4)
  (h5 : sum = a + b + c)
  (h6 : a / prop1 = b / prop2 ∧ b / prop2 = c / prop3)
  (h7 : largest = max a (max b c)) :
  largest = 112 + 8 / 11 :=
by
  sorry

end largest_part_of_proportional_division_l124_124428


namespace greatest_value_of_sum_l124_124086

theorem greatest_value_of_sum (x y : ℝ) (h₁ : x^2 + y^2 = 100) (h₂ : x * y = 40) :
  x + y = 6 * Real.sqrt 5 :=
by
  sorry

end greatest_value_of_sum_l124_124086


namespace cindy_correct_answer_l124_124394

theorem cindy_correct_answer (x : ℝ) (h : (x - 10) / 5 = 50) : (x - 5) / 10 = 25.5 :=
sorry

end cindy_correct_answer_l124_124394


namespace union_of_M_and_N_l124_124321

def M : Set ℕ := {1, 2, 4, 5}
def N : Set ℕ := {2, 3, 4}

theorem union_of_M_and_N : M ∪ N = {1, 2, 3, 4, 5} :=
by
  sorry

end union_of_M_and_N_l124_124321


namespace salt_amount_evaporation_l124_124168

-- Define the conditions as constants
def total_volume : ℕ := 2 -- 2 liters
def salt_concentration : ℝ := 0.2 -- 20%

-- The volume conversion factor from liters to milliliters.
def liter_to_ml : ℕ := 1000

-- Define the statement to prove
theorem salt_amount_evaporation : total_volume * (salt_concentration * liter_to_ml) = 400 := 
by 
  -- We'll skip the proof steps here
  sorry

end salt_amount_evaporation_l124_124168


namespace lean_proof_l124_124893

noncomputable def proof_problem (f : ℕ → ℤ) (p q : ℤ) : Prop :=
  f(2) = p ∧ f(3) = q ∧ (∀ a b : ℕ, f(a * b) = f(a) + f(b)) → f(72) = 3 * p + 2 * q

-- Here is the statement without the proof
theorem lean_proof (f : ℕ → ℤ) (p q : ℤ) (h1 : f(2) = p) (h2 : f(3) = q)
  (h3 : ∀ a b : ℕ, f(a * b) = f(a) + f(b)) : f(72) = 3 * p + 2 * q :=
by
  sorry

end lean_proof_l124_124893


namespace roger_steps_time_l124_124925

theorem roger_steps_time (steps_per_30_min : ℕ := 2000) (time_for_2000_steps : ℕ := 30) (goal_steps : ℕ := 10000) : 
  (goal_steps * time_for_2000_steps) / steps_per_30_min = 150 :=
by 
  -- This is the statement. Proof is omitted as per instruction.
  sorry

end roger_steps_time_l124_124925


namespace resulting_polygon_sides_l124_124115

/-
Problem statement: 

Construct a regular pentagon on one side of a regular heptagon.
On one non-adjacent side of the pentagon, construct a regular hexagon.
On a non-adjacent side of the hexagon, construct an octagon.
Continue to construct regular polygons in the same way, until you construct a nonagon.
How many sides does the resulting polygon have?

Given facts:
1. Start with a heptagon (7 sides).
2. Construct a pentagon (5 sides) on one side of the heptagon.
3. Construct a hexagon (6 sides) on a non-adjacent side of the pentagon.
4. Construct an octagon (8 sides) on a non-adjacent side of the hexagon.
5. Construct a nonagon (9 sides) on a non-adjacent side of the octagon.
-/

def heptagon_sides : ℕ := 7
def pentagon_sides : ℕ := 5
def hexagon_sides : ℕ := 6
def octagon_sides : ℕ := 8
def nonagon_sides : ℕ := 9

theorem resulting_polygon_sides : 
  (heptagon_sides + nonagon_sides - 2 * 1) + (pentagon_sides + hexagon_sides + octagon_sides - 3 * 2) = 27 := by
  sorry

end resulting_polygon_sides_l124_124115


namespace determine_b_l124_124898

noncomputable def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem determine_b (a b c m1 m2 : ℝ) (h1 : a > b) (h2 : b > c) (h3 : f a b c 1 = 0)
  (h4 : a^2 + (f a b c m1 + f a b c m2) * a + (f a b c m1) * (f a b c m2) = 0) : 
  b ≥ 0 := 
by
  -- Proof logic goes here
  sorry

end determine_b_l124_124898


namespace circumcenter_rational_l124_124613

theorem circumcenter_rational {a1 b1 a2 b2 a3 b3 : ℚ} :
  ∃ (x y : ℚ), 
    ((x - a1)^2 + (y - b1)^2 = (x - a2)^2 + (y - b2)^2) ∧
    ((x - a1)^2 + (y - b1)^2 = (x - a3)^2 + (y - b3)^2) :=
sorry

end circumcenter_rational_l124_124613


namespace toby_remaining_money_l124_124079

theorem toby_remaining_money : 
  let initial_amount : ℕ := 343
  let fraction_given : ℚ := 1/7
  let total_given := 2 * (initial_amount * fraction_given).to_nat
  initial_amount - total_given = 245 :=
by
  let initial_amount := 343
  let fraction_given := (1/7 : ℚ)
  let given_each := (fraction_given * initial_amount).to_nat
  let total_given := 2 * given_each
  have h : initial_amount - total_given = 245 := by
    calc
      initial_amount - total_given
          = 343 - 2 * 49 : by rw [given_each, (343 : ℕ).mul_div_cancel' (7 : ℕ).nat_abs_pos]
      ... = 343 - 98 : by norm_num
      ... = 245 : by norm_num
  exact h

end toby_remaining_money_l124_124079


namespace find_larger_number_l124_124341

theorem find_larger_number (x y : ℕ) (h1 : y - x = 1365) (h2 : y = 4 * x + 15) : y = 1815 :=
sorry

end find_larger_number_l124_124341


namespace people_not_in_any_club_l124_124215

def num_people_company := 120
def num_people_club_A := 25
def num_people_club_B := 34
def num_people_club_C := 21
def num_people_club_D := 16
def num_people_club_E := 10
def overlap_C_D := 8
def overlap_D_E := 4

theorem people_not_in_any_club :
  num_people_company - 
  (num_people_club_A + num_people_club_B + 
  (num_people_club_C + (num_people_club_D - overlap_C_D) + (num_people_club_E - overlap_D_E))) = 26 :=
by
  unfold num_people_company num_people_club_A num_people_club_B num_people_club_C num_people_club_D num_people_club_E overlap_C_D overlap_D_E
  sorry

end people_not_in_any_club_l124_124215


namespace propositions_correctness_l124_124859

noncomputable def correctPropositions : set ℕ :=
  {1, 4, 5}

theorem propositions_correctness :
  (∀ x : ℝ, ∀ x_increase : ℝ, y_increase : ℝ,
    (λ (x : ℝ), 3 + 2 * x) (x + x_increase) - (λ (x : ℝ), 3 + 2 * x) x = y_increase
    ↔ x_increase = 2 ∧ y_increase = 4) ∧
  (∀ r : ℝ, (r ≠ 0) →
    (|r| > 1 → false) ∧ (|r| < 1 → true)) ∧
  (∀ ξ : ℝ, ∀ σ : ℝ, σ > 0 →
    (∫ ξ in set.interval 0 (2 * σ), NormalDist 0 σ ξ) = 0.5 →
    (∫ ξ in set.interval (-2 * σ) 0, NormalDist 0 σ ξ) = 0.4 →
    (∫ ξ in set.interval 2 (∞), NormalDist 0 σ ξ) = 0.1) ∧
  (∫ x in 0..π, sin x dx = 2) ∧
  (∀ n : ℤ, (n ≠ 4) →
    (n / (n-4) + (8-n) / ((8-n)-4) = 2)) →
  correctPropositions = {1, 4, 5} := 
begin
  sorry
end

end propositions_correctness_l124_124859


namespace present_worth_proof_l124_124348

-- Define the conditions
def banker's_gain (BG : ℝ) : Prop := BG = 16
def true_discount (TD : ℝ) : Prop := TD = 96

-- Define the relationship from the problem
def relationship (BG TD PW : ℝ) : Prop := BG = TD - PW

-- Define the present worth of the sum
def present_worth : ℝ := 80

-- Theorem stating that the present worth of the sum is Rs. 80 given the conditions
theorem present_worth_proof (BG TD PW : ℝ)
  (hBG : banker's_gain BG)
  (hTD : true_discount TD)
  (hRelation : relationship BG TD PW) :
  PW = present_worth := by
  sorry

end present_worth_proof_l124_124348


namespace subsets_containing_six_l124_124748

theorem subsets_containing_six :
  ∃ s : Finset (Fin 6), s = {1, 2, 3, 4, 5, 6} ∧ (∃ n : ℕ, n = 32 ∧ n = 2 ^ 5) := by
  sorry

end subsets_containing_six_l124_124748


namespace red_minus_white_more_l124_124695

variable (flowers_total yellow_white red_yellow red_white : ℕ)
variable (h1 : flowers_total = 44)
variable (h2 : yellow_white = 13)
variable (h3 : red_yellow = 17)
variable (h4 : red_white = 14)

theorem red_minus_white_more : 
  (red_yellow + red_white) - (yellow_white + red_white) = 4 :=
by sorry

end red_minus_white_more_l124_124695


namespace smallest_perimeter_iso_triangle_l124_124080

theorem smallest_perimeter_iso_triangle :
  ∃ (x y : ℕ), (PQ = PR ∧ PQ = x ∧ PR = x ∧ QR = y ∧ QJ = 10 ∧ PQ + PR + QR = 416 ∧ 
  PQ = PR ∧ y = 8 ∧ 2 * (x + y) = 416 ∧ y^2 - 50 > 0 ∧ y < 10) :=
sorry

end smallest_perimeter_iso_triangle_l124_124080


namespace second_month_sales_l124_124236

def sales_first_month : ℝ := 7435
def sales_third_month : ℝ := 7855
def sales_fourth_month : ℝ := 8230
def sales_fifth_month : ℝ := 7562
def sales_sixth_month : ℝ := 5991
def average_sales : ℝ := 7500

theorem second_month_sales : 
  ∃ (second_month_sale : ℝ), 
    (sales_first_month + second_month_sale + sales_third_month + sales_fourth_month + sales_fifth_month + sales_sixth_month) / 6 = average_sales ∧
    second_month_sale = 7927 := by
  sorry

end second_month_sales_l124_124236


namespace correct_result_l124_124224

theorem correct_result (x : ℕ) (h: (325 - x) * 5 = 1500) : 325 - x * 5 = 200 := 
by
  -- placeholder for proof
  sorry

end correct_result_l124_124224


namespace disjoint_sets_condition_l124_124419

theorem disjoint_sets_condition (A B : Set ℕ) (h_disjoint: Disjoint A B) (h_union: A ∪ B = Set.univ) :
  ∀ n : ℕ, ∃ a b : ℕ, a > n ∧ b > n ∧ a ≠ b ∧ 
             ((a ∈ A ∧ b ∈ A ∧ a + b ∈ A) ∨ (a ∈ B ∧ b ∈ B ∧ a + b ∈ B)) := 
by
  sorry

end disjoint_sets_condition_l124_124419


namespace composite_product_quotient_l124_124268

def first_seven_composite := [4, 6, 8, 9, 10, 12, 14]
def next_eight_composite := [15, 16, 18, 20, 21, 22, 24, 25]

noncomputable def product {α : Type*} [Monoid α] (l : List α) : α :=
  l.foldl (· * ·) 1

theorem composite_product_quotient : 
  (product first_seven_composite : ℚ) / (product next_eight_composite : ℚ) = 1 / 2475 := 
by 
  sorry

end composite_product_quotient_l124_124268


namespace greatest_integer_gcd_24_eq_4_l124_124498

theorem greatest_integer_gcd_24_eq_4 : ∃ n < 200, n % 4 = 0 ∧ n % 3 ≠ 0 ∧ n % 8 ≠ 0 ∧ n = 196 :=
begin
  sorry
end

end greatest_integer_gcd_24_eq_4_l124_124498


namespace sqrt_expression_equality_l124_124543

theorem sqrt_expression_equality :
  Real.sqrt (25 * Real.sqrt (25 * Real.sqrt 25)) = 5 * 5^(3/4) :=
by
  sorry

end sqrt_expression_equality_l124_124543


namespace num_students_yes_R_l124_124379

noncomputable def num_students_total : ℕ := 800
noncomputable def num_students_yes_only_M : ℕ := 150
noncomputable def num_students_no_to_both : ℕ := 250

theorem num_students_yes_R : (num_students_total - num_students_no_to_both) - num_students_yes_only_M = 400 :=
by
  sorry

end num_students_yes_R_l124_124379


namespace carolyn_practice_time_l124_124872

theorem carolyn_practice_time :
  let minutes_piano := 20
  let days_per_week := 6
  let weeks_per_month := 4
  let multiplier_violin := 3
  let daily_minutes_piano := minutes_piano
  let daily_minutes_violin := multiplier_violin * minutes_piano
  let daily_minutes_total := daily_minutes_piano + daily_minutes_violin
  let weekly_minutes_total := daily_minutes_total * days_per_week
  let monthly_minutes_total := weekly_minutes_total * weeks_per_month
  monthly_minutes_total = 1920 :=
by
  let minutes_piano := 20
  let days_per_week := 6
  let weeks_per_month := 4
  let multiplier_violin := 3
  let daily_minutes_piano := minutes_piano
  let daily_minutes_violin := multiplier_violin * minutes_piano
  let daily_minutes_total := daily_minutes_piano + daily_minutes_violin
  let weekly_minutes_total := daily_minutes_total * days_per_week
  let monthly_minutes_total := weekly_minutes_total * weeks_per_month
  sorry

end carolyn_practice_time_l124_124872


namespace kw_price_approx_4266_percent_l124_124704

noncomputable def kw_price_percentage (A B C D E : ℝ) (hA : KW = 1.5 * A) (hB : KW = 2 * B) (hC : KW = 2.5 * C) (hD : KW = 2.25 * D) (hE : KW = 3 * E) : ℝ :=
  let total_assets := A + B + C + D + E
  let price_kw := 1.5 * A
  (price_kw / total_assets) * 100

theorem kw_price_approx_4266_percent (A B C D E KW : ℝ)
  (hA : KW = 1.5 * A) (hB : KW = 2 * B) (hC : KW = 2.5 * C) (hD : KW = 2.25 * D) (hE : KW = 3 * E)
  (hB_from_A : B = 0.75 * A) (hC_from_A : C = 0.6 * A) (hD_from_A : D = 0.6667 * A) (hE_from_A : E = 0.5 * A) :
  abs ((kw_price_percentage A B C D E hA hB hC hD hE) - 42.66) < 1 :=
by sorry

end kw_price_approx_4266_percent_l124_124704


namespace determine_d_l124_124906

variables (a b c d : ℝ)

-- Conditions given in the problem
def condition1 (a b d : ℝ) : Prop := d / a = (d - 25) / b
def condition2 (b c d : ℝ) : Prop := d / b = (d - 15) / c
def condition3 (a c d : ℝ) : Prop := d / a = (d - 35) / c

-- Final statement to prove
theorem determine_d (a b c : ℝ) (d : ℝ) :
    condition1 a b d ∧ condition2 b c d ∧ condition3 a c d → d = 75 :=
by sorry

end determine_d_l124_124906


namespace rabbit_probability_l124_124103

def cube_vertices : ℕ := 8
def cube_edges : ℕ := 12
def moves : ℕ := 11
def paths_after_11_moves : ℕ := 3 ^ moves
def favorable_paths : ℕ := 24

theorem rabbit_probability :
  (favorable_paths : ℚ) / paths_after_11_moves = 24 / 177147 := by
  sorry

end rabbit_probability_l124_124103


namespace gcd_1978_2017_l124_124220

theorem gcd_1978_2017 : Int.gcd 1978 2017 = 1 :=
sorry

end gcd_1978_2017_l124_124220


namespace intersection_point_l124_124402

variables (g : ℤ → ℤ) (b a : ℤ)
def g_def := ∀ x : ℤ, g x = 4 * x + b
def inv_def := ∀ y : ℤ, g y = -4 → y = a
def point_intersection := ∀ y : ℤ, (g y = -4) → (y = a) → (a = -16 + b)
def solution : ℤ := -4

theorem intersection_point (b a : ℤ) (h₁ : g_def g b) (h₂ : inv_def g a) (h₃ : point_intersection g a b) :
  a = solution :=
  sorry

end intersection_point_l124_124402


namespace smallest_perimeter_triangle_l124_124083

theorem smallest_perimeter_triangle (PQ PR QR : ℕ) (J : Point) :
  PQ = PR →
  QJ = 10 →
  QR = 2 * 10 →
  PQ + PR + QR = 40 :=
by
  sorry

structure Point : Type :=
mk :: (QJ : ℕ)

noncomputable def smallest_perimeter_triangle : Prop :=
  ∃ (PQ PR QR : ℕ) (J : Point), PQ = PR ∧ J.QJ = 10 ∧ QR = 2 * 10 ∧ PQ + PR + QR = 40

end smallest_perimeter_triangle_l124_124083


namespace probability_slope_geq_2_over_5_l124_124361

theorem probability_slope_geq_2_over_5 :
  let events := [(a, b) | a <- [1, 2, 3, 4, 5, 6], b <- [1, 2, 3, 4, 5, 6]],
      favorable_events := [(a, b) ∈ events | (b: ℚ) / a ≤ 2 / 5] in
  (favorable_events.length : ℚ) / events.length = 1 / 6 :=
by
  let events := [(a, b) | a <- [1, 2, 3, 4, 5, 6], b <- [1, 2, 3, 4, 5, 6]],
      favorable_events := events.filter (λ (ab : ℕ × ℕ), (ab.2 : ℚ) / ab.1 ≤ 2 / 5)
  have h1 : events.length = 36 := sorry
  have h2 : favorable_events.length = 6 := sorry
  rw [h1, h2]
  norm_num
  simp

end probability_slope_geq_2_over_5_l124_124361


namespace scalene_triangle_process_l124_124849

theorem scalene_triangle_process (a b c : ℝ) 
  (h1: a > 0) (h2: b > 0) (h3: c > 0) 
  (h4: a + b > c) (h5: b + c > a) (h6: a + c > b) : 
  ¬(∃ k : ℝ, (k > 0) ∧ 
    ((k * a = a + b - c) ∧ 
     (k * b = b + c - a) ∧ 
     (k * c = a + c - b))) ∧ 
  (∀ n: ℕ, n > 0 → (a + b - c)^n + (b + c - a)^n + (a + c - b)^n < 1) :=
by
  sorry

end scalene_triangle_process_l124_124849


namespace augmented_matrix_solution_l124_124286

theorem augmented_matrix_solution (c1 c2 : ℚ) 
    (h1 : 2 * (3 : ℚ) + 3 * (5 : ℚ) = c1)
    (h2 : (5 : ℚ) = c2) : 
    c1 - c2 = 16 := 
by 
  sorry

end augmented_matrix_solution_l124_124286


namespace coefficient_of_q_is_correct_l124_124151

theorem coefficient_of_q_is_correct (q' : ℕ → ℕ) : 
  (∀ q : ℕ, q' q = 3 * q - 3) ∧  q' (q' 7) = 306 → ∃ a : ℕ, (∀ q : ℕ, q' q = a * q - 3) ∧ a = 17 :=
by
  sorry

end coefficient_of_q_is_correct_l124_124151


namespace moles_of_Cu_CN_2_is_1_l124_124718

def moles_of_HCN : Nat := 2
def moles_of_CuSO4 : Nat := 1
def moles_of_Cu_CN_2_formed (hcn : Nat) (cuso4 : Nat) : Nat :=
  if hcn = 2 ∧ cuso4 = 1 then 1 else 0

theorem moles_of_Cu_CN_2_is_1 : moles_of_Cu_CN_2_formed moles_of_HCN moles_of_CuSO4 = 1 :=
by
  sorry

end moles_of_Cu_CN_2_is_1_l124_124718


namespace greatest_possible_subway_takers_l124_124663

/-- In a company with 48 employees, some part-time and some full-time, exactly (1/3) of the part-time
employees and (1/4) of the full-time employees take the subway to work. Prove that the greatest
possible number of employees who take the subway to work is 15. -/
theorem greatest_possible_subway_takers
  (P F : ℕ)
  (h : P + F = 48)
  (h_subway_part : ∀ p, p = P → 0 ≤ p ∧ p ≤ 48)
  (h_subway_full : ∀ f, f = F → 0 ≤ f ∧ f ≤ 48) :
  ∃ y, y = 15 := 
sorry

end greatest_possible_subway_takers_l124_124663


namespace probability_of_same_color_is_correct_l124_124117

def probability_same_color (blue_balls yellow_balls : ℕ) : ℚ :=
  let total_balls := blue_balls + yellow_balls
  let prob_blue := (blue_balls / total_balls : ℚ)
  let prob_yellow := (yellow_balls / total_balls : ℚ)
  (prob_blue ^ 2) + (prob_yellow ^ 2)

theorem probability_of_same_color_is_correct :
  probability_same_color 8 5 = 89 / 169 :=
by 
  sorry

end probability_of_same_color_is_correct_l124_124117


namespace unique_shell_arrangements_l124_124445

theorem unique_shell_arrangements : 
  let shells := 12
  let symmetry_ops := 12
  let total_arrangements := Nat.factorial shells
  let distinct_arrangements := total_arrangements / symmetry_ops
  distinct_arrangements = 39916800 := by
  sorry

end unique_shell_arrangements_l124_124445


namespace pencil_count_l124_124462

theorem pencil_count (a : ℕ) (h1 : 200 ≤ a) (h2 : a ≤ 300)
    (h3 : a % 10 = 7) (h4 : a % 12 = 9) : a = 237 ∨ a = 297 :=
by {
  sorry
}

end pencil_count_l124_124462


namespace pencil_count_l124_124463

theorem pencil_count (a : ℕ) (h1 : 200 ≤ a) (h2 : a ≤ 300)
    (h3 : a % 10 = 7) (h4 : a % 12 = 9) : a = 237 ∨ a = 297 :=
by {
  sorry
}

end pencil_count_l124_124463


namespace find_x_l124_124994

variables (z y x : Int)

def condition1 : Prop := z + 1 = 0
def condition2 : Prop := y - 1 = 1
def condition3 : Prop := x + 2 = -1

theorem find_x (h1 : condition1 z) (h2 : condition2 y) (h3 : condition3 x) : x = -3 :=
by
  sorry

end find_x_l124_124994


namespace value_of_f_log_20_l124_124787

variable (f : ℝ → ℝ)
variable (h₁ : ∀ x : ℝ, f (-x) = -f x)
variable (h₂ : ∀ x : ℝ, f (x - 2) = f (x + 2))
variable (h₃ : ∀ x : ℝ, x > -1 ∧ x < 0 → f x = 2^x + 1/5)

theorem value_of_f_log_20 : f (Real.log 20 / Real.log 2) = -1 := sorry

end value_of_f_log_20_l124_124787


namespace triangle_area_proof_l124_124999

-- Define the triangle sides and median
variables (AB BC BD AC : ℝ)

-- Assume given values
def AB_value : AB = 1 := by sorry 
def BC_value : BC = Real.sqrt 15 := by sorry
def BD_value : BD = 2 := by sorry

-- Assume AC calculated from problem
def AC_value : AC = 4 := by sorry

-- Final proof statement
theorem triangle_area_proof 
  (hAB : AB = 1)
  (hBC : BC = Real.sqrt 15)
  (hBD : BD = 2)
  (hAC : AC = 4) :
  (1 / 2) * AB * BC = (Real.sqrt 15) / 2 := 
sorry

end triangle_area_proof_l124_124999


namespace minimum_perimeter_l124_124494

/-
Given:
1. (a: ℤ), (b: ℤ), (c: ℤ)
2. (a ≠ b)
3. 2 * a + 10 * c = 2 * b + 8 * c
4. 25 * a^2 - 625 * c^2 = 16 * b^2 - 256 * c^2
5. 10 * c / 8 * c = 5 / 4

Prove:
The minimum perimeter is 1180 
-/

theorem minimum_perimeter (a b c : ℤ) 
(h1 : a ≠ b)
(h2 : 2 * a + 10 * c = 2 * b + 8 * c)
(h3 : 25 * a^2 - 625 * c^2 = 16 * b^2 - 256 * c^2)
(h4 : 10 * c / 8 * c = 5 / 4) :
2 * a + 10 * c = 1180 ∨ 2 * b + 8 * c = 1180 :=
sorry

end minimum_perimeter_l124_124494


namespace find_a_l124_124533

def g (x : ℝ) : ℝ := 5 * x - 7

theorem find_a (a : ℝ) : g a = 0 → a = 7 / 5 := by
  intro h
  sorry

end find_a_l124_124533


namespace find_two_digit_number_l124_124719

theorem find_two_digit_number : 
  ∃ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ (b = 0 ∨ b = 5) ∧ (10 * a + b = 5 * (a + b)) ∧ (10 * a + b = 45) :=
by
  sorry

end find_two_digit_number_l124_124719


namespace train_speed_l124_124853

theorem train_speed (length_of_train : ℝ) (time_to_cross : ℝ) (conversion_factor : ℝ) :
  length_of_train = 200 → 
  time_to_cross = 24 → 
  conversion_factor = 3600 → 
  (length_of_train / 1000) / (time_to_cross / conversion_factor) = 30 := 
by
  sorry

end train_speed_l124_124853


namespace population_growth_l124_124951

theorem population_growth (P : ℝ) (x : ℝ) (y : ℝ) 
  (h₁ : P = 5.48) 
  (h₂ : y = P * (1 + x / 100)^8) : 
  y = 5.48 * (1 + x / 100)^8 := 
by
  sorry

end population_growth_l124_124951


namespace multinomial_expansion_terms_l124_124935

theorem multinomial_expansion_terms :
  let terms := { (a, b, c) : ℕ × ℕ × ℕ // a + b + c = 10 }
  in terms.finite.to_finset.card = 66 :=
by {
  sorry
}

end multinomial_expansion_terms_l124_124935


namespace find_m_plus_n_l124_124424

noncomputable def f (x : ℝ) : ℝ := - (1 / 2) * x^2 + x

theorem find_m_plus_n (m n : ℝ) (h1 : m < n ∧ n ≤ 1) (h2 : ∀ (x : ℝ), m ≤ x ∧ x ≤ n → 3 * m ≤ f x ∧ f x ≤ 3 * n) : m + n = -4 :=
by
  have H1 : - (1 / 2) * m^2 + m = 3 * m := sorry
  have H2 : - (1 / 2) * n^2 + n = 3 * n := sorry
  sorry

end find_m_plus_n_l124_124424


namespace derivative_of_sin_squared_minus_cos_squared_l124_124478

noncomputable def func (x : ℝ) : ℝ := (Real.sin x)^2 - (Real.cos x)^2

theorem derivative_of_sin_squared_minus_cos_squared (x : ℝ) :
  deriv func x = 2 * Real.sin (2 * x) :=
sorry

end derivative_of_sin_squared_minus_cos_squared_l124_124478


namespace find_d_l124_124179

noncomputable def f (x : ℝ) (a b c d : ℝ) : ℝ := x^4 + a*x^3 + b*x^2 + c*x + d

theorem find_d (a b c d : ℝ) (roots_negative_integers : ∀ x, f x a b c d = 0 → x < 0) (sum_is_2023 : a + b + c + d = 2023) :
  d = 17020 :=
sorry

end find_d_l124_124179


namespace convex_polyhedron_even_faces_not_necessarily_two_color_edges_l124_124546

open SimpleGraph

theorem convex_polyhedron_even_faces_not_necessarily_two_color_edges (P : Type) [Fintype P] 
  (f : P → SimpleGraph P) (h : ∀ p ∈ P, (f p).adjacencyMatrix.even) :
  ¬∀ (c : P → Fin 2), ∀ (p ∈ P), (f p).EdgeSet.card / 2 = Cardinal.mk {e ∈ (f p).EdgeSet | c e = 0} :=
by
  sorry

end convex_polyhedron_even_faces_not_necessarily_two_color_edges_l124_124546


namespace fraction_addition_l124_124964

theorem fraction_addition :
  let a := (1 : ℚ) / 6
  let b := (5 : ℚ) / 12
  a + b = 7 / 12 :=
by
  let a := (1 : ℚ) / 6
  let b := (5 : ℚ) / 12
  have : a + b = 7 / 12 := sorry
  exact this

end fraction_addition_l124_124964


namespace find_OP_l124_124929

variable (a b c d e f : ℝ)
variable (P : ℝ)

-- Given conditions
axiom AP_PD_ratio : (a - P) / (P - d) = 2 / 3
axiom BP_PC_ratio : (b - P) / (P - c) = 3 / 4

-- Conclusion to prove
theorem find_OP : P = (3 * a + 2 * d) / 5 :=
by
  sorry

end find_OP_l124_124929


namespace total_pears_l124_124108

theorem total_pears (S P C : ℕ) (hS : S = 20) (hP : P = (S - S / 2)) (hC : C = (P + P / 5)) : S + P + C = 42 :=
by
  -- We state the theorem with the given conditions and the goal of proving S + P + C = 42.
  sorry

end total_pears_l124_124108


namespace find_m_l124_124161

theorem find_m (m : ℝ) : 
  (m^2 + 3 * m + 3 ≠ 0) ∧ (m^2 + 2 * m - 3 ≠ 0) ∧ 
  (m^2 + 3 * m + 3 = 1) → m = -2 := 
by
  sorry

end find_m_l124_124161


namespace oblique_projection_correct_statements_l124_124645

-- Definitions of conditions
def oblique_projection_parallel_invariant : Prop :=
  ∀ (x_parallel y_parallel : Prop), x_parallel ∧ y_parallel

def oblique_projection_length_changes : Prop :=
  ∀ (x y : ℝ), x = y / 2 ∨ x = y

def triangle_is_triangle : Prop :=
  ∀ (t : Type), t = t

def square_is_rhombus : Prop :=
  ∀ (s : Type), s = s → false

def isosceles_trapezoid_is_parallelogram : Prop :=
  ∀ (it : Type), it = it → false

def rhombus_is_rhombus : Prop :=
  ∀ (r : Type), r = r → false

-- Math proof problem
theorem oblique_projection_correct_statements :
  (triangle_is_triangle ∧ oblique_projection_parallel_invariant ∧ oblique_projection_length_changes)
  → ¬square_is_rhombus ∧ ¬isosceles_trapezoid_is_parallelogram ∧ ¬rhombus_is_rhombus :=
by 
  sorry

end oblique_projection_correct_statements_l124_124645


namespace radius_triple_area_l124_124064

variable (r n : ℝ)

theorem radius_triple_area (h : π * (r + n) ^ 2 = 3 * π * r ^ 2) : r = (n / 2) * (Real.sqrt 3 - 1) :=
sorry

end radius_triple_area_l124_124064


namespace sum_of_roots_eq_14_l124_124360

theorem sum_of_roots_eq_14 : 
  (∀ x : ℝ, (x - 7)^2 = 16 → (x = 11 ∨ x = 3)) → 11 + 3 = 14 :=
by
  intro h
  have x1 : 11 = 11 := rfl
  have x2 : 3 = 3 := rfl
  exact rfl

end sum_of_roots_eq_14_l124_124360


namespace solution_x_percentage_of_alcohol_l124_124688

variable (P : ℝ) -- percentage of alcohol by volume in solution x, in decimal form

theorem solution_x_percentage_of_alcohol :
  (0.30 : ℝ) * 200 + P * 200 = 0.20 * 400 → P = 0.10 :=
by
  intro h
  sorry

end solution_x_percentage_of_alcohol_l124_124688


namespace gain_percentage_l124_124226

theorem gain_percentage (selling_price gain : ℝ) (h_selling : selling_price = 90) (h_gain : gain = 15) : 
  (gain / (selling_price - gain)) * 100 = 20 := 
by
  sorry

end gain_percentage_l124_124226


namespace problem_statement_l124_124330

theorem problem_statement (a b c : ℝ) (h : a^2 + b^2 - a * b = c^2) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a - c) * (b - c) ≤ 0 :=
by sorry

end problem_statement_l124_124330


namespace square_side_length_l124_124066

theorem square_side_length (d : ℝ) (s : ℝ) (h : d = Real.sqrt 2) (h2 : d = Real.sqrt 2 * s) : s = 1 :=
by
  sorry

end square_side_length_l124_124066


namespace price_change_theorem_l124_124990

-- Define initial prices
def candy_box_price_before : ℝ := 10
def soda_can_price_before : ℝ := 9
def popcorn_bag_price_before : ℝ := 5
def gum_pack_price_before : ℝ := 2

-- Define price changes
def candy_box_price_increase := candy_box_price_before * 0.25
def soda_can_price_decrease := soda_can_price_before * 0.15
def popcorn_bag_price_factor := 2
def gum_pack_price_change := 0

-- Compute prices after the policy changes
def candy_box_price_after := candy_box_price_before + candy_box_price_increase
def soda_can_price_after := soda_can_price_before - soda_can_price_decrease
def popcorn_bag_price_after := popcorn_bag_price_before * popcorn_bag_price_factor
def gum_pack_price_after := gum_pack_price_before

-- Compute total costs
def total_cost_before := candy_box_price_before + soda_can_price_before + popcorn_bag_price_before + gum_pack_price_before
def total_cost_after := candy_box_price_after + soda_can_price_after + popcorn_bag_price_after + gum_pack_price_after

-- The statement to be proven
theorem price_change_theorem :
  total_cost_before = 26 ∧ total_cost_after = 32.15 :=
by
  -- This part requires proof, add 'sorry' for now
  sorry

end price_change_theorem_l124_124990


namespace total_tickets_sold_l124_124077

theorem total_tickets_sold 
  (ticket_price : ℕ) 
  (discount_40_percent : ℕ → ℕ) 
  (discount_15_percent : ℕ → ℕ) 
  (revenue : ℕ) 
  (people_10_discount_40 : ℕ) 
  (people_20_discount_15 : ℕ) 
  (people_full_price : ℕ)
  (h_ticket_price : ticket_price = 20)
  (h_discount_40 : ∀ n, discount_40_percent n = n * 12)
  (h_discount_15 : ∀ n, discount_15_percent n = n * 17)
  (h_revenue : revenue = 760)
  (h_people_10_discount_40 : people_10_discount_40 = 10)
  (h_people_20_discount_15 : people_20_discount_15 = 20)
  (h_people_full_price : people_full_price * ticket_price = 300) :
  (people_10_discount_40 + people_20_discount_15 + people_full_price = 45) :=
by
  sorry

end total_tickets_sold_l124_124077


namespace pencil_eraser_cost_l124_124027

/-- Oscar buys 13 pencils and 3 erasers for 100 cents. A pencil costs more than an eraser, 
    and both items cost a whole number of cents. 
    We need to prove that the total cost of one pencil and one eraser is 10 cents. -/
theorem pencil_eraser_cost (p e : ℕ) (h1 : 13 * p + 3 * e = 100) (h2 : p > e) : p + e = 10 :=
sorry

end pencil_eraser_cost_l124_124027


namespace find_f_7_l124_124285

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_7 (h_odd : ∀ x, f (-x) = -f x)
                 (h_periodic : ∀ x, f (x + 4) = f x)
                 (h_interval : ∀ x, 0 < x ∧ x < 2 → f x = 2 * x ^ 2) :
  f 7 = -2 := 
sorry

end find_f_7_l124_124285


namespace lunch_cost_before_tip_l124_124954

theorem lunch_cost_before_tip (C : ℝ) (h : C + 0.2 * C = 60.6) : C = 50.5 :=
sorry

end lunch_cost_before_tip_l124_124954


namespace ratio_of_b_l124_124607

theorem ratio_of_b (a b k a1 a2 b1 b2 : ℝ) (h_nonzero_a2 : a2 ≠ 0) (h_nonzero_b12: b1 ≠ 0 ∧ b2 ≠ 0) :
  (a * b = k) →
  (a1 * b1 = a2 * b2) →
  (a1 / a2 = 3 / 5) →
  (b1 / b2 = 5 / 3) := 
sorry

end ratio_of_b_l124_124607


namespace length_PQ_l124_124579

theorem length_PQ (AB BC CA AH : ℝ) (P Q : ℝ) : 
  AB = 7 → BC = 8 → CA = 9 → 
  AH = 3 * Real.sqrt 5 → 
  PQ = AQ - AP → 
  AQ = 7 * (Real.sqrt 5) / 3 → 
  AP = 9 * (Real.sqrt 5) / 5 → 
  PQ = Real.sqrt 5 * 8 / 15 :=
by
  intros hAB hBC hCA hAH hPQ hAQ hAP
  sorry

end length_PQ_l124_124579


namespace smallest_sum_l124_124143

theorem smallest_sum (x y : ℕ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hxy_neq : x ≠ y) 
  (h_fraction : (1 : ℚ) / x + (1 : ℚ) / y = 1 / 15) : x + y = 64 :=
sorry

end smallest_sum_l124_124143


namespace sample_size_correct_l124_124683

-- Define the conditions as lean variables
def total_employees := 120
def male_employees := 90
def female_sample := 9

-- Define the proof problem statement
theorem sample_size_correct : ∃ n : ℕ, (total_employees - male_employees) / total_employees = female_sample / n ∧ n = 36 := by 
  sorry

end sample_size_correct_l124_124683


namespace probability_one_white_ball_initial_find_n_if_one_red_ball_l124_124231

-- Define the initial conditions: 5 red balls and 3 white balls
def initial_red_balls := 5
def initial_white_balls := 3
def total_initial_balls := initial_red_balls + initial_white_balls

-- Define the probability of drawing exactly one white ball initially
def prob_draw_one_white := initial_white_balls / total_initial_balls

-- Define the number of white balls added
variable (n : ℕ)

-- Define the total number of balls after adding n white balls
def total_balls_after_adding := total_initial_balls + n

-- Define the probability of drawing exactly one red ball after adding n white balls
def prob_draw_one_red := initial_red_balls / total_balls_after_adding

-- Prove that the probability of drawing one white ball initially is 3/8
theorem probability_one_white_ball_initial : prob_draw_one_white = 3 / 8 := by
  sorry

-- Prove that, if the probability of drawing one red ball after adding n white balls is 1/2, then n = 2
theorem find_n_if_one_red_ball : prob_draw_one_red = 1 / 2 -> n = 2 := by
  sorry

end probability_one_white_ball_initial_find_n_if_one_red_ball_l124_124231


namespace total_weight_of_8_bags_total_sales_amount_of_qualified_products_l124_124633

-- Definitions
def deviations : List ℤ := [-6, -3, -2, 0, 1, 4, 5, -1]
def standard_weight_per_bag : ℤ := 450
def threshold : ℤ := 4
def price_per_bag : ℤ := 3

-- Part 1: Total weight of the 8 bags of laundry detergent
theorem total_weight_of_8_bags : 
  8 * standard_weight_per_bag + deviations.sum = 3598 := 
by
  sorry

-- Part 2: Total sales amount of qualified products
theorem total_sales_amount_of_qualified_products : 
  price_per_bag * (deviations.filter (fun x => abs x ≤ threshold)).length = 18 := 
by
  sorry

end total_weight_of_8_bags_total_sales_amount_of_qualified_products_l124_124633


namespace range_of_a_l124_124725

noncomputable def f (a x : ℝ) : ℝ := x^2 - 2*a*x + 2

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, x ≥ -1 → f a x ≥ a) : -3 ≤ a ∧ a ≤ 1 :=
by
  sorry

end range_of_a_l124_124725


namespace three_pow_2010_mod_eight_l124_124502

theorem three_pow_2010_mod_eight : (3^2010) % 8 = 1 :=
  sorry

end three_pow_2010_mod_eight_l124_124502


namespace roger_steps_to_minutes_l124_124928

theorem roger_steps_to_minutes (h1 : ∃ t: ℕ, t = 30 ∧ ∃ s: ℕ, s = 2000)
                               (h2 : ∃ g: ℕ, g = 10000) :
  ∃ m: ℕ, m = 150 :=
by 
  sorry

end roger_steps_to_minutes_l124_124928


namespace maximize_q_l124_124318

noncomputable def maximum_q (X Y Z : ℕ) : ℕ :=
X * Y * Z + X * Y + Y * Z + Z * X

theorem maximize_q : ∃ (X Y Z : ℕ), X + Y + Z = 15 ∧ (∀ (A B C : ℕ), A + B + C = 15 → X * Y * Z + X * Y + Y * Z + Z * X ≥ A * B * C + A * B + B * C + C * A) ∧ maximum_q X Y Z = 200 :=
by
  sorry

end maximize_q_l124_124318


namespace xiaoming_problem_l124_124940

theorem xiaoming_problem :
  (- 1 / 24) / (1 / 3 - 1 / 6 + 3 / 8) = - 1 / 13 :=
by
  sorry

end xiaoming_problem_l124_124940


namespace time_for_A_to_complete_race_l124_124002

open Real

theorem time_for_A_to_complete_race (V_A V_B : ℝ) (T_A : ℝ) :
  (V_B = 4) →
  (V_B = 960 / T_A) →
  T_A = 1000 / V_A →
  T_A = 240 := by
  sorry

end time_for_A_to_complete_race_l124_124002


namespace calculate_shot_cost_l124_124260

theorem calculate_shot_cost :
  let num_pregnant_dogs := 3
  let puppies_per_dog := 4
  let shots_per_puppy := 2
  let cost_per_shot := 5
  let total_puppies := num_pregnant_dogs * puppies_per_dog
  let total_shots := total_puppies * shots_per_puppy
  let total_cost := total_shots * cost_per_shot
  total_cost = 120 :=
by
  sorry

end calculate_shot_cost_l124_124260


namespace discriminant_is_four_l124_124635

-- Define the quadratic equation components
def quadratic_a (a : ℝ) := 1
def quadratic_b (a : ℝ) := 2 * a
def quadratic_c (a : ℝ) := a^2 - 1

-- Define the discriminant of the quadratic equation
def discriminant (a : ℝ) := quadratic_b a ^ 2 - 4 * quadratic_a a * quadratic_c a

-- Statement to prove: The discriminant is 4
theorem discriminant_is_four (a : ℝ) : discriminant a = 4 :=
by {
  sorry
}

end discriminant_is_four_l124_124635


namespace pencil_count_l124_124458

theorem pencil_count (a : ℕ) :
  200 ≤ a ∧ a ≤ 300 ∧ a % 10 = 7 ∧ a % 12 = 9 → (a = 237 ∨ a = 297) :=
by sorry

end pencil_count_l124_124458


namespace rotated_intersection_point_l124_124891

theorem rotated_intersection_point (θ : ℝ) (hθ : 0 < θ ∧ θ < π / 2) : 
  ∃ P : ℝ × ℝ, P = (-Real.sin θ, Real.cos θ) ∧ 
    ∃ φ : ℝ, φ = θ + π / 2 ∧ 
      P = (Real.cos φ, Real.sin φ) := 
by
  sorry

end rotated_intersection_point_l124_124891


namespace kitten_weight_l124_124375

theorem kitten_weight :
  ∃ (x y z : ℝ), x + y + z = 36 ∧ x + z = 3 * y ∧ x + y = 1 / 2 * z ∧ x = 3 := 
by
  sorry

end kitten_weight_l124_124375


namespace circumcenter_rational_l124_124626

theorem circumcenter_rational {a1 b1 a2 b2 a3 b3 : ℚ} 
  (h1 : a1 ≠ a2 ∨ b1 ≠ b2) 
  (h2 : a1 ≠ a3 ∨ b1 ≠ b3) 
  (h3 : a2 ≠ a3 ∨ b2 ≠ b3) :
  ∃ (x y : ℚ), 
    (x - a1)^2 + (y - b1)^2 = (x - a2)^2 + (y - b2)^2 ∧
    (x - a1)^2 + (y - b1)^2 = (x - a3)^2 + (y - b3)^2 :=
sorry

end circumcenter_rational_l124_124626


namespace solve_equations_l124_124407

theorem solve_equations (x : ℝ) (h1 : x^2 - 9 = 0) (h2 : (-x)^3 = (-8)^2) : x = 3 ∨ x = -3 ∨ x = -4 :=
by 
  sorry

end solve_equations_l124_124407


namespace first_player_can_always_make_A_eq_6_l124_124025

def maxSum3x3In5x5Board (board : Fin 5 → Fin 5 → ℕ) (i j : Fin 3) : ℕ :=
  (i + 3 : Fin 5) * (j + 3 : Fin 5) + 
  (i + 3 : Fin 5) * (j + 4 : Fin 5) + 
  (i + 3 : Fin 5) * (j + 5 : Fin 5) + 
  (i + 4 : Fin 5) * (j + 3 : Fin 5) + 
  (i + 4 : Fin 5) * (j + 4 : Fin 5) + 
  (i + 4 : Fin 5) * (j + 5 : Fin 5) + 
  (i + 5 : Fin 5) * (j + 3 : Fin 5) + 
  (i + 5 : Fin 5) * (j + 4 : Fin 5) + 
  (i + 5 : Fin 5) * (j + 5 : Fin 5)

theorem first_player_can_always_make_A_eq_6 :
  ∀ (board : Fin 5 → Fin 5 → ℕ), 
  (∀ (i j : Fin 3), maxSum3x3In5x5Board board i j = 6)
  :=
by
  intros board i j
  sorry

end first_player_can_always_make_A_eq_6_l124_124025


namespace add_fractions_l124_124961

theorem add_fractions : (1 / 6 : ℚ) + (5 / 12) = 7 / 12 := 
by
  sorry

end add_fractions_l124_124961


namespace value_of_a_l124_124537

theorem value_of_a :
  ∀ (g : ℝ → ℝ), (∀ x, g x = 5*x - 7) → ∃ a, g a = 0 ∧ a = 7 / 5 :=
by
  sorry

end value_of_a_l124_124537


namespace group_selection_l124_124000

theorem group_selection (m f : ℕ) (h1 : m + f = 8) (h2 : (m * (m - 1) / 2) * f = 30) : f = 3 :=
sorry

end group_selection_l124_124000


namespace helen_chocolate_chip_cookies_l124_124569

theorem helen_chocolate_chip_cookies :
  let cookies_yesterday := 527
  let cookies_morning := 554
  cookies_yesterday + cookies_morning = 1081 :=
by
  let cookies_yesterday := 527
  let cookies_morning := 554
  show cookies_yesterday + cookies_morning = 1081
  -- The proof is omitted according to the provided instructions 
  sorry

end helen_chocolate_chip_cookies_l124_124569


namespace g_symmetry_solutions_l124_124911

noncomputable def g : ℝ → ℝ := sorry

theorem g_symmetry_solutions (g_def: ∀ (x : ℝ), x ≠ 0 → g x + 3 * g (1 / x) = 6 * x^2) :
  ∀ (x : ℝ), g x = g (-x) → x = 1 ∨ x = -1 :=
by
  sorry

end g_symmetry_solutions_l124_124911


namespace mark_last_shots_l124_124693

theorem mark_last_shots (h1 : 0.60 * 15 = 9) (h2 : 0.65 * 25 = 16.25) : 
  ∀ (successful_shots_first_15 successful_shots_total: ℤ),
  successful_shots_first_15 = 9 ∧ 
  successful_shots_total = 16 → 
  successful_shots_total - successful_shots_first_15 = 7 := by
  sorry

end mark_last_shots_l124_124693


namespace average_of_seven_consecutive_l124_124466

theorem average_of_seven_consecutive (
  a : ℤ 
  ) (c : ℤ) 
  (h1 : c = (a + 1 + a + 2 + a + 3 + a + 4 + a + 5 + a + 6 + a + 7) / 7) : 
  (c + (c + 1) + (c + 2) + (c + 3) + (c + 4) + (c + 5) + (c + 6)) / 7 = a + 7 := 
by 
  sorry

end average_of_seven_consecutive_l124_124466


namespace last_three_digits_of_5_power_odd_l124_124205

theorem last_three_digits_of_5_power_odd (n : ℕ) (h : n % 2 = 1) : (5 ^ n) % 1000 = 125 :=
sorry

end last_three_digits_of_5_power_odd_l124_124205


namespace floor_area_cannot_exceed_10_square_meters_l124_124938

theorem floor_area_cannot_exceed_10_square_meters
  (a b : ℝ)
  (h : 3 > 0)
  (floor_lt_wall1 : a * b < 3 * a)
  (floor_lt_wall2 : a * b < 3 * b) :
  a * b ≤ 9 :=
by
  -- This is where the proof would go
  sorry

end floor_area_cannot_exceed_10_square_meters_l124_124938


namespace find_circle_center_l124_124977

theorem find_circle_center :
  ∃ (a b : ℝ), a = 1 / 2 ∧ b = 7 / 6 ∧
  (0 - a)^2 + (1 - b)^2 = (1 - a)^2 + (1 - b)^2 ∧
  (1 - a) * 3 = b - 1 :=
by {
  sorry
}

end find_circle_center_l124_124977


namespace employees_count_l124_124489

theorem employees_count (E M : ℝ) (h1 : M = 0.99 * E) (h2 : M - 299.9999999999997 = 0.98 * E) :
  E = 30000 :=
by sorry

end employees_count_l124_124489


namespace volleyball_tournament_l124_124670

theorem volleyball_tournament (teams : Finset ℕ) (H : ∀ s : Finset ℕ, s.card = 55 → ∃ t ∈ s, (s.erase t).filter (λ x, (x, t) ∈ losses ∨ (t, x) ∈ losses).card ≤ 4) :
  ∃ t ∈ teams, (teams.erase t).filter (λ x, (x, t) ∈ losses ∨ (t, x) ∈ losses).card ≤ 4 := 
sorry

end volleyball_tournament_l124_124670


namespace find_midpoint_in_polar_l124_124438

noncomputable def midpoint_polar_coordinates (r₁ θ₁ r₂ θ₂ : ℝ) [h₁ : 0 ≤ θ₁] [h₂ : θ₁ < 2 * Real.pi] [h₃ : 0 ≤ θ₂] [h₄ : θ₂ < 2 * Real.pi] :
  ℝ × ℝ := let x₁ := r₁ * Real.cos θ₁
              y₁ := r₁ * Real.sin θ₁
              x₂ := r₂ * Real.cos θ₂
              y₂ := r₂ * Real.sin θ₂
              mx := (x₁ + x₂) / 2
              my := (y₁ + y₂) / 2
              mr := Real.sqrt (mx^2 + my^2)
              mθ := Real.atan2 my mx
  in (mr, mθ)

theorem find_midpoint_in_polar :
  midpoint_polar_coordinates 10 (Real.pi / 4) 10 (3 * Real.pi / 4) = (5 * Real.sqrt 2, Real.pi / 2) := 
sorry

end find_midpoint_in_polar_l124_124438


namespace average_age_of_contestants_l124_124307

theorem average_age_of_contestants :
  let numFemales := 12
  let avgAgeFemales := 25
  let numMales := 18
  let avgAgeMales := 40
  let sumAgesFemales := avgAgeFemales * numFemales
  let sumAgesMales := avgAgeMales * numMales
  let totalSumAges := sumAgesFemales + sumAgesMales
  let totalContestants := numFemales + numMales
  (totalSumAges / totalContestants) = 34 := by
  sorry

end average_age_of_contestants_l124_124307


namespace football_cost_is_correct_l124_124857

def total_spent_on_toys : ℝ := 12.30
def spent_on_marbles : ℝ := 6.59
def spent_on_football := total_spent_on_toys - spent_on_marbles

theorem football_cost_is_correct : spent_on_football = 5.71 :=
by
  sorry

end football_cost_is_correct_l124_124857


namespace find_a_l124_124295

theorem find_a (a b c : ℕ) (h1 : a + b = c) (h2 : b + c = 5) (h3 : c = 3) : a = 1 := by
  sorry

end find_a_l124_124295


namespace ratio_side_length_to_brush_width_l124_124980

theorem ratio_side_length_to_brush_width (s w : ℝ) (h1 : w = s / 4) (h2 : s^2 / 3 = w^2 + ((s - w)^2) / 2) :
    s / w = 4 := by
  sorry

end ratio_side_length_to_brush_width_l124_124980


namespace bags_of_hammers_to_load_l124_124266

noncomputable def total_crate_capacity := 15 * 20
noncomputable def weight_of_nails := 4 * 5
noncomputable def weight_of_planks := 10 * 30
noncomputable def weight_to_be_left_out := 80
noncomputable def effective_capacity := total_crate_capacity - weight_to_be_left_out
noncomputable def weight_of_loaded_planks := 220

theorem bags_of_hammers_to_load : (effective_capacity - weight_of_nails - weight_of_loaded_planks = 0) :=
by
  sorry

end bags_of_hammers_to_load_l124_124266


namespace original_number_l124_124377

theorem original_number (x : ℕ) (h : ∃ k, 14 * x = 112 * k) : x = 8 :=
sorry

end original_number_l124_124377


namespace ratio_of_oranges_l124_124437

def num_good_oranges : ℕ := 24
def num_bad_oranges : ℕ := 8
def ratio_good_to_bad : ℕ := num_good_oranges / num_bad_oranges

theorem ratio_of_oranges : ratio_good_to_bad = 3 := by
  show 24 / 8 = 3
  sorry

end ratio_of_oranges_l124_124437


namespace prove_optionC_is_suitable_l124_124657

def OptionA := "Understanding the height of students in Class 7(1)"
def OptionB := "Companies recruiting and interviewing job applicants"
def OptionC := "Investigating the impact resistance of a batch of cars"
def OptionD := "Selecting the fastest runner in our school to participate in the city-wide competition"

def is_suitable_for_sampling_survey (option : String) : Prop :=
  option = OptionC

theorem prove_optionC_is_suitable :
  is_suitable_for_sampling_survey OptionC :=
by
  sorry

end prove_optionC_is_suitable_l124_124657


namespace seniors_selected_correct_l124_124105

-- Definitions based on the conditions problem
def total_freshmen : ℕ := 210
def total_sophomores : ℕ := 270
def total_seniors : ℕ := 300
def selected_freshmen : ℕ := 7

-- Problem statement to prove
theorem seniors_selected_correct : 
  (total_seniors / (total_freshmen / selected_freshmen)) = 10 := 
by 
  sorry

end seniors_selected_correct_l124_124105


namespace base7_divisible_by_19_l124_124404

theorem base7_divisible_by_19 (y : ℕ) (h : y ≤ 6) :
  (7 * y + 247) % 19 = 0 ↔ y = 0 :=
by sorry

end base7_divisible_by_19_l124_124404


namespace total_sheep_l124_124854

-- Define the conditions as hypotheses
variables (Aaron_sheep Beth_sheep : ℕ)
def condition1 := Aaron_sheep = 7 * Beth_sheep
def condition2 := Aaron_sheep = 532
def condition3 := Beth_sheep = 76

-- Assert that under these conditions, the total number of sheep is 608.
theorem total_sheep
  (h1 : condition1 Aaron_sheep Beth_sheep)
  (h2 : condition2 Aaron_sheep)
  (h3 : condition3 Beth_sheep) :
  Aaron_sheep + Beth_sheep = 608 :=
by sorry

end total_sheep_l124_124854


namespace product_three_consecutive_not_power_l124_124971

theorem product_three_consecutive_not_power (n k m : ℕ) (hn : n > 0) (hm : m ≥ 2) : 
  (n-1) * n * (n+1) ≠ k^m :=
by sorry

end product_three_consecutive_not_power_l124_124971


namespace pool_capacity_l124_124364

theorem pool_capacity (C : ℝ) (h1 : C * 0.70 = C * 0.40 + 300)
  (h2 : 300 = C * 0.30) : C = 1000 :=
sorry

end pool_capacity_l124_124364


namespace distribute_candies_l124_124837

theorem distribute_candies (n : ℕ) (h : ∃ m : ℕ, n = 2^m) : 
  ∀ k : ℕ, ∃ i : ℕ, (1 / 2) * i * (i + 1) % n = k :=
sorry

end distribute_candies_l124_124837


namespace sawyer_joined_coaching_l124_124337

variable (daily_fees total_fees : ℕ)
variable (year_not_leap : Prop)
variable (discontinue_day : ℕ)

theorem sawyer_joined_coaching :
  daily_fees = 39 → 
  total_fees = 11895 → 
  year_not_leap → 
  discontinue_day = 307 → 
  ∃ start_day, start_day = 30 := 
by
  intros h_daily_fees h_total_fees h_year_not_leap h_discontinue_day
  sorry

end sawyer_joined_coaching_l124_124337


namespace biscuits_afternoon_eq_40_l124_124968

-- Define the initial conditions given in the problem.
def butter_cookies_afternoon : Nat := 10
def additional_biscuits : Nat := 30

-- Define the number of biscuits based on the initial conditions.
def biscuits_afternoon : Nat := butter_cookies_afternoon + additional_biscuits

-- The statement to prove according to the problem.
theorem biscuits_afternoon_eq_40 : biscuits_afternoon = 40 := by
  -- The proof is to be done, hence we use 'sorry'.
  sorry

end biscuits_afternoon_eq_40_l124_124968


namespace sum_infinite_series_l124_124113

theorem sum_infinite_series : (∑' n : ℕ, (n + 1) / 8^(n + 1)) = 8 / 49 := sorry

end sum_infinite_series_l124_124113


namespace tumblonian_words_count_l124_124457

def numTumblonianWords : ℕ :=
  let alphabet_size := 6
  let max_word_length := 4
  let num_words n := alphabet_size ^ n
  (num_words 1) + (num_words 2) + (num_words 3) + (num_words 4)

theorem tumblonian_words_count : numTumblonianWords = 1554 := by
  sorry

end tumblonian_words_count_l124_124457


namespace h_oplus_h_op_h_equals_h_l124_124401

def op (x y : ℝ) : ℝ := x^3 - y

theorem h_oplus_h_op_h_equals_h (h : ℝ) : op h (op h h) = h := by
  sorry

end h_oplus_h_op_h_equals_h_l124_124401


namespace sample_capacity_is_480_l124_124584

-- Problem conditions
def total_people : ℕ := 500 + 400 + 300
def selection_probability : ℝ := 0.4

-- Statement: Prove that sample capacity n equals 480
theorem sample_capacity_is_480 (n : ℕ) (h : n / total_people = selection_probability) : n = 480 := by
  sorry

end sample_capacity_is_480_l124_124584


namespace triangle_inequality_half_perimeter_l124_124189

theorem triangle_inequality_half_perimeter 
  (a b c : ℝ)
  (h_a : a < b + c)
  (h_b : b < a + c)
  (h_c : c < a + b) :
  a < (a + b + c) / 2 ∧ b < (a + b + c) / 2 ∧ c < (a + b + c) / 2 := 
sorry

end triangle_inequality_half_perimeter_l124_124189


namespace simplify_and_evaluate_expression_l124_124468

-- Define the parameters for m and n.
def m : ℚ := -1 / 3
def n : ℚ := 1 / 2

-- Define the expression to simplify and evaluate.
def complex_expr (m n : ℚ) : ℚ :=
  -2 * (m * n - 3 * m^2) + 3 * (2 * m * n - 5 * m^2)

-- State the theorem that proves the expression equals -5/3.
theorem simplify_and_evaluate_expression :
  complex_expr m n = -5 / 3 :=
by
  sorry

end simplify_and_evaluate_expression_l124_124468


namespace max_net_income_meeting_point_l124_124467

theorem max_net_income_meeting_point :
  let A := (9 : ℝ)
  let B := (6 : ℝ)
  let cost_per_mile := 1
  let payment_per_mile := 2
  ∃ x : ℝ, 
  let AP := Real.sqrt ((x - 9)^2 + 12^2)
  let PB := Real.sqrt ((x - 6)^2 + 3^2)
  let net_income := payment_per_mile * PB - (AP + PB)
  x = -12.5 := 
sorry

end max_net_income_meeting_point_l124_124467


namespace librarian_took_books_l124_124789

-- Define variables and conditions
def total_books : ℕ := 46
def books_per_shelf : ℕ := 4
def shelves_needed : ℕ := 9

-- Define the number of books Oliver has left to put away
def books_left : ℕ := shelves_needed * books_per_shelf

-- Define the number of books the librarian took
def books_taken : ℕ := total_books - books_left

-- State the theorem
theorem librarian_took_books : books_taken = 10 := by
  sorry

end librarian_took_books_l124_124789


namespace sum_series_eq_two_l124_124705

theorem sum_series_eq_two : ∑' n : ℕ, (4 * (n + 1) - 2) / (3 ^ (n + 1)) = 2 :=
sorry

end sum_series_eq_two_l124_124705


namespace circumcenter_rational_l124_124611

theorem circumcenter_rational {a1 b1 a2 b2 a3 b3 : ℚ} :
  ∃ (x y : ℚ), 
    ((x - a1)^2 + (y - b1)^2 = (x - a2)^2 + (y - b2)^2) ∧
    ((x - a1)^2 + (y - b1)^2 = (x - a3)^2 + (y - b3)^2) :=
sorry

end circumcenter_rational_l124_124611


namespace exists_nonidentical_subset_with_common_elements_l124_124180

open Finset

theorem exists_nonidentical_subset_with_common_elements {k n : ℕ} (X : Finset (Fin k)) (A : Finset (Finset (Fin k)))
  (hA : A.card = n)
  (hX : ∀ (A1 A2 : Finset (Fin k)), A1 ∈ A → A2 ∈ A → A1 ≠ A2 → (A1 ∩ A2).nonempty)
  (hn : n < 2 ^ (k - 1)) :
  ∃ (C : Finset (Fin k)), C ∉ A ∧ ∀ (A_i : Finset (Fin k)), A_i ∈ A → (C ∩ A_i).nonempty := sorry

end exists_nonidentical_subset_with_common_elements_l124_124180


namespace hyperbola_asymptotes_l124_124628

theorem hyperbola_asymptotes :
  ∀ (x y : ℝ), (y^2 / 9 - x^2 / 4 = 1 →
  (y = (3 / 2) * x ∨ y = - (3 / 2) * x)) :=
by
  intros x y h
  sorry

end hyperbola_asymptotes_l124_124628


namespace rational_coordinates_of_circumcenter_l124_124615

open Classical

noncomputable theory

theorem rational_coordinates_of_circumcenter
  {a1 b1 a2 b2 a3 b3 : ℚ}
  (h1 : ∃ (x y : ℚ), (x - a1)^2 + (y - b1)^2 = (x - a2)^2 + (y - b2)^2 ∧
                      (x - a1)^2 + (y - b1)^2 = (x - a3)^2 + (y - b3)^2) :
  ∃ (x y : ℚ),
    (x - a1)^2 + (y - b1)^2 = (x - a2)^2 + (y - b2)^2 ∧
    (x - a1)^2 + (y - b1)^2 = (x - a3)^2 + (y - b3)^2 := 
begin
  obtain ⟨x, y, hx⟩ := h1,
  use [x, y],
  exact hx,
end

end rational_coordinates_of_circumcenter_l124_124615


namespace find_m_plus_M_l124_124780

-- Given conditions
def cond1 (x y z : ℝ) := x + y + z = 4
def cond2 (x y z : ℝ) := x^2 + y^2 + z^2 = 6

-- Proof statement: The sum of the smallest and largest possible values of x is 8/3
theorem find_m_plus_M :
  ∀ (x y z : ℝ), cond1 x y z → cond2 x y z → (min (x : ℝ) (max x y) + max (x : ℝ) (min x y) = 8 / 3) :=
by
  sorry

end find_m_plus_M_l124_124780


namespace time_to_fill_tank_l124_124983

theorem time_to_fill_tank (T : ℝ) :
  (1 / 2 * T) + ((1 / 2 * T) / 4) = 10 → T = 16 :=
by { sorry }

end time_to_fill_tank_l124_124983


namespace soccer_balls_per_class_l124_124102

-- Definitions for all conditions in the problem
def elementary_classes_per_school : ℕ := 4
def middle_school_classes_per_school : ℕ := 5
def number_of_schools : ℕ := 2
def total_soccer_balls_donated : ℕ := 90

-- The total number of classes in one school
def classes_per_school : ℕ := elementary_classes_per_school + middle_school_classes_per_school

-- The total number of classes in both schools
def total_classes : ℕ := classes_per_school * number_of_schools

-- Prove that the number of soccer balls donated per class is 5
theorem soccer_balls_per_class : total_soccer_balls_donated / total_classes = 5 :=
  by sorry

end soccer_balls_per_class_l124_124102


namespace solve_first_system_solve_second_system_solve_third_system_l124_124792

-- First system of equations
theorem solve_first_system (x y : ℝ) 
  (h1 : 2*x + 3*y = 16)
  (h2 : x + 4*y = 13) : 
  x = 5 ∧ y = 2 := 
sorry

-- Second system of equations
theorem solve_second_system (x y : ℝ) 
  (h1 : 0.3*x - y = 1)
  (h2 : 0.2*x - 0.5*y = 19) : 
  x = 370 ∧ y = 110 := 
sorry

-- Third system of equations
theorem solve_third_system (x y : ℝ) 
  (h1 : 3 * (x - 1) = y + 5)
  (h2 : (x + 2) / 2 = ((y - 1) / 3) + 1) : 
  x = 6 ∧ y = 10 := 
sorry

end solve_first_system_solve_second_system_solve_third_system_l124_124792


namespace simplify_fraction_multiplication_l124_124339

theorem simplify_fraction_multiplication :
  8 * (15 / 4) * (-40 / 45) = -64 / 9 :=
by
  sorry

end simplify_fraction_multiplication_l124_124339


namespace simplify_sqrt_sum_l124_124041

theorem simplify_sqrt_sum : sqrt 72 + sqrt 32 = 10 * sqrt 2 := sorry

end simplify_sqrt_sum_l124_124041


namespace period_tan_2x_3_l124_124087

noncomputable def period_of_tan_transformed : Real :=
  let period_tan := Real.pi
  let coeff := 2/3
  (period_tan / coeff : Real)

theorem period_tan_2x_3 : period_of_tan_transformed = 3 * Real.pi / 2 :=
  sorry

end period_tan_2x_3_l124_124087


namespace number_of_students_absent_l124_124812

def classes := 18
def students_per_class := 28
def students_present := 496
def students_absent := (classes * students_per_class) - students_present

theorem number_of_students_absent : students_absent = 8 := 
by
  sorry

end number_of_students_absent_l124_124812


namespace hyperbola_eccentricity_condition_l124_124153

theorem hyperbola_eccentricity_condition (m : ℝ) (h : m > 0) : 
  (∃ e : ℝ, e = Real.sqrt (1 + m) ∧ e > Real.sqrt 2) → m > 1 :=
by
  sorry

end hyperbola_eccentricity_condition_l124_124153


namespace maria_remaining_towels_l124_124228

def total_towels_initial := 40 + 44
def towels_given_away := 65

theorem maria_remaining_towels : (total_towels_initial - towels_given_away) = 19 := by
  sorry

end maria_remaining_towels_l124_124228


namespace simplify_sqrt72_add_sqrt32_l124_124060

theorem simplify_sqrt72_add_sqrt32 : (sqrt 72) + (sqrt 32) = 10 * (sqrt 2) :=
by sorry

end simplify_sqrt72_add_sqrt32_l124_124060


namespace smallest_sum_l124_124149

theorem smallest_sum (x y : ℕ) (hx : x ≠ y) (hxy : (1/x:ℚ) + (1/y:ℚ) = 1/15) : x + y = 64 :=
sorry

end smallest_sum_l124_124149


namespace polynomial_no_linear_term_l124_124955

theorem polynomial_no_linear_term (m n : ℝ) :
  (∀ x : ℝ, (x - m) * (x - n) = x^2 + mn → n + m = 0) :=
sorry

end polynomial_no_linear_term_l124_124955


namespace daily_reading_goal_l124_124595

-- Define the problem conditions
def total_days : ℕ := 30
def goal_pages : ℕ := 600
def busy_days_13_16 : ℕ := 4
def busy_days_20_25 : ℕ := 6
def flight_day : ℕ := 1
def flight_pages : ℕ := 100

-- Define the mathematical equivalent proof problem in Lean 4
theorem daily_reading_goal :
  (total_days - busy_days_13_16 - busy_days_20_25 - flight_day) * 27 + flight_pages >= goal_pages :=
by
  sorry

end daily_reading_goal_l124_124595


namespace arithmetic_sequence_problem_l124_124167

theorem arithmetic_sequence_problem 
    (a : ℕ → ℝ)  -- Define the arithmetic sequence as a function from natural numbers to reals
    (a1 : ℝ)  -- Represent a₁ as a1
    (a8 : ℝ)  -- Represent a₈ as a8
    (a9 : ℝ)  -- Represent a₉ as a9
    (a10 : ℝ)  -- Represent a₁₀ as a10
    (a15 : ℝ)  -- Represent a₁₅ as a15
    (h1 : a 1 = a1)  -- Hypothesis that a(1) is represented by a1
    (h8 : a 8 = a8)  -- Hypothesis that a(8) is represented by a8
    (h9 : a 9 = a9)  -- Hypothesis that a(9) is represented by a9
    (h10 : a 10 = a10)  -- Hypothesis that a(10) is represented by a10
    (h15 : a 15 = a15)  -- Hypothesis that a(15) is represented by a15
    (h_condition : a1 + 2 * a8 + a15 = 96)  -- Condition of the problem
    : 2 * a9 - a10 = 24 := 
sorry

end arithmetic_sequence_problem_l124_124167


namespace f_x_minus_1_pass_through_l124_124734

variable (a : ℝ)

def f (x : ℝ) : ℝ := a * x^2 + x

theorem f_x_minus_1_pass_through (a : ℝ) : f a (1 - 1) = 0 :=
by
  -- Proof is omitted here
  sorry

end f_x_minus_1_pass_through_l124_124734


namespace polygon_sides_l124_124519

theorem polygon_sides (perimeter side_length : ℕ) (h₁ : perimeter = 150) (h₂ : side_length = 15): 
  (perimeter / side_length) = 10 := 
by
  -- Here goes the proof part
  sorry

end polygon_sides_l124_124519


namespace find_F_16_l124_124783

noncomputable def F : ℝ → ℝ := sorry

lemma F_condition_1 : ∀ x, (x + 4) ≠ 0 ∧ (x + 2) ≠ 0 → (F (4 * x) / F (x + 4) = 16 - (64 * x + 64) / (x^2 + 6 * x + 8)) := sorry

lemma F_condition_2 : F 8 = 33 := sorry

theorem find_F_16 : F 16 = 136 :=
by
  have h1 := F_condition_1
  have h2 := F_condition_2
  sorry

end find_F_16_l124_124783


namespace diane_head_start_l124_124436

theorem diane_head_start (x : ℝ) :
  (100 - 11.91) / (88.09 + x) = 99.25 / 100 ->
  abs (x - 12.68) < 0.01 := 
by
  sorry

end diane_head_start_l124_124436


namespace paperboy_delivery_count_l124_124878

def no_miss_four_consecutive (n : ℕ) (E : ℕ → ℕ) : Prop :=
  ∀ k > 3, E k = E (k - 1) + E (k - 2) + E (k - 3)

def base_conditions (E : ℕ → ℕ) : Prop :=
  E 1 = 2 ∧ E 2 = 4 ∧ E 3 = 8

theorem paperboy_delivery_count : ∃ (E : ℕ → ℕ), 
  base_conditions E ∧ no_miss_four_consecutive 12 E ∧ E 12 = 1854 :=
by
  sorry

end paperboy_delivery_count_l124_124878


namespace verify_a_l124_124535

def g (x : ℝ) : ℝ := 5 * x - 7

theorem verify_a (a : ℝ) : g a = 0 ↔ a = 7 / 5 := by
  sorry

end verify_a_l124_124535


namespace partial_fraction_decomposition_l124_124547

theorem partial_fraction_decomposition (x : ℝ) :
  (5 * x - 3) / (x^2 - 5 * x - 14) = (32 / 9) / (x - 7) + (13 / 9) / (x + 2) := by
  sorry

end partial_fraction_decomposition_l124_124547


namespace find_k_l124_124861

theorem find_k (angle_BAC : ℝ) (angle_D : ℝ)
  (h1 : 0 < angle_BAC ∧ angle_BAC < π)
  (h2 : 0 < angle_D ∧ angle_D < π)
  (h3 : (π - angle_BAC) / 2 = 3 * angle_D) :
  angle_BAC = (5 / 11) * π :=
by sorry

end find_k_l124_124861


namespace value_of_a_l124_124539

theorem value_of_a :
  ∀ (g : ℝ → ℝ), (∀ x, g x = 5*x - 7) → ∃ a, g a = 0 ∧ a = 7 / 5 :=
by
  sorry

end value_of_a_l124_124539


namespace proofSmallestM_l124_124178

def LeanProb (a b c d e f : ℕ) : Prop :=
  a + b + c + d + e + f = 2512 →
  (0 < a) ∧ (0 < b) ∧ (0 < c) ∧ (0 < d) ∧ (0 < e) ∧ (0 < f) →
  ∃ M, (M = 1005) ∧ (M = max (a+b) (max (b+c) (max (c+d) (max (d+e) (e+f)))))

theorem proofSmallestM (a b c d e f : ℕ) (h1 : a + b + c + d + e + f = 2512) 
(h2 : 0 < a) (h3 : 0 < b) (h4 : 0 < c) (h5 : 0 < d) (h6 : 0 < e) (h7 : 0 < f) : 
  ∃ M, (M = 1005) ∧ (M = max (a+b) (max (b+c) (max (c+d) (max (d+e) (e+f))))):=
by
  sorry

end proofSmallestM_l124_124178


namespace monotonic_on_interval_l124_124796

theorem monotonic_on_interval (k : ℝ) :
  (∀ x y : ℝ, x ≤ y → x ≤ 8 → y ≤ 8 → (4 * x ^ 2 - k * x - 8) ≤ (4 * y ^ 2 - k * y - 8)) ↔ (64 ≤ k) :=
sorry

end monotonic_on_interval_l124_124796


namespace minimum_amount_spent_on_boxes_l124_124667

theorem minimum_amount_spent_on_boxes
  (box_length : ℕ) (box_width : ℕ) (box_height : ℕ) 
  (cost_per_box : ℝ) (total_volume_needed : ℕ) :
  box_length = 20 →
  box_width = 20 →
  box_height = 12 →
  cost_per_box = 0.50 →
  total_volume_needed = 2400000 →
  (total_volume_needed / (box_length * box_width * box_height) * cost_per_box) = 250 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  sorry

end minimum_amount_spent_on_boxes_l124_124667


namespace ratio_of_larger_to_smaller_l124_124026

theorem ratio_of_larger_to_smaller (S L k : ℕ) 
  (hS : S = 32)
  (h_sum : S + L = 96)
  (h_multiple : L = k * S) : L / S = 2 :=
by
  sorry

end ratio_of_larger_to_smaller_l124_124026


namespace reflection_of_P_across_y_axis_l124_124203

-- Define the initial point P as a tuple
def P : ℝ × ℝ := (1, -2)

-- Define the reflection across the y-axis function
def reflect_y_axis (point : ℝ × ℝ) : ℝ × ℝ :=
  (-point.1, point.2)

-- State the theorem that we want to prove
theorem reflection_of_P_across_y_axis :
  reflect_y_axis P = (-1, -2) :=
by
  -- placeholder for the proof steps
  sorry

end reflection_of_P_across_y_axis_l124_124203


namespace water_strider_probability_l124_124839

-- Defining the events A and B, and their probabilities
def A : Prop := true -- Event {the taken water contains water strider a}
def B : Prop := true -- Event {the taken water contains water strider b}

axiom prob_A : Prob A = 0.1
axiom prob_B : Prob B = 0.1

-- Axiom for independence
axiom independence : independent A B

-- Prove that the probability of finding a water strider in the taken water is 0.19
theorem water_strider_probability : Prob (A ∪ B) = 0.19 :=
by
  -- Using the formula for the probability of the union of two independent events
  have h_union : Prob (A ∪ B) = Prob A + Prob B - Prob (A ∩ B) := by sorry
  -- Using the given probabilities and the fact that A and B are independent
  have h_intersection : Prob (A ∩ B) = Prob A * Prob B := by sorry
  rw [h_union, h_intersection]
  exact sorry

end water_strider_probability_l124_124839


namespace trigonometric_identity_l124_124114

theorem trigonometric_identity :
  (1 / Real.cos (40 * Real.pi / 180) - 2 * Real.sqrt 3 / Real.sin (40 * Real.pi / 180)) = -4 * Real.tan (20 * Real.pi / 180) := 
sorry

end trigonometric_identity_l124_124114


namespace imaginary_part_z_is_correct_l124_124754

open Complex

noncomputable def problem_conditions (z : ℂ) : Prop :=
  (3 - 4 * Complex.I) * z = Complex.abs (4 + 3 * Complex.I)

theorem imaginary_part_z_is_correct (z : ℂ) (hz : problem_conditions z) :
  z.im = 4 / 5 :=
sorry

end imaginary_part_z_is_correct_l124_124754


namespace Nina_total_problems_l124_124455

def Ruby_math_problems := 12
def Ruby_reading_problems := 4
def Ruby_science_problems := 5

def Nina_math_problems := 5 * Ruby_math_problems
def Nina_reading_problems := 9 * Ruby_reading_problems
def Nina_science_problems := 3 * Ruby_science_problems

def total_problems := Nina_math_problems + Nina_reading_problems + Nina_science_problems

theorem Nina_total_problems : total_problems = 111 :=
by
  sorry

end Nina_total_problems_l124_124455


namespace no_solution_system_of_inequalities_l124_124510

theorem no_solution_system_of_inequalities :
  ¬ ∃ (x y : ℝ),
    11 * x^2 - 10 * x * y + 3 * y^2 ≤ 3 ∧
    5 * x + y ≤ -10 :=
by
  sorry

end no_solution_system_of_inequalities_l124_124510


namespace swimmer_speeds_l124_124669

variable (a s r : ℝ)
variable (x z y : ℝ)

theorem swimmer_speeds (h : s < r) (h' : r < 100 * s / (50 + s)) :
    (100 * s - 50 * r - r * s) / ((3 * s - r) * a) = x ∧ 
    (100 * s - 50 * r - r * s) / ((r - s) * a) = z := by
    sorry

end swimmer_speeds_l124_124669


namespace correct_answers_l124_124130

open Set Finset

noncomputable def problem (teams_A teams_B : Finset ℕ) (hpart : teams_A ∪ teams_B = {0, 1, 2, 3, 4, 5, 6, 7}) (hw : ∀ t ∈ teams_A, t < 4 ∨ t >= 4 → t < 3 ∨ t >= 3 ) : ℚ :=
  (calc_probability teams_A teams_B)/2

theorem correct_answers:
  ∀ (teams : Finset ℕ) (h_div_in_half : (|teams| = 8) -> ( (teams = teams_A ∪ teams_B) ∧ (|teams_A| = 4) ∧ (|teams_B| = 4))),
  (Probability({ one_group_exactly_two_weak teams_A teams_B }) = 6 / 7) ∧ (Probability({ at_least_two_weak teams_A }) = 1 / 2) :=
begin
  sorry
end

end correct_answers_l124_124130


namespace triangles_congruence_l124_124768

theorem triangles_congruence (A_1 B_1 C_1 A_2 B_2 C_2 : ℝ)
  (angle_A1 angle_B1 angle_C1 angle_A2 angle_B2 angle_C2 : ℝ)
  (h_side1 : A_1 = A_2) 
  (h_side2 : B_1 = B_2)
  (h_angle1 : angle_A1 = angle_A2)
  (h_angle2 : angle_B1 = angle_B2)
  (h_angle3 : angle_C1 = angle_C2) : 
  ¬((A_1 = C_1) ∧ (B_1 = C_2) ∧ (angle_A1 = angle_B2) ∧ (angle_B1 = angle_A2) ∧ (angle_C1 = angle_B2) → 
     (A_1 = A_2) ∧ (B_1 = B_2) ∧ (C_1 = C_2)) :=
by {
  sorry
}

end triangles_congruence_l124_124768


namespace parabola_value_f_l124_124609

theorem parabola_value_f (d e f : ℝ) :
  (∀ y : ℝ, x = d * y ^ 2 + e * y + f) →
  (∀ x y : ℝ, (x + 3) = d * (y - 1) ^ 2) →
  (x = -1 ∧ y = 3) →
  y = 0 →
  f = -2.5 :=
sorry

end parabola_value_f_l124_124609


namespace valid_for_expression_c_l124_124442

def expression_a_defined (x : ℝ) : Prop := x ≠ 2
def expression_b_defined (x : ℝ) : Prop := x ≠ 3
def expression_c_defined (x : ℝ) : Prop := x ≥ 2
def expression_d_defined (x : ℝ) : Prop := x ≥ 3

theorem valid_for_expression_c :
  (expression_a_defined 2 = false ∧ expression_a_defined 3 = true) ∧
  (expression_b_defined 2 = true ∧ expression_b_defined 3 = false) ∧
  (expression_c_defined 2 = true ∧ expression_c_defined 3 = true) ∧
  (expression_d_defined 2 = false ∧ expression_d_defined 3 = true) ∧
  (expression_c_defined 2 = true ∧ expression_c_defined 3 = true) := by
  sorry

end valid_for_expression_c_l124_124442


namespace pond_fish_count_l124_124304

theorem pond_fish_count :
  (∃ (N : ℕ), (2 / 50 : ℚ) = (40 / N : ℚ)) → N = 1000 :=
by
  sorry

end pond_fish_count_l124_124304


namespace cans_restocked_after_second_day_l124_124391

theorem cans_restocked_after_second_day :
  let initial_cans := 2000
  let first_day_taken := 500 
  let first_day_restock := 1500
  let second_day_taken := 1000 * 2
  let total_given_away := 2500
  let remaining_after_second_day_before_restock := initial_cans - first_day_taken + first_day_restock - second_day_taken
  (total_given_away - remaining_after_second_day_before_restock) = 1500 := 
by {
  sorry
}

end cans_restocked_after_second_day_l124_124391


namespace pencil_count_l124_124459

theorem pencil_count (a : ℕ) :
  200 ≤ a ∧ a ≤ 300 ∧ a % 10 = 7 ∧ a % 12 = 9 → (a = 237 ∨ a = 297) :=
by sorry

end pencil_count_l124_124459


namespace correct_statements_l124_124415

variable (a b : ℝ)

theorem correct_statements (hab : a * b > 0) :
  (|a + b| > |a| ∧ |a + b| > |a - b|) ∧ (¬ (|a + b| < |b|)) ∧ (¬ (|a + b| < |a - b|)) :=
by
  -- The proof is omitted as per instructions
  sorry

end correct_statements_l124_124415


namespace simplify_sqrt_sum_l124_124044

theorem simplify_sqrt_sum : sqrt 72 + sqrt 32 = 10 * sqrt 2 := sorry

end simplify_sqrt_sum_l124_124044


namespace tens_digit_11_pow_2045_l124_124879

theorem tens_digit_11_pow_2045 : 
    ((11 ^ 2045) % 100) / 10 % 10 = 5 :=
by
    sorry

end tens_digit_11_pow_2045_l124_124879


namespace other_number_eq_462_l124_124803

theorem other_number_eq_462 (a b : ℕ) 
  (lcm_ab : Nat.lcm a b = 4620) 
  (gcd_ab : Nat.gcd a b = 21) 
  (a_eq : a = 210) : b = 462 := 
by
  sorry

end other_number_eq_462_l124_124803


namespace dividend_ratio_l124_124682

theorem dividend_ratio
  (expected_earnings_per_share : ℝ)
  (actual_earnings_per_share : ℝ)
  (dividend_per_share_increase : ℝ)
  (threshold_earnings_increase : ℝ)
  (shares_owned : ℕ)
  (h_expected_earnings : expected_earnings_per_share = 0.8)
  (h_actual_earnings : actual_earnings_per_share = 1.1)
  (h_dividend_increase : dividend_per_share_increase = 0.04)
  (h_threshold_increase : threshold_earnings_increase = 0.1)
  (h_shares_owned : shares_owned = 100)
  : (shares_owned * (expected_earnings_per_share + 
      (actual_earnings_per_share - expected_earnings_per_share) / threshold_earnings_increase * dividend_per_share_increase)) /
    (shares_owned * actual_earnings_per_share) = 46 / 55 :=
by
  sorry

end dividend_ratio_l124_124682


namespace ratio_diminished_to_total_l124_124685

-- Definitions related to the conditions
def N := 240
def P := 60
def fifth_part_increased (N : ℕ) : ℕ := (N / 5) + 6
def part_diminished (P : ℕ) : ℕ := P - 6

-- The proof problem statement
theorem ratio_diminished_to_total 
  (h1 : fifth_part_increased N = part_diminished P) : 
  (P - 6) / N = 9 / 40 :=
by sorry

end ratio_diminished_to_total_l124_124685


namespace sum_of_roots_eq_three_l124_124781

theorem sum_of_roots_eq_three (x1 x2 : ℝ) 
  (h1 : x1^2 - 3*x1 + 2 = 0)
  (h2 : x2^2 - 3*x2 + 2 = 0) 
  (h3 : x1 ≠ x2) : 
  x1 + x2 = 3 := 
sorry

end sum_of_roots_eq_three_l124_124781


namespace collinear_condition_perpendicular_condition_l124_124902

namespace Vectors

-- Definitions for vectors a and b
def a : ℝ × ℝ := (4, -2)
def b (x : ℝ) : ℝ × ℝ := (x, 1)

-- Collinear condition
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2

-- Perpendicular condition
def perpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

-- Proof statement for collinear condition
theorem collinear_condition (x : ℝ) (h : collinear a (b x)) : x = -2 := sorry

-- Proof statement for perpendicular condition
theorem perpendicular_condition (x : ℝ) (h : perpendicular a (b x)) : x = 1 / 2 := sorry

end Vectors

end collinear_condition_perpendicular_condition_l124_124902


namespace minimum_value_ineq_l124_124773

noncomputable def problem_statement (a b c : ℝ) (h : a + b + c = 3) : Prop :=
  (a > 0) ∧ (b > 0) ∧ (c > 0) → (1 / (a + b) + 1 / (b + c) + 1 / (c + a) ≥ 3 / 2)

theorem minimum_value_ineq (a b c : ℝ) (h : a + b + c = 3) : problem_statement a b c h :=
  sorry

end minimum_value_ineq_l124_124773


namespace min_sum_xy_l124_124134

theorem min_sum_xy (x y : ℕ) (hx : x ≠ y) (hcond : ↑(1 / x) + ↑(1 / y) = 1 / 15) : x + y = 64 :=
sorry

end min_sum_xy_l124_124134


namespace coprime_mk_has_distinct_products_not_coprime_mk_has_congruent_products_l124_124672

def coprime_distinct_remainders (m k : ℕ) (coprime_mk : Nat.gcd m k = 1) : Prop :=
  ∃ (a : Fin m → ℤ) (b : Fin k → ℤ),
    (∀ (i : Fin m) (j : Fin k), ∀ (s : Fin m) (t : Fin k),
      (i ≠ s ∨ j ≠ t) → (a i * b j) % (m * k) ≠ (a s * b t) % (m * k))

def not_coprime_congruent_product (m k : ℕ) (not_coprime_mk : Nat.gcd m k > 1) : Prop :=
  ∀ (a : Fin m → ℤ) (b : Fin k → ℤ),
    ∃ (i : Fin m) (j : Fin k) (s : Fin m) (t : Fin k),
      (i ≠ s ∨ j ≠ t) ∧ (a i * b j) % (m * k) = (a s * b t) % (m * k)

-- Example statement to assert the existence of the above properties
theorem coprime_mk_has_distinct_products 
  (m k : ℕ) (coprime_mk : Nat.gcd m k = 1) : coprime_distinct_remainders m k coprime_mk :=
sorry

theorem not_coprime_mk_has_congruent_products 
  (m k : ℕ) (not_coprime_mk : Nat.gcd m k > 1) : not_coprime_congruent_product m k not_coprime_mk :=
sorry

end coprime_mk_has_distinct_products_not_coprime_mk_has_congruent_products_l124_124672


namespace circumcenter_is_rational_l124_124622

theorem circumcenter_is_rational (a1 a2 a3 b1 b2 b3 : ℚ)
  (h1 : (a2 - a1) ≠ 0 ∨ (b2 - b1) ≠ 0)
  (h2 : (a3 - a1) ≠ 0 ∨ (b3 - b1) ≠ 0) :
  ∃ x y : ℚ,
    ((a2 - a1) * x + (b2 - b1) * y = (a2^2 - a1^2 + b2^2 - b1^2) / 2) ∧
    ((a3 - a1) * x + (b3 - b1) * y = (a3^2 - a1^2 + b3^2 - b1^2) / 2) :=
begin
  -- proof goes here
  sorry,
end

end circumcenter_is_rational_l124_124622


namespace six_digits_sum_l124_124061

theorem six_digits_sum 
  (a b c d e f g : ℕ) 
  (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d) (h4 : a ≠ e) (h5 : a ≠ f) (h6 : a ≠ g)
  (h7 : b ≠ c) (h8 : b ≠ d) (h9 : b ≠ e) (h10 : b ≠ f) (h11 : b ≠ g)
  (h12 : c ≠ d) (h13 : c ≠ e) (h14 : c ≠ f) (h15 : c ≠ g)
  (h16 : d ≠ e) (h17 : d ≠ f) (h18 : d ≠ g)
  (h19 : e ≠ f) (h20 : e ≠ g)
  (h21 : f ≠ g)
  (h22 : 2 ≤ a) (h23 : a ≤ 9) 
  (h24 : 2 ≤ b) (h25 : b ≤ 9) 
  (h26 : 2 ≤ c) (h27 : c ≤ 9)
  (h28 : 2 ≤ d) (h29 : d ≤ 9)
  (h30 : 2 ≤ e) (h31 : e ≤ 9)
  (h32 : 2 ≤ f) (h33 : f ≤ 9)
  (h34 : 2 ≤ g) (h35 : g ≤ 9)
  (h36 : a + b + c = 25)
  (h37 : d + e + f + g = 15)
  (h38 : b = e) :
  a + b + c + d + f + g = 31 := 
sorry

end six_digits_sum_l124_124061


namespace zachary_pushups_l124_124399

theorem zachary_pushups (d z : ℕ) (h1 : d = z + 30) (h2 : d = 37) : z = 7 := by
  sorry

end zachary_pushups_l124_124399


namespace num_subsets_containing_6_l124_124739

open Finset

-- Define the set S
def S : Finset ℕ := {1, 2, 3, 4, 5, 6}

-- Define the subset containing number 6
def subsets_with_6 (s : Finset ℕ) : Finset (Finset ℕ) :=
  s.powerset.filter (λ x => 6 ∈ x)

-- Theorem: The number of subsets of {1, 2, 3, 4, 5, 6} containing the number 6 is 32
theorem num_subsets_containing_6 : (subsets_with_6 S).card = 32 := by
  sorry

end num_subsets_containing_6_l124_124739


namespace find_c_and_d_l124_124598

noncomputable def N : Matrix (Fin 2) (Fin 2) ℚ := ![![3, 0], ![2, -4]]
noncomputable def I : Matrix (Fin 2) (Fin 2) ℚ := 1 -- Identity matrix

def c := (1 : ℚ) / 36
def d := - (1 : ℚ) / 12

theorem find_c_and_d :
  (N⁻¹ = c • (N * N) + d • I) := by
  sorry

end find_c_and_d_l124_124598


namespace isosceles_triangle_perimeter_l124_124347

theorem isosceles_triangle_perimeter (a b : ℕ) (h1 : a = 4 ∨ a = 7) (h2 : b = 4 ∨ b = 7) (h3 : a ≠ b) :
  (a + a + b = 15 ∨ a + a + b = 18) ∨ (a + b + b = 15 ∨ a + b + b = 18) :=
sorry

end isosceles_triangle_perimeter_l124_124347


namespace total_clothes_l124_124989

-- Defining the conditions
def shirts := 12
def pants := 5 * shirts
def shorts := (1 / 4) * pants

-- Theorem to prove the total number of pieces of clothes
theorem total_clothes : shirts + pants + shorts = 87 := by
  -- using sorry to skip the proof
  sorry

end total_clothes_l124_124989


namespace probability_of_black_ball_l124_124302

/-- Let the probability of drawing a red ball be 0.42, and the probability of drawing a white ball be 0.28. Prove that the probability of drawing a black ball is 0.3. -/
theorem probability_of_black_ball (p_red p_white p_black : ℝ) (h1 : p_red = 0.42) (h2 : p_white = 0.28) (h3 : p_red + p_white + p_black = 1) : p_black = 0.3 :=
by
  sorry

end probability_of_black_ball_l124_124302


namespace problem_solution_l124_124701

theorem problem_solution :
  -20 + 7 * (8 - 2 / 2) = 29 :=
by 
  sorry

end problem_solution_l124_124701


namespace students_not_making_the_cut_l124_124953

-- Define the total number of girls, boys, and the number of students called back
def number_of_girls : ℕ := 39
def number_of_boys : ℕ := 4
def students_called_back : ℕ := 26

-- Define the total number of students trying out
def total_students : ℕ := number_of_girls + number_of_boys

-- Formulate the problem statement as a theorem
theorem students_not_making_the_cut : total_students - students_called_back = 17 := 
by 
  -- Omitted proof, just the statement
  sorry

end students_not_making_the_cut_l124_124953


namespace roots_magnitude_order_l124_124901

theorem roots_magnitude_order (m : ℝ) (a b c d : ℝ)
  (h1 : m > 0)
  (h2 : a ^ 2 - m * a - 1 = 0)
  (h3 : b ^ 2 - m * b - 1 = 0)
  (h4 : c ^ 2 + m * c - 1 = 0)
  (h5 : d ^ 2 + m * d - 1 = 0)
  (ha_pos : a > 0) (hb_neg : b < 0)
  (hc_pos : c > 0) (hd_neg : d < 0) :
  |a| > |c| ∧ |c| > |b| ∧ |b| > |d| :=
sorry

end roots_magnitude_order_l124_124901


namespace problem_part1_problem_part2_l124_124417

theorem problem_part1 (x y : ℝ) (h1 : x - 2 * y = 3) (h2 : x^2 - 2 * x * y + 4 * y^2 = 11) :
  x * y = 1 :=
sorry

theorem problem_part2 (x y : ℝ) (h1 : x - 2 * y = 3) (h2 : x^2 - 2 * x * y + 4 * y^2 = 11) :
  x^2 * y - 2 * x * y^2 = 3 :=
sorry

end problem_part1_problem_part2_l124_124417


namespace train_length_l124_124985

open Real

/--
A train of a certain length can cross an electric pole in 30 sec with a speed of 43.2 km/h.
Prove that the length of the train is 360 meters.
-/
theorem train_length (t : ℝ) (v_kmh : ℝ) (length : ℝ) 
  (h_time : t = 30) 
  (h_speed_kmh : v_kmh = 43.2) 
  (h_length : length = v_kmh * (t * (1000 / 3600))) : 
  length = 360 := 
by
  -- skip the actual proof steps
  sorry

end train_length_l124_124985


namespace complex_product_polar_form_l124_124390

theorem complex_product_polar_form :
  ∃ r θ, r > 0 ∧ 0 ≤ θ ∧ θ < 360 ∧ 
  (r = 12 ∧ θ = 245) :=
by
  sorry

end complex_product_polar_form_l124_124390


namespace smallest_prime_p_l124_124496

theorem smallest_prime_p 
  (p q r : ℕ) 
  (h1 : Nat.Prime p) 
  (h2 : Nat.Prime q) 
  (h3 : r > 0) 
  (h4 : p + q = r) 
  (h5 : q < p) 
  (h6 : q = 2) 
  (h7 : Nat.Prime r)  
  : p = 3 := 
sorry

end smallest_prime_p_l124_124496


namespace roger_steps_time_l124_124926

theorem roger_steps_time (steps_per_30_min : ℕ := 2000) (time_for_2000_steps : ℕ := 30) (goal_steps : ℕ := 10000) : 
  (goal_steps * time_for_2000_steps) / steps_per_30_min = 150 :=
by 
  -- This is the statement. Proof is omitted as per instruction.
  sorry

end roger_steps_time_l124_124926


namespace edge_ratio_of_cubes_l124_124832

theorem edge_ratio_of_cubes (a b : ℝ) (h : a^3 / b^3 = 27 / 8) : a / b = 3 / 2 :=
by
  sorry

end edge_ratio_of_cubes_l124_124832


namespace range_of_a_for_p_range_of_a_for_p_and_q_l124_124420

variable (a : ℝ)

/-- For any x ∈ ℝ, ax^2 - x + 3 > 0 if and only if a > 1/12 -/
def condition_p : Prop := ∀ x : ℝ, a * x^2 - x + 3 > 0

/-- There exists x ∈ [1, 2] such that 2^x * a ≥ 1 -/
def condition_q : Prop := ∃ x : ℝ, 1 ≤ x ∧ x ≤ 2 ∧ a * 2^x ≥ 1

/-- Theorem (1): The range of values for a such that condition_p holds true is (1/12, +∞) -/
theorem range_of_a_for_p (h : condition_p a) : a > 1/12 :=
sorry

/-- Theorem (2): The range of values for a such that condition_p and condition_q have different truth values is (1/12, 1/4) -/
theorem range_of_a_for_p_and_q (h₁ : condition_p a) (h₂ : ¬condition_q a) : 1/12 < a ∧ a < 1/4 :=
sorry

end range_of_a_for_p_range_of_a_for_p_and_q_l124_124420


namespace find_distance_from_home_to_airport_l124_124400

variable (d t : ℝ)

-- Conditions
def condition1 := d = 40 * (t + 0.75)
def condition2 := d - 40 = 60 * (t - 1.25)

-- Proof statement
theorem find_distance_from_home_to_airport (hd : condition1 d t) (ht : condition2 d t) : d = 160 :=
by
  sorry

end find_distance_from_home_to_airport_l124_124400


namespace largest_angle_in_pentagon_l124_124763

def pentagon_angle_sum : ℝ := 540

def angle_A : ℝ := 70
def angle_B : ℝ := 90
def angle_C (x : ℝ) : ℝ := x
def angle_D (x : ℝ) : ℝ := x
def angle_E (x : ℝ) : ℝ := 3 * x - 10

theorem largest_angle_in_pentagon
  (x : ℝ)
  (h_sum : angle_A + angle_B + angle_C x + angle_D x + angle_E x = pentagon_angle_sum) :
  angle_E x = 224 :=
sorry

end largest_angle_in_pentagon_l124_124763


namespace perpendicular_lines_condition_l124_124154

theorem perpendicular_lines_condition (A1 B1 C1 A2 B2 C2 : ℝ) :
  (A1 * A2 + B1 * B2 = 0) ↔ (A1 * A2) / (B1 * B2) = -1 := sorry

end perpendicular_lines_condition_l124_124154


namespace range_of_m_l124_124425

noncomputable def setA : Set ℝ := { x : ℝ | -3 ≤ x ∧ x ≤ 4 }
noncomputable def setB (m : ℝ) : Set ℝ := { x : ℝ | m + 1 ≤ x ∧ x ≤ 2 * m - 1 }

theorem range_of_m (m : ℝ) : (setB m ⊆ setA) ↔ (m ∈ Iic (5 / 2)) := by
  sorry

end range_of_m_l124_124425


namespace second_number_multiple_of_seven_l124_124798

theorem second_number_multiple_of_seven (x : ℕ) (h : gcd (gcd 105 x) 2436 = 7) : 7 ∣ x :=
sorry

end second_number_multiple_of_seven_l124_124798


namespace kelly_total_apples_l124_124446

variable (initial_apples : ℕ) (additional_apples : ℕ)

theorem kelly_total_apples (h1 : initial_apples = 56) (h2 : additional_apples = 49) :
  initial_apples + additional_apples = 105 :=
by
  sorry

end kelly_total_apples_l124_124446


namespace total_amount_paid_l124_124366

theorem total_amount_paid (grapes_kg mangoes_kg rate_grapes rate_mangoes : ℕ) 
    (h1 : grapes_kg = 8) (h2 : mangoes_kg = 8) 
    (h3 : rate_grapes = 70) (h4 : rate_mangoes = 55) : 
    (grapes_kg * rate_grapes + mangoes_kg * rate_mangoes) = 1000 :=
by
  sorry

end total_amount_paid_l124_124366


namespace symmetric_circle_eqn_l124_124342

theorem symmetric_circle_eqn (x y : ℝ) :
  (∃ (x0 y0 : ℝ), (x - 2)^2 + (y - 2)^2 = 7 ∧ x + y = 2) → x^2 + y^2 = 7 :=
by
  sorry

end symmetric_circle_eqn_l124_124342


namespace minimum_ribbon_length_l124_124952

def side_length : ℚ := 13 / 12

def perimeter_of_equilateral_triangle (a : ℚ) : ℚ := 3 * a

theorem minimum_ribbon_length :
  perimeter_of_equilateral_triangle side_length = 3.25 := 
by
  sorry

end minimum_ribbon_length_l124_124952


namespace balance_relationship_l124_124164

theorem balance_relationship (x : ℕ) (hx : 0 ≤ x ∧ x ≤ 5) : 
  ∃ y : ℝ, y = 200 - 36 * x := 
sorry

end balance_relationship_l124_124164


namespace circle_and_line_cartesian_equations_l124_124305

noncomputable def circle_in_cartesian : (ℝ × ℝ) × ℝ → (ℝ × ℝ) → Prop
| ((r, θ), R), (x, y) => (x - r * cos(θ))^2 + (y - r * sin(θ))^2 = R^2

noncomputable def line_in_cartesian : ℝ → (ℝ × ℝ) → Prop
| θ, (x, y) => y = x * tan(θ)

theorem circle_and_line_cartesian_equations :
    (circle_in_cartesian ((sqrt 2, π / 4), sqrt 3) (1, 1)) ∧
    (line_in_cartesian (π / 4) (x, y)) ↔
    ((x - 1)^2 + (y - 1)^2 = 3 ∧ y = x) :=
by
  sorry

end circle_and_line_cartesian_equations_l124_124305


namespace sqrt_sum_l124_124047

theorem sqrt_sum (a b : ℕ) (ha : a = 72) (hb : b = 32) : 
  Real.sqrt a + Real.sqrt b = 10 * Real.sqrt 2 := 
by 
  rw [ha, hb] 
  -- Insert any further required simplifications as a formal proof or leave it abstracted.
  exact sorry -- skipping the proof to satisfy this step.

end sqrt_sum_l124_124047


namespace scientific_notation_of_122254_l124_124186

theorem scientific_notation_of_122254 :
  122254 = 1.22254 * 10^5 :=
sorry

end scientific_notation_of_122254_l124_124186


namespace question_true_l124_124449
noncomputable def a := (1/2) * Real.cos (7 * Real.pi / 180) - (Real.sqrt 3 / 2) * Real.sin (7 * Real.pi / 180)
noncomputable def b := (2 * Real.tan (12 * Real.pi / 180)) / (1 + Real.tan (12 * Real.pi / 180)^2)
noncomputable def c := Real.sqrt ((1 - Real.cos (44 * Real.pi / 180)) / 2)

theorem question_true :
  b > a ∧ a > c :=
by
  sorry

end question_true_l124_124449


namespace find_third_month_sale_l124_124374

def sales_first_month : ℕ := 3435
def sales_second_month : ℕ := 3927
def sales_fourth_month : ℕ := 4230
def sales_fifth_month : ℕ := 3562
def sales_sixth_month : ℕ := 1991
def required_average_sale : ℕ := 3500

theorem find_third_month_sale (S3 : ℕ) :
  (sales_first_month + sales_second_month + S3 + sales_fourth_month + sales_fifth_month + sales_sixth_month) / 6 = required_average_sale →
  S3 = 3855 := by
  sorry

end find_third_month_sale_l124_124374


namespace simplify_expression_l124_124032

theorem simplify_expression (b : ℝ) : 3 * b * (3 * b ^ 2 + 2 * b) - 4 * b ^ 2 = 9 * b ^ 3 + 2 * b ^ 2 :=
by
  sorry

end simplify_expression_l124_124032


namespace find_angle_A_find_area_l124_124594

-- Define the geometric and trigonometric conditions of the triangle
def triangle (A B C a b c : ℝ) :=
  a = 4 * Real.sqrt 3 ∧ b + c = 8 ∧
  2 * Real.sin A * Real.cos B + Real.sin B = 2 * Real.sin C

-- Prove angle A is 60 degrees
theorem find_angle_A (A B C a b c : ℝ) 
  (h : triangle A B C a b c) : A = Real.pi / 3 := sorry

-- Prove the area of triangle ABC is 4 * sqrt(3) / 3
theorem find_area (A B C a b c : ℝ) 
  (h : triangle A B C a b c) : 
  (1 / 2) * (a * b * Real.sin C) = (4 * Real.sqrt 3) / 3 := sorry

end find_angle_A_find_area_l124_124594


namespace find_min_n_l124_124287

variable (a : Nat → Int)
variable (S : Nat → Int)
variable (d : Nat)
variable (n : Nat)

-- Definitions based on given conditions
def arithmetic_sequence (a : Nat → Int) (d : Nat) : Prop :=
  ∀ n, a (n + 1) = a n + d

def a1_eq_neg3 (a : Nat → Int) : Prop :=
  a 1 = -3

def condition (a : Nat → Int) (d : Nat) : Prop :=
  11 * a 5 = 5 * a 8

-- Correct answer condition
def minimized_sum_condition (a : Nat → Int) (S : Nat → Int) (d : Nat) (n : Nat) : Prop :=
  S n ≤ S (n + 1)

theorem find_min_n (a : Nat → Int) (S : Nat → Int) (d : Nat) :
  arithmetic_sequence a d ->
  a1_eq_neg3 a ->
  condition a 2 ->
  minimized_sum_condition a S 2 2 :=
by
  sorry

end find_min_n_l124_124287


namespace eval_expression_pow_i_l124_124881

theorem eval_expression_pow_i :
  i^(12345 : ℤ) + i^(12346 : ℤ) + i^(12347 : ℤ) + i^(12348 : ℤ) = (0 : ℂ) :=
by
  -- Since this statement doesn't need the full proof, we use sorry to leave it open 
  sorry

end eval_expression_pow_i_l124_124881


namespace sales_tax_difference_l124_124699

theorem sales_tax_difference (price : ℝ) (rate1 rate2 : ℝ) : 
  rate1 = 0.085 → rate2 = 0.07 → price = 50 → 
  (price * rate1 - price * rate2) = 0.75 := 
by 
  intros h_rate1 h_rate2 h_price
  rw [h_rate1, h_rate2, h_price] 
  simp
  sorry

end sales_tax_difference_l124_124699


namespace initial_people_count_l124_124722

theorem initial_people_count (x : ℕ) 
  (h1 : (x + 15) % 5 = 0)
  (h2 : (x + 15) / 5 = 12) : 
  x = 45 := 
by
  sorry

end initial_people_count_l124_124722


namespace number_of_true_propositions_l124_124860

-- Definitions based on the problem
def proposition1 (α β : ℝ) : Prop := (α + β = 180) → (α + β = 90)
def proposition2 (α β γ δ : ℝ) : Prop := (α = β) → (γ = δ)
def proposition3 (α β γ δ : ℝ) : Prop := (α = β) → (γ = δ)

-- Proof problem statement
theorem number_of_true_propositions : ∃ n : ℕ, n = 2 :=
by
  let p1 := false
  let p2 := false
  let p3 := true
  existsi (if p3 then 1 else 0 + if p2 then 1 else 0 + if p1 then 1 else 0)
  simp
  sorry

end number_of_true_propositions_l124_124860


namespace project_hours_l124_124227

variable (K P M : ℕ)

theorem project_hours
  (h1 : P + K + M = 144)
  (h2 : P = 2 * K)
  (h3 : P = M / 3) :
  M - K = 80 :=
sorry

end project_hours_l124_124227


namespace polynomial_divisible_by_24_l124_124605

-- Defining the function
def f (n : ℕ) : ℕ :=
n^4 + 2*n^3 + 11*n^2 + 10*n

-- Statement of the theorem
theorem polynomial_divisible_by_24 (n : ℕ) (h : n > 0) : f n % 24 = 0 :=
sorry

end polynomial_divisible_by_24_l124_124605


namespace subsets_containing_six_l124_124744

theorem subsets_containing_six : 
  let S := {1, 2, 3, 4, 5, 6}
  in set.count (λ s, 6 ∈ s ∧ s ⊆ S) = 32 :=
sorry

end subsets_containing_six_l124_124744


namespace carmen_distance_from_start_l124_124526

noncomputable def distance_carmen (start : ℝ × ℝ) : ℝ :=
  let B := (start.1, start.2 + 3)
  let C := (B.1 + 8 * Real.cos (Real.pi / 4), B.2 + 8 * Real.sin (Real.pi / 4))
  Real.sqrt ((C.1 - start.1)^2 + (C.2 - start.2)^2)

theorem carmen_distance_from_start : 
  distance_carmen (0, 0) = Real.sqrt (73 + 24 * Real.sqrt 2) :=
by
  sorry

end carmen_distance_from_start_l124_124526


namespace largest_divisor_of_n_l124_124508

theorem largest_divisor_of_n 
  (n : ℕ) (h_pos : n > 0) (h_div : 72 ∣ n^2) : 
  ∃ v : ℕ, v = 12 ∧ v ∣ n :=
by
  use 12
  sorry

end largest_divisor_of_n_l124_124508


namespace shots_cost_l124_124265

theorem shots_cost (n_dogs : ℕ) (puppies_per_dog : ℕ) (shots_per_puppy : ℕ) (cost_per_shot : ℕ) :
  n_dogs = 3 →
  puppies_per_dog = 4 →
  shots_per_puppy = 2 →
  cost_per_shot = 5 →
  n_dogs * puppies_per_dog * shots_per_puppy * cost_per_shot = 120 := by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  calc
    3 * 4 * 2 * 5 = 12 * 2 * 5 : by rfl
    ... = 24 * 5 : by rfl
    ... = 120 : by rfl

end shots_cost_l124_124265


namespace find_m_value_l124_124434

def power_function_increasing (m : ℝ) : Prop :=
  (m^2 - m - 1 = 1) ∧ (m^2 - 2*m - 1 > 0)

theorem find_m_value (m : ℝ) (h : power_function_increasing m) : m = -1 :=
  sorry

end find_m_value_l124_124434


namespace subsets_with_six_l124_124741

open Finset

theorem subsets_with_six (S : Finset ℕ) (hS : S = {1, 2, 3, 4, 5, 6}) :
  (S.filter (λ x, x = 6)).powerset.card = 32 :=
by
  rw [hS]
  have T : {1, 2, 3, 4, 5}.powerset = T.powerset := rfl
  sorry

end subsets_with_six_l124_124741


namespace compute_P_2_4_8_l124_124063

noncomputable def P : ℝ → ℝ → ℝ → ℝ := sorry

axiom homogeneity (x y z k : ℝ) : P (k * x) (k * y) (k * z) = (k ^ 4) * P x y z

axiom symmetry (a b c : ℝ) : P a b c = P b c a

axiom zero_cond (a b : ℝ) : P a a b = 0

axiom initial_cond : P 1 2 3 = 1

theorem compute_P_2_4_8 : P 2 4 8 = 56 := sorry

end compute_P_2_4_8_l124_124063


namespace find_abc_l124_124340

theorem find_abc (a b c : ℝ) (h1 : a * (b + c) = 198) (h2 : b * (c + a) = 210) (h3 : c * (a + b) = 222) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) : 
  a * b * c = 1069 :=
by
  sorry

end find_abc_l124_124340


namespace equilateral_triangles_formed_l124_124797

theorem equilateral_triangles_formed :
  ∀ k : ℤ, -8 ≤ k ∧ k ≤ 8 →
  (∃ triangles : ℕ, triangles = 426) :=
by sorry

end equilateral_triangles_formed_l124_124797


namespace ratio_eliminated_to_remaining_l124_124703

theorem ratio_eliminated_to_remaining (initial_racers : ℕ) (final_racers : ℕ)
  (eliminations_1st_segment : ℕ) (eliminations_2nd_segment : ℕ) :
  initial_racers = 100 →
  final_racers = 30 →
  eliminations_1st_segment = 10 →
  eliminations_2nd_segment = initial_racers - eliminations_1st_segment - (initial_racers - eliminations_1st_segment) / 3 - final_racers →
  (eliminations_2nd_segment / (initial_racers - eliminations_1st_segment - (initial_racers - eliminations_1st_segment) / 3)) = 1 / 2 :=
by
  sorry

end ratio_eliminated_to_remaining_l124_124703


namespace decrease_by_150_percent_l124_124505

theorem decrease_by_150_percent (x : ℝ) (h : x = 80) : x - 1.5 * x = -40 :=
by
  sorry

end decrease_by_150_percent_l124_124505


namespace power_mod_8_l124_124500

theorem power_mod_8 (n : ℕ) (h : n % 2 = 0) : 3^n % 8 = 1 :=
by sorry

end power_mod_8_l124_124500


namespace people_happy_correct_l124_124987

-- Define the size and happiness percentage of an institution.
variables (size : ℕ) (happiness_percentage : ℚ)

-- Assume the size is between 100 and 200.
axiom size_range : 100 ≤ size ∧ size ≤ 200

-- Assume the happiness percentage is between 0.6 and 0.95.
axiom happiness_percentage_range : 0.6 ≤ happiness_percentage ∧ happiness_percentage ≤ 0.95

-- Define the number of people made happy at an institution.
def people_made_happy (size : ℕ) (happiness_percentage : ℚ) : ℚ := 
  size * happiness_percentage

-- Theorem stating that the number of people made happy is as expected.
theorem people_happy_correct : 
  ∀ (size : ℕ) (happiness_percentage : ℚ), 
  100 ≤ size → size ≤ 200 → 
  0.6 ≤ happiness_percentage → happiness_percentage ≤ 0.95 → 
  people_made_happy size happiness_percentage = size * happiness_percentage := 
by 
  intros size happiness_percentage hsize1 hsize2 hperc1 hperc2
  unfold people_made_happy
  sorry

end people_happy_correct_l124_124987


namespace almost_monotonic_digits_0_to_8_l124_124525

def binom (n k : ℕ) : ℕ := Nat.choose n k

def almost_monotonic_count : ℕ :=
  let count_nondecreasing := (Finset.range 9).sum (λ n => binom (n + 8) 8)
  2 * count_nondecreasing - 9

theorem almost_monotonic_digits_0_to_8 :
  almost_monotonic_count = 97227 := by
  sorry

end almost_monotonic_digits_0_to_8_l124_124525


namespace no_solution_for_99_l124_124018

theorem no_solution_for_99 :
  ∃ n : ℕ, (¬ ∃ x y : ℕ, 0 < x ∧ 0 < y ∧ 9 * x + 11 * y = n) ∧
  (∀ m : ℕ, n < m → ∃ x y : ℕ, 0 < x ∧ 0 < y ∧ 9 * x + 11 * y = m) ∧
  n = 99 :=
by
  sorry

end no_solution_for_99_l124_124018


namespace distinct_rectangles_l124_124848

theorem distinct_rectangles :
  ∃! (l w : ℝ), l * w = 100 ∧ l + w = 24 :=
sorry

end distinct_rectangles_l124_124848


namespace find_a4_l124_124208

variable (a : ℕ → ℤ)
variable (d : ℤ)

-- Conditions
def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

axiom hyp1 : is_arithmetic_sequence a d
axiom hyp2 : a 5 = 9
axiom hyp3 : a 7 + a 8 = 28

-- Goal
theorem find_a4 : a 4 = 7 :=
by
  sorry

end find_a4_l124_124208


namespace simplify_sqrt72_add_sqrt32_l124_124057

theorem simplify_sqrt72_add_sqrt32 : (sqrt 72) + (sqrt 32) = 10 * (sqrt 2) :=
by sorry

end simplify_sqrt72_add_sqrt32_l124_124057


namespace fraction_addition_l124_124965

theorem fraction_addition :
  let a := (1 : ℚ) / 6
  let b := (5 : ℚ) / 12
  a + b = 7 / 12 :=
by
  let a := (1 : ℚ) / 6
  let b := (5 : ℚ) / 12
  have : a + b = 7 / 12 := sorry
  exact this

end fraction_addition_l124_124965


namespace jacket_final_price_l124_124851

/-- 
The initial price of the jacket is $20, 
the first discount is 40%, and the second discount is 25%. 
We need to prove that the final price of the jacket is $9.
-/
theorem jacket_final_price :
  let initial_price := 20
  let first_discount := 0.40
  let second_discount := 0.25
  let price_after_first_discount := initial_price * (1 - first_discount)
  let final_price := price_after_first_discount * (1 - second_discount)
  final_price = 9 :=
by
  sorry

end jacket_final_price_l124_124851


namespace cannot_make_120_cents_with_6_coins_l124_124194

def Coin := ℕ → ℕ -- represents a number of each type of coin

noncomputable def coin_value (c : Coin) : ℕ :=
  c 0 * 1 + c 1 * 5 + c 2 * 10 + c 3 * 25

def total_coins (c : Coin) : ℕ :=
  c 0 + c 1 + c 2 + c 3

theorem cannot_make_120_cents_with_6_coins (c : Coin) (h1 : total_coins c = 6) :
  coin_value c ≠ 120 :=
sorry

end cannot_make_120_cents_with_6_coins_l124_124194


namespace equilateral_triangle_perimeter_l124_124629

theorem equilateral_triangle_perimeter (x : ℕ) (h : 2 * x = x + 15) : 
  3 * (2 * x) = 90 :=
by
  -- Definitions & hypothesis
  sorry

end equilateral_triangle_perimeter_l124_124629


namespace students_exceed_goldfish_l124_124269

theorem students_exceed_goldfish 
    (num_classrooms : ℕ) 
    (students_per_classroom : ℕ) 
    (goldfish_per_classroom : ℕ) 
    (h1 : num_classrooms = 5) 
    (h2 : students_per_classroom = 20) 
    (h3 : goldfish_per_classroom = 3) 
    : (students_per_classroom * num_classrooms) - (goldfish_per_classroom * num_classrooms) = 85 := by
  sorry

end students_exceed_goldfish_l124_124269


namespace number_of_ways_to_assign_volunteers_l124_124711

/-- Theorem: The number of ways to assign 5 volunteers to 3 venues such that each venue has at least one volunteer is 150. -/
theorem number_of_ways_to_assign_volunteers :
  let total_ways := 3^5
  let subtract_one_empty := 3 * 2^5
  let add_back_two_empty := 3 * 1^5
  (total_ways - subtract_one_empty + add_back_two_empty) = 150 :=
by
  sorry

end number_of_ways_to_assign_volunteers_l124_124711


namespace max_unbounded_xy_sum_l124_124499

theorem max_unbounded_xy_sum (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  ∃ M : ℝ, ∀ z : ℝ, z > 0 → ∃ (x y : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ (xy + 1)^2 + (x - y)^2 > z := 
  sorry

end max_unbounded_xy_sum_l124_124499


namespace metallic_sheet_dimension_l124_124515

theorem metallic_sheet_dimension (x : ℝ) (h₁ : ∀ (l w h : ℝ), l = x - 8 → w = 28 → h = 4 → l * w * h = 4480) : x = 48 :=
sorry

end metallic_sheet_dimension_l124_124515


namespace find_x_l124_124578

theorem find_x (x : ℝ) (h : (2 * x) / 16 = 25) : x = 200 :=
sorry

end find_x_l124_124578


namespace power_mod_8_l124_124501

theorem power_mod_8 (n : ℕ) (h : n % 2 = 0) : 3^n % 8 = 1 :=
by sorry

end power_mod_8_l124_124501


namespace Jean_calls_thursday_l124_124013

theorem Jean_calls_thursday :
  ∃ (thursday_calls : ℕ), thursday_calls = 61 ∧ 
  (∃ (mon tue wed fri : ℕ),
    mon = 35 ∧ 
    tue = 46 ∧ 
    wed = 27 ∧ 
    fri = 31 ∧ 
    (mon + tue + wed + thursday_calls + fri = 40 * 5)) :=
sorry

end Jean_calls_thursday_l124_124013


namespace ellipse_standard_equation_and_point_l124_124561
  
noncomputable def ellipse_equation (x y : ℝ) : Prop :=
  (x^2) / 25 + (y^2) / 9 = 1

def exists_dot_product_zero_point (P : ℝ × ℝ) : Prop :=
  let F1 := (-4, 0)
  let F2 := (4, 0)
  (P.1 + 4) * (P.1 - 4) + P.2 * P.2 = 0

theorem ellipse_standard_equation_and_point :
  ∃ (P : ℝ × ℝ), ellipse_equation P.1 P.2 ∧ exists_dot_product_zero_point P ∧ 
    ((P = ((5 * Real.sqrt 7) / 4, 9 / 4)) ∨ (P = (-(5 * Real.sqrt 7) / 4, 9 / 4)) ∨ 
    (P = ((5 * Real.sqrt 7) / 4, -(9 / 4))) ∨ (P = (-(5 * Real.sqrt 7) / 4, -(9 / 4)))) :=
by 
  sorry

end ellipse_standard_equation_and_point_l124_124561


namespace circumcenter_rational_l124_124625

theorem circumcenter_rational {a1 b1 a2 b2 a3 b3 : ℚ} 
  (h1 : a1 ≠ a2 ∨ b1 ≠ b2) 
  (h2 : a1 ≠ a3 ∨ b1 ≠ b3) 
  (h3 : a2 ≠ a3 ∨ b2 ≠ b3) :
  ∃ (x y : ℚ), 
    (x - a1)^2 + (y - b1)^2 = (x - a2)^2 + (y - b2)^2 ∧
    (x - a1)^2 + (y - b1)^2 = (x - a3)^2 + (y - b3)^2 :=
sorry

end circumcenter_rational_l124_124625


namespace least_prime_factor_of_11_pow4_minus_11_pow3_l124_124821

open Nat

theorem least_prime_factor_of_11_pow4_minus_11_pow3 : 
  Nat.minFac (11^4 - 11^3) = 2 :=
  sorry

end least_prime_factor_of_11_pow4_minus_11_pow3_l124_124821


namespace output_value_of_y_l124_124883

/-- Define the initial conditions -/
def l : ℕ := 2
def m : ℕ := 3
def n : ℕ := 5

/-- Define the function that executes the flowchart operations -/
noncomputable def flowchart_operation (l m n : ℕ) : ℕ := sorry

/-- Main theorem statement -/
theorem output_value_of_y : flowchart_operation l m n = 68 := sorry

end output_value_of_y_l124_124883


namespace find_some_number_l124_124378

theorem find_some_number (n m : ℕ) (h : (n / 20) * (n / m) = 1) (n_eq_40 : n = 40) : m = 2 :=
by
  sorry

end find_some_number_l124_124378


namespace solve_for_m_l124_124723

theorem solve_for_m (a_0 a_1 a_2 a_3 a_4 a_5 m : ℝ)
  (h1 : (x : ℝ) → (x + m)^5 = a_0 + a_1 * (x+1) + a_2 * (x+1)^2 + a_3 * (x+1)^3 + a_4 * (x+1)^4 + a_5 * (x+1)^5)
  (h2 : a_0 + a_1 + a_2 + a_3 + a_4 + a_5 = 32) :
  m = 2 :=
sorry

end solve_for_m_l124_124723


namespace average_salary_correct_l124_124810

def A_salary := 10000
def B_salary := 5000
def C_salary := 11000
def D_salary := 7000
def E_salary := 9000

def total_salary := A_salary + B_salary + C_salary + D_salary + E_salary
def num_individuals := 5

def average_salary := total_salary / num_individuals

theorem average_salary_correct : average_salary = 8600 := by
  sorry

end average_salary_correct_l124_124810


namespace bella_items_l124_124388

theorem bella_items (M F D : ℕ) 
  (h1 : M = 60)
  (h2 : M = 2 * F)
  (h3 : F = D + 20) :
  (7 * M + 7 * F + 7 * D) / 5 = 140 := 
by
  sorry

end bella_items_l124_124388


namespace triangle_at_most_one_obtuse_l124_124354

theorem triangle_at_most_one_obtuse (A B C : ℝ) (h1 : 0 < A ∧ A < 180) (h2 : 0 < B ∧ B < 180) (h3 : 0 < C ∧ C < 180) (h4 : A + B + C = 180) : A ≤ 90 ∨ B ≤ 90 ∨ C ≤ 90 :=
by
  sorry

end triangle_at_most_one_obtuse_l124_124354


namespace solve_ordered_pairs_l124_124552

theorem solve_ordered_pairs (a b : ℕ) (h : a^2 + b^2 = ab * (a + b)) : 
  (a, b) = (1, 1) ∨ (a, b) = (1, 1) :=
by 
  sorry

end solve_ordered_pairs_l124_124552


namespace ordered_pairs_divide_square_sum_l124_124549

theorem ordered_pairs_divide_square_sum :
  { (m, n) : ℕ × ℕ | m > 0 ∧ n > 0 ∧ (mn - 1) ∣ (m^2 + n^2) } = { (1, 2), (1, 3), (2, 1), (3, 1) } := 
sorry

end ordered_pairs_divide_square_sum_l124_124549


namespace y_coordinate_equidistant_l124_124958

theorem y_coordinate_equidistant :
  ∃ y : ℝ, (∀ ptC ptD : ℝ × ℝ, ptC = (-3, 0) → ptD = (4, 5) → 
    dist (0, y) ptC = dist (0, y) ptD) ∧ y = 16 / 5 :=
by
  sorry

end y_coordinate_equidistant_l124_124958


namespace batsman_average_17th_innings_l124_124512

theorem batsman_average_17th_innings:
  ∀ (A : ℝ), 
  (16 * A + 85 = 17 * (A + 3)) →
  (A + 3 = 37) :=
by
  intros A h
  sorry

end batsman_average_17th_innings_l124_124512


namespace part_1_part_2_l124_124283

-- Define proposition p
def proposition_p (a : ℝ) : Prop :=
  ∃ x : ℝ, x ∈ Set.Ioo (-1 : ℝ) (1 : ℝ) ∧ (x^2 - (a + 2) * x + 2 * a = 0)

-- Proposition q: x₁ and x₂ are two real roots of the equation x^2 - 2mx - 3 = 0
def proposition_q (m x₁ x₂ : ℝ) : Prop :=
  x₁ ^ 2 - 2 * m * x₁ - 3 = 0 ∧ x₂ ^ 2 - 2 * m * x₂ - 3 = 0

-- Inequality condition
def inequality_condition (a m x₁ x₂ : ℝ) : Prop :=
  a ^ 2 - 3 * a ≥ abs (x₁ - x₂)

-- Part 1: If proposition p is true, find the range of the real number a
theorem part_1 (a : ℝ) (h_p : proposition_p a) : -1 < a ∧ a < 1 :=
  sorry

-- Part 2: If exactly one of propositions p or q is true, find the range of the real number a
theorem part_2 (a m x₁ x₂ : ℝ) (h_p_or_q : (proposition_p a ∧ ¬(proposition_q m x₁ x₂)) ∨ (¬(proposition_p a) ∧ (proposition_q m x₁ x₂))) : (a < 1) ∨ (a ≥ 4) :=
  sorry

end part_1_part_2_l124_124283


namespace checkerboard_no_identical_numbers_l124_124844

theorem checkerboard_no_identical_numbers :
  ∀ (i j : ℕ), 1 ≤ i ∧ i ≤ 11 ∧ 1 ≤ j ∧ j ≤ 19 → 19 * (i - 1) + j = 11 * (j - 1) + i → false :=
by
  sorry

end checkerboard_no_identical_numbers_l124_124844


namespace students_at_1544_l124_124111

noncomputable def students_in_lab : Nat := 44

theorem students_at_1544 :
  let initial_students := 20
  let enter_interval := 3
  let enter_students := 4
  let leave_interval := 10
  let leave_students := 8

  ∃ students : Nat,
    students = initial_students
    + (34 / enter_interval) * enter_students
    - (34 / leave_interval) * leave_students
    ∧ students = students_in_lab :=
by
  let initial_students := 20
  let enter_interval := 3
  let enter_students := 4
  let leave_interval := 10
  let leave_students := 8
  use 20 + (34 / 3) * 4 - (34 / 10) * 8
  sorry

end students_at_1544_l124_124111


namespace solve_for_m_l124_124132

theorem solve_for_m 
  (m : ℝ) 
  (h : (m - 1) * 2^2 + 3 * 2 - 5 * m + 4 = 0) : m = 6 :=
sorry

end solve_for_m_l124_124132


namespace fraction_value_l124_124257

theorem fraction_value :
  (2015^2 : ℤ) / (2014^2 + 2016^2 - 2) = (1 : ℚ) / 2 :=
by
  sorry

end fraction_value_l124_124257


namespace determine_pairs_of_positive_integers_l124_124267

open Nat

theorem determine_pairs_of_positive_integers (n p : ℕ) (hp : Nat.Prime p) (hn_le_2p : n ≤ 2 * p)
    (hdiv : (p - 1)^n + 1 ∣ n^(p - 1)) : (n = 1) ∨ (n = 2 ∧ p = 2) ∨ (n = 3 ∧ p = 3) :=
  sorry

end determine_pairs_of_positive_integers_l124_124267


namespace time_to_cross_bridge_l124_124381

-- Define the given conditions in Lean.
def length_of_train : ℝ := 140
def speed_of_train_kmph : ℝ := 45
def length_of_bridge : ℝ := 235

-- Define the conversion factor from km/hr to m/s.
def kmph_to_mps (speed: ℝ) : ℝ := speed * (1000 / 3600)

-- Define the speed in meters per second.
def speed_of_train_mps : ℝ := kmph_to_mps speed_of_train_kmph

-- Define the total distance the train needs to cover.
def total_distance : ℝ := length_of_train + length_of_bridge

-- The goal is to prove that the time to cross the bridge is 30 seconds.
theorem time_to_cross_bridge : total_distance / speed_of_train_mps = 30 := by
  sorry

end time_to_cross_bridge_l124_124381


namespace total_age_l124_124829

theorem total_age (a b c : ℕ) (h1 : a = b + 2) (h2 : b = 2 * c) (h3 : b = 8) : a + b + c = 22 :=
by
  sorry

end total_age_l124_124829


namespace sqrt_prime_geometric_progression_impossible_l124_124702

theorem sqrt_prime_geometric_progression_impossible {p1 p2 p3 : ℕ} (hp1 : Nat.Prime p1) (hp2 : Nat.Prime p2) (hp3 : Nat.Prime p3) (hneq12 : p1 ≠ p2) (hneq23 : p2 ≠ p3) (hneq31 : p3 ≠ p1) :
  ¬ ∃ (a r : ℝ) (n1 n2 n3 : ℤ), (a * r^n1 = Real.sqrt p1) ∧ (a * r^n2 = Real.sqrt p2) ∧ (a * r^n3 = Real.sqrt p3) := sorry

end sqrt_prime_geometric_progression_impossible_l124_124702


namespace range_of_m_l124_124416

noncomputable def p (x : ℝ) : Prop := |x - 3| ≤ 2
noncomputable def q (x m : ℝ) : Prop := (x - m + 1) * (x - m - 1) ≤ 0

theorem range_of_m {m : ℝ} (H : ∀ (x : ℝ), ¬p x → ¬q x m) :
  2 ≤ m ∧ m ≤ 4 :=
sorry

end range_of_m_l124_124416


namespace roots_k_m_l124_124912

theorem roots_k_m (k m : ℝ) 
  (h1 : ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ a + b + c = 11 ∧ a * b + b * c + c * a = k ∧ a * b * c = m)
  : k + m = 52 :=
sorry

end roots_k_m_l124_124912


namespace smallest_sum_l124_124147

theorem smallest_sum (x y : ℕ) (hx : x ≠ y) (hxy : (1/x:ℚ) + (1/y:ℚ) = 1/15) : x + y = 64 :=
sorry

end smallest_sum_l124_124147


namespace minimum_possible_perimeter_l124_124492

theorem minimum_possible_perimeter (a b c : ℤ) (h1 : 2 * a + 8 * c = 2 * b + 10 * c) 
                                  (h2 : 4 * c * (sqrt (a^2 - (4 * c)^2)) = 5 * c * (sqrt (b^2 - (5 * c)^2))) 
                                  (h3 : a - b = c) : 
    2 * a + 8 * c = 740 :=
by
  sorry

end minimum_possible_perimeter_l124_124492


namespace total_shots_cost_l124_124263

def numDogs : ℕ := 3
def puppiesPerDog : ℕ := 4
def shotsPerPuppy : ℕ := 2
def costPerShot : ℕ := 5

theorem total_shots_cost : (numDogs * puppiesPerDog * shotsPerPuppy * costPerShot) = 120 := by
  sorry

end total_shots_cost_l124_124263


namespace sqrt_sum_l124_124048

theorem sqrt_sum (a b : ℕ) (ha : a = 72) (hb : b = 32) : 
  Real.sqrt a + Real.sqrt b = 10 * Real.sqrt 2 := 
by 
  rw [ha, hb] 
  -- Insert any further required simplifications as a formal proof or leave it abstracted.
  exact sorry -- skipping the proof to satisfy this step.

end sqrt_sum_l124_124048


namespace andrew_eggs_l124_124523

def andrew_eggs_problem (a b : ℕ) (half_eggs_given_away : ℚ) (remaining_eggs : ℕ) : Prop :=
  a + b - (a + b) * half_eggs_given_away = remaining_eggs

theorem andrew_eggs :
  andrew_eggs_problem 8 62 (1/2 : ℚ) 35 :=
by
  sorry

end andrew_eggs_l124_124523


namespace weight_of_replaced_person_l124_124202

theorem weight_of_replaced_person 
  (avg_increase : ℝ)
  (num_persons : ℕ)
  (new_person_weight : ℝ)
  (weight_increase : ℝ)
  (new_person_might_be_90_kg : new_person_weight = 90)
  (average_increase_by_3_5_kg : avg_increase = 3.5)
  (group_of_8_persons : num_persons = 8)
  (total_weight_increase_formula : weight_increase = num_persons * avg_increase)
  (weight_of_replaced_person : ℝ)
  (weight_difference_formula : weight_of_replaced_person = new_person_weight - weight_increase) :
  weight_of_replaced_person = 62 :=
sorry

end weight_of_replaced_person_l124_124202


namespace solution_exists_l124_124405

open Real

theorem solution_exists (x : ℝ) (h1 : x > 9) (h2 : sqrt (x - 3 * sqrt (x - 9)) + 3 = sqrt (x + 3 * sqrt (x - 9)) - 3) : x ≥ 18 :=
sorry

end solution_exists_l124_124405


namespace correct_assignment_statement_l124_124655

theorem correct_assignment_statement (n m : ℕ) : 
  ¬ (4 = n) ∧ ¬ (n + 1 = m) ∧ ¬ (m + n = 0) :=
by
  sorry

end correct_assignment_statement_l124_124655


namespace rectangle_area_l124_124687

theorem rectangle_area (d : ℝ) (w : ℝ) (h : w^2 + (3*w)^2 = d^2) : (3 * w ^ 2 = 3 * d ^ 2 / 10) :=
by
  sorry

end rectangle_area_l124_124687


namespace angle_in_third_quadrant_l124_124229

theorem angle_in_third_quadrant (θ : ℝ) (hθ : θ = 2014) : 180 < θ % 360 ∧ θ % 360 < 270 :=
by
  sorry

end angle_in_third_quadrant_l124_124229


namespace local_value_proof_l124_124222

-- Definitions based on the conditions
def face_value_7 : ℕ := 7
def local_value_6_in_7098060 : ℕ := 6000
def product_of_face_value_and_local_value : ℕ := face_value_7 * local_value_6_in_7098060
def local_value_6_in_product : ℕ := 6000

-- Theorem statement
theorem local_value_proof : local_value_6_in_product = 6000 :=
by
  -- Direct restatement of the condition in Lean
  sorry

end local_value_proof_l124_124222


namespace integer_solutions_yk_eq_x2_plus_x_l124_124716

-- Define the problem in Lean
theorem integer_solutions_yk_eq_x2_plus_x (k : ℕ) (hk : k > 1) :
  ∀ (x y : ℤ), y^k = x^2 + x → (x = 0 ∧ y = 0) ∨ (x = -1 ∧ y = 0) :=
by
  sorry

end integer_solutions_yk_eq_x2_plus_x_l124_124716


namespace tony_graduate_degree_years_l124_124642

-- Define the years spent for each degree and the total time
def D1 := 4 -- years for the first degree in science
def D2 := 4 -- years for each of the two additional degrees
def T := 14 -- total years spent in school
def G := 2 -- years spent for the graduate degree in physics

-- Theorem: Given the conditions, prove that Tony spent 2 years on his graduate degree in physics
theorem tony_graduate_degree_years : 
  D1 + 2 * D2 + G = T :=
by
  sorry

end tony_graduate_degree_years_l124_124642


namespace sin_pi_six_minus_two_alpha_l124_124888

theorem sin_pi_six_minus_two_alpha (α : ℝ) (h : Real.sin (π / 3 - α) = 1 / 3) : 
  Real.sin (π / 6 - 2 * α) = - 7 / 9 :=
by
  sorry

end sin_pi_six_minus_two_alpha_l124_124888


namespace fewer_VIP_tickets_sold_l124_124365

variable (V G : ℕ)

-- Definitions: total number of tickets sold and the total revenue from tickets sold
def total_tickets : Prop := V + G = 320
def total_revenue : Prop := 45 * V + 20 * G = 7500

-- Definition of the number of fewer VIP tickets than general admission tickets
def fewer_VIP_tickets : Prop := G - V = 232

-- The theorem to be proven
theorem fewer_VIP_tickets_sold (h1 : total_tickets V G) (h2 : total_revenue V G) : fewer_VIP_tickets V G :=
sorry

end fewer_VIP_tickets_sold_l124_124365


namespace possible_values_l124_124155

theorem possible_values (a b : ℕ → ℕ) (h1 : ∀ n, a n < (a (n + 1)))
  (h2 : ∀ n, b n < (b (n + 1)))
  (h3 : a 10 = b 10)
  (h4 : a 10 < 2017)
  (h5 : ∀ n, a (n + 2) = a (n + 1) + a n)
  (h6 : ∀ n, b (n + 1) = 2 * b n) :
  ∃ (a1 b1 : ℕ), (a 1 = a1) ∧ (b 1 = b1) ∧ (a1 + b1 = 13 ∨ a1 + b1 = 20) := sorry

end possible_values_l124_124155


namespace james_marbles_left_l124_124444

def total_initial_marbles : Nat := 28
def marbles_in_bag_A : Nat := 4
def marbles_in_bag_B : Nat := 6
def marbles_in_bag_C : Nat := 2
def marbles_in_bag_D : Nat := 8
def marbles_in_bag_E : Nat := 4
def marbles_in_bag_F : Nat := 4

theorem james_marbles_left : total_initial_marbles - marbles_in_bag_D = 20 := by
  -- James has 28 marbles initially.
  -- He gives away Bag D which has 8 marbles.
  -- 28 - 8 = 20
  sorry

end james_marbles_left_l124_124444


namespace problem_statement_l124_124252

def approx_digit_place (num : ℕ) : ℕ :=
if num = 3020000 then 0 else sorry

theorem problem_statement :
  approx_digit_place (3 * 10^6 + 2 * 10^4) = 0 :=
by
  sorry

end problem_statement_l124_124252


namespace sarees_shirts_cost_l124_124809

variable (S T : ℕ)

-- Definition of conditions
def condition1 : Prop := 2 * S + 4 * T = 2 * S + 4 * T
def condition2 : Prop := (S + 6 * T) = (2 * S + 4 * T)
def condition3 : Prop := 12 * T = 2400

-- Proof goal
theorem sarees_shirts_cost :
  condition1 S T → condition2 S T → condition3 T → 2 * S + 4 * T = 1600 := by
  sorry

end sarees_shirts_cost_l124_124809


namespace sphere_surface_area_l124_124606

theorem sphere_surface_area (R h : ℝ) (R_pos : 0 < R) (h_pos : 0 < h) :
  ∃ A : ℝ, A = 2 * Real.pi * R * h := 
sorry

end sphere_surface_area_l124_124606


namespace explicit_formula_l124_124280

variable (f : ℝ → ℝ)
variable (is_quad : ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c)
variable (max_value : ∀ x, f x ≤ 13)
variable (value_at_3 : f 3 = 5)
variable (value_at_neg1 : f (-1) = 5)

theorem explicit_formula :
  (∀ x, f x = -2 * x^2 + 4 * x + 11) :=
by
  sorry

end explicit_formula_l124_124280


namespace oil_consumption_relation_l124_124976

noncomputable def initial_oil : ℝ := 62

noncomputable def remaining_oil (x : ℝ) : ℝ :=
  if x = 100 then 50
  else if x = 200 then 38
  else if x = 300 then 26
  else if x = 400 then 14
  else 62 - 0.12 * x

theorem oil_consumption_relation (x : ℝ) :
  remaining_oil x = 62 - 0.12 * x := by
  sorry

end oil_consumption_relation_l124_124976


namespace negation_of_square_positive_l124_124346

open Real

-- Define the original proposition
def prop_square_positive : Prop :=
  ∀ x : ℝ, x^2 > 0

-- Define the negation of the original proposition
def prop_square_not_positive : Prop :=
  ∃ x : ℝ, ¬ (x^2 > 0)

-- The theorem that asserts the logical equivalence for the negation
theorem negation_of_square_positive :
  ¬ prop_square_positive ↔ prop_square_not_positive :=
by sorry

end negation_of_square_positive_l124_124346


namespace chris_money_l124_124528

-- Define conditions
def grandmother_gift : Nat := 25
def aunt_uncle_gift : Nat := 20
def parents_gift : Nat := 75
def total_after_birthday : Nat := 279

-- Define the proof problem to show Chris had $159 before his birthday
theorem chris_money (x : Nat) (h : x + grandmother_gift + aunt_uncle_gift + parents_gift = total_after_birthday) :
  x = 159 :=
by
  -- Leave the proof blank
  sorry

end chris_money_l124_124528


namespace find_a_l124_124531

def g (x : ℝ) : ℝ := 5 * x - 7

theorem find_a (a : ℝ) : g a = 0 → a = 7 / 5 := by
  intro h
  sorry

end find_a_l124_124531


namespace maximum_m_value_l124_124129

theorem maximum_m_value (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) : 
  ∃ m, m = 4 ∧ (∀ a b : ℝ, a > 0 → b > 0 → a + b = 1 → (1 / a + 1 / b) ≥ m) :=
sorry

end maximum_m_value_l124_124129


namespace expected_total_rain_correct_l124_124516

-- Define the probabilities and rain amounts for one day.
def prob_sun : ℝ := 0.30
def prob_rain3 : ℝ := 0.40
def prob_rain8 : ℝ := 0.30
def rain_sun : ℝ := 0
def rain_three : ℝ := 3
def rain_eight : ℝ := 8
def days : ℕ := 7

-- Define the expected value of daily rain.
def E_daily_rain : ℝ :=
  prob_sun * rain_sun + prob_rain3 * rain_three + prob_rain8 * rain_eight

-- Define the expected total rain over seven days.
def E_total_rain : ℝ :=
  days * E_daily_rain

-- Statement of the proof problem.
theorem expected_total_rain_correct : E_total_rain = 25.2 := by
  -- Proof goes here
  sorry

end expected_total_rain_correct_l124_124516


namespace multiple_of_denominator_l124_124936

def denominator := 5
def numerator := denominator + 4

theorem multiple_of_denominator:
  (numerator + 6) = 3 * denominator :=
by
  -- Proof steps go here
  sorry

end multiple_of_denominator_l124_124936


namespace solve_for_x_l124_124410

theorem solve_for_x (x : ℝ) (h : (4 + x) / (6 + x) = (1 + x) / (2 + x)) : x = 2 :=
sorry

end solve_for_x_l124_124410


namespace angle_F_calculation_l124_124010

theorem angle_F_calculation (D E F : ℝ) :
  D = 80 ∧ E = 2 * F + 30 ∧ D + E + F = 180 → F = 70 / 3 :=
by
  intro h
  cases' h with hD h_remaining
  cases' h_remaining with hE h_sum
  sorry

end angle_F_calculation_l124_124010


namespace largest_mersenne_prime_lt_1000_l124_124230

def is_prime (p : ℕ) : Prop := ∀ d : ℕ, d ∣ p → d = 1 ∨ d = p

def is_mersenne_prime (n : ℕ) : Prop :=
  is_prime n ∧ ∃ p : ℕ, is_prime p ∧ n = 2^p - 1

theorem largest_mersenne_prime_lt_1000 : ∃ (n : ℕ), is_mersenne_prime n ∧ n < 1000 ∧ ∀ (m : ℕ), is_mersenne_prime m ∧ m < 1000 → m ≤ n :=
by
  -- Proof goes here
  sorry

end largest_mersenne_prime_lt_1000_l124_124230


namespace max_value_f_diff_l124_124565

open Real

noncomputable def f (A ω : ℝ) (x : ℝ) := A * sin (ω * x + π / 6) - 1

theorem max_value_f_diff {A ω : ℝ} (hA : A > 0) (hω : ω > 0)
  (h_sym : (π / 2) = π / (2 * ω))
  (h_initial : f A ω (π / 6) = 1) :
  ∀ (x1 x2 : ℝ), (0 ≤ x1 ∧ x1 ≤ π / 2) ∧ (0 ≤ x2 ∧ x2 ≤ π / 2) →
  (f A ω x1 - f A ω x2 ≤ 3) :=
sorry

end max_value_f_diff_l124_124565


namespace inverse_sum_l124_124778

noncomputable def g (x : ℝ) : ℝ :=
  if x ≤ 2 then 3 - x else 3 * x - x^2

theorem inverse_sum :
  (∃ x₁, g x₁ = -2 ∧ x₁ ≠ 5) ∨ (∃ x₂, g x₂ = 0 ∧ x₂ = 3) ∨ (∃ x₃, g x₃ = 4 ∧ x₃ = -1) → 
  g⁻¹ (-2) + g⁻¹ (0) + g⁻¹ (4) = 6 :=
by
  sorry

end inverse_sum_l124_124778


namespace circumcenter_is_rational_l124_124619

theorem circumcenter_is_rational (a1 a2 a3 b1 b2 b3 : ℚ)
  (h1 : (a2 - a1) ≠ 0 ∨ (b2 - b1) ≠ 0)
  (h2 : (a3 - a1) ≠ 0 ∨ (b3 - b1) ≠ 0) :
  ∃ x y : ℚ,
    ((a2 - a1) * x + (b2 - b1) * y = (a2^2 - a1^2 + b2^2 - b1^2) / 2) ∧
    ((a3 - a1) * x + (b3 - b1) * y = (a3^2 - a1^2 + b3^2 - b1^2) / 2) :=
begin
  -- proof goes here
  sorry,
end

end circumcenter_is_rational_l124_124619


namespace basketball_team_free_throws_l124_124483

theorem basketball_team_free_throws (a b x : ℕ) 
  (h1 : 3 * b = 2 * a)
  (h2 : x = 2 * a - 1)
  (h3 : 2 * a + 3 * b + x = 89) : 
  x = 29 :=
by
  sorry

end basketball_team_free_throws_l124_124483


namespace johns_age_l124_124573

-- Define variables for ages of John and Matt
variables (J M : ℕ)

-- Define the conditions based on the problem statement
def condition1 : Prop := M = 4 * J - 3
def condition2 : Prop := J + M = 52

-- The goal: prove that John is 11 years old
theorem johns_age (J M : ℕ) (h1 : condition1 J M) (h2 : condition2 J M) : J = 11 := by
  -- proof will go here
  sorry

end johns_age_l124_124573


namespace find_x_l124_124779

theorem find_x (x : ℤ) (h_pos : x > 0) 
  (n := x^2 + 2 * x + 17) 
  (d := 2 * x + 5)
  (h_div : n = d * x + 7) : x = 2 := 
sorry

end find_x_l124_124779


namespace perimeter_of_rectangle_EFGH_l124_124332

noncomputable def rectangle_ellipse_problem (u v c d : ℝ) : Prop :=
  (u * v = 3000) ∧
  (3000 = c * d) ∧
  ((u + v) = 2 * c) ∧
  ((u^2 + v^2).sqrt = 2 * (c^2 - d^2).sqrt) ∧
  (d = 3000 / c) ∧
  (4 * c = 8 * (1500).sqrt)

theorem perimeter_of_rectangle_EFGH :
  ∃ (u v c d : ℝ), rectangle_ellipse_problem u v c d ∧ 2 * (u + v) = 8 * (1500).sqrt := sorry

end perimeter_of_rectangle_EFGH_l124_124332


namespace smallest_possible_area_of_ellipse_l124_124696

theorem smallest_possible_area_of_ellipse
  (a b : ℝ)
  (h_ellipse : ∀ (x y : ℝ), (x^2 / a^2 + y^2 / b^2 = 1) → 
    (((x - 1/2)^2 + y^2 = 1/4) ∨ ((x + 1/2)^2 + y^2 = 1/4))) :
  ∃ (k : ℝ), (a * b * π = 4 * π) :=
by
  sorry

end smallest_possible_area_of_ellipse_l124_124696


namespace average_ratio_one_l124_124689

theorem average_ratio_one (scores : List ℝ) (h_len : scores.length = 50) :
  let A := (scores.sum / 50)
  let scores_with_averages := scores ++ [A, A]
  let A' := (scores_with_averages.sum / 52)
  A' = A :=
by
  sorry

end average_ratio_one_l124_124689


namespace speed_in_still_water_l124_124091

variable (upstream downstream : ℝ)

-- Conditions
def upstream_speed : Prop := upstream = 26
def downstream_speed : Prop := downstream = 40

-- Question and correct answer
theorem speed_in_still_water (h1 : upstream_speed upstream) (h2 : downstream_speed downstream) :
  (upstream + downstream) / 2 = 33 := by
  sorry

end speed_in_still_water_l124_124091


namespace find_x_l124_124089

theorem find_x :
  ∃ x : ℝ, (2020 + x)^2 = x^2 ∧ x = -1010 :=
sorry

end find_x_l124_124089


namespace weight_of_dry_grapes_l124_124367

def fresh_grapes : ℝ := 10 -- weight of fresh grapes in kg
def fresh_water_content : ℝ := 0.90 -- fresh grapes contain 90% water by weight
def dried_water_content : ℝ := 0.20 -- dried grapes contain 20% water by weight

theorem weight_of_dry_grapes : 
  (fresh_grapes * (1 - fresh_water_content)) / (1 - dried_water_content) = 1.25 := 
by 
  sorry

end weight_of_dry_grapes_l124_124367


namespace percentage_increase_l124_124698

theorem percentage_increase (G P : ℝ) (h1 : G = 15 + (P / 100) * 15) 
                            (h2 : 15 + 2 * G = 51) : P = 20 :=
by 
  sorry

end percentage_increase_l124_124698


namespace total_beads_sue_necklace_l124_124694

theorem total_beads_sue_necklace (purple blue green : ℕ) (h1 : purple = 7)
  (h2 : blue = 2 * purple) (h3 : green = blue + 11) : 
  purple + blue + green = 46 := 
by 
  sorry

end total_beads_sue_necklace_l124_124694


namespace abs_eq_ax_plus_1_one_negative_root_no_positive_roots_l124_124637

theorem abs_eq_ax_plus_1_one_negative_root_no_positive_roots (a : ℝ) :
  (∃ x : ℝ, |x| = a * x + 1 ∧ x < 0) ∧ (∀ x : ℝ, |x| = a * x + 1 → x ≤ 0) → a > -1 :=
by
  sorry

end abs_eq_ax_plus_1_one_negative_root_no_positive_roots_l124_124637


namespace alan_spent_total_amount_l124_124249

-- Conditions
def eggs_bought : ℕ := 20
def price_per_egg : ℕ := 2
def chickens_bought : ℕ := 6
def price_per_chicken : ℕ := 8

-- Total cost calculation 
def cost_of_eggs : ℕ := eggs_bought * price_per_egg
def cost_of_chickens : ℕ := chickens_bought * price_per_chicken
def total_cost : ℕ := cost_of_eggs + cost_of_chickens

-- Proof statement
theorem alan_spent_total_amount : total_cost = 88 :=
by
  unfold total_cost cost_of_eggs cost_of_chickens
  simp [eggs_bought, price_per_egg, chickens_bought, price_per_chicken]
  sorry

end alan_spent_total_amount_l124_124249


namespace max_ab_ac_bc_l124_124774

theorem max_ab_ac_bc (a b c : ℝ) (h : a + 3 * b + c = 6) : 
    ab + ac + bc <= 8 :=
sorry

end max_ab_ac_bc_l124_124774


namespace acute_isosceles_inscribed_in_circle_l124_124862

noncomputable def solve_problem : ℝ := by
  -- Let x be the angle BAC
  let x : ℝ := π * 5 / 11
  -- Considering the value of k in the problem statement
  let k : ℝ := 5 / 11
  -- Providing the value of k obtained from solving the problem
  exact k

theorem acute_isosceles_inscribed_in_circle (ABC : Type)
  [inhabited ABC]
  (inscribed : ∀ {A B C : ABC}, A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (B_tangent C_tangent : ∀ {A B C : ABC}, A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (D : ABC)
  (angle_eq : ∀ {A B C : ABC}, ∠ABC = ∠ACB)
  (triple_angle : ∀ {A B C D : ABC}, ∠ABC = 3 * ∠BAC) :
  solve_problem = 5 / 11 := 
sorry

end acute_isosceles_inscribed_in_circle_l124_124862


namespace solve_for_x_l124_124791

theorem solve_for_x : 
  ∀ x : ℝ, 
    (x ≠ 2) ∧ (4 * x^2 + 3 * x + 1) / (x - 2) = 4 * x + 5 → 
    x = -11 / 6 :=
by
  intro x
  intro h 
  sorry

end solve_for_x_l124_124791


namespace repeating_decimal_fractional_representation_l124_124477

theorem repeating_decimal_fractional_representation :
  (0.36 : ℝ) = (4 / 11 : ℝ) :=
sorry

end repeating_decimal_fractional_representation_l124_124477


namespace arithmetic_sequence_fifth_term_l124_124766

theorem arithmetic_sequence_fifth_term :
  let a1 := 3
  let d := 4
  let a5 := a1 + (5 - 1) * d
  a5 = 19 :=
by
  let a1 := 3
  let d := 4
  let a5 := a1 + (5 - 1) * d
  show a5 = 19
  sorry

end arithmetic_sequence_fifth_term_l124_124766


namespace equal_values_on_plane_l124_124865

theorem equal_values_on_plane (f : ℤ × ℤ → ℕ)
    (h_avg : ∀ (i j : ℤ), f (i, j) = (f (i+1, j) + f (i-1, j) + f (i, j+1) + f (i, j-1)) / 4) :
  ∃ c : ℕ, ∀ (i j : ℤ), f (i, j) = c :=
by
  sorry

end equal_values_on_plane_l124_124865


namespace product_of_cosines_value_l124_124875

noncomputable def product_of_cosines : ℝ :=
  (1 + Real.cos (Real.pi / 12)) * (1 + Real.cos (5 * Real.pi / 12)) *
  (1 + Real.cos (7 * Real.pi / 12)) * (1 + Real.cos (11 * Real.pi / 12))

theorem product_of_cosines_value :
  product_of_cosines = 1 / 16 :=
by
  sorry

end product_of_cosines_value_l124_124875


namespace number_of_real_solutions_of_equation_l124_124631

theorem number_of_real_solutions_of_equation :
  (∀ x : ℝ, ((2 : ℝ)^(4 * x + 2)) * ((4 : ℝ)^(2 * x + 8)) = ((8 : ℝ)^(3 * x + 7))) ↔ x = -3 :=
by sorry

end number_of_real_solutions_of_equation_l124_124631


namespace max_intersection_points_circles_lines_l124_124357

-- Definitions based on the conditions
def num_circles : ℕ := 3
def num_lines : ℕ := 2

-- Function to calculate the number of points of intersection
def max_points_of_intersection (num_circles num_lines : ℕ) : ℕ :=
  (num_circles * (num_circles - 1) / 2) * 2 + 
  num_circles * num_lines * 2 + 
  (num_lines * (num_lines - 1) / 2)

-- The proof statement
theorem max_intersection_points_circles_lines :
  max_points_of_intersection num_circles num_lines = 19 :=
by
  sorry

end max_intersection_points_circles_lines_l124_124357


namespace probability_green_l124_124076

def total_marbles : ℕ := 100

def P_white : ℚ := 1 / 4

def P_red_or_blue : ℚ := 0.55

def P_sum : ℚ := 1

theorem probability_green :
  P_sum = P_white + P_red_or_blue + P_green →
  P_green = 0.2 :=
sorry

end probability_green_l124_124076


namespace distance_P_to_y_axis_l124_124764

-- Definition: Given point P in Cartesian coordinates
def P : ℝ × ℝ := (-3, -4)

-- Definition: Function to calculate distance to y-axis
def distance_to_y_axis (p : ℝ × ℝ) : ℝ := abs p.1

-- Theorem: The distance from point P to the y-axis is 3
theorem distance_P_to_y_axis : distance_to_y_axis P = 3 :=
by
  sorry

end distance_P_to_y_axis_l124_124764


namespace greatest_integer_l124_124497

-- Define the conditions for the problem
def isMultiple4 (n : ℕ) : Prop := n % 4 = 0
def notMultiple8 (n : ℕ) : Prop := n % 8 ≠ 0
def notMultiple12 (n : ℕ) : Prop := n % 12 ≠ 0
def gcf4 (n : ℕ) : Prop := Nat.gcd n 24 = 4
def lessThan200 (n : ℕ) : Prop := n < 200

-- State the main theorem
theorem greatest_integer : ∃ n : ℕ, lessThan200 n ∧ gcf4 n ∧ n = 196 :=
by
  sorry

end greatest_integer_l124_124497
