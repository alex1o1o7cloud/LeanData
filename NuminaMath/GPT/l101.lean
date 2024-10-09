import Mathlib

namespace perfect_square_mod_3_l101_10110

theorem perfect_square_mod_3 (n : ℤ) : n^2 % 3 = 0 ∨ n^2 % 3 = 1 :=
sorry

end perfect_square_mod_3_l101_10110


namespace largest_n_divisible_l101_10146

theorem largest_n_divisible (n : ℕ) (h : (n : ℤ) > 0) : 
  (n^3 + 105) % (n + 12) = 0 ↔ n = 93 :=
sorry

end largest_n_divisible_l101_10146


namespace inequality_equivalence_l101_10107

theorem inequality_equivalence (a : ℝ) :
  (∀ (x : ℝ), |x + 1| + |x - 1| ≥ a) ↔ (a ≤ 2) :=
sorry

end inequality_equivalence_l101_10107


namespace t_shirt_cost_l101_10116

theorem t_shirt_cost (n_tshirts : ℕ) (total_cost : ℝ) (cost_per_tshirt : ℝ)
  (h1 : n_tshirts = 25)
  (h2 : total_cost = 248) :
  cost_per_tshirt = 9.92 :=
by
  sorry

end t_shirt_cost_l101_10116


namespace min_crossing_time_proof_l101_10170

def min_crossing_time (times : List ℕ) : ℕ :=
  -- Function to compute the minimum crossing time. Note: Actual implementation skipped.
sorry

theorem min_crossing_time_proof
  (times : List ℕ)
  (h_times : times = [2, 4, 8, 16]) :
  min_crossing_time times = 30 :=
sorry

end min_crossing_time_proof_l101_10170


namespace three_digit_powers_of_two_l101_10157

theorem three_digit_powers_of_two : 
  ∃ (N : ℕ), N = 3 ∧ ∀ (n : ℕ), (100 ≤ 2^n ∧ 2^n < 1000) ↔ (n = 7 ∨ n = 8 ∨ n = 9) :=
by
  sorry

end three_digit_powers_of_two_l101_10157


namespace find_a1_l101_10103

theorem find_a1 (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h_rec : ∀ n ≥ 2, a n + 2 * S n * S (n - 1) = 0)
  (h_S5 : S 5 = 1 / 11) : 
  a 1 = 1 / 3 := 
sorry

end find_a1_l101_10103


namespace curious_number_is_digit_swap_divisor_l101_10111

theorem curious_number_is_digit_swap_divisor (a b : ℕ) (hab : a ≠ 0 ∧ b ≠ 0) :
  (10 * a + b) ∣ (10 * b + a) → (10 * a + b) = 11 ∨ (10 * a + b) = 22 ∨ (10 * a + b) = 33 ∨ 
  (10 * a + b) = 44 ∨ (10 * a + b) = 55 ∨ (10 * a + b) = 66 ∨ 
  (10 * a + b) = 77 ∨ (10 * a + b) = 88 ∨ (10 * a + b) = 99 :=
by
  sorry

end curious_number_is_digit_swap_divisor_l101_10111


namespace tip_percentage_l101_10118

theorem tip_percentage (T : ℝ) 
  (total_cost meal_cost sales_tax : ℝ)
  (h1 : meal_cost = 61.48)
  (h2 : sales_tax = 0.07 * meal_cost)
  (h3 : total_cost = meal_cost + sales_tax + T * meal_cost)
  (h4 : total_cost ≤ 75) :
  T ≤ 0.1499 :=
by
  -- main proof goes here
  sorry

end tip_percentage_l101_10118


namespace initial_gift_card_value_l101_10184

-- The price per pound of coffee
def cost_per_pound : ℝ := 8.58

-- The number of pounds of coffee bought by Rita
def pounds_bought : ℝ := 4.0

-- The remaining balance on Rita's gift card after buying coffee
def remaining_balance : ℝ := 35.68

-- The total cost of the coffee Rita bought
def total_cost_of_coffee : ℝ := cost_per_pound * pounds_bought

-- The initial value of Rita's gift card
def initial_value_of_gift_card : ℝ := total_cost_of_coffee + remaining_balance

-- Statement of the proof problem
theorem initial_gift_card_value : initial_value_of_gift_card = 70.00 :=
by
  -- Placeholder for the proof
  sorry

end initial_gift_card_value_l101_10184


namespace range_of_a_l101_10194

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x > 1 → x + 1 / (x - 1) ≥ a) → a ≤ 3 :=
by
  intro h
  sorry

end range_of_a_l101_10194


namespace prob_at_least_two_correct_l101_10191

-- Probability of guessing a question correctly
def prob_correct := 1 / 6

-- Probability of guessing a question incorrectly
def prob_incorrect := 5 / 6

-- Binomial probability mass function for k successes out of n trials
def binom_pmf (n k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k : ℚ) * (p ^ k) * ((1 - p) ^ (n - k))

-- Calculate probability P(X = 0)
def prob_X0 := binom_pmf 6 0 prob_correct

-- Calculate probability P(X = 1)
def prob_X1 := binom_pmf 6 1 prob_correct

-- Theorem for the desired probability
theorem prob_at_least_two_correct : 
  1 - (prob_X0 + prob_X1) = 34369 / 58420 := by
  sorry

end prob_at_least_two_correct_l101_10191


namespace three_digit_number_is_275_l101_10176

noncomputable def digits (n : ℕ) : ℕ × ℕ × ℕ :=
  (n / 100 % 10, n / 10 % 10, n % 10)

theorem three_digit_number_is_275 :
  ∃ (n : ℕ), n / 100 % 10 + n % 10 = n / 10 % 10 ∧
              7 * (n / 100 % 10) = n % 10 + n / 10 % 10 + 2 ∧
              n / 100 % 10 + n / 10 % 10 + n % 10 = 14 ∧
              n = 275 :=
by
  sorry

end three_digit_number_is_275_l101_10176


namespace find_side_b_l101_10137

variable {a b c : ℝ} -- sides of the triangle
variable {A B C : ℝ} -- angles of the triangle
variable {area : ℝ}

axiom sides_form_arithmetic_sequence : 2 * b = a + c
axiom angle_B_is_60_degrees : B = Real.pi / 3
axiom area_is_3sqrt3 : area = 3 * Real.sqrt 3
axiom area_formula : area = 1 / 2 * a * c * Real.sin (B)

theorem find_side_b : b = 2 * Real.sqrt 3 := by
  sorry

end find_side_b_l101_10137


namespace f_5_eq_2_l101_10180

def f : ℕ → ℤ :=
sorry

axiom f_initial_condition : f 1 = 2

axiom f_functional_eq (a b : ℕ) : f (a + b) = 2 * f a + 2 * f b - 3 * f (a * b)

theorem f_5_eq_2 : f 5 = 2 :=
sorry

end f_5_eq_2_l101_10180


namespace simultaneous_equations_solution_l101_10112

-- Definition of the two equations
def eq1 (m x y : ℝ) : Prop := y = m * x + 5
def eq2 (m x y : ℝ) : Prop := y = (3 * m - 2) * x + 6

-- Lean theorem statement to check if the equations have a solution
theorem simultaneous_equations_solution (m : ℝ) :
  (m ≠ 1) ↔ ∃ x y : ℝ, eq1 m x y ∧ eq2 m x y := 
sorry

end simultaneous_equations_solution_l101_10112


namespace probability_dice_sum_12_l101_10172

def total_outcomes : ℕ := 216
def favorable_outcomes : ℕ := 25

theorem probability_dice_sum_12 :
  (favorable_outcomes : ℚ) / total_outcomes = 25 / 216 := by
  sorry

end probability_dice_sum_12_l101_10172


namespace B_and_C_complementary_l101_10177

def EventA (selected : List String) : Prop :=
  selected.count "boy" = 1

def EventB (selected : List String) : Prop :=
  selected.count "boy" ≥ 1

def EventC (selected : List String) : Prop :=
  selected.count "girl" = 2

theorem B_and_C_complementary :
  ∀ selected : List String,
    (selected.length = 2 ∧ (EventB selected ∨ EventC selected)) ∧ 
    (¬ (EventB selected ∧ EventC selected)) →
    (EventB selected → ¬ EventC selected) ∧ (EventC selected → ¬ EventB selected) :=
  sorry

end B_and_C_complementary_l101_10177


namespace positive_integers_satisfy_condition_l101_10108

theorem positive_integers_satisfy_condition :
  ∃! n : ℕ, (n > 0 ∧ 30 - 6 * n > 18) :=
by
  sorry

end positive_integers_satisfy_condition_l101_10108


namespace daps_to_dips_l101_10145

theorem daps_to_dips : 
  (∀ a b c d : ℝ, (5 * a = 4 * b) → (3 * b = 8 * c) → (c = 48 * d) → (a = 22.5 * d)) := 
by
  intros a b c d h1 h2 h3
  sorry

end daps_to_dips_l101_10145


namespace overtaking_time_l101_10179

theorem overtaking_time :
  ∀ t t_k : ℕ,
  (30 * t = 40 * (t - 5)) ∧ 
  (30 * t = 60 * t_k) →
  t = 20 ∧ t_k = 10 ∧ (20 - 10 = 10) :=
by
  sorry

end overtaking_time_l101_10179


namespace greatest_value_b_l101_10114

-- Define the polynomial and the inequality condition
def polynomial (b : ℝ) : ℝ := -b^2 + 8*b - 12
#check polynomial
-- State the main theorem with the given condition and the result
theorem greatest_value_b (b : ℝ) : -b^2 + 8*b - 12 ≥ 0 → b ≤ 6 :=
sorry

end greatest_value_b_l101_10114


namespace halfway_fraction_eq_l101_10121

-- Define the fractions
def one_seventh := 1 / 7
def one_fourth := 1 / 4

-- Define the common denominators
def common_denom_1 := 4 / 28
def common_denom_2 := 7 / 28

-- Define the addition of the common denominators
def addition := common_denom_1 + common_denom_2

-- Define the average of the fractions
noncomputable def average := addition / 2

-- State the theorem
theorem halfway_fraction_eq : average = 11 / 56 :=
by
  -- Provide the steps which will be skipped here
  sorry

end halfway_fraction_eq_l101_10121


namespace price_for_3years_service_l101_10120

def full_price : ℝ := 85
def discount_price_1year (price : ℝ) : ℝ := price - (0.20 * price)
def discount_price_3years (price : ℝ) : ℝ := price - (0.25 * price)

theorem price_for_3years_service : discount_price_3years (discount_price_1year full_price) = 51 := 
by 
  sorry

end price_for_3years_service_l101_10120


namespace scrabble_champions_l101_10153

theorem scrabble_champions :
  let total_champions := 10
  let men_percentage := 0.40
  let men_champions := total_champions * men_percentage
  let bearded_percentage := 0.40
  let non_bearded_percentage := 0.60

  let bearded_men_champions := men_champions * bearded_percentage
  let non_bearded_men_champions := men_champions * non_bearded_percentage

  let bearded_bald_percentage := 0.60
  let bearded_with_hair_percentage := 0.40
  let non_bearded_bald_percentage := 0.30
  let non_bearded_with_hair_percentage := 0.70

  (bearded_men_champions * bearded_bald_percentage).round = 2 ∧
  (bearded_men_champions * bearded_with_hair_percentage).round = 1 ∧
  (non_bearded_men_champions * non_bearded_bald_percentage).round = 2 ∧
  (non_bearded_men_champions * non_bearded_with_hair_percentage).round = 4 :=
by 
sorry

end scrabble_champions_l101_10153


namespace calc_man_dividend_l101_10162

noncomputable def calc_dividend (investment : ℝ) (face_value : ℝ) (premium : ℝ) (dividend_percent : ℝ) : ℝ :=
  let cost_per_share := face_value * (1 + premium / 100)
  let number_of_shares := investment / cost_per_share
  let dividend_per_share := dividend_percent / 100 * face_value
  let total_dividend := dividend_per_share * number_of_shares
  total_dividend

theorem calc_man_dividend :
  calc_dividend 14400 100 20 5 = 600 :=
by
  sorry

end calc_man_dividend_l101_10162


namespace minyoung_gave_nine_notebooks_l101_10198

theorem minyoung_gave_nine_notebooks (original left given : ℕ) (h1 : original = 17) (h2 : left = 8) (h3 : given = original - left) : given = 9 :=
by
  rw [h1, h2] at h3
  exact h3

end minyoung_gave_nine_notebooks_l101_10198


namespace painted_cubes_even_faces_l101_10151

theorem painted_cubes_even_faces :
  let L := 6 -- length of the block
  let W := 2 -- width of the block
  let H := 2 -- height of the block
  let total_cubes := 24 -- the block is cut into 24 1-inch cubes
  let cubes_even_faces := 12 -- the number of 1-inch cubes with even number of blue faces
  -- each cube has a total of 6 faces,
  -- we need to count how many cubes have an even number of painted faces.
  L * W * H = total_cubes → 
  cubes_even_faces = 12 := sorry

end painted_cubes_even_faces_l101_10151


namespace candies_per_packet_l101_10129

-- Define the given conditions
def monday_to_friday_candies_per_day := 2
def weekend_candies_per_day := 1
def weekdays := 5
def weekends := 2
def weeks := 3
def packets := 2

-- Calculate the number of candies Bobby eats in a week
def candies_per_week := (monday_to_friday_candies_per_day * weekdays) + (weekend_candies_per_day * weekends)

-- Calculate the total number of candies Bobby eats in the given 3 weeks
def total_candies_in_3_weeks := candies_per_week * weeks

-- Divide the total number of candies by the number of packets to find the candies per packet
theorem candies_per_packet : total_candies_in_3_weeks / packets = 18 := 
by
  -- Adding the proof placeholder
  sorry

end candies_per_packet_l101_10129


namespace find_x_l101_10182

def operation_star (a b c d : ℤ) : ℤ × ℤ :=
  (a + c, b - 2 * d)

theorem find_x (x y : ℤ) (h : operation_star (x+1) (y-1) 1 3 = (2, -4)) : x = 0 :=
by 
  sorry

end find_x_l101_10182


namespace manufacturing_percentage_l101_10124

theorem manufacturing_percentage (a b : ℕ) (h1 : a = 108) (h2 : b = 360) : (a / b : ℚ) * 100 = 30 :=
by
  sorry

end manufacturing_percentage_l101_10124


namespace kitchen_supplies_sharon_wants_l101_10154

theorem kitchen_supplies_sharon_wants (P : ℕ) (plates_angela cutlery_angela pots_sharon plates_sharon cutlery_sharon : ℕ) 
  (h1 : plates_angela = 3 * P + 6) 
  (h2 : cutlery_angela = (3 * P + 6) / 2) 
  (h3 : pots_sharon = P / 2) 
  (h4 : plates_sharon = 3 * (3 * P + 6) - 20) 
  (h5 : cutlery_sharon = 2 * (3 * P + 6) / 2) 
  (h_total : pots_sharon + plates_sharon + cutlery_sharon = 254) : 
  P = 20 :=
sorry

end kitchen_supplies_sharon_wants_l101_10154


namespace john_initial_money_l101_10125

variable (X S : ℕ)
variable (L : ℕ := 500)
variable (cond1 : L = S - 600)
variable (cond2 : X = S + L)

theorem john_initial_money : X = 1600 :=
by
  sorry

end john_initial_money_l101_10125


namespace complete_square_l101_10136

theorem complete_square (x : ℝ) (h : x^2 + 8 * x + 9 = 0) : (x + 4)^2 = 7 := by
  sorry

end complete_square_l101_10136


namespace length_of_CD_l101_10187

theorem length_of_CD (L : ℝ) (r : ℝ) (V_total : ℝ) (cylinder_vol : ℝ) (hemisphere_vol : ℝ) : 
  r = 5 ∧ V_total = 900 * Real.pi ∧ cylinder_vol = Real.pi * r^2 * L ∧ hemisphere_vol = (2/3) *Real.pi * r^3 → 
  V_total = cylinder_vol + 2 * hemisphere_vol → 
  L = 88 / 3 := 
by
  sorry

end length_of_CD_l101_10187


namespace relationship_between_length_and_width_l101_10106

theorem relationship_between_length_and_width 
  (x y : ℝ) (h : 2 * (x + y) = 20) : y = 10 - x := 
by
  sorry

end relationship_between_length_and_width_l101_10106


namespace rowing_distance_l101_10141

theorem rowing_distance
  (rowing_speed_in_still_water : ℝ)
  (velocity_of_current : ℝ)
  (total_time : ℝ)
  (H1 : rowing_speed_in_still_water = 5)
  (H2 : velocity_of_current = 1)
  (H3 : total_time = 1) :
  ∃ (D : ℝ), D = 2.4 := 
sorry

end rowing_distance_l101_10141


namespace largest_rectangle_area_l101_10192

theorem largest_rectangle_area (x y : ℝ) (h1 : 2*x + 2*y = 60) (h2 : x ≥ 2*y) : ∃ A, A = x*y ∧ A ≤ 200 := by
  sorry

end largest_rectangle_area_l101_10192


namespace symmetry_center_of_g_l101_10178

open Real

noncomputable def g (x : ℝ) : ℝ := cos ((1 / 2) * x - π / 6)

def center_of_symmetry : Set (ℝ × ℝ) := { p | ∃ k : ℤ, p = (2 * k * π + 4 * π / 3, 0) }

theorem symmetry_center_of_g :
  (∃ p : ℝ × ℝ, p ∈ center_of_symmetry) :=
sorry

end symmetry_center_of_g_l101_10178


namespace f_value_at_2_9_l101_10138

-- Define the function f with its properties as conditions
noncomputable def f (x : ℝ) : ℝ := sorry

-- Define the domain of f
axiom f_domain : ∀ x, 0 ≤ x ∧ x ≤ 1

-- Condition (i)
axiom f_0_eq : f 0 = 0

-- Condition (ii)
axiom f_monotone : ∀ (x y : ℝ), 0 ≤ x ∧ x < y ∧ y ≤ 1 → f x ≤ f y

-- Condition (iii)
axiom f_symmetry : ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → f (1 - x) = 3/4 - f x / 2

-- Condition (iv)
axiom f_scale : ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → f (x / 3) = f x / 3

-- Proof goal
theorem f_value_at_2_9 : f (2/9) = 5/24 := by
  sorry

end f_value_at_2_9_l101_10138


namespace least_number_divisible_by_6_has_remainder_4_is_40_l101_10147

-- Define the least number N which leaves a remainder of 4 when divided by 6
theorem least_number_divisible_by_6_has_remainder_4_is_40 :
  ∃ (N : ℕ), (∀ (k : ℕ), N = 6 * k + 4) ∧ N = 40 := by
  sorry

end least_number_divisible_by_6_has_remainder_4_is_40_l101_10147


namespace find_h_of_root_l101_10193

theorem find_h_of_root :
  ∀ h : ℝ, (-3)^3 + h * (-3) - 10 = 0 → h = -37/3 := by
  sorry

end find_h_of_root_l101_10193


namespace person_B_work_days_l101_10142

-- Let a be the work rate for person A, and b be the work rate for person B.
-- a completes the work in 20 days
-- b completes the work in x days
-- When working together, a and b complete 0.375 of the work in 5 days


theorem person_B_work_days (x : ℝ) :
  ((5 : ℝ) * ((1 / 20) + 1 / x) = 0.375) -> x = 40 := 
by 
  sorry

end person_B_work_days_l101_10142


namespace frustum_smaller_cone_height_l101_10155

theorem frustum_smaller_cone_height (H frustum_height radius1 radius2 : ℝ) 
  (h : ℝ) (h_eq : h = 30 - 18) : 
  radius1 = 6 → radius2 = 10 → frustum_height = 18 → H = 30 → h = 12 := 
by
  intros
  sorry

end frustum_smaller_cone_height_l101_10155


namespace no_int_solutions_l101_10149

open Nat

theorem no_int_solutions (p1 p2 α n : ℕ)
  (hp1_prime : p1.Prime)
  (hp2_prime : p2.Prime)
  (hp1_odd : p1 % 2 = 1)
  (hp2_odd : p2 % 2 = 1)
  (hα_pos : 0 < α)
  (hn_pos : 0 < n)
  (hα_gt1 : 1 < α)
  (hn_gt1 : 1 < n) :
  ¬(let lhs := ((p2 - 1) / 2) ^ p1 + ((p2 + 1) / 2) ^ p1
    lhs = α ^ n) :=
sorry

end no_int_solutions_l101_10149


namespace members_do_not_play_either_l101_10166

noncomputable def total_members := 30
noncomputable def badminton_players := 16
noncomputable def tennis_players := 19
noncomputable def both_players := 7

theorem members_do_not_play_either : 
  (total_members - (badminton_players + tennis_players - both_players)) = 2 :=
by
  sorry

end members_do_not_play_either_l101_10166


namespace number_of_cats_l101_10135

variable (C D : ℕ)

-- Conditions
def condition1 : Prop := C = 15 * D / 7
def condition2 : Prop := C = 15 * (D + 12) / 11

-- Proof problem
theorem number_of_cats (h1 : condition1 C D) (h2 : condition2 C D) : C = 45 := sorry

end number_of_cats_l101_10135


namespace value_of_a4_l101_10173

open Nat

def sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 2 ∧ ∀ n, a (n + 1) = 2 * a n + 1

theorem value_of_a4 (a : ℕ → ℕ) (h : sequence a) : a 4 = 23 :=
by
  -- Proof to be provided or implemented
  sorry

end value_of_a4_l101_10173


namespace space_between_trees_l101_10143

theorem space_between_trees (tree_count : ℕ) (tree_space : ℕ) (road_length : ℕ)
  (h1 : tree_space = 1) (h2 : tree_count = 13) (h3 : road_length = 157) :
  (road_length - tree_count * tree_space) / (tree_count - 1) = 12 := by
  sorry

end space_between_trees_l101_10143


namespace quadratic_inequality_solution_l101_10104

theorem quadratic_inequality_solution (x : ℝ) : 
  (x^2 - 6 * x + 5 > 0) ↔ (x < 1 ∨ x > 5) :=
by sorry

end quadratic_inequality_solution_l101_10104


namespace integer_solutions_for_exponential_equation_l101_10150

theorem integer_solutions_for_exponential_equation :
  ∃ (a b c : ℕ), 
  2 ^ a * 3 ^ b + 9 = c ^ 2 ∧ 
  (a = 4 ∧ b = 0 ∧ c = 5) ∨ 
  (a = 3 ∧ b = 2 ∧ c = 9) ∨ 
  (a = 4 ∧ b = 3 ∧ c = 21) ∨ 
  (a = 3 ∧ b = 3 ∧ c = 15) ∨ 
  (a = 4 ∧ b = 5 ∧ c = 51) :=
by {
  -- This is where the proof would go.
  sorry
}

end integer_solutions_for_exponential_equation_l101_10150


namespace simplify_expression_l101_10122

variable {a : ℝ} (h1 : a ≠ -3) (h2 : a ≠ 3) (h3 : a ≠ 2) (h4 : 2 * a + 6 ≠ 0)

theorem simplify_expression : (1 / (a + 3) + 1 / (a ^ 2 - 9)) / ((a - 2) / (2 * a + 6)) = 2 / (a - 3) :=
by
  sorry

end simplify_expression_l101_10122


namespace first_term_exceeding_1000_l101_10196

variable (a₁ : Int := 2)
variable (d : Int := 3)

def arithmetic_sequence (n : Int) : Int :=
  a₁ + (n - 1) * d

theorem first_term_exceeding_1000 :
  ∃ n : Int, n = 334 ∧ arithmetic_sequence n > 1000 := by
  sorry

end first_term_exceeding_1000_l101_10196


namespace remainder_when_3n_plus_2_squared_divided_by_11_l101_10140

theorem remainder_when_3n_plus_2_squared_divided_by_11 (n : ℕ) (h : n % 7 = 5) : ((3 * n + 2)^2) % 11 = 3 :=
  sorry

end remainder_when_3n_plus_2_squared_divided_by_11_l101_10140


namespace twice_as_many_juniors_as_seniors_l101_10190

theorem twice_as_many_juniors_as_seniors (j s : ℕ) (h : (1/3 : ℝ) * j = (2/3 : ℝ) * s) : j = 2 * s :=
by
  --proof steps here
  sorry

end twice_as_many_juniors_as_seniors_l101_10190


namespace find_four_digit_number_l101_10102

variable {N : ℕ} {a x y : ℕ}

theorem find_four_digit_number :
  (∃ a x y : ℕ, y < 10 ∧ 10 + a = x * y ∧ x = 9 + a ∧ N = 1000 + a + 10 * b + 100 * b ∧
  (N = 1014 ∨ N = 1035 ∨ N = 1512)) :=
by
  sorry

end find_four_digit_number_l101_10102


namespace advantageous_bank_l101_10169

variable (C : ℝ) (p n : ℝ)

noncomputable def semiAnnualCompounding (p : ℝ) (n : ℝ) : ℝ :=
  (1 + p / (2 * 100)) ^ n

noncomputable def monthlyCompounding (p : ℝ) (n : ℝ) : ℝ :=
  (1 + p / (12 * 100)) ^ (6 * n)

theorem advantageous_bank (p n : ℝ) :
  monthlyCompounding p n - semiAnnualCompounding p n > 0 := sorry

#check advantageous_bank

end advantageous_bank_l101_10169


namespace smallest_possible_n_l101_10132

theorem smallest_possible_n
  (n : ℕ)
  (d : ℕ)
  (h_d_pos : d > 0)
  (h_profit : 10 * n - 30 = 100)
  (h_cost_multiple : ∃ k, d = 2 * n * k) :
  n = 13 :=
by {
  sorry
}

end smallest_possible_n_l101_10132


namespace slope_range_l101_10139

theorem slope_range (a : ℝ) (ha : a ∈ Set.Icc (Real.pi / 4) (Real.pi / 2)) :
  ∃ k : ℝ, k = Real.tan a ∧ k ∈ Set.Ici 1 :=
by {
  sorry
}

end slope_range_l101_10139


namespace range_of_x_l101_10181

def interval1 : Set ℝ := {x | 2 ≤ x ∧ x ≤ 5}
def interval2 : Set ℝ := {x | x < 1 ∨ x > 4}
def false_statement (x : ℝ) : Prop := x ∈ interval1 ∨ x ∈ interval2

theorem range_of_x (x : ℝ) (h : ¬ false_statement x) : x ∈ Set.Ico 1 2 :=
by
  sorry

end range_of_x_l101_10181


namespace height_of_model_l101_10160

noncomputable def original_monument_height : ℝ := 100
noncomputable def original_monument_radius : ℝ := 20
noncomputable def original_monument_volume : ℝ := 125600
noncomputable def model_volume : ℝ := 1.256

theorem height_of_model : original_monument_height / (original_monument_volume / model_volume)^(1/3) = 1 :=
by
  sorry

end height_of_model_l101_10160


namespace rectangular_plot_perimeter_l101_10189

theorem rectangular_plot_perimeter (w : ℝ) (P : ℝ) (cost_per_meter : ℝ) (total_cost : ℝ) :
  (cost_per_meter = 6.5) →
  (total_cost = 1430) →
  (P = 2 * (w + (w + 10))) →
  (cost_per_meter * P = total_cost) →
  P = 220 :=
by
  sorry

end rectangular_plot_perimeter_l101_10189


namespace valid_six_digit_numbers_l101_10175

def is_divisible_by_4 (n : Nat) : Prop :=
  n % 4 = 0

def digit_sum (n : Nat) : Nat :=
  (Nat.digits 10 n).sum

def is_divisible_by_9 (n : Nat) : Prop :=
  digit_sum n % 9 = 0

def is_valid_six_digit_number (n : Nat) : Prop :=
  ∃ (a b : Nat), n = b * 100000 + 20140 + a ∧ is_divisible_by_4 (10 * 2014 + a) ∧ is_divisible_by_9 (b * 100000 + 20140 + a)

theorem valid_six_digit_numbers :
  { n | is_valid_six_digit_number n } = {220140, 720144, 320148} :=
by
  sorry

end valid_six_digit_numbers_l101_10175


namespace cyclic_inequality_l101_10167

theorem cyclic_inequality (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  2 * (x^3 + y^3 + z^3) ≥ x^2 * y + x^2 * z + y^2 * z + y^2 * x + z^2 * x + z^2 * y := 
by
  sorry

end cyclic_inequality_l101_10167


namespace mary_candies_l101_10171

-- The conditions
def bob_candies : Nat := 10
def sue_candies : Nat := 20
def john_candies : Nat := 5
def sam_candies : Nat := 10
def total_candies : Nat := 50

-- The theorem to prove
theorem mary_candies :
  total_candies - (bob_candies + sue_candies + john_candies + sam_candies) = 5 := by
  -- Here is where the proof would go; currently using sorry to skip the proof
  sorry

end mary_candies_l101_10171


namespace find_siblings_l101_10100

-- Define the characteristics of each child
structure Child where
  name : String
  eyeColor : String
  hairColor : String
  age : Nat

-- List of children
def Olivia : Child := { name := "Olivia", eyeColor := "Green", hairColor := "Red", age := 12 }
def Henry  : Child := { name := "Henry", eyeColor := "Gray", hairColor := "Brown", age := 12 }
def Lucas  : Child := { name := "Lucas", eyeColor := "Green", hairColor := "Red", age := 10 }
def Emma   : Child := { name := "Emma", eyeColor := "Green", hairColor := "Brown", age := 12 }
def Mia    : Child := { name := "Mia", eyeColor := "Gray", hairColor := "Red", age := 10 }
def Noah   : Child := { name := "Noah", eyeColor := "Gray", hairColor := "Brown", age := 12 }

-- Define a family as a set of children who share at least one characteristic
def isFamily (c1 c2 c3 : Child) : Prop :=
  (c1.eyeColor = c2.eyeColor ∨ c1.eyeColor = c3.eyeColor ∨ c2.eyeColor = c3.eyeColor) ∨
  (c1.hairColor = c2.hairColor ∨ c1.hairColor = c3.hairColor ∨ c2.hairColor = c3.hairColor) ∨
  (c1.age = c2.age ∨ c1.age = c3.age ∨ c2.age = c3.age)

-- The main theorem
theorem find_siblings : isFamily Olivia Lucas Emma :=
by
  sorry

end find_siblings_l101_10100


namespace at_most_n_zeros_l101_10161

-- Definitions of conditions
variables {α : Type*} [Inhabited α]

/-- Define the structure of the sheet of numbers with the given properties -/
structure sheet :=
(n : ℕ)
(val : ℕ → ℤ)

-- Assuming infinite sheet and the properties
variable (s : sheet)

-- Predicate for a row having only positive integers
def all_positive (r : ℕ → ℤ) : Prop := ∀ i, r i > 0

-- Define the initial row R which has all positive integers
variable {R : ℕ → ℤ}

-- Statement that each element in the row below is sum of element above and to the left
def below_sum (r R : ℕ → ℤ) (n : ℕ) : Prop := ∀ i, r i = R i + (if i = 0 then 0 else R (i - 1))

-- Variable for the row n below R
variable {Rn : ℕ → ℤ}

-- Main theorem statement
theorem at_most_n_zeros (n : ℕ) (hr : all_positive R) (hs : below_sum R Rn n) : 
  ∃ k ≤ n, Rn k = 0 ∨ Rn k > 0 := sorry

end at_most_n_zeros_l101_10161


namespace simplify_and_evaluate_l101_10199

-- Define the constants
def a : ℤ := -1
def b : ℤ := 2

-- Declare the expression
def expr : ℤ := 7 * a ^ 2 * b + (-4 * a ^ 2 * b + 5 * a * b ^ 2) - (2 * a ^ 2 * b - 3 * a * b ^ 2)

-- Declare the final evaluated result
def result : ℤ := 2 * ((-1 : ℤ) ^ 2) + 8 * (-1) * (2 : ℤ) ^ 2 

-- The theorem we want to prove
theorem simplify_and_evaluate : expr = result :=
by
  sorry

end simplify_and_evaluate_l101_10199


namespace point_C_correct_l101_10115

-- Definitions of point A and B
def A : ℝ × ℝ := (4, -4)
def B : ℝ × ℝ := (18, 6)

-- Coordinate of C obtained from the conditions of the problem
def C : ℝ × ℝ := (25, 11)

-- Proof statement
theorem point_C_correct :
  ∃ C : ℝ × ℝ, (∃ (BC : ℝ × ℝ), BC = (1/2) • (B.1 - A.1, B.2 - A.2) ∧ C = (B.1 + BC.1, B.2 + BC.2)) ∧ C = (25, 11) :=
by
  sorry

end point_C_correct_l101_10115


namespace trig_identity_l101_10158

noncomputable def sin (x : ℝ) : ℝ := sorry
noncomputable def cos (x : ℝ) : ℝ := sorry

theorem trig_identity (θ : ℝ) : sin (θ + 75 * Real.pi / 180) + cos (θ + 45 * Real.pi / 180) - Real.sqrt 3 * cos (θ + 15 * Real.pi / 180) = 0 :=
by
  sorry

end trig_identity_l101_10158


namespace fixed_point_difference_l101_10131

noncomputable def func (a x : ℝ) : ℝ := a^x + Real.log a

theorem fixed_point_difference (a m n : ℝ) (h_pos : a > 0) (h_ne_one : a ≠ 1) :
  (func a 0 = n) ∧ (y = func a x → (x = m) ∧ (y = n)) → (m - n = -2) :=
by 
  intro h
  sorry

end fixed_point_difference_l101_10131


namespace curve_statements_incorrect_l101_10119

theorem curve_statements_incorrect (t : ℝ) :
  (1 < t ∧ t < 3 → ¬ ∀ x y : ℝ, (x^2 / (3 - t) + y^2 / (t - 1) = 1 → x^2 + y^2 ≠ 1)) ∧
  ((3 - t) * (t - 1) < 0 → ¬ t < 1) :=
by
  sorry

end curve_statements_incorrect_l101_10119


namespace rudy_first_run_rate_l101_10164

def first_run_rate (R : ℝ) : Prop :=
  let time_first_run := 5 * R
  let time_second_run := 4 * 9.5
  let total_time := time_first_run + time_second_run
  total_time = 88

theorem rudy_first_run_rate : first_run_rate 10 :=
by
  unfold first_run_rate
  simp
  sorry

end rudy_first_run_rate_l101_10164


namespace directrix_of_given_parabola_l101_10109

noncomputable def parabola_directrix (y : ℝ) : Prop :=
  let focus_x : ℝ := -1
  let point_on_parabola : ℝ × ℝ := (-1 / 4 * y^2, y)
  let PF_sq := (point_on_parabola.1 - focus_x)^2 + (point_on_parabola.2)^2
  let PQ_sq := (point_on_parabola.1 - 1)^2
  PF_sq = PQ_sq

theorem directrix_of_given_parabola : parabola_directrix y :=
sorry

end directrix_of_given_parabola_l101_10109


namespace oil_ratio_l101_10156

theorem oil_ratio (x : ℝ) (initial_small_tank : ℝ) (initial_large_tank : ℝ) (total_capacity_large : ℝ)
  (half_capacity_large : ℝ) (additional_needed : ℝ) :
  initial_small_tank = 4000 ∧ initial_large_tank = 3000 ∧ total_capacity_large = 20000 ∧
  half_capacity_large = total_capacity_large / 2 ∧ additional_needed = 4000 ∧
  (initial_large_tank + x + additional_needed = half_capacity_large) →
  x / initial_small_tank = 3 / 4 :=
by
  intro h
  rcases h with ⟨h1, h2, h3, h4, h5, h6⟩
  sorry

end oil_ratio_l101_10156


namespace find_g_inv_f_neg7_l101_10127

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry
noncomputable def f_inv : ℝ → ℝ := sorry
noncomputable def g_inv : ℝ → ℝ := sorry

axiom f_inv_def : ∀ x, f_inv (g x) = 5 * x + 3

theorem find_g_inv_f_neg7 : g_inv (f (-7)) = -2 :=
by
  sorry

end find_g_inv_f_neg7_l101_10127


namespace percentage_of_apples_sold_l101_10105

variables (A P : ℝ) 

theorem percentage_of_apples_sold :
  (A = 700) →
  (A * (1 - P / 100) = 420) →
  (P = 40) :=
by
  intros h1 h2
  sorry

end percentage_of_apples_sold_l101_10105


namespace smallest_number_divisible_by_given_numbers_diminished_by_10_is_55450_l101_10113

theorem smallest_number_divisible_by_given_numbers_diminished_by_10_is_55450 :
  ∃ n : ℕ, (n - 10) % 12 = 0 ∧
           (n - 10) % 16 = 0 ∧
           (n - 10) % 18 = 0 ∧
           (n - 10) % 21 = 0 ∧
           (n - 10) % 28 = 0 ∧
           (n - 10) % 35 = 0 ∧
           (n - 10) % 40 = 0 ∧
           (n - 10) % 45 = 0 ∧
           (n - 10) % 55 = 0 ∧
           n = 55450 :=
by
  sorry

end smallest_number_divisible_by_given_numbers_diminished_by_10_is_55450_l101_10113


namespace inequality_no_solution_l101_10174

theorem inequality_no_solution : 
  ∀ x : ℝ, -2 < (x^2 - 10 * x + 9) / (x^2 - 4 * x + 8) ∧ (x^2 - 10 * x + 9) / (x^2 - 4 * x + 8) < 2 → false :=
by sorry

end inequality_no_solution_l101_10174


namespace maximize_profit_l101_10195

def revenue (x : ℝ) : ℝ := 17 * x^2
def cost (x : ℝ) : ℝ := 2 * x^3 - x^2
def profit (x : ℝ) : ℝ := revenue x - cost x

theorem maximize_profit : ∃ x > 0, profit x = 18 * x^2 - 2 * x^3 ∧ (∀ y > 0, y ≠ x → profit y < profit x) :=
by
  sorry

end maximize_profit_l101_10195


namespace min_k_period_at_least_15_l101_10185

theorem min_k_period_at_least_15 (a b : ℚ) (h_pos_a : 0 < a) (h_pos_b : 0 < b)
    (h_period_a : ∃ m, a = m / (10^30 - 1))
    (h_period_b : ∃ n, b = n / (10^30 - 1))
    (h_period_ab : ∃ p, (a - b) = p / (10^30 - 1) ∧ 10^15 + 1 ∣ p) :
    ∃ k : ℕ, k = 6 ∧ (∃ q, (a + k * b) = q / (10^30 - 1) ∧ 10^15 + 1 ∣ q) :=
sorry

end min_k_period_at_least_15_l101_10185


namespace conner_collected_on_day_two_l101_10168

variable (s0 : ℕ) (c0 : ℕ) (s1 : ℕ) (c1 : ℕ) (c2 : ℕ) (s3 : ℕ) (c3 : ℕ) (total_sydney : ℕ) (total_conner : ℕ)

theorem conner_collected_on_day_two :
  s0 = 837 ∧ c0 = 723 ∧ 
  s1 = 4 ∧ c1 = 8 * s1 ∧
  s3 = 2 * c1 ∧ c3 = 27 ∧
  total_sydney = s0 + s1 + s3 ∧
  total_conner = c0 + c1 + c2 + c3 ∧
  total_conner >= total_sydney
  → c2 = 123 :=
by
  sorry

end conner_collected_on_day_two_l101_10168


namespace allison_greater_probability_l101_10133

-- Definitions and conditions for the problem
def faceRollAllison : Nat := 6
def facesBrian : List Nat := [1, 3, 3, 5, 5, 6]
def facesNoah : List Nat := [4, 4, 4, 4, 5, 5]

-- Function to calculate probability
def probability_less_than (faces : List Nat) (value : Nat) : ℚ :=
  (faces.filter (fun x => x < value)).length / faces.length

-- Main theorem statement
theorem allison_greater_probability :
  probability_less_than facesBrian 6 * probability_less_than facesNoah 6 = 5 / 6 := by
  sorry

end allison_greater_probability_l101_10133


namespace no_discount_profit_percentage_l101_10148

noncomputable def cost_price : ℝ := 100
noncomputable def discount_percentage : ℝ := 4 / 100  -- 4%
noncomputable def profit_percentage_with_discount : ℝ := 20 / 100  -- 20%

theorem no_discount_profit_percentage : 
  (1 + profit_percentage_with_discount) * cost_price / (1 - discount_percentage) / cost_price - 1 = 0.25 := by
  sorry

end no_discount_profit_percentage_l101_10148


namespace math_study_time_l101_10152

-- Conditions
def science_time : ℕ := 25
def total_time : ℕ := 60

-- Theorem statement
theorem math_study_time :
  total_time - science_time = 35 := by
  -- Proof placeholder
  sorry

end math_study_time_l101_10152


namespace calculate_total_interest_l101_10163

theorem calculate_total_interest :
  let total_money := 9000
  let invested_at_8_percent := 4000
  let invested_at_9_percent := total_money - invested_at_8_percent
  let interest_rate_8 := 0.08
  let interest_rate_9 := 0.09
  let interest_from_8_percent := invested_at_8_percent * interest_rate_8
  let interest_from_9_percent := invested_at_9_percent * interest_rate_9
  let total_interest := interest_from_8_percent + interest_from_9_percent
  total_interest = 770 :=
by
  sorry

end calculate_total_interest_l101_10163


namespace find_alpha_plus_beta_l101_10186

open Real

theorem find_alpha_plus_beta 
  (α β : ℝ)
  (h1 : sin α = sqrt 5 / 5)
  (h2 : sin β = sqrt 10 / 10)
  (h3 : π / 2 < α ∧ α < π)
  (h4 : π / 2 < β ∧ β < π) :
  α + β = 7 * π / 4 :=
sorry

end find_alpha_plus_beta_l101_10186


namespace find_joe_age_l101_10128

noncomputable def billy_age (joe_age : ℕ) : ℕ := 3 * joe_age
noncomputable def emily_age (billy_age joe_age : ℕ) : ℕ := (billy_age + joe_age) / 2

theorem find_joe_age (joe_age : ℕ) 
    (h1 : billy_age joe_age = 3 * joe_age)
    (h2 : emily_age (billy_age joe_age) joe_age = (billy_age joe_age + joe_age) / 2)
    (h3 : billy_age joe_age + joe_age + emily_age (billy_age joe_age) joe_age = 90) : 
    joe_age = 15 :=
by
  sorry

end find_joe_age_l101_10128


namespace problem_l101_10188

theorem problem (x : ℝ) (h : 8 * x = 3) : 200 * (1 / x) = 533.33 := by
  sorry

end problem_l101_10188


namespace length_of_bridge_is_255_l101_10144

noncomputable def bridge_length (train_length : ℕ) (train_speed_kph : ℕ) (cross_time_sec : ℕ) : ℕ :=
  let train_speed_mps := train_speed_kph * 1000 / (60 * 60)
  let total_distance := train_speed_mps * cross_time_sec
  total_distance - train_length

theorem length_of_bridge_is_255 :
  ∀ (train_length : ℕ) (train_speed_kph : ℕ) (cross_time_sec : ℕ), 
    train_length = 120 →
    train_speed_kph = 45 →
    cross_time_sec = 30 →
    bridge_length train_length train_speed_kph cross_time_sec = 255 :=
by
  intros train_length train_speed_kph cross_time_sec htl htsk hcts
  simp [bridge_length]
  rw [htl, htsk, hcts]
  norm_num
  sorry

end length_of_bridge_is_255_l101_10144


namespace f_inequality_l101_10123

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := a * (Real.exp x + a) - x

theorem f_inequality (a : ℝ) (h : a > 0) : ∀ x : ℝ, f x a > 2 * Real.log a + 3 / 2 :=
sorry

end f_inequality_l101_10123


namespace count_integers_abs_inequality_l101_10159

theorem count_integers_abs_inequality : 
  ∃ n : ℕ, n = 15 ∧ ∀ x : ℤ, |(x: ℝ) - 3| ≤ 7.2 ↔ x ∈ {i : ℤ | -4 ≤ i ∧ i ≤ 10} := 
by 
  sorry

end count_integers_abs_inequality_l101_10159


namespace range_of_y_l101_10197

theorem range_of_y (a b y : ℝ) (hab : a + b = 2) (hbl : b ≤ 2) (hy : y = a^2 + 2*a - 2) : y ≥ -2 :=
by
  sorry

end range_of_y_l101_10197


namespace set_difference_lt3_gt0_1_leq_x_leq_2_l101_10130

def A := {x : ℝ | |x| < 3}
def B := {x : ℝ | x^2 - 3 * x + 2 > 0}

theorem set_difference_lt3_gt0_1_leq_x_leq_2 : {x : ℝ | x ∈ A ∧ x ∉ (A ∩ B)} = {x : ℝ | 1 ≤ x ∧ x ≤ 2} := by
  sorry

end set_difference_lt3_gt0_1_leq_x_leq_2_l101_10130


namespace line_passes_through_vertex_count_l101_10101

theorem line_passes_through_vertex_count :
  (∃ a : ℝ, ∀ (x : ℝ), x = 0 → (x + a = a^2)) ↔ (∀ a : ℝ, (a = 0 ∨ a = 1)) :=
by
  sorry

end line_passes_through_vertex_count_l101_10101


namespace intersection_PQ_eq_23_l101_10165

def P : Set ℝ := {x : ℝ | 1 < x ∧ x < 3}
def Q : Set ℝ := {x : ℝ | 2 < x}

theorem intersection_PQ_eq_23 : P ∩ Q = {x : ℝ | 2 < x ∧ x < 3} := 
by {
  sorry
}

end intersection_PQ_eq_23_l101_10165


namespace metal_waste_l101_10134

theorem metal_waste (l b : ℝ) (h : l > b) : l * b - (b^2 / 2) = 
  (l * b - (π * (b / 2)^2)) + (π * (b / 2)^2 - (b^2 / 2)) := by
  sorry

end metal_waste_l101_10134


namespace johns_gas_usage_per_week_l101_10126

def mpg : ℕ := 30
def miles_to_work_each_way : ℕ := 20
def days_per_week_to_work : ℕ := 5
def leisure_miles_per_week : ℕ := 40

theorem johns_gas_usage_per_week : 
  (2 * miles_to_work_each_way * days_per_week_to_work + leisure_miles_per_week) / mpg = 8 :=
by
  sorry

end johns_gas_usage_per_week_l101_10126


namespace hundredth_odd_integer_is_199_sum_of_first_100_odd_integers_is_10000_l101_10117

noncomputable def nth_odd_positive_integer (n : ℕ) : ℕ :=
  2 * n - 1

noncomputable def sum_first_n_odd_positive_integers (n : ℕ) : ℕ :=
  n * n

theorem hundredth_odd_integer_is_199 : nth_odd_positive_integer 100 = 199 :=
  by
  sorry

theorem sum_of_first_100_odd_integers_is_10000 : sum_first_n_odd_positive_integers 100 = 10000 :=
  by
  sorry

end hundredth_odd_integer_is_199_sum_of_first_100_odd_integers_is_10000_l101_10117


namespace total_right_handed_players_l101_10183

-- Defining the conditions and the given values
def total_players : ℕ := 61
def throwers : ℕ := 37
def non_throwers : ℕ := total_players - throwers
def left_handed_non_throwers : ℕ := non_throwers / 3
def right_handed_non_throwers : ℕ := non_throwers - left_handed_non_throwers

-- The proof goal
theorem total_right_handed_players 
  (h1 : total_players = 61)
  (h2 : throwers = 37)
  (h3 : non_throwers = total_players - throwers)
  (h4 : left_handed_non_throwers = non_throwers / 3)
  (h5 : right_handed_non_throwers = non_throwers - left_handed_non_throwers)
  (h6 : left_handed_non_throwers * 3 = non_throwers)
  : throwers + right_handed_non_throwers = 53 :=
sorry

end total_right_handed_players_l101_10183
