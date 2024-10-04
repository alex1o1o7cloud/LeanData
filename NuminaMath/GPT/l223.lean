import Mathlib

namespace count_valid_four_digit_numbers_l223_223810

theorem count_valid_four_digit_numbers : 
  let valid_first_digits := (4*5 + 4*4)
  let valid_last_digits := (5*5 + 4*4)
  valid_first_digits * valid_last_digits = 1476 :=
by
  sorry

end count_valid_four_digit_numbers_l223_223810


namespace specificTriangle_perimeter_l223_223449

-- Assume a type to represent triangle sides
structure IsoscelesTriangle (a b : ℕ) : Prop :=
  (equal_sides : a = b ∨ a + b > max a b)

-- Define the condition where we have specific sides
def specificTriangle : Prop :=
  IsoscelesTriangle 5 2

-- Prove that given the specific sides, the perimeter is 12
theorem specificTriangle_perimeter : specificTriangle → 5 + 5 + 2 = 12 :=
by
  intro h
  cases h
  sorry

end specificTriangle_perimeter_l223_223449


namespace amount_left_after_pool_l223_223811

def amount_left (total_earned : ℝ) (cost_per_person : ℝ) (num_people : ℕ) : ℝ :=
  total_earned - (cost_per_person * num_people)

theorem amount_left_after_pool :
  amount_left 30 2.5 10 = 5 :=
by
  sorry

end amount_left_after_pool_l223_223811


namespace expression_value_l223_223555

theorem expression_value : 
  29^2 - 27^2 + 25^2 - 23^2 + 21^2 - 19^2 + 17^2 - 15^2 + 13^2 - 11^2 + 9^2 - 7^2 + 5^2 - 3^2 + 1^2 = 389 :=
by
  sorry

end expression_value_l223_223555


namespace ellipse_condition_sufficient_not_necessary_l223_223981

theorem ellipse_condition_sufficient_not_necessary (n : ℝ) :
  (-1 < n) ∧ (n < 2) → 
  (2 - n > 0) ∧ (n + 1 > 0) ∧ (2 - n > n + 1) :=
by
  intro h
  sorry

end ellipse_condition_sufficient_not_necessary_l223_223981


namespace Sammy_has_8_bottle_caps_l223_223063

-- Definitions representing the conditions
def BilliesBottleCaps := 2
def JaninesBottleCaps := 3 * BilliesBottleCaps
def SammysBottleCaps := JaninesBottleCaps + 2

-- Goal: Prove that Sammy has 8 bottle caps
theorem Sammy_has_8_bottle_caps : 
  SammysBottleCaps = 8 := 
sorry

end Sammy_has_8_bottle_caps_l223_223063


namespace sin_sum_diff_l223_223320

theorem sin_sum_diff (α β : ℝ) 
  (hα : Real.sin α = 1/3) 
  (hβ : Real.sin β = 1/2) : 
  Real.sin (α + β) * Real.sin (α - β) = -5/36 := 
sorry

end sin_sum_diff_l223_223320


namespace pool_capacity_l223_223225

theorem pool_capacity (hose_rate leak_rate : ℝ) (fill_time : ℝ) (net_rate := hose_rate - leak_rate) (total_water := net_rate * fill_time) :
  hose_rate = 1.6 → 
  leak_rate = 0.1 → 
  fill_time = 40 → 
  total_water = 60 := by
  intros
  sorry

end pool_capacity_l223_223225


namespace angle_measure_l223_223074

theorem angle_measure (x : ℝ) (h : x + (3 * x - 10) = 180) : x = 47.5 := 
by
  sorry

end angle_measure_l223_223074


namespace compute_f_seven_halves_l223_223220

theorem compute_f_seven_halves 
  (f : ℝ → ℝ)
  (h_odd : ∀ x, f (-x) = -f x)
  (h_shift : ∀ x, f (x + 2) = -f x)
  (h_interval : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = x) :
  f (7 / 2) = -1 / 2 :=
  sorry

end compute_f_seven_halves_l223_223220


namespace farmer_apples_count_l223_223236

-- Definitions from the conditions in step a)
def initial_apples : ℕ := 127
def apples_given_away : ℕ := 88

-- Proof goal from step c)
theorem farmer_apples_count : initial_apples - apples_given_away = 39 :=
by
  sorry

end farmer_apples_count_l223_223236


namespace tv_price_with_tax_l223_223462

-- Define the original price of the TV
def originalPrice : ℝ := 1700

-- Define the value-added tax rate
def taxRate : ℝ := 0.15

-- Calculate the total price including tax
theorem tv_price_with_tax : originalPrice * (1 + taxRate) = 1955 :=
by
  sorry

end tv_price_with_tax_l223_223462


namespace judy_shopping_trip_l223_223560

-- Define the quantities and prices of the items
def num_carrots : ℕ := 5
def price_carrot : ℕ := 1
def num_milk : ℕ := 4
def price_milk : ℕ := 3
def num_pineapples : ℕ := 2
def price_pineapple : ℕ := 4
def num_flour : ℕ := 2
def price_flour : ℕ := 5
def price_ice_cream : ℕ := 7

-- Define the promotion conditions
def pineapple_promotion : ℕ := num_pineapples / 2

-- Define the coupon condition
def coupon_threshold : ℕ := 40
def coupon_value : ℕ := 10

-- Define the total cost without coupon
def total_cost : ℕ := 
  (num_carrots * price_carrot) + 
  (num_milk * price_milk) +
  (pineapple_promotion * price_pineapple) +
  (num_flour * price_flour) +
  price_ice_cream

-- Define the final cost considering the coupon condition
def final_cost : ℕ :=
  if total_cost < coupon_threshold then total_cost else total_cost - coupon_value

-- The theorem to be proven
theorem judy_shopping_trip : final_cost = 38 := by
  sorry

end judy_shopping_trip_l223_223560


namespace power_of_powers_eval_powers_l223_223130

theorem power_of_powers (a m n : ℕ) : (a^m)^n = a^(m * n) :=
  by sorry

theorem eval_powers : (3^2)^4 = 6561 :=
  by
    rw [power_of_powers]
    -- further proof of (3^8 = 6561) would go here
    sorry

end power_of_powers_eval_powers_l223_223130


namespace expand_and_simplify_l223_223432

variable (y : ℝ)

theorem expand_and_simplify :
  -2 * (5 * y^3 - 4 * y^2 + 3 * y - 6) = -10 * y^3 + 8 * y^2 - 6 * y + 12 :=
  sorry

end expand_and_simplify_l223_223432


namespace min_odd_in_A_P_l223_223613

-- Define the polynomial P of degree 8
variables (P : Polynomial ℝ)
hypothesis degree_P : P.natDegree = 8

-- Define the set A_P
def A_P (P : Polynomial ℝ) : Set ℝ := { x : ℝ | True }

-- Hypothesis that the set A_P includes the number 8
hypothesis A_P_contains_8 : 8 ∈ A_P P

-- The minimum number of odd numbers in the set A_P
theorem min_odd_in_A_P : ∃ n : ℕ, odd n ∧ n = 1 := sorry

end min_odd_in_A_P_l223_223613


namespace average_distance_per_day_l223_223640

def distance_Monday : ℝ := 4.2
def distance_Tuesday : ℝ := 3.8
def distance_Wednesday : ℝ := 3.6
def distance_Thursday : ℝ := 4.4

def total_distance : ℝ := distance_Monday + distance_Tuesday + distance_Wednesday + distance_Thursday

def number_of_days : ℕ := 4

theorem average_distance_per_day : total_distance / number_of_days = 4 := by
  sorry

end average_distance_per_day_l223_223640


namespace alfred_gain_percent_l223_223148

theorem alfred_gain_percent :
  let purchase_price := 4700
  let repair_costs := 800
  let selling_price := 5800
  let total_cost := purchase_price + repair_costs
  let gain := selling_price - total_cost
  let gain_percent := (gain / total_cost) * 100
  gain_percent = 5.45 := 
by
  sorry

end alfred_gain_percent_l223_223148


namespace opposite_of_5_is_neg5_l223_223991

def opposite (n x : ℤ) := n + x = 0

theorem opposite_of_5_is_neg5 : opposite 5 (-5) :=
by
  sorry

end opposite_of_5_is_neg5_l223_223991


namespace gcd_sequence_l223_223964

theorem gcd_sequence (n : ℕ) : gcd ((7^n - 1)/6) ((7^(n+1) - 1)/6) = 1 := by
  sorry

end gcd_sequence_l223_223964


namespace smallest_possible_sum_l223_223327

theorem smallest_possible_sum (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (hneq : a ≠ b) 
  (heq : (1 / a : ℚ) + (1 / b) = 1 / 12) : a + b = 49 :=
sorry

end smallest_possible_sum_l223_223327


namespace power_of_powers_l223_223082

theorem power_of_powers :
  (3^2)^4 = 6561 := 
by 
  sorry

end power_of_powers_l223_223082


namespace opposite_of_five_is_neg_five_l223_223992

theorem opposite_of_five_is_neg_five :
  ∃ (x : ℤ), (5 + x = 0) ∧ x = -5 :=
by
  use -5
  split
  · simp
  · rfl

end opposite_of_five_is_neg_five_l223_223992


namespace probability_roll_2_given_sum_3_l223_223860

open ProbabilityTheory

-- Define the structure of the unusual die
noncomputable def unusual_die : finset ℕ := {1, 2, 1, 1, 1, 2}

-- Define the probability space of rolling this die
noncomputable def die_probability_space := Finset.universalMeasure unusual_die

-- Define the event that the sum of rolled results is 3
def event_sum_3 (rolls : list ℕ) : Prop := rolls.sum = 3

-- Define the event that a roll resulted in a 2
def event_roll_2 (rolls : list ℕ) : Prop := 2 ∈ rolls

-- The theorem to prove the probability
theorem probability_roll_2_given_sum_3 (S : list ℕ) (hS : event_sum_3 S) :
  condProb (event_roll_2 S) (event_sum_3 S) die_probability_space = 0.6 := 
sorry

end probability_roll_2_given_sum_3_l223_223860


namespace farmer_apples_count_l223_223234

theorem farmer_apples_count (initial : ℕ) (given : ℕ) (remaining : ℕ) 
  (h1 : initial = 127) (h2 : given = 88) : remaining = initial - given := 
by
  sorry

end farmer_apples_count_l223_223234


namespace root_in_interval_l223_223313

def polynomial (x : ℝ) := x^3 + 3 * x^2 - x + 1

noncomputable def A : ℤ := -4
noncomputable def B : ℤ := -3

theorem root_in_interval : (∃ x : ℝ, polynomial x = 0 ∧ (A : ℝ) < x ∧ x < (B : ℝ)) :=
sorry

end root_in_interval_l223_223313


namespace triangle_inequality_for_n6_l223_223722

variables {a b c : ℝ} {n : ℕ}
open Real

-- Define the main statement as a theorem
theorem triangle_inequality_for_n6 (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c)
  (ineq : 6 * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2)) :
  a + b > c ∧ b + c > a ∧ c + a > b :=
sorry

end triangle_inequality_for_n6_l223_223722


namespace fraction_of_total_calls_l223_223857

-- Definitions based on conditions
variable (B : ℚ) -- Calls processed by each member of Team B
variable (N : ℚ) -- Number of members in Team B

-- The fraction of calls processed by each member of Team A
def team_A_call_fraction : ℚ := 1 / 5

-- The fraction of calls processed by each member of Team C
def team_C_call_fraction : ℚ := 7 / 8

-- The fraction of agents in Team A relative to Team B
def team_A_agents_fraction : ℚ := 5 / 8

-- The fraction of agents in Team C relative to Team B
def team_C_agents_fraction : ℚ := 3 / 4

-- Total calls processed by Team A, Team B, and Team C
def total_calls_team_A : ℚ := (B * team_A_call_fraction) * (N * team_A_agents_fraction)
def total_calls_team_B : ℚ := B * N
def total_calls_team_C : ℚ := (B * team_C_call_fraction) * (N * team_C_agents_fraction)

-- Sum of total calls processed by all teams
def total_calls_all_teams : ℚ := total_calls_team_A B N + total_calls_team_B B N + total_calls_team_C B N

-- Potential total calls if all teams were as efficient as Team B
def potential_total_calls : ℚ := 3 * (B * N)

-- Fraction of total calls processed by all teams combined
def processed_fraction : ℚ := total_calls_all_teams B N / potential_total_calls B N

theorem fraction_of_total_calls : processed_fraction B N = 19 / 32 :=
by
  sorry -- Proof omitted

end fraction_of_total_calls_l223_223857


namespace exponentiation_example_l223_223116

theorem exponentiation_example : (3^2)^4 = 6561 := by
  sorry

end exponentiation_example_l223_223116


namespace cole_round_trip_time_l223_223179

-- Define the relevant quantities
def speed_to_work : ℝ := 70 -- km/h
def speed_to_home : ℝ := 105 -- km/h
def time_to_work_mins : ℝ := 72 -- minutes

-- Define the theorem to be proved
theorem cole_round_trip_time : 
  (time_to_work_mins / 60 + (speed_to_work * time_to_work_mins / 60) / speed_to_home) = 2 :=
by
  sorry

end cole_round_trip_time_l223_223179


namespace find_d_value_l223_223888

theorem find_d_value (a b : ℚ) (d : ℚ) (h1 : a = 2) (h2 : b = 11) 
  (h3 : ∀ x, 2 * x^2 + 11 * x + d = 0 ↔ x = (-11 + Real.sqrt 15) / 4 ∨ x = (-11 - Real.sqrt 15) / 4) : 
  d = 53 / 4 :=
sorry

end find_d_value_l223_223888


namespace min_odd_in_A_P_l223_223612

-- Define the polynomial P of degree 8
variables (P : Polynomial ℝ)
hypothesis degree_P : P.natDegree = 8

-- Define the set A_P
def A_P (P : Polynomial ℝ) : Set ℝ := { x : ℝ | True }

-- Hypothesis that the set A_P includes the number 8
hypothesis A_P_contains_8 : 8 ∈ A_P P

-- The minimum number of odd numbers in the set A_P
theorem min_odd_in_A_P : ∃ n : ℕ, odd n ∧ n = 1 := sorry

end min_odd_in_A_P_l223_223612


namespace neg_p_equiv_exists_leq_l223_223334

-- Define the given proposition p
def p : Prop := ∀ x : ℝ, x^2 - x + 1 > 0

-- State the equivalence we need to prove
theorem neg_p_equiv_exists_leq :
  ¬ p ↔ ∃ x : ℝ, x^2 - x + 1 ≤ 0 :=
by {
  sorry  -- Proof is skipped as per instructions
}

end neg_p_equiv_exists_leq_l223_223334


namespace total_hours_proof_l223_223657

-- Conditions
def half_hour_show_episodes : ℕ := 24
def one_hour_show_episodes : ℕ := 12
def half_hour_per_episode : ℝ := 0.5
def one_hour_per_episode : ℝ := 1.0

-- Define the total hours Tim watched
def total_hours_watched : ℝ :=
  half_hour_show_episodes * half_hour_per_episode + one_hour_show_episodes * one_hour_per_episode

-- Prove that the total hours watched is 24
theorem total_hours_proof : total_hours_watched = 24 := by
  sorry

end total_hours_proof_l223_223657


namespace trajectory_equation_l223_223744

theorem trajectory_equation (x y : ℝ) (M O A : ℝ × ℝ)
    (hO : O = (0, 0)) (hA : A = (3, 0))
    (h_ratio : dist M O / dist M A = 1 / 2) : 
    x^2 + y^2 + 2 * x - 3 = 0 :=
by
  -- Definition of points
  let M := (x, y)
  exact sorry

end trajectory_equation_l223_223744


namespace combined_total_score_l223_223203

-- Define the conditions
def num_single_answer_questions : ℕ := 50
def num_multiple_answer_questions : ℕ := 20
def single_answer_score : ℕ := 2
def multiple_answer_score : ℕ := 4
def wrong_single_penalty : ℕ := 1
def wrong_multiple_penalty : ℕ := 2
def jose_wrong_single : ℕ := 10
def jose_wrong_multiple : ℕ := 5
def jose_lost_marks : ℕ := (jose_wrong_single * wrong_single_penalty) + (jose_wrong_multiple * wrong_multiple_penalty)
def jose_correct_single : ℕ := num_single_answer_questions - jose_wrong_single
def jose_correct_multiple : ℕ := num_multiple_answer_questions - jose_wrong_multiple
def jose_single_score : ℕ := jose_correct_single * single_answer_score
def jose_multiple_score : ℕ := jose_correct_multiple * multiple_answer_score
def jose_score : ℕ := (jose_single_score + jose_multiple_score) - jose_lost_marks
def alison_score : ℕ := jose_score - 50
def meghan_score : ℕ := jose_score - 30

-- Prove the combined total score
theorem combined_total_score :
  jose_score + alison_score + meghan_score = 280 :=
by
  sorry

end combined_total_score_l223_223203


namespace final_price_jacket_l223_223293

-- Defining the conditions as per the problem
def original_price : ℚ := 250
def first_discount_rate : ℚ := 0.40
def second_discount_rate : ℚ := 0.15
def tax_rate : ℚ := 0.05

-- Defining the calculation steps
def first_discounted_price : ℚ := original_price * (1 - first_discount_rate)
def second_discounted_price : ℚ := first_discounted_price * (1 - second_discount_rate)
def final_price_inclusive_tax : ℚ := second_discounted_price * (1 + tax_rate)

-- The proof problem statement
theorem final_price_jacket : final_price_inclusive_tax = 133.88 := sorry

end final_price_jacket_l223_223293


namespace problem_U_complement_eq_l223_223916

universe u
variable {α : Type u} [DecidableEq α]
variable (U M : Set α)

theorem problem_U_complement_eq
  (hU : U = ({1, 2, 3, 4, 5} : Set ℕ))
  (hM : U \ M = ({1, 3} : Set ℕ)) : 
  (2 : ℕ) ∈ M := 
by
  sorry

end problem_U_complement_eq_l223_223916


namespace hypotenuse_length_triangle_l223_223862

theorem hypotenuse_length_triangle (a b c : ℝ) (h1 : a + b + c = 40) (h2 : (1/2) * a * b = 30) 
  (h3 : a = b) : c = 2 * Real.sqrt 30 :=
by
  sorry

end hypotenuse_length_triangle_l223_223862


namespace cos_double_angle_at_origin_l223_223898

noncomputable def vertex : ℝ × ℝ := (0, 0)
noncomputable def initial_side : ℝ × ℝ := (1, 0)
noncomputable def terminal_side : ℝ × ℝ := (-1, 3)
noncomputable def cos2alpha (v i t : ℝ × ℝ) : ℝ :=
  2 * ((t.1) / (Real.sqrt (t.1 ^ 2 + t.2 ^ 2))) ^ 2 - 1

theorem cos_double_angle_at_origin :
  cos2alpha vertex initial_side terminal_side = -4 / 5 :=
by
  sorry

end cos_double_angle_at_origin_l223_223898


namespace unique_zero_function_l223_223983

theorem unique_zero_function
    (f : ℝ → ℝ)
    (H : ∀ x y : ℝ, x + y ≠ 0 → f (x * y) = (f x + f y) / (x + y)) :
    ∀ x : ℝ, f x = 0 := 
by 
     sorry

end unique_zero_function_l223_223983


namespace greatest_integer_for_prime_abs_expression_l223_223512

open Int

-- Define the quadratic expression and the prime condition
def quadratic_expression (x : ℤ) : ℤ := 6 * x^2 - 47 * x + 15

-- Statement that |quadratic_expression x| is prime
def is_prime_quadratic_expression (x : ℤ) : Prop :=
  Prime (abs (quadratic_expression x))

-- Prove that the greatest integer x such that |quadratic_expression x| is prime is 8
theorem greatest_integer_for_prime_abs_expression :
  ∃ (x : ℤ), is_prime_quadratic_expression x ∧ (∀ (y : ℤ), is_prime_quadratic_expression y → y ≤ x) → x = 8 :=
by
  sorry

end greatest_integer_for_prime_abs_expression_l223_223512


namespace totalPeaches_l223_223828

-- Definition of conditions in the problem
def redPeaches : Nat := 4
def greenPeaches : Nat := 6
def numberOfBaskets : Nat := 1

-- Mathematical proof problem
theorem totalPeaches : numberOfBaskets * (redPeaches + greenPeaches) = 10 := by
  sorry

end totalPeaches_l223_223828


namespace kate_change_l223_223216

def first_candy_cost : ℝ := 0.54
def second_candy_cost : ℝ := 0.35
def third_candy_cost : ℝ := 0.68
def amount_given : ℝ := 5.00

theorem kate_change : amount_given - (first_candy_cost + second_candy_cost + third_candy_cost) = 3.43 := by
  sorry

end kate_change_l223_223216


namespace part_one_part_two_l223_223702

-- First part: Prove that \( (1)(-1)^{2017}+(\frac{1}{2})^{-2}+(3.14-\pi)^{0} = 4\)
theorem part_one : (1 * (-1:ℤ)^2017 + (1/2)^(-2:ℤ) + (3.14 - Real.pi)^0 : ℝ) = 4 := 
  sorry

-- Second part: Prove that \( ((-2*x^2)^3 + 4*x^3*x^3) = -4*x^6 \)
theorem part_two (x : ℝ) : ((-2*x^2)^3 + 4*x^3*x^3) = -4*x^6 := 
  sorry

end part_one_part_two_l223_223702


namespace exp_eval_l223_223103

theorem exp_eval : (3^2)^4 = 6561 := by
sorry

end exp_eval_l223_223103


namespace div_by_19_l223_223057

theorem div_by_19 (n : ℕ) : 19 ∣ (26^n - 7^n) :=
sorry

end div_by_19_l223_223057


namespace probability_top_red_second_black_l223_223406

def num_red_cards : ℕ := 39
def num_black_cards : ℕ := 39
def total_cards : ℕ := 78

theorem probability_top_red_second_black :
  (num_red_cards * num_black_cards) / (total_cards * (total_cards - 1)) = 507 / 2002 := 
sorry

end probability_top_red_second_black_l223_223406


namespace isosceles_triangle_base_length_l223_223796

-- Define the conditions
def side_length : ℕ := 7
def perimeter : ℕ := 23

-- Define the theorem to prove the length of the base
theorem isosceles_triangle_base_length (b : ℕ) (h : 2 * side_length + b = perimeter) : b = 9 :=
by
  sorry

end isosceles_triangle_base_length_l223_223796


namespace geometric_sequence_b_l223_223242

theorem geometric_sequence_b (b : ℝ) (h : b > 0) (s : ℝ) 
  (h1 : 30 * s = b) (h2 : b * s = 15 / 4) : 
  b = 15 * Real.sqrt 2 / 2 := 
by
  sorry

end geometric_sequence_b_l223_223242


namespace joint_purchases_popular_l223_223272

-- Define the conditions stating what makes joint purchases feasible
structure Conditions where
  cost_saving : Prop  -- Joint purchases allow significant cost savings.
  shared_overhead : Prop  -- Overhead costs are distributed among all members.
  collective_quality_assessment : Prop  -- Enhanced quality assessment via collective feedback.
  community_trust : Prop  -- Trust within the community encourages honest feedback.

-- Define the proposition stating the popularity of joint purchases
theorem joint_purchases_popular (cond : Conditions) : 
  cond.cost_saving ∧ cond.shared_overhead ∧ cond.collective_quality_assessment ∧ cond.community_trust → 
  Prop := 
by 
  intro h
  sorry

end joint_purchases_popular_l223_223272


namespace original_bill_l223_223789

theorem original_bill (m : ℝ) (h1 : 10 * (m / 10) = m)
                      (h2 : 9 * ((m - 10) / 10 + 3) = m - 10) :
  m = 180 :=
  sorry

end original_bill_l223_223789


namespace power_calc_l223_223129

theorem power_calc : (3^2)^4 = 6561 := 
by
  sorry

end power_calc_l223_223129


namespace wall_area_l223_223873

theorem wall_area (width : ℝ) (height : ℝ) (h1 : width = 2) (h2 : height = 4) : width * height = 8 := by
  sorry

end wall_area_l223_223873


namespace alice_prank_combinations_l223_223172

theorem alice_prank_combinations : 
  let monday_choices := 1
  let tuesday_choices := 3
  let wednesday_choices := 5
  let thursday_choices := 4
  let friday_choices := 1
  monday_choices * tuesday_choices * wednesday_choices * thursday_choices * friday_choices = 60 :=
by
  let monday_choices := 1
  let tuesday_choices := 3
  let wednesday_choices := 5
  let thursday_choices := 4
  let friday_choices := 1
  exact (show 1 * 3 * 5 * 4 * 1 = 60 from sorry)

end alice_prank_combinations_l223_223172


namespace find_number_l223_223155

theorem find_number : ∃ x : ℝ, x + 5 * 12 / (180 / 3) = 51 ∧ x = 50 :=
by
  sorry

end find_number_l223_223155


namespace triple_divisor_sum_6_l223_223438

-- Summarize the definition of the divisor sum function excluding the number itself
def divisorSumExcluding (n : ℕ) : ℕ :=
  (Finset.filter (λ x => x ≠ n) (Finset.range (n + 1))).sum id

-- This is the main statement that we need to prove
theorem triple_divisor_sum_6 : divisorSumExcluding (divisorSumExcluding (divisorSumExcluding 6)) = 6 := 
by sorry

end triple_divisor_sum_6_l223_223438


namespace arithmetic_sequence_30th_term_l223_223065

-- Definitions
def a₁ : ℤ := 8
def d : ℤ := -3
def n : ℕ := 30

-- The statement to be proved
theorem arithmetic_sequence_30th_term :
  a₁ + (n - 1) * d = -79 :=
by
  sorry

end arithmetic_sequence_30th_term_l223_223065


namespace reflection_xy_plane_reflection_across_point_l223_223036

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def reflect_across_xy_plane (p : Point3D) : Point3D :=
  {x := p.x, y := p.y, z := -p.z}

def reflect_across_point (a p : Point3D) : Point3D :=
  {x := 2 * a.x - p.x, y := 2 * a.y - p.y, z := 2 * a.z - p.z}

theorem reflection_xy_plane :
  reflect_across_xy_plane {x := -2, y := 1, z := 4} = {x := -2, y := 1, z := -4} :=
by sorry

theorem reflection_across_point :
  reflect_across_point {x := 1, y := 0, z := 2} {x := -2, y := 1, z := 4} = {x := -5, y := -1, z := 0} :=
by sorry

end reflection_xy_plane_reflection_across_point_l223_223036


namespace trig_identity_1_trig_identity_2_l223_223333

noncomputable def point := ℚ × ℚ

namespace TrigProblem

open Real

def point_on_terminal_side (α : ℝ) (p : point) : Prop :=
  let (x, y) := p
  ∃ r : ℝ, r = sqrt (x^2 + y^2) ∧ x/r = cos α ∧ y/r = sin α

theorem trig_identity_1 {α : ℝ} (h : point_on_terminal_side α (-4, 3)) :
  (sin (π / 2 + α) - cos (π + α)) / (sin (π / 2 - α) - sin (π - α)) = 8 / 7 :=
sorry

theorem trig_identity_2 {α : ℝ} (h : point_on_terminal_side α (-4, 3)) :
  sin α * cos α = -12 / 25 :=
sorry

end TrigProblem

end trig_identity_1_trig_identity_2_l223_223333


namespace sqrt_comparison_l223_223475

theorem sqrt_comparison :
  let a := Real.sqrt 2
  let b := Real.sqrt 7 - Real.sqrt 3
  let c := Real.sqrt 6 - Real.sqrt 2
  a > c ∧ c > b := by
{
  sorry
}

end sqrt_comparison_l223_223475


namespace mr_williams_land_percentage_l223_223809

-- Given conditions
def farm_tax_percent : ℝ := 60
def total_tax_collected : ℝ := 5000
def mr_williams_tax_paid : ℝ := 480

-- Theorem statement
theorem mr_williams_land_percentage :
  (mr_williams_tax_paid / total_tax_collected) * 100 = 9.6 := by
  sorry

end mr_williams_land_percentage_l223_223809


namespace minimum_odd_in_A_P_l223_223617

-- Define the polynomial and its properties
def polynomial_degree8 (P : ℝ[X]) : Prop :=
  P.degree = 8

-- Define the set A_P
def A_P (P : ℝ[X]) : set ℝ :=
  {x | P.eval x = 8}

-- Main theorem statement
theorem minimum_odd_in_A_P (P : ℝ[X]) (hP : polynomial_degree8 P) : ∃ n : ℕ, n = 1 ∧ ∃ x ∈ A_P P, odd x :=
sorry

end minimum_odd_in_A_P_l223_223617


namespace gingerbread_percentage_red_hats_l223_223699

def total_gingerbread_men (n_red_hats : ℕ) (n_blue_boots : ℕ) (n_both : ℕ) : ℕ :=
  n_red_hats + n_blue_boots - n_both

def percentage_with_red_hats (n_red_hats : ℕ) (total : ℕ) : ℕ :=
  (n_red_hats * 100) / total

theorem gingerbread_percentage_red_hats 
  (n_red_hats : ℕ) (n_blue_boots : ℕ) (n_both : ℕ)
  (h_red_hats : n_red_hats = 6)
  (h_blue_boots : n_blue_boots = 9)
  (h_both : n_both = 3) : 
  percentage_with_red_hats n_red_hats (total_gingerbread_men n_red_hats n_blue_boots n_both) = 50 := by
  sorry

end gingerbread_percentage_red_hats_l223_223699


namespace tan_value_l223_223007

open Real

theorem tan_value (α : ℝ) 
  (h1 : sin (α + π / 6) = -3 / 5)
  (h2 : -2 * π / 3 < α ∧ α < -π / 6) : 
  tan (4 * π / 3 - α) = -4 / 3 :=
sorry

end tan_value_l223_223007


namespace exponentiation_rule_example_l223_223092

theorem exponentiation_rule_example : (3^2)^4 = 6561 := 
by {
  -- This proof is based on the power rule (ab)^c = a^(b*c)
  have step1 : (3^2)^4 = 3^(2 * 4), 
  {
    rw pow_mul,
  },
  -- Simplification of 3^(2*4) to 3^8
  have step2 : 3^(2 * 4) = 3^8,
  {
    refl,
  },
  -- Actual computation for 3^8 = 6561
  have step3 : 3^8 = 6561,
  {
    norm_num,
  },
  -- Combining all steps
  rw [step1, step2, step3],
  exact rfl,
}

end exponentiation_rule_example_l223_223092


namespace sqrt_diff_nat_l223_223577

open Nat

theorem sqrt_diff_nat (a b : ℕ) (h : 2015 * a^2 + a = 2016 * b^2 + b) : ∃ k : ℕ, a - b = k^2 := 
by
  sorry

end sqrt_diff_nat_l223_223577


namespace exponentiation_identity_l223_223109

theorem exponentiation_identity (a m n : ℕ) : (a^m)^n = a^(m * n) :=
by sorry

example : (3^2)^4 = 6561 :=
by
  have h := exponentiation_identity 3 2 4
  rw [←h]
  norm_num

end exponentiation_identity_l223_223109


namespace rectangle_width_l223_223002

theorem rectangle_width (x y : ℝ) (h1 : x * y = 12) (h2 : x^2 + y^2 = 25) : y = 3 := 
by 
  sorry

end rectangle_width_l223_223002


namespace isosceles_triangle_base_length_l223_223801

theorem isosceles_triangle_base_length (a b P : ℕ) (h1 : a = 7) (h2 : P = 23) (h3 : P = 2 * a + b) : b = 9 :=
sorry

end isosceles_triangle_base_length_l223_223801


namespace exist_ints_a_b_l223_223734

theorem exist_ints_a_b (n : ℕ) : (∃ a b : ℤ, (n : ℤ) + a^2 = b^2) ↔ ¬ n % 4 = 2 := 
by
  sorry

end exist_ints_a_b_l223_223734


namespace gasoline_amount_added_l223_223526

noncomputable def initial_fill (capacity : ℝ) : ℝ := (3 / 4) * capacity
noncomputable def final_fill (capacity : ℝ) : ℝ := (9 / 10) * capacity
noncomputable def gasoline_added (capacity : ℝ) : ℝ := final_fill capacity - initial_fill capacity

theorem gasoline_amount_added :
  ∀ (capacity : ℝ), capacity = 24 → gasoline_added capacity = 3.6 :=
  by
    intros capacity h
    rw [h]
    have initial_fill_24 : initial_fill 24 = 18 := by norm_num [initial_fill]
    have final_fill_24 : final_fill 24 = 21.6 := by norm_num [final_fill]
    have gasoline_added_24 : gasoline_added 24 = 3.6 :=
      by rw [gasoline_added, initial_fill_24, final_fill_24]; norm_num
    exact gasoline_added_24

end gasoline_amount_added_l223_223526


namespace acute_triangle_exists_l223_223314

theorem acute_triangle_exists {a1 a2 a3 a4 a5 : ℝ} 
  (h1 : a1 + a2 > a3) (h2 : a1 + a3 > a2) (h3 : a2 + a3 > a1)
  (h4 : a2 + a3 > a4) (h5 : a3 + a4 > a2) (h6 : a2 + a4 > a3)
  (h7 : a3 + a4 > a5) (h8 : a4 + a5 > a3) (h9 : a3 + a5 > a4) : 
  ∃ (t1 t2 t3 : ℝ), (t1 + t2 > t3) ∧ (t1 + t3 > t2) ∧ (t2 + t3 > t1) ∧ (t3 ^ 2 < t1 ^ 2 + t2 ^ 2) :=
sorry

end acute_triangle_exists_l223_223314


namespace same_number_assigned_to_each_point_l223_223999

namespace EqualNumberAssignment

def is_arithmetic_mean (f : ℤ × ℤ → ℕ) (p : ℤ × ℤ) : Prop :=
  let (x, y) := p
  f (x, y) = (f (x + 1, y) + f (x - 1, y) + f (x, y + 1) + f (x, y - 1)) / 4

theorem same_number_assigned_to_each_point (f : ℤ × ℤ → ℕ) :
  (∀ p : ℤ × ℤ, is_arithmetic_mean f p) → ∃ m : ℕ, ∀ p : ℤ × ℤ, f p = m :=
by
  intros h
  sorry

end EqualNumberAssignment

end same_number_assigned_to_each_point_l223_223999


namespace range_of_b_l223_223571

def M := {p : ℝ × ℝ | p.1 ^ 2 + 2 * p.2 ^ 2 = 3}
def N (m b : ℝ) := {p : ℝ × ℝ | p.2 = m * p.1 + b}

theorem range_of_b (b : ℝ) : (∀ (m : ℝ), (∃ (p : ℝ × ℝ), p ∈ M ∧ p ∈ N m b)) ↔ 
  -Real.sqrt (6) / 2 ≤ b ∧ b ≤ Real.sqrt (6) / 2 :=
by
  sorry

end range_of_b_l223_223571


namespace ellipse_condition_l223_223889

variables (m n : ℝ)

-- Definition of the curve
def curve_eqn (x y : ℝ) := m * x^2 + n * y^2 = 1

-- Define the condition for being an ellipse
def is_ellipse (m n : ℝ) : Prop :=
  m > 0 ∧ n > 0 ∧ m ≠ n

def mn_positive (m n : ℝ) : Prop := m * n > 0

-- Prove that mn > 0 is a necessary but not sufficient condition
theorem ellipse_condition (m n : ℝ) : mn_positive m n → is_ellipse m n → False := sorry

end ellipse_condition_l223_223889


namespace satisfies_properties_l223_223265

noncomputable def f (x : ℝ) : ℝ := x^2

theorem satisfies_properties :
  (∀ x1 x2 : ℝ, f (x1 * x2) = f x1 * f x2) ∧
  (∀ x : ℝ, 0 < x → 0 < (f' x)) ∧
  (∀ x : ℝ, f' (-x) = - f' x) := 
sorry

end satisfies_properties_l223_223265


namespace find_the_number_l223_223246

theorem find_the_number (x : ℝ) (h : 150 - x = x + 68) : x = 41 :=
sorry

end find_the_number_l223_223246


namespace min_odd_in_A_P_l223_223607

-- Define the polynomial P of degree 8
variable (P : ℝ → ℝ)
-- Assume P is a polynomial of degree 8
axiom degree_P : degree P = 8

-- Define the set A_P as the set of all x for which P(x) gives a certain value
def A_P (c : ℝ) : Set ℝ := { x | P x = c }

-- Given condition
axiom include_8 : 8 ∈ A_P (P 8)

-- Define if a number is odd
def is_odd (n : ℝ) : Prop := ∃k : ℤ, n = 2 * k + 1

-- The minimum number of odd numbers that are in the set A_P,
-- given that the number 8 is included in A_P, is 1

theorem min_odd_in_A_P : ∀ P, (degree P = 8) → 8 ∈ A_P (P 8) → ∃ x ∈ A_P (P 8), is_odd x := 
by
  sorry -- proof goes here

end min_odd_in_A_P_l223_223607


namespace sum_of_roots_l223_223515

theorem sum_of_roots (x : ℝ) : (x - 4)^2 = 16 → x = 8 ∨ x = 0 := by
  intro h
  have h1 : x - 4 = 4 ∨ x - 4 = -4 := by
    sorry
  cases h1
  case inl h2 =>
    rw [h2] at h
    exact Or.inl (by linarith)
  case inr h2 =>
    rw [h2] at h
    exact Or.inr (by linarith)

end sum_of_roots_l223_223515


namespace smallest_b_factors_l223_223730

theorem smallest_b_factors (b p q : ℤ) (hb : b = p + q) (hpq : p * q = 2052) : b = 132 :=
sorry

end smallest_b_factors_l223_223730


namespace expected_value_decagonal_die_l223_223844

-- Given conditions
def decagonal_die_faces : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
def probability (n : ℕ) : ℚ := 1 / 10

-- The mathematical proof problem (statement only, no proof required)
theorem expected_value_decagonal_die : 
  (List.sum decagonal_die_faces : ℚ) / List.length decagonal_die_faces = 5.5 := by
  sorry

end expected_value_decagonal_die_l223_223844


namespace negation_proposition_equivalence_l223_223072

theorem negation_proposition_equivalence :
  (¬ ∃ x₀ : ℝ, (2 / x₀ + Real.log x₀ ≤ 0)) ↔ (∀ x : ℝ, 2 / x + Real.log x > 0) := 
sorry

end negation_proposition_equivalence_l223_223072


namespace compute_expr1_factorize_expr2_l223_223402

-- Definition for Condition 1: None explicitly stated.

-- Theorem for Question 1
theorem compute_expr1 (y : ℝ) : (y - 1) * (y + 5) = y^2 + 4*y - 5 :=
by sorry

-- Definition for Condition 2: None explicitly stated.

-- Theorem for Question 2
theorem factorize_expr2 (x y : ℝ) : -x^2 + 4*x*y - 4*y^2 = -((x - 2*y)^2) :=
by sorry

end compute_expr1_factorize_expr2_l223_223402


namespace exists_color_removal_connected_l223_223387

noncomputable theory

open_locale classical

def K20_colored := simple_graph (fin 20) -- Define the complete graph K20

-- Define the complete graph with edges being one of five colors
axiom colored_edges : K20_colored → fin 5

theorem exists_color_removal_connected :
  ∃ c : fin 5, ∀ e, colored_edges e ≠ c → (K20_colored - {e}) .conn :=
sorry

end exists_color_removal_connected_l223_223387


namespace abs_distance_equation_1_abs_distance_equation_2_l223_223255

theorem abs_distance_equation_1 (x : ℚ) : |x - (3 : ℚ)| = 5 ↔ x = 8 ∨ x = -2 := 
sorry

theorem abs_distance_equation_2 (x : ℚ) : |x - (3 : ℚ)| = |x + (1 : ℚ)| ↔ x = 1 :=
sorry

end abs_distance_equation_1_abs_distance_equation_2_l223_223255


namespace probability_of_mutual_generation_l223_223470

theorem probability_of_mutual_generation :
  let elements := ["gold", "wood", "water", "fire", "earth"],
      mutual_generation := [(gold, water), (water, wood), (wood, fire), (fire, earth), (earth, gold)],
      total_pairs := Nat.choose 5 2
  in (mutual_generation.length : ℚ) / (total_pairs : ℚ) = 1 / 2 :=
by
  sorry

end probability_of_mutual_generation_l223_223470


namespace unique_integer_triplet_solution_l223_223435

theorem unique_integer_triplet_solution (x y z : ℤ) (h : x^2 + y^2 + z^2 = 2 * x * y * z) : 
    (x = 0 ∧ y = 0 ∧ z = 0) :=
by
  sorry

end unique_integer_triplet_solution_l223_223435


namespace obtuse_angles_in_second_quadrant_l223_223848

theorem obtuse_angles_in_second_quadrant
  (θ : ℝ) 
  (is_obtuse : θ > 90 ∧ θ < 180) :
  90 < θ ∧ θ < 180 :=
by sorry

end obtuse_angles_in_second_quadrant_l223_223848


namespace Sammy_has_8_bottle_caps_l223_223060

def Billie_caps : Nat := 2
def Janine_caps (B : Nat) : Nat := 3 * B
def Sammy_caps (J : Nat) : Nat := J + 2

theorem Sammy_has_8_bottle_caps : 
  Sammy_caps (Janine_caps Billie_caps) = 8 := 
by
  sorry

end Sammy_has_8_bottle_caps_l223_223060


namespace correct_statement_l223_223911

universe u
variable (α : Type u)

noncomputable def U : Set ℕ := {1, 2, 3, 4, 5}

noncomputable def complement_U_M : Set ℕ := {1, 3}

noncomputable def M : Set ℕ := U α \ complement_U_M

theorem correct_statement : 2 ∈ M :=
by {
  sorry
}

end correct_statement_l223_223911


namespace isosceles_triangle_base_length_l223_223808

-- Definitions based on the conditions
def congruent_side : Nat := 7
def perimeter : Nat := 23

-- Statement to prove
theorem isosceles_triangle_base_length :
  let b := perimeter - 2 * congruent_side in b = 9 :=
by
  sorry

end isosceles_triangle_base_length_l223_223808


namespace sum_first_12_terms_l223_223040

theorem sum_first_12_terms (S : ℕ → ℕ) (a : ℕ → ℕ) 
  (h1 : ∀ n, S n = n * a n)
  (h2 : a 6 + a 7 = 18) : 
  S 12 = 108 :=
sorry

end sum_first_12_terms_l223_223040


namespace distance_traveled_by_light_in_10_seconds_l223_223332

theorem distance_traveled_by_light_in_10_seconds :
  ∃ (a : ℝ) (n : ℕ), (300000 * 10 : ℝ) = a * 10 ^ n ∧ n = 6 :=
sorry

end distance_traveled_by_light_in_10_seconds_l223_223332


namespace red_paint_intensity_l223_223974

theorem red_paint_intensity (x : ℝ) (h1 : 0.5 * 10 + 0.5 * x = 15) : x = 20 :=
sorry

end red_paint_intensity_l223_223974


namespace exponentiation_example_l223_223121

theorem exponentiation_example : (3^2)^4 = 6561 := by
  sorry

end exponentiation_example_l223_223121


namespace math_problem_l223_223294

variables (a b c : ℤ)

theorem math_problem (h1 : a - (b - 2 * c) = 19) (h2 : a - b - 2 * c = 7) : a - b = 13 := by
  sorry

end math_problem_l223_223294


namespace solve_exp_log_eq_correct_l223_223602

noncomputable def solve_exp_log_eq (a x : ℝ) : Prop :=
  x > 0 ∧ a ≠ 1 ∧ a > 0 → a ^ x = x ^ x + log a (log a x) → x = a

theorem solve_exp_log_eq_correct (a : ℝ) (h₁ : a ≠ 1) (h₂ : a > 0) :
  ∀ x : ℝ, solve_exp_log_eq a x :=
begin
  intros x h,
  sorry
end

end solve_exp_log_eq_correct_l223_223602


namespace engineers_percentage_calculation_l223_223598

noncomputable def percentageEngineers (num_marketers num_engineers num_managers total_salary: ℝ) : ℝ := 
  let num_employees := num_marketers + num_engineers + num_managers 
  if num_employees = 0 then 0 else num_engineers / num_employees * 100

theorem engineers_percentage_calculation : 
  let marketers_percentage := 0.7 
  let engineers_salary := 80000
  let average_salary := 80000
  let marketers_salary_total := 50000 * marketers_percentage 
  let managers_total_percent := 1 - marketers_percentage - x / 100
  let managers_salary := 370000 * managers_total_percent 
  marketers_salary_total + engineers_salary * x / 100 + managers_salary = average_salary -> 
  x = 22.76 
:= 
sorry

end engineers_percentage_calculation_l223_223598


namespace tim_watched_total_hours_tv_l223_223655

-- Define the conditions
def short_show_episodes : ℕ := 24
def short_show_duration_per_episode : ℝ := 0.5

def long_show_episodes : ℕ := 12
def long_show_duration_per_episode : ℝ := 1

-- Define the total duration for each show
def short_show_total_duration : ℝ :=
  short_show_episodes * short_show_duration_per_episode

def long_show_total_duration : ℝ :=
  long_show_episodes * long_show_duration_per_episode

-- Define the total TV hours watched
def total_tv_hours_watched : ℝ :=
  short_show_total_duration + long_show_total_duration

-- Write the theorem statement
theorem tim_watched_total_hours_tv : total_tv_hours_watched = 24 := 
by
  -- proof goes here
  sorry

end tim_watched_total_hours_tv_l223_223655


namespace sin_cos_sum_l223_223504

theorem sin_cos_sum (α : ℝ) (h : ∃ (c : ℝ), Real.sin α = -1 / c ∧ Real.cos α = 2 / c ∧ c = Real.sqrt 5) :
  Real.sin α + Real.cos α = Real.sqrt 5 / 5 :=
by sorry

end sin_cos_sum_l223_223504


namespace farmer_steven_total_days_l223_223466

theorem farmer_steven_total_days 
(plow_acres_per_day : ℕ)
(mow_acres_per_day : ℕ)
(farmland_acres : ℕ)
(grassland_acres : ℕ)
(h_plow : plow_acres_per_day = 10)
(h_mow : mow_acres_per_day = 12)
(h_farmland : farmland_acres = 55)
(h_grassland : grassland_acres = 30) :
((farmland_acres / plow_acres_per_day) + (grassland_acres / mow_acres_per_day) = 8) := by
  sorry

end farmer_steven_total_days_l223_223466


namespace geometric_series_second_term_l223_223171

theorem geometric_series_second_term (a : ℝ) (r : ℝ) (sum : ℝ) 
  (h1 : r = 1/4) 
  (h2 : sum = 40) 
  (sum_formula : sum = a / (1 - r)) : a * r = 7.5 :=
by {
  -- Proof to be filled in later
  sorry
}

end geometric_series_second_term_l223_223171


namespace maryville_population_increase_l223_223421

def average_people_added_per_year (P2000 P2005 : ℕ) (period : ℕ) : ℕ :=
  (P2005 - P2000) / period
  
theorem maryville_population_increase :
  let P2000 := 450000
  let P2005 := 467000
  let period := 5
  average_people_added_per_year P2000 P2005 period = 3400 :=
by
  sorry

end maryville_population_increase_l223_223421


namespace fiona_prob_reaches_12_l223_223652

/-- Lily pads are numbered from 0 to 15 -/
def is_valid_pad (n : ℕ) : Prop := 0 ≤ n ∧ n ≤ 15

/-- Predators are on lily pads 4 and 7 -/
def predator (n : ℕ) : Prop := n = 4 ∨ n = 7

/-- Fiona the frog's probability to hop to the next pad -/
def hop : ℚ := 1 / 2

/-- Fiona the frog's probability to jump 2 pads -/
def jump_two : ℚ := 1 / 2

/-- Probability that Fiona reaches pad 12 without landing on pads 4 or 7 is 1/32 -/
theorem fiona_prob_reaches_12 :
  ∀ p : ℕ, 
    (is_valid_pad p ∧ ¬ predator p ∧ (p = 12) ∧ 
    ((∀ k : ℕ, is_valid_pad k → ¬ predator k → k ≤ 3 → (hop ^ k) = 1 / 2) ∧
    hop * hop = 1 / 4 ∧ hop * jump_two = 1 / 8 ∧
    (jump_two * (hop * hop + jump_two)) = 1 / 4 → hop * 1 / 4 = 1 / 32)) := 
by intros; sorry

end fiona_prob_reaches_12_l223_223652


namespace each_boy_earns_14_dollars_l223_223835

theorem each_boy_earns_14_dollars :
  let Victor_shrimp := 26 in
  let Austin_shrimp := Victor_shrimp - 8 in
  let total_Victor_Austin_shrimp := Victor_shrimp + Austin_shrimp in
  let Brian_shrimp := total_Victor_Austin_shrimp / 2 in
  let total_shrimp := Victor_shrimp + Austin_shrimp + Brian_shrimp in
  let total_money := (total_shrimp / 11) * 7 in
  let money_per_boy := total_money / 3 in
  money_per_boy = 14 :=
by
  sorry

end each_boy_earns_14_dollars_l223_223835


namespace intersection_A_B_l223_223328

def A : Set ℝ := { x | Real.sqrt x ≤ 3 }
def B : Set ℝ := { x | x^2 ≤ 9 }

theorem intersection_A_B : A ∩ B = { x | 0 ≤ x ∧ x ≤ 3 } :=
by
  sorry

end intersection_A_B_l223_223328


namespace minimize_f_a_n_distance_l223_223454

noncomputable def f (x : ℝ) : ℝ :=
  2^x + Real.log x

noncomputable def a (n : ℕ) : ℝ :=
  0.1 * n

theorem minimize_f_a_n_distance :
  ∃ n : ℕ, n = 110 ∧ ∀ m : ℕ, (m > 0) -> |f (a 110) - 2012| ≤ |f (a m) - 2012| :=
by
  sorry

end minimize_f_a_n_distance_l223_223454


namespace find_natural_numbers_l223_223729

def divisors (n m : ℕ) : Prop := m ∣ n
def is_prime (n : ℕ) : Prop := nat.prime n

theorem find_natural_numbers (a b : ℕ) :
  (a = 3 ∧ b = 1) ∨ (a = 7 ∧ b = 2) ∨ (a = 11 ∧ b = 3) ↔
  ¬divisors (a - b) 3 ∧ is_prime (a + 2 * b) ∧ a = 4 * b - 1 ∧ divisors (a + 7) b := by
  sorry

end find_natural_numbers_l223_223729


namespace units_digit_35_pow_35_mul_17_pow_17_l223_223563

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_35_pow_35_mul_17_pow_17:
  units_digit (35 ^ (35 * 17 ^ 17)) = 5 := 
by {
  -- Here we're skipping the proof.
  sorry
}

end units_digit_35_pow_35_mul_17_pow_17_l223_223563


namespace range_of_a_l223_223901

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 2 * x + 1 > 0) ↔ (0 ≤ a ∧ a ≤ 1) := 
sorry

end range_of_a_l223_223901


namespace problem_solution_l223_223925

universe u

variable (U : Set Nat) (M : Set Nat)
variable (complement_U_M : Set Nat)

axiom U_def : U = {1, 2, 3, 4, 5}
axiom complement_U_M_def : complement_U_M = {1, 3}
axiom M_def : M = U \ complement_U_M

theorem problem_solution : 2 ∈ M := by
  sorry

end problem_solution_l223_223925


namespace length_GP_l223_223958

open Triangle

-- Definitions for triangle, centroids, and altitudes
noncomputable def triangleABC : Triangle :=
{ a := (11 : ℝ), b := (13 : ℝ), c := (20 : ℝ), A := (0, 0), B := (11, 0), C := (5, nonneg_of_real 13), }

noncomputable def G : Point := centroid triangleABC

noncomputable def P : Point := foot_of_altitude G triangleABC.bc

-- Statement to prove the length of GP is 11/5
theorem length_GP : length (line_segment G P) = 11 / 5 := by
  sorry

end length_GP_l223_223958


namespace books_bound_l223_223765

theorem books_bound (x : ℕ) (w c : ℕ) (h₀ : w = 92) (h₁ : c = 135) 
(h₂ : 92 - x = 2 * (135 - x)) :
x = 178 :=
by
  sorry

end books_bound_l223_223765


namespace number_of_packages_l223_223259

theorem number_of_packages (total_tshirts : ℕ) (tshirts_per_package : ℕ) 
  (h1 : total_tshirts = 56) (h2 : tshirts_per_package = 2) : 
  (total_tshirts / tshirts_per_package) = 28 := 
  by
    sorry

end number_of_packages_l223_223259


namespace negative_solutions_iff_l223_223942

theorem negative_solutions_iff (m x y : ℝ) (h1 : x - y = 2 * m + 7) (h2 : x + y = 4 * m - 3) :
  (x < 0 ∧ y < 0) ↔ m < -2 / 3 :=
by
  sorry

end negative_solutions_iff_l223_223942


namespace joe_eats_at_least_two_kinds_l223_223769

noncomputable def probability_at_least_two_kinds_of_fruit : ℚ :=
  1 - (3 * (1 / 3)^4)

theorem joe_eats_at_least_two_kinds :
  probability_at_least_two_kinds_of_fruit = 26 / 27 := 
by
  sorry

end joe_eats_at_least_two_kinds_l223_223769


namespace maximum_area_of_region_l223_223355

/-- Given four circles with radii 2, 4, 6, and 8, tangent to the same point B 
on a line ℓ, with the two largest circles (radii 6 and 8) on the same side of ℓ,
prove that the maximum possible area of the region consisting of points lying
inside exactly one of these circles is 120π. -/
theorem maximum_area_of_region 
  (radius1 : ℝ) (radius2 : ℝ) (radius3 : ℝ) (radius4 : ℝ)
  (line : ℝ → Prop) (B : ℝ)
  (tangent1 : ∀ x, line x → dist x B = radius1) 
  (tangent2 : ∀ x, line x → dist x B = radius2)
  (tangent3 : ∀ x, line x → dist x B = radius3)
  (tangent4 : ∀ x, line x → dist x B = radius4)
  (side1 : ℕ)
  (side2 : ℕ)
  (equal_side : side1 = side2)
  (r1 : ℝ := 2) 
  (r2 : ℝ := 4)
  (r3 : ℝ := 6) 
  (r4 : ℝ := 8) :
  (π * (radius1 * radius1) + π * (radius2 * radius2) + π * (radius3 * radius3) + π * (radius4 * radius4)) = 120 * π := 
sorry

end maximum_area_of_region_l223_223355


namespace correct_statement_l223_223905

open Set

variable (U : Set ℕ) (M : Set ℕ)
variables (hU : U = {1, 2, 3, 4, 5}) (hM : U \ M = {1, 3})

theorem correct_statement : 2 ∈ M :=
by
  sorry

end correct_statement_l223_223905


namespace total_shells_is_correct_l223_223774

def morning_shells : Nat := 292
def afternoon_shells : Nat := 324
def total_shells : Nat := morning_shells + afternoon_shells

theorem total_shells_is_correct : total_shells = 616 :=
by
  sorry

end total_shells_is_correct_l223_223774


namespace at_least_one_non_negative_l223_223081

variable (x : ℝ)
def a : ℝ := x^2 - 1
def b : ℝ := 2*x + 2

theorem at_least_one_non_negative (x : ℝ) : ¬ (a x < 0 ∧ b x < 0) :=
by
  sorry

end at_least_one_non_negative_l223_223081


namespace distribution_scheme_count_l223_223708

noncomputable def NumberOfDistributionSchemes : Nat :=
  let plumbers := 5
  let residences := 4
  Nat.choose plumbers (residences - 1) * Nat.factorial residences

theorem distribution_scheme_count :
  NumberOfDistributionSchemes = 240 :=
by
  sorry

end distribution_scheme_count_l223_223708


namespace intersection_polar_sum_l223_223208

variables {α ρ θ : ℝ}

def curve_C1 (α : ℝ) : Prop :=
  (λ (x y : ℝ), x = 2 + Real.cos α ∧ y = 2 + Real.sin α)

def line_C2 (x y : ℝ) : Prop :=
  y = Real.sqrt 3 * x

noncomputable def polar_coordinates_oa_ob (A B : ℝ × ℝ) (O : ℝ × ℝ) :=
  let OA := Real.sqrt ((A.1 - O.1)^2 + (A.2 - O.2)^2) in
  let OB := Real.sqrt ((B.1 - O.1)^2 + (B.2 - O.2)^2) in
  1 / |OA| + 1 / |OB|

theorem intersection_polar_sum (OA OB : ℝ) (O : ℝ × ℝ) :
  (∀ α, ∃ x y, curve_C1 α x y) →
  (∀ x y, line_C2 x y) →
  polar_coordinates_oa_ob (2 + Real.cos OA, 2 + Real.sin OA) (2 + Real.cos OB, 2 + Real.sin OB) O = (2 + 2 * Real.sqrt 3) / 7 :=
sorry

end intersection_polar_sum_l223_223208


namespace interest_earned_l223_223493

theorem interest_earned :
  let P : ℝ := 1500
  let r : ℝ := 0.02
  let n : ℕ := 3
  let A : ℝ := P * (1 + r) ^ n
  let interest : ℝ := A - P
  interest = 92 := 
by
  sorry

end interest_earned_l223_223493


namespace six_divides_p_plus_one_l223_223488

theorem six_divides_p_plus_one 
  (p : ℕ) 
  (prime_p : Nat.Prime p) 
  (gt_three_p : p > 3) 
  (prime_p_plus_two : Nat.Prime (p + 2)) 
  (gt_three_p_plus_two : p + 2 > 3) : 
  6 ∣ (p + 1) := 
sorry

end six_divides_p_plus_one_l223_223488


namespace three_digit_integer_equal_sum_factorials_l223_223517

open Nat

theorem three_digit_integer_equal_sum_factorials :
  ∃ (a b c : ℕ), a = 1 ∧ b = 4 ∧ c = 5 ∧ 100 * a + 10 * b + c = a.factorial + b.factorial + c.factorial :=
by
  use 1, 4, 5
  simp
  sorry

end three_digit_integer_equal_sum_factorials_l223_223517


namespace yoojung_namjoon_total_flowers_l223_223266

theorem yoojung_namjoon_total_flowers
  (yoojung_flowers : ℕ)
  (namjoon_flowers : ℕ)
  (yoojung_condition : yoojung_flowers = 4 * namjoon_flowers)
  (yoojung_count : yoojung_flowers = 32) :
  yoojung_flowers + namjoon_flowers = 40 :=
by
  sorry

end yoojung_namjoon_total_flowers_l223_223266


namespace average_people_added_each_year_l223_223422

-- a) Identifying questions and conditions
-- Question: What is the average number of people added each year?
-- Conditions: In 2000, about 450,000 people lived in Maryville. In 2005, about 467,000 people lived in Maryville.

-- c) Mathematically equivalent proof problem
-- Mathematically equivalent proof problem: Prove that the average number of people added each year is 3400 given the conditions.

-- d) Lean 4 statement
theorem average_people_added_each_year :
  let population_2000 := 450000
  let population_2005 := 467000
  let years_passed := 2005 - 2000
  let total_increase := population_2005 - population_2000
  total_increase / years_passed = 3400 := by
    sorry

end average_people_added_each_year_l223_223422


namespace permissible_range_n_l223_223580

theorem permissible_range_n (n x y m : ℝ) (hn : n ≤ x) (hxy : x < y) (hy : y ≤ n+1)
  (hm_in: x < m ∧ m < y) (habs_eq : |y| = |m| + |x|): 
  -1 < n ∧ n < 1 := sorry

end permissible_range_n_l223_223580


namespace circle_equation_polar_to_rectangular_line_parametric_to_standard_area_of_triangle_ABC_l223_223957

theorem circle_equation_polar_to_rectangular :
  ∀ (x y : ℝ), 
  (x^2 + y^2)^2 = 16 * (x^2 + y^2) - 32 * x * y → 
  (x - 2)^2 + (y + 2)^2 = 8 :=
by sorry

theorem line_parametric_to_standard :
  ∀ (t : ℝ) (x y : ℝ), 
  (x = t + 1) ∧ (y = t - 1) → 
  x - y = 2 :=
by sorry

theorem area_of_triangle_ABC :
  ∀ (A B : (ℝ × ℝ)) (x y : ℝ),
  (A = (2 + sqrt(2), -2 - sqrt(2))) ∧ (B = (2 - sqrt(2), -2 + sqrt(2))) →
  ((x - 2)^2 + (y + 2)^2 = 8 ∧ x - y = 2) →
  let d := sqrt(2) in
  let h := sqrt(8) - d in
  let AB := 2 * sqrt(6) in
  let S := (1 / 2) * AB * h in
  S = 2 * sqrt(3) :=
by sorry

end circle_equation_polar_to_rectangular_line_parametric_to_standard_area_of_triangle_ABC_l223_223957


namespace geometric_sequence_a4_l223_223645

theorem geometric_sequence_a4 (a : ℕ → ℝ) (q : ℝ) 
  (h1 : ∀ n, a (n + 1) = a n * q)
  (h2 : a 2 = 4)
  (h3 : a 6 = 16) : 
  a 4 = 8 :=
by
  -- Proof goes here
  sorry

end geometric_sequence_a4_l223_223645


namespace unique_y_for_star_l223_223557

def star (x y : ℝ) : ℝ := 5 * x - 2 * y + 2 * x * y - 3

theorem unique_y_for_star : (∀ y : ℝ, star 4 y = 17 → y = 0) ∧ (∃! y : ℝ, star 4 y = 17) := by
  sorry

end unique_y_for_star_l223_223557


namespace harry_book_pages_correct_l223_223783

-- Define the total pages in Selena's book.
def selena_book_pages : ℕ := 400

-- Define Harry's book pages as 20 fewer than half of Selena's book pages.
def harry_book_pages : ℕ := (selena_book_pages / 2) - 20

-- The theorem to prove the number of pages in Harry's book.
theorem harry_book_pages_correct : harry_book_pages = 180 := by
  sorry

end harry_book_pages_correct_l223_223783


namespace Sandy_goal_water_l223_223489

-- Definitions based on the conditions in problem a)
def milliliters_per_interval := 500
def time_per_interval := 2
def total_time := 12
def milliliters_to_liters := 1000

-- The goal statement that proves the question == answer given conditions.
theorem Sandy_goal_water : (milliliters_per_interval * (total_time / time_per_interval)) / milliliters_to_liters = 3 := by
  sorry

end Sandy_goal_water_l223_223489


namespace total_carpet_area_correct_l223_223600

-- Define dimensions of the rooms
def room1_width : ℝ := 12
def room1_length : ℝ := 15
def room2_width : ℝ := 7
def room2_length : ℝ := 9
def room3_width : ℝ := 10
def room3_length : ℝ := 11

-- Define the areas of the rooms
def room1_area : ℝ := room1_width * room1_length
def room2_area : ℝ := room2_width * room2_length
def room3_area : ℝ := room3_width * room3_length

-- Total carpet area
def total_carpet_area : ℝ := room1_area + room2_area + room3_area

-- The theorem to prove
theorem total_carpet_area_correct :
  total_carpet_area = 353 :=
sorry

end total_carpet_area_correct_l223_223600


namespace function_properties_l223_223260

noncomputable def f (x : ℝ) : ℝ := x^2

theorem function_properties :
  (∀ x1 x2 : ℝ, f (x1 * x2) = f x1 * f x2) ∧
  (∀ x : ℝ, 0 < x → deriv f x > 0) ∧
  (∀ x : ℝ, deriv f (-x) = -deriv f x) :=
by
  sorry

end function_properties_l223_223260


namespace Caitlin_Sara_weight_l223_223291

variable (A C S : ℕ)

theorem Caitlin_Sara_weight 
  (h1 : A + C = 95) 
  (h2 : A = S + 8) : 
  C + S = 87 := by
  sorry

end Caitlin_Sara_weight_l223_223291


namespace min_value_of_quadratic_l223_223005

open Real

theorem min_value_of_quadratic 
  (x y z : ℝ) 
  (h : 3 * x + 2 * y + z = 1) : 
  x^2 + 2 * y^2 + 3 * z^2 ≥ 3 / 34 := 
sorry

end min_value_of_quadratic_l223_223005


namespace necessary_but_not_sufficient_l223_223018

-- Define \(\frac{1}{x} < 2\) and \(x > \frac{1}{2}\)
def condition1 (x : ℝ) : Prop := 1 / x < 2
def condition2 (x : ℝ) : Prop := x > 1 / 2

-- Theorem stating that condition1 is necessary but not sufficient for condition2
theorem necessary_but_not_sufficient (x : ℝ) : condition1 x → condition2 x ↔ true :=
sorry

end necessary_but_not_sufficient_l223_223018


namespace proof_2_in_M_l223_223938

def U : Set ℕ := {1, 2, 3, 4, 5}

def M : Set ℕ := { x | x ∈ U ∧ x ≠ 1 ∧ x ≠ 3 }

theorem proof_2_in_M : 2 ∈ M :=
by
  sorry

end proof_2_in_M_l223_223938


namespace geometric_sequence_property_l223_223468

theorem geometric_sequence_property 
  (a : ℕ → ℝ) 
  (h_geom: ∀ n, a (n + 1) = a n * r) 
  (h_pos: ∀ n, a n > 0)
  (h_root1: a 3 * a 15 = 8)
  (h_root2: a 3 + a 15 = 6) :
  a 1 * a 17 / a 9 = 2 * Real.sqrt 2 :=
by
  sorry

end geometric_sequence_property_l223_223468


namespace calculate_F_2_f_3_l223_223604

def f (a : ℕ) : ℕ := a ^ 2 - 3 * a + 2

def F (a b : ℕ) : ℕ := b ^ 2 + a + 1

theorem calculate_F_2_f_3 : F 2 (f 3) = 7 :=
by
  show F 2 (f 3) = 7
  sorry

end calculate_F_2_f_3_l223_223604


namespace decreasing_geometric_sums_implications_l223_223210

variable (X : Type)
variable (a1 q : ℝ)
variable (S : ℕ → ℝ)

def is_geometric_sequence (a : ℕ → ℝ) :=
∀ n : ℕ, a (n + 1) = a1 * q^n

def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) :=
S 0 = a 0 ∧ ∀ n : ℕ, S (n + 1) = S n + a (n + 1)

def is_decreasing_sequence (S : ℕ → ℝ) :=
∀ n : ℕ, S (n + 1) < S n

theorem decreasing_geometric_sums_implications (a1 q : ℝ) (S : ℕ → ℝ) :
  (∀ n : ℕ, S (n + 1) < S n) → a1 < 0 ∧ q > 0 := 
by 
  sorry

end decreasing_geometric_sums_implications_l223_223210


namespace points_on_opposite_sides_l223_223578

theorem points_on_opposite_sides (a : ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ) 
    (hA : A = (a, 1)) 
    (hB : B = (2, a)) 
    (opposite_sides : A.1 < 0 ∧ B.1 > 0 ∨ A.1 > 0 ∧ B.1 < 0) 
    : a < 0 := 
  sorry

end points_on_opposite_sides_l223_223578


namespace coefficient_of_8th_term_l223_223791

-- Define the general term of the binomial expansion
def binomial_expansion_term (n r : ℕ) (a b : ℕ) : ℕ := 
  Nat.choose n r * a^(n - r) * b^r

-- Define the specific scenario given in the problem
def specific_binomial_expansion_term : ℕ := 
  binomial_expansion_term 8 7 2 1  -- a = 2, b = x (consider b as 1 for coefficient calculation)

-- Problem statement to prove the coefficient of the 8th term is 16
theorem coefficient_of_8th_term : specific_binomial_expansion_term = 16 := by
  sorry

end coefficient_of_8th_term_l223_223791


namespace determine_g1_l223_223367

variable (g : ℝ → ℝ)
variable (h : ∀ x y : ℝ, g (g x + y) = g (x + y) + x * g y - x^2 * y - x^3 + 1)

theorem determine_g1 : g 1 = 2 := sorry

end determine_g1_l223_223367


namespace sum_of_roots_l223_223516

theorem sum_of_roots (x : ℝ) : (x - 4)^2 = 16 → x = 8 ∨ x = 0 := by
  intro h
  have h1 : x - 4 = 4 ∨ x - 4 = -4 := by
    sorry
  cases h1
  case inl h2 =>
    rw [h2] at h
    exact Or.inl (by linarith)
  case inr h2 =>
    rw [h2] at h
    exact Or.inr (by linarith)

end sum_of_roots_l223_223516


namespace exponentiation_example_l223_223114

theorem exponentiation_example : (3^2)^4 = 6561 := by
  sorry

end exponentiation_example_l223_223114


namespace eccentricity_of_ellipse_l223_223465

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem eccentricity_of_ellipse :
  let P := (2, 3)
  let F1 := (-2, 0)
  let F2 := (2, 0)
  let d1 := distance P F1
  let d2 := distance P F2
  let a := (d1 + d2) / 2
  let c := distance F1 F2 / 2
  let e := c / a
  e = 1 / 2 := 
by 
  sorry

end eccentricity_of_ellipse_l223_223465


namespace geom_seq_result_l223_223583

variable (a : ℕ → ℚ) (S : ℕ → ℚ)
variable (n : ℕ)

-- Conditions
axiom h1 : a 1 + a 3 = 5 / 2
axiom h2 : a 2 + a 4 = 5 / 4

-- General properties
axiom geom_seq_common_ratio : ∃ q : ℚ, ∀ n, a (n + 1) = a n * q

-- Sum of the first n terms of the geometric sequence
axiom S_def : S n = (2 * (1 - (1 / 2)^n)) / (1 - 1 / 2)

-- General term of the geometric sequence
axiom a_n_def : a n = 2 * (1 / 2)^(n - 1)

-- Result to be proved
theorem geom_seq_result : S n / a n = 2^n - 1 := 
  by sorry

end geom_seq_result_l223_223583


namespace bus_cost_proof_l223_223864

-- Define conditions
def train_cost (bus_cost : ℚ) : ℚ := bus_cost + 6.85
def discount_rate : ℚ := 0.15
def service_fee : ℚ := 1.25
def combined_cost : ℚ := 10.50

-- Formula for the total cost after discount
def discounted_train_cost (bus_cost : ℚ) : ℚ := (train_cost bus_cost) * (1 - discount_rate)
def total_cost (bus_cost : ℚ) : ℚ := discounted_train_cost bus_cost + bus_cost + service_fee

-- Lean 4 statement asserting the cost of the bus ride before service fee
theorem bus_cost_proof : ∃ (B : ℚ), total_cost B = combined_cost ∧ B = 1.85 :=
sorry

end bus_cost_proof_l223_223864


namespace train_speed_l223_223544

theorem train_speed
    (train_length : ℕ := 800)
    (tunnel_length : ℕ := 500)
    (time_minutes : ℕ := 1)
    : (train_length + tunnel_length) * (60 / time_minutes) / 1000 = 78 := by
  sorry

end train_speed_l223_223544


namespace sequence_inequality_l223_223223

theorem sequence_inequality
  (a : ℕ → ℝ)
  (h_cond : ∀ k m : ℕ, |a (k + m) - a k - a m| ≤ 1) :
  ∀ p q : ℕ, |a p / p - a q / q| < 1 / p + 1 / q :=
by
  intros p q
  sorry

end sequence_inequality_l223_223223


namespace original_price_of_cycle_l223_223677

theorem original_price_of_cycle (SP : ℝ) (gain_percent : ℝ) (P : ℝ)
  (h_SP : SP = 1080)
  (h_gain_percent: gain_percent = 60)
  (h_relation : SP = 1.6 * P)
  : P = 675 :=
by {
  sorry
}

end original_price_of_cycle_l223_223677


namespace solve_for_other_diagonal_l223_223068

noncomputable def length_of_other_diagonal
  (area : ℝ) (d2 : ℝ) : ℝ :=
  (2 * area) / d2

theorem solve_for_other_diagonal 
  (h_area : ℝ) (h_d2 : ℝ) (h_condition : h_area = 75 ∧ h_d2 = 15) :
  length_of_other_diagonal h_area h_d2 = 10 :=
by
  -- using h_condition, prove the required theorem
  sorry

end solve_for_other_diagonal_l223_223068


namespace sum_of_squares_l223_223622

theorem sum_of_squares (k₁ k₂ k₃ : ℝ)
  (h_sum : k₁ + k₂ + k₃ = 1) : k₁^2 + k₂^2 + k₃^2 ≥ 1/3 :=
by sorry

end sum_of_squares_l223_223622


namespace probability_all_white_l223_223673

noncomputable def balls_in_box :=
  {white := 6, black := 7, red := 3}

noncomputable def total_balls :=
  balls_in_box.white + balls_in_box.black + balls_in_box.red

noncomputable def drawn_balls :=
  8

theorem probability_all_white (w : nat := balls_in_box.white) (t : nat := total_balls) (d : nat := drawn_balls) :
  let prob := if d > w then 0 else (nat.choose w d) / (nat.choose t d) in
  prob = 0 :=
by
  have out_of_bounds : drawn_balls > balls_in_box.white := by sorry
  rw [out_of_bounds]
  exact rfl

end probability_all_white_l223_223673


namespace sand_height_when_inverted_l223_223281

noncomputable def sand_height_inverted (r h hf: ℝ) (h_cylinder: ℝ) : ℝ :=
  let volume_cone := (1 / 3) * real.pi * r * r * h,
      volume_cylinder_part := real.pi * r * r * h_cylinder,
      total_volume := volume_cone + volume_cylinder_part,
      volume_new_cylinder := total_volume - volume_cone,
      new_height := volume_new_cylinder / (real.pi * r * r)
  in (h + new_height)

theorem sand_height_when_inverted (r h hf : ℝ) (h_cylinder: ℝ) : 
  r = 12 → h = 20 → hf = 20 → h_cylinder = 5 → 
  sand_height_inverted r h hf h_cylinder = 25 :=
by { intros, unfold sand_height_inverted, sorry }

end sand_height_when_inverted_l223_223281


namespace total_hours_proof_l223_223656

-- Conditions
def half_hour_show_episodes : ℕ := 24
def one_hour_show_episodes : ℕ := 12
def half_hour_per_episode : ℝ := 0.5
def one_hour_per_episode : ℝ := 1.0

-- Define the total hours Tim watched
def total_hours_watched : ℝ :=
  half_hour_show_episodes * half_hour_per_episode + one_hour_show_episodes * one_hour_per_episode

-- Prove that the total hours watched is 24
theorem total_hours_proof : total_hours_watched = 24 := by
  sorry

end total_hours_proof_l223_223656


namespace number_of_lemons_l223_223638

theorem number_of_lemons
  (total_fruits : ℕ)
  (mangoes : ℕ)
  (pears : ℕ)
  (pawpaws : ℕ)
  (kiwis : ℕ)
  (lemons : ℕ)
  (h_total : total_fruits = 58)
  (h_mangoes : mangoes = 18)
  (h_pears : pears = 10)
  (h_pawpaws : pawpaws = 12)
  (h_kiwis_lemons_equal : kiwis = lemons) :
  lemons = 9 :=
by
  sorry

end number_of_lemons_l223_223638


namespace exponentiation_rule_example_l223_223095

theorem exponentiation_rule_example : (3^2)^4 = 6561 := 
by {
  -- This proof is based on the power rule (ab)^c = a^(b*c)
  have step1 : (3^2)^4 = 3^(2 * 4), 
  {
    rw pow_mul,
  },
  -- Simplification of 3^(2*4) to 3^8
  have step2 : 3^(2 * 4) = 3^8,
  {
    refl,
  },
  -- Actual computation for 3^8 = 6561
  have step3 : 3^8 = 6561,
  {
    norm_num,
  },
  -- Combining all steps
  rw [step1, step2, step3],
  exact rfl,
}

end exponentiation_rule_example_l223_223095


namespace find_nabla_l223_223339

theorem find_nabla : ∀ (nabla : ℤ), 5 * (-4) = nabla + 2 → nabla = -22 :=
by
  intros nabla h
  sorry

end find_nabla_l223_223339


namespace power_of_powers_l223_223088

theorem power_of_powers :
  (3^2)^4 = 6561 := 
by 
  sorry

end power_of_powers_l223_223088


namespace contrapositive_proposition_l223_223232

theorem contrapositive_proposition (x y : ℝ) :
  (¬ (x = 0 ∧ y = 0)) → (x^2 + y^2 ≠ 0) :=
sorry

end contrapositive_proposition_l223_223232


namespace circle_and_tangent_lines_pass_through_point_l223_223323

def circle_passing_through (r : ℝ) (Mx My : ℝ) : Prop :=
  Mx^2 + My^2 = r^2

def tangent_lines (r : ℝ) (Px Py : ℝ) (k1 k2 : ℝ) : Prop :=
  (Py - 2) * sqrt ((k1)^2 + 1) = 2 ∧ (Px - 3) * k1 = Py - 2 ∧
  (k2 = 0 ∨ k2 = 12/5)

theorem circle_and_tangent_lines_pass_through_point :
  (circle_passing_through 2 0 2) →
  (∃ t1 t2 : ℝ, tangent_lines 2 3 2 t1 t2) :=
by sorry

end circle_and_tangent_lines_pass_through_point_l223_223323


namespace marbles_in_jar_l223_223536

theorem marbles_in_jar (x : ℕ)
  (h1 : \frac{1}{2} * x + \frac{1}{4} * x + 27 + 14 = x) : x = 164 := sorry

end marbles_in_jar_l223_223536


namespace factor_difference_of_squares_l223_223182

theorem factor_difference_of_squares (t : ℝ) : t^2 - 144 = (t - 12) * (t + 12) :=
by
  sorry

end factor_difference_of_squares_l223_223182


namespace smallest_positive_multiple_of_37_l223_223663

theorem smallest_positive_multiple_of_37 :
  ∃ n, n > 0 ∧ (∃ a, n = 37 * a) ∧ (∃ k, n = 76 * k + 7) ∧ n = 2405 := 
by
  sorry

end smallest_positive_multiple_of_37_l223_223663


namespace correct_statement_l223_223909

universe u
variable (α : Type u)

noncomputable def U : Set ℕ := {1, 2, 3, 4, 5}

noncomputable def complement_U_M : Set ℕ := {1, 3}

noncomputable def M : Set ℕ := U α \ complement_U_M

theorem correct_statement : 2 ∈ M :=
by {
  sorry
}

end correct_statement_l223_223909


namespace x_in_terms_of_y_y_in_terms_of_x_l223_223209

-- Define the main equation
variable (x y : ℝ)

-- First part: Expressing x in terms of y given the condition
theorem x_in_terms_of_y (h : x + 3 * y = 3) : x = 3 - 3 * y :=
by
  sorry

-- Second part: Expressing y in terms of x given the condition
theorem y_in_terms_of_x (h : x + 3 * y = 3) : y = (3 - x) / 3 :=
by
  sorry

end x_in_terms_of_y_y_in_terms_of_x_l223_223209


namespace y_coord_vertex_C_l223_223627

/-- The coordinates of vertices A, B, and D are given as A(0,0), B(0,1), and D(3,1).
 Vertex C is directly above vertex B. The quadrilateral ABCD has a vertical line of symmetry 
 and the area of quadrilateral ABCD is 18 square units.
 Prove that the y-coordinate of vertex C is 11. -/
theorem y_coord_vertex_C (h : ℝ) 
  (A : ℝ × ℝ := (0, 0)) 
  (B : ℝ × ℝ := (0, 1)) 
  (D : ℝ × ℝ := (3, 1)) 
  (C : ℝ × ℝ := (0, h)) 
  (symmetry : C.fst = B.fst) 
  (area : 18 = 3 * 1 + (1 / 2) * 3 * (h - 1)) :
  h = 11 := 
by
  sorry

end y_coord_vertex_C_l223_223627


namespace division_631938_by_625_l223_223336

theorem division_631938_by_625 :
  (631938 : ℚ) / 625 = 1011.1008 :=
by
  -- Add a placeholder proof. We do not provide the solution steps.
  sorry

end division_631938_by_625_l223_223336


namespace lloyd_normal_hours_l223_223371

-- Definitions based on the conditions
def regular_rate : ℝ := 3.50
def overtime_rate : ℝ := 1.5 * regular_rate
def total_hours_worked : ℝ := 10.5
def total_earnings : ℝ := 42
def normal_hours_worked (h : ℝ) : Prop := 
  h * regular_rate + (total_hours_worked - h) * overtime_rate = total_earnings

-- The theorem to prove
theorem lloyd_normal_hours : ∃ h : ℝ, normal_hours_worked h ∧ h = 7.5 := sorry

end lloyd_normal_hours_l223_223371


namespace min_m_value_l223_223593

noncomputable def f (x a : ℝ) : ℝ := 2 ^ (abs (x - a))

theorem min_m_value :
  ∀ a, (∀ x, f (1 + x) a = f (1 - x) a) →
  ∃ m : ℝ, (∀ x : ℝ, x ≥ m → ∀ y : ℝ, y ≥ x → f y a ≥ f x a) ∧ m = 1 :=
by
  intros a h
  sorry

end min_m_value_l223_223593


namespace power_calc_l223_223122

theorem power_calc : (3^2)^4 = 6561 := 
by
  sorry

end power_calc_l223_223122


namespace square_area_eq_l223_223287

-- Define the side length of the square and the diagonal relationship
variables (s : ℝ) (h : s * Real.sqrt 2 = s + 1)

-- State the theorem to solve
theorem square_area_eq :
  s * Real.sqrt 2 = s + 1 → (s ^ 2 = 3 + 2 * Real.sqrt 2) :=
by
  -- Assume the given condition
  intro h
  -- Insert proof steps here, analysis follows the provided solution steps.
  sorry

end square_area_eq_l223_223287


namespace exponentiation_identity_l223_223113

theorem exponentiation_identity (a m n : ℕ) : (a^m)^n = a^(m * n) :=
by sorry

example : (3^2)^4 = 6561 :=
by
  have h := exponentiation_identity 3 2 4
  rw [←h]
  norm_num

end exponentiation_identity_l223_223113


namespace nitin_ranks_from_last_l223_223776

def total_students : ℕ := 75

def math_rank_start : ℕ := 24
def english_rank_start : ℕ := 18

def rank_from_last (total : ℕ) (rank_start : ℕ) : ℕ :=
  total - rank_start + 1

theorem nitin_ranks_from_last :
  rank_from_last total_students math_rank_start = 52 ∧
  rank_from_last total_students english_rank_start = 58 :=
by
  sorry

end nitin_ranks_from_last_l223_223776


namespace volume_maximized_at_r_5_h_8_l223_223545

noncomputable def V (r : ℝ) : ℝ := (Real.pi / 5) * (300 * r - 4 * r^3)

/-- (1) Given that the total construction cost is 12000π yuan, 
express the volume V as a function of the radius r, and determine its domain. -/
def volume_function (r : ℝ) (h : ℝ) (cost : ℝ) : Prop :=
  cost = 12000 * Real.pi ∧
  h = 1 / (5 * r) * (300 - 4 * r^2) ∧
  V r = Real.pi * r^2 * h ∧
  0 < r ∧ r < 5 * Real.sqrt 3

/-- (2) Prove V(r) is maximized when r = 5 and h = 8 -/
theorem volume_maximized_at_r_5_h_8 :
  ∀ (r : ℝ) (h : ℝ) (cost : ℝ), volume_function r h cost → 
  ∃ (r_max : ℝ) (h_max : ℝ), r_max = 5 ∧ h_max = 8 ∧ ∀ x, 0 < x → x < 5 * Real.sqrt 3 → V x ≤ V r_max :=
by
  intros r h cost hvolfunc
  sorry

end volume_maximized_at_r_5_h_8_l223_223545


namespace find_C_l223_223851

theorem find_C (A B C : ℝ) (h1 : A + B + C = 500) (h2 : A + C = 200) (h3 : B + C = 340) : C = 40 :=
by sorry

end find_C_l223_223851


namespace friends_count_l223_223632

-- Define the conditions
def num_kids : ℕ := 2
def shonda_present : Prop := True  -- Shonda is present, we may just incorporate it as part of count for clarity
def num_adults : ℕ := 7
def num_baskets : ℕ := 15
def eggs_per_basket : ℕ := 12
def eggs_per_person : ℕ := 9

-- Define the total number of eggs
def total_eggs : ℕ := num_baskets * eggs_per_basket

-- Define the total number of people
def total_people : ℕ := total_eggs / eggs_per_person

-- Define the number of known people (Shonda, her kids, and the other adults)
def known_people : ℕ := num_kids + 1 + num_adults  -- 1 represents Shonda

-- Define the number of friends
def num_friends : ℕ := total_people - known_people

-- The theorem we need to prove
theorem friends_count : num_friends = 10 :=
by
  sorry

end friends_count_l223_223632


namespace mineral_samples_per_shelf_l223_223295

theorem mineral_samples_per_shelf (total_samples : ℕ) (num_shelves : ℕ) (h1 : total_samples = 455) (h2 : num_shelves = 7) :
  total_samples / num_shelves = 65 :=
by
  sorry

end mineral_samples_per_shelf_l223_223295


namespace expectation_zero_l223_223641

noncomputable def X_distribution : List (ℝ × ℝ) :=
  [(1, 0.1), (2, 0.3), (3, 0.2), (4, 0.3), (5, 0.1)]

theorem expectation_zero :
  let E_X := ∑ x in X_distribution, x.1 * x.2 in
  E_X = 3 → -- Given that E[X] = 3
  ∑ x in X_distribution, (x.1 - E_X) * x.2 = 0 :=
by
  intros E_X hE
  sorry

end expectation_zero_l223_223641


namespace find_a6_l223_223359

variable {a : ℕ → ℤ} -- Assume we have a sequence of integers
variable (d : ℤ) -- Common difference of the arithmetic sequence

-- Conditions
axiom h1 : a 3 = 7
axiom h2 : a 5 = a 2 + 6

-- Define arithmetic sequence property
def arithmetic_seq (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ (n : ℕ), a (n + 1) = a n + d

-- Theorem to prove
theorem find_a6 (h1 : a 3 = 7) (h2 : a 5 = a 2 + 6) (h3 : arithmetic_seq a d) : a 6 = 13 :=
by
  sorry

end find_a6_l223_223359


namespace water_tank_equilibrium_l223_223546

theorem water_tank_equilibrium :
  (1 / 15 : ℝ) + (1 / 10 : ℝ) - (1 / 6 : ℝ) = 0 :=
by
  sorry

end water_tank_equilibrium_l223_223546


namespace expression_equals_5_l223_223400

theorem expression_equals_5 : (3^2 - 2^2) = 5 := by
  calc
    (3^2 - 2^2) = 5 := by sorry

end expression_equals_5_l223_223400


namespace horse_revolutions_l223_223284

theorem horse_revolutions (r1 r2 : ℝ) (rev1 rev2 : ℕ) (h1 : r1 = 30) (h2 : rev1 = 25) (h3 : r2 = 10) : 
  rev2 = 75 :=
by 
  sorry

end horse_revolutions_l223_223284


namespace kenny_jumps_l223_223217

theorem kenny_jumps (M : ℕ) (h : 34 + M + 0 + 123 + 64 + 23 + 61 = 325) : M = 20 :=
by
  sorry

end kenny_jumps_l223_223217


namespace angle_terminal_side_eq_l223_223377

noncomputable def has_same_terminal_side (θ₁ θ₂ : ℝ) : Prop :=
  ∃ k : ℤ, θ₂ = θ₁ + 2 * k * Real.pi

theorem angle_terminal_side_eq (k : ℤ) :
  has_same_terminal_side (- (Real.pi / 3)) (5 * Real.pi / 3) :=
by
  use 1
  sorry

end angle_terminal_side_eq_l223_223377


namespace find_a_l223_223766

theorem find_a (P : ℝ) (hP : P ≠ 0) (S : ℕ → ℝ) (a_n : ℕ → ℝ)
  (hSn : ∀ n, S n = 3^n + a)
  (ha_n : ∀ n, a_n (n + 1) = P * a_n n)
  (hS1 : S 1 = a_n 1)
  (hS2 : S 2 = S 1 + a_n 2 - a_n 1)
  (hS3 : S 3 = S 2 + a_n 3 - a_n 2) :
  a = -1 := sorry

end find_a_l223_223766


namespace find_principal_sum_l223_223166

theorem find_principal_sum
  (R : ℝ) (P : ℝ)
  (H1 : 0 < R)
  (H2 : 8 * 10 * P / 100 = 150) :
  P = 187.50 :=
by
  sorry

end find_principal_sum_l223_223166


namespace range_of_a_minus_b_l223_223338

theorem range_of_a_minus_b (a b : ℝ) (ha : 1 < a ∧ a < 4) (hb : -2 < b ∧ b < 4) : 
  -3 < a - b ∧ a - b < 6 :=
by
  sorry

end range_of_a_minus_b_l223_223338


namespace polygon_with_20_diagonals_is_octagon_l223_223858

theorem polygon_with_20_diagonals_is_octagon :
  ∃ (n : ℕ), n ≥ 3 ∧ (n * (n - 3)) / 2 = 20 ∧ n = 8 :=
by
  sorry

end polygon_with_20_diagonals_is_octagon_l223_223858


namespace find_son_l223_223684

variable (SonAge ManAge : ℕ)

def age_relationship (SonAge ManAge : ℕ) : Prop :=
  ManAge = SonAge + 20 ∧ ManAge + 2 = 2 * (SonAge + 2)

theorem find_son's_age (S M : ℕ) (h : age_relationship S M) : S = 18 :=
by
  unfold age_relationship at h
  obtain ⟨h1, h2⟩ := h
  sorry

end find_son_l223_223684


namespace sylvia_time_to_complete_job_l223_223492

theorem sylvia_time_to_complete_job (S : ℝ) (h₁ : 18 ≠ 0) (h₂ : 30 ≠ 0)
  (together_rate : (1 / S) + (1 / 30) = 1 / 18) :
  S = 45 :=
by
  -- Proof will be provided here
  sorry

end sylvia_time_to_complete_job_l223_223492


namespace arithmetic_geometric_ratio_l223_223366

theorem arithmetic_geometric_ratio
  (a : ℕ → ℤ) 
  (d : ℤ)
  (h_seq : ∀ n, a (n+1) = a n + d)
  (h_geometric : (a 3)^2 = a 1 * a 9)
  (h_nonzero_d : d ≠ 0) :
  a 11 / a 5 = 5 / 2 :=
by sorry

end arithmetic_geometric_ratio_l223_223366


namespace largest_possible_percent_error_l223_223053

open Real

theorem largest_possible_percent_error :
  let length := 15
  let width := 10
  let length_error := 0.1
  let width_error := 0.1
  let min_length := length * (1 - length_error)
  let max_length := length * (1 + length_error)
  let min_width := width * (1 - width_error)
  let max_width := width * (1 + width_error)
  let actual_area := length * width
  let min_area := min_length * min_width
  let max_area := max_length * max_width
  let percent_error (computed_area : ℝ) : ℝ := ((computed_area - actual_area) / actual_area) * 100
  max (percent_error min_area) (percent_error max_area) = 21 :=
by
  sorry

end largest_possible_percent_error_l223_223053


namespace scott_earnings_l223_223630

theorem scott_earnings
  (price_smoothie : ℝ)
  (price_cake : ℝ)
  (cups_sold : ℝ)
  (cakes_sold : ℝ)
  (earnings_smoothies : ℝ := cups_sold * price_smoothie)
  (earnings_cakes : ℝ := cakes_sold * price_cake) :
  price_smoothie = 3 → price_cake = 2 → cups_sold = 40 → cakes_sold = 18 → 
  (earnings_smoothies + earnings_cakes) = 156 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end scott_earnings_l223_223630


namespace time_wandered_l223_223962

-- Definitions and Hypotheses
def distance : ℝ := 4
def speed : ℝ := 2

-- Proof statement
theorem time_wandered : distance / speed = 2 := by
  sorry

end time_wandered_l223_223962


namespace son_age_l223_223681

theorem son_age:
  ∃ S M : ℕ, 
  (M = S + 20) ∧ 
  (M + 2 = 2 * (S + 2)) ∧ 
  (S = 18) := 
by
  sorry

end son_age_l223_223681


namespace duration_trip_for_cyclist1_l223_223511

-- Definitions
variable (s : ℝ) -- the speed of Cyclist 1 without wind in km/h
variable (t : ℝ) -- the time in hours it takes for Cyclist 1 to travel from A to B
variable (wind_speed : ℝ := 3) -- wind modifies speed by 3 km/h
variable (total_time : ℝ := 4) -- total time after which cyclists meet

-- Conditions
axiom consistent_speed_aid : ∀ (s t : ℝ), t > 0 → (s + wind_speed) * t + (s - wind_speed) * (total_time - t) / 2 = s - wind_speed * total_time

-- Goal (equivalent proof problem)
theorem duration_trip_for_cyclist1 : t = 2 := by
  sorry

end duration_trip_for_cyclist1_l223_223511


namespace find_y_in_set_l223_223582

noncomputable def arithmetic_mean (s : List ℝ) : ℝ :=
  s.sum / s.length

theorem find_y_in_set :
  ∀ (y : ℝ), arithmetic_mean [8, 15, 20, 5, y] = 12 ↔ y = 12 :=
by
  intro y
  unfold arithmetic_mean
  simp [List.sum_cons, List.length_cons]
  sorry

end find_y_in_set_l223_223582


namespace min_value_of_a_l223_223748

/-- Given the inequality |x - 1| + |x + a| ≤ 8, prove that the minimum value of a is -9 -/

theorem min_value_of_a (a : ℝ) (h : ∀ x : ℝ, |x - 1| + |x + a| ≤ 8) : a = -9 :=
sorry

end min_value_of_a_l223_223748


namespace converse_xy_implies_x_is_true_l223_223143

/-- Prove that the converse of the proposition "If \(xy = 0\), then \(x = 0\)" is true. -/
theorem converse_xy_implies_x_is_true {x y : ℝ} (h : x = 0) : x * y = 0 :=
by sorry

end converse_xy_implies_x_is_true_l223_223143


namespace opposite_of_five_l223_223995

theorem opposite_of_five : -5 = -5 :=
by
sorry

end opposite_of_five_l223_223995


namespace drying_time_l223_223767

theorem drying_time
  (time_short : ℕ := 10) -- Time to dry a short-haired dog in minutes
  (time_full : ℕ := time_short * 2) -- Time to dry a full-haired dog in minutes, which is twice as long
  (num_short : ℕ := 6) -- Number of short-haired dogs
  (num_full : ℕ := 9) -- Number of full-haired dogs
  : (time_short * num_short + time_full * num_full) / 60 = 4 := 
by
  sorry

end drying_time_l223_223767


namespace average_waiting_time_l223_223417

-- Define the problem conditions
def light_period : ℕ := 3  -- Total cycle time in minutes
def green_time : ℕ := 1    -- Green light duration in minutes
def red_time : ℕ := 2      -- Red light duration in minutes

-- Define the probabilities of each light state
def P_G : ℚ := green_time / light_period
def P_R : ℚ := red_time / light_period

-- Define the expected waiting times given each state
def E_T_G : ℚ := 0
def E_T_R : ℚ := red_time / 2

-- Calculate the expected waiting time using the law of total expectation
def E_T : ℚ := E_T_G * P_G + E_T_R * P_R

-- Convert the expected waiting time to seconds
def E_T_seconds : ℚ := E_T * 60

-- Prove that the expected waiting time in seconds is 40 seconds
theorem average_waiting_time : E_T_seconds = 40 := by
  sorry

end average_waiting_time_l223_223417


namespace compare_logs_l223_223321

noncomputable def log_base (b x : ℝ) := Real.log x / Real.log b

theorem compare_logs (a b c : ℝ) (h1 : a = log_base 4 1.25) (h2 : b = log_base 5 1.2) (h3 : c = log_base 4 8) :
  c > a ∧ a > b :=
by
  sorry

end compare_logs_l223_223321


namespace probability_square_area_l223_223055

theorem probability_square_area (AB : ℝ) (M : ℝ) (h1 : AB = 12) (h2 : 0 ≤ M) (h3 : M ≤ AB) :
  (∃ (AM : ℝ), (AM = M) ∧ (36 ≤ AM^2 ∧ AM^2 ≤ 81)) → 
  (∃ (p : ℝ), p = 1/4) :=
by
  sorry

end probability_square_area_l223_223055


namespace solve_for_square_l223_223592

theorem solve_for_square (x : ℝ) 
  (h : 10 + 9 + 8 * 7 / x + 6 - 5 * 4 - 3 * 2 = 1) : 
  x = 28 := 
by 
  sorry

end solve_for_square_l223_223592


namespace maria_total_flowers_l223_223733

-- Define the initial conditions
def dozens := 3
def flowers_per_dozen := 12
def free_flowers_per_dozen := 2

-- Define the total number of flowers
def total_flowers := dozens * flowers_per_dozen + dozens * free_flowers_per_dozen

-- Assert the proof statement
theorem maria_total_flowers : total_flowers = 42 := sorry

end maria_total_flowers_l223_223733


namespace probability_of_A_winning_is_correct_l223_223037

def even (n : ℕ) : Prop :=
  n % 2 = 0

def probability_A_wins : ℚ :=
  13 / 25

noncomputable def game_probability_of_A_winning : ℚ :=
  let outcomes : List (ℕ × ℕ) := [(1,1), (1,2), (1,3), (1,4), (1,5),
                                     (2,1), (2,2), (2,3), (2,4), (2,5),
                                     (3,1), (3,2), (3,3), (3,4), (3,5),
                                     (4,1), (4,2), (4,3), (4,4), (4,5),
                                     (5,1), (5,2), (5,3), (5,4), (5,5)]
    let winning_outcomes := outcomes.filter (λ pair, even (pair.1 + pair.2))
    (winning_outcomes.length : ℚ) / (outcomes.length : ℚ)

theorem probability_of_A_winning_is_correct : game_probability_of_A_winning = probability_A_wins := 
  sorry

end probability_of_A_winning_is_correct_l223_223037


namespace each_boy_makes_14_l223_223834

/-- Proof that each boy makes 14 dollars given the initial conditions and sales scheme. -/
theorem each_boy_makes_14 (victor_shrimp : ℕ)
                          (austin_shrimp : ℕ)
                          (brian_shrimp : ℕ)
                          (total_shrimp : ℕ)
                          (sets_sold : ℕ)
                          (total_earnings : ℕ)
                          (individual_earnings : ℕ)
                          (h1 : victor_shrimp = 26)
                          (h2 : austin_shrimp = victor_shrimp - 8)
                          (h3 : brian_shrimp = (victor_shrimp + austin_shrimp) / 2)
                          (h4 : total_shrimp = victor_shrimp + austin_shrimp + brian_shrimp)
                          (h5 : sets_sold = total_shrimp / 11)
                          (h6 : total_earnings = sets_sold * 7)
                          (h7 : individual_earnings = total_earnings / 3):
  individual_earnings = 14 := 
by
  sorry

end each_boy_makes_14_l223_223834


namespace correctStatement_l223_223921

variable (U : Set ℕ) (M : Set ℕ)

namespace Proof

-- Given conditions
def universalSet := {1, 2, 3, 4, 5}
def complementM := {1, 3}
def isComplement (M : Set ℕ) : Prop := U \ M = complementM

-- Target statement to be proved
theorem correctStatement (h1 : U = universalSet) (h2 : isComplement M) : 2 ∈ M := by
  sorry

end Proof

end correctStatement_l223_223921


namespace marble_count_l223_223531

theorem marble_count (x : ℝ) (h1 : 0.5 * x = x / 2) (h2 : 0.25 * x = x / 4) (h3 : (27 + 14) = 41) (h4 : 0.25 * x = 27 + 14)
  : x = 164 :=
by
  sorry

end marble_count_l223_223531


namespace minimize_sum_distances_l223_223325

open Real
open Set

variables {A B C O : EuclideanSpace ℝ (Fin 2)}

/-- For a triangle with all angles less than 120 degrees, the Fermat point minimizes
    the sum of distances to the vertices. For a triangle with one angle greater than or equal to 120 degrees,
    the vertex of this angle minimizes the sum of distances to the vertices. -/
theorem minimize_sum_distances (h₁: angle C A B < 120°) (h₂: angle A B C < 120°) (h₃: angle B C A < 120°) 
  (h₄: ¬ angle C A B < 120° ∨ ¬ angle A B C < 120° ∨ ¬ angle B C A < 120°) :
  (∀ O', inside_triangle ABC O' → dist O' A + dist O' B + dist O' C ≥ dist O A + dist O B + dist O C) :=
begin
  sorry
end

end minimize_sum_distances_l223_223325


namespace triangle_inequality_satisfied_for_n_six_l223_223721

theorem triangle_inequality_satisfied_for_n_six :
  ∀ (a b c : ℝ), 0 < a → 0 < b → 0 < c → 6 * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2) → 
  (a + b > c ∧ a + c > b ∧ b + c > a) := sorry

end triangle_inequality_satisfied_for_n_six_l223_223721


namespace beta_max_two_day_ratio_l223_223549

noncomputable def alpha_first_day_score : ℚ := 160 / 300
noncomputable def alpha_second_day_score : ℚ := 140 / 200
noncomputable def alpha_two_day_ratio : ℚ := 300 / 500

theorem beta_max_two_day_ratio :
  ∃ (p q r : ℕ), 
  p < 300 ∧
  q < (8 * p / 15) ∧
  r < ((3500 - 7 * p) / 10) ∧
  q + r = 299 ∧
  gcd 299 500 = 1 ∧
  (299 + 500) = 799 := 
sorry

end beta_max_two_day_ratio_l223_223549


namespace Jia_age_is_24_l223_223827

variable (Jia Yi Bing Ding : ℕ)

theorem Jia_age_is_24
  (h1 : (Jia + Yi + Bing) / 3 = (Jia + Yi + Bing + Ding) / 4 + 1)
  (h2 : (Jia + Yi) / 2 = (Jia + Yi + Bing) / 3 + 1)
  (h3 : Jia = Yi + 4)
  (h4 : Ding = 17) :
  Jia = 24 :=
by
  sorry

end Jia_age_is_24_l223_223827


namespace find_a_plus_b_l223_223478

theorem find_a_plus_b (a b : ℕ) 
  (h1 : 2^(2 * a) + 2^b + 5 = k^2) : a + b = 4 ∨ a + b = 5 :=
sorry

end find_a_plus_b_l223_223478


namespace minimum_f_value_g_ge_f_implies_a_ge_4_l223_223187

noncomputable def f (x : ℝ) : ℝ := 2 * x * Real.log x
noncomputable def g (x a : ℝ) : ℝ := -x^2 + a * x - 3

theorem minimum_f_value : (∃ x : ℝ, f x = 2 / Real.exp 1) :=
  sorry

theorem g_ge_f_implies_a_ge_4 (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ f x ≤ g x a) → a ≥ 4 :=
  sorry

end minimum_f_value_g_ge_f_implies_a_ge_4_l223_223187


namespace problem_U_complement_eq_l223_223914

universe u
variable {α : Type u} [DecidableEq α]
variable (U M : Set α)

theorem problem_U_complement_eq
  (hU : U = ({1, 2, 3, 4, 5} : Set ℕ))
  (hM : U \ M = ({1, 3} : Set ℕ)) : 
  (2 : ℕ) ∈ M := 
by
  sorry

end problem_U_complement_eq_l223_223914


namespace factor_t_squared_minus_81_l223_223304

theorem factor_t_squared_minus_81 (t : ℂ) : (t^2 - 81) = (t - 9) * (t + 9) := 
by
  -- We apply the identity a^2 - b^2 = (a - b) * (a + b)
  let a := t
  let b := 9
  have eq : t^2 - 81 = a^2 - b^2 := by sorry
  rw [eq]
  exact (mul_sub_mul_add_eq_sq_sub_sq a b).symm
  -- Concluding the proof
  sorry -- skipping detailed proof steps for now

end factor_t_squared_minus_81_l223_223304


namespace exponentiation_example_l223_223117

theorem exponentiation_example : (3^2)^4 = 6561 := by
  sorry

end exponentiation_example_l223_223117


namespace boys_to_admit_or_expel_l223_223392

-- Definitions from the conditions
def total_students : ℕ := 500

def girls_percent (x : ℕ) : ℕ := (x * total_students) / 100

-- Definition of the calculation under the new policy
def required_boys : ℕ := (total_students * 3) / 5

-- Main statement we need to prove
theorem boys_to_admit_or_expel (x : ℕ) (htotal : x + girls_percent x = total_students) :
  required_boys - x = 217 := by
  sorry

end boys_to_admit_or_expel_l223_223392


namespace find_a_l223_223587

noncomputable def f (x a : ℝ) : ℝ := x + Real.exp (x - a)
noncomputable def g (x a : ℝ) : ℝ := Real.log (x + 2) - 4 * Real.exp (a - x)

theorem find_a (x₀ a : ℝ) (h : f x₀ a - g x₀ a = 3) : a = -Real.log 2 - 1 :=
by
  sorry

end find_a_l223_223587


namespace committee_with_one_boy_one_girl_prob_l223_223998

def total_members := 30
def boys := 12
def girls := 18
def committee_size := 6

theorem committee_with_one_boy_one_girl_prob :
  let total_ways := Nat.choose total_members committee_size
  let all_boys_ways := Nat.choose boys committee_size
  let all_girls_ways := Nat.choose girls committee_size
  let prob_all_boys_or_all_girls := (all_boys_ways + all_girls_ways) / total_ways
  let desired_prob := 1 - prob_all_boys_or_all_girls
  desired_prob = 19145 / 19793 :=
by
  sorry

end committee_with_one_boy_one_girl_prob_l223_223998


namespace books_sold_correct_l223_223856

-- Define the initial number of books, number of books added, and the final number of books.
def initial_books : ℕ := 41
def added_books : ℕ := 2
def final_books : ℕ := 10

-- Define the number of books sold.
def sold_books : ℕ := initial_books + added_books - final_books

-- The theorem we need to prove: the number of books sold is 33.
theorem books_sold_correct : sold_books = 33 := by
  sorry

end books_sold_correct_l223_223856


namespace problem_U_complement_eq_l223_223917

universe u
variable {α : Type u} [DecidableEq α]
variable (U M : Set α)

theorem problem_U_complement_eq
  (hU : U = ({1, 2, 3, 4, 5} : Set ℕ))
  (hM : U \ M = ({1, 3} : Set ℕ)) : 
  (2 : ℕ) ∈ M := 
by
  sorry

end problem_U_complement_eq_l223_223917


namespace multiple_of_B_share_l223_223288

theorem multiple_of_B_share (A B C : ℝ) (k : ℝ) 
    (h1 : 3 * A = k * B) 
    (h2 : k * B = 7 * 84) 
    (h3 : C = 84)
    (h4 : A + B + C = 427) :
    k = 4 :=
by
  -- We do not need the detailed proof steps here.
  sorry

end multiple_of_B_share_l223_223288


namespace set_union_intersection_l223_223458

def A : Set ℕ := {1, 2}
def B : Set ℕ := {1, 2, 3}
def C : Set ℕ := {2, 3, 4}

theorem set_union_intersection :
  (A ∩ B) ∪ C = {1, 2, 3, 4} := 
by
  sorry

end set_union_intersection_l223_223458


namespace ab_over_a_minus_b_l223_223568

theorem ab_over_a_minus_b (a b : ℝ) (h : (1 / a) - (1 / b) = 1 / 3) : (a * b) / (a - b) = -3 := by
  sorry

end ab_over_a_minus_b_l223_223568


namespace general_term_arithmetic_sum_first_n_terms_geometric_l223_223742

-- Definitions and assumptions based on given conditions
def a (n : ℕ) : ℤ := 2 * n + 1

-- Given conditions
def initial_a1 : ℤ := 3
def common_difference : ℤ := 2

-- Validate the general formula for the arithmetic sequence
theorem general_term_arithmetic : ∀ n : ℕ, a n = 2 * n + 1 := 
by sorry

-- Definitions and assumptions for geometric sequence
def b (n : ℕ) : ℤ := 3^n

-- Sum of the first n terms of the geometric sequence
def Sn (n : ℕ) : ℤ := 3 / 2 * (3^n - 1)

-- Validate the sum formula for the geometric sequence
theorem sum_first_n_terms_geometric (n : ℕ) : Sn n = 3 / 2 * (3^n - 1) := 
by sorry

end general_term_arithmetic_sum_first_n_terms_geometric_l223_223742


namespace sum_in_range_l223_223427

def a : ℚ := 4 + 1/4
def b : ℚ := 2 + 3/4
def c : ℚ := 7 + 1/8

theorem sum_in_range : 14 < a + b + c ∧ a + b + c < 15 := by
  sorry

end sum_in_range_l223_223427


namespace find_f0_f1_l223_223679

noncomputable def f : ℤ → ℤ := sorry

theorem find_f0_f1 :
  (∀ x : ℤ, f (x+5) - f x = 10 * x + 25) →
  (∀ x : ℤ, f (x^3 - 1) = (f x - x)^3 + x^3 - 3) →
  f 0 = -1 ∧ f 1 = 0 := by
  intros h1 h2
  sorry

end find_f0_f1_l223_223679


namespace exponentiation_rule_example_l223_223091

theorem exponentiation_rule_example : (3^2)^4 = 6561 := 
by {
  -- This proof is based on the power rule (ab)^c = a^(b*c)
  have step1 : (3^2)^4 = 3^(2 * 4), 
  {
    rw pow_mul,
  },
  -- Simplification of 3^(2*4) to 3^8
  have step2 : 3^(2 * 4) = 3^8,
  {
    refl,
  },
  -- Actual computation for 3^8 = 6561
  have step3 : 3^8 = 6561,
  {
    norm_num,
  },
  -- Combining all steps
  rw [step1, step2, step3],
  exact rfl,
}

end exponentiation_rule_example_l223_223091


namespace line_through_points_l223_223331

theorem line_through_points :
  ∀ x y : ℝ, (∃ t : ℝ, (x, y) = (2 * t, -3 * (1 - t))) ↔ (x / 2) - (y / 3) = 1 :=
by
  sorry

end line_through_points_l223_223331


namespace binomial_20_13_l223_223894

theorem binomial_20_13 (h₁ : Nat.choose 21 13 = 203490) (h₂ : Nat.choose 21 14 = 116280) :
  Nat.choose 20 13 = 58140 :=
by
  sorry

end binomial_20_13_l223_223894


namespace sum_of_digits_of_d_l223_223226

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
n.digits 10 |>.sum

theorem sum_of_digits_of_d (d : ℕ) 
  (h_exchange : 15 * d = 9 * (d * 5 / 3)) 
  (h_spending : (5 * d / 3) - 120 = d) 
  (h_d_eq : d = 180) : sum_of_digits d = 9 := by
  -- This is where the proof would go
  sorry

end sum_of_digits_of_d_l223_223226


namespace f_ln_inv_4_eq_l223_223015

-- Define the function f as per the given problem
def f : ℝ → ℝ :=
λ x, if x > 0 then Real.exp x else sorry -- We use sorry here to skip the proof

-- Define the specific value ln(1/4)
def ln_inv_4 : ℝ := Real.log (1 / 4)

-- State the theorem to prove that f(ln(1/4)) == e^2 / 4
theorem f_ln_inv_4_eq : f ln_inv_4 = Real.exp 2 / 4 := sorry

end f_ln_inv_4_eq_l223_223015


namespace find_x2_times_x1_plus_x3_l223_223477

noncomputable def a := Real.sqrt 2023
noncomputable def x1 := -Real.sqrt 7
noncomputable def x2 := 1 / a
noncomputable def x3 := Real.sqrt 7

theorem find_x2_times_x1_plus_x3 :
  let x1 := -Real.sqrt 7
  let x2 := 1 / Real.sqrt 2023
  let x3 := Real.sqrt 7
  x2 * (x1 + x3) = 0 :=
by
  sorry

end find_x2_times_x1_plus_x3_l223_223477


namespace snow_at_least_once_l223_223243

noncomputable def prob_snow_at_least_once (p1 p2 p3: ℚ) : ℚ :=
  1 - (1 - p1) * (1 - p2) * (1 - p3)

theorem snow_at_least_once : 
  prob_snow_at_least_once (1/2) (2/3) (3/4) = 23 / 24 := 
by
  sorry

end snow_at_least_once_l223_223243


namespace percent_of_part_l223_223847

variable (Part : ℕ) (Whole : ℕ)

theorem percent_of_part (hPart : Part = 70) (hWhole : Whole = 280) :
  (Part / Whole) * 100 = 25 := by
  sorry

end percent_of_part_l223_223847


namespace rat_op_neg2_3_rat_op_4_neg2_eq_neg2_4_l223_223316

namespace RationalOperation

-- Definition of the operation ⊗ for rational numbers
def rat_op (a b : ℚ) : ℚ := a * b - a - b - 2

-- Proof problem 1: (-2) ⊗ 3 = -9
theorem rat_op_neg2_3 : rat_op (-2) 3 = -9 :=
by
  sorry

-- Proof problem 2: 4 ⊗ (-2) = (-2) ⊗ 4
theorem rat_op_4_neg2_eq_neg2_4 : rat_op 4 (-2) = rat_op (-2) 4 :=
by
  sorry

end RationalOperation

end rat_op_neg2_3_rat_op_4_neg2_eq_neg2_4_l223_223316


namespace trig_inequality_2016_l223_223219

theorem trig_inequality_2016 :
  let a := Real.sin (Real.cos (2016 * Real.pi / 180))
  let b := Real.sin (Real.sin (2016 * Real.pi / 180))
  let c := Real.cos (Real.sin (2016 * Real.pi / 180))
  let d := Real.cos (Real.cos (2016 * Real.pi / 180))
  c > d ∧ d > b ∧ b > a := by
  sorry

end trig_inequality_2016_l223_223219


namespace emily_gardens_and_seeds_l223_223883

variables (total_seeds planted_big_garden tom_seeds lettuce_seeds pepper_seeds tom_gardens lettuce_gardens pepper_gardens : ℕ)

def seeds_left (total_seeds planted_big_garden : ℕ) : ℕ :=
  total_seeds - planted_big_garden

def seeds_used_tomatoes (tom_seeds tom_gardens : ℕ) : ℕ :=
  tom_seeds * tom_gardens

def seeds_used_lettuce (lettuce_seeds lettuce_gardens : ℕ) : ℕ :=
  lettuce_seeds * lettuce_gardens

def seeds_used_peppers (pepper_seeds pepper_gardens : ℕ) : ℕ :=
  pepper_seeds * pepper_gardens

def remaining_seeds (total_seeds planted_big_garden tom_seeds tom_gardens lettuce_seeds lettuce_gardens : ℕ) : ℕ :=
  seeds_left total_seeds planted_big_garden - (seeds_used_tomatoes tom_seeds tom_gardens + seeds_used_lettuce lettuce_seeds lettuce_gardens)

def total_small_gardens (tom_gardens lettuce_gardens pepper_gardens : ℕ) : ℕ :=
  tom_gardens + lettuce_gardens + pepper_gardens

theorem emily_gardens_and_seeds :
  total_seeds = 42 ∧
  planted_big_garden = 36 ∧
  tom_seeds = 4 ∧
  lettuce_seeds = 3 ∧
  pepper_seeds = 2 ∧
  tom_gardens = 3 ∧
  lettuce_gardens = 2 →
  seeds_used_peppers pepper_seeds pepper_gardens = 0 ∧
  total_small_gardens tom_gardens lettuce_gardens pepper_gardens = 5 :=
by
  sorry

end emily_gardens_and_seeds_l223_223883


namespace heartsuit_calc_l223_223556

-- Define the operation x ♡ y = 4x + 6y
def heartsuit (x y : ℝ) : ℝ := 4 * x + 6 * y

-- State the theorem
theorem heartsuit_calc : heartsuit 5 3 = 38 := by
  -- Proof omitted
  sorry

end heartsuit_calc_l223_223556


namespace distance_between_lamps_l223_223625

/-- 
A rectangular classroom measures 10 meters in length. Two lamps emitting conical light beams with a 90° opening angle 
are installed on the ceiling. The first lamp is located at the center of the ceiling and illuminates a circle on the 
floor with a diameter of 6 meters. The second lamp is adjusted such that the illuminated area along the length 
of the classroom spans a 10-meter section without reaching the opposite walls. Prove that the distance between the 
two lamps is 4 meters.
-/
theorem distance_between_lamps : 
  ∀ (length width height : ℝ) (center_illum_radius illum_length : ℝ) (d_center_to_lamp1 d_center_to_lamp2 dist_lamps : ℝ),
  length = 10 ∧ d_center_to_lamp1 = 3 ∧ d_center_to_lamp2 = 1 ∧ dist_lamps = 4 → d_center_to_lamp1 - d_center_to_lamp2 = dist_lamps :=
by
  intros length width height center_illum_radius illum_length d_center_to_lamp1 d_center_to_lamp2 dist_lamps conditions
  sorry

end distance_between_lamps_l223_223625


namespace guilt_of_X_and_Y_l223_223700

-- Definitions
variable (X Y : Prop)

-- Conditions
axiom condition1 : ¬X ∨ Y
axiom condition2 : X

-- Conclusion to prove
theorem guilt_of_X_and_Y : X ∧ Y := by
  sorry

end guilt_of_X_and_Y_l223_223700


namespace digits_arithmetic_l223_223849

theorem digits_arithmetic :
  (12 / 3 / 4) * (56 / 7 / 8) = 1 :=
by
  sorry

end digits_arithmetic_l223_223849


namespace solution_of_system_l223_223887

theorem solution_of_system :
  ∃ x y : ℝ, (x^4 + y^4 = 17) ∧ (x + y = 3) ∧ ((x = 1 ∧ y = 2) ∨ (x = 2 ∧ y = 1)) :=
by
  sorry

end solution_of_system_l223_223887


namespace solution_set_fraction_inequality_l223_223822

theorem solution_set_fraction_inequality : 
  { x : ℝ | 0 < x ∧ x < 1/3 } = { x : ℝ | 1/x > 3 } :=
by
  sorry

end solution_set_fraction_inequality_l223_223822


namespace parabola_satisfies_given_condition_l223_223382

variable {p : ℝ}
variable {x1 x2 : ℝ}

-- Condition 1: The equation of the parabola is y^2 = 2px where p > 0.
def parabola_equation (p : ℝ) (x y : ℝ) : Prop :=
  y^2 = 2 * p * x

-- Condition 2: The parabola has a focus F.
-- Condition 3: A line passes through the focus F with an inclination angle of π/3.
def line_through_focus (p : ℝ) (x y : ℝ) : Prop :=
  y = Real.sqrt 3 * (x - p / 2)

-- Condition 4 & 5: The line intersects the parabola at points A and B with distance |AB| = 8.
def intersection_points (p : ℝ) (x1 x2 : ℝ) : Prop :=
  x1 ≠ x2 ∧ parabola_equation p x1 (Real.sqrt 3 * (x1 - p / 2)) ∧ parabola_equation p x2 (Real.sqrt 3 * (x2 - p / 2)) ∧
  abs (x1 - x2) * Real.sqrt (1 + 3) = 8

-- The proof statement
theorem parabola_satisfies_given_condition (hp : 0 < p) (hintersect : intersection_points p x1 x2) : 
  parabola_equation 3 x1 (Real.sqrt 3 * (x1 - 3 / 2)) ∧ parabola_equation 3 x2 (Real.sqrt 3 * (x2 - 3 / 2)) := sorry

end parabola_satisfies_given_condition_l223_223382


namespace tessa_needs_more_apples_l223_223790

/-- Tessa starts with 4 apples.
    Anita gives her 5 more apples.
    She needs 10 apples to make a pie.
    Prove that she needs 1 more apple to make the pie.
-/
theorem tessa_needs_more_apples:
  ∀ initial_apples extra_apples total_needed extra_needed: ℕ,
    initial_apples = 4 → extra_apples = 5 → total_needed = 10 →
    extra_needed = total_needed - (initial_apples + extra_apples) →
    extra_needed = 1 :=
by
  intros initial_apples extra_apples total_needed extra_needed hi he ht heq
  rw [hi, he, ht] at heq
  simp at heq
  assumption

end tessa_needs_more_apples_l223_223790


namespace worst_player_is_son_l223_223169

-- Define the types of players and relationships
inductive Sex
| male
| female

structure Player where
  name : String
  sex : Sex
  age : Nat

-- Define the four players
def woman := Player.mk "woman" Sex.female 30  -- Age is arbitrary
def brother := Player.mk "brother" Sex.male 30
def son := Player.mk "son" Sex.male 10
def daughter := Player.mk "daughter" Sex.female 10

-- Define the conditions
def opposite_sex (p1 p2 : Player) : Prop := p1.sex ≠ p2.sex
def same_age (p1 p2 : Player) : Prop := p1.age = p2.age

-- Define the worst player and the best player
variable (worst_player : Player) (best_player : Player)

-- Conditions as hypotheses
axiom twin_condition : ∃ twin : Player, (twin ≠ worst_player) ∧ (opposite_sex twin best_player)
axiom age_condition : same_age worst_player best_player
axiom not_same_player : worst_player ≠ best_player

-- Prove that the worst player is the son
theorem worst_player_is_son : worst_player = son :=
by
  sorry

end worst_player_is_son_l223_223169


namespace least_number_to_add_l223_223846

theorem least_number_to_add (n divisor : ℕ) (h₁ : n = 27306) (h₂ : divisor = 151) : 
  ∃ k : ℕ, k = 25 ∧ (n + k) % divisor = 0 := 
by
  sorry

end least_number_to_add_l223_223846


namespace area_of_square_eq_36_l223_223689

theorem area_of_square_eq_36 :
  ∃ (s q : ℝ), q = 6 ∧ s = 10 ∧ (∃ p : ℝ, p = 24 ∧ (p / 4) * (p / 4) = 36) := 
by
  sorry

end area_of_square_eq_36_l223_223689


namespace malia_berries_second_bush_l223_223481

theorem malia_berries_second_bush :
  ∀ (b2 : ℕ), ∃ (d1 d2 d3 d4 : ℕ),
  d1 = 3 → d2 = 7 → d3 = 12 → d4 = 19 →
  d2 - d1 = (d3 - d2) - 2 →
  d3 - d2 = (d4 - d3) - 2 →
  b2 = d1 + (d2 - d1 - 2) →
  b2 = 6 :=
by
  sorry

end malia_berries_second_bush_l223_223481


namespace ball_bounces_l223_223855

theorem ball_bounces (k : ℕ) :
  1500 * (2 / 3 : ℝ)^k < 2 ↔ k ≥ 19 :=
sorry

end ball_bounces_l223_223855


namespace Portia_school_students_l223_223779

theorem Portia_school_students:
  ∃ (P L : ℕ), P = 2 * L ∧ P + L = 3000 ∧ P = 2000 :=
by
  sorry

end Portia_school_students_l223_223779


namespace probability_three_students_different_courses_probability_two_courses_not_chosen_l223_223826

/--
Problem (Ⅰ):
Calculate the probability that all three students choose different elective courses.
Given:
- There are four elective courses.
- Each student must choose exactly one elective course.
We need to prove that the probability that all three students choose different elective courses is 3 / 8.
-/
theorem probability_three_students_different_courses : 
  let total_ways := 4^3 in
  let different_ways := Nat.factorial 4 / Nat.factorial (4 - 3) in
  (different_ways : ℚ) / total_ways = 3 / 8 :=
by
  let total_ways := 4^3
  let different_ways := Nat.factorial 4 / Nat.factorial (4 - 3)
  have h : (3 : ℚ) / 8 = 3 / 8 := rfl
  rw [h]
  sorry

/--
Problem (Ⅱ):
Calculate the probability that exactly two elective courses are not chosen by any of the three students.
Given:
- There are four elective courses.
- Each student must choose exactly one elective course.
We need to prove that the probability that exactly two elective courses are not chosen by any of the three students is 9 / 16.
-/
theorem probability_two_courses_not_chosen : 
  let total_ways := 4^3 in
  let num_ways_not_chosen := (Nat.choose 4 2) * (Nat.choose 3 2) * (Nat.factorial 2 / Nat.factorial (2 - 2)) in
  (num_ways_not_chosen : ℚ) / total_ways = 9 / 16 :=
by
  let total_ways := 4^3
  let num_ways_not_chosen := (Nat.choose 4 2) * (Nat.choose 3 2) * (Nat.factorial 2 / Nat.factorial (2 - 2))
  have h : (9 : ℚ) / 16 = 9 / 16 := rfl
  rw [h]
  sorry

end probability_three_students_different_courses_probability_two_courses_not_chosen_l223_223826


namespace largest_8_11_double_l223_223704

def is_8_11_double (M : ℕ) : Prop :=
  let digits_8 := (Nat.digits 8 M)
  let M_11 := Nat.ofDigits 11 digits_8
  M_11 = 2 * M

theorem largest_8_11_double : ∃ (M : ℕ), is_8_11_double M ∧ ∀ (N : ℕ), is_8_11_double N → N ≤ M :=
sorry

end largest_8_11_double_l223_223704


namespace ratio_distance_l223_223442

-- Definitions based on conditions
def speed_ferry_P : ℕ := 6 -- speed of ferry P in km/h
def time_ferry_P : ℕ := 3 -- travel time of ferry P in hours
def speed_ferry_Q : ℕ := speed_ferry_P + 3 -- speed of ferry Q in km/h
def time_ferry_Q : ℕ := time_ferry_P + 1 -- travel time of ferry Q in hours

-- Calculating the distances
def distance_ferry_P : ℕ := speed_ferry_P * time_ferry_P -- distance covered by ferry P
def distance_ferry_Q : ℕ := speed_ferry_Q * time_ferry_Q -- distance covered by ferry Q

-- Main theorem to prove
theorem ratio_distance (d_P d_Q : ℕ) (h_dP : d_P = distance_ferry_P) (h_dQ : d_Q = distance_ferry_Q) : d_Q / d_P = 2 :=
by
  sorry

end ratio_distance_l223_223442


namespace no_positive_integers_m_n_l223_223064

theorem no_positive_integers_m_n (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  m^3 + 11^3 ≠ n^3 :=
sorry

end no_positive_integers_m_n_l223_223064


namespace circumscribed_circle_area_l223_223280

noncomputable def circumradius (s : ℝ) : ℝ := s / Real.sqrt 3
noncomputable def circle_area (r : ℝ) : ℝ := Real.pi * r^2

theorem circumscribed_circle_area (s : ℝ) (hs : s = 15) : circle_area (circumradius s) = 75 * Real.pi :=
by
  sorry

end circumscribed_circle_area_l223_223280


namespace sum_of_three_ints_product_5_4_l223_223816

theorem sum_of_three_ints_product_5_4 :
  ∃ (a b c: ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ a * b * c = 5^4 ∧ a + b + c = 51 :=
by
  sorry

end sum_of_three_ints_product_5_4_l223_223816


namespace f_shift_l223_223585

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x

theorem f_shift (x : ℝ) : f (x - 1) = x^2 - 4 * x + 3 := by
  sorry

end f_shift_l223_223585


namespace alpha_eq_beta_l223_223319

variable {α β : ℝ}

theorem alpha_eq_beta
  (h_alpha : 0 < α ∧ α < (π / 2))
  (h_beta : 0 < β ∧ β < (π / 2))
  (h_sin : Real.sin (α + β) + Real.sin (α - β) = Real.sin (2 * β)) :
  α = β :=
by
  sorry

end alpha_eq_beta_l223_223319


namespace geometric_sequence_a5_l223_223575

theorem geometric_sequence_a5 (a : ℕ → ℝ) (r : ℝ) (h1 : ∀ n, a (n + 1) = a n * r) (h2 : a 2 * a 8 = 4) : a 5 = 2 :=
sorry

end geometric_sequence_a5_l223_223575


namespace area_enclosed_by_region_l223_223839

theorem area_enclosed_by_region : ∀ (x y : ℝ), (x^2 + y^2 - 8*x + 6*y = -9) → (π * (4 ^ 2) = 16 * π) :=
by
  intro x y h
  sorry

end area_enclosed_by_region_l223_223839


namespace cody_initial_tickets_l223_223425

theorem cody_initial_tickets (T : ℕ) (h1 : T - 25 + 6 = 30) : T = 49 :=
sorry

end cody_initial_tickets_l223_223425


namespace isosceles_triangle_base_length_l223_223805

-- Definitions based on the conditions
def congruent_side : Nat := 7
def perimeter : Nat := 23

-- Statement to prove
theorem isosceles_triangle_base_length :
  let b := perimeter - 2 * congruent_side in b = 9 :=
by
  sorry

end isosceles_triangle_base_length_l223_223805


namespace inequality_solution_l223_223715

theorem inequality_solution (x : ℝ) :
  (1 / (x * (x + 1)) - 1 / ((x + 1) * (x + 2)) < 1/4) ∧ (x - 2 > 0) → x > 2 :=
by {
  sorry
}

end inequality_solution_l223_223715


namespace number_of_ordered_triples_l223_223963

noncomputable def count_triples : Nat := 50

theorem number_of_ordered_triples 
    (x y z : Nat)
    (hx : x > 0)
    (hy : y > 0)
    (hz : z > 0)
    (H1 : Nat.lcm x y = 500)
    (H2 : Nat.lcm y z = 1000)
    (H3 : Nat.lcm z x = 1000) :
    ∃ (n : Nat), n = count_triples := 
by
    use 50
    sorry

end number_of_ordered_triples_l223_223963


namespace power_of_powers_eval_powers_l223_223136

theorem power_of_powers (a m n : ℕ) : (a^m)^n = a^(m * n) :=
  by sorry

theorem eval_powers : (3^2)^4 = 6561 :=
  by
    rw [power_of_powers]
    -- further proof of (3^8 = 6561) would go here
    sorry

end power_of_powers_eval_powers_l223_223136


namespace Sammy_has_8_bottle_caps_l223_223062

-- Definitions representing the conditions
def BilliesBottleCaps := 2
def JaninesBottleCaps := 3 * BilliesBottleCaps
def SammysBottleCaps := JaninesBottleCaps + 2

-- Goal: Prove that Sammy has 8 bottle caps
theorem Sammy_has_8_bottle_caps : 
  SammysBottleCaps = 8 := 
sorry

end Sammy_has_8_bottle_caps_l223_223062


namespace exp_eval_l223_223105

theorem exp_eval : (3^2)^4 = 6561 := by
sorry

end exp_eval_l223_223105


namespace same_number_written_every_vertex_l223_223541

theorem same_number_written_every_vertex (a : ℕ → ℝ) (h1 : ∀ i : ℕ, 1 ≤ i ∧ i ≤ 2019 → a i > 0) 
(h2 : ∀ i : ℕ, 1 ≤ i ∧ i ≤ 2019 → (a i) ^ 2 = a (i - 1) + a (i + 1) ) :
  ∀ i : ℕ, 1 ≤ i ∧ i ≤ 2019 → a i = 2 :=
by
  sorry

end same_number_written_every_vertex_l223_223541


namespace minimum_value_l223_223008

noncomputable def min_value (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : 1 / (a + 1) + 1 / (b + 1) = 1) :=
  a + 2 * b

theorem minimum_value (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : 1 / (a + 1) + 1 / (b + 1) = 1) :
  min_value a b h₁ h₂ h₃ ≥ 2 * Real.sqrt 2 :=
sorry

end minimum_value_l223_223008


namespace total_hours_charged_l223_223626

variables (K P M : ℕ)

theorem total_hours_charged (h1 : P = 2 * K) 
                            (h2 : P = (1/3 : ℚ) * (M : ℚ)) 
                            (h3 : M = K + 85) : 
  K + P + M = 153 := 
by 
  sorry

end total_hours_charged_l223_223626


namespace joint_purchases_popular_l223_223271

-- Define the conditions stating what makes joint purchases feasible
structure Conditions where
  cost_saving : Prop  -- Joint purchases allow significant cost savings.
  shared_overhead : Prop  -- Overhead costs are distributed among all members.
  collective_quality_assessment : Prop  -- Enhanced quality assessment via collective feedback.
  community_trust : Prop  -- Trust within the community encourages honest feedback.

-- Define the proposition stating the popularity of joint purchases
theorem joint_purchases_popular (cond : Conditions) : 
  cond.cost_saving ∧ cond.shared_overhead ∧ cond.collective_quality_assessment ∧ cond.community_trust → 
  Prop := 
by 
  intro h
  sorry

end joint_purchases_popular_l223_223271


namespace largest_divisor_l223_223349

theorem largest_divisor (n : ℕ) (hn : n > 0) (h : 360 ∣ n^3) :
  ∃ w : ℕ, w > 0 ∧ w ∣ n ∧ ∀ d : ℕ, (d > 0 ∧ d ∣ n) → d ≤ 30 := 
sorry

end largest_divisor_l223_223349


namespace power_of_powers_l223_223089

theorem power_of_powers :
  (3^2)^4 = 6561 := 
by 
  sorry

end power_of_powers_l223_223089


namespace total_distance_walked_l223_223850

theorem total_distance_walked 
  (d1 : ℝ) (d2 : ℝ)
  (h1 : d1 = 0.75)
  (h2 : d2 = 0.25) :
  d1 + d2 = 1 :=
by
  sorry

end total_distance_walked_l223_223850


namespace friend_redistribute_l223_223731

-- Definition and total earnings
def earnings : List Int := [30, 45, 15, 10, 60]
def total_earnings := earnings.sum

-- Number of friends
def number_of_friends : Int := 5

-- Calculate the equal share
def equal_share := total_earnings / number_of_friends

-- Calculate the amount to redistribute by the friend who earned 60
def amount_to_give := 60 - equal_share

theorem friend_redistribute :
  earnings.sum = 160 ∧ equal_share = 32 ∧ amount_to_give = 28 :=
by
  -- Proof goes here, skipped with 'sorry'
  sorry

end friend_redistribute_l223_223731


namespace exponentiation_identity_l223_223112

theorem exponentiation_identity (a m n : ℕ) : (a^m)^n = a^(m * n) :=
by sorry

example : (3^2)^4 = 6561 :=
by
  have h := exponentiation_identity 3 2 4
  rw [←h]
  norm_num

end exponentiation_identity_l223_223112


namespace power_of_powers_l223_223083

theorem power_of_powers :
  (3^2)^4 = 6561 := 
by 
  sorry

end power_of_powers_l223_223083


namespace large_posters_count_l223_223165

theorem large_posters_count (total_posters small_ratio medium_ratio : ℕ) (h_total : total_posters = 50) (h_small_ratio : small_ratio = 2/5) (h_medium_ratio : medium_ratio = 1/2) :
  let small_posters := (small_ratio * total_posters) in
  let medium_posters := (medium_ratio * total_posters) in
  let large_posters := total_posters - (small_posters + medium_posters) in
  large_posters = 5 := by
{
  sorry
}

end large_posters_count_l223_223165


namespace find_interest_rate_l223_223675

theorem find_interest_rate (P : ℕ) (diff : ℕ) (T : ℕ) (I2_rate : ℕ) (r : ℚ) 
  (hP : P = 15000) (hdiff : diff = 900) (hT : T = 2) (hI2_rate : I2_rate = 12)
  (h : P * (r / 100) * T = P * (I2_rate / 100) * T + diff) :
  r = 15 :=
sorry

end find_interest_rate_l223_223675


namespace solve_for_x_l223_223952

variables {x y : ℝ}

theorem solve_for_x (h : x / (x - 3) = (y^2 + 3 * y + 1) / (y^2 + 3 * y - 4)) : 
  x = (3 * y^2 + 9 * y + 3) / 5 :=
sorry

end solve_for_x_l223_223952


namespace all_options_valid_l223_223984

-- Definition of the line equation
def line_eq (x y : ℝ) : Prop := y = 2 * x - 4

-- Definitions of parameterizations for each option
def option_A (t : ℝ) : ℝ × ℝ := ⟨2 + (-1) * t, 0 + (-2) * t⟩
def option_B (t : ℝ) : ℝ × ℝ := ⟨6 + 4 * t, 8 + 8 * t⟩
def option_C (t : ℝ) : ℝ × ℝ := ⟨1 + 1 * t, -2 + 2 * t⟩
def option_D (t : ℝ) : ℝ × ℝ := ⟨0 + 0.5 * t, -4 + 1 * t⟩
def option_E (t : ℝ) : ℝ × ℝ := ⟨-2 + (-2) * t, -8 + (-4) * t⟩

-- The main statement to prove
theorem all_options_valid :
  (∀ t, line_eq (option_A t).1 (option_A t).2) ∧
  (∀ t, line_eq (option_B t).1 (option_B t).2) ∧
  (∀ t, line_eq (option_C t).1 (option_C t).2) ∧
  (∀ t, line_eq (option_D t).1 (option_D t).2) ∧
  (∀ t, line_eq (option_E t).1 (option_E t).2) :=
by sorry -- proof omitted

end all_options_valid_l223_223984


namespace chairs_bought_l223_223837

theorem chairs_bought (C : ℕ) (tables chairs total_time time_per_furniture : ℕ)
  (h1 : tables = 4)
  (h2 : time_per_furniture = 6)
  (h3 : total_time = 48)
  (h4 : total_time = time_per_furniture * (tables + chairs)) :
  C = 4 :=
by
  -- proof steps are omitted
  sorry

end chairs_bought_l223_223837


namespace sum_of_numbers_l223_223150

theorem sum_of_numbers : 3 + 33 + 333 + 33.3 = 402.3 :=
  by
    sorry

end sum_of_numbers_l223_223150


namespace isosceles_triangle_circumscribed_radius_and_height_l223_223239

/-
Conditions:
- The isosceles triangle has two equal sides of 20 inches.
- The base of the triangle is 24 inches.

Prove:
1. The radius of the circumscribed circle is 5 inches.
2. The height of the triangle is 16 inches.
-/

theorem isosceles_triangle_circumscribed_radius_and_height 
  (h_eq_sides : ∀ A B C : Type, ∀ (AB AC : ℝ), ∀ (BC : ℝ), AB = 20 → AC = 20 → BC = 24) 
  (R : ℝ) (h : ℝ) : 
  R = 5 ∧ h = 16 := 
sorry

end isosceles_triangle_circumscribed_radius_and_height_l223_223239


namespace isosceles_triangle_base_length_l223_223795

-- Define the conditions
def side_length : ℕ := 7
def perimeter : ℕ := 23

-- Define the theorem to prove the length of the base
theorem isosceles_triangle_base_length (b : ℕ) (h : 2 * side_length + b = perimeter) : b = 9 :=
by
  sorry

end isosceles_triangle_base_length_l223_223795


namespace night_rides_total_l223_223664

-- Definitions corresponding to the conditions in the problem
def total_ferris_wheel_rides : Nat := 13
def total_roller_coaster_rides : Nat := 9
def ferris_wheel_day_rides : Nat := 7
def roller_coaster_day_rides : Nat := 4

-- The total night rides proof problem
theorem night_rides_total :
  let ferris_wheel_night_rides := total_ferris_wheel_rides - ferris_wheel_day_rides
  let roller_coaster_night_rides := total_roller_coaster_rides - roller_coaster_day_rides
  ferris_wheel_night_rides + roller_coaster_night_rides = 11 :=
by
  -- Proof skipped
  sorry

end night_rides_total_l223_223664


namespace arccos_cos_eq_x_div_3_solutions_l223_223491

theorem arccos_cos_eq_x_div_3_solutions (x : ℝ) :
  (Real.arccos (Real.cos x) = x / 3) ∧ (-3 * Real.pi / 2 ≤ x ∧ x ≤ 3 * Real.pi / 2) 
  ↔ x = -3 * Real.pi / 2 ∨ x = 0 ∨ x = 3 * Real.pi / 2 :=
by
  sorry

end arccos_cos_eq_x_div_3_solutions_l223_223491


namespace angle_between_hands_at_3_40_l223_223869

def degrees_per_minute_minute_hand := 360 / 60
def minutes_passed := 40
def degrees_minute_hand := degrees_per_minute_minute_hand * minutes_passed -- 240 degrees

def degrees_per_hour_hour_hand := 360 / 12
def hours_passed := 3
def degrees_hour_hand_at_hour := degrees_per_hour_hour_hand * hours_passed -- 90 degrees

def degrees_per_minute_hour_hand := degrees_per_hour_hour_hand / 60
def degrees_hour_hand_additional := degrees_per_minute_hour_hand * minutes_passed -- 20 degrees

def total_degrees_hour_hand := degrees_hour_hand_at_hour + degrees_hour_hand_additional -- 110 degrees

def expected_angle_between_hands := 130

theorem angle_between_hands_at_3_40
  (h1: degrees_minute_hand = 240)
  (h2: total_degrees_hour_hand = 110):
  (degrees_minute_hand - total_degrees_hour_hand = expected_angle_between_hands) :=
by
  sorry

end angle_between_hands_at_3_40_l223_223869


namespace find_base_of_numeral_system_l223_223502

def base_of_numeral_system (x : ℕ) : Prop :=
  (3 * x + 4)^2 = x^3 + 5 * x^2 + 5 * x + 2

theorem find_base_of_numeral_system :
  ∃ x : ℕ, base_of_numeral_system x ∧ x = 7 := sorry

end find_base_of_numeral_system_l223_223502


namespace power_of_powers_l223_223084

theorem power_of_powers :
  (3^2)^4 = 6561 := 
by 
  sorry

end power_of_powers_l223_223084


namespace girls_left_class_l223_223383

variable (G B G₂ B₁ : Nat)

theorem girls_left_class (h₁ : 5 * B = 6 * G) 
                         (h₂ : B = 120)
                         (h₃ : 2 * B₁ = 3 * G₂)
                         (h₄ : B₁ = B) : 
                         G - G₂ = 20 :=
by
  sorry

end girls_left_class_l223_223383


namespace power_of_powers_eval_powers_l223_223131

theorem power_of_powers (a m n : ℕ) : (a^m)^n = a^(m * n) :=
  by sorry

theorem eval_powers : (3^2)^4 = 6561 :=
  by
    rw [power_of_powers]
    -- further proof of (3^8 = 6561) would go here
    sorry

end power_of_powers_eval_powers_l223_223131


namespace factor_expression_l223_223297

theorem factor_expression (x : ℝ) : 9 * x^2 + 3 * x = 3 * x * (3 * x + 1) := 
by
  sorry

end factor_expression_l223_223297


namespace final_cost_is_35_l223_223788

-- Definitions based on conditions
def original_price : ℕ := 50
def discount_rate : ℚ := 0.30
def discount_amount : ℚ := original_price * discount_rate
def final_cost : ℚ := original_price - discount_amount

-- The theorem we need to prove
theorem final_cost_is_35 : final_cost = 35 := by
  sorry

end final_cost_is_35_l223_223788


namespace sum_of_edges_rectangular_solid_l223_223650

theorem sum_of_edges_rectangular_solid
  (a r : ℝ)
  (hr : r ≠ 0)
  (volume_eq : (a / r) * a * (a * r) = 512)
  (surface_area_eq : 2 * ((a ^ 2) / r + a ^ 2 + (a ^ 2) * r) = 384)
  (geo_progression : true) : -- This is implicitly understood in the construction
  4 * ((a / r) + a + (a * r)) = 112 :=
by
  -- The proof will be placed here
  sorry

end sum_of_edges_rectangular_solid_l223_223650


namespace maximize_profit_l223_223671

-- Definitions
def initial_employees := 320
def profit_per_employee := 200000
def profit_increase_per_layoff := 20000
def expense_per_laid_off_employee := 60000
def min_employees := (3 * initial_employees) / 4
def profit_function (x : ℝ) := -0.2 * x^2 + 38 * x + 6400

-- The main statement
theorem maximize_profit : ∃ x : ℝ, 0 ≤ x ∧ x ≤ 80 ∧ (∀ y : ℝ, 0 ≤ y ∧ y ≤ 80 → profit_function y ≤ profit_function x) ∧ x = 80 :=
by
  sorry

end maximize_profit_l223_223671


namespace product_of_p_r_s_l223_223345

theorem product_of_p_r_s (p r s : ℕ) 
  (h1 : 4^p + 4^3 = 280)
  (h2 : 3^r + 29 = 56) 
  (h3 : 7^s + 6^3 = 728) : 
  p * r * s = 27 :=
by
  sorry

end product_of_p_r_s_l223_223345


namespace exponentiation_example_l223_223120

theorem exponentiation_example : (3^2)^4 = 6561 := by
  sorry

end exponentiation_example_l223_223120


namespace janous_inequality_l223_223771

theorem janous_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x ≥ y + z) :
  (x + y) / z + (y + z) / x + (z + x) / y ≥ 7 :=
sorry

end janous_inequality_l223_223771


namespace power_calc_l223_223127

theorem power_calc : (3^2)^4 = 6561 := 
by
  sorry

end power_calc_l223_223127


namespace monotonic_intervals_range_of_k_l223_223584

noncomputable def f (x a : ℝ) : ℝ :=
  (x - a - 1) * Real.exp x - (1/2) * x^2 + a * x

-- Conditions: a > 0
variables (a : ℝ) (h_a : 0 < a)

-- Part (1): Monotonic Intervals
theorem monotonic_intervals :
  (∀ x, f x a < f (x + 1) a ↔ x < 0 ∨ a < x) ∧
  (∀ x, f (x + 1) a < f x a ↔ 0 < x ∧ x < a) :=
  sorry

-- Part (2): Range of k
theorem range_of_k (x1 x2 : ℝ) (hx1 : f x1 a = 0) (hx2 : f x2 a = 0) :
  (f x1 a - f x2 a < k * a^3) ↔ k ≥ -1/6 :=
  sorry

end monotonic_intervals_range_of_k_l223_223584


namespace positive_integers_satisfy_inequality_l223_223499

theorem positive_integers_satisfy_inequality :
  ∀ (n : ℕ), 2 * n - 5 < 5 - 2 * n ↔ n = 1 ∨ n = 2 :=
by
  intro n
  sorry

end positive_integers_satisfy_inequality_l223_223499


namespace seating_arrangement_l223_223977

-- Define the conditions first
def martians : ℕ := 4
def venusians : ℕ := 4
def earthlings : ℕ := 4
def chairs : ℕ := 12
def chair1_occ := "Martian"
def chair12_occ := "Earthling"
def no_earthling_left_of_martian := true
def no_martian_left_of_venusian := true
def no_venusian_left_of_earthling := true

-- Assertion of the problem statement
theorem seating_arrangement (chairs = 12) 
    (martians = 4) 
    (venusians = 4) 
    (earthlings = 4)
    (chair1_occ = "Martian") 
    (chair12_occ = "Earthling") 
    (no_earthling_left_of_martian = true)
    (no_martian_left_of_venusian = true) 
    (no_venusian_left_of_earthling = true) : 
    ∃ N : ℕ, 
    (N * (factorial martians) * (factorial venusians) * (factorial earthlings)) = N * (4!)^3 
    ∧ N = 50 :=
by 
    sorry

end seating_arrangement_l223_223977


namespace minimum_n_for_obtuse_triangle_l223_223324

def α₀ : ℝ := 60 
def β₀ : ℝ := 59.999
def γ₀ : ℝ := 60.001

def α (n : ℕ) : ℝ := (-2)^n * (α₀ - 60) + 60
def β (n : ℕ) : ℝ := (-2)^n * (β₀ - 60) + 60
def γ (n : ℕ) : ℝ := (-2)^n * (γ₀ - 60) + 60

theorem minimum_n_for_obtuse_triangle : ∃ n : ℕ, β n > 90 ∧ ∀ m : ℕ, m < n → β m ≤ 90 :=
by sorry

end minimum_n_for_obtuse_triangle_l223_223324


namespace jeep_initial_distance_l223_223285

theorem jeep_initial_distance (D : ℝ) (h1 : ∀ t : ℝ, t = 4 → D / t = 103.33 * (3 / 8)) :
  D = 275.55 :=
sorry

end jeep_initial_distance_l223_223285


namespace exponentiation_identity_l223_223108

theorem exponentiation_identity (a m n : ℕ) : (a^m)^n = a^(m * n) :=
by sorry

example : (3^2)^4 = 6561 :=
by
  have h := exponentiation_identity 3 2 4
  rw [←h]
  norm_num

end exponentiation_identity_l223_223108


namespace ones_digit_of_power_35_35_pow_17_17_is_five_l223_223561

theorem ones_digit_of_power_35_35_pow_17_17_is_five :
  (35 ^ (35 * (17 ^ 17))) % 10 = 5 := by
  sorry

end ones_digit_of_power_35_35_pow_17_17_is_five_l223_223561


namespace smallest_m_value_l223_223043

theorem smallest_m_value : ∃ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ (m + 6) % 9 = 0 ∧ (m - 9) % 6 = 0 ∧ m = 111 := by
  sorry

end smallest_m_value_l223_223043


namespace find_quotient_l223_223205

theorem find_quotient (D d R Q : ℤ) (hD : D = 729) (hd : d = 38) (hR : R = 7)
  (h : D = d * Q + R) : Q = 19 := by
  sorry

end find_quotient_l223_223205


namespace proof_2_in_M_l223_223935

def U : Set ℕ := {1, 2, 3, 4, 5}

def M : Set ℕ := { x | x ∈ U ∧ x ≠ 1 ∧ x ≠ 3 }

theorem proof_2_in_M : 2 ∈ M :=
by
  sorry

end proof_2_in_M_l223_223935


namespace triangle_side_relation_l223_223378

theorem triangle_side_relation (a b c : ℝ) 
    (h_angles : 55 = 55 ∧ 15 = 15 ∧ 110 = 110) :
    c^2 - a^2 = a * b :=
  sorry

end triangle_side_relation_l223_223378


namespace farmhands_work_hours_l223_223197

def apples_per_pint (variety: String) : ℕ :=
  match variety with
  | "golden_delicious" => 20
  | "pink_lady" => 40
  | _ => 0

def total_apples_for_pints (pints: ℕ) : ℕ :=
  (apples_per_pint "golden_delicious") * pints + (apples_per_pint "pink_lady") * pints

def apples_picked_per_hour_per_farmhand : ℕ := 240

def num_farmhands : ℕ := 6

def total_apples_picked_per_hour : ℕ :=
  num_farmhands * apples_picked_per_hour_per_farmhand

def ratio_golden_to_pink : ℕ × ℕ := (1, 2)

def haley_cider_pints : ℕ := 120

def hours_worked (pints: ℕ) (picked_per_hour: ℕ): ℕ :=
  (total_apples_for_pints pints) / picked_per_hour

theorem farmhands_work_hours :
  hours_worked haley_cider_pints total_apples_picked_per_hour = 5 := by
  sorry

end farmhands_work_hours_l223_223197


namespace additional_cost_per_kg_l223_223867

theorem additional_cost_per_kg (l m : ℝ) 
  (h1 : 168 = 30 * l + 3 * m) 
  (h2 : 186 = 30 * l + 6 * m) 
  (h3 : 20 * l = 100) : 
  m = 6 := 
by
  sorry

end additional_cost_per_kg_l223_223867


namespace distance_from_circumcenter_to_orthocenter_l223_223207

variables {A B C A1 H O : Type}

-- Condition Definitions
variable (acute_triangle : Prop)
variable (is_altitude : Prop)
variable (is_orthocenter : Prop)
variable (AH_dist : ℝ := 3)
variable (A1H_dist : ℝ := 2)
variable (circum_radius : ℝ := 4)

-- Prove the distance from O to H
theorem distance_from_circumcenter_to_orthocenter
  (h1 : acute_triangle)
  (h2 : is_altitude)
  (h3 : is_orthocenter)
  (h4 : AH_dist = 3)
  (h5 : A1H_dist = 2)
  (h6 : circum_radius = 4) : 
  ∃ (d : ℝ), d = 2 := 
sorry

end distance_from_circumcenter_to_orthocenter_l223_223207


namespace width_of_room_l223_223643

-- Definitions from conditions
def length : ℝ := 8
def total_cost : ℝ := 34200
def cost_per_sqm : ℝ := 900

-- Theorem stating the width of the room
theorem width_of_room : (total_cost / cost_per_sqm) / length = 4.75 := by 
  sorry

end width_of_room_l223_223643


namespace sum_of_digits_of_repeating_decimal_l223_223245

theorem sum_of_digits_of_repeating_decimal : 
  ∃ (c d : ℕ), 0 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9 ∧ (5/13 : ℚ) = 0 + Nat.digits 10 (c + d*10).sum.to_rnor ≠ 0 → c + d = 11 := 
by sorry

end sum_of_digits_of_repeating_decimal_l223_223245


namespace jony_stop_block_correct_l223_223215

-- Jony's walk parameters
def start_time : ℕ := 7 -- In hours, but it is not used directly
def start_block : ℕ := 10
def end_block : ℕ := 90
def stop_time : ℕ := 40 -- Jony stops walking after 40 minutes starting from 07:00
def speed : ℕ := 100 -- meters per minute
def block_length : ℕ := 40 -- meters

-- Function to calculate the stop block given the parameters
def stop_block (start_block end_block stop_time speed block_length : ℕ) : ℕ :=
  let total_distance := stop_time * speed
  let outbound_distance := (end_block - start_block) * block_length
  let remaining_distance := total_distance - outbound_distance
  let blocks_walked_back := remaining_distance / block_length
  end_block - blocks_walked_back

-- The statement to prove
theorem jony_stop_block_correct :
  stop_block start_block end_block stop_time speed block_length = 70 :=
by
  sorry

end jony_stop_block_correct_l223_223215


namespace zeta_1_8_add_zeta_2_8_add_zeta_3_8_l223_223275

noncomputable def compute_s8 (s : ℕ → ℂ) : ℂ :=
  s 8

theorem zeta_1_8_add_zeta_2_8_add_zeta_3_8 {ζ : ℕ → ℂ} 
  (h1 : ζ 1 + ζ 2 + ζ 3 = 2)
  (h2 : ζ 1^2 + ζ 2^2 + ζ 3^2 = 6)
  (h3 : ζ 1^3 + ζ 2^3 + ζ 3^3 = 18)
  (rec : ∀ n, ζ (n + 3) = 2 * ζ (n + 2) + ζ (n + 1) - (4 / 3) * ζ n)
  (s0 : ζ 0 = 3)
  (s1 : ζ 1 = 2)
  (s2 : ζ 2 = 6)
  (s3 : ζ 3 = 18)
  : ζ 8 = compute_s8 ζ := 
sorry

end zeta_1_8_add_zeta_2_8_add_zeta_3_8_l223_223275


namespace walk_usual_time_l223_223836

theorem walk_usual_time (T : ℝ) (S : ℝ) (h1 : (5 / 4 : ℝ) = (T + 10) / T) : T = 40 :=
sorry

end walk_usual_time_l223_223836


namespace stocking_stuffers_total_l223_223944

theorem stocking_stuffers_total 
  (candy_canes_per_child beanie_babies_per_child books_per_child : ℕ)
  (num_children : ℕ)
  (h1 : candy_canes_per_child = 4)
  (h2 : beanie_babies_per_child = 2)
  (h3 : books_per_child = 1)
  (h4 : num_children = 3) :
  candy_canes_per_child + beanie_babies_per_child + books_per_child * num_children = 21 :=
by
  sorry

end stocking_stuffers_total_l223_223944


namespace china_math_olympiad_34_2023_l223_223369

-- Defining the problem conditions and verifying the minimum and maximum values of S.
theorem china_math_olympiad_34_2023 {a b c d e : ℝ}
  (h1 : a ≥ -1)
  (h2 : b ≥ -1)
  (h3 : c ≥ -1)
  (h4 : d ≥ -1)
  (h5 : e ≥ -1)
  (h6 : a + b + c + d + e = 5) :
  (-512 ≤ (a + b) * (b + c) * (c + d) * (d + e) * (e + a)) ∧
  ((a + b) * (b + c) * (c + d) * (d + e) * (e + a) ≤ 288) :=
sorry

end china_math_olympiad_34_2023_l223_223369


namespace age_problem_l223_223424

theorem age_problem (S Sh K : ℕ) 
  (h1 : S / Sh = 4 / 3)
  (h2 : S / K = 4 / 2)
  (h3 : K + 10 = S)
  (h4 : S + 8 = 30) :
  S = 22 ∧ Sh = 17 ∧ K = 10 := 
sorry

end age_problem_l223_223424


namespace opposite_of_five_l223_223986

theorem opposite_of_five : ∃ y : ℤ, 5 + y = 0 ∧ y = -5 := by
  use -5
  constructor
  . exact rfl
  . sorry

end opposite_of_five_l223_223986


namespace basketball_team_selection_l223_223278

noncomputable def count_ways_excluding_twins (n k : ℕ) : ℕ :=
  let total_ways := Nat.choose n k
  let exhaustive_cases := Nat.choose (n - 2) (k - 2)
  total_ways - exhaustive_cases

theorem basketball_team_selection :
  count_ways_excluding_twins 12 5 = 672 :=
by
  sorry

end basketball_team_selection_l223_223278


namespace first_bag_brown_mms_l223_223781

theorem first_bag_brown_mms :
  ∀ (x : ℕ),
  (12 + 8 + 8 + 3 + x) / 5 = 8 → x = 9 :=
by
  intros x h
  sorry

end first_bag_brown_mms_l223_223781


namespace common_root_conds_l223_223233

theorem common_root_conds (α a b c d : ℝ) (h₁ : a ≠ c)
  (h₂ : α^2 + a * α + b = 0)
  (h₃ : α^2 + c * α + d = 0) :
  α = (d - b) / (a - c) :=
by 
  sorry

end common_root_conds_l223_223233


namespace sum_of_squares_of_coeffs_l223_223875

   theorem sum_of_squares_of_coeffs :
     let expr := 3 * (X^3 - 4 * X^2 + X) - 5 * (X^3 + 2 * X^2 - 5 * X + 3)
     let simplified_expr := -2 * X^3 - 22 * X^2 + 28 * X - 15
     let coefficients := [-2, -22, 28, -15]
     (coefficients.map (λ a => a^2)).sum = 1497 := 
   by 
     -- expending, simplifying and summing up the coefficients 
     sorry
   
end sum_of_squares_of_coeffs_l223_223875


namespace average_of_r_s_t_l223_223199

theorem average_of_r_s_t (r s t : ℝ) (h : (5/4) * (r + s + t) = 20) : (r + s + t) / 3 = 16 / 3 :=
by
  sorry

end average_of_r_s_t_l223_223199


namespace sandy_initial_payment_l223_223374

theorem sandy_initial_payment (P : ℝ) (H1 : P + 300 < P + 1320)
  (H2 : 1320 = 1.10 * (P + 300)) : P = 900 :=
sorry

end sandy_initial_payment_l223_223374


namespace intersection_M_N_l223_223750

def M : Set ℝ := { x | x^2 - x - 2 = 0 }
def N : Set ℝ := { -1, 0 }

theorem intersection_M_N : M ∩ N = {-1} :=
by
  sorry

end intersection_M_N_l223_223750


namespace sin_double_angle_condition_l223_223317

theorem sin_double_angle_condition (θ : ℝ) (h : Real.sin θ + Real.cos θ = 1 / 3) : Real.sin (2 * θ) = -8 / 9 := 
sorry

end sin_double_angle_condition_l223_223317


namespace correct_statement_l223_223907

open Set

variable (U : Set ℕ) (M : Set ℕ)
variables (hU : U = {1, 2, 3, 4, 5}) (hM : U \ M = {1, 3})

theorem correct_statement : 2 ∈ M :=
by
  sorry

end correct_statement_l223_223907


namespace recurring_decimal_addition_l223_223433

noncomputable def recurring_decimal_sum : ℚ :=
  (23 / 99) + (14 / 999) + (6 / 9999)

theorem recurring_decimal_addition :
  recurring_decimal_sum = 2469 / 9999 :=
sorry

end recurring_decimal_addition_l223_223433


namespace robot_path_length_l223_223159

/--
A robot moves in the plane in a straight line, but every one meter it turns 90° to the right or to the left. At some point it reaches its starting point without having visited any other point more than once, and stops immediately. Prove that the possible path lengths of the robot are 4k for some integer k with k >= 3.
-/
theorem robot_path_length (n : ℕ) (h : n > 0) (Movement : n % 4 = 0) :
  ∃ k : ℕ, n = 4 * k ∧ k ≥ 3 :=
sorry

end robot_path_length_l223_223159


namespace simplify_and_evaluate_l223_223785

theorem simplify_and_evaluate (x : ℤ) (h : x = 2) :
  (2 * x + 1) ^ 2 - (x + 3) * (x - 3) = 30 :=
by
  rw [h]
  sorry

end simplify_and_evaluate_l223_223785


namespace choose_3_computers_l223_223441

theorem choose_3_computers (A B : Type) (a : A) (b : B) :
  (∃ C : set (A ⊕ B), 
  (card C = 3) ∧ 
  (∃ a_count b_count : ℕ, 
  (a_count + b_count = 3) ∧ (0 < a_count) ∧ (0 < b_count) ∧
  (a_count = 2 ∨ a_count = 1) ∧ (b_count = 2 ∨ b_count = 1))) → 18 := sorry

end choose_3_computers_l223_223441


namespace least_element_of_S_is_4_l223_223047

theorem least_element_of_S_is_4 :
  ∃ S : Finset ℕ, S.card = 7 ∧ (S ⊆ Finset.range 16) ∧
  (∀ {a b : ℕ}, a ∈ S → b ∈ S → a < b → ¬ (b % a = 0)) ∧
  (∀ T : Finset ℕ, T.card = 7 → (T ⊆ Finset.range 16) →
  (∀ {a b : ℕ}, a ∈ T → b ∈ T → a < b → ¬ (b % a = 0)) →
  ∃ x : ℕ, x ∈ T ∧ x = 4) :=
by
  sorry

end least_element_of_S_is_4_l223_223047


namespace find_xyz_sum_cube_l223_223464

variable (x y z c d : ℝ) 

theorem find_xyz_sum_cube (h1 : x * y * z = c) (h2 : 1 / x^3 + 1 / y^3 + 1 / z^3 = d) :
  (x + y + z)^3 = d * c^3 + 3 * c - 3 * c * d := 
by
  sorry

end find_xyz_sum_cube_l223_223464


namespace product_of_differences_of_squares_is_diff_of_square_l223_223046

-- Define when an integer is a difference of squares of positive integers
def diff_of_squares (n : ℕ) : Prop :=
  ∃ x y : ℕ, 0 < x ∧ 0 < y ∧ n = x^2 - y^2

-- State the main theorem
theorem product_of_differences_of_squares_is_diff_of_square 
  (a b c d : ℕ) (h₁ : diff_of_squares a) (h₂ : diff_of_squares b) (h₃ : diff_of_squares c) (h₄ : diff_of_squares d) : 
  diff_of_squares (a * b * c * d) := by
  sorry

end product_of_differences_of_squares_is_diff_of_square_l223_223046


namespace product_of_terms_l223_223473

variable (a : ℕ → ℝ)

-- Conditions: the sequence is geometric, a_1 = 1, a_10 = 3.
axiom geometric_sequence : ∀ n m : ℕ, a n * a m = a 1 * a (n + m - 1)

axiom a_1_eq_one : a 1 = 1
axiom a_10_eq_three : a 10 = 3

-- We need to prove that the product a_2a_3a_4a_5a_6a_7a_8a_9 = 81.
theorem product_of_terms : a 2 * a 3 * a 4 * a 5 * a 6 * a 7 * a 8 * a 9 = 81 := by
  sorry

end product_of_terms_l223_223473


namespace matrix_addition_correct_l223_223874

def matrixA : Matrix (Fin 2) (Fin 2) ℤ := ![![4, -3], ![0, 5]]
def matrixB : Matrix (Fin 2) (Fin 2) ℤ := ![![-6, 2], ![7, -10]]
def matrixC : Matrix (Fin 2) (Fin 2) ℤ := ![![-2, -1], ![7, -5]]

theorem matrix_addition_correct : matrixA + matrixB = matrixC := by
  sorry

end matrix_addition_correct_l223_223874


namespace speed_difference_is_zero_l223_223672

theorem speed_difference_is_zero :
  let distance_bike := 72
  let time_bike := 9
  let distance_truck := 72
  let time_truck := 9
  let speed_bike := distance_bike / time_bike
  let speed_truck := distance_truck / time_truck
  (speed_truck - speed_bike) = 0 := by
  sorry

end speed_difference_is_zero_l223_223672


namespace intersect_setA_setB_l223_223195

def setA : Set ℝ := {x | x < 2}
def setB : Set ℝ := {x | 3 - 2 * x > 0}

theorem intersect_setA_setB :
  setA ∩ setB = {x | x < 3 / 2} :=
by
  -- proof goes here
  sorry

end intersect_setA_setB_l223_223195


namespace tim_weekly_earnings_l223_223508

-- Definitions based on the conditions
def tasks_per_day : ℕ := 100
def pay_per_task : ℝ := 1.2
def days_per_week : ℕ := 6

-- The theorem that we need to prove
theorem tim_weekly_earnings :
  (tasks_per_day * pay_per_task * days_per_week : ℝ) = 720 :=
by
  sorry -- Skipping the proof

end tim_weekly_earnings_l223_223508


namespace problem1_problem2_problem3_l223_223177

theorem problem1 : 2013^2 - 2012 * 2014 = 1 := 
by 
  sorry

variables (m n : ℤ)

theorem problem2 : ((m-n)^6 / (n-m)^4) * (m-n)^3 = (m-n)^5 :=
by 
  sorry

variables (a b c : ℤ)

theorem problem3 : (a - 2*b + 3*c) * (a - 2*b - 3*c) = a^2 - 4*a*b + 4*b^2 - 9*c^2 :=
by 
  sorry

end problem1_problem2_problem3_l223_223177


namespace find_g_75_l223_223982

variable (g : ℝ → ℝ)

def prop_1 := ∀ x y : ℝ, x > 0 → y > 0 → g (x * y) = g x / y
def prop_2 := g 50 = 30

theorem find_g_75 (h1 : prop_1 g) (h2 : prop_2 g) : g 75 = 20 :=
by
  sorry

end find_g_75_l223_223982


namespace A_empty_iff_A_single_element_iff_and_value_A_at_most_one_element_iff_l223_223590

-- Define the set A
def A (a : ℝ) : Set ℝ := {x | a * x^2 - 3 * x + 2 = 0}

-- Statement for (1)
theorem A_empty_iff (a : ℝ) : A a = ∅ ↔ a ∈ Set.Ioi 0 :=
sorry

-- Statement for (2)
theorem A_single_element_iff_and_value (a : ℝ) : 
  (∃ x, A a = {x}) ↔ (a = 0 ∨ a = 9 / 8) ∧ A a = {2 / 3} :=
sorry

-- Statement for (3)
theorem A_at_most_one_element_iff (a : ℝ) : 
  (∃ x, A a = {x} ∨ A a = ∅) ↔ (a = 0 ∨ a ∈ Set.Ici (9 / 8)) :=
sorry

end A_empty_iff_A_single_element_iff_and_value_A_at_most_one_element_iff_l223_223590


namespace rectangular_prism_edge_sum_l223_223157

theorem rectangular_prism_edge_sum
  (V A : ℝ)
  (hV : V = 8)
  (hA : A = 32)
  (l w h : ℝ)
  (geom_prog : l = w / h ∧ w = l * h ∧ h = l * (w / l)) :
  4 * (l + w + h) = 28 :=
by 
  sorry

end rectangular_prism_edge_sum_l223_223157


namespace angle_in_fourth_quadrant_l223_223010

theorem angle_in_fourth_quadrant (α : ℝ) (h1 : Real.sin (2 * α) < 0) (h2 : Real.sin α - Real.cos α < 0) :
  (π < α ∧ α < 2 * π) ∨ (-0 * π < α ∧ α < 0 * π) := sorry

end angle_in_fourth_quadrant_l223_223010


namespace exponentiation_example_l223_223118

theorem exponentiation_example : (3^2)^4 = 6561 := by
  sorry

end exponentiation_example_l223_223118


namespace executive_board_ways_l223_223775

-- Conditions stated in Lean 4
def total_members : ℕ := 40
def board_size : ℕ := 6

-- Helper functions for combinatorial calculations
noncomputable def choose (n k : ℕ) : ℕ := nat.choose n k

noncomputable def permutations (n : ℕ) : ℕ := nat.fact n
noncomputable def arrangements(n k : ℕ) : ℕ := permutations k / permutations (k - n)

-- Define the problem statement in Lean 4 
theorem executive_board_ways : 
  (choose total_members board_size) * 30 = 115151400 := 
by
  -- Using sorry to skip the proof
  sorry

end executive_board_ways_l223_223775


namespace probability_toner_never_displayed_l223_223510

theorem probability_toner_never_displayed:
  let total_votes := 129
  let toner_votes := 63
  let celery_votes := 66
  (toner_votes + celery_votes = total_votes) →
  let probability := (celery_votes - toner_votes) / (celery_votes + toner_votes)
  probability = 1 / 43 := 
by
  sorry

end probability_toner_never_displayed_l223_223510


namespace samir_climbed_318_stairs_l223_223629

theorem samir_climbed_318_stairs 
  (S : ℕ)
  (h1 : ∀ {V : ℕ}, V = (S / 2) + 18 → S + V = 495) 
  (half_S : ∃ k : ℕ, S = k * 2) -- assumes S is even 
  : S = 318 := 
by
  sorry

end samir_climbed_318_stairs_l223_223629


namespace nat_pow_eq_iff_divides_l223_223071

theorem nat_pow_eq_iff_divides (a b n : ℕ) (h : ∀ k : ℕ, k ≠ b → (b - k) ∣ (a - k^n)) : a = b^n :=
sorry

end nat_pow_eq_iff_divides_l223_223071


namespace right_angled_triangle_solution_l223_223273

-- Define the necessary constants
def t : ℝ := 504 -- area in cm^2
def c : ℝ := 65 -- hypotenuse in cm

-- The definitions of the right-angled triangle's properties
def is_right_angled_triangle (a b : ℝ) : Prop :=
  a ^ 2 + b ^ 2 = c ^ 2 ∧ a * b = 2 * t

-- The proof problem statement
theorem right_angled_triangle_solution :
  ∃ (a b : ℝ), is_right_angled_triangle a b ∧ ((a = 63 ∧ b = 16) ∨ (a = 16 ∧ b = 63)) :=
sorry

end right_angled_triangle_solution_l223_223273


namespace johns_age_in_8_years_l223_223186

theorem johns_age_in_8_years :
  let current_age := 18
  let age_five_years_ago := current_age - 5
  let twice_age_five_years_ago := 2 * age_five_years_ago
  current_age + 8 = twice_age_five_years_ago :=
by
  let current_age := 18
  let age_five_years_ago := current_age - 5
  let twice_age_five_years_ago := 2 * age_five_years_ago
  sorry

end johns_age_in_8_years_l223_223186


namespace sum_of_reciprocals_of_squares_l223_223149

theorem sum_of_reciprocals_of_squares (x y : ℕ) (hxy : x * y = 17) : 
  1 / (x:ℚ)^2 + 1 / (y:ℚ)^2 = 290 / 289 := 
by
  sorry

end sum_of_reciprocals_of_squares_l223_223149


namespace ratio_problem_l223_223596

theorem ratio_problem (a b c d : ℚ) (h1 : a / b = 5 / 4) (h2 : c / d = 4 / 1) (h3 : d / b = 1 / 8) :
  a / c = 5 / 2 := by
  sorry

end ratio_problem_l223_223596


namespace sum_of_roots_eq_l223_223513

theorem sum_of_roots_eq (x : ℝ) : (x - 4)^2 = 16 → (∃ r1 r2 : ℝ, (x - 4) = 4 ∨ (x - 4) = -4 ∧ r1 + r2 = 8) :=
by
  have h := (x - 4) ^ 2 = 16
  sorry  -- You would proceed with the proof here.

end sum_of_roots_eq_l223_223513


namespace exponentiation_rule_example_l223_223096

theorem exponentiation_rule_example : (3^2)^4 = 6561 := 
by {
  -- This proof is based on the power rule (ab)^c = a^(b*c)
  have step1 : (3^2)^4 = 3^(2 * 4), 
  {
    rw pow_mul,
  },
  -- Simplification of 3^(2*4) to 3^8
  have step2 : 3^(2 * 4) = 3^8,
  {
    refl,
  },
  -- Actual computation for 3^8 = 6561
  have step3 : 3^8 = 6561,
  {
    norm_num,
  },
  -- Combining all steps
  rw [step1, step2, step3],
  exact rfl,
}

end exponentiation_rule_example_l223_223096


namespace order_of_t_t2_neg_t_l223_223576

theorem order_of_t_t2_neg_t (t : ℝ) (h : t^2 + t < 0) : t < t^2 ∧ t^2 < -t :=
by
  sorry

end order_of_t_t2_neg_t_l223_223576


namespace find_a_l223_223200

theorem find_a (a x : ℝ) : 
  ((x + a)^2 / (3 * x + 65) = 2) 
  ∧ (∃ x1 x2 : ℝ,  x1 ≠ x2 ∧ (x1 = x2 + 22 ∨ x2 = x1 + 22 )) 
  → a = 3 := 
sorry

end find_a_l223_223200


namespace gingerbread_red_hats_percentage_l223_223698
-- We import the required libraries

-- Define the sets and their cardinalities
def A := {x : Nat | x < 6}
def B := {x : Nat | x < 9}
def A_inter_B := {x : Nat | x < 3}

-- Define the total number of unique gingerbread men
def total_unique := (A ∪ B).card - A_inter_B.card

-- Define the percentage calculation
def percentage_red_hats (total_unique : Nat) : Nat := (A.card * 100) / total_unique

-- The theorem to prove that the percentage of gingerbread men with red hats is 50%
theorem gingerbread_red_hats_percentage : percentage_red_hats total_unique = 50 := by
  sorry

end gingerbread_red_hats_percentage_l223_223698


namespace max_value_expr_max_l223_223050

noncomputable def max_value_expr (x : ℝ) : ℝ :=
  (x^2 + 3 - (x^4 + 9).sqrt) / x

theorem max_value_expr_max (x : ℝ) (hx : 0 < x) :
  max_value_expr x ≤ (6 * (6:ℝ).sqrt) / (6 + 3 * (2:ℝ).sqrt) :=
sorry

end max_value_expr_max_l223_223050


namespace annual_interest_rate_l223_223204

variable (P : ℝ) (t : ℝ)
variable (h1 : t = 25)
variable (h2 : ∀ r : ℝ, P * 2 = P * (1 + r * t))

theorem annual_interest_rate : ∃ r : ℝ, P * 2 = P * (1 + r * t) ∧ r = 0.04 := by
  sorry

end annual_interest_rate_l223_223204


namespace parabola_equation_l223_223879

-- Definitions of the conditions
def parabola_passes_through (x y : ℝ) : Prop :=
  y^2 = -2 * (3 * x)

def focus_on_line (x y : ℝ) : Prop :=
  3 * x - 2 * y - 6 = 0

theorem parabola_equation (x y : ℝ) (hM : x = -6 ∧ y = 6) (hF : ∃ (x y : ℝ), focus_on_line x y) :
  parabola_passes_through x y = (y^2 = -6 * x) :=
by 
  sorry

end parabola_equation_l223_223879


namespace complex_problem_l223_223899

noncomputable def z : ℂ := Complex.exp (2 * Real.pi * Complex.I / 5)

theorem complex_problem :
  (1 - z) * (1 - z^2) * (1 - z^3) * (1 - z^4) = 5 :=
by
  sorry

end complex_problem_l223_223899


namespace area_of_given_region_l223_223841

noncomputable def radius_squared : ℝ := 16 -- Completing the square gives us a radius squared value of 16.
def area_of_circle (r : ℝ) : ℝ := π * r ^ 2

theorem area_of_given_region : area_of_circle (real.sqrt radius_squared) = 16 * π := by
  sorry

end area_of_given_region_l223_223841


namespace wheat_acres_l223_223376

theorem wheat_acres (x y : ℤ) 
  (h1 : x + y = 4500) 
  (h2 : 42 * x + 35 * y = 165200) : 
  y = 3400 :=
sorry

end wheat_acres_l223_223376


namespace domain_of_f_lg_x_l223_223029

theorem domain_of_f_lg_x : 
  ({x : ℝ | -1 ≤ x ∧ x ≤ 1} = {x | 10 ≤ x ∧ x ≤ 100}) ↔ (∃ f : ℝ → ℝ, ∀ x ∈ {x : ℝ | -1 ≤ x ∧ x ≤ 1}, f (x * x + 1) = f (Real.log x)) :=
sorry

end domain_of_f_lg_x_l223_223029


namespace negation_of_universal_proposition_l223_223497

theorem negation_of_universal_proposition :
  ¬ (∀ x : ℝ, 1 < x ∧ x < 2 → x^2 > 1) ↔ ∃ x : ℝ, 1 < x ∧ x < 2 ∧ x^2 ≤ 1 := 
sorry

end negation_of_universal_proposition_l223_223497


namespace transformed_mean_stddev_l223_223452

variables (n : ℕ) (x : Fin n → ℝ)

-- Given conditions
def mean_is_4 (mean : ℝ) : Prop :=
  mean = 4

def stddev_is_7 (stddev : ℝ) : Prop :=
  stddev = 7

-- Definitions for transformations and the results
def transformed_mean (mean : ℝ) : ℝ :=
  3 * mean + 2

def transformed_stddev (stddev : ℝ) : ℝ :=
  3 * stddev

-- The proof problem
theorem transformed_mean_stddev (mean stddev : ℝ) 
  (h_mean : mean_is_4 mean) 
  (h_stddev : stddev_is_7 stddev) :
  transformed_mean mean = 14 ∧ transformed_stddev stddev = 21 :=
by
  rw [h_mean, h_stddev]
  unfold transformed_mean transformed_stddev
  rw [← h_mean, ← h_stddev]
  sorry

end transformed_mean_stddev_l223_223452


namespace min_adults_at_amusement_park_l223_223701

def amusement_park_problem : Prop :=
  ∃ (x y z : ℕ), 
    x + y + z = 100 ∧
    3 * x + 2 * y + (3 / 10) * z = 100 ∧
    (∀ (x' : ℕ), x' < 2 → ¬(∃ (y' z' : ℕ), x' + y' + z' = 100 ∧ 3 * x' + 2 * y' + (3 / 10) * z' = 100))

theorem min_adults_at_amusement_park : amusement_park_problem := sorry

end min_adults_at_amusement_park_l223_223701


namespace sum_of_roots_eq_l223_223514

theorem sum_of_roots_eq (x : ℝ) : (x - 4)^2 = 16 → (∃ r1 r2 : ℝ, (x - 4) = 4 ∨ (x - 4) = -4 ∧ r1 + r2 = 8) :=
by
  have h := (x - 4) ^ 2 = 16
  sorry  -- You would proceed with the proof here.

end sum_of_roots_eq_l223_223514


namespace sin_alpha_given_cos_alpha_plus_pi_over_3_l223_223950

theorem sin_alpha_given_cos_alpha_plus_pi_over_3 
  (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.cos (α + π / 3) = 1 / 5) : 
  Real.sin α = (2 * Real.sqrt 6 - Real.sqrt 3) / 10 := 
by 
  sorry

end sin_alpha_given_cos_alpha_plus_pi_over_3_l223_223950


namespace correct_statement_l223_223904

open Set

variable (U : Set ℕ) (M : Set ℕ)
variables (hU : U = {1, 2, 3, 4, 5}) (hM : U \ M = {1, 3})

theorem correct_statement : 2 ∈ M :=
by
  sorry

end correct_statement_l223_223904


namespace price_of_orange_is_60_l223_223173

-- Given: 
-- 1. The price of each apple is 40 cents.
-- 2. Mary selects 10 pieces of fruit in total.
-- 3. The average price of these 10 pieces is 56 cents.
-- 4. Mary must put back 6 oranges so that the remaining average price is 50 cents.
-- Prove: The price of each orange is 60 cents.

theorem price_of_orange_is_60 (a o : ℕ) (x : ℕ) 
  (h1 : a + o = 10)
  (h2 : 40 * a + x * o = 560)
  (h3 : 40 * a + x * (o - 6) = 200) : 
  x = 60 :=
by
  have eq1 : 40 * a + x * o = 560 := h2
  have eq2 : 40 * a + x * (o - 6) = 200 := h3
  sorry

end price_of_orange_is_60_l223_223173


namespace at_least_one_zero_l223_223450

theorem at_least_one_zero (a b c : ℝ) 
  (h1 : (a + b) * (b + c) * (c + a) = a * b * c)
  (h2 : (a^9 + b^9) * (b^9 + c^9) * (c^9 + a^9) = (a * b * c)^9) : 
  a = 0 ∨ b = 0 ∨ c = 0 := 
sorry

end at_least_one_zero_l223_223450


namespace regular_polygon_sides_l223_223711

theorem regular_polygon_sides (n : ℕ) (h : 180 * (n - 2) / n = 150) : n = 12 := by
  sorry

end regular_polygon_sides_l223_223711


namespace find_c_eq_neg_9_over_4_l223_223495

theorem find_c_eq_neg_9_over_4 (c x : ℚ) (h₁ : 3 * x + 5 = 1) (h₂ : c * x - 8 = -5) :
  c = -9 / 4 :=
sorry

end find_c_eq_neg_9_over_4_l223_223495


namespace alice_meeting_distance_l223_223394

noncomputable def distanceAliceWalks (t : ℝ) : ℝ :=
  6 * t

theorem alice_meeting_distance :
  ∃ t : ℝ, 
    distanceAliceWalks t = 
      (900 * Real.sqrt 2 - Real.sqrt 630000) / 11 ∧
    (5 * t) ^ 2 =
      (6 * t) ^ 2 + 150 ^ 2 - 2 * 6 * t * 150 * Real.cos (Real.pi / 4) :=
sorry

end alice_meeting_distance_l223_223394


namespace power_calc_l223_223125

theorem power_calc : (3^2)^4 = 6561 := 
by
  sorry

end power_calc_l223_223125


namespace shirt_cost_is_43_l223_223482

def pantsCost : ℕ := 140
def tieCost : ℕ := 15
def totalPaid : ℕ := 200
def changeReceived : ℕ := 2

def totalCostWithoutShirt := totalPaid - changeReceived
def totalCostWithPantsAndTie := pantsCost + tieCost
def shirtCost := totalCostWithoutShirt - totalCostWithPantsAndTie

theorem shirt_cost_is_43 : shirtCost = 43 := by
  have h1 : totalCostWithoutShirt = 198 := by rfl
  have h2 : totalCostWithPantsAndTie = 155 := by rfl
  have h3 : shirtCost = totalCostWithoutShirt - totalCostWithPantsAndTie := by rfl
  rw [h1, h2] at h3
  exact h3

end shirt_cost_is_43_l223_223482


namespace num_values_divisible_by_120_l223_223461

theorem num_values_divisible_by_120 (n : ℕ) (h_seq : ∀ n, ∃ k, n = k * (k + 1)) :
  ∃ k, k = 8 := sorry

end num_values_divisible_by_120_l223_223461


namespace proof_2_in_M_l223_223934

def U : Set ℕ := {1, 2, 3, 4, 5}

def M : Set ℕ := { x | x ∈ U ∧ x ≠ 1 ∧ x ≠ 3 }

theorem proof_2_in_M : 2 ∈ M :=
by
  sorry

end proof_2_in_M_l223_223934


namespace final_cost_of_dress_l223_223787

theorem final_cost_of_dress (original_price : ℝ) (discount_percentage : ℝ) 
  (h1 : original_price = 50) (h2 : discount_percentage = 0.30) : 
  let discount := discount_percentage * original_price in
  let final_cost := original_price - discount in
  final_cost = 35 := 
by
  sorry

end final_cost_of_dress_l223_223787


namespace focus_of_parabola_y_eq_x_sq_l223_223067

theorem focus_of_parabola_y_eq_x_sq : ∃ (f : ℝ × ℝ), f = (0, 1/4) ∧ (∃ (p : ℝ), p = 1/2 ∧ ∀ x, y = x^2 → y = 2 * p * (0, y).snd) :=
by
  sorry

end focus_of_parabola_y_eq_x_sq_l223_223067


namespace probability_yellow_second_l223_223552

section MarbleProbabilities

def bag_A := (5, 6)     -- (white marbles, black marbles)
def bag_B := (3, 7)     -- (yellow marbles, blue marbles)
def bag_C := (5, 6)     -- (yellow marbles, blue marbles)

def P_white_A := 5 / 11
def P_black_A := 6 / 11
def P_yellow_given_B := 3 / 10
def P_yellow_given_C := 5 / 11

theorem probability_yellow_second :
  P_white_A * P_yellow_given_B + P_black_A * P_yellow_given_C = 33 / 121 :=
by
  -- Proof would be provided here
  sorry

end MarbleProbabilities

end probability_yellow_second_l223_223552


namespace neg_or_false_of_or_true_l223_223031

variable {p q : Prop}

theorem neg_or_false_of_or_true (h : ¬ (p ∨ q) = false) : p ∨ q :=
by {
  sorry
}

end neg_or_false_of_or_true_l223_223031


namespace hannah_stocking_stuffers_l223_223947

theorem hannah_stocking_stuffers (candy_caness : ℕ) (beanie_babies : ℕ) (books : ℕ) (kids : ℕ) : 
  candy_caness = 4 → 
  beanie_babies = 2 → 
  books = 1 → 
  kids = 3 → 
  candy_caness + beanie_babies + books = 7 → 
  7 * kids = 21 := 
by sorry

end hannah_stocking_stuffers_l223_223947


namespace symmetry_condition_l223_223016

theorem symmetry_condition (a : ℝ) :
  (∀ x : ℝ, |x + 1| + |x - a| = |(2 - x) + 1| + |(2 - x) - a|) ↔ a = 3 :=
by
  sorry

end symmetry_condition_l223_223016


namespace marbles_in_jar_l223_223535

theorem marbles_in_jar (x : ℕ)
  (h1 : \frac{1}{2} * x + \frac{1}{4} * x + 27 + 14 = x) : x = 164 := sorry

end marbles_in_jar_l223_223535


namespace exponentiation_rule_example_l223_223093

theorem exponentiation_rule_example : (3^2)^4 = 6561 := 
by {
  -- This proof is based on the power rule (ab)^c = a^(b*c)
  have step1 : (3^2)^4 = 3^(2 * 4), 
  {
    rw pow_mul,
  },
  -- Simplification of 3^(2*4) to 3^8
  have step2 : 3^(2 * 4) = 3^8,
  {
    refl,
  },
  -- Actual computation for 3^8 = 6561
  have step3 : 3^8 = 6561,
  {
    norm_num,
  },
  -- Combining all steps
  rw [step1, step2, step3],
  exact rfl,
}

end exponentiation_rule_example_l223_223093


namespace sin_square_pi_over_4_l223_223570

theorem sin_square_pi_over_4 (β : ℝ) (h : Real.sin (2 * β) = 2 / 3) : 
  Real.sin (β + π/4) ^ 2 = 5 / 6 :=
by
  sorry

end sin_square_pi_over_4_l223_223570


namespace integer_roots_count_l223_223379

theorem integer_roots_count (b c d e f : ℚ) :
  ∃ (n : ℕ), (n = 0 ∨ n = 1 ∨ n = 2 ∨ n = 4 ∨ n = 5) ∧
  (∃ (r : ℕ → ℤ), ∀ i, i < n → (∀ z : ℤ, (∃ m, z = r m) → (z^5 + b * z^4 + c * z^3 + d * z^2 + e * z + f = 0))) :=
sorry

end integer_roots_count_l223_223379


namespace sequence_formula_minimum_m_l223_223189

variable (a_n : ℕ → ℕ) (S_n : ℕ → ℕ)

/-- The sequence a_n with sum of its first n terms S_n, the first term a_1 = 1, and the terms
   1, a_n, S_n forming an arithmetic sequence, satisfies a_n = 2^(n-1). -/
theorem sequence_formula (h1 : a_n 1 = 1)
    (h2 : ∀ n : ℕ, 1 + n * (a_n n - 1) = S_n n) :
    ∀ n : ℕ, a_n n = 2 ^ (n - 1) := by
  sorry

/-- T_n being the sum of the sequence {n / a_n}, if T_n < (m - 4) / 3 for all n in ℕ*, 
    then the minimum value of m is 16. -/
theorem minimum_m (T_n : ℕ → ℝ) (m : ℕ)
    (hT : ∀ n : ℕ, n > 0 → T_n n < (m - 4) / 3) :
    m ≥ 16 := by
  sorry

end sequence_formula_minimum_m_l223_223189


namespace train_speed_km_per_hr_l223_223854

theorem train_speed_km_per_hr
  (train_length : ℝ) 
  (platform_length : ℝ)
  (time_seconds : ℝ) 
  (h_train_length : train_length = 470) 
  (h_platform_length : platform_length = 520) 
  (h_time_seconds : time_seconds = 64.79481641468682) :
  (train_length + platform_length) / time_seconds * 3.6 = 54.975 := 
sorry

end train_speed_km_per_hr_l223_223854


namespace min_value_of_quadratic_l223_223139

def quadratic_function (x : ℝ) : ℝ :=
  x^2 - 12 * x + 35

theorem min_value_of_quadratic :
  ∀ x : ℝ, quadratic_function x ≥ quadratic_function 6 :=
by sorry

end min_value_of_quadratic_l223_223139


namespace john_total_spending_l223_223042

def t_shirt_price : ℝ := 20
def num_t_shirts : ℝ := 3
def t_shirt_offer_discount : ℝ := 0.50
def t_shirt_total_cost : ℝ := (2 * t_shirt_price) + (t_shirt_price * t_shirt_offer_discount)

def pants_price : ℝ := 50
def num_pants : ℝ := 2
def pants_total_cost : ℝ := pants_price * num_pants

def jacket_original_price : ℝ := 80
def jacket_discount : ℝ := 0.25
def jacket_total_cost : ℝ := jacket_original_price * (1 - jacket_discount)

def hat_price : ℝ := 15

def shoes_original_price : ℝ := 60
def shoes_discount : ℝ := 0.10
def shoes_total_cost : ℝ := shoes_original_price * (1 - shoes_discount)

def clothes_tax_rate : ℝ := 0.05
def shoes_tax_rate : ℝ := 0.08

def clothes_total_cost : ℝ := t_shirt_total_cost + pants_total_cost + jacket_total_cost + hat_price
def total_cost_before_tax : ℝ := clothes_total_cost + shoes_total_cost

def clothes_tax : ℝ := clothes_total_cost * clothes_tax_rate
def shoes_tax : ℝ := shoes_total_cost * shoes_tax_rate

def total_cost_including_tax : ℝ := total_cost_before_tax + clothes_tax + shoes_tax

theorem john_total_spending :
  total_cost_including_tax = 294.57 := by
  sorry

end john_total_spending_l223_223042


namespace number_of_distinct_products_l223_223198

def distinct_products (A : Finset ℕ) : Finset ℕ :=
  (Finset.powerset A).filter (λ s, 2 ≤ s.card).image (λ s, s.prod id)

theorem number_of_distinct_products : 
  distinct_products (Finset.of_list [1, 2, 3, 5, 11]).card = 15 :=
by sorry

end number_of_distinct_products_l223_223198


namespace range_of_a_l223_223953

theorem range_of_a (a : ℝ) :
  (∃ x y : ℝ, x^2 + 4 * (y - a)^2 = 4 ∧ x^2 = 2 * y) ↔ -1 ≤ a ∧ a ≤ 17 / 8 :=
by
  sorry

end range_of_a_l223_223953


namespace express_in_scientific_notation_l223_223559

theorem express_in_scientific_notation : 
  ∃ (a : ℝ) (n : ℤ), 388800 = a * 10 ^ n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ a = 3.888 ∧ n = 5 :=
by
  sorry

end express_in_scientific_notation_l223_223559


namespace smallest_positive_integer_l223_223257

theorem smallest_positive_integer (n : ℕ) : 3 * n ≡ 568 [MOD 34] → n = 18 := 
sorry

end smallest_positive_integer_l223_223257


namespace length_of_PS_l223_223356

theorem length_of_PS
  (PT TR QT TS PQ : ℝ)
  (h1 : PT = 5)
  (h2 : TR = 7)
  (h3 : QT = 9)
  (h4 : TS = 4)
  (h5 : PQ = 7) :
  PS = Real.sqrt 66.33 := 
  sorry

end length_of_PS_l223_223356


namespace parallel_line_through_P_perpendicular_line_through_P_l223_223743

-- Define the line equations
def line1 (x y : ℝ) : Prop := 2 * x + y - 5 = 0
def line2 (x y : ℝ) : Prop := x - 2 * y = 0
def line_l (x y : ℝ) : Prop := 3 * x - y - 7 = 0

-- Define the equations for parallel and perpendicular lines through point P
def parallel_line (x y : ℝ) : Prop := 3 * x - y - 5 = 0
def perpendicular_line (x y : ℝ) : Prop := x + 3 * y - 5 = 0

-- Define the point P where the lines intersect
def point_P : (ℝ × ℝ) := (2, 1)

-- Assert the proof statements
theorem parallel_line_through_P : parallel_line point_P.1 point_P.2 :=
by 
  -- proof content skipped with sorry
  sorry
  
theorem perpendicular_line_through_P : perpendicular_line point_P.1 point_P.2 :=
by 
  -- proof content skipped with sorry
  sorry

end parallel_line_through_P_perpendicular_line_through_P_l223_223743


namespace toms_balloons_l223_223831

-- Define the original number of balloons that Tom had
def original_balloons : ℕ := 30

-- Define the number of balloons that Tom gave to Fred
def balloons_given_to_Fred : ℕ := 16

-- Define the number of balloons that Tom has now
def balloons_left : ℕ := original_balloons - balloons_given_to_Fred

-- The theorem to prove
theorem toms_balloons : balloons_left = 14 := 
by
  -- The proof steps would go here
  sorry

end toms_balloons_l223_223831


namespace problem_solution_l223_223928

universe u

variable (U : Set Nat) (M : Set Nat)
variable (complement_U_M : Set Nat)

axiom U_def : U = {1, 2, 3, 4, 5}
axiom complement_U_M_def : complement_U_M = {1, 3}
axiom M_def : M = U \ complement_U_M

theorem problem_solution : 2 ∈ M := by
  sorry

end problem_solution_l223_223928


namespace marble_count_l223_223532

theorem marble_count (x : ℝ) (h1 : 0.5 * x = x / 2) (h2 : 0.25 * x = x / 4) (h3 : (27 + 14) = 41) (h4 : 0.25 * x = 27 + 14)
  : x = 164 :=
by
  sorry

end marble_count_l223_223532


namespace subset_of_intervals_l223_223196

def A (x : ℝ) := -2 ≤ x ∧ x ≤ 5
def B (m x : ℝ) := m + 1 ≤ x ∧ x ≤ 2 * m - 1
def is_subset_of (B A : ℝ → Prop) := ∀ x, B x → A x
def possible_values_m (m : ℝ) := m ≤ 3

theorem subset_of_intervals (m : ℝ) :
  is_subset_of (B m) A ↔ possible_values_m m := by
  sorry

end subset_of_intervals_l223_223196


namespace triangle_inequality_l223_223896

theorem triangle_inequality 
  (a b c : ℝ) (A B C : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : A + B + C = π) 
  (h5 : a^2 = b^2 + c^2 - 2 * b * c * Real.cos A)
  (h6 : b^2 = a^2 + c^2 - 2 * a * c * Real.cos B)
  (h7 : c^2 = a^2 + b^2 - 2 * a * b * Real.cos C) :
  3 / 2 ≤ a^2 / (b^2 + c^2) + b^2 / (c^2 + a^2) + c^2 / (a^2 + b^2) ∧
  (a^2 / (b^2 + c^2) + b^2 / (c^2 + a^2) + c^2 / (a^2 + b^2) ≤ 
     2 * ((Real.cos A)^2 + (Real.cos B)^2 + (Real.cos C)^2)) :=
sorry

end triangle_inequality_l223_223896


namespace maryville_population_increase_l223_223420

def average_people_added_per_year (P2000 P2005 : ℕ) (period : ℕ) : ℕ :=
  (P2005 - P2000) / period
  
theorem maryville_population_increase :
  let P2000 := 450000
  let P2005 := 467000
  let period := 5
  average_people_added_per_year P2000 P2005 period = 3400 :=
by
  sorry

end maryville_population_increase_l223_223420


namespace product_bc_l223_223252

theorem product_bc (b c : ℤ)
    (h1 : ∀ s : ℤ, s^2 = 2 * s + 1 → s^6 - b * s - c = 0) :
    b * c = 2030 :=
sorry

end product_bc_l223_223252


namespace product_of_three_integers_sum_l223_223817
-- Import necessary libraries

-- Define the necessary conditions and the goal
theorem product_of_three_integers_sum (a b c : ℕ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c)
(h4 : a * b * c = 11^3) : a + b + c = 133 :=
sorry

end product_of_three_integers_sum_l223_223817


namespace correct_statement_l223_223908

open Set

variable (U : Set ℕ) (M : Set ℕ)
variables (hU : U = {1, 2, 3, 4, 5}) (hM : U \ M = {1, 3})

theorem correct_statement : 2 ∈ M :=
by
  sorry

end correct_statement_l223_223908


namespace mary_score_is_95_l223_223961

theorem mary_score_is_95
  (s c w : ℕ)
  (h1 : s > 90)
  (h2 : s = 35 + 5 * c - w)
  (h3 : c + w = 30)
  (h4 : ∀ c' w', s = 35 + 5 * c' - w' → c + w = c' + w' → (c', w') = (c, w)) :
  s = 95 :=
by
  sorry

end mary_score_is_95_l223_223961


namespace exponentiation_identity_l223_223111

theorem exponentiation_identity (a m n : ℕ) : (a^m)^n = a^(m * n) :=
by sorry

example : (3^2)^4 = 6561 :=
by
  have h := exponentiation_identity 3 2 4
  rw [←h]
  norm_num

end exponentiation_identity_l223_223111


namespace geom_seq_a_sum_first_n_terms_l223_223188

noncomputable def a (n : ℕ) : ℕ := 2^(n + 1)

def b (n : ℕ) : ℕ := 3 * (n + 1) - 2

def a_b_product (n : ℕ) : ℕ := (3 * (n + 1) - 2) * 2^(n + 1)

def S (n : ℕ) : ℕ :=
  (Finset.range n).sum (λ k => a_b_product k)

theorem geom_seq_a (n : ℕ) : a (n + 1) = 2 * a n :=
by sorry

theorem sum_first_n_terms (n : ℕ) : S n = 10 + (3 * n - 5) * 2^(n + 1) :=
by sorry

end geom_seq_a_sum_first_n_terms_l223_223188


namespace Q_has_negative_root_l223_223299

def Q (x : ℝ) : ℝ := x^7 + 2 * x^5 + 5 * x^3 - x + 12

theorem Q_has_negative_root : ∃ x : ℝ, x < 0 ∧ Q x = 0 :=
by
  sorry

end Q_has_negative_root_l223_223299


namespace son_age_l223_223682

theorem son_age:
  ∃ S M : ℕ, 
  (M = S + 20) ∧ 
  (M + 2 = 2 * (S + 2)) ∧ 
  (S = 18) := 
by
  sorry

end son_age_l223_223682


namespace positive_integral_solution_exists_l223_223434

theorem positive_integral_solution_exists :
  ∃ n : ℕ, n > 0 ∧
  ( (n * (n + 1) * (2 * n + 1)) * 100 = 27 * 6 * (n * (n + 1))^2 ) ∧ n = 5 :=
by {
  sorry
}

end positive_integral_solution_exists_l223_223434


namespace initial_stock_of_coffee_l223_223680

theorem initial_stock_of_coffee (x : ℝ) (h : x ≥ 0) 
  (h1 : 0.30 * x + 60 = 0.36 * (x + 100)) : x = 400 :=
by sorry

end initial_stock_of_coffee_l223_223680


namespace inequality_proof_l223_223892

theorem inequality_proof (x y z : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : 0 < z) (h₄ : x * y * z ≥ 1) : 
  (x^5 - x^2) / (x^5 + y^2 + z^2) + (y^5 - y^2) / (y^5 + z^2 + x^2) + (z^5 - z^2) / (z^5 + x^2 + y^2) ≥ 0 :=
by
  sorry

end inequality_proof_l223_223892


namespace toy_spending_ratio_l223_223360

theorem toy_spending_ratio :
  ∃ T : ℝ, 204 - T > 0 ∧ 51 = (204 - T) / 2 ∧ (T / 204) = 1 / 2 :=
by
  sorry

end toy_spending_ratio_l223_223360


namespace P_started_following_J_l223_223550

theorem P_started_following_J :
  ∀ (t : ℝ),
    (6 * 7.3 + 3 = 8 * (7.3 - t)) → t = 1.45 → t + 12 = 13.45 :=
by
  sorry

end P_started_following_J_l223_223550


namespace not_prime_41_squared_plus_41_plus_41_l223_223959

def is_prime (n : ℕ) : Prop := ∀ m k : ℕ, m * k = n → m = 1 ∨ k = 1

theorem not_prime_41_squared_plus_41_plus_41 :
  ¬ is_prime (41^2 + 41 + 41) :=
by {
  sorry
}

end not_prime_41_squared_plus_41_plus_41_l223_223959


namespace delta_k_f_l223_223738

open Nat

-- Define the function
def f (n : ℕ) : ℕ := 3^n

-- Define the discrete difference operator
def Δ (g : ℕ → ℕ) (n : ℕ) : ℕ := g (n + 1) - g n

-- Define the k-th discrete difference
def Δk (g : ℕ → ℕ) (k : ℕ) (n : ℕ) : ℕ :=
  if k = 0 then g n else Δk (Δ g) (k - 1) n

-- State the theorem
theorem delta_k_f (k : ℕ) (n : ℕ) (h : k ≥ 1) : Δk f k n = 2^k * 3^n := by
  sorry

end delta_k_f_l223_223738


namespace average_words_per_page_l223_223710

theorem average_words_per_page
  (sheets_to_pages : ℕ := 16)
  (total_sheets : ℕ := 12)
  (total_word_count : ℕ := 240000) :
  (total_word_count / (total_sheets * sheets_to_pages)) = 1250 :=
by
  sorry

end average_words_per_page_l223_223710


namespace isosceles_triangle_base_length_l223_223807

-- Definitions based on the conditions
def congruent_side : Nat := 7
def perimeter : Nat := 23

-- Statement to prove
theorem isosceles_triangle_base_length :
  let b := perimeter - 2 * congruent_side in b = 9 :=
by
  sorry

end isosceles_triangle_base_length_l223_223807


namespace number_is_eight_l223_223342

theorem number_is_eight (x : ℤ) (h : x - 2 = 6) : x = 8 := 
sorry

end number_is_eight_l223_223342


namespace power_calc_l223_223123

theorem power_calc : (3^2)^4 = 6561 := 
by
  sorry

end power_calc_l223_223123


namespace B_share_is_102_l223_223058

variables (A B C : ℝ)
variables (total : ℝ)
variables (rA_B : ℝ) (rB_C : ℝ)

-- Conditions
def conditions : Prop :=
  (total = 578) ∧
  (rA_B = 2 / 3) ∧
  (rB_C = 1 / 4) ∧
  (A = rA_B * B) ∧
  (B = rB_C * C) ∧
  (A + B + C = total)

-- Theorem to prove B's share
theorem B_share_is_102 (h : conditions A B C total rA_B rB_C) : B = 102 :=
by sorry

end B_share_is_102_l223_223058


namespace probability_distribution_l223_223960

noncomputable def mean_variance_binomial (n : ℕ) (p : ℝ) : ℝ × ℝ :=
  (n * p, n * p * (1 - p))

theorem probability_distribution (P : ℕ → ℝ) :
  (∀ n : ℕ, n > 2 → P n = 1/3 * P (n - 1) + 2/3 * P (n - 2)) →
  P 1 = 1/3 →
  P 2 = 7/9 →
  ∀ n : ℕ, n ≥ 1 →
  P n = 3/5 - 4/15 * (-2/3)^(n-1) :=
sorry

example : mean_variance_binomial 5 (2/3) = (10/3, 10/9) :=
begin
  simp [mean_variance_binomial],
  norm_num,
end

end probability_distribution_l223_223960


namespace problem_statement_l223_223340

def h (x : ℝ) : ℝ := 3 * x + 2
def k (x : ℝ) : ℝ := 2 * x - 3

theorem problem_statement : (h (k (h 3))) / (k (h (k 3))) = 59 / 19 := by
  sorry

end problem_statement_l223_223340


namespace additional_savings_correct_l223_223054

def initial_order_amount : ℝ := 10000

def option1_discount1 : ℝ := 0.20
def option1_discount2 : ℝ := 0.20
def option1_discount3 : ℝ := 0.10
def option2_discount1 : ℝ := 0.40
def option2_discount2 : ℝ := 0.05
def option2_discount3 : ℝ := 0.05

def final_price_option1 : ℝ :=
  initial_order_amount * (1 - option1_discount1) *
  (1 - option1_discount2) *
  (1 - option1_discount3)

def final_price_option2 : ℝ :=
  initial_order_amount * (1 - option2_discount1) *
  (1 - option2_discount2) *
  (1 - option2_discount3)

def additional_savings : ℝ :=
  final_price_option1 - final_price_option2

theorem additional_savings_correct : additional_savings = 345 :=
by
  sorry

end additional_savings_correct_l223_223054


namespace fraction_result_l223_223975

theorem fraction_result (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : (2 * x + 3 * y) / (x - 2 * y) = 3) : 
  (x + 2 * y) / (2 * x - y) = 11 / 17 :=
sorry

end fraction_result_l223_223975


namespace isosceles_triangle_base_length_l223_223806

-- Definitions based on the conditions
def congruent_side : Nat := 7
def perimeter : Nat := 23

-- Statement to prove
theorem isosceles_triangle_base_length :
  let b := perimeter - 2 * congruent_side in b = 9 :=
by
  sorry

end isosceles_triangle_base_length_l223_223806


namespace jordan_more_novels_than_maxime_l223_223300

def jordan_french_novels : ℕ := 130
def jordan_spanish_novels : ℕ := 20

def alexandre_french_novels : ℕ := jordan_french_novels / 10
def alexandre_spanish_novels : ℕ := 3 * jordan_spanish_novels

def camille_french_novels : ℕ := 2 * alexandre_french_novels
def camille_spanish_novels : ℕ := jordan_spanish_novels / 2

def total_french_novels : ℕ := jordan_french_novels + alexandre_french_novels + camille_french_novels

def maxime_french_novels : ℕ := total_french_novels / 2 - 5
def maxime_spanish_novels : ℕ := 2 * camille_spanish_novels

def jordan_total_novels : ℕ := jordan_french_novels + jordan_spanish_novels
def maxime_total_novels : ℕ := maxime_french_novels + maxime_spanish_novels

def novels_difference : ℕ := jordan_total_novels - maxime_total_novels

theorem jordan_more_novels_than_maxime : novels_difference = 51 :=
sorry

end jordan_more_novels_than_maxime_l223_223300


namespace ratio_of_new_time_to_previous_time_l223_223419

-- Given conditions
def distance : ℕ := 288
def initial_time : ℕ := 6
def new_speed : ℕ := 32

-- Question: Prove the ratio of the new time to the previous time is 3:2
theorem ratio_of_new_time_to_previous_time :
  (distance / new_speed) / initial_time = 3 / 2 :=
by
  sorry

end ratio_of_new_time_to_previous_time_l223_223419


namespace instantaneous_rate_of_change_at_0_l223_223298

noncomputable def f (x : ℝ) : ℝ := 2 * Real.exp (Real.sin x)

theorem instantaneous_rate_of_change_at_0 : (deriv f 0) = 2 :=
  by
  sorry

end instantaneous_rate_of_change_at_0_l223_223298


namespace find_e_l223_223052

noncomputable def quadratic (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem find_e (r s d e : ℝ) 
  (h1 : quadratic 2 (-4) (-6) r = 0)
  (h2 : quadratic 2 (-4) (-6) s = 0)
  (h3 : r + s = 2) 
  (h4 : r * s = -3)
  (h5 : d = -(r + s - 6))
  (h6 : e = (r - 3) * (s - 3)) : 
  e = 0 :=
sorry

end find_e_l223_223052


namespace check_numbers_has_property_P_l223_223620

def has_property_P (n : ℤ) : Prop :=
  ∃ x y z : ℤ, n = x^3 + y^3 + z^3 - 3 * x * y * z

theorem check_numbers_has_property_P :
  has_property_P 1 ∧ has_property_P 5 ∧ has_property_P 2014 ∧ ¬has_property_P 2013 :=
by
  sorry

end check_numbers_has_property_P_l223_223620


namespace length_of_goods_train_l223_223409

theorem length_of_goods_train (speed_kmph : ℝ) (platform_length : ℝ) (crossing_time : ℝ) (length_of_train : ℝ) :
  speed_kmph = 96 → platform_length = 360 → crossing_time = 32 → length_of_train = (26.67 * 32 - 360) :=
by
  sorry

end length_of_goods_train_l223_223409


namespace fraction_sum_l223_223498

theorem fraction_sum (y : ℝ) (a b : ℤ) (h : y = 3.834834834) (h_frac : y = (a : ℝ) / b) (h_coprime : Int.gcd a b = 1) : a + b = 4830 :=
sorry

end fraction_sum_l223_223498


namespace main_theorem_l223_223283

def f (m: ℕ) : ℕ := m * (m + 1) / 2

lemma f_1 : f 1 = 1 := by 
  -- placeholder for proof
  sorry

lemma f_functional_eq (m n : ℕ) : f m + f n = f (m + n) - m * n := by
  -- placeholder for proof
  sorry

theorem main_theorem (m : ℕ) : f m = m * (m + 1) / 2 := by
  -- Combining the conditions to conclude the result
  sorry

end main_theorem_l223_223283


namespace range_of_independent_variable_l223_223819

theorem range_of_independent_variable (x : ℝ) : (x - 4) ≠ 0 ↔ x ≠ 4 :=
by
  sorry

end range_of_independent_variable_l223_223819


namespace triangle_area_on_ellipse_l223_223649

def onEllipse (p : ℝ × ℝ) : Prop := (p.1)^2 + 4 * (p.2)^2 = 4

def isCentroid (C : ℝ × ℝ) (A B : ℝ × ℝ) : Prop :=
  A ≠ B ∧ B ≠ C ∧ C ≠ A ∧
  C = ((A.1 + B.1) / 3, (A.2 + B.2) / 3)

theorem triangle_area_on_ellipse
  (A B C : ℝ × ℝ)
  (h₁ : A ≠ B)
  (h₂ : B ≠ C)
  (h₃ : C ≠ A)
  (h₄ : onEllipse A)
  (h₅ : onEllipse B)
  (h₆ : onEllipse C)
  (h₇ : isCentroid C A B)
  (h₈ : C = (0, 0))  : 
  1 / 2 * (A.1 - B.1) * (B.2 - A.2) = 1 :=
by
  sorry

end triangle_area_on_ellipse_l223_223649


namespace root_range_of_quadratic_eq_l223_223642

theorem root_range_of_quadratic_eq (k : ℝ) :
  (∃ x1 x2 : ℝ, x1 < x2 ∧ x1^2 + k * x1 - k = 0 ∧ x2^2 + k * x2 - k = 0 ∧ 1 < x1 ∧ x1 < 2 ∧ 2 < x2 ∧ x2 < 3) ↔  (-9 / 2) < k ∧ k < -4 :=
by
  sorry

end root_range_of_quadratic_eq_l223_223642


namespace wire_cut_example_l223_223519

theorem wire_cut_example (total_length piece_ratio : ℝ) (h1 : total_length = 28) (h2 : piece_ratio = 2.00001 / 5) :
  ∃ (shorter_piece : ℝ), shorter_piece + piece_ratio * shorter_piece = total_length ∧ shorter_piece = 20 :=
by
  sorry

end wire_cut_example_l223_223519


namespace total_spider_legs_l223_223666

variable (numSpiders : ℕ)
variable (legsPerSpider : ℕ)
axiom h1 : numSpiders = 5
axiom h2 : legsPerSpider = 8

theorem total_spider_legs : numSpiders * legsPerSpider = 40 :=
by
  -- necessary for build without proof.
  sorry

end total_spider_legs_l223_223666


namespace triangle_circumradius_l223_223859

theorem triangle_circumradius (a b c : ℝ) (h1 : a = 8) (h2 : b = 6) (h3 : c = 10) : 
  ∃ r : ℝ, r = 5 :=
by
  sorry

end triangle_circumradius_l223_223859


namespace area_of_given_region_l223_223840

noncomputable def radius_squared : ℝ := 16 -- Completing the square gives us a radius squared value of 16.
def area_of_circle (r : ℝ) : ℝ := π * r ^ 2

theorem area_of_given_region : area_of_circle (real.sqrt radius_squared) = 16 * π := by
  sorry

end area_of_given_region_l223_223840


namespace sufficient_not_necessary_condition_l223_223274

theorem sufficient_not_necessary_condition (x : ℝ) : (x > 0 → |x| = x) ∧ (|x| = x → x ≥ 0) :=
by
  sorry

end sufficient_not_necessary_condition_l223_223274


namespace num_of_subsets_is_16_l223_223553

open Set
open Finset

noncomputable def numValidSubsets : ℕ :=
  let S := {1, 2, 3, 4, 5, 6} in
  (powerset S.filter (λ x, x ≠ 1 ∧ x ≠ 2)).card

theorem num_of_subsets_is_16 : numValidSubsets = 16 := by
  sorry

end num_of_subsets_is_16_l223_223553


namespace type_C_count_l223_223077

theorem type_C_count (A B C C1 C2 : ℕ) (h1 : A + B + C = 25) (h2 : A + B + C2 = 17) (h3 : B + C2 = 12) (h4 : C2 = 8) (h5: B = 4) (h6: A = 5) : C = 16 :=
by {
  -- Directly use the given hypotheses.
  sorry
}

end type_C_count_l223_223077


namespace time_after_2500_minutes_l223_223518

/-- 
To prove that adding 2500 minutes to midnight on January 1, 2011 results in 
January 2 at 5:40 PM.
-/
theorem time_after_2500_minutes :
  let minutes_in_a_day := 1440 -- 24 hours * 60 minutes
  let minutes_in_an_hour := 60
  let start_time_minutes := 0 -- Midnight January 1, 2011 as zero minutes
  let total_minutes := 2500
  let resulting_minutes := start_time_minutes + total_minutes
  let days_passed := resulting_minutes / minutes_in_a_day
  let remaining_minutes := resulting_minutes % minutes_in_a_day
  let hours := remaining_minutes / minutes_in_an_hour
  let minutes := remaining_minutes % minutes_in_an_hour
  days_passed = 1 ∧ hours = 17 ∧ minutes = 40 :=
by
  -- Proof to be filled in
  sorry

end time_after_2500_minutes_l223_223518


namespace systematic_sampling_employee_l223_223865

theorem systematic_sampling_employee
    (n : ℕ)
    (employees : Finset ℕ)
    (sample : Finset ℕ)
    (h_n_52 : n = 52)
    (h_employees : employees = Finset.range 52)
    (h_sample_size : sample.card = 4)
    (h_systematic_sample : sample ⊆ employees)
    (h_in_sample : {6, 32, 45} ⊆ sample) :
    19 ∈ sample :=
by
  -- conditions 
  have h0 : 6 ∈ sample := Finset.mem_of_subset h_in_sample (by simp)
  have h1 : 32 ∈ sample := Finset.mem_of_subset h_in_sample (by simp)
  have h2 : 45 ∈ sample := Finset.mem_of_subset h_in_sample (by simp)
  have h_arith : 6 + 45 = 32 + 19 :=
    by linarith
  sorry

end systematic_sampling_employee_l223_223865


namespace option_B_can_be_factored_l223_223693

theorem option_B_can_be_factored (a b : ℝ) : 
  (-a^2 + b^2) = (b+a)*(b-a) := 
by
  sorry

end option_B_can_be_factored_l223_223693


namespace length_of_de_l223_223522

theorem length_of_de
  {a b c d e : ℝ} 
  (h1 : b - a = 5) 
  (h2 : c - a = 11) 
  (h3 : e - a = 22) 
  (h4 : c - b = 2 * (d - c)) :
  e - d = 8 :=
by 
  sorry

end length_of_de_l223_223522


namespace estimate_students_l223_223471

noncomputable def mean : ℝ := 90
noncomputable def std_dev : ℝ := σ -- σ > 0
noncomputable def prob_range : ℝ := 0.8
noncomputable def total_students : ℕ := 780
noncomputable def prob_gt_120 : ℝ := (1 - prob_range) / 2
noncomputable def estimated_students_gt_120 : ℕ := (prob_gt_120 * total_students).to_nat

theorem estimate_students {σ : ℝ} (hσ : σ > 0) :
  estimated_students_gt_120 = 78 :=
by 
  sorry

end estimate_students_l223_223471


namespace isosceles_triangle_base_length_l223_223794

-- Define the conditions
def side_length : ℕ := 7
def perimeter : ℕ := 23

-- Define the theorem to prove the length of the base
theorem isosceles_triangle_base_length (b : ℕ) (h : 2 * side_length + b = perimeter) : b = 9 :=
by
  sorry

end isosceles_triangle_base_length_l223_223794


namespace percent_of_x_is_y_l223_223670

variables (x y : ℝ)

theorem percent_of_x_is_y (h : 0.30 * (x - y) = 0.20 * (x + y)) : y = 0.20 * x :=
by sorry

end percent_of_x_is_y_l223_223670


namespace average_speed_v2_l223_223659

theorem average_speed_v2 (v1 : ℝ) (t : ℝ) (S1 : ℝ) (S2 : ℝ) : 
  (v1 = 30) → (t = 30) → (S1 = 800) → (S2 = 200) → 
  (v2 = (v1 - (S1 - S2) / t) ∨ v2 = (v1 + (S1 - S2) / t)) :=
by
  intros h1 h2 h3 h4
  sorry

end average_speed_v2_l223_223659


namespace bullfinches_are_50_l223_223253

theorem bullfinches_are_50 :
  ∃ N : ℕ, (N > 50 ∨ N < 50 ∨ N ≥ 1) ∧ (¬(N > 50) ∨ ¬(N < 50) ∨ ¬(N ≥ 1)) ∧
  (N > 50 ∧ ¬(N < 50) ∨ N < 50 ∧ ¬(N > 50) ∨ N ≥ 1 ∧ (¬(N > 50) ∧ ¬(N < 50))) ∧
  N = 50 :=
by
  sorry

end bullfinches_are_50_l223_223253


namespace cost_of_each_adult_meal_is_8_l223_223871

/- Define the basic parameters and conditions -/
def total_people : ℕ := 11
def kids : ℕ := 2
def total_cost : ℕ := 72
def kids_eat_free (k : ℕ) := k = 0

/- The number of adults is derived from the total people minus kids -/
def num_adults : ℕ := total_people - kids

/- The cost per adult meal can be defined and we need to prove it equals to $8 -/
def cost_per_adult (total_cost : ℕ) (num_adults : ℕ) : ℕ := total_cost / num_adults

/- The statement to prove that the cost per adult meal is $8 -/
theorem cost_of_each_adult_meal_is_8 : cost_per_adult total_cost num_adults = 8 := by
  sorry

end cost_of_each_adult_meal_is_8_l223_223871


namespace kim_average_increase_l223_223170

noncomputable def avg (scores : List ℚ) : ℚ :=
  (scores.sum) / (scores.length)

theorem kim_average_increase :
  let scores_initial := [85, 89, 90, 92]  -- Initial scores
  let score_fifth := 95  -- Fifth score
  let original_average := avg scores_initial
  let new_average := avg (scores_initial ++ [score_fifth])
  new_average - original_average = 1.2 := by
  let scores_initial : List ℚ := [85, 89, 90, 92]
  let score_fifth : ℚ := 95
  let original_average : ℚ := avg scores_initial
  let new_average : ℚ := avg (scores_initial ++ [score_fifth])
  have : new_average - original_average = 1.2 := sorry
  exact this

end kim_average_increase_l223_223170


namespace value_of_abs_sum_l223_223820

noncomputable def cos_squared (θ : ℝ) : ℝ := (Real.cos θ) ^ 2

theorem value_of_abs_sum (θ x : ℝ) (h : Real.log x / Real.log 2 = 3 - 2 * cos_squared θ) :
  |x - 2| + |x - 8| = 6 := by
    sorry

end value_of_abs_sum_l223_223820


namespace boxes_same_number_oranges_l223_223168

theorem boxes_same_number_oranges 
  (total_boxes : ℕ) (min_oranges : ℕ) (max_oranges : ℕ) 
  (boxes : ℕ) (range_oranges : ℕ) :
  total_boxes = 150 →
  min_oranges = 130 →
  max_oranges = 160 →
  range_oranges = max_oranges - min_oranges + 1 →
  boxes = total_boxes / range_oranges →
  31 = range_oranges →
  4 ≤ boxes :=
by sorry

end boxes_same_number_oranges_l223_223168


namespace difference_in_earnings_in_currency_B_l223_223784

-- Definitions based on conditions
def num_red_stamps : Nat := 30
def num_white_stamps : Nat := 80
def price_per_red_stamp_currency_A : Nat := 5
def price_per_white_stamp_currency_B : Nat := 50
def exchange_rate_A_to_B : Nat := 2

-- Theorem based on the question and correct answer
theorem difference_in_earnings_in_currency_B : 
  num_white_stamps * price_per_white_stamp_currency_B - 
  (num_red_stamps * price_per_red_stamp_currency_A * exchange_rate_A_to_B) = 3700 := 
  by
  sorry

end difference_in_earnings_in_currency_B_l223_223784


namespace part1_part2_l223_223455

def f (x : ℝ) : ℝ := abs (2 * x - 1) + abs (x - 3)
noncomputable def M := 3 / 2

theorem part1 (x : ℝ) (m : ℝ) : (∀ x, f x ≥ abs (m + 1)) → m ≤ M := sorry

theorem part2 (a b c : ℝ) : a > 0 → b > 0 → c > 0 → a + b + c = M →  (b^2 / a + c^2 / b + a^2 / c) ≥ M := sorry

end part1_part2_l223_223455


namespace ellipse_has_correct_equation_l223_223003

noncomputable def ellipse_Equation (a b : ℝ) (eccentricity : ℝ) (triangle_perimeter : ℝ) : Prop :=
  let c := a * eccentricity
  (a > b) ∧ (b > 0) ∧ (eccentricity = (Real.sqrt 3) / 3) ∧ (triangle_perimeter = 4 * (Real.sqrt 3)) ∧
  (a = Real.sqrt 3) ∧ (b^2 = a^2 - c^2) ∧
  (c = 1) ∧
  (b = Real.sqrt 2) ∧
  (∀ x y : ℝ, ((x^2 / a^2) + (y^2 / b^2) = 1) ↔ ((x^2 / 3) + (y^2 / 2) = 1))

theorem ellipse_has_correct_equation : ellipse_Equation (Real.sqrt 3) (Real.sqrt 2) ((Real.sqrt 3) / 3) (4 * (Real.sqrt 3)) := 
sorry

end ellipse_has_correct_equation_l223_223003


namespace distance_symmetric_line_eq_l223_223013

noncomputable def distance_from_point_to_line : ℝ :=
  let x0 := 2
  let y0 := -1
  let A := 2
  let B := 3
  let C := 0
  (|A * x0 + B * y0 + C|) / (Real.sqrt (A^2 + B^2))

theorem distance_symmetric_line_eq : distance_from_point_to_line = 1 / (Real.sqrt 13) := by
  sorry

end distance_symmetric_line_eq_l223_223013


namespace cos_B_of_triangle_l223_223211

theorem cos_B_of_triangle (A B : ℝ) (a b : ℝ) (h1 : A = 2 * B) (h2 : a = 6) (h3 : b = 4) :
  Real.cos B = 3 / 4 :=
by
  sorry

end cos_B_of_triangle_l223_223211


namespace pieces_present_l223_223426

def total_pieces : ℕ := 32
def missing_pieces : ℕ := 10

theorem pieces_present : total_pieces - missing_pieces = 22 :=
by {
  sorry
}

end pieces_present_l223_223426


namespace limit_f_l223_223193

open Real

noncomputable def f (x : ℝ) := (5 / 3) * x - log (2 * x + 1)

theorem limit_f (f' : ℝ → ℝ) (h : ∀ x, f' x = (5 / 3) - (2 / (2 * x + 1))) :
  Tendsto (λ Δx, (f (1 + Δx) - f 1) / Δx) (𝓝 0) (𝓝 1) :=
begin
  have f'_def : f' 1 = 1,
  { rw h, simp },
  rw ← f'_def,
  exact has_deriv_at_iff_tendsto_slope.mp (Deriv.deriv f (1 : ℝ)),
  sorry -- Proof required here
end

end limit_f_l223_223193


namespace proof_f_prime_at_2_l223_223980

noncomputable def f_prime (x : ℝ) (f_prime_2 : ℝ) : ℝ :=
  2 * x + 2 * f_prime_2 - (1 / x)

theorem proof_f_prime_at_2 :
  ∃ (f_prime_2 : ℝ), f_prime 2 f_prime_2 = -7 / 2 :=
by
  sorry

end proof_f_prime_at_2_l223_223980


namespace engineers_to_designers_ratio_l223_223979

-- Define the given conditions for the problem
variables (e d : ℕ) -- e is the number of engineers, d is the number of designers
variables (h1 : (48 * e + 60 * d) / (e + d) = 52)

-- Theorem statement: The ratio of the number of engineers to the number of designers is 2:1
theorem engineers_to_designers_ratio (h1 : (48 * e + 60 * d) / (e + d) = 52) : e = 2 * d :=
by {
  sorry  
}

end engineers_to_designers_ratio_l223_223979


namespace find_a_l223_223588

noncomputable def f (x a : ℝ) : ℝ := x + Real.exp (x - a)
noncomputable def g (x a : ℝ) : ℝ := Real.log (x + 2) - 4 * Real.exp (a - x)

theorem find_a {x0 a : ℝ} (h : f x0 a - g x0 a = 3) : a = -Real.log 2 - 1 :=
sorry

end find_a_l223_223588


namespace coefficient_x_squared_l223_223312

theorem coefficient_x_squared (a : ℝ) (x : ℝ) (h : x = 0.5) (eqn : a * x^2 + 9 * x - 5 = 0) : a = 2 :=
by
  sorry

end coefficient_x_squared_l223_223312


namespace prob_rain_next_day_given_today_rain_l223_223752

variable (P_rain : ℝ) (P_rain_2_days : ℝ)
variable (p_given_rain : ℝ)

-- Given conditions
def condition_P_rain : Prop := P_rain = 1/3
def condition_P_rain_2_days : Prop := P_rain_2_days = 1/5

-- The question to prove
theorem prob_rain_next_day_given_today_rain (h1 : condition_P_rain P_rain) (h2 : condition_P_rain_2_days P_rain_2_days) :
  p_given_rain = 3/5 :=
by
  sorry

end prob_rain_next_day_given_today_rain_l223_223752


namespace min_odd_in_A_P_l223_223606

-- Define the polynomial P of degree 8
variable (P : ℝ → ℝ)
-- Assume P is a polynomial of degree 8
axiom degree_P : degree P = 8

-- Define the set A_P as the set of all x for which P(x) gives a certain value
def A_P (c : ℝ) : Set ℝ := { x | P x = c }

-- Given condition
axiom include_8 : 8 ∈ A_P (P 8)

-- Define if a number is odd
def is_odd (n : ℝ) : Prop := ∃k : ℤ, n = 2 * k + 1

-- The minimum number of odd numbers that are in the set A_P,
-- given that the number 8 is included in A_P, is 1

theorem min_odd_in_A_P : ∀ P, (degree P = 8) → 8 ∈ A_P (P 8) → ∃ x ∈ A_P (P 8), is_odd x := 
by
  sorry -- proof goes here

end min_odd_in_A_P_l223_223606


namespace negation_of_p_l223_223194

noncomputable def p : Prop := ∀ x : ℝ, x > 0 → 2 * x^2 + 1 > 0

theorem negation_of_p : (∃ x : ℝ, x > 0 ∧ 2 * x^2 + 1 ≤ 0) ↔ ¬p :=
by
  sorry

end negation_of_p_l223_223194


namespace triangle_inequality_satisfied_for_n_six_l223_223720

theorem triangle_inequality_satisfied_for_n_six :
  ∀ (a b c : ℝ), 0 < a → 0 < b → 0 < c → 6 * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2) → 
  (a + b > c ∧ a + c > b ∧ b + c > a) := sorry

end triangle_inequality_satisfied_for_n_six_l223_223720


namespace power_calc_l223_223126

theorem power_calc : (3^2)^4 = 6561 := 
by
  sorry

end power_calc_l223_223126


namespace complex_div_imaginary_unit_eq_l223_223476

theorem complex_div_imaginary_unit_eq :
  (∀ i : ℂ, i^2 = -1 → (1 / (1 + i)) = ((1 - i) / 2)) :=
by
  intro i
  intro hi
  /- The proof will be inserted here -/
  sorry

end complex_div_imaginary_unit_eq_l223_223476


namespace range_of_m_l223_223941

theorem range_of_m (x y m : ℝ) (h1 : x - y = 2 * m + 7) (h2 : x + y = 4 * m - 3) 
  (h3 : x < 0) (h4 : y < 0) : m < -2 / 3 := 
by 
  sorry

end range_of_m_l223_223941


namespace net_increase_correct_l223_223051

-- Definitions for the given conditions
def S1 : ℕ := 10
def B1 : ℕ := 15
def S2 : ℕ := 12
def B2 : ℕ := 8
def S3 : ℕ := 9
def B3 : ℕ := 11

def P1 : ℕ := 250
def P2 : ℕ := 275
def P3 : ℕ := 260
def C1 : ℕ := 100
def C2 : ℕ := 110
def C3 : ℕ := 120

def Sale_profit1 : ℕ := S1 * P1
def Sale_profit2 : ℕ := S2 * P2
def Sale_profit3 : ℕ := S3 * P3

def Repair_cost1 : ℕ := B1 * C1
def Repair_cost2 : ℕ := B2 * C2
def Repair_cost3 : ℕ := B3 * C3

def Net_profit1 : ℕ := Sale_profit1 - Repair_cost1
def Net_profit2 : ℕ := Sale_profit2 - Repair_cost2
def Net_profit3 : ℕ := Sale_profit3 - Repair_cost3

def Total_net_profit : ℕ := Net_profit1 + Net_profit2 + Net_profit3

def Net_Increase : ℕ := (B1 - S1) + (B2 - S2) + (B3 - S3)

-- The theorem to be proven
theorem net_increase_correct : Net_Increase = 3 := by
  sorry

end net_increase_correct_l223_223051


namespace greatest_num_fruit_in_each_basket_l223_223968

theorem greatest_num_fruit_in_each_basket : 
  let oranges := 15
  let peaches := 9
  let pears := 18
  let gcd := Nat.gcd (Nat.gcd oranges peaches) pears
  gcd = 3 :=
by
  sorry

end greatest_num_fruit_in_each_basket_l223_223968


namespace simplify_fraction_l223_223876

def expr1 : ℚ := 3
def expr2 : ℚ := 2
def expr3 : ℚ := 3
def expr4 : ℚ := 4
def expected : ℚ := 12 / 5

theorem simplify_fraction : (expr1 / (expr2 - (expr3 / expr4))) = expected := by
  sorry

end simplify_fraction_l223_223876


namespace proof_problem_l223_223667

noncomputable def expr (a b : ℚ) : ℚ :=
  ((a / b + b / a + 2) * ((a + b) / (2 * a) - (b / (a + b)))) /
  ((a + 2 * b + b^2 / a) * (a / (a + b) + b / (a - b)))

theorem proof_problem : expr (3/4 : ℚ) (4/3 : ℚ) = -7/24 :=
by
  sorry

end proof_problem_l223_223667


namespace correct_statement_l223_223906

open Set

variable (U : Set ℕ) (M : Set ℕ)
variables (hU : U = {1, 2, 3, 4, 5}) (hM : U \ M = {1, 3})

theorem correct_statement : 2 ∈ M :=
by
  sorry

end correct_statement_l223_223906


namespace inverse_matrix_proof_l223_223307

variable (A : Matrix (Fin 2) (Fin 2) ℤ)
variable (B : Matrix (Fin 2) (Fin 2) ℤ)
variable (zeroMatrix : Matrix (Fin 2) (Fin 2) ℤ := ![(0, 0), (0, 0)])

-- Condition: The given matrices
def matrixA := ![(5, -3), (-2, 1)]
def matrixB := ![(-1, -3), (-2, -5)]

-- Property to prove: matrixB is the inverse of matrixA
theorem inverse_matrix_proof : 
  (∀ A : Matrix (Fin 2) (Fin 2) ℤ, A = matrixA) →
  (∀ B : Matrix (Fin 2) (Fin 2) ℤ, B = matrixB) →
  (B ⬝ A = 1) := 
  by sorry

end inverse_matrix_proof_l223_223307


namespace negation_of_universal_l223_223486

variable (f : ℝ → ℝ) (m : ℝ)

theorem negation_of_universal :
  (∀ x : ℝ, f x ≥ m) → ¬ (∀ x : ℝ, f x ≥ m) → ∃ x : ℝ, f x < m :=
by
  sorry

end negation_of_universal_l223_223486


namespace solution_set_fraction_inequality_l223_223821

theorem solution_set_fraction_inequality : 
  { x : ℝ | 0 < x ∧ x < 1/3 } = { x : ℝ | 1/x > 3 } :=
by
  sorry

end solution_set_fraction_inequality_l223_223821


namespace square_completing_l223_223707

theorem square_completing (b c : ℤ) (h : (x^2 - 10 * x + 15 = 0) → ((x + b)^2 = c)) : 
  b + c = 5 :=
sorry

end square_completing_l223_223707


namespace probability_of_5_pieces_of_candy_l223_223152

-- Define the conditions
def total_eggs : ℕ := 100 -- Assume total number of eggs is 100 for simplicity
def blue_eggs : ℕ := 4 * total_eggs / 5
def purple_eggs : ℕ := total_eggs / 5
def blue_eggs_with_5_candies : ℕ := blue_eggs / 4
def purple_eggs_with_5_candies : ℕ := purple_eggs / 2
def total_eggs_with_5_candies : ℕ := blue_eggs_with_5_candies + purple_eggs_with_5_candies

-- The proof problem
theorem probability_of_5_pieces_of_candy : (total_eggs_with_5_candies : ℚ) / (total_eggs : ℚ) = 3 / 10 := 
by
  sorry

end probability_of_5_pieces_of_candy_l223_223152


namespace Annette_Caitlin_total_weight_l223_223292

variable (A C S : ℕ)

-- Conditions
axiom cond1 : C + S = 87
axiom cond2 : A = S + 8

-- Theorem
theorem Annette_Caitlin_total_weight : A + C = 95 := by
  sorry

end Annette_Caitlin_total_weight_l223_223292


namespace equality_of_coefficients_l223_223480

open Real

theorem equality_of_coefficients (a b c x : ℝ)
  (h1 : a * x^2 - b * x - c = b * x^2 - c * x - a)
  (h2 : b * x^2 - c * x - a = c * x^2 - a * x - b)
  (h3 : c * x^2 - a * x - b = a * x^2 - b * x - c):
  a = b ∧ b = c :=
sorry

end equality_of_coefficients_l223_223480


namespace greatest_sum_l223_223397

theorem greatest_sum (x y : ℝ) (h1 : x^2 + y^2 = 100) (h2 : x * y = 40) : x + y = 6 * Real.sqrt 5 :=
sorry

end greatest_sum_l223_223397


namespace sum_arithmetic_series_base8_l223_223296

theorem sum_arithmetic_series_base8 : 
  let n := 36
  let a := 1
  let l := 30 -- 36_8 in base 10 is 30
  let S := (n * (a + l)) / 2
  let sum_base10 := 558
  let sum_base8 := 1056 -- 558 in base 8 is 1056
  S = sum_base10 ∧ sum_base10 = 1056 :=
by
  sorry

end sum_arithmetic_series_base8_l223_223296


namespace find_z_l223_223891

theorem find_z (z : ℂ) (h : (Complex.I * z = 4 + 3 * Complex.I)) : z = 3 - 4 * Complex.I :=
by
  sorry

end find_z_l223_223891


namespace count_L_shapes_l223_223735

theorem count_L_shapes (m n : ℕ) (hm : 1 ≤ m) (hn : 1 ≤ n) : 
  ∃ k, k = 4 * (m - 1) * (n - 1) :=
by
  sorry

end count_L_shapes_l223_223735


namespace smallest_number_is_42_l223_223250

theorem smallest_number_is_42 (x : ℤ) 
  (h1 : x + (x + 1) + (x + 2) + (x + 3) + (x + 4) = 225)
  (h2 : x % 7 = 0) : 
  x = 42 := 
sorry

end smallest_number_is_42_l223_223250


namespace container_capacity_l223_223404

variable (C : ℝ)
variable (h1 : 0.30 * C + 27 = (3/4) * C)

theorem container_capacity : C = 60 := by
  sorry

end container_capacity_l223_223404


namespace exponentiation_identity_l223_223107

theorem exponentiation_identity (a m n : ℕ) : (a^m)^n = a^(m * n) :=
by sorry

example : (3^2)^4 = 6561 :=
by
  have h := exponentiation_identity 3 2 4
  rw [←h]
  norm_num

end exponentiation_identity_l223_223107


namespace triangle_inequality_condition_l223_223717

theorem triangle_inequality_condition (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) (ineq : 6 * (a * b + b * c + c * a) > 5 * (a ^ 2 + b ^ 2 + c ^ 2)) : 
  (a < b + c ∧ b < a + c ∧ c < a + b) :=
sorry

end triangle_inequality_condition_l223_223717


namespace range_of_function_l223_223244

open Set

noncomputable def log_base_2 (x : ℝ) : ℝ := Real.log x / Real.log 2

theorem range_of_function (S : Set ℝ) : 
    S = {y : ℝ | ∃ x : ℝ, x ≥ 1 ∧ y = 2 + log_base_2 x} 
    ↔ S = {y : ℝ | y ≥ 2} :=
by 
  sorry

end range_of_function_l223_223244


namespace power_of_powers_l223_223087

theorem power_of_powers :
  (3^2)^4 = 6561 := 
by 
  sorry

end power_of_powers_l223_223087


namespace license_plates_count_l223_223537

noncomputable def number_of_distinct_license_plates : ℕ :=
  let digits_choices := 10^5
  let letter_block_choices := 26^3
  let positions_choices := 6
  positions_choices * digits_choices * letter_block_choices

theorem license_plates_count : number_of_distinct_license_plates = 105456000 := by
  unfold number_of_distinct_license_plates
  calc
    6 * 10^5 * 26^3 = 6 * 100000 * 17576 : by norm_num
                  ... = 105456000 : by norm_num
  sorry

end license_plates_count_l223_223537


namespace sufficient_condition_for_reciprocal_inequality_l223_223009

theorem sufficient_condition_for_reciprocal_inequality (a b : ℝ) (h : b < a ∧ a < 0) : (1 / a) < (1 / b) :=
sorry

end sufficient_condition_for_reciprocal_inequality_l223_223009


namespace units_digit_35_pow_35_mul_17_pow_17_l223_223564

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_35_pow_35_mul_17_pow_17:
  units_digit (35 ^ (35 * 17 ^ 17)) = 5 := 
by {
  -- Here we're skipping the proof.
  sorry
}

end units_digit_35_pow_35_mul_17_pow_17_l223_223564


namespace number_of_valid_programs_l223_223690

/--
A student must choose a program of five courses from an expanded list of courses that includes English, Algebra, Geometry, Calculus, Biology, History, Art, and Latin. 
The program must include both English and at least two mathematics courses. 
Prove that the number of valid ways to choose such a program is 22.
-/
theorem number_of_valid_programs : 
  let courses := ["English", "Algebra", "Geometry", "Calculus", "Biology", "History", "Art", "Latin"]
  ∃ valid_programs, 
  valid_programs = 22 := 
by 
  -- Define the total number of courses excluding English
  let total_courses_excl_english := 7

  -- Calculate the total number of combinations without any restrictions
  let total_combinations := Nat.choose total_courses_excl_english 4

  -- Calculate invalid cases
  let no_math_courses := Nat.choose 4 4 -- choosing all non-math courses
  let one_math_course := Nat.choose 3 1 * Nat.choose 4 3 -- choosing 1 out of 3 math courses and rest non-math

  -- Total invalid combinations that don’t meet the mathematics course requirement
  let invalid_combinations := no_math_courses + one_math_course

  -- Calculate the number of valid programs
  let valid_programs := total_combinations - invalid_combinations

  -- The number of valid programs should be 22
  existsi valid_programs
  assume valid_programs_eq
  show valid_programs_eq = 22 by sorry

end number_of_valid_programs_l223_223690


namespace isosceles_triangle_base_length_l223_223802

theorem isosceles_triangle_base_length (a b P : ℕ) (h1 : a = 7) (h2 : P = 23) (h3 : P = 2 * a + b) : b = 9 :=
sorry

end isosceles_triangle_base_length_l223_223802


namespace trig_identity_l223_223506

noncomputable def sin (x : ℝ) : ℝ := sorry
noncomputable def cos (x : ℝ) : ℝ := sorry

theorem trig_identity (θ : ℝ) : sin (θ + 75 * Real.pi / 180) + cos (θ + 45 * Real.pi / 180) - Real.sqrt 3 * cos (θ + 15 * Real.pi / 180) = 0 :=
by
  sorry

end trig_identity_l223_223506


namespace tank_salt_solution_l223_223669

theorem tank_salt_solution (x : ℝ) (h1 : (0.20 * x + 14) / ((3 / 4) * x + 21) = 1 / 3) : x = 140 :=
sorry

end tank_salt_solution_l223_223669


namespace cube_div_identity_l223_223138

theorem cube_div_identity (a b : ℕ) (h1 : a = 6) (h2 : b = 3) : 
  (a^3 - b^3) / (a^2 + a * b + b^2) = 3 :=
by {
  sorry
}

end cube_div_identity_l223_223138


namespace water_usage_in_May_l223_223527

theorem water_usage_in_May (x : ℝ) (h_cost : 45 = if x ≤ 12 then 2 * x 
                                                else if x ≤ 18 then 24 + 2.5 * (x - 12) 
                                                else 39 + 3 * (x - 18)) : x = 20 :=
sorry

end water_usage_in_May_l223_223527


namespace vector_sum_correct_l223_223311

def vec1 : Fin 3 → ℤ := ![-7, 3, 5]
def vec2 : Fin 3 → ℤ := ![4, -1, -6]
def vec3 : Fin 3 → ℤ := ![1, 8, 2]
def expectedSum : Fin 3 → ℤ := ![-2, 10, 1]

theorem vector_sum_correct :
  (fun i => vec1 i + vec2 i + vec3 i) = expectedSum := 
by
  sorry

end vector_sum_correct_l223_223311


namespace prime_remainder_30_l223_223972

theorem prime_remainder_30 (p : ℕ) (hp : Nat.Prime p) (hgt : p > 30) (hmod2 : p % 2 ≠ 0) 
(hmod3 : p % 3 ≠ 0) (hmod5 : p % 5 ≠ 0) : 
  ∃ (r : ℕ), r < 30 ∧ (p % 30 = r) ∧ (r = 1 ∨ Nat.Prime r) := 
by
  sorry

end prime_remainder_30_l223_223972


namespace general_pattern_specific_computation_l223_223624

theorem general_pattern (n : ℕ) : n * (n + 1) * (n + 2) * (n + 3) + 1 = (n^2 + 3*n + 1)^2 :=
by
  sorry

theorem specific_computation : 2000 * 2001 * 2002 * 2003 + 1 = 4006001^2 :=
by
  have h := general_pattern 2000
  exact h

end general_pattern_specific_computation_l223_223624


namespace range_of_m_l223_223019

noncomputable def f (x : ℝ) : ℝ := (Real.exp x) / (x * (Real.exp x) - x + 1)
noncomputable def m_range := Set.Icc (Real.exp 2 / (2 * (Real.exp 2) - 1)) 1

theorem range_of_m :
  ∃ m : ℝ, (∀ x : ℝ, m < f x ↔ f x = 0 ∨ f x = 1) ↔ m ∈ m_range := sorry

end range_of_m_l223_223019


namespace student_ticket_price_l223_223830

theorem student_ticket_price
  (S : ℕ)
  (num_tickets : ℕ := 2000)
  (num_student_tickets : ℕ := 520)
  (price_non_student : ℕ := 11)
  (total_revenue : ℕ := 20960)
  (h : 520 * S + (2000 - 520) * 11 = 20960) :
  S = 9 :=
sorry

end student_ticket_price_l223_223830


namespace question_l223_223930

variable (U : Set ℕ) (M : Set ℕ)

theorem question :
  U = {1, 2, 3, 4, 5} →
  (U \ M = {1, 3}) →
  2 ∈ M :=
by
  intros
  sorry

end question_l223_223930


namespace correct_statement_l223_223910

universe u
variable (α : Type u)

noncomputable def U : Set ℕ := {1, 2, 3, 4, 5}

noncomputable def complement_U_M : Set ℕ := {1, 3}

noncomputable def M : Set ℕ := U α \ complement_U_M

theorem correct_statement : 2 ∈ M :=
by {
  sorry
}

end correct_statement_l223_223910


namespace probability_A_B_C_adjacent_l223_223566

theorem probability_A_B_C_adjacent (students : Fin 5 → Prop) (A B C : Fin 5) :
  (students A ∧ students B ∧ students C) →
  (∃ n m : ℕ, n = 48 ∧ m = 12 ∧ m / n = (1 : ℚ) / 4) :=
by
  sorry

end probability_A_B_C_adjacent_l223_223566


namespace find_semi_perimeter_l223_223628

noncomputable def semi_perimeter_of_rectangle (a b : ℝ) (h₁ : a * b = 4024) (h₂ : a = 2 * b) : ℝ :=
  (a + b) / 2

theorem find_semi_perimeter (a b : ℝ) (h₁ : a * b = 4024) (h₂ : a = 2 * b) : semi_perimeter_of_rectangle a b h₁ h₂ = (3 / 2) * Real.sqrt 2012 :=
  sorry

end find_semi_perimeter_l223_223628


namespace find_x_l223_223881

theorem find_x (x : ℝ) (h : (x / 2) + 6 = 2 * x - 6) : x = 8 :=
by
  sorry

end find_x_l223_223881


namespace total_stocking_stuffers_total_stocking_stuffers_hannah_buys_l223_223949

def candy_canes_per_kid : ℕ := 4
def beanie_babies_per_kid : ℕ := 2
def books_per_kid : ℕ := 1
def kids : ℕ := 3

theorem total_stocking_stuffers : candy_canes_per_kid + beanie_babies_per_kid + books_per_kid = 7 :=
by { 
  -- by trusted computation
  sorry
}

theorem total_stocking_stuffers_hannah_buys : 3 * (candy_canes_per_kid + beanie_babies_per_kid + books_per_kid) = 21 :=
by {
  have h : candy_canes_per_kid + beanie_babies_per_kid + books_per_kid = 7 := total_stocking_stuffers,
  rw h,
  norm_num,
}

end total_stocking_stuffers_total_stocking_stuffers_hannah_buys_l223_223949


namespace S_11_eq_zero_l223_223895

noncomputable def S (n : ℕ) : ℝ := sorry
variable (a_n : ℕ → ℝ) (d : ℝ)
variable (h1 : ∀ n, a_n (n+1) = a_n n + d) -- common difference d ≠ 0
variable (h2 : S 5 = S 6)

theorem S_11_eq_zero (h_nonzero : d ≠ 0) : S 11 = 0 := by
  sorry

end S_11_eq_zero_l223_223895


namespace chess_tournament_green_teams_l223_223599

theorem chess_tournament_green_teams :
  ∀ (R G total_teams : ℕ)
  (red_team_count : ℕ → ℕ)
  (green_team_count : ℕ → ℕ)
  (mixed_team_count : ℕ → ℕ),
  R = 64 → G = 68 → total_teams = 66 →
  red_team_count R = 20 →
  (R + G = 132) →
  -- Details derived from mixed_team_count and green_team_count
  -- are inferred from the conditions provided
  mixed_team_count R + red_team_count R = 32 → 
  -- Total teams by definition including mixed teams 
  mixed_team_count G = G - (2 * red_team_count R) - green_team_count G →
  green_team_count (G - (mixed_team_count R)) = 2 → 
  2 * (green_team_count G) = 22 :=
by sorry

end chess_tournament_green_teams_l223_223599


namespace find_a_range_for_two_distinct_roots_l223_223017

def f (x : ℝ) : ℝ := x^3 - 3 * x + 5

theorem find_a_range_for_two_distinct_roots :
  ∀ (a : ℝ), 3 ≤ a ∧ a ≤ 7 → ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ f x1 = a ∧ f x2 = a :=
by
  -- The proof will be here
  sorry

end find_a_range_for_two_distinct_roots_l223_223017


namespace initial_cars_l223_223829

theorem initial_cars (X : ℕ) : (X - 13 + (13 + 5) = 85) → (X = 80) :=
by
  sorry

end initial_cars_l223_223829


namespace f_D_not_mapping_to_B_l223_223222

def A := {x : ℝ | 1 ≤ x ∧ x ≤ 2}
def B := {y : ℝ | 1 ≤ y ∧ y <= 4}
def f_D (x : ℝ) := 4 - x^2

theorem f_D_not_mapping_to_B : ¬ (∀ x ∈ A, f_D x ∈ B) := sorry

end f_D_not_mapping_to_B_l223_223222


namespace find_number_l223_223440

theorem find_number (x : ℤ) (h : 42 + 3 * x - 10 = 65) : x = 11 := 
by 
  sorry 

end find_number_l223_223440


namespace find_z_l223_223348

theorem find_z (a z : ℕ) (h₁ : a = 105) (h₂ : a^3 = 21 * 25 * 45 * z) : z = 49 :=
sorry

end find_z_l223_223348


namespace total_carrots_grown_l223_223782

theorem total_carrots_grown :
  let Sandy := 6.5
  let Sam := 3.25
  let Sophie := 2.75 * Sam
  let Sara := (Sandy + Sam + Sophie) - 7.5
  Sandy + Sam + Sophie + Sara = 29.875 :=
by
  sorry

end total_carrots_grown_l223_223782


namespace power_of_powers_l223_223085

theorem power_of_powers :
  (3^2)^4 = 6561 := 
by 
  sorry

end power_of_powers_l223_223085


namespace exists_integers_for_expression_l223_223772

theorem exists_integers_for_expression (n : ℤ) : 
  ∃ a b c d : ℤ, n = a^2 + b^2 - c^2 - d^2 := 
sorry

end exists_integers_for_expression_l223_223772


namespace baker_cake_count_l223_223872

theorem baker_cake_count :
  let initial_cakes := 62
  let additional_cakes := 149
  let sold_cakes := 144
  initial_cakes + additional_cakes - sold_cakes = 67 :=
by
  sorry

end baker_cake_count_l223_223872


namespace find_a_l223_223589

noncomputable def f (x a : ℝ) : ℝ := x + Real.exp (x - a)
noncomputable def g (x a : ℝ) : ℝ := Real.log (x + 2) - 4 * Real.exp (a - x)

theorem find_a {x0 a : ℝ} (h : f x0 a - g x0 a = 3) : a = -Real.log 2 - 1 :=
sorry

end find_a_l223_223589


namespace max_sum_a_b_c_d_l223_223346

theorem max_sum_a_b_c_d (a c d : ℤ) (b : ℕ) (h1 : a + b = c) (h2 : b + c = d) (h3 : c + d = a) : 
  a + b + c + d = -5 := 
sorry

end max_sum_a_b_c_d_l223_223346


namespace exponential_decreasing_iff_frac_inequality_l223_223443

theorem exponential_decreasing_iff_frac_inequality (a : ℝ) :
  (0 < a ∧ a < 1) ↔ (a ≠ 1 ∧ a * (a - 1) ≤ 0) :=
by
  sorry

end exponential_decreasing_iff_frac_inequality_l223_223443


namespace xy_in_N_l223_223773

def M := {x : ℤ | ∃ m : ℤ, x = 3 * m + 1}
def N := {y : ℤ | ∃ n : ℤ, y = 3 * n + 2}

theorem xy_in_N (x y : ℤ) (hx : x ∈ M) (hy : y ∈ N) : (x * y) ∈ N :=
by
  sorry

end xy_in_N_l223_223773


namespace union_sets_l223_223364

-- Define the sets A and B using their respective conditions.
def A : Set ℝ := {x : ℝ | 3 < x ∧ x ≤ 7}
def B : Set ℝ := {x : ℝ | 4 < x ∧ x ≤ 10}

-- The theorem we aim to prove.
theorem union_sets : A ∪ B = {x : ℝ | 3 < x ∧ x ≤ 10} := 
by
  sorry

end union_sets_l223_223364


namespace number_of_lemons_l223_223637

theorem number_of_lemons
  (total_fruits : ℕ)
  (mangoes : ℕ)
  (pears : ℕ)
  (pawpaws : ℕ)
  (kiwis : ℕ)
  (lemons : ℕ)
  (h_total : total_fruits = 58)
  (h_mangoes : mangoes = 18)
  (h_pears : pears = 10)
  (h_pawpaws : pawpaws = 12)
  (h_kiwis_lemons_equal : kiwis = lemons) :
  lemons = 9 :=
by
  sorry

end number_of_lemons_l223_223637


namespace gcd_1680_1683_l223_223728

theorem gcd_1680_1683 :
  ∀ (n : ℕ), n = 1683 →
  (∀ m, (m = 5 ∨ m = 67 ∨ m = 8) → n % m = 3) →
  (∃ d, d > 1 ∧ d ∣ 1683 ∧ d = Nat.gcd 1680 n ∧ Nat.gcd 1680 n = 3) :=
by
  sorry

end gcd_1680_1683_l223_223728


namespace minimum_odd_numbers_in_set_l223_223619

-- Definitions
variable (P : ℝ → ℝ)
variable (degree_P : ℕ)
variable (A_P : Set ℝ)

-- The conditions: P is a polynomial of degree 8, and 8 is included in A_P
def is_polynomial_of_degree_eight (P : ℝ → ℝ) (degree_P : ℕ) : Prop :=
  degree_P = 8

def set_includes_eight (A_P : Set ℝ) : Prop := 
  8 ∈ A_P

-- The goal: prove the minimum number of odd numbers in A_P is 1
theorem minimum_odd_numbers_in_set {P : ℝ → ℝ} {degree_P : ℕ} {A_P : Set ℝ} :
  is_polynomial_of_degree_eight P degree_P → 
  set_includes_eight A_P → 
  ∃ odd_numbers : ℕ, odd_numbers = 1 :=
sorry

end minimum_odd_numbers_in_set_l223_223619


namespace convince_jury_l223_223390

def not_guilty : Prop := sorry  -- definition indicating the defendant is not guilty
def not_liar : Prop := sorry    -- definition indicating the defendant is not a liar
def innocent_knight_statement : Prop := sorry  -- statement "I am an innocent knight"

theorem convince_jury (not_guilty : not_guilty) (not_liar : not_liar) : innocent_knight_statement :=
sorry

end convince_jury_l223_223390


namespace neither_sufficient_nor_necessary_l223_223792

theorem neither_sufficient_nor_necessary (a b : ℝ) (h1 : a ≠ 5) (h2 : b ≠ -5) : ¬((a + b ≠ 0) ↔ (a ≠ 5 ∧ b ≠ -5)) :=
by sorry

end neither_sufficient_nor_necessary_l223_223792


namespace apples_per_basket_l223_223507

theorem apples_per_basket (total_apples : ℕ) (num_baskets : ℕ) (h : total_apples = 629) (k : num_baskets = 37) :
  total_apples / num_baskets = 17 :=
by
  -- proof omitted
  sorry

end apples_per_basket_l223_223507


namespace number_of_possible_SC_values_l223_223370

open Finset

variable (𝒞 : Set ℕ)
variable (n a d : ℕ)

def sum_arithmetic_sequence (n a d : ℕ) : ℕ :=
  (n * (2 * a + (n - 1) * d)) / 2

noncomputable def count_possible_sums : ℕ :=
  let S_min := sum_arithmetic_sequence 75 10 1
  let S_max := sum_arithmetic_sequence 75 126 1
  S_max - S_min + 1

theorem number_of_possible_SC_values 
  (𝒞 : Finset ℕ) 
  (h₁ : 𝒞.card = 75) 
  (h₂ : ∀ x ∈ 𝒞, 10 ≤ x ∧ x ≤ 200) :
  count_possible_sums = 8476 := by
  sorry

end number_of_possible_SC_values_l223_223370


namespace f_0_eq_0_l223_223463

-- Define a function f with the given condition
def f (x : ℤ) : ℤ := if x = 0 then 0
                     else (x-1)^2 + 2*(x-1) + 1

-- State the theorem
theorem f_0_eq_0 : f 0 = 0 :=
by sorry

end f_0_eq_0_l223_223463


namespace reflected_curve_equation_l223_223183

-- Define the original curve equation
def original_curve (x y : ℝ) : Prop :=
  2 * x^2 + 4 * x * y + 5 * y^2 - 22 = 0

-- Define the line of reflection
def line_of_reflection (x y : ℝ) : Prop :=
  x - 2 * y + 1 = 0

-- Define the equation of the reflected curve
def reflected_curve (x y : ℝ) : Prop :=
  146 * x^2 - 44 * x * y + 29 * y^2 + 152 * x - 64 * y - 494 = 0

-- Problem: Prove the equation of the reflected curve is as given
theorem reflected_curve_equation (x y : ℝ) :
  (∃ x1 y1 : ℝ, original_curve x1 y1 ∧ line_of_reflection x1 y1 ∧ (x, y) = (x1, y1)) →
  reflected_curve x y :=
by
  intros
  sorry

end reflected_curve_equation_l223_223183


namespace exists_integers_for_S_geq_100_l223_223228

theorem exists_integers_for_S_geq_100 (S : ℤ) (hS : S ≥ 100) :
  ∃ (T C B : ℤ) (P : ℤ),
    T > 0 ∧ C > 0 ∧ B > 0 ∧
    T > C ∧ C > B ∧
    T + C + B = S ∧
    T * C * B = P ∧
    (∀ (T₁ C₁ B₁ T₂ C₂ B₂ : ℤ), 
      T₁ > 0 ∧ C₁ > 0 ∧ B₁ > 0 ∧ 
      T₂ > 0 ∧ C₂ > 0 ∧ B₂ > 0 ∧ 
      T₁ > C₁ ∧ C₁ > B₁ ∧ 
      T₂ > C₂ ∧ C₂ > B₂ ∧ 
      T₁ + C₁ + B₁ = S ∧ 
      T₂ + C₂ + B₂ = S ∧ 
      T₁ * C₁ * B₁ = T₂ * C₂ * B₂ → 
      (T₁ = T₂) ∧ (C₁ = C₂) ∧ (B₁ = B₂) → false) :=
sorry

end exists_integers_for_S_geq_100_l223_223228


namespace problem_solution_l223_223926

universe u

variable (U : Set Nat) (M : Set Nat)
variable (complement_U_M : Set Nat)

axiom U_def : U = {1, 2, 3, 4, 5}
axiom complement_U_M_def : complement_U_M = {1, 3}
axiom M_def : M = U \ complement_U_M

theorem problem_solution : 2 ∈ M := by
  sorry

end problem_solution_l223_223926


namespace three_pow_gt_pow_three_for_n_ne_3_l223_223487

theorem three_pow_gt_pow_three_for_n_ne_3 (n : ℕ) (h : n ≠ 3) : 3^n > n^3 :=
sorry

end three_pow_gt_pow_three_for_n_ne_3_l223_223487


namespace exp_eval_l223_223099

theorem exp_eval : (3^2)^4 = 6561 := by
sorry

end exp_eval_l223_223099


namespace find_m_l223_223352

theorem find_m (m : ℝ) (h : |m - 4| = |2 * m + 7|) : m = -11 ∨ m = -1 :=
sorry

end find_m_l223_223352


namespace find_son_l223_223683

variable (SonAge ManAge : ℕ)

def age_relationship (SonAge ManAge : ℕ) : Prop :=
  ManAge = SonAge + 20 ∧ ManAge + 2 = 2 * (SonAge + 2)

theorem find_son's_age (S M : ℕ) (h : age_relationship S M) : S = 18 :=
by
  unfold age_relationship at h
  obtain ⟨h1, h2⟩ := h
  sorry

end find_son_l223_223683


namespace faucets_fill_time_l223_223890

theorem faucets_fill_time (fill_time_4faucets_200gallons_12min : 4 * 12 * faucet_rate = 200) 
    (fill_time_m_50gallons_seconds : ∃ (rate: ℚ), 8 * t_to_seconds * rate = 50) : 
    8 * t_to_seconds / 33.33 = 90 :=
by sorry


end faucets_fill_time_l223_223890


namespace exists_initial_segment_of_power_of_2_l223_223780

theorem exists_initial_segment_of_power_of_2 (m : ℕ) : ∃ n : ℕ, ∃ k : ℕ, k ≥ m ∧ 2^n = 10^k * m ∨ 2^n = 10^k * (m+1) := 
by
  sorry

end exists_initial_segment_of_power_of_2_l223_223780


namespace f_value_at_5pi_over_6_l223_223350

noncomputable def f (x ω : ℝ) := 2 * Real.sin (ω * x + (Real.pi / 3))

theorem f_value_at_5pi_over_6
  (ω : ℝ) (ω_pos : ω > 0)
  (α β : ℝ)
  (h1 : f α ω = 2)
  (h2 : f β ω = 0)
  (h3 : Real.sqrt ((α - β)^2 + 4) = Real.sqrt (4 + (Real.pi^2 / 4))) :
  f (5 * Real.pi / 6) ω = -1 := 
sorry

end f_value_at_5pi_over_6_l223_223350


namespace sum_of_reciprocals_of_squares_l223_223777

open BigOperators

theorem sum_of_reciprocals_of_squares (n : ℕ) (h : n ≥ 2) :
   (∑ k in Finset.range n, 1 / (k + 1)^2) < (2 * n - 1) / n :=
sorry

end sum_of_reciprocals_of_squares_l223_223777


namespace union_of_M_N_l223_223524

def M : Set ℝ := { x | x^2 + 2*x = 0 }

def N : Set ℝ := { x | x^2 - 2*x = 0 }

theorem union_of_M_N : M ∪ N = {0, -2, 2} := sorry

end union_of_M_N_l223_223524


namespace abs_inequality_interval_notation_l223_223301

variable (x : ℝ)

theorem abs_inequality_interval_notation :
  {x : ℝ | |x - 1| < 1} = {x : ℝ | 0 < x ∧ x < 2} :=
by
  sorry

end abs_inequality_interval_notation_l223_223301


namespace speed_of_current_is_2_l223_223154

noncomputable def speed_current : ℝ :=
  let still_water_speed := 14  -- kmph
  let distance_m := 40         -- meters
  let time_s := 8.9992800576   -- seconds
  let distance_km := distance_m / 1000
  let time_h := time_s / 3600
  let downstream_speed := distance_km / time_h
  downstream_speed - still_water_speed

theorem speed_of_current_is_2 :
  speed_current = 2 :=
by
  sorry

end speed_of_current_is_2_l223_223154


namespace total_surface_area_correct_l223_223158

noncomputable def total_surface_area_of_cylinder (radius height : ℝ) : ℝ :=
  let lateral_surface_area := 2 * Real.pi * radius * height
  let top_and_bottom_area := 2 * Real.pi * radius^2
  lateral_surface_area + top_and_bottom_area

theorem total_surface_area_correct : total_surface_area_of_cylinder 3 10 = 78 * Real.pi :=
by
  sorry

end total_surface_area_correct_l223_223158


namespace Berta_winning_strategy_l223_223697

theorem Berta_winning_strategy:
  ∃ (N : ℕ), 
  N ≥ 100000 ∧ 
  (∀ (n : ℕ), n = N →
    (∀ (k : ℕ), (k ≥ 1 ∧ ((k % 2 = 0 ∧ k ≤ n / 2) ∨ (k % 2 = 1 ∧ n / 2 ≤ k ∧ k ≤ n))) →
      ∃ (m : ℕ), m = n - k ∧ (m + m.succ = n ∨ m + 2.msucc = n)) ∧
        ((N = 2 ^ x - 2) ∧ ∀ x, N = (2 ^ x - 2) → N = 131070 :=
begin
  sorry
end

/- Theorem's Description:
We claim that there exists a number \( N \ge 100000 \) such that Berta has a winning strategy under the given game rules. For \( n \) marbles on the table, the conditions for removing marbles are:
- \( k \ge 1 \)
- \( k \) is either an even number not more than half the total marbles, or an odd number not less than half the total marbles and not more than the total marbles.
We also prove that Berta's winning strategy guarantees that \( N = 131070 \).
-/

end Berta_winning_strategy_l223_223697


namespace gcd_2210_145_l223_223396

-- defining the constants a and b
def a : ℕ := 2210
def b : ℕ := 145

-- theorem stating that gcd(a, b) = 5
theorem gcd_2210_145 : Nat.gcd a b = 5 :=
sorry

end gcd_2210_145_l223_223396


namespace factor_t_squared_minus_81_l223_223306

theorem factor_t_squared_minus_81 (t : ℝ) : t^2 - 81 = (t - 9) * (t + 9) :=
by
  sorry

end factor_t_squared_minus_81_l223_223306


namespace shooter_hit_rate_l223_223863

noncomputable def shooter_prob := 2 / 3

theorem shooter_hit_rate:
  ∀ (x : ℚ), (1 - x)^4 = 1 / 81 → x = shooter_prob :=
by
  intro x h
  -- Proof is omitted
  sorry

end shooter_hit_rate_l223_223863


namespace triangle_side_ratios_l223_223028

theorem triangle_side_ratios
    (A B C : ℝ) (a b c : ℝ)
    (h1 : 2 * b * Real.sin (2 * A) = a * Real.sin B)
    (h2 : c = 2 * b) :
    a / b = 2 :=
by
  sorry

end triangle_side_ratios_l223_223028


namespace total_cost_all_children_l223_223254

-- Defining the constants and conditions
def regular_tuition : ℕ := 45
def early_bird_discount : ℕ := 15
def first_sibling_discount : ℕ := 15
def additional_sibling_discount : ℕ := 10
def weekend_class_extra_cost : ℕ := 20
def multi_instrument_discount : ℕ := 10

def Ali_cost : ℕ := regular_tuition - early_bird_discount
def Matt_cost : ℕ := regular_tuition - first_sibling_discount
def Jane_cost : ℕ := regular_tuition - additional_sibling_discount + weekend_class_extra_cost - multi_instrument_discount
def Sarah_cost : ℕ := regular_tuition - additional_sibling_discount + weekend_class_extra_cost - multi_instrument_discount

-- Proof statement
theorem total_cost_all_children : Ali_cost + Matt_cost + Jane_cost + Sarah_cost = 150 := by
  sorry

end total_cost_all_children_l223_223254


namespace smallest_positive_number_among_options_l223_223694

theorem smallest_positive_number_among_options :
  (10 > 3 * Real.sqrt 11) →
  (51 > 10 * Real.sqrt 26) →
  min (10 - 3 * Real.sqrt 11) (51 - 10 * Real.sqrt 26) = 51 - 10 * Real.sqrt 26 :=
by
  intros h1 h2
  sorry

end smallest_positive_number_among_options_l223_223694


namespace exponentiation_rule_example_l223_223094

theorem exponentiation_rule_example : (3^2)^4 = 6561 := 
by {
  -- This proof is based on the power rule (ab)^c = a^(b*c)
  have step1 : (3^2)^4 = 3^(2 * 4), 
  {
    rw pow_mul,
  },
  -- Simplification of 3^(2*4) to 3^8
  have step2 : 3^(2 * 4) = 3^8,
  {
    refl,
  },
  -- Actual computation for 3^8 = 6561
  have step3 : 3^8 = 6561,
  {
    norm_num,
  },
  -- Combining all steps
  rw [step1, step2, step3],
  exact rfl,
}

end exponentiation_rule_example_l223_223094


namespace triangle_inequality_condition_l223_223716

theorem triangle_inequality_condition (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) (ineq : 6 * (a * b + b * c + c * a) > 5 * (a ^ 2 + b ^ 2 + c ^ 2)) : 
  (a < b + c ∧ b < a + c ∧ c < a + b) :=
sorry

end triangle_inequality_condition_l223_223716


namespace william_total_tickets_l223_223665

def initial_tickets : ℕ := 15
def additional_tickets : ℕ := 3
def total_tickets : ℕ := initial_tickets + additional_tickets

theorem william_total_tickets :
  total_tickets = 18 := by
  -- proof goes here
  sorry

end william_total_tickets_l223_223665


namespace base_of_second_exponent_l223_223027

theorem base_of_second_exponent (a b : ℕ) (x : ℕ) 
  (h1 : (18^a) * (x^(3 * a - 1)) = (2^6) * (3^b)) 
  (h2 : a = 6) 
  (h3 : 0 < a)
  (h4 : 0 < b) : x = 3 := 
by
  sorry

end base_of_second_exponent_l223_223027


namespace probability_prime_or_odd_ball_l223_223882

def isPrime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def isOdd (n : ℕ) : Prop :=
  n % 2 = 1

def isPrimeOrOdd (n : ℕ) : Prop :=
  isPrime n ∨ isOdd n

theorem probability_prime_or_odd_ball :
  (1+2+3+5+7)/8 = 5/8 := by
  sorry

end probability_prime_or_odd_ball_l223_223882


namespace sequence_sum_a_b_l223_223447

theorem sequence_sum_a_b (a b : ℕ) (a_seq : ℕ → ℕ) 
  (h1 : a_seq 1 = a)
  (h2 : a_seq 2 = b)
  (h3 : ∀ n ≥ 1, a_seq (n+2) = (a_seq n + 2018) / (a_seq (n+1) + 1)) :
  a + b = 1011 ∨ a + b = 2019 :=
sorry

end sequence_sum_a_b_l223_223447


namespace problem_ineq_l223_223621

theorem problem_ineq (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z)
(h4 : x * y * z = 1) :
    (x^3 / ((1 + y)*(1 + z)) + y^3 / ((1 + z)*(1 + x)) + z^3 / ((1 + x)*(1 + y))) ≥ 3 / 4 := 
sorry

end problem_ineq_l223_223621


namespace min_sum_six_l223_223579

theorem min_sum_six (a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h : a * b = a + b + 3) :
  a + b ≥ 6 :=
sorry

end min_sum_six_l223_223579


namespace inequality_solution_set_l223_223823

theorem inequality_solution_set (x : ℝ) (h : x ≠ 0) : 
  (1 / x > 3) ↔ (0 < x ∧ x < 1 / 3) := 
by 
  sorry

end inequality_solution_set_l223_223823


namespace exponentiation_rule_example_l223_223090

theorem exponentiation_rule_example : (3^2)^4 = 6561 := 
by {
  -- This proof is based on the power rule (ab)^c = a^(b*c)
  have step1 : (3^2)^4 = 3^(2 * 4), 
  {
    rw pow_mul,
  },
  -- Simplification of 3^(2*4) to 3^8
  have step2 : 3^(2 * 4) = 3^8,
  {
    refl,
  },
  -- Actual computation for 3^8 = 6561
  have step3 : 3^8 = 6561,
  {
    norm_num,
  },
  -- Combining all steps
  rw [step1, step2, step3],
  exact rfl,
}

end exponentiation_rule_example_l223_223090


namespace min_odd_in_A_P_l223_223610

-- Define the polynomial P of degree 8
variable {R : Type*} [Ring R] (P : Polynomial R)
variable (hP : degree P = 8)

-- Define the set A_P for some value c in R
def A_P (c : R) : Set R := { x | P.eval x = c }

-- Given that the number 8 is included in the set A_P
variable (c : R) (h8 : (8 : R) ∈ A_P c)

-- Prove that the minimum number of odd numbers in the set A_P is 1
theorem min_odd_in_A_P : ∃ x ∈ A_P c, (x % 2 = 1) := sorry

end min_odd_in_A_P_l223_223610


namespace calculation_l223_223554

-- Define the exponents and base values as conditions
def exponent : ℕ := 3 ^ 2
def neg_base : ℤ := -2
def pos_base : ℤ := 2

-- The calculation expressions as conditions
def term1 : ℤ := neg_base^exponent
def term2 : ℤ := pos_base^exponent

-- The proof statement: Show that the sum of the terms equals 0
theorem calculation : term1 + term2 = 0 := sorry

end calculation_l223_223554


namespace sample_size_l223_223528

theorem sample_size 
  (n_A n_B n_C : ℕ)
  (h1 : n_A = 15)
  (h2 : 3 * n_B = 4 * n_A)
  (h3 : 3 * n_C = 7 * n_A) :
  n_A + n_B + n_C = 70 :=
by
sorry

end sample_size_l223_223528


namespace pain_subsided_days_l223_223212

-- Define the problem conditions in Lean
variable (x : ℕ) -- the number of days it takes for the pain to subside

-- Condition 1: The injury takes 5 times the pain subsiding period to fully heal
def injury_healing_days := 5 * x

-- Condition 2: James waits an additional 3 days after the injury is fully healed
def workout_waiting_days := injury_healing_days + 3

-- Condition 3: James waits another 3 weeks (21 days) before lifting heavy
def total_days_until_lifting_heavy := workout_waiting_days + 21

-- Given the total days until James can lift heavy is 39 days, prove x = 3
theorem pain_subsided_days : 
    total_days_until_lifting_heavy x = 39 → x = 3 := by
  sorry

end pain_subsided_days_l223_223212


namespace solution_set_of_inequality_l223_223191

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f' : ℝ → ℝ := sorry

theorem solution_set_of_inequality (H1 : f 1 = 1)
  (H2 : ∀ x : ℝ, x * f' x < 1 / 2) :
  {x : ℝ | f (Real.log x ^ 2) < (Real.log x ^ 2) / 2 + 1 / 2} = 
  {x : ℝ | 0 < x ∧ x < 1 / 10} ∪ {x : ℝ | x > 10} :=
sorry

end solution_set_of_inequality_l223_223191


namespace simplify_and_evaluate_l223_223231

theorem simplify_and_evaluate :
  let a := 1
  let b := 2
  (a - b) ^ 2 - a * (a - b) + (a + b) * (a - b) = -1 := by
  sorry

end simplify_and_evaluate_l223_223231


namespace farmer_apples_count_l223_223235

theorem farmer_apples_count (initial : ℕ) (given : ℕ) (remaining : ℕ) 
  (h1 : initial = 127) (h2 : given = 88) : remaining = initial - given := 
by
  sorry

end farmer_apples_count_l223_223235


namespace count_unique_elements_in_set_l223_223073

def f (x : ℝ) : ℝ := Real.floor x + Real.floor (2 * x) + Real.floor (3 * x)

theorem count_unique_elements_in_set : 
  (Finset.image f (Finset.Icc 1 100)).card = 67 := 
sorry

end count_unique_elements_in_set_l223_223073


namespace population_proof_l223_223685

def population (tosses : ℕ) (values : ℕ) : Prop :=
  (tosses = 7768) ∧ (values = 6)

theorem population_proof : 
  population 7768 6 :=
by
  unfold population
  exact And.intro rfl rfl

end population_proof_l223_223685


namespace graph_movement_l223_223754

noncomputable def f (x : ℝ) : ℝ := -2 * (x - 1) ^ 2 + 3

noncomputable def g (x : ℝ) : ℝ := -2 * x ^ 2

theorem graph_movement :
  ∀ (x y : ℝ),
  y = f x →
  g x = y → 
  (∃ Δx Δy, Δx = -1 ∧ Δy = -3 ∧ g (x + Δx) = y + Δy) :=
by
  sorry

end graph_movement_l223_223754


namespace quadratic_positivity_range_l223_223739

variable (a : ℝ)

def quadratic_function (x : ℝ) : ℝ :=
  a * x^2 - 2 * a * x + 3

theorem quadratic_positivity_range :
  (∀ x, 0 < x ∧ x < 3 → quadratic_function a x > 0)
  ↔ (-1 ≤ a ∧ a < 0) ∨ (0 < a ∧ a < 3) := sorry

end quadratic_positivity_range_l223_223739


namespace expression_eq_one_l223_223603

theorem expression_eq_one (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (h : a + b + c = 1) :
  (a^2 * b^2 / ((a^2 - b * c) * (b^2 - a * c)) +
   a^2 * c^2 / ((a^2 - b * c) * (c^2 - a * b)) +
   b^2 * c^2 / ((b^2 - a * c) * (c^2 - a * b))) = 1 := 
by
  sorry

end expression_eq_one_l223_223603


namespace opposite_of_five_l223_223987

theorem opposite_of_five : ∃ y : ℤ, 5 + y = 0 ∧ y = -5 := by
  use -5
  constructor
  . exact rfl
  . sorry

end opposite_of_five_l223_223987


namespace group_purchase_cheaper_l223_223269

-- Define the initial conditions
def initial_price : ℕ := 10
def bulk_price : ℕ := 7
def delivery_cost : ℕ := 100
def group_size : ℕ := 50

-- Define the costs for individual and group purchases
def individual_cost : ℕ := initial_price
def group_cost : ℕ := bulk_price + (delivery_cost / group_size)

-- Statement to prove: cost per participant in a group purchase is less than cost per participant in individual purchases
theorem group_purchase_cheaper : group_cost < individual_cost := by
  sorry

end group_purchase_cheaper_l223_223269


namespace perpendicular_lines_a_eq_1_l223_223939

-- Definitions for the given conditions
def line1 (a : ℝ) (x y : ℝ) : Prop := a * x + y + 3 = 0
def line2 (a : ℝ) (x y : ℝ) : Prop := x + (2 * a - 3) * y = 4

-- Condition that the lines are perpendicular
def perpendicular_lines (a : ℝ) : Prop := a + (2 * a - 3) = 0

-- Proof problem to be solved
theorem perpendicular_lines_a_eq_1 (a : ℝ) (h : perpendicular_lines a) : a = 1 :=
by
  sorry

end perpendicular_lines_a_eq_1_l223_223939


namespace probability_of_multiple_6_or_8_l223_223490

def is_probability_of_multiple_6_or_8 (n : ℕ) : Prop := 
  let num_multiples (k : ℕ) := n / k
  let multiples_6 := num_multiples 6
  let multiples_8 := num_multiples 8
  let multiples_24 := num_multiples 24
  let total_multiples := multiples_6 + multiples_8 - multiples_24
  total_multiples / n = 1 / 4

theorem probability_of_multiple_6_or_8 : is_probability_of_multiple_6_or_8 72 :=
  by sorry

end probability_of_multiple_6_or_8_l223_223490


namespace find_x_l223_223403

theorem find_x (x : ℚ) (h : 2 / 5 = (4 / 3) / x) : x = 10 / 3 :=
by
sorry

end find_x_l223_223403


namespace correct_statement_l223_223913

universe u
variable (α : Type u)

noncomputable def U : Set ℕ := {1, 2, 3, 4, 5}

noncomputable def complement_U_M : Set ℕ := {1, 3}

noncomputable def M : Set ℕ := U α \ complement_U_M

theorem correct_statement : 2 ∈ M :=
by {
  sorry
}

end correct_statement_l223_223913


namespace power_calc_l223_223128

theorem power_calc : (3^2)^4 = 6561 := 
by
  sorry

end power_calc_l223_223128


namespace arrange_in_ascending_order_l223_223337

theorem arrange_in_ascending_order (x : ℝ) (h1 : -1 < x) (h2 : x < 0) : 5 * x < 0.5 * x ∧ 0.5 * x < 5 - x := by
  sorry

end arrange_in_ascending_order_l223_223337


namespace touching_squares_same_color_probability_l223_223023

theorem touching_squares_same_color_probability :
  let m := 0
  let n := 1
  100 * m + n = 1 :=
by
  let m := 0
  let n := 1
  sorry -- Proof is omitted as per instructions

end touching_squares_same_color_probability_l223_223023


namespace silk_original_amount_l223_223956

theorem silk_original_amount (s r : ℕ) (l d x : ℚ)
  (h1 : s = 30)
  (h2 : r = 3)
  (h3 : d = 12)
  (h4 : 30 - 3 = 27)
  (h5 : x / 12 = 30 / 27):
  x = 40 / 3 :=
by
  sorry

end silk_original_amount_l223_223956


namespace depth_notation_l223_223038

theorem depth_notation (x y : ℤ) (hx : x = 9050) (hy : y = -10907) : -y = x :=
by
  sorry

end depth_notation_l223_223038


namespace maximum_possible_value_of_e_l223_223048

noncomputable def b (n : ℕ) : ℤ := (10^n - 2) / 8

def e (n : ℕ) : ℤ := Int.gcd (b n) (b (n + 2))

theorem maximum_possible_value_of_e : ∀ n : ℕ, ∃ k : ℤ, e n = k ∧ k = 1 := by
  intros n
  use 1
  sorry

end maximum_possible_value_of_e_l223_223048


namespace initial_teach_count_l223_223290

theorem initial_teach_count :
  ∃ (x y : ℕ), (x + x * y + (x + x * y) * (y + x * y) = 195) ∧
               (y + x * y + (y + x * y) * (x + x * y) = 192) ∧
               x = 5 ∧ y = 2 :=
by {
  sorry
}

end initial_teach_count_l223_223290


namespace exp_eval_l223_223100

theorem exp_eval : (3^2)^4 = 6561 := by
sorry

end exp_eval_l223_223100


namespace disjoint_subsets_less_elements_l223_223080

open Nat

theorem disjoint_subsets_less_elements (m : ℕ) (A B : Finset ℕ) (hA : A ⊆ Finset.range (m + 1))
  (hB : B ⊆ Finset.range (m + 1)) (h_disjoint : Disjoint A B)
  (h_sum : A.sum id = B.sum id) : ↑(A.card) < m / Real.sqrt 2 ∧ ↑(B.card) < m / Real.sqrt 2 := 
sorry

end disjoint_subsets_less_elements_l223_223080


namespace area_of_triangle_l223_223032

theorem area_of_triangle (A : ℝ) (b : ℝ) (a : ℝ) (hA : A = 60) (hb : b = 4) (ha : a = 2 * Real.sqrt 3) : 
  1 / 2 * a * b * Real.sin (60 * Real.pi / 180) = 2 * Real.sqrt 3 :=
by
  sorry

end area_of_triangle_l223_223032


namespace bus_arrival_time_at_first_station_l223_223970

noncomputable def time_to_first_station (start_time end_time first_station_to_work: ℕ) : ℕ :=
  (end_time - start_time) - first_station_to_work

theorem bus_arrival_time_at_first_station :
  time_to_first_station 360 540 140 = 40 :=
by
  -- provide the proof here, which has been omitted per the instructions
  sorry

end bus_arrival_time_at_first_station_l223_223970


namespace range_of_t_l223_223746

noncomputable def f (a x : ℝ) : ℝ :=
  a / x - x + a * Real.log x

noncomputable def g (a x : ℝ) : ℝ :=
  f a x + 1/2 * x^2 - (a - 1) * x - a / x

theorem range_of_t (a x₁ x₂ t : ℝ) (h1 : f a x₁ = f a x₂) (h2 : x₁ + x₂ = a)
  (h3 : x₁ * x₂ = a) (h4 : a > 4) (h5 : g a x₁ + g a x₂ > t * (x₁ + x₂)) :
  t < Real.log 4 - 3 :=
  sorry

end range_of_t_l223_223746


namespace problem_statement_l223_223448

noncomputable def f : ℝ → ℝ := sorry -- Placeholder for f

-- Theorem stating the axis of symmetry and increasing interval for the transformed function
theorem problem_statement (hf_even : ∀ x, f x = f (-x))
  (hf_increasing : ∀ x₁ x₂, 3 < x₁ → x₁ < x₂ → x₂ < 5 → f x₁ < f x₂) :
  -- For y = f(x - 1), the following holds:
  (∀ x, (f (x - 1)) = f (-(x - 1))) ∧
  (∀ x₁ x₂, 4 < x₁ → x₁ < x₂ → x₂ < 6 → f (x₁ - 1) < f (x₂ - 1)) :=
sorry

end problem_statement_l223_223448


namespace fixed_monthly_fee_l223_223178

theorem fixed_monthly_fee (x y : ℝ)
  (h₁ : x + y = 18.70)
  (h₂ : x + 3 * y = 34.10) : x = 11.00 :=
by sorry

end fixed_monthly_fee_l223_223178


namespace minimum_odd_numbers_in_set_l223_223618

-- Definitions
variable (P : ℝ → ℝ)
variable (degree_P : ℕ)
variable (A_P : Set ℝ)

-- The conditions: P is a polynomial of degree 8, and 8 is included in A_P
def is_polynomial_of_degree_eight (P : ℝ → ℝ) (degree_P : ℕ) : Prop :=
  degree_P = 8

def set_includes_eight (A_P : Set ℝ) : Prop := 
  8 ∈ A_P

-- The goal: prove the minimum number of odd numbers in A_P is 1
theorem minimum_odd_numbers_in_set {P : ℝ → ℝ} {degree_P : ℕ} {A_P : Set ℝ} :
  is_polynomial_of_degree_eight P degree_P → 
  set_includes_eight A_P → 
  ∃ odd_numbers : ℕ, odd_numbers = 1 :=
sorry

end minimum_odd_numbers_in_set_l223_223618


namespace smallest_solution_proof_l223_223310

noncomputable def smallest_solution : ℝ :=
  let n := 11
  let a := 0.533
  n + a

theorem smallest_solution_proof :
  ∃ (x : ℝ), ⌊x^2⌋ - ⌊x⌋^2 = 21 ∧ x = smallest_solution :=
by
  use smallest_solution
  sorry

end smallest_solution_proof_l223_223310


namespace standard_concession_l223_223542

theorem standard_concession (x : ℝ) : 
  (∀ (x : ℝ), (2000 - (x / 100) * 2000) - 0.2 * (2000 - (x / 100) * 2000) = 1120) → x = 30 := 
by 
  sorry

end standard_concession_l223_223542


namespace quadratic_condition_solutions_specific_a_for_x_values_l223_223591

theorem quadratic_condition_solutions:
  ∀ a : ℝ,
  (∀ x : ℝ, (-6 < x ∧ x ≤ -2) → x ≠ -5 ∧ x ≠ -4 ∧ x ≠ -3 →
    (x^2 - (a - 12) * x + 36 - 5 * a = 0)) →
    (a ∈ Ioo 4 (4.5) ∨ a ∈ Ioc 4.5 ((16 : ℝ) / 3)) := by
  sorry

theorem specific_a_for_x_values:
  (∀ x : ℝ, x = -4 → ∃ a : ℝ, a = 4 ∧ (x^2 - (a - 12) * x + 36 - 5 * a = 0)) ∧
  (∀ x : ℝ, x = -3 → ∃ a : ℝ, a = 4.5 ∧ (x^2 - (a - 12) * x + 36 - 5 * a = 0)) := by
  sorry

end quadratic_condition_solutions_specific_a_for_x_values_l223_223591


namespace hyperbola_eccentricity_l223_223456

theorem hyperbola_eccentricity (a b c e : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : a > b)
  (h4 : c = 3 * b) 
  (h5 : c * c = a * a + b * b)
  (h6 : e = c / a) :
  e = 3 * Real.sqrt 2 / 4 :=
by
  sorry

end hyperbola_eccentricity_l223_223456


namespace central_angle_of_sector_l223_223818

theorem central_angle_of_sector (r l : ℝ) (h1 : r = 1) (h2 : l = 4 - 2*r) : 
    ∃ α : ℝ, α = 2 :=
by
  use l / r
  have hr : r = 1 := h1
  have hl : l = 4 - 2*r := h2
  sorry

end central_angle_of_sector_l223_223818


namespace count_four_digit_integers_l223_223460

theorem count_four_digit_integers :
    ∃! (a b c d : ℕ), 1 ≤ a ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧
    (10 * b + c)^2 = (10 * a + b) * (10 * c + d) := sorry

end count_four_digit_integers_l223_223460


namespace correct_statement_l223_223912

universe u
variable (α : Type u)

noncomputable def U : Set ℕ := {1, 2, 3, 4, 5}

noncomputable def complement_U_M : Set ℕ := {1, 3}

noncomputable def M : Set ℕ := U α \ complement_U_M

theorem correct_statement : 2 ∈ M :=
by {
  sorry
}

end correct_statement_l223_223912


namespace average_income_A_B_l223_223494

def monthly_incomes (A B C : ℝ) : Prop :=
  (A = 4000) ∧
  ((B + C) / 2 = 6250) ∧
  ((A + C) / 2 = 5200)

theorem average_income_A_B (A B C X : ℝ) (h : monthly_incomes A B C) : X = 5050 :=
by
  have hA : A = 4000 := h.1
  have hBC : (B + C) / 2 = 6250 := h.2.1
  have hAC : (A + C) / 2 = 5200 := h.2.2
  sorry

end average_income_A_B_l223_223494


namespace problem1_problem2_l223_223276

-- Problem 1
theorem problem1 {a b c : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (1 / a) + (1 / b) + (1 / c) ≥ (1 / (Real.sqrt (a * b))) + (1 / (Real.sqrt (b * c))) + (1 / (Real.sqrt (a * c))) :=
sorry

-- Problem 2
theorem problem2 {x y : ℝ} :
  Real.sin x + Real.sin y ≤ 1 + Real.sin x * Real.sin y :=
sorry

end problem1_problem2_l223_223276


namespace LeonaEarnsGivenHourlyRate_l223_223384

theorem LeonaEarnsGivenHourlyRate :
  (∀ (c: ℝ) (t h e: ℝ), 
    (c = 24.75) → 
    (t = 3) → 
    (h = c / t) → 
    (e = h * 5) →
    e = 41.25) :=
by
  intros c t h e h1 h2 h3 h4
  sorry

end LeonaEarnsGivenHourlyRate_l223_223384


namespace bricks_needed_to_build_wall_l223_223753

def volume_of_brick (length_brick height_brick thickness_brick : ℤ) : ℤ :=
  length_brick * height_brick * thickness_brick

def volume_of_wall (length_wall height_wall thickness_wall : ℤ) : ℤ :=
  length_wall * height_wall * thickness_wall

def number_of_bricks_needed (length_wall height_wall thickness_wall length_brick height_brick thickness_brick : ℤ) : ℤ :=
  (volume_of_wall length_wall height_wall thickness_wall + volume_of_brick length_brick height_brick thickness_brick - 1) / 
  volume_of_brick length_brick height_brick thickness_brick

theorem bricks_needed_to_build_wall : number_of_bricks_needed 800 100 5 25 11 6 = 243 := 
  by 
    sorry

end bricks_needed_to_build_wall_l223_223753


namespace min_cards_to_guarantee_four_same_suit_l223_223762

theorem min_cards_to_guarantee_four_same_suit (n : ℕ) (suits : Fin n) (cards_per_suit : ℕ) (total_cards : ℕ)
  (h1 : n = 4) (h2 : cards_per_suit = 13) : total_cards ≥ 13 :=
by
  sorry

end min_cards_to_guarantee_four_same_suit_l223_223762


namespace division_remainder_l223_223206

-- Define the conditions
def dividend : ℝ := 9087.42
def divisor : ℝ := 417.35
def quotient : ℝ := 21

-- Define the expected remainder
def expected_remainder : ℝ := 323.07

-- Statement of the problem
theorem division_remainder : dividend - divisor * quotient = expected_remainder :=
by
  sorry

end division_remainder_l223_223206


namespace triangle_inequality_satisfied_for_n_six_l223_223719

theorem triangle_inequality_satisfied_for_n_six :
  ∀ (a b c : ℝ), 0 < a → 0 < b → 0 < c → 6 * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2) → 
  (a + b > c ∧ a + c > b ∧ b + c > a) := sorry

end triangle_inequality_satisfied_for_n_six_l223_223719


namespace consecutive_days_without_meeting_l223_223761

/-- In March 1987, there are 31 days, starting on a Sunday.
There are 11 club meetings to be held, and no meetings are on Saturdays or Sundays.
This theorem proves that there will be at least three consecutive days without a meeting. -/
theorem consecutive_days_without_meeting (meetings : Finset ℕ) :
  (∀ x ∈ meetings, 1 ≤ x ∧ x ≤ 31 ∧ ¬ ∃ k, x = 7 * k + 1 ∨ x = 7 * k + 2) →
  meetings.card = 11 →
  ∃ i, 1 ≤ i ∧ i + 2 ≤ 31 ∧ ¬ (i ∈ meetings ∨ (i + 1) ∈ meetings ∨ (i + 2) ∈ meetings) :=
by
  sorry

end consecutive_days_without_meeting_l223_223761


namespace problem_die_rolls_four_times_l223_223691

theorem problem_die_rolls_four_times :
  let total_outcomes := 10000,
      b4 := 1264 in
  let probability := (b4 : ℚ) / total_outcomes in
  let frac := probability.num.gcd probability.denom in
  (probability.num / frac) + (probability.denom / frac) = 2816 :=
by
  sorry

end problem_die_rolls_four_times_l223_223691


namespace trig_identity_simplification_l223_223634

theorem trig_identity_simplification (α : ℝ) :
  (2 * Real.sin (Real.pi - α) + Real.sin (2 * α)) / (2 * (Real.cos (α / 2))^2) = 2 * Real.sin α :=
by sorry

end trig_identity_simplification_l223_223634


namespace work_problem_l223_223668

theorem work_problem (W : ℝ) (A B C : ℝ)
  (h1 : B + C = W / 24)
  (h2 : C + A = W / 12)
  (h3 : C = W / 32) : A + B = W / 16 := 
by
  sorry

end work_problem_l223_223668


namespace range_of_a_l223_223025

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, 0 < x ∧ x < 3 ↔ x > Real.log a / Real.log 2) → 0 < a ∧ a ≤ 1 := 
by 
  sorry

end range_of_a_l223_223025


namespace intersection_complement_l223_223451

noncomputable def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x^2 - 2 * x - 3 > 0}
def B : Set ℝ := {x | 2 < x ∧ x < 4}
def complement_U (A : Set ℝ) : Set ℝ := {x | ¬ (A x)}

theorem intersection_complement (U : Set ℝ) (A : Set ℝ) (B : Set ℝ) :
  B ∩ (complement_U A) = {x | 2 < x ∧ x ≤ 3} := by
  sorry

end intersection_complement_l223_223451


namespace area_of_circle_eq_sixteen_pi_l223_223843

theorem area_of_circle_eq_sixteen_pi :
  ∃ (x y : ℝ), (x^2 + y^2 - 8*x + 6*y = -9) ↔ (π * 4^2 = 16 * π) :=
by
  sorry

end area_of_circle_eq_sixteen_pi_l223_223843


namespace cheenu_time_difference_l223_223399

-- Define the conditions in terms of Cheenu's activities

variable (boy_run_distance : ℕ) (boy_run_time : ℕ)
variable (midage_bike_distance : ℕ) (midage_bike_time : ℕ)
variable (old_walk_distance : ℕ) (old_walk_time : ℕ)

-- Define the problem with these variables
theorem cheenu_time_difference:
    boy_run_distance = 20 ∧ boy_run_time = 240 ∧
    midage_bike_distance = 30 ∧ midage_bike_time = 120 ∧
    old_walk_distance = 8 ∧ old_walk_time = 240 →
    (old_walk_time / old_walk_distance - midage_bike_time / midage_bike_distance) = 26 := by
    sorry

end cheenu_time_difference_l223_223399


namespace max_value_of_expr_l223_223049

noncomputable theory

open Real

/-- Let x be a positive real number. Prove that the maximum possible value of 
    (x² + 3 - sqrt(x⁴ + 9)) / x is 3 - sqrt(6). -/
theorem max_value_of_expr (x : ℝ) (hx : 0 < x) :
  Sup {y : ℝ | ∃ x : ℝ, 0 < x ∧ y = (x^2 + 3 - sqrt (x^4 + 9)) / x} = 3 - sqrt 6 :=
sorry

end max_value_of_expr_l223_223049


namespace least_number_of_cans_l223_223852

theorem least_number_of_cans (maaza pepsi sprite : ℕ) (h_maaza : maaza = 80) (h_pepsi : pepsi = 144) (h_sprite : sprite = 368) :
  ∃ n, n = 37 := sorry

end least_number_of_cans_l223_223852


namespace deer_families_stayed_l223_223075

-- Define the initial number of deer families
def initial_deer_families : ℕ := 79

-- Define the number of deer families that moved out
def moved_out_deer_families : ℕ := 34

-- The theorem stating how many deer families stayed
theorem deer_families_stayed : initial_deer_families - moved_out_deer_families = 45 :=
by
  -- Proof will be provided here
  sorry

end deer_families_stayed_l223_223075


namespace problem_solution_l223_223927

universe u

variable (U : Set Nat) (M : Set Nat)
variable (complement_U_M : Set Nat)

axiom U_def : U = {1, 2, 3, 4, 5}
axiom complement_U_M_def : complement_U_M = {1, 3}
axiom M_def : M = U \ complement_U_M

theorem problem_solution : 2 ∈ M := by
  sorry

end problem_solution_l223_223927


namespace roots_sum_eq_product_l223_223900

theorem roots_sum_eq_product (m : ℝ) :
  (∀ x : ℝ, 2 * (x - 1) * (x - 3 * m) = x * (m - 4)) →
  (∀ a b : ℝ, 2 * a * b = 2 * (5 * m + 6) / -2 ∧ 2 * a * b = 6 * m / 2) →
  m = -2 / 3 :=
by
  sorry

end roots_sum_eq_product_l223_223900


namespace pedestrian_avg_waiting_time_at_traffic_light_l223_223418

theorem pedestrian_avg_waiting_time_at_traffic_light :
  ∀ cycle_time green_time red_time : ℕ,
  cycle_time = green_time + red_time →
  green_time = 1 →
  red_time = 2 →
  let prob_green := green_time / cycle_time in
  let prob_red := red_time / cycle_time in
  let E_T_given_green := 0 in
  let E_T_given_red := (0 + 2) / 2 in
  (E_T_given_green * prob_green + E_T_given_red * prob_red) * 60 = 40 := 
by
  -- Insert proof here
  sorry

end pedestrian_avg_waiting_time_at_traffic_light_l223_223418


namespace vasya_max_points_l223_223485

theorem vasya_max_points (cards : Finset (Fin 36)) 
  (petya_hand vasya_hand : Finset (Fin 36)) 
  (h_disjoint : Disjoint petya_hand vasya_hand)
  (h_union : petya_hand ∪ vasya_hand = cards)
  (h_card : cards.card = 36)
  (h_half : petya_hand.card = 18 ∧ vasya_hand.card = 18) : 
  ∃ max_points : ℕ, max_points = 15 := 
sorry

end vasya_max_points_l223_223485


namespace lcm_3_4_6_15_l223_223256

noncomputable def lcm_is_60 : ℕ := 60

theorem lcm_3_4_6_15 : lcm (lcm (lcm 3 4) 6) 15 = lcm_is_60 := 
by 
    sorry

end lcm_3_4_6_15_l223_223256


namespace farmer_apples_count_l223_223237

-- Definitions from the conditions in step a)
def initial_apples : ℕ := 127
def apples_given_away : ℕ := 88

-- Proof goal from step c)
theorem farmer_apples_count : initial_apples - apples_given_away = 39 :=
by
  sorry

end farmer_apples_count_l223_223237


namespace inequality_always_holds_l223_223897

theorem inequality_always_holds (a b c : ℝ) (h : a > b) : (a - b) * c^2 ≥ 0 :=
sorry

end inequality_always_holds_l223_223897


namespace total_fruits_in_baskets_l223_223639

structure Baskets where
  mangoes : ℕ
  pears : ℕ
  pawpaws : ℕ
  kiwis : ℕ
  lemons : ℕ

def taniaBaskets : Baskets := {
  mangoes := 18,
  pears := 10,
  pawpaws := 12,
  kiwis := 9,
  lemons := 9
}

theorem total_fruits_in_baskets : taniaBaskets.mangoes + taniaBaskets.pears + taniaBaskets.pawpaws + taniaBaskets.kiwis + taniaBaskets.lemons = 58 :=
by
  sorry

end total_fruits_in_baskets_l223_223639


namespace total_volume_l223_223221

-- Defining the volumes for different parts as per the conditions.
variables (V_A V_C V_B' V_C' : ℝ)
variables (V : ℝ)

-- The given conditions
axiom V_A_eq_40 : V_A = 40
axiom V_C_eq_300 : V_C = 300
axiom V_B'_eq_360 : V_B' = 360
axiom V_C'_eq_90 : V_C' = 90

-- The proof goal: total volume of the parallelepiped
theorem total_volume (V_A V_C V_B' V_C' : ℝ) 
  (V_A_eq_40 : V_A = 40) (V_C_eq_300 : V_C = 300) 
  (V_B'_eq_360 : V_B' = 360) (V_C'_eq_90 : V_C' = 90) :
  V = V_A + V_C + V_B' + V_C' :=
by
  sorry

end total_volume_l223_223221


namespace parallel_lines_solution_l223_223030

theorem parallel_lines_solution (a : ℝ) :
  (∀ x y : ℝ, (1 + a) * x + y + 1 = 0 → 2 * x + a * y + 2 = 0 → (a = 1 ∨ a = -2)) :=
by
  sorry

end parallel_lines_solution_l223_223030


namespace correctStatement_l223_223923

variable (U : Set ℕ) (M : Set ℕ)

namespace Proof

-- Given conditions
def universalSet := {1, 2, 3, 4, 5}
def complementM := {1, 3}
def isComplement (M : Set ℕ) : Prop := U \ M = complementM

-- Target statement to be proved
theorem correctStatement (h1 : U = universalSet) (h2 : isComplement M) : 2 ∈ M := by
  sorry

end Proof

end correctStatement_l223_223923


namespace circle_passes_through_fixed_point_l223_223011

theorem circle_passes_through_fixed_point :
  ∀ (C : ℝ × ℝ), (C.2 ^ 2 = 4 * C.1) ∧ (C.1 = -1 + (C.1 + 1)) → ∃ P : ℝ × ℝ, P = (1, 0) ∧
    (P.1 - C.1) ^ 2 + (P.2 - C.2) ^ 2 = (C.1 + 1) ^ 2 + (0 - C.2) ^ 2 :=
by
  sorry

end circle_passes_through_fixed_point_l223_223011


namespace ones_digit_of_power_35_35_pow_17_17_is_five_l223_223562

theorem ones_digit_of_power_35_35_pow_17_17_is_five :
  (35 ^ (35 * (17 ^ 17))) % 10 = 5 := by
  sorry

end ones_digit_of_power_35_35_pow_17_17_is_five_l223_223562


namespace question2_l223_223573

noncomputable def a (n : ℕ) : ℕ :=
  2^n

noncomputable def b (n : ℕ) : ℕ :=
  4^n - 2^n

noncomputable def S (n : ℕ) : ℕ :=
  2^n + 4 * (4^n - 2^n)

noncomputable def P (n : ℕ) : ℚ :=
  a n / S n

theorem question2 (n : ℕ) : ∑ i in range n, P (i+1) < 3/2 := by
  sorry

end question2_l223_223573


namespace area_enclosed_by_region_l223_223838

theorem area_enclosed_by_region : ∀ (x y : ℝ), (x^2 + y^2 - 8*x + 6*y = -9) → (π * (4 ^ 2) = 16 * π) :=
by
  intro x y h
  sorry

end area_enclosed_by_region_l223_223838


namespace final_digit_is_two_l223_223372

-- Define initial conditions
def initial_ones : ℕ := 10
def initial_twos : ℕ := 10

-- Define the possible moves and the parity properties
def erase_identical (ones twos : ℕ) : ℕ × ℕ :=
  if ones ≥ 2 then (ones - 2, twos + 1)
  else (ones, twos - 1) -- for the case where two twos are removed

def erase_different (ones twos : ℕ) : ℕ × ℕ :=
  (ones, twos - 1)

-- Theorem stating that the final digit must be a two
theorem final_digit_is_two : 
∀ (ones twos : ℕ), ones = initial_ones → twos = initial_twos → 
(∃ n, ones + twos = n ∧ n = 1 ∧ (ones % 2 = 0)) → 
(∃ n, ones + twos = n ∧ n = 0 ∧ twos = 1) := 
by
  intros ones twos h_ones h_twos condition
  -- Constructing the proof should be done here
  sorry

end final_digit_is_two_l223_223372


namespace chord_length_sqrt_10_l223_223326

/-
  Given a line L: 3x - y - 6 = 0 and a circle C: x^2 + y^2 - 2x - 4y = 0,
  prove that the length of the chord AB formed by their intersection is sqrt(10).
-/

noncomputable def line_L : Set (ℝ × ℝ) := {p | ∃ x y, p = (x, y) ∧ 3 * x - y - 6 = 0}

noncomputable def circle_C : Set (ℝ × ℝ) := {p | ∃ x y, p = (x, y) ∧ x^2 + y^2 - 2 * x - 4 * y = 0}

noncomputable def chord_length (L C : Set (ℝ × ℝ)) : ℝ :=
  let center := (1, 2)
  let r := Real.sqrt 5
  let d := |3 * 1 - 2 - 6| / Real.sqrt (1 + 3^2)
  2 * Real.sqrt (r^2 - d^2)

theorem chord_length_sqrt_10 : chord_length line_L circle_C = Real.sqrt 10 := sorry

end chord_length_sqrt_10_l223_223326


namespace strictly_increasing_difference_l223_223238

variable {a b : ℝ}
variable {f g : ℝ → ℝ}

theorem strictly_increasing_difference
  (h_diff : ∀ x ∈ Set.Icc a b, DifferentiableAt ℝ f x ∧ DifferentiableAt ℝ g x)
  (h_eq : f a = g a)
  (h_diff_ineq : ∀ x ∈ Set.Ioo a b, (deriv f x : ℝ) > (deriv g x : ℝ)) :
  ∀ x ∈ Set.Ioo a b, f x > g x := by
  sorry

end strictly_increasing_difference_l223_223238


namespace exp_eval_l223_223102

theorem exp_eval : (3^2)^4 = 6561 := by
sorry

end exp_eval_l223_223102


namespace sum_divisors_of_24_is_60_and_not_prime_l223_223176

def divisors (n : Nat) : List Nat :=
  List.filter (λ d => n % d = 0) (List.range (n + 1))

def sum_divisors (n : Nat) : Nat :=
  (divisors n).sum

def is_prime (n : Nat) : Bool :=
  n > 1 ∧ (List.filter (λ d => d > 1 ∧ d < n ∧ n % d = 0) (List.range (n + 1))).length = 0

theorem sum_divisors_of_24_is_60_and_not_prime :
  sum_divisors 24 = 60 ∧ ¬ is_prime 60 := 
by
  sorry

end sum_divisors_of_24_is_60_and_not_prime_l223_223176


namespace exponentiation_example_l223_223115

theorem exponentiation_example : (3^2)^4 = 6561 := by
  sorry

end exponentiation_example_l223_223115


namespace minimum_odd_numbers_in_A_P_l223_223615

-- Polynomials and assumptions
def degree (P : Polynomial ℝ) : ℕ := P.natDegree

-- The set A_P is defined as the set of all numbers x for which P(x) gives a certain value
def A_P (P : Polynomial ℝ) : Set ℝ := {x : ℝ | P.eval x = P.eval 8}

-- Define the main theorem statement
theorem minimum_odd_numbers_in_A_P (P : Polynomial ℝ) (hdeg : degree P = 8) (h8 : 8 ∈ A_P P) : 
  ∃ n, n = 1 ∧ ∃ (x : ℝ), x ∈ A_P P ∧ ¬ (x % 2 = 0) := sorry

end minimum_odd_numbers_in_A_P_l223_223615


namespace exponent_proof_l223_223318

theorem exponent_proof (n m : ℕ) (h1 : 4^n = 3) (h2 : 8^m = 5) : 2^(2*n + 3*m) = 15 :=
by
  -- Proof steps
  sorry

end exponent_proof_l223_223318


namespace problem1_problem2_l223_223277

-- Define the first problem
theorem problem1 (x : ℝ) : (x - 2) ^ 2 = 2 * x - 4 ↔ (x = 2 ∨ x = 4) := 
by 
  sorry

-- Define the second problem using completing the square method
theorem problem2 (x : ℝ) : x ^ 2 - 4 * x - 1 = 0 ↔ (x = 2 + Real.sqrt 5 ∨ x = 2 - Real.sqrt 5) := 
by 
  sorry

end problem1_problem2_l223_223277


namespace c_share_of_profit_l223_223147

theorem c_share_of_profit
  (a_investment : ℝ)
  (b_investment : ℝ)
  (c_investment : ℝ)
  (total_profit : ℝ)
  (ha : a_investment = 30000)
  (hb : b_investment = 45000)
  (hc : c_investment = 50000)
  (hp : total_profit = 90000) :
  (c_investment / (a_investment + b_investment + c_investment)) * total_profit = 36000 := 
by
  sorry

end c_share_of_profit_l223_223147


namespace smallest_lcm_l223_223347

theorem smallest_lcm (k l : ℕ) (hk : k ≥ 1000) (hl : l ≥ 1000) (huk : k < 10000) (hul : l < 10000) (hk_pos : 0 < k) (hl_pos : 0 < l) (h_gcd: Nat.gcd k l = 5) :
  Nat.lcm k l = 201000 :=
by
  sorry

end smallest_lcm_l223_223347


namespace license_plates_count_l223_223538

theorem license_plates_count : (6 * 10^5 * 26^3) = 10584576000 := by
  sorry

end license_plates_count_l223_223538


namespace sum_arithmetic_seq_nine_terms_l223_223190

theorem sum_arithmetic_seq_nine_terms
  (a_n : ℕ → ℝ)
  (S_n : ℕ → ℝ)
  (k : ℝ)
  (h1 : ∀ n, a_n n = k * n + 4 - 5 * k)
  (h2 : ∀ n, S_n n = (n / 2) * (a_n 1 + a_n n))
  : S_n 9 = 36 :=
sorry

end sum_arithmetic_seq_nine_terms_l223_223190


namespace ink_length_figure_4_ink_length_difference_9_8_ink_length_figure_100_l223_223411

-- Define the ink length of a figure
def ink_length (n : ℕ) : ℕ := 5 * n

-- Part (a): Determine the ink length of Figure 4.
theorem ink_length_figure_4 : ink_length 4 = 20 := by
  sorry

-- Part (b): Determine the difference between the ink length of Figure 9 and the ink length of Figure 8.
theorem ink_length_difference_9_8 : ink_length 9 - ink_length 8 = 5 := by
  sorry

-- Part (c): Determine the ink length of Figure 100.
theorem ink_length_figure_100 : ink_length 100 = 500 := by
  sorry

end ink_length_figure_4_ink_length_difference_9_8_ink_length_figure_100_l223_223411


namespace total_games_played_l223_223076

def number_of_games (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem total_games_played :
  number_of_games 9 2 = 36 :=
by
  -- Proof to be filled in later
  sorry

end total_games_played_l223_223076


namespace min_odd_numbers_in_A_P_l223_223608

theorem min_odd_numbers_in_A_P (P : ℝ → ℝ) 
  (hP : ∃ a8 a7 a6 a5 a4 a3 a2 a1 a0 : ℝ, P = λ x, a8 * x^8 + a7 * x^7 + a6 * x^6 + a5 * x^5 + a4 * x^4 + a3 * x^3 + a2 * x^2 + a1 * x + a0)
  (h8_in_A_P : ∃ x : ℝ, P x = 8) : 
  (∀ x : ℝ, P x ∈ Set.A_P) → 
  (∃! y : ℝ, odd y ∧ y ∈ Set.A_P) := 
sorry

end min_odd_numbers_in_A_P_l223_223608


namespace lcm_of_three_numbers_is_180_l223_223500

-- Define the three numbers based on the ratio and HCF condition
def a : ℕ := 2 * 6
def b : ℕ := 3 * 6
def c : ℕ := 5 * 6

-- State the theorem regarding the LCM
theorem lcm_of_three_numbers_is_180 : Nat.lcm (Nat.lcm a b) c = 180 :=
by
  sorry

end lcm_of_three_numbers_is_180_l223_223500


namespace lemon_count_l223_223636

theorem lemon_count {total_fruits mangoes pears pawpaws : ℕ} (kiwi lemon : ℕ) :
  total_fruits = 58 ∧ 
  mangoes = 18 ∧ 
  pears = 10 ∧ 
  pawpaws = 12 ∧ 
  (kiwi = lemon) →
  lemon = 9 :=
by 
  sorry

end lemon_count_l223_223636


namespace susan_min_packages_l223_223976

theorem susan_min_packages (n : ℕ) (cost_per_package : ℕ := 5) (earnings_per_package : ℕ := 15) (initial_cost : ℕ := 1200) :
  15 * n - 5 * n ≥ 1200 → n ≥ 120 :=
by {
  sorry -- Proof goes here
}

end susan_min_packages_l223_223976


namespace power_calc_l223_223124

theorem power_calc : (3^2)^4 = 6561 := 
by
  sorry

end power_calc_l223_223124


namespace boris_stopped_saving_in_may_2020_l223_223174

theorem boris_stopped_saving_in_may_2020 :
  ∀ (B V : ℕ) (start_date_B start_date_V stop_date : ℕ), 
    (∀ t, start_date_B + t ≤ stop_date → B = 200 * t) →
    (∀ t, start_date_V + t ≤ stop_date → V = 300 * t) → 
    V = 6 * B →
    stop_date = 17 → 
    B / 200 = 4 → 
    stop_date - B/200 = 2020 * 12 + 5 :=
by
  sorry

end boris_stopped_saving_in_may_2020_l223_223174


namespace exponentiation_example_l223_223119

theorem exponentiation_example : (3^2)^4 = 6561 := by
  sorry

end exponentiation_example_l223_223119


namespace tim_watched_total_hours_tv_l223_223654

-- Define the conditions
def short_show_episodes : ℕ := 24
def short_show_duration_per_episode : ℝ := 0.5

def long_show_episodes : ℕ := 12
def long_show_duration_per_episode : ℝ := 1

-- Define the total duration for each show
def short_show_total_duration : ℝ :=
  short_show_episodes * short_show_duration_per_episode

def long_show_total_duration : ℝ :=
  long_show_episodes * long_show_duration_per_episode

-- Define the total TV hours watched
def total_tv_hours_watched : ℝ :=
  short_show_total_duration + long_show_total_duration

-- Write the theorem statement
theorem tim_watched_total_hours_tv : total_tv_hours_watched = 24 := 
by
  -- proof goes here
  sorry

end tim_watched_total_hours_tv_l223_223654


namespace band_first_set_songs_count_l223_223501

theorem band_first_set_songs_count 
  (total_repertoire : ℕ) (second_set : ℕ) (encore : ℕ) (avg_third_fourth : ℕ)
  (h_total_repertoire : total_repertoire = 30)
  (h_second_set : second_set = 7)
  (h_encore : encore = 2)
  (h_avg_third_fourth : avg_third_fourth = 8)
  : ∃ (x : ℕ), x + second_set + encore + avg_third_fourth * 2 = total_repertoire := 
  sorry

end band_first_set_songs_count_l223_223501


namespace min_colors_required_l223_223662

-- Defining the color type
def Color := ℕ

-- Defining a 6x6 grid
def Grid := Fin 6 → Fin 6 → Color

-- Defining the conditions of the problem for a valid coloring
def is_valid_coloring (c : Grid) : Prop :=
  (∀ i j k, i ≠ j → c i k ≠ c j k) ∧ -- each row has all cells with different colors
  (∀ i j k, i ≠ j → c k i ≠ c k j) ∧ -- each column has all cells with different colors
  (∀ i j, i ≠ j → c i (i+j) ≠ c j (i+j)) ∧ -- each 45° diagonal has all different colors
  (∀ i j, i ≠ j → (i-j ≥ 0 → c (i-j) i ≠ c (i-j) j) ∧ (j-i ≥ 0 → c i (j-i) ≠ c j (j-i))) -- each 135° diagonal has all different colors

-- The formal statement of the math problem
theorem min_colors_required : ∃ (n : ℕ), (∀ c : Grid, is_valid_coloring c → n ≥ 7) :=
sorry

end min_colors_required_l223_223662


namespace isosceles_base_length_l223_223799

theorem isosceles_base_length (b : ℝ) (h1 : 7 + 7 + b = 23) : b = 9 :=
sorry

end isosceles_base_length_l223_223799


namespace symmetric_about_origin_implies_odd_l223_223012

variable {F : Type} [Field F] (f : F → F)
variable (x : F)

theorem symmetric_about_origin_implies_odd (H : ∀ x, f (-x) = -f x) : f x + f (-x) = 0 := 
by 
  sorry

end symmetric_about_origin_implies_odd_l223_223012


namespace smallest_s_for_F_l223_223706

def F (a b c d : ℕ) : ℕ := a * b^(c^d)

theorem smallest_s_for_F :
  ∃ s : ℕ, F s s 2 2 = 65536 ∧ ∀ t : ℕ, F t t 2 2 = 65536 → s ≤ t :=
sorry

end smallest_s_for_F_l223_223706


namespace hannah_stocking_stuffers_l223_223946

theorem hannah_stocking_stuffers (candy_caness : ℕ) (beanie_babies : ℕ) (books : ℕ) (kids : ℕ) : 
  candy_caness = 4 → 
  beanie_babies = 2 → 
  books = 1 → 
  kids = 3 → 
  candy_caness + beanie_babies + books = 7 → 
  7 * kids = 21 := 
by sorry

end hannah_stocking_stuffers_l223_223946


namespace num_natural_a_l223_223558

theorem num_natural_a (a b : ℕ) : 
  (a^2 + a + 100 = b^2) → ∃ n : ℕ, n = 4 := sorry

end num_natural_a_l223_223558


namespace simplify_and_evaluate_l223_223633

noncomputable def simplifyExpression (a : ℚ) : ℚ :=
  (a - 3 + (1 / (a - 1))) / ((a^2 - 4) / (a^2 + 2*a)) * (1 / (a - 2))

theorem simplify_and_evaluate
  (h : ∀ a, a ∈ [-2, -1, 0, 1, 2]) :
  ∀ a, (a - 1) ≠ 0 → a ≠ 0 → a ≠ 2  →
  simplifyExpression a = a / (a - 1) ∧ simplifyExpression (-1) = 1 / 2 :=
by
  intro a ha_ne_zero ha_ne_two
  sorry

end simplify_and_evaluate_l223_223633


namespace smallest_sum_twice_perfect_square_l223_223249

-- Definitions based directly on conditions:
def sum_of_20_consecutive_integers (n : ℕ) : ℕ := (2 * n + 19) * 10

def twice_perfect_square (x : ℕ) : Prop := ∃ m : ℕ, x = 2 * m^2

-- Proposition to prove the smallest possible value satisfying these conditions:
theorem smallest_sum_twice_perfect_square : 
  ∃ n S, S = sum_of_20_consecutive_integers n ∧ twice_perfect_square S ∧ S = 450 :=
begin
  sorry
end

end smallest_sum_twice_perfect_square_l223_223249


namespace stocking_stuffers_total_l223_223945

theorem stocking_stuffers_total 
  (candy_canes_per_child beanie_babies_per_child books_per_child : ℕ)
  (num_children : ℕ)
  (h1 : candy_canes_per_child = 4)
  (h2 : beanie_babies_per_child = 2)
  (h3 : books_per_child = 1)
  (h4 : num_children = 3) :
  candy_canes_per_child + beanie_babies_per_child + books_per_child * num_children = 21 :=
by
  sorry

end stocking_stuffers_total_l223_223945


namespace smallest_arithmetic_geometric_seq_sum_l223_223644

variable (A B C D : ℕ)

noncomputable def arithmetic_seq (A B C : ℕ) (d : ℕ) : Prop :=
  B - A = d ∧ C - B = d

noncomputable def geometric_seq (B C D : ℕ) : Prop :=
  C = (5 / 3) * B ∧ D = (25 / 9) * B

theorem smallest_arithmetic_geometric_seq_sum :
  ∃ A B C D : ℕ, 
    arithmetic_seq A B C 12 ∧ 
    geometric_seq B C D ∧ 
    (A + B + C + D = 104) :=
sorry

end smallest_arithmetic_geometric_seq_sum_l223_223644


namespace gideon_fraction_of_marbles_l223_223000

variable (f : ℝ)

theorem gideon_fraction_of_marbles (marbles : ℝ) (age_now : ℝ) (age_future : ℝ) (remaining_marbles : ℝ) (future_age_with_remaining_marbles : Bool)
  (h1 : marbles = 100)
  (h2 : age_now = 45)
  (h3 : age_future = age_now + 5)
  (h4 : remaining_marbles = 2 * (1 - f) * marbles)
  (h5 : remaining_marbles = age_future)
  (h6 : future_age_with_remaining_marbles = (age_future = 50)) :
  f = 3 / 4 :=
by
  sorry

end gideon_fraction_of_marbles_l223_223000


namespace isosceles_base_length_l223_223798

theorem isosceles_base_length (b : ℝ) (h1 : 7 + 7 + b = 23) : b = 9 :=
sorry

end isosceles_base_length_l223_223798


namespace sum_of_three_integers_with_product_5_pow_4_l223_223813

noncomputable def a : ℕ := 1
noncomputable def b : ℕ := 5
noncomputable def c : ℕ := 125

theorem sum_of_three_integers_with_product_5_pow_4 (h : a * b * c = 5^4) : 
  a + b + c = 131 := by
  have ha : a = 1 := rfl
  have hb : b = 5 := rfl
  have hc : c = 125 := rfl
  rw [ha, hb, hc, mul_assoc] at h
  exact sorry

end sum_of_three_integers_with_product_5_pow_4_l223_223813


namespace triangle_angles_l223_223971

theorem triangle_angles (A B C : ℝ) (h1 : A + B + C = 180) (h2 : A = 45) : B = 90 ∧ C = 45 :=
sorry

end triangle_angles_l223_223971


namespace necessary_sufficient_condition_geometric_sequence_l223_223192

noncomputable def an_geometric (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = r * a n

theorem necessary_sufficient_condition_geometric_sequence
  (a : ℕ → ℝ) (S : ℕ → ℝ) (p q : ℝ) (h_sum : ∀ n : ℕ, S (n + 1) = S n + a (n + 1))
  (h_eq : ∀ n : ℕ, a (n + 1) = p * S n + q) :
  (a 1 = q) ↔ (∃ r : ℝ, an_geometric a r) :=
sorry

end necessary_sufficient_condition_geometric_sequence_l223_223192


namespace f_properties_l223_223263

open Real

-- Define the function f(x) = x^2
noncomputable def f (x : ℝ) : ℝ := x^2

-- Define the statement to be proved
theorem f_properties (x₁ x₂ : ℝ) (x : ℝ) (h : 0 < x) :
  (f (x₁ * x₂) = f x₁ * f x₂) ∧ 
  (deriv f x > 0) ∧
  (∀ x : ℝ, deriv f (-x) = -deriv f x) :=
by
  sorry

end f_properties_l223_223263


namespace area_of_triangle_EOF_correct_l223_223240

noncomputable def area_of_triangle_eof : ℝ :=
  let line := λ x y : ℝ, x - 2 * y - 3 = 0
  let circle := λ x y : ℝ, (x - 2)^2 + (y + 3)^2 = 9
  let O := (0 : ℝ, 0 : ℝ)
  let E := sorry  -- placeholder for the intersection point
  let F := sorry  -- placeholder for the intersection point
  let OE := real.sqrt ((E.1 - O.1)^2 + (E.2 - O.2)^2)
  let OF := real.sqrt ((F.1 - O.1)^2 + (F.2 - O.2)^2)
  let EF := real.sqrt ((F.1 - E.1)^2 + (F.2 - E.2)^2)
  let p := (OE + OF + EF) / 2
  real.sqrt (p * (p - OE) * (p - OF) * (p - EF))

theorem area_of_triangle_EOF_correct : area_of_triangle_eof = 6 * real.sqrt 5 / 5 := sorry

end area_of_triangle_EOF_correct_l223_223240


namespace question_l223_223933

variable (U : Set ℕ) (M : Set ℕ)

theorem question :
  U = {1, 2, 3, 4, 5} →
  (U \ M = {1, 3}) →
  2 ∈ M :=
by
  intros
  sorry

end question_l223_223933


namespace hyperbola_asymptote_b_value_l223_223747

theorem hyperbola_asymptote_b_value (b : ℝ) (hb : 0 < b) : 
  (∀ x y, x^2 - y^2 / b^2 = 1 → y = 3 * x ∨ y = -3 * x) → b = 3 := 
by
  sorry

end hyperbola_asymptote_b_value_l223_223747


namespace geometric_sequence_sum_l223_223574

variable {a : ℕ → ℝ} -- Sequence terms
variable {S : ℕ → ℝ} -- Sum of the first n terms

-- Conditions
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) := ∀ n, a (n+1) = a n * q
def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) := ∀ n, S n = a 0 * (1 - (a n)) / (1 - a 1)
def is_arithmetic_sequence (x y z : ℝ) := 2 * y = x + z
def term_1_equals_1 (a : ℕ → ℝ) := a 0 = 1

-- Question: Prove that given the conditions, S_5 = 31
theorem geometric_sequence_sum (q : ℝ) (h_geom : is_geometric_sequence a q) 
  (h_sum : sum_of_first_n_terms a S) (h_arith : is_arithmetic_sequence (4 * a 0) (2 * a 1) (a 2)) 
  (h_a1 : term_1_equals_1 a) : S 5 = 31 :=
sorry

end geometric_sequence_sum_l223_223574


namespace painting_combinations_l223_223410

-- Define the conditions and the problem statement
def top_row_paint_count := 2
def total_lockers_per_row := 4
def valid_paintings := Nat.choose total_lockers_per_row top_row_paint_count

theorem painting_combinations : valid_paintings = 6 := by
  -- Use the derived conditions to provide the proof
  sorry

end painting_combinations_l223_223410


namespace trig_identity_cos_sin_l223_223437

theorem trig_identity_cos_sin : 
  (Real.cos (π / 12))^2 - (Real.sin (π / 12))^2 = Real.cos (π / 6) :=
sorry

end trig_identity_cos_sin_l223_223437


namespace total_stocking_stuffers_total_stocking_stuffers_hannah_buys_l223_223948

def candy_canes_per_kid : ℕ := 4
def beanie_babies_per_kid : ℕ := 2
def books_per_kid : ℕ := 1
def kids : ℕ := 3

theorem total_stocking_stuffers : candy_canes_per_kid + beanie_babies_per_kid + books_per_kid = 7 :=
by { 
  -- by trusted computation
  sorry
}

theorem total_stocking_stuffers_hannah_buys : 3 * (candy_canes_per_kid + beanie_babies_per_kid + books_per_kid) = 21 :=
by {
  have h : candy_canes_per_kid + beanie_babies_per_kid + books_per_kid = 7 := total_stocking_stuffers,
  rw h,
  norm_num,
}

end total_stocking_stuffers_total_stocking_stuffers_hannah_buys_l223_223948


namespace arithmetic_sequence_a2_a9_l223_223330

theorem arithmetic_sequence_a2_a9 (a : ℕ → ℚ) (d : ℚ)
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_sum : a 5 + a 6 = 12) :
  a 2 + a 9 = 12 :=
sorry

end arithmetic_sequence_a2_a9_l223_223330


namespace marbles_count_l223_223534

theorem marbles_count (M : ℕ)
  (h_blue : (M / 2) = n_blue)
  (h_red : (M / 4) = n_red)
  (h_green : 27 = n_green)
  (h_yellow : 14 = n_yellow)
  (h_total : (n_blue + n_red + n_green + n_yellow) = M) :
  M = 164 :=
by
  sorry

end marbles_count_l223_223534


namespace max_value_of_cubes_l223_223368

theorem max_value_of_cubes 
  (x y z : ℝ) 
  (h : x^2 + y^2 + z^2 = 9) : 
  x^3 + y^3 + z^3 ≤ 27 :=
  sorry

end max_value_of_cubes_l223_223368


namespace isosceles_base_length_l223_223797

theorem isosceles_base_length (b : ℝ) (h1 : 7 + 7 + b = 23) : b = 9 :=
sorry

end isosceles_base_length_l223_223797


namespace remainder_8357_to_8361_div_9_l223_223309

theorem remainder_8357_to_8361_div_9 :
  (8357 + 8358 + 8359 + 8360 + 8361) % 9 = 3 := 
by
  sorry

end remainder_8357_to_8361_div_9_l223_223309


namespace volume_ratio_of_sphere_surface_area_l223_223020

theorem volume_ratio_of_sphere_surface_area 
  {V1 V2 V3 : ℝ} 
  (h : V1/V3 = 1/27 ∧ V2/V3 = 8/27) 
  : V1 + V2 = (1/3) * V3 := 
sorry

end volume_ratio_of_sphere_surface_area_l223_223020


namespace intersection_eq_l223_223329

open Set

variable {α : Type*}

def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

theorem intersection_eq : M ∩ N = {2, 3} := by
  apply Set.ext
  intro x
  simp [M, N]
  sorry

end intersection_eq_l223_223329


namespace sara_initial_pears_l223_223230

theorem sara_initial_pears (given_to_dan : ℕ) (left_with_sara : ℕ) (total : ℕ) :
  given_to_dan = 28 ∧ left_with_sara = 7 ∧ total = given_to_dan + left_with_sara → total = 35 :=
by
  sorry

end sara_initial_pears_l223_223230


namespace correctStatement_l223_223920

variable (U : Set ℕ) (M : Set ℕ)

namespace Proof

-- Given conditions
def universalSet := {1, 2, 3, 4, 5}
def complementM := {1, 3}
def isComplement (M : Set ℕ) : Prop := U \ M = complementM

-- Target statement to be proved
theorem correctStatement (h1 : U = universalSet) (h2 : isComplement M) : 2 ∈ M := by
  sorry

end Proof

end correctStatement_l223_223920


namespace proof_2_in_M_l223_223936

def U : Set ℕ := {1, 2, 3, 4, 5}

def M : Set ℕ := { x | x ∈ U ∧ x ≠ 1 ∧ x ≠ 3 }

theorem proof_2_in_M : 2 ∈ M :=
by
  sorry

end proof_2_in_M_l223_223936


namespace Paula_needs_52_tickets_l223_223056

theorem Paula_needs_52_tickets :
  let g := 2
  let b := 4
  let r := 3
  let f := 1
  let t_g := 4
  let t_b := 5
  let t_r := 7
  let t_f := 3
  g * t_g + b * t_b + r * t_r + f * t_f = 52 := by
  intros
  sorry

end Paula_needs_52_tickets_l223_223056


namespace problem_U_complement_eq_l223_223918

universe u
variable {α : Type u} [DecidableEq α]
variable (U M : Set α)

theorem problem_U_complement_eq
  (hU : U = ({1, 2, 3, 4, 5} : Set ℕ))
  (hM : U \ M = ({1, 3} : Set ℕ)) : 
  (2 : ℕ) ∈ M := 
by
  sorry

end problem_U_complement_eq_l223_223918


namespace original_number_exists_l223_223969

theorem original_number_exists :
  ∃ x : ℝ, 10 * x = x + 2.7 ∧ x = 0.3 :=
by {
  sorry
}

end original_number_exists_l223_223969


namespace solve_eq1_solve_eq2_l223_223786

-- Define the problem for equation (1)
theorem solve_eq1 (x : Real) : (x - 1)^2 = 2 ↔ (x = 1 + Real.sqrt 2 ∨ x = 1 - Real.sqrt 2) :=
by 
  sorry

-- Define the problem for equation (2)
theorem solve_eq2 (x : Real) : x^2 - 6 * x - 7 = 0 ↔ (x = -1 ∨ x = 7) :=
by 
  sorry

end solve_eq1_solve_eq2_l223_223786


namespace mike_picked_l223_223041

-- Define the number of pears picked by Jason, Keith, and the total number of pears picked.
def jason_picked : ℕ := 46
def keith_picked : ℕ := 47
def total_picked : ℕ := 105

-- Define the goal that we need to prove: the number of pears Mike picked.
theorem mike_picked (jason_picked keith_picked total_picked : ℕ) 
  (h1 : jason_picked = 46) 
  (h2 : keith_picked = 47) 
  (h3 : total_picked = 105) 
  : (total_picked - (jason_picked + keith_picked)) = 12 :=
by sorry

end mike_picked_l223_223041


namespace sum_of_squares_not_divisible_by_4_or_8_l223_223218

theorem sum_of_squares_not_divisible_by_4_or_8 (n : ℤ) (h : n % 2 = 1) :
  let a := n - 2
  let b := n
  let c := n + 2
  let sum_squares := a^2 + b^2 + c^2
  ¬(4 ∣ sum_squares ∨ 8 ∣ sum_squares) :=
by
  let a := n - 2
  let b := n
  let c := n + 2
  let sum_squares := a^2 + b^2 + c^2
  sorry

end sum_of_squares_not_divisible_by_4_or_8_l223_223218


namespace imons_no_entanglements_l223_223530

-- Define the fundamental structure for imons and their entanglements.
universe u
variable {α : Type u}

-- Define a graph structure to represent imons and their entanglement.
structure Graph (α : Type u) where
  vertices : Finset α
  edges : Finset (α × α)
  edge_sym : ∀ {x y}, (x, y) ∈ edges → (y, x) ∈ edges

-- Define the operations that can be performed on imons.
structure ImonOps (G : Graph α) where
  destroy : {v : α} → G.vertices.card % 2 = 1
  double : Graph α

-- Prove the main theorem
theorem imons_no_entanglements (G : Graph α) (op : ImonOps G) : 
  ∃ seq : List (ImonOps G), ∀ g : Graph α, g ∈ (seq.map (λ h => h.double)) → g.edges = ∅ :=
by
  sorry -- The proof would be constructed here.

end imons_no_entanglements_l223_223530


namespace sequence_formula_l223_223885

theorem sequence_formula (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : ∀ n > 1, a n - a (n - 1) = 2^(n-1)) : a n = 2^n - 1 := 
sorry

end sequence_formula_l223_223885


namespace ticket_price_for_children_l223_223044

open Nat

theorem ticket_price_for_children
  (C : ℕ)
  (adult_ticket_price : ℕ := 12)
  (num_adults : ℕ := 3)
  (num_children : ℕ := 3)
  (total_cost : ℕ := 66)
  (H : num_adults * adult_ticket_price + num_children * C = total_cost) :
  C = 10 :=
sorry

end ticket_price_for_children_l223_223044


namespace part_I_interval_part_II_range_l223_223751

noncomputable def f (x : ℝ) : ℝ := 2 * (cos x)^2 + 2 * sqrt 3 * sin x * cos x - 1

theorem part_I_interval :
  ∀ k : ℤ, ∀ x : ℝ, (π/6 + k*π ≤ x ∧ x ≤ 2*π/3 + k*π) → 
  (f x has_deriv_at (2 * (cos (2*x + π/6)))) x → 
  2 * (cos (2*x + π/6)) < 0 := sorry

theorem part_II_range :
  ∀ a b c : ℝ, ∀ A : ℝ, (tan B = sqrt 3 * a * c / (a^2 + c^2 - b^2)) -> 
  (π/6 < A ∧ A < π/2) → 
  (f A ≥ -1 ∧ f A < 2) := sorry

end part_I_interval_part_II_range_l223_223751


namespace number_of_large_posters_is_5_l223_223163

theorem number_of_large_posters_is_5
  (total_posters : ℕ)
  (small_posters_ratio : ℚ)
  (medium_posters_ratio : ℚ)
  (h_total : total_posters = 50)
  (h_small_ratio : small_posters_ratio = 2 / 5)
  (h_medium_ratio : medium_posters_ratio = 1 / 2) :
  (total_posters * (1 - small_posters_ratio - medium_posters_ratio)) = 5 :=
by sorry

end number_of_large_posters_is_5_l223_223163


namespace part1_part2_l223_223365

open Set

def A : Set ℝ := {x | x^2 - 8 * x + 15 = 0}
def B (a : ℝ) : Set ℝ := {x | a * x = 1}

theorem part1 : B (1/5) ⊆ A ∧ ¬ A ⊆ B (1/5) := by
  sorry
  
theorem part2 (a : ℝ) : (B a ⊆ A) ↔ a ∈ ({0, 1/3, 1/5} : Set ℝ) := by
  sorry

end part1_part2_l223_223365


namespace baseball_card_ratio_l223_223778

-- Define the conditions
variable (T : ℤ) -- Number of baseball cards on Tuesday

-- Given conditions
-- On Monday, Buddy has 30 baseball cards
def monday_cards : ℤ := 30

-- On Wednesday, Buddy has T + 12 baseball cards
def wednesday_cards : ℤ := T + 12

-- On Thursday, Buddy buys a third of what he had on Tuesday
def thursday_additional_cards : ℤ := T / 3

-- Total number of cards on Thursday is 32
def thursday_cards (T : ℤ) : ℤ := T + 12 + T / 3

-- We are given that Buddy has 32 baseball cards on Thursday
axiom thursday_total : thursday_cards T = 32

-- The theorem we want to prove: the ratio of Tuesday's to Monday's cards is 1:2
theorem baseball_card_ratio
  (T : ℤ)
  (htotal : thursday_cards T = 32)
  (hmon : monday_cards = 30) :
  T = 15 ∧ (T : ℚ) / monday_cards = 1 / 2 := by
  -- Proof goes here
  sorry

end baseball_card_ratio_l223_223778


namespace grandpa_age_times_jungmin_age_l223_223141

-- Definitions based on the conditions
def grandpa_age_last_year : ℕ := 71
def jungmin_age_last_year : ℕ := 8
def grandpa_age_this_year : ℕ := grandpa_age_last_year + 1
def jungmin_age_this_year : ℕ := jungmin_age_last_year + 1

-- The statement to prove
theorem grandpa_age_times_jungmin_age :
  grandpa_age_this_year / jungmin_age_this_year = 8 :=
by
  sorry

end grandpa_age_times_jungmin_age_l223_223141


namespace triangle_inequality_for_n6_l223_223724

variables {a b c : ℝ} {n : ℕ}
open Real

-- Define the main statement as a theorem
theorem triangle_inequality_for_n6 (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c)
  (ineq : 6 * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2)) :
  a + b > c ∧ b + c > a ∧ c + a > b :=
sorry

end triangle_inequality_for_n6_l223_223724


namespace city_population_l223_223405

theorem city_population (p : ℝ) (hp : 0.85 * (p + 2000) = p + 2050) : p = 2333 :=
by
  sorry

end city_population_l223_223405


namespace abs_conditions_iff_l223_223322

theorem abs_conditions_iff (x y : ℝ) :
  (|x| < 1 ∧ |y| < 1) ↔ (|x + y| + |x - y| < 2) :=
by
  sorry

end abs_conditions_iff_l223_223322


namespace greatest_integer_value_x_l223_223661

theorem greatest_integer_value_x :
  ∀ x : ℤ, (∃ k : ℤ, x^2 + 2 * x + 9 = k * (x - 5)) ↔ x ≤ 49 :=
by
  sorry

end greatest_integer_value_x_l223_223661


namespace film_finishes_earlier_on_first_channel_l223_223153

-- Definitions based on conditions
def DurationSegmentFirstChannel (n : ℕ) : ℝ := n * 22
def DurationSegmentSecondChannel (k : ℕ) : ℝ := k * 11

-- The time when first channel starts the n-th segment
def StartNthSegmentFirstChannel (n : ℕ) : ℝ := (n - 1) * 22

-- The number of segments second channel shows by the time first channel starts the n-th segment
def SegmentsShownSecondChannel (n : ℕ) : ℕ := ((n - 1) * 22) / 11

-- If first channel finishes earlier than second channel
theorem film_finishes_earlier_on_first_channel (n : ℕ) (hn : 1 < n) :
  DurationSegmentFirstChannel n < DurationSegmentSecondChannel (SegmentsShownSecondChannel n + 1) :=
sorry

end film_finishes_earlier_on_first_channel_l223_223153


namespace inequality_solution_set_l223_223824

theorem inequality_solution_set (x : ℝ) (h : x ≠ 0) : 
  (1 / x > 3) ↔ (0 < x ∧ x < 1 / 3) := 
by 
  sorry

end inequality_solution_set_l223_223824


namespace percentage_le_29_l223_223740

def sample_size : ℕ := 100
def freq_17_19 : ℕ := 1
def freq_19_21 : ℕ := 1
def freq_21_23 : ℕ := 3
def freq_23_25 : ℕ := 3
def freq_25_27 : ℕ := 18
def freq_27_29 : ℕ := 16
def freq_29_31 : ℕ := 28
def freq_31_33 : ℕ := 30

theorem percentage_le_29 : (freq_17_19 + freq_19_21 + freq_21_23 + freq_23_25 + freq_25_27 + freq_27_29) * 100 / sample_size = 42 :=
by
  sorry

end percentage_le_29_l223_223740


namespace meadow_trees_count_l223_223631

theorem meadow_trees_count (n : ℕ) (f s m : ℕ → ℕ) :
  (f 20 = s 7) ∧ (f 7 = s 94) ∧ (s 7 > f 20) → 
  n = 100 :=
by
  sorry

end meadow_trees_count_l223_223631


namespace more_knights_than_liars_l223_223696

theorem more_knights_than_liars 
  (k l : Nat)
  (h1 : (k + l) % 2 = 1)
  (h2 : ∀ i : Nat, i < k → ∃ j : Nat, j < l)
  (h3 : ∀ j : Nat, j < l → ∃ i : Nat, i < k) :
  k > l := 
sorry

end more_knights_than_liars_l223_223696


namespace min_odd_numbers_in_A_P_l223_223609

theorem min_odd_numbers_in_A_P (P : ℝ → ℝ) 
  (hP : ∃ a8 a7 a6 a5 a4 a3 a2 a1 a0 : ℝ, P = λ x, a8 * x^8 + a7 * x^7 + a6 * x^6 + a5 * x^5 + a4 * x^4 + a3 * x^3 + a2 * x^2 + a1 * x + a0)
  (h8_in_A_P : ∃ x : ℝ, P x = 8) : 
  (∀ x : ℝ, P x ∈ Set.A_P) → 
  (∃! y : ℝ, odd y ∧ y ∈ Set.A_P) := 
sorry

end min_odd_numbers_in_A_P_l223_223609


namespace train_passes_man_in_approx_18_seconds_l223_223401

noncomputable def train_length : ℝ := 300 -- meters
noncomputable def train_speed : ℝ := 68 -- km/h
noncomputable def man_speed : ℝ := 8 -- km/h
noncomputable def kmh_to_mps (v : ℝ) : ℝ := v * 1000 / 3600
noncomputable def relative_speed_mps : ℝ := kmh_to_mps (train_speed - man_speed)
noncomputable def time_to_pass_man : ℝ := train_length / relative_speed_mps

theorem train_passes_man_in_approx_18_seconds :
  abs (time_to_pass_man - 18) < 1 :=
by
  sorry

end train_passes_man_in_approx_18_seconds_l223_223401


namespace nontrivial_power_of_nat_l223_223439

theorem nontrivial_power_of_nat (n : ℕ) :
  (∃ A p : ℕ, 2^n + 1 = A^p ∧ p > 1) → n = 3 :=
by
  sorry

end nontrivial_power_of_nat_l223_223439


namespace fifth_term_sum_of_powers_of_4_l223_223705

theorem fifth_term_sum_of_powers_of_4 :
  (4^0 + 4^1 + 4^2 + 4^3 + 4^4) = 341 := 
by
  sorry

end fifth_term_sum_of_powers_of_4_l223_223705


namespace bells_ring_together_l223_223529

theorem bells_ring_together (church school day_care library noon : ℕ) :
  church = 18 ∧ school = 24 ∧ day_care = 30 ∧ library = 35 ∧ noon = 0 →
  ∃ t : ℕ, t = 2520 ∧ ∀ n, (t - noon) % n = 0 := by
  sorry

end bells_ring_together_l223_223529


namespace constant_term_binomial_l223_223741

noncomputable def integral_value : ℝ :=
  ∫ x in 0..(real.pi / 2), 6 * real.sin x

theorem constant_term_binomial (n : ℝ) (h : n = integral_value) :
  (∑ r in finset.range 7, nat.choose 6 r * (x^(6-r) * ((-2 / x^2)^r))).filter (λ term, term = 0) = 60 :=
by sorry

end constant_term_binomial_l223_223741


namespace order_of_a_b_c_l223_223737

noncomputable def a := Real.sqrt 3 - Real.sqrt 2
noncomputable def b := Real.sqrt 6 - Real.sqrt 5
noncomputable def c := Real.sqrt 7 - Real.sqrt 6

theorem order_of_a_b_c : a > b ∧ b > c :=
by
  sorry

end order_of_a_b_c_l223_223737


namespace max_value_proof_l223_223004

noncomputable def maximum_value (x y z : ℝ) : ℝ := 
  (2/x) + (1/y) - (2/z) + 2

theorem max_value_proof {x y z : ℝ} 
  (h_pos : 0 < x ∧ 0 < y ∧ 0 < z)
  (h_eq : x^2 - 3*x*y + 4*y^2 - z = 0):
  maximum_value x y z ≤ 3 :=
sorry

end max_value_proof_l223_223004


namespace shaded_area_l223_223868

noncomputable def area_of_shaded_region (AB : ℝ) (pi_approx : ℝ) : ℝ :=
  let R := AB / 2
  let r := R / 2
  let A_large := (1/2) * pi_approx * R^2
  let A_small := (1/2) * pi_approx * r^2
  2 * A_large - 4 * A_small

theorem shaded_area (h : area_of_shaded_region 40 3.14 = 628) : true :=
  sorry

end shaded_area_l223_223868


namespace original_faculty_members_l223_223687

theorem original_faculty_members (X : ℝ) (H0 : X > 0) 
  (H1 : 0.75 * X ≤ X)
  (H2 : ((0.75 * X + 35) * 1.10 * 0.80 = 195)) :
  X = 253 :=
by {
  sorry
}

end original_faculty_members_l223_223687


namespace problem_U_complement_eq_l223_223915

universe u
variable {α : Type u} [DecidableEq α]
variable (U M : Set α)

theorem problem_U_complement_eq
  (hU : U = ({1, 2, 3, 4, 5} : Set ℕ))
  (hM : U \ M = ({1, 3} : Set ℕ)) : 
  (2 : ℕ) ∈ M := 
by
  sorry

end problem_U_complement_eq_l223_223915


namespace value_of_expression_l223_223258

theorem value_of_expression : 3 ^ (0 ^ (2 ^ 11)) + ((3 ^ 0) ^ 2) ^ 11 = 2 := by
  sorry

end value_of_expression_l223_223258


namespace smallest_sum_of_consecutive_integers_l223_223247

theorem smallest_sum_of_consecutive_integers:
  ∃ (n m : ℕ), (n > 0) ∧ (20 * n + 190 = 2 * m^2) ∧ (20 * n + 190 = 450)  :=
by
  use 13, 15
  split; norm_num
  -- the proof steps would then follow
  sorry

end smallest_sum_of_consecutive_integers_l223_223247


namespace problem1_problem2_l223_223878

theorem problem1 : -20 + 3 + 5 - 7 = -19 := by
  sorry

theorem problem2 : (-3)^2 * 5 + (-2)^3 / 4 - |-3| = 40 := by
  sorry

end problem1_problem2_l223_223878


namespace average_age_combined_rooms_l223_223034

theorem average_age_combined_rooms :
  (8 * 30 + 5 * 22) / (8 + 5) = 26.9 := by
  sorry

end average_age_combined_rooms_l223_223034


namespace new_ratio_milk_to_water_l223_223033

def total_volume : ℕ := 100
def initial_milk_ratio : ℚ := 3
def initial_water_ratio : ℚ := 2
def additional_water : ℕ := 48

def new_milk_volume := (initial_milk_ratio / (initial_milk_ratio + initial_water_ratio)) * total_volume
def new_water_volume := (initial_water_ratio / (initial_milk_ratio + initial_water_ratio)) * total_volume + additional_water

theorem new_ratio_milk_to_water :
  new_milk_volume / (new_water_volume : ℚ) = 15 / 22 :=
by
  sorry

end new_ratio_milk_to_water_l223_223033


namespace determine_m_l223_223903

def setA_is_empty (m: ℝ) : Prop :=
  { x : ℝ | m * x = 1 } = ∅

theorem determine_m (m: ℝ) (h: setA_is_empty m) : m = 0 :=
by sorry

end determine_m_l223_223903


namespace eval_expression_l223_223431

theorem eval_expression : (503 * 503 - 502 * 504) = 1 :=
by
  sorry

end eval_expression_l223_223431


namespace fewer_mpg_in_city_l223_223674

def city_miles : ℕ := 336
def highway_miles : ℕ := 462
def city_mpg : ℕ := 24

def tank_size : ℕ := city_miles / city_mpg
def highway_mpg : ℕ := highway_miles / tank_size

theorem fewer_mpg_in_city : highway_mpg - city_mpg = 9 :=
by
  sorry

end fewer_mpg_in_city_l223_223674


namespace snow_white_seven_piles_l223_223070

def split_pile_action (piles : List ℕ) : Prop :=
  ∃ pile1 pile2, pile1 > 0 ∧ pile2 > 0 ∧ pile1 + pile2 + 1 ∈ piles

theorem snow_white_seven_piles :
  ∃ piles : List ℕ, piles.length = 7 ∧ ∀ pile ∈ piles, pile = 3 :=
sorry

end snow_white_seven_piles_l223_223070


namespace measure_of_MNP_l223_223039

-- Define the conditions of the pentagon
variables {M N P Q S : Type} -- Define the vertices of the pentagon
variables {MN NP PQ QS SM : ℝ} -- Define the lengths of the sides
variables (MNP QNS : ℝ) -- Define the measures of the involved angles

-- State the conditions
-- Pentagon sides are equal
axiom equal_sides : MN = NP ∧ NP = PQ ∧ PQ = QS ∧ QS = SM ∧ SM = MN 
-- Angle relation
axiom angle_relation : MNP = 2 * QNS

-- The goal is to prove that measure of angle MNP is 60 degrees
theorem measure_of_MNP : MNP = 60 :=
by {
  sorry -- The proof goes here
}

end measure_of_MNP_l223_223039


namespace question_l223_223929

variable (U : Set ℕ) (M : Set ℕ)

theorem question :
  U = {1, 2, 3, 4, 5} →
  (U \ M = {1, 3}) →
  2 ∈ M :=
by
  intros
  sorry

end question_l223_223929


namespace solve_for_x_l223_223344

theorem solve_for_x : ∀ (x : ℝ), 
  (x + 2 * x + 3 * x + 4 * x = 5) → (x = 1 / 2) :=
by 
  intros x H
  sorry

end solve_for_x_l223_223344


namespace right_triangle_acute_angles_l223_223955

theorem right_triangle_acute_angles (a b : ℝ)
  (h_right_triangle : a + b = 90)
  (h_ratio : a / b = 3 / 2) :
  (a = 54) ∧ (b = 36) :=
by
  sorry

end right_triangle_acute_angles_l223_223955


namespace probability_same_color_probability_different_color_and_odd_l223_223251

open Finset

def balls := {1, 2, 3, 4, 5}
def red_balls := {1, 2, 3}
def white_balls := {4, 5}
def events := balls.powerset.filter (λ s, s.card = 2)

-- Defining event A: Draw two balls of the same color
def event_A := ({1, 2}, {1, 3}, {2, 3}, {4, 5} : Finset (Finset ℕ))

-- Defining event B: Draw two balls of different colors, and at least one has an odd number
def event_B := ({1, 4}, {1, 5}, {2, 5}, {3, 4}, {3, 5} : Finset (Finset ℕ))

-- Calculate the probability of the events
noncomputable def probability {α} (e : Finset α) (Ω : Finset α) : ℚ :=
  (e.card : ℚ) / Ω.card

theorem probability_same_color : 
  probability event_A events = 2 / 5 := by
  sorry

theorem probability_different_color_and_odd : 
  probability event_B events = 1 / 2 := by 
  sorry

end probability_same_color_probability_different_color_and_odd_l223_223251


namespace question_l223_223932

variable (U : Set ℕ) (M : Set ℕ)

theorem question :
  U = {1, 2, 3, 4, 5} →
  (U \ M = {1, 3}) →
  2 ∈ M :=
by
  intros
  sorry

end question_l223_223932


namespace points_collinear_sum_l223_223358

theorem points_collinear_sum (x y : ℝ) :
  ∃ k : ℝ, (x - 1 = 3 * k ∧ 1 = k * (y - 2) ∧ -1 = 2 * k) → 
  x + y = -1 / 2 :=
by
  sorry

end points_collinear_sum_l223_223358


namespace number_of_tests_initially_l223_223214

-- Given conditions
variables (n S : ℕ)
variables (h1 : S / n = 70)
variables (h2 : S = 70 * n)
variables (h3 : (S - 55) / (n - 1) = 75)

-- Prove the number of tests initially, n, is 4.
theorem number_of_tests_initially (n : ℕ) (S : ℕ)
  (h1 : S / n = 70) (h2 : S = 70 * n) (h3 : (S - 55) / (n - 1) = 75) :
  n = 4 :=
sorry

end number_of_tests_initially_l223_223214


namespace calculate_total_parts_l223_223732

theorem calculate_total_parts (sample_size : ℕ) (draw_probability : ℚ) (N : ℕ) 
  (h_sample_size : sample_size = 30) 
  (h_draw_probability : draw_probability = 0.25) 
  (h_relation : sample_size = N * draw_probability) : 
  N = 120 :=
by
  rw [h_sample_size, h_draw_probability] at h_relation
  sorry

end calculate_total_parts_l223_223732


namespace jason_total_spending_l223_223213

def cost_of_shorts : ℝ := 14.28
def cost_of_jacket : ℝ := 4.74
def total_spent : ℝ := 19.02

theorem jason_total_spending : cost_of_shorts + cost_of_jacket = total_spent :=
by
  sorry

end jason_total_spending_l223_223213


namespace largest_possible_n_l223_223551

theorem largest_possible_n (b g : ℕ) (n : ℕ) (h1 : g = 3 * b)
  (h2 : ∀ (boy : ℕ), boy < b → ∀ (girlfriend : ℕ), girlfriend < g → girlfriend ≤ 2013)
  (h3 : ∀ (girl : ℕ), girl < g → ∀ (boyfriend : ℕ), boyfriend < b → boyfriend ≥ n) :
  n ≤ 671 := by
    sorry

end largest_possible_n_l223_223551


namespace sum_of_coordinates_l223_223386

-- Define the given conditions as hypotheses
def isThreeUnitsFromLine (x y : ℝ) : Prop := y = 18 ∨ y = 12
def isTenUnitsFromPoint (x y : ℝ) : Prop := (x - 5)^2 + (y - 15)^2 = 100

-- We aim to prove the sum of the coordinates of the points satisfying these conditions
theorem sum_of_coordinates (x1 x2 x3 x4 y1 y2 y3 y4 : ℝ) 
  (h1 : isThreeUnitsFromLine x1 y1 ∧ isTenUnitsFromPoint x1 y1)
  (h2 : isThreeUnitsFromLine x2 y2 ∧ isTenUnitsFromPoint x2 y2)
  (h3 : isThreeUnitsFromLine x3 y3 ∧ isTenUnitsFromPoint x3 y3)
  (h4 : isThreeUnitsFromLine x4 y4 ∧ isTenUnitsFromPoint x4 y4) :
  x1 + x2 + x3 + x4 + y1 + y2 + y3 + y4 = 50 :=
  sorry

end sum_of_coordinates_l223_223386


namespace irrational_roots_of_quadratic_l223_223343

theorem irrational_roots_of_quadratic (p q : ℤ) (h1 : p % 2 = 1) (h2 : q % 2 = 1) (h3 : p^2 - 2*q ≥ 0) :
  ∀ x, ¬ ∃ r : ℚ, x = -p + Real.sqrt (p^2 - 2*q) ∨ x = -p - Real.sqrt (p^2 - 2*q) ∧ r = (x : ℝ) := sorry

end irrational_roots_of_quadratic_l223_223343


namespace opposite_of_five_is_neg_five_l223_223994

theorem opposite_of_five_is_neg_five :
  ∃ (x : ℤ), (5 + x = 0) ∧ x = -5 :=
by
  use -5
  split
  · simp
  · rfl

end opposite_of_five_is_neg_five_l223_223994


namespace opposite_of_five_is_neg_five_l223_223993

theorem opposite_of_five_is_neg_five :
  ∃ (x : ℤ), (5 + x = 0) ∧ x = -5 :=
by
  use -5
  split
  · simp
  · rfl

end opposite_of_five_is_neg_five_l223_223993


namespace decagonal_die_expected_value_is_correct_l223_223845

def decagonalDieExpectedValue : ℕ := 5 -- A decagonal die has faces 1 to 10

def expectedValueDecagonalDie : ℝ := 5.5 -- The expected value as calculated.

theorem decagonal_die_expected_value_is_correct (p : fin 10 → ℝ) (i : fin 10) :
  p i = 1 / 10 ∧ (∑ i in finset.univ, p i * (i + 1 : ℝ)) = expectedValueDecagonalDie := by
    sorry

end decagonal_die_expected_value_is_correct_l223_223845


namespace lemon_count_l223_223635

theorem lemon_count {total_fruits mangoes pears pawpaws : ℕ} (kiwi lemon : ℕ) :
  total_fruits = 58 ∧ 
  mangoes = 18 ∧ 
  pears = 10 ∧ 
  pawpaws = 12 ∧ 
  (kiwi = lemon) →
  lemon = 9 :=
by 
  sorry

end lemon_count_l223_223635


namespace rectangle_area_l223_223415

theorem rectangle_area (p : ℝ) (l : ℝ) (h1 : 2 * (l + 2 * l) = p) :
  l * 2 * l = p^2 / 18 :=
by
  sorry

end rectangle_area_l223_223415


namespace find_integer_pairs_l223_223714

theorem find_integer_pairs :
  {ab : ℤ × ℤ | ∃ (p : Polynomial ℤ), let (a, b) := ab in
    ((Polynomial.C a * X + Polynomial.C b) * p).coeffs.all (λ c, c = 1 ∨ c = -1)} =
  {ab : ℤ × ℤ | ab = (1, 1) ∨ ab = (1, -1) ∨ ab = (-1, 1) ∨ ab = (-1, -1) ∨
                  ab = (0, 1) ∨ ab = (0, -1) ∨
                  ab = (2, 1) ∨ ab = (2, -1) ∨ ab = (-2, 1) ∨ ab = (-2, -1)} :=
begin
  sorry
end

end find_integer_pairs_l223_223714


namespace calculate_expression_l223_223877

theorem calculate_expression : 
  (3^2 - 2 * 3) - (5^2 - 2 * 5) + (7^2 - 2 * 7) = 23 := 
by sorry

end calculate_expression_l223_223877


namespace sequences_count_equals_fibonacci_n_21_l223_223886

noncomputable def increasing_sequences_count (n: ℕ) : ℕ := 
  -- Function to count the number of valid increasing sequences
  sorry

def fibonacci : ℕ → ℕ 
  | 0 => 0
  | 1 => 1
  | (n + 2) => fibonacci n + fibonacci (n + 1)

theorem sequences_count_equals_fibonacci_n_21 :
  increasing_sequences_count 20 = fibonacci 21 :=
sorry

end sequences_count_equals_fibonacci_n_21_l223_223886


namespace isosceles_triangle_perimeter_l223_223764

theorem isosceles_triangle_perimeter (a b : ℕ) (h_a : a = 8 ∨ a = 9) (h_b : b = 8 ∨ b = 9) 
(h_iso : a = a) (h_tri_ineq : a + a > b ∧ a + b > a ∧ b + a > a) :
  a + a + b = 25 ∨ a + a + b = 26 := 
by
  sorry

end isosceles_triangle_perimeter_l223_223764


namespace no_three_times_age_ago_l223_223407

theorem no_three_times_age_ago (F D : ℕ) (h₁ : F = 40) (h₂ : D = 40) (h₃ : F = 2 * D) :
  ¬ ∃ x, F - x = 3 * (D - x) :=
by
  sorry

end no_three_times_age_ago_l223_223407


namespace solve_inequality_f_ge_x_no_positive_a_b_satisfy_conditions_l223_223745

noncomputable def f (x : ℝ) : ℝ :=
  |2 * x - 1| - |2 * x - 2|

theorem solve_inequality_f_ge_x :
  {x : ℝ | f x >= x} = {x : ℝ | x <= -1 ∨ x = 1} :=
by sorry

theorem no_positive_a_b_satisfy_conditions :
  ∀ (a b : ℝ), a > 0 → b > 0 → (a + 2 * b = 1) → (2 / a + 1 / b = 4 - 1 / (a * b)) → false :=
by sorry

end solve_inequality_f_ge_x_no_positive_a_b_satisfy_conditions_l223_223745


namespace inradius_circumradius_inequality_l223_223601

variable {R r a b c : ℝ}

def inradius (ABC : Triangle) := r
def circumradius (ABC : Triangle) := R
def side_a (ABC : Triangle) := a
def side_b (ABC : Triangle) := b
def side_c (ABC : Triangle) := c

theorem inradius_circumradius_inequality (ABC : Triangle) :
  R / (2 * r) ≥ (64 * a^2 * b^2 * c^2 / ((4 * a^2 - (b - c)^2) * (4 * b^2 - (c - a)^2) * (4 * c^2 - (a - b)^2)))^2 :=
sorry

end inradius_circumradius_inequality_l223_223601


namespace number_subtracted_l223_223565

theorem number_subtracted (x : ℝ) : 3 + 2 * (8 - x) = 24.16 → x = -2.58 :=
by
  intro h
  sorry

end number_subtracted_l223_223565


namespace exp_eval_l223_223104

theorem exp_eval : (3^2)^4 = 6561 := by
sorry

end exp_eval_l223_223104


namespace eval_expression_l223_223884

theorem eval_expression : -30 + 12 * (8 / 4)^2 = 18 :=
by
  sorry

end eval_expression_l223_223884


namespace power_of_powers_eval_powers_l223_223137

theorem power_of_powers (a m n : ℕ) : (a^m)^n = a^(m * n) :=
  by sorry

theorem eval_powers : (3^2)^4 = 6561 :=
  by
    rw [power_of_powers]
    -- further proof of (3^8 = 6561) would go here
    sorry

end power_of_powers_eval_powers_l223_223137


namespace solve_logarithmic_equation_l223_223375

theorem solve_logarithmic_equation (x : ℝ) (h : log 8 x + 3 * log 2 (x^2) - log 4 x = 14) :
  x = 2^(12/5) :=
by
  sorry

end solve_logarithmic_equation_l223_223375


namespace population_increase_duration_l223_223954

noncomputable def birth_rate := 6 / 2 -- people every 2 seconds = 3 people per second
noncomputable def death_rate := 2 / 2 -- people every 2 seconds = 1 person per second
noncomputable def net_increase_per_second := (birth_rate - death_rate) -- net increase per second

def total_net_increase := 172800

theorem population_increase_duration :
  (total_net_increase / net_increase_per_second) / 3600 = 24 :=
by
  sorry

end population_increase_duration_l223_223954


namespace train_speed_in_km_per_hour_l223_223692

-- Definitions based on the conditions
def train_length : ℝ := 240  -- The length of the train in meters.
def time_to_pass_tree : ℝ := 8  -- The time to pass the tree in seconds.
def meters_per_second_to_kilometers_per_hour : ℝ := 3.6  -- Conversion factor from meters/second to kilometers/hour.

-- Statement based on the question and the correct answer
theorem train_speed_in_km_per_hour : (train_length / time_to_pass_tree) * meters_per_second_to_kilometers_per_hour = 108 :=
by
  sorry

end train_speed_in_km_per_hour_l223_223692


namespace sum_of_three_integers_with_product_5_pow_4_l223_223814

noncomputable def a : ℕ := 1
noncomputable def b : ℕ := 5
noncomputable def c : ℕ := 125

theorem sum_of_three_integers_with_product_5_pow_4 (h : a * b * c = 5^4) : 
  a + b + c = 131 := by
  have ha : a = 1 := rfl
  have hb : b = 5 := rfl
  have hc : c = 125 := rfl
  rw [ha, hb, hc, mul_assoc] at h
  exact sorry

end sum_of_three_integers_with_product_5_pow_4_l223_223814


namespace number_of_large_posters_is_5_l223_223162

theorem number_of_large_posters_is_5
  (total_posters : ℕ)
  (small_posters_ratio : ℚ)
  (medium_posters_ratio : ℚ)
  (h_total : total_posters = 50)
  (h_small_ratio : small_posters_ratio = 2 / 5)
  (h_medium_ratio : medium_posters_ratio = 1 / 2) :
  (total_posters * (1 - small_posters_ratio - medium_posters_ratio)) = 5 :=
by sorry

end number_of_large_posters_is_5_l223_223162


namespace winner_last_year_ounces_l223_223658

/-- Definition of the problem conditions -/
def ouncesPerHamburger : ℕ := 4
def hamburgersTonyaAte : ℕ := 22

/-- Theorem stating the desired result -/
theorem winner_last_year_ounces :
  hamburgersTonyaAte * ouncesPerHamburger = 88 :=
by
  sorry

end winner_last_year_ounces_l223_223658


namespace max_towns_meeting_criteria_l223_223398

-- Define the graph with edges of types air, bus, and train
inductive Link
| air
| bus
| train

-- Define a structure for the town network
structure Network (V : Type*) :=
(edges : V → V → Option Link)
(pairwise_linked : ∀ u v : V, u ≠ v → ∃ (lk : Link), edges u v = some lk)
(has_air_link : ∃ u v : V, u ≠ v ∧ edges u v = some Link.air)
(has_bus_link : ∃ u v : V, u ≠ v ∧ edges u v = some Link.bus)
(has_train_link : ∃ u v : V, u ≠ v ∧ edges u v = some Link.train)
(no_all_three_types : ∀ v : V, ¬ (∃ u w: V, u ≠ v ∧ w ≠ v ∧ u ≠ w ∧
  edges v u = some Link.air ∧ edges v w = some Link.bus ∧ edges v w = some Link.train))
(no_same_type_triangle : ∀ u v w : V, u ≠ v ∧ v ≠ w ∧ u ≠ w →
  ¬ (edges u v = edges u w ∧ edges u w = edges v w))

-- The theorem stating the maximum number of towns meeting the criteria
theorem max_towns_meeting_criteria (V : Type*) [Fintype V] [DecidableEq V] (h : ∀ e : V, ∃ p : Network V, p.edges = e) : 
  Fintype.card V ≤ 4 :=
by sorry

end max_towns_meeting_criteria_l223_223398


namespace round_table_chairs_l223_223548

theorem round_table_chairs :
  ∃ x : ℕ, (2 * x + 2 * 7 = 26) ∧ x = 6 :=
by
  sorry

end round_table_chairs_l223_223548


namespace valid_triangle_inequality_l223_223726

theorem valid_triangle_inequality (n : ℕ) (h : n = 6) :
  ∀ (a b c : ℝ), 0 < a → 0 < b → 0 < c →
  n * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2) →
  (a + b > c ∧ b + c > a ∧ c + a > b) :=
by
  intros a b c ha hb hc hineq
  have h₁ : n = 6 := h
  simplify_eq [h₁] at hineq
  have h₂ := nat.add_comm a b
  exact sorry

end valid_triangle_inequality_l223_223726


namespace ratio_of_c_to_b_l223_223978

    theorem ratio_of_c_to_b (a b c : ℤ) (h0 : a = 0) (h1 : a < b) (h2 : b < c)
      (h3 : (a + b + c) / 3 = b / 2) : c / b = 1 / 2 :=
    by
      -- proof steps go here
      sorry
    
end ratio_of_c_to_b_l223_223978


namespace inverse_of_matrix_l223_223308

noncomputable def my_matrix := matrix ([[5, -3], [-2, 1]])

theorem inverse_of_matrix :
  ∃ (M_inv : matrix ℕ ℕ ℝ), (my_matrix.det ≠ 0) ∧ (my_matrix * M_inv = 1) → M_inv = matrix ([[ -1, -3 ], [-2, -5 ]]) :=
by
  sorry

end inverse_of_matrix_l223_223308


namespace dogwood_trees_proof_l223_223597

def dogwood_trees_left (a b c : Float) : Float :=
  a + b - c

theorem dogwood_trees_proof : dogwood_trees_left 5.0 4.0 7.0 = 2.0 :=
by
  -- The proof itself is left out intentionally as per the instructions
  sorry

end dogwood_trees_proof_l223_223597


namespace total_distance_l223_223709

theorem total_distance (D : ℝ) 
  (h1 : 1/4 * (3/8 * D) = 210) : D = 840 := 
by
  -- proof steps would go here
  sorry

end total_distance_l223_223709


namespace symmetry_axis_is_2_range_of_a_l223_223445

-- Definitions given in the conditions
def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Condition 1: Constants a, b, c and a ≠ 0
variables (a b c : ℝ) (a_ne_zero : a ≠ 0)

-- Condition 2: Inequality constraint
axiom inequality_constraint : a^2 + 2 * a * c + c^2 < b^2

-- Condition 3: y-values are the same when x=t+2 and x=-t+2
axiom y_symmetry (t : ℝ) : quadratic_function a b c (t + 2) = quadratic_function a b c (-t + 2)

-- Question 1: Proving the symmetry axis is x=2
theorem symmetry_axis_is_2 : ∀ t : ℝ, (t + 2 + (-t + 2)) / 2 = 2 :=
by sorry

-- Question 2: Proving the range of a if y=2 when x=-2
theorem range_of_a (h : quadratic_function a b c (-2) = 2) (b_eq_neg4a : b = -4 * a) : 2 / 15 < a ∧ a < 2 / 7 :=
by sorry

end symmetry_axis_is_2_range_of_a_l223_223445


namespace ball_probability_l223_223146

theorem ball_probability (total_balls white green yellow red purple : ℕ) 
  (h_total : total_balls = 60)
  (h_white : white = 22)
  (h_green : green = 18)
  (h_yellow : yellow = 5)
  (h_red : red = 6)
  (h_purple : purple = 9) :
  (total_balls - red - purple) / total_balls = 3 / 4 :=
by
  sorry

end ball_probability_l223_223146


namespace add_in_base8_l223_223547

def base8_add (a b : ℕ) (n : ℕ): ℕ :=
  a * (8 ^ n) + b

theorem add_in_base8 : base8_add 123 56 0 = 202 := by
  sorry

end add_in_base8_l223_223547


namespace g_at_6_is_zero_l223_223479

def g (x : ℝ) : ℝ := 3*x^4 - 18*x^3 + 31*x^2 - 29*x - 72

theorem g_at_6_is_zero : g 6 = 0 :=
by {
  sorry
}

end g_at_6_is_zero_l223_223479


namespace min_odd_in_A_P_l223_223611

-- Define the polynomial P of degree 8
variable {R : Type*} [Ring R] (P : Polynomial R)
variable (hP : degree P = 8)

-- Define the set A_P for some value c in R
def A_P (c : R) : Set R := { x | P.eval x = c }

-- Given that the number 8 is included in the set A_P
variable (c : R) (h8 : (8 : R) ∈ A_P c)

-- Prove that the minimum number of odd numbers in the set A_P is 1
theorem min_odd_in_A_P : ∃ x ∈ A_P c, (x % 2 = 1) := sorry

end min_odd_in_A_P_l223_223611


namespace max_m_value_l223_223014

theorem max_m_value (m : ℕ) (h1 : m > 0) (h2 : ∃ k : ℕ, m^4 + 16 * m + 8 = k * (k + 1)) : m ≤ 2 :=
sorry

end max_m_value_l223_223014


namespace problem_statement_l223_223758

theorem problem_statement :
  ∀ (x : ℝ),
    (5 * x - 10 = 15 * x + 5) →
    (5 * (x + 3) = 15 / 2) :=
by
  intros x h
  sorry

end problem_statement_l223_223758


namespace opposite_of_5_is_neg5_l223_223990

def opposite (n x : ℤ) := n + x = 0

theorem opposite_of_5_is_neg5 : opposite 5 (-5) :=
by
  sorry

end opposite_of_5_is_neg5_l223_223990


namespace correctStatement_l223_223919

variable (U : Set ℕ) (M : Set ℕ)

namespace Proof

-- Given conditions
def universalSet := {1, 2, 3, 4, 5}
def complementM := {1, 3}
def isComplement (M : Set ℕ) : Prop := U \ M = complementM

-- Target statement to be proved
theorem correctStatement (h1 : U = universalSet) (h2 : isComplement M) : 2 ∈ M := by
  sorry

end Proof

end correctStatement_l223_223919


namespace isosceles_base_length_l223_223800

theorem isosceles_base_length (b : ℝ) (h1 : 7 + 7 + b = 23) : b = 9 :=
sorry

end isosceles_base_length_l223_223800


namespace inequality_solution_set_l223_223646

theorem inequality_solution_set : 
  { x : ℝ | (1 - x) * (x + 1) ≤ 0 ∧ x ≠ -1 } = { x : ℝ | x < -1 ∨ x ≥ 1 } :=
sorry

end inequality_solution_set_l223_223646


namespace minimum_value_of_f_on_neg_interval_l223_223966

theorem minimum_value_of_f_on_neg_interval (f : ℝ → ℝ) 
    (h_even : ∀ x, f (-x) = f x) 
    (h_increasing : ∀ x y, 1 ≤ x → x ≤ y → y ≤ 2 → f x ≤ f y) 
  : ∀ x, -2 ≤ x → x ≤ -1 → f (-1) ≤ f x := 
by
  sorry

end minimum_value_of_f_on_neg_interval_l223_223966


namespace repeating_decimal_fraction_sum_l223_223985

/-- The repeating decimal 3.171717... can be written as a fraction. When reduced to lowest
terms, the sum of the numerator and denominator of this fraction is 413. -/
theorem repeating_decimal_fraction_sum :
  let y := 3.17171717 -- The repeating decimal
  let frac_num := 314
  let frac_den := 99
  let sum := frac_num + frac_den
  y = frac_num / frac_den ∧ sum = 413 := by
  sorry

end repeating_decimal_fraction_sum_l223_223985


namespace pupils_count_l223_223469

-- Definitions based on given conditions
def number_of_girls : ℕ := 692
def girls_more_than_boys : ℕ := 458
def number_of_boys : ℕ := number_of_girls - girls_more_than_boys
def total_pupils : ℕ := number_of_girls + number_of_boys

-- The statement that the total number of pupils is 926
theorem pupils_count : total_pupils = 926 := by
  sorry

end pupils_count_l223_223469


namespace operation_addition_l223_223484

theorem operation_addition (a b c : ℝ) (op : ℝ → ℝ → ℝ)
  (H : ∀ a b c : ℝ, op (op a b) c = a + b + c) :
  ∀ a b : ℝ, op a b = a + b :=
sorry

end operation_addition_l223_223484


namespace possible_value_of_a_eq_neg1_l223_223267

theorem possible_value_of_a_eq_neg1 (a : ℝ) : (-6 * a ^ 2 = 3 * (4 * a + 2)) → (a = -1) :=
by
  intro h
  have H : a^2 + 2*a + 1 = 0
  · sorry
  show a = -1
  · sorry

end possible_value_of_a_eq_neg1_l223_223267


namespace fraction_problem_l223_223760

theorem fraction_problem (N D : ℚ) (h1 : 1.30 * N / (0.85 * D) = 25 / 21) : 
  N / D = 425 / 546 :=
sorry

end fraction_problem_l223_223760


namespace distinct_values_for_even_integers_lt_20_l223_223428

open Set

theorem distinct_values_for_even_integers_lt_20 :
  let evens := {2, 4, 6, 8, 10, 12, 14, 16, 18}
  let results := { (p + 1) * (q + 1) - 1 | p in evens, q in evens }
  results.finite :=
by sorry

end distinct_values_for_even_integers_lt_20_l223_223428


namespace area_of_circle_eq_sixteen_pi_l223_223842

theorem area_of_circle_eq_sixteen_pi :
  ∃ (x y : ℝ), (x^2 + y^2 - 8*x + 6*y = -9) ↔ (π * 4^2 = 16 * π) :=
by
  sorry

end area_of_circle_eq_sixteen_pi_l223_223842


namespace correctStatement_l223_223922

variable (U : Set ℕ) (M : Set ℕ)

namespace Proof

-- Given conditions
def universalSet := {1, 2, 3, 4, 5}
def complementM := {1, 3}
def isComplement (M : Set ℕ) : Prop := U \ M = complementM

-- Target statement to be proved
theorem correctStatement (h1 : U = universalSet) (h2 : isComplement M) : 2 ∈ M := by
  sorry

end Proof

end correctStatement_l223_223922


namespace determine_numbers_l223_223648

theorem determine_numbers (a b c : ℕ) (h₁ : a + b + c = 15) 
  (h₂ : (1 / (a : ℝ)) + (1 / (b : ℝ)) + (1 / (c : ℝ)) = 71 / 105) : 
  (a = 3 ∧ b = 5 ∧ c = 7) ∨ (a = 3 ∧ b = 7 ∧ c = 5) ∨ (a = 5 ∧ b = 3 ∧ c = 7) ∨ 
  (a = 5 ∧ b = 7 ∧ c = 3) ∨ (a = 7 ∧ b = 3 ∧ c = 5) ∨ (a = 7 ∧ b = 5 ∧ c = 3) :=
sorry

end determine_numbers_l223_223648


namespace probability_of_equal_numbers_when_throwing_two_fair_dice_l223_223142

theorem probability_of_equal_numbers_when_throwing_two_fair_dice :
  let total_outcomes := 36
  let favorable_outcomes := 6
  favorable_outcomes / total_outcomes = 1 / 6 :=
by
  sorry

end probability_of_equal_numbers_when_throwing_two_fair_dice_l223_223142


namespace opposite_of_five_l223_223996

theorem opposite_of_five : -5 = -5 :=
by
sorry

end opposite_of_five_l223_223996


namespace problem_statement_l223_223623

theorem problem_statement :
  ∃ (a b c : ℕ), gcd a (gcd b c) = 1 ∧
  (∃ x y : ℝ, 2 * y = 8 * x - 7) ∧
  a ^ 2 + b ^ 2 + (c:ℤ) ^ 2 = 117 :=
sorry

end problem_statement_l223_223623


namespace cylinder_section_volume_l223_223572

theorem cylinder_section_volume (a : ℝ) :
  let volume := (π * a^3 / 4)
  let section1_volume := volume * (1 / 3)
  let section2_volume := volume * (1 / 4)
  let enclosed_volume := (section1_volume - section2_volume) / 2
  enclosed_volume = π * a^3 / 24 := by
  sorry

end cylinder_section_volume_l223_223572


namespace milk_jars_good_for_sale_l223_223224

noncomputable def good_whole_milk_jars : ℕ := 
  let initial_jars := 60 * 30
  let short_deliveries := 20 * 30 * 2
  let damaged_jars_1 := 3 * 5
  let damaged_jars_2 := 4 * 6
  let totally_damaged_cartons := 2 * 30
  let received_jars := initial_jars - short_deliveries - damaged_jars_1 - damaged_jars_2 - totally_damaged_cartons
  let spoilage := (5 * received_jars) / 100
  received_jars - spoilage

noncomputable def good_skim_milk_jars : ℕ := 
  let initial_jars := 40 * 40
  let short_delivery := 10 * 40
  let damaged_jars := 5 * 4
  let totally_damaged_carton := 1 * 40
  let received_jars := initial_jars - short_delivery - damaged_jars - totally_damaged_carton
  let spoilage := (3 * received_jars) / 100
  received_jars - spoilage

noncomputable def good_almond_milk_jars : ℕ := 
  let initial_jars := 30 * 20
  let short_delivery := 5 * 20
  let damaged_jars := 2 * 3
  let received_jars := initial_jars - short_delivery - damaged_jars
  let spoilage := (1 * received_jars) / 100
  received_jars - spoilage

theorem milk_jars_good_for_sale : 
  good_whole_milk_jars = 476 ∧
  good_skim_milk_jars = 1106 ∧
  good_almond_milk_jars = 489 :=
by
  sorry

end milk_jars_good_for_sale_l223_223224


namespace exp_eval_l223_223101

theorem exp_eval : (3^2)^4 = 6561 := by
sorry

end exp_eval_l223_223101


namespace exactly_two_succeed_probability_l223_223653

/-- Define the probabilities of three independent events -/
def P1 : ℚ := 1 / 2
def P2 : ℚ := 1 / 3
def P3 : ℚ := 3 / 4

/-- Define the probability that exactly two out of the three people successfully decrypt the password -/
def prob_exactly_two_succeed : ℚ := P1 * P2 * (1 - P3) + P1 * (1 - P2) * P3 + (1 - P1) * P2 * P3

theorem exactly_two_succeed_probability :
  prob_exactly_two_succeed = 5 / 12 :=
sorry

end exactly_two_succeed_probability_l223_223653


namespace exists_d_for_m_divides_f_of_f_n_l223_223362

noncomputable def f : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => 23 * f (n + 1) + f n

theorem exists_d_for_m_divides_f_of_f_n (m : ℕ) : 
  ∃ (d : ℕ), ∀ (n : ℕ), m ∣ f (f n) ↔ d ∣ n := 
sorry

end exists_d_for_m_divides_f_of_f_n_l223_223362


namespace minimum_odd_numbers_in_A_P_l223_223614

-- Polynomials and assumptions
def degree (P : Polynomial ℝ) : ℕ := P.natDegree

-- The set A_P is defined as the set of all numbers x for which P(x) gives a certain value
def A_P (P : Polynomial ℝ) : Set ℝ := {x : ℝ | P.eval x = P.eval 8}

-- Define the main theorem statement
theorem minimum_odd_numbers_in_A_P (P : Polynomial ℝ) (hdeg : degree P = 8) (h8 : 8 ∈ A_P P) : 
  ∃ n, n = 1 ∧ ∃ (x : ℝ), x ∈ A_P P ∧ ¬ (x % 2 = 0) := sorry

end minimum_odd_numbers_in_A_P_l223_223614


namespace triangle_inequality_condition_l223_223718

theorem triangle_inequality_condition (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) (ineq : 6 * (a * b + b * c + c * a) > 5 * (a ^ 2 + b ^ 2 + c ^ 2)) : 
  (a < b + c ∧ b < a + c ∧ c < a + b) :=
sorry

end triangle_inequality_condition_l223_223718


namespace log_0_333_eq_neg1_l223_223185

theorem log_0_333_eq_neg1 : log 3 0.333 = -1 := by
  have h1 : 0.333 = 1 / 3 := sorry -- This would be assured by a separate lemma about decimal fractions
  have h2 : log 3 (1 / 3) = log 3 1 - log 3 3 := by
    rw [log_div]
  have h3 : log 3 1 = 0 := by
    exact log_one_eq_zero
  have h4 : log 3 3 = 1 := by
    exact log_self_eq_one
  rw [←h1, h2, h3, h4]
  exact by norm_num

end log_0_333_eq_neg1_l223_223185


namespace cos_alpha_minus_pi_over_4_l223_223581

theorem cos_alpha_minus_pi_over_4 (α : ℝ) (hα1 : 0 < α) (hα2 : α < π / 2) (hα3 : Real.tan α = 2) :
  Real.cos (α - π / 4) = 3 * Real.sqrt 10 / 10 :=
sorry

end cos_alpha_minus_pi_over_4_l223_223581


namespace opposite_of_five_l223_223997

theorem opposite_of_five : -5 = -5 :=
by
sorry

end opposite_of_five_l223_223997


namespace max_chords_through_line_l223_223651

noncomputable def maxChords (n : ℕ) : ℕ :=
  let k := n / 2
  k * k + n

theorem max_chords_through_line (points : ℕ) (h : points = 2017) : maxChords 2016 = 1018080 :=
by
  have h1 : (2016 / 2) * (2016 / 2) + 2016 = 1018080 := by norm_num
  rw [← h1]; sorry

end max_chords_through_line_l223_223651


namespace smallest_sum_of_20_consecutive_integers_twice_perfect_square_l223_223248

theorem smallest_sum_of_20_consecutive_integers_twice_perfect_square :
  ∃ n : ℕ, ∃ k : ℕ, (∀ m : ℕ, m ≥ n → 0 < m) ∧ 10 * (2 * n + 19) = 2 * k^2 ∧ 10 * (2 * n + 19) = 450 :=
by
  sorry

end smallest_sum_of_20_consecutive_integers_twice_perfect_square_l223_223248


namespace xyz_sum_56_l223_223756

theorem xyz_sum_56 (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x * y + z = 55) (h2 : y * z + x = 55) (h3 : z * x + y = 55)
  (even_cond : x % 2 = 0 ∨ y % 2 = 0 ∨ z % 2 = 0) :
  x + y + z = 56 :=
sorry

end xyz_sum_56_l223_223756


namespace total_wheels_in_parking_lot_l223_223035

-- Definitions (conditions)
def cars := 14
def wheels_per_car := 4
def missing_wheels_per_missing_car := 1
def missing_cars := 2

def bikes := 5
def wheels_per_bike := 2

def unicycles := 3
def wheels_per_unicycle := 1

def twelve_wheeler_trucks := 2
def wheels_per_twelve_wheeler_truck := 12
def damaged_wheels_per_twelve_wheeler_truck := 3
def damaged_twelve_wheeler_trucks := 1

def eighteen_wheeler_trucks := 1
def wheels_per_eighteen_wheeler_truck := 18

-- The total wheels calculation proof
theorem total_wheels_in_parking_lot :
  ((cars * wheels_per_car - missing_cars * missing_wheels_per_missing_car) +
   (bikes * wheels_per_bike) +
   (unicycles * wheels_per_unicycle) +
   (twelve_wheeler_trucks * wheels_per_twelve_wheeler_truck - damaged_twelve_wheeler_trucks * damaged_wheels_per_twelve_wheeler_truck) +
   (eighteen_wheeler_trucks * wheels_per_eighteen_wheeler_truck)) = 106 := by
  sorry

end total_wheels_in_parking_lot_l223_223035


namespace twenty_percent_l223_223757

-- Given condition
def condition (X : ℝ) : Prop := 0.4 * X = 160

-- Theorem to show that 20% of X equals 80 given the condition
theorem twenty_percent (X : ℝ) (h : condition X) : 0.2 * X = 80 :=
by sorry

end twenty_percent_l223_223757


namespace cos_double_angle_l223_223006

theorem cos_double_angle (α β : Real) 
    (h1 : Real.sin α = Real.cos β) 
    (h2 : Real.sin α * Real.cos β - 2 * Real.cos α * Real.sin β = 1 / 2) :
    Real.cos (2 * β) = 2 / 3 :=
by
  sorry

end cos_double_angle_l223_223006


namespace path_problem_l223_223605

noncomputable def path_bounds (N : ℕ) (h : 0 < N) : Prop :=
  ∃ p : ℕ, 4 * N ≤ p ∧ p ≤ 2 * N^2 + 2 * N

theorem path_problem (N : ℕ) (h : 0 < N) : path_bounds N h :=
  sorry

end path_problem_l223_223605


namespace inequality_ab_gt_ac_l223_223736

theorem inequality_ab_gt_ac {a b c : ℝ} (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) : a * b > a * c :=
sorry

end inequality_ab_gt_ac_l223_223736


namespace no_solution_in_natural_numbers_l223_223521

theorem no_solution_in_natural_numbers :
  ¬ ∃ (x y : ℕ), 2^x + 21^x = y^3 :=
sorry

end no_solution_in_natural_numbers_l223_223521


namespace sum_modulo_9_l223_223880

theorem sum_modulo_9 :
  (2 + 33 + 444 + 5555 + 66666 + 777777 + 8888888 + 99999999) % 9 = 6 := 
by
  -- Skipping the detailed proof steps
  sorry

end sum_modulo_9_l223_223880


namespace probability_at_least_one_3_l223_223282

open Probability

noncomputable def fair_six_sided_die : Pmf ℤ := Pmf.uniform_of_finite 6 (λ (x : ℕ), x + 1)

def valid_tosses (X1 X2 X3 X4 : ℕ) := 
  (1, 1) ∈ fair_six_sided_die.to_finset ∧
  (1, 1) ∈ fair_six_sided_die.to_finset ∧
  (1, 1) ∈ fair_six_sided_die.to_finset ∧
  (1, 1) ∈ fair_six_sided_die.to_finset ∧
  (X1 + X2 + X3 = X4)

def at_least_one_three (X1 X2 X3 X4 : ℕ) := X1 = 3 ∨ X2 = 3 ∨ X3 = 3 ∨ X4 = 3

theorem probability_at_least_one_3 (h : valid_tosses X1 X2 X3 X4) : 
  probability fair_six_sided_die (at_least_one_three X1 X2 X3 X4) = 9 / 20 := 
sorry

end probability_at_least_one_3_l223_223282


namespace y_six_power_eq_44_over_27_l223_223414

theorem y_six_power_eq_44_over_27
  (y : ℝ)
  (h_pos : 0 < y)
  (h_equation : ∛(2 - y^3) + ∛(2 + y^3) = 2)
  : y^6 = 44 / 27 :=
sorry

end y_six_power_eq_44_over_27_l223_223414


namespace expression_value_as_fraction_l223_223302

theorem expression_value_as_fraction :
  2 + (3 / (2 + (5 / (4 + (7 / 3))))) = 91 / 19 :=
by
  sorry

end expression_value_as_fraction_l223_223302


namespace sum_geom_seq_nine_l223_223472

noncomputable def geom_seq (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n * q

theorem sum_geom_seq_nine {a : ℕ → ℝ} {q : ℝ} (h_geom : geom_seq a q)
  (h1 : a 1 * (1 + q + q^2) = 30) 
  (h2 : a 4 * (1 + q + q^2) = 120) :
  a 7 + a 8 + a 9 = 480 :=
  sorry

end sum_geom_seq_nine_l223_223472


namespace proof_2_in_M_l223_223937

def U : Set ℕ := {1, 2, 3, 4, 5}

def M : Set ℕ := { x | x ∈ U ∧ x ≠ 1 ∧ x ≠ 3 }

theorem proof_2_in_M : 2 ∈ M :=
by
  sorry

end proof_2_in_M_l223_223937


namespace ratio_of_capital_l223_223391

variable (C A B : ℝ)
variable (h1 : B = 4 * C)
variable (h2 : B / (A + 5 * C) = 6000 / 16500)

theorem ratio_of_capital : A / B = 17 / 4 :=
by
  sorry

end ratio_of_capital_l223_223391


namespace rachel_pool_fill_time_l223_223229

theorem rachel_pool_fill_time :
  ∀ (pool_volume : ℕ) (num_hoses : ℕ) (hose_rate : ℕ),
  pool_volume = 30000 →
  num_hoses = 5 →
  hose_rate = 3 →
  (pool_volume / (num_hoses * hose_rate * 60) : ℤ) = 33 :=
by
  intros pool_volume num_hoses hose_rate h1 h2 h3
  sorry

end rachel_pool_fill_time_l223_223229


namespace deformable_to_triangle_l223_223540

-- Definition of the planar polygon with n sides
structure Polygon (n : ℕ) := 
  (vertices : Fin n → ℝ × ℝ) -- This is a simplified representation of a planar polygon using vertex coordinates

noncomputable def canDeformToTriangle (poly : Polygon n) : Prop := sorry

theorem deformable_to_triangle (n : ℕ) (h : n > 4) (poly : Polygon n) : canDeformToTriangle poly := 
  sorry

end deformable_to_triangle_l223_223540


namespace prove_healthy_diet_multiple_l223_223059

variable (rum_on_pancakes rum_earlier rum_after_pancakes : ℝ)
variable (healthy_multiple : ℝ)

-- Definitions from conditions
def Sally_gave_rum_on_pancakes : Prop := rum_on_pancakes = 10
def Don_had_rum_earlier : Prop := rum_earlier = 12
def Don_can_have_rum_after_pancakes : Prop := rum_after_pancakes = 8

-- Concluding multiple for healthy diet
def healthy_diet_multiple : Prop := healthy_multiple = (rum_on_pancakes + rum_after_pancakes - rum_earlier) / rum_on_pancakes

theorem prove_healthy_diet_multiple :
  Sally_gave_rum_on_pancakes rum_on_pancakes →
  Don_had_rum_earlier rum_earlier →
  Don_can_have_rum_after_pancakes rum_after_pancakes →
  healthy_diet_multiple rum_on_pancakes rum_earlier rum_after_pancakes healthy_multiple →
  healthy_multiple = 0.8 := 
by
  intros h1 h2 h3 h4
  sorry

end prove_healthy_diet_multiple_l223_223059


namespace option_C_correct_l223_223026

variable (a b : ℝ)

theorem option_C_correct (h : a > b) : -15 * a < -15 * b := 
  sorry

end option_C_correct_l223_223026


namespace sequence_term_general_formula_l223_223446

theorem sequence_term_general_formula (S : ℕ → ℚ) (a : ℕ → ℚ) :
  (∀ n, S n = n^2 + (1/2)*n + 5) →
  (∀ n, (n ≥ 2) → a n = S n - S (n - 1)) →
  a 1 = 13/2 →
  (∀ n, a n = if n = 1 then 13/2 else 2*n - 1/2) :=
by
  intros hS ha h1
  sorry

end sequence_term_general_formula_l223_223446


namespace largest_prime_divisor_l223_223286

-- Let n be a positive integer
def is_positive_integer (n : ℕ) : Prop :=
  n > 0

-- Define that n equals the sum of the squares of its four smallest positive divisors
def is_sum_of_squares_of_smallest_divisors (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), a = 2 ∧ b = 5 ∧ c = 10 ∧ n = 1 + a^2 + b^2 + c^2

-- Prove that the largest prime divisor of n is 13
theorem largest_prime_divisor (n : ℕ) (h1 : is_positive_integer n) (h2 : is_sum_of_squares_of_smallest_divisors n) :
  ∃ p : ℕ, Prime p ∧ p ∣ n ∧ ∀ q : ℕ, Prime q ∧ q ∣ n → q ≤ p ∧ p = 13 :=
by
  sorry

end largest_prime_divisor_l223_223286


namespace AD_mutually_exclusive_not_complementary_l223_223140

-- Define the sets representing the outcomes of the events
def A : Set ℕ := {1, 3, 5}
def B : Set ℕ := {2, 4, 6}
def C : Set ℕ := {2, 4, 6}
def D : Set ℕ := {2, 4}

-- Define mutually exclusive
def mutually_exclusive (X Y : Set ℕ) : Prop := X ∩ Y = ∅

-- Define complementary
def complementary (X Y : Set ℕ) : Prop := X ∪ Y = {1, 2, 3, 4, 5, 6}

-- The statement to prove that events A and D are mutually exclusive but not complementary
theorem AD_mutually_exclusive_not_complementary :
  mutually_exclusive A D ∧ ¬ complementary A D :=
by
  sorry

end AD_mutually_exclusive_not_complementary_l223_223140


namespace team_total_points_l223_223520

theorem team_total_points 
  (n : ℕ)
  (best_score actual : ℕ)
  (desired_avg : ℕ)
  (hypothetical_score : ℕ)
  (current_best_score : ℕ)
  (team_size : ℕ)
  (h1 : team_size = 8)
  (h2 : current_best_score = 85)
  (h3 : hypothetical_score = 92)
  (h4 : desired_avg = 84)
  (h5 : hypothetical_score - current_best_score = 7)
  (h6 : team_size * desired_avg = 672) :
  (actual = 665) :=
sorry

end team_total_points_l223_223520


namespace han_xin_troop_min_soldiers_l223_223335

theorem han_xin_troop_min_soldiers (n : ℕ) : 
  (n % 3 = 2) ∧ (n % 5 = 3) ∧ (n % 7 = 4) → n = 53 :=
  sorry

end han_xin_troop_min_soldiers_l223_223335


namespace sum_of_cousins_ages_l223_223066

theorem sum_of_cousins_ages :
  ∃ (a b c d e : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e ∧
    1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ 1 ≤ c ∧ c ≤ 9 ∧
    1 ≤ d ∧ d ≤ 9 ∧ 1 ≤ e ∧ e ≤ 9 ∧
    a * b = 36 ∧ c * d = 40 ∧ a + b + c + d + e = 33 :=
by
  sorry

end sum_of_cousins_ages_l223_223066


namespace triangle_inequality_for_n6_l223_223723

variables {a b c : ℝ} {n : ℕ}
open Real

-- Define the main statement as a theorem
theorem triangle_inequality_for_n6 (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c)
  (ineq : 6 * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2)) :
  a + b > c ∧ b + c > a ∧ c + a > b :=
sorry

end triangle_inequality_for_n6_l223_223723


namespace initial_mean_calculated_l223_223496

theorem initial_mean_calculated (M : ℝ) (h1 : 25 * M - 35 = 25 * 191.4 - 35) : M = 191.4 := 
  sorry

end initial_mean_calculated_l223_223496


namespace find_n_l223_223965

-- Define the function to sum the digits of a natural number n
def digit_sum (n : ℕ) : ℕ := 
  -- This is a dummy implementation for now
  -- Normally, we would implement the sum of the digits of n
  sorry 

-- The main theorem that we want to prove
theorem find_n : ∃ (n : ℕ), digit_sum n + n = 2011 ∧ n = 1991 :=
by
  -- Proof steps would go here, but we're skipping those with sorry.
  sorry

end find_n_l223_223965


namespace road_renovation_l223_223509

theorem road_renovation (x : ℕ) (h : 200 / (x + 20) = 150 / x) : 
  x = 60 ∧ (x + 20) = 80 :=
by {
  sorry
}

end road_renovation_l223_223509


namespace melted_ice_cream_depth_l223_223543

theorem melted_ice_cream_depth :
  ∀ (r_sphere r_cylinder : ℝ) (h : ℝ),
    r_sphere = 3 →
    r_cylinder = 10 →
    (4 / 3) * π * r_sphere^3 = 100 * π * h →
    h = 9 / 25 :=
  by
    intros r_sphere r_cylinder h
    intros hr_sphere hr_cylinder
    intros h_volume_eq
    sorry

end melted_ice_cream_depth_l223_223543


namespace find_a_l223_223586

noncomputable def f (x a : ℝ) : ℝ := x + Real.exp (x - a)
noncomputable def g (x a : ℝ) : ℝ := Real.log (x + 2) - 4 * Real.exp (a - x)

theorem find_a (x₀ a : ℝ) (h : f x₀ a - g x₀ a = 3) : a = -Real.log 2 - 1 :=
by
  sorry

end find_a_l223_223586


namespace age_of_youngest_child_l223_223268

theorem age_of_youngest_child (x : ℕ) (h : x + (x + 3) + (x + 6) + (x + 9) + (x + 12) = 50) : x = 4 :=
by
  sorry

end age_of_youngest_child_l223_223268


namespace opposite_of_5_is_neg5_l223_223989

def opposite (n x : ℤ) := n + x = 0

theorem opposite_of_5_is_neg5 : opposite 5 (-5) :=
by
  sorry

end opposite_of_5_is_neg5_l223_223989


namespace trig_value_ordering_l223_223180

theorem trig_value_ordering : 
  let a := Real.sin (17 * Real.pi / 12)
  let b := Real.cos (4 * Real.pi / 9)
  let c := Real.tan (7 * Real.pi / 4)
  c < a ∧ a < b := 
by 
  -- Definitions based on given conditions
  let a := Real.sin (17 * Real.pi / 12)
  let b := Real.cos (4 * Real.pi / 9)
  let c := Real.tan (7 * Real.pi / 4)
  -- The proof will go here
  sorry

end trig_value_ordering_l223_223180


namespace total_chewing_gums_l223_223380

-- Definitions for the conditions
def mary_gums : Nat := 5
def sam_gums : Nat := 10
def sue_gums : Nat := 15

-- Lean 4 Theorem statement to prove the total chewing gums
theorem total_chewing_gums : mary_gums + sam_gums + sue_gums = 30 := by
  sorry

end total_chewing_gums_l223_223380


namespace time_to_fill_pool_l223_223967

def LindasPoolCapacity : ℕ := 30000
def CurrentVolume : ℕ := 6000
def NumberOfHoses : ℕ := 6
def RatePerHosePerMinute : ℕ := 3
def GallonsNeeded : ℕ := LindasPoolCapacity - CurrentVolume
def RatePerHosePerHour : ℕ := RatePerHosePerMinute * 60
def TotalHourlyRate : ℕ := NumberOfHoses * RatePerHosePerHour

theorem time_to_fill_pool : (GallonsNeeded / TotalHourlyRate) = 22 :=
by
  sorry

end time_to_fill_pool_l223_223967


namespace problem_solution_l223_223924

universe u

variable (U : Set Nat) (M : Set Nat)
variable (complement_U_M : Set Nat)

axiom U_def : U = {1, 2, 3, 4, 5}
axiom complement_U_M_def : complement_U_M = {1, 3}
axiom M_def : M = U \ complement_U_M

theorem problem_solution : 2 ∈ M := by
  sorry

end problem_solution_l223_223924


namespace m_plus_n_is_172_l223_223833

-- defining the conditions for m
def m := 3

-- helper function to count divisors
def count_divisors (x : ℕ) : ℕ :=
  (List.range x).filter (λ d, x % (d + 1) = 0).length.succ

-- defining the conditions for n
noncomputable def n :=
  let primes := List.filter nat.prime (List.range 100) in
  let candidates := primes.map (λ p, p * p) in
  (candidates.filter (λ x, x < 200)).maximum' sorry

theorem m_plus_n_is_172 : m + n = 172 :=
by
  -- filling in that m is 3
  let m : ℕ := 3
  -- filling in that n is 169
  let n : ℕ := 13 * 13
  show m + n = 172
  calc
    m + n = 3 + 169 := by rfl
    ... = 172 := by rfl

end m_plus_n_is_172_l223_223833


namespace power_of_powers_l223_223086

theorem power_of_powers :
  (3^2)^4 = 6561 := 
by 
  sorry

end power_of_powers_l223_223086


namespace height_relationship_l223_223395

theorem height_relationship (r1 r2 h1 h2 : ℝ) (h_radii : r2 = 1.2 * r1) (h_volumes : π * r1^2 * h1 = π * r2^2 * h2) : h1 = 1.44 * h2 :=
by
  sorry

end height_relationship_l223_223395


namespace length_more_than_breadth_l223_223381

theorem length_more_than_breadth (length cost_per_metre total_cost : ℝ) (breadth : ℝ) :
  length = 60 → cost_per_metre = 26.50 → total_cost = 5300 → 
  (total_cost = (2 * length + 2 * breadth) * cost_per_metre) → length - breadth = 20 :=
by
  intros hlength hcost_per_metre htotal_cost hperimeter_cost
  rw [hlength, hcost_per_metre] at hperimeter_cost
  sorry

end length_more_than_breadth_l223_223381


namespace intersection_M_N_l223_223457

noncomputable def M : Set ℝ := { x | x^2 - x ≤ 0 }
noncomputable def N : Set ℝ := { x | 1 - abs x > 0 }
noncomputable def intersection : Set ℝ := { x | x ≥ 0 ∧ x < 1 }

theorem intersection_M_N : M ∩ N = intersection :=
by
  sorry

end intersection_M_N_l223_223457


namespace rug_area_calculation_l223_223156

theorem rug_area_calculation (length_floor width_floor strip_width : ℕ)
  (h_length : length_floor = 10)
  (h_width : width_floor = 8)
  (h_strip : strip_width = 2) :
  (length_floor - 2 * strip_width) * (width_floor - 2 * strip_width) = 24 := by
  sorry

end rug_area_calculation_l223_223156


namespace function_properties_l223_223261

noncomputable def f (x : ℝ) : ℝ := x^2

theorem function_properties :
  (∀ x1 x2 : ℝ, f (x1 * x2) = f x1 * f x2) ∧
  (∀ x : ℝ, 0 < x → deriv f x > 0) ∧
  (∀ x : ℝ, deriv f (-x) = -deriv f x) :=
by
  sorry

end function_properties_l223_223261


namespace valid_triangle_inequality_l223_223725

theorem valid_triangle_inequality (n : ℕ) (h : n = 6) :
  ∀ (a b c : ℝ), 0 < a → 0 < b → 0 < c →
  n * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2) →
  (a + b > c ∧ b + c > a ∧ c + a > b) :=
by
  intros a b c ha hb hc hineq
  have h₁ : n = 6 := h
  simplify_eq [h₁] at hineq
  have h₂ := nat.add_comm a b
  exact sorry

end valid_triangle_inequality_l223_223725


namespace calculate_expression_l223_223703

variable (a : ℝ)

theorem calculate_expression : (-a) ^ 2 * (-a ^ 5) ^ 4 / a ^ 12 * (-2 * a ^ 4) = -2 * a ^ 14 := 
by sorry

end calculate_expression_l223_223703


namespace initial_roses_l223_223678

theorem initial_roses (x : ℕ) (h : x - 2 + 32 = 41) : x = 11 :=
sorry

end initial_roses_l223_223678


namespace group_purchase_cheaper_l223_223270

-- Define the initial conditions
def initial_price : ℕ := 10
def bulk_price : ℕ := 7
def delivery_cost : ℕ := 100
def group_size : ℕ := 50

-- Define the costs for individual and group purchases
def individual_cost : ℕ := initial_price
def group_cost : ℕ := bulk_price + (delivery_cost / group_size)

-- Statement to prove: cost per participant in a group purchase is less than cost per participant in individual purchases
theorem group_purchase_cheaper : group_cost < individual_cost := by
  sorry

end group_purchase_cheaper_l223_223270


namespace part_I_part_II_l223_223893

-- Definitions of the sets A, B, and C
def A : Set ℝ := { x | x ≤ -1 ∨ x ≥ 3 }
def B : Set ℝ := { x | 1 ≤ x ∧ x ≤ 6 }
def C (m : ℝ) : Set ℝ := { x | m + 1 ≤ x ∧ x ≤ 2 * m }

-- Proof statements
theorem part_I : A ∩ B = { x | 3 ≤ x ∧ x ≤ 6 } :=
by sorry

theorem part_II (m : ℝ) : (B ∪ C m = B) → (m ≤ 3) :=
by sorry

end part_I_part_II_l223_223893


namespace provider_choices_count_l223_223045

theorem provider_choices_count :
  let num_providers := 25
  let num_s_providers := 6
  let remaining_providers_after_laura := num_providers - 1
  let remaining_providers_after_brother := remaining_providers_after_laura - 1

  (num_providers * num_s_providers * remaining_providers_after_laura * remaining_providers_after_brother) = 75900 :=
by
  sorry

end provider_choices_count_l223_223045


namespace range_of_m_l223_223940

theorem range_of_m (x y m : ℝ) (h1 : x - y = 2 * m + 7) (h2 : x + y = 4 * m - 3) 
  (h3 : x < 0) (h4 : y < 0) : m < -2 / 3 := 
by 
  sorry

end range_of_m_l223_223940


namespace wall_length_l223_223459

theorem wall_length
    (brick_length brick_width brick_height : ℝ)
    (wall_height wall_width : ℝ)
    (num_bricks : ℕ)
    (wall_length_cm : ℝ)
    (h_brick_volume : brick_length * brick_width * brick_height = 1687.5)
    (h_wall_volume :
        wall_length_cm * wall_height * wall_width
        = (brick_length * brick_width * brick_height) * num_bricks)
    (h_wall_height : wall_height = 600)
    (h_wall_width : wall_width = 22.5)
    (h_num_bricks : num_bricks = 7200) :
    wall_length_cm / 100 = 9 := 
by
  sorry

end wall_length_l223_223459


namespace solution_l223_223341

noncomputable def problem_statement : Prop :=
  ∃ (x y : ℝ), (x > 0 ∧ y > 0) ∧ (6 * x^2 + 18 * x * y = 2 * x^3 + 3 * x^2 * y^2) ∧ x = (3 + Real.sqrt 153) / 4

theorem solution : problem_statement :=
by
  sorry

end solution_l223_223341


namespace total_spent_after_discount_and_tax_l223_223861

-- Define prices for each item
def price_bracelet := 4
def price_keychain := 5
def price_coloring_book := 3
def price_sticker := 1
def price_toy_car := 6

-- Define discounts and tax rates
def discount_bracelet := 0.10
def sales_tax := 0.05

-- Define the quantity of each item purchased by Paula, Olive, and Nathan
def quantity_paula_bracelets := 3
def quantity_paula_keychains := 2
def quantity_paula_coloring_books := 1
def quantity_paula_stickers := 4

def quantity_olive_coloring_books := 1
def quantity_olive_bracelets := 2
def quantity_olive_toy_cars := 1
def quantity_olive_stickers := 3

def quantity_nathan_toy_cars := 4
def quantity_nathan_stickers := 5
def quantity_nathan_keychains := 1

-- Function to calculate total cost before discount and tax
def total_cost_before_discount_and_tax (bracelets keychains coloring_books stickers toy_cars : Nat) : Float :=
  Float.ofNat (bracelets * price_bracelet) +
  Float.ofNat (keychains * price_keychain) +
  Float.ofNat (coloring_books * price_coloring_book) +
  Float.ofNat (stickers * price_sticker) +
  Float.ofNat (toy_cars * price_toy_car)

-- Function to calculate discount on bracelets
def bracelet_discount (bracelets : Nat) : Float :=
  Float.ofNat (bracelets * price_bracelet) * discount_bracelet

-- Function to calculate total cost after discount and before tax
def total_cost_after_discount (total_cost discount : Float) : Float :=
  total_cost - discount

-- Function to calculate total cost after tax
def total_cost_after_tax (total_cost : Float) (tax_rate : Float) : Float :=
  total_cost * (1 + tax_rate)

-- Proof statement (no proof provided, only the statement)
theorem total_spent_after_discount_and_tax : 
  total_cost_after_tax (
    total_cost_after_discount
      (total_cost_before_discount_and_tax quantity_paula_bracelets quantity_paula_keychains quantity_paula_coloring_books quantity_paula_stickers 0)
      (bracelet_discount quantity_paula_bracelets)
    +
    total_cost_after_discount
      (total_cost_before_discount_and_tax quantity_olive_bracelets 0 quantity_olive_coloring_books quantity_olive_stickers quantity_olive_toy_cars)
      (bracelet_discount quantity_olive_bracelets)
    +
    total_cost_before_discount_and_tax 0 quantity_nathan_keychains 0 quantity_nathan_stickers quantity_nathan_toy_cars
  ) sales_tax = 85.05 := 
sorry

end total_spent_after_discount_and_tax_l223_223861


namespace knicks_win_tournament_probability_l223_223763

noncomputable def knicks_win_probability : ℚ :=
  let knicks_win_proba := 2 / 5
  let heat_win_proba := 3 / 5
  let first_4_games_scenarios := 6 * (knicks_win_proba^2 * heat_win_proba^2)
  first_4_games_scenarios * knicks_win_proba

theorem knicks_win_tournament_probability :
  knicks_win_probability = 432 / 3125 :=
by
  sorry

end knicks_win_tournament_probability_l223_223763


namespace power_of_powers_eval_powers_l223_223132

theorem power_of_powers (a m n : ℕ) : (a^m)^n = a^(m * n) :=
  by sorry

theorem eval_powers : (3^2)^4 = 6561 :=
  by
    rw [power_of_powers]
    -- further proof of (3^8 = 6561) would go here
    sorry

end power_of_powers_eval_powers_l223_223132


namespace power_of_powers_eval_powers_l223_223133

theorem power_of_powers (a m n : ℕ) : (a^m)^n = a^(m * n) :=
  by sorry

theorem eval_powers : (3^2)^4 = 6561 :=
  by
    rw [power_of_powers]
    -- further proof of (3^8 = 6561) would go here
    sorry

end power_of_powers_eval_powers_l223_223133


namespace tom_overall_profit_l223_223079

def initial_purchase_cost : ℝ := 20 * 3 + 30 * 5 + 15 * 10
def purchase_commission : ℝ := 0.02 * initial_purchase_cost
def total_initial_cost : ℝ := initial_purchase_cost + purchase_commission

def sale_revenue_before_commission : ℝ := 10 * 4 + 20 * 7 + 5 * 12
def sales_commission : ℝ := 0.02 * sale_revenue_before_commission
def total_sales_revenue : ℝ := sale_revenue_before_commission - sales_commission

def remaining_stock_a_value : ℝ := 10 * (3 * 2)
def remaining_stock_b_value : ℝ := 10 * (5 * 1.20)
def remaining_stock_c_value : ℝ := 10 * (10 * 0.90)
def total_remaining_value : ℝ := remaining_stock_a_value + remaining_stock_b_value + remaining_stock_c_value

def overall_profit_or_loss : ℝ := total_sales_revenue + total_remaining_value - total_initial_cost

theorem tom_overall_profit : overall_profit_or_loss = 78 := by
  sorry

end tom_overall_profit_l223_223079


namespace valid_triangle_inequality_l223_223727

theorem valid_triangle_inequality (n : ℕ) (h : n = 6) :
  ∀ (a b c : ℝ), 0 < a → 0 < b → 0 < c →
  n * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2) →
  (a + b > c ∧ b + c > a ∧ c + a > b) :=
by
  intros a b c ha hb hc hineq
  have h₁ : n = 6 := h
  simplify_eq [h₁] at hineq
  have h₂ := nat.add_comm a b
  exact sorry

end valid_triangle_inequality_l223_223727


namespace fraction_identity_l223_223569

theorem fraction_identity (a b : ℝ) (h1 : 1/a + 2/b = 1) (h2 : a ≠ -b) : 
  (ab - a)/(a + b) = 1 := 
by 
  sorry

end fraction_identity_l223_223569


namespace bulgarian_inequality_l223_223523

theorem bulgarian_inequality (a b c d : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) (d_pos : 0 < d) :
    (a^4 / (a^3 + a^2 * b + a * b^2 + b^3) + 
     b^4 / (b^3 + b^2 * c + b * c^2 + c^3) + 
     c^4 / (c^3 + c^2 * d + c * d^2 + d^3) + 
     d^4 / (d^3 + d^2 * a + d * a^2 + a^3)) 
    ≥ (a + b + c + d) / 4 :=
sorry

end bulgarian_inequality_l223_223523


namespace point_between_lines_l223_223202

theorem point_between_lines (b : ℝ) (h1 : 6 * 5 - 8 * b + 1 < 0) (h2 : 3 * 5 - 4 * b + 5 > 0) : b = 4 :=
  sorry

end point_between_lines_l223_223202


namespace Sammy_has_8_bottle_caps_l223_223061

def Billie_caps : Nat := 2
def Janine_caps (B : Nat) : Nat := 3 * B
def Sammy_caps (J : Nat) : Nat := J + 2

theorem Sammy_has_8_bottle_caps : 
  Sammy_caps (Janine_caps Billie_caps) = 8 := 
by
  sorry

end Sammy_has_8_bottle_caps_l223_223061


namespace domain_of_c_is_all_reals_l223_223430

theorem domain_of_c_is_all_reals (k : ℝ) : 
  (∀ x : ℝ, -3 * x^2 + 5 * x + k ≠ 0) ↔ k < -(25 / 12) :=
by
  sorry

end domain_of_c_is_all_reals_l223_223430


namespace sufficient_not_necessary_condition_l223_223001

noncomputable def sufficient_but_not_necessary (x y : ℝ) : Prop :=
  (x > 1 ∧ y > 1) → (x + y > 2) ∧ (x + y > 2 → ¬(x > 1 ∧ y > 1))

theorem sufficient_not_necessary_condition (x y : ℝ) :
  sufficient_but_not_necessary x y :=
sorry

end sufficient_not_necessary_condition_l223_223001


namespace power_of_powers_eval_powers_l223_223134

theorem power_of_powers (a m n : ℕ) : (a^m)^n = a^(m * n) :=
  by sorry

theorem eval_powers : (3^2)^4 = 6561 :=
  by
    rw [power_of_powers]
    -- further proof of (3^8 = 6561) would go here
    sorry

end power_of_powers_eval_powers_l223_223134


namespace isosceles_triangle_base_length_l223_223804

theorem isosceles_triangle_base_length (a b P : ℕ) (h1 : a = 7) (h2 : P = 23) (h3 : P = 2 * a + b) : b = 9 :=
sorry

end isosceles_triangle_base_length_l223_223804


namespace sum_of_pairs_l223_223853

theorem sum_of_pairs (a : ℕ → ℝ) (h1 : ∀ n, a n ≠ 0)
  (h2 : ∀ n, a n * a (n + 3) = a (n + 2) * a (n + 5))
  (h3 : a 1 * a 2 + a 3 * a 4 + a 5 * a 6 = 6) :
  a 1 * a 2 + a 3 * a 4 + a 5 * a 6 + a 7 * a 8 + a 9 * a 10 + a 11 * a 12 + 
  a 13 * a 14 + a 15 * a 16 + a 17 * a 18 + a 19 * a 20 + a 21 * a 22 + 
  a 23 * a 24 + a 25 * a 26 + a 27 * a 28 + a 29 * a 30 + a 31 * a 32 + 
  a 33 * a 34 + a 35 * a 36 + a 37 * a 38 + a 39 * a 40 + a 41 * a 42 = 42 := 
sorry

end sum_of_pairs_l223_223853


namespace flag_arrangements_l223_223393

theorem flag_arrangements (B R : ℕ) (M : ℕ) : 
  B = 12 ∧ R = 11 ∧ (∃ M, 
    M = (13 * Nat.choose 13 11 - 2 * Nat.choose 13 11)) →
  M % 1000 = 858 :=
by 
  intros h,
  obtain ⟨hB, hR, hM⟩ := h, 
  rw [hB, hR] at hM,
  simp [Nat.choose_eq_factorial_div_factorial, hM],
  sorry

end flag_arrangements_l223_223393


namespace line_parallel_to_plane_l223_223315

-- Defining conditions
def vector_a : ℝ × ℝ × ℝ := (1, -1, 3)
def vector_n : ℝ × ℝ × ℝ := (0, 3, 1)

-- Lean theorem statement
theorem line_parallel_to_plane : 
  let ⟨a1, a2, a3⟩ := vector_a;
  let ⟨n1, n2, n3⟩ := vector_n;
  a1 * n1 + a2 * n2 + a3 * n3 = 0 :=
by 
  -- Proof omitted
  sorry

end line_parallel_to_plane_l223_223315


namespace cube_of_720_diamond_1001_l223_223241

-- Define the operation \diamond
def diamond (a b : ℕ) : ℕ :=
  (Nat.factors (a * b)).toFinset.card

-- Define the specific numbers 720 and 1001
def n1 : ℕ := 720
def n2 : ℕ := 1001

-- Calculate the cubic of the result of diamond operation
def cube_of_diamond : ℕ := (diamond n1 n2) ^ 3

-- The statement to be proved
theorem cube_of_720_diamond_1001 : cube_of_diamond = 216 :=
by {
  sorry
}

end cube_of_720_diamond_1001_l223_223241


namespace interval_intersection_l223_223184

theorem interval_intersection :
  {x : ℝ | 1 < 3 * x ∧ 3 * x < 2 ∧ 1 < 5 * x ∧ 5 * x < 2} =
  {x : ℝ | (1 / 3 : ℝ) < x ∧ x < (2 / 5 : ℝ)} :=
by
  -- Need a proof here
  sorry

end interval_intersection_l223_223184


namespace part1_part2_l223_223749

theorem part1 (a : ℝ) : (a - 3 ≠ 0) ∧ (16 - 4 * (a-3) * (-1) = 0) → 
  a = -1 ∧ ∀ x : ℝ, (4 * x^2 + 4 * x + 1 = 0 ↔ x = -1/2) :=
sorry

theorem part2 (a : ℝ) : (a - 3 ≠ 0) ∧ (16 - 4 * (a-3) * (-1) > 0) → 
  a > -1 ∧ a ≠ 3 :=
sorry

end part1_part2_l223_223749


namespace k_20_coloring_connected_l223_223388

open SimpleGraph

theorem k_20_coloring_connected :
  ∃ c : Fin 5, 
  ∀ (K : SimpleGraph (Fin 20)) 
    (colored_K : ∀ e : K.edgeSet, Fin 5),
    (K = completeGraph (Fin 20)) → 
    ¬(K.deleteEdges (colored_K⁻¹' {c})).Disconnected :=
sorry

end k_20_coloring_connected_l223_223388


namespace john_paintball_times_l223_223770

theorem john_paintball_times (x : ℕ) (cost_per_box : ℕ) (boxes_per_play : ℕ) (monthly_spending : ℕ) :
  (cost_per_box = 25) → (boxes_per_play = 3) → (monthly_spending = 225) → (boxes_per_play * cost_per_box * x = monthly_spending) → x = 3 :=
by
  intros h1 h2 h3 h4
  -- proof would go here
  sorry

end john_paintball_times_l223_223770


namespace marbles_count_l223_223533

theorem marbles_count (M : ℕ)
  (h_blue : (M / 2) = n_blue)
  (h_red : (M / 4) = n_red)
  (h_green : 27 = n_green)
  (h_yellow : 14 = n_yellow)
  (h_total : (n_blue + n_red + n_green + n_yellow) = M) :
  M = 164 :=
by
  sorry

end marbles_count_l223_223533


namespace find_y_six_l223_223413

theorem find_y_six (y : ℝ) (h : y > 0) (h_eq : (2 - y^3)^(1/3) + (2 + y^3)^(1/3) = 2) : 
    y^6 = 116 / 27 :=
by
  sorry

end find_y_six_l223_223413


namespace quadratic_discriminant_constraint_l223_223201

theorem quadratic_discriminant_constraint (c : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 - 4*x1 + c = 0 ∧ x2^2 - 4*x2 + c = 0) ↔ c < 4 := 
by
  sorry

end quadratic_discriminant_constraint_l223_223201


namespace walking_running_distance_ratio_l223_223412

variables (d_w d_r : ℝ)

theorem walking_running_distance_ratio :
  d_w + d_r = 12 ∧ (d_w / 4 + d_r / 8 = 2.25) → d_w / d_r = 1 :=
by
  assume h : d_w + d_r = 12 ∧ (d_w / 4 + d_r / 8 = 2.25)
  sorry

end walking_running_distance_ratio_l223_223412


namespace sqrt_square_eq_self_l223_223951

theorem sqrt_square_eq_self (a : ℝ) (h : a ≥ 1/2) :
  Real.sqrt ((2 * a - 1) ^ 2) = 2 * a - 1 :=
by
  sorry

end sqrt_square_eq_self_l223_223951


namespace exponentiation_rule_example_l223_223097

theorem exponentiation_rule_example : (3^2)^4 = 6561 := 
by {
  -- This proof is based on the power rule (ab)^c = a^(b*c)
  have step1 : (3^2)^4 = 3^(2 * 4), 
  {
    rw pow_mul,
  },
  -- Simplification of 3^(2*4) to 3^8
  have step2 : 3^(2 * 4) = 3^8,
  {
    refl,
  },
  -- Actual computation for 3^8 = 6561
  have step3 : 3^8 = 6561,
  {
    norm_num,
  },
  -- Combining all steps
  rw [step1, step2, step3],
  exact rfl,
}

end exponentiation_rule_example_l223_223097


namespace rational_k_quadratic_solution_count_l223_223385

theorem rational_k_quadratic_solution_count (N : ℕ) :
  (N = 98) ↔ 
  (∃ (k : ℚ) (x : ℤ), |k| < 500 ∧ (3 * x^2 + k * x + 7 = 0)) :=
sorry

end rational_k_quadratic_solution_count_l223_223385


namespace number_of_possible_IDs_l223_223567

theorem number_of_possible_IDs : 
  ∃ (n : ℕ), 
  (∀ (a b : Fin 26) (x y : Fin 10),
    a = b ∨ x = y ∨ (a = b ∧ x = y) → 
    n = 9100) :=
sorry

end number_of_possible_IDs_l223_223567


namespace minimum_odd_in_A_P_l223_223616

-- Define the polynomial and its properties
def polynomial_degree8 (P : ℝ[X]) : Prop :=
  P.degree = 8

-- Define the set A_P
def A_P (P : ℝ[X]) : set ℝ :=
  {x | P.eval x = 8}

-- Main theorem statement
theorem minimum_odd_in_A_P (P : ℝ[X]) (hP : polynomial_degree8 P) : ∃ n : ℕ, n = 1 ∧ ∃ x ∈ A_P P, odd x :=
sorry

end minimum_odd_in_A_P_l223_223616


namespace f_properties_l223_223262

open Real

-- Define the function f(x) = x^2
noncomputable def f (x : ℝ) : ℝ := x^2

-- Define the statement to be proved
theorem f_properties (x₁ x₂ : ℝ) (x : ℝ) (h : 0 < x) :
  (f (x₁ * x₂) = f x₁ * f x₂) ∧ 
  (deriv f x > 0) ∧
  (∀ x : ℝ, deriv f (-x) = -deriv f x) :=
by
  sorry

end f_properties_l223_223262


namespace cars_meet_in_two_hours_l223_223832

theorem cars_meet_in_two_hours (t : ℝ) (d : ℝ) (v1 v2 : ℝ) (h1 : d = 60) (h2 : v1 = 13) (h3 : v2 = 17) (h4 : v1 * t + v2 * t = d) : t = 2 := 
by
  sorry

end cars_meet_in_two_hours_l223_223832


namespace percent_increase_bike_helmet_l223_223474

theorem percent_increase_bike_helmet :
  let old_bike_cost := 160
  let old_helmet_cost := 40
  let bike_increase_rate := 0.05
  let helmet_increase_rate := 0.10
  let new_bike_cost := old_bike_cost * (1 + bike_increase_rate)
  let new_helmet_cost := old_helmet_cost * (1 + helmet_increase_rate)
  let old_total_cost := old_bike_cost + old_helmet_cost
  let new_total_cost := new_bike_cost + new_helmet_cost
  let increase_amount := new_total_cost - old_total_cost
  let percent_increase := (increase_amount / old_total_cost) * 100
  percent_increase = 6 :=
by
  sorry

end percent_increase_bike_helmet_l223_223474


namespace find_b_value_l223_223812

-- Define the conditions: line equation and given range for b
def line_eq (x : ℝ) (b : ℝ) : ℝ := b - x

-- Define the points P, Q, S
def P (b : ℝ) : ℝ × ℝ := ⟨0, b⟩
def Q (b : ℝ) : ℝ × ℝ := ⟨b, 0⟩
def S (b : ℝ) : ℝ × ℝ := ⟨6, b - 6⟩

-- Define the area ratio condition
def area_ratio_condition (b : ℝ) : Prop :=
  (0 < b ∧ b < 6) ∧ ((6 - b) / b) ^ 2 = 4 / 25

-- Define the main theorem to prove
theorem find_b_value (b : ℝ) : area_ratio_condition b → b = 4.3 := by
  sorry

end find_b_value_l223_223812


namespace sequence_a7_l223_223902

/-- 
  Given a sequence {a_n} such that a_1 + a_{2n-1} = 4n - 6, 
  prove that a_7 = 11 
-/
theorem sequence_a7 (a : ℕ → ℤ)
  (h : ∀ n : ℕ, a 1 + a (2 * n - 1) = 4 * n - 6) : a 7 = 11 :=
by
  sorry

end sequence_a7_l223_223902


namespace right_triangle_primes_l223_223357

-- Define what it means for a number to be a prime
def is_prime (n : ℕ) : Prop := ∃ m : ℕ, m > 1 ∧ m < n ∧ n % m = 0

-- State the problem
theorem right_triangle_primes
  (a b : ℕ)
  (ha : is_prime a)
  (hb : is_prime b)
  (a_gt_b : a > b)
  (a_plus_b : a + b = 90)
  (a_minus_b_prime : is_prime (a - b)) :
  b = 17 :=
sorry

end right_triangle_primes_l223_223357


namespace first_term_of_geometric_series_l223_223695

/-- An infinite geometric series with common ratio -1/3 has a sum of 24.
    Prove that the first term of the series is 32. -/
theorem first_term_of_geometric_series (r : ℝ) (S : ℝ) (a : ℝ) 
  (h1 : r = -1/3) 
  (h2 : S = 24) 
  (h3 : S = a / (1 - r)) : 
  a = 32 := 
sorry

end first_term_of_geometric_series_l223_223695


namespace value_of_n_l223_223647

theorem value_of_n 
  {a b n : ℕ} (ha : a > 0) (hb : b > 0) 
  (h : (1 + b)^n = 243) : 
  n = 5 := by 
  sorry

end value_of_n_l223_223647


namespace sum_eq_prod_nat_numbers_l223_223825

theorem sum_eq_prod_nat_numbers (A B C D E F : ℕ) :
  A + B + C + D + E + F = A * B * C * D * E * F →
  (A = 0 ∧ B = 0 ∧ C = 0 ∧ D = 0 ∧ E = 0 ∧ F = 0) ∨
  (A = 1 ∧ B = 1 ∧ C = 1 ∧ D = 1 ∧ E = 2 ∧ F = 6) :=
by
  sorry

end sum_eq_prod_nat_numbers_l223_223825


namespace zhen_zhen_test_score_l223_223144

theorem zhen_zhen_test_score
  (avg1 avg2 : ℝ) (n m : ℝ)
  (h1 : avg1 = 88)
  (h2 : avg2 = 90)
  (h3 : n = 4)
  (h4 : m = 5) :
  avg2 * m - avg1 * n = 98 :=
by
  -- Given the hypotheses h1, h2, h3, and h4,
  -- we need to show that avg2 * m - avg1 * n = 98.
  sorry

end zhen_zhen_test_score_l223_223144


namespace smallest_AAB_value_exists_l223_223167

def is_consecutive_digits (A B : ℕ) : Prop :=
  (B = A + 1 ∨ A = B + 1) ∧ 1 ≤ A ∧ A ≤ 9 ∧ 1 ≤ B ∧ B ≤ 9

def two_digit_to_int (A B : ℕ) : ℕ :=
  10 * A + B

def three_digit_to_int (A B : ℕ) : ℕ :=
  110 * A + B

theorem smallest_AAB_value_exists :
  ∃ (A B: ℕ), is_consecutive_digits A B ∧ two_digit_to_int A B = (1 / 7 : ℝ) * ↑(three_digit_to_int A B) ∧ three_digit_to_int A B = 889 :=
sorry

end smallest_AAB_value_exists_l223_223167


namespace fourth_term_of_sequence_l223_223408

-- Given conditions
def first_term : ℕ := 5
def fifth_term : ℕ := 1280

-- Definition of the common ratio
def common_ratio (a : ℕ) (b : ℕ) : ℕ := (b / a)^(1 / 4)

-- Function to calculate the nth term of a geometric sequence
def nth_term (a r n : ℕ) : ℕ := a * r^(n - 1)

-- Prove the fourth term of the geometric sequence is 320
theorem fourth_term_of_sequence 
    (a : ℕ) (b : ℕ) (a_pos : a = first_term) (b_eq : nth_term a (common_ratio a b) 5 = b) : 
    nth_term a (common_ratio a b) 4 = 320 := by
  sorry

end fourth_term_of_sequence_l223_223408


namespace vector_sum_is_zero_l223_223021

variables {V : Type*} [AddCommGroup V]

variables (AB CF BC FA : V)

-- Condition: Vectors form a closed polygon
def vectors_form_closed_polygon (AB CF BC FA : V) : Prop :=
  AB + BC + CF + FA = 0

theorem vector_sum_is_zero
  (h : vectors_form_closed_polygon AB CF BC FA) :
  AB + BC + CF + FA = 0 :=
  h

end vector_sum_is_zero_l223_223021


namespace max_tickets_sold_l223_223539

theorem max_tickets_sold (bus_capacity : ℕ) (num_stations : ℕ) (max_capacity : bus_capacity = 25) 
  (total_stations : num_stations = 14) : 
  ∃ (tickets : ℕ), tickets = 67 :=
by 
  sorry

end max_tickets_sold_l223_223539


namespace third_median_length_l223_223660

variable (a b : ℝ) (A : ℝ)

def two_medians (m₁ m₂ : ℝ) : Prop :=
  m₁ = 4.5 ∧ m₂ = 7.5

def triangle_area (area : ℝ) : Prop :=
  area = 6 * Real.sqrt 20

theorem third_median_length (m₁ m₂ m₃ : ℝ) (area : ℝ) (h₁ : two_medians m₁ m₂)
  (h₂ : triangle_area area) : m₃ = 3 * Real.sqrt 5 := by
  sorry

end third_median_length_l223_223660


namespace total_divisions_is_48_l223_223525

-- Definitions based on the conditions
def initial_cells := 1
def final_cells := 1993
def cells_added_division_42 := 41
def cells_added_division_44 := 43

-- The main statement we want to prove
theorem total_divisions_is_48 (a b : ℕ) 
  (h1 : cells_added_division_42 = 41)
  (h2 : cells_added_division_44 = 43)
  (h3 : cells_added_division_42 * a + cells_added_division_44 * b = final_cells - initial_cells) :
  a + b = 48 := 
sorry

end total_divisions_is_48_l223_223525


namespace range_of_a_l223_223453

theorem range_of_a (a : ℝ) (h : a > 0) : (∀ x : ℝ, x > 0 → 9 * x + a^2 / x ≥ a^2 + 8) → 2 ≤ a ∧ a ≤ 4 :=
by
  intros h1
  sorry

end range_of_a_l223_223453


namespace quadratic_radicals_x_le_10_l223_223351

theorem quadratic_radicals_x_le_10 (a x : ℝ) (h1 : 3 * a - 8 = 17 - 2 * a) (h2 : 4 * a - 2 * x ≥ 0) : x ≤ 10 :=
by
  sorry

end quadratic_radicals_x_le_10_l223_223351


namespace rhombus_area_l223_223069

theorem rhombus_area (d1 d2 : ℕ) (h1 : d1 = 30) (h2 : d2 = 16) : (d1 * d2) / 2 = 240 := by
  sorry

end rhombus_area_l223_223069


namespace people_got_off_at_first_stop_l223_223389

theorem people_got_off_at_first_stop 
  (X : ℕ)
  (h1 : 50 - X - 6 - 1 = 28) :
  X = 15 :=
by
  sorry

end people_got_off_at_first_stop_l223_223389


namespace largest_circle_area_215_l223_223686

theorem largest_circle_area_215
  (length width : ℝ)
  (h1 : length = 16)
  (h2 : width = 10)
  (P : ℝ := 2 * (length + width))
  (C : ℝ := P)
  (r : ℝ := C / (2 * Real.pi))
  (A : ℝ := Real.pi * r^2) :
  round A = 215 := by sorry

end largest_circle_area_215_l223_223686


namespace exponentiation_identity_l223_223110

theorem exponentiation_identity (a m n : ℕ) : (a^m)^n = a^(m * n) :=
by sorry

example : (3^2)^4 = 6561 :=
by
  have h := exponentiation_identity 3 2 4
  rw [←h]
  norm_num

end exponentiation_identity_l223_223110


namespace satisfies_properties_l223_223264

noncomputable def f (x : ℝ) : ℝ := x^2

theorem satisfies_properties :
  (∀ x1 x2 : ℝ, f (x1 * x2) = f x1 * f x2) ∧
  (∀ x : ℝ, 0 < x → 0 < (f' x)) ∧
  (∀ x : ℝ, f' (-x) = - f' x) := 
sorry

end satisfies_properties_l223_223264


namespace age_of_child_l223_223151

theorem age_of_child 
  (avg_age_3_years_ago : ℕ)
  (family_size_3_years_ago : ℕ)
  (current_family_size : ℕ)
  (current_avg_age : ℕ)
  (h1 : avg_age_3_years_ago = 17)
  (h2 : family_size_3_years_ago = 5)
  (h3 : current_family_size = 6)
  (h4 : current_avg_age = 17)
  : ∃ age_of_baby : ℕ, age_of_baby = 2 := 
by
  sorry

end age_of_child_l223_223151


namespace question_l223_223931

variable (U : Set ℕ) (M : Set ℕ)

theorem question :
  U = {1, 2, 3, 4, 5} →
  (U \ M = {1, 3}) →
  2 ∈ M :=
by
  intros
  sorry

end question_l223_223931


namespace power_of_powers_eval_powers_l223_223135

theorem power_of_powers (a m n : ℕ) : (a^m)^n = a^(m * n) :=
  by sorry

theorem eval_powers : (3^2)^4 = 6561 :=
  by
    rw [power_of_powers]
    -- further proof of (3^8 = 6561) would go here
    sorry

end power_of_powers_eval_powers_l223_223135


namespace comic_book_arrangement_l223_223483

theorem comic_book_arrangement :
  let spiderman_books := 7
  let archie_books := 6
  let garfield_books := 5
  let groups := 3
  Nat.factorial spiderman_books * Nat.factorial archie_books * Nat.factorial garfield_books * Nat.factorial groups = 248005440 :=
by
  sorry

end comic_book_arrangement_l223_223483


namespace hare_overtakes_tortoise_l223_223354

noncomputable def hare_distance (t: ℕ) : ℕ := 
  if t ≤ 5 then 10 * t
  else if t ≤ 20 then 50
  else 50 + 20 * (t - 20)

noncomputable def tortoise_distance (t: ℕ) : ℕ :=
  2 * t

theorem hare_overtakes_tortoise : 
  ∃ t : ℕ, t ≤ 60 ∧ hare_distance t = tortoise_distance t ∧ 60 - t = 22 :=
sorry

end hare_overtakes_tortoise_l223_223354


namespace polynomial_expansion_l223_223713

theorem polynomial_expansion :
  (7 * x^2 + 3 * x + 1) * (5 * x^3 + 2 * x + 6) = 
  35 * x^5 + 15 * x^4 + 19 * x^3 + 48 * x^2 + 20 * x + 6 := 
by
  sorry

end polynomial_expansion_l223_223713


namespace large_posters_count_l223_223164

theorem large_posters_count (total_posters small_ratio medium_ratio : ℕ) (h_total : total_posters = 50) (h_small_ratio : small_ratio = 2/5) (h_medium_ratio : medium_ratio = 1/2) :
  let small_posters := (small_ratio * total_posters) in
  let medium_posters := (medium_ratio * total_posters) in
  let large_posters := total_posters - (small_posters + medium_posters) in
  large_posters = 5 := by
{
  sorry
}

end large_posters_count_l223_223164


namespace opposite_of_five_l223_223988

theorem opposite_of_five : ∃ y : ℤ, 5 + y = 0 ∧ y = -5 := by
  use -5
  constructor
  . exact rfl
  . sorry

end opposite_of_five_l223_223988


namespace positive_solutions_count_l223_223024

theorem positive_solutions_count :
  ∃ n : ℕ, n = 9 ∧
  (∀ (x y : ℕ), 5 * x + 10 * y = 100 → 0 < x ∧ 0 < y → (∃ k : ℕ, k < 10 ∧ n = 9)) :=
sorry

end positive_solutions_count_l223_223024


namespace num_large_posters_l223_223161

-- Define the constants
def total_posters : ℕ := 50
def small_posters : ℕ := total_posters * 2 / 5
def medium_posters : ℕ := total_posters / 2
def large_posters : ℕ := total_posters - (small_posters + medium_posters)

-- Theorem to prove the number of large posters
theorem num_large_posters : large_posters = 5 :=
by
  sorry

end num_large_posters_l223_223161


namespace factor_difference_of_squares_l223_223181

theorem factor_difference_of_squares (t : ℝ) : t^2 - 144 = (t - 12) * (t + 12) :=
by
  sorry

end factor_difference_of_squares_l223_223181


namespace quotient_remainder_scaled_l223_223227

theorem quotient_remainder_scaled (a b q r k : ℤ) (hb : b > 0) (hk : k ≠ 0) (h1 : a = b * q + r) (h2 : 0 ≤ r) (h3 : r < b) :
  a * k = (b * k) * q + (r * k) ∧ (k ∣ r → (a / k = (b / k) * q + (r / k) ∧ 0 ≤ (r / k) ∧ (r / k) < (b / k))) :=
by
  sorry

end quotient_remainder_scaled_l223_223227


namespace negative_solutions_iff_l223_223943

theorem negative_solutions_iff (m x y : ℝ) (h1 : x - y = 2 * m + 7) (h2 : x + y = 4 * m - 3) :
  (x < 0 ∧ y < 0) ↔ m < -2 / 3 :=
by
  sorry

end negative_solutions_iff_l223_223943


namespace coefficient_condition_l223_223467

theorem coefficient_condition (m : ℝ) (h : m^3 * Nat.choose 6 3 = -160) : m = -2 := sorry

end coefficient_condition_l223_223467


namespace isosceles_triangle_base_length_l223_223793

-- Define the conditions
def side_length : ℕ := 7
def perimeter : ℕ := 23

-- Define the theorem to prove the length of the base
theorem isosceles_triangle_base_length (b : ℕ) (h : 2 * side_length + b = perimeter) : b = 9 :=
by
  sorry

end isosceles_triangle_base_length_l223_223793


namespace min_value_xy_inv_xy_l223_223444

theorem min_value_xy_inv_xy (x y : ℝ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hxy_sum : x + y = 2) :
  ∃ m : ℝ, m = xy + 4 / xy ∧ m ≥ 5 :=
by
  sorry

end min_value_xy_inv_xy_l223_223444


namespace exp_eval_l223_223098

theorem exp_eval : (3^2)^4 = 6561 := by
sorry

end exp_eval_l223_223098


namespace necessary_but_not_sufficient_condition_l223_223503

variables {a b : ℤ}

theorem necessary_but_not_sufficient_condition : (¬(a = 1) ∨ ¬(b = 2)) ↔ ¬(a + b = 3) :=
by
  sorry

end necessary_but_not_sufficient_condition_l223_223503


namespace isosceles_triangle_base_length_l223_223803

theorem isosceles_triangle_base_length (a b P : ℕ) (h1 : a = 7) (h2 : P = 23) (h3 : P = 2 * a + b) : b = 9 :=
sorry

end isosceles_triangle_base_length_l223_223803


namespace factor_t_squared_minus_81_l223_223305

theorem factor_t_squared_minus_81 (t : ℝ) : t^2 - 81 = (t - 9) * (t + 9) :=
by
  sorry

end factor_t_squared_minus_81_l223_223305


namespace darnel_jog_laps_l223_223429

theorem darnel_jog_laps (x : ℝ) (h1 : 0.88 = x + 0.13) : x = 0.75 := by
  sorry

end darnel_jog_laps_l223_223429


namespace chris_current_age_l223_223373

def praveens_age_after_10_years (P : ℝ) : ℝ := P + 10
def praveens_age_3_years_back (P : ℝ) : ℝ := P - 3

def praveens_age_condition (P : ℝ) : Prop :=
  praveens_age_after_10_years P = 3 * praveens_age_3_years_back P

def chris_age (P : ℝ) : ℝ := (P - 4) - 2

theorem chris_current_age (P : ℝ) (h₁ : praveens_age_condition P) :
  chris_age P = 3.5 :=
sorry

end chris_current_age_l223_223373


namespace solve_recursive_fraction_l223_223973

noncomputable def recursive_fraction (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0     => x
  | (n+1) => 1 + 1 / (recursive_fraction n x)

theorem solve_recursive_fraction (x : ℝ) (n : ℕ) :
  (recursive_fraction n x = x) ↔ (x = (1 + Real.sqrt 5) / 2 ∨ x = (1 - Real.sqrt 5) / 2) :=
sorry

end solve_recursive_fraction_l223_223973


namespace quadratic_has_two_distinct_roots_l223_223759

theorem quadratic_has_two_distinct_roots (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x₁^2 - 2*x₁ + k = 0) ∧ (x₂^2 - 2*x₂ + k = 0))
  ↔ k < 1 :=
by sorry

end quadratic_has_two_distinct_roots_l223_223759


namespace distinct_students_count_l223_223870

-- Definition of the initial parameters
def num_gauss : Nat := 12
def num_euler : Nat := 10
def num_fibonnaci : Nat := 7
def overlap : Nat := 1

-- The main theorem to prove
theorem distinct_students_count : num_gauss + num_euler + num_fibonnaci - overlap = 28 := by
  sorry

end distinct_students_count_l223_223870


namespace intersection_A_B_l223_223363

-- Define the set A
def A : Set ℕ := {1, 2, 3, 4, 5}

-- Define the set B as the set of natural numbers greater than 2.5
def B : Set ℕ := {x : ℕ | 2 * x > 5}

-- Prove that the intersection of A and B is {3, 4, 5}
theorem intersection_A_B : A ∩ B = {3, 4, 5} :=
by sorry

end intersection_A_B_l223_223363


namespace gcd_digits_le_two_l223_223595

open Nat

theorem gcd_digits_le_two (a b : ℕ) (ha : a < 10^5) (hb : b < 10^5)
  (hlcm_digits : 10^8 ≤ lcm a b ∧ lcm a b < 10^9) : gcd a b < 100 :=
by
  -- Proof here
  sorry

end gcd_digits_le_two_l223_223595


namespace b_remaining_work_days_l223_223145

-- Definitions of the conditions
def together_work (a b: ℕ) := a + b = 12
def alone_work (a: ℕ) := a = 20
def c_work (c: ℕ) := c = 30
def initial_work_days := 5

-- Question to prove:
theorem b_remaining_work_days (a b c : ℕ) (h1 : together_work a b) (h2 : alone_work a) (h3 : c_work c) : 
  let b_rate := 1 / 30 
  let remaining_work := 25 / 60
  let work_to_days := remaining_work / b_rate
  work_to_days = 12.5 := 
sorry

end b_remaining_work_days_l223_223145


namespace gcd_two_5_digit_integers_l223_223594

theorem gcd_two_5_digit_integers (a b : ℕ) 
  (h1 : 10^4 ≤ a ∧ a < 10^5)
  (h2 : 10^4 ≤ b ∧ b < 10^5)
  (h3 : 10^8 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10^9) :
  Nat.gcd a b < 10^2 :=
by
  sorry  -- Skip the proof

end gcd_two_5_digit_integers_l223_223594


namespace length_of_train_is_400_meters_l223_223289

noncomputable def relative_speed (speed_train speed_man : ℝ) : ℝ :=
  speed_train - speed_man

noncomputable def km_per_hr_to_m_per_s (speed_km_per_hr : ℝ) : ℝ :=
  speed_km_per_hr * (1000 / 3600)

noncomputable def length_of_train (relative_speed_m_per_s time_seconds : ℝ) : ℝ :=
  relative_speed_m_per_s * time_seconds

theorem length_of_train_is_400_meters :
  let speed_train := 30 -- km/hr
  let speed_man := 6 -- km/hr
  let time_to_cross := 59.99520038396929 -- seconds
  let rel_speed := km_per_hr_to_m_per_s (relative_speed speed_train speed_man)
  length_of_train rel_speed time_to_cross = 400 :=
by
  sorry

end length_of_train_is_400_meters_l223_223289


namespace minimum_basketballs_sold_l223_223416

theorem minimum_basketballs_sold :
  ∃ (F B K : ℕ), F + B + K = 180 ∧ 3 * F + 5 * B + 10 * K = 800 ∧ F > B ∧ B > K ∧ K = 2 :=
by
  sorry

end minimum_basketballs_sold_l223_223416


namespace microphotonics_budget_allocation_l223_223676

theorem microphotonics_budget_allocation
    (home_electronics : ℕ)
    (food_additives : ℕ)
    (gen_mod_microorg : ℕ)
    (ind_lubricants : ℕ)
    (basic_astrophysics_degrees : ℕ)
    (full_circle_degrees : ℕ := 360)
    (total_budget_percentage : ℕ := 100)
    (basic_astrophysics_percentage : ℕ) :
  home_electronics = 24 →
  food_additives = 15 →
  gen_mod_microorg = 19 →
  ind_lubricants = 8 →
  basic_astrophysics_degrees = 72 →
  basic_astrophysics_percentage = (basic_astrophysics_degrees * total_budget_percentage) / full_circle_degrees →
  (total_budget_percentage -
    (home_electronics + food_additives + gen_mod_microorg + ind_lubricants + basic_astrophysics_percentage)) = 14 :=
by
  intros he fa gmm il bad bp
  sorry

end microphotonics_budget_allocation_l223_223676


namespace correct_answer_l223_223022

def vector := (Int × Int)

-- Definitions of vectors given in conditions
def m : vector := (2, 1)
def n : vector := (0, -2)

def vec_add (v1 v2 : vector) : vector :=
  (v1.1 + v2.1, v1.2 + v2.2)

def vec_scalar_mult (c : Int) (v : vector) : vector :=
  (c * v.1, c * v.2)

def vec_dot (v1 v2 : vector) : Int :=
  v1.1 * v2.1 + v1.2 * v2.2

-- The condition vector combined
def combined_vector := vec_add m (vec_scalar_mult 2 n)

-- The problem is to prove this:
theorem correct_answer : vec_dot (3, 2) combined_vector = 0 :=
  sorry

end correct_answer_l223_223022


namespace solve_for_x_l223_223505

theorem solve_for_x (x : ℝ) (h : 3 * x + 15 = 1 / 3 * (6 * x + 45)) : x = 0 :=
sorry

end solve_for_x_l223_223505


namespace exponentiation_identity_l223_223106

theorem exponentiation_identity (a m n : ℕ) : (a^m)^n = a^(m * n) :=
by sorry

example : (3^2)^4 = 6561 :=
by
  have h := exponentiation_identity 3 2 4
  rw [←h]
  norm_num

end exponentiation_identity_l223_223106


namespace maximum_unique_numbers_in_circle_l223_223866

theorem maximum_unique_numbers_in_circle :
  ∀ (n : ℕ) (numbers : ℕ → ℤ), n = 2023 →
  (∀ i, numbers i = numbers ((i + 1) % n) * numbers ((i + n - 1) % n)) →
  ∀ i j, numbers i = numbers j :=
by
  sorry

end maximum_unique_numbers_in_circle_l223_223866


namespace car_distance_calculation_l223_223279

noncomputable def total_distance (u a v t1 t2: ℝ) : ℝ :=
  let d1 := (u * t1) + (1 / 2) * a * t1^2
  let d2 := v * t2
  d1 + d2

theorem car_distance_calculation :
  total_distance 30 5 60 2 3 = 250 :=
by
  unfold total_distance
  -- next steps include simplifying the math, but we'll defer details to proof
  sorry

end car_distance_calculation_l223_223279


namespace kyle_delivers_daily_papers_l223_223361

theorem kyle_delivers_daily_papers (x : ℕ) (h : 6 * x + (x - 10) + 30 = 720) : x = 100 :=
by
  sorry

end kyle_delivers_daily_papers_l223_223361


namespace compute_f_1986_l223_223755

noncomputable def f : ℕ → ℤ := sorry

axiom f_defined_for_nonneg_integers : ∀ x : ℕ, ∃ y : ℤ, f x = y
axiom f_one : f 1 = 1
axiom f_functional_equation : ∀ (a b : ℕ), f (a + b) = f a + f b - 2 * f (a * b)

theorem compute_f_1986 : f 1986 = 0 :=
  sorry

end compute_f_1986_l223_223755


namespace evaluate_expression_l223_223712

theorem evaluate_expression :
  (Int.floor ((Int.ceil ((11/5:ℚ)^2)) * (19/3:ℚ))) = 31 :=
by
  sorry

end evaluate_expression_l223_223712


namespace factor_t_squared_minus_81_l223_223303

theorem factor_t_squared_minus_81 (t : ℂ) : (t^2 - 81) = (t - 9) * (t + 9) := 
by
  -- We apply the identity a^2 - b^2 = (a - b) * (a + b)
  let a := t
  let b := 9
  have eq : t^2 - 81 = a^2 - b^2 := by sorry
  rw [eq]
  exact (mul_sub_mul_add_eq_sq_sub_sq a b).symm
  -- Concluding the proof
  sorry -- skipping detailed proof steps for now

end factor_t_squared_minus_81_l223_223303


namespace find_A_l223_223175

theorem find_A (A B C : ℕ) (h1 : A ≠ B ∧ A ≠ C ∧ B ≠ C) 
  (h2 : 1 ≤ A ∧ A ≤ 9 ∧ 1 ≤ B ∧ B ≤ 9 ∧ 0 ≤ C ∧ C ≤ 9)
  (h3 : A * 10 + B + B * 10 + C = B * 100 + C * 10 + B) : 
  A = 9 :=
  sorry

end find_A_l223_223175


namespace expected_value_is_minus_one_fifth_l223_223688

-- Define the parameters given in the problem
def p_heads := 2 / 5
def p_tails := 3 / 5
def win_heads := 4
def loss_tails := -3

-- Calculate the expected value for heads and tails
def expected_heads := p_heads * win_heads
def expected_tails := p_tails * loss_tails

-- The theorem stating that the expected value is -1/5
theorem expected_value_is_minus_one_fifth :
  expected_heads + expected_tails = -1 / 5 :=
by
  -- The proof can be filled in here
  sorry

end expected_value_is_minus_one_fifth_l223_223688


namespace imag_part_of_complex_l223_223436

open Complex

theorem imag_part_of_complex : (im ((5 + I) / (1 + I))) = -2 :=
by
  sorry

end imag_part_of_complex_l223_223436


namespace num_large_posters_l223_223160

-- Define the constants
def total_posters : ℕ := 50
def small_posters : ℕ := total_posters * 2 / 5
def medium_posters : ℕ := total_posters / 2
def large_posters : ℕ := total_posters - (small_posters + medium_posters)

-- Theorem to prove the number of large posters
theorem num_large_posters : large_posters = 5 :=
by
  sorry

end num_large_posters_l223_223160


namespace sum_of_three_ints_product_5_4_l223_223815

theorem sum_of_three_ints_product_5_4 :
  ∃ (a b c: ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ a * b * c = 5^4 ∧ a + b + c = 51 :=
by
  sorry

end sum_of_three_ints_product_5_4_l223_223815


namespace james_total_socks_l223_223768

-- Definitions based on conditions
def red_pairs : ℕ := 20
def black_pairs : ℕ := red_pairs / 2
def white_pairs : ℕ := 2 * (red_pairs + black_pairs)
def green_pairs : ℕ := (red_pairs + black_pairs + white_pairs) + 5

-- Total number of pairs
def total_pairs := red_pairs + black_pairs + white_pairs + green_pairs

-- Total number of socks
def total_socks := total_pairs * 2

-- The main theorem to prove the total number of socks
theorem james_total_socks : total_socks = 370 :=
  by
  -- proof is skipped
  sorry

end james_total_socks_l223_223768


namespace average_people_added_each_year_l223_223423

-- a) Identifying questions and conditions
-- Question: What is the average number of people added each year?
-- Conditions: In 2000, about 450,000 people lived in Maryville. In 2005, about 467,000 people lived in Maryville.

-- c) Mathematically equivalent proof problem
-- Mathematically equivalent proof problem: Prove that the average number of people added each year is 3400 given the conditions.

-- d) Lean 4 statement
theorem average_people_added_each_year :
  let population_2000 := 450000
  let population_2005 := 467000
  let years_passed := 2005 - 2000
  let total_increase := population_2005 - population_2000
  total_increase / years_passed = 3400 := by
    sorry

end average_people_added_each_year_l223_223423


namespace greatest_non_fiction_books_l223_223353

def is_prime (p : ℕ) := p > 1 ∧ (∀ d : ℕ, d ∣ p → d = 1 ∨ d = p)

theorem greatest_non_fiction_books (n f k : ℕ) :
  (n + f = 100 ∧ f = n + k ∧ is_prime k) → n ≤ 49 :=
by
  sorry

end greatest_non_fiction_books_l223_223353


namespace maria_min_score_fourth_quarter_l223_223078

theorem maria_min_score_fourth_quarter (x : ℝ) :
  (82 + 77 + 78 + x) / 4 ≥ 85 ↔ x ≥ 103 :=
by
  sorry

end maria_min_score_fourth_quarter_l223_223078
