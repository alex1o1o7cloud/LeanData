import Mathlib

namespace NUMINAMATH_GPT_reciprocal_of_neg3_l2310_231097

theorem reciprocal_of_neg3 : ∃ x : ℝ, -3 * x = 1 ∧ x = -1/3 :=
by
  sorry

end NUMINAMATH_GPT_reciprocal_of_neg3_l2310_231097


namespace NUMINAMATH_GPT_janine_read_pages_in_two_months_l2310_231035

theorem janine_read_pages_in_two_months :
  (let books_last_month := 5
   let books_this_month := 2 * books_last_month
   let total_books := books_last_month + books_this_month
   let pages_per_book := 10
   total_books * pages_per_book = 150) := by
   sorry

end NUMINAMATH_GPT_janine_read_pages_in_two_months_l2310_231035


namespace NUMINAMATH_GPT_triangle_inequality_l2310_231063

theorem triangle_inequality
  (a b c x y z : ℝ)
  (h_order : a < b ∧ b < c ∧ 0 < x)
  (h_area_eq : c * x = a * y + b * z) :
  x < y + z :=
by
  sorry

end NUMINAMATH_GPT_triangle_inequality_l2310_231063


namespace NUMINAMATH_GPT_part1_part2_l2310_231082

def A (t : ℝ) : Prop :=
  ∀ x : ℝ, (t+2)*x^2 + 2*x + 1 > 0

def B (a x : ℝ) : Prop :=
  (a*x - 1)*(x + a) > 0

theorem part1 (t : ℝ) : A t ↔ t < -1 :=
sorry

theorem part2 (a : ℝ) : (∀ t : ℝ, t < -1 → ∀ x : ℝ, B a x) → (0 ≤ a ∧ a ≤ 1) :=
sorry

end NUMINAMATH_GPT_part1_part2_l2310_231082


namespace NUMINAMATH_GPT_total_obstacle_course_time_l2310_231009

-- Definitions for the given conditions
def first_part_time : Nat := 7 * 60 + 23
def second_part_time : Nat := 73
def third_part_time : Nat := 5 * 60 + 58

-- State the main theorem
theorem total_obstacle_course_time :
  first_part_time + second_part_time + third_part_time = 874 :=
by
  sorry

end NUMINAMATH_GPT_total_obstacle_course_time_l2310_231009


namespace NUMINAMATH_GPT_sales_tax_paid_l2310_231071

theorem sales_tax_paid 
  (total_spent : ℝ) 
  (tax_free_cost : ℝ) 
  (tax_rate : ℝ) 
  (cost_of_taxable_items : ℝ) 
  (sales_tax : ℝ) 
  (h1 : total_spent = 40) 
  (h2 : tax_free_cost = 34.7) 
  (h3 : tax_rate = 0.06) 
  (h4 : cost_of_taxable_items = 5) 
  (h5 : sales_tax = 0.3) 
  (h6 : 1.06 * cost_of_taxable_items + tax_free_cost = total_spent) : 
  sales_tax = tax_rate * cost_of_taxable_items :=
sorry

end NUMINAMATH_GPT_sales_tax_paid_l2310_231071


namespace NUMINAMATH_GPT_reggie_marbles_l2310_231031

/-- Given that Reggie and his friend played 9 games in total,
    Reggie lost 1 game, and they bet 10 marbles per game.
    Prove that Reggie has 70 marbles after all games. -/
theorem reggie_marbles (total_games : ℕ) (lost_games : ℕ) (marbles_per_game : ℕ) (marbles_initial : ℕ) 
  (h_total_games : total_games = 9) (h_lost_games : lost_games = 1) (h_marbles_per_game : marbles_per_game = 10) 
  (h_marbles_initial : marbles_initial = 0) : 
  marbles_initial + (total_games - lost_games) * marbles_per_game - lost_games * marbles_per_game = 70 :=
by
  -- We proved this in the solution steps, but will skip the proof here with sorry.
  sorry

end NUMINAMATH_GPT_reggie_marbles_l2310_231031


namespace NUMINAMATH_GPT_midpoint_trajectory_l2310_231096

   -- Defining the given conditions
   def P_moves_on_circle (x1 y1 : ℝ) : Prop :=
     (x1 + 1)^2 + y1^2 = 4

   def Q_coordinates : (ℝ × ℝ) := (4, 3)

   -- Defining the midpoint relationship
   def midpoint_relation (x y x1 y1 : ℝ) : Prop :=
     x1 + Q_coordinates.1 = 2 * x ∧ y1 + Q_coordinates.2 = 2 * y

   -- Proving the trajectory equation of the midpoint M
   theorem midpoint_trajectory (x y : ℝ) : 
     (∃ x1 y1 : ℝ, midpoint_relation x y x1 y1 ∧ P_moves_on_circle x1 y1) →
     (x - 3/2)^2 + (y - 3/2)^2 = 1 :=
   by
     intros h
     sorry
   
end NUMINAMATH_GPT_midpoint_trajectory_l2310_231096


namespace NUMINAMATH_GPT_symmetrical_line_range_l2310_231062

theorem symmetrical_line_range {k : ℝ} :
  (∀ x y : ℝ, (y = k * x - 1) ∧ (x + y - 1 = 0) → y ≠ -x + 1) → k > 1 ↔ k > 1 :=
by
  sorry

end NUMINAMATH_GPT_symmetrical_line_range_l2310_231062


namespace NUMINAMATH_GPT_children_absent_on_independence_day_l2310_231069

theorem children_absent_on_independence_day
  (total_children : ℕ)
  (bananas_per_child : ℕ)
  (extra_bananas : ℕ)
  (total_possible_children : total_children = 780)
  (bananas_distributed : bananas_per_child = 2)
  (additional_bananas : extra_bananas = 2) :
  ∃ (A : ℕ), A = 390 := 
sorry

end NUMINAMATH_GPT_children_absent_on_independence_day_l2310_231069


namespace NUMINAMATH_GPT_domain_of_sqrt_1_minus_2_cos_l2310_231005

theorem domain_of_sqrt_1_minus_2_cos (x : ℝ) (k : ℤ) :
  1 - 2 * Real.cos x ≥ 0 ↔ ∃ k : ℤ, (π / 3 + 2 * k * π ≤ x ∧ x ≤ 5 * π / 3 + 2 * k * π) :=
by
  sorry

end NUMINAMATH_GPT_domain_of_sqrt_1_minus_2_cos_l2310_231005


namespace NUMINAMATH_GPT_soybeans_to_oil_l2310_231033

theorem soybeans_to_oil 
    (kg_soybeans_to_tofu : ℝ)
    (kg_soybeans_to_oil : ℝ)
    (price_soybeans : ℝ)
    (price_tofu : ℝ)
    (price_oil : ℝ)
    (purchase_amount : ℝ)
    (sales_amount : ℝ)
    (amount_to_oil : ℝ)
    (used_soybeans_for_oil : ℝ) :
    kg_soybeans_to_tofu = 3 →
    kg_soybeans_to_oil = 6 →
    price_soybeans = 2 →
    price_tofu = 3 →
    price_oil = 15 →
    purchase_amount = 920 →
    sales_amount = 1800 →
    used_soybeans_for_oil = 360 →
    (6 * amount_to_oil) = 360 →
    15 * amount_to_oil + 3 * (460 - 6 * amount_to_oil) = 1800 :=
by sorry

end NUMINAMATH_GPT_soybeans_to_oil_l2310_231033


namespace NUMINAMATH_GPT_decimal_to_base7_l2310_231051

-- Define the decimal number
def decimal_number : ℕ := 2011

-- Define the base-7 conversion function
def to_base7 (n : ℕ) : List ℕ :=
  if n < 7 then [n]
  else to_base7 (n / 7) ++ [n % 7]

-- Calculate the base-7 representation of 2011
def base7_representation : List ℕ := to_base7 decimal_number

-- Prove that the base-7 representation of 2011 is [5, 6, 0, 2]
theorem decimal_to_base7 : base7_representation = [5, 6, 0, 2] :=
  by sorry

end NUMINAMATH_GPT_decimal_to_base7_l2310_231051


namespace NUMINAMATH_GPT_ellipse_equation_from_hyperbola_l2310_231034

theorem ellipse_equation_from_hyperbola :
  (∃ (a b : ℝ), ∀ x y : ℝ, (x^2 / 4 - y^2 / 12 = 1) →
  (x^2 / 16 + y^2 / 12 = 1)) :=
by
  sorry

end NUMINAMATH_GPT_ellipse_equation_from_hyperbola_l2310_231034


namespace NUMINAMATH_GPT_estimated_total_fish_population_l2310_231076

-- Definitions of the initial conditions
def tagged_fish_in_first_catch : ℕ := 100
def total_fish_in_second_catch : ℕ := 300
def tagged_fish_in_second_catch : ℕ := 15

-- The theorem to prove the estimated number of total fish in the pond
theorem estimated_total_fish_population (tagged_fish_in_first_catch : ℕ) (total_fish_in_second_catch : ℕ) (tagged_fish_in_second_catch : ℕ) : ℕ :=
  2000

-- Assertion of the theorem with actual numbers
example : estimated_total_fish_population tagged_fish_in_first_catch total_fish_in_second_catch tagged_fish_in_second_catch = 2000 := by
  sorry

end NUMINAMATH_GPT_estimated_total_fish_population_l2310_231076


namespace NUMINAMATH_GPT_consecutive_integers_greatest_l2310_231088

theorem consecutive_integers_greatest (n : ℤ) (h : n + 2 = 8) : 
  (n + 2 = 8) → (max n (max (n + 1) (n + 2)) = 8) :=
by {
  sorry
}

end NUMINAMATH_GPT_consecutive_integers_greatest_l2310_231088


namespace NUMINAMATH_GPT_XY_sym_diff_l2310_231015

-- The sets X and Y
def X : Set ℤ := {1, 3, 5, 7}
def Y : Set ℤ := { x | x < 4 ∧ x ∈ Set.univ }

-- Definition of set operation (A - B)
def set_sub (A B : Set ℤ) : Set ℤ := { x | x ∈ A ∧ x ∉ B }

-- Definition of set operation (A * B)
def set_sym_diff (A B : Set ℤ) : Set ℤ := (set_sub A B) ∪ (set_sub B A)

-- Prove that X * Y = {-3, -2, -1, 0, 2, 5, 7}
theorem XY_sym_diff : set_sym_diff X Y = {-3, -2, -1, 0, 2, 5, 7} :=
by
  sorry

end NUMINAMATH_GPT_XY_sym_diff_l2310_231015


namespace NUMINAMATH_GPT_positive_difference_of_squares_l2310_231002

theorem positive_difference_of_squares (a b : ℕ) (h1 : a + b = 40) (h2 : a - b = 8) : a^2 - b^2 = 320 :=
by
  sorry

end NUMINAMATH_GPT_positive_difference_of_squares_l2310_231002


namespace NUMINAMATH_GPT_pounds_of_fudge_sold_l2310_231053

variable (F : ℝ)
variable (price_fudge price_truffles price_pretzels total_revenue : ℝ)

def conditions := 
  price_fudge = 2.50 ∧
  price_truffles = 60 * 1.50 ∧
  price_pretzels = 36 * 2.00 ∧
  total_revenue = 212 ∧
  total_revenue = (price_fudge * F) + price_truffles + price_pretzels

theorem pounds_of_fudge_sold (F : ℝ) (price_fudge price_truffles price_pretzels total_revenue : ℝ) 
  (h : conditions F price_fudge price_truffles price_pretzels total_revenue ) :
  F = 20 :=
by
  sorry

end NUMINAMATH_GPT_pounds_of_fudge_sold_l2310_231053


namespace NUMINAMATH_GPT_identifyNewEnergySources_l2310_231086

-- Definitions of energy types as elements of a set.
inductive EnergySource 
| NaturalGas
| Coal
| OceanEnergy
| Petroleum
| SolarEnergy
| BiomassEnergy
| WindEnergy
| HydrogenEnergy

open EnergySource

-- Set definition for types of new energy sources
def newEnergySources : Set EnergySource := 
  { OceanEnergy, SolarEnergy, BiomassEnergy, WindEnergy, HydrogenEnergy }

-- Set definition for the correct answer set of new energy sources identified by Option B
def optionB : Set EnergySource := 
  { OceanEnergy, SolarEnergy, BiomassEnergy, WindEnergy, HydrogenEnergy }

-- The theorem asserting the equivalence between the identified new energy sources and the set option B
theorem identifyNewEnergySources : newEnergySources = optionB :=
  sorry

end NUMINAMATH_GPT_identifyNewEnergySources_l2310_231086


namespace NUMINAMATH_GPT_cyclist_overtake_points_l2310_231099

theorem cyclist_overtake_points (p c : ℝ) (track_length : ℝ) (h1 : c = 1.55 * p) (h2 : track_length = 55) : 
  ∃ n, n = 11 :=
by
  -- we'll add the proof steps later
  sorry

end NUMINAMATH_GPT_cyclist_overtake_points_l2310_231099


namespace NUMINAMATH_GPT_distance_to_axes_l2310_231038

def point (P : ℝ × ℝ) : Prop :=
  P = (3, 5)

theorem distance_to_axes (P : ℝ × ℝ) (hx : P = (3, 5)) : 
  abs P.2 = 5 ∧ abs P.1 = 3 :=
by 
  sorry

end NUMINAMATH_GPT_distance_to_axes_l2310_231038


namespace NUMINAMATH_GPT_sum_six_terms_l2310_231073

variable (S : ℕ → ℝ)
variable (n : ℕ)
variable (S_2 S_4 S_6 : ℝ)

-- Given conditions
axiom sum_two_terms : S 2 = 4
axiom sum_four_terms : S 4 = 16

-- Problem statement
theorem sum_six_terms : S 6 = 52 :=
by
  -- Insert the proof here
  sorry

end NUMINAMATH_GPT_sum_six_terms_l2310_231073


namespace NUMINAMATH_GPT_largest_possible_P10_l2310_231029

noncomputable def P (x : ℤ) : ℤ := x^2 + 3*x + 3

theorem largest_possible_P10 : P 10 = 133 := by
  sorry

end NUMINAMATH_GPT_largest_possible_P10_l2310_231029


namespace NUMINAMATH_GPT_cafeteria_orders_green_apples_l2310_231079

theorem cafeteria_orders_green_apples (G : ℕ) (h1 : 6 + G = 5 + 16) : G = 15 :=
by
  sorry

end NUMINAMATH_GPT_cafeteria_orders_green_apples_l2310_231079


namespace NUMINAMATH_GPT_number_of_valid_integers_l2310_231007

theorem number_of_valid_integers (n : ℕ) (h1 : n ≤ 2021) (h2 : ∀ m : ℕ, m^2 ≤ n → n < (m + 1)^2 → ((m^2 + 1) ∣ (n^2 + 1))) : 
  ∃ k, k = 47 :=
by
  sorry

end NUMINAMATH_GPT_number_of_valid_integers_l2310_231007


namespace NUMINAMATH_GPT_cost_of_pencil_l2310_231074

theorem cost_of_pencil (x y : ℕ) (h1 : 4 * x + 3 * y = 224) (h2 : 2 * x + 5 * y = 154) : y = 12 := 
by
  sorry

end NUMINAMATH_GPT_cost_of_pencil_l2310_231074


namespace NUMINAMATH_GPT_jane_total_worth_l2310_231090

open Nat

theorem jane_total_worth (q d : ℕ) (h1 : q + d = 30)
  (h2 : 25 * q + 10 * d + 150 = 10 * q + 25 * d) :
  25 * q + 10 * d = 450 :=
by
  sorry

end NUMINAMATH_GPT_jane_total_worth_l2310_231090


namespace NUMINAMATH_GPT_solve_for_k_l2310_231092

theorem solve_for_k (k : ℕ) (h : 16 / k = 4) : k = 4 :=
sorry

end NUMINAMATH_GPT_solve_for_k_l2310_231092


namespace NUMINAMATH_GPT_problem_part_1_problem_part_2_l2310_231014
open Set Real

noncomputable def A (a : ℝ) : Set ℝ := {x | a < x ∧ x ≤ a^2 - 2}
noncomputable def B : Set ℝ := {x | 1 < x ∧ x < 5}

theorem problem_part_1 : A 3 ∪ B = {x | 1 < x ∧ x ≤ 7} := 
  by
  sorry

theorem problem_part_2 : (∀ a : ℝ, A a ∪ B = B → 2 < a ∧ a < sqrt 7) :=
  by 
  sorry

end NUMINAMATH_GPT_problem_part_1_problem_part_2_l2310_231014


namespace NUMINAMATH_GPT_smallest_possible_n_l2310_231032

theorem smallest_possible_n (n : ℕ) (h1 : 0 < n) (h2 : 0 < 60) 
  (h3 : (Nat.lcm 60 n) / (Nat.gcd 60 n) = 24) : n = 20 :=
by sorry

end NUMINAMATH_GPT_smallest_possible_n_l2310_231032


namespace NUMINAMATH_GPT_Brad_age_l2310_231049

theorem Brad_age (shara_age : ℕ) (h_shara : shara_age = 10)
  (jaymee_age : ℕ) (h_jaymee : jaymee_age = 2 * shara_age + 2)
  (brad_age : ℕ) (h_brad : brad_age = (shara_age + jaymee_age) / 2 - 3) : brad_age = 13 := by
  sorry

end NUMINAMATH_GPT_Brad_age_l2310_231049


namespace NUMINAMATH_GPT_boy_travel_speed_l2310_231094

theorem boy_travel_speed 
  (v : ℝ)
  (travel_distance : ℝ := 10) 
  (return_speed : ℝ := 2) 
  (total_time : ℝ := 5.8)
  (distance : ℝ := 9.999999999999998) :
  (v = 12.5) → (travel_distance = distance) →
  (total_time = (travel_distance / v) + (travel_distance / return_speed)) :=
by
  sorry

end NUMINAMATH_GPT_boy_travel_speed_l2310_231094


namespace NUMINAMATH_GPT_percentage_reduction_l2310_231045

variable (P R : ℝ)
variable (ReducedPrice : R = 15)
variable (AmountMore : 900 / 15 - 900 / P = 6)

theorem percentage_reduction (ReducedPrice : R = 15) (AmountMore : 900 / 15 - 900 / P = 6) :
  (P - R) / P * 100 = 10 :=
by
  sorry

end NUMINAMATH_GPT_percentage_reduction_l2310_231045


namespace NUMINAMATH_GPT_reciprocal_power_l2310_231018

theorem reciprocal_power (a : ℝ) (h : 1 / a = -1) : a ^ 2023 = -1 :=
by sorry

end NUMINAMATH_GPT_reciprocal_power_l2310_231018


namespace NUMINAMATH_GPT_intersection_two_elements_l2310_231013

open Real Set

-- Definitions
def M (k : ℝ) : Set (ℝ × ℝ) := {p | ∃ x y : ℝ, p = (x, y) ∧ y = k * (x - 1) + 1}
def N : Set (ℝ × ℝ) := {p | ∃ x y : ℝ, p = (x, y) ∧ x^2 + y^2 - 2 * y = 0}

-- Statement of the problem
theorem intersection_two_elements (k : ℝ) (hk : k ≠ 0) :
  ∃ x1 y1 x2 y2 : ℝ,
    (x1, y1) ∈ M k ∧ (x1, y1) ∈ N ∧ 
    (x2, y2) ∈ M k ∧ (x2, y2) ∈ N ∧ 
    (x1, y1) ≠ (x2, y2) := sorry

end NUMINAMATH_GPT_intersection_two_elements_l2310_231013


namespace NUMINAMATH_GPT_total_marbles_in_bowls_l2310_231026

theorem total_marbles_in_bowls :
  let second_bowl := 600
  let first_bowl := 3 / 4 * second_bowl
  let third_bowl := 1 / 2 * first_bowl
  let fourth_bowl := 1 / 3 * second_bowl
  first_bowl + second_bowl + third_bowl + fourth_bowl = 1475 :=
by
  sorry

end NUMINAMATH_GPT_total_marbles_in_bowls_l2310_231026


namespace NUMINAMATH_GPT_eight_distinct_solutions_l2310_231003

noncomputable def f (x : ℝ) : ℝ := x^2 - 2

theorem eight_distinct_solutions : 
  ∃ S : Finset ℝ, S.card = 8 ∧ ∀ x ∈ S, f (f (f x)) = x :=
sorry

end NUMINAMATH_GPT_eight_distinct_solutions_l2310_231003


namespace NUMINAMATH_GPT_sarah_interview_combinations_l2310_231021

theorem sarah_interview_combinations : 
  (1 * 2 * (2 + 3) * 5 * 1) = 50 := 
by
  sorry

end NUMINAMATH_GPT_sarah_interview_combinations_l2310_231021


namespace NUMINAMATH_GPT_find_q_sum_of_bn_l2310_231048

-- Defining the sequences and conditions
def a (n : ℕ) (q : ℝ) : ℝ := q^(n-1)

def b (n : ℕ) (q : ℝ) : ℝ := a n q + n

-- Given that 2a_1, (1/2)a_3, a_2 form an arithmetic sequence
def condition_arithmetic_sequence (q : ℝ) : Prop :=
  2 * a 1 q + a 2 q = (1 / 2) * a 3 q + (1 / 2) * a 3 q

-- To be proved: Given conditions, prove q = 2
theorem find_q : ∃ q > 0, a 1 q = 1 ∧ a 2 q = q ∧ a 3 q = q^2 ∧ condition_arithmetic_sequence q ∧ q = 2 :=
by {
  sorry
}

-- Given b_n = a_n + n, prove T_n = (n(n+1))/2 + 2^n - 1
theorem sum_of_bn (n : ℕ) : 
  ∃ T_n : ℕ → ℝ, T_n n = (n * (n + 1)) / 2 + (2^n) - 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_q_sum_of_bn_l2310_231048


namespace NUMINAMATH_GPT_cost_of_later_purchase_l2310_231087

-- Define the costs of bats and balls as constants.
def cost_of_bat : ℕ := 500
def cost_of_ball : ℕ := 100

-- Define the quantities involved in the later purchase.
def bats_purchased_later : ℕ := 3
def balls_purchased_later : ℕ := 5

-- Define the expected total cost for the later purchase.
def expected_total_cost_later : ℕ := 2000

-- The theorem to be proved: the cost of the later purchase of bats and balls is $2000.
theorem cost_of_later_purchase :
  bats_purchased_later * cost_of_bat + balls_purchased_later * cost_of_ball = expected_total_cost_later :=
sorry

end NUMINAMATH_GPT_cost_of_later_purchase_l2310_231087


namespace NUMINAMATH_GPT_integer_values_count_l2310_231025

theorem integer_values_count (x : ℤ) :
  ∃ k, (∀ n : ℤ, (3 ≤ Real.sqrt (3 * n + 1) ∧ Real.sqrt (3 * n + 1) < 5) ↔ ((n = 3) ∨ (n = 4) ∨ (n = 5) ∨ (n = 6) ∨ (n = 7)) ∧ k = 5) :=
by
  sorry

end NUMINAMATH_GPT_integer_values_count_l2310_231025


namespace NUMINAMATH_GPT_rhombus_diagonal_sum_maximum_l2310_231057

theorem rhombus_diagonal_sum_maximum 
    (x y : ℝ) 
    (h1 : x^2 + y^2 = 100) 
    (h2 : x ≥ 6) 
    (h3 : y ≤ 6) : 
    x + y = 14 :=
sorry

end NUMINAMATH_GPT_rhombus_diagonal_sum_maximum_l2310_231057


namespace NUMINAMATH_GPT_triangle_inequality_from_inequality_l2310_231078

theorem triangle_inequality_from_inequality
  (a b c : ℝ)
  (h : 0 < a ∧ 0 < b ∧ 0 < c)
  (ineq : (a^2 + b^2 + c^2)^2 > 2 * (a^4 + b^4 + c^4)) :
  a + b > c ∧ b + c > a ∧ c + a > b :=
by
  sorry

end NUMINAMATH_GPT_triangle_inequality_from_inequality_l2310_231078


namespace NUMINAMATH_GPT_find_n_l2310_231077

-- Defining necessary conditions and declarations
def isThreeDigit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def sumOfDigits (n : ℕ) : ℕ :=
  n / 100 + (n / 10) % 10 + n % 10

def productOfDigits (n : ℕ) : ℕ :=
  (n / 100) * ((n / 10) % 10) * (n % 10)

theorem find_n (n : ℕ) (s : ℕ) (p : ℕ) 
  (h1 : isThreeDigit n) 
  (h2 : isPerfectSquare n) 
  (h3 : sumOfDigits n = s) 
  (h4 : productOfDigits n = p) 
  (h5 : 10 ≤ s ∧ s < 100)
  (h6 : ∀ m : ℕ, isThreeDigit m → isPerfectSquare m → sumOfDigits m = s → productOfDigits m = p → (m = n → false))
  (h7 : ∃ m : ℕ, isThreeDigit m ∧ isPerfectSquare m ∧ sumOfDigits m = s ∧ productOfDigits m = p ∧ (∃ k : ℕ, k ≠ m → true)) :
  n = 841 :=
sorry

end NUMINAMATH_GPT_find_n_l2310_231077


namespace NUMINAMATH_GPT_neznaika_made_mistake_l2310_231030

-- Define the total digits used from 1 to N pages
def totalDigits (N : ℕ) : ℕ :=
  let single_digit_pages := min N 9
  let double_digit_pages := if N > 9 then N - 9 else 0
  single_digit_pages * 1 + double_digit_pages * 2

-- The main statement we want to prove
theorem neznaika_made_mistake : ¬ ∃ N : ℕ, totalDigits N = 100 :=
by
  sorry

end NUMINAMATH_GPT_neznaika_made_mistake_l2310_231030


namespace NUMINAMATH_GPT_kim_paid_with_amount_l2310_231059

-- Define the conditions
def meal_cost : ℝ := 10
def drink_cost : ℝ := 2.5
def tip_rate : ℝ := 0.20
def change_received : ℝ := 5

-- Define the total amount paid formula
def total_cost_before_tip := meal_cost + drink_cost
def tip_amount := tip_rate * total_cost_before_tip
def total_cost_after_tip := total_cost_before_tip + tip_amount
def amount_paid := total_cost_after_tip + change_received

-- Statement of the theorem
theorem kim_paid_with_amount : amount_paid = 20 := by
  sorry

end NUMINAMATH_GPT_kim_paid_with_amount_l2310_231059


namespace NUMINAMATH_GPT_largest_integer_solution_l2310_231095

theorem largest_integer_solution (x : ℤ) (h : (x : ℚ) / 3 + 4 / 5 < 5 / 3) : x ≤ 2 :=
sorry

end NUMINAMATH_GPT_largest_integer_solution_l2310_231095


namespace NUMINAMATH_GPT_sum_of_roots_of_abs_quadratic_is_zero_l2310_231008

theorem sum_of_roots_of_abs_quadratic_is_zero : 
  ∀ x : ℝ, (|x|^2 + |x| - 6 = 0) → (x = 2 ∨ x = -2) → (2 + (-2) = 0) :=
by
  intros x h h1
  sorry

end NUMINAMATH_GPT_sum_of_roots_of_abs_quadratic_is_zero_l2310_231008


namespace NUMINAMATH_GPT_sum_of_numbers_l2310_231083

theorem sum_of_numbers : 
  (87 + 91 + 94 + 88 + 93 + 91 + 89 + 87 + 92 + 86 + 90 + 92 + 88 + 90 + 91 + 86 + 89 + 92 + 95 + 88) = 1799 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_numbers_l2310_231083


namespace NUMINAMATH_GPT_find_radius_l2310_231042

theorem find_radius (AB EO : ℝ) (AE BE : ℝ) (h1 : AB = AE + BE) (h2 : AE = 2 * BE) (h3 : EO = 7) :
  ∃ R : ℝ, R = 11 := by
  sorry

end NUMINAMATH_GPT_find_radius_l2310_231042


namespace NUMINAMATH_GPT_symmetric_axis_and_vertex_l2310_231060

theorem symmetric_axis_and_vertex (x : ℝ) : 
  (∀ x y, y = (1 / 2) * (x - 1)^2 + 6 → x = 1) 
  ∧ (1, 6) = (1, 6) :=
by 
  sorry

end NUMINAMATH_GPT_symmetric_axis_and_vertex_l2310_231060


namespace NUMINAMATH_GPT_dad_caught_more_l2310_231047

theorem dad_caught_more {trouts_caleb : ℕ} (h₁ : trouts_caleb = 2) 
    (h₂ : ∃ trouts_dad : ℕ, trouts_dad = 3 * trouts_caleb) : 
    ∃ more_trouts : ℕ, more_trouts = 4 := by
  sorry

end NUMINAMATH_GPT_dad_caught_more_l2310_231047


namespace NUMINAMATH_GPT_company_fund_initial_amount_l2310_231085

theorem company_fund_initial_amount (n : ℕ) (fund_initial : ℤ) 
  (h1 : ∃ n, fund_initial = 60 * n - 10)
  (h2 : ∃ n, 55 * n + 120 = fund_initial + 130)
  : fund_initial = 1550 := 
sorry

end NUMINAMATH_GPT_company_fund_initial_amount_l2310_231085


namespace NUMINAMATH_GPT_percentage_of_gold_coins_is_35_percent_l2310_231016

-- Definitions of conditions
def percentage_of_objects_that_are_beads : ℝ := 0.30
def percentage_of_coins_that_are_silver : ℝ := 0.25
def percentage_of_coins_that_are_gold : ℝ := 0.50

-- Problem Statement
theorem percentage_of_gold_coins_is_35_percent 
  (h_beads : percentage_of_objects_that_are_beads = 0.30) 
  (h_silver_coins : percentage_of_coins_that_are_silver = 0.25) 
  (h_gold_coins : percentage_of_coins_that_are_gold = 0.50) :
  0.35 = 0.35 := 
sorry

end NUMINAMATH_GPT_percentage_of_gold_coins_is_35_percent_l2310_231016


namespace NUMINAMATH_GPT_john_total_spent_l2310_231011

noncomputable def calculate_total_spent : ℝ :=
  let orig_price_A := 900.0
  let discount_A := 0.15 * orig_price_A
  let price_A := orig_price_A - discount_A
  let tax_A := 0.06 * price_A
  let total_A := price_A + tax_A
  let orig_price_B := 600.0
  let discount_B := 0.25 * orig_price_B
  let price_B := orig_price_B - discount_B
  let tax_B := 0.09 * price_B
  let total_B := price_B + tax_B
  let total_other_toys := total_A + total_B
  let price_lightsaber := 2 * total_other_toys
  let tax_lightsaber := 0.04 * price_lightsaber
  let total_lightsaber := price_lightsaber + tax_lightsaber
  total_other_toys + total_lightsaber

theorem john_total_spent : calculate_total_spent = 4008.312 := by
  sorry

end NUMINAMATH_GPT_john_total_spent_l2310_231011


namespace NUMINAMATH_GPT_unique_three_digit_numbers_l2310_231064

noncomputable def three_digit_numbers_no_repeats : Nat :=
  let total_digits := 10
  let permutations := total_digits * (total_digits - 1) * (total_digits - 2)
  let invalid_start_with_zero := (total_digits - 1) * (total_digits - 2)
  permutations - invalid_start_with_zero

theorem unique_three_digit_numbers : three_digit_numbers_no_repeats = 648 := by
  sorry

end NUMINAMATH_GPT_unique_three_digit_numbers_l2310_231064


namespace NUMINAMATH_GPT_gcd_1337_382_l2310_231080

theorem gcd_1337_382 : Nat.gcd 1337 382 = 191 := by
  sorry

end NUMINAMATH_GPT_gcd_1337_382_l2310_231080


namespace NUMINAMATH_GPT_remainder_proof_l2310_231024

def nums : List ℕ := [83, 84, 85, 86, 87, 88, 89, 90]
def mod : ℕ := 17

theorem remainder_proof : (nums.sum % mod) = 3 := by sorry

end NUMINAMATH_GPT_remainder_proof_l2310_231024


namespace NUMINAMATH_GPT_min_shots_for_probability_at_least_075_l2310_231084

theorem min_shots_for_probability_at_least_075 (hit_rate : ℝ) (target_probability : ℝ) :
  hit_rate = 0.25 → target_probability = 0.75 → ∃ n : ℕ, n = 4 ∧ (1 - hit_rate)^n ≤ 1 - target_probability := by
  intros h_hit_rate h_target_probability
  sorry

end NUMINAMATH_GPT_min_shots_for_probability_at_least_075_l2310_231084


namespace NUMINAMATH_GPT_min_value_16_l2310_231056

noncomputable def min_value_expr (a b : ℝ) : ℝ :=
  1 / a + 3 / b

theorem min_value_16 (a b : ℝ) (h : a > 0 ∧ b > 0) (h_constraint : a + 3 * b = 1) :
  min_value_expr a b ≥ 16 :=
sorry

end NUMINAMATH_GPT_min_value_16_l2310_231056


namespace NUMINAMATH_GPT_value_in_box_l2310_231065

theorem value_in_box (x : ℤ) (h : 5 + x = 10 + 20) : x = 25 := by
  sorry

end NUMINAMATH_GPT_value_in_box_l2310_231065


namespace NUMINAMATH_GPT_art_piece_increase_is_correct_l2310_231022

-- Define the conditions
def initial_price : ℝ := 4000
def future_multiplier : ℝ := 3
def future_price : ℝ := future_multiplier * initial_price

-- Define the goal
-- Proof that the increase in price is equal to $8000
theorem art_piece_increase_is_correct : future_price - initial_price = 8000 := 
by {
  -- We put sorry here to skip the actual proof
  sorry
}

end NUMINAMATH_GPT_art_piece_increase_is_correct_l2310_231022


namespace NUMINAMATH_GPT_find_y_value_l2310_231039

/-- Given angles and conditions, find the value of y in the geometric figure. -/
theorem find_y_value
  (AB_parallel_DC : true) -- AB is parallel to DC
  (ACE_straight_line : true) -- ACE is a straight line
  (angle_ACF : ℝ := 130) -- ∠ACF = 130°
  (angle_CBA : ℝ := 60) -- ∠CBA = 60°
  (angle_ACB : ℝ := 100) -- ∠ACB = 100°
  (angle_ADC : ℝ := 125) -- ∠ADC = 125°
  : 35 = 35 := -- y = 35°
by
  sorry

end NUMINAMATH_GPT_find_y_value_l2310_231039


namespace NUMINAMATH_GPT_smallest_k_for_repeating_representation_l2310_231067

theorem smallest_k_for_repeating_representation:
  ∃ k : ℕ, (k > 0) ∧ (∀ m : ℕ, m > 0 → m < k → ¬(97*(5*m + 6) = 11*(m^2 - 1))) ∧ 97*(5*k + 6) = 11*(k^2 - 1) := by
  sorry

end NUMINAMATH_GPT_smallest_k_for_repeating_representation_l2310_231067


namespace NUMINAMATH_GPT_find_abc_unique_solution_l2310_231068

theorem find_abc_unique_solution (N a b c : ℕ) 
  (hN : N > 3 ∧ N % 2 = 1)
  (h_eq : a^N = b^N + 2^N + a * b * c)
  (h_c : c ≤ 5 * 2^(N-1)) : 
  N = 5 ∧ a = 3 ∧ b = 1 ∧ c = 70 := 
sorry

end NUMINAMATH_GPT_find_abc_unique_solution_l2310_231068


namespace NUMINAMATH_GPT_eq_of_divisible_l2310_231040

theorem eq_of_divisible (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h : a + b ∣ 5 * a + 3 * b) : a = b :=
sorry

end NUMINAMATH_GPT_eq_of_divisible_l2310_231040


namespace NUMINAMATH_GPT_expression_value_l2310_231019

theorem expression_value (a b c d m : ℝ) 
  (h1 : a = -b) 
  (h2 : c * d = 1) 
  (h3 : |m| = 3) :
  (a + b) / m - c * d + m = 2 ∨ (a + b) / m - c * d + m = -4 := 
by
  sorry

end NUMINAMATH_GPT_expression_value_l2310_231019


namespace NUMINAMATH_GPT_machines_produce_12x_boxes_in_expected_time_l2310_231028

-- Definitions corresponding to the conditions
def rate_A (x : ℕ) := x / 10
def rate_B (x : ℕ) := 2 * x / 5
def rate_C (x : ℕ) := 3 * x / 8
def rate_D (x : ℕ) := x / 4

-- Total combined rate when working together
def combined_rate (x : ℕ) := rate_A x + rate_B x + rate_C x + rate_D x

-- The time taken to produce 12x boxes given their combined rate
def time_to_produce (x : ℕ) : ℕ := 12 * x / combined_rate x

-- Goal: Time taken should be 32/3 minutes
theorem machines_produce_12x_boxes_in_expected_time (x : ℕ) : time_to_produce x = 32 / 3 :=
sorry

end NUMINAMATH_GPT_machines_produce_12x_boxes_in_expected_time_l2310_231028


namespace NUMINAMATH_GPT_similarity_coordinates_l2310_231043

theorem similarity_coordinates {B B1 : ℝ × ℝ} 
  (h₁ : ∃ (k : ℝ), k = 2 ∧ 
         (∀ (x y : ℝ), B = (x, y) → ∀ (x₁ y₁ : ℝ), B1 = (x₁, y₁) → x₁ = x / k ∨ x₁ = x / -k) ∧ 
         (∀ (x y : ℝ), B = (x, y) → ∀ (x₁ y₁ : ℝ), B1 = (x₁, y₁) → y₁ = y / k ∨ y₁ = y / -k))
  (h₂ : B = (-4, -2)) :
  B1 = (-2, -1) ∨ B1 = (2, 1) :=
sorry

end NUMINAMATH_GPT_similarity_coordinates_l2310_231043


namespace NUMINAMATH_GPT_problem_inequality_l2310_231046

variable {a b c : ℝ}

-- Assuming a, b, c are positive real numbers
variables (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)

-- Assuming abc = 1
variable (h_abc : a * b * c = 1)

theorem problem_inequality :
  (1 / (a^3 * (b + c))) + (1 / (b^3 * (c + a))) + (1 / (c^3 * (a + b))) ≥ (3 / 2) :=
by sorry

end NUMINAMATH_GPT_problem_inequality_l2310_231046


namespace NUMINAMATH_GPT_solve_for_y_l2310_231006

theorem solve_for_y (y : ℚ) : 
  y + 1 / 3 = 3 / 8 - 1 / 4 → y = -5 / 24 := 
by
  sorry

end NUMINAMATH_GPT_solve_for_y_l2310_231006


namespace NUMINAMATH_GPT_violet_ticket_cost_l2310_231058

theorem violet_ticket_cost :
  (2 * 35 + 5 * 20 = 170) ∧
  (((35 - 17.50) + 35 + 5 * 20) = 152.50) ∧
  ((152.50 - 150) = 2.50) :=
by
  sorry

end NUMINAMATH_GPT_violet_ticket_cost_l2310_231058


namespace NUMINAMATH_GPT_linear_function_implies_m_value_l2310_231089

variable (x m : ℝ)

theorem linear_function_implies_m_value :
  (∃ y : ℝ, y = (m-3)*x^(m^2-8) + m + 1 ∧ ∀ x1 x2 : ℝ, y = y * (x2 - x1) + y * x1) → m = -3 :=
by
  sorry

end NUMINAMATH_GPT_linear_function_implies_m_value_l2310_231089


namespace NUMINAMATH_GPT_jacket_cost_l2310_231050

noncomputable def cost_of_shorts : ℝ := 13.99
noncomputable def cost_of_shirt : ℝ := 12.14
noncomputable def total_spent : ℝ := 33.56
noncomputable def cost_of_jacket : ℝ := total_spent - (cost_of_shorts + cost_of_shirt)

theorem jacket_cost : cost_of_jacket = 7.43 := by
  sorry

end NUMINAMATH_GPT_jacket_cost_l2310_231050


namespace NUMINAMATH_GPT_smallest_N_sum_of_digits_eq_six_l2310_231054

def bernardo_wins (N : ℕ) : Prop :=
  let b1 := 3 * N
  let s1 := b1 - 30
  let b2 := 3 * s1
  let s2 := b2 - 30
  let b3 := 3 * s2
  let s3 := b3 - 30
  let b4 := 3 * s3
  b4 < 800

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n
  else sum_of_digits (n / 10) + (n % 10)

theorem smallest_N_sum_of_digits_eq_six :
  ∃ N : ℕ, bernardo_wins N ∧ sum_of_digits N = 6 :=
by
  sorry

end NUMINAMATH_GPT_smallest_N_sum_of_digits_eq_six_l2310_231054


namespace NUMINAMATH_GPT_b_minus_a_equals_two_l2310_231066

open Set

variables {a b : ℝ}

theorem b_minus_a_equals_two (h₀ : {1, a + b, a} = ({0, b / a, b} : Finset ℝ)) (h₁ : a ≠ 0) : b - a = 2 :=
sorry

end NUMINAMATH_GPT_b_minus_a_equals_two_l2310_231066


namespace NUMINAMATH_GPT_solve_system_of_equations_l2310_231010

theorem solve_system_of_equations (x y z : ℝ) :
  x + y + z = 1 ∧ x^3 + y^3 + z^3 = 1 ∧ xyz = -16 ↔ 
  (x = 1 ∧ y = 4 ∧ z = -4) ∨ (x = 1 ∧ y = -4 ∧ z = 4) ∨ 
  (x = 4 ∧ y = 1 ∧ z = -4) ∨ (x = 4 ∧ y = -4 ∧ z = 1) ∨ 
  (x = -4 ∧ y = 1 ∧ z = 4) ∨ (x = -4 ∧ y = 4 ∧ z = 1) := 
by
  sorry

end NUMINAMATH_GPT_solve_system_of_equations_l2310_231010


namespace NUMINAMATH_GPT_new_oranges_added_l2310_231072

-- Defining the initial conditions
def initial_oranges : Nat := 40
def thrown_away_oranges : Nat := 37
def total_oranges_now : Nat := 10
def remaining_oranges : Nat := initial_oranges - thrown_away_oranges
def new_oranges := total_oranges_now - remaining_oranges

-- The theorem we want to prove
theorem new_oranges_added : new_oranges = 7 := by
  sorry

end NUMINAMATH_GPT_new_oranges_added_l2310_231072


namespace NUMINAMATH_GPT_sum_of_cubes_eq_neg_27_l2310_231036

theorem sum_of_cubes_eq_neg_27
  (a b c : ℝ)
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_condition : (a^3 + 9) / a = (b^3 + 9) / b ∧ (b^3 + 9) / b = (c^3 + 9) / c) :
  a^3 + b^3 + c^3 = -27 :=
sorry

end NUMINAMATH_GPT_sum_of_cubes_eq_neg_27_l2310_231036


namespace NUMINAMATH_GPT_pieces_info_at_most_two_identical_digits_l2310_231037

def num_pieces_of_information_with_at_most_two_positions_as_0110 : Nat :=
  (Nat.choose 4 2 + Nat.choose 4 1 + Nat.choose 4 0)

theorem pieces_info_at_most_two_identical_digits :
  num_pieces_of_information_with_at_most_two_positions_as_0110 = 11 :=
by
  sorry

end NUMINAMATH_GPT_pieces_info_at_most_two_identical_digits_l2310_231037


namespace NUMINAMATH_GPT_Owen_spending_on_burgers_in_June_l2310_231081

theorem Owen_spending_on_burgers_in_June (daily_burgers : ℕ) (cost_per_burger : ℕ) (days_in_June : ℕ) :
  daily_burgers = 2 → 
  cost_per_burger = 12 → 
  days_in_June = 30 → 
  daily_burgers * cost_per_burger * days_in_June = 720 :=
by
  intros
  sorry

end NUMINAMATH_GPT_Owen_spending_on_burgers_in_June_l2310_231081


namespace NUMINAMATH_GPT_suzanne_donation_l2310_231001

theorem suzanne_donation :
  let base_donation := 10
  let total_distance := 5
  let total_donation := (List.range total_distance).foldl (fun acc km => acc + base_donation * 2 ^ km) 0
  total_donation = 310 :=
by
  let base_donation := 10
  let total_distance := 5
  let total_donation := (List.range total_distance).foldl (fun acc km => acc + base_donation * 2 ^ km) 0
  sorry

end NUMINAMATH_GPT_suzanne_donation_l2310_231001


namespace NUMINAMATH_GPT_p_sufficient_not_necessary_for_q_l2310_231061

-- Define the propositions p and q
def is_ellipse (m : ℝ) : Prop := (1 / 4 < m) ∧ (m < 1)
def is_hyperbola (m : ℝ) : Prop := (0 < m) ∧ (m < 1)

-- Define the theorem to prove the relationship between p and q
theorem p_sufficient_not_necessary_for_q (m : ℝ) :
  (is_ellipse m → is_hyperbola m) ∧ ¬(is_hyperbola m → is_ellipse m) :=
sorry

end NUMINAMATH_GPT_p_sufficient_not_necessary_for_q_l2310_231061


namespace NUMINAMATH_GPT_last_place_is_Fedya_l2310_231012

def position_is_valid (position : ℕ) := position >= 1 ∧ position <= 4

variable (Misha Anton Petya Fedya : ℕ)

axiom Misha_statement: position_is_valid Misha → Misha ≠ 1 ∧ Misha ≠ 4
axiom Anton_statement: position_is_valid Anton → Anton ≠ 4
axiom Petya_statement: position_is_valid Petya → Petya = 1
axiom Fedya_statement: position_is_valid Fedya → Fedya = 4

theorem last_place_is_Fedya : ∃ (x : ℕ), x = Fedya ∧ Fedya = 4 :=
by
  sorry

end NUMINAMATH_GPT_last_place_is_Fedya_l2310_231012


namespace NUMINAMATH_GPT_pow_mod_remainder_l2310_231098

theorem pow_mod_remainder (x : ℕ) (h : x = 3) : x^1988 % 8 = 1 := by
  sorry

end NUMINAMATH_GPT_pow_mod_remainder_l2310_231098


namespace NUMINAMATH_GPT_vertex_of_parabola_is_correct_l2310_231020

theorem vertex_of_parabola_is_correct :
  ∀ x y : ℝ, y = -5 * (x + 2) ^ 2 - 6 → (x = -2 ∧ y = -6) :=
by
  sorry

end NUMINAMATH_GPT_vertex_of_parabola_is_correct_l2310_231020


namespace NUMINAMATH_GPT_solve_equation_l2310_231000

theorem solve_equation (y : ℝ) (z : ℝ) (hz : z = y^(1/3)) :
  (6 * y^(1/3) - 3 * y^(4/3) = 12 + y^(1/3) + y) ↔ (3 * z^4 + z^3 - 5 * z + 12 = 0) :=
by sorry

end NUMINAMATH_GPT_solve_equation_l2310_231000


namespace NUMINAMATH_GPT_option_C_correct_l2310_231023

theorem option_C_correct : 5 + (-6) - (-7) = 5 - 6 + 7 := 
by
  sorry

end NUMINAMATH_GPT_option_C_correct_l2310_231023


namespace NUMINAMATH_GPT_curve_equation_represents_line_l2310_231027

noncomputable def curve_is_line (x y : ℝ) : Prop :=
(x^2 + y^2 - 2*x) * (x + y - 3)^(1/2) = 0

theorem curve_equation_represents_line (x y : ℝ) :
curve_is_line x y ↔ (x + y = 3) :=
by sorry

end NUMINAMATH_GPT_curve_equation_represents_line_l2310_231027


namespace NUMINAMATH_GPT_system_solutions_l2310_231093

theorem system_solutions (x y z a b c : ℝ) :
  (a = 1 ∨ b = 1 ∨ c = 1 ∨ a + b + c + a * b * c = 0) → (¬(x = 1 ∨ y = 1 ∨ z = 1) → 
  ∃ (x y z : ℝ), (x - y) / (z - 1) = a ∧ (y - z) / (x - 1) = b ∧ (z - x) / (y - 1) = c) ∨
  (a ≠ 1 ∧ b ≠ 1 ∧ c ≠ 1 ∧ a + b + c + a * b * c ≠ 0) → 
  ¬∃ (x y z : ℝ), (x - y) / (z - 1) = a ∧ (y - z) / (x - 1) = b ∧ (z - x) / (y - 1) = c :=
by
    sorry

end NUMINAMATH_GPT_system_solutions_l2310_231093


namespace NUMINAMATH_GPT_proof_problem_l2310_231075

noncomputable def a : ℝ := (11 + Real.sqrt 337) ^ (1 / 3)
noncomputable def b : ℝ := (11 - Real.sqrt 337) ^ (1 / 3)
noncomputable def x : ℝ := a + b

theorem proof_problem : x^3 + 18 * x = 22 := by
  sorry

end NUMINAMATH_GPT_proof_problem_l2310_231075


namespace NUMINAMATH_GPT_mark_donates_cans_l2310_231091

-- Definitions coming directly from the conditions
def num_shelters : ℕ := 6
def people_per_shelter : ℕ := 30
def cans_per_person : ℕ := 10

-- The final statement to be proven
theorem mark_donates_cans : (num_shelters * people_per_shelter * cans_per_person) = 1800 :=
by sorry

end NUMINAMATH_GPT_mark_donates_cans_l2310_231091


namespace NUMINAMATH_GPT_smallest_int_ending_in_9_divisible_by_11_l2310_231041

theorem smallest_int_ending_in_9_divisible_by_11:
  ∃ x : ℕ, (∃ k : ℤ, x = 10 * k + 9) ∧ x % 11 = 0 ∧ x = 99 :=
by
  sorry

end NUMINAMATH_GPT_smallest_int_ending_in_9_divisible_by_11_l2310_231041


namespace NUMINAMATH_GPT_true_statement_l2310_231070

theorem true_statement :
  -8 < -2 := 
sorry

end NUMINAMATH_GPT_true_statement_l2310_231070


namespace NUMINAMATH_GPT_remainder_1235678_div_127_l2310_231055

theorem remainder_1235678_div_127 : 1235678 % 127 = 69 := by
  sorry

end NUMINAMATH_GPT_remainder_1235678_div_127_l2310_231055


namespace NUMINAMATH_GPT_car_speeds_l2310_231052

theorem car_speeds (u v w : ℝ) (hu : 0 < u) (hv : 0 < v) (hw : 0 < w) :
  (3 / (1 / u + 1 / v + 1 / w)) ≤ ((u + v) / 2) :=
sorry

end NUMINAMATH_GPT_car_speeds_l2310_231052


namespace NUMINAMATH_GPT_geometric_sequence_divisibility_l2310_231017

theorem geometric_sequence_divisibility 
  (a1 : ℚ) (h1 : a1 = 1 / 2) 
  (a2 : ℚ) (h2 : a2 = 10) 
  (n : ℕ) :
  ∃ (n : ℕ), a_n = (a1 * 20^(n - 1)) ∧ (n ≥ 4) ∧ (5000 ∣ a_n) :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_divisibility_l2310_231017


namespace NUMINAMATH_GPT_pages_read_in_7_days_l2310_231004

-- Definitions of the conditions
def total_hours : ℕ := 10
def days : ℕ := 5
def pages_per_hour : ℕ := 50
def reading_days : ℕ := 7

-- Compute intermediate steps
def hours_per_day : ℕ := total_hours / days
def pages_per_day : ℕ := pages_per_hour * hours_per_day

-- Lean statement to prove Tom reads 700 pages in 7 days
theorem pages_read_in_7_days :
  pages_per_day * reading_days = 700 :=
by
  -- We can add the intermediate steps here as sorry, as we will not do the proof
  sorry

end NUMINAMATH_GPT_pages_read_in_7_days_l2310_231004


namespace NUMINAMATH_GPT_christopher_age_l2310_231044

variable (C G F : ℕ)

theorem christopher_age (h1 : G = C + 8) (h2 : F = C - 2) (h3 : C + G + F = 60) : C = 18 := by
  sorry

end NUMINAMATH_GPT_christopher_age_l2310_231044
