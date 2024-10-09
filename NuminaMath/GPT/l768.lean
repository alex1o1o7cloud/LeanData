import Mathlib

namespace remainder_when_3_pow_305_div_13_l768_76809

theorem remainder_when_3_pow_305_div_13 :
  (3 ^ 305) % 13 = 9 := 
by {
  sorry
}

end remainder_when_3_pow_305_div_13_l768_76809


namespace monthly_revenue_l768_76803

variable (R : ℝ) -- The monthly revenue

-- Conditions
def after_taxes (R : ℝ) : ℝ := R * 0.90
def after_marketing (R : ℝ) : ℝ := (after_taxes R) * 0.95
def after_operational_costs (R : ℝ) : ℝ := (after_marketing R) * 0.80
def total_employee_wages (R : ℝ) : ℝ := (after_operational_costs R) * 0.15

-- Number of employees and their wages
def number_of_employees : ℝ := 10
def wage_per_employee : ℝ := 4104
def total_wages : ℝ := number_of_employees * wage_per_employee

-- Proof problem
theorem monthly_revenue : R = 400000 ↔ total_employee_wages R = total_wages := by
  sorry

end monthly_revenue_l768_76803


namespace part_1_part_2_l768_76834

open Set

variable (U : Set ℝ) (A B : Set ℝ)

def A_def : A = {x : ℝ | 0 < x ∧ x ≤ 2} := by
  ext x
  sorry
  
def B_def : B = {x : ℝ | x^2 + 2*x - 3 > 0} := by
  ext x
  sorry

theorem part_1 (hU : U = univ) (hA : A = {x : ℝ | 0 < x ∧ x ≤ 2}) (hB : B = {x : ℝ | x^2 + 2 * x - 3 > 0}) :
  compl (A ∪ B) = {x | -3 ≤ x ∧ x ≤ 0} := by
  rw [hA, hB]
  sorry

theorem part_2 (hU : U = univ) (hA : A = {x : ℝ | 0 < x ∧ x ≤ 2}) (hB : B = {x : ℝ | x^2 + 2 * x - 3 > 0}) :
  (compl A ∩ B) = {x | x > 1 ∨ x < -3} := by
  rw [hA, hB]
  sorry

end part_1_part_2_l768_76834


namespace patty_weighs_more_l768_76892

variable (R : ℝ) (P_0 : ℝ) (L : ℝ) (P : ℝ) (D : ℝ)

theorem patty_weighs_more :
  (R = 100) →
  (P_0 = 4.5 * R) →
  (L = 235) →
  (P = P_0 - L) →
  (D = P - R) →
  D = 115 := by
  sorry

end patty_weighs_more_l768_76892


namespace no_integer_solution_2_to_2x_minus_3_to_2y_eq_58_l768_76837

theorem no_integer_solution_2_to_2x_minus_3_to_2y_eq_58
  (x y : ℕ)
  (h1 : 2 ^ (2 * x) - 3 ^ (2 * y) = 58) : false :=
by
  sorry

end no_integer_solution_2_to_2x_minus_3_to_2y_eq_58_l768_76837


namespace probability_no_shaded_square_l768_76870

theorem probability_no_shaded_square :
  let num_rects := 1003 * 2005
  let num_rects_with_shaded := 1002^2
  let probability_no_shaded := 1 - (num_rects_with_shaded / num_rects)
  probability_no_shaded = 1 / 1003 := by
  -- The proof steps go here
  sorry

end probability_no_shaded_square_l768_76870


namespace Courtney_total_marbles_l768_76851

theorem Courtney_total_marbles (first_jar second_jar third_jar : ℕ) 
  (h1 : first_jar = 80)
  (h2 : second_jar = 2 * first_jar)
  (h3 : third_jar = first_jar / 4) :
  first_jar + second_jar + third_jar = 260 := 
by
  sorry

end Courtney_total_marbles_l768_76851


namespace pesticide_residue_comparison_l768_76819

noncomputable def f (x : ℝ) : ℝ := 1 / (1 + x^2)

theorem pesticide_residue_comparison (a : ℝ) (ha : a > 0) :
  (f a = (1 / (1 + a^2))) ∧ 
  (if a = 2 * Real.sqrt 2 then f a = 16 / (4 + a^2)^2 else 
   if a > 2 * Real.sqrt 2 then f a > 16 / (4 + a^2)^2 else 
   f a < 16 / (4 + a^2)^2) ∧
  (f 0 = 1) ∧ 
  (f 1 = 1 / 2) := sorry

end pesticide_residue_comparison_l768_76819


namespace total_roses_tom_sent_l768_76889

theorem total_roses_tom_sent
  (roses_in_dozen : ℕ := 12)
  (dozens_per_day : ℕ := 2)
  (days_in_week : ℕ := 7) :
  7 * (2 * 12) = 168 := by
  sorry

end total_roses_tom_sent_l768_76889


namespace dog_catches_rabbit_in_4_minutes_l768_76850

def dog_speed_mph : ℝ := 24
def rabbit_speed_mph : ℝ := 15
def rabbit_head_start : ℝ := 0.6

theorem dog_catches_rabbit_in_4_minutes : 
  (∃ t : ℝ, t > 0 ∧ 0.4 * t = 0.25 * t + 0.6) → ∃ t : ℝ, t = 4 :=
sorry

end dog_catches_rabbit_in_4_minutes_l768_76850


namespace find_initial_apples_l768_76875

theorem find_initial_apples (A : ℤ)
  (h1 : 6 * ((A / 8) + 8 - 30) = 12) :
  A = 192 :=
sorry

end find_initial_apples_l768_76875


namespace solve_trig_problem_l768_76865

theorem solve_trig_problem (α : ℝ) (h : Real.tan α = 1 / 3) :
  (Real.cos α)^2 - 2 * (Real.sin α)^2 / (Real.cos α)^2 = 7 / 9 := 
sorry

end solve_trig_problem_l768_76865


namespace teacher_age_l768_76894

theorem teacher_age (avg_age_students : ℕ) (num_students : ℕ) (new_avg_with_teacher : ℕ) (num_total : ℕ) 
  (total_age_students : ℕ)
  (h1 : avg_age_students = 10)
  (h2 : num_students = 15)
  (h3 : new_avg_with_teacher = 11)
  (h4 : num_total = 16)
  (h5 : total_age_students = num_students * avg_age_students) :
  num_total * new_avg_with_teacher - total_age_students = 26 :=
by sorry

end teacher_age_l768_76894


namespace algebraic_expression_value_l768_76844

noncomputable def a : ℝ := 2 * Real.sin (Real.pi / 4) + 1
noncomputable def b : ℝ := 2 * Real.cos (Real.pi / 4) - 1

theorem algebraic_expression_value :
  ((a^2 + b^2) / (2 * a * b) - 1) / ((a^2 - b^2) / (a^2 * b + a * b^2)) = 1 :=
by sorry

end algebraic_expression_value_l768_76844


namespace longest_segment_in_cylinder_l768_76881

theorem longest_segment_in_cylinder (r h : ℝ) (hr : r = 5) (hh : h = 12) :
  ∃ (d : ℝ), d = 2 * Real.sqrt 61 ∧ d = Real.sqrt (h^2 + (2*r)^2) :=
by
  sorry

end longest_segment_in_cylinder_l768_76881


namespace tyson_one_point_count_l768_76824

def tyson_three_points := 3 * 15
def tyson_two_points := 2 * 12
def total_points := 75
def points_from_three_and_two := tyson_three_points + tyson_two_points

theorem tyson_one_point_count :
  ∃ n : ℕ, n % 2 = 0 ∧ (n = total_points - points_from_three_and_two) :=
sorry

end tyson_one_point_count_l768_76824


namespace even_function_exists_l768_76863

def f (x m : ℝ) : ℝ := x^2 + m * x

theorem even_function_exists : ∃ m : ℝ, ∀ x : ℝ, f x m = f (-x) m :=
by
  use 0
  intros x
  unfold f
  simp

end even_function_exists_l768_76863


namespace distinct_constructions_l768_76887

def num_cube_constructions (white_cubes : Nat) (blue_cubes : Nat) : Nat :=
  if white_cubes = 5 ∧ blue_cubes = 3 then 5 else 0

theorem distinct_constructions : num_cube_constructions 5 3 = 5 :=
by
  sorry

end distinct_constructions_l768_76887


namespace necessary_and_sufficient_condition_l768_76871

def U (a : ℕ) : Set ℕ := { x | x > 0 ∧ x ≤ a }
def P : Set ℕ := {1, 2, 3}
def Q : Set ℕ := {4, 5, 6}
def C_U (S : Set ℕ) (a : ℕ) : Set ℕ := U a ∩ Sᶜ

theorem necessary_and_sufficient_condition (a : ℕ) (h : 6 ≤ a ∧ a < 7) : 
  C_U P a = Q ↔ (6 ≤ a ∧ a < 7) :=
by
  sorry

end necessary_and_sufficient_condition_l768_76871


namespace marble_draw_probability_l768_76890

theorem marble_draw_probability :
  let total_marbles := 12
  let red_marbles := 5
  let white_marbles := 4
  let blue_marbles := 3

  let p_red_first := (red_marbles / total_marbles : ℚ)
  let p_white_second := (white_marbles / (total_marbles - 1) : ℚ)
  let p_blue_third := (blue_marbles / (total_marbles - 2) : ℚ)
  
  p_red_first * p_white_second * p_blue_third = (1/22 : ℚ) :=
by
  sorry

end marble_draw_probability_l768_76890


namespace jessica_total_payment_l768_76860

-- Definitions based on the conditions
def basic_cable_cost : Nat := 15
def movie_channels_cost : Nat := 12
def sports_channels_cost : Nat := movie_channels_cost - 3

-- Definition of the total monthly payment given Jessica adds both movie and sports channels
def total_monthly_payment : Nat :=
  basic_cable_cost + (movie_channels_cost + sports_channels_cost)

-- The proof statement
theorem jessica_total_payment : total_monthly_payment = 36 :=
by
  -- skip the proof
  sorry

end jessica_total_payment_l768_76860


namespace expectation_variance_comparison_l768_76899

variable {p1 p2 : ℝ}
variable {ξ1 ξ2 : ℝ}

theorem expectation_variance_comparison
  (h_p1 : 0 < p1)
  (h_p2 : p1 < p2)
  (h_p3 : p2 < 1 / 2)
  (h_ξ1 : ξ1 = p1)
  (h_ξ2 : ξ2 = p2):
  (ξ1 < ξ2) ∧ (ξ1 * (1 - ξ1) < ξ2 * (1 - ξ2)) := by
  sorry

end expectation_variance_comparison_l768_76899


namespace kids_still_awake_l768_76845

-- Definition of the conditions
def num_kids_initial : ℕ := 20

def kids_asleep_first_5_minutes : ℕ := num_kids_initial / 2

def kids_awake_after_first_5_minutes : ℕ := num_kids_initial - kids_asleep_first_5_minutes

def kids_asleep_next_5_minutes : ℕ := kids_awake_after_first_5_minutes / 2

def kids_awake_final : ℕ := kids_awake_after_first_5_minutes - kids_asleep_next_5_minutes

-- Theorem that needs to be proved
theorem kids_still_awake : kids_awake_final = 5 := by
  sorry

end kids_still_awake_l768_76845


namespace ab_ac_bc_all_real_l768_76800

theorem ab_ac_bc_all_real (a b c : ℝ) (h : a + b + c = 1) : ∃ x : ℝ, ab + ac + bc = x := by
  sorry

end ab_ac_bc_all_real_l768_76800


namespace pieces_by_first_team_correct_l768_76818

-- Define the number of pieces required.
def total_pieces : ℕ := 500

-- Define the number of pieces made by the second team.
def pieces_by_second_team : ℕ := 131

-- Define the number of pieces made by the third team.
def pieces_by_third_team : ℕ := 180

-- Define the number of pieces made by the first team.
def pieces_by_first_team : ℕ := total_pieces - (pieces_by_second_team + pieces_by_third_team)

-- Statement to prove
theorem pieces_by_first_team_correct : pieces_by_first_team = 189 := 
by 
  -- Proof to be filled in
  sorry

end pieces_by_first_team_correct_l768_76818


namespace range_of_a_l768_76884

noncomputable def f (a x : ℝ) : ℝ := x^3 + x^2 - a * x - 4
noncomputable def f_derivative (a x : ℝ) : ℝ := 3 * x^2 + 2 * x - a

def has_exactly_one_extremum_in_interval (a : ℝ) : Prop :=
  (f_derivative a (-1)) * (f_derivative a 1) < 0

theorem range_of_a (a : ℝ) :
  has_exactly_one_extremum_in_interval a ↔ (1 < a ∧ a < 5) :=
sorry

end range_of_a_l768_76884


namespace probability_exactly_two_heads_and_two_tails_l768_76883

noncomputable def probability_two_heads_two_tails (n : ℕ) (k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k : ℚ) * (p ^ n)

theorem probability_exactly_two_heads_and_two_tails
  (tosses : ℕ) (k : ℕ) (p : ℚ) (h_tosses : tosses = 4) (h_k : k = 2) (h_p : p = 1/2) :
  probability_two_heads_two_tails tosses k p = 3 / 8 := by
  sorry

end probability_exactly_two_heads_and_two_tails_l768_76883


namespace total_spent_amount_l768_76812

-- Define the conditions
def spent_relation (B D : ℝ) : Prop := D = 0.75 * B
def payment_difference (B D : ℝ) : Prop := B = D + 12.50

-- Define the theorem to prove
theorem total_spent_amount (B D : ℝ) 
  (h1 : spent_relation B D) 
  (h2 : payment_difference B D) : 
  B + D = 87.50 :=
sorry

end total_spent_amount_l768_76812


namespace probability_A_to_B_in_8_moves_l768_76804

-- Define vertices
inductive Vertex : Type
| A | B | C | D | E | F

open Vertex

-- Define the probability of ending up at Vertex B after 8 moves starting from Vertex A
noncomputable def probability_at_B_after_8_moves : ℚ :=
  let prob := (3 : ℚ) / 16
  prob

-- Theorem statement
theorem probability_A_to_B_in_8_moves :
  (probability_at_B_after_8_moves = (3 : ℚ) / 16) :=
by
  -- Proof to be provided
  sorry

end probability_A_to_B_in_8_moves_l768_76804


namespace football_defense_stats_l768_76826

/-- Given:
1. Team 1 has an average of 1.5 goals conceded per match.
2. Team 1 has a standard deviation of 1.1 for the total number of goals conceded throughout the year.
3. Team 2 has an average of 2.1 goals conceded per match.
4. Team 2 has a standard deviation of 0.4 for the total number of goals conceded throughout the year.

Prove:
There are exactly 3 correct statements out of the 4 listed statements. -/
theorem football_defense_stats
  (avg_goals_team1 : ℝ := 1.5)
  (std_dev_team1 : ℝ := 1.1)
  (avg_goals_team2 : ℝ := 2.1)
  (std_dev_team2 : ℝ := 0.4) :
  ∃ correct_statements : ℕ, correct_statements = 3 := 
by
  sorry

end football_defense_stats_l768_76826


namespace solve_for_y_l768_76808

theorem solve_for_y (x y : ℝ) (h1 : x * y = 1) (h2 : x / y = 36) (h3 : 0 < x) (h4 : 0 < y) : 
  y = 1 / 6 := 
sorry

end solve_for_y_l768_76808


namespace fabric_sales_fraction_l768_76816

def total_sales := 36
def stationery_sales := 15
def jewelry_sales := total_sales / 4
def fabric_sales := total_sales - jewelry_sales - stationery_sales

theorem fabric_sales_fraction:
  (fabric_sales : ℝ) / total_sales = 1 / 3 :=
by
  sorry

end fabric_sales_fraction_l768_76816


namespace smallest_prime_perimeter_l768_76801

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_scalene_triangle (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a + b > c ∧ a + c > b ∧ b + c > a

def is_prime_perimeter_scalene_triangle (a b c : ℕ) : Prop :=
  is_prime a ∧ is_prime b ∧ is_prime c ∧
  is_scalene_triangle a b c ∧ is_prime (a + b + c)

theorem smallest_prime_perimeter (a b c : ℕ) :
  (a = 5 ∧ a < b ∧ a < c ∧ is_prime_perimeter_scalene_triangle a b c) →
  (a + b + c = 23) :=
by
  sorry

end smallest_prime_perimeter_l768_76801


namespace vinces_bus_ride_length_l768_76898

theorem vinces_bus_ride_length (zachary_ride : ℝ) (vince_extra : ℝ) (vince_ride : ℝ) :
  zachary_ride = 0.5 →
  vince_extra = 0.13 →
  vince_ride = zachary_ride + vince_extra →
  vince_ride = 0.63 :=
by
  intros hz hv he
  -- proof steps here
  sorry

end vinces_bus_ride_length_l768_76898


namespace ratio_difference_l768_76869

theorem ratio_difference (x : ℕ) (h : 7 * x = 70) : 70 - 3 * x = 40 :=
by
  -- proof would go here
  sorry

end ratio_difference_l768_76869


namespace find_x_l768_76861

open Real

theorem find_x (x : ℝ) (h : (x / 6) / 3 = 6 / (x / 3)) : x = 18 ∨ x = -18 :=
by
  sorry

end find_x_l768_76861


namespace particular_solutions_of_diff_eq_l768_76828

variable {x y : ℝ}

theorem particular_solutions_of_diff_eq
  (h₁ : ∀ C : ℝ, x^2 = C * (y - C))
  (h₂ : x > 0) :
  (y = 2 * x ∨ y = -2 * x) ↔ (x * (y')^2 - 2 * y * y' + 4 * x = 0) := 
sorry

end particular_solutions_of_diff_eq_l768_76828


namespace min_distance_between_lines_t_l768_76859

theorem min_distance_between_lines_t (t : ℝ) :
  (∀ x y : ℝ, x + 2 * y + t^2 = 0) ∧ (∀ x y : ℝ, 2 * x + 4 * y + 2 * t - 3 = 0) →
  t = 1 / 2 := by
  sorry

end min_distance_between_lines_t_l768_76859


namespace fixed_monthly_fee_l768_76848

-- Define the problem parameters and assumptions
variables (x y : ℝ)
axiom february_bill : x + y = 20.72
axiom march_bill : x + 3 * y = 35.28

-- State the Lean theorem that we want to prove
theorem fixed_monthly_fee : x = 13.44 :=
by
  sorry

end fixed_monthly_fee_l768_76848


namespace pencils_given_l768_76854

-- Define the conditions
def a : Nat := 9
def b : Nat := 65

-- Define the goal statement: the number of pencils Kathryn gave to Anthony
theorem pencils_given (a b : Nat) (h₁ : a = 9) (h₂ : b = 65) : b - a = 56 :=
by
  -- Omitted proof part
  sorry

end pencils_given_l768_76854


namespace second_store_earns_at_least_72000_more_l768_76831

-- Conditions as definitions in Lean.
def discount_price := 900000 -- 10% discount on 1 million yuan.
def full_price := 1000000 -- Full price for 1 million yuan without discount.

-- Prize calculation for the second department store.
def prize_first := 1000 * 5
def prize_second := 500 * 10
def prize_third := 200 * 20
def prize_fourth := 100 * 40
def prize_fifth := 10 * 1000

def total_prizes := prize_first + prize_second + prize_third + prize_fourth + prize_fifth

def second_store_net_income := full_price - total_prizes -- Net income after subtracting prizes.

-- The proof problem statement.
theorem second_store_earns_at_least_72000_more :
  second_store_net_income - discount_price >= 72000 := sorry

end second_store_earns_at_least_72000_more_l768_76831


namespace length_third_altitude_l768_76885

theorem length_third_altitude (a b c : ℝ) (S : ℝ) 
  (h_altitude_a : 4 = 2 * S / a)
  (h_altitude_b : 12 = 2 * S / b)
  (h_scalene : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h_third_integer : ∃ n : ℕ, h = n):
  h = 5 :=
by
  -- Proof is omitted
  sorry

end length_third_altitude_l768_76885


namespace ceil_y_squared_possibilities_l768_76895

theorem ceil_y_squared_possibilities (y : ℝ) (h : ⌈y⌉ = 15) : 
  ∃ n : ℕ, (n = 29) ∧ (∀ z : ℕ, ⌈y^2⌉ = z → (197 ≤ z ∧ z ≤ 225)) :=
by
  sorry

end ceil_y_squared_possibilities_l768_76895


namespace moles_NaClO4_formed_l768_76822

-- Condition: Balanced chemical reaction
def reaction : Prop := ∀ (NaOH HClO4 NaClO4 H2O : ℕ), NaOH + HClO4 = NaClO4 + H2O

-- Given: 3 moles of NaOH and 3 moles of HClO4
def initial_moles_NaOH : ℕ := 3
def initial_moles_HClO4 : ℕ := 3

-- Question: number of moles of NaClO4 formed
def final_moles_NaClO4 : ℕ := 3

-- Proof Problem: Given the balanced chemical reaction and initial moles, prove the final moles of NaClO4
theorem moles_NaClO4_formed : reaction → initial_moles_NaOH = 3 → initial_moles_HClO4 = 3 → final_moles_NaClO4 = 3 :=
by
  intros
  sorry

end moles_NaClO4_formed_l768_76822


namespace blue_pill_cost_l768_76814

theorem blue_pill_cost (y : ℕ) :
  -- Conditions
  (∀ t d : ℕ, t = 21 → 
     d = 14 → 
     (735 - d * 2 = t * ((2 * y) + (y + 2)) / t) →
     2 * y + (y + 2) = 35) →
  -- Conclusion
  y = 11 :=
by
  sorry

end blue_pill_cost_l768_76814


namespace sufficient_but_not_necessary_l768_76838

theorem sufficient_but_not_necessary (a : ℝ) (h : a > 1) : (1 / a < 1) := 
by
  sorry

end sufficient_but_not_necessary_l768_76838


namespace simplest_form_l768_76842

theorem simplest_form (b : ℝ) (h : b ≠ 2) : 2 - (2 / (2 + b / (2 - b))) = 4 / (4 - b) :=
by sorry

end simplest_form_l768_76842


namespace store_profit_l768_76876

theorem store_profit {C : ℝ} (h₁ : C > 0) : 
  let SP1 := 1.20 * C
  let SP2 := 1.25 * SP1
  let SPF := 0.80 * SP2
  SPF - C = 0.20 * C := 
by 
  let SP1 := 1.20 * C
  let SP2 := 1.25 * SP1
  let SPF := 0.80 * SP2
  sorry

end store_profit_l768_76876


namespace common_measure_of_segments_l768_76893

theorem common_measure_of_segments (a b : ℚ) (h₁ : a = 4 / 15) (h₂ : b = 8 / 21) : 
  (∃ (c : ℚ), c = 1 / 105 ∧ ∃ (n₁ n₂ : ℕ), a = n₁ * c ∧ b = n₂ * c) := 
by {
  sorry
}

end common_measure_of_segments_l768_76893


namespace abs_expression_eq_6500_l768_76802

def given_expression (x : ℝ) : ℝ := 
  abs (abs x - x - abs x + 500) - x

theorem abs_expression_eq_6500 (x : ℝ) (h : x = -3000) : given_expression x = 6500 := by
  sorry

end abs_expression_eq_6500_l768_76802


namespace half_abs_diff_of_squares_l768_76877

theorem half_abs_diff_of_squares (a b : ℤ) (h1 : a = 21) (h2 : b = 19) :
  (abs (a^2 - b^2)) / 2 = 40 := by
  sorry

end half_abs_diff_of_squares_l768_76877


namespace number_of_people_in_group_l768_76805

theorem number_of_people_in_group (P : ℕ) : 
  (∃ (P : ℕ), 0 < P ∧ (364 / P - 1 = 364 / (P + 2))) → P = 26 :=
by
  sorry

end number_of_people_in_group_l768_76805


namespace probability_window_opens_correct_l768_76835

noncomputable def probability_window_opens_no_later_than_3_minutes_after_scientist_arrives 
  (arrival_times : Fin 6 → ℝ) : ℝ :=
  if (∀ i, arrival_times i ∈ Set.Icc 0 15) ∧ 
     (∀ i j, i ≠ j → arrival_times i < arrival_times j) ∧ 
     ((∃ i, arrival_times i ≥ 12)) then
    1 - (0.8 ^ 6)
  else
    0

theorem probability_window_opens_correct : 
  ∀ (arrival_times : Fin 6 → ℝ),
    (∀ i, arrival_times i ∈ Set.Icc 0 15) →
    (∀ i j, i ≠ j → arrival_times i < arrival_times j) →
    (∃ i, arrival_times i = arrival_times 5) →
    abs (probability_window_opens_no_later_than_3_minutes_after_scientist_arrives arrival_times - 0.738) < 0.001 :=
by
  sorry

end probability_window_opens_correct_l768_76835


namespace sin_tan_relation_l768_76833

theorem sin_tan_relation (θ : ℝ) (h : Real.tan θ = 2) : 
  Real.sin θ * Real.sin (3 * Real.pi / 2 + θ) = -(2 / 5) := 
sorry

end sin_tan_relation_l768_76833


namespace ny_mets_fans_count_l768_76896

theorem ny_mets_fans_count (Y M R : ℕ) (h1 : 3 * M = 2 * Y) (h2 : 4 * R = 5 * M) (h3 : Y + M + R = 390) : M = 104 := 
by
  sorry

end ny_mets_fans_count_l768_76896


namespace pizza_slices_left_l768_76880

def initial_slices : ℕ := 16
def eaten_during_dinner : ℕ := initial_slices / 4
def remaining_after_dinner : ℕ := initial_slices - eaten_during_dinner
def yves_eaten : ℕ := remaining_after_dinner / 4
def remaining_after_yves : ℕ := remaining_after_dinner - yves_eaten
def siblings_eaten : ℕ := 2 * 2
def remaining_after_siblings : ℕ := remaining_after_yves - siblings_eaten

theorem pizza_slices_left : remaining_after_siblings = 5 := by
  sorry

end pizza_slices_left_l768_76880


namespace mathematicians_correctness_l768_76815

theorem mathematicians_correctness :
  (2 / 5 + 3 / 8) / (5 + 8) = 5 / 13 ∧
  (4 / 10 + 3 / 8) / (10 + 8) = 7 / 18 ∧
  (3 / 8 < 2 / 5 ∧ 2 / 5 < 17 / 40) → false :=
by 
  sorry

end mathematicians_correctness_l768_76815


namespace trigonometric_identity_l768_76867

theorem trigonometric_identity
    (α φ : ℝ) :
    4 * Real.cos α * Real.cos φ * Real.cos (α - φ) - 2 * (Real.cos (α - φ))^2 - Real.cos (2 * φ) = Real.cos (2 * α) :=
by
  sorry

end trigonometric_identity_l768_76867


namespace max_students_distribution_l768_76813

theorem max_students_distribution (pens toys : ℕ) (h_pens : pens = 451) (h_toys : toys = 410) :
  Nat.gcd pens toys = 41 :=
by
  sorry

end max_students_distribution_l768_76813


namespace factor_expression_correct_l768_76864

variable (y : ℝ)

def expression := 4 * y * (y + 2) + 6 * (y + 2)

theorem factor_expression_correct : expression y = (y + 2) * (2 * (2 * y + 3)) :=
by
  sorry

end factor_expression_correct_l768_76864


namespace find_all_waldo_time_l768_76857

theorem find_all_waldo_time (b : ℕ) (p : ℕ) (t : ℕ) :
  b = 15 → p = 30 → t = 3 → b * p * t = 1350 := by
sorry

end find_all_waldo_time_l768_76857


namespace count_selection_4_balls_count_selection_5_balls_score_at_least_7_points_l768_76823

-- Setup the basic context
def Pocket := Finset (Fin 11)

-- The pocket contains 4 red balls and 7 white balls
def red_balls : Finset (Fin 11) := {0, 1, 2, 3}
def white_balls : Finset (Fin 11) := {4, 5, 6, 7, 8, 9, 10}

-- Question 1
theorem count_selection_4_balls :
  (red_balls.card.choose 4) + (red_balls.card.choose 3 * white_balls.card.choose 1) +
  (red_balls.card.choose 2 * white_balls.card.choose 2) = 115 := 
sorry

-- Question 2
theorem count_selection_5_balls_score_at_least_7_points :
  (red_balls.card.choose 2 * white_balls.card.choose 3) +
  (red_balls.card.choose 3 * white_balls.card.choose 2) +
  (red_balls.card.choose 4 * white_balls.card.choose 1) = 301 := 
sorry

end count_selection_4_balls_count_selection_5_balls_score_at_least_7_points_l768_76823


namespace mike_total_hours_l768_76825

-- Define the number of hours Mike worked each day.
def hours_per_day : ℕ := 3

-- Define the number of days Mike worked.
def days : ℕ := 5

-- Define the total number of hours Mike worked.
def total_hours : ℕ := hours_per_day * days

-- State and prove that the total hours Mike worked is 15.
theorem mike_total_hours : total_hours = 15 := by
  -- Proof goes here
  sorry

end mike_total_hours_l768_76825


namespace Billy_weight_is_159_l768_76862

def Carl_weight : ℕ := 145
def Brad_weight : ℕ := Carl_weight + 5
def Billy_weight : ℕ := Brad_weight + 9

theorem Billy_weight_is_159 : Billy_weight = 159 := by
  sorry

end Billy_weight_is_159_l768_76862


namespace molecular_weight_calculation_l768_76836

theorem molecular_weight_calculation :
  let atomic_weight_K := 39.10
  let atomic_weight_Br := 79.90
  let atomic_weight_O := 16.00
  let num_K := 1
  let num_Br := 1
  let num_O := 3
  let molecular_weight := (num_K * atomic_weight_K) + (num_Br * atomic_weight_Br) + (num_O * atomic_weight_O)
  molecular_weight = 167.00 :=
by
  sorry

end molecular_weight_calculation_l768_76836


namespace time_to_paint_remaining_rooms_l768_76811

-- Definitions for the conditions
def total_rooms : ℕ := 11
def time_per_room : ℕ := 7
def painted_rooms : ℕ := 2

-- Statement of the problem
theorem time_to_paint_remaining_rooms : 
  total_rooms - painted_rooms = 9 →
  (total_rooms - painted_rooms) * time_per_room = 63 := 
by 
  intros h1
  sorry

end time_to_paint_remaining_rooms_l768_76811


namespace cos_of_angle_between_lines_l768_76868

noncomputable def cosTheta (a b : ℝ × ℝ) : ℝ :=
  let dotProduct := a.1 * b.1 + a.2 * b.2
  let magA := Real.sqrt (a.1 ^ 2 + a.2 ^ 2)
  let magB := Real.sqrt (b.1 ^ 2 + b.2 ^ 2)
  dotProduct / (magA * magB)

theorem cos_of_angle_between_lines :
  cosTheta (3, 4) (1, 3) = 3 / Real.sqrt 10 :=
by
  sorry

end cos_of_angle_between_lines_l768_76868


namespace cat_finishes_food_on_next_monday_l768_76855

noncomputable def cat_food_consumption_per_day : ℚ := (1 / 4) + (1 / 6)

theorem cat_finishes_food_on_next_monday :
  ∃ n : ℕ, n = 8 ∧ (n * cat_food_consumption_per_day > 8) := sorry

end cat_finishes_food_on_next_monday_l768_76855


namespace find_c_value_l768_76810

theorem find_c_value (A B C : ℝ) (S1_area S2_area : ℝ) (b : ℝ) :
  S1_area = 40 * b + 1 →
  S2_area = 40 * b →
  ∃ c, AC + CB = c ∧ c = 462 :=
by
  intro hS1 hS2
  sorry

end find_c_value_l768_76810


namespace functional_solution_l768_76852

def functional_property (f : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ), f (x * f y + 2 * x) = x * y + 2 * f x

theorem functional_solution (f : ℝ → ℝ) (h : functional_property f) : f 1 = 0 :=
by sorry

end functional_solution_l768_76852


namespace vertex_of_parabola_l768_76827

theorem vertex_of_parabola :
  ∃ (x y : ℝ), (∀ x : ℝ, y = x^2 - 12 * x + 9) → (x, y) = (6, -27) :=
sorry

end vertex_of_parabola_l768_76827


namespace total_number_of_matches_l768_76841

-- Define the total number of teams
def numberOfTeams : ℕ := 10

-- Define the number of matches each team competes against each other team
def matchesPerPair : ℕ := 4

-- Calculate the total number of unique matches
def calculateUniqueMatches (teams : ℕ) : ℕ :=
  (teams * (teams - 1)) / 2

-- Main statement to be proved
theorem total_number_of_matches : calculateUniqueMatches numberOfTeams * matchesPerPair = 180 := by
  -- Placeholder for the proof
  sorry

end total_number_of_matches_l768_76841


namespace square_root_ratio_area_l768_76840

theorem square_root_ratio_area (side_length_C side_length_D : ℕ) (hC : side_length_C = 45) (hD : side_length_D = 60) : 
  Real.sqrt ((side_length_C^2 : ℝ) / (side_length_D^2 : ℝ)) = 3 / 4 :=
by
  rw [hC, hD]
  sorry

end square_root_ratio_area_l768_76840


namespace arrival_time_l768_76878

def minutes_to_hours (minutes : ℕ) : ℕ := minutes / 60

theorem arrival_time (departure_time : ℕ) (stop1 stop2 stop3 travel_hours : ℕ) (stops_total_time := stop1 + stop2 + stop3) (stops_total_hours := minutes_to_hours stops_total_time) : 
  departure_time = 7 → 
  stop1 = 25 → 
  stop2 = 10 → 
  stop3 = 25 → 
  travel_hours = 12 → 
  (departure_time + (travel_hours - stops_total_hours)) % 24 = 18 :=
by
  sorry

end arrival_time_l768_76878


namespace no_real_quadruples_solutions_l768_76897

theorem no_real_quadruples_solutions :
  ¬ ∃ (a b c d : ℝ),
    a^3 + c^3 = 2 ∧
    a^2 * b + c^2 * d = 0 ∧
    b^3 + d^3 = 1 ∧
    a * b^2 + c * d^2 = -6 := 
sorry

end no_real_quadruples_solutions_l768_76897


namespace prove_equation_l768_76843

theorem prove_equation (x : ℚ) (h : 5 * x - 3 = 15 * x + 21) : 3 * (2 * x + 5) = 3 / 5 :=
by
  sorry

end prove_equation_l768_76843


namespace find_n_l768_76829

theorem find_n :
  ∃ (n : ℤ), 50 ≤ n ∧ n ≤ 120 ∧ (n % 8 = 0) ∧ (n % 7 = 5) ∧ (n % 6 = 3) ∧ n = 208 := 
by {
  sorry
}

end find_n_l768_76829


namespace distinct_real_numbers_satisfying_system_l768_76849

theorem distinct_real_numbers_satisfying_system :
  ∃! (x y z : ℝ),
  x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
  (x^2 + y^2 = -x + 3 * y + z) ∧
  (y^2 + z^2 = x + 3 * y - z) ∧
  (x^2 + z^2 = 2 * x + 2 * y - z) ∧
  ((x = 0 ∧ y = 1 ∧ z = -2) ∨ (x = -3/2 ∧ y = 5/2 ∧ z = -1/2)) :=
sorry

end distinct_real_numbers_satisfying_system_l768_76849


namespace max_AMC_AM_MC_CA_l768_76856

theorem max_AMC_AM_MC_CA (A M C : ℕ) (h_sum : A + M + C = 15) :
  A * M * C + A * M + M * C + C * A ≤ 200 :=
sorry

end max_AMC_AM_MC_CA_l768_76856


namespace inequality_problem_l768_76821

theorem inequality_problem (a b c d : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d)
    (h_sum : a + b + c + d = 4) : 
    a^2 * b * c + b^2 * d * a + c^2 * d * a + d^2 * b * c ≤ 4 := 
sorry

end inequality_problem_l768_76821


namespace f_log₂_20_l768_76879

noncomputable def f (x : ℝ) : ℝ := sorry -- This is a placeholder for the function f.

lemma f_neg (x : ℝ) : f (-x) = -f (x) := sorry
lemma f_shift (x : ℝ) : f (x + 1) = f (1 - x) := sorry
lemma f_special (x : ℝ) (hx : -1 < x ∧ x < 0) : f (x) = 2^x + 6 / 5 := sorry

theorem f_log₂_20 : f (Real.log 20 / Real.log 2) = -2 := by
  -- Proof details would go here.
  sorry

end f_log₂_20_l768_76879


namespace number_of_friends_l768_76891

def money_emma : ℕ := 8

def money_daya : ℕ := money_emma + (money_emma * 25 / 100)

def money_jeff : ℕ := (2 * money_daya) / 5

def money_brenda : ℕ := money_jeff + 4

def money_brenda_condition : Prop := money_brenda = 8

def friends_pooling_pizza : ℕ := 4

theorem number_of_friends (h : money_brenda_condition) : friends_pooling_pizza = 4 := by
  sorry

end number_of_friends_l768_76891


namespace club_membership_l768_76806

theorem club_membership:
  (∃ (committee : ℕ → Prop) (member_assign : (ℕ × ℕ) → ℕ → Prop),
    (∀ i, i < 5 → ∃! m, member_assign (i, m) 2) ∧
    (∀ i j, i < 5 ∧ j < 5 ∧ i ≠ j → ∃! m, m < 10 ∧ member_assign (i, j) m)
  ) → 
  ∃ n, n = 10 :=
by
  sorry

end club_membership_l768_76806


namespace sequence_perfect_square_l768_76847

variable (a : ℕ → ℤ)

axiom a1 : a 1 = 1
axiom a2 : a 2 = 1
axiom recurrence : ∀ n ≥ 3, a n = 7 * (a (n - 1)) - (a (n - 2))

theorem sequence_perfect_square (n : ℕ) (hn : n > 0) : ∃ k : ℤ, a n + a (n + 1) + 2 = k * k :=
by
  sorry

end sequence_perfect_square_l768_76847


namespace finance_specialization_percentage_l768_76873

theorem finance_specialization_percentage (F : ℝ) :
  (76 - 43.333333333333336) = (90 - F) → 
  F = 57.333333333333336 :=
by
  sorry

end finance_specialization_percentage_l768_76873


namespace second_dog_average_miles_l768_76888

theorem second_dog_average_miles
  (total_miles_week : ℕ)
  (first_dog_miles_day : ℕ)
  (days_in_week : ℕ)
  (h1 : total_miles_week = 70)
  (h2 : first_dog_miles_day = 2)
  (h3 : days_in_week = 7) :
  (total_miles_week - (first_dog_miles_day * days_in_week)) / days_in_week = 8 := sorry

end second_dog_average_miles_l768_76888


namespace TimTotalRunHoursPerWeek_l768_76820

def TimUsedToRunTimesPerWeek : ℕ := 3
def TimAddedExtraDaysPerWeek : ℕ := 2
def MorningRunHours : ℕ := 1
def EveningRunHours : ℕ := 1

theorem TimTotalRunHoursPerWeek :
  (TimUsedToRunTimesPerWeek + TimAddedExtraDaysPerWeek) * (MorningRunHours + EveningRunHours) = 10 :=
by
  sorry

end TimTotalRunHoursPerWeek_l768_76820


namespace binom_10_4_l768_76839

theorem binom_10_4 : Nat.choose 10 4 = 210 := 
by sorry

end binom_10_4_l768_76839


namespace moles_H2O_formed_l768_76886

-- Define the balanced equation as a struct
structure Reaction :=
(reactants : List (String × ℕ)) -- List of reactants with their stoichiometric coefficients
(products : List (String × ℕ)) -- List of products with their stoichiometric coefficients

-- Example reaction: NaHCO3 + HC2H3O2 -> NaC2H3O2 + H2O + CO2
def example_reaction : Reaction :=
{ reactants := [("NaHCO3", 1), ("HC2H3O2", 1)],
  products := [("NaC2H3O2", 1), ("H2O", 1), ("CO2", 1)] }

-- We need a predicate to determine the number of moles of a product based on the reaction
def moles_of_product (reaction : Reaction) (product : String) (moles_reactant₁ moles_reactant₂ : ℕ) : ℕ :=
if product = "H2O" then moles_reactant₁ else 0  -- Only considering H2O for simplicity

-- Now we define our main theorem
theorem moles_H2O_formed : 
  moles_of_product example_reaction "H2O" 3 3 = 3 :=
by
  -- The proof will go here; for now, we use sorry to skip it
  sorry

end moles_H2O_formed_l768_76886


namespace find_fourth_term_l768_76866

variable (a_n : ℕ → ℕ)
variable (S_n : ℕ → ℕ)
variable (a_1 a_4 d : ℕ)

-- Conditions
axiom sum_first_5 : S_n 5 = 35
axiom sum_first_9 : S_n 9 = 117
axiom sum_closed_form_first_5 : 5 * a_1 + (5 * (5 - 1)) / 2 * d = 35
axiom sum_closed_form_first_9 : 9 * a_1 + (9 * (9 - 1)) / 2 * d = 117
axiom nth_term_closed_form : ∀ n, a_n n = a_1 + (n-1)*d

-- Target
theorem find_fourth_term : a_4 = 10 := by
  sorry

end find_fourth_term_l768_76866


namespace trains_crossing_time_l768_76807

theorem trains_crossing_time :
  let length_first_train := 500
  let length_second_train := 800
  let speed_first_train := 80 * (5/18 : ℚ)  -- convert km/hr to m/s
  let speed_second_train := 100 * (5/18 : ℚ)  -- convert km/hr to m/s
  let relative_speed := speed_first_train + speed_second_train
  let total_distance := length_first_train + length_second_train
  let time_taken := total_distance / relative_speed
  time_taken = 26 :=
by
  sorry

end trains_crossing_time_l768_76807


namespace monotonic_increase_interval_l768_76817

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem monotonic_increase_interval : ∀ x : ℝ, 0 < x ∧ x < Real.exp 1 → 0 < (Real.log x) / x :=
by sorry

end monotonic_increase_interval_l768_76817


namespace log_a_plus_b_eq_zero_l768_76830

open Complex

noncomputable def a_b_expression : ℂ := (⟨2, 1⟩ / ⟨1, 1⟩ : ℂ)

noncomputable def a : ℝ := a_b_expression.re

noncomputable def b : ℝ := a_b_expression.im

theorem log_a_plus_b_eq_zero : log (a + b) = 0 := by
  sorry

end log_a_plus_b_eq_zero_l768_76830


namespace spinner_sections_equal_size_l768_76874

theorem spinner_sections_equal_size 
  (p : ℕ → Prop)
  (h1 : ∀ n, p n ↔ (1 - (1: ℝ) / n) ^ 2 = 0.5625) : 
  p 4 :=
by
  sorry

end spinner_sections_equal_size_l768_76874


namespace arithmetic_sequence_a101_eq_52_l768_76858

theorem arithmetic_sequence_a101_eq_52 (a : ℕ → ℝ)
  (h₁ : a 1 = 2)
  (h₂ : ∀ n : ℕ, a (n + 1) - a n = 1 / 2) :
  a 101 = 52 :=
by
  sorry

end arithmetic_sequence_a101_eq_52_l768_76858


namespace digit_2567_l768_76872

def nth_digit_in_concatenation (n : ℕ) : ℕ :=
  sorry

theorem digit_2567 : nth_digit_in_concatenation 2567 = 8 :=
by
  sorry

end digit_2567_l768_76872


namespace geom_seq_min_value_l768_76846

theorem geom_seq_min_value (r : ℝ) : 
  (1 : ℝ) = a_1 → a_2 = r → a_3 = r^2 → ∃ r : ℝ, 6 * a_2 + 7 * a_3 = -9/7 := 
by 
  intros h1 h2 h3 
  use -3/7 
  rw [h2, h3] 
  ring 
  sorry

end geom_seq_min_value_l768_76846


namespace max_value_a7_b7_c7_d7_l768_76853

-- Assume a, b, c, d are real numbers such that a^6 + b^6 + c^6 + d^6 = 64
-- Prove that the maximum value of a^7 + b^7 + c^7 + d^7 is 128
theorem max_value_a7_b7_c7_d7 (a b c d : ℝ) (h : a^6 + b^6 + c^6 + d^6 = 64) : 
  ∃ a b c d, a^6 + b^6 + c^6 + d^6 = 64 ∧ a^7 + b^7 + c^7 + d^7 = 128 :=
by sorry

end max_value_a7_b7_c7_d7_l768_76853


namespace cos_sq_sub_sin_sq_l768_76882

noncomputable def cos_sq_sub_sin_sq_eq := 
  ∀ (α : ℝ), α ∈ Set.Ioo 0 Real.pi → (Real.sin α + Real.cos α = Real.sqrt 3 / 3) →
  (Real.cos α) ^ 2 - (Real.sin α) ^ 2 = -Real.sqrt 5 / 3

theorem cos_sq_sub_sin_sq :
  cos_sq_sub_sin_sq_eq := 
by
  intros α hα h_eq
  sorry

end cos_sq_sub_sin_sq_l768_76882


namespace extreme_point_a_zero_l768_76832

noncomputable def f (a x : ℝ) : ℝ := a * x^3 + x^2 - (a + 2) * x + 1
def f_prime (a x : ℝ) : ℝ := 3 * a * x^2 + 2 * x - (a + 2)

theorem extreme_point_a_zero (a : ℝ) (h : f_prime a 1 = 0) : a = 0 :=
by
  sorry

end extreme_point_a_zero_l768_76832
