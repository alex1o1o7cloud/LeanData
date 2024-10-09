import Mathlib

namespace part_a_l1183_118322

-- Power tower with 100 twos
def power_tower_100_t2 : ℕ := sorry

theorem part_a : power_tower_100_t2 > 3 := sorry

end part_a_l1183_118322


namespace part_1_part_2_equality_case_l1183_118306

variables {m n : ℝ}

-- Definition of positive real numbers and given condition m > n and n > 1
def conditions_1 (m n : ℝ) : Prop := m > 0 ∧ n > 0 ∧ m > n ∧ n > 1

-- Prove that given conditions, m^2 + n > mn + m
theorem part_1 (m n : ℝ) (h : conditions_1 m n) : m^2 + n > m * n + m :=
  by sorry

-- Definition of the condition m + 2n = 1
def conditions_2 (m n : ℝ) : Prop := m > 0 ∧ n > 0 ∧ m + 2 * n = 1

-- Prove that given conditions, (2/m) + (1/n) ≥ 8
theorem part_2 (m n : ℝ) (h : conditions_2 m n) : (2 / m) + (1 / n) ≥ 8 :=
  by sorry

-- Prove that the minimum value is obtained when m = 2n = 1/2
theorem equality_case (m n : ℝ) (h : conditions_2 m n) : 
  (2 / m) + (1 / n) = 8 ↔ m = 1/2 ∧ n = 1/4 :=
  by sorry

end part_1_part_2_equality_case_l1183_118306


namespace quadratic_complete_square_l1183_118307

theorem quadratic_complete_square : ∀ x : ℝ, (x^2 - 8*x - 1) = (x - 4)^2 - 17 :=
by sorry

end quadratic_complete_square_l1183_118307


namespace min_expression_value_l1183_118395

theorem min_expression_value (x y z : ℝ) : ∃ x y z : ℝ, (xy - z)^2 + (x + y + z)^2 = 0 :=
by
  sorry

end min_expression_value_l1183_118395


namespace initial_cakes_count_l1183_118315

theorem initial_cakes_count (f : ℕ) (a b : ℕ) 
  (condition1 : f = 5)
  (condition2 : ∀ i, i ∈ Finset.range f → a = 4)
  (condition3 : ∀ i, i ∈ Finset.range f → b = 20 / 2)
  (condition4 : f * a = 2 * b) : 
  b = 40 := 
by
  sorry

end initial_cakes_count_l1183_118315


namespace reconstruct_quadrilateral_l1183_118386

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (A A'' B'' C'' D'' : V)

def trisect_segment (P Q R : V) : Prop :=
  Q = (1 / 3 : ℝ) • P + (2 / 3 : ℝ) • R

theorem reconstruct_quadrilateral
  (hB : trisect_segment A B A'')
  (hC : trisect_segment B C B'')
  (hD : trisect_segment C D C'')
  (hA : trisect_segment D A D'') :
  A = (2 / 26) • A'' + (6 / 26) • B'' + (6 / 26) • C'' + (12 / 26) • D'' :=
sorry

end reconstruct_quadrilateral_l1183_118386


namespace Alec_goal_ratio_l1183_118320

theorem Alec_goal_ratio (total_students half_votes thinking_votes more_needed fifth_votes : ℕ)
  (h_class : total_students = 60)
  (h_half : half_votes = total_students / 2)
  (remaining_students : ℕ := total_students - half_votes)
  (h_thinking : thinking_votes = 5)
  (h_fifth : fifth_votes = (remaining_students - thinking_votes) / 5)
  (h_current_votes : half_votes + thinking_votes + fifth_votes = 40)
  (h_needed : more_needed = 5)
  :
  (half_votes + thinking_votes + fifth_votes + more_needed) / total_students = 3 / 4 :=
by sorry

end Alec_goal_ratio_l1183_118320


namespace trace_bags_weight_l1183_118348

theorem trace_bags_weight :
  ∀ (g1 g2 t1 t2 t3 t4 t5 : ℕ),
    g1 = 3 →
    g2 = 7 →
    (g1 + g2) = (t1 + t2 + t3 + t4 + t5) →
    (t1 = t2 ∧ t2 = t3 ∧ t3 = t4 ∧ t4 = t5) →
    t1 = 2 :=
by
  intros g1 g2 t1 t2 t3 t4 t5 hg1 hg2 hsum hsame
  sorry

end trace_bags_weight_l1183_118348


namespace terminal_angle_quadrant_l1183_118334

theorem terminal_angle_quadrant : 
  let angle := -558
  let reduced_angle := angle % 360
  90 < reduced_angle ∧ reduced_angle < 180 →
  SecondQuadrant := 
by 
  intro angle reduced_angle h 
  sorry

end terminal_angle_quadrant_l1183_118334


namespace number_of_boys_l1183_118379

theorem number_of_boys (n : ℕ)
    (incorrect_avg_weight : ℝ)
    (misread_weight new_weight : ℝ)
    (correct_avg_weight : ℝ)
    (h1 : incorrect_avg_weight = 58.4)
    (h2 : misread_weight = 56)
    (h3 : new_weight = 66)
    (h4 : correct_avg_weight = 58.9)
    (h5 : n * correct_avg_weight = n * incorrect_avg_weight + (new_weight - misread_weight)) :
  n = 20 := by
  sorry

end number_of_boys_l1183_118379


namespace length_of_arc_l1183_118378

def radius : ℝ := 5
def area_of_sector : ℝ := 10
def expected_length_of_arc : ℝ := 4

theorem length_of_arc (r : ℝ) (A : ℝ) (l : ℝ) (h₁ : r = radius) (h₂ : A = area_of_sector) : l = expected_length_of_arc := by
  sorry

end length_of_arc_l1183_118378


namespace maximize_profit_l1183_118313

variable {k : ℝ} (hk : k > 0)
variable {x : ℝ} (hx : 0 < x ∧ x < 0.06)

def deposit_volume (x : ℝ) : ℝ := k * x
def interest_paid (x : ℝ) : ℝ := k * x ^ 2
def profit (x : ℝ) : ℝ := (0.06 * k^2 * x) - (k * x^2)

theorem maximize_profit : 0.03 = x :=
by
  sorry

end maximize_profit_l1183_118313


namespace calculateTotalProfit_l1183_118399

-- Defining the initial investments and changes
def initialInvestmentA : ℕ := 5000
def initialInvestmentB : ℕ := 8000
def initialInvestmentC : ℕ := 9000

def additionalInvestmentA : ℕ := 2000
def withdrawnInvestmentB : ℕ := 1000
def additionalInvestmentC : ℕ := 3000

-- Defining the durations
def months1 : ℕ := 4
def months2 : ℕ := 8
def months3 : ℕ := 6

-- C's share of the profit
def shareOfC : ℕ := 45000

-- Total profit to be proved
def totalProfit : ℕ := 103571

-- Lean 4 theorem statement
theorem calculateTotalProfit :
  let ratioA := (initialInvestmentA * months1) + ((initialInvestmentA + additionalInvestmentA) * months2)
  let ratioB := (initialInvestmentB * months1) + ((initialInvestmentB - withdrawnInvestmentB) * months2)
  let ratioC := (initialInvestmentC * months3) + ((initialInvestmentC + additionalInvestmentC) * months3)
  let totalRatio := ratioA + ratioB + ratioC
  (shareOfC / ratioC : ℚ) = (totalProfit / totalRatio : ℚ) :=
sorry

end calculateTotalProfit_l1183_118399


namespace average_score_girls_proof_l1183_118338

noncomputable def average_score_girls_all_schools (A a B b C c : ℕ)
  (adams_boys : ℕ) (adams_girls : ℕ) (adams_comb : ℕ)
  (baker_boys : ℕ) (baker_girls : ℕ) (baker_comb : ℕ)
  (carter_boys : ℕ) (carter_girls : ℕ) (carter_comb : ℕ)
  (all_boys_comb : ℕ) : ℕ :=
  -- Assume number of boys and girls per school A, B, C (boys) and a, b, c (girls)
  if (adams_boys * A + adams_girls * a) / (A + a) = adams_comb ∧
     (baker_boys * B + baker_girls * b) / (B + b) = baker_comb ∧
     (carter_boys * C + carter_girls * c) / (C + c) = carter_comb ∧
     (adams_boys * A + baker_boys * B + carter_boys * C) / (A + B + C) = all_boys_comb
  then (85 * a + 92 * b + 80 * c) / (a + b + c) else 0

theorem average_score_girls_proof (A a B b C c : ℕ)
  (adams_boys : ℕ := 82) (adams_girls : ℕ := 85) (adams_comb : ℕ := 83)
  (baker_boys : ℕ := 87) (baker_girls : ℕ := 92) (baker_comb : ℕ := 91)
  (carter_boys : ℕ := 78) (carter_girls : ℕ := 80) (carter_comb : ℕ := 80)
  (all_boys_comb : ℕ := 84) :
  average_score_girls_all_schools A a B b C c adams_boys adams_girls adams_comb baker_boys baker_girls baker_comb carter_boys carter_girls carter_comb all_boys_comb = 85 :=
by
  sorry

end average_score_girls_proof_l1183_118338


namespace car_gas_cost_l1183_118302

def car_mpg_city : ℝ := 30
def car_mpg_highway : ℝ := 40
def city_distance_one_way : ℝ := 60
def highway_distance_one_way : ℝ := 200
def gas_cost_per_gallon : ℝ := 3
def total_gas_cost : ℝ := 42

theorem car_gas_cost :
  (city_distance_one_way / car_mpg_city * 2 + highway_distance_one_way / car_mpg_highway * 2) * gas_cost_per_gallon = total_gas_cost := 
  sorry

end car_gas_cost_l1183_118302


namespace algebraic_expression_is_product_l1183_118361

def algebraicExpressionMeaning (x : ℝ) : Prop :=
  -7 * x = -7 * x

theorem algebraic_expression_is_product (x : ℝ) :
  algebraicExpressionMeaning x :=
by
  sorry

end algebraic_expression_is_product_l1183_118361


namespace unique_int_pair_exists_l1183_118327

theorem unique_int_pair_exists (a b : ℤ) : 
  ∃! (x y : ℤ), (x + 2 * y - a)^2 + (2 * x - y - b)^2 ≤ 1 :=
by
  sorry

end unique_int_pair_exists_l1183_118327


namespace perpendicular_vectors_l1183_118376

theorem perpendicular_vectors (k : ℝ) (a b : ℝ × ℝ) 
  (ha : a = (0, 2)) 
  (hb : b = (Real.sqrt 3, 1)) 
  (h : (a.1 - k * b.1) * (k * a.1 + b.1) + (a.2 - k * b.2) * (k * a.2 + b.2) = 0) :
  k = -1 ∨ k = 1 :=
sorry

end perpendicular_vectors_l1183_118376


namespace speed_of_woman_in_still_water_l1183_118344

noncomputable def V_w : ℝ := 5
variable (V_s : ℝ)

-- Conditions:
def downstream_condition : Prop := (V_w + V_s) * 6 = 54
def upstream_condition : Prop := (V_w - V_s) * 6 = 6

theorem speed_of_woman_in_still_water 
    (h1 : downstream_condition V_s) 
    (h2 : upstream_condition V_s) : 
    V_w = 5 :=
by
    -- Proof omitted
    sorry

end speed_of_woman_in_still_water_l1183_118344


namespace range_of_m_for_distinct_real_roots_l1183_118329

theorem range_of_m_for_distinct_real_roots (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2 * x₁ + m = 0 ∧ x₂^2 + 2 * x₂ + m = 0) ↔ m < 1 :=
by sorry

end range_of_m_for_distinct_real_roots_l1183_118329


namespace solution_is_three_l1183_118352

def equation (x : ℝ) : Prop := 
  Real.sqrt (4 - 3 * Real.sqrt (10 - 3 * x)) = x - 2

theorem solution_is_three : equation 3 :=
by sorry

end solution_is_three_l1183_118352


namespace k_value_l1183_118336

theorem k_value (k m : ℤ) (h : (m - 8) ∣ (m^2 - k * m - 24)) : k = 5 := by
  have : (m - 8) ∣ (m^2 - 8 * m - 24) := sorry
  sorry

end k_value_l1183_118336


namespace geometric_sequence_min_l1183_118377

theorem geometric_sequence_min (a : ℕ → ℝ) (q : ℝ)
  (h_positive : ∀ n, a n > 0)
  (h_condition : 2 * (a 4) + (a 3) - 2 * (a 2) - (a 1) = 8)
  (h_geometric : ∀ n, a (n+1) = a n * q) :
  ∃ min_val, min_val = 12 * Real.sqrt 3 ∧ min_val = 2 * (a 5) + (a 4) :=
sorry

end geometric_sequence_min_l1183_118377


namespace jessica_monthly_car_insurance_payment_l1183_118300

theorem jessica_monthly_car_insurance_payment
  (rent_last_year : ℤ := 1000)
  (food_last_year : ℤ := 200)
  (car_insurance_last_year : ℤ)
  (rent_increase_rate : ℕ := 3 / 10)
  (food_increase_rate : ℕ := 1 / 2)
  (car_insurance_increase_rate : ℕ := 3)
  (additional_expenses_this_year : ℤ := 7200) :
  car_insurance_last_year = 300 :=
by
  sorry

end jessica_monthly_car_insurance_payment_l1183_118300


namespace scientific_notation_113700_l1183_118388

theorem scientific_notation_113700 :
  ∃ (a : ℝ) (b : ℤ), 113700 = a * 10 ^ b ∧ a = 1.137 ∧ b = 5 :=
by
  sorry

end scientific_notation_113700_l1183_118388


namespace mike_practice_hours_l1183_118394

def weekday_practice_hours_per_day : ℕ := 3
def days_per_weekday_practice : ℕ := 5
def saturday_practice_hours : ℕ := 5
def weeks_until_game : ℕ := 3

def total_weekday_practice_hours : ℕ := weekday_practice_hours_per_day * days_per_weekday_practice
def total_weekly_practice_hours : ℕ := total_weekday_practice_hours + saturday_practice_hours
def total_practice_hours : ℕ := total_weekly_practice_hours * weeks_until_game

theorem mike_practice_hours :
  total_practice_hours = 60 := by
  sorry

end mike_practice_hours_l1183_118394


namespace nth_term_arithmetic_seq_l1183_118373

theorem nth_term_arithmetic_seq (a : ℕ → ℝ) (d : ℝ)
  (h_arithmetic : ∀ n : ℕ, ∃ m : ℝ, a (n + 1) = a n + m)
  (h_d_neg : d < 0)
  (h_condition1 : a 2 * a 4 = 12)
  (h_condition2 : a 2 + a 4 = 8):
  ∀ n : ℕ, a n = -2 * n + 10 :=
by
  sorry

end nth_term_arithmetic_seq_l1183_118373


namespace number_of_prize_orders_l1183_118355

/-- At the end of a professional bowling tournament, the top 6 bowlers have a playoff.
    - #6 and #5 play a game. The loser receives the 6th prize and the winner plays #4.
    - The loser of the second game receives the 5th prize and the winner plays #3.
    - The loser of the third game receives the 4th prize and the winner plays #2.
    - The loser of the fourth game receives the 3rd prize and the winner plays #1.
    - The winner of the final game gets 1st prize and the loser gets 2nd prize.

    We want to determine the number of possible orders in which the bowlers can receive the prizes.
-/
theorem number_of_prize_orders : 2^5 = 32 := by
  sorry

end number_of_prize_orders_l1183_118355


namespace linda_age_difference_l1183_118384

/-- 
Linda is some more than 2 times the age of Jane.
In five years, the sum of their ages will be 28.
Linda's age at present is 13.
Prove that Linda's age is 3 years more than 2 times Jane's age.
-/
theorem linda_age_difference {L J : ℕ} (h1 : L = 13)
  (h2 : (L + 5) + (J + 5) = 28) : L - 2 * J = 3 :=
by sorry

end linda_age_difference_l1183_118384


namespace factorization_correct_l1183_118326

noncomputable def factorize_diff_of_squares (a b : ℝ) : ℝ :=
  36 * a * a - 4 * b * b

theorem factorization_correct (a b : ℝ) : factorize_diff_of_squares a b = 4 * (3 * a + b) * (3 * a - b) :=
by
  sorry

end factorization_correct_l1183_118326


namespace max_height_l1183_118303

-- Given definitions
def height_eq (t : ℝ) : ℝ := -16 * t^2 + 64 * t + 10

def max_height_problem : Prop :=
  ∃ t : ℝ, height_eq t = 74 ∧ ∀ t' : ℝ, height_eq t' ≤ height_eq t

-- Statement of the proof
theorem max_height : max_height_problem := sorry

end max_height_l1183_118303


namespace sufficient_but_not_necessary_condition_for_parallelism_l1183_118312

-- Define the two lines
def line1 (x y : ℝ) (m : ℝ) : Prop := 2 * x - m * y = 1
def line2 (x y : ℝ) (m : ℝ) : Prop := (m - 1) * x - y = 1

-- Define the parallel condition for the two lines
def parallel (m : ℝ) : Prop :=
  (∃ x1 y1 x2 y2 : ℝ, line1 x1 y1 m ∧ line2 x2 y2 m ∧ (2 * m + 1 = 0 ∧ m^2 - m - 2 = 0)) ∨ 
  (∃ x1 y1 x2 y2 : ℝ, line1 x1 y1 2 ∧ line2 x2 y2 2)

theorem sufficient_but_not_necessary_condition_for_parallelism :
  ∀ m, (parallel m) ↔ (m = 2) :=
by sorry

end sufficient_but_not_necessary_condition_for_parallelism_l1183_118312


namespace expectedAdjacentBlackPairs_l1183_118365

noncomputable def numberOfBlackPairsInCircleDeck (totalCards blackCards redCards : ℕ) : ℚ := 
  let probBlackNext := (blackCards - 1) / (totalCards - 1)
  blackCards * probBlackNext

theorem expectedAdjacentBlackPairs (totalCards blackCards redCards expectedPairs : ℕ) : 
  totalCards = 52 → 
  blackCards = 30 → 
  redCards = 22 → 
  expectedPairs = 870 / 51 → 
  numberOfBlackPairsInCircleDeck totalCards blackCards redCards = expectedPairs :=
by 
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end expectedAdjacentBlackPairs_l1183_118365


namespace find_meeting_time_l1183_118398

-- Define the context and the problem parameters
def lisa_speed : ℝ := 9  -- Lisa's speed in mph
def adam_speed : ℝ := 7  -- Adam's speed in mph
def initial_distance : ℝ := 6  -- Initial distance in miles

-- The time in minutes for Lisa to meet Adam
theorem find_meeting_time : (initial_distance / (lisa_speed + adam_speed)) * 60 = 22.5 := by
  -- The proof is omitted for this statement
  sorry

end find_meeting_time_l1183_118398


namespace count_jianzhan_count_gift_boxes_l1183_118341

-- Definitions based on given conditions
def firewood_red_clay : Int := 90
def firewood_white_clay : Int := 60
def electric_red_clay : Int := 75
def electric_white_clay : Int := 75
def total_red_clay : Int := 1530
def total_white_clay : Int := 1170

-- Proof problem 1: Number of "firewood firing" and "electric firing" Jianzhan produced
theorem count_jianzhan (x y : Int) (hx : firewood_red_clay * x + electric_red_clay * y = total_red_clay)
  (hy : firewood_white_clay * x + electric_white_clay * y = total_white_clay) : 
  x = 12 ∧ y = 6 :=
sorry

-- Definitions based on given conditions for Part 2
def total_jianzhan : Int := 18
def box_a_capacity : Int := 2
def box_b_capacity : Int := 6

-- Proof problem 2: Number of purchasing plans for gift boxes
theorem count_gift_boxes (m n : Int) (h : box_a_capacity * m + box_b_capacity * n = total_jianzhan) : 
  ∃ s : Finset (Int × Int), s.card = 4 ∧ ∀ (p : Int × Int), p ∈ s ↔ (p = (9, 0) ∨ p = (6, 1) ∨ p = (3, 2) ∨ p = (0, 3)) :=
sorry

end count_jianzhan_count_gift_boxes_l1183_118341


namespace minimum_excellence_percentage_l1183_118335

theorem minimum_excellence_percentage (n : ℕ) (h : n = 100)
    (m c b : ℕ) 
    (h_math : m = 70)
    (h_chinese : c = 75) 
    (h_min_both : b = c - (n - m))
    (h_percent : b = 45) :
    b = 45 :=
    sorry

end minimum_excellence_percentage_l1183_118335


namespace polygon_sides_eq_seven_l1183_118390

theorem polygon_sides_eq_seven (n : ℕ) :
  ((n - 2) * 180 = 3 * 360 - 180) → n = 7 :=
by
  sorry

end polygon_sides_eq_seven_l1183_118390


namespace contradiction_method_at_most_one_positive_l1183_118369

theorem contradiction_method_at_most_one_positive :
  (∃ a b c : ℝ, (a > 0 → (b ≤ 0 ∧ c ≤ 0)) ∧ (b > 0 → (a ≤ 0 ∧ c ≤ 0)) ∧ (c > 0 → (a ≤ 0 ∧ b ≤ 0))) → 
  (¬(∃ a b c : ℝ, (a > 0 ∧ b > 0) ∨ (b > 0 ∧ c > 0) ∨ (a > 0 ∧ c > 0))) :=
by sorry

end contradiction_method_at_most_one_positive_l1183_118369


namespace hyperbola_through_point_has_asymptotes_l1183_118350

-- Definitions based on condition (1)
def hyperbola_asymptotes (x y : ℝ) : Prop := y = 2 * x ∨ y = -2 * x

-- Definition of the problem
def hyperbola_eqn (x y : ℝ) : Prop := (x^2 / 5) - (y^2 / 20) = 1

-- Main statement including all conditions and proving the correct answer
theorem hyperbola_through_point_has_asymptotes :
  ∀ x y : ℝ, hyperbola_eqn x y ↔ (hyperbola_asymptotes x y ∨ (x, y) = (-3, 4)) :=
by
  -- The proof part is skipped with sorry
  sorry

end hyperbola_through_point_has_asymptotes_l1183_118350


namespace same_cost_duration_l1183_118324

-- Define the cost function for Plan A
def cost_plan_a (x : ℕ) : ℚ :=
 if x ≤ 8 then 0.60 else 0.60 + 0.06 * (x - 8)

-- Define the cost function for Plan B
def cost_plan_b (x : ℕ) : ℚ :=
 0.08 * x

-- The duration of a call for which the company charges the same under Plan A and Plan B is 14 minutes
theorem same_cost_duration (x : ℕ) : cost_plan_a x = cost_plan_b x ↔ x = 14 :=
by
  -- The proof is not required, using sorry to skip the proof steps
  sorry

end same_cost_duration_l1183_118324


namespace functional_equation_solution_l1183_118347

noncomputable def f : ℝ → ℝ := sorry

theorem functional_equation_solution :
  (∀ x y : ℝ, f (f x * f y) + f (x + y) = f (x * y)) ↔ (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = x - 1) ∨ (∀ x : ℝ, f x = 1 - x) :=
sorry

end functional_equation_solution_l1183_118347


namespace hydrogen_moles_l1183_118353

-- Define the balanced chemical reaction as a relation between moles
def balanced_reaction (NaH H₂O NaOH H₂ : ℕ) : Prop :=
  NaH = NaOH ∧ H₂ = NaOH ∧ NaH = H₂

-- Given conditions
def given_conditions (NaH H₂O : ℕ) : Prop :=
  NaH = 2 ∧ H₂O = 2

-- Problem statement to prove
theorem hydrogen_moles (NaH H₂O NaOH H₂ : ℕ)
  (h₁ : balanced_reaction NaH H₂O NaOH H₂)
  (h₂ : given_conditions NaH H₂O) :
  H₂ = 2 :=
by sorry

end hydrogen_moles_l1183_118353


namespace total_amount_l1183_118318

noncomputable def initial_amounts (a j t : ℕ) := (t = 24)
noncomputable def redistribution_amounts (a j t a' j' t' : ℕ) :=
  a' = 3 * (2 * (a - 2 * j - 24)) ∧
  j' = 3 * (3 * j - (a - 2 * j - 24 + 48)) ∧
  t' = 144 - (6 * (a - 2 * j - 24) + 9 * j - 3 * (a - 2 * j - 24 + 48))

theorem total_amount (a j t a' j' t' : ℕ) (h1 : t = 24)
  (h2 : redistribution_amounts a j t a' j' t')
  (h3 : t' = 24) : 
  a + j + t = 72 :=
sorry

end total_amount_l1183_118318


namespace min_value_frac_f1_f_l1183_118340

theorem min_value_frac_f1_f'0 (a b c : ℝ) (h_a : a > 0) (h_b : b > 0) (h_c : c > 0) (h_discriminant : b^2 ≤ 4 * a * c) :
  (a + b + c) / b ≥ 2 := 
by
  -- Here goes the proof
  sorry

end min_value_frac_f1_f_l1183_118340


namespace four_digit_number_divisibility_l1183_118374

theorem four_digit_number_divisibility 
  (E V I L : ℕ) 
  (hE : 0 ≤ E ∧ E < 10) 
  (hV : 0 ≤ V ∧ V < 10) 
  (hI : 0 ≤ I ∧ I < 10) 
  (hL : 0 ≤ L ∧ L < 10)
  (h1 : (1000 * E + 100 * V + 10 * I + L) % 73 = 0) 
  (h2 : (1000 * V + 100 * I + 10 * L + E) % 74 = 0)
  : 1000 * L + 100 * I + 10 * V + E = 5499 := 
  sorry

end four_digit_number_divisibility_l1183_118374


namespace box_surface_area_l1183_118331

theorem box_surface_area (w l s tab : ℕ):
  w = 40 → l = 60 → s = 8 → tab = 2 →
  (40 * 60 - 4 * 8 * 8 + 2 * (2 * (60 - 2 * 8) + 2 * (40 - 2 * 8))) = 2416 :=
by
  intros _ _ _ _
  sorry

end box_surface_area_l1183_118331


namespace positive_integer_solution_lcm_eq_sum_l1183_118366

def is_lcm (x y z m : Nat) : Prop :=
  ∃ (d : Nat), x = d * (Nat.gcd y z) ∧ y = d * (Nat.gcd x z) ∧ z = d * (Nat.gcd x y) ∧
  x * y * z / Nat.gcd x (Nat.gcd y z) = m

theorem positive_integer_solution_lcm_eq_sum :
  ∀ (a b c : Nat), 0 < a → 0 < b → 0 < c → is_lcm a b c (a + b + c) → (a, b, c) = (a, 2 * a, 3 * a) := by
    sorry

end positive_integer_solution_lcm_eq_sum_l1183_118366


namespace bridge_length_is_correct_l1183_118319

-- Train length in meters
def train_length : ℕ := 130

-- Train speed in km/hr
def train_speed_kmh : ℕ := 45

-- Time to cross bridge in seconds
def time_to_cross_bridge : ℕ := 30

-- Conversion factor from km/hr to m/s
def kmh_to_mps (kmh : ℕ) : ℚ := (kmh * 1000) / 3600

-- Train speed in m/s
def train_speed_mps := kmh_to_mps train_speed_kmh

-- Total distance covered by the train in 30 seconds
def total_distance := train_speed_mps * time_to_cross_bridge

-- Length of the bridge
def bridge_length := total_distance - train_length

theorem bridge_length_is_correct : bridge_length = 245 := by
  sorry

end bridge_length_is_correct_l1183_118319


namespace min_abs_diff_is_11_l1183_118393

noncomputable def min_abs_diff (k l : ℕ) : ℤ := abs (36^k - 5^l)

theorem min_abs_diff_is_11 :
  ∃ k l : ℕ, min_abs_diff k l = 11 :=
by
  sorry

end min_abs_diff_is_11_l1183_118393


namespace labourer_income_l1183_118371

noncomputable def monthly_income : ℤ := 75

theorem labourer_income:
  ∃ (I D : ℤ),
  (80 * 6 = 480) ∧
  (I * 6 - D + (I * 4) = 480 + 240 + D + 30) →
  I = monthly_income :=
by
  sorry

end labourer_income_l1183_118371


namespace apple_production_total_l1183_118370

def apples_first_year := 40
def apples_second_year := 8 + 2 * apples_first_year
def apples_third_year := apples_second_year - (1 / 4) * apples_second_year
def total_apples := apples_first_year + apples_second_year + apples_third_year

-- Math proof problem statement
theorem apple_production_total : total_apples = 194 :=
  sorry

end apple_production_total_l1183_118370


namespace count_unique_lists_of_five_l1183_118333

theorem count_unique_lists_of_five :
  (∃ (f : ℕ → ℕ), ∀ (i j : ℕ), i < j → f (i + 1) - f i = 3 ∧ j = 5 → f 5 % f 1 = 0) →
  (∃ (n : ℕ), n = 6) :=
by
  sorry

end count_unique_lists_of_five_l1183_118333


namespace nickel_ate_3_chocolates_l1183_118316

theorem nickel_ate_3_chocolates (R N : ℕ) (h1 : R = 7) (h2 : R = N + 4) : N = 3 := by
  sorry

end nickel_ate_3_chocolates_l1183_118316


namespace ratio_when_volume_maximized_l1183_118308

-- Definitions based on conditions
def cylinder_perimeter := 24

-- Definition of properties derived from maximizing the volume
def max_volume_height := 4

def max_volume_circumference := 12 - max_volume_height

-- The ratio of the circumference of the cylinder's base to its height when the volume is maximized
def max_volume_ratio := max_volume_circumference / max_volume_height

-- The theorem to be proved
theorem ratio_when_volume_maximized :
  max_volume_ratio = 2 :=
by sorry

end ratio_when_volume_maximized_l1183_118308


namespace range_of_alpha_minus_beta_l1183_118345

theorem range_of_alpha_minus_beta (α β : Real) (h₁ : -180 < α) (h₂ : α < β) (h₃ : β < 180) :
  -360 < α - β ∧ α - β < 0 :=
by
  sorry

end range_of_alpha_minus_beta_l1183_118345


namespace point_M_in_second_quadrant_l1183_118346

-- Given conditions
def m : ℤ := -2
def n : ℤ := 1

-- Definitions to identify the quadrants
def point_in_second_quadrant (x y : ℤ) : Prop :=
  x < 0 ∧ y > 0

-- Problem statement to prove
theorem point_M_in_second_quadrant : 
  point_in_second_quadrant m n :=
by
  sorry

end point_M_in_second_quadrant_l1183_118346


namespace find_ratio_l1183_118354

theorem find_ratio (a b : ℝ) (h1 : ∀ x, ax^2 + bx + 2 < 0 ↔ (x < -1/2 ∨ x > 1/3)) :
  (a - b) / a = 5 / 6 := 
sorry

end find_ratio_l1183_118354


namespace solution_set_l1183_118368

variable {f : ℝ → ℝ}
variable {f' : ℝ → ℝ}

-- Given conditions
axiom differentiable_f : Differentiable ℝ f
axiom f_deriv : ∀ x, deriv f x = f' x
axiom f_at_3 : f 3 = 1
axiom inequality : ∀ x, 3 * f x + x * f' x > 1

-- Goal to prove
theorem solution_set :
  {x : ℝ | (x - 2017) ^ 3 * f (x - 2017) - 27 > 0} = {x | 2020 < x} :=
  sorry

end solution_set_l1183_118368


namespace problem_lean_l1183_118385

theorem problem_lean (k b : ℤ) : 
  ∃ n : ℤ, n = 25 ∧ n^2 = (k + 1)^4 - k^4 ∧ 3 * n + 100 = b^2 :=
sorry

end problem_lean_l1183_118385


namespace algebraic_expression_evaluation_l1183_118372

theorem algebraic_expression_evaluation (a b : ℝ) (h : -2 * a + 3 * b + 8 = 18) : 9 * b - 6 * a + 2 = 32 := by
  sorry

end algebraic_expression_evaluation_l1183_118372


namespace width_of_box_l1183_118339

theorem width_of_box (w : ℝ) (h1 : w > 0) 
    (length : ℝ) (h2 : length = 60) 
    (area_lawn : ℝ) (h3 : area_lawn = 2109) 
    (width_road : ℝ) (h4 : width_road = 3) 
    (crossroads : ℝ) (h5 : crossroads = 2 * (60 / 3 * 3)) :
    60 * w - 120 = 2109 → w = 37.15 := 
by 
  intro h6
  sorry

end width_of_box_l1183_118339


namespace find_greater_number_l1183_118381

theorem find_greater_number (a b : ℕ) (h1 : a * b = 4107) (h2 : Nat.gcd a b = 37) (h3 : a > b) : a = 111 :=
sorry

end find_greater_number_l1183_118381


namespace total_price_is_correct_l1183_118349

-- Define the cost of an adult ticket
def cost_adult : ℕ := 22

-- Define the cost of a children ticket
def cost_child : ℕ := 7

-- Define the number of adults in the family
def num_adults : ℕ := 2

-- Define the number of children in the family
def num_children : ℕ := 2

-- Define the total price the family will pay
def total_price : ℕ := cost_adult * num_adults + cost_child * num_children

-- The proof to check the total price
theorem total_price_is_correct : total_price = 58 :=
by 
  -- Here we would solve the proof
  sorry

end total_price_is_correct_l1183_118349


namespace linear_eq_with_one_variable_is_B_l1183_118362

-- Define the equations
def eqA (x y : ℝ) : Prop := 2 * x = 3 * y
def eqB (x : ℝ) : Prop := 7 * x + 5 = 6 * (x - 1)
def eqC (x : ℝ) : Prop := x^2 + (1 / 2) * (x - 1) = 1
def eqD (x : ℝ) : Prop := (1 / x) - 2 = x

-- State the problem
theorem linear_eq_with_one_variable_is_B :
  ∃ x : ℝ, ¬ (∃ y : ℝ, eqA x y) ∧ eqB x ∧ ¬ eqC x ∧ ¬ eqD x :=
by {
  -- mathematical content goes here
  sorry
}

end linear_eq_with_one_variable_is_B_l1183_118362


namespace find_x_value_l1183_118310

-- Definitions based on the conditions
def varies_inversely_as_square (k : ℝ) (x y : ℝ) : Prop := x = k / y^2

def given_condition (k : ℝ) : Prop := 1 = k / 3^2

-- The main proof problem to solve
theorem find_x_value (k : ℝ) (y : ℝ) (h1 : varies_inversely_as_square k 1 3) (h2 : y = 9) : 
  varies_inversely_as_square k (1/9) y :=
sorry

end find_x_value_l1183_118310


namespace SamBalloonsCount_l1183_118363

-- Define the conditions
def FredBalloons : ℕ := 10
def DanBalloons : ℕ := 16
def TotalBalloons : ℕ := 72

-- Define the function to calculate Sam's balloons and the main theorem to prove
def SamBalloons := TotalBalloons - (FredBalloons + DanBalloons)

theorem SamBalloonsCount : SamBalloons = 46 := by
  -- The proof is omitted here
  sorry

end SamBalloonsCount_l1183_118363


namespace shaded_percentage_correct_l1183_118356

def total_squares : ℕ := 6 * 6
def shaded_squares : ℕ := 18
def percentage_shaded (total shaded : ℕ) : ℕ := (shaded * 100) / total

theorem shaded_percentage_correct : percentage_shaded total_squares shaded_squares = 50 := by
  sorry

end shaded_percentage_correct_l1183_118356


namespace train_length_is_499_96_l1183_118397

-- Define the conditions
def speed_train_kmh : ℕ := 75   -- Speed of the train in km/h
def speed_man_kmh : ℕ := 3     -- Speed of the man in km/h
def time_cross_s : ℝ := 24.998 -- Time taken for the train to cross the man in seconds

-- Define the conversion factors
def km_to_m : ℕ := 1000        -- Conversion from kilometers to meters
def hr_to_s : ℕ := 3600        -- Conversion from hours to seconds

-- Define relative speed in m/s
def relative_speed_ms : ℕ := (speed_train_kmh - speed_man_kmh) * km_to_m / hr_to_s

-- Prove the length of the train in meters
def length_of_train : ℝ := relative_speed_ms * time_cross_s

theorem train_length_is_499_96 : length_of_train = 499.96 := sorry

end train_length_is_499_96_l1183_118397


namespace right_triangle_inequality_equality_condition_l1183_118382

theorem right_triangle_inequality (a b c : ℝ) (h : a^2 + b^2 = c^2) :
  3 * a + 4 * b ≤ 5 * c :=
by 
  sorry

theorem equality_condition (a b c : ℝ) (h : a^2 + b^2 = c^2) :
  3 * a + 4 * b = 5 * c ↔ a / b = 3 / 4 :=
by
  sorry

end right_triangle_inequality_equality_condition_l1183_118382


namespace solve_x_of_det_8_l1183_118389

variable (x : ℝ)

def matrix_det (a b c d : ℝ) : ℝ := a * d - b * c

theorem solve_x_of_det_8
  (h : matrix_det (x + 1) (1 - x) (1 - x) (x + 1) = 8) : x = 2 := by
  sorry

end solve_x_of_det_8_l1183_118389


namespace fixed_point_parabola_l1183_118332

theorem fixed_point_parabola (t : ℝ) : 4 * 3^2 + t * 3 - t^2 - 3 * t = 36 := by
  sorry

end fixed_point_parabola_l1183_118332


namespace smaller_number_is_25_l1183_118309

theorem smaller_number_is_25 (a b : ℕ) (h1 : a + b = 60) (h2 : a - b = 10) : b = 25 :=
sorry

end smaller_number_is_25_l1183_118309


namespace product_of_three_numbers_l1183_118387

theorem product_of_three_numbers (a b c : ℚ) 
  (h₁ : a + b + c = 30)
  (h₂ : a = 6 * (b + c))
  (h₃ : b = 5 * c) : 
  a * b * c = 22500 / 343 := 
sorry

end product_of_three_numbers_l1183_118387


namespace prism_faces_l1183_118358

-- Define a structure for a prism with a given number of edges
def is_prism (edges : ℕ) := 
  ∃ (n : ℕ), 3 * n = edges

-- Define the theorem to prove the number of faces in a prism given it has 21 edges
theorem prism_faces (h : is_prism 21) : ∃ (faces : ℕ), faces = 9 :=
by
  sorry

end prism_faces_l1183_118358


namespace people_in_the_theater_l1183_118321

theorem people_in_the_theater : ∃ P : ℕ, P = 100 ∧ 
  P = 19 + (1/2 : ℚ) * P + (1/4 : ℚ) * P + 6 := by
  sorry

end people_in_the_theater_l1183_118321


namespace average_remaining_two_l1183_118301

theorem average_remaining_two (a b c d e : ℝ) 
  (h1 : (a + b + c + d + e) / 5 = 12) 
  (h2 : (a + b + c) / 3 = 4) : 
  (d + e) / 2 = 24 :=
by 
  sorry

end average_remaining_two_l1183_118301


namespace distance_is_twenty_cm_l1183_118380

noncomputable def distance_between_pictures_and_board (picture_width: ℕ) (board_width_m: ℕ) (board_width_cm: ℕ) (number_of_pictures: ℕ) : ℕ :=
  let board_total_width := board_width_m * 100 + board_width_cm
  let total_pictures_width := number_of_pictures * picture_width
  let total_distance := board_total_width - total_pictures_width
  let total_gaps := number_of_pictures + 1
  total_distance / total_gaps

theorem distance_is_twenty_cm :
  distance_between_pictures_and_board 30 3 20 6 = 20 :=
by
  sorry

end distance_is_twenty_cm_l1183_118380


namespace valid_p_interval_l1183_118383

theorem valid_p_interval :
  ∀ p, (∀ q, q > 0 → (4 * (p * q^2 + p^2 * q + 4 * q^2 + 4 * p * q)) / (p + q) > 3 * p^2 * q) ↔ 0 ≤ p ∧ p < 4 :=
sorry

end valid_p_interval_l1183_118383


namespace combined_molecular_weight_l1183_118342

theorem combined_molecular_weight :
  let CaO_molecular_weight := 56.08
  let CO2_molecular_weight := 44.01
  let HNO3_molecular_weight := 63.01
  let moles_CaO := 5
  let moles_CO2 := 3
  let moles_HNO3 := 2
  moles_CaO * CaO_molecular_weight + moles_CO2 * CO2_molecular_weight + moles_HNO3 * HNO3_molecular_weight = 538.45 :=
by sorry

end combined_molecular_weight_l1183_118342


namespace volume_to_surface_area_ratio_l1183_118337

-- Define the structure of the object consisting of unit cubes
structure CubicObject where
  volume : ℕ
  surface_area : ℕ

-- Define a specific cubic object based on given conditions
def specialCubicObject : CubicObject := {
  volume := 8,
  surface_area := 29
}

-- Statement to prove the ratio of the volume to the surface area
theorem volume_to_surface_area_ratio :
  (specialCubicObject.volume : ℚ) / (specialCubicObject.surface_area : ℚ) = 8 / 29 := by
  sorry

end volume_to_surface_area_ratio_l1183_118337


namespace complement_union_sets_l1183_118357

open Set

theorem complement_union_sets :
  ∀ (U A B : Set ℕ), (U = {1, 2, 3, 4}) → (A = {2, 3}) → (B = {3, 4}) → (U \ (A ∪ B) = {1}) :=
by
  intros U A B hU hA hB
  rw [hU, hA, hB]
  simp 
  sorry

end complement_union_sets_l1183_118357


namespace root_of_quadratic_l1183_118392

theorem root_of_quadratic {x a : ℝ} (h : x = 2 ∧ x^2 - x + a = 0) : a = -2 := 
by
  sorry

end root_of_quadratic_l1183_118392


namespace find_f_neg1_l1183_118305

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_neg1 :
  (∀ x, f (-x) = -f x) →
  (∀ x, (0 < x) → f x = 2 * x * (x + 1)) →
  f (-1) = -4 := by
  intros h1 h2
  sorry

end find_f_neg1_l1183_118305


namespace range_of_a_l1183_118304

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 2 * x - a * Real.log x

theorem range_of_a (a : ℝ) :
  (∀ x > 0, f x a ≥ 2 * a - (1 / 2) * a^2) ↔ 0 ≤ a :=
by
  sorry

end range_of_a_l1183_118304


namespace problem_I_number_of_zeros_problem_II_inequality_l1183_118391

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x * Real.exp 1 - 1

theorem problem_I_number_of_zeros : 
  ∃! (x1 x2 : ℝ), f x1 = 0 ∧ f x2 = 0 ∧ x1 ≠ x2 := 
sorry

theorem problem_II_inequality (a : ℝ) (h_a : a ≤ 0) (x : ℝ) (h_x : x ≥ 1) : 
  f x ≥ a * Real.log x - 1 := 
sorry

end problem_I_number_of_zeros_problem_II_inequality_l1183_118391


namespace sin_half_angle_identity_l1183_118311

theorem sin_half_angle_identity (theta : ℝ) (h : Real.sin (Real.pi / 2 + theta) = - 1 / 2) :
  2 * Real.sin (theta / 2) ^ 2 - 1 = 1 / 2 := 
by
  sorry

end sin_half_angle_identity_l1183_118311


namespace simplify_expression_l1183_118359

variable (x y : ℤ)

theorem simplify_expression : 
  (15 * x + 45 * y) + (7 * x + 18 * y) - (6 * x + 35 * y) = 16 * x + 28 * y :=
by
  sorry

end simplify_expression_l1183_118359


namespace chimney_bricks_l1183_118314

theorem chimney_bricks (x : ℕ) 
  (h1 : Brenda_rate = x / 8) 
  (h2 : Brandon_rate = x / 12) 
  (h3 : Brian_rate = x / 16) 
  (h4 : effective_combined_rate = (Brenda_rate + Brandon_rate + Brian_rate) - 15) 
  (h5 : total_time = 4) :
  (4 * effective_combined_rate) = x := 
  sorry

end chimney_bricks_l1183_118314


namespace calculate_difference_l1183_118330

theorem calculate_difference :
  let a := 3.56
  let b := 2.1
  let c := 1.5
  a - (b * c) = 0.41 :=
by
  let a := 3.56
  let b := 2.1
  let c := 1.5
  show a - (b * c) = 0.41
  sorry

end calculate_difference_l1183_118330


namespace ring_matching_possible_iff_odd_l1183_118317

theorem ring_matching_possible_iff_odd (n : ℕ) (hn : n ≥ 3) :
  (∃ f : ℕ → ℕ, (∀ k : ℕ, k < n → ∃ j : ℕ, j < n ∧ f (j + k) % n = k % n) ↔ Odd n) :=
sorry

end ring_matching_possible_iff_odd_l1183_118317


namespace grape_juice_percentage_l1183_118325

theorem grape_juice_percentage
  (initial_volume : ℝ) (initial_percentage : ℝ) (added_juice : ℝ)
  (h_initial_volume : initial_volume = 50)
  (h_initial_percentage : initial_percentage = 0.10)
  (h_added_juice : added_juice = 10) :
  ((initial_percentage * initial_volume + added_juice) / (initial_volume + added_juice) * 100) = 25 := 
by
  sorry

end grape_juice_percentage_l1183_118325


namespace josanna_minimum_test_score_l1183_118351

def test_scores := [90, 80, 70, 60, 85]

def target_average_increase := 3

def current_average (scores : List ℕ) : ℚ :=
  (scores.sum : ℚ) / scores.length

def sixth_test_score_needed (scores : List ℕ) (increase : ℚ) : ℚ :=
  let current_avg := current_average scores
  let target_avg := current_avg + increase
  target_avg * (scores.length + 1) - scores.sum

theorem josanna_minimum_test_score :
  sixth_test_score_needed test_scores target_average_increase = 95 := sorry

end josanna_minimum_test_score_l1183_118351


namespace geo_seq_4th_term_l1183_118343

theorem geo_seq_4th_term (a r : ℝ) (h₀ : a = 512) (h₆ : a * r^5 = 32) :
  a * r^3 = 64 :=
by 
  sorry

end geo_seq_4th_term_l1183_118343


namespace selling_price_l1183_118364

theorem selling_price (cost_price : ℕ) (profit_percent : ℕ) (selling_price : ℕ) : 
  cost_price = 2400 ∧ profit_percent = 6 → selling_price = 2544 := by
  sorry

end selling_price_l1183_118364


namespace trees_to_plant_l1183_118367

def road_length : ℕ := 156
def interval : ℕ := 6
def trees_needed (road_length interval : ℕ) := road_length / interval + 1

theorem trees_to_plant : trees_needed road_length interval = 27 := by
  sorry

end trees_to_plant_l1183_118367


namespace sum_of_fractions_l1183_118375

theorem sum_of_fractions : (1 / 3 : ℚ) + (2 / 7) = 13 / 21 :=
by
  sorry

end sum_of_fractions_l1183_118375


namespace rubber_ball_radius_l1183_118396

theorem rubber_ball_radius (r : ℝ) (radius_exposed_section : ℝ) (depth : ℝ) 
  (h1 : radius_exposed_section = 20) 
  (h2 : depth = 12) 
  (h3 : (r - depth)^2 + radius_exposed_section^2 = r^2) : 
  r = 22.67 :=
by
  sorry

end rubber_ball_radius_l1183_118396


namespace inv_mod_35_l1183_118360

theorem inv_mod_35 : ∃ x : ℕ, 5 * x ≡ 1 [MOD 35] :=
by
  use 29
  sorry

end inv_mod_35_l1183_118360


namespace probability_heads_equals_7_over_11_l1183_118323

theorem probability_heads_equals_7_over_11 (p : ℝ) (q : ℝ)
  (h1 : q = 1 - p)
  (h2 : 120 * p^7 * q^3 = 210 * p^6 * q^4) :
  p = 7 / 11 :=
by {
  sorry
}

end probability_heads_equals_7_over_11_l1183_118323


namespace parabola_standard_equation_l1183_118328

variable {a : ℝ} (h : a < 0)

theorem parabola_standard_equation (h : a < 0) :
  ∃ (p : ℝ), p = -2 * a ∧ (∀ (x y : ℝ), y^2 = -2 * p * x ↔ y^2 = 4 * a * x) :=
sorry

end parabola_standard_equation_l1183_118328
