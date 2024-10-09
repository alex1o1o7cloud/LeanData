import Mathlib

namespace compute_f_1986_l1716_171678

noncomputable def f : ℕ → ℤ := sorry

axiom f_defined_for_nonneg_integers : ∀ x : ℕ, ∃ y : ℤ, f x = y
axiom f_one : f 1 = 1
axiom f_functional_equation : ∀ (a b : ℕ), f (a + b) = f a + f b - 2 * f (a * b)

theorem compute_f_1986 : f 1986 = 0 :=
  sorry

end compute_f_1986_l1716_171678


namespace soccer_ball_cost_l1716_171622

theorem soccer_ball_cost (x : ℕ) (h : 5 * x + 4 * 65 = 980) : x = 144 :=
by
  sorry

end soccer_ball_cost_l1716_171622


namespace speed_in_still_water_l1716_171660

/--
A man can row upstream at 55 kmph and downstream at 65 kmph.
Prove that his speed in still water is 60 kmph.
-/
theorem speed_in_still_water (upstream_speed downstream_speed : ℕ) (h_upstream : upstream_speed = 55) (h_downstream : downstream_speed = 65) : 
  (upstream_speed + downstream_speed) / 2 = 60 := by
  sorry

end speed_in_still_water_l1716_171660


namespace eq_from_conditions_l1716_171694

theorem eq_from_conditions (a b : ℂ) :
  (1 / (a + b)) ^ 2003 = 1 ∧ (-a + b) ^ 2005 = 1 → a ^ 2003 + b ^ 2004 = 1 := 
by
  sorry

end eq_from_conditions_l1716_171694


namespace order_of_abc_l1716_171695

noncomputable def a : ℝ := (0.3)^3
noncomputable def b : ℝ := (3)^3
noncomputable def c : ℝ := Real.log 0.3 / Real.log 3

theorem order_of_abc : b > a ∧ a > c :=
by
  have ha : a = (0.3)^3 := rfl
  have hb : b = (3)^3 := rfl
  have hc : c = Real.log 0.3 / Real.log 3 := rfl
  sorry

end order_of_abc_l1716_171695


namespace GCF_LCM_15_21_14_20_l1716_171675

def GCF (a b : ℕ) : ℕ := Nat.gcd a b
def LCM (a b : ℕ) : ℕ := Nat.lcm a b

theorem GCF_LCM_15_21_14_20 :
  GCF (LCM 15 21) (LCM 14 20) = 35 :=
by
  sorry

end GCF_LCM_15_21_14_20_l1716_171675


namespace student_arrangement_count_l1716_171639

theorem student_arrangement_count :
  let males := 4
  let females := 5
  let select_males := 2
  let select_females := 3
  let total_selected := select_males + select_females
  (Nat.choose males select_males) * (Nat.choose females select_females) * (Nat.factorial total_selected) = 7200 := 
by
  sorry

end student_arrangement_count_l1716_171639


namespace max_ratio_lemma_l1716_171615

theorem max_ratio_lemma (a : ℕ → ℝ) (S : ℕ → ℝ)
  (hSn : ∀ n, S n = (n + 1) / 2 * a n)
  (hSn_minus_one : ∀ n, S (n - 1) = n / 2 * a (n - 1)) :
  ∀ n > 1, (a n / a (n - 1) ≤ 2) ∧ (a 2 / a 1 = 2) := sorry

end max_ratio_lemma_l1716_171615


namespace max_piece_length_total_pieces_l1716_171670

-- Definitions based on the problem's conditions
def length1 : ℕ := 42
def length2 : ℕ := 63
def gcd_length : ℕ := Nat.gcd length1 length2

-- Theorem statements based on the realized correct answers
theorem max_piece_length (h1 : length1 = 42) (h2 : length2 = 63) :
  gcd_length = 21 := by
  sorry

theorem total_pieces (h1 : length1 = 42) (h2 : length2 = 63) :
  (length1 / gcd_length) + (length2 / gcd_length) = 5 := by
  sorry

end max_piece_length_total_pieces_l1716_171670


namespace tan_arithmetic_geometric_l1716_171649

noncomputable def a_seq : ℕ → ℝ := sorry -- Define a_n as an arithmetic sequence (details abstracted)
noncomputable def b_seq : ℕ → ℝ := sorry -- Define b_n as a geometric sequence (details abstracted)

axiom a_seq_is_arithmetic : ∀ n m : ℕ, a_seq (n + 1) - a_seq n = a_seq (m + 1) - a_seq m
axiom b_seq_is_geometric : ∀ n : ℕ, ∃ r : ℝ, b_seq (n + 1) = b_seq n * r
axiom a_seq_sum : a_seq 2017 + a_seq 2018 = Real.pi
axiom b_seq_square : b_seq 20 ^ 2 = 4

theorem tan_arithmetic_geometric : 
  (Real.tan ((a_seq 2 + a_seq 4033) / (b_seq 1 * b_seq 39)) = 1) :=
sorry

end tan_arithmetic_geometric_l1716_171649


namespace hyperbola_eccentricity_l1716_171692

theorem hyperbola_eccentricity (a b c e : ℝ) 
  (h_eq1 : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1)
  (h_eq2 : c = Real.sqrt (a^2 + b^2))
  (h_dist : ∀ x, x = b * c / Real.sqrt (a^2 + b^2))
  (h_eq3 : a = b) :
  e = Real.sqrt 2 :=
by
  sorry

end hyperbola_eccentricity_l1716_171692


namespace set_star_result_l1716_171604

-- Define the sets A and B
def A : Set ℕ := {1, 2}
def B : Set ℕ := {1, 2, 3}

-- Define the operation ∗ between sets A and B
def set_star (A B : Set ℕ) : Set ℕ := {x | ∃ x1 ∈ A, ∃ x2 ∈ B, x = x1 + x2}

-- Rewrite the main theorem to be proven
theorem set_star_result : set_star A B = {2, 3, 4, 5} :=
  sorry

end set_star_result_l1716_171604


namespace find_num_round_balloons_l1716_171602

variable (R : ℕ) -- Number of bags of round balloons that Janeth bought
variable (RoundBalloonsPerBag : ℕ := 20)
variable (LongBalloonsPerBag : ℕ := 30)
variable (BagsLongBalloons : ℕ := 4)
variable (BurstRoundBalloons : ℕ := 5)
variable (BalloonsLeft : ℕ := 215)

def total_long_balloons : ℕ := BagsLongBalloons * LongBalloonsPerBag
def total_balloons : ℕ := R * RoundBalloonsPerBag + total_long_balloons - BurstRoundBalloons

theorem find_num_round_balloons :
  BalloonsLeft = total_balloons → R = 5 := by
  sorry

end find_num_round_balloons_l1716_171602


namespace fractions_simplify_to_prime_denominator_2023_l1716_171679

def num_fractions_simplifying_to_prime_denominator (n: ℕ) (p q: ℕ) : ℕ :=
  let multiples (m: ℕ) : ℕ := (n - 1) / m
  multiples p + multiples (p * q)

theorem fractions_simplify_to_prime_denominator_2023 :
  num_fractions_simplifying_to_prime_denominator 2023 17 7 = 22 :=
by
  sorry

end fractions_simplify_to_prime_denominator_2023_l1716_171679


namespace shirt_cost_l1716_171617

theorem shirt_cost (J S : ℕ) 
  (h₁ : 3 * J + 2 * S = 69) 
  (h₂ : 2 * J + 3 * S = 61) :
  S = 9 :=
by 
  sorry

end shirt_cost_l1716_171617


namespace total_money_spent_l1716_171638

def time_in_minutes_at_arcade : ℕ := 3 * 60
def cost_per_interval : ℕ := 50 -- in cents
def interval_duration : ℕ := 6 -- in minutes
def total_intervals : ℕ := time_in_minutes_at_arcade / interval_duration

theorem total_money_spent :
  ((total_intervals * cost_per_interval) = 1500) := 
by
  sorry

end total_money_spent_l1716_171638


namespace remainder_div_9_l1716_171643

theorem remainder_div_9 (x y : ℤ) (h : 9 ∣ (x + 2 * y)) : (2 * (5 * x - 8 * y - 4)) % 9 = -8 ∨ (2 * (5 * x - 8 * y - 4)) % 9 = 1 :=
by
  sorry

end remainder_div_9_l1716_171643


namespace tv_price_increase_percentage_l1716_171619

theorem tv_price_increase_percentage (P Q : ℝ) (x : ℝ) :
  (P * (1 + x / 100) * Q * 0.8 = P * Q * 1.28) → x = 60 :=
by sorry

end tv_price_increase_percentage_l1716_171619


namespace work_increase_percentage_l1716_171611

theorem work_increase_percentage (p : ℕ) (hp : p > 0) : 
  let absent_fraction := 1 / 6
  let work_per_person_original := 1 / p
  let present_people := p - p * absent_fraction
  let work_per_person_new := 1 / present_people
  let work_increase := work_per_person_new - work_per_person_original
  let percentage_increase := (work_increase / work_per_person_original) * 100
  percentage_increase = 20 :=
by
  sorry

end work_increase_percentage_l1716_171611


namespace triangle_area_correct_l1716_171606

def point := ℝ × ℝ

def triangle_area (A B C : point) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem triangle_area_correct :
  let A : point := (3, -1)
  let B : point := (3, 6)
  let C : point := (8, 6)
  triangle_area A B C = 17.5 :=
by
  sorry

end triangle_area_correct_l1716_171606


namespace stock_profit_percentage_l1716_171683

theorem stock_profit_percentage 
  (total_stock : ℝ) (total_loss : ℝ) (profit_percentage : ℝ) (loss_percentage : ℝ)
  (percentage_sold_at_profit : ℝ) :
  total_stock = 12499.99 →
  total_loss = 500 →
  profit_percentage = 0.20 →
  loss_percentage = 0.10 →
  (0.10 * ((100 - percentage_sold_at_profit) / 100) * 12499.99) - (0.20 * (percentage_sold_at_profit / 100) * 12499.99) = 500 →
  percentage_sold_at_profit = 20 :=
sorry

end stock_profit_percentage_l1716_171683


namespace find_third_side_l1716_171693

theorem find_third_side (a b : ℝ) (c : ℕ) 
  (h1 : a = 3.14)
  (h2 : b = 0.67)
  (h_triangle_ineq : a + b > ↑c ∧ a + ↑c > b ∧ b + ↑c > a) : 
  c = 3 := 
by
  -- Proof goes here
  sorry

end find_third_side_l1716_171693


namespace total_students_in_college_l1716_171655

theorem total_students_in_college (B G : ℕ) (h_ratio: 8 * G = 5 * B) (h_girls: G = 175) :
  B + G = 455 := 
  sorry

end total_students_in_college_l1716_171655


namespace order_of_logs_l1716_171685

open Real

noncomputable def a := log 10 / log 5
noncomputable def b := log 12 / log 6
noncomputable def c := 1 + log 2 / log 7

theorem order_of_logs : a > b ∧ b > c :=
by
  sorry

end order_of_logs_l1716_171685


namespace complex_number_pow_two_l1716_171687

theorem complex_number_pow_two (i : ℂ) (hi : i^2 = -1) : (1 + i)^2 = 2 * i :=
by sorry

end complex_number_pow_two_l1716_171687


namespace bug_meeting_point_l1716_171697
-- Import the necessary library

-- Define the side lengths of the triangle
variables (DE EF FD : ℝ)
variables (bugs_meet : ℝ)

-- State the conditions and the result
theorem bug_meeting_point
  (h1 : DE = 6)
  (h2 : EF = 8)
  (h3 : FD = 10)
  (h4 : bugs_meet = 1 / 2 * (DE + EF + FD)) :
  bugs_meet - DE = 6 :=
by
  sorry

end bug_meeting_point_l1716_171697


namespace net_increase_proof_l1716_171600

def initial_cars := 50
def initial_motorcycles := 75
def initial_vans := 25

def car_arrival_rate : ℝ := 70
def car_departure_rate : ℝ := 40
def motorcycle_arrival_rate : ℝ := 120
def motorcycle_departure_rate : ℝ := 60
def van_arrival_rate : ℝ := 30
def van_departure_rate : ℝ := 20

def play_duration : ℝ := 2.5

def net_increase_car : ℝ := play_duration * (car_arrival_rate - car_departure_rate)
def net_increase_motorcycle : ℝ := play_duration * (motorcycle_arrival_rate - motorcycle_departure_rate)
def net_increase_van : ℝ := play_duration * (van_arrival_rate - van_departure_rate)

theorem net_increase_proof :
  net_increase_car = 75 ∧
  net_increase_motorcycle = 150 ∧
  net_increase_van = 25 :=
by
  -- Proof would go here.
  sorry

end net_increase_proof_l1716_171600


namespace evaluate_expression_l1716_171609

theorem evaluate_expression :
  -1^2008 + 3*(-1)^2007 + 1^2008 - 2*(-1)^2009 = -5 := 
by
  sorry

end evaluate_expression_l1716_171609


namespace derivative_of_sin_squared_minus_cos_squared_l1716_171642

noncomputable def func (x : ℝ) : ℝ := (Real.sin x)^2 - (Real.cos x)^2

theorem derivative_of_sin_squared_minus_cos_squared (x : ℝ) :
  deriv func x = 2 * Real.sin (2 * x) :=
sorry

end derivative_of_sin_squared_minus_cos_squared_l1716_171642


namespace multiplication_72519_9999_l1716_171658

theorem multiplication_72519_9999 :
  72519 * 9999 = 725117481 :=
by
  sorry

end multiplication_72519_9999_l1716_171658


namespace max_volume_of_pyramid_PABC_l1716_171656

noncomputable def max_pyramid_volume (PA PB AB BC CA : ℝ) (hPA : PA = 3) (hPB : PB = 3) 
(hAB : AB = 2) (hBC : BC = 2) (hCA : CA = 2) : ℝ :=
  let D := 1 -- Midpoint of segment AB
  let PD : ℝ := Real.sqrt (PA ^ 2 - D ^ 2) -- Distance PD using Pythagorean theorem
  let S_ABC : ℝ := (Real.sqrt 3 / 4) * (AB ^ 2) -- Area of triangle ABC
  let V_PABC : ℝ := (1 / 3) * S_ABC * PD -- Volume of the pyramid
  V_PABC -- Return the volume

theorem max_volume_of_pyramid_PABC : 
  max_pyramid_volume 3 3 2 2 2  (rfl) (rfl) (rfl) (rfl) (rfl) = (2 * Real.sqrt 6) / 3 :=
by
  sorry

end max_volume_of_pyramid_PABC_l1716_171656


namespace tank_empty_time_l1716_171644

theorem tank_empty_time (V : ℝ) (r_inlet r_outlet1 r_outlet2 : ℝ) (I : V = 20 * 12^3)
  (r_inlet_val : r_inlet = 5) (r_outlet1_val : r_outlet1 = 9) 
  (r_outlet2_val : r_outlet2 = 8) : 
  (V / ((r_outlet1 + r_outlet2) - r_inlet) = 2880) :=
by
  sorry

end tank_empty_time_l1716_171644


namespace calculation_1500_increased_by_45_percent_l1716_171661

theorem calculation_1500_increased_by_45_percent :
  1500 * (1 + 45 / 100) = 2175 := 
by
  sorry

end calculation_1500_increased_by_45_percent_l1716_171661


namespace marias_workday_end_time_l1716_171659

theorem marias_workday_end_time :
  ∀ (start_time : ℕ) (lunch_time : ℕ) (work_duration : ℕ) (lunch_break : ℕ) (total_work_time : ℕ),
  start_time = 8 ∧ lunch_time = 13 ∧ work_duration = 8 ∧ lunch_break = 1 →
  (total_work_time = work_duration - (lunch_time - start_time - lunch_break)) →
  lunch_time + 1 + (work_duration - (lunch_time - start_time)) = 17 :=
by
  sorry

end marias_workday_end_time_l1716_171659


namespace patricia_earns_more_than_jose_l1716_171673

noncomputable def jose_final_amount : ℝ :=
  50000 * (1 + 0.04)^2

noncomputable def patricia_final_amount : ℝ :=
  50000 * (1 + 0.01)^8

theorem patricia_earns_more_than_jose :
  patricia_final_amount - jose_final_amount = 63 :=
by
  -- from solution steps
  /-
  jose_final_amount = 50000 * (1 + 0.04)^2 = 54080
  patricia_final_amount = 50000 * (1 + 0.01)^8 ≈ 54143
  patricia_final_amount - jose_final_amount ≈ 63
  -/
  sorry

end patricia_earns_more_than_jose_l1716_171673


namespace solution_set_of_inequality_l1716_171603

theorem solution_set_of_inequality :
  {x : ℝ | (x-1)*(2-x) ≥ 0} = {x : ℝ | 1 ≤ x ∧ x ≤ 2} :=
sorry

end solution_set_of_inequality_l1716_171603


namespace remove_one_and_average_l1716_171667

theorem remove_one_and_average (l : List ℕ) (n : ℕ) (avg : ℚ) :
  l = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] →
  avg = 8.5 →
  (l.sum - n : ℚ) = 14 * avg →
  n = 1 :=
by
  intros hlist havg hsum
  sorry

end remove_one_and_average_l1716_171667


namespace travelers_cross_river_l1716_171684

variables (traveler1 traveler2 traveler3 : ℕ)  -- weights of travelers
variable (raft_capacity : ℕ)  -- maximum carrying capacity of the raft

-- Given conditions
def conditions :=
  traveler1 = 3 ∧ traveler2 = 3 ∧ traveler3 = 5 ∧ raft_capacity = 7

-- Prove that the travelers can all cross the river successfully
theorem travelers_cross_river :
  conditions traveler1 traveler2 traveler3 raft_capacity →
  (traveler1 + traveler2 ≤ raft_capacity) ∧
  (traveler1 ≤ raft_capacity) ∧
  (traveler3 ≤ raft_capacity) ∧
  (traveler1 + traveler2 ≤ raft_capacity) →
  true :=
by
  intros h_conditions h_validity
  sorry

end travelers_cross_river_l1716_171684


namespace linear_function_quadrants_l1716_171668

theorem linear_function_quadrants (k : ℝ) :
  (k - 3 > 0) ∧ (-k + 2 < 0) → k > 3 :=
by
  intro h
  sorry

end linear_function_quadrants_l1716_171668


namespace intersection_eq_l1716_171629

def setA : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 3}
def setB : Set ℝ := {x : ℝ | x > 2 ∨ x < -1}

theorem intersection_eq : (setA ∩ setB) = {x : ℝ | 2 < x ∧ x ≤ 3} := by
  sorry

end intersection_eq_l1716_171629


namespace least_n_div_mod_l1716_171610

theorem least_n_div_mod (n : ℕ) (h_pos : n > 1) (h_mod25 : n % 25 = 1) (h_mod7 : n % 7 = 1) : n = 176 :=
by
  sorry

end least_n_div_mod_l1716_171610


namespace lowest_value_meter_can_record_l1716_171646

theorem lowest_value_meter_can_record (A B C : ℝ) 
  (h1 : (A + B + C) / 3 = 6) (h2 : A = 2) : A = 2 :=
by sorry

end lowest_value_meter_can_record_l1716_171646


namespace find_number_l1716_171640

theorem find_number (N : ℝ) (h : (5/4 : ℝ) * N = (4/5 : ℝ) * N + 27) : N = 60 :=
by
  sorry

end find_number_l1716_171640


namespace subtract_correctly_l1716_171612

theorem subtract_correctly (x : ℕ) (h : x + 35 = 77) : x - 35 = 7 :=
sorry

end subtract_correctly_l1716_171612


namespace determine_quadrant_l1716_171624

def pointInWhichQuadrant (x y : ℤ) : String :=
  if x > 0 ∧ y > 0 then "First quadrant"
  else if x < 0 ∧ y > 0 then "Second quadrant"
  else if x < 0 ∧ y < 0 then "Third quadrant"
  else if x > 0 ∧ y < 0 then "Fourth quadrant"
  else "On axis or origin"

theorem determine_quadrant : pointInWhichQuadrant (-7) 3 = "Second quadrant" :=
by
  sorry

end determine_quadrant_l1716_171624


namespace simplify_sqrt_of_square_l1716_171637

-- The given condition
def x : ℤ := -9

-- The theorem stating the simplified form
theorem simplify_sqrt_of_square : (Real.sqrt ((x : ℝ) ^ 2) = 9) := by    
    sorry

end simplify_sqrt_of_square_l1716_171637


namespace parallelepiped_volume_k_l1716_171625

theorem parallelepiped_volume_k (k : ℝ) : 
    abs (3 * k^2 - 13 * k + 27) = 20 ↔ k = (13 + Real.sqrt 85) / 6 ∨ k = (13 - Real.sqrt 85) / 6 := 
by sorry

end parallelepiped_volume_k_l1716_171625


namespace union_set_when_m_neg3_range_of_m_for_intersection_l1716_171635

def setA (x : ℝ) : Prop := x^2 - x - 12 ≤ 0
def setB (x m : ℝ) : Prop := 2*m - 1 ≤ x ∧ x ≤ m + 1

theorem union_set_when_m_neg3 : 
  (∀ x, setA x ∨ setB x (-3) ↔ -7 ≤ x ∧ x ≤ 4) := 
by sorry

theorem range_of_m_for_intersection :
  (∀ m x, (setA x ∧ setB x m ↔ setB x m) → m ≥ -1) := 
by sorry

end union_set_when_m_neg3_range_of_m_for_intersection_l1716_171635


namespace smaller_rectangle_length_ratio_l1716_171689

theorem smaller_rectangle_length_ratio 
  (s : ℝ)
  (h1 : 5 = 5)
  (h2 : ∃ r : ℝ, r = s)
  (h3 : ∀ x : ℝ, x = s)
  (h4 : ∀ y : ℝ, y / 2 = s / 2)
  (h5 : ∀ z : ℝ, z = 3 * s)
  (h6 : ∀ w : ℝ, w = s) :
  ∃ l : ℝ, l / s = 4 :=
sorry

end smaller_rectangle_length_ratio_l1716_171689


namespace probability_of_at_least_one_three_l1716_171669

def probability_at_least_one_three_shows : ℚ :=
  let total_outcomes : ℚ := 64
  let favorable_outcomes : ℚ := 15
  favorable_outcomes / total_outcomes

theorem probability_of_at_least_one_three (a b : ℕ) (ha : 1 ≤ a ∧ a ≤ 8) (hb : 1 ≤ b ∧ b ≤ 8) :
    (a = 3 ∨ b = 3) → probability_at_least_one_three_shows = 15 / 64 := by
  sorry

end probability_of_at_least_one_three_l1716_171669


namespace min_n_for_triangle_pattern_l1716_171651

/-- 
There are two types of isosceles triangles with a waist length of 1:
-  Type 1: An acute isosceles triangle with a vertex angle of 30 degrees.
-  Type 2: A right isosceles triangle with a vertex angle of 90 degrees.
They are placed around a point in a clockwise direction in a sequence such that:
- The 1st and 2nd are acute isosceles triangles (30 degrees),
- The 3rd is a right isosceles triangle (90 degrees),
- The 4th and 5th are acute isosceles triangles (30 degrees),
- The 6th is a right isosceles triangle (90 degrees), and so on.

Prove that the minimum value of n such that the nth triangle coincides exactly with
the 1st triangle is 23.
-/
theorem min_n_for_triangle_pattern : ∃ n : ℕ, n = 23 ∧ (∀ m < 23, m ≠ 23) :=
sorry

end min_n_for_triangle_pattern_l1716_171651


namespace num_pairs_with_math_book_l1716_171613

theorem num_pairs_with_math_book (books : Finset String) (h : books = {"Chinese", "Mathematics", "English", "Biology", "History"}):
  (∃ pairs : Finset (Finset String), pairs.card = 4 ∧ ∀ pair ∈ pairs, "Mathematics" ∈ pair) :=
by
  sorry

end num_pairs_with_math_book_l1716_171613


namespace point_on_y_axis_l1716_171608

theorem point_on_y_axis (a : ℝ) 
  (h : (a - 2) = 0) : a = 2 := 
  by 
    sorry

end point_on_y_axis_l1716_171608


namespace part1_part2_l1716_171618

variables {A B C : ℝ} {a b c : ℝ}

-- conditions of the problem
def condition_1 (a b c : ℝ) (C : ℝ) : Prop :=
  a * Real.cos C + Real.sqrt 3 * Real.sin C - b - c = 0

def condition_2 (C : ℝ) : Prop :=
  0 < C ∧ C < Real.pi

-- Part 1: Proving the value of angle A
theorem part1 (a b c C : ℝ) (h1 : condition_1 a b c C) (h2 : condition_2 C) : 
  A = Real.pi / 3 :=
sorry

-- Part 2: Range of possible values for the perimeter, given c = 3
def is_acute_triangle (A B C : ℝ) : Prop :=
  A < Real.pi / 2 ∧ B < Real.pi / 2 ∧ C < Real.pi / 2

theorem part2 (a b A B C : ℝ) (h1 : condition_1 a b 3 C) (h2 : condition_2 C) 
           (h3 : A = Real.pi / 3) (h4 : is_acute_triangle A B C) :
  ∃ p, p ∈ Set.Ioo ((3 * Real.sqrt 3 + 9) / 2) (9 + 3 * Real.sqrt 3) :=
sorry

end part1_part2_l1716_171618


namespace range_of_x_l1716_171631

noncomputable def f : ℝ → ℝ := sorry

theorem range_of_x (f_even : ∀ x : ℝ, f x = f (-x))
                   (f_increasing : ∀ x y : ℝ, 0 ≤ x → x ≤ y → f x ≤ f y)
                   (f_at_one_third : f (1/3) = 0) :
  {x : ℝ | f (Real.log x / Real.log (1/8)) > 0} = {x : ℝ | (0 < x ∧ x < 1/2) ∨ 2 < x} :=
sorry

end range_of_x_l1716_171631


namespace probability_standard_weight_l1716_171699

noncomputable def total_students : ℕ := 500
noncomputable def standard_students : ℕ := 350

theorem probability_standard_weight : (standard_students : ℚ) / (total_students : ℚ) = 7 / 10 :=
by {
  sorry
}

end probability_standard_weight_l1716_171699


namespace intersection_M_N_l1716_171665

open Set

def M := {x : ℝ | x^2 < 4}
def N := {x : ℝ | x^2 - 2*x - 3 < 0}
def intersection_sets := {x : ℝ | -1 < x ∧ x < 2}

theorem intersection_M_N : M ∩ N = intersection_sets :=
  sorry

end intersection_M_N_l1716_171665


namespace compute_Q3_Qneg3_l1716_171663

noncomputable def Q (x : ℝ) (a b c m : ℝ) : ℝ :=
  a * x^3 + b * x^2 + c * x + m

theorem compute_Q3_Qneg3 (a b c m : ℝ)
  (h1 : Q 1 a b c m = 3 * m)
  (h2 : Q (-1) a b c m = 4 * m)
  (h3 : Q 0 a b c m = m) :
  Q 3 a b c m + Q (-3) a b c m = 47 * m :=
by
  sorry

end compute_Q3_Qneg3_l1716_171663


namespace length_of_bridge_l1716_171636

theorem length_of_bridge 
  (lenA : ℝ) (speedA : ℝ) (lenB : ℝ) (speedB : ℝ) (timeA : ℝ) (timeB : ℝ) (startAtSameTime : Prop)
  (h1 : lenA = 120) (h2 : speedA = 12.5) (h3 : lenB = 150) (h4 : speedB = 15.28) 
  (h5 : timeA = 30) (h6 : timeB = 25) : 
  (∃ X : ℝ, X = 757) :=
by
  sorry

end length_of_bridge_l1716_171636


namespace fujian_provincial_games_distribution_count_l1716_171696

theorem fujian_provincial_games_distribution_count 
  (staff_members : Finset String)
  (locations : Finset String)
  (A B C D E F : String)
  (A_in_B : A ∈ staff_members)
  (B_in_B : B ∈ staff_members)
  (C_in_B : C ∈ staff_members)
  (D_in_B : D ∈ staff_members)
  (E_in_B : E ∈ staff_members)
  (F_in_B : F ∈ staff_members)
  (locations_count : locations.card = 2)
  (staff_count : staff_members.card = 6)
  (must_same_group : ∀ g₁ g₂ : Finset String, A ∈ g₁ → B ∈ g₁ → g₁ ∪ g₂ = staff_members)
  (min_two_people : ∀ g : Finset String, 2 ≤ g.card) :
  ∃ distrib_methods : ℕ, distrib_methods = 22 := 
by
  sorry

end fujian_provincial_games_distribution_count_l1716_171696


namespace yanna_baked_butter_cookies_in_morning_l1716_171676

-- Define the conditions
def biscuits_morning : ℕ := 40
def biscuits_afternoon : ℕ := 20
def cookies_afternoon : ℕ := 10
def total_more_biscuits : ℕ := 30

-- Define the statement to be proved
theorem yanna_baked_butter_cookies_in_morning (B : ℕ) : 
  (biscuits_morning + biscuits_afternoon = (B + cookies_afternoon) + total_more_biscuits) → B = 20 :=
by
  sorry

end yanna_baked_butter_cookies_in_morning_l1716_171676


namespace geometric_sequence_k_value_l1716_171688

theorem geometric_sequence_k_value
  (k : ℤ)
  (S : ℕ → ℤ)
  (a : ℕ → ℤ)
  (h1 : ∀ n, S n = 3 * 2^n + k)
  (h2 : ∀ n, n ≥ 2 → a n = S n - S (n - 1))
  (h3 : ∃ r, ∀ n, a (n + 1) = r * a n) : k = -3 :=
sorry

end geometric_sequence_k_value_l1716_171688


namespace sandy_friend_puppies_l1716_171620

theorem sandy_friend_puppies (original_puppies friend_puppies final_puppies : ℕ)
    (h1 : original_puppies = 8) (h2 : final_puppies = 12) :
    friend_puppies = final_puppies - original_puppies := by
    sorry

end sandy_friend_puppies_l1716_171620


namespace solve_quadratic_substitution_l1716_171653

theorem solve_quadratic_substitution (x : ℝ) : 
  (x^2 + x)^2 - 4*(x^2 + x) - 12 = 0 ↔ x = -3 ∨ x = 2 := 
by sorry

end solve_quadratic_substitution_l1716_171653


namespace triangle_is_isosceles_l1716_171677

noncomputable def is_isosceles_triangle (A B C a b c : ℝ) : Prop := ∃ (s : ℝ), a = s ∧ b = s

theorem triangle_is_isosceles 
  (A B C a b c : ℝ) 
  (h_sides_angles : a = c ∧ b = c) 
  (h_cos_eq : a * Real.cos B = b * Real.cos A) : 
  is_isosceles_triangle A B C a b c := 
by 
  sorry

end triangle_is_isosceles_l1716_171677


namespace smallest_a_b_sum_l1716_171630

theorem smallest_a_b_sum :
∀ (a b : ℕ), 
  (5 * a + 6 = 6 * b + 5) ∧ 
  (∀ d : ℕ, d < 10 → d < a) ∧ 
  (∀ d : ℕ, d < 10 → d < b) ∧ 
  (0 < a) ∧ 
  (0 < b) 
  → a + b = 13 :=
by
  sorry

end smallest_a_b_sum_l1716_171630


namespace correct_calculation_l1716_171626

-- Definitions of calculations based on conditions
def calc_A (a : ℝ) := a^2 + a^2 = a^4
def calc_B (a : ℝ) := (a^2)^3 = a^5
def calc_C (a : ℝ) := a + 2 = 2 * a
def calc_D (a b : ℝ) := (a * b)^3 = a^3 * b^3

-- Theorem stating that only the fourth calculation is correct
theorem correct_calculation (a b : ℝ) :
  ¬(calc_A a) ∧ ¬(calc_B a) ∧ ¬(calc_C a) ∧ calc_D a b :=
by sorry

end correct_calculation_l1716_171626


namespace simplify_expression_l1716_171648

theorem simplify_expression :
  (1 / (1 / (Real.sqrt 3 + 1) + 2 / (Real.sqrt 5 - 2))) = (Real.sqrt 3 - 2 * Real.sqrt 5 - 3) :=
by
  sorry

end simplify_expression_l1716_171648


namespace find_13_numbers_l1716_171601

theorem find_13_numbers :
  ∃ (a : Fin 13 → ℕ),
    (∀ i, a i % 21 = 0) ∧
    (∀ i j, i ≠ j → ¬(a i ∣ a j) ∧ ¬(a j ∣ a i)) ∧
    (∀ i j, i ≠ j → (a i ^ 5) % (a j ^ 4) = 0) :=
sorry

end find_13_numbers_l1716_171601


namespace weekly_goal_cans_l1716_171607

theorem weekly_goal_cans (c₁ c₂ c₃ c₄ c₅ : ℕ) (h₁ : c₁ = 20) (h₂ : c₂ = c₁ + 5) (h₃ : c₃ = c₂ + 5) 
  (h₄ : c₄ = c₃ + 5) (h₅ : c₅ = c₄ + 5) : 
  c₁ + c₂ + c₃ + c₄ + c₅ = 150 :=
by
  sorry

end weekly_goal_cans_l1716_171607


namespace find_A_l1716_171672

variable (U A CU_A : Set ℕ)

axiom U_is_universal : U = {1, 3, 5, 7, 9}
axiom CU_A_is_complement : CU_A = {5, 7}

theorem find_A (h1 : U = {1, 3, 5, 7, 9}) (h2 : CU_A = {5, 7}) : 
  A = {1, 3, 9} :=
by
  sorry

end find_A_l1716_171672


namespace ratio_area_triangles_to_square_l1716_171627

theorem ratio_area_triangles_to_square (x : ℝ) :
  let A := (0, x)
  let B := (x, x)
  let C := (x, 0)
  let D := (0, 0)
  let M := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  let N := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  let P := ((C.1 + D.1) / 2, (C.2 + D.2) / 2)
  let area_AMN := 1/2 * ((M.1 - A.1) * (N.2 - A.2) - (M.2 - A.2) * (N.1 - A.1))
  let area_MNP := 1/2 * ((N.1 - M.1) * (P.2 - M.2) - (N.2 - M.2) * (P.1 - M.1))
  let total_area_triangles := area_AMN + area_MNP
  let area_square := x * x
  total_area_triangles / area_square = 1/4 := 
by
  sorry

end ratio_area_triangles_to_square_l1716_171627


namespace segment_length_of_absolute_value_l1716_171616

theorem segment_length_of_absolute_value (x : ℝ) (h : abs (x - (27 : ℝ)^(1/3)) = 5) : 
  |8 - (-2)| = 10 :=
by
  sorry

end segment_length_of_absolute_value_l1716_171616


namespace unique_t_digit_l1716_171628

theorem unique_t_digit (t : ℕ) (ht : t < 100) (ht2 : 10 ≤ t) (h : 13 * t ≡ 42 [MOD 100]) : t = 34 := 
by
-- Proof is omitted
sorry

end unique_t_digit_l1716_171628


namespace max_value_xyz_l1716_171645

theorem max_value_xyz (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : 2 * x + 3 * x * y^2 + 2 * z = 36) : 
  x^2 * y^2 * z ≤ 144 :=
sorry

end max_value_xyz_l1716_171645


namespace Tiffany_bags_l1716_171666

theorem Tiffany_bags (x : ℕ) 
  (h1 : 8 = x + 1) : 
  x = 7 :=
by
  sorry

end Tiffany_bags_l1716_171666


namespace c_sub_a_eq_60_l1716_171623

theorem c_sub_a_eq_60 (a b c : ℝ) 
  (h1 : (a + b) / 2 = 30) 
  (h2 : (b + c) / 2 = 60) : 
  c - a = 60 := 
by 
  sorry

end c_sub_a_eq_60_l1716_171623


namespace library_table_count_l1716_171632

def base6_to_base10 (n : Nat) : Nat :=
  let d0 := (n % 10)
  let d1 := (n / 10) % 10
  let d2 := (n / 100) % 10
  d2 * 36 + d1 * 6 + d0 

theorem library_table_count (chairs people_per_table : Nat) (h1 : chairs = 231) (h2 : people_per_table = 3) :
    Nat.ceil ((base6_to_base10 chairs) / people_per_table) = 31 :=
by
  sorry

end library_table_count_l1716_171632


namespace solution_set_of_inequality_l1716_171664

noncomputable def f : ℝ → ℝ := sorry

axiom ax1 : ∀ (x1 x2 : ℝ), (0 < x1) → (0 < x2) → (x1 ≠ x2) → 
  (x1 * f x2 - x2 * f x1) / (x2 - x1) > 1

axiom ax2 : f 3 = 2

theorem solution_set_of_inequality :
  {x : ℝ | 0 < x ∧ f x < x - 1} = {x : ℝ | 0 < x ∧ x < 3} := by
  sorry

end solution_set_of_inequality_l1716_171664


namespace apple_distribution_l1716_171662

theorem apple_distribution (total_apples : ℝ)
  (time_anya time_varya time_sveta total_time : ℝ)
  (work_anya work_varya work_sveta : ℝ) :
  total_apples = 10 →
  time_anya = 20 →
  time_varya = 35 →
  time_sveta = 45 →
  total_time = (time_anya + time_varya + time_sveta) →
  work_anya = (total_apples * time_anya / total_time) →
  work_varya = (total_apples * time_varya / total_time) →
  work_sveta = (total_apples * time_sveta / total_time) →
  work_anya = 2 ∧ work_varya = 3.5 ∧ work_sveta = 4.5 := by
  sorry

end apple_distribution_l1716_171662


namespace inverse_function_fixed_point_l1716_171652

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the condition that graph of y = f(x-1) passes through the point (1, 2)
def passes_through (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  f (a - 1) = b

-- State the main theorem to prove
theorem inverse_function_fixed_point {f : ℝ → ℝ} (h : passes_through f 1 2) :
  ∃ x, x = 2 ∧ f x = 0 :=
sorry

end inverse_function_fixed_point_l1716_171652


namespace percentage_error_in_area_l1716_171690

theorem percentage_error_in_area (s : ℝ) (h : s ≠ 0) :
  let s' := 1.02 * s
  let A := s^2
  let A' := s'^2
  ((A' - A) / A) * 100 = 4.04 := by
  sorry

end percentage_error_in_area_l1716_171690


namespace probability_of_condition1_before_condition2_l1716_171634

-- Definitions for conditions
def condition1 (draw_counts : List ℕ) : Prop :=
  ∃ count ∈ draw_counts, count ≥ 3

def condition2 (draw_counts : List ℕ) : Prop :=
  ∀ count ∈ draw_counts, count ≥ 1

-- Probability function
def probability_condition1_before_condition2 : ℚ :=
  13 / 27

-- The proof statement
theorem probability_of_condition1_before_condition2 :
  (∃ draw_counts : List ℕ, (condition1 draw_counts) ∧  ¬(condition2 draw_counts)) →
  probability_condition1_before_condition2 = 13 / 27 :=
sorry

end probability_of_condition1_before_condition2_l1716_171634


namespace interest_calculation_years_l1716_171698

theorem interest_calculation_years (P r : ℝ) (diff : ℝ) (n : ℕ) 
  (hP : P = 3600) (hr : r = 0.10) (hdiff : diff = 36) 
  (h_eq : P * (1 + r)^n - P - (P * r * n) = diff) : n = 2 :=
sorry

end interest_calculation_years_l1716_171698


namespace power_function_value_at_4_l1716_171650

noncomputable def power_function (a : ℝ) (x : ℝ) : ℝ := x^a

theorem power_function_value_at_4 :
  ∃ a : ℝ, power_function a 2 = (Real.sqrt 2) / 2 → power_function a 4 = 1 / 2 :=
by
  sorry

end power_function_value_at_4_l1716_171650


namespace sum_of_squares_l1716_171657

theorem sum_of_squares (a b c : ℝ) (h1 : a * b + a * c + b * c = 131) (h2 : a + b + c = 22) : a^2 + b^2 + c^2 = 222 :=
by
  sorry

end sum_of_squares_l1716_171657


namespace value_of_expression_l1716_171681

theorem value_of_expression (x y : ℝ) (h₁ : x * y = -3) (h₂ : x + y = -4) :
  x^2 + 3 * x * y + y^2 = 13 :=
by
  sorry

end value_of_expression_l1716_171681


namespace rows_seating_8_people_l1716_171682

theorem rows_seating_8_people (x : ℕ) (h₁ : x ≡ 4 [MOD 7]) (h₂ : x ≤ 6) :
  x = 4 := by
  sorry

end rows_seating_8_people_l1716_171682


namespace total_houses_in_lincoln_county_l1716_171641

theorem total_houses_in_lincoln_county 
  (original_houses : ℕ) 
  (built_houses : ℕ) 
  (h_original : original_houses = 20817) 
  (h_built : built_houses = 97741) : 
  original_houses + built_houses = 118558 := 
by
  -- Sorry is used to skip the proof.
  sorry

end total_houses_in_lincoln_county_l1716_171641


namespace mixture_concentration_l1716_171691

-- Definitions reflecting the given conditions
def sol1_concentration : ℝ := 0.30
def sol1_volume : ℝ := 8

def sol2_concentration : ℝ := 0.50
def sol2_volume : ℝ := 5

def sol3_concentration : ℝ := 0.70
def sol3_volume : ℝ := 7

-- The proof problem stating that the resulting concentration is 49%
theorem mixture_concentration :
  (sol1_concentration * sol1_volume + sol2_concentration * sol2_volume + sol3_concentration * sol3_volume) /
  (sol1_volume + sol2_volume + sol3_volume) * 100 = 49 :=
by
  sorry

end mixture_concentration_l1716_171691


namespace fraction_eq_zero_has_solution_l1716_171621

theorem fraction_eq_zero_has_solution :
  ∀ (x : ℝ), x^2 - x - 2 = 0 ∧ x + 1 ≠ 0 → x = 2 :=
by
  sorry

end fraction_eq_zero_has_solution_l1716_171621


namespace co_presidents_included_probability_l1716_171674

-- Let the number of students in each club
def club_sizes : List ℕ := [6, 8, 9, 10]

-- Function to calculate binomial coefficient
def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Function to calculate probability for a given club size
noncomputable def co_president_probability (n : ℕ) : ℚ :=
  (choose (n - 2) 2 : ℚ) / (choose n 4)

-- List of probabilities for each club
noncomputable def probabilities : List ℚ :=
  List.map co_president_probability club_sizes

-- Aggregate total probability by averaging the individual probabilities
noncomputable def total_probability : ℚ :=
  (1 / 4 : ℚ) * probabilities.sum

-- The proof problem: proving the total probability equals 119/700
theorem co_presidents_included_probability :
  total_probability = 119 / 700 := by
  sorry

end co_presidents_included_probability_l1716_171674


namespace university_cost_per_box_l1716_171605

def box_volume (length width height : ℕ) : ℕ :=
  length * width * height

def num_boxes (total_volume box_volume : ℕ) : ℕ :=
  total_volume / box_volume

def cost_per_box (total_cost num_boxes : ℚ) : ℚ :=
  total_cost / num_boxes

theorem university_cost_per_box :
  let length := 20
  let width := 20
  let height := 15
  let total_volume := 3060000
  let total_cost := 459
  let box_vol := box_volume length width height
  let boxes := num_boxes total_volume box_vol
  cost_per_box total_cost boxes = 0.90 :=
by
  sorry

end university_cost_per_box_l1716_171605


namespace sum_of_three_squares_l1716_171680

theorem sum_of_three_squares (a b : ℝ)
  (h1 : 3 * a + 2 * b = 18)
  (h2 : 2 * a + 3 * b = 22) :
  3 * b = 18 :=
sorry

end sum_of_three_squares_l1716_171680


namespace find_breadth_of_rectangle_l1716_171671

noncomputable def breadth_of_rectangle (A : ℝ) (length_to_breadth_ratio : ℝ) (breadth : ℝ) : Prop :=
  A = length_to_breadth_ratio * breadth * breadth → breadth = 20

-- Now we can state the theorem.
theorem find_breadth_of_rectangle (A : ℝ) (length_to_breadth_ratio : ℝ) : breadth_of_rectangle A length_to_breadth_ratio 20 :=
by
  intros h
  sorry

end find_breadth_of_rectangle_l1716_171671


namespace vacant_seats_calculation_l1716_171614

noncomputable def seats_vacant (total_seats : ℕ) (percentage_filled : ℚ) : ℚ := 
  total_seats * (1 - percentage_filled)

theorem vacant_seats_calculation: 
  seats_vacant 600 0.45 = 330 := 
by 
    -- sorry to skip the proof.
    sorry

end vacant_seats_calculation_l1716_171614


namespace profit_percentage_B_l1716_171654

-- Definitions based on conditions:
def CP_A : ℝ := 150  -- Cost price for A
def profit_percentage_A : ℝ := 0.20  -- Profit percentage for A
def SP_C : ℝ := 225  -- Selling price for C

-- Lean statement for the problem:
theorem profit_percentage_B : (SP_C - (CP_A * (1 + profit_percentage_A))) / (CP_A * (1 + profit_percentage_A)) * 100 = 25 := 
by 
  sorry

end profit_percentage_B_l1716_171654


namespace negation_equiv_l1716_171686

theorem negation_equiv (x : ℝ) : ¬ (x^2 - 1 < 0) ↔ (x^2 - 1 ≥ 0) :=
by
  sorry

end negation_equiv_l1716_171686


namespace range_of_a_l1716_171647

variable (a : ℝ) (x : ℝ) (x₀ : ℝ)

def p (a : ℝ) : Prop := ∀ (x : ℝ), 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0
def q (a : ℝ) : Prop := ∃ (x₀ : ℝ), ∃ (x : ℝ), x + 2 * a * x₀ + 2 - a = 0

theorem range_of_a (h : p a ∧ q a) : a ≤ -2 ∨ a = 1 :=
sorry

end range_of_a_l1716_171647


namespace complex_number_solution_l1716_171633

theorem complex_number_solution (a b : ℝ) (z : ℂ) :
  z = a + b * I →
  (a - 2) ^ 2 + b ^ 2 = 25 →
  (a + 4) ^ 2 + b ^ 2 = 25 →
  a ^ 2 + (b - 2) ^ 2 = 25 →
  z = -1 - 4 * I :=
sorry

end complex_number_solution_l1716_171633
