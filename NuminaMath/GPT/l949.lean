import Mathlib

namespace determine_triangle_ratio_l949_94940

theorem determine_triangle_ratio (a d : ℝ) (h : (a + d) ^ 2 = (a - d) ^ 2 + a ^ 2) : a / d = 2 + Real.sqrt 3 :=
sorry

end determine_triangle_ratio_l949_94940


namespace recorded_instances_l949_94951

-- Define the conditions
def interval := 5
def total_time := 60 * 60  -- one hour in seconds

-- Define the theorem to prove the expected number of instances recorded
theorem recorded_instances : total_time / interval = 720 := by
  sorry

end recorded_instances_l949_94951


namespace ammonia_formation_l949_94998

theorem ammonia_formation (Li3N H2O LiOH NH3 : ℕ) (h₁ : Li3N = 1) (h₂ : H2O = 54) (h₃ : Li3N + 3 * H2O = 3 * LiOH + NH3) :
  NH3 = 1 :=
by
  sorry

end ammonia_formation_l949_94998


namespace count_four_digit_integers_l949_94967

theorem count_four_digit_integers :
    ∃! (a b c d : ℕ), 1 ≤ a ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧
    (10 * b + c)^2 = (10 * a + b) * (10 * c + d) := sorry

end count_four_digit_integers_l949_94967


namespace inequality_holds_l949_94910

noncomputable def f (x : ℝ) := x^2 + 2 * Real.cos x

theorem inequality_holds (x1 x2 : ℝ) : 
  f x1 > f x2 → x1 > |x2| := 
sorry

end inequality_holds_l949_94910


namespace find_radii_l949_94945

-- Definitions based on the problem conditions
def tangent_lengths (TP T'Q r r' PQ: ℝ) : Prop :=
  TP = 6 ∧ T'Q = 10 ∧ PQ = 16 ∧ r < r'

-- The main theorem to prove the radii are 15 and 5
theorem find_radii (TP T'Q r r' PQ: ℝ) 
  (h : tangent_lengths TP T'Q r r' PQ) :
  r = 15 ∧ r' = 5 :=
sorry

end find_radii_l949_94945


namespace abs_nonneg_position_l949_94986

theorem abs_nonneg_position (a : ℝ) : 0 ≤ |a| ∧ |a| ≥ 0 → (exists x : ℝ, x = |a| ∧ x ≥ 0) :=
by 
  sorry

end abs_nonneg_position_l949_94986


namespace find_m_value_l949_94973

-- Definitions of the hyperbola and its focus condition
def hyperbola_eq (m : ℝ) : Prop :=
  ∀ x y : ℝ, (x^2 / m) - (y^2 / (3 + m)) = 1

def focus_condition (m : ℝ) : Prop :=
  4 = (m) + (3 + m)

-- Theorem stating the value of m
theorem find_m_value (m : ℝ) : hyperbola_eq m → focus_condition m → m = 1 / 2 :=
by
  intros
  sorry

end find_m_value_l949_94973


namespace sum_abs_a_l949_94957

def S (n : ℕ) : ℤ := n^2 - 4 * n + 1

def a (n : ℕ) : ℤ :=
  if n = 1 then S 1
  else S n - S (n - 1)

theorem sum_abs_a :
  (|a 1| + |a 2| + |a 3| + |a 4| + |a 5| + 
   |a 6| + |a 7| + |a 8| + |a 9| + |a 10| = 67) :=
by
  sorry

end sum_abs_a_l949_94957


namespace Arman_hours_worked_l949_94966

/--
  Given:
  - LastWeekHours = 35
  - LastWeekRate = 10 (in dollars per hour)
  - IncreaseRate = 0.5 (in dollars per hour)
  - TotalEarnings = 770 (in dollars)
  Prove that:
  - ThisWeekHours = 40
-/
theorem Arman_hours_worked (LastWeekHours : ℕ) (LastWeekRate : ℕ) (IncreaseRate : ℕ) (TotalEarnings : ℕ)
  (h1 : LastWeekHours = 35)
  (h2 : LastWeekRate = 10)
  (h3 : IncreaseRate = 1/2)  -- because 0.5 as a fraction is 1/2
  (h4 : TotalEarnings = 770)
  : ∃ ThisWeekHours : ℕ, ThisWeekHours = 40 :=
by
  sorry

end Arman_hours_worked_l949_94966


namespace sequence_x_2022_l949_94929

theorem sequence_x_2022 :
  ∃ (x : ℕ → ℤ), x 1 = 1 ∧ x 2 = 1 ∧ x 3 = -1 ∧
  (∀ n, 4 ≤ n → x n = x (n-1) * x (n-3)) ∧ x 2022 = 1 := by
  sorry

end sequence_x_2022_l949_94929


namespace central_angle_l949_94960

variable (O : Type)
variable (A B C : O)
variable (angle_ABC : ℝ) 

theorem central_angle (h : angle_ABC = 50) : 2 * angle_ABC = 100 := by
  sorry

end central_angle_l949_94960


namespace solve_a_l949_94956

def custom_op (a b : ℝ) : ℝ := 2 * a - b^2

theorem solve_a :
  ∃ a : ℝ, custom_op a 7 = -20 ∧ a = 29 / 2 :=
by
  sorry

end solve_a_l949_94956


namespace expression_value_l949_94978

theorem expression_value :
  2 - (-3) - 4 - (-5) - 6 - (-7) * 2 = 14 :=
by sorry

end expression_value_l949_94978


namespace lcm_36_105_l949_94972

theorem lcm_36_105 : Nat.lcm 36 105 = 1260 := by
  sorry

end lcm_36_105_l949_94972


namespace complex_vector_PQ_l949_94907

theorem complex_vector_PQ (P Q : ℂ) (hP : P = 3 + 1 * I) (hQ : Q = 2 + 3 * I) : 
  (Q - P) = -1 + 2 * I :=
by sorry

end complex_vector_PQ_l949_94907


namespace sequence_formula_l949_94921

theorem sequence_formula (a : ℕ → ℚ) (h1 : a 2 = 3 / 2) (h2 : a 3 = 7 / 3) 
  (h3 : ∀ n : ℕ, ∃ r : ℚ, (∀ m : ℕ, m ≥ 2 → (m * a m + 1) / (n * a n + 1) = r ^ (m - n))) :
  a n = (2^n - 1) / n := 
sorry

end sequence_formula_l949_94921


namespace percent_notebooks_staplers_clips_l949_94918

def percent_not_special (n s c: ℝ) (h_n: n = 25) (h_s: s = 20) (h_c: c = 30) : ℝ :=
  100 - (n + s + c)

theorem percent_notebooks_staplers_clips (n s c: ℝ) (h_n: n = 25) (h_s: s = 20) (h_c: c = 30) :
  percent_not_special n s c h_n h_s h_c = 25 :=
by
  unfold percent_not_special
  rw [h_n, h_s, h_c]
  norm_num

end percent_notebooks_staplers_clips_l949_94918


namespace max_soap_boxes_in_carton_l949_94963

-- Define the measurements of the carton
def L_carton := 25
def W_carton := 42
def H_carton := 60

-- Define the measurements of the soap box
def L_soap_box := 7
def W_soap_box := 12
def H_soap_box := 5

-- Calculate the volume of the carton
def V_carton := L_carton * W_carton * H_carton

-- Calculate the volume of the soap box
def V_soap_box := L_soap_box * W_soap_box * H_soap_box

-- Define the number of soap boxes that can fit in the carton
def number_of_soap_boxes := V_carton / V_soap_box

-- Prove that the number of soap boxes that can fit in the carton is 150
theorem max_soap_boxes_in_carton : number_of_soap_boxes = 150 :=
by
  -- Placeholder for the proof
  sorry

end max_soap_boxes_in_carton_l949_94963


namespace probability_odd_divisor_of_15_factorial_l949_94949

-- Define the factorial function
def fact : ℕ → ℕ
  | 0 => 1
  | (n+1) => (n+1) * fact n

-- Probability function for choosing an odd divisor
noncomputable def probability_odd_divisor (n : ℕ) : ℚ :=
  let prime_factors := [(2, 11), (3, 6), (5, 3), (7, 2), (11, 1), (13, 1)]
  let total_factors := prime_factors.foldr (λ p acc => (p.2 + 1) * acc) 1
  let odd_factors := ((prime_factors.filter (λ p => p.1 ≠ 2)).foldr (λ p acc => (p.2 + 1) * acc) 1)
  (odd_factors : ℚ) / (total_factors : ℚ)

-- Statement to prove the probability of an odd divisor
theorem probability_odd_divisor_of_15_factorial :
  probability_odd_divisor 15 = 1 / 12 :=
by
  -- Proof goes here, which is omitted as per the instructions
  sorry

end probability_odd_divisor_of_15_factorial_l949_94949


namespace oak_total_after_planting_l949_94965

-- Let oak_current represent the current number of oak trees in the park.
def oak_current : ℕ := 9

-- Let oak_new represent the number of new oak trees being planted.
def oak_new : ℕ := 2

-- The problem is to prove the total number of oak trees after planting equals 11
theorem oak_total_after_planting : oak_current + oak_new = 11 :=
by
  sorry

end oak_total_after_planting_l949_94965


namespace total_eggs_emily_collected_l949_94997

theorem total_eggs_emily_collected :
  let number_of_baskets := 303
  let eggs_per_basket := 28
  number_of_baskets * eggs_per_basket = 8484 :=
by
  let number_of_baskets := 303
  let eggs_per_basket := 28
  sorry -- Proof to be provided

end total_eggs_emily_collected_l949_94997


namespace fred_walking_speed_l949_94958

/-- 
Fred and Sam are standing 55 miles apart and they start walking in a straight line toward each other
at the same time. Fred walks at a certain speed and Sam walks at a constant speed of 5 miles per hour.
Sam has walked 25 miles when they meet.
-/
theorem fred_walking_speed
  (initial_distance : ℕ) 
  (sam_speed : ℕ)
  (sam_distance : ℕ) 
  (meeting_time : ℕ)
  (fred_distance : ℕ) 
  (fred_speed : ℕ)
  (h_initial_distance : initial_distance = 55)
  (h_sam_speed : sam_speed = 5)
  (h_sam_distance : sam_distance = 25)
  (h_meeting_time : meeting_time = 5)
  (h_fred_distance : fred_distance = 30)
  (h_fred_speed : fred_speed = 6)
  : fred_speed = fred_distance / meeting_time :=
by sorry

end fred_walking_speed_l949_94958


namespace tens_place_of_8_pow_1234_l949_94942

theorem tens_place_of_8_pow_1234 : (8^1234 / 10) % 10 = 0 := by
  sorry

end tens_place_of_8_pow_1234_l949_94942


namespace compute_P_2_4_8_l949_94974

noncomputable def P : ℝ → ℝ → ℝ → ℝ := sorry

axiom homogeneity (x y z k : ℝ) : P (k * x) (k * y) (k * z) = (k ^ 4) * P x y z

axiom symmetry (a b c : ℝ) : P a b c = P b c a

axiom zero_cond (a b : ℝ) : P a a b = 0

axiom initial_cond : P 1 2 3 = 1

theorem compute_P_2_4_8 : P 2 4 8 = 56 := sorry

end compute_P_2_4_8_l949_94974


namespace dan_present_age_l949_94904

theorem dan_present_age : ∃ x : ℕ, (x + 18 = 8 * (x - 3)) ∧ x = 6 :=
by
  -- We skip the proof steps
  sorry

end dan_present_age_l949_94904


namespace hannah_spent_on_dessert_l949_94987

theorem hannah_spent_on_dessert
  (initial_money : ℕ)
  (money_left : ℕ)
  (half_spent_on_rides : ℕ)
  (total_spent : ℕ)
  (spent_on_dessert : ℕ)
  (H1 : initial_money = 30)
  (H2 : money_left = 10)
  (H3 : half_spent_on_rides = initial_money / 2)
  (H4 : total_spent = initial_money - money_left)
  (H5 : spent_on_dessert = total_spent - half_spent_on_rides) : spent_on_dessert = 5 :=
by
  sorry

end hannah_spent_on_dessert_l949_94987


namespace find_x_floor_mul_eq_100_l949_94938

theorem find_x_floor_mul_eq_100 (x : ℝ) (h1 : 0 < x) (h2 : (⌊x⌋ : ℝ) * x = 100) : x = 10 :=
by
  sorry

end find_x_floor_mul_eq_100_l949_94938


namespace arithmetic_sequence_solution_l949_94930

theorem arithmetic_sequence_solution
  (a b c : ℤ)
  (h1 : a + 1 = b - a)
  (h2 : b - a = c - b)
  (h3 : c - b = -9 - c) :
  b = -5 ∧ a * c = 21 :=
by sorry

end arithmetic_sequence_solution_l949_94930


namespace problem_U_l949_94970

theorem problem_U :
  ( (1 : ℝ) / (4 - Real.sqrt 15) - (1 / (Real.sqrt 15 - Real.sqrt 14))
  + (1 / (Real.sqrt 14 - 3)) - (1 / (3 - Real.sqrt 12))
  + (1 / (Real.sqrt 12 - Real.sqrt 11)) ) = 10 + Real.sqrt 11 :=
by
  sorry

end problem_U_l949_94970


namespace correct_calculation_l949_94979

theorem correct_calculation :
  (2 * Real.sqrt 3 + 3 * Real.sqrt 2 ≠ 5) ∧
  (Real.sqrt 8 / Real.sqrt 2 = 2) ∧
  (5 * Real.sqrt 3 * 5 * Real.sqrt 2 ≠ 5 * Real.sqrt 6) ∧
  (Real.sqrt (4 + 1/2) ≠ 2 * Real.sqrt (1/2)) :=
by
  sorry

end correct_calculation_l949_94979


namespace main_theorem_l949_94932

variables {m n : ℕ} {x : ℝ}
variables {a : ℕ → ℕ}
noncomputable def relatively_prime (a : ℕ → ℕ) (n : ℕ) : Prop :=
∀ i j, i ≠ j → i < n → j < n → Nat.gcd (a i) (a j) = 1

noncomputable def distinct (a : ℕ → ℕ) (n : ℕ) : Prop :=
∀ i j, i ≠ j → i < n → j < n → a i ≠ a j

theorem main_theorem (hm : 1 < m) (hn : 1 < n) (hge : m ≥ n)
  (hrel_prime : relatively_prime a n)
  (hdistinct : distinct a n)
  (hbound : ∀ i, i < n → a i ≤ m)
  : ∃ i, i < n ∧ ‖a i * x‖ ≥ (2 / (m * (m + 1))) * ‖x‖ := 
sorry

end main_theorem_l949_94932


namespace painted_cube_ways_l949_94913

theorem painted_cube_ways (b r g : ℕ) (cubes : ℕ) : 
  b = 1 → r = 2 → g = 3 → cubes = 3 := 
by
  intros
  sorry

end painted_cube_ways_l949_94913


namespace probability_boarding_251_l949_94925

theorem probability_boarding_251 :
  let interval_152 := 5
  let interval_251 := 7
  let total_events := interval_152 * interval_251
  let favorable_events := (interval_152 * interval_152) / 2
  (favorable_events / total_events : ℚ) = 5 / 14 :=
by 
  sorry

end probability_boarding_251_l949_94925


namespace main_theorem_l949_94980

variable (a : ℝ)

def M : Set ℝ := {x | x > 1 / 2 ∧ x < 1} ∪ {x | x > 1}
def N : Set ℝ := {x | x > 0 ∧ x ≤ 1 / 2}

theorem main_theorem : M ∩ N = ∅ :=
by
  sorry

end main_theorem_l949_94980


namespace part1_part2_l949_94992

-- Definition of Set A
def A : Set ℝ := { x | 2 ≤ x ∧ x ≤ 6 }

-- Definition of Set B
def B : Set ℝ := { x | x ≥ 3 }

-- The Complement of the Intersection of A and B
def C_R (S : Set ℝ) : Set ℝ := { x | ¬ (x ∈ S) }

-- Set C
def C (a : ℝ) : Set ℝ := { x | x ≤ a }

-- Lean statement for part 1
theorem part1 : C_R (A ∩ B) = { x | x < 3 ∨ x > 6 } :=
by sorry

-- Lean statement for part 2
theorem part2 (a : ℝ) (hA_C : A ⊆ C a) : a ≥ 6 :=
by sorry

end part1_part2_l949_94992


namespace seq_ratio_l949_94928

theorem seq_ratio (a : ℕ → ℝ) (h₁ : a 1 = 5) (h₂ : ∀ n, a n * a (n + 1) = 2^n) : 
  a 7 / a 3 = 4 := 
by 
  sorry

end seq_ratio_l949_94928


namespace tan_neg_5pi_over_4_l949_94926

theorem tan_neg_5pi_over_4 : Real.tan (-5 * Real.pi / 4) = -1 :=
by 
  sorry

end tan_neg_5pi_over_4_l949_94926


namespace remaining_students_l949_94903

def students_remaining (n1 n2 n_leaving1 n_leaving2 : Nat) : Nat :=
  (n1 * 4 - n_leaving1) + (n2 * 2 - n_leaving2)

theorem remaining_students :
  students_remaining 15 18 8 5 = 83 := 
by
  sorry

end remaining_students_l949_94903


namespace james_total_oop_correct_l949_94924

-- Define the costs and insurance coverage percentages as given conditions.
def cost_consultation : ℝ := 300
def coverage_consultation : ℝ := 0.80

def cost_xray : ℝ := 150
def coverage_xray : ℝ := 0.70

def cost_prescription : ℝ := 75
def coverage_prescription : ℝ := 0.50

def cost_therapy : ℝ := 120
def coverage_therapy : ℝ := 0.60

-- Define the out-of-pocket calculation for each service
def oop_consultation := cost_consultation * (1 - coverage_consultation)
def oop_xray := cost_xray * (1 - coverage_xray)
def oop_prescription := cost_prescription * (1 - coverage_prescription)
def oop_therapy := cost_therapy * (1 - coverage_therapy)

-- Define the total out-of-pocket cost
def total_oop : ℝ := oop_consultation + oop_xray + oop_prescription + oop_therapy

-- Proof statement
theorem james_total_oop_correct : total_oop = 190.50 := by
  sorry

end james_total_oop_correct_l949_94924


namespace intersection_P_Q_l949_94911

section set_intersection

variable (x : ℝ)

def P := { x : ℝ | x ≤ 1 }
def Q := { x : ℝ | -1 ≤ x ∧ x ≤ 2 }

theorem intersection_P_Q : { x | x ∈ P ∧ x ∈ Q } = { x | -1 ≤ x ∧ x ≤ 1 } :=
by
  sorry

end set_intersection

end intersection_P_Q_l949_94911


namespace solve_for_x_l949_94902

def determinant_2x2 (a b c d : ℝ) : ℝ :=
  a * d - b * c

theorem solve_for_x (x : ℝ) (h : determinant_2x2 (x+1) (x+2) (x-3) (x-1) = 2023) :
  x = 2018 :=
by {
  sorry
}

end solve_for_x_l949_94902


namespace total_number_of_candles_l949_94931

theorem total_number_of_candles
  (candles_bedroom : ℕ)
  (candles_living_room : ℕ)
  (candles_donovan : ℕ)
  (h1 : candles_bedroom = 20)
  (h2 : candles_bedroom = 2 * candles_living_room)
  (h3 : candles_donovan = 20) :
  candles_bedroom + candles_living_room + candles_donovan = 50 :=
by
  sorry

end total_number_of_candles_l949_94931


namespace problem_statement_l949_94914

theorem problem_statement (a b c d : ℝ) 
  (hab : a ≤ b)
  (hbc : b ≤ c)
  (hcd : c ≤ d)
  (hsum : a + b + c + d = 0)
  (hinv_sum : 1/a + 1/b + 1/c + 1/d = 0) :
  a + d = 0 :=
sorry

end problem_statement_l949_94914


namespace first_player_winning_strategy_l949_94990
noncomputable def golden_ratio := (1 + Real.sqrt 5) / 2

theorem first_player_winning_strategy (x1 y1 : ℕ)
    (h1 : x1 > 0) (h2 : y1 > 0) :
    (x1 / y1 = 1) ∨ 
    (x1 / y1 > golden_ratio) ∨ 
    (x1 / y1 < 1 / golden_ratio) :=
sorry

end first_player_winning_strategy_l949_94990


namespace find_constants_l949_94953

theorem find_constants (P Q R : ℚ) :
  (∀ x : ℚ, x ≠ 1 ∧ x ≠ 4 ∧ x ≠ -2 →
    (x^2 + x - 8) / ((x - 1) * (x - 4) * (x + 2)) = 
    P / (x - 1) + Q / (x - 4) + R / (x + 2))
  → (P = 2 / 3 ∧ Q = 8 / 9 ∧ R = -5 / 9) :=
by
  sorry

end find_constants_l949_94953


namespace sum_of_roots_equation_l949_94920

noncomputable def sum_of_roots (a b c : ℝ) : ℝ :=
  (-b) / a

theorem sum_of_roots_equation :
  let a := 3
  let b := -15
  let c := 20
  sum_of_roots a b c = 5 := 
  by {
    sorry
  }

end sum_of_roots_equation_l949_94920


namespace delta_four_equal_zero_l949_94943

-- Define the sequence u_n
def u (n : ℕ) : ℤ := n^3 + n

-- Define the ∆ operator
def delta1 (u : ℕ → ℤ) (n : ℕ) : ℤ := u (n + 1) - u n

def delta (k : ℕ) (u : ℕ → ℤ) : ℕ → ℤ :=
  match k with
  | 0   => u
  | k+1 => delta1 (delta k u)

-- The theorem statement
theorem delta_four_equal_zero (n : ℕ) : delta 4 u n = 0 :=
by sorry

end delta_four_equal_zero_l949_94943


namespace find_xy_l949_94933

theorem find_xy (x y : ℝ) (h : x * (x + 2 * y) = x^2 + 10) : x * y = 5 :=
by
  sorry

end find_xy_l949_94933


namespace kelly_peanut_weight_l949_94952

-- Define the total weight of snacks and the weight of raisins
def total_snacks_weight : ℝ := 0.5
def raisins_weight : ℝ := 0.4

-- Define the weight of peanuts as the remaining part
def peanuts_weight : ℝ := total_snacks_weight - raisins_weight

-- Theorem stating Kelly bought 0.1 pounds of peanuts
theorem kelly_peanut_weight : peanuts_weight = 0.1 :=
by
  -- proof would go here
  sorry

end kelly_peanut_weight_l949_94952


namespace range_of_m_l949_94981

theorem range_of_m (x m : ℝ) (h1 : -1 ≤ x ∧ x ≤ 1) (h2 : |x - m| ≤ 2) : -1 ≤ m ∧ m ≤ 1 :=
sorry

end range_of_m_l949_94981


namespace solution_for_x2_l949_94971

def eq1 (x : ℝ) := 2 * x = 6
def eq2 (x : ℝ) := x + 2 = 0
def eq3 (x : ℝ) := x - 5 = 3
def eq4 (x : ℝ) := 3 * x - 6 = 0

theorem solution_for_x2 : ∀ x : ℝ, x = 2 → ¬eq1 x ∧ ¬eq2 x ∧ ¬eq3 x ∧ eq4 x :=
by 
  sorry

end solution_for_x2_l949_94971


namespace compound_bar_chart_must_clearly_indicate_legend_l949_94988

-- Definitions of the conditions
structure CompoundBarChart where
  distinguishes_two_quantities : Bool
  uses_bars_of_different_colors : Bool

-- The theorem stating that a compound bar chart must clearly indicate the legend
theorem compound_bar_chart_must_clearly_indicate_legend 
  (chart : CompoundBarChart)
  (distinguishes_quantities : chart.distinguishes_two_quantities = true)
  (uses_colors : chart.uses_bars_of_different_colors = true) :
  ∃ legend : String, legend ≠ "" := by
  sorry

end compound_bar_chart_must_clearly_indicate_legend_l949_94988


namespace student_ticket_cost_l949_94917

theorem student_ticket_cost 
  (total_tickets_sold : ℕ) 
  (total_revenue : ℕ) 
  (nonstudent_ticket_cost : ℕ) 
  (student_tickets_sold : ℕ) 
  (cost_per_student_ticket : ℕ) 
  (nonstudent_tickets_sold : ℕ) 
  (H1 : total_tickets_sold = 821) 
  (H2 : total_revenue = 1933)
  (H3 : nonstudent_ticket_cost = 3)
  (H4 : student_tickets_sold = 530) 
  (H5 : nonstudent_tickets_sold = total_tickets_sold - student_tickets_sold)
  (H6 : 530 * cost_per_student_ticket + nonstudent_tickets_sold * 3 = 1933) : 
  cost_per_student_ticket = 2 := 
by
  sorry

end student_ticket_cost_l949_94917


namespace worker_payment_l949_94922

theorem worker_payment (x : ℕ) (daily_return : ℕ) (non_working_days : ℕ) (total_days : ℕ) 
    (net_earning : ℕ) 
    (H1 : daily_return = 25) 
    (H2 : non_working_days = 24) 
    (H3 : total_days = 30) 
    (H4 : net_earning = 0) 
    (H5 : ∀ w, net_earning = w * x - non_working_days * daily_return) : 
  x = 100 :=
by
  sorry

end worker_payment_l949_94922


namespace max_non_cyclic_handshakes_l949_94995

theorem max_non_cyclic_handshakes (n : ℕ) (h : n = 18) : 
  (n * (n - 1)) / 2 = 153 := by
  sorry

end max_non_cyclic_handshakes_l949_94995


namespace age_double_condition_l949_94969

theorem age_double_condition (S M X : ℕ) (h1 : S = 44) (h2 : M = S + 46) (h3 : M + X = 2 * (S + X)) : X = 2 :=
by
  sorry

end age_double_condition_l949_94969


namespace martin_travel_time_l949_94916

-- Definitions based on the conditions
def distance : ℕ := 12
def speed : ℕ := 2

-- Statement of the problem to be proven
theorem martin_travel_time : (distance / speed) = 6 := by sorry

end martin_travel_time_l949_94916


namespace first_day_revenue_l949_94946

theorem first_day_revenue :
  ∀ (S : ℕ), (12 * S + 90 = 246) → (4 * S + 3 * 9 = 79) :=
by
  intros S h1
  sorry

end first_day_revenue_l949_94946


namespace major_axis_of_ellipse_l949_94999

-- Define the given ellipse equation
def ellipse_eq (x y : ℝ) : Prop := 4 * x^2 + y^2 = 16

-- Define the length of the major axis
def major_axis_length : ℝ := 8

-- The theorem to prove
theorem major_axis_of_ellipse : 
  (∀ x y : ℝ, ellipse_eq x y) → major_axis_length = 8 :=
by
  sorry

end major_axis_of_ellipse_l949_94999


namespace angle_in_fourth_quadrant_l949_94977

theorem angle_in_fourth_quadrant (θ : ℝ) (h : θ = -1445) : (θ % 360) > 270 ∧ (θ % 360) < 360 :=
by
  sorry

end angle_in_fourth_quadrant_l949_94977


namespace adele_age_fraction_l949_94912

theorem adele_age_fraction 
  (jackson_age : ℕ) 
  (mandy_age : ℕ) 
  (adele_age_fraction : ℚ) 
  (total_age_10_years : ℕ)
  (H1 : jackson_age = 20)
  (H2 : mandy_age = jackson_age + 10)
  (H3 : total_age_10_years = (jackson_age + 10) + (mandy_age + 10) + (jackson_age * adele_age_fraction + 10))
  (H4 : total_age_10_years = 95) : 
  adele_age_fraction = 3 / 4 := 
sorry

end adele_age_fraction_l949_94912


namespace expand_product_l949_94975

def poly1 (x : ℝ) := 4 * x + 2
def poly2 (x : ℝ) := 3 * x - 1
def poly3 (x : ℝ) := x + 6

theorem expand_product (x : ℝ) :
  (poly1 x) * (poly2 x) * (poly3 x) = 12 * x^3 + 74 * x^2 + 10 * x - 12 :=
by
  sorry

end expand_product_l949_94975


namespace platform_protection_l949_94935

noncomputable def max_distance (r : ℝ) (n : ℕ) : ℝ :=
  if n > 2 then r / (Real.sin (180.0 / n)) else 0

noncomputable def coverage_ring_area (r : ℝ) (w : ℝ) : ℝ :=
  let inner_radius := r * (Real.sin 20.0)
  let outer_radius := inner_radius + w
  Real.pi * (outer_radius^2 - inner_radius^2)

theorem platform_protection :
  let r := 61
  let w := 22
  let n := 9
  max_distance r n = 60 / Real.sin 20.0 ∧
  coverage_ring_area r w = 2640 * Real.pi / Real.tan 20.0 := by
  sorry

end platform_protection_l949_94935


namespace greene_family_total_spent_l949_94955

def adm_cost : ℕ := 45

def food_cost : ℕ := adm_cost - 13

def total_cost : ℕ := adm_cost + food_cost

theorem greene_family_total_spent : total_cost = 77 := 
by 
  sorry

end greene_family_total_spent_l949_94955


namespace range_of_a_l949_94983

/-- Given a fixed point A(a, 3) is outside the circle x^2 + y^2 - 2ax - 3y + a^2 + a = 0,
we want to show that the range of values for a is (0, 9/4). -/
theorem range_of_a (a : ℝ) :
  (∃ (A : ℝ × ℝ), A = (a, 3) ∧ ¬(∃ (x y : ℝ), x^2 + y^2 - 2*a*x - 3*y + a^2 + a = 0))
  ↔ (0 < a ∧ a < 9/4) :=
sorry

end range_of_a_l949_94983


namespace expected_value_is_750_l949_94923

def winnings (roll : ℕ) : ℕ :=
  if roll % 2 = 0 then 3 * roll else 0

def expected_value : ℚ :=
  (winnings 2 / 8) + (winnings 4 / 8) + (winnings 6 / 8) + (winnings 8 / 8)

theorem expected_value_is_750 : expected_value = 7.5 := by
  sorry

end expected_value_is_750_l949_94923


namespace area_of_rectangle_l949_94959

theorem area_of_rectangle (side radius length breadth : ℕ) (h1 : side^2 = 784) (h2 : radius = side) (h3 : length = radius / 4) (h4 : breadth = 5) : length * breadth = 35 :=
by
  -- proof to be filled here
  sorry

end area_of_rectangle_l949_94959


namespace product_of_successive_numbers_l949_94993

-- Given conditions
def n : ℝ := 51.49757275833493

-- Proof statement
theorem product_of_successive_numbers : n * (n + 1) = 2703.0000000000005 :=
by
  -- Proof would be supplied here
  sorry

end product_of_successive_numbers_l949_94993


namespace fg_of_3_l949_94989

def f (x : ℝ) : ℝ := x - 2
def g (x : ℝ) : ℝ := x^2 - 3 * x

theorem fg_of_3 : f (g 3) = -2 := by
  sorry

end fg_of_3_l949_94989


namespace weight_of_fresh_grapes_is_40_l949_94982

-- Define the weight of fresh grapes and dried grapes
variables (F D : ℝ)

-- Fresh grapes contain 90% water by weight, so 10% is non-water
def fresh_grapes_non_water_content (F : ℝ) : ℝ := 0.10 * F

-- Dried grapes contain 20% water by weight, so 80% is non-water
def dried_grapes_non_water_content (D : ℝ) : ℝ := 0.80 * D

-- Given condition: weight of dried grapes is 5 kg
def weight_of_dried_grapes : ℝ := 5

-- The main theorem to prove
theorem weight_of_fresh_grapes_is_40 :
  fresh_grapes_non_water_content F = dried_grapes_non_water_content weight_of_dried_grapes →
  F = 40 := 
by
  sorry

end weight_of_fresh_grapes_is_40_l949_94982


namespace largest_divisible_l949_94927

theorem largest_divisible (n : ℕ) (h1 : n > 0) (h2 : (n^3 + 200) % (n - 8) = 0) : n = 5376 :=
by
  sorry

end largest_divisible_l949_94927


namespace weight_difference_l949_94968

variables (W_A W_B W_C W_D W_E : ℝ)

def condition1 : Prop := (W_A + W_B + W_C) / 3 = 84
def condition2 : Prop := (W_A + W_B + W_C + W_D) / 4 = 80
def condition3 : Prop := (W_B + W_C + W_D + W_E) / 4 = 79
def condition4 : Prop := W_A = 80

theorem weight_difference (h1 : condition1 W_A W_B W_C) 
                          (h2 : condition2 W_A W_B W_C W_D) 
                          (h3 : condition3 W_B W_C W_D W_E) 
                          (h4 : condition4 W_A) : 
                          W_E - W_D = 8 :=
by
  sorry

end weight_difference_l949_94968


namespace hours_learning_english_each_day_l949_94950

theorem hours_learning_english_each_day (E : ℕ) 
  (h_chinese_each_day : ∀ (d : ℕ), d = 7) 
  (days : ℕ) 
  (h_total_days : days = 5) 
  (h_total_hours : ∀ (t : ℕ), t = 65) 
  (total_learning_time : 5 * (E + 7) = 65) :
  E = 6 :=
by
  sorry

end hours_learning_english_each_day_l949_94950


namespace no_natural_m_n_prime_l949_94919

theorem no_natural_m_n_prime (m n : ℕ) : ¬Prime (n^2 + 2018 * m * n + 2019 * m + n - 2019 * m^2) :=
by
  sorry

end no_natural_m_n_prime_l949_94919


namespace third_number_is_507_l949_94939

theorem third_number_is_507 (x : ℕ) 
  (h1 : (55 + 48 + x + 2 + 684 + 42) / 6 = 223) : 
  x = 507 := by
  sorry

end third_number_is_507_l949_94939


namespace LCM_of_two_numbers_l949_94947

theorem LCM_of_two_numbers (a b : ℕ) (h1 : Nat.gcd a b = 14) (h2 : a * b = 2562) : Nat.lcm a b = 183 :=
by
  sorry

end LCM_of_two_numbers_l949_94947


namespace probability_of_same_suit_or_number_but_not_both_l949_94941

def same_suit_or_number_but_not_both : Prop :=
  let total_outcomes := 52 * 52
  let prob_same_suit := 12 / 51
  let prob_same_number := 3 / 51
  let prob_same_suit_and_number := 1 / 51
  (prob_same_suit + prob_same_number - 2 * prob_same_suit_and_number) = 15 / 52

theorem probability_of_same_suit_or_number_but_not_both :
  same_suit_or_number_but_not_both :=
by sorry

end probability_of_same_suit_or_number_but_not_both_l949_94941


namespace trapezium_other_side_length_l949_94905

theorem trapezium_other_side_length (side1 : ℝ) (distance : ℝ) (area : ℝ) (side2 : ℝ) :
  side1 = 20 ∧ distance = 15 ∧ area = 285 → side2 = 18 :=
by
  intro h
  sorry

end trapezium_other_side_length_l949_94905


namespace grasshopper_cannot_move_3_cm_grasshopper_can_move_2_cm_grasshopper_can_move_1234_cm_l949_94961

def can_jump (x : Int) : Prop :=
  ∃ (k m : Int), x = k * 36 + m * 14

theorem grasshopper_cannot_move_3_cm :
  ¬ can_jump 3 :=
by
  sorry

theorem grasshopper_can_move_2_cm :
  can_jump 2 :=
by
  sorry

theorem grasshopper_can_move_1234_cm :
  can_jump 1234 :=
by
  sorry

end grasshopper_cannot_move_3_cm_grasshopper_can_move_2_cm_grasshopper_can_move_1234_cm_l949_94961


namespace unique_solution_of_quadratic_l949_94908

theorem unique_solution_of_quadratic (b c x : ℝ) (h_eqn : 9 * x^2 + b * x + c = 0) (h_one_solution : ∀ y: ℝ, 9 * y^2 + b * y + c = 0 → y = x) (h_b2_4c : b^2 = 4 * c) : 
  x = -b / 18 := 
by 
  sorry

end unique_solution_of_quadratic_l949_94908


namespace interest_rate_B_to_C_l949_94901

theorem interest_rate_B_to_C
  (P : ℕ)                -- Principal amount
  (r_A : ℚ)              -- Interest rate A charges B per annum
  (t : ℕ)                -- Time period in years
  (gain_B : ℚ)           -- Gain of B in 3 years
  (H_P : P = 3500)
  (H_r_A : r_A = 0.10)
  (H_t : t = 3)
  (H_gain_B : gain_B = 315) :
  ∃ R : ℚ, R = 0.13 := 
by
  sorry

end interest_rate_B_to_C_l949_94901


namespace clothing_factory_exceeded_tasks_l949_94996

theorem clothing_factory_exceeded_tasks :
  let first_half := (2 : ℚ) / 3
  let second_half := (3 : ℚ) / 5
  first_half + second_half - 1 = (4 : ℚ) / 15 :=
by
  sorry

end clothing_factory_exceeded_tasks_l949_94996


namespace Janet_earnings_l949_94900

/--
Janet works as an exterminator and also sells molten metal casts of fire ant nests on the Internet.
She gets paid $70 an hour for exterminator work and makes $20/pound on her ant nest sculptures.
Given that she does 20 hours of exterminator work and sells a 5-pound sculpture and a 7-pound sculpture,
prove that Janet's total earnings are $1640.
-/
theorem Janet_earnings :
  let hourly_rate_exterminator := 70
  let hours_worked := 20
  let rate_per_pound := 20
  let sculpture_one_weight := 5
  let sculpture_two_weight := 7

  let exterminator_earnings := hourly_rate_exterminator * hours_worked
  let total_sculpture_weight := sculpture_one_weight + sculpture_two_weight
  let sculpture_earnings := rate_per_pound * total_sculpture_weight

  let total_earnings := exterminator_earnings + sculpture_earnings
  total_earnings = 1640 := 
by
  sorry

end Janet_earnings_l949_94900


namespace complex_z_solution_l949_94909

theorem complex_z_solution (z : ℂ) (i : ℂ) (h : i * z = 1 - i) (hi : i * i = -1) : z = -1 - i :=
by sorry

end complex_z_solution_l949_94909


namespace evaluate_expression_l949_94915

theorem evaluate_expression :
  (2:ℝ) ^ ((0:ℝ) ^ (Real.sin (Real.pi / 2)) ^ 2) + ((3:ℝ) ^ 0) ^ 1 ^ 4 = 2 := by
  -- Given conditions
  have h1 : Real.sin (Real.pi / 2) = 1 := by sorry
  have h2 : (3:ℝ) ^ 0 = 1 := by sorry
  have h3 : (0:ℝ) ^ 1 = 0 := by sorry
  -- Proof omitted
  sorry

end evaluate_expression_l949_94915


namespace same_grade_percentage_l949_94954

theorem same_grade_percentage (total_students: ℕ)
  (a_students: ℕ) (b_students: ℕ) (c_students: ℕ) (d_students: ℕ)
  (total: total_students = 30)
  (a: a_students = 2) (b: b_students = 4) (c: c_students = 5) (d: d_students = 1)
  : (a_students + b_students + c_students + d_students) * 100 / total_students = 40 := by
  sorry

end same_grade_percentage_l949_94954


namespace bob_questions_created_l949_94994

theorem bob_questions_created :
  let q1 := 13
  let q2 := 2 * q1
  let q3 := 2 * q2
  q1 + q2 + q3 = 91 :=
by
  sorry

end bob_questions_created_l949_94994


namespace negation_of_universal_statement_l949_94964

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^2 + 2 > 0) ↔ ∃ x : ℝ, x^2 + 2 ≤ 0 :=
sorry

end negation_of_universal_statement_l949_94964


namespace evaluate_expression_l949_94936

theorem evaluate_expression : (-7)^3 / 7^2 - 4^4 + 5^2 = -238 := by
  sorry

end evaluate_expression_l949_94936


namespace infinite_series_closed_form_l949_94944

noncomputable def series (a : ℝ) : ℝ :=
  ∑' (k : ℕ), (2 * (k + 1) - 1) / a^k

theorem infinite_series_closed_form (a : ℝ) (ha : 1 < a) : 
  series a = (a^2 + a) / (a - 1)^2 :=
sorry

end infinite_series_closed_form_l949_94944


namespace find_difference_l949_94906

theorem find_difference (x0 y0 : ℝ) 
  (h1 : x0^3 - 2023 * x0 = y0^3 - 2023 * y0 + 2020)
  (h2 : x0^2 + x0 * y0 + y0^2 = 2022) : 
  x0 - y0 = -2020 :=
by
  sorry

end find_difference_l949_94906


namespace system1_solution_system2_solution_l949_94948

theorem system1_solution :
  ∃ (x y : ℝ), 3 * x - 2 * y = -1 ∧ 2 * x + 3 * y = 8 ∧ x = 1 ∧ y = 2 :=
by {
  -- Proof skipped
  sorry
}

theorem system2_solution :
  ∃ (x y : ℝ), 2 * x + y = 1 ∧ 2 * x - y = 7 ∧ x = 2 ∧ y = -3 :=
by {
  -- Proof skipped
  sorry
}

end system1_solution_system2_solution_l949_94948


namespace hyperbola_eccentricity_correct_l949_94984

open Real

noncomputable def hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) : ℝ :=
    let PF1 := (12 * a / 5)
    let PF2 := PF1 - 2 * a
    let c := (2 * sqrt 37 * a / 5)
    sqrt (1 + (b^2 / a^2))  -- Assuming the geometric properties hold, the eccentricity should match
-- Lean function to verify the result
def verify_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) : Prop :=
    hyperbola_eccentricity a b ha hb = sqrt 37 / 5

-- Statement to be verified
theorem hyperbola_eccentricity_correct (a b : ℝ) (ha : a > 0) (hb : b > 0) :
    verify_eccentricity a b ha hb := sorry

end hyperbola_eccentricity_correct_l949_94984


namespace reciprocal_sum_l949_94991

theorem reciprocal_sum (x1 x2 x3 k : ℝ) (h : ∀ x, x^2 + k * x - k * x3 = 0 ∧ x ≠ 0 → x = x1 ∨ x = x2) :
  (1 / x1 + 1 / x2 = 1 / x3) := by
  sorry

end reciprocal_sum_l949_94991


namespace no_100_roads_l949_94934

theorem no_100_roads (k : ℕ) (hk : 3 * k % 2 = 0) : 100 ≠ 3 * k / 2 := 
by
  sorry

end no_100_roads_l949_94934


namespace female_athletes_drawn_is_7_l949_94937

-- Given conditions as definitions
def male_athletes := 64
def female_athletes := 56
def drawn_male_athletes := 8

-- The function that represents the equation in stratified sampling
def stratified_sampling_eq (x : Nat) : Prop :=
  (drawn_male_athletes : ℚ) / (male_athletes) = (x : ℚ) / (female_athletes)

-- The theorem which states that the solution to the problem is x = 7
theorem female_athletes_drawn_is_7 : ∃ x : Nat, stratified_sampling_eq x ∧ x = 7 :=
by
  sorry

end female_athletes_drawn_is_7_l949_94937


namespace trapezoid_area_l949_94962

theorem trapezoid_area (h_base : ℕ) (sum_bases : ℕ) (height : ℕ) (hsum : sum_bases = 36) (hheight : height = 15) :
    (sum_bases * height) / 2 = 270 := by
  sorry

end trapezoid_area_l949_94962


namespace area_bounded_region_l949_94985

theorem area_bounded_region : 
  (∃ x y : ℝ, x^2 + y^2 = 2 * abs (x - y) + 2 * abs (x + y)) →
  (bounded_area : ℝ) = 16 * Real.pi :=
by
  sorry

end area_bounded_region_l949_94985


namespace _l949_94976

section BoxProblem

open Nat

def volume_box (l w h : ℕ) : ℕ := l * w * h
def volume_block (l w h : ℕ) : ℕ := l * w * h

def can_fit_blocks (box_l box_w box_h block_l block_w block_h n_blocks : ℕ) : Prop :=
  (volume_box box_l box_w box_h) = (n_blocks * volume_block block_l block_w block_h)

example : can_fit_blocks 4 3 3 3 2 1 6 :=
by
  -- calculation that proves the theorem goes here, but no need to provide proof steps
  sorry

end BoxProblem

end _l949_94976
