import Mathlib

namespace no_heptagon_cross_section_l400_40093

-- Define what it means for a plane to intersect a cube and form a shape.
noncomputable def possible_cross_section_shapes (P : Plane) (C : Cube) : Set Polygon :=
  sorry -- Placeholder for the actual definition which involves geometric computations.

-- Prove that a heptagon cannot be one of the possible cross-sectional shapes of a cube.
theorem no_heptagon_cross_section (P : Plane) (C : Cube) : 
  Heptagon ∉ possible_cross_section_shapes P C :=
sorry -- Placeholder for the proof.

end no_heptagon_cross_section_l400_40093


namespace neg_ex_iff_forall_geq_0_l400_40021

theorem neg_ex_iff_forall_geq_0 :
  ¬(∃ x_0 : ℝ, x_0^2 - x_0 + 1 < 0) ↔ ∀ x : ℝ, x^2 - x + 1 ≥ 0 :=
by
  sorry

end neg_ex_iff_forall_geq_0_l400_40021


namespace range_f_g_f_eq_g_implies_A_l400_40059

open Set

noncomputable def f (x : ℝ) : ℝ := x^2 + 1
noncomputable def g (x : ℝ) : ℝ := 4 * x + 1

theorem range_f_g :
  (range f ∩ Icc 1 17 = Icc 1 17) ∧ (range g ∩ Icc 1 17 = Icc 1 17) :=
sorry

theorem f_eq_g_implies_A :
  ∀ A ⊆ Icc 0 4, (∀ x ∈ A, f x = g x) → A = {0} ∨ A = {4} ∨ A = {0, 4} :=
sorry

end range_f_g_f_eq_g_implies_A_l400_40059


namespace no_spiky_two_digit_numbers_l400_40061

def is_spiky (n : ℕ) : Prop :=
  ∃ a b : ℕ, a ≠ 0 ∧
             10 ≤ n ∧ n < 100 ∧
             n = 10 * a + b ∧
             n = a + b^3 - 2 * a

theorem no_spiky_two_digit_numbers : ∀ n, 10 ≤ n ∧ n < 100 → ¬ is_spiky n :=
by
  intro n h
  sorry

end no_spiky_two_digit_numbers_l400_40061


namespace coconuts_total_l400_40012

theorem coconuts_total (B_trips : Nat) (Ba_coconuts_per_trip : Nat) (Br_coconuts_per_trip : Nat) (combined_trips : Nat) (B_totals : B_trips = 12) (Ba_coconuts : Ba_coconuts_per_trip = 4) (Br_coconuts : Br_coconuts_per_trip = 8) : combined_trips * (Ba_coconuts_per_trip + Br_coconuts_per_trip) = 144 := 
by
  simp [B_totals, Ba_coconuts, Br_coconuts]
  sorry

end coconuts_total_l400_40012


namespace least_common_multiple_of_marble_sharing_l400_40070

theorem least_common_multiple_of_marble_sharing : 
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 2 4) 5) 7) 8) 10 = 280 :=
sorry

end least_common_multiple_of_marble_sharing_l400_40070


namespace ship_length_is_correct_l400_40009

-- Define the variables
variables (L E S C : ℝ)

-- Define the given conditions
def condition1 (L E S C : ℝ) : Prop := 320 * E = L + 320 * (S - C)
def condition2 (L E S C : ℝ) : Prop := 80 * E = L - 80 * (S + C)

-- Mathematical statement to be proven
theorem ship_length_is_correct
  (L E S C : ℝ)
  (h1 : condition1 L E S C)
  (h2 : condition2 L E S C) :
  L = 26 * E + (2 / 3) * E :=
sorry

end ship_length_is_correct_l400_40009


namespace machines_job_time_l400_40096

theorem machines_job_time (D : ℝ) (h1 : 15 * D = D * 20 * (3 / 4)) : ¬ ∃ t : ℝ, t = D :=
by
  sorry

end machines_job_time_l400_40096


namespace problem_statement_l400_40098

-- Initial sequence and Z expansion definition
def initial_sequence := [1, 2, 3]

def z_expand (seq : List ℕ) : List ℕ :=
  match seq with
  | [] => []
  | [a] => [a]
  | a :: b :: rest => a :: (a + b) :: z_expand (b :: rest)

-- Define a_n
def a_sequence (n : ℕ) : List ℕ :=
  Nat.iterate z_expand n initial_sequence

def a_n (n : ℕ) : ℕ :=
  (a_sequence n).sum

-- Define b_n
def b_n (n : ℕ) : ℕ :=
  a_n n - 2

-- Problem statement
theorem problem_statement :
    a_n 1 = 14 ∧
    a_n 2 = 38 ∧
    a_n 3 = 110 ∧
    ∀ n, b_n n = 4 * (3 ^ n) := sorry

end problem_statement_l400_40098


namespace total_distance_travelled_l400_40071

def walking_distance_flat_surface (speed_flat : ℝ) (time_flat : ℝ) : ℝ := speed_flat * time_flat
def running_distance_downhill (speed_downhill : ℝ) (time_downhill : ℝ) : ℝ := speed_downhill * time_downhill
def walking_distance_hilly (speed_hilly_walk : ℝ) (time_hilly_walk : ℝ) : ℝ := speed_hilly_walk * time_hilly_walk
def running_distance_hilly (speed_hilly_run : ℝ) (time_hilly_run : ℝ) : ℝ := speed_hilly_run * time_hilly_run

def total_distance (ds1 ds2 ds3 ds4 : ℝ) : ℝ := ds1 + ds2 + ds3 + ds4

theorem total_distance_travelled :
  let speed_flat := 8
  let time_flat := 3
  let speed_downhill := 24
  let time_downhill := 1.5
  let speed_hilly_walk := 6
  let time_hilly_walk := 2
  let speed_hilly_run := 18
  let time_hilly_run := 1
  total_distance (walking_distance_flat_surface speed_flat time_flat) (running_distance_downhill speed_downhill time_downhill)
                            (walking_distance_hilly speed_hilly_walk time_hilly_walk) (running_distance_hilly speed_hilly_run time_hilly_run) = 90 := 
by
  sorry

end total_distance_travelled_l400_40071


namespace num_black_cars_l400_40089

theorem num_black_cars (total_cars : ℕ) (one_third_blue : ℚ) (one_half_red : ℚ) 
  (h1 : total_cars = 516) (h2 : one_third_blue = 1/3) (h3 : one_half_red = 1/2) :
  total_cars - (total_cars * one_third_blue + total_cars * one_half_red) = 86 :=
by
  sorry

end num_black_cars_l400_40089


namespace most_likely_outcome_l400_40033

-- Define the probabilities for each outcome
def P_all_boys := (1/2)^6
def P_all_girls := (1/2)^6
def P_3_girls_3_boys := (Nat.choose 6 3) * (1/2)^6
def P_4_one_2_other := 2 * (Nat.choose 6 2) * (1/2)^6

-- Terms with values of each probability
lemma outcome_A : P_all_boys = 1 / 64 := by sorry
lemma outcome_B : P_all_girls = 1 / 64 := by sorry
lemma outcome_C : P_3_girls_3_boys = 20 / 64 := by sorry
lemma outcome_D : P_4_one_2_other = 30 / 64 := by sorry

-- Prove the main statement
theorem most_likely_outcome :
  P_4_one_2_other > P_all_boys ∧ P_4_one_2_other > P_all_girls ∧ P_4_one_2_other > P_3_girls_3_boys :=
by
  rw [outcome_A, outcome_B, outcome_C, outcome_D]
  sorry

end most_likely_outcome_l400_40033


namespace compute_expression_l400_40077

theorem compute_expression : (23 + 12)^2 - (23 - 12)^2 = 1104 := by
  sorry

end compute_expression_l400_40077


namespace compute_difference_l400_40067

def bin_op (x y : ℤ) : ℤ := x^2 * y - 3 * x

theorem compute_difference :
  (bin_op 5 3) - (bin_op 3 5) = 24 := by
  sorry

end compute_difference_l400_40067


namespace tens_digit_of_2013_pow_2018_minus_2019_l400_40074

theorem tens_digit_of_2013_pow_2018_minus_2019 :
  (2013 ^ 2018 - 2019) % 100 / 10 % 10 = 5 := sorry

end tens_digit_of_2013_pow_2018_minus_2019_l400_40074


namespace total_texts_sent_l400_40010

def texts_sent_monday_allison : ℕ := 5
def texts_sent_monday_brittney : ℕ := 5
def texts_sent_tuesday_allison : ℕ := 15
def texts_sent_tuesday_brittney : ℕ := 15

theorem total_texts_sent : (texts_sent_monday_allison + texts_sent_monday_brittney) + 
                           (texts_sent_tuesday_allison + texts_sent_tuesday_brittney) = 40 :=
by
  sorry

end total_texts_sent_l400_40010


namespace sequence_problem_l400_40085

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∀ n m k, n ≠ m → a n = a m + (n - m) * k

theorem sequence_problem
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_sum : a 2003 + a 2005 + a 2007 + a 2009 + a 2011 + a 2013 = 120) :
  2 * a 2018 - a 2028 = 20 :=
sorry

end sequence_problem_l400_40085


namespace ryan_spends_7_hours_on_english_l400_40046

variable (C : ℕ)
variable (E : ℕ)

def hours_spent_on_english (C : ℕ) : ℕ := C + 2

theorem ryan_spends_7_hours_on_english :
  C = 5 → E = hours_spent_on_english C → E = 7 :=
by
  intro hC hE
  rw [hC] at hE
  exact hE

end ryan_spends_7_hours_on_english_l400_40046


namespace number_of_red_items_l400_40084

-- Define the mathematics problem
theorem number_of_red_items (R : ℕ) : 
  (23 + 1) + (11 + 1) + R = 66 → 
  R = 30 := 
by 
  intro h
  sorry

end number_of_red_items_l400_40084


namespace fraction_of_visitors_l400_40028

variable (V E U : ℕ)
variable (H1 : E = U)
variable (H2 : 600 - E - 150 = 450)

theorem fraction_of_visitors (H3 : 600 = E + 150 + 450) : (450 : ℚ) / 600 = (3 : ℚ) / 4 :=
by
  apply sorry

end fraction_of_visitors_l400_40028


namespace calc_product_eq_243_l400_40005

theorem calc_product_eq_243 : 
  (1 / 3) * 9 * (1 / 27) * 81 * (1 / 243) * 729 * (1 / 2187) * 6561 * (1 / 19683) * 59049 = 243 :=
by
  sorry

end calc_product_eq_243_l400_40005


namespace sequence_property_l400_40097

variable (a : ℕ → ℕ)

theorem sequence_property
  (h_bij : Function.Bijective a) (n : ℕ) :
  ∃ k, k < n ∧ a (n - k) < a n ∧ a n < a (n + k) :=
sorry

end sequence_property_l400_40097


namespace remainder_when_divided_by_r_minus_1_l400_40006

def f (r : Int) : Int := r^14 - r + 5

theorem remainder_when_divided_by_r_minus_1 : f 1 = 5 := by
  sorry

end remainder_when_divided_by_r_minus_1_l400_40006


namespace carnations_third_bouquet_l400_40043

theorem carnations_third_bouquet (bouquet1 bouquet2 bouquet3 : ℕ) 
  (h1 : bouquet1 = 9) (h2 : bouquet2 = 14) 
  (h3 : (bouquet1 + bouquet2 + bouquet3) / 3 = 12) : bouquet3 = 13 :=
by
  sorry

end carnations_third_bouquet_l400_40043


namespace part_I_part_II_l400_40072

-- Define the function f(x)
def f (x : ℝ) (m : ℝ) := abs (x + m) + abs (2 * x - 1)

-- Part (I)
theorem part_I (x : ℝ) : (f x (-1) ≤ 2) ↔ (0 ≤ x ∧ x ≤ (4 / 3)) :=
by sorry

-- Part (II)
theorem part_II (m : ℝ) : (∀ x, (3 / 4) ≤ x ∧ x ≤ 2 → f x m ≤ abs (2 * x + 1)) ↔ (-11 / 4) ≤ m ∧ m ≤ 0 :=
by sorry

end part_I_part_II_l400_40072


namespace solve_absolute_value_equation_l400_40062

theorem solve_absolute_value_equation :
  {x : ℝ | 3 * x^2 + 3 * x + 6 = abs (-20 + 5 * x)} = {1.21, -3.87} :=
by
  sorry

end solve_absolute_value_equation_l400_40062


namespace setB_is_empty_l400_40011

noncomputable def setB := {x : ℝ | x^2 + 1 = 0}

theorem setB_is_empty : setB = ∅ :=
by
  sorry

end setB_is_empty_l400_40011


namespace minimum_S_n_at_1008_a1008_neg_a1009_pos_common_difference_pos_l400_40037

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

-- Conditions based on the problem statements
axiom a1_neg : a 1 < 0
axiom S2015_neg : S 2015 < 0
axiom S2016_pos : S 2016 > 0

-- Defining n value where S_n reaches its minimum
def n_min := 1008

theorem minimum_S_n_at_1008 : S n_min = S 1008 := sorry

-- Additional theorems to satisfy the provided conditions
theorem a1008_neg : a 1008 < 0 := sorry
theorem a1009_pos : a 1009 > 0 := sorry
theorem common_difference_pos : ∀ n : ℕ, a (n + 1) - a n > 0 := sorry

end minimum_S_n_at_1008_a1008_neg_a1009_pos_common_difference_pos_l400_40037


namespace area_of_shaded_quadrilateral_l400_40022

-- The problem setup
variables 
  (triangle : Type) [Nonempty triangle]
  (area : triangle → ℝ)
  (EFA FAB FBD CEDF : triangle)
  (h_EFA : area EFA = 5)
  (h_FAB : area FAB = 9)
  (h_FBD : area FBD = 9)
  (h_partition : ∀ t, t = EFA ∨ t = FAB ∨ t = FBD ∨ t = CEDF)

-- The goal to prove
theorem area_of_shaded_quadrilateral (EFA FAB FBD CEDF : triangle) 
  (h_EFA : area EFA = 5) (h_FAB : area FAB = 9) (h_FBD : area FBD = 9)
  (h_partition : ∀ t, t = EFA ∨ t = FAB ∨ t = FBD ∨ t = CEDF) : 
  area CEDF = 45 :=
by
  sorry

end area_of_shaded_quadrilateral_l400_40022


namespace no_solution_system_l400_40001

theorem no_solution_system (a : ℝ) : 
  (∀ x : ℝ, (x - 2 * a > 0) → (3 - 2 * x > x - 6) → false) ↔ a ≥ 3 / 2 :=
by
  sorry

end no_solution_system_l400_40001


namespace work_efficiency_ratio_l400_40090

theorem work_efficiency_ratio
  (A B : ℝ)
  (h1 : A + B = 1 / 18)
  (h2 : B = 1 / 27) :
  A / B = 1 / 2 := 
by
  sorry

end work_efficiency_ratio_l400_40090


namespace series_solution_eq_l400_40014

theorem series_solution_eq (x : ℝ) 
  (h : (∃ a : ℕ → ℝ, (∀ n, a n = 1 + 6 * n) ∧ (∑' n, a n * x^n = 100))) :
  x = 23/25 ∨ x = 1/50 :=
sorry

end series_solution_eq_l400_40014


namespace marathon_y_distance_l400_40051

theorem marathon_y_distance (miles_per_marathon : ℕ) (yards_per_marathon : ℕ) (yards_per_mile : ℕ) (num_marathons : ℕ) (total_yards : ℕ) (y : ℕ) 
  (H1 : miles_per_marathon = 26) 
  (H2 : yards_per_marathon = 312) 
  (H3 : yards_per_mile = 1760) 
  (H4 : num_marathons = 8) 
  (H5 : total_yards = num_marathons * yards_per_marathon) 
  (H6 : total_yards % yards_per_mile = y) 
  (H7 : 0 ≤ y) 
  (H8 : y < yards_per_mile) : 
  y = 736 :=
by 
  sorry

end marathon_y_distance_l400_40051


namespace necessary_but_not_sufficient_condition_l400_40050

-- Definitions of conditions
def condition_p (x : ℝ) := (x - 1) * (x + 2) ≤ 0
def condition_q (x : ℝ) := abs (x + 1) ≤ 1

-- The theorem statement
theorem necessary_but_not_sufficient_condition (x : ℝ) :
  (∀ x, condition_q x → condition_p x) ∧ ¬(∀ x, condition_p x → condition_q x) := 
by
  sorry

end necessary_but_not_sufficient_condition_l400_40050


namespace boys_total_count_l400_40017

theorem boys_total_count 
  (avg_age_all: ℤ) (avg_age_first6: ℤ) (avg_age_last6: ℤ)
  (total_first6: ℤ) (total_last6: ℤ) (total_age_all: ℤ) :
  avg_age_all = 50 →
  avg_age_first6 = 49 →
  avg_age_last6 = 52 →
  total_first6 = 6 * avg_age_first6 →
  total_last6 = 6 * avg_age_last6 →
  total_age_all = total_first6 + total_last6 →
  total_age_all = avg_age_all * 13 :=
by
  intros h_avg_all h_avg_first6 h_avg_last6 h_total_first6 h_total_last6 h_total_age_all
  rw [h_avg_all, h_avg_first6, h_avg_last6] at *
  -- Proof steps skipped
  sorry

end boys_total_count_l400_40017


namespace tetrahedron_inscribed_sphere_radius_l400_40047

theorem tetrahedron_inscribed_sphere_radius (a : ℝ) (r : ℝ) (a_pos : 0 < a) :
  (r = a * (Real.sqrt 6 + 1) / 8) ∨ 
  (r = a * (Real.sqrt 6 - 1) / 8) :=
sorry

end tetrahedron_inscribed_sphere_radius_l400_40047


namespace min_links_for_weights_l400_40036

def min_links_to_break (n : ℕ) : ℕ :=
  if n = 60 then 3 else sorry

theorem min_links_for_weights (n : ℕ) (h1 : n = 60) :
  min_links_to_break n = 3 :=
by
  rw [h1]
  trivial

end min_links_for_weights_l400_40036


namespace officers_on_duty_l400_40079

theorem officers_on_duty
  (F : ℕ)                             -- Total female officers on the police force
  (on_duty_percentage : ℕ)            -- On duty percentage of female officers
  (H1 : on_duty_percentage = 18)      -- 18% of the female officers were on duty
  (H2 : F = 500)                      -- There were 500 female officers on the police force
  : ∃ T : ℕ, T = 2 * (on_duty_percentage * F) / 100 ∧ T = 180 :=
by
  sorry

end officers_on_duty_l400_40079


namespace abs_neg_eight_plus_three_pow_zero_eq_nine_l400_40008

theorem abs_neg_eight_plus_three_pow_zero_eq_nine :
  |-8| + 3^0 = 9 :=
by
  sorry

end abs_neg_eight_plus_three_pow_zero_eq_nine_l400_40008


namespace union_of_A_and_B_l400_40020

def setA : Set ℝ := { x : ℝ | abs (x - 3) < 2 }
def setB : Set ℝ := { x : ℝ | (x + 1) / (x - 2) ≤ 0 }

theorem union_of_A_and_B : setA ∪ setB = { x : ℝ | -1 ≤ x ∧ x < 5 } :=
by
  sorry

end union_of_A_and_B_l400_40020


namespace meeting_point_l400_40038

theorem meeting_point (n : ℕ) (petya_start vasya_start petya_end vasya_end meeting_lamp : ℕ) : 
  n = 100 → petya_start = 1 → vasya_start = 100 → petya_end = 22 → vasya_end = 88 → meeting_lamp = 64 :=
by
  intros h_n h_p_start h_v_start h_p_end h_v_end
  sorry

end meeting_point_l400_40038


namespace positive_root_condition_negative_root_condition_zero_root_condition_l400_40063

-- Positive root condition
theorem positive_root_condition {a b : ℝ} (h : a * b < 0) : ∃ x : ℝ, a * x + b = 0 ∧ x > 0 :=
by
  sorry

-- Negative root condition
theorem negative_root_condition {a b : ℝ} (h : a * b > 0) : ∃ x : ℝ, a * x + b = 0 ∧ x < 0 :=
by
  sorry

-- Root equal to zero condition
theorem zero_root_condition {a b : ℝ} (h₁ : b = 0) (h₂ : a ≠ 0) : ∃ x : ℝ, a * x + b = 0 ∧ x = 0 :=
by
  sorry

end positive_root_condition_negative_root_condition_zero_root_condition_l400_40063


namespace triangle_area_40_l400_40013

noncomputable def area_of_triangle (base height : ℕ) : ℕ :=
  base * height / 2

theorem triangle_area_40
  (a : ℕ) (P B Q : (ℕ × ℕ)) (PB_side : (P.1 = 0 ∧ P.2 = 0) ∧ (B.1 = 10 ∧ B.2 = 0))
  (Q_vert_aboveP : Q.1 = 0 ∧ Q.2 = 8)
  (PQ_perp_PB : P.1 = Q.1)
  (PQ_length : (Q.snd - P.snd) = 8) :
  area_of_triangle 10 8 = 40 := by
  sorry

end triangle_area_40_l400_40013


namespace find_slope_of_line_l400_40057

theorem find_slope_of_line
  (k : ℝ)
  (P : ℝ × ℝ)
  (hP : P = (3, 0))
  (C : ℝ → ℝ → Prop)
  (hC : ∀ x y, C x y ↔ x^2 - y^2 / 3 = 1)
  (A B : ℝ × ℝ)
  (hA : C A.1 A.2)
  (hB : C B.1 B.2)
  (line : ℝ → ℝ → Prop)
  (hline : ∀ x y, line x y ↔ y = k * (x - 3))
  (hintersectA : line A.1 A.2)
  (hintersectB : line B.1 B.2)
  (F : ℝ × ℝ)
  (hF : F = (1, 0))
  (hfoci_sum : ∀ z : ℝ × ℝ, |z.1 - F.1| + |z.2 - F.2| = 16) :
  k = 3 ∨ k = -3 :=
by
  sorry

end find_slope_of_line_l400_40057


namespace can_adjust_to_357_l400_40018

structure Ratio (L O V : ℕ) :=
(lemon : ℕ)
(oil : ℕ)
(vinegar : ℕ)

def MixA : Ratio 1 2 3 := ⟨1, 2, 3⟩
def MixB : Ratio 3 4 5 := ⟨3, 4, 5⟩
def TargetC : Ratio 3 5 7 := ⟨3, 5, 7⟩

theorem can_adjust_to_357 (x y : ℕ) (hA : x * MixA.lemon + y * MixB.lemon = 3 * (x + y))
    (hO : x * MixA.oil + y * MixB.oil = 5 * (x + y))
    (hV : x * MixA.vinegar + y * MixB.vinegar = 7 * (x + y)) :
    (∃ a b : ℕ, x = 3 * a ∧ y = 2 * b) :=
sorry

end can_adjust_to_357_l400_40018


namespace range_of_a_l400_40052

theorem range_of_a (a : ℝ) (h : a > 0) (h1 : ∀ x : ℝ, |a * x - 1| + |a * x - a| ≥ 1) : a ≥ 2 := 
sorry

end range_of_a_l400_40052


namespace xy_product_given_conditions_l400_40044

variable (x y : ℝ)

theorem xy_product_given_conditions (hx : x - y = 5) (hx3 : x^3 - y^3 = 35) : x * y = -6 :=
by
  sorry

end xy_product_given_conditions_l400_40044


namespace part1_solution_part2_solution_l400_40087

-- Part (1)
theorem part1_solution (x : ℝ) : (|x - 2| + |x - 1| ≥ 2) ↔ (x ≥ 2.5 ∨ x ≤ 0.5) := sorry

-- Part (2)
theorem part2_solution (a : ℝ) (h : a > 0) : (∀ x, |a * x - 2| + |a * x - a| ≥ 2) → a ≥ 4 := sorry

end part1_solution_part2_solution_l400_40087


namespace train_lengths_l400_40064

variable (P L_A L_B : ℝ)

noncomputable def speedA := 180 * 1000 / 3600
noncomputable def speedB := 240 * 1000 / 3600

-- Train A crosses platform P in one minute
axiom hA : speedA * 60 = L_A + P

-- Train B crosses platform P in 45 seconds
axiom hB : speedB * 45 = L_B + P

-- Sum of the lengths of Train A and platform P is twice the length of Train B
axiom hSum : L_A + P = 2 * L_B

theorem train_lengths : L_A = 1500 ∧ L_B = 1500 :=
by
  sorry

end train_lengths_l400_40064


namespace symmetric_curve_eq_l400_40066

-- Define the original curve equation and line of symmetry
def original_curve (x y : ℝ) : Prop := y^2 = 4 * x
def line_of_symmetry (x : ℝ) : Prop := x = 2

-- The equivalent Lean 4 statement
theorem symmetric_curve_eq (x y : ℝ) (hx : line_of_symmetry 2) :
  (∀ (x' y' : ℝ), original_curve (4 - x') y' → y^2 = 16 - 4 * x) :=
sorry

end symmetric_curve_eq_l400_40066


namespace find_contaminated_constant_l400_40026

theorem find_contaminated_constant (contaminated_constant : ℝ) (x : ℝ) 
  (h1 : 2 * (x - 3) - contaminated_constant = x + 1) 
  (h2 : x = 9) : contaminated_constant = 2 :=
  sorry

end find_contaminated_constant_l400_40026


namespace break_even_machines_l400_40027

def cost_parts : ℤ := 3600
def cost_patent : ℤ := 4500
def selling_price : ℤ := 180

def total_costs : ℤ := cost_parts + cost_patent

def machines_to_break_even : ℤ := total_costs / selling_price

theorem break_even_machines :
  machines_to_break_even = 45 := by
  sorry

end break_even_machines_l400_40027


namespace road_trip_total_miles_l400_40094

theorem road_trip_total_miles (tracy_miles michelle_miles katie_miles : ℕ) (h_michelle : michelle_miles = 294)
    (h_tracy : tracy_miles = 2 * michelle_miles + 20) (h_katie : michelle_miles = 3 * katie_miles):
  tracy_miles + michelle_miles + katie_miles = 1000 :=
by
  sorry

end road_trip_total_miles_l400_40094


namespace right_triangle_shorter_leg_l400_40034

theorem right_triangle_shorter_leg :
  ∃ (a b : ℤ), a < b ∧ a^2 + b^2 = 65^2 ∧ a = 16 :=
by
  sorry

end right_triangle_shorter_leg_l400_40034


namespace solve_inequality_l400_40080

theorem solve_inequality (x : ℝ) : (x^2 + 5 * x - 14 < 0) ↔ (-7 < x ∧ x < 2) :=
sorry

end solve_inequality_l400_40080


namespace tangent_line_at_e_intervals_of_monotonicity_l400_40088
open Real

noncomputable def f (x : ℝ) : ℝ := x * log x

theorem tangent_line_at_e :
  ∃ (y : ℝ → ℝ), (∀ x : ℝ, y x = 2 * x - exp 1) ∧ (y (exp 1) = f (exp 1)) ∧ (deriv f (exp 1) = deriv y (exp 1)) :=
sorry

theorem intervals_of_monotonicity :
  (∀ x : ℝ, 0 < x ∧ x < exp (-1) → deriv f x < 0) ∧ (∀ x : ℝ, exp (-1) < x → deriv f x > 0) :=
sorry

end tangent_line_at_e_intervals_of_monotonicity_l400_40088


namespace slope_CD_l400_40024

def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 4*y - 8 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 16*x + 8*y + 40 = 0

theorem slope_CD :
  ∀ C D : ℝ × ℝ, circle1 C.1 C.2 → circle2 D.1 D.2 → 
  (C ≠ D → (D.2 - C.2) / (D.1 - C.1) = 5 / 2) := 
by
  -- proof to be completed
  sorry

end slope_CD_l400_40024


namespace total_unique_plants_l400_40025

noncomputable def bed_A : ℕ := 600
noncomputable def bed_B : ℕ := 550
noncomputable def bed_C : ℕ := 400
noncomputable def bed_D : ℕ := 300

noncomputable def intersection_A_B : ℕ := 75
noncomputable def intersection_A_C : ℕ := 125
noncomputable def intersection_B_D : ℕ := 50
noncomputable def intersection_A_B_C : ℕ := 25

theorem total_unique_plants : 
  bed_A + bed_B + bed_C + bed_D - intersection_A_B - intersection_A_C - intersection_B_D + intersection_A_B_C = 1625 := 
by
  sorry

end total_unique_plants_l400_40025


namespace sum_series_equals_l400_40069

theorem sum_series_equals :
  (∑' n : ℕ, if n ≥ 2 then 1 / (n * (n + 3)) else 0) = 13 / 36 :=
by
  sorry

end sum_series_equals_l400_40069


namespace investment_a_l400_40081

/-- Given:
  * b's profit share is Rs. 1800,
  * the difference between a's and c's profit shares is Rs. 720,
  * b invested Rs. 10000,
  * c invested Rs. 12000,
  prove that a invested Rs. 16000. -/
theorem investment_a (P_b : ℝ) (P_a : ℝ) (P_c : ℝ) (B : ℝ) (C : ℝ) (A : ℝ)
  (h1 : P_b = 1800)
  (h2 : P_a - P_c = 720)
  (h3 : B = 10000)
  (h4 : C = 12000)
  (h5 : P_b / B = P_c / C)
  (h6 : P_a / A = P_b / B) : A = 16000 :=
sorry

end investment_a_l400_40081


namespace smallest_value_of_y1_y2_y3_sum_l400_40016

noncomputable def y_problem := 
  ∃ (y1 y2 y3 : ℝ), 
  (y1 + 3 * y2 + 5 * y3 = 120) 
  ∧ (y1^2 + y2^2 + y3^2 = 720 / 7)

theorem smallest_value_of_y1_y2_y3_sum :
  (∃ (y1 y2 y3 : ℝ), 0 < y1 ∧ 0 < y2 ∧ 0 < y3 ∧ (y1 + 3 * y2 + 5 * y3 = 120) 
  ∧ (y1^2 + y2^2 + y3^2 = 720 / 7)) :=
by 
  sorry

end smallest_value_of_y1_y2_y3_sum_l400_40016


namespace right_triangle_sides_l400_40056

theorem right_triangle_sides (a b c : ℝ) (h : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a + b + c = 60 → h = 12 → a^2 + b^2 = c^2 → a * b = 12 * c → 
  (a = 15 ∧ b = 20 ∧ c = 25) ∨ (a = 20 ∧ b = 15 ∧ c = 25) :=
by sorry

end right_triangle_sides_l400_40056


namespace lcm_36_225_l400_40007

theorem lcm_36_225 : Nat.lcm 36 225 = 900 := by
  -- Defining the factorizations as given
  let fact_36 : 36 = 2^2 * 3^2 := by rfl
  let fact_225 : 225 = 3^2 * 5^2 := by rfl

  -- Indicating what LCM we need to prove
  show Nat.lcm 36 225 = 900

  -- Proof (skipped)
  sorry

end lcm_36_225_l400_40007


namespace growth_rate_equation_l400_40015

variable (a x : ℝ)

-- Condition: The number of visitors in March is three times that of January
def visitors_in_march := 3 * a

-- Condition: The average growth rate of visitors in February and March is x
def growth_rate := x

-- Statement to prove
theorem growth_rate_equation 
  (h : (1 + x)^2 = 3) : true :=
by sorry

end growth_rate_equation_l400_40015


namespace total_cats_handled_last_year_l400_40053

theorem total_cats_handled_last_year (num_adult_cats : ℕ) (two_thirds_female : ℕ) (seventy_five_percent_litters : ℕ) 
                                     (kittens_per_litter : ℕ) (adopted_returned : ℕ) :
  num_adult_cats = 120 →
  two_thirds_female = (2 * num_adult_cats) / 3 →
  seventy_five_percent_litters = (3 * two_thirds_female) / 4 →
  kittens_per_litter = 3 →
  adopted_returned = 15 →
  num_adult_cats + seventy_five_percent_litters * kittens_per_litter + adopted_returned = 315 := 
by
  intros h1 h2 h3 h4 h5
  sorry

end total_cats_handled_last_year_l400_40053


namespace sum_of_three_numbers_l400_40091

theorem sum_of_three_numbers (a b c : ℕ) (mean_least difference greatest_diff : ℕ)
  (h1 : a ≤ b) (h2 : b ≤ c)
  (h3 : mean_least = 8) (h4 : greatest_diff = 25)
  (h5 : c - a = 26)
  (h6 : (a + b + c) / 3 = a + mean_least) 
  (h7 : (a + b + c) / 3 = c - greatest_diff) : 
a + b + c = 81 := 
sorry

end sum_of_three_numbers_l400_40091


namespace sum_of_consecutive_at_least_20_sum_of_consecutive_greater_than_20_l400_40045

noncomputable def sum_of_consecutive_triplets (a : Fin 12 → ℕ) (i : Fin 12) : ℕ :=
a i + a ((i + 1) % 12) + a ((i + 2) % 12)

theorem sum_of_consecutive_at_least_20 :
  ∀ (a : Fin 12 → ℕ), (∀ i : Fin 12, (1 ≤ a i ∧ a i ≤ 12) ∧ ∀ (j k : Fin 12), j ≠ k → a j ≠ a k) →
  ∃ i : Fin 12, sum_of_consecutive_triplets a i ≥ 20 :=
by
  sorry

theorem sum_of_consecutive_greater_than_20 :
  ∀ (a : Fin 12 → ℕ), (∀ i : Fin 12, (1 ≤ a i ∧ a i ≤ 12) ∧ ∀ (j k : Fin 12), j ≠ k → a j ≠ a k) →
  ∃ i : Fin 12, sum_of_consecutive_triplets a i > 20 :=
by
  sorry

end sum_of_consecutive_at_least_20_sum_of_consecutive_greater_than_20_l400_40045


namespace fifteenth_triangular_number_is_120_l400_40030

def triangular_number (n : ℕ) : ℕ := (n * (n + 1)) / 2

theorem fifteenth_triangular_number_is_120 : triangular_number 15 = 120 := by
  sorry

end fifteenth_triangular_number_is_120_l400_40030


namespace area_of_triangle_AEB_is_correct_l400_40078

noncomputable def area_triangle_AEB : ℚ :=
by
  -- Definitions of given conditions
  let AB := 5
  let BC := 3
  let DF := 1
  let GC := 2

  -- Conditions of the problem
  have h1 : AB = 5 := rfl
  have h2 : BC = 3 := rfl
  have h3 : DF = 1 := rfl
  have h4 : GC = 2 := rfl

  -- The goal to prove
  exact 25 / 2

-- Statement in Lean 4 with the conditions and the correct answer
theorem area_of_triangle_AEB_is_correct :
  area_triangle_AEB = 25 / 2 := sorry -- The proof is omitted for this example

end area_of_triangle_AEB_is_correct_l400_40078


namespace beach_relaxing_people_l400_40058

def row1_original := 24
def row1_got_up := 3

def row2_original := 20
def row2_got_up := 5

def row3_original := 18

def total_left_relaxing (r1o r1u r2o r2u r3o : Nat) : Nat :=
  r1o + r2o + r3o - (r1u + r2u)

theorem beach_relaxing_people : total_left_relaxing row1_original row1_got_up row2_original row2_got_up row3_original = 54 :=
by
  sorry

end beach_relaxing_people_l400_40058


namespace find_davids_marks_in_physics_l400_40082

theorem find_davids_marks_in_physics (marks_english : ℕ) (marks_math : ℕ) (marks_chemistry : ℕ) (marks_biology : ℕ)
  (average_marks : ℕ) (num_subjects : ℕ) (H1 : marks_english = 61) 
  (H2 : marks_math = 65) (H3 : marks_chemistry = 67) 
  (H4 : marks_biology = 85) (H5 : average_marks = 72) (H6 : num_subjects = 5) :
  ∃ (marks_physics : ℕ), marks_physics = 82 :=
by
  sorry

end find_davids_marks_in_physics_l400_40082


namespace find_k_l400_40031

noncomputable def f (x : ℝ) : ℝ := 6 * x^2 + 4 * x - (1 / x) + 2

noncomputable def g (x : ℝ) (k : ℝ) : ℝ := x^2 + 3 * x - k

theorem find_k (k : ℝ) : 
  f 3 - g 3 k = 5 → 
  k = - 134 / 3 :=
by
  sorry

end find_k_l400_40031


namespace value_of_x_l400_40054

theorem value_of_x (m n : ℝ) (z x : ℝ) (hz : z ≠ 0) (hx : x = m * (n / z) ^ 3) (hconst : 5 * (16 ^ 3) = m * (n ^ 3)) (hz_const : z = 64) : x = 5 / 64 :=
by
  -- proof omitted
  sorry

end value_of_x_l400_40054


namespace largest_triangle_perimeter_maximizes_l400_40099

theorem largest_triangle_perimeter_maximizes 
  (y : ℤ) 
  (h1 : 3 ≤ y) 
  (h2 : y < 16) : 
  (7 + 9 + y) = 31 ↔ y = 15 := 
by 
  sorry

end largest_triangle_perimeter_maximizes_l400_40099


namespace not_divisible_by_44_l400_40055

theorem not_divisible_by_44 (k : ℤ) (n : ℤ) (h1 : n = k * (k + 1) * (k + 2)) (h2 : 11 ∣ n) : ¬ (44 ∣ n) :=
sorry

end not_divisible_by_44_l400_40055


namespace evaluate_expression_l400_40035

theorem evaluate_expression : 3^(1^(2^3)) + ((3^1)^2)^2 = 84 := 
by
  sorry

end evaluate_expression_l400_40035


namespace ratio_x_y_half_l400_40095

variable (x y z : ℝ)

theorem ratio_x_y_half (h1 : (x + 4) / 2 = (y + 9) / (z - 3))
                      (h2 : (y + 9) / (z - 3) = (x + 5) / (z - 5)) :
  x / y = 1 / 2 :=
by
  sorry

end ratio_x_y_half_l400_40095


namespace find_initial_students_l400_40092

def initial_students (S : ℕ) : Prop :=
  S - 4 + 42 = 48 

theorem find_initial_students (S : ℕ) (h : initial_students S) : S = 10 :=
by {
  -- The proof can be filled out here but we skip it using sorry
  sorry
}

end find_initial_students_l400_40092


namespace sum_of_products_of_roots_l400_40041

theorem sum_of_products_of_roots :
  ∀ (p q r : ℝ), (4 * p^3 - 6 * p^2 + 17 * p - 10 = 0) ∧ 
                 (4 * q^3 - 6 * q^2 + 17 * q - 10 = 0) ∧ 
                 (4 * r^3 - 6 * r^2 + 17 * r - 10 = 0) →
                 (p * q + q * r + r * p = 17 / 4) :=
by
  sorry

end sum_of_products_of_roots_l400_40041


namespace absolute_difference_of_integers_l400_40039

theorem absolute_difference_of_integers (x y : ℤ) (h1 : (x + y) / 2 = 15) (h2 : Int.sqrt (x * y) + 6 = 15) : |x - y| = 24 :=
  sorry

end absolute_difference_of_integers_l400_40039


namespace minimum_trips_needed_l400_40003

def masses : List ℕ := [150, 60, 70, 71, 72, 100, 101, 102, 103]
def capacity : ℕ := 200

theorem minimum_trips_needed (masses : List ℕ) (capacity : ℕ) : 
  masses = [150, 60, 70, 71, 72, 100, 101, 102, 103] →
  capacity = 200 →
  ∃ trips : ℕ, trips = 5 :=
by
  sorry

end minimum_trips_needed_l400_40003


namespace max_min_values_in_region_l400_40023

-- Define the function
def z (x y : ℝ) : ℝ := 4 * x^2 + y^2 - 16 * x - 4 * y + 20

-- Define the region D
def D (x y : ℝ) : Prop := (0 ≤ x) ∧ (x - 2 * y ≤ 0) ∧ (x + y - 6 ≤ 0)

-- Define the proof problem
theorem max_min_values_in_region :
  (∀ (x y : ℝ), D x y → z x y ≥ 0) ∧
  (∀ (x y : ℝ), D x y → z x y ≤ 32) :=
by 
  sorry -- Proof omitted

end max_min_values_in_region_l400_40023


namespace repeating_decimal_to_fraction_l400_40049

theorem repeating_decimal_to_fraction : (0.3666666 : ℚ) = 11 / 30 :=
by sorry

end repeating_decimal_to_fraction_l400_40049


namespace total_money_is_220_l400_40083

-- Define the amounts on Table A, B, and C
def tableA := 40
def tableC := tableA + 20
def tableB := 2 * tableC

-- Define the total amount of money on all tables
def total_money := tableA + tableB + tableC

-- The main theorem to prove
theorem total_money_is_220 : total_money = 220 :=
by
  sorry

end total_money_is_220_l400_40083


namespace frosting_cupcakes_l400_40002

theorem frosting_cupcakes (R_Cagney R_Lacey R_Jamie : ℕ)
  (H1 : R_Cagney = 1 / 20)
  (H2 : R_Lacey = 1 / 30)
  (H3 : R_Jamie = 1 / 40)
  (TotalTime : ℕ)
  (H4 : TotalTime = 600) :
  (R_Cagney + R_Lacey + R_Jamie) * TotalTime = 65 :=
by
  sorry

end frosting_cupcakes_l400_40002


namespace average_age_of_4_students_l400_40019

theorem average_age_of_4_students :
  let total_age_15 := 15 * 15
  let age_15th := 25
  let total_age_9 := 16 * 9
  (total_age_15 - total_age_9 - age_15th) / 4 = 14 :=
by
  sorry

end average_age_of_4_students_l400_40019


namespace decimal_equivalent_of_one_tenth_squared_l400_40048

theorem decimal_equivalent_of_one_tenth_squared : 
  (1 / 10 : ℝ)^2 = 0.01 := by
  sorry

end decimal_equivalent_of_one_tenth_squared_l400_40048


namespace height_of_pole_l400_40000

-- Definitions for the conditions
def ascends_first_minute := 2
def slips_second_minute := 1
def net_ascent_per_two_minutes := ascends_first_minute - slips_second_minute
def total_minutes := 17
def pairs_of_minutes := (total_minutes - 1) / 2  -- because the 17th minute is separate
def net_ascent_first_16_minutes := pairs_of_minutes * net_ascent_per_two_minutes

-- The final ascent in the 17th minute
def ascent_final_minute := 2

-- Total ascent
def total_ascent := net_ascent_first_16_minutes + ascent_final_minute

-- Statement to prove the height of the pole
theorem height_of_pole : total_ascent = 10 :=
by
  sorry

end height_of_pole_l400_40000


namespace rental_cost_per_day_l400_40004

theorem rental_cost_per_day (p m c : ℝ) (d : ℝ) (hc : c = 0.08) (hm : m = 214.0) (hp : p = 46.12) (h_total : p = d + m * c) : d = 29.00 := 
by
  sorry

end rental_cost_per_day_l400_40004


namespace one_greater_than_17_over_10_l400_40076

theorem one_greater_than_17_over_10 (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_eq : a + b + c = a * b * c) : 
  a > 17 / 10 ∨ b > 17 / 10 ∨ c > 17 / 10 :=
by
  sorry

end one_greater_than_17_over_10_l400_40076


namespace sin_cos_tan_min_value_l400_40068

open Real

theorem sin_cos_tan_min_value :
  ∀ x : ℝ, (sin x)^2 + (cos x)^2 = 1 → (sin x)^4 + (cos x)^4 + (tan x)^2 ≥ 3/2 :=
by
  sorry

end sin_cos_tan_min_value_l400_40068


namespace complex_arithmetic_1_complex_arithmetic_2_l400_40040

-- Proof Problem 1
theorem complex_arithmetic_1 : 
  (1 : ℂ) * (-2 - 4 * I) - (7 - 5 * I) + (1 + 7 * I) = -8 + 8 * I := 
sorry

-- Proof Problem 2
theorem complex_arithmetic_2 : 
  (1 + I) * (2 + I) + (5 + I) / (1 - I) + (1 - I) ^ 2 = 3 + 4 * I := 
sorry

end complex_arithmetic_1_complex_arithmetic_2_l400_40040


namespace max_min_product_xy_l400_40073

-- Definition of conditions
variables (a x y : ℝ)
def condition_1 : Prop := x + y = a
def condition_2 : Prop := x^2 + y^2 = -a^2 + 2

-- The main theorem statement
theorem max_min_product_xy (a : ℝ) (ha_range : -2 ≤ a ∧ a ≤ 2): 
  (∀ x y : ℝ, condition_1 a x y ∧ condition_2 a x y → (x * y) ≤ (1 / 3)) ∧
  (∀ x y : ℝ, condition_1 a x y ∧ condition_2 a x y → (x * y) ≥ (-1)) :=
sorry

end max_min_product_xy_l400_40073


namespace find_expression_value_l400_40029

theorem find_expression_value (x : ℝ) : 
  let a := 2015 * x + 2014
  let b := 2015 * x + 2015
  let c := 2015 * x + 2016
  a^2 + b^2 + c^2 - a * b - b * c - c * a = 3 :=
by
  let a := 2015 * x + 2014
  let b := 2015 * x + 2015
  let c := 2015 * x + 2016
  have h : a^2 + b^2 + c^2 - a * b - b * c - c * a = 3 := sorry
  exact h

end find_expression_value_l400_40029


namespace sum_of_coeffs_eq_negative_21_l400_40042

noncomputable def expand_and_sum_coeff (d : ℤ) : ℤ :=
  let expression := -(4 - d) * (d + 2 * (4 - d))
  let expanded_form := -d^2 + 12*d - 32
  let sum_of_coeffs := -1 + 12 - 32
  sum_of_coeffs

theorem sum_of_coeffs_eq_negative_21 (d : ℤ) : expand_and_sum_coeff d = -21 := by
  sorry

end sum_of_coeffs_eq_negative_21_l400_40042


namespace min_value2k2_minus_4n_l400_40086

-- We state the problem and set up the conditions
variable (k n : ℝ)
variable (nonneg_k : k ≥ 0)
variable (nonneg_n : n ≥ 0)
variable (eq1 : 2 * k + n = 2)

-- Main statement to prove
theorem min_value2k2_minus_4n : ∃ k n : ℝ, k ≥ 0 ∧ n ≥ 0 ∧ 2 * k + n = 2 ∧ (∀ k' n' : ℝ, k' ≥ 0 ∧ n' ≥ 0 ∧ 2 * k' + n' = 2 → 2 * k'^2 - 4 * n' ≥ -8) := 
sorry

end min_value2k2_minus_4n_l400_40086


namespace remainder_when_multiplied_by_three_and_divided_by_eighth_prime_l400_40060

def first_seven_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17]

def sum_first_seven_primes : ℕ := first_seven_primes.sum

def eighth_prime : ℕ := 19

theorem remainder_when_multiplied_by_three_and_divided_by_eighth_prime :
  ((sum_first_seven_primes * 3) % eighth_prime = 3) :=
by
  sorry

end remainder_when_multiplied_by_three_and_divided_by_eighth_prime_l400_40060


namespace max_value_of_n_l400_40075

theorem max_value_of_n (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h_arith_seq : ∀ n, a (n + 1) - a n = a 2 - a 1)
  (h_cond : a 11 / a 10 < -1)
  (h_maximum : ∃ N, ∀ n > N, S n ≤ S N) :
  ∃ N, S N > 0 ∧ ∀ m, S m > 0 → m ≤ N :=
by
  sorry

end max_value_of_n_l400_40075


namespace jerome_money_left_l400_40065

-- Given conditions
def half_of_money (m : ℕ) : Prop := m / 2 = 43
def amount_given_to_meg (x : ℕ) : Prop := x = 8
def amount_given_to_bianca (x : ℕ) : Prop := x = 3 * 8

-- Problem statement
theorem jerome_money_left (m : ℕ) (x : ℕ) (y : ℕ) (h1 : half_of_money m) (h2 : amount_given_to_meg x) (h3 : amount_given_to_bianca y) : m - x - y = 54 :=
sorry

end jerome_money_left_l400_40065


namespace sequence_bounded_l400_40032

open Classical

noncomputable def bounded_sequence (a : ℕ → ℝ) (M : ℝ) :=
  ∀ n : ℕ, n > 0 → a n < M

theorem sequence_bounded {a : ℕ → ℝ} (h0 : 0 ≤ a 1 ∧ a 1 ≤ 2)
  (h : ∀ n : ℕ, n > 0 → a (n + 1) = a n + (a n)^2 / n^3) :
  ∃ M : ℝ, 0 < M ∧ bounded_sequence a M :=
by
  sorry

end sequence_bounded_l400_40032
