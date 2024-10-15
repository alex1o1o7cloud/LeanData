import Mathlib

namespace NUMINAMATH_GPT_average_headcount_correct_l106_10663

def avg_headcount_03_04 : ℕ := 11500
def avg_headcount_04_05 : ℕ := 11600
def avg_headcount_05_06 : ℕ := 11300

noncomputable def average_headcount : ℕ :=
  (avg_headcount_03_04 + avg_headcount_04_05 + avg_headcount_05_06) / 3

theorem average_headcount_correct :
  average_headcount = 11467 :=
by
  sorry

end NUMINAMATH_GPT_average_headcount_correct_l106_10663


namespace NUMINAMATH_GPT_shop_length_l106_10684

def monthly_rent : ℝ := 2244
def width : ℝ := 18
def annual_rent_per_sqft : ℝ := 68

theorem shop_length : 
  (monthly_rent * 12 / annual_rent_per_sqft / width) = 22 := 
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_shop_length_l106_10684


namespace NUMINAMATH_GPT_vacation_cost_correct_l106_10640

namespace VacationCost

-- Define constants based on conditions
def starting_charge_per_dog : ℝ := 2
def charge_per_block : ℝ := 1.25
def number_of_dogs : ℕ := 20
def total_blocks : ℕ := 128
def family_members : ℕ := 5

-- Define total earnings from walking dogs
def total_earnings : ℝ :=
  (number_of_dogs * starting_charge_per_dog) + (total_blocks * charge_per_block)

-- Define the total cost of the vacation
noncomputable def total_cost_of_vacation : ℝ :=
  total_earnings / family_members * family_members

-- Proof statement: The total cost of the vacation is $200
theorem vacation_cost_correct : total_cost_of_vacation = 200 := by
  sorry

end VacationCost

end NUMINAMATH_GPT_vacation_cost_correct_l106_10640


namespace NUMINAMATH_GPT_binary_divisible_by_136_l106_10612

theorem binary_divisible_by_136 :
  let N := 2^139 + 2^105 + 2^15 + 2^13
  N % 136 = 0 :=
by {
  let N := 2^139 + 2^105 + 2^15 + 2^13;
  sorry
}

end NUMINAMATH_GPT_binary_divisible_by_136_l106_10612


namespace NUMINAMATH_GPT_det_matrix_4x4_l106_10697

def matrix_4x4 : Matrix (Fin 4) (Fin 4) ℤ :=
  ![
    ![3, 0, 2, 0],
    ![2, 3, -1, 4],
    ![0, 4, -2, 3],
    ![5, 2, 0, 1]
  ]

theorem det_matrix_4x4 : Matrix.det matrix_4x4 = -84 :=
by
  sorry

end NUMINAMATH_GPT_det_matrix_4x4_l106_10697


namespace NUMINAMATH_GPT_proof_problem_l106_10616

-- Definitions
variable (T : Type) (Sam : T)
variable (solves_all : T → Prop) (passes : T → Prop)

-- Given condition (Dr. Evans's statement)
axiom dr_evans_statement : ∀ x : T, solves_all x → passes x

-- Statement to be proven
theorem proof_problem : ¬ (passes Sam) → ¬ (solves_all Sam) :=
  by sorry

end NUMINAMATH_GPT_proof_problem_l106_10616


namespace NUMINAMATH_GPT_dihedral_angle_equivalence_l106_10648

namespace CylinderGeometry

variables {α β γ : ℝ} 

-- Given conditions
axiom axial_cross_section : Type
axiom point_on_circumference (C : axial_cross_section) : Prop
axiom dihedral_angle (α: ℝ) : Prop
axiom angle_CAB (β : ℝ) : Prop
axiom angle_CA1B (γ : ℝ) : Prop

-- Proven statement
theorem dihedral_angle_equivalence
    (hx : point_on_circumference C)
    (hα : dihedral_angle α)
    (hβ : angle_CAB β)
    (hγ : angle_CA1B γ):
  α = Real.arcsin (Real.cos β / Real.cos γ) :=
sorry

end CylinderGeometry

end NUMINAMATH_GPT_dihedral_angle_equivalence_l106_10648


namespace NUMINAMATH_GPT_find_k_for_sum_of_cubes_l106_10652

theorem find_k_for_sum_of_cubes (k : ℝ) (r s : ℝ)
  (h1 : r + s = -2)
  (h2 : r * s = k / 3)
  (h3 : r^3 + s^3 = r + s) : k = 3 :=
by
  -- Sorry will be replaced by the actual proof
  sorry

end NUMINAMATH_GPT_find_k_for_sum_of_cubes_l106_10652


namespace NUMINAMATH_GPT_polar_to_rectangular_l106_10626

theorem polar_to_rectangular (r θ : ℝ) (hr : r = 5) (hθ : θ = 5 * Real.pi / 4) :
  ∃ x y : ℝ, x = r * Real.cos θ ∧ y = r * Real.sin θ ∧ x = -5 * Real.sqrt 2 / 2 ∧ y = -5 * Real.sqrt 2 / 2 :=
by
  rw [hr, hθ]
  sorry

end NUMINAMATH_GPT_polar_to_rectangular_l106_10626


namespace NUMINAMATH_GPT_number_of_trees_in_garden_l106_10667

def total_yard_length : ℕ := 600
def distance_between_trees : ℕ := 24
def tree_at_each_end : ℕ := 1

theorem number_of_trees_in_garden : (total_yard_length / distance_between_trees) + tree_at_each_end = 26 := by
  sorry

end NUMINAMATH_GPT_number_of_trees_in_garden_l106_10667


namespace NUMINAMATH_GPT_value_of_x_l106_10642

theorem value_of_x (x y z : ℕ) (h1 : x = y / 3) (h2 : y = z / 4) (h3 : z = 48) : x = 4 := 
by
  sorry

end NUMINAMATH_GPT_value_of_x_l106_10642


namespace NUMINAMATH_GPT_probability_neither_test_l106_10664

theorem probability_neither_test (P_hist : ℚ) (P_geo : ℚ) (indep : Prop) 
  (H1 : P_hist = 5/9) (H2 : P_geo = 1/3) (H3 : indep) :
  (1 - P_hist) * (1 - P_geo) = 8/27 := by
  sorry

end NUMINAMATH_GPT_probability_neither_test_l106_10664


namespace NUMINAMATH_GPT_lucas_change_l106_10602

-- Define the costs of items and the initial amount.
def initial_amount : ℝ := 20.00
def cost_avocados : ℝ := 1.50 + 2.25 + 3.00
def cost_water : ℝ := 2 * 1.75
def cost_apples : ℝ := 4 * 0.75

-- Define the total cost.
def total_cost : ℝ := cost_avocados + cost_water + cost_apples

-- Define the expected change.
def expected_change : ℝ := initial_amount - total_cost

-- The proposition (statement) we want to prove.
theorem lucas_change : expected_change = 6.75 :=
by
  sorry -- Proof to be completed.

end NUMINAMATH_GPT_lucas_change_l106_10602


namespace NUMINAMATH_GPT_train_pass_time_l106_10610

def train_length : ℕ := 250
def train_speed_kmph : ℕ := 36
def station_length : ℕ := 200

def total_distance : ℕ := train_length + station_length

noncomputable def train_speed_mps : ℚ := (train_speed_kmph : ℚ) * 1000 / 3600

noncomputable def time_to_pass_station : ℚ := total_distance / train_speed_mps

theorem train_pass_time : time_to_pass_station = 45 := by
  sorry

end NUMINAMATH_GPT_train_pass_time_l106_10610


namespace NUMINAMATH_GPT_woman_lawyer_probability_l106_10668

theorem woman_lawyer_probability (total_members women_count lawyer_prob : ℝ) 
  (h1: total_members = 100) 
  (h2: women_count = 0.70 * total_members) 
  (h3: lawyer_prob = 0.40) : 
  (0.40 * 0.70) = 0.28 := by sorry

end NUMINAMATH_GPT_woman_lawyer_probability_l106_10668


namespace NUMINAMATH_GPT_harriet_siblings_product_l106_10662

-- Definitions based on conditions
def Harry_sisters : ℕ := 6
def Harry_brothers : ℕ := 3
def Harriet_sisters : ℕ := Harry_sisters - 1
def Harriet_brothers : ℕ := Harry_brothers

-- Statement to prove
theorem harriet_siblings_product : Harriet_sisters * Harriet_brothers = 15 := by
  -- Proof is skipped
  sorry

end NUMINAMATH_GPT_harriet_siblings_product_l106_10662


namespace NUMINAMATH_GPT_negation_of_p_is_neg_p_l106_10683

-- Define the original proposition p
def p : Prop := ∃ n : ℕ, 2^n > 100

-- Define what it means for the negation of p to be satisfied
def neg_p := ∀ n : ℕ, 2^n ≤ 100

-- Statement to prove the logical equivalence between the negation of p and neg_p
theorem negation_of_p_is_neg_p : ¬ p ↔ neg_p := by
  sorry

end NUMINAMATH_GPT_negation_of_p_is_neg_p_l106_10683


namespace NUMINAMATH_GPT_school_anniversary_problem_l106_10623

theorem school_anniversary_problem
    (total_cost : ℕ)
    (cost_commemorative_albums cost_bone_china_cups : ℕ)
    (num_commemorative_albums num_bone_china_cups : ℕ)
    (price_commemorative_album price_bone_china_cup : ℕ)
    (H1 : total_cost = 312000)
    (H2 : cost_commemorative_albums + cost_bone_china_cups = total_cost)
    (H3 : cost_commemorative_albums = 3 * cost_bone_china_cups)
    (H4 : price_commemorative_album = 3 / 2 * price_bone_china_cup)
    (H5 : num_bone_china_cups = 4 * num_commemorative_albums + 1600) :
    (cost_commemorative_albums = 72000 ∧ cost_bone_china_cups = 240000) ∧
    (price_commemorative_album = 45 ∧ price_bone_china_cup = 30) :=
by
  sorry

end NUMINAMATH_GPT_school_anniversary_problem_l106_10623


namespace NUMINAMATH_GPT_trig_expression_value_l106_10656

theorem trig_expression_value (θ : ℝ) (h1 : Real.tan (2 * θ) = -2 * Real.sqrt 2)
  (h2 : 2 * θ > Real.pi / 2 ∧ 2 * θ < Real.pi) : 
  (2 * Real.cos θ / 2 ^ 2 - Real.sin θ - 1) / (Real.sqrt 2 * Real.sin (θ + Real.pi / 4)) = 2 * Real.sqrt 2 - 3 :=
by
  sorry

end NUMINAMATH_GPT_trig_expression_value_l106_10656


namespace NUMINAMATH_GPT_simple_interest_rate_l106_10624

theorem simple_interest_rate (P : ℝ) (R : ℝ) (T : ℝ) (I : ℝ) : 
  T = 6 → I = (7/6) * P - P → I = P * R * T / 100 → R = 100 / 36 :=
by
  intros T_eq I_eq simple_interest_eq
  sorry

end NUMINAMATH_GPT_simple_interest_rate_l106_10624


namespace NUMINAMATH_GPT_min_value_f_l106_10609

theorem min_value_f (x : ℝ) (h : 0 < x) : 
  ∃ c: ℝ, c = 2.5 ∧ (∀ x, 0 < x → x^2 + 1 / x^2 + 1 / (x^2 + 1 / x^2) ≥ c) :=
by sorry

end NUMINAMATH_GPT_min_value_f_l106_10609


namespace NUMINAMATH_GPT_train_arrival_problem_shooting_problem_l106_10665

-- Define trials and outcome types
inductive OutcomeTrain : Type
| onTime
| notOnTime

inductive OutcomeShooting : Type
| hitTarget
| missTarget

-- Scenario 1: Train Arrival Problem
def train_arrival_trials_refers_to (n : Nat) : Prop := 
  ∃ trials : List OutcomeTrain, trials.length = 3

-- Scenario 2: Shooting Problem
def shooting_trials_refers_to (n : Nat) : Prop :=
  ∃ trials : List OutcomeShooting, trials.length = 2

theorem train_arrival_problem : train_arrival_trials_refers_to 3 :=
by
  sorry

theorem shooting_problem : shooting_trials_refers_to 2 :=
by
  sorry

end NUMINAMATH_GPT_train_arrival_problem_shooting_problem_l106_10665


namespace NUMINAMATH_GPT_original_numbers_product_l106_10675

theorem original_numbers_product (a b c d x : ℕ) 
  (h1 : a + b + c + d = 243)
  (h2 : a + 8 = x)
  (h3 : b - 8 = x)
  (h4 : c * 8 = x)
  (h5 : d / 8 = x) : 
  (min (min a (min b (min c d))) * max a (max b (max c d))) = 576 :=
by 
  sorry

end NUMINAMATH_GPT_original_numbers_product_l106_10675


namespace NUMINAMATH_GPT_prob_four_of_a_kind_after_re_roll_l106_10618

noncomputable def probability_of_four_of_a_kind : ℚ :=
sorry

theorem prob_four_of_a_kind_after_re_roll :
  (probability_of_four_of_a_kind =
    (1 : ℚ) / 6) :=
sorry

end NUMINAMATH_GPT_prob_four_of_a_kind_after_re_roll_l106_10618


namespace NUMINAMATH_GPT_probability_both_heads_l106_10672

-- Define the sample space and the probability of each outcome
def sample_space : List (Bool × Bool) := [(true, true), (true, false), (false, true), (false, false)]

-- Define the function to check for both heads
def both_heads (outcome : Bool × Bool) : Bool :=
  outcome = (true, true)

-- Calculate the probability of both heads
theorem probability_both_heads :
  (sample_space.filter both_heads).length / sample_space.length = 1 / 4 := sorry

end NUMINAMATH_GPT_probability_both_heads_l106_10672


namespace NUMINAMATH_GPT_infinite_solutions_l106_10611

theorem infinite_solutions (x y : ℕ) (h : x ≥ 1 ∧ y ≥ 1) : ∃ (x y : ℕ), x^2 + y^2 = x^3 :=
by {
  sorry 
}

end NUMINAMATH_GPT_infinite_solutions_l106_10611


namespace NUMINAMATH_GPT_is_hexagonal_number_2016_l106_10691

theorem is_hexagonal_number_2016 :
  ∃ (n : ℕ), 2 * n^2 - n = 2016 :=
sorry

end NUMINAMATH_GPT_is_hexagonal_number_2016_l106_10691


namespace NUMINAMATH_GPT_correct_statements_arithmetic_seq_l106_10654

/-- For an arithmetic sequence {a_n} with a1 > 0 and common difference d ≠ 0, 
    the correct statements among options A, B, C, and D are B and C. -/
theorem correct_statements_arithmetic_seq (a : ℕ → ℤ) (S : ℕ → ℤ) (d : ℤ) (h_seq : ∀ n, a (n + 1) = a n + d) 
  (h_sum : ∀ n, S n = (n * (a 1 + a n)) / 2) (h_a1_pos : a 1 > 0) (h_d_ne_0 : d ≠ 0) : 
  (S 5 = S 9 → 
   S 7 = (10 * a 4) / 2) ∧ 
  (S 6 > S 7 → S 7 > S 8) := 
sorry

end NUMINAMATH_GPT_correct_statements_arithmetic_seq_l106_10654


namespace NUMINAMATH_GPT_rob_travel_time_to_park_l106_10670

theorem rob_travel_time_to_park : 
  ∃ R : ℝ, 
    (∀ Tm : ℝ, Tm = 3 * R) ∧ -- Mark's travel time is three times Rob's travel time
    (∀ Tr : ℝ, Tm - 2 = R) → -- Considering Mark's head start of 2 hours
    R = 1 :=
sorry

end NUMINAMATH_GPT_rob_travel_time_to_park_l106_10670


namespace NUMINAMATH_GPT_soda_cans_purchasable_l106_10655

theorem soda_cans_purchasable (S Q : ℕ) (t D : ℝ) (hQ_pos : Q > 0) :
    let quarters_from_dollars := 4 * D
    let total_quarters_with_tax := quarters_from_dollars * (1 + t)
    (total_quarters_with_tax / Q) * S = (4 * D * S * (1 + t)) / Q :=
sorry

end NUMINAMATH_GPT_soda_cans_purchasable_l106_10655


namespace NUMINAMATH_GPT_original_price_of_article_l106_10677

theorem original_price_of_article (x : ℝ) (h : 0.80 * x = 620) : x = 775 := 
by 
  sorry

end NUMINAMATH_GPT_original_price_of_article_l106_10677


namespace NUMINAMATH_GPT_heidi_zoe_paint_fraction_l106_10698

theorem heidi_zoe_paint_fraction (H_period : ℝ) (HZ_period : ℝ) :
  (H_period = 60 → HZ_period = 40 → (8 / 40) = (1 / 5)) :=
by intros H_period_eq HZ_period_eq
   sorry

end NUMINAMATH_GPT_heidi_zoe_paint_fraction_l106_10698


namespace NUMINAMATH_GPT_sin_150_eq_half_l106_10620

theorem sin_150_eq_half : Real.sin (150 * Real.pi / 180) = 1 / 2 := 
by sorry

end NUMINAMATH_GPT_sin_150_eq_half_l106_10620


namespace NUMINAMATH_GPT_minimum_value_of_x_plus_2y_l106_10617

open Real

theorem minimum_value_of_x_plus_2y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 8 / x + 1 / y = 1) : x + 2 * y ≥ 18 := by
  sorry

end NUMINAMATH_GPT_minimum_value_of_x_plus_2y_l106_10617


namespace NUMINAMATH_GPT_total_apples_correct_l106_10629

def cost_per_apple : ℤ := 2
def money_emmy_has : ℤ := 200
def money_gerry_has : ℤ := 100
def total_money : ℤ := money_emmy_has + money_gerry_has
def total_apples_bought : ℤ := total_money / cost_per_apple

theorem total_apples_correct :
    total_apples_bought = 150 := by
    sorry

end NUMINAMATH_GPT_total_apples_correct_l106_10629


namespace NUMINAMATH_GPT_initial_overs_l106_10625

theorem initial_overs (x : ℝ) (r1 : ℝ) (r2 : ℝ) (target : ℝ) (overs_remaining : ℝ) :
  r1 = 3.2 ∧ overs_remaining = 22 ∧ r2 = 11.363636363636363 ∧ target = 282 ∧
  (r1 * x + r2 * overs_remaining = target) → x = 10 :=
by
  intro h
  obtain ⟨hr1, ho, hr2, ht, heq⟩ := h
  sorry

end NUMINAMATH_GPT_initial_overs_l106_10625


namespace NUMINAMATH_GPT_intervals_of_monotonicity_and_extreme_values_l106_10685

noncomputable def f (x : ℝ) : ℝ := x * Real.exp (-x)

theorem intervals_of_monotonicity_and_extreme_values :
  (∀ x : ℝ, x < 1 → deriv f x > 0) ∧
  (∀ x : ℝ, x > 1 → deriv f x < 0) ∧
  (∀ x : ℝ, f 1 = 1 / Real.exp 1) :=
by
  sorry

end NUMINAMATH_GPT_intervals_of_monotonicity_and_extreme_values_l106_10685


namespace NUMINAMATH_GPT_closest_ratio_adults_children_l106_10634

theorem closest_ratio_adults_children 
  (a c : ℕ) 
  (H1 : 30 * a + 15 * c = 2550) 
  (H2 : a > 0) 
  (H3 : c > 0) : 
  (a = 57 ∧ c = 56) ∨ (a = 56 ∧ c = 58) :=
by
  sorry

end NUMINAMATH_GPT_closest_ratio_adults_children_l106_10634


namespace NUMINAMATH_GPT_sqrt_ac_bd_le_sqrt_ef_l106_10657

noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

theorem sqrt_ac_bd_le_sqrt_ef
  (a b c d e f : ℝ)
  (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d ∧ 0 ≤ e ∧ 0 ≤ f)
  (h1 : a + b ≤ e)
  (h2 : c + d ≤ f) :
  sqrt (a * c) + sqrt (b * d) ≤ sqrt (e * f) :=
by
  sorry

end NUMINAMATH_GPT_sqrt_ac_bd_le_sqrt_ef_l106_10657


namespace NUMINAMATH_GPT_quadratic_completion_l106_10604

theorem quadratic_completion (x d e f : ℤ) (h1 : 100*x^2 + 80*x - 144 = 0) (hd : d > 0) 
  (hde : (d * x + e)^2 = f) : d + e + f = 174 :=
sorry

end NUMINAMATH_GPT_quadratic_completion_l106_10604


namespace NUMINAMATH_GPT_value_ranges_l106_10635

theorem value_ranges 
  (a b c : ℝ)
  (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c)
  (h_eq1 : 3 * a + 2 * b + c = 5)
  (h_eq2 : 2 * a + b - 3 * c = 1) :
  (3 / 7 ≤ c ∧ c ≤ 7 / 11) ∧ 
  (-5 / 7 ≤ (3 * a + b - 7 * c) ∧ (3 * a + b - 7 * c) ≤ -1 / 11) :=
by 
  sorry

end NUMINAMATH_GPT_value_ranges_l106_10635


namespace NUMINAMATH_GPT_find_y_l106_10682

open Real

def vecV (y : ℝ) : ℝ × ℝ := (1, y)
def vecW : ℝ × ℝ := (6, 4)

noncomputable def dotProduct (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

noncomputable def projection (v w : ℝ × ℝ) : ℝ × ℝ :=
  let scalar := (dotProduct v w) / (dotProduct w w)
  (scalar * w.1, scalar * w.2)

theorem find_y (y : ℝ) (h : projection (vecV y) vecW = (3, 2)) : y = 5 := by
  sorry

end NUMINAMATH_GPT_find_y_l106_10682


namespace NUMINAMATH_GPT_quadratic_real_roots_k_le_one_fourth_l106_10622

theorem quadratic_real_roots_k_le_one_fourth (k : ℝ) : 
  (∃ x : ℝ, 4 * x^2 - (4 * k - 2) * x + k^2 = 0) ↔ k ≤ 1/4 :=
sorry

end NUMINAMATH_GPT_quadratic_real_roots_k_le_one_fourth_l106_10622


namespace NUMINAMATH_GPT_one_quarter_between_l106_10679

def one_quarter_way (a b : ℚ) : ℚ :=
  a + 1 / 4 * (b - a)

theorem one_quarter_between :
  one_quarter_way (1 / 7) (1 / 4) = 23 / 112 :=
by
  sorry

end NUMINAMATH_GPT_one_quarter_between_l106_10679


namespace NUMINAMATH_GPT_common_intersection_implies_cd_l106_10676

theorem common_intersection_implies_cd (a b c d : ℝ) (h : a ≠ b) (x y : ℝ) 
  (H1 : y = a * x + a) (H2 : y = b * x + b) (H3 : y = c * x + d) : c = d := by
  sorry

end NUMINAMATH_GPT_common_intersection_implies_cd_l106_10676


namespace NUMINAMATH_GPT_solution_set_M_abs_ineq_l106_10600

-- Define the function f
def f (x : ℝ) : ℝ := |x - 3| + |x - 2|

-- Define the set M
def M : Set ℝ := {x | 1 < x ∧ x < 4}

-- The first statement to prove the solution set M for the inequality
theorem solution_set_M : ∀ x, f x < 3 ↔ x ∈ M :=
by sorry

-- The second statement to prove the inequality when a, b ∈ M
theorem abs_ineq (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) : |a + b| < |1 + ab| :=
by sorry

end NUMINAMATH_GPT_solution_set_M_abs_ineq_l106_10600


namespace NUMINAMATH_GPT_sum_of_areas_is_72_l106_10694

def base : ℕ := 2
def length1 : ℕ := 1
def length2 : ℕ := 8
def length3 : ℕ := 27

theorem sum_of_areas_is_72 : base * length1 + base * length2 + base * length3 = 72 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_areas_is_72_l106_10694


namespace NUMINAMATH_GPT_sum_of_roots_eq_two_l106_10643

theorem sum_of_roots_eq_two {b x1 x2 : ℝ} 
  (h : x1 ^ 2 - 2 * x1 + b = 0) 
  (k : x2 ^ 2 - 2 * x2 + b = 0) 
  (neq : x1 ≠ x2) : 
  x1 + x2 = 2 := 
sorry

end NUMINAMATH_GPT_sum_of_roots_eq_two_l106_10643


namespace NUMINAMATH_GPT_fixed_monthly_fee_l106_10681

theorem fixed_monthly_fee :
  ∀ (x y : ℝ), 
  x + y = 20.00 → 
  x + 2 * y = 30.00 → 
  x + 3 * y = 40.00 → 
  x = 10.00 :=
by
  intros x y H1 H2 H3
  -- Proof can be filled out here
  sorry

end NUMINAMATH_GPT_fixed_monthly_fee_l106_10681


namespace NUMINAMATH_GPT_no_naturals_satisfy_m_squared_eq_n_squared_plus_2014_l106_10608

theorem no_naturals_satisfy_m_squared_eq_n_squared_plus_2014 :
  ∀ (m n : ℕ), ¬ (m^2 = n^2 + 2014) :=
by
  intro m n
  sorry

end NUMINAMATH_GPT_no_naturals_satisfy_m_squared_eq_n_squared_plus_2014_l106_10608


namespace NUMINAMATH_GPT_part_one_part_two_l106_10650

noncomputable def M := Set.Ioo (-(1 : ℝ)/2) (1/2)

namespace Problem

variables {a b : ℝ}
def in_M (x : ℝ) := x ∈ M

theorem part_one (ha : in_M a) (hb : in_M b) :
  |(1/3 : ℝ) * a + (1/6) * b| < 1/4 :=
sorry

theorem part_two (ha : in_M a) (hb : in_M b) :
  |1 - 4 * a * b| > 2 * |a - b| :=
sorry

end Problem

end NUMINAMATH_GPT_part_one_part_two_l106_10650


namespace NUMINAMATH_GPT_probability_genuine_given_equal_weight_l106_10680

noncomputable def total_coins : ℕ := 15
noncomputable def genuine_coins : ℕ := 12
noncomputable def counterfeit_coins : ℕ := 3

def condition_A : Prop := true
def condition_B (weights : Fin 6 → ℝ) : Prop :=
  weights 0 + weights 1 = weights 2 + weights 3 ∧
  weights 0 + weights 1 = weights 4 + weights 5

noncomputable def P_A_and_B : ℚ := (44 / 70) * (15 / 26) * (28 / 55)
noncomputable def P_B : ℚ := 44 / 70

theorem probability_genuine_given_equal_weight :
  P_A_and_B / P_B = 264 / 443 :=
by
  sorry

end NUMINAMATH_GPT_probability_genuine_given_equal_weight_l106_10680


namespace NUMINAMATH_GPT_mike_earnings_l106_10601

theorem mike_earnings (total_games non_working_games price_per_game : ℕ) 
  (h1 : total_games = 15) (h2 : non_working_games = 9) (h3 : price_per_game = 5) : 
  total_games - non_working_games * price_per_game = 30 :=
by
  rw [h1, h2, h3]
  show 15 - 9 * 5 = 30
  sorry

end NUMINAMATH_GPT_mike_earnings_l106_10601


namespace NUMINAMATH_GPT_robin_spent_on_leftover_drinks_l106_10658

-- Define the number of each type of drink bought and consumed
def sodas_bought : Nat := 30
def sodas_price : Nat := 2
def sodas_consumed : Nat := 10

def energy_drinks_bought : Nat := 20
def energy_drinks_price : Nat := 3
def energy_drinks_consumed : Nat := 14

def smoothies_bought : Nat := 15
def smoothies_price : Nat := 4
def smoothies_consumed : Nat := 5

-- Define the total cost calculation
def total_spent_on_leftover_drinks : Nat :=
  (sodas_bought * sodas_price - sodas_consumed * sodas_price) +
  (energy_drinks_bought * energy_drinks_price - energy_drinks_consumed * energy_drinks_price) +
  (smoothies_bought * smoothies_price - smoothies_consumed * smoothies_price)

theorem robin_spent_on_leftover_drinks : total_spent_on_leftover_drinks = 98 := by
  -- Provide the proof steps here (not required for this task)
  sorry

end NUMINAMATH_GPT_robin_spent_on_leftover_drinks_l106_10658


namespace NUMINAMATH_GPT_negation_proof_l106_10666

open Classical

variable {x : ℝ}

theorem negation_proof :
  (∀ x : ℝ, (x + 1) ≥ 0 ∧ (x^2 - x) ≤ 0) ↔ ¬ (∃ x_0 : ℝ, (x_0 + 1) < 0 ∨ (x_0^2 - x_0) > 0) := 
by
  sorry

end NUMINAMATH_GPT_negation_proof_l106_10666


namespace NUMINAMATH_GPT_find_number_of_olives_l106_10633

theorem find_number_of_olives (O : ℕ)
  (lettuce_choices : 2 = 2)
  (tomato_choices : 3 = 3)
  (soup_choices : 2 = 2)
  (total_combos : 2 * 3 * O * 2 = 48) :
  O = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_number_of_olives_l106_10633


namespace NUMINAMATH_GPT_triangle_perimeter_ABC_l106_10645

noncomputable def perimeter_triangle (AP PB r : ℕ) (hAP : AP = 23) (hPB : PB = 27) (hr : r = 21) : ℕ :=
  2 * (50 + 245 / 2)

theorem triangle_perimeter_ABC (AP PB r : ℕ) 
  (hAP : AP = 23) 
  (hPB : PB = 27) 
  (hr : r = 21) : 
  perimeter_triangle AP PB r hAP hPB hr = 345 :=
by
  sorry

end NUMINAMATH_GPT_triangle_perimeter_ABC_l106_10645


namespace NUMINAMATH_GPT_golden_ratio_in_range_l106_10687

noncomputable def golden_ratio := (Real.sqrt 5 - 1) / 2

theorem golden_ratio_in_range : 0.6 < golden_ratio ∧ golden_ratio < 0.7 :=
by
  sorry

end NUMINAMATH_GPT_golden_ratio_in_range_l106_10687


namespace NUMINAMATH_GPT_parker_total_weight_l106_10632

def twenty_pound := 20
def thirty_pound := 30
def forty_pound := 40

def first_set_weight := (2 * twenty_pound) + (1 * thirty_pound) + (1 * forty_pound)
def second_set_weight := (1 * twenty_pound) + (2 * thirty_pound) + (2 * forty_pound)
def third_set_weight := (3 * thirty_pound) + (3 * forty_pound)

def total_weight := first_set_weight + second_set_weight + third_set_weight

theorem parker_total_weight :
  total_weight = 480 := by
  sorry

end NUMINAMATH_GPT_parker_total_weight_l106_10632


namespace NUMINAMATH_GPT_worth_of_presents_l106_10614

def ring_value := 4000
def car_value := 2000
def bracelet_value := 2 * ring_value

def total_worth := ring_value + car_value + bracelet_value

theorem worth_of_presents : total_worth = 14000 := by
  sorry

end NUMINAMATH_GPT_worth_of_presents_l106_10614


namespace NUMINAMATH_GPT_f_of_8_l106_10628

variable (f : ℝ → ℝ)

-- Conditions
axiom odd_function : ∀ x : ℝ, f (-x) = -f (x)
axiom function_property : ∀ x : ℝ, f (x + 2) = -1 / f (x)

-- Statement to prove
theorem f_of_8 : f 8 = 0 :=
sorry

end NUMINAMATH_GPT_f_of_8_l106_10628


namespace NUMINAMATH_GPT_solve_for_y_l106_10636

theorem solve_for_y (y : ℕ) (h : 5 * (2^y) = 320) : y = 6 :=
by sorry

end NUMINAMATH_GPT_solve_for_y_l106_10636


namespace NUMINAMATH_GPT_neither_sufficient_nor_necessary_l106_10607

theorem neither_sufficient_nor_necessary (x : ℝ) : 
  ¬(-1 < x ∧ x < 2 → |x - 2| < 1) ∧ ¬(|x - 2| < 1 → -1 < x ∧ x < 2) :=
by
  sorry

end NUMINAMATH_GPT_neither_sufficient_nor_necessary_l106_10607


namespace NUMINAMATH_GPT_find_d_l106_10653

-- Definitions for the functions f and g
def f (x : ℝ) (c : ℝ) : ℝ := 5 * x + c
def g (x : ℝ) (c : ℝ) : ℝ := c * x + 3

-- Statement to prove the value of d
theorem find_d (c d : ℝ) (h1 : ∀ x : ℝ, f (g x c) c = 15 * x + d) : d = 18 :=
by
  -- inserting custom logic for proof
  sorry

end NUMINAMATH_GPT_find_d_l106_10653


namespace NUMINAMATH_GPT_inequality_proof_l106_10638

variable (a b c : ℝ)
variable (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
variable (h_abc : a * b * c = 1)

theorem inequality_proof :
  (1 / (a ^ 3 * (b + c)) + 1 / (b ^ 3 * (c + a)) + 1 / (c ^ 3 * (a + b))) 
  ≥ (3 / 2) + (1 / 4) * (a * (c - b) ^ 2 / (c + b) + b * (c - a) ^ 2 / (c + a) + c * (b - a) ^ 2 / (b + a)) :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l106_10638


namespace NUMINAMATH_GPT_Josanna_min_avg_score_l106_10695

theorem Josanna_min_avg_score (scores : List ℕ) (cur_avg target_avg : ℚ)
  (next_test_bonus : ℚ) (additional_avg_points : ℚ) : ℚ :=
  let cur_avg := (92 + 81 + 75 + 65 + 88) / 5
  let target_avg := cur_avg + 6
  let needed_total := target_avg * 7
  let additional_points := 401 + 5
  let needed_sum := needed_total - additional_points
  needed_sum / 2

noncomputable def min_avg_score : ℚ :=
  Josanna_min_avg_score [92, 81, 75, 65, 88] 80.2 86.2 5 6

example : min_avg_score = 99 :=
by
  sorry

end NUMINAMATH_GPT_Josanna_min_avg_score_l106_10695


namespace NUMINAMATH_GPT_katie_new_games_l106_10678

theorem katie_new_games (K : ℕ) (h : K + 8 = 92) : K = 84 :=
by
  sorry

end NUMINAMATH_GPT_katie_new_games_l106_10678


namespace NUMINAMATH_GPT_Jane_mom_jars_needed_l106_10673

theorem Jane_mom_jars_needed : 
  ∀ (total_tomatoes jar_capacity : ℕ), 
  total_tomatoes = 550 → 
  jar_capacity = 14 → 
  ⌈(total_tomatoes: ℚ) / jar_capacity⌉ = 40 := 
by 
  intros total_tomatoes jar_capacity htotal hcapacity
  sorry

end NUMINAMATH_GPT_Jane_mom_jars_needed_l106_10673


namespace NUMINAMATH_GPT_methane_needed_l106_10639

theorem methane_needed (total_benzene_g : ℝ) (molar_mass_benzene : ℝ) (toluene_moles : ℝ) : 
  total_benzene_g = 156 ∧ molar_mass_benzene = 78 ∧ toluene_moles = 2 → 
  toluene_moles = total_benzene_g / molar_mass_benzene := 
by
  intros
  sorry

end NUMINAMATH_GPT_methane_needed_l106_10639


namespace NUMINAMATH_GPT_arctan_sum_is_pi_over_4_l106_10696

open Real

theorem arctan_sum_is_pi_over_4 (a b c : ℝ) (h1 : b = c) (h2 : c / (a + b) + a / (b + c) = 1) :
  arctan (c / (a + b)) + arctan (a / (b + c)) = π / 4 :=
by 
  sorry

end NUMINAMATH_GPT_arctan_sum_is_pi_over_4_l106_10696


namespace NUMINAMATH_GPT_butterflies_equal_distribution_l106_10659

theorem butterflies_equal_distribution (N : ℕ) : (∃ t : ℕ, 
    (N - t) % 8 = 0 ∧ (N - t) / 8 > 0) ↔ ∃ k : ℕ, N = 45 * k :=
by sorry

end NUMINAMATH_GPT_butterflies_equal_distribution_l106_10659


namespace NUMINAMATH_GPT_tan_945_equals_1_l106_10605

noncomputable def tan_circular (x : ℝ) : ℝ := Real.tan x

theorem tan_945_equals_1 :
  tan_circular 945 = 1 := 
by
  sorry

end NUMINAMATH_GPT_tan_945_equals_1_l106_10605


namespace NUMINAMATH_GPT_sequence_properties_l106_10699

noncomputable def arithmetic_sequence (a1 d : ℤ) (n : ℕ) : ℤ :=
a1 + d * (n - 1)

theorem sequence_properties (d a1 : ℤ) (h_d_ne_zero : d ≠ 0)
(h1 : arithmetic_sequence a1 d 2 + arithmetic_sequence a1 d 4 = 10)
(h2 : (arithmetic_sequence a1 d 2)^2 = (arithmetic_sequence a1 d 1) * (arithmetic_sequence a1 d 5)) :
a1 = 1 ∧ ∀ n : ℕ, n > 0 → arithmetic_sequence 1 2 n = 2 * n - 1 :=
by
  sorry

end NUMINAMATH_GPT_sequence_properties_l106_10699


namespace NUMINAMATH_GPT_machine_a_sprockets_per_hour_l106_10606

theorem machine_a_sprockets_per_hour (s h : ℝ)
    (H1 : 1.1 * s * h = 550)
    (H2 : s * (h + 10) = 550) : s = 5 := by
  sorry

end NUMINAMATH_GPT_machine_a_sprockets_per_hour_l106_10606


namespace NUMINAMATH_GPT_sum_of_primes_product_166_l106_10669

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m < n → m > 0 → n % m ≠ 0

theorem sum_of_primes_product_166
    (p1 p2 : ℕ)
    (prime_p1 : is_prime p1)
    (prime_p2 : is_prime p2)
    (product_condition : p1 * p2 = 166) :
    p1 + p2 = 85 :=
    sorry

end NUMINAMATH_GPT_sum_of_primes_product_166_l106_10669


namespace NUMINAMATH_GPT_boxes_needed_l106_10692

-- Define Marilyn's total number of bananas
def num_bananas : Nat := 40

-- Define the number of bananas per box
def bananas_per_box : Nat := 5

-- Calculate the number of boxes required for the given number of bananas and bananas per box
def num_boxes (total_bananas : Nat) (bananas_each_box : Nat) : Nat :=
  total_bananas / bananas_each_box

-- Statement to be proved: given the specific conditions, the result should be 8
theorem boxes_needed : num_boxes num_bananas bananas_per_box = 8 :=
sorry

end NUMINAMATH_GPT_boxes_needed_l106_10692


namespace NUMINAMATH_GPT_difference_xy_l106_10603

theorem difference_xy (x y : ℝ) (h1 : x + y = 9) (h2 : x^2 - y^2 = 27) : x - y = 3 := sorry

end NUMINAMATH_GPT_difference_xy_l106_10603


namespace NUMINAMATH_GPT_find_m_if_f_even_l106_10690

variable (m : ℝ)

def f (x : ℝ) : ℝ := x^2 + (m + 2) * x + 3

theorem find_m_if_f_even (h : ∀ x, f m x = f m (-x)) : m = -2 :=
by
  sorry

end NUMINAMATH_GPT_find_m_if_f_even_l106_10690


namespace NUMINAMATH_GPT_steve_needs_28_feet_of_wood_l106_10646

-- Define the required lengths
def lengths_4_feet : Nat := 6
def lengths_2_feet : Nat := 2

-- Define the wood length in feet for each type
def wood_length_4 : Nat := 4
def wood_length_2 : Nat := 2

-- Total feet of wood required
def total_wood : Nat := lengths_4_feet * wood_length_4 + lengths_2_feet * wood_length_2

-- The theorem to prove that the total amount of wood required is 28 feet
theorem steve_needs_28_feet_of_wood : total_wood = 28 :=
by
  sorry

end NUMINAMATH_GPT_steve_needs_28_feet_of_wood_l106_10646


namespace NUMINAMATH_GPT_bobby_total_l106_10651

-- Define the conditions
def initial_candy : ℕ := 33
def additional_candy : ℕ := 4
def chocolate : ℕ := 14

-- Define the total pieces of candy Bobby ate
def total_candy : ℕ := initial_candy + additional_candy

-- Define the total pieces of candy and chocolate Bobby ate
def total_candy_and_chocolate : ℕ := total_candy + chocolate

-- Theorem to prove the total pieces of candy and chocolate Bobby ate
theorem bobby_total : total_candy_and_chocolate = 51 :=
by sorry

end NUMINAMATH_GPT_bobby_total_l106_10651


namespace NUMINAMATH_GPT_vectors_parallel_x_eq_four_l106_10637

theorem vectors_parallel_x_eq_four (x : ℝ) :
  (x > 0) →
  (∃ k : ℝ, (8 + 1/2 * x, x) = k • (x + 1, 2)) →
  x = 4 :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_vectors_parallel_x_eq_four_l106_10637


namespace NUMINAMATH_GPT_original_number_of_matchsticks_l106_10671

-- Define the conditions
def matchsticks_per_house : ℕ := 10
def houses_created : ℕ := 30
def total_matchsticks_used := houses_created * matchsticks_per_house

-- Define the question and the proof goal
theorem original_number_of_matchsticks (h : total_matchsticks_used = (Michael's_original_matchsticks / 2)) :
  (Michael's_original_matchsticks = 600) :=
by
  sorry

end NUMINAMATH_GPT_original_number_of_matchsticks_l106_10671


namespace NUMINAMATH_GPT_count_integers_satisfying_inequality_l106_10619

theorem count_integers_satisfying_inequality :
  ∃ (S : Finset ℤ), S.card = 8 ∧ ∀ n ∈ S, -11 ≤ n ∧ n ≤ 11 ∧ (n - 2) * (n + 4) * (n + 8) < 0 :=
by
  sorry

end NUMINAMATH_GPT_count_integers_satisfying_inequality_l106_10619


namespace NUMINAMATH_GPT_imaginary_part_of_z_l106_10686

def z : ℂ := 1 - 2 * Complex.I

theorem imaginary_part_of_z : Complex.im z = -2 := by
  sorry

end NUMINAMATH_GPT_imaginary_part_of_z_l106_10686


namespace NUMINAMATH_GPT_angle_B_l106_10647

theorem angle_B (a b c A B : ℝ) (h : a * Real.cos B - b * Real.cos A = c) (C : ℝ) (hC : C = Real.pi / 5) (h_triangle : A + B + C = Real.pi) : B = 3 * Real.pi / 10 :=
sorry

end NUMINAMATH_GPT_angle_B_l106_10647


namespace NUMINAMATH_GPT_solution_set_max_value_l106_10649

-- Given function f(x)
def f (x : ℝ) : ℝ := |2 * x - 1| + |x - 1|

-- (I) Prove the solution set of f(x) ≤ 4 is {x | -2/3 ≤ x ≤ 2}
theorem solution_set : {x : ℝ | f x ≤ 4} = {x : ℝ | -2/3 ≤ x ∧ x ≤ 2} :=
sorry

-- (II) Given m is the minimum value of f(x)
def m := 1 / 2

-- Given a, b, c ∈ ℝ^+ and a + b + c = m
variables (a b c : ℝ)
variable (h1 : 0 < a ∧ 0 < b ∧ 0 < c)
variable (h2 : a + b + c = m)

-- Prove the maximum value of √(2a + 1) + √(2b + 1) + √(2c + 1) is 2√3
theorem max_value : (Real.sqrt (2 * a + 1) + Real.sqrt (2 * b + 1) + Real.sqrt (2 * c + 1)) ≤ 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_solution_set_max_value_l106_10649


namespace NUMINAMATH_GPT_total_travel_time_l106_10613

theorem total_travel_time (subway_time : ℕ) (train_multiplier : ℕ) (bike_time : ℕ) 
  (h_subway : subway_time = 10) 
  (h_train_multiplier : train_multiplier = 2) 
  (h_bike : bike_time = 8) : 
  subway_time + train_multiplier * subway_time + bike_time = 38 :=
by
  sorry

end NUMINAMATH_GPT_total_travel_time_l106_10613


namespace NUMINAMATH_GPT_mul_72518_9999_eq_725107482_l106_10674

theorem mul_72518_9999_eq_725107482 : 72518 * 9999 = 725107482 := by
  sorry

end NUMINAMATH_GPT_mul_72518_9999_eq_725107482_l106_10674


namespace NUMINAMATH_GPT_degree_f_x2_g_x3_l106_10689

open Polynomial

noncomputable def degree_of_composite_polynomials (f g : Polynomial ℝ) : ℕ :=
  let f_degree := Polynomial.degree f
  let g_degree := Polynomial.degree g
  match (f_degree, g_degree) with
  | (some 3, some 6) => 24
  | _ => 0

theorem degree_f_x2_g_x3 (f g : Polynomial ℝ) (h_f : Polynomial.degree f = 3) (h_g : Polynomial.degree g = 6) :
  Polynomial.degree (Polynomial.comp f (X^2) * Polynomial.comp g (X^3)) = 24 := by
  -- content Logic Here
  sorry

end NUMINAMATH_GPT_degree_f_x2_g_x3_l106_10689


namespace NUMINAMATH_GPT_max_product_sum_300_l106_10641

theorem max_product_sum_300 : ∃ (x : ℤ), x * (300 - x) = 22500 :=
by
  sorry

end NUMINAMATH_GPT_max_product_sum_300_l106_10641


namespace NUMINAMATH_GPT_solve_equation_l106_10660

theorem solve_equation (Y : ℝ) : (3.242 * 10 * Y) / 100 = 0.3242 * Y := 
by 
  sorry

end NUMINAMATH_GPT_solve_equation_l106_10660


namespace NUMINAMATH_GPT_range_of_q_l106_10615

variable (x : ℝ)

def q (x : ℝ) := (3 * x^2 + 1)^2

theorem range_of_q : ∀ y, (∃ x : ℝ, x ≥ 0 ∧ y = q x) ↔ y ≥ 1 := by
  sorry

end NUMINAMATH_GPT_range_of_q_l106_10615


namespace NUMINAMATH_GPT_smallest_N_is_14_l106_10621

-- Definition of depicted number and cyclic arrangement
def depicted_number : Type := List (Fin 2) -- Depicted numbers are lists of digits (0 corresponds to 1, 1 corresponds to 2)

-- A condition representing the function that checks if a list contains all possible four-digit combinations
def contains_all_four_digit_combinations (arr: List (Fin 2)) : Prop :=
  ∀ (seq: List (Fin 2)), seq.length = 4 → seq ⊆ arr

-- The problem statement: find the smallest N where an arrangement contains all four-digit combinations
def smallest_N (N: Nat) (arr: List (Fin 2)) : Prop :=
  N = arr.length ∧ contains_all_four_digit_combinations arr

theorem smallest_N_is_14 : ∃ (N : Nat) (arr: List (Fin 2)), smallest_N N arr ∧ N = 14 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_smallest_N_is_14_l106_10621


namespace NUMINAMATH_GPT_find_n_divisible_by_11_l106_10688

theorem find_n_divisible_by_11 : ∃ n : ℕ, 0 < n ∧ n < 11 ∧ (18888 - n) % 11 = 0 :=
by
  use 1
  -- proof steps would go here, but we're only asked for the statement
  sorry

end NUMINAMATH_GPT_find_n_divisible_by_11_l106_10688


namespace NUMINAMATH_GPT_max_value_of_x0_l106_10661

noncomputable def sequence_max_value (seq : Fin 1996 → ℝ) (pos_seq : ∀ i, seq i > 0) : Prop :=
  seq 0 = seq 1995 ∧
  (∀ i : Fin 1995, seq i + 2 / seq i = 2 * seq (i + 1) + 1 / seq (i + 1)) ∧
  (seq 0 ≤ 2^997)

theorem max_value_of_x0 :
  ∃ seq : Fin 1996 → ℝ, ∀ pos_seq : ∀ i, seq i > 0, sequence_max_value seq pos_seq :=
sorry

end NUMINAMATH_GPT_max_value_of_x0_l106_10661


namespace NUMINAMATH_GPT_find_angle_measure_l106_10644

theorem find_angle_measure (x : ℝ) (h : x = 2 * (90 - x) + 30) : x = 70 :=
by
  exact sorry

end NUMINAMATH_GPT_find_angle_measure_l106_10644


namespace NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l106_10693

theorem problem1 : 
  (3 / 5 : ℚ) - ((2 / 15) + (1 / 3)) = (2 / 15) := 
  by 
  sorry

theorem problem2 : 
  (-2 : ℤ) - 12 * ((1 / 3 : ℚ) - (1 / 4 : ℚ) + (1 / 2 : ℚ)) = -8 := 
  by 
  sorry

theorem problem3 : 
  (2 : ℤ) * (-3) ^ 2 - (6 / (-2) : ℚ) * (-1 / 3) = 17 := 
  by 
  sorry

theorem problem4 : 
  (-1 ^ 4 : ℤ) + ((abs (2 ^ 3 - 10)) : ℤ) - ((-3 : ℤ) / (-1) ^ 2019) = -2 := 
  by 
  sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l106_10693


namespace NUMINAMATH_GPT_total_distance_covered_l106_10627

theorem total_distance_covered :
  let t1 := 30 / 60 -- time in hours for first walking session
  let s1 := 3       -- speed in mph for first walking session
  let t2 := 20 / 60 -- time in hours for running session
  let s2 := 8       -- speed in mph for running session
  let t3 := 10 / 60 -- time in hours for second walking session
  let s3 := 2       -- speed in mph for second walking session
  let d1 := s1 * t1 -- distance for first walking session
  let d2 := s2 * t2 -- distance for running session
  let d3 := s3 * t3 -- distance for second walking session
  d1 + d2 + d3 = 4.5 :=
by
  sorry

end NUMINAMATH_GPT_total_distance_covered_l106_10627


namespace NUMINAMATH_GPT_perimeter_of_square_l106_10630

theorem perimeter_of_square (s : ℝ) (h : s^2 = 588) : (4 * s) = 56 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_perimeter_of_square_l106_10630


namespace NUMINAMATH_GPT_find_difference_l106_10631

theorem find_difference (P : ℝ) (hP : P > 150) :
  let q := P - 150
  let A := 0.2 * P
  let B := 40
  let C := 0.3 * q
  ∃ w z, (0.2 * (150 + 50) >= B) ∧ (30 + 0.2 * q >= 0.3 * q) ∧ 150 + 50 = w ∧ 150 + 300 = z ∧ z - w = 250 :=
by
  sorry

end NUMINAMATH_GPT_find_difference_l106_10631
