import Mathlib

namespace NUMINAMATH_GPT_initial_mat_weavers_l929_92937

variable (num_weavers : ℕ) (rate : ℕ → ℕ → ℕ) -- rate weaver_count duration_in_days → mats_woven

-- Given Conditions
def condition1 := rate num_weavers 4 = 4
def condition2 := rate (2 * num_weavers) 8 = 16

-- Theorem to Prove
theorem initial_mat_weavers : num_weavers = 4 :=
by
  sorry

end NUMINAMATH_GPT_initial_mat_weavers_l929_92937


namespace NUMINAMATH_GPT_ab_ac_plus_bc_range_l929_92951

theorem ab_ac_plus_bc_range (a b c : ℝ) (h : a + b + 2 * c = 0) :
  ∃ (k : ℝ), k ≤ 0 ∧ k = ab + ac + bc :=
sorry

end NUMINAMATH_GPT_ab_ac_plus_bc_range_l929_92951


namespace NUMINAMATH_GPT_second_spray_kill_percent_l929_92948

-- Conditions
def first_spray_kill_percent : ℝ := 50
def both_spray_kill_percent : ℝ := 5
def germs_left_after_both : ℝ := 30

-- Lean 4 statement
theorem second_spray_kill_percent (x : ℝ) 
  (H : 100 - (first_spray_kill_percent + x - both_spray_kill_percent) = germs_left_after_both) :
  x = 15 :=
by
  sorry

end NUMINAMATH_GPT_second_spray_kill_percent_l929_92948


namespace NUMINAMATH_GPT_a2018_is_4035_l929_92987

noncomputable def f : ℝ → ℝ := sorry
def a (n : ℕ) : ℝ := sorry

axiom domain : ∀ x : ℝ, true 
axiom condition_2 : ∀ x : ℝ, x < 0 → f x > 1
axiom condition_3 : ∀ x y : ℝ, f x * f y = f (x + y)
axiom sequence_def : ∀ n : ℕ, n > 0 → a 1 = f 0 ∧ f (a (n + 1)) = 1 / f (-2 - a n)

theorem a2018_is_4035 : a 2018 = 4035 :=
sorry

end NUMINAMATH_GPT_a2018_is_4035_l929_92987


namespace NUMINAMATH_GPT_first_dilution_volume_l929_92980

theorem first_dilution_volume (x : ℝ) (V : ℝ) (red_factor : ℝ) (p : ℝ) :
  V = 1000 →
  red_factor = 25 / 3 →
  (1000 - 2 * x) * (1000 - x) = 1000 * 1000 * (3 / 25) →
  x = 400 :=
by
  intros hV hred hf
  sorry

end NUMINAMATH_GPT_first_dilution_volume_l929_92980


namespace NUMINAMATH_GPT_equation_of_ellipse_AN_BM_constant_l929_92952

noncomputable def a := 2
noncomputable def b := 1
noncomputable def e := (Real.sqrt 3) / 2
noncomputable def c := Real.sqrt 3

def ellipse (x y : ℝ) : Prop := (x^2 / 4) + y^2 = 1

theorem equation_of_ellipse :
  ellipse a b
:=
by
  sorry

theorem AN_BM_constant (x0 y0 : ℝ) (hx : x0^2 + 4 * y0^2 = 4) :
  let AN := 2 + x0 / (y0 - 1)
  let BM := 1 + 2 * y0 / (x0 - 2)
  abs (AN * BM) = 4
:=
by
  sorry

end NUMINAMATH_GPT_equation_of_ellipse_AN_BM_constant_l929_92952


namespace NUMINAMATH_GPT_smallest_12_digit_proof_l929_92982

def is_12_digit_number (n : ℕ) : Prop :=
  n >= 10^11 ∧ n < 10^12

def contains_each_digit_0_to_9 (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] → d ∈ n.digits 10

def is_divisible_by_36 (n : ℕ) : Prop :=
  n % 36 = 0

noncomputable def smallest_12_digit_divisible_by_36_and_contains_each_digit : ℕ :=
  100023457896

theorem smallest_12_digit_proof :
  is_12_digit_number smallest_12_digit_divisible_by_36_and_contains_each_digit ∧
  contains_each_digit_0_to_9 smallest_12_digit_divisible_by_36_and_contains_each_digit ∧
  is_divisible_by_36 smallest_12_digit_divisible_by_36_and_contains_each_digit ∧
  ∀ m : ℕ, is_12_digit_number m ∧ contains_each_digit_0_to_9 m ∧ is_divisible_by_36 m →
  m >= smallest_12_digit_divisible_by_36_and_contains_each_digit :=
by
  sorry

end NUMINAMATH_GPT_smallest_12_digit_proof_l929_92982


namespace NUMINAMATH_GPT_ratio_of_larger_to_smaller_l929_92957

theorem ratio_of_larger_to_smaller (x y : ℝ) (h1 : x > y) (h2 : x + y = 7 * (x - y)) : x / y = 2 :=
sorry

end NUMINAMATH_GPT_ratio_of_larger_to_smaller_l929_92957


namespace NUMINAMATH_GPT_interval_of_decrease_l929_92976

noncomputable def f : ℝ → ℝ := fun x => x^2 - 2 * x

theorem interval_of_decrease : 
  ∃ a b : ℝ, a = -2 ∧ b = 1 ∧ ∀ x1 x2 : ℝ, a ≤ x1 ∧ x1 < x2 ∧ x2 ≤ b → f x1 ≥ f x2 :=
by 
  use -2, 1
  sorry

end NUMINAMATH_GPT_interval_of_decrease_l929_92976


namespace NUMINAMATH_GPT_seven_whole_numbers_cannot_form_ten_consecutive_pairwise_sums_l929_92970

theorem seven_whole_numbers_cannot_form_ten_consecutive_pairwise_sums
  (a1 a2 a3 a4 a5 a6 a7 : Nat) :
  ¬ ∃ (s : Finset Nat), (s = {a1 + a2, a1 + a3, a1 + a4, a1 + a5, a1 + a6, a1 + a7,
                             a2 + a3, a2 + a4, a2 + a5, a2 + a6, a2 + a7,
                             a3 + a4, a3 + a5, a3 + a6, a3 + a7,
                             a4 + a5, a4 + a6, a4 + a7,
                             a5 + a6, a5 + a7,
                             a6 + a7}) ∧
  (∃ (n : Nat), s = {n, n+1, n+2, n+3, n+4, n+5, n+6, n+7, n+8, n+9}) := 
sorry

end NUMINAMATH_GPT_seven_whole_numbers_cannot_form_ten_consecutive_pairwise_sums_l929_92970


namespace NUMINAMATH_GPT_box_height_is_55_cm_l929_92964

noncomputable def height_of_box 
  (ceiling_height_m : ℝ) 
  (light_fixture_below_ceiling_cm : ℝ) 
  (bob_height_m : ℝ) 
  (bob_reach_cm : ℝ) 
  : ℝ :=
  let ceiling_height_cm := ceiling_height_m * 100
  let bob_height_cm := bob_height_m * 100
  let light_fixture_from_floor := ceiling_height_cm - light_fixture_below_ceiling_cm
  let bob_total_reach := bob_height_cm + bob_reach_cm
  light_fixture_from_floor - bob_total_reach

-- Theorem statement
theorem box_height_is_55_cm 
  (ceiling_height_m : ℝ) 
  (light_fixture_below_ceiling_cm : ℝ) 
  (bob_height_m : ℝ) 
  (bob_reach_cm : ℝ) 
  (h : height_of_box ceiling_height_m light_fixture_below_ceiling_cm bob_height_m bob_reach_cm = 55) 
  : height_of_box 3 15 1.8 50 = 55 :=
by
  unfold height_of_box
  sorry

end NUMINAMATH_GPT_box_height_is_55_cm_l929_92964


namespace NUMINAMATH_GPT_not_possible_to_form_triangle_l929_92939

-- Define the conditions
variables (a : ℝ)

-- State the problem in Lean 4
theorem not_possible_to_form_triangle (h : a > 0) :
  ¬ (a + a > 2 * a ∧ a + 2 * a > a ∧ a + 2 * a > a) :=
by
  sorry

end NUMINAMATH_GPT_not_possible_to_form_triangle_l929_92939


namespace NUMINAMATH_GPT_paul_erasers_l929_92916

theorem paul_erasers (E : ℕ) (E_crayons : E + 353 = 391) : E = 38 := 
by
  sorry

end NUMINAMATH_GPT_paul_erasers_l929_92916


namespace NUMINAMATH_GPT_hexagon_equilateral_triangles_l929_92930

theorem hexagon_equilateral_triangles (hexagon_area: ℝ) (num_hexagons : ℕ) (tri_area: ℝ) 
    (h1 : hexagon_area = 6) (h2 : num_hexagons = 4) (h3 : tri_area = 4) : 
    ∃ (num_triangles : ℕ), num_triangles = 8 := 
by
  sorry

end NUMINAMATH_GPT_hexagon_equilateral_triangles_l929_92930


namespace NUMINAMATH_GPT_domain_of_function_l929_92917

noncomputable def domain_is_valid (x z : ℝ) : Prop :=
  1 < x ∧ x < 2 ∧ (|x| - z) ≠ 0

theorem domain_of_function (x z : ℝ) : domain_is_valid x z :=
by
  sorry

end NUMINAMATH_GPT_domain_of_function_l929_92917


namespace NUMINAMATH_GPT_smallest_number_divisibility_l929_92932

theorem smallest_number_divisibility :
  ∃ x, (x + 3) % 70 = 0 ∧ (x + 3) % 100 = 0 ∧ (x + 3) % 84 = 0 ∧ x = 6303 :=
sorry

end NUMINAMATH_GPT_smallest_number_divisibility_l929_92932


namespace NUMINAMATH_GPT_sum_of_legs_of_larger_triangle_l929_92958

theorem sum_of_legs_of_larger_triangle (area_small : ℝ) (area_large : ℝ) (hypotenuse_small : ℝ) :
    (area_small = 8 ∧ area_large = 200 ∧ hypotenuse_small = 6) →
    ∃ sum_of_legs : ℝ, sum_of_legs = 41.2 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_legs_of_larger_triangle_l929_92958


namespace NUMINAMATH_GPT_renata_lottery_winnings_l929_92905

def initial_money : ℕ := 10
def donation : ℕ := 4
def prize_won : ℕ := 90
def water_cost : ℕ := 1
def lottery_ticket_cost : ℕ := 1
def final_money : ℕ := 94

theorem renata_lottery_winnings :
  ∃ (lottery_winnings : ℕ), 
  initial_money - donation + prize_won 
  - water_cost - lottery_ticket_cost 
  = final_money ∧ 
  lottery_winnings = 2 :=
by
  -- Proof steps will go here
  sorry

end NUMINAMATH_GPT_renata_lottery_winnings_l929_92905


namespace NUMINAMATH_GPT_find_f_37_5_l929_92936

noncomputable def f (x : ℝ) : ℝ := sorry

/--
Given that \( f \) is an odd function defined on \( \mathbb{R} \) and satisfies
\( f(x+2) = -f(x) \). When \( 0 \leqslant x \leqslant 1 \), \( f(x) = x \),
prove that \( f(37.5) = 0.5 \).
-/
theorem find_f_37_5 (f : ℝ → ℝ) (odd_f : ∀ x : ℝ, f (-x) = -f x)
  (periodic_f : ∀ x : ℝ, f (x + 2) = -f x)
  (interval_f : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = x) : f 37.5 = 0.5 :=
sorry

end NUMINAMATH_GPT_find_f_37_5_l929_92936


namespace NUMINAMATH_GPT_min_value_l929_92910

variables (a b c : ℝ)
variable (hpos : a > 0 ∧ b > 0 ∧ c > 0)
variable (hsum : a + b + c = 1)

theorem min_value (hpos : a > 0 ∧ b > 0 ∧ c > 0) (hsum : a + b + c = 1) :
  9 * a^2 + 4 * b^2 + (1/4) * c^2 = 36 / 157 := 
sorry

end NUMINAMATH_GPT_min_value_l929_92910


namespace NUMINAMATH_GPT_ratio_eliminated_to_remaining_l929_92959

theorem ratio_eliminated_to_remaining (initial_racers : ℕ) (final_racers : ℕ)
  (eliminations_1st_segment : ℕ) (eliminations_2nd_segment : ℕ) :
  initial_racers = 100 →
  final_racers = 30 →
  eliminations_1st_segment = 10 →
  eliminations_2nd_segment = initial_racers - eliminations_1st_segment - (initial_racers - eliminations_1st_segment) / 3 - final_racers →
  (eliminations_2nd_segment / (initial_racers - eliminations_1st_segment - (initial_racers - eliminations_1st_segment) / 3)) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_eliminated_to_remaining_l929_92959


namespace NUMINAMATH_GPT_min_value_on_neg_infinite_l929_92933

def odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f x

def max_value_on_interval (F : ℝ → ℝ) (a b : ℝ) (max_val : ℝ) : Prop :=
∀ x, (0 < x → F x ≤ max_val) ∧ (∃ y, 0 < y ∧ F y = max_val)

theorem min_value_on_neg_infinite (f g : ℝ → ℝ) (a b : ℝ) (F : ℝ → ℝ)
  (h_odd_f : odd_function f) (h_odd_g : odd_function g)
  (h_def_F : ∀ x, F x = a * f x + b * g x + 2)
  (h_max_F_on_0_inf : max_value_on_interval F a b 8) :
  ∃ x, x < 0 ∧ F x = -4 :=
sorry

end NUMINAMATH_GPT_min_value_on_neg_infinite_l929_92933


namespace NUMINAMATH_GPT_cost_of_new_shoes_l929_92960

theorem cost_of_new_shoes :
  ∃ P : ℝ, P = 32 ∧ (P / 2 = 14.50 + 0.10344827586206897 * 14.50) :=
sorry

end NUMINAMATH_GPT_cost_of_new_shoes_l929_92960


namespace NUMINAMATH_GPT_possible_scenario_l929_92907

variable {a b c d : ℝ}

-- Conditions
def abcd_positive : a * b * c * d > 0 := sorry
def a_less_than_c : a < c := sorry
def bcd_negative : b * c * d < 0 := sorry

-- Statement
theorem possible_scenario :
  (a < 0) ∧ (b > 0) ∧ (c < 0) ∧ (d > 0) :=
sorry

end NUMINAMATH_GPT_possible_scenario_l929_92907


namespace NUMINAMATH_GPT_opposite_vertices_equal_l929_92913

-- Define the angles of a regular convex hexagon
variables {α β γ δ ε ζ : ℝ}

-- Regular hexagon condition: The sum of the alternating angles
axiom angle_sum_condition :
  α + γ + ε = β + δ + ε

-- Define the final theorem to prove that the opposite vertices have equal angles
theorem opposite_vertices_equal (h : α + γ + ε = β + δ + ε) :
  α = δ ∧ β = ε ∧ γ = ζ :=
sorry

end NUMINAMATH_GPT_opposite_vertices_equal_l929_92913


namespace NUMINAMATH_GPT_max_sum_of_squares_diff_l929_92950

theorem max_sum_of_squares_diff {x y : ℕ} (h : x > 0 ∧ y > 0) (h_diff : x^2 - y^2 = 2016) :
  x + y ≤ 1008 ∧ ∃ x' y' : ℕ, x'^2 - y'^2 = 2016 ∧ x' + y' = 1008 :=
sorry

end NUMINAMATH_GPT_max_sum_of_squares_diff_l929_92950


namespace NUMINAMATH_GPT_intersection_is_3_l929_92944

def setA : Set ℕ := {5, 2, 3}
def setB : Set ℕ := {9, 3, 6}

theorem intersection_is_3 : setA ∩ setB = {3} := by
  sorry

end NUMINAMATH_GPT_intersection_is_3_l929_92944


namespace NUMINAMATH_GPT_cat_food_inequality_l929_92953

theorem cat_food_inequality (B S : ℝ) (h1 : B > S) (h2 : B < 2 * S) : 4 * B + 4 * S < 3 * (B + 2 * S) :=
sorry

end NUMINAMATH_GPT_cat_food_inequality_l929_92953


namespace NUMINAMATH_GPT_flavored_drink_ratio_l929_92989

theorem flavored_drink_ratio :
  ∃ (F C W: ℚ), F / C = 1 / 7.5 ∧ F / W = 1 / 56.25 ∧ C/W = 6/90 ∧ F / C / 3 = ((F / W) * 2)
:= sorry

end NUMINAMATH_GPT_flavored_drink_ratio_l929_92989


namespace NUMINAMATH_GPT_math_problem_l929_92981

variable {x p q r : ℝ}

-- Conditions and Theorem
theorem math_problem (h1 : ∀ x, (x ≤ -5 ∨ 20 ≤ x ∧ x ≤ 30) ↔ (0 ≤ (x - p) * (x - q) / (x - r)))
  (h2 : p < q) : p + 2 * q + 3 * r = 65 := 
sorry

end NUMINAMATH_GPT_math_problem_l929_92981


namespace NUMINAMATH_GPT_y_affected_by_other_factors_l929_92998

-- Given the linear regression model
def linear_regression_model (b a e x : ℝ) : ℝ := b * x + a + e

-- Theorem: Prove that the dependent variable \( y \) may be affected by factors other than the independent variable \( x \)
theorem y_affected_by_other_factors (b a e x : ℝ) :
  ∃ y, (y = linear_regression_model b a e x ∧ e ≠ 0) :=
sorry

end NUMINAMATH_GPT_y_affected_by_other_factors_l929_92998


namespace NUMINAMATH_GPT_similar_right_triangles_hypotenuse_relation_similar_right_triangles_reciprocal_relation_l929_92985

variable {a b c m_c a' b' c' m_c' : ℝ}

/- The first proof problem -/
theorem similar_right_triangles_hypotenuse_relation (h_sim : (a = k * a') ∧ (b = k * b') ∧ (c = k * c')) :
  a * a' + b * b' = c * c' := by
  sorry

/- The second proof problem -/
theorem similar_right_triangles_reciprocal_relation (h_sim : (a = k * a') ∧ (b = k * b') ∧ (c = k * c') ∧ (m_c = k * m_c')) :
  (1 / (a * a') + 1 / (b * b')) = 1 / (m_c * m_c') := by
  sorry

end NUMINAMATH_GPT_similar_right_triangles_hypotenuse_relation_similar_right_triangles_reciprocal_relation_l929_92985


namespace NUMINAMATH_GPT_interval_sum_l929_92906

theorem interval_sum (m n : ℚ) (h : ∀ x : ℚ, m < x ∧ x < n ↔ (mx - 1) / (x + 3) > 0) :
  m + n = -10 / 3 :=
sorry

end NUMINAMATH_GPT_interval_sum_l929_92906


namespace NUMINAMATH_GPT_solve_for_x_l929_92915

def f (x : ℝ) : ℝ := x^2 - 5 * x + 6

theorem solve_for_x : {x : ℝ | f (f x) = f x} = {0, 2, 3, 5} :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l929_92915


namespace NUMINAMATH_GPT_average_of_abc_l929_92931

theorem average_of_abc (A B C : ℚ) 
  (h1 : 2002 * C + 4004 * A = 8008) 
  (h2 : 3003 * B - 5005 * A = 7007) : 
  (A + B + C) / 3 = 22 / 9 := 
by 
  sorry

end NUMINAMATH_GPT_average_of_abc_l929_92931


namespace NUMINAMATH_GPT_patients_before_doubling_l929_92963

theorem patients_before_doubling (C P : ℕ) 
    (h1 : (1 / 4) * C = 13) 
    (h2 : C = 2 * P) : 
    P = 26 := 
sorry

end NUMINAMATH_GPT_patients_before_doubling_l929_92963


namespace NUMINAMATH_GPT_Bryan_deposit_amount_l929_92997

theorem Bryan_deposit_amount (deposit_mark : ℕ) (deposit_bryan : ℕ)
  (h1 : deposit_mark = 88)
  (h2 : deposit_bryan = 5 * deposit_mark - 40) : 
  deposit_bryan = 400 := 
by
  sorry

end NUMINAMATH_GPT_Bryan_deposit_amount_l929_92997


namespace NUMINAMATH_GPT_find_a2_l929_92940

-- Definitions from the conditions
def is_geometric_sequence (a : ℕ → ℕ) (q : ℕ) := ∀ n, a (n + 1) = q * a n
def sum_geom_seq (a : ℕ → ℕ) (q : ℕ) (n : ℕ) := (a 0 * (1 - q^(n + 1))) / (1 - q)

-- Given conditions
def a_n : ℕ → ℕ := sorry -- Define the sequence a_n
def q : ℕ := 2
def S_4 := 60

-- The theorem to be proved
theorem find_a2 (h1: is_geometric_sequence a_n q)
                (h2: sum_geom_seq a_n q 3 = S_4) : 
                a_n 1 = 8 :=
sorry

end NUMINAMATH_GPT_find_a2_l929_92940


namespace NUMINAMATH_GPT_max_marks_400_l929_92974

theorem max_marks_400 {M : ℝ} (h1 : 0.35 * M = 140) : M = 400 :=
by 
-- skipping the proof using sorry
sorry

end NUMINAMATH_GPT_max_marks_400_l929_92974


namespace NUMINAMATH_GPT_gcd_ab_conditions_l929_92924

theorem gcd_ab_conditions 
  (a b : ℕ) (h1 : a > b) (h2 : Nat.gcd a b = 1) : 
  Nat.gcd (a + b) (a - b) = 1 ∨ Nat.gcd (a + b) (a - b) = 2 := 
sorry

end NUMINAMATH_GPT_gcd_ab_conditions_l929_92924


namespace NUMINAMATH_GPT_total_molecular_weight_correct_l929_92999

-- Defining the molecular weights of elements
def mol_weight_C : ℝ := 12.01
def mol_weight_H : ℝ := 1.01
def mol_weight_Cl : ℝ := 35.45
def mol_weight_O : ℝ := 16.00

-- Defining the number of moles of compounds
def moles_C2H5Cl : ℝ := 15
def moles_O2 : ℝ := 12

-- Calculating the molecular weights of compounds
def mol_weight_C2H5Cl : ℝ := (2 * mol_weight_C) + (5 * mol_weight_H) + mol_weight_Cl
def mol_weight_O2 : ℝ := 2 * mol_weight_O

-- Calculating the total weight of each compound
def total_weight_C2H5Cl : ℝ := moles_C2H5Cl * mol_weight_C2H5Cl
def total_weight_O2 : ℝ := moles_O2 * mol_weight_O2

-- Defining the final total weight
def total_weight : ℝ := total_weight_C2H5Cl + total_weight_O2

-- Statement to prove
theorem total_molecular_weight_correct :
  total_weight = 1351.8 := by
  sorry

end NUMINAMATH_GPT_total_molecular_weight_correct_l929_92999


namespace NUMINAMATH_GPT_Walter_bus_time_l929_92969

theorem Walter_bus_time :
  let start_time := 7 * 60 + 30 -- 7:30 a.m. in minutes
  let end_time := 16 * 60 + 15 -- 4:15 p.m. in minutes
  let away_time := end_time - start_time -- total time away from home in minutes
  let classes_time := 7 * 45 -- 7 classes 45 minutes each
  let lunch_time := 40 -- lunch time in minutes
  let additional_school_time := 1.5 * 60 -- additional time at school in minutes
  let school_time := classes_time + lunch_time + additional_school_time -- total school activities time
  (away_time - school_time) = 80 :=
by
  sorry

end NUMINAMATH_GPT_Walter_bus_time_l929_92969


namespace NUMINAMATH_GPT_zachary_pushups_l929_92946

theorem zachary_pushups (david_pushups zachary_pushups : ℕ) (h₁ : david_pushups = 44) (h₂ : david_pushups = zachary_pushups + 9) :
  zachary_pushups = 35 :=
by
  sorry

end NUMINAMATH_GPT_zachary_pushups_l929_92946


namespace NUMINAMATH_GPT_symmetric_function_is_periodic_l929_92972

theorem symmetric_function_is_periodic {f : ℝ → ℝ} {a b y0 : ℝ}
  (h1 : ∀ x, f (a + x) - y0 = y0 - f (a - x))
  (h2 : ∀ x, f (b + x) = f (b - x))
  (hb : b > a) :
  ∀ x, f (x + 4 * (b - a)) = f x := sorry

end NUMINAMATH_GPT_symmetric_function_is_periodic_l929_92972


namespace NUMINAMATH_GPT_jerrys_current_average_score_l929_92945

theorem jerrys_current_average_score (A : ℝ) (h1 : 3 * A + 98 = 4 * (A + 2)) : A = 90 :=
by
  sorry

end NUMINAMATH_GPT_jerrys_current_average_score_l929_92945


namespace NUMINAMATH_GPT_group_card_exchanges_l929_92988

theorem group_card_exchanges (x : ℕ) (hx : x * (x - 1) = 90) : x = 10 :=
by { sorry }

end NUMINAMATH_GPT_group_card_exchanges_l929_92988


namespace NUMINAMATH_GPT_inequality_unique_solution_l929_92949

theorem inequality_unique_solution (p : ℝ) :
  (∃ x : ℝ, 0 ≤ x^2 + p * x + 5 ∧ x^2 + p * x + 5 ≤ 1) →
  (∃ x : ℝ, x^2 + p * x + 4 = 0) → p = 4 ∨ p = -4 :=
sorry

end NUMINAMATH_GPT_inequality_unique_solution_l929_92949


namespace NUMINAMATH_GPT_extra_men_needed_approx_is_60_l929_92921

noncomputable def extra_men_needed : ℝ :=
  let total_distance := 15.0   -- km
  let total_days := 300.0      -- days
  let initial_workforce := 40.0 -- men
  let completed_distance := 2.5 -- km
  let elapsed_days := 100.0    -- days

  let remaining_distance := total_distance - completed_distance -- km
  let remaining_days := total_days - elapsed_days               -- days

  let current_rate := completed_distance / elapsed_days -- km/day
  let required_rate := remaining_distance / remaining_days -- km/day

  let required_factor := required_rate / current_rate
  let new_workforce := initial_workforce * required_factor
  let extra_men := new_workforce - initial_workforce

  extra_men

theorem extra_men_needed_approx_is_60 :
  abs (extra_men_needed - 60) < 1 :=
sorry

end NUMINAMATH_GPT_extra_men_needed_approx_is_60_l929_92921


namespace NUMINAMATH_GPT_max_handshakes_without_cycles_l929_92965

open BigOperators

theorem max_handshakes_without_cycles :
  ∀ n : ℕ, n = 20 → ∑ i in Finset.range (n - 1), i = 190 :=
by intros;
   sorry

end NUMINAMATH_GPT_max_handshakes_without_cycles_l929_92965


namespace NUMINAMATH_GPT_cody_tickets_l929_92984

theorem cody_tickets (initial_tickets spent_tickets won_tickets : ℕ) (h_initial : initial_tickets = 49) (h_spent : spent_tickets = 25) (h_won : won_tickets = 6) : initial_tickets - spent_tickets + won_tickets = 30 := 
by 
  sorry

end NUMINAMATH_GPT_cody_tickets_l929_92984


namespace NUMINAMATH_GPT_a719_divisible_by_11_l929_92934

theorem a719_divisible_by_11 (a : ℕ) (h : a < 10) : (∃ k : ℤ, a - 15 = 11 * k) ↔ a = 4 :=
by
  sorry

end NUMINAMATH_GPT_a719_divisible_by_11_l929_92934


namespace NUMINAMATH_GPT_scientific_notation_of_600000_l929_92993

theorem scientific_notation_of_600000 :
  600000 = 6 * 10^5 :=
sorry

end NUMINAMATH_GPT_scientific_notation_of_600000_l929_92993


namespace NUMINAMATH_GPT_num_digits_of_prime_started_numerals_l929_92992

theorem num_digits_of_prime_started_numerals (n : ℕ) (h : 4 * 10^(n-1) = 400) : n = 3 := 
  sorry

end NUMINAMATH_GPT_num_digits_of_prime_started_numerals_l929_92992


namespace NUMINAMATH_GPT_broken_stick_triangle_probability_l929_92978

noncomputable def probability_of_triangle (x y z : ℕ) : ℚ := sorry

theorem broken_stick_triangle_probability :
  ∀ x y z : ℕ, (x < y + z ∧ y < x + z ∧ z < x + y) → probability_of_triangle x y z = 1 / 4 := 
by
  sorry

end NUMINAMATH_GPT_broken_stick_triangle_probability_l929_92978


namespace NUMINAMATH_GPT_original_angle_measure_l929_92971

theorem original_angle_measure : 
  ∃ x : ℝ, (90 - x) = 3 * x - 2 ∧ x = 23 :=
by
  sorry

end NUMINAMATH_GPT_original_angle_measure_l929_92971


namespace NUMINAMATH_GPT_problem1_problem2_l929_92923

-- Problem 1 Statement
theorem problem1 (a : ℝ) (h : a ≠ 1) : (a^2 / (a - 1) - a - 1) = (1 / (a - 1)) :=
by
  sorry

-- Problem 2 Statement
theorem problem2 (x y : ℝ) (h1 : x ≠ y) (h2 : x ≠ -y) : 
  (2 * x * y / (x^2 - y^2)) / ((1 / (x - y)) + (1 / (x + y))) = y :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l929_92923


namespace NUMINAMATH_GPT_percentage_of_two_is_point_eight_l929_92919

theorem percentage_of_two_is_point_eight (p : ℝ) : (p / 100) * 2 = 0.8 ↔ p = 40 := 
by
  sorry

end NUMINAMATH_GPT_percentage_of_two_is_point_eight_l929_92919


namespace NUMINAMATH_GPT_dot_product_parallel_a_b_l929_92904

noncomputable def a : ℝ × ℝ := (-1, 1)
def b (x : ℝ) : ℝ × ℝ := (2, x)

-- Definition of parallel vectors
def parallel (v1 v2 : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v2 = (k * v1.1, k * v1.2)

-- Given conditions and result to prove
theorem dot_product_parallel_a_b : ∀ (x : ℝ), parallel a (b x) → x = -2 → (a.1 * (b x).1 + a.2 * (b x).2) = -4 := 
by
  intros x h_parallel h_x
  subst h_x
  sorry

end NUMINAMATH_GPT_dot_product_parallel_a_b_l929_92904


namespace NUMINAMATH_GPT_product_consecutive_natural_number_square_l929_92941

theorem product_consecutive_natural_number_square (n : ℕ) : 
  ∃ k : ℕ, 100 * (n^2 + n) + 25 = k^2 :=
by
  sorry

end NUMINAMATH_GPT_product_consecutive_natural_number_square_l929_92941


namespace NUMINAMATH_GPT_intersection_A_B_l929_92926

def A : Set ℕ := {0, 1, 2, 3, 4, 5}
def B : Set ℕ := {x | x^2 < 10}
def intersection_of_A_and_B : Set ℕ := {0, 1, 2, 3}

theorem intersection_A_B :
  A ∩ B = intersection_of_A_and_B :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l929_92926


namespace NUMINAMATH_GPT_maximal_N8_value_l929_92938

noncomputable def max_permutations_of_projections (A : Fin 8 → ℝ × ℝ) : ℕ := sorry

theorem maximal_N8_value (A : Fin 8 → ℝ × ℝ) :
  max_permutations_of_projections A = 56 :=
sorry

end NUMINAMATH_GPT_maximal_N8_value_l929_92938


namespace NUMINAMATH_GPT_not_make_all_numbers_equal_l929_92918

theorem not_make_all_numbers_equal (n : ℕ) (h : n ≥ 3)
  (a : Fin n → ℕ) (h1 : ∃ (i : Fin n), a i = 1 ∧ (∀ (j : Fin n), j ≠ i → a j = 0)) :
  ¬ ∃ x, ∀ i : Fin n, a i = x :=
by
  sorry

end NUMINAMATH_GPT_not_make_all_numbers_equal_l929_92918


namespace NUMINAMATH_GPT_smallest_positive_z_l929_92947

open Real

-- Definitions for the conditions
def sin_zero_condition (x : ℝ) : Prop := sin x = 0
def sin_half_condition (x z : ℝ) : Prop := sin (x + z) = 1 / 2

-- Theorem for the proof objective
theorem smallest_positive_z (x z : ℝ) (hx : sin_zero_condition x) (hz : sin_half_condition x z) : z = π / 6 := 
sorry

end NUMINAMATH_GPT_smallest_positive_z_l929_92947


namespace NUMINAMATH_GPT_mean_yoga_practice_days_l929_92900

noncomputable def mean_number_of_days (counts : List ℕ) (days : List ℕ) : ℚ :=
  let total_days := List.zipWith (λ c d => c * d) counts days |>.sum
  let total_students := counts.sum
  total_days / total_students

def counts : List ℕ := [2, 4, 5, 3, 2, 1, 3]
def days : List ℕ := [1, 2, 3, 4, 5, 6, 7]

theorem mean_yoga_practice_days : mean_number_of_days counts days = 37 / 10 := 
by 
  sorry

end NUMINAMATH_GPT_mean_yoga_practice_days_l929_92900


namespace NUMINAMATH_GPT_solve_system_eq_l929_92935

theorem solve_system_eq (x y z : ℝ) 
  (h1 : x * y = 6 * (x + y))
  (h2 : x * z = 4 * (x + z))
  (h3 : y * z = 2 * (y + z)) :
  (x = 0 ∧ y = 0 ∧ z = 0) ∨ 
  (x = -24 ∧ y = 24 / 5 ∧ z = 24 / 7) :=
  sorry

end NUMINAMATH_GPT_solve_system_eq_l929_92935


namespace NUMINAMATH_GPT_Sarah_collected_40_today_l929_92968

noncomputable def Sarah_yesterday : ℕ := 50
noncomputable def Lara_yesterday : ℕ := Sarah_yesterday + 30
noncomputable def Lara_today : ℕ := 70
noncomputable def Total_yesterday : ℕ := Sarah_yesterday + Lara_yesterday
noncomputable def Total_today : ℕ := Total_yesterday - 20
noncomputable def Sarah_today : ℕ := Total_today - Lara_today

theorem Sarah_collected_40_today : Sarah_today = 40 := 
by
  sorry

end NUMINAMATH_GPT_Sarah_collected_40_today_l929_92968


namespace NUMINAMATH_GPT_no_nat_number_divisible_by_1998_has_digit_sum_lt_27_l929_92991

-- Definition of a natural number being divisible by another
def divisible (m n : ℕ) : Prop := ∃ k : ℕ, m = k * n

-- Definition of the sum of the digits of a natural number
def sum_of_digits (n : ℕ) : ℕ := 
  n.digits 10 |>.sum

-- Statement of the problem
theorem no_nat_number_divisible_by_1998_has_digit_sum_lt_27 :
  ¬ ∃ n : ℕ, divisible n 1998 ∧ sum_of_digits n < 27 :=
by 
  sorry

end NUMINAMATH_GPT_no_nat_number_divisible_by_1998_has_digit_sum_lt_27_l929_92991


namespace NUMINAMATH_GPT_log5_square_simplification_l929_92956

noncomputable def log5 (x : ℝ) : ℝ := Real.log x / Real.log 5

theorem log5_square_simplification : (log5 (7 * log5 25))^2 = (log5 14)^2 :=
by
  sorry

end NUMINAMATH_GPT_log5_square_simplification_l929_92956


namespace NUMINAMATH_GPT_average_number_of_ducks_l929_92942

def average_ducks (A E K : ℕ) : ℕ :=
  (A + E + K) / 3

theorem average_number_of_ducks :
  ∀ (A E K : ℕ), A = 2 * E → E = K - 45 → A = 30 → average_ducks A E K = 35 :=
by 
  intros A E K h1 h2 h3
  sorry

end NUMINAMATH_GPT_average_number_of_ducks_l929_92942


namespace NUMINAMATH_GPT_van_helsing_earnings_l929_92961

theorem van_helsing_earnings (V W : ℕ) 
  (h1 : W = 4 * V) 
  (h2 : W = 8) :
  let E_v := 5 * (V / 2)
  let E_w := 10 * 8
  let E_total := E_v + E_w
  E_total = 85 :=
by
  sorry

end NUMINAMATH_GPT_van_helsing_earnings_l929_92961


namespace NUMINAMATH_GPT_number_of_juniors_l929_92914

theorem number_of_juniors
  (T : ℕ := 28)
  (hT : T = 28)
  (x y : ℕ)
  (hxy : x = y)
  (J S : ℕ)
  (hx : x = J / 4)
  (hy : y = S / 10)
  (hJS : J + S = T) :
  J = 8 :=
by sorry

end NUMINAMATH_GPT_number_of_juniors_l929_92914


namespace NUMINAMATH_GPT_number_of_math_books_l929_92955

-- Definitions based on the conditions in the problem
def total_books (M H : ℕ) : Prop := M + H = 90
def total_cost (M H : ℕ) : Prop := 4 * M + 5 * H = 390

-- Proof statement
theorem number_of_math_books (M H : ℕ) (h1 : total_books M H) (h2 : total_cost M H) : M = 60 :=
  sorry

end NUMINAMATH_GPT_number_of_math_books_l929_92955


namespace NUMINAMATH_GPT_prove_temperature_on_Thursday_l929_92909

def temperature_on_Thursday 
  (temps : List ℝ)   -- List of temperatures for 6 days.
  (avg : ℝ)          -- Average temperature for the week.
  (sum_six_days : ℝ) -- Sum of temperature readings for 6 days.
  (days : ℕ := 7)    -- Number of days in the week.
  (missing_day : ℕ := 1)  -- One missing day (Thursday).
  (thurs_temp : ℝ := 99.8) -- Temperature on Thursday to be proved.
: Prop := (avg * days) - sum_six_days = thurs_temp

theorem prove_temperature_on_Thursday 
  : temperature_on_Thursday [99.1, 98.2, 98.7, 99.3, 99, 98.9] 99 593.2 :=
by
  sorry

end NUMINAMATH_GPT_prove_temperature_on_Thursday_l929_92909


namespace NUMINAMATH_GPT_survey_preference_l929_92962

theorem survey_preference (X Y : ℕ) 
  (ratio_condition : X / Y = 5)
  (total_respondents : X + Y = 180) :
  X = 150 := 
sorry

end NUMINAMATH_GPT_survey_preference_l929_92962


namespace NUMINAMATH_GPT_gcd_840_1764_evaluate_polynomial_at_2_l929_92901

-- Define the Euclidean algorithm steps and prove the gcd result
theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

-- Define the polynomial and evaluate it using Horner's method
def polynomial := λ x : ℕ => 2 * (x ^ 4) + 3 * (x ^ 3) + 5 * x - 4

theorem evaluate_polynomial_at_2 : polynomial 2 = 62 := by
  sorry

end NUMINAMATH_GPT_gcd_840_1764_evaluate_polynomial_at_2_l929_92901


namespace NUMINAMATH_GPT_karlson_expenditure_exceeds_2000_l929_92911

theorem karlson_expenditure_exceeds_2000 :
  ∃ n m : ℕ, 25 * n + 340 * m > 2000 :=
by {
  -- proof must go here
  sorry
}

end NUMINAMATH_GPT_karlson_expenditure_exceeds_2000_l929_92911


namespace NUMINAMATH_GPT_groups_needed_for_sampling_l929_92994

def total_students : ℕ := 600
def sample_size : ℕ := 20

theorem groups_needed_for_sampling : (total_students / sample_size = 30) :=
by
  sorry

end NUMINAMATH_GPT_groups_needed_for_sampling_l929_92994


namespace NUMINAMATH_GPT_budget_circle_salaries_degrees_l929_92925

theorem budget_circle_salaries_degrees :
  let transportation := 20
  let research_development := 9
  let utilities := 5
  let equipment := 4
  let supplies := 2
  let total_percent := 100
  let full_circle_degrees := 360
  let total_allocated_percent := transportation + research_development + utilities + equipment + supplies
  let salaries_percent := total_percent - total_allocated_percent
  let salaries_degrees := (salaries_percent * full_circle_degrees) / total_percent
  salaries_degrees = 216 :=
by
  sorry

end NUMINAMATH_GPT_budget_circle_salaries_degrees_l929_92925


namespace NUMINAMATH_GPT_annual_subscription_cost_l929_92908

-- Definitions based on the conditions

def monthly_cost : ℝ := 10
def months_per_year : ℕ := 12
def discount_rate : ℝ := 0.20

-- The statement based on the correct answer
theorem annual_subscription_cost : 
  (monthly_cost * months_per_year) * (1 - discount_rate) = 96 := 
by
  sorry

end NUMINAMATH_GPT_annual_subscription_cost_l929_92908


namespace NUMINAMATH_GPT_area_of_field_l929_92966

noncomputable def area_square_field (speed_kmh : ℕ) (time_min : ℕ) : ℝ :=
  let speed_m_per_min := (speed_kmh * 1000) / 60
  let distance := speed_m_per_min * time_min
  let side_length := distance / Real.sqrt 2
  side_length ^ 2

-- Given conditions
theorem area_of_field : area_square_field 4 3 = 20000 := by
  sorry

end NUMINAMATH_GPT_area_of_field_l929_92966


namespace NUMINAMATH_GPT_find_difference_in_ticket_costs_l929_92929

-- Conditions
def num_adults : ℕ := 9
def num_children : ℕ := 7
def cost_adult_ticket : ℕ := 11
def cost_child_ticket : ℕ := 7

def total_cost_adults : ℕ := num_adults * cost_adult_ticket
def total_cost_children : ℕ := num_children * cost_child_ticket
def total_tickets : ℕ := num_adults + num_children

-- Discount conditions (not needed for this proof since they don't apply)
def apply_discount (total_tickets : ℕ) (total_cost : ℕ) : ℕ :=
  if total_tickets >= 10 ∧ total_tickets <= 12 then
    total_cost * 9 / 10
  else if total_tickets >= 13 ∧ total_tickets <= 15 then
    total_cost * 85 / 100
  else
    total_cost

-- The main statement to prove
theorem find_difference_in_ticket_costs : total_cost_adults - total_cost_children = 50 := by
  sorry

end NUMINAMATH_GPT_find_difference_in_ticket_costs_l929_92929


namespace NUMINAMATH_GPT_tracy_initial_balloons_l929_92943

theorem tracy_initial_balloons (T : ℕ) : 
  (12 + 8 + (T + 24) / 2 = 35) → T = 6 :=
by
  sorry

end NUMINAMATH_GPT_tracy_initial_balloons_l929_92943


namespace NUMINAMATH_GPT_abs_ineq_solution_set_l929_92954

theorem abs_ineq_solution_set {x : ℝ} : (|x - 1| ≤ 2) ↔ (-1 ≤ x ∧ x ≤ 3) :=
by
  sorry

end NUMINAMATH_GPT_abs_ineq_solution_set_l929_92954


namespace NUMINAMATH_GPT_bottles_remaining_after_2_days_l929_92967

def total_bottles := 48 

def first_day_father_consumption := total_bottles / 4
def first_day_mother_consumption := total_bottles / 6
def first_day_son_consumption := total_bottles / 8

def total_first_day_consumption := first_day_father_consumption + first_day_mother_consumption + first_day_son_consumption 
def remaining_after_first_day := total_bottles - total_first_day_consumption

def second_day_father_consumption := remaining_after_first_day / 5
def remaining_after_father := remaining_after_first_day - second_day_father_consumption
def second_day_mother_consumption := remaining_after_father / 7
def remaining_after_mother := remaining_after_father - second_day_mother_consumption
def second_day_son_consumption := remaining_after_mother / 9
def remaining_after_son := remaining_after_mother - second_day_son_consumption
def second_day_daughter_consumption := remaining_after_son / 9
def remaining_after_daughter := remaining_after_son - second_day_daughter_consumption

theorem bottles_remaining_after_2_days : ∀ (total_bottles : ℕ), remaining_after_daughter = 14 := 
by
  sorry

end NUMINAMATH_GPT_bottles_remaining_after_2_days_l929_92967


namespace NUMINAMATH_GPT_calc_expr_l929_92990

theorem calc_expr :
  (-1) * (-3) + 3^2 / (8 - 5) = 6 :=
by
  sorry

end NUMINAMATH_GPT_calc_expr_l929_92990


namespace NUMINAMATH_GPT_find_x_l929_92986

theorem find_x (x : ℝ) (h1 : x ≠ 0) (h2 : x = (1 / x) * (-x) + 3) : x = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l929_92986


namespace NUMINAMATH_GPT_factorize_expression_l929_92977

theorem factorize_expression (x : ℝ) : 2 * x - x^2 = x * (2 - x) := sorry

end NUMINAMATH_GPT_factorize_expression_l929_92977


namespace NUMINAMATH_GPT_ribbon_left_after_wrapping_l929_92975

def total_ribbon_needed (gifts : ℕ) (ribbon_per_gift : ℝ) : ℝ :=
  gifts * ribbon_per_gift

def remaining_ribbon (initial_ribbon : ℝ) (used_ribbon : ℝ) : ℝ :=
  initial_ribbon - used_ribbon

theorem ribbon_left_after_wrapping : 
  ∀ (gifts : ℕ) (ribbon_per_gift initial_ribbon : ℝ),
  gifts = 8 →
  ribbon_per_gift = 1.5 →
  initial_ribbon = 15 →
  remaining_ribbon initial_ribbon (total_ribbon_needed gifts ribbon_per_gift) = 3 :=
by
  intros gifts ribbon_per_gift initial_ribbon h1 h2 h3
  rw [h1, h2, h3]
  simp [total_ribbon_needed, remaining_ribbon]
  sorry

end NUMINAMATH_GPT_ribbon_left_after_wrapping_l929_92975


namespace NUMINAMATH_GPT_rectangular_prism_diagonals_l929_92996

theorem rectangular_prism_diagonals (length width height : ℕ) (length_eq : length = 4) (width_eq : width = 3) (height_eq : height = 2) : 
  ∃ (total_diagonals : ℕ), total_diagonals = 16 :=
by
  let face_diagonals := 12
  let space_diagonals := 4
  let total_diagonals := face_diagonals + space_diagonals
  use total_diagonals
  sorry

end NUMINAMATH_GPT_rectangular_prism_diagonals_l929_92996


namespace NUMINAMATH_GPT_bricks_lay_calculation_l929_92927

theorem bricks_lay_calculation (b c d : ℕ) (h1 : 0 < c) (h2 : 0 < d) : 
  ∃ y : ℕ, y = (b * (b + d) * (c + d))/(c * d) :=
sorry

end NUMINAMATH_GPT_bricks_lay_calculation_l929_92927


namespace NUMINAMATH_GPT_combination_sum_l929_92922

-- Define the combination function
def combination (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

-- Given conditions
axiom combinatorial_identity (n r : ℕ) : combination n r + combination n (r + 1) = combination (n + 1) (r + 1)

-- The theorem we aim to prove
theorem combination_sum : combination 8 2 + combination 8 3 + combination 9 2 = 120 := 
by
  sorry

end NUMINAMATH_GPT_combination_sum_l929_92922


namespace NUMINAMATH_GPT_coffee_consumption_l929_92995

theorem coffee_consumption (h1 h2 g1 h3: ℕ) (k : ℕ) (g2 : ℕ) :
  (k = h1 * g1) → (h1 = 9) → (g1 = 2) → (h2 = 6) → (k / h2 = g2) → (g2 = 3) :=
by
  sorry

end NUMINAMATH_GPT_coffee_consumption_l929_92995


namespace NUMINAMATH_GPT_factor_theorem_example_l929_92983

theorem factor_theorem_example (t : ℚ) : (4 * t^3 + 6 * t^2 + 11 * t - 6 = 0) ↔ (t = 1/2) :=
by sorry

end NUMINAMATH_GPT_factor_theorem_example_l929_92983


namespace NUMINAMATH_GPT_power_greater_than_linear_l929_92973

theorem power_greater_than_linear (n : ℕ) (h : n ≥ 3) : 2^n > 2 * n + 1 := 
by {
  sorry
}

end NUMINAMATH_GPT_power_greater_than_linear_l929_92973


namespace NUMINAMATH_GPT_problem_solution_l929_92928

theorem problem_solution (x y m : ℝ) (hx : x > 0) (hy : y > 0) : 
  (∀ x y, (2 * y / x) + (8 * x / y) > m^2 + 2 * m) → -4 < m ∧ m < 2 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_problem_solution_l929_92928


namespace NUMINAMATH_GPT_probability_A_not_lose_l929_92912

theorem probability_A_not_lose (p_win p_draw : ℝ) (h_win : p_win = 0.3) (h_draw : p_draw = 0.5) :
  (p_win + p_draw = 0.8) :=
by
  rw [h_win, h_draw]
  norm_num

end NUMINAMATH_GPT_probability_A_not_lose_l929_92912


namespace NUMINAMATH_GPT_lcm_36_98_is_1764_l929_92902

theorem lcm_36_98_is_1764 : Nat.lcm 36 98 = 1764 := by
  sorry

end NUMINAMATH_GPT_lcm_36_98_is_1764_l929_92902


namespace NUMINAMATH_GPT_value_of_r_when_n_is_2_l929_92903

-- Define the given conditions
def s : ℕ := 2 ^ 2 + 1
def r : ℤ := 3 ^ s - s

-- Prove that r equals 238 when n = 2
theorem value_of_r_when_n_is_2 : r = 238 := by
  sorry

end NUMINAMATH_GPT_value_of_r_when_n_is_2_l929_92903


namespace NUMINAMATH_GPT_prime_divides_sum_l929_92979

theorem prime_divides_sum 
  (a b c : ℕ) 
  (h1 : a^3 + 4 * b + c = a * b * c)
  (h2 : a ≥ c)
  (h3 : Prime (a^2 + 2 * a + 2)) : 
  (a^2 + 2 * a + 2) ∣ (a + 2 * b + 2) := 
sorry

end NUMINAMATH_GPT_prime_divides_sum_l929_92979


namespace NUMINAMATH_GPT_directrix_parabola_l929_92920

theorem directrix_parabola (x y : ℝ) :
  (x^2 = (1/4 : ℝ) * y) → (y = -1/16) :=
sorry

end NUMINAMATH_GPT_directrix_parabola_l929_92920
