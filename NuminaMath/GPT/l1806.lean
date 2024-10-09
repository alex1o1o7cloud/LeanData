import Mathlib

namespace colorable_graph_l1806_180661

variable (V : Type) [Fintype V] [DecidableEq V] (E : V → V → Prop) [DecidableRel E]

/-- Each city has at least one road leading out of it -/
def has_one_road (v : V) : Prop := ∃ w : V, E v w

/-- No city is connected by roads to all other cities -/
def not_connected_to_all (v : V) : Prop := ¬ ∀ w : V, E v w ↔ w ≠ v

/-- A set of cities D is dominating if every city not in D is connected by a road to at least one city in D -/
def is_dominating_set (D : Finset V) : Prop :=
  ∀ v : V, v ∉ D → ∃ d ∈ D, E v d

noncomputable def dominating_set_min_card (k : ℕ) : Prop :=
  ∀ D : Finset V, is_dominating_set V E D → D.card ≥ k

/-- Prove that the graph can be colored using 2001 - k colors such that no two adjacent vertices share the same color -/
theorem colorable_graph (k : ℕ) (hk : dominating_set_min_card V E k) :
    ∃ (colors : V → Fin (2001 - k)), ∀ v w : V, E v w → colors v ≠ colors w := 
by 
  sorry

end colorable_graph_l1806_180661


namespace triangle_inequality_l1806_180667

theorem triangle_inequality (a b c : ℝ) (h1 : b + c > a) (h2 : c + a > b) (h3 : a + b > c) :
  ab + bc + ca ≤ a^2 + b^2 + c^2 ∧ a^2 + b^2 + c^2 < 2 * (ab + bc + ca) :=
by
  sorry

end triangle_inequality_l1806_180667


namespace oldest_son_park_visits_l1806_180657

theorem oldest_son_park_visits 
    (season_pass_cost : ℕ)
    (cost_per_trip : ℕ)
    (youngest_son_trips : ℕ) 
    (remaining_value : ℕ)
    (oldest_son_trips : ℕ) : 
    season_pass_cost = 100 →
    cost_per_trip = 4 →
    youngest_son_trips = 15 →
    remaining_value = season_pass_cost - youngest_son_trips * cost_per_trip →
    oldest_son_trips = remaining_value / cost_per_trip →
    oldest_son_trips = 10 := 
by sorry

end oldest_son_park_visits_l1806_180657


namespace remainder_of_division_l1806_180632

theorem remainder_of_division (d : ℝ) (q : ℝ) (r : ℝ) : 
  d = 187.46067415730337 → q = 89 → 16698 = (d * q) + r → r = 14 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  simp at h3
  sorry

end remainder_of_division_l1806_180632


namespace barrels_in_one_ton_l1806_180691

-- Definitions (conditions)
def barrel_weight : ℕ := 10 -- in kilograms
def ton_in_kilograms : ℕ := 1000

-- Theorem Statement
theorem barrels_in_one_ton : ton_in_kilograms / barrel_weight = 100 :=
by
  sorry

end barrels_in_one_ton_l1806_180691


namespace find_first_week_customers_l1806_180600

def commission_per_customer := 1
def first_week_customers (C : ℕ) := C
def second_week_customers (C : ℕ) := 2 * C
def third_week_customers (C : ℕ) := 3 * C
def salary := 500
def bonus := 50
def total_earnings := 760

theorem find_first_week_customers (C : ℕ) (H : salary + bonus + commission_per_customer * (first_week_customers C + second_week_customers C + third_week_customers C) = total_earnings) : 
  C = 35 :=
by
  sorry

end find_first_week_customers_l1806_180600


namespace total_number_of_bricks_l1806_180609

/-- Given bricks of volume 80 unit cubes and 42 unit cubes,
 and a box of volume 1540 unit cubes,
 prove the total number of bricks that can fill the box exactly is 24. -/
theorem total_number_of_bricks (x y : ℕ) (vol_a vol_b total_vol : ℕ)
  (vol_a_def : vol_a = 80)
  (vol_b_def : vol_b = 42)
  (total_vol_def : total_vol = 1540)
  (volume_filled : x * vol_a + y * vol_b = total_vol) :
  x + y = 24 :=
  sorry

end total_number_of_bricks_l1806_180609


namespace probability_equivalence_l1806_180654

-- Definitions for the conditions:
def total_products : ℕ := 7
def genuine_products : ℕ := 4
def defective_products : ℕ := 3

-- Function to return the probability of selecting a genuine product on the second draw, given first is defective
def probability_genuine_given_defective : ℚ := 
  (defective_products / total_products) * (genuine_products / (total_products - 1))

-- The theorem we need to prove:
theorem probability_equivalence :
  probability_genuine_given_defective = 2 / 3 :=
by
  sorry -- Proof placeholder

end probability_equivalence_l1806_180654


namespace kim_shirts_left_l1806_180674

-- Define the total number of shirts initially
def initial_shirts : ℕ := 4 * 12

-- Define the number of shirts given to the sister as 1/3 of the total
def shirts_given_to_sister : ℕ := initial_shirts / 3

-- Define the number of shirts left after giving some to the sister
def shirts_left : ℕ := initial_shirts - shirts_given_to_sister

-- The theorem we need to prove: Kim has 32 shirts left
theorem kim_shirts_left : shirts_left = 32 := by
  -- Proof is omitted
  sorry

end kim_shirts_left_l1806_180674


namespace sin_double_angle_l1806_180605

theorem sin_double_angle (α : ℝ) (h : Real.sin (α + Real.pi / 4) = 1 / 2) : Real.sin (2 * α) = -1 / 2 :=
sorry

end sin_double_angle_l1806_180605


namespace speed_of_train_l1806_180683

-- Conditions
def length_of_train : ℝ := 100
def time_to_cross : ℝ := 12

-- Question and answer
theorem speed_of_train : length_of_train / time_to_cross = 8.33 := 
by 
  sorry

end speed_of_train_l1806_180683


namespace find_a_l1806_180660

noncomputable def S_n (n : ℕ) (a : ℝ) : ℝ := 2 * 3^n + a
noncomputable def a_1 (a : ℝ) : ℝ := S_n 1 a
noncomputable def a_2 (a : ℝ) : ℝ := S_n 2 a - S_n 1 a
noncomputable def a_3 (a : ℝ) : ℝ := S_n 3 a - S_n 2 a

theorem find_a (a : ℝ) : a_1 a * a_3 a = (a_2 a)^2 → a = -2 :=
by
  sorry

end find_a_l1806_180660


namespace scientific_notation_of_0_000815_l1806_180603

theorem scientific_notation_of_0_000815 :
  (∃ (c : ℝ) (n : ℤ), 0.000815 = c * 10^n ∧ 1 ≤ c ∧ c < 10) ∧ (0.000815 = 8.15 * 10^(-4)) :=
by
  sorry

end scientific_notation_of_0_000815_l1806_180603


namespace savings_difference_correct_l1806_180665

noncomputable def savings_1989_dick : ℝ := 5000
noncomputable def savings_1989_jane : ℝ := 5000

noncomputable def savings_1990_dick : ℝ := savings_1989_dick + 0.10 * savings_1989_dick
noncomputable def savings_1990_jane : ℝ := savings_1989_jane - 0.05 * savings_1989_jane

noncomputable def savings_1991_dick : ℝ := savings_1990_dick + 0.07 * savings_1990_dick
noncomputable def savings_1991_jane : ℝ := savings_1990_jane + 0.08 * savings_1990_jane

noncomputable def savings_1992_dick : ℝ := savings_1991_dick - 0.12 * savings_1991_dick
noncomputable def savings_1992_jane : ℝ := savings_1991_jane + 0.15 * savings_1991_jane

noncomputable def total_savings_dick : ℝ :=
savings_1989_dick + savings_1990_dick + savings_1991_dick + savings_1992_dick

noncomputable def total_savings_jane : ℝ :=
savings_1989_jane + savings_1990_jane + savings_1991_jane + savings_1992_jane

noncomputable def difference_of_savings : ℝ :=
total_savings_dick - total_savings_jane

theorem savings_difference_correct :
  difference_of_savings = 784.30 :=
by sorry

end savings_difference_correct_l1806_180665


namespace area_of_folded_shape_is_two_units_squared_l1806_180641

/-- 
A square piece of paper with each side of length 2 units is divided into 
four equal squares along both its length and width. From the top left corner to 
bottom right corner, a line is drawn through the center dividing the square diagonally.
The paper is folded along this line to form a new shape.
We prove that the area of the folded shape is 2 units².
-/
theorem area_of_folded_shape_is_two_units_squared
  (side_len : ℝ)
  (area_original : ℝ)
  (area_folded : ℝ)
  (h1 : side_len = 2)
  (h2 : area_original = side_len * side_len)
  (h3 : area_folded = area_original / 2) :
  area_folded = 2 := by
  -- Place proof here
  sorry

end area_of_folded_shape_is_two_units_squared_l1806_180641


namespace rectangle_perimeter_ratio_l1806_180648

theorem rectangle_perimeter_ratio
    (initial_height : ℕ)
    (initial_width : ℕ)
    (H_initial_height : initial_height = 2)
    (H_initial_width : initial_width = 4)
    (fold1_height : ℕ)
    (fold1_width : ℕ)
    (H_fold1_height : fold1_height = initial_height / 2)
    (H_fold1_width : fold1_width = initial_width)
    (fold2_height : ℕ)
    (fold2_width : ℕ)
    (H_fold2_height : fold2_height = fold1_height)
    (H_fold2_width : fold2_width = fold1_width / 2)
    (cut_height : ℕ)
    (cut_width : ℕ)
    (H_cut_height : cut_height = fold2_height)
    (H_cut_width : cut_width = fold2_width) :
    (2 * (cut_height + cut_width)) / (2 * (fold1_height + fold1_width)) = 3 / 5 := 
    by sorry

end rectangle_perimeter_ratio_l1806_180648


namespace total_dreams_correct_l1806_180666

def dreams_per_day : Nat := 4
def days_in_year : Nat := 365
def current_year_dreams : Nat := dreams_per_day * days_in_year
def last_year_dreams : Nat := 2 * current_year_dreams
def total_dreams : Nat := current_year_dreams + last_year_dreams

theorem total_dreams_correct : total_dreams = 4380 :=
by
  -- prime verification needed here
  sorry

end total_dreams_correct_l1806_180666


namespace total_acorns_proof_l1806_180634

variable (x y : ℝ)

def total_acorns (x y : ℝ) : ℝ :=
  let shawna := x
  let sheila := 5.3 * x
  let danny := 5.3 * x + y
  let ella := 2 * (4.3 * x + y)
  shawna + sheila + danny + ella

theorem total_acorns_proof (x y : ℝ) :
  total_acorns x y = 20.2 * x + 3 * y :=
by
  unfold total_acorns
  sorry

end total_acorns_proof_l1806_180634


namespace smallest_divisor_subtracted_l1806_180664

theorem smallest_divisor_subtracted (a b d : ℕ) (h1: a = 899830) (h2: b = 6) (h3: a - b = 899824) (h4 : 6 < d) 
(h5 : d ∣ (a - b)) : d = 8 :=
by
  sorry

end smallest_divisor_subtracted_l1806_180664


namespace proof_problem_l1806_180601

def M : Set ℝ := { x | x > -1 }

theorem proof_problem : {0} ⊆ M := by
  sorry

end proof_problem_l1806_180601


namespace cube_surface_area_l1806_180677

theorem cube_surface_area (V : ℝ) (s : ℝ) (A : ℝ) :
  V = 729 ∧ V = s^3 ∧ A = 6 * s^2 → A = 486 := by
  sorry

end cube_surface_area_l1806_180677


namespace largest_integer_a_can_be_less_than_l1806_180628

theorem largest_integer_a_can_be_less_than (a b : ℕ) (h1 : 9 < a) (h2 : 19 < b) (h3 : b < 31) (h4 : a / b = 2 / 3) :
  a < 21 :=
sorry

end largest_integer_a_can_be_less_than_l1806_180628


namespace bob_needs_50_percent_improvement_l1806_180690

def bob_time_in_seconds : ℕ := 640
def sister_time_in_seconds : ℕ := 320
def percentage_improvement_needed (bob_time sister_time : ℕ) : ℚ :=
  ((bob_time - sister_time) / bob_time : ℚ) * 100

theorem bob_needs_50_percent_improvement :
  percentage_improvement_needed bob_time_in_seconds sister_time_in_seconds = 50 := by
  sorry

end bob_needs_50_percent_improvement_l1806_180690


namespace carter_reading_pages_l1806_180613

theorem carter_reading_pages (c l o : ℕ)
  (h1: c = l / 2)
  (h2: l = o + 20)
  (h3: o = 40) : c = 30 := by
  sorry

end carter_reading_pages_l1806_180613


namespace M_inter_N_l1806_180636

def M : Set ℝ := {y | ∃ x : ℝ, y = 2^(-x)}
def N : Set ℝ := {y | ∃ x : ℝ, y = Real.sin x}

theorem M_inter_N : M ∩ N = {y | 0 < y ∧ y ≤ 1} :=
by
  sorry

end M_inter_N_l1806_180636


namespace notebooks_left_l1806_180651

theorem notebooks_left (bundles : ℕ) (notebooks_per_bundle : ℕ) (groups : ℕ) (students_per_group : ℕ) : 
  bundles = 5 ∧ notebooks_per_bundle = 25 ∧ groups = 8 ∧ students_per_group = 13 →
  bundles * notebooks_per_bundle - groups * students_per_group = 21 := 
by sorry

end notebooks_left_l1806_180651


namespace slope_of_line_l1806_180694

theorem slope_of_line (x y : ℝ) :
  (∀ (x y : ℝ), (x / 4 + y / 5 = 1) → (∃ m b : ℝ, y = m * x + b ∧ m = -5 / 4)) :=
by
  sorry

end slope_of_line_l1806_180694


namespace evaluate_expression_l1806_180689

theorem evaluate_expression : 15 * ((1 / 3 : ℚ) + (1 / 4) + (1 / 6))⁻¹ = 20 := 
by 
  sorry

end evaluate_expression_l1806_180689


namespace compute_r_l1806_180631

noncomputable def r (side_length : ℝ) : ℝ :=
  let a := (0.5 * side_length, 0.5 * side_length)
  let b := (1.5 * side_length, 2.5 * side_length)
  let c := (2.5 * side_length, 1.5 * side_length)
  let ab := Real.sqrt ((b.1 - a.1)^2 + (b.2 - a.2)^2)
  let ac := Real.sqrt ((c.1 - a.1)^2 + (c.2 - a.2)^2)
  let bc := Real.sqrt ((c.1 - b.1)^2 + (c.2 - b.2)^2)
  let s := (ab + ac + bc) / 2
  let area_ABC := Real.sqrt (s * (s - ab) * (s - ac) * (s - bc))
  let circumradius := ab * ac * bc / (4 * area_ABC)
  circumradius - (side_length / 2)

theorem compute_r :
  r 1 = (5 * Real.sqrt 2 - 3) / 6 :=
by
  unfold r
  sorry

end compute_r_l1806_180631


namespace find_symmetric_sequence_l1806_180633

noncomputable def symmetric_sequence (b : ℕ → ℤ) (n : ℕ) : Prop :=
  ∀ k, 1 ≤ k ∧ k ≤ n → b k = b (n - k + 1)

noncomputable def arithmetic_sequence (b : ℕ → ℤ) : Prop :=
  ∃ d, b 2 = b 1 + d ∧ b 3 = b 2 + d ∧ b 4 = b 3 + d

theorem find_symmetric_sequence :
  ∃ b : ℕ → ℤ, symmetric_sequence b 7 ∧ arithmetic_sequence b ∧ b 1 = 2 ∧ b 2 + b 4 = 16 ∧
  (b 1 = 2 ∧ b 2 = 5 ∧ b 3 = 8 ∧ b 4 = 11 ∧ b 5 = 8 ∧ b 6 = 5 ∧ b 7 = 2) :=
by {
  sorry
}

end find_symmetric_sequence_l1806_180633


namespace required_jogging_speed_l1806_180698

-- Definitions based on the conditions
def blocks_to_miles (blocks : ℕ) : ℚ := blocks * (1 / 8 : ℚ)
def time_in_hours (minutes : ℕ) : ℚ := minutes / 60

-- Constants provided by the problem
def beach_distance_in_blocks : ℕ := 16
def ice_cream_melt_time_in_minutes : ℕ := 10

-- The main statement to prove
theorem required_jogging_speed :
  let distance := blocks_to_miles beach_distance_in_blocks
  let time := time_in_hours ice_cream_melt_time_in_minutes
  (distance / time) = 12 := by
  sorry

end required_jogging_speed_l1806_180698


namespace joey_pills_l1806_180663

-- Definitions for the initial conditions
def TypeA_initial := 2
def TypeA_increment := 1

def TypeB_initial := 3
def TypeB_increment := 2

def TypeC_initial := 4
def TypeC_increment := 3

def days := 42

-- Function to calculate the sum of an arithmetic series
def arithmetic_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (a₁ + (a₁ + (n - 1) * d)) / 2

-- The theorem to be proved
theorem joey_pills :
  arithmetic_sum TypeA_initial TypeA_increment days = 945 ∧
  arithmetic_sum TypeB_initial TypeB_increment days = 1848 ∧
  arithmetic_sum TypeC_initial TypeC_increment days = 2751 :=
by sorry

end joey_pills_l1806_180663


namespace number_of_oranges_l1806_180639

def apples : ℕ := 14
def more_oranges : ℕ := 10

theorem number_of_oranges (o : ℕ) (apples_eq : apples = 14) (more_oranges_eq : more_oranges = 10) :
  o = apples + more_oranges :=
by
  sorry

end number_of_oranges_l1806_180639


namespace ice_cost_l1806_180606

def people : Nat := 15
def ice_needed_per_person : Nat := 2
def pack_size : Nat := 10
def cost_per_pack : Nat := 3

theorem ice_cost : 
  let total_ice_needed := people * ice_needed_per_person
  let number_of_packs := total_ice_needed / pack_size
  total_ice_needed = 30 ∧ number_of_packs = 3 ∧ number_of_packs * cost_per_pack = 9 :=
by
  let total_ice_needed := people * ice_needed_per_person
  let number_of_packs := total_ice_needed / pack_size
  have h1 : total_ice_needed = 30 := by sorry
  have h2 : number_of_packs = 3 := by sorry
  have h3 : number_of_packs * cost_per_pack = 9 := by sorry
  exact And.intro h1 (And.intro h2 h3)

end ice_cost_l1806_180606


namespace add_percentages_10_30_15_50_l1806_180610

-- Define the problem conditions:
def ten_percent (x : ℝ) : ℝ := 0.10 * x
def fifteen_percent (y : ℝ) : ℝ := 0.15 * y
def add_percentages (x y : ℝ) : ℝ := ten_percent x + fifteen_percent y

theorem add_percentages_10_30_15_50 :
  add_percentages 30 50 = 10.5 :=
by
  sorry

end add_percentages_10_30_15_50_l1806_180610


namespace find_a_l1806_180607

theorem find_a (a : ℝ) :
  (∀ x, x < 2 → 0 < a - 3 * x) ↔ (a = 6) :=
by
  sorry

end find_a_l1806_180607


namespace repeating_decimal_as_fraction_l1806_180629

-- Define the repeating decimal x as .overline{37}
def x : ℚ := 37 / 99

-- The theorem we need to prove
theorem repeating_decimal_as_fraction : x = 37 / 99 := by
  sorry

end repeating_decimal_as_fraction_l1806_180629


namespace find_value_of_2a_minus_b_l1806_180680

def A : Set ℝ := {x | x < 1 ∨ x > 5}
def B (a b : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ b}

theorem find_value_of_2a_minus_b (a b : ℝ) (h1 : A ∪ B a b = Set.univ) (h2 : A ∩ B a b = {x | 5 < x ∧ x ≤ 6}) : 2 * a - b = -4 :=
by
  sorry

end find_value_of_2a_minus_b_l1806_180680


namespace fraction_of_eggs_hatched_l1806_180685

variable (x : ℚ)
variable (survived_first_month_fraction : ℚ := 3/4)
variable (survived_first_year_fraction : ℚ := 2/5)
variable (geese_survived : ℕ := 100)
variable (total_eggs : ℕ := 500)

theorem fraction_of_eggs_hatched :
  (x * survived_first_month_fraction * survived_first_year_fraction * total_eggs : ℚ) = geese_survived → x = 2/3 :=
by 
  intro h
  sorry

end fraction_of_eggs_hatched_l1806_180685


namespace max_ab_bc_ca_a2_div_bc_plus_b2_div_ca_plus_c2_div_ab_geq_1_over_2_l1806_180621

variable (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
variable (h : a + b + c = 1)

theorem max_ab_bc_ca : ab + bc + ca ≤ 1 / 3 :=
by sorry

theorem a2_div_bc_plus_b2_div_ca_plus_c2_div_ab_geq_1_over_2 :
  (a^2 / (b + c)) + (b^2 / (c + a)) + (c^2 / (a + b)) ≥ 1 / 2 :=
by sorry

end max_ab_bc_ca_a2_div_bc_plus_b2_div_ca_plus_c2_div_ab_geq_1_over_2_l1806_180621


namespace triangle_problem_proof_l1806_180675

-- Given conditions
variables {a b c : ℝ}
variables {A B C : ℝ}
variables (h1 : a * (Real.sin A - Real.sin B) = (c - b) * (Real.sin C + Real.sin B))
variables (h2 : c = Real.sqrt 7)
variables (area : ℝ := 3 * Real.sqrt 3 / 2)

-- Prove angle C = π / 3 and perimeter of triangle
theorem triangle_problem_proof 
(h1 : a * (Real.sin A - Real.sin B) = (c - b) * (Real.sin C + Real.sin B))
(h2 : c = Real.sqrt 7)
(area_condition : (1 / 2) * a * b * (Real.sin C) = area) :
  (C = Real.pi / 3) ∧ (a + b + c = 5 + Real.sqrt 7) := 
by
  sorry

end triangle_problem_proof_l1806_180675


namespace solution_inequality_l1806_180626

open Set

theorem solution_inequality (x : ℝ) : (x > 3 ∨ x < -3) ↔ (x > 9 / x) := by
  sorry

end solution_inequality_l1806_180626


namespace ratio_a_c_l1806_180614

-- Define variables and conditions
variables (a b c d : ℚ)

-- Conditions
def ratio_a_b : Prop := a / b = 5 / 4
def ratio_c_d : Prop := c / d = 4 / 3
def ratio_d_b : Prop := d / b = 1 / 5

-- Theorem statement
theorem ratio_a_c (h1 : ratio_a_b a b)
                  (h2 : ratio_c_d c d)
                  (h3 : ratio_d_b d b) : 
  (a / c = 75 / 16) :=
sorry

end ratio_a_c_l1806_180614


namespace max_area_triangle_after_t_seconds_l1806_180616

-- Define the problem conditions and question
def second_hand_rotation_rate : ℝ := 6 -- degrees per second
def minute_hand_rotation_rate : ℝ := 0.1 -- degrees per second
def perpendicular_angle : ℝ := 90 -- degrees

theorem max_area_triangle_after_t_seconds : 
  ∃ (t : ℝ), (second_hand_rotation_rate - minute_hand_rotation_rate) * t = perpendicular_angle ∧ t = 15 + 15 / 59 :=
by
  -- This is a statement of the proof problem; the proof itself is omitted.
  sorry

end max_area_triangle_after_t_seconds_l1806_180616


namespace prob_white_first_yellow_second_l1806_180635

-- Defining the number of yellow and white balls
def yellow_balls : ℕ := 6
def white_balls : ℕ := 4

-- Defining the total number of balls
def total_balls : ℕ := yellow_balls + white_balls

-- Define the events A and B
def event_A : Prop := true -- event A: drawing a white ball first
def event_B : Prop := true -- event B: drawing a yellow ball second

-- Conditional probability P(B|A)
def prob_B_given_A : ℚ := 6 / (total_balls - 1)

-- Main theorem stating the proof problem
theorem prob_white_first_yellow_second : prob_B_given_A = 2 / 3 :=
by
  sorry

end prob_white_first_yellow_second_l1806_180635


namespace problem1_problem2_l1806_180688

noncomputable def triangle_boscos_condition (a b c A B : ℝ) : Prop :=
  b * Real.cos A = (2 * c + a) * Real.cos (Real.pi - B)

noncomputable def triangle_area (a b c : ℝ) (S : ℝ) : Prop :=
  S = (1 / 2) * a * c * Real.sin (2 * Real.pi / 3)

noncomputable def triangle_perimeter (a b c : ℝ) (P : ℝ) : Prop :=
  P = b + a + c

theorem problem1 (a b c A : ℝ) (h : triangle_boscos_condition a b c A (2 * Real.pi / 3)) : 
  ∃ B : ℝ, B = 2 * Real.pi / 3 :=
by
  sorry

theorem problem2 (a c : ℝ) (b : ℝ := 4) (area : ℝ := Real.sqrt 3) (P : ℝ) (h : triangle_area a b c area) (h_perim : triangle_perimeter a b c P) :
  ∃ x : ℝ, x = 4 + 2 * Real.sqrt 5 :=
by
  sorry

end problem1_problem2_l1806_180688


namespace uncle_bruce_dough_weight_l1806_180669

-- Definitions based on the conditions
variable {TotalChocolate : ℕ} (h1 : TotalChocolate = 13)
variable {ChocolateLeftOver : ℕ} (h2 : ChocolateLeftOver = 4)
variable {ChocolatePercentage : ℝ} (h3 : ChocolatePercentage = 0.2) 
variable {WeightOfDough : ℝ}

-- Target statement expressing the final question and answer
theorem uncle_bruce_dough_weight 
  (h1 : TotalChocolate = 13) 
  (h2 : ChocolateLeftOver = 4) 
  (h3 : ChocolatePercentage = 0.2) : 
  WeightOfDough = 36 := by
  sorry

end uncle_bruce_dough_weight_l1806_180669


namespace least_possible_value_of_expression_l1806_180652

noncomputable def min_expression_value (x : ℝ) : ℝ :=
  (x + 1) * (x + 2) * (x + 3) * (x + 4) + 2023

theorem least_possible_value_of_expression :
  ∃ x : ℝ, min_expression_value x = 2022 :=
by
  sorry

end least_possible_value_of_expression_l1806_180652


namespace problem_part1_problem_part2_l1806_180686

-- Statement part (1)
theorem problem_part1 : ( (2 / 3) - (1 / 4) - (1 / 6) ) * 24 = 6 :=
sorry

-- Statement part (2)
theorem problem_part2 : (-2)^3 + (-9 + (-3)^2 * (1 / 3)) = -14 :=
sorry

end problem_part1_problem_part2_l1806_180686


namespace gadgets_selling_prices_and_total_amount_l1806_180618

def cost_price_mobile : ℕ := 16000
def cost_price_laptop : ℕ := 25000
def cost_price_camera : ℕ := 18000

def loss_percentage_mobile : ℕ := 20
def gain_percentage_laptop : ℕ := 15
def loss_percentage_camera : ℕ := 10

def selling_price_mobile : ℕ := cost_price_mobile - (cost_price_mobile * loss_percentage_mobile / 100)
def selling_price_laptop : ℕ := cost_price_laptop + (cost_price_laptop * gain_percentage_laptop / 100)
def selling_price_camera : ℕ := cost_price_camera - (cost_price_camera * loss_percentage_camera / 100)

def total_amount_received : ℕ := selling_price_mobile + selling_price_laptop + selling_price_camera

theorem gadgets_selling_prices_and_total_amount :
  selling_price_mobile = 12800 ∧
  selling_price_laptop = 28750 ∧
  selling_price_camera = 16200 ∧
  total_amount_received = 57750 := by
  sorry

end gadgets_selling_prices_and_total_amount_l1806_180618


namespace find_g_values_l1806_180622

open Function

-- Defining the function g and its properties
axiom g : ℝ → ℝ
axiom g_domain : ∀ x, 0 ≤ x → 0 ≤ g x
axiom g_proper : ∀ x, 0 ≤ x → 0 ≤ g (g x)
axiom g_func : ∀ x, 0 ≤ x → g (g x) = 3 * x / (x + 3)
axiom g_interval : ∀ x, 2 ≤ x ∧ x ≤ 3 → g x = (x + 1) / 2

-- Problem statement translating to Lean
theorem find_g_values :
  g 2021 = 2021.5 ∧ g (1 / 2021) = 6 := by {
  sorry 
}

end find_g_values_l1806_180622


namespace stratified_sampling_middle_schools_l1806_180647

theorem stratified_sampling_middle_schools (high_schools : ℕ) (middle_schools : ℕ) (elementary_schools : ℕ) (total_selected : ℕ) 
    (h_high_schools : high_schools = 10) (h_middle_schools : middle_schools = 30) (h_elementary_schools : elementary_schools = 60)
    (h_total_selected : total_selected = 20) : 
    middle_schools * (total_selected / (high_schools + middle_schools + elementary_schools)) = 6 := 
by 
  sorry

end stratified_sampling_middle_schools_l1806_180647


namespace total_seeds_l1806_180645

theorem total_seeds (seeds_per_watermelon : ℕ) (number_of_watermelons : ℕ) 
(seeds_each : seeds_per_watermelon = 100)
(watermelons_count : number_of_watermelons = 4) :
(seeds_per_watermelon * number_of_watermelons) = 400 := by
  sorry

end total_seeds_l1806_180645


namespace find_b_l1806_180617

-- Define the problem based on the conditions identified
theorem find_b (b : ℕ) (h₁ : b > 0) (h₂ : (b : ℝ)/(b+15) = 0.75) : b = 45 := 
  sorry

end find_b_l1806_180617


namespace find_y_l1806_180615

-- Given conditions
def x : Int := 129
def student_operation (y : Int) : Int := x * y - 148
def result : Int := 110

-- The theorem statement
theorem find_y :
  ∃ y : Int, student_operation y = result ∧ y = 2 := 
sorry

end find_y_l1806_180615


namespace vertex_on_x_axis_l1806_180619

theorem vertex_on_x_axis (c : ℝ) : (∃ (h : ℝ), (h, 0) = ((-(-8) / (2 * 1)), c - (-8)^2 / (4 * 1))) → c = 16 :=
by
  sorry

end vertex_on_x_axis_l1806_180619


namespace cone_volume_l1806_180643

theorem cone_volume (d : ℝ) (h : ℝ) (π : ℝ) (volume : ℝ) 
  (hd : d = 10) (hh : h = 0.8 * d) (hπ : π = Real.pi) : 
  volume = (200 / 3) * π :=
by
  sorry

end cone_volume_l1806_180643


namespace ratio_of_children_l1806_180630

theorem ratio_of_children (C H : ℕ) 
  (hC1 : C / 8 = 16)
  (hC2 : C * (C / 8) = 512)
  (hH : H * 16 = 512) :
  H / C = 1 / 2 :=
by
  sorry

end ratio_of_children_l1806_180630


namespace graph_represents_two_intersecting_lines_l1806_180649

theorem graph_represents_two_intersecting_lines (x y : ℝ) :
  (x - 1) * (x + y + 2) = (y - 1) * (x + y + 2) → 
  (x + y + 2 = 0 ∨ x = y) ∧ 
  (∃ (x y : ℝ), (x = -1 ∧ y = -1 ∧ x = y ∨ x = -y - 2) ∧ (y = x ∨ y = -x - 2)) :=
by
  sorry

end graph_represents_two_intersecting_lines_l1806_180649


namespace final_score_is_80_l1806_180638

def adam_final_score : ℕ :=
  let first_half := 8
  let second_half := 2
  let points_per_question := 8
  (first_half + second_half) * points_per_question

theorem final_score_is_80 : adam_final_score = 80 := by
  sorry

end final_score_is_80_l1806_180638


namespace simplify_expression_and_find_ratio_l1806_180624

theorem simplify_expression_and_find_ratio:
  ∀ (k : ℤ), (∃ (a b : ℤ), (a = 1 ∧ b = 3) ∧ (6 * k + 18 = 6 * (a * k + b))) →
  (1 : ℤ) / (3 : ℤ) = (1 : ℤ) / (3 : ℤ) :=
by
  intro k
  intro h
  sorry

end simplify_expression_and_find_ratio_l1806_180624


namespace problem_1_problem_2_l1806_180681

noncomputable def f (x : ℝ) : ℝ := |x - 3| + |x + 2|

theorem problem_1 (m : ℝ) (h : ∀ x : ℝ, f x ≥ |m + 1|) : m ≤ 4 :=
by
  sorry

theorem problem_2 (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + 2 * b + c = 4) : 
  1 / (a + b) + 1 / (b + c) ≥ 1 :=
by
  sorry

end problem_1_problem_2_l1806_180681


namespace Sawyer_cleans_in_6_hours_l1806_180644

theorem Sawyer_cleans_in_6_hours (N : ℝ) (S : ℝ) (h1 : S = (2/3) * N) 
                                 (h2 : 1/S + 1/N = 1/3.6) : S = 6 :=
by
  sorry

end Sawyer_cleans_in_6_hours_l1806_180644


namespace find_u_l1806_180640

theorem find_u 
    (a b c p q u : ℝ) 
    (H₁: (∀ x, x^3 + 2*x^2 + 5*x - 8 = 0 → x = a ∨ x = b ∨ x = c))
    (H₂: (∀ x, x^3 + p*x^2 + q*x + u = 0 → x = a+b ∨ x = b+c ∨ x = c+a)) :
    u = 18 :=
by 
    sorry

end find_u_l1806_180640


namespace complement_intersection_l1806_180699

section SetTheory

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 1, 3}
def B : Set ℕ := {2, 3, 4}

theorem complement_intersection (hU : U = {0, 1, 2, 3, 4}) (hA : A = {0, 1, 3}) (hB : B = {2, 3, 4}) : 
  ((U \ A) ∩ B) = {2, 4} :=
by
  sorry

end SetTheory

end complement_intersection_l1806_180699


namespace bus_minibus_seats_l1806_180646

theorem bus_minibus_seats (x y : ℕ) 
    (h1 : x = y + 20) 
    (h2 : 5 * x + 5 * y = 300) : 
    x = 40 ∧ y = 20 := 
by
  sorry

end bus_minibus_seats_l1806_180646


namespace total_cost_of_pets_l1806_180678

theorem total_cost_of_pets 
  (num_puppies num_kittens num_parakeets : ℕ)
  (cost_parakeet cost_puppy cost_kitten : ℕ)
  (h1 : num_puppies = 2)
  (h2 : num_kittens = 2)
  (h3 : num_parakeets = 3)
  (h4 : cost_parakeet = 10)
  (h5 : cost_puppy = 3 * cost_parakeet)
  (h6 : cost_kitten = 2 * cost_parakeet) : 
  num_puppies * cost_puppy + num_kittens * cost_kitten + num_parakeets * cost_parakeet = 130 :=
by
  sorry

end total_cost_of_pets_l1806_180678


namespace gym_class_students_correct_l1806_180659

noncomputable def check_gym_class_studens :=
  let P1 := 15
  let P2 := 5
  let P3 := 12.5
  let P4 := 9.166666666666666
  let P5 := 8.333333333333334
  P1 = P2 + 10 ∧
  P2 = 2 * P3 - 20 ∧
  P3 = P4 + P5 - 5 ∧
  P4 = (1 / 2) * P5 + 5

theorem gym_class_students_correct : check_gym_class_studens := by
  simp [check_gym_class_studens]
  sorry

end gym_class_students_correct_l1806_180659


namespace jane_wins_l1806_180608

/-- Define the total number of possible outcomes and the number of losing outcomes -/
def total_outcomes := 64
def losing_outcomes := 12

/-- Define the probability that Jane wins -/
def jane_wins_probability := (total_outcomes - losing_outcomes) / total_outcomes

/-- Problem: Jane wins with a probability of 13/16 given the conditions -/
theorem jane_wins :
  jane_wins_probability = 13 / 16 :=
sorry

end jane_wins_l1806_180608


namespace milly_needs_flamingoes_l1806_180672

theorem milly_needs_flamingoes
  (flamingo_feathers : ℕ)
  (pluck_percent : ℚ)
  (num_boas : ℕ)
  (feathers_per_boa : ℕ)
  (pluckable_feathers_per_flamingo : ℕ)
  (total_feathers_needed : ℕ)
  (num_flamingoes : ℕ)
  (h1 : flamingo_feathers = 20)
  (h2 : pluck_percent = 0.25)
  (h3 : num_boas = 12)
  (h4 : feathers_per_boa = 200)
  (h5 : pluckable_feathers_per_flamingo = flamingo_feathers * pluck_percent)
  (h6 : total_feathers_needed = num_boas * feathers_per_boa)
  (h7 : num_flamingoes = total_feathers_needed / pluckable_feathers_per_flamingo)
  : num_flamingoes = 480 := 
by
  sorry

end milly_needs_flamingoes_l1806_180672


namespace tylenol_interval_l1806_180670

/-- Mark takes 2 Tylenol tablets of 500 mg each at certain intervals for 12 hours, and he ends up taking 3 grams of Tylenol in total. Prove that the interval in hours at which he takes the tablets is 2.4 hours. -/
theorem tylenol_interval 
    (total_dose_grams : ℝ)
    (tablet_mg : ℝ)
    (hours : ℝ)
    (tablets_taken_each_time : ℝ) 
    (total_tablets : ℝ) 
    (interval_hours : ℝ) :
    total_dose_grams = 3 → 
    tablet_mg = 500 → 
    hours = 12 → 
    tablets_taken_each_time = 2 → 
    total_tablets = (total_dose_grams * 1000) / tablet_mg → 
    interval_hours = hours / (total_tablets / tablets_taken_each_time - 1) → 
    interval_hours = 2.4 :=
by
  intros
  sorry

end tylenol_interval_l1806_180670


namespace smallest_positive_period_l1806_180637

open Real

-- Define conditions
def max_value_condition (b a : ℝ) : Prop := b + a = -1
def min_value_condition (b a : ℝ) : Prop := b - a = -5

-- Define the period of the function
def period (f : ℝ → ℝ) (T : ℝ) : Prop := ∀ x, f (x + T) = f x

-- Main theorem
theorem smallest_positive_period (a b : ℝ) (h1 : a < 0) 
  (h2 : max_value_condition b a) 
  (h3 : min_value_condition b a) : 
  period (fun x => tan ((3 * a + b) * x)) (π / 9) :=
by
  sorry

end smallest_positive_period_l1806_180637


namespace find_divisor_l1806_180620

-- Define the problem specifications
def divisor_problem (D Q R d : ℕ) : Prop :=
  D = d * Q + R

-- The specific instance with given values
theorem find_divisor :
  divisor_problem 15968 89 37 179 :=
by
  -- Proof omitted
  sorry

end find_divisor_l1806_180620


namespace number_of_men_l1806_180671

variable (M : ℕ)

-- Define the first condition: M men reaping 80 hectares in 24 days.
def first_work_rate (M : ℕ) : ℚ := (80 : ℚ) / (M * 24)

-- Define the second condition: 36 men reaping 360 hectares in 30 days.
def second_work_rate : ℚ := (360 : ℚ) / (36 * 30)

-- Lean 4 statement: Prove the equivalence given conditions.
theorem number_of_men (h : first_work_rate M = second_work_rate) : M = 45 :=
by
  sorry

end number_of_men_l1806_180671


namespace year_2023_ad_is_written_as_positive_2023_l1806_180612

theorem year_2023_ad_is_written_as_positive_2023 :
  (∀ (year : Int), year = -500 → year = -500) → -- This represents the given condition that year 500 BC is -500
  (∀ (year : Int), year > 0) → -- This represents the condition that AD years are postive
  2023 = 2023 := -- The problem conclusion

by
  intros
  trivial -- The solution is quite trivial due to the conditions.

end year_2023_ad_is_written_as_positive_2023_l1806_180612


namespace geometric_sequence_a_div_n_sum_first_n_terms_l1806_180662

variable {a : ℕ → ℝ} -- sequence a_n
variable {S : ℕ → ℝ} -- sum of first n terms S_n

axiom S_recurrence {n : ℕ} (hn : n > 0) : 
  S (n + 1) = S n + (n + 1) / (3 * n) * a n

axiom a_1 : a 1 = 1

theorem geometric_sequence_a_div_n :
  ∃ (r : ℝ), ∀ {n : ℕ} (hn : n > 0), (a n / n) = r^n := 
sorry

theorem sum_first_n_terms (n : ℕ) :
  S n = (9 / 4) - ((9 / 4) + (3 * n / 2)) * (1 / 3) ^ n :=
sorry

end geometric_sequence_a_div_n_sum_first_n_terms_l1806_180662


namespace directrix_of_parabola_l1806_180627

theorem directrix_of_parabola (x y : ℝ) : y = 3 * x^2 - 6 * x + 1 → y = -25 / 12 :=
sorry

end directrix_of_parabola_l1806_180627


namespace shirts_sold_l1806_180676

theorem shirts_sold (pants shorts shirts jackets credit_remaining : ℕ) 
  (price_shirt1 price_shirt2 price_pants : ℕ) 
  (discount tax : ℝ) :
  (pants = 3) →
  (shorts = 5) →
  (jackets = 2) →
  (price_shirt1 = 10) →
  (price_shirt2 = 12) →
  (price_pants = 15) →
  (discount = 0.10) →
  (tax = 0.05) →
  (credit_remaining = 25) →
  (store_credit : ℕ) →
  (store_credit = pants * 5 + shorts * 3 + jackets * 7 + shirts * 4) →
  (total_cost : ℝ) →
  (total_cost = (price_shirt1 + price_shirt2 + price_pants) * (1 - discount) * (1 + tax)) →
  (total_store_credit_used : ℝ) →
  (total_store_credit_used = total_cost - credit_remaining) →
  (initial_credit : ℝ) →
  (initial_credit = total_store_credit_used + (pants * 5 + shorts * 3 + jackets * 7)) →
  shirts = 2 :=
by
  intros
  sorry

end shirts_sold_l1806_180676


namespace quadratic_inequality_l1806_180684

theorem quadratic_inequality (a b c : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0) 
  (h : ∀ x : ℝ, a * x^2 + b * x + c > c * x) : ∀ x : ℝ, c * x^2 - b * x + a > c * x - b := 
by
  sorry

end quadratic_inequality_l1806_180684


namespace perimeter_of_region_proof_l1806_180682

noncomputable def perimeter_of_region (total_area : ℕ) (num_squares : ℕ) (arrangement : String) : ℕ :=
  if total_area = 512 ∧ num_squares = 8 ∧ arrangement = "vertical rectangle" then 160 else 0

theorem perimeter_of_region_proof :
  perimeter_of_region 512 8 "vertical rectangle" = 160 :=
by
  sorry

end perimeter_of_region_proof_l1806_180682


namespace calculation_l1806_180604

theorem calculation : 120 / 5 / 3 * 2 = 16 := by
  sorry

end calculation_l1806_180604


namespace b_completion_days_l1806_180668

theorem b_completion_days (x : ℝ) :
  (7 * (1 / 24 + 1 / x + 1 / 40) + 4 * (1 / 24 + 1 / x) = 1) → x = 26.25 := 
by 
  sorry

end b_completion_days_l1806_180668


namespace total_questions_to_review_is_1750_l1806_180679

-- Define the relevant conditions
def num_classes := 5
def students_per_class := 35
def questions_per_exam := 10

-- The total number of questions to be reviewed by Professor Oscar
def total_questions : Nat := num_classes * students_per_class * questions_per_exam

-- The theorem stating the equivalent proof problem
theorem total_questions_to_review_is_1750 : total_questions = 1750 := by
  -- proof steps are skipped here 
  sorry

end total_questions_to_review_is_1750_l1806_180679


namespace hallway_length_l1806_180658

theorem hallway_length (s t d : ℝ) (h1 : 3 * s * t = 12) (h2 : s * t = d - 12) : d = 16 :=
sorry

end hallway_length_l1806_180658


namespace box_mass_calculation_l1806_180656

variable (h₁ w₁ l₁ : ℝ) (m₁ : ℝ)
variable (h₂ w₂ l₂ density₁ density₂ : ℝ)

theorem box_mass_calculation
  (h₁_eq : h₁ = 3)
  (w₁_eq : w₁ = 4)
  (l₁_eq : l₁ = 6)
  (m₁_eq : m₁ = 72)
  (h₂_eq : h₂ = 1.5 * h₁)
  (w₂_eq : w₂ = 2.5 * w₁)
  (l₂_eq : l₂ = l₁)
  (density₂_eq : density₂ = 2 * density₁)
  (density₁_eq : density₁ = m₁ / (h₁ * w₁ * l₁)) :
  h₂ * w₂ * l₂ * density₂ = 540 := by
  sorry

end box_mass_calculation_l1806_180656


namespace smallest_x_of_quadratic_eqn_l1806_180650

theorem smallest_x_of_quadratic_eqn : ∃ x : ℝ, (12*x^2 - 44*x + 40 = 0) ∧ x = 5 / 3 :=
by
  sorry

end smallest_x_of_quadratic_eqn_l1806_180650


namespace min_sum_of_abc_conditions_l1806_180611

theorem min_sum_of_abc_conditions
  (a b c d : ℕ)
  (hab : a + b = 2)
  (hac : a + c = 3)
  (had : a + d = 4)
  (hbc : b + c = 5)
  (hbd : b + d = 6)
  (hcd : c + d = 7) :
  a + b + c + d = 9 :=
sorry

end min_sum_of_abc_conditions_l1806_180611


namespace longest_side_of_triangle_l1806_180655

theorem longest_side_of_triangle (a d : ℕ) (h1 : d = 2) (h2 : a - d > 0) (h3 : a + d > 0)
    (h_angle : ∃ C : ℝ, C = 120) 
    (h_arith_seq : ∃ (b c : ℕ), b = a - d ∧ c = a ∧ b + 2 * d = c + d) : 
    a + d = 7 :=
by
  -- The proof will be provided here
  sorry

end longest_side_of_triangle_l1806_180655


namespace problem_solution_l1806_180653

open Function

-- Definitions of the points
structure Point where
  x : ℝ
  y : ℝ

def A : Point := ⟨-3, 2⟩
def B : Point := ⟨1, 0⟩
def C : Point := ⟨4, 1⟩
def D : Point := ⟨-2, 4⟩

-- Definitions of vectors
def vec (P Q : Point) : Point := ⟨Q.x - P.x, Q.y - P.y⟩

-- Definitions of conditions
def AB := vec A B
def AD := vec A D
def DC := vec D C

-- Definitions of dot product to check orthogonality
def dot (v w : Point) : ℝ := v.x * w.x + v.y * w.y

-- Lean statement to prove the conditions
theorem problem_solution :
  AB ≠ ⟨-4, 2⟩ ∧
  dot AB AD = 0 ∧
  AB.y * DC.x = AB.x * DC.y ∧
  ((AB.y * DC.x = AB.x * DC.y) ∧ (dot AB AD = 0) → 
  (∃ a b : ℝ, a ≠ b ∧ (a = 0 ∨ b = 0) ∧ AB = ⟨a, -a⟩  ∧ DC = ⟨3 * a, -3 * a⟩)) :=
by
  -- Proof omitted
  sorry

end problem_solution_l1806_180653


namespace average_marks_l1806_180602

-- Define the conditions
variables (M P C : ℝ)
variables (h1 : M + P = 60) (h2 : C = P + 10)

-- Define the theorem statement
theorem average_marks : (M + C) / 2 = 35 :=
by {
  sorry -- Placeholder for the proof.
}

end average_marks_l1806_180602


namespace maria_total_cost_l1806_180625

-- Define the costs of the items
def pencil_cost : ℕ := 8
def pen_cost : ℕ := pencil_cost / 2
def eraser_cost : ℕ := 2 * pen_cost

-- Define the total cost
def total_cost : ℕ := pen_cost + pencil_cost + eraser_cost

-- The theorem to prove
theorem maria_total_cost : total_cost = 20 := by
  sorry

end maria_total_cost_l1806_180625


namespace odd_square_mod_eight_l1806_180623

theorem odd_square_mod_eight (k : ℤ) : ((2 * k + 1) ^ 2) % 8 = 1 := 
sorry

end odd_square_mod_eight_l1806_180623


namespace polynomial_simplification_l1806_180695

theorem polynomial_simplification (x : ℝ) : 
  (3*x - 2)*(5*x^12 + 3*x^11 + 2*x^10 - x^9) = 15*x^13 - x^12 - 7*x^10 + 2*x^9 :=
by {
  sorry
}

end polynomial_simplification_l1806_180695


namespace ratio_of_areas_is_correct_l1806_180673

-- Definition of the lengths of the sides of the triangles
def triangle_XYZ_sides := (7, 24, 25)
def triangle_PQR_sides := (9, 40, 41)

-- Definition of the areas of the right triangles
def area_triangle_XYZ := (7 * 24) / 2
def area_triangle_PQR := (9 * 40) / 2

-- The ratio of the areas of the triangles
def ratio_of_areas := area_triangle_XYZ / area_triangle_PQR

-- The expected answer
def expected_ratio := 7 / 15

-- The theorem proving that ratio_of_areas is equal to expected_ratio
theorem ratio_of_areas_is_correct :
  ratio_of_areas = expected_ratio := by
  -- Add the proof here
  sorry

end ratio_of_areas_is_correct_l1806_180673


namespace distinct_ordered_pairs_eq_49_l1806_180697

theorem distinct_ordered_pairs_eq_49 (x y : ℕ) (hx : 1 ≤ x ∧ x ≤ 49) (hy : 1 ≤ y ∧ y ≤ 49) (h_eq : x + y = 50) :
  ∃ xs : List (ℕ × ℕ), (∀ p ∈ xs, p.1 + p.2 = 50 ∧ 1 ≤ p.1 ∧ p.1 ≤ 49 ∧ 1 ≤ p.2 ∧ p.2 ≤ 49) ∧ xs.length = 49 :=
sorry

end distinct_ordered_pairs_eq_49_l1806_180697


namespace product_of_two_large_integers_l1806_180642

theorem product_of_two_large_integers :
  ∃ a b : ℕ, a > 2009^182 ∧ b > 2009^182 ∧ 3^2008 + 4^2009 = a * b :=
by { sorry }

end product_of_two_large_integers_l1806_180642


namespace ratio_Laura_to_Ken_is_2_to_1_l1806_180696

def Don_paint_tiles_per_minute : ℕ := 3

def Ken_paint_tiles_per_minute : ℕ := Don_paint_tiles_per_minute + 2

def multiple : ℕ := sorry -- Needs to be introduced, not directly from the solution steps

def Laura_paint_tiles_per_minute : ℕ := multiple * Ken_paint_tiles_per_minute

def Kim_paint_tiles_per_minute : ℕ := Laura_paint_tiles_per_minute - 3

def total_tiles_in_15_minutes : ℕ := 375

def total_tiles_per_minute : ℕ := total_tiles_in_15_minutes / 15

def total_tiles_equation : Prop :=
  Don_paint_tiles_per_minute + Ken_paint_tiles_per_minute + Laura_paint_tiles_per_minute + Kim_paint_tiles_per_minute = total_tiles_per_minute

theorem ratio_Laura_to_Ken_is_2_to_1 :
  (total_tiles_equation → Laura_paint_tiles_per_minute / Ken_paint_tiles_per_minute = 2) := sorry

end ratio_Laura_to_Ken_is_2_to_1_l1806_180696


namespace chocolate_cookies_initial_count_l1806_180693

theorem chocolate_cookies_initial_count
  (andy_ate : ℕ) (brother : ℕ) (friends_each : ℕ) (num_friends : ℕ)
  (team_members : ℕ) (first_share : ℕ) (common_diff : ℕ)
  (last_member_share : ℕ) (total_sum_team : ℕ)
  (total_cookies : ℕ) :
  andy_ate = 4 →
  brother = 6 →
  friends_each = 2 →
  num_friends = 3 →
  team_members = 10 →
  first_share = 2 →
  common_diff = 2 →
  last_member_share = first_share + (team_members - 1) * common_diff →
  total_sum_team = team_members / 2 * (first_share + last_member_share) →
  total_cookies = andy_ate + brother + (friends_each * num_friends) + total_sum_team →
  total_cookies = 126 :=
by
  intros ha hb hf hn ht hf1 hc hl hs ht
  sorry

end chocolate_cookies_initial_count_l1806_180693


namespace quadratic_inequality_l1806_180687

theorem quadratic_inequality (a b c : ℝ) (h : (a + b + c) * c < 0) : b^2 > 4 * a * c :=
sorry

end quadratic_inequality_l1806_180687


namespace joanne_total_weekly_earnings_l1806_180692

-- Define the earnings per hour and hours worked per day for the main job
def mainJobHourlyWage : ℝ := 16
def mainJobDailyHours : ℝ := 8

-- Compute daily earnings from the main job
def mainJobDailyEarnings : ℝ := mainJobHourlyWage * mainJobDailyHours

-- Define the earnings per hour and hours worked per day for the part-time job
def partTimeJobHourlyWage : ℝ := 13.5
def partTimeJobDailyHours : ℝ := 2

-- Compute daily earnings from the part-time job
def partTimeJobDailyEarnings : ℝ := partTimeJobHourlyWage * partTimeJobDailyHours

-- Compute total daily earnings from both jobs
def totalDailyEarnings : ℝ := mainJobDailyEarnings + partTimeJobDailyEarnings

-- Define the number of workdays per week
def workDaysPerWeek : ℝ := 5

-- Compute total weekly earnings
def totalWeeklyEarnings : ℝ := totalDailyEarnings * workDaysPerWeek

-- The problem statement to prove: Joanne's total weekly earnings = 775
theorem joanne_total_weekly_earnings :
  totalWeeklyEarnings = 775 :=
by
  sorry

end joanne_total_weekly_earnings_l1806_180692
