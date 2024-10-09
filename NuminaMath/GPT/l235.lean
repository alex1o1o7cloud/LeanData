import Mathlib

namespace cos_alpha_values_l235_23585

theorem cos_alpha_values (α : ℝ) (h : Real.sin (π + α) = -3 / 5) :
  Real.cos α = 4 / 5 ∨ Real.cos α = -4 / 5 := 
sorry

end cos_alpha_values_l235_23585


namespace sum_of_digits_0_to_2012_l235_23520

def sum_of_digits (n : Nat) : Nat :=
  (Nat.digits 10 n).sum

def sum_of_digits_in_range (a b : Nat) : Nat :=
  ((List.range (b + 1)).drop a).map sum_of_digits |>.sum

theorem sum_of_digits_0_to_2012 : 
  sum_of_digits_in_range 0 2012 = 28077 := 
by
  sorry

end sum_of_digits_0_to_2012_l235_23520


namespace find_m_l235_23557

theorem find_m (m : ℝ) :
  (∃ x : ℝ, (m - 1) * x^2 + 2 * x + m^2 - 1 = 0) ∧ (m - 1 ≠ 0) → m = -1 :=
by
  sorry

end find_m_l235_23557


namespace solve_for_x_l235_23562

theorem solve_for_x : (42 / (7 - 3 / 7) = 147 / 23) :=
by
  sorry

end solve_for_x_l235_23562


namespace number_of_red_balls_l235_23524

theorem number_of_red_balls
    (black_balls : ℕ)
    (frequency : ℝ)
    (total_balls : ℕ)
    (red_balls : ℕ) 
    (h_black : black_balls = 5)
    (h_frequency : frequency = 0.25)
    (h_total : total_balls = black_balls / frequency) :
    red_balls = total_balls - black_balls → red_balls = 15 :=
by
  intros h_red
  sorry

end number_of_red_balls_l235_23524


namespace smallest_perimeter_of_square_sides_l235_23591

/-
  Define a predicate for the triangle inequality condition for squares of integers.
-/
def triangle_ineq_squares (a b c : ℕ) : Prop :=
  (a < b) ∧ (b < c) ∧ (a^2 + b^2 > c^2) ∧ (a^2 + c^2 > b^2) ∧ (b^2 + c^2 > a^2)

/-
  Statement that proves the smallest possible perimeter given the conditions.
-/
theorem smallest_perimeter_of_square_sides : 
  ∃ a b c : ℕ, a < b ∧ b < c ∧ triangle_ineq_squares a b c ∧ a^2 + b^2 + c^2 = 77 :=
sorry

end smallest_perimeter_of_square_sides_l235_23591


namespace surface_area_of_tunneled_cube_l235_23537

-- Definition of the initial cube and its properties.
def cube (side_length : ℕ) := side_length * side_length * side_length

-- Initial side length of the large cube
def large_cube_side : ℕ := 12

-- Each small cube side length
def small_cube_side : ℕ := 3

-- Number of small cubes that fit into the large cube
def num_small_cubes : ℕ := (cube large_cube_side) / (cube small_cube_side)

-- Number of cubes removed initially
def removed_cubes : ℕ := 27

-- Number of remaining cubes after initial removal
def remaining_cubes : ℕ := num_small_cubes - removed_cubes

-- Surface area of each unmodified small cube
def small_cube_surface : ℕ := 54

-- Additional surface area due to removal of center units
def additional_surface : ℕ := 24

-- Surface area of each modified small cube
def modified_cube_surface : ℕ := small_cube_surface + additional_surface

-- Total surface area before adjustment for shared faces
def total_surface_before_adjustment : ℕ := remaining_cubes * modified_cube_surface

-- Shared surface area to be subtracted
def shared_surface : ℕ := 432

-- Final surface area of the resulting figure
def final_surface_area : ℕ := total_surface_before_adjustment - shared_surface

-- Theorem statement
theorem surface_area_of_tunneled_cube : final_surface_area = 2454 :=
by {
  -- Proof required here
  sorry
}

end surface_area_of_tunneled_cube_l235_23537


namespace part_I_part_II_l235_23586

noncomputable def f (a x : ℝ) : ℝ := |x - 1| + a * |x - 2|

theorem part_I (a : ℝ) (h_min : ∃ m, ∀ x, f a x ≥ m) : -1 ≤ a ∧ a ≤ 1 :=
sorry

theorem part_II (a : ℝ) (h_bound : ∀ x, f a x ≥ 1/2) : a = 1/3 :=
sorry

end part_I_part_II_l235_23586


namespace distinct_valid_sets_count_l235_23559

-- Define non-negative powers of 2 and 3
def is_non_neg_power (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = 2^a ∨ n = 3^b

-- Define the condition for sum of elements in set S to be 2014
def valid_sets (S : Finset ℕ) : Prop :=
  (∀ x ∈ S, is_non_neg_power x) ∧ (S.sum id = 2014)

theorem distinct_valid_sets_count : ∃ (number_of_distinct_sets : ℕ), number_of_distinct_sets = 64 :=
  sorry

end distinct_valid_sets_count_l235_23559


namespace row_speed_with_stream_l235_23552

theorem row_speed_with_stream (v : ℝ) (s : ℝ) (h1 : s = 2) (h2 : v - s = 12) : v + s = 16 := by
  -- Placeholder for the proof
  sorry

end row_speed_with_stream_l235_23552


namespace original_triangle_area_l235_23551

-- Define the variables
variable (A_new : ℝ) (r : ℝ)

-- The conditions from the problem
def conditions := r = 5 ∧ A_new = 100

-- Goal: Prove that the original area is 4
theorem original_triangle_area (A_orig : ℝ) (h : conditions r A_new) : A_orig = 4 := by
  sorry

end original_triangle_area_l235_23551


namespace average_score_girls_l235_23542

theorem average_score_girls (num_boys num_girls : ℕ) (avg_boys avg_class : ℕ) : 
  num_boys = 12 → 
  num_girls = 4 → 
  avg_boys = 84 → 
  avg_class = 86 → 
  ∃ avg_girls : ℕ, avg_girls = 92 :=
by
  intros h1 h2 h3 h4
  sorry

end average_score_girls_l235_23542


namespace tan_product_identity_l235_23568

theorem tan_product_identity : (1 + Real.tan (Real.pi / 180 * 17)) * (1 + Real.tan (Real.pi / 180 * 28)) = 2 := by
  sorry

end tan_product_identity_l235_23568


namespace and_15_and_l235_23590

def x_and (x : ℝ) : ℝ := 8 - x
def and_x (x : ℝ) : ℝ := x - 8

theorem and_15_and : and_x (x_and 15) = -15 :=
by
  sorry

end and_15_and_l235_23590


namespace min_value_of_a_for_inverse_l235_23579

theorem min_value_of_a_for_inverse (a : ℝ) : 
  (∀ x y : ℝ, x ≥ a → y ≥ a → (x^2 + 4*x ≤ y^2 + 4*y ↔ x ≤ y)) → a = -2 :=
by
  sorry

end min_value_of_a_for_inverse_l235_23579


namespace complement_union_l235_23553

open Set

universe u

variable {U : Type u} [Fintype U] [DecidableEq U]
variable {A B : Set U}

def complement (s : Set U) : Set U := {x | x ∉ s}

theorem complement_union {U : Set ℕ} (A B : Set ℕ) 
  (h1 : complement A ∩ B = {1})
  (h2 : A ∩ B = {3})
  (h3 : complement A ∩ complement B = {2}) :
  complement (A ∪ B) = {2} :=
by sorry

end complement_union_l235_23553


namespace max_problems_to_miss_to_pass_l235_23581

theorem max_problems_to_miss_to_pass (total_problems : ℕ) (pass_percentage : ℝ) :
  total_problems = 50 → pass_percentage = 0.85 → 7 = ↑total_problems * (1 - pass_percentage) :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end max_problems_to_miss_to_pass_l235_23581


namespace gerald_paid_l235_23588

theorem gerald_paid (G : ℝ) (h : 0.8 * G = 200) : G = 250 := by
  sorry

end gerald_paid_l235_23588


namespace eq_squares_diff_l235_23539

theorem eq_squares_diff {x y z : ℝ} :
  x = (y - z)^2 ∧ y = (x - z)^2 ∧ z = (x - y)^2 →
  (x = 0 ∧ y = 0 ∧ z = 0) ∨ 
  (x = 1 ∧ y = 1 ∧ z = 0) ∨ 
  (x = 0 ∧ y = 1 ∧ z = 1) ∨ 
  (x = 1 ∧ y = 0 ∧ z = 1) :=
by
  sorry

end eq_squares_diff_l235_23539


namespace solution_set_abs_inequality_l235_23544

theorem solution_set_abs_inequality (x : ℝ) :
  (|2 - x| ≥ 1) ↔ (x ≤ 1 ∨ x ≥ 3) :=
by
  sorry

end solution_set_abs_inequality_l235_23544


namespace total_oranges_picked_l235_23521

-- Defining the number of oranges picked by Mary, Jason, and Sarah
def maryOranges := 122
def jasonOranges := 105
def sarahOranges := 137

-- The theorem to prove that the total number of oranges picked is 364
theorem total_oranges_picked : maryOranges + jasonOranges + sarahOranges = 364 := by
  sorry

end total_oranges_picked_l235_23521


namespace sin_inequality_of_triangle_l235_23546

theorem sin_inequality_of_triangle (B C : ℝ) (hB : 0 < B) (hB_lt_pi : B < π) 
(hC : 0 < C) (hC_lt_pi : C < π) :
  (B > C) ↔ (Real.sin B > Real.sin C) := 
  sorry

end sin_inequality_of_triangle_l235_23546


namespace isosceles_obtuse_triangle_l235_23507

theorem isosceles_obtuse_triangle (A B C : ℝ) (h_isosceles: A = B)
  (h_obtuse: A + B + C = 180) 
  (h_max_angle: C = 157.5): A = 11.25 :=
by
  sorry

end isosceles_obtuse_triangle_l235_23507


namespace A_remaining_time_equals_B_remaining_time_l235_23547

variable (d_A d_B remaining_Distance_A remaining_Time_A remaining_Distance_B remaining_Time_B total_Distance : ℝ)

-- Given conditions as definitions
def A_traveled_more : d_A = d_B + 180 := sorry
def total_distance_between_X_Y : total_Distance = 900 := sorry
def sum_distance_traveled : d_A + d_B = total_Distance := sorry
def B_remaining_time : remaining_Time_B = 4.5 := sorry
def B_remaining_distance : remaining_Distance_B = total_Distance - d_B := sorry

-- Prove that: A travels the same remaining distance in the same time as B
theorem A_remaining_time_equals_B_remaining_time :
  remaining_Distance_A = remaining_Distance_B ∧ remaining_Time_A = remaining_Time_B := sorry

end A_remaining_time_equals_B_remaining_time_l235_23547


namespace determine_h_l235_23530

noncomputable def h (x : ℝ) : ℝ :=
  -12*x^4 + 2*x^3 + 8*x^2 - 8*x + 3

theorem determine_h :
  (12*x^4 + 4*x^3 - 2*x + 3 + h x = 6*x^3 + 8*x^2 - 10*x + 6) ↔
  (h x = -12*x^4 + 2*x^3 + 8*x^2 - 8*x + 3) :=
by 
  sorry

end determine_h_l235_23530


namespace total_bills_inserted_l235_23517

theorem total_bills_inserted (x y : ℕ) (h1 : x = 175) (h2 : x + 5 * y = 300) : 
  x + y = 200 :=
by {
  -- Since we focus strictly on the statement per instruction, the proof is omitted
  sorry 
}

end total_bills_inserted_l235_23517


namespace max_profit_l235_23516

noncomputable def total_cost (Q : ℝ) : ℝ := 5 * Q^2

noncomputable def demand_non_slytherin (P : ℝ) : ℝ := 26 - 2 * P

noncomputable def demand_slytherin (P : ℝ) : ℝ := 10 - P

noncomputable def combined_demand (P : ℝ) : ℝ :=
  if P >= 13 then demand_non_slytherin P else demand_non_slytherin P + demand_slytherin P

noncomputable def inverse_demand (Q : ℝ) : ℝ :=
  if Q <= 6 then 13 - Q / 2 else 12 - Q / 3

noncomputable def revenue (Q : ℝ) : ℝ :=
  if Q <= 6 then Q * (13 - Q / 2) else Q * (12 - Q / 3)

noncomputable def marginal_revenue (Q : ℝ) : ℝ :=
  if Q <= 6 then 13 - Q else 12 - 2 * Q / 3

noncomputable def marginal_cost (Q : ℝ) : ℝ := 10 * Q

theorem max_profit :
  ∃ Q P TR TC π,
    P = inverse_demand Q ∧
    TR = P * Q ∧
    TC = total_cost Q ∧
    π = TR - TC ∧
    π = 7.69 :=
sorry

end max_profit_l235_23516


namespace ratio_of_numbers_l235_23548

-- Definitions for the conditions
variable (S L : ℕ)

-- Given conditions
def condition1 : Prop := S + L = 44
def condition2 : Prop := S = 20
def condition3 : Prop := L = 6 * S

-- The theorem to be proven
theorem ratio_of_numbers (h1 : condition1 S L) (h2 : condition2 S) (h3 : condition3 S L) : L / S = 6 := 
  sorry

end ratio_of_numbers_l235_23548


namespace intersection_is_singleton_zero_l235_23534

-- Define the sets M and N
def M : Set ℤ := {-1, 0, 1, 2, 3}
def N : Set ℤ := {-2, 0}

-- Define the theorem to be proved
theorem intersection_is_singleton_zero : M ∩ N = {0} :=
by
  -- Proof is provided by the steps above but not needed here
  sorry

end intersection_is_singleton_zero_l235_23534


namespace negation_of_exists_is_forall_l235_23523

theorem negation_of_exists_is_forall :
  (¬ ∃ x : ℝ, x^3 + 1 = 0) ↔ ∀ x : ℝ, x^3 + 1 ≠ 0 :=
by 
  sorry

end negation_of_exists_is_forall_l235_23523


namespace range_of_a_l235_23509

theorem range_of_a (a : ℝ) (h : 2 * a - 1 ≤ 11) : a < 6 :=
by
  sorry

end range_of_a_l235_23509


namespace total_donation_l235_23518

-- Define the conditions in the problem
def Barbara_stuffed_animals : ℕ := 9
def Trish_stuffed_animals : ℕ := 2 * Barbara_stuffed_animals
def Barbara_sale_price : ℝ := 2
def Trish_sale_price : ℝ := 1.5

-- Define the goal as a theorem to be proven
theorem total_donation : Barbara_sale_price * Barbara_stuffed_animals + Trish_sale_price * Trish_stuffed_animals = 45 := by
  sorry

end total_donation_l235_23518


namespace eating_contest_l235_23595

variables (hotdog_weight burger_weight pie_weight : ℕ)
variable (noah_burgers jacob_pies mason_hotdogs : ℕ)
variable (total_weight_mason_hotdogs : ℕ)

theorem eating_contest :
  hotdog_weight = 2 →
  burger_weight = 5 →
  pie_weight = 10 →
  noah_burgers = 8 →
  jacob_pies = noah_burgers - 3 →
  mason_hotdogs = 3 * jacob_pies →
  total_weight_mason_hotdogs = mason_hotdogs * hotdog_weight →
  total_weight_mason_hotdogs = 30 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end eating_contest_l235_23595


namespace emily_purchased_9_wall_prints_l235_23505

/-
  Given the following conditions:
  - cost_of_each_pair_of_curtains = 30
  - num_of_pairs_of_curtains = 2
  - installation_cost = 50
  - cost_of_each_wall_print = 15
  - total_order_cost = 245

  Prove that Emily purchased 9 wall prints
-/
noncomputable def num_wall_prints_purchased 
  (cost_of_each_pair_of_curtains : ℝ) 
  (num_of_pairs_of_curtains : ℝ) 
  (installation_cost : ℝ) 
  (cost_of_each_wall_print : ℝ) 
  (total_order_cost : ℝ) 
  : ℝ :=
  (total_order_cost - (num_of_pairs_of_curtains * cost_of_each_pair_of_curtains + installation_cost)) / cost_of_each_wall_print

theorem emily_purchased_9_wall_prints
  (cost_of_each_pair_of_curtains : ℝ := 30) 
  (num_of_pairs_of_curtains : ℝ := 2) 
  (installation_cost : ℝ := 50) 
  (cost_of_each_wall_print : ℝ := 15) 
  (total_order_cost : ℝ := 245) :
  num_wall_prints_purchased cost_of_each_pair_of_curtains num_of_pairs_of_curtains installation_cost cost_of_each_wall_print total_order_cost = 9 :=
sorry

end emily_purchased_9_wall_prints_l235_23505


namespace find_value_of_a5_l235_23549

variable (a : ℕ → ℝ)

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a_1 d : ℝ), ∀ n, a n = a_1 + (n - 1) * d

variable (h_arith : is_arithmetic_sequence a)
variable (h : a 2 + a 8 = 12)

theorem find_value_of_a5 : a 5 = 6 :=
by
  sorry

end find_value_of_a5_l235_23549


namespace trapezoid_area_l235_23527

noncomputable def area_trapezoid : ℝ :=
  let x1 := 10
  let x2 := -10
  let y1 := 10
  let h := 10
  let a := 20  -- length of top side at y = 10
  let b := 10  -- length of lower side
  (a + b) * h / 2

theorem trapezoid_area : area_trapezoid = 150 := by
  sorry

end trapezoid_area_l235_23527


namespace min_value_proof_l235_23593

noncomputable def min_value (x y : ℝ) : ℝ :=
  (y / x) + (1 / y)

theorem min_value_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + 2 * y = 1) : 
  (min_value x y) ≥ 4 :=
by
  sorry

end min_value_proof_l235_23593


namespace evaluate_expression_l235_23515

theorem evaluate_expression (m n : ℤ) (hm : m = 2) (hn : n = -3) : (m + n) ^ 2 - 2 * m * (m + n) = 5 := by
  -- Proof skipped
  sorry

end evaluate_expression_l235_23515


namespace max_distance_circle_to_line_l235_23566

-- Definitions for the circle and line
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2 * x - 2 * y = 0
def line_eq (x y : ℝ) : Prop := x + y + 2 = 0

-- Proof statement
theorem max_distance_circle_to_line 
  (x y : ℝ)
  (h_circ : circle_eq x y)
  (h_line : ∀ (x y : ℝ), line_eq x y → true) :
  ∃ (d : ℝ), d = 3 * Real.sqrt 2 :=
sorry

end max_distance_circle_to_line_l235_23566


namespace range_of_a_l235_23529

theorem range_of_a (a : ℝ) : 
  (∀ x y : ℝ, x < y → (2 * a + 1)^x > (2 * a + 1)^y) → (-1/2 < a ∧ a < 0) :=
by
  sorry

end range_of_a_l235_23529


namespace exists_real_root_iff_l235_23587

theorem exists_real_root_iff {m : ℝ} :
  (∃x : ℝ, 25 - abs (x + 1) - 4 * 5 - abs (x + 1) - m = 0) ↔ (-3 < m ∧ m < 0) :=
by
  sorry

end exists_real_root_iff_l235_23587


namespace john_annual_payment_l235_23555

open Real

-- Definitions extracted from the problem:
def epipen_cost : ℝ := 500
def insurance_coverage : ℝ := 0.75
def epipen_frequency_per_year : ℕ := 2
def john_payment_per_epipen : ℝ := epipen_cost * (1 - insurance_coverage)

-- The statement to be proved:
theorem john_annual_payment :
  john_payment_per_epipen * epipen_frequency_per_year = 250 :=
by
  sorry

end john_annual_payment_l235_23555


namespace gcd_119_34_l235_23535

theorem gcd_119_34 : Nat.gcd 119 34 = 17 := by
  sorry

end gcd_119_34_l235_23535


namespace equation_solution_l235_23502

noncomputable def solve_equation (x : ℝ) : Prop :=
  (1/4) * x^(1/2 * Real.log x / Real.log 2) = 2^(1/4 * (Real.log x / Real.log 2)^2)

theorem equation_solution (x : ℝ) (hx : 0 < x) : solve_equation x → (x = 2^(2*Real.sqrt 2) ∨ x = 2^(-2*Real.sqrt 2)) :=
  by
  intro h
  sorry

end equation_solution_l235_23502


namespace remainder_when_150_divided_by_k_is_2_l235_23578

theorem remainder_when_150_divided_by_k_is_2
  (k : ℕ) (q : ℤ)
  (hk_pos : k > 0)
  (hk_condition : 120 = q * k^2 + 8) :
  150 % k = 2 :=
sorry

end remainder_when_150_divided_by_k_is_2_l235_23578


namespace least_k_for_168_l235_23571

theorem least_k_for_168 (k : ℕ) :
  (k^3 % 168 = 0) ↔ k ≥ 42 :=
sorry

end least_k_for_168_l235_23571


namespace m_div_x_eq_4_div_5_l235_23597

variable (a b : ℝ)
variable (h_pos_a : 0 < a)
variable (h_pos_b : 0 < b)
variable (h_ratio : a / b = 4 / 5)

def x := a * 1.25

def m := b * 0.80

theorem m_div_x_eq_4_div_5 : m / x = 4 / 5 :=
by
  sorry

end m_div_x_eq_4_div_5_l235_23597


namespace radius_of_smaller_molds_l235_23506

noncomputable def volumeOfHemisphere (r : ℝ) : ℝ := (2 / 3) * Real.pi * r^3

theorem radius_of_smaller_molds (r : ℝ) :
  volumeOfHemisphere 2 = 64 * volumeOfHemisphere r → r = 1 / 2 :=
by
  intro h
  sorry

end radius_of_smaller_molds_l235_23506


namespace find_m_value_l235_23519

theorem find_m_value (m : ℝ) (x : ℝ) (y : ℝ) :
  (∀ (x y : ℝ), x + m * y + 3 - 2 * m = 0) →
  (∃ (y : ℝ), x = 0 ∧ y = -1) →
  m = 1 :=
by
  sorry

end find_m_value_l235_23519


namespace quadratic_has_distinct_real_roots_l235_23514

def discriminant (a b c : ℝ) : ℝ := b ^ 2 - 4 * a * c

theorem quadratic_has_distinct_real_roots :
  let a := 5
  let b := -2
  let c := -7
  discriminant a b c > 0 :=
by
  sorry

end quadratic_has_distinct_real_roots_l235_23514


namespace James_uses_150_sheets_of_paper_l235_23532

-- Define the conditions
def number_of_books := 2
def pages_per_book := 600
def pages_per_side := 4
def sides_per_sheet := 2

-- Statement to prove
theorem James_uses_150_sheets_of_paper :
  number_of_books * pages_per_book / (pages_per_side * sides_per_sheet) = 150 :=
by sorry

end James_uses_150_sheets_of_paper_l235_23532


namespace reserve_bird_percentage_l235_23536

theorem reserve_bird_percentage (total_birds hawks paddyfield_warbler_percentage kingfisher_percentage woodpecker_percentage owl_percentage : ℕ) 
  (h1 : total_birds = 5000)
  (h2 : hawks = 30 * total_birds / 100)
  (h3 : paddyfield_warbler_percentage = 40)
  (h4 : kingfisher_percentage = 25)
  (h5 : woodpecker_percentage = 15)
  (h6 : owl_percentage = 15) :
  let non_hawks := total_birds - hawks
  let paddyfield_warblers := paddyfield_warbler_percentage * non_hawks / 100
  let kingfishers := kingfisher_percentage * paddyfield_warblers / 100
  let woodpeckers := woodpecker_percentage * non_hawks / 100
  let owls := owl_percentage * non_hawks / 100
  let specified_non_hawks := paddyfield_warblers + kingfishers + woodpeckers + owls
  let unspecified_non_hawks := non_hawks - specified_non_hawks
  let percentage_unspecified := unspecified_non_hawks * 100 / total_birds
  percentage_unspecified = 14 := by
  sorry

end reserve_bird_percentage_l235_23536


namespace compare_sqrt_l235_23512

noncomputable def a : ℝ := 3 * Real.sqrt 2
noncomputable def b : ℝ := Real.sqrt 15

theorem compare_sqrt : a > b :=
by
  sorry

end compare_sqrt_l235_23512


namespace rectangle_width_l235_23501

-- Definitions and Conditions
variables (L W : ℕ)

-- Condition 1: The perimeter of the rectangle is 16 cm
def perimeter_eq : Prop := 2 * (L + W) = 16

-- Condition 2: The width is 2 cm longer than the length
def width_eq : Prop := W = L + 2

-- Proof Statement: Given the above conditions, the width of the rectangle is 5 cm
theorem rectangle_width (h1 : perimeter_eq L W) (h2 : width_eq L W) : W = 5 := 
by
  sorry

end rectangle_width_l235_23501


namespace correct_sunset_time_l235_23596

-- Definitions corresponding to the conditions
def length_of_daylight : ℕ × ℕ := (10, 30) -- (hours, minutes)
def sunrise_time : ℕ × ℕ := (6, 50) -- (hours, minutes)

-- The reaching goal is to prove the sunset time
def sunset_time (sunrise : ℕ × ℕ) (daylight : ℕ × ℕ) : ℕ × ℕ :=
  let (sh, sm) := sunrise
  let (dh, dm) := daylight
  let total_minutes := sm + dm
  let extra_hour := total_minutes / 60
  let final_minutes := total_minutes % 60
  (sh + dh + extra_hour, final_minutes)

-- The theorem to prove
theorem correct_sunset_time :
  sunset_time sunrise_time length_of_daylight = (17, 20) := sorry

end correct_sunset_time_l235_23596


namespace correct_difference_is_nine_l235_23569

-- Define the conditions
def misunderstood_number : ℕ := 35
def actual_number : ℕ := 53
def incorrect_difference : ℕ := 27

-- Define the two-digit number based on Yoongi's incorrect calculation
def original_number : ℕ := misunderstood_number + incorrect_difference

-- State the theorem
theorem correct_difference_is_nine : (original_number - actual_number) = 9 :=
by
  -- Proof steps go here
  sorry

end correct_difference_is_nine_l235_23569


namespace inequality_system_solution_l235_23589

theorem inequality_system_solution (x : ℝ) :
  (3 * x - 1 > 2 * (x + 1) ∧ (x + 2) / 3 > x - 2) ↔ (3 < x ∧ x < 4) := 
by
  sorry

end inequality_system_solution_l235_23589


namespace variance_of_yield_l235_23513

/-- Given a data set representing annual average yields,
    prove that the variance of this data set is approximately 171. --/
theorem variance_of_yield {yields : List ℝ} 
  (h_yields : yields = [450, 430, 460, 440, 450, 440, 470, 460]) :
  let mean := (yields.sum / yields.length : ℝ)
  let squared_diffs := (yields.map (fun x => (x - mean)^2))
  let variance := (squared_diffs.sum / (yields.length - 1 : ℝ))
  abs (variance - 171) < 1 :=
by
  sorry

end variance_of_yield_l235_23513


namespace arithmetic_sequence_first_term_l235_23550

theorem arithmetic_sequence_first_term (a : ℕ → ℤ) (d : ℤ) 
  (h1 : a 5 = 9) (h2 : 2 * a 3 = a 2 + 6) : a 1 = -3 :=
by
  -- a_5 = a_1 + 4d
  have h3 : a 5 = a 1 + 4 * d := sorry
  
  -- 2a_3 = a_2 + 6, which means 2 * (a_1 + 2d) = (a_1 + d) + 6
  have h4 : 2 * (a 1 + 2 * d) = (a 1 + d) + 6 := sorry
  
  -- solve the system of linear equations to find a_1 = -3
  sorry

end arithmetic_sequence_first_term_l235_23550


namespace negation_of_universal_proposition_l235_23560

theorem negation_of_universal_proposition :
  (∀ x : ℝ, x^2 + 1 > 0) → ¬(∃ x : ℝ, x^2 + 1 ≤ 0) := sorry

end negation_of_universal_proposition_l235_23560


namespace shells_collected_by_savannah_l235_23556

def num_shells_jillian : ℕ := 29
def num_shells_clayton : ℕ := 8
def total_shells_distributed : ℕ := 54

theorem shells_collected_by_savannah (S : ℕ) :
  num_shells_jillian + S + num_shells_clayton = total_shells_distributed → S = 17 :=
by
  sorry

end shells_collected_by_savannah_l235_23556


namespace types_of_problems_l235_23533

def bill_problems : ℕ := 20
def ryan_problems : ℕ := 2 * bill_problems
def frank_problems : ℕ := 3 * ryan_problems
def problems_per_type : ℕ := 30

theorem types_of_problems : (frank_problems / problems_per_type) = 4 := by
  sorry

end types_of_problems_l235_23533


namespace find_t2_l235_23545

variable {P A1 A2 t1 r t2 : ℝ}
def conditions (P A1 A2 t1 r t2 : ℝ) :=
  P = 650 ∧
  A1 = 815 ∧
  A2 = 870 ∧
  t1 = 3 ∧
  A1 = P + (P * r * t1) / 100 ∧
  A2 = P + (P * r * t2) / 100

theorem find_t2
  (P A1 A2 t1 r t2 : ℝ)
  (hc : conditions P A1 A2 t1 r t2) :
  t2 = 4 :=
by
  sorry

end find_t2_l235_23545


namespace clowns_attended_l235_23510

-- Definition of the problem's conditions
def num_children : ℕ := 30
def initial_candies : ℕ := 700
def candies_sold_per_person : ℕ := 20
def remaining_candies : ℕ := 20

-- Main theorem stating that 4 clowns attended the carousel
theorem clowns_attended (num_clowns : ℕ) (candies_left: num_clowns * candies_sold_per_person + num_children * candies_sold_per_person = initial_candies - remaining_candies) : num_clowns = 4 := by
  sorry

end clowns_attended_l235_23510


namespace xy_divides_x2_plus_y2_plus_one_l235_23576

theorem xy_divides_x2_plus_y2_plus_one 
    (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (h : (x * y) ∣ (x^2 + y^2 + 1)) :
  (x^2 + y^2 + 1) / (x * y) = 3 := by
  sorry

end xy_divides_x2_plus_y2_plus_one_l235_23576


namespace min_knights_proof_l235_23522

-- Noncomputable theory as we are dealing with existence proofs
noncomputable def min_knights (n : ℕ) : ℕ :=
  -- Given the table contains 1001 people
  if n = 1001 then 502 else 0

-- The proof problem statement, we need to ensure that minimum number of knights is 502
theorem min_knights_proof : min_knights 1001 = 502 := 
  by
    -- Sketch of proof: Deriving that the minimum number of knights must be 502 based on the problem constraints
    sorry

end min_knights_proof_l235_23522


namespace circle_tangent_to_line_iff_m_eq_zero_l235_23504

theorem circle_tangent_to_line_iff_m_eq_zero (m : ℝ) :
  (∃ x y : ℝ, x^2 + y^2 = m^2 ∧ x - y = m) ↔ m = 0 :=
by 
  sorry

end circle_tangent_to_line_iff_m_eq_zero_l235_23504


namespace combined_salaries_correct_l235_23598

noncomputable def combined_salaries_BCDE (A B C D E : ℕ) : Prop :=
  (A = 8000) →
  ((A + B + C + D + E) / 5 = 8600) →
  (B + C + D + E = 35000)

theorem combined_salaries_correct 
  (A B C D E : ℕ) 
  (hA : A = 8000) 
  (havg : (A + B + C + D + E) / 5 = 8600) : 
  B + C + D + E = 35000 :=
sorry

end combined_salaries_correct_l235_23598


namespace cost_plane_l235_23574

def cost_boat : ℝ := 254.00
def savings_boat : ℝ := 346.00

theorem cost_plane : cost_boat + savings_boat = 600 := 
by 
  sorry

end cost_plane_l235_23574


namespace happy_valley_zoo_animal_arrangement_l235_23582

theorem happy_valley_zoo_animal_arrangement :
  let parrots := 5
  let dogs := 3
  let cats := 4
  let total_animals := parrots + dogs + cats
  (total_animals = 12) →
    (∃ no_of_ways_to_arrange,
      no_of_ways_to_arrange = 2 * (parrots.factorial) * (dogs.factorial) * (cats.factorial) ∧
      no_of_ways_to_arrange = 34560) :=
by
  sorry

end happy_valley_zoo_animal_arrangement_l235_23582


namespace cubic_yard_to_cubic_feet_l235_23580

theorem cubic_yard_to_cubic_feet (h : 1 = 3) : 1 = 27 := 
by
  sorry

end cubic_yard_to_cubic_feet_l235_23580


namespace fermats_little_theorem_l235_23565

theorem fermats_little_theorem 
  (a n : ℕ) 
  (h₁ : 0 < a) 
  (h₂ : 0 < n) 
  (h₃ : Nat.gcd a n = 1) 
  (phi : ℕ := (Nat.totient n)) 
  : n ∣ (a ^ phi - 1) := sorry

end fermats_little_theorem_l235_23565


namespace previous_job_salary_is_correct_l235_23511

-- Define the base salary and commission structure.
def base_salary_new_job : ℝ := 45000
def commission_rate : ℝ := 0.15
def sale_amount : ℝ := 750
def minimum_sales : ℝ := 266.67

-- Define the total salary from the new job with the minimum sales.
def new_job_total_salary : ℝ :=
  base_salary_new_job + (commission_rate * sale_amount * minimum_sales)

-- Define Tom's previous job's salary.
def previous_job_salary : ℝ := 75000

-- Prove that Tom's previous job salary matches the new job total salary with the minimum sales.
theorem previous_job_salary_is_correct :
  (new_job_total_salary = previous_job_salary) :=
by
  -- This is where you would include the proof steps, but it's sufficient to put 'sorry' for now.
  sorry

end previous_job_salary_is_correct_l235_23511


namespace set_intersection_nonempty_l235_23583

theorem set_intersection_nonempty {a : ℕ} (h : ({0, a} ∩ {1, 2} : Set ℕ) ≠ ∅) :
  a = 1 ∨ a = 2 := by
  sorry

end set_intersection_nonempty_l235_23583


namespace race_outcomes_210_l235_23570

-- Define the participants
def participants : List String := ["Abe", "Bobby", "Charles", "Devin", "Edwin", "Fern", "Grace"]

-- The question is to prove the number of different 1st-2nd-3rd place outcomes is 210.
theorem race_outcomes_210 (h : participants.length = 7) : (7 * 6 * 5 = 210) :=
  by sorry

end race_outcomes_210_l235_23570


namespace least_number_to_subtract_l235_23538

theorem least_number_to_subtract (n : ℕ) : (n = 5) → (5000 - n) % 37 = 0 :=
by sorry

end least_number_to_subtract_l235_23538


namespace gift_cost_l235_23508

theorem gift_cost (half_cost : ℝ) (h : half_cost = 14) : 2 * half_cost = 28 :=
by
  sorry

end gift_cost_l235_23508


namespace proof_P_and_Q_l235_23594

/-!
Proposition P: The line y=2x is perpendicular to the line x+2y=0.
Proposition Q: The projections of skew lines in the same plane could be parallel lines.
Prove: P ∧ Q is true.
-/

def proposition_P : Prop := 
  let slope1 := 2
  let slope2 := -1 / 2
  slope1 * slope2 = -1

def proposition_Q : Prop :=
  ∃ (a b : ℝ), (∃ (p q r s : ℝ),
    (a * r + b * p = 0) ∧ (a * s + b * q = 0)) ∧
    (a ≠ 0 ∨ b ≠ 0)

theorem proof_P_and_Q : proposition_P ∧ proposition_Q :=
  by
  -- We need to prove the conjunction of both propositions is true.
  sorry

end proof_P_and_Q_l235_23594


namespace finite_set_elements_at_least_half_m_l235_23541

theorem finite_set_elements_at_least_half_m (m : ℕ) (A : Finset ℤ) (B : ℕ → Finset ℤ) 
  (hm : 2 ≤ m) 
  (hB : ∀ k : ℕ, 1 ≤ k → k ≤ m → (B k).sum id = (m : ℤ) ^ k) : 
  ∃ n : ℕ, (A.card ≥ n) ∧ (n ≥ m / 2) :=
by
  sorry

end finite_set_elements_at_least_half_m_l235_23541


namespace betty_total_oranges_l235_23575

-- Definitions for the given conditions
def boxes : ℝ := 3.0
def oranges_per_box : ℝ := 24

-- Theorem statement to prove the correct answer to the problem
theorem betty_total_oranges : boxes * oranges_per_box = 72 := by
  sorry

end betty_total_oranges_l235_23575


namespace divisible_by_65_l235_23572

theorem divisible_by_65 (n : ℕ) : 65 ∣ (5^n * (2^(2*n) - 3^n) + 2^n - 7^n) :=
sorry

end divisible_by_65_l235_23572


namespace jack_time_to_school_l235_23525

noncomputable def dave_speed : ℚ := 8000 -- cm/min
noncomputable def distance_to_school : ℚ := 160000 -- cm
noncomputable def jack_speed : ℚ := 7650 -- cm/min
noncomputable def jack_start_delay : ℚ := 10 -- min

theorem jack_time_to_school : (distance_to_school / jack_speed) - jack_start_delay = 10.92 :=
by
  sorry

end jack_time_to_school_l235_23525


namespace locus_of_M_equation_of_l_l235_23564
open Real

-- Step 1: Define the given circles
def circle_F1 (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 4
def circle_F2 (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 36

-- Step 2: Define the condition of tangency for the moving circle M
def external_tangent_F1 (cx cy r : ℝ) : Prop := (cx + 2)^2 + cy^2 = (2 + r)^2
def internal_tangent_F2 (cx cy r : ℝ) : Prop := (cx - 2)^2 + cy^2 = (6 - r)^2

-- Step 4: Prove the locus C is an ellipse with the equation excluding x = -4
theorem locus_of_M (cx cy : ℝ) : 
  (∃ r : ℝ, external_tangent_F1 cx cy r ∧ internal_tangent_F2 cx cy r) ↔
  (cx ≠ -4 ∧ (cx^2) / 16 + (cy^2) / 12 = 1) :=
sorry

-- Step 5: Define the conditions for the midpoint of segment AB
def midpoint_Q (x1 y1 x2 y2 : ℝ) : Prop := (x1 + x2) / 2 = 2 ∧ (y1 + y2) / 2 = -1

-- Step 6: Prove the equation of line l
theorem equation_of_l (x1 y1 x2 y2 : ℝ) (h1 : midpoint_Q x1 y1 x2 y2) 
  (h2 : (x1^2 / 16 + y1^2 / 12 = 1) ∧ (x2^2 / 16 + y2^2 / 12 = 1)) :
  3 * (x1 - x2) - 2 * (y1 - y2) = 8 :=
sorry

end locus_of_M_equation_of_l_l235_23564


namespace Mina_stops_in_D_or_A_l235_23531

-- Define the relevant conditions and problem statement
def circumference := 60
def total_distance := 6000
def quarters := ["A", "B", "C", "D"]
def start_position := "S"
def stop_position := if (total_distance % circumference) == 0 then "S" else ""

theorem Mina_stops_in_D_or_A : stop_position = start_position → start_position = "D" ∨ start_position = "A" :=
by
  sorry

end Mina_stops_in_D_or_A_l235_23531


namespace Benny_total_hours_l235_23554

def hours_per_day : ℕ := 7
def days_worked : ℕ := 14

theorem Benny_total_hours : hours_per_day * days_worked = 98 := by
  sorry

end Benny_total_hours_l235_23554


namespace work_together_days_l235_23526

theorem work_together_days (A_rate B_rate : ℝ) (x B_alone_days : ℝ)
  (hA : A_rate = 1 / 5)
  (hB : B_rate = 1 / 15)
  (h_total_work : (A_rate + B_rate) * x + B_rate * B_alone_days = 1) :
  x = 2 :=
by
  -- Set up the equation based on given rates and solving for x.
  sorry

end work_together_days_l235_23526


namespace percentage_voting_for_biff_equals_45_l235_23592

variable (total : ℕ) (votingForMarty : ℕ) (undecidedPercent : ℝ)

theorem percentage_voting_for_biff_equals_45 :
  total = 200 →
  votingForMarty = 94 →
  undecidedPercent = 0.08 →
  let totalDecided := (1 - undecidedPercent) * total
  let votingForBiff := totalDecided - votingForMarty
  let votingForBiffPercent := (votingForBiff / total) * 100
  votingForBiffPercent = 45 :=
by
  intros h1 h2 h3
  let totalDecided := (1 - 0.08 : ℝ) * 200
  let votingForBiff := totalDecided - 94
  let votingForBiffPercent := (votingForBiff / 200) * 100
  sorry

end percentage_voting_for_biff_equals_45_l235_23592


namespace area_of_right_triangle_l235_23599

theorem area_of_right_triangle (A B C : ℝ) (hA : A = 64) (hB : B = 36) (hC : C = 100) : 
  (1 / 2) * (Real.sqrt A) * (Real.sqrt B) = 24 :=
by
  sorry

end area_of_right_triangle_l235_23599


namespace log_negative_l235_23584

open Real

theorem log_negative (a : ℝ) (h : a > 0) : log (-a) = log a := sorry

end log_negative_l235_23584


namespace find_missing_score_l235_23528

theorem find_missing_score
  (scores : List ℕ)
  (h_scores : scores = [73, 83, 86, 73, x])
  (mean : ℚ)
  (h_mean : mean = 79.2)
  (h_length : scores.length = 5)
  : x = 81 := by
  sorry

end find_missing_score_l235_23528


namespace fixed_point_tangent_circle_l235_23567

theorem fixed_point_tangent_circle (x y a b t : ℝ) :
  (x ^ 2 + (y - 2) ^ 2 = 16) ∧ (a * 0 + b * 2 - 12 = 0) ∧ (y = -6) ∧ 
  (t * x - 8 * y = 0) → 
  (0, 0) = (0, 0) :=
by 
  sorry

end fixed_point_tangent_circle_l235_23567


namespace compute_expression_l235_23558

theorem compute_expression : -8 * 4 - (-6 * -3) + (-10 * -5) = 0 := by sorry

end compute_expression_l235_23558


namespace jack_jill_meet_distance_l235_23503

theorem jack_jill_meet_distance : 
  ∀ (total_distance : ℝ) (uphill_distance : ℝ) (headstart : ℝ) 
  (jack_speed_up : ℝ) (jack_speed_down : ℝ)
  (jill_speed_up : ℝ) (jill_speed_down : ℝ), 
  total_distance = 12 → 
  uphill_distance = 6 → 
  headstart = 1 / 4 → 
  jack_speed_up = 12 → 
  jack_speed_down = 18 → 
  jill_speed_up = 14 → 
  jill_speed_down = 20 → 
  ∃ meet_position : ℝ, meet_position = 15.75 :=
by
  sorry

end jack_jill_meet_distance_l235_23503


namespace area_of_tangency_triangle_l235_23561

theorem area_of_tangency_triangle 
  (r1 r2 r3 : ℝ) 
  (h1 : r1 = 2)
  (h2 : r2 = 3)
  (h3 : r3 = 4) 
  (mutually_tangent : ∀ {c1 c2 c3 : ℝ}, c1 + c2 = r1 + r2 ∧ c2 + c3 = r2 + r3 ∧ c1 + c3 = r1 + r3 ) :
  ∃ area : ℝ, area = 3 * (Real.sqrt 6) / 2 :=
by
  sorry

end area_of_tangency_triangle_l235_23561


namespace no_integer_solutions_for_inequality_l235_23500

open Int

theorem no_integer_solutions_for_inequality : ∀ x : ℤ, (x - 4) * (x - 5) < 0 → False :=
by
  sorry

end no_integer_solutions_for_inequality_l235_23500


namespace odd_function_condition_l235_23540

-- Definitions for real numbers and absolute value function
def f (x a b : ℝ) : ℝ := (x + a) * |x + b|

-- Theorem statement
theorem odd_function_condition (a b : ℝ) (h1 : ∀ x : ℝ, f x a b = (x + a) * |x + b|) :
  (∀ x : ℝ, f x a b = -f (-x) a b) ↔ (a = 0 ∨ b = 0) :=
by
  sorry

end odd_function_condition_l235_23540


namespace hyewon_painted_colors_l235_23543

def pentagonal_prism := 
  let num_rectangular_faces := 5 
  let num_pentagonal_faces := 2
  num_rectangular_faces + num_pentagonal_faces

theorem hyewon_painted_colors : pentagonal_prism = 7 := 
by
  sorry

end hyewon_painted_colors_l235_23543


namespace coordinates_of_point_l235_23563

theorem coordinates_of_point (a : ℝ) (h : a - 3 = 0) : (a + 2, a - 3) = (5, 0) :=
by
  sorry

end coordinates_of_point_l235_23563


namespace radius_of_triangle_DEF_l235_23577

noncomputable def radius_of_inscribed_circle (DE DF EF : ℝ) : ℝ :=
  let s := (DE + DF + EF) / 2
  let K := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF))
  K / s

theorem radius_of_triangle_DEF :
  radius_of_inscribed_circle 26 15 17 = 121 / 29 := by
sorry

end radius_of_triangle_DEF_l235_23577


namespace distribution_count_l235_23573

def num_distributions (novels poetry students : ℕ) : ℕ :=
  -- This is where the formula for counting would go, but we'll just define it as sorry for now
  sorry

theorem distribution_count : num_distributions 3 2 4 = 28 :=
by
  sorry

end distribution_count_l235_23573
