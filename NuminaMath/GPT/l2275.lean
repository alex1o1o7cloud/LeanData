import Mathlib

namespace johns_final_push_time_l2275_227511

-- Definitions and initial conditions.
def john_initial_distance_behind_steve : ℝ := 12
def john_speed : ℝ := 4.2
def steve_speed : ℝ := 3.7
def john_final_distance_ahead_of_steve : ℝ := 2

-- The statement we want to prove:
theorem johns_final_push_time : ∃ t : ℝ, john_speed * t = steve_speed * t + john_initial_distance_behind_steve + john_final_distance_ahead_of_steve ∧ t = 28 := 
by 
  -- Adding blank proof body
  sorry

end johns_final_push_time_l2275_227511


namespace complex_number_properties_l2275_227508

open Complex

-- Defining the imaginary unit
def i : ℂ := Complex.I

-- Given conditions in Lean: \( z \) satisfies \( z(2+i) = i^{10} \)
def satisfies_condition (z : ℂ) : Prop :=
  z * (2 + i) = i^10

-- Theorem stating the required proofs
theorem complex_number_properties (z : ℂ) (hc : satisfies_condition z) :
  Complex.abs z = Real.sqrt 5 / 5 ∧ 
  (z.re < 0 ∧ z.im > 0) := by
  -- Placeholders for the proof steps
  sorry

end complex_number_properties_l2275_227508


namespace sin_cos_unique_solution_l2275_227526

theorem sin_cos_unique_solution (α : ℝ) (hα1 : 0 < α) (hα2 : α < (π / 2)) :
  ∃! x : ℝ, (Real.sin α) ^ x + (Real.cos α) ^ x = 1 :=
sorry

end sin_cos_unique_solution_l2275_227526


namespace smallest_fraction_of_land_l2275_227541

noncomputable def smallest_share (n : ℕ) : ℚ :=
  if n = 150 then 1 / (2 * 3^49) else 0

theorem smallest_fraction_of_land :
  smallest_share 150 = 1 / (2 * 3^49) :=
sorry

end smallest_fraction_of_land_l2275_227541


namespace alice_bob_meet_same_point_in_5_turns_l2275_227582

theorem alice_bob_meet_same_point_in_5_turns :
  ∃ k : ℕ, k = 5 ∧ 
  (∀ n, (1 + 7 * n) % 24 = 12 ↔ (n = k)) :=
by
  sorry

end alice_bob_meet_same_point_in_5_turns_l2275_227582


namespace gcd_6Pn_n_minus_2_l2275_227559

-- Auxiliary definition to calculate the nth pentagonal number
def pentagonal (n : ℕ) : ℕ := n ^ 2

-- Statement of the theorem
theorem gcd_6Pn_n_minus_2 (n : ℕ) (hn : 0 < n) : 
  ∃ d, d = Int.gcd (6 * pentagonal n) (n - 2) ∧ d ≤ 24 ∧ (∀ k, Int.gcd (6 * pentagonal k) (k - 2) ≤ 24) :=
sorry

end gcd_6Pn_n_minus_2_l2275_227559


namespace perfect_square_A_plus_B_plus1_l2275_227536

-- Definitions based on conditions
def A (m : ℕ) : ℕ := (10^2*m - 1) / 9
def B (m : ℕ) : ℕ := 4 * (10^m - 1) / 9

-- Proof statement
theorem perfect_square_A_plus_B_plus1 (m : ℕ) : A m + B m + 1 = ((10^m + 2) / 3)^2 :=
by
  sorry

end perfect_square_A_plus_B_plus1_l2275_227536


namespace cone_cannot_have_rectangular_cross_section_l2275_227507

noncomputable def solid := Type

def is_cylinder (s : solid) : Prop := sorry
def is_cone (s : solid) : Prop := sorry
def is_rectangular_prism (s : solid) : Prop := sorry
def is_cube (s : solid) : Prop := sorry

def has_rectangular_cross_section (s : solid) : Prop := sorry

axiom cylinder_has_rectangular_cross_section (s : solid) : is_cylinder s → has_rectangular_cross_section s
axiom rectangular_prism_has_rectangular_cross_section (s : solid) : is_rectangular_prism s → has_rectangular_cross_section s
axiom cube_has_rectangular_cross_section (s : solid) : is_cube s → has_rectangular_cross_section s

theorem cone_cannot_have_rectangular_cross_section (s : solid) : is_cone s → ¬has_rectangular_cross_section s := 
sorry

end cone_cannot_have_rectangular_cross_section_l2275_227507


namespace right_obtuse_triangle_impossible_l2275_227505

def triangle_interior_angles_sum (α β γ : ℝ) : Prop :=
  α + β + γ = 180

def is_right_angle (α : ℝ) : Prop :=
  α = 90

def is_obtuse_angle (α : ℝ) : Prop :=
  α > 90

theorem right_obtuse_triangle_impossible (α β γ : ℝ) (h1 : triangle_interior_angles_sum α β γ) (h2 : is_right_angle α) (h3 : is_obtuse_angle β) : false :=
  sorry

end right_obtuse_triangle_impossible_l2275_227505


namespace election_votes_l2275_227560

variable (V : ℝ)

theorem election_votes (h1 : 0.70 * V - 0.30 * V = 192) : V = 480 :=
by
  sorry

end election_votes_l2275_227560


namespace divisors_end_with_1_l2275_227571

theorem divisors_end_with_1 (n : ℕ) (h : n > 0) :
  ∀ d : ℕ, d ∣ (10^(5^n) - 1) / 9 → d % 10 = 1 :=
sorry

end divisors_end_with_1_l2275_227571


namespace find_points_l2275_227564

noncomputable def f (x y z : ℝ) : ℝ := (x^2 + y^2 + z^2) / (x + y + z)

theorem find_points :
  (∃ (x₀ y₀ z₀ : ℝ), 0 < x₀^2 + y₀^2 + z₀^2 ∧ x₀^2 + y₀^2 + z₀^2 < 1 / 1999 ∧
    1.999 < f x₀ y₀ z₀ ∧ f x₀ y₀ z₀ < 2) :=
  sorry

end find_points_l2275_227564


namespace house_height_proof_l2275_227580

noncomputable def height_of_house (house_shadow tree_height tree_shadow : ℕ) : ℕ :=
  house_shadow * tree_height / tree_shadow

theorem house_height_proof
  (house_shadow_length : ℕ)
  (tree_height : ℕ)
  (tree_shadow_length : ℕ)
  (expected_house_height : ℕ)
  (Hhouse_shadow_length : house_shadow_length = 56)
  (Htree_height : tree_height = 21)
  (Htree_shadow_length : tree_shadow_length = 24)
  (Hexpected_house_height : expected_house_height = 49) :
  height_of_house house_shadow_length tree_height tree_shadow_length = expected_house_height :=
by
  rw [Hhouse_shadow_length, Htree_height, Htree_shadow_length, Hexpected_house_height]
  -- Here we should compute the value and show it is equal to 49
  sorry

end house_height_proof_l2275_227580


namespace max_reflections_l2275_227539

theorem max_reflections (A B D : Point) (n : ℕ) (angle_CDA : ℝ) (incident_angle : ℕ → ℝ)
  (h1 : angle_CDA = 12)
  (h2 : ∀ k : ℕ, k ≤ n → incident_angle k = k * angle_CDA)
  (h3 : incident_angle n = 90) :
  n = 7 := 
sorry

end max_reflections_l2275_227539


namespace cos_of_angle_C_l2275_227550

theorem cos_of_angle_C (A B C : ℝ)
  (h1 : Real.sin (π - A) = 3 / 5)
  (h2 : Real.tan (π + B) = 12 / 5)
  (h_cos_A : Real.cos A = 4 / 5) :
  Real.cos C = 16 / 65 :=
sorry

end cos_of_angle_C_l2275_227550


namespace gcd_of_q_and_r_l2275_227593

theorem gcd_of_q_and_r (p q r : ℕ) (hpq : p > 0) (hqr : q > 0) (hpr : r > 0)
    (gcd_pq : Nat.gcd p q = 240) (gcd_pr : Nat.gcd p r = 540) : Nat.gcd q r = 60 := by
  sorry

end gcd_of_q_and_r_l2275_227593


namespace girls_at_picnic_l2275_227534

variables (g b : ℕ)

-- Conditions
axiom total_students : g + b = 1500
axiom students_at_picnic : (3/4) * g + (2/3) * b = 900

-- Goal: Prove number of girls who attended the picnic
theorem girls_at_picnic (hg : (3/4 : ℚ) * 1200 = 900) : (3/4 : ℚ) * 1200 = 900 :=
by sorry

end girls_at_picnic_l2275_227534


namespace water_added_16_l2275_227531

theorem water_added_16 (W : ℝ) 
  (h1 : ∃ W, 24 * 0.90 = 0.54 * (24 + W)) : 
  W = 16 := 
by {
  sorry
}

end water_added_16_l2275_227531


namespace total_dog_weight_l2275_227568

theorem total_dog_weight (weight_evans_dog weight_ivans_dog : ℕ)
  (h₁ : weight_evans_dog = 63)
  (h₂ : weight_evans_dog = 7 * weight_ivans_dog) :
  weight_evans_dog + weight_ivans_dog = 72 :=
sorry

end total_dog_weight_l2275_227568


namespace min_value_3x_4y_l2275_227504

theorem min_value_3x_4y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) 
  (h : 3 / x + 1 / y = 1) : 
  3 * x + 4 * y ≥ 25 :=
sorry

end min_value_3x_4y_l2275_227504


namespace unit_digit_25_pow_2010_sub_3_pow_2012_l2275_227561

theorem unit_digit_25_pow_2010_sub_3_pow_2012 :
  (25^2010 - 3^2012) % 10 = 4 :=
by 
  sorry

end unit_digit_25_pow_2010_sub_3_pow_2012_l2275_227561


namespace exist_a_sequence_l2275_227554

theorem exist_a_sequence (n : ℕ) (h : n ≥ 2) (x : Fin n → ℝ) (hx : ∀ i, 0 ≤ x i ∧ x i ≤ 1) :
  ∃ (a : Fin (n+1) → ℝ), (a 0 + a n = 0) ∧ (∀ i, |a i| ≤ 1) ∧ (∀ i : Fin n, |a i.succ - a i| = x i) :=
by
  sorry

end exist_a_sequence_l2275_227554


namespace algebraic_expression_value_l2275_227549

theorem algebraic_expression_value (a b : ℝ) (h : a - b = 2) : a^2 - b^2 - 4*a = -4 := 
sorry

end algebraic_expression_value_l2275_227549


namespace number_of_roses_ian_kept_l2275_227584

-- Definitions representing the conditions
def initial_roses : ℕ := 20
def roses_to_mother : ℕ := 6
def roses_to_grandmother : ℕ := 9
def roses_to_sister : ℕ := 4

-- The theorem statement we want to prove
theorem number_of_roses_ian_kept : (initial_roses - (roses_to_mother + roses_to_grandmother + roses_to_sister) = 1) :=
by
  sorry

end number_of_roses_ian_kept_l2275_227584


namespace unique_function_l2275_227523

noncomputable def f : ℝ → ℝ := sorry

theorem unique_function 
  (h_f : ∀ x > 0, ∀ y > 0, f x * f y = 2 * f (x + y * f x)) : ∀ x > 0, f x = 2 :=
by
  sorry

end unique_function_l2275_227523


namespace sin_neg_270_eq_one_l2275_227566

theorem sin_neg_270_eq_one : Real.sin (-(270 : ℝ) * (Real.pi / 180)) = 1 := by
  sorry

end sin_neg_270_eq_one_l2275_227566


namespace faulty_keys_l2275_227503

noncomputable def faulty_digits (typed_sequence : List ℕ) : Set ℕ :=
  { d | d = 7 ∨ d = 9 }

theorem faulty_keys (typed_sequence : List ℕ) (h : typed_sequence.length = 10) :
  (∃ faulty_keys : Set ℕ, ∃ missing_digits : ℕ, missing_digits = 3 ∧ faulty_keys = {7, 9}) :=
sorry

end faulty_keys_l2275_227503


namespace pencils_and_notebooks_cost_l2275_227565

theorem pencils_and_notebooks_cost
    (p n : ℝ)
    (h1 : 8 * p + 10 * n = 5.36)
    (h2 : 12 * (p - 0.05) + 5 * n = 4.05) :
    15 * (p - 0.05) + 12 * n = 7.01 := 
sorry

end pencils_and_notebooks_cost_l2275_227565


namespace valid_rod_count_l2275_227510

open Nat

theorem valid_rod_count :
  ∃ valid_rods : Finset ℕ,
    (∀ d ∈ valid_rods, 6 ≤ d ∧ d < 35 ∧ d ≠ 5 ∧ d ≠ 10 ∧ d ≠ 20) ∧ 
    valid_rods.card = 26 := sorry

end valid_rod_count_l2275_227510


namespace all_statements_false_l2275_227522

theorem all_statements_false (r1 r2 : ℝ) (h1 : r1 ≠ r2) (h2 : r1 + r2 = 5) (h3 : r1 * r2 = 6) :
  ¬(|r1 + r2| > 6) ∧ ¬(3 < |r1 * r2| ∧ |r1 * r2| < 8) ∧ ¬(r1 < 0 ∧ r2 < 0) :=
by
  sorry

end all_statements_false_l2275_227522


namespace customer_ordered_bags_l2275_227590

def bags_per_batch : Nat := 10
def initial_bags : Nat := 20
def days : Nat := 4
def batches_per_day : Nat := 1

theorem customer_ordered_bags : 
  initial_bags + days * batches_per_day * bags_per_batch = 60 :=
by
  sorry

end customer_ordered_bags_l2275_227590


namespace minimize_expression_l2275_227521

theorem minimize_expression (n : ℕ) (h : n > 0) : (n = 10) ↔ (∀ m : ℕ, m > 0 → (n / 2 + 50 / n: ℝ) ≤ (m / 2 + 50 / m: ℝ)) :=
sorry

end minimize_expression_l2275_227521


namespace find_base_number_l2275_227572

theorem find_base_number (y : ℕ) (base : ℕ) (h : 9^y = base ^ 16) (hy : y = 8) : base = 3 :=
by
  -- We skip the proof steps and insert sorry here
  sorry

end find_base_number_l2275_227572


namespace log_sum_eq_two_l2275_227553

theorem log_sum_eq_two (log6_3 log6_4 : ℝ) (H1 : Real.logb 6 3 = log6_3) (H2 : Real.logb 6 4 = log6_4) : 
  log6_3 + log6_4 = 2 := 
by 
  sorry

end log_sum_eq_two_l2275_227553


namespace problem_solution_l2275_227519

noncomputable def ellipse_properties (F1 F2 : ℝ × ℝ) (sum_dists : ℝ) : ℝ × ℝ × ℝ × ℝ := 
  let a := sum_dists / 2 
  let c := (Real.sqrt ((F1.1 - F2.1)^2 + (F1.2 - F2.2)^2)) / 2
  let b := Real.sqrt (a^2 - c^2)
  let h := (F1.1 + F2.1) / 2
  let k := (F1.2 + F2.2) / 2
  (h, k, a, b)

theorem problem_solution :
  let F1 := (0, 1)
  let F2 := (6, 1)
  let sum_dists := 10
  let (h, k, a, b) := ellipse_properties F1 F2 sum_dists
  h + k + a + b = 13 :=
by
  -- assuming the proof here
  sorry

end problem_solution_l2275_227519


namespace sum_of_possible_values_of_d_l2275_227547

def base_digits (n : ℕ) (b : ℕ) : ℕ := 
  if n = 0 then 1 else Nat.log (n + 1) b

theorem sum_of_possible_values_of_d :
  let min_val_7 := 1 * 7^3
  let max_val_7 := 6 * 7^3 + 6 * 7^2 + 6 * 7^1 + 6 * 7^0
  let min_val_10 := 343
  let max_val_10 := 2400
  let d1 := base_digits min_val_10 3
  let d2 := base_digits max_val_10 3
  d1 + d2 = 13 := sorry

end sum_of_possible_values_of_d_l2275_227547


namespace sin_cos_equiv_l2275_227546

theorem sin_cos_equiv (x : ℝ) (h : Real.cos x - 5 * Real.sin x = 2) :
  Real.sin x + 5 * Real.cos x = -1/2 ∨ Real.sin x + 5 * Real.cos x = 17/13 := 
by
  sorry

end sin_cos_equiv_l2275_227546


namespace rectangular_solid_volume_l2275_227517

theorem rectangular_solid_volume 
  (a b c : ℝ) 
  (h1 : a * b = 18) 
  (h2 : b * c = 50) 
  (h3 : a * c = 45) : 
  a * b * c = 150 * Real.sqrt 3 := 
by 
  sorry

end rectangular_solid_volume_l2275_227517


namespace divide_by_3_result_l2275_227545

-- Definitions
def n : ℕ := 4 * 12

theorem divide_by_3_result (h : n / 4 = 12) : n / 3 = 16 :=
by
  sorry

end divide_by_3_result_l2275_227545


namespace distance_between_cyclists_l2275_227524

def cyclist_distance (v1 : ℝ) (v2 : ℝ) (t : ℝ) : ℝ :=
  (v1 + v2) * t

theorem distance_between_cyclists :
  cyclist_distance 10 25 1.4285714285714286 = 50 := by
  sorry

end distance_between_cyclists_l2275_227524


namespace all_odd_digits_n_squared_l2275_227586

/-- Helper function to check if all digits in a number are odd -/
def all_odd_digits (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, d % 2 = 1

/-- Main theorem stating that the only positive integers n such that all the digits of n^2 are odd are 1 and 3 -/
theorem all_odd_digits_n_squared (n : ℕ) :
  (n > 0) → (all_odd_digits (n^2)) → (n = 1 ∨ n = 3) :=
by
  sorry

end all_odd_digits_n_squared_l2275_227586


namespace constant_chromosome_number_l2275_227597

theorem constant_chromosome_number (rabbits : Type) 
  (sex_reproduction : rabbits → Prop)
  (maintain_chromosome_number : Prop)
  (meiosis : Prop)
  (fertilization : Prop) : 
  (meiosis ∧ fertilization) ↔ maintain_chromosome_number :=
sorry

end constant_chromosome_number_l2275_227597


namespace range_of_a_l2275_227551

open Set Real

noncomputable def f (x a : ℝ) := x ^ 2 + 2 * x + a

theorem range_of_a (a : ℝ) :
  (∃ x, 1 ≤ x ∧ x ≤ 2 ∧ f x a ≥ 0) → a ≥ -8 :=
by
  intro h
  sorry

end range_of_a_l2275_227551


namespace ab_leq_1_l2275_227596

theorem ab_leq_1 {a b : ℝ} (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : a + b = 2) : ab ≤ 1 :=
sorry

end ab_leq_1_l2275_227596


namespace find_k_l2275_227543

theorem find_k (k : ℕ) :
  (∑' n : ℕ, (5 + n * k) / 5 ^ n) = 12 → k = 90 :=
by
  sorry

end find_k_l2275_227543


namespace person2_speed_l2275_227579

variables (v_1 : ℕ) (v_2 : ℕ)

def meet_time := 4
def catch_up_time := 16

def meet_equation : Prop := v_1 + v_2 = 22
def catch_up_equation : Prop := v_2 - v_1 = 4

theorem person2_speed :
  meet_equation v_1 v_2 → catch_up_equation v_1 v_2 →
  v_1 = 6 → v_2 = 10 :=
by
  intros h1 h2 h3
  sorry

end person2_speed_l2275_227579


namespace estimate_first_year_students_l2275_227525

noncomputable def number_of_first_year_students (N : ℕ) : Prop :=
  let p1 := (N - 90) / N
  let p2 := (N - 100) / N
  let p_both := 1 - p1 * p2
  p_both = 20 / N → N = 450

theorem estimate_first_year_students : ∃ N : ℕ, number_of_first_year_students N :=
by
  use 450
  -- sorry added to skip the proof part
  sorry

end estimate_first_year_students_l2275_227525


namespace op_evaluation_l2275_227556

-- Define the custom operation ⊕
def op (a b c : ℝ) : ℝ := b^2 - 3 * a * c

-- Statement of the theorem we want to prove
theorem op_evaluation : op 2 3 4 = -15 :=
by 
  -- This is a placeholder for the actual proof,
  -- which in a real scenario would involve computing the operation.
  sorry

end op_evaluation_l2275_227556


namespace grace_is_14_l2275_227513

def GraceAge (G F C E D : ℕ) : Prop :=
  G = F - 6 ∧ F = C + 2 ∧ E = C + 3 ∧ D = E - 4 ∧ D = 17

theorem grace_is_14 (G F C E D : ℕ) (h : GraceAge G F C E D) : G = 14 :=
by sorry

end grace_is_14_l2275_227513


namespace M_lt_N_l2275_227535

variables (a b c : ℝ)

def f (x : ℝ) : ℝ := a * x^2 + b * x + c

def N : ℝ := |a + b + c| + |2 * a - b|
def M : ℝ := |a - b + c| + |2 * a + b|

axiom h1 : f 1 < 0  -- a + b + c < 0
axiom h2 : f (-1) > 0  -- a - b + c > 0
axiom h3 : a > 0
axiom h4 : -b / (2 * a) > 1

theorem M_lt_N : M a b c < N a b c :=
by
  sorry

end M_lt_N_l2275_227535


namespace bounded_fx_range_a_l2275_227538

-- Part (1)
theorem bounded_fx :
  ∃ M > 0, ∀ x ∈ Set.Icc (-(1/2):ℝ) (1/2), abs (x / (x + 1)) ≤ M :=
by
  sorry

-- Part (2)
theorem range_a (a : ℝ) :
  (∀ x ≥ 0, abs (1 + a * (1/2)^x + (1/4)^x) ≤ 3) → -5 ≤ a ∧ a ≤ 1 :=
by
  sorry

end bounded_fx_range_a_l2275_227538


namespace find_r_value_l2275_227555

theorem find_r_value (n : ℕ) (r s : ℕ) (h_s : s = 2^n - 1) (h_r : r = 3^s - s) (h_n : n = 3) : r = 2180 :=
by
  sorry

end find_r_value_l2275_227555


namespace common_volume_of_tetrahedra_l2275_227598

open Real

noncomputable def volume_of_common_part (a b c : ℝ) : ℝ :=
  min (a * sqrt 3 / 12) (min (b * sqrt 3 / 12) (c * sqrt 3 / 12))

theorem common_volume_of_tetrahedra (a b c : ℝ) :
  volume_of_common_part a b c =
  min (a * sqrt 3 / 12) (min (b * sqrt 3 / 12) (c * sqrt 3 / 12)) :=
by sorry

end common_volume_of_tetrahedra_l2275_227598


namespace ratio_noah_to_joe_l2275_227528

def noah_age_after_10_years : ℕ := 22
def years_elapsed : ℕ := 10
def joe_age : ℕ := 6
def noah_age : ℕ := noah_age_after_10_years - years_elapsed

theorem ratio_noah_to_joe : noah_age / joe_age = 2 := by
  -- calculation omitted for brevity
  sorry

end ratio_noah_to_joe_l2275_227528


namespace jane_paid_five_l2275_227567

noncomputable def cost_of_apple : ℝ := 0.75
noncomputable def change_received : ℝ := 4.25
noncomputable def amount_paid : ℝ := cost_of_apple + change_received

theorem jane_paid_five : amount_paid = 5.00 :=
by
  sorry

end jane_paid_five_l2275_227567


namespace unique_m_power_function_increasing_l2275_227500

theorem unique_m_power_function_increasing : 
  ∃! (m : ℝ), (∀ x : ℝ, x > 0 → (m^2 - m - 5) * x^(m-1) > 0) ∧ (m^2 - m - 5 = 1) ∧ (m - 1 > 0) :=
by
  sorry

end unique_m_power_function_increasing_l2275_227500


namespace at_least_two_equal_l2275_227533

-- Define the problem
theorem at_least_two_equal (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h : (x^2 / y) + (y^2 / z) + (z^2 / x) = (x^2 / z) + (y^2 / x) + (z^2 / y)) :
  x = y ∨ y = z ∨ z = x := 
by 
  sorry

end at_least_two_equal_l2275_227533


namespace part1_solution_set_part2_range_a_l2275_227595

noncomputable def inequality1 (a x : ℝ) : Prop :=
|a * x - 2| + |a * x - a| ≥ 2

theorem part1_solution_set : 
  (∀ x : ℝ, inequality1 1 x ↔ x ≥ 2.5 ∨ x ≤ 0.5) := 
sorry

theorem part2_range_a :
  (∀ x : ℝ, inequality1 a x) ↔ a ≥ 4 :=
sorry

end part1_solution_set_part2_range_a_l2275_227595


namespace smallest_b_is_2_plus_sqrt_3_l2275_227591

open Real

noncomputable def smallest_b (a b : ℝ) : ℝ :=
  if (2 < a ∧ a < b ∧ (¬(2 + a > b ∧ 2 + b > a ∧ a + b > 2)) ∧
    (¬(1 / b + 1 / a > 2 ∧ 1 / a + 2 > 1 / b ∧ 2 + 1 / b > 1 / a)))
  then b else 0

theorem smallest_b_is_2_plus_sqrt_3 (a b : ℝ) :
  2 < a ∧ a < b ∧ (¬(2 + a > b ∧ 2 + b > a ∧ a + b > 2)) ∧
    (¬(1 / b + 1 / a > 2 ∧ 1 / a + 2 > 1 / b ∧ 2 + 1 / b > 1 / a)) →
  b = 2 + sqrt 3 := sorry

end smallest_b_is_2_plus_sqrt_3_l2275_227591


namespace probability_red_joker_is_1_over_54_l2275_227592

-- Define the conditions as given in the problem
def total_cards : ℕ := 54
def red_joker_count : ℕ := 1

-- Define the function to calculate the probability
def probability_red_joker_top_card : ℚ := red_joker_count / total_cards

-- Problem: Prove that the probability of drawing the red joker as the top card is 1/54
theorem probability_red_joker_is_1_over_54 :
  probability_red_joker_top_card = 1 / 54 :=
by
  sorry

end probability_red_joker_is_1_over_54_l2275_227592


namespace min_expression_value_l2275_227594

def distinct_elements (s : Set ℤ) : Prop := s = {-8, -6, -4, -1, 1, 3, 5, 14}

theorem min_expression_value :
  ∃ (p q r s t u v w : ℤ),
    distinct_elements {p, q, r, s, t, u, v, w} ∧
    (p + q + r + s) ≥ 5 ∧
    (p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ p ≠ u ∧ p ≠ v ∧ p ≠ w ∧
     q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ q ≠ u ∧ q ≠ v ∧ q ≠ w ∧
     r ≠ s ∧ r ≠ t ∧ r ≠ u ∧ r ≠ v ∧ r ≠ w ∧
     s ≠ t ∧ s ≠ u ∧ s ≠ v ∧ s ≠ w ∧
     t ≠ u ∧ t ≠ v ∧ t ≠ w ∧
     u ≠ v ∧ u ≠ w ∧
     v ≠ w) →
    (p + q + r + s)^2 + (t + u + v + w)^2 = 26 :=
sorry

end min_expression_value_l2275_227594


namespace calc_expression_value_l2275_227583

open Real

theorem calc_expression_value :
  sqrt ((16: ℝ) ^ 12 + (8: ℝ) ^ 15) / ((16: ℝ) ^ 5 + (8: ℝ) ^ 16) = (3 * sqrt 2) / 4 := sorry

end calc_expression_value_l2275_227583


namespace min_value_expression_l2275_227575

theorem min_value_expression (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a * b = 3) :
  ∃ x : ℝ, (x = (a^2 + b^2 + 22) / (a + b)) ∧ (x = 8) :=
by
  sorry

end min_value_expression_l2275_227575


namespace largest_two_digit_divisible_by_6_and_ends_in_4_l2275_227589

-- Define what it means to be a two-digit number
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

-- Define what it means to be divisible by 6
def divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

-- Define what it means to end in 4
def ends_in_4 (n : ℕ) : Prop := n % 10 = 4

-- Final theorem statement
theorem largest_two_digit_divisible_by_6_and_ends_in_4 : 
  ∀ n, is_two_digit n ∧ divisible_by_6 n ∧ ends_in_4 n → n ≤ 84 :=
by
  -- sorry is used here as we are not providing the proof
  sorry

end largest_two_digit_divisible_by_6_and_ends_in_4_l2275_227589


namespace geometric_sequence_sum_l2275_227537

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ)
    (h1 : a 1 = 3)
    (h2 : a 4 = 24)
    (hn : ∀ n, a n = a 1 * q ^ (n - 1)) :
    (a 3 + a 4 + a 5 = 84) :=
by
  -- Proof will go here
  sorry

end geometric_sequence_sum_l2275_227537


namespace range_of_a_l2275_227587

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, a*x^2 - a*x - 2 ≤ 0) → (-8 ≤ a ∧ a ≤ 0) :=
by
  sorry

end range_of_a_l2275_227587


namespace sum_of_ages_is_nineteen_l2275_227569

-- Definitions representing the conditions
def Bella_age : ℕ := 5
def Brother_is_older : ℕ := 9
def Brother_age : ℕ := Bella_age + Brother_is_older
def Sum_of_ages : ℕ := Bella_age + Brother_age

-- Mathematical statement (theorem) to be proved
theorem sum_of_ages_is_nineteen : Sum_of_ages = 19 := by
  sorry

end sum_of_ages_is_nineteen_l2275_227569


namespace isosceles_triangle_base_l2275_227530

noncomputable def base_of_isosceles_triangle
  (height_to_base : ℝ)
  (height_to_side : ℝ)
  (is_isosceles : Bool) : ℝ :=
if is_isosceles then 7.5 else 0

theorem isosceles_triangle_base :
  base_of_isosceles_triangle 5 6 true = 7.5 :=
by
  -- The proof would go here, just placeholder for now
  sorry

end isosceles_triangle_base_l2275_227530


namespace same_grade_percentage_is_correct_l2275_227599

def total_students : ℕ := 40

def grade_distribution : ℕ × ℕ × ℕ × ℕ :=
  (17, 40, 100)

def same_grade_percentage (total_students : ℕ) (same_grade_students : ℕ) : ℚ :=
  (same_grade_students / total_students) * 100

theorem same_grade_percentage_is_correct :
  let same_grade_students := 3 + 5 + 6 + 3
  same_grade_percentage total_students same_grade_students = 42.5 :=
by 
let same_grade_students := 3 + 5 + 6 + 3
show same_grade_percentage total_students same_grade_students = 42.5
sorry

end same_grade_percentage_is_correct_l2275_227599


namespace minimum_number_of_gloves_l2275_227502

theorem minimum_number_of_gloves (participants : ℕ) (gloves_per_participant : ℕ) (total_participants : participants = 63) (each_participant_needs_2_gloves : gloves_per_participant = 2) : 
  participants * gloves_per_participant = 126 :=
by
  rcases participants, gloves_per_participant, total_participants, each_participant_needs_2_gloves
  -- sorry to skip the proof
  sorry

end minimum_number_of_gloves_l2275_227502


namespace william_marbles_l2275_227570

theorem william_marbles :
  let initial_marbles := 10
  let shared_marbles := 3
  (initial_marbles - shared_marbles) = 7 := 
by
  sorry

end william_marbles_l2275_227570


namespace champion_team_exists_unique_champion_wins_all_not_exactly_two_champions_l2275_227563

-- Define the structure and relationship between teams in the tournament
structure Tournament (Team : Type) :=
  (competes : Team → Team → Prop) -- teams play against each other
  (no_ties : ∀ A B : Team, (competes A B ∧ ¬competes B A) ∨ (competes B A ∧ ¬competes A B)) -- no ties
  (superior : Team → Team → Prop) -- superiority relationship
  (superior_def : ∀ A B : Team, superior A B ↔ (competes A B ∧ ¬competes B A) ∨ (∃ C : Team, superior A C ∧ superior C B))

-- The main theorem based on the given questions
theorem champion_team_exists {Team : Type} (tournament : Tournament Team) :
  ∃ champion : Team, (∀ B : Team, champion ≠ B → tournament.superior champion B) :=
  sorry

theorem unique_champion_wins_all {Team : Type} (tournament : Tournament Team)
  (h : ∃! champion : Team, (∀ B : Team, champion ≠ B → tournament.superior champion B)) :
  ∃! champion : Team, (∀ B : Team, champion ≠ B → tournament.superior champion B ∧ tournament.competes champion B ∧ ¬tournament.competes B champion) :=
  sorry

theorem not_exactly_two_champions {Team : Type} (tournament : Tournament Team) :
  ¬∃ A B : Team, A ≠ B ∧ (∀ C : Team, C ≠ A → tournament.superior A C) ∧ (∀ C : Team, C ≠ B → tournament.superior B C) :=
  sorry

end champion_team_exists_unique_champion_wins_all_not_exactly_two_champions_l2275_227563


namespace reflection_matrix_solution_l2275_227562

variable (a b : ℚ)

def matrix_R : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![a, b], ![-(3/4 : ℚ), (4/5 : ℚ)]]

theorem reflection_matrix_solution (h : matrix_R a b ^ 2 = 1) :
    (a, b) = (-4/5, -3/5) := sorry

end reflection_matrix_solution_l2275_227562


namespace number_of_triangles_l2275_227576

theorem number_of_triangles (n : ℕ) (hn : 0 < n) :
  ∃ t, t = (n + 2) ^ 2 - 2 * (⌊ (n : ℝ) / 2 ⌋) / 4 :=
by
  sorry

end number_of_triangles_l2275_227576


namespace set_union_is_correct_l2275_227509

noncomputable def M (a : ℝ) : Set ℝ := {3, 2^a}
noncomputable def N (a b : ℝ) : Set ℝ := {a, b}

variable (a b : ℝ)
variable (h₁ : M a ∩ N a b = {2})
variable (h₂ : ∃ a b, N a b = {1, 2} ∧ M a = {3, 2} ∧ M a ∪ N a b = {1, 2, 3})

theorem set_union_is_correct :
  M 1 ∪ N 1 2 = {1, 2, 3} :=
by
  sorry

end set_union_is_correct_l2275_227509


namespace number_of_valid_pairs_is_34_l2275_227527

noncomputable def countValidPairs : Nat :=
  let primes : List Nat := [2, 3, 5, 7, 11, 13]
  let nonprimes : List Nat := [1, 4, 6, 8, 9, 10, 12, 14, 15]
  let countForN (n : Nat) : Nat :=
    match n with
    | 2 => Nat.choose 8 1
    | 3 => Nat.choose 7 2
    | 5 => Nat.choose 5 4
    | _ => 0
  primes.map countForN |>.sum

theorem number_of_valid_pairs_is_34 : countValidPairs = 34 :=
  sorry

end number_of_valid_pairs_is_34_l2275_227527


namespace train_length_l2275_227518

theorem train_length (speed_km_hr : ℕ) (time_sec : ℕ) (h_speed : speed_km_hr = 72) (h_time : time_sec = 12) : 
  ∃ length_m : ℕ, length_m = 240 := 
by
  sorry

end train_length_l2275_227518


namespace cross_section_prism_in_sphere_l2275_227574

noncomputable def cross_section_area 
  (a R : ℝ) 
  (h1 : a > 0) 
  (h2 : R > 0) 
  (h3 : a < 2 * R) : ℝ :=
  a * Real.sqrt (4 * R^2 - a^2)

theorem cross_section_prism_in_sphere 
  (a R : ℝ) 
  (h1 : a > 0) 
  (h2 : R > 0) 
  (h3 : a < 2 * R) :
  cross_section_area a R h1 h2 h3 = a * Real.sqrt (4 * R^2 - a^2) := 
  by
    sorry

end cross_section_prism_in_sphere_l2275_227574


namespace inequality_holds_for_real_numbers_l2275_227558

theorem inequality_holds_for_real_numbers (x y z : ℝ) : 
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 := 
by
  sorry

end inequality_holds_for_real_numbers_l2275_227558


namespace no_intersection_abs_value_graphs_l2275_227540

theorem no_intersection_abs_value_graphs : 
  ∀ (x : ℝ), ¬ (|3 * x + 6| = -|4 * x - 1|) :=
by
  intro x
  sorry

end no_intersection_abs_value_graphs_l2275_227540


namespace arithmetic_sequence_fifth_term_l2275_227588

theorem arithmetic_sequence_fifth_term (x y : ℚ) 
  (h1 : a₁ = x + y) 
  (h2 : a₂ = x - y) 
  (h3 : a₃ = x * y) 
  (h4 : a₄ = x / y) 
  (h5 : a₂ - a₁ = -2 * y) 
  (h6 : a₃ - a₂ = -2 * y) 
  (h7 : a₄ - a₃ = -2 * y) 
  (hx : x = -9 / 8)
  (hy : y = -3 / 5) : 
  a₅ = 123 / 40 :=
by
  sorry

end arithmetic_sequence_fifth_term_l2275_227588


namespace calculate_expression_l2275_227529

noncomputable def f (x : ℝ) : ℝ :=
  (x^3 + 5 * x^2 + 6 * x) / (x^3 - x^2 - 2 * x)

def num_holes (f : ℝ → ℝ) : ℕ := 1 -- hole at x = -2
def num_vertical_asymptotes (f : ℝ → ℝ) : ℕ := 2 -- vertical asymptotes at x = 0 and x = 1
def num_horizontal_asymptotes (f : ℝ → ℝ) : ℕ := 0 -- no horizontal asymptote
def num_oblique_asymptotes (f : ℝ → ℝ) : ℕ := 1 -- oblique asymptote at y = x + 4

theorem calculate_expression : num_holes f + 2 * num_vertical_asymptotes f + 3 * num_horizontal_asymptotes f + 4 * num_oblique_asymptotes f = 9 :=
by
  -- Provide the proof here
  sorry

end calculate_expression_l2275_227529


namespace sum_of_coeffs_l2275_227514

theorem sum_of_coeffs (A B C D : ℤ) (h₁ : A = 1) (h₂ : B = -1) (h₃ : C = -12) (h₄ : D = 3) :
  A + B + C + D = -9 := 
by
  rw [h₁, h₂, h₃, h₄]
  norm_num

end sum_of_coeffs_l2275_227514


namespace num_sets_satisfying_union_l2275_227577

theorem num_sets_satisfying_union : 
  ∃! (A : Set ℕ), ({1, 3} ∪ A = {1, 3, 5}) :=
by
  sorry

end num_sets_satisfying_union_l2275_227577


namespace total_pencils_given_out_l2275_227557

theorem total_pencils_given_out (n p : ℕ) (h1 : n = 10) (h2 : p = 5) : n * p = 50 :=
by
  sorry

end total_pencils_given_out_l2275_227557


namespace correlation_coefficient_correct_option_l2275_227501

variable (r : ℝ)

-- Definitions of Conditions
def positive_correlation : Prop := r > 0 → ∀ x y : ℝ, x * y > 0
def range_r : Prop := -1 < r ∧ r < 1
def correlation_strength : Prop := |r| < 1 → (∀ ε : ℝ, 0 < ε → ∃ δ : ℝ, 0 < δ ∧ δ < ε ∧ |r| < δ)

-- Theorem statement
theorem correlation_coefficient_correct_option :
  (positive_correlation r) ∧
  (range_r r) ∧
  (correlation_strength r) →
  (r ≠ 0 → |r| < 1) :=
by
  sorry

end correlation_coefficient_correct_option_l2275_227501


namespace orchids_cut_l2275_227532

-- defining the initial conditions
def initial_orchids : ℕ := 3
def final_orchids : ℕ := 7

-- the question: prove the number of orchids cut
theorem orchids_cut : final_orchids - initial_orchids = 4 := by
  sorry

end orchids_cut_l2275_227532


namespace expand_and_simplify_l2275_227516

theorem expand_and_simplify :
  ∀ x : ℝ, (x^3 - 3*x + 3)*(x^2 + 3*x + 3) = x^5 + 3*x^4 - 6*x^2 + 9 := by sorry

end expand_and_simplify_l2275_227516


namespace sequence_a_5_l2275_227573

theorem sequence_a_5 (S : ℕ → ℝ) (a : ℕ → ℝ) (h1 : ∀ n : ℕ, n > 0 → S n = 2 * a n - 3) (h2 : ∀ n : ℕ, n > 0 → a n = S n - S (n - 1)) :
  a 5 = 48 := by
  -- The proof and implementations are omitted
  sorry

end sequence_a_5_l2275_227573


namespace base_extension_1_kilometer_l2275_227544

-- Definition of the original triangle with hypotenuse length and inclination angle
def original_triangle (hypotenuse : ℝ) (angle : ℝ) : Prop :=
  hypotenuse = 1 ∧ angle = 20

-- Definition of the extension required for the new inclination angle
def extension_required (new_angle : ℝ) (extension : ℝ) : Prop :=
  new_angle = 10 ∧ extension = 1

-- The proof problem statement
theorem base_extension_1_kilometer :
  ∀ (hypotenuse : ℝ) (original_angle : ℝ) (new_angle : ℝ),
    original_triangle hypotenuse original_angle →
    new_angle = 10 →
    ∃ extension : ℝ, extension_required new_angle extension :=
by
  -- Sorry is a placeholder for the actual proof
  sorry

end base_extension_1_kilometer_l2275_227544


namespace N_intersect_M_complement_l2275_227548

-- Definitions based on given conditions
def U : Set ℝ := Set.univ
def M : Set ℝ := { x | -2 ≤ x ∧ x ≤ 3 }
def N : Set ℝ := { x | -1 ≤ x ∧ x ≤ 4 }
def M_complement : Set ℝ := { x | x < -2 ∨ x > 3 }  -- complement of M in ℝ

-- Lean statement for the proof problem
theorem N_intersect_M_complement :
  N ∩ M_complement = { x | 3 < x ∧ x ≤ 4 } :=
sorry

end N_intersect_M_complement_l2275_227548


namespace ratio_H_over_G_l2275_227512

theorem ratio_H_over_G (G H : ℤ)
  (h : ∀ x : ℝ, x ≠ -5 → x ≠ 0 → x ≠ 4 →
    (G : ℝ)/(x + 5) + (H : ℝ)/(x^2 - 4*x) = (x^2 - 2*x + 10) / (x^3 + x^2 - 20*x)) :
  H / G = 2 :=
  sorry

end ratio_H_over_G_l2275_227512


namespace clara_total_points_l2275_227581

-- Define the constants
def percentage_three_point_shots : ℝ := 0.25
def points_per_successful_three_point_shot : ℝ := 3
def percentage_two_point_shots : ℝ := 0.40
def points_per_successful_two_point_shot : ℝ := 2
def total_attempts : ℕ := 40

-- Define the function to calculate the total score
def total_score (x y : ℕ) : ℝ :=
  (percentage_three_point_shots * points_per_successful_three_point_shot) * x +
  (percentage_two_point_shots * points_per_successful_two_point_shot) * y

-- The proof statement
theorem clara_total_points (x y : ℕ) (h : x + y = total_attempts) : 
  total_score x y = 32 :=
by
  -- This is a placeholder for the actual proof
  sorry

end clara_total_points_l2275_227581


namespace coordinates_of_point_M_l2275_227552

theorem coordinates_of_point_M 
  (M : ℝ × ℝ) 
  (dist_x_axis : abs M.2 = 5) 
  (dist_y_axis : abs M.1 = 4) 
  (second_quadrant : M.1 < 0 ∧ M.2 > 0) : 
  M = (-4, 5) := 
sorry

end coordinates_of_point_M_l2275_227552


namespace holloway_soccer_team_l2275_227542

theorem holloway_soccer_team (P M : Finset ℕ) (hP_union_M : (P ∪ M).card = 20) 
(hP : P.card = 12) (h_int : (P ∩ M).card = 6) : M.card = 14 := 
by
  sorry

end holloway_soccer_team_l2275_227542


namespace symbols_in_P_l2275_227578
-- Importing the necessary library

-- Define the context P and the operations
def context_P : Type := sorry

def mul_op (P : context_P) : String := "*"
def div_op (P : context_P) : String := "/"
def exp_op (P : context_P) : String := "∧"
def sqrt_op (P : context_P) : String := "SQR"
def abs_op (P : context_P) : String := "ABS"

-- Define what each symbol represents in the context of P
theorem symbols_in_P (P : context_P) :
  (mul_op P = "*") ∧
  (div_op P = "/") ∧
  (exp_op P = "∧") ∧
  (sqrt_op P = "SQR") ∧
  (abs_op P = "ABS") := 
sorry

end symbols_in_P_l2275_227578


namespace joan_total_spent_l2275_227520

theorem joan_total_spent (cost_basketball cost_racing total_spent : ℝ) 
  (h1 : cost_basketball = 5.20) 
  (h2 : cost_racing = 4.23) 
  (h3 : total_spent = cost_basketball + cost_racing) : 
  total_spent = 9.43 := 
by 
  sorry

end joan_total_spent_l2275_227520


namespace brock_buys_7_cookies_l2275_227506

variable (cookies_total : ℕ)
variable (sold_to_stone : ℕ)
variable (left_after_sale : ℕ)
variable (cookies_brock_buys : ℕ)
variable (cookies_katy_buys : ℕ)

theorem brock_buys_7_cookies
  (h1 : cookies_total = 5 * 12)
  (h2 : sold_to_stone = 2 * 12)
  (h3 : left_after_sale = 15)
  (h4 : cookies_total - sold_to_stone - (cookies_brock_buys + cookies_katy_buys) = left_after_sale)
  (h5 : cookies_katy_buys = 2 * cookies_brock_buys) :
  cookies_brock_buys = 7 :=
by
  -- Proof is skipped
  sorry

end brock_buys_7_cookies_l2275_227506


namespace pentagonal_tiles_count_l2275_227585

theorem pentagonal_tiles_count (t s p : ℕ) 
  (h1 : t + s + p = 30) 
  (h2 : 3 * t + 4 * s + 5 * p = 120) : 
  p = 10 := by
  sorry

end pentagonal_tiles_count_l2275_227585


namespace num_solutions_x_squared_minus_y_squared_eq_2001_l2275_227515

theorem num_solutions_x_squared_minus_y_squared_eq_2001 
  (x y : ℕ) (hx : x > 0) (hy : y > 0) :
  x^2 - y^2 = 2001 ↔ (x, y) = (1001, 1000) ∨ (x, y) = (335, 332) := sorry

end num_solutions_x_squared_minus_y_squared_eq_2001_l2275_227515
