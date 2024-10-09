import Mathlib

namespace toby_sharing_proof_l1458_145878

theorem toby_sharing_proof (initial_amt amount_left num_brothers : ℕ) 
(h_init : initial_amt = 343)
(h_left : amount_left = 245)
(h_bros : num_brothers = 2) : 
(initial_amt - amount_left) / (initial_amt * num_brothers) = 1 / 7 := 
sorry

end toby_sharing_proof_l1458_145878


namespace desired_ellipse_properties_l1458_145857

def is_ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  (y^2)/(a^2) + (x^2)/(b^2) = 1

def ellipse_has_foci (a b : ℝ) (c : ℝ) : Prop :=
  c^2 = a^2 - b^2

def desired_ellipse_passes_through_point (a b : ℝ) (P : ℝ × ℝ) : Prop :=
  is_ellipse a b P.1 P.2

def foci_of_ellipse (a b : ℝ) (c : ℝ) : Prop :=
  ellipse_has_foci a b c

axiom given_ellipse_foci : foci_of_ellipse 3 2 (Real.sqrt 5)

theorem desired_ellipse_properties :
  desired_ellipse_passes_through_point 4 (Real.sqrt 11) (0, 4) ∧
  foci_of_ellipse 4 (Real.sqrt 11) (Real.sqrt 5) :=
by
  sorry

end desired_ellipse_properties_l1458_145857


namespace part1_part2_l1458_145839

variables {a b c : ℝ}

-- Condition 1: a, b, and c are positive numbers
variables (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
-- Condition 2: a² + b² + 4c² = 3
variables (h4 : a^2 + b^2 + 4*c^2 = 3)

-- Part 1: Prove a + b + 2c ≤ 3
theorem part1 : a + b + 2*c ≤ 3 :=
by
  sorry

-- Condition for Part 2: b = 2c
variables (h5 : b = 2 * c)

-- Part 2: Prove 1/a + 1/c ≥ 3
theorem part2 : 1/a + 1/c ≥ 3 :=
by
  sorry

end part1_part2_l1458_145839


namespace correct_operation_l1458_145836

theorem correct_operation (a : ℝ) :
  (a^2)^3 = a^6 :=
by
  sorry

end correct_operation_l1458_145836


namespace find_prime_p_l1458_145867

theorem find_prime_p (p x y : ℕ) (hp : Nat.Prime p) (hx : x > 0) (hy : y > 0) :
  (p + 49 = 2 * x^2) ∧ (p^2 + 49 = 2 * y^2) ↔ p = 23 :=
by
  sorry

end find_prime_p_l1458_145867


namespace product_of_p_r_s_l1458_145803

theorem product_of_p_r_s :
  ∃ p r s : ℕ, 3^p + 3^5 = 252 ∧ 2^r + 58 = 122 ∧ 5^3 * 6^s = 117000 ∧ p * r * s = 36 :=
by
  sorry

end product_of_p_r_s_l1458_145803


namespace chips_per_cookie_l1458_145896

theorem chips_per_cookie (total_cookies : ℕ) (uneaten_chips : ℕ) (uneaten_cookies : ℕ) (h1 : total_cookies = 4 * 12) (h2 : uneaten_cookies = total_cookies / 2) (h3 : uneaten_chips = 168) : 
  uneaten_chips / uneaten_cookies = 7 :=
by sorry

end chips_per_cookie_l1458_145896


namespace min_value_expression_l1458_145844

theorem min_value_expression (a b m n : ℝ) 
    (h_a_pos : 0 < a) (h_b_pos : 0 < b) 
    (h_m_pos : 0 < m) (h_n_pos : 0 < n) 
    (h_sum_one : a + b = 1) 
    (h_prod_two : m * n = 2) :
    (a * m + b * n) * (b * m + a * n) = 2 :=
sorry

end min_value_expression_l1458_145844


namespace smaller_of_two_integers_l1458_145847

noncomputable def smaller_integer (m n : ℕ) : ℕ :=
if m < n then m else n

theorem smaller_of_two_integers :
  ∀ (m n : ℕ),
  100 ≤ m ∧ m < 1000 ∧ 100 ≤ n ∧ n < 1000 ∧
  (m + n) / 2 = m + n / 200 →
  smaller_integer m n = 891 :=
by
  intros m n h
  -- Assuming m, n are positive three-digit integers and satisfy the condition
  have h1 := h.1
  have h2 := h.2.1
  have h3 := h.2.2.1
  have h4 := h.2.2.2.1
  have h5 := h.2.2.2.2
  sorry

end smaller_of_two_integers_l1458_145847


namespace painters_time_l1458_145812

-- Define the initial conditions
def n1 : ℕ := 3
def d1 : ℕ := 2
def W := n1 * d1
def n2 : ℕ := 2
def d2 := W / n2
def d_r := (3 * d2) / 4

-- Theorem statement
theorem painters_time (h : d_r = 9 / 4) : d_r = 9 / 4 := by
  sorry

end painters_time_l1458_145812


namespace trigonometric_identity_l1458_145862

theorem trigonometric_identity :
  (1 - Real.sin (Real.pi / 6)) * (1 - Real.sin (5 * Real.pi / 6)) = 1 / 4 :=
by
  have h1 : Real.sin (5 * Real.pi / 6) = Real.sin (Real.pi - Real.pi / 6) := by sorry
  have h2 : Real.sin (Real.pi / 6) = 1 / 2 := by sorry
  sorry

end trigonometric_identity_l1458_145862


namespace correct_student_mark_l1458_145873

theorem correct_student_mark
  (avg_wrong : ℕ) (num_students : ℕ) (wrong_mark : ℕ) (avg_correct : ℕ)
  (h1 : num_students = 10) (h2 : avg_wrong = 100) (h3 : wrong_mark = 90) (h4 : avg_correct = 92) :
  ∃ (x : ℕ), x = 10 :=
by
  sorry

end correct_student_mark_l1458_145873


namespace lola_pop_tarts_baked_l1458_145817

theorem lola_pop_tarts_baked :
  ∃ P : ℕ, (13 + P + 8) + (16 + 12 + 14) = 73 ∧ P = 10 := by
  sorry

end lola_pop_tarts_baked_l1458_145817


namespace smallest_stamps_l1458_145861

theorem smallest_stamps : ∃ S, 1 < S ∧ (S % 9 = 1) ∧ (S % 10 = 1) ∧ (S % 11 = 1) ∧ S = 991 :=
by
  sorry

end smallest_stamps_l1458_145861


namespace recyclable_cans_and_bottles_collected_l1458_145890

-- Define the conditions in Lean
def people_at_picnic : ℕ := 90
def soda_cans : ℕ := 50
def plastic_bottles_sparkling_water : ℕ := 50
def glass_bottles_juice : ℕ := 50
def guests_drank_soda : ℕ := people_at_picnic / 2
def guests_drank_sparkling_water : ℕ := people_at_picnic / 3
def juice_consumed : ℕ := (glass_bottles_juice * 4) / 5

-- The theorem statement
theorem recyclable_cans_and_bottles_collected :
  (soda_cans + guests_drank_sparkling_water + juice_consumed) = 120 :=
by
  sorry

end recyclable_cans_and_bottles_collected_l1458_145890


namespace find_b_find_area_of_ABC_l1458_145840

variable {a b c : ℝ}
variable {B : ℝ}

-- Given Conditions
def given_conditions (a b c B : ℝ) := a = 4 ∧ c = 3 ∧ B = Real.arccos (1 / 8)

-- Proving b = sqrt(22)
theorem find_b (h : given_conditions a b c B) : b = Real.sqrt (a^2 + c^2 - 2 * a * c * Real.cos B) :=
by
  sorry

-- Proving the area of triangle ABC
theorem find_area_of_ABC (h : given_conditions a b c B) 
  (sinB : Real.sin B = 3 * Real.sqrt 7 / 8) : 
  (1 / 2) * a * c * Real.sin B = 9 * Real.sqrt 7 / 4 :=
by
  sorry

end find_b_find_area_of_ABC_l1458_145840


namespace sum_first_five_terms_arith_seq_l1458_145879

theorem sum_first_five_terms_arith_seq (a : ℕ → ℤ)
  (h4 : a 4 = 3) (h5 : a 5 = 7) (h6 : a 6 = 11) :
  a 1 + a 2 + a 3 + a 4 + a 5 = -5 :=
by
  sorry

end sum_first_five_terms_arith_seq_l1458_145879


namespace simplify_trig_expr_l1458_145822

noncomputable def sin15 := Real.sin (Real.pi / 12)
noncomputable def sin30 := Real.sin (Real.pi / 6)
noncomputable def sin45 := Real.sin (Real.pi / 4)
noncomputable def sin60 := Real.sin (Real.pi / 3)
noncomputable def sin75 := Real.sin (5 * Real.pi / 12)
noncomputable def cos10 := Real.cos (Real.pi / 18)
noncomputable def cos20 := Real.cos (Real.pi / 9)
noncomputable def cos30 := Real.cos (Real.pi / 6)

theorem simplify_trig_expr :
  (sin15 + sin30 + sin45 + sin60 + sin75) / (cos10 * cos20 * cos30) = 5.128 :=
sorry

end simplify_trig_expr_l1458_145822


namespace intersection_points_l1458_145838

-- Definition of curve C by the polar equation
def curve_C (ρ : ℝ) (θ : ℝ) : Prop := ρ = 2 * Real.cos θ

-- Definition of line l by the polar equation
def line_l (ρ : ℝ) (θ : ℝ) (m : ℝ) : Prop := ρ * Real.sin (θ + Real.pi / 6) = m

-- Proof statement that line l intersects curve C exactly once for specific values of m
theorem intersection_points (m : ℝ) : 
  (∀ ρ θ, curve_C ρ θ → line_l ρ θ m → ρ = 0 ∧ θ = 0) ↔ (m = -1/2 ∨ m = 3/2) :=
by
  sorry

end intersection_points_l1458_145838


namespace toms_nickels_l1458_145899

variables (q n : ℕ)

theorem toms_nickels (h1 : q + n = 12) (h2 : 25 * q + 5 * n = 220) : n = 4 :=
by {
  sorry
}

end toms_nickels_l1458_145899


namespace final_score_l1458_145828

theorem final_score (questions_first_half questions_second_half : Nat)
  (points_correct points_incorrect : Int)
  (correct_first_half incorrect_first_half correct_second_half incorrect_second_half : Nat) :
  questions_first_half = 10 →
  questions_second_half = 15 →
  points_correct = 3 →
  points_incorrect = -1 →
  correct_first_half = 6 →
  incorrect_first_half = 4 →
  correct_second_half = 10 →
  incorrect_second_half = 5 →
  (points_correct * correct_first_half + points_incorrect * incorrect_first_half 
   + points_correct * correct_second_half + points_incorrect * incorrect_second_half) = 39 := 
by
  intros
  sorry

end final_score_l1458_145828


namespace fourth_guard_ran_150_meters_l1458_145872

def rectangle_width : ℕ := 200
def rectangle_length : ℕ := 300
def total_perimeter : ℕ := 2 * (rectangle_width + rectangle_length)
def three_guards_total_distance : ℕ := 850

def fourth_guard_distance : ℕ := total_perimeter - three_guards_total_distance

theorem fourth_guard_ran_150_meters :
  fourth_guard_distance = 150 :=
by
  -- calculation skipped here
  -- proving fourth_guard_distance as derived being 150 meters
  sorry

end fourth_guard_ran_150_meters_l1458_145872


namespace find_radii_of_circles_l1458_145898

theorem find_radii_of_circles (d : ℝ) (ext_tangent : ℝ) (int_tangent : ℝ)
  (hd : d = 65) (hext : ext_tangent = 63) (hint : int_tangent = 25) :
  ∃ (R r : ℝ), R = 38 ∧ r = 22 :=
by 
  sorry

end find_radii_of_circles_l1458_145898


namespace bug_total_distance_l1458_145883

theorem bug_total_distance :
  let pos1 := 3
  let pos2 := -5
  let pos3 := 8
  let final_pos := 0
  let distance1 := |pos1 - pos2|
  let distance2 := |pos2 - pos3|
  let distance3 := |pos3 - final_pos|
  let total_distance := distance1 + distance2 + distance3
  total_distance = 29 := by
    sorry

end bug_total_distance_l1458_145883


namespace notAPrpos_l1458_145863

def isProposition (s : String) : Prop :=
  s = "6 > 4" ∨ s = "If f(x) is a sine function, then f(x) is a periodic function." ∨ s = "1 ∈ {1, 2, 3}"

theorem notAPrpos (s : String) : ¬isProposition "Is a linear function an increasing function?" :=
by
  sorry

end notAPrpos_l1458_145863


namespace no_adjacent_same_color_probability_zero_l1458_145888

-- Define the number of each color bead
def num_red_beads : ℕ := 5
def num_white_beads : ℕ := 3
def num_blue_beads : ℕ := 2

-- Define the total number of beads
def total_beads : ℕ := num_red_beads + num_white_beads + num_blue_beads

-- Calculate the probability that no two neighboring beads are the same color
noncomputable def probability_no_adjacent_same_color : ℚ :=
  if (num_red_beads > num_white_beads + num_blue_beads + 1) then 0 else sorry

theorem no_adjacent_same_color_probability_zero :
  probability_no_adjacent_same_color = 0 :=
by {
  sorry
}

end no_adjacent_same_color_probability_zero_l1458_145888


namespace smallest_solution_of_abs_eq_l1458_145874

theorem smallest_solution_of_abs_eq (x : ℝ) : 
  (x * |x| = 3 * x + 2 → x ≥ 0 → x = (3 + Real.sqrt 17) / 2) ∧
  (x * |x| = 3 * x + 2 → x < 0 → x = -2) ∧
  (x * |x| = 3 * x + 2 → x = -2 → x = -2) :=
by
  sorry

end smallest_solution_of_abs_eq_l1458_145874


namespace problem_ratio_l1458_145824

-- Define the conditions
variables 
  (R : ℕ) 
  (Bill_problems : ℕ := 20) 
  (Frank_problems_per_type : ℕ := 30)
  (types : ℕ := 4)

-- State the problem to prove
theorem problem_ratio (h1 : 3 * R = Frank_problems_per_type * types) :
  R / Bill_problems = 2 :=
by
  -- placeholder for proof
  sorry

end problem_ratio_l1458_145824


namespace max_true_statements_l1458_145860

theorem max_true_statements (c d : ℝ) : 
  (∃ n, 1 ≤ n ∧ n ≤ 5 ∧ 
    (n = (if (1/c > 1/d) then 1 else 0) +
          (if (c^2 < d^2) then 1 else 0) +
          (if (c > d) then 1 else 0) +
          (if (c > 0) then 1 else 0) +
          (if (d > 0) then 1 else 0))) → 
  n ≤ 3 := 
sorry

end max_true_statements_l1458_145860


namespace power_of_10_digits_l1458_145880

theorem power_of_10_digits (n : ℕ) (hn : n > 1) :
  (∃ k : ℕ, (2^(n-1) < 10^k ∧ 10^k < 2^n) ∨ (5^(n-1) < 10^k ∧ 10^k < 5^n)) ∧ ¬((∃ k : ℕ, 2^(n-1) < 10^k ∧ 10^k < 2^n) ∧ (∃ k : ℕ, 5^(n-1) < 10^k ∧ 10^k < 5^n)) :=
sorry

end power_of_10_digits_l1458_145880


namespace minimum_manhattan_distance_l1458_145807

open Real

def ellipse (P : ℝ × ℝ) : Prop := P.1^2 / 2 + P.2^2 = 1

def line (Q : ℝ × ℝ) : Prop := 3 * Q.1 + 4 * Q.2 = 12

def manhattan_distance (P Q : ℝ × ℝ) : ℝ :=
  |P.1 - Q.1| + |P.2 - Q.2|

theorem minimum_manhattan_distance :
  ∃ P Q, ellipse P ∧ line Q ∧
    ∀ P' Q', ellipse P' → line Q' → manhattan_distance P Q ≤ manhattan_distance P' Q' :=
  sorry

end minimum_manhattan_distance_l1458_145807


namespace total_puff_pastries_l1458_145823

theorem total_puff_pastries (batches trays puff_pastry volunteers : ℕ) 
  (h_batches : batches = 1) 
  (h_trays : trays = 8) 
  (h_puff_pastry : puff_pastry = 25) 
  (h_volunteers : volunteers = 1000) : 
  (volunteers * trays * puff_pastry) = 200000 := 
by 
  have h_total_trays : volunteers * trays = 1000 * 8 := by sorry
  have h_total_puff_pastries_per_volunteer : trays * puff_pastry = 8 * 25 := by sorry
  have h_total_puff_pastries : volunteers * trays * puff_pastry = 1000 * 8 * 25 := by sorry
  sorry

end total_puff_pastries_l1458_145823


namespace diophantine_solution_exists_if_prime_divisor_l1458_145875

theorem diophantine_solution_exists_if_prime_divisor (b : ℕ) (hb : 0 < b) (gcd_b_6 : Nat.gcd b 6 = 1) :
  (∃ x y : ℕ, 0 < x ∧ 0 < y ∧ (1 / (x : ℚ) + 1 / (y : ℚ) = 3 / (b : ℚ))) ↔ 
  ∃ p : ℕ, Nat.Prime p ∧ (∃ k : ℕ, p = 6 * k - 1) ∧ p ∣ b := 
by 
  sorry

end diophantine_solution_exists_if_prime_divisor_l1458_145875


namespace geometric_sequence_sum_l1458_145815

theorem geometric_sequence_sum :
  ∃ (a : ℕ → ℚ),
  (∀ n, 3 * a (n + 1) + a n = 0) ∧
  a 2 = -2/3 ∧
  (a 0 + a 1 + a 2 + a 3 + a 4) = 122/81 :=
sorry

end geometric_sequence_sum_l1458_145815


namespace max_days_for_process_C_l1458_145845

/- 
  A project consists of four processes: A, B, C, and D, which require 2, 5, x, and 4 days to complete, respectively.
  The following conditions are given:
  - A and B can start at the same time.
  - C can start after A is completed.
  - D can start after both B and C are completed.
  - The total duration of the project is 9 days.
  We need to prove that the maximum number of days required to complete process C is 3.
-/
theorem max_days_for_process_C
  (A B C D : ℕ)
  (hA : A = 2)
  (hB : B = 5)
  (hD : D = 4)
  (total_duration : ℕ)
  (h_total : total_duration = 9)
  (h_condition1 : A + C + D = total_duration) : 
  C = 3 :=
by
  rw [hA, hD, h_total] at h_condition1
  linarith

#check max_days_for_process_C

end max_days_for_process_C_l1458_145845


namespace prob_D_correct_l1458_145816

def prob_A : ℚ := 1 / 4
def prob_B : ℚ := 1 / 3
def prob_C : ℚ := 1 / 6
def total_prob (prob_D : ℚ) : Prop := prob_A + prob_B + prob_C + prob_D = 1

theorem prob_D_correct : ∃ (prob_D : ℚ), total_prob prob_D ∧ prob_D = 1 / 4 :=
by
  -- Proof omitted
  sorry

end prob_D_correct_l1458_145816


namespace average_speed_of_car_l1458_145825

/-- The average speed of a car over four hours given specific distances covered each hour. -/
theorem average_speed_of_car
  (d1 d2 d3 d4 : ℝ)
  (t1 t2 t3 t4 : ℝ)
  (h1 : d1 = 20)
  (h2 : d2 = 40)
  (h3 : d3 = 60)
  (h4 : d4 = 100)
  (h5 : t1 = 1)
  (h6 : t2 = 1)
  (h7 : t3 = 1)
  (h8 : t4 = 1) :
  (d1 + d2 + d3 + d4) / (t1 + t2 + t3 + t4) = 55 :=
by sorry

end average_speed_of_car_l1458_145825


namespace simplify_expression_l1458_145893

variable {R : Type*} [Field R]

theorem simplify_expression (x y z : R) (h₁ : x ≠ 0) (h₂ : y ≠ 0) (h₃ : z ≠ 0) :
  (x + y + z)⁻¹ * (x⁻¹ + y⁻¹ + z⁻¹) = x⁻¹ * y⁻¹ * z⁻¹ :=
sorry

end simplify_expression_l1458_145893


namespace average_tree_height_l1458_145866

def mixed_num_to_improper (whole: ℕ) (numerator: ℕ) (denominator: ℕ) : Rat :=
  whole + (numerator / denominator)

theorem average_tree_height 
  (elm : Rat := mixed_num_to_improper 11 2 3)
  (oak : Rat := mixed_num_to_improper 17 5 6)
  (pine : Rat := mixed_num_to_improper 15 1 2)
  (num_trees : ℕ := 3) :
  ((elm + oak + pine) / num_trees) = (15 : Rat) := 
  sorry

end average_tree_height_l1458_145866


namespace find_range_of_m_l1458_145819

open Real

-- Definition for proposition p (the discriminant condition)
def real_roots (m : ℝ) : Prop := (3 * 3) - 4 * m ≥ 0

-- Definition for proposition q (ellipse with foci on x-axis conditions)
def is_ellipse (m : ℝ) : Prop := 
  9 - m > 0 ∧ 
  m - 2 > 0 ∧ 
  9 - m > m - 2

-- Lean statement for the mathematically equivalent proof problem
theorem find_range_of_m (m : ℝ) : (real_roots m ∧ is_ellipse m) → (2 < m ∧ m ≤ 9 / 4) := 
by
  sorry

end find_range_of_m_l1458_145819


namespace probability_of_negative_m_l1458_145813

theorem probability_of_negative_m (m : ℤ) (h₁ : -2 ≤ m) (h₂ : m < (9 : ℤ) / 4) :
  ∃ (neg_count total_count : ℤ), 
    (neg_count = 2) ∧ (total_count = 5) ∧ (m ∈ {i : ℤ | -2 ≤ i ∧ i < 2 ∧ i < 9 / 4}) → 
    (neg_count / total_count = 2 / 5) :=
sorry

end probability_of_negative_m_l1458_145813


namespace annual_increase_rate_l1458_145887

theorem annual_increase_rate (r : ℝ) : 
  (6400 * (1 + r) * (1 + r) = 8100) → r = 0.125 :=
by sorry

end annual_increase_rate_l1458_145887


namespace hyperbola_eccentricity_sqrt5_l1458_145834

noncomputable def eccentricity_of_hyperbola (a b : ℝ) : ℝ :=
  Real.sqrt (1 + (b/a)^2)

theorem hyperbola_eccentricity_sqrt5
  (a b : ℝ)
  (h : ∀ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1) → (y = x^2 + 1) → (x, y) = (1, 2)) :
  eccentricity_of_hyperbola a b = Real.sqrt 5 :=
by sorry

end hyperbola_eccentricity_sqrt5_l1458_145834


namespace domain_composite_function_l1458_145859

theorem domain_composite_function (f : ℝ → ℝ) :
  (∀ x, 1 ≤ x ∧ x ≤ 3 → f x = y) →
  (∀ x, 1 ≤ x ∧ x ≤ 2 → f (2^x - 1) = y) :=
by
  sorry

end domain_composite_function_l1458_145859


namespace slope_of_monotonically_decreasing_function_l1458_145806

theorem slope_of_monotonically_decreasing_function
  (k b : ℝ)
  (H : ∀ x₁ x₂, x₁ ≤ x₂ → k * x₁ + b ≥ k * x₂ + b) : k < 0 := sorry

end slope_of_monotonically_decreasing_function_l1458_145806


namespace abs_eq_neg_iff_non_positive_l1458_145830

theorem abs_eq_neg_iff_non_positive (a : ℝ) : |a| = -a → a ≤ 0 :=
by
  intro h
  sorry

end abs_eq_neg_iff_non_positive_l1458_145830


namespace hundredth_odd_integer_not_divisible_by_five_l1458_145821

def odd_positive_integer (n : ℕ) : ℕ := 2 * n - 1

theorem hundredth_odd_integer_not_divisible_by_five :
  odd_positive_integer 100 = 199 ∧ ¬ (199 % 5 = 0) :=
by
  sorry

end hundredth_odd_integer_not_divisible_by_five_l1458_145821


namespace larger_number_of_hcf_and_lcm_factors_l1458_145850

theorem larger_number_of_hcf_and_lcm_factors :
  ∃ (a b : ℕ), (∀ d, d ∣ a ∧ d ∣ b → d ≤ 20) ∧ (∃ x y, x * y * 20 = a * b ∧ x * 20 = a ∧ y * 20 = b ∧ x > y ∧ x = 15 ∧ y = 11) → max a b = 300 :=
by sorry

end larger_number_of_hcf_and_lcm_factors_l1458_145850


namespace meaningful_expr_iff_x_ne_neg_5_l1458_145800

theorem meaningful_expr_iff_x_ne_neg_5 (x : ℝ) : (x + 5 ≠ 0) ↔ (x ≠ -5) :=
by
  sorry

end meaningful_expr_iff_x_ne_neg_5_l1458_145800


namespace function_intersects_line_at_most_once_l1458_145895

variable {α β : Type} [Nonempty α]

def function_intersects_at_most_once (f : α → β) (a : α) : Prop :=
  ∀ (b b' : β), f a = b → f a = b' → b = b'

theorem function_intersects_line_at_most_once {α β : Type} [Nonempty α] (f : α → β) (a : α) :
  function_intersects_at_most_once f a :=
by
  sorry

end function_intersects_line_at_most_once_l1458_145895


namespace sale_book_cost_l1458_145870

variable (x : ℝ)

def fiveSaleBooksCost (x : ℝ) : ℝ :=
  5 * x

def onlineBooksCost : ℝ :=
  40

def bookstoreBooksCost : ℝ :=
  3 * 40

def totalCost (x : ℝ) : ℝ :=
  fiveSaleBooksCost x + onlineBooksCost + bookstoreBooksCost

theorem sale_book_cost :
  totalCost x = 210 → x = 10 := by
  sorry

end sale_book_cost_l1458_145870


namespace martha_profit_l1458_145841

theorem martha_profit :
  let loaves_baked := 60
  let cost_per_loaf := 1
  let morning_price := 3
  let afternoon_price := 3 * 0.75
  let evening_price := 2
  let morning_loaves := loaves_baked / 3
  let afternoon_loaves := (loaves_baked - morning_loaves) / 2
  let evening_loaves := loaves_baked - morning_loaves - afternoon_loaves
  let morning_revenue := morning_loaves * morning_price
  let afternoon_revenue := afternoon_loaves * afternoon_price
  let evening_revenue := evening_loaves * evening_price
  let total_revenue := morning_revenue + afternoon_revenue + evening_revenue
  let total_cost := loaves_baked * cost_per_loaf
  let profit := total_revenue - total_cost
  profit = 85 := 
by
  sorry

end martha_profit_l1458_145841


namespace triangle_interior_angle_at_least_one_leq_60_l1458_145832

theorem triangle_interior_angle_at_least_one_leq_60 {α β γ : ℝ} :
  α + β + γ = 180 →
  (α > 60 ∧ β > 60 ∧ γ > 60) → false :=
by
  intro hsum hgt
  have hα : α > 60 := hgt.1
  have hβ : β > 60 := hgt.2.1
  have hγ : γ > 60 := hgt.2.2
  have h_total: α + β + γ > 60 + 60 + 60 := add_lt_add (add_lt_add hα hβ) hγ
  linarith

end triangle_interior_angle_at_least_one_leq_60_l1458_145832


namespace least_value_l1458_145848

theorem least_value : ∀ x y : ℝ, (xy + 1)^2 + (x - y)^2 ≥ 1 :=
by
  sorry

end least_value_l1458_145848


namespace negation_of_universal_sin_pos_l1458_145882

theorem negation_of_universal_sin_pos :
  ¬ (∀ x : ℝ, Real.sin x > 0) ↔ ∃ x : ℝ, Real.sin x ≤ 0 :=
by sorry

end negation_of_universal_sin_pos_l1458_145882


namespace intersection_of_lines_l1458_145876

theorem intersection_of_lines :
  ∃ x y : ℚ, (12 * x - 3 * y = 33) ∧ (8 * x + 2 * y = 18) ∧ (x = 29 / 12 ∧ y = -2 / 3) :=
by {
  sorry
}

end intersection_of_lines_l1458_145876


namespace area_of_sheet_is_correct_l1458_145881

noncomputable def area_of_rolled_sheet (length width height thickness : ℝ) : ℝ :=
  (length * width * height) / thickness

theorem area_of_sheet_is_correct :
  area_of_rolled_sheet 80 20 5 0.1 = 80000 :=
by
  -- The proof is omitted (sorry).
  sorry

end area_of_sheet_is_correct_l1458_145881


namespace find_line_through_M_and_parallel_l1458_145833
-- Lean code to represent the proof problem

def M : Prop := ∃ (x y : ℝ), 3 * x + 4 * y - 5 = 0 ∧ 2 * x - 3 * y + 8 = 0 

def line_parallel : Prop := ∃ (m b : ℝ), 2 * m + b = 0

theorem find_line_through_M_and_parallel :
  M → line_parallel → ∃ (a b c : ℝ), (a = 2) ∧ (b = 1) ∧ (c = 0) :=
by
  intros hM hLineParallel
  sorry

end find_line_through_M_and_parallel_l1458_145833


namespace num_ways_to_pay_16_rubles_l1458_145891

theorem num_ways_to_pay_16_rubles :
  ∃! (n : ℕ), n = 13 ∧ ∀ (x y z : ℕ), (x ≥ 0) ∧ (y ≥ 0) ∧ (z ≥ 0) ∧ 
  (10 * x + 2 * y + 1 * z = 16) ∧ (x < 2) ∧ (y + z > 0) := sorry

end num_ways_to_pay_16_rubles_l1458_145891


namespace infinite_series_evaluation_l1458_145808

theorem infinite_series_evaluation :
  (∑' m : ℕ, ∑' n : ℕ, 1 / (m * n * (m + n + 2))) = 3 :=
  sorry

end infinite_series_evaluation_l1458_145808


namespace find_share_of_C_l1458_145858

-- Definitions and assumptions
def share_in_ratio (x : ℕ) : Prop :=
  let a := 2 * x
  let b := 3 * x
  let c := 4 * x
  a + b + c = 945

-- Statement to prove
theorem find_share_of_C :
  ∃ x : ℕ, share_in_ratio x ∧ 4 * x = 420 :=
by
  -- We skip the proof here.
  sorry

end find_share_of_C_l1458_145858


namespace find_k_l1458_145802

theorem find_k (d : ℤ) (h : d ≠ 0) (a : ℤ → ℤ) 
  (a_def : ∀ n, a n = 4 * d + (n - 1) * d) 
  (geom_mean_condition : ∃ k, a k * a k = a 1 * a 6) : 
  ∃ k, k = 3 := 
by
  sorry

end find_k_l1458_145802


namespace screen_width_l1458_145884

theorem screen_width
  (A : ℝ) -- Area of the screen
  (h : ℝ) -- Height of the screen
  (w : ℝ) -- Width of the screen
  (area_eq : A = 21) -- Condition 1: Area is 21 sq ft
  (height_eq : h = 7) -- Condition 2: Height is 7 ft
  (area_formula : A = w * h) -- Condition 3: Area formula
  : w = 3 := -- Conclusion: Width is 3 ft
sorry

end screen_width_l1458_145884


namespace problem_statements_correct_l1458_145892

theorem problem_statements_correct :
    (∀ (select : ℕ) (male female : ℕ), male = 4 → female = 3 → 
      (select = (4 * 3 + 3)) → select ≥ 12 = false) ∧
    (∀ (a1 a2 a3 : ℕ), 
      a2 = 0 ∨ a2 = 1 ∨ a2 = 2 →
      (∃ (cases : ℕ), cases = 14) →
      cases = 14) ∧
    (∀ (ways enter exit : ℕ), enter = 4 → exit = 4 - 1 →
      (ways = enter * exit) → ways = 12 = false) ∧
    (∀ (a b : ℕ),
      a > 0 ∧ a < 10 ∧ b > 0 ∧ b < 10 →
      (∃ (log_val : ℕ), log_val = 54) →
      log_val = 54) := by
  admit

end problem_statements_correct_l1458_145892


namespace hyperbola_range_of_k_l1458_145865

theorem hyperbola_range_of_k (k : ℝ) :
  (∃ (x y : ℝ), (x^2)/(k-3) + (y^2)/(k+3) = 1 ∧ 
  (k-3 < 0) ∧ (k+3 > 0)) → (-3 < k ∧ k < 3) :=
by
  sorry

end hyperbola_range_of_k_l1458_145865


namespace smallest_angle_between_radii_l1458_145849

theorem smallest_angle_between_radii (n : ℕ) (k : ℕ) (angle_step : ℕ) (angle_smallest : ℕ) 
(h_n : n = 40) 
(h_k : k = 23) 
(h_angle_step : angle_step = k) 
(h_angle_smallest : angle_smallest = 23) : 
angle_smallest = 23 :=
sorry

end smallest_angle_between_radii_l1458_145849


namespace continuous_polynomial_continuous_cosecant_l1458_145851

-- Prove that the function \( f(x) = 2x^2 - 1 \) is continuous on \(\mathbb{R}\)
theorem continuous_polynomial : Continuous (fun x : ℝ => 2 * x^2 - 1) :=
sorry

-- Prove that the function \( g(x) = (\sin x)^{-1} \) is continuous on \(\mathbb{R}\) \setminus \(\{ k\pi \mid k \in \mathbb{Z} \} \)
theorem continuous_cosecant : ∀ x : ℝ, x ∉ Set.range (fun k : ℤ => k * Real.pi) → ContinuousAt (fun x : ℝ => (Real.sin x)⁻¹) x :=
sorry

end continuous_polynomial_continuous_cosecant_l1458_145851


namespace ratio_of_spinsters_to_cats_l1458_145809

def spinsters := 22
def cats := spinsters + 55

theorem ratio_of_spinsters_to_cats : (spinsters : ℝ) / (cats : ℝ) = 2 / 7 := 
by
  sorry

end ratio_of_spinsters_to_cats_l1458_145809


namespace scientific_notation_gdp_2022_l1458_145854

def gdp_2022_fujian : ℝ := 53100 * 10^9

theorem scientific_notation_gdp_2022 : 
  (53100 * 10^9) = 5.31 * 10^12 :=
by
  -- The proof is based on the understanding that 53100 * 10^9 can be rewritten as 5.31 * 10^12
  -- However, this proof is currently omitted with a placeholder.
  sorry

end scientific_notation_gdp_2022_l1458_145854


namespace eval_floor_ceil_sum_l1458_145894

noncomputable def floor (x : ℝ) : ℤ := Int.floor x
noncomputable def ceil (x : ℝ) : ℤ := Int.ceil x

theorem eval_floor_ceil_sum : floor (-3.67) + ceil 34.7 = 31 := by
  sorry

end eval_floor_ceil_sum_l1458_145894


namespace sum_of_m_integers_l1458_145889

theorem sum_of_m_integers :
  ∀ (m : ℤ), 
    (∀ (x : ℚ), (x - 10) / 5 ≤ -1 - x / 5 ∧ x - 1 > -m / 2) → 
    (∃ x_max x_min : ℤ, x_max + x_min = -2 ∧ 
                        (x_max ≤ 5 / 2 ∧ x_min ≤ 5 / 2) ∧ 
                        (1 - m / 2 < x_min ∧ 1 - m / 2 < x_max)) →
  (10 < m ∧ m ≤ 12) → m = 11 ∨ m = 12 → 11 + 12 = 23 :=
by sorry

end sum_of_m_integers_l1458_145889


namespace total_current_ages_l1458_145897

theorem total_current_ages (T : ℕ) : (T - 12 = 54) → T = 66 :=
by
  sorry

end total_current_ages_l1458_145897


namespace regular_polygon_sides_l1458_145843

-- Define the central angle and number of sides of a regular polygon
def central_angle (θ : ℝ) := θ = 30
def number_of_sides (n : ℝ) := 360 / 30 = n

-- Theorem to prove that the number of sides of the regular polygon is 12 given the central angle is 30 degrees
theorem regular_polygon_sides (θ n : ℝ) (hθ : central_angle θ) : number_of_sides n → n = 12 :=
sorry

end regular_polygon_sides_l1458_145843


namespace nonneg_sol_eq_l1458_145820

theorem nonneg_sol_eq {a b c : ℝ} (a_nonneg : 0 ≤ a) (b_nonneg : 0 ≤ b) (c_nonneg : 0 ≤ c) 
  (h1 : a * (a + b) = b * (b + c)) (h2 : b * (b + c) = c * (c + a)) : 
  a = b ∧ b = c := 
sorry

end nonneg_sol_eq_l1458_145820


namespace average_temp_addington_l1458_145886

def temperatures : List ℚ := [60, 59, 56, 53, 49, 48, 46]

def average_temp (temps : List ℚ) : ℚ := (temps.sum) / temps.length

theorem average_temp_addington :
  average_temp temperatures = 53 := by
  sorry

end average_temp_addington_l1458_145886


namespace finite_ring_identity_l1458_145885

variable {A : Type} [Ring A] [Fintype A]
variables (a b : A)

theorem finite_ring_identity (h : (ab - 1) * b = 0) : b * (ab - 1) = 0 :=
sorry

end finite_ring_identity_l1458_145885


namespace even_sum_probability_l1458_145869

-- Definition of probabilities for the first wheel
def prob_first_even : ℚ := 2 / 6
def prob_first_odd  : ℚ := 4 / 6

-- Definition of probabilities for the second wheel
def prob_second_even : ℚ := 3 / 8
def prob_second_odd  : ℚ := 5 / 8

-- The expected probability of the sum being even
theorem even_sum_probability : prob_first_even * prob_second_even + prob_first_odd * prob_second_odd = 13 / 24 := by
  sorry

end even_sum_probability_l1458_145869


namespace average_of_new_sequence_l1458_145855

theorem average_of_new_sequence (c d : ℕ) (h : d = (c + (c + 1) + (c + 2) + (c + 3) + (c + 4) + (c + 5) + (c + 6)) / 7) : 
  (d + (d + 1) + (d + 2) + (d + 3) + (d + 4) + (d + 5) + (d + 6)) / 7 = c + 6 :=
by
  sorry

end average_of_new_sequence_l1458_145855


namespace martha_saving_l1458_145852

-- Definitions for the conditions
def daily_allowance : ℕ := 12
def half_daily_allowance : ℕ := daily_allowance / 2
def quarter_daily_allowance : ℕ := daily_allowance / 4
def days_saving_half : ℕ := 6
def day_saving_quarter : ℕ := 1

-- Statement to be proved
theorem martha_saving :
  (days_saving_half * half_daily_allowance) + (day_saving_quarter * quarter_daily_allowance) = 39 := by
  sorry

end martha_saving_l1458_145852


namespace right_triangle_hypotenuse_segment_ratio_l1458_145826

theorem right_triangle_hypotenuse_segment_ratio
  (x : ℝ)
  (h₀ : 0 < x)
  (AB BC : ℝ)
  (h₁ : AB = 3 * x)
  (h₂ : BC = 4 * x) :
  ∃ AD DC : ℝ, AD / DC = 3 := 
by
  sorry

end right_triangle_hypotenuse_segment_ratio_l1458_145826


namespace range_of_p_l1458_145818

def f (x : ℝ) : ℝ := x^3 - x^2 - 10 * x

def A : Set ℝ := {x | 3*x^2 - 2*x - 10 ≤ 0}

def B (p : ℝ) : Set ℝ := {x | p + 1 ≤ x ∧ x ≤ 2*p - 1}

theorem range_of_p (p : ℝ) (h : A ∪ B p = A) : p ≤ 3 :=
by
  sorry

end range_of_p_l1458_145818


namespace tan_210_eq_neg_sqrt3_over_3_l1458_145827

noncomputable def angle_210 : ℝ := 210 * (Real.pi / 180)
noncomputable def angle_30 : ℝ := 30 * (Real.pi / 180)

theorem tan_210_eq_neg_sqrt3_over_3 : Real.tan angle_210 = -Real.sqrt 3 / 3 :=
by
  sorry -- Proof omitted

end tan_210_eq_neg_sqrt3_over_3_l1458_145827


namespace mark_charged_more_hours_l1458_145829

theorem mark_charged_more_hours (P K M : ℕ) 
  (h1 : P + K + M = 135)
  (h2 : P = 2 * K)
  (h3 : P = M / 3) :
  M - K = 75 := by {

sorry
}

end mark_charged_more_hours_l1458_145829


namespace inequality_xy_l1458_145811

theorem inequality_xy {x y : ℝ} (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : x + y ≤ 1) :
  12 * x * y ≤ 4 * x * (1 - y) + 9 * y * (1 - x) := by
  sorry

end inequality_xy_l1458_145811


namespace actual_time_before_storm_is_18_18_l1458_145804

theorem actual_time_before_storm_is_18_18 :
  ∃ h m : ℕ, (h = 18) ∧ (m = 18) ∧ 
            ((09 = (if h == 0 then 1 else h - 1) ∨ 09 = (if h == 23 then 0 else h + 1)) ∧ 
             (09 = (if m == 0 then 1 else m - 1) ∨ 09 = (if m == 59 then 0 else m + 1))) := 
  sorry

end actual_time_before_storm_is_18_18_l1458_145804


namespace semicircles_problem_l1458_145835

-- Define the problem in Lean
theorem semicircles_problem 
  (D : ℝ) -- Diameter of the large semicircle
  (N : ℕ) -- Number of small semicircles
  (r : ℝ) -- Radius of each small semicircle
  (H1 : D = 2 * N * r) -- Combined diameter of small semicircles is equal to the large semicircle's diameter
  (H2 : (N * (π * r^2 / 2)) / ((π * (N * r)^2 / 2) - (N * (π * r^2 / 2))) = 1 / 10) -- Ratio of areas condition
  : N = 11 :=
   sorry -- Proof to be filled in later

end semicircles_problem_l1458_145835


namespace not_perfect_square_l1458_145871

theorem not_perfect_square (n : ℕ) : ¬ ∃ k : ℕ, k^2 = 3^n + 2 * 17^n := sorry

end not_perfect_square_l1458_145871


namespace expand_and_simplify_product_l1458_145868

theorem expand_and_simplify_product :
  5 * (x + 6) * (x + 2) * (x + 7) = 5 * x^3 + 75 * x^2 + 340 * x + 420 := 
by
  sorry

end expand_and_simplify_product_l1458_145868


namespace smallest_natural_number_with_condition_l1458_145842

theorem smallest_natural_number_with_condition {N : ℕ} :
  (N % 10 = 6) ∧ (4 * N = (6 * 10 ^ ((Nat.digits 10 (N / 10)).length) + (N / 10))) ↔ N = 153846 :=
by
  sorry

end smallest_natural_number_with_condition_l1458_145842


namespace cos_theta_equal_neg_inv_sqrt_5_l1458_145831

noncomputable def f (x : ℝ) : ℝ :=
  2 * Real.sin x - Real.cos x

theorem cos_theta_equal_neg_inv_sqrt_5 (θ : ℝ) (h_max : ∀ x : ℝ, f θ ≥ f x) : Real.cos θ = -1 / Real.sqrt 5 :=
by
  sorry

end cos_theta_equal_neg_inv_sqrt_5_l1458_145831


namespace range_of_a_l1458_145856

def f (x : ℝ) : ℝ := x^3 + x + 1

theorem range_of_a (a : ℝ) : (∀ x : ℝ, f (x^2 + a) + f (a * x) > 2) → 0 < a ∧ a < 4 := 
by 
  sorry

end range_of_a_l1458_145856


namespace perpendicular_lines_l1458_145805

theorem perpendicular_lines (a : ℝ) : (x + 2*y + 1 = 0) ∧ (ax + y - 2 = 0) → a = -2 :=
by
  sorry

end perpendicular_lines_l1458_145805


namespace initial_person_count_l1458_145853

theorem initial_person_count
  (avg_weight_increase : ℝ)
  (weight_old_person : ℝ)
  (weight_new_person : ℝ)
  (h1 : avg_weight_increase = 4.2)
  (h2 : weight_old_person = 65)
  (h3 : weight_new_person = 98.6) :
  ∃ n : ℕ, weight_new_person - weight_old_person = avg_weight_increase * n ∧ n = 8 := 
by
  sorry

end initial_person_count_l1458_145853


namespace fraction_of_visitors_who_both_enjoyed_and_understood_is_3_over_8_l1458_145810

def visitors_enjoyed_understood_fraction (E U : ℕ) (total_visitors no_enjoy_no_understood : ℕ) : Prop :=
  E = U ∧
  no_enjoy_no_understood = 110 ∧
  total_visitors = 440 ∧
  E = (total_visitors - no_enjoy_no_understood) / 2 ∧
  E = 165 ∧
  (E / total_visitors) = 3 / 8

theorem fraction_of_visitors_who_both_enjoyed_and_understood_is_3_over_8 :
  ∃ (E U : ℕ), visitors_enjoyed_understood_fraction E U 440 110 :=
by
  sorry

end fraction_of_visitors_who_both_enjoyed_and_understood_is_3_over_8_l1458_145810


namespace perpendicular_lines_l1458_145837

noncomputable def l1_slope (a : ℝ) : ℝ := -a / (1 - a)
noncomputable def l2_slope (a : ℝ) : ℝ := -(a - 1) / (2 * a + 3)

theorem perpendicular_lines (a : ℝ) 
  (h1 : ∀ x y : ℝ, a * x + (1 - a) * y = 3 → Prop) 
  (h2 : ∀ x y : ℝ, (a - 1) * x + (2 * a + 3) * y = 2 → Prop) 
  (hp : l1_slope a * l2_slope a = -1) : a = -3 := by
  sorry

end perpendicular_lines_l1458_145837


namespace number_of_ordered_pairs_l1458_145846

noncomputable def max (x y : ℕ) : ℕ := if x > y then x else y

def valid_pair_count (k : ℕ) : ℕ := 2 * k + 1

def pairs_count (a b : ℕ) : ℕ := 
  valid_pair_count 5 * valid_pair_count 3 * valid_pair_count 2 * valid_pair_count 1

theorem number_of_ordered_pairs : pairs_count 2 3 = 1155 := 
  sorry

end number_of_ordered_pairs_l1458_145846


namespace length_each_stitch_l1458_145814

theorem length_each_stitch 
  (hem_length_feet : ℝ) 
  (stitches_per_minute : ℝ) 
  (hem_time_minutes : ℝ) 
  (hem_length_inches : ℝ) 
  (total_stitches : ℝ) 
  (stitch_length_inches : ℝ) 
  (h1 : hem_length_feet = 3) 
  (h2 : stitches_per_minute = 24) 
  (h3 : hem_time_minutes = 6) 
  (h4 : hem_length_inches = hem_length_feet * 12) 
  (h5 : total_stitches = stitches_per_minute * hem_time_minutes) 
  (h6 : stitch_length_inches = hem_length_inches / total_stitches) :
  stitch_length_inches = 0.25 :=
by
  sorry

end length_each_stitch_l1458_145814


namespace intersection_of_A_and_B_union_of_A_and_B_l1458_145877

def A : Set ℝ := {x | x * (9 - x) > 0}
def B : Set ℝ := {x | x ≤ 3}

theorem intersection_of_A_and_B : A ∩ B = {x | 0 < x ∧ x ≤ 3} :=
sorry

theorem union_of_A_and_B : A ∪ B = {x | x < 9} :=
sorry

end intersection_of_A_and_B_union_of_A_and_B_l1458_145877


namespace difference_of_same_prime_factors_l1458_145864

theorem difference_of_same_prime_factors (n : ℕ) :
  ∃ a b : ℕ, a - b = n ∧ (a.primeFactors.card = b.primeFactors.card) :=
by
  sorry

end difference_of_same_prime_factors_l1458_145864


namespace water_left_l1458_145801

theorem water_left (initial_water: ℚ) (science_experiment_use: ℚ) (plant_watering_use: ℚ)
  (h1: initial_water = 3)
  (h2: science_experiment_use = 5 / 4)
  (h3: plant_watering_use = 1 / 2) :
  (initial_water - science_experiment_use - plant_watering_use = 5 / 4) :=
by
  rw [h1, h2, h3]
  norm_num

end water_left_l1458_145801
