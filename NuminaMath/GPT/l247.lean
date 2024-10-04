import Mathlib

namespace total_area_of_union_of_six_triangles_l247_247114

theorem total_area_of_union_of_six_triangles :
  let s := 2 * Real.sqrt 2
  let area_one_triangle := (Real.sqrt 3 / 4) * s^2
  let total_area_without_overlaps := 6 * area_one_triangle
  let side_overlap := Real.sqrt 2
  let area_one_overlap := (Real.sqrt 3 / 4) * side_overlap ^ 2
  let total_overlap_area := 5 * area_one_overlap
  let net_area := total_area_without_overlaps - total_overlap_area
  net_area = 9.5 * Real.sqrt 3 := 
by
  sorry

end total_area_of_union_of_six_triangles_l247_247114


namespace unique_solution_f_l247_247448

def f : ℚ → ℚ := sorry

theorem unique_solution_f (f : ℚ → ℚ) (h1 : f 1 = 2)
  (h2 : ∀ x y : ℚ, f (x * y) = f x * f y - f (x + y) + 1) : 
  ∀ x : ℚ, f x = x + 1 := 
by
  sorry 

end unique_solution_f_l247_247448


namespace ceil_pow_sq_cardinality_l247_247307

noncomputable def ceil_pow_sq_values (x : ℝ) (h : 11 < x ∧ x ≤ 12) : ℕ :=
  ((Real.ceil(x^2)) - (Real.ceil(121)) + 1)

theorem ceil_pow_sq_cardinality :
  ∀ (x : ℝ), (11 < x ∧ x ≤ 12) → ceil_pow_sq_values x _ = 23 :=
by
  intro x hx
  let attrs := (11 < x ∧ x ≤ 12)
  sorry

end ceil_pow_sq_cardinality_l247_247307


namespace reflect_y_axis_matrix_l247_247733

theorem reflect_y_axis_matrix : 
  ∃ A : Matrix (Fin 2) (Fin 2) ℝ, (
    (∀ v : (Fin 2 → ℝ), v = ![1, 0] → A.mulVec v = ![-1, 0]) ∧ 
    (∀ v : (Fin 2 → ℝ), v = ![0, 1] → A.mulVec v = ![0, 1])
  ) ∧ A = ![![-1, 0], ![0, 1]] :=
begin
  sorry
end

end reflect_y_axis_matrix_l247_247733


namespace region_area_l247_247677

theorem region_area (x y : ℝ) : (x^2 + y^2 + 6*x - 4*y - 11 = 0) → (∃ (A : ℝ), A = 24 * Real.pi) :=
by
  sorry

end region_area_l247_247677


namespace solution_set_of_inequality_l247_247809

theorem solution_set_of_inequality : {x : ℝ | x^2 < 2 * x} = {x : ℝ | 0 < x ∧ x < 2} :=
by
  sorry

end solution_set_of_inequality_l247_247809


namespace num_distinct_prime_factors_2310_l247_247853

def is_prime (p : ℕ) : Prop :=
  2 ≤ p ∧ ∀ (n : ℕ), n ∣ p → n = 1 ∨ n = p

def distinct_prime_factors (n : ℕ) (s : Set ℕ) : Prop :=
  (∀ p, p ∈ s → is_prime p) ∧ (∀ p, is_prime p → p ∣ n → p ∈ s) ∧ (∀ p₁ p₂, p₁ ∈ s → p₂ ∈ s → p₁ ≠ p₂)

theorem num_distinct_prime_factors_2310 : 
  ∃ s : Set ℕ, distinct_prime_factors 2310 s ∧ s.card = 5 :=
sorry

end num_distinct_prime_factors_2310_l247_247853


namespace a3_mul_a7_eq_36_l247_247462

-- Definition of a geometric sequence term
def geometric_sequence (a : ℕ → ℤ) : Prop :=
  ∃ r : ℤ, ∀ n : ℕ, a (n + 1) = r * a n

-- Given conditions
def a (n : ℕ) : ℤ := sorry  -- Placeholder for the geometric sequence

axiom a5_eq_6 : a 5 = 6  -- Given that a_5 = 6

axiom geo_seq : geometric_sequence a  -- The sequence is geometric

-- Problem statement: Prove that a_3 * a_7 = 36
theorem a3_mul_a7_eq_36 : a 3 * a 7 = 36 :=
  sorry

end a3_mul_a7_eq_36_l247_247462


namespace least_common_multiple_1008_672_l247_247679

theorem least_common_multiple_1008_672 : Nat.lcm 1008 672 = 2016 := by
  -- Add the prime factorizations and show the LCM calculation
  have h1 : 1008 = 2^4 * 3^2 * 7 := by sorry
  have h2 : 672 = 2^5 * 3 * 7 := by sorry
  -- Utilize the factorizations to compute LCM
  have calc1 : Nat.lcm (2^4 * 3^2 * 7) (2^5 * 3 * 7) = 2^5 * 3^2 * 7 := by sorry
  -- Show the calculation of 2^5 * 3^2 * 7
  have calc2 : 2^5 * 3^2 * 7 = 2016 := by sorry
  -- Therefore, LCM of 1008 and 672 is 2016
  exact calc2

end least_common_multiple_1008_672_l247_247679


namespace complex_power_difference_l247_247625

theorem complex_power_difference (i : ℂ) (h : i^2 = -1) : (1 + i) ^ 18 - (1 - i) ^ 18 = 1024 * i :=
by
  sorry

end complex_power_difference_l247_247625


namespace age_hence_l247_247567

theorem age_hence (A x : ℕ) (hA : A = 24) (hx : 4 * (A + x) - 4 * (A - 3) = A) : x = 3 :=
by {
  sorry
}

end age_hence_l247_247567


namespace sum_of_solutions_l247_247680

theorem sum_of_solutions : 
  ∃ x1 x2 x3 : ℝ, (x1 = 10 ∧ x2 = 50/7 ∧ x3 = 50 ∧ (x1 + x2 + x3 = 470 / 7) ∧ 
  (∀ x : ℝ, x = abs (3 * x - abs (50 - 3 * x)) → (x = x1 ∨ x = x2 ∨ x = x3))) := 
sorry

end sum_of_solutions_l247_247680


namespace weierstrass_limit_l247_247793

theorem weierstrass_limit (a_n : ℕ → ℝ) (M : ℝ) :
  (∀ n m, n ≤ m → a_n n ≤ a_n m) → 
  (∀ n, a_n n ≤ M ) → 
  ∃ c, ∀ ε > 0, ∃ N, ∀ n ≥ N, |a_n n - c| < ε :=
by
  sorry

end weierstrass_limit_l247_247793


namespace find_b_l247_247848

-- Define the function f(x)
def f (x : ℝ) : ℝ := 5 * x - 7

-- State the theorem
theorem find_b (b : ℝ) : f b = 0 ↔ b = 7 / 5 := by
  sorry

end find_b_l247_247848


namespace ratio_Y_to_Z_l247_247994

variables (X Y Z : ℕ)

def population_relation1 (X Y : ℕ) : Prop := X = 3 * Y
def population_relation2 (X Z : ℕ) : Prop := X = 6 * Z

theorem ratio_Y_to_Z (h1 : population_relation1 X Y) (h2 : population_relation2 X Z) : Y / Z = 2 :=
  sorry

end ratio_Y_to_Z_l247_247994


namespace net_gain_difference_l247_247700

def first_applicant_salary : ℝ := 42000
def first_applicant_training_cost_per_month : ℝ := 1200
def first_applicant_training_months : ℝ := 3
def first_applicant_revenue : ℝ := 93000

def second_applicant_salary : ℝ := 45000
def second_applicant_hiring_bonus_percentage : ℝ := 0.01
def second_applicant_revenue : ℝ := 92000

def first_applicant_total_cost : ℝ := first_applicant_salary + first_applicant_training_cost_per_month * first_applicant_training_months
def first_applicant_net_gain : ℝ := first_applicant_revenue - first_applicant_total_cost

def second_applicant_hiring_bonus : ℝ := second_applicant_salary * second_applicant_hiring_bonus_percentage
def second_applicant_total_cost : ℝ := second_applicant_salary + second_applicant_hiring_bonus
def second_applicant_net_gain : ℝ := second_applicant_revenue - second_applicant_total_cost

theorem net_gain_difference :
  first_applicant_net_gain - second_applicant_net_gain = 850 := by
  sorry

end net_gain_difference_l247_247700


namespace greatest_multiple_of_5_and_6_less_than_1000_l247_247392

theorem greatest_multiple_of_5_and_6_less_than_1000 : 
    ∃ k, (k % 5 = 0) ∧ (k % 6 = 0) ∧ (k < 1000) ∧ (∀ m, (m % 5 = 0) ∧ (m % 6 = 0) ∧ (m < 1000) → m ≤ k) :=
begin
  use 990,
  repeat { split },
  { norm_num }, -- 990 % 5 = 0
  { norm_num }, -- 990 % 6 = 0
  { norm_num }, -- 990 < 1000
  { intros m hm, 
    cases hm with h5 h6,
    cases h6 with h6 hlt,
    have : m % 30 = 0 := by { sorry }, -- Show that m is a multiple of LCM(5, 6)
    apply le_of_lt,
    have hle : m/30 < ↑(1000/30) := by { sorry }, -- Compare the greatest multiple of 30 less than 1000
    exact hle,
  }
end

end greatest_multiple_of_5_and_6_less_than_1000_l247_247392


namespace power_i_2015_l247_247460

theorem power_i_2015 (i : ℂ) (hi : i^2 = -1) : i^2015 = -i :=
by
  have h1 : i^4 = 1 := by sorry
  have h2 : 2015 = 4 * 503 + 3 := by norm_num
  sorry

end power_i_2015_l247_247460


namespace Smarties_remainder_l247_247726

theorem Smarties_remainder (m : ℕ) (h : m % 11 = 5) : (4 * m) % 11 = 9 :=
by
  sorry

end Smarties_remainder_l247_247726


namespace trigonometric_identity_l247_247763

theorem trigonometric_identity (α : ℝ) 
  (h : Real.sin α = 1 / 3) : 
  Real.cos (Real.pi / 2 + α) = - 1 / 3 := 
by
  sorry

end trigonometric_identity_l247_247763


namespace number_of_small_cubes_l247_247554

theorem number_of_small_cubes (X : ℕ) (h1 : ∃ k, k = 29 - X) (h2 : 4 * 4 * 4 = 64) (h3 : X + 8 * (29 - X) = 64) : X = 24 :=
by
  sorry

end number_of_small_cubes_l247_247554


namespace proof_problem_l247_247551

-- Define the conditions for the problem

def is_factor (a b : ℕ) : Prop :=
  ∃ n : ℕ, b = a * n

def is_divisor (a b : ℕ) : Prop :=
  b % a = 0

-- Statement that needs to be proven
theorem proof_problem :
  is_factor 5 65 ∧ ¬(is_divisor 19 361 ∧ ¬is_divisor 19 190) ∧ ¬(¬is_divisor 36 144 ∨ ¬is_divisor 36 73) ∧ ¬(is_divisor 14 28 ∧ ¬is_divisor 14 56) ∧ is_factor 9 144 :=
by sorry

end proof_problem_l247_247551


namespace solve_abs_inequality_l247_247214

theorem solve_abs_inequality (x : ℝ) :
  |x + 2| + |x - 2| < x + 7 ↔ -7 / 3 < x ∧ x < 7 :=
sorry

end solve_abs_inequality_l247_247214


namespace a1_greater_than_500_l247_247928

-- Set up conditions
variables (a : ℕ → ℕ) (h1 : ∀ n, 0 < a n ∧ a n < 20000)
variables (h2 : ∀ i j, i < j → gcd (a i) (a j) < a i)
variables (h3 : ∀ i j, i < j ∧ 1 ≤ i ∧ j ≤ 10000 → a i < a j)

/-- Statement to prove / lean concept as per mathematical problem  --/
theorem a1_greater_than_500 : 500 < a 1 :=
sorry

end a1_greater_than_500_l247_247928


namespace fraction_of_trunks_l247_247713

theorem fraction_of_trunks (h1 : 0.38 ≤ 1) (h2 : 0.63 ≤ 1) : 
  0.63 - 0.38 = 0.25 :=
by
  sorry

end fraction_of_trunks_l247_247713


namespace jeremy_school_distance_l247_247637

theorem jeremy_school_distance :
  ∃ d : ℝ, d = 9.375 ∧
  (∃ v : ℝ, (d = v * (15 / 60)) ∧ (d = (v + 25) * (9 / 60))) := by
  sorry

end jeremy_school_distance_l247_247637


namespace incorrect_judgment_l247_247161

-- Define propositions p and q
def p : Prop := 2 + 2 = 5
def q : Prop := 3 > 2

-- The incorrect judgment in Lean statement
theorem incorrect_judgment : ¬((p ∧ q) ∧ ¬p) :=
by
  sorry

end incorrect_judgment_l247_247161


namespace trivia_game_points_l247_247859

theorem trivia_game_points (first_round_points second_round_points points_lost last_round_points : ℤ) 
    (h1 : first_round_points = 16)
    (h2 : second_round_points = 33)
    (h3 : points_lost = 48) : 
    first_round_points + second_round_points - points_lost = 1 :=
by
    rw [h1, h2, h3]
    rfl

end trivia_game_points_l247_247859


namespace units_digit_13_times_41_l247_247153

def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_13_times_41 :
  units_digit (13 * 41) = 3 :=
sorry

end units_digit_13_times_41_l247_247153


namespace total_weight_l247_247080

def weights (M D C : ℕ): Prop :=
  D = 46 ∧ D + C = 60 ∧ C = M / 5

theorem total_weight (M D C : ℕ) (h : weights M D C) : M + D + C = 130 :=
by
  cases h with
  | intro h1 h2 =>
    cases h2 with
    | intro h2_1 h2_2 => 
      sorry

end total_weight_l247_247080


namespace cost_comparison_l247_247416

-- Definitions based on the given conditions
def suit_price : ℕ := 200
def tie_price : ℕ := 40
def num_suits : ℕ := 20
def discount_rate : ℚ := 0.9

-- Define cost expressions for the two options
def option1_cost (x : ℕ) : ℕ :=
  (suit_price * num_suits) + (tie_price * (x - num_suits))

def option2_cost (x : ℕ) : ℚ :=
  ((suit_price * num_suits + tie_price * x) * discount_rate : ℚ)

-- Main theorem to prove the given answers
theorem cost_comparison (x : ℕ) (hx : x > 20) :
  option1_cost x = 40 * x + 3200 ∧
  option2_cost x = 3600 + 36 * x ∧
  (x = 30 → option1_cost 30 < option2_cost 30) :=
by
  sorry

end cost_comparison_l247_247416


namespace problem_solution_l247_247620

theorem problem_solution
  (x y : ℝ)
  (h : 5 * x^2 - 4 * x * y + y^2 - 2 * x + 1 = 0) :
  (x - y) ^ 2007 = -1 := by
  sorry

end problem_solution_l247_247620


namespace no_solution_in_natural_numbers_l247_247508

theorem no_solution_in_natural_numbers (x y z : ℕ) (hxy : x ≠ 0) (hyz : y ≠ 0) (hzx : z ≠ 0) :
  ¬ (x / y + y / z + z / x = 1) :=
by sorry

end no_solution_in_natural_numbers_l247_247508


namespace rectangular_prism_sum_l247_247638

theorem rectangular_prism_sum :
  let edges := 12
  let corners := 8
  let faces := 6
  edges + corners + faces = 26 := by
  sorry

end rectangular_prism_sum_l247_247638


namespace area_shaded_region_l247_247102

theorem area_shaded_region :
  let r_s := 3   -- Radius of the smaller circle
  let r_l := 3 * r_s  -- Radius of the larger circle
  let A_l := π * r_l^2  -- Area of the larger circle
  let A_s := π * r_s^2  -- Area of the smaller circle
  A_l - A_s = 72 * π := 
by
  sorry

end area_shaded_region_l247_247102


namespace chi_square_association_l247_247329

theorem chi_square_association (k : ℝ) :
  (k > 3.841 → (∃ A B, A ∧ B)) ∧ (k ≤ 2.076 → (∃ A B, ¬(A ∧ B))) :=
by
  sorry

end chi_square_association_l247_247329


namespace necessarily_positive_l247_247346

-- Conditions
variables {x y z : ℝ}

-- Statement to prove
theorem necessarily_positive (h1 : 0 < x) (h2 : x < 1) (h3 : -2 < y) (h4 : y < 0) (h5 : 2 < z) (h6 : z < 3) :
  0 < y + 2 * z :=
sorry

end necessarily_positive_l247_247346


namespace length_of_plot_l247_247355

theorem length_of_plot 
  (b : ℝ)
  (H1 : 2 * (b + 20) + 2 * b = 5300 / 26.50)
  : (b + 20 = 60) :=
sorry

end length_of_plot_l247_247355


namespace a_b_c_sum_l247_247802

-- Definitions of the conditions
def f (x : ℝ) (a b c : ℝ) : ℝ := a * x^2 + b * x + c

theorem a_b_c_sum (a b c : ℝ) :
  (∀ x : ℝ, f (x + 4) a b c = 4 * x^2 + 9 * x + 5) ∧ (∀ x : ℝ, f x a b c = a * x^2 + b * x + c) →
  a + b + c = 14 :=
by
  intros h
  sorry

end a_b_c_sum_l247_247802


namespace quadratic_equation_solution_diff_l247_247599

theorem quadratic_equation_solution_diff :
  let a := 1
  let b := -6
  let c := -40
  let discriminant := b^2 - 4 * a * c
  let root1 := (-b + Real.sqrt discriminant) / (2 * a)
  let root2 := (-b - Real.sqrt discriminant) / (2 * a)
  abs (root1 - root2) = 14 := by
  -- placeholder for the proof
  sorry

end quadratic_equation_solution_diff_l247_247599


namespace homes_termite_ridden_but_not_collapsing_fraction_l247_247062

variable (H : Type) -- Representing Homes on Gotham Street

def termite_ridden_fraction : ℚ := 1 / 3
def collapsing_fraction_given_termite_ridden : ℚ := 7 / 10

theorem homes_termite_ridden_but_not_collapsing_fraction :
  (termite_ridden_fraction * (1 - collapsing_fraction_given_termite_ridden)) = 1 / 10 :=
by
  sorry

end homes_termite_ridden_but_not_collapsing_fraction_l247_247062


namespace dealer_can_determine_values_l247_247320

def card_value_determined (a : Fin 100 → Fin 100) : Prop :=
  (∀ i j : Fin 100, i > j → a i > a j) ∧ (a 0 > a 99) ∧
  (∀ k : Fin 100, a k = k + 1)

theorem dealer_can_determine_values :
  ∃ (messages : Fin 100 → Fin 100), card_value_determined messages :=
sorry

end dealer_can_determine_values_l247_247320


namespace cost_per_serving_in_cents_after_coupon_l247_247175

def oz_per_serving : ℝ := 1
def price_per_bag : ℝ := 25
def bag_weight : ℝ := 40
def coupon : ℝ := 5
def dollars_to_cents (d : ℝ) : ℝ := d * 100

theorem cost_per_serving_in_cents_after_coupon : 
  dollars_to_cents ((price_per_bag - coupon) / bag_weight) = 50 := by
  sorry

end cost_per_serving_in_cents_after_coupon_l247_247175


namespace probability_X_gt_4_l247_247614

noncomputable def normal_dist 
  (μ : ℝ) (σ2 : ℝ) (X : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, X x = exp (-(x - μ)^2 / (2 * σ2)) / sqrt (2 * π * σ2)

theorem probability_X_gt_4 
  (σ2 : ℝ) (P : set ℝ → ℝ)
  (h₀ : ∀ S, P S = ∫ x in S, exp (-(x - 2)^2 / (2 * σ2)) / sqrt (2 * π * σ2))
  (h₁ : P {x | 0 < x ∧ x < 4} = 0.8) :
  P {x | x > 4} = 0.1 :=
sorry

end probability_X_gt_4_l247_247614


namespace incircle_radius_of_right_triangle_l247_247503

noncomputable def radius_of_incircle (a b c : ℝ) : ℝ := (a + b - c) / 2

theorem incircle_radius_of_right_triangle
  (a : ℝ) (b_proj_hypotenuse : ℝ) (r : ℝ) :
  a = 15 ∧ b_proj_hypotenuse = 16 ∧ r = 5 :=
by
  sorry

end incircle_radius_of_right_triangle_l247_247503


namespace international_sales_correct_option_l247_247997

theorem international_sales_correct_option :
  (∃ (A B C D : String),
     A = "who" ∧
     B = "what" ∧
     C = "whoever" ∧
     D = "whatever" ∧
     (∃ x, x = C → "Could I speak to " ++ x ++ " is in charge of International Sales please?" = "Could I speak to whoever is in charge of International Sales please?")) :=
sorry

end international_sales_correct_option_l247_247997


namespace sentence_structure_diff_l247_247578

-- Definitions based on sentence structures.
def sentence_A := "得不焚，殆有神护者" -- passive
def sentence_B := "重为乡党所笑" -- passive
def sentence_C := "而文采不表于后也" -- post-positioned prepositional
def sentence_D := "是以见放" -- passive

-- Definition to check if the given sentence is passive
def is_passive (s : String) : Prop :=
  s = sentence_A ∨ s = sentence_B ∨ s = sentence_D

-- Definition to check if the given sentence is post-positioned prepositional
def is_post_positioned_prepositional (s : String) : Prop :=
  s = sentence_C

-- Theorem to prove
theorem sentence_structure_diff :
  (is_post_positioned_prepositional sentence_C) ∧ ¬(is_passive sentence_C) :=
by
  sorry

end sentence_structure_diff_l247_247578


namespace prime_and_n_eq_m_minus_1_l247_247647

theorem prime_and_n_eq_m_minus_1 (n m : ℕ) (h1 : n ≥ 2) (h2 : m ≥ 2)
  (h3 : ∀ k : ℕ, k ∈ Finset.range n.succ → k^n % m = 1) : Nat.Prime m ∧ n = m - 1 := 
sorry

end prime_and_n_eq_m_minus_1_l247_247647


namespace find_m_l247_247013

theorem find_m (m : ℝ) : (m - 2) * (0 : ℝ)^2 + 4 * (0 : ℝ) + 2 - |m| = 0 → m = -2 :=
by
  intros h
  sorry

end find_m_l247_247013


namespace condition_sufficient_but_not_necessary_condition_not_necessary_combined_condition_l247_247626

theorem condition_sufficient_but_not_necessary (x y : ℝ) :
  (x^2 + y^2 + 4*x + 3 ≤ 0) → ((x + 4) * (x + 3) ≥ 0) :=
sorry

theorem condition_not_necessary (x y : ℝ) :
  ((x + 4) * (x + 3) ≥ 0) → ¬ (x^2 + y^2 + 4*x + 3 ≤ 0) :=
sorry

-- Combine both into a single statement using conjunction
theorem combined_condition (x y : ℝ) :
  ((x^2 + y^2 + 4*x + 3 ≤ 0) → ((x + 4) * (x + 3) ≥ 0))
  ∧ ((x + 4) * (x + 3) ≥ 0 → ¬(x^2 + y^2 + 4*x + 3 ≤ 0)) :=
sorry

end condition_sufficient_but_not_necessary_condition_not_necessary_combined_condition_l247_247626


namespace total_painting_cost_l247_247964

variable (house_area : ℕ) (price_per_sqft : ℕ)

theorem total_painting_cost (h1 : house_area = 484) (h2 : price_per_sqft = 20) :
  house_area * price_per_sqft = 9680 :=
by
  sorry

end total_painting_cost_l247_247964


namespace greatest_multiple_of_5_and_6_less_than_1000_l247_247395

open Nat

theorem greatest_multiple_of_5_and_6_less_than_1000 :
  ∃ a, (a % 5 = 0) ∧ (a % 6 = 0) ∧ (a < 1000) ∧ (a = 990) :=
by
  use 990
  simp [Nat.mod_eq_zero_of_dvd, Nat.le_zero_iff]
  repeat {split}; try {exact Int.ofNat_zero}
  { sorry }

end greatest_multiple_of_5_and_6_less_than_1000_l247_247395


namespace frank_spends_more_l247_247676

def cost_computer_table : ℕ := 140
def cost_computer_chair : ℕ := 100
def cost_joystick : ℕ := 20
def frank_share_joystick : ℕ := cost_joystick / 4
def eman_share_joystick : ℕ := cost_joystick * 3 / 4

def total_spent_frank : ℕ := cost_computer_table + frank_share_joystick
def total_spent_eman : ℕ := cost_computer_chair + eman_share_joystick

theorem frank_spends_more : total_spent_frank - total_spent_eman = 30 :=
by
  sorry

end frank_spends_more_l247_247676


namespace tire_cost_l247_247445

theorem tire_cost (total_cost : ℝ) (num_tires : ℕ)
    (h1 : num_tires = 8) (h2 : total_cost = 4) : 
    total_cost / num_tires = 0.50 := 
by
  sorry

end tire_cost_l247_247445


namespace laura_has_435_dollars_l247_247209

-- Define the monetary values and relationships
def darwin_money := 45
def mia_money := 2 * darwin_money + 20
def combined_money := mia_money + darwin_money
def laura_money := 3 * combined_money - 30

-- The theorem to prove: Laura's money is $435
theorem laura_has_435_dollars : laura_money = 435 := by
  sorry

end laura_has_435_dollars_l247_247209


namespace optimal_washing_effect_l247_247141

noncomputable def optimal_laundry_addition (x y : ℝ) : Prop :=
  (5 + 0.02 * 2 + x + y = 20) ∧
  (0.02 * 2 + x = (20 - 5) * 0.004)

theorem optimal_washing_effect :
  ∃ x y : ℝ, optimal_laundry_addition x y ∧ x = 0.02 ∧ y = 14.94 :=
by
  sorry

end optimal_washing_effect_l247_247141


namespace num_nonnegative_real_values_l247_247157

theorem num_nonnegative_real_values :
  ∃ n : ℕ, ∀ x : ℝ, (x ≥ 0) → (∃ k : ℕ, (169 - (x^(1/3))) = k^2) → n = 27 := 
sorry

end num_nonnegative_real_values_l247_247157


namespace existence_of_point_N_l247_247458

-- Given conditions
def is_point_on_ellipse (x y a b : ℝ) : Prop := 
  (x^2 / a^2) + (y^2 / b^2) = 1

def is_ellipse (a b : ℝ) : Prop := 
  a > b ∧ b > 0 ∧ (a^2 = b^2 + (a * (Real.sqrt 2) / 2)^2)

def passes_through_point (x y a b : ℝ) (px py : ℝ) : Prop :=
  (px^2 / a^2) + (py^2 / b^2) = 1

def ellipse_with_eccentricity (a : ℝ) : Prop :=
  (Real.sqrt 2) / 2 = (Real.sqrt (a^2 - (a * (Real.sqrt 2) / 2)^2)) / a

def line_through_point (k : ℝ) (x y : ℝ) : Prop :=
  y = k * x + 1

def lines_intersect_ellipse (k a b : ℝ) : Prop :=
  ∃ x1 y1 x2 y2, line_through_point k x1 y1 ∧ line_through_point k x2 y2 ∧ is_point_on_ellipse x1 y1 a b ∧ is_point_on_ellipse x2 y2 a b

def angle_condition (k t a b : ℝ) : Prop :=
  ∃ x1 y1 x2 y2, line_through_point k x1 y1 ∧ line_through_point k x2 y2 ∧ is_point_on_ellipse x1 y1 a b ∧ is_point_on_ellipse x2 y2 a b ∧ 
  ((y1 - t) / x1) + ((y2 - t) / x2) = 0

-- Lean 4 statement
theorem existence_of_point_N (a b k t : ℝ) (hx : is_ellipse a b) (hp : passes_through_point 2 (Real.sqrt 2) a b 2 (Real.sqrt 2)) (he : ellipse_with_eccentricity a) (hl : ∀ (x1 y1 x2 y2 : ℝ), lines_intersect_ellipse k a b) :
  ∃ (N : ℝ), N = 4 ∧ angle_condition k N a b :=
sorry

end existence_of_point_N_l247_247458


namespace problem_l247_247495

variable (g : ℝ → ℝ)
variables (x y : ℝ)

noncomputable def cond1 : Prop := ∀ x y : ℝ, 0 < x → 0 < y → g (x^2 * y) = g x / y^2
noncomputable def cond2 : Prop := g 800 = 4

-- The statement to be proved
theorem problem (h1 : cond1 g) (h2 : cond2 g) : g 7200 = 4 / 81 :=
by
  sorry

end problem_l247_247495


namespace find_y_l247_247931

-- Let s be the result of tripling both the base and exponent of c^d
-- Given the condition s = c^d * y^d, we need to prove y = 27c^2

variable (c d y : ℝ)
variable (h_d : d > 0)
variable (h : (3 * c)^(3 * d) = c^d * y^d)

theorem find_y (h_d : d > 0) (h : (3 * c)^(3 * d) = c^d * y^d) : y = 27 * c ^ 2 :=
by sorry

end find_y_l247_247931


namespace third_side_length_not_12_l247_247193

theorem third_side_length_not_12 (x : ℕ) (h1 : x % 2 = 0) (h2 : 5 < x) (h3 : x < 11) : x ≠ 12 := 
sorry

end third_side_length_not_12_l247_247193


namespace system_is_inconsistent_l247_247659

def system_of_equations (x1 x2 x3 : ℝ) : Prop :=
  (x1 + 4*x2 + 10*x3 = 1) ∧
  (0*x1 - 5*x2 - 13*x3 = -1.25) ∧
  (0*x1 + 0*x2 + 0*x3 = 1.25)

theorem system_is_inconsistent : 
  ∀ x1 x2 x3, ¬ system_of_equations x1 x2 x3 :=
by
  intro x1 x2 x3
  sorry

end system_is_inconsistent_l247_247659


namespace count_divisors_of_54_greater_than_7_l247_247468

theorem count_divisors_of_54_greater_than_7 : ∃ (S : Finset ℕ), S.card = 4 ∧ ∀ n ∈ S, n ∣ 54 ∧ n > 7 :=
by
  -- proof goes here
  sorry

end count_divisors_of_54_greater_than_7_l247_247468


namespace find_a_l247_247039

theorem find_a 
  (a : ℝ) 
  (h : 1 - 2 * a = a - 2) 
  (h1 : 1 - 2 * a = a - 2) 
  : a = 1 := 
by 
  -- proof goes here
  sorry

end find_a_l247_247039


namespace factorize_a_cubed_minus_a_l247_247862

theorem factorize_a_cubed_minus_a (a : ℝ) : a^3 - a = a * (a + 1) * (a - 1) :=
by
  sorry

end factorize_a_cubed_minus_a_l247_247862


namespace a5_value_l247_247289

variable {a : ℕ → ℝ} (q : ℝ) (a2 a3 : ℝ)

-- Assume the conditions: geometric sequence, a_2 = 2, a_3 = -4
def is_geometric_sequence (a : ℕ → ℝ) : Prop := ∃ q, ∀ n, a (n + 1) = a n * q

-- Given conditions
axiom h1 : is_geometric_sequence a
axiom h2 : a 2 = 2
axiom h3 : a 3 = -4

-- Theorem to prove
theorem a5_value : a 5 = -16 :=
by
  -- Here you would provide the proof based on the conditions
  sorry

end a5_value_l247_247289


namespace min_value_3x_plus_4y_l247_247474

theorem min_value_3x_plus_4y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 3 * y = 5 * x * y) :
  3 * x + 4 * y ≥ 5 :=
sorry

end min_value_3x_plus_4y_l247_247474


namespace Z_real_iff_m_eq_neg3_or_5_Z_pure_imaginary_iff_m_eq_neg2_Z_in_fourth_quadrant_iff_neg2_lt_m_lt_5_l247_247615

open Complex

noncomputable def Z (m : ℝ) : ℂ :=
  (m ^ 2 + 5 * m + 6) + (m ^ 2 - 2 * m - 15) * Complex.I

namespace ComplexNumbersProofs

-- Prove that Z is a real number if and only if m = -3 or m = 5
theorem Z_real_iff_m_eq_neg3_or_5 (m : ℝ) :
  (Z m).im = 0 ↔ (m = -3 ∨ m = 5) := 
by
  sorry

-- Prove that Z is a pure imaginary number if and only if m = -2
theorem Z_pure_imaginary_iff_m_eq_neg2 (m : ℝ) :
  (Z m).re = 0 ↔ (m = -2) := 
by
  sorry

-- Prove that the point corresponding to Z lies in the fourth quadrant if and only if -2 < m < 5
theorem Z_in_fourth_quadrant_iff_neg2_lt_m_lt_5 (m : ℝ) :
  (Z m).re > 0 ∧ (Z m).im < 0 ↔ (-2 < m ∧ m < 5) :=
by
  sorry

end ComplexNumbersProofs

end Z_real_iff_m_eq_neg3_or_5_Z_pure_imaginary_iff_m_eq_neg2_Z_in_fourth_quadrant_iff_neg2_lt_m_lt_5_l247_247615


namespace percentage_reduction_is_10_percent_l247_247063

-- Definitions based on the given conditions
def rooms_rented_for_40 : ℕ := sorry
def rooms_rented_for_60 : ℕ := sorry
def total_rent : ℕ := 2000
def rent_per_room_40 : ℕ := 40
def rent_per_room_60 : ℕ := 60
def rooms_switch_count : ℕ := 10

-- Define the hypothetical new total if the rooms were rented at different rates
def new_total_rent : ℕ := (rent_per_room_40 * (rooms_rented_for_40 + rooms_switch_count)) + (rent_per_room_60 * (rooms_rented_for_60 - rooms_switch_count))

-- Calculate the percentage reduction
noncomputable def percentage_reduction : ℝ := (((total_rent: ℝ) - (new_total_rent: ℝ)) / (total_rent: ℝ)) * 100

-- Statement to prove
theorem percentage_reduction_is_10_percent : percentage_reduction = 10 := by
  sorry

end percentage_reduction_is_10_percent_l247_247063


namespace determine_function_l247_247752

theorem determine_function (f : ℝ → ℝ)
    (h1 : f 1 = 0)
    (h2 : ∀ x y : ℝ, |f x - f y| = |x - y|) :
    (∀ x : ℝ, f x = x - 1) ∨ (∀ x : ℝ, f x = 1 - x) := by
  sorry

end determine_function_l247_247752


namespace factorize_cubic_l247_247875

theorem factorize_cubic (a : ℝ) : a^3 - a = a * (a + 1) * (a - 1) :=
sorry

end factorize_cubic_l247_247875


namespace find_m_l247_247017

theorem find_m
  (h1 : ∃ (m : ℝ), ∃ (focus_parabola : ℝ × ℝ), focus_parabola = (0, 1/2)
       ∧ ∃ (focus_ellipse : ℝ × ℝ), focus_ellipse = (0, Real.sqrt (m - 2))
       ∧ focus_parabola = focus_ellipse) :
  ∃ (m : ℝ), m = 9/4 :=
by
  sorry

end find_m_l247_247017


namespace find_x_l247_247821

-- Definitions for the median and mean calculations
def mean (a b c d e : ℝ) : ℝ :=
  (a + b + c + d + e) / 5

def median (a b c d e : ℝ) : ℝ :=
  let sorted_list := List.sort [a, b, c, d, e]
  sorted_list.nthLe 2 sorry -- second index is the median in zero-indexed list

theorem find_x :
  ∃ x : ℝ, median 3 7 12 21 x = 9 + (mean 3 7 12 21 x) ^ (1/4) ∧ x = 362 :=
sorry

end find_x_l247_247821


namespace greatest_multiple_of_5_and_6_less_than_1000_l247_247366

theorem greatest_multiple_of_5_and_6_less_than_1000 : 
  ∃ n, n < 1000 ∧ n % 5 = 0 ∧ n % 6 = 0 ∧ ∀ m, m < 1000 ∧ m % 5 = 0 ∧ m % 6 = 0 → m ≤ n :=
exists.intro 990 (by {
  -- proof goes here
  sorry
})

end greatest_multiple_of_5_and_6_less_than_1000_l247_247366


namespace fourth_vertex_of_square_l247_247535

theorem fourth_vertex_of_square (A B C D : ℂ) : 
  A = (2 + 3 * I) ∧ B = (-3 + 2 * I) ∧ C = (-2 - 3 * I) →
  D = (0 - 0.5 * I) :=
sorry

end fourth_vertex_of_square_l247_247535


namespace parabola_directrix_l247_247352

theorem parabola_directrix :
  ∀ (p : ℝ), (y^2 = 6 * x) → (x = -3/2) :=
by
  sorry

end parabola_directrix_l247_247352


namespace decimal_equivalent_of_one_half_squared_l247_247107

theorem decimal_equivalent_of_one_half_squared : (1 / 2 : ℝ) ^ 2 = 0.25 := 
sorry

end decimal_equivalent_of_one_half_squared_l247_247107


namespace sum_of_angles_l247_247014

open Real

theorem sum_of_angles (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h1 : sin α = sqrt 5 / 5) (h2 : sin β = sqrt 10 / 10) : α + β = π / 4 :=
sorry

end sum_of_angles_l247_247014


namespace intersection_complement_A_B_l247_247757

def Universe : Set ℝ := Set.univ

def A : Set ℝ := {x | abs (x - 1) > 2}

def B : Set ℝ := {x | x^2 - 6*x + 8 < 0}

theorem intersection_complement_A_B :
  (Universe \ A) ∩ B = {x : ℝ | 2 < x ∧ x ≤ 3} :=
by
  sorry

end intersection_complement_A_B_l247_247757


namespace indefinite_integral_eq_l247_247410

theorem indefinite_integral_eq : 
  ∫ (x : ℝ) in -∞..∞, (2 * x^3 + 3 * x^2 + 3 * x + 2) / ((x^2 + x + 1) * (x^2 + 1)) = 
    (1 / 2 * log (|x^2 + x + 1|)) + (1 / (sqrt 3) * arctan ((2 * x + 1) / (sqrt 3))) + (1 / 2 * log (|x^2 + 1|)) + arctan x + C :=
sorry

end indefinite_integral_eq_l247_247410


namespace find_third_number_l247_247950

theorem find_third_number :
  let total_sum := 121526
  let first_addend := 88888
  let second_addend := 1111
  (total_sum = first_addend + second_addend + 31527) :=
by
  sorry

end find_third_number_l247_247950


namespace curve_crosses_itself_l247_247428

-- Definitions of the parametric equations
def x (t : ℝ) : ℝ := t^2 - 4
def y (t : ℝ) : ℝ := t^3 - 6*t + 3

-- The theorem statement
theorem curve_crosses_itself :
  ∃ t₁ t₂ : ℝ, t₁ ≠ t₂ ∧ x t₁ = x t₂ ∧ y t₁ = y t₂ ∧ (x t₁, y t₁) = (2, 3) :=
by
  -- Proof would go here
  sorry

end curve_crosses_itself_l247_247428


namespace min_value_3x_plus_4y_l247_247473

theorem min_value_3x_plus_4y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 3 * y = 5 * x * y) :
  3 * x + 4 * y ≥ 5 :=
sorry

end min_value_3x_plus_4y_l247_247473


namespace plane_ratio_l247_247257

section

variables (D B T P : ℕ)

-- Given conditions
axiom total_distance : D = 1800
axiom distance_by_bus : B = 720
axiom distance_by_train : T = (2 * B) / 3

-- Prove the ratio of the distance traveled by plane to the whole trip
theorem plane_ratio :
  D = 1800 →
  B = 720 →
  T = (2 * B) / 3 →
  P = D - (T + B) →
  P / D = 1 / 3 := by
  intros h1 h2 h3 h4
  sorry

end

end plane_ratio_l247_247257


namespace circles_are_intersecting_l247_247974

-- Define the circles and the distances given
def radius_O1 : ℝ := 3
def radius_O2 : ℝ := 5
def distance_O1O2 : ℝ := 2

-- Define the positional relationships
inductive PositionalRelationship
| externally_tangent
| intersecting
| internally_tangent
| contained_within_each_other

open PositionalRelationship

-- State the theorem to be proved
theorem circles_are_intersecting :
  distance_O1O2 > 0 ∧ distance_O1O2 < (radius_O1 + radius_O2) ∧ distance_O1O2 > abs (radius_O1 - radius_O2) →
  PositionalRelationship := 
by
  intro h
  exact PositionalRelationship.intersecting

end circles_are_intersecting_l247_247974


namespace min_abs_sum_l247_247051

noncomputable def abs (x : ℤ) : ℤ := Int.natAbs x

noncomputable def M (p q r s: ℤ) : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![p, q], ![r, s]]

theorem min_abs_sum (p q r s : ℤ)
  (h_nonzero : p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0 ∧ s ≠ 0)
  (h_matrix_square : (M p q r s) * (M p q r s) = ![![8, 0], ![0, 8]]) :
  abs p + abs q + abs r + abs s = 9 :=
  sorry

end min_abs_sum_l247_247051


namespace secondChapterPages_is_18_l247_247830

-- Define conditions as variables and constants
def thirdChapterPages : ℕ := 3
def additionalPages : ℕ := 15

-- The main statement to prove
theorem secondChapterPages_is_18 : (thirdChapterPages + additionalPages) = 18 := by
  -- Proof would go here, but we skip it with sorry
  sorry

end secondChapterPages_is_18_l247_247830


namespace greatest_multiple_l247_247372

theorem greatest_multiple (n : ℕ) (h1 : n < 1000) (h2 : n % 5 = 0) (h3 : n % 6 = 0) : n = 990 :=
sorry

end greatest_multiple_l247_247372


namespace find_PF2_l247_247336

open Real

noncomputable def hyperbola_equation (x y : ℝ) := (x^2 / 16) - (y^2 / 20) = 1

noncomputable def distance (P F : ℝ × ℝ) : ℝ := 
  let (px, py) := P
  let (fx, fy) := F
  sqrt ((px - fx)^2 + (py - fy)^2)

theorem find_PF2
  (P : ℝ × ℝ)
  (F1 F2 : ℝ × ℝ)
  (on_hyperbola : hyperbola_equation P.1 P.2)
  (foci_F1_F2 : F1 = (-6, 0) ∧ F2 = (6, 0))
  (distance_PF1 : distance P F1 = 9) : 
  distance P F2 = 17 := 
by
  sorry

end find_PF2_l247_247336


namespace nth_equation_l247_247343

theorem nth_equation (n : ℕ) (hn: n ≥ 1) : 
  (n+1) / ((n+1)^2 - 1) - 1 / (n * (n+1) * (n+2)) = 1 / (n+1) :=
by
  sorry

end nth_equation_l247_247343


namespace yoongi_initial_books_l247_247091

theorem yoongi_initial_books 
  (Y E U : ℕ)
  (h1 : Y - 5 + 15 = 45)
  (h2 : E + 5 - 10 = 45)
  (h3 : U - 15 + 10 = 45) : 
  Y = 35 := 
by 
  -- To be completed with proof
  sorry

end yoongi_initial_books_l247_247091


namespace expected_value_proof_l247_247952

noncomputable def expected_value_xi : ℚ :=
  let p_xi_2 : ℚ := 3/5
  let p_xi_3 : ℚ := 3/10
  let p_xi_4 : ℚ := 1/10
  2 * p_xi_2 + 3 * p_xi_3 + 4 * p_xi_4

theorem expected_value_proof :
  expected_value_xi = 5/2 :=
by
  sorry

end expected_value_proof_l247_247952


namespace find_cool_triple_x_eq_5_find_cool_triple_x_eq_7_two_distinct_cool_triples_for_odd_x_find_cool_triple_x_even_l247_247547

-- Define the nature of a "cool" triple.
def is_cool_triple (x y z : ℕ) : Prop :=
  x > 0 ∧ y > 1 ∧ z > 0 ∧ x^2 - 3 * y^2 = z^2 - 3

-- Part (a) i: For x = 5.
theorem find_cool_triple_x_eq_5 : ∃ (y z : ℕ), is_cool_triple 5 y z := sorry

-- Part (a) ii: For x = 7.
theorem find_cool_triple_x_eq_7 : ∃ (y z : ℕ), is_cool_triple 7 y z := sorry

-- Part (b): For every x ≥ 5 and odd, there are at least two distinct cool triples.
theorem two_distinct_cool_triples_for_odd_x (x : ℕ) (h1 : x ≥ 5) (h2 : x % 2 = 1) : 
  ∃ (y₁ z₁ y₂ z₂ : ℕ), is_cool_triple x y₁ z₁ ∧ is_cool_triple x y₂ z₂ ∧ (y₁, z₁) ≠ (y₂, z₂) := sorry

-- Part (c): Find a cool type triple with x even.
theorem find_cool_triple_x_even : ∃ (x y z : ℕ), x % 2 = 0 ∧ is_cool_triple x y z := sorry

end find_cool_triple_x_eq_5_find_cool_triple_x_eq_7_two_distinct_cool_triples_for_odd_x_find_cool_triple_x_even_l247_247547


namespace reflectionY_matrix_correct_l247_247732

-- Define the basis vectors e₁ and e₂
def e1 : Vector := ⟨1, 0⟩
def e2 : Vector := ⟨0, 1⟩

-- Define the transformation that reflects over the y-axis
def reflectY : Vector → Vector 
| ⟨x, y⟩ => ⟨-x, y⟩

-- Conditions given in the problem
lemma reflectY_e1 : reflectY e1 = ⟨-1, 0⟩ := sorry
lemma reflectY_e2 : reflectY e2 = ⟨0, 1⟩ := sorry

-- The goal is to find the transformation matrix for reflection over the y-axis
def reflectionMatrixY : Matrix 2 2 ℝ :=
  Matrix.of_vec ([-1, 0, 0, 1])

theorem reflectionY_matrix_correct :
  ∀ (v : Vector), reflectY v = reflectionMatrixY.mul_vec v := sorry

end reflectionY_matrix_correct_l247_247732


namespace range_of_g_area_of_triangle_ABC_l247_247750

-- Part (1)
theorem range_of_g :
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ π / 2 →
  0 ≤ (Real.sin (2 * x + π / 3) + Real.sqrt 3 / 2) ∧ (Real.sin (2 * x + π / 3) + Real.sqrt 3 / 2) ≤ (Real.sqrt 3 / 2 + 1) :=
by sorry

-- Part (2)
theorem area_of_triangle_ABC (a b c : ℝ) (A : ℝ) :
  A = π / 3 ∧ a = 4 ∧ (b + c = 5) ∧ Real.sin A = Real.sqrt 3 / 2 →
  let area := 1 / 2 * b * c * Real.sin A in
  area = 9 * Real.sqrt 3 / 4 :=
by sorry

end range_of_g_area_of_triangle_ABC_l247_247750


namespace factorial_comparison_l247_247550

open scoped BigOperators

theorem factorial_comparison : (100.factorial)!.factorial < (99.factorial) ^ (100.factorial) * (100.factorial) ^ (99.factorial) := by
  sorry

end factorial_comparison_l247_247550


namespace gcd_228_1995_l247_247816

-- Define the gcd function according to the Euclidean algorithm
def gcd (a b : ℕ) : ℕ :=
  if b = 0 then a
  else gcd b (a % b)

-- Prove that the gcd of 228 and 1995 is 57
theorem gcd_228_1995 : gcd 228 1995 = 57 :=
by
  sorry

end gcd_228_1995_l247_247816


namespace find_n_l247_247738

noncomputable def e : ℝ := Real.exp 1

-- lean cannot compute non-trivial transcendental solutions, this would need numerical methods
theorem find_n (n : ℝ) (x : ℝ) (y : ℝ) (h1 : x = 3) (h2 : y = 27) :
  Real.log n ^ (n / (2 * Real.sqrt (Real.pi + x))) = y :=
by
  rw [h1, h2]
  sorry

end find_n_l247_247738


namespace length_AB_l247_247132

open Real

noncomputable def parabola_focus : (ℝ × ℝ) := (1, 0)

theorem length_AB (x1 y1 x2 y2 : ℝ) 
  (hA : y1^2 = 4 * x1) (hB : y2^2 = 4 * x2) 
  (hLine: (y2 - y1) * 1 = (x2 - x1) *0)
  (hSum : x1 + x2 = 6) : 
  dist (x1, y1) (x2, y2) = 8 := 
sorry

end length_AB_l247_247132


namespace simplify_fraction_l247_247514

theorem simplify_fraction
  (a b c : ℝ)
  (h : 2 * a - 3 * c - 4 - b ≠ 0)
  : (6 * a ^ 2 - 2 * b ^ 2 + 6 * c ^ 2 + a * b - 13 * a * c - 4 * b * c - 18 * a - 5 * b + 17 * c + 12) /
    (4 * a ^ 2 - b ^ 2 + 9 * c ^ 2 - 12 * a * c - 16 * a + 24 * c + 16) =
    (3 * a - 2 * c - 3 + 2 * b) / (2 * a - 3 * c - 4 + b) :=
  sorry

end simplify_fraction_l247_247514


namespace solution_is_option_C_l247_247683

-- Define the equation.
def equation (x y : ℤ) : Prop := x - 2 * y = 3

-- Define the given conditions as terms in Lean.
def option_A := (1, 1)   -- (x = 1, y = 1)
def option_B := (-1, 1)  -- (x = -1, y = 1)
def option_C := (1, -1)  -- (x = 1, y = -1)
def option_D := (-1, -1) -- (x = -1, y = -1)

-- The goal is to prove that option C is a solution to the equation.
theorem solution_is_option_C : equation 1 (-1) :=
by {
  -- Proof will go here
  sorry
}

end solution_is_option_C_l247_247683


namespace range_of_x_l247_247806

theorem range_of_x (a : ℝ) (x : ℝ) (h_a : 1 ≤ a) : 
  ax^2 + (a - 3) * x + (a - 4) > 0 ↔ x < -1 ∨ x > 3 :=
by
  sorry

end range_of_x_l247_247806


namespace tomatoes_sold_to_mr_wilson_l247_247500

theorem tomatoes_sold_to_mr_wilson :
  let T := 245.5
  let S_m := 125.5
  let N := 42
  let S_w := T - S_m - N
  S_w = 78 := 
by
  sorry

end tomatoes_sold_to_mr_wilson_l247_247500


namespace range_of_a_l247_247007

def discriminant (a : ℝ) : ℝ := 4 * a^2 - 16
def P (a : ℝ) : Prop := discriminant a < 0
def Q (a : ℝ) : Prop := 5 - 2 * a > 1

theorem range_of_a (a : ℝ) (h1 : P a ∨ Q a) (h2 : ¬ (P a ∧ Q a)) : a ≤ -2 := by
  sorry

end range_of_a_l247_247007


namespace mod_pow_difference_l247_247268

theorem mod_pow_difference (a b n : ℕ) (h1 : a ≡ 47 [MOD n]) (h2 : b ≡ 22 [MOD n]) (h3 : n = 8) : (a ^ 2023 - b ^ 2023) % n = 1 :=
by
  sorry

end mod_pow_difference_l247_247268


namespace matching_pair_probability_l247_247363

theorem matching_pair_probability :
  let gray_socks := 12
  let white_socks := 10
  let black_socks := 6
  let total_socks := gray_socks + white_socks + black_socks
  let total_ways := total_socks.choose 2
  let gray_matching := gray_socks.choose 2
  let white_matching := white_socks.choose 2
  let black_matching := black_socks.choose 2
  let matching_ways := gray_matching + white_matching + black_matching
  let probability := matching_ways / total_ways
  probability = 1 / 3 :=
by sorry

end matching_pair_probability_l247_247363


namespace total_turtles_30_l247_247814

variable (Kristen_turtles : ℕ)
variable (Kris_turtles : ℕ)
variable (Trey_turtles : ℕ)
variable (total_turtles : ℕ)

def Kris_turtle_condition : Prop :=
  Kris_turtles = Kristen_turtles / 4

def Trey_turtle_condition : Prop :=
  Trey_turtles = 5 * Kris_turtles

def total_turtle_condition : Prop :=
  total_turtles = Kristen_turtles + Kris_turtles + Trey_turtles

theorem total_turtles_30
  (h1 : Kristen_turtles = 12)
  (h2 : Kris_turtle_condition)
  (h3 : Trey_turtle_condition)
  (h4 : total_turtle_condition) :
  total_turtles = 30 :=
by
  sorry

end total_turtles_30_l247_247814


namespace ratio_of_chicken_to_beef_l247_247044

theorem ratio_of_chicken_to_beef
  (beef_pounds : ℕ)
  (chicken_price_per_pound : ℕ)
  (total_cost : ℕ)
  (beef_price_per_pound : ℕ)
  (beef_cost : ℕ)
  (chicken_cost : ℕ)
  (chicken_pounds : ℕ) :
  beef_pounds = 1000 →
  beef_price_per_pound = 8 →
  total_cost = 14000 →
  beef_cost = beef_pounds * beef_price_per_pound →
  chicken_cost = total_cost - beef_cost →
  chicken_price_per_pound = 3 →
  chicken_pounds = chicken_cost / chicken_price_per_pound →
  chicken_pounds / beef_pounds = 2 :=
by
  intros
  sorry

end ratio_of_chicken_to_beef_l247_247044


namespace min_value_expression_l247_247594

theorem min_value_expression : ∃ (x y : ℝ), x^2 + 2*x*y + 3*y^2 - 6*x - 2*y = -11 := by
  sorry

end min_value_expression_l247_247594


namespace pow_mod_eq_residue_l247_247272

theorem pow_mod_eq_residue :
  (3 : ℤ)^(2048) % 11 = 5 :=
sorry

end pow_mod_eq_residue_l247_247272


namespace tangent_line_at_origin_l247_247353

noncomputable def curve (x : ℝ) : ℝ := x * Real.exp x + 2 * x - 1

def tangent_line (x₀ y₀ : ℝ) (k : ℝ) (x : ℝ) := y₀ + k * (x - x₀)

theorem tangent_line_at_origin : 
  tangent_line 0 (-1) 3 = λ x => 3 * x - 1 :=
by
  sorry

end tangent_line_at_origin_l247_247353


namespace trigonometric_identity_l247_247237

theorem trigonometric_identity : 
  let sin := Real.sin
  let cos := Real.cos
  sin 18 * cos 63 - sin 72 * sin 117 = - (Real.sqrt 2 / 2) :=
by
  -- The proof would go here
  sorry

end trigonometric_identity_l247_247237


namespace power_division_l247_247099

theorem power_division (a b : ℕ) (h₁ : 64 = 8^2) (h₂ : a = 15) (h₃ : b = 7) : 8^a / 64^b = 8 :=
by
  -- Equivalent to 8^15 / 64^7 = 8, given that 64 = 8^2
  sorry

end power_division_l247_247099


namespace percentage_exceeds_l247_247031

theorem percentage_exceeds (x y : ℝ) (h₁ : x < y) (h₂ : y = x + 0.35 * x) : ((y - x) / x) * 100 = 35 :=
by sorry

end percentage_exceeds_l247_247031


namespace find_d_l247_247770

theorem find_d (c : ℝ) (d : ℝ) (α : ℝ) (β : ℝ) (γ : ℝ) (ω : ℝ)  
  (h1 : α = c) 
  (h2 : β = 43)
  (h3 : γ = 59)
  (h4 : ω = d)
  (h5 : α + d + β + γ = 180) :
  d = 42 :=
by
  sorry

end find_d_l247_247770


namespace alice_profit_l247_247711

noncomputable def total_bracelets : ℕ := 52
noncomputable def cost_of_materials : ℝ := 3.00
noncomputable def bracelets_given_away : ℕ := 8
noncomputable def price_per_bracelet : ℝ := 0.25

theorem alice_profit :
  let bracelets_remaining := total_bracelets - bracelets_given_away;
      total_revenue := bracelets_remaining * price_per_bracelet;
      profit := total_revenue - cost_of_materials
  in profit = 8.00 := 
by 
  sorry

end alice_profit_l247_247711


namespace prime_factors_count_l247_247856

def number := 2310

theorem prime_factors_count : (nat.factors number).nodup.length = 5 := sorry

end prime_factors_count_l247_247856


namespace min_checkout_counters_l247_247714

variable (n : ℕ)
variable (x y : ℝ)

-- Conditions based on problem statement
axiom cond1 : 40 * y = 20 * x + n
axiom cond2 : 36 * y = 12 * x + n

theorem min_checkout_counters (m : ℕ) (h : 6 * m * y > 6 * x + n) : m ≥ 6 :=
  sorry

end min_checkout_counters_l247_247714


namespace eggs_left_in_jar_l247_247444

def eggs_after_removal (original removed : Nat) : Nat :=
  original - removed

theorem eggs_left_in_jar : eggs_after_removal 27 7 = 20 :=
by
  sorry

end eggs_left_in_jar_l247_247444


namespace remainder_of_s_minus_t_plus_t_minus_u_l247_247055

theorem remainder_of_s_minus_t_plus_t_minus_u (s t u : ℕ) (hs : s % 12 = 4) (ht : t % 12 = 5) (hu : u % 12 = 7) (h_order : s > t ∧ t > u) :
  ((s - t) + (t - u)) % 12 = 9 :=
by sorry

end remainder_of_s_minus_t_plus_t_minus_u_l247_247055


namespace ab_divides_a_squared_plus_b_squared_l247_247340

theorem ab_divides_a_squared_plus_b_squared (a b : ℕ) (hab : a ≠ 1 ∨ b ≠ 1) (hpos : 0 < a ∧ 0 < b) (hdiv : (ab - 1) ∣ (a^2 + b^2)) :
  a^2 + b^2 = 5 * a * b - 5 := 
by
  sorry

end ab_divides_a_squared_plus_b_squared_l247_247340


namespace rex_has_399_cards_left_l247_247210

def Nicole_cards := 700

def Cindy_cards := 3 * Nicole_cards + (40 / 100) * (3 * Nicole_cards)
def Tim_cards := (4 / 5) * Cindy_cards
def combined_total := Nicole_cards + Cindy_cards + Tim_cards
def Rex_and_Joe_cards := (60 / 100) * combined_total

def cards_per_person := Nat.floor (Rex_and_Joe_cards / 9)

theorem rex_has_399_cards_left : cards_per_person = 399 := by
  sorry

end rex_has_399_cards_left_l247_247210


namespace determine_a_l247_247273

theorem determine_a (a p q : ℚ) (h1 : p^2 = a) (h2 : 2 * p * q = 28) (h3 : q^2 = 9) : a = 196 / 9 :=
by
  sorry

end determine_a_l247_247273


namespace range_of_abs_2z_minus_1_l247_247163

open Complex

theorem range_of_abs_2z_minus_1
  (z : ℂ)
  (h : abs (z + 2 - I) = 1) :
  abs (2 * z - 1) ∈ Set.Icc (Real.sqrt 29 - 2) (Real.sqrt 29 + 2) :=
sorry

end range_of_abs_2z_minus_1_l247_247163


namespace intersection_M_N_l247_247755

def M : Set ℕ := { y | y < 6 }
def N : Set ℕ := {2, 3, 6}

theorem intersection_M_N : M ∩ N = {2, 3} := by
  sorry

end intersection_M_N_l247_247755


namespace intersection_of_A_and_B_union_of_A_and_B_complement_of_A_with_respect_to_U_l247_247201

open Set

universe u

def U : Set ℝ := univ

def A : Set ℝ := { x | x ≥ 1 ∨ x ≤ -3 }

def B : Set ℝ := { x | -4 < x ∧ x < 0 }

theorem intersection_of_A_and_B :
  A ∩ B = { x | -4 < x ∧ x ≤ -3 } :=
by sorry

theorem union_of_A_and_B :
  A ∪ B = { x | x < 0 ∨ x ≥ 1 } :=
by sorry

theorem complement_of_A_with_respect_to_U :
  U \ A = { x | -3 < x ∧ x < 1 } :=
by sorry

end intersection_of_A_and_B_union_of_A_and_B_complement_of_A_with_respect_to_U_l247_247201


namespace statement_I_l247_247075

section Problem
variable (g : ℝ → ℝ)

-- Conditions
def cond1 : Prop := ∀ x : ℝ, g x > 0
def cond2 : Prop := ∀ a b : ℝ, g a * g b = g (a + 2 * b)

-- Statement I to be proved
theorem statement_I (h1 : cond1 g) (h2 : cond2 g) : g 0 = 1 :=
by
  -- Proof is omitted
  sorry
end Problem

end statement_I_l247_247075


namespace factorize_a_cubed_minus_a_l247_247868

theorem factorize_a_cubed_minus_a (a : ℝ) : a^3 - a = a * (a + 1) * (a - 1) :=
sorry

end factorize_a_cubed_minus_a_l247_247868


namespace perpendicular_plane_line_sum_l247_247896

theorem perpendicular_plane_line_sum (x y : ℝ)
  (h1 : ∃ k : ℝ, (2, -4 * x, 1) = (6 * k, 12 * k, -3 * k * y))
  : x + y = -2 :=
sorry

end perpendicular_plane_line_sum_l247_247896


namespace total_cost_proof_l247_247662

def F : ℝ := 20.50
def R : ℝ := 61.50
def M : ℝ := 1476

def total_cost (mangos : ℝ) (rice : ℝ) (flour : ℝ) : ℝ :=
  (M * mangos) + (R * rice) + (F * flour)

theorem total_cost_proof:
  total_cost 4 3 5 = 6191 := by
  sorry

end total_cost_proof_l247_247662


namespace greatest_multiple_of_5_and_6_under_1000_l247_247375

theorem greatest_multiple_of_5_and_6_under_1000 : 
  ∃ x, (x % 5 = 0 ∧ x % 6 = 0 ∧ x < 1000) ∧ 
  (∀ y, (y % 5 = 0 ∧ y % 6 = 0 ∧ y < 1000) → y ≤ x) ∧ 
  x = 990 :=
begin
  sorry
end

end greatest_multiple_of_5_and_6_under_1000_l247_247375


namespace square_land_perimeter_l247_247118

theorem square_land_perimeter (a p : ℝ) (h1 : a = p^2 / 16) (h2 : 5*a = 10*p + 45) : p = 36 :=
by sorry

end square_land_perimeter_l247_247118


namespace gcd_polynomial_example_l247_247461

theorem gcd_polynomial_example (b : ℕ) (h : ∃ k : ℕ, b = 2 * 7784 * k) : 
  gcd (5 * b ^ 2 + 68 * b + 143) (3 * b + 14) = 25 :=
by 
  sorry

end gcd_polynomial_example_l247_247461


namespace frank_spend_more_l247_247673

noncomputable def table_cost : ℝ := 140
noncomputable def chair_cost : ℝ := 100
noncomputable def joystick_cost : ℝ := 20
noncomputable def frank_joystick : ℝ := joystick_cost * (1 / 4)
noncomputable def eman_joystick : ℝ := joystick_cost - frank_joystick
noncomputable def frank_total : ℝ := table_cost + frank_joystick
noncomputable def eman_total : ℝ := chair_cost + eman_joystick

theorem frank_spend_more :
  frank_total - eman_total = 30 :=
  sorry

end frank_spend_more_l247_247673


namespace inequality_abc_l247_247054

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (1 / (a * (1 + b))) + (1 / (b * (1 + c))) + (1 / (c * (1 + a))) ≥ 3 / (1 + a * b * c) :=
by 
  sorry

end inequality_abc_l247_247054


namespace power_division_l247_247100

theorem power_division (a b : ℕ) (h₁ : 64 = 8^2) (h₂ : a = 15) (h₃ : b = 7) : 8^a / 64^b = 8 :=
by
  -- Equivalent to 8^15 / 64^7 = 8, given that 64 = 8^2
  sorry

end power_division_l247_247100


namespace f_max_iff_l247_247982

noncomputable def f : ℚ → ℝ := sorry

axiom f_zero : f 0 = 0
axiom f_pos (a : ℚ) (h : a ≠ 0) : f a > 0
axiom f_mul (a b : ℚ) : f (a * b) = f a * f b
axiom f_add_le (a b : ℚ) : f (a + b) ≤ f a + f b
axiom f_bound (m : ℤ) : f m ≤ 1989

theorem f_max_iff (a b : ℚ) (h : f a ≠ f b) : f (a + b) = max (f a) (f b) := 
sorry

end f_max_iff_l247_247982


namespace researcher_can_cross_desert_l247_247136

structure Condition :=
  (distance_to_oasis : ℕ)  -- total distance to be covered
  (travel_per_day : ℕ)     -- distance covered per day
  (carry_capacity : ℕ)     -- maximum days of supplies they can carry
  (ensure_return : Bool)   -- flag to ensure porters can return
  (cannot_store_food : Bool) -- flag indicating no food storage in desert

def condition_instance : Condition :=
{ distance_to_oasis := 380,
  travel_per_day := 60,
  carry_capacity := 4,
  ensure_return := true,
  cannot_store_food := true }

theorem researcher_can_cross_desert (cond : Condition) : cond.distance_to_oasis = 380 
  ∧ cond.travel_per_day = 60 
  ∧ cond.carry_capacity = 4 
  ∧ cond.ensure_return = true 
  ∧ cond.cannot_store_food = true 
  → true := 
by 
  sorry

end researcher_can_cross_desert_l247_247136


namespace zachary_more_pushups_l247_247407

def zachary_pushups : ℕ := 51
def david_pushups : ℕ := 44

theorem zachary_more_pushups : zachary_pushups - david_pushups = 7 := by
  sorry

end zachary_more_pushups_l247_247407


namespace slope_correct_l247_247722

-- Coordinates of the vertices of the polygon
def vertex_A := (0, 0)
def vertex_B := (0, 4)
def vertex_C := (4, 4)
def vertex_D := (4, 2)
def vertex_E := (6, 2)
def vertex_F := (6, 0)

-- Define the total area of the polygon
def total_area : ℝ := 20

-- Define the slope of the line through the origin dividing the area in half
def slope_line_dividing_area (slope : ℝ) : Prop :=
  ∃ l : ℝ, l = 5 / 3 ∧
  ∃ area_divided : ℝ, area_divided = total_area / 2

-- Prove the slope is 5/3
theorem slope_correct :
  slope_line_dividing_area (5 / 3) :=
by
  sorry

end slope_correct_l247_247722


namespace smallest_n_l247_247046

open Real

def a : ℕ → ℝ
| 0       := 1 / 5
| 1       := 1 / 5
| (n + 2) := (a (n + 1) + a n) / (1 + a (n + 1) * a n)

theorem smallest_n (n : ℕ) : a n > 1 - 5^(-2022) ↔ n = 21 :=
sorry

end smallest_n_l247_247046


namespace symmetric_point_correct_l247_247003

def point : Type := ℝ × ℝ × ℝ

def symmetric_with_respect_to_y_axis (A : point) : point :=
  let (x, y, z) := A
  (-x, y, z)

def A : point := (-4, 8, 6)

theorem symmetric_point_correct :
  symmetric_with_respect_to_y_axis A = (4, 8, 6) := by
  sorry

end symmetric_point_correct_l247_247003


namespace find_f_prime_at_2_l247_247769

noncomputable def f (f' : ℝ → ℝ) (x : ℝ) : ℝ :=
  x^2 + 2 * x * f' 2 - Real.log x

theorem find_f_prime_at_2 (f' : ℝ → ℝ) (h : ∀ x, deriv (f f') x = f' x) :
  f' 2 = -7 / 2 :=
by
  have H := h 2
  sorry

end find_f_prime_at_2_l247_247769


namespace ratio_d_s_l247_247323

theorem ratio_d_s (s d : ℝ) 
  (h : (25 * 25 * s^2) / (25 * s + 50 * d)^2 = 0.81) :
  d / s = 1 / 18 :=
by
  sorry

end ratio_d_s_l247_247323


namespace rectangle_area_l247_247354

-- Definitions:
variables (l w : ℝ)

-- Conditions:
def condition1 : Prop := l = 4 * w
def condition2 : Prop := 2 * l + 2 * w = 200

-- Theorem statement:
theorem rectangle_area (h1 : condition1 l w) (h2 : condition2 l w) : l * w = 1600 :=
sorry

end rectangle_area_l247_247354


namespace find_triangle_CAN_angles_l247_247983

open Real

variables {A B C D P Q N : Type}
variables [square A B C D] 
variables (angle_CAP : angle A C P = 15)
variables (angle_BCP : angle B C P = 15)
variables (APCQ_isosceles_trap : isosceles_trapezoid A P C Q ∧ parallel P C A Q ∧ equal_length A P C Q)
variables (N : midpoint P Q)

theorem find_triangle_CAN_angles :
  ∠ C A N = 15 ∧ ∠ A N C = 90 ∧ ∠ N C A = 75 :=
sorry

end find_triangle_CAN_angles_l247_247983


namespace sin_identity_l247_247612

open Real

noncomputable def alpha : ℝ := π  -- since we are considering angles in radians

theorem sin_identity (h1 : sin α = 3/5) (h2 : π/2 < α ∧ α < 3 * π / 2) :
  sin (5 * π / 2 - α) = -4 / 5 :=
by sorry

end sin_identity_l247_247612


namespace sin_double_angle_value_l247_247318

theorem sin_double_angle_value 
  (α : ℝ) 
  (hα1 : π / 2 < α) 
  (hα2 : α < π)
  (h : 3 * Real.cos (2 * α) = Real.cos (π / 4 + α)) : 
  Real.sin (2 * α) = - 17 / 18 := 
by
  sorry

end sin_double_angle_value_l247_247318


namespace find_x_given_total_area_l247_247146

theorem find_x_given_total_area :
  ∃ x : ℝ, (16 * x^2 + 36 * x^2 + 6 * x^2 + 3 * x^2 = 1100) ∧ (x = Real.sqrt (1100 / 61)) :=
sorry

end find_x_given_total_area_l247_247146


namespace quadratic_complete_square_l247_247818

theorem quadratic_complete_square :
  ∀ x : ℝ, (x^2 + 2 * x + 3) = ((x + 1)^2 + 2) :=
by
  intro x
  sorry

end quadratic_complete_square_l247_247818


namespace sum_of_squares_l247_247083

theorem sum_of_squares (x y : ℝ) (h1 : x * y = 120) (h2 : x + y = 23) : x^2 + y^2 = 289 :=
by
  sorry

end sum_of_squares_l247_247083


namespace arithmetic_seq_sum_2013_l247_247917

noncomputable def a1 : ℤ := -2013
noncomputable def S (n d : ℤ) : ℤ := n * (2 * a1 + (n - 1) * d) / 2

theorem arithmetic_seq_sum_2013 :
  ∃ d : ℤ, (S 12 d / 12 - S 10 d / 10 = 2) → S 2013 d = -2013 :=
by
  sorry

end arithmetic_seq_sum_2013_l247_247917


namespace fraction_part_of_twenty_five_l247_247561

open Nat

def eighty_percent (x : ℕ) : ℕ := (85 * x) / 100

theorem fraction_part_of_twenty_five (x y : ℕ) (h1 : eighty_percent 40 = 34) (h2 : 34 - y = 14) (h3 : y = (4 * 25) / 5) : y = 20 :=
by 
  -- Given h1: eighty_percent 40 = 34
  -- And h2: 34 - y = 14
  -- And h3: y = (4 * 25) / 5
  -- Show y = 20
  sorry

end fraction_part_of_twenty_five_l247_247561


namespace brick_fence_depth_l247_247996

theorem brick_fence_depth (length height total_bricks : ℕ) 
    (h1 : length = 20) 
    (h2 : height = 5) 
    (h3 : total_bricks = 800) : 
    (total_bricks / (4 * length * height) = 2) := 
by
  sorry

end brick_fence_depth_l247_247996


namespace area_of_region_l247_247790

def region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (abs p.1 - p.1)^2 + (abs p.2 - p.2)^2 ≤ 16 ∧ 2 * p.2 + p.1 ≤ 0}

noncomputable def area : ℝ := sorry

theorem area_of_region : area = 5 + Real.pi := by
  sorry

end area_of_region_l247_247790


namespace expected_value_ball_draw_l247_247953

noncomputable def E_xi : ℚ :=
  let prob_xi_2 := 3/5
  let prob_xi_3 := 3/10
  let prob_xi_4 := 1/10
  2 * prob_xi_2 + 3 * prob_xi_3 + 4 * prob_xi_4

theorem expected_value_ball_draw : E_xi = 5 / 2 := by
  sorry

end expected_value_ball_draw_l247_247953


namespace find_point_Q_l247_247526

theorem find_point_Q {a b c : ℝ} 
  (h1 : ∀ x y z : ℝ, (x + 1)^2 + (y - 3)^2 + (z + 2)^2 = (x - a)^2 + (y - b)^2 + (z - c)^2) 
  (h2 : ∀ x y z: ℝ, 8 * x - 6 * y + 12 * z = 34) : 
  (a = 3) ∧ (b = -6) ∧ (c = 8) :=
by
  sorry

end find_point_Q_l247_247526


namespace gcd_228_1995_l247_247817

theorem gcd_228_1995 : Nat.gcd 228 1995 = 57 := by
  sorry

end gcd_228_1995_l247_247817


namespace compare_log_exp_l247_247287

theorem compare_log_exp (x y z : ℝ) 
  (hx : x = Real.log 2 / Real.log 5) 
  (hy : y = Real.log 2) 
  (hz : z = Real.sqrt 2) : 
  x < y ∧ y < z := 
sorry

end compare_log_exp_l247_247287


namespace circle_ring_ratio_l247_247562

theorem circle_ring_ratio
  (r R c d : ℝ)
  (hr : 0 < r)
  (hR : 0 < R)
  (hc : 0 < c)
  (hd : 0 < d)
  (h_areas : π * R^2 = (c / d) * (π * R^2 - π * r^2)) :
  R / r = (Real.sqrt c) / (Real.sqrt (d - c)) := 
by 
  sorry

end circle_ring_ratio_l247_247562


namespace evaluate_expression_l247_247724

theorem evaluate_expression (a b c : ℚ) (ha : a = 1/2) (hb : b = 1/4) (hc : c = 5) :
  a^2 * b^3 * c = 5 / 256 :=
by
  rw [ha, hb, hc]
  norm_num

end evaluate_expression_l247_247724


namespace sum_seven_terms_l247_247607

-- Definitions and conditions
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (d : ℝ), ∀ n : ℕ, a (n + 1) - a n = d

def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = (n : ℝ) * (a 1 + a n) / 2

-- Given condition
def given_condition (a : ℕ → ℝ) : Prop :=
  a 2 + a 4 + a 6 = 42

-- Proof statement
theorem sum_seven_terms (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : sum_of_first_n_terms a S) 
  (h_cond : given_condition a) : 
  S 7 = 98 := 
sorry

end sum_seven_terms_l247_247607


namespace at_least_one_woman_selected_l247_247627

noncomputable def probability_at_least_one_woman_selected (men women : ℕ) (total_selected : ℕ) : ℚ :=
  let total_people := men + women
  let prob_no_woman := (men / total_people) * ((men - 1) / (total_people - 1)) * ((men - 2) / (total_people - 2))
  1 - prob_no_woman

theorem at_least_one_woman_selected (men women : ℕ) (total_selected : ℕ) :
  men = 5 → women = 5 → total_selected = 3 → 
  probability_at_least_one_woman_selected men women total_selected = 11 / 12 := by
  intros hmen hwomen hselected
  rw [hmen, hwomen, hselected]
  unfold probability_at_least_one_woman_selected
  sorry

end at_least_one_woman_selected_l247_247627


namespace bobs_total_profit_l247_247435

-- Definitions of the conditions
def cost_per_dog : ℕ := 250
def num_dogs : ℕ := 2
def num_puppies : ℕ := 6
def selling_price_per_puppy : ℕ := 350

-- Definition of the problem statement
theorem bobs_total_profit : (num_puppies * selling_price_per_puppy) - (num_dogs * cost_per_dog) = 1600 := 
by
  sorry

end bobs_total_profit_l247_247435


namespace number_of_integers_l247_247176

theorem number_of_integers (n : ℕ) (h₁ : 300 < n^2) (h₂ : n^2 < 1200) : ∃ k, k = 17 :=
by
  sorry

end number_of_integers_l247_247176


namespace smallest_nine_l247_247736

def satisfies_conditions (n : ℕ) (x : Fin n → ℕ) : Prop :=
  (∀ i, 1 ≤ x i ∧ x i ≤ n) ∧
  (Finset.univ.sum (λ i => x i) = n * (n + 1) / 2) ∧
  (Finset.univ.prod (λ i => x i) = Nat.factorial n) ∧
  (Multiset.of_finset (Finset.univ.image x) ≠ Multiset.range (n + 1))

def has_solution (n : ℕ) : Prop :=
  ∃ x : Fin n → ℕ, satisfies_conditions n x

theorem smallest_nine : ∃ n, has_solution n ∧ (∀ m, m < n → ¬ has_solution m) :=
  sorry

end smallest_nine_l247_247736


namespace ab_sum_pow_eq_neg_one_l247_247001

theorem ab_sum_pow_eq_neg_one (a b : ℝ) (h : |a - 3| + (b + 4)^2 = 0) : (a + b) ^ 2003 = -1 := 
by
  sorry

end ab_sum_pow_eq_neg_one_l247_247001


namespace greatest_multiple_l247_247370

theorem greatest_multiple (n : ℕ) (h1 : n < 1000) (h2 : n % 5 = 0) (h3 : n % 6 = 0) : n = 990 :=
sorry

end greatest_multiple_l247_247370


namespace mary_bought_48_cards_l247_247341

variable (M T F C B : ℕ)

theorem mary_bought_48_cards
  (h1 : M = 18)
  (h2 : T = 8)
  (h3 : F = 26)
  (h4 : C = 84) :
  B = C - (M - T + F) :=
by
  -- Proof would go here
  sorry

end mary_bought_48_cards_l247_247341


namespace power_division_l247_247096

theorem power_division (a b : ℕ) (h : 64 = 8^2) : (8 ^ 15) / (64 ^ 7) = 8 := by
  sorry

end power_division_l247_247096


namespace perfect_square_trinomial_m_l247_247907

theorem perfect_square_trinomial_m (m : ℤ) :
  (∃ a b : ℤ, (b^2 = 25) ∧ (a + b)^2 = x^2 - (m - 3) * x + 25) → (m = 13 ∨ m = -7) :=
by
  sorry

end perfect_square_trinomial_m_l247_247907


namespace part_a_part_b_part_c_l247_247825

def Q (x : ℝ) (n : ℕ) : ℝ := (x + 1) ^ n - x ^ n - 1
def P (x : ℝ) : ℝ := x ^ 2 + x + 1

theorem part_a (n : ℕ) : 
  (∃ k : ℤ, n = 6 * k + 1 ∨ n = 6 * k - 1) ↔ (∀ x : ℝ, P x ∣ Q x n) := sorry

theorem part_b (n : ℕ) : 
  (∃ k : ℤ, n = 6 * k + 1) ↔ (∀ x : ℝ, (P x)^2 ∣ Q x n) := sorry

theorem part_c (n : ℕ) : 
  n = 1 ↔ (∀ x : ℝ, (P x)^3 ∣ Q x n) := sorry

end part_a_part_b_part_c_l247_247825


namespace find_unknown_number_l247_247958

theorem find_unknown_number (y : ℝ) (h : 25 / y = 80 / 100) : y = 31.25 :=
sorry

end find_unknown_number_l247_247958


namespace num_positive_integers_between_300_and_1200_count_positive_integers_between_300_and_1200_l247_247177

/-- The number of positive integers n such that 300 < n^2 < 1200 is 17. -/
theorem num_positive_integers_between_300_and_1200 (n : ℕ) :
  (300 < n^2 ∧ n^2 < 1200) ↔ n ∈ {18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34} :=
by {
  sorry
}

/-- There are 17 positive integers n such that 300 < n^2 < 1200. -/
theorem count_positive_integers_between_300_and_1200 :
  fintype.card {n : ℕ // 300 < n^2 ∧ n^2 < 1200} = 17 :=
by {
  sorry
}

end num_positive_integers_between_300_and_1200_count_positive_integers_between_300_and_1200_l247_247177


namespace num_distinct_prime_factors_2310_l247_247854

def is_prime (p : ℕ) : Prop :=
  2 ≤ p ∧ ∀ (n : ℕ), n ∣ p → n = 1 ∨ n = p

def distinct_prime_factors (n : ℕ) (s : Set ℕ) : Prop :=
  (∀ p, p ∈ s → is_prime p) ∧ (∀ p, is_prime p → p ∣ n → p ∈ s) ∧ (∀ p₁ p₂, p₁ ∈ s → p₂ ∈ s → p₁ ≠ p₂)

theorem num_distinct_prime_factors_2310 : 
  ∃ s : Set ℕ, distinct_prime_factors 2310 s ∧ s.card = 5 :=
sorry

end num_distinct_prime_factors_2310_l247_247854


namespace stream_current_rate_l247_247834

theorem stream_current_rate (r c : ℝ) (h1 : 20 / (r + c) + 6 = 20 / (r - c)) (h2 : 20 / (3 * r + c) + 1.5 = 20 / (3 * r - c)) 
  : c = 3 :=
  sorry

end stream_current_rate_l247_247834


namespace range_of_m_l247_247171

theorem range_of_m (a b m : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0)
  (h_eq : a * b = a + b + 3) (h_ineq : a * b ≥ m) : m ≤ 9 :=
sorry

end range_of_m_l247_247171


namespace ceil_square_range_count_l247_247306

theorem ceil_square_range_count (x : ℝ) (h : ⌈x⌉ = 12) : 
  ∃ n : ℕ, n = 23 ∧ (∀ y : ℝ, 11 < y ∧ y ≤ 12 → ⌈y^2⌉ = n) := 
sorry

end ceil_square_range_count_l247_247306


namespace min_value_of_ab_l247_247312

theorem min_value_of_ab {a b : ℝ} (ha : a > 0) (hb : b > 0) 
  (h_eq : ∀ (x y : ℝ), (x / a + y / b = 1) → (x^2 + y^2 = 1)) : a * b = 2 :=
by sorry

end min_value_of_ab_l247_247312


namespace ratio_of_colored_sheets_l247_247361

theorem ratio_of_colored_sheets
    (total_sheets : ℕ)
    (num_binders : ℕ)
    (sheets_colored_by_justine : ℕ)
    (sheets_per_binder : ℕ)
    (h1 : total_sheets = 2450)
    (h2 : num_binders = 5)
    (h3 : sheets_colored_by_justine = 245)
    (h4 : sheets_per_binder = total_sheets / num_binders) :
    (sheets_colored_by_justine / Nat.gcd sheets_colored_by_justine sheets_per_binder) /
    (sheets_per_binder / Nat.gcd sheets_colored_by_justine sheets_per_binder) = 1 / 2 := by
  sorry

end ratio_of_colored_sheets_l247_247361


namespace krishan_money_l247_247121

/-- Given that the ratio of money between Ram and Gopal is 7:17, the ratio of money between Gopal and Krishan is 7:17, and Ram has Rs. 588, prove that Krishan has Rs. 12,065. -/
theorem krishan_money (R G K : ℝ) (h1 : R / G = 7 / 17) (h2 : G / K = 7 / 17) (h3 : R = 588) : K = 12065 :=
by
  sorry

end krishan_money_l247_247121


namespace counterexample_exists_l247_247985

theorem counterexample_exists : 
  ∃ (m : ℤ), (∃ (k1 : ℤ), m = 2 * k1) ∧ ¬(∃ (k2 : ℤ), m = 4 * k2) := 
sorry

end counterexample_exists_l247_247985


namespace triangle_is_isosceles_l247_247775

theorem triangle_is_isosceles (a b c : ℝ) (A B C : ℝ) 
  (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) 
  (hABC_sum : A + B + C = π) 
  (cos_rule : a * Real.cos B + b * Real.cos A = a) :
  a = c :=
by
  sorry

end triangle_is_isosceles_l247_247775


namespace no_3_digit_numbers_sum_27_even_l247_247467

-- Define the conditions
def is_digit_sum_27 (n : ℕ) : Prop :=
  (n ≥ 100 ∧ n < 1000) ∧ ((n / 100) + (n / 10 % 10) + (n % 10) = 27)

def is_even (n : ℕ) : Prop :=
  n % 2 = 0

-- Define the theorem
theorem no_3_digit_numbers_sum_27_even :
  ¬ ∃ n : ℕ, is_digit_sum_27 n ∧ is_even n :=
by
  sorry

end no_3_digit_numbers_sum_27_even_l247_247467


namespace sum_less_than_addends_then_both_negative_l247_247405

theorem sum_less_than_addends_then_both_negative {a b : ℝ} (h : a + b < a ∧ a + b < b) : a < 0 ∧ b < 0 := 
sorry

end sum_less_than_addends_then_both_negative_l247_247405


namespace coefficient_x3_y7_expansion_l247_247239

theorem coefficient_x3_y7_expansion : 
  let n := 10
  let a := (2 : ℚ) / 3
  let b := -(3 : ℚ) / 5
  let k := 3
  let binom := Nat.choose n k
  let term := binom * (a ^ k) * (b ^ (n - k))
  term = -(256 : ℚ) / 257 := 
by
  -- Proof omitted
  sorry

end coefficient_x3_y7_expansion_l247_247239


namespace prime_factors_2310_l247_247851

-- Define the number in question
def number : ℕ := 2310

-- The main theorem statement
theorem prime_factors_2310 : ∃ s : Finset ℕ, (∀ p ∈ s, Nat.Prime p) ∧ 
Finset.prod s id = number ∧ Finset.card s = 5 := by
  sorry

end prime_factors_2310_l247_247851


namespace circular_board_area_l247_247832

theorem circular_board_area (C : ℝ) (R T : ℝ) (h1 : R = 62.8) (h2 : T = 10) (h3 : C = R / T) (h4 : C = 2 * Real.pi) : 
  ∀ r A : ℝ, (r = C / (2 * Real.pi)) → (A = Real.pi * r^2)  → A = Real.pi :=
by
  intro r A
  intro hr hA
  sorry

end circular_board_area_l247_247832


namespace onion_harvest_weight_l247_247536

theorem onion_harvest_weight :
  let bags_per_trip := 10 in
  let weight_per_bag := 50 in
  let trips := 20 in
  let total_weight := (bags_per_trip * weight_per_bag) * trips in
  total_weight = 10000 := by
  sorry

end onion_harvest_weight_l247_247536


namespace number_of_large_balls_l247_247539

def smallBallRubberBands : ℕ := 50
def largeBallRubberBands : ℕ := 300
def totalRubberBands : ℕ := 5000
def smallBallsMade : ℕ := 22

def rubberBandsUsedForSmallBalls := smallBallsMade * smallBallRubberBands
def remainingRubberBands := totalRubberBands - rubberBandsUsedForSmallBalls

theorem number_of_large_balls :
  (remainingRubberBands / largeBallRubberBands) = 13 := by
  sorry

end number_of_large_balls_l247_247539


namespace linear_function_intersects_x_axis_at_two_units_l247_247805

theorem linear_function_intersects_x_axis_at_two_units (k : ℝ) :
  (∃ x : ℝ, y = k * x + 2 ∧ y = 0 ∧ |x| = 2) ↔ k = 1 ∨ k = -1 :=
by
  sorry

end linear_function_intersects_x_axis_at_two_units_l247_247805


namespace number_of_diagonals_of_nonagon_l247_247595

theorem number_of_diagonals_of_nonagon:
  (9 * (9 - 3)) / 2 = 27 := by
  sorry

end number_of_diagonals_of_nonagon_l247_247595


namespace distinct_prime_factors_2310_l247_247858

theorem distinct_prime_factors_2310 : 
  ∃ (p1 p2 p3 p4 p5 : ℕ), p1 = 2 ∧ p2 = 3 ∧ p3 = 5 ∧ p4 = 7 ∧ p5 = 11 ∧ 
    Nat.Prime p1 ∧ Nat.Prime p2 ∧ Nat.Prime p3 ∧ Nat.Prime p4 ∧ Nat.Prime p5 ∧ 
    (p1 * p2 * p3 * p4 * p5 = 2310) :=
by
  sorry

end distinct_prime_factors_2310_l247_247858


namespace equation_of_parallel_line_l247_247604

theorem equation_of_parallel_line 
  (l : ℝ → ℝ) 
  (passes_through : l 0 = 7) 
  (parallel_to : ∀ x : ℝ, l x = -4 * x + (l 0)) :
  ∀ x : ℝ, l x = -4 * x + 7 :=
by
  sorry

end equation_of_parallel_line_l247_247604


namespace find_other_number_l247_247249

open Nat

theorem find_other_number (A B lcm hcf : ℕ) (h_lcm : lcm = 2310) (h_hcf : hcf = 30) (h_A : A = 231) (h_eq : lcm * hcf = A * B) : 
  B = 300 :=
  sorry

end find_other_number_l247_247249


namespace find_a_3_l247_247608

noncomputable def a_n (n : ℕ) : ℤ := 2 + (n - 1)  -- Definition of the arithmetic sequence

theorem find_a_3 (d : ℤ) (a : ℕ → ℤ) 
  (h1 : a 1 = 2)
  (h2 : a 5 + a 7 = 2 * a 4 + 4) : a 3 = 4 :=
by 
  sorry

end find_a_3_l247_247608


namespace maximize_Miraflores_win_l247_247534

-- Definitions based on given conditions
def voters_count (n : ℕ) : ℕ := 2 * n
def support_Miraflores (n : ℕ) : ℕ := n + 1
def support_opponent (n : ℕ) : ℕ := n - 1

-- Theorem statement
theorem maximize_Miraflores_win (n : ℕ) (hn : n > 0) : 
  ∃ (d1 d2 : ℕ), d1 = 1 ∧ d2 = 2 * n - 1 ∧ support_Miraflores n > support_opponent n := 
sorry

end maximize_Miraflores_win_l247_247534


namespace rate_of_change_area_at_t4_l247_247234

variable (t : ℝ)

def a (t : ℝ) : ℝ := 2 * t + 1

def b (t : ℝ) : ℝ := 3 * t + 2

def S (t : ℝ) : ℝ := a t * b t

theorem rate_of_change_area_at_t4 :
  (deriv S 4) = 55 := by
  sorry

end rate_of_change_area_at_t4_l247_247234


namespace find_number_of_children_l247_247787

theorem find_number_of_children (N : ℕ) (B : ℕ) 
    (h1 : B = 2 * N) 
    (h2 : B = 4 * (N - 160)) 
    : N = 320 := 
by
  sorry

end find_number_of_children_l247_247787


namespace crayon_production_correct_l247_247130

def numColors := 4
def crayonsPerColor := 2
def boxesPerHour := 5
def hours := 4

def crayonsPerBox := numColors * crayonsPerColor
def crayonsPerHour := boxesPerHour * crayonsPerBox
def totalCrayons := hours * crayonsPerHour

theorem crayon_production_correct :
  totalCrayons = 160 :=  
by
  sorry

end crayon_production_correct_l247_247130


namespace michael_large_balls_l247_247542

theorem michael_large_balls (total_rubber_bands : ℕ) (small_ball_rubber_bands : ℕ) (large_ball_rubber_bands : ℕ) (small_balls_made : ℕ)
  (h_total_rubber_bands : total_rubber_bands = 5000)
  (h_small_ball_rubber_bands : small_ball_rubber_bands = 50)
  (h_large_ball_rubber_bands : large_ball_rubber_bands = 300)
  (h_small_balls_made : small_balls_made = 22) :
  (total_rubber_bands - small_balls_made * small_ball_rubber_bands) / large_ball_rubber_bands = 13 :=
by {
  sorry
}

end michael_large_balls_l247_247542


namespace fraction_simplification_l247_247070

theorem fraction_simplification :
  8 * (15 / 11) * (-25 / 40) = -15 / 11 :=
by
  sorry

end fraction_simplification_l247_247070


namespace y_intercept_of_line_l247_247284

theorem y_intercept_of_line (x y : ℝ) (h : 4 * x + 7 * y = 28) : y = 4 :=
by
  -- The proof steps will go here.
  sorry

end y_intercept_of_line_l247_247284


namespace problem_statement_l247_247920

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

end problem_statement_l247_247920


namespace percentage_of_red_shirts_l247_247632

variable (total_students : ℕ) (blue_percent green_percent : ℕ) (other_students : ℕ)
  (H_total : total_students = 800)
  (H_blue : blue_percent = 45)
  (H_green : green_percent = 15)
  (H_other : other_students = 136)
  (H_blue_students : 0.45 * 800 = 360)
  (H_green_students : 0.15 * 800 = 120)
  (H_sum : 360 + 120 + 136 = 616)
  
theorem percentage_of_red_shirts :
  ((total_students - (360 + 120 + other_students)) / total_students) * 100 = 23 := 
by {
  sorry
}

end percentage_of_red_shirts_l247_247632


namespace directrix_of_parabola_l247_247024

theorem directrix_of_parabola : 
  (∀ (y x: ℝ), y^2 = 12 * x → x = -3) :=
sorry

end directrix_of_parabola_l247_247024


namespace ways_to_place_7_balls_into_3_boxes_l247_247027

theorem ways_to_place_7_balls_into_3_boxes :
  (Nat.choose (7 + 3 - 1) (3 - 1)) = 36 :=
by
  sorry

end ways_to_place_7_balls_into_3_boxes_l247_247027


namespace smallest_pos_int_mult_4410_sq_l247_247409

noncomputable def smallest_y : ℤ := 10

theorem smallest_pos_int_mult_4410_sq (y : ℕ) (hy : y > 0) :
  (∃ z : ℕ, 4410 * y = z^2) ↔ y = smallest_y :=
sorry

end smallest_pos_int_mult_4410_sq_l247_247409


namespace greatest_multiple_of_5_and_6_less_than_1000_l247_247380

theorem greatest_multiple_of_5_and_6_less_than_1000 : 
  ∃ n : ℕ, n < 1000 ∧ n % 5 = 0 ∧ n % 6 = 0 ∧ ∀ m : ℕ, m < 1000 ∧ m % 5 = 0 ∧ m % 6 = 0 → m ≤ n ∧ n = 990 :=
by
  sorry

end greatest_multiple_of_5_and_6_less_than_1000_l247_247380


namespace min_value_of_b_plus_3_div_a_l247_247603

theorem min_value_of_b_plus_3_div_a (a : ℝ) (b : ℝ) :
  0 < a →
  (∀ x, 0 < x → (a * x - 2) * (-x^2 - b * x + 4) ≤ 0) →
  b + 3 / a = 2 * Real.sqrt 2 :=
by
  sorry

end min_value_of_b_plus_3_div_a_l247_247603


namespace rational_numbers_inequality_l247_247784

theorem rational_numbers_inequality (a b : ℚ) (h1 : a * b > 0) (h2 : a + b < 0) : a < 0 :=
sorry

end rational_numbers_inequality_l247_247784


namespace min_segments_for_7_points_l247_247890

theorem min_segments_for_7_points (points : Fin 7 → ℝ × ℝ) : 
  ∃ (segments : Finset (Fin 7 × Fin 7)), 
    (∀ (a b c : Fin 7), a ≠ b ∧ b ≠ c ∧ c ≠ a → (a, b) ∈ segments ∨ (b, c) ∈ segments ∨ (c, a) ∈ segments) ∧
    segments.card = 9 :=
sorry

end min_segments_for_7_points_l247_247890


namespace opposite_sides_of_line_l247_247294

theorem opposite_sides_of_line 
  (x₀ y₀ : ℝ) 
  (h : (3 * x₀ + 2 * y₀ - 8) * (3 * 1 + 2 * 2 - 8) < 0) :
  3 * x₀ + 2 * y₀ > 8 :=
by
  sorry

end opposite_sides_of_line_l247_247294


namespace prove_a_range_l247_247463

noncomputable def f (x : ℝ) : ℝ := 1 / (2 ^ x + 2)

theorem prove_a_range (a : ℝ) :
  (∀ x, 2 ≤ x ∧ x ≤ 3 → f x + f (a - 2 * x) ≤ 1 / 2) → 5 ≤ a :=
by
  sorry

end prove_a_range_l247_247463


namespace red_clover_probability_l247_247600

open Probability
open MeasureTheory

noncomputable def problem_statement : Prop :=
  let n : ℕ := 60
  let p : ℚ := 0.84
  let m : ℕ := 52
  let mean : ℚ := n * p
  let variance : ℚ := n * p * (1 - p)
  let std_dev : ℚ := Real.sqrt variance
  let z_score : ℚ := (m - mean) / std_dev
  let φ := @std_normal_cdf ℝ _ -- This should be the CDF of the standard normal distribution.
  let approx_prob := φ z_score
  approx_prob ≈ 0.1201

theorem red_clover_probability : problem_statement := 
by sorry

end red_clover_probability_l247_247600


namespace central_angle_is_2_radians_l247_247768

namespace CircleAngle

def radius : ℝ := 2
def arc_length : ℝ := 4

theorem central_angle_is_2_radians : arc_length / radius = 2 := by
  sorry

end CircleAngle

end central_angle_is_2_radians_l247_247768


namespace manufacturer_A_more_cost_effective_manufacturer_B_more_cost_effective_l247_247544

-- Definitions of costs and the problem conditions.
def cost_manufacturer_A (desks chairs : ℕ) : ℝ :=
  200 * desks + 50 * (chairs - desks)

def cost_manufacturer_B (desks chairs : ℕ) : ℝ :=
  0.9 * (200 * desks + 50 * chairs)

-- Given condition: School needs 60 desks.
def desks : ℕ := 60

-- (1) Prove manufacturer A is more cost-effective when x < 360.
theorem manufacturer_A_more_cost_effective (chairs : ℕ) (h : chairs ≥ 60) :
  chairs < 360 → cost_manufacturer_A desks chairs < cost_manufacturer_B desks chairs :=
by sorry

-- (2) Prove manufacturer B is more cost-effective when x > 360.
theorem manufacturer_B_more_cost_effective (chairs : ℕ) (h : chairs ≥ 60) :
  chairs > 360 → cost_manufacturer_A desks chairs > cost_manufacturer_B desks chairs :=
by sorry

end manufacturer_A_more_cost_effective_manufacturer_B_more_cost_effective_l247_247544


namespace committee_count_with_president_l247_247481

-- Define the conditions
def total_people : ℕ := 12
def committee_size : ℕ := 5
def remaining_people : ℕ := 11
def president_inclusion : ℕ := 1

-- Define the calculation of binomial coefficient
noncomputable def binomial (n k : ℕ) : ℕ :=
  if h : k ≤ n then (Nat.choose n k) else 0

-- State the problem in Lean 4
theorem committee_count_with_president : 
  binomial remaining_people (committee_size - president_inclusion) = 330 :=
sorry

end committee_count_with_president_l247_247481


namespace arithmetic_square_root_16_l247_247943

theorem arithmetic_square_root_16 : Real.sqrt 16 = 4 := by
  sorry

end arithmetic_square_root_16_l247_247943


namespace tangent_line_ratio_l247_247916

variables {x1 x2 : ℝ}

theorem tangent_line_ratio (h1 : 2 * x1 = 3 * x2^2) (h2 : x1^2 = 2 * x2^3) : (x1 / x2) = 4 / 3 :=
by sorry

end tangent_line_ratio_l247_247916


namespace Anne_Katherine_savings_l247_247117

theorem Anne_Katherine_savings :
  ∃ A K : ℕ, (A - 150 = K / 3) ∧ (2 * K = 3 * A) ∧ (A + K = 750) := 
sorry

end Anne_Katherine_savings_l247_247117


namespace least_positive_x_multiple_l247_247960

theorem least_positive_x_multiple (x : ℕ) : 
  (∃ k : ℕ, (2 * x + 41) = 53 * k) → 
  x = 6 :=
sorry

end least_positive_x_multiple_l247_247960


namespace vampire_daily_needs_l247_247575

theorem vampire_daily_needs :
  (7 * 8) / 2 / 7 = 4 :=
by
  sorry

end vampire_daily_needs_l247_247575


namespace greatest_multiple_of_5_and_6_lt_1000_l247_247384

theorem greatest_multiple_of_5_and_6_lt_1000 : 
  ∃ n, n % 5 = 0 ∧ n % 6 = 0 ∧ n < 1000 ∧ (∀ m, m % 5 = 0 ∧ m % 6 = 0 ∧ m < 1000 → m ≤ n) :=
  sorry

end greatest_multiple_of_5_and_6_lt_1000_l247_247384


namespace remainder_of_A_div_by_9_l247_247088

theorem remainder_of_A_div_by_9 (A B : ℕ) (h : A = B * 9 + 13) : A % 9 = 4 := by
  sorry

end remainder_of_A_div_by_9_l247_247088


namespace expected_winnings_is_minus_half_l247_247979

-- Define the given condition in Lean
noncomputable def prob_win_side_1 : ℚ := 1 / 4
noncomputable def prob_win_side_2 : ℚ := 1 / 4
noncomputable def prob_lose_side_3 : ℚ := 1 / 3
noncomputable def prob_no_change_side_4 : ℚ := 1 / 6

noncomputable def win_amount_side_1 : ℚ := 2
noncomputable def win_amount_side_2 : ℚ := 4
noncomputable def lose_amount_side_3 : ℚ := -6
noncomputable def no_change_amount_side_4 : ℚ := 0

-- Define the expected value function
noncomputable def expected_winnings : ℚ :=
  (prob_win_side_1 * win_amount_side_1) +
  (prob_win_side_2 * win_amount_side_2) +
  (prob_lose_side_3 * lose_amount_side_3) +
  (prob_no_change_side_4 * no_change_amount_side_4)

-- Statement to prove
theorem expected_winnings_is_minus_half : expected_winnings = -1 / 2 := 
by
  sorry

end expected_winnings_is_minus_half_l247_247979


namespace michael_large_balls_l247_247541

theorem michael_large_balls (total_rubber_bands : ℕ) (small_ball_rubber_bands : ℕ) (large_ball_rubber_bands : ℕ) (small_balls_made : ℕ)
  (h_total_rubber_bands : total_rubber_bands = 5000)
  (h_small_ball_rubber_bands : small_ball_rubber_bands = 50)
  (h_large_ball_rubber_bands : large_ball_rubber_bands = 300)
  (h_small_balls_made : small_balls_made = 22) :
  (total_rubber_bands - small_balls_made * small_ball_rubber_bands) / large_ball_rubber_bands = 13 :=
by {
  sorry
}

end michael_large_balls_l247_247541


namespace alpha_plus_beta_l247_247028

theorem alpha_plus_beta (α β : ℝ) (hα_range : -Real.pi / 2 < α ∧ α < Real.pi / 2)
    (hβ_range : -Real.pi / 2 < β ∧ β < Real.pi / 2)
    (h_roots : ∃ (x1 x2 : ℝ), x1 = Real.tan α ∧ x2 = Real.tan β ∧ (x1^2 + 3 * Real.sqrt 3 * x1 + 4 = 0) ∧ (x2^2 + 3 * Real.sqrt 3 * x2 + 4 = 0)) :
    α + β = -2 * Real.pi / 3 :=
sorry

end alpha_plus_beta_l247_247028


namespace find_percentage_find_percentage_as_a_percentage_l247_247275

variable (P : ℝ)

theorem find_percentage (h : P / 2 = 0.02) : P = 0.04 :=
by
  sorry

theorem find_percentage_as_a_percentage (h : P / 2 = 0.02) : P = 4 :=
by
  sorry

end find_percentage_find_percentage_as_a_percentage_l247_247275


namespace num_divisors_count_l247_247469

theorem num_divisors_count (n : ℕ) (m : ℕ) (H : m = 32784) :
  (∃ S : Finset ℕ, (∀ x ∈ S, x ∈ (Finset.range 10) ∧ m % x = 0) ∧ S.card = n) ↔ n = 7 :=
by
  sorry

end num_divisors_count_l247_247469


namespace greatest_multiple_of_5_and_6_less_than_1000_l247_247386

theorem greatest_multiple_of_5_and_6_less_than_1000 : ∃ n, (n % 5 = 0) ∧ (n % 6 = 0) ∧ (n < 1000) ∧ (∀ m, (m % 5 = 0) ∧ (m % 6 = 0) ∧ (m < 1000) → m ≤ n) ∧ n = 990 :=
by sorry

end greatest_multiple_of_5_and_6_less_than_1000_l247_247386


namespace ellipse_formula_max_area_triangle_l247_247897

-- Definitions for Ellipse part
def ellipse_eq (x y a : ℝ) := (x^2 / a^2) + (y^2 / 3) = 1
def eccentricity (a : ℝ) := (Real.sqrt (a^2 - 3)) / a = 1 / 2

-- Definition for Circle intersection part
def circle_intersection_cond (t : ℝ) := (0 < t) ∧ (t < (2 * Real.sqrt 21) / 7)

-- Main theorem for ellipse equation
theorem ellipse_formula (a : ℝ) (h1 : a > Real.sqrt 3) (h2 : eccentricity a) :
  ellipse_eq x y 2 :=
sorry

-- Main theorem for maximum area of triangle ABC
theorem max_area_triangle (t : ℝ) (h : circle_intersection_cond t) :
  ∃ S, S = (3 * Real.sqrt 7) / 7 :=
sorry

end ellipse_formula_max_area_triangle_l247_247897


namespace shifted_quadratic_roots_l247_247912

theorem shifted_quadratic_roots {a h k : ℝ} (h_root_neg3 : a * (-3 + h) ^ 2 + k = 0)
                                 (h_root_2 : a * (2 + h) ^ 2 + k = 0) :
  (a * (-2 + h) ^ 2 + k = 0) ∧ (a * (3 + h) ^ 2 + k = 0) := by
  sorry

end shifted_quadratic_roots_l247_247912


namespace simplify_expression_l247_247938

theorem simplify_expression :
  (2^8 + 5^5) * (2^3 - (-2)^3)^7 = 9077567990336 :=
by
  sorry

end simplify_expression_l247_247938


namespace tan_eq_243_deg_l247_247449

theorem tan_eq_243_deg (n : ℤ) : -90 < n ∧ n < 90 ∧ Real.tan (n * Real.pi / 180) = Real.tan (243 * Real.pi / 180) ↔ n = 63 :=
by sorry

end tan_eq_243_deg_l247_247449


namespace find_number_l247_247690

theorem find_number (x : ℝ) (h : 0.5 * x = 0.1667 * x + 10) : x = 30 :=
sorry

end find_number_l247_247690


namespace find_common_ratio_geometric_l247_247488

variable {α : Type*} [Field α] {a : ℕ → α} {S : ℕ → α} {q : α} (h₁ : a 3 = 2 * S 2 + 1) (h₂ : a 4 = 2 * S 3 + 1)

def common_ratio_geometric : α := 3

theorem find_common_ratio_geometric (ha₃ : a 3 = 2 * S 2 + 1) (ha₄ : a 4 = 2 * S 3 + 1) :
  q = common_ratio_geometric := 
  sorry

end find_common_ratio_geometric_l247_247488


namespace miraflores_optimal_strategy_l247_247530

-- Definitions based on conditions
variable (n : ℕ)
def total_voters := 2 * n
def miraflores_supporters := n
def dick_maloney_supporters := n
def miraflores_is_a_voter := 1
def law_allows_division := true
def election_winner (district1 district2 : Set ℕ) : ℕ := 
  if (district1.card = 1 ∧ miraflores_is_a_voter ∈ district1) then miraflores_is_a_voter else dick_maloney_supporters

-- Mathematically equivalent proof problem
theorem miraflores_optimal_strategy (hall : law_allows_division) :
  (exists (district1 district2 : Set ℕ),
    ∀ v, v ∈ district1 ∨ v ∈ district2 ∧ district1.card + district2.card = total_voters ∧
    miraflores_supporters = 1 ∧ district1 = {miraflores_is_a_voter} ∧
    (election_winner district1 district2) = miraflores_is_a_voter) :=
sorry

end miraflores_optimal_strategy_l247_247530


namespace value_of_x_l247_247645

theorem value_of_x (g : ℝ → ℝ) (h : ∀ x, g (5 * x + 2) = 3 * x - 4) : g (-13) = -13 :=
by {
  sorry
}

end value_of_x_l247_247645


namespace inequality_holds_l247_247794

theorem inequality_holds (x y : ℝ) (hx : x ≠ -1) (hy : y ≠ -1) (hxy : x * y = 1) :
  ( (2 + x) / (1 + x) )^2 + ( (2 + y) / (1 + y) )^2 ≥ 9 / 2 :=
by
  sorry

end inequality_holds_l247_247794


namespace greatest_multiple_of_5_and_6_less_than_1000_l247_247397

open Nat

theorem greatest_multiple_of_5_and_6_less_than_1000 :
  ∃ a, (a % 5 = 0) ∧ (a % 6 = 0) ∧ (a < 1000) ∧ (a = 990) :=
by
  use 990
  simp [Nat.mod_eq_zero_of_dvd, Nat.le_zero_iff]
  repeat {split}; try {exact Int.ofNat_zero}
  { sorry }

end greatest_multiple_of_5_and_6_less_than_1000_l247_247397


namespace average_first_21_multiples_of_4_l247_247678

-- Define conditions
def n : ℕ := 21
def a1 : ℕ := 4
def an : ℕ := 4 * n
def sum_series (n a1 an : ℕ) : ℕ := (n * (a1 + an)) / 2

-- The problem statement in Lean 4
theorem average_first_21_multiples_of_4 : 
    (sum_series n a1 an) / n = 44 :=
by
  -- skipping the proof
  sorry

end average_first_21_multiples_of_4_l247_247678


namespace smallest_positive_integer_l247_247962

theorem smallest_positive_integer :
  ∃ x : ℕ, x > 0 ∧ x % 5 = 2 ∧ x % 3 = 1 ∧ x % 7 = 3 ∧ ∀ y : ℕ, y > 0 ∧ y % 5 = 2 ∧ y % 3 = 1 ∧ y % 7 = 3 → x ≤ y :=
by
  sorry

end smallest_positive_integer_l247_247962


namespace first_rocket_height_l247_247490

theorem first_rocket_height (h : ℝ) (combined_height : ℝ) (second_rocket_height : ℝ) 
  (H1 : second_rocket_height = 2 * h) 
  (H2 : combined_height = h + second_rocket_height) 
  (H3 : combined_height = 1500) : h = 500 := 
by 
  -- The proof would go here but is not required as per the instruction.
  sorry

end first_rocket_height_l247_247490


namespace comic_books_l247_247472

variables (x y : ℤ)

def condition1 (x y : ℤ) : Prop := y + 7 = 5 * (x - 7)
def condition2 (x y : ℤ) : Prop := y - 9 = 3 * (x + 9)

theorem comic_books (x y : ℤ) (h₁ : condition1 x y) (h₂ : condition2 x y) : x = 39 ∧ y = 153 :=
by
  sorry

end comic_books_l247_247472


namespace part_I_part_II_l247_247899

theorem part_I (a b : ℝ) (h1 : 0 < a) (h2 : b * a = 2)
  (h3 : (1 + b) * a = 3) :
  (a = 1) ∧ (b = 2) :=
by {
  sorry
}

theorem part_II (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : (1 : ℝ) / x + 2 / y = 1)
  (k : ℝ) : 2 * x + y ≥ k^2 + k + 2 → (-3 ≤ k) ∧ (k ≤ 2) :=
by {
  sorry
}

end part_I_part_II_l247_247899


namespace interest_received_l247_247218

theorem interest_received
  (total_investment : ℝ)
  (part_invested_6 : ℝ)
  (rate_6 : ℝ)
  (rate_9 : ℝ) :
  part_invested_6 = 7200 →
  rate_6 = 0.06 →
  rate_9 = 0.09 →
  total_investment = 10000 →
  (total_investment - part_invested_6) * rate_9 + part_invested_6 * rate_6 = 684 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end interest_received_l247_247218


namespace factorize_a_cubed_minus_a_l247_247861

theorem factorize_a_cubed_minus_a (a : ℝ) : a^3 - a = a * (a + 1) * (a - 1) :=
by
  sorry

end factorize_a_cubed_minus_a_l247_247861


namespace range_of_a_monotonically_decreasing_l247_247892

-- Definitions
def f (a x : ℝ) : ℝ := x^2 + 2 * (a - 1) * x + 2

-- Lean statement
theorem range_of_a_monotonically_decreasing {a : ℝ} : 
  (∀ x y : ℝ, -2 ≤ x → x ≤ 4 → -2 ≤ y → y ≤ 4 → x < y → f a y < f a x) ↔ a ≤ -3 := 
by 
  sorry

end range_of_a_monotonically_decreasing_l247_247892


namespace tetrahedron_equal_reciprocal_squares_l247_247212

noncomputable def tet_condition_heights (h_1 h_2 h_3 h_4 : ℝ) : Prop :=
True

noncomputable def tet_condition_distances (d_1 d_2 d_3 : ℝ) : Prop :=
True

theorem tetrahedron_equal_reciprocal_squares
  (h_1 h_2 h_3 h_4 d_1 d_2 d_3 : ℝ)
  (hc_hts : tet_condition_heights h_1 h_2 h_3 h_4)
  (hc_dsts : tet_condition_distances d_1 d_2 d_3) :
  1 / (h_1 ^ 2) + 1 / (h_2 ^ 2) + 1 / (h_3 ^ 2) + 1 / (h_4 ^ 2) =
  1 / (d_1 ^ 2) + 1 / (d_2 ^ 2) + 1 / (d_3 ^ 2) :=
sorry

end tetrahedron_equal_reciprocal_squares_l247_247212


namespace unique_ids_div_10_l247_247697

noncomputable def num_unique_ids (n : ℕ) : ℕ :=
  let no_repeats := nat.factorial 7 / nat.factorial (7 - 5)
  let repeats_with_zero := 5 * (nat.choose 6 4) * nat.factorial 4
  no_repeats + repeats_with_zero

theorem unique_ids_div_10 : num_unique_ids 5 / 10 = 432 := by
  sorry

end unique_ids_div_10_l247_247697


namespace student_can_escape_l247_247918

open Real

/-- The student can escape the pool given the following conditions:
 1. R is the radius of the circular pool.
 2. The teacher runs 4 times faster than the student swims.
 3. The teacher's running speed is v_T.
 4. The student's swimming speed is v_S = v_T / 4.
 5. The student swims along a circular path of radius r, where
    (1 - π / 4) * R < r < R / 4 -/
theorem student_can_escape (R v_T v_S r : ℝ) (h1 : v_S = v_T / 4)
  (h2 : (1 - π / 4) * R < r) (h3 : r < R / 4) : 
  True :=
sorry

end student_can_escape_l247_247918


namespace satisfied_probability_expected_satisfied_men_result_l247_247510

variable {men women : ℕ}

-- The total number of men and women
def total_people : ℕ := men + women

-- A man is satisfied if at least one woman sits next to him
def is_satisfied (men women : ℕ) : Prop :=
  let prob_discontent := (men - 1) * (men - 2) / (total_people * total_people - 2) in
  prob_discontent < 1 - 25/33

-- Expected number of satisfied men
def expected_satisfied_men (men women : ℕ) : ℚ :=
  let single_prob_satisfied := 25/33 in
  men * single_prob_satisfied

theorem satisfied_probability :
  men = 50 → women = 50 →
  (1 - (men - 1) * (men - 2) / (total_people * total_people - 2)) = 25/33 :=
by intros; sorry

theorem expected_satisfied_men_result :
  men = 50 → women = 50 →
  expected_satisfied_men men women = 1250/33 :=
by intros; sorry

end satisfied_probability_expected_satisfied_men_result_l247_247510


namespace product_of_two_integers_l247_247478

theorem product_of_two_integers (x y : ℕ) (h1 : x + y = 22) (h2 : x^2 - y^2 = 44) : x * y = 120 :=
by
  sorry

end product_of_two_integers_l247_247478


namespace triangle_side_lengths_l247_247694

theorem triangle_side_lengths 
  (r : ℝ) (CD : ℝ) (DB : ℝ) 
  (h_r : r = 4) 
  (h_CD : CD = 8) 
  (h_DB : DB = 10) :
  ∃ (AB AC : ℝ), AB = 14.5 ∧ AC = 12.5 :=
by
  sorry

end triangle_side_lengths_l247_247694


namespace f_monotonically_decreasing_iff_l247_247020

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then x^2 - 4 * a * x + 3 else (2 - 3 * a) * x + 1

theorem f_monotonically_decreasing_iff (a : ℝ) : 
  (∀ x₁ x₂, x₁ ≤ x₂ → f a x₁ ≥ f a x₂) ↔ (1/2 ≤ a ∧ a < 2/3) :=
by 
  sorry

end f_monotonically_decreasing_iff_l247_247020


namespace union_area_of_reflected_triangles_l247_247984

open Real

noncomputable def pointReflected (P : ℝ × ℝ) (line_y : ℝ) : ℝ × ℝ :=
  (P.1, 2 * line_y - P.2)

def areaOfTriangle (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem union_area_of_reflected_triangles :
  let A := (2, 6)
  let B := (5, -2)
  let C := (7, 3)
  let line_y := 2
  let A' := pointReflected A line_y
  let B' := pointReflected B line_y
  let C' := pointReflected C line_y
  areaOfTriangle A B C + areaOfTriangle A' B' C' = 29 := sorry

end union_area_of_reflected_triangles_l247_247984


namespace first_candidate_more_gain_l247_247702

-- Definitions for the salaries, revenues, training costs, and bonuses
def salary1 : ℕ := 42000
def revenue1 : ℕ := 93000
def training_cost_per_month : ℕ := 1200
def training_months : ℕ := 3

def salary2 : ℕ := 45000
def revenue2 : ℕ := 92000
def bonus2_percentage : ℕ := 1

-- Calculate net gains
def net_gain1 : ℕ :=
  revenue1 - salary1 - (training_cost_per_month * training_months)

def net_gain2 : ℕ :=
  revenue2 - salary2 - (salary2 * bonus2_percentage / 100)

def difference_in_gain : ℕ :=
  net_gain1 - net_gain2

-- Theorem statement
theorem first_candidate_more_gain :
  difference_in_gain = 850 :=
by
  -- Proof goes here
  sorry

end first_candidate_more_gain_l247_247702


namespace PR_length_right_triangle_l247_247328

theorem PR_length_right_triangle
  (P Q R : Type)
  (cos_R : ℝ)
  (PQ PR : ℝ)
  (h1 : cos_R = 5 * Real.sqrt 34 / 34)
  (h2 : PQ = Real.sqrt 34)
  (h3 : cos_R = PR / PQ) : PR = 5 := by
  sorry

end PR_length_right_triangle_l247_247328


namespace sum_of_coefficients_l247_247941

theorem sum_of_coefficients (a_0 a_1 a_2 a_3 a_4 a_5 a_6 : ℝ) :
  (∀ x : ℝ, (3 * x - 2)^6 = a_0 + a_1 * (2 * x - 1) + a_2 * (2 * x - 1)^2 + a_3 * (2 * x - 1)^3 + a_4 * (2 * x - 1)^4 + a_5 * (2 * x - 1)^5 + a_6 * (2 * x - 1)^6) ->
  a_1 + a_3 + a_5 = -63 / 2 := by
  sorry

end sum_of_coefficients_l247_247941


namespace sheets_per_class_per_day_l247_247148

theorem sheets_per_class_per_day
  (weekly_sheets : ℕ)
  (school_days_per_week : ℕ)
  (num_classes : ℕ)
  (h1 : weekly_sheets = 9000)
  (h2 : school_days_per_week = 5)
  (h3 : num_classes = 9) :
  (weekly_sheets / school_days_per_week) / num_classes = 200 :=
by
  sorry

end sheets_per_class_per_day_l247_247148


namespace general_form_of_numbers_whose_square_ends_with_9_l247_247729

theorem general_form_of_numbers_whose_square_ends_with_9 (x : ℤ) (h : (x^2 % 10 = 9)) :
  ∃ a : ℤ, x = 10 * a + 3 ∨ x = 10 * a + 7 :=
sorry

end general_form_of_numbers_whose_square_ends_with_9_l247_247729


namespace pentagon_rectangle_ratio_l247_247570

theorem pentagon_rectangle_ratio (p w : ℝ) 
    (pentagon_perimeter : 5 * p = 30) 
    (rectangle_perimeter : ∃ l, 2 * w + 2 * l = 30 ∧ l = 2 * w) : 
    p / w = 6 / 5 := 
by
  sorry

end pentagon_rectangle_ratio_l247_247570


namespace find_sum_principal_l247_247822

theorem find_sum_principal (P R : ℝ) :
  (P * (R + 5) * 10) / 100 = (P * R * 10) / 100 + 150 → P = 300 :=
by
  sorry

end find_sum_principal_l247_247822


namespace A_investment_amount_l247_247576

-- Conditions
variable (B_investment : ℝ) (C_investment : ℝ) (total_profit : ℝ) (A_profit : ℝ)
variable (B_investment_value : B_investment = 4200)
variable (C_investment_value : C_investment = 10500)
variable (total_profit_value : total_profit = 13600)
variable (A_profit_value : A_profit = 4080)

-- Proof statement
theorem A_investment_amount : 
  (∃ x : ℝ, x = 4410) :=
by
  sorry

end A_investment_amount_l247_247576


namespace preimage_of_3_1_l247_247753

theorem preimage_of_3_1 (a b : ℝ) (f : ℝ × ℝ → ℝ × ℝ) (h : ∀ (a b : ℝ), f (a, b) = (a + 2 * b, 2 * a - b)) :
  f (1, 1) = (3, 1) :=
by {
  sorry
}

end preimage_of_3_1_l247_247753


namespace at_least_20_percent_convex_l247_247891

theorem at_least_20_percent_convex {n : ℕ} (h₁ : n > 4) (no_three_collinear : ∀ u v w : ℕ, (u ≠ v ∧ v ≠ w ∧ u ≠ w) → ¬ collinear ℝ ({u, v, w} : set ℕ)) :
  ∃ (convex_quadrilaterals total_quadrilaterals : ℕ), ↑convex_quadrilaterals / ↑total_quadrilaterals ≥ (1/5 : ℚ) := sorry

end at_least_20_percent_convex_l247_247891


namespace probability_of_one_from_each_name_l247_247992

theorem probability_of_one_from_each_name (cards_total : ℕ)
    (letters_bill : ℕ) (letters_john : ℕ) :
    cards_total = 12 → letters_bill = 4 → letters_john = 5 →
    (letters_bill / cards_total) * 
    ((letters_john : ℚ) / (cards_total - 1)) +
    (letters_john / cards_total) * 
    ((letters_bill : ℚ) / (cards_total - 1)) = 
    (10 / 33) := by
  intros h_total h_bill h_john
  sorry

end probability_of_one_from_each_name_l247_247992


namespace area_shaded_region_l247_247101

theorem area_shaded_region :
  let r_s := 3   -- Radius of the smaller circle
  let r_l := 3 * r_s  -- Radius of the larger circle
  let A_l := π * r_l^2  -- Area of the larger circle
  let A_s := π * r_s^2  -- Area of the smaller circle
  A_l - A_s = 72 * π := 
by
  sorry

end area_shaded_region_l247_247101


namespace maximum_n_value_l247_247909

theorem maximum_n_value (a b c d : ℝ) (n : ℕ) (h₀ : a > b) (h₁ : b > c) (h₂ : c > d) 
(h₃ : (1 / (a - b)) + (1 / (b - c)) + (1 / (c - d)) ≥ (n / (a - d))) : n ≤ 9 :=
sorry

end maximum_n_value_l247_247909


namespace eiffel_tower_model_ratio_l247_247660

/-- Define the conditions of the problem as a structure -/
structure ModelCondition where
  eiffelTowerHeight : ℝ := 984 -- in feet
  modelHeight : ℝ := 6        -- in inches

/-- The main theorem statement -/
theorem eiffel_tower_model_ratio (cond : ModelCondition) : cond.eiffelTowerHeight / cond.modelHeight = 164 := 
by
  -- We can leave the proof out with 'sorry' for now.
  sorry

end eiffel_tower_model_ratio_l247_247660


namespace circumference_of_tank_a_l247_247517

def is_circumference_of_tank_a (h_A h_B C_B : ℝ) (V_A_eq : ℝ → Prop) : Prop :=
  ∃ (C_A : ℝ), 
    C_B = 10 ∧ 
    h_A = 10 ∧
    h_B = 7 ∧
    V_A_eq 0.7 ∧ 
    C_A = 7

theorem circumference_of_tank_a (h_A : ℝ) (h_B : ℝ) (C_B : ℝ) (V_A_eq : ℝ → Prop) : 
  is_circumference_of_tank_a h_A h_B C_B V_A_eq := 
by
  sorry

end circumference_of_tank_a_l247_247517


namespace subtract_angles_l247_247250

theorem subtract_angles :
  (90 * 60 * 60 - (78 * 60 * 60 + 28 * 60 + 56)) = (11 * 60 * 60 + 31 * 60 + 4) :=
by
  sorry

end subtract_angles_l247_247250


namespace expr_eval_l247_247811

theorem expr_eval : 3^3 - 3^2 + 3^1 - 3^0 = 20 := by
  sorry

end expr_eval_l247_247811


namespace find_a4_b4_c4_l247_247894

variables {a b c : ℝ}

theorem find_a4_b4_c4 (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 = 0.1) : a^4 + b^4 + c^4 = 0.005 :=
sorry

end find_a4_b4_c4_l247_247894


namespace shortest_ribbon_length_l247_247778

theorem shortest_ribbon_length :
  ∃ (L : ℕ), (∀ (n : ℕ), n = 2 ∨ n = 5 ∨ n = 7 → L % n = 0) ∧ L = 70 :=
by
  sorry

end shortest_ribbon_length_l247_247778


namespace ratio_c_d_l247_247269

theorem ratio_c_d (x y c d : ℝ) (h1 : 4 * x + 5 * y = c) (h2 : 8 * y - 10 * x = d) (hx : x ≠ 0) (hy : y ≠ 0) (hd : d ≠ 0) : c / d = 1 / 2 :=
by
  sorry

end ratio_c_d_l247_247269


namespace value_of_a_squared_plus_b_squared_l247_247902

theorem value_of_a_squared_plus_b_squared (a b : ℝ) 
  (h1 : (a + b) ^ 2 = 8) 
  (h2 : (a - b) ^ 2 = 12) : 
  a^2 + b^2 = 10 :=
sorry

end value_of_a_squared_plus_b_squared_l247_247902


namespace cubic_repeated_root_b_eq_100_l247_247074

theorem cubic_repeated_root_b_eq_100 (b : ℝ) (h1 : b ≠ 0)
  (h2 : ∃ x : ℝ, (b * x^3 + 15 * x^2 + 9 * x + 2 = 0) ∧ 
                 (3 * b * x^2 + 30 * x + 9 = 0)) :
  b = 100 :=
sorry

end cubic_repeated_root_b_eq_100_l247_247074


namespace melanie_total_value_l247_247058

-- Define the initial number of dimes Melanie had
def initial_dimes : ℕ := 7

-- Define the number of dimes given by her dad
def dimes_from_dad : ℕ := 8

-- Define the number of dimes given by her mom
def dimes_from_mom : ℕ := 4

-- Calculate the total number of dimes Melanie has now
def total_dimes : ℕ := initial_dimes + dimes_from_dad + dimes_from_mom

-- Define the value of each dime in dollars
def value_per_dime : ℝ := 0.10

-- Calculate the total value of dimes in dollars
def total_value_in_dollars : ℝ := total_dimes * value_per_dime

-- The theorem states that the total value in dollars is 1.90
theorem melanie_total_value : total_value_in_dollars = 1.90 := 
by
  -- Using the established definitions, the goal follows directly.
  sorry

end melanie_total_value_l247_247058


namespace candy_bar_sales_ratio_l247_247206

theorem candy_bar_sales_ratio
    (candy_bar_cost : ℕ := 2)
    (marvin_candy_sold : ℕ := 35)
    (tina_extra_earnings : ℕ := 140)
    (marvin_earnings := marvin_candy_sold * candy_bar_cost)
    (tina_earnings := marvin_earnings + tina_extra_earnings)
    (tina_candy_sold := tina_earnings / candy_bar_cost):
  tina_candy_sold / marvin_candy_sold = 3 :=
by
  sorry

end candy_bar_sales_ratio_l247_247206


namespace intersection_M_N_eq_l247_247497

-- Define the set M
def M : Set ℝ := {0, 1, 2}

-- Define the set N based on the given inequality
def N : Set ℝ := {x | x^2 - 3 * x + 2 ≤ 0}

-- The statement we want to prove
theorem intersection_M_N_eq {M N: Set ℝ} (hm: M = {0, 1, 2}) 
  (hn: N = {x | x^2 - 3 * x + 2 ≤ 0}) : 
  M ∩ N = {1, 2} :=
sorry

end intersection_M_N_eq_l247_247497


namespace polynomial_is_quadratic_l247_247226

theorem polynomial_is_quadratic (m : ℤ) (h : (m - 2 ≠ 0) ∧ (|m| = 2)) : m = -2 :=
by sorry

end polynomial_is_quadratic_l247_247226


namespace complex_modulus_l247_247807

open Complex

noncomputable def modulus_of_complex : ℂ :=
  (1 - 2 * Complex.I) * (1 - 2 * Complex.I) / Complex.I

theorem complex_modulus : Complex.abs modulus_of_complex = 5 :=
  sorry

end complex_modulus_l247_247807


namespace smallest_x_exists_l247_247565

theorem smallest_x_exists (x k m : ℤ) 
    (h1 : x + 3 = 7 * k) 
    (h2 : x - 5 = 8 * m) 
    (h3 : ∀ n : ℤ, ((n + 3) % 7 = 0) ∧ ((n - 5) % 8 = 0) → x ≤ n) : 
    x = 53 := by
  sorry

end smallest_x_exists_l247_247565


namespace men_in_first_group_l247_247127

theorem men_in_first_group (M : ℕ) (h1 : ∀ W, W = M * 30) (h2 : ∀ W, W = 10 * 36) : 
  M = 12 :=
by
  sorry

end men_in_first_group_l247_247127


namespace lcm_of_3_8_9_12_l247_247109

theorem lcm_of_3_8_9_12 : Nat.lcm (Nat.lcm 3 8) (Nat.lcm 9 12) = 72 :=
by
  sorry

end lcm_of_3_8_9_12_l247_247109


namespace mats_length_l247_247696

open Real

theorem mats_length (r : ℝ) (n : ℤ) (w : ℝ) (y : ℝ) (h₁ : r = 6) (h₂ : n = 8) (h₃ : w = 1):
  y = 6 * sqrt (2 - sqrt 2) :=
sorry

end mats_length_l247_247696


namespace num_of_possible_outcomes_l247_247425

def participants : Fin 6 := sorry  -- Define the participants as elements of Fin 6

theorem num_of_possible_outcomes : (6 * 5 * 4 = 120) :=
by {
  -- Prove this mathematical statement
  rfl
}

end num_of_possible_outcomes_l247_247425


namespace factorize_a_cubed_minus_a_l247_247869

theorem factorize_a_cubed_minus_a (a : ℝ) : a^3 - a = a * (a + 1) * (a - 1) :=
sorry

end factorize_a_cubed_minus_a_l247_247869


namespace sin_beta_value_l247_247047

theorem sin_beta_value (alpha beta : ℝ) (h1 : 0 < alpha) (h2 : alpha < beta) (h3 : beta < π / 2)
  (h4 : Real.sin alpha = 3 / 5) (h5 : Real.cos (alpha - beta) = 12 / 13) : Real.sin beta = 56 / 65 := by
  sorry

end sin_beta_value_l247_247047


namespace greatest_multiple_of_5_and_6_lt_1000_l247_247385

theorem greatest_multiple_of_5_and_6_lt_1000 : 
  ∃ n, n % 5 = 0 ∧ n % 6 = 0 ∧ n < 1000 ∧ (∀ m, m % 5 = 0 ∧ m % 6 = 0 ∧ m < 1000 → m ≤ n) :=
  sorry

end greatest_multiple_of_5_and_6_lt_1000_l247_247385


namespace count_integers_between_square_bounds_l247_247178

theorem count_integers_between_square_bounds :
  (n : ℕ) (300 < n^2 ∧ n^2 < 1200) → 17 :=
sorry

end count_integers_between_square_bounds_l247_247178


namespace greatest_multiple_of_5_and_6_lt_1000_l247_247383

theorem greatest_multiple_of_5_and_6_lt_1000 : 
  ∃ n, n % 5 = 0 ∧ n % 6 = 0 ∧ n < 1000 ∧ (∀ m, m % 5 = 0 ∧ m % 6 = 0 ∧ m < 1000 → m ≤ n) :=
  sorry

end greatest_multiple_of_5_and_6_lt_1000_l247_247383


namespace sale_price_of_sarees_l247_247949

theorem sale_price_of_sarees 
  (P : ℝ) 
  (d1 d2 d3 d4 tax_rate : ℝ) 
  (P_initial : P = 510) 
  (d1_val : d1 = 0.12) 
  (d2_val : d2 = 0.15) 
  (d3_val : d3 = 0.20) 
  (d4_val : d4 = 0.10) 
  (tax_val : tax_rate = 0.10) :
  let discount_step (price discount : ℝ) := price * (1 - discount)
  let tax_step (price tax_rate : ℝ) := price * (1 + tax_rate)
  let P1 := discount_step P d1
  let P2 := discount_step P1 d2
  let P3 := discount_step P2 d3
  let P4 := discount_step P3 d4
  let final_price := tax_step P4 tax_rate
  abs (final_price - 302.13) < 0.01 := 
sorry

end sale_price_of_sarees_l247_247949


namespace area_of_shaded_region_l247_247106

theorem area_of_shaded_region (r R : ℝ) (π : ℝ) (h1 : R = 3 * r) (h2 : 2 * r = 6) : 
  (π * R^2) - (π * r^2) = 72 * π :=
by
  sorry

end area_of_shaded_region_l247_247106


namespace functional_equation_solution_l247_247446

-- Define the conditions of the problem.
variable (f : ℝ → ℝ) 
variable (h : ∀ x y u v : ℝ, (f x + f y) * (f u + f v) = f (x * u - y * v) + f (x * v + y * u))

-- Formalize the statement that no other functions satisfy the conditions except f(x) = x^2.
theorem functional_equation_solution : (∀ x : ℝ, f x = x^2) :=
by
  -- The proof goes here, but since the proof is not required, we skip it.
  sorry

end functional_equation_solution_l247_247446


namespace sum_n_k_eq_eight_l247_247521

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Define the theorem to prove that n + k = 8 given the conditions
theorem sum_n_k_eq_eight {n k : ℕ} 
  (h1 : binom n k * 3 = binom n (k + 1))
  (h2 : binom n (k + 1) * 5 = binom n (k + 2) * 3) : n + k = 8 := by
  sorry

end sum_n_k_eq_eight_l247_247521


namespace hyungjun_initial_paint_count_l247_247112

theorem hyungjun_initial_paint_count (X : ℝ) (h1 : X / 2 - (X / 6 + 5) = 5) : X = 30 :=
sorry

end hyungjun_initial_paint_count_l247_247112


namespace part1_part2_l247_247021

noncomputable def f (x a : ℝ) := |x - a|

theorem part1 (a m : ℝ) :
  (∀ x, f x a ≤ m ↔ -1 ≤ x ∧ x ≤ 5) → a = 2 ∧ m = 3 :=
by
  sorry

theorem part2 (t x : ℝ) (h_t : 0 ≤ t ∧ t < 2) :
  f x 2 + t ≥ f (x + 2) 2 ↔ x ≤ (t + 2) / 2 :=
by
  sorry

end part1_part2_l247_247021


namespace geometric_series_sum_l247_247267

theorem geometric_series_sum :
  let a := 1 / 4
  let r := 1 / 4
  let n := 7
  (a * (1 - r^n) / (1 - r)) = (16383 / 49152) :=
by
  sorry

end geometric_series_sum_l247_247267


namespace necklace_price_l247_247060

variable (N : ℝ)

def price_of_bracelet : ℝ := 15.00
def price_of_earring : ℝ := 10.00
def num_necklaces_sold : ℝ := 5
def num_bracelets_sold : ℝ := 10
def num_earrings_sold : ℝ := 20
def num_complete_ensembles_sold : ℝ := 2
def price_of_complete_ensemble : ℝ := 45.00
def total_amount_made : ℝ := 565.0

theorem necklace_price :
  5 * N + 10 * price_of_bracelet + 20 * price_of_earring
  + 2 * price_of_complete_ensemble = total_amount_made → N = 25 :=
by
  intro h
  sorry

end necklace_price_l247_247060


namespace eleven_not_sum_of_two_primes_l247_247682

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem eleven_not_sum_of_two_primes :
  ¬ ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p + q = 11 :=
by sorry

end eleven_not_sum_of_two_primes_l247_247682


namespace arrangements_21_leaders_l247_247479

-- Define the factorial function
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Define the permutations A_n^k
def permutations (n k : ℕ) : ℕ :=
  if k ≤ n then factorial n / factorial (n - k) else 0

theorem arrangements_21_leaders : permutations 2 2 * permutations 18 18 = factorial 18 ^ 2 :=
by 
  sorry

end arrangements_21_leaders_l247_247479


namespace min_value_b_minus_a_l247_247022

noncomputable def f (x : ℝ) : ℝ := Real.exp x
noncomputable def g (x : ℝ) : ℝ := Real.log (x / 2) + 1 / 2

theorem min_value_b_minus_a :
  ∀ (a : ℝ), ∃ (b : ℝ), b > 0 ∧ f a = g b ∧ ∀ (y : ℝ), b - a = 2 * Real.exp (y - 1 / 2) - Real.log y → y = 1 / 2 → b - a = 2 + Real.log 2 := by
  sorry

end min_value_b_minus_a_l247_247022


namespace integer_divisibility_l247_247045

theorem integer_divisibility (m n : ℕ) (hm : m > 1) (hn : n > 1) (h1 : n ∣ 4^m - 1) (h2 : 2^m ∣ n - 1) : n = 2^m + 1 :=
by sorry

end integer_divisibility_l247_247045


namespace find_m_ineq_soln_set_min_value_a2_b2_l247_247023

-- Problem 1
theorem find_m_ineq_soln_set (m x : ℝ) (h1 : m - |x - 2| ≥ 1) (h2 : x ∈ Set.Icc 0 4) : m = 3 := by
  sorry

-- Problem 2
theorem min_value_a2_b2 (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 3) : a^2 + b^2 ≥ 9 / 2 := by
  sorry

end find_m_ineq_soln_set_min_value_a2_b2_l247_247023


namespace top_card_is_red_l247_247421

noncomputable def standard_deck (ranks : ℕ) (suits : ℕ) : ℕ := ranks * suits

def red_cards_in_deck (hearts : ℕ) (diamonds : ℕ) : ℕ := hearts + diamonds

noncomputable def probability_red_card (red_cards : ℕ) (total_cards : ℕ) : ℚ := red_cards / total_cards

theorem top_card_is_red (hearts diamonds spades clubs : ℕ) (deck_size : ℕ)
  (H1 : hearts = 13) (H2 : diamonds = 13) (H3 : spades = 13) (H4 : clubs = 13) (H5 : deck_size = 52):
  probability_red_card (red_cards_in_deck hearts diamonds) deck_size = 1/2 :=
by 
  sorry

end top_card_is_red_l247_247421


namespace total_cost_calc_l247_247717

variable (a b : ℝ)

def total_cost (a b : ℝ) := 2 * a + 3 * b

theorem total_cost_calc (a b : ℝ) : total_cost a b = 2 * a + 3 * b := by
  sorry

end total_cost_calc_l247_247717


namespace part_a_part_b_l247_247204

open Matrix

-- Conditions
variables {R : Type*} [CommRing R]

-- Part (a)
theorem part_a (A B : Matrix (Fin 2) (Fin 2) R) (h : (A - B) * (A - B) = 0) :
  det (A * A - B * B) = (det A - det B) * (det A - det B) :=
sorry

-- Part (b)
theorem part_b (A B : Matrix (Fin 2) (Fin 2) R) (h : (A - B) * (A - B) = 0) :
  det (A * B - B * A) = 0 ↔ det A = det B :=
sorry

end part_a_part_b_l247_247204


namespace sum_q_p_evaluation_l247_247203

def p (x : Int) : Int := x^2 - 3
def q (x : Int) : Int := x - 2

def T : List Int := [-4, -3, -2, -1, 0, 1, 2, 3, 4]

noncomputable def f (x : Int) : Int := q (p x)

noncomputable def sum_f_T : Int := List.sum (List.map f T)

theorem sum_q_p_evaluation :
  sum_f_T = 15 :=
by
  sorry

end sum_q_p_evaluation_l247_247203


namespace find_k_l247_247138

theorem find_k (a b k : ℝ) (h1 : a ≠ b ∨ a = b)
    (h2 : a^2 - 12 * a + k + 2 = 0)
    (h3 : b^2 - 12 * b + k + 2 = 0)
    (h4 : 4^2 - 12 * 4 + k + 2 = 0) :
    k = 34 ∨ k = 30 :=
by
  sorry

end find_k_l247_247138


namespace shire_total_population_l247_247667

theorem shire_total_population :
  let n := 25
  let avg_pop_min := 5400
  let avg_pop_max := 5700
  let avg_pop := (avg_pop_min + avg_pop_max) / 2
  n * avg_pop = 138750 :=
by
  let n := 25
  let avg_pop_min := 5400
  let avg_pop_max := 5700
  let avg_pop := (avg_pop_min + avg_pop_max) / 2
  show n * avg_pop = 138750
  sorry

end shire_total_population_l247_247667


namespace area_fraction_of_square_hole_l247_247708

theorem area_fraction_of_square_hole (A B C M N : ℝ)
  (h1 : B = C)
  (h2 : M = 0.5 * A)
  (h3 : N = 0.5 * A) :
  (M * N) / (B * C) = 1 / 4 :=
by
  sorry

end area_fraction_of_square_hole_l247_247708


namespace point_distance_5_5_l247_247502

-- Define the distance function in the context of the problem
def distance_from_origin (x : ℝ) : ℝ := abs x

-- Formalize the proposition
theorem point_distance_5_5 (x : ℝ) : distance_from_origin x = 5.5 → (x = -5.5 ∨ x = 5.5) :=
by
  intro h
  simp [distance_from_origin] at h
  sorry

end point_distance_5_5_l247_247502


namespace deceased_member_income_l247_247220

theorem deceased_member_income (a b c d : ℝ)
    (h1 : a = 735) 
    (h2 : b = 650)
    (h3 : c = 4 * 735)
    (h4 : d = 3 * 650) :
    c - d = 990 := by
  sorry

end deceased_member_income_l247_247220


namespace Mike_monthly_time_is_200_l247_247211

def tv_time (days : Nat) (hours_per_day : Nat) : Nat := days * hours_per_day

def video_game_time (total_tv_time_per_week : Nat) (num_days_playing : Nat) : Nat :=
  (total_tv_time_per_week / 7 / 2) * num_days_playing

def piano_time (weekday_hours : Nat) (weekend_hours : Nat) : Nat :=
  weekday_hours * 5 + weekend_hours * 2

def weekly_time (tv_time : Nat) (video_game_time : Nat) (piano_time : Nat) : Nat :=
  tv_time + video_game_time + piano_time

def monthly_time (weekly_time : Nat) (weeks : Nat) : Nat :=
  weekly_time * weeks

theorem Mike_monthly_time_is_200 : monthly_time
  (weekly_time 
     (tv_time 3 4 + tv_time 2 3 + tv_time 2 5) 
     (video_game_time 28 3) 
     (piano_time 2 3))
  4 = 200 :=
  by
  sorry

end Mike_monthly_time_is_200_l247_247211


namespace isabel_remaining_pages_l247_247489

def total_problems : ℕ := 72
def finished_problems : ℕ := 32
def problems_per_page : ℕ := 8

theorem isabel_remaining_pages :
  (total_problems - finished_problems) / problems_per_page = 5 := 
sorry

end isabel_remaining_pages_l247_247489


namespace greatest_multiple_of_5_and_6_less_than_1000_l247_247390

theorem greatest_multiple_of_5_and_6_less_than_1000 : 
    ∃ k, (k % 5 = 0) ∧ (k % 6 = 0) ∧ (k < 1000) ∧ (∀ m, (m % 5 = 0) ∧ (m % 6 = 0) ∧ (m < 1000) → m ≤ k) :=
begin
  use 990,
  repeat { split },
  { norm_num }, -- 990 % 5 = 0
  { norm_num }, -- 990 % 6 = 0
  { norm_num }, -- 990 < 1000
  { intros m hm, 
    cases hm with h5 h6,
    cases h6 with h6 hlt,
    have : m % 30 = 0 := by { sorry }, -- Show that m is a multiple of LCM(5, 6)
    apply le_of_lt,
    have hle : m/30 < ↑(1000/30) := by { sorry }, -- Compare the greatest multiple of 30 less than 1000
    exact hle,
  }
end

end greatest_multiple_of_5_and_6_less_than_1000_l247_247390


namespace maximum_members_in_dance_troupe_l247_247330

theorem maximum_members_in_dance_troupe (m : ℕ) (h1 : 25 * m % 31 = 7) (h2 : 25 * m < 1300) : 25 * m = 875 :=
by {
  sorry
}

end maximum_members_in_dance_troupe_l247_247330


namespace Carlos_earnings_l247_247196

theorem Carlos_earnings :
  ∃ (wage : ℝ), 
  (18 * wage) = (12 * wage + 36) ∧ 
  wage = 36 / 6 ∧ 
  (12 * wage + 18 * wage) = 180 :=
by
  sorry

end Carlos_earnings_l247_247196


namespace steven_apples_minus_peaches_l247_247636

-- Define the number of apples and peaches Steven has.
def steven_apples : ℕ := 19
def steven_peaches : ℕ := 15

-- Problem statement: Prove that the number of apples minus the number of peaches is 4.
theorem steven_apples_minus_peaches : steven_apples - steven_peaches = 4 := by
  sorry

end steven_apples_minus_peaches_l247_247636


namespace john_final_price_l247_247334

theorem john_final_price : 
  let goodA_price := 2500
  let goodA_rebate := 0.06 * goodA_price
  let goodA_price_after_rebate := goodA_price - goodA_rebate
  let goodA_sales_tax := 0.10 * goodA_price_after_rebate
  let goodA_final_price := goodA_price_after_rebate + goodA_sales_tax
  
  let goodB_price := 3150
  let goodB_rebate := 0.08 * goodB_price
  let goodB_price_after_rebate := goodB_price - goodB_rebate
  let goodB_sales_tax := 0.12 * goodB_price_after_rebate
  let goodB_final_price := goodB_price_after_rebate + goodB_sales_tax

  let goodC_price := 1000
  let goodC_rebate := 0.05 * goodC_price
  let goodC_price_after_rebate := goodC_price - goodC_rebate
  let goodC_sales_tax := 0.07 * goodC_price_after_rebate
  let goodC_final_price := goodC_price_after_rebate + goodC_sales_tax

  let total_amount := goodA_final_price + goodB_final_price + goodC_final_price

  let special_voucher_discount := 0.03 * total_amount
  let final_price := total_amount - special_voucher_discount
  let rounded_final_price := Float.round final_price

  rounded_final_price = 6642 := by
  sorry

end john_final_price_l247_247334


namespace alice_profit_l247_247709

def total_bracelets : ℕ := 52
def cost_materials : ℝ := 3.0
def bracelets_given_away : ℕ := 8
def price_per_bracelet : ℝ := 0.25

theorem alice_profit :
  let bracelets_sold := total_bracelets - bracelets_given_away in
  let total_revenue := bracelets_sold * price_per_bracelet in
  let profit := total_revenue - cost_materials in
  profit = 8.00 :=
by
  sorry

end alice_profit_l247_247709


namespace product_value_l247_247400

theorem product_value : 
  (1 / 2) * 4 * (1 / 8) * 16 * (1 / 32) * 64 * (1 / 128) * 256 * (1 / 512) * 1024 = 32 := 
by
  sorry

end product_value_l247_247400


namespace average_payment_is_460_l247_247968

theorem average_payment_is_460 :
  let n := 52
  let first_payment := 410
  let extra := 65
  let num_first_payments := 12
  let num_rest_payments := n - num_first_payments
  let rest_payment := first_payment + extra
  (num_first_payments * first_payment + num_rest_payments * rest_payment) / n = 460 := by
  sorry

end average_payment_is_460_l247_247968


namespace annie_start_crayons_l247_247989

def start_crayons (end_crayons : ℕ) (added_crayons : ℕ) : ℕ := end_crayons - added_crayons

theorem annie_start_crayons (added_crayons end_crayons : ℕ) (h1 : added_crayons = 36) (h2 : end_crayons = 40) :
  start_crayons end_crayons added_crayons = 4 :=
by
  rw [h1, h2]
  exact Nat.sub_eq_of_eq_add sorry  -- skips the detailed proof

end annie_start_crayons_l247_247989


namespace candidate_net_gain_difference_l247_247704

theorem candidate_net_gain_difference :
  let salary1 := 42000
      revenue1 := 93000
      training_cost_per_month := 1200
      training_months := 3
      salary2 := 45000
      revenue2 := 92000
      hiring_bonus_percent := 1 / 100 in
  let total_training_cost1 := training_cost_per_month * training_months in
  let hiring_bonus2 := salary2 * hiring_bonus_percent in
  let net_gain1 := revenue1 - salary1 - total_training_cost1 in
  let net_gain2 := revenue2 - salary2 - hiring_bonus2 in
  net_gain1 - net_gain2 = 850 :=
by
  sorry

end candidate_net_gain_difference_l247_247704


namespace marketing_percentage_l247_247258

-- Define the conditions
variable (monthly_budget : ℝ)
variable (rent : ℝ := monthly_budget / 5)
variable (remaining_after_rent : ℝ := monthly_budget - rent)
variable (food_beverages : ℝ := remaining_after_rent / 4)
variable (remaining_after_food_beverages : ℝ := remaining_after_rent - food_beverages)
variable (employee_salaries : ℝ := remaining_after_food_beverages / 3)
variable (remaining_after_employee_salaries : ℝ := remaining_after_food_beverages - employee_salaries)
variable (utilities : ℝ := remaining_after_employee_salaries / 7)
variable (remaining_after_utilities : ℝ := remaining_after_employee_salaries - utilities)
variable (marketing : ℝ := 0.15 * remaining_after_utilities)

-- Define the theorem we want to prove
theorem marketing_percentage : marketing / monthly_budget * 100 = 5.14 := by
  sorry

end marketing_percentage_l247_247258


namespace license_plate_combinations_l247_247142

-- Definitions based on the conditions
def num_letters := 26
def num_digits := 10
def num_positions := 5

def choose (n k : ℕ) : ℕ := Nat.choose n k

-- Main theorem statement
theorem license_plate_combinations :
  choose num_letters 2 * (num_letters - 2) * choose num_positions 2 * choose (num_positions - 2) 2 * num_digits * (num_digits - 1) * (num_digits - 2) = 7776000 :=
by
  sorry

end license_plate_combinations_l247_247142


namespace cos_eight_arccos_one_fourth_l247_247276

theorem cos_eight_arccos_one_fourth :
  Real.cos (8 * Real.arccos (1 / 4)) = 172546 / 1048576 :=
sorry

end cos_eight_arccos_one_fourth_l247_247276


namespace bobs_total_profit_l247_247434

-- Definitions of the conditions
def cost_per_dog : ℕ := 250
def num_dogs : ℕ := 2
def num_puppies : ℕ := 6
def selling_price_per_puppy : ℕ := 350

-- Definition of the problem statement
theorem bobs_total_profit : (num_puppies * selling_price_per_puppy) - (num_dogs * cost_per_dog) = 1600 := 
by
  sorry

end bobs_total_profit_l247_247434


namespace no_solution_in_natural_numbers_l247_247506

theorem no_solution_in_natural_numbers (x y z : ℕ) : 
  (x / y : ℚ) + (y / z : ℚ) + (z / x : ℚ) ≠ 1 := 
by sorry

end no_solution_in_natural_numbers_l247_247506


namespace B_time_l247_247414

-- Define the work rates of A, B, and C in terms of how long they take to complete the work
variable (A B C : ℝ)

-- Conditions provided in the problem
axiom A_rate : A = 1 / 3
axiom BC_rate : B + C = 1 / 3
axiom AC_rate : A + C = 1 / 2

-- Prove that B alone will take 6 hours to complete the work
theorem B_time : B = 1 / 6 → (1 / B) = 6 := by
  intro hB
  sorry

end B_time_l247_247414


namespace Proof_l247_247957

-- Definitions for the conditions
def Snakes : Type := {s : Fin 20 // s < 20}
def Purple (s : Snakes) : Prop := s.val < 6
def Happy (s : Snakes) : Prop := s.val >= 6 ∧ s.val < 14
def CanAdd (s : Snakes) : Prop := ∃ h ∈ Finset.Ico 6 14, h = s.val
def CanSubtract (s : Snakes) : Prop := ¬Purple s

-- Conditions extraction
axiom SomeHappyCanAdd : ∃ s : Snakes, Happy s ∧ CanAdd s
axiom NoPurpleCanSubtract : ∀ s : Snakes, Purple s → ¬CanSubtract s
axiom CantSubtractCantAdd : ∀ s : Snakes, ¬CanSubtract s → ¬CanAdd s

-- Theorem statement depending on conditions
theorem Proof :
    (∀ s : Snakes, CanSubtract s → ¬Purple s) ∧
    (∃ s : Snakes, Happy s ∧ ¬Purple s) ∧
    (∃ s : Snakes, Happy s ∧ ¬CanSubtract s) :=
by {
  sorry -- Proof required here
}

end Proof_l247_247957


namespace total_points_seven_players_l247_247629

theorem total_points_seven_players (S : ℕ) (x : ℕ) 
  (hAlex : Alex_scored = S / 4)
  (hBen : Ben_scored = 2 * S / 7)
  (hCharlie : Charlie_scored = 15)
  (hTotal : S / 4 + 2 * S / 7 + 15 + x = S)
  (hMultiple : S = 56) : 
  x = 11 := 
sorry

end total_points_seven_players_l247_247629


namespace number_of_boys_l247_247251

theorem number_of_boys
  (M W B : Nat)
  (total_earnings wages_of_men earnings_of_men : Nat)
  (num_men_eq_women : 5 * M = W)
  (num_men_eq_boys : 5 * M = B)
  (earnings_eq_90 : total_earnings = 90)
  (men_wages_6 : wages_of_men = 6)
  (men_earnings_eq_30 : earnings_of_men = M * wages_of_men) : 
  B = 5 := 
by
  sorry

end number_of_boys_l247_247251


namespace parabola_focus_eq_l247_247593

/-- Given the equation of a parabola y = -4x^2 - 8x + 1, prove that its focus is at (-1, 79/16). -/
theorem parabola_focus_eq :
  ∀ x y : ℝ, y = -4 * x ^ 2 - 8 * x + 1 → 
  ∃ h k p : ℝ, y = -4 * (x + 1)^2 + 5 ∧ 
  h = -1 ∧ k = 5 ∧ p = -1 / 16 ∧ (h, k + p) = (-1, 79/16) :=
by
  sorry

end parabola_focus_eq_l247_247593


namespace shekar_average_marks_l247_247797

-- Define the scores for each subject
def mathematics := 76
def science := 65
def social_studies := 82
def english := 67
def biology := 55
def computer_science := 89
def history := 74
def geography := 63
def physics := 78
def chemistry := 71

-- Define the total number of subjects
def number_of_subjects := 10

-- State the theorem to prove the average marks
theorem shekar_average_marks :
  (mathematics + science + social_studies + english + biology +
   computer_science + history + geography + physics + chemistry) 
   / number_of_subjects = 72 := 
by
  -- Proof is omitted
  sorry

end shekar_average_marks_l247_247797


namespace smallest_x_satisfies_equation_l247_247071

theorem smallest_x_satisfies_equation : 
  ∀ x : ℚ, 7 * (10 * x^2 + 10 * x + 11) = x * (10 * x - 45) → x = -7 / 5 :=
by {
  sorry
}

end smallest_x_satisfies_equation_l247_247071


namespace part_a_part_b_part_c_part_d_l247_247646

-- define the partitions function
def P (k l n : ℕ) : ℕ := sorry

-- Part (a) statement
theorem part_a (k l n : ℕ) :
  P k l n - P k (l - 1) n = P (k - 1) l (n - l) :=
sorry

-- Part (b) statement
theorem part_b (k l n : ℕ) :
  P k l n - P (k - 1) l n = P k (l - 1) (n - k) :=
sorry

-- Part (c) statement
theorem part_c (k l n : ℕ) :
  P k l n = P l k n :=
sorry

-- Part (d) statement
theorem part_d (k l n : ℕ) :
  P k l n = P k l (k * l - n) :=
sorry

end part_a_part_b_part_c_part_d_l247_247646


namespace number_multiplies_a_l247_247030

theorem number_multiplies_a (a b x : ℝ) (h₀ : x * a = 8 * b) (h₁ : a ≠ 0 ∧ b ≠ 0) (h₂ : (a / 8) / (b / 7) = 1) : x = 7 :=
by
  sorry

end number_multiplies_a_l247_247030


namespace homothety_maps_C_to_E_l247_247513

-- Defining Points and Circles
variable {Point Circle : Type}
variable [Inhabited Point] -- assuming Point type is inhabited

-- Definitions for points H, K_A, I_A, K_B, I_B, K_C, I_C
variables (H K_A I_A K_B I_B K_C I_C : Point)

-- Define midpoints
def is_midpoint (A B M : Point) : Prop := sorry -- In a real proof, you would define midpoint in terms of coordinates

-- Define homothety function
def homothety (center : Point) (ratio : ℝ) (P : Point) : Point := sorry -- In a real proof, you would define the homothety transformation

-- Defining Circles
variables (C E : Circle)

-- Define circumcircle of a triangle
def is_circumcircle (a b c : Point) (circle : Circle) : Prop := sorry

-- Statements from conditions
axiom midpointA : is_midpoint H K_A I_A
axiom midpointB : is_midpoint H K_B I_B
axiom midpointC : is_midpoint H K_C I_C

axiom circumcircle_C : is_circumcircle K_A K_B K_C C
axiom circumcircle_E : is_circumcircle I_A I_B I_C E

-- Lean theorem stating the proof problem
theorem homothety_maps_C_to_E :
  ∀ (H K_A I_A K_B I_B K_C I_C : Point) (C E : Circle),
  (is_midpoint H K_A I_A) →
  (is_midpoint H K_B I_B) →
  (is_midpoint H K_C I_C) →
  (is_circumcircle K_A K_B K_C C) →
  (is_circumcircle I_A I_B I_C E) →
  (homothety H 0.5 K_A = I_A ) →
  (homothety H 0.5 K_B = I_B ) →
  (homothety H 0.5 K_C = I_C ) →
  C = E :=
by intro; sorry

end homothety_maps_C_to_E_l247_247513


namespace seq_prime_l247_247073

/-- A strictly increasing sequence of positive integers. -/
def is_strictly_increasing (a : ℕ → ℕ) : Prop :=
  ∀ n m, n < m → a n < a m

/-- An infinite strictly increasing sequence of positive integers. -/
def strictly_increasing_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n, 0 < a n ∧ is_strictly_increasing a

/-- A sequence of distinct primes. -/
def distinct_primes (p : ℕ → ℕ) : Prop :=
  ∀ m n, m ≠ n → p m ≠ p n ∧ Nat.Prime (p n)

/-- The main theorem to be proved. -/
theorem seq_prime (a p : ℕ → ℕ) (h1 : strictly_increasing_sequence a) (h2 : distinct_primes p)
  (h3 : ∀ n, p n ∣ a n) (h4 : ∀ n k, a n - a k = p n - p k) : ∀ n, Nat.Prime (a n) := 
by
  sorry

end seq_prime_l247_247073


namespace value_of_x_l247_247548

theorem value_of_x (x : ℝ) : (12 - x)^3 = x^3 → x = 12 :=
by
  sorry

end value_of_x_l247_247548


namespace greatest_C_inequality_l247_247605

theorem greatest_C_inequality (α x y z : ℝ) (hα_pos : 0 < α) (hx_pos : 0 < x) (hy_pos : 0 < y) (hz_pos : 0 < z) (h_xyz_sum : x * y + y * z + z * x = α) : 
  16 ≤ (1 + α / x^2) * (1 + α / y^2) * (1 + α / z^2) / (x / z + z / x + 2) :=
sorry

end greatest_C_inequality_l247_247605


namespace probability_equal_2s_after_4040_rounds_l247_247274

/-- 
Given three players Diana, Nathan, and Olivia each starting with $2, each player (with at least $1) 
simultaneously gives $1 to one of the other two players randomly every 20 seconds. 
Prove that the probability that after the bell has rung 4040 times, 
each player will have $2$ is $\frac{1}{4}$.
-/
theorem probability_equal_2s_after_4040_rounds 
  (n_rounds : ℕ) (start_money : ℕ) (probability_outcome : ℚ) :
  n_rounds = 4040 →
  start_money = 2 →
  probability_outcome = 1 / 4 :=
by
  sorry

end probability_equal_2s_after_4040_rounds_l247_247274


namespace find_b_l247_247181

variables {a b : ℝ}

theorem find_b (h1 : (x - 3) * (x - a) = x^2 - b * x - 10) : b = -1/3 :=
  sorry

end find_b_l247_247181


namespace unattainable_y_l247_247898

theorem unattainable_y (x : ℝ) (h : x ≠ -4 / 3) :
  ¬ ∃ y : ℝ, y = (2 - x) / (3 * x + 4) ∧ y = -1 / 3 :=
by
  sorry

end unattainable_y_l247_247898


namespace exists_integers_cd_iff_divides_l247_247216

theorem exists_integers_cd_iff_divides (a b : ℤ) :
  (∃ c d : ℤ, a + b + c + d = 0 ∧ a * c + b * d = 0) ↔ (a - b) ∣ (2 * a * b) := 
by
  sorry

end exists_integers_cd_iff_divides_l247_247216


namespace miguel_socks_probability_l247_247934

theorem miguel_socks_probability :
  let total_socks := 10
  let socks_per_color := 2
  let colors := 5
  let draw_socks := 5
  let total_combinations := Nat.choose total_socks draw_socks
  let desired_combinations :=
    (Nat.choose colors 2) * (Nat.choose (colors - 2 + 1) 1) * socks_per_color
  let probability := desired_combinations / total_combinations
  probability = 5 / 21 :=
by
  let total_socks := 10
  let socks_per_color := 2
  let colors := 5
  let draw_socks := 5
  let total_combinations := Nat.choose total_socks draw_socks
  let desired_combinations :=
    (Nat.choose colors 2) * (Nat.choose (colors - 2 + 1) 1) * socks_per_color
  let probability := desired_combinations / total_combinations
  sorry

end miguel_socks_probability_l247_247934


namespace frank_spends_more_l247_247675

def cost_computer_table : ℕ := 140
def cost_computer_chair : ℕ := 100
def cost_joystick : ℕ := 20
def frank_share_joystick : ℕ := cost_joystick / 4
def eman_share_joystick : ℕ := cost_joystick * 3 / 4

def total_spent_frank : ℕ := cost_computer_table + frank_share_joystick
def total_spent_eman : ℕ := cost_computer_chair + eman_share_joystick

theorem frank_spends_more : total_spent_frank - total_spent_eman = 30 :=
by
  sorry

end frank_spends_more_l247_247675


namespace no_integers_six_digit_cyclic_permutation_l247_247655

theorem no_integers_six_digit_cyclic_permutation (n : ℕ) (a b c d e f : ℕ) (h : 10 ≤ a ∧ a < 10) :
  ¬(n = 5 ∨ n = 6 ∨ n = 8 ∧
    n * (a * 10^5 + b * 10^4 + c * 10^3 + d * 10^2 + e * 10 + f) =
    b * 10^5 + c * 10^4 + d * 10^3 + e * 10^2 + f * 10 + a) :=
by sorry

end no_integers_six_digit_cyclic_permutation_l247_247655


namespace greatest_multiple_of_5_and_6_less_than_1000_l247_247387

theorem greatest_multiple_of_5_and_6_less_than_1000 : ∃ n, (n % 5 = 0) ∧ (n % 6 = 0) ∧ (n < 1000) ∧ (∀ m, (m % 5 = 0) ∧ (m % 6 = 0) ∧ (m < 1000) → m ≤ n) ∧ n = 990 :=
by sorry

end greatest_multiple_of_5_and_6_less_than_1000_l247_247387


namespace distance_to_destination_l247_247566

theorem distance_to_destination 
  (speed : ℝ) (time : ℝ) 
  (h_speed : speed = 100) 
  (h_time : time = 5) : 
  speed * time = 500 :=
by
  rw [h_speed, h_time]
  -- This simplifies to 100 * 5 = 500
  norm_num

end distance_to_destination_l247_247566


namespace cos_squared_value_l247_247748

theorem cos_squared_value (α : ℝ) (h : Real.sin (2 * α) = 2 / 3) :
  Real.cos (α + π / 4) ^ 2 = 1 / 6 :=
sorry

end cos_squared_value_l247_247748


namespace f_is_periodic_with_period_4a_l247_247288

variable (f : ℝ → ℝ) (a : ℝ)

theorem f_is_periodic_with_period_4a (h : ∀ x : ℝ, f (x + a) = (1 + f x) / (1 - f x)) : ∀ x : ℝ, f (x + 4 * a) = f x :=
by
  sorry

end f_is_periodic_with_period_4a_l247_247288


namespace initially_calculated_avg_height_l247_247661

theorem initially_calculated_avg_height
  (A : ℕ)
  (initially_calculated_total_height : ℕ := 35 * A)
  (wrong_height : ℕ := 166)
  (actual_height : ℕ := 106)
  (height_overestimation : ℕ := wrong_height - actual_height)
  (actual_avg_height : ℕ := 179)
  (correct_total_height : ℕ := 35 * actual_avg_height)
  (initially_calculate_total_height_is_more : initially_calculated_total_height = correct_total_height + height_overestimation) :
  A = 181 :=
by
  sorry

end initially_calculated_avg_height_l247_247661


namespace fraction_of_paper_per_book_l247_247780

theorem fraction_of_paper_per_book (total_fraction_used : ℚ) (num_books : ℕ) (h1 : total_fraction_used = 5 / 8) (h2 : num_books = 5) : 
  (total_fraction_used / num_books) = 1 / 8 :=
by
  sorry

end fraction_of_paper_per_book_l247_247780


namespace pet_store_earnings_l247_247134

theorem pet_store_earnings :
  let kitten_price := 6
  let puppy_price := 5
  let kittens_sold := 2
  let puppies_sold := 1 
  let total_earnings := kittens_sold * kitten_price + puppies_sold * puppy_price
  total_earnings = 17 :=
by
  sorry

end pet_store_earnings_l247_247134


namespace ratio_area_triangles_to_square_l247_247773

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

end ratio_area_triangles_to_square_l247_247773


namespace problem1_correctness_problem2_correctness_l247_247844

noncomputable def problem1_solution_1 (x : ℝ) : Prop := x = Real.sqrt 5 - 1
noncomputable def problem1_solution_2 (x : ℝ) : Prop := x = -Real.sqrt 5 - 1
noncomputable def problem2_solution_1 (x : ℝ) : Prop := x = 5
noncomputable def problem2_solution_2 (x : ℝ) : Prop := x = -1 / 3

theorem problem1_correctness (x : ℝ) :
  (x^2 + 2*x - 4 = 0) → (problem1_solution_1 x ∨ problem1_solution_2 x) :=
by sorry

theorem problem2_correctness (x : ℝ) :
  (3 * x * (x - 5) = 5 - x) → (problem2_solution_1 x ∨ problem2_solution_2 x) :=
by sorry

end problem1_correctness_problem2_correctness_l247_247844


namespace task_candy_distribution_l247_247067

noncomputable def candy_distribution_eq_eventually (n : ℕ) (a : ℕ → ℕ) : Prop :=
  ∃ k : ℕ, ∀ m : ℕ, ∀ j : ℕ, m ≥ k → a (j + m * n) = a (0 + m * n)

theorem task_candy_distribution :
  ∀ n : ℕ, n > 0 →
  ∀ a : ℕ → ℕ,
  (∀ i : ℕ, a i = if a i % 2 = 1 then (a i) + 1 else a i) →
  (∀ i : ℕ, a (i + 1) = a i / 2 + a (i - 1) / 2) →
  candy_distribution_eq_eventually n a :=
by
  intros n n_positive a h_even h_transfer
  sorry

end task_candy_distribution_l247_247067


namespace side_length_is_36_l247_247833

variable (a : ℝ)

def side_length_of_largest_square (a : ℝ) := 
  2 * (a / 2) ^ 2 + 2 * (a / 4) ^ 2 = 810

theorem side_length_is_36 (h : side_length_of_largest_square a) : a = 36 :=
by
  sorry

end side_length_is_36_l247_247833


namespace combined_fraction_correct_l247_247936

def standard_fractions :=
  { soda_water := 8/21,
    lemon_juice := 4/21,
    sugar := 3/21,
    papaya_puree := 3/21,
    spice_blend := 2/21,
    lime_extract := 1/21 }

def malfunction_fractions :=
  { soda_water := 1/2 * standard_fractions.soda_water,
    sugar := 2 * standard_fractions.sugar,
    spice_blend := 1/5 * standard_fractions.spice_blend }

def combined_fraction :=
  malfunction_fractions.soda_water + malfunction_fractions.sugar + malfunction_fractions.spice_blend

theorem combined_fraction_correct :
  combined_fraction = 52/105 :=
by
  sorry

end combined_fraction_correct_l247_247936


namespace product_of_numbers_l247_247085

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 72) (h2 : x - y = 20) : x * y = 1196 := 
sorry

end product_of_numbers_l247_247085


namespace morning_snowfall_l247_247192

theorem morning_snowfall (afternoon_snowfall total_snowfall : ℝ) (h₀ : afternoon_snowfall = 0.5) (h₁ : total_snowfall = 0.63):
  total_snowfall - afternoon_snowfall = 0.13 :=
by 
  sorry

end morning_snowfall_l247_247192


namespace geometrical_shapes_OABC_l247_247746

/-- Given distinct points A(x₁, y₁), B(x₂, y₂), and C(2x₁ - x₂, 2y₁ - y₂) on a coordinate plane
    and the origin O(0,0), determine the possible geometrical shapes that the figure OABC can form
    among these three possibilities: (1) parallelogram (2) straight line (3) rhombus.
    
    Prove that the figure OABC can form either a parallelogram or a straight line,
    but not a rhombus.
-/
theorem geometrical_shapes_OABC (x₁ y₁ x₂ y₂ : ℝ) (h_distinct : (x₁, y₁) ≠ (x₂, y₂) ∧ (x₁, y₁) ≠ (2 * x₁ - x₂, 2 * y₁ - y₂) ∧ (x₂, y₂) ≠ (2 * x₁ - x₂, 2 * y₁ - y₂)) :
  (∃ t : ℝ, t ≠ 0 ∧ t ≠ 1 ∧ x₂ = t * x₁ ∧ y₂ = t * y₁) ∨
  (2 * x₁ = x₁ + x₂ ∧ 2 * y₁ = y₁ + y₂) :=
sorry

end geometrical_shapes_OABC_l247_247746


namespace probability_A_wins_probability_A_wins_2_l247_247959

def binomial (n k : ℕ) := Nat.choose n k

noncomputable def P (n : ℕ) : ℚ := 
  1/2 * (1 - binomial (2 * n) n / 2 ^ (2 * n))

theorem probability_A_wins (n : ℕ) : P n = 1/2 * (1 - binomial (2 * n) n / 2 ^ (2 * n)) := 
by sorry

theorem probability_A_wins_2 : P 2 = 5 / 16 := 
by sorry

end probability_A_wins_probability_A_wins_2_l247_247959


namespace factorial_comparison_l247_247549

theorem factorial_comparison :
  (Nat.factorial (Nat.factorial 100)) <
  (Nat.factorial 99)^(Nat.factorial 100) * (Nat.factorial 100)^(Nat.factorial 99) :=
  sorry

end factorial_comparison_l247_247549


namespace one_third_sugar_l247_247837

theorem one_third_sugar (s : ℚ) (h : s = 23 / 4) : (1 / 3) * s = 1 + 11 / 12 :=
by {
  sorry
}

end one_third_sugar_l247_247837


namespace generalized_inequality_combinatorial_inequality_l247_247829

-- Part 1: Generalized Inequality
theorem generalized_inequality (n : ℕ) (a b : Fin n → ℝ) 
  (ha : ∀ i, 0 < a i) (hb : ∀ i, 0 < b i) :
  (Finset.univ.sum (fun i => (b i)^2 / (a i))) ≥
  ((Finset.univ.sum (fun i => b i))^2 / (Finset.univ.sum (fun i => a i))) :=
sorry

-- Part 2: Combinatorial Inequality
theorem combinatorial_inequality (n : ℕ) (hn : 0 < n) :
  (Finset.range (n + 1)).sum (fun k => (2 * k + 1) / (Nat.choose n k)) ≥
  ((n + 1)^3 / (2^n : ℝ)) :=
sorry

end generalized_inequality_combinatorial_inequality_l247_247829


namespace ratio_pentagon_rectangle_l247_247569

-- Definitions of conditions.
def pentagon_side_length (p : ℕ) : Prop := 5 * p = 30
def rectangle_width (w : ℕ) : Prop := 6 * w = 30

-- The theorem to prove.
theorem ratio_pentagon_rectangle (p w : ℕ) (h1 : pentagon_side_length p) (h2 : rectangle_width w) :
  p / w = 6 / 5 :=
by sorry

end ratio_pentagon_rectangle_l247_247569


namespace sum_of_positive_numbers_is_360_l247_247946

variable (x y : ℝ)
variable (h1 : x * y = 50 * (x + y))
variable (h2 : x * y = 75 * (x - y))

theorem sum_of_positive_numbers_is_360 (hx : 0 < x) (hy : 0 < y) : x + y = 360 :=
by sorry

end sum_of_positive_numbers_is_360_l247_247946


namespace tan_B_eq_one_third_l247_247197

theorem tan_B_eq_one_third
  (A B : ℝ)
  (h1 : Real.cos A = 4 / 5)
  (h2 : Real.tan (A - B) = 1 / 3) :
  Real.tan B = 1 / 3 := by
  sorry

end tan_B_eq_one_third_l247_247197


namespace even_n_if_fraction_is_integer_l247_247589

theorem even_n_if_fraction_is_integer (n : ℕ) (h_pos : 0 < n) :
  (∃ a b : ℕ, 0 < b ∧ (a^2 + n^2) % (b^2 - n^2) = 0) → n % 2 = 0 := 
sorry

end even_n_if_fraction_is_integer_l247_247589


namespace rate_of_interest_first_year_l247_247152

-- Define the conditions
def principal : ℝ := 9000
def rate_second_year : ℝ := 0.05
def total_amount_after_2_years : ℝ := 9828

-- Define the problem statement which we need to prove
theorem rate_of_interest_first_year (R : ℝ) :
  (principal + (principal * R / 100)) + 
  ((principal + (principal * R / 100)) * rate_second_year) = 
  total_amount_after_2_years → 
  R = 4 := 
by
  sorry

end rate_of_interest_first_year_l247_247152


namespace new_pyramid_volume_l247_247836

/-- Given an original pyramid with volume 40 cubic inches, where the length is doubled, 
    the width is tripled, and the height is increased by 50%, 
    prove that the volume of the new pyramid is 360 cubic inches. -/
theorem new_pyramid_volume (V : ℝ) (l w h : ℝ) 
  (h_volume : V = 1 / 3 * l * w * h) 
  (h_original : V = 40) : 
  (2 * l) * (3 * w) * (1.5 * h) / 3 = 360 :=
by
  sorry

end new_pyramid_volume_l247_247836


namespace minimum_value_of_f_l247_247765

noncomputable def f (x : ℝ) : ℝ := (x^2 - 4 * x + 5) / (2 * x - 4)

theorem minimum_value_of_f (x : ℝ) (h : x ≥ 5 / 2) : ∃ y, y = f x ∧ y = 1 :=
by
  sorry

end minimum_value_of_f_l247_247765


namespace trader_allows_discount_l247_247116

-- Definitions for cost price, marked price, and selling price
variable (cp : ℝ)
def mp := cp + 0.12 * cp
def sp := cp - 0.01 * cp

-- The statement to prove
theorem trader_allows_discount :
  mp cp - sp cp = 13 :=
sorry

end trader_allows_discount_l247_247116


namespace sum_of_A_and_B_l247_247253

theorem sum_of_A_and_B:
  ∃ A B : ℕ, (A = 2 + 4) ∧ (B - 3 = 1) ∧ (A < 10) ∧ (B < 10) ∧ (A + B = 10) :=
by 
  sorry

end sum_of_A_and_B_l247_247253


namespace nature_of_roots_of_quadratic_l247_247740

theorem nature_of_roots_of_quadratic (k : ℝ) (h1 : k > 0) (h2 : 3 * k^2 - 2 = 10) :
  let a := 1
  let b := -(4 * k - 3)
  let c := 3 * k^2 - 2
  let Δ := b^2 - 4 * a * c
  Δ < 0 :=
by
  sorry

end nature_of_roots_of_quadratic_l247_247740


namespace barbata_interest_rate_l247_247586

theorem barbata_interest_rate
  (initial_investment: ℝ)
  (additional_investment: ℝ)
  (additional_rate: ℝ)
  (total_income_rate: ℝ)
  (total_income: ℝ)
  (h_total_investment_eq: initial_investment + additional_investment = 4800)
  (h_total_income_eq: 0.06 * (initial_investment + additional_investment) = total_income):
  (initial_investment * (r : ℝ) + additional_investment * additional_rate = total_income) →
  r = 0.04 := sorry

end barbata_interest_rate_l247_247586


namespace audrey_ratio_in_3_years_l247_247585

-- Define the ages and the conditions
def Heracles_age : ℕ := 10
def Audrey_age := Heracles_age + 7
def Audrey_age_in_3_years := Audrey_age + 3

-- Statement: Prove that the ratio of Audrey's age in 3 years to Heracles' current age is 2:1
theorem audrey_ratio_in_3_years : (Audrey_age_in_3_years / Heracles_age) = 2 := sorry

end audrey_ratio_in_3_years_l247_247585


namespace greatest_multiple_of_5_and_6_less_than_1000_l247_247396

open Nat

theorem greatest_multiple_of_5_and_6_less_than_1000 :
  ∃ a, (a % 5 = 0) ∧ (a % 6 = 0) ∧ (a < 1000) ∧ (a = 990) :=
by
  use 990
  simp [Nat.mod_eq_zero_of_dvd, Nat.le_zero_iff]
  repeat {split}; try {exact Int.ofNat_zero}
  { sorry }

end greatest_multiple_of_5_and_6_less_than_1000_l247_247396


namespace range_of_a_l247_247754

theorem range_of_a (a : ℝ) :
  (∀ x, a * x^2 - x + (1 / 16 * a) > 0 → a > 2) →
  (0 < a - 3 / 2 ∧ a - 3 / 2 < 1 → 3 / 2 < a ∧ a < 5 / 2) →
  (¬ ((∀ x, a * x^2 - x + (1 / 16 * a) > 0) ∧ (0 < a - 3 / 2 ∧ a - 3 / 2 < 1))) →
  ((3 / 2 < a) ∧ (a ≤ 2)) ∨ (a ≥ 5 / 2) :=
by
  sorry

end range_of_a_l247_247754


namespace ratio_boys_to_girls_l247_247628

theorem ratio_boys_to_girls
  (b g : ℕ) 
  (h1 : b = g + 6) 
  (h2 : b + g = 36) : b / g = 7 / 5 :=
sorry

end ratio_boys_to_girls_l247_247628


namespace find_all_k_l247_247151

theorem find_all_k :
  ∃ (k : ℝ), ∃ (v : ℝ × ℝ), v ≠ 0 ∧ (∃ (v₀ v₁ : ℝ), v = (v₀, v₁) 
  ∧ (3 * v₀ + 6 * v₁) = k * v₀ ∧ (4 * v₀ + 3 * v₁) = k * v₁) 
  ↔ k = 3 + 2 * Real.sqrt 6 ∨ k = 3 - 2 * Real.sqrt 6 :=
by
  -- here goes the proof
  sorry

end find_all_k_l247_247151


namespace shaded_area_l247_247103

theorem shaded_area (d_small : ℝ) (r_large : ℝ) (shaded_area : ℝ) :
  (d_small = 6) → (r_large = 3 * (d_small / 2)) → shaded_area = (π * r_large^2 - π * (d_small / 2)^2) → shaded_area = 72 * π :=
by
  intro h_d_small h_r_large h_shaded_area
  rw [h_d_small, h_r_large, h_shaded_area]
  sorry

end shaded_area_l247_247103


namespace range_of_x_l247_247228

noncomputable def is_valid_x (x : ℝ) : Prop :=
  x ≥ 0 ∧ x ≠ 4

theorem range_of_x (x : ℝ) : 
  is_valid_x x ↔ x ≥ 0 ∧ x ≠ 4 :=
by sorry

end range_of_x_l247_247228


namespace fraction_of_time_to_cover_distance_l247_247126

-- Definitions for the given conditions
def distance : ℝ := 540
def initial_time : ℝ := 12
def new_speed : ℝ := 60

-- The statement we need to prove
theorem fraction_of_time_to_cover_distance :
  ∃ (x : ℝ), (x = 3 / 4) ∧ (distance / (initial_time * x) = new_speed) :=
by
  -- Proof steps would go here
  sorry

end fraction_of_time_to_cover_distance_l247_247126


namespace f_decreasing_ln_inequality_limit_inequality_l247_247494

-- Definitions of the given conditions
noncomputable def f (x : ℝ) : ℝ := (Real.log (1 + x)) / x

-- Statements we need to prove

-- (I) Prove that f(x) is decreasing on (0, +∞)
theorem f_decreasing : ∀ x y : ℝ, 0 < x → x < y → f y < f x := sorry

-- (II) Prove that for the inequality ln(1 + x) < ax to hold for all x in (0, +∞), a must be at least 1
theorem ln_inequality (a : ℝ) : (∀ x : ℝ, 0 < x → Real.log (1 + x) < a * x) ↔ 1 ≤ a := sorry

-- (III) Prove that (1 + 1/n)^n < e for all n in ℕ*
theorem limit_inequality (n : ℕ) (h : n ≠ 0) : (1 + 1 / n) ^ n < Real.exp 1 := sorry

end f_decreasing_ln_inequality_limit_inequality_l247_247494


namespace negation_of_exists_proposition_l247_247357

theorem negation_of_exists_proposition :
  ¬ (∃ x₀ : ℝ, x₀^2 - 1 < 0) ↔ ∀ x : ℝ, x^2 - 1 ≥ 0 :=
by
  sorry

end negation_of_exists_proposition_l247_247357


namespace none_of_the_choices_sum_of_150_consecutive_integers_l247_247406

theorem none_of_the_choices_sum_of_150_consecutive_integers :
  ¬(∃ k : ℕ, 678900 = 150 * k + 11325) ∧
  ¬(∃ k : ℕ, 1136850 = 150 * k + 11325) ∧
  ¬(∃ k : ℕ, 1000000 = 150 * k + 11325) ∧
  ¬(∃ k : ℕ, 2251200 = 150 * k + 11325) ∧
  ¬(∃ k : ℕ, 1876800 = 150 * k + 11325) :=
by
  sorry

end none_of_the_choices_sum_of_150_consecutive_integers_l247_247406


namespace ceil_square_values_l247_247308

theorem ceil_square_values (x : ℝ) (h : ⌈x⌉ = 12) : 
  ∃ n : ℕ, (∀ m : ℕ, m = n ↔ (121 < x^2 ∧ x^2 ≤ 144) ∧ (⌈x^2⌉ = m)) ∧ n = 23 :=
by
  sorry

end ceil_square_values_l247_247308


namespace bob_needs_50_percent_improvement_l247_247842

def bob_time_in_seconds : ℕ := 640
def sister_time_in_seconds : ℕ := 320
def percentage_improvement_needed (bob_time sister_time : ℕ) : ℚ :=
  ((bob_time - sister_time) / bob_time : ℚ) * 100

theorem bob_needs_50_percent_improvement :
  percentage_improvement_needed bob_time_in_seconds sister_time_in_seconds = 50 := by
  sorry

end bob_needs_50_percent_improvement_l247_247842


namespace find_numbers_l247_247358

theorem find_numbers (x y z u n : ℤ)
  (h1 : x + y + z + u = 36)
  (h2 : x + n = y - n)
  (h3 : x + n = z * n)
  (h4 : x + n = u / n) :
  n = 1 ∧ x = 8 ∧ y = 10 ∧ z = 9 ∧ u = 9 :=
sorry

end find_numbers_l247_247358


namespace work_duration_l247_247972

theorem work_duration (p q r : ℕ) (Wp Wq Wr : ℕ) (t1 t2 : ℕ) (T : ℝ) :
  (Wp = 20) → (Wq = 12) → (Wr = 30) →
  (t1 = 4) → (t2 = 4) →
  (T = (t1 + t2 + (4/15 * Wr) / (1/(Wr) + 1/(Wq) + 1/(Wp)))) →
  T = 9.6 :=
by
  intros;
  sorry

end work_duration_l247_247972


namespace sufficient_but_not_necessary_l247_247008

variable (p q : Prop)

theorem sufficient_but_not_necessary : (¬p → ¬(p ∧ q)) ∧ (¬(¬p) → ¬(p ∧ q) → False) :=
by {
  sorry
}

end sufficient_but_not_necessary_l247_247008


namespace unique_integral_root_x_minus_9_over_x_minus_2_eq_5_minus_9_over_x_minus_2_l247_247583

theorem unique_integral_root_x_minus_9_over_x_minus_2_eq_5_minus_9_over_x_minus_2 : ∃! (x : ℤ), x - 9 / (x - 2) = 5 - 9 / (x - 2) := 
by
  sorry

end unique_integral_root_x_minus_9_over_x_minus_2_eq_5_minus_9_over_x_minus_2_l247_247583


namespace geometric_sequence_general_term_l247_247895

theorem geometric_sequence_general_term (a : ℕ → ℝ) (q : ℝ) (a1 : ℝ) 
  (h1 : a 5 = a1 * q^4)
  (h2 : a 10 = a1 * q^9)
  (h3 : ∀ n, 2 * (a n + a (n + 2)) = 5 * a (n + 1))
  (h4 : ∀ n, a n = a1 * q^(n - 1))
  (h_inc : q > 1) :
  ∀ n, a n = 2^n :=
by
  sorry

end geometric_sequence_general_term_l247_247895


namespace number_of_terms_arithmetic_sequence_l247_247084

-- Definitions for the arithmetic sequence conditions
open Nat

noncomputable def S4 := 26
noncomputable def Sn := 187
noncomputable def last4_sum (n : ℕ) (a d : ℕ) := 
  (n - 3) * a + 3 * (n - 2) * d + 3 * (n - 1) * d + n * d

-- Statement for the problem
theorem number_of_terms_arithmetic_sequence 
  (a d n : ℕ) (h1 : 4 * a + 6 * d = S4) (h2 : n * (2 * a + (n - 1) * d) / 2 = Sn) 
  (h3 : last4_sum n a d = 110) : 
  n = 11 :=
sorry

end number_of_terms_arithmetic_sequence_l247_247084


namespace largest_divisor_of_expression_l247_247224

theorem largest_divisor_of_expression (n : ℤ) : ∃ k, ∀ n : ℤ, n^4 - n^2 = k * 12 :=
by sorry

end largest_divisor_of_expression_l247_247224


namespace multiplication_counts_l247_247450

open Polynomial

noncomputable def horner_multiplications (n : ℕ) : ℕ := n

noncomputable def direct_summation_multiplications (n : ℕ) : ℕ := n * (n + 1) / 2

theorem multiplication_counts (P : Polynomial ℝ) (x₀ : ℝ) (n : ℕ)
  (h_degree : P.degree = n) :
  horner_multiplications n = n ∧ direct_summation_multiplications n = (n * (n + 1)) / 2 :=
by
  sorry

end multiplication_counts_l247_247450


namespace horse_revolutions_l247_247255

-- Defining the problem conditions
def radius_outer : ℝ := 30
def radius_inner : ℝ := 10
def revolutions_outer : ℕ := 25

-- The question we need to prove
theorem horse_revolutions :
  (revolutions_outer : ℝ) * (radius_outer / radius_inner) = 75 := 
by
  sorry

end horse_revolutions_l247_247255


namespace largest_possible_value_l247_247764

noncomputable def largest_log_expression (a b : ℝ) (h1 : a ≥ b) (h2 : b > 2) : ℝ := 
  Real.log (a^2 / b^2) / Real.log a + Real.log (b^2 / a^2) / Real.log b

theorem largest_possible_value (a b : ℝ) (h1 : a ≥ b) (h2 : b > 2) (h3 : a = b) : 
  largest_log_expression a b h1 h2 = 0 :=
by
  sorry

end largest_possible_value_l247_247764


namespace speed_in_still_water_l247_247248

/-- Conditions -/
def upstream_speed : ℝ := 30
def downstream_speed : ℝ := 40

/-- Theorem: The speed of the man in still water is 35 kmph. -/
theorem speed_in_still_water : 
  (upstream_speed + downstream_speed) / 2 = 35 := 
by 
  sorry

end speed_in_still_water_l247_247248


namespace train_speed_l247_247574

noncomputable def train_length : ℝ := 1500
noncomputable def bridge_length : ℝ := 1200
noncomputable def crossing_time : ℝ := 30

theorem train_speed :
  (train_length + bridge_length) / crossing_time = 90 := by
  sorry

end train_speed_l247_247574


namespace bob_total_profit_l247_247438

def initial_cost (num_dogs : ℕ) (cost_per_dog : ℕ) : ℕ := num_dogs * cost_per_dog

def revenue (num_puppies : ℕ) (price_per_puppy : ℕ) : ℕ := num_puppies * price_per_puppy

def total_profit (initial_cost : ℕ) (revenue : ℕ) : ℕ := revenue - initial_cost

theorem bob_total_profit (c1 : initial_cost 2 250 = 500)
                        (c2 : revenue 6 350 = 2100)
                        (c3 : total_profit 500 2100 = 1600) :
  total_profit (initial_cost 2 250) (revenue 6 350) = 1600 := by
  sorry

end bob_total_profit_l247_247438


namespace f_order_l247_247905

variable (f : ℝ → ℝ)

-- Given conditions
axiom even_f : ∀ x : ℝ, f (-x) = f x
axiom incr_f : ∀ x y : ℝ, x < y ∧ y ≤ -1 → f x < f y

-- Prove that f(2) < f (-3/2) < f(-1)
theorem f_order : f 2 < f (-3/2) ∧ f (-3/2) < f (-1) :=
by
  sorry

end f_order_l247_247905


namespace calculate_total_notebooks_given_to_tom_l247_247266

noncomputable def total_notebooks_given_to_tom : ℝ :=
  let initial_red := 15
  let initial_blue := 17
  let initial_white := 19
  let red_given_day1 := 4.5
  let blue_given_day1 := initial_blue / 3
  let remaining_red_day1 := initial_red - red_given_day1
  let remaining_blue_day1 := initial_blue - blue_given_day1
  let white_given_day2 := initial_white / 2
  let blue_given_day2 := remaining_blue_day1 * 0.25
  let remaining_white_day2 := initial_white - white_given_day2
  let remaining_blue_day2 := remaining_blue_day1 - blue_given_day2
  let red_given_day3 := 3.5
  let blue_given_day3 := (remaining_blue_day2 * 2) / 5
  let remaining_red_day3 := remaining_red_day1 - red_given_day3
  let remaining_blue_day3 := remaining_blue_day2 - blue_given_day3
  let white_kept_day3 := remaining_white_day2 / 4
  let remaining_white_day3 := initial_white - white_kept_day3
  let remaining_notebooks_day3 := remaining_red_day3 + remaining_blue_day3 + remaining_white_day3
  let notebooks_total_day3 := initial_red + initial_blue + initial_white - red_given_day1 - blue_given_day1 - white_given_day2 - blue_given_day2 - red_given_day3 - blue_given_day3 - white_kept_day3
  let tom_notebooks := red_given_day1 + blue_given_day1
  notebooks_total_day3

theorem calculate_total_notebooks_given_to_tom : total_notebooks_given_to_tom = 10.17 :=
  sorry

end calculate_total_notebooks_given_to_tom_l247_247266


namespace trapezoid_area_correct_l247_247820

noncomputable def trapezoid_area : ℝ := 
  let base1 : ℝ := 8
  let base2 : ℝ := 4
  let height : ℝ := 2
  (1 / 2) * (base1 + base2) * height

theorem trapezoid_area_correct :
  trapezoid_area = 12.0 :=
by
  -- sorry is used here to skip the actual proof
  sorry

end trapezoid_area_correct_l247_247820


namespace cos_identity_l247_247160

theorem cos_identity 
  (x : ℝ) 
  (h : Real.sin (x - π / 3) = 3 / 5) : 
  Real.cos (x + π / 6) = -3 / 5 := 
by 
  sorry

end cos_identity_l247_247160


namespace diagonals_in_octagon_l247_247759

/-- The formula to calculate the number of diagonals in a polygon -/
def number_of_diagonals (n : Nat) : Nat :=
  (n * (n - 3)) / 2

/-- The number of sides in an octagon -/
def sides_of_octagon : Nat := 8

/-- The number of diagonals in an octagon is 20. -/
theorem diagonals_in_octagon : number_of_diagonals sides_of_octagon = 20 :=
by
  sorry

end diagonals_in_octagon_l247_247759


namespace train_length_is_approx_l247_247424

noncomputable def train_length : ℝ :=
  let speed_kmh : ℝ := 54
  let conversion_factor : ℝ := 1000 / 3600
  let speed_ms : ℝ := speed_kmh * conversion_factor
  let time_seconds : ℝ := 11.999040076793857
  speed_ms * time_seconds

theorem train_length_is_approx : abs (train_length - 179.99) < 0.001 := 
by
  sorry

end train_length_is_approx_l247_247424


namespace smallest_M_exists_l247_247737

theorem smallest_M_exists :
  ∃ M : ℕ, M = 249 ∧
  (∃ k1 : ℕ, (M + k1 = 8 * k1 ∨ M + k1 + 1 = 8 * k1 ∨ M + k1 + 2 = 8 * k1)) ∧
  (∃ k2 : ℕ, (M + k2 = 27 * k2 ∨ M + k2 + 1 = 27 * k2 ∨ M + k2 + 2 = 27 * k2)) ∧
  (∃ k3 : ℕ, (M + k3 = 125 * k3 ∨ M + k3 + 1 = 125 * k3 ∨ M + k3 + 2 = 125 * k3)) :=
by
  sorry

end smallest_M_exists_l247_247737


namespace crayons_produced_l247_247129

theorem crayons_produced (colors : ℕ) (crayons_per_color : ℕ) (boxes_per_hour : ℕ) (hours : ℕ) 
  (h_colors : colors = 4) (h_crayons_per_color : crayons_per_color = 2) 
  (h_boxes_per_hour : boxes_per_hour = 5) (h_hours : hours = 4) : 
  colors * crayons_per_color * boxes_per_hour * hours = 160 := 
by
  rw [h_colors, h_crayons_per_color, h_boxes_per_hour, h_hours]
  norm_num

end crayons_produced_l247_247129


namespace factorize_cubic_l247_247879

theorem factorize_cubic (a : ℝ) : a^3 - a = a * (a + 1) * (a - 1) :=
sorry

end factorize_cubic_l247_247879


namespace solve_for_nabla_l247_247903

theorem solve_for_nabla (nabla : ℤ) (h : 5 * (-4) = nabla + 4) : nabla = -24 :=
by {
  sorry
}

end solve_for_nabla_l247_247903


namespace fraction_exponentiation_and_multiplication_l247_247995

theorem fraction_exponentiation_and_multiplication :
  ( (2 : ℚ) / 3 ) ^ 3 * (1 / 4) = 2 / 27 :=
by
  sorry

end fraction_exponentiation_and_multiplication_l247_247995


namespace probability_major_A_less_than_25_l247_247322

def total_students : ℕ := 100 -- assuming a total of 100 students for simplicity

def male_percent : ℝ := 0.40
def major_A_percent : ℝ := 0.50
def major_B_percent : ℝ := 0.30
def major_C_percent : ℝ := 0.20
def major_A_25_or_older_percent : ℝ := 0.60
def major_A_less_than_25_percent : ℝ := 1 - major_A_25_or_older_percent

theorem probability_major_A_less_than_25 :
  (major_A_percent * major_A_less_than_25_percent) = 0.20 :=
by
  sorry

end probability_major_A_less_than_25_l247_247322


namespace find_max_marks_l247_247971

variable (M : ℕ) (P : ℕ)

theorem find_max_marks (h1 : M = 332) (h2 : P = 83) : 
  let Max_Marks := M / (P / 100)
  Max_Marks = 400 := 
by 
  sorry

end find_max_marks_l247_247971


namespace price_of_necklace_l247_247362

-- Define the necessary conditions.
def num_charms_per_necklace : ℕ := 10
def cost_per_charm : ℕ := 15
def num_necklaces_sold : ℕ := 30
def total_profit : ℕ := 1500

-- Calculation of selling price per necklace
def cost_per_necklace := num_charms_per_necklace * cost_per_charm
def total_cost := cost_per_necklace * num_necklaces_sold
def total_revenue := total_cost + total_profit
def selling_price_per_necklace := total_revenue / num_necklaces_sold

-- Statement of the problem in Lean 4
theorem price_of_necklace : selling_price_per_necklace = 200 := by
  sorry

end price_of_necklace_l247_247362


namespace min_sum_m_n_l247_247032

noncomputable def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem min_sum_m_n (m n : ℕ) (h : (binomial m 2) * 2 = binomial (m + n) 2) : m + n = 4 := by
  sorry

end min_sum_m_n_l247_247032


namespace final_problem_l247_247884

-- Define the function f
def f (x p q : ℝ) : ℝ := x * abs x + p * x + q

-- Proposition ①: When q=0, f(x) is an odd function
def prop1 (p : ℝ) : Prop :=
  ∀ x : ℝ, f x p 0 = - f (-x) p 0

-- Proposition ②: The graph of y=f(x) is symmetric with respect to the point (0,q)
def prop2 (p q : ℝ) : Prop :=
  ∀ x : ℝ, f x p q = f (-x) p q + 2 * q

-- Proposition ③: When p=0 and q > 0, the equation f(x)=0 has exactly one real root
def prop3 (q : ℝ) : Prop :=
  q > 0 → ∃! x : ℝ, f x 0 q = 0

-- Proposition ④: The equation f(x)=0 has at most two real roots
def prop4 (p q : ℝ) : Prop :=
  ∀ x1 x2 x3 : ℝ, f x1 p q = 0 ∧ f x2 p q = 0 ∧ f x3 p q = 0 → x1 = x2 ∨ x1 = x3 ∨ x2 = x3

-- The final problem to prove that propositions ①, ②, and ③ are true and proposition ④ is false
theorem final_problem (p q : ℝ) :
  prop1 p ∧ prop2 p q ∧ prop3 q ∧ ¬prop4 p q :=
sorry

end final_problem_l247_247884


namespace minutes_between_bathroom_visits_l247_247927

-- Definition of the conditions
def movie_duration_hours : ℝ := 2.5
def bathroom_uses : ℕ := 3
def minutes_per_hour : ℝ := 60

-- Theorem statement for the proof
theorem minutes_between_bathroom_visits :
  let total_movie_minutes := movie_duration_hours * minutes_per_hour
  let intervals := bathroom_uses + 1
  total_movie_minutes / intervals = 37.5 :=
by
  sorry

end minutes_between_bathroom_visits_l247_247927


namespace value_of_x_minus_y_l247_247309

theorem value_of_x_minus_y (x y : ℝ) 
  (h1 : |x| = 2) 
  (h2 : y^2 = 9) 
  (h3 : x + y < 0) : 
  x - y = 1 ∨ x - y = 5 := 
by 
  sorry

end value_of_x_minus_y_l247_247309


namespace instantaneous_velocity_at_t4_l247_247986

-- Definition of the motion equation
def s (t : ℝ) : ℝ := 1 - t + t^2

-- The proof problem statement: Proving that the derivative of s at t = 4 is 7
theorem instantaneous_velocity_at_t4 : deriv s 4 = 7 :=
by sorry

end instantaneous_velocity_at_t4_l247_247986


namespace projection_of_AB_on_AC_l247_247004

noncomputable def A : ℝ × ℝ := (-1, 1)
noncomputable def B : ℝ × ℝ := (0, 3)
noncomputable def C : ℝ × ℝ := (3, 4)

noncomputable def vectorAB := (B.1 - A.1, B.2 - A.2)
noncomputable def vectorAC := (C.1 - A.1, C.2 - A.2)

noncomputable def dotProduct (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem projection_of_AB_on_AC :
  (dotProduct vectorAB vectorAC) / (magnitude vectorAC) = 2 :=
  sorry

end projection_of_AB_on_AC_l247_247004


namespace cubic_sum_div_pqr_eq_three_l247_247932

theorem cubic_sum_div_pqr_eq_three (p q r : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (h : p + q + r = 6) :
  (p^3 + q^3 + r^3) / (p * q * r) = 3 := 
by
  sorry

end cubic_sum_div_pqr_eq_three_l247_247932


namespace pet_store_cages_l247_247980

def initial_puppies : ℕ := 18
def puppies_sold : ℕ := 3
def puppies_per_cage : ℕ := 5

def remaining_puppies : ℕ := initial_puppies - puppies_sold
def number_of_cages : ℕ := remaining_puppies / puppies_per_cage

theorem pet_store_cages : number_of_cages = 3 :=
by sorry

end pet_store_cages_l247_247980


namespace T_value_l247_247644

variable (x : ℝ)

def T : ℝ := (x-2)^4 + 4 * (x-2)^3 + 6 * (x-2)^2 + 4 * (x-2) + 1

theorem T_value : T x = (x-1)^4 := by
  sorry

end T_value_l247_247644


namespace science_students_count_l247_247034

def total_students := 400 + 120
def local_arts_students := 0.50 * 400
def local_commerce_students := 0.85 * 120
def total_local_students := 327

theorem science_students_count :
  0.25 * S = 25 →
  S = 100 :=
by
  sorry

end science_students_count_l247_247034


namespace subset_condition_l247_247783

def A (a : ℝ) : Set ℝ := {0, -a}
def B (a : ℝ) : Set ℝ := {1, a - 2, 2 * a - 2}

theorem subset_condition (a : ℝ) (h : A a ⊆ B a) : a = 1 :=
by
  sorry

end subset_condition_l247_247783


namespace Stonewall_marching_band_max_members_l247_247668

theorem Stonewall_marching_band_max_members (n : ℤ) (h1 : 30 * n % 34 = 2) (h2 : 30 * n < 1500) : 30 * n = 1260 :=
by
  sorry

end Stonewall_marching_band_max_members_l247_247668


namespace total_potatoes_brought_home_l247_247923

def number_of_potatoes_each : ℕ := 8

theorem total_potatoes_brought_home (jane_potatoes mom_potatoes dad_potatoes : ℕ) :
  jane_potatoes = number_of_potatoes_each →
  mom_potatoes = number_of_potatoes_each →
  dad_potatoes = number_of_potatoes_each →
  jane_potatoes + mom_potatoes + dad_potatoes = 24 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  exact rfl

end total_potatoes_brought_home_l247_247923


namespace ellipse_distance_CD_l247_247847

theorem ellipse_distance_CD :
  ∃ (CD : ℝ), 
    (∀ (x y : ℝ),
    4 * (x - 2)^2 + 16 * y^2 = 64) → 
      CD = 2*Real.sqrt 5 :=
by sorry

end ellipse_distance_CD_l247_247847


namespace cats_not_eating_either_l247_247772

/-- In a shelter with 80 cats, 15 cats like tuna, 60 cats like chicken, 
and 10 like both tuna and chicken, prove that 15 cats do not eat either. -/
theorem cats_not_eating_either (total_cats : ℕ) (like_tuna : ℕ) (like_chicken : ℕ) (like_both : ℕ)
    (h1 : total_cats = 80) (h2 : like_tuna = 15) (h3 : like_chicken = 60) (h4 : like_both = 10) :
    (total_cats - (like_tuna - like_both + like_chicken - like_both + like_both) = 15) := 
by
    sorry

end cats_not_eating_either_l247_247772


namespace part1_part2_l247_247002

open Real

noncomputable def f (x a : ℝ) : ℝ := exp x - x^a

theorem part1 (a : ℝ) : (∀ x > 0, f x a ≥ 0) → a ≤ exp 1 :=
sorry

theorem part2 (a x1 x2 : ℝ) (hx1 : 0 < x1) (hx2 : 0 < x2) (hx : x1 > x2) :
  f x1 a = 0 → f x2 a = 0 → x1 + x2 > 2 * a :=
sorry

end part1_part2_l247_247002


namespace english_vocab_related_to_reading_level_l247_247364

theorem english_vocab_related_to_reading_level (N : ℕ) (K_squared : ℝ) (critical_value : ℝ) (p_value : ℝ)
  (hN : N = 100)
  (hK_squared : K_squared = 7)
  (h_critical_value : critical_value = 6.635)
  (h_p_value : p_value = 0.010) :
  p_value <= 0.01 → K_squared > critical_value → true :=
by
  intro h_p_value_le h_K_squared_gt
  sorry

end english_vocab_related_to_reading_level_l247_247364


namespace net_gain_difference_l247_247701

def first_applicant_salary : ℝ := 42000
def first_applicant_training_cost_per_month : ℝ := 1200
def first_applicant_training_months : ℝ := 3
def first_applicant_revenue : ℝ := 93000

def second_applicant_salary : ℝ := 45000
def second_applicant_hiring_bonus_percentage : ℝ := 0.01
def second_applicant_revenue : ℝ := 92000

def first_applicant_total_cost : ℝ := first_applicant_salary + first_applicant_training_cost_per_month * first_applicant_training_months
def first_applicant_net_gain : ℝ := first_applicant_revenue - first_applicant_total_cost

def second_applicant_hiring_bonus : ℝ := second_applicant_salary * second_applicant_hiring_bonus_percentage
def second_applicant_total_cost : ℝ := second_applicant_salary + second_applicant_hiring_bonus
def second_applicant_net_gain : ℝ := second_applicant_revenue - second_applicant_total_cost

theorem net_gain_difference :
  first_applicant_net_gain - second_applicant_net_gain = 850 := by
  sorry

end net_gain_difference_l247_247701


namespace first_year_after_2023_with_digit_sum_8_l247_247915

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem first_year_after_2023_with_digit_sum_8 : ∃ (y : ℕ), y > 2023 ∧ sum_of_digits y = 8 ∧ ∀ z, (z > 2023 ∧ sum_of_digits z = 8) → y ≤ z :=
by sorry

end first_year_after_2023_with_digit_sum_8_l247_247915


namespace width_of_foil_covered_prism_l247_247089

theorem width_of_foil_covered_prism (L W H : ℝ) 
    (hW1 : W = 2 * L)
    (hW2 : W = 2 * H)
    (hvol : L * W * H = 128) :
    W + 2 = 8 := 
sorry

end width_of_foil_covered_prism_l247_247089


namespace sum_not_prime_30_l247_247699

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem sum_not_prime_30 (p1 p2 : ℕ) (hp1 : is_prime p1) (hp2 : is_prime p2) (h : p1 + p2 = 30) : false :=
sorry

end sum_not_prime_30_l247_247699


namespace range_of_x_given_p_and_q_range_of_m_given_neg_q_sufficient_for_neg_p_l247_247933

variable {x m : ℝ}

-- First statement: Given m = 4 and p ∧ q, prove the range of x is 4 < x < 5
theorem range_of_x_given_p_and_q (m : ℝ) (h : m = 4) :
  (x^2 - 7*x + 10 < 0) ∧ (x^2 - 4*m*x + 3*m^2 < 0) → (4 < x ∧ x < 5) :=
sorry

-- Second statement: Prove the range of m given ¬q is a sufficient but not necessary condition for ¬p
theorem range_of_m_given_neg_q_sufficient_for_neg_p :
  (m ≤ 2) ∧ (3*m ≥ 5) ∧ (m > 0) → (5/3 ≤ m ∧ m ≤ 2) :=
sorry

end range_of_x_given_p_and_q_range_of_m_given_neg_q_sufficient_for_neg_p_l247_247933


namespace number_line_y_l247_247164

theorem number_line_y (step_length : ℕ) (steps_total : ℕ) (total_distance : ℕ) (y_step : ℕ) (y : ℕ) 
    (H1 : steps_total = 6) 
    (H2 : total_distance = 24) 
    (H3 : y_step = 4)
    (H4 : step_length = total_distance / steps_total) 
    (H5 : y = step_length * y_step) : 
  y = 16 := 
  by 
    sorry

end number_line_y_l247_247164


namespace total_weight_of_onions_l247_247537

def weight_per_bag : ℕ := 50
def bags_per_trip : ℕ := 10
def trips : ℕ := 20

theorem total_weight_of_onions : bags_per_trip * weight_per_bag * trips = 10000 := by
  sorry

end total_weight_of_onions_l247_247537


namespace profit_is_eight_dollars_l247_247710

-- Define the given quantities and costs
def total_bracelets : ℕ := 52
def bracelets_given_away : ℕ := 8
def cost_of_materials : ℝ := 3.00
def selling_price_per_bracelet : ℝ := 0.25

-- Define the number of bracelets sold
def bracelets_sold := total_bracelets - bracelets_given_away

-- Calculate the total money earned from selling the bracelets
def total_earnings := bracelets_sold * selling_price_per_bracelet

-- Calculate the profit made by Alice
def profit := total_earnings - cost_of_materials

-- Prove that the profit is $8.00
theorem profit_is_eight_dollars : profit = 8.00 := by
  sorry

end profit_is_eight_dollars_l247_247710


namespace cone_shorter_height_ratio_l247_247839

theorem cone_shorter_height_ratio 
  (circumference : ℝ) (original_height : ℝ) (volume_shorter_cone : ℝ) 
  (shorter_height : ℝ) (radius : ℝ) :
  circumference = 24 * Real.pi ∧ 
  original_height = 40 ∧ 
  volume_shorter_cone = 432 * Real.pi ∧ 
  2 * Real.pi * radius = circumference ∧ 
  volume_shorter_cone = (1 / 3) * Real.pi * radius^2 * shorter_height
  → shorter_height / original_height = 9 / 40 :=
by
  sorry

end cone_shorter_height_ratio_l247_247839


namespace irrational_of_sqrt_3_l247_247261

theorem irrational_of_sqrt_3 :
  ¬ (∃ (a b : ℤ), b ≠ 0 ∧ ↑a / ↑b = Real.sqrt 3) :=
sorry

end irrational_of_sqrt_3_l247_247261


namespace find_N_l247_247652

theorem find_N (N : ℕ) (h₁ : ∃ (d₁ d₂ : ℕ), d₁ + d₂ = 3333 ∧ N = max d₁ d₂ ∧ (max d₁ d₂) / (min d₁ d₂) = 2) : 
  N = 2222 := sorry

end find_N_l247_247652


namespace analysis_duration_unknown_l247_247991

-- Definitions based on the given conditions
def number_of_bones : Nat := 206
def analysis_duration_per_bone (bone: Nat) : Nat := 5  -- assumed fixed for simplicity
-- Time spent analyzing all bones (which needs more information to be accurately known)
def total_analysis_time (bones_analyzed: Nat) (hours_per_bone: Nat) : Nat := bones_analyzed * hours_per_bone

-- Given the number of bones and duration per bone, there isn't enough information to determine the total analysis duration
theorem analysis_duration_unknown (total_bones : Nat) (duration_per_bone : Nat) (bones_remaining: Nat) (analysis_already_done : Nat) :
  total_bones = number_of_bones →
  (∀ bone, analysis_duration_per_bone bone = duration_per_bone) →
  analysis_already_done ≠ (total_bones - bones_remaining) ->
  ∃ hours_needed, hours_needed = total_analysis_time (total_bones - bones_remaining) duration_per_bone :=
by
  intros
  sorry

end analysis_duration_unknown_l247_247991


namespace draw_at_least_two_first_grade_products_l247_247577

theorem draw_at_least_two_first_grade_products :
  let total_products := 9
  let first_grade := 4
  let second_grade := 3
  let third_grade := 2
  let total_draws := 4
  let ways_to_draw := Nat.choose total_products total_draws
  let ways_no_first_grade := Nat.choose (second_grade + third_grade) total_draws
  let ways_one_first_grade := Nat.choose first_grade 1 * Nat.choose (second_grade + third_grade) (total_draws - 1)
  ways_to_draw - ways_no_first_grade - ways_one_first_grade = 81 := sorry

end draw_at_least_two_first_grade_products_l247_247577


namespace trig_identity_l247_247245

open Real

theorem trig_identity :
  3.4173 * sin (2 * pi / 17) + sin (4 * pi / 17) - sin (6 * pi / 17) - (1/2) * sin (8 * pi / 17) =
  8 * (sin (2 * pi / 17))^3 * (cos (pi / 17))^2 :=
by sorry

end trig_identity_l247_247245


namespace geometric_series_second_term_l247_247580

theorem geometric_series_second_term 
  (r : ℚ) (S : ℚ) (a : ℚ) (second_term : ℚ)
  (h1 : r = 1 / 4)
  (h2 : S = 16)
  (h3 : S = a / (1 - r))
  : second_term = a * r := 
sorry

end geometric_series_second_term_l247_247580


namespace gcd_lcm_product_eq_prod_l247_247065

theorem gcd_lcm_product_eq_prod (a b : ℕ) : Nat.gcd a b * Nat.lcm a b = a * b := 
sorry

end gcd_lcm_product_eq_prod_l247_247065


namespace factorial_eq_l247_247727

theorem factorial_eq (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (Nat.factorial a) * (Nat.factorial b) = (Nat.factorial a) + (Nat.factorial b) + (Nat.factorial c) → 
  (a = 3 ∧ b = 3 ∧ c = 4) := by
  sorry

end factorial_eq_l247_247727


namespace sum_eq_prod_S1_sum_eq_prod_S2_l247_247442

def S1 : List ℕ := [1, 1, 1, 1, 1, 1, 2, 8]
def S2 : List ℕ := [1, 1, 1, 1, 1, 2, 2, 3]

def sum_list (l : List ℕ) : ℕ := l.foldr Nat.add 0
def prod_list (l : List ℕ) : ℕ := l.foldr Nat.mul 1

theorem sum_eq_prod_S1 : sum_list S1 = prod_list S1 := 
by
  sorry

theorem sum_eq_prod_S2 : sum_list S2 = prod_list S2 := 
by
  sorry

end sum_eq_prod_S1_sum_eq_prod_S2_l247_247442


namespace set_D_is_empty_l247_247263

theorem set_D_is_empty :
  {x : ℝ | x > 6 ∧ x < 1} = ∅ :=
by
  sorry

end set_D_is_empty_l247_247263


namespace power_division_l247_247094

theorem power_division (h : 64 = 8^2) : 8^15 / 64^7 = 8 :=
by
  rw [h]
  rw [pow_mul]
  rw [div_eq_mul_inv]
  rw [pow_sub]
  rw [mul_inv_cancel]
  exact rfl

end power_division_l247_247094


namespace max_varphi_l247_247512

noncomputable def f (x φ : ℝ) : ℝ := 2 * Real.sin (2 * x + φ)
noncomputable def g (x φ : ℝ) : ℝ := 2 * Real.sin (2 * x + φ + (2 * Real.pi / 3))

theorem max_varphi (φ : ℝ) (h : φ < 0) (hE : ∀ x, g x φ = g (-x) φ) : φ = -Real.pi / 6 :=
by
  sorry

end max_varphi_l247_247512


namespace greatest_multiple_of_5_and_6_less_than_1000_l247_247389

theorem greatest_multiple_of_5_and_6_less_than_1000 : ∃ n, (n % 5 = 0) ∧ (n % 6 = 0) ∧ (n < 1000) ∧ (∀ m, (m % 5 = 0) ∧ (m % 6 = 0) ∧ (m < 1000) → m ≤ n) ∧ n = 990 :=
by sorry

end greatest_multiple_of_5_and_6_less_than_1000_l247_247389


namespace number_of_large_balls_l247_247540

def smallBallRubberBands : ℕ := 50
def largeBallRubberBands : ℕ := 300
def totalRubberBands : ℕ := 5000
def smallBallsMade : ℕ := 22

def rubberBandsUsedForSmallBalls := smallBallsMade * smallBallRubberBands
def remainingRubberBands := totalRubberBands - rubberBandsUsedForSmallBalls

theorem number_of_large_balls :
  (remainingRubberBands / largeBallRubberBands) = 13 := by
  sorry

end number_of_large_balls_l247_247540


namespace jordan_rectangle_width_l247_247144

theorem jordan_rectangle_width
  (w : ℝ)
  (len_carol : ℝ := 5)
  (wid_carol : ℝ := 24)
  (len_jordan : ℝ := 12)
  (area_carol_eq_area_jordan : (len_carol * wid_carol) = (len_jordan * w)) :
  w = 10 := by
  sorry

end jordan_rectangle_width_l247_247144


namespace smallest_munificence_monic_cubic_polynomial_l247_247285

theorem smallest_munificence_monic_cubic_polynomial :
  ∃ (f : ℝ → ℝ), (∀ (x : ℝ), f x = x^3 + a * x^2 + b * x + c) ∧
  (∀ (x : ℝ), -1 ≤ x ∧ x ≤ 1 → |f x| ≤ 1) ∧
  (∀ (M : ℝ), (∀ (x : ℝ), -1 ≤ x ∧ x ≤ 1 → |f x| ≤ M) → M ≥ 1) :=
by
  sorry

end smallest_munificence_monic_cubic_polynomial_l247_247285


namespace smallest_n_l247_247910

theorem smallest_n 
    (h1 : ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a * r = b ∧ b * r = c ∧ 7 * n + 1 = a + b + c)
    (h2 : ∃ (x y z : ℕ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x * s = y ∧ y * s = z ∧ 8 * n + 1 = x + y + z) :
    n = 22 :=
sorry

end smallest_n_l247_247910


namespace sum_incircle_radii_invariant_l247_247053

variables (A B C D E : Point)
  
-- Definition of convex, inscriptible pentagon
def is_convex_inscriptible_pentagon (A B C D E : Point) : Prop :=
convex_poly ABCDE ∧ inscriptible_poly ABCDE

-- Definition of triangulations
def triangulations (A B C D E : Point) : list (Triangle × Triangle × Triangle) :=
  list.triangulations_of (Pentagon.mk A B C D E)

-- Radius of the incircle of a triangle
def incircle_radius (t : Triangle) : Real :=
t.incircle.radius

-- Sum of incircle radii for a triangulation
def sum_incircle_radii (triang : Triangle × Triangle × Triangle) : Real :=
incircle_radius triang.1 + incircle_radius triang.2 + incircle_radius triang.3

-- Main theorem statement
theorem sum_incircle_radii_invariant (h : is_convex_inscriptible_pentagon A B C D E) (t1 t2 : triangulations A B C D E) :
  sum_incircle_radii t1 = sum_incircle_radii t2 :=
sorry

end sum_incircle_radii_invariant_l247_247053


namespace paper_area_difference_l247_247761

def area (length width : ℕ) : ℕ := length * width

def combined_area (length width : ℕ) : ℕ := 2 * (area length width)

def sq_inch_to_sq_ft (sq_inch : ℕ) : ℕ := sq_inch / 144

theorem paper_area_difference :
  sq_inch_to_sq_ft (combined_area 15 24 - combined_area 12 18) = 2 :=
by
  sorry

end paper_area_difference_l247_247761


namespace total_wheels_l247_247244

-- Definitions of given conditions
def bicycles : ℕ := 50
def tricycles : ℕ := 20
def wheels_per_bicycle : ℕ := 2
def wheels_per_tricycle : ℕ := 3

-- Theorem stating the total number of wheels for bicycles and tricycles combined
theorem total_wheels : bicycles * wheels_per_bicycle + tricycles * wheels_per_tricycle = 160 :=
by
  sorry

end total_wheels_l247_247244


namespace no_savings_if_purchased_together_l247_247137

def window_price : ℕ := 120

def free_windows (purchased_windows : ℕ) : ℕ :=
  (purchased_windows / 10) * 2

def total_cost (windows_needed : ℕ) : ℕ :=
  (windows_needed - free_windows windows_needed) * window_price

def separate_cost : ℕ :=
  total_cost 9 + total_cost 11 + total_cost 10

def joint_cost : ℕ :=
  total_cost 30

theorem no_savings_if_purchased_together :
  separate_cost = joint_cost :=
by
  -- Proof will be provided here, currently skipped.
  sorry

end no_savings_if_purchased_together_l247_247137


namespace prob_math_page_l247_247967

/-- Define the total number of pages in Xiao Ming's folder. -/
def total_pages : ℕ := 12

/-- Define the number of Mathematics test pages in Xiao Ming's folder. -/
def math_pages : ℕ := 2

/-- Calculate the probability of drawing a Mathematics test paper. -/
theorem prob_math_page :
  (math_pages : ℚ) / total_pages = 1 / 6 := by
    sorry

end prob_math_page_l247_247967


namespace probability_of_independent_events_l247_247524

namespace Probability

variable (A B : Prop)
variable [pA : Decidable A] [pB : Decidable B]

noncomputable def P : Prop → ℚ := sorry

theorem probability_of_independent_events
  (hA : P A = 5/7)
  (hB : P B = 4/5)
  (hIndep : P (A ∧ B) = P A * P B) :
  P (A ∧ B) = 4/7 :=
by
  rw [hA, hB]
  rw [hIndep]
  norm_num
  sorry

end Probability

end probability_of_independent_events_l247_247524


namespace max_n_value_is_9_l247_247162

variable (a b c d n : ℝ)
variable (h1 : a > b)
variable (h2 : b > c)
variable (h3 : c > d)
variable (h : (1 / (a - b)) + (1 / (b - c)) + (1 / (c - d)) ≥ (n / (a - d)))

theorem max_n_value_is_9 (h1 : a > b) (h2 : b > c) (h3 : c > d)
    (h : (1 / (a - b)) + (1 / (b - c)) + (1 / (c - d)) ≥ (n / (a - d))) : n ≤ 9 :=
sorry

end max_n_value_is_9_l247_247162


namespace ravenswood_forest_percentage_l247_247518

def ravenswood_gnomes (westerville_gnomes : ℕ) : ℕ := 4 * westerville_gnomes
def remaining_gnomes (total_gnomes taken_percentage: ℕ) : ℕ := (total_gnomes * (100 - taken_percentage)) / 100

theorem ravenswood_forest_percentage:
  ∀ (westerville_gnomes : ℕ) (remaining : ℕ) (total_gnomes : ℕ),
  westerville_gnomes = 20 →
  total_gnomes = ravenswood_gnomes westerville_gnomes →
  remaining = 48 →
  remaining_gnomes total_gnomes 40 = remaining :=
by
  sorry

end ravenswood_forest_percentage_l247_247518


namespace avg_age_new_students_l247_247520

-- Definitions for the conditions
def initial_avg_age : ℕ := 14
def initial_student_count : ℕ := 10
def new_student_count : ℕ := 5
def new_avg_age : ℕ := initial_avg_age + 1

-- Lean statement for the proof problem
theorem avg_age_new_students :
  (initial_avg_age * initial_student_count + new_avg_age * new_student_count) / new_student_count = 17 :=
by
  sorry

end avg_age_new_students_l247_247520


namespace factorize_a_cubed_minus_a_l247_247860

theorem factorize_a_cubed_minus_a (a : ℝ) : a^3 - a = a * (a + 1) * (a - 1) :=
by
  sorry

end factorize_a_cubed_minus_a_l247_247860


namespace cookie_sheet_perimeter_l247_247123

def width : ℕ := 10
def length : ℕ := 2

def perimeter (w l : ℕ) : ℕ := 2 * w + 2 * l

theorem cookie_sheet_perimeter : 
  perimeter width length = 24 := by
  sorry

end cookie_sheet_perimeter_l247_247123


namespace symmetric_point_product_l247_247184

theorem symmetric_point_product (x y : ℤ) (h1 : (2008, y) = (-x, -1)) : x * y = -2008 :=
by {
  sorry
}

end symmetric_point_product_l247_247184


namespace fraction_of_women_married_l247_247408

-- Definitions for the conditions
def total_employees : ℕ := 100
def women_percent : ℚ := 61 / 100
def married_percent : ℚ := 60 / 100
def men_single_fraction : ℚ := 2 / 3

-- Calculate number of employees corresponding to conditions
def women : ℕ := (women_percent * total_employees).natAbs
def men : ℕ := total_employees - women
def married : ℕ := (married_percent * total_employees).natAbs

def men_married : ℕ := ((1 - men_single_fraction) * men).natAbs
def women_married : ℕ := married - men_married

-- Final fraction to prove
def fraction_women_married : ℚ := women_married / women

-- Lean theorem statement, ensuring conditions and goal
theorem fraction_of_women_married : fraction_women_married = 47 / 61 :=
by
  sorry

end fraction_of_women_married_l247_247408


namespace total_tape_length_l247_247429

-- Definitions based on the problem conditions
def first_side_songs : ℕ := 6
def second_side_songs : ℕ := 4
def song_length : ℕ := 4

-- Statement to prove the total tape length is 40 minutes
theorem total_tape_length : (first_side_songs + second_side_songs) * song_length = 40 := by
  sorry

end total_tape_length_l247_247429


namespace evaluate_combinations_l247_247725

theorem evaluate_combinations (n : ℕ) (h1 : 0 ≤ 5 - n) (h2 : 5 - n ≤ n) (h3 : 0 ≤ 10 - n) (h4 : 10 - n ≤ n + 1) (h5 : n > 0) :
  Nat.choose n (5 - n) + Nat.choose (n + 1) (10 - n) = 7 :=
sorry

end evaluate_combinations_l247_247725


namespace sector_angle_radian_measure_l247_247297

theorem sector_angle_radian_measure (r l : ℝ) (h1 : r = 1) (h2 : l = 2) : l / r = 2 := by
  sorry

end sector_angle_radian_measure_l247_247297


namespace f_neg_a_l247_247944

noncomputable def f (x : ℝ) : ℝ := x^3 + Real.sin x + 1

theorem f_neg_a (a : ℝ) (h : f a = 2) : f (-a) = 0 :=
by
  sorry

end f_neg_a_l247_247944


namespace range_a_l247_247618

def A : Set ℝ :=
  {x | x^2 + 5 * x + 6 ≤ 0}

def B : Set ℝ :=
  {x | -3 ≤ x ∧ x ≤ 5}

def C (a : ℝ) : Set ℝ :=
  {x | a < x ∧ x < a + 1}

theorem range_a (a : ℝ) : ((A ∪ B) ∩ C a = ∅) → (a ≥ 5 ∨ a ≤ -4) :=
  sorry

end range_a_l247_247618


namespace min_value_3x_plus_4y_l247_247475

theorem min_value_3x_plus_4y (x y : ℝ) (h_pos : 0 < x ∧ 0 < y) (h_eq : x + 3*y = 5*x*y) :
  ∃ (c : ℝ), (∀ x y : ℝ, 0 < x ∧ 0 < y ∧ x + 3 * y = 5 * x * y → 3 * x + 4 * y ≥ c) ∧ c = 5 :=
sorry

end min_value_3x_plus_4y_l247_247475


namespace coin_flip_sequences_l247_247742

theorem coin_flip_sequences (n : ℕ) (h_pos : 0 < n) (p : ℚ) (h_p : 0 < p ∧ p < 1) :
  (∃ a : ℕ, ∃ b : ℕ, a < b ∧ b ≠ 0 ∧ p = a / b ∧ gcd(a, b) = 1) ∧
  (∃ a_r : fin (n + 1) → ℕ, (∀ r, 0 ≤ a_r r ∧ a_r r ≤ nat.choose n r) ∧
     (∑ r in finset.range (n + 1), (a_r r) * (p ^ r) * ((1 - p) ^ (n - r))) = 1 / 2)
    → p = 1 / 2 :=
begin
  sorry
end

end coin_flip_sequences_l247_247742


namespace complex_purely_imaginary_l247_247317

theorem complex_purely_imaginary (a : ℂ) (h1 : a^2 - 3 * a + 2 = 0) (h2 : a - 1 ≠ 0) : a = 2 :=
sorry

end complex_purely_imaginary_l247_247317


namespace equation_of_ellipse_HN_passes_through_fixed_point_l247_247169

-- Definitions of points
def A : ℝ × ℝ := (0, -2)
def B : ℝ × ℝ := (3/2, -1)
def P : ℝ × ℝ := (1, -2)

-- Center and axes of the ellipse
def center : ℝ × ℝ := (0, 0)
def x_axis_symmetry := True
def y_axis_symmetry := True

-- The ellipse passes through A and B
def ellipse (x y : ℝ) : Prop := (3 * x^2 + 4 * y^2 = 12)

theorem equation_of_ellipse :
  ellipse A.1 A.2 ∧ ellipse B.1 B.2 :=
begin
  split;
  unfold ellipse;
  norm_num,
end

theorem HN_passes_through_fixed_point :
  ∃ K : ℝ × ℝ, (K = (0, -2)) ∧
  ∀ (M N T H : ℝ × ℝ), (ellipse M.1 M.2) ∧ (ellipse N.1 N.2) ∧ 
  (∃ (k : ℝ), N.1 = k * M.1 ∧ N.2 = k * M.2) ∧ -- line MN
  (T.1 = M.1) ∧ (∃ (yT : ℝ), (T.1, yT) = T) ∧ -- T on line AB
  (H.1 = 2 * M.1 - T.1) ∧ (H.2 = 2 * M.2 - T.2) -> -- H's coordinates
  ((N.1 - H.1) = 0∧ (N.2 - H.2) = -2) := -- checking HN passes through (0,-2)
sorry

end equation_of_ellipse_HN_passes_through_fixed_point_l247_247169


namespace largest_4_digit_congruent_15_mod_22_l247_247240

theorem largest_4_digit_congruent_15_mod_22 :
  ∃ (x : ℤ), x < 10000 ∧ x % 22 = 15 ∧ (∀ (y : ℤ), y < 10000 ∧ y % 22 = 15 → y ≤ x) → x = 9981 :=
sorry

end largest_4_digit_congruent_15_mod_22_l247_247240


namespace distance_from_y_axis_l247_247687

theorem distance_from_y_axis (x : ℝ) : abs x = 10 :=
by
  -- Define distances
  let d_x := 5
  let d_y := abs x
  -- Given condition
  have h : d_x = (1 / 2) * d_y := sorry
  -- Use the given condition to prove the required statement
  sorry

end distance_from_y_axis_l247_247687


namespace frank_spend_more_l247_247674

noncomputable def table_cost : ℝ := 140
noncomputable def chair_cost : ℝ := 100
noncomputable def joystick_cost : ℝ := 20
noncomputable def frank_joystick : ℝ := joystick_cost * (1 / 4)
noncomputable def eman_joystick : ℝ := joystick_cost - frank_joystick
noncomputable def frank_total : ℝ := table_cost + frank_joystick
noncomputable def eman_total : ℝ := chair_cost + eman_joystick

theorem frank_spend_more :
  frank_total - eman_total = 30 :=
  sorry

end frank_spend_more_l247_247674


namespace hand_position_at_8PM_yesterday_l247_247420

-- Define the conditions of the problem
def positions : ℕ := 20
def jump_interval_min : ℕ := 7
def jump_positions : ℕ := 9
def start_position : ℕ := 0
def end_position : ℕ := 8 -- At 8:00 AM, the hand is at position 9, hence moving forward 8 positions from position 0

-- Define the total time from 8:00 PM yesterday to 8:00 AM today
def total_minutes : ℕ := 720

-- Calculate the number of full jumps
def num_full_jumps : ℕ := total_minutes / jump_interval_min

-- Calculate the hand's final position from 8:00 PM yesterday
def final_hand_position : ℕ := (start_position + num_full_jumps * jump_positions) % positions

-- Prove that the final hand position is 2
theorem hand_position_at_8PM_yesterday : final_hand_position = 2 :=
by
  sorry

end hand_position_at_8PM_yesterday_l247_247420


namespace polar_distance_l247_247041

theorem polar_distance {r1 θ1 r2 θ2 : ℝ} (A : r1 = 1 ∧ θ1 = π/6) (B : r2 = 3 ∧ θ2 = 5*π/6) : 
  (r1^2 + r2^2 - 2*r1*r2 * Real.cos (θ2 - θ1)) = 13 :=
  sorry

end polar_distance_l247_247041


namespace combined_basketballs_l247_247349

-- Conditions as definitions
def spursPlayers := 22
def rocketsPlayers := 18
def basketballsPerPlayer := 11

-- Math Proof Problem statement
theorem combined_basketballs : 
  (spursPlayers * basketballsPerPlayer) + (rocketsPlayers * basketballsPerPlayer) = 440 :=
by
  sorry

end combined_basketballs_l247_247349


namespace range_of_m_l247_247182

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, (x < m - 1 ∨ x > m + 1) → (x^2 - 2*x - 3 > 0)) ∧ 
  ¬(∀ x : ℝ, (x < m - 1 ∨ x > m + 1) ↔ (x^2 - 2*x - 3 > 0)) 
  ↔ 0 ≤ m ∧ m ≤ 2 :=
by 
  sorry

end range_of_m_l247_247182


namespace track_circumference_l247_247827

theorem track_circumference (A_speed B_speed : ℝ) (y : ℝ) (c : ℝ)
  (A_initial B_initial : ℝ := 0)
  (B_meeting_distance_A_first_meeting : ℝ := 150)
  (A_meeting_distance_B_second_meeting : ℝ := y - 150)
  (A_second_distance : ℝ := 2 * y - 90)
  (B_second_distance : ℝ := y + 90) 
  (first_meeting_eq : B_meeting_distance_A_first_meeting = 150)
  (second_meeting_eq : A_second_distance + 90 = 2 * y)
  (uniform_speed : A_speed / B_speed = (y + 90)/(2 * y - 90)) :
  c = 2 * y → c = 720 :=
by
  sorry

end track_circumference_l247_247827


namespace a_n_is_arithmetic_sequence_b_n_is_right_sequence_sum_first_n_terms_b_n_l247_247749

noncomputable def a_n (n : ℕ) : ℕ := 3 * n

noncomputable def b_n (n : ℕ) : ℕ := 3 * n + 2^(n - 1)

noncomputable def S_n (n : ℕ) : ℕ := (3 * n * (n + 1) / 2) + (2^n - 1)

theorem a_n_is_arithmetic_sequence (n : ℕ) :
  (a_n 1 = 3) ∧ (a_n 4 = 12) ∧ (∀ n : ℕ, a_n n = 3 * n) :=
by
  sorry

theorem b_n_is_right_sequence (n : ℕ) :
  (b_n 1 = 4) ∧ (b_n 4 = 20) ∧ (∀ n : ℕ, b_n n = 3 * n + 2^(n - 1)) ∧ 
  (∀ n : ℕ, b_n n - a_n n = 2^(n - 1)) :=
by
  sorry

theorem sum_first_n_terms_b_n (n : ℕ) :
  S_n n = 3 * (n * (n + 1) / 2) + 2^n - 1 :=
by
  sorry

end a_n_is_arithmetic_sequence_b_n_is_right_sequence_sum_first_n_terms_b_n_l247_247749


namespace ratio_pentagon_rectangle_l247_247568

-- Definitions of conditions.
def pentagon_side_length (p : ℕ) : Prop := 5 * p = 30
def rectangle_width (w : ℕ) : Prop := 6 * w = 30

-- The theorem to prove.
theorem ratio_pentagon_rectangle (p w : ℕ) (h1 : pentagon_side_length p) (h2 : rectangle_width w) :
  p / w = 6 / 5 :=
by sorry

end ratio_pentagon_rectangle_l247_247568


namespace line_equation_l247_247223

theorem line_equation {a b c : ℝ} (x : ℝ) (y : ℝ)
  (point : ∃ p: ℝ × ℝ, p = (-1, 0))
  (perpendicular : ∀ k: ℝ, k = 1 → 
    ∀ m: ℝ, m = -1 → 
      ∀ b1: ℝ, b1 = 0 → 
        ∀ x1: ℝ, x1 = -1 →
          ∀ y1: ℝ, y1 = 0 →
            ∀ l: ℝ, l = b1 + k * (x1 - (-1)) + m * (y1 - 0) → 
              x - y + 1 = 0) :
  x - y + 1 = 0 :=
sorry

end line_equation_l247_247223


namespace factorize_cubic_expression_l247_247872

variable (a : ℝ)

theorem factorize_cubic_expression : a^3 - a = a * (a + 1) * (a - 1) := 
sorry

end factorize_cubic_expression_l247_247872


namespace school_starts_at_8_l247_247581

def minutes_to_time (minutes : ℕ) : ℕ × ℕ :=
  let hour := minutes / 60
  let minute := minutes % 60
  (hour, minute)

def add_minutes_to_time (h : ℕ) (m : ℕ) (added_minutes : ℕ) : ℕ × ℕ :=
  minutes_to_time ((h * 60 + m) + added_minutes)

def subtract_minutes_from_time (h : ℕ) (m : ℕ) (subtracted_minutes : ℕ) : ℕ × ℕ :=
  minutes_to_time ((h * 60 + m) - subtracted_minutes)

theorem school_starts_at_8 : True := by
  let normal_commute := 30
  let red_light_stops := 3 * 4
  let construction_delay := 10
  let total_additional_time := red_light_stops + construction_delay
  let total_commute_time := normal_commute + total_additional_time
  let depart_time := (7, 15)
  let arrival_time := add_minutes_to_time depart_time.1 depart_time.2 total_commute_time
  let start_time := subtract_minutes_from_time arrival_time.1 arrival_time.2 7

  have : start_time = (8, 0) := by
    sorry

  exact trivial

end school_starts_at_8_l247_247581


namespace divisibility_by_5_l247_247689

theorem divisibility_by_5 (x y : ℤ) (h : 5 ∣ (x + 9 * y)) : 5 ∣ (8 * x + 7 * y) :=
sorry

end divisibility_by_5_l247_247689


namespace probability_two_boys_l247_247331

def number_of_students : ℕ := 5
def number_of_boys : ℕ := 2
def number_of_girls : ℕ := 3
def total_pairs : ℕ := Nat.choose number_of_students 2
def boys_pairs : ℕ := Nat.choose number_of_boys 2

theorem probability_two_boys :
  number_of_students = 5 →
  number_of_boys = 2 →
  number_of_girls = 3 →
  (boys_pairs : ℝ) / (total_pairs : ℝ) = 1 / 10 :=
by
  sorry

end probability_two_boys_l247_247331


namespace simplify_333_div_9999_mul_99_l247_247069

theorem simplify_333_div_9999_mul_99 :
  (333 / 9999) * 99 = 37 / 101 :=
by
  -- Sorry for skipping proof
  sorry

end simplify_333_div_9999_mul_99_l247_247069


namespace smallest_pos_n_l247_247110

theorem smallest_pos_n (n : ℕ) (h : 435 * n % 30 = 867 * n % 30) : n = 5 :=
by
  sorry

end smallest_pos_n_l247_247110


namespace exponent_of_9_in_9_pow_7_l247_247365

theorem exponent_of_9_in_9_pow_7 : ∀ x : ℕ, (3 ^ x ∣ 9 ^ 7) ↔ x ≤ 14 := by
  sorry

end exponent_of_9_in_9_pow_7_l247_247365


namespace sequence_property_l247_247232

theorem sequence_property : 
  (∀ (a : ℕ → ℝ), a 1 = 2 ∧ (∀ n ≥ 1, a (n + 1) = a n + (2 * a n) / n) → a 200 = 40200) :=
by
  sorry

end sequence_property_l247_247232


namespace johnny_selection_process_l247_247630

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem johnny_selection_process : 
  binomial_coefficient 10 4 * binomial_coefficient 4 2 = 1260 :=
by
  sorry

end johnny_selection_process_l247_247630


namespace function_ordering_l247_247165

-- Definitions for the function and conditions
variable (f : ℝ → ℝ)

-- Assuming properties of the function
axiom odd_function : ∀ x, f (-x) = -f x
axiom periodicity : ∀ x, f (x + 4) = -f x
axiom increasing_on : ∀ ⦃x y⦄, 0 ≤ x → x < y → y ≤ 2 → f x < f y

-- Main theorem statement
theorem function_ordering : f (-25) < f 80 ∧ f 80 < f 11 :=
by 
  sorry

end function_ordering_l247_247165


namespace solution_set_of_tan_eq_two_l247_247527

open Real

theorem solution_set_of_tan_eq_two :
  {x | ∃ k : ℤ, x = k * π + (-1 : ℤ) ^ k * arctan 2} = {x | tan x = 2} :=
by
  sorry

end solution_set_of_tan_eq_two_l247_247527


namespace negation_of_proposition_l247_247025

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ ∃ x : ℝ, x^3 - x^2 + 1 > 0 := 
sorry

end negation_of_proposition_l247_247025


namespace total_students_l247_247572

theorem total_students (T : ℝ) 
  (h1 : 0.28 * T = 280) : 
  T = 1000 :=
by {
  sorry
}

end total_students_l247_247572


namespace xyz_problem_l247_247643

/-- Given x = 36^2 + 48^2 + 64^3 + 81^2, prove the following:
    - x is a multiple of 3. 
    - x is a multiple of 4.
    - x is a multiple of 9.
    - x is not a multiple of 16. 
-/
theorem xyz_problem (x : ℕ) (h_x : x = 36^2 + 48^2 + 64^3 + 81^2) :
  (x % 3 = 0) ∧ (x % 4 = 0) ∧ (x % 9 = 0) ∧ ¬(x % 16 = 0) := 
by
  have h1 : 36^2 = 1296 := by norm_num
  have h2 : 48^2 = 2304 := by norm_num
  have h3 : 64^3 = 262144 := by norm_num
  have h4 : 81^2 = 6561 := by norm_num
  have hx : x = 1296 + 2304 + 262144 + 6561 := by rw [h_x, h1, h2, h3, h4]
  sorry

end xyz_problem_l247_247643


namespace polynomial_divisible_l247_247850

theorem polynomial_divisible (p q : ℤ) (h_p : p = -26) (h_q : q = 25) :
  ∀ x : ℤ, (x^4 + p*x^2 + q) % (x^2 - 6*x + 5) = 0 :=
by
  sorry

end polynomial_divisible_l247_247850


namespace m_lt_n_l247_247887

theorem m_lt_n (a t : ℝ) (h : 0 < t ∧ t < 1) : 
  abs (Real.log (1 + t) / Real.log a) < abs (Real.log (1 - t) / Real.log a) :=
sorry

end m_lt_n_l247_247887


namespace smallest_perimeter_of_consecutive_even_triangle_l247_247399

theorem smallest_perimeter_of_consecutive_even_triangle (n : ℕ) :
  (2 * n + 2 * n + 2 > 2 * n + 4) ∧
  (2 * n + 2 * n + 4 > 2 * n + 2) ∧
  (2 * n + 2 + 2 * n + 4 > 2 * n) →
  2 * n + (2 * n + 2) + (2 * n + 4) = 18 :=
by 
  sorry

end smallest_perimeter_of_consecutive_even_triangle_l247_247399


namespace circle_condition_tangent_lines_right_angle_triangle_l247_247617

-- Part (1): Range of m for the equation to represent a circle
theorem circle_condition {m : ℝ} : 
  (∀ x y : ℝ, x^2 + y^2 - 2*x + 2*m*y + m^2 - 2*m - 2 = 0 →
  (m > -3 / 2)) :=
sorry

-- Part (2): Equation of tangent line to circle C
theorem tangent_lines {m : ℝ} (h : m = -1) : 
  ∀ x y : ℝ,
  ((x - 1)^2 + (y - 1)^2 = 1 →
  ((x = 2) ∨ (4*x - 3*y + 4 = 0))) :=
sorry

-- Part (3): Value of t for the line intersecting circle at a right angle
theorem right_angle_triangle {t : ℝ} :
  (∀ x y : ℝ, 
  (x + y + t = 0) →
  (t = -3 ∨ t = -1)) :=
sorry

end circle_condition_tangent_lines_right_angle_triangle_l247_247617


namespace tangent_product_le_one_third_l247_247505

theorem tangent_product_le_one_third (α β : ℝ) (h : α + β = π / 3) (hα : 0 < α) (hβ : 0 < β) : 
  Real.tan α * Real.tan β ≤ 1 / 3 :=
sorry

end tangent_product_le_one_third_l247_247505


namespace jason_needs_87_guppies_per_day_l247_247779

def guppies_needed_per_day (moray_eel_guppies : Nat)
  (betta_fish_number : Nat) (betta_fish_guppies : Nat)
  (angelfish_number : Nat) (angelfish_guppies : Nat)
  (lionfish_number : Nat) (lionfish_guppies : Nat) : Nat :=
  moray_eel_guppies +
  betta_fish_number * betta_fish_guppies +
  angelfish_number * angelfish_guppies +
  lionfish_number * lionfish_guppies

theorem jason_needs_87_guppies_per_day :
  guppies_needed_per_day 20 5 7 3 4 2 10 = 87 := by
  sorry

end jason_needs_87_guppies_per_day_l247_247779


namespace remaining_amoeba_is_blue_l247_247981

-- Define the initial number of amoebas for red, blue, and yellow types.
def n1 := 47
def n2 := 40
def n3 := 53

-- Define the property that remains constant, i.e., the parity of differences
def parity_diff (a b : ℕ) : Bool := (a - b) % 2 == 1

-- Initial conditions based on the given problem
def initial_conditions : Prop :=
  parity_diff n1 n2 = true ∧  -- odd
  parity_diff n1 n3 = false ∧ -- even
  parity_diff n2 n3 = true    -- odd

-- Final statement: Prove that the remaining amoeba is blue
theorem remaining_amoeba_is_blue : Prop :=
  initial_conditions ∧ (∀ final : String, final = "Blue")

end remaining_amoeba_is_blue_l247_247981


namespace triangle_inequality_l247_247798

open Real

theorem triangle_inequality (A B C : ℝ) (hA : 0 < A ∧ A < π) (hB : 0 < B ∧ B < π) (hC : 0 < C ∧ C < π) (h_sum : A + B + C = π) :
  sin A * cos C + A * cos B > 0 :=
by
  sorry

end triangle_inequality_l247_247798


namespace xiao_dong_not_both_understand_english_and_french_l247_247113

variables (P Q : Prop)

theorem xiao_dong_not_both_understand_english_and_french (h : ¬ (P ∧ Q)) : P → ¬ Q :=
sorry

end xiao_dong_not_both_understand_english_and_french_l247_247113


namespace loss_percentage_on_first_book_l247_247303

theorem loss_percentage_on_first_book 
    (C1 C2 SP : ℝ) 
    (H1 : C1 = 210) 
    (H2 : C1 + C2 = 360) 
    (H3 : SP = 1.19 * C2) 
    (H4 : SP = 178.5) :
    ((C1 - SP) / C1) * 100 = 15 :=
by
  sorry

end loss_percentage_on_first_book_l247_247303


namespace total_amount_paid_l247_247553

theorem total_amount_paid (cost_lunch : ℝ) (sales_tax_rate : ℝ) (tip_rate : ℝ) (sales_tax : ℝ) (tip : ℝ) 
  (h1 : cost_lunch = 100) 
  (h2 : sales_tax_rate = 0.04) 
  (h3 : tip_rate = 0.06) 
  (h4 : sales_tax = cost_lunch * sales_tax_rate) 
  (h5 : tip = cost_lunch * tip_rate) :
  cost_lunch + sales_tax + tip = 110 :=
by
  sorry

end total_amount_paid_l247_247553


namespace geometric_sequence_property_l247_247457

variable (a : ℕ → ℤ)
-- Assume the sequence is geometric with ratio r
variable (r : ℤ)

-- Define the sequence a_n as a geometric sequence
def geometric_sequence (a : ℕ → ℤ) (r : ℤ) : Prop :=
  ∀ (n : ℕ), a (n + 1) = a n * r

-- Given condition: a_4 + a_8 = -2
axiom condition : a 4 + a 8 = -2

theorem geometric_sequence_property
  (h : geometric_sequence a r) : a 6 * (a 2 + 2 * a 6 + a 10) = 4 :=
sorry

end geometric_sequence_property_l247_247457


namespace profit_percentage_l247_247691

theorem profit_percentage (SP CP : ℝ) (h₁ : SP = 300) (h₂ : CP = 250) : ((SP - CP) / CP) * 100 = 20 := by
  sorry

end profit_percentage_l247_247691


namespace angle_greater_difference_l247_247230

theorem angle_greater_difference (A B C : ℕ) (h1 : B = 5 * A) (h2 : A + B + C = 180) (h3 : A = 24) 
: C - A = 12 := 
by
  -- Proof omitted
  sorry

end angle_greater_difference_l247_247230


namespace merchant_marked_price_l247_247978

variable (L C M S : ℝ)

-- Conditions
def condition1 : Prop := C = 0.7 * L
def condition2 : Prop := C = 0.7 * S
def condition3 : Prop := S = 0.8 * M

-- The main statement
theorem merchant_marked_price (h1 : condition1 L C) (h2 : condition2 C S) (h3 : condition3 S M) : M = 1.25 * L :=
by
  sorry

end merchant_marked_price_l247_247978


namespace length_is_62_l247_247522

noncomputable def length_of_plot (b : ℝ) := b + 24

theorem length_is_62 (b : ℝ) (h1 : length_of_plot b = b + 24) 
  (h2 : 2 * (length_of_plot b + b) = 200) : 
  length_of_plot b = 62 :=
by sorry

end length_is_62_l247_247522


namespace quadratic_fraction_equality_l247_247154

theorem quadratic_fraction_equality (r : ℝ) (h1 : r ≠ 4) (h2 : r ≠ 6) (h3 : r ≠ 5) 
(h4 : r ≠ -4) (h5 : r ≠ -3): 
  (r^2 - 6*r + 8) / (r^2 - 9*r + 20) = (r^2 - 3*r - 18) / (r^2 - 2*r - 24) →
  r = -7/4 :=
by {
  sorry
}

end quadratic_fraction_equality_l247_247154


namespace total_marbles_l247_247998

-- Define the given conditions 
def bags : ℕ := 20
def marbles_per_bag : ℕ := 156

-- The theorem stating that the total number of marbles is 3120
theorem total_marbles : bags * marbles_per_bag = 3120 := by
  sorry

end total_marbles_l247_247998


namespace simplest_fraction_is_D_l247_247426

def fractionA (x : ℕ) : ℚ := 10 / (15 * x)
def fractionB (a b : ℕ) : ℚ := (2 * a * b) / (3 * a * a)
def fractionC (x : ℕ) : ℚ := (x + 1) / (3 * x + 3)
def fractionD (x : ℕ) : ℚ := (x + 1) / (x * x + 1)

theorem simplest_fraction_is_D (x a b : ℕ) :
  ¬ ∃ c, c ≠ 1 ∧
    (fractionA x = (fractionA x / c) ∨
     fractionB a b = (fractionB a b / c) ∨
     fractionC x = (fractionC x / c)) ∧
    ∀ d, d ≠ 1 → fractionD x ≠ (fractionD x / d) := 
  sorry

end simplest_fraction_is_D_l247_247426


namespace Hayley_l247_247466

-- Definitions based on the given conditions
def num_friends : ℕ := 9
def stickers_per_friend : ℕ := 8

-- Theorem statement
theorem Hayley's_total_stickers : num_friends * stickers_per_friend = 72 := by
  sorry

end Hayley_l247_247466


namespace total_profit_l247_247433

-- Definitions based on the conditions
def cost_per_dog : ℕ := 250
def num_dogs : ℕ := 2
def num_puppies : ℕ := 6
def selling_price_per_puppy : ℕ := 350

-- Statement of the theorem
theorem total_profit : (num_puppies * selling_price_per_puppy) - (num_dogs * cost_per_dog) = 1600 := 
by
  sorry

end total_profit_l247_247433


namespace quadratic_other_root_l247_247015

theorem quadratic_other_root (k : ℝ) (h : ∀ x, x^2 - k*x - 4 = 0 → x = 2 ∨ x = -2) :
  ∀ x, x^2 - k*x - 4 = 0 → x = -2 :=
by
  sorry

end quadratic_other_root_l247_247015


namespace arithmetic_sequence_number_of_terms_l247_247621

def arithmetic_sequence_terms_count (a d l : ℕ) : ℕ :=
  sorry

theorem arithmetic_sequence_number_of_terms :
  arithmetic_sequence_terms_count 13 3 73 = 21 :=
sorry

end arithmetic_sequence_number_of_terms_l247_247621


namespace pentagon_rectangle_ratio_l247_247571

theorem pentagon_rectangle_ratio (p w : ℝ) 
    (pentagon_perimeter : 5 * p = 30) 
    (rectangle_perimeter : ∃ l, 2 * w + 2 * l = 30 ∧ l = 2 * w) : 
    p / w = 6 / 5 := 
by
  sorry

end pentagon_rectangle_ratio_l247_247571


namespace ratio_adult_women_to_men_event_l247_247351

theorem ratio_adult_women_to_men_event :
  ∀ (total_members men_ratio women_ratio children : ℕ), 
  total_members = 2000 →
  men_ratio = 30 →
  children = 200 →
  women_ratio = men_ratio →
  women_ratio / men_ratio = 1 / 1 := 
by
  intros total_members men_ratio women_ratio children
  sorry

end ratio_adult_women_to_men_event_l247_247351


namespace stratified_sampling_first_grade_selection_l247_247698

theorem stratified_sampling_first_grade_selection
  (total_students : ℕ)
  (students_grade1 : ℕ)
  (sample_size : ℕ)
  (h_total : total_students = 2000)
  (h_grade1 : students_grade1 = 400)
  (h_sample : sample_size = 200) :
  sample_size * students_grade1 / total_students = 40 := by
  sorry

end stratified_sampling_first_grade_selection_l247_247698


namespace train_crossing_time_l247_247762

noncomputable def length_first_train : ℝ := 200  -- meters
noncomputable def speed_first_train_kmph : ℝ := 72  -- km/h
noncomputable def speed_first_train : ℝ := speed_first_train_kmph * (1000 / 3600)  -- m/s

noncomputable def length_second_train : ℝ := 300  -- meters
noncomputable def speed_second_train_kmph : ℝ := 36  -- km/h
noncomputable def speed_second_train : ℝ := speed_second_train_kmph * (1000 / 3600)  -- m/s

noncomputable def relative_speed : ℝ := speed_first_train - speed_second_train -- m/s
noncomputable def total_length : ℝ := length_first_train + length_second_train  -- meters
noncomputable def time_to_cross : ℝ := total_length / relative_speed  -- seconds

theorem train_crossing_time :
  time_to_cross = 50 := by
  sorry

end train_crossing_time_l247_247762


namespace largest_quantity_l247_247243

noncomputable def A := (2006 / 2005) + (2006 / 2007)
noncomputable def B := (2006 / 2007) + (2008 / 2007)
noncomputable def C := (2007 / 2006) + (2007 / 2008)

theorem largest_quantity : A > B ∧ A > C := by
  sorry

end largest_quantity_l247_247243


namespace staircase_problem_l247_247796

theorem staircase_problem :
  ∃ (n : ℕ), (n > 20) ∧ (n % 5 = 4) ∧ (n % 6 = 3) ∧ (n % 7 = 5) ∧ n = 159 :=
by sorry

end staircase_problem_l247_247796


namespace sum_coordinates_l247_247653

theorem sum_coordinates (x : ℝ) : 
  let C := (x, 8)
  let D := (-x, 8)
  (C.1 + C.2 + D.1 + D.2) = 16 := 
by
  sorry

end sum_coordinates_l247_247653


namespace monkeys_and_bananas_l247_247801

theorem monkeys_and_bananas (m1 m2 t b1 b2 : ℕ) (h1 : m1 = 8) (h2 : t = 8) (h3 : b1 = 8) (h4 : b2 = 3) : m2 = 3 :=
by
  -- Here we will include the formal proof steps
  sorry

end monkeys_and_bananas_l247_247801


namespace area_of_rectangle_is_108_l247_247601

-- Define the conditions and parameters
variables (P Q R S : Type) (diameter : ℝ) (height : ℝ) (width : ℝ) (area : ℝ)
variable (isTangentToSides : Prop)
variable (centersFormLineParallelToLongerSide : Prop)

-- Assume the given conditions
axiom h1 : diameter = 6
axiom h2 : isTangentToSides
axiom h3 : centersFormLineParallelToLongerSide

-- Define the goal to prove
theorem area_of_rectangle_is_108 (P Q R S : Type) (diameter : ℝ) (height : ℝ) (width : ℝ) (area : ℝ)
    (isTangentToSides : Prop) (centersFormLineParallelToLongerSide : Prop)
    (h1 : diameter = 6)
    (h2 : isTangentToSides)
    (h3 : centersFormLineParallelToLongerSide) :
    area = 108 :=
by
  -- Lean code requires an actual proof here, but for now, we'll use sorry.
  sorry

end area_of_rectangle_is_108_l247_247601


namespace Marty_combinations_l247_247499

theorem Marty_combinations:
  let colors := ({blue, green, yellow, black, white} : Finset String)
  let tools := ({brush, roller, sponge, spray_gun} : Finset String)
  colors.card * tools.card = 20 := 
by
  sorry

end Marty_combinations_l247_247499


namespace probability_X_equals_3_l247_247666

def total_score (a b : ℕ) : ℕ :=
  a + b

def prob_event_A_draws_yellow_B_draws_white : ℚ :=
  (2 / 5) * (3 / 4)

def prob_event_A_draws_white_B_draws_yellow : ℚ :=
  (3 / 5) * (2 / 4)

def prob_X_equals_3 : ℚ :=
  prob_event_A_draws_yellow_B_draws_white + prob_event_A_draws_white_B_draws_yellow

theorem probability_X_equals_3 :
  prob_X_equals_3 = 3 / 5 :=
by
  sorry

end probability_X_equals_3_l247_247666


namespace part_one_a_increasing_on_1_inf_part_one_a_decreasing_on_0_1_part_one_a_minimum_value_part_two_range_of_a_l247_247464

noncomputable def f (a x : ℝ) : ℝ := a * x + (a - 1) / x - Real.log x - 1

theorem part_one_a_increasing_on_1_inf (x : ℝ) : 
  (∀ x > 1, f 1 x > f 1 x) := sorry

theorem part_one_a_decreasing_on_0_1 (x : ℝ) : 
  (∀ x > 0, x < 1, f 1 x < f 1 x) := sorry

theorem part_one_a_minimum_value : f 1 1 = 0 := sorry

theorem part_two_range_of_a (a : ℝ) (x : ℝ) (h1 : a > 0) 
  (h2 : ∀ x ∈ set.Ici (1:ℝ), f a x ≥ 0) : 
  a ∈ set.Ici (1:ℝ) := sorry

end part_one_a_increasing_on_1_inf_part_one_a_decreasing_on_0_1_part_one_a_minimum_value_part_two_range_of_a_l247_247464


namespace ellipse_tangent_line_equation_l247_247616

variable {r a b x0 y0 x y : ℝ}
variable (h_r_pos : r > 0) (h_a_pos : a > 0) (h_b_pos : b > 0) (h_ineq : a > b)
variable (ellipse_eq : (x / a)^2 + (y / b)^2 = 1)
variable (tangent_circle_eq : x0 * x / r^2 + y0 * y / r^2 = 1)

theorem ellipse_tangent_line_equation :
  (a > b) → (a > 0) → (b > 0) → (x0 ≠ 0 ∨ y0 ≠ 0) → (x/a)^2 + (y/b)^2 = 1 →
  (x0 * x / a^2 + y0 * y / b^2 = 1) :=
by
  sorry

end ellipse_tangent_line_equation_l247_247616


namespace box_dimensions_l247_247919

theorem box_dimensions (a b c : ℕ) (h1 : a + c = 17) (h2 : a + b = 13) (h3 : b + c = 20) :
  a = 5 ∧ b = 8 ∧ c = 12 :=
by
  sorry

end box_dimensions_l247_247919


namespace fraction_zero_implies_x_half_l247_247681

theorem fraction_zero_implies_x_half (x : ℝ) (h₁ : (2 * x - 1) / (x + 2) = 0) (h₂ : x ≠ -2) : x = 1 / 2 :=
by sorry

end fraction_zero_implies_x_half_l247_247681


namespace arithmetic_sequence_S6_by_S4_l247_247747

-- Define the arithmetic sequence and the sum function
def sum_arithmetic_sequence (a d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

-- Given conditions
def S1 : ℕ := 1
def r (S2 S4 : ℕ) : Prop := S4 / S2 = 4

-- Proof statement
theorem arithmetic_sequence_S6_by_S4 :
  ∀ (a d : ℕ), 
  (sum_arithmetic_sequence a d 1 = S1) → (r (sum_arithmetic_sequence a d 2) (sum_arithmetic_sequence a d 4)) → 
  (sum_arithmetic_sequence a d 6 / sum_arithmetic_sequence a d 4 = 9 / 4) := 
by
  sorry

end arithmetic_sequence_S6_by_S4_l247_247747


namespace find_a_l247_247401

theorem find_a (a : ℝ) (h : 1 / Real.log 5 / Real.log a + 1 / Real.log 6 / Real.log a + 1 / Real.log 10 / Real.log a = 1) : a = 300 :=
sorry

end find_a_l247_247401


namespace sum_super_cool_rectangle_areas_eq_84_l247_247706

theorem sum_super_cool_rectangle_areas_eq_84 :
  ∀ (a b : ℕ), 
  (a * b = 3 * (a + b)) → 
  ∃ (S : ℕ), 
  S = 84 :=
by
  sorry

end sum_super_cool_rectangle_areas_eq_84_l247_247706


namespace sum_of_possible_a_l247_247019

theorem sum_of_possible_a (a : ℤ) :
  (∃ x : ℕ, x - (2 - a * x) / 6 = x / 3 - 1) →
  a = -19 :=
sorry

end sum_of_possible_a_l247_247019


namespace find_natural_numbers_l247_247880

theorem find_natural_numbers (n k : ℕ) (h : 2^n - 5^k = 7) : n = 5 ∧ k = 2 :=
by
  sorry

end find_natural_numbers_l247_247880


namespace solve_exponents_l247_247558

theorem solve_exponents (x y z : ℕ) (hx : x < y) (hy : y < z) 
  (h : 3^x + 3^y + 3^z = 179415) : x = 4 ∧ y = 7 ∧ z = 11 :=
by sorry

end solve_exponents_l247_247558


namespace max_sum_x_y_l247_247792

theorem max_sum_x_y {x y a b : ℝ} 
  (hx : 0 < x) (hy : 0 < y) (ha : 0 ≤ a ∧ a ≤ x) (hb : 0 ≤ b ∧ b ≤ y)
  (h1 : a^2 + y^2 = 2) (h2 : b^2 + x^2 = 1) (h3 : a * x + b * y = 1) : 
  x + y ≤ 2 :=
sorry

end max_sum_x_y_l247_247792


namespace concentration_replacement_l247_247128

theorem concentration_replacement 
  (initial_concentration : ℝ)
  (new_concentration : ℝ)
  (fraction_replaced : ℝ)
  (replacing_concentration : ℝ)
  (h1 : initial_concentration = 0.45)
  (h2 : new_concentration = 0.35)
  (h3 : fraction_replaced = 0.5) :
  replacing_concentration = 0.25 := by
  sorry

end concentration_replacement_l247_247128


namespace positive_difference_g_b_values_l247_247492

noncomputable def g (n : ℤ) : ℤ :=
if n < 0 then n^2 + 5 * n + 6 else 3 * n - 30

theorem positive_difference_g_b_values : 
  let g_neg_3 := g (-3)
  let g_3 := g 3
  g_neg_3 = 0 → g_3 = -21 → 
  ∃ b1 b2 : ℤ, g_neg_3 + g_3 + g b1 = 0 ∧ g_neg_3 + g_3 + g b2 = 0 ∧ 
  b1 ≠ b2 ∧ b1 < b2 ∧ b1 < 0 ∧ b2 > 0 ∧ b2 - b1 = 22 :=
by
  sorry

end positive_difference_g_b_values_l247_247492


namespace find_original_number_l247_247731

theorem find_original_number (N : ℕ) (h : ∃ k : ℕ, N - 5 = 13 * k) : N = 18 :=
sorry

end find_original_number_l247_247731


namespace decreasing_interval_l247_247356

noncomputable def func (x : ℝ) := 2 * x^3 - 6 * x^2 + 11

theorem decreasing_interval : ∀ x : ℝ, 0 < x ∧ x < 2 → deriv func x < 0 :=
by
  sorry

end decreasing_interval_l247_247356


namespace trigonometric_identity_l247_247452

theorem trigonometric_identity (α : ℝ) (h : Real.sin (α - Real.pi / 4) = 1 / 3) : 
  Real.cos (Real.pi / 4 + α) = - (1 / 3) := 
by
  sorry

end trigonometric_identity_l247_247452


namespace length_of_hypotenuse_l247_247324

theorem length_of_hypotenuse (a b c : ℝ) (h1 : a^2 + b^2 + c^2 = 2450) (h2 : c = b + 10) (h3 : a^2 + b^2 = c^2) : c = 35 :=
by
  sorry

end length_of_hypotenuse_l247_247324


namespace sum_of_coefficients_l247_247883

def polynomial (x : ℤ) : ℤ := 3 * (x^8 - 2 * x^5 + 4 * x^3 - 7) - 5 * (2 * x^4 - 3 * x^2 + 8) + 6 * (x^6 - 3)

theorem sum_of_coefficients : polynomial 1 = -59 := 
by
  sorry

end sum_of_coefficients_l247_247883


namespace quadratic_function_expr_value_of_b_minimum_value_of_m_l247_247606

-- Problem 1: Proving the quadratic function expression
theorem quadratic_function_expr (x : ℝ) (b c : ℝ)
  (h1 : (0:ℝ) = x^2 + b * 0 + c)
  (h2 : -b / 2 = (1:ℝ)) :
  x^2 - 2 * x + 4 = x^2 + b * x + c := sorry

-- Problem 2: Proving specific values of b
theorem value_of_b (b c : ℝ)
  (h1 : b^2 - c = 0)
  (h2 : ∀ x : ℝ, (b - 3 ≤ x ∧ x ≤ b → (x^2 + b * x + c ≥ 21))) :
  b = -Real.sqrt 7 ∨ b = 4 := sorry

-- Problem 3: Proving the minimum value of m
theorem minimum_value_of_m (x : ℝ) (m : ℝ)
  (h1 : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → 2 * x^2 + x + m ≥ x^2 - 2 * x + 4) :
  m = 4 := sorry

end quadratic_function_expr_value_of_b_minimum_value_of_m_l247_247606


namespace henry_jill_age_ratio_l247_247236

theorem henry_jill_age_ratio :
  ∀ (H J : ℕ), (H + J = 48) → (H = 29) → (J = 19) → ((H - 9) / (J - 9) = 2) :=
by
  intros H J h_sum h_henry h_jill
  sorry

end henry_jill_age_ratio_l247_247236


namespace number_of_solutions_l247_247596

theorem number_of_solutions :
  ∃ n : ℕ,  (1 + ⌊(102 * n : ℚ) / 103⌋ = ⌈(101 * n : ℚ) / 102⌉) ↔ (n < 10506) := 
sorry

end number_of_solutions_l247_247596


namespace tail_length_10_l247_247043

theorem tail_length_10 (length_body tail_length head_length width height overall_length: ℝ) 
  (h1 : tail_length = (1 / 2) * length_body)
  (h2 : head_length = (1 / 6) * length_body)
  (h3 : height = 1.5 * width)
  (h4 : overall_length = length_body + tail_length)
  (h5 : overall_length = 30)
  (h6 : width = 12) :
  tail_length = 10 :=
by
  sorry

end tail_length_10_l247_247043


namespace P_investment_calculation_l247_247788

variable {P_investment : ℝ}
variable (Q_investment : ℝ := 36000)
variable (total_profit : ℝ := 18000)
variable (Q_profit : ℝ := 6001.89)

def P_profit : ℝ := total_profit - Q_profit

theorem P_investment_calculation :
  P_investment = (P_profit * Q_investment) / Q_profit :=
by
  sorry

end P_investment_calculation_l247_247788


namespace polar_to_cartesian_l247_247082

theorem polar_to_cartesian :
  ∀ (ρ θ : ℝ), ρ = 3 ∧ θ = π / 6 → 
  (ρ * Real.cos θ, ρ * Real.sin θ) = (3 * Real.sqrt 3 / 2, 3 / 2) :=
by
  intro ρ θ
  rintro ⟨hρ, hθ⟩
  rw [hρ, hθ]
  sorry

end polar_to_cartesian_l247_247082


namespace minimum_schoolchildren_l247_247669

theorem minimum_schoolchildren (candies : ℕ) (h : candies = 200) : ∃ n : ℕ, n = 21 ∧ ∀ (dist : Fin n → ℕ), (∃ i j, i ≠ j ∧ dist i = dist j) :=
by
  use 21
  split
  { refl }
  { intro dist
    have hn : 21 * 20 / 2 = 210 := by norm_num
    rw ←h at hn
    linarith }

end minimum_schoolchildren_l247_247669


namespace difference_in_nickels_is_correct_l247_247845

variable (q : ℤ)

def charles_quarters : ℤ := 7 * q + 2
def richard_quarters : ℤ := 3 * q + 8

theorem difference_in_nickels_is_correct :
  5 * (charles_quarters - richard_quarters) = 20 * q - 30 :=
by
  sorry

end difference_in_nickels_is_correct_l247_247845


namespace rtl_to_conventional_notation_l247_247326

theorem rtl_to_conventional_notation (a b c d e : ℚ) :
  (a / (b - (c * (d + e)))) = a / (b - c * (d + e)) := by
  sorry

end rtl_to_conventional_notation_l247_247326


namespace longer_side_of_rectangle_l247_247771

noncomputable def circle_radius : ℝ := 6
noncomputable def circle_area : ℝ := Real.pi * circle_radius^2
noncomputable def rectangle_area : ℝ := 3 * circle_area
noncomputable def shorter_side : ℝ := 2 * circle_radius

theorem longer_side_of_rectangle :
    ∃ (l : ℝ), l = rectangle_area / shorter_side ∧ l = 9 * Real.pi :=
by
  sorry

end longer_side_of_rectangle_l247_247771


namespace sum_of_squares_of_roots_of_quadratic_l247_247823

noncomputable def sum_of_squares_of_roots (p q : ℝ) (a b : ℝ) : Prop :=
  a^2 + b^2 = 4 * p^2 - 6 * q

theorem sum_of_squares_of_roots_of_quadratic
  (p q a b : ℝ)
  (h1 : a + b = 2 * p / 3)
  (h2 : a * b = q / 3)
  (h3 : a * a + b * b = 4 * p^2 - 6 * q) :
  sum_of_squares_of_roots p q a b :=
by
  sorry

end sum_of_squares_of_roots_of_quadratic_l247_247823


namespace possible_values_ceil_square_l247_247305

noncomputable def num_possible_values (x : ℝ) (hx : ⌈x⌉ = 12) : ℕ := 23

theorem possible_values_ceil_square (x : ℝ) (hx : ⌈x⌉ = 12) :
  let n := num_possible_values x hx in n = 23 :=
by
  let n := num_possible_values x hx
  exact rfl

end possible_values_ceil_square_l247_247305


namespace reachable_target_l247_247988

-- Define the initial state of the urn
def initial_urn_state : (ℕ × ℕ) := (150, 50)

-- Define the operations as changes in counts of black and white marbles
def operation1 (state : ℕ × ℕ) := (state.1 - 2, state.2)
def operation2 (state : ℕ × ℕ) := (state.1 - 1, state.2)
def operation3 (state : ℕ × ℕ) := (state.1, state.2 - 2)
def operation4 (state : ℕ × ℕ) := (state.1 + 2, state.2 - 3)

-- Define a predicate that a state can be reached from the initial state
def reachable (target : ℕ × ℕ) : Prop :=
  ∃ n1 n2 n3 n4 : ℕ, 
    operation1^[n1] (operation2^[n2] (operation3^[n3] (operation4^[n4] initial_urn_state))) = target

-- The theorem to be proved
theorem reachable_target : reachable (1, 2) :=
sorry

end reachable_target_l247_247988


namespace equivalent_modulo_l247_247092

theorem equivalent_modulo :
  ∃ (n : ℤ), 0 ≤ n ∧ n < 31 ∧ -250 ≡ n [ZMOD 31] ∧ n = 29 := 
by
  sorry

end equivalent_modulo_l247_247092


namespace minimum_value_of_3x_plus_4y_l247_247012

theorem minimum_value_of_3x_plus_4y :
  ∀ (x y : ℝ), 0 < x → 0 < y → x + 3 * y = 5 * x * y → (3 * x + 4 * y) ≥ 24 / 5 :=
by
  sorry

end minimum_value_of_3x_plus_4y_l247_247012


namespace infinite_double_perfect_squares_l247_247183

-- Definition of a double number
def is_double_number (n : ℕ) : Prop :=
  ∃ (k : ℕ), ∃ (d : ℕ), d ≠ 0 ∧ 10^k * d + d = n ∧ 10^k ≤ d ∧ d < 10^(k+1)

-- The theorem statement
theorem infinite_double_perfect_squares :
  ∃ (S : Set ℕ), (∀ n ∈ S, is_double_number n ∧ ∃ m, m * m = n) ∧
  Set.Infinite S :=
sorry

end infinite_double_perfect_squares_l247_247183


namespace value_of_business_l247_247684

theorem value_of_business 
  (ownership : ℚ)
  (sale_fraction : ℚ)
  (sale_value : ℚ) 
  (h_ownership : ownership = 2/3) 
  (h_sale_fraction : sale_fraction = 3/4) 
  (h_sale_value : sale_value = 6500) : 
  2 * sale_value = 13000 := 
by
  -- mathematical equivalent proof here
  -- This is a placeholder.
  sorry

end value_of_business_l247_247684


namespace min_a1_value_l247_247930

theorem min_a1_value (a : ℕ → ℝ) :
  (∀ n > 1, a n = 9 * a (n-1) - 2 * n) →
  (∀ n, a n > 0) →
  (∀ x, (∀ n > 1, a n = 9 * a (n-1) - 2 * n) → (∀ n, a n > 0) → x ≥ a 1) →
  a 1 = 499.25 / 648 :=
sorry

end min_a1_value_l247_247930


namespace sets_equal_l247_247262

-- Defining the sets and proving their equality
theorem sets_equal : { x : ℝ | x^2 + 1 = 0 } = (∅ : Set ℝ) :=
  sorry

end sets_equal_l247_247262


namespace seq_proof_l247_247009

noncomputable def arithmetic_seq (a1 a2 : ℤ) : Prop :=
  ∃ (d : ℤ), a1 = -1 + d ∧ a2 = a1 + d ∧ -4 = a1 + 3 * d

noncomputable def geometric_seq (b : ℤ) : Prop :=
  b = 2 ∨ b = -2

theorem seq_proof (a1 a2 b : ℤ) 
  (h1 : arithmetic_seq a1 a2) 
  (h2 : geometric_seq b) : 
  (a2 + a1 : ℚ) / b = 5 / 2 ∨ (a2 + a1 : ℚ) / b = -5 / 2 := by
  sorry

end seq_proof_l247_247009


namespace final_total_cost_l247_247707

def initial_spiral_cost : ℝ := 15
def initial_planner_cost : ℝ := 10
def spiral_discount_rate : ℝ := 0.20
def planner_discount_rate : ℝ := 0.15
def num_spirals : ℝ := 4
def num_planners : ℝ := 8
def sales_tax_rate : ℝ := 0.07

theorem final_total_cost :
  let discounted_spiral_cost := initial_spiral_cost * (1 - spiral_discount_rate)
  let discounted_planner_cost := initial_planner_cost * (1 - planner_discount_rate)
  let total_before_tax := num_spirals * discounted_spiral_cost + num_planners * discounted_planner_cost
  let total_tax := total_before_tax * sales_tax_rate
  let total_cost := total_before_tax + total_tax
  total_cost = 124.12 :=
by
  sorry

end final_total_cost_l247_247707


namespace total_turtles_l247_247813

variable (Kristen_turtles Kris_turtles Trey_turtles : ℕ)

-- Kristen has 12 turtles
def Kristen_turtles_count : Kristen_turtles = 12 := sorry

-- Kris has 1/4 the number of turtles Kristen has
def Kris_turtles_count (hK : Kristen_turtles = 12) : Kris_turtles = Kristen_turtles / 4 := sorry

-- Trey has 5 times as many turtles as Kris
def Trey_turtles_count (hK : Kristen_turtles = 12) (hKr : Kris_turtles = Kristen_turtles / 4) : Trey_turtles = 5 * Kris_turtles := sorry

-- Total number of turtles
theorem total_turtles (hK : Kristen_turtles = 12) (hKr : Kris_turtles = Kristen_turtles / 4) 
  (hT : Trey_turtles = 5 * Kris_turtles) : Kristen_turtles + Kris_turtles + Trey_turtles = 30 := sorry

end total_turtles_l247_247813


namespace no_solution_in_natural_numbers_l247_247509

theorem no_solution_in_natural_numbers (x y z : ℕ) (hxy : x ≠ 0) (hyz : y ≠ 0) (hzx : z ≠ 0) :
  ¬ (x / y + y / z + z / x = 1) :=
by sorry

end no_solution_in_natural_numbers_l247_247509


namespace S_nine_l247_247893

noncomputable def S : ℕ → ℚ
| 3 => 8
| 6 => 10
| _ => 0  -- Placeholder for other values, as we're interested in these specific ones

theorem S_nine (S_3_eq : S 3 = 8) (S_6_eq : S 6 = 10) : S 9 = 21 / 2 :=
by
  -- Construct the proof here
  sorry

end S_nine_l247_247893


namespace infection_equation_correct_l247_247194

theorem infection_equation_correct (x : ℝ) :
  1 + x + x * (x + 1) = 196 :=
sorry

end infection_equation_correct_l247_247194


namespace intersection_of_domains_l247_247205

def A_domain : Set ℝ := { x : ℝ | 4 - x^2 ≥ 0 }
def B_domain : Set ℝ := { x : ℝ | 1 - x > 0 }

theorem intersection_of_domains :
  (A_domain ∩ B_domain) = { x : ℝ | -2 ≤ x ∧ x < 1 } :=
by
  sorry

end intersection_of_domains_l247_247205


namespace factorize_cubic_l247_247876

theorem factorize_cubic (a : ℝ) : a^3 - a = a * (a + 1) * (a - 1) :=
sorry

end factorize_cubic_l247_247876


namespace arithmetic_sequence_l247_247486

theorem arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) 
  (h1 : a 1 + a 2 + a 3 = 32) 
  (h2 : a 11 + a 12 + a 13 = 118) 
  (arith_seq : ∀ n, a (n + 1) = a n + d) : 
  a 4 + a 10 = 50 :=
by 
  sorry

end arithmetic_sequence_l247_247486


namespace ten_pow_n_plus_eight_div_nine_is_integer_l247_247345

theorem ten_pow_n_plus_eight_div_nine_is_integer (n : ℕ) : ∃ k : ℤ, 10^n + 8 = 9 * k := 
sorry

end ten_pow_n_plus_eight_div_nine_is_integer_l247_247345


namespace derivative_of_f_l247_247173

noncomputable def f (x : ℝ) : ℝ := Real.exp (2 * x) * Real.cos x

theorem derivative_of_f :
  ∀ x : ℝ, deriv f x = Real.exp (2 * x) * (2 * Real.cos x - Real.sin x) :=
by
  intro x
  -- We skip the proof here
  sorry

end derivative_of_f_l247_247173


namespace central_angle_of_sector_l247_247295

noncomputable def circumference (r : ℝ) : ℝ := 2 * Real.pi * r
noncomputable def arc_length (r α : ℝ) : ℝ := r * α

theorem central_angle_of_sector :
  ∀ (r α : ℝ),
    circumference r = 2 * Real.pi + 2 →
    arc_length r α = 2 * Real.pi - 2 →
    α = Real.pi - 1 :=
by
  intros r α hcirc harc
  sorry

end central_angle_of_sector_l247_247295


namespace f_of_x_squared_domain_l247_247170

structure FunctionDomain (f : ℝ → ℝ) :=
  (domain : Set ℝ)
  (domain_eq : domain = Set.Icc 0 1)

theorem f_of_x_squared_domain (f : ℝ → ℝ) (h : FunctionDomain f) :
  FunctionDomain (fun x => f (x ^ 2)) :=
{
  domain := Set.Icc (-1) 1,
  domain_eq := sorry
}

end f_of_x_squared_domain_l247_247170


namespace solve_for_y_l247_247686

-- Define the conditions and the goal to prove in Lean 4
theorem solve_for_y
  (x y : ℤ) 
  (h1 : x + y = 250) 
  (h2 : x - y = 200) : 
  y = 25 :=
by
  sorry

end solve_for_y_l247_247686


namespace calculate_expression_l247_247440

theorem calculate_expression : (-1) ^ 47 + 2 ^ (3 ^ 3 + 4 ^ 2 - 6 ^ 2) = 127 := 
by 
  sorry

end calculate_expression_l247_247440


namespace isabella_hair_ratio_l247_247635

-- Conditions in the problem
variable (hair_before : ℕ) (hair_after : ℕ)
variable (hb : hair_before = 18)
variable (ha : hair_after = 36)

-- Definitions based on conditions
def hair_ratio (after : ℕ) (before : ℕ) : ℚ := (after : ℚ) / (before : ℚ)

theorem isabella_hair_ratio : 
  hair_ratio hair_after hair_before = 2 :=
by
  -- plug in the known values
  rw [hb, ha]
  -- show the equation
  norm_num
  sorry

end isabella_hair_ratio_l247_247635


namespace maximize_victory_probability_l247_247533

-- Define the conditions
variables {n : ℕ}
def number_of_voters := 2 * n
def half_support_miraflores := n
def half_support_dick_maloney := n
def miraflores_is_voter := true

-- Define the districts
def district1 := {miraflores}
def district2 := {voters | voters ≠ miraflores}

theorem maximize_victory_probability (n : ℕ) (h₁ : nat.odd (2*n + 1) = true) : 
  (let district1_voters := 1 in
   let district2_voters := 2*n - 1 in
   maximize_probability_of_winning(district1_voters, district2_voters) = true) :=
sorry

end maximize_victory_probability_l247_247533


namespace greatest_multiple_of_5_and_6_under_1000_l247_247377

theorem greatest_multiple_of_5_and_6_under_1000 : 
  ∃ x, (x % 5 = 0 ∧ x % 6 = 0 ∧ x < 1000) ∧ 
  (∀ y, (y % 5 = 0 ∧ y % 6 = 0 ∧ y < 1000) → y ≤ x) ∧ 
  x = 990 :=
begin
  sorry
end

end greatest_multiple_of_5_and_6_under_1000_l247_247377


namespace urn_gold_coins_percentage_l247_247264

theorem urn_gold_coins_percentage (obj_perc_beads : ℝ) (coins_perc_gold : ℝ) : 
    obj_perc_beads = 0.15 → coins_perc_gold = 0.65 → 
    (1 - obj_perc_beads) * coins_perc_gold = 0.5525 := 
by
  intros h_obj_perc_beads h_coins_perc_gold
  sorry

end urn_gold_coins_percentage_l247_247264


namespace max_sin_product_proof_l247_247487

noncomputable def max_sin_product : ℝ :=
  let A := (-8, 0)
  let B := (8, 0)
  let C (t : ℝ) := (t, 6)
  let AB : ℝ := 16
  let AC (t : ℝ) := Real.sqrt ((t + 8)^2 + 36)
  let BC (t : ℝ) := Real.sqrt ((t - 8)^2 + 36)
  let area : ℝ := 48
  let sin_ACB (t : ℝ) := 96 / Real.sqrt (((t + 8)^2 + 36) * ((t - 8)^2 + 36))
  let sin_CAB_CBA : ℝ := 3 / 8
  sin_CAB_CBA

theorem max_sin_product_proof : ∀ t : ℝ, max_sin_product = 3 / 8 :=
by
  sorry

end max_sin_product_proof_l247_247487


namespace blue_red_area_equal_l247_247286

theorem blue_red_area_equal (n : ℕ) (h : n ≥ 2) : 
  (∃ blue_area red_area, blue_area = red_area) ↔ (n % 2 = 1) :=
by
  sorry

end blue_red_area_equal_l247_247286


namespace intersection_A_B_l247_247166

def A : Set ℤ := {-1, 1, 2}
def B : Set ℤ := {x | x^2 = x}

theorem intersection_A_B : A ∩ B = {1} := 
by
  sorry

end intersection_A_B_l247_247166


namespace find_y_l247_247403

theorem find_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (hrem : x % y = 11.52) (hdiv : x / y = 96.12) : y = 96 := 
sorry

end find_y_l247_247403


namespace polyhedron_faces_same_edges_l247_247937

theorem polyhedron_faces_same_edges (n : ℕ) (h_n : n ≥ 4) : 
  ∃ (f1 f2 : ℕ), f1 ≠ f2 ∧ 3 ≤ f1 ∧ f1 ≤ n - 1 ∧ 3 ≤ f2 ∧ f2 ≤ n - 1 ∧ f1 = f2 := 
by
  sorry

end polyhedron_faces_same_edges_l247_247937


namespace probability_at_least_one_boy_and_one_girl_correct_l247_247076

-- Define the size of the club and subgroups
def total_members : ℕ := 30
def boys : ℕ := 12
def girls : ℕ := 18
def committee_size : ℕ := 6

-- Defining the calculations as per the problem
def total_committees : ℕ := Nat.choose total_members committee_size
def all_boys_committees : ℕ := Nat.choose boys committee_size
def all_girls_committees : ℕ := Nat.choose girls committee_size

def probability_all_boys_or_all_girls : ℚ := (all_boys_committees + all_girls_committees) / total_committees
def probability_at_least_one_boy_and_one_girl : ℚ := 1 - probability_all_boys_or_all_girls

-- Statement of the theorem
theorem probability_at_least_one_boy_and_one_girl_correct : 
  probability_at_least_one_boy_and_one_girl = 574287 / 593775 := by 
  sorry

end probability_at_least_one_boy_and_one_girl_correct_l247_247076


namespace win_sector_area_l247_247695

theorem win_sector_area (r : ℝ) (P : ℝ) (h0 : r = 8) (h1 : P = 3 / 8) :
    let area_total := Real.pi * r ^ 2
    let area_win := P * area_total
    area_win = 24 * Real.pi :=
by 
  sorry

end win_sector_area_l247_247695


namespace haley_seeds_l247_247302

theorem haley_seeds (total_seeds seeds_big_garden total_small_gardens seeds_per_small_garden : ℕ) 
  (h1 : total_seeds = 56)
  (h2 : seeds_big_garden = 35)
  (h3 : total_small_gardens = 7)
  (h4 : total_seeds - seeds_big_garden = 21)
  (h5 : 21 / total_small_gardens = seeds_per_small_garden) :
  seeds_per_small_garden = 3 :=
by sorry

end haley_seeds_l247_247302


namespace prime_factors_count_l247_247855

def number := 2310

theorem prime_factors_count : (nat.factors number).nodup.length = 5 := sorry

end prime_factors_count_l247_247855


namespace households_used_both_brands_l247_247835

/-- 
A marketing firm determined that, of 160 households surveyed, 80 used neither brand A nor brand B soap.
60 used only brand A soap and for every household that used both brands of soap, 3 used only brand B soap.
--/
theorem households_used_both_brands (X: ℕ) (H: 4*X + 140 = 160): X = 5 :=
by
  sorry

end households_used_both_brands_l247_247835


namespace average_tree_height_l247_247413

theorem average_tree_height :
  let tree1 := 8
  let tree2 := if tree3 = 16 then 4 else 16
  let tree3 := 16
  let tree4 := if tree5 = 32 then 8 else 32
  let tree5 := 32
  let tree6 := if tree5 = 32 then 64 else 16
  let total_sum := tree1 + tree2 + tree3 + tree4 + tree5 + tree6
  let average_height := total_sum / 6
  average_height = 14 :=
by
  sorry

end average_tree_height_l247_247413


namespace gallons_per_cubic_foot_l247_247333

theorem gallons_per_cubic_foot (mix_per_pound : ℝ) (capacity_cubic_feet : ℕ) (weight_per_gallon : ℝ)
    (price_per_tbs : ℝ) (total_cost : ℝ) (total_gallons : ℝ) :
  mix_per_pound = 1.5 →
  capacity_cubic_feet = 6 →
  weight_per_gallon = 8 →
  price_per_tbs = 0.5 →
  total_cost = 270 →
  total_gallons = total_cost / (price_per_tbs * mix_per_pound * weight_per_gallon) →
  total_gallons / capacity_cubic_feet = 7.5 :=
by
  intro h1 h2 h3 h4 h5 h6
  rw [h2, h6]
  sorry

end gallons_per_cubic_foot_l247_247333


namespace odd_number_as_diff_of_squares_l247_247504

theorem odd_number_as_diff_of_squares (n : ℤ) : ∃ a b : ℤ, a^2 - b^2 = 2 * n + 1 :=
by
  use (n + 1), n
  sorry

end odd_number_as_diff_of_squares_l247_247504


namespace book_costs_and_scenarios_l247_247538

theorem book_costs_and_scenarios :
  (∃ (x y : ℕ), x + 3 * y = 180 ∧ 3 * x + y = 140 ∧ 
    (x = 30) ∧ (y = 50)) ∧ 
  (∀ (m : ℕ), (30 * m + 75 * m) ≤ 700 → (∃ (m_values : Finset ℕ), 
    m_values = {2, 4, 6} ∧ (m ∈ m_values))) :=
  sorry

end book_costs_and_scenarios_l247_247538


namespace prove_billy_age_l247_247143

-- Define B and J as real numbers representing the ages of Billy and Joe respectively
variables (B J : ℝ)

-- State the conditions
def billy_triple_of_joe : Prop := B = 3 * J
def sum_of_ages : Prop := B + J = 63

-- State the proposition to prove
def billy_age_proof : Prop := B = 47.25

-- Main theorem combining the conditions and the proof statement
theorem prove_billy_age (h1 : billy_triple_of_joe B J) (h2 : sum_of_ages B J) : billy_age_proof B :=
by
  sorry

end prove_billy_age_l247_247143


namespace jelly_beans_in_jar_y_l247_247087

-- Definitions of the conditions
def total_beans : ℕ := 1200
def number_beans_in_jar_y (y : ℕ) := y
def number_beans_in_jar_x (y : ℕ) := 3 * y - 400

-- The main theorem to be proven
theorem jelly_beans_in_jar_y (y : ℕ) :
  number_beans_in_jar_x y + number_beans_in_jar_y y = total_beans → 
  y = 400 := 
by
  sorry

end jelly_beans_in_jar_y_l247_247087


namespace max_Sn_in_arithmetic_sequence_l247_247048

theorem max_Sn_in_arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h_arith_seq : ∀ {m n p q : ℕ}, m + n = p + q → a m + a n = a p + a q)
  (h_a4 : a 4 = 1)
  (h_S5 : S 5 = 10)
  (h_S : ∀ n, S n = (n / 2) * (2 * a 1 + (n - 1) * (a 2 - a 1))) :
  ∃ n, n = 4 ∨ n = 5 ∧ ∀ m ≠ n, S m ≤ S n := by
  sorry

end max_Sn_in_arithmetic_sequence_l247_247048


namespace total_marbles_correct_l247_247033

variable (r : ℝ) -- number of red marbles
variable (b : ℝ) -- number of blue marbles
variable (g : ℝ) -- number of green marbles

-- Conditions
def red_blue_ratio : Prop := r = 1.5 * b
def green_red_ratio : Prop := g = 1.8 * r

-- Total number of marbles
def total_marbles (r b g : ℝ) : ℝ := r + b + g

theorem total_marbles_correct (r b g : ℝ) (h1 : red_blue_ratio r b) (h2 : green_red_ratio r g) : 
  total_marbles r b g = 3.467 * r :=
by 
  sorry

end total_marbles_correct_l247_247033


namespace ballet_class_members_l247_247124

theorem ballet_class_members (large_groups : ℕ) (members_per_large_group : ℕ) (total_members : ℕ) 
    (h1 : large_groups = 12) (h2 : members_per_large_group = 7) (h3 : total_members = large_groups * members_per_large_group) : 
    total_members = 84 :=
sorry

end ballet_class_members_l247_247124


namespace probability_all_letters_SUPERBLOOM_l247_247639

noncomputable def choose (n k : ℕ) : ℕ := sorry

theorem probability_all_letters_SUPERBLOOM :
  let P1 := 1 / (choose 6 3)
  let P2 := 9 / (choose 8 5)
  let P3 := 1 / (choose 5 4)
  P1 * P2 * P3 = 9 / 1120 :=
by
  sorry

end probability_all_letters_SUPERBLOOM_l247_247639


namespace b_k_divisible_by_11_is_5_l247_247785

def b (n : ℕ) : ℕ :=
  -- Function to concatenate numbers from 1 to n
  let digits := List.join (List.map (λ x => Nat.digits 10 x) (List.range' 1 n.succ))
  digits.foldl (λ acc d => acc * 10 + d) 0

def g (n : ℕ) : ℤ :=
  let digits := Nat.digits 10 n
  digits.enum.foldl (λ acc ⟨i, d⟩ => if i % 2 = 0 then acc + Int.ofNat d else acc - Int.ofNat d) 0

def isDivisibleBy11 (n : ℕ) : Bool :=
  g n % 11 = 0

def count_b_k_divisible_by_11 : ℕ :=
  List.length (List.filter isDivisibleBy11 (List.map b (List.range' 1 51)))

theorem b_k_divisible_by_11_is_5 : count_b_k_divisible_by_11 = 5 := by
  sorry

end b_k_divisible_by_11_is_5_l247_247785


namespace age_difference_l247_247563

theorem age_difference (A B C : ℕ) (hB : B = 14) (hBC : B = 2 * C) (hSum : A + B + C = 37) : A - B = 2 :=
by
  sorry

end age_difference_l247_247563


namespace more_birds_than_storks_l247_247560

-- Defining the initial number of birds
def initial_birds : ℕ := 2

-- Defining the number of birds that joined
def additional_birds : ℕ := 5

-- Defining the number of storks that joined
def storks : ℕ := 4

-- Defining the total number of birds
def total_birds : ℕ := initial_birds + additional_birds

-- Defining the problem statement in Lean 4
theorem more_birds_than_storks : (total_birds - storks) = 3 := by
  sorry

end more_birds_than_storks_l247_247560


namespace value_of_last_installment_l247_247252

noncomputable def total_amount_paid_without_processing_fee : ℝ :=
  36 * 2300

noncomputable def total_interest_paid : ℝ :=
  total_amount_paid_without_processing_fee - 35000

noncomputable def last_installment_value : ℝ :=
  2300 + 1000

theorem value_of_last_installment :
  last_installment_value = 3300 :=
  by
    sorry

end value_of_last_installment_l247_247252


namespace jenn_has_five_jars_l247_247924

/-- Each jar can hold 160 quarters, the bike costs 180 dollars, 
    Jenn will have 20 dollars left over, 
    and a quarter is worth 0.25 dollars.
    Prove that Jenn has 5 jars full of quarters. -/
theorem jenn_has_five_jars :
  let quarters_per_jar := 160
  let bike_cost := 180
  let money_left := 20
  let total_money_needed := bike_cost + money_left
  let quarter_value := 0.25
  let total_quarters_needed := total_money_needed / quarter_value
  let jars := total_quarters_needed / quarters_per_jar
  
  jars = 5 :=
by
  sorry

end jenn_has_five_jars_l247_247924


namespace bob_total_profit_l247_247439

def initial_cost (num_dogs : ℕ) (cost_per_dog : ℕ) : ℕ := num_dogs * cost_per_dog

def revenue (num_puppies : ℕ) (price_per_puppy : ℕ) : ℕ := num_puppies * price_per_puppy

def total_profit (initial_cost : ℕ) (revenue : ℕ) : ℕ := revenue - initial_cost

theorem bob_total_profit (c1 : initial_cost 2 250 = 500)
                        (c2 : revenue 6 350 = 2100)
                        (c3 : total_profit 500 2100 = 1600) :
  total_profit (initial_cost 2 250) (revenue 6 350) = 1600 := by
  sorry

end bob_total_profit_l247_247439


namespace det_B_squared_minus_3IB_l247_247622

open Matrix

def B : Matrix (Fin 2) (Fin 2) ℝ := ![![2, 4], ![3, 1]]
def I : Matrix (Fin 2) (Fin 2) ℝ := 1

theorem det_B_squared_minus_3IB :
  det (B * B - 3 * I * B) = 100 := by
  sorry

end det_B_squared_minus_3IB_l247_247622


namespace saree_sale_price_l247_247948

theorem saree_sale_price (original_price : ℝ) (d1 d2 d3 d4 t : ℝ) :
  original_price = 510 ∧ d1 = 0.12 ∧ d2 = 0.15 ∧ d3 = 0.20 ∧ d4 = 0.10 ∧ t = 0.10 →
  let price1 := original_price * (1 - d1) in
  let price2 := price1 * (1 - d2) in
  let price3 := price2 * (1 - d3) in
  let price4 := price3 * (1 - d4) in
  let final_price := price4 * (1 + t) in
  final_price ≈ 302 :=
by
  intros h
  sorry

end saree_sale_price_l247_247948


namespace traceable_edges_l247_247339

-- Define the vertices of the rectangle
def vertex (x y : ℕ) : ℕ × ℕ := (x, y)

-- Define the edges of the rectangle
def edges : List (ℕ × ℕ) :=
  [vertex 0 0, vertex 0 1,    -- vertical edges
   vertex 1 0, vertex 1 1,
   vertex 2 0, vertex 2 1,
   vertex 0 0, vertex 1 0,    -- horizontal edges
   vertex 1 0, vertex 2 0,
   vertex 0 1, vertex 1 1,
   vertex 1 1, vertex 2 1]

-- Define the theorem to be proved
theorem traceable_edges :
  ∃ (count : ℕ), count = 61 :=
by
  sorry

end traceable_edges_l247_247339


namespace area_of_enclosed_figure_l247_247519

theorem area_of_enclosed_figure:
  ∫ (x : ℝ) in (1/2)..2, x⁻¹ = 2 * Real.log 2 :=
by
  sorry

end area_of_enclosed_figure_l247_247519


namespace remainder_zero_l247_247271

theorem remainder_zero (x : ℤ) :
  (x^5 - 1) * (x^3 - 1) % (x^2 + x + 1) = 0 := by
sorry

end remainder_zero_l247_247271


namespace value_of_expression_l247_247441

theorem value_of_expression : 5 * 405 + 4 * 405 - 3 * 405 + 404 = 2834 :=
by sorry

end value_of_expression_l247_247441


namespace smallest_non_representable_l247_247283

def isRepresentable (n : ℕ) : Prop :=
  ∃ a b c d : ℕ, 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ n = (2^a - 2^b) / (2^c - 2^d)

theorem smallest_non_representable : ∀ n : ℕ, 0 < n → ¬ isRepresentable 11 ∧ ∀ k : ℕ, 0 < k ∧ k < 11 → isRepresentable k :=
by sorry

end smallest_non_representable_l247_247283


namespace factorize_a_cubed_minus_a_l247_247864

theorem factorize_a_cubed_minus_a (a : ℝ) : a^3 - a = a * (a + 1) * (a - 1) :=
by
  sorry

end factorize_a_cubed_minus_a_l247_247864


namespace joe_initial_paint_l247_247925

noncomputable def total_paint (P : ℕ) : Prop :=
  let used_first_week := (1 / 4 : ℚ) * P
  let remaining_after_first := (3 / 4 : ℚ) * P
  let used_second_week := (1 / 6 : ℚ) * remaining_after_first
  let total_used := used_first_week + used_second_week
  total_used = 135

theorem joe_initial_paint (P : ℕ) (h : total_paint P) : P = 463 :=
sorry

end joe_initial_paint_l247_247925


namespace factorize_a_cubed_minus_a_l247_247865

theorem factorize_a_cubed_minus_a (a : ℝ) : a^3 - a = a * (a + 1) * (a - 1) :=
sorry

end factorize_a_cubed_minus_a_l247_247865


namespace find_m_l247_247186

theorem find_m (m : ℝ) : (∀ x : ℝ, x^2 - 4 * x + m = 0) → m = 4 :=
by
  intro h
  sorry

end find_m_l247_247186


namespace right_triangle_c_l247_247889

theorem right_triangle_c (a b c : ℝ) (h1 : a = 3) (h2 : b = 4)
  (h3 : (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) ∧ (c^2 = a^2 + b^2 ∨ a^2 = b^2 + c^2 ∨ b^2 = a^2 + c^2)) :
  c = 5 ∨ c = Real.sqrt 7 :=
by
  -- Proof omitted
  sorry

end right_triangle_c_l247_247889


namespace area_arccos_cos_eq_pi_sq_l247_247881

noncomputable def area_bounded_by_arccos_cos : ℝ :=
  ∫ x in (0 : ℝ)..2 * Real.pi, Real.arccos (Real.cos x)

theorem area_arccos_cos_eq_pi_sq :
  area_bounded_by_arccos_cos = Real.pi ^ 2 :=
sorry

end area_arccos_cos_eq_pi_sq_l247_247881


namespace greatest_multiple_of_5_and_6_lt_1000_l247_247382

theorem greatest_multiple_of_5_and_6_lt_1000 : 
  ∃ n, n % 5 = 0 ∧ n % 6 = 0 ∧ n < 1000 ∧ (∀ m, m % 5 = 0 ∧ m % 6 = 0 ∧ m < 1000 → m ≤ n) :=
  sorry

end greatest_multiple_of_5_and_6_lt_1000_l247_247382


namespace no_solution_in_natural_numbers_l247_247507

theorem no_solution_in_natural_numbers (x y z : ℕ) : 
  (x / y : ℚ) + (y / z : ℚ) + (z / x : ℚ) ≠ 1 := 
by sorry

end no_solution_in_natural_numbers_l247_247507


namespace lemonade_glasses_l247_247922

def lemons_total : ℕ := 18
def lemons_per_glass : ℕ := 2

theorem lemonade_glasses : lemons_total / lemons_per_glass = 9 := by
  sorry

end lemonade_glasses_l247_247922


namespace prime_factors_2310_l247_247852

-- Define the number in question
def number : ℕ := 2310

-- The main theorem statement
theorem prime_factors_2310 : ∃ s : Finset ℕ, (∀ p ∈ s, Nat.Prime p) ∧ 
Finset.prod s id = number ∧ Finset.card s = 5 := by
  sorry

end prime_factors_2310_l247_247852


namespace books_not_read_l247_247360

theorem books_not_read (total_books read_books : ℕ) (h1 : total_books = 20) (h2 : read_books = 15) : total_books - read_books = 5 := by
  sorry

end books_not_read_l247_247360


namespace reflection_proof_l247_247815

def reflect_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

noncomputable def initial_point : ℝ × ℝ := (3, -3)
noncomputable def reflected_over_y_axis := reflect_y initial_point
noncomputable def reflected_over_x_axis := reflect_x reflected_over_y_axis

theorem reflection_proof : reflected_over_x_axis = (-3, 3) :=
  by
    -- proof goes here
    sorry

end reflection_proof_l247_247815


namespace initial_apples_correct_l247_247350

-- Define the conditions
def apples_handout : Nat := 5
def pies_made : Nat := 9
def apples_per_pie : Nat := 5

-- Calculate the number of apples used for pies
def apples_for_pies := pies_made * apples_per_pie

-- Define the total number of apples initially
def apples_initial := apples_for_pies + apples_handout

-- State the theorem to prove
theorem initial_apples_correct : apples_initial = 50 :=
by
  sorry

end initial_apples_correct_l247_247350


namespace sufficient_condition_inequalities_l247_247744

theorem sufficient_condition_inequalities (x a : ℝ) :
  (¬ (a-4 < x ∧ x < a+4) → ¬ (1 < x ∧ x < 2)) ↔ -2 ≤ a ∧ a ≤ 5 :=
by
  sorry

end sufficient_condition_inequalities_l247_247744


namespace num_special_matrices_l247_247480

open Matrix

theorem num_special_matrices :
  ∃ (M : Matrix (Fin 4) (Fin 4) ℕ), 
    (∀ i j, 1 ≤ M i j ∧ M i j ≤ 16) ∧ 
    (∀ i j, i < j → M i j < M i (j + 1)) ∧ 
    (∀ i j, i < j → M i j < M (i + 1) j) ∧ 
    (∀ i, i < 3 → M i i < M (i + 1) (i + 1)) ∧ 
    (∀ i, i < 3 → M i (3 - i) < M (i + 1) (2 - i)) ∧ 
    (∃ n, n = 144) :=
sorry

end num_special_matrices_l247_247480


namespace digit_sum_of_4_digit_number_l247_247688

theorem digit_sum_of_4_digit_number (abcd : ℕ) (H1 : 1000 ≤ abcd ∧ abcd < 10000) (erased_digit: ℕ) (H2: erased_digit < 10) (H3 : 100*(abcd / 1000) + 10*(abcd % 1000 / 100) + (abcd % 100 / 10) + erased_digit = 6031): 
    (abcd / 1000 + abcd % 1000 / 100 + abcd % 100 / 10 + abcd % 10 = 20) :=
sorry

end digit_sum_of_4_digit_number_l247_247688


namespace greatest_multiple_of_5_and_6_less_than_1000_l247_247378

theorem greatest_multiple_of_5_and_6_less_than_1000 : 
  ∃ n : ℕ, n < 1000 ∧ n % 5 = 0 ∧ n % 6 = 0 ∧ ∀ m : ℕ, m < 1000 ∧ m % 5 = 0 ∧ m % 6 = 0 → m ≤ n ∧ n = 990 :=
by
  sorry

end greatest_multiple_of_5_and_6_less_than_1000_l247_247378


namespace exists_f_condition_l247_247456

open Nat

-- Define the function φ from ℕ to ℕ
variable (ϕ : ℕ → ℕ)

-- The formal statement capturing the given math proof problem
theorem exists_f_condition (ϕ : ℕ → ℕ) : 
  ∃ (f : ℕ → ℤ), (∀ x : ℕ, f x > f (ϕ x)) :=
  sorry

end exists_f_condition_l247_247456


namespace max_smart_winners_min_total_prize_l247_247590

-- Define relevant constants and conditions
def total_winners := 25
def prize_smart : ℕ := 15
def prize_comprehensive : ℕ := 30

-- Problem 1: Maximum number of winners in "Smartest Brain" competition
theorem max_smart_winners (x : ℕ) (h1 : total_winners = 25)
  (h2 : total_winners - x ≥ 5 * x) : x ≤ 4 :=
sorry

-- Problem 2: Minimum total prize amount
theorem min_total_prize (y : ℕ) (h1 : y ≤ 4)
  (h2 : total_winners = 25)
  (h3 : (total_winners - y) ≥ 5 * y)
  (h4 : prize_smart = 15)
  (h5 : prize_comprehensive = 30) :
  15 * y + 30 * (25 - y) = 690 :=
sorry

end max_smart_winners_min_total_prize_l247_247590


namespace number_of_solutions_of_trig_eq_l247_247760

open Real

/-- The number of values of θ in the interval 0 < θ ≤ 4π that satisfy the equation 
    2 + 4 * sin (2 * θ) - 3 * cos (4 * θ) + 2 * tan(θ) = 0 is 16. -/
theorem number_of_solutions_of_trig_eq : 
  (set_of (λ θ : ℝ, 0 < θ ∧ θ ≤ 4 * π ∧ 2 + 4 * sin (2 * θ) - 3 * cos (4 * θ) + 2 * tan θ = 0)).finite.to_finset.card = 16 :=
sorry

end number_of_solutions_of_trig_eq_l247_247760


namespace function_maximum_at_1_l247_247300

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x - x ^ 2

theorem function_maximum_at_1 :
  ∀ x > 0, (f x ≤ f 1) :=
by
  intro x hx
  have hx_pos : 0 < x := hx
  sorry

end function_maximum_at_1_l247_247300


namespace trigonometric_identity_l247_247888

theorem trigonometric_identity (x : ℝ) (h : Real.tan x = 2) : 
  (6 * Real.sin (2 * x) + 2 * Real.cos (2 * x)) / (Real.cos (2 * x) - 3 * Real.sin (2 * x)) = -2 / 5 := by
  sorry

end trigonometric_identity_l247_247888


namespace quarters_remaining_l247_247656

-- Define the number of quarters Sally originally had
def initialQuarters : Nat := 760

-- Define the number of quarters Sally spent
def spentQuarters : Nat := 418

-- Prove that the number of quarters she has now is 342
theorem quarters_remaining : initialQuarters - spentQuarters = 342 :=
by
  sorry

end quarters_remaining_l247_247656


namespace yellow_yarns_count_l247_247208

theorem yellow_yarns_count (total_scarves red_yarn_count blue_yarn_count yellow_yarns scarves_per_yarn : ℕ) 
  (h1 : 3 = scarves_per_yarn)
  (h2 : red_yarn_count = 2)
  (h3 : blue_yarn_count = 6)
  (h4 : total_scarves = 36)
  :
  yellow_yarns = 4 :=
by 
  sorry

end yellow_yarns_count_l247_247208


namespace amount_of_CaCO3_required_l247_247591

-- Define the balanced chemical reaction
def balanced_reaction (CaCO3 HCl CaCl2 CO2 H2O : ℕ) : Prop :=
  CaCO3 + 2 * HCl = CaCl2 + CO2 + H2O

-- Define the required conditions
def conditions (HCl_req CaCl2_req CO2_req H2O_req : ℕ) : Prop :=
  HCl_req = 4 ∧ CaCl2_req = 2 ∧ CO2_req = 2 ∧ H2O_req = 2

-- The main theorem to be proved
theorem amount_of_CaCO3_required :
  ∃ (CaCO3_req : ℕ), conditions 4 2 2 2 ∧ balanced_reaction CaCO3_req 4 2 2 2 ∧ CaCO3_req = 2 :=
by 
  sorry

end amount_of_CaCO3_required_l247_247591


namespace problem_8_div_64_pow_7_l247_247097

theorem problem_8_div_64_pow_7:
  (64 : ℝ) = (8 : ℝ)^2 →
  8^15 / 64^7 = 8 :=
by
  intro h
  rw [h]
  have : (64^7 : ℝ) = (8^2)^7 := by rw [h]
  rw [this]
  rw [pow_mul]
  field_simp
  norm_num

end problem_8_div_64_pow_7_l247_247097


namespace number_of_people_in_group_l247_247221

theorem number_of_people_in_group :
  ∃ (N : ℕ), (∀ (avg_weight : ℝ), 
  ∃ (new_person_weight : ℝ) (replaced_person_weight : ℝ),
  new_person_weight = 85 ∧ replaced_person_weight = 65 ∧
  avg_weight + 2.5 = ((N * avg_weight + (new_person_weight - replaced_person_weight)) / N) ∧ 
  N = 8) :=
by
  sorry

end number_of_people_in_group_l247_247221


namespace ellipse_equation_fixed_point_l247_247609

/-- Given an ellipse with equation x^2 / a^2 + y^2 / b^2 = 1 where a > b > 0 and eccentricity e = 1/2,
    prove that the equation of the ellipse is x^2 / 4 + y^2 / 3 = 1 -/
theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : a^2 = b^2 + (a / 2)^2) :
  (∀ x y : ℝ, (x^2 / 4 + y^2 / 3 = 1) ↔ (x^2 / a^2 + y^2 / b^2 = 1)) :=
by sorry

/-- Given an ellipse with equation x^2 / 4 + y^2 / 3 = 1,
    if a line l: y = kx + m intersects the ellipse at two points A and B (which are not the left and right vertices),
    and a circle passing through the right vertex of the ellipse has AB as its diameter,
    prove that the line passes through a fixed point and find its coordinates -/
theorem fixed_point (k m : ℝ) :
  (∃ x y, (x = 2 / 7 ∧ y = 0)) :=
by sorry

end ellipse_equation_fixed_point_l247_247609


namespace factor_expression_l247_247150

theorem factor_expression (x : ℝ) : 60 * x + 45 = 15 * (4 * x + 3) :=
by
  sorry

end factor_expression_l247_247150


namespace infinite_n_dividing_2n_minus_n_l247_247066

theorem infinite_n_dividing_2n_minus_n (p : ℕ) (hp : Nat.Prime p) : 
  ∃ᶠ n in at_top, p ∣ 2^n - n :=
sorry

end infinite_n_dividing_2n_minus_n_l247_247066


namespace discrim_of_quadratic_eqn_l247_247404

theorem discrim_of_quadratic_eqn : 
  let a := 3
  let b := -2
  let c := -1
  b^2 - 4 * a * c = 16 := 
by
  sorry

end discrim_of_quadratic_eqn_l247_247404


namespace first_candidate_more_gain_l247_247703

-- Definitions for the salaries, revenues, training costs, and bonuses
def salary1 : ℕ := 42000
def revenue1 : ℕ := 93000
def training_cost_per_month : ℕ := 1200
def training_months : ℕ := 3

def salary2 : ℕ := 45000
def revenue2 : ℕ := 92000
def bonus2_percentage : ℕ := 1

-- Calculate net gains
def net_gain1 : ℕ :=
  revenue1 - salary1 - (training_cost_per_month * training_months)

def net_gain2 : ℕ :=
  revenue2 - salary2 - (salary2 * bonus2_percentage / 100)

def difference_in_gain : ℕ :=
  net_gain1 - net_gain2

-- Theorem statement
theorem first_candidate_more_gain :
  difference_in_gain = 850 :=
by
  -- Proof goes here
  sorry

end first_candidate_more_gain_l247_247703


namespace total_notes_proof_l247_247133

variable (x : Nat)

def total_money := 10350
def fifty_notes_count := 17
def fifty_notes_value := 850  -- 17 * 50
def five_hundred_notes_value := 500 * x
def total_value_proposition := fifty_notes_value + five_hundred_notes_value = total_money

theorem total_notes_proof :
  total_value_proposition -> (fifty_notes_count + x) = 36 :=
by
  intros h
  -- The proof steps would go here, but we use sorry for now.
  sorry

end total_notes_proof_l247_247133


namespace volume_relation_l247_247800

-- Definitions for points and geometry structures
structure Point3D :=
(x : ℝ) (y : ℝ) (z : ℝ)

structure Tetrahedron :=
(A B C D : Point3D)

-- Volume function for Tetrahedron
noncomputable def volume (t : Tetrahedron) : ℝ := sorry

-- Given conditions
variable {A B C D D1 A1 B1 C1 : Point3D} 

-- D_1 is the centroid of triangle ABC
axiom centroid_D1 (A B C D1 : Point3D) : D1 = Point3D.mk ((A.x + B.x + C.x) / 3) ((A.y + B.y + C.y) / 3) ((A.z + B.z + C.z) / 3)

-- Line through A parallel to DD_1 intersects plane BCD at A1
axiom A1_condition (A B C D D1 A1 : Point3D) : sorry
-- Line through B parallel to DD_1 intersects plane ACD at B1
axiom B1_condition (A B C D D1 B1 : Point3D) : sorry
-- Line through C parallel to DD_1 intersects plane ABD at C1
axiom C1_condition (A B C D D1 C1 : Point3D) : sorry

-- Volume relation to be proven
theorem volume_relation (t1 t2 : Tetrahedron) (h : t1.A = A ∧ t1.B = B ∧ t1.C = C ∧ t1.D = D ∧
                                                t2.A = A1 ∧ t2.B = B1 ∧ t2.C = C1 ∧ t2.D = D1) :
  volume t1 = 2 * volume t2 := 
sorry

end volume_relation_l247_247800


namespace part1_part2_l247_247174

open Set

namespace ProofProblem

variable (m : ℝ)

def A (m : ℝ) := {x : ℝ | 0 < x - m ∧ x - m < 3}
def B := {x : ℝ | x ≤ 0 ∨ x ≥ 3}

theorem part1 : (A 1 ∩ B) = {x : ℝ | 3 ≤ x ∧ x < 4} := by
  sorry

theorem part2 : (∀ m, (A m ∪ B) = B ↔ (m ≥ 3 ∨ m ≤ -3)) := by
  sorry

end ProofProblem

end part1_part2_l247_247174


namespace art_museum_visitors_l247_247824

theorem art_museum_visitors 
  (V : ℕ)
  (H1 : ∃ (d : ℕ), d = 130)
  (H2 : ∃ (e u : ℕ), e = u)
  (H3 : ∃ (x : ℕ), x = (3 * V) / 4)
  (H4 : V = (3 * V) / 4 + 130) :
  V = 520 :=
sorry

end art_museum_visitors_l247_247824


namespace range_half_diff_l247_247292

theorem range_half_diff (α β : ℝ) (h1 : -π/2 ≤ α) (h2 : α < β) (h3 : β ≤ π/2) : 
    -π/2 ≤ (α - β) / 2 ∧ (α - β) / 2 < 0 := 
    sorry

end range_half_diff_l247_247292


namespace number_of_correct_judgments_is_zero_l247_247081

theorem number_of_correct_judgments_is_zero :
  (¬ ∀ (x : ℚ), -x ≠ |x|) ∧
  (¬ ∀ (x y : ℚ), -x = y → y = 1 / x) ∧
  (¬ ∀ (x y : ℚ), |x| = |y| → x = y) →
  0 = 0 :=
by
  intros h
  exact rfl

end number_of_correct_judgments_is_zero_l247_247081


namespace value_of_x_l247_247908

theorem value_of_x (x : ℝ) (h : 0.5 * x = 0.25 * 1500 - 30) : x = 690 :=
by
  sorry

end value_of_x_l247_247908


namespace selling_price_before_brokerage_l247_247120

theorem selling_price_before_brokerage (cash_realized : ℝ) (brokerage_rate : ℝ) (final_cash : ℝ) : 
  final_cash = 104.25 → brokerage_rate = 1 / 400 → cash_realized = 104.51 :=
by
  intro h1 h2
  sorry

end selling_price_before_brokerage_l247_247120


namespace factorial_not_div_by_two_pow_l247_247064

theorem factorial_not_div_by_two_pow (n : ℕ) : ¬ (2^n ∣ n!) :=
sorry

end factorial_not_div_by_two_pow_l247_247064


namespace geometric_sequence_sum_terms_l247_247631

noncomputable def geometric_sequence (a_1 : ℕ) (q : ℕ) (n : ℕ) : ℕ :=
  a_1 * q ^ (n - 1)

theorem geometric_sequence_sum_terms :
  ∀ (a_1 q : ℕ), a_1 = 3 → 
  (geometric_sequence 3 q 1 + geometric_sequence 3 q 2 + geometric_sequence 3 q 3 = 21) →
  (q > 0) →
  (geometric_sequence 3 q 3 + geometric_sequence 3 q 4 + geometric_sequence 3 q 5 = 84) :=
by
  intros a_1 q h1 hsum hqpos
  sorry

end geometric_sequence_sum_terms_l247_247631


namespace part1_part2_l247_247459

def setA := {x : ℝ | -3 < x ∧ x < 4}
def setB (a : ℝ) := {x : ℝ | x^2 - 4 * a * x + 3 * a^2 = 0}

theorem part1 (a : ℝ) : (setA ∩ setB a = ∅) ↔ (a ≤ -3 ∨ a ≥ 4) :=
sorry

theorem part2 (a : ℝ) : (setA ∪ setB a = setA) ↔ (-1 < a ∧ a < 4/3) :=
sorry

end part1_part2_l247_247459


namespace mixtape_length_l247_247430

theorem mixtape_length (songs_side1 songs_side2 song_duration : ℕ) 
  (h1 : songs_side1 = 6) 
  (h2 : songs_side2 = 4) 
  (h3 : song_duration = 4) : 
  (songs_side1 + songs_side2) * song_duration = 40 :=
by
  rw [h1, h2, h3]
  norm_num

end mixtape_length_l247_247430


namespace quadrilateral_with_equal_sides_is_rhombus_l247_247552

theorem quadrilateral_with_equal_sides_is_rhombus (a b c d : ℝ) (h1 : a = b) (h2 : b = c) (h3 : c = d) : a = d :=
by
  sorry

end quadrilateral_with_equal_sides_is_rhombus_l247_247552


namespace people_after_five_years_l247_247987

noncomputable def population_in_year : ℕ → ℕ
| 0       => 20
| (k + 1) => 4 * population_in_year k - 18

theorem people_after_five_years : population_in_year 5 = 14382 := by
  sorry

end people_after_five_years_l247_247987


namespace find_sum_of_cubes_l247_247052

-- Define the distinct real numbers p, q, and r
variables {p q r : ℝ}

-- Conditions
-- Distinctness condition
axiom h_distinct : p ≠ q ∧ q ≠ r ∧ r ≠ p

-- Given condition
axiom h_eq : (p^3 + 7) / p = (q^3 + 7) / q ∧ (q^3 + 7) / q = (r^3 + 7) / r

-- Proof goal
theorem find_sum_of_cubes : p^3 + q^3 + r^3 = -21 :=
sorry

end find_sum_of_cubes_l247_247052


namespace total_profit_l247_247432

-- Definitions based on the conditions
def cost_per_dog : ℕ := 250
def num_dogs : ℕ := 2
def num_puppies : ℕ := 6
def selling_price_per_puppy : ℕ := 350

-- Statement of the theorem
theorem total_profit : (num_puppies * selling_price_per_puppy) - (num_dogs * cost_per_dog) = 1600 := 
by
  sorry

end total_profit_l247_247432


namespace combined_average_age_l247_247804

theorem combined_average_age :
  (8 * 35 + 6 * 30) / (8 + 6) = 33 :=
by
  sorry

end combined_average_age_l247_247804


namespace find_angle_QEP_l247_247828

open EuclideanGeometry

-- Define the given conditions
variables {P Q R E : Point}
-- PQR is a right triangle with the right angle at Q
variables (h1 : Triangle P Q R)
variables (hQ : Angle Q P R = π / 2)
-- ∠P is 30 degrees
variables (hP : Angle P Q R = π / 6)
-- PE is the angle bisector of ∠QPR
variables (hBis : AngleBisector P Q R E)

-- The goal: prove ∠QEP = 60 degrees
theorem find_angle_QEP : Angle Q E P = π / 3 := 
by 
  sorry

end find_angle_QEP_l247_247828


namespace solve_system_correct_l247_247929

noncomputable def solve_system (n : ℕ) (x : ℕ → ℝ) : Prop :=
  n > 2 ∧ (∀ k : ℕ, 1 ≤ k ∧ k ≤ n → x k + x (k + 1) = x (k + 2) ^ 2) ∧ 
  x (n + 1) = x 1 ∧ x (n + 2) = x 2 →
  ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → x i = 2

theorem solve_system_correct (n : ℕ) (x : ℕ → ℝ) : solve_system n x := 
sorry

end solve_system_correct_l247_247929


namespace find_numbers_l247_247588

noncomputable def sum_nat (a b : ℕ) : ℕ :=
  (b - a + 1) * (a + b) / 2

theorem find_numbers : 
  {n : ℕ | ∃ (a b : ℕ), 1 ≤ a ∧ a < b ∧ b ≤ 9 ∧ n = 10 * a + b ∧ n = sum_nat a b} = {14, 26, 37, 48, 59} :=
by {
  sorry
}

end find_numbers_l247_247588


namespace rate_of_interest_l247_247840

theorem rate_of_interest (P T SI CI : ℝ) (hP : P = 4000) (hT : T = 2) (hSI : SI = 400) (hCI : CI = 410) :
  ∃ r : ℝ, SI = (P * r * T) / 100 ∧ CI = P * ((1 + r / 100) ^ T - 1) ∧ r = 5 :=
by
  sorry

end rate_of_interest_l247_247840


namespace hard_candy_food_colouring_l247_247692

noncomputable def food_colouring_per_hard_candy (lollipop_use : ℕ) (gummy_use : ℕ)
    (lollipops_per_day : ℕ) (gummies_per_day : ℕ) (hard_candies_per_day : ℕ)
    (total_food_colouring : ℕ) : ℕ := 
by
  -- Let ml_lollipops be the total amount needed for lollipops
  let ml_lollipops := lollipop_use * lollipops_per_day
  -- Let ml_gummy be the total amount needed for gummy candies
  let ml_gummy := gummy_use * gummies_per_day
  -- Let ml_non_hard be the amount for lollipops and gummy candies combined
  let ml_non_hard := ml_lollipops + ml_gummy
  -- Let ml_hard be the amount used for hard candies alone
  let ml_hard := total_food_colouring - ml_non_hard
  -- Compute the food colouring used per hard candy
  exact ml_hard / hard_candies_per_day

theorem hard_candy_food_colouring :
  food_colouring_per_hard_candy 8 3 150 50 20 1950 = 30 :=
by
  unfold food_colouring_per_hard_candy
  sorry

end hard_candy_food_colouring_l247_247692


namespace rhombus_area_2400_l247_247447

noncomputable def area_of_rhombus (x y : ℝ) : ℝ :=
  2 * x * y

theorem rhombus_area_2400 (x y : ℝ) 
  (hx : x = 15) 
  (hy : y = (16 / 3) * x) 
  (rx : 18.75 * 4 * x * y = x * y * (78.75)) 
  (ry : 50 * 4 * x * y = x * y * (200)) : 
  area_of_rhombus 15 80 = 2400 :=
by
  sorry

end rhombus_area_2400_l247_247447


namespace greatest_multiple_of_5_and_6_less_than_1000_l247_247379

theorem greatest_multiple_of_5_and_6_less_than_1000 : 
  ∃ n : ℕ, n < 1000 ∧ n % 5 = 0 ∧ n % 6 = 0 ∧ ∀ m : ℕ, m < 1000 ∧ m % 5 = 0 ∧ m % 6 = 0 → m ≤ n ∧ n = 990 :=
by
  sorry

end greatest_multiple_of_5_and_6_less_than_1000_l247_247379


namespace num_combinations_two_dresses_l247_247131

def num_colors : ℕ := 4
def num_patterns : ℕ := 5

def combinations_first_dress : ℕ := num_colors * num_patterns
def combinations_second_dress : ℕ := (num_colors - 1) * (num_patterns - 1)

theorem num_combinations_two_dresses :
  (combinations_first_dress * combinations_second_dress) = 240 := by
  sorry

end num_combinations_two_dresses_l247_247131


namespace fraction_operation_correct_l247_247242

theorem fraction_operation_correct (a b : ℝ) (h : 0.2 * a + 0.5 * b ≠ 0) : 
  (0.3 * a + b) / (0.2 * a + 0.5 * b) = (3 * a + 10 * b) / (2 * a + 5 * b) :=
sorry

end fraction_operation_correct_l247_247242


namespace percentage_decrease_stock_l247_247719

theorem percentage_decrease_stock (F J M : ℝ)
  (h1 : J = F - 0.10 * F)
  (h2 : M = J - 0.20 * J) :
  (F - M) / F * 100 = 28 := by
sorry

end percentage_decrease_stock_l247_247719


namespace A_worked_days_l247_247125

theorem A_worked_days 
  (W : ℝ)                              -- Total work in arbitrary units
  (A_work_days : ℕ)                    -- Days A can complete the work 
  (B_work_days_remaining : ℕ)          -- Days B takes to complete remaining work
  (B_work_days : ℕ)                    -- Days B can complete the work alone
  (hA : A_work_days = 15)              -- A can do the work in 15 days
  (hB : B_work_days_remaining = 12)    -- B completes the remaining work in 12 days
  (hB_alone : B_work_days = 18)        -- B alone can do the work in 18 days
  :
  ∃ (x : ℕ), x = 5                     -- A worked for 5 days before leaving the job
  := 
  sorry                                 -- Proof not provided

end A_worked_days_l247_247125


namespace cars_per_client_l247_247418

-- Define the conditions
def num_cars : ℕ := 18
def selections_per_car : ℕ := 3
def num_clients : ℕ := 18

-- Define the proof problem as a theorem
theorem cars_per_client :
  (num_cars * selections_per_car) / num_clients = 3 :=
sorry

end cars_per_client_l247_247418


namespace bob_total_profit_l247_247437

def initial_cost (num_dogs : ℕ) (cost_per_dog : ℕ) : ℕ := num_dogs * cost_per_dog

def revenue (num_puppies : ℕ) (price_per_puppy : ℕ) : ℕ := num_puppies * price_per_puppy

def total_profit (initial_cost : ℕ) (revenue : ℕ) : ℕ := revenue - initial_cost

theorem bob_total_profit (c1 : initial_cost 2 250 = 500)
                        (c2 : revenue 6 350 = 2100)
                        (c3 : total_profit 500 2100 = 1600) :
  total_profit (initial_cost 2 250) (revenue 6 350) = 1600 := by
  sorry

end bob_total_profit_l247_247437


namespace value_of_mn_l247_247199

theorem value_of_mn (m n : ℕ) (h_pos_m : 0 < m) (h_pos_n : 0 < n) (h_eq : m^4 - n^4 = 3439) : m * n = 90 := 
by sorry

end value_of_mn_l247_247199


namespace combined_avg_score_l247_247115

theorem combined_avg_score (x : ℕ) : 
  let avgA := 65
  let avgB := 90 
  let avgC := 77 
  let ratioA := 4 
  let ratioB := 6 
  let ratioC := 5 
  let total_students := 15 * x 
  let total_score := (ratioA * avgA + ratioB * avgB + ratioC * avgC) * x
  (total_score / total_students) = 79 := 
by
  sorry

end combined_avg_score_l247_247115


namespace onions_shelf_correct_l247_247670

def onions_on_shelf (initial: ℕ) (sold: ℕ) (added: ℕ) (given_away: ℕ): ℕ :=
  initial - sold + added - given_away

theorem onions_shelf_correct :
  onions_on_shelf 98 65 20 10 = 43 :=
by
  sorry

end onions_shelf_correct_l247_247670


namespace geometric_sequence_increasing_iff_q_gt_one_l247_247482

variables {a_n : ℕ → ℝ} {q : ℝ}

def is_geometric_sequence (a_n : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a_n (n + 1) = a_n n * q

def is_increasing_sequence (a_n : ℕ → ℝ) : Prop :=
  ∀ n, a_n (n + 1) > a_n n

theorem geometric_sequence_increasing_iff_q_gt_one 
  (h1 : ∀ n, 0 < a_n n)
  (h2 : is_geometric_sequence a_n q) :
  is_increasing_sequence a_n ↔ q > 1 :=
by
  sorry

end geometric_sequence_increasing_iff_q_gt_one_l247_247482


namespace k_is_square_l247_247291

theorem k_is_square (a b : ℕ) (h_a : a > 0) (h_b : b > 0) (k : ℕ) (h_k : k > 0)
    (h : (a^2 + b^2) = k * (a * b + 1)) : ∃ (n : ℕ), n^2 = k :=
sorry

end k_is_square_l247_247291


namespace f_plus_2011_is_odd_l247_247723

def f (x : ℝ) : ℝ

constant condition : ∀ α β : ℝ, f (α + β) - (f α + f β) = 2011

theorem f_plus_2011_is_odd : ∀ x : ℝ, f(x) + 2011 = -(f(-x) + 2011) :=
by
  intro x
  sorry

end f_plus_2011_is_odd_l247_247723


namespace events_A_and_D_independent_l247_247546

open_locale big_operators

-- Events: 
def event_A (x : ℕ) : Prop := x % 2 = 1  -- first die is odd
def event_B (y : ℕ) : Prop := y % 2 = 0  -- second die is even
def event_C (x y : ℕ) : Prop := x + y = 6  -- sum of the points is 6
def event_D (x y : ℕ) : Prop := x + y = 7  -- sum of the points is 7

/-- Theorem to verify that events A and D are independent -/
theorem events_A_and_D_independent : 
  (∀ x y : ℕ, 1 ≤ x ∧ x ≤ 6 ∧ 1 ≤ y ∧ y ≤ 6 →
  event_A x ∧ event_D x y → 
  prob (event_A x ∧ event_D x y) = 
  (prob event_A) * (prob (event_D x y))) := sorry

end events_A_and_D_independent_l247_247546


namespace seedlings_total_l247_247079

theorem seedlings_total (seeds_per_packet : ℕ) (packets : ℕ) (total_seedlings : ℕ) 
  (h1 : seeds_per_packet = 7) (h2 : packets = 60) : total_seedlings = 420 :=
by {
  sorry
}

end seedlings_total_l247_247079


namespace compare_y_values_l247_247623

theorem compare_y_values (y1 y2 : ℝ) 
  (hA : y1 = (-1)^2 - 4*(-1) - 3) 
  (hB : y2 = 1^2 - 4*1 - 3) : y1 > y2 :=
by
  sorry

end compare_y_values_l247_247623


namespace a2_equals_3_l247_247911

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

theorem a2_equals_3 (a : ℕ → ℕ) (S3 : ℕ) (h1 : a 1 = 1) (h2 : a 1 + a 2 + a 3 = 9) : a 2 = 3 :=
by
  sorry

end a2_equals_3_l247_247911


namespace factorize_cubic_l247_247878

theorem factorize_cubic (a : ℝ) : a^3 - a = a * (a + 1) * (a - 1) :=
sorry

end factorize_cubic_l247_247878


namespace smallest_r_minus_p_l247_247411

theorem smallest_r_minus_p 
  (p q r : ℕ) (h₀ : p * q * r = 362880) (h₁ : p < q) (h₂ : q < r) : 
  r - p = 126 :=
sorry

end smallest_r_minus_p_l247_247411


namespace total_time_to_pump_540_gallons_l247_247348

-- Definitions for the conditions
def initial_rate : ℝ := 360  -- gallons per hour
def increased_rate : ℝ := 480 -- gallons per hour
def target_volume : ℝ := 540  -- total gallons
def first_interval : ℝ := 0.5 -- first 30 minutes as fraction of hour

-- Proof problem statement
theorem total_time_to_pump_540_gallons : 
  (first_interval * initial_rate) + ((target_volume - (first_interval * initial_rate)) / increased_rate) * 60 = 75 := by
  sorry

end total_time_to_pump_540_gallons_l247_247348


namespace initial_apps_count_l247_247849

theorem initial_apps_count (x A : ℕ) 
  (h₁ : A - 18 + x = 5) : A = 23 - x :=
by
  sorry

end initial_apps_count_l247_247849


namespace pattern_equation_l247_247940

theorem pattern_equation (n : ℕ) (h : 0 < n) : n * (n + 2) + 1 = (n + 1) ^ 2 := by
  sorry

end pattern_equation_l247_247940


namespace hyperbola_foci_l247_247078

-- Define the conditions and the question
def hyperbola_equation (x y : ℝ) : Prop := 
  x^2 - 4 * y^2 - 6 * x + 24 * y - 11 = 0

-- The foci of the hyperbola 
def foci (x1 x2 y1 y2 : ℝ) : Prop := 
  (x1, y1) = (3, 3 + 2 * Real.sqrt 5) ∨ (x2, y2) = (3, 3 - 2 * Real.sqrt 5)

-- The proof statement
theorem hyperbola_foci :
  ∃ x1 x2 y1 y2 : ℝ, hyperbola_equation x1 y1 ∧ foci x1 x2 y1 y2 :=
sorry

end hyperbola_foci_l247_247078


namespace labourer_saving_after_debt_clearance_l247_247219

variable (averageExpenditureFirst6Months : ℕ)
variable (monthlyIncome : ℕ)
variable (reducedMonthlyExpensesNext4Months : ℕ)

theorem labourer_saving_after_debt_clearance (h1 : averageExpenditureFirst6Months = 90)
                                              (h2 : monthlyIncome = 81)
                                              (h3 : reducedMonthlyExpensesNext4Months = 60) :
    (monthlyIncome * 4) - ((reducedMonthlyExpensesNext4Months * 4) + 
    ((averageExpenditureFirst6Months * 6) - (monthlyIncome * 6))) = 30 := by
  sorry

end labourer_saving_after_debt_clearance_l247_247219


namespace correct_equation_l247_247774

theorem correct_equation (x : ℕ) :
  (30 * x + 8 = 31 * x - 26) := by
  sorry

end correct_equation_l247_247774


namespace percent_within_one_standard_deviation_l247_247693

variable (m d : ℝ)
variable (distribution : ℝ → ℝ)
variable (symmetric_about_mean : ∀ x, distribution (m + x) = distribution (m - x))
variable (percent_less_than_m_plus_d : distribution (m + d) = 0.84)

theorem percent_within_one_standard_deviation :
  distribution (m + d) - distribution (m - d) = 0.68 :=
sorry

end percent_within_one_standard_deviation_l247_247693


namespace largest_value_n_under_100000_l247_247398

theorem largest_value_n_under_100000 :
  ∃ n : ℕ,
    0 ≤ n ∧
    n < 100000 ∧
    (10 * (n - 3)^5 - n^2 + 20 * n - 30) % 7 = 0 ∧
    n = 99999 :=
sorry

end largest_value_n_under_100000_l247_247398


namespace factorize_cubic_l247_247877

theorem factorize_cubic (a : ℝ) : a^3 - a = a * (a + 1) * (a - 1) :=
sorry

end factorize_cubic_l247_247877


namespace common_difference_zero_l247_247633

theorem common_difference_zero (a b c : ℕ) 
  (h_seq : ∃ d : ℕ, a = b + d ∧ b = c + d)
  (h_eq : (c - b) / a + (a - c) / b + (b - a) / c = 0) : 
  ∀ d : ℕ, d = 0 :=
by sorry

end common_difference_zero_l247_247633


namespace distance_travelled_l247_247247

variables (S D : ℝ)

-- conditions
def cond1 : Prop := D = S * 7
def cond2 : Prop := D = (S + 12) * 5

-- Define the main theorem
theorem distance_travelled (h1 : cond1 S D) (h2 : cond2 S D) : D = 210 :=
by {
  sorry
}

end distance_travelled_l247_247247


namespace optimal_voter_split_l247_247532

-- Definitions
variables (Voters : Type) [fintype Voters] (n : ℕ)
variables (supports_miraflores : Voters → Prop)
variables [decidable_pred supports_miraflores]

-- Conditions
def half_supports_miraflores := fintype.card { v // supports_miraflores v } = n
def half_supports_maloney := fintype.card { v // ¬ supports_miraflores v } = n

-- Question (translated to a theorem)
theorem optimal_voter_split (h_m : half_supports_miraflores Voters n supports_miraflores)
    (h_d: half_supports_maloney Voters n supports_miraflores) :
  ∃ (D1 D2 : finset Voters), 
    ((D1 = {v | supports_miraflores v}) ∧ 
    (D2 = {v | ¬supports_miraflores v}) ∧ 
    (∀ v, v ∈ D1 ∨ v ∈ D2) ∧ 
    (∀ v, ¬ (v ∈ D1 ∧ v ∈ D2)) ∧ 
    (finset.card D1 = 1) ∧ 
    (finset.card D2 = 2 * n - 1)) :=
sorry

end optimal_voter_split_l247_247532


namespace average_weight_of_whole_class_l247_247556

def num_students_a : ℕ := 50
def num_students_b : ℕ := 70
def avg_weight_a : ℚ := 50
def avg_weight_b : ℚ := 70

theorem average_weight_of_whole_class :
  (num_students_a * avg_weight_a + num_students_b * avg_weight_b) / (num_students_a + num_students_b) = 61.67 := by
  sorry

end average_weight_of_whole_class_l247_247556


namespace intersection_A_B_l247_247016

def A : Set ℤ := {0, 1, 2}
def B : Set ℤ := {x | x - 2 < 0}

theorem intersection_A_B : A ∩ B = {0, 1} :=
by
  sorry

end intersection_A_B_l247_247016


namespace main_theorem_l247_247465

variable (x : ℝ)

-- Define proposition p
def p : Prop := ∃ x0 : ℝ, x0^2 < x0

-- Define proposition q
def q : Prop := ∀ x : ℝ, x^2 - x + 1 > 0

-- Main proof problem
theorem main_theorem : p ∧ q := 
by {
  sorry
}

end main_theorem_l247_247465


namespace fourth_quadrangle_area_l247_247332

theorem fourth_quadrangle_area (S1 S2 S3 S4 : ℝ) (h : S1 + S4 = S2 + S3) : S4 = S2 + S3 - S1 :=
by
  sorry

end fourth_quadrangle_area_l247_247332


namespace power_division_l247_247093

theorem power_division (h : 64 = 8^2) : 8^15 / 64^7 = 8 :=
by
  rw [h]
  rw [pow_mul]
  rw [div_eq_mul_inv]
  rw [pow_sub]
  rw [mul_inv_cancel]
  exact rfl

end power_division_l247_247093


namespace sum_of_possible_a_values_l247_247018

-- Define the original equation as a predicate
def equation (a x : ℤ) : Prop :=
  x - (2 - a * x) / 6 = x / 3 - 1

-- State that x is a non-negative integer
def nonneg_integer (x : ℤ) : Prop := x ≥ 0

-- The main theorem to prove
theorem sum_of_possible_a_values : 
  (∑ a in {a : ℤ | ∃ x : ℤ, nonneg_integer x ∧ equation a x}, a) = -19 :=
sorry

end sum_of_possible_a_values_l247_247018


namespace flower_cost_l247_247077

theorem flower_cost (F : ℕ) (h1 : F + (F + 20) + (F - 2) = 45) : F = 9 :=
by
  sorry

end flower_cost_l247_247077


namespace simplify_expression_l247_247939

noncomputable def simplify_expr (a b : ℝ) : ℝ :=
  (3 * a^5 * b^3 + a^4 * b^2) / (-(a^2 * b)^2) - (2 + a) * (2 - a) - a * (a - 5 * b)

theorem simplify_expression (a b : ℝ) :
  simplify_expr a b = 8 * a * b - 3 := 
by
  sorry

end simplify_expression_l247_247939


namespace last_number_written_on_sheet_l247_247651

/-- The given problem is to find the last number written on a sheet with specific rules. 
Given:
- The sheet has dimensions of 100 characters in width and 100 characters in height.
- Numbers are written successively with a space between each number.
- If the end of a line is reached, the next number continues at the beginning of the next line.

We need to prove that the last number written on the sheet is 2220.
-/
theorem last_number_written_on_sheet :
  ∃ (n : ℕ), n = 2220 ∧ 
    let width := 100
    let height := 100
    let sheet_size := width * height
    let write_number size occupied_space := occupied_space + size + 1 
    ∃ (numbers : ℕ → ℕ) (space_per_number : ℕ → ℕ),
      ( ∀ i, space_per_number i = if numbers i < 10 then 2 else if numbers i < 100 then 3 else if numbers i < 1000 then 4 else 5 ) ∧
      ∃ (current_space : ℕ), 
        (current_space ≤ sheet_size) ∧
        (∀ i, current_space = write_number (space_per_number i) current_space ) :=
sorry

end last_number_written_on_sheet_l247_247651


namespace greatest_multiple_of_5_and_6_less_than_1000_l247_247393

theorem greatest_multiple_of_5_and_6_less_than_1000 : 
    ∃ k, (k % 5 = 0) ∧ (k % 6 = 0) ∧ (k < 1000) ∧ (∀ m, (m % 5 = 0) ∧ (m % 6 = 0) ∧ (m < 1000) → m ≤ k) :=
begin
  use 990,
  repeat { split },
  { norm_num }, -- 990 % 5 = 0
  { norm_num }, -- 990 % 6 = 0
  { norm_num }, -- 990 < 1000
  { intros m hm, 
    cases hm with h5 h6,
    cases h6 with h6 hlt,
    have : m % 30 = 0 := by { sorry }, -- Show that m is a multiple of LCM(5, 6)
    apply le_of_lt,
    have hle : m/30 < ↑(1000/30) := by { sorry }, -- Compare the greatest multiple of 30 less than 1000
    exact hle,
  }
end

end greatest_multiple_of_5_and_6_less_than_1000_l247_247393


namespace cost_expression_A_cost_expression_B_cost_comparison_10_students_cost_comparison_4_students_l247_247415

-- Define the conditions
def ticket_full_price : ℕ := 240
def discount_A : ℕ := ticket_full_price / 2
def discount_B (x : ℕ) : ℕ := 144 * (x + 1)

-- Algebraic expressions provided in the answer
def cost_A (x : ℕ) : ℕ := discount_A * x + ticket_full_price
def cost_B (x : ℕ) : ℕ := 144 * (x + 1)

-- Proofs for the specific cases
theorem cost_expression_A (x : ℕ) : cost_A x = 120 * x + 240 := by
  sorry

theorem cost_expression_B (x : ℕ) : cost_B x = 144 * (x + 1) := by
  sorry

theorem cost_comparison_10_students : cost_A 10 < cost_B 10 := by
  sorry

theorem cost_comparison_4_students : cost_A 4 = cost_B 4 := by
  sorry

end cost_expression_A_cost_expression_B_cost_comparison_10_students_cost_comparison_4_students_l247_247415


namespace greatest_multiple_of_5_and_6_less_than_1000_l247_247391

theorem greatest_multiple_of_5_and_6_less_than_1000 : 
    ∃ k, (k % 5 = 0) ∧ (k % 6 = 0) ∧ (k < 1000) ∧ (∀ m, (m % 5 = 0) ∧ (m % 6 = 0) ∧ (m < 1000) → m ≤ k) :=
begin
  use 990,
  repeat { split },
  { norm_num }, -- 990 % 5 = 0
  { norm_num }, -- 990 % 6 = 0
  { norm_num }, -- 990 < 1000
  { intros m hm, 
    cases hm with h5 h6,
    cases h6 with h6 hlt,
    have : m % 30 = 0 := by { sorry }, -- Show that m is a multiple of LCM(5, 6)
    apply le_of_lt,
    have hle : m/30 < ↑(1000/30) := by { sorry }, -- Compare the greatest multiple of 30 less than 1000
    exact hle,
  }
end

end greatest_multiple_of_5_and_6_less_than_1000_l247_247391


namespace number_of_ways_to_choose_officers_l247_247327

open Nat

theorem number_of_ways_to_choose_officers (n : ℕ) (h : n = 8) : 
  n * (n - 1) * (n - 2) = 336 := by
  sorry

end number_of_ways_to_choose_officers_l247_247327


namespace only_triple_l247_247728

theorem only_triple (a b c : ℕ) (h1 : (a * b + 1) % c = 0)
                                (h2 : (a * c + 1) % b = 0)
                                (h3 : (b * c + 1) % a = 0) :
    (a = 1 ∧ b = 1 ∧ c = 1) :=
by
  sorry

end only_triple_l247_247728


namespace factorize_cubic_expression_l247_247874

variable (a : ℝ)

theorem factorize_cubic_expression : a^3 - a = a * (a + 1) * (a - 1) := 
sorry

end factorize_cubic_expression_l247_247874


namespace find_number_of_people_l247_247222

-- Assuming the following conditions:
-- The average weight of a group increases by 2.5 kg when a new person replaces one weighing 65 kg
-- The weight of the new person is 85 kg

def avg_weight_increase (N : ℕ) (old_weight new_weight : ℚ) : Prop :=
  let weight_diff := new_weight - old_weight
  let total_increase := 2.5 * N
  weight_diff = total_increase

theorem find_number_of_people :
  avg_weight_increase N 65 85 → N = 8 :=
by
  intros h
  sorry -- complete proof is not required

end find_number_of_people_l247_247222


namespace compute_expression_l247_247720

theorem compute_expression :
  18 * (216 / 3 + 36 / 6 + 4 / 9 + 2 + 1 / 18) = 1449 :=
by
  sorry

end compute_expression_l247_247720


namespace melanies_plums_l247_247342

variable (pickedPlums : ℕ)
variable (gavePlums : ℕ)

theorem melanies_plums (h1 : pickedPlums = 7) (h2 : gavePlums = 3) : (pickedPlums - gavePlums) = 4 :=
by
  sorry

end melanies_plums_l247_247342


namespace train_length_l247_247573

-- Definitions of speeds and times
def speed_person_A := 5 / 3.6 -- in meters per second
def speed_person_B := 15 / 3.6 -- in meters per second
def time_to_overtake_A := 36 -- in seconds
def time_to_overtake_B := 45 -- in seconds

-- The length of the train
theorem train_length :
  ∃ x : ℝ, x = 500 :=
by
  sorry

end train_length_l247_247573


namespace tangent_curves_line_exists_l247_247913

theorem tangent_curves_line_exists (a : ℝ) :
  (∃ l : ℝ → ℝ, ∃ x₀ : ℝ, l 1 = 0 ∧ ∀ x, (l x = x₀^3 ∧ l x = a * x^2 + (15 / 4) * x - 9)) →
  a = -25/64 ∨ a = -1 :=
by
  sorry

end tangent_curves_line_exists_l247_247913


namespace tail_growth_problem_l247_247557

def initial_tail_length : ℕ := 1
def final_tail_length : ℕ := 864
def transformations (ordinary_count cowardly_count : ℕ) : ℕ := initial_tail_length * 2^ordinary_count * 3^cowardly_count

theorem tail_growth_problem (ordinary_count cowardly_count : ℕ) :
  transformations ordinary_count cowardly_count = final_tail_length ↔ ordinary_count = 5 ∧ cowardly_count = 3 :=
by
  sorry

end tail_growth_problem_l247_247557


namespace greatest_multiple_of_5_and_6_less_than_1000_l247_247381

theorem greatest_multiple_of_5_and_6_less_than_1000 : 
  ∃ n : ℕ, n < 1000 ∧ n % 5 = 0 ∧ n % 6 = 0 ∧ ∀ m : ℕ, m < 1000 ∧ m % 5 = 0 ∧ m % 6 = 0 → m ≤ n ∧ n = 990 :=
by
  sorry

end greatest_multiple_of_5_and_6_less_than_1000_l247_247381


namespace largest_three_digit_in_pascal_triangle_l247_247241

-- Define Pascal's triangle and binomial coefficient
def pascal (n k : ℕ) : ℕ := Nat.choose n k

-- State the theorem about the first appearance of the number 999 in Pascal's triangle
theorem largest_three_digit_in_pascal_triangle :
  ∃ (n : ℕ), n = 1000 ∧ ∃ (k : ℕ), pascal n k = 999 :=
sorry

end largest_three_digit_in_pascal_triangle_l247_247241


namespace positive_integer_pairs_count_l247_247282

theorem positive_integer_pairs_count :
  (∃ (n : ℕ), n = 31 ∧ 
  (∀ (a b : ℕ), a^2 + b^2 < 2013 ∧ a^2 * b ∣ b^3 - a^3 → a = b)) :=
by
  sorry

end positive_integer_pairs_count_l247_247282


namespace gcd_154_308_462_l247_247108

theorem gcd_154_308_462 : Nat.gcd (Nat.gcd 154 308) 462 = 154 := by
  sorry

end gcd_154_308_462_l247_247108


namespace reflection_y_axis_l247_247735

open Matrix

def reflection_y_axis_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![(-1 : ℝ), 0], ![0, (1 : ℝ)]]

theorem reflection_y_axis (M : Matrix (Fin 2) (Fin 2) ℝ) :
  (M ⬝ ![![1, 0], ![0, 1]]) = reflection_y_axis_matrix :=
by sorry

end reflection_y_axis_l247_247735


namespace value_of_y_l247_247188

theorem value_of_y (x y z : ℕ) (h_positive_x : 0 < x) (h_positive_y : 0 < y) (h_positive_z : 0 < z)
    (h_sum : x + y + z = 37) (h_eq : 4 * x = 6 * z) : y = 32 :=
sorry

end value_of_y_l247_247188


namespace bag_contains_twenty_cookies_l247_247412

noncomputable def cookies_in_bag 
  (total_calories : ℕ) 
  (calories_per_cookie : ℕ)
  (bags_in_box : ℕ)
  : ℕ :=
  total_calories / (calories_per_cookie * bags_in_box)

theorem bag_contains_twenty_cookies 
  (H1 : total_calories = 1600) 
  (H2 : calories_per_cookie = 20) 
  (H3 : bags_in_box = 4)
  : cookies_in_bag total_calories calories_per_cookie bags_in_box = 20 := 
by
  have h1 : total_calories = 1600 := H1
  have h2 : calories_per_cookie = 20 := H2
  have h3 : bags_in_box = 4 := H3
  sorry

end bag_contains_twenty_cookies_l247_247412


namespace find_second_term_geometric_sequence_l247_247359

noncomputable def second_term_geometric_sequence (a r : ℝ) : ℝ :=
  a * r

theorem find_second_term_geometric_sequence:
  ∀ (a r : ℝ),
    a * r^2 = 12 →
    a * r^3 = 18 →
    second_term_geometric_sequence a r = 8 :=
by
  intros a r h1 h2
  sorry

end find_second_term_geometric_sequence_l247_247359


namespace square_lawn_area_l247_247501

theorem square_lawn_area (map_scale : ℝ) (map_edge_length_cm : ℝ) (actual_edge_length_m : ℝ) (actual_area_m2 : ℝ) 
  (h1 : map_scale = 1 / 5000) 
  (h2 : map_edge_length_cm = 4) 
  (h3 : actual_edge_length_m = (map_edge_length_cm / map_scale) / 100)
  (h4 : actual_area_m2 = actual_edge_length_m^2)
  : actual_area_m2 = 400 := 
by 
  sorry

end square_lawn_area_l247_247501


namespace bridge_length_proof_l247_247555

open Real

def train_length : ℝ := 100
def train_speed_kmh : ℝ := 45
def crossing_time_s: ℝ := 30

noncomputable def bridge_length : ℝ :=
  let train_speed_ms := (train_speed_kmh * 1000) / 3600
  let total_distance := train_speed_ms * crossing_time_s
  total_distance - train_length

theorem bridge_length_proof : bridge_length = 275 := 
by
  sorry

end bridge_length_proof_l247_247555


namespace johns_eggs_per_week_l247_247781

noncomputable def total_emus (pens : ℕ) (emus_per_pen : ℕ) : ℕ :=
  pens * emus_per_pen

noncomputable def female_emus (total : ℕ) : ℕ :=
  total / 2

noncomputable def eggs_per_week (females : ℕ) (days_per_week : ℕ) : ℕ :=
  females * days_per_week

theorem johns_eggs_per_week :
  let pens := 4 in
  let emus_per_pen := 6 in
  let days_per_week := 7 in
  let total := total_emus pens emus_per_pen in
  let females := female_emus total in
  eggs_per_week females days_per_week = 84 :=
by
  sorry

end johns_eggs_per_week_l247_247781


namespace prob_2_out_of_5_exactly_A_and_B_l247_247483

noncomputable def probability_exactly_A_and_B_selected (students : List String) : ℚ :=
  if students = ["A", "B", "C", "D", "E"] then 1 / 10 else 0

theorem prob_2_out_of_5_exactly_A_and_B :
  probability_exactly_A_and_B_selected ["A", "B", "C", "D", "E"] = 1 / 10 :=
by 
  sorry

end prob_2_out_of_5_exactly_A_and_B_l247_247483


namespace min_value_3x_plus_4y_l247_247476

theorem min_value_3x_plus_4y (x y : ℝ) (h_pos : 0 < x ∧ 0 < y) (h_eq : x + 3*y = 5*x*y) :
  ∃ (c : ℝ), (∀ x y : ℝ, 0 < x ∧ 0 < y ∧ x + 3 * y = 5 * x * y → 3 * x + 4 * y ≥ c) ∧ c = 5 :=
sorry

end min_value_3x_plus_4y_l247_247476


namespace legs_total_l247_247975

def number_of_legs_bee := 6
def number_of_legs_spider := 8
def number_of_bees := 5
def number_of_spiders := 2
def total_legs := number_of_bees * number_of_legs_bee + number_of_spiders * number_of_legs_spider

theorem legs_total : total_legs = 46 := by
  sorry

end legs_total_l247_247975


namespace not_possible_linear_poly_conditions_l247_247338

theorem not_possible_linear_poly_conditions (a b : ℝ):
    ¬ (abs (b - 1) < 1 ∧ abs (a + b - 3) < 1 ∧ abs (2 * a + b - 9) < 1) := 
by
    sorry

end not_possible_linear_poly_conditions_l247_247338


namespace measure_of_angle_S_l247_247036

-- Define the angles in the pentagon PQRST
variables (P Q R S T : ℝ)
-- Assume the conditions from the problem
variables (h1 : P = Q)
variables (h2 : Q = R)
variables (h3 : S = T)
variables (h4 : P = S - 30)
-- Assume the sum of angles in a pentagon is 540 degrees
variables (h5 : P + Q + R + S + T = 540)

theorem measure_of_angle_S :
  S = 126 := by
  -- placeholder for the actual proof
  sorry

end measure_of_angle_S_l247_247036


namespace constant_term_2x3_minus_1_over_sqrtx_pow_7_l247_247592

noncomputable def constant_term_in_expansion (n : ℕ) (x : ℝ) : ℝ :=
  (2 : ℝ) * (Nat.choose 7 6 : ℝ)

theorem constant_term_2x3_minus_1_over_sqrtx_pow_7 :
  constant_term_in_expansion 7 (2 : ℝ) = 14 :=
by
  -- proof is omitted
  sorry

end constant_term_2x3_minus_1_over_sqrtx_pow_7_l247_247592


namespace minimum_value_xyz_l247_247337

theorem minimum_value_xyz (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 1) : 
  ∃ m : ℝ, m = 16 ∧ ∀ w, w = (x + y) / (x * y * z) → w ≥ m :=
by
  sorry

end minimum_value_xyz_l247_247337


namespace sin_double_angle_l247_247011

theorem sin_double_angle (x : ℝ)
  (h : Real.sin (x + Real.pi / 4) = 4 / 5) :
  Real.sin (2 * x) = 7 / 25 :=
by
  sorry

end sin_double_angle_l247_247011


namespace gg3_eq_585_over_368_l247_247611

def g (x : ℚ) : ℚ := 2 * x⁻¹ + (2 * x⁻¹) / (1 + 2 * x⁻¹)

theorem gg3_eq_585_over_368 : g (g 3) = 585 / 368 := 
  sorry

end gg3_eq_585_over_368_l247_247611


namespace prob_sum_eighteen_l247_247111

-- Define the probability space of a single die.
def die_prob_space := {n : ℕ | 1 ≤ n ∧ n ≤ 6}

-- Define the total probability space for three dice being rolled.
def three_dice_prob_space := {abc : ℕ × ℕ × ℕ | (abc.1 ∈ die_prob_space) ∧ (abc.2 ∈ die_prob_space) ∧ (abc.3 ∈ die_prob_space)}

-- Define the event of interest: the sum of the three dice equals 18.
def event_sum_eighteen (abc : ℕ × ℕ × ℕ) : Prop := abc.1 + abc.2 + abc.3 = 18

-- Define the probability of a single die landing on 6.
def prob_single_die_six := 1 / 6

-- Define the probability of three dice landing on a specific result (all sixes).
noncomputable def prob_all_six := prob_single_die_six ^ 3

-- State the theorem that the probability of the event "sum = 18" is 1 / 216.
theorem prob_sum_eighteen : probability (three_dice_prob_space) (event_sum_eighteen) = 1 / 216 :=
sorry

end prob_sum_eighteen_l247_247111


namespace cost_per_load_is_25_cents_l247_247159

-- Define the given conditions
def loads_per_bottle : ℕ := 80
def usual_price_per_bottle : ℕ := 2500 -- in cents
def sale_price_per_bottle : ℕ := 2000 -- in cents
def bottles_bought : ℕ := 2

-- Defining the total cost and total loads
def total_cost : ℕ := bottles_bought * sale_price_per_bottle
def total_loads : ℕ := bottles_bought * loads_per_bottle

-- Define the cost per load in cents
def cost_per_load_in_cents : ℕ := (total_cost * 100) / total_loads

-- Formal proof statement
theorem cost_per_load_is_25_cents 
    (h1 : loads_per_bottle = 80)
    (h2 : usual_price_per_bottle = 2500)
    (h3 : sale_price_per_bottle = 2000)
    (h4 : bottles_bought = 2)
    (h5 : total_cost = bottles_bought * sale_price_per_bottle)
    (h6 : total_loads = bottles_bought * loads_per_bottle)
    (h7 : cost_per_load_in_cents = (total_cost * 100) / total_loads):
  cost_per_load_in_cents = 25 := by
  sorry

end cost_per_load_is_25_cents_l247_247159


namespace solve_system_l247_247347

-- Define the system of equations
def eq1 (x y : ℚ) : Prop := 4 * x - 3 * y = -10
def eq2 (x y : ℚ) : Prop := 6 * x + 5 * y = -13

-- Define the solution
def solution (x y : ℚ) : Prop := x = -89 / 38 ∧ y = 0.21053

-- Prove that the given solution satisfies both equations
theorem solve_system : ∃ x y : ℚ, eq1 x y ∧ eq2 x y ∧ solution x y :=
by
  sorry

end solve_system_l247_247347


namespace three_digit_minuends_count_l247_247061

theorem three_digit_minuends_count :
  ∀ a b c : ℕ, a - c = 4 ∧ 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 →
  (∃ n : ℕ, n = 100 * a + 10 * b + c ∧ 100 * a + 10 * b + c - 396 = 100 * c + 10 * b + a) →
  ∃ count : ℕ, count = 50 :=
by
  sorry

end three_digit_minuends_count_l247_247061


namespace solution_set_of_inequality_l247_247235

-- We define the inequality condition
def inequality (x : ℝ) : Prop := (x - 3) * (x + 2) < 0

-- We need to state that for all real numbers x, iff x satisfies the inequality,
-- then x must be within the interval (-2, 3).
theorem solution_set_of_inequality :
  ∀ x : ℝ, inequality x ↔ -2 < x ∧ x < 3 :=
by {
   sorry
}

end solution_set_of_inequality_l247_247235


namespace babysitting_earnings_l247_247145

theorem babysitting_earnings
  (cost_video_game : ℕ)
  (cost_candy : ℕ)
  (hours_worked : ℕ)
  (amount_left : ℕ)
  (total_earned : ℕ)
  (earnings_per_hour : ℕ) :
  cost_video_game = 60 →
  cost_candy = 5 →
  hours_worked = 9 →
  amount_left = 7 →
  total_earned = cost_video_game + cost_candy + amount_left →
  earnings_per_hour = total_earned / hours_worked →
  earnings_per_hour = 8 :=
by
  intros h_game h_candy h_hours h_left h_total_earned h_earn_per_hour
  rw [h_game, h_candy] at h_total_earned
  simp at h_total_earned
  have h_total_earned : total_earned = 72 := by linarith
  rw [h_total_earned, h_hours] at h_earn_per_hour
  simp at h_earn_per_hour
  assumption

end babysitting_earnings_l247_247145


namespace matt_books_second_year_l247_247344

-- Definitions based on the conditions
variables (M : ℕ) -- number of books Matt read last year
variables (P : ℕ) -- number of books Pete read last year

-- Pete read twice as many books as Matt last year
def pete_read_last_year (M : ℕ) : ℕ := 2 * M

-- This year, Pete doubles the number of books he read last year
def pete_read_this_year (M : ℕ) : ℕ := 2 * (2 * M)

-- Matt reads 50% more books this year than he did last year
def matt_read_this_year (M : ℕ) : ℕ := M + M / 2

-- Pete read 300 books across both years
def total_books_pete_read_last_and_this_year (M : ℕ) : ℕ :=
  pete_read_last_year M + pete_read_this_year M

-- Prove that Matt read 75 books in his second year
theorem matt_books_second_year (M : ℕ) (h : total_books_pete_read_last_and_this_year M = 300) :
  matt_read_this_year M = 75 :=
by sorry

end matt_books_second_year_l247_247344


namespace find_a_l247_247042

theorem find_a (a b c d : ℤ) 
  (h1 : d + 0 = 2)
  (h2 : c + 2 = 2)
  (h3 : b + 0 = 4)
  (h4 : a + 4 = 0) : 
  a = -4 := 
sorry

end find_a_l247_247042


namespace minimum_value_l247_247029

theorem minimum_value (n : ℝ) (h : n > 0) : n + 32 / n^2 ≥ 6 := 
sorry

end minimum_value_l247_247029


namespace sequence_a_200_l247_247231

theorem sequence_a_200 :
  let a : ℕ → ℕ := λ n, if n = 1 then 2 else a (n - 1) + 2 * a (n - 1) / (n - 1)
  in a 200 = 40200 := by
  let a : ℕ → ℕ
  sorry

end sequence_a_200_l247_247231


namespace geometric_sequence_S4_l247_247040

-- Definitions from the conditions
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
∃ q : ℝ, (∀ n : ℕ, a (n + 1) = q * a n)

def sum_of_geometric_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
∀ n : ℕ, S n = a 1 * ((1 - (a 2 / a 1)^(n+1)) / (1 - (a 2 / a 1)))

def given_condition (S : ℕ → ℝ) : Prop :=
S 7 - 4 * S 6 + 3 * S 5 = 0

-- Problem statement to prove
theorem geometric_sequence_S4 (a : ℕ → ℝ) (S : ℕ → ℝ) (h_geom : is_geometric_sequence a)
  (h_a1 : a 1 = 1) (h_sum : sum_of_geometric_sequence a S) (h_cond : given_condition S) :
  S 4 = 40 := 
sorry

end geometric_sequence_S4_l247_247040


namespace find_n_l247_247167

theorem find_n : (∃ n : ℕ, 2^3 * 8^3 = 2^(2 * n)) ↔ n = 6 :=
by
  sorry

end find_n_l247_247167


namespace anthony_lunch_money_l247_247990

-- Define the costs as given in the conditions
def juice_box_cost : ℕ := 27
def cupcake_cost : ℕ := 40
def amount_left : ℕ := 8

-- Define the total amount needed for lunch every day
def total_amount_for_lunch : ℕ := juice_box_cost + cupcake_cost + amount_left

theorem anthony_lunch_money : total_amount_for_lunch = 75 := by
  -- This is where the proof would go.
  sorry

end anthony_lunch_money_l247_247990


namespace smaller_triangle_area_14_365_l247_247685

noncomputable def smaller_triangle_area (A : ℝ) (H_reduction : ℝ) : ℝ :=
  A * (H_reduction)^2

theorem smaller_triangle_area_14_365 :
  smaller_triangle_area 34 0.65 = 14.365 :=
by
  -- Proof will be provided here
  sorry

end smaller_triangle_area_14_365_l247_247685


namespace angle_sum_triangle_l247_247190

theorem angle_sum_triangle (A B C : Type) (angle_A angle_B angle_C : ℝ) 
(h1 : angle_A = 45) (h2 : angle_B = 25) 
(h3 : angle_A + angle_B + angle_C = 180) : 
angle_C = 110 := 
sorry

end angle_sum_triangle_l247_247190


namespace sum_of_abcd_is_1_l247_247158

theorem sum_of_abcd_is_1
  (a b c d : ℤ)
  (h1 : (x^2 + a*x + b)*(x^2 + c*x + d) = x^4 + 2*x^3 + x^2 + 8*x - 12) :
  a + b + c + d = 1 := by
  sorry

end sum_of_abcd_is_1_l247_247158


namespace part1_part2_l247_247751

def f (x : ℝ) : ℝ := |2 * x + 1| - |x - 1|

theorem part1 : {x : ℝ | f x < 2} = {x : ℝ | -4 < x ∧ x < 2 / 3} :=
by
  sorry

theorem part2 : ∀ a : ℝ, (∃ x : ℝ, f x ≤ a - a^2 / 2) → (-1 ≤ a ∧ a ≤ 3) :=
by
  sorry

end part1_part2_l247_247751


namespace ellipse_equation_and_line_intersection_unique_l247_247290

-- Definitions from conditions
def ellipse (x y : ℝ) : Prop := (x^2)/4 + (y^2)/3 = 1
def line (x0 y0 x y : ℝ) : Prop := 3*x0*x + 4*y0*y - 12 = 0
def on_ellipse (x0 y0 : ℝ) : Prop := ellipse x0 y0

theorem ellipse_equation_and_line_intersection_unique :
  ∀ (x0 y0 : ℝ), on_ellipse x0 y0 → ∀ (x y : ℝ), line x0 y0 x y → ellipse x y → x = x0 ∧ y = y0 :=
by
  sorry

end ellipse_equation_and_line_intersection_unique_l247_247290


namespace sid_money_left_after_purchases_l247_247068

theorem sid_money_left_after_purchases : 
  ∀ (original_money money_spent_on_computer money_spent_on_snacks half_of_original_money money_left final_more_than_half),
  original_money = 48 → 
  money_spent_on_computer = 12 → 
  money_spent_on_snacks = 8 →
  half_of_original_money = original_money / 2 → 
  money_left = original_money - (money_spent_on_computer + money_spent_on_snacks) → 
  final_more_than_half = money_left - half_of_original_money →
  final_more_than_half = 4 := 
by
  intros original_money money_spent_on_computer money_spent_on_snacks half_of_original_money money_left final_more_than_half
  intros h1 h2 h3 h4 h5 h6
  sorry

end sid_money_left_after_purchases_l247_247068


namespace prove_fractions_sum_equal_11_l247_247516

variable (a b c : ℝ)

-- Given conditions
axiom h1 : (a * c) / (a + b) + (b * a) / (b + c) + (c * b) / (c + a) = -9
axiom h2 : (b * c) / (a + b) + (c * a) / (b + c) + (a * b) / (c + a) = 10

-- The proof problem statement
theorem prove_fractions_sum_equal_11 : (b / (a + b) + c / (b + c) + a / (c + a)) = 11 :=
by
  sorry

end prove_fractions_sum_equal_11_l247_247516


namespace blue_crayons_l247_247254

variables (B G : ℕ)

theorem blue_crayons (h1 : 24 = 8 + B + G + 6) (h2 : G = (2 / 3) * B) : B = 6 :=
by 
-- This is where the proof would go
sorry

end blue_crayons_l247_247254


namespace factorize_cubic_expression_l247_247870

variable (a : ℝ)

theorem factorize_cubic_expression : a^3 - a = a * (a + 1) * (a - 1) := 
sorry

end factorize_cubic_expression_l247_247870


namespace range_of_a_l247_247227

theorem range_of_a (a : ℝ) :
  (¬ ∀ x : ℝ, x^2 + 2*x + a > 0) ↔ a ≤ 1 :=
sorry

end range_of_a_l247_247227


namespace exactly_one_defective_l247_247238

theorem exactly_one_defective (p_A p_B : ℝ) (hA : p_A = 0.04) (hB : p_B = 0.05) :
  ((p_A * (1 - p_B)) + ((1 - p_A) * p_B)) = 0.086 :=
by
  sorry

end exactly_one_defective_l247_247238


namespace incorrect_inequality_l247_247293

-- Given definitions
variables {a b : ℝ}
axiom h : a < b ∧ b < 0

-- Equivalent theorem statement
theorem incorrect_inequality (ha : a < b) (hb : b < 0) : (1 / (a - b)) < (1 / a) := 
sorry

end incorrect_inequality_l247_247293


namespace water_depth_upright_l247_247259

def tank_is_right_cylindrical := true
def tank_height := 18.0
def tank_diameter := 6.0
def tank_initial_position_is_flat := true
def water_depth_flat := 4.0

theorem water_depth_upright : water_depth_flat = 4.0 :=
by
  sorry

end water_depth_upright_l247_247259


namespace muirheadable_decreasing_columns_iff_l247_247642

def isMuirheadable (n : ℕ) (grid : List (List ℕ)) : Prop :=
  -- Placeholder definition; the actual definition should specify the conditions
  sorry

theorem muirheadable_decreasing_columns_iff (n : ℕ) (h : n > 0) :
  (∃ grid : List (List ℕ), isMuirheadable n grid) ↔ n ≠ 3 :=
by 
  sorry

end muirheadable_decreasing_columns_iff_l247_247642


namespace range_x_range_a_l247_247006

variable {x a : ℝ}
def p (a x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def q (x : ℝ) : Prop := (x - 3) * (2 - x) ≥ 0

-- (1) If a = 1, find the range of x for which p ∧ q is true.
theorem range_x (h : a = 1) : 2 ≤ x ∧ x < 3 ↔ p 1 x ∧ q x := sorry

-- (2) If ¬p is a necessary but not sufficient condition for ¬q, find the range of real number a.
theorem range_a : (¬p a x → ¬q x) → (∃ a : ℝ, 1 < a ∧ a < 2) := sorry

end range_x_range_a_l247_247006


namespace min_le_max_condition_l247_247641

variable (a b c : ℝ)

theorem min_le_max_condition
  (h1 : a ≠ 0)
  (h2 : ∃ t : ℝ, 2*a*t^2 + b*t + c = 0 ∧ |t| ≤ 1) :
  min c (a + c + 1) ≤ max (|b - a + 1|) (|b + a - 1|) :=
sorry

end min_le_max_condition_l247_247641


namespace john_mean_score_l247_247640

-- Define John's quiz scores as a list
def johnQuizScores := [95, 88, 90, 92, 94, 89]

-- Define the function to calculate the mean of a list of integers
def mean_scores (scores : List ℕ) : ℚ :=
  (scores.sum : ℚ) / scores.length

-- Prove that the mean of John's quiz scores is 91.3333 
theorem john_mean_score :
  mean_scores johnQuizScores = 91.3333 := by
  -- sorry is a placeholder for the missing proof
  sorry

end john_mean_score_l247_247640


namespace total_yellow_marbles_l247_247207

theorem total_yellow_marbles (mary_marbles : ℕ) (joan_marbles : ℕ) (h1 : mary_marbles = 9) (h2 : joan_marbles = 3) : mary_marbles + joan_marbles = 12 := 
by 
  sorry

end total_yellow_marbles_l247_247207


namespace awards_distribution_correct_answer_awards_distribution_l247_247213

theorem awards_distribution :
  let awards := 6
  let students := 4
  ∃ f : Fin awards → Fin students, 
    (∀ s : Fin students, ∃ a : Fin awards, f a = s) ∧ 
    (∑ s, 1) = 6 :=
begin
  -- Here we define the number of ways to distribute the awards
  -- We assert the number of ways equals 1560
  sorry
end

theorem correct_answer_awards_distribution :
  let awards := 6
  let students := 4
  count_distributions(awards, students) = 1560 :=
begin
  sorry
end

end awards_distribution_correct_answer_awards_distribution_l247_247213


namespace greatest_multiple_of_5_and_6_less_than_1000_l247_247394

open Nat

theorem greatest_multiple_of_5_and_6_less_than_1000 :
  ∃ a, (a % 5 = 0) ∧ (a % 6 = 0) ∧ (a < 1000) ∧ (a = 990) :=
by
  use 990
  simp [Nat.mod_eq_zero_of_dvd, Nat.le_zero_iff]
  repeat {split}; try {exact Int.ofNat_zero}
  { sorry }

end greatest_multiple_of_5_and_6_less_than_1000_l247_247394


namespace product_of_remainders_one_is_one_l247_247795

theorem product_of_remainders_one_is_one (a b : ℕ) (h1 : a % 3 = 1) (h2 : b % 3 = 1) : (a * b) % 3 = 1 :=
sorry

end product_of_remainders_one_is_one_l247_247795


namespace complement_of_A_in_U_l247_247026

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | 1 < x ∧ x ≤ 3}
def complement_U_A : Set ℝ := {x | x ≤ 1 ∨ x > 3}

theorem complement_of_A_in_U : (U \ A) = complement_U_A := by
  simp only [U, A, complement_U_A]
  sorry

end complement_of_A_in_U_l247_247026


namespace volume_of_TABC_l247_247791

noncomputable def volume_pyramid_TABC : ℝ :=
  let TA : ℝ := 15
  let TB : ℝ := 15
  let TC : ℝ := 5 * Real.sqrt 3
  let area_ABT : ℝ := (1 / 2) * TA * TB
  (1 / 3) * area_ABT * TC

theorem volume_of_TABC :
  volume_pyramid_TABC = 187.5 * Real.sqrt 3 :=
sorry

end volume_of_TABC_l247_247791


namespace count_integers_in_range_l247_247523

theorem count_integers_in_range : 
  let lower_bound := -2.8
  let upper_bound := Real.pi
  let in_range (x : ℤ) := (lower_bound : ℝ) < (x : ℝ) ∧ (x : ℝ) ≤ upper_bound
  (Finset.filter in_range (Finset.Icc (Int.floor lower_bound) (Int.floor upper_bound))).card = 6 :=
by
  sorry

end count_integers_in_range_l247_247523


namespace total_week_cost_proof_l247_247265

-- Defining variables for costs and consumption
def cost_brand_a_biscuit : ℝ := 0.25
def cost_brand_b_biscuit : ℝ := 0.35
def cost_small_rawhide : ℝ := 1
def cost_large_rawhide : ℝ := 1.50

def odd_days_biscuits_brand_a : ℕ := 3
def odd_days_biscuits_brand_b : ℕ := 2
def odd_days_small_rawhide : ℕ := 1
def odd_days_large_rawhide : ℕ := 1

def even_days_biscuits_brand_a : ℕ := 4
def even_days_small_rawhide : ℕ := 2

def odd_day_cost : ℝ :=
  odd_days_biscuits_brand_a * cost_brand_a_biscuit +
  odd_days_biscuits_brand_b * cost_brand_b_biscuit +
  odd_days_small_rawhide * cost_small_rawhide +
  odd_days_large_rawhide * cost_large_rawhide

def even_day_cost : ℝ :=
  even_days_biscuits_brand_a * cost_brand_a_biscuit +
  even_days_small_rawhide * cost_small_rawhide

def total_cost_per_week : ℝ :=
  4 * odd_day_cost + 3 * even_day_cost

theorem total_week_cost_proof :
  total_cost_per_week = 24.80 :=
  by
    unfold total_cost_per_week
    unfold odd_day_cost
    unfold even_day_cost
    norm_num
    sorry

end total_week_cost_proof_l247_247265


namespace units_digit_of_m3_plus_2m_l247_247050

def m : ℕ := 2021^2 + 2^2021

theorem units_digit_of_m3_plus_2m : (m^3 + 2^m) % 10 = 5 := by
  sorry

end units_digit_of_m3_plus_2m_l247_247050


namespace cucumbers_for_20_apples_l247_247315

-- Definitions for all conditions
def apples := ℕ
def bananas := ℕ
def cucumbers := ℕ

def cost_equivalence_apples_bananas (a b : ℕ) : Prop := 10 * a = 5 * b
def cost_equivalence_bananas_cucumbers (b c : ℕ) : Prop := 3 * b = 4 * c

-- Main theorem statement
theorem cucumbers_for_20_apples :
  ∀ (a b c : ℕ),
    cost_equivalence_apples_bananas a b →
    cost_equivalence_bananas_cucumbers b c →
    ∃ k : ℕ, k = 13 :=
by
  intros
  sorry

end cucumbers_for_20_apples_l247_247315


namespace solve_system_of_inequalities_l247_247215

theorem solve_system_of_inequalities (x : ℝ) 
  (h1 : -3 * x^2 + 7 * x + 6 > 0) 
  (h2 : 4 * x - 4 * x^2 > -3) : 
  -1/2 < x ∧ x < 3/2 :=
sorry

end solve_system_of_inequalities_l247_247215


namespace numValidPairs_l247_247299

open Finset

-- Defining the universal set
def universalSet : Finset ℕ := {1, 2, 3}

-- Condition: Elements from the universal set
variable {A B : Finset ℕ}
variable (AneqB : A ≠ B)
variable (unionCond : A ∪ B = universalSet)

-- Defining the set of all possible pairs
def possiblePairs (U : Finset ℕ) : Finset (Finset ℕ × Finset ℕ) :=
  U.powerset.product U.powerset

-- Filtering pairs satisfying conditions
def validPairs (U : Finset ℕ) : Finset (Finset ℕ × Finset ℕ) :=
  (possiblePairs U).filter (λ p, p.fst ∪ p.snd = U ∧ p.fst ≠ p.snd)

-- The theorem statement
theorem numValidPairs : (validPairs universalSet).card = 26 := 
  sorry

end numValidPairs_l247_247299


namespace range_of_m_l247_247225

/-- The point (m^2, m) is within the planar region defined by x - 3y + 2 > 0. 
    Find the range of m. -/
theorem range_of_m {m : ℝ} : (m^2 - 3 * m + 2 > 0) ↔ (m < 1 ∨ m > 2) := 
by 
  sorry

end range_of_m_l247_247225


namespace no_triangles_if_all_horizontal_removed_l247_247743

/-- 
Given a figure that consists of 40 identical toothpicks, making up a symmetric figure with 
additional rows on the top and bottom. We need to prove that removing all 40 horizontal toothpicks 
ensures there are no remaining triangles in the figure.
-/
theorem no_triangles_if_all_horizontal_removed
  (initial_toothpicks : ℕ)
  (horizontal_toothpicks_in_figure : ℕ) 
  (rows : ℕ)
  (top_row : ℕ)
  (second_row : ℕ)
  (third_row : ℕ)
  (fourth_row : ℕ)
  (bottom_row : ℕ)
  (additional_rows : ℕ)
  (triangles_for_upward : ℕ)
  (triangles_for_downward : ℕ):
  initial_toothpicks = 40 →
  horizontal_toothpicks_in_figure = top_row + second_row + third_row + fourth_row + bottom_row →
  rows = 5 →
  top_row = 5 →
  second_row = 10 →
  third_row = 10 →
  fourth_row = 10 →
  bottom_row = 5 →
  additional_rows = 2 →
  triangles_for_upward = 15 →
  triangles_for_downward = 10 →
  horizontal_toothpicks_in_figure = 40 → 
  ∀ toothpicks_removed, toothpicks_removed = 40 →
  no_triangles_remain :=
by
  intros
  sorry

end no_triangles_if_all_horizontal_removed_l247_247743


namespace count_board_configurations_l247_247325

-- Define the 3x3 board as a type with 9 positions
inductive Position 
| top_left | top_center | top_right
| middle_left | center | middle_right
| bottom_left | bottom_center | bottom_right

-- Define an enum for players' moves
inductive Mark
| X | O | Empty

-- Define a board as a mapping from positions to marks
def Board : Type := Position → Mark

-- Define the win condition for Carl
def win_condition (b : Board) : Prop := 
(b Position.center = Mark.O) ∧ 
((b Position.top_left = Mark.O ∧ b Position.top_center = Mark.O) ∨ 
(b Position.middle_left = Mark.O ∧ b Position.middle_right = Mark.O) ∨ 
(b Position.bottom_left = Mark.O ∧ b Position.bottom_center = Mark.O))

-- Define the condition for a filled board
def filled_board (b : Board) : Prop :=
∀ p : Position, b p ≠ Mark.Empty

-- The proof problem to show the total number of configurations is 30
theorem count_board_configurations : 
  ∃ (n : ℕ), n = 30 ∧
  (∃ b : Board, win_condition b ∧ filled_board b) := 
sorry

end count_board_configurations_l247_247325


namespace factorize_cubic_expression_l247_247873

variable (a : ℝ)

theorem factorize_cubic_expression : a^3 - a = a * (a + 1) * (a - 1) := 
sorry

end factorize_cubic_expression_l247_247873


namespace greatest_multiple_of_5_and_6_less_than_1000_l247_247368

theorem greatest_multiple_of_5_and_6_less_than_1000 : 
  ∃ n, n < 1000 ∧ n % 5 = 0 ∧ n % 6 = 0 ∧ ∀ m, m < 1000 ∧ m % 5 = 0 ∧ m % 6 = 0 → m ≤ n :=
exists.intro 990 (by {
  -- proof goes here
  sorry
})

end greatest_multiple_of_5_and_6_less_than_1000_l247_247368


namespace abs_diff_roots_eq_sqrt_13_l247_247296

theorem abs_diff_roots_eq_sqrt_13 {x₁ x₂ : ℝ} (h : x₁^2 + x₁ - 3 = 0 ∧ x₂^2 + x₂ - 3 = 0) :
  |x₁ - x₂| = Real.sqrt 13 :=
sorry

end abs_diff_roots_eq_sqrt_13_l247_247296


namespace percentage_increase_l247_247766

variable (E : ℝ) (P : ℝ)
variable (h1 : 1.36 * E = 495)
variable (h2 : (1 + P) * E = 454.96)

theorem percentage_increase :
  P = 0.25 :=
by
  sorry

end percentage_increase_l247_247766


namespace spadesuit_eval_l247_247741

def spadesuit (x y : ℝ) : ℝ :=
  (x + y) * (x - y)

theorem spadesuit_eval :
  spadesuit 5 (spadesuit 6 3) = -704 := by
  sorry

end spadesuit_eval_l247_247741


namespace HCl_moles_formed_l247_247597

-- Define the conditions for the problem:
def moles_H2SO4 := 1 -- moles of H2SO4
def moles_NaCl := 1 -- moles of NaCl
def reaction : List (Int × String) :=
  [(1, "H2SO4"), (2, "NaCl"), (2, "HCl"), (1, "Na2SO4")]  -- the reaction coefficients in (coefficient, chemical) pairs

-- Define the function that calculates the product moles based on limiting reactant
def calculate_HCl (moles_H2SO4 : Int) (moles_NaCl : Int) : Int :=
  if moles_NaCl < 2 then moles_NaCl else 2 * (moles_H2SO4 / 1)

-- Specify the theorem to be proven with the given conditions
theorem HCl_moles_formed :
  calculate_HCl moles_H2SO4 moles_NaCl = 1 :=
by
  sorry -- Proof can be filled in later

end HCl_moles_formed_l247_247597


namespace time_to_fill_bucket_completely_l247_247187

-- Define the conditions given in the problem
def time_to_fill_two_thirds (time_filled: ℕ) : ℕ := 90

-- Define what we need to prove
theorem time_to_fill_bucket_completely (time_filled: ℕ) : 
  time_to_fill_two_thirds time_filled = 90 → time_filled = 135 :=
by
  sorry

end time_to_fill_bucket_completely_l247_247187


namespace part_1_part_2_l247_247453

noncomputable def f (x a : ℝ) : ℝ := x^2 * |x - a|

theorem part_1 (a : ℝ) (h : a = 2) : {x : ℝ | f x a = x} = {0, 1, 1 + Real.sqrt 2} :=
by 
  sorry

theorem part_2 (a : ℝ) : 
  ∃ m : ℝ, m = 
    if a ≤ 1 then 1 - a 
    else if 1 < a ∧ a ≤ 2 then 0 
    else if 2 < a ∧ a ≤ (7 / 3 : ℝ) then 4 * (a - 2) 
    else a - 1 :=
by 
  sorry

end part_1_part_2_l247_247453


namespace cos_double_angle_l247_247624

theorem cos_double_angle (θ : ℝ) (h : Real.cos θ = 1/2) : Real.cos (2 * θ) = -1/2 := by
  sorry

end cos_double_angle_l247_247624


namespace find_missing_id_l247_247955

theorem find_missing_id
  (total_students : ℕ)
  (sample_size : ℕ)
  (known_ids : Finset ℕ)
  (k : ℕ)
  (missing_id : ℕ) : 
  total_students = 52 ∧ 
  sample_size = 4 ∧ 
  known_ids = {3, 29, 42} ∧ 
  k = total_students / sample_size ∧ 
  missing_id = 16 :=
by
  sorry

end find_missing_id_l247_247955


namespace find_ax5_by5_l247_247767

variable (a b x y : ℝ)

theorem find_ax5_by5 
  (h1 : a * x + b * y = 3)
  (h2 : a * x^2 + b * y^2 = 7)
  (h3 : a * x^3 + b * y^3 = 16)
  (h4 : a * x^4 + b * y^4 = 42) :
  a * x^5 + b * y^5 = 20 := 
sorry

end find_ax5_by5_l247_247767


namespace distinct_prime_factors_2310_l247_247857

theorem distinct_prime_factors_2310 : 
  ∃ (p1 p2 p3 p4 p5 : ℕ), p1 = 2 ∧ p2 = 3 ∧ p3 = 5 ∧ p4 = 7 ∧ p5 = 11 ∧ 
    Nat.Prime p1 ∧ Nat.Prime p2 ∧ Nat.Prime p3 ∧ Nat.Prime p4 ∧ Nat.Prime p5 ∧ 
    (p1 * p2 * p3 * p4 * p5 = 2310) :=
by
  sorry

end distinct_prime_factors_2310_l247_247857


namespace greatest_multiple_of_5_and_6_under_1000_l247_247374

theorem greatest_multiple_of_5_and_6_under_1000 : 
  ∃ x, (x % 5 = 0 ∧ x % 6 = 0 ∧ x < 1000) ∧ 
  (∀ y, (y % 5 = 0 ∧ y % 6 = 0 ∧ y < 1000) → y ≤ x) ∧ 
  x = 990 :=
begin
  sorry
end

end greatest_multiple_of_5_and_6_under_1000_l247_247374


namespace greatest_multiple_of_5_and_6_less_than_1000_l247_247388

theorem greatest_multiple_of_5_and_6_less_than_1000 : ∃ n, (n % 5 = 0) ∧ (n % 6 = 0) ∧ (n < 1000) ∧ (∀ m, (m % 5 = 0) ∧ (m % 6 = 0) ∧ (m < 1000) → m ≤ n) ∧ n = 990 :=
by sorry

end greatest_multiple_of_5_and_6_less_than_1000_l247_247388


namespace first_term_geometric_progression_l247_247086

theorem first_term_geometric_progression (S a : ℝ) (r : ℝ) 
  (h1 : S = 10) 
  (h2 : a = 10 * (1 - r)) 
  (h3 : a * (1 + r) = 7) : 
  a = 10 * (1 - Real.sqrt (3 / 10)) ∨ a = 10 * (1 + Real.sqrt (3 / 10)) := 
by 
  sorry

end first_term_geometric_progression_l247_247086


namespace solve_inequality_l247_247799

def numerator (x : ℝ) : ℝ := x ^ 2 - 4 * x + 3
def denominator (x : ℝ) : ℝ := (x - 2) ^ 2

theorem solve_inequality : { x : ℝ | numerator x / denominator x < 0 } = { x : ℝ | 1 < x ∧ x < 3 } :=
by
  sorry

end solve_inequality_l247_247799


namespace greatest_multiple_of_5_and_6_less_than_1000_l247_247369

theorem greatest_multiple_of_5_and_6_less_than_1000 : 
  ∃ n, n < 1000 ∧ n % 5 = 0 ∧ n % 6 = 0 ∧ ∀ m, m < 1000 ∧ m % 5 = 0 ∧ m % 6 = 0 → m ≤ n :=
exists.intro 990 (by {
  -- proof goes here
  sorry
})

end greatest_multiple_of_5_and_6_less_than_1000_l247_247369


namespace x_squared_minus_y_squared_l247_247970

theorem x_squared_minus_y_squared (x y : ℝ) (h1 : x + y = 9) (h2 : x - y = 3) : x^2 - y^2 = 27 := 
sorry

end x_squared_minus_y_squared_l247_247970


namespace num_elements_intersection_l247_247610

def A : Finset ℕ := {1, 2, 3, 4}
def B : Finset ℕ := {2, 4, 6, 8}

theorem num_elements_intersection : (A ∩ B).card = 2 := by
  sorry

end num_elements_intersection_l247_247610


namespace co_captains_probability_l247_247528

open Finset

theorem co_captains_probability (T1_members T2_members T3_members T1_cocaptains T2_cocaptains T3_cocaptains : Finset ℕ) 
  (hT1 : T1_members.card = 6) (hT2 : T2_members.card = 9) (hT3 : T3_members.card = 10) 
  (hT1_co : T1_cocaptains.card = 3) (hT2_co : T2_cocaptains.card = 2) (hT3_co : T3_cocaptains.card = 4) 
  (hT1_sub : T1_cocaptains ⊆ T1_members) (hT2_sub : T2_cocaptains ⊆ T2_members) (hT3_sub : T3_cocaptains ⊆ T3_members) :
  ((1 / 3 : ℚ) * (((3.choose 2 : ℚ) / (6.choose 2)) + ((2.choose 2 : ℚ) / (9.choose 2)) + ((4.choose 2 : ℚ) / (10.choose 2)))) = 65 / 540 := 
by sorry

end co_captains_probability_l247_247528


namespace add_two_inequality_l247_247304

theorem add_two_inequality (a b : ℝ) (h : a > b) : a + 2 > b + 2 :=
sorry

end add_two_inequality_l247_247304


namespace rook_placement_5x5_l247_247311

theorem rook_placement_5x5 :
  ∀ (board : Fin 5 → Fin 5) (distinct : Function.Injective board),
  ∃ (ways : Nat), ways = 120 := by
  sorry

end rook_placement_5x5_l247_247311


namespace quadratic_root_sum_l247_247172

theorem quadratic_root_sum (k : ℝ) (h : k ≤ 1 / 2) : 
  ∃ (α β : ℝ), (α + β = 2 - 2 * k) ∧ (α^2 - 2 * (1 - k) * α + k^2 = 0) ∧ (β^2 - 2 * (1 - k) * β + k^2 = 0) ∧ (α + β ≥ 1) :=
sorry

end quadratic_root_sum_l247_247172


namespace marked_price_percentage_fixed_l247_247419

-- Definitions based on the conditions
def discount_percentage : ℝ := 0.18461538461538467
def profit_percentage : ℝ := 0.06

-- The final theorem statement
theorem marked_price_percentage_fixed (CP MP SP : ℝ) 
  (h1 : SP = CP * (1 + profit_percentage))  
  (h2 : SP = MP * (1 - discount_percentage)) :
  (MP / CP - 1) * 100 = 30 := 
sorry

end marked_price_percentage_fixed_l247_247419


namespace anthony_total_pencils_l247_247841

theorem anthony_total_pencils (initial_pencils : ℕ) (pencils_given_by_kathryn : ℕ) (total_pencils : ℕ) :
  initial_pencils = 9 →
  pencils_given_by_kathryn = 56 →
  total_pencils = initial_pencils + pencils_given_by_kathryn →
  total_pencils = 65 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end anthony_total_pencils_l247_247841


namespace tobias_charges_for_mowing_l247_247543

/-- Tobias is buying a new pair of shoes that costs $95.
He has been saving up his money each month for the past three months.
He gets a $5 allowance a month.
He mowed 4 lawns and shoveled 5 driveways.
He charges $7 to shovel a driveway.
After buying the shoes, he has $15 in change.
Prove that Tobias charges $15 to mow a lawn.
--/
theorem tobias_charges_for_mowing 
  (shoes_cost : ℕ)
  (monthly_allowance : ℕ)
  (months_saving : ℕ)
  (lawns_mowed : ℕ)
  (driveways_shoveled : ℕ)
  (charge_per_shovel : ℕ)
  (money_left : ℕ)
  (total_money_before_purchase : ℕ)
  (x : ℕ)
  (h1 : shoes_cost = 95)
  (h2 : monthly_allowance = 5)
  (h3 : months_saving = 3)
  (h4 : lawns_mowed = 4)
  (h5 : driveways_shoveled = 5)
  (h6 : charge_per_shovel = 7)
  (h7 : money_left = 15)
  (h8 : total_money_before_purchase = shoes_cost + money_left)
  (h9 : total_money_before_purchase = (months_saving * monthly_allowance) + (lawns_mowed * x) + (driveways_shoveled * charge_per_shovel)) :
  x = 15 := 
sorry

end tobias_charges_for_mowing_l247_247543


namespace trip_time_l247_247758

theorem trip_time (T : ℝ) (x : ℝ) : 
  (150 / 4 = 50 / 30 + (x - 50) / 4 + (150 - x) / 30) → (T = 37.5) :=
by
  sorry

end trip_time_l247_247758


namespace stephen_total_distance_l247_247582

noncomputable def total_distance : ℝ :=
let speed1 : ℝ := 16
let time1 : ℝ := 10 / 60
let distance1 : ℝ := speed1 * time1

let speed2 : ℝ := 12 - 2 -- headwind reduction
let time2 : ℝ := 20 / 60
let distance2 : ℝ := speed2 * time2

let speed3 : ℝ := 20 + 4 -- tailwind increase
let time3 : ℝ := 15 / 60
let distance3 : ℝ := speed3 * time3

distance1 + distance2 + distance3

theorem stephen_total_distance :
  total_distance = 12 :=
by sorry

end stephen_total_distance_l247_247582


namespace detergent_per_pound_l247_247650

theorem detergent_per_pound (detergent clothes_per_det: ℝ) (h: detergent = 18 ∧ clothes_per_det = 9) :
  detergent / clothes_per_det = 2 :=
by
  sorry

end detergent_per_pound_l247_247650


namespace find_y1_l247_247010

noncomputable def y1_proof : Prop :=
∃ (y1 y2 y3 : ℝ), 0 ≤ y3 ∧ y3 ≤ y2 ∧ y2 ≤ y1 ∧ y1 ≤ 1 ∧
(1 - y1)^2 + (y1 - y2)^2 + (y2 - y3)^2 + y3^2 = 1 / 9 ∧
y1 = 1 / 2

-- Statement to be proven:
theorem find_y1 : y1_proof :=
sorry

end find_y1_l247_247010


namespace combined_cost_is_450_l247_247977

-- Given conditions
def bench_cost : ℕ := 150
def table_cost : ℕ := 2 * bench_cost

-- The statement we want to prove
theorem combined_cost_is_450 : bench_cost + table_cost = 450 :=
by
  sorry

end combined_cost_is_450_l247_247977


namespace total_profit_l247_247431

-- Definitions based on the conditions
def cost_per_dog : ℕ := 250
def num_dogs : ℕ := 2
def num_puppies : ℕ := 6
def selling_price_per_puppy : ℕ := 350

-- Statement of the theorem
theorem total_profit : (num_puppies * selling_price_per_puppy) - (num_dogs * cost_per_dog) = 1600 := 
by
  sorry

end total_profit_l247_247431


namespace remainder_expression_mod_l247_247648

/-- 
Let the positive integers s, t, u, and v leave remainders of 6, 9, 13, and 17, respectively, 
when divided by 23. Also, let s > t > u > v.
We want to prove that the remainder when 2 * (s - t) - 3 * (t - u) + 4 * (u - v) is divided by 23 is 12.
-/
theorem remainder_expression_mod (s t u v : ℕ) (hs : s % 23 = 6) (ht : t % 23 = 9) (hu : u % 23 = 13) (hv : v % 23 = 17)
  (h_gt : s > t ∧ t > u ∧ u > v) : (2 * (s - t) - 3 * (t - u) + 4 * (u - v)) % 23 = 12 :=
by
  sorry

end remainder_expression_mod_l247_247648


namespace candidate_net_gain_difference_l247_247705

theorem candidate_net_gain_difference :
  let salary1 := 42000
      revenue1 := 93000
      training_cost_per_month := 1200
      training_months := 3
      salary2 := 45000
      revenue2 := 92000
      hiring_bonus_percent := 1 / 100 in
  let total_training_cost1 := training_cost_per_month * training_months in
  let hiring_bonus2 := salary2 * hiring_bonus_percent in
  let net_gain1 := revenue1 - salary1 - total_training_cost1 in
  let net_gain2 := revenue2 - salary2 - hiring_bonus2 in
  net_gain1 - net_gain2 = 850 :=
by
  sorry

end candidate_net_gain_difference_l247_247705


namespace initial_price_of_iphone_l247_247664

variable (P : ℝ)

def initial_price_conditions : Prop :=
  (P > 0) ∧ (0.72 * P = 720)

theorem initial_price_of_iphone (h : initial_price_conditions P) : P = 1000 :=
by
  sorry

end initial_price_of_iphone_l247_247664


namespace T_description_l247_247202

def is_single_point {x y : ℝ} : Prop := (x = 2) ∧ (y = 11)

theorem T_description :
  ∀ (T : Set (ℝ × ℝ)),
  (∀ x y : ℝ, 
    (T (x, y) ↔ 
    ((5 = x + 3 ∧ 5 = y - 6) ∨ 
     (5 = x + 3 ∧ x + 3 = y - 6) ∨ 
     (5 = y - 6 ∧ x + 3 = y - 6)) ∧ 
    ((x = 2) ∧ (y = 11))
    )
  ) →
  (T = { (2, 11) }) :=
by
  sorry

end T_description_l247_247202


namespace reflect_over_y_axis_matrix_l247_247734

theorem reflect_over_y_axis_matrix :
  ∃ M : Matrix (Fin 2) (Fin 2) ℝ, M = ![![ -1, 0], ![0, 1]] :=
  -- Proof
  sorry

end reflect_over_y_axis_matrix_l247_247734


namespace weston_academy_geography_players_l247_247140

theorem weston_academy_geography_players
  (total_players : ℕ)
  (history_players : ℕ)
  (both_players : ℕ) :
  total_players = 18 →
  history_players = 10 →
  both_players = 6 →
  ∃ (geo_players : ℕ), geo_players = 14 := 
by 
  intros h1 h2 h3
  use 18 - (10 - 6) + 6
  sorry

end weston_academy_geography_players_l247_247140


namespace john_eggs_per_week_l247_247782

theorem john_eggs_per_week
  (pens : ℕ)
  (emus_per_pen : ℕ)
  (female_ratio : ℚ)
  (eggs_per_female_per_day : ℕ)
  (days_in_week : ℕ) :
  pens = 4 →
  emus_per_pen = 6 →
  female_ratio = 1/2 →
  eggs_per_female_per_day = 1 →
  days_in_week = 7 →
  (pens * emus_per_pen * female_ratio * eggs_per_female_per_day * days_in_week = 84) :=
by
  intros h_pens h_emus h_ratio h_eggs h_days
  rw [h_pens, h_emus, h_ratio, h_eggs, h_days]
  norm_num

end john_eggs_per_week_l247_247782


namespace number_of_b_values_l247_247812

theorem number_of_b_values (b : ℤ) :
  (∃ (x1 x2 x3 : ℤ), ∀ (x : ℤ), x^2 + b * x + 6 ≤ 0 ↔ x = x1 ∨ x = x2 ∨ x = x3) ↔ (b = -6 ∨ b = -5 ∨ b = 5 ∨ b = 6) :=
by
  sorry

end number_of_b_values_l247_247812


namespace sculptures_not_on_display_count_l247_247712

noncomputable def total_art_pieces : ℕ := 1800
noncomputable def pieces_on_display : ℕ := total_art_pieces / 3
noncomputable def pieces_not_on_display : ℕ := total_art_pieces - pieces_on_display
noncomputable def sculptures_on_display : ℕ := pieces_on_display / 6
noncomputable def sculptures_not_on_display : ℕ := pieces_not_on_display * 2 / 3

theorem sculptures_not_on_display_count : sculptures_not_on_display = 800 :=
by {
  -- Since this is a statement only as requested, we use sorry to skip the proof
  sorry
}

end sculptures_not_on_display_count_l247_247712


namespace expected_value_proof_l247_247951

noncomputable def expected_value_xi : ℚ :=
  let p_xi_2 : ℚ := 3/5
  let p_xi_3 : ℚ := 3/10
  let p_xi_4 : ℚ := 1/10
  2 * p_xi_2 + 3 * p_xi_3 + 4 * p_xi_4

theorem expected_value_proof :
  expected_value_xi = 5/2 :=
by
  sorry

end expected_value_proof_l247_247951


namespace tom_paid_450_l247_247671

-- Define the conditions
def hours_per_day : ℕ := 2
def number_of_days : ℕ := 3
def cost_per_hour : ℕ := 75

-- Calculated total number of hours Tom rented the helicopter
def total_hours_rented : ℕ := hours_per_day * number_of_days

-- Calculated total cost for renting the helicopter
def total_cost_rented : ℕ := total_hours_rented * cost_per_hour

-- Theorem stating that Tom paid $450 to rent the helicopter
theorem tom_paid_450 : total_cost_rented = 450 := by
  sorry

end tom_paid_450_l247_247671


namespace factorize_a_cubed_minus_a_l247_247867

theorem factorize_a_cubed_minus_a (a : ℝ) : a^3 - a = a * (a + 1) * (a - 1) :=
sorry

end factorize_a_cubed_minus_a_l247_247867


namespace total_votes_election_l247_247038

theorem total_votes_election
  (pct_candidate1 pct_candidate2 pct_candidate3 pct_candidate4 : ℝ)
  (votes_candidate4 total_votes : ℝ)
  (h1 : pct_candidate1 = 0.42)
  (h2 : pct_candidate2 = 0.30)
  (h3 : pct_candidate3 = 0.20)
  (h4 : pct_candidate4 = 0.08)
  (h5 : votes_candidate4 = 720)
  (h6 : votes_candidate4 = pct_candidate4 * total_votes) :
  total_votes = 9000 :=
sorry

end total_votes_election_l247_247038


namespace ray_inequality_l247_247279

theorem ray_inequality (a : ℝ) :
  (∀ x : ℝ, x^3 - (a^2 + a + 1) * x^2 + (a^3 + a^2 + a) * x - a^3 ≥ 0 ↔ x ≥ 1)
  ∨ (∀ x : ℝ, x^3 - (a^2 + a + 1) * x^2 + (a^3 + a^2 + a) * x - a^3 ≥ 0 ↔ x ≥ -1) :=
sorry

end ray_inequality_l247_247279


namespace quadratic_inequality_range_l247_247947

theorem quadratic_inequality_range (a : ℝ) : (∀ x : ℝ, a * x^2 + a * x + a - 1 < 0) → a ≤ 0 :=
sorry

end quadratic_inequality_range_l247_247947


namespace tan_sum_identity_sin_2alpha_l247_247602

theorem tan_sum_identity_sin_2alpha (α : ℝ) (h : Real.tan (π/4 + α) = 2) : Real.sin (2*α) = 3/5 :=
by
  sorry

end tan_sum_identity_sin_2alpha_l247_247602


namespace autograph_value_after_changes_l247_247057

def initial_value : ℝ := 100
def drop_percent : ℝ := 0.30
def increase_percent : ℝ := 0.40

theorem autograph_value_after_changes :
  let value_after_drop := initial_value * (1 - drop_percent)
  let value_after_increase := value_after_drop * (1 + increase_percent)
  value_after_increase = 98 :=
by
  sorry

end autograph_value_after_changes_l247_247057


namespace find_m_l247_247904

theorem find_m
  (θ : Real)
  (m : Real)
  (h_sin_cos_roots : ∀ x : Real, 4 * x^2 + 2 * m * x + m = 0 → x = Real.sin θ ∨ x = Real.cos θ)
  (h_real_roots : ∃ x : Real, 4 * x^2 + 2 * m * x + m = 0) :
  m = 1 - Real.sqrt 5 :=
sorry

end find_m_l247_247904


namespace max_value_of_function_l247_247961

noncomputable def y (x : ℝ) : ℝ := 
  Real.sin x - Real.cos x - Real.sin x * Real.cos x

theorem max_value_of_function :
  ∃ x : ℝ, y x = (1 / 2) + Real.sqrt 2 :=
sorry

end max_value_of_function_l247_247961


namespace range_of_m_l247_247745

-- Given definitions and conditions
def sequence_a (n : ℕ) : ℕ := if n = 1 then 2 else n * 2^n

def vec_a : ℕ × ℤ := (2, -1)

def vec_b (n : ℕ) : ℕ × ℤ := (sequence_a n + 2^n, sequence_a (n + 1))

def orthogonal (v1 v2 : ℕ × ℤ) : Prop := v1.1 * v2.1 + v1.2 * v2.2 = 0

-- Translate the proof problem
theorem range_of_m (n : ℕ) (m : ℝ) (h1 : orthogonal vec_a (vec_b n))
  (h2 : ∀ n : ℕ, n > 0 → (sequence_a n) / (n * (n + 1)^2) > (m^2 - 3 * m) / 9) :
  -1 < m ∧ m < 4 :=
by
  sorry

end range_of_m_l247_247745


namespace max_value_of_expression_l247_247496

noncomputable def f (x y : ℝ) := x * y^2 * (x^2 + x + 1) * (y^2 + y + 1)

theorem max_value_of_expression (x y : ℝ) (h : x + y = 5) :
  ∃ m, m = 951625 / 256 ∧ ∀ a b : ℝ, a + b = 5 → f a b ≤ m :=
sorry

end max_value_of_expression_l247_247496


namespace fraction_of_salary_spent_on_house_rent_l247_247256

theorem fraction_of_salary_spent_on_house_rent
    (S : ℕ) (H : ℚ)
    (cond1 : S = 180000)
    (cond2 : S / 5 + H * S + 3 * S / 5 + 18000 = S) :
    H = 1 / 10 := by
  sorry

end fraction_of_salary_spent_on_house_rent_l247_247256


namespace value_of_each_walmart_gift_card_l247_247776

variable (best_buy_value : ℕ) (best_buy_count : ℕ) (walmart_count : ℕ) (points_sent_bb : ℕ) (points_sent_wm : ℕ) (total_returnable : ℕ)

-- Conditions given in the problem
def conditions : Prop :=
  best_buy_value = 500 ∧
  best_buy_count = 6 ∧
  walmart_count = 9 ∧
  points_sent_bb = 1 ∧
  points_sent_wm = 2 ∧
  total_returnable = 3900

-- Result to prove
theorem value_of_each_walmart_gift_card : conditions best_buy_value best_buy_count walmart_count points_sent_bb points_sent_wm total_returnable →
  (total_returnable - ((best_buy_count - points_sent_bb) * best_buy_value)) / (walmart_count - points_sent_wm) = 200 :=
by
  intros h
  rcases h with
    ⟨hbv, hbc, hwc, hsbb, hswm, htr⟩
  sorry

end value_of_each_walmart_gift_card_l247_247776


namespace distance_between_points_forms_right_triangle_l247_247843

def point_1 : (ℝ × ℝ) := (5, -3)
def point_2 : (ℝ × ℝ) := (-7, 4)
def point_3 : (ℝ × ℝ) := (5, 4)

def dist (p1 p2 : (ℝ × ℝ)) : ℝ := Real.sqrt (((p2.1 - p1.1) ^ 2) + ((p2.2 - p1.2) ^ 2))

def is_right_triangle (a b c : ℝ) : Prop := a^2 + b^2 = c^2

theorem distance_between_points : dist point_1 point_2 = Real.sqrt 193 := by 
  sorry

theorem forms_right_triangle : is_right_triangle (dist point_1 point_3) (dist point_2 point_3) (dist point_1 point_2) := by 
  sorry

end distance_between_points_forms_right_triangle_l247_247843


namespace factorize_a_cubed_minus_a_l247_247863

theorem factorize_a_cubed_minus_a (a : ℝ) : a^3 - a = a * (a + 1) * (a - 1) :=
by
  sorry

end factorize_a_cubed_minus_a_l247_247863


namespace find_QE_l247_247200

noncomputable def QE (QD DE : ℝ) : ℝ :=
  QD + DE

theorem find_QE :
  ∀ (Q C R D E : Type) (QR QD DE QE : ℝ), 
  QD = 5 →
  QE = QD + DE →
  QR = DE - QD →
  QR^2 = QD * QE →
  QE = (QD + 5 + 5 * Real.sqrt 5) / 2 :=
by
  intros
  sorry

end find_QE_l247_247200


namespace problem1_problem2_l247_247587

-- Problem 1: Prove that (1) - 8 + 12 - 16 - 23 = -35
theorem problem1 : (1 - 8 + 12 - 16 - 23 = -35) :=
by
  sorry

-- Problem 2: Prove that (3 / 4) + (-1 / 6) - (1 / 3) - (-1 / 8) = 3 / 8
theorem problem2 : (3 / 4 + (-1 / 6) - 1 / 3 + 1 / 8 = 3 / 8) :=
by
  sorry

end problem1_problem2_l247_247587


namespace area_bounded_by_curves_l247_247280

open Set Filter

noncomputable def f (x : ℝ) : ℝ := x^3 - 4 * x

theorem area_bounded_by_curves : 
  abs (∫ x in 0..2, f x) = 4 := by
-- Proof goes here
  sorry

end area_bounded_by_curves_l247_247280


namespace x_intercept_of_line_is_7_over_2_l247_247422

-- Definitions for the conditions
def point1 : ℝ × ℝ := (2, -3)
def point2 : ℝ × ℝ := (6, 5)

-- Define what it means to be the x-intercept of the line
def x_intercept_of_line (x : ℝ) : Prop :=
  ∃ m b : ℝ, (point1.snd) = m * (point1.fst) + b ∧ (point2.snd) = m * (point2.fst) + b ∧ 0 = m * x + b

-- The theorem stating the x-intercept
theorem x_intercept_of_line_is_7_over_2 : x_intercept_of_line (7 / 2) :=
sorry

end x_intercept_of_line_is_7_over_2_l247_247422


namespace intersection_P_Q_intersection_complementP_Q_l247_247900

-- Define the universal set U
def U := Set.univ (ℝ)

-- Define set P
def P := {x : ℝ | |x| > 2}

-- Define set Q
def Q := {x : ℝ | x^2 - 4*x + 3 < 0}

-- Complement of P with respect to U
def complement_P : Set ℝ := {x : ℝ | |x| ≤ 2}

theorem intersection_P_Q : P ∩ Q = ({x : ℝ | 2 < x ∧ x < 3}) :=
by {
  sorry
}

theorem intersection_complementP_Q : complement_P ∩ Q = ({x : ℝ | 1 < x ∧ x ≤ 2}) :=
by {
  sorry
}

end intersection_P_Q_intersection_complementP_Q_l247_247900


namespace M_inter_N_eq_singleton_l247_247756

def M (x y : ℝ) : Prop := x + y = 2
def N (x y : ℝ) : Prop := x - y = 4

theorem M_inter_N_eq_singleton :
  {p : ℝ × ℝ | M p.1 p.2} ∩ {p : ℝ × ℝ | N p.1 p.2} = { (3, -1) } :=
by
  sorry

end M_inter_N_eq_singleton_l247_247756


namespace ellipse_equation_line_HN_fixed_point_l247_247168

open Real

-- Given conditions
def center (E : Type) : Point := (0, 0)
def axes_of_symmetry (E : Type) : Prop := true -- x-axis and y-axis are assumed
def passes_through (E : Type) (p q r : Point) := p ∈ E ∧ q ∈ E ∧ r ∈ E
def equation (E : Type) (x y : ℝ) : Prop := (x^2 / 3) + (y^2 / 4) = 1

def point := (ℝ × ℝ) -- Defining a type for point

-- Definitions for other points and line conditions
def A : point := (0, -2)
def B : point := (3/2, -1)
def P : point := (1, -2)

def intersects (E : Type) (P : point) : Point := sorry
def line_passes_through (P Q : point) : Prop := sorry
def M (E : Type) (P : point) : Point := intersects E P
def N (E : Type) (P : point) : Point := intersects E P

def parallel (P : point) : Prop := sorry -- line parallel to x-axis passing through P
def T (A B M: point) : Point := sorry -- intersection of line through M parallel to x-axis and line segment AB
def H (M T: point) : point := sorry -- point satisfying MT = TH

-- Proof Problem 1: Equation of the Ellipse
theorem ellipse_equation : ∀ E, 
  center E = (0, 0) →
  axes_of_symmetry E →
  passes_through E A B →
  ∃ x y, equation E x y :=
by
  intro E
  assume h1 h2 h3
  sorry

-- Proof Problem 2: Line HN passes through fixed point
theorem line_HN_fixed_point : 
  ∀ E, 
  center E = (0, 0) →
  axes_of_symmetry E →
  passes_through E A B →
  ∃ (x : ℝ) (y : ℝ), 
  ∀ (P : point), 
  intersects E P →
  let M := M E P, 
      N := N E P, 
      T := T A B M,
      H := H M T in 
  line_passes_through (1,-2) N
  →
  (HN_line : y = 2 + 2 * sqrt 6/3 * x - 2)
  ∧ line_passes_through (0, -2) H :=
by
  intro E
  assume h1 h2 h3
  sorry

end ellipse_equation_line_HN_fixed_point_l247_247168


namespace problem_1_problem_2_l247_247658

-- Problem (1): Proving the solutions for \( x^2 - 3x = 0 \)
theorem problem_1 : ∀ x : ℝ, x^2 - 3 * x = 0 ↔ (x = 0 ∨ x = 3) :=
by
  intro x
  sorry

-- Problem (2): Proving the solutions for \( 5x + 2 = 3x^2 \)
theorem problem_2 : ∀ x : ℝ, 5 * x + 2 = 3 * x^2 ↔ (x = -1/3 ∨ x = 2) :=
by
  intro x
  sorry

end problem_1_problem_2_l247_247658


namespace sequence_is_increasing_l247_247454

theorem sequence_is_increasing (a : ℕ → ℝ) (h : ∀ n : ℕ, a (n + 1) - a n = 2) : ∀ n : ℕ, a (n + 1) > a n :=
by
  intro n
  have h2 : a (n + 1) - a n = 2 := h n
  linarith

end sequence_is_increasing_l247_247454


namespace intersecting_points_radius_squared_l247_247901

noncomputable def parabola1 (x : ℝ) : ℝ := (x - 2) ^ 2
noncomputable def parabola2 (y : ℝ) : ℝ := (y - 5) ^ 2 - 1

theorem intersecting_points_radius_squared :
  ∃ (x y : ℝ), (y = parabola1 x ∧ x = parabola2 y) → (x - 2) ^ 2 + (y - 5) ^ 2 = 16 := by
sorry

end intersecting_points_radius_squared_l247_247901


namespace find_A_l247_247663

theorem find_A : ∃ (A : ℕ), 
  (A > 0) ∧ (A ∣ (270 * 2 - 312)) ∧ (A ∣ (211 * 2 - 270)) ∧ 
  (∃ (rA rB rC : ℕ), 312 % A = rA ∧ 270 % A = rB ∧ 211 % A = rC ∧ 
                      rA = 2 * rB ∧ rB = 2 * rC ∧ A = 19) :=
by sorry

end find_A_l247_247663


namespace student_problem_perfect_matching_l247_247035

theorem student_problem_perfect_matching :
  ∃ (matching : Finset (Fin 20 × Fin 20)),
    -- Define that matching is a finite set of pairs (student, problem)
    matching.card = 20 ∧ 
    -- The number of pairs (edges in the matching)
    ∀ (s : Fin 20), ∃ (p : Fin 20), (s, p) ∈ matching ∧ 
    -- Each student presents exactly one of the problems they solved
    ∀ (p : Fin 20), ∃ (s : Fin 20), (s, p) ∈ matching ∧ 
    -- Each problem is reviewed by exactly one student who solved it
    ∀ (e ∈ matching), ∃ (s : Fin 20) (p : Fin 20), (s, p) = e ∧ 
    -- Each element in the matching is a valid pair (student, problem)
    (∃ (graph : Fin 20 → Finset (Fin 20)),
    -- Define the bipartite graph as a map from students to sets of problems
    (∀ (s : Fin 20), (graph s).card = 2) ∧ 
    -- Each student solved exactly 2 problems
    (∀ (p : Fin 20), ((Finset.card (Finset.filter (λ (x : Fin 20 × Fin 20), x.2 = p) matching)) = 2))) sorry
    -- Each problem is solved by exactly 2 students.
    -- Using Hall's Theorem or equivalent criteria to establish the existence of a perfect matching.

end student_problem_perfect_matching_l247_247035


namespace geometric_sequence_properties_l247_247195

theorem geometric_sequence_properties (a : ℕ → ℝ) (n : ℕ) (q : ℝ) 
  (h_geom : ∀ (m k : ℕ), a (m + k) = a m * q ^ k) 
  (h_sum : a 1 + a n = 66) 
  (h_prod : a 3 * a (n - 2) = 128) 
  (h_s_n : (a 1 * (1 - q ^ n)) / (1 - q) = 126) : 
  n = 6 ∧ (q = 2 ∨ q = 1/2) :=
sorry

end geometric_sequence_properties_l247_247195


namespace at_least_one_is_zero_l247_247654

theorem at_least_one_is_zero (a b : ℕ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) : false := by sorry

end at_least_one_is_zero_l247_247654


namespace bruce_total_cost_l247_247716

def cost_of_grapes : ℕ := 8 * 70
def cost_of_mangoes : ℕ := 11 * 55
def cost_of_oranges : ℕ := 5 * 45
def cost_of_apples : ℕ := 3 * 90
def cost_of_cherries : ℕ := (45 / 10) * 120  -- use rational division and then multiplication

def total_cost : ℕ :=
  cost_of_grapes + cost_of_mangoes + cost_of_oranges + cost_of_apples + cost_of_cherries

theorem bruce_total_cost : total_cost = 2200 := by
  sorry

end bruce_total_cost_l247_247716


namespace prove_values_of_a_and_b_prove_range_of_k_l247_247455

variable {f : ℝ → ℝ}

-- (1) Prove values of a and b
theorem prove_values_of_a_and_b (h_odd : ∀ x : ℝ, f (-x) = -f x) :
  (∀ x, f x = 2 * x - 1) := by
sorry

-- (2) Prove range of k
theorem prove_range_of_k (h_fx_2x_minus_1 : ∀ x : ℝ, f x = 2 * x - 1) :
  (∀ t : ℝ, f (t^2 - 2 * t) + f (2 * t^2 - k) < 0) ↔ k < -1 / 3 := by
sorry

end prove_values_of_a_and_b_prove_range_of_k_l247_247455


namespace arrange_balls_l247_247484

theorem arrange_balls (w b : ℕ) (h_w : w = 7) (h_b : b = 5) :
    ∃ n : ℕ, n = 56 ∧ ∀ arr : List (char × Nat), (arr.filter (λ x, x.1 = 'B')).length = b ∧
               (arr.filter (λ x, x.1 = 'W')).length = w ∧
               (arr.dropWhile (λ x, x.1 ≠ 'B')).drop 1.filter (λ x, x.1 = 'B').length = 0 → 
               n :=
by sorry

end arrange_balls_l247_247484


namespace mary_time_l247_247056

-- Define the main entities for the problem
variables (mary_days : ℕ) (rosy_days : ℕ)
variable (rosy_efficiency_factor : ℝ) -- Rosy's efficiency factor compared to Mary

-- Given conditions
def rosy_efficient := rosy_efficiency_factor = 1.4
def rosy_time := rosy_days = 20

-- Problem Statement
theorem mary_time (h1 : rosy_efficient rosy_efficiency_factor) (h2 : rosy_time rosy_days) : mary_days = 28 :=
by
  sorry

end mary_time_l247_247056


namespace sandwich_total_calories_l247_247715

-- Given conditions
def bacon_calories := 2 * 125
def bacon_percentage := 20 / 100

-- Statement to prove
theorem sandwich_total_calories :
  bacon_calories / bacon_percentage = 1250 := 
sorry

end sandwich_total_calories_l247_247715


namespace problem1_l247_247122

theorem problem1 (a b : ℝ) (h1 : a ≥ b) (h2 : b > 0) : 2 * a^3 - b^3 ≥ 2 * a * b^2 - a^2 * b := 
sorry

end problem1_l247_247122


namespace cucumbers_for_20_apples_l247_247313

theorem cucumbers_for_20_apples (A B C : ℝ) (h1 : 10 * A = 5 * B) (h2 : 3 * B = 4 * C) :
  20 * A = 40 / 3 * C :=
by
  sorry

end cucumbers_for_20_apples_l247_247313


namespace eggs_left_after_cupcakes_l247_247672

-- Definitions derived from the given conditions
def dozen := 12
def initial_eggs := 3 * dozen
def crepes_fraction := 1 / 4
def cupcakes_fraction := 2 / 3

theorem eggs_left_after_cupcakes :
  let eggs_after_crepes := initial_eggs - crepes_fraction * initial_eggs;
  let eggs_after_cupcakes := eggs_after_crepes - cupcakes_fraction * eggs_after_crepes;
  eggs_after_cupcakes = 9 := sorry

end eggs_left_after_cupcakes_l247_247672


namespace prime_square_sum_of_cubes_equals_three_l247_247277

open Nat

theorem prime_square_sum_of_cubes_equals_three (p : ℕ) (h_prime : p.Prime) :
  (∃ a b : ℕ, a > 0 ∧ b > 0 ∧ p^2 = a^3 + b^3) → (p = 3) :=
by
  sorry

end prime_square_sum_of_cubes_equals_three_l247_247277


namespace probability_of_hitting_target_at_least_once_l247_247619

theorem probability_of_hitting_target_at_least_once :
  (∀ (p1 p2 : ℝ), p1 = 0.5 → p2 = 0.7 → (1 - (1 - p1) * (1 - p2)) = 0.85) :=
by
  intros p1 p2 h1 h2
  rw [h1, h2]
  -- This rw step simplifies (1 - (1 - 0.5) * (1 - 0.7)) to the desired result.
  sorry

end probability_of_hitting_target_at_least_once_l247_247619


namespace area_of_shaded_region_l247_247105

theorem area_of_shaded_region (r R : ℝ) (π : ℝ) (h1 : R = 3 * r) (h2 : 2 * r = 6) : 
  (π * R^2) - (π * r^2) = 72 * π :=
by
  sorry

end area_of_shaded_region_l247_247105


namespace find_rate_of_new_machine_l247_247564

noncomputable def rate_of_new_machine (R : ℝ) : Prop :=
  let old_rate := 100
  let total_bolts := 350
  let time_in_hours := 84 / 60
  let bolts_by_old_machine := old_rate * time_in_hours
  let bolts_by_new_machine := total_bolts - bolts_by_old_machine
  R = bolts_by_new_machine / time_in_hours

theorem find_rate_of_new_machine : rate_of_new_machine 150 :=
by
  sorry

end find_rate_of_new_machine_l247_247564


namespace det_B_eq_2_l247_247493

theorem det_B_eq_2 {x y : ℝ}
  (hB : ∃ (B : Matrix (Fin 2) (Fin 2) ℝ), B = ![![x, 2], ![-3, y]])
  (h_eqn : ∃ (B_inv : Matrix (Fin 2) (Fin 2) ℝ),
    B_inv = (1 / (x * y + 6)) • ![![y, -2], ![3, x]] ∧
    ![![x, 2], ![-3, y]] + 2 • B_inv = 0) : 
  Matrix.det ![![x, 2], ![-3, y]] = 2 :=
by
  sorry

end det_B_eq_2_l247_247493


namespace prime_numbers_satisfy_equation_l247_247999

noncomputable def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem prime_numbers_satisfy_equation :
  ∃ (p q r : ℕ), is_prime p ∧ is_prime q ∧ is_prime r ∧ (p + q^2 = r^4) ∧ 
  (p = 7) ∧ (q = 3) ∧ (r = 2) :=
by
  sorry

end prime_numbers_satisfy_equation_l247_247999


namespace white_roses_per_table_decoration_l247_247059

theorem white_roses_per_table_decoration (x : ℕ) :
  let bouquets := 5
  let table_decorations := 7
  let roses_per_bouquet := 5
  let total_roses := 109
  5 * roses_per_bouquet + 7 * x = total_roses → x = 12 :=
by
  intros
  sorry

end white_roses_per_table_decoration_l247_247059


namespace bobs_total_profit_l247_247436

-- Definitions of the conditions
def cost_per_dog : ℕ := 250
def num_dogs : ℕ := 2
def num_puppies : ℕ := 6
def selling_price_per_puppy : ℕ := 350

-- Definition of the problem statement
theorem bobs_total_profit : (num_puppies * selling_price_per_puppy) - (num_dogs * cost_per_dog) = 1600 := 
by
  sorry

end bobs_total_profit_l247_247436


namespace sufficient_but_not_necessary_condition_l247_247005

theorem sufficient_but_not_necessary_condition (x y : ℝ) :
  (x = 1 ∧ y = 1 → x + y = 2) ∧ (¬(x + y = 2 → x = 1 ∧ y = 1)) :=
by
  sorry

end sufficient_but_not_necessary_condition_l247_247005


namespace g_range_excludes_zero_l247_247270

noncomputable def g (x : ℝ) : ℤ :=
if x > -1 then ⌈1 / (x + 1)⌉
else ⌊1 / (x + 1)⌋

theorem g_range_excludes_zero : ¬ ∃ x : ℝ, g x = 0 := 
by 
  sorry

end g_range_excludes_zero_l247_247270


namespace greatest_multiple_l247_247373

theorem greatest_multiple (n : ℕ) (h1 : n < 1000) (h2 : n % 5 = 0) (h3 : n % 6 = 0) : n = 990 :=
sorry

end greatest_multiple_l247_247373


namespace age_of_B_l247_247119

variables (A B C : ℕ)

theorem age_of_B (h1 : (A + B + C) / 3 = 25) (h2 : (A + C) / 2 = 29) : B = 17 := 
by
  -- Skipping the proof steps
  sorry

end age_of_B_l247_247119


namespace cleanup_drive_weight_per_mile_per_hour_l247_247498

theorem cleanup_drive_weight_per_mile_per_hour :
  let duration := 4
  let lizzie_group := 387
  let second_group := lizzie_group - 39
  let third_group := 560 / 16
  let total_distance := 8
  let total_garbage := lizzie_group + second_group + third_group
  total_garbage / total_distance / duration = 24.0625 := 
by {
  sorry
}

end cleanup_drive_weight_per_mile_per_hour_l247_247498


namespace number_of_integers_in_original_list_l247_247402

theorem number_of_integers_in_original_list :
  ∃ n m : ℕ, (m + 2) * (n + 1) = m * n + 15 ∧
             (m + 1) * (n + 2) = m * n + 16 ∧
             n = 4 :=
by {
  sorry
}

end number_of_integers_in_original_list_l247_247402


namespace max_value_is_27_l247_247185

noncomputable def max_value_of_expression (a b c : ℝ) : ℝ :=
  (a - b)^2 + (b - c)^2 + (c - a)^2

theorem max_value_is_27 (a b c : ℝ)
  (h : a^2 + b^2 + c^2 = 9) : max_value_of_expression a b c = 27 :=
by
  sorry

end max_value_is_27_l247_247185


namespace ratio_spaghetti_to_fettuccine_l247_247037

def spg : Nat := 300
def fet : Nat := 80

theorem ratio_spaghetti_to_fettuccine : spg / gcd spg fet = 300 / 20 ∧ fet / gcd spg fet = 80 / 20 ∧ (spg / gcd spg fet) / (fet / gcd spg fet) = 15 / 4 := by
  sorry

end ratio_spaghetti_to_fettuccine_l247_247037


namespace three_gorges_dam_capacity_scientific_notation_l247_247810

theorem three_gorges_dam_capacity_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), (16780000 : ℝ) = a * 10^n ∧ 1 ≤ a ∧ a < 10 ∧ a = 1.678 ∧ n = 7 :=
by
  sorry

end three_gorges_dam_capacity_scientific_notation_l247_247810


namespace four_angles_for_shapes_l247_247838

-- Definitions for the shapes
def is_rectangle (fig : Type) : Prop :=
  ∀ a b c d : fig, ∃ angles : ℕ, angles = 4

def is_square (fig : Type) : Prop :=
  ∀ a b c d : fig, ∃ angles : ℕ, angles = 4

def is_parallelogram (fig : Type) : Prop :=
  ∀ a b c d : fig, ∃ angles : ℕ, angles = 4

-- Main proposition
theorem four_angles_for_shapes {fig : Type} :
  (is_rectangle fig) ∧ (is_square fig) ∧ (is_parallelogram fig) →
  ∀ shape : fig, ∃ angles : ℕ, angles = 4 := by
  sorry

end four_angles_for_shapes_l247_247838


namespace fourth_derivative_of_function_y_l247_247973

noncomputable def log_base_3 (x : ℝ) : ℝ := (Real.log x) / (Real.log 3)

noncomputable def function_y (x : ℝ) : ℝ := (log_base_3 x) / (x ^ 2)

theorem fourth_derivative_of_function_y (x : ℝ) (h : 0 < x) : 
    (deriv^[4] (fun x => function_y x)) x = (-154 + 120 * (Real.log x)) / (x ^ 6 * Real.log 3) :=
  sorry

end fourth_derivative_of_function_y_l247_247973


namespace cucumbers_for_20_apples_l247_247316

-- Definitions for all conditions
def apples := ℕ
def bananas := ℕ
def cucumbers := ℕ

def cost_equivalence_apples_bananas (a b : ℕ) : Prop := 10 * a = 5 * b
def cost_equivalence_bananas_cucumbers (b c : ℕ) : Prop := 3 * b = 4 * c

-- Main theorem statement
theorem cucumbers_for_20_apples :
  ∀ (a b c : ℕ),
    cost_equivalence_apples_bananas a b →
    cost_equivalence_bananas_cucumbers b c →
    ∃ k : ℕ, k = 13 :=
by
  intros
  sorry

end cucumbers_for_20_apples_l247_247316


namespace factorize_cubic_expression_l247_247871

variable (a : ℝ)

theorem factorize_cubic_expression : a^3 - a = a * (a + 1) * (a - 1) := 
sorry

end factorize_cubic_expression_l247_247871


namespace Mr_Spacek_birds_l247_247935

theorem Mr_Spacek_birds :
  ∃ N : ℕ, 50 < N ∧ N < 100 ∧ N % 9 = 0 ∧ N % 4 = 0 ∧ N = 72 :=
by
  sorry

end Mr_Spacek_birds_l247_247935


namespace dot_product_ABC_l247_247191

open Real

noncomputable def a : ℝ := 5
noncomputable def b : ℝ := 6
noncomputable def angleC : ℝ := π / 6  -- 30 degrees in radians

theorem dot_product_ABC :
  let CB := a
  let CA := b
  let angle_between := π - angleC  -- 150 degrees in radians
  let cos_angle := - (sqrt 3) / 2  -- cos(150 degrees)
  ∃ (dot_product : ℝ), dot_product = CB * CA * cos_angle :=
by
  have CB := a
  have CA := b
  have angle_between := π - angleC
  have cos_angle := - (sqrt 3) / 2
  use CB * CA * cos_angle
  sorry

end dot_product_ABC_l247_247191


namespace probability_one_project_not_selected_l247_247427

noncomputable def calc_probability : ℚ :=
  let n := 4 ^ 4
  let m := Nat.choose 4 2 * Nat.factorial 4
  let p := m / n
  p

theorem probability_one_project_not_selected :
  calc_probability = 9 / 16 :=
by
  sorry

end probability_one_project_not_selected_l247_247427


namespace members_on_fathers_side_are_10_l247_247926

noncomputable def members_father_side (total : ℝ) (ratio : ℝ) (members_mother_side_more: ℝ) : Prop :=
  let F := total / (1 + ratio)
  F = 10

theorem members_on_fathers_side_are_10 :
  ∀ (total : ℝ) (ratio : ℝ), 
  total = 23 → 
  ratio = 0.30 →
  members_father_side total ratio (ratio * total) :=
by
  intros total ratio htotal hratio
  have h1 : total = 23 := htotal
  have h2 : ratio = 0.30 := hratio
  rw [h1, h2]
  sorry

end members_on_fathers_side_are_10_l247_247926


namespace find_m_plus_n_l247_247613

theorem find_m_plus_n (m n : ℤ) 
  (H1 : (x^3 + m*x + n) * (x^2 - 3*x + 1) ≠ 1 * x^2 + 1 * x^3) 
  (H2 : (x^3 + m*x + n) * (x^2 - 3*x + 1) ≠ 1 * x^2 + 1 * x^3) : 
  m + n = -4 := 
by
  sorry

end find_m_plus_n_l247_247613


namespace concentric_circles_ratio_l247_247321

theorem concentric_circles_ratio
  (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h : π * b^2 - π * a^2 = 4 * (π * a^2)) :
  a / b = 1 / Real.sqrt 5 :=
by
  sorry

end concentric_circles_ratio_l247_247321


namespace percentage_increase_l247_247914

def x (y: ℝ) : ℝ := 1.25 * y
def z : ℝ := 250
def total_amount (x y z : ℝ) : ℝ := x + y + z

theorem percentage_increase (y: ℝ) : (total_amount (x y) y z = 925) → ((y - z) / z) * 100 = 20 := by
  sorry

end percentage_increase_l247_247914


namespace ones_digit_largest_power_of_3_dividing_18_factorial_l247_247598

theorem ones_digit_largest_power_of_3_dividing_18_factorial :
  (3^8 % 10) = 1 :=
by sorry

end ones_digit_largest_power_of_3_dividing_18_factorial_l247_247598


namespace probability_adjacent_l247_247310

open Finset

-- Given condition
def total_permutations (s : Finset (Fin 3)) : ℕ :=
  card (s.permutations)

-- Given condition
def adjacent_permutations (s : Finset (Fin 2)) (s' : Finset (Fin 1)) : ℕ :=
  card (s.permutations) * card (s'.permutations)

-- Theorem statement
theorem probability_adjacent :
  let s := univ : Finset (Fin 3),
      s_ab := univ : Finset (Fin 2),
      s_c := singleton ⟨0⟩ : Finset (Fin 1) in
  (adjacent_permutations s_ab s_c : ℚ) / (total_permutations s : ℚ) = 2 / 3 :=
by
  sorry

end probability_adjacent_l247_247310


namespace min_sum_abc_l247_247808

theorem min_sum_abc (a b c : ℕ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_abc : a * b * c = 1020) : a + b + c = 33 :=
sorry

end min_sum_abc_l247_247808


namespace logs_quadratic_sum_l247_247180

theorem logs_quadratic_sum (a b : ℝ) (ha_pos : 0 < a) (hb_pos : 0 < b)
  (h_roots : ∀ x, 2 * x^2 + 4 * x + 1 = 0 → (x = Real.log a) ∨ (x = Real.log b)) :
  (Real.log a)^2 + Real.log (a^2) + a * b = 1 / Real.exp 2 - 1 / 2 :=
by
  sorry

end logs_quadratic_sum_l247_247180


namespace largest_integer_remainder_l247_247730

theorem largest_integer_remainder :
  ∃ (a : ℤ), a < 61 ∧ a % 6 = 5 ∧ ∀ b : ℤ, b < 61 ∧ b % 6 = 5 → b ≤ a :=
by
  sorry

end largest_integer_remainder_l247_247730


namespace tetrad_does_not_have_four_chromosomes_l247_247965

noncomputable def tetrad_has_two_centromeres : Prop := -- The condition: a tetrad has two centromeres
  sorry

noncomputable def tetrad_contains_four_dna_molecules : Prop := -- The condition: a tetrad contains four DNA molecules
  sorry

noncomputable def tetrad_consists_of_two_pairs_of_sister_chromatids : Prop := -- The condition: a tetrad consists of two pairs of sister chromatids
  sorry

theorem tetrad_does_not_have_four_chromosomes 
  (h1: tetrad_has_two_centromeres)
  (h2: tetrad_contains_four_dna_molecules)
  (h3: tetrad_consists_of_two_pairs_of_sister_chromatids) 
  : ¬ (tetrad_has_four_chromosomes : Prop) :=
sorry

end tetrad_does_not_have_four_chromosomes_l247_247965


namespace problem_statement_l247_247298

variable {a x y : ℝ}

theorem problem_statement (hx : 0 < a) (ha : a < 1) (h : a^x < a^y) : x^3 > y^3 :=
sorry

end problem_statement_l247_247298


namespace cube_root_expression_l247_247149

theorem cube_root_expression (x : ℝ) (hx : x ≥ 0) : (x * Real.sqrt (x * x^(1/3)))^(1/3) = x^(5/9) :=
by
  sorry

end cube_root_expression_l247_247149


namespace amy_hours_per_week_l247_247579

theorem amy_hours_per_week (hours_summer_per_week : ℕ) (weeks_summer : ℕ) (earnings_summer : ℕ)
  (weeks_school_year : ℕ) (earnings_school_year_goal : ℕ) :
  (hours_summer_per_week = 40) →
  (weeks_summer = 12) →
  (earnings_summer = 4800) →
  (weeks_school_year = 36) →
  (earnings_school_year_goal = 7200) →
  (∃ hours_school_year_per_week : ℕ, hours_school_year_per_week = 20) :=
by
  sorry

end amy_hours_per_week_l247_247579


namespace find_solutions_l247_247278

noncomputable def equation (x : ℝ) : ℝ :=
  (1 / (x^2 + 11*x - 8)) + (1 / (x^2 + 2*x - 8)) + (1 / (x^2 - 13*x - 8))

theorem find_solutions : 
  {x : ℝ | equation x = 0} = {1, -8, 8, -1} := by
  sorry

end find_solutions_l247_247278


namespace greatest_multiple_of_5_and_6_less_than_1000_l247_247367

theorem greatest_multiple_of_5_and_6_less_than_1000 : 
  ∃ n, n < 1000 ∧ n % 5 = 0 ∧ n % 6 = 0 ∧ ∀ m, m < 1000 ∧ m % 5 = 0 ∧ m % 6 = 0 → m ≤ n :=
exists.intro 990 (by {
  -- proof goes here
  sorry
})

end greatest_multiple_of_5_and_6_less_than_1000_l247_247367


namespace perfect_square_polynomial_l247_247906

theorem perfect_square_polynomial (m : ℝ) :
  (∃ f : ℝ → ℝ, ∀ x, x^2 - (m + 1) * x + 1 = (f x) * (f x)) → (m = 1 ∨ m = -3) :=
by
  sorry

end perfect_square_polynomial_l247_247906


namespace simplify_fraction_l247_247718

theorem simplify_fraction (x : ℝ) (hx : x ≠ 1) : (x^2 / (x-1)) - (1 / (x-1)) = x + 1 :=
by 
  sorry

end simplify_fraction_l247_247718


namespace sin_cos_identity_l247_247559

theorem sin_cos_identity :
  (Real.sin (75 * Real.pi / 180) * Real.cos (30 * Real.pi / 180) -
   Real.sin (15 * Real.pi / 180) * Real.sin (150 * Real.pi / 180)) = (Real.sqrt 2 / 2) :=
by
  sorry

end sin_cos_identity_l247_247559


namespace cylinder_radius_l247_247189

open Real

theorem cylinder_radius (r : ℝ) 
  (h₁ : ∀(V₁ : ℝ), V₁ = π * (r + 4)^2 * 3)
  (h₂ : ∀(V₂ : ℝ), V₂ = π * r^2 * 9)
  (h₃ : ∀(V₁ V₂ : ℝ), V₁ = V₂) :
  r = 2 + 2 * sqrt 3 :=
by
  sorry

end cylinder_radius_l247_247189


namespace problem1_problem2_l247_247072

-- For problem 1: Prove the quotient is 5.
def f (n : ℕ) : ℕ := 
  let a := n / 100
  let b := (n % 100) / 10
  let c := n % 10
  a + b + c + a * b + b * c + c * a + a * b * c

theorem problem1 : (625 / f 625) = 5 :=
by
  sorry

-- For problem 2: Prove the set of numbers.
def three_digit_numbers_satisfying_quotient : Finset ℕ :=
  {199, 299, 399, 499, 599, 699, 799, 899, 999}

theorem problem2 (n : ℕ) : (100 ≤ n ∧ n < 1000) ∧ n / f n = 1 ↔ n ∈ three_digit_numbers_satisfying_quotient :=
by
  sorry

end problem1_problem2_l247_247072


namespace equal_savings_l247_247525

theorem equal_savings (U B UE BE US BS : ℕ) (h1 : U / B = 8 / 7) 
                      (h2 : U = 16000) (h3 : UE / BE = 7 / 6) (h4 : US = BS) :
                      US = 2000 ∧ BS = 2000 :=
by
  sorry

end equal_savings_l247_247525


namespace solve_for_q_l247_247470

theorem solve_for_q (p q : ℚ) (h1 : 5 * p + 6 * q = 14) (h2 : 6 * p + 5 * q = 17) : q = -1 / 11 :=
by
  sorry

end solve_for_q_l247_247470


namespace same_color_socks_prob_l247_247976

noncomputable def probability_of_at_least_one_pair_same_color :
    ℕ → ℕ → ℕ → ℕ → ℚ
  | total_socks, white_socks, red_socks, black_socks =>
    let total_ways := Nat.choose total_socks 3
    let diff_colors_ways := white_socks * red_socks * black_socks
    (total_ways - diff_colors_ways) / total_ways

theorem same_color_socks_prob :
  probability_of_at_least_one_pair_same_color 40 10 12 18 = 193 / 247 :=
by
  sorry

end same_color_socks_prob_l247_247976


namespace greatest_area_difference_l247_247545

theorem greatest_area_difference (l₁ w₁ l₂ w₂ : ℕ) 
  (h₁ : 2 * l₁ + 2 * w₁ = 160) 
  (h₂ : 2 * l₂ + 2 * w₂ = 160) : 
  1521 = (l₁ * w₁ - l₂ * w₂) → 
  (∃ l w : ℕ, 2 * l + 2 * w = 160 ∧ l * w = 1600 ∧ (l₁ = l ∧ w₁ = w) ∨ (l₂ = l ∧ w₂ = w)) ∧ 
  (∃ l w : ℕ, 2 * l + 2 * w = 160 ∧ l * w = 79 ∧ (l₁ = l ∧ w₁ = w) ∨ (l₂ = l ∧ w₂ = w)) :=
sorry

end greatest_area_difference_l247_247545


namespace part_1_odd_function_part_2_decreasing_l247_247049

noncomputable def f (x : ℝ) : ℝ := (1 - 2^x) / (1 + 2^x)

theorem part_1_odd_function : ∀ x : ℝ, f (-x) = -f x := by
  intro x
  sorry

theorem part_2_decreasing : ∀ x1 x2 : ℝ, x1 < x2 → f x1 > f x2 := by
  intros x1 x2 h
  sorry

end part_1_odd_function_part_2_decreasing_l247_247049


namespace number_is_28_l247_247471

-- Definitions from conditions in part a
def inner_expression := 15 - 15
def middle_expression := 37 - inner_expression
def outer_expression (some_number : ℕ) := 45 - (some_number - middle_expression)

-- Lean 4 statement to state the proof problem
theorem number_is_28 (some_number : ℕ) (h : outer_expression some_number = 54) : some_number = 28 := by
  sorry

end number_is_28_l247_247471


namespace smallest_n_divides_999_l247_247739

/-- 
Given \( 1 \leq n < 1000 \), \( n \) divides 999, and \( n+6 \) divides 99,
prove that the smallest possible value of \( n \) is 27.
 -/
theorem smallest_n_divides_999 (n : ℕ) 
  (h1 : 1 ≤ n) 
  (h2 : n < 1000) 
  (h3 : n ∣ 999) 
  (h4 : n + 6 ∣ 99) : 
  n = 27 :=
  sorry

end smallest_n_divides_999_l247_247739


namespace multiple_proof_l247_247831

theorem multiple_proof (n m : ℝ) (h1 : n = 25) (h2 : m * n = 3 * n - 25) : m = 2 := by
  sorry

end multiple_proof_l247_247831


namespace sum_last_two_digits_of_powers_l247_247963

theorem sum_last_two_digits_of_powers (h₁ : 9 = 10 - 1) (h₂ : 11 = 10 + 1) :
  (9^20 + 11^20) % 100 / 10 + (9^20 + 11^20) % 10 = 2 :=
by
  sorry

end sum_last_two_digits_of_powers_l247_247963


namespace sequence_even_odd_l247_247956

theorem sequence_even_odd : ∃ a1 : ℕ, (∀ n : ℕ, 1 ≤ n ∧ n ≤ 100000 → (a n) % 2 = 0) ∧ (a 100001 % 2 = 1) :=
by
  let a : ℕ → ℕ
  assume h : ∀ n : ℕ, a (n + 1) = Nat.floor (1.5 * ↑(a n)) + 1
  sorry

end sequence_even_odd_l247_247956


namespace find_y_given_area_l247_247229

-- Define the problem parameters and conditions
namespace RectangleArea

variables {y : ℝ} (y_pos : y > 0)

-- Define the vertices, they can be expressed but are not required in the statement
def vertices := [(-2, y), (8, y), (-2, 3), (8, 3)]

-- Define the area condition
def area_condition := 10 * (y - 3) = 90

-- Lean statement proving y = 12 given the conditions
theorem find_y_given_area (y_pos : y > 0) (h : 10 * (y - 3) = 90) : y = 12 :=
by
  sorry

end RectangleArea

end find_y_given_area_l247_247229


namespace greatest_multiple_of_5_and_6_under_1000_l247_247376

theorem greatest_multiple_of_5_and_6_under_1000 : 
  ∃ x, (x % 5 = 0 ∧ x % 6 = 0 ∧ x < 1000) ∧ 
  (∀ y, (y % 5 = 0 ∧ y % 6 = 0 ∧ y < 1000) → y ≤ x) ∧ 
  x = 990 :=
begin
  sorry
end

end greatest_multiple_of_5_and_6_under_1000_l247_247376


namespace parallelogram_height_l247_247281

theorem parallelogram_height (b A : ℝ) (h : ℝ) (h_base : b = 28) (h_area : A = 896) : h = A / b := by
  simp [h_base, h_area]
  norm_num
  sorry

end parallelogram_height_l247_247281


namespace fifth_number_21st_row_is_809_l247_247945

-- Define the sequence of positive odd numbers
def nth_odd_number (n : ℕ) : ℕ :=
  2 * n - 1

-- Define the last odd number in the nth row
def last_odd_number_in_row (n : ℕ) : ℕ :=
  nth_odd_number (n * n)

-- Define the position of the 5th number in the 21st row
def pos_5th_in_21st_row : ℕ :=
  let sum_first_20_rows := 400
  sum_first_20_rows + 5

-- The 5th number from the left in the 21st row
def fifth_number_in_21st_row : ℕ :=
  nth_odd_number pos_5th_in_21st_row

-- The proof statement
theorem fifth_number_21st_row_is_809 : fifth_number_in_21st_row = 809 :=
by
  -- proof omitted
  sorry

end fifth_number_21st_row_is_809_l247_247945


namespace shaded_area_l247_247104

theorem shaded_area (d_small : ℝ) (r_large : ℝ) (shaded_area : ℝ) :
  (d_small = 6) → (r_large = 3 * (d_small / 2)) → shaded_area = (π * r_large^2 - π * (d_small / 2)^2) → shaded_area = 72 * π :=
by
  intro h_d_small h_r_large h_shaded_area
  rw [h_d_small, h_r_large, h_shaded_area]
  sorry

end shaded_area_l247_247104


namespace problem_proof_l247_247156

theorem problem_proof (p : ℕ) (hodd : p % 2 = 1) (hgt : p > 3):
  ((p - 3) ^ (1 / 2 * (p - 1)) - 1 ∣ p - 4) ∧
  ¬((p - 3) ^ (1 / 2 * (p - 1)) + 1 ∣ p) ∧
  ¬((p - 3) ^ (1 / 2 * (p - 1)) ∣ p) ∧
  ¬((p - 3) ^ (1 / 2 * (p - 1)) + 1 ∣ p + 1) ∧
  ¬((p - 3) ^ (1 / 2 * (p - 1)) - 1 ∣ p - 3) :=
by
  sorry

end problem_proof_l247_247156


namespace total_amount_l247_247969

theorem total_amount (x_share : ℝ) (y_share : ℝ) (w_share : ℝ) (hx : x_share = 0.30) (hy : y_share = 0.20) (hw : w_share = 10) :
  (w_share * (1 + x_share + y_share)) = 15 := by
  sorry

end total_amount_l247_247969


namespace gail_working_hours_x_l247_247886

theorem gail_working_hours_x (x : ℕ) (hx : x < 12) : 
  let hours_am := 12 - x
  let hours_pm := x
  hours_am + hours_pm = 12 := 
by {
  sorry
}

end gail_working_hours_x_l247_247886


namespace sum_fraction_nonnegative_le_one_l247_247786

theorem sum_fraction_nonnegative_le_one 
  (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (habc : a + b + c = 2) :
  a * b / (c^2 + 1) + b * c / (a^2 + 1) + c * a / (b^2 + 1) ≤ 1 :=
sorry

end sum_fraction_nonnegative_le_one_l247_247786


namespace expected_value_ball_draw_l247_247954

noncomputable def E_xi : ℚ :=
  let prob_xi_2 := 3/5
  let prob_xi_3 := 3/10
  let prob_xi_4 := 1/10
  2 * prob_xi_2 + 3 * prob_xi_3 + 4 * prob_xi_4

theorem expected_value_ball_draw : E_xi = 5 / 2 := by
  sorry

end expected_value_ball_draw_l247_247954


namespace assign_grades_l247_247135

def num_students : ℕ := 15
def options_per_student : ℕ := 4

theorem assign_grades:
  options_per_student ^ num_students = 1073741824 := by
  sorry

end assign_grades_l247_247135


namespace compute_scalar_dot_product_l247_247846

open Matrix 

def vec1 : Fin 2 → ℤ
| 0 => -2
| 1 => 3

def vec2 : Fin 2 → ℤ
| 0 => 4
| 1 => -5

def dot_product (v1 v2 : Fin 2 → ℤ) : ℤ :=
  (v1 0) * (v2 0) + (v1 1) * (v2 1)

theorem compute_scalar_dot_product :
  3 * dot_product vec1 vec2 = -69 := 
by 
  sorry

end compute_scalar_dot_product_l247_247846


namespace pirate_treasure_chest_coins_l247_247139

theorem pirate_treasure_chest_coins:
  ∀ (gold_coins silver_coins bronze_coins: ℕ) (chests: ℕ),
    gold_coins = 3500 →
    silver_coins = 500 →
    bronze_coins = 2 * silver_coins →
    chests = 5 →
    (gold_coins / chests + silver_coins / chests + bronze_coins / chests = 1000) :=
by
  intros gold_coins silver_coins bronze_coins chests gold_eq silv_eq bron_eq chest_eq
  sorry

end pirate_treasure_chest_coins_l247_247139


namespace adventure_club_probability_l247_247942

theorem adventure_club_probability :
  let total_members := 30
  let boys := 12
  let girls := 18
  let committee_size := 5
  let total_ways := Nat.choose total_members committee_size
  let ways_all_boys := Nat.choose boys committee_size
  let ways_all_girls := Nat.choose girls committee_size
  let ways_all_boys_or_girls := ways_all_boys + ways_all_girls
  let probability := 1 - (ways_all_boys_or_girls / total_ways : ℚ)
  (probability = (59/63 : ℚ)) :=
by
  sorry

end adventure_club_probability_l247_247942


namespace cos_double_angle_l247_247000

theorem cos_double_angle (α : ℝ) (h : Real.sin α = 1 / 5) : Real.cos (2 * α) = 23 / 25 :=
sorry

end cos_double_angle_l247_247000


namespace algebraic_expression_is_200_l247_247819

-- Define the condition
def satisfies_ratio (x : ℕ) : Prop :=
  x / 10 = 20

-- The proof problem statement
theorem algebraic_expression_is_200 : ∃ x : ℕ, satisfies_ratio x ∧ x = 200 :=
by
  -- Providing the necessary proof infrastructure
  use 200
  -- Assuming the proof is correct
  sorry


end algebraic_expression_is_200_l247_247819


namespace math_proof_problem_l247_247993

theorem math_proof_problem : (10^8 / (2 * 10^5) - 50) = 450 := 
  by
  sorry

end math_proof_problem_l247_247993


namespace ratio_major_minor_is_15_4_l247_247198

-- Define the given conditions
def main_characters : ℕ := 5
def minor_characters : ℕ := 4
def minor_character_pay : ℕ := 15000
def total_payment : ℕ := 285000

-- Define the total pay to minor characters
def minor_total_pay : ℕ := minor_characters * minor_character_pay

-- Define the total pay to major characters
def major_total_pay : ℕ := total_payment - minor_total_pay

-- Define the ratio computation
def ratio_major_minor : ℕ × ℕ := (major_total_pay / 15000, minor_total_pay / 15000)

-- State the theorem
theorem ratio_major_minor_is_15_4 : ratio_major_minor = (15, 4) :=
by
  -- Proof goes here
  sorry

end ratio_major_minor_is_15_4_l247_247198


namespace indolent_student_probability_l247_247529

-- Define the constants of the problem
def n : ℕ := 30  -- total number of students
def k : ℕ := 3   -- number of students selected each lesson
def m : ℕ := 10  -- number of students from the previous lesson

-- Define the probabilities
def P_asked_in_one_lesson : ℚ := 1 / k
def P_asked_twice_in_a_row : ℚ := 1 / n
def P_overall : ℚ := P_asked_in_one_lesson + P_asked_in_one_lesson - P_asked_twice_in_a_row
def P_avoid_reciting : ℚ := 1 - P_overall

theorem indolent_student_probability : P_avoid_reciting = 11 / 30 := 
  sorry

end indolent_student_probability_l247_247529


namespace faster_train_cross_time_l247_247090

noncomputable def time_to_cross (speed_fast_kmph : ℝ) (speed_slow_kmph : ℝ) (length_fast_m : ℝ) : ℝ :=
  let speed_diff_kmph := speed_fast_kmph - speed_slow_kmph
  let speed_diff_mps := (speed_diff_kmph * 1000) / 3600
  length_fast_m / speed_diff_mps

theorem faster_train_cross_time :
  time_to_cross 72 36 120 = 12 :=
by
  sorry

end faster_train_cross_time_l247_247090


namespace max_tan_A_minus_B_l247_247634

noncomputable def triangle_max_tan_A_minus_B (a b c : ℝ) (A B C : ℝ) 
  (h1 : a * Real.cos B - b * Real.cos C = (1 / 2) * c) : ℝ :=
  if h2 : True then (√3 / 3) else 0

theorem max_tan_A_minus_B {a b c A B C : ℝ} 
  (h1 : a * Real.cos B - b * Real.cos C = (1 / 2) * c) :
  triangle_max_tan_A_minus_B a b c A B C h1 = √(3) / 3 :=
sorry

end max_tan_A_minus_B_l247_247634


namespace a_plus_b_equals_4_l247_247477

theorem a_plus_b_equals_4 (f : ℝ → ℝ) (a b : ℝ) (h_dom : ∀ x, 1 ≤ x ∧ x ≤ b → f x = (1/2) * (x-1)^2 + a)
  (h_range : ∀ y, 1 ≤ y ∧ y ≤ b → ∃ x, 1 ≤ x ∧ x ≤ b ∧ f x = y) (h_b_pos : b > 1) : a + b = 4 :=
sorry

end a_plus_b_equals_4_l247_247477


namespace quadratic_minimization_l247_247155

theorem quadratic_minimization : 
  ∃ x : ℝ, ∀ y : ℝ, (x^2 - 12 * x + 36 ≤ y^2 - 12 * y + 36) ∧ x^2 - 12 * x + 36 = 0 :=
by
  sorry

end quadratic_minimization_l247_247155


namespace distance_traveled_by_both_cars_l247_247665

def car_R_speed := 34.05124837953327
def car_P_speed := 44.05124837953327
def car_R_time := 8.810249675906654
def car_P_time := car_R_time - 2

def distance_car_R := car_R_speed * car_R_time
def distance_car_P := car_P_speed * car_P_time

theorem distance_traveled_by_both_cars :
  distance_car_R = 300 :=
by
  sorry

end distance_traveled_by_both_cars_l247_247665


namespace pythagorean_triple_example_l247_247966

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

theorem pythagorean_triple_example :
  is_pythagorean_triple 7 24 25 :=
sorry

end pythagorean_triple_example_l247_247966


namespace find_roots_of_complex_quadratic_eq_l247_247882

theorem find_roots_of_complex_quadratic_eq {z : ℂ} : z^2 + 2 * z + (3 - 4i) = 0 ↔ (z = -1 + 2i ∨ z = -3 - 2i) := by
  sorry

end find_roots_of_complex_quadratic_eq_l247_247882


namespace solve_tan_equation_l247_247246

theorem solve_tan_equation (x : ℝ) (k : ℤ) :
  8.456 * (Real.tan x)^2 * (Real.tan (3 * x))^2 * Real.tan (4 * x) = 
  (Real.tan x)^2 - (Real.tan (3 * x))^2 + Real.tan (4 * x) ->
  x = π * k ∨ x = π / 4 * (2 * k + 1) := sorry

end solve_tan_equation_l247_247246


namespace total_distance_thrown_l247_247789

theorem total_distance_thrown (D : ℝ) (total_distance : ℝ) 
  (h1 : total_distance = 20 * D + 60 * D) : 
  total_distance = 1600 := 
by
  sorry

end total_distance_thrown_l247_247789


namespace total_food_items_in_one_day_l247_247491

-- Define the food consumption for each individual
def JorgeCroissants := 7
def JorgeCakes := 18
def JorgePizzas := 30

def GiulianaCroissants := 5
def GiulianaCakes := 14
def GiulianaPizzas := 25

def MatteoCroissants := 6
def MatteoCakes := 16
def MatteoPizzas := 28

-- Define the total number of each food type consumed
def totalCroissants := JorgeCroissants + GiulianaCroissants + MatteoCroissants
def totalCakes := JorgeCakes + GiulianaCakes + MatteoCakes
def totalPizzas := JorgePizzas + GiulianaPizzas + MatteoPizzas

-- The theorem statement
theorem total_food_items_in_one_day : 
  totalCroissants + totalCakes + totalPizzas = 149 :=
by
  -- Proof is omitted
  sorry

end total_food_items_in_one_day_l247_247491


namespace prod_mod_6_l247_247721

theorem prod_mod_6 (h1 : 2015 % 6 = 3) (h2 : 2016 % 6 = 0) (h3 : 2017 % 6 = 1) (h4 : 2018 % 6 = 2) : 
  (2015 * 2016 * 2017 * 2018) % 6 = 0 := 
by 
  sorry

end prod_mod_6_l247_247721


namespace line_problems_l247_247301

noncomputable def l1 : (ℝ → ℝ) := λ x => x - 1
noncomputable def l2 (k : ℝ) : (ℝ → ℝ) := λ x => -(k + 1) / k * x - 1

theorem line_problems (k : ℝ) :
  ∃ k, k = 0 → (l2 k 1) = 90 →      -- A
  (∀ k, (l1 1 = l2 k 1 → True)) →   -- B
  (∀ k, (l1 1 ≠ l2 k 1 → True)) →   -- C (negated conclusion from False in C)
  (∀ k, (l1 1 * l2 k 1 ≠ -1))       -- D
:=
sorry

end line_problems_l247_247301


namespace miraflores_optimal_split_l247_247531

-- Define the total number of voters as 2n, and initialize half supporters for each candidate.
variable (n : ℕ) (voters : Fin (2 * n) → Bool)

-- Define the condition that exactly half of the voters including Miraflores support him
def half_support_miraflores : Prop :=
  ∃ (supporters_miraflores : Fin n) (supporters_maloney : Fin n), 
    (voters supporters_miraflores.val = true) ∧ (voters.supporters_maloney.val = false) 

-- Define the condition of drawing a single random ballot in each district.
def draw_random_ballot (d : Fin n → Prop) : Fin n := sorry

-- Define the condition that Miraflores wins if he wins both districts.
def wins_election (d1 d2 : Fin n → Prop) : Prop := 
  (draw_random_ballot d1 = true) ∧ (draw_random_ballot d2 = true)

-- Miraflores should split the voters such that his maximum probability of winning is achieved.
def optimal_split : Prop :=
  ∃ (d1 d2 : Fin n → Bool), 
    (d1.supporters_miraflores.val = true ∧ d2.supporters_maloney.val = false) ∧
    (wins_election d1 d2 = true)

theorem miraflores_optimal_split (n : ℕ) (voters : Fin (2 * n) → Bool) (half_support : half_support_miraflores n voters) : optimal_split n :=
sorry

end miraflores_optimal_split_l247_247531


namespace fill_parentheses_correct_l247_247179

theorem fill_parentheses_correct (a b : ℝ) :
  (3 * b + a) * (3 * b - a) = 9 * b^2 - a^2 :=
by 
  sorry

end fill_parentheses_correct_l247_247179


namespace number_of_bouncy_balls_per_package_l247_247649

theorem number_of_bouncy_balls_per_package (x : ℕ) (h : 4 * x + 8 * x + 4 * x = 160) : x = 10 :=
by
  sorry

end number_of_bouncy_balls_per_package_l247_247649


namespace trigonometric_identity_l247_247451

theorem trigonometric_identity 
  (α : ℝ) 
  (h : Real.sin (α + π / 3) = 12 / 13) 
  : Real.cos (π / 6 - α) = 12 / 13 := 
sorry

end trigonometric_identity_l247_247451


namespace greatest_multiple_l247_247371

theorem greatest_multiple (n : ℕ) (h1 : n < 1000) (h2 : n % 5 = 0) (h3 : n % 6 = 0) : n = 990 :=
sorry

end greatest_multiple_l247_247371


namespace seth_oranges_l247_247657

def initial_boxes := 9
def boxes_given_to_mother := 1

def remaining_boxes_after_giving_to_mother := initial_boxes - boxes_given_to_mother
def boxes_given_away := remaining_boxes_after_giving_to_mother / 2
def boxes_left := remaining_boxes_after_giving_to_mother - boxes_given_away

theorem seth_oranges : boxes_left = 4 := by
  sorry

end seth_oranges_l247_247657


namespace total_bill_l247_247803

theorem total_bill (m : ℝ) (h1 : m = 10 * (m / 10 + 3) - 27) : m = 270 :=
by
  sorry

end total_bill_l247_247803


namespace wang_payment_correct_l247_247260

noncomputable def first_trip_payment (x : ℝ) : ℝ := 0.9 * x
noncomputable def second_trip_payment (y : ℝ) : ℝ := 300 * 0.9 + (y - 300) * 0.8

theorem wang_payment_correct (x y: ℝ) 
  (cond1: 0.1 * x = 19)
  (cond2: (x + y) - (0.9 * x + ((y - 300) * 0.8 + 300 * 0.9)) = 67) :
  first_trip_payment x = 171 ∧ second_trip_payment y = 342 := 
by
  sorry

end wang_payment_correct_l247_247260


namespace james_total_cost_l247_247921

def milk_cost : ℝ := 4.50
def milk_tax_rate : ℝ := 0.20
def banana_cost : ℝ := 3.00
def banana_tax_rate : ℝ := 0.15
def baguette_cost : ℝ := 2.50
def baguette_tax_rate : ℝ := 0.0
def cereal_cost : ℝ := 6.00
def cereal_discount_rate : ℝ := 0.20
def cereal_tax_rate : ℝ := 0.12
def eggs_cost : ℝ := 3.50
def eggs_coupon : ℝ := 1.00
def eggs_tax_rate : ℝ := 0.18

theorem james_total_cost :
  let milk_total := milk_cost * (1 + milk_tax_rate)
  let banana_total := banana_cost * (1 + banana_tax_rate)
  let baguette_total := baguette_cost * (1 + baguette_tax_rate)
  let cereal_discounted := cereal_cost * (1 - cereal_discount_rate)
  let cereal_total := cereal_discounted * (1 + cereal_tax_rate)
  let eggs_discounted := eggs_cost - eggs_coupon
  let eggs_total := eggs_discounted * (1 + eggs_tax_rate)
  milk_total + banana_total + baguette_total + cereal_total + eggs_total = 19.68 := 
by
  sorry

end james_total_cost_l247_247921


namespace cucumbers_for_20_apples_l247_247314

theorem cucumbers_for_20_apples (A B C : ℝ) (h1 : 10 * A = 5 * B) (h2 : 3 * B = 4 * C) :
  20 * A = 40 / 3 * C :=
by
  sorry

end cucumbers_for_20_apples_l247_247314


namespace sam_coins_and_value_l247_247511

-- Define initial conditions
def initial_dimes := 9
def initial_nickels := 5
def initial_pennies := 12

def dimes_from_dad := 7
def nickels_taken_by_dad := 3

def pennies_exchanged := 12
def dimes_from_exchange := 2
def pennies_from_exchange := 2

-- Define final counts of coins after transactions
def final_dimes := initial_dimes + dimes_from_dad + dimes_from_exchange
def final_nickels := initial_nickels - nickels_taken_by_dad
def final_pennies := initial_pennies - pennies_exchanged + pennies_from_exchange

-- Define the total count of coins
def total_coins := final_dimes + final_nickels + final_pennies

-- Define the total value in cents
def value_dimes := final_dimes * 10
def value_nickels := final_nickels * 5
def value_pennies := final_pennies * 1

def total_value := value_dimes + value_nickels + value_pennies

-- Proof statement
theorem sam_coins_and_value :
  total_coins = 22 ∧ total_value = 192 := by
  -- Proof details would go here
  sorry

end sam_coins_and_value_l247_247511


namespace train_length_is_200_l247_247423

noncomputable def train_length 
  (speed_kmh : ℕ) 
  (time_s: ℕ) : ℕ := 
  ((speed_kmh * 1000) / 3600) * time_s

theorem train_length_is_200
  (h_speed : 40 = 40)
  (h_time : 18 = 18) :
  train_length 40 18 = 200 :=
sorry

end train_length_is_200_l247_247423


namespace solve_system_of_equations_l247_247515

theorem solve_system_of_equations :
  ∀ (x1 x2 x3 x4 x5: ℝ), 
  (x3 + x4 + x5)^5 = 3 * x1 ∧ 
  (x4 + x5 + x1)^5 = 3 * x2 ∧ 
  (x5 + x1 + x2)^5 = 3 * x3 ∧ 
  (x1 + x2 + x3)^5 = 3 * x4 ∧ 
  (x2 + x3 + x4)^5 = 3 * x5 →
  (x1 = 0 ∧ x2 = 0 ∧ x3 = 0 ∧ x4 = 0 ∧ x5 = 0) ∨ 
  (x1 = 1/3 ∧ x2 = 1/3 ∧ x3 = 1/3 ∧ x4 = 1/3 ∧ x5 = 1/3) ∨ 
  (x1 = -1/3 ∧ x2 = -1/3 ∧ x3 = -1/3 ∧ x4 = -1/3 ∧ x5 = -1/3) := 
by 
  sorry

end solve_system_of_equations_l247_247515


namespace nat_set_eq_l247_247233

open Finset

noncomputable def nat_set := {x : ℕ | 8 < x ∧ x < 12}

theorem nat_set_eq : nat_set = {9, 10, 11} :=
by
  ext x
  simp only [mem_set_of, mem_insert, mem_singleton, mem_empty]
  constructor
  · rintro ⟨h₈, h₁₂⟩
    interval_cases x
    · left
      refl
    · right
      left
      refl
    · right
      right
      left
      refl
  · rintro (rfl | rfl | (rfl | h))
    · exact ⟨(by linarith), (by linarith)⟩
    · exact ⟨(by linarith), (by linarith)⟩
    · exact ⟨(by linarith), (by linarith)⟩
    · cases h

end nat_set_eq_l247_247233


namespace uniformly_integrable_iff_l247_247826

variables {α : Type*} [MeasurableSpace α] {μ : MeasureTheory.Measure α}

-- Define the sequence of random variables
def xi (n : ℕ) : α → ℝ := sorry

-- Define the function G
noncomputable def G : ℝ → ℝ := sorry

-- Define the required conditions on G
axiom G_condition_1 : ∀ x, 0 ≤ G(x)
axiom G_condition_2 : ∀ x y, (x ≤ y) → (G(x) ≤ G(y))
axiom G_condition_3 : ∀ ε > 0, ∃ M > 0, ∀ x > M, G(x)/x > ε
axiom G_condition_4 : convex_on ℝ (set.univ) G

-- Define uniformly integrable sequence
def uniformly_integrable (seq : ℕ → α → ℝ) (μ : MeasureTheory.Measure α) :=
  ∀ ε > 0, ∃ δ > 0, ∀ s, μ s < δ → ∑ i, μ[| seq i | * indicator s] < ε

-- Prove the equivalence statement
theorem uniformly_integrable_iff :
  uniformly_integrable xi μ ↔ ∃ G, (∀ x, 0 ≤ G(x)) ∧ (∀ x y, (x ≤ y) → (G(x) ≤ G(y))) 
                              ∧ (∀ ε > 0, ∃ M > 0, ∀ x > M, G(x)/x > ε) 
                              ∧ convex_on ℝ (set.univ) G 
                              ∧ (∀ n, ∃ M, μ[| (G(xi n)) |] < M) :=
by sorry

end uniformly_integrable_iff_l247_247826


namespace fifteen_pow_mn_eq_PnQm_l247_247217

-- Definitions
def P (m : ℕ) := 3^m
def Q (n : ℕ) := 5^n

-- Theorem statement
theorem fifteen_pow_mn_eq_PnQm (m n : ℕ) : 15^(m * n) = (P m)^n * (Q n)^m :=
by
  -- Placeholder for the proof, which isn't required
  sorry

end fifteen_pow_mn_eq_PnQm_l247_247217


namespace meanScore_is_91_666_l247_247777

-- Define Jane's quiz scores
def janesScores : List ℕ := [85, 88, 90, 92, 95, 100]

-- Define the total sum of Jane's quiz scores
def sumScores (scores : List ℕ) : ℕ := scores.foldl (· + ·) 0

-- The number of Jane's quiz scores
def numberOfScores (scores : List ℕ) : ℕ := scores.length

-- Define the mean of Jane's quiz scores
def meanScore (scores : List ℕ) : ℚ := sumScores scores / numberOfScores scores

-- The theorem to be proven
theorem meanScore_is_91_666 (h : janesScores = [85, 88, 90, 92, 95, 100]) :
  meanScore janesScores = 91.66666666666667 := by 
  sorry

end meanScore_is_91_666_l247_247777


namespace gcd_97_pow_10_plus_1_and_97_pow_10_plus_97_pow_3_plus_1_l247_247443

theorem gcd_97_pow_10_plus_1_and_97_pow_10_plus_97_pow_3_plus_1 :
  Int.gcd (97 ^ 10 + 1) (97 ^ 10 + 97 ^ 3 + 1) = 1 := sorry

end gcd_97_pow_10_plus_1_and_97_pow_10_plus_97_pow_3_plus_1_l247_247443


namespace cost_of_toys_target_weekly_price_l247_247147

-- First proof problem: Cost of Plush Toy and Metal Ornament
theorem cost_of_toys (x : ℝ) (hx : 6400 / x = 2 * (4000 / (x + 20))) : 
  x = 80 :=
by sorry

-- Second proof problem: Price to achieve target weekly profit
theorem target_weekly_price (y : ℝ) (hy : (y - 80) * (10 + (150 - y) / 5) = 720) :
  y = 140 :=
by sorry

end cost_of_toys_target_weekly_price_l247_247147


namespace max_shortest_part_duration_l247_247417

theorem max_shortest_part_duration (film_duration : ℕ) (part1 part2 part3 part4 : ℕ)
  (h_total : part1 + part2 + part3 + part4 = 192)
  (h_diff1 : part2 ≥ part1 + 6)
  (h_diff2 : part3 ≥ part2 + 6)
  (h_diff3 : part4 ≥ part3 + 6) :
  part1 ≤ 39 := 
sorry

end max_shortest_part_duration_l247_247417


namespace smallest_N_value_l247_247885

theorem smallest_N_value (a b c d : ℕ)
  (h1 : gcd a b = 1 ∧ gcd a c = 2 ∧ gcd a d = 4 ∧ gcd b c = 5 ∧ gcd b d = 3 ∧ gcd c d = N)
  (h2 : N > 5) : N = 14 := sorry

end smallest_N_value_l247_247885


namespace area_of_PDCE_l247_247319

/-- A theorem to prove the area of quadrilateral PDCE given conditions in triangle ABC. -/
theorem area_of_PDCE
  (ABC_area : ℝ)
  (BD_to_CD_ratio : ℝ)
  (E_is_midpoint : Prop)
  (AD_intersects_BE : Prop)
  (P : Prop)
  (area_PDCE : ℝ) :
  (ABC_area = 1) →
  (BD_to_CD_ratio = 2 / 1) →
  E_is_midpoint →
  AD_intersects_BE →
  ∃ P, P →
    area_PDCE = 7 / 30 :=
by sorry

end area_of_PDCE_l247_247319


namespace polar_equations_and_ratios_l247_247485

open Real

theorem polar_equations_and_ratios (α β : ℝ)
    (h_line : ∀ (r θ : ℝ), r * cos θ = 2 ↔ r = 2 / cos θ)
    (h_curve : ∀ (α : ℝ), ∃ (r θ : ℝ), r = 2 * sin θ ∧ θ = β ∧ 0 < β ∧ β < π / 2) :
    ( ∀ (r θ : ℝ), r * cos θ = 2 ↔ r = 2 / cos θ) ∧
    ( ∃ (r θ : ℝ), r = 2 * sin θ ∧ θ = β ∧ 0 < β ∧ β < π / 2 → 
    0 < r * sin 2 * θ / (r / cos θ) ∧ r * sin 2 * θ / (r / cos θ) ≤ 1 / 2) :=
by
  sorry

end polar_equations_and_ratios_l247_247485


namespace power_division_l247_247095

theorem power_division (a b : ℕ) (h : 64 = 8^2) : (8 ^ 15) / (64 ^ 7) = 8 := by
  sorry

end power_division_l247_247095


namespace problem_8_div_64_pow_7_l247_247098

theorem problem_8_div_64_pow_7:
  (64 : ℝ) = (8 : ℝ)^2 →
  8^15 / 64^7 = 8 :=
by
  intro h
  rw [h]
  have : (64^7 : ℝ) = (8^2)^7 := by rw [h]
  rw [this]
  rw [pow_mul]
  field_simp
  norm_num

end problem_8_div_64_pow_7_l247_247098


namespace students_voted_both_l247_247584

def total_students : Nat := 300
def students_voted_first : Nat := 230
def students_voted_second : Nat := 190
def students_voted_none : Nat := 40

theorem students_voted_both :
  students_voted_first + students_voted_second - (total_students - students_voted_none) = 160 :=
by
  sorry

end students_voted_both_l247_247584


namespace pears_total_l247_247335

-- Conditions
def keith_initial_pears : ℕ := 47
def keith_given_pears : ℕ := 46
def mike_initial_pears : ℕ := 12

-- Define the remaining pears
def keith_remaining_pears : ℕ := keith_initial_pears - keith_given_pears
def mike_remaining_pears : ℕ := mike_initial_pears

-- Theorem statement
theorem pears_total :
  keith_remaining_pears + mike_remaining_pears = 13 :=
by
  sorry

end pears_total_l247_247335


namespace factorize_a_cubed_minus_a_l247_247866

theorem factorize_a_cubed_minus_a (a : ℝ) : a^3 - a = a * (a + 1) * (a - 1) :=
sorry

end factorize_a_cubed_minus_a_l247_247866
