import Mathlib

namespace evaluate_expression_l1938_193899

theorem evaluate_expression (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (hxy : x > y) (hyz : y > z) :
  (x ^ (y + z) * z ^ (x + y)) / (y ^ (x + z) * z ^ (y + x)) = (x / y) ^ (y + z) :=
by
  sorry

end evaluate_expression_l1938_193899


namespace ratio_of_books_l1938_193884

theorem ratio_of_books (books_last_week : ℕ) (pages_per_book : ℕ) (pages_this_week : ℕ)
  (h_books_last_week : books_last_week = 5)
  (h_pages_per_book : pages_per_book = 300)
  (h_pages_this_week : pages_this_week = 4500) :
  (pages_this_week / pages_per_book) / books_last_week = 3 := by
  sorry

end ratio_of_books_l1938_193884


namespace socks_ratio_l1938_193872

-- Definitions based on the conditions
def initial_black_socks : ℕ := 6
def initial_white_socks (B : ℕ) : ℕ := 4 * B
def remaining_white_socks (B : ℕ) : ℕ := B + 6

-- The theorem to prove the ratio is 1/2
theorem socks_ratio (B : ℕ) (hB : B = initial_black_socks) :
  ((initial_white_socks B - remaining_white_socks B) : ℚ) / initial_white_socks B = 1 / 2 :=
by
  sorry

end socks_ratio_l1938_193872


namespace find_B_l1938_193839

-- Define the polynomial function and its properties
def polynomial (z : ℤ) (A B : ℤ) : ℤ :=
  z^4 - 6 * z^3 + A * z^2 + B * z + 9

-- Prove that B = -9 under the given conditions
theorem find_B (A B : ℤ) (r1 r2 r3 r4 : ℤ)
  (h1 : polynomial r1 A B = 0)
  (h2 : polynomial r2 A B = 0)
  (h3 : polynomial r3 A B = 0)
  (h4 : polynomial r4 A B = 0)
  (h5 : r1 + r2 + r3 + r4 = 6)
  (h6 : r1 > 0)
  (h7 : r2 > 0)
  (h8 : r3 > 0)
  (h9 : r4 > 0) :
  B = -9 :=
by
  sorry

end find_B_l1938_193839


namespace intersection_complement_eq_l1938_193847

open Set

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 2, 4}
def B : Set ℕ := {1, 4}

theorem intersection_complement_eq : A ∩ (U \ B) = {0, 2} := by
  sorry

end intersection_complement_eq_l1938_193847


namespace min_value_xyz_l1938_193871

theorem min_value_xyz (x y z : ℝ) (h1 : xy + 2 * z = 1) (h2 : x^2 + y^2 + z^2 = 10 ) : xyz ≥ -28 :=
by
  sorry

end min_value_xyz_l1938_193871


namespace triangle_area_l1938_193883

variable (a b c : ℕ)
variable (s : ℕ := 21)
variable (area : ℕ := 84)

theorem triangle_area 
(h1 : c = a + b - 12) 
(h2 : (a + b + c) / 2 = s) 
(h3 : c - a = 2) 
: (21 * (21 - a) * (21 - b) * (21 - c)).sqrt = area := 
sorry

end triangle_area_l1938_193883


namespace sum_999_is_1998_l1938_193827

theorem sum_999_is_1998 : 999 + 999 = 1998 :=
by
  sorry

end sum_999_is_1998_l1938_193827


namespace max_volume_prism_l1938_193832

theorem max_volume_prism (a b h : ℝ) (h_congruent_lateral : a = b) (sum_areas_eq_48 : a * h + b * h + a * b = 48) : 
  ∃ V : ℝ, V = 64 :=
by
  sorry

end max_volume_prism_l1938_193832


namespace solve_quadratic_eq_l1938_193816

theorem solve_quadratic_eq (x : ℝ) : x^2 - 4 = 0 ↔ x = 2 ∨ x = -2 :=
by
  sorry

end solve_quadratic_eq_l1938_193816


namespace BD_value_l1938_193886

def quadrilateral_ABCD_sides (AB BC CD DA : ℕ) (BD : ℕ) : Prop :=
  AB = 5 ∧ BC = 17 ∧ CD = 5 ∧ DA = 9 ∧ 12 < BD ∧ BD < 14 ∧ BD = 13

theorem BD_value (AB BC CD DA : ℕ) (BD : ℕ) : 
  quadrilateral_ABCD_sides AB BC CD DA BD → BD = 13 :=
by
  sorry

end BD_value_l1938_193886


namespace selling_price_of_cycle_l1938_193820

theorem selling_price_of_cycle (original_price : ℝ) (loss_percentage : ℝ) (loss_amount : ℝ) (selling_price : ℝ) :
  original_price = 2000 →
  loss_percentage = 10 →
  loss_amount = (loss_percentage / 100) * original_price →
  selling_price = original_price - loss_amount →
  selling_price = 1800 :=
by
  intros
  sorry

end selling_price_of_cycle_l1938_193820


namespace solve_for_x_l1938_193878

variable (x y : ℝ)
variable (hx : x > 0)
variable (hy : y > 0)
variable (h : 3 * x^2 + 9 * x * y = x^3 + 3 * x^2 * y)

theorem solve_for_x : x = 3 :=
by
  sorry

end solve_for_x_l1938_193878


namespace right_triangle_iff_sum_excircles_a_right_triangle_iff_sum_excircles_b_l1938_193855

-- Definitions for part (a)
def is_right_angled_triangle_a (a b c r r_a r_b r_c : ℝ) :=
  r + r_a + r_b + r_c = a + b + c

def right_angled_triangle_a (a b c : ℝ) :=
  a^2 + b^2 = c^2 -- Assuming c is the hypotenuse

theorem right_triangle_iff_sum_excircles_a (a b c r r_a r_b r_c : ℝ) :
  right_angled_triangle_a a b c ↔ is_right_angled_triangle_a a b c r r_a r_b r_c := sorry

-- Definitions for part (b)
def is_right_angled_triangle_b (a b c r r_a r_b r_c : ℝ) :=
  r^2 + r_a^2 + r_b^2 + r_c^2 = a^2 + b^2 + c^2

def right_angled_triangle_b (a b c : ℝ) :=
  a^2 + b^2 = c^2 -- Assuming c is the hypotenuse

theorem right_triangle_iff_sum_excircles_b (a b c r r_a r_b r_c : ℝ) :
  right_angled_triangle_b a b c ↔ is_right_angled_triangle_b a b c r r_a r_b r_c := sorry

end right_triangle_iff_sum_excircles_a_right_triangle_iff_sum_excircles_b_l1938_193855


namespace books_for_sale_l1938_193809

theorem books_for_sale (initial_books found_books : ℕ) (h1 : initial_books = 33) (h2 : found_books = 26) :
  initial_books + found_books = 59 :=
by
  sorry

end books_for_sale_l1938_193809


namespace find_min_value_l1938_193800

theorem find_min_value (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : 1 / a + 1 / b = 1) : a + 2 * b ≥ 3 + 2 * Real.sqrt 2 :=
by
  sorry

end find_min_value_l1938_193800


namespace bigger_part_is_45_l1938_193888

variable (x y : ℕ)

theorem bigger_part_is_45
  (h1 : x + y = 60)
  (h2 : 10 * x + 22 * y = 780) :
  max x y = 45 := by
  sorry

end bigger_part_is_45_l1938_193888


namespace net_gain_mr_A_l1938_193866

def home_worth : ℝ := 12000
def sale1 : ℝ := home_worth * 1.2
def sale2 : ℝ := sale1 * 0.85
def sale3 : ℝ := sale2 * 1.1

theorem net_gain_mr_A : sale1 - sale2 + sale3 = 3384 := by
  sorry -- Proof will be provided here

end net_gain_mr_A_l1938_193866


namespace gold_beads_cannot_be_determined_without_cost_per_bead_l1938_193879

-- Carly's bead conditions
def purple_rows : ℕ := 50
def purple_beads_per_row : ℕ := 20
def blue_rows : ℕ := 40
def blue_beads_per_row : ℕ := 18
def total_cost : ℝ := 180

-- The calculation of total purple and blue beads
def purple_beads : ℕ := purple_rows * purple_beads_per_row
def blue_beads : ℕ := blue_rows * blue_beads_per_row
def total_beads_without_gold : ℕ := purple_beads + blue_beads

-- Given the lack of cost per bead, the number of gold beads cannot be determined
theorem gold_beads_cannot_be_determined_without_cost_per_bead :
  ¬ (∃ cost_per_bead : ℝ, ∃ gold_beads : ℕ, (purple_beads + blue_beads + gold_beads) * cost_per_bead = total_cost) :=
sorry

end gold_beads_cannot_be_determined_without_cost_per_bead_l1938_193879


namespace compute_exponent_problem_l1938_193802

noncomputable def exponent_problem : ℤ :=
  3 * (3^4) - (9^60) / (9^57)

theorem compute_exponent_problem : exponent_problem = -486 := by
  sorry

end compute_exponent_problem_l1938_193802


namespace log_tangent_ratio_l1938_193804

open Real

theorem log_tangent_ratio (α β : ℝ) 
  (h1 : sin (α + β) = 1 / 2) 
  (h2 : sin (α - β) = 1 / 3) : 
  log 5 * (tan α / tan β) = 1 := 
sorry

end log_tangent_ratio_l1938_193804


namespace fair_collection_l1938_193877

theorem fair_collection 
  (children : ℕ) (fee_child : ℝ) (adults : ℕ) (fee_adult : ℝ) 
  (total_people : ℕ) (count_children : ℕ) (count_adults : ℕ)
  (total_collected: ℝ) :
  children = 700 →
  fee_child = 1.5 →
  adults = 1500 →
  fee_adult = 4.0 →
  total_people = children + adults →
  count_children = 700 →
  count_adults = 1500 →
  total_collected = (count_children * fee_child) + (count_adults * fee_adult) →
  total_collected = 7050 :=
by
  intros
  sorry

end fair_collection_l1938_193877


namespace problem1_problem2_l1938_193858

theorem problem1 (a b : ℝ) : (-(2 : ℝ) * a ^ 2 * b) ^ 3 / (-(2 * a * b)) * (1 / 3 * a ^ 2 * b ^ 3) = (4 / 3) * a ^ 7 * b ^ 5 :=
  by
  sorry

theorem problem2 (x : ℝ) : (27 * x ^ 3 + 18 * x ^ 2 - 3 * x) / -3 * x = -9 * x ^ 2 - 6 * x + 1 :=
  by
  sorry

end problem1_problem2_l1938_193858


namespace ellipse_focal_length_l1938_193868

theorem ellipse_focal_length :
  let a_squared := 20
    let b_squared := 11
    let c := Real.sqrt (a_squared - b_squared)
    let focal_length := 2 * c
  11 * x^2 + 20 * y^2 = 220 →
  focal_length = 6 :=
by
  sorry

end ellipse_focal_length_l1938_193868


namespace find_x_l1938_193803

theorem find_x (x : ℝ) (h : 9 / (x + 4) = 1) : x = 5 :=
sorry

end find_x_l1938_193803


namespace multiply_fractions_l1938_193862

theorem multiply_fractions :
  (1 / 3) * (4 / 7) * (9 / 13) * (2 / 5) = 72 / 1365 :=
by sorry

end multiply_fractions_l1938_193862


namespace correct_optionD_l1938_193895

def operationA (a : ℝ) : Prop := a^3 + 3 * a^3 = 5 * a^6
def operationB (a : ℝ) : Prop := 7 * a^2 * a^3 = 7 * a^6
def operationC (a : ℝ) : Prop := (-2 * a^3)^2 = 4 * a^5
def operationD (a : ℝ) : Prop := a^8 / a^2 = a^6

theorem correct_optionD (a : ℝ) : ¬ operationA a ∧ ¬ operationB a ∧ ¬ operationC a ∧ operationD a :=
by
  unfold operationA operationB operationC operationD
  sorry

end correct_optionD_l1938_193895


namespace exam_standard_deviation_l1938_193890

-- Define the mean score
def mean_score : ℝ := 74

-- Define the standard deviation and conditions
def standard_deviation (σ : ℝ) : Prop :=
  mean_score - 2 * σ = 58

-- Define the condition to prove
def standard_deviation_above_mean (σ : ℝ) : Prop :=
  (98 - mean_score) / σ = 3

theorem exam_standard_deviation {σ : ℝ} (h1 : standard_deviation σ) : standard_deviation_above_mean σ :=
by
  -- proof is omitted
  sorry

end exam_standard_deviation_l1938_193890


namespace cost_of_each_top_l1938_193865

theorem cost_of_each_top
  (total_spent : ℝ)
  (num_shorts : ℕ)
  (price_per_short : ℝ)
  (num_shoes : ℕ)
  (price_per_shoe : ℝ)
  (num_tops : ℕ)
  (total_cost_shorts : ℝ)
  (total_cost_shoes : ℝ)
  (amount_spent_on_tops : ℝ)
  (cost_per_top : ℝ) :
  total_spent = 75 →
  num_shorts = 5 →
  price_per_short = 7 →
  num_shoes = 2 →
  price_per_shoe = 10 →
  num_tops = 4 →
  total_cost_shorts = num_shorts * price_per_short →
  total_cost_shoes = num_shoes * price_per_shoe →
  amount_spent_on_tops = total_spent - (total_cost_shorts + total_cost_shoes) →
  cost_per_top = amount_spent_on_tops / num_tops →
  cost_per_top = 5 :=
by
  sorry

end cost_of_each_top_l1938_193865


namespace planted_fraction_correct_l1938_193806

noncomputable def field_planted_fraction (leg1 leg2 : ℕ) (square_distance : ℕ) : ℚ :=
  let hypotenuse := Real.sqrt (leg1^2 + leg2^2)
  let total_area := (leg1 * leg2) / 2
  let square_side := square_distance
  let square_area := square_side^2
  let planted_area := total_area - square_area
  planted_area / total_area

theorem planted_fraction_correct :
  field_planted_fraction 5 12 4 = 367 / 375 :=
by
  sorry

end planted_fraction_correct_l1938_193806


namespace probability_of_detecting_non_conforming_l1938_193845

noncomputable def prob_detecting_non_conforming (total_cans non_conforming_cans selected_cans : ℕ) : ℚ :=
  let total_outcomes := Nat.choose total_cans selected_cans
  let outcomes_with_one_non_conforming := Nat.choose non_conforming_cans 1 * Nat.choose (total_cans - non_conforming_cans) (selected_cans - 1)
  let outcomes_with_two_non_conforming := Nat.choose non_conforming_cans 2
  (outcomes_with_one_non_conforming + outcomes_with_two_non_conforming) / total_outcomes

theorem probability_of_detecting_non_conforming :
  prob_detecting_non_conforming 5 2 2 = 7 / 10 :=
by
  -- Placeholder for the actual proof
  sorry

end probability_of_detecting_non_conforming_l1938_193845


namespace solve_for_x_l1938_193876

theorem solve_for_x (x : ℚ) (h : 3 / x - 3 / x / (9 / x) = 0.5) : x = 6 / 5 :=
sorry

end solve_for_x_l1938_193876


namespace exactly_one_true_l1938_193891

-- Given conditions
def p (x : ℝ) : Prop := (x^2 - 3 * x + 2 ≠ 0) → (x ≠ 2)

-- Define the contrapositive of p
def contrapositive_p (x : ℝ) : Prop := (x = 2) → (x^2 - 3 * x + 2 = 0)

-- Define the converse of p
def converse_p (x : ℝ) : Prop := (x ≠ 2) → (x^2 - 3 * x + 2 ≠ 0)

-- Define the inverse of p
def inverse_p (x : ℝ) : Prop := (x = 2 → x^2 - 3 * x + 2 = 0)

-- Formalize the problem: Prove that exactly one of the converse, inverse, and contrapositive of p is true.
theorem exactly_one_true :
  (∀ x : ℝ, p x) →
  ((∃ x : ℝ, contrapositive_p x) ∧ ¬(∀ x : ℝ, converse_p x) ∧ ¬(∀ x : ℝ, inverse_p x) ∨
   ¬(∃ x : ℝ, contrapositive_p x) ∧ (∀ x : ℝ, converse_p x) ∧ ¬(∀ x : ℝ, inverse_p x) ∨
   ¬(∃ x : ℝ, contrapositive_p x) ∧ ¬(∀ x : ℝ, converse_p x) ∧ (∀ x : ℝ, inverse_p x)) :=
sorry

end exactly_one_true_l1938_193891


namespace correct_reaction_equation_l1938_193861

noncomputable def reaction_equation (vA vB vC : ℝ) : Prop :=
  vB = 3 * vA ∧ 3 * vC = 2 * vB

theorem correct_reaction_equation (vA vB vC : ℝ) (h : reaction_equation vA vB vC) :
  ∃ (α β γ : ℕ), α = 1 ∧ β = 3 ∧ γ = 2 :=
sorry

end correct_reaction_equation_l1938_193861


namespace no_six_digit_starting_with_five_12_digit_square_six_digit_starting_with_one_12_digit_square_smallest_k_for_n_digit_number_square_l1938_193811

-- Part (a)
theorem no_six_digit_starting_with_five_12_digit_square : ∀ (x y : ℕ), (5 * 10^5 ≤ x) → (x < 6 * 10^5) → (10^5 ≤ y) → (y < 10^6) → ¬∃ z : ℕ, (10^11 ≤ z) ∧ (z < 10^12) ∧ x * 10^6 + y = z^2 := sorry

-- Part (b)
theorem six_digit_starting_with_one_12_digit_square : ∀ (x y : ℕ), (10^5 ≤ x) → (x < 2 * 10^5) → (10^5 ≤ y) → (y < 10^6) → ∃ z : ℕ, (10^11 ≤ z) ∧ (z < 2 * 10^11) ∧ x * 10^6 + y = z^2 := sorry

-- Part (c)
theorem smallest_k_for_n_digit_number_square : ∀ (n : ℕ), ∃ (k : ℕ), k = n + 1 ∧ ∀ (x : ℕ), (10^(n-1) ≤ x) → (x < 10^n) → ∃ y : ℕ, (10^(n + k - 1) ≤ x * 10^k + y) ∧ (x * 10^k + y) < 10^(n + k) ∧ ∃ z : ℕ, x * 10^k + y = z^2 := sorry

end no_six_digit_starting_with_five_12_digit_square_six_digit_starting_with_one_12_digit_square_smallest_k_for_n_digit_number_square_l1938_193811


namespace smallest_d_for_divisibility_by_3_l1938_193823

def sum_of_digits (d : ℕ) : ℕ := 5 + 4 + 7 + d + 0 + 6

theorem smallest_d_for_divisibility_by_3 (d : ℕ) :
  (sum_of_digits 2) % 3 = 0 ∧ ∀ k, k < 2 → sum_of_digits k % 3 ≠ 0 := 
sorry

end smallest_d_for_divisibility_by_3_l1938_193823


namespace number_of_shirts_that_weigh_1_pound_l1938_193808

/-- 
Jon's laundry machine can do 5 pounds of laundry at a time. 
Some number of shirts weigh 1 pound. 
2 pairs of pants weigh 1 pound. 
Jon needs to wash 20 shirts and 20 pants. 
Jon has to do 3 loads of laundry. 
-/
theorem number_of_shirts_that_weigh_1_pound
    (machine_capacity : ℕ)
    (num_shirts : ℕ)
    (shirts_per_pound : ℕ)
    (pairs_of_pants_per_pound : ℕ)
    (num_pants : ℕ)
    (loads : ℕ)
    (weight_per_load : ℕ)
    (total_pants_weight : ℕ)
    (total_weight : ℕ)
    (shirt_weight_per_pound : ℕ)
    (shirts_weighing_one_pound : ℕ) :
  machine_capacity = 5 → 
  num_shirts = 20 → 
  pairs_of_pants_per_pound = 2 →
  num_pants = 20 →
  loads = 3 →
  weight_per_load = 5 → 
  total_pants_weight = (num_pants / pairs_of_pants_per_pound) →
  total_weight = (loads * weight_per_load) →
  shirts_weighing_one_pound = (total_weight - total_pants_weight) / num_shirts → 
  shirts_weighing_one_pound = 4 :=
by sorry

end number_of_shirts_that_weigh_1_pound_l1938_193808


namespace solution_set_product_positive_l1938_193846

variable {R : Type*} [LinearOrderedField R]

def is_odd (f : R → R) : Prop := ∀ x : R, f (-x) = -f (x)

variable (f g : R → R)

noncomputable def solution_set_positive_f : Set R := { x | 4 < x ∧ x < 10 }
noncomputable def solution_set_positive_g : Set R := { x | 2 < x ∧ x < 5 }

theorem solution_set_product_positive :
  is_odd f →
  is_odd g →
  (∀ x, f x > 0 ↔ x ∈ solution_set_positive_f) →
  (∀ x, g x > 0 ↔ x ∈ solution_set_positive_g) →
  { x | f x * g x > 0 } = { x | (4 < x ∧ x < 5) ∨ (-5 < x ∧ x < -4) } :=
by
  sorry

end solution_set_product_positive_l1938_193846


namespace sum_of_arithmetic_seq_minimum_value_n_equals_5_l1938_193817

variable {a : ℕ → ℝ} -- Define a sequence of real numbers
variable {S : ℕ → ℝ} -- Define the sum function for the sequence

-- Assume conditions
axiom a3_a8_neg : a 3 + a 8 < 0
axiom S11_pos : S 11 > 0

-- Prove the minimum value of S_n occurs at n = 5
theorem sum_of_arithmetic_seq_minimum_value_n_equals_5 :
  ∃ n, (∀ m < 5, S m ≥ S n) ∧ (∀ m > 5, S m > S n) ∧ n = 5 :=
sorry

end sum_of_arithmetic_seq_minimum_value_n_equals_5_l1938_193817


namespace alcohol_mixture_l1938_193819

variable {a b c d : ℝ} (ha : a ≠ d) (hbc : d ≠ c)

theorem alcohol_mixture (hcd : a ≥ d ∧ d ≥ c ∨ a ≤ d ∧ d ≤ c) :
  x = b * (d - c) / (a - d) :=
by 
  sorry

end alcohol_mixture_l1938_193819


namespace min_a_for_inequality_l1938_193881

theorem min_a_for_inequality (a : ℝ) : 
  (∀ x : ℝ, (0 < x ∧ x ≤ 1/2) → (x^2 + a*x + 1 ≥ 0)) ↔ a ≥ -5/2 :=
by
  sorry

end min_a_for_inequality_l1938_193881


namespace arithmetic_mean_eq_one_l1938_193833

theorem arithmetic_mean_eq_one 
  (x a b : ℝ) 
  (hx : x ≠ 0) 
  (hb : b ≠ 0) : 
  (1 / 2 * ((x + a + b) / x + (x - a - b) / x)) = 1 := by
  sorry

end arithmetic_mean_eq_one_l1938_193833


namespace math_proof_equiv_l1938_193825

def A := 5
def B := 3
def C := 2
def D := 0
def E := 0
def F := 1
def G := 0

theorem math_proof_equiv : (A * 1000 + B * 100 + C * 10 + D) + (E * 100 + F * 10 + G) = 5300 :=
by
  sorry

end math_proof_equiv_l1938_193825


namespace items_count_l1938_193863

variable (N : ℕ)

-- Conditions
def item_price : ℕ := 50
def discount_rate : ℕ := 80
def sell_percentage : ℕ := 90
def creditors_owed : ℕ := 15000
def money_left : ℕ := 3000

-- Definitions based on the conditions
def sale_price : ℕ := (item_price * (100 - discount_rate)) / 100
def money_before_paying_creditors : ℕ := money_left + creditors_owed
def total_revenue (N : ℕ) : ℕ := (sell_percentage * N * sale_price) / 100

-- Problem statement
theorem items_count : total_revenue N = money_before_paying_creditors → N = 2000 := by
  intros h
  sorry

end items_count_l1938_193863


namespace clothing_order_equation_l1938_193828

open Real

-- Definitions and conditions
def total_pieces : ℕ := 720
def initial_rate : ℕ := 48
def days_earlier : ℕ := 5

-- Statement that we need to prove
theorem clothing_order_equation (x : ℕ) :
    (720 / 48 : ℝ) - (720 / (x + 48) : ℝ) = 5 := 
sorry

end clothing_order_equation_l1938_193828


namespace pet_store_initial_puppies_l1938_193831

theorem pet_store_initial_puppies
  (sold: ℕ) (cages: ℕ) (puppies_per_cage: ℕ)
  (remaining_puppies: ℕ)
  (h1: sold = 30)
  (h2: cages = 6)
  (h3: puppies_per_cage = 8)
  (h4: remaining_puppies = cages * puppies_per_cage):
  (sold + remaining_puppies) = 78 :=
by
  sorry

end pet_store_initial_puppies_l1938_193831


namespace sector_angle_l1938_193875

theorem sector_angle (r θ : ℝ) 
  (h1 : r * θ + 2 * r = 6) 
  (h2 : 1/2 * r^2 * θ = 2) : 
  θ = 1 ∨ θ = 4 :=
by 
  sorry

end sector_angle_l1938_193875


namespace find_m_n_diff_l1938_193889

theorem find_m_n_diff (a : ℝ) (n m: ℝ) (h_a_pos : 0 < a) (h_a_ne_one : a ≠ 1)
  (h_pass : a^(2 * m - 6) + n = 2) :
  m - n = 2 :=
sorry

end find_m_n_diff_l1938_193889


namespace find_m_l1938_193822

theorem find_m (x y m : ℤ) 
  (h1 : x + 2 * y = 5 * m) 
  (h2 : x - 2 * y = 9 * m) 
  (h3 : 3 * x + 2 * y = 19) : 
  m = 1 := 
by 
  sorry

end find_m_l1938_193822


namespace combined_score_is_210_l1938_193801

theorem combined_score_is_210 :
  ∀ (total_questions : ℕ) (marks_per_question : ℕ) (jose_wrong : ℕ) 
    (meghan_less_than_jose : ℕ) (jose_more_than_alisson : ℕ) (jose_total : ℕ),
  total_questions = 50 →
  marks_per_question = 2 →
  jose_wrong = 5 →
  meghan_less_than_jose = 20 →
  jose_more_than_alisson = 40 →
  jose_total = total_questions * marks_per_question - (jose_wrong * marks_per_question) →
  (jose_total - meghan_less_than_jose) + jose_total + (jose_total - jose_more_than_alisson) = 210 :=
by
  intros total_questions marks_per_question jose_wrong meghan_less_than_jose jose_more_than_alisson jose_total
  intros h1 h2 h3 h4 h5 h6
  sorry

end combined_score_is_210_l1938_193801


namespace complement_intersection_l1938_193892

open Set

-- Define the universal set U
def U : Set ℕ := {x | 0 < x ∧ x < 7}

-- Define Set A
def A : Set ℕ := {2, 3, 5}

-- Define Set B
def B : Set ℕ := {1, 4}

-- Define the complement of A in U
def CU_A : Set ℕ := U \ A

-- Define the complement of B in U
def CU_B : Set ℕ := U \ B

-- Define the intersection of CU_A and CU_B
def intersection_CU_A_CU_B : Set ℕ := CU_A ∩ CU_B

-- The theorem statement
theorem complement_intersection :
  intersection_CU_A_CU_B = {6} := by
  sorry

end complement_intersection_l1938_193892


namespace f_eq_g_l1938_193887

noncomputable def f : ℕ → ℕ := sorry
noncomputable def g : ℕ → ℕ := sorry

variable (f_onto : ∀ m : ℕ, ∃ n : ℕ, f n = m)
variable (g_one_one : ∀ m n : ℕ, g m = g n → m = n)
variable (f_ge_g : ∀ n : ℕ, f n ≥ g n)

theorem f_eq_g : f = g :=
sorry

end f_eq_g_l1938_193887


namespace wholesale_price_of_pen_l1938_193898

-- Definitions and conditions
def wholesale_price (P : ℝ) : Prop :=
  (5 - P = 10 - 3 * P)

-- Statement of the proof problem
theorem wholesale_price_of_pen : ∃ P : ℝ, wholesale_price P ∧ P = 2.5 :=
by {
  sorry
}

end wholesale_price_of_pen_l1938_193898


namespace volleyball_team_arrangements_l1938_193852

theorem volleyball_team_arrangements (n : ℕ) (n_pos : 0 < n) :
  ∃ arrangements : ℕ, arrangements = 2^n * (Nat.factorial n)^2 :=
sorry

end volleyball_team_arrangements_l1938_193852


namespace cylinder_radius_l1938_193854

theorem cylinder_radius
  (diameter_c : ℝ) (altitude_c : ℝ) (height_relation : ℝ → ℝ)
  (same_axis : Bool) (radius_cylinder : ℝ → ℝ)
  (h1 : diameter_c = 14)
  (h2 : altitude_c = 20)
  (h3 : ∀ r, height_relation r = 3 * r)
  (h4 : same_axis = true)
  (h5 : ∀ r, radius_cylinder r = r) :
  ∃ r, r = 140 / 41 :=
by {
  sorry
}

end cylinder_radius_l1938_193854


namespace quadratic_to_binomial_square_l1938_193853

theorem quadratic_to_binomial_square (m : ℝ) : 
  (∃ c : ℝ, (x : ℝ) → x^2 - 12 * x + m = (x + c)^2) ↔ m = 36 := 
sorry

end quadratic_to_binomial_square_l1938_193853


namespace exponential_inequality_l1938_193821

theorem exponential_inequality (n : ℕ) (h : n ≥ 5) : 2^n > n^2 + 1 :=
sorry

end exponential_inequality_l1938_193821


namespace nickel_ate_4_chocolates_l1938_193873

theorem nickel_ate_4_chocolates (R N : ℕ) (h1 : R = 13) (h2 : R = N + 9) : N = 4 :=
by
  sorry

end nickel_ate_4_chocolates_l1938_193873


namespace jason_grass_cutting_time_l1938_193874

def total_minutes (hours : ℕ) : ℕ := hours * 60
def minutes_per_yard : ℕ := 30
def total_yards_per_weekend : ℕ := 8 * 2
def total_minutes_per_weekend : ℕ := minutes_per_yard * total_yards_per_weekend
def convert_minutes_to_hours (minutes : ℕ) : ℕ := minutes / 60

theorem jason_grass_cutting_time : 
  convert_minutes_to_hours total_minutes_per_weekend = 8 := by
  sorry

end jason_grass_cutting_time_l1938_193874


namespace grazing_area_of_goat_l1938_193810

/-- 
Consider a circular park with a diameter of 50 feet, and a square monument with 10 feet on each side.
Sally ties her goat on one corner of the monument with a 20-foot rope. Calculate the total grazing area
around the monument considering the space limited by the park's boundary.
-/
theorem grazing_area_of_goat : 
  let park_radius := 25
  let monument_side := 10
  let rope_length := 20
  let monument_radius := monument_side / 2 
  let grazing_quarter_circle := (1 / 4) * Real.pi * rope_length^2
  let ungrazable_area := (1 / 4) * Real.pi * monument_radius^2
  grazing_quarter_circle - ungrazable_area = 93.75 * Real.pi :=
by
  sorry

end grazing_area_of_goat_l1938_193810


namespace ratio_M_N_l1938_193896

theorem ratio_M_N (M Q P R N : ℝ) 
(h1 : M = 0.40 * Q) 
(h2 : Q = 0.25 * P) 
(h3 : R = 0.60 * P) 
(h4 : N = 0.75 * R) : 
  M / N = 2 / 9 := 
by
  sorry

end ratio_M_N_l1938_193896


namespace probability_even_sum_l1938_193880

-- Defining the probabilities for the first wheel
def P_even_1 : ℚ := 2/3
def P_odd_1 : ℚ := 1/3

-- Defining the probabilities for the second wheel
def P_even_2 : ℚ := 1/2
def P_odd_2 : ℚ := 1/2

-- Prove that the probability that the sum of the two selected numbers is even is 1/2
theorem probability_even_sum : 
  P_even_1 * P_even_2 + P_odd_1 * P_odd_2 = 1/2 :=
by
  sorry

end probability_even_sum_l1938_193880


namespace sequence_a5_l1938_193869

/-- In the sequence {a_n}, with a_1 = 1, a_2 = 2, and a_(n+2) = 2 * a_(n+1) + a_n, prove that a_5 = 29. -/
theorem sequence_a5 (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : a 2 = 2) (h_rec : ∀ n, a (n + 2) = 2 * a (n + 1) + a n) :
  a 5 = 29 :=
sorry

end sequence_a5_l1938_193869


namespace number_of_intersections_of_lines_l1938_193824

theorem number_of_intersections_of_lines : 
  let L1 := {p : ℝ × ℝ | 3 * p.1 + 4 * p.2 = 12}
  let L2 := {p : ℝ × ℝ | 5 * p.1 - 2 * p.2 = 10}
  let L3 := {p : ℝ × ℝ | p.1 = 3}
  let L4 := {p : ℝ × ℝ | p.2 = 1}
  ∃ p1 p2 : ℝ × ℝ, p1 ≠ p2 ∧ p1 ∈ L1 ∧ p1 ∈ L2 ∧ p2 ∈ L3 ∧ p2 ∈ L4 :=
by
  sorry

end number_of_intersections_of_lines_l1938_193824


namespace relationship_xy_l1938_193867

variable (x y : ℝ)

theorem relationship_xy (h₁ : x - y > x + 2) (h₂ : x + y + 3 < y - 1) : x < -4 ∧ y < -2 := 
by sorry

end relationship_xy_l1938_193867


namespace distance_to_moscow_at_4PM_l1938_193829

noncomputable def exact_distance_at_4PM (d12: ℝ) (d13: ℝ) (d15: ℝ) : ℝ :=
  d15 - 12

theorem distance_to_moscow_at_4PM  (h12 : 81.5 ≤ 82 ∧ 82 ≤ 82.5)
                                  (h13 : 70.5 ≤ 71 ∧ 71 ≤ 71.5)
                                  (h15 : 45.5 ≤ 46 ∧ 46 ≤ 46.5) :
  exact_distance_at_4PM 82 71 46 = 34 :=
by
  sorry

end distance_to_moscow_at_4PM_l1938_193829


namespace max_value_proof_l1938_193856

noncomputable def maximum_value (x y z : ℝ) : ℝ :=
  x + y^3 + z^4

theorem max_value_proof
  (x y z : ℝ)
  (hx : 0 ≤ x)
  (hy : 0 ≤ y)
  (hz : 0 ≤ z)
  (h1 : x + y + z = 1)
  (h2 : x^2 + y^2 + z^2 = 1) :
  maximum_value x y z ≤ 1 :=
sorry

end max_value_proof_l1938_193856


namespace relationship_y1_y2_y3_l1938_193842

variables {m y_1 y_2 y_3 : ℝ}

theorem relationship_y1_y2_y3 :
  (∃ (m : ℝ), (y_1 = (-1)^2 - 2*(-1) + m) ∧ (y_2 = 2^2 - 2*2 + m) ∧ (y_3 = 3^2 - 2*3 + m)) →
  y_2 < y_1 ∧ y_1 = y_3 :=
by
  sorry

end relationship_y1_y2_y3_l1938_193842


namespace prove_lesser_fraction_l1938_193851

noncomputable def lesser_fraction (x y : ℚ) : Prop :=
  x + y = 8/9 ∧ x * y = 1/8 ∧ min x y = 7/40

theorem prove_lesser_fraction :
  ∃ x y : ℚ, lesser_fraction x y :=
sorry

end prove_lesser_fraction_l1938_193851


namespace quadratic_eq_k_value_l1938_193848

theorem quadratic_eq_k_value (k : ℤ) : (∀ x : ℝ, (k - 1) * x ^ (|k| + 1) - x + 5 = 0 → (k - 1) ≠ 0 ∧ |k| + 1 = 2) -> k = -1 :=
by
  sorry

end quadratic_eq_k_value_l1938_193848


namespace part_1_select_B_prob_part_2_select_BC_prob_l1938_193815

-- Definitions for the four students
inductive Student
| A
| B
| C
| D

open Student

-- Definition for calculating probability
def probability (favorable total : Nat) : Rat :=
  favorable / total

-- Part (1)
theorem part_1_select_B_prob : probability 1 4 = 1 / 4 :=
  sorry

-- Part (2)
theorem part_2_select_BC_prob : probability 2 12 = 1 / 6 :=
  sorry

end part_1_select_B_prob_part_2_select_BC_prob_l1938_193815


namespace order_of_a_b_c_l1938_193834

noncomputable def a := Real.sqrt 3 - Real.sqrt 2
noncomputable def b := Real.sqrt 6 - Real.sqrt 5
noncomputable def c := Real.sqrt 7 - Real.sqrt 6

theorem order_of_a_b_c : a > b ∧ b > c :=
by
  sorry

end order_of_a_b_c_l1938_193834


namespace value_of_A_l1938_193813

theorem value_of_A {α : Type} [LinearOrderedSemiring α] 
  (L A D E : α) (L_value : L = 15) (LEAD DEAL DELL : α)
  (LEAD_value : LEAD = 50)
  (DEAL_value : DEAL = 55)
  (DELL_value : DELL = 60)
  (LEAD_condition : L + E + A + D = LEAD)
  (DEAL_condition : D + E + A + L = DEAL)
  (DELL_condition : D + E + L + L = DELL) :
  A = 25 :=
by
  sorry

end value_of_A_l1938_193813


namespace amoeba_count_after_two_weeks_l1938_193864

theorem amoeba_count_after_two_weeks :
  let initial_day_count := 1
  let days_double_split := 7
  let days_triple_split := 7
  let end_of_first_phase := initial_day_count * 2 ^ days_double_split
  let final_amoeba_count := end_of_first_phase * 3 ^ days_triple_split
  final_amoeba_count = 279936 :=
by
  sorry

end amoeba_count_after_two_weeks_l1938_193864


namespace square_plot_area_l1938_193812

theorem square_plot_area (s : ℕ) 
  (cost_per_foot : ℕ) 
  (total_cost : ℕ) 
  (H1 : cost_per_foot = 58) 
  (H2 : total_cost = 1624) 
  (H3 : total_cost = 232 * s) : 
  s * s = 49 := 
  by sorry

end square_plot_area_l1938_193812


namespace fraction_simplification_l1938_193860

theorem fraction_simplification : 
  (3 + 9 - 27 + 81 + 243 - 729) / (9 + 27 - 81 + 243 + 729 - 2187) = (1 / 3) := 
sorry

end fraction_simplification_l1938_193860


namespace tangent_slope_of_circle_l1938_193837

theorem tangent_slope_of_circle {x1 y1 x2 y2 : ℝ}
  (hx1 : x1 = 1) (hy1 : y1 = 1) (hx2 : x2 = 6) (hy2 : y2 = 4) :
  ∀ m : ℝ, m = -5 / 3 ↔
    (∃ (r : ℝ), r = (y2 - y1) / (x2 - x1) ∧ m = -1 / r) :=
by
  sorry

end tangent_slope_of_circle_l1938_193837


namespace interest_rate_difference_l1938_193830

-- Definitions for given conditions
def principal : ℝ := 3000
def time : ℝ := 9
def additional_interest : ℝ := 1350

-- The Lean 4 statement for the equivalence
theorem interest_rate_difference 
  (R H : ℝ) 
  (h_interest_formula_original : principal * R * time / 100 = principal * R * time / 100) 
  (h_interest_formula_higher : principal * H * time / 100 = principal * R * time / 100 + additional_interest) 
  : (H - R) = 5 :=
sorry

end interest_rate_difference_l1938_193830


namespace aerith_is_correct_l1938_193818

theorem aerith_is_correct :
  ∀ x : ℝ, x = 1.4 → (x ^ (x ^ x)) < 2 → ∃ y : ℝ, y = x ^ (x ^ x) :=
by
  sorry

end aerith_is_correct_l1938_193818


namespace factorization_correct_l1938_193807

theorem factorization_correct (x y : ℝ) : 
  x * (x - y) - y * (x - y) = (x - y) ^ 2 :=
by 
  sorry

end factorization_correct_l1938_193807


namespace number_of_combinations_with_constraints_l1938_193841

theorem number_of_combinations_with_constraints :
  let n : ℕ := 6
  let k : ℕ := 2
  let total_combinations := Nat.choose n k
  let invalid_combinations := 2
  total_combinations - invalid_combinations = 13 :=
by
  let n : ℕ := 6
  let k : ℕ := 2
  let total_combinations := Nat.choose 6 2
  let invalid_combinations := 2
  show total_combinations - invalid_combinations = 13
  sorry

end number_of_combinations_with_constraints_l1938_193841


namespace num_trombone_players_l1938_193835

def weight_per_trumpet := 5
def weight_per_clarinet := 5
def weight_per_trombone := 10
def weight_per_tuba := 20
def weight_per_drum := 15

def num_trumpets := 6
def num_clarinets := 9
def num_tubas := 3
def num_drummers := 2
def total_weight := 245

theorem num_trombone_players : 
  let weight_trumpets := num_trumpets * weight_per_trumpet
  let weight_clarinets := num_clarinets * weight_per_clarinet
  let weight_tubas := num_tubas * weight_per_tuba
  let weight_drums := num_drummers * weight_per_drum
  let weight_others := weight_trumpets + weight_clarinets + weight_tubas + weight_drums
  let weight_trombones := total_weight - weight_others
  weight_trombones / weight_per_trombone = 8 :=
by
  sorry

end num_trombone_players_l1938_193835


namespace complex_trajectory_is_ellipse_l1938_193897

open Complex

theorem complex_trajectory_is_ellipse (z : ℂ) (h : abs (z - i) + abs (z + i) = 3) : 
  true := 
sorry

end complex_trajectory_is_ellipse_l1938_193897


namespace total_gum_l1938_193894

-- Define the conditions
def original_gum : ℕ := 38
def additional_gum : ℕ := 16

-- Define the statement to be proved
theorem total_gum : original_gum + additional_gum = 54 :=
by
  -- Proof omitted
  sorry

end total_gum_l1938_193894


namespace oranges_given_to_friend_l1938_193870

theorem oranges_given_to_friend (initial_oranges : ℕ) 
  (given_to_brother : ℕ)
  (given_to_friend : ℕ)
  (h1 : initial_oranges = 60)
  (h2 : given_to_brother = (1 / 3 : ℚ) * initial_oranges)
  (h3 : given_to_friend = (1 / 4 : ℚ) * (initial_oranges - given_to_brother)) : 
  given_to_friend = 10 := 
by 
  sorry

end oranges_given_to_friend_l1938_193870


namespace equivalent_single_discount_l1938_193859

theorem equivalent_single_discount (p : ℝ) : 
  let discount1 := 0.15
  let discount2 := 0.25
  let price_after_first_discount := (1 - discount1) * p
  let price_after_second_discount := (1 - discount2) * price_after_first_discount
  let equivalent_single_discount := 1 - price_after_second_discount / p
  equivalent_single_discount = 0.3625 :=
by
  sorry

end equivalent_single_discount_l1938_193859


namespace find_x_l1938_193885

theorem find_x (x : ℝ) (h1 : |x + 7| = 3) (h2 : x^2 + 2*x - 3 = 5) : x = -4 :=
by
  sorry

end find_x_l1938_193885


namespace largest_number_of_gold_coins_l1938_193882

theorem largest_number_of_gold_coins (n : ℕ) (h1 : n % 15 = 4) (h2 : n < 150) : n ≤ 139 :=
by {
  -- This is where the proof would go.
  sorry
}

end largest_number_of_gold_coins_l1938_193882


namespace smallest_even_sum_l1938_193838

theorem smallest_even_sum :
  ∃ (a b c : Int), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a ∈ ({8, -4, 3, 27, 10} : Set Int) ∧ b ∈ ({8, -4, 3, 27, 10} : Set Int) ∧ c ∈ ({8, -4, 3, 27, 10} : Set Int) ∧ (a + b + c) % 2 = 0 ∧ (a + b + c) = 14 := sorry

end smallest_even_sum_l1938_193838


namespace fraction_inequality_solution_l1938_193844

theorem fraction_inequality_solution (x : ℝ) (h : 0 ≤ x ∧ x ≤ 3) :
  3 * x + 2 < 2 * (5 * x - 4) → (10 / 7) < x ∧ x ≤ 3 :=
by
  sorry

end fraction_inequality_solution_l1938_193844


namespace opposite_sides_range_a_l1938_193840

theorem opposite_sides_range_a (a: ℝ) :
  ((1 - 2 * a + 1) * (a + 4 + 1) < 0) ↔ (a < -5 ∨ a > 1) :=
by
  sorry

end opposite_sides_range_a_l1938_193840


namespace grape_juice_percentage_l1938_193836

theorem grape_juice_percentage
  (original_mixture : ℝ)
  (percent_grape_juice : ℝ)
  (added_grape_juice : ℝ)
  (h1 : original_mixture = 50)
  (h2 : percent_grape_juice = 0.10)
  (h3 : added_grape_juice = 10)
  : (percent_grape_juice * original_mixture + added_grape_juice) / (original_mixture + added_grape_juice) * 100 = 25 :=
by
  sorry

end grape_juice_percentage_l1938_193836


namespace acai_berry_cost_correct_l1938_193850

def cost_superfruit_per_litre : ℝ := 1399.45
def cost_mixed_fruit_per_litre : ℝ := 262.85
def litres_mixed_fruit : ℝ := 36
def litres_acai_berry : ℝ := 24
def total_litres : ℝ := litres_mixed_fruit + litres_acai_berry
def expected_cost_acai_per_litre : ℝ := 3104.77

theorem acai_berry_cost_correct :
  cost_superfruit_per_litre * total_litres -
  cost_mixed_fruit_per_litre * litres_mixed_fruit = 
  expected_cost_acai_per_litre * litres_acai_berry :=
by sorry

end acai_berry_cost_correct_l1938_193850


namespace g_nine_l1938_193849

variable (g : ℝ → ℝ)

theorem g_nine : (∀ x y : ℝ, g (x + y) = g x * g y) → g 3 = 4 → g 9 = 64 :=
by intros h1 h2; sorry

end g_nine_l1938_193849


namespace evaluate_at_3_l1938_193893

def f (x : ℝ) : ℝ := 9 * x^4 + 7 * x^3 - 5 * x^2 + 3 * x - 6

theorem evaluate_at_3 : f 3 = 876 := by
  sorry

end evaluate_at_3_l1938_193893


namespace classify_event_l1938_193814

-- Define the conditions of the problem
def involves_variables_and_uncertainties (event: String) : Prop := 
  event = "Turning on the TV and broadcasting 'Chinese Intangible Cultural Heritage'"

-- Define the type of event as a string
def event_type : String := "random"

-- The theorem to prove the classification of the event
theorem classify_event : involves_variables_and_uncertainties "Turning on the TV and broadcasting 'Chinese Intangible Cultural Heritage'" →
  event_type = "random" :=
by
  intro h
  -- Proof is skipped
  sorry

end classify_event_l1938_193814


namespace necessary_condition_for_inequality_l1938_193843

theorem necessary_condition_for_inequality (a b : ℝ) (h : a * b > 0) : 
  (a ≠ b) → (a ≠ 0) → (b ≠ 0) → ((b / a) + (a / b) > 2) :=
by
  sorry

end necessary_condition_for_inequality_l1938_193843


namespace find_ratio_b_a_l1938_193826

theorem find_ratio_b_a (a b : ℝ) 
  (h : ∀ x : ℝ, (2 * a - b) * x + (a + b) > 0 ↔ x > -3) : 
  b / a = 5 / 4 :=
sorry

end find_ratio_b_a_l1938_193826


namespace students_pass_both_subjects_l1938_193805

theorem students_pass_both_subjects
  (F_H F_E F_HE : ℝ)
  (h1 : F_H = 0.25)
  (h2 : F_E = 0.48)
  (h3 : F_HE = 0.27) :
  (100 - (F_H + F_E - F_HE) * 100) = 54 :=
by
  sorry

end students_pass_both_subjects_l1938_193805


namespace min_value_exists_l1938_193857

noncomputable def point_on_circle (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 2)^2 = 9 ∧ y ≥ 2

theorem min_value_exists : ∃ x y : ℝ, point_on_circle x y ∧ x + Real.sqrt 3 * y = 2 * Real.sqrt 3 - 2 := 
sorry

end min_value_exists_l1938_193857
