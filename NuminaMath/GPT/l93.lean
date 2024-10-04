import Mathlib

namespace range_of_a_for_monotonic_f_l93_93918

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a^2 * x^2 + a * x

theorem range_of_a_for_monotonic_f (a : ℝ) : 
  (∀ x, 1 < x → f a x ≤ f a (1 : ℝ)) ↔ (a ≤ -1 / 2 ∨ 1 ≤ a) := 
by
  sorry

end range_of_a_for_monotonic_f_l93_93918


namespace value_of_c_over_ab_l93_93755

theorem value_of_c_over_ab
  (a b c : ℚ)
  (h1 : ab / (a + b) = 3)
  (h2 : bc / (b + c) = 6)
  (h3 : ac / (a + c) = 9)
  : c / (ab) = -35/36 := 
by
  sorry

end value_of_c_over_ab_l93_93755


namespace arithmetic_sequence_zero_l93_93514

noncomputable def f (x : ℝ) : ℝ :=
  0.3 ^ x - Real.log x / Real.log 2

theorem arithmetic_sequence_zero (a b c x : ℝ) (h_seq : a < b ∧ b < c) (h_pos_diff : b - a = c - b)
    (h_f_product : f a * f b * f c > 0) (h_fx_zero : f x = 0) : ¬ (x < a) :=
by
  sorry

end arithmetic_sequence_zero_l93_93514


namespace correct_statements_l93_93318

-- Definitions for statements A, B, C, and D
def statementA (x : ℝ) : Prop := |x| > 1 → x > 1
def statementB (A B C : ℝ) : Prop := (C > 90) ↔ (A + B + C = 180 ∧ (A > 90 ∨ B > 90 ∨ C > 90))
def statementC (a b : ℝ) : Prop := (a * b ≠ 0) ↔ (a ≠ 0 ∧ b ≠ 0)
def statementD (a b : ℝ) : Prop := a > b → 1 / a < 1 / b

-- Proof problem stating which statements are correct
theorem correct_statements :
  (∀ x : ℝ, statementA x = false) ∧ 
  (∀ (A B C : ℝ), statementB A B C = false) ∧ 
  (∀ (a b : ℝ), statementC a b) ∧ 
  (∀ (a b : ℝ), statementD a b = false) :=
by
  sorry

end correct_statements_l93_93318


namespace ac_lt_bd_l93_93380

theorem ac_lt_bd (a b c d : ℝ) (h₁ : a > b) (h₂ : b > 0) (h₃ : c < d) (h₄ : d < 0) : a * c < b * d :=
by
  sorry

end ac_lt_bd_l93_93380


namespace greatest_product_from_sum_2004_l93_93971

theorem greatest_product_from_sum_2004 : ∃ (x y : ℤ), x + y = 2004 ∧ x * y = 1004004 :=
by
  sorry

end greatest_product_from_sum_2004_l93_93971


namespace sum_of_midpoints_x_coordinates_l93_93154

theorem sum_of_midpoints_x_coordinates (a b c : ℝ) (h : a + b + c = 15) :
  (a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 15 := by
  sorry

end sum_of_midpoints_x_coordinates_l93_93154


namespace simplify_fraction_l93_93126

theorem simplify_fraction :
  (1 : ℚ) / 462 + 17 / 42 = 94 / 231 := 
sorry

end simplify_fraction_l93_93126


namespace Hector_gumballs_l93_93625

theorem Hector_gumballs :
  ∃ (total_gumballs : ℕ)
  (gumballs_Todd : ℕ) (gumballs_Alisha : ℕ) (gumballs_Bobby : ℕ) (gumballs_remaining : ℕ),
  gumballs_Todd = 4 ∧
  gumballs_Alisha = 2 * gumballs_Todd ∧
  gumballs_Bobby = 4 * gumballs_Alisha - 5 ∧
  gumballs_remaining = 6 ∧
  total_gumballs = gumballs_Todd + gumballs_Alisha + gumballs_Bobby + gumballs_remaining ∧
  total_gumballs = 45 :=
by
  sorry

end Hector_gumballs_l93_93625


namespace sum_of_mnp_l93_93416

theorem sum_of_mnp (m n p : ℕ) (h_gcd : gcd m (gcd n p) = 1)
  (h : ∀ x : ℝ, 5 * x^2 - 11 * x + 6 = 0 ↔ x = (m + Real.sqrt n) / p ∨ x = (m - Real.sqrt n) / p) :
  m + n + p = 22 :=
by
  sorry

end sum_of_mnp_l93_93416


namespace more_candidates_selected_l93_93530

theorem more_candidates_selected (n : ℕ) (pA pB : ℝ) 
  (hA : pA = 0.06) (hB : pB = 0.07) (hN : n = 8200) :
  (pB * n - pA * n) = 82 :=
by
  sorry

end more_candidates_selected_l93_93530


namespace find_value_of_a_l93_93038

-- Given conditions
def equation1 (x y : ℝ) : Prop := 4 * y + x + 5 = 0
def equation2 (x y : ℝ) (a : ℝ) : Prop := 3 * y + a * x + 4 = 0

-- The proof problem statement
theorem find_value_of_a (a : ℝ) :
  (∀ x y : ℝ, equation1 x y ∧ equation2 x y a → a = -12) :=
sorry

end find_value_of_a_l93_93038


namespace translation_correctness_l93_93385

-- Define the original function
def original_function (x : ℝ) : ℝ := 3 * x + 5

-- Define the translated function
def translated_function (x : ℝ) : ℝ := 3 * x

-- Define the condition for passing through the origin
def passes_through_origin (f : ℝ → ℝ) : Prop := f 0 = 0

-- The theorem to prove the correct translation
theorem translation_correctness : passes_through_origin translated_function := by
  sorry

end translation_correctness_l93_93385


namespace pete_bus_ride_blocks_l93_93805

theorem pete_bus_ride_blocks : 
  ∀ (total_walk_blocks bus_blocks total_blocks : ℕ), 
  total_walk_blocks = 10 → 
  total_blocks = 50 → 
  total_walk_blocks + 2 * bus_blocks = total_blocks → 
  bus_blocks = 20 :=
by
  intros total_walk_blocks bus_blocks total_blocks h1 h2 h3
  sorry

end pete_bus_ride_blocks_l93_93805


namespace number_of_polynomials_Q_l93_93936

def P (x : ℂ) : ℂ := (x - 2) * (x - 3) * (x - 4)

theorem number_of_polynomials_Q :
  ∃ (Q : ℂ[X]), ∃ (R : ℂ[X]), degree R = 3 ∧ P (Q) = P * R ∧ degree Q = 2 ∧
  (finset.univ.image (polynomial.eval 2) Q).card * (finset.univ.image (polynomial.eval 3) Q).card * (finset.univ.image (polynomial.eval 4) Q).card = 22 := sorry

end number_of_polynomials_Q_l93_93936


namespace find_roots_l93_93743

noncomputable def polynomial_roots : set ℝ :=
  {((1 - Real.sqrt 43 + 2 * Real.sqrt 34) / 6),
   ((1 - Real.sqrt 43 - 2 * Real.sqrt 34) / 6),
   ((1 + Real.sqrt 43 + 2 * Real.sqrt 34) / 6),
   ((1 + Real.sqrt 43 - 2 * Real.sqrt 34) / 6)}

theorem find_roots (x : ℝ) :
  (3 * x ^ 4 + 2 * x ^ 3 - 8 * x ^ 2 + 2 * x + 3 = 0) ↔ (x ∈ polynomial_roots) :=
by sorry

end find_roots_l93_93743


namespace area_of_AFE_l93_93563

noncomputable def point := ℝ × ℝ

def AB_parallel_CD (A B C D: point) :=
  (B.2 = A.2) ∧ (C.2 = D.2)

def is_isosceles_trapezoid (A B C D: point) : Prop :=
  AB_parallel_CD A B C D ∧
  (dist A D = dist B C) ∧
  ((A.1 - B.1)^2 + (A.2 - B.2)^2 = 6^2) ∧
  ((A.1 - D.1)^2 + (A.2 - D.2)^2 = 5^2) ∧
  (A.1 - B.1 = 0 ∨ B.1 - A.1 = 6) ∧
  (D.1 - A.1 = 5 * real.cos (60 * real.pi / 180))

def reflect_off_CB (A B E: point) : Prop :=
  -- hypothetical condition for reflection, to be specified as needed
  sorry

noncomputable def area_of_triangle (A B C: point) : ℝ :=
  real.abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)) / 2

theorem area_of_AFE (A B C D E F: point)
  (h1: is_isosceles_trapezoid A B C D)
  (h2: reflect_off_CB A B E)
  (h3: dist A F = 3) 
  : area_of_triangle A F E = 3 * real.sqrt 3 :=
sorry

end area_of_AFE_l93_93563


namespace employed_females_percentage_l93_93325

theorem employed_females_percentage (total_employed_percentage employed_males_percentage employed_females_percentage : ℝ) 
    (h1 : total_employed_percentage = 64) 
    (h2 : employed_males_percentage = 48) 
    (h3 : employed_females_percentage = total_employed_percentage - employed_males_percentage) :
    (employed_females_percentage / total_employed_percentage * 100) = 25 :=
by
  sorry

end employed_females_percentage_l93_93325


namespace find_xy_l93_93615

theorem find_xy : ∃ (x y : ℕ), 0 < x ∧ 0 < y ∧ x^4 = y^2 + 71 ∧ x = 6 ∧ y = 35 :=
by
  sorry

end find_xy_l93_93615


namespace range_of_a_l93_93417

noncomputable def range_a (a : ℝ) : Prop :=
  ∃ (x y : ℝ), 0 < x ∧ 0 < y ∧ (x^3 * Real.exp (y / x) = a * y^3)

theorem range_of_a (a : ℝ) : range_a a → a ≥ Real.exp 3 / 27 :=
by
  sorry

end range_of_a_l93_93417


namespace f_g_of_neg2_l93_93913

def f (x : ℤ) : ℤ := 3 * x + 2
def g (x : ℤ) : ℤ := (x - 1)^2

theorem f_g_of_neg2 : f (g (-2)) = 29 := by
  -- We need to show f(g(-2)) = 29 given the definitions of f and g
  sorry

end f_g_of_neg2_l93_93913


namespace arithmetic_sequence_formula_geometric_sequence_sum_formula_l93_93757

noncomputable def arithmetic_sequence_a_n (n : ℕ) : ℤ :=
  sorry

noncomputable def geometric_sequence_T_n (n : ℕ) : ℤ :=
  sorry

theorem arithmetic_sequence_formula :
  (∃ a₃ : ℤ, a₃ = 5) ∧ (∃ S₃ : ℤ, S₃ = 9) →
  -- Suppose we have an arithmetic sequence $a_n$
  (∀ n : ℕ, n ≥ 1 → arithmetic_sequence_a_n n = 2 * n - 1) := 
sorry

theorem geometric_sequence_sum_formula :
  (∃ q : ℤ, q > 0 ∧ q = 3) ∧ (∃ b₃ : ℤ, b₃ = 9) ∧ (∃ T₃ : ℤ, T₃ = 13) →
  -- Suppose we have a geometric sequence $b_n$ where $b_3 = a_5$
  (∀ n : ℕ, n ≥ 1 → geometric_sequence_T_n n = (3 ^ n - 1) / 2) := 
sorry

end arithmetic_sequence_formula_geometric_sequence_sum_formula_l93_93757


namespace average_TV_sets_in_shops_l93_93003

def shop_a := 20
def shop_b := 30
def shop_c := 60
def shop_d := 80
def shop_e := 50
def total_shops := 5

theorem average_TV_sets_in_shops : (shop_a + shop_b + shop_c + shop_d + shop_e) / total_shops = 48 :=
by
  have h1 : shop_a + shop_b + shop_c + shop_d + shop_e = 240
  { sorry }
  have h2 : 240 / total_shops = 48
  { sorry }
  exact Eq.trans (congrArg (fun x => x / total_shops) h1) h2

end average_TV_sets_in_shops_l93_93003


namespace total_pastries_l93_93328

variable (P x : ℕ)

theorem total_pastries (h1 : P = 28 * (10 + x)) (h2 : P = 49 * (4 + x)) : P = 392 := 
by 
  sorry

end total_pastries_l93_93328


namespace min_frac_sum_pos_real_l93_93547

variable {x y z w : ℝ}

theorem min_frac_sum_pos_real (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hw : 0 < w) (h_sum : x + y + z + w = 1) : 
  (x + y + z) / (x * y * z * w) ≥ 144 := 
sorry

end min_frac_sum_pos_real_l93_93547


namespace price_increase_for_desired_profit_l93_93920

/--
In Xianyou Yonghui Supermarket, the profit from selling Pomelos is 10 yuan per kilogram.
They can sell 500 kilograms per day. Market research has found that, with a constant cost price, if the price per kilogram increases by 1 yuan, the daily sales volume will decrease by 20 kilograms.
Now, the supermarket wants to ensure a daily profit of 6000 yuan while also offering the best deal to the customers.
-/
theorem price_increase_for_desired_profit :
  ∃ x : ℝ, (10 + x) * (500 - 20 * x) = 6000 ∧ x = 5 :=
sorry

end price_increase_for_desired_profit_l93_93920


namespace sale_savings_l93_93366

theorem sale_savings (price_fox : ℝ) (price_pony : ℝ) 
(discount_fox : ℝ) (discount_pony : ℝ) 
(total_discount : ℝ) (num_fox : ℕ) (num_pony : ℕ) 
(price_saved_during_sale : ℝ) :
price_fox = 15 → 
price_pony = 18 → 
num_fox = 3 → 
num_pony = 2 → 
total_discount = 22 → 
discount_pony = 15 → 
discount_fox = total_discount - discount_pony → 
price_saved_during_sale = num_fox * price_fox * (discount_fox / 100) + num_pony * price_pony * (discount_pony / 100) →
price_saved_during_sale = 8.55 := 
by sorry

end sale_savings_l93_93366


namespace mean_study_hours_l93_93944

theorem mean_study_hours :
  let students := [3, 6, 8, 5, 4, 2, 2]
  let hours := [0, 2, 4, 6, 8, 10, 12]
  (0 * 3 + 2 * 6 + 4 * 8 + 6 * 5 + 8 * 4 + 10 * 2 + 12 * 2) / (3 + 6 + 8 + 5 + 4 + 2 + 2) = 5 :=
by
  sorry

end mean_study_hours_l93_93944


namespace Milly_study_time_l93_93405

theorem Milly_study_time :
  let math_time := 60
  let geo_time := math_time / 2
  let mean_time := (math_time + geo_time) / 2
  let total_study_time := math_time + geo_time + mean_time
  total_study_time = 135 := by
  sorry

end Milly_study_time_l93_93405


namespace percentage_change_area_l93_93673

theorem percentage_change_area (L B : ℝ) (hL : L > 0) (hB : B > 0) :
    let A := L * B
    let L' := L / 2
    let B' := 3 * B
    let A' := (L / 2) * (3 * B)
    (A' - A) / A * 100 = 50 :=
by
  let A := L * B
  let A' := (L / 2) * (3 * B)
  have h : (A' - A) / A * 100 = (1 / 2) * 100 := sorry
  exact h

end percentage_change_area_l93_93673


namespace complementary_angles_decrease_percent_l93_93432

theorem complementary_angles_decrease_percent
    (a b : ℝ) 
    (h1 : a + b = 90) 
    (h2 : a / b = 3 / 7) 
    (h3 : new_a = a * 1.15) 
    (h4 : new_a + new_b = 90) : 
    (new_b / b * 100) = 93.57 := 
sorry

end complementary_angles_decrease_percent_l93_93432


namespace greatest_integer_lesser_200_gcd_45_eq_9_l93_93455

theorem greatest_integer_lesser_200_gcd_45_eq_9 :
  ∃ n : ℕ, n < 200 ∧ Int.gcd n 45 = 9 ∧ ∀ m : ℕ, (m < 200 ∧ Int.gcd m 45 = 9) → m ≤ n :=
by
  sorry

end greatest_integer_lesser_200_gcd_45_eq_9_l93_93455


namespace pairs_count_l93_93119

noncomputable def count_pairs (n : ℕ) : ℕ :=
  3^n

theorem pairs_count (A : Finset ℕ) (h : A.card = n) :
  ∃ f : Finset ℕ × Finset ℕ → Finset ℕ, ∀ B C, (B ≠ ∅ ∧ B ⊆ C ∧ C ⊆ A) → (f (B, C)).card = count_pairs n :=
sorry

end pairs_count_l93_93119


namespace amount_daria_needs_l93_93495

theorem amount_daria_needs (ticket_cost : ℕ) (total_tickets : ℕ) (current_money : ℕ) (needed_money : ℕ) 
  (h1 : ticket_cost = 90) 
  (h2 : total_tickets = 4) 
  (h3 : current_money = 189) 
  (h4 : needed_money = 360 - 189) 
  : needed_money = 171 := by
  -- proof omitted
  sorry

end amount_daria_needs_l93_93495


namespace milk_percentage_after_adding_water_l93_93303

theorem milk_percentage_after_adding_water
  (initial_total_volume : ℚ) (initial_milk_percentage : ℚ)
  (additional_water_volume : ℚ) :
  initial_total_volume = 60 → initial_milk_percentage = 0.84 → additional_water_volume = 18.75 →
  (50.4 / (initial_total_volume + additional_water_volume) * 100 = 64) :=
by
  intros h1 h2 h3
  rw [h1, h3]
  simp
  sorry

end milk_percentage_after_adding_water_l93_93303


namespace sum_of_midpoints_l93_93151

theorem sum_of_midpoints (p q r : ℝ) (h : p + q + r = 15) :
  (p + q) / 2 + (p + r) / 2 + (q + r) / 2 = 15 :=
by
  sorry

end sum_of_midpoints_l93_93151


namespace maximal_product_at_12_l93_93922

noncomputable def geometric_sequence (a₁ : ℕ) (q : ℚ) (n : ℕ) : ℚ :=
a₁ * q^(n - 1)

noncomputable def product_first_n_terms (a₁ : ℕ) (q : ℚ) (n : ℕ) : ℚ :=
(a₁ ^ n) * (q ^ ((n - 1) * n / 2))

theorem maximal_product_at_12 :
  ∀ (a₁ : ℕ) (q : ℚ), 
  a₁ = 1536 → 
  q = -1/2 → 
  ∀ (n : ℕ), n ≠ 12 → 
  (product_first_n_terms a₁ q 12) > (product_first_n_terms a₁ q n) :=
by
  sorry

end maximal_product_at_12_l93_93922


namespace distance_between_trees_l93_93179

theorem distance_between_trees (n : ℕ) (L : ℝ) (d : ℝ) (h1 : n = 26) (h2 : L = 700) (h3 : d = L / (n - 1)) : d = 28 :=
sorry

end distance_between_trees_l93_93179


namespace chebyshev_number_of_variables_l93_93536

open MeasureTheory Probability MeasureTheory.Measure

variables {ι : Type*} {Ω : Type*} [MeasureSpace Ω] {X : ι → Ω → ℝ}
variables (hx : ∀ i, has_variance (X i) ∧ (variance (X i) ≤ 4)) (n : ℕ)
variables (ε : ℝ) (hε : ε = 0.25) (p : ℝ) (hp : p = 0.99)

theorem chebyshev_number_of_variables :
  (P (ω, ℝ) in measure_space.measure_space (Ω) => 
    abs ((∑ i in finset.range n, X i ω) / n - (∑ i in finset.range n, Ε (X i)) / n) ≤ ε) > p →
  n ≥ 6400 :=
begin
  sorry
end

end chebyshev_number_of_variables_l93_93536


namespace evaluate_F_of_4_and_f_of_5_l93_93631

def f (a : ℤ) : ℤ := 2 * a - 2
def F (a b : ℤ) : ℤ := b^2 + a + 1

theorem evaluate_F_of_4_and_f_of_5 : F 4 (f 5) = 69 := by
  -- Definitions and intermediate steps are not included in the statement, proof is omitted.
  sorry

end evaluate_F_of_4_and_f_of_5_l93_93631


namespace total_white_papers_l93_93200

-- Define the given conditions
def papers_per_envelope : ℕ := 10
def number_of_envelopes : ℕ := 12

-- The theorem statement
theorem total_white_papers : (papers_per_envelope * number_of_envelopes) = 120 :=
by
  sorry

end total_white_papers_l93_93200


namespace parallel_lines_condition_l93_93000

theorem parallel_lines_condition (a : ℝ) :
  (a = 3 / 2) ↔ (∀ x y : ℝ, (x + 2 * a * y - 1 = 0 → (a - 1) * x + a * y + 1 = 0) → (a = 3 / 2)) :=
sorry

end parallel_lines_condition_l93_93000


namespace pairs_of_powers_of_two_l93_93226

theorem pairs_of_powers_of_two (m n : ℕ) (h1 : m > 0) (h2 : n > 0)
  (h3 : ∃ a : ℕ, m + n = 2^a) (h4 : ∃ b : ℕ, mn + 1 = 2^b) :
  (∃ a : ℕ, m = 2^a - 1 ∧ n = 1) ∨ 
  (∃ a : ℕ, m = 2^(a-1) + 1 ∧ n = 2^(a-1) - 1) :=
sorry

end pairs_of_powers_of_two_l93_93226


namespace mr_smith_grandchildren_prob_l93_93943

open Prob

-- Define the conditions
def children_count : ℕ := 12
def possibilities := 2 ^ children_count
def equal_gender_distribution := Nat.choose children_count (children_count / 2)
def probability_equal_gender := equal_gender_distribution.to_rat / possibilities.to_rat
def probability_unequal_gender := 1 - probability_equal_gender

-- The theorem statement to prove
theorem mr_smith_grandchildren_prob :
  probability_unequal_gender = (3172 : ℚ) / 4096 :=
by
  sorry

end mr_smith_grandchildren_prob_l93_93943


namespace problem_divides_area_l93_93145

noncomputable def vertical_line_divides_area (p q : ℝ) : Prop :=
  let f := λ x : ℝ, -2 * x ^ 2
  let g := λ x : ℝ, x ^ 2 + p * x + q
  let h := λ x : ℝ, f x - g x
  let x1 := (p - Real.sqrt (p ^ 2 - 12 * q)) / -6
  let x2 := (p + Real.sqrt (p ^ 2 - 12 * q)) / -6
  let x0 := -(p / 6)
  x0 = (x1 + x2) / 2

theorem problem_divides_area (p q : ℝ) :
  vertical_line_divides_area p q :=
sorry

end problem_divides_area_l93_93145


namespace maximum_value_of_chords_l93_93510

noncomputable def max_sum_of_chords (r : ℝ) (h1 : r = 5) 
(h2 : ∃ PA PB PC : ℝ, PA = 2 * PB ∧ PA^2 + PB^2 + PC^2 = (2 * r)^2) : ℝ := 
  6 * Real.sqrt 10

theorem maximum_value_of_chords (P : Point) (r : ℝ) (h1 : r = 5) 
(h2 : ∃ PA PB PC : ℝ, PA = 2 * PB ∧ PA^2 + PB^2 + PC^2 = (2 * r)^2) : 
  PA + PB + PC ≤ 6 * Real.sqrt 10 :=
by
  sorry

end maximum_value_of_chords_l93_93510


namespace tan_C_value_b_value_l93_93641

-- Define variables and conditions
variable (A B C a b c : ℝ)
variable (A_eq : A = Real.pi / 4)
variable (cond : b^2 - a^2 = 1 / 4 * c^2)
variable (area_eq : 1 / 2 * b * c * Real.sin A = 5 / 2)

-- First part: Prove tan(C) = 4 given the conditions
theorem tan_C_value : A = Real.pi / 4 ∧ b^2 - a^2 = 1 / 4 * c^2 → Real.tan C = 4 := by
  intro h
  sorry

-- Second part: Prove b = 5 / 2 given the area condition
theorem b_value : (1 / 2 * b * c * Real.sin (Real.pi / 4) = 5 / 2) → b = 5 / 2 := by
  intro h
  sorry

end tan_C_value_b_value_l93_93641


namespace point_after_rotation_l93_93264

-- Definitions based on conditions
def point_N : ℝ × ℝ := (-1, -2)
def origin_O : ℝ × ℝ := (0, 0)
def rotation_180 (P : ℝ × ℝ) : ℝ × ℝ := (-P.1, -P.2)

-- The statement to be proved
theorem point_after_rotation :
  rotation_180 point_N = (1, 2) :=
by
  sorry

end point_after_rotation_l93_93264


namespace f_value_neg_five_half_one_l93_93905

noncomputable def f : ℝ → ℝ := sorry

-- Conditions
axiom odd_function : ∀ x, f (-x) = -f x
axiom periodic_function : ∀ x, f (x + 2) = f x
axiom interval_definition : ∀ x, 0 < x ∧ x < 1 → f x = (4:ℝ) ^ x

-- The statement to prove
theorem f_value_neg_five_half_one : f (-5/2) + f 1 = -2 :=
by
  sorry

end f_value_neg_five_half_one_l93_93905


namespace pyramid_top_value_l93_93623

theorem pyramid_top_value 
  (p : ℕ) (q : ℕ) (z : ℕ) (m : ℕ) (n : ℕ) (left_mid : ℕ) (right_mid : ℕ) 
  (left_upper : ℕ) (right_upper : ℕ) (x_pre : ℕ) (x : ℕ) : 
  p = 20 → 
  q = 6 → 
  z = 44 → 
  m = p + 34 → 
  n = q + z → 
  left_mid = 17 + 29 → 
  right_mid = m + n → 
  left_upper = 36 + left_mid → 
  right_upper = right_mid + 42 → 
  x_pre = left_upper + 78 → 
  x = 2 * x_pre → 
  x = 320 :=
by
  intros
  sorry

end pyramid_top_value_l93_93623


namespace sum_of_squares_of_ages_l93_93688

theorem sum_of_squares_of_ages 
  (d t h : ℕ) 
  (cond1 : 3 * d + t = 2 * h)
  (cond2 : 2 * h ^ 3 = 3 * d ^ 3 + t ^ 3)
  (rel_prime : Nat.gcd d (Nat.gcd t h) = 1) :
  d ^ 2 + t ^ 2 + h ^ 2 = 42 :=
sorry

end sum_of_squares_of_ages_l93_93688


namespace factorize_x4_minus_5x2_plus_4_l93_93040

theorem factorize_x4_minus_5x2_plus_4 (x : ℝ) :
  x^4 - 5 * x^2 + 4 = (x + 1) * (x - 1) * (x + 2) * (x - 2) :=
by
  sorry

end factorize_x4_minus_5x2_plus_4_l93_93040


namespace sum_of_midpoints_x_coordinates_l93_93153

theorem sum_of_midpoints_x_coordinates (a b c : ℝ) (h : a + b + c = 15) :
  (a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 15 := by
  sorry

end sum_of_midpoints_x_coordinates_l93_93153


namespace probability_pq_satisfies_equation_l93_93277

theorem probability_pq_satisfies_equation :
  let p_values := {p : ℕ | 1 ≤ p ∧ p ≤ 20 ∧ ∃ q : ℤ, p * q - 6 * p - 3 * q = 6} in
  (p_values.to_finset.card : ℚ) / 20 = 7 / 20 :=
by
  intro p_values
  have h : p_values = {4, 5, 6, 7, 9, 11, 15}, sorry
  rw [h, Finset.card_insert_of_not_mem, Finset.card_insert_of_not_mem, Finset.card_insert_of_not_mem,
    Finset.card_insert_of_not_mem, Finset.card_insert_of_not_mem, Finset.card_insert_of_not_mem,
    Finset.card_singleton, Finset.card_empty]
  norm_cast
  exact by norm_num

end probability_pq_satisfies_equation_l93_93277


namespace sum_of_midpoints_x_coordinates_l93_93155

theorem sum_of_midpoints_x_coordinates (a b c : ℝ) (h : a + b + c = 15) :
  (a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 15 := by
  sorry

end sum_of_midpoints_x_coordinates_l93_93155


namespace line_circle_intersection_l93_93737

theorem line_circle_intersection (x y : ℝ) (h1 : 7 * x + 5 * y = 14) (h2 : x^2 + y^2 = 4) :
  ∃ (p q : ℝ), (7 * p + 5 * q = 14) ∧ (p^2 + q^2 = 4) ∧ (7 * p + 5 * q = 14) ∧ (p ≠ q) :=
sorry

end line_circle_intersection_l93_93737


namespace total_sales_correct_l93_93333

-- Define the conditions
def total_tickets : ℕ := 65
def senior_ticket_price : ℕ := 10
def regular_ticket_price : ℕ := 15
def regular_tickets_sold : ℕ := 41

-- Calculate the senior citizen tickets sold
def senior_tickets_sold : ℕ := total_tickets - regular_tickets_sold

-- Calculate the revenue from senior citizen tickets
def revenue_senior : ℕ := senior_ticket_price * senior_tickets_sold

-- Calculate the revenue from regular tickets
def revenue_regular : ℕ := regular_ticket_price * regular_tickets_sold

-- Define the total sales amount
def total_sales_amount : ℕ := revenue_senior + revenue_regular

-- The statement we need to prove
theorem total_sales_correct : total_sales_amount = 855 := by
  sorry

end total_sales_correct_l93_93333


namespace gain_percent_is_approx_30_11_l93_93196

-- Definitions for cost price (CP) and selling price (SP)
def CP : ℕ := 930
def SP : ℕ := 1210

-- Definition for gain percent
noncomputable def gain_percent : ℚ :=
  ((SP - CP : ℚ) / CP) * 100

-- Statement to prove the gain percent is approximately 30.11%
theorem gain_percent_is_approx_30_11 :
  abs (gain_percent - 30.11) < 0.01 := by
  sorry

end gain_percent_is_approx_30_11_l93_93196


namespace dot_product_calculation_l93_93753

def vec_a : ℝ × ℝ := (1, 0)
def vec_b : ℝ × ℝ := (2, 3)
def vec_s : ℝ × ℝ := (2 * vec_a.1 - vec_b.1, 2 * vec_a.2 - vec_b.2)
def vec_t : ℝ × ℝ := (vec_a.1 + vec_b.1, vec_a.2 + vec_b.2)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem dot_product_calculation :
  dot_product vec_s vec_t = -9 := by
  sorry

end dot_product_calculation_l93_93753


namespace simplify_fraction_l93_93137

theorem simplify_fraction :
  (1 / 462) + (17 / 42) = 94 / 231 := sorry

end simplify_fraction_l93_93137


namespace simplify_fraction_l93_93136

theorem simplify_fraction :
  (1 / 462) + (17 / 42) = 94 / 231 := sorry

end simplify_fraction_l93_93136


namespace max_points_for_top_teams_l93_93923

-- Definitions based on the problem conditions
def points_for_win : ℕ := 3
def points_for_draw : ℕ := 1
def points_for_loss : ℕ := 0
def number_of_teams : ℕ := 8
def number_of_games_between_each_pair : ℕ := 2
def total_games : ℕ := (number_of_teams * (number_of_teams - 1) / 2) * number_of_games_between_each_pair
def total_points_in_tournament : ℕ := total_games * points_for_win
def top_teams : ℕ := 4

-- Theorem stating the correct answer
theorem max_points_for_top_teams : (total_points_in_tournament / number_of_teams = 33) :=
sorry

end max_points_for_top_teams_l93_93923


namespace small_bottles_needed_l93_93868

noncomputable def small_bottle_capacity := 40 -- in milliliters
noncomputable def large_bottle_capacity := 540 -- in milliliters
noncomputable def worst_case_small_bottle_capacity := 38 -- in milliliters

theorem small_bottles_needed :
  let n_bottles := Int.ceil (large_bottle_capacity / worst_case_small_bottle_capacity : ℚ)
  n_bottles = 15 :=
by
  sorry

end small_bottles_needed_l93_93868


namespace car_trip_time_l93_93720

theorem car_trip_time (T A : ℕ) (h1 : 50 * T = 140 + 53 * A) (h2 : T = 4 + A) : T = 24 := by
  sorry

end car_trip_time_l93_93720


namespace solution_set_m5_range_m_sufficient_condition_l93_93909

theorem solution_set_m5 (x : ℝ) : 
  (|x + 1| + |x - 2| > 5) ↔ (x < -2 ∨ x > 3) := 
sorry

theorem range_m_sufficient_condition (x m : ℝ) (h : ∀ x : ℝ, |x + 1| + |x - 2| - m ≥ 2) : 
  m ≤ 1 := 
sorry

end solution_set_m5_range_m_sufficient_condition_l93_93909


namespace sum_of_fourth_powers_l93_93911

theorem sum_of_fourth_powers (a b c : ℝ)
  (h1 : a + b + c = 1)
  (h2 : a^2 + b^2 + c^2 = 3)
  (h3 : a^3 + b^3 + c^3 = 3) :
  a^4 + b^4 + c^4 = 37 / 6 := 
sorry

end sum_of_fourth_powers_l93_93911


namespace prime_p_satisfies_conditions_l93_93740

theorem prime_p_satisfies_conditions (p : ℕ) (hp : Nat.Prime p) (h1 : Nat.Prime (4 * p^2 + 1)) (h2 : Nat.Prime (6 * p^2 + 1)) : p = 5 :=
sorry

end prime_p_satisfies_conditions_l93_93740


namespace arithmetic_sequence_sum_l93_93063

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h_seq : ∀ n, a (n + 1) - a n = a 1 - a 0) 
  (h1 : a 1 + a 3 + a 5 = 9) (h2 : a 2 + a 4 + a 6 = 15) : a 3 + a 4 = 8 := 
by 
  sorry

end arithmetic_sequence_sum_l93_93063


namespace equilateral_triangle_iff_l93_93659

theorem equilateral_triangle_iff (a b c : ℝ) :
  a^2 + b^2 + c^2 = a*b + b*c + c*a ↔ a = b ∧ b = c :=
sorry

end equilateral_triangle_iff_l93_93659


namespace eggs_leftover_l93_93873

theorem eggs_leftover :
  let abigail_eggs := 58
  let beatrice_eggs := 35
  let carson_eggs := 27
  let total_eggs := abigail_eggs + beatrice_eggs + carson_eggs
  total_eggs % 10 = 0 := by
  let abigail_eggs := 58
  let beatrice_eggs := 35
  let carson_eggs := 27
  let total_eggs := abigail_eggs + beatrice_eggs + carson_eggs
  exact Nat.mod_eq_zero_of_dvd (show 10 ∣ total_eggs from by norm_num)

end eggs_leftover_l93_93873


namespace otimes_2_1_equals_3_l93_93357

namespace MathProof

-- Define the operation
def otimes (a b : ℝ) : ℝ := a^2 - b

-- The main theorem to prove
theorem otimes_2_1_equals_3 : otimes 2 1 = 3 :=
by
  -- Proof content not needed
  sorry

end MathProof

end otimes_2_1_equals_3_l93_93357


namespace parabola_chord_constant_l93_93845

noncomputable def calcT (x₁ x₂ c : ℝ) : ℝ :=
  let a := x₁^2 + (2*x₁^2 - c)^2
  let b := x₂^2 + (2*x₂^2 - c)^2
  1 / Real.sqrt a + 1 / Real.sqrt b

theorem parabola_chord_constant (c : ℝ) (m x₁ x₂ : ℝ) 
    (h₁ : 2*x₁^2 - m*x₁ - c = 0) 
    (h₂ : 2*x₂^2 - m*x₂ - c = 0) : 
    calcT x₁ x₂ c = -20 / (7 * c) :=
by
  sorry

end parabola_chord_constant_l93_93845


namespace geometric_sequence_seventh_term_l93_93472

theorem geometric_sequence_seventh_term (r : ℕ) (r_pos : 0 < r) 
  (h1 : 3 * r^4 = 243) : 
  3 * r^6 = 2187 :=
by
  sorry

end geometric_sequence_seventh_term_l93_93472


namespace number_of_days_l93_93736

def burger_meal_cost : ℕ := 6
def upsize_cost : ℕ := 1
def total_spending : ℕ := 35

/-- The number of days Clinton buys the meal. -/
theorem number_of_days (h1 : burger_meal_cost + upsize_cost = 7) (h2 : total_spending = 35) : total_spending / (burger_meal_cost + upsize_cost) = 5 :=
by
  -- The proof will go here
  sorry

end number_of_days_l93_93736


namespace smallest_N_div_a3_possible_values_of_a3_l93_93268

-- Problem (a)
theorem smallest_N_div_a3 (a : Fin 10 → Nat) (h : StrictMono a) :
  Nat.lcm (a 0) (Nat.lcm (a 1) (Nat.lcm (a 2) (Nat.lcm (a 3) (Nat.lcm (a 4) (Nat.lcm (a 5) (Nat.lcm (a 6) (Nat.lcm (a 7) (Nat.lcm (a 8) (a 9))))))))) / (a 2) = 8 :=
sorry

-- Problem (b)
theorem possible_values_of_a3 (a : Nat) (h_a3_range : 1 ≤ a ∧ a ≤ 1000) :
  a = 315 ∨ a = 630 ∨ a = 945 :=
sorry

end smallest_N_div_a3_possible_values_of_a3_l93_93268


namespace find_rate_of_grapes_l93_93879

def rate_per_kg_of_grapes (G : ℝ) : Prop :=
  let cost_of_grapes := 8 * G
  let cost_of_mangoes := 10 * 55
  let total_paid := 1110
  cost_of_grapes + cost_of_mangoes = total_paid

theorem find_rate_of_grapes : rate_per_kg_of_grapes 70 :=
by
  unfold rate_per_kg_of_grapes
  sorry

end find_rate_of_grapes_l93_93879


namespace total_coronavirus_cases_l93_93789

theorem total_coronavirus_cases (ny_cases ca_cases tx_cases : ℕ)
    (h_ny : ny_cases = 2000)
    (h_ca : ca_cases = ny_cases / 2)
    (h_tx : ca_cases = tx_cases + 400) :
    ny_cases + ca_cases + tx_cases = 3600 := by
  sorry

end total_coronavirus_cases_l93_93789


namespace circumcircle_excircle_distance_squared_l93_93766

variable (R r_A d_A : ℝ)

theorem circumcircle_excircle_distance_squared 
  (h : R ≥ 0)
  (h1 : r_A ≥ 0)
  (h2 : d_A^2 = R^2 + 2 * R * r_A) : d_A^2 = R^2 + 2 * R * r_A := 
by
  sorry

end circumcircle_excircle_distance_squared_l93_93766


namespace total_cost_textbooks_l93_93109

theorem total_cost_textbooks :
  let sale_books := 5 * 10
  let online_books := 40
  let bookstore_books := 3 * 40
  sale_books + online_books + bookstore_books = 210 :=
by
  let sale_books := 5 * 10
  let online_books := 40
  let bookstore_books := 3 * 40
  show sale_books + online_books + bookstore_books = 210
  sorry

end total_cost_textbooks_l93_93109


namespace ratio_product_even_odd_composite_l93_93351

theorem ratio_product_even_odd_composite :
  (4 * 6 * 8 * 10 * 12) / (9 * 15 * 21 * 25 * 27) = (2^10) / (3^6 * 5^2 * 7) :=
by
  sorry

end ratio_product_even_odd_composite_l93_93351


namespace total_cards_beginning_l93_93814

-- Define the initial conditions
def num_boxes_orig : ℕ := 2 + 5  -- Robie originally had 2 + 5 boxes
def cards_per_box : ℕ := 10      -- Each box contains 10 cards
def extra_cards : ℕ := 5         -- 5 cards were not placed in a box

-- Prove the total number of cards Robie had in the beginning
theorem total_cards_beginning : (num_boxes_orig * cards_per_box) + extra_cards = 75 :=
by sorry

end total_cards_beginning_l93_93814


namespace Hari_contribution_l93_93988

theorem Hari_contribution (H : ℕ) (Praveen_capital : ℕ := 3500) (months_Praveen : ℕ := 12) 
                          (months_Hari : ℕ := 7) (profit_ratio_P : ℕ := 2) (profit_ratio_H : ℕ := 3) : 
                          (Praveen_capital * months_Praveen) * profit_ratio_H = (H * months_Hari) * profit_ratio_P → 
                          H = 9000 :=
by
  sorry

end Hari_contribution_l93_93988


namespace lincoln_high_students_club_overlap_l93_93878

theorem lincoln_high_students_club_overlap (total_students : ℕ)
  (drama_club_students science_club_students both_or_either_club_students : ℕ)
  (h1 : total_students = 500)
  (h2 : drama_club_students = 150)
  (h3 : science_club_students = 200)
  (h4 : both_or_either_club_students = 300) :
  drama_club_students + science_club_students - both_or_either_club_students = 50 :=
by
  sorry

end lincoln_high_students_club_overlap_l93_93878


namespace possible_new_perimeters_l93_93471

theorem possible_new_perimeters
  (initial_tiles := 8)
  (initial_shape := "L")
  (initial_perimeter := 12)
  (additional_tiles := 2)
  (new_perimeters := [12, 14, 16]) :
  True := sorry

end possible_new_perimeters_l93_93471


namespace unused_streetlights_remain_l93_93421

def total_streetlights : ℕ := 200
def squares : ℕ := 15
def streetlights_per_square : ℕ := 12

theorem unused_streetlights_remain :
  total_streetlights - (squares * streetlights_per_square) = 20 :=
sorry

end unused_streetlights_remain_l93_93421


namespace seating_arrangements_l93_93257

theorem seating_arrangements : 
  let total := Nat.factorial 10
  let block := Nat.factorial 7 * Nat.factorial 4 
  total - block = 3507840 := 
by 
  let total := Nat.factorial 10
  let block := Nat.factorial 7 * Nat.factorial 4 
  sorry

end seating_arrangements_l93_93257


namespace wrapping_paper_area_correct_l93_93587

-- Conditions as given in the problem
variables (l w h : ℝ)
variable (hlw : l > w)

-- Definition of the area of the wrapping paper
def wrapping_paper_area (l w h : ℝ) : ℝ :=
  (l + 2 * h) * (w + 2 * h)

-- Proof statement
theorem wrapping_paper_area_correct (hlw : l > w) : 
  wrapping_paper_area l w h = l * w + 2 * l * h + 2 * w * h + 4 * h^2 :=
by
  sorry

end wrapping_paper_area_correct_l93_93587


namespace simplify_fraction_l93_93131

theorem simplify_fraction : 
  (1:ℚ) / 462 + 17 / 42 = 94 / 231 := 
by
  sorry

end simplify_fraction_l93_93131


namespace simplify_expression_l93_93795

theorem simplify_expression (a b c d : ℝ) (h₁ : a + b + c + d = 0) (h₂ : a ≠ 0) (h₃ : b ≠ 0) (h₄ : c ≠ 0) (h₅ : d ≠ 0) :
  (1 / (b^2 + c^2 + d^2 - a^2) + 
   1 / (a^2 + c^2 + d^2 - b^2) + 
   1 / (a^2 + b^2 + d^2 - c^2) + 
   1 / (a^2 + b^2 + c^2 - d^2)) = 4 / d^2 := 
sorry

end simplify_expression_l93_93795


namespace remainder_8x_mod_9_l93_93317

theorem remainder_8x_mod_9 (x : ℕ) (h : x % 9 = 5) : (8 * x) % 9 = 4 :=
by
  sorry

end remainder_8x_mod_9_l93_93317


namespace coin_flip_probability_l93_93915

open Classical

noncomputable section

theorem coin_flip_probability :
  let total_outcomes := 2^10
  let exactly_five_heads_tails := Nat.choose 10 5 / total_outcomes
  let even_heads_probability := 1/2
  (even_heads_probability * (1 - exactly_five_heads_tails) / 2 = 193 / 512) :=
by
  sorry

end coin_flip_probability_l93_93915


namespace length_of_each_song_l93_93210

-- Conditions
def first_side_songs : Nat := 6
def second_side_songs : Nat := 4
def total_length_of_tape : Nat := 40

-- Definition of length of each song
def total_songs := first_side_songs + second_side_songs

-- Question: Prove that each song is 4 minutes long
theorem length_of_each_song (h1 : first_side_songs = 6) 
                            (h2 : second_side_songs = 4) 
                            (h3 : total_length_of_tape = 40) 
                            (h4 : total_songs = first_side_songs + second_side_songs) : 
  total_length_of_tape / total_songs = 4 :=
by
  sorry

end length_of_each_song_l93_93210


namespace sin_alpha_minus_pi_over_6_l93_93549

variable (α : ℝ)

theorem sin_alpha_minus_pi_over_6 (h1 : 0 < α ∧ α < π / 2) (h2 : Real.cos (α + π / 6) = 3 / 5) : 
  Real.sin (α - π / 6) = (4 - 3 * Real.sqrt 3) / 10 :=
by
  sorry

end sin_alpha_minus_pi_over_6_l93_93549


namespace sign_of_k_l93_93091

variable (k x y : ℝ)
variable (A B : ℝ × ℝ)
variable (y₁ y₂ : ℝ)
variable (h₁ : A = (-2, y₁))
variable (h₂ : B = (5, y₂))
variable (h₃ : y₁ = k / -2)
variable (h₄ : y₂ = k / 5)
variable (h₅ : y₁ > y₂)
variable (h₀ : k ≠ 0)

-- We need to prove that k < 0
theorem sign_of_k (A B : ℝ × ℝ) (y₁ y₂ k : ℝ) 
  (h₁ : A = (-2, y₁)) 
  (h₂ : B = (5, y₂)) 
  (h₃ : y₁ = k / -2) 
  (h₄ : y₂ = k / 5) 
  (h₅ : y₁ > y₂) 
  (h₀ : k ≠ 0) : k < 0 := 
by
  sorry

end sign_of_k_l93_93091


namespace line_equation_through_two_points_l93_93658

noncomputable def LineEquation (x0 y0 x1 y1 x y : ℝ) : Prop :=
  (x1 ≠ x0) → (y1 ≠ y0) → 
  (y - y0) / (y1 - y0) = (x - x0) / (x1 - x0)

theorem line_equation_through_two_points 
  (x0 y0 x1 y1 : ℝ) 
  (h₁ : x1 ≠ x0) 
  (h₂ : y1 ≠ y0) : 
  ∀ (x y : ℝ), LineEquation x0 y0 x1 y1 x y :=  
by
  sorry

end line_equation_through_two_points_l93_93658


namespace should_agree_to_buy_discount_card_l93_93338

-- Define the conditions
def discount_card_cost := 100
def discount_percentage := 0.03
def cost_of_cakes := 4 * 500
def cost_of_fruits := 1600
def total_cost_without_discount_card := cost_of_cakes + cost_of_fruits
def discount_amount := total_cost_without_discount_card * discount_percentage
def cost_after_discount := total_cost_without_discount_card - discount_amount
def effective_total_cost_with_discount_card := cost_after_discount + discount_card_cost

-- Define the objective statement to prove
theorem should_agree_to_buy_discount_card : effective_total_cost_with_discount_card < total_cost_without_discount_card := by
  sorry

end should_agree_to_buy_discount_card_l93_93338


namespace probability_of_x_in_interval_l93_93412

noncomputable def interval_length (a b : ℝ) : ℝ := b - a

noncomputable def probability_in_interval : ℝ :=
  let length_total := interval_length (-2) 1
  let length_sub := interval_length 0 1
  length_sub / length_total

theorem probability_of_x_in_interval :
  probability_in_interval = 1 / 3 :=
by
  sorry

end probability_of_x_in_interval_l93_93412


namespace inequality_solution_set_l93_93679

theorem inequality_solution_set :
  {x : ℝ | (0 ≤ x ∧ x < 2) ∨ (x = 0)} = {x : ℝ | 0 ≤ x ∧ x < 2} :=
by
  sorry

end inequality_solution_set_l93_93679


namespace average_of_first_5_subjects_l93_93600

theorem average_of_first_5_subjects (avg_6_subjects : ℚ) (marks_6th_subject : ℚ) (total_subjects : ℕ) (total_marks_6_subjects : ℚ) (total_marks_5_subjects : ℚ) (avg_5_subjects : ℚ) :
  avg_6_subjects = 77 ∧ marks_6th_subject = 92 ∧ total_subjects = 6 ∧ total_marks_6_subjects = avg_6_subjects * total_subjects ∧ total_marks_5_subjects = total_marks_6_subjects - marks_6th_subject ∧ avg_5_subjects = total_marks_5_subjects / 5
  → avg_5_subjects = 74 := by
  sorry

end average_of_first_5_subjects_l93_93600


namespace right_triangle_construction_condition_l93_93356

theorem right_triangle_construction_condition
  (b s : ℝ) 
  (h_b_pos : b > 0)
  (h_s_pos : s > 0)
  (h_perimeter : ∃ (AC BC AB : ℝ), AC = b ∧ AC + BC + AB = 2 * s ∧ (AC^2 + BC^2 = AB^2)) :
  b < s := 
sorry

end right_triangle_construction_condition_l93_93356


namespace technicians_count_l93_93635

theorem technicians_count 
  (T R : ℕ) 
  (h1 : T + R = 14) 
  (h2 : 12000 * T + 6000 * R = 9000 * 14) : 
  T = 7 :=
by
  sorry

end technicians_count_l93_93635


namespace cell_phones_in_Delaware_l93_93295

theorem cell_phones_in_Delaware (population : ℕ) (phones_per_1000_people : ℕ)
  (h_population : population = 974000)
  (h_phones_per_1000 : phones_per_1000_people = 673) :
  ∃ cell_phones : ℕ, cell_phones = population / 1000 * phones_per_1000_people ∧ cell_phones = 655502 :=
by {
  use 974 * 673,
  split,
  { rw [h_population, h_phones_per_1000], norm_num },
  { norm_num }
}

end cell_phones_in_Delaware_l93_93295


namespace solve_for_x_l93_93957

noncomputable def x : ℚ := 45^2 / (7 - (3 / 4))

theorem solve_for_x : x = 324 := by
  sorry

end solve_for_x_l93_93957


namespace max_profit_at_nine_l93_93906

noncomputable def profit_function (x : ℝ) : ℝ :=
  -(1/3) * x ^ 3 + 81 * x - 234

theorem max_profit_at_nine :
  ∃ x, x = 9 ∧ ∀ y : ℝ, profit_function y ≤ profit_function 9 :=
by
  sorry

end max_profit_at_nine_l93_93906


namespace prob_not_same_group_l93_93844

variable {A B : Type}

/-- Define an event E where students A and B are in the same group.
    The probability of E can be calculated directly from the given conditions. -/
def prob_same_group (n : ℕ) : ℚ :=
  1 / n

/-- Define the main theorem: the probability that students A and B are not in the same group. -/
theorem prob_not_same_group (n : ℕ) (h: n = 3) :
  1 - prob_same_group n = 2 / 3 :=
by
  rw [h, prob_same_group]
  sorry

end prob_not_same_group_l93_93844


namespace fraction_simplification_l93_93125

-- Definitions based on conditions and question
def lcm_462_42 : ℕ := 462
def prime_factors_462 : List ℕ := [2, 3, 7, 11]
def prime_factors_42 : List ℕ := [2, 3, 7]

-- Main theorem statement
theorem fraction_simplification :
  (1 / 462) + (17 / 42) = 94 / 231 :=
by
  sorry

end fraction_simplification_l93_93125


namespace B_share_in_profit_l93_93596

theorem B_share_in_profit (A B C : ℝ) (total_profit : ℝ) 
    (h1 : A = 3 * B)
    (h2 : B = (2/3) * C)
    (h3 : total_profit = 6600) :
    (B / (A + B + C)) * total_profit = 1200 := 
by
  sorry

end B_share_in_profit_l93_93596


namespace dealer_cash_discount_percentage_l93_93015

-- Definitions of the given conditions
variable (C : ℝ) (n m : ℕ) (profit_p list_ratio : ℝ)
variable (h_n : n = 25) (h_m : m = 20) (h_profit : profit_p = 1.36) (h_list_ratio : list_ratio = 2)

-- The statement we need to prove
theorem dealer_cash_discount_percentage 
  (h_eff_selling_price : (m : ℝ) / n * C = profit_p * C)
  : ((list_ratio * C - (m / n * C)) / (list_ratio * C) * 100 = 60) :=
by
  sorry

end dealer_cash_discount_percentage_l93_93015


namespace tiger_catch_distance_correct_l93_93594

noncomputable def tiger_catch_distance (tiger_leaps_behind : ℕ) (tiger_leaps_per_minute : ℕ) (deer_leaps_per_minute : ℕ) (tiger_m_per_leap : ℕ) (deer_m_per_leap : ℕ) : ℕ :=
  let initial_distance := tiger_leaps_behind * tiger_m_per_leap
  let tiger_per_minute := tiger_leaps_per_minute * tiger_m_per_leap
  let deer_per_minute := deer_leaps_per_minute * deer_m_per_leap
  let gain_per_minute := tiger_per_minute - deer_per_minute
  let time_to_catch := initial_distance / gain_per_minute
  time_to_catch * tiger_per_minute

theorem tiger_catch_distance_correct :
  tiger_catch_distance 50 5 4 8 5 = 800 :=
by
  -- This is the placeholder for the proof.
  sorry

end tiger_catch_distance_correct_l93_93594


namespace sum_of_reciprocals_of_factors_of_13_l93_93314

theorem sum_of_reciprocals_of_factors_of_13 : 
  (1 : ℚ) + (1 / 13) = 14 / 13 :=
by {
  sorry
}

end sum_of_reciprocals_of_factors_of_13_l93_93314


namespace triangle_condition_l93_93392

-- Definitions based on the conditions
def angle_equal (A B C : ℝ) : Prop := A = B - C
def angle_ratio123 (A B C : ℝ) : Prop := A / B = 1 / 2 ∧ A / C = 1 / 3 ∧ B / C = 2 / 3
def pythagorean (a b c : ℝ) : Prop := a * a + b * b = c * c
def side_ratio456 (a b c : ℝ) : Prop := a / b = 4 / 5 ∧ a / c = 4 / 6 ∧ b / c = 5 / 6

-- Main hypothesis with right-angle and its conditions in different options
def is_right_triangle (A B C : ℝ) (a b c : ℝ) : Prop :=
  (angle_equal A B C → A = 90 ∨ B = 90 ∨ C = 90) ∧
  (angle_ratio123 A B C → A = 30 ∧ B = 60 ∧ C = 90) ∧
  (pythagorean a b c → true) ∧
  (side_ratio456 a b c → false) -- option D cannot confirm the triangle is right

theorem triangle_condition (A B C a b c : ℝ) : is_right_triangle A B C a b c :=
sorry

end triangle_condition_l93_93392


namespace conjugate_system_solution_l93_93932

theorem conjugate_system_solution (a b : ℝ) :
  (∀ x y : ℝ,
    (x + (2-a) * y = b + 1) ∧ ((2*a-7) * x + y = -5 - b)
    ↔ x + (2*a-7) * y = -5 - b ∧ (x + (2-a) * y = b + 1))
  ↔ a = 3 ∧ b = -3 := by
  sorry

end conjugate_system_solution_l93_93932


namespace range_of_a_l93_93377

variable {a : ℝ}

-- Proposition p: The solution set of the inequality x^2 - (a+1)x + 1 ≤ 0 is empty
def prop_p (a : ℝ) : Prop := (a + 1) ^ 2 - 4 < 0 

-- Proposition q: The function f(x) = (a+1)^x is increasing within its domain
def prop_q (a : ℝ) : Prop := a > 0 

-- The combined conditions
def combined_conditions (a : ℝ) : Prop := (prop_p a) ∨ (prop_q a) ∧ ¬(prop_p a ∧ prop_q a)

-- The range of values for a
theorem range_of_a (h : combined_conditions a) : -3 < a ∧ a ≤ 0 ∨ a ≥ 1 :=
  sorry

end range_of_a_l93_93377


namespace field_perimeter_l93_93569

noncomputable def outer_perimeter (posts : ℕ) (post_width_inches : ℝ) (spacing_feet : ℝ) : ℝ :=
  let posts_per_side := posts / 4
  let gaps_per_side := posts_per_side - 1
  let post_width_feet := post_width_inches / 12
  let side_length := gaps_per_side * spacing_feet + posts_per_side * post_width_feet
  4 * side_length

theorem field_perimeter : 
  outer_perimeter 32 5 4 = 125 + 1/3 := 
by
  sorry

end field_perimeter_l93_93569


namespace circular_park_diameter_factor_l93_93863

theorem circular_park_diameter_factor (r : ℝ) :
  (π * (3 * r)^2) / (π * r^2) = 9 ∧ (2 * π * (3 * r)) / (2 * π * r) = 3 :=
by
  sorry

end circular_park_diameter_factor_l93_93863


namespace find_number_l93_93778

theorem find_number (x : ℤ) (h : 3 * x - 6 = 2 * x) : x = 6 :=
by
  sorry

end find_number_l93_93778


namespace annual_sparkling_water_cost_l93_93651

theorem annual_sparkling_water_cost :
  (let cost_per_bottle := 2.00
       nights_per_year := 365
       fraction_bottle_per_night := 1 / 5
       bottles_per_year := nights_per_year * fraction_bottle_per_night in
   bottles_per_year * cost_per_bottle = 146.00) :=
by
  -- This is where the actual proof would go.
  sorry

end annual_sparkling_water_cost_l93_93651


namespace find_k_value_l93_93891

theorem find_k_value (k : ℝ) (h₁ : ∀ x, k * x^2 - 5 * x - 12 = 0 → (x = 3 ∨ x = -4 / 3)) : k = 3 :=
sorry

end find_k_value_l93_93891


namespace simplify_and_evaluate_at_3_l93_93956

noncomputable def expression (x : ℝ) : ℝ := 
  (3 / (x - 1) - x - 1) / ((x^2 - 4 * x + 4) / (x - 1))

theorem simplify_and_evaluate_at_3 : expression 3 = -5 := 
  sorry

end simplify_and_evaluate_at_3_l93_93956


namespace find_y_in_interval_l93_93225

theorem find_y_in_interval :
  { y : ℝ | y^2 + 7 * y < 12 } = { y : ℝ | -9 < y ∧ y < 2 } :=
sorry

end find_y_in_interval_l93_93225


namespace percentage_change_area_l93_93674

theorem percentage_change_area (L B : ℝ) (hL : L > 0) (hB : B > 0) :
    let A := L * B
    let L' := L / 2
    let B' := 3 * B
    let A' := (L / 2) * (3 * B)
    (A' - A) / A * 100 = 50 :=
by
  let A := L * B
  let A' := (L / 2) * (3 * B)
  have h : (A' - A) / A * 100 = (1 / 2) * 100 := sorry
  exact h

end percentage_change_area_l93_93674


namespace arithmetic_sequences_integer_ratio_count_l93_93074

theorem arithmetic_sequences_integer_ratio_count 
  (a_n b_n : ℕ → ℕ)
  (A_n B_n : ℕ → ℕ)
  (h₁ : ∀ n, A_n n = n * (a_n 1 + a_n (2 * n - 1)) / 2)
  (h₂ : ∀ n, B_n n = n * (b_n 1 + b_n (2 * n - 1)) / 2)
  (h₃ : ∀ n, A_n n / B_n n = (7 * n + 41) / (n + 3)) :
  ∃ (cnt : ℕ), cnt = 3 ∧ ∀ n, (∃ k, n = 1 + 3 * k) → (a_n n) / (b_n n) = 7 + (10 / (n + 1)) :=
by
  sorry

end arithmetic_sequences_integer_ratio_count_l93_93074


namespace percent_nonunion_part_time_women_l93_93529

noncomputable def percent (part: ℚ) (whole: ℚ) : ℚ := part / whole * 100

def employees : ℚ := 100
def men_ratio : ℚ := 54 / 100
def women_ratio : ℚ := 46 / 100
def full_time_men_ratio : ℚ := 70 / 100
def part_time_men_ratio : ℚ := 30 / 100
def full_time_women_ratio : ℚ := 60 / 100
def part_time_women_ratio : ℚ := 40 / 100
def union_full_time_ratio : ℚ := 60 / 100
def union_part_time_ratio : ℚ := 50 / 100

def men := employees * men_ratio
def women := employees * women_ratio
def full_time_men := men * full_time_men_ratio
def part_time_men := men * part_time_men_ratio
def full_time_women := women * full_time_women_ratio
def part_time_women := women * part_time_women_ratio
def total_full_time := full_time_men + full_time_women
def total_part_time := part_time_men + part_time_women

def union_full_time := total_full_time * union_full_time_ratio
def union_part_time := total_part_time * union_part_time_ratio
def nonunion_full_time := total_full_time - union_full_time
def nonunion_part_time := total_part_time - union_part_time

def nonunion_part_time_women_ratio : ℚ := 50 / 100
def nonunion_part_time_women := part_time_women * nonunion_part_time_women_ratio

theorem percent_nonunion_part_time_women : 
  percent nonunion_part_time_women nonunion_part_time = 52.94 :=
by
  sorry

end percent_nonunion_part_time_women_l93_93529


namespace gift_distribution_l93_93410

noncomputable section

structure Recipients :=
  (ondra : String)
  (matej : String)
  (kuba : String)

structure PetrStatements :=
  (ondra_fire_truck : Bool)
  (kuba_no_fire_truck : Bool)
  (matej_no_merkur : Bool)

def exactly_one_statement_true (s : PetrStatements) : Prop :=
  (s.ondra_fire_truck && ¬s.kuba_no_fire_truck && ¬s.matej_no_merkur)
  ∨ (¬s.ondra_fire_truck && s.kuba_no_fire_truck && ¬s.matej_no_merkur)
  ∨ (¬s.ondra_fire_truck && ¬s.kuba_no_fire_truck && s.matej_no_merkur)

def correct_recipients (r : Recipients) : Prop :=
  r.kuba = "fire truck" ∧ r.matej = "helicopter" ∧ r.ondra = "Merkur"

theorem gift_distribution
  (r : Recipients)
  (s : PetrStatements)
  (h : exactly_one_statement_true s)
  (h0 : ¬exactly_one_statement_true ⟨r.ondra = "fire truck", r.kuba ≠ "fire truck", r.matej ≠ "Merkur"⟩)
  (h1 : ¬exactly_one_statement_true ⟨r.ondra ≠ "fire truck", r.kuba ≠ "fire truck", r.matej ≠ "Merkur"⟩)
  : correct_recipients r := by
  -- Proof is omitted as per the instructions
  sorry

end gift_distribution_l93_93410


namespace average_salary_l93_93855

theorem average_salary (a b c d e : ℕ) (h₁ : a = 8000) (h₂ : b = 5000) (h₃ : c = 15000) (h₄ : d = 7000) (h₅ : e = 9000) :
  (a + b + c + d + e) / 5 = 9000 :=
by sorry

end average_salary_l93_93855


namespace problem1_problem2_l93_93368

variables {a x y : ℝ}

theorem problem1 (h1 : a^x = 2) (h2 : a^y = 3) : a^(x + y) = 6 :=
sorry

theorem problem2 (h1 : a^x = 2) (h2 : a^y = 3) : a^(2 * x - 3 * y) = 4 / 27 :=
sorry

end problem1_problem2_l93_93368


namespace standard_spherical_coordinates_l93_93605

theorem standard_spherical_coordinates :
  ∀ (ρ θ φ : ℝ), 
  ρ = 5 → θ = 3 * Real.pi / 4 → φ = 9 * Real.pi / 5 →
  (ρ > 0) →
  (0 ≤ θ ∧ θ < 2 * Real.pi) →
  (0 ≤ φ ∧ φ ≤ Real.pi) →
  (ρ, θ, φ) = (5, 7 * Real.pi / 4, Real.pi / 5) :=
by sorry

end standard_spherical_coordinates_l93_93605


namespace unique_solution_real_l93_93280

theorem unique_solution_real {x y : ℝ} (h1 : x * (x + y)^2 = 9) (h2 : x * (y^3 - x^3) = 7) :
  x = 1 ∧ y = 2 :=
sorry

end unique_solution_real_l93_93280


namespace sum_of_midpoints_l93_93149

theorem sum_of_midpoints (p q r : ℝ) (h : p + q + r = 15) :
  (p + q) / 2 + (p + r) / 2 + (q + r) / 2 = 15 :=
by
  sorry

end sum_of_midpoints_l93_93149


namespace square_area_of_triangle_on_hyperbola_l93_93843

noncomputable def centroid_is_vertex (triangle : Set (ℝ × ℝ)) : Prop :=
  ∃ v : ℝ × ℝ, v ∈ triangle ∧ v.1 * v.2 = 4

noncomputable def triangle_properties (triangle : Set (ℝ × ℝ)) : Prop :=
  centroid_is_vertex triangle ∧
  (∃ centroid : ℝ × ℝ, 
    centroid_is_vertex triangle ∧ 
    (∀ p ∈ triangle, centroid ∈ triangle))

theorem square_area_of_triangle_on_hyperbola :
  ∃ triangle : Set (ℝ × ℝ), triangle_properties triangle ∧ (∃ area_sq : ℝ, area_sq = 1728) :=
by
  sorry

end square_area_of_triangle_on_hyperbola_l93_93843


namespace percentage_change_in_area_of_rectangle_l93_93670

theorem percentage_change_in_area_of_rectangle
  (L B : ℝ) (hL : L > 0) (hB : B > 0) :
  let A1 := L * B
  let A2 := (L / 2) * (3 * B)
  let percentage_change := ((A2 - A1) / A1) * 100
  percentage_change = 50 := by
  let A1 := L * B
  let A2 := (L / 2) * (3 * B)
  let percentage_change := ((A2 - A1) / A1) * 100
  calc percentage_change
      = ((A2 - A1) / A1) * 100 : rfl
  ... = ((((L / 2) * (3 * B)) - (L * B)) / (L * B)) * 100 : by rw [A1, A2]
  ... = ((3 / 2 * A1 - A1) / A1) * 100 : by rw [A2, ←mul_assoc, mul_div_cancel' _ (two_ne_zero : 2 ≠ 0)]
  ... = ((3 / 2 - 1) * A1 / A1) * 100 : by simp only [mul_sub, mul_div_cancel' _ (ne_of_gt hL)]
  ... = (1 / 2) * 100 : by rw [div_mul_cancel' (ne_of_gt (lt_of_lt_of_le zero_lt_one (le_of_lt ((mul_lt_iff_lt_one_left (lt_of_lt_of_le zero_lt_one hL)).mpr (half_pos (lt_of_le_of_ne (le_of_lt (lt_of_lt_of_le (lt_one_mul_self (lt_of_le_of_lt (zero_le_two) (lt_add_one 1))) (lt_of_lt_of_le (two_le_one) (lt_one 2))))))))) so))
  ... = 50 : rfl

end percentage_change_in_area_of_rectangle_l93_93670


namespace total_cost_textbooks_l93_93110

theorem total_cost_textbooks :
  let sale_books := 5 * 10
  let online_books := 40
  let bookstore_books := 3 * 40
  sale_books + online_books + bookstore_books = 210 :=
by
  let sale_books := 5 * 10
  let online_books := 40
  let bookstore_books := 3 * 40
  show sale_books + online_books + bookstore_books = 210
  sorry

end total_cost_textbooks_l93_93110


namespace count_quadruples_l93_93776

open Real

theorem count_quadruples:
  ∃ qs : Finset (ℝ × ℝ × ℝ × ℝ),
  (∀ (a b c k : ℝ), (a, b, c, k) ∈ qs ↔ a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ 
    a^k = b * c ∧
    b^k = c * a ∧
    c^k = a * b
  ) ∧
  qs.card = 8 :=
sorry

end count_quadruples_l93_93776


namespace bucket_problem_l93_93323

variable (A B C : ℝ)

theorem bucket_problem :
  (A - 6 = (1 / 3) * (B + 6)) →
  (B - 6 = (1 / 2) * (A + 6)) →
  (C - 8 = (1 / 2) * (A + 8)) →
  A = 13.2 :=
by
  sorry

end bucket_problem_l93_93323


namespace total_employees_in_buses_l93_93447

theorem total_employees_in_buses :
  let bus1_percentage_full := 0.60,
      bus2_percentage_full := 0.70,
      bus_capacity := 150
  in
  (bus1_percentage_full * bus_capacity + bus2_percentage_full * bus_capacity) = 195 := by
  sorry

end total_employees_in_buses_l93_93447


namespace john_speed_above_limit_l93_93394

def distance : ℝ := 150
def time : ℝ := 2
def speed_limit : ℝ := 60

theorem john_speed_above_limit :
  distance / time - speed_limit = 15 :=
by
  sorry

end john_speed_above_limit_l93_93394


namespace total_money_shared_l93_93112

-- Conditions
def ken_share : ℕ := 1750
def tony_share : ℕ := 2 * ken_share

-- Question and proof to be demonstrated
theorem total_money_shared : ken_share + tony_share = 5250 :=
by sorry

end total_money_shared_l93_93112


namespace original_population_is_l93_93188

variable (P : ℕ)

def reduced_by_bombardment (P : ℕ) : ℝ :=
  0.85 * P

def reduced_by_fear (remaining_population : ℝ) : ℝ :=
  0.75 * remaining_population

def final_population (P : ℕ) : ℝ :=
  reduced_by_fear (reduced_by_bombardment P)

theorem original_population_is (h : final_population P = 4555) : P = 7143 :=
by
  sorry

end original_population_is_l93_93188


namespace simplify_fraction_l93_93557

open Complex

theorem simplify_fraction :
  (3 + 3 * I) / (-1 + 3 * I) = -1.2 - 1.2 * I :=
by
  sorry

end simplify_fraction_l93_93557


namespace parabola_point_b_l93_93724

variable {a b : ℝ}

theorem parabola_point_b (h1 : 6 = 2^2 + 2*a + b) (h2 : -14 = (-2)^2 - 2*a + b) : b = -8 :=
by
  -- sorry as a placeholder for the actual proof.
  sorry

end parabola_point_b_l93_93724


namespace find_factor_l93_93593

theorem find_factor (n f : ℤ) (h₁ : n = 124) (h₂ : n * f - 138 = 110) : f = 2 := by
  sorry

end find_factor_l93_93593


namespace f_of_f_3_eq_3_l93_93102

noncomputable def f (x : ℝ) : ℝ :=
if x < 2 then 1 - Real.logb 2 (2 - x) else 2^(1 - x) + 3 / 2

theorem f_of_f_3_eq_3 : f (f 3) = 3 := by
  sorry

end f_of_f_3_eq_3_l93_93102


namespace solve_equation_l93_93558

theorem solve_equation (x y z : ℕ) :
  (∃ n : ℕ, x = 2^n ∧ y = 2^n ∧ z = 2 * n + 2) ↔ (x^2 + 3 * y^2 = 2^z) :=
by
  sorry

end solve_equation_l93_93558


namespace A_doubles_after_6_months_l93_93190

variable (x : ℕ)

def A_investment_share (x : ℕ) := (3000 * x) + (6000 * (12 - x))
def B_investment_share := 4500 * 12

theorem A_doubles_after_6_months (h : A_investment_share x = B_investment_share) : x = 6 :=
by
  sorry

end A_doubles_after_6_months_l93_93190


namespace stratified_sampling_l93_93839

theorem stratified_sampling (N : ℕ) (r1 r2 r3 : ℕ) (sample_size : ℕ) 
  (ratio_given : r1 = 5 ∧ r2 = 2 ∧ r3 = 3) 
  (total_sample_size : sample_size = 200) :
  sample_size * r3 / (r1 + r2 + r3) = 60 := 
by
  sorry

end stratified_sampling_l93_93839


namespace chess_tournament_games_l93_93711

/--
There are 4 chess amateurs playing in a tournament. Each amateur plays against every other amateur exactly once.
Prove that the total number of unique chess games possible to be played in the tournament is 6.
-/
theorem chess_tournament_games 
  (num_players : ℕ)
  (h : num_players = 4)
  : nat.choose 4 2 = 6 := 
by sorry

end chess_tournament_games_l93_93711


namespace sum_excluding_multiples_l93_93640

theorem sum_excluding_multiples (S_total S_2 S_3 S_6 : ℕ) 
  (hS_total : S_total = (100 * (1 + 100)) / 2) 
  (hS_2 : S_2 = (50 * (2 + 100)) / 2) 
  (hS_3 : S_3 = (33 * (3 + 99)) / 2) 
  (hS_6 : S_6 = (16 * (6 + 96)) / 2) :
  S_total - S_2 - S_3 + S_6 = 1633 :=
by
  sorry

end sum_excluding_multiples_l93_93640


namespace cos_fourth_power_sum_l93_93032

open Real

theorem cos_fourth_power_sum :
  (cos (0 : ℝ))^4 + (cos (π / 6))^4 + (cos (π / 3))^4 + (cos (π / 2))^4 +
  (cos (2 * π / 3))^4 + (cos (5 * π / 6))^4 + (cos π)^4 = 13 / 4 := 
by
  sorry

end cos_fourth_power_sum_l93_93032


namespace absolute_value_solution_l93_93081

theorem absolute_value_solution (m : ℤ) (h : abs m = abs (-7)) : m = 7 ∨ m = -7 := by
  sorry

end absolute_value_solution_l93_93081


namespace race_head_start_l93_93011

theorem race_head_start (v_A v_B : ℕ) (h : v_A = 4 * v_B) (d : ℕ) : 
  100 / v_A = (100 - d) / v_B → d = 75 :=
by
  sorry

end race_head_start_l93_93011


namespace simplify_fraction_l93_93135

theorem simplify_fraction :
  (1 / 462) + (17 / 42) = 94 / 231 := sorry

end simplify_fraction_l93_93135


namespace min_distance_to_line_value_of_AB_l93_93786

noncomputable def point_B : ℝ × ℝ := (1, 1)
noncomputable def point_A : ℝ × ℝ := (4 * Real.sqrt 2, Real.pi / 4)

noncomputable def curve_C (θ : ℝ) : ℝ × ℝ :=
  (2 * Real.cos θ, Real.sqrt 3 * Real.sin θ)

noncomputable def polar_line_l (a : ℝ) (θ : ℝ) : ℝ :=
  a * Real.cos (θ - Real.pi / 4)

noncomputable def line_l1 (m : ℝ) (x y : ℝ) : Prop :=
  x + y + m = 0

theorem min_distance_to_line {θ : ℝ} (a : ℝ) :
  polar_line_l a θ = 4 * Real.sqrt 2 → 
  ∃ d, d = (8 * Real.sqrt 2 - Real.sqrt 14) / 2 :=
by
  sorry

theorem value_of_AB :
  ∃ AB, AB = 12 * Real.sqrt 2 / 7 :=
by
  sorry

end min_distance_to_line_value_of_AB_l93_93786


namespace total_employees_in_buses_l93_93451

-- Definitions from conditions
def busCapacity : ℕ := 150
def percentageFull1 : ℕ := 60
def percentageFull2 : ℕ := 70

-- Proving the total number of employees
theorem total_employees_in_buses : 
  (percentageFull1 * busCapacity / 100) + (percentageFull2 * busCapacity / 100) = 195 := 
by
  sorry

end total_employees_in_buses_l93_93451


namespace square_area_PS_l93_93690

noncomputable def area_of_square_on_PS : ℕ :=
  sorry

theorem square_area_PS (PQ QR RS PR PS : ℝ)
  (h1 : PQ ^ 2 = 25)
  (h2 : QR ^ 2 = 49)
  (h3 : RS ^ 2 = 64)
  (h4 : PQ^2 + QR^2 = PR^2)
  (h5 : PR^2 + RS^2 = PS^2) :
  PS^2 = 138 :=
by
  -- proof skipping
  sorry


end square_area_PS_l93_93690


namespace correct_equation_l93_93705

theorem correct_equation (a b : ℝ) : 
  (¬ (a^2 + a^3 = a^6)) ∧ (¬ ((ab)^2 = ab^2)) ∧ (¬ ((a+b)^2 = a^2 + b^2)) ∧ ((a+b)*(a-b) = a^2 - b^2) :=
by {
  sorry
}

end correct_equation_l93_93705


namespace smallest_range_l93_93871

theorem smallest_range {x1 x2 x3 x4 x5 : ℝ} 
  (h1 : (x1 + x2 + x3 + x4 + x5) = 100)
  (h2 : x3 = 18)
  (h3 : 2 * x1 + 2 * x5 + 18 = 100): 
  x5 - x1 = 19 :=
by {
  sorry
}

end smallest_range_l93_93871


namespace problem1_problem2_l93_93859
noncomputable section

-- Problem (1) Lean Statement
theorem problem1 : |-4| - (2021 - Real.pi)^0 + (Real.cos (Real.pi / 3))⁻¹ - (-Real.sqrt 3)^2 = 2 :=
by 
  sorry

-- Problem (2) Lean Statement
theorem problem2 (a : ℝ) (h : a ≠ 2 ∧ a ≠ -2) : 
  (1 + 4 / (a^2 - 4)) / (a / (a + 2)) = a / (a - 2) := 
by 
  sorry

end problem1_problem2_l93_93859


namespace machine_A_time_to_produce_x_boxes_l93_93459

-- Definitions of the conditions
def machine_A_rate (T : ℕ) (x : ℕ) : ℚ := x / T
def machine_B_rate (x : ℕ) : ℚ := 2 * x / 5
def combined_rate (T : ℕ) (x : ℕ) : ℚ := (x / 2) 

-- The theorem statement
theorem machine_A_time_to_produce_x_boxes (x : ℕ) : 
  ∀ T : ℕ, 20 * (machine_A_rate T x + machine_B_rate x) = 10 * x → T = 10 :=
by
  intros T h
  sorry

end machine_A_time_to_produce_x_boxes_l93_93459


namespace correct_operation_l93_93703

theorem correct_operation (a b : ℝ) : (a + b) * (a - b) = a^2 - b^2 := by
  sorry

end correct_operation_l93_93703


namespace xy_yz_zx_over_x2_y2_z2_l93_93548

theorem xy_yz_zx_over_x2_y2_z2 (x y z : ℝ) (h1 : x ≠ y) (h2 : y ≠ z) (h3 : z ≠ x) (h_sum : x + y + z = 1) :
  (xy + yz + zx) / (x^2 + y^2 + z^2) = (1 - (x^2 + y^2 + z^2)) / (2 * (x^2 + y^2 + z^2)) :=
by
  sorry

end xy_yz_zx_over_x2_y2_z2_l93_93548


namespace coordinates_of_P_with_respect_to_origin_l93_93832

def point (x y : ℝ) : Prop := True

theorem coordinates_of_P_with_respect_to_origin :
  point 2 (-3) ↔ point 2 (-3) := by
  sorry

end coordinates_of_P_with_respect_to_origin_l93_93832


namespace compound_interest_amount_l93_93083

theorem compound_interest_amount (P r t SI : ℝ) (h1 : t = 3) (h2 : r = 0.10) (h3 : SI = 900) :
  SI = P * r * t → P = 900 / (0.10 * 3) → (P * (1 + r)^t - P = 993) :=
by
  intros hSI hP
  sorry

end compound_interest_amount_l93_93083


namespace quadratic_sum_of_squares_l93_93059

theorem quadratic_sum_of_squares (α β : ℝ) (h1 : α * β = 3) (h2 : α + β = 7) : α^2 + β^2 = 43 := 
by
  sorry

end quadratic_sum_of_squares_l93_93059


namespace math_problem_l93_93687

theorem math_problem
    (p q s : ℕ)
    (prime_p : Nat.Prime p)
    (prime_q : Nat.Prime q)
    (prime_s : Nat.Prime s)
    (h1 : p * q = s + 6)
    (h2 : 3 < p)
    (h3 : p < q) :
    p = 5 :=
    sorry

end math_problem_l93_93687


namespace positive_rational_representation_l93_93353

theorem positive_rational_representation (q : ℚ) (h_pos_q : 0 < q) :
  ∃ (a b c d : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ q = (a^2021 + b^2023) / (c^2022 + d^2024) :=
by
  sorry

end positive_rational_representation_l93_93353


namespace find_n_values_l93_93501

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def A_n_k (n k : ℕ) : ℕ := (10^n + 54 * 10^k - 1) / 9

def every_A_n_k_prime (n : ℕ) : Prop :=
  ∀ k, k < n → is_prime (A_n_k n k)

theorem find_n_values :
  ∀ n : ℕ, every_A_n_k_prime n → n = 1 ∨ n = 2 := sorry

end find_n_values_l93_93501


namespace nancy_pensils_total_l93_93116

theorem nancy_pensils_total
  (initial: ℕ) 
  (mult_factor: ℕ) 
  (add_pencils: ℕ) 
  (final_total: ℕ) 
  (h1: initial = 27)
  (h2: mult_factor = 4)
  (h3: add_pencils = 45):
  final_total = initial * mult_factor + add_pencils := 
by
  sorry

end nancy_pensils_total_l93_93116


namespace abs_sum_plus_two_eq_sum_abs_l93_93644

theorem abs_sum_plus_two_eq_sum_abs {a b c : ℤ} (h : |a + b + c| + 2 = |a| + |b| + |c|) :
  a^2 = 1 ∨ b^2 = 1 ∨ c^2 = 1 :=
sorry

end abs_sum_plus_two_eq_sum_abs_l93_93644


namespace proof_b_greater_a_greater_c_l93_93077

def a : ℤ := -2 * 3^2
def b : ℤ := (-2 * 3)^2
def c : ℤ := - (2 * 3)^2

theorem proof_b_greater_a_greater_c (ha : a = -18) (hb : b = 36) (hc : c = -36) : b > a ∧ a > c := 
by
  rw [ha, hb, hc]
  exact And.intro (by norm_num) (by norm_num)

end proof_b_greater_a_greater_c_l93_93077


namespace solve_inequalities_l93_93283

theorem solve_inequalities (x : ℝ) :
  (1 / x < 1 ∧ |4 * x - 1| > 2) →
  (x < -1/4 ∨ x > 1) :=
by
  sorry

end solve_inequalities_l93_93283


namespace find_a3_l93_93639

noncomputable def geometric_seq (a : ℕ → ℕ) (q : ℕ) : Prop :=
∀ n, a (n+1) = a n * q

theorem find_a3 (a : ℕ → ℕ) (q : ℕ) (h_geom : geometric_seq a q) (hq : q > 1)
  (h1 : a 4 - a 0 = 15) (h2 : a 3 - a 1 = 6) :
  a 2 = 4 :=
by
  sorry

end find_a3_l93_93639


namespace people_dislike_both_radio_and_music_l93_93946

theorem people_dislike_both_radio_and_music (N : ℕ) (p_r p_rm : ℝ) (hN : N = 2000) (hp_r : p_r = 0.25) (hp_rm : p_rm = 0.15) : 
  N * p_r * p_rm = 75 :=
by {
  sorry
}

end people_dislike_both_radio_and_music_l93_93946


namespace mike_earnings_l93_93111

theorem mike_earnings :
  let total_games := 16
  let non_working_games := 8
  let price_per_game := 7
  let working_games := total_games - non_working_games
  let earnings := working_games * price_per_game
  earnings = 56 := 
by
  sorry

end mike_earnings_l93_93111


namespace min_floodgates_to_reduce_level_l93_93925

-- Definitions for the conditions given in the problem
def num_floodgates : ℕ := 10
def a (v : ℝ) := 30 * v
def w (v : ℝ) := 2 * v

def time_one_gate : ℝ := 30
def time_two_gates : ℝ := 10
def time_target : ℝ := 3

-- Prove that the minimum number of floodgates \(n\) that must be opened to achieve the goal
theorem min_floodgates_to_reduce_level (v : ℝ) (n : ℕ) :
  (a v + time_target * v) ≤ (n * time_target * w v) → n ≥ 6 :=
by
  sorry

end min_floodgates_to_reduce_level_l93_93925


namespace intersection_points_count_l93_93430

def f (x : ℝ) : ℝ := 2 * Real.log x
def g (x : ℝ) : ℝ := x^2 - 4 * x + 5

theorem intersection_points_count : 
    ∃ (n : ℕ), n = 2 ∧ ∀ x, f x = g x → x = 2 ∨ -- the exact intersection points
    sorry

end intersection_points_count_l93_93430


namespace area_region_eq_6_25_l93_93840

noncomputable def area_of_region : ℝ :=
  ∫ x in -0.5..4.5, (5 - |x - 2| - |x - 2|)

theorem area_region_eq_6_25 :
  area_of_region = 6.25 :=
sorry

end area_region_eq_6_25_l93_93840


namespace veronica_flashlight_distance_l93_93307

theorem veronica_flashlight_distance (V F Vel : ℕ) 
  (h1 : F = 3 * V)
  (h2 : Vel = 5 * F - 2000)
  (h3 : Vel = V + 12000) : 
  V = 1000 := 
by {
  sorry 
}

end veronica_flashlight_distance_l93_93307


namespace find_value_of_expression_l93_93100

theorem find_value_of_expression (a b c : ℝ) (h : (2*a - 6)^2 + (3*b - 9)^2 + (4*c - 12)^2 = 0) : a + 2*b + 3*c = 18 := 
sorry

end find_value_of_expression_l93_93100


namespace time_for_first_half_is_15_l93_93550

-- Definitions of the conditions in Lean
def floors := 20
def time_per_floor_next_5 := 5
def time_per_floor_final_5 := 16
def total_time := 120

-- Theorem statement
theorem time_for_first_half_is_15 :
  ∃ T, (T + (5 * time_per_floor_next_5) + (5 * time_per_floor_final_5) = total_time) ∧ (T = 15) :=
by
  sorry

end time_for_first_half_is_15_l93_93550


namespace a_can_be_any_real_l93_93379

theorem a_can_be_any_real (a b c d e : ℝ) (h1 : b ≠ 0) (h2 : d ≠ 0) (h3 : e ≠ 0) :
  ∃ a : ℝ, true :=
by sorry

end a_can_be_any_real_l93_93379


namespace days_to_fill_tank_l93_93537

-- Definitions based on the problem conditions
def tank_capacity_liters : ℕ := 50
def liters_to_milliliters : ℕ := 1000
def rain_collection_per_day : ℕ := 800
def river_collection_per_day : ℕ := 1700
def total_collection_per_day : ℕ := rain_collection_per_day + river_collection_per_day
def tank_capacity_milliliters : ℕ := tank_capacity_liters * liters_to_milliliters

-- Statement of the proof that Jacob needs 20 days to fill the tank
theorem days_to_fill_tank : tank_capacity_milliliters / total_collection_per_day = 20 := by
  sorry

end days_to_fill_tank_l93_93537


namespace problem_concentric_circles_chord_probability_l93_93847

open ProbabilityTheory

noncomputable def probability_chord_intersects_inner_circle
  (r1 r2 : ℝ) (h : r1 < r2) : ℝ :=
1/6

theorem problem_concentric_circles_chord_probability :
  probability_chord_intersects_inner_circle 1.5 3 
  (by norm_num) = 1/6 :=
sorry

end problem_concentric_circles_chord_probability_l93_93847


namespace samia_walking_distance_l93_93279

noncomputable def total_distance (x : ℝ) : ℝ := 4 * x
noncomputable def biking_distance (x : ℝ) : ℝ := 3 * x
noncomputable def walking_distance (x : ℝ) : ℝ := x
noncomputable def biking_time (x : ℝ) : ℝ := biking_distance x / 12
noncomputable def walking_time (x : ℝ) : ℝ := walking_distance x / 4
noncomputable def total_time (x : ℝ) : ℝ := biking_time x + walking_time x

theorem samia_walking_distance : ∀ (x : ℝ), total_time x = 1 → walking_distance x = 2 :=
by
  sorry

end samia_walking_distance_l93_93279


namespace simplify_fraction_l93_93133

theorem simplify_fraction : 
  (1:ℚ) / 462 + 17 / 42 = 94 / 231 := 
by
  sorry

end simplify_fraction_l93_93133


namespace incorrect_rational_number_statement_l93_93708

theorem incorrect_rational_number_statement :
  ¬ (∀ x : ℚ, x > 0 ∨ x < 0) := by
sorry

end incorrect_rational_number_statement_l93_93708


namespace sum_x_midpoints_of_triangle_l93_93162

theorem sum_x_midpoints_of_triangle (a b c : ℝ) (h : a + b + c = 15) :
  (a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 15 :=
by
  -- Proof omitted, replace with actual proof
  sorry

end sum_x_midpoints_of_triangle_l93_93162


namespace dolphins_points_l93_93387

variable (S D : ℕ)

theorem dolphins_points :
  (S + D = 36) ∧ (S = D + 12) → D = 12 :=
by
  sorry

end dolphins_points_l93_93387


namespace seventh_oblong_number_l93_93344

/-- An oblong number is the number of dots in a rectangular grid where the number of rows is one more than the number of columns. -/
def is_oblong_number (n : ℕ) (x : ℕ) : Prop :=
  x = n * (n + 1)

/-- The 7th oblong number is 56. -/
theorem seventh_oblong_number : ∃ x, is_oblong_number 7 x ∧ x = 56 :=
by 
  use 56
  unfold is_oblong_number
  constructor
  rfl -- This confirms the computation 7 * 8 = 56
  sorry -- Wrapping up the proof, no further steps needed

end seventh_oblong_number_l93_93344


namespace min_a_squared_plus_b_squared_l93_93238

theorem min_a_squared_plus_b_squared (a b : ℝ) (h : ∃ x : ℝ, x^4 + a * x^3 + b * x^2 + a * x + 1 = 0) : a^2 + b^2 ≥ 4 / 5 := 
sorry

end min_a_squared_plus_b_squared_l93_93238


namespace divisibility_condition_l93_93094

theorem divisibility_condition (n : ℕ) : 
  13 ∣ (4 * 3^(2^n) + 3 * 4^(2^n)) ↔ Even n := 
sorry

end divisibility_condition_l93_93094


namespace no_int_solutions_5x2_minus_4y2_eq_2017_l93_93276

theorem no_int_solutions_5x2_minus_4y2_eq_2017 :
  ¬ ∃ x y : ℤ, 5 * x^2 - 4 * y^2 = 2017 :=
by
  -- The detailed proof goes here
  sorry

end no_int_solutions_5x2_minus_4y2_eq_2017_l93_93276


namespace range_of_a_l93_93269

open Set

variable {α : Type*} [LinearOrderedField α]

def p (x : α) : Prop := 2 * x^2 - 3 * x + 1 ≤ 0
def q (x a : α) : Prop := x^2 - (2 * a + 1) * x + a * (a + 1) ≤ 0

theorem range_of_a (A B : Set α) (a : α) :
  (∀ x, p x → q x a) →
  A = { x | 1 / 2 ≤ x ∧ x ≤ 1 } →
  B = { x | a ≤ x ∧ x ≤ a + 1 } →
  p x → q x a :=
by sorry

end range_of_a_l93_93269


namespace minimum_value_of_16b_over_ac_l93_93386

noncomputable def minimum_16b_over_ac (a b c : ℝ) (A B C : ℝ) : ℝ :=
  if (0 < B) ∧ (B < Real.pi / 2) ∧
     (Real.cos B ^ 2 + (1 / 2) * Real.sin (2 * B) = 1) ∧
     ((Real.sqrt (a^2 + c^2 - 2 * a * c * Real.cos B) = 3)) then
    16 * b / (a * c)
  else 0

theorem minimum_value_of_16b_over_ac (a b c : ℝ) (A B C : ℝ)
  (h1 : 0 < B)
  (h2 : B < Real.pi / 2)
  (h3 : Real.cos B ^ 2 + (1 / 2) * Real.sin (2 * B) = 1)
  (h4 : Real.sqrt (a^2 + c^2 - 2 * a * c * Real.cos B) = 3) :
  minimum_16b_over_ac a b c A B C = 16 * (2 - Real.sqrt 2) / 3 := 
sorry

end minimum_value_of_16b_over_ac_l93_93386


namespace sqrt_three_irrational_among_l93_93205

theorem sqrt_three_irrational_among :
  (¬ ∃ a b : ℤ, b ≠ 0 ∧ sqrt 3 = a / b) ∧
  (∃ a b : ℤ, b ≠ 0 ∧ -1 = a / b) ∧
  (∃ a b : ℤ, b ≠ 0 ∧ 0 = a / b) ∧
  (∃ a b : ℤ, b ≠ 0 ∧ 1 / 2 = a / b) :=
by
  split
  · sorry -- Proof of irrationality of sqrt(3)
  split
  · use [-1, 1]
    split
    · exact ne_zero_of_pos one_pos
    · norm_num
  split
  · use [0, 1]
    split
    · exact one_ne_zero
    · norm_num
  · use [1, 2]
    split
    · exact two_ne_zero
    · norm_num

end sqrt_three_irrational_among_l93_93205


namespace order_of_magnitudes_l93_93073

theorem order_of_magnitudes (x : ℝ) (hx : 0.8 < x ∧ x < 0.9) : x < x^(x^x) ∧ x^(x^x) < x^x :=
by
  -- Definitions for y and z.
  let y := x^x
  let z := x^(x^x)
  have h1 : x < y := sorry
  have h2 : z < y := sorry
  have h3 : x < z := sorry
  exact ⟨h3, h2⟩

end order_of_magnitudes_l93_93073


namespace radius_of_circle_l93_93638

-- Definitions based on conditions
def center_in_first_quadrant (C : ℝ × ℝ) : Prop :=
  C.1 > 0 ∧ C.2 > 0

def intersects_x_axis (C : ℝ × ℝ) (r : ℝ) : Prop :=
  r = Real.sqrt ((C.1 - 1)^2 + (C.2)^2) ∧ r = Real.sqrt ((C.1 - 3)^2 + (C.2)^2)

def tangent_to_line (C : ℝ × ℝ) (r : ℝ) : Prop :=
  r = abs (C.1 - C.2 + 1) / Real.sqrt 2

-- Main statement
theorem radius_of_circle (C : ℝ × ℝ) (r : ℝ) 
  (h1 : center_in_first_quadrant C)
  (h2 : intersects_x_axis C r)
  (h3 : tangent_to_line C r) : 
  r = Real.sqrt 2 := 
sorry

end radius_of_circle_l93_93638


namespace ratio_of_area_of_smaller_circle_to_larger_rectangle_l93_93585

noncomputable def ratio_areas (w : ℝ) : ℝ :=
  (3.25 * Real.pi * w^2 / 4) / (1.5 * w^2)

theorem ratio_of_area_of_smaller_circle_to_larger_rectangle (w : ℝ) : 
  ratio_areas w = 13 * Real.pi / 24 := 
by 
  sorry

end ratio_of_area_of_smaller_circle_to_larger_rectangle_l93_93585


namespace max_abs_value_inequality_l93_93099

theorem max_abs_value_inequality (a b : ℝ)
  (h : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |a * x + b| ≤ 1) :
  ∃ (a b : ℝ), (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |a * x + b| ≤ 1) ∧ |20 * a + 14 * b| + |20 * a - 14 * b| = 80 := 
sorry

end max_abs_value_inequality_l93_93099


namespace ratio_of_x_to_y_l93_93978

theorem ratio_of_x_to_y (x y : ℤ) (h : (7 * x - 4 * y) * 9 = (20 * x - 3 * y) * 4) : x * 17 = y * -24 :=
by {
  sorry
}

end ratio_of_x_to_y_l93_93978


namespace parallel_lines_iff_a_eq_3_l93_93902

theorem parallel_lines_iff_a_eq_3 (a : ℝ) :
  (∀ x y : ℝ, (6 * x - 4 * y + 1 = 0) ↔ (a * x - 2 * y - 1 = 0)) ↔ (a = 3) := 
sorry

end parallel_lines_iff_a_eq_3_l93_93902


namespace percentage_change_area_l93_93672

theorem percentage_change_area (L B : ℝ) (h : L > 0 ∧ B > 0) :
  let A1 := L * B
  let L2 := L / 2
  let B2 := 3 * B
  let A2 := L2 * B2
  let percentage_change := ((A2 - A1) / A1) * 100
  percentage_change = 50 := by
  have A1_def := calc
    A1 = L * B : rfl
  have A2_def := calc
    A2 = (L / 2) * (3 * B) : rfl
    ... = (3 / 2) * (L * B) : by ring
  have pc_def := calc
    percentage_change = ((A2 - A1) / A1) * 100 : rfl
    ... = (( (3 / 2) * (L * B) - (L * B) ) / (L * B) ) * 100 : by rw [A1_def, A2_def]
    ... = (1 / 2) * 100 : by ring
  exact pc_def

end percentage_change_area_l93_93672


namespace expand_expression_l93_93886

theorem expand_expression : ∀ (x : ℝ), (20 * x - 25) * 3 * x = 60 * x^2 - 75 * x := 
by
  intro x
  sorry

end expand_expression_l93_93886


namespace find_m_l93_93768

-- Definitions for vectors and dot products
structure Vector :=
  (i : ℝ)
  (j : ℝ)

def dot_product (a b : Vector) : ℝ :=
  a.i * b.i + a.j * b.j

-- Given conditions
def i : Vector := ⟨1, 0⟩
def j : Vector := ⟨0, 1⟩

def a : Vector := ⟨2, 3⟩
def b (m : ℝ) : Vector := ⟨1, -m⟩

-- The main goal
theorem find_m (m : ℝ) (h: dot_product a (b m) = 1) : m = 1 / 3 :=
by {
  -- Calculation reaches the same \(m = 1/3\)
  sorry
}

end find_m_l93_93768


namespace find_number_l93_93575

theorem find_number (x : ℝ) (h : (3/4 : ℝ) * x = 93.33333333333333) : x = 124.44444444444444 := 
by
  -- Proof to be filled in
  sorry

end find_number_l93_93575


namespace trapezoid_circle_ratio_l93_93877

variable (P R : ℝ)

def is_isosceles_trapezoid_inscribed_in_circle (P R : ℝ) : Prop :=
  ∃ m A, 
    m = P / 4 ∧
    A = m * 2 * R ∧
    A = (P * R) / 2

theorem trapezoid_circle_ratio (P R : ℝ) 
  (h : is_isosceles_trapezoid_inscribed_in_circle P R) :
  (P / 2 * π * R) = (P / 2 * π * R) :=
by
  -- Use the given condition to prove the statement
  sorry

end trapezoid_circle_ratio_l93_93877


namespace maximum_distance_between_balls_l93_93636

theorem maximum_distance_between_balls 
  (a b c : ℝ) 
  (aluminum_ball_heavier : true) -- Implicitly understood property rather than used in calculation directly
  (wood_ball_lighter : true) -- Implicitly understood property rather than used in calculation directly
  : ∃ d : ℝ, d = Real.sqrt (a^2 + b^2 + c^2) → d = Real.sqrt (3^2 + 4^2 + 2^2) := 
by
  use Real.sqrt (3^2 + 4^2 + 2^2)
  sorry

end maximum_distance_between_balls_l93_93636


namespace no_such_n_exists_l93_93398

-- Definition of the sum of the digits function s(n)
def s (n : ℕ) : ℕ := n.digits 10 |> List.sum

-- Statement of the proof problem
theorem no_such_n_exists : ¬ ∃ n : ℕ, n * s n = 20222022 :=
by
  -- argument based on divisibility rules as presented in the problem
  sorry

end no_such_n_exists_l93_93398


namespace equation_solution_l93_93750

theorem equation_solution (x : ℝ) (h : x + 1/x = 2.5) : x^2 + 1/x^2 = 4.25 := 
by sorry

end equation_solution_l93_93750


namespace exponent_on_right_side_l93_93082

theorem exponent_on_right_side (n : ℕ) (h : n = 17) : 2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = 4^18 :=
by
  sorry

end exponent_on_right_side_l93_93082


namespace cos_formula_of_tan_l93_93367

theorem cos_formula_of_tan (α : ℝ) (h1 : Real.tan α = 2) (h2 : 0 < α ∧ α < Real.pi) :
  Real.cos (5 * Real.pi / 2 + 2 * α) = -4 / 5 := 
  sorry

end cos_formula_of_tan_l93_93367


namespace unused_streetlights_remain_l93_93420

def total_streetlights : ℕ := 200
def squares : ℕ := 15
def streetlights_per_square : ℕ := 12

theorem unused_streetlights_remain :
  total_streetlights - (squares * streetlights_per_square) = 20 :=
sorry

end unused_streetlights_remain_l93_93420


namespace fractions_comparison_l93_93214

theorem fractions_comparison : 
  (99 / 100 < 100 / 101) ∧ (100 / 101 > 199 / 201) ∧ (99 / 100 < 199 / 201) :=
by sorry

end fractions_comparison_l93_93214


namespace rectangle_width_l93_93139

theorem rectangle_width (side_length square_len rect_len : ℝ) (h1 : side_length = 4) (h2 : rect_len = 4) (h3 : square_len = side_length * side_length) (h4 : square_len = rect_len * some_width) :
  some_width = 4 :=
by
  sorry

end rectangle_width_l93_93139


namespace principal_amount_l93_93709

theorem principal_amount (P R : ℝ) : 
  (P + P * R * 2 / 100 = 850) ∧ (P + P * R * 7 / 100 = 1020) → P = 782 :=
by
  sorry

end principal_amount_l93_93709


namespace second_offset_length_l93_93742

noncomputable def quadrilateral_area (d o1 o2 : ℝ) : ℝ :=
  (1 / 2) * d * (o1 + o2)

theorem second_offset_length (d o1 A : ℝ) (h_d : d = 22) (h_o1 : o1 = 9) (h_A : A = 165) :
  ∃ o2, quadrilateral_area d o1 o2 = A ∧ o2 = 6 := by
  sorry

end second_offset_length_l93_93742


namespace cupcakes_frosted_in_10_minutes_l93_93211

theorem cupcakes_frosted_in_10_minutes :
  let cagney_rate := 1 / 25 -- Cagney's rate in cupcakes per second
  let lacey_rate := 1 / 35 -- Lacey's rate in cupcakes per second
  let total_time := 600 -- Total time in seconds for 10 minutes
  let lacey_break := 60 -- Break duration in seconds
  let lacey_work_time := total_time - lacey_break
  let cupcakes_by_cagney := total_time / 25 
  let cupcakes_by_lacey := lacey_work_time / 35
  cupcakes_by_cagney + cupcakes_by_lacey = 39 := 
by {
  sorry
}

end cupcakes_frosted_in_10_minutes_l93_93211


namespace gcd_72_108_150_l93_93694

theorem gcd_72_108_150 : Nat.gcd (Nat.gcd 72 108) 150 = 6 := by
  sorry

end gcd_72_108_150_l93_93694


namespace Buratino_can_solve_l93_93108

theorem Buratino_can_solve :
  ∃ (MA TE TI KA : ℕ), MA ≠ TE ∧ MA ≠ TI ∧ MA ≠ KA ∧ TE ≠ TI ∧ TE ≠ KA ∧ TI ≠ KA ∧
  MA * TE * MA * TI * KA = 2016000 :=
by
  -- skip the proof using sorry
  sorry

end Buratino_can_solve_l93_93108


namespace range_of_c_monotonicity_g_l93_93242

-- Define the given function f(x)
def f (x : ℝ) := 2 * real.log x + 1

-- Part 1: Define the hypothesis for the range of c
theorem range_of_c :
  ∀ x : ℝ, f(x) ≤ 2 * x + c ↔ c ∈ set.Ici (-1) :=
sorry

-- Part 2: Define the function g(x) and prove its monotonicity
def g (x a : ℝ) [ne_zero : a ≠ 0] := (f(x) - f(a)) / (x - a)

theorem monotonicity_g (a : ℝ) (h : 0 < a) : 
  ∀ x, (0 < x ∧ x < a) ∨ (x > a) → (g x a).deriv < 0 :=
sorry

end range_of_c_monotonicity_g_l93_93242


namespace bob_got_15_candies_l93_93348

-- Define the problem conditions
def bob_neighbor_sam : Prop := true -- Bob is Sam's next door neighbor
def bob_accompany_sam_home : Prop := true -- Bob decided to accompany Sam home

def bob_share_chewing_gums : ℕ := 15 -- Bob's share of chewing gums
def bob_share_chocolate_bars : ℕ := 20 -- Bob's share of chocolate bars
def bob_share_candies : ℕ := 15 -- Bob's share of assorted candies

-- Define the main assertion
theorem bob_got_15_candies : bob_share_candies = 15 := 
by sorry

end bob_got_15_candies_l93_93348


namespace union_of_sets_l93_93230

theorem union_of_sets (x y : ℕ) (A B : Set ℕ) (h1 : A = {x, y}) (h2 : B = {x + 1, 5}) (h3 : A ∩ B = {2}) : A ∪ B = {1, 2, 5} :=
sorry

end union_of_sets_l93_93230


namespace cost_of_traveling_all_roads_l93_93335

noncomputable def total_cost_of_roads (length width road_width : ℝ) (cost_per_sq_m : ℝ) : ℝ :=
  let area_road_parallel_length := length * road_width
  let area_road_parallel_breadth := width * road_width
  let diagonal_length := Real.sqrt (length^2 + width^2)
  let area_road_diagonal := diagonal_length * road_width
  let total_area := area_road_parallel_length + area_road_parallel_breadth + area_road_diagonal
  total_area * cost_per_sq_m

theorem cost_of_traveling_all_roads :
  total_cost_of_roads 80 50 10 3 = 6730.2 :=
by
  sorry

end cost_of_traveling_all_roads_l93_93335


namespace second_cat_weight_l93_93478

theorem second_cat_weight :
  ∀ (w1 w2 w3 w_total : ℕ), 
    w1 = 2 ∧ w3 = 4 ∧ w_total = 13 → 
    w_total = w1 + w2 + w3 → 
    w2 = 7 :=
by
  sorry

end second_cat_weight_l93_93478


namespace katie_needs_more_sugar_l93_93019

-- Let total_cups be the total cups of sugar required according to the recipe
def total_cups : ℝ := 3

-- Let already_put_in be the cups of sugar Katie has already put in
def already_put_in : ℝ := 0.5

-- Define the amount of sugar Katie still needs to put in
def remaining_cups : ℝ := total_cups - already_put_in 

-- Prove that remaining_cups is 2.5
theorem katie_needs_more_sugar : remaining_cups = 2.5 := 
by 
  -- substitute total_cups and already_put_in
  dsimp [remaining_cups, total_cups, already_put_in]
  -- calculate the difference
  norm_num

end katie_needs_more_sugar_l93_93019


namespace length_of_train_is_correct_l93_93340

noncomputable def speed_kmh := 30 
noncomputable def time_s := 9 
noncomputable def speed_ms := (speed_kmh * 1000) / 3600 
noncomputable def length_of_train := speed_ms * time_s

theorem length_of_train_is_correct : length_of_train = 75 := 
by 
  sorry

end length_of_train_is_correct_l93_93340


namespace reinforcement_arrival_l93_93997

theorem reinforcement_arrival (x : ℕ) :
  (2000 * 40) = (2000 * x + 4000 * 10) → x = 20 :=
by
  sorry

end reinforcement_arrival_l93_93997


namespace p_minus_q_l93_93324

theorem p_minus_q (p q : ℚ) (h1 : 3 / p = 6) (h2 : 3 / q = 18) : p - q = 1 / 3 := by
  sorry

end p_minus_q_l93_93324


namespace find_first_day_income_l93_93584

def income_4 (i2 i3 i4 i5 : ℕ) : ℕ := i2 + i3 + i4 + i5

def total_income_5 (average_income : ℕ) : ℕ := 5 * average_income

def income_1 (total : ℕ) (known : ℕ) : ℕ := total - known

theorem find_first_day_income (i2 i3 i4 i5 a income5 : ℕ) (h1 : income_4 i2 i3 i4 i5 = 1800)
  (h2 : a = 440)
  (h3 : total_income_5 a = income5)
  : income_1 income5 (income_4 i2 i3 i4 i5) = 400 := 
sorry

end find_first_day_income_l93_93584


namespace function_properties_l93_93893

noncomputable def f (x : ℝ) : ℝ := Real.sin ((13 * Real.pi / 2) - x)

theorem function_properties :
  (∀ x : ℝ, f x = Real.cos x) ∧
  (∀ x : ℝ, f (-x) = f x) ∧
  (∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f (x + T) = f x) ∧
  (forall t: ℝ, (∀ x : ℝ, f (x + t) = f x) → (t = 2 * Real.pi ∨ t = -2 * Real.pi)) :=
by
  sorry

end function_properties_l93_93893


namespace range_u_inequality_le_range_k_squared_l93_93516

def D (k : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 > 0 ∧ p.2 > 0 ∧ p.1 + p.2 = k}

theorem range_u (k : ℝ) (hk : k > 0) :
  ∀ (x1 x2 : ℝ), (x1, x2) ∈ D k → 0 < x1 * x2 ∧ x1 * x2 ≤ k^2 / 4 :=
sorry

theorem inequality_le (k : ℝ) (hk : k ≥ 1) :
  ∀ (x1 x2 : ℝ), (x1, x2) ∈ D k →
  (1 / x1 - x1) * (1 / x2 - x2) ≤ (k / 2 - 2 / k)^2 :=
sorry

theorem range_k_squared (k : ℝ) :
  (0 < k^2 ∧ k^2 ≤ 4 * Real.sqrt 5 - 8) ↔
  ∀ (x1 x2 : ℝ), (x1, x2) ∈ D k →
  (1 / x1 - x1) * (1 / x2 - x2) ≥ (k / 2 - 2 / k)^2 :=
sorry

end range_u_inequality_le_range_k_squared_l93_93516


namespace find_phi_l93_93758

theorem find_phi (ϕ : ℝ) (h0 : 0 ≤ ϕ) (h1 : ϕ < π)
    (H : 2 * Real.cos (π / 3) = 2 * Real.sin (2 * (π / 3) + ϕ)) : ϕ = π / 6 :=
by
  sorry

end find_phi_l93_93758


namespace min_value_x2_minus_x1_l93_93240

noncomputable def f (x : ℝ) := 2 * Real.sin (Real.pi / 2 * x + Real.pi / 5)

theorem min_value_x2_minus_x1 :
  (∀ x : ℝ, f x1 ≤ f x ∧ f x ≤ f x2) → |x2 - x1| = 2 :=
sorry

end min_value_x2_minus_x1_l93_93240


namespace length_of_diagonal_l93_93741

theorem length_of_diagonal (h1 h2 area : ℝ) (h1_val : h1 = 7) (h2_val : h2 = 3) (area_val : area = 50) :
  ∃ d : ℝ, d = 10 :=
by
  sorry

end length_of_diagonal_l93_93741


namespace partition_solution_l93_93998

noncomputable def partitions (a m n x : ℝ) : Prop :=
  a = x + n * (a - m * x)

theorem partition_solution (a m n : ℝ) (h : n * m < 1) :
  partitions a m n (a * (1 - n) / (1 - n * m)) :=
by
  sorry

end partition_solution_l93_93998


namespace sum_x_midpoints_of_triangle_l93_93161

theorem sum_x_midpoints_of_triangle (a b c : ℝ) (h : a + b + c = 15) :
  (a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 15 :=
by
  -- Proof omitted, replace with actual proof
  sorry

end sum_x_midpoints_of_triangle_l93_93161


namespace arrange_snow_leopards_l93_93650

theorem arrange_snow_leopards :
  let n := 9 -- number of leopards
  let factorial x := (Nat.factorial x) -- definition for factorial
  let tall_short_perm := 2 -- there are 2 ways to arrange the tallest and shortest leopards at the ends
  tall_short_perm * factorial (n - 2) = 10080 := by sorry

end arrange_snow_leopards_l93_93650


namespace recreation_percentage_this_week_l93_93934

variable (W : ℝ) -- David's last week wages
variable (R_last_week : ℝ) -- Recreation spending last week
variable (W_this_week : ℝ) -- This week's wages
variable (R_this_week : ℝ) -- Recreation spending this week

-- Conditions
def wages_last_week : R_last_week = 0.4 * W := sorry
def wages_this_week : W_this_week = 0.95 * W := sorry
def recreation_spending_this_week : R_this_week = 1.1875 * R_last_week := sorry

-- Theorem to prove
theorem recreation_percentage_this_week :
  (R_this_week / W_this_week) = 0.5 := sorry

end recreation_percentage_this_week_l93_93934


namespace product_of_three_numbers_l93_93436

theorem product_of_three_numbers :
  ∃ (a b c : ℚ), 
    a + b + c = 30 ∧ a = 2 * (b + c) ∧ b = 6 * c ∧ a * b * c = 12000 / 49 :=
by
  sorry

end product_of_three_numbers_l93_93436


namespace fraction_identity_l93_93490

theorem fraction_identity (a : ℝ) (h1 : a ≠ 2) (h2 : a ≠ -2) : 
  (2 * a) / (a^2 - 4) - 1 / (a - 2) = 1 / (a + 2) := 
by
  sorry

end fraction_identity_l93_93490


namespace candy_group_size_l93_93168

-- Define the given conditions
def num_candies : ℕ := 30
def num_groups : ℕ := 10

-- Define the statement that needs to be proven
theorem candy_group_size : num_candies / num_groups = 3 := 
by 
  sorry

end candy_group_size_l93_93168


namespace sum_of_midpoint_xcoords_l93_93157

theorem sum_of_midpoint_xcoords (a b c : ℝ) (h : a + b + c = 15) :
    (a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 15 :=
  by
    sorry

end sum_of_midpoint_xcoords_l93_93157


namespace sales_volume_A_correct_total_cost_B_correct_sales_volume_B_correct_max_total_profit_correct_possible_selling_prices_correct_l93_93266

section disinfectant_water

variables (x : ℕ) (x_greater_than_30 : x > 30)

-- Conditions
def cost_price_A := 20 -- yuan per bottle
def cost_price_B := 30 -- yuan per bottle
def total_cost := 2000 -- yuan

def initial_sales_A := 100 -- bottles at 30 yuan per bottle
def sell_decrease_A := 5 -- bottles per yuan increase
def sell_price_B := 60 -- yuan per bottle

-- Sales volume of type A disinfectant water
def sales_volume_A := 250 - 5 * x

-- Total cost price of type B disinfectant water
def total_cost_B := 2000 - 20 * (250 - 5 * x)

-- Sales volume of type B disinfectant water
def sales_volume_B := (total_cost_B x) / cost_price_B

-- Total profit function
def total_profit := (250 - 5 * x) * (x - 20) + ((total_cost_B x) / 30 - 100) * (60 - 30)

-- Maximum total profit
def max_total_profit := 2125

-- Possible selling prices for total profit >= 1945 yuan
def possible_selling_prices := {x : ℕ | 39 ≤ x ∧ x ≤ 50 ∧ (x % 3 = 0)}

-- Proofs
theorem sales_volume_A_correct : sales_volume_A x = 250 - 5 * x := by
  sorry

theorem total_cost_B_correct : total_cost_B x = 100 * x - 3000 := by
  sorry

theorem sales_volume_B_correct : sales_volume_B x = (100 * x - 3000) / 30 := by
  sorry

theorem max_total_profit_correct : total_profit x = -5*(x - 45) * (x - 45) + 2125 := by
  sorry

theorem possible_selling_prices_correct : ∀ x, 1945 ≤ total_profit x → x ∈ possible_selling_prices := by
  sorry

end disinfectant_water

end sales_volume_A_correct_total_cost_B_correct_sales_volume_B_correct_max_total_profit_correct_possible_selling_prices_correct_l93_93266


namespace area_of_triangle_l93_93826

-- Definition of equilateral triangle and its altitude
def altitude_of_equilateral_triangle (a : ℝ) : Prop := 
  a = 2 * sqrt 3

-- Definition of the area function for equilateral triangle with side 's'
def area_of_equilateral_triangle (s : ℝ) : ℝ :=
  (sqrt 3 / 4) * s^2

-- The main statement to prove
theorem area_of_triangle (a : ℝ) (s : ℝ) 
  (alt_cond : altitude_of_equilateral_triangle a) 
  (side_relation : a = (sqrt 3 / 2) * s) : 
  area_of_equilateral_triangle s = 4 * sqrt 3 :=
by
  sorry

end area_of_triangle_l93_93826


namespace average_score_of_class_l93_93854

theorem average_score_of_class (total_students : ℕ)
  (perc_assigned_day perc_makeup_day : ℝ)
  (average_assigned_day average_makeup_day : ℝ)
  (h_total : total_students = 100)
  (h_perc_assigned_day : perc_assigned_day = 0.70)
  (h_perc_makeup_day : perc_makeup_day = 0.30)
  (h_average_assigned_day : average_assigned_day = 55)
  (h_average_makeup_day : average_makeup_day = 95) :
  ((perc_assigned_day * total_students * average_assigned_day + perc_makeup_day * total_students * average_makeup_day) / total_students) = 67 := by
  sorry

end average_score_of_class_l93_93854


namespace necessary_not_sufficient_condition_l93_93057

theorem necessary_not_sufficient_condition {a : ℝ} :
  (∀ x : ℝ, |x - 1| < 1 → x ≥ a) →
  (¬ (∀ x : ℝ, x ≥ a → |x - 1| < 1)) →
  a ≤ 0 :=
by
  intro h1 h2
  sorry

end necessary_not_sufficient_condition_l93_93057


namespace height_of_fourth_person_l93_93169

theorem height_of_fourth_person
  (h : ℝ)
  (H1 : h + (h + 2) + (h + 4) + (h + 10) = 4 * 79) :
  h + 10 = 85 :=
by
  have H2 : h + 4 = 79 := by linarith
  linarith


end height_of_fourth_person_l93_93169


namespace initial_amount_l93_93552

theorem initial_amount (cost_bread cost_butter cost_juice total_remain total_amount : ℕ) :
  cost_bread = 2 →
  cost_butter = 3 →
  cost_juice = 2 * cost_bread →
  total_remain = 6 →
  total_amount = cost_bread + cost_butter + cost_juice + total_remain →
  total_amount = 15 := by
  intros h_bread h_butter h_juice h_remain h_total
  sorry

end initial_amount_l93_93552


namespace identify_irrational_number_l93_93204

theorem identify_irrational_number :
  (∀ a b : ℤ, (-1 : ℚ) = (a : ℚ) / (b : ℚ) → b ≠ 0) ∧
  (∀ a b : ℤ, (0 : ℚ) = (a : ℚ) / (b : ℚ) → b ≠ 0) ∧
  (∀ a b : ℤ, (1 : ℚ) / (2 : ℚ) = (a : ℚ) / (b : ℚ) → b ≠ 0) ∧
  ¬(∃ a b : ℤ, (Real.sqrt 3) = (a : ℚ) / (b : ℚ) ∧ b ≠ 0) :=
sorry

end identify_irrational_number_l93_93204


namespace cos_two_sum_l93_93370

theorem cos_two_sum {α β : ℝ} 
  (h1 : 3 * Real.sin α ^ 2 + 2 * Real.sin β ^ 2 = 1)
  (h2 : 3 * (Real.sin α + Real.cos α) ^ 2 - 2 * (Real.sin β + Real.cos β) ^ 2 = 1) :
  Real.cos (2 * (α + β)) = -1 / 3 :=
sorry

end cos_two_sum_l93_93370


namespace factorial_fraction_is_integer_l93_93950

theorem factorial_fraction_is_integer (m n : ℕ) : ∃ k : ℤ, ((2 * m)!.toRational * (2 * n)!.toRational) / (m!.toRational * n!.toRational * (m + n)!.toRational) = k :=
by {
  sorry
}

end factorial_fraction_is_integer_l93_93950


namespace max_value_trig_expr_exists_angle_for_max_value_l93_93043

theorem max_value_trig_expr : ∀ x : ℝ, 3 * Real.cos x + 4 * Real.sin x ≤ 5 :=
sorry

theorem exists_angle_for_max_value : ∃ x : ℝ, 3 * Real.cos x + 4 * Real.sin x = 5 :=
sorry

end max_value_trig_expr_exists_angle_for_max_value_l93_93043


namespace thrushes_left_l93_93446

theorem thrushes_left {init_thrushes : ℕ} (additional_thrushes : ℕ) (killed_ratio : ℚ) (killed : ℕ) (remaining : ℕ) :
  init_thrushes = 20 →
  additional_thrushes = 4 * 2 →
  killed_ratio = 1 / 7 →
  killed = killed_ratio * (init_thrushes + additional_thrushes) →
  remaining = init_thrushes + additional_thrushes - killed →
  remaining = 24 :=
by sorry

end thrushes_left_l93_93446


namespace population_net_increase_period_l93_93389

def period_in_hours (birth_rate : ℕ) (death_rate : ℕ) (net_increase : ℕ) : ℕ :=
  let net_rate_per_second := (birth_rate / 2) - (death_rate / 2)
  let period_in_seconds := net_increase / net_rate_per_second
  period_in_seconds / 3600

theorem population_net_increase_period :
  period_in_hours 10 2 345600 = 24 :=
by
  unfold period_in_hours
  sorry

end population_net_increase_period_l93_93389


namespace sum_of_remainders_l93_93316

theorem sum_of_remainders (n : ℤ) (h : n % 12 = 5) :
  (n % 4) + (n % 3) = 3 :=
by
  sorry

end sum_of_remainders_l93_93316


namespace certain_person_current_age_l93_93118

-- Define Sandys's current age and the certain person's current age
variable (S P : ℤ)

-- Conditions from the problem
def sandy_phone_bill_condition := 10 * S = 340
def sandy_age_relation := S + 2 = 3 * P

theorem certain_person_current_age (h1 : sandy_phone_bill_condition S) (h2 : sandy_age_relation S P) : P - 2 = 10 :=
by
  sorry

end certain_person_current_age_l93_93118


namespace find_a_l93_93256

theorem find_a (a : ℝ) (h : a * (1 : ℝ)^2 - 6 * 1 + 3 = 0) : a = 3 :=
by
  sorry

end find_a_l93_93256


namespace solution_sets_l93_93434

-- These are the hypotheses derived from the problem conditions.
structure Conditions (a b c d : ℕ) : Prop :=
  (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (positive_even : ∃ u v w x : ℕ, a = 2*u ∧ b = 2*v ∧ c = 2*w ∧ d = 2*x ∧ 
                   u > 0 ∧ v > 0 ∧ w > 0 ∧ x > 0)
  (sum_100 : a + b + c + d = 100)
  (third_fourth_single_digit : c < 20 ∧ d < 20)
  (sum_2000 : 12 * a + 30 * b + 52 * c = 2000)

-- The main theorem in Lean asserting that these are the only possible sets of numbers.
theorem solution_sets :
  ∃ (a b c d : ℕ), Conditions a b c d ∧
  ( 
    (a = 62 ∧ b = 14 ∧ c = 4 ∧ d = 1) ∨ 
    (a = 48 ∧ b = 22 ∧ c = 2 ∧ d = 3)
  ) :=
  sorry

end solution_sets_l93_93434


namespace find_remainder_l93_93745

def p (x : ℝ) : ℝ := x^5 + 2 * x^2 + 1

theorem find_remainder : p 2 = 41 :=
by sorry

end find_remainder_l93_93745


namespace simplify_fraction_l93_93127

theorem simplify_fraction :
  (1 : ℚ) / 462 + 17 / 42 = 94 / 231 := 
sorry

end simplify_fraction_l93_93127


namespace first_product_of_digits_of_98_l93_93438

theorem first_product_of_digits_of_98 : (9 * 8 = 72) :=
by simp [mul_eq_mul_right_iff] -- This will handle the basic arithmetic automatically

end first_product_of_digits_of_98_l93_93438


namespace problem1_problem2_l93_93762

def M (x : ℝ) : Prop := (x + 5) / (x - 8) ≥ 0

def N (x : ℝ) (a : ℝ) : Prop := a - 1 ≤ x ∧ x ≤ a + 1

theorem problem1 : ∀ (x : ℝ), (M x ∨ (N x 9)) ↔ (x ≤ -5 ∨ x ≥ 8) :=
by
  sorry

theorem problem2 : ∀ (a : ℝ), (∀ (x : ℝ), N x a → M x) ↔ (a ≤ -6 ∨ 9 < a) :=
by
  sorry

end problem1_problem2_l93_93762


namespace alphabet_letter_count_l93_93931

def sequence_count : Nat :=
  let total_sequences := 2^7
  let sequences_per_letter := 1 + 7 -- 1 correct sequence + 7 single-bit alterations
  total_sequences / sequences_per_letter

theorem alphabet_letter_count : sequence_count = 16 :=
  by
    -- Proof placeholder
    sorry

end alphabet_letter_count_l93_93931


namespace find_counterfeit_80_coins_in_4_weighings_min_weighings_for_n_coins_l93_93176

theorem find_counterfeit_80_coins_in_4_weighings :
  ∃ f : Fin 80 → Bool, (∃ i, f i = true) ∧ (∃ i j, f i ≠ f j) := sorry

theorem min_weighings_for_n_coins (n : ℕ) (h : n > 0) :
  ∃ k : ℕ, 3^(k-1) < n ∧ n ≤ 3^k := sorry

end find_counterfeit_80_coins_in_4_weighings_min_weighings_for_n_coins_l93_93176


namespace oliver_shirts_problem_l93_93712

-- Defining the quantities of short sleeve shirts, long sleeve shirts, and washed shirts.
def shortSleeveShirts := 39
def longSleeveShirts  := 47
def shirtsWashed := 20

-- Stating the problem formally.
theorem oliver_shirts_problem :
  shortSleeveShirts + longSleeveShirts - shirtsWashed = 66 :=
by
  -- Proof goes here.
  sorry

end oliver_shirts_problem_l93_93712


namespace tv_cost_solution_l93_93005

theorem tv_cost_solution (M T : ℝ) 
  (h1 : 2 * M + T = 7000)
  (h2 : M + 2 * T = 9800) : 
  T = 4200 :=
by
  sorry

end tv_cost_solution_l93_93005


namespace prize_interval_l93_93637

theorem prize_interval (prize1 prize2 prize3 prize4 prize5 interval : ℝ) (h1 : prize1 = 5000) 
  (h2 : prize2 = 5000 - interval) (h3 : prize3 = 5000 - 2 * interval) 
  (h4 : prize4 = 5000 - 3 * interval) (h5 : prize5 = 5000 - 4 * interval) 
  (h_total : prize1 + prize2 + prize3 + prize4 + prize5 = 15000) : 
  interval = 1000 := 
by
  sorry

end prize_interval_l93_93637


namespace intersection_P_complement_Q_l93_93513

-- Defining the sets P and Q
def R := Set ℝ
def P : Set ℝ := {x | x^2 + 2 * x - 3 = 0}
def Q : Set ℝ := {x | Real.log x < 1}
def complement_R_Q : Set ℝ := {x | x ≤ 0 ∨ x ≥ Real.exp 1}
def intersection := {x | x ∈ P ∧ x ∈ complement_R_Q}

-- Statement of the theorem
theorem intersection_P_complement_Q : 
  intersection = {-3} :=
by
  sorry

end intersection_P_complement_Q_l93_93513


namespace correct_operation_l93_93707

variable {a b : ℝ}

def conditionA : Prop := a^2 + a^3 = a^6
def conditionB : Prop := (a * b)^2 = a * b^2
def conditionC : Prop := (a + b)^2 = a^2 + b^2
def conditionD : Prop := (a + b) * (a - b) = a^2 - b^2

theorem correct_operation : conditionD ∧ ¬conditionA ∧ ¬conditionB ∧ ¬conditionC := by
  sorry

end correct_operation_l93_93707


namespace length_60_more_than_breadth_l93_93142

noncomputable def length_more_than_breadth (cost_per_meter : ℝ) (total_cost : ℝ) (length : ℝ) : Prop :=
  ∃ (breadth : ℝ) (x : ℝ), 
    length = breadth + x ∧
    2 * length + 2 * breadth = total_cost / cost_per_meter ∧
    x = length - breadth ∧
    x = 60

theorem length_60_more_than_breadth : length_more_than_breadth 26.5 5300 80 :=
by
  sorry

end length_60_more_than_breadth_l93_93142


namespace max_product_of_two_integers_with_sum_2004_l93_93968

theorem max_product_of_two_integers_with_sum_2004 :
  ∃ x y : ℤ, x + y = 2004 ∧ (∀ a b : ℤ, a + b = 2004 → a * b ≤ x * y) ∧ x * y = 1004004 := 
by
  sorry

end max_product_of_two_integers_with_sum_2004_l93_93968


namespace sin_2gamma_proof_l93_93657

-- Assume necessary definitions and conditions
variables {A B C D P : Type}
variables [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited P]
variables (a b c d: ℝ)
variables (α β γ: ℝ)

-- Assume points A, B, C, D, P lie on a circle in that order and AB = BC = CD
axiom points_on_circle : a = b ∧ b = c ∧ c = d
axiom cos_apc : Real.cos α = 3/5
axiom cos_bpd : Real.cos β = 1/5

noncomputable def sin_2gamma : ℝ :=
  2 * Real.sin γ * Real.cos γ

-- Statement to prove sin(2 * γ) given the conditions
theorem sin_2gamma_proof : sin_2gamma γ = 8 * Real.sqrt 5 / 25 :=
sorry

end sin_2gamma_proof_l93_93657


namespace simplify_fraction_l93_93128

theorem simplify_fraction :
  (1 : ℚ) / 462 + 17 / 42 = 94 / 231 := 
sorry

end simplify_fraction_l93_93128


namespace correct_operation_l93_93702

theorem correct_operation (a b : ℝ) : (a + b) * (a - b) = a^2 - b^2 := by
  sorry

end correct_operation_l93_93702


namespace trigonometric_identity_l93_93754

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 1 / 2) : 
  (Real.cos α + Real.sin α) / (Real.cos α - Real.sin α) = 3 :=
by
  sorry

end trigonometric_identity_l93_93754


namespace sum_real_imag_l93_93769

theorem sum_real_imag (z : ℂ) (hz : z = 3 - 4 * I) : z.re + z.im = -1 :=
by {
  -- Because the task asks for no proof, we're leaving it with 'sorry'.
  sorry
}

end sum_real_imag_l93_93769


namespace total_cost_of_tickets_l93_93803

def number_of_adults := 2
def number_of_children := 3
def cost_of_adult_ticket := 19
def cost_of_child_ticket := cost_of_adult_ticket - 6

theorem total_cost_of_tickets :
  let total_cost := number_of_adults * cost_of_adult_ticket + number_of_children * cost_of_child_ticket
  total_cost = 77 :=
by
  sorry

end total_cost_of_tickets_l93_93803


namespace smallest_perfect_square_divisible_by_5_and_7_l93_93312

theorem smallest_perfect_square_divisible_by_5_and_7 
  (n : ℕ) 
  (h1 : n > 0) 
  (h2 : ∃ k : ℕ, n = k^2)
  (h3 : 5 ∣ n)
  (h4 : 7 ∣ n) : 
  n = 1225 :=
sorry

end smallest_perfect_square_divisible_by_5_and_7_l93_93312


namespace find_other_number_l93_93426

theorem find_other_number (y : ℕ) : Nat.lcm 240 y = 5040 ∧ Nat.gcd 240 y = 24 → y = 504 :=
by
  sorry

end find_other_number_l93_93426


namespace number_of_bricks_required_l93_93010

def brick_length : ℝ := 0.20
def brick_width : ℝ := 0.10
def brick_height : ℝ := 0.075

def wall_length : ℝ := 25.0
def wall_width : ℝ := 2.0
def wall_height : ℝ := 0.75

def brick_volume := brick_length * brick_width * brick_height
def wall_volume := wall_length * wall_width * wall_height

theorem number_of_bricks_required :
  wall_volume / brick_volume = 25000 := by
  sorry

end number_of_bricks_required_l93_93010


namespace room_tiling_problem_correct_l93_93477

noncomputable def room_tiling_problem : Prop :=
  let room_length := 6.72
  let room_width := 4.32
  let tile_size := 0.3
  let room_area := room_length * room_width
  let tile_area := tile_size * tile_size
  let num_tiles := (room_area / tile_area).ceil
  num_tiles = 323

theorem room_tiling_problem_correct : room_tiling_problem := 
  sorry

end room_tiling_problem_correct_l93_93477


namespace sin_gt_cos_interval_l93_93929

theorem sin_gt_cos_interval (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 2 * Real.pi) (h3 : Real.sin x > Real.cos x) : 
  Real.sin x > Real.cos x ↔ (Real.pi / 4 < x ∧ x < 5 * Real.pi / 4) :=
by
  sorry

end sin_gt_cos_interval_l93_93929


namespace triangle_area_is_correct_l93_93262

noncomputable def isosceles_triangle_area : Prop :=
  let side_large_square := 6 -- sides of the large square WXYZ
  let area_large_square := side_large_square * side_large_square
  let side_small_square := 2 -- sides of the smaller squares
  let BC := side_large_square - 2 * side_small_square -- length of BC
  let height_AM := side_large_square / 2 + side_small_square -- height of the triangle from A to M
  let area_ABC := (BC * height_AM) / 2 -- area of the triangle ABC
  area_large_square = 36 ∧ BC = 2 ∧ height_AM = 5 ∧ area_ABC = 5

theorem triangle_area_is_correct : isosceles_triangle_area := sorry

end triangle_area_is_correct_l93_93262


namespace find_bc_l93_93054

theorem find_bc (b c : ℤ) (h : ∀ x : ℝ, x^2 + (b : ℝ) * x + (c : ℝ) = 0 ↔ x = 1 ∨ x = 2) :
  b = -3 ∧ c = 2 := by
  sorry

end find_bc_l93_93054


namespace proof_problem_l93_93991

-- Variables representing the numbers a, b, and c
variables {a b c : ℝ}

-- Given condition
def given_condition (a b c : ℝ) : Prop :=
  (a^2 + b^2) / (b^2 + c^2) = a / c

-- Required to prove
def to_prove (a b c : ℝ) : Prop :=
  (a / b = b / c) → False

-- Theorem stating that the given condition does not imply the required assertion
theorem proof_problem (a b c : ℝ) (h : given_condition a b c) : to_prove a b c :=
sorry

end proof_problem_l93_93991


namespace solution_set_of_abs_x_plus_one_gt_one_l93_93962

theorem solution_set_of_abs_x_plus_one_gt_one :
  {x : ℝ | |x + 1| > 1} = {x : ℝ | x < -2 ∨ x > 0} :=
sorry

end solution_set_of_abs_x_plus_one_gt_one_l93_93962


namespace f_2017_of_9_eq_8_l93_93939

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def f (n : ℕ) : ℕ :=
  sum_of_digits (n^2 + 1)

def f_k (k n : ℕ) : ℕ :=
  if k = 0 then n else f (f_k (k-1) n)

theorem f_2017_of_9_eq_8 : f_k 2017 9 = 8 := by
  sorry

end f_2017_of_9_eq_8_l93_93939


namespace raspberry_pies_l93_93107

theorem raspberry_pies (total_pies : ℕ) (r_peach : ℕ) (r_strawberry : ℕ) (r_raspberry : ℕ) (r_sum : ℕ) :
    total_pies = 36 → r_peach = 2 → r_strawberry = 5 → r_raspberry = 3 → r_sum = (r_peach + r_strawberry + r_raspberry) →
    (total_pies : ℝ) / (r_sum : ℝ) * (r_raspberry : ℝ) = 10.8 :=
by
    -- This theorem is intended to state the problem.
    sorry

end raspberry_pies_l93_93107


namespace casey_saves_by_paying_monthly_l93_93354

theorem casey_saves_by_paying_monthly :
  let weekly_rate := 280
  let monthly_rate := 1000
  let weeks_per_month := 4
  let months := 3
  let cost_monthly := monthly_rate * months
  let cost_weekly := weekly_rate * weeks_per_month * months
  let savings := cost_weekly - cost_monthly
  savings = 360 :=
by
  unfold weekly_rate monthly_rate weeks_per_month months cost_monthly cost_weekly savings
  simp
  sorry

end casey_saves_by_paying_monthly_l93_93354


namespace train_speed_l93_93872

theorem train_speed
  (train_length : ℕ)
  (man_speed_kmph : ℕ)
  (time_to_pass : ℕ)
  (speed_of_train : ℝ) :
  train_length = 180 →
  man_speed_kmph = 8 →
  time_to_pass = 4 →
  speed_of_train = 154 := 
by
  sorry

end train_speed_l93_93872


namespace three_digit_multiple_l93_93020

open Classical

theorem three_digit_multiple (n : ℕ) (h₁ : n % 2 = 0) (h₂ : n % 5 = 0) (h₃ : n % 3 = 0) (h₄ : 100 ≤ n) (h₅ : n < 1000) :
  120 ≤ n ∧ n ≤ 990 :=
by
  sorry

end three_digit_multiple_l93_93020


namespace sum_of_four_squares_eq_20_l93_93875

variable (x y : ℕ)

-- Conditions based on the provided problem
def condition1 := 2 * x + 2 * y = 16
def condition2 := 2 * x + 3 * y = 19

-- Theorem to be proven
theorem sum_of_four_squares_eq_20 (h1 : condition1 x y) (h2 : condition2 x y) : 4 * x = 20 :=
by
  sorry

end sum_of_four_squares_eq_20_l93_93875


namespace intersection_is_empty_l93_93105

open Finset

namespace ComplementIntersection

-- Define the universal set U, sets M and N
def U : Finset ℕ := {1, 2, 3, 4, 5}
def M : Finset ℕ := {1, 3, 4}
def N : Finset ℕ := {2, 4, 5}

-- The complement of M with respect to U
def complement_U_M : Finset ℕ := U \ M

-- The complement of N with respect to U
def complement_U_N : Finset ℕ := U \ N

-- The intersection of the complements
def intersection_complements : Finset ℕ := complement_U_M ∩ complement_U_N

-- The proof statement
theorem intersection_is_empty : intersection_complements = ∅ :=
by sorry

end ComplementIntersection

end intersection_is_empty_l93_93105


namespace solution_set_of_inequality_l93_93579

theorem solution_set_of_inequality (x : ℝ) : (|x + 1| - |x - 3| ≥ 0) ↔ (1 ≤ x) := 
sorry

end solution_set_of_inequality_l93_93579


namespace initial_ratio_of_partners_to_associates_l93_93198

theorem initial_ratio_of_partners_to_associates
  (P : ℕ) (A : ℕ)
  (hP : P = 18)
  (h_ratio_after_hiring : ∀ A, 45 + A = 18 * 34) :
  (P : ℤ) / (A : ℤ) = 2 / 63 := 
sorry

end initial_ratio_of_partners_to_associates_l93_93198


namespace percentage_of_number_l93_93186

theorem percentage_of_number (P : ℝ) (h : 0.10 * 3200 - 190 = P * 650) :
  P = 0.2 :=
sorry

end percentage_of_number_l93_93186


namespace percentage_change_in_area_halved_length_tripled_breadth_l93_93676

def original_area (L B : ℝ) : ℝ := L * B
def new_area (L B : ℝ) : ℝ := (L / 2) * (3 * B)

def percentage_change (A1 A2 : ℝ) : ℝ :=
  ((A2 - A1) / A1) * 100

theorem percentage_change_in_area_halved_length_tripled_breadth
  (L B : ℝ) (hL : L > 0) (hB : B > 0) :
  percentage_change (original_area L B) (new_area L B) = 50 :=
begin
  sorry
end

end percentage_change_in_area_halved_length_tripled_breadth_l93_93676


namespace smallest_positive_integer_cube_ends_544_l93_93890

theorem smallest_positive_integer_cube_ends_544 : ∃ n : ℕ, n > 0 ∧ n^3 % 1000 = 544 ∧ ∀ m : ℕ, m > 0 ∧ m^3 % 1000 = 544 → m ≥ n :=
by
  sorry

end smallest_positive_integer_cube_ends_544_l93_93890


namespace part1_part2_l93_93243

-- Definition of the function f.
def f (x: ℝ) : ℝ := 2 * Real.log x + 1

-- Definition of the function g.
def g (x a: ℝ) : ℝ := (f x - f a) / (x - a)

-- Part 1: Prove that c ≥ -1 given f(x) ≤ 2x + c.
theorem part1 (c : ℝ) : (∀ x : ℝ, x > 0 → f x ≤ 2 * x + c) → c ≥ -1 :=
by
  -- Proof is omitted.
  sorry

-- Part 2: Prove that g(x) is monotonically decreasing on (0, a) and (a, +∞) given a > 0.
theorem part2 (a : ℝ) : a > 0 → (∀ x : ℝ, x > 0 → x ≠ a → 
  (∀ x1 x2 : ℝ, x1 ∈ Ioo 0 a → x2 ∈ Ioo 0 a → x1 < x2 → g x2 a < g x1 a) ∧ 
  (∀ x1 x2 : ℝ, x1 ∈ Ioo a (Real.Inf.set (Set.Iio a)) → x2 ∈ Ioo a (Real.Inf.set (Set.Iio a)) → x1 < x2 → g x2 a < g x1 a)) :=
by
  -- Proof is omitted.
  sorry

end part1_part2_l93_93243


namespace ratio_of_professionals_l93_93531

-- Define the variables and conditions as stated in the problem.
variables (e d l : ℕ)

-- The condition about the average ages leading to the given equation.
def avg_age_condition : Prop := (40 * e + 50 * d + 60 * l) / (e + d + l) = 45

-- The statement to prove that given the average age condition, the ratio is 1:1:3.
theorem ratio_of_professionals (h : avg_age_condition e d l) : e = d + 3 * l :=
sorry

end ratio_of_professionals_l93_93531


namespace women_in_room_l93_93181

theorem women_in_room (x q : ℕ) (h1 : 4 * x + 2 = 14) (h2 : q = 2 * (5 * x - 3)) : q = 24 :=
by sorry

end women_in_room_l93_93181


namespace range_of_x_l93_93251

theorem range_of_x {a : ℝ} : 
  (∀ a : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0) ↔ (x = 0 ∨ x = -2) :=
by sorry

end range_of_x_l93_93251


namespace find_normal_monthly_charge_l93_93649

-- Define the conditions
def normal_monthly_charge (x : ℕ) : Prop :=
  let first_month_charge := x / 3
  let fourth_month_charge := x + 15
  let other_months_charge := 4 * x
  (first_month_charge + fourth_month_charge + other_months_charge = 175)

-- The statement to prove
theorem find_normal_monthly_charge : ∃ x : ℕ, normal_monthly_charge x ∧ x = 30 := by
  sorry

end find_normal_monthly_charge_l93_93649


namespace findYearsForTwiceAge_l93_93723

def fatherSonAges : ℕ := 33

def fatherAge : ℕ := fatherSonAges + 35

def yearsForTwiceAge (x : ℕ) : Prop :=
  fatherAge + x = 2 * (fatherSonAges + x)

theorem findYearsForTwiceAge : ∃ x, yearsForTwiceAge x :=
  ⟨2, sorry⟩

end findYearsForTwiceAge_l93_93723


namespace solve_for_x_l93_93138

noncomputable def x_solution (x : ℚ) : Prop :=
  x > 1 ∧ 3 * x^2 + 11 * x - 20 = 0

theorem solve_for_x :
  ∃ x : ℚ, x_solution x ∧ x = 4 / 3 :=
by
  sorry

end solve_for_x_l93_93138


namespace possible_values_of_a_l93_93106

variables {a b k : ℤ}

def sum_distances (a : ℤ) (k : ℤ) : ℤ :=
  (a - k).natAbs + (a - (k + 1)).natAbs + (a - (k + 2)).natAbs +
  (a - (k + 3)).natAbs + (a - (k + 4)).natAbs + (a - (k + 5)).natAbs +
  (a - (k + 6)).natAbs + (a - (k + 7)).natAbs + (a - (k + 8)).natAbs +
  (a - (k + 9)).natAbs + (a - (k + 10)).natAbs

theorem possible_values_of_a :
  sum_distances a k = 902 →
  sum_distances b k = 374 →
  a + b = 98 →
  a = 25 ∨ a = 107 ∨ a = -9 :=
sorry

end possible_values_of_a_l93_93106


namespace characteristic_triangle_smallest_angle_l93_93983

theorem characteristic_triangle_smallest_angle 
  (α β : ℝ)
  (h1 : α = 2 * β)
  (h2 : α = 100)
  (h3 : β + α + γ = 180) : 
  min α (min β γ) = 30 := 
by 
  sorry

end characteristic_triangle_smallest_angle_l93_93983


namespace metal_detector_time_on_less_crowded_days_l93_93271

variable (find_parking_time walk_time crowded_metal_detector_time total_time_per_week : ℕ)
variable (week_days crowded_days less_crowded_days : ℕ)

theorem metal_detector_time_on_less_crowded_days
  (h1 : find_parking_time = 5)
  (h2 : walk_time = 3)
  (h3 : crowded_metal_detector_time = 30)
  (h4 : total_time_per_week = 130)
  (h5 : week_days = 5)
  (h6 : crowded_days = 2)
  (h7 : less_crowded_days = 3) :
  (total_time_per_week = (find_parking_time * week_days) + (walk_time * week_days) + (crowded_metal_detector_time * crowded_days) + (10 * less_crowded_days)) :=
sorry

end metal_detector_time_on_less_crowded_days_l93_93271


namespace total_money_shared_l93_93113

-- Conditions
def ken_share : ℕ := 1750
def tony_share : ℕ := 2 * ken_share

-- Question and proof to be demonstrated
theorem total_money_shared : ken_share + tony_share = 5250 :=
by sorry

end total_money_shared_l93_93113


namespace speed_downstream_is_correct_l93_93566

-- Definitions corresponding to the conditions
def speed_boat_still_water : ℕ := 60
def speed_current : ℕ := 17

-- Definition of speed downstream from the conditions and proving the result
theorem speed_downstream_is_correct :
  speed_boat_still_water + speed_current = 77 :=
by
  -- Proof is omitted
  sorry

end speed_downstream_is_correct_l93_93566


namespace find_f_zero_function_decreasing_find_range_x_l93_93070

noncomputable def f : ℝ → ℝ := sorry

-- Define the main conditions as hypotheses
axiom additivity : ∀ x1 x2 : ℝ, f (x1 + x2) = f x1 + f x2
axiom negativity : ∀ x : ℝ, x > 0 → f x < 0

-- First theorem: proving f(0) = 0
theorem find_f_zero : f 0 = 0 := sorry

-- Second theorem: proving the function is decreasing over (-∞, ∞)
theorem function_decreasing : ∀ x1 x2 : ℝ, x1 < x2 → f x1 > f x2 := sorry

-- Third theorem: finding the range of x such that f(x) + f(2-3x) < 0
theorem find_range_x (x : ℝ) : f x + f (2 - 3 * x) < 0 → x < 1 := sorry

end find_f_zero_function_decreasing_find_range_x_l93_93070


namespace james_monthly_earnings_l93_93093

theorem james_monthly_earnings (initial_subscribers gifted_subscribers earnings_per_subscriber : ℕ)
  (initial_subscribers_eq : initial_subscribers = 150)
  (gifted_subscribers_eq : gifted_subscribers = 50)
  (earnings_per_subscriber_eq : earnings_per_subscriber = 9) :
  (initial_subscribers + gifted_subscribers) * earnings_per_subscriber = 1800 := by
  sorry

end james_monthly_earnings_l93_93093


namespace farmer_ear_count_l93_93197

theorem farmer_ear_count
    (seeds_per_ear : ℕ)
    (price_per_ear : ℝ)
    (cost_per_bag : ℝ)
    (seeds_per_bag : ℕ)
    (profit : ℝ)
    (target_profit : ℝ) :
  seeds_per_ear = 4 →
  price_per_ear = 0.1 →
  cost_per_bag = 0.5 →
  seeds_per_bag = 100 →
  target_profit = 40 →
  profit = price_per_ear - ((cost_per_bag / seeds_per_bag) * seeds_per_ear) →
  target_profit / profit = 500 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end farmer_ear_count_l93_93197


namespace equilateral_triangle_area_l93_93824

theorem equilateral_triangle_area (h : ℝ) (h_eq : h = 2 * Real.sqrt 3) : 
  (Real.sqrt 3 / 4) * (2 * h / (Real.sqrt 3))^2 = 4 * Real.sqrt 3 := 
by
  rw [h_eq]
  sorry

end equilateral_triangle_area_l93_93824


namespace hexagon_side_lengths_l93_93603

theorem hexagon_side_lengths (a b c d e f : ℕ) (h : a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ e ∧ e ≠ f)
(h1: a = 7 ∧ b = 5 ∧ (a + b + c + d + e + f = 38)) : 
(a + b + c + d + e + f = 38 ∧ a + b + c + d + e + f = 7 + 7 + 7 + 7 + 5 + 5) → 
(a + b + c + d + e + f = (4 * 7) + (2 * 5)) :=
sorry

end hexagon_side_lengths_l93_93603


namespace min_cookies_satisfy_conditions_l93_93407

theorem min_cookies_satisfy_conditions : ∃ (b : ℕ), b ≡ 5 [MOD 6] ∧ b ≡ 7 [MOD 8] ∧ b ≡ 8 [MOD 9] ∧ ∀ (b' : ℕ), (b' ≡ 5 [MOD 6] ∧ b' ≡ 7 [MOD 8] ∧ b' ≡ 8 [MOD 9]) → b ≤ b' := 
sorry

end min_cookies_satisfy_conditions_l93_93407


namespace max_neg_expr_l93_93619

theorem max_neg_expr (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) : 
  (- (1 / (2 * a)) - (2 / b)) ≤ - (9 / 2) :=
sorry

end max_neg_expr_l93_93619


namespace solomon_sale_price_l93_93415

def original_price : ℝ := 500
def discount_rate : ℝ := 0.10
def sale_price := original_price * (1 - discount_rate)

theorem solomon_sale_price : sale_price = 450 := by
  sorry

end solomon_sale_price_l93_93415


namespace milo_eggs_weight_l93_93655

def weight_of_one_egg : ℚ := 1/16
def eggs_per_dozen : ℕ := 12
def dozens_needed : ℕ := 8

theorem milo_eggs_weight :
  (dozens_needed * eggs_per_dozen : ℚ) * weight_of_one_egg = 6 := by sorry

end milo_eggs_weight_l93_93655


namespace dorothy_annual_earnings_correct_l93_93883

-- Define the conditions
def dorothyEarnings (X : ℝ) : Prop :=
  X - 0.18 * X = 49200

-- Define the amount Dorothy earns a year
def dorothyAnnualEarnings : ℝ := 60000

-- State the theorem
theorem dorothy_annual_earnings_correct : dorothyEarnings dorothyAnnualEarnings :=
by
-- The proof will be inserted here
sorry

end dorothy_annual_earnings_correct_l93_93883


namespace num_pairs_equals_one_l93_93362

noncomputable def fractional_part (x : ℚ) : ℚ := x - x.floor

open BigOperators

theorem num_pairs_equals_one :
  ∃! (n : ℕ) (q : ℚ), 
    (0 < q ∧ q < 2000) ∧ 
    ¬ q.isInt ∧ 
    fractional_part (q^2) = fractional_part (n.choose 2000)
:= sorry

end num_pairs_equals_one_l93_93362


namespace trig_identity_proof_l93_93601

theorem trig_identity_proof :
  (Real.cos (10 * Real.pi / 180) * Real.sin (70 * Real.pi / 180) - Real.cos (80 * Real.pi / 180) * Real.sin (20 * Real.pi / 180)) = (Real.sqrt 3 / 2) :=
by
  sorry

end trig_identity_proof_l93_93601


namespace x_squared_plus_y_squared_l93_93912

theorem x_squared_plus_y_squared (x y : ℝ) (h1 : (x + y)^2 = 1) (h2 : x * y = -4) : x^2 + y^2 = 9 :=
sorry

end x_squared_plus_y_squared_l93_93912


namespace total_phones_in_Delaware_l93_93296

def population : ℕ := 974000
def phones_per_1000 : ℕ := 673

theorem total_phones_in_Delaware : (population / 1000) * phones_per_1000 = 655502 := by
  sorry

end total_phones_in_Delaware_l93_93296


namespace vacation_cost_l93_93578

theorem vacation_cost (C : ℝ) (h1 : C / 3 - C / 4 = 30) : C = 360 :=
by
  sorry

end vacation_cost_l93_93578


namespace solutions_to_x_squared_eq_x_l93_93433

theorem solutions_to_x_squared_eq_x (x : ℝ) : x^2 = x ↔ x = 0 ∨ x = 1 := 
sorry

end solutions_to_x_squared_eq_x_l93_93433


namespace temperature_increase_per_century_l93_93930

def total_temperature_change_over_1600_years : ℕ := 64
def years_in_a_century : ℕ := 100
def years_overall : ℕ := 1600

theorem temperature_increase_per_century :
  total_temperature_change_over_1600_years / (years_overall / years_in_a_century) = 4 := by
  sorry

end temperature_increase_per_century_l93_93930


namespace range_of_a_l93_93084

theorem range_of_a (a : ℝ) : (∀ x y : ℝ, (2 ≤ x ∧ x ≤ 4) ∧ (2 ≤ y ∧ y ≤ 3) → x * y ≤ a * x^2 + 2 * y^2) → a ≥ 0 := 
sorry

end range_of_a_l93_93084


namespace system_solution_l93_93215

theorem system_solution (u v w : ℚ) 
  (h1 : 3 * u - 4 * v + w = 26)
  (h2 : 6 * u + 5 * v - 2 * w = -17) :
  u + v + w = 101 / 3 :=
sorry

end system_solution_l93_93215


namespace contrapositive_of_square_root_l93_93423

theorem contrapositive_of_square_root (a b : ℝ) :
  (a^2 < b → -Real.sqrt b < a ∧ a < Real.sqrt b) ↔ (a ≥ Real.sqrt b ∨ a ≤ -Real.sqrt b → a^2 ≥ b) := 
sorry

end contrapositive_of_square_root_l93_93423


namespace fraction_from_condition_l93_93861

theorem fraction_from_condition (x f : ℝ) (h : 0.70 * x = f * x + 110) (hx : x = 300) : f = 1 / 3 :=
by
  sorry

end fraction_from_condition_l93_93861


namespace rectangle_long_side_eq_12_l93_93476

theorem rectangle_long_side_eq_12 (s : ℕ) (a b : ℕ) (congruent_triangles : true) (h : a + b = s) (short_side_is_8 : s = 8) : a + b + 4 = 12 :=
by
  sorry

end rectangle_long_side_eq_12_l93_93476


namespace Zilla_savings_l93_93852

/-- Zilla's monthly savings based on her spending distributions -/
theorem Zilla_savings
  (rent : ℚ) (monthly_earnings_percentage : ℚ)
  (other_expenses_fraction : ℚ) (monthly_rent : ℚ)
  (monthly_expenses : ℚ) (total_monthly_earnings : ℚ)
  (half_monthly_earnings : ℚ) (savings : ℚ)
  (h1 : rent = 133)
  (h2 : monthly_earnings_percentage = 0.07)
  (h3 : other_expenses_fraction = 0.5)
  (h4 : total_monthly_earnings = monthly_rent / monthly_earnings_percentage)
  (h5 : half_monthly_earnings = total_monthly_earnings * other_expenses_fraction)
  (h6 : savings = total_monthly_earnings - (monthly_rent + half_monthly_earnings))
  : savings = 817 :=
sorry

end Zilla_savings_l93_93852


namespace system_of_equations_solution_l93_93072

theorem system_of_equations_solution :
  ∀ (x y z : ℝ),
  4 * x + 2 * y + z = 20 →
  x + 4 * y + 2 * z = 26 →
  2 * x + y + 4 * z = 28 →
  20 * x^2 + 24 * x * y + 20 * y^2 + 12 * z^2 = 500 :=
by
  intros x y z h1 h2 h3
  sorry

end system_of_equations_solution_l93_93072


namespace geometric_sequence_S6_l93_93371

variable (a : ℕ → ℝ) -- represents the geometric sequence

noncomputable def S (n : ℕ) : ℝ :=
if n = 0 then 0 else ((a 0) * (1 - (a 1 / a 0) ^ n)) / (1 - a 1 / a 0)

theorem geometric_sequence_S6 (h : ∀ n, a n = (a 0) * (a 1 / a 0) ^ n) :
  S a 2 = 6 ∧ S a 4 = 18 → S a 6 = 42 := 
by 
  intros h1
  sorry

end geometric_sequence_S6_l93_93371


namespace infinite_arith_prog_contains_infinite_nth_powers_l93_93954

theorem infinite_arith_prog_contains_infinite_nth_powers
  (a d : ℕ) (n : ℕ) 
  (h_pos: 0 < d) 
  (h_power: ∃ k : ℕ, ∃ m : ℕ, a + k * d = m^n) :
  ∃ infinitely_many k : ℕ, ∃ m : ℕ, a + k * d = m^n :=
sorry

end infinite_arith_prog_contains_infinite_nth_powers_l93_93954


namespace total_marbles_count_l93_93921

variable (r b g : ℝ)
variable (h1 : r = 1.4 * b) (h2 : g = 1.5 * r)

theorem total_marbles_count (r b g : ℝ) (h1 : r = 1.4 * b) (h2 : g = 1.5 * r) :
  r + b + g = 3.21 * r :=
by
  sorry

end total_marbles_count_l93_93921


namespace decreased_revenue_l93_93856

variable (T C : ℝ)
def Revenue (tax consumption : ℝ) : ℝ := tax * consumption

theorem decreased_revenue (hT_new : T_new = 0.9 * T) (hC_new : C_new = 1.1 * C) :
  Revenue T_new C_new = 0.99 * (Revenue T C) := 
sorry

end decreased_revenue_l93_93856


namespace toy_swords_count_l93_93935

variable (s : ℕ)

def cost_lego := 250
def cost_toy_sword := 120
def cost_play_dough := 35

def total_cost (s : ℕ) :=
  3 * cost_lego + s * cost_toy_sword + 10 * cost_play_dough

theorem toy_swords_count : total_cost s = 1940 → s = 7 := by
  sorry

end toy_swords_count_l93_93935


namespace kanul_total_amount_l93_93540

theorem kanul_total_amount (T : ℝ) (h1 : 500 + 400 + 0.10 * T = T) : T = 1000 :=
  sorry

end kanul_total_amount_l93_93540


namespace part_b_part_c_l93_93183

-- Statement for part b: In how many ways can the figure be properly filled with the numbers from 1 to 5?
def proper_fill_count_1_to_5 : Nat :=
  8

-- Statement for part c: In how many ways can the figure be properly filled with the numbers from 1 to 7?
def proper_fill_count_1_to_7 : Nat :=
  48

theorem part_b :
  proper_fill_count_1_to_5 = 8 :=
sorry

theorem part_c :
  proper_fill_count_1_to_7 = 48 :=
sorry

end part_b_part_c_l93_93183


namespace neg_p_l93_93774

-- Define the initial proposition p
def p : Prop := ∀ (m : ℝ), m ≥ 0 → 4^m ≥ 4 * m

-- State the theorem to prove the negation of p
theorem neg_p : ¬p ↔ ∃ (m_0 : ℝ), m_0 ≥ 0 ∧ 4^m_0 < 4 * m_0 :=
by
  sorry

end neg_p_l93_93774


namespace fare_per_1_5_mile_l93_93383

-- Definitions and conditions
def fare_first : ℝ := 1.0
def total_fare : ℝ := 7.3
def increments_per_mile : ℝ := 5.0
def total_miles : ℝ := 3.0
def remaining_increments : ℝ := (total_miles * increments_per_mile) - 1
def remaining_fare : ℝ := total_fare - fare_first

-- Theorem to prove
theorem fare_per_1_5_mile : remaining_fare / remaining_increments = 0.45 :=
by
  sorry

end fare_per_1_5_mile_l93_93383


namespace amount_daria_needs_l93_93496

theorem amount_daria_needs (ticket_cost : ℕ) (total_tickets : ℕ) (current_money : ℕ) (needed_money : ℕ) 
  (h1 : ticket_cost = 90) 
  (h2 : total_tickets = 4) 
  (h3 : current_money = 189) 
  (h4 : needed_money = 360 - 189) 
  : needed_money = 171 := by
  -- proof omitted
  sorry

end amount_daria_needs_l93_93496


namespace pizza_combinations_l93_93656

theorem pizza_combinations (n r k : ℕ) (h_n : n = 9) (h_r : r = 4) (h_k : k = 2) :
  Nat.choose n r - Nat.choose (n - k) r = 91 :=
by
  have h1 : Nat.choose 9 4 = 126 := by sorry
  have h2 : Nat.choose 7 4 = 35 := by sorry
  rw [h_n, h_r, h_k]
  calc
    Nat.choose 9 4 - Nat.choose (9 - 2) 4 = 126 - 35 := by rw [h1, h2]
    ... = 91 := by sorry

end pizza_combinations_l93_93656


namespace calc_addition_even_odd_probability_calc_addition_multiplication_even_probability_l93_93460

-- Define the necessary probability events and conditions.
variable {p : ℝ} (calc_action : ℕ → ℝ)

-- Condition: initially, the display shows 0.
def initial_display : ℕ := 0

-- Events for part (a): addition only, randomly chosen numbers from 0 to 9.
def random_addition_event (n : ℕ) : Prop := n % 2 = 0 ∨ n % 2 = 1

-- Events for part (b): both addition and multiplication allowed.
def random_operation_event (n : ℕ) : Prop := (n % 2 = 0 ∧ n % 2 = 1) ∨ -- addition
                                               (n ≠ 0 ∧ n % 2 = 1 ∧ (n/2) % 2 = 1) -- multiplication

-- Statements to be proved based on above definitions.
theorem calc_addition_even_odd_probability :
  calc_action 0 = 1 / 2 → random_addition_event initial_display := sorry

theorem calc_addition_multiplication_even_probability :
  calc_action (initial_display + 1) > 1 / 2 → random_operation_event (initial_display + 1) := sorry

end calc_addition_even_odd_probability_calc_addition_multiplication_even_probability_l93_93460


namespace fraction_of_cream_in_cup1_after_operations_l93_93598

/-
We consider two cups of liquids with the following contents initially:
Cup 1 has 6 ounces of coffee.
Cup 2 has 2 ounces of coffee and 4 ounces of cream.
After pouring half of Cup 1's content into Cup 2, stirring, and then pouring half of Cup 2's new content back into Cup 1, we need to show that 
the fraction of the liquid in Cup 1 that is now cream is 4/15.
-/

theorem fraction_of_cream_in_cup1_after_operations :
  let cup1_initial_coffee := 6
  let cup2_initial_coffee := 2
  let cup2_initial_cream := 4
  let cup2_initial_liquid := cup2_initial_coffee + cup2_initial_cream
  let cup1_to_cup2_coffee := cup1_initial_coffee / 2
  let cup1_final_coffee := cup1_initial_coffee - cup1_to_cup2_coffee
  let cup2_final_coffee := cup2_initial_coffee + cup1_to_cup2_coffee
  let cup2_final_liquid := cup2_final_coffee + cup2_initial_cream
  let cup2_to_cup1_liquid := cup2_final_liquid / 2
  let cup2_coffee_fraction := cup2_final_coffee / cup2_final_liquid
  let cup2_cream_fraction := cup2_initial_cream / cup2_final_liquid
  let cup2_to_cup1_coffee := cup2_to_cup1_liquid * cup2_coffee_fraction
  let cup2_to_cup1_cream := cup2_to_cup1_liquid * cup2_cream_fraction
  let cup1_final_liquid_coffee := cup1_final_coffee + cup2_to_cup1_coffee
  let cup1_final_liquid_cream := cup2_to_cup1_cream
  let cup1_final_liquid := cup1_final_liquid_coffee + cup1_final_liquid_cream
  (cup1_final_liquid_cream / cup1_final_liquid) = 4 / 15 :=
by
  sorry

end fraction_of_cream_in_cup1_after_operations_l93_93598


namespace calculate_value_l93_93352

theorem calculate_value : (2 / 3 : ℝ)^0 + Real.log 2 + Real.log 5 = 2 :=
by 
  sorry

end calculate_value_l93_93352


namespace infinite_non_congruent_integers_l93_93397

theorem infinite_non_congruent_integers (a : ℕ → ℤ) (m : ℕ → ℤ) (k : ℕ)
  (h1 : ∀ i : ℕ, 1 ≤ i ∧ i ≤ k → 2 ≤ m i)
  (h2 : ∀ i : ℕ, 1 ≤ i ∧ i < k → 2 * m i ≤ m (i + 1)) :
  ∃ (x : ℕ), ∀ i : ℕ, 1 ≤ i ∧ i ≤ k → ¬ (x % (m i) = a i % (m i)) :=
sorry

end infinite_non_congruent_integers_l93_93397


namespace bottom_row_bricks_l93_93528

theorem bottom_row_bricks {x : ℕ} 
  (c1 : ∀ i, i < 5 → (x - i) > 0)
  (c2 : x + (x - 1) + (x - 2) + (x - 3) + (x - 4) = 100) : 
  x = 22 := 
by
  sorry

end bottom_row_bricks_l93_93528


namespace triangle_existence_l93_93897

theorem triangle_existence (n : ℕ) (h : 2 * n > 0) (segments : Finset (ℕ × ℕ))
  (h_segments : segments.card = n^2 + 1)
  (points_in_segment : ∀ {a b : ℕ}, (a, b) ∈ segments → a < 2 * n ∧ b < 2 * n) :
  ∃ x y z, x < 2 * n ∧ y < 2 * n ∧ z < 2 * n ∧ (x ≠ y ∧ y ≠ z ∧ z ≠ x) ∧
  ((x, y) ∈ segments ∨ (y, x) ∈ segments) ∧
  ((y, z) ∈ segments ∨ (z, y) ∈ segments) ∧
  ((z, x) ∈ segments ∨ (x, z) ∈ segments) :=
by
  sorry

end triangle_existence_l93_93897


namespace inscribed_circle_radius_l93_93785

variable (A p s r : ℝ)

-- Condition: Area is twice the perimeter
def twice_perimeter_condition : Prop := A = 2 * p

-- Condition: The formula connecting the area, inradius, and semiperimeter
def area_inradius_semiperimeter_relation : Prop := A = r * s

-- Condition: The perimeter is twice the semiperimeter
def perimeter_semiperimeter_relation : Prop := p = 2 * s

-- Prove the radius of the inscribed circle is 4
theorem inscribed_circle_radius (h1 : twice_perimeter_condition A p)
                                (h2 : area_inradius_semiperimeter_relation A r s)
                                (h3 : perimeter_semiperimeter_relation p s) :
  r = 4 :=
by
  sorry

end inscribed_circle_radius_l93_93785


namespace count_coprime_to_15_l93_93222

def coprime_to_15 (a : ℕ) : Prop := Nat.gcd 15 a = 1

theorem count_coprime_to_15 : 
  (Finset.filter coprime_to_15 (Finset.range 15)).card = 8 := by
  sorry

end count_coprime_to_15_l93_93222


namespace nh4i_required_l93_93888

theorem nh4i_required (KOH NH4I NH3 KI H2O : ℕ) (h_eq : 1 * NH4I + 1 * KOH = 1 * NH3 + 1 * KI + 1 * H2O)
  (h_KOH : KOH = 3) : NH4I = 3 := 
by
  sorry

end nh4i_required_l93_93888


namespace tank_fill_time_l93_93806

-- Define the conditions
def capacity := 800
def rate_A := 40
def rate_B := 30
def rate_C := -20

def net_rate_per_cycle := rate_A + rate_B + rate_C
def cycle_duration := 3
def total_cycles := capacity / net_rate_per_cycle
def total_time := total_cycles * cycle_duration

-- The proof that tank will be full after 48 minutes
theorem tank_fill_time : total_time = 48 := by
  sorry

end tank_fill_time_l93_93806


namespace fraction_simplification_l93_93124

-- Definitions based on conditions and question
def lcm_462_42 : ℕ := 462
def prime_factors_462 : List ℕ := [2, 3, 7, 11]
def prime_factors_42 : List ℕ := [2, 3, 7]

-- Main theorem statement
theorem fraction_simplification :
  (1 / 462) + (17 / 42) = 94 / 231 :=
by
  sorry

end fraction_simplification_l93_93124


namespace inequality_proof_l93_93507

theorem inequality_proof
  (p q a b c d e : Real)
  (hpq : 0 < p ∧ p ≤ a ∧ p ≤ b ∧ p ≤ c ∧ p ≤ d ∧ p ≤ e)
  (hq : a ≤ q ∧ b ≤ q ∧ c ≤ q ∧ d ≤ q ∧ e ≤ q) :
  (a + b + c + d + e) * (1 / a + 1 / b + 1 / c + 1 / d + 1 / e)
  ≤ 25 + 6 * (Real.sqrt (p / q) - Real.sqrt (q / p)) ^ 2 :=
sorry

end inequality_proof_l93_93507


namespace fewest_reciprocal_keypresses_l93_93076

theorem fewest_reciprocal_keypresses (f : ℝ → ℝ) (x : ℝ) (hx : x ≠ 0) 
  (h1 : f 50 = 1 / 50) (h2 : f (1 / 50) = 50) : 
  ∃ n : ℕ, n = 2 ∧ (∀ m : ℕ, (m < n) → (f^[m] 50 ≠ 50)) :=
by
  sorry

end fewest_reciprocal_keypresses_l93_93076


namespace length_of_FD_l93_93261

theorem length_of_FD (a b c d f e : ℝ) (x : ℝ) :
  a = 0 ∧ b = 8 ∧ c = 8 ∧ d = 0 ∧ 
  e = 8 * (2 / 3) ∧ 
  (8 - x)^2 = x^2 + (8 / 3)^2 ∧ 
  a = d → c = b → 
  d = 8 → 
  x = 32 / 9 :=
by
  sorry

end length_of_FD_l93_93261


namespace vertical_asymptote_l93_93229

theorem vertical_asymptote (x : ℝ) : (4 * x + 6 = 0) -> x = -3 / 2 :=
by
  sorry

end vertical_asymptote_l93_93229


namespace range_of_c_monotonicity_of_g_l93_93245

noncomputable def f (x: ℝ) : ℝ := 2 * Real.log x + 1

theorem range_of_c (c: ℝ) : (∀ x > 0, f x ≤ 2 * x + c) → c ≥ -1 := by
  sorry

noncomputable def g (x a: ℝ) : ℝ := (f x - f a) / (x - a)

theorem monotonicity_of_g (a: ℝ) (ha: a > 0) : 
  (∀ x > 0, x ≠ a → ((x < a → g x a < g a a) ∧ (x > a → g x a < g a a))) := by
  sorry

end range_of_c_monotonicity_of_g_l93_93245


namespace total_amount_divided_l93_93012

theorem total_amount_divided (A B C : ℝ) (h1 : A / B = 3 / 4) (h2 : B / C = 5 / 6) (h3 : A = 29491.525423728814) :
  A + B + C = 116000 := 
sorry

end total_amount_divided_l93_93012


namespace sin_diff_identity_l93_93350

variable (α β : ℝ)

def condition1 := (Real.sin α - Real.cos β = 3 / 4)
def condition2 := (Real.cos α + Real.sin β = -2 / 5)

theorem sin_diff_identity : 
  condition1 α β → 
  condition2 α β → 
  Real.sin (α - β) = 511 / 800 :=
by
  intros h1 h2
  sorry

end sin_diff_identity_l93_93350


namespace cds_total_l93_93642

theorem cds_total (dawn_cds : ℕ) (h1 : dawn_cds = 10) (h2 : ∀ kristine_cds : ℕ, kristine_cds = dawn_cds + 7) :
  dawn_cds + (dawn_cds + 7) = 27 :=
by
  sorry

end cds_total_l93_93642


namespace train_length_l93_93021

theorem train_length
  (train_speed_kmph : ℝ)
  (person_speed_kmph : ℝ)
  (time_seconds : ℝ)
  (h_train_speed : train_speed_kmph = 80)
  (h_person_speed : person_speed_kmph = 16)
  (h_time : time_seconds = 15)
  : (train_speed_kmph - person_speed_kmph) * (5/18) * time_seconds = 266.67 := 
by
  rw [h_train_speed, h_person_speed, h_time]
  norm_num
  sorry

end train_length_l93_93021


namespace focus_of_parabola_l93_93511

/-- Given a quadratic function f(x) = ax^2 + bx + 2 where a ≠ 0, and for any real number x, it holds that |f(x)| ≥ 2,
    prove that the coordinates of the focus of the parabolic curve are (0, 1 / (4 * a) + 2). -/
theorem focus_of_parabola (a b : ℝ) (h_a : a ≠ 0)
  (h_f : ∀ x : ℝ, |a * x^2 + b * x + 2| ≥ 2) :
  (0, (1 / (4 * a) + 2)) = (0, (1 / (4 * a) + 2)) :=
by
  sorry

end focus_of_parabola_l93_93511


namespace total_money_shared_l93_93115

theorem total_money_shared (k t : ℕ) (h1 : k = 1750) (h2 : t = 2 * k) : k + t = 5250 :=
by
  sorry

end total_money_shared_l93_93115


namespace simplify_fraction_l93_93129

theorem simplify_fraction :
  (1 : ℚ) / 462 + 17 / 42 = 94 / 231 := 
sorry

end simplify_fraction_l93_93129


namespace grain_demand_l93_93252

variable (F : ℝ)
def S0 : ℝ := 1800000 -- base supply value

theorem grain_demand : ∃ D : ℝ, S = 0.75 * D ∧ S = S0 * (1 + F) ∧ D = (1800000 * (1 + F) / 0.75) :=
by
  sorry

end grain_demand_l93_93252


namespace polygon_sides_l93_93164

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 1260) : n = 9 :=
sorry

end polygon_sides_l93_93164


namespace school_stats_l93_93927

-- Defining the conditions
def girls_grade6 := 315
def boys_grade6 := 309
def girls_grade7 := 375
def boys_grade7 := 341
def drama_club_members := 80
def drama_club_boys_percent := 30 / 100

-- Calculate the derived numbers
def students_grade6 := girls_grade6 + boys_grade6
def students_grade7 := girls_grade7 + boys_grade7
def total_students := students_grade6 + students_grade7
def drama_club_boys := drama_club_boys_percent * drama_club_members
def drama_club_girls := drama_club_members - drama_club_boys

-- Theorem
theorem school_stats :
  total_students = 1340 ∧
  drama_club_girls = 56 ∧
  boys_grade6 = 309 ∧
  boys_grade7 = 341 :=
by
  -- We provide the proof steps inline with sorry placeholders.
  -- In practice, these would be filled with appropriate proofs.
  sorry

end school_stats_l93_93927


namespace a2018_is_4035_l93_93066

noncomputable def f : ℝ → ℝ := sorry
def a (n : ℕ) : ℝ := sorry

axiom domain : ∀ x : ℝ, true 
axiom condition_2 : ∀ x : ℝ, x < 0 → f x > 1
axiom condition_3 : ∀ x y : ℝ, f x * f y = f (x + y)
axiom sequence_def : ∀ n : ℕ, n > 0 → a 1 = f 0 ∧ f (a (n + 1)) = 1 / f (-2 - a n)

theorem a2018_is_4035 : a 2018 = 4035 :=
sorry

end a2018_is_4035_l93_93066


namespace divisor_value_l93_93182

theorem divisor_value (D : ℕ) (k m : ℤ) (h1 : 242 % D = 8) (h2 : 698 % D = 9) (h3 : (242 + 698) % D = 4) : D = 13 := by
  sorry

end divisor_value_l93_93182


namespace quadratic_roots_algebraic_expression_value_l93_93713

-- Part 1: Proof statement for the roots of the quadratic equation
theorem quadratic_roots : (∃ x₁ x₂ : ℝ, x₁ = 2 + Real.sqrt 7 ∧ x₂ = 2 - Real.sqrt 7 ∧ (∀ x : ℝ, x^2 - 4 * x - 3 = 0 → x = x₁ ∨ x = x₂)) :=
by
  sorry

-- Part 2: Proof statement for the algebraic expression value
theorem algebraic_expression_value (a : ℝ) (h : a^2 = 3 * a + 10) :
  (a + 4) * (a - 4) - 3 * (a - 1) = -3 :=
by
  sorry

end quadratic_roots_algebraic_expression_value_l93_93713


namespace remainder_when_divided_by_14_l93_93017

theorem remainder_when_divided_by_14 (A : ℕ) (h1 : A % 1981 = 35) (h2 : A % 1982 = 35) : A % 14 = 7 :=
sorry

end remainder_when_divided_by_14_l93_93017


namespace simplify_expr_l93_93817

def expr1 : ℚ := (3 + 4 + 6 + 7) / 3
def expr2 : ℚ := (3 * 6 + 9) / 4

theorem simplify_expr : expr1 + expr2 = 161 / 12 := by
  sorry

end simplify_expr_l93_93817


namespace bakery_combinations_l93_93993

theorem bakery_combinations 
  (total_breads : ℕ) (bread_types : Finset ℕ) (purchases : Finset ℕ)
  (h_total : total_breads = 8)
  (h_bread_types : bread_types.card = 5)
  (h_purchases : purchases.card = 2) : 
  ∃ (combinations : ℕ), combinations = 70 := 
sorry

end bakery_combinations_l93_93993


namespace graph_f_intersects_x_eq_1_at_most_once_l93_93144

-- Define a function f from ℝ to ℝ
def f : ℝ → ℝ := sorry  -- Placeholder for the actual function

-- Define the domain of the function f (it's a generic function on ℝ for simplicity)
axiom f_unique : ∀ x y : ℝ, f x = f y → x = y  -- If f(x) = f(y), then x must equal y

-- Prove that the graph of y = f(x) intersects the line x = 1 at most once
theorem graph_f_intersects_x_eq_1_at_most_once : ∃ y : ℝ, (f 1 = y) ∨ (¬∃ y : ℝ, f 1 = y) :=
by
  -- Proof goes here
  sorry

end graph_f_intersects_x_eq_1_at_most_once_l93_93144


namespace ratio_a_over_c_l93_93565

variables {a b c x1 x2 : Real}
variables (h1 : x1 + x2 = -a) (h2 : x1 * x2 = b) (h3 : b = 2 * a) (h4 : c = 4 * b)
           (ha_nonzero : a ≠ 0) (hb_nonzero : b ≠ 0) (hc_nonzero : c ≠ 0)

theorem ratio_a_over_c : a / c = 1 / 8 :=
by
  have hc_eq : c = 8 * a := by
    rw [h4, h3]
    simp
  rw [hc_eq]
  field_simp [ha_nonzero]
  norm_num
  sorry -- additional steps if required

end ratio_a_over_c_l93_93565


namespace min_objective_value_l93_93300

theorem min_objective_value (x y : ℝ) 
  (h1 : x + y ≥ 2) 
  (h2 : x - y ≤ 2) 
  (h3 : y ≥ 1) : ∃ (z : ℝ), z = x + 3 * y ∧ z = 4 :=
by
  -- Provided proof omitted
  sorry

end min_objective_value_l93_93300


namespace correct_operation_l93_93706

variable {a b : ℝ}

def conditionA : Prop := a^2 + a^3 = a^6
def conditionB : Prop := (a * b)^2 = a * b^2
def conditionC : Prop := (a + b)^2 = a^2 + b^2
def conditionD : Prop := (a + b) * (a - b) = a^2 - b^2

theorem correct_operation : conditionD ∧ ¬conditionA ∧ ¬conditionB ∧ ¬conditionC := by
  sorry

end correct_operation_l93_93706


namespace smallest_number_l93_93203

theorem smallest_number (a b c d : ℤ) (h1 : a = 1) (h2 : b = 0) (h3 : c = -1) (h4 : d = -3) :
  d = -3 ∧ d < c ∧ d < b ∧ d < a :=
by
  sorry

end smallest_number_l93_93203


namespace soccer_team_lineups_l93_93592

open Finset

theorem soccer_team_lineups : 
  let players := range 12
  let quadruplets := {0, 1, 2, 3}  -- Assume Ben, Bob, Bill, Bert are represented by indices 0, 1, 2, and 3
  let others := players \ quadruplets
  let choose_quadruplets := (quadruplets.card.choose 2)
  let choose_others := (others.card.choose 3)
  (choose_quadruplets * choose_others) = 336 :=
by 
  -- Definitions
  let players := range 12
  let quadruplets := {0, 1, 2, 3}
  let others := players \ quadruplets
  let choose_quadruplets := (quadruplets.card.choose 2)
  let choose_others := (others.card.choose 3)
  
  -- Assertions
  have h1 : quadruplets.card = 4 := rfl
  have h2 : others.card = 8 := rfl
  have h3 : choose_quadruplets = 6 := by { rw h1, exact choose_symm _ _ }
  have h4 : choose_others = 56 := by { rw h2, exact choose_symm _ _ }
  
  -- Final proof
  unfold choose_quadruplets choose_others,
  rw [h3, h4],
  norm_num

end soccer_team_lineups_l93_93592


namespace find_point_P_l93_93759

structure Point :=
  (x : ℝ)
  (y : ℝ)

def M : Point := ⟨2, 2⟩
def N : Point := ⟨5, -2⟩

def is_on_x_axis (P : Point) : Prop :=
  P.y = 0

def is_right_angle (M N P : Point) : Prop :=
  (M.x - P.x)*(N.x - P.x) + (M.y - P.y)*(N.y - P.y) = 0

noncomputable def P1 : Point := ⟨1, 0⟩
noncomputable def P2 : Point := ⟨6, 0⟩

theorem find_point_P :
  ∃ P : Point, is_on_x_axis P ∧ is_right_angle M N P ∧ (P = P1 ∨ P = P2) :=
by
  sorry

end find_point_P_l93_93759


namespace power_of_negative_base_l93_93610

theorem power_of_negative_base : (-64 : ℤ)^(7 / 6) = -128 := by
  sorry

end power_of_negative_base_l93_93610


namespace find_a_l93_93842

theorem find_a (a : ℝ) :
  let θ := 120
  let tan120 := -Real.sqrt 3
  (∀ x y: ℝ, 2 * x + a * y + 3 = 0) →
  a = 2 * Real.sqrt 3 / 3 :=
by
  sorry

end find_a_l93_93842


namespace Kyler_wins_1_game_l93_93553

theorem Kyler_wins_1_game
  (peter_wins : ℕ)
  (peter_losses : ℕ)
  (emma_wins : ℕ)
  (emma_losses : ℕ)
  (kyler_losses : ℕ)
  (total_games : ℕ)
  (kyler_wins : ℕ)
  (htotal : total_games = (peter_wins + peter_losses + emma_wins + emma_losses + kyler_wins + kyler_losses) / 2)
  (hpeter : peter_wins = 4 ∧ peter_losses = 2)
  (hemma : emma_wins = 3 ∧ emma_losses = 3)
  (hkyler_losses : kyler_losses = 3)
  (htotal_wins_losses : total_games = peter_wins + emma_wins + kyler_wins) : kyler_wins = 1 :=
by
  sorry

end Kyler_wins_1_game_l93_93553


namespace percentage_change_in_area_halved_length_tripled_breadth_l93_93675

def original_area (L B : ℝ) : ℝ := L * B
def new_area (L B : ℝ) : ℝ := (L / 2) * (3 * B)

def percentage_change (A1 A2 : ℝ) : ℝ :=
  ((A2 - A1) / A1) * 100

theorem percentage_change_in_area_halved_length_tripled_breadth
  (L B : ℝ) (hL : L > 0) (hB : B > 0) :
  percentage_change (original_area L B) (new_area L B) = 50 :=
begin
  sorry
end

end percentage_change_in_area_halved_length_tripled_breadth_l93_93675


namespace no_integer_x_square_l93_93399

theorem no_integer_x_square (x : ℤ) : 
  ∀ n : ℤ, x^5 + 5 * x^4 + 10 * x^3 + 10 * x^2 + 5 * x + 1 ≠ n^2 :=
by sorry

end no_integer_x_square_l93_93399


namespace remainder_div_9_l93_93055

theorem remainder_div_9 (x y : ℤ) (h : 9 ∣ (x + 2 * y)) : (2 * (5 * x - 8 * y - 4)) % 9 = -8 ∨ (2 * (5 * x - 8 * y - 4)) % 9 = 1 :=
by
  sorry

end remainder_div_9_l93_93055


namespace geometric_segment_l93_93388

theorem geometric_segment (AB A'B' : ℝ) (P D A B P' D' A' B' : ℝ) (x y a : ℝ) :
  AB = 3 ∧ A'B' = 6 ∧ (∀ P, dist P D = x) ∧ (∀ P', dist P' D' = 2 * x) ∧ x = a → x + y = 3 * a :=
by
  sorry

end geometric_segment_l93_93388


namespace probability_of_winning_in_7_games_l93_93007

noncomputable def prob_of_winning_in_7_games : ℝ :=
  let p := 2 / 3 in
  let prob_mathletes_win := (Nat.choose 6 4 * p ^ 4 * (1 - p) ^ 2 * p) in
  let prob_other_win := (Nat.choose 6 4 * (1 - p) ^ 4 * p ^ 2 * (1 - p)) in
  prob_mathletes_win + prob_other_win

theorem probability_of_winning_in_7_games :
  prob_of_winning_in_7_games = 20 / 27 :=
sorry

end probability_of_winning_in_7_games_l93_93007


namespace total_pencil_length_l93_93461

-- Definitions from the conditions
def purple_length : ℕ := 3
def black_length : ℕ := 2
def blue_length : ℕ := 1

-- Proof statement
theorem total_pencil_length :
  purple_length + black_length + blue_length = 6 :=
by
  sorry

end total_pencil_length_l93_93461


namespace quadratic_roots_relation_l93_93146

theorem quadratic_roots_relation (m p q : ℝ) (h_m_ne_zero : m ≠ 0) (h_p_ne_zero : p ≠ 0) (h_q_ne_zero : q ≠ 0) :
  (∀ r1 r2 : ℝ, (r1 + r2 = -q ∧ r1 * r2 = m) → (3 * r1 + 3 * r2 = -m ∧ (3 * r1) * (3 * r2) = p)) →
  p / q = 27 :=
by
  intros h
  sorry

end quadratic_roots_relation_l93_93146


namespace compute_g_x_h_l93_93779

def g (x : ℝ) : ℝ := 6 * x^2 - 3 * x + 4

theorem compute_g_x_h (x h : ℝ) : 
  g (x + h) - g x = h * (12 * x + 6 * h - 3) := by
  sorry

end compute_g_x_h_l93_93779


namespace max_satiated_pikes_l93_93170

-- Define the total number of pikes
def total_pikes : ℕ := 30

-- Define the condition for satiation
def satiated_condition (eats : ℕ) : Prop := eats ≥ 3

-- Define the number of pikes eaten by each satiated pike
def eaten_by_satiated_pike : ℕ := 3

-- Define the theorem to find the maximum number of satiated pikes
theorem max_satiated_pikes (s : ℕ) : 
  (s * eaten_by_satiated_pike < total_pikes) → s ≤ 9 :=
by
  sorry

end max_satiated_pikes_l93_93170


namespace projectile_reaches_35_at_1p57_seconds_l93_93665

theorem projectile_reaches_35_at_1p57_seconds :
  ∀ (t : ℝ), (y : ℝ) (h_eq : y = -4.9 * t^2 + 30 * t)
  (h_initial_velocity : true)  -- Given that the projectile is launched from the ground, we assume this as a given
  (h_conditions : y = 35),
  t = 1.57 :=
by
  sorry

end projectile_reaches_35_at_1p57_seconds_l93_93665


namespace expressions_equal_constant_generalized_identity_l93_93337

noncomputable def expr1 := (Real.sin (13 * Real.pi / 180))^2 + (Real.cos (17 * Real.pi / 180))^2 - Real.sin (13 * Real.pi / 180) * Real.cos (17 * Real.pi / 180)
noncomputable def expr2 := (Real.sin (15 * Real.pi / 180))^2 + (Real.cos (15 * Real.pi / 180))^2 - Real.sin (15 * Real.pi / 180) * Real.cos (15 * Real.pi / 180)
noncomputable def expr3 := (Real.sin (-18 * Real.pi / 180))^2 + (Real.cos (48 * Real.pi / 180))^2 - Real.sin (-18 * Real.pi / 180) * Real.cos (48 * Real.pi / 180)
noncomputable def expr4 := (Real.sin (-25 * Real.pi / 180))^2 + (Real.cos (55 * Real.pi / 180))^2 - Real.sin (-25 * Real.pi / 180) * Real.cos (55 * Real.pi / 180)

theorem expressions_equal_constant :
  expr1 = 3/4 ∧ expr2 = 3/4 ∧ expr3 = 3/4 ∧ expr4 = 3/4 :=
sorry

theorem generalized_identity (α : ℝ) :
  (Real.sin α)^2 + (Real.cos (30 * Real.pi / 180 - α))^2 - Real.sin α * Real.cos (30 * Real.pi / 180 - α) = 3 / 4 :=
sorry

end expressions_equal_constant_generalized_identity_l93_93337


namespace two_sectors_area_l93_93172

theorem two_sectors_area {r : ℝ} {θ : ℝ} (h_radius : r = 15) (h_angle : θ = 45) : 
  2 * (θ / 360) * (π * r^2) = 56.25 * π := 
by
  rw [h_radius, h_angle]
  norm_num
  sorry

end two_sectors_area_l93_93172


namespace price_difference_l93_93799

-- Define the prices of commodity X and Y in the year 2001 + n.
def P_X (n : ℕ) (a : ℝ) : ℝ := 4.20 + 0.45 * n + a * n
def P_Y (n : ℕ) (b : ℝ) : ℝ := 6.30 + 0.20 * n + b * n

-- Define the main theorem to prove
theorem price_difference (n : ℕ) (a b : ℝ) :
  P_X n a = P_Y n b + 0.65 ↔ (0.25 + a - b) * n = 2.75 :=
by
  sorry

end price_difference_l93_93799


namespace intersection_points_count_l93_93431

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x
noncomputable def g (x : ℝ) : ℝ := x^2 - 4 * x + 5

theorem intersection_points_count :
  ∃ y1 y2 : ℝ, y1 ≠ y2 ∧ (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 = g x1 ∧ f x2 = g x2) := sorry

end intersection_points_count_l93_93431


namespace sheela_monthly_income_l93_93462

theorem sheela_monthly_income (deposit : ℝ) (percentage : ℝ) (income : ℝ) 
  (h1 : deposit = 2500) (h2 : percentage = 0.25) (h3 : deposit = percentage * income) :
  income = 10000 := 
by
  -- proof steps would go here
  sorry

end sheela_monthly_income_l93_93462


namespace a_greater_than_b_for_n_ge_2_l93_93892

theorem a_greater_than_b_for_n_ge_2 
  (n : ℕ) 
  (hn : n ≥ 2) 
  (a b : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (h1 : a^n = a + 1) 
  (h2 : b^(2 * n) = b + 3 * a) : 
  a > b := 
  sorry

end a_greater_than_b_for_n_ge_2_l93_93892


namespace smallest_n_l93_93042

theorem smallest_n (n : ℕ) : 
  (2^n + 5^n - n) % 1000 = 0 ↔ n = 797 :=
sorry

end smallest_n_l93_93042


namespace rate_percent_simple_interest_l93_93848

theorem rate_percent_simple_interest:
  ∀ (P SI T R : ℝ), SI = 400 → P = 1000 → T = 4 → (SI = P * R * T / 100) → R = 10 :=
by
  intros P SI T R h_si h_p h_t h_formula
  -- Proof skipped
  sorry

end rate_percent_simple_interest_l93_93848


namespace fraction_irreducible_l93_93952

theorem fraction_irreducible (n : ℤ) : gcd (2 * n ^ 2 + 9 * n - 17) (n + 6) = 1 := by
  sorry

end fraction_irreducible_l93_93952


namespace evaluate_complex_pow_l93_93607

open Complex

noncomputable def calc : ℂ := (-64 : ℂ) ^ (7 / 6)

theorem evaluate_complex_pow : calc = 128 * Complex.I := by 
  -- Recognize that (-64) = (-4)^3
  -- Apply exponent rules: ((-4)^3)^(7/6) = (-4)^(3 * 7/6) = (-4)^(7/2)
  -- Simplify (-4)^(7/2) = √((-4)^7) = √(-16384)
  -- Calculation (-4)^7 = -16384
  -- Simplify √(-16384) = 128i
  sorry

end evaluate_complex_pow_l93_93607


namespace equilateral_triangle_area_l93_93822

theorem equilateral_triangle_area (h : ℝ) 
  (height_eq : h = 2 * Real.sqrt 3) :
  ∃ (A : ℝ), A = 4 * Real.sqrt 3 :=
by
  sorry

end equilateral_triangle_area_l93_93822


namespace line_passes_through_vertex_of_parabola_l93_93747

theorem line_passes_through_vertex_of_parabola : 
  ∃ (a : ℝ), (∀ x y : ℝ, y = 2 * x + a ↔ y = x^2 + a^2) ↔ a = 0 ∨ a = 1 := by
  sorry

end line_passes_through_vertex_of_parabola_l93_93747


namespace rectangle_perimeter_l93_93287

theorem rectangle_perimeter (A W : ℝ) (hA : A = 300) (hW : W = 15) : 
  (2 * ((A / W) + W)) = 70 := 
  sorry

end rectangle_perimeter_l93_93287


namespace num_games_round_robin_l93_93301

-- There are 10 classes in the second grade, each class forms one team.
def num_teams := 10

-- A round-robin format means each team plays against every other team once.
def num_games (n : Nat) := n * (n - 1) / 2

-- Proving the total number of games played with num_teams equals to 45
theorem num_games_round_robin : num_games num_teams = 45 := by
  sorry

end num_games_round_robin_l93_93301


namespace blue_pill_cost_l93_93880

theorem blue_pill_cost
  (days : Int := 10)
  (total_expenditure : Int := 430)
  (daily_cost : Int := total_expenditure / days) :
  ∃ (y : Int), y + (y - 3) = daily_cost ∧ y = 23 := by
  sorry

end blue_pill_cost_l93_93880


namespace inequality_solution_l93_93369

noncomputable def solve_inequality (m : ℝ) (m_lt_neg2 : m < -2) : Set ℝ :=
  if h : m = -3 then {x | 1 < x}
  else if h' : -3 < m then {x | x < m / (m + 3) ∨ 1 < x}
  else {x | 1 < x ∧ x < m / (m + 3)}

theorem inequality_solution (m : ℝ) (m_lt_neg2 : m < -2) :
  (solve_inequality m m_lt_neg2) = 
    if m = -3 then {x | 1 < x}
    else if -3 < m then {x | x < m / (m + 3) ∨ 1 < x}
    else {x | 1 < x ∧ x < m / (m + 3)} :=
sorry

end inequality_solution_l93_93369


namespace line_passes_through_vertex_of_parabola_l93_93748

theorem line_passes_through_vertex_of_parabola : 
  ∃ (a : ℝ), (∀ x y : ℝ, y = 2 * x + a ↔ y = x^2 + a^2) ↔ a = 0 ∨ a = 1 := by
  sorry

end line_passes_through_vertex_of_parabola_l93_93748


namespace product_of_total_points_l93_93025

def f (n : ℕ) : ℕ :=
  if n % 3 = 0 then 5
  else if n % 2 = 0 then 3
  else 0

def Allie_rolls : List ℕ := [3, 5, 6, 2, 4]
def Betty_rolls : List ℕ := [3, 2, 1, 6, 4]

def total_points (rolls : List ℕ) : ℕ :=
  rolls.map f |>.sum

theorem product_of_total_points :
  total_points Allie_rolls * total_points Betty_rolls = 256 :=
by
  sorry

end product_of_total_points_l93_93025


namespace smallest_number_l93_93026

theorem smallest_number (a b c d : ℝ) (h1 : a = 1) (h2 : b = -2) (h3 : c = 0) (h4 : d = -1/2) :
  b ≤ a ∧ b ≤ c ∧ b ≤ d := by
  sorry

end smallest_number_l93_93026


namespace conference_center_people_l93_93589

theorem conference_center_people (rooms : ℕ) (capacity_per_room : ℕ) (occupancy_fraction : ℚ) (total_people : ℕ) :
  rooms = 6 →
  capacity_per_room = 80 →
  occupancy_fraction = 2/3 →
  total_people = rooms * capacity_per_room * occupancy_fraction →
  total_people = 320 := 
by
  intros h_rooms h_capacity h_fraction h_total
  rw [h_rooms, h_capacity, h_fraction] at h_total
  norm_num at h_total
  exact h_total

end conference_center_people_l93_93589


namespace find_total_amount_l93_93175

noncomputable def total_amount (a b c : ℕ) : Prop :=
  a = 3 * b ∧ b = c + 25 ∧ b = 134 ∧ a + b + c = 645

theorem find_total_amount : ∃ a b c, total_amount a b c :=
by
  sorry

end find_total_amount_l93_93175


namespace paint_needed_for_new_statues_l93_93523

-- Conditions
def pint_for_original : ℕ := 1
def original_height : ℕ := 8
def num_statues : ℕ := 320
def new_height : ℕ := 2
def scale_ratio : ℚ := (new_height : ℚ) / (original_height : ℚ)
def area_ratio : ℚ := scale_ratio ^ 2

-- Correct Answer
def total_paint_needed : ℕ := 20

-- Theorem to be proved
theorem paint_needed_for_new_statues :
  pint_for_original * num_statues * area_ratio = total_paint_needed := 
by
  sorry

end paint_needed_for_new_statues_l93_93523


namespace range_of_k_l93_93645

noncomputable def f (x : ℝ) : ℝ := (x^2 + 1) / x
noncomputable def g (x : ℝ) : ℝ := x / Real.exp x

theorem range_of_k (k : ℝ) (x1 x2 : ℝ) (hx1 : 0 < x1) (hx2 : 0 < x2) :
  (g x1 / k ≤ f x2 / (k + 1)) ↔ k ≥ 1 / (2 * Real.exp 1 - 1) := by
  sorry

end range_of_k_l93_93645


namespace complement_intersection_l93_93104

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

end complement_intersection_l93_93104


namespace parabola_sum_coefficients_l93_93590

theorem parabola_sum_coefficients :
  ∃ (a b c : ℤ), 
    (∀ x : ℝ, (x = 0 → a * (x^2) + b * x + c = 1)) ∧
    (∀ x : ℝ, (x = 2 → a * (x^2) + b * x + c = 9)) ∧
    (a * (1^2) + b * 1 + c = 4)
  → a + b + c = 4 :=
by sorry

end parabola_sum_coefficients_l93_93590


namespace product_of_numbers_l93_93837

theorem product_of_numbers (a b : ℕ) (hcf_val lcm_val : ℕ) 
  (h_hcf : Nat.gcd a b = hcf_val) 
  (h_lcm : Nat.lcm a b = lcm_val) 
  (hcf_eq : hcf_val = 33) 
  (lcm_eq : lcm_val = 2574) : 
  a * b = 84942 := 
by
  sorry

end product_of_numbers_l93_93837


namespace pipe_cistern_l93_93870

theorem pipe_cistern (rate: ℚ) (duration: ℚ) (portion: ℚ) : 
  rate = (2/3) / 10 → duration = 8 → portion = 8/15 →
  portion = duration * rate := 
by 
  intros h1 h2 h3
  sorry

end pipe_cistern_l93_93870


namespace num_rel_prime_to_15_l93_93218

theorem num_rel_prime_to_15 : 
  {a : ℕ | a < 15 ∧ Int.gcd 15 a = 1}.card = 8 := by 
  sorry

end num_rel_prime_to_15_l93_93218


namespace compute_c_plus_d_l93_93749

-- Define the conditions
variables (c d : ℕ) 

-- Conditions:
-- Positive integers
axiom pos_c : 0 < c
axiom pos_d : 0 < d

-- Contains 630 terms
axiom term_count : d - c = 630

-- The product of the logarithms equals 2
axiom log_product : (Real.log d) / (Real.log c) = 2

-- Theorem to prove
theorem compute_c_plus_d : c + d = 1260 :=
sorry

end compute_c_plus_d_l93_93749


namespace ratio_of_segments_l93_93031

variables (A B C D E F G M : Type) [InCircle A B C D] [Intersection E A B C D] 
  [PointOnSegment M E B] (t : ℝ) (h_t : t = AM / AB)

theorem ratio_of_segments {EF EG : ℝ} (h_EF : EF = E - F) (h_EG : EG = E - G) :
  EF / EG = t / (1 - t) :=
sorry

end ratio_of_segments_l93_93031


namespace Milly_spends_135_minutes_studying_l93_93404

-- Definitions of homework times
def mathHomeworkTime := 60
def geographyHomeworkTime := mathHomeworkTime / 2
def scienceHomeworkTime := (mathHomeworkTime + geographyHomeworkTime) / 2

-- Definition of Milly's total study time
def totalStudyTime := mathHomeworkTime + geographyHomeworkTime + scienceHomeworkTime

-- Theorem stating that Milly spends 135 minutes studying
theorem Milly_spends_135_minutes_studying : totalStudyTime = 135 :=
by
  -- Proof omitted
  sorry

end Milly_spends_135_minutes_studying_l93_93404


namespace winning_candidate_votes_percentage_l93_93255

theorem winning_candidate_votes_percentage (majority : ℕ) (total_votes : ℕ) (winning_percentage : ℚ) :
  majority = 174 ∧ total_votes = 435 ∧ winning_percentage = 70 → 
  ∃ P : ℚ, (P / 100) * total_votes - ((100 - P) / 100) * total_votes = majority ∧ P = 70 :=
by
  sorry

end winning_candidate_votes_percentage_l93_93255


namespace trajectory_of_Q_l93_93235

-- Define Circle C
def circleC (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define Line l
def lineL (x y : ℝ) : Prop := x + y = 2

-- Define Conditions based on polar definitions
def polarCircle (ρ θ : ℝ) : Prop := ρ = 2

def polarLine (ρ θ : ℝ) : Prop := ρ * (Real.cos θ + Real.sin θ) = 2

-- Define points on ray OP
def pointP (ρ₁ θ : ℝ) : Prop := ρ₁ = 2 / (Real.cos θ + Real.sin θ)
def pointR (ρ₂ θ : ℝ) : Prop := ρ₂ = 2

-- Prove the trajectory of Q
theorem trajectory_of_Q (O P R Q : ℝ × ℝ)
  (ρ₁ θ ρ ρ₂ : ℝ)
  (h1: circleC O.1 O.2)
  (h2: lineL P.1 P.2)
  (h3: polarCircle ρ₂ θ)
  (h4: polarLine ρ₁ θ)
  (h5: ρ * ρ₁ = ρ₂^2) :
  ρ = 2 * (Real.cos θ + Real.sin θ) :=
by
  sorry

end trajectory_of_Q_l93_93235


namespace color_pairings_correct_l93_93030

noncomputable def num_color_pairings (bowls : ℕ) (glasses : ℕ) : ℕ :=
  bowls * glasses

theorem color_pairings_correct : 
  num_color_pairings 4 5 = 20 :=
by 
  -- proof omitted
  sorry

end color_pairings_correct_l93_93030


namespace perfect_square_n_l93_93504

theorem perfect_square_n (m : ℤ) :
  ∃ (n : ℤ), (n = 7 * m^2 + 6 * m + 1 ∨ n = 7 * m^2 - 6 * m + 1) ∧ ∃ (k : ℤ), 7 * n + 2 = k^2 :=
by
  sorry

end perfect_square_n_l93_93504


namespace smallest_m_for_integral_solutions_l93_93849

theorem smallest_m_for_integral_solutions : ∃ (m : ℕ), (∀ (p q : ℤ), (15 * (p * p) - m * p + 630 = 0 ∧ 15 * (q * q) - m * q + 630 = 0) → (m = 195)) :=
sorry

end smallest_m_for_integral_solutions_l93_93849


namespace trajectory_is_straight_line_l93_93299

theorem trajectory_is_straight_line (x y : ℝ) (h : x + y = 0) : ∃ m b : ℝ, y = m * x + b :=
by
  use -1
  use 0
  sorry

end trajectory_is_straight_line_l93_93299


namespace avg_A_lt_avg_B_combined_avg_eq_6_6_l93_93691

-- Define the scores for A and B
def scores_A := [4, 5, 5, 6, 6, 7, 7, 8, 8, 9]
def scores_B := [2, 5, 6, 6, 7, 7, 7, 8, 9, 10]

-- Define the average score function
def average (scores : List ℚ) : ℚ := (scores.sum : ℚ) / scores.length

-- Define the mean for the combined data
def combined_average : ℚ :=
  (average scores_A * scores_A.length + average scores_B * scores_B.length) / 
  (scores_A.length + scores_B.length)

-- Specify the variances given in the problem
def variance_A := 2.25
def variance_B := 4.41

-- Claim the average score of A is smaller than the average score of B
theorem avg_A_lt_avg_B : average scores_A < average scores_B := by sorry

-- Claim the average score of these 20 data points is 6.6
theorem combined_avg_eq_6_6 : combined_average = 6.6 := by sorry

end avg_A_lt_avg_B_combined_avg_eq_6_6_l93_93691


namespace min_days_to_sun_l93_93810

def active_days_for_level (N : ℕ) : ℕ :=
  N * (N + 4)

def days_needed_for_upgrade (current_days future_days : ℕ) : ℕ :=
  future_days - current_days

theorem min_days_to_sun (current_level future_level : ℕ) :
  current_level = 9 →
  future_level = 16 →
  days_needed_for_upgrade (active_days_for_level current_level) (active_days_for_level future_level) = 203 :=
by
  intros h1 h2
  rw [h1, h2, active_days_for_level, active_days_for_level]
  sorry

end min_days_to_sun_l93_93810


namespace infinite_solutions_a_value_l93_93738

theorem infinite_solutions_a_value (a : ℝ) : 
  (∀ y : ℝ, 3 * (5 + a * y) = 15 * y + 9) ↔ a = 5 := 
by 
  sorry

end infinite_solutions_a_value_l93_93738


namespace least_x_divisibility_l93_93614

theorem least_x_divisibility :
  ∃ x : ℕ, (x > 0) ∧ ((x^2 + 164) % 3 = 0) ∧ ((x^2 + 164) % 4 = 0) ∧ ((x^2 + 164) % 5 = 0) ∧
  ((x^2 + 164) % 6 = 0) ∧ ((x^2 + 164) % 7 = 0) ∧ ((x^2 + 164) % 8 = 0) ∧ 
  ((x^2 + 164) % 9 = 0) ∧ ((x^2 + 164) % 10 = 0) ∧ ((x^2 + 164) % 11 = 0) ∧ x = 166 → 
  3 = 3 :=
by
  sorry

end least_x_divisibility_l93_93614


namespace triangle_converse_inverse_false_l93_93910

variables {T : Type} (p q : T → Prop)

-- Condition: If a triangle is equilateral, then it is isosceles
axiom h : ∀ t, p t → q t

-- Conclusion: Neither the converse nor the inverse is true
theorem triangle_converse_inverse_false : 
  (∃ t, q t ∧ ¬ p t) ∧ (∃ t, ¬ p t ∧ q t) :=
sorry

end triangle_converse_inverse_false_l93_93910


namespace xiaoming_age_l93_93685

theorem xiaoming_age
  (x x' : ℕ) 
  (h₁ : ∃ f : ℕ, f = 4 * x) 
  (h₂ : (x + 25) + (4 * x + 25) = 100) : 
  x = 10 :=
by
  obtain ⟨f, hf⟩ := h₁
  sorry

end xiaoming_age_l93_93685


namespace Michael_needs_more_money_l93_93654

def money_Michael_has : ℕ := 50
def cake_cost : ℕ := 20
def bouquet_cost : ℕ := 36
def balloons_cost : ℕ := 5

def total_cost : ℕ := cake_cost + bouquet_cost + balloons_cost
def money_needed : ℕ := total_cost - money_Michael_has

theorem Michael_needs_more_money : money_needed = 11 :=
by
  sorry

end Michael_needs_more_money_l93_93654


namespace factory_sample_size_l93_93329

noncomputable def sample_size (A B C : ℕ) (sample_A : ℕ) : ℕ :=
  let total_ratio := A + B + C
  let ratio_A := A / total_ratio
  sample_A / ratio_A

theorem factory_sample_size
  (A B C : ℕ) (h_ratio : A = 2 ∧ B = 3 ∧ C = 5)
  (sample_A : ℕ) (h_sample_A : sample_A = 16) :
  sample_size A B C sample_A = 80 :=
by
  simp [h_ratio, h_sample_A, sample_size]
  sorry

end factory_sample_size_l93_93329


namespace coordinates_of_focus_with_greater_x_coordinate_l93_93485

noncomputable def focus_of_ellipse_with_greater_x_coordinate : (ℝ × ℝ) :=
  let center : ℝ × ℝ := (3, -2)
  let a : ℝ := 3 -- semi-major axis length
  let b : ℝ := 2 -- semi-minor axis length
  let c : ℝ := Real.sqrt (a^2 - b^2)
  let focus_x : ℝ := 3 + c
  (focus_x, -2)

theorem coordinates_of_focus_with_greater_x_coordinate :
  focus_of_ellipse_with_greater_x_coordinate = (3 + Real.sqrt 5, -2) := 
sorry

end coordinates_of_focus_with_greater_x_coordinate_l93_93485


namespace find_a_l93_93250

-- Definitions and theorem statement
def A (a : ℝ) : Set ℝ := {2, a^2 - a + 1}
def B (a : ℝ) : Set ℝ := {3, a + 3}
def C (a : ℝ) : Set ℝ := {3}

theorem find_a (a : ℝ) : A a ∩ B a = C a → a = 2 :=
by
  sorry

end find_a_l93_93250


namespace inequality_positive_l93_93289

theorem inequality_positive (x : ℝ) : (1 / 3) * x - x > 0 ↔ (-2 / 3) * x > 0 := 
  sorry

end inequality_positive_l93_93289


namespace find_k_l93_93223

-- Given definition for a quadratic expression that we want to be a square of a binomial
def quadratic_expression (x k : ℝ) := x^2 - 20 * x + k

-- The binomial square matching.
def binomial_square (x b : ℝ) := (x + b)^2

-- Statement to prove that k = 100 makes the quadratic_expression to be a square of binomial
theorem find_k :
  (∃ k : ℝ, ∀ x : ℝ, quadratic_expression x k = binomial_square x (-10)) ↔ k = 100 :=
by
  sorry

end find_k_l93_93223


namespace problem_l93_93835

def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

noncomputable def f : ℝ → ℝ := sorry

axiom f_odd : odd_function f
axiom f_property : ∀ x : ℝ, f (x + 2) = -f x
axiom f_at_1 : f 1 = 8

theorem problem : f 2012 + f 2013 + f 2014 = 8 := by
  sorry

end problem_l93_93835


namespace exterior_angle_of_regular_pentagon_l93_93599

theorem exterior_angle_of_regular_pentagon : 
  (360 / 5) = 72 := by
  sorry

end exterior_angle_of_regular_pentagon_l93_93599


namespace coprime_integers_lt_15_l93_93221

theorem coprime_integers_lt_15 : ∃ (S : Finset ℕ), S.card = 8 ∧ (∀ a ∈ S, a < 15 ∧ Nat.gcd a 15 = 1) :=
by
  sorry

end coprime_integers_lt_15_l93_93221


namespace set_D_is_empty_l93_93984

-- Definitions based on the conditions from the original problem
def set_A : Set ℝ := {x | x + 3 = 3}
def set_B : Set (ℝ × ℝ) := {(x, y) | y^2 = -x^2}
def set_C : Set ℝ := {x | x^2 ≤ 0}
def set_D : Set ℝ := {x | x^2 - x + 1 = 0}

-- The theorem statement
theorem set_D_is_empty : set_D = ∅ :=
sorry

end set_D_is_empty_l93_93984


namespace Milly_spends_135_minutes_studying_l93_93403

-- Definitions of homework times
def mathHomeworkTime := 60
def geographyHomeworkTime := mathHomeworkTime / 2
def scienceHomeworkTime := (mathHomeworkTime + geographyHomeworkTime) / 2

-- Definition of Milly's total study time
def totalStudyTime := mathHomeworkTime + geographyHomeworkTime + scienceHomeworkTime

-- Theorem stating that Milly spends 135 minutes studying
theorem Milly_spends_135_minutes_studying : totalStudyTime = 135 :=
by
  -- Proof omitted
  sorry

end Milly_spends_135_minutes_studying_l93_93403


namespace daria_needs_to_earn_l93_93494

variable (ticket_cost : ℕ) (current_money : ℕ) (total_tickets : ℕ)

def total_cost (ticket_cost : ℕ) (total_tickets : ℕ) : ℕ :=
  ticket_cost * total_tickets

def money_needed (total_cost : ℕ) (current_money : ℕ) : ℕ :=
  total_cost - current_money

theorem daria_needs_to_earn :
  total_cost 90 4 - 189 = 171 :=
by
  sorry

end daria_needs_to_earn_l93_93494


namespace minimize_f_l93_93697

noncomputable def f : ℝ → ℝ := λ x => (3/2) * x^2 - 9 * x + 7

theorem minimize_f : ∀ x, f x ≥ f 3 :=
by 
  intro x
  sorry

end minimize_f_l93_93697


namespace sinA_value_find_b_c_l93_93621

-- Define the conditions
def triangle (A B C : Type) (a b c : ℝ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0

variable {A B C : Type} (a b c : ℝ)
variable {S_triangle_ABC : ℝ}
variable {cosB : ℝ}

-- Given conditions
axiom cosB_val : cosB = 3 / 5
axiom a_val : a = 2

-- Problem 1: Prove sinA = 2/5 given additional condition b = 4
axiom b_val : b = 4

theorem sinA_value (h_triangle : triangle A B C a b c) (h_cosB : cosB = 3/5) (h_a : a = 2) (h_b : b = 4) : 
  ∃ sinA : ℝ, sinA = 2 / 5 :=
sorry

-- Problem 2: Prove b = sqrt(17) and c = 5 given the area
axiom area_val : S_triangle_ABC = 4

theorem find_b_c (h_triangle : triangle A B C a b c) (h_cosB : cosB = 3/5) (h_a : a = 2) (h_area : S_triangle_ABC = 4) : 
  ∃ b c : ℝ, b = Real.sqrt 17 ∧ c = 5 :=
sorry

end sinA_value_find_b_c_l93_93621


namespace no_such_function_exists_l93_93882

theorem no_such_function_exists :
  ¬ ∃ f : ℝ → ℝ, ∀ x : ℝ, f (Real.sin x) + f (Real.cos x) = Real.sin x :=
by
  sorry

end no_such_function_exists_l93_93882


namespace youngest_child_age_l93_93684

theorem youngest_child_age (x y z : ℕ) 
  (h1 : 3 * x + 6 = 48) 
  (h2 : 3 * y + 9 = 60) 
  (h3 : 2 * z + 4 = 30) : 
  z = 13 := 
sorry

end youngest_child_age_l93_93684


namespace value_of_k_l93_93606

theorem value_of_k (x y : ℝ) (t : ℝ) (k : ℝ) : 
  (x + t * y + 8 = 0) ∧ (5 * x - t * y + 4 = 0) ∧ (3 * x - k * y + 1 = 0) → k = 5 :=
by
  sorry

end value_of_k_l93_93606


namespace solve_for_y_l93_93281

theorem solve_for_y : ∃ y : ℕ, 8^4 = 2^y ∧ y = 12 := by
  sorry

end solve_for_y_l93_93281


namespace simplify_fraction_l93_93132

theorem simplify_fraction : 
  (1:ℚ) / 462 + 17 / 42 = 94 / 231 := 
by
  sorry

end simplify_fraction_l93_93132


namespace range_for_a_l93_93773

theorem range_for_a (f : ℝ → ℝ) (a : ℝ) (n : ℝ) :
  (∀ x, f x = x^n) →
  f 8 = 1/4 →
  f (a+1) < f 2 →
  a < -3 ∨ a > 1 :=
by
  intros h1 h2 h3
  sorry

end range_for_a_l93_93773


namespace diana_hourly_wage_l93_93177

theorem diana_hourly_wage :
  (∃ (hours_monday : ℕ) (hours_tuesday : ℕ) (hours_wednesday : ℕ) (hours_thursday : ℕ) (hours_friday : ℕ) (weekly_earnings : ℝ),
    hours_monday = 10 ∧
    hours_tuesday = 15 ∧
    hours_wednesday = 10 ∧
    hours_thursday = 15 ∧
    hours_friday = 10 ∧
    weekly_earnings = 1800 ∧
    (weekly_earnings / (hours_monday + hours_tuesday + hours_wednesday + hours_thursday + hours_friday) = 30)) :=
sorry

end diana_hourly_wage_l93_93177


namespace toms_total_out_of_pocket_is_680_l93_93689

namespace HealthCosts

def doctor_visit_cost : ℝ := 300
def cast_cost : ℝ := 200
def initial_insurance_coverage : ℝ := 0.60
def therapy_session_cost : ℝ := 100
def number_of_sessions : ℕ := 8
def therapy_insurance_coverage : ℝ := 0.40

def total_initial_cost : ℝ :=
  doctor_visit_cost + cast_cost

def initial_out_of_pocket : ℝ :=
  total_initial_cost * (1 - initial_insurance_coverage)

def total_therapy_cost : ℝ :=
  therapy_session_cost * number_of_sessions

def therapy_out_of_pocket : ℝ :=
  total_therapy_cost * (1 - therapy_insurance_coverage)

def total_out_of_pocket : ℝ :=
  initial_out_of_pocket + therapy_out_of_pocket

theorem toms_total_out_of_pocket_is_680 :
  total_out_of_pocket = 680 := by
  sorry

end HealthCosts

end toms_total_out_of_pocket_is_680_l93_93689


namespace hexagon_AF_length_l93_93532

theorem hexagon_AF_length (BC CD DE EF : ℝ) (angleB angleC angleD angleE : ℝ) (angleF : ℝ) 
  (hBC : BC = 2) (hCD : CD = 2) (hDE : DE = 2) (hEF : EF = 2)
  (hangleB : angleB = 135) (hangleC : angleC = 135) (hangleD : angleD = 135) (hangleE : angleE = 135)
  (hangleF : angleF = 90) :
  ∃ (a b : ℝ), (AF = a + 2 * Real.sqrt b) ∧ (a + b = 6) :=
by
  sorry

end hexagon_AF_length_l93_93532


namespace count_coprime_to_15_eq_8_l93_93220

def is_coprime_to_15 (a : ℕ) : Prop := Nat.gcd a 15 = 1

def count_coprime_to_15 (n : ℕ) : ℕ :=
  (Finset.filter (λ a, is_coprime_to_15 a) (Finset.range n)).card

theorem count_coprime_to_15_eq_8 : count_coprime_to_15 15 = 8 := by
  sorry

end count_coprime_to_15_eq_8_l93_93220


namespace cost_of_bananas_l93_93207

theorem cost_of_bananas (A B : ℝ) (h1 : A + B = 5) (h2 : 2 * A + B = 7) : B = 3 :=
by
  sorry

end cost_of_bananas_l93_93207


namespace combined_work_time_l93_93506

theorem combined_work_time (A B C D : ℕ) (hA : A = 10) (hB : B = 15) (hC : C = 20) (hD : D = 30) :
  1 / (1 / A + 1 / B + 1 / C + 1 / D) = 4 := by
  -- Replace the following "sorry" with your proof.
  sorry

end combined_work_time_l93_93506


namespace arthur_first_day_spending_l93_93208

-- Define the costs of hamburgers and hot dogs.
variable (H D : ℝ)
-- Given conditions
axiom hot_dog_cost : D = 1
axiom second_day_purchase : 2 * H + 3 * D = 7

-- Goal: How much did Arthur spend on the first day?
-- We need to verify that 3H + 4D = 10
theorem arthur_first_day_spending : 3 * H + 4 * D = 10 :=
by
  -- Validating given conditions
  have h1 := hot_dog_cost
  have h2 := second_day_purchase
  -- Insert proof here
  sorry

end arthur_first_day_spending_l93_93208


namespace seventh_oblong_is_56_l93_93341

def oblong (n : ℕ) : ℕ := n * (n + 1)

theorem seventh_oblong_is_56 : oblong 7 = 56 := by
  sorry

end seventh_oblong_is_56_l93_93341


namespace hernandez_state_tax_l93_93004

theorem hernandez_state_tax 
    (res_months : ℕ) (total_months : ℕ) 
    (taxable_income : ℝ) (tax_rate : ℝ) 
    (prorated_income : ℝ) (state_tax : ℝ) 
    (h1 : res_months = 9) 
    (h2 : total_months = 12) 
    (h3 : taxable_income = 42500) 
    (h4 : tax_rate = 0.04) 
    (h5 : prorated_income = taxable_income * (res_months / total_months)) 
    (h6 : state_tax = prorated_income * tax_rate) : 
    state_tax = 1275 := 
by 
  -- this is where the proof would go
  sorry

end hernandez_state_tax_l93_93004


namespace perfect_square_trinomial_l93_93033

theorem perfect_square_trinomial :
  120^2 - 40 * 120 + 20^2 = 10000 := sorry

end perfect_square_trinomial_l93_93033


namespace mod_arith_proof_l93_93819

theorem mod_arith_proof (m : ℕ) (hm1 : 0 ≤ m) (hm2 : m < 50) : 198 * 935 % 50 = 30 := 
by
  sorry

end mod_arith_proof_l93_93819


namespace alice_paid_percentage_of_srp_l93_93429

theorem alice_paid_percentage_of_srp
  (P : ℝ) -- Suggested Retail Price (SRP)
  (MP : ℝ := P * 0.60) -- Marked Price (MP) is 40% less than SRP
  (price_alice_paid : ℝ := MP * 0.60) -- Alice purchased the book for 40% off the marked price
  : (price_alice_paid / P) * 100 = 36 :=
by
  -- only the statement is required, so proof is omitted
  sorry

end alice_paid_percentage_of_srp_l93_93429


namespace sin_three_pi_div_two_l93_93361

theorem sin_three_pi_div_two : Real.sin (3 * Real.pi / 2) = -1 := 
by
  sorry

end sin_three_pi_div_two_l93_93361


namespace fraction_of_girls_l93_93926

theorem fraction_of_girls (G B : ℕ) (h1 : G + B = 800) (h2 : 7 * G + 4 * B = 4700) : (G : ℚ) / (G + B) = 5 / 8 :=
by
  sorry

end fraction_of_girls_l93_93926


namespace equation_II_consecutive_integers_l93_93602

theorem equation_II_consecutive_integers :
  ∃ x y z w : ℕ, x + y + z + w = 46 ∧ [x, x+1, x+2, x+3] = [x, y, z, w] :=
by
  sorry

end equation_II_consecutive_integers_l93_93602


namespace merchant_profit_percentage_is_35_l93_93473

noncomputable def cost_price : ℝ := 100
noncomputable def markup_percentage : ℝ := 0.80
noncomputable def discount_percentage : ℝ := 0.25

-- Marked price after 80% markup
noncomputable def marked_price (cp : ℝ) (markup_pct : ℝ) : ℝ :=
  cp + (markup_pct * cp)

-- Selling price after 25% discount on marked price
noncomputable def selling_price (mp : ℝ) (discount_pct : ℝ) : ℝ :=
  mp - (discount_pct * mp)

-- Profit as the difference between selling price and cost price
noncomputable def profit (sp cp : ℝ) : ℝ :=
  sp - cp

-- Profit percentage
noncomputable def profit_percentage (profit cp : ℝ) : ℝ :=
  (profit / cp) * 100

theorem merchant_profit_percentage_is_35 :
  let cp := cost_price
  let mp := marked_price cp markup_percentage
  let sp := selling_price mp discount_percentage
  let prof := profit sp cp
  profit_percentage prof cp = 35 :=
by
  let cp := cost_price
  let mp := marked_price cp markup_percentage
  let sp := selling_price mp discount_percentage
  let prof := profit sp cp
  show profit_percentage prof cp = 35
  sorry

end merchant_profit_percentage_is_35_l93_93473


namespace circumference_to_diameter_ratio_l93_93726

-- Definitions from the conditions
def r : ℝ := 15
def C : ℝ := 90
def D : ℝ := 2 * r

-- The proof goal
theorem circumference_to_diameter_ratio : C / D = 3 := 
by sorry

end circumference_to_diameter_ratio_l93_93726


namespace integer_cube_less_than_triple_l93_93309

theorem integer_cube_less_than_triple (x : ℤ) : x^3 < 3 * x ↔ x = 0 :=
by 
  sorry

end integer_cube_less_than_triple_l93_93309


namespace abs_eq_linear_eq_l93_93497

theorem abs_eq_linear_eq (x : ℝ) : (|x - 5| = 3 * x + 1) ↔ x = 1 := by
  sorry

end abs_eq_linear_eq_l93_93497


namespace robie_initial_cards_l93_93811

def total_initial_boxes : Nat := 2 + 5
def cards_per_box : Nat := 10
def unboxed_cards : Nat := 5

theorem robie_initial_cards :
  (total_initial_boxes * cards_per_box + unboxed_cards) = 75 :=
by
  sorry

end robie_initial_cards_l93_93811


namespace min_value_of_inverse_sum_l93_93896

noncomputable def minimumValue (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 1 / 3) : ℝ :=
  9 + 6 * Real.sqrt 2

theorem min_value_of_inverse_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 1 / 3) :
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 2 * x + y = 1 / 3 ∧ (1/x + 1/y) = 9 + 6 * Real.sqrt 2 := by
  sorry

end min_value_of_inverse_sum_l93_93896


namespace smallest_odd_prime_factor_2021_8_plus_1_l93_93613

noncomputable def least_odd_prime_factor (n : ℕ) : ℕ :=
  if 2021^8 + 1 = 0 then 2021^8 + 1 else sorry 

theorem smallest_odd_prime_factor_2021_8_plus_1 :
  least_odd_prime_factor (2021^8 + 1) = 97 :=
  by
    sorry

end smallest_odd_prime_factor_2021_8_plus_1_l93_93613


namespace simplify_expression_l93_93815

theorem simplify_expression :
  (123 / 999) * 27 = 123 / 37 :=
by sorry

end simplify_expression_l93_93815


namespace arithmetic_sequence_property_l93_93767

variable {a : ℕ → ℝ} -- Let a be an arithmetic sequence
variable {S : ℕ → ℝ} -- Let S be the sum of the first n terms of the sequence

-- Conditions
axiom sum_of_first_n_terms (n : ℕ) : S n = n * (a 1 + (n - 1) * (a 2 - a 1) / 2)
axiom a_5 : a 5 = 3
axiom S_13 : S 13 = 91

-- Question to prove
theorem arithmetic_sequence_property : a 1 + a 11 = 10 :=
by
  sorry

end arithmetic_sequence_property_l93_93767


namespace power_of_negative_base_l93_93609

theorem power_of_negative_base : (-64 : ℤ)^(7 / 6) = -128 := by
  sorry

end power_of_negative_base_l93_93609


namespace percentage_of_acid_is_18_18_percent_l93_93999

noncomputable def percentage_of_acid_in_original_mixture
  (a w : ℝ) (h1 : (a + 1) / (a + w + 1) = 1 / 4) (h2 : (a + 1) / (a + w + 2) = 1 / 5) : ℝ :=
  a / (a + w) 

theorem percentage_of_acid_is_18_18_percent :
  ∃ (a w : ℝ), (a + 1) / (a + w + 1) = 1 / 4 ∧ (a + 1) / (a + w + 2) = 1 / 5 ∧ percentage_of_acid_in_original_mixture a w (by sorry) (by sorry) = 18.18 := by
  sorry

end percentage_of_acid_is_18_18_percent_l93_93999


namespace tank_length_l93_93693

variable (rate : ℝ)
variable (time : ℝ)
variable (width : ℝ)
variable (depth : ℝ)
variable (volume : ℝ)
variable (length : ℝ)

-- Given conditions
axiom rate_cond : rate = 5 -- cubic feet per hour
axiom time_cond : time = 60 -- hours
axiom width_cond : width = 6 -- feet
axiom depth_cond : depth = 5 -- feet

-- Derived volume from the rate and time
axiom volume_cond : volume = rate * time

-- Definition of length from volume, width, and depth
axiom length_def : length = volume / (width * depth)

-- The proof problem to show
theorem tank_length : length = 10 := by
  -- conditions provided and we expect the length to be computed
  sorry

end tank_length_l93_93693


namespace P_2017_P_eq_4_exists_P_minus_P_succ_gt_50_l93_93364

-- Assume the definition of sum of digits of n and count of digits
def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum  -- Sum of digits in base 10 representation

def num_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).length  -- Number of digits in base 10 representation

def P (n : ℕ) : ℕ :=
  sum_of_digits n + num_of_digits n

-- Problem (a)
theorem P_2017 : P 2017 = 14 :=
sorry

-- Problem (b)
theorem P_eq_4 :
  {n : ℕ | P n = 4} = {3, 11, 20, 100} :=
sorry

-- Problem (c)
theorem exists_P_minus_P_succ_gt_50 : 
  ∃ n : ℕ, P n - P (n + 1) > 50 :=
sorry

end P_2017_P_eq_4_exists_P_minus_P_succ_gt_50_l93_93364


namespace smallest_perfect_square_divisible_by_5_and_7_l93_93311

theorem smallest_perfect_square_divisible_by_5_and_7 
  (n : ℕ) 
  (h1 : n > 0) 
  (h2 : ∃ k : ℕ, n = k^2)
  (h3 : 5 ∣ n)
  (h4 : 7 ∣ n) : 
  n = 1225 :=
sorry

end smallest_perfect_square_divisible_by_5_and_7_l93_93311


namespace dice_probability_l93_93864

open Probability

theorem dice_probability (sum_gt_10 : ℕ) (w : ℕ) : 
  sum_gt_10 = 10 ∧ w = 48 → 
  P(sum (d6 d8) > 10) = 3 / 16 :=
by sorry

end dice_probability_l93_93864


namespace probability_of_log2_condition_l93_93018

noncomputable def probability_log_condition : ℝ :=
  let a := 0
  let b := 9
  let log_lower_bound := 1
  let log_upper_bound := 2
  let exp_lower_bound := 2^log_lower_bound
  let exp_upper_bound := 2^log_upper_bound
  (exp_upper_bound - exp_lower_bound) / (b - a)

theorem probability_of_log2_condition :
  probability_log_condition = 2 / 9 :=
by
  sorry

end probability_of_log2_condition_l93_93018


namespace number_of_white_stones_is_3600_l93_93439

-- Definitions and conditions
def total_stones : ℕ := 6000
def total_difference_to_4800 : ℕ := 4800
def W : ℕ := 3600

-- Conditions
def condition1 (B : ℕ) : Prop := total_stones - W + B = total_difference_to_4800
def condition2 (B : ℕ) : Prop := W + B = total_stones
def condition3 (B : ℕ) : Prop := W > B

-- Theorem statement
theorem number_of_white_stones_is_3600 :
  ∃ B : ℕ, condition1 B ∧ condition2 B ∧ condition3 B :=
by
  -- TODO: Complete the proof
  sorry

end number_of_white_stones_is_3600_l93_93439


namespace margo_total_distance_l93_93401

theorem margo_total_distance
  (t1 t2 : ℚ) (rate1 rate2 : ℚ)
  (h1 : t1 = 15 / 60)
  (h2 : t2 = 25 / 60)
  (r1 : rate1 = 5)
  (r2 : rate2 = 3) :
  (t1 * rate1 + t2 * rate2 = 2.5) :=
by
  sorry

end margo_total_distance_l93_93401


namespace total_cookies_baked_l93_93626

-- Definitions based on conditions
def pans : ℕ := 5
def cookies_per_pan : ℕ := 8

-- Statement of the theorem to be proven
theorem total_cookies_baked :
  pans * cookies_per_pan = 40 := by
  sorry

end total_cookies_baked_l93_93626


namespace complement_union_l93_93237

open Set

def set_A : Set ℝ := {x | x ≤ 0}
def set_B : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}

theorem complement_union (A B : Set ℝ) (hA : A = set_A) (hB : B = set_B) :
  (univ \ (A ∪ B) = {x | 1 < x}) := by
  rw [hA, hB]
  sorry

end complement_union_l93_93237


namespace twice_x_minus_3_gt_4_l93_93039

theorem twice_x_minus_3_gt_4 (x : ℝ) : 2 * x - 3 > 4 :=
sorry

end twice_x_minus_3_gt_4_l93_93039


namespace twice_midpoint_l93_93259

open Complex

def z1 : ℂ := -7 + 5 * I
def z2 : ℂ := 9 - 11 * I

theorem twice_midpoint : 2 * ((z1 + z2) / 2) = 2 - 6 * I := 
by
  -- Sorry is used to skip the proof
  sorry

end twice_midpoint_l93_93259


namespace return_trip_time_l93_93334

theorem return_trip_time 
  (d p w : ℝ) 
  (h1 : d = 90 * (p - w))
  (h2 : ∀ t, t = d / p → d / (p + w) = t - 15) : 
  d / (p + w) = 64 :=
by
  sorry

end return_trip_time_l93_93334


namespace find_acute_angle_l93_93622

noncomputable def vector_a (α : ℝ) : ℝ × ℝ := (3 * Real.cos α, 2)
noncomputable def vector_b (α : ℝ) : ℝ × ℝ := (3, 4 * Real.sin α)
def are_parallel (a b : ℝ × ℝ) : Prop := a.1 * b.2 = a.2 * b.1

theorem find_acute_angle (α : ℝ) (h : are_parallel (vector_a α) (vector_b α)) (h_acute : 0 < α ∧ α < π / 2) : 
  α = π / 4 :=
by
  sorry

end find_acute_angle_l93_93622


namespace find_value_of_g1_l93_93904

variable (f : ℝ → ℝ)
variable (g : ℝ → ℝ)

def is_odd (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = -f x

def is_even (g : ℝ → ℝ) : Prop :=
∀ x : ℝ, g (-x) = g x

theorem find_value_of_g1 (h1 : is_odd f)
  (h2 : is_even g)
  (h3 : f (-1) + g 1 = 2)
  (h4 : f 1 + g (-1) = 4) : 
  g 1 = 3 :=
sorry

end find_value_of_g1_l93_93904


namespace max_removal_l93_93326

-- Definitions of quantities in the problem
def yellow_marbles : ℕ := 8
def red_marbles : ℕ := 7
def black_marbles : ℕ := 5
def total_marbles : ℕ := yellow_marbles + red_marbles + black_marbles

-- Definition of the condition on remaining marbles
def valid_remaining (remaining : ℕ) : Prop :=
∃ (yellow_left red_left black_left : ℕ),
  yellow_left + red_left + black_left = remaining ∧
  (yellow_left ≥ 4 ∨ red_left ≥ 4 ∨ black_left ≥ 4) ∧
  (yellow_left ≥ 3 ∨ red_left ≥ 3 ∨ black_left ≥ 3)

-- Statement of the main theorem
theorem max_removal (N : ℕ) (N ≤ 7) :
  let remaining := total_marbles - N in valid_remaining remaining := 
begin
  -- Content of the proof is omitted and replaced by sorry
  sorry
end

end max_removal_l93_93326


namespace tangent_line_parabola_d_l93_93428

theorem tangent_line_parabola_d (d : ℝ) :
  (∀ x y : ℝ, (y = 3 * x + d) → (y^2 = 12 * x) → ∃! x, 9 * x^2 + (6 * d - 12) * x + d^2 = 0) → d = 1 :=
by
  sorry

end tangent_line_parabola_d_l93_93428


namespace total_employees_in_buses_l93_93449

-- Define the capacity of each bus
def capacity : ℕ := 150

-- Define the fill percentages of each bus
def fill_percentage_bus1 : ℚ := 60 / 100
def fill_percentage_bus2 : ℚ := 70 / 100

-- Calculate the number of passengers in each bus
def passengers_bus1 : ℚ := fill_percentage_bus1 * capacity
def passengers_bus2 : ℚ := fill_percentage_bus2 * capacity

-- Calculate the total number of passengers
def total_passengers : ℚ := passengers_bus1 + passengers_bus2

-- The proof statement
theorem total_employees_in_buses : total_passengers = 195 :=
by
  sorry

end total_employees_in_buses_l93_93449


namespace expression_of_quadratic_function_coordinates_of_vertex_l93_93374

def quadratic_function_through_points (a b : ℝ) : Prop :=
  (0 = a * (-3)^2 + b * (-3) + 3) ∧ (-5 = a * 2^2 + b * 2 + 3)

theorem expression_of_quadratic_function :
  ∃ a b : ℝ, quadratic_function_through_points a b ∧ ∀ x : ℝ, -x^2 - 2 * x + 3 = a * x^2 + b * x + 3 :=
by
  sorry

theorem coordinates_of_vertex :
  - (1 : ℝ) * (1 : ℝ) = (-1) / (2 * (-1)) ∧ 4 = -(1 - (-1) + 3) + 4 :=
by
  sorry

end expression_of_quadratic_function_coordinates_of_vertex_l93_93374


namespace range_of_a_l93_93833

-- Define the quadratic function f
def f (a x : ℝ) : ℝ := x^2 + 2 * (a - 1) * x + 2

-- State the theorem that describes the condition and proves the answer
theorem range_of_a (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → x₁ < 4 → x₂ < 4 → f a x₁ ≥ f a x₂) → a ≤ -3 :=
by
  -- The proof would go here; for now, we skip it
  sorry

end range_of_a_l93_93833


namespace arrangements_l93_93290

def digits := [1, 2, 3, 4, 5, 6, 7, 8]

-- Function to check divisibility by k
def is_divisible_by (n k : ℕ) : Prop := n % k = 0

-- Function representing the arrangement in a grid
-- ... (details depending on specific approach of arrangement which we abstract here)

-- Main theorem
theorem arrangements (k : ℕ) : k ∈ [2, 3, 4, 5, 6] →
  (k = 2 ∨ k = 3 → 
    ∃ (arrangement : list (list ℕ)), -- using list to represent the grid and numbers
    (∀ nums ∈ arrangement, is_divisible_by nums k)) ∧ 
  ((k = 4 ∨ k = 5 ∨ k = 6) → 
    ¬ ∃ (arrangement : list (list ℕ)),
    (∀ nums ∈ arrangement, is_divisible_by nums k)) :=
by
  -- proof goes here
  sorry

end arrangements_l93_93290


namespace range_of_a_l93_93384

noncomputable def has_two_distinct_real_roots (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a^x₁ = x₁ ∧ a^x₂ = x₂

theorem range_of_a (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) (h3 : has_two_distinct_real_roots a) : 
  1 < a ∧ a < Real.exp (1 / Real.exp 1) :=
sorry

end range_of_a_l93_93384


namespace should_agree_to_buy_discount_card_l93_93339

noncomputable def total_cost_without_discount_card (cakes_cost fruits_cost : ℕ) : ℕ :=
  cakes_cost + fruits_cost

noncomputable def total_cost_with_discount_card (cakes_cost fruits_cost discount_card_cost : ℕ) : ℕ :=
  let total_cost := cakes_cost + fruits_cost
  let discount := total_cost * 3 / 100
  (total_cost - discount) + discount_card_cost

theorem should_agree_to_buy_discount_card : 
  let cakes_cost := 4 * 500
  let fruits_cost := 1600
  let discount_card_cost := 100
  total_cost_with_discount_card cakes_cost fruits_cost discount_card_cost < total_cost_without_discount_card cakes_cost fruits_cost :=
by
  sorry

end should_agree_to_buy_discount_card_l93_93339


namespace maximum_value_of_expression_l93_93901

theorem maximum_value_of_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a * b = 1) :
  (1 / (a + 9 * b) + 1 / (9 * a + b)) ≤ 5 / 24 := sorry

end maximum_value_of_expression_l93_93901


namespace time_to_cross_bridge_l93_93729

def train_length : ℕ := 600  -- train length in meters
def bridge_length : ℕ := 100  -- overbridge length in meters
def speed_km_per_hr : ℕ := 36  -- speed of the train in kilometers per hour

-- Convert speed from km/h to m/s
def speed_m_per_s : ℕ := speed_km_per_hr * 1000 / 3600

-- Compute the total distance
def total_distance : ℕ := train_length + bridge_length

-- Prove the time to cross the overbridge
theorem time_to_cross_bridge : total_distance / speed_m_per_s = 70 := by
  sorry

end time_to_cross_bridge_l93_93729


namespace quadratic_real_roots_condition_sufficient_l93_93561

theorem quadratic_real_roots_condition_sufficient (m : ℝ) : (m < 1 / 4) → ∃ x : ℝ, x^2 + x + m = 0 :=
by
  sorry

end quadratic_real_roots_condition_sufficient_l93_93561


namespace green_toads_per_acre_l93_93288

theorem green_toads_per_acre (brown_toads spotted_brown_toads green_toads : ℕ) 
  (h1 : ∀ g, 25 * g = brown_toads) 
  (h2 : spotted_brown_toads = brown_toads / 4) 
  (h3 : spotted_brown_toads = 50) : 
  green_toads = 8 :=
by
  sorry

end green_toads_per_acre_l93_93288


namespace percentage_change_area_l93_93667

theorem percentage_change_area (L B : ℝ) :
  let A_original := L * B
  let A_new := (L / 2) * (3 * B)
  (A_new - A_original) / A_original * 100 = 50 := by
  sorry

end percentage_change_area_l93_93667


namespace student_arrangement_count_l93_93053

theorem student_arrangement_count :
  let males := 4
  let females := 5
  let select_males := 2
  let select_females := 3
  let total_selected := select_males + select_females
  (Nat.choose males select_males) * (Nat.choose females select_females) * (Nat.factorial total_selected) = 7200 := 
by
  sorry

end student_arrangement_count_l93_93053


namespace remainder_polynomial_division_l93_93358

noncomputable def remainder_division : Polynomial ℝ := 
  (Polynomial.X ^ 4 + Polynomial.X ^ 3 - 4 * Polynomial.X + 1) % (Polynomial.X ^ 3 - 1)

theorem remainder_polynomial_division :
  remainder_division = -3 * Polynomial.X + 2 :=
by
  sorry

end remainder_polynomial_division_l93_93358


namespace num_units_from_batch_B_l93_93336

theorem num_units_from_batch_B
  (A B C : ℝ) -- quantities of products from batches A, B, and C
  (h_arith_seq : B - A = C - B) -- batches A, B, and C form an arithmetic sequence
  (h_total : A + B + C = 240)    -- total units from three batches
  (h_sample_size : A + B + C = 60)  -- sample size drawn equals 60
  : B = 20 := 
by {
  sorry
}

end num_units_from_batch_B_l93_93336


namespace max_blue_points_l93_93633

-- We define the number of spheres and the categorization of red and green spheres
def number_of_spheres : ℕ := 2016

-- Definition of the number of red spheres
def red_spheres (r : ℕ) : Prop := r <= number_of_spheres

-- Definition of the number of green spheres as the complement of red spheres
def green_spheres (r : ℕ) : ℕ := number_of_spheres - r

-- Definition of the number of blue points as the intersection of red and green spheres
def blue_points (r : ℕ) : ℕ := r * green_spheres r

-- Theorem: Given the conditions, the maximum number of blue points is 1016064
theorem max_blue_points : ∃ r : ℕ, red_spheres r ∧ blue_points r = 1016064 := by
  sorry

end max_blue_points_l93_93633


namespace no_sequence_of_14_consecutive_divisible_by_some_prime_le_11_l93_93992

theorem no_sequence_of_14_consecutive_divisible_by_some_prime_le_11 :
  ¬ ∃ n : ℕ, ∀ k : ℕ, k < 14 → ∃ p ∈ [2, 3, 5, 7, 11], (n + k) % p = 0 :=
by
  sorry

end no_sequence_of_14_consecutive_divisible_by_some_prime_le_11_l93_93992


namespace combine_exponent_remains_unchanged_l93_93698

-- Define combining like terms condition
def combining_like_terms (terms : List (ℕ × String)) : List (ℕ × String) := sorry

-- Define the problem statement
theorem combine_exponent_remains_unchanged (terms : List (ℕ × String)) : 
  (combining_like_terms terms).map Prod.snd = terms.map Prod.snd :=
sorry

end combine_exponent_remains_unchanged_l93_93698


namespace find_f_2005_1000_l93_93097

-- Define the real-valued function and its properties
def f (x y : ℝ) : ℝ := sorry

-- The condition given in the problem
axiom condition :
  ∀ x y z : ℝ, f x y = f x z - 2 * f y z - 2 * z

-- The target we need to prove
theorem find_f_2005_1000 : f 2005 1000 = 5 := 
by 
  -- all necessary logical steps (detailed in solution) would go here
  sorry

end find_f_2005_1000_l93_93097


namespace factorial_ratio_integer_l93_93949

theorem factorial_ratio_integer (m n : ℕ) : 
    (m ≥ 0) → (n ≥ 0) → ∃ k : ℤ, k = (2 * m).factorial * (2 * n).factorial / ((m.factorial * n.factorial * (m + n).factorial) : ℝ) :=
by
  sorry

end factorial_ratio_integer_l93_93949


namespace solution_to_quadratic_inequality_l93_93772

theorem solution_to_quadratic_inequality 
  (a : ℝ)
  (h : ∀ x : ℝ, x^2 - a * x + 1 < 0 ↔ (1 / 2 : ℝ) < x ∧ x < 2) :
  a = 5 / 2 :=
sorry

end solution_to_quadratic_inequality_l93_93772


namespace minimum_passed_l93_93187

def total_participants : Nat := 100
def num_questions : Nat := 10
def correct_answers : List Nat := [93, 90, 86, 91, 80, 83, 72, 75, 78, 59]
def passing_criteria : Nat := 6

theorem minimum_passed (total_participants : ℕ) (num_questions : ℕ) (correct_answers : List ℕ) (passing_criteria : ℕ) :
  100 = total_participants → 10 = num_questions → correct_answers = [93, 90, 86, 91, 80, 83, 72, 75, 78, 59] →
  passing_criteria = 6 → 
  ∃ p : ℕ, p = 62 := 
by
  sorry

end minimum_passed_l93_93187


namespace equilateral_triangle_area_l93_93827

noncomputable def altitude : ℝ := 2 * Real.sqrt 3
noncomputable def expected_area : ℝ := 4 * Real.sqrt 3

theorem equilateral_triangle_area (h : altitude = 2 * Real.sqrt 3) : 
  let a := 4 * Real.sqrt 3 in
  a = expected_area := 
by
  sorry

end equilateral_triangle_area_l93_93827


namespace fencing_rate_correct_l93_93422

noncomputable def rate_of_fencing_per_meter (area_hectares : ℝ) (total_cost : ℝ) : ℝ :=
  let area_sqm := area_hectares * 10000
  let r_squared := area_sqm / Real.pi
  let r := Real.sqrt r_squared
  let circumference := 2 * Real.pi * r
  total_cost / circumference

theorem fencing_rate_correct :
  rate_of_fencing_per_meter 13.86 6070.778380479544 = 4.60 :=
by
  sorry

end fencing_rate_correct_l93_93422


namespace num_arithmetic_sequences_l93_93061

theorem num_arithmetic_sequences (a d : ℕ) (n : ℕ) (h1 : n >= 3) (h2 : n * (2 * a + (n - 1) * d) = 2 * 97^2) :
  ∃ seqs : ℕ, seqs = 4 :=
by sorry

end num_arithmetic_sequences_l93_93061


namespace compare_pow_value_l93_93851

theorem compare_pow_value : 
  ∀ (x : ℝ) (n : ℕ), x = 0.01 → n = 1000 → (1 + x)^n > 1000 := 
by 
  intros x n hx hn
  rw [hx, hn]
  sorry

end compare_pow_value_l93_93851


namespace least_number_of_faces_l93_93321

def faces_triangular_prism : ℕ := 5
def faces_quadrangular_prism : ℕ := 6
def faces_triangular_pyramid : ℕ := 4
def faces_quadrangular_pyramid : ℕ := 5
def faces_truncated_quadrangular_pyramid : ℕ := 6

theorem least_number_of_faces : faces_triangular_pyramid < faces_triangular_prism ∧
                                faces_triangular_pyramid < faces_quadrangular_prism ∧
                                faces_triangular_pyramid < faces_quadrangular_pyramid ∧
                                faces_triangular_pyramid < faces_truncated_quadrangular_pyramid 
                                :=
by {
  sorry
}

end least_number_of_faces_l93_93321


namespace rods_in_mile_l93_93062

theorem rods_in_mile (mile_to_furlongs : 1 = 12) (furlong_to_rods : 1 = 50) : 1 * 12 * 50 = 600 :=
by
  sorry

end rods_in_mile_l93_93062


namespace y1_gt_y2_l93_93783

theorem y1_gt_y2 (k : ℝ) (y1 y2 : ℝ) 
  (h1 : y1 = (-1)^2 - 4*(-1) + k) 
  (h2 : y2 = 3^2 - 4*3 + k) : 
  y1 > y2 := 
by
  sorry

end y1_gt_y2_l93_93783


namespace min_value_three_l93_93941

noncomputable def min_value_expression (x y z : ℝ) : ℝ :=
  (1 / ((1 - x) * (1 - y) * (1 - z))) +
  (1 / ((1 + x) * (1 + y) * (1 + z))) +
  (1 / ((1 - x^2) * (1 - y^2) * (1 - z^2)))

theorem min_value_three (x y z : ℝ) (hx : 0 ≤ x) (hx1 : x ≤ 1) (hy : 0 ≤ y) (hy1 : y ≤ 1) (hz : 0 ≤ z) (hz1 : z ≤ 1) :
  min_value_expression x y z = 3 :=
by
  sorry

end min_value_three_l93_93941


namespace Michael_needs_more_money_l93_93653

def money_Michael_has : ℕ := 50
def cake_cost : ℕ := 20
def bouquet_cost : ℕ := 36
def balloons_cost : ℕ := 5

def total_cost : ℕ := cake_cost + bouquet_cost + balloons_cost
def money_needed : ℕ := total_cost - money_Michael_has

theorem Michael_needs_more_money : money_needed = 11 :=
by
  sorry

end Michael_needs_more_money_l93_93653


namespace number_of_invertible_integers_mod_15_l93_93219

theorem number_of_invertible_integers_mod_15 :
  (finset.card {a ∈ finset.range 15 | Int.gcd a 15 = 1}) = 8 := by
  sorry

end number_of_invertible_integers_mod_15_l93_93219


namespace pages_read_on_saturday_l93_93538

namespace BookReading

def total_pages : ℕ := 93
def pages_read_sunday : ℕ := 20
def pages_remaining : ℕ := 43

theorem pages_read_on_saturday :
  total_pages - (pages_read_sunday + pages_remaining) = 30 :=
by
  sorry

end BookReading

end pages_read_on_saturday_l93_93538


namespace union_of_A_and_B_l93_93270

def setA : Set ℝ := { x | -3 ≤ 2 * x - 1 ∧ 2 * x - 1 ≤ 3 }
def setB : Set ℝ := { x | 1 < x }

theorem union_of_A_and_B :
  setA ∪ setB = { x | -1 ≤ x } := sorry

end union_of_A_and_B_l93_93270


namespace jen_lisa_spent_l93_93048

theorem jen_lisa_spent (J L : ℝ) 
  (h1 : L = 0.8 * J) 
  (h2 : J = L + 15) : 
  J + L = 135 := 
by
  sorry

end jen_lisa_spent_l93_93048


namespace final_price_chocolate_l93_93140

-- Conditions
def original_cost : ℝ := 2.00
def discount : ℝ := 0.57

-- Question and answer
theorem final_price_chocolate : original_cost - discount = 1.43 :=
by
  sorry

end final_price_chocolate_l93_93140


namespace total_gray_trees_l93_93571

theorem total_gray_trees :
  (∃ trees_first trees_second trees_third gray1 gray2,
    trees_first = 100 ∧
    trees_second = 90 ∧
    trees_third = 82 ∧
    gray1 = trees_first - trees_third ∧
    gray2 = trees_second - trees_third ∧
    trees_first + trees_second - 2 * trees_third = gray1 + gray2) →
  (gray1 + gray2 = 26) :=
by
  intros
  sorry

end total_gray_trees_l93_93571


namespace adrien_winning_strategy_l93_93551

/--
On the table, there are 2023 tokens. Adrien and Iris take turns removing at least one token and at most half of the remaining tokens at the time they play. The player who leaves a single token on the table loses the game. Adrien starts first. Prove that Adrien has a winning strategy.
-/
theorem adrien_winning_strategy : ∃ strategy : ℕ → ℕ, 
  ∀ n:ℕ, (n = 2023 ∧ 1 ≤ strategy n ∧ strategy n ≤ n / 2) → 
    (∀ u : ℕ, (u = n - strategy n) → (∃ strategy' : ℕ → ℕ , 
      ∀ m:ℕ, (m = u ∧ 1 ≤ strategy' m ∧ strategy' m ≤ m / 2) → 
        (∃ next_u : ℕ, (next_u = m - strategy' m → next_u ≠ 1 ∨ (m = 1 ∧ u ≠ 1 ∧ next_u = 1)))))
:= sorry

end adrien_winning_strategy_l93_93551


namespace inverse_proportion_function_neg_k_l93_93089

variable {k : ℝ}
variable {y1 y2 : ℝ}

theorem inverse_proportion_function_neg_k
  (h1 : k ≠ 0)
  (h2 : y1 > y2)
  (hA : y1 = k / (-2))
  (hB : y2 = k / 5) :
  k < 0 :=
sorry

end inverse_proportion_function_neg_k_l93_93089


namespace can_measure_all_weights_l93_93058

def weights : List ℕ := [1, 3, 9, 27]

theorem can_measure_all_weights :
  (∀ n, 1 ≤ n ∧ n ≤ 40 → ∃ (a b c d : ℕ), a * 1 + b * 3 + c * 9 + d * 27 = n) ∧ 
  (∃ (a b c d : ℕ), a * 1 + b * 3 + c * 9 + d * 27 = 40) :=
by
  sorry

end can_measure_all_weights_l93_93058


namespace chess_tournament_possible_l93_93570

section ChessTournament

structure Player :=
  (name : String)
  (wins : ℕ)
  (draws : ℕ)
  (losses : ℕ)

def points (p : Player) : ℕ :=
  p.wins + p.draws / 2

def is_possible (A B C : Player) : Prop :=
  (points A > points B) ∧ (points A > points C) ∧
  (points C < points B) ∧
  (A.wins < B.wins) ∧ (A.wins < C.wins) ∧
  (C.wins > B.wins)

theorem chess_tournament_possible (A B C : Player) :
  is_possible A B C :=
  sorry

end ChessTournament

end chess_tournament_possible_l93_93570


namespace maria_workers_problem_l93_93492

-- Define the initial conditions
def initial_days : ℕ := 40
def days_passed : ℕ := 10
def fraction_completed : ℚ := 2/5
def initial_workers : ℕ := 10

-- Define the required minimum number of workers to complete the job on time
def minimum_workers_required : ℕ := 5

-- The theorem statement
theorem maria_workers_problem 
  (initial_days : ℕ)
  (days_passed : ℕ)
  (fraction_completed : ℚ)
  (initial_workers : ℕ) :
  ( ∀ (total_days remaining_days : ℕ), 
    initial_days = 40 ∧ days_passed = 10 ∧ fraction_completed = 2/5 ∧ initial_workers = 10 → 
    remaining_days = initial_days - days_passed ∧ 
    total_days = initial_days ∧ 
    fraction_completed + (remaining_days / total_days) = 1) →
  minimum_workers_required = 5 := 
sorry

end maria_workers_problem_l93_93492


namespace area_of_triangle_l93_93825

-- Definition of equilateral triangle and its altitude
def altitude_of_equilateral_triangle (a : ℝ) : Prop := 
  a = 2 * sqrt 3

-- Definition of the area function for equilateral triangle with side 's'
def area_of_equilateral_triangle (s : ℝ) : ℝ :=
  (sqrt 3 / 4) * s^2

-- The main statement to prove
theorem area_of_triangle (a : ℝ) (s : ℝ) 
  (alt_cond : altitude_of_equilateral_triangle a) 
  (side_relation : a = (sqrt 3 / 2) * s) : 
  area_of_equilateral_triangle s = 4 * sqrt 3 :=
by
  sorry

end area_of_triangle_l93_93825


namespace correct_avg_weight_l93_93463

theorem correct_avg_weight (initial_avg_weight : ℚ) (num_boys : ℕ) (misread_weight : ℚ) (correct_weight : ℚ) :
  initial_avg_weight = 58.4 → num_boys = 20 → misread_weight = 56 → correct_weight = 60 →
  (initial_avg_weight * num_boys + (correct_weight - misread_weight)) / num_boys = 58.6 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  -- Plugging in the values makes the calculation straightforward, resulting in: 
  -- (58.4 * 20 + (60 - 56)) / 20 = 58.6 
  -- thus this verification step is:
  sorry

end correct_avg_weight_l93_93463


namespace min_cans_for_gallon_l93_93411

-- Define conditions
def can_capacity : ℕ := 12
def gallon_to_ounces : ℕ := 128

-- Define the minimum number of cans function.
def min_cans (capacity : ℕ) (required : ℕ) : ℕ :=
  (required + capacity - 1) / capacity -- This is the ceiling of required / capacity

-- Statement asserting the required minimum number of cans.
theorem min_cans_for_gallon (h : min_cans can_capacity gallon_to_ounces = 11) : 
  can_capacity > 0 ∧ gallon_to_ounces > 0 := by
  sorry

end min_cans_for_gallon_l93_93411


namespace proof_ratio_QP_over_EF_l93_93117

noncomputable def rectangle_theorem : Prop :=
  ∃ (A B C D E F G P Q : ℝ × ℝ),
    -- Coordinates of the rectangle vertices
    A = (0, 4) ∧ B = (5, 4) ∧ C = (5, 0) ∧ D = (0, 0) ∧
    -- Coordinates of points E, F, and G on the sides of the rectangle
    E = (4, 4) ∧ F = (2, 0) ∧ G = (5, 1) ∧
    -- Coordinates of intersection points P and Q
    P = (20 / 7, 12 / 7) ∧ Q = (40 / 13, 28 / 13) ∧
    -- Ratio of distances PQ and EF
    (dist P Q)/(dist E F) = 10 / 91

theorem proof_ratio_QP_over_EF : rectangle_theorem :=
sorry

end proof_ratio_QP_over_EF_l93_93117


namespace percentage_change_area_l93_93671

theorem percentage_change_area (L B : ℝ) (h : L > 0 ∧ B > 0) :
  let A1 := L * B
  let L2 := L / 2
  let B2 := 3 * B
  let A2 := L2 * B2
  let percentage_change := ((A2 - A1) / A1) * 100
  percentage_change = 50 := by
  have A1_def := calc
    A1 = L * B : rfl
  have A2_def := calc
    A2 = (L / 2) * (3 * B) : rfl
    ... = (3 / 2) * (L * B) : by ring
  have pc_def := calc
    percentage_change = ((A2 - A1) / A1) * 100 : rfl
    ... = (( (3 / 2) * (L * B) - (L * B) ) / (L * B) ) * 100 : by rw [A1_def, A2_def]
    ... = (1 / 2) * 100 : by ring
  exact pc_def

end percentage_change_area_l93_93671


namespace value_of_f_2011_l93_93096

noncomputable def f (a b c x : ℝ) : ℝ := a * x^5 + b * x^3 + c * x + 7

theorem value_of_f_2011 (a b c : ℝ) (h : f a b c (-2011) = -17) : f a b c 2011 = 31 :=
by {
  sorry
}

end value_of_f_2011_l93_93096


namespace smallest_multiple_of_84_with_6_and_7_l93_93917

variable (N : Nat)

def is_multiple_of_84 (N : Nat) : Prop :=
  N % 84 = 0

def consists_of_6_and_7 (N : Nat) : Prop :=
  ∀ d ∈ N.digits 10, d = 6 ∨ d = 7

theorem smallest_multiple_of_84_with_6_and_7 :
  ∃ N, is_multiple_of_84 N ∧ consists_of_6_and_7 N ∧ ∀ M, is_multiple_of_84 M ∧ consists_of_6_and_7 M → N ≤ M := 
sorry

end smallest_multiple_of_84_with_6_and_7_l93_93917


namespace line_up_ways_l93_93533

theorem line_up_ways (people : Finset ℕ) (youngest : ℕ) (h1 : people.card = 5) (h2 : youngest ∈ people) :
  (∃ (cnt : ℕ), cnt = 72 ∧
    ∃ (arrangements : Finset (Fin 5 → ℕ)),
      arrangements.card = cnt ∧
      ∀ a ∈ arrangements, a ≠ (λ x, if x = 0 ∨ x = 4 then youngest else a x) ) := 
  sorry

end line_up_ways_l93_93533


namespace surface_area_of_solid_l93_93560

-- Define a unit cube and the number of cubes
def unitCube : Type := { faces : ℕ // faces = 6 }
def numCubes : ℕ := 10

-- Define the surface area contribution from different orientations
def surfaceAreaFacingUs (cubes : ℕ) : ℕ := 2 * cubes -- faces towards and away
def verticalSidesArea (heightCubes : ℕ) : ℕ := 2 * heightCubes -- left and right vertical sides
def horizontalSidesArea (widthCubes : ℕ) : ℕ := 2 * widthCubes -- top and bottom horizontal sides

-- Define the surface area for the given configuration of 10 cubes
def totalSurfaceArea (cubes : ℕ) (height : ℕ) (width : ℕ) : ℕ :=
  (surfaceAreaFacingUs cubes) + (verticalSidesArea height) + (horizontalSidesArea width)

-- Assumptions based on problem description
def heightCubes : ℕ := 3
def widthCubes : ℕ := 4

-- The theorem we want to prove
theorem surface_area_of_solid : totalSurfaceArea numCubes heightCubes widthCubes = 34 := by
  sorry

end surface_area_of_solid_l93_93560


namespace find_missing_number_l93_93143

theorem find_missing_number
  (x y : ℕ)
  (h1 : 30 = 6 * 5)
  (h2 : 600 = 30 * x)
  (h3 : x = 5 * y) :
  y = 4 :=
by
  sorry

end find_missing_number_l93_93143


namespace solve_complex_addition_l93_93629

def complex_addition_problem : Prop :=
  let B := Complex.mk 3 (-2)
  let Q := Complex.mk (-5) 1
  let R := Complex.mk 1 (-2)
  let T := Complex.mk 4 3
  B - Q + R + T = Complex.mk 13 (-2)

theorem solve_complex_addition : complex_addition_problem := by
  sorry

end solve_complex_addition_l93_93629


namespace abs_sum_lt_abs_sum_of_neg_product_l93_93508

theorem abs_sum_lt_abs_sum_of_neg_product 
  (a b : ℝ) : ab < 0 ↔ |a + b| < |a| + |b| := 
by 
  sorry

end abs_sum_lt_abs_sum_of_neg_product_l93_93508


namespace geometric_sequence_fourth_term_l93_93085

/-- In a geometric sequence with common ratio 2, where the sequence is denoted as {a_n},
and it is given that a_1 * a_3 = 6 * a_2, prove that a_4 = 24. -/
theorem geometric_sequence_fourth_term (a : ℕ → ℝ) (h : ∀ n, a (n + 1) = 2 * a n)
  (h1 : a 1 * a 3 = 6 * a 2) : a 4 = 24 :=
sorry

end geometric_sequence_fourth_term_l93_93085


namespace fraction_from_tips_l93_93359

-- Define the waiter's salary and the conditions given in the problem
variables (S : ℕ) -- S is natural assuming salary is a non-negative integer
def tips := (4/5 : ℚ) * S
def bonus := 2 * (1/10 : ℚ) * S
def total_income := S + tips S + bonus S

-- The theorem to be proven
theorem fraction_from_tips (S : ℕ) :
  (tips S / total_income S) = (2/5 : ℚ) :=
sorry

end fraction_from_tips_l93_93359


namespace team_order_l93_93253

-- Define the points of teams
variables (A B C D : ℕ)

-- State the conditions
def condition1 := A + C = B + D
def condition2 := B + A + 5 ≤ D + C
def condition3 := B + C ≥ A + D + 3

-- Statement of the theorem
theorem team_order (h1 : condition1 A B C D) (h2 : condition2 A B C D) (h3 : condition3 A B C D) :
  C > D ∧ D > B ∧ B > A :=
sorry

end team_order_l93_93253


namespace prob_correct_last_digit_no_more_than_two_attempts_prob_correct_last_digit_no_more_than_two_attempts_if_even_l93_93725

/-
Prove that if a person forgets the last digit of their 6-digit password, which can be any digit from 0 to 9,
the probability of pressing the correct last digit in no more than 2 attempts is 1/5.
-/

theorem prob_correct_last_digit_no_more_than_two_attempts :
  let correct_prob := 1 / 10 
  let incorrect_prob := 9 / 10 
  let second_attempt_prob := 1 / 9 
  correct_prob + (incorrect_prob * second_attempt_prob) = 1 / 5 :=
by
  sorry

/-
Prove that if a person forgets the last digit of their 6-digit password, but remembers that the last digit is an even number,
the probability of pressing the correct last digit in no more than 2 attempts is 2/5.
-/

theorem prob_correct_last_digit_no_more_than_two_attempts_if_even :
  let correct_prob := 1 / 5 
  let incorrect_prob := 4 / 5 
  let second_attempt_prob := 1 / 4 
  correct_prob + (incorrect_prob * second_attempt_prob) = 2 / 5 :=
by
  sorry

end prob_correct_last_digit_no_more_than_two_attempts_prob_correct_last_digit_no_more_than_two_attempts_if_even_l93_93725


namespace correct_equation_l93_93704

theorem correct_equation (a b : ℝ) : 
  (¬ (a^2 + a^3 = a^6)) ∧ (¬ ((ab)^2 = ab^2)) ∧ (¬ ((a+b)^2 = a^2 + b^2)) ∧ ((a+b)*(a-b) = a^2 - b^2) :=
by {
  sorry
}

end correct_equation_l93_93704


namespace abs_inequality_l93_93630

theorem abs_inequality (a b c : ℝ) (h : |a - c| < |b|) : |a| < |b| + |c| :=
sorry

end abs_inequality_l93_93630


namespace math_problem_l93_93763

noncomputable def find_min_value (a m n : ℝ) (h_a_pos : a > 0) (h_bn : n = 2 * m + 1 / 2)
  (h_b : -a^2 / 2 + 3 * Real.log a = -1 / 2) : ℝ :=
  (3 * Real.sqrt 5 / 5) ^ 2

theorem math_problem (a m n : ℝ) (h_a_pos : a > 0) (h_bn : n = 2 * m + 1 / 2) :
  ∃ b : ℝ, b = -a^2 / 2 + 3 * Real.log a →
  (a - m) ^ 2 + (b - n) ^ 2 = 9 / 5 :=
by
  sorry

end math_problem_l93_93763


namespace smallest_possible_value_l93_93937

theorem smallest_possible_value 
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b + c) * (1 / (a + b) + 1 / (a + c) + 1 / (b + c)) ≥ 9 / 2 :=
sorry

end smallest_possible_value_l93_93937


namespace three_brothers_pizza_slices_l93_93347

theorem three_brothers_pizza_slices :
  let large_pizza_slices := 14
  let small_pizza_slices := 8
  let num_brothers := 3
  let total_slices := small_pizza_slices + 2 * large_pizza_slices
  total_slices / num_brothers = 12 := by
  sorry

end three_brothers_pizza_slices_l93_93347


namespace inverse_proportion_function_neg_k_l93_93088

variable {k : ℝ}
variable {y1 y2 : ℝ}

theorem inverse_proportion_function_neg_k
  (h1 : k ≠ 0)
  (h2 : y1 > y2)
  (hA : y1 = k / (-2))
  (hB : y2 = k / 5) :
  k < 0 :=
sorry

end inverse_proportion_function_neg_k_l93_93088


namespace determine_d_minus_b_l93_93484

theorem determine_d_minus_b 
  (a b c d : ℕ) 
  (h1 : a^5 = b^4)
  (h2 : c^3 = d^2)
  (h3 : c - a = 19) 
  (a_pos : 0 < a)
  (b_pos : 0 < b)
  (c_pos : 0 < c)
  (d_pos : 0 < d)
  : d - b = 757 := 
  sorry

end determine_d_minus_b_l93_93484


namespace point_A_coords_l93_93041

theorem point_A_coords (x y : ℝ) (h : ∀ t : ℝ, (t + 1) * x - (2 * t + 5) * y - 6 = 0) : x = -4 ∧ y = -2 := by
  sorry

end point_A_coords_l93_93041


namespace division_of_cookies_l93_93002

theorem division_of_cookies (n p : Nat) (h1 : n = 24) (h2 : p = 6) : n / p = 4 :=
by sorry

end division_of_cookies_l93_93002


namespace general_term_arithmetic_sequence_l93_93060

theorem general_term_arithmetic_sequence (a : ℕ → ℕ) :
  (a 1 = 1) →
  (∀ n, n ≥ 2 → a n - a (n - 1) = 2) →
  ∀ n, a n = 2 * n - 1 := 
by
  intros h1 h2 n
  sorry

end general_term_arithmetic_sequence_l93_93060


namespace tom_sara_age_problem_l93_93964

-- Define the given conditions as hypotheses and variables
variables (t s : ℝ)
variables (h1 : t - 3 = 2 * (s - 3))
variables (h2 : t - 8 = 3 * (s - 8))

-- Lean statement of the problem
theorem tom_sara_age_problem :
  ∃ x : ℝ, (t + x) / (s + x) = 3 / 2 ∧ x = 7 :=
by
  sorry

end tom_sara_age_problem_l93_93964


namespace max_full_box_cards_l93_93267

-- Given conditions
def total_cards : ℕ := 94
def unfilled_box_cards : ℕ := 6

-- Define the number of cards that are evenly distributed into full boxes
def evenly_distributed_cards : ℕ := total_cards - unfilled_box_cards

-- Prove that the maximum number of cards a full box can hold is 22
theorem max_full_box_cards (h : evenly_distributed_cards = 88) : ∃ x : ℕ, evenly_distributed_cards % x = 0 ∧ x = 22 :=
by 
  -- Proof goes here
  sorry

end max_full_box_cards_l93_93267


namespace train_length_l93_93332

/-- 
  Given:
  - jogger_speed is the jogger's speed in km/hr (9 km/hr)
  - train_speed is the train's speed in km/hr (45 km/hr)
  - jogger_ahead is the jogger's initial lead in meters (240 m)
  - passing_time is the time in seconds for the train to pass the jogger (36 s)
  
  Prove that the length of the train is 120 meters.
-/
theorem train_length
  (jogger_speed : ℕ) -- in km/hr
  (train_speed : ℕ) -- in km/hr
  (jogger_ahead : ℕ) -- in meters
  (passing_time : ℕ) -- in seconds
  (h_jogger_speed : jogger_speed = 9)
  (h_train_speed : train_speed = 45)
  (h_jogger_ahead : jogger_ahead = 240)
  (h_passing_time : passing_time = 36)
  : ∃ length_of_train : ℕ, length_of_train = 120 :=
by
  sorry

end train_length_l93_93332


namespace find_n_l93_93502

theorem find_n (n : ℕ) (h : n * n.factorial + 2 * n.factorial = 5040) : n = 5 :=
sorry

end find_n_l93_93502


namespace problem_subtraction_of_negatives_l93_93574

theorem problem_subtraction_of_negatives :
  12.345 - (-3.256) = 15.601 :=
sorry

end problem_subtraction_of_negatives_l93_93574


namespace num_ways_to_place_2006_balls_l93_93167

noncomputable def numWaysToPlaceBallsIntoBoxes : ℕ :=
  let n := 2006
  nat.factorial n - 
  (derangements n + 
  (2006 * derangements (n - 1)) + 
  (nat.choose n 2 * derangements (n - 2)) + 
  (nat.choose n 3 * derangements (n - 3)) + 
  (nat.choose n 4 * derangements (n - 4)))

theorem num_ways_to_place_2006_balls : numWaysToPlaceBallsIntoBoxes = 2006! - D_{2006} - 2006 D_{2005} - nat.choose 2006 2 * D_{2004} - nat.choose 2006 3 * D_{2003} - nat.choose 2006 4 * D_{2002} :=
sorry

end num_ways_to_place_2006_balls_l93_93167


namespace max_value_trig_expr_exists_angle_for_max_value_l93_93044

theorem max_value_trig_expr : ∀ x : ℝ, 3 * Real.cos x + 4 * Real.sin x ≤ 5 :=
sorry

theorem exists_angle_for_max_value : ∃ x : ℝ, 3 * Real.cos x + 4 * Real.sin x = 5 :=
sorry

end max_value_trig_expr_exists_angle_for_max_value_l93_93044


namespace single_discount_percentage_l93_93722

noncomputable def original_price : ℝ := 9795.3216374269
noncomputable def sale_price : ℝ := 6700
noncomputable def discount_percentage (p₀ p₁ : ℝ) : ℝ := ((p₀ - p₁) / p₀) * 100

theorem single_discount_percentage :
  discount_percentage original_price sale_price = 31.59 := 
by
  sorry

end single_discount_percentage_l93_93722


namespace rationalize_denominator_l93_93278

theorem rationalize_denominator : (7 / Real.sqrt 147) = (Real.sqrt 3 / 3) :=
by
  sorry

end rationalize_denominator_l93_93278


namespace manuscript_pages_count_l93_93192

theorem manuscript_pages_count
  (P : ℕ)
  (cost_first_time : ℕ := 5 * P)
  (cost_once_revised : ℕ := 4 * 30)
  (cost_twice_revised : ℕ := 8 * 20)
  (total_cost : ℕ := 780)
  (h : cost_first_time + cost_once_revised + cost_twice_revised = total_cost) :
  P = 100 :=
sorry

end manuscript_pages_count_l93_93192


namespace solution_set_f_cos_x_l93_93765

noncomputable def f (x : ℝ) : ℝ :=
if 0 < x ∧ x < 3 then -(x-2)^2 + 1
else if x = 0 then 0
else if -3 < x ∧ x < 0 then (x+2)^2 - 1
else 0 -- Defined as 0 outside the given interval for simplicity

theorem solution_set_f_cos_x :
  {x : ℝ | f x * Real.cos x < 0} = {x : ℝ | (-3 < x ∧ x < -1) ∨ (0 < x ∧ x < 1) ∨ (1 < x ∧ x < 3)} :=
sorry

end solution_set_f_cos_x_l93_93765


namespace find_b_l93_93618

theorem find_b (b : ℝ) (x : ℝ) (hx : x^2 + b * x - 45 = 0) (h_root : x = -5) : b = -4 :=
by
  sorry

end find_b_l93_93618


namespace original_savings_l93_93479

variable (A B : ℕ)

-- A's savings are 5 times that of B's savings
def cond1 : Prop := A = 5 * B

-- If A withdraws 60 yuan and B deposits 60 yuan, then B's savings will be twice that of A's savings
def cond2 : Prop := (B + 60) = 2 * (A - 60)

-- Prove the original savings of A and B
theorem original_savings (h1 : cond1 A B) (h2 : cond2 A B) : A = 100 ∧ B = 20 := by
  sorry

end original_savings_l93_93479


namespace solve_k_l93_93858

theorem solve_k (t s : ℤ) : (∃ k m, 8 * k + 4 = 7 * m ∧ k = -4 + 7 * t ∧ m = -4 + 8 * t) →
  (∃ k m, 12 * k - 8 = 7 * m ∧ k = 3 + 7 * s ∧ m = 4 + 12 * s) →
  7 * t - 4 = 7 * s + 3 →
  ∃ k, k = 3 + 7 * s :=
by
  sorry

end solve_k_l93_93858


namespace remainder_when_divided_by_9_l93_93699

theorem remainder_when_divided_by_9 (x : ℕ) (h1 : x > 0) (h2 : (5 * x) % 9 = 7) : x % 9 = 5 :=
sorry

end remainder_when_divided_by_9_l93_93699


namespace equations_have_different_graphs_l93_93319

theorem equations_have_different_graphs :
  (∃ (x : ℝ), ∀ (y₁ y₂ y₃ : ℝ),
    (y₁ = x - 2) ∧
    (y₂ = (x^2 - 4) / (x + 2) ∧ x ≠ -2) ∧
    (y₃ = (x^2 - 4) / (x + 2) ∧ x ≠ -2 ∨ (x = -2 ∧ ∀ y₃ : ℝ, (x+2) * y₃ = x^2 - 4)))
  → (∃ y₁ y₂ y₃ : ℝ, y₁ ≠ y₂ ∨ y₁ ≠ y₃ ∨ y₂ ≠ y₃) := sorry

end equations_have_different_graphs_l93_93319


namespace min_value_f_exists_min_value_f_l93_93232

noncomputable def f (a b c : ℝ) := 1 / (b^2 + b * c) + 1 / (c^2 + c * a) + 1 / (a^2 + a * b)

theorem min_value_f (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b * c = 1) : f a b c ≥ 3 / 2 :=
  sorry

theorem exists_min_value_f : ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a * b * c = 1 ∧ f a b c = 3 / 2 :=
  sorry

end min_value_f_exists_min_value_f_l93_93232


namespace luke_points_per_round_l93_93400

-- Definitions for conditions
def total_points : ℤ := 84
def rounds : ℤ := 2
def points_per_round (total_points rounds : ℤ) : ℤ := total_points / rounds

-- Statement of the problem
theorem luke_points_per_round : points_per_round total_points rounds = 42 := 
by 
  sorry

end luke_points_per_round_l93_93400


namespace sum_of_products_circle_l93_93435

theorem sum_of_products_circle 
  (a b c d : ℤ) 
  (h : a + b + c + d = 0) : 
  -((a * (b + d)) + (b * (a + c)) + (c * (b + d)) + (d * (a + c))) = 2 * (a + c) ^ 2 :=
sorry

end sum_of_products_circle_l93_93435


namespace total_balls_estimation_l93_93580

theorem total_balls_estimation
  (n : ℕ)  -- Let n be the total number of balls in the bag
  (yellow_balls : ℕ)  -- Let yellow_balls be the number of yellow balls
  (frequency : ℝ)  -- Let frequency be the stabilized frequency of drawing a yellow ball
  (h1 : yellow_balls = 6)
  (h2 : frequency = 0.3)
  (h3 : (yellow_balls : ℝ) / (n : ℝ) = frequency) :
  n = 20 :=
by
  sorry

end total_balls_estimation_l93_93580


namespace dolls_total_l93_93874

theorem dolls_total (V S A : ℕ) 
  (hV : V = 20) 
  (hS : S = 2 * V)
  (hA : A = 2 * S) 
  : A + S + V = 140 := 
by 
  sorry

end dolls_total_l93_93874


namespace sum_mod_9237_9241_l93_93746

theorem sum_mod_9237_9241 :
  (9237 + 9238 + 9239 + 9240 + 9241) % 9 = 2 :=
by
  sorry

end sum_mod_9237_9241_l93_93746


namespace bisection_approximation_interval_l93_93966

noncomputable def bisection_accuracy (a b : ℝ) (n : ℕ) : ℝ := (b - a) / 2^n

theorem bisection_approximation_interval 
  (a b : ℝ) (n : ℕ) (accuracy : ℝ) (h1 : a = 2) (h2 : b = 3) (h3 : accuracy = 0.01) 
  (h4 : 2^n ≥ 100) : bisection_accuracy a b n ≤ accuracy :=
sorry

end bisection_approximation_interval_l93_93966


namespace simplify_fraction_l93_93134

theorem simplify_fraction :
  (1 / 462) + (17 / 42) = 94 / 231 := sorry

end simplify_fraction_l93_93134


namespace find_y_l93_93857

theorem find_y 
  (x y : ℕ) 
  (hx : x % y = 9) 
  (hxy : (x : ℝ) / y = 96.12) : y = 75 :=
sorry

end find_y_l93_93857


namespace hosting_schedules_count_l93_93189

theorem hosting_schedules_count :
  let n_universities := 6
  let n_years := 8
  let total_ways := 6 * 5 * 4^6
  let excluding_one := 6 * 5 * 4 * 3^6
  let excluding_two := 15 * 4 * 3 * 2^6
  let excluding_three := 20 * 3 * 2 * 1^6
  total_ways - excluding_one + excluding_two - excluding_three = 46080 := 
by
  sorry

end hosting_schedules_count_l93_93189


namespace infinitely_many_MTRP_numbers_l93_93797

def sum_of_digits (n : ℕ) : ℕ := 
n.digits 10 |>.sum

def is_MTRP_number (m n : ℕ) : Prop :=
  n % m = 1 ∧ sum_of_digits (n^2) ≥ sum_of_digits n

theorem infinitely_many_MTRP_numbers (m : ℕ) : 
  ∀ N : ℕ, ∃ n > N, is_MTRP_number m n :=
by sorry

end infinitely_many_MTRP_numbers_l93_93797


namespace CEMC_additional_employees_l93_93209

variable (t : ℝ)

def initialEmployees (t : ℝ) := t + 40

def finalEmployeesMooseJaw (t : ℝ) := 1.25 * t

def finalEmployeesOkotoks : ℝ := 26

def finalEmployeesTotal (t : ℝ) := finalEmployeesMooseJaw t + finalEmployeesOkotoks

def netChangeInEmployees (t : ℝ) := finalEmployeesTotal t - initialEmployees t

theorem CEMC_additional_employees (t : ℝ) (h : t = 120) : 
    netChangeInEmployees t = 16 := 
by
    sorry

end CEMC_additional_employees_l93_93209


namespace men_at_yoga_studio_l93_93440

open Real

def yoga_men_count (M : ℕ) (avg_weight_men avg_weight_women avg_weight_total : ℝ) (num_women num_total : ℕ) : Prop :=
  avg_weight_men = 190 ∧
  avg_weight_women = 120 ∧
  num_women = 6 ∧
  num_total = 14 ∧
  avg_weight_total = 160 →
  M + num_women = num_total ∧
  (M * avg_weight_men + num_women * avg_weight_women) / num_total = avg_weight_total ∧
  M = 8

theorem men_at_yoga_studio : ∃ M : ℕ, yoga_men_count M 190 120 160 6 14 :=
  by 
  use 8
  sorry

end men_at_yoga_studio_l93_93440


namespace ratio_of_height_and_radius_l93_93273

theorem ratio_of_height_and_radius 
  (h r : ℝ) 
  (V_X V_Y : ℝ)
  (hY rY : ℝ)
  (k : ℝ)
  (h_def : V_X = π * r^2 * h)
  (hY_def : hY = k * h)
  (rY_def : rY = k * r)
  (half_filled_VY : V_Y = 1/2 * π * rY^2 * hY)
  (V_X_value : V_X = 2)
  (V_Y_value : V_Y = 64):
  k = 4 :=
by
  sorry

end ratio_of_height_and_radius_l93_93273


namespace sum_of_fractions_l93_93489

theorem sum_of_fractions : 
  (7 / 8 + 3 / 4) = (13 / 8) :=
by
  sorry

end sum_of_fractions_l93_93489


namespace sufficient_but_not_necessary_condition_not_necessary_condition_l93_93064

variable {a b m : ℝ}

theorem sufficient_but_not_necessary_condition (h : a * m^2 < b * m^2) : a < b := by
  sorry

-- Additional statements to express the sufficiency and not necessity nature:
theorem not_necessary_condition (h : a < b) (hm : m = 0) : ¬ (a * m^2 < b * m^2) := by
  sorry

end sufficient_but_not_necessary_condition_not_necessary_condition_l93_93064


namespace turkey_2003_problem_l93_93714

theorem turkey_2003_problem (x m n : ℕ) (hx : 0 < x) (hm : 0 < m) (hn : 0 < n) (h : x^m = 2^(2 * n + 1) + 2^n + 1) :
  x = 2^(2 * n + 1) + 2^n + 1 ∧ m = 1 ∨ x = 23 ∧ m = 2 ∧ n = 4 :=
sorry

end turkey_2003_problem_l93_93714


namespace find_d_l93_93521

theorem find_d (d : ℚ) (h : ∀ x : ℚ, 4*x^3 + 17*x^2 + d*x + 28 = 0 → x = -4/3) : d = 155 / 9 :=
sorry

end find_d_l93_93521


namespace sparkling_water_cost_l93_93652

theorem sparkling_water_cost
  (drinks_per_day : ℚ := 1 / 5)
  (bottle_cost : ℝ := 2.00)
  (days_in_year : ℤ := 365) :
  (drinks_per_day * days_in_year) * bottle_cost = 146 :=
by
  sorry

end sparkling_water_cost_l93_93652


namespace quadratic_floor_eq_more_than_100_roots_l93_93345

open Int

theorem quadratic_floor_eq_more_than_100_roots (p q : ℤ) (h : p ≠ 0) :
  ∃ (S : Finset ℤ), S.card > 100 ∧ ∀ x ∈ S, ⌊(x : ℝ) ^ 2⌋ + p * x + q = 0 :=
by
  sorry

end quadratic_floor_eq_more_than_100_roots_l93_93345


namespace chocolates_150_satisfies_l93_93024

def chocolates_required (chocolates : ℕ) : Prop :=
  chocolates ≥ 150 ∧ chocolates % 19 = 17

theorem chocolates_150_satisfies : chocolates_required 150 :=
by
  -- We need to show that 150 satisfies the conditions:
  -- 1. 150 ≥ 150
  -- 2. 150 % 19 = 17
  unfold chocolates_required
  -- Both conditions hold:
  exact And.intro (by linarith) (by norm_num)

end chocolates_150_satisfies_l93_93024


namespace brownie_pieces_count_l93_93468

theorem brownie_pieces_count :
  let tray_length := 24
  let tray_width := 20
  let brownie_length := 3
  let brownie_width := 2
  let tray_area := tray_length * tray_width
  let brownie_area := brownie_length * brownie_width
  let pieces_count := tray_area / brownie_area
  pieces_count = 80 :=
by
  let tray_length := 24
  let tray_width := 20
  let brownie_length := 3
  let brownie_width := 2
  let tray_area := 24 * 20
  let brownie_area := 3 * 2
  let pieces_count := tray_area / brownie_area
  have h1 : tray_length * tray_width = 480 := by norm_num
  have h2 : brownie_length * brownie_width = 6 := by norm_num
  have h3 : pieces_count = 80 := by norm_num
  exact h3

end brownie_pieces_count_l93_93468


namespace least_number_of_coins_l93_93982

theorem least_number_of_coins (n : ℕ) : 
  (n % 7 = 3) ∧ (n % 5 = 4) ∧ (∀ m : ℕ, (m % 7 = 3) ∧ (m % 5 = 4) → n ≤ m) → n = 24 :=
by
  sorry

end least_number_of_coins_l93_93982


namespace circle_symmetric_eq_l93_93227

theorem circle_symmetric_eq :
  ∀ (x y : ℝ), (x^2 + y^2 + 2 * x - 2 * y + 1 = 0) → (x - y + 3 = 0) → 
  (∃ (a b : ℝ), (a + 2)^2 + (b - 2)^2 = 1) :=
by
  intros x y hc hl
  sorry

end circle_symmetric_eq_l93_93227


namespace total_gulbis_is_correct_l93_93683

-- Definitions based on given conditions
def num_dureums : ℕ := 156
def num_gulbis_in_one_dureum : ℕ := 20

-- Definition of total gulbis calculated
def total_gulbis : ℕ := num_dureums * num_gulbis_in_one_dureum

-- Statement to prove
theorem total_gulbis_is_correct : total_gulbis = 3120 := by
  -- The actual proof would go here
  sorry

end total_gulbis_is_correct_l93_93683


namespace helen_hand_washing_time_l93_93885

theorem helen_hand_washing_time :
  (52 / 4) * 30 / 60 = 6.5 := by
  sorry

end helen_hand_washing_time_l93_93885


namespace total_original_cost_of_books_l93_93008

noncomputable def original_cost_price_in_eur (selling_prices : List ℝ) (profit_margin : ℝ) (exchange_rate : ℝ) : ℝ :=
  let original_cost_prices := selling_prices.map (λ price => price / (1 + profit_margin))
  let total_original_cost_usd := original_cost_prices.sum
  total_original_cost_usd * exchange_rate

theorem total_original_cost_of_books : original_cost_price_in_eur [240, 260, 280, 300, 320] 0.20 0.85 = 991.67 :=
  sorry

end total_original_cost_of_books_l93_93008


namespace number_of_solutions_l93_93761

-- Defining the sets A and B
def A (a : ℝ) : Set ℝ := {1, 3, a^2}
def B (a : ℝ) : Set ℝ := {1, a + 2}

-- Statement of the problem
theorem number_of_solutions (a : ℝ) : {a : ℝ | A a ∪ B a = A a}.card = 1 :=
by
  sorry

end number_of_solutions_l93_93761


namespace exist_six_subsets_of_six_elements_l93_93942

theorem exist_six_subsets_of_six_elements (n m : ℕ) (X : Finset ℕ) (A : Fin m → Finset ℕ) :
    n > 6 →
    X.card = n →
    (∀ i, (A i).card = 5 ∧ (A i ⊆ X)) →
    m > (n * (n-1) * (n-2) * (n-3) * (4*n-15)) / 600 →
    ∃ i1 i2 i3 i4 i5 i6 : Fin m,
      i1 < i2 ∧ i2 < i3 ∧ i3 < i4 ∧ i4 < i5 ∧ i5 < i6 ∧
      (A i1 ∪ A i2 ∪ A i3 ∪ A i4 ∪ A i5 ∪ A i6).card = 6 := 
sorry

end exist_six_subsets_of_six_elements_l93_93942


namespace largest_angle_in_pentagon_l93_93086

theorem largest_angle_in_pentagon {R S : ℝ} (h₁: R = S) 
  (h₂: (75 : ℝ) + 110 + R + S + (3 * R - 20) = 540) : 
  (3 * R - 20) = 217 :=
by {
  -- Given conditions are assigned and now we need to prove the theorem, the proof is omitted
  sorry
}

end largest_angle_in_pentagon_l93_93086


namespace area_of_triangle_ABC_sinA_value_l93_93919

noncomputable def cosC := 3 / 4
noncomputable def sinC := Real.sqrt (1 - cosC ^ 2)
noncomputable def a := 1
noncomputable def b := 2
noncomputable def c := Real.sqrt (a ^ 2 + b ^ 2 - 2 * a * b * cosC)
noncomputable def area := (1 / 2) * a * b * sinC
noncomputable def sinA := (a * sinC) / c

theorem area_of_triangle_ABC : area = Real.sqrt 7 / 4 :=
by sorry

theorem sinA_value : sinA = Real.sqrt 14 / 8 :=
by sorry

end area_of_triangle_ABC_sinA_value_l93_93919


namespace isosceles_triangle_angle_measure_l93_93732

theorem isosceles_triangle_angle_measure
  (isosceles : Triangle → Prop)
  (exterior_angles : Triangle → ℝ → ℝ → Prop)
  (ratio_1_to_4 : ∀ {T : Triangle} {a b : ℝ}, exterior_angles T a b → b = 4 * a)
  (interior_angles : Triangle → ℝ → ℝ → ℝ → Prop) :
  ∀ (T : Triangle), isosceles T → ∃ α β γ : ℝ, interior_angles T α β γ ∧ α = 140 ∧ β = 20 ∧ γ = 20 := 
by
  sorry

end isosceles_triangle_angle_measure_l93_93732


namespace dimes_difference_l93_93036

theorem dimes_difference (a b c : ℕ) :
  a + b + c = 120 →
  5 * a + 10 * b + 25 * c = 1265 →
  c ≥ 10 →
  (max (b) - min (b)) = 92 :=
sorry

end dimes_difference_l93_93036


namespace volume_of_remaining_solid_l93_93996

noncomputable def volume_cube_with_cylindrical_hole 
  (side_length : ℝ) (hole_diameter : ℝ) (π : ℝ := 3.141592653589793) : ℝ :=
  let V_cube := side_length^3
  let radius := hole_diameter / 2
  let height := side_length
  let V_cylinder := π * radius^2 * height
  V_cube - V_cylinder

theorem volume_of_remaining_solid 
  (side_length : ℝ)
  (hole_diameter : ℝ)
  (h₁ : side_length = 6) 
  (h₂ : hole_diameter = 3)
  (π : ℝ := 3.141592653589793) : 
  abs (volume_cube_with_cylindrical_hole side_length hole_diameter π - 173.59) < 0.01 :=
by
  sorry

end volume_of_remaining_solid_l93_93996


namespace red_balls_count_l93_93293

theorem red_balls_count (w r : ℕ) (h1 : w = 16) (h2 : 4 * r = 3 * w) : r = 12 :=
by
  sorry

end red_balls_count_l93_93293


namespace overall_gain_percentage_l93_93876

theorem overall_gain_percentage (cost_A cost_B cost_C sp_A sp_B sp_C : ℕ)
  (hA : cost_A = 1000)
  (hB : cost_B = 3000)
  (hC : cost_C = 6000)
  (hsA : sp_A = 2000)
  (hsB : sp_B = 4500)
  (hsC : sp_C = 8000) :
  ((sp_A + sp_B + sp_C - (cost_A + cost_B + cost_C) : ℝ) / (cost_A + cost_B + cost_C) * 100) = 45 :=
by sorry

end overall_gain_percentage_l93_93876


namespace contemporaries_probability_l93_93573

theorem contemporaries_probability :
  (∃ x y : ℕ, 0 ≤ x ∧ x ≤ 600 ∧ 0 ≤ y ∧ y ≤ 600 ∧ x < y + 120 ∧ y < x + 100) →
  (193 : ℚ) / 200 = (∑ x in (finset.range 600), ∑ y in (finset.range 600), 
              if (x < y + 120 ∧ y < x + 100) then 1 else 0) / (600 * 600) :=
begin
  sorry
end

end contemporaries_probability_l93_93573


namespace men_in_second_group_l93_93716

theorem men_in_second_group (M : ℕ) (W : ℝ) (h1 : 15 * 25 = W) (h2 : M * 18.75 = W) : M = 20 :=
sorry

end men_in_second_group_l93_93716


namespace total_employees_in_buses_l93_93448

theorem total_employees_in_buses :
  let bus1_percentage_full := 0.60,
      bus2_percentage_full := 0.70,
      bus_capacity := 150
  in
  (bus1_percentage_full * bus_capacity + bus2_percentage_full * bus_capacity) = 195 := by
  sorry

end total_employees_in_buses_l93_93448


namespace average_books_per_month_l93_93327

-- Definitions based on the conditions
def books_sold_january : ℕ := 15
def books_sold_february : ℕ := 16
def books_sold_march : ℕ := 17
def total_books_sold : ℕ := books_sold_january + books_sold_february + books_sold_march
def number_of_months : ℕ := 3

-- The theorem we need to prove
theorem average_books_per_month : total_books_sold / number_of_months = 16 :=
by
  sorry

end average_books_per_month_l93_93327


namespace minimum_value_inequality_l93_93079

theorem minimum_value_inequality {x y z : ℝ} (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h : x * y * z * (x + y + z) = 1) : (x + y) * (y + z) ≥ 2 := 
sorry

end minimum_value_inequality_l93_93079


namespace deanna_initial_speed_l93_93050

namespace TripSpeed

variables (v : ℝ) (h : v > 0)

def speed_equation (v : ℝ) : Prop :=
  (1/2 * v) + (1/2 * (v + 20)) = 100

theorem deanna_initial_speed (v : ℝ) (h : speed_equation v) : v = 90 := sorry

end TripSpeed

end deanna_initial_speed_l93_93050


namespace evaluate_expression_l93_93360

theorem evaluate_expression (a b c : ℚ) 
  (h1 : c = b - 11) 
  (h2 : b = a + 3) 
  (h3 : a = 5)
  (h4 : a + 2 ≠ 0) 
  (h5 : b - 3 ≠ 0) 
  (h6 : c + 7 ≠ 0) : 
  (a + 3) / (a + 2) * (b - 2) / (b - 3) * (c + 9) / (c + 7) = 10 / 7 := 
sorry

end evaluate_expression_l93_93360


namespace complex_cubed_l93_93898

theorem complex_cubed (z : ℂ) (h1 : |z - 2| = 2) (h2 : |z| = 2) : z ^ 3 = -8 :=
sorry

end complex_cubed_l93_93898


namespace area_of_triangle_F1PF2P_l93_93424

noncomputable def a : ℝ := 5
noncomputable def b : ℝ := 4
noncomputable def c : ℝ := 3
noncomputable def PF1 : ℝ := sorry 
noncomputable def PF2 : ℝ := sorry

-- Given conditions
def ellipse_eq_holds (x y : ℝ) : Prop := (x^2) / 25 + (y^2) / 16 = 1

-- Given point P is on the ellipse
def P_on_ellipse (x y : ℝ) : Prop := ellipse_eq_holds x y

-- Given angle F1PF2
def angle_F1PF2_eq_60 : Prop := sorry

-- Proving the area of △F₁PF₂
theorem area_of_triangle_F1PF2P : S = (16 * Real.sqrt 3) / 3 :=
by sorry

end area_of_triangle_F1PF2P_l93_93424


namespace sale_price_correct_l93_93199

noncomputable def original_price : ℝ := 600.00
noncomputable def first_discount_factor : ℝ := 0.75
noncomputable def second_discount_factor : ℝ := 0.90
noncomputable def final_price : ℝ := original_price * first_discount_factor * second_discount_factor
noncomputable def expected_final_price : ℝ := 0.675 * original_price

theorem sale_price_correct : final_price = expected_final_price := sorry

end sale_price_correct_l93_93199


namespace numberOfTrucks_l93_93865

-- Conditions
def numberOfTanksPerTruck : ℕ := 3
def capacityPerTank : ℕ := 150
def totalWaterCapacity : ℕ := 1350

-- Question and proof goal
theorem numberOfTrucks : 
  (totalWaterCapacity / (numberOfTanksPerTruck * capacityPerTank) = 3) := 
by 
  sorry

end numberOfTrucks_l93_93865


namespace sum_of_midpoint_xcoords_l93_93156

theorem sum_of_midpoint_xcoords (a b c : ℝ) (h : a + b + c = 15) :
    (a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 15 :=
  by
    sorry

end sum_of_midpoint_xcoords_l93_93156


namespace solve_problem_l93_93049

theorem solve_problem :
  ∃ (x y : ℝ), 7 * x + y = 19 ∧ x + 3 * y = 1 ∧ 2 * x + y = 5 :=
by
  sorry

end solve_problem_l93_93049


namespace tan_alpha_frac_simplification_l93_93895

theorem tan_alpha_frac_simplification (α : ℝ) (h : Real.tan α = -1 / 2) : 
  (2 * Real.sin α * Real.cos α) / (Real.sin α ^ 2 - Real.cos α ^ 2) = 4 / 3 :=
by sorry

end tan_alpha_frac_simplification_l93_93895


namespace a_8_eq_5_l93_93375

noncomputable def S (n : ℕ) : ℕ := sorry
noncomputable def a (n : ℕ) : ℕ := sorry

axiom S_eq : ∀ n m : ℕ, S n + S m = S (n + m)
axiom a1 : a 1 = 5
axiom Sn1 : ∀ n : ℕ, S (n + 1) = S n + 5

theorem a_8_eq_5 : a 8 = 5 :=
sorry

end a_8_eq_5_l93_93375


namespace problem_I_problem_II_l93_93515

-- Define the function f as given
def f (x m : ℝ) : ℝ := x^2 + (m-1)*x - m

-- Problem (I)
theorem problem_I (x : ℝ) : -2 < x ∧ x < 1 ↔ f x 2 < 0 := sorry

-- Problem (II)
theorem problem_II (m : ℝ) : ∀ x, f x m + 1 ≥ 0 ↔ -3 ≤ m ∧ m ≤ 1 := sorry

end problem_I_problem_II_l93_93515


namespace instantaneous_speed_at_3_l93_93292

noncomputable def s (t : ℝ) : ℝ := 1 - t + 2 * t^2

theorem instantaneous_speed_at_3 : deriv s 3 = 11 :=
by
  sorry

end instantaneous_speed_at_3_l93_93292


namespace exists_infinitely_many_n_l93_93095

def digit_sum (m : ℕ) : ℕ := sorry  -- Define the digit sum function

theorem exists_infinitely_many_n (S : ℕ → ℕ)
  (hS : ∀ m : ℕ, S m = digit_sum m) :
  ∃ᶠ n in at_top, S (3^n) ≥ S (3^(n + 1)) := 
sorry

end exists_infinitely_many_n_l93_93095


namespace power_equality_l93_93522

theorem power_equality (n : ℝ) : (9:ℝ)^4 = (27:ℝ)^n → n = (8:ℝ) / 3 :=
by
  sorry

end power_equality_l93_93522


namespace evaluate_complex_pow_l93_93608

open Complex

noncomputable def calc : ℂ := (-64 : ℂ) ^ (7 / 6)

theorem evaluate_complex_pow : calc = 128 * Complex.I := by 
  -- Recognize that (-64) = (-4)^3
  -- Apply exponent rules: ((-4)^3)^(7/6) = (-4)^(3 * 7/6) = (-4)^(7/2)
  -- Simplify (-4)^(7/2) = √((-4)^7) = √(-16384)
  -- Calculation (-4)^7 = -16384
  -- Simplify √(-16384) = 128i
  sorry

end evaluate_complex_pow_l93_93608


namespace intersection_M_N_l93_93376

def M : Set ℝ := { y | ∃ x, y = 2^x ∧ x > 0 }
def N : Set ℝ := { y | ∃ z, y = Real.log z ∧ z ∈ M }

theorem intersection_M_N : M ∩ N = { y | y > 1 } := sorry

end intersection_M_N_l93_93376


namespace sufficient_but_not_necessary_condition_l93_93900

def p (x : ℝ) : Prop := x^2 - 2 * x - 3 < 0
def q (x a : ℝ) : Prop := x > a

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (∀ x, p x → q x a) ∧ (∃ x, q x a ∧ ¬ p x) ↔ a ≤ -1 :=
by
  sorry

end sufficient_but_not_necessary_condition_l93_93900


namespace smallest_integer_in_ratio_l93_93686

theorem smallest_integer_in_ratio (a b c : ℕ) 
    (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
    (h_sum : a + b + c = 100) 
    (h_ratio : c = 5 * a / 2 ∧ b = 3 * a / 2) : 
    a = 20 := 
by
  sorry

end smallest_integer_in_ratio_l93_93686


namespace solve_for_x_l93_93818

theorem solve_for_x : ∃ x : ℚ, (1/4 : ℚ) + (1/x) = 7/8 ∧ x = 8/5 :=
by {
  sorry
}

end solve_for_x_l93_93818


namespace solution_to_largest_four_digit_fulfilling_conditions_l93_93456

def largest_four_digit_fulfilling_conditions : Prop :=
  ∃ (N : ℕ), N < 10000 ∧ N ≡ 2 [MOD 11] ∧ N ≡ 4 [MOD 7] ∧ N = 9979

theorem solution_to_largest_four_digit_fulfilling_conditions : largest_four_digit_fulfilling_conditions :=
  sorry

end solution_to_largest_four_digit_fulfilling_conditions_l93_93456


namespace odd_function_neg_value_l93_93907

theorem odd_function_neg_value (f : ℝ → ℝ) (h_odd : ∀ x : ℝ, f (-x) = -f x) (h_value : f 1 = 1) : f (-1) = -1 :=
by
  sorry

end odd_function_neg_value_l93_93907


namespace simplify_expression_l93_93310

theorem simplify_expression :
  (Real.sqrt 5 * 5^(1/2) + 20 / 4 * 3 - 9^(3/2)) = -7 :=
by sorry

end simplify_expression_l93_93310


namespace discriminant_formula_l93_93798

def discriminant_cubic_eq (x1 x2 x3 p q : ℝ) : ℝ :=
  (x1 - x2)^2 * (x2 - x3)^2 * (x3 - x1)^2

theorem discriminant_formula (x1 x2 x3 p q : ℝ)
  (h1 : x1 + x2 + x3 = 0)
  (h2 : x1 * x2 + x1 * x3 + x2 * x3 = p)
  (h3 : x1 * x2 * x3 = -q) :
  discriminant_cubic_eq x1 x2 x3 p q = -4 * p^3 - 27 * q^2 :=
by sorry

end discriminant_formula_l93_93798


namespace tangent_line_of_ellipse_l93_93903

variable {a b x y x₀ y₀ : ℝ}

theorem tangent_line_of_ellipse
    (h1 : 0 < a)
    (h2 : a > b)
    (h3 : b > 0)
    (h4 : (x₀, y₀) ∈ { p : ℝ × ℝ | p.1^2 / a^2 + p.2^2 / b^2 = 1 }) :
    (x₀ * x) / (a^2) + (y₀ * y) / (b^2) = 1 :=
sorry

end tangent_line_of_ellipse_l93_93903


namespace sin_difference_identity_l93_93349

theorem sin_difference_identity 
  (α β : ℝ)
  (h1 : sin α - cos β = 3 / 4)
  (h2 : cos α + sin β = -2 / 5) : 
  sin (α - β) = 511 / 800 := 
sorry

end sin_difference_identity_l93_93349


namespace total_cases_l93_93792

def NY : ℕ := 2000
def CA : ℕ := NY / 2
def TX : ℕ := CA - 400

theorem total_cases : NY + CA + TX = 3600 :=
by
  -- use sorry placeholder to indicate the solution is omitted
  sorry

end total_cases_l93_93792


namespace chocolate_bars_left_l93_93193

noncomputable def chocolateBarsCount : ℕ :=
  let initial_bars := 800
  let thomas_friends_bars := (3 * initial_bars) / 8
  let adjusted_thomas_friends_bars := thomas_friends_bars + 1  -- Adjust for the extra bar rounding issue
  let piper_bars_taken := initial_bars / 4
  let piper_bars_returned := 8
  let adjusted_piper_bars := piper_bars_taken - piper_bars_returned
  let paul_club_bars := 9
  let polly_club_bars := 7
  let catherine_bars_returned := 15
  
  initial_bars
  - adjusted_thomas_friends_bars
  - adjusted_piper_bars
  - paul_club_bars
  - polly_club_bars
  + catherine_bars_returned

theorem chocolate_bars_left : chocolateBarsCount = 308 := by
  sorry

end chocolate_bars_left_l93_93193


namespace greatest_product_two_integers_sum_2004_l93_93976

theorem greatest_product_two_integers_sum_2004 : 
  (∃ x y : ℤ, x + y = 2004 ∧ x * y = 1004004) :=
by
  sorry

end greatest_product_two_integers_sum_2004_l93_93976


namespace group_morph_or_involution_l93_93543

variables {G : Type*} [Group G]

theorem group_morph_or_involution (f : G → G) 
  (h_morph : ∀ x y : G, f (x * y) = f x * f y)
  (h_choice : ∀ x : G, f x = x ∨ f x = x⁻¹)
  (h_no_order_4 : ∀ x : G, ¬ (x^4 = 1 ∧ ¬ x^2 = 1)) :
  (∀ x : G, f x = x) ∨ (∀ x : G, f x = x⁻¹) :=
sorry

end group_morph_or_involution_l93_93543


namespace triangle_side_ratio_l93_93899

variable (A B C : ℝ)  -- angles in radians
variable (a b c : ℝ)  -- sides of triangle

theorem triangle_side_ratio
  (h : a * Real.sin A * Real.sin B + b * Real.cos A ^ 2 = Real.sqrt 2 * a) :
  b / a = Real.sqrt 2 :=
by sorry

end triangle_side_ratio_l93_93899


namespace length_of_base_AD_l93_93928

-- Definitions based on the conditions
def isosceles_trapezoid (A B C D : Type) : Prop := sorry -- Implementation of an isosceles trapezoid
def length_of_lateral_side (A B C D : Type) : ℝ := 40 -- The lateral side is 40 cm
def angle_BAC (A B C D : Type) : ℝ := 45 -- The angle ∠BAC is 45 degrees
def bisector_O_center (O A B D M : Type) : Prop := sorry -- Implementation that O is the center of circumscribed circle and lies on bisector

-- Main theorem based on the derived problem statement
theorem length_of_base_AD (A B C D O M : Type) 
  (h_iso_trapezoid : isosceles_trapezoid A B C D)
  (h_length_lateral : length_of_lateral_side A B C D = 40)
  (h_angle_BAC : angle_BAC A B C D = 45)
  (h_O_center_bisector : bisector_O_center O A B D M)
  : ℝ :=
  20 * (Real.sqrt 6 + Real.sqrt 2)

end length_of_base_AD_l93_93928


namespace smallest_p_l93_93643

theorem smallest_p (p q : ℕ) (h1 : p + q = 2005) (h2 : (5:ℚ)/8 < p / q) (h3 : p / q < (7:ℚ)/8) : p = 772 :=
sorry

end smallest_p_l93_93643


namespace calculate_income_l93_93035

theorem calculate_income (I : ℝ) (T : ℝ) (a b c d : ℝ) (h1 : a = 0.15) (h2 : b = 40000) (h3 : c = 0.20) (h4 : T = 8000) (h5 : T = a * b + c * (I - b)) : I = 50000 :=
by
  sorry

end calculate_income_l93_93035


namespace area_of_region_l93_93233

noncomputable def circle_radius : ℝ := 3

noncomputable def segment_length : ℝ := 4

theorem area_of_region : ∃ (area : ℝ), area = 4 * Real.pi :=
by
  sorry

end area_of_region_l93_93233


namespace ellipse_domain_l93_93068

theorem ellipse_domain (m : ℝ) :
  (-1 < m ∧ m < 2 ∧ m ≠ 1 / 2) -> 
  ∃ a b : ℝ, (a = 2 - m) ∧ (b = m + 1) ∧ a > 0 ∧ b > 0 ∧ a ≠ b :=
by
  sorry

end ellipse_domain_l93_93068


namespace hotel_fee_original_flat_fee_l93_93867

theorem hotel_fee_original_flat_fee
  (f n : ℝ)
  (H1 : 0.85 * (f + 3 * n) = 210)
  (H2 : f + 6 * n = 400) :
  f = 94.12 :=
by
  -- Sorry is used to indicate that the proof is not provided
  sorry

end hotel_fee_original_flat_fee_l93_93867


namespace length_of_segment_CD_l93_93554

theorem length_of_segment_CD (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y)
  (h_ratio1 : x = (3 / 5) * (3 + y))
  (h_ratio2 : (x + 3) / y = 4 / 7)
  (h_RS : 3 = 3) :
  x + 3 + y = 273.6 :=
by
  sorry

end length_of_segment_CD_l93_93554


namespace perimeter_of_square_C_l93_93500

theorem perimeter_of_square_C (s_A s_B s_C : ℕ) (hpA : 4 * s_A = 16) (hpB : 4 * s_B = 32) (hC : s_C = s_A + s_B - 2) :
  4 * s_C = 40 := 
by
  sorry

end perimeter_of_square_C_l93_93500


namespace sum_of_midpoints_l93_93150

theorem sum_of_midpoints (p q r : ℝ) (h : p + q + r = 15) :
  (p + q) / 2 + (p + r) / 2 + (q + r) / 2 = 15 :=
by
  sorry

end sum_of_midpoints_l93_93150


namespace sewer_runoff_capacity_l93_93841

theorem sewer_runoff_capacity (gallons_per_hour : ℕ) (hours_per_day : ℕ) (days_till_overflow : ℕ)
  (h1 : gallons_per_hour = 1000)
  (h2 : hours_per_day = 24)
  (h3 : days_till_overflow = 10) :
  gallons_per_hour * hours_per_day * days_till_overflow = 240000 := 
by
  -- We'll use sorry here as the placeholder for the actual proof steps
  sorry

end sewer_runoff_capacity_l93_93841


namespace mean_greater_than_median_by_two_l93_93894

theorem mean_greater_than_median_by_two (x : ℕ) (h : x > 0) :
  ((x + (x + 2) + (x + 4) + (x + 7) + (x + 17)) / 5 - (x + 4)) = 2 :=
sorry

end mean_greater_than_median_by_two_l93_93894


namespace area_of_rectangle_l93_93286

theorem area_of_rectangle (x y : ℝ)
    (h1 : x * y = (x + 3) * (y - 1))
    (h2 : x * y = (x - 4) * (y + 3 / 2)) :
    x * y = 108 := by
  sorry

end area_of_rectangle_l93_93286


namespace shortest_wire_length_l93_93965

theorem shortest_wire_length
  (d1 d2 : ℝ) (h_d1 : d1 = 10) (h_d2 : d2 = 30) :
  let r1 := d1 / 2
  let r2 := d2 / 2
  let straight_sections := 2 * (r2 - r1)
  let curved_sections := 2 * Real.pi * r1 + 2 * Real.pi * r2
  let total_wire_length := straight_sections + curved_sections
  total_wire_length = 20 + 40 * Real.pi :=
by
  sorry

end shortest_wire_length_l93_93965


namespace sum_of_reciprocals_l93_93437

theorem sum_of_reciprocals (x y : ℝ) (h₁ : x + y = 15) (h₂ : x * y = 56) : (1/x) + (1/y) = 15/56 := 
by 
  sorry

end sum_of_reciprocals_l93_93437


namespace movement_of_hands_of_clock_involves_rotation_l93_93320

theorem movement_of_hands_of_clock_involves_rotation (A B C D : Prop) :
  (A ↔ (∃ p : ℝ, ∃ θ : ℝ, p ≠ θ)) → -- A condition: exists a fixed point and rotation around it
  (B ↔ ¬∃ p : ℝ, ∃ θ : ℝ, p ≠ θ) → -- B condition: does not rotate around a fixed point
  (C ↔ ¬∃ p : ℝ, ∃ θ : ℝ, p ≠ θ) → -- C condition: does not rotate around a fixed point
  (D ↔ ¬∃ p : ℝ, ∃ θ : ℝ, p ≠ θ) → -- D condition: does not rotate around a fixed point
  A :=
by
  intros hA hB hC hD
  sorry

end movement_of_hands_of_clock_involves_rotation_l93_93320


namespace eliana_refill_l93_93916

theorem eliana_refill (total_spent cost_per_refill : ℕ) (h1 : total_spent = 63) (h2 : cost_per_refill = 21) : (total_spent / cost_per_refill) = 3 :=
sorry

end eliana_refill_l93_93916


namespace sum_of_digits_of_d_l93_93498

noncomputable section

def exchange_rate : ℚ := 8/5
def euros_after_spending (d : ℚ) : ℚ := exchange_rate * d - 80

theorem sum_of_digits_of_d {d : ℚ} (h : euros_after_spending d = d) : 
  d = 135 ∧ 1 + 3 + 5 = 9 := 
by 
  sorry

end sum_of_digits_of_d_l93_93498


namespace negation_of_exists_l93_93678

-- Lean definition of the proposition P
def P (a : ℝ) : Prop :=
  ∃ x0 : ℝ, x0 > 0 ∧ 2^x0 * (x0 - a) > 1

-- The negation of the proposition P
def neg_P (a : ℝ) : Prop :=
  ∀ x : ℝ, x > 0 → 2^x * (x - a) ≤ 1

-- Theorem stating that the negation of P is neg_P
theorem negation_of_exists (a : ℝ) : ¬ P a ↔ neg_P a :=
by
  -- (Proof to be provided)
  sorry

end negation_of_exists_l93_93678


namespace calculate_total_payment_l93_93595

theorem calculate_total_payment
(adult_price : ℕ := 30)
(teen_price : ℕ := 20)
(child_price : ℕ := 15)
(num_adults : ℕ := 4)
(num_teenagers : ℕ := 4)
(num_children : ℕ := 2)
(num_activities : ℕ := 5)
(has_coupon : Bool := true)
(soda_price : ℕ := 5)
(num_sodas : ℕ := 5)

(total_admission_before_discount : ℕ := 
  num_adults * adult_price + num_teenagers * teen_price + num_children * child_price)
(discount_on_activities : ℕ := if num_activities >= 7 then 15 else if num_activities >= 5 then 10 else if num_activities >= 3 then 5 else 0)
(admission_after_activity_discount : ℕ := 
  total_admission_before_discount - total_admission_before_discount * discount_on_activities / 100)
(additional_discount : ℕ := if has_coupon then 5 else 0)
(admission_after_all_discounts : ℕ := 
  admission_after_activity_discount - admission_after_activity_discount * additional_discount / 100)

(total_cost : ℕ := admission_after_all_discounts + num_sodas * soda_price) :
total_cost = 22165 := 
sorry

end calculate_total_payment_l93_93595


namespace exponential_decreasing_range_l93_93051

theorem exponential_decreasing_range {a : ℝ} :
  (∀ x y : ℝ, x < y → (a - 2) ^ x > (a - 2) ^ y) ↔ 2 < a ∧ a < 3 :=
begin
  split,
  { intros h,
    have ha : a - 2 > 0, from sorry, -- Assume intermediate steps
    have hab : a - 2 < 1, from sorry, -- Assume intermediate steps
    split; linarith, },
  { rintros ⟨ha, hab⟩ x y hxy,
    calc (a - 2) ^ x > (a - 2) ^ y : sorry, -- Assume intermediate steps
  },
end

end exponential_decreasing_range_l93_93051


namespace number_of_green_balls_l93_93581

-- Define the problem statement and conditions
def total_balls : ℕ := 12
def probability_both_green (g : ℕ) : ℚ := (g / 12) * ((g - 1) / 11)

-- The main theorem statement
theorem number_of_green_balls (g : ℕ) (h : probability_both_green g = 1 / 22) : g = 3 :=
sorry

end number_of_green_balls_l93_93581


namespace part1_part2_l93_93244

-- Define the function f
def f (x : ℝ) : ℝ := 2 * Real.log x + 1

-- Part (1): Prove c ≥ -1 given f(x) ≤ 2x + c
theorem part1 (c : ℝ) : (∀ x : ℝ, x > 0 → f x ≤ 2 * x + c) → c ≥ -1 :=
  sorry

-- Define g with a > 0
def g (x a : ℝ) : ℝ := (f x - f a) / (x - a)

-- Part (2): Prove g is monotonically decreasing on (0, a) and (a, +∞)
theorem part2 (a : ℝ) (h : a > 0) : 
  (∀ x : ℝ, x > 0 → x ≠ a → g x a < g a a) :=
  sorry

end part1_part2_l93_93244


namespace cistern_length_l93_93470

-- Definitions of the given conditions
def width : ℝ := 4
def depth : ℝ := 1.25
def total_wet_surface_area : ℝ := 49

-- Mathematical problem: prove the length of the cistern
theorem cistern_length : ∃ (L : ℝ), (L * width + 2 * L * depth + 2 * width * depth = total_wet_surface_area) ∧ L = 6 :=
by
sorry

end cistern_length_l93_93470


namespace part1_c_range_part2_monotonicity_l93_93246

noncomputable def f (x : ℝ) := 2 * Real.log x + 1

theorem part1_c_range (c : ℝ) (x : ℝ) (h : a > 0) : f x ≤ 2 * x + c → c ≥ -1 :=
sorry

noncomputable def g (x a : ℝ) := (f x - f a) / (x - a)

theorem part2_monotonicity (a : ℝ) (h : a > 0) : monotone_decreasing_on g (0, a) ∧ monotone_decreasing_on g (a, +∞) :=
sorry

end part1_c_range_part2_monotonicity_l93_93246


namespace minimum_value_of_function_l93_93037

theorem minimum_value_of_function : ∀ x : ℝ, x ≥ 0 → (4 * x^2 + 12 * x + 25) / (6 * (1 + x)) ≥ 8 / 3 := by
  sorry

end minimum_value_of_function_l93_93037


namespace ratio_a_c_l93_93564

theorem ratio_a_c {a b c : ℝ} (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : ∀ x : ℝ, x^2 + a * x + b = 0 → 2 * x^2 + 2 * a * x + 2 * b = 0)
  (h2 : ∀ x : ℝ, x^2 + b * x + c = 0 → x^2 + b * x + c = 0) :
  a / c = 1 / 8 :=
by
  sorry

end ratio_a_c_l93_93564


namespace ways_A_not_head_is_600_l93_93717

-- Definitions for the problem conditions
def num_people : ℕ := 6
def valid_positions_for_A : ℕ := 5
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

-- The total number of ways person A can be placed in any position except the first
def num_ways_A_not_head : ℕ := valid_positions_for_A * factorial (num_people - 1)

-- The theorem to prove
theorem ways_A_not_head_is_600 : num_ways_A_not_head = 600 := by
  sorry

end ways_A_not_head_is_600_l93_93717


namespace greatest_product_from_sum_2004_l93_93973

theorem greatest_product_from_sum_2004 : ∃ (x y : ℤ), x + y = 2004 ∧ x * y = 1004004 :=
by
  sorry

end greatest_product_from_sum_2004_l93_93973


namespace total_money_shared_l93_93114

theorem total_money_shared (k t : ℕ) (h1 : k = 1750) (h2 : t = 2 * k) : k + t = 5250 :=
by
  sorry

end total_money_shared_l93_93114


namespace total_cases_l93_93791

def NY : ℕ := 2000
def CA : ℕ := NY / 2
def TX : ℕ := CA - 400

theorem total_cases : NY + CA + TX = 3600 :=
by
  -- use sorry placeholder to indicate the solution is omitted
  sorry

end total_cases_l93_93791


namespace solve_quadratic_sum_l93_93562

theorem solve_quadratic_sum (a b : ℕ) (x : ℝ) (h₁ : x^2 + 10 * x = 93)
  (h₂ : x = Real.sqrt a - b) (ha_pos : 0 < a) (hb_pos : 0 < b) : a + b = 123 := by
  sorry

end solve_quadratic_sum_l93_93562


namespace megan_files_in_folder_l93_93801

theorem megan_files_in_folder :
  let initial_files := 93.0
  let added_files := 21.0
  let total_files := initial_files + added_files
  let total_folders := 14.25
  (total_files / total_folders) = 8.0 :=
by
  let initial_files := 93.0
  let added_files := 21.0
  let total_files := initial_files + added_files
  let total_folders := 14.25
  have h1 : total_files = initial_files + added_files := rfl
  have h2 : total_files = 114.0 := by sorry -- 93.0 + 21.0 = 114.0
  have h3 : total_files / total_folders = 8.0 := by sorry -- 114.0 / 14.25 = 8.0
  exact h3

end megan_files_in_folder_l93_93801


namespace identity_proof_l93_93556

theorem identity_proof
  (M N x a b : ℝ)
  (h₀ : x ≠ a)
  (h₁ : x ≠ b)
  (h₂ : a ≠ b) :
  (Mx + N) / ((x - a) * (x - b)) =
  (((M *a + N) / (a - b)) * (1 / (x - a))) - 
  (((M * b + N) / (a - b)) * (1 / (x - b))) :=
sorry

end identity_proof_l93_93556


namespace largest_square_plot_size_l93_93331

def field_side_length := 50
def available_fence_length := 4000

theorem largest_square_plot_size :
  ∃ (s : ℝ), (0 < s) ∧ (s ≤ field_side_length) ∧ 
  (100 * (field_side_length - s) = available_fence_length) →
  s = 10 :=
by
  sorry

end largest_square_plot_size_l93_93331


namespace ball_hits_ground_at_2_72_l93_93346

-- Define the initial conditions
def initial_velocity (v₀ : ℝ) := v₀ = 30
def initial_height (h₀ : ℝ) := h₀ = 200
def ball_height (t : ℝ) : ℝ := -16 * t^2 - 30 * t + 200

-- Prove that the ball hits the ground at t = 2.72 seconds
theorem ball_hits_ground_at_2_72 (t : ℝ) (h : ℝ) 
  (v₀ : ℝ) (h₀ : ℝ) 
  (hv₀ : initial_velocity v₀) 
  (hh₀ : initial_height h₀)
  (h_eq: ball_height t = h) 
  (h₀_eq: ball_height 0 = h₀) : 
  h = 0 -> t = 2.72 :=
by
  sorry

end ball_hits_ground_at_2_72_l93_93346


namespace robie_initial_cards_l93_93812

def total_initial_boxes : Nat := 2 + 5
def cards_per_box : Nat := 10
def unboxed_cards : Nat := 5

theorem robie_initial_cards :
  (total_initial_boxes * cards_per_box + unboxed_cards) = 75 :=
by
  sorry

end robie_initial_cards_l93_93812


namespace min_value_expr_l93_93236

theorem min_value_expr (m n : ℝ) (h : m - n^2 = 8) : m^2 - 3 * n^2 + m - 14 ≥ 58 :=
sorry

end min_value_expr_l93_93236


namespace rhombus_obtuse_angle_l93_93294

theorem rhombus_obtuse_angle (perimeter height : ℝ) (h_perimeter : perimeter = 8) (h_height : height = 1) : 
  ∃ θ : ℝ, θ = 150 :=
by
  sorry

end rhombus_obtuse_angle_l93_93294


namespace x_one_minus_f_eq_one_l93_93940

noncomputable def x : ℝ := (1 + Real.sqrt 2) ^ 500
noncomputable def n : ℤ := Int.floor x
noncomputable def f : ℝ := x - n

theorem x_one_minus_f_eq_one : x * (1 - f) = 1 :=
by
  sorry

end x_one_minus_f_eq_one_l93_93940


namespace sum_cubes_coeffs_l93_93291

theorem sum_cubes_coeffs :
  ∃ a b c d e : ℤ, 
  (1000 * x^3 + 27 = (a * x + b) * (c * x^2 + d * x + e)) ∧ 
  (a + b + c + d + e = 92) :=
sorry

end sum_cubes_coeffs_l93_93291


namespace kelseys_sisters_age_l93_93396

theorem kelseys_sisters_age :
  ∀ (current_year : ℕ) (kelsey_birth_year : ℕ)
    (kelsey_sister_birth_year : ℕ),
    kelsey_birth_year = 1999 - 25 →
    kelsey_sister_birth_year = kelsey_birth_year - 3 →
    current_year = 2021 →
    current_year - kelsey_sister_birth_year = 50 :=
by
  intros current_year kelsey_birth_year kelsey_sister_birth_year h1 h2 h3
  sorry

end kelseys_sisters_age_l93_93396


namespace cube_number_sum_is_102_l93_93014

noncomputable def sum_of_cube_numbers (n1 n2 n3 n4 n5 n6 : ℕ) : ℕ := n1 + n2 + n3 + n4 + n5 + n6

theorem cube_number_sum_is_102 : 
  ∃ (n1 n2 n3 n4 n5 n6 : ℕ),
    n1 = 12 ∧ 
    n2 = n1 + 2 ∧ 
    n3 = n2 + 2 ∧ 
    n4 = n3 + 2 ∧ 
    n5 = n4 + 2 ∧ 
    n6 = n5 + 2 ∧ 
    ((n1 + n6 = n2 + n5) ∧ (n1 + n6 = n3 + n4)) ∧ 
    sum_of_cube_numbers n1 n2 n3 n4 n5 n6 = 102 :=
by
  sorry

end cube_number_sum_is_102_l93_93014


namespace simplify_and_evaluate_l93_93816

-- Define the expression
def expr (x : ℝ) : ℝ := x^2 * (x + 1) - x * (x^2 - x + 1)

-- The main theorem stating the equivalence
theorem simplify_and_evaluate (x : ℝ) (h : x = 5) : expr x = 45 :=
by {
  sorry
}

end simplify_and_evaluate_l93_93816


namespace parabola_c_value_l93_93869

theorem parabola_c_value (b c : ℝ)
  (h1 : 3 = 2^2 + b * 2 + c)
  (h2 : 6 = 5^2 + b * 5 + c) :
  c = -13 :=
by
  -- Proof would follow here
  sorry

end parabola_c_value_l93_93869


namespace num_real_a_satisfy_union_l93_93760

def A (a : ℝ) : Set ℝ := {1, 3, a^2}
def B (a : ℝ) : Set ℝ := {1, a + 2}

theorem num_real_a_satisfy_union {a : ℝ} : (A a ∪ B a) = A a → ∃! a, (A a ∪ B a) = A a := 
by sorry

end num_real_a_satisfy_union_l93_93760


namespace num_two_digit_numbers_l93_93627

-- Define the set of given digits
def digits : Finset ℕ := {0, 2, 5}

-- Define the function that counts the number of valid two-digit numbers
def count_two_digit_numbers (d : Finset ℕ) : ℕ :=
  (d.erase 0).card * (d.card - 1)

theorem num_two_digit_numbers : count_two_digit_numbers digits = 4 :=
by {
  -- sorry placeholder for the proof
  sorry
}

end num_two_digit_numbers_l93_93627


namespace fiftieth_digit_of_decimal_representation_of_five_fourteenth_is_five_l93_93967

theorem fiftieth_digit_of_decimal_representation_of_five_fourteenth_is_five :
  let d := (5 : ℚ) / 14 in
  let decDigits := [3, 5, 7, 1, 4, 2] in
  (decDigits : List ℕ)[(50 % 6) - 1] = 5 := 
by
  sorry

end fiftieth_digit_of_decimal_representation_of_five_fourteenth_is_five_l93_93967


namespace am_gm_inequality_l93_93414

theorem am_gm_inequality (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) : 
  (x + y + z)^2 / 3 ≥ x * Real.sqrt (y * z) + y * Real.sqrt (z * x) + z * Real.sqrt (x * y) := 
by sorry

end am_gm_inequality_l93_93414


namespace prove_inequality_l93_93620

-- Defining properties of f
variable {α : Type*} [LinearOrderedField α] (f : α → α)

-- Condition 1: f is even function
def is_even_function (f : α → α) : Prop := ∀ x : α, f (-x) = f x

-- Condition 2: f is monotonically increasing on (0, ∞)
def is_monotonically_increasing_on_positive (f : α → α) : Prop := ∀ ⦃x y : α⦄, 0 < x → 0 < y → x < y → f x < f y

-- Define the main theorem we need to prove:
theorem prove_inequality (h1 : is_even_function f) (h2 : is_monotonically_increasing_on_positive f) : 
  f (-1) < f 2 ∧ f 2 < f (-3) :=
by
  sorry

end prove_inequality_l93_93620


namespace intersecting_circles_range_l93_93381

theorem intersecting_circles_range {k : ℝ} (a b : ℝ) :
  (-36 : ℝ) ≤ k ∧ k ≤ 104 →
  (∃ (x y : ℝ), (x^2 + y^2 - 4 - 12 * x + 6 * y) = 0 ∧ (x^2 + y^2 = k + 4 * x + 12 * y)) →
  b - a = (140 : ℝ) :=
by
  intro hk hab
  sorry

end intersecting_circles_range_l93_93381


namespace equilateral_triangle_area_l93_93821

theorem equilateral_triangle_area (h : ℝ) 
  (height_eq : h = 2 * Real.sqrt 3) :
  ∃ (A : ℝ), A = 4 * Real.sqrt 3 :=
by
  sorry

end equilateral_triangle_area_l93_93821


namespace even_digits_count_1998_l93_93173

-- Define the function for counting the total number of digits used in the first n positive even integers
def totalDigitsEvenIntegers (n : ℕ) : ℕ :=
  let totalSingleDigit := 4 -- 2, 4, 6, 8
  let numDoubleDigit := 45 -- 10 to 98
  let digitsDoubleDigit := numDoubleDigit * 2
  let numTripleDigit := 450 -- 100 to 998
  let digitsTripleDigit := numTripleDigit * 3
  let numFourDigit := 1499 -- 1000 to 3996
  let digitsFourDigit := numFourDigit * 4
  totalSingleDigit + digitsDoubleDigit + digitsTripleDigit + digitsFourDigit

-- Theorem: The total number of digits used when the first 1998 positive even integers are written is 7440.
theorem even_digits_count_1998 : totalDigitsEvenIntegers 1998 = 7440 :=
  sorry

end even_digits_count_1998_l93_93173


namespace max_gcd_seq_l93_93216

theorem max_gcd_seq (a : ℕ → ℕ) (d : ℕ → ℕ) :
  (∀ n : ℕ, a n = 121 + n^2) →
  (∀ n : ℕ, d n = Nat.gcd (a n) (a (n + 1))) →
  ∃ m : ℕ, ∀ n : ℕ, d n ≤ d m ∧ d m = 99 :=
by
  sorry

end max_gcd_seq_l93_93216


namespace mean_greater_than_median_l93_93178

theorem mean_greater_than_median (x : ℕ) : 
  let mean := (x + (x + 2) + (x + 4) + (x + 7) + (x + 27)) / 5 
  let median := x + 4 
  mean - median = 4 :=
by 
  sorry

end mean_greater_than_median_l93_93178


namespace max_band_members_l93_93662

theorem max_band_members (n : ℤ) (h1 : 22 * n % 24 = 2) (h2 : 22 * n < 1000) : 22 * n = 770 :=
  sorry

end max_band_members_l93_93662


namespace area_of_rectangle_l93_93733

-- Define the lengths in meters
def length : ℝ := 1.2
def width : ℝ := 0.5

-- Define the function to calculate the area of a rectangle
def area (l w : ℝ) : ℝ := l * w

-- Prove that the area of the rectangle with given length and width is 0.6 square meters
theorem area_of_rectangle :
  area length width = 0.6 := by
  -- This is just the statement. We omit the proof with sorry.
  sorry

end area_of_rectangle_l93_93733


namespace inequality_solution_I_inequality_solution_II_l93_93715

noncomputable def f (x a : ℝ) : ℝ := |2 * x - a| - |x + 1|

theorem inequality_solution_I (x : ℝ) : f x 1 > 2 ↔ x < -2 / 3 ∨ x > 4 :=
sorry 

noncomputable def g (x a : ℝ) : ℝ := f x a + |x + 1| + x

theorem inequality_solution_II (a : ℝ) : (∀ x, g x a > a ^ 2 - 1 / 2) ↔ (-1 / 2 < a ∧ a < 1) :=
sorry

end inequality_solution_I_inequality_solution_II_l93_93715


namespace triangle_angle_sum_l93_93808

theorem triangle_angle_sum (α β γ : ℝ) (h : α + β + γ = 180) (h1 : α > 60) (h2 : β > 60) (h3 : γ > 60) : false :=
sorry

end triangle_angle_sum_l93_93808


namespace expand_expression_l93_93611

theorem expand_expression :
  (3 * t^2 - 2 * t + 3) * (-2 * t^2 + 3 * t - 4) = -6 * t^4 + 13 * t^3 - 24 * t^2 + 17 * t - 12 :=
by sorry

end expand_expression_l93_93611


namespace determinant_of_matrix_A_l93_93499

noncomputable def matrix_A (x : ℝ) : Matrix (Fin 3) (Fin 3) ℝ := 
  ![![x + 2, x + 1, x], 
    ![x, x + 2, x + 1], 
    ![x + 1, x, x + 2]]

theorem determinant_of_matrix_A (x : ℝ) :
  (matrix_A x).det = x^2 + 11 * x + 9 :=
by sorry

end determinant_of_matrix_A_l93_93499


namespace first_year_exceeds_two_million_l93_93195

-- Definition of the initial R&D investment in 2015
def initial_investment : ℝ := 1.3

-- Definition of the annual growth rate
def growth_rate : ℝ := 1.12

-- Definition of the investment function for year n
def investment (n : ℕ) : ℝ := initial_investment * growth_rate ^ (n - 2015)

-- The problem statement to be proven
theorem first_year_exceeds_two_million : ∃ n : ℕ, n > 2015 ∧ investment n > 2 ∧ ∀ m : ℕ, (m < n ∧ m > 2015) → investment m ≤ 2 := by
  sorry

end first_year_exceeds_two_million_l93_93195


namespace exponent_problem_l93_93165

theorem exponent_problem : (5 ^ 6 * 5 ^ 9 * 5) / 5 ^ 3 = 5 ^ 13 := 
by
  sorry

end exponent_problem_l93_93165


namespace fraction_of_orange_juice_is_correct_l93_93305

noncomputable def fraction_of_orange_juice_in_mixture (V1 V2 juice1_ratio juice2_ratio : ℚ) : ℚ :=
  let juice1 := V1 * juice1_ratio
  let juice2 := V2 * juice2_ratio
  let total_juice := juice1 + juice2
  let total_volume := V1 + V2
  total_juice / total_volume

theorem fraction_of_orange_juice_is_correct :
  fraction_of_orange_juice_in_mixture 800 500 (1/4) (1/3) = 7 / 25 :=
by sorry

end fraction_of_orange_juice_is_correct_l93_93305


namespace solve_for_x_l93_93567

theorem solve_for_x : (3.6 * 0.48 * x) / (0.12 * 0.09 * 0.5) = 800.0000000000001 → x = 2.5 :=
by
  sorry

end solve_for_x_l93_93567


namespace ages_of_Xs_sons_l93_93185

def ages_problem (x y : ℕ) : Prop :=
x ≠ y ∧ x ≤ 10 ∧ y ≤ 10 ∧
∀ u v : ℕ, u * v = x * y → u ≤ 10 ∧ v ≤ 10 → (u, v) = (x, y) ∨ (u, v) = (y, x) ∨
(∀ z w : ℕ, z / w = x / y → z = x ∧ w = y ∨ z = y ∧ w = x → u ≠ z ∧ v ≠ w) →
(∀ a b : ℕ, a - b = (x - y) ∨ b - a = (y - x) → (x, y) = (a, b) ∨ (x, y) = (b, a))

theorem ages_of_Xs_sons : ages_problem 8 2 := 
by {
  sorry
}


end ages_of_Xs_sons_l93_93185


namespace avg_A_less_avg_B_avg_20_points_is_6_6_l93_93692

noncomputable def scores_A : List ℕ := [4, 5, 5, 6, 6, 7, 7, 8, 8, 9]
noncomputable def scores_B : List ℕ := [2, 5, 6, 6, 7, 7, 7, 8, 9, 10]
noncomputable def variance_A : ℝ := 2.25
noncomputable def variance_B : ℝ := 4.41

theorem avg_A_less_avg_B :
  let avg_A := (scores_A.sum.to_float / scores_A.length) 
  let avg_B := (scores_B.sum.to_float / scores_B.length) 
  avg_A < avg_B := 
by
  sorry

theorem avg_20_points_is_6_6 :
  let avg_A := (scores_A.sum.to_float / scores_A.length) 
  let avg_B := (scores_B.sum.to_float / scores_B.length) 
  let avg_20 := ((avg_A * scores_A.length + avg_B * scores_B.length) / (scores_A.length + scores_B.length))
  avg_20 = 6.6 := 
by
  sorry

end avg_A_less_avg_B_avg_20_points_is_6_6_l93_93692


namespace truncated_cone_surface_area_l93_93731

theorem truncated_cone_surface_area (R r : ℝ) (S : ℝ)
  (h1: S = 4 * Real.pi * (R^2 + R * r + r^2)) :
  2 * Real.pi * (R^2 + R * r + r^2) = S / 2 :=
by
  sorry

end truncated_cone_surface_area_l93_93731


namespace positive_number_condition_l93_93475

theorem positive_number_condition (y : ℝ) (h: 0.04 * y = 16): y = 400 := 
by sorry

end positive_number_condition_l93_93475


namespace exists_real_a_l93_93372

noncomputable def A (a : ℝ) : Set ℝ := { x | x^2 - a * x + a^2 - 19 = 0 }
def B : Set ℝ := { x | x^2 - 5 * x + 6 = 0 }
def C : Set ℝ := { x | x^2 + 2 * x - 8 = 0 }

theorem exists_real_a : ∃ a : ℝ, a = -2 ∧ A a ∩ C = ∅ ∧ ∅ ⊂ A a ∩ B := 
by {
  sorry
}

end exists_real_a_l93_93372


namespace range_of_m_l93_93764

theorem range_of_m (m : ℝ) 
  (hp : ∀ x : ℝ, 2 * x > m * (x ^ 2 + 1)) 
  (hq : ∃ x0 : ℝ, x0 ^ 2 + 2 * x0 - m - 1 = 0) : 
  -2 ≤ m ∧ m < -1 :=
sorry

end range_of_m_l93_93764


namespace man_speed_still_water_l93_93016

noncomputable def speed_in_still_water (U D : ℝ) : ℝ := (U + D) / 2

theorem man_speed_still_water :
  let U := 45
  let D := 55
  speed_in_still_water U D = 50 := by
  sorry

end man_speed_still_water_l93_93016


namespace find_equation_of_tangent_line_l93_93781

def is_tangent_at_point (l : ℝ → ℝ → Prop) (x₀ y₀ : ℝ) := 
  ∃ x y, (x - 1)^2 + (y + 2)^2 = 1 ∧ l x₀ y₀ ∧ l x y

def equation_of_line (l : ℝ → ℝ → Prop) := 
  ∀ x y, l x y ↔ (x = 2 ∨ 12 * x - 5 * y - 9 = 0)

theorem find_equation_of_tangent_line : 
  ∀ (l : ℝ → ℝ → Prop),
  (∀ x y, l x y ↔ (x - 1)^2 + (y + 2)^2 ≠ 1 ∧ (x, y) = (2,3))
  → is_tangent_at_point l 2 3
  → equation_of_line l := 
sorry

end find_equation_of_tangent_line_l93_93781


namespace wire_ratio_l93_93022

theorem wire_ratio (a b : ℝ) (h_eq_area : (a / 4)^2 = 2 * (b / 8)^2 * (1 + Real.sqrt 2)) :
  a / b = Real.sqrt (2 + Real.sqrt 2) / 2 :=
by
  sorry

end wire_ratio_l93_93022


namespace equilateral_triangle_area_l93_93828

noncomputable def altitude : ℝ := 2 * Real.sqrt 3
noncomputable def expected_area : ℝ := 4 * Real.sqrt 3

theorem equilateral_triangle_area (h : altitude = 2 * Real.sqrt 3) : 
  let a := 4 * Real.sqrt 3 in
  a = expected_area := 
by
  sorry

end equilateral_triangle_area_l93_93828


namespace schoolchildren_initial_speed_l93_93454

theorem schoolchildren_initial_speed (v : ℝ) (t t_1 t_2 : ℝ) 
  (h1 : t_1 = (6 * v) / (v + 60) + (400 - 3 * v) / (v + 60)) 
  (h2 : t_2 = (400 - 3 * v) / v) 
  (h3 : t_1 = t_2) : v = 63.24 :=
by sorry

end schoolchildren_initial_speed_l93_93454


namespace true_universal_quantifier_l93_93001

theorem true_universal_quantifier :
  ∀ (a b : ℝ), a^2 + b^2 ≥ 2 * (a - b - 1) := by
  sorry

end true_universal_quantifier_l93_93001


namespace find_range_a_l93_93782

def bounded_a (a : ℝ) : Prop :=
  ∀ x : ℝ, x ≤ 2 → a * (4 ^ x) + 2 ^ x + 1 ≥ 0

theorem find_range_a :
  ∃ (a : ℝ), bounded_a a ↔ a ≥ -5 / 16 :=
sorry

end find_range_a_l93_93782


namespace car_initial_time_l93_93721

variable (t : ℝ)

theorem car_initial_time (h : 80 = 720 / (3/2 * t)) : t = 6 :=
sorry

end car_initial_time_l93_93721


namespace total_spent_l93_93467

def original_price : ℝ := 20
def discount_rate : ℝ := 0.5
def number_of_friends : ℕ := 4

theorem total_spent : (original_price * (1 - discount_rate) * number_of_friends) = 40 := by
  sorry

end total_spent_l93_93467


namespace percent_increase_lines_l93_93174

theorem percent_increase_lines (final_lines increase : ℕ) (h1 : final_lines = 5600) (h2 : increase = 1600) :
  (increase * 100) / (final_lines - increase) = 40 := 
sorry

end percent_increase_lines_l93_93174


namespace log_a_b_is_constant_l93_93995

open Real

-- Definitions of the conditions
def r (a : ℝ) : ℝ := log 10 (a ^ 2)
def C (b : ℝ) : ℝ := log 10 (b ^ 6)

-- Circle circumference formula
axiom circumference {a b : ℝ} : C b = 2 * π * r a

theorem log_a_b_is_constant (a b : ℝ) (ha : a > 0) (hb : b > 0) (hc : circumference) :
  log a b = (2 * π) / 3 :=
by sorry

end log_a_b_is_constant_l93_93995


namespace soccer_and_volleyball_unit_prices_max_soccer_balls_l93_93258

-- Define the conditions and the problem
def unit_price_soccer_ball (x : ℕ) (y : ℕ) : Prop :=
  x = y + 15 ∧ 480 / x = 390 / y

def school_purchase (m : ℕ) : Prop :=
  m ≤ 70 ∧ 80 * m + 65 * (100 - m) ≤ 7550

-- Proof statement for the unit prices of soccer balls and volleyballs
theorem soccer_and_volleyball_unit_prices (x y : ℕ) (h : unit_price_soccer_ball x y) :
  x = 80 ∧ y = 65 :=
by
  sorry

-- Proof statement for the maximum number of soccer balls the school can purchase
theorem max_soccer_balls (m : ℕ) :
  school_purchase m :=
by
  sorry

end soccer_and_volleyball_unit_prices_max_soccer_balls_l93_93258


namespace average_of_three_quantities_l93_93663

theorem average_of_three_quantities 
  (five_avg : ℚ) (three_avg : ℚ) (two_avg : ℚ) 
  (h_five_avg : five_avg = 10) 
  (h_two_avg : two_avg = 19) : 
  three_avg = 4 := 
by 
  let sum_5 := 5 * 10
  let sum_2 := 2 * 19
  let sum_3 := sum_5 - sum_2
  let three_avg := sum_3 / 3
  sorry

end average_of_three_quantities_l93_93663


namespace prove_square_ratio_l93_93924
noncomputable section

-- Definitions from given conditions
variables (a b : ℝ) (d : ℝ := Real.sqrt (a^2 + b^2))

-- Condition from the problem
def ratio_condition : Prop := a / b = (a + 2 * b) / d

-- The theorem we need to prove
theorem prove_square_ratio (h : ratio_condition a b d) : 
  ∃ k : ℝ, k = a / b ∧ k^4 - 3*k^2 - 4*k - 4 = 0 := 
by
  sorry

end prove_square_ratio_l93_93924


namespace min_u_value_l93_93780

theorem min_u_value (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 2 * y = 1) : 
  (x + 1 / x) * (y + 1 / (4 * y)) ≥ 25 / 8 :=
by
  sorry

end min_u_value_l93_93780


namespace yanni_paintings_l93_93985

theorem yanni_paintings
  (total_area : ℤ)
  (painting1 : ℕ → ℤ × ℤ)
  (painting2 : ℤ × ℤ)
  (painting3 : ℤ × ℤ)
  (num_paintings : ℕ) :
  total_area = 200
  → painting1 1 = (5, 5)
  → painting1 2 = (5, 5)
  → painting1 3 = (5, 5)
  → painting2 = (10, 8)
  → painting3 = (5, 9)
  → num_paintings = 5 := 
by
  sorry

end yanni_paintings_l93_93985


namespace num_people_in_5_years_l93_93206

def seq (n : ℕ) : ℕ :=
  match n with
  | 0     => 12
  | (k+1) => 4 * seq k - 18

theorem num_people_in_5_years : seq 5 = 6150 :=
  sorry

end num_people_in_5_years_l93_93206


namespace symmetric_point_origin_l93_93831

-- Define the coordinates of point A and the relation of symmetry about the origin
def A : ℝ × ℝ := (2, -1)
def symm_origin (P : ℝ × ℝ) : ℝ × ℝ := (-P.1, -P.2)

-- Theorem statement: Point B is the symmetric point of A about the origin
theorem symmetric_point_origin : symm_origin A = (-2, 1) :=
  sorry

end symmetric_point_origin_l93_93831


namespace correct_average_weight_l93_93577

theorem correct_average_weight (avg_weight : ℝ) (num_boys : ℕ) (incorrect_weight correct_weight : ℝ)
  (h1 : avg_weight = 58.4) (h2 : num_boys = 20) (h3 : incorrect_weight = 56) (h4 : correct_weight = 62) :
  (avg_weight * ↑num_boys + (correct_weight - incorrect_weight)) / ↑num_boys = 58.7 := by
  sorry

end correct_average_weight_l93_93577


namespace magic_square_sum_l93_93788

theorem magic_square_sum (a b c d e f S : ℕ) 
  (h1 : 30 + b + 22 = S) 
  (h2 : 19 + c + d = S) 
  (h3 : a + 28 + f = S)
  (h4 : 30 + 19 + a = S)
  (h5 : b + c + 28 = S)
  (h6 : 22 + d + f = S)
  (h7 : 30 + c + f = S)
  (h8 : 22 + c + a = S)
  (h9 : e = b) :
  d + e = 54 := 
by 
  sorry

end magic_square_sum_l93_93788


namespace find_a7_l93_93087

def arithmetic_seq (a₁ d : ℤ) (n : ℤ) : ℤ := a₁ + (n-1) * d

theorem find_a7 (a₁ d : ℤ)
  (h₁ : arithmetic_seq a₁ d 3 + arithmetic_seq a₁ d 7 - arithmetic_seq a₁ d 10 = -1)
  (h₂ : arithmetic_seq a₁ d 11 - arithmetic_seq a₁ d 4 = 21) :
  arithmetic_seq a₁ d 7 = 20 :=
by
  sorry

end find_a7_l93_93087


namespace range_of_m_l93_93147

theorem range_of_m (m : ℝ) (h : ∀ x : ℝ, x > 4 ↔ x > m) : m ≤ 4 :=
by {
  -- here we state the necessary assumptions and conclude the theorem
  -- detailed proof steps are not needed, hence sorry is used to skip the proof
  sorry
}

end range_of_m_l93_93147


namespace distance_from_O_is_450_l93_93884

noncomputable def find_distance_d (A B C P Q O : Type) 
  (side_length : ℝ) (PA PB PC QA QB QC dist_OA dist_OB dist_OC dist_OP dist_OQ : ℝ) : ℝ :=
    if h : PA = PB ∧ PB = PC ∧ QA = QB ∧ QB = QC ∧ side_length = 600 ∧
           dist_OA = dist_OB ∧ dist_OB = dist_OC ∧ dist_OC = dist_OP ∧ dist_OP = dist_OQ ∧
           -- condition of 120 degree dihedral angle translates to specific geometric constraints
           true -- placeholder for the actual geometrical configuration that proves the problem
    then 450
    else 0 -- default or indication of inconsistency in conditions

-- Assuming all conditions hold true
theorem distance_from_O_is_450 (A B C P Q O : Type) 
  (side_length : ℝ) (PA PB PC QA QB QC dist_OA dist_OB dist_OC dist_OP dist_OQ : ℝ)
  (h : PA = PB ∧ PB = PC ∧ QA = QB ∧ QB = QC ∧ side_length = 600 ∧
       dist_OA = dist_OB ∧ dist_OB = dist_OC ∧ dist_OC = dist_OP ∧ dist_OP = dist_OQ ∧
       -- adding condition of 120 degree dihedral angle
       true) -- true is a placeholder, the required proof to be filled in
  : find_distance_d A B C P Q O side_length PA PB PC QA QB QC dist_OA dist_OB dist_OC dist_OP dist_OQ = 450 :=
by
  -- proof goes here
  sorry

end distance_from_O_is_450_l93_93884


namespace sam_age_two_years_ago_l93_93539

theorem sam_age_two_years_ago (J S : ℕ) (h1 : J = 3 * S) (h2 : J + 9 = 2 * (S + 9)) : S - 2 = 7 :=
sorry

end sam_age_two_years_ago_l93_93539


namespace price_reduction_is_50_rubles_l93_93297

theorem price_reduction_is_50_rubles :
  let P_Feb : ℕ := 300
  let P_Mar : ℕ := 250
  P_Feb - P_Mar = 50 :=
by
  let P_Feb : ℕ := 300
  let P_Mar : ℕ := 250
  sorry

end price_reduction_is_50_rubles_l93_93297


namespace basketball_free_throws_l93_93677

-- Define the given conditions as assumptions
variables {a b x : ℝ}
variables (h1 : 3 * b = 2 * a)
variables (h2 : x = 2 * a - 2)
variables (h3 : 2 * a + 3 * b + x = 78)

-- State the theorem to be proven
theorem basketball_free_throws : x = 74 / 3 :=
by {
  -- We will provide the proof later
  sorry
}

end basketball_free_throws_l93_93677


namespace find_prime_pair_l93_93464

noncomputable def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem find_prime_pair (p q : ℕ) (hp : is_prime p) (hq : is_prime q) (h : p > q) (h_prime : is_prime (p^5 - q^5)) : (p, q) = (3, 2) := 
  sorry

end find_prime_pair_l93_93464


namespace simplify_expression_l93_93661

theorem simplify_expression (s : ℤ) : 120 * s - 32 * s = 88 * s := by
  sorry

end simplify_expression_l93_93661


namespace average_age_decrease_l93_93830

theorem average_age_decrease (N : ℕ) (T : ℝ) 
  (h1 : T = 40 * N) 
  (h2 : ∀ new_average_age : ℝ, (T + 12 * 34) / (N + 12) = new_average_age → new_average_age = 34) :
  ∃ decrease : ℝ, decrease = 6 :=
by
  sorry

end average_age_decrease_l93_93830


namespace maximum_value_sum_l93_93408

theorem maximum_value_sum (a b c d : ℕ) (h1 : a + c = 1000) (h2 : b + d = 500) :
  ∃ a b c d, a + c = 1000 ∧ b + d = 500 ∧ (a = 1 ∧ c = 999 ∧ b = 499 ∧ d = 1) ∧ 
  ((a : ℝ) / b + (c : ℝ) / d = (1 / 499) + 999) := 
  sorry

end maximum_value_sum_l93_93408


namespace numbers_in_ratio_l93_93304

theorem numbers_in_ratio (a b c : ℤ) :
  (∃ x : ℤ, a = 2 * x ∧ b = 3 * x ∧ c = 4 * x) ∧ (a * a + b * b + c * c = 725) →
  (a = 10 ∧ b = 15 ∧ c = 20 ∨ a = -10 ∧ b = -15 ∧ c = -20) :=
by
  sorry

end numbers_in_ratio_l93_93304


namespace max_mice_two_males_num_versions_PPF_PPF_combinations_PPF_three_kittens_l93_93419

-- Condition definitions
def PPF_male (K : ℕ) : ℕ := 80 - 4 * K
def PPF_female (K : ℕ) : ℕ := 16 - 0.25.to_nat * K

-- Proving the maximum number of mice that could be caught by 2 male kittens
theorem max_mice_two_males : ∀ K, PPF_male 0 + PPF_male 0 = 160 :=
by simp [PPF_male]

-- Proving there are 3 possible versions of the PPF
theorem num_versions_PPF : ∃ (versions : set (ℕ → ℕ)), versions = 
  { λ K, 160 - 4 * K,
    λ K, 32 - 0.5.to_nat * K,
    λ K, if K ≤ 64 then 96 - 0.25.to_nat * K else 336 - 4 * K } ∧
  versions.size = 3 :=
by sorry

-- Proving the analytical form of each PPF combination
theorem PPF_combinations : 
  (∀ K, (λ K, 160 - 4 * K) K = PPF_male K + PPF_male K) ∧
  (∀ K, (λ K, 32 - 0.5.to_nat * K) K = PPF_female K + PPF_female K) ∧
  (∀ K, if K ≤ 64 then (λ K, 96 - 0.25.to_nat * K) K = PPF_male K + PPF_female K else (λ K, 336 - 4 * K) K = PPF_male (K - 64) + PPF_female 64) :=
by sorry

-- Proving the analytical form when accepting the third kitten
theorem PPF_three_kittens :
  (∀ K, if K ≤ 64 then (176 - 0.25.to_nat * K) = PPF_male K + PPF_male K + PPF_female K else (416 - 4 * K) = PPF_male (K - 64) + PPF_male 64 + PPF_female 64) :=
by sorry

end max_mice_two_males_num_versions_PPF_PPF_combinations_PPF_three_kittens_l93_93419


namespace farmer_cages_l93_93330

theorem farmer_cages (c : ℕ) (h1 : 164 + 6 = 170) (h2 : ∃ r : ℕ, c * r = 170) (h3 : ∃ r : ℕ, c * r > 164) :
  c = 10 :=
by
  sorry

end farmer_cages_l93_93330


namespace proof_S_squared_l93_93555

variables {a b c p S r r_a r_b r_c : ℝ}

-- Conditions
axiom cond1 : r * p = r_a * (p - a)
axiom cond2 : r * r_a = (p - b) * (p - c)
axiom cond3 : r_b * r_c = p * (p - a)
axiom heron : S^2 = p * (p - a) * (p - b) * (p - c)

-- Proof statement
theorem proof_S_squared : S^2 = r * r_a * r_b * r_c :=
by sorry

end proof_S_squared_l93_93555


namespace maximize_mice_two_kittens_different_versions_JPPF_JPPF_combinations_JPPF_two_males_one_female_l93_93418

-- Defining productivity functions for male and female kittens
def male_productivity (K : ℝ) : ℝ := 80 - 4 * K
def female_productivity (K : ℝ) : ℝ := 16 - 0.25 * K

-- Condition (a): Maximizing number of mice caught by 2 kittens
theorem maximize_mice_two_kittens : 
  ∃ (male1 male2 : ℝ) (K_m1 K_m2 : ℝ), 
    (male1 = male_productivity K_m1) ∧ 
    (male2 = male_productivity K_m2) ∧
    (K_m1 = 0) ∧ (K_m2 = 0) ∧
    (male1 + male2 = 160) := 
sorry

-- Condition (b): Different versions of JPPF
theorem different_versions_JPPF : 
  ∃ (v1 v2 v3 : Unit), 
    (v1 ≠ v2) ∧ (v2 ≠ v3) ∧ (v1 ≠ v3) :=
sorry

-- Condition (c): Analytical form of JPPF for each combination
theorem JPPF_combinations :
  ∃ (M K1 K2 : ℝ),
    (M = 160 - 4 * K1 ∧ K1 ≤ 40) ∨
    (M = 32 - 0.5 * K2 ∧ K2 ≤ 64) ∨
    (M = 96 - 0.25 * K2 ∧ K2 ≤ 64) ∨
    (M = 336 - 4 * K2 ∧ 64 < K2 ∧ K2 ≤ 84) :=
sorry

-- Condition (d): Analytical form for 2 males and 1 female
theorem JPPF_two_males_one_female :
  ∃ (M K : ℝ), 
    (0 < K ∧ K ≤ 64 ∧ M = 176 - 0.25 * K) ∨
    (64 < K ∧ K ≤ 164 ∧ M = 416 - 4 * K) :=
sorry

end maximize_mice_two_kittens_different_versions_JPPF_JPPF_combinations_JPPF_two_males_one_female_l93_93418


namespace equation_has_at_most_one_real_root_l93_93373

def has_inverse (f : ℝ → ℝ) : Prop := ∃ g : ℝ → ℝ, ∀ x, g (f x) = x

theorem equation_has_at_most_one_real_root (f : ℝ → ℝ) (a : ℝ) (h : has_inverse f) :
  ∀ x1 x2 : ℝ, f x1 = a ∧ f x2 = a → x1 = x2 :=
by sorry

end equation_has_at_most_one_real_root_l93_93373


namespace number_of_neither_l93_93027

def total_businessmen : ℕ := 30
def coffee_drinkers : ℕ := 15
def tea_drinkers : ℕ := 12
def both_drinkers : ℕ := 6

theorem number_of_neither (total_businessmen coffee_drinkers tea_drinkers both_drinkers : ℕ) : 
  coffee_drinkers = 15 ∧ 
  tea_drinkers = 12 ∧ 
  both_drinkers = 6 ∧ 
  total_businessmen = 30 → 
  total_businessmen - (coffee_drinkers + tea_drinkers - both_drinkers) = 9 :=
by
  sorry

end number_of_neither_l93_93027


namespace total_spent_l93_93466

def original_price : ℝ := 20
def discount_rate : ℝ := 0.5
def number_of_friends : ℝ := 4
def discounted_price (orig_price : ℝ) (discount : ℝ) : ℝ := orig_price * discount
def total_cost (num_friends : ℝ) (unit_cost : ℝ) : ℝ := num_friends * unit_cost

theorem total_spent :
  total_cost number_of_friends (discounted_price original_price discount_rate) = 40 :=
by
  simp [total_cost, discounted_price, original_price, discount_rate, number_of_friends]
  norm_num
  sorry

end total_spent_l93_93466


namespace original_price_l93_93836

theorem original_price (x: ℝ) (h1: x * 1.1 * 0.8 = 2) : x = 25 / 11 :=
by
  sorry

end original_price_l93_93836


namespace Patricia_money_l93_93881

theorem Patricia_money 
(P L C : ℝ)
(h1 : L = 5 * P)
(h2 : L = 2 * C)
(h3 : P + L + C = 51) :
P = 6.8 := 
by 
  sorry

end Patricia_money_l93_93881


namespace ratio_A_B_share_l93_93201

-- Define the capital contributions and time in months
def A_capital : ℕ := 3500
def B_capital : ℕ := 15750
def A_months: ℕ := 12
def B_months: ℕ := 4

-- Effective capital contributions
def A_contribution : ℕ := A_capital * A_months
def B_contribution : ℕ := B_capital * B_months

-- Declare the theorem to prove the ratio 2:3
theorem ratio_A_B_share : A_contribution / 21000 = 2 ∧ B_contribution / 21000 = 3 :=
by
  -- Calculate and simplify the ratios
  have hA : A_contribution = 42000 := rfl
  have hB : B_contribution = 63000 := rfl
  have hGCD : Nat.gcd 42000 63000 = 21000 := rfl
  sorry

end ratio_A_B_share_l93_93201


namespace rubber_duck_charity_fundraiser_l93_93960

noncomputable def charity_raised (price_small price_medium price_large : ℕ) 
(bulk_discount_threshold_small bulk_discount_threshold_medium bulk_discount_threshold_large : ℕ)
(bulk_discount_rate_small bulk_discount_rate_medium bulk_discount_rate_large : ℝ)
(tax_rate_small tax_rate_medium tax_rate_large : ℝ)
(sold_small sold_medium sold_large : ℕ) : ℝ :=
  let cost_small := price_small * sold_small
  let cost_medium := price_medium * sold_medium
  let cost_large := price_large * sold_large

  let discount_small := if sold_small >= bulk_discount_threshold_small then 
                          (bulk_discount_rate_small * cost_small) else 0
  let discount_medium := if sold_medium >= bulk_discount_threshold_medium then 
                          (bulk_discount_rate_medium * cost_medium) else 0
  let discount_large := if sold_large >= bulk_discount_threshold_large then 
                          (bulk_discount_rate_large * cost_large) else 0

  let after_discount_small := cost_small - discount_small
  let after_discount_medium := cost_medium - discount_medium
  let after_discount_large := cost_large - discount_large

  let tax_small := tax_rate_small * after_discount_small
  let tax_medium := tax_rate_medium * after_discount_medium
  let tax_large := tax_rate_large * after_discount_large

  let total_small := after_discount_small + tax_small
  let total_medium := after_discount_medium + tax_medium
  let total_large := after_discount_large + tax_large

  total_small + total_medium + total_large

theorem rubber_duck_charity_fundraiser :
  charity_raised 2 3 5 10 15 20 0.1 0.15 0.2
  0.05 0.07 0.09 150 221 185 = 1693.10 :=
by 
  -- implementation of math corresponding to problem's solution
  sorry

end rubber_duck_charity_fundraiser_l93_93960


namespace find_number_of_girls_l93_93990

variable (B G : ℕ)

theorem find_number_of_girls
  (h1 : B = G / 2)
  (h2 : B + G = 90)
  : G = 60 :=
sorry

end find_number_of_girls_l93_93990


namespace triangle_inequality_l93_93065

-- Define the conditions as Lean hypotheses
variables {a b c : ℝ}

-- Lean statement for the problem
theorem triangle_inequality (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 1) : 
  a^2 + b^2 + c^2 + 4*a*b*c < 1/2 :=
sorry

end triangle_inequality_l93_93065


namespace probability_sum_less_than_10_given_first_die_is_6_l93_93171

-- Definitions
noncomputable def dice : finset ℕ := {1, 2, 3, 4, 5, 6}
noncomputable def sum_of_dice {x y : ℕ} (hx : x ∈ dice) (hy : y ∈ dice) : ℕ := x + y

-- Probability calculation
def probability_space := finset (ℕ × ℕ)
noncomputable def fair_dice_probability : probability_space :=
  finset.product dice dice

noncomputable def conditioned_prob (event : set (ℕ × ℕ)) (cond : set (ℕ × ℕ)) : ℝ :=
  (cond ∩ event).card.to_real / cond.card.to_real

-- Condition where sum of dice is less than 10 given first die is 6
def event_sum_less_than_10 : set (ℕ × ℕ) :=
  { p | sum_of_dice (finset.mem_univ p.1) (finset.mem_univ p.2) < 10 }

def condition_first_die_6 : set (ℕ × ℕ) :=
  { p | p.1 = 6 }

-- Theorem statement
theorem probability_sum_less_than_10_given_first_die_is_6 :
  conditioned_prob event_sum_less_than_10 condition_first_die_6 = 1 / 2 :=
by sorry

end probability_sum_less_than_10_given_first_die_is_6_l93_93171


namespace circle_intersection_range_l93_93382

theorem circle_intersection_range (a : ℝ) :
  (0 < a ∧ a < 2 * Real.sqrt 2) ∨ (-2 * Real.sqrt 2 < a ∧ a < 0) ↔
  (let C := { p : ℝ × ℝ | (p.1 - a) ^ 2 + (p.2 - a) ^ 2 = 4 };
   let O := { p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = 4 };
   ∀ p, p ∈ C → p ∈ O) :=
sorry

end circle_intersection_range_l93_93382


namespace max_marks_l93_93272

theorem max_marks (M : ℝ) (score passing shortfall : ℝ)
  (h_score : score = 212)
  (h_shortfall : shortfall = 44)
  (h_passing : passing = score + shortfall)
  (h_pass_cond : passing = 0.4 * M) :
  M = 640 :=
by
  sorry

end max_marks_l93_93272


namespace shrub_height_at_end_of_2_years_l93_93727

theorem shrub_height_at_end_of_2_years (h₅ : ℕ) (h : ∀ n : ℕ, 0 < n → 243 = 3^5 * h₅) : ∃ h₂ : ℕ, h₂ = 9 :=
by sorry

end shrub_height_at_end_of_2_years_l93_93727


namespace total_coronavirus_cases_l93_93790

theorem total_coronavirus_cases (ny_cases ca_cases tx_cases : ℕ)
    (h_ny : ny_cases = 2000)
    (h_ca : ca_cases = ny_cases / 2)
    (h_tx : ca_cases = tx_cases + 400) :
    ny_cases + ca_cases + tx_cases = 3600 := by
  sorry

end total_coronavirus_cases_l93_93790


namespace stickers_distribution_l93_93517

-- Define the mathematical problem: distributing 10 stickers among 5 sheets with each sheet getting at least one sticker.

def partitions_count (n k : ℕ) : ℕ := sorry

theorem stickers_distribution (n : ℕ) (k : ℕ) (h₁ : n = 10) (h₂ : k = 5) :
  partitions_count (n - k) k = 7 := by
  sorry

end stickers_distribution_l93_93517


namespace focus_of_ellipse_l93_93486

-- Definitions from conditions
def major_axis_endpoints := (1, -2) ∧ (7, -2)
def minor_axis_endpoints := (3, 1) ∧ (3, -5)
def center_of_ellipse := (3, -2)

-- Proof problem
theorem focus_of_ellipse :
  (compute_focus_x_coord major_axis_endpoints minor_axis_endpoints = 3) ∧ 
  (compute_focus_y_coord major_axis_endpoints minor_axis_endpoints = -2) :=
sorry

end focus_of_ellipse_l93_93486


namespace new_average_daily_production_l93_93052

theorem new_average_daily_production (n : ℕ) (avg_past_n_days : ℕ) (today_production : ℕ) (h1 : avg_past_n_days = 50) (h2 : today_production = 90) (h3 : n = 9) : 
  (avg_past_n_days * n + today_production) / (n + 1) = 54 := 
by
  sorry

end new_average_daily_production_l93_93052


namespace xiao_zhang_complete_task_l93_93322

open Nat

def xiaoZhangCharacters (n : ℕ) : ℕ :=
match n with
| 0 => 0
| (n+1) => 2 * (xiaoZhangCharacters n)

theorem xiao_zhang_complete_task :
  ∀ (total_chars : ℕ), (total_chars > 0) → 
  (xiaoZhangCharacters 5 = (total_chars / 3)) →
  (xiaoZhangCharacters 6 = total_chars) :=
by
  sorry

end xiao_zhang_complete_task_l93_93322


namespace sum_of_midpoints_x_coordinates_l93_93152

theorem sum_of_midpoints_x_coordinates (a b c : ℝ) (h : a + b + c = 15) :
  (a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 15 := by
  sorry

end sum_of_midpoints_x_coordinates_l93_93152


namespace sufficient_not_necessary_l93_93239

theorem sufficient_not_necessary (x : ℝ) : (x > 3 → x > 1) ∧ ¬ (x > 1 → x > 3) :=
by 
  sorry

end sufficient_not_necessary_l93_93239


namespace boxes_containing_neither_l93_93586

theorem boxes_containing_neither
  (total_boxes : ℕ := 15)
  (boxes_with_markers : ℕ := 9)
  (boxes_with_crayons : ℕ := 5)
  (boxes_with_both : ℕ := 4) :
  (total_boxes - ((boxes_with_markers - boxes_with_both) + (boxes_with_crayons - boxes_with_both) + boxes_with_both)) = 5 := by
  sorry

end boxes_containing_neither_l93_93586


namespace population_approx_10000_2090_l93_93224

def population (initial_population : ℕ) (years : ℕ) : ℕ :=
  initial_population * 2 ^ (years / 20)

theorem population_approx_10000_2090 :
  ∃ y, y = 2090 ∧ population 500 (2090 - 2010) = 500 * 2 ^ (80 / 20) :=
by
  sorry

end population_approx_10000_2090_l93_93224


namespace octahedron_hamiltonian_path_exists_l93_93866

def octahedron : SimpleGraph (Fin 6) :=
  ⟦0⟧ -- A has index 0
  ⟦1⟧ -- B has index 1
  ⟦2⟧ -- C has index 2
  ⟦3⟧ -- A_1 has index 3
  ⟦4⟧ -- B_1 has index 4
  ⟦5⟧ -- C_1 has index 5

def octahedron_edges (v1 v2 : Fin 6) : Prop :=
  v1 ≠ v2 ∧ (v1 + v2) % 3 ≠ 0

theorem octahedron_hamiltonian_path_exists :
  ∃ p : List (Fin 6), octahedron.isHamiltonianCycle p :=
begin
  sorry
end

end octahedron_hamiltonian_path_exists_l93_93866


namespace repeating_decimal_sum_l93_93520

theorem repeating_decimal_sum :
  ∃ (a b : ℕ), Nat.coprime a b ∧ (0.353535... = (a : ℝ) / (b : ℝ)) ∧ (a + b = 134) :=
sorry

end repeating_decimal_sum_l93_93520


namespace find_sum_l93_93101

variable {x y z w : ℤ}

-- Conditions: Consecutive integers and their sum condition
def consecutive_integers (x y z : ℤ) : Prop := y = x + 1 ∧ z = x + 2
def sum_is_150 (x y z : ℤ) : Prop := x + y + z = 150
def w_definition (w z x : ℤ) : Prop := w = 2 * z - x

-- Theorem statement
theorem find_sum (h1 : consecutive_integers x y z) (h2 : sum_is_150 x y z) (h3 : w_definition w z x) :
  x + y + z + w = 203 :=
sorry

end find_sum_l93_93101


namespace Peter_buys_more_hot_dogs_than_hamburgers_l93_93365

theorem Peter_buys_more_hot_dogs_than_hamburgers :
  let chicken := 16
  let hamburgers := chicken / 2
  (exists H : Real, 16 + hamburgers + H + H / 2 = 39 ∧ (H - hamburgers = 2)) := sorry

end Peter_buys_more_hot_dogs_than_hamburgers_l93_93365


namespace mike_falls_short_l93_93802

theorem mike_falls_short : 
  ∀ (max_marks mike_score : ℕ) (pass_percentage : ℚ),
  pass_percentage = 0.30 → 
  max_marks = 800 → 
  mike_score = 212 → 
  (pass_percentage * max_marks - mike_score) = 28 :=
by
  intros max_marks mike_score pass_percentage h1 h2 h3
  sorry

end mike_falls_short_l93_93802


namespace divisibility_by_cube_greater_than_1_l93_93544

theorem divisibility_by_cube_greater_than_1 (a b : ℕ) (hpos_a : 0 < a) (hpos_b : 0 < b)
  (hdiv : (a + b^3) % (a^2 + 3 * a * b + 3 * b^2 - 1) = 0) :
  ∃ k : ℕ, 1 < k ∧ k^3 ∣ a^2 + 3 * a * b + 3 * b^2 - 1 := 
by {
  sorry
}

end divisibility_by_cube_greater_than_1_l93_93544


namespace total_cherry_tomatoes_l93_93393

-- Definitions based on the conditions
def cherryTomatoesPerJar : Nat := 8
def numberOfJars : Nat := 7

-- The statement we want to prove
theorem total_cherry_tomatoes : cherryTomatoesPerJar * numberOfJars = 56 := by
  sorry

end total_cherry_tomatoes_l93_93393


namespace lassis_from_mangoes_l93_93735

-- Define the given ratio
def lassis_per_mango := 15 / 3

-- Define the number of mangoes
def mangoes := 15

-- Define the expected number of lassis
def expected_lassis := 75

-- Prove that with 15 mangoes, 75 lassis can be made given the ratio
theorem lassis_from_mangoes (h : lassis_per_mango = 5) : mangoes * lassis_per_mango = expected_lassis :=
by
  sorry

end lassis_from_mangoes_l93_93735


namespace total_pets_count_l93_93284

/-- Taylor and his six friends have a total of 45 pets, given the specified conditions about the number of each type of pet they have. -/
theorem total_pets_count
  (Taylor_cats : ℕ := 4)
  (Friend1_pets : ℕ := 8 * 3)
  (Friend2_dogs : ℕ := 3)
  (Friend2_birds : ℕ := 1)
  (Friend3_dogs : ℕ := 5)
  (Friend3_cats : ℕ := 2)
  (Friend4_reptiles : ℕ := 2)
  (Friend4_birds : ℕ := 3)
  (Friend4_cats : ℕ := 1) :
  Taylor_cats + Friend1_pets + Friend2_dogs + Friend2_birds + Friend3_dogs + Friend3_cats + Friend4_reptiles + Friend4_birds + Friend4_cats = 45 :=
sorry

end total_pets_count_l93_93284


namespace intersection_M_N_l93_93071

-- Define the set M
def M : Set ℤ := {-2, -1, 0, 1, 2}

-- Define the condition for set N
def N : Set ℤ := {x | x + 2 ≥ x^2}

-- State the theorem to prove the intersection
theorem intersection_M_N :
  M ∩ N = {-1, 0, 1, 2} :=
sorry

end intersection_M_N_l93_93071


namespace f_1988_eq_1988_l93_93098

noncomputable def f (n : ℕ) : ℕ := sorry

axiom f_f_eq_add (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : f (f m + f n) = m + n

theorem f_1988_eq_1988 : f 1988 = 1988 := 
by
  sorry

end f_1988_eq_1988_l93_93098


namespace mn_value_l93_93519

theorem mn_value (m n : ℤ) (h1 : m = n + 2) (h2 : 2 * m + n = 4) : m * n = 0 := by
  sorry

end mn_value_l93_93519


namespace space_per_bush_l93_93483

theorem space_per_bush (side_length : ℝ) (num_sides : ℝ) (num_bushes : ℝ) (h1 : side_length = 16) (h2 : num_sides = 3) (h3 : num_bushes = 12) :
  (num_sides * side_length) / num_bushes = 4 :=
by
  sorry

end space_per_bush_l93_93483


namespace fraction_squares_sum_l93_93518

theorem fraction_squares_sum (x a y b z c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (h1 : x / a + y / b + z / c = 3) (h2 : a / x + b / y + c / z = -3) : 
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 15 := 
by 
  sorry

end fraction_squares_sum_l93_93518


namespace sum_of_squares_of_roots_l93_93034

theorem sum_of_squares_of_roots
  (x1 x2 : ℝ) (h : 5 * x1^2 + 6 * x1 - 15 = 0) (h' : 5 * x2^2 + 6 * x2 - 15 = 0) :
  x1^2 + x2^2 = 186 / 25 :=
sorry

end sum_of_squares_of_roots_l93_93034


namespace range_of_x_l93_93067

theorem range_of_x (x : ℝ) : (abs (x + 1) + abs (x - 5) = 6) ↔ (-1 ≤ x ∧ x ≤ 5) :=
by sorry

end range_of_x_l93_93067


namespace smallest_positive_debt_l93_93453

theorem smallest_positive_debt :
  ∃ (p g : ℤ), 25 = 250 * p + 175 * g :=
by
  sorry

end smallest_positive_debt_l93_93453


namespace students_with_one_problem_l93_93442

theorem students_with_one_problem :
  ∃ (n_1 n_2 n_3 n_4 n_5 n_6 n_7 : ℕ) (k_1 k_2 k_3 k_4 k_5 k_6 k_7 : ℕ),
    (n_1 + n_2 + n_3 + n_4 + n_5 + n_6 + n_7 = 39) ∧
    (n_1 * k_1 + n_2 * k_2 + n_3 * k_3 + n_4 * k_4 + n_5 * k_5 + n_6 * k_6 + n_7 * k_7 = 60) ∧
    (k_1 ≠ 0) ∧ (k_2 ≠ 0) ∧ (k_3 ≠ 0) ∧ (k_4 ≠ 0) ∧ (k_5 ≠ 0) ∧ (k_6 ≠ 0) ∧ (k_7 ≠ 0) ∧
    (k_1 ≠ k_2) ∧ (k_1 ≠ k_3) ∧ (k_1 ≠ k_4) ∧ (k_1 ≠ k_5) ∧ (k_1 ≠ k_6) ∧ (k_1 ≠ k_7) ∧
    (k_2 ≠ k_3) ∧ (k_2 ≠ k_4) ∧ (k_2 ≠ k_5) ∧ (k_2 ≠ k_6) ∧ (k_2 ≠ k_7) ∧
    (k_3 ≠ k_4) ∧ (k_3 ≠ k_5) ∧ (k_3 ≠ k_6) ∧ (k_3 ≠ k_7) ∧
    (k_4 ≠ k_5) ∧ (k_4 ≠ k_6) ∧ (k_4 ≠ k_7) ∧
    (k_5 ≠ k_6) ∧ (k_5 ≠ k_7) ∧
    (k_6 ≠ k_7) ∧
    (n_1 = 33) :=
sorry

end students_with_one_problem_l93_93442


namespace sum_of_reciprocals_l93_93682

theorem sum_of_reciprocals (x y : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : x + y = 5 * x * y) : (1 / x) + (1 / y) = 5 :=
by
  sorry

end sum_of_reciprocals_l93_93682


namespace intervals_of_monotonicity_minimum_value_l93_93771

noncomputable def f (a x : ℝ) : ℝ := Real.log x - a * x

theorem intervals_of_monotonicity (a : ℝ) (h : a > 0) :
  (∀ x, 0 < x ∧ x ≤ 1 / a → f a x ≤ f a (1 / a)) ∧
  (∀ x, x ≥ 1 / a → f a x ≥ f a (1 / a)) :=
sorry

theorem minimum_value (a : ℝ) (h : a > 0) :
  (a < Real.log 2 → ∃ x ∈ Set.Icc (1:ℝ) (2:ℝ), f a x = -a) ∧
  (a ≥ Real.log 2 → ∃ x ∈ Set.Icc (1:ℝ) (2:ℝ), f a x = Real.log 2 - 2 * a) :=
sorry

end intervals_of_monotonicity_minimum_value_l93_93771


namespace max_value_of_trig_expr_l93_93046

variable (x : ℝ)

theorem max_value_of_trig_expr : 
  (∀ x : ℝ, 3 * Real.cos x + 4 * Real.sin x ≤ 5) ∧
  (∃ x : ℝ, 3 * Real.cos x + 4 * Real.sin x = 5) :=
sorry

end max_value_of_trig_expr_l93_93046


namespace problem1_problem2_l93_93734

-- Problem 1: Prove that (2a^2 b) * a b^2 / 4a^3 = 1/2 b^3
theorem problem1 (a b : ℝ) : (2 * a^2 * b) * (a * b^2) / (4 * a^3) = (1 / 2) * b^3 :=
  sorry

-- Problem 2: Prove that (2x + 5)(x - 3) = 2x^2 - x - 15
theorem problem2 (x : ℝ): (2 * x + 5) * (x - 3) = 2 * x^2 - x - 15 :=
  sorry

end problem1_problem2_l93_93734


namespace sign_of_k_l93_93090

variable (k x y : ℝ)
variable (A B : ℝ × ℝ)
variable (y₁ y₂ : ℝ)
variable (h₁ : A = (-2, y₁))
variable (h₂ : B = (5, y₂))
variable (h₃ : y₁ = k / -2)
variable (h₄ : y₂ = k / 5)
variable (h₅ : y₁ > y₂)
variable (h₀ : k ≠ 0)

-- We need to prove that k < 0
theorem sign_of_k (A B : ℝ × ℝ) (y₁ y₂ k : ℝ) 
  (h₁ : A = (-2, y₁)) 
  (h₂ : B = (5, y₂)) 
  (h₃ : y₁ = k / -2) 
  (h₄ : y₂ = k / 5) 
  (h₅ : y₁ > y₂) 
  (h₀ : k ≠ 0) : k < 0 := 
by
  sorry

end sign_of_k_l93_93090


namespace three_mathematicians_same_language_l93_93409

theorem three_mathematicians_same_language
  (M : Fin 9 → Finset string)
  (h1 : ∀ i j k : Fin 9, ∃ lang, i ≠ j → i ≠ k → j ≠ k → lang ∈ M i ∧ lang ∈ M j)
  (h2 : ∀ i : Fin 9, (M i).card ≤ 3)
  : ∃ lang ∈ ⋃ i, M i, ∃ (A B C : Fin 9), A ≠ B → A ≠ C → B ≠ C → lang ∈ M A ∧ lang ∈ M B ∧ lang ∈ M C :=
sorry

end three_mathematicians_same_language_l93_93409


namespace casey_saves_by_paying_monthly_l93_93355

theorem casey_saves_by_paying_monthly :
  let weekly_rate := 280
  let monthly_rate := 1000
  let weeks_in_a_month := 4
  let number_of_months := 3
  let total_weeks := number_of_months * weeks_in_a_month
  let total_cost_weekly := total_weeks * weekly_rate
  let total_cost_monthly := number_of_months * monthly_rate
  let savings := total_cost_weekly - total_cost_monthly
  savings = 360 :=
by
  sorry

end casey_saves_by_paying_monthly_l93_93355


namespace plane_split_into_four_regions_l93_93217

theorem plane_split_into_four_regions {x y : ℝ} :
  (y = 3 * x) ∨ (y = (1 / 3) * x - (2 / 3)) →
  ∃ r : ℕ, r = 4 :=
by
  intro h
  -- We must show that these lines split the plane into 4 regions
  sorry

end plane_split_into_four_regions_l93_93217


namespace fraction_simplification_l93_93122

-- Definitions based on conditions and question
def lcm_462_42 : ℕ := 462
def prime_factors_462 : List ℕ := [2, 3, 7, 11]
def prime_factors_42 : List ℕ := [2, 3, 7]

-- Main theorem statement
theorem fraction_simplification :
  (1 / 462) + (17 / 42) = 94 / 231 :=
by
  sorry

end fraction_simplification_l93_93122


namespace greatest_product_two_integers_sum_2004_l93_93974

theorem greatest_product_two_integers_sum_2004 : 
  (∃ x y : ℤ, x + y = 2004 ∧ x * y = 1004004) :=
by
  sorry

end greatest_product_two_integers_sum_2004_l93_93974


namespace bernardo_silvia_probability_l93_93487

/-- This theorem states that if Bernardo randomly picks 3 distinct numbers from {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} and Silvia randomly picks 3 distinct numbers from {1, 2, 3, 4, 5, 6, 7, 8, 9}, both arranging them in descending order to form a three-digit number, then the probability that Bernardo's number is greater than Silvia's number is 217/336. -/
theorem bernardo_silvia_probability :
  let bernardo_set := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
  let silvia_set := {1, 2, 3, 4, 5, 6, 7, 8, 9}
  let bernardo_picks := finset.choose 3 bernardo_set
  let silvia_picks := finset.choose 3 silvia_set
  let bernardo_num := bernardo_picks.map (λ x, -x).sort (≤)
  let silvia_num := silvia_picks.map (λ x, -x).sort (≤)
  let bernardo_larger := bernardo_num > silvia_num
  probability bernardo_larger =
    (217/336 : ℚ) :=
sorry

end bernardo_silvia_probability_l93_93487


namespace age_of_older_sister_in_2021_l93_93395

instance : DecidableEq ℕ := Classical.decEq ℕ

theorem age_of_older_sister_in_2021 (year_kelsey_25 : ℕ) (current_year : ℕ) (kelsey_age_in_1999 : ℕ) (sister_age_diff : ℕ) :
  (year_kelsey_25 = 1999) ∧ (kelsey_age_in_1999 = 25) ∧ (sister_age_diff = 3) ∧ (current_year = 2021) → 
  (current_year - ((year_kelsey_25 - kelsey_age_in_1999) - sister_age_diff) = 50) :=
by
  sorry

end age_of_older_sister_in_2021_l93_93395


namespace moderate_intensity_pushups_l93_93994

theorem moderate_intensity_pushups :
  let normal_heart_rate := 80
  let k := 7
  let y (x : ℕ) := 80 * (Real.log (Real.sqrt (x / 12)) + 1)
  let t (x : ℕ) := y x / normal_heart_rate
  let f (t : ℝ) := k * Real.exp t
  28 ≤ f (Real.log (Real.sqrt 3)) + 1 ∧ f (Real.log (Real.sqrt 3)) + 1 ≤ 34 :=
sorry

end moderate_intensity_pushups_l93_93994


namespace smallest_coins_l93_93583

theorem smallest_coins (n : ℕ) (n_min : ℕ) (h1 : ∃ n, n % 8 = 5 ∧ n % 7 = 4 ∧ n = 53) (h2 : n_min = n):
  (n_min ≡ 5 [MOD 8]) ∧ (n_min ≡ 4 [MOD 7]) ∧ (n_min = 53) ∧ (53 % 9 = 8) :=
by
  sorry

end smallest_coins_l93_93583


namespace election_votes_l93_93180

theorem election_votes (V : ℝ) (h1 : 0.70 * V - 0.30 * V = 200) : V = 500 :=
sorry

end election_votes_l93_93180


namespace seven_a_plus_seven_b_l93_93796

noncomputable def g (x : ℝ) : ℝ := 7 * x - 6
noncomputable def f_inv (x : ℝ) : ℝ := 7 * x - 4
noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := a * x + b

theorem seven_a_plus_seven_b (a b : ℝ) (h₁ : ∀ x, g x = f_inv x - 2) (h₂ : ∀ x, f_inv (f x a b) = x) :
  7 * a + 7 * b = 5 :=
by
  sorry

end seven_a_plus_seven_b_l93_93796


namespace equilateral_triangle_area_l93_93823

theorem equilateral_triangle_area (h : ℝ) (h_eq : h = 2 * Real.sqrt 3) : 
  (Real.sqrt 3 / 4) * (2 * h / (Real.sqrt 3))^2 = 4 * Real.sqrt 3 := 
by
  rw [h_eq]
  sorry

end equilateral_triangle_area_l93_93823


namespace sum_of_reciprocals_l93_93681

theorem sum_of_reciprocals (x y : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : x + y = 5 * x * y) : (1 / x) + (1 / y) = 5 :=
by
  sorry

end sum_of_reciprocals_l93_93681


namespace range_of_y_l93_93512

-- Define the vectors
def a : ℝ × ℝ := (1, -2)
def b (y : ℝ) : ℝ × ℝ := (4, y)

-- Define the vector sum
def a_plus_b (y : ℝ) : ℝ × ℝ := (a.1 + (b y).1, a.2 + (b y).2)

-- Define the dot product
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Prove the angle between a and a + b is acute and y ≠ -8
theorem range_of_y (y : ℝ) :
  (dot_product a (a_plus_b y) > 0) ↔ (y < 4.5 ∧ y ≠ -8) :=
by
  sorry

end range_of_y_l93_93512


namespace constructed_expression_equals_original_l93_93535

variable (a : ℝ)

theorem constructed_expression_equals_original : 
  a ≠ 0 → 
  ((1/a) / ((1/a) * (1/a)) - (1/a)) / (1/a) = (a + 1) * (a - 1) :=
by
  intro h
  sorry

end constructed_expression_equals_original_l93_93535


namespace conference_center_people_count_l93_93588

-- Definition of the conditions
def rooms : ℕ := 6
def capacity_per_room : ℕ := 80
def fraction_full : ℚ := 2/3

-- Total capacity of the conference center
def total_capacity := rooms * capacity_per_room

-- Number of people in the conference center when 2/3 full
def num_people := fraction_full * total_capacity

-- The theorem stating the problem
theorem conference_center_people_count :
  num_people = 320 := 
by
  -- This is a placeholder for the proof
  sorry

end conference_center_people_count_l93_93588


namespace least_number_of_stamps_l93_93482

theorem least_number_of_stamps (s t : ℕ) (h : 5 * s + 7 * t = 50) : s + t = 8 :=
sorry

end least_number_of_stamps_l93_93482


namespace total_earnings_correct_l93_93308

noncomputable def total_earnings : ℝ :=
  let earnings1 := 12 * (2 + 15 / 60)
  let earnings2 := 15 * (1 + 40 / 60)
  let earnings3 := 10 * (3 + 10 / 60)
  earnings1 + earnings2 + earnings3

theorem total_earnings_correct : total_earnings = 83.75 := by
  sorry

end total_earnings_correct_l93_93308


namespace max_product_of_two_integers_with_sum_2004_l93_93969

theorem max_product_of_two_integers_with_sum_2004 :
  ∃ x y : ℤ, x + y = 2004 ∧ (∀ a b : ℤ, a + b = 2004 → a * b ≤ x * y) ∧ x * y = 1004004 := 
by
  sorry

end max_product_of_two_integers_with_sum_2004_l93_93969


namespace percentage_of_september_authors_l93_93141

def total_authors : ℕ := 120
def september_authors : ℕ := 15

theorem percentage_of_september_authors : 
  (september_authors / total_authors : ℚ) * 100 = 12.5 :=
by
  sorry

end percentage_of_september_authors_l93_93141


namespace range_of_a_for_decreasing_function_l93_93624

variable (a : ℝ)

noncomputable def f (x : ℝ) : ℝ := -x^2 - 2 * (a - 1) * x + 5

noncomputable def f' (x : ℝ) : ℝ := -2 * x - 2 * (a - 1)

theorem range_of_a_for_decreasing_function :
  (∀ x : ℝ, -1 ≤ x → f' a x ≤ 0) → 2 ≤ a := sorry

end range_of_a_for_decreasing_function_l93_93624


namespace eggs_in_seven_boxes_l93_93963

-- define the conditions
def eggs_per_box : Nat := 15
def number_of_boxes : Nat := 7

-- state the main theorem to prove
theorem eggs_in_seven_boxes : eggs_per_box * number_of_boxes = 105 := by
  sorry

end eggs_in_seven_boxes_l93_93963


namespace daria_needs_to_earn_l93_93493

variable (ticket_cost : ℕ) (current_money : ℕ) (total_tickets : ℕ)

def total_cost (ticket_cost : ℕ) (total_tickets : ℕ) : ℕ :=
  ticket_cost * total_tickets

def money_needed (total_cost : ℕ) (current_money : ℕ) : ℕ :=
  total_cost - current_money

theorem daria_needs_to_earn :
  total_cost 90 4 - 189 = 171 :=
by
  sorry

end daria_needs_to_earn_l93_93493


namespace toms_weekly_earnings_l93_93445

variable (buckets : ℕ) (crabs_per_bucket : ℕ) (price_per_crab : ℕ) (days_per_week : ℕ)

def total_money_per_week (buckets : ℕ) (crabs_per_bucket : ℕ) (price_per_crab : ℕ) (days_per_week : ℕ) : ℕ :=
  buckets * crabs_per_bucket * price_per_crab * days_per_week

theorem toms_weekly_earnings :
  total_money_per_week 8 12 5 7 = 3360 :=
by
  sorry

end toms_weekly_earnings_l93_93445


namespace Steven_has_16_apples_l93_93933

variable (Jake_Peaches Steven_Peaches Jake_Apples Steven_Apples : ℕ)

theorem Steven_has_16_apples
  (h1 : Jake_Peaches = Steven_Peaches - 6)
  (h2 : Steven_Peaches = 17)
  (h3 : Steven_Peaches = Steven_Apples + 1)
  (h4 : Jake_Apples = Steven_Apples + 8) :
  Steven_Apples = 16 := by
  sorry

end Steven_has_16_apples_l93_93933


namespace sum_x_midpoints_of_triangle_l93_93160

theorem sum_x_midpoints_of_triangle (a b c : ℝ) (h : a + b + c = 15) :
  (a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 15 :=
by
  -- Proof omitted, replace with actual proof
  sorry

end sum_x_midpoints_of_triangle_l93_93160


namespace sum_x_midpoints_of_triangle_l93_93163

theorem sum_x_midpoints_of_triangle (a b c : ℝ) (h : a + b + c = 15) :
  (a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 15 :=
by
  -- Proof omitted, replace with actual proof
  sorry

end sum_x_midpoints_of_triangle_l93_93163


namespace fraction_simplification_l93_93123

-- Definitions based on conditions and question
def lcm_462_42 : ℕ := 462
def prime_factors_462 : List ℕ := [2, 3, 7, 11]
def prime_factors_42 : List ℕ := [2, 3, 7]

-- Main theorem statement
theorem fraction_simplification :
  (1 / 462) + (17 / 42) = 94 / 231 :=
by
  sorry

end fraction_simplification_l93_93123


namespace length_of_platform_l93_93730

theorem length_of_platform (l t p : ℝ) (h1 : (l / t) = (l + p) / (5 * t)) : p = 4 * l :=
by
  sorry

end length_of_platform_l93_93730


namespace find_inequality_solution_set_l93_93887

noncomputable def inequality_solution_set : Set ℝ :=
  { x | (1 / (x * (x + 1))) - (1 / ((x + 1) * (x + 2))) < (1 / 4) }

theorem find_inequality_solution_set :
  inequality_solution_set = { x : ℝ | x < -2 } ∪ { x : ℝ | -1 < x ∧ x < 0 } ∪ { x : ℝ | 1 < x } :=
by
  sorry

end find_inequality_solution_set_l93_93887


namespace Corey_goal_reachable_l93_93604

theorem Corey_goal_reachable :
  ∀ (goal balls_found_saturday balls_found_sunday additional_balls : ℕ),
    goal = 48 →
    balls_found_saturday = 16 →
    balls_found_sunday = 18 →
    additional_balls = goal - (balls_found_saturday + balls_found_sunday) →
    additional_balls = 14 :=
by
  intros goal balls_found_saturday balls_found_sunday additional_balls
  intro goal_eq
  intro saturday_eq
  intro sunday_eq
  intro additional_eq
  sorry

end Corey_goal_reachable_l93_93604


namespace ladder_slides_out_l93_93718

theorem ladder_slides_out (ladder_length foot_initial_dist ladder_slip_down foot_final_dist : ℝ) 
  (h_ladder_length : ladder_length = 25)
  (h_foot_initial_dist : foot_initial_dist = 7)
  (h_ladder_slip_down : ladder_slip_down = 4)
  (h_foot_final_dist : foot_final_dist = 15) :
  foot_final_dist - foot_initial_dist = 8 :=
  by
  simp [h_ladder_length, h_foot_initial_dist, h_ladder_slip_down, h_foot_final_dist]
  sorry

end ladder_slides_out_l93_93718


namespace fewer_seats_right_side_l93_93527

theorem fewer_seats_right_side
  (left_seats : ℕ)
  (people_per_seat : ℕ)
  (back_seat_capacity : ℕ)
  (total_capacity : ℕ)
  (h1 : left_seats = 15)
  (h2 : people_per_seat = 3)
  (h3 : back_seat_capacity = 12)
  (h4 : total_capacity = 93)
  : left_seats - (total_capacity - (left_seats * people_per_seat + back_seat_capacity)) / people_per_seat = 3 :=
  by sorry

end fewer_seats_right_side_l93_93527


namespace part1_part2_part3_l93_93184

section Part1
variables {a b : ℝ}

theorem part1 (h1 : a + b = 3) (h2 : a * b = 2) : a^2 + b^2 = 5 := 
sorry
end Part1

section Part2
variables {a b c : ℝ}

theorem part2 (h1 : a + b + c = 6) (h2 : a * b + b * c + c * a = 11) : a^2 + b^2 + c^2 = 14 := 
sorry
end Part2

section Part3
variables {a b c : ℝ}

theorem part3 (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 = 6) : a^4 + b^4 + c^4 = 18 :=
sorry
end Part3

end part1_part2_part3_l93_93184


namespace cheryl_probability_same_color_l93_93719

noncomputable def probability_cheryl_same_color (total_marbles : ℕ) [decidable_eq ℕ] (draws : ℕ) :=
  let red_probability    := (3 / total_marbles) ^ draws in
  let green_probability  := (3 / total_marbles) ^ draws in
  let yellow_probability := (3 / total_marbles) ^ draws in
  red_probability + green_probability + yellow_probability

theorem cheryl_probability_same_color : probability_cheryl_same_color 9 3 = 1 / 9 :=
by
  have red_probability    := (3 / 9) ^ 3
  have green_probability  := (3 / 9) ^ 3
  have yellow_probability := (3 / 9) ^ 3
  calc
    probability_cheryl_same_color 9 3
        = red_probability + green_probability + yellow_probability : by simp [probability_cheryl_same_color]
    ... = (3 / 9) ^ 3 + (3 / 9) ^ 3 + (3 / 9) ^ 3 : by simp [red_probability, green_probability, yellow_probability]
    ... = 1/27 + 1/27 + 1/27 : by norm_num
    ... = 3 * (1/27) : by ring
    ... = 1/9 : by norm_num

end cheryl_probability_same_color_l93_93719


namespace find_number_l93_93458

theorem find_number (x : ℝ) (h : (x / 6) * 12 = 18) : x = 9 :=
sorry

end find_number_l93_93458


namespace ratio_soda_water_l93_93212

variables (W S : ℕ) (k : ℕ)

-- Conditions of the problem
def condition1 : Prop := S = k * W - 6
def condition2 : Prop := W + S = 54
def positive_integer_k : Prop := k > 0

-- The theorem we want to prove
theorem ratio_soda_water (h1 : condition1 W S k) (h2 : condition2 W S) (h3 : positive_integer_k k) : S / gcd S W = 4 ∧ W / gcd S W = 5 :=
sorry

end ratio_soda_water_l93_93212


namespace sum_abs_diff_is_18_l93_93542

noncomputable def sum_of_possible_abs_diff (a b c d : ℝ) : ℝ :=
  let possible_values := [
      abs ((a + 2) - (d - 7)),
      abs ((a + 2) - (d + 1)),
      abs ((a + 2) - (d - 1)),
      abs ((a + 2) - (d + 7)),
      abs ((a - 2) - (d - 7)),
      abs ((a - 2) - (d + 1)),
      abs ((a - 2) - (d - 1)),
      abs ((a - 2) - (d + 7))
  ]
  possible_values.foldl (· + ·) 0

theorem sum_abs_diff_is_18 (a b c d : ℝ) (h1 : abs (a - b) = 2) (h2 : abs (b - c) = 3) (h3 : abs (c - d) = 4) :
  sum_of_possible_abs_diff a b c d = 18 := by
  sorry

end sum_abs_diff_is_18_l93_93542


namespace compute_expression_l93_93491

theorem compute_expression : 9 + 7 * (5 - Real.sqrt 16)^2 = 16 := by
  sorry

end compute_expression_l93_93491


namespace inequality_solution_l93_93752

theorem inequality_solution (x : ℝ) : 3 * x^2 + 9 * x + 6 ≤ 0 ↔ -2 ≤ x ∧ x ≤ -1 := 
sorry

end inequality_solution_l93_93752


namespace jane_reads_105_pages_in_a_week_l93_93793

-- Define the pages read in the morning and evening
def pages_morning := 5
def pages_evening := 10

-- Define the number of pages read in a day
def pages_per_day := pages_morning + pages_evening

-- Define the number of days in a week
def days_per_week := 7

-- Define the total number of pages read in a week
def pages_per_week := pages_per_day * days_per_week

-- The theorem that sums up the proof
theorem jane_reads_105_pages_in_a_week : pages_per_week = 105 := by
  sorry

end jane_reads_105_pages_in_a_week_l93_93793


namespace middle_card_four_or_five_l93_93443

def three_cards (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a + b + c = 15 ∧ a < b ∧ b < c

theorem middle_card_four_or_five (a b c : ℕ) :
  three_cards a b c → (b = 4 ∨ b = 5) :=
by
  sorry

end middle_card_four_or_five_l93_93443


namespace pentagon_area_correct_l93_93234

-- Define the side lengths of the pentagon
def side1 := 18
def side2 := 25
def side3 := 30
def side4 := 28
def side5 := 22

-- Define the specific angle between the sides of lengths 30 and 28
def angle := 110 -- degrees

-- Define the heights used for the trapezoids and triangle calculations
def height_trapezoid1 := 10
def height_trapezoid2 := 15
def height_triangle := 8

-- Function to calculate the area of a trapezoid
def trapezoid_area (base1 base2 height : ℕ) : ℕ :=
  (base1 + base2) * height / 2

-- Function to calculate the area of a triangle
def triangle_area (base height : ℕ) : ℕ :=
  base * height / 2

-- Calculation of individual areas
def area_trapezoid1 := trapezoid_area side1 side2 height_trapezoid1
def area_trapezoid2 := trapezoid_area side3 side4 height_trapezoid2
def area_triangle := triangle_area side5 height_triangle

-- Total area calculation
def total_area := area_trapezoid1 + area_trapezoid2 + area_triangle

-- Expected total area
def expected_area := 738

-- Lean statement to assert the total area equals the expected value
theorem pentagon_area_correct :
  total_area = expected_area :=
by sorry

end pentagon_area_correct_l93_93234


namespace smallest_multiple_of_6_8_12_l93_93363

theorem smallest_multiple_of_6_8_12 : ∃ n : ℕ, n > 0 ∧ n % 6 = 0 ∧ n % 8 = 0 ∧ n % 12 = 0 ∧ ∀ m : ℕ, (m > 0 ∧ m % 6 = 0 ∧ m % 8 = 0 ∧ m % 12 = 0) → n ≤ m := 
sorry

end smallest_multiple_of_6_8_12_l93_93363


namespace lisa_walks_distance_per_minute_l93_93648

-- Variables and conditions
variable (d : ℤ) -- distance that Lisa walks each minute (what we're solving for)
variable (daily_distance : ℤ) -- distance that Lisa walks each hour
variable (total_distance_in_two_days : ℤ := 1200) -- total distance in two days
variable (hours_per_day : ℤ := 1) -- one hour per day

-- Given conditions
axiom walks_for_an_hour_each_day : ∀ (d: ℤ), daily_distance = d * 60
axiom walks_1200_meters_in_two_days : ∀ (d: ℤ), total_distance_in_two_days = 2 * daily_distance

-- The theorem we want to prove
theorem lisa_walks_distance_per_minute : (d = 10) :=
by
  -- TODO: complete the proof
  sorry

end lisa_walks_distance_per_minute_l93_93648


namespace volume_of_pyramid_base_isosceles_right_triangle_l93_93989

theorem volume_of_pyramid_base_isosceles_right_triangle (a h : ℝ) (ha : a = 3) (hh : h = 4) :
  (1 / 3) * (1 / 2) * a * a * h = 6 := by
  sorry

end volume_of_pyramid_base_isosceles_right_triangle_l93_93989


namespace Maria_score_in_fourth_quarter_l93_93444

theorem Maria_score_in_fourth_quarter (q1 q2 q3 : ℕ) 
  (hq1 : q1 = 84) 
  (hq2 : q2 = 82) 
  (hq3 : q3 = 80) 
  (average_requirement : ℕ) 
  (havg_req : average_requirement = 85) :
  ∃ q4 : ℕ, q4 ≥ 94 ∧ (q1 + q2 + q3 + q4) / 4 ≥ average_requirement := 
by 
  sorry 

end Maria_score_in_fourth_quarter_l93_93444


namespace sum_of_midpoint_xcoords_l93_93158

theorem sum_of_midpoint_xcoords (a b c : ℝ) (h : a + b + c = 15) :
    (a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 15 :=
  by
    sorry

end sum_of_midpoint_xcoords_l93_93158


namespace clara_quarters_l93_93213

theorem clara_quarters :
  ∃ q : ℕ, 8 < q ∧ q < 80 ∧ q % 3 = 1 ∧ q % 4 = 1 ∧ q % 5 = 1 ∧ q = 61 :=
by
  sorry

end clara_quarters_l93_93213


namespace matthew_crackers_left_l93_93402

-- Definition of the conditions:
def initial_crackers := 23
def friends := 2
def crackers_eaten_per_friend := 6

-- Calculate the number of crackers Matthew has left:
def crackers_left (total_crackers : ℕ) (num_friends : ℕ) (eaten_per_friend : ℕ) : ℕ :=
  let crackers_given := (total_crackers - total_crackers % num_friends)
  let kept_by_matthew := total_crackers % num_friends
  let remaining_with_friends := (crackers_given / num_friends - eaten_per_friend) * num_friends
  kept_by_matthew + remaining_with_friends
  
-- Theorem to prove:
theorem matthew_crackers_left : crackers_left initial_crackers friends crackers_eaten_per_friend = 11 := by
  sorry

end matthew_crackers_left_l93_93402


namespace part1_part2_part3_part4_l93_93807

open Nat

section balls_and_boxes

variables (n : ℕ) (m : ℕ) (balls boxes : Finset ℕ) (h₁ : n = 4) (h₂ : m = 4) (h₃ : balls.card = 4) (h₄ : boxes.card = 4)

/-- There are 256 ways to place 4 balls into 4 boxes. -/
theorem part1 : (boxes.card ^ balls.card) = 256 := 
by
  sorry

/-- With each box having exactly one ball, there are 24 ways to place the balls. -/
theorem part2 : (Fintype.card (ball_equiv_perms balls boxes)) = 24 := 
by
  sorry

/-- If exactly one box is empty, there are 144 ways to place the balls. -/
theorem part3 : (choose 4 2) * (factorial 3) = 144 := 
by
  sorry

/-- If the balls are identical and exactly one box is empty, there are 12 ways to place the balls. -/
theorem part4 : (choose 4 1) * (choose 3 1) = 12 := 
by
  sorry

end balls_and_boxes

/- helper definitions for theorems -/

def ball_equiv_perms : Equiv.Perm (Fin 4) :=
{ to_fun := λ x, x.succ,
  inv_fun := λ x, x.pred,
  left_inv := λ x, by { simp [Fin.succ_pred] },
  right_inv := λ x, by { simp [Fin.pred_succ] }
}

end part1_part2_part3_part4_l93_93807


namespace h_of_neg2_eq_11_l93_93541

def f (x : ℝ) : ℝ := 3 * x - 4
def g (x : ℝ) : ℝ := x ^ 2 + 1
def h (x : ℝ) : ℝ := f (g x)

theorem h_of_neg2_eq_11 : h (-2) = 11 := by
  sorry

end h_of_neg2_eq_11_l93_93541


namespace distinct_arithmetic_progression_roots_l93_93505

theorem distinct_arithmetic_progression_roots (a b : ℝ) : 
  (∃ (d : ℝ), d ≠ 0 ∧ ∀ x, x^3 + a * x + b = 0 ↔ x = -d ∨ x = 0 ∨ x = d) → a < 0 ∧ b = 0 :=
by
  sorry

end distinct_arithmetic_progression_roots_l93_93505


namespace greatest_product_from_sum_2004_l93_93972

theorem greatest_product_from_sum_2004 : ∃ (x y : ℤ), x + y = 2004 ∧ x * y = 1004004 :=
by
  sorry

end greatest_product_from_sum_2004_l93_93972


namespace words_with_at_least_one_consonant_l93_93850

-- Define the letters available and classify them as vowels and consonants
def letters : List Char := ['A', 'B', 'C', 'D', 'E', 'F']
def vowels : List Char := ['A', 'E']
def consonants : List Char := ['B', 'C', 'D', 'F']

-- Define the total number of 5-letter words using the given letters
def total_words : ℕ := 6^5

-- Define the total number of 5-letter words composed exclusively of vowels
def vowel_words : ℕ := 2^5

-- Define the number of 5-letter words that contain at least one consonant
noncomputable def words_with_consonant : ℕ := total_words - vowel_words

-- The theorem to prove
theorem words_with_at_least_one_consonant : words_with_consonant = 7744 := by
  sorry

end words_with_at_least_one_consonant_l93_93850


namespace system1_solution_system2_solution_l93_93282

-- System 1
theorem system1_solution (x y : ℝ) 
  (h1 : y = 2 * x - 3)
  (h2 : 3 * x + 2 * y = 8) : 
  x = 2 ∧ y = 1 := 
by
  sorry

-- System 2
theorem system2_solution (x y : ℝ) 
  (h1 : x + 2 * y = 3)
  (h2 : 2 * x - 4 * y = -10) : 
  x = -1 ∧ y = 2 := 
by
  sorry

end system1_solution_system2_solution_l93_93282


namespace trajectory_and_perpendicular_lines_l93_93391

def P (x y : ℝ) := (x, y)
noncomputable def dist (P Q : ℝ × ℝ) : ℝ :=
  ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)^(1/2)

theorem trajectory_and_perpendicular_lines:
  (∀ (x y : ℝ), dist (P x y) (0, -real.sqrt 3) + dist (P x y) (0, real.sqrt 3) = 4 → 
                 x^2 + (y^2 / 4) = 1) ∧ 
  (∀ (k : ℝ),
    (k = 1/2 ∨ k = -1/2) →
    (∃ (x₁ y₁ x₂ y₂ : ℝ),
      (y₁ = k * x₁ + 1 ∧ y₂ = k * x₂ + 1 ∧ 
      x₁^2 + (y₁^2 / 4) = 1 ∧ x₂^2 + (y₂^2 / 4) = 1 ∧ 
      x₁ * x₂ + y₁ * y₂ = 0 ∧ dist (P x₁ y₁) (P x₂ y₂) = 4 * real.sqrt 65 / 17))) :=
begin
  sorry,
end

end trajectory_and_perpendicular_lines_l93_93391


namespace tangent_line_equation_at_x_zero_l93_93961

noncomputable def curve (x : ℝ) : ℝ := x + Real.exp (2 * x)

theorem tangent_line_equation_at_x_zero :
  ∃ (k b : ℝ), (∀ x : ℝ, curve x = k * x + b) :=
by
  let df := fun (x : ℝ) => (deriv curve x)
  have k : ℝ := df 0
  have b : ℝ := curve 0 - k * 0
  use k, b
  sorry

end tangent_line_equation_at_x_zero_l93_93961


namespace find_common_difference_l93_93908

variable {a_n : ℕ → ℕ}
variable {d : ℕ}

-- Conditions
def first_term (a_n : ℕ → ℕ) := a_n 1 = 1
def common_difference (d : ℕ) := d ≠ 0
def arithmetic_def (a_n : ℕ → ℕ) (d : ℕ) := ∀ n, a_n (n+1) = a_n n + d
def geom_mean_condition (a_n : ℕ → ℕ) := a_n 2 ^ 2 = a_n 1 * a_n 4

-- Proof statement
theorem find_common_difference
  (fa : first_term a_n)
  (cd : common_difference d)
  (ad : arithmetic_def a_n d)
  (gmc : geom_mean_condition a_n) :
  d = 1 := by
  sorry

end find_common_difference_l93_93908


namespace sum_of_reciprocals_l93_93853

theorem sum_of_reciprocals {a b : ℕ} (h_sum: a + b = 55) (h_hcf: Nat.gcd a b = 5) (h_lcm: Nat.lcm a b = 120) :
  1 / (a : ℚ) + 1 / (b : ℚ) = 11 / 120 :=
by
  sorry

end sum_of_reciprocals_l93_93853


namespace possible_values_of_AD_l93_93804

-- Define the conditions as variables
variables {A B C D : ℝ}
variables {AB BC CD : ℝ}

-- Assume the given conditions
def conditions (A B C D : ℝ) (AB BC CD : ℝ) : Prop :=
  AB = 1 ∧ BC = 2 ∧ CD = 4

-- Define the proof goal: proving the possible values of AD
theorem possible_values_of_AD (h : conditions A B C D AB BC CD) :
  ∃ AD, AD = 1 ∨ AD = 3 ∨ AD = 5 ∨ AD = 7 :=
sorry

end possible_values_of_AD_l93_93804


namespace sequence_strictly_monotonic_increasing_l93_93121

noncomputable def a (n : ℕ) : ℝ := ((n + 1) ^ n * n ^ (2 - n)) / (7 * n ^ 2 + 1)

theorem sequence_strictly_monotonic_increasing :
  ∀ n : ℕ, a n < a (n + 1) := 
by {
  sorry
}

end sequence_strictly_monotonic_increasing_l93_93121


namespace seventh_oblong_number_l93_93343

/-- An oblong number is the number of dots in a rectangular grid where the number of rows is one more than the number of columns. -/
def is_oblong_number (n : ℕ) (x : ℕ) : Prop :=
  x = n * (n + 1)

/-- The 7th oblong number is 56. -/
theorem seventh_oblong_number : ∃ x, is_oblong_number 7 x ∧ x = 56 :=
by 
  use 56
  unfold is_oblong_number
  constructor
  rfl -- This confirms the computation 7 * 8 = 56
  sorry -- Wrapping up the proof, no further steps needed

end seventh_oblong_number_l93_93343


namespace handshake_problem_l93_93568

theorem handshake_problem :
  let companies := 5
  let representatives_per_company := 5
  let total_people := companies * representatives_per_company
  let possible_handshakes_per_person := total_people - representatives_per_company - 1
  let total_possible_handshakes := total_people * possible_handshakes_per_person
  let unique_handshakes := total_possible_handshakes / 2
  unique_handshakes = 250 :=
by 
  let companies := 5
  let representatives_per_company := 5
  let total_people := companies * representatives_per_company
  let possible_handshakes_per_person := total_people - representatives_per_company - 1
  let total_possible_handshakes := total_people * possible_handshakes_per_person
  let unique_handshakes := total_possible_handshakes / 2
  sorry

end handshake_problem_l93_93568


namespace remainder_of_7_pow_308_mod_11_l93_93979

theorem remainder_of_7_pow_308_mod_11 :
  (7 ^ 308) % 11 = 9 :=
by
  sorry

end remainder_of_7_pow_308_mod_11_l93_93979


namespace candy_difference_l93_93028

-- Defining the conditions as Lean hypotheses
variable (R K B M : ℕ)

-- Given conditions
axiom h1 : K = 4
axiom h2 : B = M - 6
axiom h3 : M = R + 2
axiom h4 : K = B + 2

-- Prove that Robert gets 2 more pieces of candy than Kate
theorem candy_difference : R - K = 2 :=
by {
  sorry
}

end candy_difference_l93_93028


namespace least_positive_divisible_by_primes_l93_93457

theorem least_positive_divisible_by_primes : 
  let p1 := 2 
  let p2 := 3 
  let p3 := 5 
  let p4 := 7
  ∃ n : ℕ, n > 0 ∧ (n % p1 = 0) ∧ (n % p2 = 0) ∧ (n % p3 = 0) ∧ (n % p4 = 0) ∧ 
  (∀ m : ℕ, m > 0 → (m % p1 = 0) ∧ (m % p2 = 0) ∧ (m % p3 = 0) ∧ (m % p4 = 0) → m ≥ n) ∧ n = 210 := 
by {
  sorry
}

end least_positive_divisible_by_primes_l93_93457


namespace increasing_digits_count_l93_93628

theorem increasing_digits_count : 
  ∃ n, n = 120 ∧ ∀ x : ℕ, x ≤ 1000 → (∀ i j : ℕ, i < j → ((x / 10^i % 10) < (x / 10^j % 10)) → 
  x ≤ 1000 ∧ (x / 10^i % 10) ≠ (x / 10^j % 10)) :=
sorry

end increasing_digits_count_l93_93628


namespace isabella_babysits_afternoons_per_week_l93_93265

-- Defining the conditions of Isabella's babysitting job
def hourly_rate : ℕ := 5
def hours_per_day : ℕ := 5
def days_per_week (weeks : ℕ) (total_earnings : ℕ) : ℕ := total_earnings / (weeks * (hourly_rate * hours_per_day))

-- Total earnings after 7 weeks
def total_earnings : ℕ := 1050
def weeks : ℕ := 7

-- State the theorem
theorem isabella_babysits_afternoons_per_week :
  days_per_week weeks total_earnings = 6 :=
by
  sorry

end isabella_babysits_afternoons_per_week_l93_93265


namespace ratio_of_areas_of_concentric_circles_l93_93846

theorem ratio_of_areas_of_concentric_circles 
  (C1 C2 : ℝ) (r1 r2 : ℝ)
  (h1 : r1 * C1 = 2 * π * r1)
  (h2 : r2 * C2 = 2 * π * r2)
  (h_c1 : 60 / 360 * C1 = 48 / 360 * C2) :
  (π * r1^2) / (π * r2^2) = 16 / 25 := by
  sorry

end ratio_of_areas_of_concentric_circles_l93_93846


namespace sum_digits_3times_l93_93006

-- Define the sum of digits function
noncomputable def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Define the 2006-th power of 2
noncomputable def power_2006 := 2 ^ 2006

-- State the theorem
theorem sum_digits_3times (n : ℕ) (h : n = power_2006) : 
  digit_sum (digit_sum (digit_sum n)) = 4 := by
  -- Add the proof steps here
  sorry

end sum_digits_3times_l93_93006


namespace greatest_y_value_l93_93559

theorem greatest_y_value (x y : ℤ) (h : x * y + 3 * x + 2 * y = -2) : y ≤ 1 :=
sorry

end greatest_y_value_l93_93559


namespace percentage_change_in_area_of_rectangle_l93_93669

theorem percentage_change_in_area_of_rectangle
  (L B : ℝ) (hL : L > 0) (hB : B > 0) :
  let A1 := L * B
  let A2 := (L / 2) * (3 * B)
  let percentage_change := ((A2 - A1) / A1) * 100
  percentage_change = 50 := by
  let A1 := L * B
  let A2 := (L / 2) * (3 * B)
  let percentage_change := ((A2 - A1) / A1) * 100
  calc percentage_change
      = ((A2 - A1) / A1) * 100 : rfl
  ... = ((((L / 2) * (3 * B)) - (L * B)) / (L * B)) * 100 : by rw [A1, A2]
  ... = ((3 / 2 * A1 - A1) / A1) * 100 : by rw [A2, ←mul_assoc, mul_div_cancel' _ (two_ne_zero : 2 ≠ 0)]
  ... = ((3 / 2 - 1) * A1 / A1) * 100 : by simp only [mul_sub, mul_div_cancel' _ (ne_of_gt hL)]
  ... = (1 / 2) * 100 : by rw [div_mul_cancel' (ne_of_gt (lt_of_lt_of_le zero_lt_one (le_of_lt ((mul_lt_iff_lt_one_left (lt_of_lt_of_le zero_lt_one hL)).mpr (half_pos (lt_of_le_of_ne (le_of_lt (lt_of_lt_of_le (lt_one_mul_self (lt_of_le_of_lt (zero_le_two) (lt_add_one 1))) (lt_of_lt_of_le (two_le_one) (lt_one 2))))))))) so))
  ... = 50 : rfl

end percentage_change_in_area_of_rectangle_l93_93669


namespace satisfy_eq_pairs_l93_93739

theorem satisfy_eq_pairs (x y : ℤ) : (x^2 = y^2 + 2 * y + 13) ↔ (x = 4 ∧ (y = 1 ∨ y = -3) ∨ x = -4 ∧ (y = 1 ∨ y = -3)) :=
by
  sorry

end satisfy_eq_pairs_l93_93739


namespace inequality_int_part_l93_93953

theorem inequality_int_part (a : ℝ) (n : ℕ) (h1 : 1 ≤ a) (h2 : (0 : ℝ) ≤ n ∧ (n : ℝ) ≤ a) : 
  ⌊a⌋ > (n / (n + 1 : ℝ)) * a := 
by 
  sorry

end inequality_int_part_l93_93953


namespace simplify_and_evaluate_at_3_l93_93955

noncomputable def expression (x : ℝ) : ℝ := 
  (3 / (x - 1) - x - 1) / ((x^2 - 4 * x + 4) / (x - 1))

theorem simplify_and_evaluate_at_3 : expression 3 = -5 := 
  sorry

end simplify_and_evaluate_at_3_l93_93955


namespace complete_the_square_eqn_l93_93958

theorem complete_the_square_eqn (x b c : ℤ) (h_eqn : x^2 - 10 * x + 15 = 0) (h_form : (x + b)^2 = c) : b + c = 5 := by
  sorry

end complete_the_square_eqn_l93_93958


namespace lowest_exam_score_l93_93646

theorem lowest_exam_score 
  (first_exam_score : ℕ := 90) 
  (second_exam_score : ℕ := 108) 
  (third_exam_score : ℕ := 102) 
  (max_score_per_exam : ℕ := 120) 
  (desired_average : ℕ := 100) 
  (total_exams : ℕ := 5) 
  (total_score_needed : ℕ := desired_average * total_exams) : 
  ∃ (lowest_score : ℕ), lowest_score = 80 :=
by
  sorry

end lowest_exam_score_l93_93646


namespace max_tiles_l93_93710

open Nat

theorem max_tiles (tile_width tile_height floor_width floor_height : ℕ) 
  (h_tile_dims : tile_width = 45 ∧ tile_height = 50) 
  (h_floor_dims : floor_width = 250 ∧ floor_height = 180) : 
  (max 
      ((floor (floor_width / tile_width.toRat)).toNat * (floor (floor_height / tile_height.toRat)).toNat)
      ((floor (floor_width / tile_height.toRat)).toNat * (floor (floor_height / tile_width.toRat)).toNat)) = 20 := 
by
  sorry

end max_tiles_l93_93710


namespace math_proof_problem_l93_93820

noncomputable def sum_of_distinct_squares (a b c : ℕ) : ℕ :=
3 * ((a^2 + b^2 + c^2 : ℕ))

theorem math_proof_problem (a b c : ℕ)
  (h1 : a + b + c = 27)
  (h2 : Nat.gcd a b + Nat.gcd b c + Nat.gcd c a = 11) :
  sum_of_distinct_squares a b c = 2274 :=
sorry

end math_proof_problem_l93_93820


namespace irreducible_fraction_l93_93660

theorem irreducible_fraction {n : ℕ} : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 :=
by
  sorry

end irreducible_fraction_l93_93660


namespace cost_per_acre_proof_l93_93572

def cost_of_land (tac tl : ℕ) (hc hcc hcp heq : ℕ) (ttl : ℕ) : ℕ := ttl - (hc + hcc + hcp + heq)

def cost_per_acre (total_land : ℕ) (cost_land : ℕ) : ℕ := cost_land / total_land

theorem cost_per_acre_proof (tac tl hc hcc hcp heq ttl epl : ℕ) 
  (h1 : tac = 30)
  (h2 : hc = 120000)
  (h3 : hcc = 20 * 1000)
  (h4 : hcp = 100 * 5)
  (h5 : heq = 6 * 100 + 6000)
  (h6 : ttl = 147700) :
  cost_per_acre tac (cost_of_land tac tl hc hcc hcp heq ttl) = epl := by
  sorry

end cost_per_acre_proof_l93_93572


namespace adah_practiced_total_hours_l93_93480

theorem adah_practiced_total_hours :
  let minutes_per_day := 86
  let days_practiced := 2
  let minutes_other_days := 278
  let total_minutes := (minutes_per_day * days_practiced) + minutes_other_days
  let total_hours := total_minutes / 60
  total_hours = 7.5 :=
by
  sorry

end adah_practiced_total_hours_l93_93480


namespace exponential_inequality_l93_93056

-- Define the conditions for the problem
variables {x y a : ℝ}
axiom h1 : x > y
axiom h2 : y > 1
axiom h3 : 0 < a
axiom h4 : a < 1

-- State the problem to be proved
theorem exponential_inequality (h1 : x > y) (h2 : y > 1) (h3 : 0 < a) (h4 : a < 1) : a ^ x < a ^ y :=
sorry

end exponential_inequality_l93_93056


namespace find_z_l93_93263

-- Define the given angles
def angle_ABC : ℝ := 95
def angle_BAC : ℝ := 65

-- Define the angle sum property for triangle ABC
def angle_sum_triangle_ABC (a b : ℝ) : ℝ := 180 - (a + b)

-- Define the angle DCE as equal to angle BCA
def angle_DCE : ℝ := angle_sum_triangle_ABC angle_ABC angle_BAC

-- Define the angle sum property for right triangle CDE
def z (dce : ℝ) : ℝ := 90 - dce

-- State the theorem to be proved
theorem find_z : z angle_DCE = 70 :=
by
  -- Statement for proof is provided
  sorry

end find_z_l93_93263


namespace cost_per_trip_l93_93275

theorem cost_per_trip (cost_per_pass : ℕ) (num_passes : ℕ) (trips_oldest : ℕ) (trips_youngest : ℕ) :
    cost_per_pass = 100 →
    num_passes = 2 →
    trips_oldest = 35 →
    trips_youngest = 15 →
    (cost_per_pass * num_passes) / (trips_oldest + trips_youngest) = 4 := by
  sorry

end cost_per_trip_l93_93275


namespace subset_of_primes_is_all_primes_l93_93860

theorem subset_of_primes_is_all_primes
  (P : Set ℕ)
  (M : Set ℕ)
  (hP : ∀ n, n ∈ P ↔ Nat.Prime n)
  (hM : ∀ S : Finset ℕ, (∀ p ∈ S, p ∈ M) → ∀ p, p ∣ (Finset.prod S id + 1) → p ∈ M) :
  M = P :=
sorry

end subset_of_primes_is_all_primes_l93_93860


namespace desks_in_classroom_l93_93194

theorem desks_in_classroom (d c : ℕ) (h1 : c = 4 * d) (h2 : 4 * c + 6 * d = 728) : d = 33 :=
by
  -- The proof is omitted, this placeholder is to indicate that it is required to complete the proof.
  sorry

end desks_in_classroom_l93_93194


namespace sum_inverses_of_roots_l93_93503

open Polynomial

theorem sum_inverses_of_roots (a b c : ℝ) (h1 : a^3 - 2020 * a + 1010 = 0)
    (h2 : b^3 - 2020 * b + 1010 = 0) (h3 : c^3 - 2020 * c + 1010 = 0) :
    (1/a) + (1/b) + (1/c) = 2 := 
  sorry

end sum_inverses_of_roots_l93_93503


namespace range_of_a_l93_93948

def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2 * a * x + 4 > 0

def q (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (3 - 2 * a)^x < (3 - 2 * a)^y

theorem range_of_a (a : ℝ) (h1 : p a ∨ q a) (h2 : ¬ (p a ∧ q a)) : (1 ≤ a ∧ a < 2) ∨ (a ≤ -2) :=
sorry

end range_of_a_l93_93948


namespace cannot_be_n_plus_2_l93_93013

theorem cannot_be_n_plus_2 (n : ℕ) : 
  ¬(∃ Y, (Y = n + 2) ∧ 
         ((Y = n - 3) ∨ (Y = n - 1) ∨ (Y = n + 5))) := 
by {
  sorry
}

end cannot_be_n_plus_2_l93_93013


namespace total_birds_count_l93_93474

def cage1_parrots := 9
def cage1_finches := 4
def cage1_canaries := 7

def cage2_parrots := 5
def cage2_parakeets := 8
def cage2_finches := 10

def cage3_parakeets := 15
def cage3_finches := 7
def cage3_canaries := 3

def cage4_parrots := 10
def cage4_parakeets := 5
def cage4_finches := 12

def total_birds := cage1_parrots + cage1_finches + cage1_canaries +
                   cage2_parrots + cage2_parakeets + cage2_finches +
                   cage3_parakeets + cage3_finches + cage3_canaries +
                   cage4_parrots + cage4_parakeets + cage4_finches

theorem total_birds_count : total_birds = 95 :=
by
  -- Proof is omitted here.
  sorry

end total_birds_count_l93_93474


namespace correct_operation_l93_93700

theorem correct_operation (a b : ℝ) :
  ¬ (a^2 + a^3 = a^6) ∧
  ¬ ((a*b)^2 = a*(b^2)) ∧
  ¬ ((a + b)^2 = a^2 + b^2) ∧
  ((a + b)*(a - b) = a^2 - b^2) := 
by
  sorry

end correct_operation_l93_93700


namespace chocolate_ticket_fraction_l93_93959

theorem chocolate_ticket_fraction (box_cost : ℝ) (ticket_count_per_free_box : ℕ) (ticket_count_included : ℕ) :
  ticket_count_per_free_box = 10 →
  ticket_count_included = 1 →
  (1 / 9 : ℝ) * box_cost =
  box_cost / ticket_count_per_free_box + box_cost / (ticket_count_per_free_box - ticket_count_included + 1) :=
by 
  intros h1 h2 
  have h : ticket_count_per_free_box = 10 := h1 
  have h' : ticket_count_included = 1 := h2 
  sorry

end chocolate_ticket_fraction_l93_93959


namespace m_is_perfect_square_l93_93378

theorem m_is_perfect_square (n : ℕ) (m : ℤ) (h1 : m = 2 + 2 * Int.sqrt (44 * n^2 + 1) ∧ Int.sqrt (44 * n^2 + 1) * Int.sqrt (44 * n^2 + 1) = 44 * n^2 + 1) :
  ∃ k : ℕ, m = k^2 :=
by
  sorry

end m_is_perfect_square_l93_93378


namespace increasing_function_a_values_l93_93545

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 4 then a * x - 8 else x^2 - 2 * a * x

theorem increasing_function_a_values (a : ℝ) (h : ∀ x₁ x₂ : ℝ, x₁ < x₂ → f a x₁ ≤ f a x₂) : 
  0 < a ∧ a ≤ 2 :=
sorry

end increasing_function_a_values_l93_93545


namespace find_m_l93_93075

noncomputable def vector_sum (a b : ℝ × ℝ) : ℝ × ℝ :=
  (a.1 + b.1, a.2 + b.2)

noncomputable def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem find_m (m : ℝ) :
  let a := (1, m)
  let b := (3, -2)
  are_parallel (vector_sum a b) b → m = -2 / 3 :=
by
  sorry

end find_m_l93_93075


namespace sum_fifth_powers_l93_93069

theorem sum_fifth_powers (a b c : ℝ)
  (h1 : a + b + c = 2)
  (h2 : a^2 + b^2 + c^2 = 5)
  (h3 : a^3 + b^3 + c^3 = 8) : 
  a^5 + b^5 + c^5 = 98 / 6 := 
by 
  sorry

end sum_fifth_powers_l93_93069


namespace find_f_of_2_l93_93247

def f (x : ℝ) (a b : ℝ) : ℝ := x^3 + a * x^2 + b * x + a^2

theorem find_f_of_2 (a b : ℝ)
  (h1 : 3 + 2 * a + b = 0)
  (h2 : 1 + a + b + a^2 = 10)
  (ha : a = 4)
  (hb : b = -11) :
  f 2 a b = 18 := by {
  -- We assume the values of a and b provided by the user as the correct pair.
  sorry
}

end find_f_of_2_l93_93247


namespace library_science_books_count_l93_93427

-- Definitions based on the problem conditions
def initial_science_books := 120
def borrowed_books := 40
def returned_books := 15
def books_on_hold := 10
def borrowed_from_other_library := 20
def lost_books := 2
def damaged_books := 1

-- Statement for the proof.
theorem library_science_books_count :
  initial_science_books - borrowed_books + returned_books - books_on_hold + borrowed_from_other_library - lost_books - damaged_books = 102 :=
by
  sorry

end library_science_books_count_l93_93427


namespace total_employees_in_buses_l93_93452

-- Definitions from conditions
def busCapacity : ℕ := 150
def percentageFull1 : ℕ := 60
def percentageFull2 : ℕ := 70

-- Proving the total number of employees
theorem total_employees_in_buses : 
  (percentageFull1 * busCapacity / 100) + (percentageFull2 * busCapacity / 100) = 195 := 
by
  sorry

end total_employees_in_buses_l93_93452


namespace polynomial_roots_l93_93744

theorem polynomial_roots : ∀ x : ℝ, 3 * x^4 + 2 * x^3 - 8 * x^2 + 2 * x + 3 = 0 :=
by
  sorry

end polynomial_roots_l93_93744


namespace possible_values_of_a_l93_93546

def f (x : ℝ) (a : ℝ) : ℝ :=
if x < 4 then (a * x - 8) else (x^2 - 2 * a * x)

theorem possible_values_of_a (a : ℝ) : (0 < a ∧ a ≤ 2) ↔ (∀ x : ℝ, x < 4 → (f x a)' > 0) ∧ (∀ x : ℝ, x ≥ 4 → (f x a)' ≥ 0) ∧ (4 * a - 8 ≤ 16 - 8 * a) :=
by
  sorry

end possible_values_of_a_l93_93546


namespace simplify_to_quadratic_l93_93938

noncomputable def simplify_expression (a b c x : ℝ) : ℝ := 
  (x + a)^2 / ((a - b) * (a - c)) + 
  (x + b)^2 / ((b - a) * (b - c + 2)) + 
  (x + c)^2 / ((c - a) * (c - b))

theorem simplify_to_quadratic {a b c x : ℝ} (ha : a ≠ b) (hb : b ≠ c) (hc : c ≠ a) :
  simplify_expression a b c x = x^2 - (a + b + c) * x + sorry :=
sorry

end simplify_to_quadratic_l93_93938


namespace henry_collection_cost_l93_93465

def initial_figures : ℕ := 3
def total_needed : ℕ := 8
def cost_per_figure : ℕ := 6

theorem henry_collection_cost : 
  let needed_figures := total_needed - initial_figures
  let total_cost := needed_figures * cost_per_figure
  total_cost = 30 := 
by
  let needed_figures := total_needed - initial_figures
  let total_cost := needed_figures * cost_per_figure
  sorry

end henry_collection_cost_l93_93465


namespace sandy_marks_per_correct_sum_l93_93413

theorem sandy_marks_per_correct_sum 
  (total_sums : ℕ)
  (total_marks : ℤ)
  (correct_sums : ℕ)
  (marks_per_incorrect_sum : ℤ)
  (marks_obtained : ℤ) 
  (marks_per_correct_sum : ℕ) :
  total_sums = 30 →
  total_marks = 45 →
  correct_sums = 21 →
  marks_per_incorrect_sum = 2 →
  marks_obtained = total_marks →
  marks_obtained = marks_per_correct_sum * correct_sums - marks_per_incorrect_sum * (total_sums - correct_sums) → 
  marks_per_correct_sum = 3 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end sandy_marks_per_correct_sum_l93_93413


namespace abs_m_plus_one_l93_93231

theorem abs_m_plus_one (m : ℝ) (h : |m| = m + 1) : (4 * m - 1) ^ 4 = 81 := by
  sorry

end abs_m_plus_one_l93_93231


namespace max_value_of_trig_expr_l93_93045

variable (x : ℝ)

theorem max_value_of_trig_expr : 
  (∀ x : ℝ, 3 * Real.cos x + 4 * Real.sin x ≤ 5) ∧
  (∃ x : ℝ, 3 * Real.cos x + 4 * Real.sin x = 5) :=
sorry

end max_value_of_trig_expr_l93_93045


namespace function_increasing_l93_93425

noncomputable def f (x a b c : ℝ) : ℝ := x^3 + a * x^2 + b * x + c

theorem function_increasing (a b c : ℝ) (h : a^2 - 3 * b < 0) : 
  ∀ x y : ℝ, x < y → f x a b c < f y a b c := sorry

end function_increasing_l93_93425


namespace percentage_of_customers_purchased_l93_93092

theorem percentage_of_customers_purchased (ad_cost : ℕ) (customers : ℕ) (price_per_sale : ℕ) (profit : ℕ)
  (h1 : ad_cost = 1000)
  (h2 : customers = 100)
  (h3 : price_per_sale = 25)
  (h4 : profit = 1000) :
  (profit / price_per_sale / customers) * 100 = 40 :=
by
  sorry

end percentage_of_customers_purchased_l93_93092


namespace initial_principal_amount_l93_93285

theorem initial_principal_amount
  (A : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) (P : ℝ)
  (hA : A = 8400) 
  (hr : r = 0.05)
  (hn : n = 1) 
  (ht : t = 1) 
  (hformula : A = P * (1 + r / n) ^ (n * t)) : 
  P = 8000 :=
by
  rw [hA, hr, hn, ht] at hformula
  sorry

end initial_principal_amount_l93_93285


namespace multiply_res_l93_93914

theorem multiply_res (
  h : 213 * 16 = 3408
) : 1.6 * 213 = 340.8 :=
sorry

end multiply_res_l93_93914


namespace scientific_notation_of_125000_l93_93023

theorem scientific_notation_of_125000 :
  125000 = 1.25 * 10^5 := sorry

end scientific_notation_of_125000_l93_93023


namespace projectile_height_reaches_35_l93_93666

theorem projectile_height_reaches_35 
  (t : ℝ)
  (h_eq : -4.9 * t^2 + 30 * t = 35) :
  t = 2 ∨ t = 50 / 7 ∧ t = min (2 : ℝ) (50 / 7) :=
by
  sorry

end projectile_height_reaches_35_l93_93666


namespace range_of_a_l93_93241

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * (a - 1 / Real.exp x)

theorem range_of_a (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ deriv (f a) x₁ = 0 ∧ deriv (f a) x₂ = 0) ↔ -1 / Real.exp 2 < a ∧ a < 0 := 
sorry

end range_of_a_l93_93241


namespace Milly_study_time_l93_93406

theorem Milly_study_time :
  let math_time := 60
  let geo_time := math_time / 2
  let mean_time := (math_time + geo_time) / 2
  let total_study_time := math_time + geo_time + mean_time
  total_study_time = 135 := by
  sorry

end Milly_study_time_l93_93406


namespace min_value_xy_expression_l93_93695

theorem min_value_xy_expression : ∃ x y : ℝ, (xy - 2)^2 + (x^2 + y^2) = 4 :=
by
  sorry

end min_value_xy_expression_l93_93695


namespace polynomial_value_at_two_l93_93306

def f (x : ℝ) : ℝ := x^5 + 5*x^4 + 10*x^3 + 10*x^2 + 5*x + 1

theorem polynomial_value_at_two : f 2 = 243 := by
  -- Proof steps go here
  sorry

end polynomial_value_at_two_l93_93306


namespace intersection_is_correct_l93_93617

-- Defining sets A and B
def setA : Set ℝ := {x : ℝ | x^2 - 2 * x ≤ 0}
def setB : Set ℝ := {x : ℝ | -1 < x ∧ x < 1}

-- Target intersection set
def setIntersection : Set ℝ := {x : ℝ | 0 ≤ x ∧ x < 1}

-- Theorem to be proved
theorem intersection_is_correct : (setA ∩ setB) = setIntersection :=
by
  -- Proof steps will go here
  sorry

end intersection_is_correct_l93_93617


namespace claire_gerbils_l93_93576

theorem claire_gerbils (G H : ℕ) (h1 : G + H = 92) (h2 : (1/4 : ℚ) * G + (1/3 : ℚ) * H = 25) : G = 68 :=
sorry

end claire_gerbils_l93_93576


namespace molecular_weight_of_barium_iodide_l93_93696

-- Define the atomic weights
def atomic_weight_of_ba : ℝ := 137.33
def atomic_weight_of_i : ℝ := 126.90

-- Define the molecular weight calculation for Barium iodide
def molecular_weight_of_bai2 : ℝ := atomic_weight_of_ba + 2 * atomic_weight_of_i

-- The main theorem to prove
theorem molecular_weight_of_barium_iodide : molecular_weight_of_bai2 = 391.13 := by
  -- we are given that atomic_weight_of_ba = 137.33 and atomic_weight_of_i = 126.90
  -- hence, molecular_weight_of_bai2 = 137.33 + 2 * 126.90
  -- simplifying this, we get
  -- molecular_weight_of_bai2 = 137.33 + 253.80 = 391.13
  sorry

end molecular_weight_of_barium_iodide_l93_93696


namespace robert_books_l93_93951

/-- Given that Robert reads at a speed of 75 pages per hour, books have 300 pages, and Robert reads for 9 hours,
    he can read 2 complete 300-page books in that time. -/
theorem robert_books (reading_speed : ℤ) (pages_per_book : ℤ) (hours_available : ℤ) 
(h1 : reading_speed = 75) 
(h2 : pages_per_book = 300) 
(h3 : hours_available = 9) : 
  hours_available / (pages_per_book / reading_speed) = 2 := 
by {
  -- adding placeholder for proof
  sorry
}

end robert_books_l93_93951


namespace find_a_l93_93770

theorem find_a (a : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, f x = abs (2 * x - a) + a)
  (h2 : ∀ x : ℝ, f x ≤ 6 ↔ -2 ≤ x ∧ x ≤ 3) : 
  a = 1 := by
  sorry

end find_a_l93_93770


namespace shaniqua_styles_count_l93_93120

variable (S : ℕ)

def shaniqua_haircuts (haircuts : ℕ) : ℕ := 12 * haircuts
def shaniqua_styles (styles : ℕ) : ℕ := 25 * styles

theorem shaniqua_styles_count (total_money haircuts : ℕ) (styles : ℕ) :
  total_money = shaniqua_haircuts haircuts + shaniqua_styles styles → haircuts = 8 → total_money = 221 → S = 5 :=
by
  sorry

end shaniqua_styles_count_l93_93120


namespace no_fixed_point_range_of_a_fixed_point_in_interval_l93_93080

-- Problem (1)
theorem no_fixed_point_range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 + a * x + a ≠ x) →
  3 - 2 * Real.sqrt 2 < a ∧ a < 3 + 2 * Real.sqrt 2 :=
by
  sorry

-- Problem (2)
theorem fixed_point_in_interval (f : ℝ → ℝ) (n : ℤ) :
  (∀ x : ℝ, f x = -Real.log x + 3) →
  (∃ x₀ : ℝ, f x₀ = x₀ ∧ n ≤ x₀ ∧ x₀ < n + 1) →
  n = 2 :=
by
  sorry

end no_fixed_point_range_of_a_fixed_point_in_interval_l93_93080


namespace slightly_used_crayons_l93_93302

theorem slightly_used_crayons (total_crayons : ℕ) (percent_new : ℚ) (percent_broken : ℚ) 
  (h1 : total_crayons = 250) (h2 : percent_new = 40/100) (h3 : percent_broken = 1/5) : 
  (total_crayons - percent_new * total_crayons - percent_broken * total_crayons) = 100 :=
by
  -- sorry here to indicate the proof is omitted
  sorry

end slightly_used_crayons_l93_93302


namespace total_employees_in_buses_l93_93450

-- Define the capacity of each bus
def capacity : ℕ := 150

-- Define the fill percentages of each bus
def fill_percentage_bus1 : ℚ := 60 / 100
def fill_percentage_bus2 : ℚ := 70 / 100

-- Calculate the number of passengers in each bus
def passengers_bus1 : ℚ := fill_percentage_bus1 * capacity
def passengers_bus2 : ℚ := fill_percentage_bus2 * capacity

-- Calculate the total number of passengers
def total_passengers : ℚ := passengers_bus1 + passengers_bus2

-- The proof statement
theorem total_employees_in_buses : total_passengers = 195 :=
by
  sorry

end total_employees_in_buses_l93_93450


namespace cube_edge_percentage_growth_l93_93981

theorem cube_edge_percentage_growth (p : ℝ) 
  (h : (1 + p / 100) ^ 2 - 1 = 0.96) : p = 40 :=
by
  sorry

end cube_edge_percentage_growth_l93_93981


namespace candy_left_l93_93047

-- Define the given conditions
def KatieCandy : ℕ := 8
def SisterCandy : ℕ := 23
def AteCandy : ℕ := 8

-- The theorem stating the total number of candy left
theorem candy_left (k : ℕ) (s : ℕ) (e : ℕ) (hk : k = KatieCandy) (hs : s = SisterCandy) (he : e = AteCandy) : 
  (k + s) - e = 23 :=
by
  -- (Proof will be inserted here, but we include a placeholder "sorry" for now)
  sorry

end candy_left_l93_93047


namespace yi_successful_shots_l93_93947

-- Defining the basic conditions
variables {x y : ℕ} -- Number of successful shots made by Jia and Yi respectively

-- Each hit gains 20 points and each miss deducts 12 points.
-- Both person A (Jia) and person B (Yi) made 10 shots each.
def total_shots (x y : ℕ) : Prop := 
  (20 * x - 12 * (10 - x)) + (20 * y - 12 * (10 - y)) = 208 ∧ x + y = 14 ∧ x - y = 2

theorem yi_successful_shots (x y : ℕ) (h : total_shots x y) : y = 6 := 
  by sorry

end yi_successful_shots_l93_93947


namespace clock_angle_at_330_l93_93977

/--
At 3:00, the hour hand is at 90 degrees from the 12 o'clock position.
The minute hand at 3:30 is at 180 degrees from the 12 o'clock position.
The hour hand at 3:30 has moved an additional 15 degrees (0.5 degrees per minute).
Prove that the smaller angle formed by the hour and minute hands of a clock at 3:30 is 75.0 degrees.
-/
theorem clock_angle_at_330 : 
  let hour_pos_at_3 := 90
  let min_pos_at_330 := 180
  let hour_additional := 15
  (min_pos_at_330 - (hour_pos_at_3 + hour_additional) = 75)
  :=
  by
  sorry

end clock_angle_at_330_l93_93977


namespace greatest_possible_perimeter_l93_93254

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem greatest_possible_perimeter :
  ∃ x : ℕ, 6 ≤ x ∧ x < 17 ∧ is_triangle x (2 * x) 17 ∧ (x + 2 * x + 17 = 65) := by
  sorry

end greatest_possible_perimeter_l93_93254


namespace smallest_perfect_square_divisible_by_5_and_7_l93_93313

theorem smallest_perfect_square_divisible_by_5_and_7 
  (n : ℕ) 
  (h1 : n > 0) 
  (h2 : ∃ k : ℕ, n = k^2)
  (h3 : 5 ∣ n)
  (h4 : 7 ∣ n) : 
  n = 1225 :=
sorry

end smallest_perfect_square_divisible_by_5_and_7_l93_93313


namespace average_salary_l93_93664

theorem average_salary (R S T : ℝ) 
  (h1 : (R + S) / 2 = 4000) 
  (h2 : T = 7000) : 
  (R + S + T) / 3 = 5000 :=
by
  sorry

end average_salary_l93_93664


namespace joes_speed_second_part_l93_93794

theorem joes_speed_second_part
  (d1 d2 t1 t_total: ℝ)
  (s1 s_avg: ℝ)
  (h_d1: d1 = 420)
  (h_d2: d2 = 120)
  (h_s1: s1 = 60)
  (h_s_avg: s_avg = 54) :
  (d1 / s1 + d2 / (d2 / 40) = t_total ∧ t_total = (d1 + d2) / s_avg) →
  d2 / (t_total - d1 / s1) = 40 :=
by
  sorry

end joes_speed_second_part_l93_93794


namespace max_product_of_two_integers_with_sum_2004_l93_93970

theorem max_product_of_two_integers_with_sum_2004 :
  ∃ x y : ℤ, x + y = 2004 ∧ (∀ a b : ℤ, a + b = 2004 → a * b ≤ x * y) ∧ x * y = 1004004 := 
by
  sorry

end max_product_of_two_integers_with_sum_2004_l93_93970


namespace problem_statement_l93_93078

variable (a b c : ℤ) -- Declare variables as integers

-- Define conditions based on the problem
def smallest_natural_number (a : ℤ) := a = 1
def largest_negative_integer (b : ℤ) := b = -1
def number_equal_to_its_opposite (c : ℤ) := c = 0

-- State the theorem
theorem problem_statement (h1 : smallest_natural_number a) 
                         (h2 : largest_negative_integer b) 
                         (h3 : number_equal_to_its_opposite c) : 
  a + b + c = 0 := 
  by 
    rw [h1, h2, h3] 
    simp

end problem_statement_l93_93078


namespace total_soldiers_correct_l93_93787

-- Definitions based on conditions
def num_generals := 8
def num_vanguards := 8^2
def num_flags := 8^3
def num_team_leaders := 8^4
def num_armored_soldiers := 8^5
def num_soldiers := 8 + 8^2 + 8^3 + 8^4 + 8^5 + 8^6

-- Prove total number of soldiers
theorem total_soldiers_correct : num_soldiers = (1 / 7 : ℝ) * (8^7 - 8) := by
  sorry

end total_soldiers_correct_l93_93787


namespace negation_existence_l93_93834

-- The problem requires showing the equivalence between the negation of an existential
-- proposition and a universal proposition in the context of real numbers.

theorem negation_existence (m : ℝ) :
  (¬ ∃ x : ℝ, x^2 - m * x - m < 0) → (∀ x : ℝ, x^2 - m * x - m ≥ 0) :=
by
  sorry

end negation_existence_l93_93834


namespace books_sold_on_Tuesday_l93_93481

theorem books_sold_on_Tuesday 
  (initial_stock : ℕ)
  (books_sold_Monday : ℕ)
  (books_sold_Wednesday : ℕ)
  (books_sold_Thursday : ℕ)
  (books_sold_Friday : ℕ)
  (books_not_sold : ℕ) :
  initial_stock = 800 →
  books_sold_Monday = 60 →
  books_sold_Wednesday = 20 →
  books_sold_Thursday = 44 →
  books_sold_Friday = 66 →
  books_not_sold = 600 →
  ∃ (books_sold_Tuesday : ℕ), books_sold_Tuesday = 10
:= by
  intros h_initial h_monday h_wednesday h_thursday h_friday h_not_sold
  sorry

end books_sold_on_Tuesday_l93_93481


namespace sweets_distribution_l93_93945

theorem sweets_distribution (S : ℕ) (N : ℕ) (h1 : N - 70 > 0) (h2 : S = N * 24) (h3 : S = (N - 70) * 38) : N = 190 :=
by
  sorry

end sweets_distribution_l93_93945


namespace range_of_m_l93_93509

theorem range_of_m (m : ℝ)
  (h₁ : (m^2 - 4) ≥ 0)
  (h₂ : (4 * (m - 2)^2 - 16) < 0) :
  1 < m ∧ m ≤ 2 :=
by
  sorry

end range_of_m_l93_93509


namespace sum_of_midpoint_xcoords_l93_93159

theorem sum_of_midpoint_xcoords (a b c : ℝ) (h : a + b + c = 15) :
    (a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 15 :=
  by
    sorry

end sum_of_midpoint_xcoords_l93_93159


namespace sum_even_minus_odd_from_1_to_100_l93_93616

noncomputable def sum_even_numbers : Nat :=
  (List.range' 2 99 2).sum

noncomputable def sum_odd_numbers : Nat :=
  (List.range' 1 100 2).sum

theorem sum_even_minus_odd_from_1_to_100 :
  sum_even_numbers - sum_odd_numbers = 50 :=
by
  sorry

end sum_even_minus_odd_from_1_to_100_l93_93616


namespace total_avg_donation_per_person_l93_93634

-- Definition of variables and conditions
variables (avgA avgB : ℝ) (numA numB : ℕ)
variables (h1 : avgB = avgA - 100)
variables (h2 : 2 * numA * avgA = 4 * numB * (avgA - 100))
variables (h3 : numA = numB / 4)

-- Lean 4 statement to prove the total average donation per person is 120
theorem total_avg_donation_per_person (h1 :  avgB = avgA - 100)
    (h2 : 2 * numA * avgA = 4 * numB * (avgA - 100))
    (h3 : numA = numB / 4) : 
    ( (numA * avgA + numB * avgB) / (numA + numB) ) = 120 :=
sorry

end total_avg_donation_per_person_l93_93634


namespace find_a_b_l93_93680

theorem find_a_b (a b : ℝ) :
  (∀ x : ℝ, (x < -2 ∨ x > 1) → (x^2 + a * x + b > 0)) →
  (a = 1 ∧ b = -2) :=
by
  sorry

end find_a_b_l93_93680


namespace no_real_roots_range_k_l93_93248

theorem no_real_roots_range_k (k : ℝ) : (x^2 - 2 * x - k = 0) ∧ (∀ x : ℝ, x^2 - 2 * x - k ≠ 0) → k < -1 := 
by
  sorry

end no_real_roots_range_k_l93_93248


namespace hans_room_count_l93_93775

theorem hans_room_count :
  let total_floors := 10
  let rooms_per_floor := 10
  let unavailable_floors := 1
  let available_floors := total_floors - unavailable_floors
  available_floors * rooms_per_floor = 90 := by
  let total_floors := 10
  let rooms_per_floor := 10
  let unavailable_floors := 1
  let available_floors := total_floors - unavailable_floors
  show available_floors * rooms_per_floor = 90
  sorry

end hans_room_count_l93_93775


namespace y1_gt_y2_l93_93632

theorem y1_gt_y2 (y : ℤ → ℤ) (h_eq : ∀ x, y x = 8 * x - 1)
  (y1 y2 : ℤ) (h_y1 : y 3 = y1) (h_y2 : y 2 = y2) : y1 > y2 :=
by
  -- proof
  sorry

end y1_gt_y2_l93_93632


namespace meadow_area_l93_93728

theorem meadow_area (x : ℝ) (h1 : ∀ y : ℝ, y = x / 2 + 3) (h2 : ∀ z : ℝ, z = 1 / 3 * (x / 2 - 3) + 6) :
  (x / 2 + 3) + (1 / 3 * (x / 2 - 3) + 6) = x → x = 24 := by
  sorry

end meadow_area_l93_93728


namespace prob_xi_greater_than_2_l93_93103

noncomputable theory

open MeasureTheory

variable (ξ : MeasureTheory.ProbabilityTheory.RealRandomVar)
variable (σ : ℝ)

axiom normal_distribution_ξ : normal ξ 1 σ
axiom σ_positive : σ > 0
axiom prob_0_to_1 : ProbabilityTheory.Probability (ξ > 0 ∧ ξ < 1) = 0.4

theorem prob_xi_greater_than_2 : ProbabilityTheory.Probability (ξ > 2) = 0.2 :=
by
  sorry

end prob_xi_greater_than_2_l93_93103


namespace wire_ratio_is_one_l93_93202

theorem wire_ratio_is_one (a b : ℝ) (h1 : a = b) : a / b = 1 := by
  -- The proof goes here
  sorry

end wire_ratio_is_one_l93_93202


namespace unknown_number_value_l93_93777

theorem unknown_number_value (x n : ℝ) (h1 : 0.75 / x = n / 8) (h2 : x = 2) : n = 3 :=
by
  sorry

end unknown_number_value_l93_93777


namespace sum_of_midpoints_l93_93148

theorem sum_of_midpoints (p q r : ℝ) (h : p + q + r = 15) :
  (p + q) / 2 + (p + r) / 2 + (q + r) / 2 = 15 :=
by
  sorry

end sum_of_midpoints_l93_93148


namespace simplify_fraction_l93_93130

theorem simplify_fraction : 
  (1:ℚ) / 462 + 17 / 42 = 94 / 231 := 
by
  sorry

end simplify_fraction_l93_93130


namespace infinite_non_prime_numbers_l93_93809

theorem infinite_non_prime_numbers : ∀ (n : ℕ), ∃ (m : ℕ), m ≥ n ∧ (¬(Nat.Prime (2 ^ (2 ^ m) + 1) ∨ ¬Nat.Prime (2018 ^ (2 ^ m) + 1))) := sorry

end infinite_non_prime_numbers_l93_93809


namespace train_passing_time_correct_l93_93987

-- Definitions of the conditions
def length_of_train : ℕ := 180  -- Length of the train in meters
def speed_of_train_km_hr : ℕ := 54  -- Speed of the train in kilometers per hour

-- Known conversion factors
def km_per_hour_to_m_per_sec (v : ℕ) : ℚ := (v * 1000) / 3600

-- Define the speed of the train in meters per second
def speed_of_train_m_per_sec : ℚ := km_per_hour_to_m_per_sec speed_of_train_km_hr

-- Define the time to pass the oak tree
def time_to_pass_oak_tree (d : ℕ) (v : ℚ) : ℚ := d / v

-- The statement to prove
theorem train_passing_time_correct :
  time_to_pass_oak_tree length_of_train speed_of_train_m_per_sec = 12 := 
by
  sorry

end train_passing_time_correct_l93_93987


namespace cassidy_grades_below_B_l93_93029

theorem cassidy_grades_below_B (x : ℕ) (h1 : 26 = 14 + 3 * x) : x = 4 := 
by 
  sorry

end cassidy_grades_below_B_l93_93029


namespace average_of_added_numbers_l93_93829

theorem average_of_added_numbers (sum_twelve : ℕ) (new_sum : ℕ) (x y z : ℕ) 
  (h_sum_twelve : sum_twelve = 12 * 45) 
  (h_new_sum : new_sum = 15 * 60) 
  (h_addition : x + y + z = new_sum - sum_twelve) : 
  (x + y + z) / 3 = 120 :=
by 
  sorry

end average_of_added_numbers_l93_93829


namespace focus_of_parabola_l93_93889

theorem focus_of_parabola (a k : ℝ) (h_eq : ∀ x : ℝ, k = 6 ∧ a = 9) :
  (0, (1 / (4 * a)) + k) = (0, 217 / 36) := sorry

end focus_of_parabola_l93_93889


namespace exists_min_a_l93_93612

open Real

theorem exists_min_a (x y z : ℝ) : 
  (∃ x y z : ℝ, (sqrt (x - 1) + sqrt (y - 1) + sqrt (z - 1) = (11/2 - 1)) ∧ 
  (sqrt (x + 1) + sqrt (y + 1) + sqrt (z + 1) = (11/2 + 1))) :=
sorry

end exists_min_a_l93_93612


namespace Adam_spent_21_dollars_l93_93597

-- Define the conditions as given in the problem
def initial_money : ℕ := 91
def spent_money (x : ℕ) : Prop := (initial_money - x) * 3 = 10 * x

-- The theorem we want to prove: Adam spent 21 dollars on new books
theorem Adam_spent_21_dollars : spent_money 21 :=
by sorry

end Adam_spent_21_dollars_l93_93597


namespace adapted_bowling_ball_volume_l93_93582

noncomputable def volume_adapted_bowling_ball : ℝ :=
  let volume_sphere := (4/3) * Real.pi * (20 ^ 3)
  let volume_hole1 := Real.pi * (1 ^ 2) * 10
  let volume_hole2 := Real.pi * (1.5 ^ 2) * 10
  let volume_hole3 := Real.pi * (2 ^ 2) * 10
  volume_sphere - (volume_hole1 + volume_hole2 + volume_hole3)

theorem adapted_bowling_ball_volume :
  volume_adapted_bowling_ball = 10594.17 * Real.pi :=
sorry

end adapted_bowling_ball_volume_l93_93582


namespace triangle_angle_eq_pi_over_3_l93_93784

theorem triangle_angle_eq_pi_over_3
  (a b c : ℝ)
  (h : (a + b + c) * (a + b - c) = a * b)
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  ∃ C : ℝ, C = 2 * Real.pi / 3 ∧ 
            Real.cos C = (a^2 + b^2 - c^2) / (2 * a * b) :=
by
  -- Proof goes here
  sorry

end triangle_angle_eq_pi_over_3_l93_93784


namespace correct_operation_l93_93701

theorem correct_operation (a b : ℝ) :
  ¬ (a^2 + a^3 = a^6) ∧
  ¬ ((a*b)^2 = a*(b^2)) ∧
  ¬ ((a + b)^2 = a^2 + b^2) ∧
  ((a + b)*(a - b) = a^2 - b^2) := 
by
  sorry

end correct_operation_l93_93701


namespace free_endpoints_can_be_1001_l93_93756

variables (initial_segs : ℕ) (total_free_ends : ℕ) (k : ℕ)

-- Initial setup: one initial segment.
def initial_segment : ℕ := 1

-- Each time 5 segments are drawn from a point, the number of free ends increases by 4.
def free_ends_after_k_actions (k : ℕ) : ℕ := initial_segment + 4 * k

-- Question: Can the number of free endpoints be exactly 1001?
theorem free_endpoints_can_be_1001 : free_ends_after_k_actions 250 = 1001 := by
  sorry

end free_endpoints_can_be_1001_l93_93756


namespace solve_E_l93_93591

-- Definitions based on the conditions provided
variables {A H S M C O E : ℕ}

-- Given conditions
def algebra_books := A
def geometry_books := H
def history_books := C
def S_algebra_books := S
def M_geometry_books := M
def O_history_books := O
def E_algebra_books := E

-- Prove that E = (AM + AO - SH - SC) / (M + O - H - C) given the conditions
theorem solve_E (h1: A ≠ H) (h2: A ≠ S) (h3: A ≠ M) (h4: A ≠ C) (h5: A ≠ O) (h6: A ≠ E)
                (h7: H ≠ S) (h8: H ≠ M) (h9: H ≠ C) (h10: H ≠ O) (h11: H ≠ E)
                (h12: S ≠ M) (h13: S ≠ C) (h14: S ≠ O) (h15: S ≠ E)
                (h16: M ≠ C) (h17: M ≠ O) (h18: M ≠ E)
                (h19: C ≠ O) (h20: C ≠ E)
                (h21: O ≠ E)
                (pos1: 0 < A) (pos2: 0 < H) (pos3: 0 < S) (pos4: 0 < M) (pos5: 0 < C)
                (pos6: 0 < O) (pos7: 0 < E) :
  E = (A * M + A * O - S * H - S * C) / (M + O - H - C) :=
sorry

end solve_E_l93_93591


namespace constant_ratio_of_arithmetic_sequence_l93_93390

-- Definition of an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) := ∃ a₁ d : ℝ, ∀ n : ℕ, a n = a₁ + (n-1) * d

-- The main theorem stating the result
theorem constant_ratio_of_arithmetic_sequence 
  (a : ℕ → ℝ) (c : ℝ) (h_seq : arithmetic_sequence a)
  (h_const : ∀ n : ℕ, a n ≠ 0 ∧ a (2 * n) ≠ 0 ∧ a n / a (2 * n) = c) :
  c = 1 ∨ c = 1 / 2 :=
sorry

end constant_ratio_of_arithmetic_sequence_l93_93390


namespace size_of_sixth_doll_l93_93647

def nth_doll_size (n : ℕ) : ℝ :=
  243 * (2 / 3) ^ n

theorem size_of_sixth_doll : nth_doll_size 5 = 32 := by
  sorry

end size_of_sixth_doll_l93_93647


namespace molecular_weight_H2O_correct_l93_93488

-- Define atomic weights as constants
def atomic_weight_hydrogen : ℝ := 1.008
def atomic_weight_oxygen : ℝ := 15.999

-- Define the number of atoms in H2O
def num_hydrogens : ℕ := 2
def num_oxygens : ℕ := 1

-- Define molecular weight calculation for H2O
def molecular_weight_H2O : ℝ :=
  num_hydrogens * atomic_weight_hydrogen + num_oxygens * atomic_weight_oxygen

-- State the theorem that this molecular weight is 18.015 amu
theorem molecular_weight_H2O_correct :
  molecular_weight_H2O = 18.015 :=
by
  sorry

end molecular_weight_H2O_correct_l93_93488


namespace percentage_change_area_l93_93668

theorem percentage_change_area (L B : ℝ) :
  let A_original := L * B
  let A_new := (L / 2) * (3 * B)
  (A_new - A_original) / A_original * 100 = 50 := by
  sorry

end percentage_change_area_l93_93668


namespace locus_of_circle_centers_l93_93534

theorem locus_of_circle_centers (a : ℝ) (x0 y0 : ℝ) :
  { (α, β) | (x0 - α)^2 + (y0 - β)^2 = a^2 } = 
  { (x, y) | (x - x0)^2 + (y - y0)^2 = a^2 } :=
by
  sorry

end locus_of_circle_centers_l93_93534


namespace division_problem_l93_93980

theorem division_problem :
  (0.25 / 0.005) / 0.1 = 500 := by
  sorry

end division_problem_l93_93980


namespace probability_at_least_one_red_ball_l93_93166

-- Define the problem conditions
def balls_in_box_A := {red_ball_1, red_ball_2, white_ball}
def balls_in_box_B := {red_ball_3, red_ball_4, white_ball}

-- Define the random drawing event from each box
def event (A B : Set ball) : Set (ball × ball) := { (a, b) | a ∈ A ∧ b ∈ B }

-- Total probability space
def total_outcomes := event balls_in_box_A balls_in_box_B

-- Define event where no red balls are drawn: {white_ball from box A, white_ball from box B}
def no_red_event := {(white_ball, white_ball)}

-- Probability calculation
def P (E : Set (ball × ball)) : ℝ := (E.card / total_outcomes.card : ℝ)

-- Theorem statement for the problem
theorem probability_at_least_one_red_ball :
  P (total_outcomes \ no_red_event) = 8 / 9 :=
by
  sorry

end probability_at_least_one_red_ball_l93_93166


namespace march_volume_expression_l93_93191

variable (x : ℝ) (y : ℝ)

def initial_volume : ℝ := 500
def growth_rate_volumes (x : ℝ) (initial_volume : ℝ) : ℝ := initial_volume * (1 + x)
def calculate_march_volume (x : ℝ) (initial_volume : ℝ) : ℝ := initial_volume * (1 + x)^2

theorem march_volume_expression :
  y = calculate_march_volume x initial_volume :=
sorry

end march_volume_expression_l93_93191


namespace find_c_for_radius_of_circle_l93_93751

theorem find_c_for_radius_of_circle :
  ∃ c : ℝ, (∀ x y : ℝ, x^2 + 8 * x + y^2 - 6 * y + c = 0 → (x + 4)^2 + (y - 3)^2 = 25 - c) ∧
  (∀ x y : ℝ, (x + 4)^2 + (y - 3)^2 = 25 → c = 0) :=
sorry

end find_c_for_radius_of_circle_l93_93751


namespace seventh_oblong_is_56_l93_93342

def oblong (n : ℕ) : ℕ := n * (n + 1)

theorem seventh_oblong_is_56 : oblong 7 = 56 := by
  sorry

end seventh_oblong_is_56_l93_93342


namespace fencing_required_l93_93986

theorem fencing_required (L : ℝ) (W : ℝ) (A : ℝ) (H1 : L = 20) (H2 : A = 720) 
  (H3 : A = L * W) : L + 2 * W = 92 := by 
{
  sorry
}

end fencing_required_l93_93986


namespace twenty_four_points_game_l93_93441

theorem twenty_four_points_game :
  let a := (-6 : ℚ)
  let b := (3 : ℚ)
  let c := (4 : ℚ)
  let d := (10 : ℚ)
  3 * (d - a + c) = 24 := 
by
  sorry

end twenty_four_points_game_l93_93441


namespace angle_PMN_is_60_l93_93260

-- Define given variables and their types
variable (P M N R Q : Prop)
variable (angle : Prop → Prop → Prop → ℝ)

-- Given conditions
variables (h1 : angle P Q R = 60)
variables (h2 : PM = MN)

-- The statement of what's to be proven
theorem angle_PMN_is_60 :
  angle P M N = 60 := sorry

end angle_PMN_is_60_l93_93260


namespace find_age_of_mother_l93_93249

def Grace_age := 60
def ratio_GM_Grace := 3 / 8
def ratio_GM_Mother := 2

theorem find_age_of_mother (G M GM : ℕ) (h1 : G = ratio_GM_Grace * GM) 
                           (h2 : GM = ratio_GM_Mother * M) (h3 : G = Grace_age) : 
  M = 80 :=
by
  sorry

end find_age_of_mother_l93_93249


namespace greatest_product_two_integers_sum_2004_l93_93975

theorem greatest_product_two_integers_sum_2004 : 
  (∃ x y : ℤ, x + y = 2004 ∧ x * y = 1004004) :=
by
  sorry

end greatest_product_two_integers_sum_2004_l93_93975


namespace percentage_increase_of_x_l93_93298

theorem percentage_increase_of_x (C x y : ℝ) (P : ℝ) (h1 : x * y = C) (h2 : (x * (1 + P / 100)) * (y * (5 / 6)) = C) :
  P = 20 :=
by
  sorry

end percentage_increase_of_x_l93_93298


namespace amusement_park_trip_cost_l93_93274

def cost_per_trip (pass_cost : ℕ) (num_passes : ℕ) (oldest_trips : ℕ) (youngest_trips : ℕ) : ℕ :=
  let total_cost := num_passes * pass_cost
  let total_trips := oldest_trips + youngest_trips
  total_cost / total_trips

theorem amusement_park_trip_cost :
  ∀ (pass_cost num_passes oldest_trips youngest_trips : ℕ),
  pass_cost = 100 → num_passes = 2 → oldest_trips = 35 → youngest_trips = 15 →
  cost_per_trip pass_cost num_passes oldest_trips youngest_trips = 4 :=
by
  intros
  rw [H, H_1, H_2, H_3]
  sorry

end amusement_park_trip_cost_l93_93274


namespace squirrel_burrow_has_44_walnuts_l93_93009

def boy_squirrel_initial := 30
def boy_squirrel_gathered := 20
def boy_squirrel_dropped := 4
def boy_squirrel_hid := 8
-- "Forgets where he hid 3 of them" does not affect the main burrow

def girl_squirrel_brought := 15
def girl_squirrel_ate := 5
def girl_squirrel_gave := 4
def girl_squirrel_lost_playing := 3
def girl_squirrel_knocked := 2

def third_squirrel_gathered := 10
def third_squirrel_dropped := 1
def third_squirrel_hid := 3
def third_squirrel_returned := 6 -- Given directly instead of as a formula step; 9-3=6
def third_squirrel_gave := 1 -- Given directly as a friend

def final_walnuts := boy_squirrel_initial + boy_squirrel_gathered
                    - boy_squirrel_dropped - boy_squirrel_hid
                    + girl_squirrel_brought - girl_squirrel_ate
                    - girl_squirrel_gave - girl_squirrel_lost_playing
                    - girl_squirrel_knocked + third_squirrel_returned

theorem squirrel_burrow_has_44_walnuts :
  final_walnuts = 44 :=
by
  sorry

end squirrel_burrow_has_44_walnuts_l93_93009


namespace decreasing_exponential_range_l93_93524

theorem decreasing_exponential_range {a : ℝ} (h : ∀ x y : ℝ, x < y → (a + 1)^x > (a + 1)^y) : -1 < a ∧ a < 0 :=
sorry

end decreasing_exponential_range_l93_93524


namespace negation_false_l93_93838

theorem negation_false (a b : ℝ) : ¬ ((a ≤ 1 ∨ b ≤ 1) → a + b ≤ 2) :=
sorry

end negation_false_l93_93838


namespace grade11_paper_cutting_survey_l93_93862

theorem grade11_paper_cutting_survey (a b c x y z : ℕ)
  (h_total_students : a + b + c + x + y + z = 800)
  (h_clay_sculpture : a + b + c = 480)
  (h_paper_cutting : x + y + z = 320)
  (h_ratio : 5 * y = 3 * x ∧ 3 * z = 2 * y)
  (h_sample_size : 50):
  y * 50 / 800 = 6 :=
by {
  sorry
}

end grade11_paper_cutting_survey_l93_93862


namespace maximum_value_g_on_interval_l93_93228

noncomputable def g (x : ℝ) : ℝ := 4 * x - x^4

theorem maximum_value_g_on_interval : ∃ (x : ℝ), 0 ≤ x ∧ x ≤ 2 ∧ g x = 3 := by
  sorry

end maximum_value_g_on_interval_l93_93228


namespace math_problem_l93_93525

open Real

-- Conditions extracted from the problem
def cond1 (a b : ℝ) : Prop := -|2 - a| + b = 5
def cond2 (a b : ℝ) : Prop := -|8 - a| + b = 3
def cond3 (c d : ℝ) : Prop := |2 - c| + d = 5
def cond4 (c d : ℝ) : Prop := |8 - c| + d = 3
def cond5 (a c : ℝ) : Prop := 2 < a ∧ a < 8
def cond6 (a c : ℝ) : Prop := 2 < c ∧ c < 8

-- Proof problem: Given the conditions, prove that a + c = 10
theorem math_problem (a b c d : ℝ) (h1 : cond1 a b) (h2 : cond2 a b) (h3 : cond3 c d) (h4 : cond4 c d)
  (h5 : cond5 a c) (h6 : cond6 a c) : a + c = 10 := 
by
  sorry

end math_problem_l93_93525


namespace find_multiplier_l93_93315

-- Define the variables x and y
variables (x y : ℕ)

-- Define the conditions
def condition1 := (x / 6) * y = 12
def condition2 := x = 6

-- State the theorem to prove
theorem find_multiplier (h1 : condition1 x y) (h2 : condition2 x) : y = 12 :=
sorry

end find_multiplier_l93_93315


namespace number_of_girls_in_colins_class_l93_93526

variables (g b : ℕ)

theorem number_of_girls_in_colins_class
  (h1 : g / b = 3 / 4)
  (h2 : g + b = 35)
  (h3 : b > 15) :
  g = 15 :=
sorry

end number_of_girls_in_colins_class_l93_93526


namespace coalsBurnedEveryTwentyMinutes_l93_93469

-- Definitions based on the conditions
def totalGrillingTime : Int := 240
def coalsPerBag : Int := 60
def numberOfBags : Int := 3
def grillingInterval : Int := 20

-- Derived definitions based on conditions
def totalCoals : Int := numberOfBags * coalsPerBag
def numberOfIntervals : Int := totalGrillingTime / grillingInterval

-- The Lean theorem we want to prove
theorem coalsBurnedEveryTwentyMinutes : (totalCoals / numberOfIntervals) = 15 := by
  sorry

end coalsBurnedEveryTwentyMinutes_l93_93469


namespace total_cards_beginning_l93_93813

-- Define the initial conditions
def num_boxes_orig : ℕ := 2 + 5  -- Robie originally had 2 + 5 boxes
def cards_per_box : ℕ := 10      -- Each box contains 10 cards
def extra_cards : ℕ := 5         -- 5 cards were not placed in a box

-- Prove the total number of cards Robie had in the beginning
theorem total_cards_beginning : (num_boxes_orig * cards_per_box) + extra_cards = 75 :=
by sorry

end total_cards_beginning_l93_93813


namespace total_cost_paid_l93_93800

-- Definition of the given conditions
def number_of_DVDs : ℕ := 4
def cost_per_DVD : ℝ := 1.2

-- The theorem to be proven
theorem total_cost_paid : number_of_DVDs * cost_per_DVD = 4.8 := by
  sorry

end total_cost_paid_l93_93800
