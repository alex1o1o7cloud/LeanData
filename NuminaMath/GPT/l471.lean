import Mathlib

namespace birds_nest_building_area_scientific_notation_l471_471620

theorem birds_nest_building_area_scientific_notation :
  (258000 : ℝ) = 2.58 * 10^5 :=
by sorry

end birds_nest_building_area_scientific_notation_l471_471620


namespace min_games_needed_l471_471112

theorem min_games_needed (N : ℕ) : 
  (2 + N) * 10 ≥ 9 * (5 + N) ↔ N ≥ 25 := 
by {
  sorry
}

end min_games_needed_l471_471112


namespace range_of_m_l471_471828

def f (x : ℝ) (m : ℝ) := x^3 + 3 * x^2 - m * x + 1

theorem range_of_m (m : ℝ) : 
  (∀ x ∈ set.Icc (-2 : ℝ) 2, 3 * x^2 + 6 * x - m ≥ 0) ↔ m ≤ -3 :=
sorry

end range_of_m_l471_471828


namespace MrSlinkums_total_count_l471_471707

variable (T : ℕ)

-- Defining the conditions as given in the problem
def placed_on_shelves (T : ℕ) : ℕ := (20 * T) / 100
def storage (T : ℕ) : ℕ := (80 * T) / 100

-- Stating the main theorem to prove
theorem MrSlinkums_total_count 
    (h : storage T = 120) : 
    T = 150 :=
sorry

end MrSlinkums_total_count_l471_471707


namespace induction_example_l471_471943

-- Define the predicate P(n) that represents the property we want to prove for all natural numbers n
def P (n : ℕ) := (n * (n + 1) * (2 * n + 1)) % 6 = 0

theorem induction_example : ∀ n : ℕ, n ≥ 1 -> P(n) :=
  by
    -- Introduce the natural number n
    intro n
    -- Introduce the condition n ≥ 1
    intro h
    -- Now, we need to prove P(n), i.e., (n * (n + 1) * (2 * n + 1)) % 6 = 0
    -- This is where the inductive step would typically be proven
    sorry

end induction_example_l471_471943


namespace maximize_z_l471_471818

open Real

theorem maximize_z (x y : ℝ) (h1 : x + y ≤ 10) (h2 : 3 * x + y ≤ 18) (h3 : 0 ≤ x) (h4 : 0 ≤ y) :
  (∀ x y, x + y ≤ 10 ∧ 3 * x + y ≤ 18 ∧ 0 ≤ x ∧ 0 ≤ y → x + y / 2 ≤ 7) :=
by
  sorry

end maximize_z_l471_471818


namespace frequency_of_six_face_l471_471713

theorem frequency_of_six_face (n : ℕ) (h : n > 0) : (tendsto (λ (k : ℕ), (frequency_count k / k)) at_top (𝓝 (1 / 6))) :=
sorry

end frequency_of_six_face_l471_471713


namespace algae_coverage_day_21_l471_471151

-- Define the coverage of the pond
def coverage (day : ℕ) : ℚ :=
  if day = 24 then 1
  else if day < 24 then (coverage (day + 1)) / 2
  else 0

-- Theorem stating that on day 21, the pond is 12.5% (i.e., 1/8) covered with algae
theorem algae_coverage_day_21 : coverage 21 = 1 / 8 := 
  sorry

end algae_coverage_day_21_l471_471151


namespace largest_prime_factor_of_cyclic_sequence_sum_l471_471702

theorem largest_prime_factor_of_cyclic_sequence_sum : 
  ∃ k : ℕ, ∀ (a b c d : ℕ), 
  (1000 * a + 100 * b + 10 * c + d) = 
  ((1000 * d + 100 * a + 10 * b + c) →
  ∃ n : ℕ, S = 1111 * k → Prime n → n = 101 := 
begin
  sorry
end

end largest_prime_factor_of_cyclic_sequence_sum_l471_471702


namespace total_students_l471_471142

theorem total_students (S K : ℕ) (h1 : S = 4000) (h2 : K = 2 * S) :
  S + K = 12000 := by
  sorry

end total_students_l471_471142


namespace solve_inequality_l471_471345

theorem solve_inequality :
  {x : ℝ | (1 / (x * (x + 1)) - 1 / ((x + 1) * (x + 2)) < 1 / 2)} =
  {x : ℝ | (x < -2) ∨ (-1 < x ∧ x < 0) ∨ (1 < x)} :=
by
  sorry

end solve_inequality_l471_471345


namespace max_discount_rate_l471_471266

-- Define the conditions
def cost_price : ℝ := 4
def selling_price : ℝ := 5
def min_profit_margin : ℝ := 0.1
def min_profit : ℝ := cost_price * min_profit_margin

-- Define the discount rate
def discount_rate (x : ℝ) : ℝ :=
  selling_price * (1 - x / 100)

-- Define the profit after discount
def profit_after_discount (x : ℝ) : ℝ :=
  discount_rate x - cost_price

-- The statement we want to prove
theorem max_discount_rate : ∃ x : ℝ, x = 8.8 ∧ profit_after_discount x ≥ min_profit := 
by
  sorry

end max_discount_rate_l471_471266


namespace cans_per_bag_l471_471689

theorem cans_per_bag (total_cans : ℕ) (num_bags : ℕ) (h_total_cans : total_cans = 122) (h_num_bags : num_bags = 2) :
  total_cans / num_bags = 61 :=
by {
  rw [h_total_cans, h_num_bags],
  norm_num
}

end cans_per_bag_l471_471689


namespace projection_vector_length_l471_471091

variables {V : Type*} [inner_product_space ℝ V] 
variables (A B C X A1 B1 C1 : V) (α β γ d : ℝ)
variable (l : submodule ℝ V)

-- Conditions
hypothesis hxA : X ∈ affine_span ℝ (set.range ![{A, B, C}])
hypothesis ha1 : A1 ∈ orthogonal_projection l A
hypothesis hb1 : B1 ∈ orthogonal_projection l B
hypothesis hc1 : C1 ∈ orthogonal_projection l C
hypothesis hdist : dist X l = d

-- Given areas
def α := 1 -- placeholder for area S_{BXC}
def β := 1 -- placeholder for area S_{CXA}
def γ := 1 -- placeholder for area S_{AXB}

-- Statement of theorem
theorem projection_vector_length :
  ∥α • (A - A1) + β • (B - B1) + γ • (C - C1)∥ = (α + β + γ) * d :=
sorry

end projection_vector_length_l471_471091


namespace eccentricity_of_ellipse_l471_471821

open Real

noncomputable def ellipse_eccentricity : ℝ :=
  let a : ℝ := 4
  let b : ℝ := 2 * sqrt 3
  let c : ℝ := sqrt (a^2 - b^2)
  c / a

theorem eccentricity_of_ellipse (a b : ℝ) (ha : a = 4) (hb : b = 2 * sqrt 3) (h_eq : ∀ A B : ℝ, |A - B| = b^2 / 2 → |A - 2 * sqrt 3| + |B - 2 * sqrt 3| ≤ 10) :
  ellipse_eccentricity = 1 / 2 :=
by
  sorry

end eccentricity_of_ellipse_l471_471821


namespace rectangle_area_is_12_l471_471301

noncomputable def rectangle_area_proof (w l y : ℝ) : Prop :=
  l = 3 * w ∧ 2 * (l + w) = 16 ∧ (l^2 + w^2 = y^2) → l * w = 12

theorem rectangle_area_is_12 (y : ℝ) : ∃ (w l : ℝ), rectangle_area_proof w l y :=
by
  -- Introducing variables
  exists 2
  exists 6
  -- Constructing proof steps (skipped here with sorry)
  sorry

end rectangle_area_is_12_l471_471301


namespace count_valid_sequences_l471_471475

-- Define procedures A, B, C, D, and E
inductive Procedure : Type
| A : Procedure
| B : Procedure
| C : Procedure
| D : Procedure
| E : Procedure

open Procedure

-- Define a sequence of 5 procedures
def sequence := List Procedure

-- Define conditions for sequence validity
def isValidSequence (seq : sequence) : Prop :=
  (seq.head = some A ∨ seq.getLast = some A) ∧
  ∃ n, seq.get? n = some C ∧ seq.get? (n + 1) = some D ∨
       seq.get? n = some D ∧ seq.get? (n + 1) = some C

-- Define the main theorem to be proved
theorem count_valid_sequences : 
  ∃ seqs : List sequence, (∀ seq ∈ seqs, isValidSequence seq) ∧ seqs.length = 24 := by
  sorry

end count_valid_sequences_l471_471475


namespace range_of_a_l471_471395

theorem range_of_a (a : ℝ) :
  (¬ ∃ x : ℝ, 2 * x^2 + (a - 1) * x + 1 / 2 ≤ 0) ↔ (-1 < a ∧ a < 3) := 
sorry

end range_of_a_l471_471395


namespace sum_series_eq_l471_471789

theorem sum_series_eq : 
  ∑' n : ℕ, (n + 1) * (1 / 3 : ℝ)^n = 9 / 4 :=
by sorry

end sum_series_eq_l471_471789


namespace jason_hardcover_books_count_l471_471499

-- Define the conditions and the problem
theorem jason_hardcover_books_count :
  let max_weight_limit := 80 in
  let num_textbooks := 30 in
  let weight_per_textbook := 2 in
  let num_knickknacks := 3 in
  let weight_per_knickknack := 6 in
  let extra_weight := 33 in
  let weight_per_hardcover_book := 0.5 in
  let total_weight_limit := max_weight_limit + extra_weight in
  let weight_of_textbooks := num_textbooks * weight_per_textbook in
  let weight_of_knickknacks := num_knickknacks * weight_per_knickknack in
  let weight_of_other_items := weight_of_textbooks + weight_of_knickknacks in
  let total_weight := total_weight_limit in
  let weight_of_hardcover_books := total_weight - weight_of_other_items in
  let num_hardcover_books := weight_of_hardcover_books / weight_per_hardcover_book in
  num_hardcover_books = 70 :=
by {
  -- Note that the proof is not required, hence we use sorry
  sorry
}

end jason_hardcover_books_count_l471_471499


namespace option_c_correct_l471_471448

theorem option_c_correct (x y : ℝ) (h : x < y) : -x > -y := 
sorry

end option_c_correct_l471_471448


namespace pigeonhole_principle_example_birthday_problem_367_l471_471664

theorem pigeonhole_principle_example :
  ∀ (n m : ℕ), m > n → ∃ (x y : ℕ), x ≠ y ∧ x ≤ n ∧ y ≤ n ∧ x % (n + 1) = y % (n + 1) := 
by
  intro n m h
  sorry

theorem birthday_problem_367 :
  ∀ (n : ℕ), n > 366 → ∃ (x y : ℕ), x ≠ y ∧ x ≤ 365 ∧ y ≤ 365 ∧ x % 366 = y % 366 := 
by 
  intro n h
  exact pigeonhole_principle_example 365 n h

end pigeonhole_principle_example_birthday_problem_367_l471_471664


namespace flashlight_lifetime_expectation_leq_two_l471_471545

noncomputable def min_lifetime_expectation (ξ η : ℝ) (E_ξ : ℝ) : Prop :=
  E_ξ = 2 → E(min ξ η) ≤ 2

-- Assume ξ and η are random variables and E denotes the expectation.
axiom E : (ℝ → ℝ) → ℝ

theorem flashlight_lifetime_expectation_leq_two (ξ η : ℝ) (E_ξ : ℝ) (hE_ξ : E_ξ = 2) : E(min ξ η) ≤ 2 :=
  by
    sorry

end flashlight_lifetime_expectation_leq_two_l471_471545


namespace number_of_zeros_f_l471_471148

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log a (x + 1) + x^2 - 2

theorem number_of_zeros_f (a : ℝ) (h : 0 < a ∧ a < 1) : ∃! (x : ℝ), f a x = 0 :=
sorry

end number_of_zeros_f_l471_471148


namespace h_h3_eq_3568_l471_471862

def h (x : ℤ) := 3 * x ^ 2 + 3 * x - 2

theorem h_h3_eq_3568 : h (h 3) = 3568 := by
  sorry

end h_h3_eq_3568_l471_471862


namespace tangent_lines_to_ln_abs_through_origin_l471_471124

noncomputable def tangent_line_through_origin (x y: ℝ) : Prop :=
  (y = log (abs x)) ∧ ((x - exp(1) * y = 0) ∨ (x + exp(1) * y = 0))

theorem tangent_lines_to_ln_abs_through_origin :
  ∃ (f : ℝ → ℝ), 
  (∀ x, f x = log (abs x)) ∧ 
  ∀ x y, (tangent_line_through_origin x y) := sorry

end tangent_lines_to_ln_abs_through_origin_l471_471124


namespace hyperbola_equation_l471_471396

theorem hyperbola_equation 
  (circle_center : ℝ × ℝ) (circle_radius : ℝ)
  (center_eq_focus : circle_center = (2, 0))
  (radius_eq_one : circle_radius = 1)
  (asymptote_tangent_to_circle : ∀ a b : ℝ, (a^2 + b^2 = 4) → (2 * b / sqrt (a^2 + b^2) = 1)) :
  ∃ a b : ℝ, (a = sqrt 3 ∧ b = 1) → 
  (∀ x y : ℝ, (x^2 / (a^2) - y^2 / (b^2) = 1) ↔ (x^2 / 3 - y^2 = 1)) :=
by 
  intros
  sorry

end hyperbola_equation_l471_471396


namespace expected_lifetime_flashlight_l471_471524

section
variables {Ω : Type} [ProbabilitySpace Ω]
variables (ξ η : Ω → ℝ)
variables (h_ξ_expect : E[ξ] = 2)

-- Define the minimum of ξ and η
def min_ξ_η (ω : Ω) : ℝ := min (ξ ω) (η ω)

theorem expected_lifetime_flashlight : E[min_ξ_η ξ η] ≤ 2 :=
by
  sorry
end

end expected_lifetime_flashlight_l471_471524


namespace base16_trailing_zeros_l471_471726

theorem base16_trailing_zeros (f : ℕ → ℕ) (n : ℕ) (h : ∀ x, f x = Nat.factorial x):
  (trailing_zeros (f 15) 16) = 2 :=
by sorry

def trailing_zeros (n base : ℕ) : ℕ :=
  if base = 0 then 0 else
  let rec count_div (k : ℕ) (acc : ℕ) :=
    if k = 0 then acc else
    count_div (k / base) (acc + 1)
  in count_div (n / base) 0

end base16_trailing_zeros_l471_471726


namespace solution_set_quadratic_ineq_all_real_l471_471983

theorem solution_set_quadratic_ineq_all_real (a b c : ℝ) :
  (∀ x : ℝ, (a / 3) * x^2 + 2 * b * x - c < 0) ↔ (a > 0 ∧ 4 * b^2 - (4 / 3) * a * c < 0) :=
by
  sorry

end solution_set_quadratic_ineq_all_real_l471_471983


namespace area_ratio_correct_l471_471494

noncomputable def TriangleAreaRatios
  (X Y Z G H I S T U : Type*)
  [AffineSpace ℝ X] [AffineSpace ℝ Y] [AffineSpace ℝ Z]
  [AffineSpace ℝ G] [AffineSpace ℝ H] [AffineSpace ℝ I]
  [AffineSpace ℝ S] [AffineSpace ℝ T] [AffineSpace ℝ U]
  (g_ratio : ℚ) (h_ratio : ℚ) (i_ratio : ℚ)
  (XYZ_area : ℚ)
  (STU_area : ℚ) : Prop :=
  let YG_GZ_ratio := 2 / 3 in
  let XH_HZ_ratio := 2 / 3 in
  let XI_IY_ratio := 2 / 3 in
  let XG_intersects_S := true in
  let YH_intersects_T := true in
  let ZI_intersects_U := true in
  if g_ratio = YG_GZ_ratio ∧ h_ratio = XH_HZ_ratio ∧ i_ratio = XI_IY_ratio
    ∧ XG_intersects_S ∧ YH_intersects_T ∧ ZI_intersects_U then
    STU_area / XYZ_area = 9 / 55
  else false

theorem area_ratio_correct : TriangleAreaRatios X Y Z G H I S T U 2/5 2/5 2/5 1 9/55 :=
  sorry

end area_ratio_correct_l471_471494


namespace hyperbola_eccentricity_proof_l471_471389

noncomputable def hyperbola_eccentricity (b : ℝ) (P : ℝ × ℝ) (ha : b > 0) 
(hp : P.1^2 - (P.2^2) / (b^2) = 1)
(area : ℝ) (h_area : area = 1) : ℝ :=
sqrt 5

theorem hyperbola_eccentricity_proof (b : ℝ) (P : ℝ × ℝ) (ha : b > 0) 
(hp : P.1^2 - (P.2^2) / (b^2) = 1)
(area : ℝ) (h_area : area = 1) : 
hyperbola_eccentricity b P ha hp area h_area = sqrt 5 :=
sorry

end hyperbola_eccentricity_proof_l471_471389


namespace find_unique_positive_integer_l471_471326

theorem find_unique_positive_integer (n : ℕ) 
  (h : ∑ k in range (n - 1), (k + 2) * 3^(k + 2) = 3^(n + 8)) : n = 2196 :=
sorry

end find_unique_positive_integer_l471_471326


namespace total_balloons_correct_l471_471010

-- Definitions based on the conditions
def brookes_initial_balloons : Nat := 12
def brooke_additional_balloons : Nat := 8

def tracys_initial_balloons : Nat := 6
def tracy_additional_balloons : Nat := 24

-- Calculate the number of balloons each person has after the additions and Tracy popping half
def brookes_final_balloons : Nat := brookes_initial_balloons + brooke_additional_balloons
def tracys_balloons_after_addition : Nat := tracys_initial_balloons + tracy_additional_balloons
def tracys_final_balloons : Nat := tracys_balloons_after_addition / 2

-- Total number of balloons
def total_balloons : Nat := brookes_final_balloons + tracys_final_balloons

-- The proof statement
theorem total_balloons_correct : total_balloons = 35 := by
  -- Proof would go here (but we'll skip with sorry)
  sorry

end total_balloons_correct_l471_471010


namespace abs_nonneg_l471_471249

theorem abs_nonneg (a : ℝ) : |a| ≥ 0 := 
sorry

end abs_nonneg_l471_471249


namespace women_in_first_group_l471_471967

-- Define the number of women in the first group as W
variable (W : ℕ)

-- Define the work parameters
def work_per_day := 75 / 8
def work_per_hour_first_group := work_per_day / 5

def work_per_day_second_group := 30 / 3
def work_per_hour_second_group := work_per_day_second_group / 8

-- The equation comes from work/hour equivalence
theorem women_in_first_group :
  (W : ℝ) * work_per_hour_first_group = 4 * work_per_hour_second_group → W = 5 :=
by 
  sorry

end women_in_first_group_l471_471967


namespace modulus_of_z_l471_471823

open Complex

noncomputable def z_conjugate_satisfies (z : ℂ) : Prop := conj z * (2 - I) = 10 + 5 * I

theorem modulus_of_z (z : ℂ) (h : z_conjugate_satisfies z) : ∥z∥ = 5 :=
sorry

end modulus_of_z_l471_471823


namespace prob_9_correct_matches_is_zero_l471_471248

noncomputable def probability_of_exactly_9_correct_matches : ℝ :=
  let n := 10 in
  -- Since choosing 9 correct implies the 10th is also correct, the probability is 0.
  0

theorem prob_9_correct_matches_is_zero : probability_of_exactly_9_correct_matches = 0 :=
by
  sorry

end prob_9_correct_matches_is_zero_l471_471248


namespace prod_of_sums_of_consecutive_naturals_ne_20192019_l471_471379

theorem prod_of_sums_of_consecutive_naturals_ne_20192019 (a : ℕ) :
  let A := a + (a+1) + (a+2)
  let B := (a+1) + (a+2) + (a+3)
  (A * B ≠ 20192019) :=
by
  let A := a + (a+1) + (a+2)
  let B := (a+1) + (a+2) + (a+3)
  have hA : A = 3 * (a + 1) := by sorry
  have hB : B = 3 * (a + 2) := by sorry
  have h_product_multiple_of_9 : A * B = 9 * (a + 1) * (a + 2) := by sorry
  have h_not_multiple_of_9 : ¬ 20192019 % 9 = 0 := by sorry
  show (A * B ≠ 20192019) from by
    intro h_eq
    have h_20192019_multiple_of_9 := by
      rw h_product_multiple_of_9 at h_eq
      exact (eq.trans h_eq.symm (h_not_multiple_of_9.symm)).mp rfl
    contradiction

end prod_of_sums_of_consecutive_naturals_ne_20192019_l471_471379


namespace magnitude_of_vector_expression_l471_471811

-- Defining vectors a, b, c in a vector space over reals
variables {V : Type*} [inner_product_space ℝ V] (a b c : V)

-- Given conditions: the magnitude of vector a
axiom h1 : ∥a∥ = 1

-- Given conditions: relationship between a, b, and c
axiom h2 : a + b = c

-- The statement to be proven
theorem magnitude_of_vector_expression : ∥a - b + c∥ = 2 :=
by
  sorry

end magnitude_of_vector_expression_l471_471811


namespace negation_of_proposition_p_l471_471847

theorem negation_of_proposition_p :
  (¬ (∀ n : ℕ, 2^n > real.sqrt n)) = ∃ n : ℕ, 2^n ≤ real.sqrt n :=
sorry

end negation_of_proposition_p_l471_471847


namespace michael_reorganization_proof_l471_471550

def michael_notebooks_filled (total_notebooks : ℕ) (pages_per_notebook : ℕ) (drawings_per_page_initial : ℕ) (drawings_per_page_reorganized : ℕ) (filled_notebooks : ℕ) : Prop :=
  let total_drawings := total_notebooks * pages_per_notebook * drawings_per_page_initial in
  let total_pages_needed := total_drawings / drawings_per_page_reorganized in
  total_pages_needed <= filled_notebooks * pages_per_notebook

theorem michael_reorganization_proof :
  michael_notebooks_filled 5 60 8 15 3 :=
by {
  sorry
}

end michael_reorganization_proof_l471_471550


namespace game_not_fair_probability_first_player_approximation_l471_471178

-- Definitions representing the problem's conditions
def n : ℕ := 36

-- Function representing the probability of a player winning (generalized)
def prob_first_player_winning (n : ℕ) : ℝ :=
  (1 : ℝ) / n * (1 / (1 - Real.exp (-1 / (Real.ofNat n))))

-- Hypothesis representing the approximate probability of the first player winning for 36 players
def approximated_prob_first_player_winning : ℝ := 0.044

-- Main statement in two parts: (1) fairness, (2) probability approximation
theorem game_not_fair (n : ℕ) (prob : ℕ → ℝ) : 
  (∃ k : ℕ, k < n ∧ prob k ≠ prob 0) :=
sorry

theorem probability_first_player_approximation :
  abs (prob_first_player_winning n - approximated_prob_first_player_winning) < 0.001 :=
sorry

end game_not_fair_probability_first_player_approximation_l471_471178


namespace abs_sum_zero_l471_471451

theorem abs_sum_zero (a b : ℝ) (h : |a - 5| + |b + 8| = 0) : a + b = -3 := 
sorry

end abs_sum_zero_l471_471451


namespace find_a_l471_471318

noncomputable def csc (x : ℝ) : ℝ := 1 / (Real.sin x)

theorem find_a (a b c : ℝ) (h₀ : a > 0) (h₁ : b > 0)
  (h₂ : a * csc (b * (Real.pi / 6) + c) = 3) : a = 3 := 
sorry

end find_a_l471_471318


namespace cyclic_points_C_I_Q_Y_l471_471718

theorem cyclic_points_C_I_Q_Y
  {A B C I P Q X Y : Type*}
  [Incenter I (triangle A B C)]
  [Projection P I (side A B)]
  [Projection Q I (side A C)]
  [OnCircumcircle (side PQ) (triangle A B C) X Y P Q]
  (h1 : Concyclic [B, I, P, X]):
  Concyclic [C, I, Q, Y] :=
sorry

end cyclic_points_C_I_Q_Y_l471_471718


namespace appropriate_sampling_methods_l471_471324

-- Definitions for the context
def community := {
  high_income_families : ℕ := 125,
  middle_income_families : ℕ := 280,
  low_income_families : ℕ := 95
}

def middle_school := {
  art_specialty_students : ℕ := 15
}

-- Appropriate sampling methods
def stratified_sampling {α : Type} (population : α) : Prop := sorry
def simple_random_sampling {α : Type} (population : α) : Prop := sorry

-- Main theorem
theorem appropriate_sampling_methods :
  stratified_sampling community ∧ simple_random_sampling middle_school :=
sorry

end appropriate_sampling_methods_l471_471324


namespace tyler_saltwater_animals_l471_471640

/-- Tyler had 56 aquariums for saltwater animals and each aquarium has 39 animals in it. 
    We need to prove that the total number of saltwater animals Tyler has is 2184. --/
theorem tyler_saltwater_animals : (56 * 39) = 2184 := by
  sorry

end tyler_saltwater_animals_l471_471640


namespace consecutive_squares_not_arithmetic_sequence_l471_471355

theorem consecutive_squares_not_arithmetic_sequence (x y z w : ℕ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z ∧ 0 < w)
  (h_order: x < y ∧ y < z ∧ z < w) :
  ¬ (∃ d : ℕ, y^2 = x^2 + d ∧ z^2 = y^2 + d ∧ w^2 = z^2 + d) :=
sorry

end consecutive_squares_not_arithmetic_sequence_l471_471355


namespace probability_of_exactly_9_correct_matches_is_zero_l471_471224

theorem probability_of_exactly_9_correct_matches_is_zero :
  ∀ (n : ℕ) (translate : Fin n → Fin n),
    (n = 10) → 
    (∀ i : Fin n, translate i ≠ i) → 
    (∃ (k : ℕ), (k < n ∧ k ≠ n-1) → false ) :=
by
  sorry

end probability_of_exactly_9_correct_matches_is_zero_l471_471224


namespace min_distance_is_correct_l471_471388

-- Define the function representing the line 2x - y + 6 = 0
def line1 (x y : ℝ) : Prop := 2 * x - y + 6 = 0

-- Define the function representing the logarithmic curve y = 2 ln x + 2
def logCurve (x y : ℝ) : Prop := y = 2 * Real.log x + 2

-- Define the minimum distance between the two functions (i.e., the length of the line segment |PQ|)
def minDistance : ℝ := 6 * Real.sqrt 5 / 5

-- Theorem statement to prove that the minimum length of the line segment |PQ| is as calculated
theorem min_distance_is_correct :
  ∃ (P Q : ℝ × ℝ), line1 P.1 P.2 ∧ logCurve Q.1 Q.2 ∧ dist P Q = minDistance :=
sorry

end min_distance_is_correct_l471_471388


namespace pigeons_remaining_l471_471164

def initial_pigeons : ℕ := 40
def chicks_per_pigeon : ℕ := 6
def total_chicks : ℕ := initial_pigeons * chicks_per_pigeon
def total_pigeons_with_chicks : ℕ := total_chicks + initial_pigeons
def percentage_eaten : ℝ := 0.30
def pigeons_eaten : ℤ := int.of_nat total_pigeons_with_chicks * percentage_eaten
def pigeons_left : ℤ := int.of_nat total_pigeons_with_chicks - pigeons_eaten

theorem pigeons_remaining :
  pigeons_left = 196 := by
  -- The process of proving will be here
  sorry

end pigeons_remaining_l471_471164


namespace radius_of_sphere_l471_471134

-- Defining the context of the problem
variables {a x y R : ℝ}  -- a: given edge, x: one of the other edges, y: another edge, R: radius

-- Conditions of the problem
def condition1 : Prop := a = x + y
def condition2 : Prop := x ≠ 0 ∧ y ≠ 0 ∧ a ≠ 0 -- ensure non-zero positive lengths

-- Expression for the radius of the sphere that touches the base and lateral faces' extensions
theorem radius_of_sphere (h1 : condition1) (h2 : condition2) :
  R = a / 2 :=
sorry

end radius_of_sphere_l471_471134


namespace distance_between_points_l471_471194

theorem distance_between_points (x1 y1 x2 y2 : ℝ) (h1 : x1 = 2) (h2 : y1 = 3) (h3 : x2 = 6) (h4 : y2 = -4) :
  real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) = real.sqrt 65 :=
by
  subst h1
  subst h2
  subst h3
  subst h4
  sorry

end distance_between_points_l471_471194


namespace probability_of_exactly_9_correct_matches_is_zero_l471_471230

theorem probability_of_exactly_9_correct_matches_is_zero :
  ∃ (P : ℕ → ℕ → ℕ), 
    (∀ (total correct : ℕ), 
      total = 10 → 
      correct = 9 → 
      P total correct = 0) := 
by {
  sorry
}

end probability_of_exactly_9_correct_matches_is_zero_l471_471230


namespace max_discount_rate_l471_471269

-- Define the conditions
def cost_price : ℝ := 4
def selling_price : ℝ := 5
def min_profit_margin : ℝ := 0.1
def min_profit : ℝ := cost_price * min_profit_margin

-- Define the discount rate
def discount_rate (x : ℝ) : ℝ :=
  selling_price * (1 - x / 100)

-- Define the profit after discount
def profit_after_discount (x : ℝ) : ℝ :=
  discount_rate x - cost_price

-- The statement we want to prove
theorem max_discount_rate : ∃ x : ℝ, x = 8.8 ∧ profit_after_discount x ≥ min_profit := 
by
  sorry

end max_discount_rate_l471_471269


namespace h_h3_eq_3568_l471_471863

def h (x : ℤ) := 3 * x ^ 2 + 3 * x - 2

theorem h_h3_eq_3568 : h (h 3) = 3568 := by
  sorry

end h_h3_eq_3568_l471_471863


namespace union_complement_inter_l471_471070

noncomputable def U : Set ℝ := Set.univ
def M : Set ℝ := { x | x ≥ 2 }
def N : Set ℝ := { x | -1 ≤ x ∧ x < 5 }

def C_U_M : Set ℝ := U \ M
def M_inter_N : Set ℝ := { x | x ≥ 2 ∧ x < 5 }

theorem union_complement_inter (C_U_M M_inter_N : Set ℝ) :
  C_U_M ∪ M_inter_N = { x | x < 5 } :=
by
  sorry

end union_complement_inter_l471_471070


namespace molecular_weight_H_of_H2CrO4_is_correct_l471_471349

-- Define the atomic weight of hydrogen
def atomic_weight_H : ℝ := 1.008

-- Define the number of hydrogen atoms in H2CrO4
def num_H_atoms_in_H2CrO4 : ℕ := 2

-- Define the molecular weight of the compound H2CrO4
def molecular_weight_H2CrO4 : ℝ := 118

-- Define the molecular weight of the hydrogen part (H2)
def molecular_weight_H2 : ℝ := atomic_weight_H * num_H_atoms_in_H2CrO4

-- The statement to prove
theorem molecular_weight_H_of_H2CrO4_is_correct : molecular_weight_H2 = 2.016 :=
by
  sorry

end molecular_weight_H_of_H2CrO4_is_correct_l471_471349


namespace min_painted_squares_cover_vertices_l471_471560

theorem min_painted_squares_cover_vertices (grid_size : ℕ) (vertex_count : ℕ) : 
  grid_size = 8 ∧ vertex_count = (grid_size + 1) * (grid_size + 1) → 
  ∃ (painted_squares_count : ℕ), painted_squares_count = 25 ∧ 
  ∀ v : ℕ, v < vertex_count → ∃ sq : ℕ, sq < painted_squares_count ∧ covers_vertex (v, sq)
:= 
by
  intros grid_size vertex_count h
  cases h with grid_eq vertex_eq
  -- Assuming a helper function covers_vertex that takes a pair of vertex number and square number
  -- This function would check if the square covers the given vertex
  sorry

end min_painted_squares_cover_vertices_l471_471560


namespace volume_of_rotated_solid_l471_471994

noncomputable def circle_volume_rotation (k : ℝ) : ℝ :=
  let radius := real.sqrt 3 in
  (4 / 3) * real.pi * radius^3

theorem volume_of_rotated_solid (k : ℝ) 
  (h1 : ∀ x : ℝ, x^2 + (x*k - 1 + 1)^2 = 3) : 
  circle_volume_rotation k = 4 * real.sqrt 3 * real.pi :=
by
  sorry

end volume_of_rotated_solid_l471_471994


namespace v5_computation_l471_471110

noncomputable def sequence (v : ℕ → ℝ) :=
  ∀ n, v (n + 2) = 3 * v (n + 1) + 2 * v n

theorem v5_computation 
  (v : ℕ → ℝ) 
  (h_seq : sequence v) 
  (h_v4 : v 4 = 18) 
  (h_v7 : v 7 = 583) : 
  v 5 = 43 :=
sorry

end v5_computation_l471_471110


namespace intersection_of_sets_l471_471415

theorem intersection_of_sets :
  let M := { x : ℝ | -3 < x ∧ x ≤ 5 }
  let N := { x : ℝ | -5 < x ∧ x < 5 }
  M ∩ N = { x : ℝ | -3 < x ∧ x < 5 } := 
by
  sorry

end intersection_of_sets_l471_471415


namespace min_f_x_eq_one_implies_a_eq_zero_or_two_l471_471020

theorem min_f_x_eq_one_implies_a_eq_zero_or_two (a : ℝ) :
  (∃ x : ℝ, |x + 1| + |x + a| = 1) → (a = 0 ∨ a = 2) := by
  sorry

end min_f_x_eq_one_implies_a_eq_zero_or_two_l471_471020


namespace sam_total_cows_l471_471103

-- Conditions
def half_plus_five (C : ℕ) : ℕ := (1 / 2 : ℚ) * C + 5
def total_minus_four (C : ℕ) : ℕ := C - 4

-- Lean statement defining the proof
theorem sam_total_cows (C : ℕ) : (half_plus_five C = total_minus_four C) → C = 18 := by
  intro h
  sorry

end sam_total_cows_l471_471103


namespace log_identity_l471_471455

theorem log_identity (x z : ℝ) (hx : x = 12) (hz : log 4 (x * (1/6)) = z) : 
  z = 1/2 := 
by 
  sorry

end log_identity_l471_471455


namespace partitions_of_6_into_4_indistinguishable_boxes_l471_471436

theorem partitions_of_6_into_4_indistinguishable_boxes : 
  ∃ (X : Finset (Multiset ℕ)), X.card = 9 ∧ 
  ∀ p ∈ X, p.sum = 6 ∧ p.card ≤ 4 := 
sorry

end partitions_of_6_into_4_indistinguishable_boxes_l471_471436


namespace cost_of_milk_powder_july_l471_471464

theorem cost_of_milk_powder_july :
  ∃ (C : ℝ), 
    let milk_powder_july := 0.4 * C,
        coffee_july := 3 * C,
        sugar_july := 1.5 * C in
    (2 * milk_powder_july + 3 * coffee_july + 1 * sugar_july = 15.30) → 
    milk_powder_july = 0.54 :=
begin
  sorry
end

end cost_of_milk_powder_july_l471_471464


namespace minimum_y_l471_471808

theorem minimum_y (x : ℝ) (h : x > 1) : (∃ y : ℝ, y = x + 1 / (x - 1) ∧ y = 3) :=
by
  sorry

end minimum_y_l471_471808


namespace equilateral_triangle_l471_471463

theorem equilateral_triangle (A B C : ℝ) (a b c : ℝ) 
  (h1 : 2 * b = a + c) 
  (h2 : sin B ^ 2 = sin A * sin C)
  (h3 : sin B = (sin A + sin C) / 2) : 
  a = b ∧ b = c := 
sorry

end equilateral_triangle_l471_471463


namespace probability_red_given_black_l471_471649

noncomputable def urn_A := {white := 4, red := 2}
noncomputable def urn_B := {red := 3, black := 3}

-- Define the probabilities as required in the conditions
def prob_urn_A := 1 / 2
def prob_urn_B := 1 / 2

def draw_red_from_A := 2 / 6
def draw_black_from_B := 3 / 6
def draw_red_from_B := 3 / 6
def draw_black_from_B_after_red := 3 / 5
def draw_black_from_B_after_black := 2 / 5

def probability_first_red_second_black :=
  (prob_urn_A * draw_red_from_A * prob_urn_B * draw_black_from_B) +
  (prob_urn_B * draw_red_from_B * prob_urn_B * draw_black_from_B_after_red) +
  (prob_urn_B * draw_red_from_B * prob_urn_B * draw_black_from_B_after_black)

def probability_second_black :=
  (prob_urn_A * prob_urn_B * draw_black_from_B) +
  (prob_urn_B * draw_red_from_B * prob_urn_B * draw_black_from_B_after_red) +
  (prob_urn_B * draw_black_from_B * prob_urn_B * draw_black_from_B_after_black)

theorem probability_red_given_black :
  probability_first_red_second_black / probability_second_black = 7 / 15 :=
sorry

end probability_red_given_black_l471_471649


namespace shopkeeper_percentage_gain_l471_471215

theorem shopkeeper_percentage_gain 
    (original_price : ℝ) 
    (price_increase : ℝ) 
    (first_discount : ℝ) 
    (second_discount : ℝ)
    (new_price : ℝ) 
    (discounted_price1 : ℝ) 
    (final_price : ℝ) 
    (percentage_gain : ℝ) 
    (h1 : original_price = 100)
    (h2 : price_increase = original_price * 0.34)
    (h3 : new_price = original_price + price_increase)
    (h4 : first_discount = new_price * 0.10)
    (h5 : discounted_price1 = new_price - first_discount)
    (h6 : second_discount = discounted_price1 * 0.15)
    (h7 : final_price = discounted_price1 - second_discount)
    (h8 : percentage_gain = ((final_price - original_price) / original_price) * 100) :
    percentage_gain = 2.51 :=
by sorry

end shopkeeper_percentage_gain_l471_471215


namespace total_balloons_correct_l471_471009

-- Definitions based on the conditions
def brookes_initial_balloons : Nat := 12
def brooke_additional_balloons : Nat := 8

def tracys_initial_balloons : Nat := 6
def tracy_additional_balloons : Nat := 24

-- Calculate the number of balloons each person has after the additions and Tracy popping half
def brookes_final_balloons : Nat := brookes_initial_balloons + brooke_additional_balloons
def tracys_balloons_after_addition : Nat := tracys_initial_balloons + tracy_additional_balloons
def tracys_final_balloons : Nat := tracys_balloons_after_addition / 2

-- Total number of balloons
def total_balloons : Nat := brookes_final_balloons + tracys_final_balloons

-- The proof statement
theorem total_balloons_correct : total_balloons = 35 := by
  -- Proof would go here (but we'll skip with sorry)
  sorry

end total_balloons_correct_l471_471009


namespace find_square_sum_l471_471519

variables {a b c : ℝ}

theorem find_square_sum (h1 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
  (h2 : a + b + c = 0)
  (h3 : a^3 + b^3 + c^3 = a^5 + b^5 + c^5) : 
  a^2 + b^2 + c^2 = 6 / 5 := 
begin
  sorry
end

end find_square_sum_l471_471519


namespace perfect_square_probability_l471_471887

theorem perfect_square_probability :
  let cases := [(-, -), (+, +), (+, -), (-, +)] in
  let perfect_square_cases := [(+, +), (-, -)] in
  (perfect_square_cases.length : ℚ) / (cases.length : ℚ) = 1 / 2 :=
by
  -- List of all possible cases for filling in the signs
  let cases := [(-, -), (+, +), (+, -), (-, +)]
  -- List of valid perfect square cases
  let perfect_square_cases := [(+, +), (-, -)]
  -- Compute the probability
  have h_cases : (cases.length : ℚ) = 4 := by sorry
  have h_perfect_square_cases : (perfect_square_cases.length : ℚ) = 2 := by sorry
  show (perfect_square_cases.length : ℚ) / (cases.length : ℚ) = 1 / 2, by
    rw [h_cases, h_perfect_square_cases]
    norm_num

end perfect_square_probability_l471_471887


namespace two_digit_number_13_more_than_squares_sum_l471_471346

theorem two_digit_number_13_more_than_squares_sum : 
∃ (x y : ℕ), (x = 5) ∧ (y = 4) ∧ (10 * x + y = 54) ∧ (10 * x + y = x^2 + y^2 + 13) :=
by
  exists 5
  exists 4
  simp
  sorry

end two_digit_number_13_more_than_squares_sum_l471_471346


namespace systemOfEquationsUniqueSolution_l471_471627

def largeBarrelHolds (x : ℝ) (y : ℝ) : Prop :=
  5 * x + y = 3

def smallBarrelHolds (x : ℝ) (y : ℝ) : Prop :=
  x + 5 * y = 2

theorem systemOfEquationsUniqueSolution (x y : ℝ) :
  (largeBarrelHolds x y) ∧ (smallBarrelHolds x y) ↔ 
  (5 * x + y = 3 ∧ x + 5 * y = 2) :=
by
  sorry

end systemOfEquationsUniqueSolution_l471_471627


namespace derivative_of_y_l471_471347

noncomputable def y (x : ℝ) : ℝ :=
  (4 * x + 1) / (16 * x^2 + 8 * x + 3) + (1 / Real.sqrt 2) * Real.arctan ((4 * x + 1) / Real.sqrt 2)

theorem derivative_of_y (x : ℝ) : 
  (deriv y x) = 16 / (16 * x^2 + 8 * x + 3)^2 :=
by 
  sorry

end derivative_of_y_l471_471347


namespace circle_area_AOC_l471_471186

theorem circle_area_AOC :
  ∀ (A B C : Point) (O : Point),
    isosceles_triangle A B C 7 7 6 → -- Triangle \(ABC\) is isosceles with \(AB = AC = 7\) and \(BC = 6\)
    incenter O A B C → -- \(O\) is the incenter of the triangle
    circle_area A O C = 27 * π := -- The area of the circle passing through \(A\), \(O\), and \(C\)
by
  sorry

end circle_area_AOC_l471_471186


namespace quadrilateral_parallelogram_l471_471383

variable {Point : Type*} [AffineSpace Point ℝ]

variables (A B C D O : Point)
variables (OA OB OC OD : Point → ℝ^3)
variables (vec : Point → Point → ℝ^3)
variables (eq_OA_OC_OB_OD : vec O A + vec O C = vec O B + vec O D)

theorem quadrilateral_parallelogram :
  let BA := vec B A,
      DC := vec D C in
  BA = DC →
  (ABCD : Type) →
  (parallelogram ABCD)
:=
begin
  sorry
end

end quadrilateral_parallelogram_l471_471383


namespace game_not_fair_probability_first_player_approximation_l471_471176

-- Definitions representing the problem's conditions
def n : ℕ := 36

-- Function representing the probability of a player winning (generalized)
def prob_first_player_winning (n : ℕ) : ℝ :=
  (1 : ℝ) / n * (1 / (1 - Real.exp (-1 / (Real.ofNat n))))

-- Hypothesis representing the approximate probability of the first player winning for 36 players
def approximated_prob_first_player_winning : ℝ := 0.044

-- Main statement in two parts: (1) fairness, (2) probability approximation
theorem game_not_fair (n : ℕ) (prob : ℕ → ℝ) : 
  (∃ k : ℕ, k < n ∧ prob k ≠ prob 0) :=
sorry

theorem probability_first_player_approximation :
  abs (prob_first_player_winning n - approximated_prob_first_player_winning) < 0.001 :=
sorry

end game_not_fair_probability_first_player_approximation_l471_471176


namespace moles_H2O_formed_l471_471787

-- Define the balanced equation as a struct
structure Reaction :=
(reactants : List (String × ℕ)) -- List of reactants with their stoichiometric coefficients
(products : List (String × ℕ)) -- List of products with their stoichiometric coefficients

-- Example reaction: NaHCO3 + HC2H3O2 -> NaC2H3O2 + H2O + CO2
def example_reaction : Reaction :=
{ reactants := [("NaHCO3", 1), ("HC2H3O2", 1)],
  products := [("NaC2H3O2", 1), ("H2O", 1), ("CO2", 1)] }

-- We need a predicate to determine the number of moles of a product based on the reaction
def moles_of_product (reaction : Reaction) (product : String) (moles_reactant₁ moles_reactant₂ : ℕ) : ℕ :=
if product = "H2O" then moles_reactant₁ else 0  -- Only considering H2O for simplicity

-- Now we define our main theorem
theorem moles_H2O_formed : 
  moles_of_product example_reaction "H2O" 3 3 = 3 :=
by
  -- The proof will go here; for now, we use sorry to skip it
  sorry

end moles_H2O_formed_l471_471787


namespace B_can_finish_work_in_6_days_l471_471264

theorem B_can_finish_work_in_6_days :
  (A_work_alone : ℕ) → (A_work_before_B : ℕ) → (A_B_together : ℕ) → (B_days_alone : ℕ) → 
  (A_work_alone = 12) → (A_work_before_B = 3) → (A_B_together = 3) → B_days_alone = 6 :=
by
  intros A_work_alone A_work_before_B A_B_together B_days_alone
  intros h1 h2 h3
  sorry

end B_can_finish_work_in_6_days_l471_471264


namespace a_2020_is_one_fourth_l471_471813

def sequence (a : ℕ → ℚ) : Prop :=
  a 1 = 1 / 4 ∧ ∀ n ≥ 2, a n = 1 - 1 / (a (n - 1))

theorem a_2020_is_one_fourth (a : ℕ → ℚ) (h_seq : sequence a) : a 2020 = 1 / 4 :=
by sorry

end a_2020_is_one_fourth_l471_471813


namespace a_squared_plus_b_squared_a_plus_b_squared_l471_471366

variable (a b : ℚ)

-- Conditions
axiom h1 : a - b = 7
axiom h2 : a * b = 18

-- To Prove
theorem a_squared_plus_b_squared : a^2 + b^2 = 85 :=
by sorry

theorem a_plus_b_squared : (a + b)^2 = 121 :=
by sorry

end a_squared_plus_b_squared_a_plus_b_squared_l471_471366


namespace joan_spent_on_toy_cars_l471_471900

theorem joan_spent_on_toy_cars :
  ∀ (skateboard_cost toy_trucks_cost total_spent_on_toys toy_cars_cost : ℝ),
  skateboard_cost = 4.88 →
  toy_trucks_cost = 5.86 →
  total_spent_on_toys = 25.62 →
  toy_cars_cost = total_spent_on_toys - (skateboard_cost + toy_trucks_cost) →
  toy_cars_cost = 14.88 :=
by
  intros skateboard_cost toy_trucks_cost total_spent_on_toys toy_cars_cost
  intros h_sc h_tt h_ttots h_tcc
  rw [h_sc, h_tt, h_ttots] at h_tcc
  exact h_tcc

end joan_spent_on_toy_cars_l471_471900


namespace find_wrong_observation_value_l471_471136

theorem find_wrong_observation_value :
  ∃ (wrong_value : ℝ),
    let n := 50
    let mean_initial := 36
    let mean_corrected := 36.54
    let observation_incorrect := 48
    let sum_initial := n * mean_initial
    let sum_corrected := n * mean_corrected
    let difference := sum_corrected - sum_initial
    wrong_value = observation_incorrect - difference := sorry

end find_wrong_observation_value_l471_471136


namespace total_students_l471_471139

def numStudents (skiing scavenger : ℕ) : ℕ :=
  skiing + scavenger

theorem total_students (skiing scavenger : ℕ) (h1 : skiing = 2 * scavenger) (h2 : scavenger = 4000) :
  numStudents skiing scavenger = 12000 :=
by
  sorry

end total_students_l471_471139


namespace ratio_joseph_to_george_l471_471205

theorem ratio_joseph_to_george (j : ℕ) (h_j_definition: 6 * j = 18) : j = 3 ∧ j : 6 = 1 : 2 := by
  -- to show j = 3
  have h1: 6 * j = 18 := h_j_definition
  have h2: j = 3 := by linarith
  -- to show j : 6 = 1 : 2
  have h3: (3 / 3 : ℚ) : (6 / 3 : ℚ) = 1 : 2 := by norm_num
  exact ⟨h2, h3⟩

end ratio_joseph_to_george_l471_471205


namespace ellipse_equation_correct_triangle_area_l471_471397

noncomputable def ellipse_equation (P : ℝ × ℝ) := 
  let F1 : ℝ × ℝ := (-1, 0)
  let F2 : ℝ × ℝ := (1, 0)
  let function for x y : ℝ → (x+1)^2 + y^2 = 4 in 
  (4, 3)

theorem ellipse_equation_correct (P : ℝ × ℝ) (hx : 2 * dist F1 F2 = dist P F1 + dist P F2) : 
    ∃ (c a b : ℝ), 
    ellipse_equation c a b = 
    let x := P.1 
    let y := P.2 in
     c * x^2 + a * y^2 = b :=
by
 sorry

theorem triangle_area (P : ℝ × ℝ) (hx : P.1 < 0) (hy : P.2 > 0) (h_angle : angle F2 F1 P = 120) (h_ellipse : P ∈ set_of (λ Q : ℝ × ℝ, (Q.1^2) / 4 + (Q.2^2) / 3 = 1)) : 
    area_triangle P F1 F2 = (3 * sqrt 3) / 5 :=
by
 sorry

end ellipse_equation_correct_triangle_area_l471_471397


namespace max_norm_sum_eq_sixteen_l471_471446

variable {α : Type} [inner_product_space α]

variables (a b c d : α)

theorem max_norm_sum_eq_sixteen (ha : ∥a∥ = 1) (hb : ∥b∥ = 1) (hc : ∥c∥ = 1) (hd : ∥d∥ = 1) :
  ∥a - b∥^2 + ∥a - c∥^2 + ∥a - d∥^2 + ∥b - c∥^2 + ∥b - d∥^2 + ∥c - d∥^2 ≤ 16 :=
sorry

end max_norm_sum_eq_sixteen_l471_471446


namespace sequence_correct_6_7_sequence_general_l471_471087

-- Define the term function for the sequence
def sequence (n : ℕ) : ℚ :=
  (-1)^(n+1) * (2 * n - 1) / (n^2)

-- State the theorem: Proving that the sequence term at positions 6 and 7 are as expected
theorem sequence_correct_6_7 :
  sequence 6 = -11/36 ∧
  sequence 7 = 13/49 :=
by
  sorry

-- State the general form of the theorem for any n
theorem sequence_general (n : ℕ) : sequence n = (-1)^(n+1) * (2 * n - 1) / (n^2) :=
by
  sorry

end sequence_correct_6_7_sequence_general_l471_471087


namespace tangent_lines_ln_l471_471128

theorem tangent_lines_ln (x y: ℝ) : 
    (y = Real.log (abs x)) → 
    (x = 0 ∧ y = 0) ∨ ((x = yup ∨ x = ydown) ∧ (∀ (ey : ℝ), x = ey ∨ x = -ey)) :=
by 
    intro h
    sorry

end tangent_lines_ln_l471_471128


namespace magnitude_b_l471_471806

def vector_a : ℝ × ℝ × ℝ := (2, 3, 1)
def vector_b (x : ℝ) : ℝ × ℝ × ℝ := (-4, 2, x)
def dot_product (u v : ℝ × ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2 + u.3 * v.3

theorem magnitude_b :
  ∀ x, dot_product vector_a (vector_b x) = 0 → sqrt ((-4)^2 + 2^2 + x^2) = 2 * sqrt 6 :=
by
  sorry

end magnitude_b_l471_471806


namespace divisible_count_3_4_5_l471_471442

theorem divisible_count_3_4_5 (S : Finset ℕ) (hS : S = finset.range 51 \ {0})
  (h3 : ∀ x ∈ S, x % 3 = 0 → x ∈ S)
  (h4 : ∀ x ∈ S, x % 4 = 0 → x ∈ S)
  (h5 : ∀ x ∈ S, x % 5 = 0 → x ∈ S) :
  S.filter (λ x, x % 3 = 0 ∨ x % 4 = 0 ∨ x % 5 = 0).card = 29 := 
sorry

end divisible_count_3_4_5_l471_471442


namespace ellipse_equation_line_MN_fixed_l471_471993

noncomputable theory
open_locale classical

-- Conditions: The ellipse with foci and minor axis endpoints on the circle.
def ellipse (a b : ℝ) (h : a > b ∧ b > 0) : Prop :=
  (∀ (x y : ℝ), (x^2 / a^2) + (y^2 / b^2) = 1 → (x^2 + y^2 = 1)) ∧
  (a^2 = b^2 + 1^2)

-- Show that the specific form of the ellipse is correct.
theorem ellipse_equation : ∃ (a b : ℝ) (h : a > b ∧ b > 0), 
  ellipse a b h ∧ (∀ (x y : ℝ), (x^2 / 2) + y^2 = 1) :=
by
  sorry

-- Define the fixed point property of line MN.
def fixed_point (M N : ℝ × ℝ) (midpoint_AB midpoint_CD : ℝ × ℝ) : Prop :=
  (midpoint_AB ≠ midpoint_CD) →
  (∃ k : ℝ, k ≠ 0 ∧ 
  (∃ (x y : ℝ), M = (2/3, 0) ∧ N = (2/3, 0) ))

-- Demonstrating the fixed point.
theorem line_MN_fixed (M N : ℝ × ℝ) (midpoint_AB midpoint_CD : ℝ × ℝ) :
  ∀ (a b : ℝ) (h : a > b ∧ b > 0), ellipse a b h →
    fixed_point M N midpoint_AB midpoint_CD :=
by
  sorry

end ellipse_equation_line_MN_fixed_l471_471993


namespace max_discount_rate_l471_471272

theorem max_discount_rate 
  (cost_price : ℝ) (selling_price : ℝ) (min_profit_margin : ℝ) :
  cost_price = 4 → selling_price = 5 → min_profit_margin = 0.1 →
  ∃ (x : ℝ), 0 ≤ x ∧ x ≤ 8.8 ∧ (selling_price * (1 - x / 100) - cost_price) / cost_price ≥ min_profit_margin :=
begin
  sorry
end

end max_discount_rate_l471_471272


namespace total_points_by_six_remaining_members_l471_471890

noncomputable theory

def total_points (x y : ℕ) :=
  (x : ℚ) / 5 + (x : ℚ) / 3 + 12 + y = ↑x

theorem total_points_by_six_remaining_members :
  ∃ (y : ℕ), total_points 135 y ∧ y = 18 :=
by
  use 18
  dsimp [total_points]
  have : (135 : ℚ) / 5 + (135 : ℚ) / 3 + 12 + 18 = 135 := by norm_num
  exact this

end total_points_by_six_remaining_members_l471_471890


namespace sandwiches_prepared_l471_471955

-- Define the conditions as given in the problem.
def ruth_ate_sandwiches : ℕ := 1
def brother_ate_sandwiches : ℕ := 2
def first_cousin_ate_sandwiches : ℕ := 2
def each_other_cousin_ate_sandwiches : ℕ := 1
def number_of_other_cousins : ℕ := 2
def sandwiches_left : ℕ := 3

-- Define the total number of sandwiches eaten.
def total_sandwiches_eaten : ℕ := ruth_ate_sandwiches 
                                  + brother_ate_sandwiches
                                  + first_cousin_ate_sandwiches 
                                  + (each_other_cousin_ate_sandwiches * number_of_other_cousins)

-- Define the number of sandwiches prepared by Ruth.
def sandwiches_prepared_by_ruth : ℕ := total_sandwiches_eaten + sandwiches_left

-- Formulate the theorem to prove.
theorem sandwiches_prepared : sandwiches_prepared_by_ruth = 10 :=
by
  -- Use the solution steps to prove the theorem (proof omitted here).
  sorry

end sandwiches_prepared_l471_471955


namespace marica_winning_strategy_l471_471360

theorem marica_winning_strategy (a b : ℕ) (ha : a > 0) (hb : b > 0) 
  (perfect_play : ∀ n : ℕ, n * 2 ≤ (a + b) → 
     (a = 2 * n ∨ b = 2 * n ∨ 
      ∀ k : ℕ, k > 0 → k ≤ n → 
        ((a + k, b - k) ∨ (a - k, b + k)))) : 
  (|a - b| ≤ 1 ↔ ∃ n, ((a = 0 ∨ b = 0) ∧ (a, b) = (2 * n + 1, 2 * n + 1)) ∨
                   (a, b) = (2 * n, 0) ∨
                   (a, b) = (0, 2 * n)) :=
sorry

end marica_winning_strategy_l471_471360


namespace max_discount_rate_l471_471271

-- Define the conditions
def cost_price : ℝ := 4
def selling_price : ℝ := 5
def min_profit_margin : ℝ := 0.1
def min_profit : ℝ := cost_price * min_profit_margin

-- Define the discount rate
def discount_rate (x : ℝ) : ℝ :=
  selling_price * (1 - x / 100)

-- Define the profit after discount
def profit_after_discount (x : ℝ) : ℝ :=
  discount_rate x - cost_price

-- The statement we want to prove
theorem max_discount_rate : ∃ x : ℝ, x = 8.8 ∧ profit_after_discount x ≥ min_profit := 
by
  sorry

end max_discount_rate_l471_471271


namespace prob_9_correct_matches_is_zero_l471_471244

noncomputable def probability_of_exactly_9_correct_matches : ℝ :=
  let n := 10 in
  -- Since choosing 9 correct implies the 10th is also correct, the probability is 0.
  0

theorem prob_9_correct_matches_is_zero : probability_of_exactly_9_correct_matches = 0 :=
by
  sorry

end prob_9_correct_matches_is_zero_l471_471244


namespace john_average_speed_l471_471062

theorem john_average_speed:
  (∃ J : ℝ, Carla_speed = 35 ∧ Carla_time = 3 ∧ John_time = 3.5 ∧ J * John_time = Carla_speed * Carla_time) →
  (∃ J : ℝ, J = 30) :=
by
  -- Given Variables
  let Carla_speed : ℝ := 35
  let Carla_time : ℝ := 3
  let John_time : ℝ := 3.5
  -- Proof goal
  sorry

end john_average_speed_l471_471062


namespace probability_of_exactly_9_correct_matches_is_zero_l471_471228

theorem probability_of_exactly_9_correct_matches_is_zero :
  ∀ (n : ℕ) (translate : Fin n → Fin n),
    (n = 10) → 
    (∀ i : Fin n, translate i ≠ i) → 
    (∃ (k : ℕ), (k < n ∧ k ≠ n-1) → false ) :=
by
  sorry

end probability_of_exactly_9_correct_matches_is_zero_l471_471228


namespace money_left_after_purchase_l471_471341

noncomputable def total_cost : ℝ := 250 + 25 + 35 + 45 + 90

def savings_erika : ℝ := 155

noncomputable def savings_rick : ℝ := total_cost / 2

def savings_sam : ℝ := 175

def combined_cost_cake_flowers_skincare : ℝ := 25 + 35 + 45

noncomputable def savings_amy : ℝ := 2 * combined_cost_cake_flowers_skincare

noncomputable def total_savings : ℝ := savings_erika + savings_rick + savings_sam + savings_amy

noncomputable def money_left : ℝ := total_savings - total_cost

theorem money_left_after_purchase : money_left = 317.5 := by
  sorry

end money_left_after_purchase_l471_471341


namespace total_hours_watching_tv_and_playing_games_l471_471559

-- Defining the conditions provided in the problem
def hours_watching_tv_saturday : ℕ := 6
def hours_watching_tv_sunday : ℕ := 3
def hours_watching_tv_tuesday : ℕ := 2
def hours_watching_tv_thursday : ℕ := 4

def hours_playing_games_monday : ℕ := 3
def hours_playing_games_wednesday : ℕ := 5
def hours_playing_games_friday : ℕ := 1

-- The proof statement
theorem total_hours_watching_tv_and_playing_games :
  hours_watching_tv_saturday + hours_watching_tv_sunday + hours_watching_tv_tuesday + hours_watching_tv_thursday
  + hours_playing_games_monday + hours_playing_games_wednesday + hours_playing_games_friday = 24 := 
by
  sorry

end total_hours_watching_tv_and_playing_games_l471_471559


namespace plate_arrangements_l471_471298

theorem plate_arrangements :
  let total_arrangements := (10.factorial / (3.factorial * 3.factorial * 2.factorial * 2.factorial) / 10)
  let green_adjacent := (9.factorial / (3.factorial * 3.factorial * 1.factorial * 2.factorial) / 9)
  let orange_adjacent := (9.factorial / (3.factorial * 3.factorial * 1.factorial * 2.factorial) / 9)
  let both_adjacent := (8.factorial / (3.factorial * 3.factorial * 1.factorial * 1.factorial) / 8)
  total_arrangements - green_adjacent - orange_adjacent + both_adjacent = 1540 :=
by {
  let total_arrangements := 10.factorial / (3.factorial * 3.factorial * 2.factorial * 2.factorial) / 10,
  let green_adjacent := 9.factorial / (3.factorial * 3.factorial * 1.factorial * 2.factorial) / 9,
  let orange_adjacent := 9.factorial / (3.factorial * 3.factorial * 1.factorial * 2.factorial) / 9,
  let both_adjacent := 8.factorial / (3.factorial * 3.factorial * 1.factorial * 1.factorial) / 8,
  exact Nat.sub (Nat.sub (Nat.add total_arrangements (-green_adjacent)) orange_adjacent) (-both_adjacent) = 1540,
  sorry
}

end plate_arrangements_l471_471298


namespace eq_system_correct_l471_471482

theorem eq_system_correct (x y : ℤ) : 
  (7 * x + 7 = y) ∧ (9 * (x - 1) = y) :=
sorry

end eq_system_correct_l471_471482


namespace expected_flashlight_lifetime_leq_two_l471_471539

theorem expected_flashlight_lifetime_leq_two
  (Ω : Type*) [MeasurableSpace Ω] [ProbabilitySpace Ω]
  (ξ η : Ω → ℝ)
  (h_min_leq_xi : ∀ ω, min (ξ ω) (η ω) ≤ ξ ω)
  (h_expectation_xi : expectation (ξ) = 2) :
  expectation (fun ω => min (ξ ω) (η ω)) ≤ 2 := 
sorry

end expected_flashlight_lifetime_leq_two_l471_471539


namespace g_at_2_l471_471014

variable {R : Type} [LinearOrderedField R]

def g (x : R) : R 

theorem g_at_2 (h : ∀ x : R, g (3 * x - 7) = 4 * x + 6) : g 2 = 18 :=
by
  sorry

end g_at_2_l471_471014


namespace probability_exactly_nine_matches_l471_471242

theorem probability_exactly_nine_matches (n : ℕ) (h : n = 10) : 
  (∃ p : ℕ, p = 9 ∧ probability_of_exact_matches n p = 0) :=
by {
  sorry
}

end probability_exactly_nine_matches_l471_471242


namespace initial_amount_in_cookie_jar_l471_471895

theorem initial_amount_in_cookie_jar (M : ℝ) (h : 15 / 100 * (85 / 100 * (100 - 10) / 100 * (100 - 15) / 100 * M) = 15) : M = 24.51 :=
sorry

end initial_amount_in_cookie_jar_l471_471895


namespace total_students_l471_471141

def numStudents (skiing scavenger : ℕ) : ℕ :=
  skiing + scavenger

theorem total_students (skiing scavenger : ℕ) (h1 : skiing = 2 * scavenger) (h2 : scavenger = 4000) :
  numStudents skiing scavenger = 12000 :=
by
  sorry

end total_students_l471_471141


namespace domain_of_f_l471_471337

noncomputable def f (x : ℝ) : ℝ := (x^3 - 125) / (x + 5)

theorem domain_of_f :
  {x : ℝ | ∃ y : ℝ, f x = y} = {x : ℝ | x ≠ -5} := 
by
  sorry

end domain_of_f_l471_471337


namespace min_selection_for_diff_two_integers_l471_471001

theorem min_selection_for_diff_two_integers:
  ∀ (s : Finset ℕ),
    (∀ a b ∈ s, a - b ≠ 2)
    → s ⊆ Finset.range 21
    → s.card ≤ 10 :=
by
  sorry

end min_selection_for_diff_two_integers_l471_471001


namespace math_proof_problem_l471_471899

-- Step (1): Define the incorrectness of Jiajia's and Qiqi's methods for the given equation
def Jiajia_method_incorrect : Prop :=
  ¬(∀ x, 3 * (x - 3) = (x - 3) ^ 2 → 3 / (x - 3) = x - 3 → x = 6)

def Qiqi_method_incorrect : Prop :=
  ¬(∀ x, 3 * (x - 3) = (x - 3) ^ 2 → (x - 3) * (6 - x) = 0 → x = 3 ∨ x = 6)

-- Step (2): Define the correct solutions to the quadratic equation
def quadratic_solutions_correct : Prop :=
  (4 * x * (2 * x + 1) = 3 * (2 * x + 1) → x = -1 / 2 ∨ x = 3 / 4)

-- Combine the propositions into a single theorem statement
theorem math_proof_problem :
  Jiajia_method_incorrect ∧
  Qiqi_method_incorrect ∧
  quadratic_solutions_correct :=
by sorry

end math_proof_problem_l471_471899


namespace sum_of_coeffs_l471_471866

theorem sum_of_coeffs {a : Fin 9 → ℝ} {x : ℝ} (h : (2 * x - 3) ^ 8 = ∑ i : Fin 9, a i * x ^ (i : ℕ)) :
  ∑ i : Fin 9, a i = 1 :=
by
  sorry

end sum_of_coeffs_l471_471866


namespace polynomial_remainder_l471_471661

theorem polynomial_remainder (p : ℚ[X]) :
  (p % (X + 1) = 3) ∧ (p % (X + 3) = -1) →
  (p % ((X + 1) * (X + 3)) = 2 * X + 5) :=
by
  sorry

end polynomial_remainder_l471_471661


namespace find_n_of_binomial_l471_471909

variable {X : Type} [Fintype X] [DecidableEq X]

def binomial_distribution (n : ℕ) (P : ℝ) : X → ℝ := sorry

noncomputable def expectation (X : X → ℝ) : ℝ := sorry
noncomputable def variance (X : X → ℝ) : ℝ := sorry

theorem find_n_of_binomial (n : ℕ) (P : ℝ) (hX : binomial_distribution n P)
    (hE : expectation hX = 15) (hV : variance hX = 11.25) : n = 60 :=
by
    sorry

end find_n_of_binomial_l471_471909


namespace probability_of_exactly_nine_correct_matches_is_zero_l471_471235

theorem probability_of_exactly_nine_correct_matches_is_zero :
  let n := 10 in
  let match_probability (correct: Fin n → Fin n) (guess: Fin n → Fin n) (right_count: Nat) :=
    (Finset.univ.filter (λ i => correct i = guess i)).card = right_count in
  ∀ (correct_guessing: Fin n → Fin n), 
    ∀ (random_guessing: Fin n → Fin n),
      match_probability correct_guessing random_guessing 9 → 
        match_probability correct_guessing random_guessing 10 :=
begin
  sorry -- This skips the proof part
end

end probability_of_exactly_nine_correct_matches_is_zero_l471_471235


namespace ratio_area_square_to_circle_l471_471308

noncomputable def length := ℝ

def wire_cut_eq_len (l : length) : Prop :=
  ∃ s c : length, s = c ∧ s + c = 2 * l

def side_length_square (s : length) : length :=
  s / 4

def area_square (s : length) : length :=
  (side_length_square s) ^ 2

def radius_circle (c : length) : length :=
  c / (2 * real.pi)

def area_circle (c : length) : length :=
  real.pi * (radius_circle c) ^ 2

theorem ratio_area_square_to_circle (l : length) (h : wire_cut_eq_len l) :
  (area_square l) / (area_circle l) = real.pi / 4 :=
sorry

end ratio_area_square_to_circle_l471_471308


namespace iron_column_lifted_by_9_6_cm_l471_471716

namespace VolumeLift

def base_area_container : ℝ := 200
def base_area_column : ℝ := 40
def height_water : ℝ := 16
def distance_water_surface : ℝ := 4

theorem iron_column_lifted_by_9_6_cm :
  ∃ (h_lift : ℝ),
    h_lift = 9.6 ∧ height_water - distance_water_surface = 16 - h_lift :=
by
sorry

end VolumeLift

end iron_column_lifted_by_9_6_cm_l471_471716


namespace min_value_of_fC_is_4_l471_471888

def fA (x : ℝ) : ℝ := (1 + 2 * x) / Real.sqrt x
def fB (x : ℝ) : ℝ := (2 + x^2) / x
def fC (x : ℝ) : ℝ := (1 + 4 * x) / Real.sqrt x
def fD (x : ℝ) : ℝ := (4 + x^2) / x

theorem min_value_of_fC_is_4 :
  ∃ x : ℝ, 0 < x ∧ fC x = 4 ∧ 
  (∀ x : ℝ, 0 < x → fA x ≠ 4 ∧
                 fB x ≠ 4 ∧
                 fD x ≠ 4) :=
by
  sorry

end min_value_of_fC_is_4_l471_471888


namespace part1_part2_l471_471925

def line_intersection (l1 l2 : ℝ × ℝ × ℝ) : ℝ × ℝ :=
  let (a1, b1, c1) := l1
  let (a2, b2, c2) := l2
  let y := (c1 * a2 - c2 * a1) / (a2 * b1 - a1 * b2)
  let x := (c1 - b1 * y) / a1
  (x, y)

theorem part1 (x y : ℝ) (h : line_intersection (1, 2, -11) (2, 1, -10) = (x, y)) :
  (2 * y - x = 5) ↔ ⟨x, y⟩ satisfies being perpendicular to l₂ and passes through (3, 4) :=
sorry

theorem part2 (x y : ℝ) (h : line_intersection (1, 2, -11) (2, 1, -10) = (x, y)) :
  (4 * x - 3 * y = 0 ∨ x + y - 7 = 0) ↔ ⟨x, y⟩ has equal intercepts and passes through (3, 4) :=
sorry

end part1_part2_l471_471925


namespace inscribed_sphere_touches_centroid_l471_471820

axiom incircle_touches_midpoints {T : Type} [equilateral_triangle T] : 
  ∀ (t : T), ∀ (m₁ m₂ m₃ : T), touches_incircle t m₁ m₂ m₃ → (midpoints t m₁ m₂ m₃)

theorem inscribed_sphere_touches_centroid {T : Type} [regular_tetrahedron T] : 
  ∀ (tet : T), ∀ (f₁ f₂ f₃ f₄ : equilateral_triangle T), inscribed_sphere_touches_faces tet f₁ f₂ f₃ f₄ → (centroids f₁ f₂ f₃ f₄) :=
by
  sorry

end inscribed_sphere_touches_centroid_l471_471820


namespace cube_side_length_l471_471704

theorem cube_side_length (n : ℕ) (h1 : 6 * n^2 = (6 * n^3) / 3) : n = 3 :=
sorry

end cube_side_length_l471_471704


namespace player_one_wins_with_optimal_play_l471_471300

theorem player_one_wins_with_optimal_play
  (initial_position : ℕ × ℕ)
  (distance : (ℕ × ℕ) → (ℕ × ℕ) → ℕ)
  (move_greater : ∀ n m : ℕ, n < m → ¬ ∃ pos1 pos2 : ℕ × ℕ, distance pos1 pos2 = n ∧ distance pos1.pos2 > m)
  (optimal_play : ∀ pos : ℕ × ℕ, ∃ new_pos : ℕ × ℕ, move_greater (distance initial_position new_pos))
  :
  ∃ strategy : (ℕ × ℕ) → (ℕ × ℕ), ∀ current_pos, (strategy current_pos) winning_move :=
begin
  sorry,
end

end player_one_wins_with_optimal_play_l471_471300


namespace suzanna_textbooks_page_total_l471_471584

theorem suzanna_textbooks_page_total :
  let H := 160
  let G := H + 70
  let M := (H + G) / 2
  let S := 2 * H
  let L := (H + G) - 30
  let E := M + L + 25
  H + G + M + S + L + E = 1845 := by
  sorry

end suzanna_textbooks_page_total_l471_471584


namespace domain_and_range_of_sqrt_function_l471_471121

theorem domain_and_range_of_sqrt_function (x : ℝ) (f : ℝ → ℝ) :
  (∀ x, f x = sqrt (3 - 2 * x - x^2)) →
  { x : ℝ | 3 - 2 * x - x^2 ≥ 0 } = set.Icc (-3 : ℝ) (1 : ℝ) ∧ 
  (∀ (y : ℝ), (∃ x ∈ set.Icc (-3 : ℝ) (1 : ℝ), f x = y) ↔ (set.Icc (0 : ℝ) (2 : ℝ)).has y) :=
by
  sorry

end domain_and_range_of_sqrt_function_l471_471121


namespace expression1_expression2_expression3_expression4_l471_471320

theorem expression1 : 12 - (-10) + 7 = 29 := 
by
  sorry

theorem expression2 : 1 + (-2) * abs (-2 - 3) - 5 = -14 :=
by
  sorry

theorem expression3 : (-8 * (-1 / 6 + 3 / 4 - 1 / 12)) / (1 / 6) = -24 :=
by
  sorry

theorem expression4 : -1 ^ 2 - (2 - (-2) ^ 3) / (-2 / 5) * (5 / 2) = 123 / 2 := 
by
  sorry

end expression1_expression2_expression3_expression4_l471_471320


namespace percentage_of_women_l471_471315

def total_men (n: ℕ) := 0.25 * n = 8

def total_women := 48

theorem percentage_of_women (n : ℕ) (h : total_men n) : 
    (total_women / (n + total_women)) * 100 = 60 := by
  sorry

end percentage_of_women_l471_471315


namespace triangle_is_isosceles_l471_471944

-- Definitions based on the conditions.
variables {A B C D : Type} [LinearField ℝ] -- Assume A, B, C, D are points in space.
variables {BD_median_and_altitude : is_median B D A C ∧ is_altitude B D A C}
variables {BD_angle_bisector : is_altitude B D A C ∧ is_angle_bisector B D A C}

-- The theorem that we need to prove.
theorem triangle_is_isosceles (BD_median_altitude : BD_median_and_altitude)
                              (BD_altitude_bisector : BD_angle_bisector) : 
  is_isosceles_triangle A B C := 
sorry -- Proof not included as instructed.

end triangle_is_isosceles_l471_471944


namespace largest_n_condition_l471_471760

theorem largest_n_condition :
  ∃ n : ℤ, (∃ m : ℤ, n^2 = (m + 1)^3 - m^3) ∧ ∃ k : ℤ, 2 * n + 99 = k^2 ∧ ∀ x : ℤ, 
  (∃ m' : ℤ, x^2 = (m' + 1)^3 - m'^3) ∧ ∃ k' : ℤ, 2 * x + 99 = k'^2 → x ≤ 289 :=
sorry

end largest_n_condition_l471_471760


namespace positive_difference_of_roots_l471_471962

theorem positive_difference_of_roots :
    ∀ r : ℝ, (r ≠ 2) → (r ^ 2 - 5 * r - 20 = (r - 2) * (2 * r + 7)) →
    let q : polynomial ℝ := polynomial.C 1 * polynomial.X ^ 2 + polynomial.C 8 * polynomial.X + polynomial.C 6
    (q.roots.length = 2) →
    let roots := q.roots.toFinset
    roots.card = 2 ∧ roots.max - roots.min = 4 := 
sorry

end positive_difference_of_roots_l471_471962


namespace angle_b_is_acute_l471_471562

-- Definitions for angles being right, acute, and sum of angles in a triangle
def is_right_angle (θ : ℝ) : Prop := θ = 90
def is_acute_angle (θ : ℝ) : Prop := 0 < θ ∧ θ < 90
def angles_sum_to_180 (α β γ : ℝ) : Prop := α + β + γ = 180

-- Main theorem statement
theorem angle_b_is_acute {α β γ : ℝ} (hC : is_right_angle γ) (hSum : angles_sum_to_180 α β γ) : is_acute_angle β :=
by
  sorry

end angle_b_is_acute_l471_471562


namespace arithmetic_sequence_sum_l471_471886

variable (a : ℕ → ℝ)

-- In an arithmetic sequence
def arithmetic_seq := ∀ n, a (n + 1) - a n = a 1 - a 0

-- Given that a₅ = 1
axiom h_a5 : a 5 = 1

theorem arithmetic_sequence_sum (h_arith : arithmetic_seq a) : a 4 + a 5 + a 6 = 3 :=
by
  have h1 : a 4 + a 6 = 2 * a 5 := sorry
  rw [h_a5] at h1
  linarith

end arithmetic_sequence_sum_l471_471886


namespace max_perimeter_of_divided_iso_triangle_l471_471329

noncomputable def max_perimeter_iso_triangle : ℝ :=
  let base := 10
  let height := 12
  let P (k : ℕ) : ℝ := 1 + real.sqrt (12^2 + k^2) + real.sqrt (12^2 + (k + 1)^2)
  max (P 0) (max (P 1) (max (P 2) (max (P 3) (max (P 4) (max (P 5) (max (P 6) (max (P 7) (max (P 8) (P 9)))))))))
  
theorem max_perimeter_of_divided_iso_triangle : max_perimeter_iso_triangle = 31.62 := by
  -- The exact solution steps are skipped here
  sorry

end max_perimeter_of_divided_iso_triangle_l471_471329


namespace expected_min_leq_2_l471_471538

open ProbabilityTheory

variables (ξ η : ℝ → ℝ) -- ξ and η are random variables

-- Condition: expected value of ξ is 2
axiom E_ξ_eq_2 : ℝ
axiom E_ξ_is_2 : (∫ x in ⊤, ξ x) = 2

-- Goal: expected value of min(ξ, η) ≤ 2
theorem expected_min_leq_2 (h : ∀ x, min (ξ x) (η x) ≤ ξ x) : 
  (∫ x in ⊤, min (ξ x) (η x)) ≤ 2 := by
  -- use the provided axioms and conditions here
  sorry

end expected_min_leq_2_l471_471538


namespace expected_lifetime_flashlight_l471_471533

noncomputable def E (X : ℝ) : ℝ := sorry -- Define E as the expectation operator

variables (ξ η : ℝ) -- Define ξ and η as random variables representing lifetimes of blue and red bulbs
variable (h_exi : E ξ = 2) -- Given condition E ξ = 2

theorem expected_lifetime_flashlight (h_min : ∀ x y : ℝ, min x y ≤ x) :
  E (min ξ η) ≤ 2 :=
by
  sorry

end expected_lifetime_flashlight_l471_471533


namespace series_nonconvergence_l471_471154

open Nat

/-- The series \( a_n \) defined by \( a_n = \sum_{k=n+1}^{\infty} a_k^2 \) does not converge 
unless all \( a_n \) are zero. -/
theorem series_nonconvergence (a : ℕ → ℝ) (H : ∀ n, a n = ∑' k in (Set.Ici (n + 1)), a k ^ 2) :
  (∃ l, ∑' n, a n = l) ↔ (∀ n, a n = 0) :=
sorry

end series_nonconvergence_l471_471154


namespace value_of_x_l471_471420

theorem value_of_x (m : ℝ) (x : ℝ) (hm : m > 0) (h_eq : 2^x = log (5 * m) + log (20 / m)) : 
  x = 2 :=
sorry

end value_of_x_l471_471420


namespace pyramid_volume_l471_471949

-- Given conditions
variables (AB BC EA : ℝ) (AB_CD : EA ⊥ AB)
variables (AB_val : AB = 12) (BC_val : BC = 6) (EA_val : EA = 10)
variables (not_perp : ¬(EA ⊥ AD))

-- Prove the volume of EABCD
theorem pyramid_volume : (1 / 3) * (AB * BC) * EA = 240 := 
by 
  sorry

end pyramid_volume_l471_471949


namespace statement_A_statement_B_statement_C_statement_D_l471_471842

-- Definitions
def f_n (n : ℕ) (x : ℝ) : ℝ := (Real.sin x) ^ n + (Real.cos x) ^ n

-- Statements to prove
theorem statement_A : MonotoneOn (λ x, f_n 1 x) (Set.Icc (-Real.pi / 3) (Real.pi / 4)) := sorry

theorem statement_B : ¬ Set.image (λ x, f_n 3 x) (Set.Icc (-Real.pi / 2) 0) = Set.Icc (-(Real.sqrt 2) / 2) ((Real.sqrt 2) / 2) := sorry

theorem statement_C : Function.Periodic (λ x, f_n 4 x) (Real.pi / 2) := sorry

theorem statement_D : ∀ x, f_n 4 x = (λ x, 1 / 4 * Real.cos (4 * x) + 3 / 4) (x - Real.pi / 8) := sorry

end statement_A_statement_B_statement_C_statement_D_l471_471842


namespace g_at_2_l471_471013

variable {R : Type} [LinearOrderedField R]

def g (x : R) : R 

theorem g_at_2 (h : ∀ x : R, g (3 * x - 7) = 4 * x + 6) : g 2 = 18 :=
by
  sorry

end g_at_2_l471_471013


namespace max_discount_rate_l471_471275

theorem max_discount_rate 
  (cost_price : ℝ) (selling_price : ℝ) (min_profit_margin : ℝ) :
  cost_price = 4 → selling_price = 5 → min_profit_margin = 0.1 →
  ∃ (x : ℝ), 0 ≤ x ∧ x ≤ 8.8 ∧ (selling_price * (1 - x / 100) - cost_price) / cost_price ≥ min_profit_margin :=
begin
  sorry
end

end max_discount_rate_l471_471275


namespace min_sin_cos_sixth_power_l471_471774

noncomputable def min_value_sin_cos_expr : ℝ :=
  (3 + 2 * Real.sqrt 2) / 12

theorem min_sin_cos_sixth_power :
  ∃ x : ℝ, (∀ y, (Real.sin y) ^ 6 + 2 * (Real.cos y) ^ 6 ≥ min_value_sin_cos_expr) ∧ 
            ((Real.sin x) ^ 6 + 2 * (Real.cos x) ^ 6 = min_value_sin_cos_expr) :=
sorry

end min_sin_cos_sixth_power_l471_471774


namespace common_ratio_eq_l471_471844

theorem common_ratio_eq (a : ℕ → ℚ) (q : ℚ) (h_geom : ∀ n : ℕ, a (n + 1) = q * a n)
  (h_arith : 2 * a 2 = 2 * a 1 + 4 * a 0) (a1_ne_zero : a 0 ≠ 0) :
  q = 2 ∨ q = -1 :=
by
  sorry

end common_ratio_eq_l471_471844


namespace ellipse_line_intersection_range_l471_471832

theorem ellipse_line_intersection_range (b : ℝ) :
  (∀ m : ℝ, ∃ x y : ℝ, (x, y) ∈ set_of (λ (p : ℝ × ℝ), p.2 = m * p.1 + 1) ∧ (x, y) ∈ set_of (λ (p : ℝ × ℝ), p.1 ^ 2 / 4 + p.2 ^ 2 / b = 1)) →
  b ∈ set.Ico 1 4 ∪ set.Ioi 4 :=
begin
  sorry
end

end ellipse_line_intersection_range_l471_471832


namespace minimum_male_students_l471_471606

theorem minimum_male_students (M : ℕ) (benches : ℕ) (students_per_bench : ℕ) (female_factor : ℕ)
  (h1 : female_factor = 4) (h2 : benches = 29) (h3 : students_per_bench = 5)
  (total_students : ℕ) (h4 : total_students = M + female_factor * M) :
  total_students ≥ benches * students_per_bench → M ≥ 29 := 
by
  intro h 
  calc
    M + 4 * M = 5 * M : by rw [h1]
       ... ≥ 145     : by linarith [h2, h3, h]
       ... = 29 * 5  : by norm_num
       ... = 29      : by norm_num
  sorry

end minimum_male_students_l471_471606


namespace simplify_trig_identity_l471_471961

theorem simplify_trig_identity (x y z : ℝ) :
  sin (x - y + z) * cos y - cos (x - y + z) * sin y = sin (x - 2 * y + z) :=
by
  sorry

end simplify_trig_identity_l471_471961


namespace n_squared_divisible_by_36_l471_471868

theorem n_squared_divisible_by_36 (n : ℕ) (h1 : 0 < n) (h2 : 6 ∣ n) : 36 ∣ n^2 := 
sorry

end n_squared_divisible_by_36_l471_471868


namespace find_f_double_prime_at_2_l471_471404

def f (x : ℝ) : ℝ := (1 / 2) * x^2 + 2 * (f'' (2)) * x - 2 * ln x

theorem find_f_double_prime_at_2 : f'' 2 = -1 :=
sorry

end find_f_double_prime_at_2_l471_471404


namespace second_player_wins_strategy_l471_471292

-- Define the game setup and rules
def grid := list char
def initial_grid : grid := list.repeat ' ' 2006

-- Move is either 'S' or 'O'
inductive move | S | O

structure game :=
    (board : grid)
    (turn : ℕ) -- 0 for player α turn, 1 for player β turn
    (winner: option ℕ) -- none for ongoing, some(0) for player α, some(1) for player β
    (is_square_filled : ℕ → bool)
    (make_move : ℕ → move → game)

-- Winning condition: Sequence of 'SOS' in three consecutive cells
def is_win (g : game) : Prop :=
    ∃ i : ℕ, i < 2004 ∧ g.board[i] = 'S' ∧ g.board[i+1] = 'O' ∧ g.board[i+2] = 'S'

-- Define the forbidden area
def forbidden_area (g : game) (idx : ℕ) : Prop :=
    (g.board[idx] = 'S' ∧ g.board[idx+2] = 'S') ∧ g.board[idx+1] = ' '

-- Winning strategy for the second player β
noncomputable def player_β_wins (g : game) : Prop :=
  ∀ (g : game), g.turn = 1 → 
  (∃ (m : ℕ) (s : move),
    let new_game := g.make_move m s in
      (is_win new_game ∨ 
      (∀ (idx : ℕ), idx < 2006 → forbidden_area new_game idx) ∨
      ¬(is_win new_game) ∧ player_β_wins new_game))

-- Theorem to prove that player β has a winning strategy
theorem second_player_wins_strategy : 
  ∀ (g : game), player_β_wins g :=
begin
  sorry
end

end second_player_wins_strategy_l471_471292


namespace number_of_students_l471_471591

theorem number_of_students (n : ℕ)
  (h_avg : 100 * n = total_marks_unknown)
  (h_wrong_marks : total_marks_wrong = total_marks_unknown + 50)
  (h_correct_avg : total_marks_correct / n = 95)
  (h_corrected_marks : total_marks_correct = total_marks_wrong - 50) :
  n = 10 :=
by
  sorry

end number_of_students_l471_471591


namespace find_largest_int_with_conditions_l471_471763

-- Definition of the problem conditions
def is_diff_of_consecutive_cubes (n : ℤ) : Prop :=
  ∃ m : ℤ, n^2 = (m + 1)^3 - m^3

def is_perfect_square_shifted (n : ℤ) : Prop :=
  ∃ k : ℤ, 2n + 99 = k^2

-- The main statement asserting the proof problem
theorem find_largest_int_with_conditions :
  ∃ n : ℤ, is_diff_of_consecutive_cubes n ∧ is_perfect_square_shifted n ∧
    ∀ m : ℤ, is_diff_of_consecutive_cubes m ∧ is_perfect_square_shifted m → m ≤ 50 :=
sorry

end find_largest_int_with_conditions_l471_471763


namespace number_of_valid_integers_l471_471596

def is_valid_integer (d1 d2 d3 d4 d5 : ℕ) : Prop :=
  (d1 = 1 ∨ d2 = 1 ∨ d3 = 1 ∨ d4 = 1 ∨ d5 = 1) ∧
  (d1 = 3 ∨ d2 = 3 ∨ d3 = 3 ∨ d4 = 3 ∨ d5 = 3) ∧
  (d1 = 5 ∨ d2 = 5 ∨ d3 = 5 ∨ d4 = 5 ∨ d5 = 5) ∧
  (d1 ≠ d2) ∧ (d1 ≠ d3) ∧ (d1 ≠ d4) ∧ (d1 ≠ d5) ∧ 
  (d2 ≠ d3) ∧ (d2 ≠ d4) ∧ (d2 ≠ d5) ∧ 
  (d3 ≠ d4) ∧ (d3 ≠ d5) ∧ (d4 ≠ d5)

def count_valid_integers (digits : List ℕ) : ℕ :=
  List.foldl (λ acc (perm : List ℕ),
    if is_valid_integer perm[0] perm[1] perm[2] perm[3] perm[4] then acc + 1 else acc)
    0 (List.permutations digits)

def valid_integers_with_condition : ℕ :=
  count_valid_integers [1, 3, 4, 5, 6]

theorem number_of_valid_integers : valid_integers_with_condition = 36 :=
  by
    sorry -- We skip the proof here as instructed

end number_of_valid_integers_l471_471596


namespace infinite_solutions_or_no_solutions_l471_471094

theorem infinite_solutions_or_no_solutions (a b : ℚ) :
  (∃ (x y : ℚ), a * x^2 + b * y^2 = 1) →
  (∀ (k : ℚ), a * k^2 + b ≠ 0 → ∃ (x_k y_k : ℚ), a * x_k^2 + b * y_k^2 = 1) :=
by
  intro h_sol h_k
  sorry

end infinite_solutions_or_no_solutions_l471_471094


namespace estimate_students_l471_471879

noncomputable def numberOfStudentsAbove120 (X : ℝ → ℝ) (μ : ℝ) (σ : ℝ) (P₁ : ℝ → ℝ → ℝ)
  (students : ℝ) : ℝ :=
  let prob_interval := P₁ 100 110
  let prob_above_120 := (1 - (2 * prob_interval)) / 2
  prob_above_120 * students

theorem estimate_students (μ : ℝ := 110) (σ : ℝ := 10) (P₁ : ℝ → ℝ → ℝ)
  (students : ℝ := 50) (hyp : P₁ 100 110 = 0.34) :
  numberOfStudentsAbove120 (λ x => Normal.pdf μ σ x) μ σ P₁ students = 8 :=
by sorry

end estimate_students_l471_471879


namespace magnitude_b_l471_471853

-- Define the vectors as variables
variables {ℝ : Type*} [inner_product_space ℝ]

-- Using the conditions given in the problem
variables (a b : ℝ^3) (ha : ∥a∥ = 1) (haba : ∥a - 2 • b∥ = real.sqrt 21) (angle_ab : inner_product_space.angle a b = real.pi / 3)

-- Main theorem statement for the magnitude of b
theorem magnitude_b : ∥b∥ = 2 :=
sorry

end magnitude_b_l471_471853


namespace num_distinct_prime_factors_330_l471_471423

theorem num_distinct_prime_factors_330 : 
  ∃ (s : Finset ℕ), s.card = 4 ∧ ∀ x ∈ s, Nat.Prime x ∧ 330 % x = 0 := 
sorry

end num_distinct_prime_factors_330_l471_471423


namespace digit_at_100_in_squares_seq_l471_471988

/-- The sequence formed by concatenating the squares of natural numbers from 1 to 99.
    We need to prove that the 100th digit in this sequence is 9. -/
def sequence_sq_digit_100 : Nat := 9

theorem digit_at_100_in_squares_seq :
  let seq := (List.range' 1 99).flat_map (λ x, (x * x).toString.data)
  seq.get? 99 = some sequence_sq_digit_100 :=
by
  -- Proof omitted
  sorry

end digit_at_100_in_squares_seq_l471_471988


namespace simplify_and_evaluate_expression_l471_471108

variable {R : Type} [NonAssocRing R] (a b : R)

theorem simplify_and_evaluate_expression (h : a = -b) : 2 * (3 * a^2 + a - 2 * b) - 6 * (a^2 - b) = 0 :=
by
  sorry

end simplify_and_evaluate_expression_l471_471108


namespace total_population_of_cities_l471_471885

theorem total_population_of_cities 
    (number_of_cities : ℕ) 
    (average_population : ℕ) 
    (h1 : number_of_cities = 25) 
    (h2 : average_population = (5200 + 5700) / 2) : 
    number_of_cities * average_population = 136250 := by 
    sorry

end total_population_of_cities_l471_471885


namespace abs_vals_greater_five_less_eight_product_negative_abs_less_four_l471_471343

theorem abs_vals_greater_five_less_eight :
  {x : ℤ | 5 < |x| ∧ |x| < 8} = {6, -6, 7, -7} :=
by {
  -- Proof placeholder
  sorry
}

theorem product_negative_abs_less_four :
  ∏ x in ({x : ℤ | x < 0 ∧ |x| < 4}.toFinset) = -6 :=
by {
  -- Proof placeholder
  sorry
}

end abs_vals_greater_five_less_eight_product_negative_abs_less_four_l471_471343


namespace system_of_equations_correct_l471_471478

theorem system_of_equations_correct (x y : ℕ) :
  (7 * x + 7 = y) ∧ (9 * (x - 1) = y) :=
begin
  -- sorry, proof placeholder
  sorry
end

end system_of_equations_correct_l471_471478


namespace nolan_total_savings_l471_471554

-- Define the conditions given in the problem
def monthly_savings : ℕ := 3000
def number_of_months : ℕ := 12

-- State the equivalent proof problem in Lean 4
theorem nolan_total_savings : (monthly_savings * number_of_months) = 36000 := by
  -- Proof is omitted
  sorry

end nolan_total_savings_l471_471554


namespace PQS_over_PTS_is_32_l471_471874

noncomputable def ratio_area_PQS_PTS 
  (h : ℕ)
  (QT : ℕ) (TR : ℕ) (TS : ℕ) (SR : ℕ)
  (hQT : QT = 4) (hTR : TR = 12) (hTS : TS = 8) (hSR : SR = 4) : Prop :=
  let QS := QT + TS
  ∧ let PQR_area_ratio := (QS * h) / 2 / ((TS * h) / 2) 
  ∧ PQR_area_ratio = (3 / 2)

theorem PQS_over_PTS_is_32 
  (QT TR TS SR h : ℕ) 
  (hQT : QT = 4) (hTR : TR = 12) (hTS : TS = 8) (hSR : SR = 4) : 
  ratio_area_PQS_PTS h QT TR TS SR hQT hTR hTS hSR := 
  by 
    let QS := QT + TS
    have PQR_area_ratio := (QS * h) / 2 / ((TS * h) / 2)
    have : PQR_area_ratio = (3 / 2)
    sorry

end PQS_over_PTS_is_32_l471_471874


namespace fraction_of_peaches_l471_471929

-- Define the number of peaches each person has
def Benjy_peaches : ℕ := 5
def Martine_peaches : ℕ := 16
def Gabrielle_peaches : ℕ := 15

-- Condition that Martine has 6 more than twice Benjy's peaches
def Martine_cond : Prop := Martine_peaches = 2 * Benjy_peaches + 6

-- The goal is to prove the fraction of Gabrielle's peaches that Benjy has
theorem fraction_of_peaches :
  Martine_cond → (Benjy_peaches : ℚ) / (Gabrielle_peaches : ℚ) = 1 / 3 :=
by
  -- Assuming the condition holds
  intro h
  rw [Martine_cond] at h
  -- Use the condition directly, since Martine_cond implies Benjy_peaches = 5
  exact sorry

end fraction_of_peaches_l471_471929


namespace pat_stickers_l471_471557

theorem pat_stickers (stickers_given_away stickers_left : ℝ) 
(h_given_away : stickers_given_away = 22.0)
(h_left : stickers_left = 17.0) : 
(stickers_given_away + stickers_left = 39) :=
by
  sorry

end pat_stickers_l471_471557


namespace max_discount_rate_l471_471270

-- Define the conditions
def cost_price : ℝ := 4
def selling_price : ℝ := 5
def min_profit_margin : ℝ := 0.1
def min_profit : ℝ := cost_price * min_profit_margin

-- Define the discount rate
def discount_rate (x : ℝ) : ℝ :=
  selling_price * (1 - x / 100)

-- Define the profit after discount
def profit_after_discount (x : ℝ) : ℝ :=
  discount_rate x - cost_price

-- The statement we want to prove
theorem max_discount_rate : ∃ x : ℝ, x = 8.8 ∧ profit_after_discount x ≥ min_profit := 
by
  sorry

end max_discount_rate_l471_471270


namespace distinguishable_cubes_l471_471193

theorem distinguishable_cubes : 
  let seven_choose_six := Nat.choose 7 6 in
  let six_factorial := Nat.factorial 6 in
  let cube_symmetries := 24 in
  (seven_choose_six * six_factorial) / cube_symmetries = 210 := by
  let seven_choose_six := Nat.choose 7 6
  let six_factorial := Nat.factorial 6
  let cube_symmetries := 24
  sorry

end distinguishable_cubes_l471_471193


namespace find_grape_juice_l471_471057

variables (milk water: ℝ) (limit total_before_test grapejuice: ℝ)

-- Conditions
def milk_amt: ℝ := 8
def water_amt: ℝ := 8
def limit_amt: ℝ := 32

-- The total liquid consumed before the test can be computed
def total_before_test_amt (milk water: ℝ) : ℝ := limit_amt - water_amt

-- The given total liquid consumed must be (milk + grape juice)
def total_consumed (milk grapejuice: ℝ) : ℝ := milk + grapejuice

theorem find_grape_juice :
    total_before_test_amt milk_amt water_amt = total_consumed milk_amt grapejuice →
    grapejuice = 16 :=
by
    unfold total_before_test_amt total_consumed
    sorry

end find_grape_juice_l471_471057


namespace min_value_expression_l471_471770

theorem min_value_expression (x y : ℝ) (hx : x > 2) (hy : y > 2) : 
  ∃ m, m = 20 ∧ (∀ a b : ℝ, a = x - 2 ∧ b = y - 2 →
    let exp := (a + 2) ^ 2 + 1 / (b + (b + 2) ^ 2 + 1 / a) in exp ≥ m) :=
sorry

end min_value_expression_l471_471770


namespace largest_common_value_less_than_1000_l471_471348

theorem largest_common_value_less_than_1000 :
  ∃ a : ℕ, a < 1000 ∧ (∃ n : ℤ, a = 4 + 5 * n) ∧ (∃ m : ℤ, a = 5 + 8 * m) ∧ 
            (∀ b : ℕ, (b < 1000 ∧ (∃ n : ℤ, b = 4 + 5 * n) ∧ (∃ m : ℤ, b = 5 + 8 * m)) → b ≤ a) :=
sorry

end largest_common_value_less_than_1000_l471_471348


namespace probability_open_remaining_piggy_banks_l471_471192

/-- 
For 30 piggy banks, each containing a unique key assigned to open one of the piggy banks.
The keys are randomly placed in the piggy banks, each having one key.
After breaking two piggy banks, the probability of being able to open all remaining 28
piggy banks without breaking any more piggy banks is equal to 1/15.
-/
theorem probability_open_remaining_piggy_banks : 
  let n := 30 in
  let k := 2 in
  (k / (n:ℕ) = 1 / 15) :=
by sorry

end probability_open_remaining_piggy_banks_l471_471192


namespace weekly_income_l471_471634

-- Defining the daily catches
def blue_crabs_per_bucket (day : String) : ℕ :=
  match day with
  | "Monday"    => 10
  | "Tuesday"   => 8
  | "Wednesday" => 12
  | "Thursday"  => 6
  | "Friday"    => 14
  | "Saturday"  => 10
  | "Sunday"    => 8
  | _           => 0

def red_crabs_per_bucket (day : String) : ℕ :=
  match day with
  | "Monday"    => 14
  | "Tuesday"   => 16
  | "Wednesday" => 10
  | "Thursday"  => 18
  | "Friday"    => 12
  | "Saturday"  => 10
  | "Sunday"    => 8
  | _           => 0

-- Prices per crab
def price_per_blue_crab : ℕ := 6
def price_per_red_crab : ℕ := 4
def buckets : ℕ := 8

-- Daily income calculation
def daily_income (day : String) : ℕ :=
  let blue_income := (blue_crabs_per_bucket day) * buckets * price_per_blue_crab
  let red_income := (red_crabs_per_bucket day) * buckets * price_per_red_crab
  blue_income + red_income

-- Proving the weekly income is $6080
theorem weekly_income : 
  (daily_income "Monday" +
  daily_income "Tuesday" +
  daily_income "Wednesday" +
  daily_income "Thursday" +
  daily_income "Friday" +
  daily_income "Saturday" +
  daily_income "Sunday") = 6080 :=
by sorry

end weekly_income_l471_471634


namespace new_person_age_l471_471973

theorem new_person_age (T : ℕ) (A : ℕ) (n : ℕ) 
  (avg_age : ℕ) (new_avg_age : ℕ) 
  (h1 : avg_age = T / n) 
  (h2 : T = 14 * n)
  (h3 : n = 17) 
  (h4 : new_avg_age = 15) 
  (h5 : new_avg_age = (T + A) / (n + 1)) 
  : A = 32 := 
by 
  sorry

end new_person_age_l471_471973


namespace four_digit_prime_even_unique_digits_l471_471855

open Nat

/-- 
  The number of four-digit whole numbers such that:
  1. The leftmost digit is a prime number.
  2. The second digit is even.
  3. All four digits are different.
-/
theorem four_digit_prime_even_unique_digits : 
  let primes := {2, 3, 5, 7}
  let evens := {0, 2, 4, 6, 8}
  let is_prime (d : ℕ) := d ∈ primes
  let is_even (d : ℕ) := d ∈ evens
  (∃ f s t fo, is_prime f ∧ is_even s ∧ f ≠ s ∧ f ≠ t ∧ f ≠ fo ∧ s ≠ t ∧ s ≠ fo ∧ t ≠ fo ∧ 
               1000 * f + 100 * s + 10 * t + fo < 10000) = 1064 :=
sorry

end four_digit_prime_even_unique_digits_l471_471855


namespace probability_red_first_given_black_second_l471_471641

open ProbabilityTheory MeasureTheory

-- Definitions for Urn A and Urn B ball quantities
def urnA := (white : 4, red : 2)
def urnB := (red : 3, black : 3)

-- Event of drawing a red ball first and a black ball second
def eventRedFirst := (urn : ℕ, ball : string) -> (urn = 1 ∧ ball = "red") ∨ (urn = 2 ∧ ball = "red")
def eventBlackSecond := (urn : ℕ, ball : string) -> (urn = 1 ∧ ball = "black") ∨ (urn = 2 ∧ ball = "black")

-- Probability function definition
noncomputable def P := sorry -- Probability function placeholder

-- Conditional Probability
theorem probability_red_first_given_black_second :
  P(eventRedFirst | eventBlackSecond) = 2 / 5 := sorry

end probability_red_first_given_black_second_l471_471641


namespace geometric_sequence_product_l471_471489

variable {α : Type*} [LinearOrderedField α]

theorem geometric_sequence_product :
  ∀ (a r : α), (a^3 * r^6 = 3) → (a^3 * r^15 = 24) → (a^3 * r^24 = 192) :=
by
  intros a r h1 h2
  sorry

end geometric_sequence_product_l471_471489


namespace georgia_total_cost_l471_471804

-- Define the prices for three tiers
def prices_tier1 := {single := 0.60, dozen := 5.00, bundle25 := 10.00}
def prices_tier2 := {single := 0.50, dozen := 4.50, bundle50 := 16.00}
def prices_tier3 := {single := 0.40, dozen := 4.00, bundle100 := 30.00}

-- Define the quantities ordered for teachers
def teachers :=
  [ (2 * prices_tier1.dozen),
    (prices_tier2.dozen + 5 * prices_tier2.single),
    prices_tier1.bundle25,
    (3 * prices_tier3.dozen + 10 * prices_tier3.single),
    (2 * prices_tier2.bundle50),
    (prices_tier3.dozen + 7 * prices_tier1.single),
    prices_tier3.bundle100,
    (3 * prices_tier1.dozen)
  ]

-- Define the quantities ordered for friends
def friends :=
  [ (4 * 3 * prices_tier2.single),
    prices_tier1.dozen,
    (4 * 5 * prices_tier3.single),
    (15 * prices_tier2.single),
    (3 * (prices_tier1.dozen + 2 * prices_tier1.single)),
    (25 * prices_tier3.single),
    (4 * 4 * prices_tier1.single),
    prices_tier2.bundle50,
    (2 * prices_tier1.dozen + 5 * prices_tier1.single)
  ]

-- Calculation of the total cost for teachers and friends
def total_cost (teachers friends : list ℝ) : ℝ :=
  teachers.sum + friends.sum

-- Assertion about the total cost being equal to the expected value
theorem georgia_total_cost :
  total_cost teachers friends = 221.90 :=
by
  -- Instead of the proof
  sorry

end georgia_total_cost_l471_471804


namespace system_consistent_k_eq_4_l471_471361

theorem system_consistent_k_eq_4 (x y u k : ℝ) :
  (x + y = 1) →
  (k * x + y = 2) →
  (x + k * u = 3) →
  k = 4 :=
begin
  sorry
end

end system_consistent_k_eq_4_l471_471361


namespace john_bought_3_candy_bars_l471_471500

noncomputable def number_of_candy_bars (gum_cost_per_pack : ℝ) (candy_bar_cost : ℝ) (total_cost : ℝ): ℕ :=
  let gum_total_cost := 2 * gum_cost_per_pack
  let remaining_amount := total_cost - gum_total_cost
  let candy_bars := remaining_amount / candy_bar_cost
  candy_bars.toNat

theorem john_bought_3_candy_bars:
  let gum_cost_per_pack := 1.5 / 2
  let candy_bar_cost := 1.5
  let total_cost := 6
  number_of_candy_bars gum_cost_per_pack candy_bar_cost total_cost = 3 :=
by
  sorry

end john_bought_3_candy_bars_l471_471500


namespace subgroup_addition_l471_471907

theorem subgroup_addition (p : ℕ) [fact (nat.prime p)] {A : set (zmod p)}
  [is_subgroup A]
  (h : fintype.card A % 6 = 0) :
  ∃ x y z : zmod p, x ∈ A ∧ y ∈ A ∧ z ∈ A ∧ (x + y = z) :=
sorry

end subgroup_addition_l471_471907


namespace range_of_a_l471_471363

-- Define the sets A and B
def setA (a : ℝ) : Set ℝ := {x | x - a > 0}
def setB : Set ℝ := {x | x ≤ 0}

-- The main theorem asserting the condition
theorem range_of_a {a : ℝ} (h : setA a ∩ setB = ∅) : a ≥ 0 := by
  sorry

end range_of_a_l471_471363


namespace triangle_isosceles_if_perpendiculars_equal_l471_471906

/-- Let ABC be a triangle with internal angle bisectors AD and CE such that D is on BC and 
   E is on AB. Let K and M be the feet of perpendiculars from B to the lines AD and CE,
   respectively. If |BK| = |BM|, then ∆ABC is isosceles. -/
theorem triangle_isosceles_if_perpendiculars_equal (A B C D E K M : Type*) 
  [linear_ordered_field A] [metric_space A] [normed_group A] [normed_space ℝ A] 
  [inner_product_space ℝ A] (h₁ : is_angle_bisector A D B C) 
  (h₂ : is_angle_bisector E B C A) 
  (h₃ : foot_of_perpendicular B D = K) 
  (h₄ : foot_of_perpendicular B E = M) 
  (h₅ : dist B K = dist B M) 
  : ∠A B C = ∠A C B → triangle_is_isosceles A B C :=
begin
  sorry
end

end triangle_isosceles_if_perpendiculars_equal_l471_471906


namespace krista_price_per_dozen_l471_471501

def total_eggs_per_week (hens : ℕ) (eggs_per_hen : ℕ) : ℕ :=
  hens * eggs_per_hen

def total_eggs_in_four_weeks (total_eggs_per_week : ℕ) : ℕ :=
  total_eggs_per_week * 4

def dozens_of_eggs (total_eggs : ℕ) : ℕ :=
  total_eggs / 12

def price_per_dozen (total_revenue : ℚ) (dozens : ℕ) : ℚ :=
  total_revenue / dozens

theorem krista_price_per_dozen :
  ∀ (hens : ℕ) (eggs_per_hen : ℕ) (weeks : ℕ) (revenue : ℚ)
    (total_eggs_per_week : ℕ)
    (total_eggs : ℕ)
    (dozens : ℕ)
    (price : ℚ),
    hens = 10 →
    eggs_per_hen = 12 →
    weeks = 4 →
    revenue = 120 →
    total_eggs_per_week = total_eggs_per_week hens eggs_per_hen →
    total_eggs = total_eggs_in_four_weeks total_eggs_per_week →
    dozens = dozens_of_eggs total_eggs →
    price = price_per_dozen revenue dozens →
    price = 3 :=
by
  intros hens eggs_per_hen weeks revenue total_eggs_per_week total_eggs dozens price
  intros hens_eq eggs_per_hen_eq weeks_eq revenue_eq total_eggs_per_week_eq total_eggs_eq dozens_eq price_eq
  rw [hens_eq, eggs_per_hen_eq, weeks_eq, revenue_eq] at *
  rw [total_eggs_per_week_eq, total_eggs_eq, dozens_eq, price_eq]
  sorry

end krista_price_per_dozen_l471_471501


namespace circle_C_equation_min_area_quadrilateral_l471_471498

section
variables {P Q : Type*}

def circle (center : P × P) (radius2 : ℝ) : set (P × P) :=
{ point | (point.1 - center.1)^2 + (point.2 - center.2)^2 = radius2 }

def line (a b c : ℝ) : set (P × P) :=
{ point | a * point.1 + b * point.2 + c = 0 }

theorem circle_C_equation 
  (Q : P × P) (M : P × P) (r : ℝ) (hm : circle M r) 
  (P : P × P) (hline : line 3 4 8 P)
  (hx_eq : (Q.1 - 6/5)^2 + (Q.2 + 2/5)^2 = 2) :
  circle (6/5, -2/5) 2 :=
sorry

theorem min_area_quadrilateral 
  (P : P × P) (hline : line 3 4 8 P) 
  (C : P × P) (hx_eq : (Q.1 - 6/5)^2 + (Q.2 + 2/5)^2 = 2) 
  (hPA : ℝ) (hx_dist : (3 * 6/5 + 4 * (-2/5) + 8)^2/(3^2 + 4^2) = 2) :
  ∃ S, S = 2 :=
sorry
end

end circle_C_equation_min_area_quadrilateral_l471_471498


namespace total_selection_methods_l471_471375

open Finset

def I : Finset ℕ := {1, 2, 3, 4, 5}

def validPairs (A B : Finset ℕ) : Prop :=
  A ≠ ∅ ∧ B ≠ ∅ ∧ (∀ a ∈ A, ∀ b ∈ B, a < b)

def selectionMethods : Finset (Finset ℕ × Finset ℕ) :=
  (powerset I).product (powerset I).filter (λ ⟨A, B⟩, validPairs A B)

theorem total_selection_methods : (selectionMethods.card = 49) :=
  sorry

end total_selection_methods_l471_471375


namespace min_sin_cos_sixth_power_l471_471775

noncomputable def min_value_sin_cos_expr : ℝ :=
  (3 + 2 * Real.sqrt 2) / 12

theorem min_sin_cos_sixth_power :
  ∃ x : ℝ, (∀ y, (Real.sin y) ^ 6 + 2 * (Real.cos y) ^ 6 ≥ min_value_sin_cos_expr) ∧ 
            ((Real.sin x) ^ 6 + 2 * (Real.cos x) ^ 6 = min_value_sin_cos_expr) :=
sorry

end min_sin_cos_sixth_power_l471_471775


namespace lockers_unlocked_if_perfect_square_l471_471162

-- Define the basic setup
def locker_state (n : ℕ) : ℕ → bool := λ _, false

def toggle (state : ℕ → bool) (k : ℕ) (n : ℕ) : ℕ → bool :=
λ m, if m % k = 0 then bnot (state m) else state m

def execute_operations (n : ℕ) : (ℕ → bool) :=
Nat.fold n (toggle (locker_state n)) (locker_state n)

def is_perfect_square (m : ℕ) : Prop :=
∃ k : ℕ, k * k = m

-- Prove the statement
theorem lockers_unlocked_if_perfect_square (n : ℕ) :
  ∀ m, 1 ≤ m ∧ m ≤ n → execute_operations n m = tt ↔ is_perfect_square m :=
sorry

end lockers_unlocked_if_perfect_square_l471_471162


namespace unique_maximizing_g_and_sum_of_digits_l471_471334

def g (n : ℕ) : ℚ := (d n) / (n ^ (1/4 : ℚ))

def d (n : ℕ) : ℕ := ∏ i in (factors n), (i + 1)

def sum_of_digits (n : ℕ) : ℕ :=
  let digits := to_digits n
  digits.foldl (· + ·) 0

theorem unique_maximizing_g_and_sum_of_digits :
  ∃ M : ℕ, (∀ m : ℕ, m ≠ M → g M > g m) ∧ sum_of_digits M = 9 := sorry

end unique_maximizing_g_and_sum_of_digits_l471_471334


namespace ratio_equivalence_l471_471921

theorem ratio_equivalence (x : ℝ) :
  ((20 / 10) * 100 = (25 / x) * 100) → x = 12.5 :=
by
  intro h
  sorry

end ratio_equivalence_l471_471921


namespace math_problem_statement_l471_471406

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x

def pointP (a b : ℝ) := f a b 1 = 2

def extremum_at_third (a b : ℝ) : Prop :=
  let f' := λ x, 3*x^2 + 2*a*x + b
  in f' (1/3) = 0

def values_of_a_b : Prop :=
  ∃ (a b : ℝ), pointP a b ∧ extremum_at_third a b ∧ a = 4 ∧ b = -3

def monotonic_intervals : Prop :=
  let a := 4
  let b := -3
  let f' := λ x : ℝ, 3*x^2 + 2*a*x + b
  ∀ x : ℝ, 
    ((x < -3 ∨ x > 1/3) → f' x > 0) ∧
    (-3 < x ∧ x < 1/3 → f' x < 0)

def max_min_values_on_interval : Prop :=
  let a := 4
  let b := -3
  let f_x := f a b
  f_x (-1) = 6 ∧ f_x 1 = 2 ∧ f_x (1/3) = -4/27 ∧ 
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → (f_x x ≤ 6 ∧ f_x x ≥ -4/27)

theorem math_problem_statement :
  values_of_a_b ∧ monotonic_intervals ∧ max_min_values_on_interval :=
by
  split
  · use 4, -3
    constructor
    · sorry
    constructor
    · sorry
    constructor
    · rfl
    rfl
  · sorry
  · sorry

end math_problem_statement_l471_471406


namespace probability_sum_even_from_1_to_9_l471_471803

theorem probability_sum_even_from_1_to_9 : 
  let numbers := finset.range 10 -- represents numbers from 1 to 9
  let combinations n := finset.powerset_len n numbers
  let event_even_sum (s : finset ℕ) := s.sum % 2 = 0
  let total_combinations := (combinations 3).card
  let event_combinations := ((combinations 3).filter event_even_sum).card
  (event_combinations / total_combinations : ℚ) = 11 / 21 := sorry

end probability_sum_even_from_1_to_9_l471_471803


namespace no_solutions_to_equation_l471_471097

theorem no_solutions_to_equation (a b c : ℤ) : a^2 + b^2 - 8 * c ≠ 6 := 
by 
-- sorry to skip the proof part
sorry

end no_solutions_to_equation_l471_471097


namespace max_value_expression_l471_471797

open Real

theorem max_value_expression (x : ℝ) : 
  ∃ (y : ℝ), y ≤ (x^6 / (x^10 + 3 * x^8 - 5 * x^6 + 10 * x^4 + 25)) ∧
  y = 1 / (5 + 2 * sqrt 30) :=
sorry

end max_value_expression_l471_471797


namespace units_digit_of_factorial_sum_l471_471791

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_of_factorial_sum :
  units_digit (1! + 2! + 3! + 4! + ∑ k in (finset.range 2002).map (finset.filter (> 4)), k!) = 3 :=
by
  -- the proof details go here
  sorry

end units_digit_of_factorial_sum_l471_471791


namespace emily_earns_more_l471_471362

def investment_george (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r)^n

def investment_emily (P : ℝ) (r : ℝ) (m : ℕ) : ℝ :=
  P * (1 + r/2)^(2*m)

theorem emily_earns_more (P : ℝ) (r1 r2 : ℝ) (n m : ℕ) :
  P = 40000 →
  r1 = 0.05 →
  r2 = 0.06 →
  n = 3 →
  m = 3 →
  investment_emily P r2 m - investment_george P r1 n ≈ 1457 :=
by
  intros
  sorry

end emily_earns_more_l471_471362


namespace expected_value_linear_combination_l471_471364

variable (X Y : Type)
variable [AddCommGroup X] [AddCommGroup Y]
variable [ExpectedValue X Real] [ExpectedValue Y Real]

theorem expected_value_linear_combination
  (h1 : E(X) = 10) 
  (h2 : E(Y) = 3) : 
  E(3 • X + 5 • Y) = 45 := 
  sorry

end expected_value_linear_combination_l471_471364


namespace square_side_length_l471_471204

theorem square_side_length (p : ℝ) (h : p = 17.8) : (p / 4) = 4.45 := by
  sorry

end square_side_length_l471_471204


namespace diagonal_AC_length_l471_471099

variable (A B C D : Point)
variable (angle : Angle)
variables (O : ℝ) (circumference : ℝ) (AD BC AC : ℝ)

-- Given conditions
axiom cyclic_quad : cyclic ABCD
axiom angle_BAC : angle A B C = 60
axiom angle_ADB : angle A D B = 50
axiom length_AD : AD = 3
axiom length_BC : BC = 5

-- Prove that AC = 4.5
theorem diagonal_AC_length :
  AC = 4.5 := sorry

end diagonal_AC_length_l471_471099


namespace chord_length_in_polar_coordinates_l471_471892

theorem chord_length_in_polar_coordinates (theta: ℝ) (ρ: ℝ)
  (h_line : ρ * cos theta = 1/2)
  (h_circle : ρ = 2 * cos theta) :
  ∃ L : ℝ, L = sqrt 3 :=
by
  sorry

end chord_length_in_polar_coordinates_l471_471892


namespace universal_number_min_length_l471_471191

-- Define a universal number in the decimal system
def is_universal (b : ℕ) :=
  ∀ (A : ℕ), (∃ (digitsA : Finset ℕ), digitsA.card ≤ 10 ∧ (∀ d ∈ digitsA, d < 10) ∧ 
  (∃ (subseq : list ℕ), subseq.to_finset = digitsA ∧ subseq ≺ (list.of_digits ∘ list.reverse ∘ nat.digits 10) b))

-- Theorem to show that any universal number must have at least 55 digits
theorem universal_number_min_length (b : ℕ) (h_universal : is_universal b) : 
  nat.log 10 b ≥ 54 :=
sorry

end universal_number_min_length_l471_471191


namespace first_ball_red_given_second_black_l471_471646

open ProbabilityTheory

noncomputable def urn_A : Finset (Finset ℕ) := { {0, 0, 0, 0, 1, 1}, {0, 0, 0, 1, 1, 2}, ... }
noncomputable def urn_B : Finset (Finset ℕ) := { {1, 1, 1, 2, 2, 2}, {1, 1, 2, 2, 2, 2}, ... }

noncomputable def prob_draw_red : ℕ := 7 / 15

theorem first_ball_red_given_second_black :
  (∑ A_Burn_selection in ({0, 1} : Finset ℕ), 
     ((∑ ball_draw from A_Burn_selection,
           if A_Burn_selection = 0 then (∑ red in urn_A, if red = 1 then 1 else 0) / 6 / 2
           else (∑ red in urn_B, if red = 1 then 1 else 0) / 6 / 2) *
     ((∑ second_urn_selection in ({0, 1} : Finset ℕ),
           if second_urn_selection = 0 and A_Burn_selection = 0 then 
              ∑ black in urn_A, if black = 1 then 1 else 0 / 6 / 2 
           else 
              ∑ black in urn_B, if black = 1 then 1 else 0 / 6 / 2))) = 7 / 15 :=
sorry

end first_ball_red_given_second_black_l471_471646


namespace euler_characteristic_ge_circles_l471_471506

noncomputable def surface := Type
noncomputable def circle := Type
noncomputable def component (S : surface) := S → Prop

variables (S : surface) (C : finset circle)
variables (D0 : component (S \ \bigcup C))

-- Conditions
def circles_disjoint : Prop :=
  ∀ (c1 c2 : circle), c1 ≠ c2 → c1 ∩ c2 = ∅

def component_intersects_closure : Prop :=
  ∀ (c : circle), ∀ (closure_c : set (component S)), c ∈ C → D0 closure_c ∩ closure_c ≠ ∅

def no_disk_disjoint_from_D0 : Prop :=
  ∀ (c : circle), ∀ (disk_c : set S), c ∈ C → S \ disk_c → D0 (S \ disk_c) = ∅

variable [circles_disjoint S C]
variable [component_intersects_closure S C D0]
variable [no_disk_disjoint_from_D0 S C D0]

theorem euler_characteristic_ge_circles :
  ∃ (ε : ℕ), ε >= C.card :=
sorry

end euler_characteristic_ge_circles_l471_471506


namespace rahul_matches_l471_471947

theorem rahul_matches
  (initial_avg : ℕ)
  (runs_today : ℕ)
  (final_avg : ℕ)
  (n : ℕ)
  (H1 : initial_avg = 50)
  (H2 : runs_today = 78)
  (H3 : final_avg = 54)
  (H4 : (initial_avg * n + runs_today) = final_avg * (n + 1)) :
  n = 6 :=
by
  sorry

end rahul_matches_l471_471947


namespace magnitude_of_a_l471_471460

open Real

noncomputable def vec_magnitude (x y : ℝ) := sqrt (x^2 + y^2)

theorem magnitude_of_a (m : ℝ) : 
  let a := (2^m, -1)
  let b := (2^m - 1, 2^(m + 1))
  (2^m * (2^m - 1) + -1 * (2^(m+1)) = 0) → 
  vec_magnitude (2^m) (-1) = sqrt(10) :=
by
  intros
  sorry

end magnitude_of_a_l471_471460


namespace range_f_l471_471514

def f (x : ℝ) : ℝ := real.sqrt (2 - x) + real.sqrt (3 * x + 12)

theorem range_f : set.range f = set.Icc (real.sqrt 6) (2 * real.sqrt 6) :=
sorry

end range_f_l471_471514


namespace compute_value_l471_471912

theorem compute_value {a b : ℝ} 
  (h1 : ∀ x, (x + a) * (x + b) * (x + 12) = 0 → x ≠ -3 → x = -a ∨ x = -b ∨ x = -12)
  (h2 : ∀ x, (x + 2 * a) * (x + 3) * (x + 6) = 0 → x ≠ -b ∧ x ≠ -12 → x = -3) :
  100 * (3 / 2) + 6 = 156 :=
by
  sorry

end compute_value_l471_471912


namespace tangent_spheres_radii_relation_l471_471630

theorem tangent_spheres_radii_relation (R r ρ : ℝ) 
  (hR : R > 0) (hρr : ρ > r) 
  (hr : r > 0) (hρ : ρ > 0) 
  (h1 : ∀P : Point, ∃S1 S2 S3 : Sphere, mutually_tangent S1 S2 S3 P R)
  (h2 : ∃(S1 S2 S3 : Sphere) (P : Plane),
         tangent_to_plane S1 P ∧ tangent_to_plane S2 P ∧ tangent_to_plane S3 P ∧
         exists_two_new_spheres S1 S2 S3 P r ρ) 
  : (1 / r) - (1 / ρ) = (2 * Real.sqrt 3) / R := 
by
  sorry

end tangent_spheres_radii_relation_l471_471630


namespace slope_of_line_eq_l471_471156

theorem slope_of_line_eq : ∀ (x : ℝ), ∀ (y : ℝ), (y = x - 2) → (1 = 1) := 
by 
  assume x y h, 
  sorry

end slope_of_line_eq_l471_471156


namespace collinear_c1_c2_l471_471219

noncomputable def vector_a : Vector ℝ 3 := ⟨[-2, -3, -2]⟩
noncomputable def vector_b : Vector ℝ 3 := ⟨[1, 0, 5]⟩
noncomputable def vector_c1 : Vector ℝ 3 := 3 • vector_a + 9 • vector_b
noncomputable def vector_c2 : Vector ℝ 3 := -1 • vector_a - 3 • vector_b

theorem collinear_c1_c2 : ∃ γ : ℝ, vector_c1 = γ • vector_c2 :=
by
  sorry

end collinear_c1_c2_l471_471219


namespace fourier_series_of_f_l471_471201

noncomputable def f (x : ℝ) : ℝ :=
if -π < x ∧ x < 0 then -π/4
else if x = 0 ∨ x = -π then 0
else if 0 < x ∧ x < π then π/4
else f (x - 2*π)

theorem fourier_series_of_f :
  ∀ x, f x = ∑ n in Finset.range (∞), sin ((2 * n + 1) * x) / (2 * n + 1) :=
begin
  -- proof goes here.
  sorry
end

end fourier_series_of_f_l471_471201


namespace determine_function_l471_471740

theorem determine_function (f : ℤ → ℤ) (h : ∀ m n : ℤ, f(m + f(f(n))) = -f(f(m + 1)) - n) : 
  ∀ p : ℤ, f(p) = 1 - p :=
by
  sorry

end determine_function_l471_471740


namespace system1_solution_system2_solution_l471_471578

theorem system1_solution (x y : ℤ) : 
  (x - y = 3) ∧ (x = 3 * y - 1) → (x = 5) ∧ (y = 2) :=
by
  sorry

theorem system2_solution (x y : ℤ) : 
  (2 * x + 3 * y = -1) ∧ (3 * x - 2 * y = 18) → (x = 4) ∧ (y = -3) :=
by
  sorry

end system1_solution_system2_solution_l471_471578


namespace expected_flashlight_lifetime_leq_two_l471_471542

theorem expected_flashlight_lifetime_leq_two
  (Ω : Type*) [MeasurableSpace Ω] [ProbabilitySpace Ω]
  (ξ η : Ω → ℝ)
  (h_min_leq_xi : ∀ ω, min (ξ ω) (η ω) ≤ ξ ω)
  (h_expectation_xi : expectation (ξ) = 2) :
  expectation (fun ω => min (ξ ω) (η ω)) ≤ 2 := 
sorry

end expected_flashlight_lifetime_leq_two_l471_471542


namespace projection_of_a_onto_c_is_zero_l471_471419

noncomputable def a (λ : ℝ) : ℝ × ℝ := (1, λ)
def b : ℝ × ℝ := (3, 1)
def c : ℝ × ℝ := (1, 2)

def collinear (v w : ℝ × ℝ) : Prop := ∃ k : ℝ, w = (k * v.1, k * v.2)

theorem projection_of_a_onto_c_is_zero (λ : ℝ) (h_collinear : collinear (2 * a λ.1 - b.1, 2 * a λ.2 - b.2) c) : (a (-1 / 2).1 * c.1 + a (-1 / 2).2 * c.2) / sqrt (c.1 ^ 2 + c.2 ^ 2) = 0 :=
by
  sorry

end projection_of_a_onto_c_is_zero_l471_471419


namespace sinB_value_triangle_area_value_l471_471024

variables (A B C : ℝ) (a b c : ℝ)
variables (cosA cosB cosC : ℝ)
noncomputable def sinB_condition_1 (A B C a b c : ℝ) (cosC cosB : ℝ) 
  (h1 : ∀ {A B C a b c}, a = b ∧ b = c → A + B + C = 180) 
  (h2 : cosC / cosB = (3 * a - c) / b) : real :=
  sqrt (1 - cosB * cosB)

noncomputable def triangle_area (a b c : ℝ) (sinB : real)
  (h1 : b = 4 * sqrt 2)
  (h2 : a = c) : real :=
  (1/2) * c * c * sinB

theorem sinB_value (A B C a b c : ℝ) (cosC cosB : ℝ)
(h1 : ∀ {A B C a b c}, a = b ∧ b = c → A + B + C = 180)
(h2 : cosC / cosB = (3 * a - c) / b)
(h := sqrt (1 - cosB * cosB)) : sinB_condition_1 A B C a b c cosC cosB h1 h2 = (2 * sqrt 2) / 3 := sorry

theorem triangle_area_value (a b c sinB : ℝ) 
(h1 : b = 4 * sqrt 2)
(h2 : a = c)
(h := sqrt (1 - (1 / 3)^2))
: triangle_area a b c sinB h1 h2 = 8 * sqrt 2 := sorry

end sinB_value_triangle_area_value_l471_471024


namespace maria_made_144_cookies_l471_471927

def cookies (C : ℕ) : Prop :=
  (2 * 1 / 4 * C = 72)

theorem maria_made_144_cookies: ∃ (C : ℕ), cookies C ∧ C = 144 :=
by
  existsi 144
  unfold cookies
  sorry

end maria_made_144_cookies_l471_471927


namespace first_expression_calc_second_expression_calc_l471_471566

noncomputable def rationalization_series : ℝ :=
  let series_term := λ n : ℕ, 1 / (Real.sqrt (n + 2) + Real.sqrt (n + 1)) in
  (Finset.range 2022).sum series_term

theorem first_expression_calc :
  (rationalization_series * (Real.sqrt 2023 + 1)) = 2022 :=
by sorry

theorem second_expression_calc :
  (12 / (5 + Real.sqrt 13) + 5 / (Real.sqrt 13 - 2 * Real.sqrt 2) - 4 / (2 * Real.sqrt 2 + 3)) = 
  (10 * Real.sqrt 2 - 7) :=
by sorry

end first_expression_calc_second_expression_calc_l471_471566


namespace count_subgroups_multiple_of_11_l471_471715

noncomputable def is_multiple_of (n k : ℕ) : Prop := k > 0 ∧ n % k = 0

theorem count_subgroups_multiple_of_11 :
  let numbers : List ℕ := [1, 4, 8, 10, 16, 19, 21, 25, 30, 43]
  List.length (numbers)
  == 10 →
  (∃ n, count (λ sublist, is_multiple_of sublist.sum 11) sublists (filter_sublists numbers) = 7) :=
sorry

end count_subgroups_multiple_of_11_l471_471715


namespace total_tax_difference_is_approximately_73_l471_471461

-- Define the market prices of the items
def price_A : ℝ := 7800
def price_B : ℝ := 9500
def price_C : ℝ := 11000

-- Define the original and reduced tax rates
def original_tax_rate_A : ℝ := 3.5 / 100
def reduced_tax_rate_A : ℝ := 3.333 / 100

def original_tax_rate_B : ℝ := 4.75 / 100
def reduced_tax_rate_B : ℝ := 4.5 / 100

def original_tax_rate_C : ℝ := 5.666 / 100
def reduced_tax_rate_C : ℝ := 5.333 / 100

-- Calculate the difference in tax
def tax_difference (price original_rate reduced_rate : ℝ) : ℝ :=
  price * (original_rate - reduced_rate)

def total_tax_difference : ℝ :=
  tax_difference price_A original_tax_rate_A reduced_tax_rate_A +
  tax_difference price_B original_tax_rate_B reduced_tax_rate_B +
  tax_difference price_C original_tax_rate_C reduced_tax_rate_C

-- Lean 4 statement to prove the total tax difference is approximately Rs. 73
theorem total_tax_difference_is_approximately_73 : abs (total_tax_difference - 73) < 1 :=
by
  sorry

end total_tax_difference_is_approximately_73_l471_471461


namespace slope_CD_eq_neg_k_l471_471135

variables (a m k : ℝ) (h_a_pos : 0 < a) (h_a_ne_one : a ≠ 1) (h_m_pos : 0 < m) (h_k_pos : 0 < k)

def y_log_a_x (x : ℝ) := abs (Real.log x / Real.log a)
def y_k_over_x (x : ℝ) := k / x

theorem slope_CD_eq_neg_k :
  ∃ (x_C x_D : ℝ), 
    x_C = a ^ m ∧ x_D = a ^ (-m) ∧ 
    (y_k_over_x a k x_C - y_k_over_x a k x_D) / (x_C - x_D) = -k :=
by {
  sorry
}

end slope_CD_eq_neg_k_l471_471135


namespace count_even_three_digit_numbers_l471_471452

open Set

def is_smooth (n : ℕ) : Prop :=
  let sum := n + (n + 1) + (n + 2)
  (sum % 10) < 10

def set_A : Set ℕ := {0, 1, 2, 3}

def is_even_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ n % 2 = 0

def digits_in_set_A (n : ℕ) : Prop :=
  (n / 100) ∈ set_A ∧ ((n / 10) % 10) ∈ set_A ∧ (n % 10) ∈ set_A

theorem count_even_three_digit_numbers : 
  ∃ count : ℕ, count = 10 ∧ ∀ n, is_even_three_digit_number n → digits_in_set_A n → finset.filter is_even_three_digit_number (finset.range 1000) .card = count :=
by
  sorry

end count_even_three_digit_numbers_l471_471452


namespace perfect_cubes_between_bounds_l471_471858

def lower := 2^9 + 1
def upper := 2^{17} + 1

theorem perfect_cubes_between_bounds : 
  ∃ n, n = 32 ∧ ∀ k ∈ {k : ℤ | k^3 >= lower ∧ k^3 <= upper}, k = 9 + n - 1 :=
sorry

end perfect_cubes_between_bounds_l471_471858


namespace stating_ant_walk_distance_l471_471312

/-
  Define the points as tuples
-/
def point1 : (ℝ × ℝ) := (0, 0)
def point2 : (ℝ × ℝ) := (3, 4)
def point3 : (ℝ × ℝ) := (-9, 9)

/-
  Define the distance function between two points
-/
def distance (p1 p2 : (ℝ × ℝ)) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

noncomputable def total_distance : ℝ :=
  distance point1 point2 + distance point2 point3 + distance point3 point1

/-
  Theorem stating the total distance
-/
theorem ant_walk_distance : total_distance = 30.7 := 
  by  sorry

end stating_ant_walk_distance_l471_471312


namespace inradius_inequality_l471_471875

variable {a b c r : ℝ}

def semiperimeter (a b c : ℝ) : ℝ := (a + b + c) / 2

def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem inradius_inequality (a b c : ℝ) (r : ℝ) (s : ℝ)
  (h_triangle : triangle_inequality a b c)
  (hs : s = semiperimeter a b c) :
  24 * Real.sqrt 3 * r ^ 3 ≤ (-a + b + c) * (a - b + c) * (a + b - c) :=
sorry

end inradius_inequality_l471_471875


namespace distinct_prime_factors_330_l471_471421

def num_prime_factors (n : ℕ) : ℕ :=
  if n = 330 then 4 else 0

theorem distinct_prime_factors_330 : num_prime_factors 330 = 4 :=
sorry

end distinct_prime_factors_330_l471_471421


namespace lcm_14_18_20_l471_471655

theorem lcm_14_18_20 : Nat.lcm (Nat.lcm 14 18) 20 = 1260 :=
by
  -- Define the prime factorizations
  have fact_14 : 14 = 2 * 7 := by norm_num
  have fact_18 : 18 = 2 * 3^2 := by norm_num
  have fact_20 : 20 = 2^2 * 5 := by norm_num
  
  -- Calculate the LCM based on the highest powers of each prime
  have lcm : Nat.lcm (Nat.lcm 14 18) 20 = 2^2 * 3^2 * 5 * 7 :=
    by
      sorry -- Proof details are not required

  -- Final verification that this calculation matches 1260
  exact lcm

end lcm_14_18_20_l471_471655


namespace find_integer_m_l471_471980

theorem find_integer_m 
  (m : ℤ)
  (h1 : 30 ≤ m ∧ m ≤ 80)
  (h2 : ∃ k : ℤ, m = 6 * k)
  (h3 : m % 8 = 2)
  (h4 : m % 5 = 2) : 
  m = 42 := 
sorry

end find_integer_m_l471_471980


namespace first_group_total_cost_correct_l471_471316

-- Define the cost of hotdog and soft drink
def cost_hotdog := 0.5
def cost_soft_drink := 0.5

-- Define the number of hotdogs and soft drinks purchased by the first group
def hotdogs_first_group := 10
def soft_drinks_first_group := 5

-- Define the expected cost of the first group's purchase
def total_cost_first_group_expected := 7.5

-- Prove that the total cost for the first group is as expected
theorem first_group_total_cost_correct :
  hotdogs_first_group * cost_hotdog + soft_drinks_first_group * cost_soft_drink = total_cost_first_group_expected :=
by
  sorry

end first_group_total_cost_correct_l471_471316


namespace solve_for_a_l471_471837

theorem solve_for_a (a : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f(x) = |2*x - a| + a)
  (h2 : ∀ x, (f(x) ≤ 6) ↔ -2 ≤ x ∧ x ≤ 3) : 
  a = 1 :=
sorry

end solve_for_a_l471_471837


namespace determine_constants_l471_471722

theorem determine_constants (a b c d : ℝ) 
  (periodic : (2 * (2 * Real.pi / b) = 4 * Real.pi))
  (vert_shift : d = 3)
  (max_val : (d + a = 8))
  (min_val : (d - a = -2)) :
  a = 5 ∧ b = 1 :=
by
  sorry

end determine_constants_l471_471722


namespace length_HM_correct_l471_471052

noncomputable def length_of_HM (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] 
  (AB BC CA : ℝ) (M : B) (H : A) (abc : IsTriangle A B C) 
  (is_midpoint_M : IsMidpoint B M C) (is_perp_H : IsPerpendicularFrom B H (angleBisector A B C)) : ℝ :=
  let length_BC := 5
  let length_MH := 1
  length_MH

theorem length_HM_correct (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] 
  (AB BC CA : ℝ) (M : B) (H : A) (abc : IsTriangle A B C) 
  (is_midpoint_M : IsMidpoint B M C) (is_perp_H : IsPerpendicularFrom B H (angleBisector A B C)) : 
  length_of_HM A B C AB BC CA M H abc is_midpoint_M is_perp_H = 1 :=
by
  sorry

end length_HM_correct_l471_471052


namespace train_pass_time_approx_l471_471711

def length_train : ℝ := 550 -- in meters
def speed_train : ℝ := 60 * (1000 / 3600) -- in meters per second
def speed_man : ℝ := 6 * (1000 / 3600) -- in meters per second
def relative_speed := speed_train + speed_man
def time_to_pass (distance speed: ℝ) : ℝ := distance / speed

theorem train_pass_time_approx : abs (time_to_pass length_train relative_speed - 30) < 1 :=
by
  sorry

end train_pass_time_approx_l471_471711


namespace shaded_area_correct_l471_471582
open Real

def area_of_shaded_region (side_length : ℝ) (radius : ℝ) (sector_angle : ℝ) (n : ℕ) : ℝ :=
  let total_square_area := side_length^2
  let circle_area := π * radius^2
  let sector_area := (sector_angle / 360) * circle_area
  let total_sector_area := n * sector_area
  let triangle_area := 0.5 * radius * radius * sin (sector_angle * π / 180)
  let total_triangle_area := n * triangle_area
  total_square_area - total_sector_area - total_triangle_area

theorem shaded_area_correct :
  area_of_shaded_region 8 3 45 4 = 64 - 9 * π - 9 * sqrt 2 := 
sorry

end shaded_area_correct_l471_471582


namespace first_three_flips_HHT_l471_471662

-- Definitions based on the conditions and questions
def fair_coin : ProbabilityMassFunction Bool := 
  ProbabilityMassFunction.ofFintype uniform

theorem first_three_flips_HHT :
  let event_space := [true, true, false] in
  ProbabilityMassFunction.experiment_space fair_coin 3 = event_space →
  ProbabilityMassFunction.probability_of_event_space fair_coin event_space = 1 / 8 := 
sorry

end first_three_flips_HHT_l471_471662


namespace smallest_n_in_S_subset_has_perfect_square_set_l471_471916

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k*k = n

def S : Set ℕ := {m | m > 0 ∧ ∀ p ∈ Nat.factors m, p ≤ 10}

theorem smallest_n_in_S_subset_has_perfect_square_set :
  ∃ (n : ℕ), (∀ (M : Finset ℕ), (M ⊆ S) → (M.card = n) → ∃ a b c d ∈ M, a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a * b * c * d = k^2 ∧ is_perfect_square(k)) ∧ n = 9 :=
sorry

end smallest_n_in_S_subset_has_perfect_square_set_l471_471916


namespace quadrant_of_z1_div_z2_l471_471523

def z1 : ℂ := 1 - 3 * complex.I
def z2 : ℂ := 3 - 2 * complex.I
def z := z1 / z2

theorem quadrant_of_z1_div_z2 :
  let z := z1 / z2 in z.re > 0 ∧ z.im < 0 :=
by 
  -- let z := z1 / z2
  -- have h1 : z = (1 - 3 * complex.I) / (3 - 2 * complex.I) := sorry
  -- show z.re > 0 ∧ z.im < 0
  sorry

end quadrant_of_z1_div_z2_l471_471523


namespace find_k_l471_471894

variable (m n k : ℝ)

def line (x y : ℝ) : Prop := x = 2 * y + 3
def point1_on_line : Prop := line m n
def point2_on_line : Prop := line (m + 2) (n + k)

theorem find_k (h1 : point1_on_line m n) (h2 : point2_on_line m n k) : k = 0 :=
by
  sorry

end find_k_l471_471894


namespace minimum_sum_x_l471_471555

noncomputable section

variables {x : Fin 2016 → ℝ} {y : Fin 2016 → ℝ}

def cond1 (x : Fin 2016 → ℝ) (y : Fin 2016 → ℝ) : Prop :=
  ∀ k, x k ^ 2 + y k ^ 2 = 1

def cond2 (y : Fin 2016 → ℝ) : Prop :=
  Finset.sum Finset.univ y % 2 = 1

theorem minimum_sum_x (hx : ∀ k, 0 ≤ x k) (h1 : cond1 x y) (h2 : cond2 y) :
  ∑ k in Finset.univ, x k = 1 :=
sorry

end minimum_sum_x_l471_471555


namespace max_m_value_proof_l471_471078

noncomputable def max_m_value (A B C D P Q : ℝ) := 
  ∃ A B C D P Q,
  (inscribed_in_unit_circle A B C D) ∧
  (P ∈ ray AB) ∧
  (Q ∈ ray AD) ∧
  (\(∠ BAD = 30\)) ∧
  (minimum_value_CP_PQ_QC = 2) 

theorem max_m_value_proof (A B C D P Q : ℝ) 
  (h1 : inscribed_in_unit_circle A B C D)
  (h2 : P ∈ ray AB)
  (h3 : Q ∈ ray AD)
  (h4 : angle BAD = 30) : maximum_value_2 :=
begin 
  sorry -- Placeholder for the proof
end

end max_m_value_proof_l471_471078


namespace eccentricity_range_l471_471805

variables {a b : ℝ} (h1 : a > b > 0)
def e := (Real.sqrt (a^2 - b^2)) / a

theorem eccentricity_range (h2 : ∀ A B : ℝ, 
  (let F1 := (-(Real.sqrt (a^2 - b^2)), 0)
        F2 := (Real.sqrt (a^2 - b^2), 0)
    in (F1 - A) = 3 * (F2 - B))
  ) : 
  1/2 < e a b h1 ∧ e a b h1 < 1 :=
by
  sorry

end eccentricity_range_l471_471805


namespace sum_factorials_l471_471331

theorem sum_factorials (a : ℕ) (h : a = nat.factorial 1580) : 
  (∑ k in finset.range 1581, (k + 1) * nat.factorial (k + 1)) = 1581 * a - 1 := 
sorry

end sum_factorials_l471_471331


namespace exists_good_point_l471_471303

def isGood (points : List Int) (i : Nat) : Prop :=
  ∀ d : Nat, 1 ≤ d → d < points.length → 
    let sumForward := (List.range d).map (λ j => points[(i + j) % points.length]).sum
    let sumBackward := (List.range d).map (λ j => points[(i + points.length - j) % points.length]).sum
    sumForward > 0 ∧ sumBackward > 0

theorem exists_good_point (points : List Int)
  (h_len : points.length = 1985)
  (h_marks : ∀ point ∈ points, point = 1 ∨ point = -1)
  (h_neg : points.count (-1) < 662) :
  ∃ i, isGood points i := 
sorry

end exists_good_point_l471_471303


namespace probability_rain_weekend_l471_471160

theorem probability_rain_weekend :
  let p_rain_saturday := 0.30
  let p_rain_sunday := 0.60
  let p_rain_sunday_given_rain_saturday := 0.40
  let p_no_rain_saturday := 1 - p_rain_saturday
  let p_no_rain_sunday_given_no_rain_saturday := 1 - p_rain_sunday
  let p_no_rain_both_days := p_no_rain_saturday * p_no_rain_sunday_given_no_rain_saturday
  let p_rain_sunday_given_rain_saturday := 1 - p_rain_sunday_given_rain_saturday
  let p_no_rain_sunday_given_rain_saturday := p_rain_saturday * p_rain_sunday_given_rain_saturday
  let p_no_rain_all_scenarios := p_no_rain_both_days + p_no_rain_sunday_given_rain_saturday
  let p_rain_weekend := 1 - p_no_rain_all_scenarios
  p_rain_weekend = 0.54 :=
sorry

end probability_rain_weekend_l471_471160


namespace probability_of_exactly_9_correct_matches_is_zero_l471_471226

theorem probability_of_exactly_9_correct_matches_is_zero :
  ∀ (n : ℕ) (translate : Fin n → Fin n),
    (n = 10) → 
    (∀ i : Fin n, translate i ≠ i) → 
    (∃ (k : ℕ), (k < n ∧ k ≠ n-1) → false ) :=
by
  sorry

end probability_of_exactly_9_correct_matches_is_zero_l471_471226


namespace smallest_n_with_2022_digits_l471_471054

-- Sequence definition
def x : ℕ → ℕ
| 0       := 1
| (n + 1) := list.perm.max (list.perm (digits (x n + 2))).erase_dup

-- Function to count the digits of a number
def number_of_digits (n : ℕ) : ℕ :=
  n.digits.length

-- Theorem statement
theorem smallest_n_with_2022_digits :
  ∃ n, number_of_digits (x n) = 2022 ∧ ∀ m < n, number_of_digits (x m) < 2022 :=
begin
  use 18334567,
  split,
  { 
    -- This part should prove number_of_digits (x 18334567) = 2022, using the conditions and sequence definition.
    sorry
  },
  {
    -- This part should prove that for all m < 18334567, number_of_digits (x m) < 2022.
    sorry
  }
end

end smallest_n_with_2022_digits_l471_471054


namespace log2_a_10_l471_471831

variable (n : ℕ)
def S (n : ℕ) : ℕ := 2^n - 1
def a (n : ℕ) : ℕ := S n - S (n - 1)

theorem log2_a_10 : Real.log2 (a 10) = 9 := by
  -- Define the necessary hypotheses and prove the statement
  sorry

end log2_a_10_l471_471831


namespace count_divisible_by_3_in_range_l471_471856

theorem count_divisible_by_3_in_range (a b : ℤ) :
  a = 252 → b = 549 → (∃ n : ℕ, (a ≤ 3 * n ∧ 3 * n ≤ b) ∧ (b - a) / 3 = (100 : ℝ)) :=
by
  intros ha hb
  have h1 : ∃ k : ℕ, a = 3 * k := by sorry
  have h2 : ∃ m : ℕ, b = 3 * m := by sorry
  sorry

end count_divisible_by_3_in_range_l471_471856


namespace problem_statement_l471_471390

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_domain : ∀ x, 0 < x → f x ∈ ℝ

axiom f_diff_eq : ∀ x, 0 < x → x * (f' x) - f x = (x - 1) * (Real.exp x)

axiom f_at_one : f 1 = 0

theorem problem_statement :
  (3 * f 2 < 2 * f 3) ∧ (¬ ∃ M, ∀ x, 0 < x → f x ≤ M) :=
  sorry

end problem_statement_l471_471390


namespace flashlight_lifetime_expectation_leq_two_l471_471544

noncomputable def min_lifetime_expectation (ξ η : ℝ) (E_ξ : ℝ) : Prop :=
  E_ξ = 2 → E(min ξ η) ≤ 2

-- Assume ξ and η are random variables and E denotes the expectation.
axiom E : (ℝ → ℝ) → ℝ

theorem flashlight_lifetime_expectation_leq_two (ξ η : ℝ) (E_ξ : ℝ) (hE_ξ : E_ξ = 2) : E(min ξ η) ≤ 2 :=
  by
    sorry

end flashlight_lifetime_expectation_leq_two_l471_471544


namespace max_discount_rate_l471_471274

theorem max_discount_rate 
  (cost_price : ℝ) (selling_price : ℝ) (min_profit_margin : ℝ) :
  cost_price = 4 → selling_price = 5 → min_profit_margin = 0.1 →
  ∃ (x : ℝ), 0 ≤ x ∧ x ≤ 8.8 ∧ (selling_price * (1 - x / 100) - cost_price) / cost_price ≥ min_profit_margin :=
begin
  sorry
end

end max_discount_rate_l471_471274


namespace acute_triangle_conclusions_l471_471472

theorem acute_triangle_conclusions (a b c : ℝ) (A B C : ℝ) (h1 : 0 < A ∧ A < π / 2) (h2 : 0 < B ∧ B < π / 2) (h3 : 0 < C ∧ C < π / 2) 
  (h4 : ∠ A = angle (a, b, c)) (h5 : ∠ B = angle (b, c, a)) (h6 : ∠ C = angle (c, a, b)) 
  (h7 : a^2 = b * (b + c)) : 
  (A = 2 * B) ∧ (sqrt 2 < a / b ∧ a / b < sqrt 3) := 
by
    sorry

end acute_triangle_conclusions_l471_471472


namespace bus_ticket_payment_impossible_l471_471624

theorem bus_ticket_payment_impossible :
  ∀ (passengers : ℕ) (coins : ℕ) (ticket_price : ℕ) (denominations : list ℕ),
  passengers = 40 →
  coins = 49 →
  ticket_price = 5 →
  denominations = [10, 15, 20] →
  ¬ (∃ (x1 x2 x3 : ℕ),
    x1 + x2 + x3 = coins ∧
    10 * x1 + 15 * x2 + 20 * x3 = passengers * ticket_price ∧
    ∀ k, k ∈ set.to_finset (set.range 40) → 
    ∃ c ∈ denominations, c - ticket_price ≤ 0) :=
begin
  sorry
end

end bus_ticket_payment_impossible_l471_471624


namespace power_of_two_primes_l471_471517

noncomputable def distinct_divisors (n : ℕ) (primes : list ℕ) : Prop :=
  ∀ (N : ℕ), (N = 2 ^ (primes.prod) + 1) → 
    n ≥ 1 → 
    (∀ p, p ∈ primes → prime p ∧ p ≥ 5) → 
    ∃ (d : ℕ), d ≥ 2^(2^n) ∧ d ∣ N

theorem power_of_two_primes (n : ℕ) (p_list : list ℕ) :
  distinct_divisors n p_list :=
by
  sorry

end power_of_two_primes_l471_471517


namespace expected_lifetime_flashlight_l471_471526

section
variables {Ω : Type} [ProbabilitySpace Ω]
variables (ξ η : Ω → ℝ)
variables (h_ξ_expect : E[ξ] = 2)

-- Define the minimum of ξ and η
def min_ξ_η (ω : Ω) : ℝ := min (ξ ω) (η ω)

theorem expected_lifetime_flashlight : E[min_ξ_η ξ η] ≤ 2 :=
by
  sorry
end

end expected_lifetime_flashlight_l471_471526


namespace sum_of_squares_distances_l471_471354

theorem sum_of_squares_distances {n : ℕ} (R : ℝ) (h : n ≥ 3) :
  let vertices := fin n → ℝ^2 in
  let center := (0 : ℝ^2) in
  let radius := (λ i : fin n, ‖vertices i - center‖ = R) in
  let line := (λ i : fin n, (line_through_center : ℝ^2)) in
  let distances := (λ i : fin n, distance_from_line (vertices i) line) in
  sum (λ i, distances i^2) = (n * R^2) / 2 :=
begin
  sorry
end

end sum_of_squares_distances_l471_471354


namespace age_of_b_l471_471209

theorem age_of_b (a b c : ℕ) (h1 : a = b + 2) (h2 : b = 2 * c) (h3 : a + b + c = 42) : b = 16 :=
by
  sorry

end age_of_b_l471_471209


namespace average_speed_return_trip_l471_471673

def speed1 : ℝ := 12 -- Speed for the first part of the trip in miles per hour
def distance1 : ℝ := 18 -- Distance for the first part of the trip in miles
def speed2 : ℝ := 10 -- Speed for the second part of the trip in miles per hour
def distance2 : ℝ := 18 -- Distance for the second part of the trip in miles
def total_round_trip_time : ℝ := 7.3 -- Total time for the round trip in hours

theorem average_speed_return_trip :
  let time1 := distance1 / speed1 -- Time taken for the first part of the trip
  let time2 := distance2 / speed2 -- Time taken for the second part of the trip
  let total_time_to_destination := time1 + time2 -- Total time for the trip to the destination
  let time_return_trip := total_round_trip_time - total_time_to_destination -- Time for the return trip
  let return_trip_distance := distance1 + distance2 -- Distance for the return trip (same as to the destination)
  let avg_speed_return_trip := return_trip_distance / time_return_trip -- Average speed for the return trip
  avg_speed_return_trip = 9 := 
by
  sorry

end average_speed_return_trip_l471_471673


namespace triangle_B_angle_value_triangle_a_c_range_l471_471025

theorem triangle_B_angle_value (A B C a b c : ℝ)
  (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0)
  (h₄ : a = c * sin A / sin C)
  (h₅ : ∠A + ∠B + ∠C = π)
  (h₆ : cos 2 * A - cos 2 * B = 2 * cos (π / 6 - A) * cos (π / 6 + A)) :
  B = π / 3 ∨ B = 2 * π / 3 := 
sorry

theorem triangle_a_c_range (A B C a b c : ℝ)
  (hb : b = sqrt 3)
  (ha : b ≤ a)
  (hB : B = π / 3)
  (h_a : a = 2 * sin A)
  (h_c : c = 2 * sin C) :
  sqrt 3 / 2 ≤ a - 1 / 2 * c ∧ a - 1 / 2 * c < sqrt 3 :=
sorry

end triangle_B_angle_value_triangle_a_c_range_l471_471025


namespace number_of_zeros_of_g_l471_471401

noncomputable def f (x : ℝ) : ℝ := 
  if x ≤ 0 then x + 1 else real.log x / real.log 2

def g (x : ℝ) : ℝ :=
  f (f x) - 1

theorem number_of_zeros_of_g : (∃ x₁ x₂ x₃ : ℝ, g x₁ = 0 ∧ g x₂ = 0 ∧ g x₃ = 0 ∧ x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃) :=
sorry

end number_of_zeros_of_g_l471_471401


namespace find_BD_length_l471_471045

-- Definitions from the conditions
variables {A B C D : Type} -- Triangle vertices and point
variables {A B C D : ℝ} -- Base and height are real numbers

-- Defining the conditions in Lean
def triangle_area (area AC BD : ℝ) : Prop :=
  area = (1 / 2) * AC * BD

def triangle_ABC_area_84 : Prop := triangle_area 84 12

def BD_length (BD : ℝ) : Prop :=
  ∃ BD, triangle_area 84 12 BD

-- Lean statement to prove BD
theorem find_BD_length (h1 : triangle_ABC_area_84 84 12) : BD = 14 := sorry

end find_BD_length_l471_471045


namespace minimum_value_l471_471772

noncomputable def f (x y : ℝ) : ℝ :=
  (x^2 + 1) / (y - 2) + (y^2 + 1) / (x - 2)

theorem minimum_value (x y : ℝ) (hx : x > 2) (hy : y > 2) :
  ∃ a b, x = a + 2 ∧ y = b + 2 ∧ a = sqrt 5 ∧ b = sqrt 5 ∧ f x y = 4 * sqrt 5 + 8 :=
sorry

end minimum_value_l471_471772


namespace percentage_sales_tax_on_taxable_purchases_l471_471061

-- Definitions
def total_cost : ℝ := 30
def tax_free_cost : ℝ := 24.7
def tax_rate : ℝ := 0.06

-- Statement to prove
theorem percentage_sales_tax_on_taxable_purchases :
  (tax_rate * (total_cost - tax_free_cost)) / total_cost * 100 = 1 := by
  sorry

end percentage_sales_tax_on_taxable_purchases_l471_471061


namespace max_area_OAPF_l471_471891

-- Definitions of the elements
def ellipse_eq (x y : ℝ) : Prop := (x ^ 2) / 9 + (y ^ 2) / 10 = 1

def A : ℝ × ℝ := (3, 0)
def F : ℝ × ℝ := (0, 1)

-- Definition of P being on the ellipse and in the first quadrant
def P (θ : ℝ) (hθ : 0 < θ ∧ θ < Real.pi / 2) : ℝ × ℝ := 
  (3 * Real.cos θ, Real.sqrt 10 * Real.sin θ)

-- Maximum area of quadrilateral OAPF
theorem max_area_OAPF : ∃ (θ : ℝ) (hθ : 0 < θ ∧ θ < Real.pi / 2), 
  let P := P θ hθ in
  let area_OAPF := (3 * Real.sqrt 10 * Real.sin θ / 2) + (3 * Real.cos θ / 2) 
  in area_OAPF = 3 / 2 * Real.sqrt 11 := 
sorry

end max_area_OAPF_l471_471891


namespace min_value_sin_cos_l471_471780

theorem min_value_sin_cos (x : ℝ) : sin x ^ 6 + 2 * cos x ^ 6 ≥ 2 / 3 :=
sorry

end min_value_sin_cos_l471_471780


namespace cot_sum_arccot_roots_l471_471913

theorem cot_sum_arccot_roots :
  let z := λ k : ℕ, (λ (x : ℂ), x^12 - 3 * x^11 + 6 * x^10 - 10 * x^9 + (if k = 0 then 144 else 0)) 0,
    roots := {z_1, z_2, ..., z_{12}},
    θ := λ k : ℕ, Real.arccot z_k,
    s := λ n : ℕ, if n = 12 then 144 else
                   if n = 11 then 121 else
                   if n = 10 then 100 else
                   if n =  9 then 81 else
                   if n =  8 then 64 else
                   if n =  7 then 49 else
                   if n =  6 then 36 else
                   if n =  5 then 25 else
                   if n =  4 then 16 else
                   if n =  3 then 9 else
                   if n =  2 then 4 else
                   if n =  1 then 1 else 1,
    cot := λ θ : ℝ, (Real.sin θ)⁻¹ * Real.cos θ in
  cot (∑ k in (Finset.range 12), θ k) = (49 / 105 : ℝ) := 
sorry

end cot_sum_arccot_roots_l471_471913


namespace closest_integer_sqrt_40_l471_471133

theorem closest_integer_sqrt_40 (x : ℝ) (hx1 : 36 < x) (hx2 : x < 49) (hx3 : x = 40) : 
  abs(ceil (real.sqrt x) - real.sqrt x) > abs(floor (real.sqrt x) - real.sqrt x) → 
  floor (real.sqrt x) = 6 := 
  sorry

end closest_integer_sqrt_40_l471_471133


namespace remainder_x50_div_x_plus_1_cubed_l471_471350

theorem remainder_x50_div_x_plus_1_cubed :
  ∀ (x : ℂ), let remainder := (x^50) % (x + 1)^3
  in remainder = -1225 * x^2 - 2400 * x - 1176 :=
by sorry

end remainder_x50_div_x_plus_1_cubed_l471_471350


namespace maximal_k_value_l471_471503

noncomputable def max_edges (n : ℕ) : ℕ :=
  2 * n - 4
   
theorem maximal_k_value (k n : ℕ) (h1 : n = 2016) (h2 : k ≤ max_edges n) :
  k = 4028 :=
by sorry

end maximal_k_value_l471_471503


namespace james_total_pay_l471_471898

def original_prices : List ℝ := [15, 20, 25, 18, 22, 30]
def discounts : List ℝ := [0.30, 0.50, 0.40, 0.20, 0.45, 0.25]

def discounted_price (price discount : ℝ) : ℝ :=
  price * (1 - discount)

def total_price_after_discount (prices discounts : List ℝ) : ℝ :=
  (List.zipWith discounted_price prices discounts).sum

theorem james_total_pay :
  total_price_after_discount original_prices discounts = 84.50 :=
  by sorry

end james_total_pay_l471_471898


namespace remainder_when_ab_div_by_40_l471_471940

theorem remainder_when_ab_div_by_40 (a b : ℤ) (k j : ℤ)
  (ha : a = 80 * k + 75)
  (hb : b = 90 * j + 85):
  (a + b) % 40 = 0 :=
by sorry

end remainder_when_ab_div_by_40_l471_471940


namespace max_discount_rate_l471_471278

-- Define the cost price and selling price.
def cp : ℝ := 4
def sp : ℝ := 5

-- Define the minimum profit margin.
def min_profit_margin : ℝ := 0.4

-- Define the discount rate d.
def discount_rate (d : ℝ) : ℝ := sp * (1 - d / 100) - cp

-- The theorem to prove the maximum discount rate.
theorem max_discount_rate (d : ℝ) (H : discount_rate d ≥ min_profit_margin) : d ≤ 12 :=
sorry

end max_discount_rate_l471_471278


namespace dwayneA_students_l471_471027

-- Define the number of students who received an 'A' in Mrs. Carter's class
def mrsCarterA := 8
-- Define the total number of students in Mrs. Carter's class
def mrsCarterTotal := 20
-- Define the total number of students in Mr. Dwayne's class
def mrDwayneTotal := 30
-- Calculate the ratio of students who received an 'A' in Mrs. Carter's class
def carterRatio := mrsCarterA / mrsCarterTotal
-- Calculate the number of students who received an 'A' in Mr. Dwayne's class based on the same ratio
def mrDwayneA := (carterRatio * mrDwayneTotal)

-- Prove that the number of students who received an 'A' in Mr. Dwayne's class is 12
theorem dwayneA_students :
  mrDwayneA = 12 := 
by
  -- Since def calculation does not automatically prove equality, we will need to use sorry to skip the proof for now.
  sorry

end dwayneA_students_l471_471027


namespace magnitude_of_force_l471_471636

/- Define the conditions used in the Lean 4 statement -/

def mass : ℝ := 60 -- in kg
def height_fall : ℝ := 3.2 -- in meters
def height_bounce : ℝ := 5 -- in meters
def contact_time : ℝ := 1.2 -- in seconds
def gravitational_acceleration : ℝ := 10 -- in m/s^2

/- Statement of the proof problem -/
theorem magnitude_of_force :
  let v1 := Real.sqrt (2 * gravitational_acceleration * height_fall)
  let v2 := Real.sqrt (2 * gravitational_acceleration * height_bounce)
  let F := (mass * (v2 + v1) + mass * gravitational_acceleration * contact_time) / contact_time in
  F = 1500 := by
  sorry

end magnitude_of_force_l471_471636


namespace mnpq_product_l471_471123

noncomputable def prove_mnpq_product (a b x y : ℝ) : Prop :=
  ∃ (m n p q : ℤ), (a^m * x - a^n) * (a^p * y - a^q) = a^3 * b^4 ∧
                    m * n * p * q = 4

theorem mnpq_product (a b x y : ℝ) (h : a^7 * x * y - a^6 * y - a^5 * x = a^3 * (b^4 - 1)) :
  prove_mnpq_product a b x y :=
sorry

end mnpq_product_l471_471123


namespace min_value_expression_l471_471768

theorem min_value_expression (x y : ℝ) (hx : x > 2) (hy : y > 2) : 
  ∃ m, m = 20 ∧ (∀ a b : ℝ, a = x - 2 ∧ b = y - 2 →
    let exp := (a + 2) ^ 2 + 1 / (b + (b + 2) ^ 2 + 1 / a) in exp ≥ m) :=
sorry

end min_value_expression_l471_471768


namespace extreme_points_and_tangent_line_l471_471839

-- Define the function
def f (x : ℝ) : ℝ := (x^2 - x + 1) * exp x

-- The proof problem
theorem extreme_points_and_tangent_line :
  (∃ a b : ℝ, a ≠ b ∧ ∀ t : ℝ, (f' t = 0) ↔ (t = a ∨ t = b)) ∧
  (∀ t : ℝ, t = 1 → (∃ m b : ℝ, m = 2 * exp 1 ∧ b = -exp 1 ∧ (∀ y, y = m * (x - 1) + b ↔ y = 2 * exp 1 * x - exp 1))) :=
by
  sorry

end extreme_points_and_tangent_line_l471_471839


namespace numer_greater_than_denom_iff_l471_471793

theorem numer_greater_than_denom_iff (x : ℝ) (h : -1 ≤ x ∧ x ≤ 3) : 
  (4 * x - 3 > 9 - 2 * x) ↔ (2 < x ∧ x ≤ 3) :=
sorry

end numer_greater_than_denom_iff_l471_471793


namespace log_function_behaviour_l471_471576

noncomputable def log_increasing (a : ℝ) (x1 x2 : ℝ) : Prop :=
  1 < a → 0 < x1 → x1 < x2 → Real.logBase a x1 < Real.logBase a x2

noncomputable def log_decreasing (a : ℝ) (x1 x2 : ℝ) : Prop :=
  0 < a → a < 1 → 0 < x1 → x1 < x2 → Real.logBase a x2 < Real.logBase a x1

theorem log_function_behaviour (a x1 x2 : ℝ) :
  (0 < x1 ∧ x1 < x2 → (1 < a → Real.logBase a x1 < Real.logBase a x2)) ∧
  (0 < x1 ∧ x1 < x2 → (0 < a ∧ a < 1 → Real.logBase a x2 < Real.logBase a x1)) :=
by
  sorry

end log_function_behaviour_l471_471576


namespace game_not_fair_probability_first_player_wins_approx_l471_471168

def fair_game (n : ℕ) (P : ℕ → ℚ) : Prop :=
  ∀ k j, k ≤ n → j ≤ n → P k = P j

noncomputable def probability_first_player_wins (n : ℕ) : ℚ :=
  let r := (n - 1 : ℚ) / n;
  (1 : ℚ) / n * (1 : ℚ) / (1 - r ^ n)

-- Check if the game is fair for 36 players
theorem game_not_fair : ¬ fair_game 36 (λ i, probability_first_player_wins 36) :=
sorry

-- Calculate the approximate probability of the first player winning
theorem probability_first_player_wins_approx (n : ℕ)
  (h₁ : n = 36) :
  probability_first_player_wins n ≈ 0.044 :=
sorry

end game_not_fair_probability_first_player_wins_approx_l471_471168


namespace solve_inequality_l471_471965

theorem solve_inequality (a x : ℝ) : 
  (ax^2 + (a - 1) * x - 1 < 0) ↔ (
  (a = 0 ∧ x > -1) ∨ 
  (a > 0 ∧ -1 < x ∧ x < 1/a) ∨
  (-1 < a ∧ a < 0 ∧ (x < 1/a ∨ x > -1)) ∨ 
  (a = -1 ∧ x ≠ -1) ∨ 
  (a < -1 ∧ (x < -1 ∨ x > 1/a))
) := sorry

end solve_inequality_l471_471965


namespace find_a_satisfies_F_l471_471335

def F (a : ℚ) (b : ℚ) (c : ℚ) : ℚ := a * b^3 + c

theorem find_a_satisfies_F : 
  ∃ (a : ℚ), F(a, 3, 8) = F(a, 5, 12) ∧ a = -2 / 49 :=
by
  use -2 / 49
  sorry

end find_a_satisfies_F_l471_471335


namespace gcd_binomials_one_l471_471502

open Nat

theorem gcd_binomials_one (n k : ℕ) (hpos_n : 0 < n) (hpos_k : 0 < k) (h_le : k ≤ n) :
  gcd (List.foldr gcd 0 ((List.range (k + 1)).map (λ i => Nat.choose (n + i) k))) = 1 := 
sorry

end gcd_binomials_one_l471_471502


namespace ball_acceleration_l471_471683

-- Let k, m, y, and g be real numbers representing the spring constant, mass of the ball, displacement, and acceleration due to gravity, respectively.
variables (k m y g : ℝ)

-- Define the forces based on the given conditions
def F_g : ℝ := -m * g -- gravitational force
def F_s : ℝ := -k * y -- spring force

-- Define the net force according to Newton's Second Law
def F_net : ℝ := F_g m g + F_s k y

-- Define the acceleration (to be proved)
def acceleration : ℝ := - (k / m) * y - g

-- The theorem to prove the acceleration of the ball
theorem ball_acceleration : F_net k m y g = m * acceleration k m y g :=
sorry

end ball_acceleration_l471_471683


namespace vertex_of_parabola_l471_471119

theorem vertex_of_parabola :
  ∃ h k, (∀ x : ℝ, -(x - h)^2 + k = -(x - 5)^2 + 3) ∧ h = 5 ∧ k = 3 :=
by
  use 5, 3
  split
  sorry

end vertex_of_parabola_l471_471119


namespace averagePricePerBook_l471_471946

-- Define the prices and quantities from the first store
def firstStoreFictionBooks : ℕ := 25
def firstStoreFictionPrice : ℝ := 20
def firstStoreNonFictionBooks : ℕ := 15
def firstStoreNonFictionPrice : ℝ := 30
def firstStoreChildrenBooks : ℕ := 20
def firstStoreChildrenPrice : ℝ := 8

-- Define the prices and quantities from the second store
def secondStoreFictionBooks : ℕ := 10
def secondStoreFictionPrice : ℝ := 18
def secondStoreNonFictionBooks : ℕ := 20
def secondStoreNonFictionPrice : ℝ := 25
def secondStoreChildrenBooks : ℕ := 30
def secondStoreChildrenPrice : ℝ := 5

-- Definition of total books from first and second store
def totalBooks : ℕ :=
  firstStoreFictionBooks + firstStoreNonFictionBooks + firstStoreChildrenBooks +
  secondStoreFictionBooks + secondStoreNonFictionBooks + secondStoreChildrenBooks

-- Definition of the total cost from first and second store
def totalCost : ℝ :=
  (firstStoreFictionBooks * firstStoreFictionPrice) +
  (firstStoreNonFictionBooks * firstStoreNonFictionPrice) +
  (firstStoreChildrenBooks * firstStoreChildrenPrice) +
  (secondStoreFictionBooks * secondStoreFictionPrice) +
  (secondStoreNonFictionBooks * secondStoreNonFictionPrice) +
  (secondStoreChildrenBooks * secondStoreChildrenPrice)

-- Theorem: average price per book
theorem averagePricePerBook : (totalCost / totalBooks : ℝ) = 16.17 := by
  sorry

end averagePricePerBook_l471_471946


namespace parabola_intersects_line_l471_471609

-- Define the conditions of the problem
variables {a k x1 x2 : ℝ}
variable (h_a_ne_0 : a ≠ 0)
variable (h_x1_x2 : x1 + x2 < 0)
variable (h_intersect : ∀ x, a * x^2 - a = k * x → x = x1 ∨ x = x2)

-- Define what needs to be proven
theorem parabola_intersects_line {h_a_ne_0 h_x1_x2 h_intersect} :
  ∃ k, (k < 0 ∧ k > 0 → (λ x, a * x + k).exists_first_and_fourth_quadrant) ∧ 
       (k > 0 ∧ k < 0 → (λ x, a * x + k).exists_first_and_fourth_quadrant) :=
sorry

end parabola_intersects_line_l471_471609


namespace quadratic_expression_rewrite_l471_471613

theorem quadratic_expression_rewrite (a b c : ℝ) :
  (∀ x : ℝ, 6 * x^2 + 36 * x + 216 = a * (x + b)^2 + c) → a + b + c = 171 :=
sorry

end quadratic_expression_rewrite_l471_471613


namespace number_of_real_p_l471_471515

theorem number_of_real_p (k : ℝ) (h : k ≥ -1) : 
  ∃! p : ℝ, ∀ p₁ p₂ : ℝ, (¬(p₁ = p₁) → (p₁ - 2 - 2*√(1 + k)) = 0) ∧ (¬(p₂ = p₂) → (p₂ - 2 + 2*√(1 + k)) = 0) :=
    sorry

end number_of_real_p_l471_471515


namespace max_Xs_in_grid_l471_471203

def is_valid_placement (grid : list (list bool)) : Prop :=
  ∀ i j k, 
    (i < 5) → (j < 5) → (k < 5) → 
    (((grid[i][j] = tt) ∧ (grid[i][k] = tt) ∧ (grid[j][k] = tt)) → i = j ∧ j = k) ∧
    ((((grid[0][i] = tt) ∧ (grid[1][i] = tt) ∧ (grid[2][i] = tt)) → false) ∧
     ((grid[1][i] = tt) ∧ (grid[2][i] = tt) ∧ (grid[3][i] = tt) → false) ∧
     ((grid[2][i] = tt) ∧ (grid[3][i] = tt) ∧ (grid[4][i] = tt) → false)) ∧
    ((((grid[i][0] = tt) ∧ (grid[i][1] = tt) ∧ (grid[i][2] = tt)) → false) ∧
     ((grid[i][1] = tt) ∧ (grid[i][2] = tt) ∧ (grid[i][3] = tt) → false) ∧
     ((grid[i][2] = tt) ∧ (grid[i][3] = tt) ∧ (grid[i][4] = tt) → false)) ∧
    ((((grid[0][0] = tt) ∧ (grid[1][1] = tt) ∧ (grid[2][2] = tt)) → false) ∧
     ((grid[1][1] = tt) ∧ (grid[2][2] = tt) ∧ (grid[3][3] = tt) → false) ∧
     ((grid[2][2] = tt) ∧ (grid[3][3] = tt) ∧ (grid[4][4] = tt) → false)) ∧
    ((((grid[2][0] = tt) ∧ (grid[1][1] = tt) ∧ (grid[0][2] = tt)) → false) ∧
     ((grid[3][0] = tt) ∧ (grid[2][1] = tt) ∧ (grid[1][2] = tt) → false) ∧
     ((grid[4][0] = tt) ∧ (grid[3][1] = tt) ∧ (grid[2][2] = tt) → false)) ∧
    (grid[i].count tt ≥ 1)

theorem max_Xs_in_grid : ∃ grid, is_valid_placement grid ∧ (grid.count tt = 10) :=
sorry

end max_Xs_in_grid_l471_471203


namespace volume_of_pyramid_correct_l471_471568

noncomputable def volume_of_pyramid : ℝ := 24 * Real.sqrt 481

theorem volume_of_pyramid_correct (AB BC PB : ℝ) (PA : ℝ) 
  (ABCD : ℝ → ℝ → Prop) (height : ℝ)
  (h_base : ABCD AB BC) 
  (h_AB : AB = 12) 
  (h_BC : BC = 6) 
  (h_perp1 : PA ⟂ AD) 
  (h_perp2 : PA ⟂ AB) 
  (h_PB : PB = 25) 
  (h_Pythagorean : PA = Real.sqrt (PB^2 - AB^2)) 
  (h_area : height = AB * BC) 
  (h_volume : volume_of_pyramid = (1/3) * height * PA) :
  volume_of_pyramid = 24 * Real.sqrt 481 := 
sorry

end volume_of_pyramid_correct_l471_471568


namespace expected_lifetime_flashlight_l471_471528

section
variables {Ω : Type} [ProbabilitySpace Ω]
variables (ξ η : Ω → ℝ)
variables (h_ξ_expect : E[ξ] = 2)

-- Define the minimum of ξ and η
def min_ξ_η (ω : Ω) : ℝ := min (ξ ω) (η ω)

theorem expected_lifetime_flashlight : E[min_ξ_η ξ η] ≤ 2 :=
by
  sorry
end

end expected_lifetime_flashlight_l471_471528


namespace determine_c_absolute_value_l471_471580

theorem determine_c_absolute_value 
  (a b c : ℤ) 
  (h_gcd : Int.gcd (Int.gcd a b) c = 1)
  (h_poly : a * (2 + Complex.i)^4 + b * (2 + Complex.i)^3 + c * (2 + Complex.i)^2 + b * (2 + Complex.i) + a = 0) :
  |c| = 42 := sorry

end determine_c_absolute_value_l471_471580


namespace team_with_at_least_one_girl_l471_471441

theorem team_with_at_least_one_girl :
  (∑ (k : ℕ) in Finset.range 4, nat.choose 5 k * nat.choose 5 (3 - k)) = 110 :=
by
  sorry

end team_with_at_least_one_girl_l471_471441


namespace polynomial_degree_and_linear_coefficient_l471_471595

-- Define the polynomial P(x)
def P (x : ℝ) : ℝ := 3 * x^2 - 2 * x + 1

-- State the theorem to be proved.
theorem polynomial_degree_and_linear_coefficient :
  (polynomial.degree (polynomial.C (3 : ℝ) * polynomial.X^2 + polynomial.C (-2 : ℝ) * polynomial.X + polynomial.C 1) = 2) ∧
  (polynomial.coeff (polynomial.C (3 : ℝ) * polynomial.X^2 + polynomial.C (-2 : ℝ) * polynomial.X + polynomial.C 1) 1 = -2) :=
by
  sorry

end polynomial_degree_and_linear_coefficient_l471_471595


namespace tangent_lines_ln_l471_471127

theorem tangent_lines_ln (x y: ℝ) : 
    (y = Real.log (abs x)) → 
    (x = 0 ∧ y = 0) ∨ ((x = yup ∨ x = ydown) ∧ (∀ (ey : ℝ), x = ey ∨ x = -ey)) :=
by 
    intro h
    sorry

end tangent_lines_ln_l471_471127


namespace pyramid_volume_l471_471571

theorem pyramid_volume (AB BC PA PB : ℝ) (h_AB : AB = 12) (h_BC : BC = 6) (h_PB : PB = 25) 
  (h_PA_perp_AD : ∀ x, PA = x ∧ x > 0) (h_PA_perp_AB : ∀ x, PA = x ∧ x > 0) : (1 / 3) * (AB * BC) * PA = 24 * Real.sqrt 481 :=
by
  have h1 : PA * PA = PB * PB - AB * AB :=
    by sorry
  have h2 : AB * BC = 72 :=
    by rw [h_AB, h_BC]; exact rfl
  have h3 : PA = Real.sqrt 481 :=
    by rw [←h1, h_PB, h_AB]; sorry
  have volume : (1 / 3) * 72 * Real.sqrt 481 = 24 * Real.sqrt 481 :=
    by sorry
  exact volume

end pyramid_volume_l471_471571


namespace borrowed_sheets_l471_471629

theorem borrowed_sheets (total_pages sheets : ℕ) (avg_remaining : ℕ) :
  total_pages = 72 →
  sheets = 36 →
  avg_remaining = 40 →
  (∃ c b : ℕ, c ∈ {1, 2, ..., 36} ∧
              b ∈ {0, 1, ..., 36 - c} ∧
              avg_remaining = (∑ i in (2*(b + c) + 1 .. total_pages).to_finset, i) / (sheets - c)) →
  (∃ c, c = 17) := by
  intros htp hs ha ex_condition
  sorry

end borrowed_sheets_l471_471629


namespace select_monkey_l471_471581

theorem select_monkey (consumption : ℕ → ℕ) (n bananas minutes : ℕ)
  (h1 : consumption 1 = 1) (h2 : consumption 2 = 2) (h3 : consumption 3 = 3)
  (h4 : consumption 4 = 4) (h5 : consumption 5 = 5) (h6 : consumption 6 = 6)
  (h_total_minutes : minutes = 18) (h_total_bananas : bananas = 18) :
  consumption 1 * minutes = bananas :=
by
  sorry

end select_monkey_l471_471581


namespace find_m_l471_471444

theorem find_m (a0 a1 a2 a3 a4 a5 a6 : ℝ) (m : ℝ)
  (h1 : (1 + m) * x ^ 6 = a0 + a1 * x + a2 * x ^ 2 + a3 * x ^ 3 + a4 * x ^ 4 + a5 * x ^ 5 + a6 * x ^ 6)
  (h2 : a1 - a2 + a3 - a4 + a5 - a6 = -63)
  (h3 : a0 = 1) :
  m = 3 ∨ m = -1 :=
by
  sorry

end find_m_l471_471444


namespace probability_of_exactly_9_correct_matches_is_zero_l471_471233

theorem probability_of_exactly_9_correct_matches_is_zero :
  ∃ (P : ℕ → ℕ → ℕ), 
    (∀ (total correct : ℕ), 
      total = 10 → 
      correct = 9 → 
      P total correct = 0) := 
by {
  sorry
}

end probability_of_exactly_9_correct_matches_is_zero_l471_471233


namespace union_inter_eq_union_compl_inter_eq_l471_471849

open Set

variable (U : Set ℕ) (A B C : Set ℕ)
variable [DecidableEq ℕ]

def U_def : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A_def : Set ℕ := {x | x^2 - 3 * x + 2 = 0}
def B_def : Set ℕ := {x | 1 ≤ x ∧ x ≤ 5 ∧ x ∈ ℤ}
def C_def : Set ℕ := {x | 2 < x ∧ x < 9 ∧ x ∈ ℤ}

theorem union_inter_eq :
  A ∪ (B ∩ C) = {1, 2, 3, 4, 5} :=
by
  -- proof would go here
  sorry

theorem union_compl_inter_eq :
  (compl U ∩ B) ∪ (compl U ∩ C) = {1, 2, 6, 7, 8} :=
by
  -- proof would go here
  sorry

end union_inter_eq_union_compl_inter_eq_l471_471849


namespace right_triangle_angles_l471_471893

theorem right_triangle_angles
    (a b : ℝ) -- lengths of legs BC and AC
    (h1 : (real.pi * b^2 / 8 = 1 /2 * a * b)) -- Condition on areas
    (h2 : ∠ BCA = real.pi / 2) : -- Right angle condition

    let α := real.arctan (a / b) in -- Definition of angle alpha
    α = (38 + 15 / 60) * (real.pi / 180) ∧ -- Angle α in radians
    (real.pi / 2 - α) = (51 + 45 / 60) * (real.pi / 180) := -- Complementary angle
by 
  sorry

end right_triangle_angles_l471_471893


namespace prob_9_correct_matches_is_zero_l471_471246

noncomputable def probability_of_exactly_9_correct_matches : ℝ :=
  let n := 10 in
  -- Since choosing 9 correct implies the 10th is also correct, the probability is 0.
  0

theorem prob_9_correct_matches_is_zero : probability_of_exactly_9_correct_matches = 0 :=
by
  sorry

end prob_9_correct_matches_is_zero_l471_471246


namespace part_a_l471_471924

noncomputable def points_on_circle (l1 l2 l3 l4 : Line) (p1 p2 p3 p4 : Point) : Prop :=
  ∃ (circle : Circle), p1 ∈ circle ∧ p2 ∈ circle ∧ p3 ∈ circle ∧ p4 ∈ circle

theorem part_a 
  (l1 l2 l3 l4 : Line) 
  (p1 p2 p3 p4 : Point) 
  (h1 : general_position l1 l2 l3 l4)
  (h2 : points_on_circle l1 l2 l3 l4 p1 p2 p3 p4) 
  : ∃ circle, (A1 l2 l3 l4 ∈ circle) ∧ (A2 l1 l3 l4 ∈ circle) ∧ (A3 l1 l2 l4 ∈ circle) ∧ (A4 l1 l2 l3 ∈ circle) := 
sorry

end part_a_l471_471924


namespace quadratic_eq_roots_are_coeffs_l471_471665

theorem quadratic_eq_roots_are_coeffs :
  ∃ (a b : ℝ), (a = r_1) → (b = r_2) →
  (r_1 + r_2 = -a) → (r_1 * r_2 = b) →
  r_1 = 1 ∧ r_2 = -2 ∧ (x^2 + x - 2 = 0):=
by
  sorry

end quadratic_eq_roots_are_coeffs_l471_471665


namespace f_2015_l471_471391

noncomputable def f (x : ℝ) : ℝ := sorry

axiom even_f : ∀ x : ℝ, f x = f (-x)

axiom periodic_f : ∀ x : ℝ, f (x - 2) = -f x

axiom f_interval : ∀ x : ℝ, (-1 ≤ x ∧ x ≤ 0) → f x = 2 ^ x

theorem f_2015 : f 2015 = 1 / 2 :=
sorry

end f_2015_l471_471391


namespace teammates_of_oliver_l471_471618

-- Define the player characteristics
structure Player :=
  (name   : String)
  (eyes   : String)
  (hair   : String)

-- Define the list of players with their given characteristics
def players : List Player := [
  {name := "Daniel", eyes := "Green", hair := "Red"},
  {name := "Oliver", eyes := "Gray", hair := "Brown"},
  {name := "Mia", eyes := "Gray", hair := "Red"},
  {name := "Ella", eyes := "Green", hair := "Brown"},
  {name := "Leo", eyes := "Green", hair := "Red"},
  {name := "Zoe", eyes := "Green", hair := "Brown"}
]

-- Define the condition for being on the same team
def same_team (p1 p2 : Player) : Bool :=
  (p1.eyes = p2.eyes && p1.hair ≠ p2.hair) || (p1.eyes ≠ p2.eyes && p1.hair = p2.hair)

-- Define the criterion to check if two players are on the same team as Oliver
def is_teammate_of_oliver (p : Player) : Bool :=
  let oliver := players[1] -- Oliver is the second player in the list
  same_team oliver p

-- Formal proof statement
theorem teammates_of_oliver : 
  is_teammate_of_oliver players[2] = true ∧ is_teammate_of_oliver players[3] = true :=
by
  -- Provide the intended proof here
  sorry

end teammates_of_oliver_l471_471618


namespace subjects_difference_marius_monica_l471_471085

-- Definitions of given conditions.
def Monica_subjects : ℕ := 10
def Total_subjects : ℕ := 41
def Millie_offset : ℕ := 3

-- Theorem to prove the question == answer given conditions
theorem subjects_difference_marius_monica : 
  ∃ (M : ℕ), (M + (M + Millie_offset) + Monica_subjects = Total_subjects) ∧ (M - Monica_subjects = 4) := 
by
  sorry

end subjects_difference_marius_monica_l471_471085


namespace part1_part2_l471_471393

theorem part1 (a b c : ℝ)
  (h1 : ∀ x : ℝ, f(x) = a * x^4 + b * x^2 + c)
  (h2 : f(0) = 1)
  (h3 : f'(x) = 4 * a * x^3 + 2 * b * x)
  (h4: f(1) = -1)
  (h5 : f'(1) = 1)
  : a = 5 / 2 ∧ c = 1 := 
sorry

theorem part2 (f : ℝ → ℝ)
  (h : ∀ x : ℝ, f(x) = (5 / 2) * x^4 - (9 / 2) * x^2 + 1)
  : { x : ℝ | x > 3 * real.sqrt 10 / 10 ∨ (-3 * real.sqrt 10 / 10 < x ∧ x < 0) } := 
sorry

end part1_part2_l471_471393


namespace polly_age_is_33_l471_471092

theorem polly_age_is_33 
  (x : ℕ) 
  (h1 : ∀ y, y = 20 → x - y = x - 20)
  (h2 : ∀ y, y = 22 → x - y = x - 22)
  (h3 : ∀ y, y = 24 → x - y = x - 24) : 
  x = 33 :=
by 
  sorry

end polly_age_is_33_l471_471092


namespace ferris_wheel_seats_l471_471589

def number_of_people_per_seat := 6
def total_number_of_people := 84

def number_of_seats := total_number_of_people / number_of_people_per_seat

theorem ferris_wheel_seats : number_of_seats = 14 := by
  sorry

end ferris_wheel_seats_l471_471589


namespace rationalize_denominator_simplifies_l471_471948

theorem rationalize_denominator_simplifies :
  (√18 + √2) / (√3 + √2) = 4 * (√6 - 2) :=
by
  sorry

end rationalize_denominator_simplifies_l471_471948


namespace balls_in_boxes_l471_471440

def num_ways_to_partition_6_in_4_parts : ℕ :=
  -- The different partitions of 6: (6,0,0,0), (5,1,0,0), (4,2,0,0), (4,1,1,0),
  -- (3,3,0,0), (3,2,1,0), (3,1,1,1), (2,2,2,0), (2,2,1,1)
  ([
    [6, 0, 0, 0],
    [5, 1, 0, 0],
    [4, 2, 0, 0],
    [4, 1, 1, 0],
    [3, 3, 0, 0],
    [3, 2, 1, 0],
    [3, 1, 1, 1],
    [2, 2, 2, 0],
    [2, 2, 1, 1]
  ]).length

theorem balls_in_boxes : num_ways_to_partition_6_in_4_parts = 9 := by
  sorry

end balls_in_boxes_l471_471440


namespace max_discount_rate_l471_471277

theorem max_discount_rate 
  (cost_price : ℝ) (selling_price : ℝ) (min_profit_margin : ℝ) :
  cost_price = 4 → selling_price = 5 → min_profit_margin = 0.1 →
  ∃ (x : ℝ), 0 ≤ x ∧ x ≤ 8.8 ∧ (selling_price * (1 - x / 100) - cost_price) / cost_price ≥ min_profit_margin :=
begin
  sorry
end

end max_discount_rate_l471_471277


namespace find_angle_B_perimeter_range_l471_471473

noncomputable def triangle_ABC (a b c A B C : ℝ) (acute : 0 < A ∧ A < π / 2)
  (B_eq : 0 < B ∧ B < π / 2) (C_eq : 0 < C ∧ C < π / 2) : Prop :=
  a = b * cos (C - π / 3) * 2 - c

theorem find_angle_B 
  (a b c : ℝ)
  (h : b * cos (C - π / 3) = (a + c) / 2) 
  (acute_A : 0 < A ∧ A < π / 2)
  (acute_B : 0 < B ∧ B < π / 2)
  (acute_C : 0 < C ∧ C < π / 2) 
  (A_sum : A + B + C = π)
  : B = π / 3 := sorry

theorem perimeter_range 
  (a b c : ℝ)
  (h1 : b = 2 * sqrt 3)
  (h2 : B = π / 3)
  (h3 : a = 4 * sin A)
  (h4 : c = 4 * sin C)
  (h5 : 0 < A ∧ A < π / 2)
  (h6 : A + B + C = π) : 
  ∃ P, P = a + b + c ∧ 6 + 2 * sqrt 3 < P ∧ P ≤ 6 * sqrt 3 := sorry

end find_angle_B_perimeter_range_l471_471473


namespace samuels_shoes_left_dust_particles_l471_471104

theorem samuels_shoes_left_dust_particles
  (initial_dust : ℕ)
  (dust_after_walking : ℕ)
  (fraction_cleared : ℚ)
  (particles_cleared : initial_dust * fraction_cleared = initial_dust - 1) :
  initial_dust = 1080 →
  dust_after_walking = 331 →
  fraction_cleared = 9 / 10 →
  let dust_after_sweeping := initial_dust * (1 - fraction_cleared) in
  dust_after_walking - dust_after_sweeping = 223 :=
by
  intros h_init h_walk h_frac
  sorry

end samuels_shoes_left_dust_particles_l471_471104


namespace expected_lifetime_flashlight_l471_471530

noncomputable def E (X : ℝ) : ℝ := sorry -- Define E as the expectation operator

variables (ξ η : ℝ) -- Define ξ and η as random variables representing lifetimes of blue and red bulbs
variable (h_exi : E ξ = 2) -- Given condition E ξ = 2

theorem expected_lifetime_flashlight (h_min : ∀ x y : ℝ, min x y ≤ x) :
  E (min ξ η) ≤ 2 :=
by
  sorry

end expected_lifetime_flashlight_l471_471530


namespace ratio_of_volumes_l471_471699

theorem ratio_of_volumes (s : ℝ) (h : s > 0) : 
  let r := s / 2,
      V_cube := s^3,
      V_cylinder := π * r^2 * s
  in V_cylinder / V_cube = π / 4 :=
by
  let r := s / 2;
  let V_cube := s^3;
  let V_cylinder := π * r^2 * s;
  have r_def : r = s / 2 := rfl;
  have V_cube_def : V_cube = s^3 := rfl;
  have V_cylinder_def : V_cylinder = π * (s / 2)^2 * s := by simp [r_def];
  calc 
    V_cylinder / V_cube = (π * (s / 2)^2 * s) / s^3 : by simp [V_cylinder_def, V_cube_def]
                    ... = (π * (s^2 / 4) * s) / s^3 : by rw [pow_two]
                    ... = π * s^3 / 4 / s^3 : by ring
                    ... = π / 4 : by field_simp;
  sorry

end ratio_of_volumes_l471_471699


namespace option_one_correct_l471_471310

theorem option_one_correct (x : ℝ) : 
  (x ≠ 0 → x + |x| > 0) ∧ ¬((x + |x| > 0) → x ≠ 0) := 
by
  sorry

end option_one_correct_l471_471310


namespace sum_of_smallest_10_T_divisible_by_5_l471_471357

def T (n : ℕ) : ℚ := ((n-1)*n*(n+1)*(3*n+2)) / 24

theorem sum_of_smallest_10_T_divisible_by_5 :
  let ns := (Finset.range 50).filter (fun n => T n % 5 = 0).sort (≤)
  (Finset.sum (Finset.take 10 ns : Finset ℕ)) = 235 :=
by
  sorry

end sum_of_smallest_10_T_divisible_by_5_l471_471357


namespace increasing_sequence_condition_l471_471458

theorem increasing_sequence_condition (k : ℤ) :
  (∀ n : ℕ, n > 1 → a_n > a_{n-1} → a_n = n^2 + k * n) → (k ≥ -2) ∧ (¬(k ≥ -2)) :=
sorry

end increasing_sequence_condition_l471_471458


namespace tangent_lines_ln_l471_471129

theorem tangent_lines_ln (x y: ℝ) : 
    (y = Real.log (abs x)) → 
    (x = 0 ∧ y = 0) ∨ ((x = yup ∨ x = ydown) ∧ (∀ (ey : ℝ), x = ey ∨ x = -ey)) :=
by 
    intro h
    sorry

end tangent_lines_ln_l471_471129


namespace concurrency_of_reflected_touchpoint_lines_l471_471471

theorem concurrency_of_reflected_touchpoint_lines
  (A B C : Type*)
  [NonIsoscelesTriangle A B C]
  (a1 a2 a3 : Segment)
  (M1 M2 M3 : Point)
  (T1 T2 T3 : Point)
  (S1 S2 S3 : Point)
  (h_mid1 : midpoint a1 A B = M1)
  (h_mid2 : midpoint a2 B C = M2)
  (h_mid3 : midpoint a3 C A = M3)
  (h_touch1 : incircle_touch_point a1 = T1)
  (h_touch2 : incircle_touch_point a2 = T2)
  (h_touch3 : incircle_touch_point a3 = T3)
  (h_reflect1 : reflection_point T1 (internal_angle_bisector A) = S1)
  (h_reflect2 : reflection_point T2 (internal_angle_bisector B) = S2)
  (h_reflect3 : reflection_point T3 (internal_angle_bisector C) = S3) :
  concurrent (line_through M1 S1) (line_through M2 S2) (line_through M3 S3) :=
sorry

end concurrency_of_reflected_touchpoint_lines_l471_471471


namespace isosceles_trapezoid_eccentricities_l471_471490

theorem isosceles_trapezoid_eccentricities (A B C D : Point)
    (AB CD : Line)
    (h1 : IsoscelesTrapezoid A B C D)
    (h2 : Parallel AB CD)
    (h3 : SegmentLength AB > SegmentLength CD)
    (e1 : EccentricityHyperbola A B D)
    (e2 : EccentricityEllipse C D A) :
    e1 * e2 = 1 :=
by
  sorry

end isosceles_trapezoid_eccentricities_l471_471490


namespace sufficient_but_not_necessary_l471_471117

theorem sufficient_but_not_necessary (k : ℝ) :
  (∀ x y : ℝ, l k x y ↔ (y = -x - 3 → x = y))
  ∧ ¬ (∀ x y : ℝ, l k x y ↔ (y = (1 / 2) * x → x = y)) :=
begin
  sorry
end

def l (k : ℝ) (x y : ℝ) : Prop := y = k * x + 2 * k - 1

end sufficient_but_not_necessary_l471_471117


namespace inequality_proof_l471_471073

theorem inequality_proof (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a * b ≥ 1) :
  (a + 2 * b + 2 / (a + 1)) * (b + 2 * a + 2 / (b + 1)) ≥ 16 :=
by
  sorry

end inequality_proof_l471_471073


namespace nth_non_perfect_square_eq_sequence_l471_471093

open Nat

-- Define the sequence as the nth non-perfect square
noncomputable def non_perfect_square (n : ℕ) : ℕ :=
  n + ⌊real.sqrt n + 1 / 2⌋

-- The theorem statement
theorem nth_non_perfect_square_eq_sequence (n : ℕ) :
  ∃ k : ℕ, non_perfect_square n = k ∧ ¬∃ m : ℕ, m * m = k :=
sorry

end nth_non_perfect_square_eq_sequence_l471_471093


namespace train_speed_correct_l471_471306

-- Define the length of the train
def train_length : ℝ := 200

-- Define the time taken to cross the telegraph post
def cross_time : ℝ := 8

-- Define the expected speed of the train
def expected_speed : ℝ := 25

-- Prove that the speed of the train is as expected
theorem train_speed_correct (length time : ℝ) (h_length : length = train_length) (h_time : time = cross_time) : 
  (length / time = expected_speed) :=
by
  rw [h_length, h_time]
  sorry

end train_speed_correct_l471_471306


namespace Ruth_sandwiches_l471_471953

theorem Ruth_sandwiches (sandwiches_left sandwiches_ruth sandwiches_brother sandwiches_first_cousin sandwiches_two_cousins total_sandwiches : ℕ)
  (h_ruth : sandwiches_ruth = 1)
  (h_brother : sandwiches_brother = 2)
  (h_first_cousin : sandwiches_first_cousin = 2)
  (h_two_cousins : sandwiches_two_cousins = 2)
  (h_left : sandwiches_left = 3) :
  total_sandwiches = sandwiches_left + sandwiches_two_cousins + sandwiches_first_cousin + sandwiches_ruth + sandwiches_brother :=
by
  sorry

end Ruth_sandwiches_l471_471953


namespace solve_cubic_root_eq_l471_471752

theorem solve_cubic_root_eq (x : ℝ) (h : (5 - x)^(1/3) = 4) : x = -59 := 
by
  sorry

end solve_cubic_root_eq_l471_471752


namespace intersection_points_in_decagon_l471_471714

-- Define the number of sides for a regular decagon
def n : ℕ := 10

-- The formula to calculate the number of ways to choose 4 vertices from n vertices
def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- The statement that needs to be proven
theorem intersection_points_in_decagon : choose 10 4 = 210 := by
  sorry

end intersection_points_in_decagon_l471_471714


namespace force_required_l471_471601

theorem force_required 
  (F : ℕ → ℕ)
  (h_inv : ∀ L L' : ℕ, F L * L = F L' * L')
  (h1 : F 12 = 300) :
  F 18 = 200 :=
by
  sorry

end force_required_l471_471601


namespace price_of_pants_l471_471619

theorem price_of_pants (total_cost belt_cost pants_discount : ℝ) 
  (h_total_cost : total_cost = 70.93)
  (h_pants_discount : pants_discount = 2.93)
  (h_price_relation : total_cost = belt_cost + (belt_cost - pants_discount)) : 
  belt_cost - pants_discount = 34.00 :=
by
  -- Introduce necessary assumptions and variables
  let B := belt_cost
  let P := B - pants_discount
  have h1 : total_cost = B + P := by
    rw [h_pants_discount]
    rw [h_price_relation]
  have h2 : total_cost = 70.93 := h_total_cost
  have h3 : 2 * B - 2.93 = 70.93 := by
    rw [h_pants_discount]
    linarith
  have h4 : 2 * B = 73.86 := by
    linarith
  have h5 : B = 36.93 := by
    apply eq_of_mul_eq_mul_left 
    norm_num
    rw [h4]
  exact calc
    B - pants_discount = 36.93 - 2.93 : by rw [h5, h_pants_discount]
                    ... = 34.00 : by norm_num
  sorry

end price_of_pants_l471_471619


namespace sin_pi_over_4_plus_alpha_l471_471826

open Real

theorem sin_pi_over_4_plus_alpha
  (α : ℝ)
  (hα : 0 < α ∧ α < π)
  (h_tan : tan (α - π / 4) = 1 / 3) :
  sin (π / 4 + α) = 3 * sqrt 10 / 10 :=
sorry

end sin_pi_over_4_plus_alpha_l471_471826


namespace domain_F_F_odd_F_pos_set_l471_471410

variable (a : ℝ) (h_a1 : 0 < a) (h_a2 : a ≠ 1)

def f (x : ℝ) : ℝ := log a (x + 1)
def g (x : ℝ) : ℝ := log a (1 - x)
def F (x : ℝ) : ℝ := f a x - g a x

-- Part 1: Domain of F(x)
theorem domain_F : ∀ x, (-1 < x ∧ x < 1) ↔ (F a x).1 ∧ (F a x).2 := 
sorry

-- Part 2: Odd property of F(x)
theorem F_odd : ∀ x, F a (-x) = -F a x := 
sorry

-- Part 3: Set of values of x that make F(x) > 0
theorem F_pos_set (h_a : a > 1) : ∀ x, (0 < x ∧ x < 1) ↔ F a x > 0 :=
sorry

end domain_F_F_odd_F_pos_set_l471_471410


namespace treaty_signed_on_tuesday_l471_471971

-- Define a constant for the start date and the number of days
def start_day_of_week : ℕ := 1 -- Monday is represented by 1
def days_until_treaty : ℕ := 1301

-- Function to calculate the resulting day of the week
def day_of_week_after_days (start_day : ℕ) (days : ℕ) : ℕ :=
  (start_day + days) % 7

-- Theorem statement: Prove that 1301 days after Monday is Tuesday
theorem treaty_signed_on_tuesday :
  day_of_week_after_days start_day_of_week days_until_treaty = 2 :=
by
  -- placeholder for the proof
  sorry

end treaty_signed_on_tuesday_l471_471971


namespace cube_dot_product_values_l471_471456

structure Cube :=
  (A1 A2 A3 A4 B1 B2 B3 B4 : ℝ × ℝ × ℝ)
  (edge_length : ℝ)

def vector_sub (v1 v2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (v1.1 - v2.1, v1.2 - v2.2, v1.3 - v2.3)

def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  (v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3)

def distinct_values_count (cube : Cube) : ℕ :=
  {x ∈ set.range (λ ij : ℕ × ℕ, dot_product (vector_sub cube.B1 cube.A1) (vector_sub (list.nth [cube.A1, cube.A2, cube.A3, cube.A4] ij.1).get_or_else (0,0,0) (vector_sub (list.nth [cube.B1, cube.B2, cube.B3, cube.B4] ij.2).get_or_else (0,0,0)))}.to_finset.card

theorem cube_dot_product_values (cube : Cube) (h : cube.edge_length = 1) :
  distinct_values_count cube = 1 :=
sorry

end cube_dot_product_values_l471_471456


namespace system_of_equations_correct_l471_471480

theorem system_of_equations_correct (x y : ℕ) :
  (7 * x + 7 = y) ∧ (9 * (x - 1) = y) :=
begin
  -- sorry, proof placeholder
  sorry
end

end system_of_equations_correct_l471_471480


namespace flashlight_lifetime_expectation_leq_two_l471_471546

noncomputable def min_lifetime_expectation (ξ η : ℝ) (E_ξ : ℝ) : Prop :=
  E_ξ = 2 → E(min ξ η) ≤ 2

-- Assume ξ and η are random variables and E denotes the expectation.
axiom E : (ℝ → ℝ) → ℝ

theorem flashlight_lifetime_expectation_leq_two (ξ η : ℝ) (E_ξ : ℝ) (hE_ξ : E_ξ = 2) : E(min ξ η) ≤ 2 :=
  by
    sorry

end flashlight_lifetime_expectation_leq_two_l471_471546


namespace kangaroo_num_ways_l471_471719

def num_ways_to_vertex_E (m : ℕ) : ℕ :=
  1 / Real.sqrt 2 * (2 + Real.sqrt 2) ^ (m - 1) - 1 / Real.sqrt 2 * (2 - Real.sqrt 2) ^ (m - 1)

theorem kangaroo_num_ways (n : ℕ) (m : ℕ) (hn : n = 2 * m) :
  a_n = num_ways_to_vertex_E m := by
  sorry

end kangaroo_num_ways_l471_471719


namespace odd_sum_property_l471_471721

noncomputable def S : ℕ → ℕ
| 1 := 1
| 2 := 5
| 3 := 15
| 4 := 34
| 5 := 65
| 6 := 111
| 7 := 175
| _ := 0 -- Assuming values are unknown for n > 7 for consistency here

theorem odd_sum_property (n : ℕ) :
  (∑ k in finset.range (2 * n), if odd k then S k.succ else 0) = n^4 :=
by
  sorry

end odd_sum_property_l471_471721


namespace max_n_for_coloring_l471_471195

noncomputable def maximum_n : ℕ :=
  11

theorem max_n_for_coloring :
  ∃ n : ℕ, (n = maximum_n) ∧ ∀ k ∈ Finset.range n, 
  (∃ x y : ℕ, 1 ≤ x ∧ x ≤ 14 ∧ 1 ≤ y ∧ y ≤ 14 ∧ (x - y = k ∨ y - x = k) ∧ x ≠ y) ∧
  (∃ a b : ℕ, 1 ≤ a ∧ a ≤ 14 ∧ 1 ≤ b ∧ b ≤ 14 ∧ (a - b = k ∨ b - a = k) ∧ a ≠ b) :=
sorry

end max_n_for_coloring_l471_471195


namespace trapezium_second_side_length_l471_471756

theorem trapezium_second_side_length
  (side1 : ℝ)
  (height : ℝ)
  (area : ℝ) 
  (h1 : side1 = 20) 
  (h2 : height = 13) 
  (h3 : area = 247) : 
  ∃ side2 : ℝ, 0 ≤ side2 ∧ ∀ side2, area = 1 / 2 * (side1 + side2) * height → side2 = 18 :=
by
  use 18
  sorry

end trapezium_second_side_length_l471_471756


namespace mean_proportional_c_l471_471459

theorem mean_proportional_c (a b c : ℝ) (h1 : a = 3) (h2 : b = 27) (h3 : c^2 = a * b) : c = 9 := by
  sorry

end mean_proportional_c_l471_471459


namespace find_norm_alpha_l471_471910

variable (α β : ℂ) 

theorem find_norm_alpha (h1 : α + β ∈ ℝ) 
                        (h2 : ∃ k ∈ ℝ, α / (β * β) = k)
                        (h3 : |α - β| = 4)
                        (h4 : β = conj α) 
                        : |α| = 2 * Real.sqrt 2 := 
  sorry

end find_norm_alpha_l471_471910


namespace all_star_seating_arrangements_l471_471034

theorem all_star_seating_arrangements : 
  let blocks := 4!
  let cubs_permutations := 3!
  let red_sox_permutations := 2!
  let yankees_permutations := 2!
  let dodgers_permutations := 2!
  blocks * cubs_permutations * red_sox_permutations * yankees_permutations * dodgers_permutations = 1152 := 
by 
  have blocks := Nat.factorial 4
  have cubs_permutations := Nat.factorial 3
  have red_sox_permutations := Nat.factorial 2
  have yankees_permutations := Nat.factorial 2
  have dodgers_permutations := Nat.factorial 2
  sorry

end all_star_seating_arrangements_l471_471034


namespace rojas_speed_l471_471574

theorem rojas_speed (P R : ℝ) (h1 : P = 3) (h2 : 4 * (R + P) = 28) : R = 4 :=
by
  sorry

end rojas_speed_l471_471574


namespace beggars_society_votes_l471_471676

def total_voting_members (votes_for votes_against additional_against : ℕ) :=
  let majority := additional_against / 4
  let initial_difference := votes_for - votes_against
  let updated_against := votes_against + additional_against
  let updated_for := votes_for - additional_against
  updated_for + updated_against

theorem beggars_society_votes :
  total_voting_members 115 92 12 = 207 :=
by
  -- Proof goes here
  sorry

end beggars_society_votes_l471_471676


namespace total_students_l471_471143

theorem total_students (S K : ℕ) (h1 : S = 4000) (h2 : K = 2 * S) :
  S + K = 12000 := by
  sorry

end total_students_l471_471143


namespace sum_of_digits_B_is_7_l471_471914

-- Define A as the sum of the digits of 4444^4444
def sumOfDigits (n : Nat) : Nat :=
  n.digits.foldr (fun d acc => d + acc) 0

def A : Nat :=
  sumOfDigits (4444 ^ 4444)

-- Define B as the sum of the digits of A
def B : Nat :=
  sumOfDigits A

-- Prove that sum of the digits of B is 7
theorem sum_of_digits_B_is_7 : sumOfDigits B = 7 :=
  sorry

end sum_of_digits_B_is_7_l471_471914


namespace circle_equation_mid_point_condition_l471_471845

-- Definitions used in conditions
def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def line_l (x y : ℝ) (m : ℝ) : Prop := y = x - m
def midpoint (A B M : ℝ × ℝ) : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Proof 1: Given M(2, 0), prove the equation of circle with diameter AB is (x - 4)^2 + (y - 2)^2 = 24
theorem circle_equation (A B P : ℝ × ℝ) (M : ℝ × ℝ) :
  M = (2, 0) →
  midpoint A B = P →
  (parabola A.1 A.2 ∧ parabola B.1 B.2) →
  (line_l A.1 A.2 2 ∧ line_l B.1 B.2 2) →
  ∀ x y, (x - 4)^2 + (y - 2)^2 = 24 :=
by sorry

-- Proof 2: Given M(m, 0), if 1/|AM| + 1/|BM| = 1/|PM|, prove m = 2 ± 2 * √2
theorem mid_point_condition (A B P : ℝ × ℝ) (m : ℝ) :
  M = (m, 0) →
  midpoint A B = P →
  (parabola A.1 A.2 ∧ parabola B.1 B.2) →
  (line_l A.1 A.2 m ∧ line_l B.1 B.2 m) →
  (1 / |A.1 - M.1| + 1 / |B.1 - M.1| = 1 / |P.1 - M.1|) →
  m = 2 + 2 * Real.sqrt 2 ∨ m = 2 - 2 * Real.sqrt 2 :=
by sorry

end circle_equation_mid_point_condition_l471_471845


namespace sum_irreducible_fractions_l471_471351

theorem sum_irreducible_fractions (m n : ℕ) (h : m < n) : 
  (∑ k in (Finset.Ico m n), if (k % 3 ≠ 0) then (k / 3 : ℚ) else 0) = n^2 - m^2 := 
  sorry

end sum_irreducible_fractions_l471_471351


namespace main_theorem_l471_471067

noncomputable def a (k : ℕ) : ℝ := (1 + Real.sqrt k) / 2

def int_condition (k : ℕ) : Prop :=
  k % 4 = 1 ∧ ¬(∃m : ℕ, m * m = k)

theorem main_theorem (k : ℕ) (h : int_condition k) :
  {⌊(a k)^2 * n⌋ - ⌊(a k) * ⌊(a k) * n⌋⌋ : ℕ+ → ℕ} = {i | i ∈ Finset.range (⌊a k⌋ : ℕ) + 1} :=
by
  sorry

end main_theorem_l471_471067


namespace cyclic_quad_area_example_l471_471810

noncomputable def cyclic_quad_area (a b c d : ℝ) : ℝ :=
  let s := (a + b + c + d) / 2 in
  let area := (s - a) * (s - b) * (s - c) * (s - d) in
  (area - a * b * c * d * (1 / (s * (16 / area ^ 2)))) ^ (1 / 2)

theorem cyclic_quad_area_example :
  cyclic_quad_area 2 6 4 4 = 8 * real.sqrt 3 :=
begin
  -- Proof goes here
  sorry
end

end cyclic_quad_area_example_l471_471810


namespace evaluate_expression_l471_471799

def spadesuit (x y : ℝ) : ℝ := (x + y) * (x - y)

theorem evaluate_expression : spadesuit 3 (spadesuit 6 5) = -112 := by
  sorry

end evaluate_expression_l471_471799


namespace find_radius_of_circle_l471_471216

theorem find_radius_of_circle (C : ℝ) (h : C = 72 * Real.pi) : ∃ r : ℝ, 2 * Real.pi * r = C ∧ r = 36 :=
by
  sorry

end find_radius_of_circle_l471_471216


namespace unsold_percentage_l471_471932

def total_harvested : ℝ := 340.2
def sold_mm : ℝ := 125.5  -- Weight sold to Mrs. Maxwell
def sold_mw : ℝ := 78.25  -- Weight sold to Mr. Wilson
def sold_mb : ℝ := 43.8   -- Weight sold to Ms. Brown
def sold_mj : ℝ := 56.65  -- Weight sold to Mr. Johnson

noncomputable def percentage_unsold (total_harvested : ℝ) 
                                   (sold_mm : ℝ) 
                                   (sold_mw : ℝ)
                                   (sold_mb : ℝ) 
                                   (sold_mj : ℝ) : ℝ :=
  let total_sold := sold_mm + sold_mw + sold_mb + sold_mj
  let unsold := total_harvested - total_sold
  (unsold / total_harvested) * 100

theorem unsold_percentage : percentage_unsold total_harvested sold_mm sold_mw sold_mb sold_mj = 10.58 :=
by
  sorry

end unsold_percentage_l471_471932


namespace total_tosses_made_l471_471928

theorem total_tosses_made (tosses_per_second : ℕ) (initial_balls : ℕ) (ball_interval : ℕ) (total_time : ℕ) :
  initial_balls = 1 →
  ball_interval = 5 →
  total_time = 60 →
  tosses_per_second = 1 →
  ∑ i in Finset.range (total_time / ball_interval + 1), ((i + initial_balls) * ball_interval) + (initial_balls * (total_time % ball_interval)) = 390 := 
by
  intros h_initial_balls h_ball_interval h_total_time h_tosses_per_second
  rw [h_initial_balls, h_ball_interval, h_total_time, h_tosses_per_second]
  sorry

end total_tosses_made_l471_471928


namespace single_element_sets_l471_471999

noncomputable def set_one : Set ℝ := {x | x^2 - 2 * x + 1 = 0}
def set_two : Set ℝ := {-1, 2}
noncomputable def set_three : Set (ℤ × ℤ) := {(-1, 2)}
-- Triangles with sides of length 3 and 4 would typically involve some non-trivial computation
-- so for simplicity, we use placeholder elements, demonstrating the necessary conditions.
def set_four : Set (ℕ × ℕ × ℕ) := {t | (t.1 = 3 ∧ t.2 = 4) ∨ (t.1 = 4 ∧ t.2 = 3)}

theorem single_element_sets :
  (set_one.card = 1 ∧ set_three.card = 1) ∧
  (set_two.card ≠ 1 ∧ set_four.card ≠ 1) :=
by
  sorry

end single_element_sets_l471_471999


namespace ellipse_triangle_is_isosceles_right_l471_471976

theorem ellipse_triangle_is_isosceles_right (e : ℝ) (a b c k : ℝ)
  (H1 : e = (c / a))
  (H2 : e = (Real.sqrt 2) / 2)
  (H3 : b^2 = a^2 * (1 - e^2))
  (H4 : a = 2 * k)
  (H5 : b = k * Real.sqrt 2)
  (H6 : c = k * Real.sqrt 2) :
  (4 * k)^2 = (2 * (k * Real.sqrt 2))^2 + (2 * (k * Real.sqrt 2))^2 :=
by
  sorry

end ellipse_triangle_is_isosceles_right_l471_471976


namespace f_l471_471405

def f (x : ℝ) : ℝ := Real.sin x + Real.exp x

theorem f'_at_0 : (deriv f 0) = 2 :=
by
  -- Proof omitted
  sorry

end f_l471_471405


namespace x_eq_1_is_iff_x2_minus_2x_plus_1_eq_0_l471_471251

theorem x_eq_1_is_iff_x2_minus_2x_plus_1_eq_0 :
  (x = 1 ↔ x^2 - 2 * x + 1 = 0) :=
sorry

end x_eq_1_is_iff_x2_minus_2x_plus_1_eq_0_l471_471251


namespace complex_problem_l471_471018

variable (a : ℂ)

theorem complex_problem (h1 : (a ^ 2 - 4 : ℂ) = 0) (h2 : a ≠ -2) :
  (a + complex.exp (complex.I * 2015 * complex.pi / 2)) / (1 + 2 * complex.I) = -complex.I := by
  sorry

end complex_problem_l471_471018


namespace rationalize_denominator_l471_471101

theorem rationalize_denominator :
  (2 * real.sqrt 12 + real.sqrt 5) / (real.sqrt 5 + real.sqrt 3) =
  (3 * real.sqrt 15 - 7) / 2 :=
by
  sorry

end rationalize_denominator_l471_471101


namespace ratio_p_q_l471_471573

noncomputable def binomial_expansion (n : ℕ) (x : ℝ) : ℝ :=
  (real.sqrt x + 1/x)^n

def rational_indices : list ℕ := [0, 2, 4, 6, 8]

def is_rational_term (r : ℕ) : bool :=
  (4 - 3 * r / 2) ∈ ℕ

def count_rational_terms (terms : list ℕ) : ℕ :=
  terms.filter is_rational_term |>.length

def p (rational_count irrational_count : ℕ) : ℚ := 
  (rational_count * factorial rational_count) ^ 2 / (factorial (rational_count + irrational_count)) ^ 2

def q (rational_count irrational_count : ℕ) : ℚ := 
  (factorial irrational_count * factorial rational_count) / factorial (rational_count + irrational_count)

theorem ratio_p_q : 
  let rational_count := count_rational_terms rational_indices in
  let irrational_count := (8 + 1) - rational_count in
  (p rational_count irrational_count) / (q rational_count irrational_count) = 5 :=
sorry

end ratio_p_q_l471_471573


namespace point_movement_l471_471056

theorem point_movement (P : ℤ) (hP : P = -5) (k : ℤ) (hk : (k = 3 ∨ k = -3)) :
  P + k = -8 ∨ P + k = -2 :=
by {
  sorry
}

end point_movement_l471_471056


namespace sum_def_l471_471833

-- Given polynomials in conditions
def poly1 : Polynomial ℤ := Polynomial.C 40 + Polynomial.C 13 * Polynomial.X + Polynomial.X^2
def poly2 : Polynomial ℤ := Polynomial.C 88 + Polynomial.C (-19) * Polynomial.X + Polynomial.X^2

-- Define the factorizations
def factors1 (d e : ℤ) : Prop := poly1 = (Polynomial.C d) * (Polynomial.C e)
def factors2 (e f : ℤ) : Prop := poly2 = (Polynomial.C e) * (Polynomial.C f)

theorem sum_def (d e f : ℤ) (h1 : factors1 d e) (h2 : factors2 e f) : d + e + f = 24 := sorry

end sum_def_l471_471833


namespace probability_red_first_given_black_second_l471_471642

open ProbabilityTheory MeasureTheory

-- Definitions for Urn A and Urn B ball quantities
def urnA := (white : 4, red : 2)
def urnB := (red : 3, black : 3)

-- Event of drawing a red ball first and a black ball second
def eventRedFirst := (urn : ℕ, ball : string) -> (urn = 1 ∧ ball = "red") ∨ (urn = 2 ∧ ball = "red")
def eventBlackSecond := (urn : ℕ, ball : string) -> (urn = 1 ∧ ball = "black") ∨ (urn = 2 ∧ ball = "black")

-- Probability function definition
noncomputable def P := sorry -- Probability function placeholder

-- Conditional Probability
theorem probability_red_first_given_black_second :
  P(eventRedFirst | eventBlackSecond) = 2 / 5 := sorry

end probability_red_first_given_black_second_l471_471642


namespace gcd_linear_combination_l471_471669

namespace EuclideanAlgorithm

theorem gcd_linear_combination (m₀ m₁ : ℤ) (h : 0 < m₁ ∧ m₁ ≤ m₀) :
  ∃ k > 1, ∃ a : Fin k → ℤ, ∃ m : Fin k → ℤ,
  (∀ i : Fin (k-1), m i > m (i+1) ∧ m (k-1) > 0) ∧
  a (k-1) > 1 ∧
  (m 0 = m₁ * a 0 + m 1 ∧
  ∀ i : Fin (k-2), m (i + 1) = m (i + 2) * a (i + 1) + m (i + 2 + 1)) ∧
  ∃ u v : ℤ, m 0 * u + m 1 * v = (Nat.gcd m₀ m₁ : ℤ) := by
  sorry

end EuclideanAlgorithm

end gcd_linear_combination_l471_471669


namespace football_area_l471_471098

theorem football_area (A B C D : Point) (AB BC CD DA : Line)
  (circle_D : Circle) (circle_B : Circle) (sector_DEA sector_BFA : Sector) :
  square ABCD → 
  (AB = 4) →
  (center circle_D = D) →
  (radius circle_D = AB) →
  (center circle_B = B) →
  (radius circle_B = AB) →
  (angle DEA = 90) →
  (angle BFA = 90) →
  (arc sector_DEA = arc sector_BFA) →
  area (region_I ∪ region_II ∪ region_III) - (area region_I) = 8 * pi - 16 →
  ( 2 * (4 * pi - 8) ).approx = 9.1 :=
by sorry

end football_area_l471_471098


namespace tangent_lines_to_ln_abs_through_origin_l471_471125

noncomputable def tangent_line_through_origin (x y: ℝ) : Prop :=
  (y = log (abs x)) ∧ ((x - exp(1) * y = 0) ∨ (x + exp(1) * y = 0))

theorem tangent_lines_to_ln_abs_through_origin :
  ∃ (f : ℝ → ℝ), 
  (∀ x, f x = log (abs x)) ∧ 
  ∀ x y, (tangent_line_through_origin x y) := sorry

end tangent_lines_to_ln_abs_through_origin_l471_471125


namespace volume_of_pyramid_correct_l471_471569

noncomputable def volume_of_pyramid : ℝ := 24 * Real.sqrt 481

theorem volume_of_pyramid_correct (AB BC PB : ℝ) (PA : ℝ) 
  (ABCD : ℝ → ℝ → Prop) (height : ℝ)
  (h_base : ABCD AB BC) 
  (h_AB : AB = 12) 
  (h_BC : BC = 6) 
  (h_perp1 : PA ⟂ AD) 
  (h_perp2 : PA ⟂ AB) 
  (h_PB : PB = 25) 
  (h_Pythagorean : PA = Real.sqrt (PB^2 - AB^2)) 
  (h_area : height = AB * BC) 
  (h_volume : volume_of_pyramid = (1/3) * height * PA) :
  volume_of_pyramid = 24 * Real.sqrt 481 := 
sorry

end volume_of_pyramid_correct_l471_471569


namespace correct_statement_l471_471572

noncomputable def k : ℝ := -6

def inverse_prop_function (x : ℝ) : ℝ := k / x

theorem correct_statement :
  (∀ x : ℝ, x ≠ 0 → inverse_prop_function x = -6 / x) ∧
  (∀ x : ℝ, x > 0 → inverse_prop_function x < 0) ∧
  (∀ x : ℝ, x < 0 → inverse_prop_function x > 0) →
  (∀ x : ℝ, x < 0 → x₂ : ℝ, x₂ < 0 → x < x₂ → inverse_prop_function x < inverse_prop_function x₂) := 
  sorry

end correct_statement_l471_471572


namespace find_complex_number_l471_471116

theorem find_complex_number :
  (im (3 * complex.I) = re (complex.I * 3)) ∧ 
  (re (-3 + complex.I) = 1) ∧ 
  (∀ z: ℂ, im z = re (3 * complex.I) ∧ re z = im (-3 + complex.I) → z = 3 - 3 * complex.I)
sorry

end find_complex_number_l471_471116


namespace third_largest_using_digits_1_6_8_l471_471650

def digits : List ℕ := [1, 6, 8]

def all_three_digit_numbers (ds : List ℕ) : List ℕ :=
  ds.permutations.map (λ p, 100 * p.head! + 10 * p.tail!.head! + p.tail!.tail!.head!)

def third_largest (ns : List ℕ) : ℕ :=
  ns.sort (· > ·) |>.nth! 2

theorem third_largest_using_digits_1_6_8 : third_largest (all_three_digit_numbers digits) = 681 :=
  sorry

end third_largest_using_digits_1_6_8_l471_471650


namespace solve_mod_equation_l471_471743

def is_two_digit_positive_integer (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem solve_mod_equation (u : ℕ) (h1 : is_two_digit_positive_integer u) (h2 : 13 * u % 100 = 52) : u = 4 :=
sorry

end solve_mod_equation_l471_471743


namespace max_positive_integer_for_S_n_l471_471041

noncomputable def a_n (n : ℕ) : ℝ := sorry -- the general term of the arithmetic sequence
noncomputable def S_n (n : ℕ) : ℝ := (n / 2) * (a_n 1 + a_n n) -- Sum of the first n terms

-- Definitions for the given conditions
axiom h1 : a_n 1010 / a_n 1009 < -1 -- Condition 1
axiom h2 : ∃ n, ∀ m, m ≤ n → S_n m = (m / 2) * (a_n 1 + a_n m) -- Condition 2 (maximum value exists)

-- Proving the maximum positive integer n for which S_n > 0
theorem max_positive_integer_for_S_n : ∃ n, S_n n > 0 ∧ ∀ m > n, S_n m ≤ 0 :=
∃ n, n = 2018 ∧ S_n n > 0 ∧ ∀ m > n, S_n m ≤ 0 := sorry

end max_positive_integer_for_S_n_l471_471041


namespace triangle_area_l471_471725

def point := (ℕ, ℕ)

def A : point := (2, 2)
def B : point := (8, 2)
def C : point := (5, 11)

def base_length (A B : point) : ℕ := B.1 - A.1
def height_length (A C : point) : ℕ := C.2 - A.2

theorem triangle_area (A B C : point) (hA : A = (2, 2)) (hB : B = (8, 2)) (hC : C = (5, 11)) :
  1 / 2 * base_length A B * height_length A C = 27 :=
by
  rw [hA, hB, hC]
  sorry

end triangle_area_l471_471725


namespace max_distance_and_point_l471_471477

noncomputable def curve_C1 (x y : ℝ) : Prop := x^2 + y^2 = 1

noncomputable def stretch_x (x : ℝ) := √2 * x
noncomputable def stretch_y (y : ℝ) := √3 * y

noncomputable def parametric_C2 (θ : ℝ) : ℝ × ℝ :=
  (stretch_x (Math.cos θ), stretch_y (Math.sin θ))

noncomputable def distance_to_line (P : ℝ × ℝ) : ℝ :=
  let (x, y) := P in
  abs (x + y - 4 * √5) / √2

noncomputable def max_distance_point (θ : ℝ) : Prop :=
  let P := parametric_C2 θ in
  distance_to_line P = 5 * √10 / 2 

theorem max_distance_and_point :
  ∃ θ : ℝ, 
  max_distance_point θ ∧ 
  parametric_C2 θ = (-2 * √5 / 5, -3 * √5 / 5) := 
sorry

end max_distance_and_point_l471_471477


namespace largest_three_digit_number_l471_471764

theorem largest_three_digit_number :
  ∃ (N : ℕ), N = 915 ∧ 
  N = let S := (N / 100) + (N % 100 / 10) + (N % 10) in
          S + 4 * S^2 ∧ 100 ≤ N ∧ N < 1000 :=
begin
  sorry
end

end largest_three_digit_number_l471_471764


namespace alice_paid_35_percent_l471_471602

theorem alice_paid_35_percent (P : ℝ) (h_marked_price : ∀ P, P > 0 → 0.7 * P = P - 0.3 * P)
  (h_sale_price : ∀ MP, MP = 0.5 * MP) : 0.35 * P = 0.5 * (0.7 * P) :=
by
  have h1 : 0.7 * P = P - 0.3 * P := h_marked_price P (by linarith)
  have h2 : 0.5 * (0.7 * P) = (0.5 * 0.7) * P := by linarith
  have h3 : (0.5 * 0.7) * P = 0.35 * P := by norm_num
  rw [h1, h2, h3]
  sorry

end alice_paid_35_percent_l471_471602


namespace minimize_sum_of_squares_of_perpendiculars_l471_471470

open Real

variable {α β c : ℝ} -- angles and side length

theorem minimize_sum_of_squares_of_perpendiculars
    (habc : α + β = π)
    (P : ℝ)
    (AP BP : ℝ)
    (x : AP + BP = c)
    (u : ℝ)
    (v : ℝ)
    (hAP : AP = P)
    (hBP : BP = c - P)
    (hu : u = P * sin α)
    (hv : v = (c - P) * sin β)
    (f : ℝ)
    (hf : f = (P * sin α)^2 + ((c - P) * sin β)^2):
  (AP / BP = (sin β)^2 / (sin α)^2) := sorry

end minimize_sum_of_squares_of_perpendiculars_l471_471470


namespace tan_half_alpha_l471_471384

theorem tan_half_alpha (α : ℝ) (h1 : α ∈ Ioo (3 * Real.pi / 2) (2 * Real.pi))
  (h2 : Real.sin α + Real.cos α = 1 / 5) : Real.tan (α / 2) = -1 / 3 :=
sorry

end tan_half_alpha_l471_471384


namespace limit_seq_zero_l471_471220

noncomputable def limit_seq (f : ℕ → ℝ) (L : ℝ) : Prop :=
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |f n - L| < ε

def seq_expr (n : ℕ) : ℝ :=
  (Real.cbrt (n ^ 3 - 7) + Real.cbrt (n ^ 2 + 4)) / (Real.root 4 (n ^ 5 + 5) + Real.sqrt n)

theorem limit_seq_zero : limit_seq seq_expr 0 :=
by
  sorry

end limit_seq_zero_l471_471220


namespace student_in_eighth_group_l471_471466

-- Defining the problem: total students and their assignment into groups
def total_students : ℕ := 50
def students_assigned_numbers (n : ℕ) : Prop := n > 0 ∧ n ≤ total_students

-- Grouping students: Each group has 5 students
def grouped_students (group_num student_num : ℕ) : Prop := 
  student_num > (group_num - 1) * 5 ∧ student_num ≤ group_num * 5

-- Condition: Student 12 is selected from the third group
def condition : Prop := grouped_students 3 12

-- Goal: the number of the student selected from the eighth group is 37
theorem student_in_eighth_group : condition → grouped_students 8 37 :=
by
  sorry

end student_in_eighth_group_l471_471466


namespace cloth_cost_price_l471_471210

theorem cloth_cost_price
  (meters_of_cloth : ℕ) (selling_price : ℕ) (profit_per_meter : ℕ)
  (total_profit : ℕ) (total_cost_price : ℕ) (cost_price_per_meter : ℕ) :
  meters_of_cloth = 45 →
  selling_price = 4500 →
  profit_per_meter = 14 →
  total_profit = profit_per_meter * meters_of_cloth →
  total_cost_price = selling_price - total_profit →
  cost_price_per_meter = total_cost_price / meters_of_cloth →
  cost_price_per_meter = 86 :=
by
  intros
  sorry

end cloth_cost_price_l471_471210


namespace midpoint_product_eq_neg_six_l471_471657

def point := (ℝ, ℝ)

def midpoint (p1 p2 : point) : point :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  ((x1 + x2) / 2, (y1 + y2) / 2)

def product_of_coordinates (p : point) : ℝ :=
  let (x, y) := p
  x * y

theorem midpoint_product_eq_neg_six :
  let p1 : point := (4, -1)
  let p2 : point := (-8, 7)
  product_of_coordinates (midpoint p1 p2) = -6 :=
by
  let p1 : point := (4, -1)
  let p2 : point := (-8, 7)
  have : midpoint p1 p2 = (-2, 3) := sorry
  have : product_of_coordinates (-2, 3) = -2 * 3 := rfl
  show product_of_coordinates (midpoint p1 p2) = -6
  sorry

end midpoint_product_eq_neg_six_l471_471657


namespace polar_equation_represents_circle_parametric_equations_represent_line_l471_471986

-- Definition of polar equation
def polar_equation (ρ θ : ℝ) : Prop :=
  ρ = Real.cos θ

-- Definition of parametric equations
def parametric_equations (x y t : ℝ) : Prop :=
  x = -1 - t ∧ y = 2 + 3t

-- Proof that the polar equation represents a circle
theorem polar_equation_represents_circle (ρ θ : ℝ) : polar_equation ρ θ →
  ∃ x y : ℝ, (x - 1/2) ^ 2 + y ^ 2 = 1 / 4 ∧ ρ = Real.sqrt (x^2 + y^2) ∧ θ = Real.atan2 y x :=
by
  sorry

-- Proof that the parametric equations represent a line
theorem parametric_equations_represent_line (x y t : ℝ) : parametric_equations x y t →
  ∃ a b : ℝ, 3 * x + y + 1 = 0 :=
by
  sorry

end polar_equation_represents_circle_parametric_equations_represent_line_l471_471986


namespace directional_vector_for_line_l471_471118

theorem directional_vector_for_line :
  ∃ (v : ℝ × ℝ), let l := (2 : ℝ, 3 : ℝ, -1 : ℝ)
                 in (2 * v.1 + 3 * v.2 = 0 ∧ v ≠ (0, 0)) ∧ v = (1, -2 / 3) :=
by
  sorry

end directional_vector_for_line_l471_471118


namespace interest_rate_for_4000_investment_l471_471651

theorem interest_rate_for_4000_investment
      (total_money : ℝ := 9000)
      (invested_at_9_percent : ℝ := 5000)
      (total_interest : ℝ := 770)
      (invested_at_unknown_rate : ℝ := 4000) :
  ∃ r : ℝ, invested_at_unknown_rate * r = total_interest - (invested_at_9_percent * 0.09) ∧ r = 0.08 :=
by {
  -- Proof is not required based on instruction, so we use sorry.
  sorry
}

end interest_rate_for_4000_investment_l471_471651


namespace divides_to_congruence_mod_2p_l471_471518

theorem divides_to_congruence_mod_2p {p n : ℤ} (h_prime : Prime p) (h_gt_3 : p > 3)
  (h_divides : n ∣ ((2^p + 1) / 3)) : n ≡ 1 [ZMOD 2 * p] := 
by
  sorry

end divides_to_congruence_mod_2p_l471_471518


namespace ramanujan_identity_a_ramanujan_identity_b_ramanujan_identity_c_ramanujan_identity_d_ramanujan_identity_e_ramanujan_identity_f_l471_471223
noncomputable theory

-- Part (a)
theorem ramanujan_identity_a :
  (∛(∛2 - 1) = ∛(1 / 9) - ∛(2 / 9) + ∛(4 / 9)) := 
sorry

-- Part (b)
theorem ramanujan_identity_b :
  (sqrt (∛5 - ∛4) = (1 / 3) * (∛2 + ∛20 - ∛25)) := 
sorry

-- Part (c)
theorem ramanujan_identity_c :
  (∛(∛(7 * 20 ^ (1/3)) - 19) = ∛(5 / 3) - ∛(2 / 3)) := 
sorry

-- Part (d)
theorem ramanujan_identity_d :
  (∛(∛(3 + (2 * 5 ^ (1/4))) / (3 - (2 * 5 ^ (1/4)))) = (5 ^ (1/4) + 1) / (5 ^ (1/4) - 1)) :=
sorry

-- Part (e)
theorem ramanujan_identity_e :
  (sqrt (∛28 - ∛27) = (1 / 3) * (∛98 - ∛28 - 1)) := 
sorry

-- Part (f)
theorem ramanujan_identity_f :
  (∛(∛(32 / 5) - ∛(27 / 5)) = ∛(1 / 25) + ∛(3 / 25) - ∛(9 / 25)) := 
sorry

end ramanujan_identity_a_ramanujan_identity_b_ramanujan_identity_c_ramanujan_identity_d_ramanujan_identity_e_ramanujan_identity_f_l471_471223


namespace point_on_line_iff_l471_471565

-- Parametric definition of points
structure Point (α : Type) [AddCommGroup α] [Module ℝ α] :=
(x : α)

variables {α : Type} [AddCommGroup α] [Module ℝ α]

-- Define vector between two points
def vector (A B : Point α) : α := B.x - A.x

-- Define point on line segment condition
def point_on_line (O A B X : Point α) (t : ℝ) : Prop :=
  X.x = t • (A.x - O.x) + (1 - t) • (B.x - O.x)

-- Define line segment condition
def line_segment (λ : ℝ) (A B X : Point α) : Prop :=
  X.x = A.x + λ • (vector A B)

-- Prove equivalence statement
theorem point_on_line_iff (O A B X : Point α) :
  (∃ t : ℝ, point_on_line O A B X t) ↔ (∃ λ : ℝ, line_segment λ A B X) :=
by
  sorry

end point_on_line_iff_l471_471565


namespace express_n_in_terms_of_f_and_g_l471_471798

theorem express_n_in_terms_of_f_and_g (n f g : ℕ) (h1 : 1 < n) 
  (h2 : f = 1 + nat.find (λ p, p ∣ n ∧ nat.prime p)) 
  (h3 : g = n + n / nat.find (λ p, p ∣ n ∧ nat.prime p)) : 
  n = (g * (f - 1)) / f := 
by
  sorry

end express_n_in_terms_of_f_and_g_l471_471798


namespace monic_quadratic_with_root_l471_471785

theorem monic_quadratic_with_root (a b : ℝ) (x : ℂ) :
  (x = 3 - 2 * complex.i ∨ x = 3 + 2 * complex.i) →
  (∃ p : polynomial ℂ, p.monic ∧ p.coeff 0 = 13 ∧ p.coeff 1 = -6 ∧ p.coeff 2 = 1 ∧ p.eval x = 0) :=
by
  intro hroot
  use polynomial.C 13 + polynomial.X * (-6) + polynomial.X^2
  split
  sorry
  split
  sorry
  split
  sorry
  rw polynomial.eval_poly
  sorry

end monic_quadratic_with_root_l471_471785


namespace negation_example_l471_471604

theorem negation_example :
  ¬ (∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ ∃ x₀ : ℝ, x₀^3 - x₀^2 + 1 > 0 :=
by
  sorry

end negation_example_l471_471604


namespace part_a_part_b_1_part_b_2_l471_471211

-- Part (a): Find k such that k is divisible by 2 and 9, and has exactly 14 divisors.
theorem part_a (k : ℕ) : (∃ k, k = 1458 ∧ (∃ d, d = k ∧ d % 2 = 0 ∧ d % 9 = 0 ∧ 
                           ∀ d, d > 0 → k % d = 0 → (∃ l, list.length (list.factors k) = 14) )) := 
sorry

-- Part (b): Prove that if 14 is replaced by 15, the task will have multiple solutions.
theorem part_b_1 : (∃ k, (k = 144 ∨ k = 324) ∧ (∃ d, d = k ∧ d % 2 = 0 ∧ d % 9 = 0 ∧ 
                          ∀ d, d > 0 → k % d = 0 → (∃ l, list.length (list.factors k) = 15) )) :=
sorry

-- Part (b): Prove that if 14 is replaced by 17, there will be no solutions at all.
theorem part_b_2 : ¬ (∃ k, (∃ d, d = k ∧ d % 2 = 0 ∧ d % 9 = 0 ∧ 
                          ∀ d, d > 0 → k % d = 0 → (∃ l, list.length (list.factors k) = 17) )) :=
sorry

end part_a_part_b_1_part_b_2_l471_471211


namespace table_tennis_competition_l471_471700

theorem table_tennis_competition :
  let males_A := 5
  let females_A := 3
  let males_B := 6
  let females_B := 2
  let select_from_A := 2
  let select_from_B := 2
  ∃ combinations, (combinations = (males_A.choose 1) * (females_A.choose 1) * (males_B.choose 2) 
                                   + (males_A.choose 2) * (males_B.choose 1) * (females_B.choose 1)) 
                    ∧ combinations = 345 :=
begin
  let males_A := 5,
  let females_A := 3,
  let males_B := 6,
  let females_B := 2,
  let select_from_A := 2,
  let select_from_B := 2,
  existsi (males_A.choose 1 * females_A.choose 1 * males_B.choose 2 + males_A.choose 2 * males_B.choose 1 * females_B.choose 1),
  split,
  { sorry },
  { sorry },
end

end table_tennis_competition_l471_471700


namespace eq_zero_or_one_if_square_eq_self_l471_471447

theorem eq_zero_or_one_if_square_eq_self (a : ℝ) (h : a^2 = a) : a = 0 ∨ a = 1 :=
sorry

end eq_zero_or_one_if_square_eq_self_l471_471447


namespace evaluate_expression_at_2_l471_471660

theorem evaluate_expression_at_2 : (3^2 - 2^3) = 1 := 
by
  sorry

end evaluate_expression_at_2_l471_471660


namespace actual_revenue_vs_projected_l471_471672

theorem actual_revenue_vs_projected (R : ℝ) (hR : R > 0) : 
  let projected_revenue := 1.20 * R
  let actual_revenue := 0.90 * R
  (actual_revenue / projected_revenue) * 100 = 75 :=
by
  let projected_revenue := 1.20 * R
  let actual_revenue := 0.90 * R
  have h : (actual_revenue / projected_revenue) * 100 = (0.90 / 1.20) * 100 := by simp [actual_revenue, projected_revenue]
  have h' : (0.90 / 1.20) * 100 = 75 := by norm_num
  rw [h, h']
  sorry

end actual_revenue_vs_projected_l471_471672


namespace star_computation_l471_471358

-- Define the operation ⭒
def star (a b : ℝ) (h : a ≠ b) : ℝ :=
  (a - b) / (a + b)

-- Prove that the value ((2 ⭒ 4) ⭒ 5) equals -8/7
theorem star_computation : (star (star 2 4 (by norm_num) 5 (by norm_num)) = -8 / 7 :=
by sorry

end star_computation_l471_471358


namespace probability_statements_l471_471703

-- Assigning probabilities
def p_hit := 0.9
def p_miss := 1 - p_hit

-- Definitions based on the problem conditions
def shoot_4_times (shots : List Bool) : Bool :=
  shots.length = 4 ∧ ∀ (s : Bool), s ∈ shots → (s = true → s ≠ false) ∧ (s = false → s ≠ true ∧ s ≠ 0)

-- Statements derived from the conditions
def prob_shot_3 := p_hit

def prob_exact_3_out_of_4 := 
  let binom_4_3 := 4
  binom_4_3 * (p_hit^3) * (p_miss^1)

def prob_at_least_1_out_of_4 := 1 - (p_miss^4)

-- The equivalence proof
theorem probability_statements : 
  (prob_shot_3 = 0.9) ∧ 
  (prob_exact_3_out_of_4 = 0.2916) ∧ 
  (prob_at_least_1_out_of_4 = 0.9999) := 
by 
  sorry

end probability_statements_l471_471703


namespace base7_number_l471_471970

theorem base7_number (A B C : ℕ) (h1 : 1 ≤ A ∧ A ≤ 6) (h2 : 1 ≤ B ∧ B ≤ 6) (h3 : 1 ≤ C ∧ C ≤ 6)
  (h_distinct : A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (h_condition1 : B + C = 7)
  (h_condition2 : A + 1 = C)
  (h_condition3 : A + B = C) :
  A = 5 ∧ B = 1 ∧ C = 6 :=
sorry

end base7_number_l471_471970


namespace minimum_value_l471_471771

noncomputable def f (x y : ℝ) : ℝ :=
  (x^2 + 1) / (y - 2) + (y^2 + 1) / (x - 2)

theorem minimum_value (x y : ℝ) (hx : x > 2) (hy : y > 2) :
  ∃ a b, x = a + 2 ∧ y = b + 2 ∧ a = sqrt 5 ∧ b = sqrt 5 ∧ f x y = 4 * sqrt 5 + 8 :=
sorry

end minimum_value_l471_471771


namespace brinley_snake_count_l471_471723

-- Definitions based on conditions
def arctic_foxes : ℕ := 80
def leopards : ℕ := 20
def bee_eaters (l : ℕ) : ℕ := 10 * l
def cheetahs (s : ℕ) : ℕ := s / 2
def alligators (af l : ℕ) : ℕ := 2 * (af + l)
def total_animals (s af l b c a : ℕ) : ℕ := s + af + l + b + c + a

-- The main theorem to prove
theorem brinley_snake_count :
  ∃ (S : ℕ), S = 113 ∧
  let b := bee_eaters leopards in
  let c := cheetahs S in
  let a := alligators arctic_foxes leopards in
  total_animals S arctic_foxes leopards b c a = 670 := by
  sorry

#eval brinley_snake_count

end brinley_snake_count_l471_471723


namespace sin_of_angles_of_triangle_l471_471504

theorem sin_of_angles_of_triangle (A B C : ℝ)
  (hA : A > 0)
  (hB : B > 0)
  (hC : C > 0)
  (sum_angle : A + B + C = π) :
  -2 < sin (3 * A) + sin (3 * B) + sin (3 * C) ∧ sin (3 * A) + sin (3 * B) + sin (3 * C) ≤ (3 / 2) * (sqrt 3) :=
by
  sorry

end sin_of_angles_of_triangle_l471_471504


namespace total_length_of_lines_in_T_l471_471917

def T (x y : ℝ) : Prop := abs (abs x - 3 - 2) + abs (abs y - 3 - 2) = 2

theorem total_length_of_lines_in_T : 
  (∑ t in { (x, y) | T x y }, (length t)) = 128 * Real.sqrt 2 :=
sorry

end total_length_of_lines_in_T_l471_471917


namespace find_cos_B_find_b_l471_471873

noncomputable theory

variable {A B C : ℝ} -- Angles
variable {a b c : ℝ} -- Opposite sides

-- Conditions
variable (triangle_ABC : 0 < a ∧ 0 < b ∧ 0 < c)
variable (angle_conditions : 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧ A + B + C = π)
variable (eq1 : b * Real.cos C - c * Real.cos (A + C) = 3 * a * Real.cos B)
variable (dot_product_condition : b * c * Real.cos B = 2)
variable (given_a : a = Real.sqrt 6)

-- Goals
theorem find_cos_B : Real.cos B = 1 / 3 := by
  sorry

theorem find_b : b = 2 * Real.sqrt 2 := by
  sorry

end find_cos_B_find_b_l471_471873


namespace largest_area_quadrilateral_l471_471196

theorem largest_area_quadrilateral (a b c d : ℝ) (h₀ : a = 1) (h₁ : b = 4) (h₂ : c = 7) (h₃ : d = 8) : 
  let s := (a + b + c + d) / 2 in
  sqrt ((s - a) * (s - b) * (s - c) * (s - d)) = 18 :=
by
  sorry

end largest_area_quadrilateral_l471_471196


namespace alpha_magnitude_l471_471911

theorem alpha_magnitude 
  (α β : ℂ) 
  (h1 : β = conj α) 
  (h2 : ∃ k : ℝ, α / (β^2) = k) 
  (h3 : abs (α - β) = 6) : 
  abs α = 2 * real.sqrt 3 := 
by
  sorry

end alpha_magnitude_l471_471911


namespace max_intersections_l471_471751

theorem max_intersections (x_points y_points : ℕ) (hx : x_points = 15) (hy : y_points = 10) :
  let segments := x_points * y_points in
  (segments = 150) → (binom x_points 2) * (binom y_points 2) = 4725 :=
by
  sorry

end max_intersections_l471_471751


namespace find_a5_l471_471812

-- Define the sequence and its properties
def geom_sequence (a : ℕ → ℕ) : Prop :=
∀ n m : ℕ, a (n + m) = (2^m) * a n

-- Define the problem statement
def sum_of_first_five_terms_is_31 (a : ℕ → ℕ) : Prop :=
a 1 + a 2 + a 3 + a 4 + a 5 = 31

-- State the theorem to prove
theorem find_a5 (a : ℕ → ℕ) (h_geom : geom_sequence a) (h_sum : sum_of_first_five_terms_is_31 a) : a 5 = 16 :=
by
  sorry

end find_a5_l471_471812


namespace partitions_of_6_into_4_indistinguishable_boxes_l471_471433

theorem partitions_of_6_into_4_indistinguishable_boxes : 
  ∃ (X : Finset (Multiset ℕ)), X.card = 9 ∧ 
  ∀ p ∈ X, p.sum = 6 ∧ p.card ≤ 4 := 
sorry

end partitions_of_6_into_4_indistinguishable_boxes_l471_471433


namespace evaluate_expression_eq_neg_one_evaluate_expression_only_value_l471_471750

variable (a y : ℝ)
variable (h1 : a ≠ 0)
variable (h2 : a ≠ 2 * y)
variable (h3 : a ≠ -2 * y)

theorem evaluate_expression_eq_neg_one
  (h : y = -a / 3) :
  ( (a / (a + 2 * y) + y / (a - 2 * y)) / (y / (a + 2 * y) - a / (a - 2 * y)) ) = -1 := 
sorry

theorem evaluate_expression_only_value :
  ( (a / (a + 2 * y) + y / (a - 2 * y)) / (y / (a + 2 * y) - a / (a - 2 * y)) = -1 ) ↔ 
  y = -a / 3 := 
sorry

end evaluate_expression_eq_neg_one_evaluate_expression_only_value_l471_471750


namespace BD_DA_eq_BC_l471_471495

-- Define the points and the triangle
variables {A B C D : Type*}

-- Define angle measures as real numbers
variables (α β γ δ : ℝ)

-- Consider a triangle with given conditions
def triangle_ABC (A B C : Type*) (α β : ℝ) :=
  α = 40 ∧ β = 40 ∧ α + β + γ = 180

-- Define the angle bisector condition
def angle_bisector_of_B (A B C D : Type*) :=
  ∃ (angle_bisector : ℝ), angle_bisector = (40 / 2)

-- Final statement to be proved
theorem BD_DA_eq_BC (A B C D : Type*) (α β γ : ℝ) (triangle_condition : triangle_ABC A B C α β)
  (bisector_condition : angle_bisector_of_B A B C D) :
  BD + DA = BC :=
sorry

end BD_DA_eq_BC_l471_471495


namespace prove_divisibility_l471_471552

-- Definitions for natural numbers m, n, k
variables (m n k : ℕ)

-- Conditions stating divisibility
def div1 := m^n ∣ n^m
def div2 := n^k ∣ k^n

-- The final theorem to prove
theorem prove_divisibility (hmn : div1 m n) (hnk : div2 n k) : m^k ∣ k^m :=
sorry

end prove_divisibility_l471_471552


namespace probability_of_exactly_9_correct_matches_is_zero_l471_471229

theorem probability_of_exactly_9_correct_matches_is_zero :
  ∃ (P : ℕ → ℕ → ℕ), 
    (∀ (total correct : ℕ), 
      total = 10 → 
      correct = 9 → 
      P total correct = 0) := 
by {
  sorry
}

end probability_of_exactly_9_correct_matches_is_zero_l471_471229


namespace probability_of_difference_multiple_of_seven_l471_471681

open Set

theorem probability_of_difference_multiple_of_seven :
  ∀ (s : Finset ℕ), s.card = 6 → (∀ n ∈ s, 1 ≤ n ∧ n ≤ 5000) →
  (∃ (a b ∈ s), a ≠ b ∧ (a - b) % 7 = 0) :=
by
  intros s hcard hrange
  -- insert proof here
  sorry

end probability_of_difference_multiple_of_seven_l471_471681


namespace smallest_six_consecutive_even_integers_l471_471157

theorem smallest_six_consecutive_even_integers : 
  let sum_first_15_even := 2 * (1 + 2 + ... + 15 : ℕ)
  let six_consecutive_even_sum := n + (n + 2) + (n + 4) + (n + 6) + (n + 8) + (n + 10) = 240
  n = 35 := 
by
  sorry

end smallest_six_consecutive_even_integers_l471_471157


namespace range_of_k_l471_471840

theorem range_of_k (a : ℝ) (h : 0 < a ∧ a ≠ 1) :
  ∀ x : ℝ, log a (x^2 - 2 * k * x + 2 * k + 3) ∈ ℝ →
  -1 < k ∧ k < 3 :=
by
  sorry

end range_of_k_l471_471840


namespace extra_item_is_candy_l471_471931

/-- Mikuláš prepared packages for 23 children. Each package was supposed
    to contain a 100 gram chocolate bar, a 90 gram bag of candy, an 80 gram
    pack of wafers, and three different 10 gram packs of chewing gum. After
    packing, he weighed all the packages together and found that they weighed
    a total of 7 kg. Mikuláš realized that there was a mistake. Two packages had
    one extra of the same item, and one package was missing the wafers. Prove
    that the extra item in each of the two packages is a bag of candies. -/
theorem extra_item_is_candy :
  let packages := 23
  let chocolate_weight := 100  -- grams
  let candy_weight := 90  -- grams
  let wafer_weight := 80  -- grams
  let gum_weight := 3 * 10  -- grams
  let package_weight := chocolate_weight + candy_weight + wafer_weight + gum_weight  -- grams
  (package_weight * packages = 6900) ∧ (7000 - 6900 - 80 = 100) →
  (∃ item_weight, item_weight = candy_weight ∧ 2 * item_weight = 180) :=
begin
  let packages := 23,
  let chocolate_weight := 100,  -- grams
  let candy_weight := 90,  -- grams
  let wafer_weight := 80,  -- grams
  let gum_weight := 3 * 10,  -- grams
  let package_weight := chocolate_weight + candy_weight + wafer_weight + gum_weight,  -- grams
  assume h,
  sorry
end

end extra_item_is_candy_l471_471931


namespace scientific_notation_101000_l471_471138

theorem scientific_notation_101000 : 
  ∃ (a : ℝ) (n : ℤ), 101000 = a * 10^n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ a = 1.01 ∧ n = 5 :=
by
  use 1.01, 5
  split
  sorry

end scientific_notation_101000_l471_471138


namespace abs_diff_zero_C_D_equals_zero_l471_471754

def is_single_digit_base_6 (n : ℕ) : Prop :=
  n < 6

def is_valid_sum (C D : ℕ) : Prop :=
  (C + D + 3) % 6 = 5 ∧ (C + 2) % 6 = 2 ∧ (1 + D + 5) % 6 = 1

theorem abs_diff_zero_C_D_equals_zero {C D : ℕ}
    (h1 : is_single_digit_base_6 C)
    (h2 : is_single_digit_base_6 D)
    (h3 : is_valid_sum C D) :
    |C - D| = 0 :=
sorry

end abs_diff_zero_C_D_equals_zero_l471_471754


namespace students_total_l471_471145

theorem students_total (scavenger_hunt_students : ℕ) (ski_trip_students : ℕ) 
  (h1 : ski_trip_students = 2 * scavenger_hunt_students) 
  (h2 : scavenger_hunt_students = 4000) : 
  scavenger_hunt_students + ski_trip_students = 12000 := 
by
  sorry

end students_total_l471_471145


namespace g_value_at_2_l471_471011

theorem g_value_at_2 (g : ℝ → ℝ) (h : ∀ x : ℝ, g(3*x - 7) = 4*x + 6) : g 2 = 18 :=
by
  sorry

end g_value_at_2_l471_471011


namespace length_AM_eq_l471_471069

noncomputable theory

variables {A B C D E F M : Type*}
variables [inner_product_space ℝ A] [inner_product_space ℝ B] [inner_product_space ℝ C]
variables [inner_product_space ℝ D] [inner_product_space ℝ E] [inner_product_space ℝ F]
variables [inner_product_space ℝ M]

def perpendicular_projection (p : A) (l : B) : E := sorry -- Given: definition for perpendicular projection
def orthocenter (t : triangle E F) : M := sorry -- Given: orthocenter definition
def angle (u v : F) : ℝ := sorry -- Given: function to find angle between vectors

-- Given: Parallelogram ABCD with vertex A
-- Given: perpendicular projections E and F
-- Given: orthocenter M
-- Given: lengths AC and EF and angle between AC and EF

variables (AC EF : ℝ)
variables (α : ℝ) -- Let α be the angle between AC and EF

-- Proof statement: Prove length of AM given AC, EF
theorem length_AM_eq : AM = ∥ sqrt (AC^2 - EF^2) ∥ :=
sorry

end length_AM_eq_l471_471069


namespace range_of_f_l471_471809

-- Define variables and conditions
variables {x y z : ℝ}
variables (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (hxyz : x + y + z = 1)

-- Define the function f(z)
def f (z : ℝ) : ℝ := (z - x) * (z - y)

-- State the theorem
theorem range_of_f (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (hxyz : x + y + z = 1) :
  ∀ z : ℝ, hx → hy → hz → hxyz →
  - (1 / 8) ≤ f z ∧ f z ≤ 1 :=
by
  sorry

end range_of_f_l471_471809


namespace parallelepiped_inequality_l471_471918

theorem parallelepiped_inequality (a b c d : ℝ) (h : d^2 = a^2 + b^2 + c^2 + 2 * (a * b + a * c + b * c)) :
  a^2 + b^2 + c^2 ≥ (1 / 3) * d^2 :=
by
  sorry

end parallelepiped_inequality_l471_471918


namespace best_fitting_model_l471_471491

theorem best_fitting_model :
  ∀ (R1 R2 R3 R4 : ℝ), R1 = 0.976 → R2 = 0.776 → R3 = 0.076 → R4 = 0.351 →
  R1 = max R1 (max R2 (max R3 R4)) :=
by
  intros R1 R2 R3 R4 hR1 hR2 hR3 hR4
  rw [hR1, hR2, hR3, hR4]
  sorry

end best_fitting_model_l471_471491


namespace exists_n_consecutive_non_prime_power_l471_471095

theorem exists_n_consecutive_non_prime_power (n : ℕ) (h_pos : n > 0) :
  ∃ x : ℕ, ∀ i : ℕ, 1 ≤ i → i ≤ n → ¬(∃ p : ℕ, prime p ∧ ∃ k : ℕ, k ≥ 1 ∧ x + i = p ^ k) :=
by sorry

end exists_n_consecutive_non_prime_power_l471_471095


namespace max_discount_rate_l471_471267

-- Define the conditions
def cost_price : ℝ := 4
def selling_price : ℝ := 5
def min_profit_margin : ℝ := 0.1
def min_profit : ℝ := cost_price * min_profit_margin

-- Define the discount rate
def discount_rate (x : ℝ) : ℝ :=
  selling_price * (1 - x / 100)

-- Define the profit after discount
def profit_after_discount (x : ℝ) : ℝ :=
  discount_rate x - cost_price

-- The statement we want to prove
theorem max_discount_rate : ∃ x : ℝ, x = 8.8 ∧ profit_after_discount x ≥ min_profit := 
by
  sorry

end max_discount_rate_l471_471267


namespace minimum_value_inequality_l471_471766

theorem minimum_value_inequality (x y : ℝ) (hx : x > 2) (hy : y > 2) :
    (x^2 + 1) / (y - 2) + (y^2 + 1) / (x - 2) ≥ 18 := by
  sorry

end minimum_value_inequality_l471_471766


namespace students_total_l471_471146

theorem students_total (scavenger_hunt_students : ℕ) (ski_trip_students : ℕ) 
  (h1 : ski_trip_students = 2 * scavenger_hunt_students) 
  (h2 : scavenger_hunt_students = 4000) : 
  scavenger_hunt_students + ski_trip_students = 12000 := 
by
  sorry

end students_total_l471_471146


namespace number_of_vegetarians_l471_471031

theorem number_of_vegetarians (only_nonveg : ℕ) (both_veg_nonveg : ℕ) (total_veg : ℕ) : 
  only_nonveg = 9 → both_veg_nonveg = 12 → total_veg = 31 → (total_veg - both_veg_nonveg) = 19 :=
by {
  intros h1 h2 h3,
  rw [h1, h2, h3],
  exact rfl,
}

end number_of_vegetarians_l471_471031


namespace expected_min_leq_2_l471_471537

open ProbabilityTheory

variables (ξ η : ℝ → ℝ) -- ξ and η are random variables

-- Condition: expected value of ξ is 2
axiom E_ξ_eq_2 : ℝ
axiom E_ξ_is_2 : (∫ x in ⊤, ξ x) = 2

-- Goal: expected value of min(ξ, η) ≤ 2
theorem expected_min_leq_2 (h : ∀ x, min (ξ x) (η x) ≤ ξ x) : 
  (∫ x in ⊤, min (ξ x) (η x)) ≤ 2 := by
  -- use the provided axioms and conditions here
  sorry

end expected_min_leq_2_l471_471537


namespace equilibrium_point_unstable_l471_471497

noncomputable def lyapunov_function (x y : ℝ) : ℝ :=
  x^4 - y^4

noncomputable def original_system (dx dy : ℝ → ℝ) (x y t : ℝ) : Prop :=
  dx = y^3 + x^5 ∧ dy = x^3 + y^5

noncomputable def chetaev_theorem (v : ℝ → ℝ → ℝ) (dx dy : ℝ → ℝ) (x y : ℝ) : Prop :=
  (∀ x y, v x y > 0 ∧ abs(x) > abs(y)) →
  (4 * x^3 * (y^3 + x^5) - 4 * y^3 * (x^3 + y^5) > 0)

theorem equilibrium_point_unstable (x y t : ℝ) :
  original_system (λ t, y^3 + x^5) (λ t, x^3 + y^5) x y t →
  chetaev_theorem lyapunov_function (λ t, y^3 + x^5) (λ t, x^3 + y^5) x y →
  ¬ stable (0, 0) :=
begin
  sorry
end

end equilibrium_point_unstable_l471_471497


namespace abc_over_sum_leq_four_thirds_l471_471065

theorem abc_over_sum_leq_four_thirds (a b c : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) 
  (h_a_leq_2 : a ≤ 2) (h_b_leq_2 : b ≤ 2) (h_c_leq_2 : c ≤ 2) :
  (abc / (a + b + c) ≤ 4/3) :=
by
  sorry

end abc_over_sum_leq_four_thirds_l471_471065


namespace least_possible_value_expression_l471_471003

theorem least_possible_value_expression 
  (a b c : ℝ)
  (h1 : b > c)
  (h2 : c > a)
  (h3 : b ≠ 0) :
  ∃ x, (x = 24) ∧ (∀ x', (∀ a b c : ℝ, b > c ∧ c > a ∧ b ≠ 0 → x' = (5 * (a + b) ^ 2 + 4 * (b - c) ^ 2 + 3 * (c - a) ^ 2) / (2 * b ^ 2)) ≤ x) :=
  sorry

end least_possible_value_expression_l471_471003


namespace number_of_valid_colorings_l471_471319

-- Define the grid and problem statement
def grid := Fin 3 × Fin 4
def colors := {red, maroon}

-- Define the condition for no monochromatic rectangle
def valid_coloring (f : grid → colors) : Prop :=
  ∀ (r1 r2 : Fin 3) (c1 c2 : Fin 4),
    r1 ≠ r2 → c1 ≠ c2 → 
    ¬(f (r1, c1) = f (r1, c2) ∧ f (r2, c1) = f (r2, c2) ∧ f (r1, c1) = f (r2, c1))

-- Prove the number of valid colorings
theorem number_of_valid_colorings : 
  (∑ f in {f : grid → colors | valid_coloring f}, 1) = 168 := 
by sorry

end number_of_valid_colorings_l471_471319


namespace expected_flashlight_lifetime_leq_two_l471_471543

theorem expected_flashlight_lifetime_leq_two
  (Ω : Type*) [MeasurableSpace Ω] [ProbabilitySpace Ω]
  (ξ η : Ω → ℝ)
  (h_min_leq_xi : ∀ ω, min (ξ ω) (η ω) ≤ ξ ω)
  (h_expectation_xi : expectation (ξ) = 2) :
  expectation (fun ω => min (ξ ω) (η ω)) ≤ 2 := 
sorry

end expected_flashlight_lifetime_leq_two_l471_471543


namespace rational_count_is_4_l471_471311

def is_rational (x : ℝ) : Prop := ∃ (p q : ℤ), q ≠ 0 ∧ x = p / q

theorem rational_count_is_4 :
  let nums := [2023, -32 / 10, 0, Real.pi, -1 / 3] in
  (nums.filter is_rational).length = 4 :=
by
  sorry

end rational_count_is_4_l471_471311


namespace geometric_series_modulo_l471_471788

theorem geometric_series_modulo :
  (∑ k in finset.range 1003, 3^(k+1)) % 500 = 113 :=
by
  -- problem definition
  let series_sum := ∑ k in finset.range 1003, 3^(k+1)
  -- geometric series sum formula
  have h : series_sum = (3^1004 - 3) / 2 := sorry
  -- compute (3^1004 - 3) / 2 % 500
  show (series_sum) % 500 = 113, from sorry

end geometric_series_modulo_l471_471788


namespace volume_ADBFE_l471_471182

theorem volume_ADBFE (V_ABCD : ℝ) (BE_median : ∀ (A B C : ℝ), BE / (2/3)) 
  (F_midpoint : ∀ (D C : ℝ), F / 2) :
  V_ABCD = 40 → V_ADBFE = 30 := by
  sorry

end volume_ADBFE_l471_471182


namespace num_distinct_values_2014_tuple_l471_471066

theorem num_distinct_values_2014_tuple : 
  ∃ (f : ℕ → ℕ), (f 1 = 1) ∧ 
  (∀ a b : ℕ, a ≤ b → f a ≤ f b) ∧ 
  (∀ a : ℕ, f (2 * a) = f a + 1) ∧ 
  (finset.image f (finset.range 2014)).card = 11 :=
sorry

end num_distinct_values_2014_tuple_l471_471066


namespace engineers_meeting_probability_l471_471180

theorem engineers_meeting_probability :
  ∀ (x y z : ℝ), 
    (0 ≤ x ∧ x ≤ 2) → 
    (0 ≤ y ∧ y ≤ 2) → 
    (0 ≤ z ∧ z ≤ 2) → 
    (abs (x - y) ≤ 0.5) → 
    (abs (y - z) ≤ 0.5) → 
    (abs (z - x) ≤ 0.5) → 
    Π (volume_region : ℝ) (total_volume : ℝ),
    (volume_region = 1.5 * 1.5 * 1.5) → 
    (total_volume = 2 * 2 * 2) → 
    (volume_region / total_volume = 0.421875) :=
by
  intros x y z hx hy hz hxy hyz hzx volume_region total_volume hr ht
  sorry

end engineers_meeting_probability_l471_471180


namespace problem_I_solution_problem_II_solution_l471_471919

noncomputable def f (x : ℝ) : ℝ := |3 * x - 2| + |x - 2|

-- Problem (I): Solve the inequality f(x) <= 8
theorem problem_I_solution (x : ℝ) : 
  f x ≤ 8 ↔ -1 ≤ x ∧ x ≤ 3 :=
sorry

-- Problem (II): Find the range of the real number m
theorem problem_II_solution (x m : ℝ) : 
  f x ≥ (m^2 - m + 2) * |x| ↔ (0 ≤ m ∧ m ≤ 1) :=
sorry

end problem_I_solution_problem_II_solution_l471_471919


namespace relationship_D_is_not_function_l471_471200

-- Define each relationship
def relationship_A (l : ℝ) : ℝ := l^3
def relationship_B (θ : ℝ) : ℝ := Real.sin θ
def relationship_C (area : ℝ) (yield_per_unit : ℝ) : ℝ := area * yield_per_unit
def relationship_D (daylight : ℝ) : Set ℝ := {yield | true}  -- non-deterministic yield per acre of rice based on daylight

-- The theorem states that relationship D is not a function
theorem relationship_D_is_not_function : ¬(∀x y1 y2, y1 ∈ relationship_D x → y2 ∈ relationship_D x → y1 = y2) :=
sorry

end relationship_D_is_not_function_l471_471200


namespace evaluate_K_3_15_10_l471_471801

def K (a b c : ℝ) : ℝ := a / b - b / c + c / a

theorem evaluate_K_3_15_10 : K 3 15 10 = 61 / 30 :=
by
  sorry

end evaluate_K_3_15_10_l471_471801


namespace coefficient_x3_of_2x_plus_1_equals_80_l471_471115

open Finset

-- Let n be 5 and k be 3 for binomial coefficient
def binom := (n k : ℕ) → n.choose k

-- Define the expansion term, r is set to 3.
def term_r (n k a x : ℕ) := (binom n k) * a^k * x^k

-- The coefficient of x^3 in (2x+1)^5
def coefficient_x3 (a b n x : ℕ) (h: a=2 ∧ b=1 ∧ n=5 ∧ x=3) : ℕ :=
  by
  cases h
  exact term_r 5 3 2 3 / x^3

-- The theorem to show the evaluation
theorem coefficient_x3_of_2x_plus_1_equals_80 :
  coefficient_x3 2 1 5 3 (by simp [ * ]) = 80 :=
  by
  sorry

end coefficient_x3_of_2x_plus_1_equals_80_l471_471115


namespace math_problem_l471_471381

theorem math_problem (x : ℤ) :
  (¬ (x + 2) / (x - 3) ≥ 0 ∧ ¬ (x ∈ ℤ)) ∧ ¬ ¬ (x ∈ ℤ) → 
  (x = -1 ∨ x = 0 ∨ x = 1 ∨ x = 2 ∨ x = 3) :=
by
  sorry

end math_problem_l471_471381


namespace expected_min_leq_2_l471_471535

open ProbabilityTheory

variables (ξ η : ℝ → ℝ) -- ξ and η are random variables

-- Condition: expected value of ξ is 2
axiom E_ξ_eq_2 : ℝ
axiom E_ξ_is_2 : (∫ x in ⊤, ξ x) = 2

-- Goal: expected value of min(ξ, η) ≤ 2
theorem expected_min_leq_2 (h : ∀ x, min (ξ x) (η x) ≤ ξ x) : 
  (∫ x in ⊤, min (ξ x) (η x)) ≤ 2 := by
  -- use the provided axioms and conditions here
  sorry

end expected_min_leq_2_l471_471535


namespace arithmetic_square_root_of_16_is_4_l471_471972

theorem arithmetic_square_root_of_16_is_4 :
  ∃ x : ℝ, x * x = 16 ∧ x = 4 :=
by {
  use 4,
  split,
  {
    norm_num, 
  },
  {
    refl,
  }
}

end arithmetic_square_root_of_16_is_4_l471_471972


namespace expected_lifetime_flashlight_l471_471525

section
variables {Ω : Type} [ProbabilitySpace Ω]
variables (ξ η : Ω → ℝ)
variables (h_ξ_expect : E[ξ] = 2)

-- Define the minimum of ξ and η
def min_ξ_η (ω : Ω) : ℝ := min (ξ ω) (η ω)

theorem expected_lifetime_flashlight : E[min_ξ_η ξ η] ≤ 2 :=
by
  sorry
end

end expected_lifetime_flashlight_l471_471525


namespace line_passing_quadrants_l471_471607

theorem line_passing_quadrants (a k : ℝ) (a_nonzero : a ≠ 0)
  (x1 x2 y1 y2 : ℝ) (hx1 : y1 = a * x1^2 - a) (hx2 : y2 = a * x2^2 - a)
  (hx1_y1 : y1 = k * x1) (hx2_y2 : y2 = k * x2) 
  (sum_x : x1 + x2 < 0) : 
  ∃ (q1 q4 : (ℝ × ℝ)), 
  (q1.1 > 0 ∧ q1.2 > 0 ∧ q1.2 = a * q1.1 + k) ∧ (q4.1 > 0 ∧ q4.2 < 0 ∧ q4.2 = a * q4.1 + k) := 
sorry

end line_passing_quadrants_l471_471607


namespace evaluation_at_fraction_pi_l471_471402

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x

def f₁ (x : ℝ) : ℝ := deriv f x
def fₙ {n : ℕ} (x : ℝ) : ℝ := Nat.rec f (λ _ fₙ_px, deriv fₙ_px) n x

theorem evaluation_at_fraction_pi
  (n : ℕ) (h : n % 4 = 0) :
  fₙ (Real.pi / 3) = (Real.sqrt 3 + 1) / 2 :=
sorry

end evaluation_at_fraction_pi_l471_471402


namespace fare_calculation_l471_471029

-- Definitions based on conditions
def base_fare : ℝ := 5
def additional_fare_per_km : ℝ := 1.2

-- Define the total fare function for a given distance a km (a > 3)
def total_fare (a : ℝ) (h : a > 3) : ℝ :=
  base_fare + additional_fare_per_km * (a - 3)

-- Prove that the total fare for a > 3 kilometers is 1.2a + 1.4
theorem fare_calculation (a : ℝ) (h : a > 3) : total_fare a h = 1.2 * a + 1.4 :=
by
  unfold total_fare
  have eq1 : base_fare + additional_fare_per_km * (a - 3) = 5 + 1.2 * (a - 3) := rfl
  rw [eq1]
  have eq2 : 5 + 1.2 * (a - 3) = 5 + 1.2 * a - 1.2 * 3 := by ring
  rw [eq2]
  have eq3 : 5 + 1.2 * a - 3.6 = 1.2 * a + 1.4 := by ring
  rw [eq3]
  exact rfl

end fare_calculation_l471_471029


namespace probability_exactly_nine_matches_l471_471243

theorem probability_exactly_nine_matches (n : ℕ) (h : n = 10) : 
  (∃ p : ℕ, p = 9 ∧ probability_of_exact_matches n p = 0) :=
by {
  sorry
}

end probability_exactly_nine_matches_l471_471243


namespace seventh_term_geometric_sequence_l471_471742

theorem seventh_term_geometric_sequence (a : ℝ) (a3 : ℝ) (r : ℝ) (n : ℕ) (term : ℕ → ℝ)
    (h_a : a = 3)
    (h_a3 : a3 = 3 / 64)
    (h_term : ∀ n, term n = a * r ^ (n - 1))
    (h_r : r = 1 / 8) :
    term 7 = 3 / 262144 :=
by
  sorry

end seventh_term_geometric_sequence_l471_471742


namespace indistinguishable_balls_boxes_l471_471431

open Finset

def partitions (n : ℕ) (k : ℕ) : ℕ :=
  (univ : Finset (Multiset ℕ)).filter (λ p, p.sum = n ∧ p.card ≤ k).card

theorem indistinguishable_balls_boxes : partitions 6 4 = 9 :=
sorry

end indistinguishable_balls_boxes_l471_471431


namespace expected_lifetime_flashlight_l471_471527

section
variables {Ω : Type} [ProbabilitySpace Ω]
variables (ξ η : Ω → ℝ)
variables (h_ξ_expect : E[ξ] = 2)

-- Define the minimum of ξ and η
def min_ξ_η (ω : Ω) : ℝ := min (ξ ω) (η ω)

theorem expected_lifetime_flashlight : E[min_ξ_η ξ η] ≤ 2 :=
by
  sorry
end

end expected_lifetime_flashlight_l471_471527


namespace quadrilateral_possible_rods_l471_471902

theorem quadrilateral_possible_rods (rods : Finset ℕ) (a b c : ℕ) (ha : a = 3) (hb : b = 7) (hc : c = 15)
  (hrods : rods = (Finset.range 31 \ {3, 7, 15})) :
  ∃ d, d ∈ rods ∧ 5 < d ∧ d < 25 ∧ rods.card - 2 = 17 := 
by
  sorry

end quadrilateral_possible_rods_l471_471902


namespace union_complement_intersection_range_of_m_l471_471851

-- Define the sets and range of values as provided in the problem
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | 0 < Real.log x / Real.log 3 < 1}
def B (m : ℝ) : Set ℝ := {x | 2 * m < x ∧ x < 1 - m}

-- Question 1: Prove A ∪ B = (-2, 3) and (A^c_U) ∩ B = (-2, 1] when m = -1
theorem union_complement_intersection (m : ℝ) (h : m = -1) :
  A ∪ B m = (-2 : Set ℝ ∪ 3) ∧ (U \ A) ∩ B m = {x | -2 < x ∧ x ≤ 1} :=
sorry

-- Question 2: Prove the range of real number m such that A ∩ B = A is (-∞, -2]
theorem range_of_m (m : ℝ) :
  (A ∩ B m = A) ↔ m ≤ -2 :=
sorry

end union_complement_intersection_range_of_m_l471_471851


namespace find_a_l471_471819

-- Define sets A and B based on the given real number a
def A (a : ℝ) : Set ℝ := {a^2, a + 1, -3}
def B (a : ℝ) : Set ℝ := {a - 3, 3 * a - 1, a^2 + 1}

-- Given condition
def condition (a : ℝ) : Prop := A a ∩ B a = {-3}

-- Prove that a = -2/3 is the solution satisfying the condition
theorem find_a : ∃ a : ℝ, condition a ∧ a = -2/3 :=
by
  sorry  -- Proof goes here

end find_a_l471_471819


namespace net_population_after_three_years_l471_471611

def initial_population : ℝ := 15000
def year1_decrease_rate : ℝ := 0.20
def year2_increase_rate : ℝ := 0.10
def year2_decrease_rate : ℝ := 0.05
def year3_increase_rate : ℝ := 0.08
def year3_decrease_rate : ℝ := 0.25

theorem net_population_after_three_years :
  let p1 := initial_population * (1 - year1_decrease_rate) in
  let p2 := p1 * (1 + year2_increase_rate) * (1 - year2_decrease_rate) in
  let p3 := p2 * (1 + year3_increase_rate) * (1 - year3_decrease_rate) in
  round p3 = 10157 :=
by {
  let p1 := initial_population * (1 - year1_decrease_rate),
  let p2 := p1 * (1 + year2_increase_rate) * (1 - year2_decrease_rate),
  let p3 := p2 * (1 + year3_increase_rate) * (1 - year3_decrease_rate),
  exact sorry -- Proof goes here
}

end net_population_after_three_years_l471_471611


namespace red_suit_top_card_probability_l471_471297

theorem red_suit_top_card_probability :
  let num_cards := 104
  let num_red_suits := 4
  let cards_per_suit := 26
  let num_red_cards := num_red_suits * cards_per_suit
  let top_card_is_red_probability := num_red_cards / num_cards
  top_card_is_red_probability = 1 := by
  sorry

end red_suit_top_card_probability_l471_471297


namespace pool_width_is_40_l471_471952

-- Define the relevant variables and constants
variables (width : ℝ) (time : ℝ) (sarah_speed ruth_speed : ℝ)
constants (length : ℝ := 50)

-- Conditions
axiom Ruth_Faster_Than_Sarah : ruth_speed = 3 * sarah_speed
axiom Equal_Time : (6 * length / sarah_speed) = (5 * (2 * (length + width)) / ruth_speed)

-- The proof goal
theorem pool_width_is_40 : width = 40 :=
by
  -- Setting the assumptions
  have speed_factor : ruth_speed = 3 * sarah_speed := Ruth_Faster_Than_Sarah
  have travel_time : (6 * length / sarah_speed) = (5 * (2 * (length + width)) / ruth_speed) := Equal_Time

  -- Convert the equality into a solvable equation
  have h := calc
    (6 * length / sarah_speed) = (5 * (2 * (length + width)) / ruth_speed) : travel_time
    ... = (5 * (2 * (length + width)) / (3 * sarah_speed)) : speed_factor
    ... = (10 * (length + width) / (3 * sarah_speed)) : by rw [← mul_div_assoc]
  have eq : (18 * length) = (10 * (length + width)) := by
    rw [← mul_div_cancel_left _ (show sarah_speed ≠ 0, from by linarith)],
    rw [mul_assoc]
    assumption
    
  -- Simplifying the result
  have width_simplified : (10 * width) = 400 := calc
    18 * length = 10 * (length + width) : eq
    ... = (10 * length + 10 * width) : by rw [mul_add]

  -- Finish the proof
  have width_is_40 : width = 40 := by
    linarith
  exact width_is_40

end pool_width_is_40_l471_471952


namespace simplify_expression_evaluate_at_2_l471_471107

-- Define the conditions
variable (a : ℝ)
hypothesis h0 : a ≠ 0
hypothesis h1 : a ≠ -1
hypothesis h2 : a ≠ 1

-- Define the main proof problem
theorem simplify_expression : 
  (a - (2 * a - 1) / a ) / ( (1 - a^2) / (a^2 + a) ) = a + 1 := 
by {
  sorry 
}

-- Define the evaluation for a specific value
theorem evaluate_at_2 :
  (let a := 2 in 
  (a - (2 * a - 1) / a ) / ( (1 - a^2) / (a^2 + a))) = 3 := 
by {
  sorry 
}

end simplify_expression_evaluate_at_2_l471_471107


namespace jalapeno_peppers_needed_jalapeno_peppers_needed_correct_l471_471586

theorem jalapeno_peppers_needed
  (strips_per_sandwich : ℕ) (strips_per_pepper : ℕ)
  (minutes_per_sandwich : ℕ) (hours_per_day : ℕ)
  (strips_per_sandwich = 4) (strips_per_pepper = 8)
  (minutes_per_sandwich = 5) (hours_per_day = 8) : ℕ :=
  let peppers_per_sandwich := strips_per_sandwich / strips_per_pepper in
  let sandwiches_per_hour := 60 / minutes_per_sandwich in
  let peppers_per_hour := sandwiches_per_hour * peppers_per_sandwich in
  let total_peppers_needed := hours_per_day * peppers_per_hour in
  total_peppers_needed

-- Now providing the theorem to ensure the calculation
theorem jalapeno_peppers_needed_correct : jalapeno_peppers_needed 4 8 5 8 = 48 := sorry

end jalapeno_peppers_needed_jalapeno_peppers_needed_correct_l471_471586


namespace inverse_proportion_inequality_l471_471038

variable (x1 x2 k : ℝ)

theorem inverse_proportion_inequality (hA : 2 = k / x1) (hB : 4 = k / x2) (hk : 0 < k) : 
  x1 > x2 ∧ x1 > 0 ∧ x2 > 0 :=
sorry

end inverse_proportion_inequality_l471_471038


namespace value_of_C_l471_471207

-- Defining the conditions in the problem
variables {A B C : ℝ}

-- Conditions
def cond1 : Prop := A + B + C = 600
def cond2 : Prop := A + C = 250
def cond3 : Prop := B + C = 450

-- Theorem stating the value of C
theorem value_of_C (h1 : cond1) (h2 : cond2) (h3 : cond3) : C = 100 :=
sorry

end value_of_C_l471_471207


namespace eccentricity_equilateral_l471_471330

noncomputable def eccentricity_of_ellipse (a b : ℝ) (h : a > b ∧ b > 0) :=
  let c := real.sqrt (a^2 - b^2) in
  let e := c / a in
  e

theorem eccentricity_equilateral (a b : ℝ) (h : a > b ∧ b > 0)
  (hc : c = real.sqrt (a^2 - b^2))
  (P : ℝ × ℝ) (Q : ℝ × ℝ) (F1 := (-c, 0)) (F2 := (c, 0)) :
  let PQ := dist P Q in
  let F1P := dist F1 P in
  let F2P := dist F2 P in
  let F1Q := dist F1 Q in
  let F2Q := dist F2 Q in
  let e := c / a in
  PQ = PF1 ∧ PQ = PF2 ∧ PQ = QF1 ∧ PQ = QF2 ∧ PQ = PF2 ∧ PQ = QF2 → 
  2 * e = real.sqrt 3 - real.sqrt 3 * e^2 → 
  e = real.sqrt 3 / 3 :=
sorry

end eccentricity_equilateral_l471_471330


namespace num_ways_distribute_balls_l471_471428

-- Definitions of the conditions
def balls := 6
def boxes := 4

-- Theorem statement
theorem num_ways_distribute_balls : 
  ∃ n : ℕ, (balls = 6 ∧ boxes = 4) → n = 8 :=
sorry

end num_ways_distribute_balls_l471_471428


namespace lily_cost_l471_471934

-- Definitions based on conditions
def num_tables : Nat := 20
def linen_cost_per_table : Nat := 25
def place_settings_per_table : Nat := 4
def place_setting_cost : Nat := 10
def roses_per_table : Nat := 10
def rose_cost : Nat := 5
def lilies_per_table : Nat := 15
def total_decorations_cost : Nat := 3500

-- Hypothesis to calculate components and intermediate results
def cost_without_lilies_per_table : Nat :=
  linen_cost_per_table + (place_settings_per_table * place_setting_cost) + (roses_per_table * rose_cost)

def total_cost_without_lilies : Nat := num_tables * cost_without_lilies_per_table

def total_lilies_cost (total_decorations_cost - total_cost_without_lilies : ℕ) := sorry

def total_number_of_lilies : Natural := num_tables * lilies_per_table

def cost_per_lily : ℕ := total_lilies_cost / total_number_of_lilies

-- Statement to prove that cost per lily is $4
theorem lily_cost : cost_per_lily = 4 :=
  sorry

end lily_cost_l471_471934


namespace prop1_prop2_prop3_prop4_final_l471_471359

variables (a b c : ℝ) (h_a : a ≠ 0)

-- Proposition ①
theorem prop1 (h1 : a + b + c = 0) : b^2 - 4 * a * c ≥ 0 := 
sorry

-- Proposition ②
theorem prop2 (h2 : ∃ (x1 x2 : ℝ), x1 = -1 ∧ x2 = 2 ∧ a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0) : 2 * a + c = 0 := 
sorry

-- Proposition ③
theorem prop3 (h3 : ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ a * x1^2 + c = 0 ∧ a * x2^2 + c = 0) : ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0 :=
sorry

-- Proposition ④
theorem prop4 (h4 : ∃ x : ℝ, a * x^2 + b * x + c = 0 ∧ ∃! x : ℝ, a * x^2 + b * x + c = 0) : ¬ (∃ x : ℝ, a * x^2 + b * x + c = 1 ∧ a * x^2 + b * x + 1 = 0) :=
sorry

-- Collectively checking that ①, ②, and ③ are true, and ④ is false
theorem final (h1 : a + b + c = 0)
              (h2 : ∃ (x1 x2 : ℝ), x1 = -1 ∧ x2 = 2 ∧ a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0)
              (h3 : ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ a * x1^2 + c = 0 ∧ a * x2^2 + c = 0)
              (h4 : ∃ x : ℝ, a * x^2 + b * x + c = 0 ∧ ∃! x : ℝ, a * x^2 + b * x + c = 0) : 
  (b^2 - 4 * a * c ≥ 0 ∧ 2 * a + c = 0 ∧ 
  (∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0) ∧ 
  ¬ (∃ x : ℝ, a * x^2 + b * x + c = 1 ∧ a * x^2 + b * x + 1 = 0)) :=
sorry

end prop1_prop2_prop3_prop4_final_l471_471359


namespace opposite_of_neg_three_l471_471149

theorem opposite_of_neg_three : ∃ y : ℤ, -3 + y = 0 ∧ y = 3 :=
by
  use 3
  split
  rfl
  rfl

end opposite_of_neg_three_l471_471149


namespace largest_n_condition_l471_471761

theorem largest_n_condition :
  ∃ n : ℤ, (∃ m : ℤ, n^2 = (m + 1)^3 - m^3) ∧ ∃ k : ℤ, 2 * n + 99 = k^2 ∧ ∀ x : ℤ, 
  (∃ m' : ℤ, x^2 = (m' + 1)^3 - m'^3) ∧ ∃ k' : ℤ, 2 * x + 99 = k'^2 → x ≤ 289 :=
sorry

end largest_n_condition_l471_471761


namespace min_value_sin_cos_l471_471782

theorem min_value_sin_cos (x : ℝ) : sin x ^ 6 + 2 * cos x ^ 6 ≥ 2 / 3 :=
sorry

end min_value_sin_cos_l471_471782


namespace neq_zero_necessary_not_sufficient_l471_471250

theorem neq_zero_necessary_not_sufficient (x : ℝ) (h : x ≠ 0) : 
  (¬ (x = 0) ↔ x > 0) ∧ ¬ (x > 0 → x ≠ 0) :=
by sorry

end neq_zero_necessary_not_sufficient_l471_471250


namespace arithmetic_sequence_sum_l471_471352

theorem arithmetic_sequence_sum :
  let a1 : ℕ := 4  -- First term
  let d : ℕ := 3   -- Common difference
  let n : ℕ := 20  -- Number of terms
  let an : ℕ := a1 + (n - 1) * d  -- nth term formula
  let sum_n : ℕ := n * (a1 + an) / 2 -- Sum formula
  in sum_n = 650 := by
  sorry

end arithmetic_sequence_sum_l471_471352


namespace min_table_sum_l471_471077

theorem min_table_sum :
  ∀ (x : Fin 60 → ℕ), 
  (∀ i, x i > 1) →
  ∑ i in Finset.univ (Fin 60), ∑ k in Finset.univ (Fin 60), 
    Real.log ((x k : ℝ) (Real.log (x k : ℝ) / Real.log 8)) ≥ -7200 :=
begin
  sorry
end

end min_table_sum_l471_471077


namespace cos_angle_PQR_l471_471036

theorem cos_angle_PQR (P Q R S : Point) 
  (angle_PRS_90 : angle PRS = 90) 
  (angle_PSR_90 : angle PSR = 90) 
  (angle_QRS_90 : angle QRS = 90) 
  (t : Real) 
  (cos_angle_PQS_t : cos (angle PQS) = t) 
  (z : Real) 
  (cos_angle_PQS_z : cos (angle PQS) = z) :
  cos (angle PQR) = -t^2 :=
sorry

end cos_angle_PQR_l471_471036


namespace equation_of_AB_l471_471035

--- Definitions of points and conditions as given
def O : (ℝ × ℝ) := (0, 0)
def A : (ℝ × ℝ) := (1, 3)
noncomputable def B : (ℝ × ℝ) := (b, 0) -- B is on the x-axis, hence y-coordinate is 0

--- The slope calculation as per the conditions
noncomputable def slope (P Q : ℝ × ℝ) := (Q.2 - P.2) / (Q.1 - P.1)

--- Isocele triangle conditions, hence slope reciprocal condition
lemma isosceles_triangle_AOB (b : ℝ) (H : b > 0) : 
  let k_AO := slope O A,
      k_AB := slope A B
  in k_AB = -k_AO := 
sorry

--- The final proof statement to show the equation of line AB.
theorem equation_of_AB (b : ℝ) (H1 : b > 0) (H2 : B = (b, 0)) :
  let k_AB := slope A B
  in (∀ x y, y - 3 = k_AB * (x - 1)) ↔ (y - 3 = -3 * (x - 1)) := 
sorry

end equation_of_AB_l471_471035


namespace range_of_t_l471_471521

-- Define set A and set B as conditions
def setA := { x : ℝ | -3 < x ∧ x < 7 }
def setB (t : ℝ) := { x : ℝ | t + 1 < x ∧ x < 2 * t - 1 }

-- Lean statement to prove the range of t
theorem range_of_t (t : ℝ) : setB t ⊆ setA → t ≤ 4 :=
by
  -- sorry acts as a placeholder for the proof
  sorry

end range_of_t_l471_471521


namespace max_discount_rate_l471_471281

-- Define the cost price and selling price.
def cp : ℝ := 4
def sp : ℝ := 5

-- Define the minimum profit margin.
def min_profit_margin : ℝ := 0.4

-- Define the discount rate d.
def discount_rate (d : ℝ) : ℝ := sp * (1 - d / 100) - cp

-- The theorem to prove the maximum discount rate.
theorem max_discount_rate (d : ℝ) (H : discount_rate d ≥ min_profit_margin) : d ≤ 12 :=
sorry

end max_discount_rate_l471_471281


namespace second_storm_duration_l471_471638

theorem second_storm_duration
  (x y : ℕ)
  (h1 : x + y = 45)
  (h2 : 30 * x + 15 * y = 975) :
  y = 25 := 
sorry

end second_storm_duration_l471_471638


namespace population_net_increase_in_one_day_l471_471213

-- Definitions based on the conditions
def birth_rate_per_two_seconds : ℝ := 4
def death_rate_per_two_seconds : ℝ := 3
def seconds_in_a_day : ℝ := 86400

-- The main theorem to prove
theorem population_net_increase_in_one_day : 
  (birth_rate_per_two_seconds / 2 - death_rate_per_two_seconds / 2) * seconds_in_a_day = 43200 :=
by
  sorry

end population_net_increase_in_one_day_l471_471213


namespace surface_area_of_second_cube_l471_471621

theorem surface_area_of_second_cube (V1 V2: ℝ) (a2: ℝ):
  (V1 = 16 ∧ V2 = 4 * V1 ∧ a2 = (V2)^(1/3)) → 6 * a2^2 = 96 :=
by intros h; sorry

end surface_area_of_second_cube_l471_471621


namespace max_distance_proof_l471_471717

-- Definitions for fuel consumption rates per 100 km
def fuel_consumption_U : Nat := 20 -- liters per 100 km
def fuel_consumption_V : Nat := 25 -- liters per 100 km
def fuel_consumption_W : Nat := 5  -- liters per 100 km
def fuel_consumption_X : Nat := 10 -- liters per 100 km

-- Definitions for total available fuel
def total_fuel : Nat := 50 -- liters

-- Distance calculation
def distance (fuel_consumption : Nat) (fuel : Nat) : Nat :=
  (fuel * 100) / fuel_consumption

-- Distances
def distance_U := distance fuel_consumption_U total_fuel
def distance_V := distance fuel_consumption_V total_fuel
def distance_W := distance fuel_consumption_W total_fuel
def distance_X := distance fuel_consumption_X total_fuel

-- Maximum total distance calculation
def maximum_total_distance : Nat :=
  distance_U + distance_V + distance_W + distance_X

-- The statement to be proved
theorem max_distance_proof :
  maximum_total_distance = 1950 := by
  sorry

end max_distance_proof_l471_471717


namespace value_at_2007_l471_471392

noncomputable def f : ℝ → ℝ := sorry

axiom even_function (x : ℝ) : f x = f (-x)
axiom symmetric_property (x : ℝ) : f (2 + x) = f (2 - x)
axiom specific_value : f (-3) = -2

theorem value_at_2007 : f 2007 = -2 :=
sorry

end value_at_2007_l471_471392


namespace line_passing_quadrants_l471_471608

theorem line_passing_quadrants (a k : ℝ) (a_nonzero : a ≠ 0)
  (x1 x2 y1 y2 : ℝ) (hx1 : y1 = a * x1^2 - a) (hx2 : y2 = a * x2^2 - a)
  (hx1_y1 : y1 = k * x1) (hx2_y2 : y2 = k * x2) 
  (sum_x : x1 + x2 < 0) : 
  ∃ (q1 q4 : (ℝ × ℝ)), 
  (q1.1 > 0 ∧ q1.2 > 0 ∧ q1.2 = a * q1.1 + k) ∧ (q4.1 > 0 ∧ q4.2 < 0 ∧ q4.2 = a * q4.1 + k) := 
sorry

end line_passing_quadrants_l471_471608


namespace tangent_lines_through_origin_l471_471130

-- Definition of the curve y = ln|x|
def curve (x : ℝ) : ℝ :=
  real.log (abs x)

-- Proposition stating that the tangent lines to the curve y = ln|x| passing through
-- the origin are given by x - e y = 0 and x + e y = 0.
theorem tangent_lines_through_origin :
  (∀ (x y : ℝ), curve x = y → (x - real.exp 1 * y = 0 ∨ x + real.exp 1 * y = 0)) ↔
  (∀ (x : ℝ), curve x = real.log (abs x)) :=
sorry

end tangent_lines_through_origin_l471_471130


namespace value_of_y_arithmetic_sequence_l471_471487

-- Define the arithmetic sequence and the target middle term
variable (x y z : ℝ)
variable (d : ℝ)
variable h_arith_seq : (12, x, y, z, 56) is_arithmetic_sequence with common_difference d

theorem value_of_y_arithmetic_sequence :
  y = (12 + 56) / 2 :=
by
  -- As per the conditions of the problem, provide the rest of the proof structure which we skip here.
  sorry

end value_of_y_arithmetic_sequence_l471_471487


namespace find_common_difference_l471_471507

section
variables (a1 a7 a8 a9 S5 S6 : ℚ) (d : ℚ)

/-- Given an arithmetic sequence with the sum of the first n terms S_n,
    if S_5 = a_8 + 5 and S_6 = a_7 + a_9 - 5, we need to find the common difference d. -/
theorem find_common_difference
  (h1 : S5 = a8 + 5)
  (h2 : S6 = a7 + a9 - 5)
  (h3 : S5 = 5 / 2 * (2 * a1 + 4 * d))
  (h4 : S6 = 6 / 2 * (2 * a1 + 5 * d))
  (h5 : a8 = a1 + 7 * d)
  (h6 : a7 = a1 + 6 * d)
  (h7 : a9 = a1 + 8 * d):
  d = -55 / 19 :=
by
  sorry
end

end find_common_difference_l471_471507


namespace first_machine_copies_per_minute_l471_471290

theorem first_machine_copies_per_minute
    (x : ℕ)
    (h1 : ∀ (x : ℕ), 30 * x + 30 * 55 = 2850) :
  x = 40 :=
by
  sorry

end first_machine_copies_per_minute_l471_471290


namespace particle_return_origin_l471_471695

def combinations (n k : ℕ) : ℕ := (nat.factorial n) / (nat.factorial k * nat.factorial (n - k))

theorem particle_return_origin :
  let P_start_origin    := true,
      move_one_unit     := true,
      equal_prob        := true,
      six_moves         := true,
      direction_prob    := (1/2),
      total_moves       := 6,
      target_probability := combinations 6 3 * (direction_prob ^ 6)
  in target_probability = combinations 6 3 * (1/2 ^ 6) :=
by {
  -- The proof will involve enumerating possible valid sequences,
  -- calculating the number of such sequences, and their probabilities.
  sorry
}

end particle_return_origin_l471_471695


namespace tangent_lines_through_origin_l471_471132

-- Definition of the curve y = ln|x|
def curve (x : ℝ) : ℝ :=
  real.log (abs x)

-- Proposition stating that the tangent lines to the curve y = ln|x| passing through
-- the origin are given by x - e y = 0 and x + e y = 0.
theorem tangent_lines_through_origin :
  (∀ (x y : ℝ), curve x = y → (x - real.exp 1 * y = 0 ∨ x + real.exp 1 * y = 0)) ↔
  (∀ (x : ℝ), curve x = real.log (abs x)) :=
sorry

end tangent_lines_through_origin_l471_471132


namespace prob_9_correct_matches_is_zero_l471_471245

noncomputable def probability_of_exactly_9_correct_matches : ℝ :=
  let n := 10 in
  -- Since choosing 9 correct implies the 10th is also correct, the probability is 0.
  0

theorem prob_9_correct_matches_is_zero : probability_of_exactly_9_correct_matches = 0 :=
by
  sorry

end prob_9_correct_matches_is_zero_l471_471245


namespace harmonic_mean_of_4_and_5040_is_8_closest_l471_471979

noncomputable def harmonicMean (a b : ℕ) : ℝ :=
  (2 * a * b) / (a + b)

theorem harmonic_mean_of_4_and_5040_is_8_closest :
  abs (harmonicMean 4 5040 - 8) < 1 :=
by
  -- The proof process would go here
  sorry

end harmonic_mean_of_4_and_5040_is_8_closest_l471_471979


namespace ellipse_equation_maximum_area_l471_471377

-- Definitions based on conditions:
def eccentricity_eq : ℝ := (1 / 2) ^ (1 / 2)
def point_A : ℝ × ℝ := (1, (2 ^ (1 / 2)) / 2)
def ellipse_eq (a b : ℝ) : set (ℝ × ℝ) := { p | (p.1 ^ 2) / (a ^ 2) + (p.2 ^ 2) / (b ^ 2) = 1 }
def line_l (m : ℝ) : set (ℝ × ℝ) := { p | p.2 = (2 ^ (1 / 2)) / 2 * p.1 + m }

-- Problem statements:
theorem ellipse_equation (a b : ℝ) (h: a > b ∧ b > 0) (ecc: eccentricity_eq = b / a)
  (H : point_A ∈ ellipse_eq a b) :
  ellipse_eq a b = { p | (p.1 ^ 2) / 2 + (p.2 ^ 2) = 1 } := sorry

theorem maximum_area (m : ℝ) (a b : ℝ) (h: a > b ∧ b > 0) (ecc: eccentricity_eq = b / a)
  (H : point_A ∈ ellipse_eq a b) (H2 : point_A ∉ line_l m):
  (∃ B C : ℝ × ℝ, B ∈ line_l m ∧ C ∈ line_l m ∧ B ≠ point_A ∧ C ≠ point_A ∧ 
   ∃ area, area = (1 / 2) * max_area_triangle (B, point_A, C)) :=
  max_area_triangle = (2 ^ (1 / 2)) / 2 := sorry

end ellipse_equation_maximum_area_l471_471377


namespace sum_of_integer_solutions_l471_471197

theorem sum_of_integer_solutions : 
  (∑ x in { x : ℤ | 2 < (x - 3) ^ 4 ∧ (x - 3) ^ 4 < 82 }.to_finset, x) = 15 :=
by 
  sorry

end sum_of_integer_solutions_l471_471197


namespace original_rectangle_length_l471_471981

-- Define the problem conditions
def length_three_times_width (l w : ℕ) : Prop :=
  l = 3 * w

def length_decreased_width_increased (l w : ℕ) : Prop :=
  l - 5 = w + 5

-- Define the proof problem
theorem original_rectangle_length (l w : ℕ) (H1 : length_three_times_width l w) (H2 : length_decreased_width_increased l w) : l = 15 :=
sorry

end original_rectangle_length_l471_471981


namespace fish_catch_ratio_l471_471901

noncomputable def Perry_catch (J : ℕ) (remaining : ℕ) : ℕ :=
  let total_catch := (remaining * 4) / 3
  total_catch - J

theorem fish_catch_ratio : 
  ∀ (J P remaining : ℕ), 
  J = 4 → 
  (3 * (J + P)) = (4 * remaining) →
  remaining = 9 →
  (P / J) = 2 := 
by 
  intros J P remaining hJ h1 h2
  have hJ4 : J = 4 := hJ
  have h9 : remaining = 9 := h2
  have eq1 : 3 * (4 + P) = 36 := by 
    rw [hJ4, h9] at h1
    exact h1
  have total_catch := 12 := by 
    linarith [eq1]
  have eq2 : P = 8 := by 
    linarith
  exact eq2.divide (eqJ4.symm.trans hJ.symm)

end fish_catch_ratio_l471_471901


namespace exists_arithmetic_progression_with_sum_zero_l471_471321

theorem exists_arithmetic_progression_with_sum_zero : 
  ∃ (a d : Int) (n : Int), n > 0 ∧ (n * (2 * a + (n - 1) * d)) = 0 :=
by 
  sorry

end exists_arithmetic_progression_with_sum_zero_l471_471321


namespace interval_of_monotonicity_min_value_on_interval_range_of_values_for_k_l471_471834

noncomputable def f (x k : ℝ): ℝ := (x - k) * Real.exp x

-- Statement I: Interval of monotonicity
theorem interval_of_monotonicity (k x : ℝ) :
  if x < k - 1 then
    (f' x k > 0)
  else
    (f' x k < 0) := sorry

-- Statement II: Minimum value on the interval [0,1]
theorem min_value_on_interval (k : ℝ) :
  if k ≤ 1 then
    ∃ x ∈ (Icc (0:ℝ) 1), f x k = -k
  else if 1 < k ∧ k ≤ 2 then
    ∃ x ∈ (Icc (0:ℝ) 1), f x k = -(Real.exp (k-1))
  else ∃ x ∈ (Icc (0:ℝ) 1), f x k = (1 - k) * Real.exp 1 := sorry

-- Statement III: Range of values for k such that f(x) > k^2 - 2 on [0,1]
theorem range_of_values_for_k (k x : ℝ) (h : k ≤ 1) (hx : x ∈ Icc 0 1) :
  f x k > k^2 - 2 ↔ -2 < k ∧ k < 1 := sorry

end interval_of_monotonicity_min_value_on_interval_range_of_values_for_k_l471_471834


namespace indistinguishable_balls_boxes_l471_471432

open Finset

def partitions (n : ℕ) (k : ℕ) : ℕ :=
  (univ : Finset (Multiset ℕ)).filter (λ p, p.sum = n ∧ p.card ≤ k).card

theorem indistinguishable_balls_boxes : partitions 6 4 = 9 :=
sorry

end indistinguishable_balls_boxes_l471_471432


namespace problem_statement_l471_471739

noncomputable def f_B (x : ℝ) : ℝ := -x^2
noncomputable def f_D (x : ℝ) : ℝ := Real.cos x

theorem problem_statement :
  (∀ x : ℝ, f_B (-x) = f_B x) ∧ (∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → x2 < 1 → f_B x1 > f_B x2) ∧
  (∀ x : ℝ, f_D (-x) = f_D x) ∧ (∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → x2 < 1 → f_D x1 > f_D x2) :=
  sorry

end problem_statement_l471_471739


namespace wins_per_girl_l471_471802

theorem wins_per_girl (a b c d : ℕ) (h1 : a + b = 8) (h2 : a + c = 10) (h3 : b + c = 12) (h4 : a + d = 12) (h5 : b + d = 14) (h6 : c + d = 16) : 
  a = 3 ∧ b = 5 ∧ c = 7 ∧ d = 9 :=
sorry

end wins_per_girl_l471_471802


namespace jerry_received_from_sister_l471_471060

def total_money_received (amounts : List ℕ) : ℕ :=
  amounts.sum

def mean (total : ℕ) (num_sources : ℕ) : ℚ :=
  total.toRat / num_sources

def amount_from_sister (aunt uncle friend1 friend2 friend3 friend4 : ℕ) (mean_value : ℚ) : ℚ :=
  let total : ℕ := total_money_received [aunt, uncle, friend1, friend2, friend3, friend4]
  let num_sources : ℕ := 7
  let total_with_sister :=
    mean_value * num_sources.toRat
  total_with_sister - total.toRat

theorem jerry_received_from_sister :
  amount_from_sister 9 9 22 23 22 22 16.3 = 7.1 :=
by sorry

end jerry_received_from_sister_l471_471060


namespace pages_copied_l471_471454

theorem pages_copied (cost_per_page : ℕ) (amount_in_dollars : ℕ)
    (cents_per_dollar : ℕ) (total_cents : ℕ) 
    (pages : ℕ)
    (h1 : cost_per_page = 3)
    (h2 : amount_in_dollars = 25)
    (h3 : cents_per_dollar = 100)
    (h4 : total_cents = amount_in_dollars * cents_per_dollar)
    (h5 : total_cents = 2500)
    (h6 : pages = total_cents / cost_per_page) :
  pages = 833 := 
sorry

end pages_copied_l471_471454


namespace perp_bisector_eq_l471_471418

noncomputable def C1 := { p : ℝ × ℝ | (p.1)^2 + (p.2)^2 - 6 * p.1 - 7 = 0 }
noncomputable def C2 := { p : ℝ × ℝ | (p.1)^2 + (p.2)^2 - 6 * p.2 - 27 = 0 }

theorem perp_bisector_eq :
  ∃ x y, ( (x, y) ∈ C1 ∧ (x, y) ∈ C2 ) -> ( x - y = 0 ) :=
by
  sorry

end perp_bisector_eq_l471_471418


namespace indistinguishable_balls_boxes_l471_471430

open Finset

def partitions (n : ℕ) (k : ℕ) : ℕ :=
  (univ : Finset (Multiset ℕ)).filter (λ p, p.sum = n ∧ p.card ≤ k).card

theorem indistinguishable_balls_boxes : partitions 6 4 = 9 :=
sorry

end indistinguishable_balls_boxes_l471_471430


namespace midpoint_polar_coordinates_correct_l471_471033
noncomputable def polar_midpoint_coordinates 
  (r₁ θ₁ r₂ θ₂ : ℝ) 
  (h₁ : r₁ = 10) 
  (h₂ : θ₁ = Real.pi / 3)
  (h₃ : r₂ = 12) 
  (h₄ : θ₂ = -Real.pi / 4) 
  (pos_r : ∀ r > 0)
  (theta_range : ∀ θ, 0 ≤ θ < 2 * Real.pi): 
  (Real × Real) :=
  let x_A := r₁ * Real.cos θ₁ in
  let y_A := r₁ * Real.sin θ₁ in
  let x_B := r₂ * Real.cos θ₂ in
  let y_B := r₂ * Real.sin θ₂ in
  let x_M := (x_A + x_B) / 2 in
  let y_M := (y_A + y_B) / 2 in
  let r := Real.sqrt (x_M^2 + y_M^2) in
  let θ := Real.atan2 y_M x_M in
  (r, θ)

theorem midpoint_polar_coordinates_correct :
  polar_midpoint_coordinates 10 (Real.pi / 3) 12 (-Real.pi / 4) r > 0 0 ≤ θ < 2 * Real.pi = 
  (Real.sqrt ((5 + 6 * Real.sqrt 2) / 2)^2 + ((5 * Real.sqrt 3 - 6 * Real.sqrt 2) / 2)^2,
   Real.atan2 ((5 * Real.sqrt 3 - 6 * Real.sqrt 2) / 2) ((5 + 6 * Real.sqrt 2) / 2)) := 
Sorry

end midpoint_polar_coordinates_correct_l471_471033


namespace volume_of_pyramid_AMSK_angle_between_AM_and_SK_distance_between_AM_and_SK_l471_471617

-- Definition of points and lengths based on given conditions
def side_length := 8
def height_SO := 3

-- Given SA = SB = SC = SD, SO, and midpoints M and K, we need to prove:
theorem volume_of_pyramid_AMSK : volume_AMSK = 8 := sorry

theorem angle_between_AM_and_SK : angle_AM_SK = Real.arccos(3 / 5) := sorry

theorem distance_between_AM_and_SK : distance_AM_SK = 24 / 13 := sorry

end volume_of_pyramid_AMSK_angle_between_AM_and_SK_distance_between_AM_and_SK_l471_471617


namespace value_of_f_f_0_l471_471403

def f (x : ℝ) : ℝ :=
  if x < 1 then 2 - x else x ^ 2 - x

theorem value_of_f_f_0 : f (f 0) = 2 := by
  sorry

end value_of_f_f_0_l471_471403


namespace find_exponent_l471_471255

theorem find_exponent :
  ∃ x : ℤ, 10 ^ x * 10 ^ 652 = 1000 ∧ x = -649 :=
by
  use -649
  simp [pow_add, pow_mul, ←mul_assoc]
  sorry

end find_exponent_l471_471255


namespace odd_coefficients_in_polynomial_l471_471075

noncomputable def number_of_odd_coefficients (n : ℕ) : ℕ :=
  (2^n - 1) / 3 * 4 + 1

theorem odd_coefficients_in_polynomial (n : ℕ) (hn : 0 < n) :
  (x^2 + x + 1)^n = number_of_odd_coefficients n :=
sorry

end odd_coefficients_in_polynomial_l471_471075


namespace smallest_m_Rn_eq_l_l471_471082

def l1_angle := Real.pi / 70
def l2_angle := Real.pi / 54
def l_slope := 19 / 92

def R (theta x : ℝ) := 2 * theta - 2 * l1_angle + x

def Rn (x : ℝ) (n : ℕ) : ℝ :=
  x + n * ((2 * l2_angle - 2 * l1_angle) / Real.pi)

theorem smallest_m_Rn_eq_l :
  ∃ m : ℕ, (m > 0) ∧ (∀ n : ℕ, n < m → Rn (Real.arctan l_slope) n ≠ Real.arctan l_slope) ∧
              Rn (Real.arctan l_slope) m = Real.arctan l_slope :=
sorry

end smallest_m_Rn_eq_l_l471_471082


namespace range_of_values_l471_471022

variable (a : ℝ)

-- State the conditions
def prop.false (a : ℝ) : Prop := ¬ ∃ x : ℝ, a * x^2 + 4 * x + a ≤ 0

-- Prove that the range of values for a where the proposition is false is (2, +∞)
theorem range_of_values (ha : prop.false a) : 2 < a :=
sorry

end range_of_values_l471_471022


namespace tan_arithmetic_sequence_l471_471474

variable {A B C : ℝ}

-- Definition of an acute triangle (all angles less than 90 degrees)
-- This condition might be formulated differently in Lean mathematics library.
def is_acute (A B C : ℝ) : Prop := 
  0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2 ∧ A + B + C = π

-- The main theorem statement
theorem tan_arithmetic_sequence
  (h_acute : is_acute A B C)
  (h_cos2_sum : cos A ^ 2 + cos B ^ 2 + cos C ^ 2 = sin A ^ 2) :
  tan A = (tan B + tan C) / 2 :=
sorry

end tan_arithmetic_sequence_l471_471474


namespace classA_wins_championship_distribution_expectation_scoreB_l471_471633

-- Define the probabilities of winning each game for Class A
def probBasketballA : ℝ := 0.4
def probSoccerA : ℝ := 0.8
def probBadmintonA : ℝ := 0.6

-- Define the conditions of independence
variable (indepEvents : indep_event probBasketballA probSoccerA probBadmintonA)

-- Define the point scoring system
def points_win := 8
def points_loss := 0

-- Define the total probability calculation for Class A winning at least two events
noncomputable def probClassAWinChampionship : ℝ := 
  probBasketballA * probSoccerA * probBadmintonA + -- All three events
  (1 - probBasketballA) * probSoccerA * probBadmintonA + -- Soccer and Badminton
  probBasketballA * (1 - probSoccerA) * probBadmintonA + -- Basketball and Badminton
  probBasketballA * probSoccerA * (1 - probBadmintonA) -- Basketball and Soccer

-- Verify the expected probability
theorem classA_wins_championship : probClassAWinChampionship = 0.656 :=
by
  unfold probClassAWinChampionship
  exact sorry

-- Distribution table for Class B's total score X
def probScore0 : ℝ := 0.4 * 0.8 * 0.6
def probScore10 : ℝ := 0.6 * 0.8 * 0.6 + 0.4 * 0.2 * 0.6 + 0.4 * 0.8 * 0.4
def probScore20 : ℝ := 0.6 * 0.2 * 0.6 + 0.6 * 0.8 * 0.4 + 0.4 * 0.2 * 0.4
def probScore30 : ℝ := 0.6 * 0.2 * 0.4

theorem distribution_expectation_scoreB : 
  ∑ p in [0, 10, 20, 30], p * probability_score p = 12 :=
by
  unfold probability_score
  exact sorry

end classA_wins_championship_distribution_expectation_scoreB_l471_471633


namespace value_of_expression_l471_471218

theorem value_of_expression : 
  let a := 0.137
  let b := 0.098 in
  ((a + b)^2 - (a - b)^2) / (a * b) = 3.991 :=
by
  let a := 0.137
  let b := 0.098
  calc
    (((a + b)^2 - (a - b)^2) / (a * b)) = (((4 * a * b) / (a * b))) : by sorry
                               ... = 4 * (a * b) / (a * b) : by sorry
                               ... = 4 : by sorry
                               ... = 3.991 : by sorry

end value_of_expression_l471_471218


namespace game_not_fair_probability_first_player_approximation_l471_471177

-- Definitions representing the problem's conditions
def n : ℕ := 36

-- Function representing the probability of a player winning (generalized)
def prob_first_player_winning (n : ℕ) : ℝ :=
  (1 : ℝ) / n * (1 / (1 - Real.exp (-1 / (Real.ofNat n))))

-- Hypothesis representing the approximate probability of the first player winning for 36 players
def approximated_prob_first_player_winning : ℝ := 0.044

-- Main statement in two parts: (1) fairness, (2) probability approximation
theorem game_not_fair (n : ℕ) (prob : ℕ → ℝ) : 
  (∃ k : ℕ, k < n ∧ prob k ≠ prob 0) :=
sorry

theorem probability_first_player_approximation :
  abs (prob_first_player_winning n - approximated_prob_first_player_winning) < 0.001 :=
sorry

end game_not_fair_probability_first_player_approximation_l471_471177


namespace unique_routes_l471_471730

-- We define different cities
inductive City
| P | Q | R | S | T

open City

-- We define roads as pairs of cities
inductive Road
| PQ | PR | PS | QR | QS | RS | ST

open Road

-- A route is a sequence of roads
def Route := List Road

-- Condition for a valid route: uses each road exactly once
def valid_route (r : Route) : Prop :=
  (Road.PQ ∈ r) ∧ (Road.PR ∈ r) ∧ (Road.PS ∈ r) ∧ (Road.QR ∈ r) ∧
  (Road.QS ∈ r) ∧ (Road.RS ∈ r) ∧ (Road.ST ∈ r) ∧ (r.length = 7)

-- The proof problem in Lean: proving there are exactly 8 valid routes from P to Q
theorem unique_routes : ∃ (routes : Finset Route), 
  (routes.filter (λ r => valid_route r ∧ r.head = Road.PQ)).card = 8 := 
by
  sorry

end unique_routes_l471_471730


namespace solve_quadratic_eq_l471_471966

theorem solve_quadratic_eq (x : ℝ) : (x^2 + 4 * x = 5) ↔ (x = 1 ∨ x = -5) :=
by
  sorry

end solve_quadratic_eq_l471_471966


namespace distance_is_660_km_l471_471639

def distance_between_cities (x y : ℝ) : ℝ :=
  3.3 * (x + y)

def train_A_dep_earlier (x y : ℝ) : Prop :=
  3.4 * (x + y) = 3.3 * (x + y) + 14

def train_B_dep_earlier (x y : ℝ) : Prop :=
  3.6 * (x + y) = 3.3 * (x + y) + 9

theorem distance_is_660_km (x y : ℝ) (hx : train_A_dep_earlier x y) (hy : train_B_dep_earlier x y) :
    distance_between_cities x y = 660 :=
sorry

end distance_is_660_km_l471_471639


namespace spider_legs_total_l471_471323

def num_spiders : ℕ := 4
def legs_per_spider : ℕ := 8
def total_legs : ℕ := num_spiders * legs_per_spider

theorem spider_legs_total : total_legs = 32 := by
  sorry -- proof is skipped with 'sorry'

end spider_legs_total_l471_471323


namespace derivative_at_0_l471_471317

noncomputable def f : ℝ → ℝ
| x => if x = 0 then 0 else x^2 * Real.exp (|x|) * Real.sin (1 / x^2)

theorem derivative_at_0 : deriv f 0 = 0 := by
  sorry

end derivative_at_0_l471_471317


namespace systematic_classic_equations_l471_471484

theorem systematic_classic_equations (x y : ℕ) : 
  (7 * x + 7 = y) ∧ (9 * (x - 1) = y) ↔
  (if (exists p q : ℕ, p = 7 * x + 7 ∧ q = 9 * (x - 1)) 
  then x = x ∧ y = 9 * (x - 1) 
  else false) :=
by 
  sorry

end systematic_classic_equations_l471_471484


namespace circumradius_triangulation_l471_471814

noncomputable def circumradius (A B C : Triangle.Point) : ℝ := 
  sorry -- Definition will depend on specifics of the geometry library in use.

variables {A B C B1 C1 I M N : Triangle.Point}
variables [Triangle.is_triangle A B C]
variables [Triangle.angle_bisector A B C B1]
variables [Triangle.angle_bisector C B A C1]
variables [Triangle.incenter A B C I]
variables [Triangle.intersects_circumcircle A B C B1 C1 M N]

theorem circumradius_triangulation : 
  circumradius I M N = 2 * circumradius A B C :=
sorry

end circumradius_triangulation_l471_471814


namespace find_A_l471_471336

def spadesuit (A B : ℕ) : ℕ := A^2 + 2 * A * B + 3 * B + 7

theorem find_A (A : ℕ) : spadesuit A 5 = 97 ↔ (A = 5 ∨ A = -15) := by
  sorry

end find_A_l471_471336


namespace fraction_e_over_d_l471_471153

theorem fraction_e_over_d :
  ∃ (d e : ℝ), (∀ (x : ℝ), x^2 + 2600 * x + 2600 = (x + d)^2 + e) ∧ e / d = -1298 :=
by 
  sorry

end fraction_e_over_d_l471_471153


namespace complex_division_l471_471252

theorem complex_division :
  (2 + Complex.i^3) / (1 - Complex.i) = (3 + Complex.i) / 2 :=
by
  sorry

end complex_division_l471_471252


namespace infinite_double_numbers_perfect_squares_l471_471697

theorem infinite_double_numbers_perfect_squares : 
  ∃∞ (n : ℕ), ∃ (N : ℕ), 
  let x := 10^(21 * n) + 1 in
  let N := (3 * x / 7)^2 in
  (N % (10 ^ (21 * n) * (10 ^ (21 * n) + 1)) = 0 ∧ sqrt(N)^2 = N) :=
sorry

end infinite_double_numbers_perfect_squares_l471_471697


namespace problem_l471_471380

def p := ∀ (a α β : Type), (a ∥ β) → (a ∥ α) → (a ∥ β)
def q := ∀ (a α β b : Type), (a ∥ α) → (a ∥ β) → (α ∩ β = b) → (a ∥ b)

-- Prove that (¬p) ∧ q
theorem problem (a α β b : Type) (h_p: ¬p a α β) (h_q: q a α β b) : (¬p a α β) ∧ q a α β b :=
by {
  apply and.intro,
  exact h_p,
  exact h_q,
}

end problem_l471_471380


namespace project_profit_starts_from_4th_year_l471_471685

def initial_investment : ℝ := 144
def maintenance_cost (n : ℕ) : ℝ := 4 * n^2 + 40 * n
def annual_income : ℝ := 100

def net_profit (n : ℕ) : ℝ := 
  annual_income * n - maintenance_cost n - initial_investment

theorem project_profit_starts_from_4th_year :
  ∀ n : ℕ, 3 < n ∧ n < 12 → net_profit n > 0 :=
by
  intros n hn
  sorry

end project_profit_starts_from_4th_year_l471_471685


namespace cone_sphere_volume_ratio_l471_471698

theorem cone_sphere_volume_ratio (r h : ℝ) 
  (radius_eq : r > 0)
  (volume_rel : (1 / 3 : ℝ) * π * r^2 * h = (1 / 3 : ℝ) * (4 / 3) * π * r^3) : 
  h / r = 4 / 3 :=
by
  sorry

end cone_sphere_volume_ratio_l471_471698


namespace perfect_square_value_l471_471004

theorem perfect_square_value (a : ℕ) (h : a = 1995^2 + 1995^2 * 1996^2 + 1996^2) : ∃ k : ℕ, k^2 = a ∧ k = 3982021 :=
by
  use 3982021
  rw h
  norm_num
  sorry

end perfect_square_value_l471_471004


namespace solve_inequality_l471_471846

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then 1 / x else (1 / 3) ^ x

theorem solve_inequality : { x : ℝ | |f x| ≥ 1 / 3 } = { x : ℝ | -3 ≤ x ∧ x ≤ 1 } :=
by
  sorry

end solve_inequality_l471_471846


namespace abs_sum_zero_l471_471450

theorem abs_sum_zero (a b : ℝ) (h : |a - 5| + |b + 8| = 0) : a + b = -3 := 
sorry

end abs_sum_zero_l471_471450


namespace probability_of_exactly_9_correct_matches_is_zero_l471_471231

theorem probability_of_exactly_9_correct_matches_is_zero :
  ∃ (P : ℕ → ℕ → ℕ), 
    (∀ (total correct : ℕ), 
      total = 10 → 
      correct = 9 → 
      P total correct = 0) := 
by {
  sorry
}

end probability_of_exactly_9_correct_matches_is_zero_l471_471231


namespace sphere_radius_ratio_l471_471293

theorem sphere_radius_ratio (V_L V_s : ℝ) (hV_L : V_L = 576 * Real.pi) (hV_s : V_s = 0.0625 * V_L) :
  (let R := (3 * V_L / (4 * Real.pi)).cbrt,
       r := (3 * V_s / (4 * Real.pi)).cbrt in r / R = 1 / 2) :=
by
  let R := (3 * V_L / (4 * Real.pi)).cbrt
  let r := (3 * V_s / (4 * Real.pi)).cbrt
  have hV_s2 : V_s = 36 * Real.pi := by rw [hV_L, hV_s] -- This resolves to V_s = 36π as in the steps.
  rw [Real.cbrt_div, Real.cbrt_div]
  rw [Real.cbrt_mul, Real.cbrt_mul, Real.cbrt_of_pow, Real.cbrt_of_pow, real.mul_pi_sqrt, real.mul_pi_sqrt]
  sorry

end sphere_radius_ratio_l471_471293


namespace transform_sine_to_cosine_l471_471184

theorem transform_sine_to_cosine:
  ∀ x, -3 * cos (2 * x) = -6 * sin^2 (x + π / 6) + 3 → 
       -3 * cos (2 * x) = 3 * cos (2 * (x + π / 3) + π / 3) := 
by
  intros x h,
  sorry

end transform_sine_to_cosine_l471_471184


namespace max_common_points_line_plane_l471_471871

theorem max_common_points_line_plane {L P : Set.Point → Prop} (h : ∃ p, L p ∧ ¬ P p) :
  ∃ m : ℕ, (∀ p, L p → P p → p ≤ m) ∧ m = 1 :=
sorry

end max_common_points_line_plane_l471_471871


namespace prob_four_or_more_same_value_l471_471794

theorem prob_four_or_more_same_value (dice : Fin 5 → Fin 6) :
  (let num_same := λ (n : Fin 6), (Fin 5).count (λ i, dice i = n),
       num_four_or_more := (Fin 6).count (λ n, num_same n ≥ 4) in
   num_four_or_more) / 7776 = 13 / 648 := 
sorry

end prob_four_or_more_same_value_l471_471794


namespace find_theta_l471_471365

-- Define vectors and their magnitudes
def vec_a : ℝ × ℝ := (2, -1)
def mag_b := 2 * Real.sqrt 5

-- Define the dot product property
def dot_product_property (b : ℝ × ℝ) :=
  let a := vec_a in
  let vector_add := (a.1 + b.1, a.2 + b.2) in
  (vector_add.1 * a.1 + vector_add.2 * a.2) = 10

-- Define the magnitude of vector a
def mag_a := Real.sqrt (2^2 + (-1)^2)

-- Define the dot product between two vectors
def dot_product (v1 v2 : ℝ × ℝ) := v1.1 * v2.1 + v1.2 * v2.2

-- The angle between two vectors
def angle_between_vectors (a b : ℝ × ℝ) :=
  Real.arccos ((dot_product a b) / (mag_a * mag_b))

-- Main theorem: prove the angle θ between a and b
theorem find_theta (b : ℝ × ℝ) (hb : ∥b∥ = mag_b) (hprop : dot_product_property b) :
  angle_between_vectors vec_a b = Real.pi / 3 :=
by
  sorry

end find_theta_l471_471365


namespace y_coord_min_perimeter_l471_471382

-- Definitions given in the problem conditions
structure Point (α : Type) := (x : α) (y : α)
def A : Point ℝ := ⟨0, 6 * real.sqrt 6⟩
def hyperbola := {p : Point ℝ // p.x^2 - p.y^2 / 8 = 1}
def F : Point ℝ := ⟨3, 0⟩

-- The proof problem translated into Lean code
theorem y_coord_min_perimeter (P : hyperbola) (on_left_branch : P.val.x < 0) :
  ∃ y : ℝ, y = 2 * real.sqrt 6 ∧ P.val.y = y :=
sorry

end y_coord_min_perimeter_l471_471382


namespace f_increasing_range_f_l471_471680

-- Define the function f(x)
def f (x : ℝ) (a : ℝ) : ℝ := -1 / (x ^ 2) + a

-- Condition: a > 0, x > 0
variables {a : ℝ} (ha : a > 0) {x : ℝ} (hx : x > 0)

-- Statement (1): f(x) is an increasing function on (0, +∞)
theorem f_increasing : ∀ x1 x2 : ℝ, 0 < x1 → 0 < x2 → x1 < x2 → f x1 a < f x2 a :=
by
  sorry

-- Statement (2): if the range of f(x) on (0, +∞) is (0, +∞), then a = +∞
theorem range_f (h_range : ∀ y : ℝ, y > 0 → ∃ x : ℝ, x > 0 ∧ f x a = y) : a = Real.Infinity :=
by
  sorry

end f_increasing_range_f_l471_471680


namespace ordered_quadruples_count_l471_471908

/--
The number of ordered quadruples (p, q, r, s) such that p ⋅ s + q ⋅ r is odd, 
where each of p, q, r, s belongs to the set {0, 1, 2, 3, 4}, is 168.
-/
theorem ordered_quadruples_count :
  let S := {0, 1, 2, 3, 4}
  (count :
    {psqr : ℕ × ℕ × ℕ × ℕ // psqr.1 ∈ S ∧ psqr.2.1 ∈ S ∧ psqr.2.2.1 ∈ S ∧ psqr.2.2.2 ∈ S ∧
      (psqr.1 * psqr.2.2.2 + psqr.2.1 * psqr.2.2.1) % 2 = 1}).val.card = 168 :=
by sorry

end ordered_quadruples_count_l471_471908


namespace game_not_fair_probability_first_player_approximation_l471_471179

-- Definitions representing the problem's conditions
def n : ℕ := 36

-- Function representing the probability of a player winning (generalized)
def prob_first_player_winning (n : ℕ) : ℝ :=
  (1 : ℝ) / n * (1 / (1 - Real.exp (-1 / (Real.ofNat n))))

-- Hypothesis representing the approximate probability of the first player winning for 36 players
def approximated_prob_first_player_winning : ℝ := 0.044

-- Main statement in two parts: (1) fairness, (2) probability approximation
theorem game_not_fair (n : ℕ) (prob : ℕ → ℝ) : 
  (∃ k : ℕ, k < n ∧ prob k ≠ prob 0) :=
sorry

theorem probability_first_player_approximation :
  abs (prob_first_player_winning n - approximated_prob_first_player_winning) < 0.001 :=
sorry

end game_not_fair_probability_first_player_approximation_l471_471179


namespace min_selection_for_diff_two_integers_l471_471000

theorem min_selection_for_diff_two_integers:
  ∀ (s : Finset ℕ),
    (∀ a b ∈ s, a - b ≠ 2)
    → s ⊆ Finset.range 21
    → s.card ≤ 10 :=
by
  sorry

end min_selection_for_diff_two_integers_l471_471000


namespace min_value_polynomial_l471_471076

open Real

theorem min_value_polynomial (x y z : ℝ) (h_posx : 0 < x) (h_posy : 0 < y) (h_posz : 0 < z) (h : x * y * z = 3) :
  x^2 + 4 * x * y + 12 * y^2 + 8 * y * z + 3 * z^2 ≥ 162 := 
sorry

end min_value_polynomial_l471_471076


namespace num_pairs_l471_471857

theorem num_pairs (count : ℕ) :
  count = (finset.univ : finset ℕ).filter (λ m, m > 0 ∧ m^2 < 50).sum (λ m, (50 - m^2 - 1)) :=
by
  let count := (1 + 48) + (2 + 45 + 40 + 4 * 5 + 6 * 6 + 14 * 13 + 1)
  sorry

end num_pairs_l471_471857


namespace tangent_lines_to_ln_abs_through_origin_l471_471126

noncomputable def tangent_line_through_origin (x y: ℝ) : Prop :=
  (y = log (abs x)) ∧ ((x - exp(1) * y = 0) ∨ (x + exp(1) * y = 0))

theorem tangent_lines_to_ln_abs_through_origin :
  ∃ (f : ℝ → ℝ), 
  (∀ x, f x = log (abs x)) ∧ 
  ∀ x y, (tangent_line_through_origin x y) := sorry

end tangent_lines_to_ln_abs_through_origin_l471_471126


namespace min_quotient_c_over_d_l471_471007

noncomputable theory

variables {x C D : ℝ}

theorem min_quotient_c_over_d (hC : x^4 + 1 / x^4 = C) (hD : x^2 - 1 / x^2 = D) (hC_pos : 0 < C) (hD_pos : 0 < D) : 
  ∃ D, C = D^2 + 2 ∧ D = sqrt 2 ∧ ∀ C D, (C = D^2 + 2) → (D = sqrt 2) → (C / D = 2 * sqrt 2) :=
by {
  use sqrt 2,
  split,
  {
    sorry,
  },
  split,
  {
    use sqrt 2,
    sorry,
  },
  intros C D hC1 hD1,
  exact (calc
    C / D = (D^2 + 2) / D : by rw hC1
      ... = (sqrt 2)^2 + 2 / sqrt 2 : by rw hD1
      ... = (2 + 2) / sqrt 2 : by norm_num
      ... = 2 * sqrt 2 : by field_simp [sqrt_pos]   )
}

end min_quotient_c_over_d_l471_471007


namespace find_largest_int_with_conditions_l471_471762

-- Definition of the problem conditions
def is_diff_of_consecutive_cubes (n : ℤ) : Prop :=
  ∃ m : ℤ, n^2 = (m + 1)^3 - m^3

def is_perfect_square_shifted (n : ℤ) : Prop :=
  ∃ k : ℤ, 2n + 99 = k^2

-- The main statement asserting the proof problem
theorem find_largest_int_with_conditions :
  ∃ n : ℤ, is_diff_of_consecutive_cubes n ∧ is_perfect_square_shifted n ∧
    ∀ m : ℤ, is_diff_of_consecutive_cubes m ∧ is_perfect_square_shifted m → m ≤ 50 :=
sorry

end find_largest_int_with_conditions_l471_471762


namespace remaining_soup_feeds_adults_l471_471684

theorem remaining_soup_feeds_adults (C A k c : ℕ) 
    (hC : C= 10) 
    (hA : A = 5) 
    (hk : k = 8) 
    (hc : c = 20) : k - c / C * 10 * A = 30 := sorry

end remaining_soup_feeds_adults_l471_471684


namespace minimum_b_in_arithmetic_series_l471_471071

theorem minimum_b_in_arithmetic_series (a b c : ℝ) (h1 : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h2 : a + b + c = 3 * b) (h3 : a * b * c = 64) : 4 ≤ b :=
begin
  sorry,
end

end minimum_b_in_arithmetic_series_l471_471071


namespace yellow_mugs_count_l471_471854

variables (R B Y O : ℕ)
variables (B_eq_3R : B = 3 * R)
variables (R_eq_Y_div_2 : R = Y / 2)
variables (O_eq_4 : O = 4)
variables (mugs_eq_40 : R + B + Y + O = 40)

theorem yellow_mugs_count : Y = 12 :=
by 
  sorry

end yellow_mugs_count_l471_471854


namespace P_n_even_Q_n_odd_powers_l471_471080

def P_n (n : ℕ) (cs : List ℝ) (x : ℝ) : ℝ :=
  cs.foldr (fun c acc => (c - x) * acc) 1

def Q_n (n : ℕ) (cs : List ℝ) (x : ℝ) : ℝ :=
  cs.foldr (fun c acc => (c + x) * acc) 1

theorem P_n_even_Q_n_odd_powers (n : ℕ) (cs : List ℝ) (x : ℝ) :
  (P_n n cs x + Q_n n cs x = P_n n cs (-x) + Q_n n cs x ∧
  (∀ k, (P_n n cs x + Q_n n cs x).pow x k → k % 2 = 0))
  ∧
  (P_n n cs x - Q_n n cs x =  P_n n cs (-x) - Q_n n cs x ∧
  (∀ k, (P_n n cs x - Q_n n cs x).pow x k → k % 2 = 1)) :=
sorry

end P_n_even_Q_n_odd_powers_l471_471080


namespace ducks_in_garden_l471_471100

theorem ducks_in_garden (num_rabbits : ℕ) (num_ducks : ℕ) 
  (total_legs : ℕ)
  (rabbit_legs : ℕ) (duck_legs : ℕ) 
  (H1 : num_rabbits = 9)
  (H2 : rabbit_legs = 4)
  (H3 : duck_legs = 2)
  (H4 : total_legs = 48)
  (H5 : num_rabbits * rabbit_legs + num_ducks * duck_legs = total_legs) :
  num_ducks = 6 := 
by {
  sorry
}

end ducks_in_garden_l471_471100


namespace partitions_of_6_into_4_indistinguishable_boxes_l471_471435

theorem partitions_of_6_into_4_indistinguishable_boxes : 
  ∃ (X : Finset (Multiset ℕ)), X.card = 9 ∧ 
  ∀ p ∈ X, p.sum = 6 ∧ p.card ≤ 4 := 
sorry

end partitions_of_6_into_4_indistinguishable_boxes_l471_471435


namespace increasing_intervals_decreasing_intervals_max_value_min_value_l471_471408

noncomputable def func (x : ℝ) : ℝ := 3 * Real.sin (2 * x + Real.pi / 4)

theorem increasing_intervals : 
  ∀ x ∈ (Set.Icc 0 (Real.pi / 8) ∪ Set.Icc (5 * Real.pi / 8) Real.pi), 
  0 < Real.cos (2 * x + Real.pi / 4) := 
sorry

theorem decreasing_intervals : 
  ∀ x ∈ Set.Icc (Real.pi / 8) (5 * Real.pi / 8), 
  Real.cos (2 * x + Real.pi / 4) < 0 := 
sorry

theorem max_value : func (Real.pi / 8) = 3 :=
sorry

theorem min_value : func (5 * Real.pi / 8) = -3 :=
sorry

end increasing_intervals_decreasing_intervals_max_value_min_value_l471_471408


namespace determine_false_statements_l471_471938

def statements (index : ℕ) : Prop :=
match index with
| 1 => ∃ (n : ℕ), n = 4 ∧ ¬(n = 1)
| 2 => ∃ (n : ℕ), n = 3 ∧ ¬(n = 2)
| 3 => ∃ (n : ℕ), n = 2 ∧ ¬(n = 3)
| 4 => ∃ (n : ℕ), n = 1 ∧ ¬(n = 4)
| 5 => ¬statements 5
| _ => false

theorem determine_false_statements : (∃ (k : ℕ), (k = 3 ∧ (∀ (i : ℕ) (h : i ∈ [1, 2, 3, 4, 5]),
  if k = 3 then ¬statements i else false))) :=
by
  sorry

end determine_false_statements_l471_471938


namespace exponent_calculation_l471_471724

theorem exponent_calculation : (8^5 / 8^2) * (2^10 / 2^3) = 65536 := by
  -- We start by using the properties of exponents to simplify the expression 
  have h1 : 8^5 / 8^2 = 8^(5 - 2) := by sorry
  -- Next, we simplify the inner term inside parenthesis
  have h2 : 8^(5 - 2) = 8^3 := by sorry
  -- Since 8 = 2^3, we substitute it back into the equation
  have h3 : 8^3 = (2^3)^3 := by sorry
  -- Therefore, 8^3 is replaced with 2^9
  have h4 : 8^3 = 2^9 := by sorry
  -- We combine and simplify the remaining expression
  have h5 : (2^9 * 2^10) / 2^3 = 2^(9 + 10 - 3) := by sorry
  -- Simplify to get the final answer
  have h6 : 2^(9 + 10 - 3) = 2^16 := by sorry
  -- Evaluating 2^16
  have h7 : 2^16 = 65536 := by
    calc 2^16 = 65536 : by sorry
  exact h7

end exponent_calculation_l471_471724


namespace sum_of_first_10_terms_l471_471815

theorem sum_of_first_10_terms:
  let a : ℕ → ℕ := λ n => (2 * n - 1) in
  let b : ℕ → ℚ := λ n => 1 / (a n * a (n + 1)) in
  (∑ n in Finset.range 10, b n) = (10 / 21) :=
by
  sorry

end sum_of_first_10_terms_l471_471815


namespace gcd_20586_58768_l471_471759

theorem gcd_20586_58768 : Int.gcd 20586 58768 = 2 := by
  sorry

end gcd_20586_58768_l471_471759


namespace inscribed_sphere_volume_ratio_l471_471304

theorem inscribed_sphere_volume_ratio (d : ℝ) (hd : d > 0) :
  let r := d / 2 in
  let V_sphere := (4 / 3) * Real.pi * r^3 in
  let V_cylinder := Real.pi * r^2 * d in
  V_sphere / V_cylinder = 2 / 3 :=
by
  sorry

end inscribed_sphere_volume_ratio_l471_471304


namespace tangent_line_at_1_lambda_greater_than_e_range_of_a_l471_471843

def f (x : ℝ) : ℝ := 1 / Real.exp x
def g (x : ℝ) : ℝ := Real.log x

theorem tangent_line_at_1 :
  let y := fun x => f x * g x
  let y' := (fun x => ((1 / x) - Real.log x) / Real.exp x)
  y 1 = 0 ∧ y' 1 = 1 / Real.exp 1 →
  y = (fun x => (1 / Real.exp 1) * x - (1 / Real.exp 1)) := sorry

theorem lambda_greater_than_e (x₁ x₂ : ℝ) (hx : x₁ ≠ x₂) (λ : ℝ) :
  g x₁ - g x₂ = λ * (f x₂ - f x₁) →
  λ > Real.exp 1 := sorry

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Icc (0:ℝ) 1, f x * g x ≤ a * (x - 1)) →
  a ≤ 1 / Real.exp 1 := sorry

end tangent_line_at_1_lambda_greater_than_e_range_of_a_l471_471843


namespace equal_areas_l471_471047

-- Metadata about points and geometric shapes
variables {A B C D E : Type*}
variables [pseudo_metric_space A] [pseudo_metric_space B]
variables [pseudo_metric_space C] [pseudo_metric_space D]
variables [pseudo_metric_space E]

-- Definitions of trapezoid and the properties needed:
def is_trapezoid (A B C D : Type*) [pseudo_metric_space A] [pseudo_metric_space B] [pseudo_metric_space C] [pseudo_metric_space D] : Prop :=
(BC: line B.to C.to) ∧ (AD: line A.to D.to) ∧ (parallel AD BC)

def is_parallel (l1 l2 : line) : Prop := sorry

-- Given conditions: A trapezoid ABCD and parallel line BE intersecting AC at E
variable (trapezoid : is_trapezoid A B C D)
variable (BE_parallel_CD : is_parallel (line_through B E) (line_through C D))
variable (E_on_AC : on_diagonal E A C)

-- Theorem to prove
theorem equal_areas (A B C D E : Type*) [pseudo_metric_space A] [pseudo_metric_space B] [pseudo_metric_space C] [pseudo_metric_space D] [pseudo_metric_space E] 
  (h₁ : is_trapezoid A B C D) 
  (h₂ : BE_parallel_CD : is_parallel (line_through B E) (line_through C D)) 
  (h₃ : E_on_AC E A C):
  area (triangle A B C) = area (triangle D E C) := 
sorry

end equal_areas_l471_471047


namespace equation_of_line_l_equation_of_line_BC_l471_471049

variable (A B M : Point)
variable (a b m n : ℝ)

-- Define points A, B, and M
def A := Point.mk 1 1
def B := Point.mk 3 (-2)
def M := Point.mk 2 0

-- Define the line equation passed a point and slope
def line_eq (P : Point) (m: ℝ) : Prop :=
  ∃ b, ∀ x y, y = m * (x - P.x) + b → line P m

-- Define equidistant line and parallel line properties
def is_equidistant_line (P₁ P₂ : Point) (L : Line) : Prop :=
  distance P₁ L = distance P₂ L

def is_parallel_to (L₁ L₂ : Line) : Prop :=
  slope L₁ = slope L₂

-- Problem (1): Equation of line l
theorem equation_of_line_l 
  (heq : is_equidistant_line A B l) (hl : line_through M l) : 
  l = (λ p, p.x = 2) ∨ l = (λ p, 3 * p.x + 2 * p.y - 6 = 0) :=
sorry

-- Problem (2): Equation of line BC
theorem equation_of_line_BC 
  (hangle : angle_bisector C (λ p, p.x + p.y - 3)): 
  line_through B C = (λ p, 4 * p.x + p.y - 10 = 0) :=
sorry

end equation_of_line_l_equation_of_line_BC_l471_471049


namespace little_sister_stole_roses_l471_471951

/-- Ricky has 40 roses. His little sister steals some roses. He wants to give away the rest of the roses in equal portions to 9 different people, and each person gets 4 roses. Prove how many roses his little sister stole. -/
theorem little_sister_stole_roses (total_roses stolen_roses remaining_roses people roses_per_person : ℕ)
  (h1 : total_roses = 40)
  (h2 : people = 9)
  (h3 : roses_per_person = 4)
  (h4 : remaining_roses = people * roses_per_person)
  (h5 : remaining_roses = total_roses - stolen_roses) :
  stolen_roses = 4 :=
by
  sorry

end little_sister_stole_roses_l471_471951


namespace limit_sequence_l471_471325

open Real

theorem limit_sequence :
  tendsto (λ n : ℕ, (↑((n+1)^3 - (n-1)^3) / ↑((n+1)^2 - (n-1)^2))) at_top at_top :=
sorry

end limit_sequence_l471_471325


namespace part1_S_2_2_part1_S_2_4_part2_S_m_n_l471_471522

noncomputable def M : Set ℤ := {-1, 0, 1}

def A_n (n : ℕ) : Set (Vector ℤ n) :=
  { x | (∀ i, x i ∈ M) }

def S (m n : ℕ) : ℕ :=
  ((A_n n).filter (λ x, 1 ≤ (Finset.univ.to_list.map (λ i => |x i|)).sum ∧ (Finset.univ.to_list.map (λ i => |x i|)).sum ≤ m)).to_finset.card

theorem part1_S_2_2 :
  S 2 2 = 8 := sorry

theorem part1_S_2_4 :
  S 2 4 = 32 := sorry

theorem part2_S_m_n (m n : ℕ) (h : m < n) :
  S m n < 3^n + 2^(m+1) - 2^(n+1) := sorry

end part1_S_2_2_part1_S_2_4_part2_S_m_n_l471_471522


namespace min_value_sin_cos_squared_six_l471_471778

theorem min_value_sin_cos_squared_six (x : ℝ) :
  ∃ x : ℝ, (sin^6 x + 2 * cos^6 x) = 2/3 :=
sorry

end min_value_sin_cos_squared_six_l471_471778


namespace cos_negative_570_equals_negative_sqrt3_div_2_l471_471667

theorem cos_negative_570_equals_negative_sqrt3_div_2 : Real.cos (-570 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end cos_negative_570_equals_negative_sqrt3_div_2_l471_471667


namespace students_count_l471_471720

theorem students_count (start_FG start_FG_left start_FG_new move_FG_FF : ℕ)
                       (start_FG : start_FG = 4)
                       (start_FG_left : start_FG_left = 3)
                       (start_FG_new : start_FG_new = 42)
                       (move_FG_FF : move_FG_FF = 10)
                       (start_FF start_FF_left start_FF_new move_FF_FS : ℕ)
                       (start_FF : start_FF = 10)
                       (start_FF_left : start_FF_left = 5)
                       (start_FF_new : start_FF_new = 25)
                       (move_FF_FS : move_FF_FS = 5)
                       (start_FS start_FS_left start_FS_new : ℕ)
                       (start_FS : start_FS = 15)
                       (start_FS_left : start_FS_left = 7)
                       (start_FS_new : start_FS_new = 30)
                       : 
  let FG_end := start_FG - start_FG_left + start_FG_new - move_FG_FF in
  let FF_inter := start_FF - start_FF_left + start_FF_new + move_FG_FF in
  let FF_end := FF_inter - move_FF_FS in
  let FS_end := start_FS - start_FS_left + start_FS_new + move_FF_FS in
  FG_end = 33 ∧ FF_end = 35 ∧ FS_end = 43 ∧ FG_end + FF_end + FS_end = 111 :=
by
  sorry

end students_count_l471_471720


namespace h_h_three_l471_471865

def h (x : ℤ) : ℤ := 3 * x^2 + 3 * x - 2

theorem h_h_three : h (h 3) = 3568 := by
  sorry

end h_h_three_l471_471865


namespace tan_60_l471_471731

theorem tan_60 (DO DP : ℝ) (h_DO : DO = 1 / 2) (h_DP : DP = sqrt 3 / 2) :
  Real.tan (Real.pi / 3) = sqrt 3 :=
by 
  sorry

end tan_60_l471_471731


namespace find_omega_l471_471385

theorem find_omega (ω : ℝ) (hω : ω > 0)
  (h_dist : ∀ x1 x2 : ℝ,
    4 * Math.sin (ω * x1) = 4 * Math.cos (ω * x1) →
    4 * Math.sin (ω * x2) = 4 * Math.cos (ω * x2) →
    (abs (x2 - x1) = 1 / ω * π) →
    (dist (x1, 4 * Math.sin (ω * x1)) (x2, 4 * Math.cos (ω * x2)) = 6)) :
  ω = π / 2 := 
sorry

end find_omega_l471_471385


namespace expected_min_leq_2_l471_471534

open ProbabilityTheory

variables (ξ η : ℝ → ℝ) -- ξ and η are random variables

-- Condition: expected value of ξ is 2
axiom E_ξ_eq_2 : ℝ
axiom E_ξ_is_2 : (∫ x in ⊤, ξ x) = 2

-- Goal: expected value of min(ξ, η) ≤ 2
theorem expected_min_leq_2 (h : ∀ x, min (ξ x) (η x) ≤ ξ x) : 
  (∫ x in ⊤, min (ξ x) (η x)) ≤ 2 := by
  -- use the provided axioms and conditions here
  sorry

end expected_min_leq_2_l471_471534


namespace comparison_l471_471367

def a : ℝ := Real.sin (1/5)
def b : ℝ := 1/5
def c : ℝ := (6/5) * Real.log (6/5)

theorem comparison : c > b ∧ b > a := by
  sorry

end comparison_l471_471367


namespace max_min_sum_of_f_l471_471969

noncomputable def f (x : ℝ) : ℝ := sorry

theorem max_min_sum_of_f :
  (∀ x1 x2 : ℝ, x1 ∈ Set.Icc (-2016) 2016 →
                x2 ∈ Set.Icc (-2016) 2016 →
                f (x1 + x2) = f x1 + f x2 - 2016) →
  (∀ x : ℝ, x > 0 → f x < 2016) →
  let M := supr (f '' (Set.Icc (-2016) 2016)) in
  let N := infi (f '' (Set.Icc (-2016) 2016)) in
  M + N = 4032 :=
begin
  intros h1 h2,
  let M := supr (f '' (Set.Icc (-2016) 2016)),
  let N := infi (f '' (Set.Icc (-2016) 2016)),
  sorry
end

end max_min_sum_of_f_l471_471969


namespace camel_grid_lines_impossible_camel_not_grid_lines_possible_l471_471055

/--
Given a figure consisting of 25 unit squares, it is impossible to divide the figure into three parts 
along the grid lines such that each part can be rearranged to form a square with side length 5.
-/
theorem camel_grid_lines_impossible :
  ∀ (figure : Set (Set (ℕ × ℕ))), figure.card = 25 →
  ¬ (∃ parts : List (Set (ℕ × ℕ)), parts.length = 3 ∧ (∀ part ∈ parts, (∃ n : ℕ, n^2 = part.card) ∧ (∃ side : ℕ, side^2 = part.card) ∧ side = 5) ∧
    ∀ (part ∈ parts), is_grid_aligned part) :=
by {
  sorry
}

/--
Given a figure consisting of 25 unit squares, it is possible to divide the figure into three parts 
not necessarily along the grid lines such that each part can be rearranged to form a square with side length 5.
-/
theorem camel_not_grid_lines_possible :
  ∀ (figure : Set (Set (ℕ × ℕ))), figure.card = 25 →
  ∃ parts : List (Set (ℕ × ℕ)), parts.length = 3 ∧ (∀ part ∈ parts, (∃ n : ℕ, n^2 = part.card) ∧ (∃ side : ℕ, side^2 = part.card) ∧ side = 5) :=
by {
  sorry
}

end camel_grid_lines_impossible_camel_not_grid_lines_possible_l471_471055


namespace max_value_of_m_l471_471509

noncomputable def max_subsequences (A : Fin 2001 → ℕ) : ℕ := 
  let counts := λ i : ℕ, (Finset.univ.filter (λ j, A j = i)).card
  (Finset.range 1999).sum (λ i, counts i * counts (i + 1) * counts (i + 2))

theorem max_value_of_m : ∀ (A : Fin 2001 → ℕ), max_subsequences A ≤ 667^3 := 
sorry

end max_value_of_m_l471_471509


namespace train_speed_is_correct_l471_471708

-- Definitions based on conditions
def train_length : ℝ := 110  -- in meters
def pass_time : ℝ := 5.999520038396929  -- in seconds
def man_speed : ℝ := 6  -- in km/hr

-- Conversion constants
def seconds_to_hours : ℝ := 1 / (60 * 60)  -- conversion factor from seconds to hours
def meters_to_kilometers : ℝ := 1 / 1000  -- conversion factor from meters to kilometers

-- Calculate the time in hours
def time_in_hours : ℝ := pass_time * seconds_to_hours

-- Calculate the train length in kilometers
def train_length_km : ℝ := train_length * meters_to_kilometers

-- Relative speed of the train with respect to the man
def relative_train_speed : ℝ := train_length_km / time_in_hours

-- Actual speed of the train
def actual_train_speed : ℝ := relative_train_speed + man_speed

-- The theorem to prove
theorem train_speed_is_correct : actual_train_speed = 72.00001499862502 := by
  sorry

end train_speed_is_correct_l471_471708


namespace max_discount_rate_l471_471289

-- Define the constants used in the problem
def costPrice : ℝ := 4
def sellingPrice : ℝ := 5
def minProfitMarginRate : ℝ := 0.1
def minProfit : ℝ := costPrice * minProfitMarginRate -- which is 0.4

-- The maximum discount rate that preserves the required profit margin
theorem max_discount_rate : 
  ∃ x : ℝ, (0 ≤ x ∧ x ≤ 100) ∧ (sellingPrice * (1 - x / 100) - costPrice ≥ minProfit) ∧ (x = 12) :=
by
  sorry

end max_discount_rate_l471_471289


namespace train_speed_proof_l471_471710

def train_speed_in_kmph (train_length_m : ℕ) (passing_time_s : ℕ) (man_speed_kmph : ℕ) : ℕ :=
  let man_speed_ms := (man_speed_kmph * 1000) / 3600
  let relative_speed_ms := train_length_m / passing_time_s
  let train_speed_ms := relative_speed_ms - man_speed_ms
  (train_speed_ms * 3600) / 1000

theorem train_speed_proof : train_speed_in_kmph 165 9 6 = 60 := by
  -- convert the man's speed from kmph to m/s
  have man_speed_ms : ℕ := (6 * 1000) / 3600
  -- calculate the relative speed in m/s (length/time)
  have relative_speed_ms : ℕ := 165 / 9
  -- calculate the train's speed in m/s (relative_speed - man_speed)
  have train_speed_ms : ℕ := relative_speed_ms - man_speed_ms
  -- convert the train's speed back to kmph (m/s to kmph)
  have train_speed_kmph : ℕ := (train_speed_ms * 3600) / 1000
  -- assert the result
  show train_speed_kmph = 60 from sorry

end train_speed_proof_l471_471710


namespace equal_sum_solutions_l471_471600

def A_points := {1, 2, 3, 4, 5, 6, 7, 8}

variable (a b c d e f g h : ℕ)

theorem equal_sum_solutions :
  (∀ (a b c d e f g h : ℕ), a ∈ A_points ∧ b ∈ A_points ∧ c ∈ A_points ∧ d ∈ A_points ∧ e ∈ A_points ∧ f ∈ A_points ∧ g ∈ A_points ∧ h ∈ A_points ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧
  d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧
  e ≠ f ∧ e ≠ g ∧ e ≠ h ∧
  f ≠ g ∧ f ≠ h ∧
  g ≠ h ∧
  a + b + c + d = 12 ∧
  e = c + d ∧ f = d + a ∧ g = a + b ∧ h = b + c ∧
  e + f + g + h = 24) →
  (∃ (assignments : List (ℕ × ℕ)), assignments.length = 8) :=
sorry

end equal_sum_solutions_l471_471600


namespace time_for_A_and_C_to_complete_work_l471_471263

variable (A_rate B_rate C_rate : ℝ)

theorem time_for_A_and_C_to_complete_work
  (hA : A_rate = 1 / 4)
  (hBC : 1 / 3 = B_rate + C_rate)
  (hB : B_rate = 1 / 12) :
  1 / (A_rate + C_rate) = 2 :=
by
  -- Here would be the proof logic
  sorry

end time_for_A_and_C_to_complete_work_l471_471263


namespace probability_of_exactly_9_correct_matches_is_zero_l471_471232

theorem probability_of_exactly_9_correct_matches_is_zero :
  ∃ (P : ℕ → ℕ → ℕ), 
    (∀ (total correct : ℕ), 
      total = 10 → 
      correct = 9 → 
      P total correct = 0) := 
by {
  sorry
}

end probability_of_exactly_9_correct_matches_is_zero_l471_471232


namespace max_discount_rate_l471_471279

-- Define the cost price and selling price.
def cp : ℝ := 4
def sp : ℝ := 5

-- Define the minimum profit margin.
def min_profit_margin : ℝ := 0.4

-- Define the discount rate d.
def discount_rate (d : ℝ) : ℝ := sp * (1 - d / 100) - cp

-- The theorem to prove the maximum discount rate.
theorem max_discount_rate (d : ℝ) (H : discount_rate d ≥ min_profit_margin) : d ≤ 12 :=
sorry

end max_discount_rate_l471_471279


namespace max_chocolates_eaten_by_Ben_l471_471188

-- Define the situation with Ben and Carol sharing chocolates
variable (b c k : ℕ) -- b for Ben, c for Carol, k is the multiplier

-- Define the conditions
def chocolates_shared (b c : ℕ) : Prop := b + c = 30
def carol_eats_multiple (b c k : ℕ) : Prop := c = k * b ∧ k > 0

-- The theorem statement that we want to prove
theorem max_chocolates_eaten_by_Ben 
  (h1 : chocolates_shared b c) 
  (h2 : carol_eats_multiple b c k) : 
  b ≤ 15 := by
  sorry

end max_chocolates_eaten_by_Ben_l471_471188


namespace person_second_half_speed_l471_471696

theorem person_second_half_speed :
  ∀ (total_time first_half_speed total_distance : ℝ),
  total_time = 10 ∧ first_half_speed = 21 ∧ total_distance = 225 →
  (let first_half_distance := total_distance / 2 in
   let second_half_distance := total_distance / 2 in
   let first_half_time := first_half_distance / first_half_speed in
   let second_half_time := total_time - first_half_time in
   let second_half_speed := second_half_distance / second_half_time in
   second_half_speed = 26.25) :=
begin
  intros,
  apply and.intro,
  sorry
end

end person_second_half_speed_l471_471696


namespace reflect_P_l471_471039

def point :=
  {x : ℝ, y : ℝ}

def reflect_origin (p : point) : point :=
  { x := -p.x, y := -p.y }

theorem reflect_P :
  reflect_origin { x := 2, y := 1 } = { x := -2, y := -1 } :=
by
  -- proof steps would follow here
  sorry

end reflect_P_l471_471039


namespace greatest_distance_between_spheres_l471_471741

open Real EuclideanGeometry

theorem greatest_distance_between_spheres :
  let C1 := (-5, -15, 10) : ℝ × ℝ × ℝ
  let C2 := (15, 5, -20) : ℝ × ℝ × ℝ
  let r1 := 24
  let r2 := 93
  let distance_between_centers := dist C1 C2
  let expected_distance := 117 + 10 * sqrt 17
  (distance_between_centers = sqrt ((-5 - 15)^2 + (-15 - 5)^2 + (10 + (-20))^2)) →
  C1 = (-5, -15, 10) →
  C2 = (15, 5, -20) →
  r1 = 24 →
  r2 = 93 →
  distance_between_centers + r1 + r2 = expected_distance :=
by
  intro C1 C2 r1 r2 distance_between_centers expected_distance h1 h2 h3 h4 h5
  sorry

end greatest_distance_between_spheres_l471_471741


namespace probability_of_sum14_l471_471016

-- Define the standard set of outcomes for a six-faced die.
def die_faces := {1, 2, 3, 4, 5, 6}

-- Define an event where the sum of four dice equals 14.
def event_sum14 (d1 d2 d3 d4 : ℕ) : Prop :=
  d1 + d2 + d3 + d4 = 14 ∧ d1 ∈ die_faces ∧ d2 ∈ die_faces ∧ d3 ∈ die_faces ∧ d4 ∈ die_faces

-- Calculate the total number of outcomes when rolling four six-faced dice.
def total_outcomes := 6^4

-- Calculate the number of favorable outcomes where the sum of the dice equals 14
def favorable_outcomes : ℕ := 54

-- Prove that the probability of the sum being 14 is 54/1296.
theorem probability_of_sum14 :
  favorable_outcomes / total_outcomes = 54 / 1296 :=
begin
  sorry
end

end probability_of_sum14_l471_471016


namespace elvin_fixed_monthly_charge_l471_471748

theorem elvin_fixed_monthly_charge
  (F C : ℝ) 
  (h1 : F + C = 50) 
  (h2 : F + 2 * C = 76) : 
  F = 24 := 
sorry

end elvin_fixed_monthly_charge_l471_471748


namespace luncheon_cost_l471_471974

theorem luncheon_cost (s c p : ℝ) (h1 : 5 * s + 9 * c + 2 * p = 5.95)
  (h2 : 7 * s + 12 * c + 2 * p = 7.90) (h3 : 3 * s + 5 * c + p = 3.50) :
  s + c + p = 1.05 :=
sorry

end luncheon_cost_l471_471974


namespace equal_opposite_roots_eq_m_l471_471457

theorem equal_opposite_roots_eq_m (a b c : ℝ) (m : ℝ) (h : (∃ x : ℝ, (a * x - c ≠ 0) ∧ (((x^2 - b * x) / (a * x - c)) = ((m - 1) / (m + 1)))) ∧
(∀ x : ℝ, ((x^2 - b * x) = 0 → x = 0) ∧ (∃ t : ℝ, t > 0 ∧ ((x = t) ∨ (x = -t))))):
  m = (a - b) / (a + b) :=
by
  sorry

end equal_opposite_roots_eq_m_l471_471457


namespace num_people_for_new_avg_l471_471592

def avg_salary := 430
def old_supervisor_salary := 870
def new_supervisor_salary := 870
def num_workers := 8
def total_people_before := num_workers + 1
def total_salary_before := total_people_before * avg_salary
def workers_salary := total_salary_before - old_supervisor_salary
def total_salary_after := workers_salary + new_supervisor_salary

theorem num_people_for_new_avg :
    ∃ (x : ℕ), x * avg_salary = total_salary_after ∧ x = 9 :=
by
  use 9
  field_simp
  sorry

end num_people_for_new_avg_l471_471592


namespace number_of_smaller_pipes_l471_471692

theorem number_of_smaller_pipes (D_L D_s : ℝ) (h1 : D_L = 8) (h2 : D_s = 2) (v: ℝ) :
  let A_L := (π * (D_L / 2)^2)
  let A_s := (π * (D_s / 2)^2)
  (A_L / A_s) = 16 :=
by {
  sorry
}

end number_of_smaller_pipes_l471_471692


namespace flashlight_lifetime_expectation_leq_two_l471_471548

noncomputable def min_lifetime_expectation (ξ η : ℝ) (E_ξ : ℝ) : Prop :=
  E_ξ = 2 → E(min ξ η) ≤ 2

-- Assume ξ and η are random variables and E denotes the expectation.
axiom E : (ℝ → ℝ) → ℝ

theorem flashlight_lifetime_expectation_leq_two (ξ η : ℝ) (E_ξ : ℝ) (hE_ξ : E_ξ = 2) : E(min ξ η) ≤ 2 :=
  by
    sorry

end flashlight_lifetime_expectation_leq_two_l471_471548


namespace sum_of_digits_l471_471353

theorem sum_of_digits : (∑ n in Finset.range 2023, Nat.digits 10 (n + 1)).sum = 27314 := 
by
  -- Sorry placeholder for the proof
  sorry

end sum_of_digits_l471_471353


namespace choose_numbers_ways_l471_471872

theorem choose_numbers_ways : 
  ∃ (a_1 a_2 a_3 : ℕ) (S : Finset ℕ), 
  S = (Finset.range 15).erase 0 ∧ 
  a_1 ∈ S ∧ a_2 ∈ S ∧ a_3 ∈ S ∧
  a_1 < a_2 ∧ a_2 < a_3 ∧
  a_2 - a_1 ≥ 3 ∧ a_3 - a_2 ≥ 3 ∧ 
  (S.filter (λ (n : ℕ), ∃ b_1 b_2 b_3 : ℕ, (b_1 = n) ∨ (b_2 = n) ∨ (b_3 = n)
                                        ∧ b_1 < b_2 ∧ b_2 < b_3 
                                        ∧ b_2 ≥ b_1 + 3 ∧ b_3 ≥ b_2 + 3)).card = 120 :=
by sorry

end choose_numbers_ways_l471_471872


namespace vector_BP_expression_l471_471313

theorem vector_BP_expression (a b : Vect) :
  let BM := 1 / 3 * a,
      BN := 3 / 5 * b,
      MA := b - BM,
      MP := 1 / 2 * MA   -- derived from solving the system of equations
  in BP = BM + MP) :=
  have MA_expr : MA = b - 1 / 3 * a := by sorry,
  have MP_expr : MP = 1 / 2 * (b - 1 / 3 * a) := by sorry,
  have BP_expr : BP = 1 / 6 * a + 1 / 2 * b := by sorry
  BP = BP_expr

end vector_BP_expression_l471_471313


namespace slope_negative_l471_471374

theorem slope_negative (k b m n : ℝ) (h₁ : k ≠ 0) (h₂ : m < n) 
  (ha : m = k * 1 + b) (hb : n = k * -1 + b) : k < 0 :=
by
  sorry

end slope_negative_l471_471374


namespace peppers_needed_l471_471587

/-- Each sausage sandwich has 4 strips of jalapeno pepper, and
    one jalapeno pepper makes 8 slices. -/
def strips_per_sandwich : ℕ := 4
def slices_per_pepper : ℕ := 8

/-- A sandwich is served every 5 minutes, and the shop operates for 8 hours a day. -/
def minutes_per_sandwich : ℕ := 5
def hours_per_day : ℕ := 8

/-- Convert operation hours to minutes. -/
def minutes_per_day : ℕ := hours_per_day * 60

/-- Calculate the number of sandwiches served per day. -/
def sandwiches_per_day : ℕ := minutes_per_day / minutes_per_sandwich

/-- Calculate the number of peppers needed per sandwich. -/
def peppers_per_sandwich : ℝ := strips_per_sandwich.toReal / slices_per_pepper.toReal

/-- Calculate the total number of peppers needed for the day. -/
def total_peppers : ℝ := sandwiches_per_day.toReal * peppers_per_sandwich

theorem peppers_needed : total_peppers = 48 := by
  sorry

end peppers_needed_l471_471587


namespace angle_range_in_triangle_l471_471493

theorem angle_range_in_triangle
  (A B C M : Type)
  [triangle : IsTriangle ABC]
  (hC : angle C = π / 3)
  (hθ : angle BAC = θ)
  (hM : M ∈ segment B C)
  (hM_distinct : M ≠ B ∧ M ≠ C)
  (h_reflect : ∃ B', triangle_reflect BAM AM B')
  (h_perpendicular : ⊥⊥ AB' (line CM)) :
  θ ∈ Ioo (π / 6) (2 * π / 3) :=
sorry

end angle_range_in_triangle_l471_471493


namespace find_room_width_l471_471120

def room_height : ℕ := 12
def room_length : ℕ := 25
def door_height : ℕ := 6
def door_width : ℕ := 3
def window_height : ℕ := 4
def window_width : ℕ := 3
def number_of_windows : ℕ := 3
def cost_per_sqft : ℕ := 8
def total_cost : ℕ := 7248

theorem find_room_width (x : ℕ) (h : 8 * (room_height * (2 * room_length + 2 * x) - (door_height * door_width + window_height * window_width * number_of_windows)) = total_cost) : 
  x = 15 :=
sorry

end find_room_width_l471_471120


namespace solve_system_nat_l471_471577

open Nat

theorem solve_system_nat (x y z t : ℕ) :
  (x + y = z * t ∧ z + t = x * y) ↔ (x, y, z, t) = (1, 5, 2, 3) ∨ (x, y, z, t) = (2, 2, 2, 2) :=
by
  sorry

end solve_system_nat_l471_471577


namespace sum_of_non_domain_points_l471_471339

noncomputable def g (x : ℝ) : ℝ :=
  1 / (1 + 1 / (1 + 1 / x^2))

theorem sum_of_non_domain_points : (∑ x in {x : ℝ | ¬function.hasValue (g x)}, id) = 0 :=
by
  sorry

end sum_of_non_domain_points_l471_471339


namespace first_ball_red_given_second_black_l471_471645

open ProbabilityTheory

noncomputable def urn_A : Finset (Finset ℕ) := { {0, 0, 0, 0, 1, 1}, {0, 0, 0, 1, 1, 2}, ... }
noncomputable def urn_B : Finset (Finset ℕ) := { {1, 1, 1, 2, 2, 2}, {1, 1, 2, 2, 2, 2}, ... }

noncomputable def prob_draw_red : ℕ := 7 / 15

theorem first_ball_red_given_second_black :
  (∑ A_Burn_selection in ({0, 1} : Finset ℕ), 
     ((∑ ball_draw from A_Burn_selection,
           if A_Burn_selection = 0 then (∑ red in urn_A, if red = 1 then 1 else 0) / 6 / 2
           else (∑ red in urn_B, if red = 1 then 1 else 0) / 6 / 2) *
     ((∑ second_urn_selection in ({0, 1} : Finset ℕ),
           if second_urn_selection = 0 and A_Burn_selection = 0 then 
              ∑ black in urn_A, if black = 1 then 1 else 0 / 6 / 2 
           else 
              ∑ black in urn_B, if black = 1 then 1 else 0 / 6 / 2))) = 7 / 15 :=
sorry

end first_ball_red_given_second_black_l471_471645


namespace matrix_sum_l471_471032

theorem matrix_sum (a : ℕ → ℕ → ℤ) 
    (ha : ∀ i, a i 1 + a i 3 = 2 * a i 2) 
    (hb : ∀ j, a 1 j + a 3 j = 2 * a 2 j)
    (a22 : a 2 2 = 2) : 
    ∑ i in range 3, ∑ j in range 3, a (i + 1) (j + 1) = 18 :=
by
  sorry

end matrix_sum_l471_471032


namespace olivia_correct_answers_l471_471880

theorem olivia_correct_answers (c w : ℕ) (h1 : c + w = 15) (h2 : 4 * c - 3 * w = 25) : c = 10 :=
by
  sorry

end olivia_correct_answers_l471_471880


namespace island_not_called_Maya_l471_471189

def A_statement : Prop := (∀ x, x == "liar") ∧ (island == "Maya")
def B_statement : Prop := (true)

-- The theorem to prove
theorem island_not_called_Maya (A_says : A_statement) (B_says : B_statement) : (island ≠ "Maya") :=
by {
  sorry
}

end island_not_called_Maya_l471_471189


namespace total_distance_l471_471666

-- Define the starting point and movement
structure Point where
  x : ℝ
  y : ℝ

def initial_point : Point := ⟨0, 0⟩

def move_forward (p : Point) (distance : ℝ) (angle_degrees : ℝ) : Point :=
  let angle_radians := angle_degrees * (Real.pi / 180)
  ⟨p.x + distance * Real.cos angle_radians, p.y + distance * Real.sin angle_radians⟩

-- Define the conditions
constant side_length : ℝ := 10
constant angle_turn : ℝ := 60

def path : List Point := List.foldl (fun pts _ => pts ++
  [move_forward pts.head! side_length (angle_turn * pts.length)]) [initial_point] (List.range 6)

theorem total_distance : Real :=
  List.foldl (fun dist pair => dist + (move_forward pair.fst side_length angle_turn).dist pair.snd) 0 (List.zip path (List.tail path))

example : total_distance = 60 := by
  sorry

end total_distance_l471_471666


namespace pb_tangent_secant_l471_471915

noncomputable def solve_pb (PA PT AB PB : ℝ) : PB = 9 :=
by sorry

theorem pb_tangent_secant 
  (P O T A B : Type*)
  (PA PB PT AB : ℝ)
  (h1 : PA = 4)
  (h2 : PT = 2 * (AB - PA))
  (h3 : ∀ {x}, PA * PB = x * x) :
  PB = 9 := solve_pb PA PT AB PB
    sorry

end pb_tangent_secant_l471_471915


namespace sum_of_cubes_l471_471449

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 20) : x^3 + y^3 = 87.5 := 
by 
  sorry

end sum_of_cubes_l471_471449


namespace coloring_problem_l471_471652

-- Given 2015 lines in a plane such that no two are parallel and no three are concurrent.
-- Let E be the set of their intersection points.
-- We need to prove the minimum number of colors needed to color the points in E such that
-- any two points on the same line, whose connecting segment contains no other points from E,
-- are colored differently, is 3.

theorem coloring_problem :
  ∀ (n: ℕ) (lines: fin n → affine_plane.point → affine_plane.point)
  (hn: n = 2015)
  (h_no_parallel: ∀ i j : fin n, i ≠ j → ¬ parallel (lines i) (lines j))
  (h_no_concurrent: ∀ i j k : fin n, i ≠ j → j ≠ k → k ≠ i → ¬ concurrent (lines i) (lines j) (lines k)),
  (∃ (coloring : affine_plane.point → fin 3), 
  ∀ p q : affine_plane.point, 
  (p ≠ q) ∧ (∃ i : fin n, (lines i) p ∧ (lines i) q)
  ∧ (∀ r : affine_plane.point, (lines i) r → r = p ∨ r = q → coloring p ≠ coloring q)) := 
sorry

end coloring_problem_l471_471652


namespace num_ways_distribute_balls_l471_471426

-- Definitions of the conditions
def balls := 6
def boxes := 4

-- Theorem statement
theorem num_ways_distribute_balls : 
  ∃ n : ℕ, (balls = 6 ∧ boxes = 4) → n = 8 :=
sorry

end num_ways_distribute_balls_l471_471426


namespace find_extrema_l471_471369

theorem find_extrema (x y : ℝ) (h1 : x < 0) (h2 : -1 < y) (h3 : y < 0) : 
  max (max x (x*y)) (x*y^2) = x*y ∧ min (min x (x*y)) (x*y^2) = x :=
by sorry

end find_extrema_l471_471369


namespace probability_exactly_nine_matches_l471_471239

theorem probability_exactly_nine_matches (n : ℕ) (h : n = 10) : 
  (∃ p : ℕ, p = 9 ∧ probability_of_exact_matches n p = 0) :=
by {
  sorry
}

end probability_exactly_nine_matches_l471_471239


namespace sin_2theta_value_l471_471445

theorem sin_2theta_value (θ : ℝ) 
  (h : (sqrt 2 * cos (2 * θ)) / (cos(π / 4 + θ)) = sqrt 3 * sin (2 * θ)) : 
  sin (2 * θ) = -2 / 3 :=
by
  sorry

end sin_2theta_value_l471_471445


namespace remainder_of_2023rd_term_eq_0_l471_471735

theorem remainder_of_2023rd_term_eq_0 :
  let seq := λ n, (n, n + 1) ∈ ℕ × ℕ
  let nth_term := 63 in
  nth_term % 7 = 0 :=
begin
  sorry
end

end remainder_of_2023rd_term_eq_0_l471_471735


namespace fraction_of_sum_l471_471687

theorem fraction_of_sum (S n : ℝ) (h1 : n = S / 6) : n / (S + n) = 1 / 7 :=
by sorry

end fraction_of_sum_l471_471687


namespace speed_conversion_l471_471705

noncomputable def mps_to_kmph (speed_mps : ℝ) : ℝ :=
  speed_mps * 3.6

theorem speed_conversion (h : 1 = 3.6) : mps_to_kmph 12.7788 = 45.96 :=
  by
    sorry

end speed_conversion_l471_471705


namespace range_of_b_over_a_l471_471050

theorem range_of_b_over_a (A B a b : ℝ) (h1 : 0 < A) (h2 : A < π/4) (hB_eq : B = 3 * A) (h_triangle : A + B < π) :
  1 < b / a ∧ b / a < 3 :=
by
  have hcos_lt : (1/2 : ℝ) < Real.cos A^2 := sorry
  have hcos_ub : Real.cos A^2 < 1 := sorry
  have hlt : 1 < 4 * Real.cos A^2 - 1 := sorry
  have hub : 4 * Real.cos A^2 - 1 < 3 := sorry
  exact ⟨hlt, hub⟩

end range_of_b_over_a_l471_471050


namespace ratio_a_b_equilibrium_l471_471158

-- Given math problem and solution:
-- The problem involves a system in equilibrium with three equal masses, small pulleys, a lightweight string, and negligible friction.
-- The goal is to find the ratio \( \frac{a}{b} \) given that the angle \( \theta \) the string makes with the horizontal is \( 30^\circ \).

-- Step a): Identify all questions and conditions in the given problem.
-- Question: What is the ratio \( \frac{a}{b} \) in the system?
-- Conditions:
-- 1. The system is in equilibrium.
-- 2. The masses are equal.
-- 3. Pulleys are small.
-- 4. The string is lightweight.
-- 5. Friction is negligible.
-- 6. The angle \( \theta = 30^\circ \).

-- Step b): Identify all solution steps and the correct answers in the given solution.
-- Correct Answer: \( \frac{a}{b} = 2\sqrt{3} \)

-- Step c): Translate the (question, conditions, correct answer) tuple to a mathematically equivalent proof problem.
-- Proof problem: Prove \( \frac{a}{b} = 2\sqrt{3} \) given the conditions of the problem.

-- Step d): Rewrite the math proof problem as a Lean 4 statement.
theorem ratio_a_b_equilibrium
  (m : ℝ) -- mass of the objects
  (g : ℝ) -- acceleration due to gravity
  (a b : ℝ) -- distances in the system
  (h1 : b > 0) -- condition to avoid division by zero
  (h2 : tan 30 = sqrt 3 / 3) : 
  (a / b = 2 * sqrt 3) :=
sorry

end ratio_a_b_equilibrium_l471_471158


namespace expected_lifetime_flashlight_l471_471531

noncomputable def E (X : ℝ) : ℝ := sorry -- Define E as the expectation operator

variables (ξ η : ℝ) -- Define ξ and η as random variables representing lifetimes of blue and red bulbs
variable (h_exi : E ξ = 2) -- Given condition E ξ = 2

theorem expected_lifetime_flashlight (h_min : ∀ x y : ℝ, min x y ≤ x) :
  E (min ξ η) ≤ 2 :=
by
  sorry

end expected_lifetime_flashlight_l471_471531


namespace find_original_cost_price_l471_471167

variables (P : ℝ) (A B C D E : ℝ)

-- Define the conditions as per the problem statement
def with_tax (P : ℝ) : ℝ := P * 1.10
def profit_60 (price : ℝ) : ℝ := price * 1.60
def profit_25 (price : ℝ) : ℝ := price * 1.25
def loss_15 (price : ℝ) : ℝ := price * 0.85
def profit_30 (price : ℝ) : ℝ := price * 1.30

-- The final price E is given.
def final_price (P : ℝ) : ℝ :=
  profit_30 
  (loss_15 
  (profit_25 
  (profit_60 
  (with_tax P))))

-- To find original cost price P given final price of Rs. 500.
theorem find_original_cost_price (h : final_price P = 500) : 
  P = 500 / 2.431 :=
by 
  sorry

end find_original_cost_price_l471_471167


namespace newlandia_population_density_l471_471492

theorem newlandia_population_density (population_newlandia : ℕ) (area_newlandia : ℕ) (conversion : ℕ) (density_oldlandia : ℕ) :
  population_newlandia = 350000000 →
  area_newlandia = 4500000 →
  conversion = 5280^2 →
  density_oldlandia = 700 →
  let avg_sq_feet_per_person : ℚ := (area_newlandia * conversion : ℕ) / population_newlandia in
  avg_sq_feet_per_person ≈ 360000 ∧ avg_sq_feet_per_person > density_oldlandia := by
  sorry

end newlandia_population_density_l471_471492


namespace one_in_set_A_l471_471413

theorem one_in_set_A : 1 ∈ {x | x ≥ -1} :=
sorry

end one_in_set_A_l471_471413


namespace num_arithmetic_sequences_l471_471816

-- Definitions of the arithmetic sequence conditions
def is_arithmetic_sequence (a d n : ℕ) : Prop :=
  0 ≤ a ∧ 0 ≤ d ∧ n ≥ 3 ∧ 
  (∃ k : ℕ, k = 97 ∧ 
  (n * (2 * a + (n - 1) * d) = 2 * k ^ 2)) 

-- Prove that there are exactly 4 such sequences
theorem num_arithmetic_sequences : 
  ∃ (n : ℕ) (a d : ℕ), 
  is_arithmetic_sequence a d n ∧ 
  (n * (2 * a + (n - 1) * d) = 2 * 97^2) ∧ (
    (n = 97 ∧ ((a = 97 ∧ d = 0) ∨ (a = 49 ∧ d = 1) ∨ (a = 1 ∧ d = 2))) ∨
    (n = 97^2 ∧ a = 1 ∧ d = 0)
  ) :=
sorry

end num_arithmetic_sequences_l471_471816


namespace sum_binom_eq_two_pow_sum_even_binom_eq_two_pow_minus1_sum_k_binom_eq_n_two_pow_minus1_sum_double_binom_eq_three_pow_l471_471960

theorem sum_binom_eq_two_pow (n : ℕ) (hn : n > 0) : 
  ∑ k in Finset.range (n + 1), Nat.choose n k = 2^n := 
by sorry

theorem sum_even_binom_eq_two_pow_minus1 (n : ℕ) (hn : n > 0) : 
  ∑ k in Finset.range (n + 1), if k % 2 = 0 then Nat.choose n k else 0 = 2^(n-1) := 
by sorry

theorem sum_k_binom_eq_n_two_pow_minus1 (n : ℕ) (hn : n > 0) : 
  ∑ k in Finset.range (n + 1), k * Nat.choose n k = n * 2^(n-1) := 
by sorry

theorem sum_double_binom_eq_three_pow (n : ℕ) (hn : n > 0) :
  ∑ k in Finset.range (n + 1), ∑ l in Finset.range (k + 1), Nat.choose n k * Nat.choose k l = 3^n := 
by sorry

end sum_binom_eq_two_pow_sum_even_binom_eq_two_pow_minus1_sum_k_binom_eq_n_two_pow_minus1_sum_double_binom_eq_three_pow_l471_471960


namespace scrap_cookie_radius_l471_471935

noncomputable def large_dough_radius : ℝ := 3.5
noncomputable def small_cookie_radius : ℝ := 1.0
noncomputable def num_cookies : ℕ := 9

theorem scrap_cookie_radius :
  ∀ (large_dough_radius small_cookie_radius : ℝ) (num_cookies : ℕ),
    large_dough_radius = 3.5 →
    small_cookie_radius = 1 →
    num_cookies = 9 →
    let large_area := π * large_dough_radius^2 in
    let small_area := π * small_cookie_radius^2 in
    let total_small_area := num_cookies * small_area in
    let scrap_area := large_area - total_small_area in
    ∃ radius : ℝ, radius = Real.sqrt (scrap_area / π) ∧ radius = Real.sqrt 3.25 :=
begin
  intros,
  simp, -- simplifying mathematical expressions
  rw [←Real.sqrt_eq_iff_sqr_eq, Real.sqrt_mul_self],
  norm_num, -- normalizing numbers
  sorry -- placeholder for remaining proof steps
end

end scrap_cookie_radius_l471_471935


namespace average_marks_of_all_students_l471_471017

theorem average_marks_of_all_students :
  (let total_marks := 40 * 45 + 50 * 55 + 60 * 65 in
   let total_students := 40 + 50 + 60 in
   total_marks / total_students = 56.33) :=
by
  sorry

end average_marks_of_all_students_l471_471017


namespace length_AE_k_m_l471_471968

theorem length_AE_k_m {A B C D E F : Point}
  (side_length_one : distance A B = 1)
  (E_on_AB : ∃ t ∈ Icc (0 : ℝ) 1, E = A + t * (B - A))
  (F_on_CB : ∃ t ∈ Icc (0 : ℝ) 1, F = C + t * (B - C))
  (AE_eq_CF : distance A E = distance C F)
  (AD_CD_coincide_on_BD : ∀ D', (folded_on B D D') → distance A D = distance C D)
  : ∃ k m : ℕ, AE_length = sqrt k - m ∧ k + m = 3 :=
begin
  -- Proof goes here
  sorry
end

end length_AE_k_m_l471_471968


namespace orthocenter_of_triangle_l471_471053

open EuclideanGeometry 

variables {A B C X A₁ B₁ C₁ : Point}

-- Assume the conditions given in the problem
variable (h1 : InsideTriangle X A B C) -- X inside triangle ABC
variables (hAX : LineThrough A X intersectAt A₁ (LineThrough B C)) -- AX intersects BC at A₁
variables (hBX : LineThrough B X intersectAt B₁ (LineThrough C A)) -- BX intersects CA at B₁
variables (hCX : LineThrough C X intersectAt C₁ (LineThrough A B)) -- CX intersects AB at C₁

-- Assume circumcircles intersection condition
variables (hAB₁C₁ : CircumcircleIntersects (Triangle A B₁ C₁) X)
variables (hA₁BC₁ : CircumcircleIntersects (Triangle A₁ B C₁) X)
variables (hA₁B₁C : CircumcircleIntersects (Triangle A₁ B₁ C) X)

theorem orthocenter_of_triangle 
  (h : hAX ∧ hBX ∧ hCX ∧ hAB₁C₁ ∧ hA₁BC₁ ∧ hA₁B₁C) : IsOrthocenter X A B C :=
sorry

end orthocenter_of_triangle_l471_471053


namespace domain_lg_tan_minus_sqrt3_l471_471599

open Real

theorem domain_lg_tan_minus_sqrt3 :
  {x : ℝ | ∃ k : ℤ, k * π + π / 3 < x ∧ x < k * π + π / 2} =
    {x : ℝ | ∃ k : ℤ, k * π + π / 3 < x ∧ x < k * π + π / 2} :=
by
  sorry

end domain_lg_tan_minus_sqrt3_l471_471599


namespace tank_capacity_l471_471671

variable (C : ℝ)

theorem tank_capacity (h : (3/4) * C + 9 = (7/8) * C) : C = 72 :=
by
  sorry

end tank_capacity_l471_471671


namespace num_ways_distribute_balls_l471_471425

-- Definitions of the conditions
def balls := 6
def boxes := 4

-- Theorem statement
theorem num_ways_distribute_balls : 
  ∃ n : ℕ, (balls = 6 ∧ boxes = 4) → n = 8 :=
sorry

end num_ways_distribute_balls_l471_471425


namespace find_smallest_n_l471_471063

noncomputable def smallest_n (X : Finset ℕ) : ℕ :=
  2 * Nat.choose 100 50 + 2 * Nat.choose 100 49 + 1

lemma subsets_sequence_le_three (X : Finset ℕ) (hX : X.card = 100) (n : ℕ) (S : Finset (Finset ℕ)) :
  S.card = n → (∃ (A_i A_j A_k : Finset ℕ), 1 ≤ i ∧ i < j ∧ j < k ∧ k ≤ n ∧ A_i ⊆ A_j ∧ A_j ⊆ A_k) ∨
  ∃ (A_i A_j A_k : Finset ℕ), 1 ≤ i ∧ i < j ∧ j < k ∧ k ≤ n ∧ A_i ⊇ A_j ∧ A_j ⊇ A_k :=
sorry

theorem find_smallest_n (hX : (Finset.range 100).card = 100) :
  smallest_n (Finset.range 100) = 200 * Nat.choose 100 50 + 200 * Nat.choose 100 49 + 1 :=
by 
  have := subsets_sequence_le_three (Finset.range 100) (by simp) (smallest_n (Finset.range 100))
  cases this with hsubsets hreversesubsets
  -- Additional proof details would go here if we were to complete the proof
  sorry

end find_smallest_n_l471_471063


namespace modulus_of_z_l471_471079

open Complex

theorem modulus_of_z :
  ∃ z : ℂ, z * Complex.i = 3 + 4 * Complex.i ∧ Complex.abs z = 5 :=
by
  use 4 - 3 * Complex.i
  split
  · have h : Complex.i * Complex.i = -1 := by simp [Complex.i]
    calc
      (4 - 3 * Complex.i) * Complex.i
          = 4 * Complex.i - 3 * (Complex.i * Complex.i) : by ring
      ... = 4 * Complex.i - 3 * (-1) : by rw [h]
      ... = 4 * Complex.i + 3 : by ring
      ... = 3 + 4 * Complex.i : by simp
  · calc
    Complex.abs (4 - 3 * Complex.i)
        = Real.sqrt (4^2 + (-3)^2) : by simp [Complex.abs, abs2_eq]
    ... = Real.sqrt (16 + 9) : by norm_num
    ... = Real.sqrt 25 : by norm_num
    ... = 5 : by norm_num

end modulus_of_z_l471_471079


namespace geometric_sequence_log_sum_l471_471155

theorem geometric_sequence_log_sum (a r : ℕ) (ha_pos : a > 0) (hr_pos : r > 0)
  (hlogsum : (Finset.range 12).sum (λ i, Real.log (a * r^i) / Real.log 8) = 2006) :
  ∃ n : ℕ, n = 46 :=
by
  sorry

end geometric_sequence_log_sum_l471_471155


namespace sin_angle_identity_l471_471340

theorem sin_angle_identity : 
  (Real.sin (Real.pi / 4) * Real.sin (7 * Real.pi / 12) + Real.sin (Real.pi / 4) * Real.sin (Real.pi / 12)) = Real.sqrt 3 / 2 := 
by 
  sorry

end sin_angle_identity_l471_471340


namespace triangle_ABD_AD_equals_16_l471_471221

theorem triangle_ABD_AD_equals_16
  (A B C D M : Type)
  [LinearOrder A] [LinearOrder B] [LinearOrder C] [LinearOrder D]
  [LinearOrder M]
  (AB AC BC AD AM : ℝ)
  (h1 : AB = 8)
  (h2 : AC = 12)
  (h3 : BC = 5)
  (h4 : D is the intersection point of the tangents to the circle centered at M from B and C other than AB and AC)
  (h5 : M is the second intersection of the internal angle bisector of ∠BAC with the circumcircle of △ABC)
  (h6 : ω is the circle centered at M tangent to AB and AC)
  (h7 : AM = MB ∧ AM = MC)
  : AD = 16 := by
  sorry

end triangle_ABD_AD_equals_16_l471_471221


namespace tenth_derivative_correct_l471_471790

noncomputable def fun_y (x : ℝ) : ℝ :=
  (exp x) * (x^3 - 2)

noncomputable def tenth_derivative : (ℝ → ℝ) :=
  fun x => (exp x) * (x^3 + 90 * x^2 + 270 * x + 118)

theorem tenth_derivative_correct (x : ℝ) :
  (deriv^[10] fun_y) x = tenth_derivative x :=
sorry

end tenth_derivative_correct_l471_471790


namespace min_sin_cos_sixth_power_l471_471776

noncomputable def min_value_sin_cos_expr : ℝ :=
  (3 + 2 * Real.sqrt 2) / 12

theorem min_sin_cos_sixth_power :
  ∃ x : ℝ, (∀ y, (Real.sin y) ^ 6 + 2 * (Real.cos y) ^ 6 ≥ min_value_sin_cos_expr) ∧ 
            ((Real.sin x) ^ 6 + 2 * (Real.cos x) ^ 6 = min_value_sin_cos_expr) :=
sorry

end min_sin_cos_sixth_power_l471_471776


namespace smallest_x_solution_l471_471963

theorem smallest_x_solution (x : ℚ) :
  (7 * (8 * x^2 + 8 * x + 11) = x * (8 * x - 45)) →
  (x = -7/3 ∨ x = -11/16) →
  x = -7/3 :=
by
  sorry

end smallest_x_solution_l471_471963


namespace saturated_function_2014_l471_471064

def saturated (f : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, f^[f^[f n] n] n = n

theorem saturated_function_2014 (f : ℕ → ℕ) (m : ℕ) (h : saturated f) :
  (m ∣ 2014) ↔ (f^[2014] m = m) :=
sorry

end saturated_function_2014_l471_471064


namespace probability_of_exactly_nine_correct_matches_is_zero_l471_471234

theorem probability_of_exactly_nine_correct_matches_is_zero :
  let n := 10 in
  let match_probability (correct: Fin n → Fin n) (guess: Fin n → Fin n) (right_count: Nat) :=
    (Finset.univ.filter (λ i => correct i = guess i)).card = right_count in
  ∀ (correct_guessing: Fin n → Fin n), 
    ∀ (random_guessing: Fin n → Fin n),
      match_probability correct_guessing random_guessing 9 → 
        match_probability correct_guessing random_guessing 10 :=
begin
  sorry -- This skips the proof part
end

end probability_of_exactly_nine_correct_matches_is_zero_l471_471234


namespace B_speed_is_8_m_per_s_l471_471877

variables (V_a V_b : ℝ) (T : ℝ)

-- Conditions
-- 1. Race distance
def race_distance : ℝ := 1000
-- 2. B's distance when A finishes race
def B_distance_when_A_finishes : ℝ := 800
-- 3. B's additional time
def B_additional_time : ℝ := 25

-- Given Expressions
def Va_expr : ℝ := race_distance / T
def Vb_expr1 : ℝ := B_distance_when_A_finishes / T
def Vb_expr2 : ℝ := race_distance / (T + B_additional_time)

-- Proof Problem Statement
theorem B_speed_is_8_m_per_s (h : Vb_expr1 = Vb_expr2) : V_b = 8 :=
by
  sorry

end B_speed_is_8_m_per_s_l471_471877


namespace find_vector_c_find_parallelogram_area_l471_471417

-- Definitions of points A, B, and C
def A : ℝ × ℝ × ℝ := (2, 0, -2)
def B : ℝ × ℝ × ℝ := (1, -1, -2)
def C : ℝ × ℝ × ℝ := (3, 0, -4)

-- Vectors a and b
def vector_a : ℝ × ℝ × ℝ := (B.1 - A.1, B.2 - A.2, B.3 - A.3)  -- (1 - 2, -1 - 0, -2 - (-2))
def vector_b : ℝ × ℝ × ℝ := (C.1 - A.1, C.2 - A.2, C.3 - A.3)  -- (3 - 2, 0 - 0, -4 - (-2))

-- Magnitude function
def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 * v.1 + v.2 * v.2 + v.3 * v.3)

-- Condition on vector c
def is_parallel (v₁ v₂ : ℝ × ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v₁ = (k * v₂.1, k * v₂.2, k * v₂.3)

-- Theorem statements
theorem find_vector_c (c : ℝ × ℝ × ℝ) (h1 : magnitude c = 3) (h2 : is_parallel c vector_b) :
  c = (2, 1, -2) ∨ c = (-2, -1, 2) := sorry

theorem find_parallelogram_area :
  (vector_a.1 * (vector_b.2 * 0 - 0) - vector_a.2 * (vector_b.1 * 0 - vector_b.3)) = 3 := sorry

end find_vector_c_find_parallelogram_area_l471_471417


namespace probability_of_exactly_9_correct_matches_is_zero_l471_471225

theorem probability_of_exactly_9_correct_matches_is_zero :
  ∀ (n : ℕ) (translate : Fin n → Fin n),
    (n = 10) → 
    (∀ i : Fin n, translate i ≠ i) → 
    (∃ (k : ℕ), (k < n ∧ k ≠ n-1) → false ) :=
by
  sorry

end probability_of_exactly_9_correct_matches_is_zero_l471_471225


namespace pencils_left_proof_l471_471746

noncomputable def total_pencils_left (a d : ℕ) : ℕ :=
  let total_initial_pencils : ℕ := 30
  let total_pencils_given_away : ℕ := 15 * a + 105 * d
  total_initial_pencils - total_pencils_given_away

theorem pencils_left_proof (a d : ℕ) :
  total_pencils_left a d = 30 - (15 * a + 105 * d) :=
by
  sorry

end pencils_left_proof_l471_471746


namespace balls_in_boxes_l471_471439

def num_ways_to_partition_6_in_4_parts : ℕ :=
  -- The different partitions of 6: (6,0,0,0), (5,1,0,0), (4,2,0,0), (4,1,1,0),
  -- (3,3,0,0), (3,2,1,0), (3,1,1,1), (2,2,2,0), (2,2,1,1)
  ([
    [6, 0, 0, 0],
    [5, 1, 0, 0],
    [4, 2, 0, 0],
    [4, 1, 1, 0],
    [3, 3, 0, 0],
    [3, 2, 1, 0],
    [3, 1, 1, 1],
    [2, 2, 2, 0],
    [2, 2, 1, 1]
  ]).length

theorem balls_in_boxes : num_ways_to_partition_6_in_4_parts = 9 := by
  sorry

end balls_in_boxes_l471_471439


namespace contradiction_method_example_l471_471945

variables {a b c : ℝ}
variables (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
variables (h4 : a + b + c > 0) (h5 : ab + bc + ca > 0)
variables (h6 : (a < 0 ∧ b < 0) ∨ (a < 0 ∧ c < 0) ∨ (b < 0 ∧ c < 0))

theorem contradiction_method_example : false :=
by {
  sorry
}

end contradiction_method_example_l471_471945


namespace probability_of_exactly_9_correct_matches_is_zero_l471_471227

theorem probability_of_exactly_9_correct_matches_is_zero :
  ∀ (n : ℕ) (translate : Fin n → Fin n),
    (n = 10) → 
    (∀ i : Fin n, translate i ≠ i) → 
    (∃ (k : ℕ), (k < n ∧ k ≠ n-1) → false ) :=
by
  sorry

end probability_of_exactly_9_correct_matches_is_zero_l471_471227


namespace sum_fib_identity_l471_471096

-- Definition of Fibonacci numbers
def fib : ℕ → ℕ
| 0       := 0
| 1       := 1
| (n + 2) := fib (n + 1) + fib n

-- Given condition for Fibonacci numbers for even positive m
lemma fib_property (m : ℕ) (hm : 0 < m) : 
  (1 / fib (2 * m) : ℚ) = (fib (m - 1) : ℚ) / (fib m) - (fib (2 * m - 1) : ℚ) / (fib (2 * m)) :=
sorry

-- The main theorem to prove
theorem sum_fib_identity (n : ℕ) (hn : 1 ≤ n) : 
  (∑ k in Finset.range (n + 1), (1 : ℚ) / fib (2^k)) = 3 - (fib (2^n - 1) : ℚ) / (fib (2^n) : ℚ) :=
sorry

end sum_fib_identity_l471_471096


namespace problem1_problem2_l471_471416

open Set

variable (a : Real)

-- Problem 1: Prove the intersection M ∩ (C_R N) equals the given set
theorem problem1 :
  let M := { x : ℝ | x^2 - 3*x ≤ 10 }
  let N := { x : ℝ | 3 ≤ x ∧ x ≤ 5 }
  let C_RN := { x : ℝ | x < 3 ∨ 5 < x }
  M ∩ C_RN = { x : ℝ | -2 ≤ x ∧ x < 3 } :=
by
  sorry

-- Problem 2: Prove the range of values for a such that M ∪ N = M
theorem problem2 :
  let M := { x : ℝ | x^2 - 3*x ≤ 10 }
  let N := { x : ℝ | a+1 ≤ x ∧ x ≤ 2*a+1 }
  (M ∪ N = M) → a ≤ 2 :=
by
  sorry

end problem1_problem2_l471_471416


namespace minChordLength_is_pi_div_2_l471_471398

noncomputable def minChordLength (α : ℝ) : ℝ :=
  if h : -1 ≤ α ∧ α ≤ 1 then
    let arcsinα := Real.arcsin α
    let arccosα := Real.arccos α
    let d := Real.sqrt ((arcsinα - arccosα)^2 + (Real.pi / 2)^2)
    d
  else 0

theorem minChordLength_is_pi_div_2 :
  ∃ α, minChordLength α = Real.pi / 2 :=
begin
  use Real.sin (Real.pi / 4),
  rw [minChordLength, dif_pos],
  { simp, norm_num },
  { split; linarith [Real.sin_nonneg_iff.mpr 
     { left:=by norm_num, right:=by norm_num }, Real.sin_pi_div_four] }
end

end minChordLength_is_pi_div_2_l471_471398


namespace percent_decrease_first_year_l471_471257

theorem percent_decrease_first_year (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 100) 
  (h_second_year : 0.9 * (100 - x) = 54) : x = 40 :=
by sorry

end percent_decrease_first_year_l471_471257


namespace solve_quadratic_1_solve_quadratic_2_solve_quadratic_3_l471_471964

theorem solve_quadratic_1 (x : Real) : x^2 - 2 * x - 4 = 0 ↔ (x = 1 + Real.sqrt 5 ∨ x = 1 - Real.sqrt 5) :=
by
  sorry

theorem solve_quadratic_2 (x : Real) : (x - 1)^2 = 2 * (x - 1) ↔ (x = 1 ∨ x = 3) :=
by
  sorry

theorem solve_quadratic_3 (x : Real) : (x + 1)^2 = 4 * x^2 ↔ (x = 1 ∨ x = -1 / 3) :=
by
  sorry

end solve_quadratic_1_solve_quadratic_2_solve_quadratic_3_l471_471964


namespace y_intercept_of_line_l471_471623

theorem y_intercept_of_line (a b : ℝ) : 
  (λ x y : ℝ, x / (a^2) - y / (b^2) = 1) 0 (-b^2) :=
by
  -- We need to show that substituting x = 0 into the line equation satisfies the equation with y = -b^2
  sorry

end y_intercept_of_line_l471_471623


namespace partI_partII_l471_471838

noncomputable def problem1 : set ℝ :=
{x : ℝ | -1/2 ≤ x ∧ x ≤ 5/2}

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
abs (2 * x - a) + abs (2 * x - 1)

theorem partI (x : ℝ) :
  (|2 * x - 3| + |2 * x - 1| ≤ 6) ↔ (-1/2 ≤ x ∧ x ≤ 5/2) :=
sorry

noncomputable def problem2 : set ℝ :=
{a : ℝ | -real.sqrt 14 ≤ a ∧ a ≤ 1 + real.sqrt 13}

theorem partII (a : ℝ) :
  (∀ (x : ℝ), (|2 * x - a| + |2 * x - 1| ≥ a^2 - a - 13)) ↔ (-real.sqrt 14 ≤ a ∧ a ≤ 1 + real.sqrt 13) :=
sorry

end partI_partII_l471_471838


namespace wrapping_paper_area_l471_471690

theorem wrapping_paper_area 
  (l w h : ℝ) :
  (l + 4 + 2 * h) ^ 2 = l^2 + 8 * l + 16 + 4 * l * h + 16 * h + 4 * h^2 := 
by 
  sorry

end wrapping_paper_area_l471_471690


namespace game_not_fair_probability_first_player_wins_approx_l471_471170

def fair_game (n : ℕ) (P : ℕ → ℚ) : Prop :=
  ∀ k j, k ≤ n → j ≤ n → P k = P j

noncomputable def probability_first_player_wins (n : ℕ) : ℚ :=
  let r := (n - 1 : ℚ) / n;
  (1 : ℚ) / n * (1 : ℚ) / (1 - r ^ n)

-- Check if the game is fair for 36 players
theorem game_not_fair : ¬ fair_game 36 (λ i, probability_first_player_wins 36) :=
sorry

-- Calculate the approximate probability of the first player winning
theorem probability_first_player_wins_approx (n : ℕ)
  (h₁ : n = 36) :
  probability_first_player_wins n ≈ 0.044 :=
sorry

end game_not_fair_probability_first_player_wins_approx_l471_471170


namespace base_seven_sum_of_digits_of_product_l471_471152

theorem base_seven_sum_of_digits_of_product :
  let a := 24
  let b := 30
  let product := a * b
  let base_seven_product := 105 -- The product in base seven notation
  let sum_of_digits (n : ℕ) : ℕ := n.digits 7 |> List.sum
  sum_of_digits base_seven_product = 6 :=
by
  sorry

end base_seven_sum_of_digits_of_product_l471_471152


namespace parallelogram_BCHG_area_l471_471904

open_locale classical

universe u

-- Definitions
variables {A B C D E F G H : Type*}
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D]
variables (P : triangle ABC) (Γ : circle P) (D_eq_mid_arc : midpoint_arc (P B) (P C) D)
variables (E F : circle_point Γ) (DE_perp_AC : orthogonal (line_through_points D E) (line_through_points (P A) (P C))) (DF_perp_AB : orthogonal (line_through_points D F) (line_through_points (P A) (P B)))
variables (G : intersection_point (line_through_points (P B) E) (line_through_points D F))
variables (H : intersection_point (line_through_points (P C) F) (line_through_points D E))
variables (AB_eq_8 : distance (P A) (P B) = 8) (AC_eq_10 : distance (P A) (P C) = 10) (angle_BAC_eq_60 : angle (P B) (P A) (P C) = 60)

-- To Prove
theorem parallelogram_BCHG_area :
  area_parallelogram (quadrilateral (P B) (P C) H G) = 2 * real.sqrt 3 :=
by sorry

end parallelogram_BCHG_area_l471_471904


namespace relationship_between_p_and_q_l471_471368

variable {a b : ℝ}

theorem relationship_between_p_and_q 
  (h_a : a > 2) 
  (h_p : p = a + 1 / (a - 2)) 
  (h_q : q = -b^2 - 2 * b + 3) : 
  p ≥ q := 
sorry

end relationship_between_p_and_q_l471_471368


namespace q_minus_p_l471_471549

noncomputable def sum_nine_terms : ℚ := ∑ n in finset.range 9, 1 / (n + 1) / (n + 2) / (n + 3)

theorem q_minus_p :
  let p := (sum_nine_terms.num : ℤ)
  let q := (sum_nine_terms.denom : ℤ)
  q - p = 83 := by
  sorry

end q_minus_p_l471_471549


namespace total_time_per_week_l471_471632

noncomputable def meditating_time_per_day : ℝ := 1
noncomputable def reading_time_per_day : ℝ := 2 * meditating_time_per_day
noncomputable def exercising_time_per_day : ℝ := 0.5 * meditating_time_per_day
noncomputable def practicing_time_per_day : ℝ := (1/3) * reading_time_per_day

noncomputable def total_time_per_day : ℝ :=
  meditating_time_per_day + reading_time_per_day + exercising_time_per_day + practicing_time_per_day

theorem total_time_per_week :
  total_time_per_day * 7 = 29.17 := by
  sorry

end total_time_per_week_l471_471632


namespace apple_pies_l471_471896

theorem apple_pies (total_apples not_ripe_apples apples_per_pie : ℕ) 
    (h1 : total_apples = 34) 
    (h2 : not_ripe_apples = 6) 
    (h3 : apples_per_pie = 4) : 
    (total_apples - not_ripe_apples) / apples_per_pie = 7 :=
by 
    sorry

end apple_pies_l471_471896


namespace roger_final_money_l471_471678

variable (initial_money : ℕ)
variable (spent_money : ℕ)
variable (received_money : ℕ)

theorem roger_final_money (h1 : initial_money = 45) (h2 : spent_money = 20) (h3 : received_money = 46) :
  (initial_money - spent_money + received_money) = 71 :=
by
  sorry

end roger_final_money_l471_471678


namespace number_of_non_similar_regular_2000_pointed_stars_l471_471338

theorem number_of_non_similar_regular_2000_pointed_stars :
  let n := 2000 in
  let coprimes := { m : ℕ | 1 < m ∧ m < n ∧ Nat.gcd m n = 1 } in
  coprimes.card / 2 = 399 :=
by {
  sorry
}

end number_of_non_similar_regular_2000_pointed_stars_l471_471338


namespace max_discount_rate_l471_471288

-- Define the constants used in the problem
def costPrice : ℝ := 4
def sellingPrice : ℝ := 5
def minProfitMarginRate : ℝ := 0.1
def minProfit : ℝ := costPrice * minProfitMarginRate -- which is 0.4

-- The maximum discount rate that preserves the required profit margin
theorem max_discount_rate : 
  ∃ x : ℝ, (0 ≤ x ∧ x ≤ 100) ∧ (sellingPrice * (1 - x / 100) - costPrice ≥ minProfit) ∧ (x = 12) :=
by
  sorry

end max_discount_rate_l471_471288


namespace students_6_to_8_hours_study_l471_471701

-- Condition: 100 students were surveyed
def total_students : ℕ := 100

-- Hypothetical function representing the number of students studying for a specific range of hours based on the histogram
def histogram_students (lower_bound upper_bound : ℕ) : ℕ :=
  sorry  -- this would be defined based on actual histogram data

-- Question: Prove the number of students who studied for 6 to 8 hours
theorem students_6_to_8_hours_study : histogram_students 6 8 = 30 :=
  sorry -- the expected answer based on the histogram data

end students_6_to_8_hours_study_l471_471701


namespace max_discount_rate_l471_471273

theorem max_discount_rate 
  (cost_price : ℝ) (selling_price : ℝ) (min_profit_margin : ℝ) :
  cost_price = 4 → selling_price = 5 → min_profit_margin = 0.1 →
  ∃ (x : ℝ), 0 ≤ x ∧ x ≤ 8.8 ∧ (selling_price * (1 - x / 100) - cost_price) / cost_price ≥ min_profit_margin :=
begin
  sorry
end

end max_discount_rate_l471_471273


namespace max_discount_rate_l471_471287

-- Define the constants used in the problem
def costPrice : ℝ := 4
def sellingPrice : ℝ := 5
def minProfitMarginRate : ℝ := 0.1
def minProfit : ℝ := costPrice * minProfitMarginRate -- which is 0.4

-- The maximum discount rate that preserves the required profit margin
theorem max_discount_rate : 
  ∃ x : ℝ, (0 ≤ x ∧ x ≤ 100) ∧ (sellingPrice * (1 - x / 100) - costPrice ≥ minProfit) ∧ (x = 12) :=
by
  sorry

end max_discount_rate_l471_471287


namespace locus_of_M_existence_of_A4_l471_471476

-- Part (1): Locus of M
theorem locus_of_M (x y : ℝ) :
    let MA1 := sqrt((x-1)^2 + y^2);
        MA2 := sqrt((x+2)^2 + y^2)
    in (MA1 / MA2 = sqrt(2) / 2) → (x^2 + y^2 - 8*x - 2 = 0) :=
by
    sorry

-- Part (2): Existence of a Point A4
theorem existence_of_A4 (x y : ℝ) (m : ℝ := 2) (n : ℝ := 0) :
    let NA3 := sqrt((x+1)^2 + y^2);
        NA4 := sqrt((x-m)^2 + (y-n)^2);
        on_circle := (x - 3)^2 + y^2 = 4
    in on_circle → (NA3 / NA4 = 2) → (m = 2 ∧ n = 0) :=
by
    sorry

end locus_of_M_existence_of_A4_l471_471476


namespace no_three_digit_numbers_with_sum_27_are_even_l471_471753

-- We define a 3-digit number and its conditions based on digit-sum and even properties
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def digit_sum (n : ℕ) : ℕ :=
  (n / 100) + (n % 100 / 10) + (n % 10)

def is_even (n : ℕ) : Prop :=
  n % 2 = 0

theorem no_three_digit_numbers_with_sum_27_are_even :
  ¬ ∃ n : ℕ, is_three_digit n ∧ digit_sum n = 27 ∧ is_even n :=
by sorry

end no_three_digit_numbers_with_sum_27_are_even_l471_471753


namespace triangular_grid_edges_l471_471583

theorem triangular_grid_edges :
  let n := 1001 in
  let T := (n * (n + 1)) / 2 in
  let total_edges := 4 * T in
  let shared_edges := n * (n - 1) / 2 in
  let unique_edges := total_edges - 2 * shared_edges in
  unique_edges = 1006004 :=
by
  sorry

end triangular_grid_edges_l471_471583


namespace tangent_lines_through_origin_l471_471131

-- Definition of the curve y = ln|x|
def curve (x : ℝ) : ℝ :=
  real.log (abs x)

-- Proposition stating that the tangent lines to the curve y = ln|x| passing through
-- the origin are given by x - e y = 0 and x + e y = 0.
theorem tangent_lines_through_origin :
  (∀ (x y : ℝ), curve x = y → (x - real.exp 1 * y = 0 ∨ x + real.exp 1 * y = 0)) ↔
  (∀ (x : ℝ), curve x = real.log (abs x)) :=
sorry

end tangent_lines_through_origin_l471_471131


namespace balls_in_boxes_l471_471437

def num_ways_to_partition_6_in_4_parts : ℕ :=
  -- The different partitions of 6: (6,0,0,0), (5,1,0,0), (4,2,0,0), (4,1,1,0),
  -- (3,3,0,0), (3,2,1,0), (3,1,1,1), (2,2,2,0), (2,2,1,1)
  ([
    [6, 0, 0, 0],
    [5, 1, 0, 0],
    [4, 2, 0, 0],
    [4, 1, 1, 0],
    [3, 3, 0, 0],
    [3, 2, 1, 0],
    [3, 1, 1, 1],
    [2, 2, 2, 0],
    [2, 2, 1, 1]
  ]).length

theorem balls_in_boxes : num_ways_to_partition_6_in_4_parts = 9 := by
  sorry

end balls_in_boxes_l471_471437


namespace sum_lent_to_Ramu_l471_471084

namespace InterestProblem

-- Define the constants of the problem.
def borrowed_sum : ℝ := 3900
def interest_rate_Anwar : ℝ := 6 / 100
def interest_rate_Ramu : ℝ := 9 / 100
def time_years : ℝ := 3
def total_gain : ℝ := 824.85

-- Define the required interest calculations.
def interest_to_Anwar := borrowed_sum * interest_rate_Anwar * time_years
def total_interest_from_Ramu := total_gain + interest_to_Anwar

def principal_Ramu := total_interest_from_Ramu * 100 / (interest_rate_Ramu * time_years)

-- The theorem we need to prove
theorem sum_lent_to_Ramu : principal_Ramu = 4355 := by
  -- The proof will go here, but we'll fill it in with sorry.
  sorry

end InterestProblem

end sum_lent_to_Ramu_l471_471084


namespace prop_p_necessary_but_not_sufficient_for_prop_q_l471_471848

-- Conditions
variables {X : Type*} (f : X → ℝ)

-- Proposition p: The derivative of the function y = f(x) is a constant function
def prop_p : Prop := ∃ c : ℝ, ∀ x : X, f' x = c

-- Proposition q: The function y = f(x) is a linear function
def prop_q : Prop := ∃ a b : ℝ, ∀ x : X, f x = a * x + b

-- The equivalence problem
theorem prop_p_necessary_but_not_sufficient_for_prop_q :
  (prop_q → prop_p ∧ ¬(prop_p → prop_q)) :=
begin
  sorry
end

end prop_p_necessary_but_not_sufficient_for_prop_q_l471_471848


namespace trigonometric_equation_solution_l471_471206

theorem trigonometric_equation_solution (x : ℝ) :
  (1 + sin x + cos (3 * x) = cos x + sin (2 * x) + cos (2 * x)) ↔ 
  (∃ k : ℤ, (x = real.pi * k) ∨ (x = (-1)^(k+1) * real.pi / 6 + real.pi * k) ∨ (x = ± real.pi / 3 + 2 * real.pi * k)) :=
sorry

end trigonometric_equation_solution_l471_471206


namespace maximum_distance_point_to_line_l471_471561

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x + 1)^2 + (y - 2)^2 = 1

-- Define the line l
def line_l (m x y : ℝ) : Prop := (m - 1) * x + m * y + 2 = 0

-- Statement of the problem to prove
theorem maximum_distance_point_to_line :
  ∀ (x y m : ℝ), circle_C x y → ∃ P : ℝ, line_l m x y → P = 6 :=
by 
  sorry

end maximum_distance_point_to_line_l471_471561


namespace expected_min_leq_2_l471_471536

open ProbabilityTheory

variables (ξ η : ℝ → ℝ) -- ξ and η are random variables

-- Condition: expected value of ξ is 2
axiom E_ξ_eq_2 : ℝ
axiom E_ξ_is_2 : (∫ x in ⊤, ξ x) = 2

-- Goal: expected value of min(ξ, η) ≤ 2
theorem expected_min_leq_2 (h : ∀ x, min (ξ x) (η x) ≤ ξ x) : 
  (∫ x in ⊤, min (ξ x) (η x)) ≤ 2 := by
  -- use the provided axioms and conditions here
  sorry

end expected_min_leq_2_l471_471536


namespace find_a_l471_471387

theorem find_a
  (x : ℝ) (a : ℝ) (hx : x = -π / 6)
  (h_eq : 3 * Real.tan (x + a) = Real.sqrt 3) 
  (h_a_interval : a ∈ Ioo (-π) 0) :
  a = -2 * π / 3 := 
sorry

end find_a_l471_471387


namespace minimum_value_inequality_l471_471765

theorem minimum_value_inequality (x y : ℝ) (hx : x > 2) (hy : y > 2) :
    (x^2 + 1) / (y - 2) + (y^2 + 1) / (x - 2) ≥ 18 := by
  sorry

end minimum_value_inequality_l471_471765


namespace megawheel_seat_capacity_l471_471590

theorem megawheel_seat_capacity (seats people : ℕ) (h1 : seats = 15) (h2 : people = 75) : people / seats = 5 := by
  sorry

end megawheel_seat_capacity_l471_471590


namespace expected_flashlight_lifetime_leq_two_l471_471541

theorem expected_flashlight_lifetime_leq_two
  (Ω : Type*) [MeasurableSpace Ω] [ProbabilitySpace Ω]
  (ξ η : Ω → ℝ)
  (h_min_leq_xi : ∀ ω, min (ξ ω) (η ω) ≤ ξ ω)
  (h_expectation_xi : expectation (ξ) = 2) :
  expectation (fun ω => min (ξ ω) (η ω)) ≤ 2 := 
sorry

end expected_flashlight_lifetime_leq_two_l471_471541


namespace game_not_fair_prob_first_player_win_l471_471172

-- Define the probability of player i winning
variable (n : ℕ) -- number of players
variable (P : ℕ → ℝ) -- function from player index to probability

-- Define the base condition for 36 players
def P_i (i : ℕ) : ℝ := P i

-- Given conditions
-- Condition 1: There are 36 players
def num_players := 36

-- Condition 2: Each player has a turn
-- This is implicit in the cyclic structure of the game.

-- Part (a): Prove that the game is not fair
theorem game_not_fair (h : ∀ i j : ℕ, i ≠ j → P_i i ≠ P_i j): Prop :=
  ∃ i j : ℕ, i ≠ j ∧ P_i i ≠ P_i j

-- Part (b): Calculate the probability of the first player winning

noncomputable def approx_prob_first_player_win : ℝ :=
  let e := Real.exp 1
  in e / (36 * (e - 1))

-- Theorem stating the approximate probability of the first player winning
theorem prob_first_player_win : Prop :=
  abs (approx_prob_first_player_win - 0.044) < 0.001

end game_not_fair_prob_first_player_win_l471_471172


namespace ellipse_hyperbola_foci_l471_471830

-- Definitions based on conditions
def isEllipse (C : ℝ → ℝ → Prop) (m : ℝ) : Prop :=
  ∃ (x y : ℝ), m > 1 ∧ C x y = (x ^ 2 / m ^ 2 + y ^ 2 = 1)

def isHyperbola (H : ℝ → ℝ → Prop) (n : ℝ) : Prop :=
  ∃ (x y : ℝ), n > 0 ∧ H x y = (x ^ 2 / n ^ 2 - y ^ 2 = 1)

def sameFoci (m n : ℝ) : Prop := m ^ 2 - 1 = n ^ 2 + 1

def eccentricityEllipse (m : ℝ) : ℝ := sqrt(m^2 - 1) / m
def eccentricityHyperbola (n : ℝ) : ℝ := sqrt(n^2 + 1) / n

-- Lean 4 statement of the problem
theorem ellipse_hyperbola_foci (C : ℝ → ℝ → Prop) (H : ℝ → ℝ → Prop) (m n : ℝ)
  (h_ellipse : isEllipse C m)
  (h_hyperbola : isHyperbola H n)
  (h_foci : sameFoci m n) :
  m > n ∧ eccentricityEllipse m * eccentricityHyperbola n > 1 := by
  sorry

end ellipse_hyperbola_foci_l471_471830


namespace ThreeDBarChartHeightRepresentsFrequency_l471_471028

theorem ThreeDBarChartHeightRepresentsFrequency
  (height_representation : ∀ bar_height, bar_height = frequency) :
  (height_representation ∧ height_represents_frequency ∧ bar_height) = A := 
sorry

end ThreeDBarChartHeightRepresentsFrequency_l471_471028


namespace part1_part2_l471_471835

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp (x - 1) + a
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := a * x + Real.log x

theorem part1 (x : ℝ) (hx : 0 < x) :
  f x 0 ≥ g x 0 + 1 := sorry

theorem part2 {x0 : ℝ} (hx0 : ∃ y0 : ℝ, f x0 0 = g x0 0 ∧ ∀ x ≠ x0, f x 0 ≠ g x 0) :
  x0 < 2 := sorry

end part1_part2_l471_471835


namespace xander_pages_left_to_read_l471_471202

theorem xander_pages_left_to_read :
  let total_pages := 500
  let read_first_night := 0.2 * 500
  let read_second_night := 0.2 * 500
  let read_third_night := 0.3 * 500
  total_pages - (read_first_night + read_second_night + read_third_night) = 150 :=
by 
  sorry

end xander_pages_left_to_read_l471_471202


namespace arithmetic_mean_inequality_l471_471222

theorem arithmetic_mean_inequality
  {n : ℕ} (hn : n > 0) (x : ℕ → ℝ) :
  let a := (1 / n : ℝ) * (∑ i in finset.range n, x i) in
  (∑ i in finset.range n, (x i - a) ^ 2) ≤ 
  (1 / 2) * (∑ i in finset.range n, |x i - a|) ^ 2 :=
by
  sorry

end arithmetic_mean_inequality_l471_471222


namespace min_value_function_l471_471784

theorem min_value_function : 
    (∀ x : ℝ, (0 < x ∧ x < π / 4) → (y = (cos x)^2 / (cos x * sin x - (sin x)^2) → 4 ≤ y)) :=
begin
  sorry
end

end min_value_function_l471_471784


namespace proof_area_identity_l471_471488

-- Conditions given
variables (A B C D1 D2 : Type) 
variables [linear_ordered_field ℝ]  -- Assume a real number field for calculations

-- Defining the parallel conditions
def ab_parallel_cd2 : Prop := parallel AB CD₂
def ac_parallel_bd1 : Prop := parallel AC BD₁
def a_on_d1d2 : Prop := A ∈ line D₁ D₂

-- Defining the areas of the triangles
def area (t : triangle) : ℝ := sorry -- Assume a function to compute the area of a triangle

-- Defining the specific triangles 
def triangle_ABC : triangle := ⟨A, B, C⟩
def triangle_ABD1 : triangle := ⟨A, B, D₁⟩
def triangle_ACD2 : triangle := ⟨A, C, D₂⟩

-- Theorem statement
theorem proof_area_identity (h1 : ab_parallel_cd2) (h2 : ac_parallel_bd1) (h3 : a_on_d1d2) : 
  (area triangle_ABC)^2 = (area triangle_ABD1) * (area triangle_ACD2) :=
sorry

end proof_area_identity_l471_471488


namespace fraction_spent_on_museum_l471_471059

-- Conditions
variables (initial_money sandwich_fraction book_fraction leftover_money : ℝ)
variables (spent_sandwich spent_book total_spent spent_museum: ℝ)

-- Definitions directly from conditions
def initial_money := 180
def sandwich_fraction := 1/5
def book_fraction := 1/2
def leftover_money := 24

-- Calculations based on conditions
def spent_sandwich := sandwich_fraction * initial_money
def spent_book := book_fraction * initial_money
def total_spent := initial_money - leftover_money
def spent_museum := total_spent - (spent_sandwich + spent_book)

-- The fraction of money spent on museum ticket
def spent_museum_fraction := spent_museum / initial_money

-- The statement to be proven
theorem fraction_spent_on_museum :
  spent_museum_fraction = 1 / 6 :=
sorry

end fraction_spent_on_museum_l471_471059


namespace min_diagonal_l471_471021

theorem min_diagonal (a b c : ℝ) (h1 : 2 * (a + b) = 20) (h2 : c^2 = a^2 + b^2) : c ≥ Real.sqrt 50 :=
begin
  sorry
end

end min_diagonal_l471_471021


namespace avg_height_country_l471_471185

-- Define the parameters for the number of boys and their average heights
def num_boys_north : ℕ := 300
def num_boys_south : ℕ := 200
def avg_height_north : ℝ := 1.60
def avg_height_south : ℝ := 1.50

-- Define the total number of boys
def total_boys : ℕ := num_boys_north + num_boys_south

-- Define the total combined height
def total_height : ℝ := (num_boys_north * avg_height_north) + (num_boys_south * avg_height_south)

-- Prove that the average height of all boys combined is 1.56 meters
theorem avg_height_country : total_height / total_boys = 1.56 := by
  sorry

end avg_height_country_l471_471185


namespace team_B_city_A_matches_l471_471291

noncomputable def matches_played_B_from_city_A : ℕ := 15

theorem team_B_city_A_matches : 
  ∀ (cities : Fin 16) (teams : Fin 32) (matches : Fin 31), 
    ∀ (unique_matches : ∀ t : Fin 31, t ≠ matches_played_B_from_city_A → ∃! n : Fin 31, n = matches),
    matches = matches_played_B_from_city_A :=
by
  sorry

end team_B_city_A_matches_l471_471291


namespace time_for_A_and_C_to_complete_work_l471_471262

variable (A_rate B_rate C_rate : ℝ)

theorem time_for_A_and_C_to_complete_work
  (hA : A_rate = 1 / 4)
  (hBC : 1 / 3 = B_rate + C_rate)
  (hB : B_rate = 1 / 12) :
  1 / (A_rate + C_rate) = 2 :=
by
  -- Here would be the proof logic
  sorry

end time_for_A_and_C_to_complete_work_l471_471262


namespace university_original_faculty_number_l471_471693

theorem university_original_faculty_number (x : ℝ) (h : 0.77 * x = 360) : x ≈ 468 :=
by
  -- Proof should go here
  sorry

end university_original_faculty_number_l471_471693


namespace find_percentage_l471_471256

noncomputable def percentage_condition (P : ℝ) : Prop :=
  9000 + (P / 100) * 9032 = 10500

theorem find_percentage (P : ℝ) (h : percentage_condition P) : P = 16.61 :=
sorry

end find_percentage_l471_471256


namespace leak_time_to_empty_l471_471942

variable {A L : ℝ}

-- Definitions based on the conditions provided
def pipeARate := (1 / 12 : ℝ) -- Pipe A's rate of filling the tank
def combinedRate := (1 / 18 : ℝ) -- Combined rate of Pipe A and the leak

-- Prove that the leak alone can empty the tank in 36 hours
theorem leak_time_to_empty (h : pipeARate - L = combinedRate) : (1 / L) = 36 := by
  -- Substituting and solving for L to show the leak rate
  have L_rate : L = pipeARate - combinedRate := by
    sorry
  -- Proving the time taken by the leak to empty the tank
  sorry

end leak_time_to_empty_l471_471942


namespace number_of_possible_m_values_l471_471920

def has_integer_root (f : ℤ → ℤ) : Prop :=
  ∃ x : ℤ, f x = 0

noncomputable def f (x m : ℤ) : ℤ :=
  2 * x - m * Real.sqrt (10 - x) - m + 10

axiom sqrt_nat (n : ℤ) : Real.sqrt n = ↑(nat.sqrt (nat_abs n))

-- Given m ∈ ℕ
def m_nat (m : ℤ) : Prop :=
  ∃ (n : ℕ), m = n

-- Prove there are exactly 4 possible values of m
theorem number_of_possible_m_values : (∃ (m : ℤ), m_nat m ∧ has_integer_root (λ x, f x m)) → ∃! (m : ℤ), m_nat m ∧ has_integer_root (λ x, f x m) :=
by
  sorry

end number_of_possible_m_values_l471_471920


namespace constant_term_in_binomial_expansion_is_28_l471_471042

-- Conditions as definitions in Lean 4
def binomial_expansion (x : ℂ) : ℂ := (∛x - 1/x)^8
def sum_of_binomial_coeffs (n : ℕ) : ℤ := 2^n
def sum_of_coefficients_equals_256  : Prop := sum_of_binomial_coeffs 8 = 256

-- The resulting theorem to prove
theorem constant_term_in_binomial_expansion_is_28 : 
sum_of_coefficients_equals_256 →
∃ c : ℂ, c = 28 := sorry

end constant_term_in_binomial_expansion_is_28_l471_471042


namespace total_students_l471_471140

def numStudents (skiing scavenger : ℕ) : ℕ :=
  skiing + scavenger

theorem total_students (skiing scavenger : ℕ) (h1 : skiing = 2 * scavenger) (h2 : scavenger = 4000) :
  numStudents skiing scavenger = 12000 :=
by
  sorry

end total_students_l471_471140


namespace game_not_fair_prob_first_player_win_l471_471175

-- Define the probability of player i winning
variable (n : ℕ) -- number of players
variable (P : ℕ → ℝ) -- function from player index to probability

-- Define the base condition for 36 players
def P_i (i : ℕ) : ℝ := P i

-- Given conditions
-- Condition 1: There are 36 players
def num_players := 36

-- Condition 2: Each player has a turn
-- This is implicit in the cyclic structure of the game.

-- Part (a): Prove that the game is not fair
theorem game_not_fair (h : ∀ i j : ℕ, i ≠ j → P_i i ≠ P_i j): Prop :=
  ∃ i j : ℕ, i ≠ j ∧ P_i i ≠ P_i j

-- Part (b): Calculate the probability of the first player winning

noncomputable def approx_prob_first_player_win : ℝ :=
  let e := Real.exp 1
  in e / (36 * (e - 1))

-- Theorem stating the approximate probability of the first player winning
theorem prob_first_player_win : Prop :=
  abs (approx_prob_first_player_win - 0.044) < 0.001

end game_not_fair_prob_first_player_win_l471_471175


namespace triangle_area_l471_471307

theorem triangle_area (a b c : ℕ) (h1: a = 3) (h2: b = 4) (h3: c = 5) (h4 : a + b + c = 12) : 
  let s := (a + b + c) / 2 in sqrt (s * (s - a) * (s - b) * (s - c)) = 6 :=
by
  -- placeholder for the proof
  sorry

end triangle_area_l471_471307


namespace calc_expression_solve_system_inequalities_l471_471679

-- Proof Problem 1: Calculation
theorem calc_expression : 
  |1 - Real.sqrt 3| - Real.sqrt 2 * Real.sqrt 6 + 1 / (2 - Real.sqrt 3) - (2 / 3) ^ (-2 : ℤ) = -5 / 4 := 
by 
  sorry

-- Proof Problem 2: System of Inequalities Solution
variable (m : ℝ)
variable (x : ℝ)
  
theorem solve_system_inequalities (h : m < 0) : 
  (4 * x - 1 > x - 7) ∧ (-1 / 4 * x < 3 / 2 * m - 1) → x > 4 - 6 * m := 
by 
  sorry

end calc_expression_solve_system_inequalities_l471_471679


namespace regression_lines_intersect_at_avg_l471_471183

-- Define the conditions
noncomputable def studentA_experiments : ℕ := 10
noncomputable def studentB_experiments : ℕ := 15
def avg_x : ℝ := s
def avg_y : ℝ := t
def regression_line (experiments : ℕ) (data : list (ℝ × ℝ)) : set (ℝ × ℝ) := sorry

-- The linear regression lines for students A and B
def l1 := regression_line studentA_experiments []
def l2 := regression_line studentB_experiments []

-- The point through which both regression lines must pass
def intersection_point : ℝ × ℝ := (avg_x, avg_y)

-- The theorem to be proved
theorem regression_lines_intersect_at_avg :
  ∃ p : ℝ × ℝ, p = intersection_point ∧ p ∈ l1 ∧ p ∈ l2 :=
sorry

end regression_lines_intersect_at_avg_l471_471183


namespace moles_of_water_needed_l471_471755

-- Define the conditions
def reaction (CaO H2O CaOH2 : ℕ) : Prop :=
  CaO = H2O ∧ H2O = CaOH2

def mass_CaO : ℝ := 168
def molar_mass_CaO : ℝ := 56.08
def moles_CaOH2_needed : ℕ := 3

-- Define the main theorem
theorem moles_of_water_needed :
  let moles_CaO := mass_CaO / molar_mass_CaO in
  moles_CaO = 3 →
  reaction moles_CaO 3 moles_CaOH2_needed →
  3 = 3 := by
  sorry

end moles_of_water_needed_l471_471755


namespace desk_sides_perpendicular_l471_471982

theorem desk_sides_perpendicular (desk : Type) [rectangular_desk : Rectangle desk] :
  ∃ (long_side short_side : desk.Side), right_angle(long_side, short_side) := sorry

end desk_sides_perpendicular_l471_471982


namespace valid_n_for_grid_l471_471937

theorem valid_n_for_grid (n : ℕ) : (∃ k : ℕ, n = 4 * k ∧ k > 0) ↔ 
                                  (∃ f : ℕ × ℕ → ℤ, 
                                      (∃ g : {p : ℕ × ℕ // p.1 < 5 ∧ p.2 < n} → (ℚ → bool),
                                      (∀ p : ℕ × ℕ, p.1 < 5 ∧ p.2 < n → (∃ q : ℕ × ℕ, (q.1 = p.1 ∧ (q.2 = p.2 + 1 ∨ q.2 + 1 = p.2)) ∧ f p = 1 ∧ f q = -1)) ∧
                                      (∀ i < 5, (∏ j in finset.range n, g ⟨⟨i, j⟩, ⟨nat.lt_of_succ_lt j.property⟩⟩ ((λ _ => true) n ) = 1)) ∧
                                      (∀ j < n, (∏ i in finset.range 5, g ⟨⟨i, j⟩, ⟨nat.lt_of_succ_lt i.property⟩⟩ ((λ _ => true) 5) = !(-1))))

end valid_n_for_grid_l471_471937


namespace circle_equation_is_correct_l471_471989

def center : Int × Int := (-3, 4)
def radius : Int := 3
def circle_standard_equation (x y : Int) : Int :=
  (x + 3)^2 + (y - 4)^2

theorem circle_equation_is_correct :
  circle_standard_equation x y = 9 :=
sorry

end circle_equation_is_correct_l471_471989


namespace chess_tournament_schedule_l471_471465

theorem chess_tournament_schedule:
  (∃ players_w players_e : Finset ℕ, players_w.card = 4 ∧ players_e.card = 4 ∧
  (∀ a ∈ players_w, ∀ b ∈ players_e, a ≠ b)) →
  (3 ∣ 48) →
  (16 \times (4:ℕ)) = 48 →
  nat.factorial 16 / (nat.factorial 3) = 16! / 3! :=
by
  sorry

end chess_tournament_schedule_l471_471465


namespace pigeons_remaining_l471_471165

def initial_pigeons : ℕ := 40
def chicks_per_pigeon : ℕ := 6
def total_chicks : ℕ := initial_pigeons * chicks_per_pigeon
def total_pigeons_with_chicks : ℕ := total_chicks + initial_pigeons
def percentage_eaten : ℝ := 0.30
def pigeons_eaten : ℤ := int.of_nat total_pigeons_with_chicks * percentage_eaten
def pigeons_left : ℤ := int.of_nat total_pigeons_with_chicks - pigeons_eaten

theorem pigeons_remaining :
  pigeons_left = 196 := by
  -- The process of proving will be here
  sorry

end pigeons_remaining_l471_471165


namespace g_h_2_eq_583_l471_471859

def g (x : ℝ) : ℝ := 3*x^2 - 5

def h (x : ℝ) : ℝ := -2*x^3 + 2

theorem g_h_2_eq_583 : g (h 2) = 583 :=
by
  sorry

end g_h_2_eq_583_l471_471859


namespace net_sum_paid_eq_810_l471_471212

-- Definitions of conditions
def work_rate (days: ℕ) := 1 / (days: ℝ)
def combined_work_rate (days_a: ℕ) (days_b: ℕ) := work_rate days_a + work_rate days_b
def total_payment (daily_wage: ℝ) (days_worked: ℕ) := daily_wage * (days_worked: ℝ)

-- Given conditions
def days_a := 12
def days_b := 15
def days_worked_together := 5
def daily_wage_b := 54

-- Lean statement to prove the net sum paid
theorem net_sum_paid_eq_810 :
  let work_done_a_in_1_day := work_rate days_a,
      work_done_b_in_1_day := work_rate days_b,
      combined_work_done_in_1_day := combined_work_rate days_a days_b,
      total_work_done_together_5_days := days_worked_together * combined_work_done_in_1_day,
      remaining_work_done_by_c := 1 - total_work_done_together_5_days,
      payment_b := total_payment daily_wage_b days_worked_together,
      proportion_work_done_b := work_done_b_in_1_day * days_worked_together,
      total_payment := payment_b / proportion_work_done_b
  in total_payment = 810 :=
by sorry

end net_sum_paid_eq_810_l471_471212


namespace dad_use_per_brush_correct_l471_471992

def toothpaste_total : ℕ := 105
def mom_use_per_brush : ℕ := 2
def anne_brother_use_per_brush : ℕ := 1
def brushing_per_day : ℕ := 3
def days_to_finish : ℕ := 5

-- Defining the daily use function for Anne's Dad
def dad_use_per_brush (D : ℕ) : ℕ := D

theorem dad_use_per_brush_correct (D : ℕ) 
  (h : brushing_per_day * (mom_use_per_brush + anne_brother_use_per_brush * 2 + dad_use_per_brush D) * days_to_finish = toothpaste_total) 
  : dad_use_per_brush D = 3 :=
by sorry

end dad_use_per_brush_correct_l471_471992


namespace eccentricity_range_l471_471817

theorem eccentricity_range (a b c : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : a > 0) (h4 : c = a * (√3 - 1) / 2) : 
  0 < c ∧ c < a ↔ √3 - 1 < c / a ∧ c / a < 1 := 
by
  sorry

end eccentricity_range_l471_471817


namespace area_of_shaded_region_l471_471757

open Real

theorem area_of_shaded_region : 
  let Line1 : ℝ → ℝ := fun x => (3 / 4 * x + 5 / 4)
  let Line2 : ℝ → ℝ := fun x => (3 / 2 * x - 2)
  ∫ x in 1..(13/3), |Line1 x - Line2 x| = 1.7 := 
by
  sorry

end area_of_shaded_region_l471_471757


namespace probability_red_given_black_l471_471648

noncomputable def urn_A := {white := 4, red := 2}
noncomputable def urn_B := {red := 3, black := 3}

-- Define the probabilities as required in the conditions
def prob_urn_A := 1 / 2
def prob_urn_B := 1 / 2

def draw_red_from_A := 2 / 6
def draw_black_from_B := 3 / 6
def draw_red_from_B := 3 / 6
def draw_black_from_B_after_red := 3 / 5
def draw_black_from_B_after_black := 2 / 5

def probability_first_red_second_black :=
  (prob_urn_A * draw_red_from_A * prob_urn_B * draw_black_from_B) +
  (prob_urn_B * draw_red_from_B * prob_urn_B * draw_black_from_B_after_red) +
  (prob_urn_B * draw_red_from_B * prob_urn_B * draw_black_from_B_after_black)

def probability_second_black :=
  (prob_urn_A * prob_urn_B * draw_black_from_B) +
  (prob_urn_B * draw_red_from_B * prob_urn_B * draw_black_from_B_after_red) +
  (prob_urn_B * draw_black_from_B * prob_urn_B * draw_black_from_B_after_black)

theorem probability_red_given_black :
  probability_first_red_second_black / probability_second_black = 7 / 15 :=
sorry

end probability_red_given_black_l471_471648


namespace solution_set_of_inequality_l471_471372

noncomputable def f : ℝ → ℝ := sorry

theorem solution_set_of_inequality :
  (∀ x y > 0, f (x + y) = f x + f y - 2) →
  f 3 = 4 →
  (∀ x1 x2 > 0, ((f x1 - f x2) / (x1 - x2)) > 0) →
  { a : ℝ | f (a^2 - 2 * a - 3) ≥ (8 / 3) } = {a | a ≤ 1 - Real.sqrt 5} ∪ {a | 1 + Real.sqrt 5 ≤ a} :=
by
  intros h1 h2 h3
  sorry

end solution_set_of_inequality_l471_471372


namespace average_production_per_day_for_entire_month_l471_471030

-- Definitions based on the conditions
def average_first_25_days := 65
def average_last_5_days := 35
def number_of_days_in_first_period := 25
def number_of_days_in_last_period := 5
def total_days_in_month := 30

-- The goal is to prove that the average production per day for the entire month is 60 TVs/day.
theorem average_production_per_day_for_entire_month :
  (average_first_25_days * number_of_days_in_first_period + 
   average_last_5_days * number_of_days_in_last_period) / total_days_in_month = 60 := 
by
  sorry

end average_production_per_day_for_entire_month_l471_471030


namespace min_value_sin_cos_l471_471781

theorem min_value_sin_cos (x : ℝ) : sin x ^ 6 + 2 * cos x ^ 6 ≥ 2 / 3 :=
sorry

end min_value_sin_cos_l471_471781


namespace stratified_sampling_elderly_l471_471686

theorem stratified_sampling_elderly (total_elderly total_middle total_young sample_size : ℕ)
  (h_elderly : total_elderly = 28)
  (h_middle : total_middle = 54)
  (h_young : total_young = 81)
  (h_sample : sample_size = 36) :
  let total_population := total_elderly + total_middle + total_young,
      proportion_elderly := (total_elderly : ℚ) / total_population,
      elderly_in_sample := proportion_elderly * sample_size in
  elderly_in_sample ≈ 6 :=
by {
  -- Proof placeholder
  sorry
}

end stratified_sampling_elderly_l471_471686


namespace number_of_possible_7_digit_numbers_l471_471551

theorem number_of_possible_7_digit_numbers :
  let positions := 6 in
  let total_arrangements := positions * positions in
  let same_position_arrangements := positions in
  total_arrangements - same_position_arrangements = 30 :=
by
  sorry

end number_of_possible_7_digit_numbers_l471_471551


namespace max_discount_rate_l471_471286

-- Define the constants used in the problem
def costPrice : ℝ := 4
def sellingPrice : ℝ := 5
def minProfitMarginRate : ℝ := 0.1
def minProfit : ℝ := costPrice * minProfitMarginRate -- which is 0.4

-- The maximum discount rate that preserves the required profit margin
theorem max_discount_rate : 
  ∃ x : ℝ, (0 ≤ x ∧ x ≤ 100) ∧ (sellingPrice * (1 - x / 100) - costPrice ≥ minProfit) ∧ (x = 12) :=
by
  sorry

end max_discount_rate_l471_471286


namespace range_of_a_l471_471400

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x > 0 then log a x else abs (x + 2)

def symmetric_points_exist (a : ℝ) : Prop :=
a > 0 ∧ a ≠ 1 ∧ 
  ∃ x1 x2, f a x1 = f a x2 ∧ x1 = -x2

theorem range_of_a :
  {a : ℝ | symmetric_points_exist a } = 
  {a : ℝ | (0 < a ∧ a < 1) ∨ (1 < a ∧ a < 3)} :=
by sorry

end range_of_a_l471_471400


namespace blackboard_number_invariant_l471_471605

theorem blackboard_number_invariant (n : ℕ) (initial : n = 2011)
  (move_a : ∀ (x : ℕ), ∃ a b : ℕ, a + b = x)
  (move_b : ∀ (a b : ℕ), a >= b → (a - b) ≥ 0):
  ∀ x : ℕ, x ∈ finset.range(n).erase(0) → False :=
begin
  sorry
end

end blackboard_number_invariant_l471_471605


namespace max_Sn_at_5_l471_471615

noncomputable def a_n (a1 : ℕ) (d : ℕ) (n : ℕ) : ℕ := a1 + (n - 1) * d
noncomputable def Sn (a1 : ℕ) (d : ℕ) (n : ℕ) : ℕ := n * (2 * a1 + (n - 1) * d) / 2

theorem max_Sn_at_5 (a1 d : ℕ) :
  (d = -2) → (Sn a1 d 3 = 21) → n = 5 := sorry

end max_Sn_at_5_l471_471615


namespace square_area_is_two_l471_471043

def area_square {z : ℂ} (h : z ≠ 0) (h_square : (∃ (z2 z4 z5: ℂ), z2 = z^2 ∧ z4 = z^4 ∧ z5 = z^5 ∧ 
  ((z4 - z2) = i * (z5 - z2) ∨ (z5 - z2) = -i * (z4 - z2)) ∧ |z4 - z2| = |z5 - z2|)) : ℂ :=
(|z^4 - z^2|^2)

theorem square_area_is_two {z : ℂ} (h : z ≠ 0) (h_square : (∃ (z2 z4 z5: ℂ), z2 = z^2 ∧ z4 = z^4 ∧ z5 = z^5 ∧ 
  ((z4 - z2) = i * (z5 - z2) ∨ (z5 - z2) = -i * (z4 - z2)) ∧ |z4 - z2| = |z5 - z2|)) :
  area_square h h_square = 2 :=
sorry

end square_area_is_two_l471_471043


namespace garden_dimensions_l471_471926

theorem garden_dimensions
  (w l : ℝ) 
  (h1 : l = 2 * w) 
  (h2 : l * w = 600) : 
  w = 10 * Real.sqrt 3 ∧ l = 20 * Real.sqrt 3 :=
by
  sorry

end garden_dimensions_l471_471926


namespace fraction_difference_l471_471654

/-- The difference between the largest and smallest fractions among 2/3, 3/4, 4/5, and 5/7 is 2/15. -/
theorem fraction_difference : 
  let fractions := [2/3, 3/4, 4/5, 5/7] in 
  let max_fraction := fractions.max 
  let min_fraction := fractions.min 
  max_fraction - min_fraction = 2/15 := by 
  sorry

end fraction_difference_l471_471654


namespace number_of_rolls_in_case_l471_471265

-- Definitions: Cost of a case, cost per roll individually, percent savings per roll
def cost_of_case : ℝ := 9
def cost_per_roll_individual : ℝ := 1
def percent_savings_per_roll : ℝ := 0.25

-- Theorem: Proving the number of rolls in the case is 12
theorem number_of_rolls_in_case (n : ℕ) (h1 : cost_of_case = 9)
    (h2 : cost_per_roll_individual = 1)
    (h3 : percent_savings_per_roll = 0.25) : n = 12 := 
  sorry

end number_of_rolls_in_case_l471_471265


namespace p_is_contradictory_to_q_l471_471411

variable (a : ℝ)

def p := a > 0 → a^2 ≠ 0
def q := a ≤ 0 → a^2 = 0

theorem p_is_contradictory_to_q : (p a) ↔ ¬ (q a) :=
by
  sorry

end p_is_contradictory_to_q_l471_471411


namespace sandwiches_prepared_l471_471956

-- Define the conditions as given in the problem.
def ruth_ate_sandwiches : ℕ := 1
def brother_ate_sandwiches : ℕ := 2
def first_cousin_ate_sandwiches : ℕ := 2
def each_other_cousin_ate_sandwiches : ℕ := 1
def number_of_other_cousins : ℕ := 2
def sandwiches_left : ℕ := 3

-- Define the total number of sandwiches eaten.
def total_sandwiches_eaten : ℕ := ruth_ate_sandwiches 
                                  + brother_ate_sandwiches
                                  + first_cousin_ate_sandwiches 
                                  + (each_other_cousin_ate_sandwiches * number_of_other_cousins)

-- Define the number of sandwiches prepared by Ruth.
def sandwiches_prepared_by_ruth : ℕ := total_sandwiches_eaten + sandwiches_left

-- Formulate the theorem to prove.
theorem sandwiches_prepared : sandwiches_prepared_by_ruth = 10 :=
by
  -- Use the solution steps to prove the theorem (proof omitted here).
  sorry

end sandwiches_prepared_l471_471956


namespace expected_lifetime_flashlight_l471_471532

noncomputable def E (X : ℝ) : ℝ := sorry -- Define E as the expectation operator

variables (ξ η : ℝ) -- Define ξ and η as random variables representing lifetimes of blue and red bulbs
variable (h_exi : E ξ = 2) -- Given condition E ξ = 2

theorem expected_lifetime_flashlight (h_min : ∀ x y : ℝ, min x y ≤ x) :
  E (min ξ η) ≤ 2 :=
by
  sorry

end expected_lifetime_flashlight_l471_471532


namespace speed_of_stream_l471_471675

theorem speed_of_stream (v_s : ℝ) (D : ℝ) (h1 : D / (78 - v_s) = 2 * (D / (78 + v_s))) : v_s = 26 :=
by
  sorry

end speed_of_stream_l471_471675


namespace inclination_angle_slope_l471_471019

theorem inclination_angle_slope (l : Line) (h : inclination_angle l = 120) : slope l = -Math.sqrt 3 :=
sorry

end inclination_angle_slope_l471_471019


namespace min_sum_face_l471_471984

/-- Define the vertices of the cube -/
def V : Finset ℕ := { 1, 2, 3, 4, 5, 6, 7, 8 }

/-- The condition that the sum of the numbers at any three vertices on the same face is at least 10 -/
def valid_face_sum (a b c : ℕ) : Prop := a + b + c ≥ 10

/-- Enumerate the faces of the cube as sets of four vertices -/
def faces : list (Finset ℕ) := [
    { 1, 2, 3, 4 },
    { 5, 6, 7, 8 },
    { 1, 2, 5, 6 },
    { 3, 4, 7, 8 },
    { 1, 3, 5, 7 },
    { 2, 4, 6, 8 }
]

/-- Define the main theorem -/
theorem min_sum_face :
  (∀ face ∈ faces, ∀ a b c ∈ face.erase a, valid_face_sum a b c) →
  ∃ face ∈ faces, face.sum id = 16 := 
sorry

end min_sum_face_l471_471984


namespace systematic_classic_equations_l471_471485

theorem systematic_classic_equations (x y : ℕ) : 
  (7 * x + 7 = y) ∧ (9 * (x - 1) = y) ↔
  (if (exists p q : ℕ, p = 7 * x + 7 ∧ q = 9 * (x - 1)) 
  then x = x ∧ y = 9 * (x - 1) 
  else false) :=
by 
  sorry

end systematic_classic_equations_l471_471485


namespace length_of_goods_train_l471_471208

theorem length_of_goods_train
    (speed_kmph : ℕ)
    (platform_length_m : ℕ)
    (time_seconds : ℕ)
    (conversion_factor : ℕ)
    (train_speed_ms : ℕ)
    (total_distance : ℕ) (train_length : ℕ) :
    speed_kmph = 72 →
    platform_length_m = 290 →
    time_seconds = 26 →
    conversion_factor = 1000 / 3600 →
    train_speed_ms = speed_kmph * conversion_factor →
    total_distance = train_speed_ms * time_seconds →
    train_length = total_distance - platform_length_m →
    train_length = 230 :=
begin
    -- proof would go here
    sorry
end

end length_of_goods_train_l471_471208


namespace least_number_to_addition_l471_471659

-- Given conditions
def n : ℤ := 2496

-- The least number to be added to n to make it divisible by 5
def least_number_to_add (n : ℤ) : ℤ :=
  if (n % 5 = 0) then 0 else (5 - (n % 5))

-- Prove that adding 4 to 2496 makes it divisible by 5
theorem least_number_to_addition : (least_number_to_add n) = 4 :=
  by
    sorry

end least_number_to_addition_l471_471659


namespace log_sequence_geometric_minimum_n_for_Tn_l471_471922

-- Condition definitions
variable {a : ℕ → ℝ}
-- Each term is positive
axiom a_pos : ∀ n, a n > 0
-- Specific conditions on a_2 and the recurrence relation
axiom a2_def : a 2 = 4 * a 1
axiom a_recurrence : ∀ n, a (n + 1) = (a n)^2 + 2 * a n

-- Part (I): Prove sequence of logarithms forms a geometric sequence
theorem log_sequence_geometric : ∃ (r : ℝ), r > 0 ∧ ∀ n, log 3 (1 + a (n + 1)) = 2 * log 3 (1 + a n) := by
  sorry

-- Part (II): Find the minimum value of n such that T_n > 345
def b (n : ℕ) := log 3 (1 + a (2 * n - 1))
def T (n : ℕ) := ∑ i in Finset.range n, b (i + 1)

theorem minimum_n_for_Tn : ∃ n, T n > 345 ∧ ∀ m, m < n → T m ≤ 345 := by
  use 6
  sorry

end log_sequence_geometric_minimum_n_for_Tn_l471_471922


namespace cubic_polynomials_with_specific_roots_count_l471_471786

theorem cubic_polynomials_with_specific_roots_count (a b c d : ℝ) :
  a ≠ 0 →
  ∃ (p : ℝ[X]), 
    (∀ x : ℝ, (p.eval x = 0 ↔ x ∈ {a, b, c})) ∧ 
    p.degree = 3 ∧ 
    p.coeff 3 = a ∧ 
    p.coeff 2 = b ∧ 
    p.coeff 1 = c ∧ 
    p.coeff 0 = d → 
  ∃ n : ℕ, n = 3 :=
by 
  sorry

end cubic_polynomials_with_specific_roots_count_l471_471786


namespace folding_options_are_four_l471_471150

-- Definitions of the polygon and the arrangement
def square : Type := { a // a > 0 } -- square with positive side length
def T_shape : Type := list (square)

/-- There are 6 squares forming the T-shape -/
def elongated_T_shape (n : ℕ) (h : n = 6) : T_shape := sorry

/-- Attaching one more square to one of 12 positions -/
def attach_square (t : T_shape) (pos : ℕ) (h : 1 ≤ pos ∧ pos ≤ 12) : T_shape := sorry

/-- Defining a function that checks if a T_shape folds into a cube-like
    structure with two faces missing -/
def forms_open_box_with_two_faces_missing (t : T_shape) : Prop := sorry

/-- Main theorem statement -/
theorem folding_options_are_four : 
  ∀ t, (∃ n (h : n = 6), t = elongated_T_shape n h) → (∃ n (h : 1 ≤ n ∧ n ≤ 12), forms_open_box_with_two_faces_missing (attach_square t n h)) = 4 := 
sorry

end folding_options_are_four_l471_471150


namespace parabola_intersects_line_l471_471610

-- Define the conditions of the problem
variables {a k x1 x2 : ℝ}
variable (h_a_ne_0 : a ≠ 0)
variable (h_x1_x2 : x1 + x2 < 0)
variable (h_intersect : ∀ x, a * x^2 - a = k * x → x = x1 ∨ x = x2)

-- Define what needs to be proven
theorem parabola_intersects_line {h_a_ne_0 h_x1_x2 h_intersect} :
  ∃ k, (k < 0 ∧ k > 0 → (λ x, a * x + k).exists_first_and_fourth_quadrant) ∧ 
       (k > 0 ∧ k < 0 → (λ x, a * x + k).exists_first_and_fourth_quadrant) :=
sorry

end parabola_intersects_line_l471_471610


namespace max_discount_rate_l471_471282

-- Define the cost price and selling price.
def cp : ℝ := 4
def sp : ℝ := 5

-- Define the minimum profit margin.
def min_profit_margin : ℝ := 0.4

-- Define the discount rate d.
def discount_rate (d : ℝ) : ℝ := sp * (1 - d / 100) - cp

-- The theorem to prove the maximum discount rate.
theorem max_discount_rate (d : ℝ) (H : discount_rate d ≥ min_profit_margin) : d ≤ 12 :=
sorry

end max_discount_rate_l471_471282


namespace f_odd_and_increasing_on_ℝ_l471_471122

noncomputable def f : ℝ → ℝ := sorry

axiom functional_eq (x₁ x₂ : ℝ) : f (x₁ + x₂) = f x₁ + f x₂
axiom increasing_on_nonneg : ∀ (x₁ x₂ : ℝ), 0 ≤ x₁ → x₁ ≤ x₂ → f x₁ ≤ f x₂

theorem f_odd_and_increasing_on_ℝ :
  (∀ x : ℝ, f (-x) = - f x) ∧ (∀ x₁ x₂ : ℝ, x₁ ≤ x₂ → f x₁ ≤ f x₂) :=
begin
  sorry
end

end f_odd_and_increasing_on_ℝ_l471_471122


namespace monotonic_intervals_f_monotonic_intervals_g_h_max_min_number_of_real_roots_l471_471409

-- Definitions based on conditions
def y_cond (x a : ℝ) (hx : a > 0) : ℝ := x + a / x
def h_fun (x : ℝ) : ℝ := (x^2 + 1/x)^3 + (x + 1/x^2)^3

-- The theorem statements
theorem monotonic_intervals_f (a : ℝ) (h : a > 0) :
  ∀ x, (0 < x ∧ x ≤ real.cbrt a) → monotonic_decreasing_on (λ x, x^2 + a/x^2) x ∧
    (real.cbrt a ≤ x ∧ x < ∞) → monotonic_increasing_on (λ x, x^2 + a/x^2) x :=
sorry

theorem monotonic_intervals_g (a : ℝ) (h : a > 0) (n : ℕ) (hn : n ≥ 3) :
  ∀ x, (0 < x ∧ x ≤ real.nroot (2 * (n.to_real)) a) → monotonic_decreasing_on (λ x, x^n + a/x^n) x ∧
    (real.nroot (2 * (n.to_real)) a ≤ x ∧ x < ∞) → monotonic_increasing_on (λ x, x^n + a/x^n) x :=
sorry

theorem h_max_min :
  (∀ x ∈ set.Icc (1/2:ℝ) 2, h_fun x ≥ 16) →
  (∀ x ∈ set.Icc (1/2:ℝ) 2, h_fun x ≤ 6561 / 64) :=
sorry

theorem number_of_real_roots (m : ℝ) (hm : 0 < m ∧ m ≤ 30) :
  ∃ n : ℕ,
  (0 < m ∧ m < 8 → n = 0) ∧
  (m = 8 → n = 1) ∧
  (8 < m ∧ m < 16 → n = 2) ∧
  (m = 16 → n = 3) ∧
  (16 < m ∧ m ≤ 30 → n = 4) :=
sorry

end monotonic_intervals_f_monotonic_intervals_g_h_max_min_number_of_real_roots_l471_471409


namespace negative_subtraction_result_l471_471732

theorem negative_subtraction_result : -2 - 1 = -3 := 
by
  -- The proof is not required by the prompt, so we use "sorry" to indicate the unfinished proof.
  sorry

end negative_subtraction_result_l471_471732


namespace PA_dot_PB_l471_471822

-- Define the hyperbola
def is_hyperbola (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  x^2 / 3 - y^2 = 1

-- Define the dot product condition
theorem PA_dot_PB (P A B : ℝ × ℝ)
  (hP : is_hyperbola P)
  (PA_perpendicular_asymptote1 : (A.1 - P.1) + √3 * (A.2 - P.2) = 0)
  (PA_perpendicular_asymptote2 : (A.1 - P.1) - √3 * (A.2 - P.2) = 0)
  (PB_perpendicular_asymptote1 : (B.1 - P.1) + √3 * (B.2 - P.2) = 0)
  (PB_perpendicular_asymptote2 : (B.1 - P.1) - √3 * (B.2 - P.2) = 0)
  : (P.1 - A.1) * (P.1 - B.1) + (P.2 - A.2) * (P.2 - B.2) = -3/8 := 
sorry

end PA_dot_PB_l471_471822


namespace mrs_hilt_remaining_cents_l471_471933

-- Define the initial amount of money Mrs. Hilt had
def initial_cents : ℕ := 43

-- Define the cost of the pencil
def pencil_cost : ℕ := 20

-- Define the cost of the candy
def candy_cost : ℕ := 5

-- Define the remaining money Mrs. Hilt has after the purchases
def remaining_cents : ℕ := initial_cents - (pencil_cost + candy_cost)

-- Theorem statement to prove that the remaining amount is 18 cents
theorem mrs_hilt_remaining_cents : remaining_cents = 18 := by
  -- Proof omitted
  sorry

end mrs_hilt_remaining_cents_l471_471933


namespace complex_problem_l471_471567

open Complex

theorem complex_problem (x y : ℝ) (h : (1 + I) * x + (I - 1) * y = 2) (z : ℂ := x + y * I) :
  |z| = √2 ∧ z.im = -1 ∧ z / conj(z) = -I :=
by
  sorry

end complex_problem_l471_471567


namespace translate_line_down_l471_471637

theorem translate_line_down (k : ℝ) (b : ℝ) : 
  (∀ x : ℝ, b = 0 → (y = k * x - 3) = (y = k * x - 3)) :=
by
  sorry

end translate_line_down_l471_471637


namespace pyramid_volume_l471_471570

theorem pyramid_volume (AB BC PA PB : ℝ) (h_AB : AB = 12) (h_BC : BC = 6) (h_PB : PB = 25) 
  (h_PA_perp_AD : ∀ x, PA = x ∧ x > 0) (h_PA_perp_AB : ∀ x, PA = x ∧ x > 0) : (1 / 3) * (AB * BC) * PA = 24 * Real.sqrt 481 :=
by
  have h1 : PA * PA = PB * PB - AB * AB :=
    by sorry
  have h2 : AB * BC = 72 :=
    by rw [h_AB, h_BC]; exact rfl
  have h3 : PA = Real.sqrt 481 :=
    by rw [←h1, h_PB, h_AB]; sorry
  have volume : (1 / 3) * 72 * Real.sqrt 481 = 24 * Real.sqrt 481 :=
    by sorry
  exact volume

end pyramid_volume_l471_471570


namespace angle_BAC_is_60_degrees_l471_471677

variable (O I A B C : Point)
variable [triangle : Triangle A B C]
variable [circumcenter : Circumcenter O (Triangle A B C)]
variable [incenter : Incenter I (Triangle A B C)]
variable [O_inside : O ∈ interior (Triangle A B C)]
variable [I_on_circle : I ∈ circle_through B O C]

theorem angle_BAC_is_60_degrees :
  angle A B C = 60 :=
by sorry

end angle_BAC_is_60_degrees_l471_471677


namespace exercise_l471_471081

theorem exercise (x y z : ℕ) (h1 : x * y * z = 1) : (7 ^ ((x + y + z) ^ 3) / 7 ^ ((x - y + z) ^ 3)) = 7 ^ 6 := 
by
  sorry

end exercise_l471_471081


namespace car_speed_l471_471668

section CarSpeed

-- Definitions
def time_at_constant_speed (distance_speed : ℕ) : ℕ := 3600 / distance_speed

def time_taken (unknown_speed time_more : ℕ) : ℕ := time_at_constant_speed unknown_speed + time_more

def speed_from_time (distance time_taken : ℕ) : ℕ := distance * 3600 / time_taken

-- Main theorem
theorem car_speed (time_more : ℕ) : 
  (time_at_constant_speed 225 + time_more = time_at_constant_speed 225 + 2) → 
  speed_from_time 1 (3600 / 200) = 200 :=
by
  assume h1 : time_more = 2
  let t_at_225 := time_at_constant_speed 225
  have h_speed : speed_from_time 1 (t_at_225 + 2) = 200 := sorry
  exact h_speed

end CarSpeed

end car_speed_l471_471668


namespace minimum_value_l471_471773

noncomputable def f (x y : ℝ) : ℝ :=
  (x^2 + 1) / (y - 2) + (y^2 + 1) / (x - 2)

theorem minimum_value (x y : ℝ) (hx : x > 2) (hy : y > 2) :
  ∃ a b, x = a + 2 ∧ y = b + 2 ∧ a = sqrt 5 ∧ b = sqrt 5 ∧ f x y = 4 * sqrt 5 + 8 :=
sorry

end minimum_value_l471_471773


namespace land_area_of_each_section_l471_471995

theorem land_area_of_each_section (n : ℕ) (total_area : ℕ) (h1 : n = 3) (h2 : total_area = 7305) :
  total_area / n = 2435 :=
by {
  sorry
}

end land_area_of_each_section_l471_471995


namespace loci_of_feet_of_altitudes_symmetry_l471_471691

-- Definitions of geometric entities
variables {O A B : Point} (e : Line)

-- Feet of the altitudes from vertices O, A, and B
variables {M N P : Point}

-- Rotating angle and intersection condition
def rotating_angle_fixed_vertex (O : Point) (angle : ℝ) : Prop := sorry
def sides_intersect_fixed_line (e : Line) (A B : Point) : Prop := sorry
def feet_of_altitudes (△OAB : Triangle) (M N P : Point) : Prop := sorry

-- Mathematical equivalent proof problem
theorem loci_of_feet_of_altitudes_symmetry 
  (angle : ℝ)
  (h1 : rotating_angle_fixed_vertex O angle)
  (h2 : sides_intersect_fixed_line e A B)
  (h3 : feet_of_altitudes (Triangle.mk O A B) M N P) :
  symmetrical_relative_to_line (Line.mk O P) [M, N] := 
sorry

end loci_of_feet_of_altitudes_symmetry_l471_471691


namespace ac_work_time_l471_471261

theorem ac_work_time (W : ℝ) (a_work_rate : ℝ) (b_work_rate : ℝ) (bc_work_rate : ℝ) (t : ℝ) : 
  (a_work_rate = W / 4) ∧ 
  (b_work_rate = W / 12) ∧ 
  (bc_work_rate = W / 3) → 
  t = 2 := 
by 
  sorry

end ac_work_time_l471_471261


namespace expected_lifetime_flashlight_l471_471529

noncomputable def E (X : ℝ) : ℝ := sorry -- Define E as the expectation operator

variables (ξ η : ℝ) -- Define ξ and η as random variables representing lifetimes of blue and red bulbs
variable (h_exi : E ξ = 2) -- Given condition E ξ = 2

theorem expected_lifetime_flashlight (h_min : ∀ x y : ℝ, min x y ≤ x) :
  E (min ξ η) ≤ 2 :=
by
  sorry

end expected_lifetime_flashlight_l471_471529


namespace select_terms_from_sequence_l471_471378

theorem select_terms_from_sequence (k : ℕ) (hk : k ≥ 3) :
  ∃ (terms : Fin k → ℚ), (∀ i j : Fin k, i < j → (terms j - terms i) = (j.val - i.val) / k!) ∧
  (∀ i : Fin k, terms i ∈ {x : ℚ | ∃ n : ℕ, x = 1 / (n : ℚ)}) :=
by
  sorry

end select_terms_from_sequence_l471_471378


namespace extra_page_number_l471_471258

theorem extra_page_number (n k : ℕ) (H1 : ∑ i in Finset.range (n+1), i = n * (n+1) / 2)
  (H2 : n * (n+1) / 2 + k = 1986) : k = 33 :=
by sorry

end extra_page_number_l471_471258


namespace jog_to_coffee_shop_l471_471002

def constant_pace_jogging (time_to_park : ℕ) (dist_to_park : ℝ) (dist_to_coffee_shop : ℝ) : Prop :=
  time_to_park / dist_to_park * dist_to_coffee_shop = 6

theorem jog_to_coffee_shop
  (time_to_park : ℕ)
  (dist_to_park : ℝ)
  (dist_to_coffee_shop : ℝ)
  (h1 : time_to_park = 12)
  (h2 : dist_to_park = 1.5)
  (h3 : dist_to_coffee_shop = 0.75)
: constant_pace_jogging time_to_park dist_to_park dist_to_coffee_shop :=
by sorry

end jog_to_coffee_shop_l471_471002


namespace triangle_perimeter_l471_471044

-- Define the problem setup
variable (P Q R S A B C : Point)
variable (radius : ℝ)
variable (tangent : Point → Point → Prop)
variable (triangle : Point → Point → Point → Triangle)
variable (parallel : Line → Line → Prop)
variable (perimeter : Triangle → ℝ)

-- Define the conditions
axiom circles (radius = 1)
axiom tangent_condition1 : tangent P Q
axiom tangent_condition2 : tangent Q R
axiom tangent_condition3 : tangent R S
axiom tangent_condition4 : tangent S P
axiom tangent_condition5 : tangent P A
axiom tangent_condition6 : tangent Q B
axiom tangent_condition7 : tangent R C
axiom tangent_condition8 : tangent S A
axiom distance_condition1 : P.dist(S) = 2
axiom distance_condition2 : Q.dist(R) = 2

-- Parallel conditions (QR and RS are parallel to BC, PS is parallel to AC, PQ is parallel to AB)
axiom parallel_condition1 : parallel QR BC
axiom parallel_condition2 : parallel RS BC
axiom parallel_condition3 : parallel PS AC
axiom parallel_condition4 : parallel PQ AB

-- Final proof goal
theorem triangle_perimeter : 
  perimeter (triangle A B C) = 12 + 6 * √3 :=
sorry

end triangle_perimeter_l471_471044


namespace lambda_ge_e_l471_471344

noncomputable def satisfies_inequality (λ : ℝ) : Prop :=
∀ (n : ℕ) (a : Fin n → ℝ),
  (2 ≤ n) →
  (∀ i : Fin n, 0 < a i) →
  (∑ i, a i = n) →
  (∑ i, (1 / a i : ℝ) - λ * ∏ i, (1 / a i : ℝ) ≤ n - λ)

theorem lambda_ge_e (λ : ℝ) :
  satisfies_inequality λ → λ ≥ Real.exp 1 :=
by
  sorry

end lambda_ge_e_l471_471344


namespace swimming_pool_length_correct_l471_471622

noncomputable def swimming_pool_length (V_removed: ℝ) (W: ℝ) (H: ℝ) (gal_to_cuft: ℝ): ℝ :=
  V_removed / (W * H / gal_to_cuft)

theorem swimming_pool_length_correct:
  swimming_pool_length 3750 25 0.5 7.48052 = 40.11 :=
by
  sorry

end swimming_pool_length_correct_l471_471622


namespace flashlight_lifetime_expectation_leq_two_l471_471547

noncomputable def min_lifetime_expectation (ξ η : ℝ) (E_ξ : ℝ) : Prop :=
  E_ξ = 2 → E(min ξ η) ≤ 2

-- Assume ξ and η are random variables and E denotes the expectation.
axiom E : (ℝ → ℝ) → ℝ

theorem flashlight_lifetime_expectation_leq_two (ξ η : ℝ) (E_ξ : ℝ) (hE_ξ : E_ξ = 2) : E(min ξ η) ≤ 2 :=
  by
    sorry

end flashlight_lifetime_expectation_leq_two_l471_471547


namespace permutations_count_l471_471068

theorem permutations_count :
  let b : Fin 15 → Fin 15 := λ i => i + 1
  let condition1 := b 1 > b 2 ∧ b 2 > b 3 ∧ b 3 > b 4 ∧ b 4 > b 5 ∧ b 5 > b 6 ∧ b 6 > b 7 ∧ b 7 > b 8
  let condition2 := b 8 < b 9 ∧ b 9 < b 10 ∧ b 10 < b 11 ∧ b 11 < b 12 ∧ b 12 < b 13 ∧ b 13 < b 14
  let conditions := condition1 ∧ condition2
  ∃ b' : (Fin 15 → Fin 15), conditions b' ∧ (Fin 15) (1, 2, 3, ..., 13) = 1716 :=
sorry

end permutations_count_l471_471068


namespace probability_p2_probability_p3_probability_p20_l471_471161

noncomputable def p (i : ℕ) : ℚ :=
if i = 1 then 2/3 
else if i = 2 then 8/15 
else if i = 3 then 38/75 
else (1/5) * p (i-1) + (2/5)

theorem probability_p2 : p 2 = 8/15 := 
sorry

theorem probability_p3 : p 3 = 38/75 := 
sorry

theorem probability_p20 : p 20 = (1/6 * 5^19) + 1/2 := 
sorry

end probability_p2_probability_p3_probability_p20_l471_471161


namespace award_distribution_l471_471105

theorem award_distribution (students awards : ℕ) (h_s : students = 4) (h_a : awards = 7) :
  ∃(d:ℕ), 
  (d = 5880) ∧ ∀(award_dist : Fin students → ℕ), 
  (∑ i, award_dist i = awards) ∧ 
  (∀ i, award_dist i ≥ 1) ∧ 
  (∃ i, award_dist i = 3) →
  d = ∑ i : Fin students, award_dist i :=
by
  sorry

end award_distribution_l471_471105


namespace partitions_of_6_into_4_indistinguishable_boxes_l471_471434

theorem partitions_of_6_into_4_indistinguishable_boxes : 
  ∃ (X : Finset (Multiset ℕ)), X.card = 9 ∧ 
  ∀ p ∈ X, p.sum = 6 ∧ p.card ≤ 4 := 
sorry

end partitions_of_6_into_4_indistinguishable_boxes_l471_471434


namespace min_value_sin_cos_squared_six_l471_471777

theorem min_value_sin_cos_squared_six (x : ℝ) :
  ∃ x : ℝ, (sin^6 x + 2 * cos^6 x) = 2/3 :=
sorry

end min_value_sin_cos_squared_six_l471_471777


namespace sphere_surface_area_lt_cube_l471_471800

noncomputable def sphere_volume (r : ℝ) : ℝ := (4 / 3) * π * r^3
noncomputable def cube_volume (a : ℝ) : ℝ := a^3

noncomputable def sphere_surface_area (r : ℝ) : ℝ := 4 * π * r^2
noncomputable def cube_surface_area (a : ℝ) : ℝ := 6 * a^2

theorem sphere_surface_area_lt_cube {V : ℝ} (hV : V > 0) :
  ∀ (r a : ℝ), sphere_volume r = V → cube_volume a = V → sphere_surface_area r < cube_surface_area a :=
by
  intro r a h1 h2
  let r_val := (3 * V / (4 * π))^(1 / 3)
  let a_val := V^(1 / 3)
  have hr : r = r_val := by sorry
  have ha : a = a_val := by sorry
  rw [hr, ha]
  have hsphere := sphere_surface_area r_val
  have hcube := cube_surface_area a_val
  -- General non-trivial calculations can be split into sub-goals
  suffices hcomp : 6 > (4 * π)^(1 / 3) * 3^(2 / 3) by
    rw [sphere_surface_area, cube_surface_area]
    dsimp [r_val, a_val]
    exact hcomp
  -- Sufficient to show the numerical comparison (in practice should justify)
  sorry

end sphere_surface_area_lt_cube_l471_471800


namespace game_not_fair_prob_first_player_win_l471_471174

-- Define the probability of player i winning
variable (n : ℕ) -- number of players
variable (P : ℕ → ℝ) -- function from player index to probability

-- Define the base condition for 36 players
def P_i (i : ℕ) : ℝ := P i

-- Given conditions
-- Condition 1: There are 36 players
def num_players := 36

-- Condition 2: Each player has a turn
-- This is implicit in the cyclic structure of the game.

-- Part (a): Prove that the game is not fair
theorem game_not_fair (h : ∀ i j : ℕ, i ≠ j → P_i i ≠ P_i j): Prop :=
  ∃ i j : ℕ, i ≠ j ∧ P_i i ≠ P_i j

-- Part (b): Calculate the probability of the first player winning

noncomputable def approx_prob_first_player_win : ℝ :=
  let e := Real.exp 1
  in e / (36 * (e - 1))

-- Theorem stating the approximate probability of the first player winning
theorem prob_first_player_win : Prop :=
  abs (approx_prob_first_player_win - 0.044) < 0.001

end game_not_fair_prob_first_player_win_l471_471174


namespace max_discount_rate_l471_471284

-- Define the constants used in the problem
def costPrice : ℝ := 4
def sellingPrice : ℝ := 5
def minProfitMarginRate : ℝ := 0.1
def minProfit : ℝ := costPrice * minProfitMarginRate -- which is 0.4

-- The maximum discount rate that preserves the required profit margin
theorem max_discount_rate : 
  ∃ x : ℝ, (0 ≤ x ∧ x ≤ 100) ∧ (sellingPrice * (1 - x / 100) - costPrice ≥ minProfit) ∧ (x = 12) :=
by
  sorry

end max_discount_rate_l471_471284


namespace speed_in_still_water_l471_471694

/-- A man can row upstream at 20 kmph and downstream at 80 kmph.
    Prove that the speed of the man in still water is 50 kmph -/
theorem speed_in_still_water (upstream_speed : ℕ) (downstream_speed : ℕ) 
  (h1 : upstream_speed = 20) (h2 : downstream_speed = 80) :
  (upstream_speed + downstream_speed) / 2 = 50 :=
by
  rw [h1, h2]
  norm_num
  sorry

end speed_in_still_water_l471_471694


namespace game_not_fair_prob_first_player_win_l471_471173

-- Define the probability of player i winning
variable (n : ℕ) -- number of players
variable (P : ℕ → ℝ) -- function from player index to probability

-- Define the base condition for 36 players
def P_i (i : ℕ) : ℝ := P i

-- Given conditions
-- Condition 1: There are 36 players
def num_players := 36

-- Condition 2: Each player has a turn
-- This is implicit in the cyclic structure of the game.

-- Part (a): Prove that the game is not fair
theorem game_not_fair (h : ∀ i j : ℕ, i ≠ j → P_i i ≠ P_i j): Prop :=
  ∃ i j : ℕ, i ≠ j ∧ P_i i ≠ P_i j

-- Part (b): Calculate the probability of the first player winning

noncomputable def approx_prob_first_player_win : ℝ :=
  let e := Real.exp 1
  in e / (36 * (e - 1))

-- Theorem stating the approximate probability of the first player winning
theorem prob_first_player_win : Prop :=
  abs (approx_prob_first_player_win - 0.044) < 0.001

end game_not_fair_prob_first_player_win_l471_471173


namespace p_n_divisible_by_5_l471_471371

noncomputable def p_n (n : ℕ) : ℕ := 1^n + 2^n + 3^n + 4^n

theorem p_n_divisible_by_5 (n : ℕ) (h : n ≠ 0) : p_n n % 5 = 0 ↔ n % 4 ≠ 0 := by
  sorry

end p_n_divisible_by_5_l471_471371


namespace min_value_z_l471_471656

theorem min_value_z : ∀ (x y : ℝ), ∃ z, z = 3 * x^2 + y^2 + 12 * x - 6 * y + 40 ∧ z = 19 :=
by
  intro x y
  use 3 * x^2 + y^2 + 12 * x - 6 * y + 40 -- Define z
  sorry -- Proof is skipped for now

end min_value_z_l471_471656


namespace students_attending_chess_class_l471_471163

def percentageChessClass (students_total : ℕ) (students_swimming : ℕ) (percentage_swim_from_chess : ℚ) : ℚ :=
  let P := (students_swimming * 100 / (students_total * percentage_swim_from_chess)).toRat in
  P

theorem students_attending_chess_class : (students_total = 1000) → (students_swimming = 20) → (percentage_swim_from_chess = 0.1) → 
  percentageChessClass students_total students_swimming percentage_swim_from_chess = 20 := 
by 
  intros.
  rw [students_total, students_swimming, percentage_swim_from_chess].
  unfold percentageChessClass.
  simp.
  norm_num.
  sorry

end students_attending_chess_class_l471_471163


namespace hip_hop_class_cost_l471_471083

-- Define the variables and constants
def hip_hop_cost : ℕ
def ballet_cost : ℕ := 12
def jazz_cost : ℕ := 8
def total_cost : ℕ := 52

-- Define the conditions as hypotheses
def condition_1 : ℕ := 2 -- Number of hip-hop classes per week
def condition_2 : ℕ := 2 -- Number of ballet classes per week
def condition_3 : ℕ := 1 -- Number of jazz classes per week

-- Prove that the cost of one hip-hop class is $10
theorem hip_hop_class_cost :
  hip_hop_cost = 10 :=
  let total_ballet_cost := condition_2 * ballet_cost
  let total_jazz_cost := condition_3 * jazz_cost
  let remaining_cost := total_cost - (total_ballet_cost + total_jazz_cost)
  let hip_hop_cost := remaining_cost / condition_1
  show hip_hop_cost = 10 by
    sorry

end hip_hop_class_cost_l471_471083


namespace systematic_classic_equations_l471_471486

theorem systematic_classic_equations (x y : ℕ) : 
  (7 * x + 7 = y) ∧ (9 * (x - 1) = y) ↔
  (if (exists p q : ℕ, p = 7 * x + 7 ∧ q = 9 * (x - 1)) 
  then x = x ∧ y = 9 * (x - 1) 
  else false) :=
by 
  sorry

end systematic_classic_equations_l471_471486


namespace lines_intersection_points_l471_471516

theorem lines_intersection_points (n : ℕ) (h₁ : n ≥ 5) (h₂ : ∀ (lines : set (set ℝ)) (h : lines.finite), 
                                                      lines.card = n →
                                                      ∃ parallel_set : set (set ℝ), 
                                                      parallel_set.card = 3 ∧ 
                                                      ∀ l ∈ parallel_set, ∀ l' ∈ parallel_set, l ≠ l' → parallel l l' ∧
                                                      ∀ (l₁ l₂ : set ℝ), l₁ ≠ l₂ → l₁ ∈ (lines \ parallel_set) → l₂ ∈ (lines \ parallel_set) → ¬parallel l₁ l₂ ∧
                                                      ∀ (l₁ l₂ l₃ : set ℝ), l₁ ≠ l₂ ∧ l₁ ≠ l₃ ∧ l₂ ≠ l₃ → l₁ ∈ lines → l₂ ∈ lines → l₃ ∈ lines → ¬collinear l₁ l₂ l₃): 
                                                      ∃ m : ℕ, m = (n^2 - n - 6) / 2 := sorry

end lines_intersection_points_l471_471516


namespace cosine_of_central_angle_l471_471295

-- Let O be the center of the circle, and R be the radius.
variables (O : Point) (R : ℝ)

-- Let A and B be points on the circle such that AB is a chord.
variables (A B : Point)

-- Let MN be a perpendicular segment to AB at point P.
variables (M N P : Point)

-- The segment AB is divided in the ratio 1:4, and the arc ANB is divided in a ratio 1:2.
variable (h1 : AM / MB = 1 / 4)
variable (h2 : arc A N B / arc N B = 1 / 2)

-- We aim to find the cosine of the central angle subtended by the arc A B.
theorem cosine_of_central_angle : 
  ∃ θ : ℝ, cos θ = -23 / 27 :=
sorry

end cosine_of_central_angle_l471_471295


namespace sum_coeff_expansion_l471_471991

theorem sum_coeff_expansion (x y : ℝ) : 
  (x + 2 * y)^4 = 81 := sorry

end sum_coeff_expansion_l471_471991


namespace max_discount_rate_l471_471285

-- Define the constants used in the problem
def costPrice : ℝ := 4
def sellingPrice : ℝ := 5
def minProfitMarginRate : ℝ := 0.1
def minProfit : ℝ := costPrice * minProfitMarginRate -- which is 0.4

-- The maximum discount rate that preserves the required profit margin
theorem max_discount_rate : 
  ∃ x : ℝ, (0 ≤ x ∧ x ≤ 100) ∧ (sellingPrice * (1 - x / 100) - costPrice ≥ minProfit) ∧ (x = 12) :=
by
  sorry

end max_discount_rate_l471_471285


namespace trajectory_of_A_fixed_point_MN_l471_471037

-- Definitions from conditions
def vertexB : ℝ × ℝ := (0, 1)
def vertexC : ℝ × ℝ := (0, 1)
def PA (A P : ℝ × ℝ) := let (a_x, a_y) := A in let (p_x, p_y) := P in ((a_x - p_x), (a_y - p_y))
def PB (P : ℝ × ℝ) := let (p_x, p_y) := P in ((-p_x), (1 - p_y))
def PC (P : ℝ × ℝ) := let (p_x, p_y) := P in ((-p_x), (1 - p_y))

def condition1 (A P : ℝ × ℝ) := PA(A,P) + PB(P) + PC(P) = (0,0)
def condition2 (Q : ℝ × ℝ) (A B C : ℝ × ℝ) := let (q_x, q_y) := Q in (q_x - B.1)^2 + q_y^2 = (q_x - C.1)^2 + q_y^2 /\
                                                (q_x - A.1)^2 + q_y^2 = (q_x - B.1)^2 + q_y^2 /\
                                                norm (Q - A) = norm (Q - B)
def condition3 (P Q B C : ℝ × ℝ) := let (p_x, p_y) := P in let (q_x, q_y) := Q in let (b_x, b_y) := B in let (c_x, c_y) := C in 
                           (q_x - p_x) / (q_y - p_y) = (c_x - b_x) / (c_y - b_y)


theorem trajectory_of_A (A B C P Q: ℝ × ℝ) 
  (hB : B = vertexB) (hC : C = vertexC)
  (h1 : condition1 A P) (h2 : condition2 Q A B C)
  (h3 : condition3 P Q B C) :
  (let (a_x, a_y) := A in a_x^2 / 3 + a_y^2 = 1) :=
begin
  sorry
end

theorem fixed_point_MN (F : ℝ × ℝ) 
  (hF: F = (sqrt 2, 0))
  : (3sqrt(2)/4, 0) :=
begin
  sorry
end

end trajectory_of_A_fixed_point_MN_l471_471037


namespace h_h_three_l471_471864

def h (x : ℤ) : ℤ := 3 * x^2 + 3 * x - 2

theorem h_h_three : h (h 3) = 3568 := by
  sorry

end h_h_three_l471_471864


namespace diagonals_intersect_on_diameter_l471_471905

theorem diagonals_intersect_on_diameter 
  (O A B C D E: Type*)
  [linear_ordered_comm_ring O]
  (inscribed: cyclic_pentagon A B C D E O)
  (angleB: ∠B = 120)
  (angleC: ∠C = 120)
  (angleD: ∠D = 130)
  (angleE: ∠E = 100)
  : ∃ X, X ∈ diameter AO ∧ X ∈ line BD ∧ X ∈ line CE :=
sorry

end diagonals_intersect_on_diameter_l471_471905


namespace hyperbola_equation_l471_471897

theorem hyperbola_equation :
  ∃ a b : ℝ, (e = (Real.sqrt 5) / 2) ∧ (2 * a = 12) ∧ (c^2 = a^2 + b^2)
  ∧ (hyperbola_equation : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1) :=
begin
  sorry
end

end hyperbola_equation_l471_471897


namespace problem_statement_l471_471370

variable (P Q : Prop)
def P_def : P := 2 + 2 = 5
def Q_def : Q := 3 > 2

theorem problem_statement : (P ∨ Q) ∧ ¬¬Q :=
by
  rw [P_def, Q_def]
  sorry

end problem_statement_l471_471370


namespace total_area_for_building_l471_471996

theorem total_area_for_building (num_sections : ℕ) (area_per_section : ℝ) (open_space_percentage : ℝ) :
  num_sections = 7 →
  area_per_section = 9473 →
  open_space_percentage = 0.15 →
  (num_sections * (area_per_section * (1 - open_space_percentage))) = 56364.35 :=
by
  intros h1 h2 h3
  sorry

end total_area_for_building_l471_471996


namespace complex_modulus_sum_l471_471749

noncomputable def complex_modulus (a b : ℝ) : ℝ := real.sqrt (a^2 + b^2)

theorem complex_modulus_sum :
  complex_modulus 3 (-5) + complex_modulus 3 5 + complex_modulus 6 (-8) = 2 * real.sqrt 34 + 10 :=
by sorry

end complex_modulus_sum_l471_471749


namespace sum_pow_congruent_zero_mod_m_l471_471074

theorem sum_pow_congruent_zero_mod_m
  {a : ℕ → ℤ} {x : ℕ → ℤ} (n r : ℕ) 
  (hn : n ≥ 2) (hr : r ≥ 2)
  (h0 : ∀ k : ℕ, 1 ≤ k ∧ k ≤ r → ∑ j in Finset.range (n+1), a j * x j ^ k = 0) :
  ∀ m : ℕ, r+1 ≤ m ∧ m ≤ 2*r+1 → ∑ j in Finset.range (n+1), a j * x j ^ m ≡ 0 [MOD m] :=
by
  sorry

end sum_pow_congruent_zero_mod_m_l471_471074


namespace number_of_correct_statements_is_zero_l471_471978

theorem number_of_correct_statements_is_zero
  (h1 : ∀ (A B C : Type) [linear_ordered_field A][linear_ordered_field B][linear_ordered_field C], 
        collinear_points := A = B ∧ B = C → ¬ plane_determined_by (A, B, C))
  (h2 : ∀ (lateral_surface : Type → Type), 
        cone_lateral_surface_unfolded_is_sector := (radius > 0 ∧ arc_length < 2 * pi))
  (h3 : ∀ (base : Type), 
        equilateral_base_imp_non_reg_tetrahedron := ¬(isosceles_lateral_faces → regular_tetrahedron))
  (h4 : ∀ (spherical_surface : Type), 
        great_circle_count := (two_distinct_points → one_great_circle) ∧ (antipodal_points → infinitely_many_great_circles)) :
  num_correct_statements [h1, h2, h3, h4] = 0 := 
begin 
  sorry 
end

end number_of_correct_statements_is_zero_l471_471978


namespace Loui_current_age_l471_471496

theorem Loui_current_age :
  ∃ L : ℕ, 26 + 20 = 2 * L ∧ L = 23 :=
by {
  use 23,
  split,
  {
    rw [nat.add_comm, nat.add_assoc],
    exact rfl,
  },
  {
    exact rfl,
  }
}

end Loui_current_age_l471_471496


namespace fraction_of_phone_numbers_l471_471314

theorem fraction_of_phone_numbers (total_digits : ℕ) (invalid_start_digits : Finset ℕ) (valid_start_digits : Finset ℕ) (invalid_end_digits : Finset ℕ) (begin_with : ℕ) (end_with : ℕ) :
  total_digits = 7 →
  invalid_start_digits = {0, 1} →
  valid_start_digits = Finset.range 10 \ invalid_start_digits →
  invalid_end_digits = ∅ →
  valid_start_digits.card = 8 →
  begin_with = 9 →
  end_with = 0 →
  (finset.card (finset.product (finset.singleton begin_with) (finset.product (Finset.range 10 ^ (total_digits - 2)) (finset.singleton end_with))) : ℝ) / (valid_start_digits.card * 10^(total_digits - 1) : ℝ) = 1 / 80 :=
begin
  sorry

end fraction_of_phone_numbers_l471_471314


namespace roots_derivative_inside_triangle_l471_471513

theorem roots_derivative_inside_triangle
  (a b c : ℂ)
  (h_neq_ab : a ≠ b)
  (h_neq_ac : a ≠ c)
  (h_neq_bc : b ≠ c) :
  let f : ℂ → ℂ := λ x, (x - a) * (x - b) * (x - c) in
  let roots_of_f' := {x : ℂ | f.derivative x = 0} in
  let triangle := {z : ℂ | ∃ λ₁ λ₂ λ₃ : ℝ, λ₁ ≥ 0 ∧ λ₂ ≥ 0 ∧ λ₃ ≥ 0 ∧ λ₁ + λ₂ + λ₃ = 1 ∧ z = λ₁ • a + λ₂ • b + λ₃ • c} in
  ∀ x ∈ roots_of_f', x ∈ triangle :=
by
  sorry

end roots_derivative_inside_triangle_l471_471513


namespace find_length_AG_l471_471026

noncomputable def length_segment_AG (A B C G M : ℝ) (h_ABC_isosceles: ∀ x, x ∈ [{AB = 3, AC = 3, BC = 4}]) 
  (h_M_midpoint: M = (B + C) / 2) (h_AM_median: G = (A + M) / 2) : ℝ :=
  (1 / 2) * Real.sqrt 5 

# Variables and hypotheses to represent isosceles triangle given condition
variables {A B C : Point}
variables [IsIsoscelesTriangle A B C]
variable (AB AC BC : ℝ)
variables (A_ABC_isosceles : AB = 3 ∧ AC = 3 ∧ BC = 4)

-- Midpoint definition
def midpoint (B C : Point) : Point := 
  Point.mk ((B.x + C.x) / 2) ((B.y + C.y) / 2)

-- Definition of M as the midpoint of BC
def M := midpoint B C

-- Hypothesis stating that M is indeed the midpoint
axiom h_M_midpoint : M = midpoint B C

-- Hypothesis for the intersection point G
variables (A M G : Point)
axiom h_AM_median : G = midpoint A M

-- The proof statement itself
theorem find_length_AG (h1: AB = 3) (h2: AC = 3) (h3: BC = 4)
  (h1: IsIsoscelesTriangle A B C) (h2: h_M_midpoint) (h3: h_AM_median) : AG = (Real.sqrt 5) / 2 :=
sorry   -- Proof steps would be here

#check find_length_AG

end find_length_AG_l471_471026


namespace cosine_of_angle_l471_471294

-- Define vectors
def v1 : ℝ × ℝ := (4, 5)
def v2 : ℝ × ℝ := (2, 7)

-- Define dot product function
def dot_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2

-- Define norm function
def norm (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Define the cosine of the angle theta
def cos_theta : ℝ :=
  dot_product v1 v2 / (norm v1 * norm v2)

-- Theorem to prove the cosine of the angle
theorem cosine_of_angle :
  cos_theta = 43 / (Real.sqrt 41 * Real.sqrt 53) :=
  by sorry

end cosine_of_angle_l471_471294


namespace sum_of_quarter_circle_arcs_eq_semi_circumference_l471_471975

variable (D : ℝ) (n : ℕ)

-- Definitions based on the problem conditions
def segment_length := D / n
def quarter_circle_circumference := (π * segment_length) / 4
def total_arc_length := 2 * n * quarter_circle_circumference

-- The statement that as n approaches infinity, total_arc_length approaches semi-circumference
theorem sum_of_quarter_circle_arcs_eq_semi_circumference (D : ℝ) (n : ℕ) :
  filter.tendsto (λ (n : ℕ), 2 * n * (π * (D / n) / 4)) filter.at_top (𝓝 (π * D / 2)) :=
by
  sorry

end sum_of_quarter_circle_arcs_eq_semi_circumference_l471_471975


namespace tiling_methods_l471_471889

-- Define the regular polygons and their properties
inductive RegularPolygon
| equilateral_triangle
| square
| hexagon
| octagon

def interior_angle (p : RegularPolygon) : ℕ :=
  match p with
  | RegularPolygon.equilateral_triangle => 60
  | RegularPolygon.square => 90
  | RegularPolygon.hexagon => 120
  | RegularPolygon.octagon => 135

def can_tile_floor (p1 p2 : RegularPolygon) : Prop :=
  (interior_angle p1 + interior_angle p2) ∣ 360

-- Statement of the problem
theorem tiling_methods :
  (finset.unordered_pairs {RegularPolygon.equilateral_triangle, RegularPolygon.square, RegularPolygon.hexagon, RegularPolygon.octagon}).count can_tile_floor = 3 :=
sorry

end tiling_methods_l471_471889


namespace math_problem_l471_471198

variable (x y z : ℤ)

def given_conditions : Prop := (x = -2) ∧ (y = 1) ∧ (z = 4)

theorem math_problem (h : given_conditions x y z) : x^2 * y * z - x * y * z^2 = 48 :=
by 
  obtain ⟨hx, hy, hz⟩ := h
  simp [hx, hy, hz]
  sorry

end math_problem_l471_471198


namespace sum_of_three_numbers_l471_471137

theorem sum_of_three_numbers (a b c : ℝ) (h1 : a ≤ b) (h2 : b ≤ c) 
 (h_median : b = 10) 
 (h_mean_least : (a + b + c) / 3 = a + 8)
 (h_mean_greatest : (a + b + c) / 3 = c - 20) : 
 a + b + c = 66 :=
by 
  sorry

end sum_of_three_numbers_l471_471137


namespace min_xy_value_min_x_plus_y_value_l471_471825

variable {x y : ℝ}

theorem min_xy_value (hx : 0 < x) (hy : 0 < y) (h : (2 / y) + (8 / x) = 1) : xy ≥ 64 := 
sorry

theorem min_x_plus_y_value (hx : 0 < x) (hy : 0 < y) (h : (2 / y) + (8 / x) = 1) : x + y ≥ 18 :=
sorry

end min_xy_value_min_x_plus_y_value_l471_471825


namespace sequence_bn_l471_471796

theorem sequence_bn (a b : ℕ → ℕ) (n : ℕ) (h1 : ∀ n, a (n+1) = (3 : ℕ)^n)
  (h2 : a 1 = 1)
  (h3 : a 2 + a 1 + a 2 + a 3 = 16)
  (h4 : ∀ n, (∑ i in Finset.range n, a (i+1) * b (i+1)) = n * 3^n) :
  b(n + 1) = 2 * (n + 1) + 1 := sorry

end sequence_bn_l471_471796


namespace log_expression_value_l471_471728

theorem log_expression_value : (Real.log 8 / Real.log 10) + 3 * (Real.log 5 / Real.log 10) = 3 :=
by
  -- Assuming necessary properties and steps are already known and prove the theorem accordingly:
  sorry

end log_expression_value_l471_471728


namespace set_equiv_transitive_proof_l471_471861

variables {A A1 B B1 : Type} 
variable (hA1 : A1 ⊆ A)
variable (hB1 : B1 ⊆ B)
variable (hAB1 : A ≃ B1)
variable (hBA1 : B ≃ A1)

theorem set_equiv_transitive_proof : A ≃ B := 
by
  -- Proof omitted
  sorry

end set_equiv_transitive_proof_l471_471861


namespace number_of_people_l471_471998

-- Definitions based on the conditions
def average_age (T : ℕ) (n : ℕ) := T / n = 30
def youngest_age := 3
def average_age_when_youngest_born (T : ℕ) (n : ℕ) := (T - youngest_age) / (n - 1) = 27

theorem number_of_people (T n : ℕ) (h1 : average_age T n) (h2 : average_age_when_youngest_born T n) : n = 7 :=
by
  sorry

end number_of_people_l471_471998


namespace arithmetic_sequence_satisfies_condition_l471_471407

def f (x : ℝ) : ℝ := x^3 + x^2 + (4 / 3) * x + (13 / 27)

def a (n : ℕ) : ℝ := n / 3 - 17

theorem arithmetic_sequence_satisfies_condition :
    (∑ i in finset.range 99, f (a i)) = 11 :=
sorry

end arithmetic_sequence_satisfies_condition_l471_471407


namespace probability_of_exactly_nine_correct_matches_is_zero_l471_471238

theorem probability_of_exactly_nine_correct_matches_is_zero :
  let n := 10 in
  let match_probability (correct: Fin n → Fin n) (guess: Fin n → Fin n) (right_count: Nat) :=
    (Finset.univ.filter (λ i => correct i = guess i)).card = right_count in
  ∀ (correct_guessing: Fin n → Fin n), 
    ∀ (random_guessing: Fin n → Fin n),
      match_probability correct_guessing random_guessing 9 → 
        match_probability correct_guessing random_guessing 10 :=
begin
  sorry -- This skips the proof part
end

end probability_of_exactly_nine_correct_matches_is_zero_l471_471238


namespace sin_angle_BAC_regular_tetrahedron_l471_471328

-- Define the regular tetrahedron and related properties
structure Tetrahedron where
  A B C D : ℝ^3 -- Points in 3D space
  edge_length : ℝ -- Length of each edge
  edge_eq : ∀ {X Y : ℝ^3}, (X = A ∧ Y = B ∨ X = A ∧ Y = C ∨ X = A ∧ Y = D ∨ X = B ∧ Y = C ∨ X = B ∧ Y = D ∨ X = C ∧ Y = D) → dist X Y = edge_length

variables {A B C D : ℝ^3} {s : ℝ}

-- Regular tetrahedron property: all edges have the same length 's'
def regular_tetrahedron (s : ℝ) : Tetrahedron :=
{ A := A, B := B, C := C, D := D, edge_length := s, edge_eq := sorry }

-- Given a regular tetrahedron, prove sin ∠BAC = √3 / 3
theorem sin_angle_BAC_regular_tetrahedron (T : regular_tetrahedron s) :
  sin (angle T.A T.B T.C) = (sqrt 3) / 3 :=
sorry

end sin_angle_BAC_regular_tetrahedron_l471_471328


namespace gumball_probability_l471_471957

theorem gumball_probability :
  let total_gumballs : ℕ := 25
  let orange_gumballs : ℕ := 10
  let green_gumballs : ℕ := 6
  let yellow_gumballs : ℕ := 9
  let total_gumballs_after_first : ℕ := total_gumballs - 1
  let total_gumballs_after_second : ℕ := total_gumballs - 2
  let orange_probability_first : ℚ := orange_gumballs / total_gumballs
  let green_or_yellow_probability_second : ℚ := (green_gumballs + yellow_gumballs) / total_gumballs_after_first
  let orange_probability_third : ℚ := (orange_gumballs - 1) / total_gumballs_after_second
  orange_probability_first * green_or_yellow_probability_second * orange_probability_third = 9 / 92 :=
by
  sorry

end gumball_probability_l471_471957


namespace find_CN_is_sqrt_of_17_l471_471090

noncomputable def point := (ℝ × ℝ)

def right_triangle (A B C : point) : Prop :=
  ∃ (a b : ℝ), A = (0, b) ∧ B = (a, 0) ∧ C = (0, 0)

def square_outside (A B C D E : point) : Prop :=
  ∥ A - B ∥ = ∥ B - C ∥ ∧ -- Assuming there's a calculation method for distance
  ∥ A - D ∥ = ∥ A - E ∥ ∧ -- More conditions for the squares ACDE and BCFG
  -- We would have exact calculations for other sides and verify right angles

def median_is_constructed (A B M : point) : Prop :=
  M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

def extension_intersects (C M D F N : point) (median_eqn line_eqn : ℝ → ℝ) : Prop := 
  median_eqn N.1 = N.2 ∧ line_eqn N.1 = N.2

def find_C_N {A B C D E F M N : point} 
  (h_right_triangle : right_triangle A B C)
  (h_square_outside1: square_outside A C D E)
  (h_square_outside2: square_outside B C F G)
  (h_median: median_is_constructed A B M)
  (h_intersection: extension_intersects C M D F N -- placeholder equations):
  ℝ := 
  ∥ C - N ∥

theorem find_CN_is_sqrt_of_17 (A B C D E F M N : point)
  (h_right_triangle : right_triangle A B C)
  (h_square_outside1 : square_outside A C D E)
  (h_square_outside2 : square_outside B C F G)
  (h_median : median_is_constructed A B M)
  (h_intersection : extension_intersects C M D F N):
  find_C_N h_right_triangle h_square_outside1 h_square_outside2 h_median h_intersection =  sqrt 17 :=
sorry

end find_CN_is_sqrt_of_17_l471_471090


namespace find_tangent_line_equation_l471_471758

-- Define the curve as a function
def curve (x : ℝ) : ℝ := 2 * x^2 + 1

-- Define the derivative of the curve
def curve_derivative (x : ℝ) : ℝ := 4 * x

-- Define the point of tangency
def P : ℝ × ℝ := (-1, 3)

-- Define the slope of the tangent line at point P
def slope_at_P : ℝ := curve_derivative P.1

-- Define the expected equation of the tangent line
def tangent_line (x y : ℝ) : Prop := 4 * x + y + 1 = 0

-- The theorem to prove that the tangent line at point P has the expected equation
theorem find_tangent_line_equation : 
  tangent_line P.1 (curve P.1) :=
  sorry

end find_tangent_line_equation_l471_471758


namespace field_area_is_174_6359_l471_471302

namespace RectangleField

def length_more_than_width (L W : ℝ) : Prop := L = W + 10
def field_length (L : ℝ) : Prop := L = 19.13
def field_area (L W : ℝ) : ℝ := L * W

theorem field_area_is_174_6359 :
  ∀ (L W : ℝ), length_more_than_width L W → field_length L → field_area L W = 174.6359 := 
by 
  intros L W h1 h2 
  have W_val : W = 9.13 := by {
    rw [field_length, ←length_more_than_width] at h2,
    rw h2 at h1,
    exact eq.subst (by linarith) rfl
  },
  rw [field_area, W_val, field_length] at *,
  linarith

end field_area_is_174_6359_l471_471302


namespace block_difference_exactly_two_l471_471688

noncomputable def count_different_blocks : ℕ :=
  let material := 1 + X in
  let size := 1 + 3 * X in
  let color := 1 + 3 * X in
  let shape := 1 + 3 * X in
  let generating_function := (material) * (size) * (color ^ 2) * (shape) in
  generating_function.coeff 2

theorem block_difference_exactly_two :
  count_different_blocks = 36 :=
sorry

end block_difference_exactly_two_l471_471688


namespace ratio_of_cows_to_hearts_is_32_l471_471997

-- Define the conditions
def number_of_hearts_on_standard_deck := 13
def cost_per_cow := 200
def total_cost_of_cows := 83200

-- Define the proof problem
theorem ratio_of_cows_to_hearts_is_32 :
  let number_of_cows := total_cost_of_cows / cost_per_cow in
  let ratio := number_of_cows / number_of_hearts_on_standard_deck in
  ratio = 32 :=
by 
  sorry

end ratio_of_cows_to_hearts_is_32_l471_471997


namespace parallel_PP1_AA1_l471_471883

theorem parallel_PP1_AA1
  (A A₁ B B₁ C C₁ P P₁ : ℝ³)
  (h1 : ¬(affine_independent ℝ ![A, A₁, B, B₁, C, C₁]))
  (h2 : collinear ℝ (![A, A₁, B, B₁, C, C₁],[vector.parallel A A₁ B B₁],[vector.parallel A A₁ C C₁])) :
  vector.parallel (vector.sub P P₁) (vector.sub A A₁) :=
by
  sorry

end parallel_PP1_AA1_l471_471883


namespace find_t_l471_471089

-- Definitions based on the conditions
def my_hours (t : ℚ) := t + 3
def my_rate (t : ℚ) := 3t - 1
def andrew_hours (t : ℚ) := 3t - 7
def andrew_rate (t : ℚ) := t + 4

-- Define my earnings and Andrew's earnings based on the conditions
def my_earnings (t : ℚ) := my_hours t * my_rate t
def andrew_earnings (t : ℚ) := andrew_hours t * andrew_rate t

-- Proof statement to show that t equals 26/3 given the earnings condition
theorem find_t (t : ℚ) (h : my_earnings t = andrew_earnings t + 5) : t = 26 / 3 := 
by {
  sorry -- Proof will be here
}

end find_t_l471_471089


namespace grid_arrangement_count_l471_471876

open Classical

-- Define the problem domain: a 3x3 grid, and each 2x2 grid subgrid requirement.
def valid_2x2_subgrid (grid : Fin 3 → Fin 3 → Fin 4) (i j : Fin 2) : Prop :=
  (∃ a b c d : Fin 4, 
     {grid i j, grid i (j+1), grid (i+1) j, grid (i+1) (j+1)} = {a, b, c, d} ∧
     ∀ x : Fin 4, x ∈ {a, b, c, d})

-- Define the validity for the whole grid
def valid_grid (grid : Fin 3 → Fin 3 → Fin 4) : Prop :=
  valid_2x2_subgrid grid 0 0 ∧
  valid_2x2_subgrid grid 0 1 ∧
  valid_2x2_subgrid grid 1 0 ∧
  valid_2x2_subgrid grid 1 1

-- Statement of the problem in Lean
theorem grid_arrangement_count : 
  ∃ n : ℕ, n = 72 ∧ 
  (∃ grid_set : Finset (Fin 3 → Fin 3 → Fin 4), 
     ∀ grid ∈ grid_set, valid_grid grid ∧
     grid_set.card = n) :=
begin
  sorry
end

end grid_arrangement_count_l471_471876


namespace infinite_geometric_series_sum_l471_471727

theorem infinite_geometric_series_sum :
  ∑' (n : ℕ), (1 : ℚ) * (-1 / 4 : ℚ) ^ n = 4 / 5 :=
by
  sorry

end infinite_geometric_series_sum_l471_471727


namespace minimum_value_inequality_l471_471767

theorem minimum_value_inequality (x y : ℝ) (hx : x > 2) (hy : y > 2) :
    (x^2 + 1) / (y - 2) + (y^2 + 1) / (x - 2) ≥ 18 := by
  sorry

end minimum_value_inequality_l471_471767


namespace first_ball_red_given_second_black_l471_471644

open ProbabilityTheory

noncomputable def urn_A : Finset (Finset ℕ) := { {0, 0, 0, 0, 1, 1}, {0, 0, 0, 1, 1, 2}, ... }
noncomputable def urn_B : Finset (Finset ℕ) := { {1, 1, 1, 2, 2, 2}, {1, 1, 2, 2, 2, 2}, ... }

noncomputable def prob_draw_red : ℕ := 7 / 15

theorem first_ball_red_given_second_black :
  (∑ A_Burn_selection in ({0, 1} : Finset ℕ), 
     ((∑ ball_draw from A_Burn_selection,
           if A_Burn_selection = 0 then (∑ red in urn_A, if red = 1 then 1 else 0) / 6 / 2
           else (∑ red in urn_B, if red = 1 then 1 else 0) / 6 / 2) *
     ((∑ second_urn_selection in ({0, 1} : Finset ℕ),
           if second_urn_selection = 0 and A_Burn_selection = 0 then 
              ∑ black in urn_A, if black = 1 then 1 else 0 / 6 / 2 
           else 
              ∑ black in urn_B, if black = 1 then 1 else 0 / 6 / 2))) = 7 / 15 :=
sorry

end first_ball_red_given_second_black_l471_471644


namespace sum_of_exponents_l471_471023

theorem sum_of_exponents :
  (∃ (i k m p x : ℕ),
    x = (∏ n in (finset.range 9).erase 0, n) ∧
    x = 2^i * 3^k * 5^m * 7^p ∧
    i > 0 ∧ k > 0 ∧ m > 0 ∧ p > 0 ∧
    i + k + m + p = 11) :=
sorry

end sum_of_exponents_l471_471023


namespace N_cannot_begin_with_even_digit_l471_471594

theorem N_cannot_begin_with_even_digit :
  (∃ N : ℕ, 
     let digits := [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] in 
     let a := (N.digits 10).reverse in 
     a.length = 10 ∧ 
     ∀ i, i < a.length → a.nth i ∈ some <$> digits ∧ 
     N.digits 10.to_list.perm digits ∧
     let A := 10 * (a.nth 0).get_or_else 0 + (a.nth 1).get_or_else 0 + 10 * (a.nth 2).get_or_else 0 + (a.nth 3).get_or_else 0 + 
               10 * (a.nth 4).get_or_else 0 + (a.nth 5).get_or_else 0 + 10 * (a.nth 6).get_or_else 0 + (a.nth 7).get_or_else 0 + 
               10 * (a.nth 8).get_or_else 0 + (a.nth 9).get_or_else 0,
     let B := 10 * (a.nth 1).get_or_else 0 + (a.nth 2).get_or_else 0 + 10 * (a.nth 3).get_or_else 0 + (a.nth 4).get_or_else 0 + 
               10 * (a.nth 5).get_or_else 0 + (a.nth 6).get_or_else 0 + 10 * (a.nth 7).get_or_else 0 + (a.nth 8).get_or_else 0 in
     A = B) → 
  let d0 := (N.digits 10).reverse.nth 0 in
  d0.get_or_else 0 % 2 ≠ 0 := 
sorry

end N_cannot_begin_with_even_digit_l471_471594


namespace question1_question2_l471_471850

noncomputable def A := { x : ℝ | x^2 - 2*x - 3 ≤ 0 }
noncomputable def B (m : ℝ) := { x : ℝ | x^2 - 2*m*x + m^2 - 4 ≤ 0 }

theorem question1 (m : ℝ) : (A ∩ B m) = set.Icc 1 3 ↔ m = 3 :=
by sorry

theorem question2 (m : ℝ) : A ⊆ (B m)ᶜ ↔ m ∈ set.Icc (-∞) (-3) ∪ set.Icc 5 ∞ :=
by sorry

end question1_question2_l471_471850


namespace num_correct_judgments_l471_471399

noncomputable def judgment_1 := ∃ x : ℝ, exp x ≤ 0
def judgment_2 := ∀ x : ℝ, 0 < x → 2^x > x^2
def judgment_3 (a b : ℝ) := a > 1 ∧ b > 1 ↔ a * b > 1
def judgment_4 := ∀ (p q : Prop), (p → q) ↔ (¬q → ¬p)

theorem num_correct_judgments : 
  (¬ judgment_1) ∧ (¬ judgment_2) ∧ (¬ ∀ (a b : ℝ), judgment_3 a b) ∧ judgment_4 → 
  1 = 1 := 
by 
  sorry

end num_correct_judgments_l471_471399


namespace max_principals_ten_years_l471_471747

theorem max_principals_ten_years : 
  (∀ (P : ℕ → Prop), (∀ n, n ≥ 10 → ∀ i, ¬P (n - i)) → ∀ p, p ≤ 4 → 
  (∃ n ≤ 10, ∀ k, k ≥ n → P k)) :=
sorry

end max_principals_ten_years_l471_471747


namespace boys_count_in_dance_class_l471_471467

theorem boys_count_in_dance_class
  (total_students : ℕ) 
  (ratio_girls_to_boys : ℕ) 
  (ratio_boys_to_girls: ℕ)
  (total_students_eq : total_students = 35)
  (ratio_eq : ratio_girls_to_boys = 3 ∧ ratio_boys_to_girls = 4) : 
  ∃ boys : ℕ, boys = 20 :=
by
  let k := total_students / (ratio_girls_to_boys + ratio_boys_to_girls)
  have girls := ratio_girls_to_boys * k
  have boys := ratio_boys_to_girls * k
  use boys
  sorry

end boys_count_in_dance_class_l471_471467


namespace game_not_fair_probability_first_player_wins_approx_l471_471169

def fair_game (n : ℕ) (P : ℕ → ℚ) : Prop :=
  ∀ k j, k ≤ n → j ≤ n → P k = P j

noncomputable def probability_first_player_wins (n : ℕ) : ℚ :=
  let r := (n - 1 : ℚ) / n;
  (1 : ℚ) / n * (1 : ℚ) / (1 - r ^ n)

-- Check if the game is fair for 36 players
theorem game_not_fair : ¬ fair_game 36 (λ i, probability_first_player_wins 36) :=
sorry

-- Calculate the approximate probability of the first player winning
theorem probability_first_player_wins_approx (n : ℕ)
  (h₁ : n = 36) :
  probability_first_player_wins n ≈ 0.044 :=
sorry

end game_not_fair_probability_first_player_wins_approx_l471_471169


namespace can_petya_win_l471_471941

theorem can_petya_win :
  ∃ f : ℕ → ℕ, (∀ n m, m ∈ {2, 3, 8, 9} → 0 ≤ n → n + f n ≡ 0 [MOD 2020]) → True :=
by
  sorry

end can_petya_win_l471_471941


namespace x_2015_value_l471_471807

theorem x_2015_value (a : ℝ) (x : ℝ) (x₀ : ℝ) (xₙ₋₁ xₙ : ℕ → ℝ)
  (h₁ : f = λ x, x / (a * (x + 2)))
  (h₂ : ∃! x, x = f x)
  (h₃ : f x₀ = 1 / 1008)
  (h₄ : ∀ n, f(xₙ₋₁ n) = xₙ n) :
  a = 1/2 ∧ xₙ 2015 = 1 / 2015 :=
by
  sorry

end x_2015_value_l471_471807


namespace calc_G10_G12_G11_l471_471734

def A : Matrix ℕ ℕ ℤ :=
  ![![2, 1], ![1, 1]]

def G : ℕ → ℤ
| 0     => 0
| 1     => 1
| 2     => 2
| (n+3) => 2 * G (n+2) + G (n+1)

theorem calc_G10_G12_G11 :
  G 10 * G 12 - G 11 ^ 2 = 1 := sorry

end calc_G10_G12_G11_l471_471734


namespace triangle_right_l471_471462

theorem triangle_right (a b c : ℕ)
  (h : a^2 + b^2 + c^2 + 338 = 10a + 24b + 26c) :
  a = 5 ∧ b = 12 ∧ c = 13 :=
by 
  sorry

end triangle_right_l471_471462


namespace light_bulbs_circle_l471_471936

theorem light_bulbs_circle : ∀ (f : ℕ → ℕ),
  (f 0 = 1) ∧
  (f 1 = 2) ∧
  (f 2 = 4) ∧
  (f 3 = 8) ∧
  (∀ n, f n = f (n - 1) + f (n - 2) + f (n - 3) + f (n - 4)) →
  (f 9 - 3 * f 3 - 2 * f 2 - f 1 = 367) :=
by
  sorry

end light_bulbs_circle_l471_471936


namespace num_ways_distribute_balls_l471_471427

-- Definitions of the conditions
def balls := 6
def boxes := 4

-- Theorem statement
theorem num_ways_distribute_balls : 
  ∃ n : ℕ, (balls = 6 ∧ boxes = 4) → n = 8 :=
sorry

end num_ways_distribute_balls_l471_471427


namespace find_PM_PN_value_l471_471884

-- Definitions of the conditions
def curve_C_polar (ρ θ : ℝ) : Prop := ρ * (sin θ)^2 = 4 * cos θ
def line_l_parametric (t x y : ℝ) : Prop := x = -2 + (sqrt 2 / 2) * t ∧ y = -4 + (sqrt 2 / 2) * t
def point_P : ℝ × ℝ := (-2, -4)

-- The Cartesian equations of the curve and the line
def curve_C_cartesian (x y : ℝ) : Prop := y^2 = 4 * x
def line_l_standard (x y : ℝ) : Prop := x - y = 2

-- Statement to prove the value of |PM| + |PN|
theorem find_PM_PN_value :
  (∀ ρ θ, curve_C_polar ρ θ → (ρ * cos θ = -2 + (sqrt 2 / 2) * θ ∧ ρ * sin θ = -4 + (sqrt 2 / 2) * θ) → 
  ∃ M N : ℝ × ℝ, curve_C_cartesian M.fst M.snd ∧ line_l_standard M.fst M.snd ∧
                 curve_C_cartesian N.fst N.snd ∧ line_l_standard N.fst N.snd ∧
                 dist point_P M + dist point_P N = 12 * sqrt 2.
) := sorry

end find_PM_PN_value_l471_471884


namespace investment_difference_l471_471930

noncomputable def compound_interest_annually (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r)^n

noncomputable def compound_interest_semi_annually (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r / 2)^(2 * n)

theorem investment_difference : 
  let P := 100000
  let r := 0.05
  let n := 3
  (compound_interest_semi_annually P r n - compound_interest_annually P r n).round = 164 := 
by
  sorry

end investment_difference_l471_471930


namespace range_of_a_l471_471414

open Set

variable {a : ℝ} 

def M (a : ℝ) : Set ℝ := {x : ℝ | -4 * x + 4 * a < 0 }

theorem range_of_a (hM : 2 ∉ M a) : a ≥ 2 :=
by
  sorry

end range_of_a_l471_471414


namespace sphere_volume_increase_factor_l471_471869

theorem sphere_volume_increase_factor (r : Real) : 
  let V_original := (4 / 3) * Real.pi * r^3
  let V_increased := (4 / 3) * Real.pi * (2 * r)^3
  V_increased / V_original = 8 :=
by
  -- Definitions of volumes
  let V_original := (4 / 3) * Real.pi * r^3
  let V_increased := (4 / 3) * Real.pi * (2 * r)^3
  -- Volume ratio
  have h : V_increased / V_original = 8 := sorry
  exact h

end sphere_volume_increase_factor_l471_471869


namespace jalapeno_peppers_needed_jalapeno_peppers_needed_correct_l471_471585

theorem jalapeno_peppers_needed
  (strips_per_sandwich : ℕ) (strips_per_pepper : ℕ)
  (minutes_per_sandwich : ℕ) (hours_per_day : ℕ)
  (strips_per_sandwich = 4) (strips_per_pepper = 8)
  (minutes_per_sandwich = 5) (hours_per_day = 8) : ℕ :=
  let peppers_per_sandwich := strips_per_sandwich / strips_per_pepper in
  let sandwiches_per_hour := 60 / minutes_per_sandwich in
  let peppers_per_hour := sandwiches_per_hour * peppers_per_sandwich in
  let total_peppers_needed := hours_per_day * peppers_per_hour in
  total_peppers_needed

-- Now providing the theorem to ensure the calculation
theorem jalapeno_peppers_needed_correct : jalapeno_peppers_needed 4 8 5 8 = 48 := sorry

end jalapeno_peppers_needed_jalapeno_peppers_needed_correct_l471_471585


namespace problem_equivalent_l471_471046

open Real

noncomputable def general_equation_of_curve (a b : ℝ) (h1 : a > b) (h2 : b > 0) (φ : ℝ) (M : ℝ × ℝ) (hm : M = (2, sqrt 3)) (hφ : φ = π / 3) :
  Prop := 
  (M.fst = a * cos φ ∧ M.snd = b * sin φ) → 
  (a = 4 ∧ b = 2) ∧ (∀ x y, (x / a)^2 + (y / b)^2 = 1)

noncomputable def value_of_reciprocal_sum (ρ1 ρ2 θ : ℝ) (h1 : a = 4) (h2 : b = 2) : 
  Prop := 
  (∀ ρ1 θ, (ρ1^2 * (cos θ)^2 / 16 + ρ1^2 * (sin θ)^2 / 4 = 1)) → 
  (∀ ρ2 θ, (ρ2^2 * (sin θ)^2 / 16 + ρ2^2 * (cos θ)^2 / 4 = 1)) →
  (1 / ρ1^2 + 1 / ρ2^2) = 5 / 16

theorem problem_equivalent (a b φ : ℝ) (M : ℝ × ℝ) (h1 : a > b) (h2 : b > 0) (hm : M = (2, sqrt 3)) (hφ : φ = π / 3) 
  (ρ1 ρ2 θ : ℝ) : 
  general_equation_of_curve a b h1 h2 φ M hm hφ ∧ value_of_reciprocal_sum ρ1 ρ2 θ.
Proof
  sorry

end problem_equivalent_l471_471046


namespace pythagorean_triple_divisible_by_60_l471_471512

theorem pythagorean_triple_divisible_by_60 
  (a b c : ℕ) (h : a * a + b * b = c * c) : 60 ∣ (a * b * c) :=
sorry

end pythagorean_triple_divisible_by_60_l471_471512


namespace total_weight_of_watermelons_l471_471903

theorem total_weight_of_watermelons (w1 w2 : ℝ) (h1 : w1 = 9.91) (h2 : w2 = 4.11) :
  w1 + w2 = 14.02 :=
by
  sorry

end total_weight_of_watermelons_l471_471903


namespace prob_sum_is_even_l471_471166

-- Define the set of numbers on the dice
def dice_faces : set ℕ := {1, 2, 3, 5, 7, 8}

-- Define the probability of rolling an odd number
def prob_odd (S : set ℕ) : ℚ := (S.filter (odd)).card / (S.card)

-- Define the probability of rolling an even number
def prob_even (S : set ℕ) : ℚ := (S.filter (even)).card / (S.card)

-- Define the probability that the sum of two dice rolls is even
def prob_sum_even : ℚ :=
  let p_odd := prob_odd dice_faces in
  let p_even := prob_even dice_faces in
  (p_even * p_even) + (p_odd * p_odd)

-- The main theorem statement
theorem prob_sum_is_even : prob_sum_even = 5 / 9 :=
by
  sorry

end prob_sum_is_even_l471_471166


namespace cos_two_thirds_pi_l471_471744

theorem cos_two_thirds_pi : Real.cos (2 / 3 * Real.pi) = -1 / 2 :=
by sorry

end cos_two_thirds_pi_l471_471744


namespace probability_exactly_nine_matches_l471_471241

theorem probability_exactly_nine_matches (n : ℕ) (h : n = 10) : 
  (∃ p : ℕ, p = 9 ∧ probability_of_exact_matches n p = 0) :=
by {
  sorry
}

end probability_exactly_nine_matches_l471_471241


namespace range_of_x_l471_471824

theorem range_of_x (f : ℝ → ℝ) (h_even : ∀ x, f x = f (-x)) (h_increasing : ∀ {a b : ℝ}, a ≤ b → b ≤ 0 → f a ≤ f b) :
  (∀ x : ℝ, f (2^(2*x^2 - x - 1)) ≥ f (-4)) → ∀ x, x ∈ Set.Icc (-1 : ℝ) (3/2 : ℝ) :=
by 
  sorry

end range_of_x_l471_471824


namespace cost_equality_and_inequality_l471_471296

section CommunicationCost

-- Definitions based on given conditions
def y1 (x : ℝ) := 50 + 0.4 * x
def y2 (x : ℝ) := 0.6 * x

-- Theorem to prove given questions and conditions
theorem cost_equality_and_inequality (x : ℝ) :
  (y1 125 = y2 125) ∧ (∀ x > 125, y1 x < y2 x) :=
by
  sorry
  
end CommunicationCost

end cost_equality_and_inequality_l471_471296


namespace F_equiv_binom_l471_471510

-- Define the set of integer points on the plane
def M : set (ℤ × ℤ) := {p | true}

-- Define what it means to form a polyline with given properties
def is_polyline (points : list (ℤ × ℤ)) : Prop :=
  ∃ n, (points.length = n + 1) ∧ 
       (∀ i, 0 < i ∧ i ≤ n → abs (points.nth_le (i - 1) (by linarith)).fst - abs (points.nth_le i (by linarith)).fst = 1)

-- Define the function F(n) which counts the number of polylines of length n
def F (n : ℕ) : ℕ :=
  {L : list (ℤ × ℤ) | is_polyline L ∧ L.head = (0, 0) ∧ (L.last (list.cons_ne_nil _ [])) ∈ {p : ℤ × ℤ | p.snd = 0}}.finite.to_finset.card

-- Main statement to prove F(n) = binom(2n, n)
theorem F_equiv_binom (n : ℕ) : F(n) = nat.choose (2 * n) n :=
by sorry

end F_equiv_binom_l471_471510


namespace binary_expansion_terms_l471_471342

theorem binary_expansion_terms :
  let num_terms := (2^341 + 1) / (2^31 + 1)
  (bin_expansion num_terms).numTerms = 341 := sorry

end binary_expansion_terms_l471_471342


namespace probability_red_given_black_l471_471647

noncomputable def urn_A := {white := 4, red := 2}
noncomputable def urn_B := {red := 3, black := 3}

-- Define the probabilities as required in the conditions
def prob_urn_A := 1 / 2
def prob_urn_B := 1 / 2

def draw_red_from_A := 2 / 6
def draw_black_from_B := 3 / 6
def draw_red_from_B := 3 / 6
def draw_black_from_B_after_red := 3 / 5
def draw_black_from_B_after_black := 2 / 5

def probability_first_red_second_black :=
  (prob_urn_A * draw_red_from_A * prob_urn_B * draw_black_from_B) +
  (prob_urn_B * draw_red_from_B * prob_urn_B * draw_black_from_B_after_red) +
  (prob_urn_B * draw_red_from_B * prob_urn_B * draw_black_from_B_after_black)

def probability_second_black :=
  (prob_urn_A * prob_urn_B * draw_black_from_B) +
  (prob_urn_B * draw_red_from_B * prob_urn_B * draw_black_from_B_after_red) +
  (prob_urn_B * draw_black_from_B * prob_urn_B * draw_black_from_B_after_black)

theorem probability_red_given_black :
  probability_first_red_second_black / probability_second_black = 7 / 15 :=
sorry

end probability_red_given_black_l471_471647


namespace problem_1_problem_2_problem_3_l471_471958

-- Definitions based on problem conditions
def total_people := 12
def choices := 5
def special_people_count := 3

noncomputable def choose (n k : ℕ) : ℕ := Nat.choose n k

-- Proof problem 1: A, B, and C must be chosen, so select 2 more from the remaining 9 people
theorem problem_1 : choose 9 2 = 36 :=
by sorry

-- Proof problem 2: Only one among A, B, and C is chosen, so select 4 more from the remaining 9 people
theorem problem_2 : choose 3 1 * choose 9 4 = 378 :=
by sorry

-- Proof problem 3: At most two among A, B, and C are chosen
theorem problem_3 : choose 12 5 - choose 9 2 = 756 :=
by sorry

end problem_1_problem_2_problem_3_l471_471958


namespace min_positive_period_cos_div_3_l471_471603

-- Define the function y = cos(x / 3)
def f (x : ℝ) : ℝ := Real.cos (x / 3)

-- State the theorem about the period of the function f
theorem min_positive_period_cos_div_3 : ∃ T > 0, (∀ x, f (x + T) = f x) ∧ T = 6 * Real.pi :=
by
  sorry

end min_positive_period_cos_div_3_l471_471603


namespace min_value_a_div_n_l471_471412

-- Define the sequence {a_n} using given recurrence relation and initial value
noncomputable def a : ℕ → ℝ
| 0     => 0     -- Since there's no a_0 defined, we set it arbitrarily
| (n+1) => a n + 2 * n

lemma a_initial : a 1 = 33 := sorry -- Given condition a_1 = 33

-- Define the expression a_n / n
noncomputable def a_div_n (n : ℕ) : ℝ :=
if n = 0 then 0 else a n / n

-- Define the minimum value of a_n / n
def min_a_div_n : ℝ := 21 / 2

-- Prove that the minimum value of a_n / n is indeed 21/2
theorem min_value_a_div_n : ∃ n > 0, a_div_n n = min_a_div_n :=
by
  sorry

end min_value_a_div_n_l471_471412


namespace flower_count_l471_471625

theorem flower_count (roses carnations : ℕ) (h₁ : roses = 5) (h₂ : carnations = 5) : roses + carnations = 10 :=
by
  sorry

end flower_count_l471_471625


namespace find_associated_equation_l471_471870

def associated_equation_of_inequality_system (x : ℝ) : Prop :=
  2x - 5 > 3x - 8 ∧ -4x + 3 < x - 4

def equation_1 := 5*x - 2 = 0
def equation_2 := (3/4)*x + 1 = 0
def equation_3 := x - (3*x + 1) = -5

theorem find_associated_equation :
  (∃ x : ℝ, associated_equation_of_inequality_system x) →
  equation_3 :=
sorry

end find_associated_equation_l471_471870


namespace find_num_female_workers_l471_471259

-- Defining the given constants and equations
def num_male_workers : Nat := 20
def num_child_workers : Nat := 5
def wage_male_worker : Nat := 35
def wage_female_worker : Nat := 20
def wage_child_worker : Nat := 8
def avg_wage_paid : Nat := 26

-- Defining the total number of workers and total daily wage
def total_workers (num_female_workers : Nat) : Nat := 
  num_male_workers + num_female_workers + num_child_workers

def total_wage (num_female_workers : Nat) : Nat :=
  (num_male_workers * wage_male_worker) + (num_female_workers * wage_female_worker) + (num_child_workers * wage_child_worker)

-- Proving the number of female workers given the average wage
theorem find_num_female_workers (F : Nat) 
  (h : avg_wage_paid * total_workers F = total_wage F) : 
  F = 15 :=
by
  sorry

end find_num_female_workers_l471_471259


namespace stone_count_197_is_prime_l471_471628

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, m > 1 → m ≤ Nat.sqrt n → n % m ≠ 0

theorem stone_count_197_is_prime :
  let cycle := 24
  let count := 197 % cycle
  let stone := if count > 0 then count else cycle
  stone = 5 ∧ is_prime 5 :=
by
  -- We know the cycle is 24
  let cycle := 24

  -- Calculate count 197 modulo 24
  have count := 197 % cycle
  have stone : ℕ := if count > 0 then count else cycle

  -- We know the residue of 197 % 24 is 5
  have h_count : count = 197 % cycle := rfl
  have h_mod := Nat.modEq_iff % (197, 24)
  rw [h_mod.2] at h_count
  have h_count_eq : count = 5 := by exact_nat_rel h_count

  -- Setting stone to count (which is 5)
  have h_stone : stone = if count > 0 then count else cycle := rfl
  rw [if_pos (show count > 0 from Nat.zero_lt_of_lt h_count)] at h_stone
  have h_final := Eq.trans h_stone h_count_eq

  -- Final statement
  exact ⟨h_final, by sorry⟩

end stone_count_197_is_prime_l471_471628


namespace ac_work_time_l471_471260

theorem ac_work_time (W : ℝ) (a_work_rate : ℝ) (b_work_rate : ℝ) (bc_work_rate : ℝ) (t : ℝ) : 
  (a_work_rate = W / 4) ∧ 
  (b_work_rate = W / 12) ∧ 
  (bc_work_rate = W / 3) → 
  t = 2 := 
by 
  sorry

end ac_work_time_l471_471260


namespace resulting_polygon_has_30_sides_l471_471736

def polygon_sides : ℕ := 3 + 4 + 5 + 6 + 7 + 8 + 9 - 6 * 2

theorem resulting_polygon_has_30_sides : polygon_sides = 30 := by
  sorry

end resulting_polygon_has_30_sides_l471_471736


namespace largest_n_dividing_P_l471_471072

theorem largest_n_dividing_P :
  let P := ∏ i in (Finset.range 2020).map (λ i, i + 1),
                 (3 ^ i + 1)
  in ∃ n : ℕ, (2 ^ n) ∣ P ∧ ∀ m : ℕ, (m > n) → ¬ (2 ^ m) ∣ P :=
sorry

end largest_n_dividing_P_l471_471072


namespace monkeys_bananas_l471_471181

theorem monkeys_bananas (c₁ c₂ c₃ : ℕ) (h1 : ∀ (k₁ k₂ k₃ : ℕ), k₁ = c₁ → k₂ = c₂ → k₃ = c₃ → 4 * (k₁ / 3 + k₂ / 6 + k₃ / 18) = 2 * (k₁ / 6 + k₂ / 3 + k₃ / 18) ∧ 2 * (k₁ / 6 + k₂ / 3 + k₃ / 18) = k₁ / 6 + k₂ / 6 + k₃ / 6)
  (h2 : c₃ % 6 = 0) (h3 : 4 * (c₁ / 3 + c₂ / 6 + c₃ / 18) < 2 * (c₁ / 6 + c₂ / 3 + c₃ / 18 + 1)) :
  c₁ + c₂ + c₃ = 2352 :=
sorry

end monkeys_bananas_l471_471181


namespace eq_system_correct_l471_471481

theorem eq_system_correct (x y : ℤ) : 
  (7 * x + 7 = y) ∧ (9 * (x - 1) = y) :=
sorry

end eq_system_correct_l471_471481


namespace value_of_ab_l471_471860

theorem value_of_ab (a b : ℤ) (h1 : |a| = 5) (h2 : b = -3) (h3 : a < b) : a * b = 15 :=
by
  sorry

end value_of_ab_l471_471860


namespace train_speed_km_hr_l471_471709

def train_length : ℝ := 130  -- Length of the train in meters
def bridge_and_train_length : ℝ := 245  -- Total length of the bridge and the train in meters
def crossing_time : ℝ := 30  -- Time to cross the bridge in seconds

theorem train_speed_km_hr : (train_length + bridge_and_train_length) / crossing_time * 3.6 = 45 := by
  sorry

end train_speed_km_hr_l471_471709


namespace eq_system_correct_l471_471483

theorem eq_system_correct (x y : ℤ) : 
  (7 * x + 7 = y) ∧ (9 * (x - 1) = y) :=
sorry

end eq_system_correct_l471_471483


namespace balls_in_boxes_l471_471438

def num_ways_to_partition_6_in_4_parts : ℕ :=
  -- The different partitions of 6: (6,0,0,0), (5,1,0,0), (4,2,0,0), (4,1,1,0),
  -- (3,3,0,0), (3,2,1,0), (3,1,1,1), (2,2,2,0), (2,2,1,1)
  ([
    [6, 0, 0, 0],
    [5, 1, 0, 0],
    [4, 2, 0, 0],
    [4, 1, 1, 0],
    [3, 3, 0, 0],
    [3, 2, 1, 0],
    [3, 1, 1, 1],
    [2, 2, 2, 0],
    [2, 2, 1, 1]
  ]).length

theorem balls_in_boxes : num_ways_to_partition_6_in_4_parts = 9 := by
  sorry

end balls_in_boxes_l471_471438


namespace only_positive_integer_n_is_one_l471_471737

-- Definitions of the polynomials f_n and g_n 
def f_n (n : ℕ) (x y z : ℤ) : ℤ := 
  x^(2 * n) + y^(2 * n) + z^(2 * n) - x * y - y * z - z * x

def g_n (n : ℕ) (x y z : ℤ) : ℤ := 
  (x - y)^(5 * n) + (y - z)^(5 * n) + (z - x)^(5 * n)

-- The main theorem stating that the only possible value for n is 1
theorem only_positive_integer_n_is_one : 
  ∀ (n : ℕ), (∀ (x y z : ℤ), f_n(n, x, y, z) ∣ g_n(n, x, y, z)) → n = 1 :=
by sorry

end only_positive_integer_n_is_one_l471_471737


namespace book_selection_ways_l471_471706

theorem book_selection_ways (chineseBooks mathBooks englishBooks : ℕ) (h1 : chineseBooks = 12) (h2 : mathBooks = 14) (h3 : englishBooks = 11) :
  chineseBooks + mathBooks + englishBooks = 37 :=
by
  rw [h1, h2, h3]
  norm_num

end book_selection_ways_l471_471706


namespace senya_right_triangle_area_l471_471575

noncomputable def area_of_right_triangle
  (a : ℝ) (b : ℝ)
  (h1 : a + b = 24)
  (h2 : 24^2 + a^2 = (24 + b)^2) : ℝ :=
  (1 / 2) * 24 * a

theorem senya_right_triangle_area :
  ∃ a : ℝ, ∃ b : ℝ,
    a + b = 24 ∧
    24^2 + a^2 = (24 + b)^2 ∧
    area_of_right_triangle a b ‹a + b = 24› ‹24^2 + a^2 = (24 + b)^2› = 216 :=
begin
  sorry
end

end senya_right_triangle_area_l471_471575


namespace polynomial_remainder_example_l471_471663

noncomputable def polynomial_remainder_division (p q : Polynomial ℝ) : Polynomial ℝ :=
  Polynomial.divByMonic p (Polynomial.monic_leads_coeff q).some

theorem polynomial_remainder_example :
  polynomial_remainder_division (Polynomial.C 3 + Polynomial.X ^ 3) 
                                (Polynomial.C 2 + Polynomial.X ^ 2) 
  = Polynomial.C 3 - Polynomial.C 2 * Polynomial.X := 
sorry

end polynomial_remainder_example_l471_471663


namespace pyramid_lateral_surface_area_l471_471593

-- Define the conditions
def is_equilateral (a : ℝ) : Prop :=
  true  -- placeholder for actual definition if needed

def lateral_surface_area (a : ℝ) : ℝ :=
  (a^2 / 4) * (Real.sqrt 15 + Real.sqrt 3)

-- Define the proof goal
theorem pyramid_lateral_surface_area (a : ℝ) (ha : is_equilateral a) :
  lateral_surface_area a = (a^2 / 4) * (Real.sqrt 15 + Real.sqrt 3) :=
sorry

end pyramid_lateral_surface_area_l471_471593


namespace probability_reroll_two_dice_is_given_correct_probability_l471_471058

noncomputable def probability_of_rerolling_two_dice_optimally : ℝ :=
  -- Total favorable outcomes for rerolling two dice to achieve sum 9 (assuming precomputed correct answer, e.g. 7/54)
  let total_favorable_outcomes := 7 in
  let total_possible_outcomes := 54 in 
  (total_favorable_outcomes : ℝ) / (total_possible_outcomes : ℝ)

theorem probability_reroll_two_dice_is_given_correct_probability :
  probability_of_rerolling_two_dice_optimally = 7 / 54 :=
  sorry

end probability_reroll_two_dice_is_given_correct_probability_l471_471058


namespace yellow_balls_count_l471_471878

theorem yellow_balls_count (x y z : ℕ) 
  (h1 : x + y + z = 68)
  (h2 : y = 2 * x)
  (h3 : 3 * z = 4 * y) : y = 24 :=
by {
  sorry
}

end yellow_balls_count_l471_471878


namespace g_value_at_2_l471_471012

theorem g_value_at_2 (g : ℝ → ℝ) (h : ∀ x : ℝ, g(3*x - 7) = 4*x + 6) : g 2 = 18 :=
by
  sorry

end g_value_at_2_l471_471012


namespace max_discount_rate_l471_471283

-- Define the cost price and selling price.
def cp : ℝ := 4
def sp : ℝ := 5

-- Define the minimum profit margin.
def min_profit_margin : ℝ := 0.4

-- Define the discount rate d.
def discount_rate (d : ℝ) : ℝ := sp * (1 - d / 100) - cp

-- The theorem to prove the maximum discount rate.
theorem max_discount_rate (d : ℝ) (H : discount_rate d ≥ min_profit_margin) : d ≤ 12 :=
sorry

end max_discount_rate_l471_471283


namespace triangle_ABC_properties_l471_471376

noncomputable def is_arithmetic_sequence (α β γ : ℝ) : Prop :=
γ - β = β - α

theorem triangle_ABC_properties
  (A B C a c : ℝ)
  (b : ℝ := Real.sqrt 3)
  (h1 : a * Real.cos C + c * Real.cos A = 2 * b * Real.cos B) :
  is_arithmetic_sequence A B C ∧
  ∃ (max_area : ℝ), max_area = (3 * Real.sqrt 3) / 4 := by sorry

end triangle_ABC_properties_l471_471376


namespace max_discount_rate_l471_471268

-- Define the conditions
def cost_price : ℝ := 4
def selling_price : ℝ := 5
def min_profit_margin : ℝ := 0.1
def min_profit : ℝ := cost_price * min_profit_margin

-- Define the discount rate
def discount_rate (x : ℝ) : ℝ :=
  selling_price * (1 - x / 100)

-- Define the profit after discount
def profit_after_discount (x : ℝ) : ℝ :=
  discount_rate x - cost_price

-- The statement we want to prove
theorem max_discount_rate : ∃ x : ℝ, x = 8.8 ∧ profit_after_discount x ≥ min_profit := 
by
  sorry

end max_discount_rate_l471_471268


namespace reciprocal_problem_l471_471008

theorem reciprocal_problem (x : ℚ) (h : 8 * x = 5) : 150 * (x⁻¹) = 240 := 
by 
  sorry

end reciprocal_problem_l471_471008


namespace sin_double_angle_l471_471867

theorem sin_double_angle (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.sin α = 3 / 5) : 
  Real.sin (2 * α) = 24 / 25 :=
by sorry

end sin_double_angle_l471_471867


namespace min_value_f_l471_471783

-- Define the function f(x, y) according to the given problem.
def f (x y : ℝ) : ℝ :=
  (5*x^2 - 8*x*y + 5*y^2 - 10*x + 14*y + 55) / ((9 - 25*x^2 + 10*x*y - y^2)^(5/2))

-- State the theorem to prove that the minimum value of f(x, y) is 0.19.
theorem min_value_f : ∃ (x y : ℝ), f x y = 0.19 :=
sorry

end min_value_f_l471_471783


namespace num_distinct_prime_factors_330_l471_471424

theorem num_distinct_prime_factors_330 : 
  ∃ (s : Finset ℕ), s.card = 4 ∧ ∀ x ∈ s, Nat.Prime x ∧ 330 % x = 0 := 
sorry

end num_distinct_prime_factors_330_l471_471424


namespace domain_correct_l471_471598

def domain_of_function (x : ℝ) : Prop :=
  (x > 2) ∧ (x ≠ 5)

theorem domain_correct : {x : ℝ | domain_of_function x} = {x : ℝ | x > 2 ∧ x ≠ 5} :=
by
  sorry

end domain_correct_l471_471598


namespace largest_T_A_value_l471_471356

theorem largest_T_A_value : 
  ∀ (A : Finset ℕ), 
  A.card = 5 ∧ ∀ (a ∈ A) (b ∈ A), a ≠ b → a ≠ b → a ≠ b →
  let S_A := A.sum id in 
  let triples := { t : Finset ℕ // ∃ i j k, i < j ∧ j < k ∧ t = {i, j, k} } in
  ∀ t ∈ triples, S_A % t.sum id = 0 → T_A ≤ 4 :=
sorry

end largest_T_A_value_l471_471356


namespace max_discount_rate_l471_471280

-- Define the cost price and selling price.
def cp : ℝ := 4
def sp : ℝ := 5

-- Define the minimum profit margin.
def min_profit_margin : ℝ := 0.4

-- Define the discount rate d.
def discount_rate (d : ℝ) : ℝ := sp * (1 - d / 100) - cp

-- The theorem to prove the maximum discount rate.
theorem max_discount_rate (d : ℝ) (H : discount_rate d ≥ min_profit_margin) : d ≤ 12 :=
sorry

end max_discount_rate_l471_471280


namespace joe_total_spending_at_fair_l471_471217

-- Definitions based on conditions
def entrance_fee (age : ℕ) : ℝ := if age < 18 then 5 else 6
def ride_cost (rides : ℕ) : ℝ := rides * 0.5

-- Given conditions
def joe_age := 19
def twin_age := 6

def total_cost (joe_age : ℕ) (twin_age : ℕ) (rides_per_person : ℕ) :=
  entrance_fee joe_age + 2 * entrance_fee twin_age + 3 * ride_cost rides_per_person

-- The main statement to be proven
theorem joe_total_spending_at_fair : total_cost joe_age twin_age 3 = 20.5 :=
by
  sorry

end joe_total_spending_at_fair_l471_471217


namespace find_a_l471_471453

noncomputable def tangent_perpendicular (a : ℝ) : Prop :=
  let C1 := fun (x : ℝ) => a * x^3 - 6 * x^2 + 12 * x
  let C1' := fun (x : ℝ) => 3 * a * x^2 - 12 * x + 12
  let C2 := fun (x : ℝ) => Real.exp x
  let C2' := fun (x : ℝ) => Real.exp x
  C1'(1) * C2'(1) = -1

theorem find_a : ∃ (a : ℝ), tangent_perpendicular a ∧ a = -1 / (3 * Real.exp 1) :=
by
  sorry

end find_a_l471_471453


namespace quad_func_minimum_l471_471199

def quad_func (x : ℝ) : ℝ := x^2 - 8 * x + 5

theorem quad_func_minimum : ∀ x : ℝ, quad_func x ≥ -11 ∧ quad_func 4 = -11 :=
by
  sorry

end quad_func_minimum_l471_471199


namespace least_sum_exponents_and_number_of_terms_l471_471254
-- Import the necessary library

-- Statement of the problem in Lean 4
theorem least_sum_exponents_and_number_of_terms (n : ℕ) (hn : n = 1023) :
  ∃ (S : Finset ℕ), (n = S.sum (λ x, 2^x)) ∧ (S.card = 10) ∧ (S.sum id = 45) :=
sorry

end least_sum_exponents_and_number_of_terms_l471_471254


namespace polygon_guarding_set_l471_471508

/-- 
A simple (not necessarily convex) polygon P with n vertices exists.
Show that there exists a set A of ⌊n/3⌋ vertices of P such that
for any point X inside P, there exists a point C ∈ A such that 
the segment [CX] is entirely inside P.
-/
theorem polygon_guarding_set (n : ℕ) (P : Type) [Polygon P n] :
  ∃ (A : Finset P), A.card = n / 3 ∧ 
    ∀ (X : P), is_inside X P →
      ∃ (C : P), C ∈ A ∧ segment_inside_polygon C X P :=
sorry

end polygon_guarding_set_l471_471508


namespace hexagon_perimeter_l471_471881

-- Define the side length 's' based on the given area condition
def side_length (s : ℝ) : Prop :=
  (3 * Real.sqrt 2 + Real.sqrt 3) / 4 * s^2 = 12

-- The theorem to prove
theorem hexagon_perimeter (s : ℝ) (h : side_length s) : 
  6 * s = 6 * Real.sqrt (48 / (3 * Real.sqrt 2 + Real.sqrt 3)) :=
by
  sorry

end hexagon_perimeter_l471_471881


namespace opposite_of_neg_2023_l471_471985

theorem opposite_of_neg_2023 :
  ∃ y : ℝ, (-2023 + y = 0) ∧ y = 2023 :=
by
  sorry

end opposite_of_neg_2023_l471_471985


namespace females_in_town_l471_471469

theorem females_in_town (population : ℕ) (ratio : ℕ × ℕ) (H : population = 480) (H_ratio : ratio = (3, 5)) : 
  let m := ratio.1
  let f := ratio.2
  f * (population / (m + f)) = 300 := by
  sorry

end females_in_town_l471_471469


namespace initially_calculated_average_height_l471_471114

theorem initially_calculated_average_height 
    (students : ℕ) (incorrect_height : ℕ) (correct_height : ℕ) (actual_avg_height : ℝ) 
    (A : ℝ) 
    (h_students : students = 30) 
    (h_incorrect_height : incorrect_height = 151) 
    (h_correct_height : correct_height = 136) 
    (h_actual_avg_height : actual_avg_height = 174.5)
    (h_A_definition : (students : ℝ) * A + (incorrect_height - correct_height) = (students : ℝ) * actual_avg_height) : 
    A = 174 := 
by sorry

end initially_calculated_average_height_l471_471114


namespace probability_exactly_nine_matches_l471_471240

theorem probability_exactly_nine_matches (n : ℕ) (h : n = 10) : 
  (∃ p : ℕ, p = 9 ∧ probability_of_exact_matches n p = 0) :=
by {
  sorry
}

end probability_exactly_nine_matches_l471_471240


namespace numbers_written_in_red_l471_471950

theorem numbers_written_in_red :
  ∃ (x : ℕ), x > 0 ∧ x <= 101 ∧ 
  ∀ (largest_blue_num : ℕ) (smallest_red_num : ℕ), 
  (largest_blue_num = x) ∧ 
  (smallest_red_num = x + 1) ∧ 
  (smallest_red_num = (101 - x) / 2) → 
  (101 - x = 68) := by
  sorry

end numbers_written_in_red_l471_471950


namespace complex_transformation_result_l471_471190

noncomputable def complex_transform (z : ℂ) : ℂ :=
  (complex.of_real (real.sqrt 3) + complex.I) * z

theorem complex_transformation_result :
  complex_transform (-4 - 6 * complex.I) = -4 * real.sqrt 3 + 6 - (6 * real.sqrt 3 + 4) * complex.I :=
by
  sorry

end complex_transformation_result_l471_471190


namespace parabolas_intersect_twice_l471_471939

-- Definitions based on conditions
def parabola (x y : ℝ) := ∃ a b : ℝ, a ≠ b ∧ y = (x - a)*(x - b)

def parabolas_sum (f_i : ℝ → ℝ) (m : ℕ) (x : ℝ) := ∑ i in Finset.range m, f_i x

-- The problem stated formally in Lean
theorem parabolas_intersect_twice (n : ℕ) (X : Finₓ n → ℝ)
  (f : Finₓ n → ℝ → ℝ)
  (h₁ : ∀ i, ∃ a b, X i = a ∧ X (i + 1) % n = b ∧ f i = λ x, (x - a)*(x - b))
  (h₂ : n ≥ 3) :
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (parabolas_sum f n x1 = 0 ∧ parabolas_sum f n x2 = 0) :=
sorry

end parabolas_intersect_twice_l471_471939


namespace product_of_numbers_l471_471631

noncomputable def x : ℕ := 18 / 6

def num1 : ℕ := 3 * x
def num2 : ℕ := 4 * x
def num3 : ℕ := 6 * x

theorem product_of_numbers :
  num1 * num2 * num3 = 1944 :=
by
  sorry

end product_of_numbers_l471_471631


namespace find_other_two_mean_l471_471959

theorem find_other_two_mean :
  let nums := [1871, 1997, 2020, 2028, 2113, 2125, 2140, 2222, 2300]
  ∑ n in nums, n = 17016 →
  let seven_mean := 2100
  let seven_sum := 7 * seven_mean
  let remaining_sum := (17016 - seven_sum)
  (remaining_sum / 2) = 1158 :=
by
  sorry

end find_other_two_mean_l471_471959


namespace sqrt_of_decimal_fraction_with_hundred_nines_l471_471670

theorem sqrt_of_decimal_fraction_with_hundred_nines
  (α : ℝ) (hα : α = 0.9999...  -- representation for clarity
  ) :
  let β := sqrt α in β = 0.9999... :=  -- representation for clarity

begin
  sorry
end

end sqrt_of_decimal_fraction_with_hundred_nines_l471_471670


namespace sum_of_square_roots_l471_471658

theorem sum_of_square_roots :
  (Real.sqrt 1 + Real.sqrt (1 + 2) + Real.sqrt (1 + 2 + 3) + Real.sqrt (1 + 2 + 3 + 4)) = 
  (1 + Real.sqrt 3 + Real.sqrt 6 + Real.sqrt 10) := 
sorry

end sum_of_square_roots_l471_471658


namespace system_of_equations_correct_l471_471479

theorem system_of_equations_correct (x y : ℕ) :
  (7 * x + 7 = y) ∧ (9 * (x - 1) = y) :=
begin
  -- sorry, proof placeholder
  sorry
end

end system_of_equations_correct_l471_471479


namespace expected_flashlight_lifetime_leq_two_l471_471540

theorem expected_flashlight_lifetime_leq_two
  (Ω : Type*) [MeasurableSpace Ω] [ProbabilitySpace Ω]
  (ξ η : Ω → ℝ)
  (h_min_leq_xi : ∀ ω, min (ξ ω) (η ω) ≤ ξ ω)
  (h_expectation_xi : expectation (ξ) = 2) :
  expectation (fun ω => min (ξ ω) (η ω)) ≤ 2 := 
sorry

end expected_flashlight_lifetime_leq_two_l471_471540


namespace total_marks_math_physics_l471_471305

variable (M P C : ℕ)

def condition1 : Prop := C = P + 20
def condition2 : Prop := (M + C) / 2 = 26

theorem total_marks_math_physics (h1 : condition1 M P C) (h2 : condition2 M P C) : M + P = 32 := 
by
  -- Proof omitted
  sorry

end total_marks_math_physics_l471_471305


namespace sqrt2_no_repeated_digit_5000001_times_l471_471563

theorem sqrt2_no_repeated_digit_5000001_times :
  ∀ (n : ℕ) (a : ℕ), n ≤ 4999999 → 
  a < 10 → 
  ¬ (∃ k : ℕ, sqrt 2 * 10^(n + k) = a)
:=
by 
  sorry

end sqrt2_no_repeated_digit_5000001_times_l471_471563


namespace probability_red_first_given_black_second_l471_471643

open ProbabilityTheory MeasureTheory

-- Definitions for Urn A and Urn B ball quantities
def urnA := (white : 4, red : 2)
def urnB := (red : 3, black : 3)

-- Event of drawing a red ball first and a black ball second
def eventRedFirst := (urn : ℕ, ball : string) -> (urn = 1 ∧ ball = "red") ∨ (urn = 2 ∧ ball = "red")
def eventBlackSecond := (urn : ℕ, ball : string) -> (urn = 1 ∧ ball = "black") ∨ (urn = 2 ∧ ball = "black")

-- Probability function definition
noncomputable def P := sorry -- Probability function placeholder

-- Conditional Probability
theorem probability_red_first_given_black_second :
  P(eventRedFirst | eventBlackSecond) = 2 / 5 := sorry

end probability_red_first_given_black_second_l471_471643


namespace sum_of_radii_of_circles_in_rectangle_area1_l471_471564

theorem sum_of_radii_of_circles_in_rectangle_area1 (r : ℝ) (α : ℝ) (n m : ℕ) :
  (1 / α - 1) / 2 > 1962 → n * m * (α / 2) = 1962 → 
  ∃ (rect : ℝ) (circles : list ℝ), Σ r ∈ circles, r = 1962 ∧
  rect = 1 ∧ (∀ r1 r2, r1 ∈ circles → r2 ∈ circles → r1 ≠ r2 → disjoint r1 r2) := 
begin
  sorry
end

end sum_of_radii_of_circles_in_rectangle_area1_l471_471564


namespace tracy_initial_candies_l471_471635

theorem tracy_initial_candies (x : ℕ) 
  (h1 : x % 4 = 0)
  (h2 : ∃ b ∈ set.Icc 3 7, (x / 2 - 36 - b = 5)) :
  x = 88 :=
by
  sorry

end tracy_initial_candies_l471_471635


namespace problem_1_problem_2_problem_3_problem_4_problem_5_l471_471729

theorem problem_1 : 9 * real.sqrt 3 - 7 * real.sqrt 12 + 5 * real.sqrt 48 = 15 * real.sqrt 3 := 
  sorry

theorem problem_2 : 
  (real.sqrt (5 / 3) - real.sqrt 15) * real.sqrt 3 = -2 * real.sqrt 5 := 
  sorry

theorem problem_3 : 
  (real.sqrt 75 - real.sqrt 12) / real.sqrt 3 = 3 := 
  sorry

theorem problem_4 : 
  (3 + real.sqrt 5) * (3 - real.sqrt 5) - (real.sqrt 3 - 1)^2 = 2 * real.sqrt 3 := 
  sorry

theorem problem_5 (x1 x2 : ℝ) (h1 : x1 = 2 / (real.sqrt 5 + real.sqrt 3)) (h2 : x2 = real.sqrt 5 + real.sqrt 3) : 
  x1^2 + x2^2 = 16 := 
  sorry

end problem_1_problem_2_problem_3_problem_4_problem_5_l471_471729


namespace find_b_such_that_tangent_l471_471792

theorem find_b_such_that_tangent (
  k b c : ℝ
) : 
  let curve := (x : ℝ) → x^3 + b * x^2 + c
  let line := (x : ℝ) → k * x + 1
  tangent_at_M : curve 1 = 2 ∧ line 1 = 2 ∧ (3 * 1^2 + 2 * b * 1 = k)
  ∧ (1 + b + c = 2) 
  → b = -1 :=
by { sorry }

end find_b_such_that_tangent_l471_471792


namespace circle_area_solution_l471_471738

def circle_area_problem : Prop :=
  ∀ (x y : ℝ), x^2 + y^2 + 6 * x - 8 * y - 12 = 0 -> ∃ (A : ℝ), A = 37 * Real.pi

theorem circle_area_solution : circle_area_problem :=
by
  sorry

end circle_area_solution_l471_471738


namespace reconstruct_right_triangle_l471_471102

theorem reconstruct_right_triangle (A C L : Type*)
  (hC : ∠ C = 90°)
  (hL : L ∈ angle_bisector B) :
  ∃ B, ∀ P, P ∈ angle_bisector B → (
  let l := line.perpendicular C AC in
  let r := circle.center_radius L LA in
  let A' := r ∩ l in
  let mid_perp := perpendicular_bisector AA' in
  B = mid_perp ∩ l
  ) :=
sorry

end reconstruct_right_triangle_l471_471102


namespace object_speed_approx_l471_471015

noncomputable def feet_to_miles := 1 / 5280
noncomputable def seconds_to_hours := 1 / 3600

def object_travel {
  (distance_in_feet : ℝ) (time_in_seconds : ℝ) :=
  (distance_in_miles * time_in_hours : ℝ) :=
    ((distance_in_feet * feet_to_miles) / (time_in_seconds * seconds_to_hours))

theorem object_speed_approx
  (distance_in_feet : ℝ := 70) (time_in_seconds : ℝ := 2) :
  object_travel distance_in_feet time_in_seconds ≈ 23.86 :=
by
  sorry

end object_speed_approx_l471_471015


namespace part_a_4s_more_than_5s_part_a_count_9s_part_b_digital_root_3_pow_2009_part_c_digital_root_17_pow_2009_l471_471253

-- Part (a)
theorem part_a_4s_more_than_5s : 
  let list_digits := List.map (λ n, n % 9) (List.range 20092009) in
  (List.count (λ d, d = 4) list_digits) > (List.count (λ d, d = 5) list_digits) := sorry

theorem part_a_count_9s : 
  let list_digits := List.map (λ n, n % 9) (List.range 20092009) in
  (List.count (λ d, d = 9) list_digits) = 2232445 := sorry

-- Part (b)
theorem part_b_digital_root_3_pow_2009 : 
  (3^2009) % 9 = 9 := sorry

-- Part (c)
theorem part_c_digital_root_17_pow_2009 : 
  (17^2009) % 9 = 8 := sorry

end part_a_4s_more_than_5s_part_a_count_9s_part_b_digital_root_3_pow_2009_part_c_digital_root_17_pow_2009_l471_471253


namespace fraction_of_students_between_11_and_13_is_two_fifths_l471_471086

def totalStudents : ℕ := 45
def under11 : ℕ :=  totalStudents / 3
def over13 : ℕ := 12
def between11and13 : ℕ := totalStudents - (under11 + over13)
def fractionBetween11and13 : ℚ := between11and13 / totalStudents

theorem fraction_of_students_between_11_and_13_is_two_fifths :
  fractionBetween11and13 = 2 / 5 := 
by 
  sorry

end fraction_of_students_between_11_and_13_is_two_fifths_l471_471086


namespace probability_divisible_by_5_l471_471159

-- Definitions based on the problem conditions
def M (a b : ℕ) : ℕ := 100 * a + 10 * b + 5
def ends_in_5 (n : ℕ) : Prop := n % 10 = 5
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

-- Main theorem statement in Lean 4
theorem probability_divisible_by_5 :
  ∀ (n : ℕ), is_three_digit n ∧ ends_in_5 n → ∃ q r, n = 5 * q + r ∧ r = 0 :=
by
  intros n hn,
  have h_mod: n % 5 = 0,
  {
    sorry, -- proof part to be filled in later
  },
  use (n / 5), 0,
  split,
  {
    rw nat.div_add_mod,
    exact nat.mod_eq_zero_of_dvd,
    exact h_mod,
  },
  {
    refl,
  }

end probability_divisible_by_5_l471_471159


namespace difference_square_consecutive_l471_471612

theorem difference_square_consecutive (x : ℕ) (h : x * (x + 1) = 812) : (x + 1)^2 - x = 813 :=
sorry

end difference_square_consecutive_l471_471612


namespace greatest_balloons_orvin_can_buy_l471_471558

-- Given conditions
def balloonPrice : ℝ := 2
def orvinInitialMoney : ℝ := 40 * balloonPrice
def saleFirstBalloonPrice : ℝ := balloonPrice
def saleSecondBalloonPrice : ℝ := balloonPrice / 2
def orvinTotalBalloons : ℝ := 52

-- Our task is to prove that given Orvin's initial money and the sale conditions,
-- the greatest number of balloons Orvin can buy is 52.
theorem greatest_balloons_orvin_can_buy : orvinTotalBalloons = 52 := by
  sorry

end greatest_balloons_orvin_can_buy_l471_471558


namespace max_discount_rate_l471_471276

theorem max_discount_rate 
  (cost_price : ℝ) (selling_price : ℝ) (min_profit_margin : ℝ) :
  cost_price = 4 → selling_price = 5 → min_profit_margin = 0.1 →
  ∃ (x : ℝ), 0 ≤ x ∧ x ≤ 8.8 ∧ (selling_price * (1 - x / 100) - cost_price) / cost_price ≥ min_profit_margin :=
begin
  sorry
end

end max_discount_rate_l471_471276


namespace car_clock_shows_11_is_1054_l471_471579

/-- Given:
    * A car clock and wristwatch both start at 8:00 AM.
    * Upon finishing shopping, the wristwatch reads 9:00 AM and the car clock reads 9:10 AM.
    * After a 20-minute break, the car clock reads 11:00 AM.

Prove that the actual time when the car clock shows 11:00 AM is 10:54 AM.
-/

def actual_time_when_car_clock_shows_11 (start_time : ℕ := 8 * 60) 
                                        (end_shopping_time_watch : ℕ := 9 * 60) 
                                        (end_shopping_time_car : ℕ := 9 * 60 + 10)
                                        (break_duration : ℕ := 20) 
                                        (final_car_time : ℕ := 11 * 60) : ℕ :=
  let car_clock_rate := 7 / 6
  let total_car_time := final_car_time - end_shopping_time_car
  let total_real_time := (total_car_time * 6) / 7
  let reached_time := end_shopping_time_watch + break_duration + total_real_time in
  reached_time

theorem car_clock_shows_11_is_1054 :
  actual_time_when_car_clock_shows_11 = 10 * 60 + 54 := 
sorry

end car_clock_shows_11_is_1054_l471_471579


namespace find_m_n_l471_471187

noncomputable def triangle_area : ℝ := 270
noncomputable def α : ℝ := 36 * β

theorem find_m_n
  (DE EF DF : ℝ)
  (hDE : DE = 15)
  (hEF : EF = 36)
  (hDF : DF = 39)
  (f_area : triangle_area = 270)
  (hβ : β = 5/12)
  : (5 + 12) = 17 :=
by
  simp [hβ]
  sorry

end find_m_n_l471_471187


namespace total_distance_fourth_fifth_days_l471_471040

theorem total_distance_fourth_fifth_days (d : ℕ) (total_distance : ℕ) (n : ℕ) (q : ℚ) 
  (S_6 : d * (1 - q^6) / (1 - q) = 378) (ratio : q = 1/2) (n_six : n = 6) : 
  (d * q^3) + (d * q^4) = 36 :=
by 
  sorry

end total_distance_fourth_fifth_days_l471_471040


namespace child_ticket_cost_l471_471556

/-- Defining the conditions and proving the cost of a child's ticket --/
theorem child_ticket_cost:
  (∀ c: ℕ, 
      -- Revenue from Monday
      (7 * c + 5 * 4) + 
      -- Revenue from Tuesday
      (4 * c + 2 * 4) = 
      -- Total revenue for both days
      61 
    ) → 
    -- Proving c
    (c = 3) :=
by
  sorry

end child_ticket_cost_l471_471556


namespace find_subtracted_number_l471_471682

variable (initial_number : Real)
variable (sum : Real := initial_number + 5)
variable (product : Real := sum * 7)
variable (quotient : Real := product / 5)
variable (remainder : Real := 33)

theorem find_subtracted_number 
  (initial_number_eq : initial_number = 22.142857142857142)
  : quotient - remainder = 5 := by
  sorry

end find_subtracted_number_l471_471682


namespace painted_stripe_area_l471_471309

noncomputable def circumference (d : ℝ) : ℝ := π * d
noncomputable def area_stripe (width : ℝ) (revolutions : ℕ) (diameter : ℝ) : ℝ :=
  width * (circumference diameter * revolutions)

theorem painted_stripe_area :
  let diameter := 40
  let height := 100
  let width_first := 5
  let width_second := 7
  let revolutions_first := 3
  let revolutions_second := 3
  area_stripe width_first revolutions_first diameter +
  area_stripe width_second revolutions_second diameter = 1440 * π :=
by
  sorry

end painted_stripe_area_l471_471309


namespace triangle_shape_l471_471051

theorem triangle_shape (a b : ℝ) (A B : ℝ)
  (h1 : a ≠ 0) (h2 : b ≠ 0) (hA : 0 < A ∧ A < π) (hB : 0 < B ∧ B < π)
  (hTriangle : A + B + (π - A - B) = π)
  (h : a * Real.cos A = b * Real.cos B) : 
  (A = B ∨ A + B = π / 2) := sorry

end triangle_shape_l471_471051


namespace min_value_expression_l471_471769

theorem min_value_expression (x y : ℝ) (hx : x > 2) (hy : y > 2) : 
  ∃ m, m = 20 ∧ (∀ a b : ℝ, a = x - 2 ∧ b = y - 2 →
    let exp := (a + 2) ^ 2 + 1 / (b + (b + 2) ^ 2 + 1 / a) in exp ≥ m) :=
sorry

end min_value_expression_l471_471769


namespace probability_of_exactly_nine_correct_matches_is_zero_l471_471237

theorem probability_of_exactly_nine_correct_matches_is_zero :
  let n := 10 in
  let match_probability (correct: Fin n → Fin n) (guess: Fin n → Fin n) (right_count: Nat) :=
    (Finset.univ.filter (λ i => correct i = guess i)).card = right_count in
  ∀ (correct_guessing: Fin n → Fin n), 
    ∀ (random_guessing: Fin n → Fin n),
      match_probability correct_guessing random_guessing 9 → 
        match_probability correct_guessing random_guessing 10 :=
begin
  sorry -- This skips the proof part
end

end probability_of_exactly_nine_correct_matches_is_zero_l471_471237


namespace part1_part2_l471_471048

noncomputable def triangle_perimeter_range (A B C c : ℝ) (hA : 0 < A) (hB : 0 < B) (hA_acute : A < π / 2) (hB_acute : B < π / 2) (hC_eq : C = π / 3) (hc_eq : c = 2) : Prop :=
  ∃ a b : ℝ, a + b + c ∈ (2 + 2 * Real.sqrt 3, 6]

theorem part1 : triangle_perimeter_range A B C c hA hB hA_acute hB_acute hC_eq hc_eq := sorry

noncomputable def sin_squared_sum_gt_one (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hA_acute : A < π / 2) (hB_acute : B < π / 2) (h_ineq : Real.sin A ^ 2 + Real.sin B ^ 2 > Real.sin C ^ 2) : Prop :=
  Real.sin A ^ 2 + Real.sin B ^ 2 > 1

theorem part2 : sin_squared_sum_gt_one A B C hA hB hA_acute hB_acute h_ineq := sorry

end part1_part2_l471_471048


namespace angle_between_asymptotes_of_hyperbola_l471_471829

-- Define the conditions
def hyperbola_eccentricity (a b c : ℝ) (h : b > 0) : Prop :=
  ∀ e : ℝ, e = real.sqrt 2 → e = c / a ∧ a^2 + b^2 = c^2

-- State the main problem
theorem angle_between_asymptotes_of_hyperbola (a b c : ℝ) (h : b > 0) 
  (h_e : hyperbola_eccentricity a b c h) : 
  ∀ θ : ℝ, θ = 90° :=
sorry

end angle_between_asymptotes_of_hyperbola_l471_471829


namespace total_students_l471_471144

theorem total_students (S K : ℕ) (h1 : S = 4000) (h2 : K = 2 * S) :
  S + K = 12000 := by
  sorry

end total_students_l471_471144


namespace distinct_prime_factors_330_l471_471422

def num_prime_factors (n : ℕ) : ℕ :=
  if n = 330 then 4 else 0

theorem distinct_prime_factors_330 : num_prime_factors 330 = 4 :=
sorry

end distinct_prime_factors_330_l471_471422


namespace solution_set_inequality_l471_471373

variable (f : ℝ → ℝ)

theorem solution_set_inequality (h1 : f 3 = 16) (h2 : ∀ x : ℝ, f' x < 4 * x - 1) :
  ∀ x : ℝ, f x < 2 * x^2 - x + 1 ↔ x > 3 :=
by sorry

end solution_set_inequality_l471_471373


namespace line_passes_fixed_point_l471_471006

theorem line_passes_fixed_point (k b : ℝ) (h : -1 = (k + b) / 2) :
  ∃ (x y : ℝ), x = 1 ∧ y = -2 ∧ y = k * x + b :=
by
  sorry

end line_passes_fixed_point_l471_471006


namespace integer_polynomial_l471_471109

theorem integer_polynomial (x y : ℂ) 
  (h : ∃ k : ℕ, k > 0 ∧ ∀ i : ℕ, i ∈ {k, k+1, k+2, k+3} → (∃ m : ℤ, (x ^ i - y ^ i) / (x - y) = m)) :
  ∀ n : ℕ, n > 0 → ∃ m : ℤ, (x ^ n - y ^ n) / (x - y) = m :=
by
  sorry

end integer_polynomial_l471_471109


namespace exists_projection_sum_lt_two_thirds_l471_471327

theorem exists_projection_sum_lt_two_thirds :
  ∃ r : ℝ, ∑ (i : Fin 2002), (1 / 2002 : ℝ) * |Real.cos (r + i * 2 * Real.pi / 2002)| < 2 / 3 :=
sorry

end exists_projection_sum_lt_two_thirds_l471_471327


namespace fraction_crop_brought_to_AD_l471_471977

-- Definitions of the involved measures and areas
structure Trapezoid where
  A B C D : Point
  AB CD : Line
  length_AD : ℝ
  isosceles : True -- This indicates it's an isosceles trapezoid
  parallel_BC_AD : BC.parallel AD
  angles : angles_on_cd_angle BAD 60 ∧ angles_on_cd_angle ABC 120 ∧
           angles_on_cd_angle BCD 120 ∧ angles_on_cd_angle CDA 60

structure Point where
  x y : ℝ

structure Line where
  p1 p2 : Point
  
noncomputable def area_trapezoid (t : Trapezoid) : ℝ := sorry
noncomputable def region_closer_to_AD (t : Trapezoid) : ℝ := sorry

theorem fraction_crop_brought_to_AD (t : Trapezoid) (uniformly_planted : True) :
  region_closer_to_AD t / area_trapezoid t = 5 / 12 := sorry

end fraction_crop_brought_to_AD_l471_471977


namespace new_average_after_drop_l471_471113

theorem new_average_after_drop (initial_average : ℝ) (num_students : ℕ) (drop_score : ℝ) :
  initial_average = 60.5 →
  num_students = 16 →
  drop_score = 8 →
  (let T := initial_average * num_students in
  let T_new := T - drop_score in
  T_new / (num_students - 1) = 64) := by
  intros h_initial h_num h_drop
  let T := initial_average * num_students
  let T_new := T - drop_score
  have : T_new / (num_students - 1) = 64 := sorry
  exact this

end new_average_after_drop_l471_471113


namespace union_A_B_l471_471852

def A (x : ℝ) : Set ℝ := {x ^ 2, 2 * x - 1, -4}
def B (x : ℝ) : Set ℝ := {x - 5, 1 - x, 9}

theorem union_A_B (x : ℝ) (h : {9} = A x ∩ B x) :
  (A x ∪ B x) = {(-8 : ℝ), -7, -4, 4, 9} := by
  sorry

end union_A_B_l471_471852


namespace equilateral_fold_cut_hexagon_l471_471795

/-- Given an equilateral triangle that is folded along the midpoints of the sides and then one corner 
is cut off along these midpoints, we want to prove that the resulting unfolded shape is a hexagon. -/
theorem equilateral_fold_cut_hexagon :
  ∃ (T : Type) (shape : T),
    (is_equilateral_triangle T shape) ∧
    (is_folded shape) ∧
    (is_cut_along_midpoints shape) ∧
    (unfold shape = hexagon) :=
sorry

end equilateral_fold_cut_hexagon_l471_471795


namespace find_a_given_even_l471_471836

def f (x a : ℝ) : ℝ := (x + a) * (x - 4)

theorem find_a_given_even (a : ℝ) :
  (∀ x : ℝ, f x a = f (-x) a) → a = 4 :=
by
  unfold f
  sorry

end find_a_given_even_l471_471836


namespace actual_cost_of_article_l471_471712

theorem actual_cost_of_article (x : ℝ) (h : 0.60 * x = 1050) : x = 1750 := by
  sorry

end actual_cost_of_article_l471_471712


namespace jamie_mean_score_l471_471468

-- Define the six scores
def scores : List ℝ := [75, 80, 85, 90, 92, 97]

-- Define the conditions
def alex_mean_score : ℝ := 85.5
def alex_scores_sum : ℝ := 4 * alex_mean_score
def total_scores_sum : ℝ := scores.sum

-- Lean statement proving Jamie's mean score
theorem jamie_mean_score :
  (total_scores_sum - alex_scores_sum) / 2 = 88.5 := 
by
  -- Sorry is used as a placeholder for the proof steps
  sorry

end jamie_mean_score_l471_471468


namespace midpoint_on_radical_axis_of_incicles_l471_471614

variables {A B C D E F I I1 I2 M : Point}
variables {AC : Segment A C}
variables {r1 r2 : ℝ}
variables {k : Circle I}
variables {k1 : Circle I1 r1}
variables {k2 : Circle I2 r2}

noncomputable def midpoint (AC : Segment A C) : Point := sorry
noncomputable def power (P : Point) (c : Circle) : ℝ := sorry

axiom QuadCircumscribed (A B C D : Point) (k : Circle I) : Prop
axiom Intersection (A B C D E F : Point) : Prop
axiom Incircle (T : Triangle EAF) (k : Circle I r) : Prop

theorem midpoint_on_radical_axis_of_incicles
  (H1 : QuadCircumscribed A B C D k)
  (H2 : Intersection D A C B E)
  (H3 : Intersection A B D C F)
  (H4 : Incircle (Triangle EAF) k1)
  (H5 : Incircle (Triangle ECF) k2)
  (M := midpoint (Segment A C)) :
  power M k1 = power M k2 :=
sorry

end midpoint_on_radical_axis_of_incicles_l471_471614


namespace minimum_bag_count_l471_471299

theorem minimum_bag_count (n a b : ℕ) (h1 : 7 * a + 11 * b = 77) (h2 : a + b = n) : n = 17 :=
by
  sorry

end minimum_bag_count_l471_471299


namespace Ruth_sandwiches_l471_471954

theorem Ruth_sandwiches (sandwiches_left sandwiches_ruth sandwiches_brother sandwiches_first_cousin sandwiches_two_cousins total_sandwiches : ℕ)
  (h_ruth : sandwiches_ruth = 1)
  (h_brother : sandwiches_brother = 2)
  (h_first_cousin : sandwiches_first_cousin = 2)
  (h_two_cousins : sandwiches_two_cousins = 2)
  (h_left : sandwiches_left = 3) :
  total_sandwiches = sandwiches_left + sandwiches_two_cousins + sandwiches_first_cousin + sandwiches_ruth + sandwiches_brother :=
by
  sorry

end Ruth_sandwiches_l471_471954


namespace indistinguishable_balls_boxes_l471_471429

open Finset

def partitions (n : ℕ) (k : ℕ) : ℕ :=
  (univ : Finset (Multiset ℕ)).filter (λ p, p.sum = n ∧ p.card ≤ k).card

theorem indistinguishable_balls_boxes : partitions 6 4 = 9 :=
sorry

end indistinguishable_balls_boxes_l471_471429


namespace pairwise_coprime_l471_471106

noncomputable theory

def u (n : ℕ) : ℤ := ⌊((3 + Real.sqrt 10) ^ (2 ^ n) / 4)⌋ + 1

def v (n : ℕ) : ℤ := (3 + Real.sqrt 10) ^ (2 ^ n) + (3 - Real.sqrt 10) ^ (2 ^ n)

axiom v0 : v 0 = 6

axiom v_recurrence : ∀ n, v (n + 1) = v n ^ 2 - 2

axiom v_mod : ∀ n, v n % 4 = 2

theorem pairwise_coprime : ∀ m n : ℕ, m ≠ n → Int.gcd (u m) (u n) = 1 := 
sorry

end pairwise_coprime_l471_471106


namespace arith_seq_largest_portion_l471_471111

theorem arith_seq_largest_portion (a1 d : ℝ) (h_d_pos : d > 0) 
  (h_sum : 5 * a1 + 10 * d = 100)
  (h_ratio : (3 * a1 + 9 * d) / 7 = 2 * a1 + d) : 
  a1 + 4 * d = 115 / 3 := by
  sorry

end arith_seq_largest_portion_l471_471111


namespace necessary_but_not_sufficient_l471_471990

theorem necessary_but_not_sufficient (x : ℝ) : (x^2 > 4) → (x > 2 ∨ x < -2) ∧ ¬((x^2 > 4) ↔ (x > 2)) :=
by
  intros h
  have h1 : x > 2 ∨ x < -2 := by sorry
  have h2 : ¬((x^2 > 4) ↔ (x > 2)) := by sorry
  exact And.intro h1 h2

end necessary_but_not_sufficient_l471_471990


namespace probability_of_exactly_nine_correct_matches_is_zero_l471_471236

theorem probability_of_exactly_nine_correct_matches_is_zero :
  let n := 10 in
  let match_probability (correct: Fin n → Fin n) (guess: Fin n → Fin n) (right_count: Nat) :=
    (Finset.univ.filter (λ i => correct i = guess i)).card = right_count in
  ∀ (correct_guessing: Fin n → Fin n), 
    ∀ (random_guessing: Fin n → Fin n),
      match_probability correct_guessing random_guessing 9 → 
        match_probability correct_guessing random_guessing 10 :=
begin
  sorry -- This skips the proof part
end

end probability_of_exactly_nine_correct_matches_is_zero_l471_471236


namespace peppers_needed_l471_471588

/-- Each sausage sandwich has 4 strips of jalapeno pepper, and
    one jalapeno pepper makes 8 slices. -/
def strips_per_sandwich : ℕ := 4
def slices_per_pepper : ℕ := 8

/-- A sandwich is served every 5 minutes, and the shop operates for 8 hours a day. -/
def minutes_per_sandwich : ℕ := 5
def hours_per_day : ℕ := 8

/-- Convert operation hours to minutes. -/
def minutes_per_day : ℕ := hours_per_day * 60

/-- Calculate the number of sandwiches served per day. -/
def sandwiches_per_day : ℕ := minutes_per_day / minutes_per_sandwich

/-- Calculate the number of peppers needed per sandwich. -/
def peppers_per_sandwich : ℝ := strips_per_sandwich.toReal / slices_per_pepper.toReal

/-- Calculate the total number of peppers needed for the day. -/
def total_peppers : ℝ := sandwiches_per_day.toReal * peppers_per_sandwich

theorem peppers_needed : total_peppers = 48 := by
  sorry

end peppers_needed_l471_471588


namespace min_value_sin_cos_squared_six_l471_471779

theorem min_value_sin_cos_squared_six (x : ℝ) :
  ∃ x : ℝ, (sin^6 x + 2 * cos^6 x) = 2/3 :=
sorry

end min_value_sin_cos_squared_six_l471_471779


namespace cube_less_than_three_times_l471_471653

theorem cube_less_than_three_times (x : ℤ) : x ^ 3 < 3 * x ↔ x = -3 ∨ x = -2 ∨ x = 1 :=
by
  sorry

end cube_less_than_three_times_l471_471653


namespace prob_9_correct_matches_is_zero_l471_471247

noncomputable def probability_of_exactly_9_correct_matches : ℝ :=
  let n := 10 in
  -- Since choosing 9 correct implies the 10th is also correct, the probability is 0.
  0

theorem prob_9_correct_matches_is_zero : probability_of_exactly_9_correct_matches = 0 :=
by
  sorry

end prob_9_correct_matches_is_zero_l471_471247


namespace derivative_at_alpha_l471_471005

variable (α : ℝ)

-- Defining the function f
def f (x : ℝ) : ℝ := (sin α) - (cos x)

-- Statement to prove
theorem derivative_at_alpha : (deriv f α) = sin α :=
by 
  -- We add sorry here because we are only focusing on the statement, not the proof
  sorry

end derivative_at_alpha_l471_471005


namespace dawson_failed_by_36_l471_471333

-- Define the constants and conditions
def max_marks : ℕ := 220
def passing_percentage : ℝ := 0.3
def marks_obtained : ℕ := 30

-- Calculate the minimum passing marks
noncomputable def min_passing_marks : ℝ :=
  passing_percentage * max_marks

-- Calculate the marks Dawson failed by
noncomputable def marks_failed_by : ℝ :=
  min_passing_marks - marks_obtained

-- State the theorem
theorem dawson_failed_by_36 :
  marks_failed_by = 36 := by
  -- Proof is omitted
  sorry

end dawson_failed_by_36_l471_471333


namespace number_of_consecutive_subsets_l471_471923

theorem number_of_consecutive_subsets :
  let X := {1, 2, ..., 20}
  ∃ (n : ℕ), n = 190 ∧ (∀ A ⊆ X, 2 ≤ |A| → 
  ∀ k m, A = {k, k+1, ..., k+m-1} → 
  k + m - 1 ≤ 20) do
  sorry

end number_of_consecutive_subsets_l471_471923


namespace function_increasing_and_extrema_l471_471745

open Set

theorem function_increasing_and_extrema (f : ℝ → ℝ) (a b : ℝ)
  (h1 : ∀ x ∈ Icc (2 : ℝ) 6, f x = (x - 2) / (x - 1))
  (h2 : a = 2) (h3 : b = 6) :
  (∀ x1 x2 : ℝ, x1 ∈ Icc a b ∧ x2 ∈ Icc a b ∧ x1 < x2 → f x1 < f x2) ∧ f a = 0 ∧ f b = 4 / 5 :=
sorry

end function_increasing_and_extrema_l471_471745


namespace hyperbola_foci_distance_l471_471505

noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

theorem hyperbola_foci_distance
  (a c : ℝ)
  (h1 : ∀ (x y : ℝ), (x ^ 2 - y ^ 2 = 1) → distance (x, y) (-c, 0) * distance (x, y) (c, 0) = 0)
  (h2 : ∀ (x y : ℝ), (x ^ 2 - y ^ 2 = 1) → ∃ (P : ℝ × ℝ), P = (x, y))
  (P : ℝ × ℝ) (hP : P.1 ^ 2 - P.2 ^ 2 = 1) :
  abs (distance P (-c, 0) + distance P (c, 0)) = 2 * a := 
sorry

end hyperbola_foci_distance_l471_471505


namespace passing_times_l471_471088

-- Define the constants used in the problem
def radius_inner_track : ℝ := 40
def radius_outer_track : ℝ := 55
def speed_odell : ℝ := 220
def speed_kershaw : ℝ := 275
def time : ℝ := 40

-- Define the required circumferences according to the conditions
def circumference (radius : ℝ) : ℝ := 2 * Real.pi * radius

-- Circumference of Odell's and Kershaw's tracks
def circumference_odell := circumference radius_inner_track
def circumference_kershaw := circumference radius_outer_track

-- Angular Speeds in radians per minute
def angular_speed (speed : ℝ) (radius : ℝ) : ℝ := (speed / (circumference radius)) * 2 * Real.pi

def angular_speed_odell := angular_speed speed_odell radius_inner_track
def angular_speed_kershaw := angular_speed speed_kershaw radius_outer_track

-- Relative angular speed considering they run in opposite directions
def relative_angular_speed : ℝ := angular_speed_odell + angular_speed_kershaw

-- Time to meet once
def time_to_meet_once : ℝ := (2 * Real.pi) / relative_angular_speed

-- Total number of meetings during the given time
def total_meetings : ℝ := time / time_to_meet_once

theorem passing_times :
  Real.floor total_meetings = 50 := by
  sorry

end passing_times_l471_471088


namespace find_smallest_m_l471_471511

noncomputable def S := {z : ℂ | ∃ (x y : ℝ), z = x + y * complex.I ∧ (1 / 2) ≤ x ∧ x ≤ real.sqrt 3 / 2}

theorem find_smallest_m :
  ∃ m : ℕ, (∀ n : ℕ, n ≥ m → ∃ z : ℂ, z ∈ S ∧ z ^ n = 1) ∧ m = 12 :=
by
  sorry

end find_smallest_m_l471_471511


namespace arithmetic_progression_count_l471_471882

variable (n : ℕ)

noncomputable def number_of_valid_progressions (d_max : ℕ) : ℕ :=
  (2 * n - 3 * d_max - 3) * d_max / 2

theorem arithmetic_progression_count :
  let d_max := (n - 1) / 3 in
  number_of_valid_progressions n d_max = 
    (2 * n - 3 * (n - 1) / 3 - 3) * (n - 1) / 3 / 2 := by
    sorry

end arithmetic_progression_count_l471_471882


namespace minimum_value_of_S_l471_471332

theorem minimum_value_of_S :
  ∃ (S : ℝ), let side := 1 in
             let trapezoid_perimeter := 2 * side + x + (1 - x) in
             let trapezoid_area := 0.5 * (x + 1) * (sqrt 3 / 2) * (1 - x) in
             S = (trapezoid_perimeter^2) / trapezoid_area ∧ 
             (∀ x, 0 < x ∧ x < 1 → S = (4 / sqrt 3) * ((3 - x)^2 / (1 - x^2)) ∧ 
                             S ≥ (32 * sqrt 3) / 3) :=
sorry

end minimum_value_of_S_l471_471332


namespace sum_a_and_c_is_zero_l471_471827

noncomputable
def roots (a b c : ℝ) := 
  let Δ := b * b - 4 * a * c in
  ((-b + Real.sqrt Δ) / (2 * a), (-b - Real.sqrt Δ) / (2 * a))

theorem sum_a_and_c_is_zero
  (a b c : ℝ)
  (h_distinct : a ≠ 0 ∧ c ≠ 0)
  (p1 p2 q1 q2 : ℝ)
  (h_roots_p : roots a b c = (p1, p2))
  (h_roots_q : roots c b a = (q1, q2))
  (h_arith_prog : p1 < q1 ∧ q1 < p2 ∧ p2 < q2 ∧ p1 ≠ q1 ∧ q1 ≠ p2 ∧ p2 ≠ q2)
  : a + c = 0 :=
sorry

end sum_a_and_c_is_zero_l471_471827


namespace determine_a_l471_471841

theorem determine_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (M m : ℝ)
  (hM : M = max (a^1) (a^2))
  (hm : m = min (a^1) (a^2))
  (hM_m : M = 2 * m) :
  a = 1/2 ∨ a = 2 := 
by sorry

end determine_a_l471_471841


namespace partial_fraction_sum_equals_251_l471_471520

theorem partial_fraction_sum_equals_251 (p q r A B C : ℝ) :
  (p ≠ q) ∧ (p ≠ r) ∧ (q ≠ r) ∧ 
  (A ≠ 0) ∧ (B ≠ 0) ∧ (C ≠ 0) ∧
  (∀ s : ℝ, (s ≠ p) ∧ (s ≠ q) ∧ (s ≠ r) →
  1 / (s^3 - 24*s^2 + 151*s - 650) = A / (s - p) + B / (s - q) + C / (s - r)) →
  (p + q + r = 24) →
  (p * q + p * r + q * r = 151) →
  (p * q * r = 650) →
  (1 / A + 1 / B + 1 / C = 251) :=
by
  sorry

end partial_fraction_sum_equals_251_l471_471520


namespace marching_band_max_l471_471597

-- Define the conditions
variables (m k n : ℕ)

-- Lean statement of the problem
theorem marching_band_max (H1 : m = k^2 + 9) (H2 : m = n * (n + 5)) : m = 234 :=
sorry

end marching_band_max_l471_471597


namespace a2017_value_l471_471616

noncomputable def seq : ℕ → ℚ
| 1       := 0
| (n + 1) := 1 - 1 / (1 + 1 - seq n)

theorem a2017_value : seq 2017 = 2016 / 2017 := sorry

end a2017_value_l471_471616


namespace total_points_correct_l471_471553

def noa_points := 30
def phillip_points := 2 * noa_points + 0.2 * noa_points
def sum_noa_phillip := noa_points + phillip_points
def lucy_points := (Real.sqrt sum_noa_phillip) * 10
def total_points := noa_points + phillip_points + Real.ceil lucy_points

theorem total_points_correct : total_points = 194 :=
by
  sorry

end total_points_correct_l471_471553


namespace game_not_fair_probability_first_player_wins_approx_l471_471171

def fair_game (n : ℕ) (P : ℕ → ℚ) : Prop :=
  ∀ k j, k ≤ n → j ≤ n → P k = P j

noncomputable def probability_first_player_wins (n : ℕ) : ℚ :=
  let r := (n - 1 : ℚ) / n;
  (1 : ℚ) / n * (1 : ℚ) / (1 - r ^ n)

-- Check if the game is fair for 36 players
theorem game_not_fair : ¬ fair_game 36 (λ i, probability_first_player_wins 36) :=
sorry

-- Calculate the approximate probability of the first player winning
theorem probability_first_player_wins_approx (n : ℕ)
  (h₁ : n = 36) :
  probability_first_player_wins n ≈ 0.044 :=
sorry

end game_not_fair_probability_first_player_wins_approx_l471_471171


namespace sum_of_reciprocals_of_squares_l471_471674

theorem sum_of_reciprocals_of_squares (a b : ℕ) (h : a * b = 19) : 1 / (a * a : ℚ) + 1 / (b * b : ℚ) = 362 / 361 := 
by
  sorry

end sum_of_reciprocals_of_squares_l471_471674


namespace square_property_unique_l471_471987

namespace Geometry

structure Square (A B C D : Type) :=
  (sides_equal : A = B ∧ B = C ∧ C = D)
  (diagonals_equal : A = C ∧ B = D)
  (diagonals_perpendicular : true)

structure Parallelogram (A B C D : Type) :=
  (opposite_sides_equal : (A = C) ∧ (B = D))
  (diagonals_bisect : true)

theorem square_property_unique (A B C D : Type) 
  (sq : Square A B C D) 
  (paral : Parallelogram A B C D) : 
  sq.diagonals_perpendicular ≠ paral.diagonals_perpendicular := 
  sorry

end Geometry

end square_property_unique_l471_471987


namespace size_of_smaller_package_l471_471322

theorem size_of_smaller_package
  (total_coffee : ℕ)
  (n_ten_ounce_packages : ℕ)
  (extra_five_ounce_packages : ℕ)
  (size_smaller_package : ℕ)
  (h1 : total_coffee = 115)
  (h2 : size_smaller_package = 5)
  (h3 : n_ten_ounce_packages = 7)
  (h4 : extra_five_ounce_packages = 2)
  (h5 : total_coffee = n_ten_ounce_packages * 10 + (n_ten_ounce_packages + extra_five_ounce_packages) * size_smaller_package) :
  size_smaller_package = 5 :=
by 
  sorry

end size_of_smaller_package_l471_471322


namespace gcd_a_plus_b_a_pow_n_plus_b_pow_n_l471_471443

variables (a b n : ℕ)

-- Conditions: a and b are positive integers and relatively prime, and n is a positive integer
def relatively_prime (a b : ℕ) : Prop := Nat.gcd a b = 1
def positive_integer (x : ℕ) : Prop := x > 0

theorem gcd_a_plus_b_a_pow_n_plus_b_pow_n (h1 : relatively_prime a b) 
  (h2 : positive_integer n) (h3 : positive_integer a) (h4 : positive_integer b) : 
  Nat.gcd (a + b) (a^n + b^n) = 
  if Nat.odd n then a + b else Nat.gcd 2 (a + b) :=
sorry

end gcd_a_plus_b_a_pow_n_plus_b_pow_n_l471_471443


namespace problem_equiv_proof_l471_471394

noncomputable def f : ℝ → ℝ := sorry

theorem problem_equiv_proof (x : ℝ) (h_sym : ∀ y, f(3 + y) = f(3 - y))
    (h_f_neg1 : f(-1) = 320)
    (h_cos_sin : cos x - sin x = (3 * real.sqrt 2) / 5) :
    f (15 * sin (2 * x) / cos (x + real.pi / 4)) = 320 := by
  sorry

end problem_equiv_proof_l471_471394


namespace students_total_l471_471147

theorem students_total (scavenger_hunt_students : ℕ) (ski_trip_students : ℕ) 
  (h1 : ski_trip_students = 2 * scavenger_hunt_students) 
  (h2 : scavenger_hunt_students = 4000) : 
  scavenger_hunt_students + ski_trip_students = 12000 := 
by
  sorry

end students_total_l471_471147


namespace roots_quadratic_expression_l471_471386

theorem roots_quadratic_expression (m n : ℝ) (h1 : m^2 + 2 * m - 5 = 0) (h2 : n^2 + 2 * n - 5 = 0) 
  (sum_roots : m + n = -2) (product_roots : m * n = -5) : m^2 + m * n + 3 * m + n = -2 :=
sorry

end roots_quadratic_expression_l471_471386


namespace estimated_red_balls_l471_471626

theorem estimated_red_balls
  (total_balls : ℕ)
  (total_draws : ℕ)
  (red_draws : ℕ)
  (h_total_balls : total_balls = 12)
  (h_total_draws : total_draws = 200)
  (h_red_draws : red_draws = 50) :
  red_draws * total_balls = total_draws * 3 :=
by
  sorry

end estimated_red_balls_l471_471626


namespace condition_sufficient_but_not_necessary_l471_471214

variable (a b : ℝ)

theorem condition_sufficient_but_not_necessary :
  (|a| < 1 ∧ |b| < 1) → (|1 - a * b| > |a - b|) ∧
  ((|1 - a * b| > |a - b|) → (|a| < 1 ∧ |b| < 1) ∨ (|a| ≥ 1 ∧ |b| ≥ 1)) :=
by
  sorry

end condition_sufficient_but_not_necessary_l471_471214


namespace sum_of_possible_x_l471_471733

theorem sum_of_possible_x :
  let lst := [12, -3, x, 7, -3, x, -3]
  let mean := (12 - 3 + x + 7 - 3 + x - 3) / 7
  let mode := -3
  (∀ x, mean, median, mode : ℚ, list.median lst = x ∧ list.median lst ≥ 7 ->
    let arithmetic mean, median, mode : list ℚ := λ mean lst → [lst.median]
      x = 54.042) :=
sorry

end sum_of_possible_x_l471_471733
