import Mathlib

namespace NUMINAMATH_GPT_number_of_true_propositions_is_zero_l2150_215026

theorem number_of_true_propositions_is_zero :
  (∀ x : ℝ, x^2 - 3 * x + 2 ≠ 0) →
  (¬ ∃ x : ℚ, x^2 = 2) →
  (¬ ∃ x : ℝ, x^2 + 1 = 0) →
  (∀ x : ℝ, 4 * x^2 ≤ 2 * x - 1 + 3 * x^2) →
  true :=  -- representing that the number of true propositions is 0
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_number_of_true_propositions_is_zero_l2150_215026


namespace NUMINAMATH_GPT_charles_draws_yesterday_after_work_l2150_215044

theorem charles_draws_yesterday_after_work :
  ∀ (initial_papers today_drawn yesterday_drawn_before_work current_papers_left yesterday_drawn_after_work : ℕ),
    initial_papers = 20 →
    today_drawn = 6 →
    yesterday_drawn_before_work = 6 →
    current_papers_left = 2 →
    (initial_papers - (today_drawn + yesterday_drawn_before_work) - yesterday_drawn_after_work = current_papers_left) →
    yesterday_drawn_after_work = 6 :=
by
  intros initial_papers today_drawn yesterday_drawn_before_work current_papers_left yesterday_drawn_after_work
  intro h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_charles_draws_yesterday_after_work_l2150_215044


namespace NUMINAMATH_GPT_two_R_theta_bounds_l2150_215011

variables {R : ℝ} (θ : ℝ)
variables (h_pos : 0 < R) (h_triangle : (R + 1 + (R + 1/2)) > 2 *R)

-- Define that θ is the angle between sides R and R + 1/2
-- Here we assume θ is defined via the cosine rule for simplicity

noncomputable def angle_between_sides (R : ℝ) := 
  Real.arccos ((R^2 + (R + 1/2)^2 - 1^2) / (2 * R * (R + 1/2)))

-- State the theorem
theorem two_R_theta_bounds (h : θ = angle_between_sides R) : 
  1 < 2 * R * θ ∧ 2 * R * θ < π :=
by
  sorry

end NUMINAMATH_GPT_two_R_theta_bounds_l2150_215011


namespace NUMINAMATH_GPT_simple_and_compound_interest_difference_l2150_215040

theorem simple_and_compound_interest_difference (r : ℝ) :
  let P := 3600
  let t := 2
  let SI := P * r * t / 100
  let CI := P * (1 + r / 100)^t - P
  CI - SI = 225 → r = 25 := by
  intros
  sorry

end NUMINAMATH_GPT_simple_and_compound_interest_difference_l2150_215040


namespace NUMINAMATH_GPT_total_holes_dug_l2150_215094

theorem total_holes_dug :
  (Pearl_digging_rate * 21 + Miguel_digging_rate * 21) = 26 :=
by
  -- Definitions based on conditions
  let Pearl_digging_rate := 4 / 7
  let Miguel_digging_rate := 2 / 3
  -- Sorry placeholder for the proof
  sorry

end NUMINAMATH_GPT_total_holes_dug_l2150_215094


namespace NUMINAMATH_GPT_determine_f_l2150_215012

theorem determine_f (d e f : ℝ) 
  (h_eq : ∀ y : ℝ, (-3) = d * y^2 + e * y + f)
  (h_vertex : ∀ k : ℝ, -1 = d * (3 - k)^2 + e * (3 - k) + f) :
  f = -5 / 2 :=
sorry

end NUMINAMATH_GPT_determine_f_l2150_215012


namespace NUMINAMATH_GPT_arcsin_sqrt_one_half_l2150_215050

theorem arcsin_sqrt_one_half : Real.arcsin (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  -- TODO: provide proof
  sorry

end NUMINAMATH_GPT_arcsin_sqrt_one_half_l2150_215050


namespace NUMINAMATH_GPT_volume_pyramid_PABCD_is_384_l2150_215046

noncomputable def volume_of_pyramid : ℝ :=
  let AB := 12
  let BC := 6
  let PA := Real.sqrt (20^2 - 12^2)
  let base_area := AB * BC
  (1 / 3) * base_area * PA

theorem volume_pyramid_PABCD_is_384 :
  volume_of_pyramid = 384 := 
by
  sorry

end NUMINAMATH_GPT_volume_pyramid_PABCD_is_384_l2150_215046


namespace NUMINAMATH_GPT_sum_of_roots_proof_l2150_215001

noncomputable def sum_of_roots (x1 x2 x3 : ℝ) : ℝ :=
  let eq1 := (11 - x1)^3 + (13 - x1)^3 = (24 - 2 * x1)^3
  let eq2 := (11 - x2)^3 + (13 - x2)^3 = (24 - 2 * x2)^3
  let eq3 := (11 - x3)^3 + (13 - x3)^3 = (24 - 2 * x3)^3
  x1 + x2 + x3

theorem sum_of_roots_proof : sum_of_roots 11 12 13 = 36 :=
  sorry

end NUMINAMATH_GPT_sum_of_roots_proof_l2150_215001


namespace NUMINAMATH_GPT_product_of_marbles_l2150_215009

theorem product_of_marbles (R B : ℕ) (h1 : R - B = 12) (h2 : R + B = 52) : R * B = 640 := by
  sorry

end NUMINAMATH_GPT_product_of_marbles_l2150_215009


namespace NUMINAMATH_GPT_reduction_percentage_toy_l2150_215014

-- Definition of key parameters
def paintings_bought : ℕ := 10
def cost_per_painting : ℕ := 40
def toys_bought : ℕ := 8
def cost_per_toy : ℕ := 20
def total_cost : ℕ := (paintings_bought * cost_per_painting) + (toys_bought * cost_per_toy) -- $560
def painting_selling_price_per_unit : ℕ := cost_per_painting - (cost_per_painting * 10 / 100) -- $36
def total_loss : ℕ := 64

-- Define percentage reduction in the selling price of a wooden toy
variable {x : ℕ} -- Define x as a percentage value to be solved

-- Theorems to prove
theorem reduction_percentage_toy (x) : 
  (paintings_bought * painting_selling_price_per_unit) 
  + (toys_bought * (cost_per_toy - (cost_per_toy * x / 100))) 
  = (total_cost - total_loss) 
  → x = 15 := 
by
  sorry

end NUMINAMATH_GPT_reduction_percentage_toy_l2150_215014


namespace NUMINAMATH_GPT_max_two_integers_abs_leq_50_l2150_215063

theorem max_two_integers_abs_leq_50
  (a b c : ℤ) (h_a : a > 100) :
  ∀ {x1 x2 x3 : ℤ}, (abs (a * x1^2 + b * x1 + c) ≤ 50) →
                    (abs (a * x2^2 + b * x2 + c) ≤ 50) →
                    (abs (a * x3^2 + b * x3 + c) ≤ 50) →
                    false :=
sorry

end NUMINAMATH_GPT_max_two_integers_abs_leq_50_l2150_215063


namespace NUMINAMATH_GPT_irrational_sqrt_2023_l2150_215099

theorem irrational_sqrt_2023 (A B C D : ℝ) :
  A = -2023 → B = Real.sqrt 2023 → C = 0 → D = 1 / 2023 →
  (¬ ∃ (p q : ℤ), q ≠ 0 ∧ B = p / q) ∧
  (∃ (p q : ℤ), q ≠ 0 ∧ A = p / q) ∧
  (∃ (p q : ℤ), q ≠ 0 ∧ C = p / q) ∧
  (∃ (p q : ℤ), q ≠ 0 ∧ D = p / q) := 
by
  intro hA hB hC hD
  sorry

end NUMINAMATH_GPT_irrational_sqrt_2023_l2150_215099


namespace NUMINAMATH_GPT_intersection_complement_l2150_215028

def set_M : Set ℝ := {x : ℝ | x^2 - x = 0}

def set_N : Set ℝ := {x : ℝ | ∃ n : ℤ, x = 2 * n + 1}

theorem intersection_complement (h : UniversalSet = Set.univ) :
  set_M ∩ (UniversalSet \ set_N) = {0} := 
sorry

end NUMINAMATH_GPT_intersection_complement_l2150_215028


namespace NUMINAMATH_GPT_isosceles_triangle_base_l2150_215035

theorem isosceles_triangle_base (b : ℝ) (h1 : 7 + 7 + b = 20) : b = 6 :=
by {
    sorry
}

end NUMINAMATH_GPT_isosceles_triangle_base_l2150_215035


namespace NUMINAMATH_GPT_solution_set_of_quadratic_inequality_l2150_215032

theorem solution_set_of_quadratic_inequality :
  {x : ℝ | x^2 - x - 6 < 0} = {x : ℝ | -2 < x ∧ x < 3} :=
sorry

end NUMINAMATH_GPT_solution_set_of_quadratic_inequality_l2150_215032


namespace NUMINAMATH_GPT_find_pairs_l2150_215033

theorem find_pairs (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  (a^3 * b - 1) % (a + 1) = 0 ∧ (b^3 * a + 1) % (b - 1) = 0 ↔ (a = 2 ∧ b = 2) ∨ (a = 1 ∧ b = 3) ∨ (a = 3 ∧ b = 3) :=
by
  sorry

end NUMINAMATH_GPT_find_pairs_l2150_215033


namespace NUMINAMATH_GPT_smallest_m_inequality_l2150_215023

theorem smallest_m_inequality (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_sum : a + b + c = 1) : 27 * (a^3 + b^3 + c^3) ≥ 6 * (a^2 + b^2 + c^2) + 1 :=
sorry

end NUMINAMATH_GPT_smallest_m_inequality_l2150_215023


namespace NUMINAMATH_GPT_machine_work_time_today_l2150_215066

theorem machine_work_time_today :
  let shirts_today := 40
  let pants_today := 50
  let shirt_rate := 5
  let pant_rate := 3
  let time_for_shirts := shirts_today / shirt_rate
  let time_for_pants := pants_today / pant_rate
  time_for_shirts + time_for_pants = 24.67 :=
by
  sorry

end NUMINAMATH_GPT_machine_work_time_today_l2150_215066


namespace NUMINAMATH_GPT_impossible_cube_configuration_l2150_215055

theorem impossible_cube_configuration :
  ∀ (cube: ℕ → ℕ) (n : ℕ), 
    (∀ n, 1 ≤ n ∧ n ≤ 27 → ∃ k, 1 ≤ k ∧ k ≤ 27 ∧ cube k = n) →
    (∀ n, 1 ≤ n ∧ n ≤ 27 → (cube 27 = 27 ∧ ∀ m, 1 ≤ m ∧ m ≤ 26 → cube m = 27 - m)) → 
    false :=
by
  intros cube n hcube htarget
  -- any detailed proof steps would go here, skipping with sorry
  sorry

end NUMINAMATH_GPT_impossible_cube_configuration_l2150_215055


namespace NUMINAMATH_GPT_percentage_problem_l2150_215018

theorem percentage_problem (x : ℝ) (h : 0.30 * 0.15 * x = 18) : 0.15 * 0.30 * x = 18 :=
by
  sorry

end NUMINAMATH_GPT_percentage_problem_l2150_215018


namespace NUMINAMATH_GPT_find_m_l2150_215002

-- Mathematical conditions definitions
def line1 (x y : ℝ) (m : ℝ) : Prop := 3 * x + m * y - 1 = 0
def line2 (x y : ℝ) (m : ℝ) : Prop := (m + 2) * x - (m - 2) * y + 2 = 0

-- Given the lines are parallel
def lines_parallel (l1 l2 : ℝ → ℝ → ℝ → Prop) (m : ℝ) : Prop :=
  ∀ x y : ℝ, l1 x y m → l2 x y m → (3 / (m + 2)) = (m / (-(m - 2)))

-- The proof problem statement
theorem find_m (m : ℝ) : 
  lines_parallel (line1) (line2) m → (m = -6 ∨ m = 1) :=
by
  sorry

end NUMINAMATH_GPT_find_m_l2150_215002


namespace NUMINAMATH_GPT_decrease_in_sales_percentage_l2150_215031

theorem decrease_in_sales_percentage (P Q : Real) :
  let P' := 1.40 * P
  let R := P * Q
  let R' := 1.12 * R
  ∃ (D : Real), Q' = Q * (1 - D / 100) ∧ R' = P' * Q' → D = 20 :=
by
  sorry

end NUMINAMATH_GPT_decrease_in_sales_percentage_l2150_215031


namespace NUMINAMATH_GPT_amusement_park_admission_fees_l2150_215019

theorem amusement_park_admission_fees
  (num_children : ℕ) (num_adults : ℕ)
  (fee_child : ℝ) (fee_adult : ℝ)
  (total_people : ℕ) (expected_total_fees : ℝ) :
  num_children = 180 →
  fee_child = 1.5 →
  fee_adult = 4.0 →
  total_people = 315 →
  expected_total_fees = 810 →
  num_children + num_adults = total_people →
  (num_children : ℝ) * fee_child + (num_adults : ℝ) * fee_adult = expected_total_fees := by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_amusement_park_admission_fees_l2150_215019


namespace NUMINAMATH_GPT_farm_cows_l2150_215047

theorem farm_cows (x y : ℕ) (h : 4 * x + 2 * y = 20 + 3 * (x + y)) : x = 20 + y :=
sorry

end NUMINAMATH_GPT_farm_cows_l2150_215047


namespace NUMINAMATH_GPT_rectangle_section_properties_l2150_215007

structure Tetrahedron where
  edge_length : ℝ

structure RectangleSection where
  perimeter : ℝ
  area : ℝ

def regular_tetrahedron : Tetrahedron :=
  { edge_length := 1 }

theorem rectangle_section_properties :
  ∀ (rect : RectangleSection), 
  (∃ tetra : Tetrahedron, tetra = regular_tetrahedron) →
  (rect.perimeter = 2) ∧ (0 ≤ rect.area) ∧ (rect.area ≤ 1/4) :=
by
  -- Provide the hypothesis of the existence of such a tetrahedron and rectangular section
  sorry

end NUMINAMATH_GPT_rectangle_section_properties_l2150_215007


namespace NUMINAMATH_GPT_marbles_problem_l2150_215096

def marbles_total : ℕ := 30
def prob_black_black : ℚ := 14 / 25
def prob_white_white : ℚ := 16 / 225

theorem marbles_problem (total_marbles : ℕ) (prob_bb prob_ww : ℚ) 
  (h_total : total_marbles = 30)
  (h_prob_bb : prob_bb = 14 / 25)
  (h_prob_ww : prob_ww = 16 / 225) :
  let m := 16
  let n := 225
  m.gcd n = 1 ∧ m + n = 241 :=
by {
  sorry
}

end NUMINAMATH_GPT_marbles_problem_l2150_215096


namespace NUMINAMATH_GPT_total_pairs_sold_l2150_215042

theorem total_pairs_sold (H S : ℕ) 
    (soft_lens_cost hard_lens_cost : ℕ)
    (total_sales : ℕ)
    (h1 : soft_lens_cost = 150)
    (h2 : hard_lens_cost = 85)
    (h3 : S = H + 5)
    (h4 : soft_lens_cost * S + hard_lens_cost * H = total_sales)
    (h5 : total_sales = 1455) :
    H + S = 11 := 
  sorry

end NUMINAMATH_GPT_total_pairs_sold_l2150_215042


namespace NUMINAMATH_GPT_quadratic_solution_sum_l2150_215075

theorem quadratic_solution_sum
  (x : ℚ)
  (m n p : ℕ)
  (h_eq : (5 * x - 11) * x = -6)
  (h_form : ∃ m n p, x = (m + Real.sqrt n) / p ∧ x = (m - Real.sqrt n) / p)
  (h_gcd : Nat.gcd (Nat.gcd m n) p = 1) :
  m + n + p = 22 := 
sorry

end NUMINAMATH_GPT_quadratic_solution_sum_l2150_215075


namespace NUMINAMATH_GPT_min_value_of_A2_minus_B2_nonneg_l2150_215015

noncomputable def A (x y z : ℝ) : ℝ :=
  Real.sqrt (x + 4) + Real.sqrt (y + 7) + Real.sqrt (z + 13)

noncomputable def B (x y z : ℝ) : ℝ :=
  Real.sqrt (x + 3) + Real.sqrt (y + 3) + Real.sqrt (z + 3)

theorem min_value_of_A2_minus_B2_nonneg (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) : 
  (A x y z) ^ 2 - (B x y z) ^ 2 ≥ 36 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_A2_minus_B2_nonneg_l2150_215015


namespace NUMINAMATH_GPT_binary_to_decimal_conversion_l2150_215092

theorem binary_to_decimal_conversion : (1 * 2^2 + 1 * 2^1 + 0 * 2^0 = 6) := by
  sorry

end NUMINAMATH_GPT_binary_to_decimal_conversion_l2150_215092


namespace NUMINAMATH_GPT_pencil_and_pen_cost_l2150_215016

theorem pencil_and_pen_cost
  (p q : ℝ)
  (h1 : 3 * p + 2 * q = 3.75)
  (h2 : 2 * p + 3 * q = 4.05) :
  p + q = 1.56 :=
by
  sorry

end NUMINAMATH_GPT_pencil_and_pen_cost_l2150_215016


namespace NUMINAMATH_GPT_monotonic_decreasing_intervals_l2150_215027

noncomputable def f (x : ℝ) : ℝ := x / Real.log x

theorem monotonic_decreasing_intervals : 
  (∀ x : ℝ, (0 < x ∧ x < 1) → ∃ ε > 0, ∀ y : ℝ, x ≤ y ∧ y ≤ x + ε → f y < f x) ∧
  (∀ x : ℝ, (1 < x ∧ x < Real.exp 1) → ∃ ε > 0, ∀ y : ℝ, x ≤ y ∧ y ≤ x + ε → f y < f x) :=
by
  sorry

end NUMINAMATH_GPT_monotonic_decreasing_intervals_l2150_215027


namespace NUMINAMATH_GPT_soccer_team_points_l2150_215052

theorem soccer_team_points 
  (total_games wins losses draws : ℕ)
  (points_per_win points_per_draw points_per_loss : ℕ)
  (h_total_games : total_games = 20)
  (h_wins : wins = 14)
  (h_losses : losses = 2)
  (h_draws : draws = total_games - (wins + losses))
  (h_points_per_win : points_per_win = 3)
  (h_points_per_draw : points_per_draw = 1)
  (h_points_per_loss : points_per_loss = 0) :
  (wins * points_per_win) + (draws * points_per_draw) + (losses * points_per_loss) = 46 :=
by
  -- the actual proof steps will be inserted here
  sorry

end NUMINAMATH_GPT_soccer_team_points_l2150_215052


namespace NUMINAMATH_GPT_remainder_when_200_divided_by_k_l2150_215039

theorem remainder_when_200_divided_by_k 
  (k : ℕ) (k_pos : 0 < k)
  (h : 120 % k^2 = 12) :
  200 % k = 2 :=
sorry

end NUMINAMATH_GPT_remainder_when_200_divided_by_k_l2150_215039


namespace NUMINAMATH_GPT_sin_double_angle_l2150_215083

variable {φ : ℝ}

theorem sin_double_angle (h : (7 / 13 : ℝ) + Real.sin φ = Real.cos φ) :
    Real.sin (2 * φ) = 120 / 169 := 
sorry

end NUMINAMATH_GPT_sin_double_angle_l2150_215083


namespace NUMINAMATH_GPT_translation_invariant_line_l2150_215056

theorem translation_invariant_line (k : ℝ) :
  (∀ x : ℝ, k * (x - 2) + 5 = k * x + 2) → k = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_translation_invariant_line_l2150_215056


namespace NUMINAMATH_GPT_circle_diameter_in_feet_l2150_215086

/-- Given: The area of a circle is 25 * pi square inches.
    Prove: The diameter of the circle in feet is 5/6 feet. -/
theorem circle_diameter_in_feet (A : ℝ) (hA : A = 25 * Real.pi) :
  ∃ d : ℝ, d = (5 / 6) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_circle_diameter_in_feet_l2150_215086


namespace NUMINAMATH_GPT_maximal_x2009_l2150_215025

theorem maximal_x2009 (x : ℕ → ℝ) 
    (h_seq : ∀ n, x n - 2 * x (n + 1) + x (n + 2) ≤ 0)
    (h_x0 : x 0 = 1)
    (h_x20 : x 20 = 9)
    (h_x200 : x 200 = 6) :
    x 2009 ≤ 6 :=
sorry

end NUMINAMATH_GPT_maximal_x2009_l2150_215025


namespace NUMINAMATH_GPT_simplify_expression_l2150_215071

open Real

theorem simplify_expression :
    (3 * (sqrt 5 + sqrt 7) / (4 * sqrt (3 + sqrt 5))) = sqrt (414 - 98 * sqrt 35) / 8 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l2150_215071


namespace NUMINAMATH_GPT_trajectory_equation_necessary_not_sufficient_l2150_215051

theorem trajectory_equation_necessary_not_sufficient :
  ∀ (x y : ℝ), (|x| = |y|) → (y = |x|) ↔ (necessary_not_sufficient) :=
by
  sorry

end NUMINAMATH_GPT_trajectory_equation_necessary_not_sufficient_l2150_215051


namespace NUMINAMATH_GPT_min_m_n_sum_divisible_by_27_l2150_215029

theorem min_m_n_sum_divisible_by_27 (m n : ℕ) (h : 180 * m * (n - 2) % 27 = 0) : m + n = 6 :=
sorry

end NUMINAMATH_GPT_min_m_n_sum_divisible_by_27_l2150_215029


namespace NUMINAMATH_GPT_average_salary_of_all_workers_is_correct_l2150_215036

noncomputable def average_salary_all_workers (n_total n_tech : ℕ) (avg_salary_tech avg_salary_others : ℝ) : ℝ :=
  let n_others := n_total - n_tech
  let total_salary_tech := n_tech * avg_salary_tech
  let total_salary_others := n_others * avg_salary_others
  let total_salary := total_salary_tech + total_salary_others
  total_salary / n_total

theorem average_salary_of_all_workers_is_correct :
  average_salary_all_workers 21 7 12000 6000 = 8000 :=
by
  unfold average_salary_all_workers
  sorry

end NUMINAMATH_GPT_average_salary_of_all_workers_is_correct_l2150_215036


namespace NUMINAMATH_GPT_sum_b_div_5_pow_eq_l2150_215073

namespace SequenceSumProblem

-- Define the sequence b_n
def b : ℕ → ℝ
| 0       => 2
| 1       => 3
| (n + 2) => b (n + 1) + b n

-- The infinite series sum we need to prove
noncomputable def sum_b_div_5_pow (Y : ℝ) : Prop :=
  Y = ∑' n : ℕ, (b n) / (5 ^ (n + 1))

-- The statement of the problem
theorem sum_b_div_5_pow_eq : sum_b_div_5_pow (2 / 25) :=
sorry

end SequenceSumProblem

end NUMINAMATH_GPT_sum_b_div_5_pow_eq_l2150_215073


namespace NUMINAMATH_GPT_value_of_a_l2150_215022

noncomputable def number : ℕ := 21 * 25 * 45 * 49

theorem value_of_a (a : ℕ) (h : a^3 = number) : a = 105 :=
sorry

end NUMINAMATH_GPT_value_of_a_l2150_215022


namespace NUMINAMATH_GPT_quad_factor_value_l2150_215030

theorem quad_factor_value (c d : ℕ) (h1 : c + d = 14) (h2 : c * d = 40) (h3 : c > d) : 4 * d - c = 6 :=
sorry

end NUMINAMATH_GPT_quad_factor_value_l2150_215030


namespace NUMINAMATH_GPT_train_people_count_l2150_215098

theorem train_people_count :
  let initial := 332
  let first_station_on := 119
  let first_station_off := 113
  let second_station_off := 95
  let second_station_on := 86
  initial + first_station_on - first_station_off - second_station_off + second_station_on = 329 := 
by
  sorry

end NUMINAMATH_GPT_train_people_count_l2150_215098


namespace NUMINAMATH_GPT_monotonicity_of_f_range_of_a_l2150_215017

noncomputable def f (a x : ℝ) : ℝ := -2 * x^3 + 6 * a * x^2 - 1
noncomputable def g (a x : ℝ) : ℝ := Real.exp x - 2 * a * x - 1

theorem monotonicity_of_f (a : ℝ) :
  (a < 0 → (∀ x, f a x ≥ f a (2 * a) → x ≤ 0 ∨ x ≤ 2 * a)) ∧
  (a = 0 → ∀ x y, x ≤ y → f a x ≥ f a y) ∧
  (a > 0 → (∀ x, f a x ≤ f a 0 → x ≤ 0) ∧
           (∀ x, 0 < x ∧ x < 2 * a → f a x ≥ f a 2 * a) ∧
           (∀ x, 2 * a < x → f a x ≤ f a (2 * a))) :=
sorry

theorem range_of_a :
  ∀ a : ℝ, a ≥ 1 / 2 →
  ∃ x1 : ℝ, x1 > 0 ∧ ∃ x2 : ℝ, f a x1 ≥ g a x2 :=
sorry

end NUMINAMATH_GPT_monotonicity_of_f_range_of_a_l2150_215017


namespace NUMINAMATH_GPT_researcher_can_cross_desert_l2150_215077

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

end NUMINAMATH_GPT_researcher_can_cross_desert_l2150_215077


namespace NUMINAMATH_GPT_prism_cutout_l2150_215062

noncomputable def original_volume : ℕ := 15 * 5 * 4 -- Volume of the original prism
noncomputable def cutout_width : ℕ := 5

variables {x y : ℕ}

theorem prism_cutout:
  -- Given conditions
  (15 > 0) ∧ (5 > 0) ∧ (4 > 0) ∧ (x > 0) ∧ (y > 0) ∧ 
  -- The volume condition
  (original_volume - y * cutout_width * x = 120) →
  -- Prove that x + y = 15
  (x + y = 15) :=
sorry

end NUMINAMATH_GPT_prism_cutout_l2150_215062


namespace NUMINAMATH_GPT_sunset_time_l2150_215020

def length_of_daylight_in_minutes := 11 * 60 + 12
def sunrise_time_in_minutes := 6 * 60 + 45
def sunset_time_in_minutes := sunrise_time_in_minutes + length_of_daylight_in_minutes
def sunset_time_hour := sunset_time_in_minutes / 60
def sunset_time_minute := sunset_time_in_minutes % 60
def sunset_time_12hr_format := if sunset_time_hour >= 12 
    then (sunset_time_hour - 12, sunset_time_minute)
    else (sunset_time_hour, sunset_time_minute)

theorem sunset_time : sunset_time_12hr_format = (5, 57) :=
by
  sorry

end NUMINAMATH_GPT_sunset_time_l2150_215020


namespace NUMINAMATH_GPT_polynomial_roots_l2150_215064

theorem polynomial_roots (p q BD DC : ℝ) (h_sum : BD + DC = p) (h_prod : BD * DC = q^2) :
    Polynomial.roots (Polynomial.C 1 * Polynomial.X^2 - Polynomial.C p * Polynomial.X + Polynomial.C (q^2)) = {BD, DC} :=
sorry

end NUMINAMATH_GPT_polynomial_roots_l2150_215064


namespace NUMINAMATH_GPT_field_ratio_l2150_215004

theorem field_ratio (w : ℝ) (h : ℝ) (pond_len : ℝ) (field_len : ℝ) 
  (h1 : pond_len = 8) 
  (h2 : field_len = 112) 
  (h3 : w > 0) 
  (h4 : field_len = w * h) 
  (h5 : pond_len * pond_len = (1 / 98) * (w * h * h)) : 
  field_len / h = 2 := 
by 
  sorry

end NUMINAMATH_GPT_field_ratio_l2150_215004


namespace NUMINAMATH_GPT_inequality_ab_bc_ca_max_l2150_215074

theorem inequality_ab_bc_ca_max (a b c : ℝ) :
  a * b + b * c + c * a + max (|a - b|) (max (|b - c|) (|c - a|))
  ≤ 1 + (1 / 3) * (a + b + c)^2 := sorry

end NUMINAMATH_GPT_inequality_ab_bc_ca_max_l2150_215074


namespace NUMINAMATH_GPT_two_digit_numbers_condition_l2150_215038

theorem two_digit_numbers_condition :
  ∃ (x y : ℕ), x > y ∧ x < 100 ∧ y < 100 ∧ x - y = 56 ∧ (x^2 % 100) = (y^2 % 100) ∧
  ((x = 78 ∧ y = 22) ∨ (x = 22 ∧ y = 78)) :=
by sorry

end NUMINAMATH_GPT_two_digit_numbers_condition_l2150_215038


namespace NUMINAMATH_GPT_prob1_prob2_prob3_l2150_215076

-- Problem 1
theorem prob1 (a b c : ℝ) : ((-8 * a^4 * b^5 * c / (4 * a * b^5)) * (3 * a^3 * b^2)) = -6 * a^6 * b^2 :=
by
  sorry

-- Problem 2
theorem prob2 (a : ℝ) : (2 * a + 1)^2 - (2 * a + 1) * (2 * a - 1) = 4 * a + 2 :=
by
  sorry

-- Problem 3
theorem prob3 (x y : ℝ) : (x - y - 2) * (x - y + 2) - (x + 2 * y) * (x - 3 * y) = 7 * y^2 - x * y - 4 :=
by
  sorry

end NUMINAMATH_GPT_prob1_prob2_prob3_l2150_215076


namespace NUMINAMATH_GPT_f_neg_val_is_minus_10_l2150_215068
-- Import the necessary Lean library

-- Define the function f with the given conditions
def f (a b x : ℝ) : ℝ := a * x^5 + b * x^3 + 3

-- Define the specific values
def x_val : ℝ := 2023
def x_neg_val : ℝ := -2023
def f_pos_val : ℝ := 16

-- Theorem to prove
theorem f_neg_val_is_minus_10 (a b : ℝ)
  (h : f a b x_val = f_pos_val) : 
  f a b x_neg_val = -10 :=
by
  -- Sorry placeholder for proof
  sorry

end NUMINAMATH_GPT_f_neg_val_is_minus_10_l2150_215068


namespace NUMINAMATH_GPT_jack_needs_more_money_l2150_215091

/--
Jack is a soccer player. He needs to buy two pairs of socks, a pair of soccer shoes, a soccer ball, and a sports bag.
Each pair of socks costs $12.75, the shoes cost $145, the soccer ball costs $38, and the sports bag costs $47.
Jack has a 5% discount coupon for the shoes and a 10% discount coupon for the sports bag.
He currently has $25. How much more money does Jack need to buy all the items?
-/
theorem jack_needs_more_money :
  let socks_cost : ℝ := 12.75
  let shoes_cost : ℝ := 145
  let ball_cost : ℝ := 38
  let bag_cost : ℝ := 47
  let shoes_discount : ℝ := 0.05
  let bag_discount : ℝ := 0.10
  let money_jack_has : ℝ := 25
  let total_cost := 2 * socks_cost + (shoes_cost - shoes_cost * shoes_discount) + ball_cost + (bag_cost - bag_cost * bag_discount)
  total_cost - money_jack_has = 218.55 :=
by
  sorry

end NUMINAMATH_GPT_jack_needs_more_money_l2150_215091


namespace NUMINAMATH_GPT_triangle_similarity_proof_l2150_215065

-- Define a structure for points in a geometric space
structure Point : Type where
  x : ℝ
  y : ℝ
  deriving Inhabited

-- Define the conditions provided in the problem
variables (A B C D E H : Point)
variables (HD HE : ℝ)

-- Condition statements
def HD_dist := HD = 6
def HE_dist := HE = 3

-- Main theorem statement
theorem triangle_similarity_proof (BD DC AE EC BH AH : ℝ) 
  (h1 : HD = 6) (h2 : HE = 3) 
  (h3 : 2 * BH = AH) : 
  (BD * DC - AE * EC = 9 * BH + 27) :=
sorry

end NUMINAMATH_GPT_triangle_similarity_proof_l2150_215065


namespace NUMINAMATH_GPT_train_cross_pole_time_l2150_215000

noncomputable def L_train : ℝ := 300 -- Length of the train in meters
noncomputable def L_platform : ℝ := 870 -- Length of the platform in meters
noncomputable def t_platform : ℝ := 39 -- Time to cross the platform in seconds

theorem train_cross_pole_time
  (L_train : ℝ)
  (L_platform : ℝ)
  (t_platform : ℝ)
  (D : ℝ := L_train + L_platform)
  (v : ℝ := D / t_platform)
  (t_pole : ℝ := L_train / v) :
  t_pole = 10 :=
by sorry

end NUMINAMATH_GPT_train_cross_pole_time_l2150_215000


namespace NUMINAMATH_GPT_find_d_l2150_215067

theorem find_d 
    (a b c d : ℝ) 
    (h_a_pos : 0 < a)
    (h_b_pos : 0 < b)
    (h_c_pos : 0 < c)
    (h_d_pos : 0 < d)
    (max_val : d + a = 7)
    (min_val : d - a = 1) :
    d = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_d_l2150_215067


namespace NUMINAMATH_GPT_age_of_15th_student_l2150_215084

theorem age_of_15th_student (avg15: ℕ) (avg5: ℕ) (avg9: ℕ) (x: ℕ)
  (h1: avg15 = 15) (h2: avg5 = 14) (h3: avg9 = 16)
  (h4: 15 * avg15 = x + 5 * avg5 + 9 * avg9) : x = 11 :=
by
  -- Proof will be added here
  sorry

end NUMINAMATH_GPT_age_of_15th_student_l2150_215084


namespace NUMINAMATH_GPT_no_common_root_l2150_215070

theorem no_common_root (a b c d : ℝ) (h0 : 0 < a) (h1 : a < b) (h2 : b < c) (h3 : c < d) :
  ¬ ∃ x : ℝ, x^2 + b * x + c = 0 ∧ x^2 + a * x + d = 0 :=
by
  sorry

end NUMINAMATH_GPT_no_common_root_l2150_215070


namespace NUMINAMATH_GPT_attendance_second_day_l2150_215082

theorem attendance_second_day (total_attendance first_day_attendance second_day_attendance third_day_attendance : ℕ) 
  (h_total : total_attendance = 2700)
  (h_second_day : second_day_attendance = first_day_attendance / 2)
  (h_third_day : third_day_attendance = 3 * first_day_attendance) :
  second_day_attendance = 300 :=
by
  sorry

end NUMINAMATH_GPT_attendance_second_day_l2150_215082


namespace NUMINAMATH_GPT_arithmetic_sequence_properties_l2150_215057

theorem arithmetic_sequence_properties
  (a_n : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : ∀ n, S n = n * ((a_n 0 + a_n (n-1)) / 2))
  (h2 : S 6 < S 7)
  (h3 : S 7 > S 8) :
  (a_n 8 - a_n 7 < 0) ∧ (S 9 < S 6) ∧ (∀ m, S m ≤ S 7) :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_properties_l2150_215057


namespace NUMINAMATH_GPT_largest_c_in_range_l2150_215037

theorem largest_c_in_range (c : ℝ) (h : ∃ x : ℝ,  2 * x ^ 2 - 4 * x + c = 5) : c ≤ 7 :=
by sorry

end NUMINAMATH_GPT_largest_c_in_range_l2150_215037


namespace NUMINAMATH_GPT_quotient_when_divided_by_44_l2150_215049

theorem quotient_when_divided_by_44 (N Q P : ℕ) (h1 : N = 44 * Q) (h2 : N = 35 * P + 3) : Q = 12 :=
by {
  -- Proof
  sorry
}

end NUMINAMATH_GPT_quotient_when_divided_by_44_l2150_215049


namespace NUMINAMATH_GPT_length_OR_coordinates_Q_area_OPQR_8_p_value_l2150_215097

noncomputable def point_R : (ℝ × ℝ) := (0, 4)

noncomputable def OR_distance : ℝ := 0 - 4 -- the vertical distance from O to R

theorem length_OR : OR_distance = 4 := sorry

noncomputable def point_Q (p : ℝ) : (ℝ × ℝ) := (p, 2 * p + 4)

theorem coordinates_Q (p : ℝ) : point_Q p = (p, 2 * p + 4) := sorry

noncomputable def area_OPQR (p : ℝ) : ℝ := 
  let OR : ℝ := 4
  let PQ : ℝ := 2 * p + 4
  let OP : ℝ := p
  1 / 2 * (OR + PQ) * OP

theorem area_OPQR_8 : area_OPQR 8 = 96 := sorry

theorem p_value (h : area_OPQR p = 77) : p = 7 := sorry

end NUMINAMATH_GPT_length_OR_coordinates_Q_area_OPQR_8_p_value_l2150_215097


namespace NUMINAMATH_GPT_weight_of_dog_l2150_215003

theorem weight_of_dog (k r d : ℕ) (h1 : k + r + d = 30) (h2 : k + r = 2 * d) (h3 : k + d = r) : d = 10 :=
by
  sorry

end NUMINAMATH_GPT_weight_of_dog_l2150_215003


namespace NUMINAMATH_GPT_number_of_days_to_catch_fish_l2150_215045

variable (fish_per_day : ℕ) (fillets_per_fish : ℕ) (total_fillets : ℕ)

theorem number_of_days_to_catch_fish (h1 : fish_per_day = 2) 
                                    (h2 : fillets_per_fish = 2) 
                                    (h3 : total_fillets = 120) : 
                                    (total_fillets / fillets_per_fish) / fish_per_day = 30 :=
by sorry

end NUMINAMATH_GPT_number_of_days_to_catch_fish_l2150_215045


namespace NUMINAMATH_GPT_clerical_percentage_after_reduction_l2150_215061

-- Define the initial conditions
def total_employees : ℕ := 3600
def clerical_fraction : ℚ := 1/4
def reduction_fraction : ℚ := 1/4

-- Define the intermediate calculations
def initial_clerical_employees : ℚ := clerical_fraction * total_employees
def clerical_reduction : ℚ := reduction_fraction * initial_clerical_employees
def new_clerical_employees : ℚ := initial_clerical_employees - clerical_reduction
def total_employees_after_reduction : ℚ := total_employees - clerical_reduction

-- State the theorem
theorem clerical_percentage_after_reduction :
  (new_clerical_employees / total_employees_after_reduction) * 100 = 20 :=
sorry

end NUMINAMATH_GPT_clerical_percentage_after_reduction_l2150_215061


namespace NUMINAMATH_GPT_find_largest_value_l2150_215005

theorem find_largest_value
  (h1: 0 < Real.sin 2) (h2: Real.sin 2 < 1)
  (h3: Real.log 2 / Real.log (1 / 3) < 0)
  (h4: Real.log (1 / 3) / Real.log (1 / 2) > 1) :
  Real.log (1 / 3) / Real.log (1 / 2) > Real.sin 2 ∧ 
  Real.log (1 / 3) / Real.log (1 / 2) > Real.log 2 / Real.log (1 / 3) := by
  sorry

end NUMINAMATH_GPT_find_largest_value_l2150_215005


namespace NUMINAMATH_GPT_product_of_two_integers_l2150_215021

def gcd_lcm_prod (x y : ℕ) :=
  Nat.gcd x y = 8 ∧ Nat.lcm x y = 48

theorem product_of_two_integers (x y : ℕ) (h : gcd_lcm_prod x y) : x * y = 384 :=
by
  sorry

end NUMINAMATH_GPT_product_of_two_integers_l2150_215021


namespace NUMINAMATH_GPT_pastries_average_per_day_l2150_215089

theorem pastries_average_per_day :
  let monday_sales := 2
  let tuesday_sales := monday_sales + 1
  let wednesday_sales := tuesday_sales + 1
  let thursday_sales := wednesday_sales + 1
  let friday_sales := thursday_sales + 1
  let saturday_sales := friday_sales + 1
  let sunday_sales := saturday_sales + 1
  let total_sales := monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales + saturday_sales + sunday_sales
  let days := 7
  total_sales / days = 5 := by
  sorry

end NUMINAMATH_GPT_pastries_average_per_day_l2150_215089


namespace NUMINAMATH_GPT_line_eq_circle_eq_l2150_215041

section
  variable (A B : ℝ × ℝ)
  variable (A_eq : A = (4, 6))
  variable (B_eq : B = (-2, 4))

  theorem line_eq : ∃ (a b c : ℝ), (a, b, c) = (1, -3, 14) ∧ ∀ x y, (y - 6) = ((4 - 6) / (-2 - 4)) * (x - 4) → a * x + b * y + c = 0 :=
  sorry

  theorem circle_eq : ∃ (h k r : ℝ), (h, k, r) = (1, 5, 10) ∧ ∀ x y, (x - 1)^2 + (y - 5)^2 = 10 :=
  sorry
end

end NUMINAMATH_GPT_line_eq_circle_eq_l2150_215041


namespace NUMINAMATH_GPT_sum_squares_nonpositive_l2150_215079

theorem sum_squares_nonpositive (a b c : ℝ) (h : a + b + c = 0) : ab + bc + ac ≤ 0 :=
by {
  sorry
}

end NUMINAMATH_GPT_sum_squares_nonpositive_l2150_215079


namespace NUMINAMATH_GPT_rectangle_area_l2150_215087

theorem rectangle_area (W : ℕ) (hW : W = 5) (L : ℕ) (hL : L = 4 * W) : ∃ (A : ℕ), A = L * W ∧ A = 100 := 
by
  use 100
  sorry

end NUMINAMATH_GPT_rectangle_area_l2150_215087


namespace NUMINAMATH_GPT_original_laborers_count_l2150_215006

theorem original_laborers_count (L : ℕ) (h1 : (L - 7) * 10 = L * 6) : L = 18 :=
sorry

end NUMINAMATH_GPT_original_laborers_count_l2150_215006


namespace NUMINAMATH_GPT_bars_sold_this_week_l2150_215054

-- Definitions based on conditions
def total_bars : Nat := 18
def bars_sold_last_week : Nat := 5
def bars_needed_to_sell : Nat := 6

-- Statement of the proof problem
theorem bars_sold_this_week : (total_bars - (bars_needed_to_sell + bars_sold_last_week)) = 2 := by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_bars_sold_this_week_l2150_215054


namespace NUMINAMATH_GPT_initial_boys_count_l2150_215090

variable (q : ℕ) -- total number of children initially in the group
variable (b : ℕ) -- number of boys initially in the group

-- Initial condition: 60% of the group are boys initially
def initial_boys (q : ℕ) : ℕ := 6 * q / 10

-- Change after event: three boys leave, three girls join
def boys_after_event (b : ℕ) : ℕ := b - 3

-- After the event, the number of boys is 50% of the total group
def boys_percentage_after_event (b : ℕ) (q : ℕ) : Prop :=
  boys_after_event b = 5 * q / 10

theorem initial_boys_count :
  ∃ b q : ℕ, b = initial_boys q ∧ boys_percentage_after_event b q → b = 18 := 
sorry

end NUMINAMATH_GPT_initial_boys_count_l2150_215090


namespace NUMINAMATH_GPT_find_angle_and_sum_of_sides_l2150_215088

noncomputable def triangle_conditions 
    (a b c : ℝ) (C : ℝ)
    (area : ℝ) : Prop :=
  a^2 + b^2 - c^2 = a * b ∧
  c = Real.sqrt 7 ∧
  area = (3 * Real.sqrt 3) / 2 

theorem find_angle_and_sum_of_sides
    (a b c C : ℝ)
    (area : ℝ)
    (h : triangle_conditions a b c C area) :
    C = Real.pi / 3 ∧ a + b = 5 := by
  sorry

end NUMINAMATH_GPT_find_angle_and_sum_of_sides_l2150_215088


namespace NUMINAMATH_GPT_geometric_sequence_ratio_l2150_215024

variable {a : ℕ → ℝ} -- Define the geometric sequence {a_n}

-- Conditions: The sequence is geometric with positive terms
variable (q : ℝ) (hq : q > 0) (hgeo : ∀ n, a (n + 1) = q * a n)

-- Additional condition: a2, 1/2 a3, and a1 form an arithmetic sequence
variable (hseq : a 1 - (1 / 2) * a 2 = (1 / 2) * a 2 - a 0)

theorem geometric_sequence_ratio :
  (a 3 + a 4) / (a 2 + a 3) = (1 + Real.sqrt 5) / 2 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_ratio_l2150_215024


namespace NUMINAMATH_GPT_even_number_divisible_by_8_l2150_215069

theorem even_number_divisible_by_8 {n : ℤ} (h : ∃ k : ℤ, n = 2 * k) : 
  (n * (n^2 + 20)) % 8 = 0 ∧ 
  (n * (n^2 - 20)) % 8 = 0 ∧ 
  (n * (n^2 + 4)) % 8 = 0 ∧ 
  (n * (n^2 - 4)) % 8 = 0 :=
by
  sorry

end NUMINAMATH_GPT_even_number_divisible_by_8_l2150_215069


namespace NUMINAMATH_GPT_polynomial_sum_l2150_215085

def f (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def g (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def h (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2
def j (x : ℝ) : ℝ := x^2 - x - 3

theorem polynomial_sum (x : ℝ) : f x + g x + h x + j x = -3 * x^2 + 11 * x - 15 := by
  sorry

end NUMINAMATH_GPT_polynomial_sum_l2150_215085


namespace NUMINAMATH_GPT_tammy_haircuts_l2150_215053

theorem tammy_haircuts (total_haircuts free_haircuts haircuts_to_next_free : ℕ) 
(h1 : free_haircuts = 5) 
(h2 : haircuts_to_next_free = 5) 
(h3 : total_haircuts = 79) : 
(haircuts_to_next_free = 5) :=
by {
  sorry
}

end NUMINAMATH_GPT_tammy_haircuts_l2150_215053


namespace NUMINAMATH_GPT_sum_of_values_l2150_215093

namespace ProofProblem

noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then 5 * x - 3 else x^2 - 4 * x + 3

theorem sum_of_values (s : Finset ℝ) : 
  (∀ x ∈ s, f x = 2) → s.sum id = 4 :=
by 
  sorry

end ProofProblem

end NUMINAMATH_GPT_sum_of_values_l2150_215093


namespace NUMINAMATH_GPT_claire_photos_l2150_215080

variable (C : ℕ) -- Claire's photos
variable (L : ℕ) -- Lisa's photos
variable (R : ℕ) -- Robert's photos

-- Conditions
axiom Lisa_photos : L = 3 * C
axiom Robert_photos : R = C + 16
axiom Lisa_Robert_same : L = R

-- Proof Goal
theorem claire_photos : C = 8 :=
by
  -- Sorry skips the proof and allows the theorem to compile
  sorry

end NUMINAMATH_GPT_claire_photos_l2150_215080


namespace NUMINAMATH_GPT_tan_C_l2150_215060

theorem tan_C (A B C : ℝ) (hABC : A + B + C = π) (tan_A : Real.tan A = 1 / 2) 
  (cos_B : Real.cos B = 3 * Real.sqrt 10 / 10) : Real.tan C = -1 :=
by
  sorry

end NUMINAMATH_GPT_tan_C_l2150_215060


namespace NUMINAMATH_GPT_odd_function_f1_eq_4_l2150_215008

theorem odd_function_f1_eq_4 (f : ℝ → ℝ) (a : ℝ) 
  (h1 : ∀ x, f (-x) = -f x)
  (h2 : ∀ x, x < 0 → f x = x^2 + a * x)
  (h3 : f 2 = 6) : 
  f 1 = 4 :=
by sorry

end NUMINAMATH_GPT_odd_function_f1_eq_4_l2150_215008


namespace NUMINAMATH_GPT_min_purchase_amount_is_18_l2150_215078

def burger_cost := 2 * 3.20
def fries_cost := 2 * 1.90
def milkshake_cost := 2 * 2.40
def current_total := burger_cost + fries_cost + milkshake_cost
def additional_needed := 3.00
def min_purchase_amount_for_free_delivery := current_total + additional_needed

theorem min_purchase_amount_is_18 : min_purchase_amount_for_free_delivery = 18 := by
  sorry

end NUMINAMATH_GPT_min_purchase_amount_is_18_l2150_215078


namespace NUMINAMATH_GPT_king_and_queen_ages_l2150_215072

variable (K Q : ℕ)

theorem king_and_queen_ages (h1 : K = 2 * (Q - (K - Q)))
                            (h2 : K + (K + (K - Q)) = 63) :
                            K = 28 ∧ Q = 21 := by
  sorry

end NUMINAMATH_GPT_king_and_queen_ages_l2150_215072


namespace NUMINAMATH_GPT_cos_alpha_value_l2150_215095

theorem cos_alpha_value (α : ℝ) (h1 : 0 < α) (h2 : α < Real.pi / 2) (h3 : Real.sin α = 3 / 5) :
  Real.cos α = 4 / 5 :=
by
  sorry

end NUMINAMATH_GPT_cos_alpha_value_l2150_215095


namespace NUMINAMATH_GPT_cubic_sum_of_roots_l2150_215034

theorem cubic_sum_of_roots (a b c : ℝ) 
  (h1 : a + b + c = -1)
  (h2 : a * b + b * c + c * a = -333)
  (h3 : a * b * c = 1001) :
  a^3 + b^3 + c^3 = 2003 :=
sorry

end NUMINAMATH_GPT_cubic_sum_of_roots_l2150_215034


namespace NUMINAMATH_GPT_min_odd_solution_l2150_215043

theorem min_odd_solution (a m1 m2 n1 n2 : ℕ)
  (h1: a = m1^2 + n1^2)
  (h2: a^2 = m2^2 + n2^2)
  (h3: m1 - n1 = m2 - n2)
  (h4: a > 5)
  (h5: a % 2 = 1) :
  a = 261 :=
sorry

end NUMINAMATH_GPT_min_odd_solution_l2150_215043


namespace NUMINAMATH_GPT_number_of_pens_each_student_gets_l2150_215081

theorem number_of_pens_each_student_gets 
    (total_pens : ℕ) (total_pencils : ℕ) (max_students : ℕ)
    (h1 : total_pens = 1001) (h2 : total_pencils = 910) (h3 : max_students = 91) :
  (total_pens / Nat.gcd total_pens total_pencils) = 11 :=
by
  sorry

end NUMINAMATH_GPT_number_of_pens_each_student_gets_l2150_215081


namespace NUMINAMATH_GPT_lightsaber_ratio_l2150_215048

theorem lightsaber_ratio (T L : ℕ) (hT : T = 1000) (hTotal : L + T = 3000) : L / T = 2 :=
by
  sorry

end NUMINAMATH_GPT_lightsaber_ratio_l2150_215048


namespace NUMINAMATH_GPT_paperclips_exceed_200_at_friday_l2150_215059

def paperclips_on_day (n : ℕ) : ℕ :=
  3 * 4^n

theorem paperclips_exceed_200_at_friday : 
  ∃ n : ℕ, n = 4 ∧ paperclips_on_day n > 200 :=
by
  sorry

end NUMINAMATH_GPT_paperclips_exceed_200_at_friday_l2150_215059


namespace NUMINAMATH_GPT_minimum_sum_distances_square_l2150_215010

noncomputable def minimum_sum_of_distances
    (A B : ℝ × ℝ)
    (d : ℝ)
    (h_dist: dist A B = d)
    : ℝ :=
(1 + Real.sqrt 2) * d

theorem minimum_sum_distances_square
    (A B : ℝ × ℝ)
    (d : ℝ)
    (h_dist: dist A B = d)
    : minimum_sum_of_distances A B d h_dist = (1 + Real.sqrt 2) * d := by
sorry

end NUMINAMATH_GPT_minimum_sum_distances_square_l2150_215010


namespace NUMINAMATH_GPT_percent_workday_in_meetings_l2150_215013

def first_meeting_duration : ℕ := 30
def second_meeting_duration : ℕ := 3 * first_meeting_duration
def third_meeting_duration : ℕ := 2 * second_meeting_duration
def total_meeting_time : ℕ := first_meeting_duration + second_meeting_duration + third_meeting_duration
def workday_duration : ℕ := 10 * 60

theorem percent_workday_in_meetings : (total_meeting_time : ℚ) / workday_duration * 100 = 50 := by
  sorry

end NUMINAMATH_GPT_percent_workday_in_meetings_l2150_215013


namespace NUMINAMATH_GPT_correct_quotient_and_remainder_l2150_215058

theorem correct_quotient_and_remainder:
  let incorrect_divisor := 47
  let incorrect_quotient := 5
  let incorrect_remainder := 8
  let incorrect_dividend := incorrect_divisor * incorrect_quotient + incorrect_remainder
  let correct_dividend := 243
  let correct_divisor := 74
  (correct_dividend / correct_divisor = 3 ∧ correct_dividend % correct_divisor = 21) :=
by sorry

end NUMINAMATH_GPT_correct_quotient_and_remainder_l2150_215058
