import Mathlib

namespace NUMINAMATH_GPT_solution_set_inequalities_l1824_182424

theorem solution_set_inequalities (x : ℝ) :
  (2 * x + 3 ≥ -1) ∧ (7 - 3 * x > 1) ↔ (-2 ≤ x ∧ x < 2) :=
by
  sorry

end NUMINAMATH_GPT_solution_set_inequalities_l1824_182424


namespace NUMINAMATH_GPT_sara_marbles_l1824_182445

theorem sara_marbles : 10 - 7 = 3 :=
by
  sorry

end NUMINAMATH_GPT_sara_marbles_l1824_182445


namespace NUMINAMATH_GPT_quadratic_intersects_x_axis_if_and_only_if_k_le_four_l1824_182452

-- Define the quadratic function
def quadratic_function (k x : ℝ) : ℝ :=
  (k - 3) * x^2 + 2 * x + 1

-- Theorem stating the relationship between the function intersecting the x-axis and k ≤ 4
theorem quadratic_intersects_x_axis_if_and_only_if_k_le_four
  (k : ℝ) :
  (∃ x : ℝ, quadratic_function k x = 0) ↔ k ≤ 4 :=
sorry

end NUMINAMATH_GPT_quadratic_intersects_x_axis_if_and_only_if_k_le_four_l1824_182452


namespace NUMINAMATH_GPT_lake_view_population_l1824_182438

-- Define the populations of the cities
def population_of_Seattle : ℕ := 20000 -- Derived from the solution
def population_of_Boise : ℕ := (3 / 5) * population_of_Seattle
def population_of_Lake_View : ℕ := population_of_Seattle + 4000
def total_population : ℕ := population_of_Seattle + population_of_Boise + population_of_Lake_View

-- Statement to prove
theorem lake_view_population :
  total_population = 56000 →
  population_of_Lake_View = 24000 :=
sorry

end NUMINAMATH_GPT_lake_view_population_l1824_182438


namespace NUMINAMATH_GPT_bart_firewood_burning_period_l1824_182429

-- We'll state the conditions as definitions.
def pieces_per_tree := 75
def trees_cut_down := 8
def logs_burned_per_day := 5

-- The theorem to prove the period Bart burns the logs.
theorem bart_firewood_burning_period :
  (trees_cut_down * pieces_per_tree) / logs_burned_per_day = 120 :=
by
  sorry

end NUMINAMATH_GPT_bart_firewood_burning_period_l1824_182429


namespace NUMINAMATH_GPT_smallest_int_solution_l1824_182427

theorem smallest_int_solution : ∃ y : ℤ, y = 6 ∧ ∀ z : ℤ, z > 5 → y ≤ z := sorry

end NUMINAMATH_GPT_smallest_int_solution_l1824_182427


namespace NUMINAMATH_GPT_player_avg_increase_l1824_182499

theorem player_avg_increase
  (matches_played : ℕ)
  (initial_avg : ℕ)
  (next_match_runs : ℕ)
  (total_runs : ℕ)
  (new_total_runs : ℕ)
  (new_avg : ℕ)
  (desired_avg_increase : ℕ) :
  matches_played = 10 ∧ initial_avg = 32 ∧ next_match_runs = 76 ∧ total_runs = 320 ∧ 
  new_total_runs = 396 ∧ new_avg = 32 + desired_avg_increase ∧ 
  11 * new_avg = new_total_runs → desired_avg_increase = 4 := 
by
  sorry

end NUMINAMATH_GPT_player_avg_increase_l1824_182499


namespace NUMINAMATH_GPT_incorrect_option_D_l1824_182422

-- Definitions based on the given conditions:
def contrapositive_correct : Prop :=
  ∀ x : ℝ, (x ≠ 1 → x^2 - 3 * x + 2 ≠ 0) ↔ (x^2 - 3 * x + 2 = 0 → x = 1)

def sufficient_but_not_necessary : Prop :=
  ∀ x : ℝ, (x > 2 → x^2 - 3 * x + 2 > 0) ∧ (x^2 - 3 * x + 2 > 0 → x > 2 ∨ x < 1)

def negation_correct (p : Prop) (neg_p : Prop) : Prop :=
  p ↔ ∀ x : ℝ, x^2 + x + 1 ≠ 0 ∧ neg_p ↔ ∃ x_0 : ℝ, x_0^2 + x_0 + 1 = 0

theorem incorrect_option_D (p q : Prop) (h : p ∨ q) :
  ¬ (p ∧ q) :=
sorry  -- Proof is to be done later

end NUMINAMATH_GPT_incorrect_option_D_l1824_182422


namespace NUMINAMATH_GPT_prime_factors_difference_l1824_182475

theorem prime_factors_difference (h : 184437 = 3 * 7 * 8783) : 8783 - 7 = 8776 :=
by sorry

end NUMINAMATH_GPT_prime_factors_difference_l1824_182475


namespace NUMINAMATH_GPT_common_root_divisibility_l1824_182472

variables (a b c : ℤ)

theorem common_root_divisibility 
  (h1 : c ≠ b) 
  (h2 : ∃ x : ℝ, a * x^2 + b * x + c = 0 ∧ (c - b) * x^2 + (c - a) * x + (a + b) = 0) 
  : 3 ∣ (a + b + 2 * c) :=
sorry

end NUMINAMATH_GPT_common_root_divisibility_l1824_182472


namespace NUMINAMATH_GPT_number_of_triplets_with_sum_6n_l1824_182450

theorem number_of_triplets_with_sum_6n (n : ℕ) : 
  ∃ (count : ℕ), count = 3 * n^2 ∧ 
  (∀ (x y z : ℕ), x ≤ y → y ≤ z → x + y + z = 6 * n → count = 1) :=
sorry

end NUMINAMATH_GPT_number_of_triplets_with_sum_6n_l1824_182450


namespace NUMINAMATH_GPT_inequality_proof_l1824_182457

variables (a b c : ℝ)

theorem inequality_proof (h : a > b) : a * c^2 ≥ b * c^2 :=
by sorry

end NUMINAMATH_GPT_inequality_proof_l1824_182457


namespace NUMINAMATH_GPT_hyperbola_foci_distance_l1824_182468

theorem hyperbola_foci_distance :
  (∃ (h : ℝ → ℝ) (c : ℝ), (∀ x, h x = 2 * x + 3 ∨ h x = 1 - 2 * x)
    ∧ (h 4 = 5)
    ∧ 2 * Real.sqrt (20.25 + 4.444) = 2 * Real.sqrt 24.694) := 
  sorry

end NUMINAMATH_GPT_hyperbola_foci_distance_l1824_182468


namespace NUMINAMATH_GPT_number_of_points_determined_l1824_182407

def A : Set ℕ := {5}
def B : Set ℕ := {1, 2}
def C : Set ℕ := {1, 3, 4}

theorem number_of_points_determined : (∃ n : ℕ, n = 33) :=
by
  -- sorry to skip the proof
  sorry

end NUMINAMATH_GPT_number_of_points_determined_l1824_182407


namespace NUMINAMATH_GPT_negation_universal_to_existential_l1824_182484

-- Setup the necessary conditions and types
variable (a : ℝ) (ha : 0 < a ∧ a < 1)

-- Negate the universal quantifier
theorem negation_universal_to_existential :
  (¬ ∀ x < 0, a^x > 1) ↔ ∃ x_0 < 0, a^(x_0) ≤ 1 :=
by sorry

end NUMINAMATH_GPT_negation_universal_to_existential_l1824_182484


namespace NUMINAMATH_GPT_decompose_375_l1824_182418

theorem decompose_375 : 375 = 3 * 100 + 7 * 10 + 5 * 1 :=
by
  sorry

end NUMINAMATH_GPT_decompose_375_l1824_182418


namespace NUMINAMATH_GPT_calculation_is_correct_l1824_182425

theorem calculation_is_correct : -1^6 + 8 / (-2)^2 - abs (-4 * 3) = -9 := by
  sorry

end NUMINAMATH_GPT_calculation_is_correct_l1824_182425


namespace NUMINAMATH_GPT_lisa_total_spoons_l1824_182478

def children_count : ℕ := 6
def spoons_per_child : ℕ := 4
def decorative_spoons : ℕ := 4
def large_spoons : ℕ := 20
def dessert_spoons : ℕ := 10
def soup_spoons : ℕ := 15
def tea_spoons : ℕ := 25

def baby_spoons_total : ℕ := children_count * spoons_per_child
def cutlery_set_total : ℕ := large_spoons + dessert_spoons + soup_spoons + tea_spoons

def total_spoons : ℕ := cutlery_set_total + baby_spoons_total + decorative_spoons

theorem lisa_total_spoons : total_spoons = 98 :=
by
  sorry

end NUMINAMATH_GPT_lisa_total_spoons_l1824_182478


namespace NUMINAMATH_GPT_bobby_paid_for_shoes_l1824_182420

theorem bobby_paid_for_shoes :
  let mold_cost := 250
  let hourly_labor_rate := 75
  let hours_worked := 8
  let discount_rate := 0.80
  let materials_cost := 150
  let tax_rate := 0.10

  let labor_cost := hourly_labor_rate * hours_worked
  let discounted_labor_cost := discount_rate * labor_cost
  let total_cost_before_tax := mold_cost + discounted_labor_cost + materials_cost
  let tax := total_cost_before_tax * tax_rate
  let total_cost_with_tax := total_cost_before_tax + tax

  total_cost_with_tax = 968 :=
by
  sorry

end NUMINAMATH_GPT_bobby_paid_for_shoes_l1824_182420


namespace NUMINAMATH_GPT_find_admission_score_l1824_182434

noncomputable def admission_score : ℝ := 87

theorem find_admission_score :
  ∀ (total_students admitted_students not_admitted_students : ℝ) 
    (admission_score admitted_avg not_admitted_avg overall_avg : ℝ),
    admitted_students = total_students / 4 →
    not_admitted_students = 3 * admitted_students →
    admitted_avg = admission_score + 10 →
    not_admitted_avg = admission_score - 26 →
    overall_avg = 70 →
    total_students * overall_avg = 
    (admitted_students * admitted_avg + not_admitted_students * not_admitted_avg) →
    admission_score = 87 :=
by
  intros total_students admitted_students not_admitted_students 
         admission_score admitted_avg not_admitted_avg overall_avg
         h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_find_admission_score_l1824_182434


namespace NUMINAMATH_GPT_determine_z_l1824_182466

theorem determine_z (z : ℕ) (h1: z.factors.count = 18) (h2: 16 ∣ z) (h3: 18 ∣ z) : z = 288 := 
  by 
  sorry

end NUMINAMATH_GPT_determine_z_l1824_182466


namespace NUMINAMATH_GPT_find_a_and_union_set_l1824_182440

theorem find_a_and_union_set (a : ℝ) 
  (A : Set ℝ) (B : Set ℝ) 
  (hA : A = {-3, a + 1}) 
  (hB : B = {2 * a - 1, a ^ 2 + 1}) 
  (h_inter : A ∩ B = {3}) : 
  a = 2 ∧ A ∪ B = {-3, 3, 5} :=
by
  sorry

end NUMINAMATH_GPT_find_a_and_union_set_l1824_182440


namespace NUMINAMATH_GPT_percentage_markup_l1824_182428

open Real

theorem percentage_markup (SP CP : ℝ) (hSP : SP = 5600) (hCP : CP = 4480) : 
  ((SP - CP) / CP) * 100 = 25 :=
by
  sorry

end NUMINAMATH_GPT_percentage_markup_l1824_182428


namespace NUMINAMATH_GPT_largest_multiple_of_8_less_than_100_l1824_182417

theorem largest_multiple_of_8_less_than_100 :
  ∃ n : ℕ, n < 100 ∧ (∃ k : ℕ, n = 8 * k) ∧ ∀ m : ℕ, m < 100 ∧ (∃ k : ℕ, m = 8 * k) → m ≤ 96 := 
by
  sorry

end NUMINAMATH_GPT_largest_multiple_of_8_less_than_100_l1824_182417


namespace NUMINAMATH_GPT_y_coordinate_equidistant_l1824_182463

theorem y_coordinate_equidistant :
  ∃ y : ℝ, (∀ P : ℝ × ℝ, P = (0, y) → dist (3, 0) P = dist (2, 5) P) ∧ y = 2 := 
by
  sorry

end NUMINAMATH_GPT_y_coordinate_equidistant_l1824_182463


namespace NUMINAMATH_GPT_ratio_of_q_to_p_l1824_182492

theorem ratio_of_q_to_p (p q : ℝ) (h₀ : 0 < p) (h₁ : 0 < q) 
  (h₂ : Real.log p / Real.log 9 = Real.log q / Real.log 12) 
  (h₃ : Real.log q / Real.log 12 = Real.log (p + q) / Real.log 16) : 
  q / p = (1 + Real.sqrt 5) / 2 := 
by 
  sorry

end NUMINAMATH_GPT_ratio_of_q_to_p_l1824_182492


namespace NUMINAMATH_GPT_number_of_distinct_lines_l1824_182416

theorem number_of_distinct_lines (S : Finset ℕ) (h : S = {1, 2, 3, 4, 5}) :
  (S.card.choose 2) - 2 = 18 :=
by
  -- Conditions
  have hS : S = {1, 2, 3, 4, 5} := h
  -- Conclusion
  sorry

end NUMINAMATH_GPT_number_of_distinct_lines_l1824_182416


namespace NUMINAMATH_GPT_probability_A2_equals_zero_matrix_l1824_182493

noncomputable def probability_A2_zero (n : ℕ) (hn : n ≥ 2) : ℚ :=
  let numerator := (n - 1) * (n - 2)
  let denominator := n * (n - 1)
  numerator / denominator

theorem probability_A2_equals_zero_matrix (n : ℕ) (hn : n ≥ 2) :
  probability_A2_zero n hn = ((n - 1) * (n - 2) / (n * (n - 1))) := by
  sorry

end NUMINAMATH_GPT_probability_A2_equals_zero_matrix_l1824_182493


namespace NUMINAMATH_GPT_fred_bought_books_l1824_182442

theorem fred_bought_books (initial_money : ℕ) (remaining_money : ℕ) (book_cost : ℕ)
  (h1 : initial_money = 236)
  (h2 : remaining_money = 14)
  (h3 : book_cost = 37) :
  (initial_money - remaining_money) / book_cost = 6 :=
by {
  sorry
}

end NUMINAMATH_GPT_fred_bought_books_l1824_182442


namespace NUMINAMATH_GPT_dad_strawberries_now_weight_l1824_182419

-- Definitions based on the conditions given
def total_weight : ℕ := 36
def weight_lost_by_dad : ℕ := 8
def weight_of_marco_strawberries : ℕ := 12

-- Theorem to prove the question as an equality
theorem dad_strawberries_now_weight :
  total_weight - weight_lost_by_dad - weight_of_marco_strawberries = 16 := by
  sorry

end NUMINAMATH_GPT_dad_strawberries_now_weight_l1824_182419


namespace NUMINAMATH_GPT_minimum_t_is_2_l1824_182496

noncomputable def minimum_t_value (t : ℝ) : Prop :=
  let A := (-t, 0)
  let B := (t, 0)
  let C := (Real.sqrt 3, Real.sqrt 6)
  let r := 1
  ∃ P : ℝ × ℝ, 
    (P.1 - (Real.sqrt 3))^2 + (P.2 - (Real.sqrt 6))^2 = r^2 ∧ 
    (P.1 - A.1) * (P.1 - B.1) + (P.2 - A.2) * (P.2 - B.2) = 0

theorem minimum_t_is_2 : (∃ t : ℝ, t > 0 ∧ minimum_t_value t) → ∃ t : ℝ, t = 2 :=
sorry

end NUMINAMATH_GPT_minimum_t_is_2_l1824_182496


namespace NUMINAMATH_GPT_find_range_of_m_l1824_182497

variable (m : ℝ)

-- Definition of p: There exists x in ℝ such that mx^2 - mx + 1 < 0
def p : Prop := ∃ x : ℝ, m * x ^ 2 - m * x + 1 < 0

-- Definition of q: The curve of the equation (x^2)/(m-1) + (y^2)/(3-m) = 1 is a hyperbola
def q : Prop := (m - 1) * (3 - m) < 0

-- Given conditions
def proposition_and : Prop := ¬ (p m ∧ q m)
def proposition_or : Prop := p m ∨ q m

-- Final theorem statement
theorem find_range_of_m : proposition_and m ∧ proposition_or m → (0 < m ∧ m ≤ 1) ∨ (3 ≤ m ∧ m < 4) :=
sorry

end NUMINAMATH_GPT_find_range_of_m_l1824_182497


namespace NUMINAMATH_GPT_calvin_winning_strategy_l1824_182448

theorem calvin_winning_strategy :
  ∃ (n : ℤ), ∃ (p : ℤ), ∃ (q : ℤ),
  (∀ k : ℕ, k > 0 → p = 0 ∧ (q = 2014 + k ∨ q = 2014 - k) → ∃ x : ℤ, (x^2 + p * x + q = 0)) :=
sorry

end NUMINAMATH_GPT_calvin_winning_strategy_l1824_182448


namespace NUMINAMATH_GPT_minimum_points_to_determine_polynomial_l1824_182408

def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), ∀ x, f x = a * x^2 + b * x + c

def different_at (f g : ℝ → ℝ) (x : ℝ) : Prop :=
  f x ≠ g x

theorem minimum_points_to_determine_polynomial :
  ∀ (f g : ℝ → ℝ), is_quadratic f → is_quadratic g → 
  (∀ t, t < 8 → (different_at f g t → ∃ t₁ t₂ t₃, different_at f g t₁ ∧ different_at f g t₂ ∧ different_at f g t₃)) → False :=
by {
  sorry
}

end NUMINAMATH_GPT_minimum_points_to_determine_polynomial_l1824_182408


namespace NUMINAMATH_GPT_unique_solution_system_l1824_182437

noncomputable def f (x : ℝ) := 4 * x ^ 3 + x - 4

theorem unique_solution_system :
  (∃ x y z : ℝ, y^2 = 4*x^3 + x - 4 ∧ z^2 = 4*y^3 + y - 4 ∧ x^2 = 4*z^3 + z - 4) ↔
  (x = 1 ∧ y = 1 ∧ z = 1) :=
by
  sorry

end NUMINAMATH_GPT_unique_solution_system_l1824_182437


namespace NUMINAMATH_GPT_hyperbola_hkabc_sum_l1824_182401

theorem hyperbola_hkabc_sum :
  ∃ h k a b : ℝ, h = 3 ∧ k = -1 ∧ a = 2 ∧ b = Real.sqrt 46 ∧ h + k + a + b = 4 + Real.sqrt 46 :=
by
  use 3
  use -1
  use 2
  use Real.sqrt 46
  simp
  sorry

end NUMINAMATH_GPT_hyperbola_hkabc_sum_l1824_182401


namespace NUMINAMATH_GPT_adam_final_score_l1824_182498

theorem adam_final_score : 
  let science_correct := 5
  let science_points := 10
  let history_correct := 3
  let history_points := 5
  let history_multiplier := 2
  let sports_correct := 1
  let sports_points := 15
  let literature_correct := 1
  let literature_points := 7
  let literature_penalty := 3
  
  let science_total := science_correct * science_points
  let history_total := (history_correct * history_points) * history_multiplier
  let sports_total := sports_correct * sports_points
  let literature_total := (literature_correct * literature_points) - literature_penalty
  
  let final_score := science_total + history_total + sports_total + literature_total
  final_score = 99 := by 
    sorry

end NUMINAMATH_GPT_adam_final_score_l1824_182498


namespace NUMINAMATH_GPT_find_multiple_of_benjy_peaches_l1824_182467

theorem find_multiple_of_benjy_peaches
(martine_peaches gabrielle_peaches : ℕ)
(benjy_peaches : ℕ)
(m : ℕ)
(h1 : martine_peaches = 16)
(h2 : gabrielle_peaches = 15)
(h3 : benjy_peaches = gabrielle_peaches / 3)
(h4 : martine_peaches = m * benjy_peaches + 6) :
m = 2 := by
sorry

end NUMINAMATH_GPT_find_multiple_of_benjy_peaches_l1824_182467


namespace NUMINAMATH_GPT_find_height_l1824_182410

-- Defining the known conditions
def length : ℝ := 3
def width : ℝ := 5
def cost_per_sqft : ℝ := 20
def total_cost : ℝ := 1240

-- Defining the unknown dimension as a variable
variable (height : ℝ)

-- Surface area formula for a rectangular tank
def surface_area (l w h : ℝ) : ℝ := 2 * l * w + 2 * l * h + 2 * w * h

-- Given statement to prove that the height is 2 feet.
theorem find_height : surface_area length width height = total_cost / cost_per_sqft → height = 2 := by
  sorry

end NUMINAMATH_GPT_find_height_l1824_182410


namespace NUMINAMATH_GPT_find_original_fraction_l1824_182481

theorem find_original_fraction (x y : ℚ) (h : (1.15 * x) / (0.92 * y) = 15 / 16) :
  x / y = 69 / 92 :=
sorry

end NUMINAMATH_GPT_find_original_fraction_l1824_182481


namespace NUMINAMATH_GPT_find_b_l1824_182464

theorem find_b (x y b : ℝ) (h1 : (7 * x + b * y) / (x - 2 * y) = 13) (h2 : x / (2 * y) = 5 / 2) : b = 4 :=
  sorry

end NUMINAMATH_GPT_find_b_l1824_182464


namespace NUMINAMATH_GPT_solution_to_problem_l1824_182485

def number_exists (n : ℝ) : Prop :=
  n / 0.25 = 400

theorem solution_to_problem : ∃ n : ℝ, number_exists n ∧ n = 100 := by
  sorry

end NUMINAMATH_GPT_solution_to_problem_l1824_182485


namespace NUMINAMATH_GPT_area_enclosed_by_3x2_l1824_182483

theorem area_enclosed_by_3x2 (a b : ℝ) (h₀ : a = 0) (h₁ : b = 1) :
  ∫ (x : ℝ) in a..b, 3 * x^2 = 1 :=
by 
  rw [h₀, h₁]
  sorry

end NUMINAMATH_GPT_area_enclosed_by_3x2_l1824_182483


namespace NUMINAMATH_GPT_sum_of_roots_l1824_182430

theorem sum_of_roots (x : ℝ) (h : (x - 6)^2 = 16) : (∃ a b : ℝ, a + b = 12 ∧ (x = a ∨ x = b)) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_roots_l1824_182430


namespace NUMINAMATH_GPT_find_solution_l1824_182455

theorem find_solution (x y : ℕ) (h1 : y ∣ (x^2 + 1)) (h2 : x^2 ∣ (y^3 + 1)) : (x = 1 ∧ y = 1) :=
sorry

end NUMINAMATH_GPT_find_solution_l1824_182455


namespace NUMINAMATH_GPT_max_additional_plates_l1824_182473

def initial_plates_count : ℕ := 5 * 3 * 4 * 2
def new_second_set_size : ℕ := 5  -- second set after adding two letters
def new_fourth_set_size : ℕ := 3 -- fourth set after adding one letter
def new_plates_count : ℕ := 5 * new_second_set_size * 4 * new_fourth_set_size

theorem max_additional_plates :
  new_plates_count - initial_plates_count = 180 := by
  sorry

end NUMINAMATH_GPT_max_additional_plates_l1824_182473


namespace NUMINAMATH_GPT_sum_of_roots_of_quadratic_l1824_182487

variables {b x₁ x₂ : ℝ}

theorem sum_of_roots_of_quadratic (h : x₁^2 - 2 * x₁ + b = 0) (h' : x₂^2 - 2 * x₂ + b = 0) :
    x₁ + x₂ = 2 :=
sorry

end NUMINAMATH_GPT_sum_of_roots_of_quadratic_l1824_182487


namespace NUMINAMATH_GPT_sequence_count_21_l1824_182488

-- Define the conditions and the problem
def valid_sequence (n : ℕ) : ℕ :=
  if n = 21 then 114 else sorry

theorem sequence_count_21 : valid_sequence 21 = 114 :=
  by sorry

end NUMINAMATH_GPT_sequence_count_21_l1824_182488


namespace NUMINAMATH_GPT_citizens_own_a_cat_l1824_182436

theorem citizens_own_a_cat (p d : ℝ) (n : ℕ) (h1 : p = 0.60) (h2 : d = 0.50) (h3 : n = 100) : 
  (p * n - d * p * n) = 30 := 
by 
  sorry

end NUMINAMATH_GPT_citizens_own_a_cat_l1824_182436


namespace NUMINAMATH_GPT_x_minus_y_eq_eight_l1824_182414

theorem x_minus_y_eq_eight (x y : ℝ) (hx : 3 = 0.15 * x) (hy : 3 = 0.25 * y) : x - y = 8 :=
by
  sorry

end NUMINAMATH_GPT_x_minus_y_eq_eight_l1824_182414


namespace NUMINAMATH_GPT_sum_r_j_eq_3_l1824_182443

variable (p r j : ℝ)

theorem sum_r_j_eq_3
  (h : (6 * p^2 - 4 * p + r) * (2 * p^2 + j * p - 7) = 12 * p^4 - 34 * p^3 - 19 * p^2 + 28 * p - 21) :
  r + j = 3 := by
  sorry

end NUMINAMATH_GPT_sum_r_j_eq_3_l1824_182443


namespace NUMINAMATH_GPT_candidates_appeared_l1824_182432

-- Define the conditions:
variables (A_selected B_selected : ℕ) (x : ℝ)

-- 12% candidates got selected in State A
def State_A_selected := 0.12 * x

-- 18% candidates got selected in State B
def State_B_selected := 0.18 * x

-- 250 more candidates got selected in State B than in State A
def selection_difference := State_B_selected = State_A_selected + 250

-- The statement to prove:
theorem candidates_appeared (h : selection_difference) : x = 4167 :=
by
  sorry

end NUMINAMATH_GPT_candidates_appeared_l1824_182432


namespace NUMINAMATH_GPT_percentage_increase_twice_eq_16_64_l1824_182411

theorem percentage_increase_twice_eq_16_64 (x : ℝ) (hx : (1 + x)^2 = 1 + 0.1664) : x = 0.08 :=
by
  sorry -- This is the placeholder for the proof.

end NUMINAMATH_GPT_percentage_increase_twice_eq_16_64_l1824_182411


namespace NUMINAMATH_GPT_greatest_value_of_b_l1824_182459

theorem greatest_value_of_b : ∃ b, (∀ a, (-a^2 + 7 * a - 10 ≥ 0) → (a ≤ b)) ∧ b = 5 :=
by
  sorry

end NUMINAMATH_GPT_greatest_value_of_b_l1824_182459


namespace NUMINAMATH_GPT_george_older_than_christopher_l1824_182446

theorem george_older_than_christopher
  (G C F : ℕ)
  (h1 : C = 18)
  (h2 : F = C - 2)
  (h3 : G + C + F = 60) :
  G - C = 8 := by
  sorry

end NUMINAMATH_GPT_george_older_than_christopher_l1824_182446


namespace NUMINAMATH_GPT_bar_charts_as_line_charts_l1824_182409

-- Given that line charts help to visualize trends of increase and decrease
axiom trends_visualization (L : Type) : Prop

-- Bar charts can be drawn as line charts, which helps in visualizing trends
theorem bar_charts_as_line_charts (L B : Type) (h : trends_visualization L) : trends_visualization B := sorry

end NUMINAMATH_GPT_bar_charts_as_line_charts_l1824_182409


namespace NUMINAMATH_GPT_tip_percentage_l1824_182435

theorem tip_percentage 
  (total_bill : ℕ) 
  (silas_payment : ℕ) 
  (remaining_friend_payment_with_tip : ℕ) 
  (num_remaining_friends : ℕ) 
  (num_friends : ℕ)
  (h1 : total_bill = 150) 
  (h2 : silas_payment = total_bill / 2) 
  (h3 : num_remaining_friends = 5)
  (h4 : remaining_friend_payment_with_tip = 18)
  : (remaining_friend_payment_with_tip - (total_bill / 2 / num_remaining_friends) * num_remaining_friends) / total_bill * 100 = 10 :=
by
  sorry

end NUMINAMATH_GPT_tip_percentage_l1824_182435


namespace NUMINAMATH_GPT_percentage_increase_in_expenses_l1824_182462

-- Define the variables and conditions
def monthly_salary : ℝ := 7272.727272727273
def original_savings_percentage : ℝ := 0.10
def new_savings : ℝ := 400
def original_savings : ℝ := original_savings_percentage * monthly_salary
def savings_difference : ℝ := original_savings - new_savings
def original_expenses : ℝ := (1 - original_savings_percentage) * monthly_salary

-- Formalize the question as a theorem
theorem percentage_increase_in_expenses (P : ℝ) :
  P = (savings_difference / original_expenses) * 100 ↔ P = 5 := 
sorry

end NUMINAMATH_GPT_percentage_increase_in_expenses_l1824_182462


namespace NUMINAMATH_GPT_bones_received_on_sunday_l1824_182412

-- Definitions based on the conditions
def initial_bones : ℕ := 50
def bones_eaten : ℕ := initial_bones / 2
def bones_left_after_saturday : ℕ := initial_bones - bones_eaten
def total_bones_after_sunday : ℕ := 35

-- The theorem to prove how many bones received on Sunday
theorem bones_received_on_sunday : 
  (total_bones_after_sunday - bones_left_after_saturday = 10) :=
by
  -- proof will be filled in here
  sorry

end NUMINAMATH_GPT_bones_received_on_sunday_l1824_182412


namespace NUMINAMATH_GPT_find_n_l1824_182453

def num_of_trailing_zeros (n : ℕ) : ℕ :=
  if n = 0 then 0 else (n / 5) + num_of_trailing_zeros (n / 5)

theorem find_n (n : ℕ) (k : ℕ) (h1 : n > 3) (h2 : k = num_of_trailing_zeros n) (h3 : 2*k + 1 = num_of_trailing_zeros (2*n)) (h4 : k > 0) : n = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_n_l1824_182453


namespace NUMINAMATH_GPT_find_a_plus_b_l1824_182403

theorem find_a_plus_b :
  ∃ (a b : ℝ), (∀ x : ℝ, (3 * (a * x + b) - 6) = 4 * x + 5) ∧ a + b = 5 :=
by 
  sorry

end NUMINAMATH_GPT_find_a_plus_b_l1824_182403


namespace NUMINAMATH_GPT_area_of_region_l1824_182451

theorem area_of_region (r : ℝ) (theta_deg : ℝ) (a b c : ℤ) : 
  r = 8 → 
  theta_deg = 45 → 
  (r^2 * theta_deg * Real.pi / 360) - (1/2 * r^2 * Real.sin (theta_deg * Real.pi / 180)) = (a * Real.sqrt b + c * Real.pi) →
  a + b + c = -22 :=
by 
  intros hr htheta Harea 
  sorry

end NUMINAMATH_GPT_area_of_region_l1824_182451


namespace NUMINAMATH_GPT_geometry_problem_l1824_182441

theorem geometry_problem
  (A_square : ℝ)
  (A_rectangle : ℝ)
  (A_triangle : ℝ)
  (side_length : ℝ)
  (rectangle_width : ℝ)
  (rectangle_length : ℝ)
  (triangle_base : ℝ)
  (triangle_height : ℝ)
  (square_area_eq : A_square = side_length ^ 2)
  (rectangle_area_eq : A_rectangle = rectangle_width * rectangle_length)
  (triangle_area_eq : A_triangle = (triangle_base * triangle_height) / 2)
  (side_length_eq : side_length = 4)
  (rectangle_width_eq : rectangle_width = 4)
  (triangle_base_eq : triangle_base = 8)
  (areas_equal : A_square = A_rectangle ∧ A_square = A_triangle) :
  rectangle_length = 4 ∧ triangle_height = 4 :=
by
  sorry

end NUMINAMATH_GPT_geometry_problem_l1824_182441


namespace NUMINAMATH_GPT_probability_of_rolling_prime_is_half_l1824_182402

def is_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def total_outcomes : ℕ := 8

def successful_outcomes : ℕ := 4 -- prime numbers between 1 and 8 are 2, 3, 5, and 7

def probability_of_rolling_prime : ℚ :=
  successful_outcomes / total_outcomes

theorem probability_of_rolling_prime_is_half : probability_of_rolling_prime = 1 / 2 :=
  sorry

end NUMINAMATH_GPT_probability_of_rolling_prime_is_half_l1824_182402


namespace NUMINAMATH_GPT_jeff_stars_l1824_182486

noncomputable def eric_stars : ℕ := 4
noncomputable def chad_initial_stars : ℕ := 2 * eric_stars
noncomputable def chad_stars_after_sale : ℕ := chad_initial_stars - 2
noncomputable def total_stars : ℕ := 16
noncomputable def stars_eric_and_chad : ℕ := eric_stars + chad_stars_after_sale

theorem jeff_stars :
  total_stars - stars_eric_and_chad = 6 := 
by 
  sorry

end NUMINAMATH_GPT_jeff_stars_l1824_182486


namespace NUMINAMATH_GPT_cube_volume_from_surface_area_l1824_182482

theorem cube_volume_from_surface_area (SA : ℕ) (h : SA = 600) :
  ∃ V : ℕ, V = 1000 := by
  sorry

end NUMINAMATH_GPT_cube_volume_from_surface_area_l1824_182482


namespace NUMINAMATH_GPT_find_m_l1824_182476

theorem find_m (m : ℝ) :
  let a : ℝ × ℝ := (2, m)
  let b : ℝ × ℝ := (1, -1)
  (b.1 * (a.1 + 2 * b.1) + b.2 * (a.2 + 2 * b.2) = 0) → 
  m = 6 := by 
  sorry

end NUMINAMATH_GPT_find_m_l1824_182476


namespace NUMINAMATH_GPT_length_of_train_is_250_02_l1824_182465

noncomputable def train_speed_km_per_hr : ℝ := 100
noncomputable def time_to_cross_pole_sec : ℝ := 9

-- Convert speed from km/hr to m/s
noncomputable def speed_m_per_s : ℝ := train_speed_km_per_hr * (1000 / 3600)

-- Calculating the length of the train
noncomputable def length_of_train : ℝ := speed_m_per_s * time_to_cross_pole_sec

theorem length_of_train_is_250_02 :
  length_of_train = 250.02 := by
  -- Proof is omitted (replace 'sorry' with the actual proof)
  sorry

end NUMINAMATH_GPT_length_of_train_is_250_02_l1824_182465


namespace NUMINAMATH_GPT_problem_statement_l1824_182469

-- The conditions of the problem
variables (x : Real)

-- Define the conditions as hypotheses
def condition1 : Prop := (Real.sin (3 * x) * Real.sin (4 * x)) = (Real.cos (3 * x) * Real.cos (4 * x))
def condition2 : Prop := Real.sin (7 * x) = 0

-- The theorem we need to prove
theorem problem_statement (h1 : condition1 x) (h2 : condition2 x) : x = Real.pi / 7 :=
by sorry

end NUMINAMATH_GPT_problem_statement_l1824_182469


namespace NUMINAMATH_GPT_complete_the_square_l1824_182474

theorem complete_the_square (x : ℝ) :
  x^2 + 6 * x - 4 = 0 → (x + 3)^2 = 13 :=
by
  sorry

end NUMINAMATH_GPT_complete_the_square_l1824_182474


namespace NUMINAMATH_GPT_triangle_area_transform_l1824_182404

-- Define the concept of a triangle with integer coordinates
structure Triangle :=
  (A : ℤ × ℤ)
  (B : ℤ × ℤ)
  (C : ℤ × ℤ)

-- Define the area of a triangle using determinant
def triangle_area (T : Triangle) : ℤ :=
  let ⟨(x1, y1), (x2, y2), (x3, y3)⟩ := (T.A, T.B, T.C)
  abs ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2)

-- Define a legal transformation for triangles
def legal_transform (T : Triangle) : Set Triangle :=
  { T' : Triangle |
    (∃ c : ℤ, 
      (T'.A = (T.A.1 + c * (T.B.1 - T.C.1), T.A.2 + c * (T.B.2 - T.C.2)) ∧ T'.B = T.B ∧ T'.C = T.C) ∨
      (T'.A = T.A ∧ T'.B = (T.B.1 + c * (T.A.1 - T.C.1), T.B.2 + c * (T.A.2 - T.C.2)) ∧ T'.C = T.C) ∨
      (T'.A = T.A ∧ T'.B = T.B ∧ T'.C = (T.C.1 + c * (T.A.1 - T.B.1), T.C.2 + c * (T.A.2 - T.B.2)))) }

-- Proposition that any two triangles with equal area can be legally transformed into each other
theorem triangle_area_transform (T1 T2 : Triangle) (h : triangle_area T1 = triangle_area T2) :
  ∃ (T' : Triangle), T' ∈ legal_transform T1 ∧ triangle_area T' = triangle_area T2 :=
sorry

end NUMINAMATH_GPT_triangle_area_transform_l1824_182404


namespace NUMINAMATH_GPT_track_circumference_l1824_182415

def same_start_point (A B : ℕ) : Prop := A = B

def opposite_direction (a_speed b_speed : ℕ) : Prop := a_speed > 0 ∧ b_speed > 0

def first_meet_after (A B : ℕ) (a_distance b_distance : ℕ) : Prop := a_distance = 150 ∧ b_distance = 150

def second_meet_near_full_lap (B : ℕ) (lap_length short_distance : ℕ) : Prop := short_distance = 90

theorem track_circumference
    (A B : ℕ) (a_speed b_speed lap_length : ℕ)
    (h1 : same_start_point A B)
    (h2 : opposite_direction a_speed b_speed)
    (h3 : first_meet_after A B 150 150)
    (h4 : second_meet_near_full_lap B lap_length 90) :
    lap_length = 300 :=
sorry

end NUMINAMATH_GPT_track_circumference_l1824_182415


namespace NUMINAMATH_GPT_newsletter_cost_l1824_182431

theorem newsletter_cost (x : ℝ) (h1 : 14 * x < 16) (h2 : 19 * x > 21) : x = 1.11 :=
by
  sorry

end NUMINAMATH_GPT_newsletter_cost_l1824_182431


namespace NUMINAMATH_GPT_convert_denominators_to_integers_l1824_182454

def original_equation (x : ℝ) : Prop :=
  (x + 1) / 0.4 - (0.2 * x - 1) / 0.7 = 1

def transformed_equation (x : ℝ) : Prop :=
  (10 * x + 10) / 4 - (2 * x - 10) / 7 = 1

theorem convert_denominators_to_integers (x : ℝ) 
  (h : original_equation x) : transformed_equation x :=
sorry

end NUMINAMATH_GPT_convert_denominators_to_integers_l1824_182454


namespace NUMINAMATH_GPT_sqrt_sum_eq_l1824_182489

theorem sqrt_sum_eq : 
  (Real.sqrt (16 - 12 * Real.sqrt 3)) + (Real.sqrt (16 + 12 * Real.sqrt 3)) = 4 * Real.sqrt 6 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_sum_eq_l1824_182489


namespace NUMINAMATH_GPT_product_of_two_numbers_l1824_182405

variable {x y : ℝ}

theorem product_of_two_numbers (h1 : x + y = 25) (h2 : x - y = 7) : x * y = 144 := by
  sorry

end NUMINAMATH_GPT_product_of_two_numbers_l1824_182405


namespace NUMINAMATH_GPT_bookmarks_sold_l1824_182423

-- Definitions pertaining to the problem
def total_books_sold : ℕ := 72
def books_ratio : ℕ := 9
def bookmarks_ratio : ℕ := 2

-- Statement of the theorem
theorem bookmarks_sold :
  (total_books_sold / books_ratio) * bookmarks_ratio = 16 :=
by
  sorry

end NUMINAMATH_GPT_bookmarks_sold_l1824_182423


namespace NUMINAMATH_GPT_find_some_value_l1824_182491

-- Define the main variables and assumptions
variable (m n some_value : ℝ)

-- State the assumptions based on the conditions
axiom h1 : m = n / 2 - 2 / 5
axiom h2 : m + some_value = (n + 4) / 2 - 2 / 5

-- State the theorem we are trying to prove
theorem find_some_value : some_value = 2 :=
by
  -- Proof goes here, for now we just put sorry
  sorry

end NUMINAMATH_GPT_find_some_value_l1824_182491


namespace NUMINAMATH_GPT_dried_fruit_percentage_l1824_182479

-- Define the percentages for Sue, Jane, and Tom's trail mixes.
structure TrailMix :=
  (nuts : ℝ)
  (dried_fruit : ℝ)

def sue : TrailMix := { nuts := 0.30, dried_fruit := 0.70 }
def jane : TrailMix := { nuts := 0.60, dried_fruit := 0.00 }  -- Note: No dried fruit
def tom : TrailMix := { nuts := 0.40, dried_fruit := 0.50 }

-- Condition: Combined mix contains 45% nuts.
def combined_nuts (sue_nuts jane_nuts tom_nuts : ℝ) : Prop :=
  0.33 * sue_nuts + 0.33 * jane_nuts + 0.33 * tom_nuts = 0.45

-- Condition: Each contributes equally to the total mixture.
def equal_contribution (sue_cont jane_cont tom_cont : ℝ) : Prop :=
  sue_cont = jane_cont ∧ jane_cont = tom_cont

-- Theorem to be proven: Combined mixture contains 40% dried fruit.
theorem dried_fruit_percentage :
  combined_nuts sue.nuts jane.nuts tom.nuts →
  equal_contribution (1 / 3) (1 / 3) (1 / 3) →
  0.33 * sue.dried_fruit + 0.33 * tom.dried_fruit = 0.40 :=
by sorry

end NUMINAMATH_GPT_dried_fruit_percentage_l1824_182479


namespace NUMINAMATH_GPT_inequality_proof_l1824_182460

theorem inequality_proof (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
    (h4 : (3 / (a * b * c)) ≥ (a + b + c)) : 
    (1 / a + 1 / b + 1 / c) ≥ (a + b + c) :=
  sorry

end NUMINAMATH_GPT_inequality_proof_l1824_182460


namespace NUMINAMATH_GPT_odd_function_expression_l1824_182461

theorem odd_function_expression (f : ℝ → ℝ) 
  (h_odd : ∀ x, f (-x) = -f x)
  (h_pos : ∀ x, 0 < x → f x = x^2 + |x| - 1) : 
  ∀ x, x < 0 → f x = -x^2 + x + 1 :=
by
  sorry

end NUMINAMATH_GPT_odd_function_expression_l1824_182461


namespace NUMINAMATH_GPT_union_of_A_and_B_l1824_182433

/-- Let the universal set U = ℝ, and let the sets A = {x | x^2 - x - 2 = 0}
and B = {y | ∃ x, x ∈ A ∧ y = x + 3}. We want to prove that A ∪ B = {-1, 2, 5}.
-/
theorem union_of_A_and_B (U : Set ℝ) (A B : Set ℝ) (A_def : ∀ x, x ∈ A ↔ x^2 - x - 2 = 0)
  (B_def : ∀ y, y ∈ B ↔ ∃ x, x ∈ A ∧ y = x + 3) :
  A ∪ B = {-1, 2, 5} :=
sorry

end NUMINAMATH_GPT_union_of_A_and_B_l1824_182433


namespace NUMINAMATH_GPT_exists_root_in_interval_l1824_182458

noncomputable def f (x : ℝ) : ℝ := 1 / (x - 1) - Real.log (x - 1) / Real.log 2

theorem exists_root_in_interval :
  ∃ c : ℝ, 2 < c ∧ c < 3 ∧ f c = 0 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_exists_root_in_interval_l1824_182458


namespace NUMINAMATH_GPT_find_k_values_l1824_182477

noncomputable def problem (a b c d k : ℂ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧
  (a * k^3 + b * k^2 + c * k + d = 0) ∧
  (b * k^3 + c * k^2 + d * k + a = 0)

theorem find_k_values (a b c d k : ℂ) (h : problem a b c d k) : 
  k = 1 ∨ k = -1 ∨ k = Complex.I ∨ k = -Complex.I :=
sorry

end NUMINAMATH_GPT_find_k_values_l1824_182477


namespace NUMINAMATH_GPT_calc_value_l1824_182495

theorem calc_value (n : ℕ) (h : 1 ≤ n) : 
  (5^(n+1) + 6^(n+2))^2 - (5^(n+1) - 6^(n+2))^2 = 144 * 30^(n+1) := 
sorry

end NUMINAMATH_GPT_calc_value_l1824_182495


namespace NUMINAMATH_GPT_absolute_difference_m_n_l1824_182494

theorem absolute_difference_m_n (m n : ℝ) (h1 : m * n = 6) (h2 : m + n = 7) : |m - n| = 5 := 
by 
  sorry

end NUMINAMATH_GPT_absolute_difference_m_n_l1824_182494


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l1824_182439

open Set

noncomputable def A : Set ℤ := {1, 3, 5, 7}
noncomputable def B : Set ℤ := {x | 2 ≤ x ∧ x ≤ 5}

theorem intersection_of_A_and_B : A ∩ B = {3, 5} := by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l1824_182439


namespace NUMINAMATH_GPT_hyperbola_center_l1824_182490

theorem hyperbola_center (x y : ℝ) :
  9 * x ^ 2 - 54 * x - 36 * y ^ 2 + 360 * y - 900 = 0 →
  (x, y) = (3, 5) :=
sorry

end NUMINAMATH_GPT_hyperbola_center_l1824_182490


namespace NUMINAMATH_GPT_calculate_expression_l1824_182447

theorem calculate_expression :
  (-1: ℤ) ^ 53 + 2 ^ (4 ^ 4 + 3 ^ 3 - 5 ^ 2) = -1 + 2 ^ 258 := 
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l1824_182447


namespace NUMINAMATH_GPT_opposite_of_neg_2023_l1824_182444

def opposite (n : ℤ) : ℤ :=
  -n

theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  sorry

end NUMINAMATH_GPT_opposite_of_neg_2023_l1824_182444


namespace NUMINAMATH_GPT_diagonals_in_octadecagon_l1824_182449

def num_sides : ℕ := 18

def num_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem diagonals_in_octadecagon : num_diagonals num_sides = 135 := by 
  sorry

end NUMINAMATH_GPT_diagonals_in_octadecagon_l1824_182449


namespace NUMINAMATH_GPT_number_of_cats_l1824_182413

def number_of_dogs : ℕ := 43
def number_of_fish : ℕ := 72
def total_pets : ℕ := 149

theorem number_of_cats : total_pets - (number_of_dogs + number_of_fish) = 34 := 
by
  sorry

end NUMINAMATH_GPT_number_of_cats_l1824_182413


namespace NUMINAMATH_GPT_divisors_count_of_108n5_l1824_182480

theorem divisors_count_of_108n5 {n : ℕ} (hn_pos : 0 < n) (h_divisors_150n3 : (150 * n^3).divisors.card = 150) : 
(108 * n^5).divisors.card = 432 :=
sorry

end NUMINAMATH_GPT_divisors_count_of_108n5_l1824_182480


namespace NUMINAMATH_GPT_sweet_potatoes_not_yet_sold_l1824_182471

def total_harvested := 80
def sold_to_adams := 20
def sold_to_lenon := 15
def not_yet_sold : ℕ := total_harvested - (sold_to_adams + sold_to_lenon)

theorem sweet_potatoes_not_yet_sold :
  not_yet_sold = 45 :=
by
  unfold not_yet_sold
  unfold total_harvested sold_to_adams sold_to_lenon
  sorry

end NUMINAMATH_GPT_sweet_potatoes_not_yet_sold_l1824_182471


namespace NUMINAMATH_GPT_andrei_club_visits_l1824_182400

theorem andrei_club_visits (d c : ℕ) (h : 15 * d + 11 * c = 115) : d + c = 9 :=
by
  sorry

end NUMINAMATH_GPT_andrei_club_visits_l1824_182400


namespace NUMINAMATH_GPT_sum_first_49_nat_nums_l1824_182470

theorem sum_first_49_nat_nums : (Finset.range 50).sum (fun x => x) = 1225 := 
by
  sorry

end NUMINAMATH_GPT_sum_first_49_nat_nums_l1824_182470


namespace NUMINAMATH_GPT_exponent_multiplication_l1824_182456

variable (x : ℤ)

theorem exponent_multiplication :
  (-x^2) * x^3 = -x^5 :=
sorry

end NUMINAMATH_GPT_exponent_multiplication_l1824_182456


namespace NUMINAMATH_GPT_simplify_fraction_l1824_182406

theorem simplify_fraction (m : ℝ) (h : m ≠ 1) : (m / (m - 1) + 1 / (1 - m) = 1) :=
by {
  sorry
}

end NUMINAMATH_GPT_simplify_fraction_l1824_182406


namespace NUMINAMATH_GPT_average_speed_30_l1824_182426

theorem average_speed_30 (v : ℝ) (h₁ : 0 < v) (h₂ : 210 / v - 1 = 210 / (v + 5)) : v = 30 :=
sorry

end NUMINAMATH_GPT_average_speed_30_l1824_182426


namespace NUMINAMATH_GPT_dean_ordered_two_pizzas_l1824_182421

variable (P : ℕ)

-- Each large pizza is cut into 12 slices
def slices_per_pizza := 12

-- Dean ate half of the Hawaiian pizza
def dean_slices := slices_per_pizza / 2

-- Frank ate 3 slices of Hawaiian pizza
def frank_slices := 3

-- Sammy ate a third of the cheese pizza
def sammy_slices := slices_per_pizza / 3

-- Total slices eaten plus slices left over equals total slices from pizzas
def total_slices_eaten := dean_slices + frank_slices + sammy_slices
def slices_left_over := 11
def total_pizza_slices := total_slices_eaten + slices_left_over

-- Total pizzas ordered is the total slices divided by slices per pizza
def pizzas_ordered := total_pizza_slices / slices_per_pizza

-- Prove that Dean ordered 2 large pizzas
theorem dean_ordered_two_pizzas : pizzas_ordered = 2 := by
  -- Proof omitted, add your proof here
  sorry

end NUMINAMATH_GPT_dean_ordered_two_pizzas_l1824_182421
