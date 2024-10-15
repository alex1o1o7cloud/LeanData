import Mathlib

namespace NUMINAMATH_GPT_focus_of_parabola_l454_45486

theorem focus_of_parabola (x y : ℝ) : (y^2 = 4 * x) → (x = 2 ∧ y = 0) :=
by
  sorry

end NUMINAMATH_GPT_focus_of_parabola_l454_45486


namespace NUMINAMATH_GPT_cos_plus_2sin_eq_one_l454_45479

theorem cos_plus_2sin_eq_one (α : ℝ) (h : (1 + Real.cos α) / Real.sin α = 1 / 2) : 
  Real.cos α + 2 * Real.sin α = 1 := 
by
  sorry

end NUMINAMATH_GPT_cos_plus_2sin_eq_one_l454_45479


namespace NUMINAMATH_GPT_find_first_term_l454_45400

open Int

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem find_first_term
  (a : ℕ → ℤ)
  (d : ℤ)
  (h_seq : arithmetic_sequence a)
  (h_a3 : a 2 = 1)
  (h_a4_a10 : a 3 + a 9 = 18) :
  a 0 = -3 :=
by
  sorry

end NUMINAMATH_GPT_find_first_term_l454_45400


namespace NUMINAMATH_GPT_proof_star_ast_l454_45439

noncomputable def star (a b : ℕ) : ℕ := sorry  -- representing binary operation for star
noncomputable def ast (a b : ℕ) : ℕ := sorry  -- representing binary operation for ast

theorem proof_star_ast :
  star 12 2 * ast 9 3 = 2 →
  (star 7 3 * ast 12 6) = 7 / 6 :=
by
  sorry

end NUMINAMATH_GPT_proof_star_ast_l454_45439


namespace NUMINAMATH_GPT_trajectory_of_P_eqn_l454_45415

theorem trajectory_of_P_eqn :
  ∀ {x y : ℝ}, -- For all real numbers x and y
  (-(x + 2)^2 + (x - 1)^2 + y^2 = 3*((x - 1)^2 + y^2)) → -- Condition |PA| = 2|PB|
  (x^2 + y^2 - 4*x = 0) := -- Prove the trajectory equation
by
  intros x y h
  sorry -- Proof to be completed

end NUMINAMATH_GPT_trajectory_of_P_eqn_l454_45415


namespace NUMINAMATH_GPT_tuples_and_triples_counts_are_equal_l454_45474

theorem tuples_and_triples_counts_are_equal (n : ℕ) (h : n > 0) :
  let countTuples := 8^n - 2 * 7^n + 6^n
  let countTriples := 8^n - 2 * 7^n + 6^n
  countTuples = countTriples :=
by
  sorry

end NUMINAMATH_GPT_tuples_and_triples_counts_are_equal_l454_45474


namespace NUMINAMATH_GPT_koby_boxes_l454_45450

theorem koby_boxes (x : ℕ) (sparklers_per_box : ℕ := 3) (whistlers_per_box : ℕ := 5) 
    (cherie_sparklers : ℕ := 8) (cherie_whistlers : ℕ := 9) (total_fireworks : ℕ := 33) : 
    (sparklers_per_box * x + cherie_sparklers) + (whistlers_per_box * x + cherie_whistlers) = total_fireworks → x = 2 :=
by
  sorry

end NUMINAMATH_GPT_koby_boxes_l454_45450


namespace NUMINAMATH_GPT_max_sector_area_l454_45404

theorem max_sector_area (r θ : ℝ) (h₁ : 2 * r + r * θ = 16) : 
  (∃ A : ℝ, A = 1/2 * r^2 * θ ∧ A ≤ 16) ∧ (∃ r θ, r = 4 ∧ θ = 2 ∧ 1/2 * r^2 * θ = 16) := 
by
  sorry

end NUMINAMATH_GPT_max_sector_area_l454_45404


namespace NUMINAMATH_GPT_farmer_harvest_correct_l454_45430

-- Define the conditions
def estimated_harvest : ℕ := 48097
def additional_harvest : ℕ := 684
def total_harvest : ℕ := 48781

-- The proof statement
theorem farmer_harvest_correct :
  estimated_harvest + additional_harvest = total_harvest :=
by
  sorry

end NUMINAMATH_GPT_farmer_harvest_correct_l454_45430


namespace NUMINAMATH_GPT_problem_m_n_sum_l454_45406

theorem problem_m_n_sum (m n : ℕ) 
  (h1 : m^2 + n^2 = 3789) 
  (h2 : Nat.gcd m n + Nat.lcm m n = 633) : 
  m + n = 87 :=
sorry

end NUMINAMATH_GPT_problem_m_n_sum_l454_45406


namespace NUMINAMATH_GPT_second_statue_weight_l454_45464

theorem second_statue_weight (S : ℕ) :
  ∃ S : ℕ,
    (80 = 10 + S + 15 + 15 + 22) → S = 18 :=
by
  sorry

end NUMINAMATH_GPT_second_statue_weight_l454_45464


namespace NUMINAMATH_GPT_min_value_of_y_l454_45455

theorem min_value_of_y (x : ℝ) (h : x > 3) : y = x + 1/(x-3) → y ≥ 5 :=
sorry

end NUMINAMATH_GPT_min_value_of_y_l454_45455


namespace NUMINAMATH_GPT_binary_equals_octal_l454_45498

-- Define that 1001101 in binary is a specific integer
def binary_value : ℕ := 0b1001101

-- Define that 115 in octal is a specific integer
def octal_value : ℕ := 0o115

-- State the theorem we need to prove
theorem binary_equals_octal : binary_value = octal_value :=
  by sorry

end NUMINAMATH_GPT_binary_equals_octal_l454_45498


namespace NUMINAMATH_GPT_cube_face_sum_l454_45409

theorem cube_face_sum (a b c d e f : ℕ) (h1 : e = b) (h2 : 2 * (a * b * c + a * b * f + d * b * c + d * b * f) = 1332) :
  a + b + c + d + e + f = 47 :=
sorry

end NUMINAMATH_GPT_cube_face_sum_l454_45409


namespace NUMINAMATH_GPT_product_of_numbers_l454_45468

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 22) (h2 : x^2 + y^2 = 460) : x * y = 40 := 
by 
  sorry

end NUMINAMATH_GPT_product_of_numbers_l454_45468


namespace NUMINAMATH_GPT_option1_cost_expression_option2_cost_expression_cost_comparison_x_20_more_cost_effective_strategy_cost_x_20_l454_45459

def teapot_price : ℕ := 20
def teacup_price : ℕ := 6
def discount_rate : ℝ := 0.9

def option1_cost (x : ℕ) : ℕ :=
  5 * teapot_price + (x - 5) * teacup_price

def option2_cost (x : ℕ) : ℝ :=
  discount_rate * (5 * teapot_price + x * teacup_price)

theorem option1_cost_expression (x : ℕ) (h : x > 5) : option1_cost x = 6 * x + 70 := by
  sorry

theorem option2_cost_expression (x : ℕ) (h : x > 5) : option2_cost x = 5.4 * x + 90 := by
  sorry

theorem cost_comparison_x_20 : option1_cost 20 < option2_cost 20 := by
  sorry

theorem more_cost_effective_strategy_cost_x_20 : (5 * teapot_price + 15 * teacup_price * discount_rate) = 181 := by
  sorry

end NUMINAMATH_GPT_option1_cost_expression_option2_cost_expression_cost_comparison_x_20_more_cost_effective_strategy_cost_x_20_l454_45459


namespace NUMINAMATH_GPT_sum_of_ages_l454_45463

theorem sum_of_ages (juliet_age maggie_age ralph_age nicky_age : ℕ)
  (h1 : juliet_age = 10)
  (h2 : juliet_age = maggie_age + 3)
  (h3 : ralph_age = juliet_age + 2)
  (h4 : nicky_age = ralph_age / 2) :
  maggie_age + ralph_age + nicky_age = 25 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_ages_l454_45463


namespace NUMINAMATH_GPT_sum_of_first_four_terms_l454_45421

def arithmetic_sequence_sum (a1 a2 : ℕ) (n : ℕ) : ℕ :=
  (n * (2 * a1 + (n - 1) * (a2 - a1))) / 2

theorem sum_of_first_four_terms : arithmetic_sequence_sum 4 6 4 = 28 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_first_four_terms_l454_45421


namespace NUMINAMATH_GPT_price_of_fruit_l454_45407

theorem price_of_fruit
  (price_milk_per_liter : ℝ)
  (milk_per_batch : ℝ)
  (fruit_per_batch : ℝ)
  (cost_for_three_batches : ℝ)
  (F : ℝ)
  (h1 : price_milk_per_liter = 1.5)
  (h2 : milk_per_batch = 10)
  (h3 : fruit_per_batch = 3)
  (h4 : cost_for_three_batches = 63)
  (h5 : 3 * (milk_per_batch * price_milk_per_liter + fruit_per_batch * F) = cost_for_three_batches) :
  F = 2 :=
by sorry

end NUMINAMATH_GPT_price_of_fruit_l454_45407


namespace NUMINAMATH_GPT_value_of_expression_l454_45483

theorem value_of_expression : 3 ^ (0 ^ (2 ^ 11)) + ((3 ^ 0) ^ 2) ^ 11 = 2 := by
  sorry

end NUMINAMATH_GPT_value_of_expression_l454_45483


namespace NUMINAMATH_GPT_modified_counting_game_53rd_term_l454_45442

theorem modified_counting_game_53rd_term :
  let a : ℕ := 1
  let d : ℕ := 2
  a + (53 - 1) * d = 105 :=
by 
  sorry

end NUMINAMATH_GPT_modified_counting_game_53rd_term_l454_45442


namespace NUMINAMATH_GPT_sqrt_log_equality_l454_45441

noncomputable def log4 (x : ℝ) : ℝ := Real.log x / Real.log 4
noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

theorem sqrt_log_equality {x y : ℝ} (hx : x > 0) (hy : y > 0) :
    Real.sqrt (log4 x + 2 * log2 y) = Real.sqrt (log2 (x * y^2)) / Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_sqrt_log_equality_l454_45441


namespace NUMINAMATH_GPT_max_A_excircle_area_ratio_max_A_excircle_area_ratio_eq_l454_45496

noncomputable def A_excircle_area_ratio (α : Real) (s : Real) : Real :=
  0.5 * Real.sin α

theorem max_A_excircle_area_ratio (α : Real) (s : Real) : (A_excircle_area_ratio α s) ≤ 0.5 :=
by
  sorry

theorem max_A_excircle_area_ratio_eq (s : Real) : 
  (A_excircle_area_ratio (Real.pi / 2) s) = 0.5 :=
by
  sorry

end NUMINAMATH_GPT_max_A_excircle_area_ratio_max_A_excircle_area_ratio_eq_l454_45496


namespace NUMINAMATH_GPT_not_in_range_l454_45434

noncomputable def g (x c: ℝ) : ℝ := x^2 + c * x + 5

theorem not_in_range (c : ℝ) (hc : -2 * Real.sqrt 2 < c ∧ c < 2 * Real.sqrt 2) :
  ∀ x : ℝ, g x c ≠ 3 :=
by
  intros
  sorry

end NUMINAMATH_GPT_not_in_range_l454_45434


namespace NUMINAMATH_GPT_exists_strictly_positive_c_l454_45458

theorem exists_strictly_positive_c {a : ℕ → ℕ → ℝ} (h_diag_pos : ∀ i, a i i > 0)
  (h_off_diag_neg : ∀ i j, i ≠ j → a i j < 0) :
  ∃ (c : ℕ → ℝ), (∀ i, 
    0 < c i) ∧ 
    ((∀ k, a k 1 * c 1 + a k 2 * c 2 + a k 3 * c 3 > 0) ∨ 
     (∀ k, a k 1 * c 1 + a k 2 * c 2 + a k 3 * c 3 < 0) ∨ 
     (∀ k, a k 1 * c 1 + a k 2 * c 2 + a k 3 * c 3 = 0)) :=
by
  sorry

end NUMINAMATH_GPT_exists_strictly_positive_c_l454_45458


namespace NUMINAMATH_GPT_quadratic_rewrite_h_l454_45480

theorem quadratic_rewrite_h (a k h x : ℝ) :
  (3 * x^2 + 9 * x + 17) = a * (x - h)^2 + k ↔ h = -3/2 :=
by sorry

end NUMINAMATH_GPT_quadratic_rewrite_h_l454_45480


namespace NUMINAMATH_GPT_solve_quadratic_eq_l454_45471

theorem solve_quadratic_eq (x : ℝ) : x^2 - 2 * x - 15 = 0 ↔ (x = -3 ∨ x = 5) :=
by
  sorry

end NUMINAMATH_GPT_solve_quadratic_eq_l454_45471


namespace NUMINAMATH_GPT_value_of_y_l454_45416

noncomputable def k : ℝ := 168.75

theorem value_of_y (x y : ℝ) (h1 : x * y = k) (h2 : x + y = 30) (h3 : x = 3 * y) : y = -16.875 :=
by 
  sorry

end NUMINAMATH_GPT_value_of_y_l454_45416


namespace NUMINAMATH_GPT_problem1_problem2_l454_45423

variables {a x y : ℝ}

theorem problem1 (h1 : a^x = 2) (h2 : a^y = 3) : a^(x + y) = 6 :=
sorry

theorem problem2 (h1 : a^x = 2) (h2 : a^y = 3) : a^(2 * x - 3 * y) = 4 / 27 :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l454_45423


namespace NUMINAMATH_GPT_cassy_initial_jars_l454_45475

theorem cassy_initial_jars (boxes1 jars1 boxes2 jars2 leftover: ℕ) (h1: boxes1 = 10) (h2: jars1 = 12) (h3: boxes2 = 30) (h4: jars2 = 10) (h5: leftover = 80) : 
  boxes1 * jars1 + boxes2 * jars2 + leftover = 500 := 
by 
  sorry

end NUMINAMATH_GPT_cassy_initial_jars_l454_45475


namespace NUMINAMATH_GPT_complement_intersection_l454_45485

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {3, 4, 5}

theorem complement_intersection (U A B : Set ℕ) (hU : U = {1, 2, 3, 4, 5}) (hA : A = {1, 2, 3}) (hB : B = {3, 4, 5}) :
  U \ (A ∩ B) = {1, 2, 4, 5} :=
by
  sorry

end NUMINAMATH_GPT_complement_intersection_l454_45485


namespace NUMINAMATH_GPT_no_rational_roots_l454_45456

theorem no_rational_roots : ¬ ∃ x : ℚ, 5 * x^3 - 4 * x^2 - 8 * x + 3 = 0 :=
by
  sorry

end NUMINAMATH_GPT_no_rational_roots_l454_45456


namespace NUMINAMATH_GPT_min_value_of_quadratic_l454_45467

theorem min_value_of_quadratic : ∀ x : ℝ, ∃ y : ℝ, y = (x - 1)^2 - 3 ∧ (∀ z : ℝ, (z - 1)^2 - 3 ≥ y) :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_quadratic_l454_45467


namespace NUMINAMATH_GPT_circle_equation_tangent_line_l454_45426

theorem circle_equation_tangent_line :
  ∃ r : ℝ, ∀ x y : ℝ, (x - 3)^2 + (y + 5)^2 = r^2 ↔ x - 7 * y + 2 = 0 :=
sorry

end NUMINAMATH_GPT_circle_equation_tangent_line_l454_45426


namespace NUMINAMATH_GPT_brad_more_pages_than_greg_l454_45473

def greg_pages_first_week : ℕ := 7 * 18
def greg_pages_next_two_weeks : ℕ := 14 * 22
def greg_total_pages : ℕ := greg_pages_first_week + greg_pages_next_two_weeks

def brad_pages_first_5_days : ℕ := 5 * 26
def brad_pages_remaining_12_days : ℕ := 12 * 20
def brad_total_pages : ℕ := brad_pages_first_5_days + brad_pages_remaining_12_days

def total_required_pages : ℕ := 800

theorem brad_more_pages_than_greg : brad_total_pages - greg_total_pages = 64 :=
by
  sorry

end NUMINAMATH_GPT_brad_more_pages_than_greg_l454_45473


namespace NUMINAMATH_GPT_quadratic_distinct_roots_iff_m_lt_four_l454_45428

theorem quadratic_distinct_roots_iff_m_lt_four (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x₁^2 - 4 * x₁ + m = 0) ∧ (x₂^2 - 4 * x₂ + m = 0)) ↔ m < 4 :=
by sorry

end NUMINAMATH_GPT_quadratic_distinct_roots_iff_m_lt_four_l454_45428


namespace NUMINAMATH_GPT_sixth_term_geometric_sequence_l454_45417

theorem sixth_term_geometric_sequence (a r : ℚ) (h_a : a = 16) (h_r : r = 1/2) : 
  a * r^(5) = 1/2 :=
by 
  rw [h_a, h_r]
  sorry

end NUMINAMATH_GPT_sixth_term_geometric_sequence_l454_45417


namespace NUMINAMATH_GPT_virginia_eggs_l454_45454

-- Definitions and conditions
variable (eggs_start : Nat)
variable (eggs_taken : Nat := 3)
variable (eggs_end : Nat := 93)

-- Problem statement to prove
theorem virginia_eggs : eggs_start - eggs_taken = eggs_end → eggs_start = 96 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_virginia_eggs_l454_45454


namespace NUMINAMATH_GPT_house_construction_days_l454_45490

theorem house_construction_days
  (D : ℕ) -- number of planned days to build the house
  (Hwork_done : 1000 + 200 * (D - 10) = 100 * (D + 90)) : 
  D = 110 :=
sorry

end NUMINAMATH_GPT_house_construction_days_l454_45490


namespace NUMINAMATH_GPT_cheryl_more_eggs_than_others_l454_45461

def kevin_eggs : ℕ := 5
def bonnie_eggs : ℕ := 13
def george_eggs : ℕ := 9
def cheryl_eggs : ℕ := 56

theorem cheryl_more_eggs_than_others : cheryl_eggs - (kevin_eggs + bonnie_eggs + george_eggs) = 29 :=
by
  sorry

end NUMINAMATH_GPT_cheryl_more_eggs_than_others_l454_45461


namespace NUMINAMATH_GPT_converse_and_inverse_false_l454_45481

-- Define the property of being a rhombus and a parallelogram
def is_rhombus (R : Type) : Prop := sorry
def is_parallelogram (P : Type) : Prop := sorry

-- Given: If a quadrilateral is a rhombus, then it is a parallelogram
def quad_imp (Q : Type) : Prop := is_rhombus Q → is_parallelogram Q

-- Prove that the converse and inverse are false
theorem converse_and_inverse_false (Q : Type) 
  (h1 : quad_imp Q) : 
  ¬(is_parallelogram Q → is_rhombus Q) ∧ ¬(¬(is_rhombus Q) → ¬(is_parallelogram Q)) :=
by
  sorry

end NUMINAMATH_GPT_converse_and_inverse_false_l454_45481


namespace NUMINAMATH_GPT_max_gcd_coprime_l454_45492

theorem max_gcd_coprime (x y : ℤ) (h : Int.gcd x y = 1) : 
  Int.gcd (x + 2015 * y) (y + 2015 * x) ≤ 4060224 :=
sorry

end NUMINAMATH_GPT_max_gcd_coprime_l454_45492


namespace NUMINAMATH_GPT_unattainable_y_l454_45487

theorem unattainable_y (x : ℝ) (h : x ≠ -5/4) : ¬∃ y : ℝ, y = (2 - 3 * x) / (4 * x + 5) ∧ y = -3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_unattainable_y_l454_45487


namespace NUMINAMATH_GPT_initial_shirts_count_l454_45414

theorem initial_shirts_count 
  (S T x : ℝ)
  (h1 : 2 * S + x * T = 1600)
  (h2 : S + 6 * T = 1600)
  (h3 : 12 * T = 2400) :
  x = 4 :=
by
  sorry

end NUMINAMATH_GPT_initial_shirts_count_l454_45414


namespace NUMINAMATH_GPT_mean_of_added_numbers_l454_45499

theorem mean_of_added_numbers (mean_seven : ℝ) (mean_ten : ℝ) (x y z : ℝ)
    (h1 : mean_seven = 40)
    (h2 : mean_ten = 55) :
    (mean_seven * 7 + x + y + z) / 10 = mean_ten → (x + y + z) / 3 = 90 :=
by sorry

end NUMINAMATH_GPT_mean_of_added_numbers_l454_45499


namespace NUMINAMATH_GPT_clara_total_cookies_l454_45437

theorem clara_total_cookies :
  let cookies_per_box1 := 12
  let cookies_per_box2 := 20
  let cookies_per_box3 := 16
  let boxes_sold1 := 50
  let boxes_sold2 := 80
  let boxes_sold3 := 70
  (boxes_sold1 * cookies_per_box1 + boxes_sold2 * cookies_per_box2 + boxes_sold3 * cookies_per_box3) = 3320 :=
by
  sorry

end NUMINAMATH_GPT_clara_total_cookies_l454_45437


namespace NUMINAMATH_GPT_rhombus_compression_problem_l454_45482

def rhombus_diagonal_lengths (side longer_diagonal : ℝ) (compression : ℝ) : ℝ × ℝ :=
  let new_longer_diagonal := longer_diagonal - compression
  let new_shorter_diagonal := 1.2 * compression + 24
  (new_longer_diagonal, new_shorter_diagonal)

theorem rhombus_compression_problem :
  let side := 20
  let longer_diagonal := 32
  let compression := 2.62
  rhombus_diagonal_lengths side longer_diagonal compression = (29.38, 27.14) :=
by sorry

end NUMINAMATH_GPT_rhombus_compression_problem_l454_45482


namespace NUMINAMATH_GPT_tanks_difference_l454_45469

theorem tanks_difference (total_tanks german_tanks allied_tanks sanchalian_tanks : ℕ)
  (h_total : total_tanks = 115)
  (h_german_allied : german_tanks = 2 * allied_tanks + 2)
  (h_allied_sanchalian : allied_tanks = 3 * sanchalian_tanks + 1)
  (h_total_eq : german_tanks + allied_tanks + sanchalian_tanks = total_tanks) :
  german_tanks - sanchalian_tanks = 59 :=
sorry

end NUMINAMATH_GPT_tanks_difference_l454_45469


namespace NUMINAMATH_GPT_interest_rate_increase_l454_45403

theorem interest_rate_increase (P : ℝ) (A1 A2 : ℝ) (T : ℝ) (R1 R2 : ℝ) (percentage_increase : ℝ) :
  P = 500 → A1 = 600 → A2 = 700 → T = 2 → 
  (A1 - P) = P * R1 * T →
  (A2 - P) = P * R2 * T →
  percentage_increase = (R2 - R1) / R1 * 100 →
  percentage_increase = 100 :=
by sorry

end NUMINAMATH_GPT_interest_rate_increase_l454_45403


namespace NUMINAMATH_GPT_dot_product_eq_half_l454_45457

noncomputable def vector_dot_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2
  
theorem dot_product_eq_half :
  vector_dot_product (Real.cos (25 * Real.pi / 180), Real.sin (25 * Real.pi / 180))
                     (Real.cos (85 * Real.pi / 180), Real.cos (5 * Real.pi / 180)) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_dot_product_eq_half_l454_45457


namespace NUMINAMATH_GPT_difference_of_smallest_integers_l454_45401

theorem difference_of_smallest_integers (n_1 n_2: ℕ) (h1 : ∀ k, 2 ≤ k ∧ k ≤ 6 → (n_1 > 1 ∧ n_1 % k = 1)) (h2 : ∀ k, 2 ≤ k ∧ k ≤ 6 → (n_2 > 1 ∧ n_2 % k = 1)) (h_smallest : n_1 = 61) (h_second_smallest : n_2 = 121) : n_2 - n_1 = 60 :=
by
  sorry

end NUMINAMATH_GPT_difference_of_smallest_integers_l454_45401


namespace NUMINAMATH_GPT_percent_difference_l454_45462

variables (w q y z x : ℝ)

-- Given conditions
def cond1 : Prop := w = 0.60 * q
def cond2 : Prop := q = 0.60 * y
def cond3 : Prop := z = 0.54 * y
def cond4 : Prop := x = 1.30 * w

-- The proof problem
theorem percent_difference (h1 : cond1 w q)
                           (h2 : cond2 q y)
                           (h3 : cond3 z y)
                           (h4 : cond4 x w) :
  ((z - x) / w) * 100 = 20 :=
by
  sorry

end NUMINAMATH_GPT_percent_difference_l454_45462


namespace NUMINAMATH_GPT_no_negative_product_l454_45429

theorem no_negative_product (x y : ℝ) (n : ℕ) (hx : x ≠ 0) (hy : y ≠ 0) 
(h1 : x ^ (2 * n) - y ^ (2 * n) > x) (h2 : y ^ (2 * n) - x ^ (2 * n) > y) : x * y ≥ 0 :=
sorry

end NUMINAMATH_GPT_no_negative_product_l454_45429


namespace NUMINAMATH_GPT_largest_rectangle_in_circle_l454_45425

theorem largest_rectangle_in_circle {r : ℝ} (h : r = 6) : 
  ∃ A : ℝ, A = 72 := 
by 
  sorry

end NUMINAMATH_GPT_largest_rectangle_in_circle_l454_45425


namespace NUMINAMATH_GPT_imaginary_part_of_z_l454_45478

-- Define complex numbers and necessary conditions
variable (z : ℂ)

-- The main statement
theorem imaginary_part_of_z (h : z * (1 + 2 * I) = 3 - 4 * I) : 
  (z.im = -2) :=
sorry

end NUMINAMATH_GPT_imaginary_part_of_z_l454_45478


namespace NUMINAMATH_GPT_find_d_l454_45420

theorem find_d (a₁: ℤ) (d : ℤ) (Sn : ℤ → ℤ) : 
  a₁ = 190 → 
  (Sn 20 > 0) → 
  (Sn 24 < 0) → 
  (Sn n = n * a₁ + (n * (n - 1)) / 2 * d) →
  d = -17 :=
by
  intros
  sorry

end NUMINAMATH_GPT_find_d_l454_45420


namespace NUMINAMATH_GPT_simplify_tan_cot_60_l454_45489

theorem simplify_tan_cot_60 :
  let tan60 := Real.sqrt 3
  let cot60 := 1 / Real.sqrt 3
  (tan60^3 + cot60^3) / (tan60 + cot60) = 7 / 3 :=
by
  let tan60 := Real.sqrt 3
  let cot60 := 1 / Real.sqrt 3
  sorry

end NUMINAMATH_GPT_simplify_tan_cot_60_l454_45489


namespace NUMINAMATH_GPT_hide_and_seek_problem_l454_45419

variable (A B V G D : Prop)

theorem hide_and_seek_problem :
  (A → (B ∧ ¬V)) →
  (B → (G ∨ D)) →
  (¬V → (¬B ∧ ¬D)) →
  (¬A → (B ∧ ¬G)) →
  ¬A ∧ B ∧ ¬V ∧ ¬G ∧ D :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_hide_and_seek_problem_l454_45419


namespace NUMINAMATH_GPT_equivalent_angle_terminal_side_l454_45472

theorem equivalent_angle_terminal_side (k : ℤ) (a : ℝ) (c : ℝ) (d : ℝ) : a = -3/10 * Real.pi → c = a * 180 / Real.pi → d = c + 360 * k →
   ∃ k : ℤ, d = 306 :=
sorry

end NUMINAMATH_GPT_equivalent_angle_terminal_side_l454_45472


namespace NUMINAMATH_GPT_tangent_line_at_point_l454_45449

def f (x : ℝ) : ℝ := x^3 + x - 16

def f' (x : ℝ) : ℝ := 3*x^2 + 1

def tangent_line (x : ℝ) (f'val : ℝ) (p_x p_y : ℝ) : ℝ := f'val * (x - p_x) + p_y

theorem tangent_line_at_point (x y : ℝ) (h : x = 2 ∧ y = -6 ∧ f 2 = -6) : 
  ∃ a b c : ℝ, a*x + b*y + c = 0 ∧ a = 13 ∧ b = -1 ∧ c = -32 :=
by
  use 13, -1, -32
  sorry

end NUMINAMATH_GPT_tangent_line_at_point_l454_45449


namespace NUMINAMATH_GPT_average_of_modified_set_l454_45466

theorem average_of_modified_set (a1 a2 a3 a4 a5 : ℝ) (h : (a1 + a2 + a3 + a4 + a5) / 5 = 8) :
  ((a1 + 10) + (a2 - 10) + (a3 + 10) + (a4 - 10) + (a5 + 10)) / 5 = 10 :=
by 
  sorry

end NUMINAMATH_GPT_average_of_modified_set_l454_45466


namespace NUMINAMATH_GPT_oblique_projection_correct_statements_l454_45477

-- Definitions of conditions
def oblique_projection_parallel_invariant : Prop :=
  ∀ (x_parallel y_parallel : Prop), x_parallel ∧ y_parallel

def oblique_projection_length_changes : Prop :=
  ∀ (x y : ℝ), x = y / 2 ∨ x = y

def triangle_is_triangle : Prop :=
  ∀ (t : Type), t = t

def square_is_rhombus : Prop :=
  ∀ (s : Type), s = s → false

def isosceles_trapezoid_is_parallelogram : Prop :=
  ∀ (it : Type), it = it → false

def rhombus_is_rhombus : Prop :=
  ∀ (r : Type), r = r → false

-- Math proof problem
theorem oblique_projection_correct_statements :
  (triangle_is_triangle ∧ oblique_projection_parallel_invariant ∧ oblique_projection_length_changes)
  → ¬square_is_rhombus ∧ ¬isosceles_trapezoid_is_parallelogram ∧ ¬rhombus_is_rhombus :=
by 
  sorry

end NUMINAMATH_GPT_oblique_projection_correct_statements_l454_45477


namespace NUMINAMATH_GPT_evaluate_expression_l454_45402

theorem evaluate_expression : (831 * 831) - (830 * 832) = 1 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l454_45402


namespace NUMINAMATH_GPT_avg_nested_l454_45497

def avg (a b c : ℚ) : ℚ := (a + b + c) / 3

theorem avg_nested {x y z : ℕ} :
  avg (avg 2 3 1) (avg 4 1 0) 5 = 26 / 9 :=
by
  sorry

end NUMINAMATH_GPT_avg_nested_l454_45497


namespace NUMINAMATH_GPT_sum_of_products_l454_45413

theorem sum_of_products (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 390) 
  (h2 : a + b + c = 20) : 
  ab + bc + ca = 5 :=
by 
  sorry

end NUMINAMATH_GPT_sum_of_products_l454_45413


namespace NUMINAMATH_GPT_sum_faces_edges_vertices_of_octagonal_pyramid_l454_45432

-- We define an octagonal pyramid with the given geometric properties.
structure OctagonalPyramid :=
  (base_vertices : ℕ) -- the number of vertices of the base
  (base_edges : ℕ)    -- the number of edges of the base
  (apex : ℕ)          -- the single apex of the pyramid
  (faces : ℕ)         -- the total number of faces: base face + triangular faces
  (edges : ℕ)         -- the total number of edges
  (vertices : ℕ)      -- the total number of vertices

-- Now we instantiate the structure based on the conditions.
def octagonalPyramid : OctagonalPyramid :=
  { base_vertices := 8,
    base_edges := 8,
    apex := 1,
    faces := 9,
    edges := 16,
    vertices := 9 }

-- We prove that the total number of faces, edges, and vertices sum to 34.
theorem sum_faces_edges_vertices_of_octagonal_pyramid : 
  (octagonalPyramid.faces + octagonalPyramid.edges + octagonalPyramid.vertices = 34) :=
by
  -- The proof steps are omitted as per instruction.
  sorry

end NUMINAMATH_GPT_sum_faces_edges_vertices_of_octagonal_pyramid_l454_45432


namespace NUMINAMATH_GPT_math_test_total_questions_l454_45470

theorem math_test_total_questions (Q : ℕ) (h : Q - 38 = 7) : Q = 45 :=
by
  sorry

end NUMINAMATH_GPT_math_test_total_questions_l454_45470


namespace NUMINAMATH_GPT_x_y_n_sum_l454_45445

theorem x_y_n_sum (x y n : ℕ) (h1 : 10 ≤ x ∧ x ≤ 99) (h2 : 10 ≤ y ∧ y ≤ 99) (h3 : y = (x % 10) * 10 + (x / 10)) (h4 : x^2 + y^2 = n^2) : x + y + n = 132 :=
sorry

end NUMINAMATH_GPT_x_y_n_sum_l454_45445


namespace NUMINAMATH_GPT_contrapositive_example_l454_45476

theorem contrapositive_example (x : ℝ) : 
  (x = 1 ∨ x = 2) → (x^2 - 3 * x + 2 ≤ 0) :=
by
  sorry

end NUMINAMATH_GPT_contrapositive_example_l454_45476


namespace NUMINAMATH_GPT_gift_boxes_in_3_days_l454_45491
-- Conditions:
def inchesPerBox := 18
def dailyWrapper := 90
-- "how many gift boxes will he be able to wrap every 3 days?"
theorem gift_boxes_in_3_days : 3 * (dailyWrapper / inchesPerBox) = 15 :=
by
  sorry

end NUMINAMATH_GPT_gift_boxes_in_3_days_l454_45491


namespace NUMINAMATH_GPT_smallest_number_of_coins_l454_45465

theorem smallest_number_of_coins (p n d q : ℕ) (total : ℕ) :
  (total < 100) →
  (total = p * 1 + n * 5 + d * 10 + q * 25) →
  (∀ k < 100, ∃ (p n d q : ℕ), k = p * 1 + n * 5 + d * 10 + q * 25) →
  p + n + d + q = 10 :=
sorry

end NUMINAMATH_GPT_smallest_number_of_coins_l454_45465


namespace NUMINAMATH_GPT_hyperbola_vertices_distance_l454_45418

noncomputable def distance_between_vertices : ℝ :=
  2 * Real.sqrt 7.5

theorem hyperbola_vertices_distance :
  ∀ (x y : ℝ), 4 * x^2 - 24 * x - y^2 + 6 * y - 3 = 0 →
  distance_between_vertices = 2 * Real.sqrt 7.5 :=
by sorry

end NUMINAMATH_GPT_hyperbola_vertices_distance_l454_45418


namespace NUMINAMATH_GPT_initial_amount_l454_45453

theorem initial_amount (H P L : ℝ) (C : ℝ) (n : ℕ) (T M : ℝ) 
  (hH : H = 10) 
  (hP : P = 2) 
  (hC : C = 1.25) 
  (hn : n = 4) 
  (hL : L = 3) 
  (hT : T = H + P + n * C) 
  (hM : M = T + L) : 
  M = 20 := 
sorry

end NUMINAMATH_GPT_initial_amount_l454_45453


namespace NUMINAMATH_GPT_knocks_to_knicks_l454_45444

-- Define the conversion relationships as conditions
variable (knicks knacks knocks : ℝ)

-- 8 knicks = 3 knacks
axiom h1 : 8 * knicks = 3 * knacks

-- 5 knacks = 6 knocks
axiom h2 : 5 * knacks = 6 * knocks

-- Proving the equivalence of 30 knocks to 66.67 knicks
theorem knocks_to_knicks : 30 * knocks = 66.67 * knicks :=
by
  sorry

end NUMINAMATH_GPT_knocks_to_knicks_l454_45444


namespace NUMINAMATH_GPT_find_8b_l454_45448

variable (a b : ℚ)

theorem find_8b (h1 : 4 * a + 3 * b = 5) (h2 : a = b - 3) : 8 * b = 136 / 7 := by
  sorry

end NUMINAMATH_GPT_find_8b_l454_45448


namespace NUMINAMATH_GPT_cos_C_of_triangle_l454_45410

theorem cos_C_of_triangle
  (sin_A : ℝ) (cos_B : ℝ) 
  (h1 : sin_A = 3/5)
  (h2 : cos_B = 5/13) :
  ∃ (cos_C : ℝ), cos_C = 16/65 :=
by
  -- Place for the proof
  sorry

end NUMINAMATH_GPT_cos_C_of_triangle_l454_45410


namespace NUMINAMATH_GPT_largest_number_sum13_product36_l454_45438

-- helper definitions for sum and product of digits
def sum_digits (n : ℕ) : ℕ := Nat.digits 10 n |> List.sum
def mul_digits (n : ℕ) : ℕ := Nat.digits 10 n |> List.foldr (· * ·) 1

theorem largest_number_sum13_product36 : 
  ∃ n : ℕ, sum_digits n = 13 ∧ mul_digits n = 36 ∧ ∀ m : ℕ, sum_digits m = 13 ∧ mul_digits m = 36 → m ≤ n :=
sorry

end NUMINAMATH_GPT_largest_number_sum13_product36_l454_45438


namespace NUMINAMATH_GPT_probability_truth_or_lies_l454_45408

def probability_truth := 0.30
def probability_lies := 0.20
def probability_both := 0.10

theorem probability_truth_or_lies :
  (probability_truth + probability_lies - probability_both) = 0.40 :=
by
  sorry

end NUMINAMATH_GPT_probability_truth_or_lies_l454_45408


namespace NUMINAMATH_GPT_A_finishes_race_in_36_seconds_l454_45440

-- Definitions of conditions
def distance_A := 130 -- A covers a distance of 130 meters
def distance_B := 130 -- B covers a distance of 130 meters
def time_B := 45 -- B covers the distance in 45 seconds
def distance_B_lag := 26 -- A beats B by 26 meters

-- Statement to prove
theorem A_finishes_race_in_36_seconds : 
  ∃ t : ℝ, distance_A / t + distance_B_lag = distance_B / time_B := sorry

end NUMINAMATH_GPT_A_finishes_race_in_36_seconds_l454_45440


namespace NUMINAMATH_GPT_area_ratio_triangle_PQR_ABC_l454_45452

noncomputable def area (A B C : ℝ×ℝ) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  (1 / 2) * (x1*y2 + x2*y3 + x3*y1 - x1*y3 - x2*y1 - x3*y2)

theorem area_ratio_triangle_PQR_ABC {A B C P Q R : ℝ×ℝ} 
  (h1 : dist A B + dist B C + dist C A = 1)
  (h2 : dist A P + dist P Q + dist Q B + dist B C + dist C A = 1)
  (h3 : dist P Q + dist Q R + dist R P = 1)
  (h4 : P.1 <= A.1 ∧ A.1 <= Q.1 ∧ Q.1 <= B.1) :
  area P Q R / area A B C > 2 / 9 :=
by
  sorry

end NUMINAMATH_GPT_area_ratio_triangle_PQR_ABC_l454_45452


namespace NUMINAMATH_GPT_k_value_range_l454_45422

noncomputable def f (x : ℝ) : ℝ := x - 1 - Real.log x

theorem k_value_range {k : ℝ} (h : ∀ x : ℝ, 0 < x → f x ≥ k * x - 2) : 
  k ≤ 1 - 1 / Real.exp 2 := 
sorry

end NUMINAMATH_GPT_k_value_range_l454_45422


namespace NUMINAMATH_GPT_product_defect_rate_correct_l454_45446

-- Definitions for the defect rates of the stages
def defect_rate_stage1 : ℝ := 0.10
def defect_rate_stage2 : ℝ := 0.03

-- Definitions for the probability of passing each stage without defects
def pass_rate_stage1 : ℝ := 1 - defect_rate_stage1
def pass_rate_stage2 : ℝ := 1 - defect_rate_stage2

-- Definition for the overall probability of a product not being defective
def pass_rate_overall : ℝ := pass_rate_stage1 * pass_rate_stage2

-- Definition for the overall defect rate based on the above probabilities
def defect_rate_product : ℝ := 1 - pass_rate_overall

-- The theorem statement to be proved
theorem product_defect_rate_correct : defect_rate_product = 0.127 :=
by
  -- Proof here
  sorry

end NUMINAMATH_GPT_product_defect_rate_correct_l454_45446


namespace NUMINAMATH_GPT_solution_set_f_over_x_lt_0_l454_45447

noncomputable def f : ℝ → ℝ := sorry

theorem solution_set_f_over_x_lt_0 :
  (∀ x, f (2 - x) = f (2 + x)) →
  (∀ x1 x2, x1 < 2 ∧ x2 < 2 ∧ x1 ≠ x2 → (f x1 - f x2) / (x1 - x2) < 0) →
  (f 4 = 0) →
  { x | f x / x < 0 } = { x | x < 0 } ∪ { x | 0 < x ∧ x < 4 } :=
by
  intros _ _ _
  sorry

end NUMINAMATH_GPT_solution_set_f_over_x_lt_0_l454_45447


namespace NUMINAMATH_GPT_system_solution_unique_l454_45493

theorem system_solution_unique (w x y z : ℝ) (h1 : w + x + y + z = 12)
  (h2 : w * x * y * z = w * x + w * y + w * z + x * y + x * z + y * z + 27) :
  w = 3 ∧ x = 3 ∧ y = 3 ∧ z = 3 := 
sorry

end NUMINAMATH_GPT_system_solution_unique_l454_45493


namespace NUMINAMATH_GPT_baseball_cards_per_pack_l454_45405

theorem baseball_cards_per_pack (cards_each : ℕ) (packs_total : ℕ) (total_cards : ℕ) (cards_per_pack : ℕ) :
    (cards_each = 540) →
    (packs_total = 108) →
    (total_cards = cards_each * 4) →
    (cards_per_pack = total_cards / packs_total) →
    cards_per_pack = 20 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_baseball_cards_per_pack_l454_45405


namespace NUMINAMATH_GPT_rational_function_eq_l454_45494

theorem rational_function_eq (f : ℚ → ℚ) 
  (h1 : f 1 = 2) 
  (h2 : ∀ x y : ℚ, f (x * y) = f x * f y - f (x + y) + 1) : 
  ∀ x : ℚ, f x = x + 1 :=
by sorry

end NUMINAMATH_GPT_rational_function_eq_l454_45494


namespace NUMINAMATH_GPT_like_term_exists_l454_45443

variable (a b : ℝ) (x y : ℝ)

theorem like_term_exists : ∃ b : ℝ, b * x^5 * y^3 = 3 * x^5 * y^3 ∧ b ≠ a :=
by
  -- existence of b
  use 3
  -- proof is omitted
  sorry

end NUMINAMATH_GPT_like_term_exists_l454_45443


namespace NUMINAMATH_GPT_monotonic_intervals_range_of_a_l454_45424

noncomputable def f (x a : ℝ) := Real.log x + (a / 2) * x^2 - (a + 1) * x
noncomputable def f' (x a : ℝ) := 1 / x + a * x - (a + 1)

theorem monotonic_intervals (a : ℝ) (ha : f 1 a = -2 ∧ f' 1 a = 0):
  (∀ x : ℝ, 0 < x ∧ x < (1 / 2) → f' x a > 0) ∧ 
  (∀ x : ℝ, x > 1 → f' x a > 0) ∧ 
  (∀ x : ℝ, (1 / 2) < x ∧ x < 1 → f' x a < 0) := sorry

theorem range_of_a (a : ℝ) 
  (h : ∀ x : ℕ, x > 0 → (f x a) / x < (f' x a) / 2):
  a > 2 * Real.exp (- (3 / 2)) - 1 := sorry

end NUMINAMATH_GPT_monotonic_intervals_range_of_a_l454_45424


namespace NUMINAMATH_GPT_number_of_real_solutions_is_one_l454_45484

noncomputable def num_real_solutions (a b c d : ℝ) : ℕ :=
  let x := Real.sin (a + b + c)
  let y := Real.sin (b + c + d)
  let z := Real.sin (c + d + a)
  let w := Real.sin (d + a + b)
  if (a + b + c + d) % 360 = 0 then 1 else 0

theorem number_of_real_solutions_is_one (a b c d : ℝ) (h : (a + b + c + d) % 360 = 0) :
  num_real_solutions a b c d = 1 :=
by
  sorry

end NUMINAMATH_GPT_number_of_real_solutions_is_one_l454_45484


namespace NUMINAMATH_GPT_least_possible_integer_l454_45460

theorem least_possible_integer :
  ∃ N : ℕ,
    (∀ k, 1 ≤ k ∧ k ≤ 30 → k ≠ 24 → k ≠ 25 → N % k = 0) ∧
    (N % 24 ≠ 0) ∧
    (N % 25 ≠ 0) ∧
    N = 659375723440 :=
by
  sorry

end NUMINAMATH_GPT_least_possible_integer_l454_45460


namespace NUMINAMATH_GPT_proof_statements_l454_45433

namespace ProofProblem

-- Definitions for each condition
def is_factor (x y : ℕ) : Prop := ∃ n : ℕ, y = n * x
def is_divisor (x y : ℕ) : Prop := is_factor x y

-- Lean 4 statement for the problem
theorem proof_statements :
  is_factor 4 20 ∧
  (is_divisor 19 209 ∧ ¬ is_divisor 19 63) ∧
  (¬ is_divisor 12 75 ∧ ¬ is_divisor 12 29) ∧
  (is_divisor 11 33 ∧ ¬ is_divisor 11 64) ∧
  is_factor 9 180 :=
by
  sorry

end ProofProblem

end NUMINAMATH_GPT_proof_statements_l454_45433


namespace NUMINAMATH_GPT_salary_based_on_tax_l454_45427

theorem salary_based_on_tax (salary tax paid_tax excess_800 excess_500 excess_500_2000 : ℤ) 
    (h1 : excess_800 = salary - 800)
    (h2 : excess_500 = min excess_800 500)
    (h3 : excess_500_2000 = excess_800 - excess_500)
    (h4 : paid_tax = (excess_500 * 5 / 100) + (excess_500_2000 * 10 / 100))
    (h5 : paid_tax = 80) :
  salary = 1850 := by
  sorry

end NUMINAMATH_GPT_salary_based_on_tax_l454_45427


namespace NUMINAMATH_GPT_unique_pair_exists_l454_45488

theorem unique_pair_exists :
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧
  (a + b + (Nat.gcd a b)^2 = Nat.lcm a b) ∧
  (Nat.lcm a b = 2 * Nat.lcm (a - 1) b) ∧
  (a, b) = (6, 15) :=
sorry

end NUMINAMATH_GPT_unique_pair_exists_l454_45488


namespace NUMINAMATH_GPT_markese_earnings_l454_45436

-- Define the conditions
def earnings_relation (E M : ℕ) : Prop :=
  M = E - 5 ∧ M + E = 37

-- The theorem to prove
theorem markese_earnings (E M : ℕ) (h : earnings_relation E M) : M = 16 :=
by
  sorry

end NUMINAMATH_GPT_markese_earnings_l454_45436


namespace NUMINAMATH_GPT_triangle_is_right_angled_l454_45411

-- Define the internal angles of a triangle
variables (A B C : ℝ)
-- Condition: A, B, C are internal angles of a triangle
-- This directly implies 0 < A, B, C < pi and A + B + C = pi

-- Internal angles of a triangle sum to π
axiom angles_sum_pi : A + B + C = Real.pi

-- Condition given in the problem
axiom sin_condition : Real.sin A = Real.sin C * Real.cos B

-- We need to prove that triangle ABC is right-angled
theorem triangle_is_right_angled : C = Real.pi / 2 :=
by
  sorry

end NUMINAMATH_GPT_triangle_is_right_angled_l454_45411


namespace NUMINAMATH_GPT_sqrt_identity_l454_45495

def condition1 (α : ℝ) : Prop := 
  ∃ P : ℝ × ℝ, P = (Real.sin 2, Real.cos 2) ∧ Real.sin α = Real.cos 2

def condition2 (P : ℝ × ℝ) : Prop := 
  P.1 ^ 2 + P.2 ^ 2 = 1

theorem sqrt_identity (α : ℝ) (P : ℝ × ℝ) 
  (h₁ : condition1 α) (h₂ : condition2 P) : 
  Real.sqrt (2 * (1 - Real.sin α)) = 2 * Real.sin 1 := by 
  sorry

end NUMINAMATH_GPT_sqrt_identity_l454_45495


namespace NUMINAMATH_GPT_population_increase_l454_45431

theorem population_increase (i j : ℝ) : 
  ∀ (m : ℝ), m * (1 + i / 100) * (1 + j / 100) = m * (1 + (i + j + i * j / 100) / 100) := 
by
  intro m
  sorry

end NUMINAMATH_GPT_population_increase_l454_45431


namespace NUMINAMATH_GPT_jackson_weekly_mileage_increase_l454_45412

theorem jackson_weekly_mileage_increase :
  ∃ (weeks : ℕ), weeks = (7 - 3) / 1 := by
  sorry

end NUMINAMATH_GPT_jackson_weekly_mileage_increase_l454_45412


namespace NUMINAMATH_GPT_largest_n_under_100000_l454_45435

theorem largest_n_under_100000 (n : ℕ) : 
  n < 100000 ∧ (9 * (n - 3)^6 - 3 * n^3 + 21 * n - 42) % 7 = 0 → n = 99996 :=
by
  sorry

end NUMINAMATH_GPT_largest_n_under_100000_l454_45435


namespace NUMINAMATH_GPT_technicians_in_workshop_l454_45451

theorem technicians_in_workshop (T R : ℕ) 
    (h1 : 700 * 15 = 800 * T + 650 * R)
    (h2 : T + R = 15) : T = 5 := 
by
  sorry

end NUMINAMATH_GPT_technicians_in_workshop_l454_45451
