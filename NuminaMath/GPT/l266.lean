import Mathlib

namespace NUMINAMATH_GPT_master_efficiency_comparison_l266_26602

theorem master_efficiency_comparison (z_parts : ℕ) (z_hours : ℕ) (l_parts : ℕ) (l_hours : ℕ)
    (hz : z_parts = 5) (hz_time : z_hours = 8)
    (hl : l_parts = 3) (hl_time : l_hours = 4) :
    (z_parts / z_hours : ℚ) < (l_parts / l_hours : ℚ) → false :=
by
  -- This is a placeholder for the proof, which is not needed as per the instructions.
  sorry

end NUMINAMATH_GPT_master_efficiency_comparison_l266_26602


namespace NUMINAMATH_GPT_decreasing_range_of_a_l266_26604

noncomputable def f (a x : ℝ) : ℝ := (Real.sqrt (2 - a * x)) / (a - 1)

theorem decreasing_range_of_a (a : ℝ) :
    (∀ x y : ℝ, 0 ≤ x → x ≤ 1/2 → 0 ≤ y → y ≤ 1/2 → x < y → f a y < f a x) ↔ (a < 0 ∨ (1 < a ∧ a ≤ 4)) :=
by
  sorry

end NUMINAMATH_GPT_decreasing_range_of_a_l266_26604


namespace NUMINAMATH_GPT_general_term_formula_of_arithmetic_seq_l266_26630

noncomputable def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem general_term_formula_of_arithmetic_seq 
  (a : ℕ → ℝ) (h_arith : arithmetic_seq a)
  (h1 : a 3 * a 7 = -16) 
  (h2 : a 4 + a 6 = 0) :
  (∀ n : ℕ, a n = 2 * n - 10) ∨ (∀ n : ℕ, a n = -2 * n + 10) :=
by
  sorry

end NUMINAMATH_GPT_general_term_formula_of_arithmetic_seq_l266_26630


namespace NUMINAMATH_GPT_market_price_article_l266_26682

theorem market_price_article (P : ℝ)
  (initial_tax_rate : ℝ := 0.035)
  (reduced_tax_rate : ℝ := 0.033333333333333)
  (difference_in_tax : ℝ := 11) :
  (initial_tax_rate * P - reduced_tax_rate * P = difference_in_tax) → 
  P = 6600 :=
by
  intro h
  /-
  We assume h: initial_tax_rate * P - reduced_tax_rate * P = difference_in_tax
  And we need to show P = 6600.
  The proof steps show that P = 6600 follows logically given h and the provided conditions.
  -/
  sorry

end NUMINAMATH_GPT_market_price_article_l266_26682


namespace NUMINAMATH_GPT_inequality_solution_set_l266_26642

theorem inequality_solution_set (x : ℝ) :
  (x - 3)^2 - 2 * Real.sqrt ((x - 3)^2) - 3 < 0 ↔ 0 < x ∧ x < 6 :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_set_l266_26642


namespace NUMINAMATH_GPT_quadratic_inequality_solution_range_l266_26660

theorem quadratic_inequality_solution_range (a : ℝ) :
  (∀ x : ℝ, a*x^2 + 2*a*x - 4 < 0) ↔ -4 < a ∧ a < 0 := 
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_range_l266_26660


namespace NUMINAMATH_GPT_regular_price_one_pound_is_20_l266_26672

variable (y : ℝ)
variable (discounted_price_quarter_pound : ℝ)

-- Conditions
axiom h1 : 0.6 * (y / 4) + 2 = discounted_price_quarter_pound
axiom h2 : discounted_price_quarter_pound = 2
axiom h3 : 0.1 * y = 2

-- Question: What is the regular price for one pound of cake?
theorem regular_price_one_pound_is_20 : y = 20 := 
  sorry

end NUMINAMATH_GPT_regular_price_one_pound_is_20_l266_26672


namespace NUMINAMATH_GPT_abs_sub_eq_three_l266_26665

theorem abs_sub_eq_three {m n : ℝ} (h1 : m * n = 4) (h2 : m + n = 5) : |m - n| = 3 := 
sorry

end NUMINAMATH_GPT_abs_sub_eq_three_l266_26665


namespace NUMINAMATH_GPT_phi_cannot_be_chosen_l266_26620

theorem phi_cannot_be_chosen (θ φ : ℝ) (hθ : -π/2 < θ ∧ θ < π/2) (hφ : 0 < φ ∧ φ < π)
  (h1 : 3 * Real.sin θ = 3 * Real.sqrt 2 / 2) 
  (h2 : 3 * Real.sin (-2*φ + θ) = 3 * Real.sqrt 2 / 2) : φ ≠ 5*π/4 :=
by
  sorry

end NUMINAMATH_GPT_phi_cannot_be_chosen_l266_26620


namespace NUMINAMATH_GPT_range_of_m_l266_26617

noncomputable def f (a x : ℝ) := a * (x^2 + 1) + Real.log x

theorem range_of_m (a m : ℝ) (h₁ : a ∈ Set.Ioo (-4 : ℝ) (-2))
  (h₂ : ∀ x ∈ Set.Icc (1 : ℝ) (3), ma - f a x > a^2) : m ≤ -2 := 
sorry

end NUMINAMATH_GPT_range_of_m_l266_26617


namespace NUMINAMATH_GPT_buy_items_ways_l266_26681

theorem buy_items_ways (headphones keyboards mice keyboard_mouse_sets headphone_mouse_sets : ℕ) :
  headphones = 9 → keyboards = 5 → mice = 13 → keyboard_mouse_sets = 4 → headphone_mouse_sets = 5 →
  (keyboard_mouse_sets * headphones) + (headphone_mouse_sets * keyboards) + (headphones * mice * keyboards) = 646 :=
by
  intros h_eq k_eq m_eq kms_eq hms_eq
  have h_eq_gen : headphones = 9 := h_eq
  have k_eq_gen : keyboards = 5 := k_eq
  have m_eq_gen : mice = 13 := m_eq
  have kms_eq_gen : keyboard_mouse_sets = 4 := kms_eq
  have hms_eq_gen : headphone_mouse_sets = 5 := hms_eq
  sorry

end NUMINAMATH_GPT_buy_items_ways_l266_26681


namespace NUMINAMATH_GPT_coin_flip_sequences_l266_26697

theorem coin_flip_sequences : (2 ^ 10 = 1024) :=
by
  sorry

end NUMINAMATH_GPT_coin_flip_sequences_l266_26697


namespace NUMINAMATH_GPT_triangle_no_real_solution_l266_26600

theorem triangle_no_real_solution (a b c : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) (habc : a + b > c ∧ b + c > a ∧ c + a > b) :
  ¬ (∀ x, x^2 - 2 * b * x + 2 * a * c = 0 ∧
       x^2 - 2 * c * x + 2 * a * b = 0 ∧
       x^2 - 2 * a * x + 2 * b * c = 0) :=
by
  intro H
  sorry

end NUMINAMATH_GPT_triangle_no_real_solution_l266_26600


namespace NUMINAMATH_GPT_evaluate_expression_l266_26647

theorem evaluate_expression : 150 * (150 - 4) - (150 * 150 - 6 + 2) = -596 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l266_26647


namespace NUMINAMATH_GPT_equation_solutions_count_l266_26696

theorem equation_solutions_count (n : ℕ) :
  (∃ (x y z : ℕ), 0 < x ∧ 0 < y ∧ 0 < z ∧ 2 * x + 3 * y + z + x^2 = n) →
  (n = 32 ∨ n = 33) :=
sorry

end NUMINAMATH_GPT_equation_solutions_count_l266_26696


namespace NUMINAMATH_GPT_k_values_l266_26693

def vector_dot (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

noncomputable def find_k (k : ℝ) : Prop :=
  (vector_dot (2, 3) (1, k) = 0) ∨
  (vector_dot (2, 3) (-1, k - 3) = 0) ∨
  (vector_dot (1, k) (-1, k - 3) = 0)

theorem k_values :
  ∃ k : ℝ, find_k k ∧ 
  (k = -2/3 ∨ k = 11/3 ∨ k = (3 + Real.sqrt 13) / 2 ∨ k = (3 - Real.sqrt 13 ) / 2) :=
by
  sorry

end NUMINAMATH_GPT_k_values_l266_26693


namespace NUMINAMATH_GPT_div_equivalence_l266_26611

theorem div_equivalence (a b c : ℝ) (h1: a / b = 3) (h2: b / c = 2 / 5) : c / a = 5 / 6 :=
by sorry

end NUMINAMATH_GPT_div_equivalence_l266_26611


namespace NUMINAMATH_GPT_ratio_w_to_y_l266_26650

variable (w x y z : ℚ)
variable (h1 : w / x = 5 / 4)
variable (h2 : y / z = 5 / 3)
variable (h3 : z / x = 1 / 5)

theorem ratio_w_to_y : w / y = 15 / 4 := sorry

end NUMINAMATH_GPT_ratio_w_to_y_l266_26650


namespace NUMINAMATH_GPT_impossible_score_53_l266_26654

def quizScoring (total_questions correct_answers incorrect_answers unanswered_questions score: ℤ) : Prop :=
  total_questions = 15 ∧
  correct_answers + incorrect_answers + unanswered_questions = 15 ∧
  score = 4 * correct_answers - incorrect_answers ∧
  unanswered_questions ≥ 0 ∧ correct_answers ≥ 0 ∧ incorrect_answers ≥ 0

theorem impossible_score_53 :
  ¬ ∃ (correct_answers incorrect_answers unanswered_questions : ℤ), quizScoring 15 correct_answers incorrect_answers unanswered_questions 53 := 
sorry

end NUMINAMATH_GPT_impossible_score_53_l266_26654


namespace NUMINAMATH_GPT_radius_of_cone_l266_26689

theorem radius_of_cone (S : ℝ) (h_S: S = 9 * Real.pi) (h_net: net_is_semi_circle) :
  ∃ (r : ℝ), r = Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_radius_of_cone_l266_26689


namespace NUMINAMATH_GPT_number_of_balls_l266_26667

noncomputable def totalBalls (frequency : ℚ) (yellowBalls : ℕ) : ℚ :=
  yellowBalls / frequency

theorem number_of_balls (h : totalBalls 0.3 6 = 20) : true :=
by
  sorry

end NUMINAMATH_GPT_number_of_balls_l266_26667


namespace NUMINAMATH_GPT_jill_sod_area_needed_l266_26608

def plot_width : ℕ := 200
def plot_length : ℕ := 50
def sidewalk_width : ℕ := 3
def sidewalk_length : ℕ := 50
def flower_bed1_depth : ℕ := 4
def flower_bed1_length : ℕ := 25
def flower_bed1_count : ℕ := 2
def flower_bed2_width : ℕ := 10
def flower_bed2_length : ℕ := 12
def flower_bed3_width : ℕ := 7
def flower_bed3_length : ℕ := 8

theorem jill_sod_area_needed :
  (plot_width * plot_length) - 
  (sidewalk_width * sidewalk_length + 
   flower_bed1_depth * flower_bed1_length * flower_bed1_count + 
   flower_bed2_width * flower_bed2_length + 
   flower_bed3_width * flower_bed3_length) = 9474 :=
by
  sorry

end NUMINAMATH_GPT_jill_sod_area_needed_l266_26608


namespace NUMINAMATH_GPT_range_of_x_l266_26622

theorem range_of_x (x : ℝ) (h1 : (x + 2) * (x - 3) ≤ 0) (h2 : |x + 1| ≥ 2) : 
  1 ≤ x ∧ x ≤ 3 :=
sorry

end NUMINAMATH_GPT_range_of_x_l266_26622


namespace NUMINAMATH_GPT_distance_between_points_l266_26699

theorem distance_between_points (x : ℝ) :
  let M := (-1, 4)
  let N := (x, 4)
  dist (M, N) = 5 →
  (x = -6 ∨ x = 4) := sorry

end NUMINAMATH_GPT_distance_between_points_l266_26699


namespace NUMINAMATH_GPT_terez_farm_pregnant_cows_percentage_l266_26663

theorem terez_farm_pregnant_cows_percentage (total_cows : ℕ) (female_percentage : ℕ) (pregnant_females : ℕ) 
  (ht : total_cows = 44) (hf : female_percentage = 50) (hp : pregnant_females = 11) :
  (pregnant_females * 100 / (female_percentage * total_cows / 100) = 50) :=
by 
  sorry

end NUMINAMATH_GPT_terez_farm_pregnant_cows_percentage_l266_26663


namespace NUMINAMATH_GPT_prop3_prop4_l266_26653

-- Definitions to represent planes and lines
variable (Plane Line : Type)

-- Predicate representing parallel planes or lines
variable (parallel : Plane → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)

-- Predicate representing perpendicular planes or lines
variable (perpendicular : Plane → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)

-- Distinct planes and a line
variables (α β γ : Plane) (l : Line)

-- Proposition 3: If l ⊥ α and l ∥ β, then α ⊥ β
theorem prop3 : perpendicular_line_plane l α ∧ parallel_line_plane l β → perpendicular α β :=
sorry

-- Proposition 4: If α ∥ β and α ⊥ γ, then β ⊥ γ
theorem prop4 : parallel α β ∧ perpendicular α γ → perpendicular β γ :=
sorry

end NUMINAMATH_GPT_prop3_prop4_l266_26653


namespace NUMINAMATH_GPT_least_number_of_stamps_is_6_l266_26678

noncomputable def exist_stamps : Prop :=
∃ (c f : ℕ), 5 * c + 7 * f = 40 ∧ c + f = 6

theorem least_number_of_stamps_is_6 : exist_stamps :=
sorry

end NUMINAMATH_GPT_least_number_of_stamps_is_6_l266_26678


namespace NUMINAMATH_GPT_speed_of_current_l266_26651

-- Definitions
def speed_boat_still_water := 60
def speed_downstream := 77
def speed_upstream := 43

-- Theorem statement
theorem speed_of_current : ∃ x, speed_boat_still_water + x = speed_downstream ∧ speed_boat_still_water - x = speed_upstream ∧ x = 17 :=
by
  unfold speed_boat_still_water speed_downstream speed_upstream
  sorry

end NUMINAMATH_GPT_speed_of_current_l266_26651


namespace NUMINAMATH_GPT_z_real_iff_m_1_or_2_z_complex_iff_not_m_1_and_2_z_pure_imaginary_iff_m_neg_half_z_in_second_quadrant_l266_26625

variables (m : ℝ)

def z_re (m : ℝ) : ℝ := 2 * m^2 - 3 * m - 2
def z_im (m : ℝ) : ℝ := m^2 - 3 * m + 2

-- Part (Ⅰ) Question 1
theorem z_real_iff_m_1_or_2 (m : ℝ) :
  z_im m = 0 ↔ (m = 1 ∨ m = 2) :=
sorry

-- Part (Ⅰ) Question 2
theorem z_complex_iff_not_m_1_and_2 (m : ℝ) :
  ¬ (m = 1 ∨ m = 2) ↔ (m ≠ 1 ∧ m ≠ 2) :=
sorry

-- Part (Ⅰ) Question 3
theorem z_pure_imaginary_iff_m_neg_half (m : ℝ) :
  z_re m = 0 ∧ z_im m ≠ 0 ↔ (m = -1/2) :=
sorry

-- Part (Ⅱ) Question
theorem z_in_second_quadrant (m : ℝ) :
  z_re m < 0 ∧ z_im m > 0 ↔ -1/2 < m ∧ m < 1 :=
sorry

end NUMINAMATH_GPT_z_real_iff_m_1_or_2_z_complex_iff_not_m_1_and_2_z_pure_imaginary_iff_m_neg_half_z_in_second_quadrant_l266_26625


namespace NUMINAMATH_GPT_cube_root_product_l266_26680

theorem cube_root_product : (343 : ℝ)^(1/3) * (125 : ℝ)^(1/3) = 35 := 
by
  sorry

end NUMINAMATH_GPT_cube_root_product_l266_26680


namespace NUMINAMATH_GPT_Charles_chocolate_milk_total_l266_26669

theorem Charles_chocolate_milk_total (milk_per_glass syrup_per_glass total_milk total_syrup : ℝ) 
(h_milk_glass : milk_per_glass = 6.5) (h_syrup_glass : syrup_per_glass = 1.5) (h_total_milk : total_milk = 130) (h_total_syrup : total_syrup = 60) :
  (min (total_milk / milk_per_glass) (total_syrup / syrup_per_glass) * (milk_per_glass + syrup_per_glass) = 160) :=
by
  sorry

end NUMINAMATH_GPT_Charles_chocolate_milk_total_l266_26669


namespace NUMINAMATH_GPT_box_weight_in_kg_l266_26694

def weight_of_one_bar : ℕ := 125 -- Weight of one chocolate bar in grams
def number_of_bars : ℕ := 16 -- Number of chocolate bars in the box
def grams_to_kg (g : ℕ) : ℕ := g / 1000 -- Function to convert grams to kilograms

theorem box_weight_in_kg : grams_to_kg (weight_of_one_bar * number_of_bars) = 2 :=
by
  sorry -- Proof is omitted

end NUMINAMATH_GPT_box_weight_in_kg_l266_26694


namespace NUMINAMATH_GPT_product_mnp_l266_26645

theorem product_mnp (a x y b : ℝ) (m n p : ℕ):
  (a ^ 8 * x * y - 2 * a ^ 7 * y - 3 * a ^ 6 * x = 2 * a ^ 5 * (b ^ 5 - 2)) ∧
  (a ^ 8 * x * y - 2 * a ^ 7 * y - 3 * a ^ 6 * x + 6 * a ^ 5 = (a ^ m * x - 2 * a ^ n) * (a ^ p * y - 3 * a ^ 3)) →
  m = 5 ∧ n = 4 ∧ p = 3 ∧ m * n * p = 60 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_product_mnp_l266_26645


namespace NUMINAMATH_GPT_identity_eq_l266_26698

theorem identity_eq (a b : ℤ) (h₁ : a = -1) (h₂ : b = 1) : 
  (∀ x : ℝ, ((2 * x + a) ^ 3) = (5 * x ^ 3 + (3 * x + b) * (x ^ 2 - x - 1) - 10 * x ^ 2 + 10 * x)) := by
  sorry

end NUMINAMATH_GPT_identity_eq_l266_26698


namespace NUMINAMATH_GPT_neg_p_necessary_but_not_sufficient_for_neg_q_l266_26675

variable (p q : Prop)

theorem neg_p_necessary_but_not_sufficient_for_neg_q
  (h1 : p → q)
  (h2 : ¬ (q → p)) : 
  (¬p → ¬q) ∧ (¬q → ¬p) := 
sorry

end NUMINAMATH_GPT_neg_p_necessary_but_not_sufficient_for_neg_q_l266_26675


namespace NUMINAMATH_GPT_solve_quadratic_inequality_l266_26631

theorem solve_quadratic_inequality (a x : ℝ) (h : a < 1) : 
  x^2 - (a + 1) * x + a < 0 ↔ (a < x ∧ x < 1) :=
by
  sorry

end NUMINAMATH_GPT_solve_quadratic_inequality_l266_26631


namespace NUMINAMATH_GPT_students_play_both_l266_26613

variable (students total_students football cricket neither : ℕ)
variable (H1 : total_students = 420)
variable (H2 : football = 325)
variable (H3 : cricket = 175)
variable (H4 : neither = 50)
  
theorem students_play_both (H1 : total_students = 420) (H2 : football = 325) 
    (H3 : cricket = 175) (H4 : neither = 50) : 
    students = 325 + 175 - (420 - 50) :=
by sorry

end NUMINAMATH_GPT_students_play_both_l266_26613


namespace NUMINAMATH_GPT_at_most_two_greater_than_one_l266_26661

theorem at_most_two_greater_than_one (a b c : ℝ) (h : a * b * c = 1) :
  ¬ (2 * a - 1 / b > 1 ∧ 2 * b - 1 / c > 1 ∧ 2 * c - 1 / a > 1) :=
by
  sorry

end NUMINAMATH_GPT_at_most_two_greater_than_one_l266_26661


namespace NUMINAMATH_GPT_determine_ordered_pair_l266_26684

theorem determine_ordered_pair (s n : ℤ)
    (h1 : ∀ t : ℤ, ∃ x y : ℤ,
        (x, y) = (s + 2 * t, -3 + n * t)) 
    (h2 : ∀ x y : ℤ, y = 2 * x - 7) :
    (s, n) = (2, 4) :=
by
  sorry

end NUMINAMATH_GPT_determine_ordered_pair_l266_26684


namespace NUMINAMATH_GPT_emily_has_28_beads_l266_26695

def beads_per_necklace : ℕ := 7
def necklaces : ℕ := 4

def total_beads : ℕ := necklaces * beads_per_necklace

theorem emily_has_28_beads : total_beads = 28 := by
  sorry

end NUMINAMATH_GPT_emily_has_28_beads_l266_26695


namespace NUMINAMATH_GPT_sufficient_condition_for_p_l266_26609

theorem sufficient_condition_for_p (m : ℝ) (h : 1 < m) : ∀ x : ℝ, x^2 - 2 * x + m > 0 :=
sorry

end NUMINAMATH_GPT_sufficient_condition_for_p_l266_26609


namespace NUMINAMATH_GPT_common_ratio_geometric_sequence_l266_26610

theorem common_ratio_geometric_sequence (a : ℕ → ℝ) (q : ℝ) (h_pos : ∀ n, 0 < a n) 
  (h_arith : 2 * (1/2 * a 5) = a 3 + a 4) : q = (1 + Real.sqrt 5) / 2 :=
sorry

end NUMINAMATH_GPT_common_ratio_geometric_sequence_l266_26610


namespace NUMINAMATH_GPT_first_divisor_l266_26632

-- Definitions
def is_divisible_by (n d : ℕ) : Prop := d > 0 ∧ n % d = 0

-- Theorem to prove
theorem first_divisor (x : ℕ) (h₁ : ∃ l, l = Nat.lcm x 35 ∧ is_divisible_by 1400 l ∧ 1400 / l = 8) : 
  x = 25 := 
sorry

end NUMINAMATH_GPT_first_divisor_l266_26632


namespace NUMINAMATH_GPT_jogging_track_circumference_l266_26677

theorem jogging_track_circumference (speed_deepak speed_wife : ℝ) (time_meet_minutes : ℝ) 
  (h1 : speed_deepak = 20) (h2 : speed_wife = 16) (h3 : time_meet_minutes = 36) : 
  let relative_speed := speed_deepak + speed_wife
  let time_meet_hours := time_meet_minutes / 60
  let circumference := relative_speed * time_meet_hours
  circumference = 21.6 :=
by
  sorry

end NUMINAMATH_GPT_jogging_track_circumference_l266_26677


namespace NUMINAMATH_GPT_bud_age_is_eight_l266_26671

def uncle_age : ℕ := 24

def bud_age (uncle_age : ℕ) : ℕ := uncle_age / 3

theorem bud_age_is_eight : bud_age uncle_age = 8 :=
by
  sorry

end NUMINAMATH_GPT_bud_age_is_eight_l266_26671


namespace NUMINAMATH_GPT_min_possible_range_l266_26618

theorem min_possible_range (A B C : ℤ) : 
  (A + 15 ≤ C ∧ B + 25 ≤ C ∧ C ≤ A + 45) → C - A ≤ 45 :=
by
  intros h
  have h1 : A + 15 ≤ C := h.1
  have h2 : B + 25 ≤ C := h.2.1
  have h3 : C ≤ A + 45 := h.2.2
  sorry

end NUMINAMATH_GPT_min_possible_range_l266_26618


namespace NUMINAMATH_GPT_total_trip_duration_proof_l266_26685

-- Naming all components
def driving_time : ℝ := 5
def first_jam_duration (pre_first_jam_drive : ℝ) : ℝ := 1.5 * pre_first_jam_drive
def second_jam_duration (between_first_and_second_drive : ℝ) : ℝ := 2 * between_first_and_second_drive
def third_jam_duration (between_second_and_third_drive : ℝ) : ℝ := 3 * between_second_and_third_drive
def pit_stop_duration : ℝ := 0.5
def pit_stops : ℕ := 2
def initial_drive : ℝ := 1
def second_drive : ℝ := 1.5

-- Additional drive time calculation
def remaining_drive : ℝ := driving_time - initial_drive - second_drive

-- Total duration calculation
def total_duration (initial_drive : ℝ) (second_drive : ℝ) (remaining_drive : ℝ) (first_jam_duration : ℝ) 
(second_jam_duration : ℝ) (third_jam_duration : ℝ) (pit_stop_duration : ℝ) (pit_stops : ℕ) : ℝ :=
  driving_time + first_jam_duration + second_jam_duration + third_jam_duration + (pit_stop_duration * pit_stops)

theorem total_trip_duration_proof :
  total_duration initial_drive second_drive remaining_drive (first_jam_duration initial_drive)
                  (second_jam_duration second_drive) (third_jam_duration remaining_drive) pit_stop_duration pit_stops 
  = 18 :=
by
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_total_trip_duration_proof_l266_26685


namespace NUMINAMATH_GPT_david_older_than_scott_l266_26637

-- Define the ages of Richard, David, and Scott
variables (R D S : ℕ)

-- Given conditions
def richard_age_eq : Prop := R = D + 6
def richard_twice_scott : Prop := R + 8 = 2 * (S + 8)
def david_current_age : Prop := D = 14

-- Prove the statement
theorem david_older_than_scott (h1 : richard_age_eq R D) (h2 : richard_twice_scott R S) (h3 : david_current_age D) :
  D - S = 8 :=
  sorry

end NUMINAMATH_GPT_david_older_than_scott_l266_26637


namespace NUMINAMATH_GPT_min_surface_area_base_edge_length_l266_26633

noncomputable def min_base_edge_length (V : ℝ) : ℝ :=
  2 * (V / (2 * Real.pi))^(1/3)

theorem min_surface_area_base_edge_length (V : ℝ) : 
  min_base_edge_length V = (4 * V)^(1/3) :=
by
  sorry

end NUMINAMATH_GPT_min_surface_area_base_edge_length_l266_26633


namespace NUMINAMATH_GPT_tangent_line_condition_l266_26686

theorem tangent_line_condition (k : ℝ) : 
  (∀ x y : ℝ, (x-2)^2 + (y-1)^2 = 1 → x - k * y - 1 = 0 → False) ↔ k = 0 :=
sorry

end NUMINAMATH_GPT_tangent_line_condition_l266_26686


namespace NUMINAMATH_GPT_f_properties_l266_26606

noncomputable def f (x : ℝ) : ℝ :=
if -2 < x ∧ x < 0 then 2^x else sorry

theorem f_properties (f_odd : ∀ x : ℝ, f (-x) = -f x)
                     (f_periodic : ∀ x : ℝ, f (x + 3 / 2) = -f x) :
  f 2014 + f 2015 + f 2016 = 0 :=
by 
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_f_properties_l266_26606


namespace NUMINAMATH_GPT_largest_sampled_item_l266_26688

theorem largest_sampled_item (n : ℕ) (m : ℕ) (a : ℕ) (k : ℕ)
  (hn : n = 360)
  (hm : m = 30)
  (hk : k = n / m)
  (ha : a = 105) :
  ∃ b, b = 433 ∧ (∃ i, i < m ∧ a = 1 + i * k) → (∃ j, j < m ∧ b = 1 + j * k) :=
by
  sorry

end NUMINAMATH_GPT_largest_sampled_item_l266_26688


namespace NUMINAMATH_GPT_minimize_notch_volume_l266_26626

noncomputable def total_volume (theta phi : ℝ) : ℝ :=
  let part1 := (2 / 3) * Real.tan phi
  let part2 := (2 / 3) * Real.tan (theta - phi)
  part1 + part2

theorem minimize_notch_volume :
  ∀ (theta : ℝ), (0 < theta ∧ theta < π) →
  ∃ (phi : ℝ), (0 < phi ∧ phi < θ) ∧
  (∀ ψ : ℝ, (0 < ψ ∧ ψ < θ) → total_volume theta ψ ≥ total_volume theta (theta / 2)) :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_minimize_notch_volume_l266_26626


namespace NUMINAMATH_GPT_solution_set_inequality_l266_26656

theorem solution_set_inequality (a m : ℝ) (h : ∀ x : ℝ, (x > m ∧ x < 1) ↔ 2 * x^2 - 3 * x + a < 0) : m = 1 / 2 :=
by
  -- Insert the proof here
  sorry

end NUMINAMATH_GPT_solution_set_inequality_l266_26656


namespace NUMINAMATH_GPT_parallel_vectors_l266_26624

noncomputable def vector_a : ℝ × ℝ := (-1, 2)
noncomputable def vector_b (m : ℝ) : ℝ × ℝ := (2, m)

theorem parallel_vectors (m : ℝ) (h : ∃ k : ℝ, vector_a = (k • vector_b m)) : m = -4 :=
by {
  sorry
}

end NUMINAMATH_GPT_parallel_vectors_l266_26624


namespace NUMINAMATH_GPT_similar_inscribed_triangle_exists_l266_26640

variable {α : Type*} [LinearOrderedField α]

-- Representing points and triangles
structure Point (α : Type*) := (x : α) (y : α)
structure Triangle (α : Type*) := (A B C : Point α)

-- Definitions for inscribed triangles and similarity conditions
def isInscribed (inner outer : Triangle α) : Prop :=
  -- Dummy definition, needs correct geometric interpretation
  sorry

def areSimilar (Δ1 Δ2 : Triangle α) : Prop :=
  -- Dummy definition, needs correct geometric interpretation
  sorry

-- Main theorem
theorem similar_inscribed_triangle_exists (Δ₁ Δ₂ : Triangle α) (h_ins : isInscribed Δ₂ Δ₁) :
  ∃ Δ₃ : Triangle α, isInscribed Δ₃ Δ₂ ∧ areSimilar Δ₁ Δ₃ :=
sorry

end NUMINAMATH_GPT_similar_inscribed_triangle_exists_l266_26640


namespace NUMINAMATH_GPT_general_term_formula_l266_26641

theorem general_term_formula (n : ℕ) (a : ℕ → ℚ) :
  (∀ n, a n = (-1)^n * (n^2)/(2 * n - 1)) :=
sorry

end NUMINAMATH_GPT_general_term_formula_l266_26641


namespace NUMINAMATH_GPT_angle_CBD_is_48_degrees_l266_26612

theorem angle_CBD_is_48_degrees :
  ∀ (A B D C : Type) (α β γ δ : ℝ), 
    α = 28 ∧ β = 46 ∧ C ∈ [B, D] ∧ γ = 30 → 
    δ = 48 := 
by 
  sorry

end NUMINAMATH_GPT_angle_CBD_is_48_degrees_l266_26612


namespace NUMINAMATH_GPT_range_of_a_l266_26646
noncomputable def exponential_quadratic (a : ℝ) : Prop :=
  ∃ x : ℝ, 0 < x ∧ (1/4)^x + (1/2)^(x-1) + a = 0

theorem range_of_a (a : ℝ) : exponential_quadratic a ↔ -3 < a ∧ a < 0 :=
sorry

end NUMINAMATH_GPT_range_of_a_l266_26646


namespace NUMINAMATH_GPT_system_of_equations_solution_l266_26639

theorem system_of_equations_solution (a b x y : ℝ) 
  (h1 : x = 1) 
  (h2 : y = 2)
  (h3 : a * x + y = -1)
  (h4 : 2 * x - b * y = 0) : 
  a + b = -2 := 
sorry

end NUMINAMATH_GPT_system_of_equations_solution_l266_26639


namespace NUMINAMATH_GPT_solve_fraction_equation_l266_26644

theorem solve_fraction_equation :
  ∀ x : ℚ, (x + 4) / (x - 3) = (x - 2) / (x + 2) → x = -2 / 11 :=
by
  intro x
  intro h
  sorry

end NUMINAMATH_GPT_solve_fraction_equation_l266_26644


namespace NUMINAMATH_GPT_range_of_y_l266_26635

theorem range_of_y (y : ℝ) (h₁ : y < 0) (h₂ : ⌈y⌉ * ⌊y⌋ = 110) : -11 < y ∧ y < -10 := 
sorry

end NUMINAMATH_GPT_range_of_y_l266_26635


namespace NUMINAMATH_GPT_arithmetic_square_root_of_nine_l266_26676

theorem arithmetic_square_root_of_nine : ∃ (x : ℝ), (x * x = 9) ∧ (x ≥ 0) ∧ (x = 3) :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_square_root_of_nine_l266_26676


namespace NUMINAMATH_GPT_quadratic_has_real_root_l266_26629

theorem quadratic_has_real_root (a : ℝ) : 
  ¬(∀ x : ℝ, x^2 + a * x + a - 1 ≠ 0) :=
sorry

end NUMINAMATH_GPT_quadratic_has_real_root_l266_26629


namespace NUMINAMATH_GPT_k_value_correct_l266_26687

theorem k_value_correct (k : ℚ) : 
  let f (x : ℚ) := 4 * x^2 - 3 * x + 5
  let g (x : ℚ) := x^2 + k * x - 8
  (f 5 - g 5 = 20) -> k = 53 / 5 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_k_value_correct_l266_26687


namespace NUMINAMATH_GPT_sum_of_reciprocals_ineq_l266_26679

theorem sum_of_reciprocals_ineq (a b c : ℝ) (h : a + b + c = 3) : 
  (1 / (5 * a ^ 2 - 4 * a + 11)) + 
  (1 / (5 * b ^ 2 - 4 * b + 11)) + 
  (1 / (5 * c ^ 2 - 4 * c + 11)) ≤ 
  (1 / 4) := 
by {
  sorry
}

end NUMINAMATH_GPT_sum_of_reciprocals_ineq_l266_26679


namespace NUMINAMATH_GPT_shanghai_mock_exam_problem_l266_26652

noncomputable def a_n : ℕ → ℝ := sorry -- Defines the arithmetic sequence 

theorem shanghai_mock_exam_problem 
  (a_is_arithmetic : ∃ d a₀, ∀ n, a_n n = a₀ + n * d)
  (h₁ : a_n 1 + a_n 3 + a_n 5 = 9)
  (h₂ : a_n 2 + a_n 4 + a_n 6 = 15) :
  a_n 3 + a_n 4 = 8 := 
  sorry

end NUMINAMATH_GPT_shanghai_mock_exam_problem_l266_26652


namespace NUMINAMATH_GPT_smallest_w_for_factors_l266_26658

theorem smallest_w_for_factors (w : ℕ) (h_pos : 0 < w) :
  (2^5 ∣ 936 * w) ∧ (3^3 ∣ 936 * w) ∧ (13^2 ∣ 936 * w) ↔ w = 156 := 
sorry

end NUMINAMATH_GPT_smallest_w_for_factors_l266_26658


namespace NUMINAMATH_GPT_total_shaded_area_is_71_l266_26605

-- Define the dimensions of the first rectangle
def rect1_length : ℝ := 4
def rect1_width : ℝ := 12

-- Define the dimensions of the second rectangle
def rect2_length : ℝ := 5
def rect2_width : ℝ := 7

-- Define the dimensions of the overlap area
def overlap_length : ℝ := 3
def overlap_width : ℝ := 4

-- Define the area calculation
def area (length width : ℝ) : ℝ := length * width

-- Calculate the areas of the rectangles and the overlap
def rect1_area : ℝ := area rect1_length rect1_width
def rect2_area : ℝ := area rect2_length rect2_width
def overlap_area : ℝ := area overlap_length overlap_width

-- Total shaded area calculation
def total_shaded_area : ℝ := rect1_area + rect2_area - overlap_area

-- Proof statement to show that the total shaded area is 71 square units
theorem total_shaded_area_is_71 : total_shaded_area = 71 := by
  sorry

end NUMINAMATH_GPT_total_shaded_area_is_71_l266_26605


namespace NUMINAMATH_GPT_necessary_and_sufficient_condition_l266_26666

theorem necessary_and_sufficient_condition :
  ∀ a b : ℝ, (a + b > 0) ↔ ((a ^ 3) + (b ^ 3) > 0) :=
by
  sorry

end NUMINAMATH_GPT_necessary_and_sufficient_condition_l266_26666


namespace NUMINAMATH_GPT_excess_percentage_l266_26655

theorem excess_percentage (A B : ℝ) (x : ℝ) 
  (hA' : A' = A * (1 + x / 100))
  (hB' : B' = B * (1 - 5 / 100))
  (h_area_err : A' * B' = 1.007 * (A * B)) : x = 6 :=
by
  sorry

end NUMINAMATH_GPT_excess_percentage_l266_26655


namespace NUMINAMATH_GPT_simplify_expression_l266_26664

theorem simplify_expression :
  2 + 1 / (2 + 1 / (2 + 1 / 2)) = 29 / 12 :=
by
  sorry  -- Proof will be provided here

end NUMINAMATH_GPT_simplify_expression_l266_26664


namespace NUMINAMATH_GPT_coeff_of_z_in_eq2_l266_26643

-- Definitions of the conditions from part a)
def equation1 (x y z : ℤ) := 6 * x - 5 * y + 3 * z = 22
def equation2 (x y z : ℤ) := 4 * x + 8 * y - z = (7 : ℚ) / 11
def equation3 (x y z : ℤ) := 5 * x - 6 * y + 2 * z = 12
def sum_xyz (x y z : ℤ) := x + y + z = 10

-- Theorem stating that the coefficient of z in equation 2 is -1.
theorem coeff_of_z_in_eq2 {x y z : ℤ} (h1 : equation1 x y z) (h2 : equation2 x y z) (h3 : equation3 x y z) (h4 : sum_xyz x y z) :
    -1 = -1 :=
by
  -- This is a placeholder for the proof.
  sorry

end NUMINAMATH_GPT_coeff_of_z_in_eq2_l266_26643


namespace NUMINAMATH_GPT_basketball_game_l266_26649

theorem basketball_game (a r b d : ℕ) (r_gt_1 : r > 1) (d_gt_0 : d > 0)
  (H1 : a = b)
  (H2 : a * (1 + r) * (1 + r^2) = 4 * b + 6 * d + 2)
  (H3 : a * (1 + r) * (1 + r^2) ≤ 100)
  (H4 : 4 * b + 6 * d ≤ 98) :
  (a + a * r) + (b + (b + d)) = 43 := 
sorry

end NUMINAMATH_GPT_basketball_game_l266_26649


namespace NUMINAMATH_GPT_find_f_neg2_l266_26662

theorem find_f_neg2 (a b : ℝ) (f : ℝ → ℝ) (h₁ : ∀ x, f x = x^5 + a*x^3 + x^2 + b*x + 2) (h₂ : f 2 = 3) : f (-2) = 9 :=
by
  sorry

end NUMINAMATH_GPT_find_f_neg2_l266_26662


namespace NUMINAMATH_GPT_total_weight_of_oranges_l266_26636

theorem total_weight_of_oranges :
  let capacity1 := 80
  let capacity2 := 50
  let capacity3 := 60
  let filled1 := 3 / 4
  let filled2 := 3 / 5
  let filled3 := 2 / 3
  let weight_per_orange1 := 0.25
  let weight_per_orange2 := 0.30
  let weight_per_orange3 := 0.40
  let num_oranges1 := capacity1 * filled1
  let num_oranges2 := capacity2 * filled2
  let num_oranges3 := capacity3 * filled3
  let total_weight1 := num_oranges1 * weight_per_orange1
  let total_weight2 := num_oranges2 * weight_per_orange2
  let total_weight3 := num_oranges3 * weight_per_orange3
  total_weight1 + total_weight2 + total_weight3 = 40 := by
  sorry

end NUMINAMATH_GPT_total_weight_of_oranges_l266_26636


namespace NUMINAMATH_GPT_find_k_such_that_product_minus_one_is_perfect_power_l266_26691

noncomputable def product_of_first_n_primes (n : ℕ) : ℕ :=
  (List.take n (List.filter (Nat.Prime) (List.range n.succ))).prod

theorem find_k_such_that_product_minus_one_is_perfect_power :
  ∀ k : ℕ, ∃ a n : ℕ, (product_of_first_n_primes k) - 1 = a^n ∧ n > 1 ∧ k = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_k_such_that_product_minus_one_is_perfect_power_l266_26691


namespace NUMINAMATH_GPT_quadratic_equation_roots_transformation_l266_26690

theorem quadratic_equation_roots_transformation (α β : ℝ) 
  (h1 : 3 * α^2 + 7 * α + 4 = 0)
  (h2 : 3 * β^2 + 7 * β + 4 = 0) :
  ∃ y : ℝ, 21 * y^2 - 23 * y + 6 = 0 :=
sorry

end NUMINAMATH_GPT_quadratic_equation_roots_transformation_l266_26690


namespace NUMINAMATH_GPT_correct_transformation_l266_26683

theorem correct_transformation (x : ℝ) (h : 3 * x - 7 = 2 * x) : 3 * x - 2 * x = 7 :=
sorry

end NUMINAMATH_GPT_correct_transformation_l266_26683


namespace NUMINAMATH_GPT_ratio_between_two_numbers_l266_26614

noncomputable def first_number : ℕ := 48
noncomputable def lcm_value : ℕ := 432
noncomputable def second_number : ℕ := 9 * 24  -- Derived from the given conditions in the problem

def ratio (a b : ℕ) : ℚ := (a : ℚ) / (b : ℚ)

theorem ratio_between_two_numbers 
  (A B : ℕ) 
  (hA : A = first_number) 
  (hLCM : Nat.lcm A B = lcm_value) 
  (hB : B = 9 * 24) : 
  ratio A B = 1 / 4.5 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_ratio_between_two_numbers_l266_26614


namespace NUMINAMATH_GPT_students_with_exactly_two_skills_l266_26619

-- Definitions based on the conditions:
def total_students : ℕ := 150
def students_can_write : ℕ := total_students - 60 -- 150 - 60 = 90
def students_can_direct : ℕ := total_students - 90 -- 150 - 90 = 60
def students_can_produce : ℕ := total_students - 40 -- 150 - 40 = 110

-- The theorem statement
theorem students_with_exactly_two_skills :
  students_can_write + students_can_direct + students_can_produce - total_students = 110 := 
sorry

end NUMINAMATH_GPT_students_with_exactly_two_skills_l266_26619


namespace NUMINAMATH_GPT_sum_of_squares_of_coeffs_l266_26670

theorem sum_of_squares_of_coeffs (a b c : ℕ) : (a = 6) → (b = 24) → (c = 12) → (a^2 + b^2 + c^2 = 756) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_of_coeffs_l266_26670


namespace NUMINAMATH_GPT_intersection_complement_A_B_l266_26601

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x < 2}
def B : Set ℝ := {x | x < 1}

theorem intersection_complement_A_B : A ∩ (U \ B) = {x | 1 ≤ x ∧ x < 2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_complement_A_B_l266_26601


namespace NUMINAMATH_GPT_find_b_l266_26607

noncomputable def curve (a : ℝ) (x : ℝ) : ℝ := a * x + Real.log x

def derivative_at_one (a : ℝ) : ℝ := a + 1

def tangent_line (b : ℝ) (x : ℝ) : ℝ := 2 * x + b

theorem find_b (a b : ℝ) (h_deriv : derivative_at_one a = 2) (h_tangent : tangent_line b 1 = curve a 1) :
  b = -1 :=
by
  sorry

end NUMINAMATH_GPT_find_b_l266_26607


namespace NUMINAMATH_GPT_security_deposit_amount_correct_l266_26627

noncomputable def daily_rate : ℝ := 125.00
noncomputable def pet_fee : ℝ := 100.00
noncomputable def service_cleaning_fee_rate : ℝ := 0.20
noncomputable def security_deposit_rate : ℝ := 0.50
noncomputable def weeks : ℝ := 2
noncomputable def days_per_week : ℝ := 7

noncomputable def number_of_days : ℝ := weeks * days_per_week
noncomputable def total_rental_fee : ℝ := number_of_days * daily_rate
noncomputable def total_rental_fee_with_pet : ℝ := total_rental_fee + pet_fee
noncomputable def service_cleaning_fee : ℝ := service_cleaning_fee_rate * total_rental_fee_with_pet
noncomputable def total_cost : ℝ := total_rental_fee_with_pet + service_cleaning_fee

theorem security_deposit_amount_correct : 
    security_deposit_rate * total_cost = 1110.00 := 
by 
  sorry

end NUMINAMATH_GPT_security_deposit_amount_correct_l266_26627


namespace NUMINAMATH_GPT_min_value_of_quadratic_l266_26628

theorem min_value_of_quadratic :
  ∃ (x y : ℝ), 2 * x^2 + 4 * x * y + 5 * y^2 - 4 * x - 6 * y + 1 = -3 :=
sorry

end NUMINAMATH_GPT_min_value_of_quadratic_l266_26628


namespace NUMINAMATH_GPT_rides_first_day_l266_26648

variable (total_rides : ℕ) (second_day_rides : ℕ)

theorem rides_first_day (h1 : total_rides = 7) (h2 : second_day_rides = 3) : total_rides - second_day_rides = 4 :=
by
  sorry

end NUMINAMATH_GPT_rides_first_day_l266_26648


namespace NUMINAMATH_GPT_julia_drove_miles_l266_26621

theorem julia_drove_miles :
  ∀ (daily_rental_cost cost_per_mile total_paid : ℝ),
    daily_rental_cost = 29 →
    cost_per_mile = 0.08 →
    total_paid = 46.12 →
    total_paid - daily_rental_cost = cost_per_mile * 214 :=
by
  intros _ _ _ d_cost_eq cpm_eq tp_eq
  -- calculation and proof steps will be filled here
  sorry

end NUMINAMATH_GPT_julia_drove_miles_l266_26621


namespace NUMINAMATH_GPT_find_a5_l266_26659

-- Define an arithmetic sequence with a given common difference
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) := ∀ n : ℕ, a (n + 1) = a n + d

-- Define that three terms form a geometric sequence
def geometric_sequence (x y z : ℝ) := y^2 = x * z

-- Given conditions for the problem
def a₁ : ℝ := 1  -- found from the geometric sequence condition
def d : ℝ := 2

-- The definition of the sequence {a_n} based on the common difference
noncomputable def a_n (n : ℕ) : ℝ := a₁ + n * d

-- Given that a_1, a_2, a_5 form a geometric sequence
axiom geo_progression : geometric_sequence a₁ (a_n 1) (a_n 4)

-- The proof goal
theorem find_a5 : a_n 4 = 9 :=
by
  -- the proof is skipped
  sorry

end NUMINAMATH_GPT_find_a5_l266_26659


namespace NUMINAMATH_GPT_tangent_line_at_point_l266_26623

noncomputable def f (x : ℝ) : ℝ := x^3 - 2 * x^2 - 4 * x + 2

def point : ℝ × ℝ := (1, -3)

def tangent_line (x y : ℝ) : Prop := 5 * x + y - 2 = 0

theorem tangent_line_at_point : tangent_line 1 (-3) :=
  sorry

end NUMINAMATH_GPT_tangent_line_at_point_l266_26623


namespace NUMINAMATH_GPT_intersection_of_M_and_N_l266_26634

def M : Set ℝ := {x | (x + 3) * (x - 2) < 0}
def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

theorem intersection_of_M_and_N : M ∩ N = {x | 1 ≤ x ∧ x < 2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_M_and_N_l266_26634


namespace NUMINAMATH_GPT_median_interval_60_64_l266_26668

theorem median_interval_60_64 
  (students : ℕ) 
  (f_45_49 f_50_54 f_55_59 f_60_64 : ℕ) :
  students = 105 ∧ 
  f_45_49 = 8 ∧ 
  f_50_54 = 15 ∧ 
  f_55_59 = 20 ∧ 
  f_60_64 = 18 ∧ 
  (8 + 15 + 20 + 18) ≥ (105 + 1) / 2
  → 60 ≤ (105 + 1) / 2  ∧ (105 + 1) / 2 ≤ 64 :=
sorry

end NUMINAMATH_GPT_median_interval_60_64_l266_26668


namespace NUMINAMATH_GPT_box_distribution_l266_26603

theorem box_distribution (A P S : ℕ) (h : A + P + S = 22) : A ≥ 8 ∨ P ≥ 8 ∨ S ≥ 8 := 
by 
-- The next step is to use proof by contradiction, assuming the opposite.
sorry

end NUMINAMATH_GPT_box_distribution_l266_26603


namespace NUMINAMATH_GPT_speed_of_stream_l266_26673

theorem speed_of_stream (v c : ℝ) (h1 : c - v = 6) (h2 : c + v = 10) : v = 2 :=
by
  sorry

end NUMINAMATH_GPT_speed_of_stream_l266_26673


namespace NUMINAMATH_GPT_min_value_one_div_a_plus_one_div_b_l266_26615

theorem min_value_one_div_a_plus_one_div_b (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) : 
  (1 / a + 1 / b) ≥ 4 :=
by
  sorry

end NUMINAMATH_GPT_min_value_one_div_a_plus_one_div_b_l266_26615


namespace NUMINAMATH_GPT_problem_1_problem_2_l266_26638

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  Real.log (x + 1) + Real.log (1 - x) + a * (x + 1)

def mono_intervals (a : ℝ) : Set ℝ × Set ℝ := 
  if a = 1 then ((Set.Ioo (-1) (Real.sqrt 2 - 1)), (Set.Ico (Real.sqrt 2 - 1) 1)) 
  else (∅, ∅)

theorem problem_1 (a : ℝ) (h_pos : a > 0) : 
  mono_intervals a = if a = 1 then ((Set.Ioo (-1) (Real.sqrt 2 - 1)), (Set.Ico (Real.sqrt 2 - 1) 1)) else (∅, ∅) :=
sorry

theorem problem_2 (h_max : f a 0 = 1) (h_pos : a > 0) : 
  a = 1 :=
sorry

end NUMINAMATH_GPT_problem_1_problem_2_l266_26638


namespace NUMINAMATH_GPT_relationship_of_x_vals_l266_26692

variables {k x1 x2 x3 : ℝ}

noncomputable def inverse_proportion_function (k x : ℝ) : ℝ := k / x

theorem relationship_of_x_vals (h1 : inverse_proportion_function k x1 = 1)
                              (h2 : inverse_proportion_function k x2 = -5)
                              (h3 : inverse_proportion_function k x3 = 3)
                              (hk : k < 0) :
                              x1 < x3 ∧ x3 < x2 :=
by
  sorry

end NUMINAMATH_GPT_relationship_of_x_vals_l266_26692


namespace NUMINAMATH_GPT_proportion_solution_l266_26657

theorem proportion_solution (x : ℝ) (h : 0.75 / x = 5 / 6) : x = 0.9 := by
  sorry

end NUMINAMATH_GPT_proportion_solution_l266_26657


namespace NUMINAMATH_GPT_quotient_is_12_l266_26674

theorem quotient_is_12 (a b q : ℕ) (h1: q = a / b) (h2: q = a / 2) (h3: q = 6 * b) : q = 12 :=
by 
  sorry

end NUMINAMATH_GPT_quotient_is_12_l266_26674


namespace NUMINAMATH_GPT_six_digit_palindromes_count_l266_26616

open Nat

theorem six_digit_palindromes_count :
  let digits := {d | 0 ≤ d ∧ d ≤ 9}
  let a_digits := {a | 1 ≤ a ∧ a ≤ 9}
  let b_digits := digits
  let c_digits := digits
  ∃ (total : ℕ), (∀ a ∈ a_digits, ∀ b ∈ b_digits, ∀ c ∈ c_digits, True) → total = 900 :=
by
  sorry

end NUMINAMATH_GPT_six_digit_palindromes_count_l266_26616
