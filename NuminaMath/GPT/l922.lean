import Mathlib

namespace NUMINAMATH_GPT_ott_fractional_part_l922_92271

theorem ott_fractional_part (M L N O x : ℝ)
  (hM : M = 6 * x)
  (hL : L = 5 * x)
  (hN : N = 4 * x)
  (hO : O = 0)
  (h_each : O + M + L + N = x + x + x) :
  (3 * x) / (M + L + N) = 1 / 5 :=
by
  sorry

end NUMINAMATH_GPT_ott_fractional_part_l922_92271


namespace NUMINAMATH_GPT_sale_in_first_month_l922_92236

theorem sale_in_first_month 
  (sale_month_2 : ℕ)
  (sale_month_3 : ℕ)
  (sale_month_4 : ℕ)
  (sale_month_5 : ℕ)
  (required_sale_month_6 : ℕ)
  (average_sale_6_months : ℕ)
  (total_sale_6_months : ℕ)
  (total_known_sales : ℕ)
  (sale_first_month : ℕ) : 
    sale_month_2 = 3920 →
    sale_month_3 = 3855 →
    sale_month_4 = 4230 →
    sale_month_5 = 3560 →
    required_sale_month_6 = 2000 →
    average_sale_6_months = 3500 →
    total_sale_6_months = 6 * average_sale_6_months →
    total_known_sales = sale_month_2 + sale_month_3 + sale_month_4 + sale_month_5 →
    total_sale_6_months - (total_known_sales + required_sale_month_6) = sale_first_month →
    sale_first_month = 3435 :=
by
  intros h2 h3 h4 h5 h6 h_avg h_total h_known h_calc
  sorry

end NUMINAMATH_GPT_sale_in_first_month_l922_92236


namespace NUMINAMATH_GPT_problem_statement_l922_92252

theorem problem_statement (x y z w : ℝ)
  (h1 : x + y + z + w = 0)
  (h7 : x^7 + y^7 + z^7 + w^7 = 0) :
  w * (w + x) * (w + y) * (w + z) = 0 := 
sorry

end NUMINAMATH_GPT_problem_statement_l922_92252


namespace NUMINAMATH_GPT_evaluate_expression_l922_92293

theorem evaluate_expression : 2009 * (2007 / 2008) + (1 / 2008) = 2008 := 
by 
  sorry

end NUMINAMATH_GPT_evaluate_expression_l922_92293


namespace NUMINAMATH_GPT_find_expression_value_l922_92226

variable (x y z : ℚ)
variable (h1 : x - y + 2 * z = 1)
variable (h2 : x + y + 4 * z = 3)

theorem find_expression_value : x + 2 * y + 5 * z = 4 := 
by {
  sorry
}

end NUMINAMATH_GPT_find_expression_value_l922_92226


namespace NUMINAMATH_GPT_minimum_area_of_triangle_l922_92213

def parabola_focus : Prop :=
  ∃ F : ℝ × ℝ, F = (1, 0)

def on_parabola (A B : ℝ × ℝ) : Prop :=
  (A.2 ^ 2 = 4 * A.1 ∧ B.2 ^ 2 = 4 * B.1) ∧ (A.2 * B.2 < 0)

def dot_product_condition (A B : ℝ × ℝ) : Prop :=
  A.1 * B.1 + A.2 * B.2 = -4

noncomputable def area (A B : ℝ × ℝ) : ℝ :=
  1 / 2 * abs (A.1 * B.2 - B.1 * A.2)

theorem minimum_area_of_triangle
  (A B : ℝ × ℝ)
  (h_focus : parabola_focus)
  (h_on_parabola : on_parabola A B)
  (h_dot : dot_product_condition A B) :
  ∃ C : ℝ, C = 4 * Real.sqrt 2 ∧ area A B = C :=
by
  sorry

end NUMINAMATH_GPT_minimum_area_of_triangle_l922_92213


namespace NUMINAMATH_GPT_necessary_and_sufficient_condition_l922_92239

theorem necessary_and_sufficient_condition (a b : ℝ) : a > b ↔ a^3 > b^3 :=
by {
  sorry
}

end NUMINAMATH_GPT_necessary_and_sufficient_condition_l922_92239


namespace NUMINAMATH_GPT_initial_weight_l922_92292

noncomputable def initial_average_weight (A : ℝ) : Prop :=
  let total_weight_initial := 20 * A
  let total_weight_new := total_weight_initial + 210
  let new_average_weight := 181.42857142857142
  total_weight_new / 21 = new_average_weight

theorem initial_weight:
  ∃ A : ℝ, initial_average_weight A ∧ A = 180 :=
by
  sorry

end NUMINAMATH_GPT_initial_weight_l922_92292


namespace NUMINAMATH_GPT_first_player_winning_strategy_l922_92242

theorem first_player_winning_strategy (num_chips : ℕ) : 
  (num_chips = 110) → 
  ∃ (moves : ℕ → ℕ × ℕ), (∀ n, 1 ≤ (moves n).1 ∧ (moves n).1 ≤ 9) ∧ 
  (∀ n, (moves n).1 ≠ (moves (n-1)).1) →
  (∃ move_sequence : ℕ → ℕ, ∀ k, move_sequence k ≤ num_chips ∧ 
  ((move_sequence (k+1) < move_sequence k) ∨ (move_sequence (k+1) = 0 ∧ move_sequence k = 1)) ∧ 
  (move_sequence k > 0) ∧ (move_sequence 0 = num_chips) →
  num_chips ≡ 14 [MOD 32]) :=
by 
  sorry

end NUMINAMATH_GPT_first_player_winning_strategy_l922_92242


namespace NUMINAMATH_GPT_monotonic_intervals_range_of_a_for_inequality_l922_92281

noncomputable def f (a x : ℝ) : ℝ := (x + a) / (a * Real.exp x)

theorem monotonic_intervals (a : ℝ) :
  (if a > 0 then
    ∀ x, (x < (1 - a) → 0 < deriv (f a) x) ∧ ((1 - a) < x → deriv (f a) x < 0)
  else
    ∀ x, (x < (1 - a) → deriv (f a) x < 0) ∧ ((1 - a) < x → 0 < deriv (f a) x)) := 
sorry

theorem range_of_a_for_inequality (a : ℝ) :
  (∀ x, 0 < x → (3 + 2 * Real.log x) / Real.exp x ≤ f a x + 2 * x) ↔
  a ∈ Set.Iio (-1/2) ∪ Set.Ioi 0 :=
sorry

end NUMINAMATH_GPT_monotonic_intervals_range_of_a_for_inequality_l922_92281


namespace NUMINAMATH_GPT_teacher_age_is_94_5_l922_92250

noncomputable def avg_age_students : ℝ := 18
noncomputable def num_students : ℝ := 50
noncomputable def avg_age_class_with_teacher : ℝ := 19.5
noncomputable def num_total : ℝ := 51

noncomputable def total_age_students : ℝ := num_students * avg_age_students
noncomputable def total_age_class_with_teacher : ℝ := num_total * avg_age_class_with_teacher

theorem teacher_age_is_94_5 : ∃ T : ℝ, total_age_students + T = total_age_class_with_teacher ∧ T = 94.5 := by
  sorry

end NUMINAMATH_GPT_teacher_age_is_94_5_l922_92250


namespace NUMINAMATH_GPT_nested_abs_expression_eval_l922_92248

theorem nested_abs_expression_eval :
  abs (abs (-abs (-2 + 3) - 2) + 3) = 6 := sorry

end NUMINAMATH_GPT_nested_abs_expression_eval_l922_92248


namespace NUMINAMATH_GPT_radius_relation_l922_92283

-- Define the conditions under which the spheres exist
variable {R r : ℝ}

-- The problem statement
theorem radius_relation (h : r = R * (2 - Real.sqrt 2)) : r = R * (2 - Real.sqrt 2) :=
sorry

end NUMINAMATH_GPT_radius_relation_l922_92283


namespace NUMINAMATH_GPT_find_some_number_l922_92255

theorem find_some_number (d : ℝ) (x : ℝ) (h1 : d = (0.889 * x) / 9.97) (h2 : d = 4.9) :
  x = 54.9 := by
  sorry

end NUMINAMATH_GPT_find_some_number_l922_92255


namespace NUMINAMATH_GPT_problem_statement_l922_92258

-- Define the expression in Lean
def expr : ℤ := 120 * (120 - 5) - (120 * 120 - 10 + 2)

-- Theorem stating the value of the expression
theorem problem_statement : expr = -592 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l922_92258


namespace NUMINAMATH_GPT_minimum_club_members_l922_92280

theorem minimum_club_members : ∃ (b : ℕ), (b = 7) ∧ ∃ (a : ℕ), (2 : ℚ) / 5 < (a : ℚ) / b ∧ (a : ℚ) / b < 1 / 2 := 
sorry

end NUMINAMATH_GPT_minimum_club_members_l922_92280


namespace NUMINAMATH_GPT_thabo_HNF_calculation_l922_92245

variable (THABO_BOOKS : ℕ)

-- Conditions as definitions
def total_books : ℕ := 500
def fiction_books : ℕ := total_books * 40 / 100
def non_fiction_books : ℕ := total_books * 60 / 100
def paperback_non_fiction_books (HNF : ℕ) : ℕ := HNF + 50
def total_non_fiction_books (HNF : ℕ) : ℕ := HNF + paperback_non_fiction_books HNF

-- Lean statement to prove
theorem thabo_HNF_calculation (HNF : ℕ) :
  total_books = 500 →
  fiction_books = 200 →
  non_fiction_books = 300 →
  total_non_fiction_books HNF = 300 →
  2 * HNF + 50 = 300 →
  HNF = 125 :=
by
  intros _
         _
         _
         _
         _
  sorry

end NUMINAMATH_GPT_thabo_HNF_calculation_l922_92245


namespace NUMINAMATH_GPT_sum_of_ages_l922_92225

-- Definitions based on conditions
variables (J S : ℝ) -- J and S are real numbers

-- First condition: Jane is five years older than Sarah
def jane_older_than_sarah := J = S + 5

-- Second condition: Nine years from now, Jane will be three times as old as Sarah was three years ago
def future_condition := J + 9 = 3 * (S - 3)

-- Conclusion to prove
theorem sum_of_ages (h1 : jane_older_than_sarah J S) (h2 : future_condition J S) : J + S = 28 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_ages_l922_92225


namespace NUMINAMATH_GPT_find_m_find_A_inter_CUB_l922_92209

-- Definitions of sets A and B given m
def A (m : ℤ) : Set ℤ := {-4, 2 * m - 1, m ^ 2}
def B (m : ℤ) : Set ℤ := {9, m - 5, 1 - m}

-- Define the universal set U
def U : Set ℤ := Set.univ

-- First part: Prove that m = -3
theorem find_m (m : ℤ) : A m ∩ B m = {9} → m = -3 := sorry

-- Condition that m = -3 is true
def m_val : ℤ := -3

-- Second part: Prove A ∩ C_U B = {-4, -7}
theorem find_A_inter_CUB: A m_val ∩ (U \ B m_val) = {-4, -7} := sorry

end NUMINAMATH_GPT_find_m_find_A_inter_CUB_l922_92209


namespace NUMINAMATH_GPT_find_possible_values_l922_92204

noncomputable def possible_values (a b : ℝ) : Set ℝ :=
  { x | ∃ (a b : ℝ), 0 < a ∧ 0 < b ∧ a + b = 2 ∧ x = (1/a + 1/b) }

theorem find_possible_values :
  (∀ (a b : ℝ), 0 < a → 0 < b → a + b = 2 → (1 / a + 1 / b) ∈ Set.Ici 2) ∧
  (∀ y, y ∈ Set.Ici 2 → ∃ (a b : ℝ), 0 < a ∧ 0 < b ∧ a + b = 2 ∧ y = (1 / a + 1 / b)) :=
by
  sorry

end NUMINAMATH_GPT_find_possible_values_l922_92204


namespace NUMINAMATH_GPT_math_problem_l922_92240

-- Definitions based on conditions
def avg2 (a b : ℚ) : ℚ := (a + b) / 2
def avg4 (a b c d : ℚ) : ℚ := (a + b + c + d) / 4

-- Main theorem statement
theorem math_problem :
  avg4 (avg4 2 2 0 2) (avg2 3 1) 0 3 = 13 / 8 :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l922_92240


namespace NUMINAMATH_GPT_maximize_x3y4_correct_l922_92270

noncomputable def maximize_x3y4 : ℝ × ℝ :=
  let x := 160 / 7
  let y := 120 / 7
  (x, y)

theorem maximize_x3y4_correct :
  ∃ x y : ℝ, 0 < x ∧ 0 < y ∧ x + y = 40 ∧ (x, y) = maximize_x3y4 ∧ 
  ∀ (x' y' : ℝ), 0 < x' ∧ 0 < y' ∧ x' + y' = 40 → x ^ 3 * y ^ 4 ≥ x' ^ 3 * y' ^ 4 :=
by
  sorry

end NUMINAMATH_GPT_maximize_x3y4_correct_l922_92270


namespace NUMINAMATH_GPT_ratio_of_larger_to_smaller_l922_92203

theorem ratio_of_larger_to_smaller (a b : ℝ) (h : a > 0) (h' : b > 0) (h_sum_diff : a + b = 7 * (a - b)) :
  a / b = 4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_larger_to_smaller_l922_92203


namespace NUMINAMATH_GPT_range_of_a_l922_92277

theorem range_of_a (a : ℝ) : (¬ (∃ x0 : ℝ, a * x0^2 + x0 + 1/2 ≤ 0)) → a > 1/2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l922_92277


namespace NUMINAMATH_GPT_pawns_on_black_squares_even_l922_92274

theorem pawns_on_black_squares_even (A : Fin 8 → Fin 8) :
  ∃ n : ℕ, ∀ i, (i + A i).val % 2 = 1 → n % 2 = 0 :=
sorry

end NUMINAMATH_GPT_pawns_on_black_squares_even_l922_92274


namespace NUMINAMATH_GPT_least_multiple_of_21_gt_380_l922_92235

theorem least_multiple_of_21_gt_380 : ∃ n : ℕ, (21 * n > 380) ∧ (21 * n = 399) :=
sorry

end NUMINAMATH_GPT_least_multiple_of_21_gt_380_l922_92235


namespace NUMINAMATH_GPT_calculation_l922_92299

-- Define the exponents and base values as conditions
def exponent : ℕ := 3 ^ 2
def neg_base : ℤ := -2
def pos_base : ℤ := 2

-- The calculation expressions as conditions
def term1 : ℤ := neg_base^exponent
def term2 : ℤ := pos_base^exponent

-- The proof statement: Show that the sum of the terms equals 0
theorem calculation : term1 + term2 = 0 := sorry

end NUMINAMATH_GPT_calculation_l922_92299


namespace NUMINAMATH_GPT_range_of_c_l922_92220

noncomputable def p (c : ℝ) : Prop := ∀ x : ℝ, (2 * c - 1) ^ x = (2 * c - 1) ^ x

def q (c : ℝ) : Prop := ∀ x : ℝ, x + |x - 2 * c| > 1

theorem range_of_c (c : ℝ) (h1 : c > 0)
  (h2 : p c ∨ q c) (h3 : ¬ (p c ∧ q c)) : c ≥ 1 :=
sorry

end NUMINAMATH_GPT_range_of_c_l922_92220


namespace NUMINAMATH_GPT_no_such_abc_l922_92211

theorem no_such_abc :
  ¬ ∃ (a b c : ℕ+),
    (∃ k1 : ℕ, a ^ 2 * b * c + 2 = k1 ^ 2) ∧
    (∃ k2 : ℕ, b ^ 2 * c * a + 2 = k2 ^ 2) ∧
    (∃ k3 : ℕ, c ^ 2 * a * b + 2 = k3 ^ 2) := 
sorry

end NUMINAMATH_GPT_no_such_abc_l922_92211


namespace NUMINAMATH_GPT_geometric_sequence_is_alternating_l922_92275

theorem geometric_sequence_is_alternating (a : ℕ → ℝ) (q : ℝ)
  (h1 : a 1 + a 2 = -3 / 2)
  (h2 : a 4 + a 5 = 12)
  (hg : ∀ n, a (n + 1) = q * a n) :
  ∃ q, q < 0 ∧ ∀ n, a n * a (n + 1) ≤ 0 :=
by sorry

end NUMINAMATH_GPT_geometric_sequence_is_alternating_l922_92275


namespace NUMINAMATH_GPT_train_speed_l922_92290

theorem train_speed (L : ℝ) (T : ℝ) (hL : L = 200) (hT : T = 20) :
  L / T = 10 := by
  rw [hL, hT]
  norm_num
  done

end NUMINAMATH_GPT_train_speed_l922_92290


namespace NUMINAMATH_GPT_points_five_units_away_from_neg_one_l922_92286

theorem points_five_units_away_from_neg_one (x : ℝ) :
  |x + 1| = 5 ↔ x = 4 ∨ x = -6 :=
by
  sorry

end NUMINAMATH_GPT_points_five_units_away_from_neg_one_l922_92286


namespace NUMINAMATH_GPT_volume_Q3_l922_92257

def Q0 : ℚ := 8
def delta : ℚ := (1 / 3) ^ 3
def ratio : ℚ := 6 / 27

def Q (i : ℕ) : ℚ :=
  match i with
  | 0 => Q0
  | 1 => Q0 + 4 * delta
  | n + 1 => Q n + delta * (ratio ^ n)

theorem volume_Q3 : Q 3 = 5972 / 729 := 
by
  sorry

end NUMINAMATH_GPT_volume_Q3_l922_92257


namespace NUMINAMATH_GPT_apples_per_pie_l922_92212

theorem apples_per_pie (total_apples handed_out_apples pies made_pies remaining_apples : ℕ) 
  (h_initial : total_apples = 86)
  (h_handout : handed_out_apples = 30)
  (h_made_pies : made_pies = 7)
  (h_remaining : remaining_apples = total_apples - handed_out_apples) :
  remaining_apples / made_pies = 8 :=
by
  sorry

end NUMINAMATH_GPT_apples_per_pie_l922_92212


namespace NUMINAMATH_GPT_prince_wish_fulfilled_l922_92263

theorem prince_wish_fulfilled
  (k : ℕ)
  (k_gt_1 : 1 < k)
  (k_lt_13 : k < 13)
  (city : Fin 13 → Fin k) 
  (initial_goblets : Fin k → Fin 13)
  (is_gold : Fin 13 → Bool) :
  ∃ i j : Fin 13, i ≠ j ∧ city i = city j ∧ is_gold i = true ∧ is_gold j = true := 
sorry

end NUMINAMATH_GPT_prince_wish_fulfilled_l922_92263


namespace NUMINAMATH_GPT_problem_l922_92296

    theorem problem (a b c : ℝ) : 
        a < b → 
        (∀ x : ℝ, (x ≤ -2 ∨ |x - 30| < 2) ↔ (0 ≤ (x - a) * (x - b) / (x - c))) → 
        a + 2 * b + 3 * c = 86 := by 
    sorry

end NUMINAMATH_GPT_problem_l922_92296


namespace NUMINAMATH_GPT_box_length_is_24_l922_92266

theorem box_length_is_24 (L : ℕ) (h1 : ∀ s : ℕ, (L * 40 * 16 = 30 * s^3) → s ∣ 40 ∧ s ∣ 16) (h2 : ∃ s : ℕ, s ∣ 40 ∧ s ∣ 16) : L = 24 :=
by
  sorry

end NUMINAMATH_GPT_box_length_is_24_l922_92266


namespace NUMINAMATH_GPT_track_circumference_l922_92202

variable (A B : Nat → ℝ)
variable (speedA speedB : ℝ)
variable (x : ℝ) -- half the circumference of the track
variable (y : ℝ) -- the circumference of the track

theorem track_circumference
  (x_pos : 0 < x)
  (y_def : y = 2 * x)
  (start_opposite : A 0 = 0 ∧ B 0 = x)
  (B_first_meet_150 : ∃ t₁, B t₁ = 150 ∧ A t₁ = x - 150)
  (A_second_meet_90 : ∃ t₂, A t₂ = 2 * x - 90 ∧ B t₂ = x + 90) :
  y = 720 := 
by 
  sorry

end NUMINAMATH_GPT_track_circumference_l922_92202


namespace NUMINAMATH_GPT_paving_stone_length_l922_92273

theorem paving_stone_length
  (length_courtyard : ℝ)
  (width_courtyard : ℝ)
  (num_paving_stones : ℝ)
  (width_paving_stone : ℝ)
  (total_area : ℝ := length_courtyard * width_courtyard)
  (area_per_paving_stone : ℝ := (total_area / num_paving_stones))
  (length_paving_stone : ℝ := (area_per_paving_stone / width_paving_stone)) :
  length_courtyard = 20 ∧
  width_courtyard = 16.5 ∧
  num_paving_stones = 66 ∧
  width_paving_stone = 2 →
  length_paving_stone = 2.5 :=
by {
   sorry
}

end NUMINAMATH_GPT_paving_stone_length_l922_92273


namespace NUMINAMATH_GPT_friends_activity_l922_92285

-- Defining the problem conditions
def total_friends : ℕ := 5
def organizers : ℕ := 3
def managers : ℕ := total_friends - organizers

-- Stating the proof problem
theorem friends_activity (h1 : organizers = 3) (h2 : managers = 2) :
  Nat.choose total_friends organizers = 10 :=
sorry

end NUMINAMATH_GPT_friends_activity_l922_92285


namespace NUMINAMATH_GPT_books_ratio_l922_92276

theorem books_ratio (c e : ℕ) (h_ratio : c / e = 2 / 5) (h_sampled : c = 10) : e = 25 :=
by
  sorry

end NUMINAMATH_GPT_books_ratio_l922_92276


namespace NUMINAMATH_GPT_fourth_term_geometric_series_l922_92264

theorem fourth_term_geometric_series (a₁ a₅ : ℕ) (r : ℕ) :
  a₁ = 6 → a₅ = 1458 → (∀ n, aₙ = a₁ * r^(n-1)) → r = 3 → (∃ a₄, a₄ = a₁ * r^(4-1) ∧ a₄ = 162) :=
by intros h₁ h₅ H r_sol
   sorry

end NUMINAMATH_GPT_fourth_term_geometric_series_l922_92264


namespace NUMINAMATH_GPT_inequality_solution_l922_92288

theorem inequality_solution (x : ℝ) :
  27 ^ (Real.log x / Real.log 3) ^ 2 - 8 * x ^ (Real.log x / Real.log 3) ≥ 3 ↔
  x ∈ Set.Icc 0 (1 / 3) ∪ Set.Ici 3 :=
sorry

end NUMINAMATH_GPT_inequality_solution_l922_92288


namespace NUMINAMATH_GPT_range_of_k_l922_92222

noncomputable def f (x : ℝ) : ℝ := x - Real.sin x

theorem range_of_k (k : ℝ) :
  (∀ x : ℝ, f (-x^2 + 3 * x) + f (x - 2 * k) ≤ 0) ↔ k ≥ 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_k_l922_92222


namespace NUMINAMATH_GPT_zero_points_of_function_l922_92234

theorem zero_points_of_function : 
  (∃ x y : ℝ, y = x - 4 / x ∧ y = 0) → (∃! x : ℝ, x = -2 ∨ x = 2) :=
by
  sorry

end NUMINAMATH_GPT_zero_points_of_function_l922_92234


namespace NUMINAMATH_GPT_time_at_2010_minutes_after_3pm_is_930pm_l922_92247

def time_after_2010_minutes (current_time : Nat) (minutes_passed : Nat) : Nat :=
  sorry

theorem time_at_2010_minutes_after_3pm_is_930pm :
  time_after_2010_minutes 900 2010 = 1290 :=
by
  sorry

end NUMINAMATH_GPT_time_at_2010_minutes_after_3pm_is_930pm_l922_92247


namespace NUMINAMATH_GPT_Raven_age_l922_92228

-- Define the conditions
def Phoebe_age_current : Nat := 10
def Phoebe_age_in_5_years : Nat := Phoebe_age_current + 5

-- Define the hypothesis that in 5 years Raven will be 4 times as old as Phoebe
def Raven_in_5_years (R : Nat) : Prop := R + 5 = 4 * Phoebe_age_in_5_years

-- State the theorem to be proved
theorem Raven_age : ∃ R : Nat, Raven_in_5_years R ∧ R = 55 :=
by
  sorry

end NUMINAMATH_GPT_Raven_age_l922_92228


namespace NUMINAMATH_GPT_fraction_food_l922_92267

-- Define the salary S and remaining amount H
def S : ℕ := 170000
def H : ℕ := 17000

-- Define fractions of the salary spent on house rent and clothes
def fraction_rent : ℚ := 1 / 10
def fraction_clothes : ℚ := 3 / 5

-- Define the fraction F to be proven
def F : ℚ := 1 / 5

-- Define the remaining fraction of the salary
def remaining_fraction : ℚ := H / S

theorem fraction_food :
  ∀ S H : ℕ,
  S = 170000 →
  H = 17000 →
  F = 1 / 5 →
  F + (fraction_rent + fraction_clothes) + remaining_fraction = 1 :=
by
  intros S H hS hH hF
  sorry

end NUMINAMATH_GPT_fraction_food_l922_92267


namespace NUMINAMATH_GPT_range_of_a_for_monotonicity_l922_92221

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then (a - 2) * x - 1 else Real.log x / Real.log a

theorem range_of_a_for_monotonicity (a : ℝ) :
  (∀ x y : ℝ, x ≤ y → f a x ≤ f a y) ↔ 2 < a ∧ a ≤ 3 :=
by sorry

end NUMINAMATH_GPT_range_of_a_for_monotonicity_l922_92221


namespace NUMINAMATH_GPT_power_of_128_div_7_eq_16_l922_92279

theorem power_of_128_div_7_eq_16 : (128 : ℝ) ^ (4 / 7) = 16 := by
  sorry

end NUMINAMATH_GPT_power_of_128_div_7_eq_16_l922_92279


namespace NUMINAMATH_GPT_second_train_speed_l922_92282

theorem second_train_speed
  (v : ℕ)
  (h1 : 8 * v - 8 * 11 = 160) :
  v = 31 :=
sorry

end NUMINAMATH_GPT_second_train_speed_l922_92282


namespace NUMINAMATH_GPT_average_test_score_45_percent_l922_92238

theorem average_test_score_45_percent (x : ℝ) 
  (h1 : 0.45 * x + 0.50 * 78 + 0.05 * 60 = 84.75) : 
  x = 95 :=
by sorry

end NUMINAMATH_GPT_average_test_score_45_percent_l922_92238


namespace NUMINAMATH_GPT_no_solutions_l922_92208

theorem no_solutions : ¬ ∃ x : ℝ, (6 * x - 2 < (x + 2)^2) ∧ ((x + 2)^2 < 8 * x - 4) := by
  sorry

end NUMINAMATH_GPT_no_solutions_l922_92208


namespace NUMINAMATH_GPT_rob_baseball_cards_l922_92256

theorem rob_baseball_cards
  (r j r_d : ℕ)
  (hj : j = 40)
  (h_double : r_d = j / 5)
  (h_cards : r = 3 * r_d) :
  r = 24 :=
by
  sorry

end NUMINAMATH_GPT_rob_baseball_cards_l922_92256


namespace NUMINAMATH_GPT_percentage_fruits_in_good_condition_l922_92243

theorem percentage_fruits_in_good_condition (oranges bananas : ℕ) (rotten_oranges_pct rotten_bananas_pct : ℚ)
    (h_oranges : oranges = 600) (h_bananas : bananas = 400)
    (h_rotten_oranges_pct : rotten_oranges_pct = 0.15) (h_rotten_bananas_pct : rotten_bananas_pct = 0.06) :
    let rotten_oranges := (rotten_oranges_pct * oranges : ℚ)
    let rotten_bananas := (rotten_bananas_pct * bananas : ℚ)
    let total_rotten := rotten_oranges + rotten_bananas
    let total_fruits := (oranges + bananas : ℚ)
    let good_fruits := total_fruits - total_rotten
    let percentage_good_fruits := (good_fruits / total_fruits) * 100
    percentage_good_fruits = 88.6 :=
by
    sorry

end NUMINAMATH_GPT_percentage_fruits_in_good_condition_l922_92243


namespace NUMINAMATH_GPT_log_identity_l922_92206

theorem log_identity (a b : ℝ) (h1 : a = Real.log 144 / Real.log 4) (h2 : b = Real.log 12 / Real.log 2) : a = b := 
by
  sorry

end NUMINAMATH_GPT_log_identity_l922_92206


namespace NUMINAMATH_GPT_shortest_path_octahedron_l922_92217

theorem shortest_path_octahedron 
  (edge_length : ℝ) (h : edge_length = 2) 
  (d : ℝ) : d = 2 :=
by
  sorry

end NUMINAMATH_GPT_shortest_path_octahedron_l922_92217


namespace NUMINAMATH_GPT_total_revenue_is_correct_l922_92294

-- Define the constants and conditions
def price_of_jeans : ℕ := 11
def price_of_tees : ℕ := 8
def quantity_of_tees_sold : ℕ := 7
def quantity_of_jeans_sold : ℕ := 4

-- Define the total revenue calculation
def total_revenue : ℕ :=
  (price_of_tees * quantity_of_tees_sold) +
  (price_of_jeans * quantity_of_jeans_sold)

-- The theorem to prove
theorem total_revenue_is_correct : total_revenue = 100 := 
by
  -- Proof is omitted for now
  sorry

end NUMINAMATH_GPT_total_revenue_is_correct_l922_92294


namespace NUMINAMATH_GPT_tammy_investment_change_l922_92268

-- Defining initial investment, losses, and gains
def initial_investment : ℝ := 100
def first_year_loss : ℝ := 0.10
def second_year_gain : ℝ := 0.25

-- Defining the final amount after two years
def final_amount (initial_investment : ℝ) (first_year_loss : ℝ) (second_year_gain : ℝ) : ℝ :=
  let remaining_after_first_year := initial_investment * (1 - first_year_loss)
  remaining_after_first_year * (1 + second_year_gain)

-- Statement to prove
theorem tammy_investment_change :
  let percentage_change := ((final_amount initial_investment first_year_loss second_year_gain - initial_investment) / initial_investment) * 100
  percentage_change = 12.5 :=
by
  sorry

end NUMINAMATH_GPT_tammy_investment_change_l922_92268


namespace NUMINAMATH_GPT_sam_runs_more_than_sarah_sue_runs_less_than_sarah_l922_92219

-- Definitions based on the problem conditions
def street_width : ℝ := 25
def block_side_length : ℝ := 500
def sarah_perimeter : ℝ := 4 * block_side_length
def sam_perimeter : ℝ := 4 * (block_side_length + 2 * street_width)
def sue_perimeter : ℝ := 4 * (block_side_length - 2 * street_width)

-- The proof problem statements
theorem sam_runs_more_than_sarah : sam_perimeter - sarah_perimeter = 200 := by
  sorry

theorem sue_runs_less_than_sarah : sarah_perimeter - sue_perimeter = 200 := by
  sorry

end NUMINAMATH_GPT_sam_runs_more_than_sarah_sue_runs_less_than_sarah_l922_92219


namespace NUMINAMATH_GPT_min_hypotenuse_of_right_triangle_l922_92233

theorem min_hypotenuse_of_right_triangle (a b c k : ℝ) (h₁ : k = a + b + c) (h₂ : a^2 + b^2 = c^2) : 
  c ≥ (Real.sqrt 2 - 1) * k := 
sorry

end NUMINAMATH_GPT_min_hypotenuse_of_right_triangle_l922_92233


namespace NUMINAMATH_GPT_files_per_folder_l922_92244

theorem files_per_folder
    (initial_files : ℕ)
    (deleted_files : ℕ)
    (folders : ℕ)
    (remaining_files : ℕ)
    (files_per_folder : ℕ)
    (initial_files_eq : initial_files = 93)
    (deleted_files_eq : deleted_files = 21)
    (folders_eq : folders = 9)
    (remaining_files_eq : remaining_files = initial_files - deleted_files)
    (files_per_folder_eq : files_per_folder = remaining_files / folders) :
    files_per_folder = 8 :=
by
    -- Here, sorry is used to skip the actual proof steps 
    sorry

end NUMINAMATH_GPT_files_per_folder_l922_92244


namespace NUMINAMATH_GPT_question1_question2_l922_92224

def f (x : ℝ) : ℝ := |x + 7| + |x - 1|

theorem question1 (x : ℝ) : ∀ m : ℝ, (∀ x : ℝ, f x ≥ m) → m ≤ 8 :=
by sorry

theorem question2 (x : ℝ) : (∀ x : ℝ, |x - 3| - 2 * x ≤ 2 * 8 - 12) ↔ (x ≥ -1/3) :=
by sorry

end NUMINAMATH_GPT_question1_question2_l922_92224


namespace NUMINAMATH_GPT_f_at_63_l922_92259

-- Define the function f: ℤ → ℤ with given properties
def f : ℤ → ℤ :=
  sorry -- Placeholder, as we are only stating the problem, not the solution

-- Conditions
axiom f_at_1 : f 1 = 6
axiom f_eq : ∀ x : ℤ, f (2 * x + 1) = 3 * f x

-- The goal is to prove f(63) = 1458
theorem f_at_63 : f 63 = 1458 :=
  sorry

end NUMINAMATH_GPT_f_at_63_l922_92259


namespace NUMINAMATH_GPT_calculateDifferentialSavings_l922_92237

/-- 
Assumptions for the tax brackets and deductions/credits.
-/
def taxBracketsCurrent (income : ℕ) : ℕ :=
  if income ≤ 15000 then
    income * 15 / 100
  else if income ≤ 45000 then
    15000 * 15 / 100 + (income - 15000) * 42 / 100
  else
    15000 * 15 / 100 + (45000 - 15000) * 42 / 100 + (income - 45000) * 50 / 100

def taxBracketsProposed (income : ℕ) : ℕ :=
  if income ≤ 15000 then
    income * 12 / 100
  else if income ≤ 45000 then
    15000 * 12 / 100 + (income - 15000) * 28 / 100
  else
    15000 * 12 / 100 + (45000 - 15000) * 28 / 100 + (income - 45000) * 50 / 100

def standardDeduction : ℕ := 3000
def childrenCredit (num_children : ℕ) : ℕ := num_children * 1000

def taxableIncome (income : ℕ) : ℕ :=
  income - standardDeduction

def totalTaxLiabilityCurrent (income num_children : ℕ) : ℕ :=
  (taxBracketsCurrent (taxableIncome income)) - (childrenCredit num_children)

def totalTaxLiabilityProposed (income num_children : ℕ) : ℕ :=
  (taxBracketsProposed (taxableIncome income)) - (childrenCredit num_children)

def differentialSavings (income num_children : ℕ) : ℕ :=
  totalTaxLiabilityCurrent income num_children - totalTaxLiabilityProposed income num_children

/-- 
Statement of the Lean 4 proof problem.
-/
theorem calculateDifferentialSavings : differentialSavings 34500 2 = 2760 :=
by
  sorry

end NUMINAMATH_GPT_calculateDifferentialSavings_l922_92237


namespace NUMINAMATH_GPT_original_integer_is_26_l922_92227

theorem original_integer_is_26 (x y z w : ℕ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : 0 < z) (h₄ : 0 < w)
(h₅ : x ≠ y) (h₆ : x ≠ z) (h₇ : x ≠ w) (h₈ : y ≠ z) (h₉ : y ≠ w) (h₁₀ : z ≠ w)
(h₁₁ : (x + y + z) / 3 + w = 34)
(h₁₂ : (x + y + w) / 3 + z = 22)
(h₁₃ : (x + z + w) / 3 + y = 26)
(h₁₄ : (y + z + w) / 3 + x = 18) :
    w = 26 := 
sorry

end NUMINAMATH_GPT_original_integer_is_26_l922_92227


namespace NUMINAMATH_GPT_sara_disproves_tom_l922_92297

-- Define the type and predicate of cards
inductive Card
| K
| M
| card5
| card7
| card8

open Card

-- Define the conditions
def is_consonant : Card → Prop
| K => true
| M => true
| _ => false

def is_odd : Card → Prop
| card5 => true
| card7 => true
| _ => false

def is_even : Card → Prop
| card8 => true
| _ => false

-- Tom's statement
def toms_statement : Prop :=
  ∀ c, is_consonant c → is_odd c

-- The card Sara turns over (card8) to disprove Tom's statement
theorem sara_disproves_tom : is_even card8 ∧ is_consonant card8 → ¬toms_statement :=
by
  sorry

end NUMINAMATH_GPT_sara_disproves_tom_l922_92297


namespace NUMINAMATH_GPT_students_exceed_guinea_pigs_and_teachers_l922_92241

def num_students_per_classroom : Nat := 25
def num_guinea_pigs_per_classroom : Nat := 3
def num_teachers_per_classroom : Nat := 1
def num_classrooms : Nat := 5

def total_students : Nat := num_students_per_classroom * num_classrooms
def total_guinea_pigs : Nat := num_guinea_pigs_per_classroom * num_classrooms
def total_teachers : Nat := num_teachers_per_classroom * num_classrooms
def total_guinea_pigs_and_teachers : Nat := total_guinea_pigs + total_teachers

theorem students_exceed_guinea_pigs_and_teachers :
  total_students - total_guinea_pigs_and_teachers = 105 :=
by
  sorry

end NUMINAMATH_GPT_students_exceed_guinea_pigs_and_teachers_l922_92241


namespace NUMINAMATH_GPT_ariel_age_l922_92287

theorem ariel_age : ∃ A : ℕ, (A + 15 = 4 * A) ∧ A = 5 :=
by
  -- Here we skip the proof
  sorry

end NUMINAMATH_GPT_ariel_age_l922_92287


namespace NUMINAMATH_GPT_solve_equation_l922_92269

theorem solve_equation :
  { x : ℝ | x * (x - 3)^2 * (5 - x) = 0 } = {0, 3, 5} :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l922_92269


namespace NUMINAMATH_GPT_apples_jackie_l922_92254

theorem apples_jackie (A : ℕ) (J : ℕ) (h1 : A = 8) (h2 : J = A + 2) : J = 10 := by
  -- Adam has 8 apples
  sorry

end NUMINAMATH_GPT_apples_jackie_l922_92254


namespace NUMINAMATH_GPT_tangent_line_equation_l922_92200

noncomputable def curve := fun x : ℝ => Real.sin (x + Real.pi / 3)

def tangent_line (x y : ℝ) : Prop :=
  x - 2 * y + Real.sqrt 3 = 0

theorem tangent_line_equation :
  tangent_line 0 (curve 0) := by
  unfold curve tangent_line
  sorry

end NUMINAMATH_GPT_tangent_line_equation_l922_92200


namespace NUMINAMATH_GPT_grunters_win_all_5_games_grunters_win_at_least_one_game_l922_92231

/-- Given that the Grunters have an independent probability of 3/4 of winning any single game, 
and they play 5 games, the probability that the Grunters will win all 5 games is 243/1024. --/
theorem grunters_win_all_5_games :
  (3/4)^5 = 243 / 1024 :=
sorry

/-- Given that the Grunters have an independent probability of 3/4 of winning any single game, 
and they play 5 games, the probability that the Grunters will win at least one game is 1023/1024. --/
theorem grunters_win_at_least_one_game :
  1 - (1/4)^5 = 1023 / 1024 :=
sorry

end NUMINAMATH_GPT_grunters_win_all_5_games_grunters_win_at_least_one_game_l922_92231


namespace NUMINAMATH_GPT_gcd_factorial_l922_92289

theorem gcd_factorial (n m l : ℕ) (h1 : n = 7) (h2 : m = 10) (h3 : l = 4): 
  Nat.gcd (Nat.factorial n) (Nat.factorial m / Nat.factorial l) = 2520 :=
by
  sorry

end NUMINAMATH_GPT_gcd_factorial_l922_92289


namespace NUMINAMATH_GPT_find_radii_l922_92229

theorem find_radii (r R : ℝ) (h₁ : R - r = 2) (h₂ : R + r = 16) : r = 7 ∧ R = 9 := by
  sorry

end NUMINAMATH_GPT_find_radii_l922_92229


namespace NUMINAMATH_GPT_largest_n_satisfying_inequality_l922_92262

theorem largest_n_satisfying_inequality : 
  ∃ (n : ℕ), (∀ k : ℕ, (8 : ℚ) / 15 < n / (n + k) ∧ n / (n + k) < (7 : ℚ) / 13) ∧ 
  ∀ n' : ℕ, (∀ k : ℕ, (8 : ℚ) / 15 < n' / (n' + k) ∧ n' / (n' + k) < (7 : ℚ) / 13) → n' ≤ n :=
sorry

end NUMINAMATH_GPT_largest_n_satisfying_inequality_l922_92262


namespace NUMINAMATH_GPT_speed_rowing_upstream_l922_92230

theorem speed_rowing_upstream (V_m V_down : ℝ) (V_s V_up : ℝ)
  (h1 : V_m = 28) (h2 : V_down = 30) (h3 : V_down = V_m + V_s) (h4 : V_up = V_m - V_s) : 
  V_up = 26 :=
by
  sorry

end NUMINAMATH_GPT_speed_rowing_upstream_l922_92230


namespace NUMINAMATH_GPT_intersection_A_B_l922_92223

def A : Set ℤ := {-2, 0, 2}
def B : Set ℤ := {x | x^2 - x - 2 = 0}

theorem intersection_A_B : A ∩ B = {2} := by
  -- Proof to be filled
  sorry

end NUMINAMATH_GPT_intersection_A_B_l922_92223


namespace NUMINAMATH_GPT_altitude_eqn_median_eqn_l922_92249

def Point := (ℝ × ℝ)

def A : Point := (4, 0)
def B : Point := (6, 7)
def C : Point := (0, 3)

theorem altitude_eqn (B C: Point) : 
  ∃ (k b : ℝ), (b = 6) ∧ (k = - 3 / 2) ∧ (∀ x y : ℝ, y = k * x + b →
  3 * x + 2 * y - 12 = 0)
:=
sorry

theorem median_eqn (A B C : Point) :
  ∃ (k b : ℝ), (b = 20) ∧ (k = -3/5) ∧ (∀ x y : ℝ, y = k * x + b →
  5 * x + y - 20 = 0)
:=
sorry

end NUMINAMATH_GPT_altitude_eqn_median_eqn_l922_92249


namespace NUMINAMATH_GPT_product_of_digits_base8_of_12345_is_0_l922_92261

def base8_representation (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else Nat.digits 8 n 

def product_of_digits (digits : List ℕ) : ℕ :=
  digits.foldl (· * ·) 1

theorem product_of_digits_base8_of_12345_is_0 :
  product_of_digits (base8_representation 12345) = 0 := 
sorry

end NUMINAMATH_GPT_product_of_digits_base8_of_12345_is_0_l922_92261


namespace NUMINAMATH_GPT_prime_gt_three_times_n_l922_92265

def nth_prime (n : ℕ) : ℕ :=
  -- Define the nth prime function, can use mathlib functionality
  sorry

theorem prime_gt_three_times_n (n : ℕ) (h : 12 ≤ n) : nth_prime n > 3 * n :=
  sorry

end NUMINAMATH_GPT_prime_gt_three_times_n_l922_92265


namespace NUMINAMATH_GPT_largest_multiple_of_11_lt_neg150_l922_92207

theorem largest_multiple_of_11_lt_neg150 : ∃ (x : ℤ), (x % 11 = 0) ∧ (x < -150) ∧ (∀ y : ℤ, y % 11 = 0 → y < -150 → y ≤ x) ∧ x = -154 :=
by
  sorry

end NUMINAMATH_GPT_largest_multiple_of_11_lt_neg150_l922_92207


namespace NUMINAMATH_GPT_angle_bisector_coordinates_distance_to_x_axis_l922_92246

structure Point where
  x : ℝ
  y : ℝ

def M (m : ℝ) : Point :=
  ⟨m - 1, 2 * m + 3⟩

theorem angle_bisector_coordinates (m : ℝ) :
  (M m = ⟨-5, -5⟩) ∨ (M m = ⟨-(5/3), 5/3⟩) := sorry

theorem distance_to_x_axis (m : ℝ) :
  (|2 * m + 3| = 1) → (M m = ⟨-2, 1⟩) ∨ (M m = ⟨-3, -1⟩) := sorry

end NUMINAMATH_GPT_angle_bisector_coordinates_distance_to_x_axis_l922_92246


namespace NUMINAMATH_GPT_solve_logarithmic_inequality_l922_92210

theorem solve_logarithmic_inequality :
  {x : ℝ | 2 * (Real.log x / Real.log 0.5)^2 + 9 * (Real.log x / Real.log 0.5) + 9 ≤ 0} = 
  {x : ℝ | 2 * Real.sqrt 2 ≤ x ∧ x ≤ 8} :=
sorry

end NUMINAMATH_GPT_solve_logarithmic_inequality_l922_92210


namespace NUMINAMATH_GPT_relativ_prime_and_divisible_exists_l922_92253

theorem relativ_prime_and_divisible_exists
  (a b c : ℕ)
  (a_pos : 0 < a)
  (b_pos : 0 < b)
  (c_pos : 0 < c) :
  ∃ r s : ℕ, Nat.gcd r s = 1 ∧ 0 < r ∧ 0 < s ∧ c ∣ (a * r + b * s) :=
by
  sorry

end NUMINAMATH_GPT_relativ_prime_and_divisible_exists_l922_92253


namespace NUMINAMATH_GPT_skitties_remainder_l922_92272

theorem skitties_remainder (m : ℕ) (h : m % 7 = 5) : (4 * m) % 7 = 6 :=
sorry

end NUMINAMATH_GPT_skitties_remainder_l922_92272


namespace NUMINAMATH_GPT_initial_people_on_train_l922_92232

theorem initial_people_on_train 
    (P : ℕ)
    (h1 : 116 = P - 4)
    (h2 : P = 120)
    : 
    P = 116 + 4 := by
have h3 : P = 120 := by sorry
exact h3

end NUMINAMATH_GPT_initial_people_on_train_l922_92232


namespace NUMINAMATH_GPT_sin_alpha_plus_half_pi_l922_92295

theorem sin_alpha_plus_half_pi (α : ℝ) 
  (h1 : Real.tan (α - Real.pi) = 3 / 4)
  (h2 : α ∈ Set.Ioo (Real.pi / 2) (3 * Real.pi / 2)) : 
  Real.sin (α + Real.pi / 2) = -4 / 5 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_sin_alpha_plus_half_pi_l922_92295


namespace NUMINAMATH_GPT_tangent_line_passes_through_origin_l922_92218

noncomputable def curve (α : ℝ) (x : ℝ) : ℝ := x^α + 1

theorem tangent_line_passes_through_origin (α : ℝ)
  (h_tangent : ∀ (x : ℝ), curve α 1 + (α * (x - 1)) - 2 = curve α x) :
  α = 2 :=
sorry

end NUMINAMATH_GPT_tangent_line_passes_through_origin_l922_92218


namespace NUMINAMATH_GPT_area_of_rectangle_perimeter_of_rectangle_l922_92215

-- Define the input conditions
variables (AB AC BC : ℕ)
def is_right_triangle (a b c : ℕ) : Prop := a * a + b * b = c * c
def area_rect (l w : ℕ) : ℕ := l * w
def perimeter_rect (l w : ℕ) : ℕ := 2 * (l + w)

-- Given the conditions for the problem
axiom AB_eq_15 : AB = 15
axiom AC_eq_17 : AC = 17
axiom right_triangle : is_right_triangle AB BC AC

-- Prove the area and perimeter of the rectangle
theorem area_of_rectangle : area_rect AB BC = 120 := by sorry

theorem perimeter_of_rectangle : perimeter_rect AB BC = 46 := by sorry

end NUMINAMATH_GPT_area_of_rectangle_perimeter_of_rectangle_l922_92215


namespace NUMINAMATH_GPT_master_codes_count_l922_92216

def num_colors : ℕ := 7
def num_slots : ℕ := 5

theorem master_codes_count : num_colors ^ num_slots = 16807 := by
  sorry

end NUMINAMATH_GPT_master_codes_count_l922_92216


namespace NUMINAMATH_GPT_waiter_income_fraction_l922_92214

theorem waiter_income_fraction (S T : ℝ) (hT : T = 5/4 * S) :
  T / (S + T) = 5 / 9 :=
by
  sorry

end NUMINAMATH_GPT_waiter_income_fraction_l922_92214


namespace NUMINAMATH_GPT_min_m_quad_eq_integral_solutions_l922_92251

theorem min_m_quad_eq_integral_solutions :
  (∃ m : ℕ, (∀ x : ℤ, 10 * x ^ 2 - m * x + 420 = 0 → ∃ p q : ℤ, p + q = m / 10 ∧ p * q = 42) ∧ m > 0) →
  (∃ m : ℕ, m = 130 ∧ (∀ x : ℤ, 10 * x ^ 2 - m * x + 420 = 0 → ∃ p q : ℤ, p + q = m / 10 ∧ p * q = 42)) :=
by
  sorry

end NUMINAMATH_GPT_min_m_quad_eq_integral_solutions_l922_92251


namespace NUMINAMATH_GPT_gcd_8251_6105_l922_92278

theorem gcd_8251_6105 : Nat.gcd 8251 6105 = 37 := by
  sorry

end NUMINAMATH_GPT_gcd_8251_6105_l922_92278


namespace NUMINAMATH_GPT_cost_relationship_l922_92284

variable {α : Type} [LinearOrderedField α]
variables (bananas_cost apples_cost pears_cost : α)

theorem cost_relationship :
  (5 * bananas_cost = 3 * apples_cost) →
  (10 * apples_cost = 6 * pears_cost) →
  (25 * bananas_cost = 9 * pears_cost) := by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_cost_relationship_l922_92284


namespace NUMINAMATH_GPT_Fedya_third_l922_92291

/-- Definitions for order of children's arrival -/
inductive Child
| Roman | Fedya | Liza | Katya | Andrew

open Child

def arrival_order (order : Child → ℕ) : Prop :=
  order Liza > order Roman ∧
  order Katya < order Liza ∧
  order Fedya = order Katya + 1 ∧
  order Katya ≠ 1

/-- Theorem stating that Fedya is third based on the given conditions -/
theorem Fedya_third (order : Child → ℕ) (H : arrival_order order) : order Fedya = 3 :=
sorry

end NUMINAMATH_GPT_Fedya_third_l922_92291


namespace NUMINAMATH_GPT_correct_answers_max_l922_92260

def max_correct_answers (c w b : ℕ) : Prop :=
  c + w + b = 25 ∧ 4 * c - 3 * w = 40

theorem correct_answers_max : ∃ c w b : ℕ, max_correct_answers c w b ∧ ∀ c', max_correct_answers c' w b → c' ≤ 13 :=
by
  sorry

end NUMINAMATH_GPT_correct_answers_max_l922_92260


namespace NUMINAMATH_GPT_extremum_value_and_min_on_interval_l922_92205

noncomputable def f (a b c x : ℝ) : ℝ := a * x^3 + b * x + c

theorem extremum_value_and_min_on_interval
  (a b c : ℝ)
  (h1_eq : 12 * a + b = 0)
  (h2_eq : 4 * a + b = -8)
  (h_max : 16 + c = 28) :
  min (min (f a b c (-3)) (f a b c 3)) (f a b c 2) = -4 :=
by sorry

end NUMINAMATH_GPT_extremum_value_and_min_on_interval_l922_92205


namespace NUMINAMATH_GPT_exists_eight_integers_sum_and_product_eight_l922_92201

theorem exists_eight_integers_sum_and_product_eight :
  ∃ (a1 a2 a3 a4 a5 a6 a7 a8 : ℤ), 
  a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 = 8 ∧ 
  a1 * a2 * a3 * a4 * a5 * a6 * a7 * a8 = 8 :=
by
  -- The existence proof can be constructed here
  sorry

end NUMINAMATH_GPT_exists_eight_integers_sum_and_product_eight_l922_92201


namespace NUMINAMATH_GPT_julios_grape_soda_l922_92298

variable (a b c d e f g : ℕ)
variable (ha : a = 4)
variable (hc : c = 1)
variable (hd : d = 3)
variable (he : e = 2)
variable (hf : f = 14)
variable (hg : g = 7)

theorem julios_grape_soda : 
  let julios_soda := a * e + b * e
  let mateos_soda := (c + d) * e
  julios_soda = mateos_soda + f
  → b = g := by
  sorry

end NUMINAMATH_GPT_julios_grape_soda_l922_92298
