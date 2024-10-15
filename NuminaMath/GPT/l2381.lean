import Mathlib

namespace NUMINAMATH_GPT_simplify_fraction_l2381_238181

variable (d : ℤ)

theorem simplify_fraction (d : ℤ) : (6 + 4 * d) / 9 + 3 = (33 + 4 * d) / 9 := 
by 
  sorry

end NUMINAMATH_GPT_simplify_fraction_l2381_238181


namespace NUMINAMATH_GPT_pencils_brought_l2381_238169

-- Given conditions
variables (A B : ℕ)

-- There are 7 people in total
def total_people : Prop := A + B = 7

-- 11 charts in total
def total_charts : Prop := A + 2 * B = 11

-- Question: Total pencils
def total_pencils : ℕ := 2 * A + B

-- Statement to be proved
theorem pencils_brought
  (h1 : total_people A B)
  (h2 : total_charts A B) :
  total_pencils A B = 10 := by
  sorry

end NUMINAMATH_GPT_pencils_brought_l2381_238169


namespace NUMINAMATH_GPT_exists_distinct_a_b_all_P_balanced_P_balanced_implies_a_eq_b_l2381_238179

-- Define the notion of a balanced integer.
def isBalanced (N : ℕ) : Prop :=
  N = 1 ∨ ∃ (p : ℕ) (k : ℕ), Nat.Prime p ∧ N = p ^ (2 * k)

-- Define the polynomial P(x) = (x + a)(x + b)
def P (a b x : ℕ) : ℕ := (x + a) * (x + b)

theorem exists_distinct_a_b_all_P_balanced :
  ∃ (a b : ℕ), a ≠ b ∧ ∀ (n : ℕ), 1 ≤ n ∧ n ≤ 50 → isBalanced (P a b n) :=
sorry

theorem P_balanced_implies_a_eq_b (a b : ℕ) :
  (∀ n : ℕ, isBalanced (P a b n)) → a = b :=
sorry

end NUMINAMATH_GPT_exists_distinct_a_b_all_P_balanced_P_balanced_implies_a_eq_b_l2381_238179


namespace NUMINAMATH_GPT_maximum_term_of_sequence_l2381_238155

open Real

noncomputable def seq (n : ℕ) : ℝ := n / (n^2 + 81)

theorem maximum_term_of_sequence : ∃ n : ℕ, seq n = 1 / 18 ∧ ∀ k : ℕ, seq k ≤ 1 / 18 :=
by
  sorry

end NUMINAMATH_GPT_maximum_term_of_sequence_l2381_238155


namespace NUMINAMATH_GPT_dollars_tina_l2381_238199

open Real

theorem dollars_tina (P Q R S T : ℤ)
  (h1 : abs (P - Q) = 21)
  (h2 : abs (Q - R) = 9)
  (h3 : abs (R - S) = 7)
  (h4 : abs (S - T) = 6)
  (h5 : abs (T - P) = 13)
  (h6 : P + Q + R + S + T = 86) :
  T = 16 :=
sorry

end NUMINAMATH_GPT_dollars_tina_l2381_238199


namespace NUMINAMATH_GPT_store_loses_out_l2381_238120

theorem store_loses_out (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a ≠ b) (x y : ℝ)
    (h1 : a = b * x) (h2 : b = a * y) : x + y > 2 :=
by
  sorry

end NUMINAMATH_GPT_store_loses_out_l2381_238120


namespace NUMINAMATH_GPT_additional_boxes_needed_l2381_238122

theorem additional_boxes_needed
  (total_chocolates : ℕ)
  (chocolates_not_in_box : ℕ)
  (boxes_filled : ℕ)
  (friend_brought_chocolates : ℕ)
  (chocolates_per_box : ℕ)
  (h1 : total_chocolates = 50)
  (h2 : chocolates_not_in_box = 5)
  (h3 : boxes_filled = 3)
  (h4 : friend_brought_chocolates = 25)
  (h5 : chocolates_per_box = 15) :
  (chocolates_not_in_box + friend_brought_chocolates) / chocolates_per_box = 2 :=
by
  sorry
  
end NUMINAMATH_GPT_additional_boxes_needed_l2381_238122


namespace NUMINAMATH_GPT_ratio_of_andy_age_in_5_years_to_rahim_age_l2381_238118

def rahim_age_now : ℕ := 6
def andy_age_now : ℕ := rahim_age_now + 1
def andy_age_in_5_years : ℕ := andy_age_now + 5
def ratio (a b : ℕ) : ℕ := a / b

theorem ratio_of_andy_age_in_5_years_to_rahim_age : ratio andy_age_in_5_years rahim_age_now = 2 := by
  sorry

end NUMINAMATH_GPT_ratio_of_andy_age_in_5_years_to_rahim_age_l2381_238118


namespace NUMINAMATH_GPT_bridge_weight_requirement_l2381_238128

def weight_soda_can : ℕ := 12
def weight_empty_soda_can : ℕ := 2
def num_soda_cans : ℕ := 6

def weight_empty_other_can : ℕ := 3
def num_other_cans : ℕ := 2

def wind_force_eq_soda_cans : ℕ := 2

def total_weight_bridge_must_hold : ℕ :=
  weight_soda_can * num_soda_cans + weight_empty_soda_can * num_soda_cans +
  weight_empty_other_can * num_other_cans +
  wind_force_eq_soda_cans * (weight_soda_can + weight_empty_soda_can)

theorem bridge_weight_requirement :
  total_weight_bridge_must_hold = 118 :=
by
  unfold total_weight_bridge_must_hold weight_soda_can weight_empty_soda_can num_soda_cans
    weight_empty_other_can num_other_cans wind_force_eq_soda_cans
  sorry

end NUMINAMATH_GPT_bridge_weight_requirement_l2381_238128


namespace NUMINAMATH_GPT_range_of_m_l2381_238144

theorem range_of_m (m : ℝ) : (∀ x : ℝ, m * x ^ 2 - m * x + 1 > 0) ↔ (0 ≤ m ∧ m < 4) :=
sorry

end NUMINAMATH_GPT_range_of_m_l2381_238144


namespace NUMINAMATH_GPT_percentage_of_students_owning_only_cats_l2381_238109

theorem percentage_of_students_owning_only_cats
  (total_students : ℕ) (students_owning_dogs : ℕ) (students_owning_cats : ℕ) (students_owning_both : ℕ)
  (h1 : total_students = 500) (h2 : students_owning_dogs = 200) (h3 : students_owning_cats = 100) (h4 : students_owning_both = 50) :
  ((students_owning_cats - students_owning_both) * 100 / total_students) = 10 :=
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_percentage_of_students_owning_only_cats_l2381_238109


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l2381_238154

theorem arithmetic_sequence_sum
  (a l : ℤ) (n d : ℤ)
  (h1 : a = -5) (h2 : l = 40) (h3 : d = 5)
  (h4 : l = a + (n - 1) * d) :
  (n / 2) * (a + l) = 175 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l2381_238154


namespace NUMINAMATH_GPT_num_seven_digit_numbers_l2381_238161

theorem num_seven_digit_numbers (a b c d e f g : ℕ)
  (h1 : a * b * c = 30)
  (h2 : c * d * e = 7)
  (h3 : e * f * g = 15) :
  ∃ n : ℕ, n = 4 := 
sorry

end NUMINAMATH_GPT_num_seven_digit_numbers_l2381_238161


namespace NUMINAMATH_GPT_evaluate_expression_l2381_238101

noncomputable def a : ℝ := Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 7
noncomputable def b : ℝ := -Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 7
noncomputable def c : ℝ := Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 7
noncomputable def d : ℝ := -Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 7

theorem evaluate_expression : ( (1 / a) + (1 / b) + (1 / c) + (1 / d) ) ^ 2 = (7 / 49) :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2381_238101


namespace NUMINAMATH_GPT_consecutive_product_plus_one_l2381_238148

theorem consecutive_product_plus_one (n : ℕ) :
  n * (n + 1) * (n + 2) * (n + 3) + 1 = (n^2 + 3 * n + 1)^2 :=
by
  sorry

end NUMINAMATH_GPT_consecutive_product_plus_one_l2381_238148


namespace NUMINAMATH_GPT_Maria_score_in_fourth_quarter_l2381_238146

theorem Maria_score_in_fourth_quarter (q1 q2 q3 : ℕ) 
  (hq1 : q1 = 84) 
  (hq2 : q2 = 82) 
  (hq3 : q3 = 80) 
  (average_requirement : ℕ) 
  (havg_req : average_requirement = 85) :
  ∃ q4 : ℕ, q4 ≥ 94 ∧ (q1 + q2 + q3 + q4) / 4 ≥ average_requirement := 
by 
  sorry 

end NUMINAMATH_GPT_Maria_score_in_fourth_quarter_l2381_238146


namespace NUMINAMATH_GPT_celine_library_charge_l2381_238129

variable (charge_per_day : ℝ) (days_in_may : ℕ) (books_borrowed : ℕ) (days_first_book : ℕ)
          (days_other_books : ℕ) (books_kept : ℕ)

noncomputable def total_charge (charge_per_day : ℝ) (days_first_book : ℕ) 
        (days_other_books : ℕ) (books_kept : ℕ) : ℝ :=
  charge_per_day * days_first_book + charge_per_day * days_other_books * books_kept

theorem celine_library_charge : 
  charge_per_day = 0.50 ∧ days_in_may = 31 ∧ books_borrowed = 3 ∧ days_first_book = 20 ∧
  days_other_books = 31 ∧ books_kept = 2 → 
  total_charge charge_per_day days_first_book days_other_books books_kept = 41.00 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_celine_library_charge_l2381_238129


namespace NUMINAMATH_GPT_hotel_charge_decrease_l2381_238145

theorem hotel_charge_decrease 
  (G R P : ℝ)
  (h1 : R = 1.60 * G)
  (h2 : P = 0.50 * R) :
  (G - P) / G * 100 = 20 := by
sorry

end NUMINAMATH_GPT_hotel_charge_decrease_l2381_238145


namespace NUMINAMATH_GPT_nonnegative_solutions_eq1_l2381_238183

theorem nonnegative_solutions_eq1 : (∃ x : ℝ, 0 ≤ x ∧ x^2 = -6 * x) ∧ (∀ x : ℝ, 0 ≤ x ∧ x^2 = -6 * x → x = 0) := by
  sorry

end NUMINAMATH_GPT_nonnegative_solutions_eq1_l2381_238183


namespace NUMINAMATH_GPT_fractional_eq_k_l2381_238186

open Real

theorem fractional_eq_k (x k : ℝ) (hx0 : x ≠ 0) (hx1 : x ≠ 1) :
  (3 / x + 6 / (x - 1) - (x + k) / (x * (x - 1)) = 0) ↔ k ≠ -3 ∧ k ≠ 5 := 
sorry

end NUMINAMATH_GPT_fractional_eq_k_l2381_238186


namespace NUMINAMATH_GPT_parallel_lines_same_slope_l2381_238100

theorem parallel_lines_same_slope (k : ℝ) : 
  (2*x + y + 1 = 0) ∧ (y = k*x + 3) → (k = -2) := 
by
  sorry

end NUMINAMATH_GPT_parallel_lines_same_slope_l2381_238100


namespace NUMINAMATH_GPT_Sandy_marks_per_correct_sum_l2381_238116

theorem Sandy_marks_per_correct_sum
  (x : ℝ)  -- number of marks Sandy gets for each correct sum
  (marks_lost_per_incorrect : ℝ := 2)  -- 2 marks lost for each incorrect sum, default value is 2
  (total_attempts : ℤ := 30)  -- Sandy attempts 30 sums, default value is 30
  (total_marks : ℝ := 60)  -- Sandy obtains 60 marks, default value is 60
  (correct_sums : ℤ := 24)  -- Sandy got 24 sums correct, default value is 24
  (incorrect_sums := total_attempts - correct_sums) -- incorrect sums are the remaining attempts
  (marks_from_correct := correct_sums * x) -- total marks from the correct sums
  (marks_lost_from_incorrect := incorrect_sums * marks_lost_per_incorrect) -- total marks lost from the incorrect sums
  (total_marks_obtained := marks_from_correct - marks_lost_from_incorrect) -- total marks obtained

  -- The theorem states that x must be 3 given the conditions above
  : total_marks_obtained = total_marks → x = 3 := by sorry

end NUMINAMATH_GPT_Sandy_marks_per_correct_sum_l2381_238116


namespace NUMINAMATH_GPT_camel_cost_l2381_238159

theorem camel_cost
  (C H O E : ℝ)
  (h1 : 10 * C = 24 * H)
  (h2 : 26 * H = 4 * O)
  (h3 : 6 * O = 4 * E)
  (h4 : 10 * E = 170000) :
  C = 4184.62 :=
by sorry

end NUMINAMATH_GPT_camel_cost_l2381_238159


namespace NUMINAMATH_GPT_transformation_correctness_l2381_238108

variable (x x' y y' : ℝ)

-- Conditions
def original_curve : Prop := y^2 = 4
def transformed_curve : Prop := (x'^2)/1 + (y'^2)/4 = 1
def transformation_formula : Prop := (x = 2 * x') ∧ (y = y')

-- Proof Statement
theorem transformation_correctness (h1 : original_curve y) (h2 : transformed_curve x' y') :
  transformation_formula x x' y y' :=
  sorry

end NUMINAMATH_GPT_transformation_correctness_l2381_238108


namespace NUMINAMATH_GPT_amount_for_second_shop_l2381_238103

-- Definitions based on conditions
def books_from_first_shop : Nat := 65
def amount_first_shop : Float := 1160.0
def books_from_second_shop : Nat := 50
def avg_price_per_book : Float := 18.08695652173913
def total_books : Nat := books_from_first_shop + books_from_second_shop
def total_amount_spent : Float := avg_price_per_book * (total_books.toFloat)

-- The Lean statement to prove
theorem amount_for_second_shop : total_amount_spent - amount_first_shop = 920.0 := by
  sorry

end NUMINAMATH_GPT_amount_for_second_shop_l2381_238103


namespace NUMINAMATH_GPT_find_ratio_of_geometric_sequence_l2381_238105

open Real

noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

def arithmetic_sequence (a1 a2 a3 : ℝ) : Prop :=
  2 * a2 = a1 + a3

theorem find_ratio_of_geometric_sequence 
  {a : ℕ → ℝ} {q : ℝ}
  (h_pos : ∀ n, 0 < a n)
  (h_geo : geometric_sequence a q)
  (h_arith : arithmetic_sequence (a 1) ((1/2) * a 3) (2 * a 2)) :
  (a 10) / (a 8) = 3 + 2 * sqrt 2 :=
sorry

end NUMINAMATH_GPT_find_ratio_of_geometric_sequence_l2381_238105


namespace NUMINAMATH_GPT_largest_x_l2381_238184

-- Definitions from the conditions
def eleven_times_less_than_150 (x : ℕ) : Prop := 11 * x < 150

-- Statement of the proof problem
theorem largest_x : ∃ x : ℕ, eleven_times_less_than_150 x ∧ ∀ y : ℕ, eleven_times_less_than_150 y → y ≤ x := 
sorry

end NUMINAMATH_GPT_largest_x_l2381_238184


namespace NUMINAMATH_GPT_total_area_is_71_l2381_238167

-- Define the lengths of the segments
def length_left : ℕ := 7
def length_top : ℕ := 6
def length_middle_1 : ℕ := 2
def length_middle_2 : ℕ := 4
def length_right : ℕ := 1
def length_right_top : ℕ := 5

-- Define the rectangles and their areas
def area_left_rect : ℕ := length_left * length_left
def area_middle_rect : ℕ := length_middle_1 * (length_top - length_left)
def area_right_rect : ℕ := length_middle_2 * length_middle_2

-- Define the total area
def total_area : ℕ := area_left_rect + area_middle_rect + area_right_rect

-- Theorem: The total area of the figure is 71 square units
theorem total_area_is_71 : total_area = 71 := by
  sorry

end NUMINAMATH_GPT_total_area_is_71_l2381_238167


namespace NUMINAMATH_GPT_inequality_solution_l2381_238119

theorem inequality_solution (x : ℝ) (h : x ≠ 1) : (x + 1) * (x + 3) / (x - 1)^2 ≤ 0 ↔ (-3 ≤ x ∧ x ≤ -1) :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l2381_238119


namespace NUMINAMATH_GPT_infinite_series_sum_l2381_238142

theorem infinite_series_sum :
  (∑' n : ℕ, if h : n ≠ 0 then 1 / (n * (n + 1) * (n + 3)) else 0) = 5 / 36 := by
  sorry

end NUMINAMATH_GPT_infinite_series_sum_l2381_238142


namespace NUMINAMATH_GPT_total_weight_of_rhinos_l2381_238172

def white_rhino_weight : ℕ := 5100
def black_rhino_weight : ℕ := 2000

theorem total_weight_of_rhinos :
  7 * white_rhino_weight + 8 * black_rhino_weight = 51700 :=
by
  sorry

end NUMINAMATH_GPT_total_weight_of_rhinos_l2381_238172


namespace NUMINAMATH_GPT_product_of_reds_is_red_sum_of_reds_is_red_l2381_238187

noncomputable def color := ℕ → Prop

variables (white red : color)
variable (r : ℕ)

axiom coloring : ∀ n, white n ∨ red n
axiom exists_white : ∃ n, white n
axiom exists_red : ∃ n, red n
axiom sum_of_white_red_is_white : ∀ m n, white m → red n → white (m + n)
axiom prod_of_white_red_is_red : ∀ m n, white m → red n → red (m * n)

theorem product_of_reds_is_red (m n : ℕ) : red m → red n → red (m * n) :=
sorry

theorem sum_of_reds_is_red (m n : ℕ) : red m → red n → red (m + n) :=
sorry

end NUMINAMATH_GPT_product_of_reds_is_red_sum_of_reds_is_red_l2381_238187


namespace NUMINAMATH_GPT_rope_purchases_l2381_238175

theorem rope_purchases (last_week_rope_feet : ℕ) (less_rope : ℕ) (feet_to_inches : ℕ) 
  (h1 : last_week_rope_feet = 6) 
  (h2 : less_rope = 4) 
  (h3 : feet_to_inches = 12) : 
  (last_week_rope_feet * feet_to_inches) + ((last_week_rope_feet - less_rope) * feet_to_inches) = 96 := 
by
  sorry

end NUMINAMATH_GPT_rope_purchases_l2381_238175


namespace NUMINAMATH_GPT_turtle_population_estimate_l2381_238174

theorem turtle_population_estimate :
  (tagged_in_june = 90) →
  (sample_november = 50) →
  (tagged_november = 4) →
  (natural_causes_removal = 0.30) →
  (new_hatchlings_november = 0.50) →
  estimate = 563 :=
by
  intros tagged_in_june sample_november tagged_november natural_causes_removal new_hatchlings_november
  sorry

end NUMINAMATH_GPT_turtle_population_estimate_l2381_238174


namespace NUMINAMATH_GPT_tetrahedron_dihedral_face_areas_l2381_238107

variables {S₁ S₂ a b : ℝ} {α φ : ℝ}

theorem tetrahedron_dihedral_face_areas :
  S₁^2 + S₂^2 - 2 * S₁ * S₂ * Real.cos α = (a * b * Real.sin φ / 4)^2 :=
sorry

end NUMINAMATH_GPT_tetrahedron_dihedral_face_areas_l2381_238107


namespace NUMINAMATH_GPT_verify_sub_by_add_verify_sub_by_sub_verify_mul_by_div1_verify_mul_by_div2_verify_mul_by_mul_l2381_238158

variable (A B C P M N : ℝ)

-- Verification of Subtraction by Addition
theorem verify_sub_by_add (h : A - B = C) : C + B = A :=
sorry

-- Verification of Subtraction by Subtraction
theorem verify_sub_by_sub (h : A - B = C) : A - C = B :=
sorry

-- Verification of Multiplication by Division (1)
theorem verify_mul_by_div1 (h : M * N = P) : P / N = M :=
sorry

-- Verification of Multiplication by Division (2)
theorem verify_mul_by_div2 (h : M * N = P) : P / M = N :=
sorry

-- Verification of Multiplication by Multiplication
theorem verify_mul_by_mul (h : M * N = P) : M * N = P :=
sorry

end NUMINAMATH_GPT_verify_sub_by_add_verify_sub_by_sub_verify_mul_by_div1_verify_mul_by_div2_verify_mul_by_mul_l2381_238158


namespace NUMINAMATH_GPT_prime_9_greater_than_perfect_square_l2381_238165

theorem prime_9_greater_than_perfect_square (p : ℕ) (hp : Nat.Prime p) :
  ∃ n m : ℕ, p - 9 = n^2 ∧ p + 2 = m^2 ∧ p = 23 :=
by
  sorry

end NUMINAMATH_GPT_prime_9_greater_than_perfect_square_l2381_238165


namespace NUMINAMATH_GPT_find_x_l2381_238156

theorem find_x :
  ∀ (x y z w : ℕ), 
    x = y + 5 →
    y = z + 10 →
    z = w + 20 →
    w = 80 →
    x = 115 :=
by
  intros x y z w h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_find_x_l2381_238156


namespace NUMINAMATH_GPT_instantaneous_velocity_at_1_2_l2381_238170

def equation_of_motion (t : ℝ) : ℝ := 2 * (1 - t^2)

def velocity_function (t : ℝ) : ℝ := -4 * t

theorem instantaneous_velocity_at_1_2 :
  velocity_function 1.2 = -4.8 :=
by sorry

end NUMINAMATH_GPT_instantaneous_velocity_at_1_2_l2381_238170


namespace NUMINAMATH_GPT_pentagon_stack_valid_sizes_l2381_238198

def valid_stack_size (n : ℕ) : Prop :=
  ¬ (n = 1) ∧ ¬ (n = 3)

theorem pentagon_stack_valid_sizes (n : ℕ) :
  valid_stack_size n :=
sorry

end NUMINAMATH_GPT_pentagon_stack_valid_sizes_l2381_238198


namespace NUMINAMATH_GPT_percentage_of_goals_by_two_players_l2381_238139

-- Definitions from conditions
def total_goals_league := 300
def goals_per_player := 30
def number_of_players := 2

-- Mathematically equivalent proof problem
theorem percentage_of_goals_by_two_players :
  let combined_goals := number_of_players * goals_per_player
  let percentage := (combined_goals / total_goals_league : ℝ) * 100 
  percentage = 20 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_goals_by_two_players_l2381_238139


namespace NUMINAMATH_GPT_total_scoops_l2381_238196

-- Define the conditions as variables
def flourCups := 3
def sugarCups := 2
def scoopSize := 1/3

-- Define what needs to be proved, i.e., the total amount of scoops needed
theorem total_scoops (flourCups sugarCups : ℚ) (scoopSize : ℚ) : 
  (flourCups / scoopSize) + (sugarCups / scoopSize) = 15 := 
by
  sorry

end NUMINAMATH_GPT_total_scoops_l2381_238196


namespace NUMINAMATH_GPT_cos2_add_3sin2_eq_2_l2381_238132

theorem cos2_add_3sin2_eq_2 (x : ℝ) (hx : -20 < x ∧ x < 100) (h : Real.cos x ^ 2 + 3 * Real.sin x ^ 2 = 2) : 
  ∃ n : ℕ, n = 38 := 
sorry

end NUMINAMATH_GPT_cos2_add_3sin2_eq_2_l2381_238132


namespace NUMINAMATH_GPT_find_n_lcm_l2381_238104

theorem find_n_lcm (m n : ℕ) (h1 : Nat.lcm m n = 690) (h2 : n ≥ 100) (h3 : n < 1000) (h4 : ¬ (3 ∣ n)) (h5 : ¬ (2 ∣ m)) : n = 230 :=
sorry

end NUMINAMATH_GPT_find_n_lcm_l2381_238104


namespace NUMINAMATH_GPT_travel_time_l2381_238136

-- Definitions from problem conditions
def scale := 3000000
def map_distance_cm := 6
def conversion_factor_cm_to_km := 30000 -- derived from 1 cm on the map equals 30,000 km in reality
def speed_kmh := 30

-- The travel time we want to prove
theorem travel_time : (map_distance_cm * conversion_factor_cm_to_km / speed_kmh) = 6000 := 
by
  sorry

end NUMINAMATH_GPT_travel_time_l2381_238136


namespace NUMINAMATH_GPT_converse_true_inverse_true_contrapositive_false_sufficiency_necessity_l2381_238138

-- Define the original proposition with conditions
def prop : Prop := ∀ (m n : ℝ), m ≤ 0 ∨ n ≤ 0 → m + n ≤ 0

-- Identify converse, inverse, and contrapositive
def converse : Prop := ∀ (m n : ℝ), m + n ≤ 0 → m ≤ 0 ∨ n ≤ 0
def inverse : Prop := ∀ (m n : ℝ), m > 0 ∧ n > 0 → m + n > 0
def contrapositive : Prop := ∀ (m n : ℝ), m + n > 0 → m > 0 ∧ n > 0

-- Identifying the conditions of sufficiency and necessity
def necessary_but_not_sufficient (p q : Prop) : Prop := 
  (¬p → ¬q) ∧ (q → p) ∧ ¬(p → q)

-- Prove or provide the statements
theorem converse_true : converse := sorry
theorem inverse_true : inverse := sorry
theorem contrapositive_false : ¬contrapositive := sorry
theorem sufficiency_necessity : necessary_but_not_sufficient 
  (∀ (m n : ℝ), m ≤ 0 ∨ n ≤ 0) 
  (∀ (m n : ℝ), m + n ≤ 0) := sorry

end NUMINAMATH_GPT_converse_true_inverse_true_contrapositive_false_sufficiency_necessity_l2381_238138


namespace NUMINAMATH_GPT_find_n_l2381_238110

noncomputable def imaginary_unit : ℂ := Complex.I

theorem find_n (n : ℝ) (h : (2 : ℂ) / (1 - imaginary_unit) = 1 + n * imaginary_unit) : n = 1 :=
sorry

end NUMINAMATH_GPT_find_n_l2381_238110


namespace NUMINAMATH_GPT_triangle_area_is_correct_l2381_238180

-- Define the points
def point1 : (ℝ × ℝ) := (0, 3)
def point2 : (ℝ × ℝ) := (5, 0)
def point3 : (ℝ × ℝ) := (0, 6)
def point4 : (ℝ × ℝ) := (4, 0)

-- Define a function to calculate the area based on the intersection points
noncomputable def area_of_triangle (p1 p2 p3 p4 : ℝ × ℝ) : ℝ :=
  let slope1 := (p2.2 - p1.2) / (p2.1 - p1.1)
  let intercept1 := p1.2 - slope1 * p1.1
  let slope2 := (p4.2 - p3.2) / (p4.1 - p3.1)
  let intercept2 := p3.2 - slope2 * p3.1
  let x_intersect := (intercept2 - intercept1) / (slope1 - slope2)
  let y_intersect := slope1 * x_intersect + intercept1
  let base := x_intersect
  let height := y_intersect
  (1 / 2) * base * height

-- The proof problem statement in Lean
theorem triangle_area_is_correct :
  area_of_triangle point1 point2 point3 point4 = 5 / 3 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_is_correct_l2381_238180


namespace NUMINAMATH_GPT_intersection_of_log_functions_l2381_238188

theorem intersection_of_log_functions : 
  ∃ x : ℝ, (3 * Real.log x = Real.log (3 * x)) ∧ x = Real.sqrt 3 := 
by 
  sorry

end NUMINAMATH_GPT_intersection_of_log_functions_l2381_238188


namespace NUMINAMATH_GPT_translation_is_elevator_l2381_238114

-- Definitions representing the conditions
def P_A : Prop := true  -- The movement of elevators constitutes translation.
def P_B : Prop := false -- Swinging on a swing does not constitute translation.
def P_C : Prop := false -- Closing an open textbook does not constitute translation.
def P_D : Prop := false -- The swinging of a pendulum does not constitute translation.

-- The goal is to prove that Option A is the phenomenon that belongs to translation
theorem translation_is_elevator : P_A ∧ ¬P_B ∧ ¬P_C ∧ ¬P_D :=
by
  sorry -- proof not required

end NUMINAMATH_GPT_translation_is_elevator_l2381_238114


namespace NUMINAMATH_GPT_range_of_b_l2381_238151

theorem range_of_b (b : ℝ) : (¬ ∃ a < 0, a + 1/a > b) → b ≥ -2 := 
by {
  sorry
}

end NUMINAMATH_GPT_range_of_b_l2381_238151


namespace NUMINAMATH_GPT_problem_statement_l2381_238182

def line : Type := sorry
def plane : Type := sorry

def perpendicular (l : line) (p : plane) : Prop := sorry
def parallel (l1 l2 : line) : Prop := sorry

variable (m n : line)
variable (α β : plane)

theorem problem_statement (h1 : perpendicular m α) 
                          (h2 : parallel m n) 
                          (h3 : parallel n β) : 
                          perpendicular α β := 
sorry

end NUMINAMATH_GPT_problem_statement_l2381_238182


namespace NUMINAMATH_GPT_find_third_smallest_three_digit_palindromic_prime_l2381_238157

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def second_smallest_three_digit_palindromic_prime : ℕ :=
  131 -- Given in the problem statement

noncomputable def third_smallest_three_digit_palindromic_prime : ℕ :=
  151 -- Answer obtained from the solution

theorem find_third_smallest_three_digit_palindromic_prime :
  ∃ n, is_palindrome n ∧ is_prime n ∧ 100 ≤ n ∧ n < 1000 ∧
  (n ≠ 101) ∧ (n ≠ 131) ∧ (∀ m, is_palindrome m ∧ is_prime m ∧ 100 ≤ m ∧ m < 1000 → second_smallest_three_digit_palindromic_prime < m → m = n) :=
by
  sorry -- This is where the proof would be, but it is not needed as per instructions.

end NUMINAMATH_GPT_find_third_smallest_three_digit_palindromic_prime_l2381_238157


namespace NUMINAMATH_GPT_Jessie_weight_loss_l2381_238190

theorem Jessie_weight_loss :
  let initial_weight := 74
  let current_weight := 67
  (initial_weight - current_weight) = 7 :=
by
  sorry

end NUMINAMATH_GPT_Jessie_weight_loss_l2381_238190


namespace NUMINAMATH_GPT_sequence_period_9_l2381_238193

def sequence_periodic (x : ℕ → ℤ) : Prop :=
  ∀ n > 1, x (n + 1) = |x n| - x (n - 1)

theorem sequence_period_9 (x : ℕ → ℤ) :
  sequence_periodic x → ∃ p, p = 9 ∧ ∀ n, x (n + p) = x n :=
by
  sorry

end NUMINAMATH_GPT_sequence_period_9_l2381_238193


namespace NUMINAMATH_GPT_q_negative_one_is_minus_one_l2381_238140

-- Define the function q and the point on the graph
def q (x : ℝ) : ℝ := sorry

-- The condition: point (-1, -1) lies on the graph of q
axiom point_on_graph : q (-1) = -1

-- The theorem to prove that q(-1) = -1
theorem q_negative_one_is_minus_one : q (-1) = -1 :=
by exact point_on_graph

end NUMINAMATH_GPT_q_negative_one_is_minus_one_l2381_238140


namespace NUMINAMATH_GPT_pipes_fill_cistern_together_time_l2381_238124

theorem pipes_fill_cistern_together_time
  (t : ℝ)
  (h1 : t * (1 / 12 + 1 / 15) + 6 * (1 / 15) = 1) : 
  t = 4 := 
by
  -- Proof is omitted here as instructed
  sorry

end NUMINAMATH_GPT_pipes_fill_cistern_together_time_l2381_238124


namespace NUMINAMATH_GPT_find_c_plus_one_over_b_l2381_238176

variable (a b c : ℝ)
variable (habc : a * b * c = 1)
variable (ha : a + (1 / c) = 7)
variable (hb : b + (1 / a) = 35)

theorem find_c_plus_one_over_b : (c + (1 / b) = 11 / 61) :=
by
  have h1 : a * b * c = 1 := habc
  have h2 : a + (1 / c) = 7 := ha
  have h3 : b + (1 / a) = 35 := hb
  sorry

end NUMINAMATH_GPT_find_c_plus_one_over_b_l2381_238176


namespace NUMINAMATH_GPT_fuel_reduction_16km_temperature_drop_16km_l2381_238127

-- Definition for fuel reduction condition
def fuel_reduction_rate (distance: ℕ) : ℕ := distance / 4 * 2

-- Definition for temperature drop condition
def temperature_drop_rate (distance: ℕ) : ℕ := distance / 8 * 1

-- Theorem to prove fuel reduction for 16 km
theorem fuel_reduction_16km : fuel_reduction_rate 16 = 8 := 
by
  -- proof will go here, but for now add sorry
  sorry

-- Theorem to prove temperature drop for 16 km
theorem temperature_drop_16km : temperature_drop_rate 16 = 2 := 
by
  -- proof will go here, but for now add sorry
  sorry

end NUMINAMATH_GPT_fuel_reduction_16km_temperature_drop_16km_l2381_238127


namespace NUMINAMATH_GPT_sum_of_altitudes_at_least_nine_times_inradius_l2381_238166

variables (a b c : ℝ)
variables (s : ℝ) -- semiperimeter
variables (Δ : ℝ) -- area
variables (r : ℝ) -- inradius
variables (h_A h_B h_C : ℝ) -- altitudes

-- The Lean statement of the problem
theorem sum_of_altitudes_at_least_nine_times_inradius
  (ha : s = (a + b + c) / 2)
  (hb : Δ = r * s)
  (hc : h_A = (2 * Δ) / a)
  (hd : h_B = (2 * Δ) / b)
  (he : h_C = (2 * Δ) / c) :
  h_A + h_B + h_C ≥ 9 * r :=
sorry

end NUMINAMATH_GPT_sum_of_altitudes_at_least_nine_times_inradius_l2381_238166


namespace NUMINAMATH_GPT_rectangle_area_l2381_238173

theorem rectangle_area (w l : ℕ) (h_sum : w + l = 14) (h_w : w = 6) : w * l = 48 := by
  sorry

end NUMINAMATH_GPT_rectangle_area_l2381_238173


namespace NUMINAMATH_GPT_barry_sotter_magic_l2381_238164

theorem barry_sotter_magic (n : ℕ) : (n + 3) / 3 = 50 → n = 147 := 
by 
  sorry

end NUMINAMATH_GPT_barry_sotter_magic_l2381_238164


namespace NUMINAMATH_GPT_element_of_M_l2381_238115

def M : Set (ℕ × ℕ) := { (2, 3) }

theorem element_of_M : (2, 3) ∈ M :=
by
  sorry

end NUMINAMATH_GPT_element_of_M_l2381_238115


namespace NUMINAMATH_GPT_largest_integer_value_neg_quadratic_l2381_238177

theorem largest_integer_value_neg_quadratic :
  ∃ m : ℤ, (4 < m ∧ m < 7) ∧ (m^2 - 11 * m + 28 < 0) ∧ ∀ n : ℤ, (4 < n ∧ n < 7 ∧ (n^2 - 11 * n + 28 < 0)) → n ≤ m :=
sorry

end NUMINAMATH_GPT_largest_integer_value_neg_quadratic_l2381_238177


namespace NUMINAMATH_GPT_number_of_sets_B_l2381_238126

theorem number_of_sets_B (A : Set ℕ) (hA : A = {1, 2}) :
    ∃ (n : ℕ), n = 4 ∧ (∀ B : Set ℕ, A ∪ B = {1, 2} → B ⊆ A) := sorry

end NUMINAMATH_GPT_number_of_sets_B_l2381_238126


namespace NUMINAMATH_GPT_inequality_proof_equality_condition_l2381_238178

theorem inequality_proof (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b < 2) :
  (1 / (1 + a^2) + 1 / (1 + b^2) ≤ 2 / (1 + a*b)) :=
sorry

theorem equality_condition (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b < 2) :
  (1 / (1 + a^2) + 1 / (1 + b^2) = 2 / (1 + a*b) ↔ a = b ∧ a < 1) :=
sorry

end NUMINAMATH_GPT_inequality_proof_equality_condition_l2381_238178


namespace NUMINAMATH_GPT_total_canoes_built_l2381_238130

-- Definitions of conditions
def initial_canoes : ℕ := 8
def common_ratio : ℕ := 2
def number_of_months : ℕ := 6

-- Sum of a geometric sequence formula
-- Sₙ = a * (r^n - 1) / (r - 1)
def sum_of_geometric_sequence (a r n : ℕ) : ℕ := 
  a * (r^n - 1) / (r - 1)

-- Statement to prove
theorem total_canoes_built : 504 = sum_of_geometric_sequence initial_canoes common_ratio number_of_months := 
  by
  sorry

end NUMINAMATH_GPT_total_canoes_built_l2381_238130


namespace NUMINAMATH_GPT_isosceles_triangle_base_length_l2381_238162

theorem isosceles_triangle_base_length (a b c : ℕ) (h_isosceles : a = b ∨ b = c ∨ c = a)
  (h_perimeter : a + b + c = 16) (h_side_length : a = 6 ∨ b = 6 ∨ c = 6) :
  (a = 4 ∨ b = 4 ∨ c = 4) ∨ (a = 6 ∨ b = 6 ∨ c = 6) :=
sorry

end NUMINAMATH_GPT_isosceles_triangle_base_length_l2381_238162


namespace NUMINAMATH_GPT_comb_eq_comb_imp_n_eq_18_l2381_238117

theorem comb_eq_comb_imp_n_eq_18 {n : ℕ} (h : Nat.choose n 14 = Nat.choose n 4) : n = 18 :=
sorry

end NUMINAMATH_GPT_comb_eq_comb_imp_n_eq_18_l2381_238117


namespace NUMINAMATH_GPT_fraction_calculation_l2381_238134

theorem fraction_calculation :
  (3 / 4) * (1 / 2) * (2 / 5) * 5060 = 759 :=
by
  sorry

end NUMINAMATH_GPT_fraction_calculation_l2381_238134


namespace NUMINAMATH_GPT_find_positions_l2381_238149

def first_column (m : ℕ) : ℕ := 4 + 3*(m-1)

def table_element (m n : ℕ) : ℕ := first_column m + (n-1)*(2*m + 1)

theorem find_positions :
  (∀ m n, table_element m n ≠ 1994) ∧
  (∃ m n, table_element m n = 1995 ∧ ((m = 6 ∧ n = 153) ∨ (m = 153 ∧ n = 6))) :=
by
  sorry

end NUMINAMATH_GPT_find_positions_l2381_238149


namespace NUMINAMATH_GPT_number_of_right_handed_players_l2381_238152

/-- 
Given:
(1) There are 70 players on a football team.
(2) 34 players are throwers.
(3) One third of the non-throwers are left-handed.
(4) All throwers are right-handed.
Prove:
The total number of right-handed players is 58.
-/
theorem number_of_right_handed_players 
  (total_players : ℕ) (throwers : ℕ) (non_throwers : ℕ) (left_handed_non_throwers : ℕ) (right_handed_non_throwers : ℕ) : 
  total_players = 70 ∧ throwers = 34 ∧ non_throwers = total_players - throwers ∧ left_handed_non_throwers = non_throwers / 3 ∧ right_handed_non_throwers = non_throwers - left_handed_non_throwers ∧ right_handed_non_throwers + throwers = 58 :=
by
  sorry

end NUMINAMATH_GPT_number_of_right_handed_players_l2381_238152


namespace NUMINAMATH_GPT_perimeter_of_triangle_l2381_238171

namespace TrianglePerimeter

variables {a b c : ℝ}

-- Conditions translated into definitions
def is_valid_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def absolute_sum_condition (a b c : ℝ) : Prop :=
  |a + b - c| + |b + c - a| + |c + a - b| = 12

-- The theorem stating the perimeter under given conditions
theorem perimeter_of_triangle (h : is_valid_triangle a b c) (h_abs_sum : absolute_sum_condition a b c) : 
  a + b + c = 12 := 
sorry

end TrianglePerimeter

end NUMINAMATH_GPT_perimeter_of_triangle_l2381_238171


namespace NUMINAMATH_GPT_square_area_proof_l2381_238191

   theorem square_area_proof (x : ℝ) (h1 : 4 * x - 15 = 20 - 3 * x) :
     (20 - 3 * x) * (4 * x - 15) = 25 :=
   by
     sorry
   
end NUMINAMATH_GPT_square_area_proof_l2381_238191


namespace NUMINAMATH_GPT_nine_pow_2048_mod_50_l2381_238185

theorem nine_pow_2048_mod_50 : (9^2048) % 50 = 21 := sorry

end NUMINAMATH_GPT_nine_pow_2048_mod_50_l2381_238185


namespace NUMINAMATH_GPT_consecutive_integers_sum_and_difference_l2381_238102

theorem consecutive_integers_sum_and_difference (x y : ℕ) 
(h1 : y = x + 1) 
(h2 : x * y = 552) 
: x + y = 47 ∧ y - x = 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_consecutive_integers_sum_and_difference_l2381_238102


namespace NUMINAMATH_GPT_max_integers_sum_power_of_two_l2381_238121

open Set

/-- Given a finite set of positive integers such that the sum of any two distinct elements is a power of two,
    the cardinality of the set is at most 2. -/
theorem max_integers_sum_power_of_two (S : Finset ℕ) (h_pos : ∀ x ∈ S, 0 < x)
  (h_sum : ∀ {a b : ℕ}, a ∈ S → b ∈ S → a ≠ b → ∃ n : ℕ, a + b = 2^n) : S.card ≤ 2 :=
sorry

end NUMINAMATH_GPT_max_integers_sum_power_of_two_l2381_238121


namespace NUMINAMATH_GPT_curve_points_satisfy_equation_l2381_238125

theorem curve_points_satisfy_equation (C : Set (ℝ × ℝ)) (f : ℝ × ℝ → ℝ) :
  (∀ p : ℝ × ℝ, p ∈ C → f p = 0) → (∀ q : ℝ × ℝ, f q ≠ 0 → q ∉ C) :=
by
  intro h₁
  intro q
  intro h₂
  sorry

end NUMINAMATH_GPT_curve_points_satisfy_equation_l2381_238125


namespace NUMINAMATH_GPT_smallest_consecutive_sum_l2381_238135

theorem smallest_consecutive_sum (n : ℤ) (h : n + (n + 1) + (n + 2) + (n + 3) + (n + 4) = 210) : 
  n = 40 := 
sorry

end NUMINAMATH_GPT_smallest_consecutive_sum_l2381_238135


namespace NUMINAMATH_GPT_function_classification_l2381_238111

theorem function_classification {f : ℝ → ℝ} 
    (h : ∀ x y : ℝ, x * f y + y * f x = (x + y) * f x * f y) : 
    ∀ x : ℝ, f x = 0 ∨ f x = 1 :=
by
  sorry

end NUMINAMATH_GPT_function_classification_l2381_238111


namespace NUMINAMATH_GPT_car_y_start_time_l2381_238150

theorem car_y_start_time : 
  ∀ (t m : ℝ), 
  (35 * (t + m) = 294) ∧ (40 * t = 294) → 
  t = 7.35 ∧ m = 1.05 → 
  m * 60 = 63 :=
by
  intros t m h1 h2
  sorry

end NUMINAMATH_GPT_car_y_start_time_l2381_238150


namespace NUMINAMATH_GPT_total_students_suggestion_l2381_238112

theorem total_students_suggestion :
  let m := 324
  let b := 374
  let t := 128
  m + b + t = 826 := by
  sorry

end NUMINAMATH_GPT_total_students_suggestion_l2381_238112


namespace NUMINAMATH_GPT_smaller_tank_capacity_l2381_238163

/-- Problem Statement:
Three-quarters of the oil from a certain tank (that was initially full) was poured into a
20000-liter capacity tanker that already had 3000 liters of oil.
To make the large tanker half-full, 4000 more liters of oil would be needed.
What is the capacity of the smaller tank?
-/

theorem smaller_tank_capacity (C : ℝ) 
  (h1 : 3 / 4 * C + 3000 + 4000 = 10000) : 
  C = 4000 :=
sorry

end NUMINAMATH_GPT_smaller_tank_capacity_l2381_238163


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l2381_238131

variable (x y : ℝ)

theorem necessary_but_not_sufficient_condition :
  (x ≠ 1 ∨ y ≠ 1) ↔ (xy ≠ 1) :=
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l2381_238131


namespace NUMINAMATH_GPT_additional_savings_is_297_l2381_238197

-- Define initial order amount
def initial_order_amount : ℝ := 12000

-- Define the first set of discounts
def discount_scheme_1 (amount : ℝ) : ℝ :=
  let first_discount := amount * 0.75
  let second_discount := first_discount * 0.85
  let final_price := second_discount * 0.90
  final_price

-- Define the second set of discounts
def discount_scheme_2 (amount : ℝ) : ℝ :=
  let first_discount := amount * 0.70
  let second_discount := first_discount * 0.90
  let final_price := second_discount * 0.95
  final_price

-- Define the amount saved selecting the better discount scheme
def additional_savings : ℝ :=
  let final_price_1 := discount_scheme_1 initial_order_amount
  let final_price_2 := discount_scheme_2 initial_order_amount
  final_price_2 - final_price_1

-- Lean statement to prove the additional savings is $297
theorem additional_savings_is_297 : additional_savings = 297 := by
  sorry

end NUMINAMATH_GPT_additional_savings_is_297_l2381_238197


namespace NUMINAMATH_GPT_cos_identity_l2381_238160

theorem cos_identity
  (α : ℝ)
  (h : Real.sin (π / 6 - α) = 1 / 3) :
  Real.cos (2 * π / 3 + 2 * α) = -7 / 9 :=
by
  sorry

end NUMINAMATH_GPT_cos_identity_l2381_238160


namespace NUMINAMATH_GPT_find_cos_sin_sum_l2381_238192

-- Define the given condition: tan θ = 5/12 and 180° ≤ θ ≤ 270°.
variable (θ : ℝ)
variable (h₁ : Real.tan θ = 5 / 12)
variable (h₂ : π ≤ θ ∧ θ ≤ 3 * π / 2)

-- Define the main statement to prove.
theorem find_cos_sin_sum : Real.cos θ + Real.sin θ = -17 / 13 := by
  sorry

end NUMINAMATH_GPT_find_cos_sin_sum_l2381_238192


namespace NUMINAMATH_GPT_part1_union_part1_complement_part2_intersect_l2381_238189

namespace MathProof

open Set Real

def A : Set ℝ := { x | 1 ≤ x ∧ x < 7 }
def B : Set ℝ := { x | 2 < x ∧ x < 10 }
def C (a : ℝ) : Set ℝ := { x | x < a }
def R : Set ℝ := univ  -- the set of all real numbers

theorem part1_union :
  A ∪ B = { x | 1 ≤ x ∧ x < 10 } :=
sorry

theorem part1_complement :
  R \ B = { x | x ≤ 2 ∨ x ≥ 10 } :=
sorry

theorem part2_intersect (a : ℝ) :
  (A ∩ C a ≠ ∅) → a > 1 :=
sorry

end MathProof

end NUMINAMATH_GPT_part1_union_part1_complement_part2_intersect_l2381_238189


namespace NUMINAMATH_GPT_ratio_HP_HA_l2381_238195

-- Given Definitions
variables (A B C P Q H : Type)
variables (h1 : Triangle A B C) (h2 : AcuteTriangle A B C) (h3 : P ≠ Q)
variables (h4 : FootOfAltitudeFrom A H B C) (h5 : OnExtendedLine P A B) (h6 : OnExtendedLine Q A C)
variables (h7 : HP = HQ) (h8 : CyclicQuadrilateral B C P Q)

-- Required Ratio
theorem ratio_HP_HA : HP = HA := sorry

end NUMINAMATH_GPT_ratio_HP_HA_l2381_238195


namespace NUMINAMATH_GPT_no_non_square_number_with_triple_product_divisors_l2381_238141

theorem no_non_square_number_with_triple_product_divisors (N : ℕ) (h_non_square : ∀ k : ℕ, k * k ≠ N) : 
  ¬ (∃ t : ℕ, ∃ d : Finset (Finset ℕ), (∀ s ∈ d, s.card = 3) ∧ (∀ s ∈ d, s.prod id = t)) := 
sorry

end NUMINAMATH_GPT_no_non_square_number_with_triple_product_divisors_l2381_238141


namespace NUMINAMATH_GPT_initial_stock_of_coffee_l2381_238194

theorem initial_stock_of_coffee (x : ℝ) (h : x ≥ 0) 
  (h1 : 0.30 * x + 60 = 0.36 * (x + 100)) : x = 400 :=
by sorry

end NUMINAMATH_GPT_initial_stock_of_coffee_l2381_238194


namespace NUMINAMATH_GPT_proposition_truthfulness_l2381_238153

-- Definitions
def is_positive (n : ℕ) : Prop := n > 0
def is_even (n : ℕ) : Prop := n % 2 = 0
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

-- Original proposition
def original_prop (n : ℕ) : Prop := is_positive n ∧ is_even n → ¬ is_prime n

-- Converse proposition
def converse_prop (n : ℕ) : Prop := ¬ is_prime n → is_positive n ∧ is_even n

-- Inverse proposition
def inverse_prop (n : ℕ) : Prop := ¬ (is_positive n ∧ is_even n) → is_prime n

-- Contrapositive proposition
def contrapositive_prop (n : ℕ) : Prop := is_prime n → ¬ (is_positive n ∧ is_even n)

-- Proof problem statement
theorem proposition_truthfulness (n : ℕ) :
  (original_prop n = False) ∧
  (converse_prop n = False) ∧
  (inverse_prop n = False) ∧
  (contrapositive_prop n = True) :=
sorry

end NUMINAMATH_GPT_proposition_truthfulness_l2381_238153


namespace NUMINAMATH_GPT_avg_height_country_l2381_238133

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

end NUMINAMATH_GPT_avg_height_country_l2381_238133


namespace NUMINAMATH_GPT_initial_music_files_l2381_238143

-- Define the conditions
def video_files : ℕ := 21
def deleted_files : ℕ := 23
def remaining_files : ℕ := 2

-- Theorem to prove the initial number of music files
theorem initial_music_files : 
  ∃ (M : ℕ), (M + video_files - deleted_files = remaining_files) → M = 4 := 
sorry

end NUMINAMATH_GPT_initial_music_files_l2381_238143


namespace NUMINAMATH_GPT_evaluate_fraction_l2381_238168

theorem evaluate_fraction (a b : ℝ) (h1 : a = 5) (h2 : b = 3) : 3 / (a + b) = 3 / 8 :=
by
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_evaluate_fraction_l2381_238168


namespace NUMINAMATH_GPT_largest_polygon_area_l2381_238137

structure Polygon :=
(unit_squares : Nat)
(right_triangles : Nat)

def area (p : Polygon) : ℝ :=
p.unit_squares + 0.5 * p.right_triangles

def polygon_A : Polygon := { unit_squares := 6, right_triangles := 2 }
def polygon_B : Polygon := { unit_squares := 7, right_triangles := 1 }
def polygon_C : Polygon := { unit_squares := 8, right_triangles := 0 }
def polygon_D : Polygon := { unit_squares := 5, right_triangles := 4 }
def polygon_E : Polygon := { unit_squares := 6, right_triangles := 2 }

theorem largest_polygon_area :
  max (area polygon_A) (max (area polygon_B) (max (area polygon_C) (max (area polygon_D) (area polygon_E)))) = area polygon_C :=
by
  sorry

end NUMINAMATH_GPT_largest_polygon_area_l2381_238137


namespace NUMINAMATH_GPT_percentage_increase_painting_l2381_238147

/-
Problem:
Given:
1. The original cost of jewelry is $30 each.
2. The original cost of paintings is $100 each.
3. The new cost of jewelry is $40 each.
4. The new cost of paintings is $100 + ($100 * P / 100).
5. A buyer purchased 2 pieces of jewelry and 5 paintings for $680.

Prove:
The percentage increase in the cost of each painting (P) is 20%.
-/

theorem percentage_increase_painting (P : ℝ) :
  let jewelry_price := 30
  let painting_price := 100
  let new_jewelry_price := 40
  let new_painting_price := 100 * (1 + P / 100)
  let total_cost := 2 * new_jewelry_price + 5 * new_painting_price
  total_cost = 680 → P = 20 := by
sorry

end NUMINAMATH_GPT_percentage_increase_painting_l2381_238147


namespace NUMINAMATH_GPT_downstream_distance_l2381_238123

theorem downstream_distance (speed_boat : ℝ) (speed_current : ℝ) (time_minutes : ℝ) (distance : ℝ) :
  speed_boat = 20 ∧ speed_current = 5 ∧ time_minutes = 24 ∧ distance = 10 →
  (speed_boat + speed_current) * (time_minutes / 60) = distance :=
by
  sorry

end NUMINAMATH_GPT_downstream_distance_l2381_238123


namespace NUMINAMATH_GPT_intersection_line_canonical_equation_l2381_238106

def plane1 (x y z : ℝ) : Prop := 6 * x - 7 * y - z - 2 = 0
def plane2 (x y z : ℝ) : Prop := x + 7 * y - 4 * z - 5 = 0
def canonical_equation (x y z : ℝ) : Prop := 
  (x - 1) / 35 = (y - 4 / 7) / 23 ∧ (y - 4 / 7) / 23 = z / 49

theorem intersection_line_canonical_equation (x y z : ℝ) :
  plane1 x y z → plane2 x y z → canonical_equation x y z :=
by
  intros h1 h2
  unfold plane1 at h1
  unfold plane2 at h2
  unfold canonical_equation
  sorry

end NUMINAMATH_GPT_intersection_line_canonical_equation_l2381_238106


namespace NUMINAMATH_GPT_complement_of_A_l2381_238113

def A : Set ℝ := {y : ℝ | ∃ (x : ℝ), y = 2^x}

theorem complement_of_A : (Set.compl A) = {y : ℝ | y ≤ 0} :=
by
  sorry

end NUMINAMATH_GPT_complement_of_A_l2381_238113
