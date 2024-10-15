import Mathlib

namespace NUMINAMATH_GPT_correct_substitution_l881_88107

theorem correct_substitution (x : ℝ) : 
    (2 * x - 7)^2 + (5 * x - 17.5)^2 = 0 → 
    x = 7 / 2 :=
by
  sorry

end NUMINAMATH_GPT_correct_substitution_l881_88107


namespace NUMINAMATH_GPT_remaining_distance_l881_88156

theorem remaining_distance (S u : ℝ) (h1 : S / (2 * u) + 24 = S) (h2 : S * u / 2 + 15 = S) : ∃ x : ℝ, x = 8 :=
by
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_remaining_distance_l881_88156


namespace NUMINAMATH_GPT_new_average_is_21_l881_88195

def initial_number_of_students : ℕ := 30
def late_students : ℕ := 4
def initial_jumping_students : ℕ := initial_number_of_students - late_students
def initial_average_score : ℕ := 20
def late_student_scores : List ℕ := [26, 27, 28, 29]
def total_jumps_initial_students : ℕ := initial_jumping_students * initial_average_score
def total_jumps_late_students : ℕ := late_student_scores.sum
def total_jumps_all_students : ℕ := total_jumps_initial_students + total_jumps_late_students
def new_average_score : ℕ := total_jumps_all_students / initial_number_of_students

theorem new_average_is_21 :
  new_average_score = 21 :=
sorry

end NUMINAMATH_GPT_new_average_is_21_l881_88195


namespace NUMINAMATH_GPT_john_books_purchase_l881_88130

theorem john_books_purchase : 
  let john_money := 4575
  let book_price := 325
  john_money / book_price = 14 :=
by
  sorry

end NUMINAMATH_GPT_john_books_purchase_l881_88130


namespace NUMINAMATH_GPT_inverse_proportion_relation_l881_88175

theorem inverse_proportion_relation (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : y₁ = 2 / x₁) 
  (h2 : y₂ = 2 / x₂) 
  (h3 : x₁ < x₂) 
  (h4 : x₂ < 0) : 
  y₂ < y₁ ∧ y₁ < 0 := 
sorry

end NUMINAMATH_GPT_inverse_proportion_relation_l881_88175


namespace NUMINAMATH_GPT_ratio_Sachin_Rahul_l881_88131

-- Definitions: Sachin's age (S) is 63, and Sachin is younger than Rahul by 18 years.
def Sachin_age : ℕ := 63
def Rahul_age : ℕ := Sachin_age + 18

-- The problem: Prove the ratio of Sachin's age to Rahul's age is 7/9.
theorem ratio_Sachin_Rahul : (Sachin_age : ℚ) / (Rahul_age : ℚ) = 7 / 9 :=
by 
  -- The proof will go here, but we are skipping the proof as per the instructions.
  sorry

end NUMINAMATH_GPT_ratio_Sachin_Rahul_l881_88131


namespace NUMINAMATH_GPT_days_c_worked_l881_88111

noncomputable def work_done_by_a_b := 1 / 10
noncomputable def work_done_by_b_c := 1 / 18
noncomputable def work_done_by_c_alone := 1 / 45

theorem days_c_worked
  (A B C : ℚ)
  (h1 : A + B = work_done_by_a_b)
  (h2 : B + C = work_done_by_b_c)
  (h3 : C = work_done_by_c_alone) :
  15 = (1/3) / work_done_by_c_alone :=
sorry

end NUMINAMATH_GPT_days_c_worked_l881_88111


namespace NUMINAMATH_GPT_symmetric_points_addition_l881_88155

theorem symmetric_points_addition (m n : ℤ) (h₁ : m = 2) (h₂ : n = -3) : m + n = -1 := by
  rw [h₁, h₂]
  norm_num

end NUMINAMATH_GPT_symmetric_points_addition_l881_88155


namespace NUMINAMATH_GPT_total_pages_in_scifi_section_l881_88149

theorem total_pages_in_scifi_section : 
  let books := 8
  let pages_per_book := 478
  books * pages_per_book = 3824 := 
by
  sorry

end NUMINAMATH_GPT_total_pages_in_scifi_section_l881_88149


namespace NUMINAMATH_GPT_base8_subtraction_l881_88110

-- Define the base 8 notation for the given numbers
def b8_256 := 256
def b8_167 := 167
def b8_145 := 145

-- Define the sum of 256_8 and 167_8 in base 8
def sum_b8 := 435

-- Define the result of subtracting 145_8 from the sum in base 8
def result_b8 := 370

-- Prove that the result of the entire operation is 370_8
theorem base8_subtraction : sum_b8 - b8_145 = result_b8 := by
  sorry

end NUMINAMATH_GPT_base8_subtraction_l881_88110


namespace NUMINAMATH_GPT_operation_8_to_cube_root_16_l881_88196

theorem operation_8_to_cube_root_16 : ∃ (x : ℕ), x = 8 ∧ (x * x = (Nat.sqrt 16)^3) :=
by
  sorry

end NUMINAMATH_GPT_operation_8_to_cube_root_16_l881_88196


namespace NUMINAMATH_GPT_quadratic_real_roots_range_l881_88128

theorem quadratic_real_roots_range (k : ℝ) : 
  (∃ x : ℝ, k * x^2 + 3 * x - 1 = 0) ↔ (k ≥ -9 / 4 ∧ k ≠ 0) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_real_roots_range_l881_88128


namespace NUMINAMATH_GPT_arthur_walked_total_miles_l881_88190

def blocks_east := 8
def blocks_north := 15
def blocks_west := 3
def block_length := 1/2

def total_blocks := blocks_east + blocks_north + blocks_west
def total_miles := total_blocks * block_length

theorem arthur_walked_total_miles : total_miles = 13 := by
  sorry

end NUMINAMATH_GPT_arthur_walked_total_miles_l881_88190


namespace NUMINAMATH_GPT_rectangle_area_diff_l881_88165

theorem rectangle_area_diff :
  ∀ (l w : ℕ), (2 * l + 2 * w = 60) → (∃ A_max A_min : ℕ, 
    A_max = (l * (30 - l)) ∧ A_min = (min (1 * (30 - 1)) (29 * (30 - 29))) ∧ (A_max - A_min = 196)) :=
by
  intros l w h
  use 15 * 15, min (1 * 29) (29 * 1)
  sorry

end NUMINAMATH_GPT_rectangle_area_diff_l881_88165


namespace NUMINAMATH_GPT_acute_angle_vector_range_l881_88169

theorem acute_angle_vector_range (m : ℝ) (a b : ℝ × ℝ) 
  (h1 : a = (1, 2)) 
  (h2 : b = (4, m)) 
  (acute : (a.1 * b.1 + a.2 * b.2) > 0) : 
  (m > -2) ∧ (m ≠ 8) := 
by 
  sorry

end NUMINAMATH_GPT_acute_angle_vector_range_l881_88169


namespace NUMINAMATH_GPT_number_of_students_l881_88123

-- Define parameters and conditions
variables (B G : ℕ) -- number of boys and girls

-- Condition: each boy is friends with exactly two girls
axiom boys_to_girls : ∀ (B G : ℕ), 2 * B = 3 * G

-- Condition: total number of children in the class
axiom total_children : ∀ (B G : ℕ), B + G = 31

-- Define the theorem that proves the correct number of students
theorem number_of_students : (B G : ℕ) → 2 * B = 3 * G → B + G = 31 → B + G = 35 :=
by
  sorry

end NUMINAMATH_GPT_number_of_students_l881_88123


namespace NUMINAMATH_GPT_derivative_at_minus_one_l881_88178
open Real

def f (x : ℝ) : ℝ := (1 + x) * (2 + x^2)^(1 / 2) * (3 + x^3)^(1 / 3)

theorem derivative_at_minus_one : deriv f (-1) = sqrt 3 * 2^(1 / 3) :=
by sorry

end NUMINAMATH_GPT_derivative_at_minus_one_l881_88178


namespace NUMINAMATH_GPT_large_marshmallows_are_eight_l881_88109

-- Definition for the total number of marshmallows
def total_marshmallows : ℕ := 18

-- Definition for the number of mini marshmallows
def mini_marshmallows : ℕ := 10

-- Definition for the number of large marshmallows
def large_marshmallows : ℕ := total_marshmallows - mini_marshmallows

-- Theorem stating that the number of large marshmallows is 8
theorem large_marshmallows_are_eight : large_marshmallows = 8 := by
  sorry

end NUMINAMATH_GPT_large_marshmallows_are_eight_l881_88109


namespace NUMINAMATH_GPT_arithmetic_geometric_sequence_a1_l881_88158

theorem arithmetic_geometric_sequence_a1 (a : ℕ → ℚ)
  (h1 : a 1 + a 6 = 11)
  (h2 : a 3 * a 4 = 32 / 9) :
  a 1 = 32 / 3 ∨ a 1 = 1 / 3 :=
sorry

end NUMINAMATH_GPT_arithmetic_geometric_sequence_a1_l881_88158


namespace NUMINAMATH_GPT_find_n_l881_88185

theorem find_n (n : ℕ) : (8 : ℝ)^(1/3) = (2 : ℝ)^n → n = 1 := by
  sorry

end NUMINAMATH_GPT_find_n_l881_88185


namespace NUMINAMATH_GPT_initial_pennies_in_each_compartment_l881_88170

theorem initial_pennies_in_each_compartment (x : ℕ) (h : 12 * (x + 6) = 96) : x = 2 :=
by sorry

end NUMINAMATH_GPT_initial_pennies_in_each_compartment_l881_88170


namespace NUMINAMATH_GPT_file_size_l881_88142

-- Definitions based on conditions
def upload_speed : ℕ := 8 -- megabytes per minute
def upload_time : ℕ := 20 -- minutes

-- Goal to prove
theorem file_size:
  (upload_speed * upload_time = 160) :=
by sorry

end NUMINAMATH_GPT_file_size_l881_88142


namespace NUMINAMATH_GPT_max_plus_min_value_of_y_eq_neg4_l881_88143

noncomputable def y (x : ℝ) : ℝ := (2 * (Real.sin x) ^ 2 + Real.sin (3 * x / 2) - 4) / ((Real.sin x) ^ 2 + 2 * (Real.cos x) ^ 2)

theorem max_plus_min_value_of_y_eq_neg4 (M m : ℝ) (hM : ∃ x : ℝ, y x = M) (hm : ∃ x : ℝ, y x = m) :
  M + m = -4 := sorry

end NUMINAMATH_GPT_max_plus_min_value_of_y_eq_neg4_l881_88143


namespace NUMINAMATH_GPT_remainder_of_3_pow_600_mod_19_l881_88154

theorem remainder_of_3_pow_600_mod_19 :
  (3 ^ 600) % 19 = 11 :=
sorry

end NUMINAMATH_GPT_remainder_of_3_pow_600_mod_19_l881_88154


namespace NUMINAMATH_GPT_find_number_satisfies_l881_88135

noncomputable def find_number (m : ℤ) (n : ℤ) : Prop :=
  (m % n = 2) ∧ (3 * m % n = 1)

theorem find_number_satisfies (m : ℤ) : ∃ n : ℤ, find_number m n ∧ n = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_number_satisfies_l881_88135


namespace NUMINAMATH_GPT_range_of_b_l881_88159

noncomputable def f : ℝ → ℝ
| x => if x < -1/2 then (2*x + 1) / (x^2) else x + 1

def g (x : ℝ) : ℝ := x^2 - 4*x - 4

theorem range_of_b (a b : ℝ) (h : f a + g b = 0) : -1 <= b ∧ b <= 5 :=
sorry

end NUMINAMATH_GPT_range_of_b_l881_88159


namespace NUMINAMATH_GPT_values_of_k_real_equal_roots_l881_88148

theorem values_of_k_real_equal_roots (k : ℝ) : 
  (∃ k, (3 - 2 * k)^2 - 4 * 3 * 12 = 0 ∧ (k = -9 / 2 ∨ k = 15 / 2)) :=
by
  sorry

end NUMINAMATH_GPT_values_of_k_real_equal_roots_l881_88148


namespace NUMINAMATH_GPT_range_of_a_l881_88137

noncomputable def setA : Set ℝ := {x | x^2 + 4 * x = 0}
noncomputable def setB (a : ℝ) : Set ℝ := {x | x^2 + a * x + a = 0}

theorem range_of_a :
  ∀ a : ℝ, (setA ∪ setB a) = setA ↔ 0 ≤ a ∧ a < 4 :=
by sorry

end NUMINAMATH_GPT_range_of_a_l881_88137


namespace NUMINAMATH_GPT_correct_inequality_l881_88101

theorem correct_inequality (a b : ℝ) (h1 : -1 < a) (h2 : a < b) (h3 : b < 0) :
  a^2 > ab ∧ ab > a :=
sorry

end NUMINAMATH_GPT_correct_inequality_l881_88101


namespace NUMINAMATH_GPT_solve_inequality_l881_88102

theorem solve_inequality :
  (4 - Real.sqrt 17 < x ∧ x < 4 - Real.sqrt 3) ∨ 
  (4 + Real.sqrt 3 < x ∧ x < 4 + Real.sqrt 17) → 
  0 < (x^2 - 8*x + 13) / (x^2 - 4*x + 7) ∧ 
  (x^2 - 8*x + 13) / (x^2 - 4*x + 7) < 2 :=
sorry

end NUMINAMATH_GPT_solve_inequality_l881_88102


namespace NUMINAMATH_GPT_find_ordered_pair_l881_88168

theorem find_ordered_pair (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0)
  (h₃ : (x : ℝ) → x^2 + 2 * a * x + b = 0 → x = a ∨ x = b) :
  (a, b) = (1, -3) :=
sorry

end NUMINAMATH_GPT_find_ordered_pair_l881_88168


namespace NUMINAMATH_GPT_number_of_stanzas_is_correct_l881_88129

-- Define the total number of words in the poem
def total_words : ℕ := 1600

-- Define the number of lines per stanza
def lines_per_stanza : ℕ := 10

-- Define the number of words per line
def words_per_line : ℕ := 8

-- Calculate the number of words per stanza
def words_per_stanza : ℕ := lines_per_stanza * words_per_line

-- Define the number of stanzas
def stanzas (total_words words_per_stanza : ℕ) := total_words / words_per_stanza

-- Theorem: Prove that given the conditions, the number of stanzas is 20
theorem number_of_stanzas_is_correct : stanzas total_words words_per_stanza = 20 :=
by
  -- Insert the proof here
  sorry

end NUMINAMATH_GPT_number_of_stanzas_is_correct_l881_88129


namespace NUMINAMATH_GPT_number_of_shelves_l881_88188

-- Given conditions
def booksBeforeTrip : ℕ := 56
def booksBought : ℕ := 26
def avgBooksPerShelf : ℕ := 20
def booksLeftOver : ℕ := 2
def totalBooks : ℕ := booksBeforeTrip + booksBought

-- Statement to prove
theorem number_of_shelves :
  totalBooks - booksLeftOver = 80 →
  80 / avgBooksPerShelf = 4 := by
  intros h
  sorry

end NUMINAMATH_GPT_number_of_shelves_l881_88188


namespace NUMINAMATH_GPT_similar_polygons_perimeter_ratio_l881_88103

-- Define the main function to assert the proportional relationship
theorem similar_polygons_perimeter_ratio (x y : ℕ) (h1 : 9 * y^2 = 64 * x^2) : x * 8 = y * 3 :=
by sorry

-- noncomputable if needed (only necessary when computation is involved, otherwise omit)

end NUMINAMATH_GPT_similar_polygons_perimeter_ratio_l881_88103


namespace NUMINAMATH_GPT_derivative_at_pi_div_3_l881_88133

noncomputable def f (x : ℝ) : ℝ := x * Real.cos x - Real.sin x

theorem derivative_at_pi_div_3 : 
  deriv f (Real.pi / 3) = - (Real.sqrt 3 * Real.pi / 6) :=
by
  sorry

end NUMINAMATH_GPT_derivative_at_pi_div_3_l881_88133


namespace NUMINAMATH_GPT_brendan_cuts_yards_l881_88186

theorem brendan_cuts_yards (x : ℝ) (h : 7 * 1.5 * x = 84) : x = 8 :=
sorry

end NUMINAMATH_GPT_brendan_cuts_yards_l881_88186


namespace NUMINAMATH_GPT_math_problem_l881_88122

noncomputable def a (b : ℝ) : ℝ := 
  sorry -- to be derived from the conditions

noncomputable def b : ℝ := 
  sorry -- to be derived from the conditions

theorem math_problem (a b: ℝ) 
  (h1: a - b = 1)
  (h2: a^2 - b^2 = -1) : 
  a^2008 - b^2008 = -1 := 
sorry

end NUMINAMATH_GPT_math_problem_l881_88122


namespace NUMINAMATH_GPT_minimum_value_expression_l881_88146

noncomputable def expression (a b c d : ℝ) : ℝ :=
  (a + b) / c + (a + c) / d + (b + d) / a + (c + d) / b

theorem minimum_value_expression (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  expression a b c d ≥ 8 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_minimum_value_expression_l881_88146


namespace NUMINAMATH_GPT_flower_problem_l881_88167

def totalFlowers (n_rows n_per_row : Nat) : Nat :=
  n_rows * n_per_row

def flowersCut (total percent_cut : Nat) : Nat :=
  total * percent_cut / 100

def flowersRemaining (total cut : Nat) : Nat :=
  total - cut

theorem flower_problem :
  let n_rows := 50
  let n_per_row := 400
  let percent_cut := 60
  let total := totalFlowers n_rows n_per_row
  let cut := flowersCut total percent_cut
  flowersRemaining total cut = 8000 :=
by
  sorry

end NUMINAMATH_GPT_flower_problem_l881_88167


namespace NUMINAMATH_GPT_total_marbles_l881_88104

def Mary_marbles : ℕ := 9
def Joan_marbles : ℕ := 3

theorem total_marbles : Mary_marbles + Joan_marbles = 12 :=
by
  -- Please provide the proof here if needed
  sorry

end NUMINAMATH_GPT_total_marbles_l881_88104


namespace NUMINAMATH_GPT_second_term_is_correct_l881_88153

noncomputable def arithmetic_sequence_second_term (a d : ℤ) (h1 : a + 9 * d = 15) (h2 : a + 10 * d = 18) : ℤ :=
  a + d

theorem second_term_is_correct (a d : ℤ) (h1 : a + 9 * d = 15) (h2 : a + 10 * d = 18) :
  arithmetic_sequence_second_term a d h1 h2 = -9 :=
sorry

end NUMINAMATH_GPT_second_term_is_correct_l881_88153


namespace NUMINAMATH_GPT_johnny_marble_combinations_l881_88100

/-- 
Johnny has 10 different colored marbles. 
The number of ways he can choose four different marbles from his bag is 210.
-/
theorem johnny_marble_combinations : (Nat.choose 10 4) = 210 := by
  sorry

end NUMINAMATH_GPT_johnny_marble_combinations_l881_88100


namespace NUMINAMATH_GPT_domain_of_w_l881_88177

theorem domain_of_w :
  {x : ℝ | x + (x - 1)^(1/3) + (8 - x)^(1/3) ≥ 0} = {x : ℝ | x ≥ 0} :=
by {
  sorry
}

end NUMINAMATH_GPT_domain_of_w_l881_88177


namespace NUMINAMATH_GPT_percentage_workday_in_meetings_l881_88132

theorem percentage_workday_in_meetings :
  let workday_minutes := 10 * 60
  let first_meeting := 30
  let second_meeting := 2 * first_meeting
  let third_meeting := first_meeting + second_meeting
  let total_meeting_minutes := first_meeting + second_meeting + third_meeting
  (total_meeting_minutes * 100) / workday_minutes = 30 :=
by
  sorry

end NUMINAMATH_GPT_percentage_workday_in_meetings_l881_88132


namespace NUMINAMATH_GPT_sqrt_fraction_expression_eq_one_l881_88127

theorem sqrt_fraction_expression_eq_one :
  (Real.sqrt (9 / 4) - Real.sqrt (4 / 9) + 1 / 6) = 1 := 
by
  sorry

end NUMINAMATH_GPT_sqrt_fraction_expression_eq_one_l881_88127


namespace NUMINAMATH_GPT_max_value_expression_l881_88157

noncomputable def target_expr (x y z : ℝ) : ℝ :=
  (x^2 - x * y + y^2) * (y^2 - y * z + z^2) * (z^2 - z * x + x^2)

theorem max_value_expression (x y z : ℝ) (h : x + y + z = 3) (hxy : x = y) (hxz : 0 ≤ x) (hyz : 0 ≤ y) (hzz : 0 ≤ z) :
  target_expr x y z ≤ 9 / 4 := by
  sorry

end NUMINAMATH_GPT_max_value_expression_l881_88157


namespace NUMINAMATH_GPT_max_pages_within_budget_l881_88105

-- Definitions based on the problem conditions
def page_cost_in_cents : ℕ := 5
def total_budget_in_cents : ℕ := 5000
def max_expenditure_in_cents : ℕ := 4500

-- Proof problem statement
theorem max_pages_within_budget : 
  ∃ (pages : ℕ), pages = max_expenditure_in_cents / page_cost_in_cents ∧ 
                  pages * page_cost_in_cents ≤ total_budget_in_cents :=
by {
  sorry
}

end NUMINAMATH_GPT_max_pages_within_budget_l881_88105


namespace NUMINAMATH_GPT_lunks_needed_for_apples_l881_88174

theorem lunks_needed_for_apples :
  (∀ l k a : ℕ, (4 * k = 2 * l) ∧ (3 * a = 5 * k ) → ∃ l', l' = (24 * l / 4)) :=
by
  intros l k a h
  obtain ⟨h1, h2⟩ := h
  have k_for_apples := 3 * a / 5
  have l_for_kunks := 4 * k / 2
  sorry

end NUMINAMATH_GPT_lunks_needed_for_apples_l881_88174


namespace NUMINAMATH_GPT_minimum_value_of_tan_sum_l881_88193

open Real

theorem minimum_value_of_tan_sum :
  ∀ {A B C : ℝ}, 
  0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2 ∧ A + B + C = π ∧ 
  2 * sin A ^ 2 + sin B ^ 2 = 2 * sin C ^ 2 ->
  ( ∃ t : ℝ, ( t = 1 / tan A + 1 / tan B + 1 / tan C ) ∧ t = sqrt 13 / 2 ) := 
sorry

end NUMINAMATH_GPT_minimum_value_of_tan_sum_l881_88193


namespace NUMINAMATH_GPT_minimum_value_expression_l881_88117

theorem minimum_value_expression (γ δ : ℝ) :
  (3 * Real.cos γ + 4 * Real.sin δ - 7)^2 + (3 * Real.sin γ + 4 * Real.cos δ - 12)^2 ≥ 81 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_expression_l881_88117


namespace NUMINAMATH_GPT_pipe_length_difference_l881_88145

theorem pipe_length_difference (total_length shorter_piece : ℕ) (h1 : total_length = 68) (h2 : shorter_piece = 28) : 
  total_length - shorter_piece * 2 = 12 := 
sorry

end NUMINAMATH_GPT_pipe_length_difference_l881_88145


namespace NUMINAMATH_GPT_log_value_between_integers_l881_88124

theorem log_value_between_integers : (1 : ℤ) < Real.log 25 / Real.log 10 ∧ Real.log 25 / Real.log 10 < (2 : ℤ) → 1 + 2 = 3 :=
by
  sorry

end NUMINAMATH_GPT_log_value_between_integers_l881_88124


namespace NUMINAMATH_GPT_negation_of_exists_gt0_and_poly_gt0_l881_88183

theorem negation_of_exists_gt0_and_poly_gt0 :
  (¬ (∃ x₀ : ℝ, x₀ > 0 ∧ x₀^2 - 5 * x₀ + 6 > 0)) ↔ 
  (∀ x : ℝ, x > 0 → x^2 - 5 * x + 6 ≤ 0) :=
by sorry

end NUMINAMATH_GPT_negation_of_exists_gt0_and_poly_gt0_l881_88183


namespace NUMINAMATH_GPT_range_of_a_l881_88176

theorem range_of_a : (∀ x : ℝ, x^2 + (a-1)*x + 1 > 0) ↔ (-1 < a ∧ a < 3) := by
  sorry

end NUMINAMATH_GPT_range_of_a_l881_88176


namespace NUMINAMATH_GPT_union_A_B_l881_88141

def set_A : Set ℝ := { x | 1 / x ≤ 0 }
def set_B : Set ℝ := { x | x^2 - 1 < 0 }

theorem union_A_B : set_A ∪ set_B = { x | x < 1 } :=
by
  sorry

end NUMINAMATH_GPT_union_A_B_l881_88141


namespace NUMINAMATH_GPT_last_number_aryana_counts_l881_88181

theorem last_number_aryana_counts (a d : ℤ) (h_start : a = 72) (h_diff : d = -11) :
  ∃ n : ℕ, (a + n * d > 0) ∧ (a + (n + 1) * d ≤ 0) ∧ a + n * d = 6 := by
  sorry

end NUMINAMATH_GPT_last_number_aryana_counts_l881_88181


namespace NUMINAMATH_GPT_hypotenuse_length_l881_88192

theorem hypotenuse_length
  (x : ℝ) 
  (h_leg_relation : 3 * x - 3 > 0) -- to ensure the legs are positive
  (hypotenuse : ℝ)
  (area_eq : 1 / 2 * x * (3 * x - 3) = 84)
  (pythagorean : hypotenuse^2 = x^2 + (3 * x - 3)^2) :
  hypotenuse = Real.sqrt 505 :=
by 
  sorry

end NUMINAMATH_GPT_hypotenuse_length_l881_88192


namespace NUMINAMATH_GPT_find_ordered_pair_l881_88197

theorem find_ordered_pair : ∃ (x y : ℚ), 
  3 * x - 4 * y = -7 ∧ 4 * x + 5 * y = 23 ∧ 
  x = 57 / 31 ∧ y = 195 / 62 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_ordered_pair_l881_88197


namespace NUMINAMATH_GPT_number_of_terms_added_l881_88118

theorem number_of_terms_added (k : ℕ) (h : 1 ≤ k) :
  (2^(k+1) - 1) - (2^k - 1) = 2^k :=
by sorry

end NUMINAMATH_GPT_number_of_terms_added_l881_88118


namespace NUMINAMATH_GPT_age_of_b_l881_88108

variable (a b : ℕ)
variable (h1 : a * 3 = b * 5)
variable (h2 : (a + 2) * 2 = (b + 2) * 3)

theorem age_of_b : b = 6 :=
by
  sorry

end NUMINAMATH_GPT_age_of_b_l881_88108


namespace NUMINAMATH_GPT_sport_flavoring_to_water_ratio_l881_88166

/-- The ratio by volume of flavoring to corn syrup to water in the 
standard formulation is 1:12:30. The sport formulation has a ratio 
of flavoring to corn syrup three times as great as in the standard formulation. 
A large bottle of the sport formulation contains 4 ounces of corn syrup and 
60 ounces of water. Prove that the ratio of the amount of flavoring to water 
in the sport formulation compared to the standard formulation is 1:2. -/
theorem sport_flavoring_to_water_ratio 
    (standard_flavoring : ℝ) 
    (standard_corn_syrup : ℝ) 
    (standard_water : ℝ) : 
  standard_flavoring = 1 → standard_corn_syrup = 12 → 
  standard_water = 30 → 
  ∃ sport_flavoring : ℝ, 
  ∃ sport_corn_syrup : ℝ, 
  ∃ sport_water : ℝ, 
  sport_corn_syrup = 4 ∧ 
  sport_water = 60 ∧ 
  (sport_flavoring / sport_water) = (standard_flavoring / standard_water) / 2 :=
by
  sorry

end NUMINAMATH_GPT_sport_flavoring_to_water_ratio_l881_88166


namespace NUMINAMATH_GPT_line_AB_equation_l881_88116

theorem line_AB_equation (m : ℝ) (A B : ℝ × ℝ)
  (hA : A = (0, 0)) (hA_line : ∀ (x y : ℝ), A = (x, y) → x + m * y = 0)
  (hB : B = (1, 3)) (hB_line : ∀ (x y : ℝ), B = (x, y) → m * x - y - m + 3 = 0) :
  ∃ (a b c : ℝ), a * 1 - b * 3 + c = 0 ∧ a * x + b * y + c * 0 = 0 ∧ 3 * x - y + 0 = 0 :=
by
  sorry

end NUMINAMATH_GPT_line_AB_equation_l881_88116


namespace NUMINAMATH_GPT_boat_distance_along_stream_l881_88164

theorem boat_distance_along_stream
  (distance_against_stream : ℝ)
  (speed_still_water : ℝ)
  (time : ℝ)
  (v_s : ℝ)
  (H1 : distance_against_stream = 5)
  (H2 : speed_still_water = 6)
  (H3 : time = 1)
  (H4 : speed_still_water - v_s = distance_against_stream / time) :
  (speed_still_water + v_s) * time = 7 :=
by
  -- Sorry to skip proof
  sorry

end NUMINAMATH_GPT_boat_distance_along_stream_l881_88164


namespace NUMINAMATH_GPT_english_alphabet_is_set_l881_88126

-- Conditions definition: Elements of a set must have the properties of definiteness, distinctness, and unorderedness.
def is_definite (A : Type) : Prop := ∀ (a b : A), a = b ∨ a ≠ b
def is_distinct (A : Type) : Prop := ∀ (a b : A), a ≠ b → (a ≠ b)
def is_unordered (A : Type) : Prop := true  -- For simplicity, we assume unorderedness holds for any set

-- Property that verifies if the 26 letters of the English alphabet can form a set
def english_alphabet_set : Prop :=
  is_definite Char ∧ is_distinct Char ∧ is_unordered Char

theorem english_alphabet_is_set : english_alphabet_set :=
  sorry

end NUMINAMATH_GPT_english_alphabet_is_set_l881_88126


namespace NUMINAMATH_GPT_more_girls_than_boys_l881_88172

variables (boys girls : ℕ)

def ratio_condition : Prop := (3 * girls = 4 * boys)
def total_students_condition : Prop := (boys + girls = 42)

theorem more_girls_than_boys (h1 : ratio_condition boys girls) (h2 : total_students_condition boys girls) :
  (girls - boys = 6) :=
sorry

end NUMINAMATH_GPT_more_girls_than_boys_l881_88172


namespace NUMINAMATH_GPT_product_not_divisible_by_prime_l881_88150

theorem product_not_divisible_by_prime (p a b : ℕ) (hp : Prime p) (ha : 1 ≤ a) (hpa : a < p) (hb : 1 ≤ b) (hpb : b < p) : ¬ (p ∣ (a * b)) :=
by
  sorry

end NUMINAMATH_GPT_product_not_divisible_by_prime_l881_88150


namespace NUMINAMATH_GPT_amount_borrowed_eq_4137_84_l881_88138

noncomputable def compound_interest (initial : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  initial * (1 + rate/100) ^ time

theorem amount_borrowed_eq_4137_84 :
  ∃ P : ℝ, 
    (compound_interest (compound_interest (compound_interest P 6 3) 8 4) 10 2 = 8110) 
    ∧ (P = 4137.84) :=
by
  sorry

end NUMINAMATH_GPT_amount_borrowed_eq_4137_84_l881_88138


namespace NUMINAMATH_GPT_true_prop_count_l881_88187

-- Define the propositions
def original_prop (x : ℝ) : Prop := x > -3 → x > -6
def converse (x : ℝ) : Prop := x > -6 → x > -3
def inverse (x : ℝ) : Prop := x ≤ -3 → x ≤ -6
def contrapositive (x : ℝ) : Prop := x ≤ -6 → x ≤ -3

-- The statement to prove
theorem true_prop_count (x : ℝ) : 
  (original_prop x → true) ∧ (contrapositive x → true) ∧ ¬(converse x) ∧ ¬(inverse x) → 
  (count_true_propositions = 2) :=
sorry

end NUMINAMATH_GPT_true_prop_count_l881_88187


namespace NUMINAMATH_GPT_solve_equation_l881_88136

noncomputable def equation (x : ℝ) : Prop := x * (x - 2) + x - 2 = 0

theorem solve_equation : ∀ x, equation x ↔ (x = 2 ∨ x = -1) :=
by sorry

end NUMINAMATH_GPT_solve_equation_l881_88136


namespace NUMINAMATH_GPT_solve_for_x_l881_88182

theorem solve_for_x (x : ℝ) (hx : x^(1/10) * (x^(3/2))^(1/10) = 3) : x = 9 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l881_88182


namespace NUMINAMATH_GPT_solution_set_f_l881_88191

noncomputable def f (x : ℝ) : ℝ := sorry -- The differentiable function f

axiom f_deriv_lt (x : ℝ) : deriv f x < x -- Condition on the derivative of f
axiom f_at_2 : f 2 = 1 -- Given f(2) = 1

theorem solution_set_f : ∀ x : ℝ, f x < (1 / 2) * x^2 - 1 ↔ x > 2 :=
by sorry

end NUMINAMATH_GPT_solution_set_f_l881_88191


namespace NUMINAMATH_GPT_probability_of_one_black_ball_l881_88115

theorem probability_of_one_black_ball (total_balls black_balls white_balls drawn_balls : ℕ) 
  (h_total : total_balls = 4)
  (h_black : black_balls = 2)
  (h_white : white_balls = 2)
  (h_drawn : drawn_balls = 2) :
  ((Nat.choose black_balls 1) * (Nat.choose white_balls 1) : ℚ) / (Nat.choose total_balls drawn_balls) = 2 / 3 :=
by {
  -- Insert proof here
  sorry
}

end NUMINAMATH_GPT_probability_of_one_black_ball_l881_88115


namespace NUMINAMATH_GPT_time_to_paint_one_room_l881_88199

theorem time_to_paint_one_room (total_rooms : ℕ) (rooms_painted : ℕ) (time_remaining : ℕ) (rooms_left : ℕ) :
  total_rooms = 9 ∧ rooms_painted = 5 ∧ time_remaining = 32 ∧ rooms_left = total_rooms - rooms_painted → time_remaining / rooms_left = 8 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_time_to_paint_one_room_l881_88199


namespace NUMINAMATH_GPT_solve_for_x_l881_88171

theorem solve_for_x (x : ℝ) : 
  x - 3 * x + 5 * x = 150 → x = 50 :=
by
  intro h
  -- sorry to skip the proof
  sorry

end NUMINAMATH_GPT_solve_for_x_l881_88171


namespace NUMINAMATH_GPT_value_of_expression_l881_88140

theorem value_of_expression : 2 - (-2 : ℝ) ^ (-2 : ℝ) = 7 / 4 := 
by 
  sorry

end NUMINAMATH_GPT_value_of_expression_l881_88140


namespace NUMINAMATH_GPT_evalCeilingOfNegativeSqrt_l881_88106

noncomputable def ceiling_of_negative_sqrt : ℤ :=
  Int.ceil (-(Real.sqrt (36 / 9)))

theorem evalCeilingOfNegativeSqrt : ceiling_of_negative_sqrt = -2 := by
  sorry

end NUMINAMATH_GPT_evalCeilingOfNegativeSqrt_l881_88106


namespace NUMINAMATH_GPT_find_constant_a_range_of_f_l881_88151

noncomputable def f (a x : ℝ) : ℝ :=
  2 * a * (Real.sin x)^2 + 2 * (Real.sin x) * (Real.cos x) - a

theorem find_constant_a (h : f a 0 = -Real.sqrt 3) : a = Real.sqrt 3 := by
  sorry

theorem range_of_f (a : ℝ) (h : a = Real.sqrt 3) (x : ℝ) (hx : 0 ≤ x ∧ x ≤ Real.pi / 2) :
  f a x ∈ Set.Icc (-Real.sqrt 3) 2 := by
  sorry

end NUMINAMATH_GPT_find_constant_a_range_of_f_l881_88151


namespace NUMINAMATH_GPT_card_distribution_l881_88121

-- Definitions of the total cards and distribution rules
def total_cards : ℕ := 363

def ratio_xiaoming_xiaohua (k : ℕ) : Prop := ∃ x y, x = 7 * k ∧ y = 6 * k
def ratio_xiaogang_xiaoming (m : ℕ) : Prop := ∃ x z, z = 8 * m ∧ x = 5 * m

-- Final values to prove
def xiaoming_cards : ℕ := 105
def xiaohua_cards : ℕ := 90
def xiaogang_cards : ℕ := 168

-- The proof statement
theorem card_distribution (x y z k m : ℕ) 
  (hk : total_cards = 7 * k + 6 * k + 8 * m)
  (hx : ratio_xiaoming_xiaohua k)
  (hz : ratio_xiaogang_xiaoming m) :
  x = xiaoming_cards ∧ y = xiaohua_cards ∧ z = xiaogang_cards :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_card_distribution_l881_88121


namespace NUMINAMATH_GPT_neg_abs_neg_three_l881_88162

theorem neg_abs_neg_three : -|(-3)| = -3 := 
by
  sorry

end NUMINAMATH_GPT_neg_abs_neg_three_l881_88162


namespace NUMINAMATH_GPT_find_c_d_l881_88180

noncomputable def g (c d x : ℝ) : ℝ := c * x^3 + 5 * x^2 + d * x + 7

theorem find_c_d : ∃ (c d : ℝ), 
  (g c d 2 = 11) ∧ (g c d (-3) = 134) ∧ c = -35 / 13 ∧ d = 16 / 13 :=
  by
  sorry

end NUMINAMATH_GPT_find_c_d_l881_88180


namespace NUMINAMATH_GPT_vinegar_final_percentage_l881_88161

def vinegar_percentage (volume1 volume2 : ℕ) (percent1 percent2 : ℚ) : ℚ :=
  let vinegar1 := volume1 * percent1 / 100
  let vinegar2 := volume2 * percent2 / 100
  (vinegar1 + vinegar2) / (volume1 + volume2) * 100

theorem vinegar_final_percentage:
  vinegar_percentage 128 128 8 13 = 10.5 :=
  sorry

end NUMINAMATH_GPT_vinegar_final_percentage_l881_88161


namespace NUMINAMATH_GPT_max_valid_words_for_AU_language_l881_88147

noncomputable def maxValidWords : ℕ :=
  2^14 - 128

theorem max_valid_words_for_AU_language 
  (letters : Finset (String)) (validLengths : Set ℕ) (noConcatenation : Prop) :
  letters = {"a", "u"} ∧ validLengths = {n | 1 ≤ n ∧ n ≤ 13} ∧ noConcatenation →
  maxValidWords = 16256 :=
by
  sorry

end NUMINAMATH_GPT_max_valid_words_for_AU_language_l881_88147


namespace NUMINAMATH_GPT_tea_maker_capacity_l881_88194

theorem tea_maker_capacity (x : ℝ) (h : 0.45 * x = 54) : x = 120 :=
by
  sorry

end NUMINAMATH_GPT_tea_maker_capacity_l881_88194


namespace NUMINAMATH_GPT_find_xz_over_y_squared_l881_88198

variable {x y z : ℝ}

noncomputable def k : ℝ := 7

theorem find_xz_over_y_squared
    (h1 : x + k * y + 4 * z = 0)
    (h2 : 4 * x + k * y - 3 * z = 0)
    (h3 : x + 3 * y - 2 * z = 0)
    (h_nz : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0) :
    (x * z) / (y ^ 2) = 26 / 9 :=
by sorry

end NUMINAMATH_GPT_find_xz_over_y_squared_l881_88198


namespace NUMINAMATH_GPT_sum_of_squares_first_15_l881_88184

def sum_of_squares (n : ℕ) : ℕ := (n * (n + 1) * (2 * n + 1)) / 6

theorem sum_of_squares_first_15 : sum_of_squares 15 = 3720 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_first_15_l881_88184


namespace NUMINAMATH_GPT_maxwell_distance_traveled_l881_88160

theorem maxwell_distance_traveled
  (distance_between_homes : ℕ)
  (maxwell_speed : ℕ)
  (brad_speed : ℕ)
  (meeting_time : ℕ)
  (h1 : distance_between_homes = 72)
  (h2 : maxwell_speed = 6)
  (h3 : brad_speed = 12)
  (h4 : meeting_time = distance_between_homes / (maxwell_speed + brad_speed)) :
  maxwell_speed * meeting_time = 24 :=
by
  sorry

end NUMINAMATH_GPT_maxwell_distance_traveled_l881_88160


namespace NUMINAMATH_GPT_problem_statement_l881_88119

variables (f : ℝ → ℝ)

def is_odd (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x
def condition (f : ℝ → ℝ) := ∀ x : ℝ, f x = f (2 - x)

theorem problem_statement (h_odd : is_odd f) (h_cond : condition f) : f 2010 = 0 := 
sorry

end NUMINAMATH_GPT_problem_statement_l881_88119


namespace NUMINAMATH_GPT_village_male_population_l881_88189

theorem village_male_population (total_population parts male_parts : ℕ) (h1 : total_population = 600) (h2 : parts = 4) (h3 : male_parts = 2) :
  male_parts * (total_population / parts) = 300 :=
by
  -- We are stating the problem as per the given conditions
  sorry

end NUMINAMATH_GPT_village_male_population_l881_88189


namespace NUMINAMATH_GPT_number_of_birds_seen_l881_88144

theorem number_of_birds_seen (dozens_seen : ℕ) (birds_per_dozen : ℕ) (h₀ : dozens_seen = 8) (h₁ : birds_per_dozen = 12) : dozens_seen * birds_per_dozen = 96 :=
by sorry

end NUMINAMATH_GPT_number_of_birds_seen_l881_88144


namespace NUMINAMATH_GPT_range_of_a_l881_88113

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, ¬ (x^2 - a * x + 1 ≤ 0)) ↔ -2 < a ∧ a < 2 := 
sorry

end NUMINAMATH_GPT_range_of_a_l881_88113


namespace NUMINAMATH_GPT_line_tangent_to_ellipse_l881_88134

theorem line_tangent_to_ellipse (m : ℝ) : 
  (∀ x y : ℝ, y = mx + 2 → x^2 + 9 * y^2 = 9 → ∃ u, y = u) → m^2 = 1 / 3 := 
by
  intro h
  sorry

end NUMINAMATH_GPT_line_tangent_to_ellipse_l881_88134


namespace NUMINAMATH_GPT_smallest_possible_value_of_M_l881_88139

theorem smallest_possible_value_of_M :
  ∀ (a b c d e f : ℕ), a > 0 → b > 0 → c > 0 → d > 0 → e > 0 → f > 0 →
  a + b + c + d + e + f = 4020 →
  (∃ M : ℕ, M = max (a + b) (max (b + c) (max (c + d) (max (d + e) (e + f)))) ∧
    (∀ (M' : ℕ), (∀ (a b c d e f : ℕ), a > 0 → b > 0 → c > 0 → d > 0 → e > 0 → f > 0 →
      a + b + c + d + e + f = 4020 →
      M' = max (a + b) (max (b + c) (max (c + d) (max (d + e) (e + f)))) → M' ≥ 804) → M = 804)) := by
  sorry

end NUMINAMATH_GPT_smallest_possible_value_of_M_l881_88139


namespace NUMINAMATH_GPT_discount_on_soap_l881_88125

theorem discount_on_soap :
  (let chlorine_price := 10
   let chlorine_discount := 0.20 * chlorine_price
   let discounted_chlorine_price := chlorine_price - chlorine_discount

   let soap_price := 16

   let total_savings := 26

   let chlorine_savings := 3 * chlorine_price - 3 * discounted_chlorine_price
   let soap_savings := total_savings - chlorine_savings

   let discount_per_soap := soap_savings / 5
   let discount_percentage_per_soap := (discount_per_soap / soap_price) * 100
   discount_percentage_per_soap = 25) := sorry

end NUMINAMATH_GPT_discount_on_soap_l881_88125


namespace NUMINAMATH_GPT_rest_area_milepost_l881_88163

theorem rest_area_milepost : 
  let fifth_exit := 30
  let fifteenth_exit := 210
  (3 / 5) * (fifteenth_exit - fifth_exit) + fifth_exit = 138 := 
by 
  let fifth_exit := 30
  let fifteenth_exit := 210
  sorry

end NUMINAMATH_GPT_rest_area_milepost_l881_88163


namespace NUMINAMATH_GPT_minimize_potato_cost_l881_88179

def potatoes_distribution (x1 x2 x3 : ℚ) : Prop :=
  x1 ≥ 0 ∧ x2 ≥ 0 ∧ x3 ≥ 0 ∧
  x1 + x2 + x3 = 12 ∧
  x1 + 4 * x2 + 3 * x3 ≤ 40 ∧
  x1 ≤ 10 ∧ x2 ≤ 8 ∧ x3 ≤ 6 ∧
  4 * x1 + 3 * x2 + 1 * x3 = (74 / 3)

theorem minimize_potato_cost :
  ∃ x1 x2 x3 : ℚ, potatoes_distribution x1 x2 x3 ∧ x1 = (2/3) ∧ x2 = (16/3) ∧ x3 = 6 :=
by
  sorry

end NUMINAMATH_GPT_minimize_potato_cost_l881_88179


namespace NUMINAMATH_GPT_exists_A_for_sqrt_d_l881_88173

def is_not_perfect_square (d : ℕ) : Prop := ∀ m : ℕ, m * m ≠ d

def s (d n : ℕ) : ℕ := 
  -- count number of 1's in the first n digits of binary representation of √d
  sorry 

theorem exists_A_for_sqrt_d (d : ℕ) (h : is_not_perfect_square d) :
  ∃ A : ℕ, ∀ n ≥ A, s d n > Int.sqrt (2 * n) - 2 :=
sorry

end NUMINAMATH_GPT_exists_A_for_sqrt_d_l881_88173


namespace NUMINAMATH_GPT_race_meeting_time_l881_88114

noncomputable def track_length : ℕ := 500
noncomputable def first_meeting_from_marie_start : ℕ := 100
noncomputable def time_until_first_meeting : ℕ := 2
noncomputable def second_meeting_time : ℕ := 12

theorem race_meeting_time
  (h1 : track_length = 500)
  (h2 : first_meeting_from_marie_start = 100)
  (h3 : time_until_first_meeting = 2)
  (h4 : ∀ t v1 v2 : ℕ, t * (v1 + v2) = track_length)
  (h5 : 12 = second_meeting_time) :
  second_meeting_time = 12 := by
  sorry

end NUMINAMATH_GPT_race_meeting_time_l881_88114


namespace NUMINAMATH_GPT_prime_factors_sum_correct_prime_factors_product_correct_l881_88120

-- The number we are considering
def n : ℕ := 172480

-- Prime factors of the number n
def prime_factors : List ℕ := [2, 3, 5, 719]

-- Sum of the prime factors
def sum_prime_factors : ℕ := 2 + 3 + 5 + 719

-- Product of the prime factors
def prod_prime_factors : ℕ := 2 * 3 * 5 * 719

theorem prime_factors_sum_correct :
  sum_prime_factors = 729 :=
by {
  -- Proof goes here
  sorry
}

theorem prime_factors_product_correct :
  prod_prime_factors = 21570 :=
by {
  -- Proof goes here
  sorry
}

end NUMINAMATH_GPT_prime_factors_sum_correct_prime_factors_product_correct_l881_88120


namespace NUMINAMATH_GPT_plan_y_cost_effective_l881_88112

theorem plan_y_cost_effective (m : ℕ) (h1 : ∀ minutes, cost_plan_x = 15 * minutes)
(h2 : ∀ minutes, cost_plan_y = 3000 + 10 * minutes) :
m ≥ 601 → 3000 + 10 * m < 15 * m :=
by
sorry

end NUMINAMATH_GPT_plan_y_cost_effective_l881_88112


namespace NUMINAMATH_GPT_probability_two_white_balls_l881_88152

noncomputable def probability_of_two_white_balls (total_balls white_balls black_balls: ℕ) : ℚ :=
  if white_balls + black_balls = total_balls ∧ total_balls = 15 ∧ white_balls = 7 ∧ black_balls = 8 then
    (white_balls / total_balls) * ((white_balls - 1) / (total_balls - 1))
  else 0

theorem probability_two_white_balls : 
  probability_of_two_white_balls 15 7 8 = 1/5
:= sorry

end NUMINAMATH_GPT_probability_two_white_balls_l881_88152
