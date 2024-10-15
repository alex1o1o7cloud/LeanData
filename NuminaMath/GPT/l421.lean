import Mathlib

namespace NUMINAMATH_GPT_bus_capacity_total_kids_l421_42168

-- Definitions based on conditions
def total_rows : ℕ := 25
def lower_deck_rows : ℕ := 15
def upper_deck_rows : ℕ := 10
def lower_deck_capacity_per_row : ℕ := 5
def upper_deck_capacity_per_row : ℕ := 3
def staff_members : ℕ := 4

-- Theorem statement
theorem bus_capacity_total_kids : 
  (lower_deck_rows * lower_deck_capacity_per_row) + 
  (upper_deck_rows * upper_deck_capacity_per_row) - staff_members = 101 := 
by
  sorry

end NUMINAMATH_GPT_bus_capacity_total_kids_l421_42168


namespace NUMINAMATH_GPT_abs_neg_one_eq_one_l421_42191

theorem abs_neg_one_eq_one : abs (-1 : ℚ) = 1 := 
by
  sorry

end NUMINAMATH_GPT_abs_neg_one_eq_one_l421_42191


namespace NUMINAMATH_GPT_max_value_expression_l421_42194

theorem max_value_expression (a b c d : ℝ) 
  (h1 : -11.5 ≤ a ∧ a ≤ 11.5)
  (h2 : -11.5 ≤ b ∧ b ≤ 11.5)
  (h3 : -11.5 ≤ c ∧ c ≤ 11.5)
  (h4 : -11.5 ≤ d ∧ d ≤ 11.5):
  a + 2 * b + c + 2 * d - a * b - b * c - c * d - d * a ≤ 552 :=
by
  sorry

end NUMINAMATH_GPT_max_value_expression_l421_42194


namespace NUMINAMATH_GPT_points_satisfying_inequality_l421_42173

theorem points_satisfying_inequality (x y : ℝ) :
  ( ( (x * y + 1) / (x + y) )^2 < 1) ↔ 
  ( (-1 < x ∧ x < 1) ∧ (y < -1 ∨ y > 1) ) ∨ 
  ( (x < -1 ∨ x > 1) ∧ (-1 < y ∧ y < 1) ) := 
sorry

end NUMINAMATH_GPT_points_satisfying_inequality_l421_42173


namespace NUMINAMATH_GPT_geometric_sequence_sum_l421_42139

noncomputable def geometric_sequence (a : ℕ → ℝ) (r: ℝ): Prop :=
  ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) (r: ℝ)
  (h_geometric : geometric_sequence a r)
  (h_ratio : r = 2)
  (h_sum_condition : a 1 + a 4 + a 7 = 10) :
  a 3 + a 6 + a 9 = 20 := 
sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l421_42139


namespace NUMINAMATH_GPT_simplified_expression_term_count_l421_42166

def even_exponents_terms_count : ℕ :=
  let n := 2008
  let k := 1004
  Nat.choose (k + 2) 2

theorem simplified_expression_term_count :
  even_exponents_terms_count = 505815 :=
sorry

end NUMINAMATH_GPT_simplified_expression_term_count_l421_42166


namespace NUMINAMATH_GPT_jasper_sold_31_drinks_l421_42180

def chips := 27
def hot_dogs := chips - 8
def drinks := hot_dogs + 12

theorem jasper_sold_31_drinks : drinks = 31 := by
  sorry

end NUMINAMATH_GPT_jasper_sold_31_drinks_l421_42180


namespace NUMINAMATH_GPT_intersection_one_point_l421_42179

open Set

def A (x y : ℝ) : Prop := x^2 - 3*x*y + 4*y^2 = 7 / 2
def B (k x y : ℝ) : Prop := k > 0 ∧ k*x + y = 2

theorem intersection_one_point (k : ℝ) (h : k > 0) :
  (∃ x y : ℝ, A x y ∧ B k x y) → (∀ x₁ y₁ x₂ y₂ : ℝ, (A x₁ y₁ ∧ B k x₁ y₁) ∧ (A x₂ y₂ ∧ B k x₂ y₂) → x₁ = x₂ ∧ y₁ = y₂) ↔ k = 1 / 4 :=
sorry

end NUMINAMATH_GPT_intersection_one_point_l421_42179


namespace NUMINAMATH_GPT_largest_class_students_l421_42186

theorem largest_class_students (n1 n2 n3 n4 n5 : ℕ) (h1 : n1 = x) (h2 : n2 = x - 2) (h3 : n3 = x - 4) (h4 : n4 = x - 6) (h5 : n5 = x - 8) (h_sum : n1 + n2 + n3 + n4 + n5 = 140) : x = 32 :=
by {
  sorry
}

end NUMINAMATH_GPT_largest_class_students_l421_42186


namespace NUMINAMATH_GPT_complement_intersection_l421_42114

open Set

variable {U : Set ℝ} (A B : Set ℝ)

def A_def : Set ℝ := { x | 3 ≤ x ∧ x < 7 }
def B_def : Set ℝ := { x | 2 < x ∧ x < 10 }

theorem complement_intersection :
  (U = univ ∧ A = A_def ∧ B = B_def) →
  (compl (A ∩ B) = {x | x < 3 ∨ x ≥ 7}) :=
by
  sorry

end NUMINAMATH_GPT_complement_intersection_l421_42114


namespace NUMINAMATH_GPT_speed_in_still_water_l421_42137

-- Given conditions
def upstream_speed : ℝ := 60
def downstream_speed : ℝ := 90

-- Proof that the speed of the man in still water is 75 kmph
theorem speed_in_still_water :
  (upstream_speed + downstream_speed) / 2 = 75 := 
by
  sorry

end NUMINAMATH_GPT_speed_in_still_water_l421_42137


namespace NUMINAMATH_GPT_time_after_midnight_1453_minutes_l421_42181

def minutes_to_time (minutes : Nat) : Nat × Nat :=
  let hours := minutes / 60
  let remaining_minutes := minutes % 60
  (hours, remaining_minutes)

def time_of_day (hours : Nat) : Nat × Nat :=
  let days := hours / 24
  let remaining_hours := hours % 24
  (days, remaining_hours)

theorem time_after_midnight_1453_minutes : 
  let midnight := (0, 0) -- Midnight as a tuple of hours and minutes
  let total_minutes := 1453
  let (total_hours, minutes) := minutes_to_time total_minutes
  let (days, hours) := time_of_day total_hours
  days = 1 ∧ hours = 0 ∧ minutes = 13
  := by
    let midnight := (0, 0)
    let total_minutes := 1453
    let (total_hours, minutes) := minutes_to_time total_minutes
    let (days, hours) := time_of_day total_hours
    sorry

end NUMINAMATH_GPT_time_after_midnight_1453_minutes_l421_42181


namespace NUMINAMATH_GPT_num_elements_intersection_l421_42159

def A : Finset ℕ := {1, 2, 3, 4}
def B : Finset ℕ := {2, 4, 6, 8}

theorem num_elements_intersection : (A ∩ B).card = 2 := by
  sorry

end NUMINAMATH_GPT_num_elements_intersection_l421_42159


namespace NUMINAMATH_GPT_evaluate_expression_l421_42195

theorem evaluate_expression (a : ℕ) (h : a = 3) : a^2 * a^5 = 2187 :=
by sorry

end NUMINAMATH_GPT_evaluate_expression_l421_42195


namespace NUMINAMATH_GPT_train_passes_jogger_time_l421_42144

theorem train_passes_jogger_time (speed_jogger_kmph : ℝ) 
                                (speed_train_kmph : ℝ) 
                                (distance_ahead_m : ℝ) 
                                (length_train_m : ℝ) : 
  speed_jogger_kmph = 9 → 
  speed_train_kmph = 45 →
  distance_ahead_m = 250 →
  length_train_m = 120 →
  (distance_ahead_m + length_train_m) / (speed_train_kmph - speed_jogger_kmph) * (1000 / 3600) = 37 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_train_passes_jogger_time_l421_42144


namespace NUMINAMATH_GPT_at_least_one_negative_root_l421_42171

theorem at_least_one_negative_root (a : ℝ) :
  (∃ x : ℝ, x < 0 ∧ (x^2 - 6*a*x - 2 + 2*a + 9*a^2 = 0)) ↔ a < (-1 + Real.sqrt 19) / 9 := by
  sorry

end NUMINAMATH_GPT_at_least_one_negative_root_l421_42171


namespace NUMINAMATH_GPT_twenty_five_percent_less_than_eighty_equals_one_fourth_more_than_what_number_l421_42163

theorem twenty_five_percent_less_than_eighty_equals_one_fourth_more_than_what_number :
  ∃ n : ℝ, (80 - 0.25 * 80) = (5 / 4) * n ∧ n = 48 := 
by
  sorry

end NUMINAMATH_GPT_twenty_five_percent_less_than_eighty_equals_one_fourth_more_than_what_number_l421_42163


namespace NUMINAMATH_GPT_three_digit_cubes_divisible_by_4_l421_42150

-- Let's define the conditions in Lean
def is_three_digit (n : ℕ) : Prop := n ≥ 100 ∧ n ≤ 999
def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k^3 = n
def is_divisible_by_4 (n : ℕ) : Prop := n % 4 = 0

-- Let's combine these conditions to define the target predicate in Lean
def is_target_number (n : ℕ) : Prop := is_three_digit n ∧ is_perfect_cube n ∧ is_divisible_by_4 n

-- The statement to be proven: that there is only one such number
theorem three_digit_cubes_divisible_by_4 : 
  (∃! n, is_target_number n) :=
sorry

end NUMINAMATH_GPT_three_digit_cubes_divisible_by_4_l421_42150


namespace NUMINAMATH_GPT_contrapositive_example_l421_42123

theorem contrapositive_example (x : ℝ) (h : x = 1 → x^2 - 3 * x + 2 = 0) :
  x^2 - 3 * x + 2 ≠ 0 → x ≠ 1 :=
by
  intro h₀
  intro h₁
  have h₂ := h h₁
  contradiction

end NUMINAMATH_GPT_contrapositive_example_l421_42123


namespace NUMINAMATH_GPT_tom_age_l421_42190

theorem tom_age (S T : ℕ) (h1 : T = 2 * S - 1) (h2 : T + S = 14) : T = 9 := by
  sorry

end NUMINAMATH_GPT_tom_age_l421_42190


namespace NUMINAMATH_GPT_find_x_l421_42154

theorem find_x (x : ℕ) (h : 1 + 2 + 3 + 4 + 5 + x = 21 + 22 + 23 + 24 + 25) : x = 100 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_x_l421_42154


namespace NUMINAMATH_GPT_rectangles_equal_area_implies_value_l421_42110

theorem rectangles_equal_area_implies_value (x y : ℝ) (h1 : x < 9) (h2 : y < 4)
  (h3 : x * (4 - y) = y * (9 - x)) : 360 * x / y = 810 :=
by
  -- We only need to state the theorem, the proof is not required.
  sorry

end NUMINAMATH_GPT_rectangles_equal_area_implies_value_l421_42110


namespace NUMINAMATH_GPT_last_ball_probability_l421_42167

theorem last_ball_probability (w b : ℕ) (H : w > 0 ∨ b > 0) :
  (w % 2 = 1 → ∃ p : ℝ, p = 1 ∧ (∃ n, (∀ (k : ℕ), k < n → (sorry))) ) ∧ 
  (w % 2 = 0 → ∃ p : ℝ, p = 0 ∧ (∃ n, (∀ (k : ℕ), k < n → (sorry))) ) :=
by sorry

end NUMINAMATH_GPT_last_ball_probability_l421_42167


namespace NUMINAMATH_GPT_sphere_ratios_l421_42103

theorem sphere_ratios (r1 r2 : ℝ) (h : r1 / r2 = 1 / 3) :
  (4 * π * r1^2) / (4 * π * r2^2) = 1 / 9 ∧ (4 / 3 * π * r1^3) / (4 / 3 * π * r2^3) = 1 / 27 :=
by
  sorry

end NUMINAMATH_GPT_sphere_ratios_l421_42103


namespace NUMINAMATH_GPT_books_taken_off_l421_42119

def books_initially : ℝ := 38.0
def books_remaining : ℝ := 28.0

theorem books_taken_off : books_initially - books_remaining = 10 := by
  sorry

end NUMINAMATH_GPT_books_taken_off_l421_42119


namespace NUMINAMATH_GPT_botanical_garden_correct_path_length_l421_42182

noncomputable def correct_path_length_on_ground
  (inch_length_on_map : ℝ)
  (inch_per_error_segment : ℝ)
  (conversion_rate : ℝ) : ℝ :=
  (inch_length_on_map * conversion_rate) - (inch_per_error_segment * conversion_rate)

theorem botanical_garden_correct_path_length :
  correct_path_length_on_ground 6.5 0.75 1200 = 6900 := 
by
  sorry

end NUMINAMATH_GPT_botanical_garden_correct_path_length_l421_42182


namespace NUMINAMATH_GPT_lisa_time_to_complete_l421_42138

theorem lisa_time_to_complete 
  (hotdogs_record : ℕ) 
  (eaten_so_far : ℕ) 
  (rate_per_minute : ℕ) 
  (remaining_hotdogs : ℕ) 
  (time_to_complete : ℕ) 
  (h1 : hotdogs_record = 75) 
  (h2 : eaten_so_far = 20) 
  (h3 : rate_per_minute = 11) 
  (h4 : remaining_hotdogs = hotdogs_record - eaten_so_far)
  (h5 : time_to_complete = remaining_hotdogs / rate_per_minute) :
  time_to_complete = 5 :=
sorry

end NUMINAMATH_GPT_lisa_time_to_complete_l421_42138


namespace NUMINAMATH_GPT_tan_ratio_l421_42135

open Real

variable (x y : ℝ)

-- Conditions
def sin_add_eq : sin (x + y) = 5 / 8 := sorry
def sin_sub_eq : sin (x - y) = 1 / 4 := sorry

-- Proof problem statement
theorem tan_ratio : sin (x + y) = 5 / 8 → sin (x - y) = 1 / 4 → (tan x) / (tan y) = 2 := sorry

end NUMINAMATH_GPT_tan_ratio_l421_42135


namespace NUMINAMATH_GPT_solve_for_y_l421_42170

theorem solve_for_y (y : ℝ) (h : 6 * y^(1/3) - 3 * (y / y^(2/3)) = 10 + 2 * y^(1/3)) : y = 1000 := 
by
  sorry

end NUMINAMATH_GPT_solve_for_y_l421_42170


namespace NUMINAMATH_GPT_goal_l421_42189

-- Definitions based on the conditions
def condition1 (x : ℕ) : Prop := 10 < x ∧ x < 17
def condition2 (x : ℕ) : Prop := 11 < x ∧ x < 18
def condition3 (x : ℕ) : Prop := x % 2 = 1

-- Definition for checking if exactly two conditions hold
def exactly_two_holds (h1 h2 h3 : Prop) : Prop :=
  (h1 ∧ h2 ∧ ¬h3) ∨ (h1 ∧ ¬h2 ∧ h3) ∨ (¬h1 ∧ h2 ∧ h3)

-- Main goal: find x where exactly two out of three conditions hold
theorem goal (x : ℕ) : exactly_two_holds (condition1 x) (condition2 x) (condition3 x) ↔ 
  x = 11 ∨ x = 12 ∨ x = 14 ∨ x = 16 ∨ x = 17 :=
sorry

end NUMINAMATH_GPT_goal_l421_42189


namespace NUMINAMATH_GPT_triangle_ABC_properties_l421_42126

theorem triangle_ABC_properties 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h1 : 2 * Real.sin B * Real.sin C * Real.cos A + Real.cos A = 3 * Real.sin A ^ 2 - Real.cos (B - C)) : 
  (2 * a = b + c) ∧ 
  (b + c = 2) →
  (Real.cos A = 3/5) → 
  (1 / 2 * b * c * Real.sin A = 3 / 8) :=
by
  sorry

end NUMINAMATH_GPT_triangle_ABC_properties_l421_42126


namespace NUMINAMATH_GPT_equation_verification_l421_42175

theorem equation_verification :
  (96 / 12 = 8) ∧ (45 - 37 = 8) := 
by
  -- We can add the necessary proofs later
  sorry

end NUMINAMATH_GPT_equation_verification_l421_42175


namespace NUMINAMATH_GPT_manfred_total_paychecks_l421_42111

-- Define the conditions
def first_paychecks : ℕ := 6
def first_paycheck_amount : ℕ := 750
def remaining_paycheck_amount : ℕ := first_paycheck_amount + 20
def average_amount : ℝ := 765.38

-- Main theorem statement
theorem manfred_total_paychecks (x : ℕ) (h : (first_paychecks * first_paycheck_amount + x * remaining_paycheck_amount) / (first_paychecks + x) = average_amount) : first_paychecks + x = 26 :=
by
  sorry

end NUMINAMATH_GPT_manfred_total_paychecks_l421_42111


namespace NUMINAMATH_GPT_elder_three_times_younger_l421_42142

-- Definitions based on conditions
def age_difference := 16
def elder_present_age := 30
def younger_present_age := elder_present_age - age_difference

-- The problem statement to prove the correct value of n (years ago)
theorem elder_three_times_younger (n : ℕ) 
  (h1 : elder_present_age = younger_present_age + age_difference)
  (h2 : elder_present_age - n = 3 * (younger_present_age - n)) : 
  n = 6 := 
sorry

end NUMINAMATH_GPT_elder_three_times_younger_l421_42142


namespace NUMINAMATH_GPT_solve_for_y_l421_42193

theorem solve_for_y (y : ℝ) (h : 7 - y = 12) : y = -5 := sorry

end NUMINAMATH_GPT_solve_for_y_l421_42193


namespace NUMINAMATH_GPT_min_value_f_l421_42187

-- Define the function f(x)
def f (x : ℝ) : ℝ := (15 - x) * (13 - x) * (15 + x) * (13 + x) + 200 * x^2

-- State the theorem to be proved
theorem min_value_f : ∃ (x : ℝ), (∀ y : ℝ, f y ≥ 33) ∧ f x = 33 := by
  sorry

end NUMINAMATH_GPT_min_value_f_l421_42187


namespace NUMINAMATH_GPT_opposite_of_seven_l421_42183

theorem opposite_of_seven : ∃ x : ℤ, 7 + x = 0 ∧ x = -7 :=
by
  sorry

end NUMINAMATH_GPT_opposite_of_seven_l421_42183


namespace NUMINAMATH_GPT_gcd_subtraction_result_l421_42131

theorem gcd_subtraction_result : gcd 8100 270 - 8 = 262 := by
  sorry

end NUMINAMATH_GPT_gcd_subtraction_result_l421_42131


namespace NUMINAMATH_GPT_both_owners_count_l421_42106

-- Define the sets and counts as given in the conditions
variable (total_students : ℕ) (rabbit_owners : ℕ) (guinea_pig_owners : ℕ) (both_owners : ℕ)

-- Assume the values given in the problem
axiom total : total_students = 50
axiom rabbits : rabbit_owners = 35
axiom guinea_pigs : guinea_pig_owners = 40

-- The theorem to prove
theorem both_owners_count : both_owners = rabbit_owners + guinea_pig_owners - total_students := by
  sorry

end NUMINAMATH_GPT_both_owners_count_l421_42106


namespace NUMINAMATH_GPT_bart_earned_14_l421_42132

variable (questions_per_survey money_per_question surveys_monday surveys_tuesday : ℕ → ℝ)
variable (total_surveys total_questions money_earned : ℕ → ℝ)

noncomputable def conditions :=
  let questions_per_survey := 10
  let money_per_question := 0.2
  let surveys_monday := 3
  let surveys_tuesday := 4
  let total_surveys := surveys_monday + surveys_tuesday
  let total_questions := questions_per_survey * total_surveys
  let money_earned := total_questions * money_per_question
  money_earned = 14

theorem bart_earned_14 : conditions :=
by
  -- proof steps
  sorry

end NUMINAMATH_GPT_bart_earned_14_l421_42132


namespace NUMINAMATH_GPT_intersection_of_A_and_complement_B_l421_42188

def A : Set ℝ := {1, 2, 3, 4, 5}
def B : Set ℝ := {x | x < 3}
def complement_B : Set ℝ := {x | x ≥ 3}

theorem intersection_of_A_and_complement_B : A ∩ complement_B = {3, 4, 5} :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_complement_B_l421_42188


namespace NUMINAMATH_GPT_letter_lock_rings_l421_42177

theorem letter_lock_rings (n : ℕ) (h : n^3 - 1 ≤ 215) : n = 6 :=
by { sorry }

end NUMINAMATH_GPT_letter_lock_rings_l421_42177


namespace NUMINAMATH_GPT_total_fish_correct_l421_42128

def Leo_fish := 40
def Agrey_fish := Leo_fish + 20
def Sierra_fish := Agrey_fish + 15
def total_fish := Leo_fish + Agrey_fish + Sierra_fish

theorem total_fish_correct : total_fish = 175 := by
  sorry


end NUMINAMATH_GPT_total_fish_correct_l421_42128


namespace NUMINAMATH_GPT_cookies_indeterminate_l421_42169

theorem cookies_indeterminate (bananas : ℕ) (boxes : ℕ) (bananas_per_box : ℕ) (cookies : ℕ)
  (h1 : bananas = 40)
  (h2 : boxes = 8)
  (h3 : bananas_per_box = 5)
  : ∃ c : ℕ, c = cookies :=
by sorry

end NUMINAMATH_GPT_cookies_indeterminate_l421_42169


namespace NUMINAMATH_GPT_geometric_sequence_sum_point_on_line_l421_42108

theorem geometric_sequence_sum_point_on_line
  (S : ℕ → ℝ) (a : ℕ → ℝ) (t : ℝ) (r : ℝ)
  (h1 : a 1 = t)
  (h2 : ∀ n : ℕ, a (n + 1) = t * r ^ n)
  (h3 : ∀ n : ℕ, S n = t * (1 - r ^ n) / (1 - r))
  (h4 : ∀ n : ℕ, (S n, a (n + 1)) ∈ {p : ℝ × ℝ | p.2 = 2 * p.1 + 1})
  : t = 1 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_point_on_line_l421_42108


namespace NUMINAMATH_GPT_intersection_of_P_with_complement_Q_l421_42130

-- Define the universal set U, and sets P and Q
def U : List ℕ := [1, 2, 3, 4]
def P : List ℕ := [1, 2]
def Q : List ℕ := [2, 3]

-- Define the complement of Q with respect to U
def complement (U Q : List ℕ) : List ℕ := U.filter (λ x => x ∉ Q)

-- Define the intersection of two sets
def intersection (A B : List ℕ) : List ℕ := A.filter (λ x => x ∈ B)

-- The proof statement we need to show
theorem intersection_of_P_with_complement_Q : intersection P (complement U Q) = [1] := by
  sorry

end NUMINAMATH_GPT_intersection_of_P_with_complement_Q_l421_42130


namespace NUMINAMATH_GPT_age_difference_l421_42127

def age1 : ℕ := 10
def age2 : ℕ := age1 - 2
def age3 : ℕ := age2 + 4
def age4 : ℕ := age3 / 2
def age5 : ℕ := age4 + 20
def avg : ℕ := (age1 + age5) / 2

theorem age_difference :
  (age3 - age2) = 4 ∧ avg = 18 := by
  sorry

end NUMINAMATH_GPT_age_difference_l421_42127


namespace NUMINAMATH_GPT_quadruple_equation_solution_count_l421_42113

theorem quadruple_equation_solution_count (
    a b c d : ℕ
) (h_pos: a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) 
  (h_order: a < b ∧ b < c ∧ c < d) 
  (h_equation: 2 * a + 2 * b + 2 * c + 2 * d = d^2 - c^2 + b^2 - a^2) : 
  num_correct_statements = 2 :=
sorry

end NUMINAMATH_GPT_quadruple_equation_solution_count_l421_42113


namespace NUMINAMATH_GPT_greatest_prime_factor_of_5_pow_7_plus_6_pow_6_l421_42116

def greatest_prime_factor (n : ℕ) : ℕ :=
  sorry -- Implementation of finding greatest prime factor goes here

theorem greatest_prime_factor_of_5_pow_7_plus_6_pow_6 : 
  greatest_prime_factor (5^7 + 6^6) = 211 := 
by 
  sorry -- Proof of the theorem goes here

end NUMINAMATH_GPT_greatest_prime_factor_of_5_pow_7_plus_6_pow_6_l421_42116


namespace NUMINAMATH_GPT_expression_value_l421_42112

-- Step c: Definitions based on conditions
def base1 : ℤ := -2
def exponent1 : ℕ := 4^2
def base2 : ℕ := 1
def exponent2 : ℕ := 3^3

-- The Lean statement for the problem
theorem expression_value :
  base1 ^ exponent1 + base2 ^ exponent2 = 65537 := by
  sorry

end NUMINAMATH_GPT_expression_value_l421_42112


namespace NUMINAMATH_GPT_find_same_color_integers_l421_42153

variable (Color : Type) (red blue green yellow : Color)

theorem find_same_color_integers
  (color : ℤ → Color)
  (m n : ℤ)
  (hm : Odd m)
  (hn : Odd n)
  (h_not_zero : m + n ≠ 0) :
  ∃ a b : ℤ, color a = color b ∧ (a - b = m ∨ a - b = n ∨ a - b = m + n ∨ a - b = m - n) :=
sorry

end NUMINAMATH_GPT_find_same_color_integers_l421_42153


namespace NUMINAMATH_GPT_no_n_repeats_stock_price_l421_42136

-- Problem statement translation
theorem no_n_repeats_stock_price (n : ℕ) (h1 : n < 100) : ¬ ∃ k l : ℕ, (100 + n) ^ k * (100 - n) ^ l = 100 ^ (k + l) :=
by
  sorry

end NUMINAMATH_GPT_no_n_repeats_stock_price_l421_42136


namespace NUMINAMATH_GPT_election_votes_l421_42147

theorem election_votes (V : ℝ) 
    (h1 : ∃ c1 c2 : ℝ, c1 + c2 = V ∧ c1 = 0.60 * V ∧ c2 = 0.40 * V)
    (h2 : ∃ m : ℝ, m = 280 ∧ 0.60 * V - 0.40 * V = m) : 
    V = 1400 :=
by
  sorry

end NUMINAMATH_GPT_election_votes_l421_42147


namespace NUMINAMATH_GPT_sin_alpha_plus_7pi_over_12_l421_42185

theorem sin_alpha_plus_7pi_over_12 (α : Real) 
  (h1 : Real.cos (α + π / 12) = 1 / 5) : 
  Real.sin (α + 7 * π / 12) = 1 / 5 :=
by
  sorry

end NUMINAMATH_GPT_sin_alpha_plus_7pi_over_12_l421_42185


namespace NUMINAMATH_GPT_average_weight_estimate_l421_42125

noncomputable def average_weight (female_students male_students : ℕ) (avg_weight_female avg_weight_male : ℕ) : ℝ :=
  (female_students / (female_students + male_students) : ℝ) * avg_weight_female +
  (male_students / (female_students + male_students) : ℝ) * avg_weight_male

theorem average_weight_estimate :
  average_weight 504 596 49 57 = (504 / 1100 : ℝ) * 49 + (596 / 1100 : ℝ) * 57 :=
by
  sorry

end NUMINAMATH_GPT_average_weight_estimate_l421_42125


namespace NUMINAMATH_GPT_average_mark_of_remaining_students_l421_42134

theorem average_mark_of_remaining_students
  (n : ℕ) (A : ℕ) (m : ℕ) (B : ℕ) (total_students : n = 10)
  (avg_class : A = 80) (excluded_students : m = 5) (avg_excluded : B = 70) :
  (A * n - B * m) / (n - m) = 90 :=
by
  sorry

end NUMINAMATH_GPT_average_mark_of_remaining_students_l421_42134


namespace NUMINAMATH_GPT_page_copy_cost_l421_42145

theorem page_copy_cost (cost_per_4_pages : ℕ) (page_count : ℕ) (dollar_to_cents : ℕ) : cost_per_4_pages = 8 → page_count = 4 → dollar_to_cents = 100 → (1500 * (page_count / cost_per_4_pages) = 750) :=
by
  intros
  sorry

end NUMINAMATH_GPT_page_copy_cost_l421_42145


namespace NUMINAMATH_GPT_change_in_nickels_l421_42172

theorem change_in_nickels (cost_bread cost_cheese given_amount : ℝ) (quarters dimes : ℕ) (nickel_value : ℝ) 
  (h1 : cost_bread = 4.2) (h2 : cost_cheese = 2.05) (h3 : given_amount = 7.0)
  (h4 : quarters = 1) (h5 : dimes = 1) (hnickel_value : nickel_value = 0.05) : 
  ∃ n : ℕ, n = 8 :=
by
  sorry

end NUMINAMATH_GPT_change_in_nickels_l421_42172


namespace NUMINAMATH_GPT_find_x_plus_y_l421_42104

theorem find_x_plus_y (x y : ℝ) (h1 : x + Real.cos y = 2010) (h2 : x + 2010 * Real.sin y = 2009) (hy : 0 ≤ y ∧ y ≤ Real.pi / 2) :
  x + y = 2009 :=
by
  sorry

end NUMINAMATH_GPT_find_x_plus_y_l421_42104


namespace NUMINAMATH_GPT_solve_rectangular_field_problem_l421_42122

-- Define the problem
def f (L W : ℝ) := L * W = 80 ∧ 2 * W + L = 28

-- Define the length of the uncovered side
def length_of_uncovered_side (L: ℝ) := L = 20

-- The statement we need to prove
theorem solve_rectangular_field_problem (L W : ℝ) (h : f L W) : length_of_uncovered_side L :=
by
  sorry

end NUMINAMATH_GPT_solve_rectangular_field_problem_l421_42122


namespace NUMINAMATH_GPT_total_goals_l421_42121

-- Define constants for goals scored in respective seasons
def goalsLastSeason : ℕ := 156
def goalsThisSeason : ℕ := 187

-- Define the theorem for the total number of goals
theorem total_goals : goalsLastSeason + goalsThisSeason = 343 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_total_goals_l421_42121


namespace NUMINAMATH_GPT_cauliflower_difference_is_401_l421_42105

-- Definitions using conditions from part a)
def garden_area_this_year : ℕ := 40401
def side_length_this_year : ℕ := Nat.sqrt garden_area_this_year
def side_length_last_year : ℕ := side_length_this_year - 1
def garden_area_last_year : ℕ := side_length_last_year ^ 2
def cauliflowers_difference : ℕ := garden_area_this_year - garden_area_last_year

-- Problem statement claiming that the difference in cauliflowers produced is 401
theorem cauliflower_difference_is_401 :
  garden_area_this_year = 40401 →
  side_length_this_year = 201 →
  side_length_last_year = 200 →
  garden_area_last_year = 40000 →
  cauliflowers_difference = 401 :=
by
  intros
  sorry

end NUMINAMATH_GPT_cauliflower_difference_is_401_l421_42105


namespace NUMINAMATH_GPT_tip_percentage_calculation_l421_42157

theorem tip_percentage_calculation :
  let a := 8
  let r := 20
  let w := 3
  let n_w := 2
  let d := 6
  let t := 38
  let discount := 0.5
  let full_cost_without_tip := a + r + (w * n_w) + d
  let discounted_meal_cost := a + (r - (r * discount)) + (w * n_w) + d
  let tip_amount := t - discounted_meal_cost
  let tip_percentage := (tip_amount / full_cost_without_tip) * 100
  tip_percentage = 20 :=
by
  sorry

end NUMINAMATH_GPT_tip_percentage_calculation_l421_42157


namespace NUMINAMATH_GPT_greatest_perimeter_of_triangle_l421_42198

theorem greatest_perimeter_of_triangle :
  ∃ (x : ℕ), 
    4 < x ∧ x < 20 / 3 ∧ 
    (x + 4 * x + 20 = 50) :=
by 
  sorry

end NUMINAMATH_GPT_greatest_perimeter_of_triangle_l421_42198


namespace NUMINAMATH_GPT_smallest_positive_integer_solution_l421_42151

theorem smallest_positive_integer_solution (x : ℕ) (h : 5 * x ≡ 17 [MOD 29]) : x = 15 :=
sorry

end NUMINAMATH_GPT_smallest_positive_integer_solution_l421_42151


namespace NUMINAMATH_GPT_weight_of_new_person_l421_42152

-- Define the problem conditions
variables (W : ℝ) -- Weight of the new person
variable (initial_weight : ℝ := 65) -- Weight of the person being replaced
variable (increase_in_avg : ℝ := 4) -- Increase in average weight
variable (num_persons : ℕ := 8) -- Number of persons

-- Define the total increase in weight due to the new person
def total_increase : ℝ := num_persons * increase_in_avg

-- The Lean statement to prove
theorem weight_of_new_person (W : ℝ) (h : total_increase = W - initial_weight) : W = 97 := sorry

end NUMINAMATH_GPT_weight_of_new_person_l421_42152


namespace NUMINAMATH_GPT_discount_percentage_is_ten_l421_42160

-- Definitions based on given conditions
def cost_price : ℝ := 42
def markup (S : ℝ) : ℝ := 0.30 * S
def selling_price (S : ℝ) : Prop := S = cost_price + markup S
def profit : ℝ := 6

-- To prove the discount percentage
theorem discount_percentage_is_ten (S SP : ℝ) 
  (h_sell_price : selling_price S) 
  (h_SP : SP = S - profit) : 
  ((S - SP) / S) * 100 = 10 := 
by
  sorry

end NUMINAMATH_GPT_discount_percentage_is_ten_l421_42160


namespace NUMINAMATH_GPT_calculate_number_of_sides_l421_42155

theorem calculate_number_of_sides (n : ℕ) (h : n ≥ 6) :
  ((6 : ℚ) / n^2) * ((6 : ℚ) / n^2) = 0.027777777777777776 →
  n = 6 :=
by
  sorry

end NUMINAMATH_GPT_calculate_number_of_sides_l421_42155


namespace NUMINAMATH_GPT_area_ratio_of_quadrilateral_ADGJ_to_decagon_l421_42178

noncomputable def ratio_of_areas (k : ℝ) : ℝ :=
  (2 * k^2 * Real.sin (72 * Real.pi / 180)) / (5 * Real.sqrt (5 + 2 * Real.sqrt 5))

theorem area_ratio_of_quadrilateral_ADGJ_to_decagon
  (k : ℝ) :
  ∃ (n m : ℝ), m / n = ratio_of_areas k :=
  sorry

end NUMINAMATH_GPT_area_ratio_of_quadrilateral_ADGJ_to_decagon_l421_42178


namespace NUMINAMATH_GPT_paving_cost_l421_42164

def length : ℝ := 5.5
def width : ℝ := 3.75
def rate : ℝ := 300
def area : ℝ := length * width
def cost : ℝ := area * rate

theorem paving_cost : cost = 6187.50 := by
  -- length = 5.5
  -- width = 3.75
  -- rate = 300
  -- area = length * width = 20.625
  -- cost = area * rate = 6187.50
  sorry

end NUMINAMATH_GPT_paving_cost_l421_42164


namespace NUMINAMATH_GPT_express_x2_y2_z2_in_terms_of_sigma1_sigma2_l421_42124

variable (x y z : ℝ)
def sigma1 := x + y + z
def sigma2 := x * y + y * z + z * x

theorem express_x2_y2_z2_in_terms_of_sigma1_sigma2 :
  x^2 + y^2 + z^2 = sigma1 x y z ^ 2 - 2 * sigma2 x y z := by
  sorry

end NUMINAMATH_GPT_express_x2_y2_z2_in_terms_of_sigma1_sigma2_l421_42124


namespace NUMINAMATH_GPT_arrangements_with_AB_together_l421_42129

theorem arrangements_with_AB_together (students : Finset α) (A B : α) (hA : A ∈ students) (hB : B ∈ students) (h_students : students.card = 5) : 
  ∃ n, n = 48 :=
sorry

end NUMINAMATH_GPT_arrangements_with_AB_together_l421_42129


namespace NUMINAMATH_GPT_integer_points_on_segment_l421_42120

noncomputable def f (n : ℕ) : ℕ :=
  if n % 3 = 0 then 2 else 0

theorem integer_points_on_segment (n : ℕ) (h : 0 < n) :
  f n = if n % 3 = 0 then 2 else 0 :=
by
  sorry

end NUMINAMATH_GPT_integer_points_on_segment_l421_42120


namespace NUMINAMATH_GPT_sqrt_16_l421_42118

theorem sqrt_16 : {x : ℝ | x^2 = 16} = {4, -4} :=
by
  sorry

end NUMINAMATH_GPT_sqrt_16_l421_42118


namespace NUMINAMATH_GPT_determine_better_robber_l421_42161

def sum_of_odd_series (k : ℕ) : ℕ := k * k
def sum_of_even_series (k : ℕ) : ℕ := k * (k + 1)

def first_robber_coins (n k : ℕ) (r : ℕ) : ℕ := 
  if r < 2 * k - 1 then (k - 1) * (k - 1) + r else k * k

def second_robber_coins (n k : ℕ) (r : ℕ) : ℕ := 
  if r < 2 * k - 1 then k * (k + 1) else k * k - k + r

theorem determine_better_robber (n k r : ℕ) :
  if 2 * k * k - 2 * k < n ∧ n < 2 * k * k then
    first_robber_coins n k r > second_robber_coins n k r
  else if 2 * k * k < n ∧ n < 2 * k * k + 2 * k then
    second_robber_coins n k r > first_robber_coins n k r
  else 
    false :=
sorry

end NUMINAMATH_GPT_determine_better_robber_l421_42161


namespace NUMINAMATH_GPT_percentage_increase_salary_l421_42101

theorem percentage_increase_salary (S : ℝ) (P : ℝ) (h1 : 1.16 * S = 348) (h2 : S + P * S = 375) : P = 0.25 :=
by
  sorry

end NUMINAMATH_GPT_percentage_increase_salary_l421_42101


namespace NUMINAMATH_GPT_solve_toenail_problem_l421_42176

def toenail_problem (b_toenails r_toenails_already r_toenails_more : ℕ) : Prop :=
  (b_toenails = 20) ∧
  (r_toenails_already = 40) ∧
  (r_toenails_more = 20) →
  (r_toenails_already + r_toenails_more = 60)

theorem solve_toenail_problem : toenail_problem 20 40 20 :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_toenail_problem_l421_42176


namespace NUMINAMATH_GPT_temperature_decrease_2C_l421_42107

variable (increase_3 : ℤ := 3)
variable (decrease_2 : ℤ := -2)

theorem temperature_decrease_2C :
  decrease_2 = -2 :=
by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_temperature_decrease_2C_l421_42107


namespace NUMINAMATH_GPT_binom_coeff_mult_l421_42192

theorem binom_coeff_mult :
  (Nat.choose 10 3) * (Nat.choose 8 3) = 6720 :=
by
  sorry

end NUMINAMATH_GPT_binom_coeff_mult_l421_42192


namespace NUMINAMATH_GPT_weight_of_second_piece_of_wood_l421_42165

/--
Given: 
1) The density and thickness of the wood are uniform.
2) The first piece of wood is a square with a side length of 3 inches and a weight of 15 ounces.
3) The second piece of wood is a square with a side length of 6 inches.
Theorem: 
The weight of the second piece of wood is 60 ounces.
-/
theorem weight_of_second_piece_of_wood (s1 s2 w1 w2 : ℕ) (h1 : s1 = 3) (h2 : w1 = 15) (h3 : s2 = 6) :
  w2 = 60 :=
sorry

end NUMINAMATH_GPT_weight_of_second_piece_of_wood_l421_42165


namespace NUMINAMATH_GPT_find_a_l421_42102

theorem find_a (a : ℝ) (h1 : a^2 + 2 * a - 15 = 0) (h2 : a^2 + 4 * a - 5 ≠ 0) :
  a = 3 :=
by
sorry

end NUMINAMATH_GPT_find_a_l421_42102


namespace NUMINAMATH_GPT_terminal_side_in_third_quadrant_l421_42141

theorem terminal_side_in_third_quadrant (α : ℝ) (h1 : Real.sin α < 0) (h2 : Real.tan α > 0) : 
  (∃ k : ℤ, α = k * π + π / 2 + π) := sorry

end NUMINAMATH_GPT_terminal_side_in_third_quadrant_l421_42141


namespace NUMINAMATH_GPT_jackson_grade_l421_42117

open Function

theorem jackson_grade :
  ∃ (grade : ℕ), 
  ∀ (hours_playing hours_studying : ℕ), 
    (hours_playing = 9) ∧ 
    (hours_studying = hours_playing / 3) ∧ 
    (grade = hours_studying * 15) →
    grade = 45 := 
by {
  sorry
}

end NUMINAMATH_GPT_jackson_grade_l421_42117


namespace NUMINAMATH_GPT_audio_per_cd_l421_42156

theorem audio_per_cd (total_audio : ℕ) (max_per_cd : ℕ) (num_cds : ℕ) 
  (h1 : total_audio = 360) 
  (h2 : max_per_cd = 60) 
  (h3 : num_cds = total_audio / max_per_cd): 
  (total_audio / num_cds = max_per_cd) :=
by
  sorry

end NUMINAMATH_GPT_audio_per_cd_l421_42156


namespace NUMINAMATH_GPT_evaluate_expression_l421_42115

theorem evaluate_expression :
  let x := (1/4 : ℚ)
  let y := (4/5 : ℚ)
  let z := (-2 : ℚ)
  x^3 * y^2 * z^2 = 1/25 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l421_42115


namespace NUMINAMATH_GPT_alex_age_thrice_ben_in_n_years_l421_42162

-- Definitions based on the problem's conditions
def Ben_current_age := 4
def Alex_current_age := Ben_current_age + 30

-- The main problem defined as a theorem to be proven
theorem alex_age_thrice_ben_in_n_years :
  ∃ n : ℕ, Alex_current_age + n = 3 * (Ben_current_age + n) ∧ n = 11 :=
by
  sorry

end NUMINAMATH_GPT_alex_age_thrice_ben_in_n_years_l421_42162


namespace NUMINAMATH_GPT_boys_left_is_31_l421_42146

def initial_children : ℕ := 85
def girls_came_in : ℕ := 24
def final_children : ℕ := 78

noncomputable def compute_boys_left (initial : ℕ) (girls_in : ℕ) (final : ℕ) : ℕ :=
  (initial + girls_in) - final

theorem boys_left_is_31 :
  compute_boys_left initial_children girls_came_in final_children = 31 :=
by
  sorry

end NUMINAMATH_GPT_boys_left_is_31_l421_42146


namespace NUMINAMATH_GPT_intersection_points_form_line_l421_42100

theorem intersection_points_form_line :
  ∀ (x y : ℝ), ((x * y = 12) ∧ ((x^2 / 16) + (y^2 / 36) = 1)) →
  ∃ (x1 x2 : ℝ) (y1 y2 : ℝ), (x, y) = (x1, y1) ∨ (x, y) = (x2, y2) ∧ (x2 - x1) * (y2 - y1) = x1 * y1 - x2 * y2 :=
by
  sorry

end NUMINAMATH_GPT_intersection_points_form_line_l421_42100


namespace NUMINAMATH_GPT_men_at_conference_l421_42140

theorem men_at_conference (M : ℕ) 
  (num_women : ℕ) (num_children : ℕ)
  (indian_men_fraction : ℚ) (indian_women_fraction : ℚ)
  (indian_children_fraction : ℚ) (non_indian_fraction : ℚ)
  (num_women_eq : num_women = 300)
  (num_children_eq : num_children = 500)
  (indian_men_fraction_eq : indian_men_fraction = 0.10)
  (indian_women_fraction_eq : indian_women_fraction = 0.60)
  (indian_children_fraction_eq : indian_children_fraction = 0.70)
  (non_indian_fraction_eq : non_indian_fraction = 0.5538461538461539) :
  M = 500 :=
by
  sorry

end NUMINAMATH_GPT_men_at_conference_l421_42140


namespace NUMINAMATH_GPT_Nicky_profit_l421_42133

-- Definitions for conditions
def card1_value : ℕ := 8
def card2_value : ℕ := 8
def received_card_value : ℕ := 21

-- The statement to be proven
theorem Nicky_profit : (received_card_value - (card1_value + card2_value)) = 5 :=
by
  sorry

end NUMINAMATH_GPT_Nicky_profit_l421_42133


namespace NUMINAMATH_GPT_who_stole_the_pan_l421_42174

def Frog_statement := "Lackey-Lech stole the pan"
def LackeyLech_statement := "I did not steal any pan"
def KnaveOfHearts_statement := "I stole the pan"

axiom no_more_than_one_liar : ∀ (frog_is_lying : Prop) (lackey_lech_is_lying : Prop) (knave_of_hearts_is_lying : Prop), (frog_is_lying → ¬ lackey_lech_is_lying) ∧ (frog_is_lying → ¬ knave_of_hearts_is_lying) ∧ (lackey_lech_is_lying → ¬ knave_of_hearts_is_lying)

theorem who_stole_the_pan : KnaveOfHearts_statement = "I stole the pan" :=
sorry

end NUMINAMATH_GPT_who_stole_the_pan_l421_42174


namespace NUMINAMATH_GPT_units_digit_k_squared_plus_three_to_the_k_mod_10_l421_42109

def k := 2025^2 + 3^2025

theorem units_digit_k_squared_plus_three_to_the_k_mod_10 : 
  (k^2 + 3^k) % 10 = 5 := by
sorry

end NUMINAMATH_GPT_units_digit_k_squared_plus_three_to_the_k_mod_10_l421_42109


namespace NUMINAMATH_GPT_rectangle_perimeter_l421_42196

theorem rectangle_perimeter (long_side short_side : ℝ) 
  (h_long : long_side = 1) 
  (h_short : short_side = long_side - 2/8) : 
  2 * long_side + 2 * short_side = 3.5 := 
by 
  sorry

end NUMINAMATH_GPT_rectangle_perimeter_l421_42196


namespace NUMINAMATH_GPT_total_distance_travelled_l421_42143

/-- Proving that the total horizontal distance traveled by the centers of two wheels with radii 1 m and 2 m 
    after one complete revolution is 6π meters. -/
theorem total_distance_travelled (R1 R2 : ℝ) (h1 : R1 = 1) (h2 : R2 = 2) : 
    2 * Real.pi * R1 + 2 * Real.pi * R2 = 6 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_total_distance_travelled_l421_42143


namespace NUMINAMATH_GPT_calculate_expression_l421_42197

theorem calculate_expression : 1010^2 - 990^2 - 1005^2 + 995^2 = 20000 := by
  sorry

end NUMINAMATH_GPT_calculate_expression_l421_42197


namespace NUMINAMATH_GPT_karting_routes_10_min_l421_42199

-- Define the recursive function for M_{n, A}
def num_routes : ℕ → ℕ
| 0 => 1   -- Starting point at A for 0 minutes (0 routes)
| 1 => 0   -- Impossible to end at A in just 1 move
| 2 => 1   -- Only one way to go A -> B -> A in 2 minutes
| n + 1 =>
  if n = 1 then 0 -- Additional base case for n=2 as defined
  else if n = 2 then 1
  else num_routes (n - 1) + num_routes (n - 2)

theorem karting_routes_10_min : num_routes 10 = 34 := by
  -- Proof steps go here
  sorry

end NUMINAMATH_GPT_karting_routes_10_min_l421_42199


namespace NUMINAMATH_GPT_cost_per_package_l421_42148

theorem cost_per_package
  (parents : ℕ)
  (brothers : ℕ)
  (spouses_per_brother : ℕ)
  (children_per_brother : ℕ)
  (total_cost : ℕ)
  (num_packages : ℕ)
  (h1 : parents = 2)
  (h2 : brothers = 3)
  (h3 : spouses_per_brother = 1)
  (h4 : children_per_brother = 2)
  (h5 : total_cost = 70)
  (h6 : num_packages = parents + brothers + brothers * spouses_per_brother + brothers * children_per_brother) :
  total_cost / num_packages = 5 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_cost_per_package_l421_42148


namespace NUMINAMATH_GPT_cylinder_properties_l421_42158

theorem cylinder_properties (h r : ℝ) (h_eq : h = 15) (r_eq : r = 5) :
  let total_surface_area := 2 * Real.pi * r^2 + 2 * Real.pi * r * h
  let volume := Real.pi * r^2 * h
  total_surface_area = 200 * Real.pi ∧ volume = 375 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_cylinder_properties_l421_42158


namespace NUMINAMATH_GPT_total_customers_l421_42184

-- Define the initial number of customers
def initial_customers : ℕ := 14

-- Define the number of customers that left
def customers_left : ℕ := 3

-- Define the number of new customers gained
def new_customers : ℕ := 39

-- Prove that the total number of customers is 50
theorem total_customers : initial_customers - customers_left + new_customers = 50 := 
by
  sorry

end NUMINAMATH_GPT_total_customers_l421_42184


namespace NUMINAMATH_GPT_sequence_term_4_l421_42149

noncomputable def sequence (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => 2 * sequence n

theorem sequence_term_4 : sequence 3 = 8 := 
by
  sorry

end NUMINAMATH_GPT_sequence_term_4_l421_42149
