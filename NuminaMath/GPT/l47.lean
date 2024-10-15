import Mathlib

namespace NUMINAMATH_GPT_tensor_op_correct_l47_4761

-- Define the operation ⊗
def tensor_op (x y : ℝ) : ℝ := x^2 + y

-- Goal: Prove h ⊗ (h ⊗ h) = 2h^2 + h for some h in ℝ
theorem tensor_op_correct (h : ℝ) : tensor_op h (tensor_op h h) = 2 * h^2 + h :=
by
  sorry

end NUMINAMATH_GPT_tensor_op_correct_l47_4761


namespace NUMINAMATH_GPT_fraction_equation_solution_l47_4745

theorem fraction_equation_solution (x : ℝ) (h₁ : x ≠ 3) (h₂ : x ≠ 4) :
  (3 / (x - 3) = 4 / (x - 4)) → x = 0 :=
by
  sorry

end NUMINAMATH_GPT_fraction_equation_solution_l47_4745


namespace NUMINAMATH_GPT_tangents_secant_intersect_l47_4785

variable {A B C O1 P Q R : Type} 
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace P] [MetricSpace Q] [MetricSpace R]
variables (AB AC : Set (MetricSpace A)) (t : Tangent AB) (s : Tangent AC)

variable (BC : line ( Set A))
variable (APQ : secant A P Q) 

theorem tangents_secant_intersect { AR AP AQ : ℝ } :
  2 / AR = 1 / AP + 1 / AQ :=
by
  sorry

end NUMINAMATH_GPT_tangents_secant_intersect_l47_4785


namespace NUMINAMATH_GPT_fraction_meaningful_l47_4721

theorem fraction_meaningful (x : ℝ) : (x ≠ 5) ↔ (x-5 ≠ 0) :=
by simp [sub_eq_zero]

end NUMINAMATH_GPT_fraction_meaningful_l47_4721


namespace NUMINAMATH_GPT_niki_money_l47_4703

variables (N A : ℕ)

def condition1 (N A : ℕ) : Prop := N = 2 * A + 15
def condition2 (N A : ℕ) : Prop := N - 30 = (A + 30) / 2

theorem niki_money : condition1 N A ∧ condition2 N A → N = 55 :=
by
  sorry

end NUMINAMATH_GPT_niki_money_l47_4703


namespace NUMINAMATH_GPT_initial_number_of_professors_l47_4784

theorem initial_number_of_professors (p : ℕ) :
  (∃ p, (6480 / p : ℚ) < (11200 / (p + 3) : ℚ) ∧ 
   6480 % p = 0 ∧ 11200 % (p + 3) = 0 ∧ p > 4) → 
  p = 5 := 
sorry

end NUMINAMATH_GPT_initial_number_of_professors_l47_4784


namespace NUMINAMATH_GPT_volume_of_rectangular_prism_l47_4711

theorem volume_of_rectangular_prism (x y z : ℝ) 
  (h1 : x * y = 30) 
  (h2 : x * z = 45) 
  (h3 : y * z = 75) : 
  x * y * z = 150 :=
sorry

end NUMINAMATH_GPT_volume_of_rectangular_prism_l47_4711


namespace NUMINAMATH_GPT_animals_not_like_either_l47_4710

def total_animals : ℕ := 75
def animals_eat_carrots : ℕ := 26
def animals_like_hay : ℕ := 56
def animals_like_both : ℕ := 14

theorem animals_not_like_either : (total_animals - (animals_eat_carrots - animals_like_both + animals_like_hay - animals_like_both + animals_like_both)) = 7 := by
  sorry

end NUMINAMATH_GPT_animals_not_like_either_l47_4710


namespace NUMINAMATH_GPT_amount_after_two_years_l47_4742

def amount_after_years (P : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  P * ((r + 1) ^ n) / (r ^ n)

theorem amount_after_two_years :
  let P : ℕ := 70400
  let r : ℕ := 8
  amount_after_years P r 2 = 89070 :=
  by
    sorry

end NUMINAMATH_GPT_amount_after_two_years_l47_4742


namespace NUMINAMATH_GPT_problem1_problem2a_problem2b_l47_4795

noncomputable def x : ℝ := Real.sqrt 6 - Real.sqrt 2
noncomputable def a : ℝ := Real.sqrt 3 + Real.sqrt 2
noncomputable def b : ℝ := Real.sqrt 3 - Real.sqrt 2

theorem problem1 : x * (Real.sqrt 6 - x) + (x + Real.sqrt 5) * (x - Real.sqrt 5) = 1 - 2 * Real.sqrt 3 := 
by
  sorry

theorem problem2a : a - b = 2 * Real.sqrt 2 := 
by 
  sorry

theorem problem2b : a^2 - 2 * a * b + b^2 = 8 := 
by 
  sorry

end NUMINAMATH_GPT_problem1_problem2a_problem2b_l47_4795


namespace NUMINAMATH_GPT_original_money_l47_4744
noncomputable def original_amount (x : ℝ) :=
  let after_first_loss := (2/3) * x
  let after_first_win := after_first_loss + 10
  let after_second_loss := after_first_win - (1/3) * after_first_win
  let after_second_win := after_second_loss + 20
  after_second_win

theorem original_money (x : ℝ) (h : original_amount x = x) : x = 48 :=
by {
  sorry
}

end NUMINAMATH_GPT_original_money_l47_4744


namespace NUMINAMATH_GPT_num_true_propositions_l47_4740

theorem num_true_propositions (x : ℝ) :
  (∀ x, x > -3 → x > -6) ∧
  (∀ x, x > -6 → x > -3 = false) ∧
  (∀ x, x ≤ -3 → x ≤ -6 = false) ∧
  (∀ x, x ≤ -6 → x ≤ -3) →
  2 = 2 :=
by
  sorry

end NUMINAMATH_GPT_num_true_propositions_l47_4740


namespace NUMINAMATH_GPT_problem_statement_l47_4754

open Complex

noncomputable def a : ℂ := 5 - 3 * I
noncomputable def b : ℂ := 2 + 4 * I

theorem problem_statement : 3 * a - 4 * b = 7 - 25 * I :=
by { sorry }

end NUMINAMATH_GPT_problem_statement_l47_4754


namespace NUMINAMATH_GPT_wage_ratio_l47_4753

-- Define the conditions
variable (M W : ℝ) -- M stands for man's daily wage, W stands for woman's daily wage
variable (h1 : 40 * 10 * M = 14400) -- Condition 1: 40 men working for 10 days earn Rs. 14400
variable (h2 : 40 * 30 * W = 21600) -- Condition 2: 40 women working for 30 days earn Rs. 21600

-- The statement to prove
theorem wage_ratio (h1 : 40 * 10 * M = 14400) (h2 : 40 * 30 * W = 21600) : M / W = 2 := by
  sorry

end NUMINAMATH_GPT_wage_ratio_l47_4753


namespace NUMINAMATH_GPT_count_valid_a_values_l47_4794

def is_perfect_square (n : ℕ) : Prop :=
∃ k : ℕ, k * k = n

def valid_a_values (a : ℕ) : Prop :=
1 ≤ a ∧ a ≤ 100 ∧ is_perfect_square (16 * a + 9)

theorem count_valid_a_values :
  ∃ N : ℕ, N = Nat.card {a : ℕ | valid_a_values a} := sorry

end NUMINAMATH_GPT_count_valid_a_values_l47_4794


namespace NUMINAMATH_GPT_determine_b_l47_4780

theorem determine_b (a b c y1 y2 : ℝ) 
  (h1 : y1 = a * 2^2 + b * 2 + c)
  (h2 : y2 = a * (-2)^2 + b * (-2) + c)
  (h3 : y1 - y2 = -12) : 
  b = -3 := 
by
  sorry

end NUMINAMATH_GPT_determine_b_l47_4780


namespace NUMINAMATH_GPT_cosine_greater_sine_cosine_cos_greater_sine_sin_l47_4782

variable {f g : ℝ → ℝ}

-- Problem 1
theorem cosine_greater_sine (h : ∀ x, - (Real.pi / 2) < f x + g x ∧ f x + g x < Real.pi / 2
                            ∧ - (Real.pi / 2) < f x - g x ∧ f x - g x < Real.pi / 2) :
  ∀ x, Real.cos (f x) > Real.sin (g x) :=
sorry

-- Problem 2
theorem cosine_cos_greater_sine_sin (x : ℝ) :  Real.cos (Real.cos x) > Real.sin (Real.sin x) :=
sorry

end NUMINAMATH_GPT_cosine_greater_sine_cosine_cos_greater_sine_sin_l47_4782


namespace NUMINAMATH_GPT_museum_revenue_l47_4726

theorem museum_revenue (V : ℕ) (H : V = 500)
  (R : ℕ) (H_R : R = 60 * V / 100)
  (C_p : ℕ) (H_C_p : C_p = 40 * R / 100)
  (S_p : ℕ) (H_S_p : S_p = 30 * R / 100)
  (A_p : ℕ) (H_A_p : A_p = 30 * R / 100)
  (C_t S_t A_t : ℕ) (H_C_t : C_t = 4) (H_S_t : S_t = 6) (H_A_t : A_t = 12) :
  C_p * C_t + S_p * S_t + A_p * A_t = 2100 :=
by 
  sorry

end NUMINAMATH_GPT_museum_revenue_l47_4726


namespace NUMINAMATH_GPT_division_addition_correct_l47_4729

theorem division_addition_correct : 0.2 / 0.005 + 0.1 = 40.1 :=
by
  sorry

end NUMINAMATH_GPT_division_addition_correct_l47_4729


namespace NUMINAMATH_GPT_oliver_shelves_needed_l47_4793

-- Definitions based on conditions
def total_books : ℕ := 46
def books_taken_by_librarian : ℕ := 10
def books_remaining (total books_taken : ℕ) : ℕ := total - books_taken
def books_per_shelf : ℕ := 4

-- Theorem statement
theorem oliver_shelves_needed :
  books_remaining total_books books_taken_by_librarian / books_per_shelf = 9 := by
  sorry

end NUMINAMATH_GPT_oliver_shelves_needed_l47_4793


namespace NUMINAMATH_GPT_find_smaller_number_l47_4779

-- Define the conditions as hypotheses and the goal as a proposition
theorem find_smaller_number (x y : ℕ) (h1 : x + y = 77) (h2 : x = 42 ∨ y = 42) (h3 : 5 * x = 6 * y) : x = 35 :=
sorry

end NUMINAMATH_GPT_find_smaller_number_l47_4779


namespace NUMINAMATH_GPT_negation_of_proposition_l47_4799

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, x^2 - 2 * x + 4 ≤ 0) ↔ (∃ x : ℝ, x^2 - 2 * x + 4 > 0) :=
by sorry

end NUMINAMATH_GPT_negation_of_proposition_l47_4799


namespace NUMINAMATH_GPT_sum_of_ten_distinct_numbers_lt_75_l47_4798

theorem sum_of_ten_distinct_numbers_lt_75 :
  ∃ (S : Finset ℕ), S.card = 10 ∧
  (∃ (S_div_5 : Finset ℕ), S_div_5 ⊆ S ∧ S_div_5.card = 3 ∧ ∀ x ∈ S_div_5, 5 ∣ x) ∧
  (∃ (S_div_4 : Finset ℕ), S_div_4 ⊆ S ∧ S_div_4.card = 4 ∧ ∀ x ∈ S_div_4, 4 ∣ x) ∧
  S.sum id < 75 :=
by { 
  sorry 
}

end NUMINAMATH_GPT_sum_of_ten_distinct_numbers_lt_75_l47_4798


namespace NUMINAMATH_GPT_yearly_water_consumption_correct_l47_4737

def monthly_water_consumption : ℝ := 182.88
def months_in_a_year : ℕ := 12
def yearly_water_consumption : ℝ := monthly_water_consumption * (months_in_a_year : ℝ)

theorem yearly_water_consumption_correct :
  yearly_water_consumption = 2194.56 :=
by
  sorry

end NUMINAMATH_GPT_yearly_water_consumption_correct_l47_4737


namespace NUMINAMATH_GPT_min_value_of_sum_l47_4787

theorem min_value_of_sum (x y : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : x * y + 2 * x + y = 4) : x + y ≥ 2 * Real.sqrt 6 - 3 :=
sorry

end NUMINAMATH_GPT_min_value_of_sum_l47_4787


namespace NUMINAMATH_GPT_no_such_sequence_exists_l47_4788

theorem no_such_sequence_exists (a : ℕ → ℝ) :
  (∀ i, 1 ≤ i ∧ i ≤ 13 → a i + a (i + 1) + a (i + 2) > 0) →
  (∀ i, 1 ≤ i ∧ i ≤ 12 → a i + a (i + 1) + a (i + 2) + a (i + 3) < 0) →
  False :=
by
  sorry

end NUMINAMATH_GPT_no_such_sequence_exists_l47_4788


namespace NUMINAMATH_GPT_whitewashing_cost_l47_4718

noncomputable def cost_of_whitewashing (l w h : ℝ) (c : ℝ) (door_area window_area : ℝ) (num_windows : ℝ) : ℝ :=
  let perimeter := 2 * (l + w)
  let total_wall_area := perimeter * h
  let total_window_area := num_windows * window_area
  let total_paintable_area := total_wall_area - (door_area + total_window_area)
  total_paintable_area * c

theorem whitewashing_cost:
  cost_of_whitewashing 25 15 12 6 (6 * 3) (4 * 3) 3 = 5436 := by
  sorry

end NUMINAMATH_GPT_whitewashing_cost_l47_4718


namespace NUMINAMATH_GPT_vacant_student_seats_given_to_parents_l47_4700

-- Definitions of the conditions
def total_seats : Nat := 150

def awardees_seats : Nat := 15
def admins_teachers_seats : Nat := 45
def students_seats : Nat := 60
def parents_seats : Nat := 30

def awardees_occupied_seats : Nat := 15
def admins_teachers_occupied_seats : Nat := 9 * admins_teachers_seats / 10
def students_occupied_seats : Nat := 4 * students_seats / 5
def parents_occupied_seats : Nat := 7 * parents_seats / 10

-- Vacant seats calculation
def awardees_vacant_seats : Nat := awardees_seats - awardees_occupied_seats
def admins_teachers_vacant_seats : Nat := admins_teachers_seats - admins_teachers_occupied_seats
def students_vacant_seats : Nat := students_seats - students_occupied_seats
def parents_vacant_seats : Nat := parents_seats - parents_occupied_seats

-- Theorem statement
theorem vacant_student_seats_given_to_parents :
  students_vacant_seats = 12 →
  parents_vacant_seats = 9 →
  9 ≤ students_vacant_seats ∧ 9 ≤ parents_vacant_seats :=
by
  sorry

end NUMINAMATH_GPT_vacant_student_seats_given_to_parents_l47_4700


namespace NUMINAMATH_GPT_total_students_in_class_l47_4790

theorem total_students_in_class (R S : ℕ) (h1 : 2 + 12 + 14 + R = S) (h2 : 2 * S = 40 + 3 * R) : S = 44 :=
by
  sorry

end NUMINAMATH_GPT_total_students_in_class_l47_4790


namespace NUMINAMATH_GPT_rachel_wrote_six_pages_l47_4738

theorem rachel_wrote_six_pages
  (write_rate : ℕ)
  (research_time : ℕ)
  (editing_time : ℕ)
  (total_time : ℕ)
  (total_time_in_minutes : ℕ := total_time * 60)
  (actual_time_writing : ℕ := total_time_in_minutes - (research_time + editing_time))
  (pages_written : ℕ := actual_time_writing / write_rate) :
  write_rate = 30 →
  research_time = 45 →
  editing_time = 75 →
  total_time = 5 →
  pages_written = 6 :=
by
  intros h1 h2 h3 h4
  subst h1
  subst h2
  subst h3
  subst h4
  have h5 : total_time_in_minutes = 300 := by sorry
  have h6 : actual_time_writing = 180 := by sorry
  have h7 : pages_written = 6 := by sorry
  exact h7

end NUMINAMATH_GPT_rachel_wrote_six_pages_l47_4738


namespace NUMINAMATH_GPT_sum_of_midpoint_coordinates_l47_4752

theorem sum_of_midpoint_coordinates :
  let (x1, y1) := (8, 16)
  let (x2, y2) := (2, -8)
  let midpoint_x := (x1 + x2) / 2
  let midpoint_y := (y1 + y2) / 2
  midpoint_x + midpoint_y = 9 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_midpoint_coordinates_l47_4752


namespace NUMINAMATH_GPT_sum_of_digits_1_to_1000_l47_4776

/--  sum_of_digits calculates the sum of digits of a given number n -/
def sum_of_digits(n : ℕ) : ℕ :=
  n.digits 10 |>.sum

/-- sum_of_digits_in_range calculates the sum of the digits 
of all numbers in the inclusive range from 1 to m -/
def sum_of_digits_in_range (m : ℕ) : ℕ :=
  (Finset.range (m + 1)).sum sum_of_digits

theorem sum_of_digits_1_to_1000 : sum_of_digits_in_range 1000 = 13501 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_digits_1_to_1000_l47_4776


namespace NUMINAMATH_GPT_min_value_of_a_plus_2b_l47_4747

theorem min_value_of_a_plus_2b (a b : ℝ) (h1: a > 0) (h2: b > 0) (h3: 1 / (a + 1) + 1 / (b + 1) = 1) : 
  a + 2 * b ≥ 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_a_plus_2b_l47_4747


namespace NUMINAMATH_GPT_clusters_of_oats_l47_4781

-- Define conditions:
def clusters_per_spoonful : Nat := 4
def spoonfuls_per_bowl : Nat := 25
def bowls_per_box : Nat := 5

-- Define the question and correct answer:
def clusters_per_box : Nat :=
  clusters_per_spoonful * spoonfuls_per_bowl * bowls_per_box

-- Theorem statement for the proof problem:
theorem clusters_of_oats:
  clusters_per_box = 500 :=
by
  sorry

end NUMINAMATH_GPT_clusters_of_oats_l47_4781


namespace NUMINAMATH_GPT_probability_of_condition_l47_4706

def Q_within_square (x y : ℝ) : Prop :=
  0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1

def condition (x y : ℝ) : Prop :=
  y > (1/2) * x

theorem probability_of_condition : 
  ∀ x y, Q_within_square x y → (0.75 = 3 / 4) :=
by
  sorry

end NUMINAMATH_GPT_probability_of_condition_l47_4706


namespace NUMINAMATH_GPT_rectangle_area_diagonal_l47_4756

theorem rectangle_area_diagonal (r l w d : ℝ) (h_ratio : r = 5 / 2) (h_diag : d^2 = l^2 + w^2) : ∃ k : ℝ, (k = 10 / 29) ∧ (l / w = r) ∧ (l^2 + w^2 = d^2) :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_diagonal_l47_4756


namespace NUMINAMATH_GPT_polynomial_factorization_example_l47_4748

open Polynomial

theorem polynomial_factorization_example
  (a_5 a_4 a_3 a_2 a_1 a_0 : ℤ) (hf : ∀ i ∈ [a_5, a_4, a_3, a_2, a_1, a_0], |i| ≤ 4)
  (b_3 b_2 b_1 b_0 : ℤ) (hg : ∀ i ∈ [b_3, b_2, b_1, b_0], |i| ≤ 1)
  (c_2 c_1 c_0 : ℤ) (hh : ∀ i ∈ [c_2, c_1, c_0], |i| ≤ 1)
  (h : (C a_5 * X^5 + C a_4 * X^4 + C a_3 * X^3 + C a_2 * X^2 + C a_1 * X + C a_0).eval 10 =
       ((C b_3 * X^3 + C b_2 * X^2 + C b_1 * X + C b_0) * (C c_2 * X^2 + C c_1 * X + C c_0)).eval 10) :
  (C a_5 * X^5 + C a_4 * X^4 + C a_3 * X^3 + C a_2 * X^2 + C a_1 * X + C a_0) =
  (C b_3 * X^3 + C b_2 * X^2 + C b_1 * X + C b_0) * (C c_2 * X^2 + C c_1 * X + C c_0) :=
sorry

end NUMINAMATH_GPT_polynomial_factorization_example_l47_4748


namespace NUMINAMATH_GPT_number_of_intersection_points_l47_4751

-- Define the standard parabola y = x^2
def parabola (x : ℝ) : ℝ := x^2

-- Define what it means for a line to be tangent to the parabola
def is_tangent (m : ℝ) (c : ℝ) : Prop :=
  ∃ x0 : ℝ, parabola x0 = m * x0 + c ∧ 2 * x0 = m

-- Define what it means for a line to intersect the parabola
def line_intersects_parabola (m : ℝ) (c : ℝ) : Prop :=
  ∃ x1 x2 : ℝ, parabola x1 = m * x1 + c ∧ parabola x2 = m * x2 + c

-- Main theorem statement
theorem number_of_intersection_points :
  (∃ m c : ℝ, is_tangent m c) → (∃ m' c' : ℝ, line_intersects_parabola m' c') →
  ∃ n : ℕ, n = 1 ∨ n = 2 ∨ n = 3 :=
sorry

end NUMINAMATH_GPT_number_of_intersection_points_l47_4751


namespace NUMINAMATH_GPT_problem1_problem2_problem3_l47_4786

theorem problem1 : -3^2 + (-1/2)^2 + (2023 - Real.pi)^0 - |-2| = -47/4 :=
by
  sorry

theorem problem2 (a : ℝ) : (-2 * a^2)^3 * a^2 + a^8 = -7 * a^8 :=
by
  sorry

theorem problem3 : 2023^2 - 2024 * 2022 = 1 :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_l47_4786


namespace NUMINAMATH_GPT_find_missing_number_l47_4765

theorem find_missing_number :
  ∀ (x y : ℝ),
    (12 + x + 42 + 78 + 104) / 5 = 62 →
    (128 + y + 511 + 1023 + x) / 5 = 398.2 →
    y = 255 :=
by
  intros x y h1 h2
  sorry

end NUMINAMATH_GPT_find_missing_number_l47_4765


namespace NUMINAMATH_GPT_largest_possible_b_l47_4727

theorem largest_possible_b (a b c : ℕ) (h1 : 1 < c) (h2 : c < b) (h3 : b < a) (h4 : a * b * c = 360) : b ≤ 12 :=
by
  sorry

end NUMINAMATH_GPT_largest_possible_b_l47_4727


namespace NUMINAMATH_GPT_lily_pads_half_lake_l47_4735

noncomputable def size (n : ℕ) : ℝ := sorry

theorem lily_pads_half_lake {n : ℕ} (h : size 48 = size 0 * 2^48) : size 47 = (size 48) / 2 :=
by 
  sorry

end NUMINAMATH_GPT_lily_pads_half_lake_l47_4735


namespace NUMINAMATH_GPT_coral_must_read_pages_to_finish_book_l47_4701

theorem coral_must_read_pages_to_finish_book
  (total_pages first_week_read second_week_percentage pages_remaining first_week_left second_week_read : ℕ)
  (initial_pages_read : ℕ := total_pages / 2)
  (remaining_after_first_week : ℕ := total_pages - initial_pages_read)
  (read_second_week : ℕ := remaining_after_first_week * second_week_percentage / 100)
  (remaining_after_second_week : ℕ := remaining_after_first_week - read_second_week)
  (final_pages_to_read : ℕ := remaining_after_second_week):
  total_pages = 600 → first_week_read = 300 → second_week_percentage = 30 →
  pages_remaining = 300 → first_week_left = 300 → second_week_read = 90 →
  remaining_after_first_week = 300 - 300 →
  remaining_after_second_week = remaining_after_first_week - second_week_read →
  third_week_read = remaining_after_second_week →
  third_week_read = 210 := by
  sorry

end NUMINAMATH_GPT_coral_must_read_pages_to_finish_book_l47_4701


namespace NUMINAMATH_GPT_runner_speed_ratio_l47_4766

noncomputable def speed_ratio (u1 u2 : ℝ) : ℝ := u1 / u2

theorem runner_speed_ratio (u1 u2 : ℝ) (h1 : u1 > u2) (h2 : u1 + u2 = 5) (h3 : u1 - u2 = 5/3) :
  speed_ratio u1 u2 = 2 :=
by
  sorry

end NUMINAMATH_GPT_runner_speed_ratio_l47_4766


namespace NUMINAMATH_GPT_relationship_f_minus_a2_f_minus_1_l47_4702

noncomputable def f (x : ℝ) : ℝ := x^3 - x^2 - x

-- Theorem statement translation
theorem relationship_f_minus_a2_f_minus_1 (a : ℝ) : f (-a^2) ≤ f (-1) := 
sorry

end NUMINAMATH_GPT_relationship_f_minus_a2_f_minus_1_l47_4702


namespace NUMINAMATH_GPT_probability_of_3_correct_answers_is_31_over_135_expected_value_of_total_score_is_50_l47_4791

noncomputable def probability_correct_answers : ℚ :=
  let pA := (1/5 : ℚ)
  let pB := (3/5 : ℚ)
  let pC := (1/5 : ℚ)
  ((pA * (3/9 : ℚ) * (2/3)^2 * (1/3)) + (pB * (6/9 : ℚ) * (2/3) * (1/3)^2) + (pC * (1/9 : ℚ) * (1/3)^3))

theorem probability_of_3_correct_answers_is_31_over_135 :
  probability_correct_answers = 31 / 135 := by
  sorry

noncomputable def expected_score : ℚ :=
  let E_m := (1/5 * 1 + 3/5 * 2 + 1/5 * 3 : ℚ)
  let E_n := (3 * (2/3 : ℚ))
  (15 * E_m + 10 * E_n)

theorem expected_value_of_total_score_is_50 :
  expected_score = 50 := by
  sorry

end NUMINAMATH_GPT_probability_of_3_correct_answers_is_31_over_135_expected_value_of_total_score_is_50_l47_4791


namespace NUMINAMATH_GPT_ratio_of_carpets_l47_4741

theorem ratio_of_carpets (h1 h2 h3 h4 : ℕ) (total : ℕ) 
  (H1 : h1 = 12) (H2 : h2 = 20) (H3 : h3 = 10) (H_total : total = 62) 
  (H_all_houses : h1 + h2 + h3 + h4 = total) : h4 / h3 = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_carpets_l47_4741


namespace NUMINAMATH_GPT_rowing_students_l47_4777

theorem rowing_students (X Y : ℕ) (N : ℕ) :
  (17 * X + 6 = N) →
  (10 * Y + 2 = N) →
  100 < N →
  N < 200 →
  5 ≤ X ∧ X ≤ 11 →
  10 ≤ Y ∧ Y ≤ 19 →
  N = 142 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_rowing_students_l47_4777


namespace NUMINAMATH_GPT_soccer_team_lineups_l47_4730

-- Define the number of players in the team
def numPlayers : Nat := 16

-- Define the number of regular players to choose (excluding the goalie)
def numRegularPlayers : Nat := 10

-- Define the total number of starting lineups, considering the goalie and the combination of regular players
def totalStartingLineups : Nat :=
  numPlayers * Nat.choose (numPlayers - 1) numRegularPlayers

-- The theorem to prove
theorem soccer_team_lineups : totalStartingLineups = 48048 := by
  sorry

end NUMINAMATH_GPT_soccer_team_lineups_l47_4730


namespace NUMINAMATH_GPT_find_lighter_ball_min_weighings_l47_4715

noncomputable def min_weighings_to_find_lighter_ball (balls : Fin 9 → ℕ) : ℕ :=
  2

-- Given: 9 balls, where 8 weigh 10 grams and 1 weighs 9 grams, and a balance scale.
theorem find_lighter_ball_min_weighings :
  (∃ i : Fin 9, balls i = 9 ∧ (∀ j : Fin 9, j ≠ i → balls j = 10)) 
  → min_weighings_to_find_lighter_ball balls = 2 :=
by
  intros
  sorry

end NUMINAMATH_GPT_find_lighter_ball_min_weighings_l47_4715


namespace NUMINAMATH_GPT_distance_A_B_l47_4722

noncomputable def distance_3d (x1 y1 z1 x2 y2 z2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2 + (z2 - z1) ^ 2)

theorem distance_A_B :
  distance_3d 4 1 9 10 (-1) 6 = 7 :=
by
  sorry

end NUMINAMATH_GPT_distance_A_B_l47_4722


namespace NUMINAMATH_GPT_total_seats_l47_4712

theorem total_seats (s : ℕ) 
  (h1 : 30 + (0.20 * s : ℝ) + (0.60 * s : ℝ) = s) : s = 150 :=
  sorry

end NUMINAMATH_GPT_total_seats_l47_4712


namespace NUMINAMATH_GPT_each_child_plays_40_minutes_l47_4733

variable (TotalMinutes : ℕ)
variable (NumChildren : ℕ)
variable (ChildPairs : ℕ)

theorem each_child_plays_40_minutes (h1 : TotalMinutes = 120) 
                                    (h2 : NumChildren = 6) 
                                    (h3 : ChildPairs = 2) :
  (ChildPairs * TotalMinutes) / NumChildren = 40 :=
by
  sorry

end NUMINAMATH_GPT_each_child_plays_40_minutes_l47_4733


namespace NUMINAMATH_GPT_find_a_b_l47_4783

theorem find_a_b (a b : ℝ) (h1 : b - a = -7) (h2 : 64 * (a + b) = 20736) :
  a = 165.5 ∧ b = 158.5 :=
by
  sorry

end NUMINAMATH_GPT_find_a_b_l47_4783


namespace NUMINAMATH_GPT_average_infection_l47_4774

theorem average_infection (x : ℕ) (h : 1 + 2 * x + x^2 = 121) : x = 10 :=
by
  sorry -- Proof to be filled.

end NUMINAMATH_GPT_average_infection_l47_4774


namespace NUMINAMATH_GPT_loisa_saves_70_l47_4707

def tablet_cash_price : ℕ := 450
def down_payment : ℕ := 100
def first_4_months_payment : ℕ := 40
def next_4_months_payment : ℕ := 35
def last_4_months_payment : ℕ := 30
def total_installment_payment : ℕ := down_payment + (4 * first_4_months_payment) + (4 * next_4_months_payment) + (4 * last_4_months_payment)
def savings : ℕ := total_installment_payment - tablet_cash_price

theorem loisa_saves_70 : savings = 70 := by
  sorry

end NUMINAMATH_GPT_loisa_saves_70_l47_4707


namespace NUMINAMATH_GPT_hall_reunion_attendees_l47_4767

theorem hall_reunion_attendees
  (total_guests : ℕ)
  (oates_attendees : ℕ)
  (both_attendees : ℕ)
  (h : total_guests = 100 ∧ oates_attendees = 50 ∧ both_attendees = 12) :
  ∃ (hall_attendees : ℕ), hall_attendees = 62 :=
by
  sorry

end NUMINAMATH_GPT_hall_reunion_attendees_l47_4767


namespace NUMINAMATH_GPT_brick_length_is_20_cm_l47_4778

theorem brick_length_is_20_cm
    (courtyard_length_m : ℕ) (courtyard_width_m : ℕ)
    (brick_length_cm : ℕ) (brick_width_cm : ℕ)
    (total_bricks_required : ℕ)
    (h1 : courtyard_length_m = 25)
    (h2 : courtyard_width_m = 16)
    (h3 : brick_length_cm = 20)
    (h4 : brick_width_cm = 10)
    (h5 : total_bricks_required = 20000) :
    brick_length_cm = 20 := 
by
    sorry

end NUMINAMATH_GPT_brick_length_is_20_cm_l47_4778


namespace NUMINAMATH_GPT_smallest_root_of_quadratic_l47_4736

theorem smallest_root_of_quadratic :
  ∃ x : ℝ, (12 * x^2 - 50 * x + 48 = 0) ∧ x = 1.333 := 
sorry

end NUMINAMATH_GPT_smallest_root_of_quadratic_l47_4736


namespace NUMINAMATH_GPT_complement_U_A_intersection_A_B_complement_U_intersection_A_B_complement_A_intersection_B_l47_4714

open Set -- Open the Set namespace for convenience

-- Define the universal set U, and sets A and B
def U : Set ℝ := univ
def A : Set ℝ := {x | -2 < x ∧ x < 3}
def B : Set ℝ := {x | -3 < x ∧ x ≤ 3}

-- Proof statements
theorem complement_U_A : U \ A = {x | x ≥ 3 ∨ x ≤ -2} :=
by sorry

theorem intersection_A_B : A ∩ B = {x | -2 < x ∧ x < 3} :=
by sorry

theorem complement_U_intersection_A_B : U \ (A ∩ B) = {x | x ≥ 3 ∨ x ≤ -2} :=
by sorry

theorem complement_A_intersection_B : (U \ A) ∩ B = {x | (-3 < x ∧ x ≤ -2) ∨ x = 3} :=
by sorry

end NUMINAMATH_GPT_complement_U_A_intersection_A_B_complement_U_intersection_A_B_complement_A_intersection_B_l47_4714


namespace NUMINAMATH_GPT_grandmother_dolls_l47_4739

-- Define the conditions
variable (S G : ℕ)

-- Rene has three times as many dolls as her sister
def rene_dolls : ℕ := 3 * S

-- The sister has two more dolls than their grandmother
def sister_dolls_eq : Prop := S = G + 2

-- Together they have a total of 258 dolls
def total_dolls : Prop := (rene_dolls S) + S + G = 258

-- Prove that the grandmother has 50 dolls given the conditions
theorem grandmother_dolls : sister_dolls_eq S G → total_dolls S G → G = 50 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_grandmother_dolls_l47_4739


namespace NUMINAMATH_GPT_eight_points_in_circle_l47_4709

theorem eight_points_in_circle :
  ∀ (P : Fin 8 → ℝ × ℝ), 
  (∀ i, (P i).1^2 + (P i).2^2 ≤ 1) → 
  ∃ (i j : Fin 8), i ≠ j ∧ ((P i).1 - (P j).1)^2 + ((P i).2 - (P j).2)^2 < 1 :=
by
  sorry

end NUMINAMATH_GPT_eight_points_in_circle_l47_4709


namespace NUMINAMATH_GPT_x_y_difference_l47_4719

theorem x_y_difference
    (x y : ℚ)
    (h1 : x + y = 780)
    (h2 : x / y = 1.25) :
    x - y = 86.66666666666667 :=
by
  sorry

end NUMINAMATH_GPT_x_y_difference_l47_4719


namespace NUMINAMATH_GPT_percent_profit_l47_4713

theorem percent_profit (cost : ℝ) (markup_percent : ℝ) (discount_percent : ℝ) (final_profit_percent : ℝ)
  (h1 : cost = 50)
  (h2 : markup_percent = 30)
  (h3 : discount_percent = 10)
  (h4 : final_profit_percent = 17)
  : (markup_percent / 100 * cost - discount_percent / 100 * (cost + markup_percent / 100 * cost)) / cost * 100 = final_profit_percent := 
by
  sorry

end NUMINAMATH_GPT_percent_profit_l47_4713


namespace NUMINAMATH_GPT_proof_solution_arithmetic_progression_l47_4792

noncomputable def system_has_solution (a b c m : ℝ) : Prop :=
  (m = 1 → a = b ∧ b = c) ∧
  (m = -2 → a + b + c = 0) ∧ 
  (m ≠ -2 ∧ m ≠ 1 → ∃ x y z : ℝ, x + y + m * z = a ∧ x + m * y + z = b ∧ m * x + y + z = c)

def abc_arithmetic_progression (a b c : ℝ) : Prop :=
  2 * b = a + c

theorem proof_solution_arithmetic_progression (a b c m : ℝ) : 
  system_has_solution a b c m → 
  (∃ x y z : ℝ, x + y + m * z = a ∧ x + m * y + z = b ∧ m * x + y + z = c ∧ 2 * y = x + z) ↔
  abc_arithmetic_progression a b c := 
by 
  sorry

end NUMINAMATH_GPT_proof_solution_arithmetic_progression_l47_4792


namespace NUMINAMATH_GPT_ratio_of_p_to_q_l47_4771

theorem ratio_of_p_to_q (p q r : ℚ) (h1: p = r * q) (h2: 18 / 7 + (2 * q - p) / (2 * q + p) = 3) : r = 29 / 10 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_p_to_q_l47_4771


namespace NUMINAMATH_GPT_boys_speed_l47_4746

-- Define the conditions
def sideLength : ℕ := 50
def timeTaken : ℕ := 72

-- Define the goal
theorem boys_speed (sideLength timeTaken : ℕ) (D T : ℝ) :
  D = (4 * sideLength : ℕ) / 1000 ∧
  T = timeTaken / 3600 →
  (D / T = 10) := by
  sorry

end NUMINAMATH_GPT_boys_speed_l47_4746


namespace NUMINAMATH_GPT_domain_of_f_eq_l47_4723

noncomputable def domain_of_f (x : ℝ) : Prop :=
  (x + 1 ≥ 0) ∧ (x ≠ 0)

theorem domain_of_f_eq :
  { x : ℝ | domain_of_f x} = { x : ℝ | -1 ≤ x ∧ x < 0 } ∪ { x : ℝ | 0 < x } :=
by
  sorry

end NUMINAMATH_GPT_domain_of_f_eq_l47_4723


namespace NUMINAMATH_GPT_flip_ratio_l47_4796

theorem flip_ratio (jen_triple_flips tyler_double_flips : ℕ)
  (hjen : jen_triple_flips = 16)
  (htyler : tyler_double_flips = 12)
  : 2 * tyler_double_flips / 3 * jen_triple_flips = 1 / 2 := 
by
  rw [hjen, htyler]
  norm_num
  sorry

end NUMINAMATH_GPT_flip_ratio_l47_4796


namespace NUMINAMATH_GPT_largest_circle_at_A_l47_4789

/--
Given a pentagon with side lengths AB = 16 cm, BC = 14 cm, CD = 17 cm, DE = 13 cm, and EA = 14 cm,
and given five circles with centers A, B, C, D, and E such that each pair of circles with centers at
the ends of a side of the pentagon touch on that side, the circle with center A
has the largest radius.
-/
theorem largest_circle_at_A
  (rA rB rC rD rE : ℝ) 
  (hAB : rA + rB = 16)
  (hBC : rB + rC = 14)
  (hCD : rC + rD = 17)
  (hDE : rD + rE = 13)
  (hEA : rE + rA = 14) :
  rA ≥ rB ∧ rA ≥ rC ∧ rA ≥ rD ∧ rA ≥ rE := 
sorry

end NUMINAMATH_GPT_largest_circle_at_A_l47_4789


namespace NUMINAMATH_GPT_current_balance_after_deduction_l47_4725

theorem current_balance_after_deduction :
  ∀ (original_balance deduction_percent : ℕ), 
  original_balance = 100000 →
  deduction_percent = 10 →
  original_balance - (deduction_percent * original_balance / 100) = 90000 :=
by
  intros original_balance deduction_percent h1 h2
  sorry

end NUMINAMATH_GPT_current_balance_after_deduction_l47_4725


namespace NUMINAMATH_GPT_odd_function_value_at_neg2_l47_4757

variable (f : ℝ → ℝ)
variable (h_odd : ∀ x, f (-x) = -f x)
variable (h_ge_one : ∀ x, 1 ≤ x → f x = 3 * x - 7)

theorem odd_function_value_at_neg2 : f (-2) = 1 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_odd_function_value_at_neg2_l47_4757


namespace NUMINAMATH_GPT_complex_expr_evaluation_l47_4762

def complex_expr : ℤ :=
  2 * (3 * (2 * (3 * (2 * (3 * (2 + 1) * 2) + 2) * 2) + 2) * 2) + 2

theorem complex_expr_evaluation : complex_expr = 5498 := by
  sorry

end NUMINAMATH_GPT_complex_expr_evaluation_l47_4762


namespace NUMINAMATH_GPT_find_reading_l47_4704

variable (a_1 a_2 a_3 a_4 : ℝ) (x : ℝ)
variable (h1 : a_1 = 2) (h2 : a_2 = 2.1) (h3 : a_3 = 2) (h4 : a_4 = 2.2)
variable (mean : (a_1 + a_2 + a_3 + a_4 + x) / 5 = 2)

theorem find_reading : x = 1.7 :=
by
  sorry

end NUMINAMATH_GPT_find_reading_l47_4704


namespace NUMINAMATH_GPT_determine_a_b_l47_4708

-- Definitions
def num (a b : ℕ) := 10000*a + 1000*6 + 100*7 + 10*9 + b

def divisible_by_72 (n : ℕ) : Prop := n % 72 = 0

noncomputable def a : ℕ := 3
noncomputable def b : ℕ := 2

-- Theorem statement
theorem determine_a_b : divisible_by_72 (num a b) :=
by
  -- The proof will be inserted here
  sorry

end NUMINAMATH_GPT_determine_a_b_l47_4708


namespace NUMINAMATH_GPT_min_omega_l47_4716

theorem min_omega (f : Real → Real) (ω φ : Real) (φ_bound : |φ| < π / 2) 
  (h1 : ω > 0) (h2 : f = fun x => Real.sin (ω * x + φ)) 
  (h3 : f 0 = 1/2) 
  (h4 : ∀ x, f x ≤ f (π / 12)) : ω = 4 := 
by
  sorry

end NUMINAMATH_GPT_min_omega_l47_4716


namespace NUMINAMATH_GPT_envelope_width_l47_4728

theorem envelope_width (Area Height Width : ℝ) (h_area : Area = 36) (h_height : Height = 6) (h_area_formula : Area = Width * Height) : Width = 6 :=
by
  sorry

end NUMINAMATH_GPT_envelope_width_l47_4728


namespace NUMINAMATH_GPT_graph_of_f_4_minus_x_l47_4775

theorem graph_of_f_4_minus_x (f : ℝ → ℝ) (h : f 0 = 1) : f (4 - 4) = 1 :=
by
  rw [sub_self]
  exact h

end NUMINAMATH_GPT_graph_of_f_4_minus_x_l47_4775


namespace NUMINAMATH_GPT_zoe_pop_albums_l47_4732

theorem zoe_pop_albums (total_songs country_albums songs_per_album : ℕ) (h1 : total_songs = 24) (h2 : country_albums = 3) (h3 : songs_per_album = 3) :
  total_songs - (country_albums * songs_per_album) = 15 ↔ (total_songs - (country_albums * songs_per_album)) / songs_per_album = 5 :=
by
  sorry

end NUMINAMATH_GPT_zoe_pop_albums_l47_4732


namespace NUMINAMATH_GPT_min_value_my_function_l47_4772

noncomputable def my_function (x : ℝ) : ℝ :=
  abs (x - 1) + 2 * abs (x - 2) + 3 * abs (x - 3) + 4 * abs (x - 4)

theorem min_value_my_function :
  ∃ (x : ℝ), my_function x = 8 ∧ (∀ y : ℝ, my_function y ≥ 8) :=
sorry

end NUMINAMATH_GPT_min_value_my_function_l47_4772


namespace NUMINAMATH_GPT_rate_percent_simple_interest_l47_4717

theorem rate_percent_simple_interest (P SI T : ℝ) (hP : P = 720) (hSI : SI = 180) (hT : T = 4) :
  (SI = P * (R / 100) * T) → R = 6.25 :=
by
  sorry

end NUMINAMATH_GPT_rate_percent_simple_interest_l47_4717


namespace NUMINAMATH_GPT_max_quotient_l47_4743

-- Define the given conditions
def conditions (a b : ℝ) :=
  100 ≤ a ∧ a ≤ 250 ∧ 700 ≤ b ∧ b ≤ 1400

-- State the theorem for the largest value of the quotient b / a
theorem max_quotient (a b : ℝ) (h : conditions a b) : b / a ≤ 14 :=
by
  sorry

end NUMINAMATH_GPT_max_quotient_l47_4743


namespace NUMINAMATH_GPT_find_sum_of_coordinates_of_other_endpoint_l47_4768

theorem find_sum_of_coordinates_of_other_endpoint :
  ∃ (x y : ℤ), (7, -5) = (10 + x / 2, 4 + y / 2) ∧ x + y = -10 :=
by
  sorry

end NUMINAMATH_GPT_find_sum_of_coordinates_of_other_endpoint_l47_4768


namespace NUMINAMATH_GPT_trig_identity_l47_4769

theorem trig_identity :
  (Real.sin (20 * Real.pi / 180) * Real.sin (50 * Real.pi / 180) + 
   Real.cos (20 * Real.pi / 180) * Real.sin (40 * Real.pi / 180)) = 
  (Real.sqrt 3 / 2) :=
by
  sorry

end NUMINAMATH_GPT_trig_identity_l47_4769


namespace NUMINAMATH_GPT_domain_all_real_l47_4731

theorem domain_all_real (p : ℝ) : 
  (∀ x : ℝ, -3 * x ^ 2 + 3 * x + p ≠ 0) ↔ p < -3 / 4 := 
by
  sorry

end NUMINAMATH_GPT_domain_all_real_l47_4731


namespace NUMINAMATH_GPT_fraction_area_outside_circle_l47_4763

theorem fraction_area_outside_circle (r : ℝ) (h1 : r > 0) :
  let side_length := 2 * r
  let area_square := side_length ^ 2
  let area_circle := π * r ^ 2
  let area_outside := area_square - area_circle
  (area_outside / area_square) = 1 - ↑π / 4 :=
by
  sorry

end NUMINAMATH_GPT_fraction_area_outside_circle_l47_4763


namespace NUMINAMATH_GPT_alphametic_puzzle_l47_4760

theorem alphametic_puzzle (I D A M E R O : ℕ) 
  (h1 : R = 0) 
  (h2 : D + E = 10)
  (h3 : I + M + 1 = O)
  (h4 : A = D + 1) :
  I + 1 + M + 10 + 1 = O + 0 + A := sorry

end NUMINAMATH_GPT_alphametic_puzzle_l47_4760


namespace NUMINAMATH_GPT_min_value_of_f_in_interval_l47_4770

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 4)

theorem min_value_of_f_in_interval :
  ∃ (x : ℝ), (0 ≤ x ∧ x ≤ Real.pi / 2) ∧ f x = -Real.sqrt 2 / 2 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_f_in_interval_l47_4770


namespace NUMINAMATH_GPT_debt_amount_is_40_l47_4755

theorem debt_amount_is_40 (l n t debt remaining : ℕ) (h_l : l = 6)
  (h_n1 : n = 5 * l) (h_n2 : n = 3 * t) (h_remaining : remaining = 6) 
  (h_share : ∀ x y z : ℕ, x = y ∧ y = z ∧ z = 2) :
  debt = 40 := 
by
  sorry

end NUMINAMATH_GPT_debt_amount_is_40_l47_4755


namespace NUMINAMATH_GPT_count_even_numbers_between_250_and_600_l47_4764

theorem count_even_numbers_between_250_and_600 : 
  ∃ n : ℕ, (n = 175 ∧ 
    ∀ k : ℕ, (250 < 2 * k ∧ 2 * k ≤ 600) ↔ (126 ≤ k ∧ k ≤ 300)) :=
by
  sorry

end NUMINAMATH_GPT_count_even_numbers_between_250_and_600_l47_4764


namespace NUMINAMATH_GPT_percentage_alcohol_second_vessel_l47_4750

theorem percentage_alcohol_second_vessel :
  (∀ (x : ℝ),
    (0.25 * 3 + (x / 100) * 5 = 0.275 * 10) -> x = 40) :=
by
  intro x h
  sorry

end NUMINAMATH_GPT_percentage_alcohol_second_vessel_l47_4750


namespace NUMINAMATH_GPT_both_selected_prob_l47_4724

noncomputable def prob_X : ℚ := 1 / 3
noncomputable def prob_Y : ℚ := 2 / 7
noncomputable def combined_prob : ℚ := prob_X * prob_Y

theorem both_selected_prob :
  combined_prob = 2 / 21 :=
by
  unfold combined_prob prob_X prob_Y
  sorry

end NUMINAMATH_GPT_both_selected_prob_l47_4724


namespace NUMINAMATH_GPT_base_of_first_term_is_two_l47_4720

-- Define h as a positive integer
variable (h : ℕ) (a b c : ℕ)

-- Conditions
variables 
  (h_positive : h > 0)
  (divisor_225 : 225 ∣ h)
  (divisor_216 : 216 ∣ h)

-- Given h can be expressed as specified and a + b + c = 8
variable (h_expression : ∃ k : ℕ, h = k^a * 3^b * 5^c)
variable (sum_eight : a + b + c = 8)

-- Prove the base of the first term in the expression for h is 2.
theorem base_of_first_term_is_two : (∃ k : ℕ, k^a * 3^b * 5^c = h) → k = 2 :=
by 
  sorry

end NUMINAMATH_GPT_base_of_first_term_is_two_l47_4720


namespace NUMINAMATH_GPT_cube_pyramid_volume_l47_4759

theorem cube_pyramid_volume (s b h : ℝ) 
  (hcube : s = 6) 
  (hbase : b = 10)
  (eq_volumes : (s ^ 3) = (1 / 3) * (b ^ 2) * h) : 
  h = 162 / 25 := 
by 
  sorry

end NUMINAMATH_GPT_cube_pyramid_volume_l47_4759


namespace NUMINAMATH_GPT_hyperbola_foci_on_x_axis_l47_4749

theorem hyperbola_foci_on_x_axis (a : ℝ) 
  (h1 : 1 - a < 0)
  (h2 : a - 3 > 0)
  (h3 : ∀ c, c = 2 → 2 * c = 4) : 
  a = 4 := 
sorry

end NUMINAMATH_GPT_hyperbola_foci_on_x_axis_l47_4749


namespace NUMINAMATH_GPT_average_income_P_Q_l47_4705

   variable (P Q R : ℝ)

   theorem average_income_P_Q
     (h1 : (Q + R) / 2 = 6250)
     (h2 : (P + R) / 2 = 5200)
     (h3 : P = 4000) :
     (P + Q) / 2 = 5050 := by
   sorry
   
end NUMINAMATH_GPT_average_income_P_Q_l47_4705


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l47_4797

def A : Set ℤ := {-1, 0, 3, 5}
def B : Set ℤ := {x | x - 2 > 0}

theorem intersection_of_A_and_B : A ∩ B = {3, 5} := 
by 
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l47_4797


namespace NUMINAMATH_GPT_sum_of_a5_a6_l47_4734

noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n * q

noncomputable def geometric_conditions (a : ℕ → ℝ) (q : ℝ) : Prop :=
a 1 + a 2 = 1 ∧ a 3 + a 4 = 4 ∧ q^2 = 4

theorem sum_of_a5_a6 (a : ℕ → ℝ) (q : ℝ) (h_seq : geometric_sequence a q) (h_cond : geometric_conditions a q) :
  a 5 + a 6 = 16 :=
sorry

end NUMINAMATH_GPT_sum_of_a5_a6_l47_4734


namespace NUMINAMATH_GPT_union_sets_l47_4758

def A := { x : ℝ | x^2 ≤ 1 }
def B := { x : ℝ | 0 < x }

theorem union_sets : A ∪ B = { x | -1 ≤ x } :=
by {
  sorry -- Proof is omitted as per the instructions
}

end NUMINAMATH_GPT_union_sets_l47_4758


namespace NUMINAMATH_GPT_positive_divisors_d17_l47_4773

theorem positive_divisors_d17 (n : ℕ) (d : ℕ → ℕ) (k : ℕ) (h_order : d 1 = 1 ∧ ∀ i, 1 ≤ i → i ≤ k → d i < d (i + 1)) 
  (h_last : d k = n) (h_pythagorean : d 7 ^ 2 + d 15 ^ 2 = d 16 ^ 2) : 
  d 17 = 28 :=
sorry

end NUMINAMATH_GPT_positive_divisors_d17_l47_4773
