import Mathlib

namespace NUMINAMATH_GPT_ella_stamps_value_l938_93832

theorem ella_stamps_value :
  let total_stamps := 18
  let value_of_6_stamps := 18
  let consistent_value_per_stamp := value_of_6_stamps / 6
  total_stamps * consistent_value_per_stamp = 54 := by
  sorry

end NUMINAMATH_GPT_ella_stamps_value_l938_93832


namespace NUMINAMATH_GPT_solve_for_k_l938_93858

theorem solve_for_k :
  (∀ x : ℤ, (2 * x + 4 = 4 * (x - 2)) ↔ ( -x + 17 = 2 * x - 1 )) :=
by
  sorry

end NUMINAMATH_GPT_solve_for_k_l938_93858


namespace NUMINAMATH_GPT_kevin_wings_record_l938_93862

-- Conditions
def alanWingsPerMinute : ℕ := 5
def additionalWingsNeeded : ℕ := 4
def kevinRecordDuration : ℕ := 8

-- Question and answer
theorem kevin_wings_record : 
  (alanWingsPerMinute + additionalWingsNeeded) * kevinRecordDuration = 72 :=
by
  sorry

end NUMINAMATH_GPT_kevin_wings_record_l938_93862


namespace NUMINAMATH_GPT_sum_of_coefficients_is_neg40_l938_93835

noncomputable def p (x : ℝ) : ℝ := 3 * (x^8 - x^5 + 2 * x^3 - 6) - 5 * (x^4 + 3 * x^2) + 2 * (x^6 - 5)

theorem sum_of_coefficients_is_neg40 : p 1 = -40 := by
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_is_neg40_l938_93835


namespace NUMINAMATH_GPT_remainder_calculation_l938_93874

theorem remainder_calculation 
  (dividend divisor quotient : ℕ)
  (h1 : dividend = 140)
  (h2 : divisor = 15)
  (h3 : quotient = 9) :
  dividend = (divisor * quotient) + (dividend - (divisor * quotient)) := by
sorry

end NUMINAMATH_GPT_remainder_calculation_l938_93874


namespace NUMINAMATH_GPT_inequality_solution_l938_93863

open Set

theorem inequality_solution (x : ℝ) : (1 - 7 / (2 * x - 1) < 0) ↔ (1 / 2 < x ∧ x < 4) := 
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l938_93863


namespace NUMINAMATH_GPT_reeya_third_subject_score_l938_93803

theorem reeya_third_subject_score (s1 s2 s3 s4 : ℝ) (average : ℝ) (num_subjects : ℝ) (total_score : ℝ) :
    s1 = 65 → s2 = 67 → s4 = 95 → average = 76.6 → num_subjects = 4 → total_score = 306.4 →
    (s1 + s2 + s3 + s4) / num_subjects = average → s3 = 79.4 :=
by
  intros h1 h2 h4 h_average h_num_subjects h_total_score h_avg_eq
  -- Proof steps can be added here
  sorry

end NUMINAMATH_GPT_reeya_third_subject_score_l938_93803


namespace NUMINAMATH_GPT_problem_1_problem_2_l938_93875

def set_A := { y : ℝ | 2 < y ∧ y < 3 }
def set_B := { x : ℝ | x > 1 ∨ x < -1 }

theorem problem_1 : { x : ℝ | x ∈ set_A ∧ x ∈ set_B } = { x : ℝ | 2 < x ∧ x < 3 } :=
by
  sorry

def set_C := { x : ℝ | x ∈ set_B ∧ ¬(x ∈ set_A) }

theorem problem_2 : set_C = { x : ℝ | x < -1 ∨ (1 < x ∧ x ≤ 2) ∨ x ≥ 3 } :=
by
  sorry

end NUMINAMATH_GPT_problem_1_problem_2_l938_93875


namespace NUMINAMATH_GPT_line_does_not_pass_through_third_quadrant_l938_93817

variable {a b c : ℝ}

theorem line_does_not_pass_through_third_quadrant
  (hac : a * c < 0) (hbc : b * c < 0) : ¬ ∃ x y, x < 0 ∧ y < 0 ∧ a * x + b * y + c = 0 :=
sorry

end NUMINAMATH_GPT_line_does_not_pass_through_third_quadrant_l938_93817


namespace NUMINAMATH_GPT_length_ab_l938_93854

section geometry

variables {A B C D E : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C]

-- Define the lengths and perimeters as needed
variables (AB AC BC CD DE CE : ℝ)

-- Isosceles Triangle properties
axiom isosceles_abc : AC = BC
axiom isosceles_cde : CD = DE

-- Conditons given in the problem
axiom perimeter_cde : CE + CD + DE = 22
axiom perimeter_abc : AB + BC + AC = 24
axiom length_ce : CE = 8

-- Goal: To prove the length of AB
theorem length_ab : AB = 10 :=
by 
  sorry

end geometry

end NUMINAMATH_GPT_length_ab_l938_93854


namespace NUMINAMATH_GPT_combined_eel_length_l938_93889

def Lengths : Type := { j : ℕ // j = 16 }

def jenna_eel_length : Lengths := ⟨16, rfl⟩

def bill_eel_length (j : Lengths) : ℕ := 3 * j.val

#check bill_eel_length

theorem combined_eel_length (j : Lengths) :
  j.val + bill_eel_length j = 64 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_combined_eel_length_l938_93889


namespace NUMINAMATH_GPT_acute_angle_proof_l938_93808

theorem acute_angle_proof
  (α β : ℝ) 
  (hα : 0 < α ∧ α < π / 2) 
  (hβ : 0 < β ∧ β < π / 2) 
  (h : Real.cos (α + β) = Real.sin (α - β)) : α = π / 4 :=
  sorry

end NUMINAMATH_GPT_acute_angle_proof_l938_93808


namespace NUMINAMATH_GPT_complex_number_identity_l938_93853

theorem complex_number_identity : |-i| + i^2018 = 0 := by
  sorry

end NUMINAMATH_GPT_complex_number_identity_l938_93853


namespace NUMINAMATH_GPT_eval_six_times_f_l938_93814

def f (x : Int) : Int :=
  if x % 2 == 0 then
    x / 2
  else
    5 * x + 1

theorem eval_six_times_f : f (f (f (f (f (f 7))))) = 116 := 
by
  -- Skipping proof body (since it's not required)
  sorry

end NUMINAMATH_GPT_eval_six_times_f_l938_93814


namespace NUMINAMATH_GPT_copper_tin_alloy_weight_l938_93806

theorem copper_tin_alloy_weight :
  let c1 := (4/5 : ℝ) * 10 -- Copper in the first alloy
  let t1 := (1/5 : ℝ) * 10 -- Tin in the first alloy
  let c2 := (1/4 : ℝ) * 16 -- Copper in the second alloy
  let t2 := (3/4 : ℝ) * 16 -- Tin in the second alloy
  let x := ((3 * 14 - 24) / 2 : ℝ) -- Pure copper added
  let total_copper := c1 + c2 + x
  let total_tin := t1 + t2
  total_copper + total_tin = 35 := 
by
  sorry

end NUMINAMATH_GPT_copper_tin_alloy_weight_l938_93806


namespace NUMINAMATH_GPT_real_roots_of_quadratic_l938_93870

noncomputable def quadratic_discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem real_roots_of_quadratic (m : ℝ) : (∃ x : ℝ, x^2 - x - m = 0) ↔ m ≥ -1/4 := by
  sorry

end NUMINAMATH_GPT_real_roots_of_quadratic_l938_93870


namespace NUMINAMATH_GPT_parabola_trajectory_l938_93886

theorem parabola_trajectory (P : ℝ × ℝ) : 
  (dist P (3, 0) = dist P (3 - 1, P.2 - 0)) → P.2^2 = 12 * P.1 := 
sorry

end NUMINAMATH_GPT_parabola_trajectory_l938_93886


namespace NUMINAMATH_GPT_sam_bikes_speed_l938_93851

noncomputable def EugeneSpeed : ℝ := 5
noncomputable def ClaraSpeed : ℝ := (3/4) * EugeneSpeed
noncomputable def SamSpeed : ℝ := (4/3) * ClaraSpeed

theorem sam_bikes_speed :
  SamSpeed = 5 :=
by
  -- Proof will be filled here.
  sorry

end NUMINAMATH_GPT_sam_bikes_speed_l938_93851


namespace NUMINAMATH_GPT_find_t_l938_93847

variable (g V V0 c S t : ℝ)
variable (h1 : V = g * t + V0 + c)
variable (h2 : S = (1/2) * g * t^2 + V0 * t + c * t^2)

theorem find_t
  (h1 : V = g * t + V0 + c)
  (h2 : S = (1/2) * g * t^2 + V0 * t + c * t^2) :
  t = 2 * S / (V + V0 - c) :=
sorry

end NUMINAMATH_GPT_find_t_l938_93847


namespace NUMINAMATH_GPT_find_garden_perimeter_l938_93824

noncomputable def garden_perimeter (a : ℝ) (P : ℝ) : Prop :=
  a = 2 * P + 14.25 ∧ a = 90.25

theorem find_garden_perimeter :
  ∃ P : ℝ, garden_perimeter 90.25 P ∧ P = 38 :=
by
  sorry

end NUMINAMATH_GPT_find_garden_perimeter_l938_93824


namespace NUMINAMATH_GPT_intersection_M_N_l938_93850

open Set

def M := {x : ℝ | -2 ≤ x ∧ x ≤ 2}
def N := {x : ℝ | x > 1}

theorem intersection_M_N :
  M ∩ N = {x : ℝ | 1 < x ∧ x ≤ 2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l938_93850


namespace NUMINAMATH_GPT_rent_3600_rents_88_max_revenue_is_4050_l938_93840

def num_total_cars : ℕ := 100
def initial_rent : ℕ := 3000
def rent_increase_step : ℕ := 50
def maintenance_cost_rented : ℕ := 150
def maintenance_cost_unrented : ℕ := 50

def rented_cars (rent : ℕ) : ℕ :=
  if rent < initial_rent then num_total_cars
  else num_total_cars - ((rent - initial_rent) / rent_increase_step)

def monthly_revenue (rent : ℕ) : ℕ :=
  let rented := rented_cars rent
  rent * rented - (rented * maintenance_cost_rented + (num_total_cars - rented) * maintenance_cost_unrented)

theorem rent_3600_rents_88 :
  rented_cars 3600 = 88 := by 
  sorry

theorem max_revenue_is_4050 :
  ∃ (rent : ℕ), rent = 4050 ∧ monthly_revenue rent = 37050 := by
  sorry

end NUMINAMATH_GPT_rent_3600_rents_88_max_revenue_is_4050_l938_93840


namespace NUMINAMATH_GPT_game_cost_l938_93884

theorem game_cost (initial_money : ℕ) (toys_count : ℕ) (toy_price : ℕ) (left_money : ℕ) : 
  initial_money = 63 ∧ toys_count = 5 ∧ toy_price = 3 ∧ left_money = 15 → 
  (initial_money - left_money = 48) :=
by
  sorry

end NUMINAMATH_GPT_game_cost_l938_93884


namespace NUMINAMATH_GPT_avg_bc_eq_70_l938_93818

-- Definitions of the given conditions
variables (a b c : ℝ)

def avg_ab (a b : ℝ) : Prop := (a + b) / 2 = 45
def diff_ca (a c : ℝ) : Prop := c - a = 50

-- The main theorem statement
theorem avg_bc_eq_70 (h1 : avg_ab a b) (h2 : diff_ca a c) : (b + c) / 2 = 70 :=
by
  sorry

end NUMINAMATH_GPT_avg_bc_eq_70_l938_93818


namespace NUMINAMATH_GPT_Benny_and_Tim_have_47_books_together_l938_93825

/-
  Definitions and conditions:
  1. Benny_has_24_books : Benny has 24 books.
  2. Benny_gave_10_books_to_Sandy : Benny gave Sandy 10 books.
  3. Tim_has_33_books : Tim has 33 books.
  
  Goal:
  Prove that together Benny and Tim have 47 books.
-/

def Benny_has_24_books : ℕ := 24
def Benny_gave_10_books_to_Sandy : ℕ := 10
def Tim_has_33_books : ℕ := 33

def Benny_remaining_books : ℕ := Benny_has_24_books - Benny_gave_10_books_to_Sandy

def Benny_and_Tim_together : ℕ := Benny_remaining_books + Tim_has_33_books

theorem Benny_and_Tim_have_47_books_together :
  Benny_and_Tim_together = 47 := by
  sorry

end NUMINAMATH_GPT_Benny_and_Tim_have_47_books_together_l938_93825


namespace NUMINAMATH_GPT_sum_of_coefficients_eq_10_l938_93899

theorem sum_of_coefficients_eq_10 
  (s : ℕ → ℝ) 
  (a b c : ℝ) 
  (h0 : s 0 = 3) 
  (h1 : s 1 = 5) 
  (h2 : s 2 = 9)
  (h : ∀ k ≥ 2, s (k + 1) = a * s k + b * s (k - 1) + c * s (k - 2)) : 
  a + b + c = 10 :=
sorry

end NUMINAMATH_GPT_sum_of_coefficients_eq_10_l938_93899


namespace NUMINAMATH_GPT_david_marks_in_english_l938_93836

theorem david_marks_in_english 
  (math : ℤ) (phys : ℤ) (chem : ℤ) (bio : ℤ) (avg : ℤ) 
  (marks_per_math : math = 85) 
  (marks_per_phys : phys = 92) 
  (marks_per_chem : chem = 87) 
  (marks_per_bio : bio = 95) 
  (avg_marks : avg = 89) 
  (num_subjects : ℤ := 5) :
  ∃ (eng : ℤ), eng + 85 + 92 + 87 + 95 = 89 * 5 ∧ eng = 86 :=
by
  sorry

end NUMINAMATH_GPT_david_marks_in_english_l938_93836


namespace NUMINAMATH_GPT_calculate_max_marks_l938_93844

theorem calculate_max_marks (shortfall_math : ℕ) (shortfall_science : ℕ) 
                            (shortfall_literature : ℕ) (shortfall_social_studies : ℕ)
                            (required_math : ℕ) (required_science : ℕ)
                            (required_literature : ℕ) (required_social_studies : ℕ)
                            (max_math : ℕ) (max_science : ℕ)
                            (max_literature : ℕ) (max_social_studies : ℕ) :
                            shortfall_math = 40 ∧ required_math = 95 ∧ max_math = 800 ∧
                            shortfall_science = 35 ∧ required_science = 92 ∧ max_science = 438 ∧
                            shortfall_literature = 30 ∧ required_literature = 90 ∧ max_literature = 300 ∧
                            shortfall_social_studies = 25 ∧ required_social_studies = 88 ∧ max_social_studies = 209 :=
by
  sorry

end NUMINAMATH_GPT_calculate_max_marks_l938_93844


namespace NUMINAMATH_GPT_distinct_real_roots_find_other_root_and_k_l938_93857

-- Definition of the quadratic equation
def quadratic_eq (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Part (1): Proving the discriminant condition
theorem distinct_real_roots (k : ℝ) : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ quadratic_eq 2 k (-1) x1 = 0 ∧ quadratic_eq 2 k (-1) x2 = 0 := by
  sorry

-- Part (2): Finding the other root and the value of k
theorem find_other_root_and_k : 
  ∃ k : ℝ, ∃ x2 : ℝ,
    quadratic_eq 2 1 (-1) (-1) = 0 ∧ quadratic_eq 2 1 (-1) x2 = 0 ∧ k = 1 ∧ x2 = 1/2 := by
  sorry

end NUMINAMATH_GPT_distinct_real_roots_find_other_root_and_k_l938_93857


namespace NUMINAMATH_GPT_lcm_of_36_and_45_l938_93864

theorem lcm_of_36_and_45 : Nat.lcm 36 45 = 180 := by
  sorry

end NUMINAMATH_GPT_lcm_of_36_and_45_l938_93864


namespace NUMINAMATH_GPT_building_height_l938_93837

-- We start by defining the heights of the stories.
def first_story_height : ℕ := 12
def additional_height_per_story : ℕ := 3
def number_of_stories : ℕ := 20
def first_ten_stories : ℕ := 10
def remaining_stories : ℕ := number_of_stories - first_ten_stories

-- Now we define what it means for the total height of the building to be 270 feet.
theorem building_height :
  first_ten_stories * first_story_height + remaining_stories * (first_story_height + additional_height_per_story) = 270 := by
  sorry

end NUMINAMATH_GPT_building_height_l938_93837


namespace NUMINAMATH_GPT_correct_assignment_statement_l938_93848

def is_assignment_statement (stmt : String) : Prop :=
  stmt = "a = 2a"

theorem correct_assignment_statement : is_assignment_statement "a = 2a" :=
by
  sorry

end NUMINAMATH_GPT_correct_assignment_statement_l938_93848


namespace NUMINAMATH_GPT_intersection_A_B_l938_93890

def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
def B : Set ℝ := {0, 2, 4, 6}

theorem intersection_A_B : A ∩ B = {0} :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l938_93890


namespace NUMINAMATH_GPT_inequality_solution_set_range_of_k_l938_93895

variable {k m x : ℝ}

theorem inequality_solution_set (k_pos : k > 0) 
  (f : ℝ → ℝ) (hf : ∀ x, f x = k * x / (x^2 + 3 * k)) 
  (sol_set_f_x_gt_m : ∀ x, f x > m ↔ (x < -3 ∨ x > -2)) :
  -1 < x ∧ x < 3 / 2 := 
sorry

theorem range_of_k (k_pos : k > 0) 
  (f : ℝ → ℝ) (hf : ∀ x, f x = k * x / (x^2 + 3 * k))
  (exists_f_x_gt_1 : ∃ x > 3, f x > 1) : 
  k > 12 :=
sorry

end NUMINAMATH_GPT_inequality_solution_set_range_of_k_l938_93895


namespace NUMINAMATH_GPT_min_value_of_sum_l938_93865

theorem min_value_of_sum (a b : ℤ) (h : a * b = 150) : a + b = -151 :=
  sorry

end NUMINAMATH_GPT_min_value_of_sum_l938_93865


namespace NUMINAMATH_GPT_quadratic_int_roots_iff_n_eq_3_or_4_l938_93892

theorem quadratic_int_roots_iff_n_eq_3_or_4 (n : ℕ) (hn : 0 < n) :
    (∃ m k : ℤ, (m ≠ k) ∧ (m^2 - 4 * m + n = 0) ∧ (k^2 - 4 * k + n = 0)) ↔ (n = 3 ∨ n = 4) := sorry

end NUMINAMATH_GPT_quadratic_int_roots_iff_n_eq_3_or_4_l938_93892


namespace NUMINAMATH_GPT_number_pairs_sum_diff_prod_quotient_l938_93827

theorem number_pairs_sum_diff_prod_quotient (x y : ℤ) (h : x ≥ y) :
  (x + y) + (x - y) + x * y + x / y = 800 ∨ (x + y) + (x - y) + x * y + x / y = 400 :=
sorry

-- Correct answers for A = 800
example : (38 + 19) + (38 - 19) + 38 * 19 + 38 / 19 = 800 := by norm_num
example : (-42 + -21) + (-42 - -21) + (-42 * -21) + (-42 / -21) = 800 := by norm_num
example : (72 + 9) + (72 - 9) + 72 * 9 + 72 / 9 = 800 := by norm_num
example : (-88 + -11) + (-88 - -11) + -(88 * -11) + (-88 / -11) = 800 := by norm_num
example : (128 + 4) + (128 - 4) + 128 * 4 + 128 / 4 = 800 := by norm_num
example : (-192 + -6) + (-192 - -6) + -192 * -6 + ( -192 / -6 ) = 800 := by norm_num
example : (150 + 3) + (150 - 3) + 150 * 3 + 150 / 3 = 800 := by norm_num
example : (-250 + -5) + (-250 - -5) + (-250 * -5) + (-250 / -5) = 800 := by norm_num
example : (200 + 1) + (200 - 1) + 200 * 1 + 200 / 1 = 800 := by norm_num
example : (-600 + -3) + (-600 - -3) + -600 * -3 + -600 / -3 = 800 := by norm_num

-- Correct answers for A = 400
example : (19 + 19) + (19 - 19) + 19 * 19 + 19 / 19 = 400 := by norm_num
example : (-21 + -21) + (-21 - -21) + (-21 * -21) + (-21 / -21) = 400 := by norm_num
example : (36 + 9) + (36 - 9) + 36 * 9 + 36 / 9 = 400 := by norm_num
example : (-44 + -11) + (-44 - -11) + (-44 * -11) + (-44 / -11) = 400 := by norm_num
example : (64 + 4) + (64 - 4) + 64 * 4 + 64 / 4 = 400 := by norm_num
example : (-96 + -6) + (-96 - -6) + (-96 * -6) + (-96 / -6) = 400 := by norm_num
example : (75 + 3) + (75 - 3) + 75 * 3 + 75 / 3 = 400 := by norm_num
example : (-125 + -5) + (-125 - -5) + (-125 * -5) + (-125 / -5) = 400 := by norm_num
example : (100 + 1) + (100 - 1) + 100 * 1 + 100 / 1 = 400 := by norm_num
example : (-300 + -3) + (-300 - -3) + (-300 * -3) + (-300 / -3) = 400 := by norm_num

end NUMINAMATH_GPT_number_pairs_sum_diff_prod_quotient_l938_93827


namespace NUMINAMATH_GPT_earnings_from_roosters_l938_93866

-- Definitions from the conditions
def price_per_kg : Float := 0.50
def weight_of_rooster1 : Float := 30.0
def weight_of_rooster2 : Float := 40.0

-- The theorem we need to prove (mathematically equivalent proof problem)
theorem earnings_from_roosters (p : Float := price_per_kg)
                               (w1 : Float := weight_of_rooster1)
                               (w2 : Float := weight_of_rooster2) :
  p * w1 + p * w2 = 35.0 := 
by {
  sorry
}

end NUMINAMATH_GPT_earnings_from_roosters_l938_93866


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_to_increasing_l938_93869

theorem sufficient_but_not_necessary_to_increasing (a : ℝ) :
  (∀ x y : ℝ, 1 ≤ x → x ≤ y → (x^2 - 2*a*x) ≤ (y^2 - 2*a*y)) ↔ (a ≤ 1) := sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_to_increasing_l938_93869


namespace NUMINAMATH_GPT_budget_spent_on_salaries_l938_93830

theorem budget_spent_on_salaries :
  ∀ (B R U E S T : ℕ),
  R = 9 ∧
  U = 5 ∧
  E = 4 ∧
  S = 2 ∧
  T = (72 * 100) / 360 → 
  B = 100 →
  (B - (R + U + E + S + T)) = 60 :=
by sorry

end NUMINAMATH_GPT_budget_spent_on_salaries_l938_93830


namespace NUMINAMATH_GPT_english_score_l938_93897

theorem english_score (s1 s2 s3 e : ℕ) :
  (s1 + s2 + s3) = 276 → (s1 + s2 + s3 + e) = 376 → e = 100 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_english_score_l938_93897


namespace NUMINAMATH_GPT_smallest_prime_factor_of_difference_l938_93855

theorem smallest_prime_factor_of_difference (A B C : ℕ) (hA : 1 ≤ A ∧ A ≤ 9) (hB : 0 ≤ B ∧ B ≤ 9) (hC : 1 ≤ C ∧ C ≤ 9) (h_diff : A ≠ C) : 
  ∃ p : ℕ, Prime p ∧ p ∣ (100 * A + 10 * B + C - (100 * C + 10 * B + A)) ∧ p = 3 :=
by
  sorry

end NUMINAMATH_GPT_smallest_prime_factor_of_difference_l938_93855


namespace NUMINAMATH_GPT_calculate_wheel_radii_l938_93816

theorem calculate_wheel_radii (rpmA rpmB : ℕ) (length : ℝ) (r R : ℝ) :
  rpmA = 1200 →
  rpmB = 1500 →
  length = 9 →
  (4 : ℝ) / 5 * r = R →
  2 * (R + r) = 9 →
  r = 2 ∧ R = 2.5 :=
by
  intros
  sorry

end NUMINAMATH_GPT_calculate_wheel_radii_l938_93816


namespace NUMINAMATH_GPT_foldable_topless_cubical_box_count_l938_93888

def isFoldable (placement : Char) : Bool :=
  placement = 'C' ∨ placement = 'E' ∨ placement = 'G'

theorem foldable_topless_cubical_box_count :
  (List.filter isFoldable ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']).length = 3 :=
by
  sorry

end NUMINAMATH_GPT_foldable_topless_cubical_box_count_l938_93888


namespace NUMINAMATH_GPT_similar_triangle_perimeter_l938_93839

theorem similar_triangle_perimeter 
  (a b c : ℝ) (a_sim : ℝ)
  (h1 : a = b) (h2 : b = c)
  (h3 : a = 15) (h4 : a_sim = 45)
  (h5 : a_sim / a = 3) :
  a_sim + a_sim + a_sim = 135 :=
by
  sorry

end NUMINAMATH_GPT_similar_triangle_perimeter_l938_93839


namespace NUMINAMATH_GPT_math_proof_problem_l938_93805

namespace Proofs

-- Definition of the arithmetic sequence {a_n}
def arithmetic_seq (a : ℕ → ℤ) : Prop := 
  ∀ m n, a n = a m + (n - m) * (a (m + 1) - a m)

-- Conditions for the arithmetic sequence
def a_conditions (a : ℕ → ℤ) : Prop := 
  a 3 = -6 ∧ a 6 = 0

-- Definition of the geometric sequence {b_n}
def geometric_seq (b : ℕ → ℤ) : Prop := 
  ∃ q, ∀ n, b (n + 1) = q * b n

-- Conditions for the geometric sequence
def b_conditions (b a : ℕ → ℤ) : Prop := 
  b 1 = -8 ∧ b 2 = a 1 + a 2 + a 3

-- The general formula for {a_n}
def a_formula (a : ℕ → ℤ) :=
  ∀ n, a n = 2 * n - 12

-- The sum formula of the first n terms of {b_n}
def S_n_formula (b : ℕ → ℤ) (S_n : ℕ → ℤ) :=
  ∀ n, S_n n = 4 * (1 - 3^n)

-- The main theorem combining all
theorem math_proof_problem (a b : ℕ → ℤ) (S_n : ℕ → ℤ) :
  arithmetic_seq a →
  a_conditions a →
  geometric_seq b →
  b_conditions b a →
  (a_formula a ∧ S_n_formula b S_n) :=
by 
  sorry

end Proofs

end NUMINAMATH_GPT_math_proof_problem_l938_93805


namespace NUMINAMATH_GPT_determine_initial_fund_l938_93838

def initial_amount_fund (n : ℕ) := 60 * n + 30 - 10

theorem determine_initial_fund (n : ℕ) (h : 50 * n + 110 = 60 * n - 10) : initial_amount_fund n = 740 :=
by
  -- we skip the proof steps here
  sorry

end NUMINAMATH_GPT_determine_initial_fund_l938_93838


namespace NUMINAMATH_GPT_total_amount_l938_93833

variable (Brad Josh Doug : ℝ)

axiom h1 : Josh = 2 * Brad
axiom h2 : Josh = (3 / 4) * Doug
axiom h3 : Doug = 32

theorem total_amount : Brad + Josh + Doug = 68 := by
  sorry

end NUMINAMATH_GPT_total_amount_l938_93833


namespace NUMINAMATH_GPT_simplify_expression_l938_93873

theorem simplify_expression (m : ℝ) (h1 : m ≠ 3) :
  (m / (m - 3) + 2 / (3 - m)) / ((m - 2) / (m^2 - 6 * m + 9)) = m - 3 := 
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l938_93873


namespace NUMINAMATH_GPT_coefficient_x2y3_in_expansion_l938_93871

-- Define the binomial coefficient function
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- State the theorem
theorem coefficient_x2y3_in_expansion (x y : ℝ) : 
  binomial 5 3 * (2 : ℝ) ^ 2 * (-1 : ℝ) ^ 3 = -40 := by
sorry

end NUMINAMATH_GPT_coefficient_x2y3_in_expansion_l938_93871


namespace NUMINAMATH_GPT_simplify_expr_C_l938_93891

theorem simplify_expr_C (x y : ℝ) : 5 * x - (x - 2 * y) = 4 * x + 2 * y :=
by
  sorry

end NUMINAMATH_GPT_simplify_expr_C_l938_93891


namespace NUMINAMATH_GPT_length_of_rectangular_sheet_l938_93813

/-- The length of each rectangular sheet is 10 cm given that:
    1. Two identical rectangular sheets each have an area of 48 square centimeters,
    2. The covered area when overlapping the sheets is 72 square centimeters,
    3. The diagonal BD of the overlapping quadrilateral ABCD is 6 centimeters. -/
theorem length_of_rectangular_sheet :
  ∀ (length width : ℝ),
    width * length = 48 ∧
    2 * 48 - 72 = width * 6 ∧
    width * 6 = 24 →
    length = 10 :=
sorry

end NUMINAMATH_GPT_length_of_rectangular_sheet_l938_93813


namespace NUMINAMATH_GPT_sum_of_fifth_powers_cannot_conclude_sum_of_fourth_powers_l938_93810

-- Definition of conditions
variables {a b c d : ℝ} 

-- First proof statement
theorem sum_of_fifth_powers 
  (h1 : a + b = c + d) 
  (h2 : a^3 + b^3 = c^3 + d^3) : 
  a^5 + b^5 = c^5 + d^5 := 
sorry

-- Second proof statement
theorem cannot_conclude_sum_of_fourth_powers 
  (h1 : a + b = c + d) 
  (h2 : a^3 + b^3 = c^3 + d^3) : 
  ¬(a^4 + b^4 = c^4 + d^4) := 
sorry

end NUMINAMATH_GPT_sum_of_fifth_powers_cannot_conclude_sum_of_fourth_powers_l938_93810


namespace NUMINAMATH_GPT_james_weekly_expenses_l938_93811

noncomputable def utility_cost (rent: ℝ):  ℝ := 0.2 * rent
noncomputable def weekly_hours_open (hours_per_day: ℕ) (days_per_week: ℕ): ℕ := hours_per_day * days_per_week
noncomputable def employee_weekly_wages (wage_per_hour: ℝ) (weekly_hours: ℕ): ℝ := wage_per_hour * weekly_hours
noncomputable def total_employee_wages (employees: ℕ) (weekly_wages: ℝ): ℝ := employees * weekly_wages
noncomputable def total_weekly_expenses (rent: ℝ) (utilities: ℝ) (employee_wages: ℝ): ℝ := rent + utilities + employee_wages

theorem james_weekly_expenses : 
  let rent := 1200
  let utility_percentage := 0.2
  let hours_per_day := 16
  let days_per_week := 5
  let employees := 2
  let wage_per_hour := 12.5
  let weekly_hours := weekly_hours_open hours_per_day days_per_week
  let utilities := utility_cost rent
  let employee_wages_per_week := employee_weekly_wages wage_per_hour weekly_hours
  let total_employee_wages_per_week := total_employee_wages employees employee_wages_per_week
  total_weekly_expenses rent utilities total_employee_wages_per_week = 3440 := 
by
  sorry

end NUMINAMATH_GPT_james_weekly_expenses_l938_93811


namespace NUMINAMATH_GPT_find_angle_A_l938_93867

theorem find_angle_A (A B C a b c : ℝ) 
  (h1 : 0 < A ∧ A < π)
  (h2 : 0 < B ∧ B < π)
  (h3 : 0 < C ∧ C < π)
  (h4 : a > 0)
  (h5 : b > 0)
  (h6 : c > 0)
  (sin_eq : Real.sin (C + π / 6) = b / (2 * a)) :
  A = π / 6 :=
sorry

end NUMINAMATH_GPT_find_angle_A_l938_93867


namespace NUMINAMATH_GPT_number_of_integer_pairs_l938_93822

theorem number_of_integer_pairs (n : ℕ) : 
  (∀ x y : ℤ, 5 * x^2 - 6 * x * y + y^2 = 6^100) → n = 19594 :=
sorry

end NUMINAMATH_GPT_number_of_integer_pairs_l938_93822


namespace NUMINAMATH_GPT_nonrational_ab_l938_93852

theorem nonrational_ab {a b : ℝ} (ha : a ≠ 0) (hb : b ≠ 0) 
    (h : a^2 * b^2 * (a^2 * b^2 + 4) = 2 * (a^6 + b^6)) : 
    ¬(∃ (p q r s : ℤ), q ≠ 0 ∧ s ≠ 0 ∧ a = p / q ∧ b = r / s) := by
  sorry

end NUMINAMATH_GPT_nonrational_ab_l938_93852


namespace NUMINAMATH_GPT_area_of_triangle_CDE_l938_93831

theorem area_of_triangle_CDE
  (DE : ℝ) (h : ℝ)
  (hDE : DE = 12) (hh : h = 15) :
  1/2 * DE * h = 90 := by
  sorry

end NUMINAMATH_GPT_area_of_triangle_CDE_l938_93831


namespace NUMINAMATH_GPT_concert_ticket_sales_l938_93887

theorem concert_ticket_sales (A C : ℕ) (total : ℕ) :
  (C = 3 * A) →
  (7 * A + 3 * C = 6000) →
  (total = A + C) →
  total = 1500 :=
by
  intros
  -- The proof is not required
  sorry

end NUMINAMATH_GPT_concert_ticket_sales_l938_93887


namespace NUMINAMATH_GPT_eccentricity_hyperbola_l938_93821

theorem eccentricity_hyperbola : 
  let a2 := 4
  let b2 := 5
  let e := Real.sqrt (1 + (b2 / a2))
  e = 3 / 2 := by
    apply sorry

end NUMINAMATH_GPT_eccentricity_hyperbola_l938_93821


namespace NUMINAMATH_GPT_total_games_l938_93868

theorem total_games (teams : ℕ) (games_per_pair : ℕ) (h_teams : teams = 12) (h_games_per_pair : games_per_pair = 4) : 
  (teams * (teams - 1) / 2) * games_per_pair = 264 :=
by
  sorry

end NUMINAMATH_GPT_total_games_l938_93868


namespace NUMINAMATH_GPT_trees_died_due_to_typhoon_l938_93820

-- defining the initial number of trees
def initial_trees : ℕ := 9

-- defining the additional trees grown after the typhoon
def additional_trees : ℕ := 5

-- defining the final number of trees after all events
def final_trees : ℕ := 10

-- we introduce D as the number of trees that died due to the typhoon
def trees_died (D : ℕ) : Prop := initial_trees - D + additional_trees = final_trees

-- the theorem we need to prove is that 4 trees died
theorem trees_died_due_to_typhoon : trees_died 4 :=
by
  sorry

end NUMINAMATH_GPT_trees_died_due_to_typhoon_l938_93820


namespace NUMINAMATH_GPT_parabola_unique_solution_l938_93882

theorem parabola_unique_solution (a : ℝ) :
  (∃ x : ℝ, (0 ≤ x^2 + a * x + 5) ∧ (x^2 + a * x + 5 ≤ 4)) → (a = 2 ∨ a = -2) :=
by
  sorry

end NUMINAMATH_GPT_parabola_unique_solution_l938_93882


namespace NUMINAMATH_GPT_marble_244_is_white_l938_93804

noncomputable def color_of_marble (n : ℕ) : String :=
  let cycle := ["white", "white", "white", "white", "gray", "gray", "gray", "gray", "gray", "black", "black", "black"]
  cycle.get! (n % 12)

theorem marble_244_is_white : color_of_marble 244 = "white" :=
by
  sorry

end NUMINAMATH_GPT_marble_244_is_white_l938_93804


namespace NUMINAMATH_GPT_quadratic_function_value_2_l938_93809

variables (a b : ℝ)
def f (x : ℝ) : ℝ := x^2 + a * x + b

theorem quadratic_function_value_2 :
  f a b 2 = 3 :=
by
  -- Definitions and assumptions to be used
  sorry

end NUMINAMATH_GPT_quadratic_function_value_2_l938_93809


namespace NUMINAMATH_GPT_beef_original_weight_l938_93885

theorem beef_original_weight (W : ℝ) (h : 0.65 * W = 546): W = 840 :=
sorry

end NUMINAMATH_GPT_beef_original_weight_l938_93885


namespace NUMINAMATH_GPT_quadratic_has_two_distinct_roots_l938_93883

noncomputable def discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

theorem quadratic_has_two_distinct_roots 
  (a b c : ℝ) (h : 2016 + a^2 + a * c < a * b) : 
  discriminant a b c > 0 :=
sorry

end NUMINAMATH_GPT_quadratic_has_two_distinct_roots_l938_93883


namespace NUMINAMATH_GPT_price_per_ticket_is_six_l938_93861

-- Definition of the conditions
def total_tickets (friends_tickets extra_tickets : ℕ) : ℕ :=
  friends_tickets + extra_tickets

def total_cost (tickets price_per_ticket : ℕ) : ℕ :=
  tickets * price_per_ticket

-- Given conditions
def friends_tickets : ℕ := 8
def extra_tickets : ℕ := 2
def total_spent : ℕ := 60

-- Formulate the problem to prove the price per ticket
theorem price_per_ticket_is_six :
  ∃ (price_per_ticket : ℕ), price_per_ticket = 6 ∧ 
  total_cost (total_tickets friends_tickets extra_tickets) price_per_ticket = total_spent :=
by
  -- The proof is not required; we assume its correctness here.
  sorry

end NUMINAMATH_GPT_price_per_ticket_is_six_l938_93861


namespace NUMINAMATH_GPT_div_by_20_l938_93877

theorem div_by_20 (n : ℕ) : 20 ∣ (9 ^ (8 * n + 4) - 7 ^ (8 * n + 4)) :=
  sorry

end NUMINAMATH_GPT_div_by_20_l938_93877


namespace NUMINAMATH_GPT_tom_bought_6_hardcover_l938_93856

-- Given conditions and statements
def toms_books_condition_1 (h p : ℕ) : Prop :=
  h + p = 10

def toms_books_condition_2 (h p : ℕ) : Prop :=
  28 * h + 18 * p = 240

-- The theorem to prove
theorem tom_bought_6_hardcover (h p : ℕ) 
  (h_condition : toms_books_condition_1 h p)
  (c_condition : toms_books_condition_2 h p) : 
  h = 6 :=
sorry

end NUMINAMATH_GPT_tom_bought_6_hardcover_l938_93856


namespace NUMINAMATH_GPT_find_k_l938_93879

noncomputable def g (a b c : ℤ) (x : ℤ) := a * x^2 + b * x + c

theorem find_k (a b c k : ℤ) 
  (h1 : g a b c (-1) = 0) 
  (h2 : 30 < g a b c 5) (h3 : g a b c 5 < 40)
  (h4 : 120 < g a b c 7) (h5 : g a b c 7 < 130)
  (h6 : 2000 * k < g a b c 50) (h7 : g a b c 50 < 2000 * (k + 1)) : 
  k = 5 := 
sorry

end NUMINAMATH_GPT_find_k_l938_93879


namespace NUMINAMATH_GPT_point_in_first_quadrant_l938_93878

-- Define the imaginary unit i
def i := Complex.I

-- Define the complex number z
def z := i * (2 - i)

-- Define a predicate that checks if a complex number is in the first quadrant
def isFirstQuadrant (x : ℂ) : Prop := x.re > 0 ∧ x.im > 0

-- State the theorem
theorem point_in_first_quadrant : isFirstQuadrant z := sorry

end NUMINAMATH_GPT_point_in_first_quadrant_l938_93878


namespace NUMINAMATH_GPT_probability_of_same_type_is_correct_l938_93815

noncomputable def total_socks : ℕ := 12 + 10 + 6
noncomputable def ways_to_pick_any_3_socks : ℕ := Nat.choose total_socks 3
noncomputable def ways_to_pick_3_black_socks : ℕ := Nat.choose 12 3
noncomputable def ways_to_pick_3_white_socks : ℕ := Nat.choose 10 3
noncomputable def ways_to_pick_3_striped_socks : ℕ := Nat.choose 6 3
noncomputable def ways_to_pick_3_same_type : ℕ := ways_to_pick_3_black_socks + ways_to_pick_3_white_socks + ways_to_pick_3_striped_socks
noncomputable def probability_same_type : ℚ := ways_to_pick_3_same_type / ways_to_pick_any_3_socks

theorem probability_of_same_type_is_correct :
  probability_same_type = 60 / 546 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_same_type_is_correct_l938_93815


namespace NUMINAMATH_GPT_solution_set_inequality_l938_93893

def custom_op (a b : ℝ) : ℝ := a * b + 2 * a + b

theorem solution_set_inequality : {x : ℝ | custom_op x (x - 2) < 0} = {x : ℝ | -2 < x ∧ x < 1} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_inequality_l938_93893


namespace NUMINAMATH_GPT_radius_of_circle_with_chords_l938_93894

theorem radius_of_circle_with_chords 
  (chord1_length : ℝ) (chord2_length : ℝ) (distance_between_midpoints : ℝ) 
  (h1 : chord1_length = 9) (h2 : chord2_length = 17) (h3 : distance_between_midpoints = 5) : 
  ∃ r : ℝ, r = 85 / 8 :=
by
  sorry

end NUMINAMATH_GPT_radius_of_circle_with_chords_l938_93894


namespace NUMINAMATH_GPT_symmetric_scanning_codes_count_l938_93859

-- Definition of a symmetric 8x8 scanning code grid under given conditions
def is_symmetric_code (grid : Fin 8 → Fin 8 → Bool) : Prop :=
  ∀ i j : Fin 8, grid i j = grid (7 - i) (7 - j) ∧ grid i j = grid j i

def at_least_one_each_color (grid : Fin 8 → Fin 8 → Bool) : Prop :=
  ∃ i j k l : Fin 8, grid i j = true ∧ grid k l = false

def total_symmetric_scanning_codes : Nat :=
  1022

theorem symmetric_scanning_codes_count :
  ∀ (grid : Fin 8 → Fin 8 → Bool), is_symmetric_code grid ∧ at_least_one_each_color grid → 
  1022 = total_symmetric_scanning_codes :=
by
  sorry

end NUMINAMATH_GPT_symmetric_scanning_codes_count_l938_93859


namespace NUMINAMATH_GPT_charlie_share_l938_93800

theorem charlie_share (A B C : ℕ) 
  (h1 : (A - 10) * 18 = (B - 20) * 11)
  (h2 : (A - 10) * 24 = (C - 15) * 11)
  (h3 : A + B + C = 1105) : 
  C = 495 := 
by
  sorry

end NUMINAMATH_GPT_charlie_share_l938_93800


namespace NUMINAMATH_GPT_largest_multiple_of_15_less_than_500_l938_93846

theorem largest_multiple_of_15_less_than_500 : ∃ (n : ℕ), 15 * n < 500 ∧ 15 * n ≥ 15 * (33 - 1) + 1 := by
  sorry

end NUMINAMATH_GPT_largest_multiple_of_15_less_than_500_l938_93846


namespace NUMINAMATH_GPT_claire_sleep_hours_l938_93880

def hours_in_day := 24
def cleaning_hours := 4
def cooking_hours := 2
def crafting_hours := 5
def tailoring_hours := crafting_hours

theorem claire_sleep_hours :
  hours_in_day - (cleaning_hours + cooking_hours + crafting_hours + tailoring_hours) = 8 := by
  sorry

end NUMINAMATH_GPT_claire_sleep_hours_l938_93880


namespace NUMINAMATH_GPT_trigonometric_identity_l938_93828

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 3) : 
  (Real.sin (Real.pi - α) - Real.sin (Real.pi / 2 + α)) / 
  (Real.cos (3 * Real.pi / 2 - α) + 2 * Real.cos (-Real.pi + α)) = -2 / 5 := 
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l938_93828


namespace NUMINAMATH_GPT_pump_fill_time_without_leak_l938_93849

variable (T : ℕ)

def rate_pump (T : ℕ) : ℚ := 1 / T
def rate_leak : ℚ := 1 / 20

theorem pump_fill_time_without_leak : rate_pump T - rate_leak = rate_leak → T = 10 := by 
  intro h
  sorry

end NUMINAMATH_GPT_pump_fill_time_without_leak_l938_93849


namespace NUMINAMATH_GPT_david_marks_in_english_l938_93826

theorem david_marks_in_english : 
  ∀ (E : ℕ), 
  let math_marks := 85 
  let physics_marks := 82 
  let chemistry_marks := 87 
  let biology_marks := 85 
  let avg_marks := 85 
  let total_subjects := 5 
  let total_marks := avg_marks * total_subjects 
  let total_known_subject_marks := math_marks + physics_marks + chemistry_marks + biology_marks 
  total_marks = total_known_subject_marks + E → 
  E = 86 :=
by 
  intros
  sorry

end NUMINAMATH_GPT_david_marks_in_english_l938_93826


namespace NUMINAMATH_GPT_sugar_ratio_l938_93876

theorem sugar_ratio (total_sugar : ℕ)  (bags : ℕ) (remaining_sugar : ℕ) (sugar_each_bag : ℕ) (sugar_fell : ℕ)
  (h1 : total_sugar = 24) (h2 : bags = 4) (h3 : total_sugar - remaining_sugar = sugar_fell) 
  (h4 : total_sugar / bags = sugar_each_bag) (h5 : remaining_sugar = 21) : 
  2 * sugar_fell = sugar_each_bag := by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_sugar_ratio_l938_93876


namespace NUMINAMATH_GPT_parallel_lines_slope_l938_93823

theorem parallel_lines_slope (k : ℝ) :
  (∀ x : ℝ, 5 * x - 3 = (3 * k) * x + 7 -> ((3 * k) = 5)) -> (k = 5 / 3) :=
by
  -- Posing the conditions on parallel lines
  intro h_eq_slopes
  -- We know 3k = 5, hence k = 5 / 3
  have slope_eq : 3 * k = 5 := by sorry
  -- Therefore k = 5 / 3 follows from the fact 3k = 5
  have k_val : k = 5 / 3 := by sorry
  exact k_val

end NUMINAMATH_GPT_parallel_lines_slope_l938_93823


namespace NUMINAMATH_GPT_factorize_expr_l938_93801

theorem factorize_expr (x : ℝ) : 2 * x^2 - 8 = 2 * (x + 2) * (x - 2) :=
sorry

end NUMINAMATH_GPT_factorize_expr_l938_93801


namespace NUMINAMATH_GPT_regular_polygon_sides_l938_93872

theorem regular_polygon_sides (n : ℕ) (h : 2 ≤ n) (h_angle : 180 * (n - 2) / n = 150) : n = 12 :=
sorry

end NUMINAMATH_GPT_regular_polygon_sides_l938_93872


namespace NUMINAMATH_GPT_gcd_35_x_eq_7_in_range_80_90_l938_93841

theorem gcd_35_x_eq_7_in_range_80_90 {n : ℕ} (h₁ : Nat.gcd 35 n = 7) (h₂ : 80 < n) (h₃ : n < 90) : n = 84 :=
by
  sorry

end NUMINAMATH_GPT_gcd_35_x_eq_7_in_range_80_90_l938_93841


namespace NUMINAMATH_GPT_five_digit_odd_and_multiples_of_5_sum_l938_93896

theorem five_digit_odd_and_multiples_of_5_sum :
  let A := 9 * 10^3 * 5
  let B := 9 * 10^3 * 1
  A + B = 45000 := by
sorry

end NUMINAMATH_GPT_five_digit_odd_and_multiples_of_5_sum_l938_93896


namespace NUMINAMATH_GPT_sin_alpha_through_point_l938_93898

theorem sin_alpha_through_point (α : ℝ) (P : ℝ × ℝ) (hP : P = (-3, -Real.sqrt 3)) :
    Real.sin α = -1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_sin_alpha_through_point_l938_93898


namespace NUMINAMATH_GPT_max_percent_liquid_X_l938_93842

theorem max_percent_liquid_X (wA wB wC : ℝ) (XA XB XC YA YB YC : ℝ)
  (hXA : XA = 0.8 / 100) (hXB : XB = 1.8 / 100) (hXC : XC = 3.0 / 100)
  (hYA : YA = 2.0 / 100) (hYB : YB = 1.0 / 100) (hYC : YC = 0.5 / 100)
  (hwA : wA = 500) (hwB : wB = 700) (hwC : wC = 300)
  (H_combined_limit : XA * wA + XB * wB + XC * wC + YA * wA + YB * wB + YC * wC ≤ 0.025 * (wA + wB + wC)) :
  XA * wA + XB * wB + XC * wC ≤ 0.0171 * (wA + wB + wC) :=
sorry

end NUMINAMATH_GPT_max_percent_liquid_X_l938_93842


namespace NUMINAMATH_GPT_distance_from_point_to_plane_l938_93845

-- Definitions representing the conditions
def side_length_base := 6
def base_area := side_length_base * side_length_base
def volume_pyramid := 96

-- Proof statement
theorem distance_from_point_to_plane (h : ℝ) : 
  (1/3) * base_area * h = volume_pyramid → h = 8 := 
by 
  sorry

end NUMINAMATH_GPT_distance_from_point_to_plane_l938_93845


namespace NUMINAMATH_GPT_ratio_of_sum_of_terms_l938_93802

theorem ratio_of_sum_of_terms (S : ℕ → ℝ) (a : ℕ → ℝ) (h1 : ∀ n, S n = n * (a 1 + a n) / 2)
  (h2 : a 5 / a 3 = 5 / 9) : S 9 / S 5 = 1 := 
  sorry

end NUMINAMATH_GPT_ratio_of_sum_of_terms_l938_93802


namespace NUMINAMATH_GPT_opposite_sign_pairs_l938_93812

def opposite_sign (a b : ℤ) : Prop := (a < 0 ∧ b > 0) ∨ (a > 0 ∧ b < 0)

theorem opposite_sign_pairs :
  ¬opposite_sign (-(-1)) 1 ∧
  ¬opposite_sign ((-1)^2) 1 ∧
  ¬opposite_sign (|(-1)|) 1 ∧
  opposite_sign (-1) 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_opposite_sign_pairs_l938_93812


namespace NUMINAMATH_GPT_find_abc_l938_93860

open Real

theorem find_abc 
  (a b c : ℝ) 
  (h_pos_a : 0 < a) 
  (h_pos_b : 0 < b) 
  (h_pos_c : 0 < c) 
  (h1 : a * (b + c) = 154)
  (h2 : b * (c + a) = 164) 
  (h3 : c * (a + b) = 172) : 
  (a * b * c = Real.sqrt 538083) := 
by 
  sorry

end NUMINAMATH_GPT_find_abc_l938_93860


namespace NUMINAMATH_GPT_num_students_third_class_num_students_second_class_l938_93819

-- Definition of conditions for both problems
def class_student_bounds (n : ℕ) : Prop := 40 < n ∧ n ≤ 50
def option_one_cost (n : ℕ) : ℕ := 40 * n * 7 / 10
def option_two_cost (n : ℕ) : ℕ := 40 * (n - 6) * 8 / 10

-- Problem Part 1
theorem num_students_third_class (x : ℕ) (h1 : class_student_bounds x) (h2 : option_one_cost x = option_two_cost x) : x = 48 := 
sorry

-- Problem Part 2
theorem num_students_second_class (y : ℕ) (h1 : class_student_bounds y) (h2 : option_one_cost y < option_two_cost y) : y = 49 ∨ y = 50 := 
sorry

end NUMINAMATH_GPT_num_students_third_class_num_students_second_class_l938_93819


namespace NUMINAMATH_GPT_number_of_customers_trimmed_l938_93843

-- Definitions based on the conditions
def total_sounds : ℕ := 60
def sounds_per_person : ℕ := 20

-- Statement to prove
theorem number_of_customers_trimmed :
  ∃ n : ℕ, n * sounds_per_person = total_sounds ∧ n = 3 :=
sorry

end NUMINAMATH_GPT_number_of_customers_trimmed_l938_93843


namespace NUMINAMATH_GPT_radius_of_circle_is_ten_l938_93829

noncomputable def radius_of_circle (diameter : ℝ) : ℝ :=
  diameter / 2

theorem radius_of_circle_is_ten :
  radius_of_circle 20 = 10 :=
by
  unfold radius_of_circle
  sorry

end NUMINAMATH_GPT_radius_of_circle_is_ten_l938_93829


namespace NUMINAMATH_GPT_find_abcdef_l938_93881

def repeating_decimal_to_fraction_abcd (a b c d : ℕ) : ℚ :=
  (1000 * a + 100 * b + 10 * c + d) / 9999

def repeating_decimal_to_fraction_abcdef (a b c d e f : ℕ) : ℚ :=
  (100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f) / 999999

theorem find_abcdef :
  ∀ a b c d e f : ℕ,
  0 ≤ a ∧ a ≤ 9 ∧
  0 ≤ b ∧ b ≤ 9 ∧
  0 ≤ c ∧ c ≤ 9 ∧
  0 ≤ d ∧ d ≤ 9 ∧
  0 ≤ e ∧ e ≤ 9 ∧
  0 ≤ f ∧ f ≤ 9 ∧
  (repeating_decimal_to_fraction_abcd a b c d + repeating_decimal_to_fraction_abcdef a b c d e f = 49 / 999) →
  (100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f = 490) :=
by
  repeat {sorry}

end NUMINAMATH_GPT_find_abcdef_l938_93881


namespace NUMINAMATH_GPT_cheryl_needed_first_material_l938_93834

noncomputable def cheryl_material (x : ℚ) : ℚ :=
  x + 1 / 3 - 3 / 8

theorem cheryl_needed_first_material
  (h_total_used : 0.33333333333333326 = 1 / 3) :
  cheryl_material x = 1 / 3 → x = 3 / 8 :=
by
  intros
  rw [h_total_used] at *
  sorry

end NUMINAMATH_GPT_cheryl_needed_first_material_l938_93834


namespace NUMINAMATH_GPT_Jane_age_proof_l938_93807

theorem Jane_age_proof (D J : ℕ) (h1 : D + 6 = (J + 6) / 2) (h2 : D + 14 = 25) : J = 28 :=
by
  sorry

end NUMINAMATH_GPT_Jane_age_proof_l938_93807
