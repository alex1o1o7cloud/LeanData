import Mathlib

namespace NUMINAMATH_GPT_question1_question2_l6_691

variables (θ : ℝ)

-- Condition: tan θ = 2
def tan_theta_eq : Prop := Real.tan θ = 2

-- Question 1: Prove (4 * sin θ - 2 * cos θ) / (3 * sin θ + 5 * cos θ) = 6 / 11
theorem question1 (h : tan_theta_eq θ) : (4 * Real.sin θ - 2 * Real.cos θ) / (3 * Real.sin θ + 5 * Real.cos θ) = 6 / 11 :=
by
  sorry

-- Question 2: Prove 1 - 4 * sin θ * cos θ + 2 * cos² θ = -1 / 5
theorem question2 (h : tan_theta_eq θ) : 1 - 4 * Real.sin θ * Real.cos θ + 2 * (Real.cos θ)^2 = -1 / 5 :=
by
  sorry

end NUMINAMATH_GPT_question1_question2_l6_691


namespace NUMINAMATH_GPT_rectangle_perimeter_is_104_l6_692

noncomputable def perimeter_of_rectangle (b : ℝ) (h1 : b > 0) (h2 : 3 * b * b = 507) : ℝ :=
  2 * (3 * b) + 2 * b

theorem rectangle_perimeter_is_104 {b : ℝ} (h1 : b > 0) (h2 : 3 * b * b = 507) :
  perimeter_of_rectangle b h1 h2 = 104 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_perimeter_is_104_l6_692


namespace NUMINAMATH_GPT_diff_of_two_numbers_l6_610

theorem diff_of_two_numbers :
  ∃ D S : ℕ, (1650 = 5 * S + 5) ∧ (D = 1650 - S) ∧ (D = 1321) :=
sorry

end NUMINAMATH_GPT_diff_of_two_numbers_l6_610


namespace NUMINAMATH_GPT_washing_whiteboards_l6_672

/-- Define the conditions from the problem:
1. Four kids can wash three whiteboards in 20 minutes.
2. It takes one kid 160 minutes to wash a certain number of whiteboards. -/
def four_kids_wash_in_20_min : ℕ := 3
def time_per_batch : ℕ := 20
def one_kid_time : ℕ := 160
def intervals : ℕ := one_kid_time / time_per_batch

/-- Proving the answer based on the conditions:
one kid can wash six whiteboards in 160 minutes given these conditions. -/
theorem washing_whiteboards : intervals * (four_kids_wash_in_20_min / 4) = 6 :=
by
  sorry

end NUMINAMATH_GPT_washing_whiteboards_l6_672


namespace NUMINAMATH_GPT_range_of_m_n_l6_637

noncomputable def tangent_condition (m n : ℝ) : Prop :=
  ∀ x y : ℝ, (x - 1)^2 + (y - 1)^2 = 1 → (m + 1) * x + (n + 1) * y - 2 = 0

theorem range_of_m_n (m n : ℝ) :
  tangent_condition m n →
  (m + n ≤ 2 - 2 * Real.sqrt 2 ∨ m + n ≥ 2 + 2 * Real.sqrt 2) :=
sorry

end NUMINAMATH_GPT_range_of_m_n_l6_637


namespace NUMINAMATH_GPT_imaginary_part_of_z_is_1_l6_605

def z := Complex.ofReal 0 + Complex.ofReal 1 * Complex.I * (Complex.ofReal 1 + Complex.ofReal 2 * Complex.I)
theorem imaginary_part_of_z_is_1 : z.im = 1 := by
  sorry

end NUMINAMATH_GPT_imaginary_part_of_z_is_1_l6_605


namespace NUMINAMATH_GPT_find_X_l6_698

theorem find_X (X : ℝ) (h : 45 * 8 = 0.40 * X) : X = 900 :=
sorry

end NUMINAMATH_GPT_find_X_l6_698


namespace NUMINAMATH_GPT_max_product_l6_669

def geometric_sequence (a1 q : ℝ) (n : ℕ) :=
  a1 * q ^ (n - 1)

def product_of_terms (a1 q : ℝ) (n : ℕ) :=
  (List.range n).foldr (λ i acc => acc * geometric_sequence a1 q (i + 1)) 1

theorem max_product (n : ℕ) (a1 q : ℝ) (h₁ : a1 = 1536) (h₂ : q = -1/2) :
  n = 11 ↔ ∀ m : ℕ, m ≤ 11 → product_of_terms a1 q m ≤ product_of_terms a1 q 11 :=
by
  sorry

end NUMINAMATH_GPT_max_product_l6_669


namespace NUMINAMATH_GPT_max_min_values_l6_687

noncomputable def y (x : ℝ) : ℝ :=
  3 - 4 * Real.sin x - 4 * (Real.cos x)^2

theorem max_min_values :
  (∀ k : ℤ, y (- (Real.pi/2) + 2 * k * Real.pi) = 7) ∧
  (∀ k : ℤ, y (Real.pi/6 + 2 * k * Real.pi) = -2) ∧
  (∀ k : ℤ, y (5 * Real.pi/6 + 2 * k * Real.pi) = -2) := by
  sorry

end NUMINAMATH_GPT_max_min_values_l6_687


namespace NUMINAMATH_GPT_circles_ordering_l6_674

theorem circles_ordering :
  let rA := 3
  let AB := 12 * Real.pi
  let AC := 28 * Real.pi
  let rB := Real.sqrt 12
  let rC := Real.sqrt 28
  (rA < rB) ∧ (rB < rC) :=
by
  let rA := 3
  let AB := 12 * Real.pi
  let AC := 28 * Real.pi
  let rB := Real.sqrt 12
  let rC := Real.sqrt 28
  have rA_lt_rB: rA < rB := by sorry
  have rB_lt_rC: rB < rC := by sorry
  exact ⟨rA_lt_rB, rB_lt_rC⟩

end NUMINAMATH_GPT_circles_ordering_l6_674


namespace NUMINAMATH_GPT_probability_white_given_popped_l6_684

theorem probability_white_given_popped :
  let P_white := 3 / 5
  let P_yellow := 2 / 5
  let P_popped_given_white := 2 / 5
  let P_popped_given_yellow := 4 / 5
  let P_white_and_popped := P_white * P_popped_given_white
  let P_yellow_and_popped := P_yellow * P_popped_given_yellow
  let P_popped := P_white_and_popped + P_yellow_and_popped
  let P_white_given_popped := P_white_and_popped / P_popped
  P_white_given_popped = 3 / 7 :=
by sorry

end NUMINAMATH_GPT_probability_white_given_popped_l6_684


namespace NUMINAMATH_GPT_golden_section_AP_l6_623

-- Definitions of the golden ratio and its reciprocal
noncomputable def phi := (1 + Real.sqrt 5) / 2
noncomputable def phi_inv := (Real.sqrt 5 - 1) / 2

-- Conditions of the problem
def isGoldenSectionPoint (A B P : ℝ) := ∃ AP BP AB, AP < BP ∧ BP = 10 ∧ P = AB ∧ AP = BP * phi_inv

theorem golden_section_AP (A B P : ℝ) (h1 : isGoldenSectionPoint A B P) : 
  ∃ AP, AP = 5 * Real.sqrt 5 - 5 :=
by
  sorry

end NUMINAMATH_GPT_golden_section_AP_l6_623


namespace NUMINAMATH_GPT_percentage_increase_l6_664

theorem percentage_increase (S P : ℝ) (h1 : (S * (1 + P / 100)) * 0.8 = 1.04 * S) : P = 30 :=
by 
  sorry

end NUMINAMATH_GPT_percentage_increase_l6_664


namespace NUMINAMATH_GPT_product_mod_23_l6_652

theorem product_mod_23 :
  (2003 * 2004 * 2005 * 2006 * 2007 * 2008) % 23 = 3 :=
by 
  sorry

end NUMINAMATH_GPT_product_mod_23_l6_652


namespace NUMINAMATH_GPT_find_xy_l6_626

-- Define the conditions as constants for clarity
def condition1 (x : ℝ) : Prop := 0.60 / x = 6 / 2
def condition2 (x y : ℝ) : Prop := x / y = 8 / 12

theorem find_xy (x y : ℝ) (hx : condition1 x) (hy : condition2 x y) : 
  x = 0.20 ∧ y = 0.30 :=
by
  sorry

end NUMINAMATH_GPT_find_xy_l6_626


namespace NUMINAMATH_GPT_intersection_of_complements_l6_630

-- Define the universal set U as a natural set with numbers <= 8
def U : Set ℕ := { x | x ≤ 8 }

-- Define the set A
def A : Set ℕ := { 1, 3, 7 }

-- Define the set B
def B : Set ℕ := { 2, 3, 8 }

-- Prove the statement for the intersection of the complements of A and B with respect to U
theorem intersection_of_complements : 
  ((U \ A) ∩ (U \ B)) = ({ 0, 4, 5, 6 } : Set ℕ) :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_complements_l6_630


namespace NUMINAMATH_GPT_parabola_focus_l6_640

theorem parabola_focus (p : ℝ) (hp : ∃ (p : ℝ), ∀ x y : ℝ, x^2 = 2 * p * y) : (∀ (hf : (0, 2) = (0, p / 2)), p = 4) :=
sorry

end NUMINAMATH_GPT_parabola_focus_l6_640


namespace NUMINAMATH_GPT_negation_proposition_l6_653

-- Define the original proposition
def unique_solution (a b : ℝ) (h : a ≠ 0) : Prop :=
  ∀ x1 x2 : ℝ, (a * x1 = b ∧ a * x2 = b) → (x1 = x2)

-- Define the negation of the proposition
def negation_unique_solution (a b : ℝ) (h : a ≠ 0) : Prop :=
  ¬ unique_solution a b h

-- Define a proposition for "no unique solution"
def no_unique_solution (a b : ℝ) (h : a ≠ 0) : Prop :=
  ∃ x1 x2 : ℝ, (a * x1 = b ∧ a * x2 = b) ∧ (x1 ≠ x2)

-- The Lean 4 statement
theorem negation_proposition (a b : ℝ) (h : a ≠ 0) :
  negation_unique_solution a b h :=
sorry

end NUMINAMATH_GPT_negation_proposition_l6_653


namespace NUMINAMATH_GPT_how_many_eyes_do_I_see_l6_688

def boys : ℕ := 23
def eyes_per_boy : ℕ := 2
def total_eyes : ℕ := boys * eyes_per_boy

theorem how_many_eyes_do_I_see : total_eyes = 46 := by
  sorry

end NUMINAMATH_GPT_how_many_eyes_do_I_see_l6_688


namespace NUMINAMATH_GPT_bob_total_spend_in_usd_l6_660

theorem bob_total_spend_in_usd:
  let coffee_cost_yen := 250
  let sandwich_cost_yen := 150
  let yen_to_usd := 110
  (coffee_cost_yen + sandwich_cost_yen) / yen_to_usd = 3.64 := by
  sorry

end NUMINAMATH_GPT_bob_total_spend_in_usd_l6_660


namespace NUMINAMATH_GPT_total_complaints_l6_602

-- Conditions as Lean definitions
def normal_complaints : ℕ := 120
def short_staffed_20 (c : ℕ) := c + c / 3
def short_staffed_40 (c : ℕ) := c + 2 * c / 3
def self_checkout_partial (c : ℕ) := c + c / 10
def self_checkout_complete (c : ℕ) := c + c / 5
def day1_complaints : ℕ := normal_complaints + normal_complaints / 3 + normal_complaints / 5
def day2_complaints : ℕ := normal_complaints + 2 * normal_complaints / 3 + normal_complaints / 10
def day3_complaints : ℕ := normal_complaints + 2 * normal_complaints / 3 + normal_complaints / 5

-- Prove the total complaints
theorem total_complaints : day1_complaints + day2_complaints + day3_complaints = 620 :=
by
  sorry

end NUMINAMATH_GPT_total_complaints_l6_602


namespace NUMINAMATH_GPT_investment_interests_l6_685

theorem investment_interests (x y : ℝ) (h₁ : x + y = 24000)
  (h₂ : 0.045 * x + 0.06 * y = 0.05 * 24000) : (x = 16000) ∧ (y = 8000) :=
  by
  sorry

end NUMINAMATH_GPT_investment_interests_l6_685


namespace NUMINAMATH_GPT_either_x_or_y_is_even_l6_650

theorem either_x_or_y_is_even (x y z : ℤ) (h : x^2 + y^2 = z^2) : (2 ∣ x) ∨ (2 ∣ y) :=
by
  sorry

end NUMINAMATH_GPT_either_x_or_y_is_even_l6_650


namespace NUMINAMATH_GPT_area_of_triangle_abe_l6_686

noncomputable def triangle_area (a b c : ℝ) : ℝ :=
10 -- Dummy definition, in actual scenario appropriate area calculation will be required.

def length_AD : ℝ := 2
def length_BD : ℝ := 3

def areas_equal (S_ABE S_DBFE : ℝ) : Prop :=
    S_ABE = S_DBFE

theorem area_of_triangle_abe
  (area_abc : ℝ)
  (length_ad length_bd : ℝ)
  (equal_areas : areas_equal (triangle_area 1 1 1) 1) -- Dummy values, should be substituted with correct arguments
  : triangle_area 1 1 1 = 6 :=
sorry -- proof will be filled later

end NUMINAMATH_GPT_area_of_triangle_abe_l6_686


namespace NUMINAMATH_GPT_fraction_sum_l6_629

theorem fraction_sum : (1 / 3 : ℚ) + (5 / 9 : ℚ) = (8 / 9 : ℚ) :=
by
  sorry

end NUMINAMATH_GPT_fraction_sum_l6_629


namespace NUMINAMATH_GPT_acute_angle_of_rhombus_l6_659

theorem acute_angle_of_rhombus (a α : ℝ) (V1 V2 : ℝ) (OA BD AN AB : ℝ) 
  (h_volumes : V1 / V2 = 1 / (2 * Real.sqrt 5)) 
  (h_V1 : V1 = (1 / 3) * Real.pi * (OA^2) * BD)
  (h_V2 : V2 = Real.pi * (AN^2) * AB)
  (h_OA : OA = a * Real.sin (α / 2))
  (h_BD : BD = 2 * a * Real.cos (α / 2))
  (h_AN : AN = a * Real.sin α)
  (h_AB : AB = a)
  : α = Real.arccos (1 / 9) :=
sorry

end NUMINAMATH_GPT_acute_angle_of_rhombus_l6_659


namespace NUMINAMATH_GPT_problem_1_problem_2_l6_682

-- Proof Problem 1: Prove A ∩ B = {x | -3 ≤ x ≤ -2} given m = -3
def A : Set ℝ := {x | -3 ≤ x ∧ x ≤ 4}
def B (m : ℝ) : Set ℝ := {x | 2 * m - 1 ≤ x ∧ x ≤ m + 1}

theorem problem_1 : B (-3) ∩ A = {x | -3 ≤ x ∧ x ≤ -2} := sorry

-- Proof Problem 2: Prove m ≥ -1 given B ⊆ A
theorem problem_2 (m : ℝ) : (B m ⊆ A) → m ≥ -1 := sorry

end NUMINAMATH_GPT_problem_1_problem_2_l6_682


namespace NUMINAMATH_GPT_solve_abs_eq_2x_plus_1_l6_662

theorem solve_abs_eq_2x_plus_1 (x : ℝ) (h : |x| = 2 * x + 1) : x = -1 / 3 :=
by 
  sorry

end NUMINAMATH_GPT_solve_abs_eq_2x_plus_1_l6_662


namespace NUMINAMATH_GPT_find_parameter_a_exactly_two_solutions_l6_648

noncomputable def system_has_two_solutions (a : ℝ) : Prop :=
∃ (x y : ℝ), |y - 3 - x| + |y - 3 + x| = 6 ∧ (|x| - 4)^2 + (|y| - 3)^2 = a

theorem find_parameter_a_exactly_two_solutions :
  {a : ℝ | system_has_two_solutions a} = {1, 25} :=
by
  sorry

end NUMINAMATH_GPT_find_parameter_a_exactly_two_solutions_l6_648


namespace NUMINAMATH_GPT_john_sleep_total_hours_l6_680

-- Defining the conditions provided in the problem statement
def days_with_3_hours : ℕ := 2
def sleep_per_day_3_hours : ℕ := 3
def remaining_days : ℕ := 7 - days_with_3_hours
def recommended_sleep : ℕ := 8
def percentage_sleep : ℝ := 0.6

-- Expressing the proof problem statement
theorem john_sleep_total_hours :
  (days_with_3_hours * sleep_per_day_3_hours
  + remaining_days * (percentage_sleep * recommended_sleep)) = 30 := by
  sorry

end NUMINAMATH_GPT_john_sleep_total_hours_l6_680


namespace NUMINAMATH_GPT_bounces_to_below_30_cm_l6_699

theorem bounces_to_below_30_cm :
  ∃ (b : ℕ), (256 * (3 / 4)^b < 30) ∧
            (∀ (k : ℕ), k < b -> 256 * (3 / 4)^k ≥ 30) :=
by 
  sorry

end NUMINAMATH_GPT_bounces_to_below_30_cm_l6_699


namespace NUMINAMATH_GPT_asep_wins_in_at_most_n_minus_5_div_4_steps_l6_670

theorem asep_wins_in_at_most_n_minus_5_div_4_steps (n : ℕ) (h : n ≥ 14) : 
  ∃ f : ℕ → ℕ, (∀ X d : ℕ, 0 < d → d ∣ X → (X' = X + d ∨ X' = X - d) → (f X' ≤ f X + 1)) ∧ f n ≤ (n - 5) / 4 := 
sorry

end NUMINAMATH_GPT_asep_wins_in_at_most_n_minus_5_div_4_steps_l6_670


namespace NUMINAMATH_GPT_circle_radius_range_l6_622

theorem circle_radius_range (r : ℝ) : 
  (∃ P₁ P₂ : ℝ × ℝ, (P₁.2 = 1 ∨ P₁.2 = -1) ∧ (P₂.2 = 1 ∨ P₂.2 = -1) ∧ 
  (P₁.1 - 3) ^ 2 + (P₁.2 + 5) ^ 2 = r^2 ∧ (P₂.1 - 3) ^ 2 + (P₂.2 + 5) ^ 2 = r^2) → (4 < r ∧ r < 6) :=
by
  sorry

end NUMINAMATH_GPT_circle_radius_range_l6_622


namespace NUMINAMATH_GPT_stratified_sampling_correct_l6_636

-- Definitions for the conditions
def total_employees : ℕ := 750
def young_employees : ℕ := 350
def middle_aged_employees : ℕ := 250
def elderly_employees : ℕ := 150
def sample_size : ℕ := 15
def sampling_proportion : ℚ := sample_size / total_employees

-- Statement to prove
theorem stratified_sampling_correct :
  (young_employees * sampling_proportion = 7) ∧
  (middle_aged_employees * sampling_proportion = 5) ∧
  (elderly_employees * sampling_proportion = 3) :=
by
  sorry

end NUMINAMATH_GPT_stratified_sampling_correct_l6_636


namespace NUMINAMATH_GPT_no_integer_n_squared_plus_one_div_by_seven_l6_678

theorem no_integer_n_squared_plus_one_div_by_seven (n : ℤ) : ¬ (n^2 + 1) % 7 = 0 := 
sorry

end NUMINAMATH_GPT_no_integer_n_squared_plus_one_div_by_seven_l6_678


namespace NUMINAMATH_GPT_div_by_9_implies_not_div_by_9_l6_625

/-- If 9 divides 10^n + 1, then it also divides 10^(n+1) + 1 -/
theorem div_by_9_implies:
  ∀ n: ℕ, (9 ∣ (10^n + 1)) → (9 ∣ (10^(n + 1) + 1)) :=
by
  intro n
  intro h
  sorry

/-- 9 does not divide 10^1 + 1 -/
theorem not_div_by_9:
  ¬(9 ∣ (10^1 + 1)) :=
by 
  sorry

end NUMINAMATH_GPT_div_by_9_implies_not_div_by_9_l6_625


namespace NUMINAMATH_GPT_incorrect_statement_B_l6_639

def two_times_root_equation (a b c x1 x2 : ℝ) : Prop :=
  a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0 ∧ (x1 = 2 * x2 ∨ x2 = 2 * x1)

theorem incorrect_statement_B (m n : ℝ) (h : (x - 2) * (m * x + n) = 0) :
  ¬(two_times_root_equation 1 (-m+n) (-mn) 2 (-n / m) -> m + n = 0) :=
sorry

end NUMINAMATH_GPT_incorrect_statement_B_l6_639


namespace NUMINAMATH_GPT_matrix_eq_value_satisfied_for_two_values_l6_612

variable (a b c d x : ℝ)

def matrix_value (a b c d : ℝ) : ℝ := a * b - c * d

-- Define the specific instance for the given matrix problem
def matrix_eq_value (x : ℝ) : Prop :=
  matrix_value (2 * x) x 1 x = 3

-- Prove that the equation is satisfied for exactly two values of x
theorem matrix_eq_value_satisfied_for_two_values :
  (∃! (x : ℝ), matrix_value (2 * x) x 1 x = 3) :=
sorry

end NUMINAMATH_GPT_matrix_eq_value_satisfied_for_two_values_l6_612


namespace NUMINAMATH_GPT_roots_numerically_equal_but_opposite_signs_l6_614

noncomputable def value_of_m (a b c : ℝ) : ℝ := (a - b) / (a + b)

theorem roots_numerically_equal_but_opposite_signs
  (a b c m : ℝ)
  (h : ∀ x : ℝ, (a ≠ 0 ∧ a + b ≠ 0) ∧ (x^2 - b*x = (ax - c) * (m - 1) / (m + 1))) 
  (root_condition : ∃ x₁ x₂ : ℝ, x₁ = -x₂ ∧ x₁ * x₂ != 0) :
  m = value_of_m a b c :=
by
  sorry

end NUMINAMATH_GPT_roots_numerically_equal_but_opposite_signs_l6_614


namespace NUMINAMATH_GPT_frisbee_total_distance_l6_671

-- Definitions for the conditions
def bess_initial_distance : ℝ := 20
def bess_throws : ℕ := 4
def bess_reduction : ℝ := 0.90
def holly_initial_distance : ℝ := 8
def holly_throws : ℕ := 5
def holly_reduction : ℝ := 0.95

-- Function to calculate the total distance for Bess
def total_distance_bess : ℝ :=
  let distances := List.range bess_throws |>.map (λ i => bess_initial_distance * bess_reduction ^ i)
  (distances.sum) * 2

-- Function to calculate the total distance for Holly
def total_distance_holly : ℝ :=
  let distances := List.range holly_throws |>.map (λ i => holly_initial_distance * holly_reduction ^ i)
  distances.sum

-- Proof statement
theorem frisbee_total_distance : 
  total_distance_bess + total_distance_holly = 173.76 :=
by
  sorry

end NUMINAMATH_GPT_frisbee_total_distance_l6_671


namespace NUMINAMATH_GPT_vehicles_with_at_least_80_kmh_equal_50_l6_643

variable (num_vehicles_80_to_89 : ℕ := 15)
variable (num_vehicles_90_to_99 : ℕ := 30)
variable (num_vehicles_100_to_109 : ℕ := 5)

theorem vehicles_with_at_least_80_kmh_equal_50 :
  num_vehicles_80_to_89 + num_vehicles_90_to_99 + num_vehicles_100_to_109 = 50 := by
  sorry

end NUMINAMATH_GPT_vehicles_with_at_least_80_kmh_equal_50_l6_643


namespace NUMINAMATH_GPT_roots_of_polynomial_fraction_l6_603

theorem roots_of_polynomial_fraction (a b c : ℝ)
  (h1 : a + b + c = 6)
  (h2 : a * b + a * c + b * c = 11)
  (h3 : a * b * c = 6) :
  a / (b * c + 2) + b / (a * c + 2) + c / (a * b + 2) = 3 / 2 := 
by
  sorry

end NUMINAMATH_GPT_roots_of_polynomial_fraction_l6_603


namespace NUMINAMATH_GPT_a8_div_b8_l6_696

variable (a b : ℕ → ℕ)
variable (S T : ℕ → ℕ)

-- Given Conditions
axiom sum_a (n : ℕ) : S n = (n * (a 1 + (n - 1) * a 2)) / 2 -- Sum of first n terms of arithmetic sequence a_n
axiom sum_b (n : ℕ) : T n = (n * (b 1 + (n - 1) * b 2)) / 2 -- Sum of first n terms of arithmetic sequence b_n
axiom ratio (n : ℕ) : S n / T n = (7 * n + 3) / (n + 3)

-- Proof statement
theorem a8_div_b8 : a 8 / b 8 = 6 := by
  sorry

end NUMINAMATH_GPT_a8_div_b8_l6_696


namespace NUMINAMATH_GPT_total_amount_l6_673

theorem total_amount
  (x y z : ℝ)
  (hy : y = 0.45 * x)
  (hz : z = 0.50 * x)
  (y_share : y = 27) :
  x + y + z = 117 :=
by
  sorry

end NUMINAMATH_GPT_total_amount_l6_673


namespace NUMINAMATH_GPT_return_speed_is_33_33_l6_621

noncomputable def return_speed (d: ℝ) (speed_to_b: ℝ) (avg_speed: ℝ): ℝ :=
  d / (3 + (d / avg_speed))

-- Conditions
def distance := 150
def speed_to_b := 50
def avg_speed := 40

-- Prove that the return speed is 33.33 miles per hour
theorem return_speed_is_33_33:
  return_speed distance speed_to_b avg_speed = 33.33 :=
by
  unfold return_speed
  sorry

end NUMINAMATH_GPT_return_speed_is_33_33_l6_621


namespace NUMINAMATH_GPT_combined_list_correct_l6_616

def james_friends : ℕ := 75
def john_friends : ℕ := 3 * james_friends
def shared_friends : ℕ := 25

def combined_list : ℕ :=
  james_friends + john_friends - shared_friends

theorem combined_list_correct : combined_list = 275 := by
  unfold combined_list
  unfold james_friends
  unfold john_friends
  unfold shared_friends
  sorry

end NUMINAMATH_GPT_combined_list_correct_l6_616


namespace NUMINAMATH_GPT_power_of_three_divides_an_l6_631

theorem power_of_three_divides_an (a : ℕ → ℕ) (k : ℕ) (h1 : a 1 = 3)
  (h2 : ∀ n, a (n + 1) = ((3 * (a n)^2 + 1) / 2) - a n)
  (h3 : ∃ m, n = 3^m) :
  3^(k + 1) ∣ a (3^k) :=
sorry

end NUMINAMATH_GPT_power_of_three_divides_an_l6_631


namespace NUMINAMATH_GPT_fifth_flower_is_e_l6_645

def flowers : List String := ["a", "b", "c", "d", "e", "f", "g"]

theorem fifth_flower_is_e : flowers.get! 4 = "e" := sorry

end NUMINAMATH_GPT_fifth_flower_is_e_l6_645


namespace NUMINAMATH_GPT_always_positive_sum_reciprocal_inequality_l6_632

-- Problem 1
theorem always_positive (x : ℝ) : x^6 - x^3 + x^2 - x + 1 > 0 :=
sorry

-- Problem 2
theorem sum_reciprocal_inequality (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 1) :
  1/a + 1/b + 1/c ≥ 9 :=
sorry

end NUMINAMATH_GPT_always_positive_sum_reciprocal_inequality_l6_632


namespace NUMINAMATH_GPT_parabola_line_non_intersect_l6_613

theorem parabola_line_non_intersect (r s : ℝ) (Q : ℝ × ℝ) (P : ℝ → ℝ)
  (hP : ∀ x, P x = x^2)
  (hQ : Q = (10, 6))
  (h_cond : ∀ m : ℝ, ¬∃ x : ℝ, (Q.snd - 6 = m * (Q.fst - 10)) ∧ (P x = x^2) ↔ r < m ∧ m < s) :
  r + s = 40 :=
sorry

end NUMINAMATH_GPT_parabola_line_non_intersect_l6_613


namespace NUMINAMATH_GPT_jasper_hot_dogs_fewer_l6_677

theorem jasper_hot_dogs_fewer (chips drinks hot_dogs : ℕ)
  (h1 : chips = 27)
  (h2 : drinks = 31)
  (h3 : drinks = hot_dogs + 12) : 27 - hot_dogs = 8 := by
  sorry

end NUMINAMATH_GPT_jasper_hot_dogs_fewer_l6_677


namespace NUMINAMATH_GPT_num_rows_of_gold_bars_l6_618

-- Definitions from the problem conditions
def num_bars_per_row : ℕ := 20
def total_worth : ℕ := 1600000

-- Statement to prove
theorem num_rows_of_gold_bars :
  (total_worth / (total_worth / num_bars_per_row)) = 1 := 
by sorry

end NUMINAMATH_GPT_num_rows_of_gold_bars_l6_618


namespace NUMINAMATH_GPT_cellphone_gifting_l6_628

theorem cellphone_gifting (n m : ℕ) (h1 : n = 20) (h2 : m = 3) : 
    (Finset.range n).card * (Finset.range (n - 1)).card * (Finset.range (n - 2)).card = 6840 := by
  sorry

end NUMINAMATH_GPT_cellphone_gifting_l6_628


namespace NUMINAMATH_GPT_solve_for_x_l6_619

theorem solve_for_x (x : ℝ) (h₁ : (x + 2) ≠ 0) (h₂ : (|x| - 2) / (x + 2) = 0) : x = 2 := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l6_619


namespace NUMINAMATH_GPT_distance_between_home_and_school_l6_641

variable (D T : ℝ)

def boy_travel_5kmhr : Prop :=
  5 * (T + 7 / 60) = D

def boy_travel_10kmhr : Prop :=
  10 * (T - 8 / 60) = D

theorem distance_between_home_and_school :
  (boy_travel_5kmhr D T) ∧ (boy_travel_10kmhr D T) → D = 2.5 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_distance_between_home_and_school_l6_641


namespace NUMINAMATH_GPT_two_digit_number_multiple_l6_604

theorem two_digit_number_multiple (x : ℕ) (h1 : x ≥ 10) (h2 : x < 100) 
(h3 : ∃ k : ℕ, x + 1 = 3 * k) 
(h4 : ∃ k : ℕ, x + 1 = 4 * k) 
(h5 : ∃ k : ℕ, x + 1 = 5 * k) 
(h6 : ∃ k : ℕ, x + 1 = 7 * k) 
: x = 83 := 
sorry

end NUMINAMATH_GPT_two_digit_number_multiple_l6_604


namespace NUMINAMATH_GPT_distinct_equilateral_triangles_in_polygon_l6_608

noncomputable def num_distinct_equilateral_triangles (P : Finset (Fin 10)) : Nat :=
  90

theorem distinct_equilateral_triangles_in_polygon (P : Finset (Fin 10)) :
  P.card = 10 →
  num_distinct_equilateral_triangles P = 90 :=
by
  intros
  sorry

end NUMINAMATH_GPT_distinct_equilateral_triangles_in_polygon_l6_608


namespace NUMINAMATH_GPT_extremum_at_neg3_l6_665

variable (a : ℝ)
def f (x : ℝ) : ℝ := x^3 + 5 * x^2 + a * x
def f_deriv (x : ℝ) : ℝ := 3 * x^2 + 10 * x + a

theorem extremum_at_neg3 (h : f_deriv a (-3) = 0) : a = 3 := 
  by
  sorry

end NUMINAMATH_GPT_extremum_at_neg3_l6_665


namespace NUMINAMATH_GPT_smaller_rectangle_area_l6_607

theorem smaller_rectangle_area (L_h S_h : ℝ) (L_v S_v : ℝ) 
  (ratio_h : L_h = (8 / 7) * S_h) 
  (ratio_v : L_v = (9 / 4) * S_v) 
  (area_large : L_h * L_v = 108) :
  S_h * S_v = 42 :=
sorry

end NUMINAMATH_GPT_smaller_rectangle_area_l6_607


namespace NUMINAMATH_GPT_cost_price_per_meter_l6_683

def total_length : ℝ := 9.25
def total_cost : ℝ := 397.75

theorem cost_price_per_meter : total_cost / total_length = 43 := sorry

end NUMINAMATH_GPT_cost_price_per_meter_l6_683


namespace NUMINAMATH_GPT_range_of_quadratic_function_l6_657

noncomputable def quadratic_function_range : Set ℝ :=
  { y : ℝ | ∃ x : ℝ, y = x^2 - 6 * x + 7 }

theorem range_of_quadratic_function :
  quadratic_function_range = { y : ℝ | y ≥ -2 } :=
by
  -- Insert proof here
  sorry

end NUMINAMATH_GPT_range_of_quadratic_function_l6_657


namespace NUMINAMATH_GPT_perpendicular_lines_iff_l6_644

theorem perpendicular_lines_iff (a : ℝ) : 
  (∀ b₁ b₂ : ℝ, b₁ ≠ b₂ → ¬ (∀ x : ℝ, a * x + b₁ = (a - 2) * x + b₂) ∧ 
   (a * (a - 2) = -1)) ↔ a = 1 :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_lines_iff_l6_644


namespace NUMINAMATH_GPT_problem_l6_667

theorem problem (a b : ℝ) (h : ∀ x : ℝ, (x + a) * (x + b) = x^2 + 4 * x + 3) : a + b = 4 :=
by
  sorry

end NUMINAMATH_GPT_problem_l6_667


namespace NUMINAMATH_GPT_lcm_of_18_and_24_l6_606

noncomputable def lcm_18_24 : ℕ :=
  Nat.lcm 18 24

theorem lcm_of_18_and_24 : lcm_18_24 = 72 :=
by
  sorry

end NUMINAMATH_GPT_lcm_of_18_and_24_l6_606


namespace NUMINAMATH_GPT_prove_tan_570_eq_sqrt_3_over_3_l6_681

noncomputable def tan_570_eq_sqrt_3_over_3 : Prop :=
  Real.tan (570 * Real.pi / 180) = Real.sqrt 3 / 3

theorem prove_tan_570_eq_sqrt_3_over_3 : tan_570_eq_sqrt_3_over_3 :=
by
  sorry

end NUMINAMATH_GPT_prove_tan_570_eq_sqrt_3_over_3_l6_681


namespace NUMINAMATH_GPT_complex_number_quadrant_l6_654

theorem complex_number_quadrant (a b : ℝ) (h1 : (2 + a * (0+1*I)) / (1 + 1*I) = b + 1*I) (h2: a = 4) (h3: b = 3) : 
  0 < a ∧ 0 < b :=
by
  sorry

end NUMINAMATH_GPT_complex_number_quadrant_l6_654


namespace NUMINAMATH_GPT_cara_total_debt_l6_627

def simple_interest (P R T : ℝ) : ℝ := P * R * T

theorem cara_total_debt :
  let P := 54
  let R := 0.05
  let T := 1
  let I := simple_interest P R T
  let total := P + I
  total = 56.7 :=
by
  sorry

end NUMINAMATH_GPT_cara_total_debt_l6_627


namespace NUMINAMATH_GPT_find_common_ratio_l6_694

variable {a : ℕ → ℝ}
variable {q : ℝ}

-- Conditions
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n * q

axiom a2 : a 2 = 9
axiom a3_plus_a4 : a 3 + a 4 = 18
axiom q_not_one : q ≠ 1

-- Proof problem
theorem find_common_ratio
  (h : is_geometric_sequence a q)
  (ha2 : a 2 = 9)
  (ha3a4 : a 3 + a 4 = 18)
  (hq : q ≠ 1) :
  q = -2 :=
sorry

end NUMINAMATH_GPT_find_common_ratio_l6_694


namespace NUMINAMATH_GPT_real_y_iff_x_l6_668

open Real

-- Definitions based on the conditions
def quadratic_eq (y x : ℝ) : ℝ := 9 * y^2 - 3 * x * y + x + 8

-- The main theorem to prove
theorem real_y_iff_x (x : ℝ) : (∃ y : ℝ, quadratic_eq y x = 0) ↔ x ≤ -4 ∨ x ≥ 8 := 
sorry

end NUMINAMATH_GPT_real_y_iff_x_l6_668


namespace NUMINAMATH_GPT_Jessica_cut_roses_l6_656

theorem Jessica_cut_roses
  (initial_roses : ℕ) (initial_orchids : ℕ)
  (new_roses : ℕ) (new_orchids : ℕ)
  (cut_roses : ℕ) :
  initial_roses = 15 → initial_orchids = 62 →
  new_roses = 17 → new_orchids = 96 →
  new_roses = initial_roses + cut_roses →
  cut_roses = 2 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h3] at h5
  linarith

end NUMINAMATH_GPT_Jessica_cut_roses_l6_656


namespace NUMINAMATH_GPT_sqrt_infinite_nest_eq_two_l6_600

theorem sqrt_infinite_nest_eq_two (y : ℝ) (h : y = Real.sqrt (2 + y)) : y = 2 := 
sorry

end NUMINAMATH_GPT_sqrt_infinite_nest_eq_two_l6_600


namespace NUMINAMATH_GPT_gcd_1978_2017_l6_609

theorem gcd_1978_2017 : Int.gcd 1978 2017 = 1 :=
sorry

end NUMINAMATH_GPT_gcd_1978_2017_l6_609


namespace NUMINAMATH_GPT_sum_of_ten_numbers_l6_695

theorem sum_of_ten_numbers (average count : ℝ) (h_avg : average = 5.3) (h_count : count = 10) : 
  average * count = 53 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_ten_numbers_l6_695


namespace NUMINAMATH_GPT_ceil_evaluation_l6_615

theorem ceil_evaluation : 
  (Int.ceil (((-7 : ℚ) / 4) ^ 2 - (1 / 8)) = 3) :=
sorry

end NUMINAMATH_GPT_ceil_evaluation_l6_615


namespace NUMINAMATH_GPT_value_of_a_l6_638

theorem value_of_a (a b c : ℂ) (h_real : a.im = 0)
  (h1 : a + b + c = 5) 
  (h2 : a * b + b * c + c * a = 7) 
  (h3 : a * b * c = 2) : a = 2 := by
  sorry

end NUMINAMATH_GPT_value_of_a_l6_638


namespace NUMINAMATH_GPT_correct_factorization_l6_690

theorem correct_factorization (a b : ℝ) : 
  ((x + 6) * (x - 1) = x^2 + 5 * x - 6) →
  ((x - 2) * (x + 1) = x^2 - x - 2) →
  (a = 1 ∧ b = -6) →
  (x^2 - x - 6 = (x + 2) * (x - 3)) :=
sorry

end NUMINAMATH_GPT_correct_factorization_l6_690


namespace NUMINAMATH_GPT_geometric_reasoning_l6_646

-- Definitions of relationships between geometric objects
inductive GeometricObject
  | Line
  | Plane

open GeometricObject

def perpendicular (a b : GeometricObject) : Prop := 
  match a, b with
  | Plane, Plane => True  -- Planes can be perpendicular
  | Line, Plane => True   -- Lines can be perpendicular to planes
  | Plane, Line => True   -- Planes can be perpendicular to lines
  | Line, Line => True    -- Lines can be perpendicular to lines (though normally in a 3D space specific context)

def parallel (a b : GeometricObject) : Prop := 
  match a, b with
  | Plane, Plane => True  -- Planes can be parallel
  | Line, Plane => True   -- Lines can be parallel to planes under certain interpretation
  | Plane, Line => True
  | Line, Line => True    -- Lines can be parallel

axiom x : GeometricObject
axiom y : GeometricObject
axiom z : GeometricObject

-- Main theorem statement
theorem geometric_reasoning (hx : perpendicular x y) (hy : parallel y z) 
  : ¬ (perpendicular x z) → (x = Plane ∧ y = Plane ∧ z = Line) :=
  sorry

end NUMINAMATH_GPT_geometric_reasoning_l6_646


namespace NUMINAMATH_GPT_bug_return_probability_twelfth_move_l6_617

-- Conditions
def P : ℕ → ℚ
| 0       => 1
| (n + 1) => (1 : ℚ) / 3 * (1 - P n)

theorem bug_return_probability_twelfth_move :
  P 12 = 14762 / 59049 := by
sorry

end NUMINAMATH_GPT_bug_return_probability_twelfth_move_l6_617


namespace NUMINAMATH_GPT_condition_an_necessary_but_not_sufficient_l6_620

-- Definitions for the sequence and properties
def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, n ≥ 1 → a (n + 1) = r * (a n)

def condition_an (a : ℕ → ℝ) : Prop :=
  ∀ n, n ≥ 2 → a n = 2 * a (n - 1)

-- The theorem statement
theorem condition_an_necessary_but_not_sufficient (a : ℕ → ℝ) :
  (∀ n, n ≥ 1 → a (n + 1) = 2 * (a n)) → (condition_an a) ∧ ¬(is_geometric_sequence a 2) :=
by
  sorry

end NUMINAMATH_GPT_condition_an_necessary_but_not_sufficient_l6_620


namespace NUMINAMATH_GPT_find_x_l6_693

theorem find_x (x : ℝ) (h : 3 * x + 15 = (1 / 3) * (7 * x + 42)) : x = -3 / 2 :=
sorry

end NUMINAMATH_GPT_find_x_l6_693


namespace NUMINAMATH_GPT_classify_tangents_through_point_l6_679

-- Definitions for the Lean theorem statement
noncomputable def curve (x : ℝ) : ℝ :=
  x^3 - x

noncomputable def phi (t x₀ y₀ : ℝ) : ℝ :=
  2*t^3 - 3*x₀*t^2 + (x₀ + y₀)

theorem classify_tangents_through_point (x₀ y₀ : ℝ) :
  (if (x₀ + y₀ < 0 ∨ y₀ > x₀^3 - x₀)
   then 1
   else if (x₀ + y₀ > 0 ∧ x₀ + y₀ - x₀^3 < 0)
   then 3
   else if (x₀ + y₀ = 0 ∨ y₀ = x₀^3 - x₀)
   then 2
   else 0) = 
  (if (x₀ + y₀ < 0 ∨ y₀ > x₀^3 - x₀)
   then 1
   else if (x₀ + y₀ > 0 ∧ x₀ + y₀ - x₀^3 < 0)
   then 3
   else if (x₀ + y₀ = 0 ∨ y₀ = x₀^3 - x₀)
   then 2
   else 0) :=
  sorry

end NUMINAMATH_GPT_classify_tangents_through_point_l6_679


namespace NUMINAMATH_GPT_length_squared_of_segment_CD_is_196_l6_649

theorem length_squared_of_segment_CD_is_196 :
  ∃ (C D : ℝ × ℝ), 
    (C.2 = 3 * C.1 ^ 2 + 6 * C.1 - 2) ∧
    (D.2 = 3 * (2 - C.1) ^ 2 + 6 * (2 - C.1) - 2) ∧
    (1 : ℝ) = (C.1 + D.1) / 2 ∧
    (0 : ℝ) = (C.2 + D.2) / 2 ∧
    ((C.1 - D.1) ^ 2 + (C.2 - D.2) ^ 2 = 196) :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_length_squared_of_segment_CD_is_196_l6_649


namespace NUMINAMATH_GPT_consecutive_probability_is_two_fifths_l6_697

-- Conditions
def total_days : ℕ := 5
def select_days : ℕ := 2

-- Total number of basic events (number of ways to choose 2 days out of 5)
def total_events : ℕ := Nat.choose total_days select_days -- This is C(5, 2)

-- Number of basic events where 2 selected days are consecutive
def consecutive_events : ℕ := 4

-- Probability that the selected 2 days are consecutive
def consecutive_probability : ℚ := consecutive_events / total_events

-- Theorem to be proved
theorem consecutive_probability_is_two_fifths :
  consecutive_probability = 2 / 5 :=
by
  sorry

end NUMINAMATH_GPT_consecutive_probability_is_two_fifths_l6_697


namespace NUMINAMATH_GPT_find_t_l6_633

variable (t : ℚ)

def point_on_line (p1 p2 p3 : ℚ × ℚ) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (y2 - y1) * (x3 - x1) = (y3 - y1) * (x2 - x1)

theorem find_t (t : ℚ) : point_on_line (3, 0) (0, 7) (t, 8) → t = -3 / 7 := by
  sorry

end NUMINAMATH_GPT_find_t_l6_633


namespace NUMINAMATH_GPT_no_sol_n4_minus_m4_eq_42_l6_647

theorem no_sol_n4_minus_m4_eq_42 :
  ¬ ∃ (n m : ℕ), 0 < n ∧ 0 < m ∧ n^4 - m^4 = 42 :=
by
  sorry

end NUMINAMATH_GPT_no_sol_n4_minus_m4_eq_42_l6_647


namespace NUMINAMATH_GPT_proof_problem_l6_624

theorem proof_problem (s t: ℤ) (h : 514 - s = 600 - t) : s < t ∧ t - s = 86 :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l6_624


namespace NUMINAMATH_GPT_inequality_holds_for_all_x_iff_l6_658

theorem inequality_holds_for_all_x_iff (m : ℝ) :
  (∀ (x : ℝ), m * x^2 + m * x - 4 < 2 * x^2 + 2 * x - 1) ↔ -10 < m ∧ m ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_holds_for_all_x_iff_l6_658


namespace NUMINAMATH_GPT_shoe_store_total_shoes_l6_651

theorem shoe_store_total_shoes (b k : ℕ) (h1 : b = 22) (h2 : k = 2 * b) : b + k = 66 :=
by
  sorry

end NUMINAMATH_GPT_shoe_store_total_shoes_l6_651


namespace NUMINAMATH_GPT_common_difference_is_1_over_10_l6_675

open Real

noncomputable def a_n (a₁ d: ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

noncomputable def S_n (a₁ d : ℝ) (n : ℕ) : ℝ := 
  n * a₁ + (n * (n - 1)) * d / 2

theorem common_difference_is_1_over_10 (a₁ d : ℝ) 
  (h : (S_n a₁ d 2017 / 2017) - (S_n a₁ d 17 / 17) = 100) : 
  d = 1 / 10 :=
by
  sorry

end NUMINAMATH_GPT_common_difference_is_1_over_10_l6_675


namespace NUMINAMATH_GPT_inequality_holds_and_equality_occurs_l6_689

theorem inequality_holds_and_equality_occurs (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x * y = 4) :
  (1 / (x + 3) + 1 / (y + 3) ≤ 2 / 5) ∧ (x = 2 ∧ y = 2 → 1 / (x + 3) + 1 / (y + 3) = 2 / 5) :=
by
  sorry

end NUMINAMATH_GPT_inequality_holds_and_equality_occurs_l6_689


namespace NUMINAMATH_GPT_at_least_one_truth_and_not_knight_l6_601

def isKnight (n : Nat) : Prop := n = 1   -- Identifier for knights
def isKnave (n : Nat) : Prop := n = 0    -- Identifier for knaves
def isRegular (n : Nat) : Prop := n = 2  -- Identifier for regular persons

def A := 2     -- Initially define A's type as regular (this can be adjusted)
def B := 2     -- Initially define B's type as regular (this can be adjusted)

def statementA : Prop := isKnight B
def statementB : Prop := ¬ isKnight A

theorem at_least_one_truth_and_not_knight :
  statementA ∧ ¬ isKnight A ∨ statementB ∧ ¬ isKnight B :=
sorry

end NUMINAMATH_GPT_at_least_one_truth_and_not_knight_l6_601


namespace NUMINAMATH_GPT_find_D_l6_666

variables (A B C D : ℤ)
axiom h1 : A + C = 15
axiom h2 : A - B = 1
axiom h3 : C + C = A
axiom h4 : B - D = 2
axiom h5 : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D

theorem find_D : D = 7 :=
by sorry

end NUMINAMATH_GPT_find_D_l6_666


namespace NUMINAMATH_GPT_articles_produced_l6_634

theorem articles_produced (a b c d f p q r g : ℕ) :
  (a * b * c = d) → 
  ((p * q * r * d * g) / (a * b * c * f) = pqr * d * g / (abc * f)) :=
by
  sorry

end NUMINAMATH_GPT_articles_produced_l6_634


namespace NUMINAMATH_GPT_complex_fraction_eval_l6_676

theorem complex_fraction_eval (a b : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a^2 + a * b + b^2 = 0) :
  (a^15 + b^15) / (a + b)^15 = -2 := by
sorry

end NUMINAMATH_GPT_complex_fraction_eval_l6_676


namespace NUMINAMATH_GPT_Winnie_the_Pooh_honey_consumption_l6_655

theorem Winnie_the_Pooh_honey_consumption (W0 W1 W2 W3 W4 : ℝ) (pot_empty : ℝ) 
  (h1 : W1 = W0 / 2)
  (h2 : W2 = W1 / 2)
  (h3 : W3 = W2 / 2)
  (h4 : W4 = W3 / 2)
  (h5 : W4 = 200)
  (h6 : pot_empty = 200) : 
  W0 - 200 = 3000 := by
  sorry

end NUMINAMATH_GPT_Winnie_the_Pooh_honey_consumption_l6_655


namespace NUMINAMATH_GPT_product_no_xx_x_eq_x_cube_plus_one_l6_661

theorem product_no_xx_x_eq_x_cube_plus_one (a c : ℝ) (h1 : a - 1 = 0) (h2 : c - a = 0) : 
  (x + a) * (x ^ 2 - x + c) = x ^ 3 + 1 :=
by {
  -- Here would be the proof steps, which we omit with "sorry"
  sorry
}

end NUMINAMATH_GPT_product_no_xx_x_eq_x_cube_plus_one_l6_661


namespace NUMINAMATH_GPT_maximum_m_value_l6_635

theorem maximum_m_value (a : ℕ → ℤ) (m : ℕ) :
  (∀ n, a (n + 1) - a n = 3) →
  a 3 = -2 →
  (∀ k : ℕ, k ≥ 4 → (3 * k - 8) * (3 * k - 5) / (3 * k - 11) ≥ 3 * m - 11) →
  m ≤ 9 :=
by
  sorry

end NUMINAMATH_GPT_maximum_m_value_l6_635


namespace NUMINAMATH_GPT_power_mod_remainder_l6_642

theorem power_mod_remainder (a b c : ℕ) (h1 : 7^40 % 500 = 1) (h2 : 7^4 % 40 = 1) : (7^(7^25) % 500 = 43) :=
sorry

end NUMINAMATH_GPT_power_mod_remainder_l6_642


namespace NUMINAMATH_GPT_find_length_of_other_diagonal_l6_611

theorem find_length_of_other_diagonal
  (area : ℝ) (d1 : ℝ) (d2 : ℝ) 
  (h1: area = 75)
  (h2: d1 = 10) :
  d2 = 15 :=
by 
  sorry

end NUMINAMATH_GPT_find_length_of_other_diagonal_l6_611


namespace NUMINAMATH_GPT_tank_capacity_l6_663

theorem tank_capacity (w c : ℕ) (h1 : w = c / 3) (h2 : w + 7 = 2 * c / 5) : c = 105 :=
sorry

end NUMINAMATH_GPT_tank_capacity_l6_663
