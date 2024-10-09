import Mathlib

namespace correct_average_l1767_176738

-- Define the conditions given in the problem
def avg_incorrect : ℕ := 46 -- incorrect average
def n : ℕ := 10 -- number of values
def incorrect_num : ℕ := 25
def correct_num : ℕ := 75
def diff : ℕ := correct_num - incorrect_num

-- Define the total sums
def total_incorrect : ℕ := avg_incorrect * n
def total_correct : ℕ := total_incorrect + diff

-- Define the correct average
def avg_correct : ℕ := total_correct / n

-- Statement in Lean 4
theorem correct_average :
  avg_correct = 51 :=
by
  -- We expect users to fill the proof here
  sorry

end correct_average_l1767_176738


namespace students_not_enrolled_in_any_classes_l1767_176721

/--
  At a particular college, 27.5% of the 1050 students are enrolled in biology,
  32.9% of the students are enrolled in mathematics, and 15% of the students are enrolled in literature classes.
  Assuming that no student is taking more than one of these specific subjects,
  the number of students at the college who are not enrolled in biology, mathematics, or literature classes is 260.

  We want to prove the statement:
    number_students_not_enrolled_in_any_classes = 260
-/
theorem students_not_enrolled_in_any_classes 
  (total_students : ℕ) 
  (biology_percent : ℝ) 
  (mathematics_percent : ℝ) 
  (literature_percent : ℝ) 
  (no_student_in_multiple : Prop) : 
  total_students = 1050 →
  biology_percent = 27.5 →
  mathematics_percent = 32.9 →
  literature_percent = 15 →
  (total_students - (⌊biology_percent / 100 * total_students⌋ + ⌊mathematics_percent / 100 * total_students⌋ + ⌊literature_percent / 100 * total_students⌋)) = 260 :=
by {
  sorry
}

end students_not_enrolled_in_any_classes_l1767_176721


namespace fraction_value_l1767_176727

variable (u v w x : ℝ)

-- Conditions
def cond1 : Prop := u / v = 5
def cond2 : Prop := w / v = 3
def cond3 : Prop := w / x = 2 / 3

theorem fraction_value (h1 : cond1 u v) (h2 : cond2 w v) (h3 : cond3 w x) : x / u = 9 / 10 := 
by
  sorry

end fraction_value_l1767_176727


namespace factor_expression_l1767_176733

theorem factor_expression (b : ℝ) : 180 * b ^ 2 + 36 * b = 36 * b * (5 * b + 1) :=
by
  -- actual proof is omitted
  sorry

end factor_expression_l1767_176733


namespace Cody_age_is_14_l1767_176743

variable (CodyGrandmotherAge CodyAge : ℕ)

theorem Cody_age_is_14 (h1 : CodyGrandmotherAge = 6 * CodyAge) (h2 : CodyGrandmotherAge = 84) : CodyAge = 14 := by
  sorry

end Cody_age_is_14_l1767_176743


namespace executiveCommittee_ways_l1767_176775

noncomputable def numberOfWaysToFormCommittee (totalMembers : ℕ) (positions : ℕ) : ℕ :=
Nat.choose (totalMembers - 1) (positions - 1)

theorem executiveCommittee_ways : numberOfWaysToFormCommittee 30 5 = 25839 := 
by
  -- skipping the proof as it's not required
  sorry

end executiveCommittee_ways_l1767_176775


namespace solution_set_of_inequality_l1767_176769

theorem solution_set_of_inequality :
  { x : ℝ | 2 * x^2 - x - 3 > 0 } = { x : ℝ | x > 3 / 2 ∨ x < -1 } :=
sorry

end solution_set_of_inequality_l1767_176769


namespace regular_polygon_with_12_degree_exterior_angle_has_30_sides_l1767_176710

def regular_polygon_sides (e : ℤ) : ℤ :=
  360 / e

theorem regular_polygon_with_12_degree_exterior_angle_has_30_sides :
  regular_polygon_sides 12 = 30 :=
by
  -- Proof is omitted
  sorry

end regular_polygon_with_12_degree_exterior_angle_has_30_sides_l1767_176710


namespace two_distinct_solutions_diff_l1767_176765

theorem two_distinct_solutions_diff (a b : ℝ) (h1 : a ≠ b) (h2 : a > b)
  (h3 : ∀ x, (x = a ∨ x = b) ↔ (6 * x - 18) / (x^2 + 3 * x - 18) = x + 3) :
  a - b = 3 :=
by
  -- Proof will be provided here.
  sorry

end two_distinct_solutions_diff_l1767_176765


namespace probability_at_least_one_defective_is_correct_l1767_176719

/-- Define a box containing 21 bulbs, 4 of which are defective -/
def total_bulbs : ℕ := 21
def defective_bulbs : ℕ := 4
def non_defective_bulbs : ℕ := total_bulbs - defective_bulbs

/-- Define probabilities of choosing non-defective bulbs -/
def prob_first_non_defective : ℚ := non_defective_bulbs / total_bulbs
def prob_second_non_defective : ℚ := (non_defective_bulbs - 1) / (total_bulbs - 1)

/-- Calculate the probability of both bulbs being non-defective -/
def prob_both_non_defective : ℚ := prob_first_non_defective * prob_second_non_defective

/-- Calculate the probability of at least one defective bulb -/
def prob_at_least_one_defective : ℚ := 1 - prob_both_non_defective

theorem probability_at_least_one_defective_is_correct :
  prob_at_least_one_defective = 37 / 105 :=
by
  -- Sorry allows us to skip the proof
  sorry

end probability_at_least_one_defective_is_correct_l1767_176719


namespace four_digit_number_l1767_176745

-- Definitions of the conditions
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

-- Statement of the theorem
theorem four_digit_number (x y : ℕ) (hx : is_two_digit x) (hy : is_two_digit y) :
    (100 * x + y) = 1000 * x + y := sorry

end four_digit_number_l1767_176745


namespace find_f_2016_l1767_176749

noncomputable def f : ℝ → ℝ :=
sorry

axiom f_0_eq_2016 : f 0 = 2016

axiom f_x_plus_2_minus_f_x_leq : ∀ x : ℝ, f (x + 2) - f x ≤ 3 * 2 ^ x

axiom f_x_plus_6_minus_f_x_geq : ∀ x : ℝ, f (x + 6) - f x ≥ 63 * 2 ^ x

theorem find_f_2016 : f 2016 = 2015 + 2 ^ 2020 :=
sorry

end find_f_2016_l1767_176749


namespace convert_deg_to_rad_l1767_176787

theorem convert_deg_to_rad (deg : ℝ) (π : ℝ) (h : deg = 50) : (deg * (π / 180) = 5 / 18 * π) :=
by
  -- Conditions
  sorry

end convert_deg_to_rad_l1767_176787


namespace license_plates_count_correct_l1767_176740

/-- Calculate the number of five-character license plates. -/
def count_license_plates : Nat :=
  let num_consonants := 20
  let num_vowels := 6
  let num_digits := 10
  num_consonants^2 * num_vowels^2 * num_digits

theorem license_plates_count_correct :
  count_license_plates = 144000 :=
by
  sorry

end license_plates_count_correct_l1767_176740


namespace veranda_width_l1767_176756

-- Defining the conditions as given in the problem
def room_length : ℝ := 21
def room_width : ℝ := 12
def veranda_area : ℝ := 148

-- The main statement to prove
theorem veranda_width :
  ∃ (w : ℝ), (21 + 2*w) * (12 + 2*w) - 21 * 12 = 148 ∧ w = 2 :=
by
  sorry

end veranda_width_l1767_176756


namespace odd_function_expression_on_negative_domain_l1767_176754

theorem odd_function_expression_on_negative_domain
  (f : ℝ → ℝ)
  (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_pos : ∀ x : ℝ, 0 < x → f x = x * (x - 1))
  (x : ℝ)
  (h_neg : x < 0)
  : f x = x * (x + 1) :=
sorry

end odd_function_expression_on_negative_domain_l1767_176754


namespace find_b_l1767_176790

theorem find_b (b p : ℝ) 
  (h1 : 3 * p + 15 = 0)
  (h2 : 15 * p + 3 = b) :
  b = -72 :=
by
  sorry

end find_b_l1767_176790


namespace pencil_length_l1767_176729

theorem pencil_length (L : ℝ) 
  (h1 : 1 / 8 * L = b) 
  (h2 : 1 / 2 * (L - 1 / 8 * L) = w) 
  (h3 : (L - 1 / 8 * L - 1 / 2 * (L - 1 / 8 * L)) = 7 / 2) :
  L = 8 :=
sorry

end pencil_length_l1767_176729


namespace problem1_problem2_l1767_176736

variable {a b : ℝ}

theorem problem1 (ha : a ≠ 0) (hb : b ≠ 0) :
  (4 * b^3 / a) / (2 * b / a^2) = 2 * a * b^2 :=
by 
  sorry

theorem problem2 (ha : a ≠ b) :
  (a^2 / (a - b)) + (b^2 / (a - b)) - (2 * a * b / (a - b)) = a - b :=
by 
  sorry

end problem1_problem2_l1767_176736


namespace find_b_l1767_176782

theorem find_b (b : ℚ) (H : ∃ x y : ℚ, x = 3 ∧ y = -7 ∧ b * x + (b - 1) * y = b + 3) : 
  b = 4 / 5 := 
by
  sorry

end find_b_l1767_176782


namespace tangent_line_at_origin_is_minus_3x_l1767_176789

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a * x^2 + (a - 3) * x
noncomputable def f_prime (a : ℝ) (x : ℝ) : ℝ := 3 * x^2 + 2 * a * x + (a - 3)

theorem tangent_line_at_origin_is_minus_3x (a : ℝ) (h : ∀ x : ℝ, f_prime a x = f_prime a (-x)) : 
  (f_prime 0 0 = -3) → ∀ x : ℝ, (f a x = -3 * x) :=
by
  sorry

end tangent_line_at_origin_is_minus_3x_l1767_176789


namespace initial_rulers_calculation_l1767_176703

variable {initial_rulers taken_rulers left_rulers : ℕ}

theorem initial_rulers_calculation 
  (h1 : taken_rulers = 25) 
  (h2 : left_rulers = 21) 
  (h3 : initial_rulers = taken_rulers + left_rulers) : 
  initial_rulers = 46 := 
by 
  sorry

end initial_rulers_calculation_l1767_176703


namespace sector_to_cone_volume_l1767_176798

theorem sector_to_cone_volume (θ : ℝ) (A : ℝ) (V : ℝ) (l r h : ℝ) :
  θ = (2 * Real.pi / 3) →
  A = (3 * Real.pi) →
  A = (1 / 2 * l^2 * θ) →
  θ = (r / l * 2 * Real.pi) →
  h = Real.sqrt (l^2 - r^2) →
  V = (1 / 3 * Real.pi * r^2 * h) →
  V = (2 * Real.sqrt 2 * Real.pi / 3) :=
by
  intros hθ hA hAeq hθeq hh hVeq
  sorry

end sector_to_cone_volume_l1767_176798


namespace flower_nectar_water_content_l1767_176797

/-- Given that to yield 1 kg of honey, 1.6 kg of flower-nectar must be processed,
    and the honey obtained from this nectar contains 20% water,
    prove that the flower-nectar contains 50% water. --/
theorem flower_nectar_water_content :
  (1.6 : ℝ) * (0.2 / 1) = (50 / 100) * (1.6 : ℝ) := by
  sorry

end flower_nectar_water_content_l1767_176797


namespace fraction_pow_zero_is_one_l1767_176718

theorem fraction_pow_zero_is_one (a b : ℤ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) : (a / (b : ℚ)) ^ 0 = 1 := by
  sorry

end fraction_pow_zero_is_one_l1767_176718


namespace linda_spent_amount_l1767_176722

theorem linda_spent_amount :
  let cost_notebooks := 3 * 1.20
  let cost_pencils := 1.50
  let cost_pens := 1.70
  let total_cost := cost_notebooks + cost_pencils + cost_pens
  total_cost = 6.80 :=
by
  let cost_notebooks := 3 * 1.20
  let cost_pencils := 1.50
  let cost_pens := 1.70
  let total_cost := cost_notebooks + cost_pencils + cost_pens
  show total_cost = 6.80
  sorry

end linda_spent_amount_l1767_176722


namespace largest_possible_perimeter_l1767_176748

noncomputable def max_perimeter (a b c: ℕ) : ℕ := 2 * (a + b + c - 6)

theorem largest_possible_perimeter :
  ∃ (a b c : ℕ), (a = c) ∧ ((a - 2) * (b - 2) = 8) ∧ (max_perimeter a b c = 42) := by
  sorry

end largest_possible_perimeter_l1767_176748


namespace problem_solution_l1767_176760

theorem problem_solution (x y : ℝ) (h1 : y = x / (3 * x + 1)) (hx : x ≠ 0) (hy : y ≠ 0) :
    (x - y + 3 * x * y) / (x * y) = 6 := by
  sorry

end problem_solution_l1767_176760


namespace minimum_value_w_l1767_176776

theorem minimum_value_w : ∃ (x y : ℝ), ∀ w, w = 3 * x^2 + 3 * y^2 + 9 * x - 6 * y + 30 → w ≥ 20.25 :=
sorry

end minimum_value_w_l1767_176776


namespace sixth_employee_salary_l1767_176788

-- We define the salaries of the five employees
def salaries : List ℝ := [1000, 2500, 3100, 1500, 2000]

-- The mean of the salaries of these 5 employees and another employee
def mean_salary : ℝ := 2291.67

-- The number of employees
def number_of_employees : ℝ := 6

-- The total salary of the first five employees
def total_salary_5 : ℝ := salaries.sum

-- The total salary based on the given mean and number of employees
def total_salary_all : ℝ := mean_salary * number_of_employees

-- The statement to prove: The salary of the sixth employee
theorem sixth_employee_salary :
  total_salary_all - total_salary_5 = 3650.02 := 
  sorry

end sixth_employee_salary_l1767_176788


namespace circle_reflection_l1767_176747

/-- The reflection of a point over the line y = -x results in swapping the x and y coordinates 
and changing their signs. Given a circle with center (3, -7), the reflected center should be (7, -3). -/
theorem circle_reflection (x y : ℝ) (h : (x, y) = (3, -7)) : (y, -x) = (7, -3) :=
by
  -- since the problem is stated to skip the proof, we use sorry
  sorry

end circle_reflection_l1767_176747


namespace M_inter_N_eq_l1767_176705

open Set

def M : Set ℝ := { m | -3 < m ∧ m < 2 }
def N : Set ℤ := { n | -1 < n ∧ n ≤ 3 }

theorem M_inter_N_eq : M ∩ (coe '' N) = {0, 1} :=
by sorry

end M_inter_N_eq_l1767_176705


namespace proof_N_union_complement_M_eq_235_l1767_176784

open Set

theorem proof_N_union_complement_M_eq_235 :
  let U := ({1,2,3,4,5} : Set ℕ)
  let M := ({1, 4} : Set ℕ)
  let N := ({2, 5} : Set ℕ)
  N ∪ (U \ M) = ({2, 3, 5} : Set ℕ) :=
by
  sorry

end proof_N_union_complement_M_eq_235_l1767_176784


namespace exists_abc_gcd_equation_l1767_176779

theorem exists_abc_gcd_equation (n : ℕ) : ∃ a b c : ℤ, n = Int.gcd a b * (c^2 - a*b) + Int.gcd b c * (a^2 - b*c) + Int.gcd c a * (b^2 - c*a) := sorry

end exists_abc_gcd_equation_l1767_176779


namespace number_of_math_books_l1767_176753

-- Definitions for conditions
variables (M H : ℕ)

-- Given conditions as a Lean proposition
def conditions : Prop :=
  M + H = 80 ∧ 4 * M + 5 * H = 368

-- The theorem to prove
theorem number_of_math_books (M H : ℕ) (h : conditions M H) : M = 32 :=
by sorry

end number_of_math_books_l1767_176753


namespace rate_of_simple_interest_l1767_176711

theorem rate_of_simple_interest (P : ℝ) (R : ℝ) (T : ℝ) (P_nonzero : P ≠ 0) : 
  (P * R * T = P / 6) → R = 1 / 42 :=
by
  intro h
  sorry

end rate_of_simple_interest_l1767_176711


namespace alia_markers_l1767_176785

theorem alia_markers (S A a : ℕ) (h1 : S = 60) (h2 : A = S / 3) (h3 : a = 2 * A) : a = 40 :=
by
  -- Proof omitted
  sorry

end alia_markers_l1767_176785


namespace class_mean_score_l1767_176795

theorem class_mean_score:
  ∀ (n: ℕ) (m: ℕ) (a b: ℕ),
  n + m = 50 →
  n * a = 3400 →
  m * b = 750 →
  a = 85 →
  b = 75 →
  (n * a + m * b) / (n + m) = 83 :=
by
  intros n m a b h1 h2 h3 h4 h5
  sorry

end class_mean_score_l1767_176795


namespace find_first_factor_of_LCM_l1767_176712

-- Conditions
def HCF : ℕ := 23
def Y : ℕ := 14
def largest_number : ℕ := 322

-- Statement
theorem find_first_factor_of_LCM
  (A B : ℕ)
  (H : Nat.gcd A B = HCF)
  (max_num : max A B = largest_number)
  (lcm_eq : Nat.lcm A B = HCF * X * Y) :
  X = 23 :=
sorry

end find_first_factor_of_LCM_l1767_176712


namespace odd_and_symmetric_f_l1767_176773

open Real

noncomputable def f (A ϕ : ℝ) (x : ℝ) := A * sin (x + ϕ)

theorem odd_and_symmetric_f (A ϕ : ℝ) (hA : A > 0) (hmin : f A ϕ (π / 4) = -1) : 
  ∃ g : ℝ → ℝ, g x = -A * sin x ∧ (∀ x, g (-x) = -g x) ∧ (∀ x, g (π / 2 - x) = g (π / 2 + x)) :=
sorry

end odd_and_symmetric_f_l1767_176773


namespace craig_apples_total_l1767_176757

-- Defining the conditions
def initial_apples_craig : ℝ := 20.0
def apples_from_eugene : ℝ := 7.0

-- Defining the total number of apples Craig will have
noncomputable def total_apples_craig : ℝ := initial_apples_craig + apples_from_eugene

-- The theorem stating that Craig will have 27.0 apples.
theorem craig_apples_total : total_apples_craig = 27.0 := by
  -- Proof here
  sorry

end craig_apples_total_l1767_176757


namespace lina_walk_probability_l1767_176751

/-- Total number of gates -/
def num_gates : ℕ := 20

/-- Distance between adjacent gates in feet -/
def gate_distance : ℕ := 50

/-- Maximum distance in feet Lina can walk to be within the desired range -/
def max_walk_distance : ℕ := 200

/-- Number of gates Lina can move within the max walk distance -/
def max_gates_within_distance : ℕ := max_walk_distance / gate_distance

/-- Total possible gate pairs for initial and new gate selection -/
def total_possible_pairs : ℕ := num_gates * (num_gates - 1)

/-- Total number of favorable gate pairs where walking distance is within the allowed range -/
def total_favorable_pairs : ℕ :=
  let edge_favorable (g : ℕ) := if g = 1 ∨ g = num_gates then 4
                                else if g = 2 ∨ g = num_gates - 1 then 5
                                else if g = 3 ∨ g = num_gates - 2 then 6
                                else if g = 4 ∨ g = num_gates - 3 then 7 else 8
  (edge_favorable 1) + (edge_favorable 2) + (edge_favorable 3) +
  (edge_favorable 4) + (num_gates - 8) * 8

/-- Probability that Lina walks 200 feet or less expressed as a reduced fraction -/
def probability_within_distance : ℚ :=
  (total_favorable_pairs : ℚ) / (total_possible_pairs : ℚ)

/-- p and q components of the fraction representing the probability -/
def p := 7
def q := 19

/-- Sum of p and q -/
def p_plus_q : ℕ := p + q

theorem lina_walk_probability : p_plus_q = 26 := by sorry

end lina_walk_probability_l1767_176751


namespace range_of_3x_minus_2y_l1767_176713

variable (x y : ℝ)

theorem range_of_3x_minus_2y (h1 : -1 ≤ x + y ∧ x + y ≤ 1) (h2 : 1 ≤ x - y ∧ x - y ≤ 5) :
  ∃ (a b : ℝ), 2 ≤ a ∧ a ≤ b ∧ b ≤ 13 ∧ (3 * x - 2 * y = a ∨ 3 * x - 2 * y = b) :=
by
  sorry

end range_of_3x_minus_2y_l1767_176713


namespace unique_zero_function_l1767_176702

variable (f : ℝ → ℝ)

theorem unique_zero_function (h : ∀ x y : ℝ, f (x + y) = f x - f y) : 
  ∀ x : ℝ, f x = 0 :=
by
  sorry

end unique_zero_function_l1767_176702


namespace negation_of_existential_l1767_176764

theorem negation_of_existential (x : ℝ) : ¬(∃ x : ℝ, x^2 - 2 * x + 3 > 0) ↔ ∀ x : ℝ, x^2 - 2 * x + 3 ≤ 0 := 
by
  sorry

end negation_of_existential_l1767_176764


namespace cubing_identity_l1767_176730

theorem cubing_identity (x : ℝ) (hx : x + 1/x = -3) : x^3 + 1/x^3 = -18 := by
  sorry

end cubing_identity_l1767_176730


namespace c_geq_one_l1767_176766

variable {α : Type*} [LinearOrderedField α]

theorem c_geq_one
  (a : ℕ → α)
  (c : α)
  (h1 : ∀ i : ℕ, 0 < i → 0 ≤ a i ∧ a i ≤ c)
  (h2 : ∀ i j : ℕ, 0 < i → 0 < j → i ≠ j → |a i - a j| ≥ 1 / (i + j)) :
  c ≥ 1 :=
sorry

end c_geq_one_l1767_176766


namespace average_age_of_inhabitants_l1767_176737

theorem average_age_of_inhabitants (H M : ℕ) (avg_age_men avg_age_women : ℕ)
  (ratio_condition : 2 * M = 3 * H)
  (men_avg_age_condition : avg_age_men = 37)
  (women_avg_age_condition : avg_age_women = 42) :
  ((H * 37) + (M * 42)) / (H + M) = 40 :=
by
  sorry

end average_age_of_inhabitants_l1767_176737


namespace cricket_average_score_l1767_176778

theorem cricket_average_score (A : ℝ)
    (h1 : 3 * 30 = 90)
    (h2 : 5 * 26 = 130) :
    2 * A + 90 = 130 → A = 20 :=
by
  intros h
  linarith

end cricket_average_score_l1767_176778


namespace find_percentage_of_male_students_l1767_176791

def percentage_of_male_students (M F : ℝ) : Prop :=
  M + F = 1 ∧ 0.40 * M + 0.60 * F = 0.52

theorem find_percentage_of_male_students (M F : ℝ) (h1 : M + F = 1) (h2 : 0.40 * M + 0.60 * F = 0.52) : M = 0.40 :=
by
  sorry

end find_percentage_of_male_students_l1767_176791


namespace range_of_y_l1767_176772

-- Define the vectors
def a : ℝ × ℝ := (1, -2)
def b (y : ℝ) : ℝ × ℝ := (4, y)

-- Define the vector sum
def a_plus_b (y : ℝ) : ℝ × ℝ := (a.1 + (b y).1, a.2 + (b y).2)

-- Define the dot product
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Prove the angle between a and a + b is acute and y ≠ -8
theorem range_of_y (y : ℝ) :
  (dot_product a (a_plus_b y) > 0) ↔ (y < 4.5 ∧ y ≠ -8) :=
by
  sorry

end range_of_y_l1767_176772


namespace alternating_sum_l1767_176731

theorem alternating_sum : 
  (1 - 3 + 5 - 7 + 9 - 11 + 13 - 15 + 17 - 19 + 21 - 23 + 25 - 27 + 29 - 31 + 33 - 35 + 37 - 39 + 41 = 21) :=
by
  sorry

end alternating_sum_l1767_176731


namespace equation_has_one_solution_l1767_176750

theorem equation_has_one_solution : ∀ x : ℝ, x - 6 / (x - 2) = 4 - 6 / (x - 2) ↔ x = 4 :=
by {
  -- proof goes here
  sorry
}

end equation_has_one_solution_l1767_176750


namespace total_units_in_building_l1767_176783

theorem total_units_in_building (x y : ℕ) (cost_1_bedroom cost_2_bedroom total_cost : ℕ)
  (h1 : cost_1_bedroom = 360) (h2 : cost_2_bedroom = 450)
  (h3 : total_cost = 4950) (h4 : y = 7) (h5 : total_cost = cost_1_bedroom * x + cost_2_bedroom * y) :
  x + y = 12 :=
sorry

end total_units_in_building_l1767_176783


namespace combinedAverageAge_l1767_176726

-- Definitions
def numFifthGraders : ℕ := 50
def avgAgeFifthGraders : ℕ := 10
def numParents : ℕ := 75
def avgAgeParents : ℕ := 40

-- Calculation of total ages
def totalAgeFifthGraders := numFifthGraders * avgAgeFifthGraders
def totalAgeParents := numParents * avgAgeParents
def combinedTotalAge := totalAgeFifthGraders + totalAgeParents

-- Calculation of total number of individuals
def totalIndividuals := numFifthGraders + numParents

-- The claim to prove
theorem combinedAverageAge : 
  combinedTotalAge / totalIndividuals = 28 := by
  -- Skipping the proof details.
  sorry

end combinedAverageAge_l1767_176726


namespace awards_distribution_l1767_176752

theorem awards_distribution :
  let num_awards := 6
  let num_students := 3 
  let min_awards_per_student := 2
  (num_awards = 6 ∧ num_students = 3 ∧ min_awards_per_student = 2) →
  ∃ (ways : ℕ), ways = 15 :=
by
  sorry

end awards_distribution_l1767_176752


namespace image_of_center_after_transformations_l1767_176735

-- Define the initial center of circle C
def initial_center : ℝ × ℝ := (3, -4)

-- Define a function to reflect a point across the x-axis
def reflect_x_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

-- Define a function to translate a point by some units left
def translate_left (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ :=
  (p.1 - d, p.2)

-- Define the final coordinates after transformations
def final_center : ℝ × ℝ :=
  translate_left (reflect_x_axis initial_center) 5

-- The theorem to prove
theorem image_of_center_after_transformations :
  final_center = (-2, 4) :=
by
  sorry

end image_of_center_after_transformations_l1767_176735


namespace f_800_l1767_176714

noncomputable def f : ℕ → ℕ := sorry

axiom axiom1 : ∀ x y : ℕ, 0 < x → 0 < y → f (x * y) = f x + f y
axiom axiom2 : f 10 = 10
axiom axiom3 : f 40 = 14

theorem f_800 : f 800 = 26 :=
by
  -- Apply the conditions here
  sorry

end f_800_l1767_176714


namespace total_sleep_time_is_correct_l1767_176755

-- Define the sleeping patterns of the animals
def cougar_sleep_even_days : ℕ := 4
def cougar_sleep_odd_days : ℕ := 6
def zebra_sleep_more : ℕ := 2

-- Define the distribution of even and odd days in a week
def even_days_in_week : ℕ := 3
def odd_days_in_week : ℕ := 4

-- Define the total weekly sleep time for the cougar
def cougar_total_weekly_sleep : ℕ := 
  (cougar_sleep_even_days * even_days_in_week) + 
  (cougar_sleep_odd_days * odd_days_in_week)

-- Define the total weekly sleep time for the zebra
def zebra_total_weekly_sleep : ℕ := 
  ((cougar_sleep_even_days + zebra_sleep_more) * even_days_in_week) + 
  ((cougar_sleep_odd_days + zebra_sleep_more) * odd_days_in_week)

-- Define the total weekly sleep time for both the cougar and the zebra
def total_weekly_sleep : ℕ := 
  cougar_total_weekly_sleep + zebra_total_weekly_sleep

-- Prove that the total weekly sleep time for both animals is 86 hours
theorem total_sleep_time_is_correct : total_weekly_sleep = 86 :=
by
  -- skipping proof
  sorry

end total_sleep_time_is_correct_l1767_176755


namespace find_B_and_C_l1767_176734

def values_of_B_and_C (B C : ℤ) : Prop :=
  5 * B - 3 = 32 ∧ 2 * B + 2 * C = 18

theorem find_B_and_C : ∃ B C : ℤ, values_of_B_and_C B C ∧ B = 7 ∧ C = 2 := by
  sorry

end find_B_and_C_l1767_176734


namespace solve_for_k_l1767_176739

theorem solve_for_k (k : ℝ) : (∀ x : ℝ, 3 * (5 + k * x) = 15 * x + 15) ↔ k = 5 :=
  sorry

end solve_for_k_l1767_176739


namespace remainder_7n_mod_4_l1767_176742

theorem remainder_7n_mod_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 := by
  sorry

end remainder_7n_mod_4_l1767_176742


namespace meaningful_expression_l1767_176701

theorem meaningful_expression (m : ℝ) :
  (2 - m ≥ 0) ∧ (m + 2 ≠ 0) ↔ (m ≤ 2 ∧ m ≠ -2) :=
by
  sorry

end meaningful_expression_l1767_176701


namespace no_four_digit_numbers_divisible_by_11_l1767_176704

theorem no_four_digit_numbers_divisible_by_11 (a b c d : ℕ) :
  (a + b + c + d = 9) ∧ ((a + c) - (b + d)) % 11 = 0 → false :=
by
  sorry

end no_four_digit_numbers_divisible_by_11_l1767_176704


namespace max_earth_to_sun_distance_l1767_176706

-- Define the semi-major axis a and semi-focal distance c
def semi_major_axis : ℝ := 1.5 * 10^8
def semi_focal_distance : ℝ := 3 * 10^6

-- Define the maximum distance from the Earth to the Sun
def max_distance (a c : ℝ) : ℝ := a + c

-- Define the Lean statement to be proved
theorem max_earth_to_sun_distance :
  max_distance semi_major_axis semi_focal_distance = 1.53 * 10^8 :=
by
  -- skipping the proof for now
  sorry

end max_earth_to_sun_distance_l1767_176706


namespace Jake_initial_balloons_l1767_176732

theorem Jake_initial_balloons (J : ℕ) 
  (h1 : 6 = (J + 3) + 1) : 
  J = 2 :=
by
  sorry

end Jake_initial_balloons_l1767_176732


namespace range_of_t_l1767_176716

theorem range_of_t (x y : ℝ) (h1 : 1 ≤ x) (h2 : x ≤ y) (h3 : x + y > 1) (h4 : x + 1 > y) (h5 : y + 1 > x) :
    1 ≤ max (1 / x) (max (x / y) y) * min (1 / x) (min (x / y) y) ∧
    max (1 / x) (max (x / y) y) * min (1 / x) (min (x / y) y) < (1 + Real.sqrt 5) / 2 := 
sorry

end range_of_t_l1767_176716


namespace intersection_points_of_curve_with_axes_l1767_176758

theorem intersection_points_of_curve_with_axes :
  (∃ t : ℝ, (-2 + 5 * t = 0) ∧ (1 - 2 * t = 1/5)) ∧
  (∃ t : ℝ, (1 - 2 * t = 0) ∧ (-2 + 5 * t = 1/2)) :=
by {
  -- Proving the intersection points with the coordinate axes
  sorry
}

end intersection_points_of_curve_with_axes_l1767_176758


namespace chocolate_bars_in_large_box_l1767_176715

theorem chocolate_bars_in_large_box
  (small_box_count : ℕ) (chocolate_per_small_box : ℕ)
  (h1 : small_box_count = 20)
  (h2 : chocolate_per_small_box = 25) :
  (small_box_count * chocolate_per_small_box) = 500 :=
by
  sorry

end chocolate_bars_in_large_box_l1767_176715


namespace two_distinct_solutions_exist_l1767_176770

theorem two_distinct_solutions_exist :
  ∃ (a1 b1 c1 d1 e1 a2 b2 c2 d2 e2 : ℕ), 
    1 ≤ a1 ∧ a1 ≤ 9 ∧ 1 ≤ b1 ∧ b1 ≤ 9 ∧ 1 ≤ c1 ∧ c1 ≤ 9 ∧ 1 ≤ d1 ∧ d1 ≤ 9 ∧ 1 ≤ e1 ∧ e1 ≤ 9 ∧
    1 ≤ a2 ∧ a2 ≤ 9 ∧ 1 ≤ b2 ∧ b2 ≤ 9 ∧ 1 ≤ c2 ∧ c2 ≤ 9 ∧ 1 ≤ d2 ∧ d2 ≤ 9 ∧ 1 ≤ e2 ∧ e2 ≤ 9 ∧
    (b1 - d1 = 2) ∧ (d1 - a1 = 3) ∧ (a1 - c1 = 1) ∧
    (b2 - d2 = 2) ∧ (d2 - a2 = 3) ∧ (a2 - c2 = 1) ∧
    ¬ (a1 = a2 ∧ b1 = b2 ∧ c1 = c2 ∧ d1 = d2 ∧ e1 = e2) :=
by
  sorry

end two_distinct_solutions_exist_l1767_176770


namespace probability_of_earning_2400_l1767_176708

noncomputable def spinner_labels := ["Bankrupt", "$700", "$900", "$200", "$3000", "$800"]
noncomputable def total_possibilities := (spinner_labels.length : ℕ) ^ 3
noncomputable def favorable_outcomes := 6

theorem probability_of_earning_2400 :
  (favorable_outcomes : ℚ) / total_possibilities = 1 / 36 := by
  sorry

end probability_of_earning_2400_l1767_176708


namespace solve_arithmetic_series_l1767_176763

theorem solve_arithmetic_series : 
  25^2 - 23^2 + 21^2 - 19^2 + 17^2 - 15^2 + 13^2 - 11^2 + 9^2 - 7^2 + 5^2 - 3^2 + 1^2 = 338 :=
by sorry

end solve_arithmetic_series_l1767_176763


namespace triangle_equilateral_of_angle_and_side_sequences_l1767_176723

theorem triangle_equilateral_of_angle_and_side_sequences 
  (A B C : ℝ) (a b c : ℝ) 
  (h_angles_arith_seq: B = (A + C) / 2)
  (h_sides_geom_seq : b^2 = a * c) 
  (h_sum_angles : A + B + C = 180) 
  (h_pos_angles : 0 < A ∧ 0 < B ∧ 0 < C) 
  (h_pos_sides : 0 < a ∧ 0 < b ∧ 0 < c) :
  A = 60 ∧ B = 60 ∧ C = 60 ∧ a = b ∧ b = c :=
by
  sorry

end triangle_equilateral_of_angle_and_side_sequences_l1767_176723


namespace angle_in_quadrants_l1767_176781

theorem angle_in_quadrants (α : ℝ) (hα : 0 < α ∧ α < π / 2) (k : ℤ) :
  (∃ i : ℤ, k = 2 * i + 1 ∧ π < (2 * i + 1) * π + α ∧ (2 * i + 1) * π + α < 3 * π / 2) ∨
  (∃ i : ℤ, k = 2 * i ∧ 0 < 2 * i * π + α ∧ 2 * i * π + α < π / 2) :=
sorry

end angle_in_quadrants_l1767_176781


namespace part1_part2_l1767_176746

def f (x : ℝ) : ℝ := |2 * x + 1| - |x - 1|

theorem part1 : {x : ℝ | f x < 2} = {x : ℝ | -4 < x ∧ x < 2 / 3} :=
by
  sorry

theorem part2 : ∀ a : ℝ, (∃ x : ℝ, f x ≤ a - a^2 / 2) → (-1 ≤ a ∧ a ≤ 3) :=
by
  sorry

end part1_part2_l1767_176746


namespace smallest_two_digit_multiple_of_17_smallest_four_digit_multiple_of_17_l1767_176741

theorem smallest_two_digit_multiple_of_17 : ∃ m, 10 ≤ m ∧ m < 100 ∧ 17 ∣ m ∧ ∀ n, 10 ≤ n ∧ n < 100 ∧ 17 ∣ n → m ≤ n :=
by
  sorry

theorem smallest_four_digit_multiple_of_17 : ∃ m, 1000 ≤ m ∧ m < 10000 ∧ 17 ∣ m ∧ ∀ n, 1000 ≤ n ∧ n < 10000 ∧ 17 ∣ n → m ≤ n :=
by
  sorry

end smallest_two_digit_multiple_of_17_smallest_four_digit_multiple_of_17_l1767_176741


namespace wheels_travel_distance_l1767_176793

noncomputable def total_horizontal_distance (R₁ R₂ : ℝ) : ℝ :=
  2 * Real.pi * R₁ + 2 * Real.pi * R₂

theorem wheels_travel_distance (R₁ R₂ : ℝ) (h₁ : R₁ = 2) (h₂ : R₂ = 3) :
  total_horizontal_distance R₁ R₂ = 10 * Real.pi :=
by
  rw [total_horizontal_distance, h₁, h₂]
  sorry

end wheels_travel_distance_l1767_176793


namespace value_of_expression_l1767_176777

variable {a b : ℝ}
variables (h1 : ∀ x, 3 * x^2 + 9 * x - 18 = 0 → x = a ∨ x = b)

theorem value_of_expression : (3 * a - 2) * (6 * b - 9) = 27 :=
by
  sorry

end value_of_expression_l1767_176777


namespace sequence_a_n_is_n_l1767_176707

-- Definitions and statements based on the conditions
def sequence_cond (a : ℕ → ℕ) (n : ℕ) : ℕ := 
1 / 2 * (a n) ^ 2 + n / 2

theorem sequence_a_n_is_n :
  ∀ (a : ℕ → ℕ), (∀ n, n > 0 → ∃ (S_n : ℕ), S_n = sequence_cond a n) → 
  (∀ n, n > 0 → a n = n) :=
by
  sorry

end sequence_a_n_is_n_l1767_176707


namespace right_triangle_area_inscribed_3_4_l1767_176724

theorem right_triangle_area_inscribed_3_4 (r1 r2: ℝ) (h1 : r1 = 3) (h2 : r2 = 4) : 
  ∃ (S: ℝ), S = 150 :=
by
  sorry

end right_triangle_area_inscribed_3_4_l1767_176724


namespace cos_A_eq_sqrt3_div3_of_conditions_l1767_176771

noncomputable def given_conditions
  (a b c : ℝ)
  (A B C : ℝ)
  (h1 : (Real.sqrt 3 * b - c) * Real.cos A = a * Real.cos C)
  (h2 : a ≠ 0) 
  (h3 : b ≠ 0) 
  (h4 : c ≠ 0) 
  (h5 : A ≠ 0) 
  (h6 : B ≠ 0) 
  (h7 : C ≠ 0) : Prop :=
  (Real.cos A = Real.sqrt 3 / 3)

theorem cos_A_eq_sqrt3_div3_of_conditions
  (a b c : ℝ)
  (A B C : ℝ)
  (h1 : (Real.sqrt 3 * b - c) * Real.cos A = a * Real.cos C)
  (h2 : a ≠ 0) 
  (h3 : b ≠ 0) 
  (h4 : c ≠ 0) 
  (h5 : A ≠ 0) 
  (h6 : B ≠ 0) 
  (h7 : C ≠ 0) :
  Real.cos A = Real.sqrt 3 / 3 :=
sorry

end cos_A_eq_sqrt3_div3_of_conditions_l1767_176771


namespace greatest_prime_factor_of_221_l1767_176794

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def greatest_prime_factor (n : ℕ) (p : ℕ) : Prop := 
  is_prime p ∧ p ∣ n ∧ ∀ q : ℕ, is_prime q → q ∣ n → q ≤ p

theorem greatest_prime_factor_of_221 : greatest_prime_factor 221 17 := by
  sorry

end greatest_prime_factor_of_221_l1767_176794


namespace vectors_parallel_l1767_176720

-- Let s and n be the direction vector and normal vector respectively
def s : ℝ × ℝ × ℝ := (2, 1, 1)
def n : ℝ × ℝ × ℝ := (-4, -2, -2)

-- Statement that vectors s and n are parallel
theorem vectors_parallel : ∃ (k : ℝ), n = (k • s) := by
  use -2
  simp [s, n]
  sorry

end vectors_parallel_l1767_176720


namespace min_sum_ab_l1767_176796

theorem min_sum_ab (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_ab : a * b = a + b + 3) : 
  a + b ≥ 6 := 
sorry

end min_sum_ab_l1767_176796


namespace investment_amount_correct_l1767_176759

noncomputable def investment_problem : Prop :=
  let initial_investment_rubles : ℝ := 10000
  let initial_exchange_rate : ℝ := 50
  let annual_return_rate : ℝ := 0.12
  let end_year_exchange_rate : ℝ := 80
  let currency_conversion_commission : ℝ := 0.05
  let broker_profit_commission_rate : ℝ := 0.3

  -- Computations
  let initial_investment_dollars := initial_investment_rubles / initial_exchange_rate
  let profit_dollars := initial_investment_dollars * annual_return_rate
  let total_dollars := initial_investment_dollars + profit_dollars
  let broker_commission_dollars := profit_dollars * broker_profit_commission_rate
  let post_commission_dollars := total_dollars - broker_commission_dollars
  let amount_in_rubles_before_conversion_commission := post_commission_dollars * end_year_exchange_rate
  let conversion_commission := amount_in_rubles_before_conversion_commission * currency_conversion_commission
  let final_amount_rubles := amount_in_rubles_before_conversion_commission - conversion_commission

  -- Proof goal
  final_amount_rubles = 16476.8

theorem investment_amount_correct : investment_problem := by {
  sorry
}

end investment_amount_correct_l1767_176759


namespace BKINGTON_appears_first_on_eighth_line_l1767_176786

-- Define the cycle lengths for letters and digits
def cycle_letters : ℕ := 8
def cycle_digits : ℕ := 4

-- Define the problem statement
theorem BKINGTON_appears_first_on_eighth_line :
  Nat.lcm cycle_letters cycle_digits = 8 := by
  sorry

end BKINGTON_appears_first_on_eighth_line_l1767_176786


namespace problem_solution_l1767_176717

def grid_side : ℕ := 4
def square_size : ℝ := 2
def ellipse_major_axis : ℝ := 4
def ellipse_minor_axis : ℝ := 2
def circle_radius : ℝ := 1
def num_circles : ℕ := 3

noncomputable def grid_area : ℝ :=
  (grid_side * grid_side) * (square_size * square_size)

noncomputable def circle_area : ℝ :=
  num_circles * (Real.pi * (circle_radius ^ 2))

noncomputable def ellipse_area : ℝ :=
  Real.pi * (ellipse_major_axis / 2) * (ellipse_minor_axis / 2)

noncomputable def visible_shaded_area (A B : ℝ) : Prop :=
  grid_area = A - B * Real.pi

theorem problem_solution : ∃ A B, visible_shaded_area A B ∧ (A + B = 69) :=
by
  sorry

end problem_solution_l1767_176717


namespace value_of_V3_l1767_176792

def f (x : ℝ) : ℝ := 3 * x^5 + 8 * x^4 - 3 * x^3 + 5 * x^2 + 12 * x - 6

def horner (a : ℝ) : ℝ :=
  let V0 := 3
  let V1 := V0 * a + 8
  let V2 := V1 * a - 3
  let V3 := V2 * a + 5
  V3

theorem value_of_V3 : horner 2 = 55 :=
  by
    simp [horner]
    sorry

end value_of_V3_l1767_176792


namespace unicorn_rope_problem_l1767_176774

/-
  A unicorn is tethered by a 24-foot golden rope to the base of a sorcerer's cylindrical tower
  whose radius is 10 feet. The rope is attached to the tower at ground level and to the unicorn
  at a height of 6 feet. The unicorn has pulled the rope taut, and the end of the rope is 6 feet
  from the nearest point on the tower.
  The length of the rope that is touching the tower is given as:
  ((96 - sqrt(36)) / 6) feet,
  where 96, 36, and 6 are positive integers, and 6 is prime.
  We need to prove that the sum of these integers is 138.
-/
theorem unicorn_rope_problem : 
  let d := 96
  let e := 36
  let f := 6
  d + e + f = 138 := by
  sorry

end unicorn_rope_problem_l1767_176774


namespace sufficient_condition_implication_l1767_176744

theorem sufficient_condition_implication {A B : Prop}
  (h : (¬A → ¬B) ∧ (B → A)): (B → A) ∧ (A → ¬¬A ∧ ¬A → ¬B) :=
by
  -- Note: We would provide the proof here normally, but we skip it for now.
  sorry

end sufficient_condition_implication_l1767_176744


namespace possible_k_values_l1767_176780

theorem possible_k_values :
  (∃ k b a c : ℤ, b = 2020 + k ∧ a * (c ^ 2) = (2020 + k) ∧ 
  (k = -404 ∨ k = -1010)) :=
sorry

end possible_k_values_l1767_176780


namespace factor_x4_minus_64_l1767_176762

theorem factor_x4_minus_64 :
  ∀ x : ℝ, (x^4 - 64 = (x - 2 * Real.sqrt 2) * (x + 2 * Real.sqrt 2) * (x^2 + 8)) :=
by
  intro x
  sorry

end factor_x4_minus_64_l1767_176762


namespace year_2023_not_lucky_l1767_176761

def is_valid_date (month day year : ℕ) : Prop :=
  month * day = year % 100

def is_lucky_year (year : ℕ) : Prop :=
  ∃ month day, month ≤ 12 ∧ day ≤ 31 ∧ is_valid_date month day year

theorem year_2023_not_lucky : ¬ is_lucky_year 2023 :=
by sorry

end year_2023_not_lucky_l1767_176761


namespace man_completion_time_l1767_176768

theorem man_completion_time (w_time : ℕ) (efficiency_increase : ℚ) (m_time : ℕ) :
  w_time = 40 → efficiency_increase = 1.25 → m_time = (w_time : ℚ) / efficiency_increase → m_time = 32 :=
by
  sorry

end man_completion_time_l1767_176768


namespace greatest_a_l1767_176728

theorem greatest_a (a : ℤ) (h_pos : a > 0) : 
  (∀ x : ℤ, (x^2 + a * x = -30) → (a = 31)) :=
by {
  sorry
}

end greatest_a_l1767_176728


namespace box_width_l1767_176709

theorem box_width (W : ℕ) (h₁ : 15 * W * 13 = 3120) : W = 16 := by
  sorry

end box_width_l1767_176709


namespace polynomial_A_l1767_176799

theorem polynomial_A (A a : ℝ) (h : A * (a + 1) = a^2 - 1) : A = a - 1 :=
sorry

end polynomial_A_l1767_176799


namespace minimum_value_of_reciprocal_sum_l1767_176767

theorem minimum_value_of_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h : 2 * a * (-1) - b * 2 + 2 = 0) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 2 * a * (-1) - b * 2 + 2 = 0 ∧ (a + b = 1) ∧ (a = 1/2 ∧ b = 1/2) ∧ (1/a + 1/b = 4) :=
by
  sorry

end minimum_value_of_reciprocal_sum_l1767_176767


namespace total_flour_used_l1767_176725

theorem total_flour_used :
  let wheat_flour := 0.2
  let white_flour := 0.1
  let rye_flour := 0.15
  let almond_flour := 0.05
  let oat_flour := 0.1
  wheat_flour + white_flour + rye_flour + almond_flour + oat_flour = 0.6 :=
by
  sorry

end total_flour_used_l1767_176725


namespace proof_a_eq_x_and_b_eq_x_pow_x_l1767_176700

theorem proof_a_eq_x_and_b_eq_x_pow_x
  {a b x : ℕ}
  (h_pos_a : 0 < a)
  (h_pos_b : 0 < b)
  (h_pos_x : 0 < x)
  (h : x^(a + b) = a^b * b) :
  a = x ∧ b = x^x := 
by
  sorry

end proof_a_eq_x_and_b_eq_x_pow_x_l1767_176700
