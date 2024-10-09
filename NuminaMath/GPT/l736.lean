import Mathlib

namespace problem_a_add_b_eq_five_l736_73687

variable {a b : ℝ}

theorem problem_a_add_b_eq_five
  (h1 : ∀ x, -2 < x ∧ x < 3 → ax^2 + x + b > 0)
  (h2 : a < 0) :
  a + b = 5 :=
sorry

end problem_a_add_b_eq_five_l736_73687


namespace find_a_l736_73656

/-- Given function -/
def f (x: ℝ) : ℝ := (x + 1)^2 - 2 * (x + 1)

/-- Problem statement -/
theorem find_a (a : ℝ) (h : f a = 3) : a = 2 ∨ a = -2 := 
by
  sorry

end find_a_l736_73656


namespace intersection_of_sets_l736_73627

def set_a : Set ℝ := { x | -x^2 + 2 * x ≥ 0 }
def set_b : Set ℝ := { x | -1 ≤ x ∧ x ≤ 1 }
def set_intersection : Set ℝ := { x | 0 ≤ x ∧ x ≤ 1 }

theorem intersection_of_sets : (set_a ∩ set_b) = set_intersection := by 
  sorry

end intersection_of_sets_l736_73627


namespace find_element_atomic_mass_l736_73698

-- Define the atomic mass of bromine
def atomic_mass_br : ℝ := 79.904

-- Define the molecular weight of the compound
def molecular_weight : ℝ := 267

-- Define the number of bromine atoms in the compound (assuming n = 1)
def n : ℕ := 1

-- Define the atomic mass of the unknown element X
def atomic_mass_x : ℝ := molecular_weight - n * atomic_mass_br

-- State the theorem to prove
theorem find_element_atomic_mass : atomic_mass_x = 187.096 :=
by
  -- placeholder for the proof
  sorry

end find_element_atomic_mass_l736_73698


namespace mason_father_age_l736_73668

theorem mason_father_age
  (Mason_age : ℕ) 
  (Sydney_age : ℕ) 
  (Father_age : ℕ)
  (h1 : Mason_age = 20)
  (h2 : Sydney_age = 3 * Mason_age)
  (h3 : Father_age = Sydney_age + 6) :
  Father_age = 66 :=
by
  sorry

end mason_father_age_l736_73668


namespace cost_of_items_l736_73631

theorem cost_of_items (M R F : ℝ)
  (h1 : 10 * M = 24 * R) 
  (h2 : F = 2 * R) 
  (h3 : F = 20.50) : 
  4 * M + 3 * R + 5 * F = 231.65 := 
by
  sorry

end cost_of_items_l736_73631


namespace solve_for_q_l736_73641

theorem solve_for_q (p q : ℚ) (h1 : 5 * p + 3 * q = 7) (h2 : 3 * p + 5 * q = 8) : q = 19 / 16 :=
by
  sorry

end solve_for_q_l736_73641


namespace jesses_room_total_area_l736_73675

-- Define the dimensions of the first rectangular part
def length1 : ℕ := 12
def width1 : ℕ := 8

-- Define the dimensions of the second rectangular part
def length2 : ℕ := 6
def width2 : ℕ := 4

-- Define the areas of both parts
def area1 : ℕ := length1 * width1
def area2 : ℕ := length2 * width2

-- Define the total area
def total_area : ℕ := area1 + area2

-- Statement of the theorem we want to prove
theorem jesses_room_total_area : total_area = 120 :=
by
  -- We would provide the proof here
  sorry

end jesses_room_total_area_l736_73675


namespace boy_present_age_l736_73658

-- Define the boy's present age
variable (x : ℤ)

-- Conditions from the problem statement
def condition_one : Prop :=
  x + 4 = 2 * (x - 6)

-- Prove that the boy's present age is 16
theorem boy_present_age (h : condition_one x) : x = 16 := 
sorry

end boy_present_age_l736_73658


namespace student_marks_l736_73694

theorem student_marks (x : ℕ) :
  let total_questions := 60
  let correct_answers := 38
  let wrong_answers := total_questions - correct_answers
  let total_marks := 130
  let marks_from_correct := correct_answers * x
  let marks_lost := wrong_answers * 1
  let net_marks := marks_from_correct - marks_lost
  net_marks = total_marks → x = 4 :=
by
  intros
  sorry

end student_marks_l736_73694


namespace tree_circumference_inequality_l736_73623

theorem tree_circumference_inequality (x : ℝ) : 
  (∀ t : ℝ, t = 10 + 3 * x ∧ t > 90 → x > 80 / 3) :=
by
  intro t ht
  obtain ⟨h_t_eq, h_t_gt_90⟩ := ht
  linarith

end tree_circumference_inequality_l736_73623


namespace binomial_inequality_l736_73611

theorem binomial_inequality (n : ℕ) (x : ℝ) (h1 : 2 ≤ n) (h2 : |x| < 1) : 
  (1 - x)^n + (1 + x)^n < 2^n := 
by 
  sorry

end binomial_inequality_l736_73611


namespace smallest_integer_value_of_m_l736_73664

theorem smallest_integer_value_of_m (x y m : ℝ) 
  (h1 : 3*x + y = m + 8) 
  (h2 : 2*x + 2*y = 2*m + 5) 
  (h3 : x - y < 1) : 
  m >= 3 := 
sorry

end smallest_integer_value_of_m_l736_73664


namespace function_passes_through_fixed_point_l736_73691

theorem function_passes_through_fixed_point (a : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) : (2 - a^(0 : ℝ) = 1) :=
by
  sorry

end function_passes_through_fixed_point_l736_73691


namespace regular_discount_rate_l736_73673

theorem regular_discount_rate (MSRP : ℝ) (s : ℝ) (sale_price : ℝ) (d : ℝ) :
  MSRP = 35 ∧ s = 0.20 ∧ sale_price = 19.6 → d = 0.3 :=
by
  intro h
  sorry

end regular_discount_rate_l736_73673


namespace value_of_a_l736_73600

theorem value_of_a (a : ℝ) : 
  ({2, 3} : Set ℝ) ⊆ ({1, 2, a} : Set ℝ) → a = 3 :=
by
  sorry

end value_of_a_l736_73600


namespace find_y_l736_73652

def vectors_orthogonal_condition (y : ℝ) : Prop :=
  (1 * -2) + (-3 * y) + (-4 * -1) = 0

theorem find_y : vectors_orthogonal_condition (2 / 3) :=
by
  sorry

end find_y_l736_73652


namespace compute_2018_square_123_Delta_4_l736_73650

namespace custom_operations

def Delta (a b : ℕ) : ℕ := a * 10 ^ b + b
def Square (a b : ℕ) : ℕ := a * 10 + b

theorem compute_2018_square_123_Delta_4 : Square 2018 (Delta 123 4) = 1250184 :=
by
  sorry

end custom_operations

end compute_2018_square_123_Delta_4_l736_73650


namespace binary_to_decimal_l736_73647

theorem binary_to_decimal (b : ℕ) (h : b = 2^3 + 2^2 + 0 * 2^1 + 2^0) : b = 13 :=
by {
  -- proof is omitted
  sorry
}

end binary_to_decimal_l736_73647


namespace value_of_y_l736_73671

theorem value_of_y (x y : ℤ) (h1 : x^2 - 3 * x + 6 = y + 2) (h2 : x = -8) : y = 92 :=
by
  sorry

end value_of_y_l736_73671


namespace complex_equation_square_sum_l736_73630

-- Lean 4 statement of the mathematical proof problem
theorem complex_equation_square_sum (a b : ℝ) (i : ℂ) (h : i^2 = -1) 
    (h1 : (a - 2 * i) * i = b - i) : a^2 + b^2 = 5 := by
  sorry

end complex_equation_square_sum_l736_73630


namespace vanya_number_l736_73669

def S (n : ℕ) : ℕ := n.digits 10 |>.sum

theorem vanya_number:
  (2014 + S 2014 = 2021) ∧ (1996 + S 1996 = 2021) := by
  sorry

end vanya_number_l736_73669


namespace repeating_decimal_exceeds_decimal_representation_l736_73663

noncomputable def repeating_decimal : ℚ := 71 / 99
def decimal_representation : ℚ := 71 / 100

theorem repeating_decimal_exceeds_decimal_representation :
  repeating_decimal - decimal_representation = 71 / 9900 := by
  sorry

end repeating_decimal_exceeds_decimal_representation_l736_73663


namespace minimum_value_of_polynomial_l736_73633

def polynomial (a b : ℝ) : ℝ := 2 * a^2 - 8 * a * b + 17 * b^2 - 16 * a + 4 * b + 1999

theorem minimum_value_of_polynomial : ∃ (a b : ℝ), polynomial a b = 1947 :=
by
  sorry

end minimum_value_of_polynomial_l736_73633


namespace total_number_of_letters_l736_73604

def jonathan_first_name_letters : Nat := 8
def jonathan_surname_letters : Nat := 10
def sister_first_name_letters : Nat := 5
def sister_surname_letters : Nat := 10

theorem total_number_of_letters : 
  jonathan_first_name_letters + jonathan_surname_letters + sister_first_name_letters + sister_surname_letters = 33 := 
by 
  sorry

end total_number_of_letters_l736_73604


namespace range_of_x2_y2_l736_73618

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - x^4

theorem range_of_x2_y2 (x y : ℝ) (h : x^2 + y^2 = 2 * x) : 
  0 ≤ x^2 * y^2 ∧ x^2 * y^2 ≤ 27 / 16 :=
sorry

end range_of_x2_y2_l736_73618


namespace value_three_std_devs_less_than_mean_l736_73616

-- Define the given conditions as constants.
def mean : ℝ := 16.2
def std_dev : ℝ := 2.3

-- Translate the question into a proof statement.
theorem value_three_std_devs_less_than_mean : mean - 3 * std_dev = 9.3 :=
by sorry

end value_three_std_devs_less_than_mean_l736_73616


namespace discriminant_divisible_l736_73612

theorem discriminant_divisible (a b: ℝ) (n: ℤ) (h: (∃ x1 x2: ℝ, 2018*x1^2 + a*x1 + b = 0 ∧ 2018*x2^2 + a*x2 + b = 0 ∧ x1 - x2 = n)): 
  ∃ k: ℤ, a^2 - 4 * 2018 * b = 2018^2 * k := 
by 
  sorry

end discriminant_divisible_l736_73612


namespace cash_refund_per_bottle_l736_73609

-- Define the constants based on the conditions
def bottles_per_month : ℕ := 15
def cost_per_bottle : ℝ := 3.0
def bottles_can_buy_with_refund : ℕ := 6
def months_per_year : ℕ := 12

-- Define the total number of bottles consumed in a year
def total_bottles_per_year : ℕ := bottles_per_month * months_per_year

-- Define the total refund in dollars after 1 year
def total_refund_amount : ℝ := bottles_can_buy_with_refund * cost_per_bottle

-- Define the statement we need to prove
theorem cash_refund_per_bottle :
  total_refund_amount / total_bottles_per_year = 0.10 :=
by
  -- This is where the steps would be completed to prove the theorem
  sorry

end cash_refund_per_bottle_l736_73609


namespace dot_product_equivalence_l736_73695

variable (a : ℝ × ℝ) 
variable (b : ℝ × ℝ)

-- Given conditions
def condition_1 : Prop := a = (2, 1)
def condition_2 : Prop := a - b = (-1, 2)

-- Goal
theorem dot_product_equivalence (h1 : condition_1 a) (h2 : condition_2 a b) : a.1 * b.1 + a.2 * b.2 = 5 :=
  sorry

end dot_product_equivalence_l736_73695


namespace largest_number_l736_73679

theorem largest_number (a b : ℕ) (hcf_ab : Nat.gcd a b = 42) (h_dvd_a : 42 ∣ a) (h_dvd_b : 42 ∣ b)
  (a_eq : a = 42 * 11) (b_eq : b = 42 * 12) : max a b = 504 := by
  sorry

end largest_number_l736_73679


namespace neither_sufficient_nor_necessary_l736_73628

theorem neither_sufficient_nor_necessary (α β : ℝ) :
  (α + β = 90) ↔ ¬((α + β = 90) ↔ (Real.sin α + Real.sin β > 1)) :=
sorry

end neither_sufficient_nor_necessary_l736_73628


namespace segment_association_l736_73696

theorem segment_association (x y : ℝ) 
  (h1 : ∃ (D : ℝ), ∀ (P : ℝ), abs (P - D) ≤ 5) 
  (h2 : ∃ (D' : ℝ), ∀ (P' : ℝ), abs (P' - D') ≤ 9)
  (h3 : 3 * x - 2 * y = 6) : 
  x + y = 12 := 
by sorry

end segment_association_l736_73696


namespace largest_integral_k_for_real_distinct_roots_l736_73676

theorem largest_integral_k_for_real_distinct_roots :
  ∃ k : ℤ, (k < 9) ∧ (∀ k' : ℤ, k' < 9 → k' ≤ k) :=
sorry

end largest_integral_k_for_real_distinct_roots_l736_73676


namespace cost_of_four_pencils_and_four_pens_l736_73681

def pencil_cost : ℝ := sorry
def pen_cost : ℝ := sorry

axiom h1 : 8 * pencil_cost + 3 * pen_cost = 5.10
axiom h2 : 3 * pencil_cost + 5 * pen_cost = 4.95

theorem cost_of_four_pencils_and_four_pens : 4 * pencil_cost + 4 * pen_cost = 4.488 :=
by
  sorry

end cost_of_four_pencils_and_four_pens_l736_73681


namespace width_rectangular_box_5_cm_l736_73692

theorem width_rectangular_box_5_cm 
  (W : ℕ)
  (h_dim_wooden_box : (8 * 10 * 6 * 100 ^ 3) = 480000000) -- dimensions of the wooden box in cm³
  (h_dim_rectangular_box : (4 * W * 6) = (24 * W)) -- dimensions of the rectangular box in cm³
  (h_max_boxes : 4000000 * (24 * W) = 480000000) -- max number of boxes that fit in the wooden box
: 
  W = 5 := 
by
  sorry

end width_rectangular_box_5_cm_l736_73692


namespace find_local_value_of_7_in_difference_l736_73660

-- Define the local value of 3 in the number 28943712.
def local_value_of_3_in_28943712 : Nat := 30000

-- Define the property that the local value of 7 in a number Y is 7000.
def local_value_of_7 (Y : Nat) : Prop := (Y / 1000 % 10) = 7

-- Define the unknown number X and its difference with local value of 3 in 28943712.
variable (X : Nat)

-- Assumption: The difference between X and local_value_of_3_in_28943712 results in a number whose local value of 7 is 7000.
axiom difference_condition : local_value_of_7 (X - local_value_of_3_in_28943712)

-- The proof problem statement to be solved.
theorem find_local_value_of_7_in_difference : local_value_of_7 (X - local_value_of_3_in_28943712) = true :=
by
  -- Proof is omitted.
  sorry

end find_local_value_of_7_in_difference_l736_73660


namespace geometric_sequence_solution_l736_73684

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ a1 r, ∀ n, a n = a1 * r ^ (n - 1)

theorem geometric_sequence_solution :
  ∀ (a : ℕ → ℝ),
    (geometric_sequence a) →
    (∃ a2 a18, a2 + a18 = -6 ∧ a2 * a18 = 4 ∧ a 2 = a2 ∧ a 18 = a18) →
    a 4 * a 16 + a 10 = 6 :=
by
  sorry

end geometric_sequence_solution_l736_73684


namespace final_lives_equals_20_l736_73621

def initial_lives : ℕ := 30
def lives_lost : ℕ := 12
def bonus_lives : ℕ := 5
def penalty_lives : ℕ := 3

theorem final_lives_equals_20 : (initial_lives - lives_lost + bonus_lives - penalty_lives) = 20 :=
by 
  sorry

end final_lives_equals_20_l736_73621


namespace triangle_tangency_perimeter_l736_73699

def triangle_perimeter (a b c : ℝ) (s : ℝ) (t : ℝ) (u : ℝ) : ℝ :=
  s + t + u

theorem triangle_tangency_perimeter (a b c : ℝ) (D E F : ℝ) (s : ℝ) (t : ℝ) (u : ℝ)
  (h1 : a = 5) (h2 : b = 7) (h3 : c = 8) 
  (h4 : s + t + u = 3) : triangle_perimeter a b c s t u = 3 :=
by
  sorry

end triangle_tangency_perimeter_l736_73699


namespace most_likely_composition_l736_73643

def event_a : Prop := (1 / 3) * (1 / 3) * 2 = (2 / 9)
def event_d : Prop := 2 * (1 / 3 * 1 / 3) = (2 / 9)

theorem most_likely_composition :
  event_a ∧ event_d :=
by sorry

end most_likely_composition_l736_73643


namespace find_m_if_polynomial_is_square_l736_73672

theorem find_m_if_polynomial_is_square (m : ℝ) :
  (∀ x, ∃ k : ℝ, x^2 + 2 * (m - 3) * x + 16 = (x + k)^2) ↔ (m = 7 ∨ m = -1) :=
by
  sorry

end find_m_if_polynomial_is_square_l736_73672


namespace value_of_f_15_l736_73610

def f (n : ℕ) : ℕ := n^2 + 2*n + 19

theorem value_of_f_15 : f 15 = 274 := 
by 
  -- Add proof here
  sorry

end value_of_f_15_l736_73610


namespace find_certain_number_l736_73642

theorem find_certain_number 
  (x : ℝ) 
  (h : ( (x + 2 - 6) * 3 ) / 4 = 3) 
  : x = 8 :=
by
  sorry

end find_certain_number_l736_73642


namespace real_mul_eq_zero_iff_l736_73622

theorem real_mul_eq_zero_iff (a b : ℝ) (h : a * b = 0) : a = 0 ∨ b = 0 :=
sorry

end real_mul_eq_zero_iff_l736_73622


namespace part_a_part_b_l736_73634

-- Part (a) Equivalent Proof Problem
theorem part_a (k : ℤ) : 
  ∃ a b c : ℤ, 3 * k - 2 = a ^ 2 + b ^ 3 + c ^ 3 := 
sorry

-- Part (b) Equivalent Proof Problem
theorem part_b (n : ℤ) : 
  ∃ a b c d : ℤ, n = a ^ 2 + b ^ 3 + c ^ 3 + d ^ 3 := 
sorry

end part_a_part_b_l736_73634


namespace compute_g_neg_101_l736_73674

variable (g : ℝ → ℝ)

def functional_eqn := ∀ x y : ℝ, g (x * y) + 2 * x = x * g y + g x
def g_neg_one := g (-1) = 3
def g_one := g (1) = 1

theorem compute_g_neg_101 (g : ℝ → ℝ)
  (H1 : functional_eqn g)
  (H2 : g_neg_one g)
  (H3 : g_one g) :
  g (-101) = 103 := 
by
  sorry

end compute_g_neg_101_l736_73674


namespace expression_equals_5_l736_73685

def expression_value : ℤ := 8 + 15 / 3 - 2^3

theorem expression_equals_5 : expression_value = 5 :=
by
  sorry

end expression_equals_5_l736_73685


namespace carrots_planted_per_hour_l736_73640

theorem carrots_planted_per_hour (rows plants_per_row hours : ℕ) (h1 : rows = 400) (h2 : plants_per_row = 300) (h3 : hours = 20) :
  (rows * plants_per_row) / hours = 6000 := by
  sorry

end carrots_planted_per_hour_l736_73640


namespace find_two_digit_ab_l736_73648

def digit_range (n : ℕ) : Prop := 0 ≤ n ∧ n ≤ 9

def different_digits (a b c d : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

theorem find_two_digit_ab (A B C D : ℕ) (hA : digit_range A) (hB : digit_range B)
                         (hC : digit_range C) (hD : digit_range D)
                         (h_diff : different_digits A B C D)
                         (h_eq : (100 * A + 10 * B + C) * (10 * A + B) + C * D = 2017) :
  10 * A + B = 14 :=
sorry

end find_two_digit_ab_l736_73648


namespace find_original_number_l736_73651

theorem find_original_number
  (x : ℤ)
  (h : 3 * (2 * x + 5) = 123) :
  x = 18 := 
sorry

end find_original_number_l736_73651


namespace part1_part2_l736_73661

noncomputable section

variables (a x : ℝ)

def P : Prop := x^2 - 4*a*x + 3*a^2 < 0
def Q : Prop := abs (x - 3) ≤ 1

-- Part 1: If a=1 and P ∨ Q, prove the range of x is 1 < x ≤ 4
theorem part1 (h1 : a = 1) (h2 : P a x ∨ Q x) : 1 < x ∧ x ≤ 4 :=
sorry

-- Part 2: If ¬P is necessary but not sufficient for ¬Q, prove the range of a is 4/3 ≤ a ≤ 2
theorem part2 (h : (¬P a x → ¬Q x) ∧ (¬Q x → ¬P a x → False)) : 4/3 ≤ a ∧ a ≤ 2 :=
sorry

end part1_part2_l736_73661


namespace brenda_ends_with_15_skittles_l736_73665

def initial_skittles : ℕ := 7
def skittles_bought : ℕ := 8

theorem brenda_ends_with_15_skittles : initial_skittles + skittles_bought = 15 := 
by {
  sorry
}

end brenda_ends_with_15_skittles_l736_73665


namespace find_ab_l736_73632

theorem find_ab (a b : ℝ) (h1 : a - b = 10) (h2 : a^2 + b^2 = 150) : a * b = 25 :=
by 
  sorry

end find_ab_l736_73632


namespace numerical_puzzle_unique_solution_l736_73670

theorem numerical_puzzle_unique_solution :
  ∃ (A X Y P : ℕ), 
    A ≠ X ∧ A ≠ Y ∧ A ≠ P ∧ X ≠ Y ∧ X ≠ P ∧ Y ≠ P ∧
    (A * 10 + X) + (Y * 10 + X) = Y * 100 + P * 10 + A ∧
    A = 8 ∧ X = 9 ∧ Y = 1 ∧ P = 0 :=
sorry

end numerical_puzzle_unique_solution_l736_73670


namespace range_of_a_l736_73624

theorem range_of_a (a : ℝ) :
  (∃ x : ℤ, 2 * (x : ℝ) - 1 > 3 ∧ x ≤ a) ∧ (∀ x : ℤ, 2 * (x : ℝ) - 1 > 3 → x ≤ a) → 5 ≤ a ∧ a < 6 :=
by
  sorry

end range_of_a_l736_73624


namespace fraction_meaningful_l736_73639

theorem fraction_meaningful (x : ℝ) : x - 3 ≠ 0 ↔ x ≠ 3 :=
by sorry

end fraction_meaningful_l736_73639


namespace max_marks_l736_73620

theorem max_marks (T : ℝ) (h : 0.33 * T = 165) : T = 500 := 
by {
  sorry
}

end max_marks_l736_73620


namespace max_value_of_f_l736_73617

def f (x : ℝ) : ℝ := 12 * x - 4 * x^2

theorem max_value_of_f : ∀ x : ℝ, f x ≤ 9 :=
by
  have h₁ : ∀ x : ℝ, 12 * x - 4 * x^2 ≤ 9
  { sorry }
  exact h₁

end max_value_of_f_l736_73617


namespace expression_approx_l736_73697

noncomputable def simplified_expression : ℝ :=
  (Real.sqrt 97 + 9 * Real.sqrt 6 + 5 * Real.sqrt 5) / (3 * Real.sqrt 6 + 7)

theorem expression_approx : abs (simplified_expression - 3.002) < 0.001 :=
by
  -- Proof omitted
  sorry

end expression_approx_l736_73697


namespace find_range_of_m_l736_73603

def has_two_distinct_real_roots (m : ℝ) : Prop :=
  m^2 - 4 > 0

def inequality_holds_for_all_real_x (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - 2 * (m + 1) * x + m * (m + 1) > 0

def p (m : ℝ) : Prop := has_two_distinct_real_roots m
def q (m : ℝ) : Prop := inequality_holds_for_all_real_x m

theorem find_range_of_m (m : ℝ) :
  (p m ∨ q m) ∧ ¬(p m ∧ q m) → (m > 2 ∨ (-2 ≤ m ∧ m < -1)) :=
sorry

end find_range_of_m_l736_73603


namespace cube_painted_surface_l736_73637

theorem cube_painted_surface (n : ℕ) (hn : n > 2) 
: 6 * (n - 2) ^ 2 = (n - 2) ^ 3 → n = 8 :=
by
  sorry

end cube_painted_surface_l736_73637


namespace girls_25_percent_less_false_l736_73638

theorem girls_25_percent_less_false (g b : ℕ) (h : b = g * 125 / 100) : (b - g) / b ≠ 25 / 100 := by
  sorry

end girls_25_percent_less_false_l736_73638


namespace gcd_consecutive_terms_l736_73667

theorem gcd_consecutive_terms (n : ℕ) : 
  Nat.gcd (2 * Nat.factorial n + n) (2 * Nat.factorial (n + 1) + (n + 1)) = 1 :=
by
  sorry

end gcd_consecutive_terms_l736_73667


namespace sin_half_alpha_l736_73644

theorem sin_half_alpha (α : ℝ) (h_cos : Real.cos α = -2/3) (h_range : π < α ∧ α < 3 * π / 2) :
  Real.sin (α / 2) = Real.sqrt 30 / 6 :=
by
  sorry

end sin_half_alpha_l736_73644


namespace average_height_40_girls_l736_73666

/-- Given conditions for a class of 50 students, where the average height of 40 girls is H,
    the average height of the remaining 10 girls is 167 cm, and the average height of the whole
    class is 168.6 cm, prove that the average height H of the 40 girls is 169 cm. -/
theorem average_height_40_girls (H : ℝ)
  (h1 : 0 < H)
  (h2 : (40 * H + 10 * 167) = 50 * 168.6) :
  H = 169 :=
by
  sorry

end average_height_40_girls_l736_73666


namespace range_f_iff_l736_73680

noncomputable def f (m x : ℝ) : ℝ :=
  Real.log ((m^2 - 3 * m + 2) * x^2 + 2 * (m - 1) * x + 5)

theorem range_f_iff (m : ℝ) :
  (∀ y ∈ Set.univ, ∃ x, f m x = y) ↔ (m = 1 ∨ (2 < m ∧ m ≤ 9/4)) := 
by
  sorry

end range_f_iff_l736_73680


namespace EDTA_Ca2_complex_weight_l736_73657

-- Definitions of atomic weights
def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.008
def atomic_weight_N : ℝ := 14.01
def atomic_weight_O : ℝ := 16.00
def atomic_weight_Ca : ℝ := 40.08

-- Number of atoms in EDTA
def num_atoms_C : ℝ := 10
def num_atoms_H : ℝ := 16
def num_atoms_N : ℝ := 2
def num_atoms_O : ℝ := 8

-- Molecular weight of EDTA
def molecular_weight_EDTA : ℝ :=
  num_atoms_C * atomic_weight_C +
  num_atoms_H * atomic_weight_H +
  num_atoms_N * atomic_weight_N +
  num_atoms_O * atomic_weight_O

-- Proof that the molecular weight of the complex is 332.328 g/mol
theorem EDTA_Ca2_complex_weight : molecular_weight_EDTA + atomic_weight_Ca = 332.328 := by
  sorry

end EDTA_Ca2_complex_weight_l736_73657


namespace how_many_candies_eaten_l736_73635

variable (candies_tuesday candies_thursday candies_friday candies_left : ℕ)

def total_candies (candies_tuesday candies_thursday candies_friday : ℕ) : ℕ :=
  candies_tuesday + candies_thursday + candies_friday

theorem how_many_candies_eaten (h_tuesday : candies_tuesday = 3)
                               (h_thursday : candies_thursday = 5)
                               (h_friday : candies_friday = 2)
                               (h_left : candies_left = 4) :
  (total_candies candies_tuesday candies_thursday candies_friday) - candies_left = 6 :=
by
  sorry

end how_many_candies_eaten_l736_73635


namespace length_of_GH_l736_73629

variable (S_A S_C S_E S_F : ℝ)
variable (AB FE CD GH : ℝ)

-- Given conditions
axiom h1 : AB = 11
axiom h2 : FE = 13
axiom h3 : CD = 5

-- Relationships between the sizes of the squares
axiom h4 : S_A = S_C + AB
axiom h5 : S_C = S_E + CD
axiom h6 : S_E = S_F + FE
axiom h7 : GH = S_A - S_F

theorem length_of_GH : GH = 29 :=
by
  -- This is where the proof would go
  sorry

end length_of_GH_l736_73629


namespace part1_part2_l736_73655

open Real

variables (α : ℝ) (A : (ℝ × ℝ)) (B : (ℝ × ℝ)) (C : (ℝ × ℝ))

def points_coordinates : Prop :=
A = (3, 0) ∧ B = (0, 3) ∧ C = (cos α, sin α) ∧ π / 2 < α ∧ α < 3 * π / 2

theorem part1 (h : points_coordinates α A B C) (h1 : dist (3, 0) (cos α, sin α) = dist (0, 3) (cos α, sin α)) : 
  α = 5 * π / 4 :=
sorry

theorem part2 (h : points_coordinates α A B C) (h2 : ((cos α - 3) * cos α + (sin α) * (sin α - 3)) = -1) : 
  (2 * sin α * sin α + sin (2 * α)) / (1 + tan α) = -5 / 9 :=
sorry

end part1_part2_l736_73655


namespace sarah_score_l736_73613

theorem sarah_score (s g : ℝ) (h1 : s = g + 50) (h2 : (s + g) / 2 = 110) : s = 135 := 
by 
  sorry

end sarah_score_l736_73613


namespace minimum_value_of_expression_l736_73625

theorem minimum_value_of_expression (x : ℝ) (h : x > 2) : 
  ∃ y, (∀ z, z > 2 → (z^2 - 4 * z + 5) / (z - 2) ≥ y) ∧ 
       y = 2 :=
by
  sorry

end minimum_value_of_expression_l736_73625


namespace min_value_f_l736_73601

noncomputable def f (x : ℝ) : ℝ := 2 * x^2 + 24 * x + 128 / x^3

theorem min_value_f : ∃ x > 0, f x = 168 :=
by
  sorry

end min_value_f_l736_73601


namespace geom_seq_necessity_geom_seq_not_sufficient_l736_73682

theorem geom_seq_necessity (a₁ q : ℝ) (h₁ : 0 < a₁) (h₂ : a₁ < a₁ * q^2) :
    q > 1 ∨ q < -1 :=
  sorry

theorem geom_seq_not_sufficient (a₁ q : ℝ) (h₁ : 0 < a₁) (h₂ : a₁ < a₁ * q^2) :
    ¬ (q > 1 → a₁ < a₁ * q^2) :=
  sorry

end geom_seq_necessity_geom_seq_not_sufficient_l736_73682


namespace exists_idempotent_l736_73659

-- Definition of the set M as the natural numbers from 1 to 1993
def M : Set ℕ := { n | 1 ≤ n ∧ n ≤ 1993 }

-- Operation * on M
noncomputable def star (a b : ℕ) : ℕ := sorry

-- Hypothesis: * is closed on M and (a * b) * a = b for any a, b in M
axiom star_closed (a b : ℕ) (ha : a ∈ M) (hb : b ∈ M) : star a b ∈ M
axiom star_property (a b : ℕ) (ha : a ∈ M) (hb : b ∈ M) : star (star a b) a = b

-- Goal: Prove that there exists a number a in M such that a * a = a
theorem exists_idempotent : ∃ a ∈ M, star a a = a := by
  sorry

end exists_idempotent_l736_73659


namespace james_correct_take_home_pay_l736_73683

noncomputable def james_take_home_pay : ℝ :=
  let main_job_hourly_rate := 20
  let second_job_hourly_rate := main_job_hourly_rate * 0.8
  let main_job_hours := 30
  let main_job_overtime_hours := 5
  let second_job_hours := 15
  let side_gig_daily_rate := 100
  let side_gig_days := 2
  let tax_deductions := 200
  let federal_tax_rate := 0.18
  let state_tax_rate := 0.05

  let regular_main_job_hours := main_job_hours - main_job_overtime_hours
  let main_job_regular_pay := regular_main_job_hours * main_job_hourly_rate
  let main_job_overtime_pay := main_job_overtime_hours * main_job_hourly_rate * 1.5
  let total_main_job_pay := main_job_regular_pay + main_job_overtime_pay

  let total_second_job_pay := second_job_hours * second_job_hourly_rate
  let total_side_gig_pay := side_gig_daily_rate * side_gig_days

  let total_earnings := total_main_job_pay + total_second_job_pay + total_side_gig_pay
  let taxable_income := total_earnings - tax_deductions
  let federal_tax := taxable_income * federal_tax_rate
  let state_tax := taxable_income * state_tax_rate
  let total_taxes := federal_tax + state_tax
  total_earnings - total_taxes

theorem james_correct_take_home_pay : james_take_home_pay = 885.30 := by
  sorry

end james_correct_take_home_pay_l736_73683


namespace books_left_correct_l736_73626

variable (initial_books : ℝ) (sold_books : ℝ)

def number_of_books_left (initial_books sold_books : ℝ) : ℝ :=
  initial_books - sold_books

theorem books_left_correct :
  number_of_books_left 51.5 45.75 = 5.75 :=
by
  sorry

end books_left_correct_l736_73626


namespace Shekar_marks_in_Science_l736_73614

theorem Shekar_marks_in_Science (S : ℕ) (h : (76 + S + 82 + 67 + 85) / 5 = 75) : S = 65 :=
sorry

end Shekar_marks_in_Science_l736_73614


namespace quadratic_inequality_for_all_x_l736_73608

theorem quadratic_inequality_for_all_x (a : ℝ) :
  (∀ x : ℝ, (a^2 + a) * x^2 - a * x + 1 > 0) ↔ (-4 / 3 < a ∧ a < -1) ∨ a = 0 :=
sorry

end quadratic_inequality_for_all_x_l736_73608


namespace find_area_of_triangle_l736_73689

noncomputable def triangle_area (a b: ℝ) (cosC: ℝ) : ℝ :=
  let sinC := Real.sqrt (1 - cosC^2)
  0.5 * a * b * sinC

theorem find_area_of_triangle :
  ∀ (a b cosC : ℝ), a = 3 * Real.sqrt 2 → b = 2 * Real.sqrt 3 → cosC = 1 / 3 →
  triangle_area a b cosC = 4 * Real.sqrt 3 :=
by
  intros a b cosC ha hb hcosC
  rw [ha, hb, hcosC]
  sorry

end find_area_of_triangle_l736_73689


namespace greatest_three_digit_number_divisible_by_3_6_5_l736_73649

theorem greatest_three_digit_number_divisible_by_3_6_5 :
  ∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ (n % 3 = 0) ∧ (n % 6 = 0) ∧ (n % 5 = 0) ∧ (∀ m : ℕ, 100 ≤ m ∧ m ≤ 999 ∧ (m % 3 = 0) ∧ (m % 6 = 0) ∧ (m % 5 = 0) → m ≤ n) ∧ n = 990 := 
by
  sorry

end greatest_three_digit_number_divisible_by_3_6_5_l736_73649


namespace ratio_of_m_div_x_l736_73662

theorem ratio_of_m_div_x (a b : ℝ) (h1 : a / b = 4 / 5) (h2 : a > 0) (h3 : b > 0) :
  let x := a + 0.25 * a
  let m := b - 0.60 * b
  (m / x) = 2 / 5 :=
by
  -- Define x and m
  let x := a + 0.25 * a
  let m := b - 0.60 * b
  -- Include the steps or assumptions here if necessary
  sorry

end ratio_of_m_div_x_l736_73662


namespace candies_per_block_l736_73619

theorem candies_per_block (candies_per_house : ℕ) (houses_per_block : ℕ) (h1 : candies_per_house = 7) (h2 : houses_per_block = 5) :
  candies_per_house * houses_per_block = 35 :=
by 
  -- Placeholder for the formal proof
  sorry

end candies_per_block_l736_73619


namespace range_of_a_l736_73646

theorem range_of_a (a : ℝ) : (-1/3 ≤ a) ∧ (a ≤ 2/3) ↔ (∀ x y : ℝ, (0 ≤ x ∧ x ≤ 1) → (0 ≤ y ∧ y ≤ 1) → y = a * x + 1/3) :=
by
  sorry

end range_of_a_l736_73646


namespace Carmela_difference_l736_73690

theorem Carmela_difference (Cecil Catherine Carmela : ℤ) (X : ℤ) (h1 : Cecil = 600) 
(h2 : Catherine = 2 * Cecil - 250) (h3 : Carmela = 2 * Cecil + X) 
(h4 : Cecil + Catherine + Carmela = 2800) : X = 50 :=
by { sorry }

end Carmela_difference_l736_73690


namespace quadrilateral_is_trapezoid_l736_73645

variables {V : Type*} [AddCommGroup V] [Module ℝ V] -- Define the type of vectors and vector space over the reals
variables (a b : V) -- Vectors a and b
variables (AB BC CD AD : V) -- Vectors representing sides of quadrilateral

-- Condition: vectors a and b are not collinear
def not_collinear (a b : V) : Prop := ∀ k : ℝ, k ≠ 0 → a ≠ k • b

-- Given Conditions
def conditions (a b AB BC CD : V) : Prop :=
  AB = a + 2 • b ∧
  BC = -4 • a - b ∧
  CD = -5 • a - 3 • b ∧
  not_collinear a b

-- The to-be-proven property
def is_trapezoid (AB BC CD AD : V) : Prop :=
  AD = 2 • BC

theorem quadrilateral_is_trapezoid 
  (a b AB BC CD : V) 
  (h : conditions a b AB BC CD)
  : is_trapezoid AB BC CD (AB + BC + CD) :=
sorry

end quadrilateral_is_trapezoid_l736_73645


namespace eq_infinite_solutions_pos_int_l736_73654

noncomputable def eq_has_inf_solutions_in_positive_integers (m : ℕ) : Prop :=
    ∀ (a b c : ℕ), 
    a > 0 ∧ b > 0 ∧ c > 0 → 
    ∃ (a' b' c' : ℕ), 
    a' > 0 ∧ b' > 0 ∧ c' > 0 ∧ 
    (1 / a + 1 / b + 1 / c + 1 / (a * b * c) = m / (a + b + c))

theorem eq_infinite_solutions_pos_int (m : ℕ) (hm : m > 0) : eq_has_inf_solutions_in_positive_integers m := 
by 
  sorry

end eq_infinite_solutions_pos_int_l736_73654


namespace franks_age_l736_73607

variable (F G : ℕ)

def gabriel_younger_than_frank : Prop := G = F - 3
def total_age_is_seventeen : Prop := F + G = 17

theorem franks_age (h1 : gabriel_younger_than_frank F G) (h2 : total_age_is_seventeen F G) : F = 10 :=
by
  sorry

end franks_age_l736_73607


namespace problem_1_solution_set_problem_2_min_value_l736_73615

-- Problem (1)
def f (x : ℝ) : ℝ := 4 - |x| - |x - 3|

theorem problem_1_solution_set :
  {x : ℝ | f (x + 3/2) ≥ 0} = {x | -2 ≤ x ∧ x ≤ 2} :=
by
  sorry

-- Problem (2)
theorem problem_2_min_value (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) 
  (h : 1/(3*p) + 1/(2*q) + 1/r = 4) : 
  3*p + 2*q + r ≥ 9/4 :=
by
  sorry

end problem_1_solution_set_problem_2_min_value_l736_73615


namespace students_enrolled_for_german_l736_73605

-- Defining the total number of students
def class_size : Nat := 40

-- Defining the number of students enrolled for both English and German
def enrolled_both : Nat := 12

-- Defining the number of students enrolled for only English and not German
def enrolled_only_english : Nat := 18

-- Using the conditions to define the number of students who enrolled for German
theorem students_enrolled_for_german (G G_only : Nat) 
  (h_class_size : G_only + enrolled_only_english + enrolled_both = class_size) 
  (h_G : G = G_only + enrolled_both) : 
  G = 22 := 
by
  -- placeholder for proof
  sorry

end students_enrolled_for_german_l736_73605


namespace quadratic_double_root_eq1_quadratic_double_root_eq2_l736_73693

theorem quadratic_double_root_eq1 :
  (∃ r : ℝ , ∃ s : ℝ, (r ≠ s) ∧ (
  (1 : ℝ) * r^2 + (-3 : ℝ) * r + (2 : ℝ) = 0 ∧
  (1 : ℝ) * s^2 + (-3 : ℝ) * s + (2 : ℝ) = 0 ∧
  (r = 2 * s ∨ s = 2 * r) 
  )) := 
  sorry

theorem quadratic_double_root_eq2 :
  (∃ a b : ℝ, a ≠ 0 ∧
  ((∃ r : ℝ, (-b / a = 2 + r) ∧ (-6 / a = 2 * r)) ∨ 
  ((-b / a = 2 + 1) ∧ (-6 / a = 2 * 1))) ∧ 
  ((a = -3/4 ∧ b = 9/2) ∨ (a = -3 ∧ b = 9))) :=
  sorry

end quadratic_double_root_eq1_quadratic_double_root_eq2_l736_73693


namespace ratio_Nikki_to_Michael_l736_73678

theorem ratio_Nikki_to_Michael
  (M Joyce Nikki Ryn : ℕ)
  (h1 : Joyce = M + 2)
  (h2 : Nikki = 30)
  (h3 : Ryn = (4 / 5) * Nikki)
  (h4 : M + Joyce + Nikki + Ryn = 76) :
  Nikki / M = 3 :=
by {
  sorry
}

end ratio_Nikki_to_Michael_l736_73678


namespace interest_first_year_correct_interest_second_year_correct_interest_third_year_correct_l736_73602

noncomputable def principal_first_year : ℝ := 9000
noncomputable def interest_rate_first_year : ℝ := 0.09
noncomputable def principal_second_year : ℝ := principal_first_year * (1 + interest_rate_first_year)
noncomputable def interest_rate_second_year : ℝ := 0.105
noncomputable def principal_third_year : ℝ := principal_second_year * (1 + interest_rate_second_year)
noncomputable def interest_rate_third_year : ℝ := 0.085

noncomputable def compute_interest (principal : ℝ) (rate : ℝ) : ℝ :=
  principal * rate

theorem interest_first_year_correct :
  compute_interest principal_first_year interest_rate_first_year = 810 := by
  sorry

theorem interest_second_year_correct :
  compute_interest principal_second_year interest_rate_second_year = 1034.55 := by
  sorry

theorem interest_third_year_correct :
  compute_interest principal_third_year interest_rate_third_year = 922.18 := by
  sorry

end interest_first_year_correct_interest_second_year_correct_interest_third_year_correct_l736_73602


namespace second_player_wins_when_2003_candies_l736_73686

def game_winning_strategy (n : ℕ) : ℕ :=
  if n % 2 = 0 then 1 else 2

theorem second_player_wins_when_2003_candies :
  game_winning_strategy 2003 = 2 :=
by 
  sorry

end second_player_wins_when_2003_candies_l736_73686


namespace gun_fan_image_equivalence_l736_73677

def gunPiercingImage : String := "point moving to form a line"
def foldingFanImage : String := "line moving to form a surface"

theorem gun_fan_image_equivalence :
  (gunPiercingImage = "point moving to form a line") ∧ 
  (foldingFanImage = "line moving to form a surface") := by
  -- Proof goes here
  sorry

end gun_fan_image_equivalence_l736_73677


namespace range_of_m_in_first_quadrant_l736_73688

theorem range_of_m_in_first_quadrant (m : ℝ) : ((m - 1 > 0) ∧ (m + 2 > 0)) ↔ m > 1 :=
by sorry

end range_of_m_in_first_quadrant_l736_73688


namespace trapezoid_area_l736_73636

theorem trapezoid_area:
  let vert1 := (10, 10)
  let vert2 := (15, 15)
  let vert3 := (0, 15)
  let vert4 := (0, 10)
  let base1 := 10
  let base2 := 15
  let height := 5
  ∃ (area : ℝ), area = 62.5 := by
  sorry

end trapezoid_area_l736_73636


namespace remaining_lives_l736_73606

theorem remaining_lives (initial_players quit1 quit2 player_lives : ℕ) (h1 : initial_players = 15) (h2 : quit1 = 5) (h3 : quit2 = 4) (h4 : player_lives = 7) :
  (initial_players - quit1 - quit2) * player_lives = 42 :=
by
  sorry

end remaining_lives_l736_73606


namespace ratio_ac_bd_l736_73653

theorem ratio_ac_bd (a b c d : ℝ) (h1 : a = 4 * b) (h2 : b = 2 * c) (h3 : c = 5 * d) : 
  (a * c) / (b * d) = 20 :=
by
  sorry

end ratio_ac_bd_l736_73653
