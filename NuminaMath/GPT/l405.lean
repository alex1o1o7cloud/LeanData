import Mathlib

namespace NUMINAMATH_GPT_max_a_plus_ab_plus_abc_l405_40541

noncomputable def f (a b c: ℝ) := a + a * b + a * b * c

theorem max_a_plus_ab_plus_abc (a b c : ℝ) (h1 : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c) (h2 : a + b + c = 1) :
  ∃ x, (f a b c ≤ x) ∧ (∀ y, f a b c ≤ y → y = 1) :=
sorry

end NUMINAMATH_GPT_max_a_plus_ab_plus_abc_l405_40541


namespace NUMINAMATH_GPT_modulo_residue_addition_l405_40511

theorem modulo_residue_addition : 
  (368 + 3 * 78 + 8 * 242 + 6 * 22) % 11 = 8 := 
by
  have h1 : 368 % 11 = 5 := by sorry
  have h2 : 78 % 11 = 1 := by sorry
  have h3 : 242 % 11 = 0 := by sorry
  have h4 : 22 % 11 = 0 := by sorry
  sorry

end NUMINAMATH_GPT_modulo_residue_addition_l405_40511


namespace NUMINAMATH_GPT_problem_1_problem_2_l405_40596

-- Proof Problem 1
theorem problem_1 (x : ℝ) : (x^2 + 2 > |x - 4| - |x - 1|) ↔ (x > 1 ∨ x ≤ -1) :=
sorry

-- Proof Problem 2
theorem problem_2 (a : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x₁ x₂, x₁^2 + 2 ≥ |x₂ - a| - |x₂ - 1|) → (-1 ≤ a ∧ a ≤ 3) :=
sorry

end NUMINAMATH_GPT_problem_1_problem_2_l405_40596


namespace NUMINAMATH_GPT_M_intersection_N_eq_M_l405_40566

def is_element_of_M (y : ℝ) : Prop := ∃ x : ℝ, y = 2^x
def is_element_of_N (y : ℝ) : Prop := ∃ x : ℝ, y = x^2

theorem M_intersection_N_eq_M : {y | is_element_of_M y} ∩ {y | is_element_of_N y} = {y | is_element_of_M y} :=
by
  sorry

end NUMINAMATH_GPT_M_intersection_N_eq_M_l405_40566


namespace NUMINAMATH_GPT_ninth_term_of_geometric_sequence_l405_40505

theorem ninth_term_of_geometric_sequence (a r : ℕ) (h1 : a = 3) (h2 : a * r^6 = 2187) : a * r^8 = 19683 := by
  sorry

end NUMINAMATH_GPT_ninth_term_of_geometric_sequence_l405_40505


namespace NUMINAMATH_GPT_derivative_at_pi_over_4_l405_40533

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x

theorem derivative_at_pi_over_4 : (deriv f) (Real.pi / 4) = 0 := 
by
  sorry

end NUMINAMATH_GPT_derivative_at_pi_over_4_l405_40533


namespace NUMINAMATH_GPT_square_side_length_tangent_circle_l405_40535

theorem square_side_length_tangent_circle (r s : ℝ) :
  (∃ (O : ℝ × ℝ) (A : ℝ × ℝ) (AB : ℝ) (AD : ℝ),
    AB = AD ∧
    O = (r, r) ∧
    A = (0, 0) ∧
    dist O A = r * Real.sqrt 2 ∧
    s = dist (O.fst, 0) A ∧
    s = dist (0, O.snd) A ∧
    ∀ x y, (O = (x, y) → x = r ∧ y = r)) → s = 2 * r :=
by
  sorry

end NUMINAMATH_GPT_square_side_length_tangent_circle_l405_40535


namespace NUMINAMATH_GPT_complement_of_A_l405_40570

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x ≤ -3} ∪ {x | x ≥ 0}

theorem complement_of_A :
  U \ A = {x | -3 < x ∧ x < 0} :=
sorry

end NUMINAMATH_GPT_complement_of_A_l405_40570


namespace NUMINAMATH_GPT_dale_pasta_l405_40520

-- Define the conditions
def original_pasta : Nat := 2
def original_servings : Nat := 7
def final_servings : Nat := 35

-- Define the required calculation for the number of pounds of pasta needed
def required_pasta : Nat := 10

-- The theorem to prove
theorem dale_pasta : (final_servings / original_servings) * original_pasta = required_pasta := 
by
  sorry

end NUMINAMATH_GPT_dale_pasta_l405_40520


namespace NUMINAMATH_GPT_ending_number_divisible_by_3_l405_40518

theorem ending_number_divisible_by_3 (n : ℕ) :
  (∀ k, 0 ≤ k ∧ k < 13 → ∃ m, 10 ≤ m ∧ m ≤ n ∧ m % 3 = 0) →
  n = 48 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_ending_number_divisible_by_3_l405_40518


namespace NUMINAMATH_GPT_peaches_picked_l405_40582

variable (o t : ℕ)
variable (p : ℕ)

theorem peaches_picked : (o = 34) → (t = 86) → (t = o + p) → p = 52 :=
by
  intros ho ht htot
  rw [ho, ht] at htot
  sorry

end NUMINAMATH_GPT_peaches_picked_l405_40582


namespace NUMINAMATH_GPT_largest_sum_distinct_factors_l405_40587

theorem largest_sum_distinct_factors (A B C : ℕ) (h1 : A ≠ B) (h2 : B ≠ C) (h3 : A ≠ C) 
  (h4 : A * B * C = 2023) : A + B + C = 297 :=
sorry

end NUMINAMATH_GPT_largest_sum_distinct_factors_l405_40587


namespace NUMINAMATH_GPT_highest_value_meter_l405_40555

theorem highest_value_meter (A B C : ℝ) 
  (h_avg : (A + B + C) / 3 = 6)
  (h_A_min : A = 2)
  (h_B_min : B = 2) : C = 14 :=
by {
  sorry
}

end NUMINAMATH_GPT_highest_value_meter_l405_40555


namespace NUMINAMATH_GPT_multiple_of_one_third_l405_40559

theorem multiple_of_one_third (x : ℚ) (h : x * (1 / 3) = 2 / 9) : x = 2 / 3 :=
sorry

end NUMINAMATH_GPT_multiple_of_one_third_l405_40559


namespace NUMINAMATH_GPT_division_quotient_difference_l405_40563

theorem division_quotient_difference :
  (32.5 / 1.3) - (60.8 / 7.6) = 17 :=
by
  sorry

end NUMINAMATH_GPT_division_quotient_difference_l405_40563


namespace NUMINAMATH_GPT_solution_set_of_inequality_l405_40564

theorem solution_set_of_inequality (x : ℝ) :  (3 ≤ |5 - 2 * x| ∧ |5 - 2 * x| < 9) ↔ (-2 < x ∧ x ≤ 1) ∨ (4 ≤ x ∧ x < 7) :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l405_40564


namespace NUMINAMATH_GPT_solution_to_equation_l405_40589

theorem solution_to_equation :
  ∃ x : ℝ, x = (11 - 3 * Real.sqrt 5) / 2 ∧ x^2 + 6 * x + 6 * x * Real.sqrt (x + 4) = 31 :=
by
  sorry

end NUMINAMATH_GPT_solution_to_equation_l405_40589


namespace NUMINAMATH_GPT_points_lie_on_line_l405_40584

noncomputable def x (t : ℝ) (ht : t ≠ 0) : ℝ := (t^2 + 2 * t + 2) / t
noncomputable def y (t : ℝ) (ht : t ≠ 0) : ℝ := (t^2 - 2 * t + 2) / t

theorem points_lie_on_line : ∀ (t : ℝ) (ht : t ≠ 0), y t ht = x t ht - 4 :=
by 
  intros t ht
  simp [x, y]
  sorry

end NUMINAMATH_GPT_points_lie_on_line_l405_40584


namespace NUMINAMATH_GPT_fred_sheets_left_l405_40594

def sheets_fred_had_initially : ℕ := 212
def sheets_jane_given : ℕ := 307
def planned_percentage_more : ℕ := 50
def given_percentage : ℕ := 25

-- Prove that after all transactions, Fred has 389 sheets left
theorem fred_sheets_left :
  let planned_sheets := (sheets_jane_given * 100) / (planned_percentage_more + 100)
  let sheets_jane_actual := planned_sheets + (planned_sheets * planned_percentage_more) / 100
  let total_sheets := sheets_fred_had_initially + sheets_jane_actual
  let charles_given := (total_sheets * given_percentage) / 100
  let fred_sheets_final := total_sheets - charles_given
  fred_sheets_final = 389 := 
by
  sorry

end NUMINAMATH_GPT_fred_sheets_left_l405_40594


namespace NUMINAMATH_GPT_cat_toy_cost_correct_l405_40592

-- Define the initial amount of money Jessica had.
def initial_amount : ℝ := 11.73

-- Define the amount left after spending.
def amount_left : ℝ := 1.51

-- Define the cost of the cat toy.
def toy_cost : ℝ := initial_amount - amount_left

-- Theorem and statement to prove the cost of the cat toy.
theorem cat_toy_cost_correct : toy_cost = 10.22 := sorry

end NUMINAMATH_GPT_cat_toy_cost_correct_l405_40592


namespace NUMINAMATH_GPT_polynomial_coefficient_a5_l405_40537

theorem polynomial_coefficient_a5 : 
  (∃ (a0 a1 a2 a3 a4 a5 a6 : ℝ), 
    (∀ (x : ℝ), ((2 * x - 1)^5 * (x + 2) = a0 + a1 * (x - 1) + a2 * (x - 1)^2 + a3 * (x - 1)^3 + a4 * (x - 1)^4 + a5 * (x - 1)^5 + a6 * (x - 1)^6)) ∧ 
    a5 = 176) := sorry

end NUMINAMATH_GPT_polynomial_coefficient_a5_l405_40537


namespace NUMINAMATH_GPT_sale_prices_correct_l405_40545

-- Define the cost prices and profit percentages
def cost_price_A : ℕ := 320
def profit_percentage_A : ℕ := 50

def cost_price_B : ℕ := 480
def profit_percentage_B : ℕ := 70

def cost_price_C : ℕ := 600
def profit_percentage_C : ℕ := 40

-- Define the expected sale prices
def sale_price_A : ℕ := 480
def sale_price_B : ℕ := 816
def sale_price_C : ℕ := 840

-- Define a function to compute sale price
def compute_sale_price (cost_price : ℕ) (profit_percentage : ℕ) : ℕ :=
  cost_price + (profit_percentage * cost_price) / 100

-- The proof statement
theorem sale_prices_correct :
  compute_sale_price cost_price_A profit_percentage_A = sale_price_A ∧
  compute_sale_price cost_price_B profit_percentage_B = sale_price_B ∧
  compute_sale_price cost_price_C profit_percentage_C = sale_price_C :=
by {
  sorry
}

end NUMINAMATH_GPT_sale_prices_correct_l405_40545


namespace NUMINAMATH_GPT_sqrt_5_is_quadratic_radical_l405_40525

variable (a : ℝ) -- a is a real number

-- Definition to check if a given expression is a quadratic radical
def is_quadratic_radical (x : ℝ) : Prop := ∃ y : ℝ, y^2 = x

theorem sqrt_5_is_quadratic_radical : is_quadratic_radical 5 :=
by
  -- Here, 'by' indicates the start of the proof block,
  -- but the actual content of the proof is replaced with 'sorry' as instructed.
  sorry

end NUMINAMATH_GPT_sqrt_5_is_quadratic_radical_l405_40525


namespace NUMINAMATH_GPT_cube_root_simplification_l405_40500

theorem cube_root_simplification (N : ℝ) (h : N > 1) : (N^3)^(1/3) * ((N^5)^(1/3) * ((N^3)^(1/3)))^(1/3) = N^(5/3) :=
by sorry

end NUMINAMATH_GPT_cube_root_simplification_l405_40500


namespace NUMINAMATH_GPT_ordered_pair_a_c_l405_40551

theorem ordered_pair_a_c (a c : ℝ) (h_quad: ∀ x : ℝ, a * x^2 + 16 * x + c = 0)
    (h_sum: a + c = 25) (h_ineq: a < c) : (a = 3 ∧ c = 22) :=
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_ordered_pair_a_c_l405_40551


namespace NUMINAMATH_GPT_work_completion_time_l405_40591

theorem work_completion_time
  (A B C : ℝ)
  (h1 : A + B = 1 / 12)
  (h2 : B + C = 1 / 15)
  (h3 : C + A = 1 / 20) :
  1 / (A + B + C) = 10 :=
by
  sorry

end NUMINAMATH_GPT_work_completion_time_l405_40591


namespace NUMINAMATH_GPT_number_of_whole_numbers_between_sqrt_18_and_sqrt_120_l405_40550

theorem number_of_whole_numbers_between_sqrt_18_and_sqrt_120 : 
  ∀ (n : ℕ), 
  (5 ≤ n ∧ n ≤ 10) ↔ (6 = 6) :=
sorry

end NUMINAMATH_GPT_number_of_whole_numbers_between_sqrt_18_and_sqrt_120_l405_40550


namespace NUMINAMATH_GPT_john_bought_six_bagels_l405_40549

theorem john_bought_six_bagels (b m : ℕ) (expenditure_in_dollars_whole : (90 * b + 60 * m) % 100 = 0) (total_items : b + m = 7) : 
b = 6 :=
by
  -- The proof goes here. For now, we skip it with sorry.
  sorry

end NUMINAMATH_GPT_john_bought_six_bagels_l405_40549


namespace NUMINAMATH_GPT_living_room_area_l405_40542

-- Define the conditions
def carpet_area (length width : ℕ) : ℕ :=
  length * width

def percentage_coverage (carpet_area living_room_area : ℕ) : ℕ :=
  (carpet_area * 100) / living_room_area

-- State the problem
theorem living_room_area (A : ℕ) (carpet_len carpet_wid : ℕ) (carpet_coverage : ℕ) :
  carpet_len = 4 → carpet_wid = 9 → carpet_coverage = 20 →
  20 * A = 36 * 100 → A = 180 :=
by
  intros h_len h_wid h_coverage h_proportion
  sorry

end NUMINAMATH_GPT_living_room_area_l405_40542


namespace NUMINAMATH_GPT_calculate_ratio_l405_40568

theorem calculate_ratio (l m n : ℝ) :
  let D := (l + 1, 1, 1)
  let E := (1, m + 1, 1)
  let F := (1, 1, n + 1)
  let AB_sq := 4 * ((n - m) ^ 2)
  let AC_sq := 4 * ((l - n) ^ 2)
  let BC_sq := 4 * ((m - l) ^ 2)
  (AB_sq + AC_sq + BC_sq + 3) / (l^2 + m^2 + n^2 + 3) = 8 := by
  sorry

end NUMINAMATH_GPT_calculate_ratio_l405_40568


namespace NUMINAMATH_GPT_remaining_cubes_l405_40530

-- The configuration of the initial cube and the properties of a layer
def initial_cube : ℕ := 10
def total_cubes : ℕ := 1000
def layer_cubes : ℕ := (initial_cube * initial_cube)

-- The proof problem: Prove that the remaining number of cubes is 900 after removing one layer
theorem remaining_cubes : total_cubes - layer_cubes = 900 := 
by 
  sorry

end NUMINAMATH_GPT_remaining_cubes_l405_40530


namespace NUMINAMATH_GPT_benches_required_l405_40508

theorem benches_required (students_base5 : ℕ := 312) (base_student_seating : ℕ := 5) (seats_per_bench : ℕ := 3) : ℕ :=
  let chairs := 3 * base_student_seating^2 + 1 * base_student_seating^1 + 2 * base_student_seating^0
  let benches := (chairs / seats_per_bench) + if (chairs % seats_per_bench > 0) then 1 else 0
  benches

example : benches_required = 28 :=
by sorry

end NUMINAMATH_GPT_benches_required_l405_40508


namespace NUMINAMATH_GPT_impossible_to_load_two_coins_l405_40579

theorem impossible_to_load_two_coins 
  (p q : ℝ) 
  (h0 : p ≠ 1 - p)
  (h1 : q ≠ 1 - q) 
  (hpq : p * q = 1/4)
  (hptq : p * (1 - q) = 1/4)
  (h1pq : (1 - p) * q = 1/4)
  (h1ptq : (1 - p) * (1 - q) = 1/4) : 
  false :=
sorry

end NUMINAMATH_GPT_impossible_to_load_two_coins_l405_40579


namespace NUMINAMATH_GPT_total_weeds_correct_l405_40504

def tuesday : ℕ := 25
def wednesday : ℕ := 3 * tuesday
def thursday : ℕ := wednesday / 5
def friday : ℕ := thursday - 10
def total_weeds : ℕ := tuesday + wednesday + thursday + friday

theorem total_weeds_correct : total_weeds = 120 :=
by
  sorry

end NUMINAMATH_GPT_total_weeds_correct_l405_40504


namespace NUMINAMATH_GPT_Nancy_money_in_dollars_l405_40529

-- Condition: Nancy has saved 1 dozen quarters
def dozen : ℕ := 12

-- Condition: Each quarter is worth 25 cents
def value_of_quarter : ℕ := 25

-- Condition: 100 cents is equal to 1 dollar
def cents_per_dollar : ℕ := 100

-- Proving that Nancy has 3 dollars
theorem Nancy_money_in_dollars :
  (dozen * value_of_quarter) / cents_per_dollar = 3 := by
  sorry

end NUMINAMATH_GPT_Nancy_money_in_dollars_l405_40529


namespace NUMINAMATH_GPT_binomials_product_evaluation_l405_40599

-- Define the binomials and the resulting polynomial
def binomial_one (x : ℝ) := 4 * x + 3
def binomial_two (x : ℝ) := 2 * x - 6
def resulting_polynomial (x : ℝ) := 8 * x^2 - 18 * x - 18

-- Define the proof problem
theorem binomials_product_evaluation :
  ∀ (x : ℝ), (binomial_one x) * (binomial_two x) = resulting_polynomial x ∧ 
  resulting_polynomial (-1) = 8 := 
by 
  intro x
  have h1 : (4 * x + 3) * (2 * x - 6) = 8 * x^2 - 18 * x - 18 := sorry
  have h2 : resulting_polynomial (-1) = 8 := sorry
  exact ⟨h1, h2⟩

end NUMINAMATH_GPT_binomials_product_evaluation_l405_40599


namespace NUMINAMATH_GPT_money_distribution_l405_40597

-- Conditions
variable (A B x y : ℝ)
variable (h1 : x + 1/2 * y = 50)
variable (h2 : 2/3 * x + y = 50)

-- Problem statement
theorem money_distribution : x = A → y = B → (x + 1/2 * y = 50 ∧ 2/3 * x + y = 50) :=
by
  intro hx hy
  rw [hx, hy]
  exfalso -- using exfalso to skip proof body
  sorry

end NUMINAMATH_GPT_money_distribution_l405_40597


namespace NUMINAMATH_GPT_negation_of_positive_l405_40569

def is_positive (x : ℝ) : Prop := x > 0
def is_non_positive (x : ℝ) : Prop := x ≤ 0

theorem negation_of_positive (a b c : ℝ) :
  (¬ (is_positive a ∨ is_positive b ∨ is_positive c)) ↔ (is_non_positive a ∧ is_non_positive b ∧ is_non_positive c) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_positive_l405_40569


namespace NUMINAMATH_GPT_mass_percentage_Al_aluminum_carbonate_l405_40578

theorem mass_percentage_Al_aluminum_carbonate :
  let m_Al := 26.98  -- molar mass of Al in g/mol
  let m_C := 12.01  -- molar mass of C in g/mol
  let m_O := 16.00  -- molar mass of O in g/mol
  let molar_mass_CO3 := m_C + 3 * m_O  -- molar mass of CO3 in g/mol
  let molar_mass_Al2CO33 := 2 * m_Al + 3 * molar_mass_CO3  -- molar mass of Al2(CO3)3 in g/mol
  let mass_Al_in_Al2CO33 := 2 * m_Al  -- mass of Al in Al2(CO3)3 in g/mol
  (mass_Al_in_Al2CO33 / molar_mass_Al2CO33) * 100 = 23.05 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_mass_percentage_Al_aluminum_carbonate_l405_40578


namespace NUMINAMATH_GPT_area_of_defined_region_l405_40543

theorem area_of_defined_region : 
  ∃ (A : ℝ), (∀ x y : ℝ, |4 * x - 20| + |3 * y + 9| ≤ 6 → A = 9) :=
sorry

end NUMINAMATH_GPT_area_of_defined_region_l405_40543


namespace NUMINAMATH_GPT_find_y_minus_x_l405_40527

theorem find_y_minus_x (x y : ℕ) (hx : x + y = 540) (hxy : (x : ℚ) / (y : ℚ) = 7 / 8) : y - x = 36 :=
by
  sorry

end NUMINAMATH_GPT_find_y_minus_x_l405_40527


namespace NUMINAMATH_GPT_simplify_expression_l405_40522

theorem simplify_expression (x : ℝ) : 
  8 * x + 15 - 3 * x + 5 * 7 = 5 * x + 50 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l405_40522


namespace NUMINAMATH_GPT_angle_C_of_quadrilateral_ABCD_l405_40501

theorem angle_C_of_quadrilateral_ABCD
  (AB CD BC AD : ℝ) (D : ℝ) (h_AB_CD : AB = CD) (h_BC_AD : BC = AD) (h_ang_D : D = 120) :
  ∃ C : ℝ, C = 60 :=
by
  sorry

end NUMINAMATH_GPT_angle_C_of_quadrilateral_ABCD_l405_40501


namespace NUMINAMATH_GPT_f_zero_derivative_not_extremum_l405_40524

noncomputable def f (x : ℝ) : ℝ := x ^ 3

theorem f_zero_derivative_not_extremum (x : ℝ) : 
  deriv f 0 = 0 ∧ ∀ (y : ℝ), y ≠ 0 → (∃ δ > 0, ∀ z, abs (z - 0) < δ → (f z / z : ℝ) ≠ 0) :=
by
  sorry

end NUMINAMATH_GPT_f_zero_derivative_not_extremum_l405_40524


namespace NUMINAMATH_GPT_compute_f5_l405_40590

-- Definitions of the logical operations used in the conditions
axiom x1 : Prop
axiom x2 : Prop
axiom x3 : Prop
axiom x4 : Prop
axiom x5 : Prop

noncomputable def x6 : Prop := x1 ∨ x3
noncomputable def x7 : Prop := x2 ∧ x6
noncomputable def x8 : Prop := x3 ∨ x5
noncomputable def x9 : Prop := x4 ∧ x8
noncomputable def f5 : Prop := x7 ∨ x9

-- Proof statement to be proven
theorem compute_f5 : f5 = (x7 ∨ x9) :=
by sorry

end NUMINAMATH_GPT_compute_f5_l405_40590


namespace NUMINAMATH_GPT_sum_odds_200_600_l405_40519

-- Define the bounds 200 and 600 for our range
def lower_bound := 200
def upper_bound := 600

-- Define first and last odd integers in the range
def first_odd := 201
def last_odd := 599

-- Define the common difference in our arithmetic sequence
def common_diff := 2

-- Number of terms in the sequence
def n := ((last_odd - first_odd) / common_diff) + 1

-- Sum of the arithmetic sequence formula
def sum_arithmetic_seq (n : ℕ) (a l : ℕ) : ℕ :=
  n * (a + l) / 2

-- Specifically, the sum of odd integers between 200 and 600
def sum_odd_integers : ℕ := sum_arithmetic_seq n first_odd last_odd

-- Theorem stating the sum is equal to 80000
theorem sum_odds_200_600 : sum_odd_integers = 80000 :=
by sorry

end NUMINAMATH_GPT_sum_odds_200_600_l405_40519


namespace NUMINAMATH_GPT_probability_of_winning_first_draw_better_chance_with_yellow_ball_l405_40546

-- The probability of winning on the first draw in the lottery promotion.
theorem probability_of_winning_first_draw :
  (1 / 4 : ℚ) = 0.25 :=
sorry

-- The optimal choice to add to the bag for the highest probability of receiving a fine gift.
theorem better_chance_with_yellow_ball :
  (3 / 5 : ℚ) > (2 / 5 : ℚ) :=
by norm_num

end NUMINAMATH_GPT_probability_of_winning_first_draw_better_chance_with_yellow_ball_l405_40546


namespace NUMINAMATH_GPT_find_a_and_b_l405_40580

theorem find_a_and_b (a b : ℕ) :
  42 = a * 6 ∧ 72 = 6 * b ∧ 504 = 42 * 12 → (a, b) = (7, 12) :=
by
  sorry

end NUMINAMATH_GPT_find_a_and_b_l405_40580


namespace NUMINAMATH_GPT_vityas_miscalculation_l405_40516

/-- Vitya's miscalculated percentages problem -/
theorem vityas_miscalculation :
  ∀ (N : ℕ)
  (acute obtuse nonexistent right depends_geometry : ℕ)
  (H_acute : acute = 5)
  (H_obtuse : obtuse = 5)
  (H_nonexistent : nonexistent = 5)
  (H_right : right = 50)
  (H_total : acute + obtuse + nonexistent + right + depends_geometry = 100),
  depends_geometry = 110 :=
by
  intros
  sorry

end NUMINAMATH_GPT_vityas_miscalculation_l405_40516


namespace NUMINAMATH_GPT_find_s_l405_40514

theorem find_s : ∃ s : ℚ, (∀ x : ℚ, (3 * x^2 - 8 * x + 9) * (5 * x^2 + s * x + 15) = 15 * x^4 - 71 * x^3 + 174 * x^2 - 215 * x + 135) ∧ s = -95 / 9 := sorry

end NUMINAMATH_GPT_find_s_l405_40514


namespace NUMINAMATH_GPT_problem_3034_1002_20_04_div_sub_l405_40598

theorem problem_3034_1002_20_04_div_sub:
  3034 - (1002 / 20.04) = 2984 :=
by
  sorry

end NUMINAMATH_GPT_problem_3034_1002_20_04_div_sub_l405_40598


namespace NUMINAMATH_GPT_exist_nat_nums_l405_40588

theorem exist_nat_nums :
  ∃ (a b c d : ℕ), (a / (b : ℚ) + c / (d : ℚ) = 1) ∧ (a / (d : ℚ) + c / (b : ℚ) = 2008) :=
sorry

end NUMINAMATH_GPT_exist_nat_nums_l405_40588


namespace NUMINAMATH_GPT_china_junior_1990_problem_l405_40567

theorem china_junior_1990_problem 
  (x y z a b c : ℝ) 
  (hx : x ≠ 0) 
  (hy : y ≠ 0) 
  (hz : z ≠ 0) 
  (ha : a ≠ -1) 
  (hb : b ≠ -1) 
  (hc : c ≠ -1)
  (h1 : a * x = y * z / (y + z))
  (h2 : b * y = x * z / (x + z))
  (h3 : c * z = x * y / (x + y)) :
  (1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) = 1) :=
sorry

end NUMINAMATH_GPT_china_junior_1990_problem_l405_40567


namespace NUMINAMATH_GPT_vertices_integer_assignment_zero_l405_40544

theorem vertices_integer_assignment_zero (f : ℕ → ℤ) (h100 : ∀ i, i < 100 → (i + 3) % 100 < 100) 
  (h : ∀ i, (i < 97 → f i + f (i + 2) = f (i + 1)) 
            ∨ (i < 97 → f (i + 1) + f (i + 3) = f (i + 2)) 
            ∨ (i < 97 → f i + f (i + 1) = f (i + 2))): 
  ∀ i, i < 100 → f i = 0 :=
by
  sorry

end NUMINAMATH_GPT_vertices_integer_assignment_zero_l405_40544


namespace NUMINAMATH_GPT_sum_of_endpoints_l405_40576

noncomputable def triangle_side_length (PQ QR PR QS PS : ℝ) (h1 : PQ = 12) (h2 : QS = 4)
  (h3 : (PQ / PR) = (PS / QS)) : ℝ :=
  if 4 < PR ∧ PR < 18 then 4 + 18 else 0

theorem sum_of_endpoints {PQ PR QS PS : ℝ} (h1 : PQ = 12) (h2 : QS = 4)
  (h3 : (PQ / PR) = ( PS / QS)) :
  triangle_side_length PQ 0 PR QS PS h1 h2 h3 = 22 := by
  sorry

end NUMINAMATH_GPT_sum_of_endpoints_l405_40576


namespace NUMINAMATH_GPT_central_cell_value_l405_40577

theorem central_cell_value (a1 a2 a3 a4 a5 a6 a7 a8 C : ℕ) 
  (h1 : a1 + a3 + C = 13) (h2 : a2 + a4 + C = 13)
  (h3 : a5 + a7 + C = 13) (h4 : a6 + a8 + C = 13)
  (h5 : a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 = 40) : 
  C = 3 := 
sorry

end NUMINAMATH_GPT_central_cell_value_l405_40577


namespace NUMINAMATH_GPT_max_sqrt_expression_l405_40560

open Real

theorem max_sqrt_expression (x y z : ℝ) (h_sum : x + y + z = 3)
  (hx : x ≥ -1) (hy : y ≥ -(2/3)) (hz : z ≥ -2) :
  sqrt (3 * x + 3) + sqrt (3 * y + 2) + sqrt (3 * z + 6) ≤ 2 * sqrt 15 := by
  sorry

end NUMINAMATH_GPT_max_sqrt_expression_l405_40560


namespace NUMINAMATH_GPT_gumballs_difference_l405_40548

variable (x y : ℕ)

def total_gumballs := 16 + 12 + 20 + x + y
def avg_gumballs (T : ℕ) := T / 5

theorem gumballs_difference (h1 : 18 <= avg_gumballs (total_gumballs x y)) 
                            (h2 : avg_gumballs (total_gumballs x y) <= 27) : (87 - 42) = 45 := by
  sorry

end NUMINAMATH_GPT_gumballs_difference_l405_40548


namespace NUMINAMATH_GPT_team_a_daily_work_rate_l405_40562

theorem team_a_daily_work_rate
  (L : ℕ) (D1 : ℕ) (D2 : ℕ) (w : ℕ → ℕ)
  (hL : L = 8250)
  (hD1 : D1 = 4)
  (hD2 : D2 = 7)
  (hwB : ∀ (x : ℕ), w x = x + 150)
  (hwork : ∀ (x : ℕ), D1 * x + D2 * (x + (w x)) = L) :
  ∃ x : ℕ, x = 400 :=
by
  sorry

end NUMINAMATH_GPT_team_a_daily_work_rate_l405_40562


namespace NUMINAMATH_GPT_batsman_average_increase_l405_40536

theorem batsman_average_increase 
  (A : ℕ)
  (h1 : ∀ n ≤ 11, (1 / (n : ℝ)) * (A * n + 60) = 38) 
  (h2 : 1 / 12 * (A * 11 + 60) = 38)
  (h3 : ∀ n ≤ 12, (A * n : ℝ) ≤ (A * (n + 1) : ℝ)) :
  38 - A = 2 := 
sorry

end NUMINAMATH_GPT_batsman_average_increase_l405_40536


namespace NUMINAMATH_GPT_find_consecutive_numbers_l405_40515

theorem find_consecutive_numbers :
  ∃ (a b c d : ℕ),
      a % 11 = 0 ∧
      b % 7 = 0 ∧
      c % 5 = 0 ∧
      d % 4 = 0 ∧
      b = a + 1 ∧
      c = a + 2 ∧
      d = a + 3 ∧
      (a % 10) = 3 ∧
      (b % 10) = 4 ∧
      (c % 10) = 5 ∧
      (d % 10) = 6 :=
sorry

end NUMINAMATH_GPT_find_consecutive_numbers_l405_40515


namespace NUMINAMATH_GPT_find_number_of_observations_l405_40553

theorem find_number_of_observations 
  (n : ℕ) 
  (mean_before_correction : ℝ)
  (incorrect_observation : ℝ)
  (correct_observation : ℝ)
  (mean_after_correction : ℝ) 
  (h0 : mean_before_correction = 36)
  (h1 : incorrect_observation = 23)
  (h2 : correct_observation = 45)
  (h3 : mean_after_correction = 36.5) 
  (h4 : (n * mean_before_correction + (correct_observation - incorrect_observation)) / n = mean_after_correction) : 
  n = 44 := 
by
  sorry

end NUMINAMATH_GPT_find_number_of_observations_l405_40553


namespace NUMINAMATH_GPT_maximum_ratio_x_over_y_l405_40556

theorem maximum_ratio_x_over_y {x y : ℕ} (hx : x > 9 ∧ x < 100) (hy : y > 9 ∧ y < 100)
  (hmean : x + y = 110) (hsquare : ∃ z : ℕ, z^2 = x * y) : x = 99 ∧ y = 11 := 
by
  -- mathematical proof
  sorry

end NUMINAMATH_GPT_maximum_ratio_x_over_y_l405_40556


namespace NUMINAMATH_GPT_part_a_part_b_l405_40585

theorem part_a (A B : ℕ) (hA : 1 ≤ A) (hB : 1 ≤ B) : 
  (A + B = 70) → 
  (A * (4 : ℚ) / 35 + B * (4 : ℚ) / 35 = 8) :=
  by
    sorry

theorem part_b (C D : ℕ) (r : ℚ) (hC : C > 1) (hD : D > 1) (hr : r > 1) :
  (C + D = 8 / r) → 
  (C * r + D * r = 8) → 
  (∃ ki : ℕ, (C + D = (70 : ℕ) / ki ∧ 1 < ki ∧ ki ∣ 70)) :=
  by
    sorry

end NUMINAMATH_GPT_part_a_part_b_l405_40585


namespace NUMINAMATH_GPT_hands_per_student_l405_40573

theorem hands_per_student (hands_without_peter : ℕ) (total_students : ℕ) (hands_peter : ℕ) 
  (h1 : hands_without_peter = 20) 
  (h2 : total_students = 11) 
  (h3 : hands_peter = 2) : 
  (hands_without_peter + hands_peter) / total_students = 2 :=
by
  sorry

end NUMINAMATH_GPT_hands_per_student_l405_40573


namespace NUMINAMATH_GPT_complex_fraction_identity_l405_40557

theorem complex_fraction_identity
  (a b : ℂ) (ζ : ℂ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : ζ ^ 3 = 1) (h4 : ζ ≠ 1) 
  (h5 : a ^ 2 + a * b + b ^ 2 = 0) :
  (a ^ 9 + b ^ 9) / ((a - b) ^ 9) = (2 : ℂ) / (81 * (ζ - 1)) :=
sorry

end NUMINAMATH_GPT_complex_fraction_identity_l405_40557


namespace NUMINAMATH_GPT_volume_of_rectangular_solid_l405_40572

theorem volume_of_rectangular_solid 
  (a b c : ℝ)
  (h1 : a * b = 15)
  (h2 : b * c = 10)
  (h3 : c * a = 6) :
  a * b * c = 30 := 
by
  -- sorry placeholder for the proof
  sorry

end NUMINAMATH_GPT_volume_of_rectangular_solid_l405_40572


namespace NUMINAMATH_GPT_polynomial_remainder_division_l405_40574

theorem polynomial_remainder_division :
  ∀ (x : ℝ), (x^4 + 2 * x^2 - 3) % (x^2 + 3 * x + 2) = -21 * x - 21 := 
by
  sorry

end NUMINAMATH_GPT_polynomial_remainder_division_l405_40574


namespace NUMINAMATH_GPT_proof_problem_l405_40517

-- Defining the statement in Lean 4.

noncomputable def p : Prop :=
  ∀ x : ℝ, x > Real.sin x

noncomputable def neg_p : Prop :=
  ∃ x : ℝ, x ≤ Real.sin x

theorem proof_problem : ¬p ↔ neg_p := 
by sorry

end NUMINAMATH_GPT_proof_problem_l405_40517


namespace NUMINAMATH_GPT_interior_angle_of_arithmetic_sequence_triangle_l405_40506

theorem interior_angle_of_arithmetic_sequence_triangle :
  ∀ (α d : ℝ), (α - d) + α + (α + d) = 180 → α = 60 :=
by 
  sorry

end NUMINAMATH_GPT_interior_angle_of_arithmetic_sequence_triangle_l405_40506


namespace NUMINAMATH_GPT_greatest_prime_factor_180_l405_40512

noncomputable def is_prime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ n : ℕ, n ∣ p → n = 1 ∨ n = p

theorem greatest_prime_factor_180 : 
  ∃ p : ℕ, is_prime p ∧ p ∣ 180 ∧ ∀ q : ℕ, is_prime q ∧ q ∣ 180 → q ≤ p :=
  sorry

end NUMINAMATH_GPT_greatest_prime_factor_180_l405_40512


namespace NUMINAMATH_GPT_shaded_region_area_l405_40554

-- Definitions based on given conditions
def num_squares : ℕ := 25
def diagonal_length : ℝ := 10
def squares_in_large_square : ℕ := 16

-- The area of the entire shaded region
def area_of_shaded_region : ℝ := 78.125

-- Theorem to prove 
theorem shaded_region_area 
  (num_squares : ℕ) 
  (diagonal_length : ℝ) 
  (squares_in_large_square : ℕ) : 
  (num_squares = 25) → 
  (diagonal_length = 10) → 
  (squares_in_large_square = 16) → 
  area_of_shaded_region = 78.125 := 
by {
  sorry -- proof to be filled
}

end NUMINAMATH_GPT_shaded_region_area_l405_40554


namespace NUMINAMATH_GPT_susan_strawberries_per_handful_l405_40523

-- Definitions of the given conditions
def total_picked := 75
def total_needed := 60
def strawberries_per_handful := 5

-- Derived conditions
def total_eaten := total_picked - total_needed
def number_of_handfuls := total_picked / strawberries_per_handful
def strawberries_eaten_per_handful := total_eaten / number_of_handfuls

-- The theorem we want to prove
theorem susan_strawberries_per_handful : strawberries_eaten_per_handful = 1 :=
by sorry

end NUMINAMATH_GPT_susan_strawberries_per_handful_l405_40523


namespace NUMINAMATH_GPT_find_p_from_binomial_distribution_l405_40571

theorem find_p_from_binomial_distribution (p : ℝ) (h₁ : 0 ≤ p ∧ p ≤ 1) 
    (h₂ : ∀ n k : ℕ, k ≤ n → 0 ≤ p^(k:ℝ) * (1-p)^((n-k):ℝ)) 
    (h₃ : (1 - (1 - p)^2 = 5 / 9)) : p = 1 / 3 :=
by sorry

end NUMINAMATH_GPT_find_p_from_binomial_distribution_l405_40571


namespace NUMINAMATH_GPT_total_hours_worked_l405_40531

theorem total_hours_worked :
  (∃ (hours_per_day : ℕ) (days : ℕ), hours_per_day = 3 ∧ days = 6) →
  (∃ (total_hours : ℕ), total_hours = 18) :=
by
  intros
  sorry

end NUMINAMATH_GPT_total_hours_worked_l405_40531


namespace NUMINAMATH_GPT_Karsyn_payment_l405_40507

def percentage : ℝ := 20
def initial_price : ℝ := 600

theorem Karsyn_payment : (percentage / 100) * initial_price = 120 :=
by 
  sorry

end NUMINAMATH_GPT_Karsyn_payment_l405_40507


namespace NUMINAMATH_GPT_distance_left_to_drive_l405_40540

theorem distance_left_to_drive (total_distance : ℕ) (distance_driven : ℕ) 
  (h1 : total_distance = 78) (h2 : distance_driven = 32) : 
  total_distance - distance_driven = 46 := by
  sorry

end NUMINAMATH_GPT_distance_left_to_drive_l405_40540


namespace NUMINAMATH_GPT_max_value_exponent_l405_40510

theorem max_value_exponent {a b : ℝ} (h : 0 < b ∧ b < a ∧ a < 1) :
  max (max (a^b) (b^a)) (max (a^a) (b^b)) = a^b :=
sorry

end NUMINAMATH_GPT_max_value_exponent_l405_40510


namespace NUMINAMATH_GPT_brad_siblings_product_l405_40561

theorem brad_siblings_product (S B : ℕ) (hS : S = 5) (hB : B = 7) : S * B = 35 :=
by
  have : S = 5 := hS
  have : B = 7 := hB
  sorry

end NUMINAMATH_GPT_brad_siblings_product_l405_40561


namespace NUMINAMATH_GPT_find_value_l405_40595

noncomputable def S2013 (x y : ℂ) (h : x ≠ 0 ∧ y ≠ 0) (h_eq : x^2 + x * y + y^2 = 0) : ℂ :=
  (x / (x + y))^2013 + (y / (x + y))^2013

theorem find_value (x y : ℂ) (h : x ≠ 0 ∧ y ≠ 0) (h_eq : x^2 + x * y + y^2 = 0) :
  S2013 x y h h_eq = -2 :=
sorry

end NUMINAMATH_GPT_find_value_l405_40595


namespace NUMINAMATH_GPT_prove_smallest_geometric_third_term_value_l405_40502

noncomputable def smallest_value_geometric_third_term : ℝ :=
  let d_1 := -5 + 10 * Real.sqrt 2
  let d_2 := -5 - 10 * Real.sqrt 2
  let g3_1 := 39 + 2 * d_1
  let g3_2 := 39 + 2 * d_2
  min g3_1 g3_2

theorem prove_smallest_geometric_third_term_value :
  smallest_value_geometric_third_term = 29 - 20 * Real.sqrt 2 := by sorry

end NUMINAMATH_GPT_prove_smallest_geometric_third_term_value_l405_40502


namespace NUMINAMATH_GPT_number_of_tables_l405_40538

noncomputable def stools_per_table : ℕ := 7
noncomputable def legs_per_stool : ℕ := 4
noncomputable def legs_per_table : ℕ := 5
noncomputable def total_legs : ℕ := 658

theorem number_of_tables : 
  ∃ t : ℕ, 
  (∃ s : ℕ, s = stools_per_table * t ∧ legs_per_stool * s + legs_per_table * t = total_legs) ∧ t = 20 :=
by {
  sorry
}

end NUMINAMATH_GPT_number_of_tables_l405_40538


namespace NUMINAMATH_GPT_mirella_read_more_pages_l405_40539

-- Define the number of books Mirella read
def num_purple_books := 8
def num_orange_books := 7
def num_blue_books := 5

-- Define the number of pages per book for each color
def pages_per_purple_book := 320
def pages_per_orange_book := 640
def pages_per_blue_book := 450

-- Calculate the total pages for each color
def total_purple_pages := num_purple_books * pages_per_purple_book
def total_orange_pages := num_orange_books * pages_per_orange_book
def total_blue_pages := num_blue_books * pages_per_blue_book

-- Calculate the combined total of orange and blue pages
def total_orange_blue_pages := total_orange_pages + total_blue_pages

-- Define the target value
def page_difference := 4170

-- State the theorem to prove
theorem mirella_read_more_pages :
  total_orange_blue_pages - total_purple_pages = page_difference := by
  sorry

end NUMINAMATH_GPT_mirella_read_more_pages_l405_40539


namespace NUMINAMATH_GPT_no_solution_m1_no_solution_m2_solution_m3_l405_40581

-- Problem 1: No positive integer solutions for m = 1
theorem no_solution_m1 (x y z : ℕ) (h: x > 0 ∧ y > 0 ∧ z > 0) : x^3 + y^3 + z^3 ≠ x * y * z := sorry

-- Problem 2: No positive integer solutions for m = 2
theorem no_solution_m2 (x y z : ℕ) (h: x > 0 ∧ y > 0 ∧ z > 0) : x^3 + y^3 + z^3 ≠ 2 * x * y * z := sorry

-- Problem 3: Only solutions for m = 3 are x = y = z = k for some k
theorem solution_m3 (x y z : ℕ) (h: x > 0 ∧ y > 0 ∧ z > 0) : x^3 + y^3 + z^3 = 3 * x * y * z ↔ x = y ∧ y = z := sorry

end NUMINAMATH_GPT_no_solution_m1_no_solution_m2_solution_m3_l405_40581


namespace NUMINAMATH_GPT_find_y_at_neg3_l405_40532

noncomputable def quadratic_solution (y x a b : ℝ) : Prop :=
  y = x ^ 2 + a * x + b

theorem find_y_at_neg3
    (a b : ℝ)
    (h1 : 1 + a + b = 2)
    (h2 : 4 - 2 * a + b = -1)
    : quadratic_solution 2 (-3) a b :=
by
  sorry

end NUMINAMATH_GPT_find_y_at_neg3_l405_40532


namespace NUMINAMATH_GPT_value_of_star_15_25_l405_40575

noncomputable def star (x y : ℝ) : ℝ := Real.log x / Real.log y

axiom condition1 (x y : ℝ) (hxy : x > 0 ∧ y > 0) : star (star (x^2) y) y = star x y
axiom condition2 (x y : ℝ) (hxy : x > 0 ∧ y > 0) : star x (star y y) = star (star x y) (star x 1)
axiom condition3 (h : 1 > 0) : star 1 1 = 0

theorem value_of_star_15_25 : star 15 25 = (Real.log 3 / (2 * Real.log 5)) + 1 / 2 := 
by 
  sorry

end NUMINAMATH_GPT_value_of_star_15_25_l405_40575


namespace NUMINAMATH_GPT_largest_x_quadratic_inequality_l405_40547

theorem largest_x_quadratic_inequality : 
  ∃ (x : ℝ), (x^2 - 10 * x + 24 ≤ 0) ∧ (∀ y, (y^2 - 10 * y + 24 ≤ 0) → y ≤ x) :=
sorry

end NUMINAMATH_GPT_largest_x_quadratic_inequality_l405_40547


namespace NUMINAMATH_GPT_evaluate_expression_l405_40513

noncomputable def a : ℝ := 2 * Real.sqrt 2 + 3 * Real.sqrt 3 + 4 * Real.sqrt 6
noncomputable def b : ℝ := -2 * Real.sqrt 2 + 3 * Real.sqrt 3 + 4 * Real.sqrt 6
noncomputable def c : ℝ := 2 * Real.sqrt 2 - 3 * Real.sqrt 3 + 4 * Real.sqrt 6
noncomputable def d : ℝ := -2 * Real.sqrt 2 - 3 * Real.sqrt 3 + 4 * Real.sqrt 6

theorem evaluate_expression : (1/a + 1/b + 1/c + 1/d)^2 = 952576 / 70225 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l405_40513


namespace NUMINAMATH_GPT_find_f_of_3pi_by_4_l405_40558

noncomputable def f (x : ℝ) : ℝ := Real.sin (x + Real.pi / 2)

theorem find_f_of_3pi_by_4 : f (3 * Real.pi / 4) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_GPT_find_f_of_3pi_by_4_l405_40558


namespace NUMINAMATH_GPT_odd_function_condition_l405_40528

noncomputable def f (x a b : ℝ) : ℝ := x * abs (x + a) + b

theorem odd_function_condition (a b : ℝ) :
  (∀ x : ℝ, f (-x) a b = -f x a b) ↔ (a = 0 ∧ b = 0) :=
by
  sorry

end NUMINAMATH_GPT_odd_function_condition_l405_40528


namespace NUMINAMATH_GPT_interior_edges_sum_l405_40583

theorem interior_edges_sum 
  (frame_thickness : ℝ)
  (frame_area : ℝ)
  (outer_length : ℝ)
  (frame_thickness_eq : frame_thickness = 2)
  (frame_area_eq : frame_area = 32)
  (outer_length_eq : outer_length = 7) 
  : ∃ interior_edges_sum : ℝ, interior_edges_sum = 8 := 
by
  sorry

end NUMINAMATH_GPT_interior_edges_sum_l405_40583


namespace NUMINAMATH_GPT_relationship_between_y_and_x_fuel_remaining_after_35_kilometers_max_distance_without_refueling_l405_40521

variable (x y : ℝ)

-- Assume the initial fuel and consumption rate
def initial_fuel : ℝ := 48
def consumption_rate : ℝ := 0.6

-- Define the fuel consumption equation
def fuel_equation (distance : ℝ) : ℝ := -consumption_rate * distance + initial_fuel

-- Theorem proving the fuel equation satisfies the specific conditions
theorem relationship_between_y_and_x :
  ∀ (x : ℝ), y = fuel_equation x :=
by
  sorry

-- Theorem proving the fuel remaining after traveling 35 kilometers
theorem fuel_remaining_after_35_kilometers :
  fuel_equation 35 = 27 :=
by
  sorry

-- Theorem proving the maximum distance the car can travel without refueling
theorem max_distance_without_refueling :
  ∃ (x : ℝ), fuel_equation x = 0 ∧ x = 80 :=
by
  sorry

end NUMINAMATH_GPT_relationship_between_y_and_x_fuel_remaining_after_35_kilometers_max_distance_without_refueling_l405_40521


namespace NUMINAMATH_GPT_area_of_T_prime_l405_40503

-- Given conditions
def AreaBeforeTransformation : ℝ := 9

def TransformationMatrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![3, 4],![-2, 5]]

def AreaAfterTransformation (M : Matrix (Fin 2) (Fin 2) ℝ) (area_before : ℝ) : ℝ :=
  (M.det) * area_before

-- Problem statement
theorem area_of_T_prime : 
  AreaAfterTransformation TransformationMatrix AreaBeforeTransformation = 207 :=
by
  sorry

end NUMINAMATH_GPT_area_of_T_prime_l405_40503


namespace NUMINAMATH_GPT_increase_in_cost_l405_40552

def initial_lumber_cost : ℝ := 450
def initial_nails_cost : ℝ := 30
def initial_fabric_cost : ℝ := 80
def lumber_inflation_rate : ℝ := 0.20
def nails_inflation_rate : ℝ := 0.10
def fabric_inflation_rate : ℝ := 0.05

def initial_total_cost : ℝ := initial_lumber_cost + initial_nails_cost + initial_fabric_cost

def new_lumber_cost : ℝ := initial_lumber_cost * (1 + lumber_inflation_rate)
def new_nails_cost : ℝ := initial_nails_cost * (1 + nails_inflation_rate)
def new_fabric_cost : ℝ := initial_fabric_cost * (1 + fabric_inflation_rate)

def new_total_cost : ℝ := new_lumber_cost + new_nails_cost + new_fabric_cost

theorem increase_in_cost :
  new_total_cost - initial_total_cost = 97 := 
sorry

end NUMINAMATH_GPT_increase_in_cost_l405_40552


namespace NUMINAMATH_GPT_price_of_each_movie_in_first_box_l405_40534

theorem price_of_each_movie_in_first_box (P : ℝ) (total_movies_box1 : ℕ) (total_movies_box2 : ℕ) (price_per_movie_box2 : ℝ) (average_price : ℝ) (total_movies : ℕ) :
  total_movies_box1 = 10 →
  total_movies_box2 = 5 →
  price_per_movie_box2 = 5 →
  average_price = 3 →
  total_movies = 15 →
  10 * P + 5 * price_per_movie_box2 = average_price * total_movies →
  P = 2 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_price_of_each_movie_in_first_box_l405_40534


namespace NUMINAMATH_GPT_find_pq_l405_40593

noncomputable def area_of_triangle (p q : ℝ) : ℝ := 1/2 * (12 / p) * (12 / q)

theorem find_pq (p q : ℝ) (hp : p > 0) (hq : q > 0) (harea : area_of_triangle p q = 12) : p * q = 6 := 
by
  sorry

end NUMINAMATH_GPT_find_pq_l405_40593


namespace NUMINAMATH_GPT_binkie_gemstones_l405_40526

noncomputable def gemstones_solution : ℕ :=
sorry

theorem binkie_gemstones : ∀ (Binkie Frankie Spaatz Whiskers Snowball : ℕ),
  Spaatz = 1 ∧
  Whiskers = Spaatz + 3 ∧
  Snowball = 2 * Whiskers ∧ 
  Snowball % 2 = 0 ∧
  Whiskers % 2 = 0 ∧
  Spaatz = (1 / 2 * Frankie) - 2 ∧
  Binkie = 4 * Frankie ∧
  Binkie + Frankie + Spaatz + Whiskers + Snowball <= 50 →
  Binkie = 24 :=
sorry

end NUMINAMATH_GPT_binkie_gemstones_l405_40526


namespace NUMINAMATH_GPT_probability_within_three_units_from_origin_l405_40565

-- Define the properties of the square Q is selected from
def isInSquare (Q : ℝ × ℝ) : Prop :=
  Q.1 ≥ -2 ∧ Q.1 ≤ 2 ∧ Q.2 ≥ -2 ∧ Q.2 ≤ 2

-- Define the condition of being within 3 units from the origin
def withinThreeUnits (Q: ℝ × ℝ) : Prop :=
  (Q.1)^2 + (Q.2)^2 ≤ 9

-- State the problem: Proving the probability is 1
theorem probability_within_three_units_from_origin : 
  ∀ (Q : ℝ × ℝ), isInSquare Q → withinThreeUnits Q := 
by 
  sorry

end NUMINAMATH_GPT_probability_within_three_units_from_origin_l405_40565


namespace NUMINAMATH_GPT_container_holds_slices_l405_40509

theorem container_holds_slices (x : ℕ) 
  (h1 : x > 1) 
  (h2 : x ≠ 332) 
  (h3 : x ≠ 166) 
  (h4 : x ∣ 332) :
  x = 83 := 
sorry

end NUMINAMATH_GPT_container_holds_slices_l405_40509


namespace NUMINAMATH_GPT_min_shift_symmetric_y_axis_l405_40586

theorem min_shift_symmetric_y_axis :
  ∃ (m : ℝ), m = 7 * Real.pi / 6 ∧ 
             (∀ x : ℝ, 2 * Real.cos (x + Real.pi / 3) = 2 * Real.cos (x + Real.pi / 3 + m)) ∧ 
             m > 0 :=
by
  sorry

end NUMINAMATH_GPT_min_shift_symmetric_y_axis_l405_40586
