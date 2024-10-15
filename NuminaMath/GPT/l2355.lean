import Mathlib

namespace NUMINAMATH_GPT_divisibility_by_cube_greater_than_1_l2355_235579

theorem divisibility_by_cube_greater_than_1 (a b : ℕ) (hpos_a : 0 < a) (hpos_b : 0 < b)
  (hdiv : (a + b^3) % (a^2 + 3 * a * b + 3 * b^2 - 1) = 0) :
  ∃ k : ℕ, 1 < k ∧ k^3 ∣ a^2 + 3 * a * b + 3 * b^2 - 1 := 
by {
  sorry
}

end NUMINAMATH_GPT_divisibility_by_cube_greater_than_1_l2355_235579


namespace NUMINAMATH_GPT_parabola_directrix_equation_l2355_235513

theorem parabola_directrix_equation (x y a : ℝ) : 
  (x^2 = 4 * y) → (a = 1) → (y = -a) := by
  intro h1 h2
  rw [h2] -- given a = 1
  sorry

end NUMINAMATH_GPT_parabola_directrix_equation_l2355_235513


namespace NUMINAMATH_GPT_dad_vacuum_time_l2355_235584

theorem dad_vacuum_time (x : ℕ) (h1 : 2 * x + 5 = 27) (h2 : x + (2 * x + 5) = 38) :
  (2 * x + 5) = 27 := by
  sorry

end NUMINAMATH_GPT_dad_vacuum_time_l2355_235584


namespace NUMINAMATH_GPT_circle_possible_values_l2355_235590

theorem circle_possible_values (a : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + a * x + 2 * a * y + 2 * a^2 + a - 1 = 0 → -2 < a ∧ a < 2/3) := sorry

end NUMINAMATH_GPT_circle_possible_values_l2355_235590


namespace NUMINAMATH_GPT_find_y_l2355_235502

theorem find_y (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 18) : y = 5 :=
sorry

end NUMINAMATH_GPT_find_y_l2355_235502


namespace NUMINAMATH_GPT_greatest_number_of_fruit_baskets_l2355_235540

def number_of_oranges : ℕ := 18
def number_of_pears : ℕ := 27
def number_of_bananas : ℕ := 12

theorem greatest_number_of_fruit_baskets :
  Nat.gcd (Nat.gcd number_of_oranges number_of_pears) number_of_bananas = 3 :=
by
  sorry

end NUMINAMATH_GPT_greatest_number_of_fruit_baskets_l2355_235540


namespace NUMINAMATH_GPT_no_valid_n_lt_200_l2355_235561

noncomputable def roots_are_consecutive (n m : ℕ) : Prop :=
  ∃ k : ℕ, m = k * (k + 1) ∧ n = 2 * k + 1

theorem no_valid_n_lt_200 :
  ¬∃ n m : ℕ, n < 200 ∧
              m % 4 = 0 ∧
              ∃ t : ℕ, t^2 = m ∧
              roots_are_consecutive n m := 
by
  sorry

end NUMINAMATH_GPT_no_valid_n_lt_200_l2355_235561


namespace NUMINAMATH_GPT_circles_equal_or_tangent_l2355_235571

theorem circles_equal_or_tangent (a b c : ℝ) 
  (h : (2 * a)^2 - 4 * (b^2 - c * (b - a)) = 0) : 
  a = b ∨ c = a + b :=
by
  -- Will fill the proof later
  sorry

end NUMINAMATH_GPT_circles_equal_or_tangent_l2355_235571


namespace NUMINAMATH_GPT_arithmetic_series_remainder_l2355_235503

-- Define the sequence parameters
def a : ℕ := 2
def l : ℕ := 12
def d : ℕ := 1
def n : ℕ := (l - a) / d + 1

-- Define the sum of the arithmetic series
def S : ℕ := n * (a + l) / 2

-- The final theorem statement
theorem arithmetic_series_remainder : S % 9 = 5 := 
by sorry

end NUMINAMATH_GPT_arithmetic_series_remainder_l2355_235503


namespace NUMINAMATH_GPT_concentration_of_salt_solution_l2355_235517

-- Conditions:
def total_volume : ℝ := 1 + 0.25
def concentration_of_mixture : ℝ := 0.15
def volume_of_salt_solution : ℝ := 0.25

-- Expression for the concentration of the salt solution used, $C$:
theorem concentration_of_salt_solution (C : ℝ) :
  (volume_of_salt_solution * (C / 100)) = (total_volume * concentration_of_mixture) → C = 75 := by
  sorry

end NUMINAMATH_GPT_concentration_of_salt_solution_l2355_235517


namespace NUMINAMATH_GPT_minimize_rental_cost_l2355_235528

def travel_agency (x y : ℕ) : ℕ := 1600 * x + 2400 * y

theorem minimize_rental_cost :
    ∃ (x y : ℕ), (x + y ≤ 21) ∧ (y ≤ x + 7) ∧ (36 * x + 60 * y = 900) ∧ 
    (∀ (a b : ℕ), (a + b ≤ 21) ∧ (b ≤ a + 7) ∧ (36 * a + 60 * b = 900) → travel_agency a b ≥ travel_agency x y) ∧
    travel_agency x y = 36800 :=
sorry

end NUMINAMATH_GPT_minimize_rental_cost_l2355_235528


namespace NUMINAMATH_GPT_quadrilateral_ABCD_is_rectangle_l2355_235583

noncomputable def point := (ℤ × ℤ)

def A : point := (-2, 0)
def B : point := (1, 6)
def C : point := (5, 4)
def D : point := (2, -2)

def vector (p1 p2 : point) : point := (p2.1 - p1.1, p2.2 - p1.2)

def dot_product (v1 v2 : point) : ℤ := (v1.1 * v2.1) + (v1.2 * v2.2)

def is_perpendicular (v1 v2 : point) : Prop := dot_product v1 v2 = 0

def is_rectangle (A B C D : point) :=
  vector A B = vector C D ∧ is_perpendicular (vector A B) (vector A D)

theorem quadrilateral_ABCD_is_rectangle : is_rectangle A B C D :=
by
  sorry

end NUMINAMATH_GPT_quadrilateral_ABCD_is_rectangle_l2355_235583


namespace NUMINAMATH_GPT_arcsin_one_half_l2355_235509

theorem arcsin_one_half : Real.arcsin (1 / 2) = Real.pi / 6 :=
by
  -- Conditions
  have h1 : -Real.pi / 2 ≤ Real.pi / 6 ∧ Real.pi / 6 ≤ Real.pi / 2 := by
    -- Proof the range of pi/6 is within [-pi/2, pi/2]
    sorry
  have h2 : ∀ x, Real.sin x = 1 / 2 → x = Real.pi / 6 := by
    -- Proof sin(pi/6) = 1 / 2
    sorry
  show Real.arcsin (1 / 2) = Real.pi / 6
  -- Proof arcsin(1/2) = pi/6 based on the above conditions
  sorry

end NUMINAMATH_GPT_arcsin_one_half_l2355_235509


namespace NUMINAMATH_GPT_slope_of_line_l2355_235599

theorem slope_of_line (P Q : ℝ × ℝ) (hP : P = (1, 2)) (hQ : Q = (4, 3)) :
  (Q.snd - P.snd) / (Q.fst - P.fst) = 1 / 3 := by
  sorry

end NUMINAMATH_GPT_slope_of_line_l2355_235599


namespace NUMINAMATH_GPT_parabola_directrix_l2355_235565

theorem parabola_directrix (x y : ℝ) : 
    (x^2 = (1/2) * y) -> (y = -1/8) :=
sorry

end NUMINAMATH_GPT_parabola_directrix_l2355_235565


namespace NUMINAMATH_GPT_sum_of_first_17_terms_l2355_235597

variable {α : Type*} [LinearOrderedField α] 

-- conditions
def arithmetic_sequence (a : ℕ → α) : Prop := 
  ∃ d : α, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → α) (S : ℕ → α) : Prop :=
  ∀ n : ℕ, S n = (n * (a 1 + a n)) / 2

variable {a : ℕ → α}
variable {S : ℕ → α}

-- main theorem
theorem sum_of_first_17_terms (h_arith : arithmetic_sequence a)
  (h_S : sum_of_first_n_terms a S)
  (h_condition : a 7 + a 12 = 12 - a 8) :
  S 17 = 68 := sorry

end NUMINAMATH_GPT_sum_of_first_17_terms_l2355_235597


namespace NUMINAMATH_GPT_eight_sharp_two_equals_six_thousand_l2355_235568

def new_operation (a b : ℕ) : ℕ :=
  (a + b) ^ 3 * (a - b)

theorem eight_sharp_two_equals_six_thousand : new_operation 8 2 = 6000 := 
  by
    sorry

end NUMINAMATH_GPT_eight_sharp_two_equals_six_thousand_l2355_235568


namespace NUMINAMATH_GPT_guide_is_knight_l2355_235577

-- Definitions
def knight (p : Prop) : Prop := p
def liar (p : Prop) : Prop := ¬p

-- Conditions
variable (GuideClaimsKnight : Prop)
variable (SecondResidentClaimsKnight : Prop)
variable (GuideReportsAccurately : Prop)

-- Proof problem
theorem guide_is_knight
  (GuideClaimsKnight : Prop)
  (SecondResidentClaimsKnight : Prop)
  (GuideReportsAccurately : (GuideClaimsKnight ↔ SecondResidentClaimsKnight)) :
  GuideClaimsKnight := 
sorry

end NUMINAMATH_GPT_guide_is_knight_l2355_235577


namespace NUMINAMATH_GPT_problem_1_problem_2_l2355_235555

def f (x : ℝ) : ℝ := |(1 - 2 * x)| - |(1 + x)|

theorem problem_1 :
  {x | f x ≥ 4} = {x | x ≤ -2 ∨ x ≥ 6} :=
sorry

theorem problem_2 (a : ℝ) : 
  (∀ x : ℝ, a^2 + 2 * a + |(1 + x)| > f x) → (a < -3 ∨ a > 1) :=
sorry

end NUMINAMATH_GPT_problem_1_problem_2_l2355_235555


namespace NUMINAMATH_GPT_cube_edge_length_l2355_235525

theorem cube_edge_length
  (length_base : ℝ) (width_base : ℝ) (rise_level : ℝ) (volume_displaced : ℝ) (volume_cube : ℝ) (edge_length : ℝ)
  (h_base : length_base = 20) (h_width : width_base = 15) (h_rise : rise_level = 3.3333333333333335)
  (h_volume_displaced : volume_displaced = length_base * width_base * rise_level)
  (h_volume_cube : volume_cube = volume_displaced)
  (h_edge_length_eq : volume_cube = edge_length ^ 3)
  : edge_length = 10 :=
by
  sorry

end NUMINAMATH_GPT_cube_edge_length_l2355_235525


namespace NUMINAMATH_GPT_A_equals_half_C_equals_half_l2355_235551

noncomputable def A := 2 * Real.sin (75 * Real.pi / 180) * Real.cos (75 * Real.pi / 180)
noncomputable def C := Real.sin (45 * Real.pi / 180) * Real.cos (15 * Real.pi / 180) - Real.cos (45 * Real.pi / 180) * Real.sin (15 * Real.pi / 180)

theorem A_equals_half : A = 1 / 2 := 
by
  sorry

theorem C_equals_half : C = 1 / 2 := 
by
  sorry

end NUMINAMATH_GPT_A_equals_half_C_equals_half_l2355_235551


namespace NUMINAMATH_GPT_negation_proposition_equiv_l2355_235556

open Classical

variable (R : Type) [OrderedRing R] (a x : R)

theorem negation_proposition_equiv :
  (¬ ∃ a : R, ∃ x : R, a * x^2 + 1 = 0) ↔ (∀ a : R, ∀ x : R, a * x^2 + 1 ≠ 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_proposition_equiv_l2355_235556


namespace NUMINAMATH_GPT_prob_truth_same_time_l2355_235515

theorem prob_truth_same_time (pA pB : ℝ) (hA : pA = 0.85) (hB : pB = 0.60) :
  pA * pB = 0.51 :=
by
  rw [hA, hB]
  norm_num

end NUMINAMATH_GPT_prob_truth_same_time_l2355_235515


namespace NUMINAMATH_GPT_complement_set_l2355_235557

-- Define the universal set U
def U : Set ℝ := Set.univ

-- Define the set M
def M : Set ℝ := {x | x^2 - 4 ≤ 0}

-- Define the complement of M in U
def complement_M_in_U : Set ℝ := {x | x < -2 ∨ x > 2}

-- The mathematical proof to be stated
theorem complement_set :
  U \ M = complement_M_in_U := sorry

end NUMINAMATH_GPT_complement_set_l2355_235557


namespace NUMINAMATH_GPT_prime_polynomial_l2355_235522

theorem prime_polynomial (n : ℕ) (h1 : 2 ≤ n)
  (h2 : ∀ k : ℕ, k ≤ Nat.sqrt (n / 3) → Nat.Prime (k^2 + k + n)) :
  ∀ k : ℕ, k ≤ n - 2 → Nat.Prime (k^2 + k + n) :=
sorry

end NUMINAMATH_GPT_prime_polynomial_l2355_235522


namespace NUMINAMATH_GPT_find_numbers_l2355_235578

def hundreds_digit (n : ℕ) : ℕ := (n / 100) % 10
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10
def units_digit (n : ℕ) : ℕ := n % 10

def is_three_digit_number (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

noncomputable def number1 := 986
noncomputable def number2 := 689

theorem find_numbers :
  is_three_digit_number number1 ∧ is_three_digit_number number2 ∧
  hundreds_digit number1 = units_digit number2 ∧ hundreds_digit number2 = units_digit number1 ∧
  number1 - number2 = 297 ∧ (hundreds_digit number2 + tens_digit number2 + units_digit number2) = 23 :=
by
  sorry

end NUMINAMATH_GPT_find_numbers_l2355_235578


namespace NUMINAMATH_GPT_B_days_to_complete_job_alone_l2355_235504

theorem B_days_to_complete_job_alone (x : ℝ) : 
  (1 / 15 + 1 / x) * 4 = 0.4666666666666667 → x = 20 :=
by
  intro h
  -- Note: The proof is omitted as we only need the statement here.
  sorry

end NUMINAMATH_GPT_B_days_to_complete_job_alone_l2355_235504


namespace NUMINAMATH_GPT_starting_current_ratio_l2355_235541

theorem starting_current_ratio (running_current : ℕ) (units : ℕ) (total_current : ℕ)
    (h1 : running_current = 40) 
    (h2 : units = 3) 
    (h3 : total_current = 240) 
    (h4 : total_current = running_current * (units * starter_ratio)) :
    starter_ratio = 2 := 
sorry

end NUMINAMATH_GPT_starting_current_ratio_l2355_235541


namespace NUMINAMATH_GPT_tax_budget_level_correct_l2355_235505

-- Definitions for tax types and their corresponding budget levels
inductive TaxType where
| property_tax_organizations : TaxType
| federal_tax : TaxType
| profit_tax_organizations : TaxType
| tax_subjects_RF : TaxType
| transport_collecting : TaxType
deriving DecidableEq

inductive BudgetLevel where
| federal_budget : BudgetLevel
| subjects_RF_budget : BudgetLevel
deriving DecidableEq

def tax_to_budget_level : TaxType → BudgetLevel
| TaxType.property_tax_organizations => BudgetLevel.subjects_RF_budget
| TaxType.federal_tax => BudgetLevel.federal_budget
| TaxType.profit_tax_organizations => BudgetLevel.subjects_RF_budget
| TaxType.tax_subjects_RF => BudgetLevel.subjects_RF_budget
| TaxType.transport_collecting => BudgetLevel.subjects_RF_budget

theorem tax_budget_level_correct :
  tax_to_budget_level TaxType.property_tax_organizations = BudgetLevel.subjects_RF_budget ∧
  tax_to_budget_level TaxType.federal_tax = BudgetLevel.federal_budget ∧
  tax_to_budget_level TaxType.profit_tax_organizations = BudgetLevel.subjects_RF_budget ∧
  tax_to_budget_level TaxType.tax_subjects_RF = BudgetLevel.subjects_RF_budget ∧
  tax_to_budget_level TaxType.transport_collecting = BudgetLevel.subjects_RF_budget :=
by
  sorry

end NUMINAMATH_GPT_tax_budget_level_correct_l2355_235505


namespace NUMINAMATH_GPT_count_even_numbers_between_500_and_800_l2355_235585

theorem count_even_numbers_between_500_and_800 :
  let a := 502
  let d := 2
  let last_term := 798
  ∃ n, a + (n - 1) * d = last_term ∧ n = 149 :=
by
  sorry

end NUMINAMATH_GPT_count_even_numbers_between_500_and_800_l2355_235585


namespace NUMINAMATH_GPT_degree_polynomial_is_13_l2355_235544

noncomputable def degree_polynomial (a b c d e f g h j : ℝ) : ℕ :=
  (7 + 4 + 2)

theorem degree_polynomial_is_13 (a b c d e f g h j : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) (he : e ≠ 0) (hf : f ≠ 0) (hg : g ≠ 0) (hh : h ≠ 0) (hj : j ≠ 0) : 
  degree_polynomial a b c d e f g h j = 13 :=
by
  rfl

end NUMINAMATH_GPT_degree_polynomial_is_13_l2355_235544


namespace NUMINAMATH_GPT_math_homework_pages_l2355_235534

-- Define Rachel's total pages, math homework pages, and reading homework pages
def total_pages : ℕ := 13
def reading_homework : ℕ := sorry
def math_homework (r : ℕ) : ℕ := r + 3

-- State the main theorem that needs to be proved
theorem math_homework_pages :
  ∃ r : ℕ, r + (math_homework r) = total_pages ∧ (math_homework r) = 8 :=
by {
  sorry
}

end NUMINAMATH_GPT_math_homework_pages_l2355_235534


namespace NUMINAMATH_GPT_original_price_l2355_235581

variable (P : ℝ)
variable (S : ℝ := 140)
variable (discount : ℝ := 0.60)

theorem original_price :
  (S = P * (1 - discount)) → (P = 350) :=
by
  sorry

end NUMINAMATH_GPT_original_price_l2355_235581


namespace NUMINAMATH_GPT_find_value_of_m_l2355_235520

open Real

theorem find_value_of_m (a b m : ℝ) (h1 : 2^a = m) (h2 : 5^b = m) (h3 : 1/a + 1/b = 2) : m = sqrt 10 := by
  sorry

end NUMINAMATH_GPT_find_value_of_m_l2355_235520


namespace NUMINAMATH_GPT_exists_integers_greater_than_N_l2355_235562

theorem exists_integers_greater_than_N (N : ℝ) : 
  ∃ (x1 x2 x3 x4 : ℤ), (x1 > N) ∧ (x2 > N) ∧ (x3 > N) ∧ (x4 > N) ∧ 
  (x1 * x1 + x2 * x2 + x3 * x3 + x4 * x4 = x1 * x2 * x3 + x1 * x2 * x4 + x1 * x3 * x4 + x2 * x3 * x4) := 
sorry

end NUMINAMATH_GPT_exists_integers_greater_than_N_l2355_235562


namespace NUMINAMATH_GPT_inequality_proof_l2355_235572

theorem inequality_proof (a b c : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) :
  a^3 * b + b^3 * c + c^3 * a ≥ a^2 * b * c + b^2 * c * a + c^2 * a * b :=
by {
  sorry
}

end NUMINAMATH_GPT_inequality_proof_l2355_235572


namespace NUMINAMATH_GPT_constant_term_binomial_expansion_l2355_235500

theorem constant_term_binomial_expansion :
  ∀ (x : ℝ), ((2 / x) + x) ^ 4 = 24 :=
by
  sorry

end NUMINAMATH_GPT_constant_term_binomial_expansion_l2355_235500


namespace NUMINAMATH_GPT_shop_width_l2355_235576

theorem shop_width 
  (monthly_rent : ℝ) 
  (shop_length : ℝ) 
  (annual_rent_per_sqft : ℝ) 
  (width : ℝ) 
  (monthly_rent_eq : monthly_rent = 2244) 
  (shop_length_eq : shop_length = 22) 
  (annual_rent_per_sqft_eq : annual_rent_per_sqft = 68) 
  (width_eq : width = 18) : 
  (12 * monthly_rent) / annual_rent_per_sqft / shop_length = width := 
by 
  sorry

end NUMINAMATH_GPT_shop_width_l2355_235576


namespace NUMINAMATH_GPT_total_length_XYZ_l2355_235587

theorem total_length_XYZ :
  let straight_segments := 7
  let slanted_segments := 7 * Real.sqrt 2
  straight_segments + slanted_segments = 7 + 7 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_total_length_XYZ_l2355_235587


namespace NUMINAMATH_GPT_angles_equal_l2355_235531

theorem angles_equal (A B C : ℝ) (h1 : A + B + C = Real.pi) (h2 : Real.sin A = 2 * Real.cos B * Real.sin C) : B = C :=
by sorry

end NUMINAMATH_GPT_angles_equal_l2355_235531


namespace NUMINAMATH_GPT_fragment_probability_l2355_235566

noncomputable def probability_fragment_in_21_digit_code : ℚ :=
  (12 * 10^11 - 30) / 10^21

theorem fragment_probability:
  ∀ (code : Fin 10 → Fin 21 → Fin 10),
  (∃ (i : Fin 12), ∀ (j : Fin 10), code (i + j) = j) → 
  probability_fragment_in_21_digit_code = (12 * 10^11 - 30) / 10^21 :=
sorry

end NUMINAMATH_GPT_fragment_probability_l2355_235566


namespace NUMINAMATH_GPT_totalNumberOfCrayons_l2355_235508

def numOrangeCrayons (numBoxes : ℕ) (crayonsPerBox : ℕ) : ℕ :=
  numBoxes * crayonsPerBox

def numBlueCrayons (numBoxes : ℕ) (crayonsPerBox : ℕ) : ℕ :=
  numBoxes * crayonsPerBox

def numRedCrayons (numBoxes : ℕ) (crayonsPerBox : ℕ) : ℕ :=
  numBoxes * crayonsPerBox

theorem totalNumberOfCrayons :
  numOrangeCrayons 6 8 + numBlueCrayons 7 5 + numRedCrayons 1 11 = 94 :=
by
  sorry

end NUMINAMATH_GPT_totalNumberOfCrayons_l2355_235508


namespace NUMINAMATH_GPT_yule_log_surface_area_increase_l2355_235564

theorem yule_log_surface_area_increase :
  let h := 10
  let d := 5
  let r := d / 2
  let n := 9
  let initial_surface_area := 2 * Real.pi * r * h + 2 * Real.pi * r^2
  let slice_height := h / n
  let slice_surface_area := 2 * Real.pi * r * slice_height + 2 * Real.pi * r^2
  let total_surface_area_slices := n * slice_surface_area
  let delta_surface_area := total_surface_area_slices - initial_surface_area
  delta_surface_area = 100 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_yule_log_surface_area_increase_l2355_235564


namespace NUMINAMATH_GPT_divisible_by_five_l2355_235529

theorem divisible_by_five {x y z : ℤ} (hxy : x ≠ y) (hyz : y ≠ z) (hzx : z ≠ x) : 
  5 ∣ ((x - y)^5 + (y - z)^5 + (z - x)^5) :=
sorry

end NUMINAMATH_GPT_divisible_by_five_l2355_235529


namespace NUMINAMATH_GPT_molecular_weight_BaCl2_l2355_235521

theorem molecular_weight_BaCl2 (mw8 : ℝ) (n : ℝ) (h : mw8 = 1656) : (mw8 / n = 207) ↔ n = 8 := 
by
  sorry

end NUMINAMATH_GPT_molecular_weight_BaCl2_l2355_235521


namespace NUMINAMATH_GPT_roots_equal_and_real_l2355_235526

theorem roots_equal_and_real (a c : ℝ) (h : 32 - 4 * a * c = 0) :
  ∃ x : ℝ, x = (2 * Real.sqrt 2) / a := 
by sorry

end NUMINAMATH_GPT_roots_equal_and_real_l2355_235526


namespace NUMINAMATH_GPT_managers_salary_l2355_235527

-- Definitions based on conditions
def avg_salary_50_employees : ℝ := 2000
def num_employees : ℕ := 50
def new_avg_salary : ℝ := 2150
def num_employees_with_manager : ℕ := 51

-- Condition statement: The manager's salary such that when added, average salary increases as given.
theorem managers_salary (M : ℝ) :
  (num_employees * avg_salary_50_employees + M) / num_employees_with_manager = new_avg_salary →
  M = 9650 := sorry

end NUMINAMATH_GPT_managers_salary_l2355_235527


namespace NUMINAMATH_GPT_quadratic_has_distinct_real_roots_l2355_235533

theorem quadratic_has_distinct_real_roots :
  ∃ (x y : ℝ), x ≠ y ∧ (x^2 - 3 * x - 1 = 0) ∧ (y^2 - 3 * y - 1 = 0) :=
by {
  sorry
}

end NUMINAMATH_GPT_quadratic_has_distinct_real_roots_l2355_235533


namespace NUMINAMATH_GPT_imo1965_cmo6511_l2355_235560

theorem imo1965_cmo6511 (x : ℝ) (h : 0 ≤ x ∧ x ≤ 2 * Real.pi) :
  2 * Real.cos x ≤ |(Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x)))| ∧
  |(Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x)))| ≤ Real.sqrt 2 ↔
  ((Real.pi / 4 ≤ x ∧ x ≤ Real.pi / 2) ∨ (3 * Real.pi / 2 ≤ x ∧ x ≤ 7 * Real.pi / 4)) :=
sorry

end NUMINAMATH_GPT_imo1965_cmo6511_l2355_235560


namespace NUMINAMATH_GPT_smallest_x_value_l2355_235512

theorem smallest_x_value :
  ∃ x, (x ≠ 9) ∧ (∀ y, (y ≠ 9) → ((x^2 - x - 72) / (x - 9) = 3 / (x + 6)) → x ≤ y) ∧ x = -9 :=
by
  sorry

end NUMINAMATH_GPT_smallest_x_value_l2355_235512


namespace NUMINAMATH_GPT_find_initial_mice_l2355_235574

theorem find_initial_mice : 
  ∃ x : ℕ, (∀ (h1 : ∀ (m : ℕ), m * 2 = m + m), (35 * x = 280) → x = 8) :=
by
  existsi 8
  intro h1 h2
  sorry

end NUMINAMATH_GPT_find_initial_mice_l2355_235574


namespace NUMINAMATH_GPT_must_be_nonzero_l2355_235532

noncomputable def Q (a b c d : ℝ) : ℝ → ℝ :=
  λ x => x^5 + a * x^4 + b * x^3 + c * x^2 + d * x

theorem must_be_nonzero (a b c d : ℝ)
  (h_roots : ∃ p q r s : ℝ, (∀ y : ℝ, Q a b c d y = 0 → y = 0 ∨ y = -1 ∨ y = p ∨ y = q ∨ y = r ∨ y = s) ∧ p ≠ 0 ∧ p ≠ -1 ∧ q ≠ 0 ∧ q ≠ -1 ∧ r ≠ 0 ∧ r ≠ -1 ∧ s ≠ 0 ∧ s ≠ -1)
  (h_distinct : (∀ x₁ x₂ : ℝ, Q a b c d x₁ = 0 ∧ Q a b c d x₂ = 0 → x₁ ≠ x₂ ∨ x₁ = x₂) → False)
  (h_f_zero : Q a b c d 0 = 0) :
  d ≠ 0 := by
  sorry

end NUMINAMATH_GPT_must_be_nonzero_l2355_235532


namespace NUMINAMATH_GPT_total_packs_l2355_235552

noncomputable def robyn_packs : ℕ := 16
noncomputable def lucy_packs : ℕ := 19

theorem total_packs : robyn_packs + lucy_packs = 35 := by
  sorry

end NUMINAMATH_GPT_total_packs_l2355_235552


namespace NUMINAMATH_GPT_Alex_final_silver_tokens_l2355_235519

variable (x y : ℕ)

def final_red_tokens (x y : ℕ) : ℕ := 90 - 3 * x + 2 * y
def final_blue_tokens (x y : ℕ) : ℕ := 65 + 2 * x - 4 * y
def silver_tokens (x y : ℕ) : ℕ := x + y

theorem Alex_final_silver_tokens (h1 : final_red_tokens x y < 3)
                                 (h2 : final_blue_tokens x y < 4) :
  silver_tokens x y = 67 := 
sorry

end NUMINAMATH_GPT_Alex_final_silver_tokens_l2355_235519


namespace NUMINAMATH_GPT_choco_delight_remainder_l2355_235510

theorem choco_delight_remainder (m : ℕ) (h : m % 7 = 5) : (4 * m) % 7 = 6 := 
by 
  sorry

end NUMINAMATH_GPT_choco_delight_remainder_l2355_235510


namespace NUMINAMATH_GPT_vector_magnitude_difference_l2355_235547

-- Defining the vectors a and b
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-2, 4)

-- Statement to prove that the magnitude of the difference of vectors a and b is 5
theorem vector_magnitude_difference : ‖a - b‖ = 5 := 
sorry -- Proof omitted

end NUMINAMATH_GPT_vector_magnitude_difference_l2355_235547


namespace NUMINAMATH_GPT_distance_PF_l2355_235589

-- Definitions for the given conditions
structure Rectangle :=
  (EF GH: ℝ)
  (interior_point : ℝ × ℝ)
  (PE : ℝ)
  (PH : ℝ)
  (PG : ℝ)

-- The theorem to prove PF equals 12 under the given conditions
theorem distance_PF 
  (r : Rectangle)
  (hPE : r.PE = 5)
  (hPH : r.PH = 12)
  (hPG : r.PG = 13) :
  ∃ PF, PF = 12 := 
sorry

end NUMINAMATH_GPT_distance_PF_l2355_235589


namespace NUMINAMATH_GPT_largest_3_digit_sum_l2355_235546

theorem largest_3_digit_sum : ∃ A B : ℕ, A ≠ B ∧ A < 10 ∧ B < 10 ∧ 100 ≤ 111 * A + 12 * B ∧ 111 * A + 12 * B = 996 := by
  sorry

end NUMINAMATH_GPT_largest_3_digit_sum_l2355_235546


namespace NUMINAMATH_GPT_cupboard_cost_price_l2355_235545

theorem cupboard_cost_price
  (C : ℝ)
  (h1 : ∀ (S : ℝ), S = 0.84 * C) -- Vijay sells a cupboard at 84% of the cost price.
  (h2 : ∀ (S_new : ℝ), S_new = 1.16 * C) -- If Vijay got Rs. 1200 more, he would have made a profit of 16%.
  (h3 : ∀ (S_new S : ℝ), S_new - S = 1200) -- The difference between new selling price and original selling price is Rs. 1200.
  : C = 3750 := 
sorry -- Proof is not required.

end NUMINAMATH_GPT_cupboard_cost_price_l2355_235545


namespace NUMINAMATH_GPT_Lin_finishes_reading_on_Monday_l2355_235591

theorem Lin_finishes_reading_on_Monday :
  let start_day := "Tuesday"
  let book_days : ℕ → ℕ := fun n => n
  let total_books := 10
  let total_days := (total_books * (total_books + 1)) / 2
  let days_in_a_week := 7
  let finish_day_offset := total_days % days_in_a_week
  let day_names := ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
  (day_names.indexOf start_day + finish_day_offset) % days_in_a_week = day_names.indexOf "Monday" :=
by
  sorry

end NUMINAMATH_GPT_Lin_finishes_reading_on_Monday_l2355_235591


namespace NUMINAMATH_GPT_solve_inequality_l2355_235573

open Set Real

noncomputable def inequality_solution_set : Set ℝ :=
  {x | -2 ≤ x ∧ x ≤ 2} ∪ {6}

theorem solve_inequality (x : ℝ) :
  (x ^ 2 - 4) * (x - 6) ^ 2 ≤ 0 ↔ x ∈ inequality_solution_set := 
sorry

end NUMINAMATH_GPT_solve_inequality_l2355_235573


namespace NUMINAMATH_GPT_domain_tan_x_plus_pi_over_3_l2355_235567

open Real Set

theorem domain_tan_x_plus_pi_over_3 :
  ∀ x : ℝ, ¬ (∃ k : ℤ, x = k * π + π / 6) ↔ x ∈ {x : ℝ | ¬ ∃ k : ℤ, x = k * π + π / 6} :=
by {
  sorry
}

end NUMINAMATH_GPT_domain_tan_x_plus_pi_over_3_l2355_235567


namespace NUMINAMATH_GPT_number_of_cells_after_9_days_l2355_235516

theorem number_of_cells_after_9_days : 
  let initial_cells := 4 
  let doubling_period := 3 
  let total_duration := 9 
  ∀ cells_after_9_days, cells_after_9_days = initial_cells * 2^(total_duration / doubling_period) 
  → cells_after_9_days = 32 :=
by
  sorry

end NUMINAMATH_GPT_number_of_cells_after_9_days_l2355_235516


namespace NUMINAMATH_GPT_sum_of_solutions_l2355_235549

theorem sum_of_solutions (x y : ℝ) (h1 : y = 9) (h2 : x^2 + y^2 = 225) : 2 * x = 0 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_solutions_l2355_235549


namespace NUMINAMATH_GPT_avg_consecutive_integers_l2355_235550

theorem avg_consecutive_integers (a : ℝ) (b : ℝ) 
  (h₁ : b = (a + (a + 1) + (a + 2) + (a + 3) + (a + 4) + (a + 5)) / 6) :
  (a + 5) = (b + (b + 1) + (b + 2) + (b + 3) + (b + 4) + (b + 5)) / 6 :=
by sorry

end NUMINAMATH_GPT_avg_consecutive_integers_l2355_235550


namespace NUMINAMATH_GPT_complete_the_square_l2355_235542

theorem complete_the_square (x : ℝ) : 
    (x^2 - 2 * x - 5 = 0) -> (x - 1)^2 = 6 :=
by sorry

end NUMINAMATH_GPT_complete_the_square_l2355_235542


namespace NUMINAMATH_GPT_max_n_positive_l2355_235575

theorem max_n_positive (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h1 : S 15 > 0)
  (h2 : S 16 < 0)
  (hs1 : S 15 = 15 * (a 8))
  (hs2 : S 16 = 8 * (a 8 + a 9)) :
  (∀ n, a n > 0 → n ≤ 8) :=
by {
    sorry
}

end NUMINAMATH_GPT_max_n_positive_l2355_235575


namespace NUMINAMATH_GPT_prime_sum_l2355_235588

theorem prime_sum (m n : ℕ) (hm : Prime m) (hn : Prime n) (h : 5 * m + 7 * n = 129) :
  m + n = 19 ∨ m + n = 25 := by
  sorry

end NUMINAMATH_GPT_prime_sum_l2355_235588


namespace NUMINAMATH_GPT_trig_identity_l2355_235501

theorem trig_identity (A : ℝ) (h : Real.cos (π + A) = -1/2) : Real.sin (π / 2 + A) = 1/2 :=
by 
sorry

end NUMINAMATH_GPT_trig_identity_l2355_235501


namespace NUMINAMATH_GPT_min_amount_for_free_shipping_l2355_235536

def book1 : ℝ := 13.00
def book2 : ℝ := 15.00
def book3 : ℝ := 10.00
def book4 : ℝ := 10.00
def discount_rate : ℝ := 0.25
def shipping_threshold : ℝ := 9.00

def total_cost_before_discount : ℝ := book1 + book2 + book3 + book4
def discount_amount : ℝ := book1 * discount_rate + book2 * discount_rate
def total_cost_after_discount : ℝ := total_cost_before_discount - discount_amount

theorem min_amount_for_free_shipping : total_cost_after_discount + shipping_threshold = 50.00 :=
by
  sorry

end NUMINAMATH_GPT_min_amount_for_free_shipping_l2355_235536


namespace NUMINAMATH_GPT_area_outside_two_small_squares_l2355_235580

theorem area_outside_two_small_squares (L S : ℝ) (hL : L = 9) (hS : S = 4) :
  let large_square_area := L^2
  let small_square_area := S^2
  let combined_small_squares_area := 2 * small_square_area
  large_square_area - combined_small_squares_area = 49 :=
by
  sorry

end NUMINAMATH_GPT_area_outside_two_small_squares_l2355_235580


namespace NUMINAMATH_GPT_more_pups_than_adult_dogs_l2355_235535

def number_of_huskies := 5
def number_of_pitbulls := 2
def number_of_golden_retrievers := 4
def pups_per_husky := 3
def pups_per_pitbull := 3
def additional_pups_per_golden_retriever := 2
def pups_per_golden_retriever := pups_per_husky + additional_pups_per_golden_retriever

def total_pups := (number_of_huskies * pups_per_husky) + (number_of_pitbulls * pups_per_pitbull) + (number_of_golden_retrievers * pups_per_golden_retriever)
def total_adult_dogs := number_of_huskies + number_of_pitbulls + number_of_golden_retrievers

theorem more_pups_than_adult_dogs : (total_pups - total_adult_dogs) = 30 :=
by
  -- proof steps, which we will skip
  sorry

end NUMINAMATH_GPT_more_pups_than_adult_dogs_l2355_235535


namespace NUMINAMATH_GPT_find_f500_l2355_235559

variable (f : ℕ → ℕ)
variable (h : ∀ x y : ℕ, f (x * y) = f x + f y)
variable (h₁ : f 10 = 16)
variable (h₂ : f 40 = 24)

theorem find_f500 : f 500 = 44 :=
sorry

end NUMINAMATH_GPT_find_f500_l2355_235559


namespace NUMINAMATH_GPT_triangle_proof_l2355_235553

-- Declare a structure for a triangle with given conditions
structure TriangleABC :=
  (a b c : ℝ) -- sides opposite to angles A, B, and C
  (A B C : ℝ) -- angles A, B, and C
  (R : ℝ) -- circumcircle radius
  (r : ℝ := 3) -- inradius is given as 3
  (area : ℝ := 6) -- area of the triangle is 6
  (h1 : a * Real.cos A + b * Real.cos B + c * Real.cos C = R / 3) -- given condition
  (h2 : ∀ a b c A B C, a * Real.sin A + b * Real.sin B + c * Real.sin C = 2 * area / (a+b+c)) -- implied area condition

-- Define the theorem using the above conditions
theorem triangle_proof (t : TriangleABC) :
  t.a + t.b + t.c = 4 ∧
  (Real.sin (2 * t.A) + Real.sin (2 * t.B) + Real.sin (2 * t.C)) = 1/3 ∧
  t.R = 6 :=
by
  sorry

end NUMINAMATH_GPT_triangle_proof_l2355_235553


namespace NUMINAMATH_GPT_initial_books_in_library_l2355_235593

theorem initial_books_in_library
  (books_out_tuesday : ℕ)
  (books_in_thursday : ℕ)
  (books_out_friday : ℕ)
  (final_books : ℕ)
  (h1 : books_out_tuesday = 227)
  (h2 : books_in_thursday = 56)
  (h3 : books_out_friday = 35)
  (h4 : final_books = 29) : 
  initial_books = 235 :=
by
  sorry

end NUMINAMATH_GPT_initial_books_in_library_l2355_235593


namespace NUMINAMATH_GPT_union_A_B_intersection_complement_A_B_l2355_235596

open Set Real

noncomputable def A : Set ℝ := {x : ℝ | x^2 - x - 2 ≥ 0}
noncomputable def B : Set ℝ := {x : ℝ | abs (2 * x + 1) ≤ 1}

theorem union_A_B : A ∪ B = {x : ℝ | x ≤ 0 ∨ x ≥ 2} := by
  sorry

theorem intersection_complement_A_B : (Aᶜ) ∩ (Bᶜ) = {x : ℝ | 0 < x ∧ x < 2} := by
  sorry

end NUMINAMATH_GPT_union_A_B_intersection_complement_A_B_l2355_235596


namespace NUMINAMATH_GPT_cobbler_hours_per_day_l2355_235523

-- Defining some conditions based on our problem statement
def cobbler_rate : ℕ := 3  -- pairs of shoes per hour
def friday_hours : ℕ := 3  -- number of hours worked on Friday
def friday_pairs : ℕ := cobbler_rate * friday_hours  -- pairs mended on Friday
def weekly_pairs : ℕ := 105  -- total pairs mended in a week
def mon_thu_pairs : ℕ := weekly_pairs - friday_pairs  -- pairs mended from Monday to Thursday
def mon_thu_hours : ℕ := mon_thu_pairs / cobbler_rate  -- total hours worked from Monday to Thursday

-- Thm statement: If a cobbler works h hours daily from Mon to Thu, then h = 8 implies total = 105 pairs
theorem cobbler_hours_per_day (h : ℕ) : (4 * h = mon_thu_hours) ↔ (h = 8) :=
by
  sorry

end NUMINAMATH_GPT_cobbler_hours_per_day_l2355_235523


namespace NUMINAMATH_GPT_linear_regression_eq_l2355_235586

noncomputable def x_vals : List ℝ := [3, 7, 11]
noncomputable def y_vals : List ℝ := [10, 20, 24]

theorem linear_regression_eq :
  ∃ a b : ℝ, (a = 5.75) ∧ (b = 1.75) ∧ (∀ x, ∃ y, y = a + b * x) := sorry

end NUMINAMATH_GPT_linear_regression_eq_l2355_235586


namespace NUMINAMATH_GPT_inequality_proof_l2355_235539

theorem inequality_proof (a b c d e f : ℝ) (h : b^2 ≥ a^2 + c^2) : 
  (a * f - c * d)^2 ≤ (a * e - b * d)^2 + (b * f - c * e)^2 := 
by 
  sorry

end NUMINAMATH_GPT_inequality_proof_l2355_235539


namespace NUMINAMATH_GPT_find_f_3_l2355_235507

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f_3 : 
  (∀ (x : ℝ), x ≠ 0 → 27 * f (-x) / x - x^2 * f (1 / x) = - 2 * x^2) →
  f 3 = 2 :=
sorry

end NUMINAMATH_GPT_find_f_3_l2355_235507


namespace NUMINAMATH_GPT_simplify_expression_evaluate_l2355_235582

theorem simplify_expression_evaluate : 
  let x := 1
  let y := 2
  (2 * x - y) * (y + 2 * x) - (2 * y + x) * (2 * y - x) = -15 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_evaluate_l2355_235582


namespace NUMINAMATH_GPT_total_area_of_removed_triangles_l2355_235558

theorem total_area_of_removed_triangles (a b : ℝ)
  (square_side : ℝ := 16)
  (triangle_hypotenuse : ℝ := 8)
  (isosceles_right_triangle : a = b ∧ a^2 + b^2 = triangle_hypotenuse^2) :
  4 * (1 / 2 * a * b) = 64 :=
by
  -- Sketch of the proof:
  -- From the isosceles right triangle property and Pythagorean theorem,
  -- a^2 + b^2 = 8^2 ⇒ 2 * a^2 = 64 ⇒ a^2 = 32 ⇒ a = b = 4√2
  -- The area of one triangle is (1/2) * a * b = 16
  -- Total area of four such triangles is 4 * 16 = 64
  sorry

end NUMINAMATH_GPT_total_area_of_removed_triangles_l2355_235558


namespace NUMINAMATH_GPT_quotient_with_zero_in_middle_l2355_235511

theorem quotient_with_zero_in_middle : 
  ∃ (op : ℕ → ℕ → ℕ), 
  (op = Nat.add ∧ ((op 6 4) / 3).digits 10 = [3, 0, 3]) := 
by 
  sorry

end NUMINAMATH_GPT_quotient_with_zero_in_middle_l2355_235511


namespace NUMINAMATH_GPT_unique_four_digit_square_l2355_235554

theorem unique_four_digit_square (n : ℕ) : 
  1000 ≤ n ∧ n < 10000 ∧ 
  (n % 10 = (n / 10) % 10) ∧ 
  ((n / 100) % 10 = (n / 1000) % 10) ∧ 
  (∃ k : ℕ, n = k^2) ↔ n = 7744 := 
by 
  sorry

end NUMINAMATH_GPT_unique_four_digit_square_l2355_235554


namespace NUMINAMATH_GPT_part_a_l2355_235530

theorem part_a {d m b : ℕ} (h_d : d = 41) (h_m : m = 28) (h_b : b = 15) :
    d - b + m - b + b = 54 :=
  by sorry

end NUMINAMATH_GPT_part_a_l2355_235530


namespace NUMINAMATH_GPT_max_difference_primes_l2355_235592

def is_prime (n : ℕ) := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def even_integer : ℕ := 138

theorem max_difference_primes (p q : ℕ) :
  is_prime p ∧ is_prime q ∧ p + q = even_integer ∧ p ≠ q →
  (q - p) = 124 :=
by
  sorry

end NUMINAMATH_GPT_max_difference_primes_l2355_235592


namespace NUMINAMATH_GPT_loom_weaving_rate_l2355_235537

noncomputable def total_cloth : ℝ := 27
noncomputable def total_time : ℝ := 210.9375

theorem loom_weaving_rate :
  (total_cloth / total_time) = 0.128 :=
by
  sorry

end NUMINAMATH_GPT_loom_weaving_rate_l2355_235537


namespace NUMINAMATH_GPT_evaluate_expression_l2355_235543

theorem evaluate_expression :
  (3 ^ 4 * 5 ^ 2 * 7 ^ 3 * 11) / (7 * 11 ^ 2) = 9025 :=
by 
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2355_235543


namespace NUMINAMATH_GPT_distance_from_hyperbola_focus_to_line_l2355_235598

-- Definitions of the problem conditions
def hyperbola : Prop := ∀ x y : ℝ, (x^2 / 4 - y^2 / 5 = 1)
def line : Prop := ∀ x y : ℝ, (x + 2 * y - 8 = 0)

-- The main theorem we wish to prove
theorem distance_from_hyperbola_focus_to_line : 
  (∀ x y : ℝ, hyperbola) ∧ (∀ x y : ℝ, line) → ∃ d : ℝ, d = Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_distance_from_hyperbola_focus_to_line_l2355_235598


namespace NUMINAMATH_GPT_math_proof_problem_l2355_235563

variables {a b c d e f k : ℝ}

theorem math_proof_problem 
  (h1 : a + b + c = d + e + f)
  (h2 : a^2 + b^2 + c^2 = d^2 + e^2 + f^2)
  (h3 : a^3 + b^3 + c^3 ≠ d^3 + e^3 + f^3) :
  (a + b + c + (d + k) + (e + k) + (f + k) = d + e + f + (a + k) + (b + k) + (c + k) ∧
   a^2 + b^2 + c^2 + (d + k)^2 + (e + k)^2 + (f + k)^2 = d^2 + e^2 + f^2 + (a + k)^2 + (b + k)^2 + (c + k)^2 ∧
   a^3 + b^3 + c^3 + (d + k)^3 + (e + k)^3 + (f + k)^3 = d^3 + e^3 + f^3 + (a + k)^3 + (b + k)^3 + (c + k)^3) 
   ∧ 
  (a^4 + b^4 + c^4 + (d + k)^4 + (e + k)^4 + (f + k)^4 ≠ d^4 + e^4 + f^4 + (a + k)^4 + (b + k)^4 + (c + k)^4) := 
  sorry

end NUMINAMATH_GPT_math_proof_problem_l2355_235563


namespace NUMINAMATH_GPT_calculate_negative_subtraction_l2355_235595

theorem calculate_negative_subtraction : -2 - (-3) = 1 :=
by sorry

end NUMINAMATH_GPT_calculate_negative_subtraction_l2355_235595


namespace NUMINAMATH_GPT_find_ABC_l2355_235570

noncomputable def g (x : ℝ) (A B C : ℝ) : ℝ := x^2 / (A * x^2 + B * x + C)

theorem find_ABC : 
  ∀ (A B C : ℝ),
  (∀ (x : ℝ), x > 2 → g x A B C > 0.3) →
  (∃ (A : ℤ), A = 4) →
  (∃ (B : ℤ), ∃ (C : ℤ), A = 4 ∧ B = 8 ∧ C = -12) →
  A + B + C = 0 :=
by
  intros A B C h1 h2 h3
  rcases h2 with ⟨intA, h2'⟩
  rcases h3 with ⟨intB, ⟨intC, h3'⟩⟩
  simp [h2', h3']
  sorry -- proof skipped

end NUMINAMATH_GPT_find_ABC_l2355_235570


namespace NUMINAMATH_GPT_sum_of_fraction_parts_l2355_235518

theorem sum_of_fraction_parts (x : ℝ) (hx : x = 0.45) : 
  (∃ (a b : ℕ), x = a / b ∧ Nat.gcd a b = 1 ∧ a + b = 16) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_fraction_parts_l2355_235518


namespace NUMINAMATH_GPT_average_temp_tues_to_fri_l2355_235538

theorem average_temp_tues_to_fri (T W Th : ℕ) 
  (h1: (42 + T + W + Th) / 4 = 48) 
  (mon: 42 = 42) 
  (fri: 10 = 10) :
  (T + W + Th + 10) / 4 = 40 := by
  sorry

end NUMINAMATH_GPT_average_temp_tues_to_fri_l2355_235538


namespace NUMINAMATH_GPT_competition_end_time_is_5_35_am_l2355_235548

def start_time : Nat := 15 * 60  -- 3:00 p.m. in minutes
def duration : Nat := 875  -- competition duration in minutes
def end_time : Nat := (start_time + duration) % (24 * 60)  -- competition end time in minutes

theorem competition_end_time_is_5_35_am :
  end_time = 5 * 60 + 35 :=  -- 5:35 a.m. in minutes
sorry

end NUMINAMATH_GPT_competition_end_time_is_5_35_am_l2355_235548


namespace NUMINAMATH_GPT_ratio_paperback_fiction_to_nonfiction_l2355_235506

-- Definitions
def total_books := 160
def hardcover_nonfiction := 25
def paperback_nonfiction := hardcover_nonfiction + 20
def paperback_fiction := total_books - hardcover_nonfiction - paperback_nonfiction

-- Theorem statement
theorem ratio_paperback_fiction_to_nonfiction : paperback_fiction / paperback_nonfiction = 2 :=
by
  -- proof details would go here
  sorry

end NUMINAMATH_GPT_ratio_paperback_fiction_to_nonfiction_l2355_235506


namespace NUMINAMATH_GPT_students_not_taken_test_l2355_235524

theorem students_not_taken_test 
  (num_enrolled : ℕ) 
  (answered_q1 : ℕ) 
  (answered_q2 : ℕ) 
  (answered_both : ℕ) 
  (H_num_enrolled : num_enrolled = 40) 
  (H_answered_q1 : answered_q1 = 30) 
  (H_answered_q2 : answered_q2 = 29) 
  (H_answered_both : answered_both = 29) : 
  num_enrolled - (answered_q1 + answered_q2 - answered_both) = 10 :=
by {
  sorry
}

end NUMINAMATH_GPT_students_not_taken_test_l2355_235524


namespace NUMINAMATH_GPT_length_of_ae_l2355_235569

-- Define the given consecutive points
variables (a b c d e : ℝ)

-- Conditions from the problem
-- 1. Points a, b, c, d, e are 5 consecutive points on a straight line - implicitly assumed on the same line
-- 2. bc = 2 * cd
-- 3. de = 4
-- 4. ab = 5
-- 5. ac = 11

theorem length_of_ae 
  (h1 : b - a = 5) -- ab = 5
  (h2 : c - a = 11) -- ac = 11
  (h3 : c - b = 2 * (d - c)) -- bc = 2 * cd
  (h4 : e - d = 4) -- de = 4
  : (e - a) = 18 := sorry

end NUMINAMATH_GPT_length_of_ae_l2355_235569


namespace NUMINAMATH_GPT_find_g_l2355_235514

noncomputable def g (x : ℝ) : ℝ := sorry

axiom functional_equation : ∀ x y : ℝ, (g x * g y - g (x * y)) / 4 = x + y + 4

theorem find_g :
  (∀ x : ℝ, g x = x + 5) ∨ (∀ x : ℝ, g x = -x - 3) := 
by
  sorry

end NUMINAMATH_GPT_find_g_l2355_235514


namespace NUMINAMATH_GPT_constant_function_odd_iff_zero_l2355_235594

theorem constant_function_odd_iff_zero (k : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = k) 
  (h2 : ∀ x, f (-x) = -f x) : 
  k = 0 :=
sorry

end NUMINAMATH_GPT_constant_function_odd_iff_zero_l2355_235594
