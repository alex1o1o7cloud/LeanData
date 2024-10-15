import Mathlib

namespace NUMINAMATH_GPT_bert_money_left_l1245_124561

theorem bert_money_left (initial_money : ℕ) (spent_hardware : ℕ) (spent_cleaners : ℕ) (spent_grocery : ℕ) :
  initial_money = 52 →
  spent_hardware = initial_money * 1 / 4 →
  spent_cleaners = 9 →
  spent_grocery = (initial_money - spent_hardware - spent_cleaners) / 2 →
  initial_money - spent_hardware - spent_cleaners - spent_grocery = 15 := 
by
  intros h_initial h_hardware h_cleaners h_grocery
  rw [h_initial, h_hardware, h_cleaners, h_grocery]
  sorry

end NUMINAMATH_GPT_bert_money_left_l1245_124561


namespace NUMINAMATH_GPT_margie_drive_distance_l1245_124573

-- Conditions
def car_mpg : ℝ := 45  -- miles per gallon
def gas_price : ℝ := 5 -- dollars per gallon
def money_spent : ℝ := 25 -- dollars

-- Question: Prove that Margie can drive 225 miles with $25 worth of gas.
theorem margie_drive_distance (h1 : car_mpg = 45) (h2 : gas_price = 5) (h3 : money_spent = 25) :
  money_spent / gas_price * car_mpg = 225 := by
  sorry

end NUMINAMATH_GPT_margie_drive_distance_l1245_124573


namespace NUMINAMATH_GPT_products_arrangement_count_l1245_124544

/--
There are five different products: A, B, C, D, and E arranged in a row on a shelf.
- Products A and B must be adjacent.
- Products C and D must not be adjacent.
Prove that there are a total of 24 distinct valid arrangements under these conditions.
-/
theorem products_arrangement_count : 
  ∃ (n : ℕ), 
  (∀ (A B C D E : Type), n = 24 ∧
  ∀ l : List (Type), l = [A, B, C, D, E] ∧
  -- A and B must be adjacent
  (∀ p : List (Type), p = [A, B] ∨ p = [B, A]) ∧
  -- C and D must not be adjacent
  ¬ (∀ q : List (Type), q = [C, D] ∨ q = [D, C])) :=
sorry

end NUMINAMATH_GPT_products_arrangement_count_l1245_124544


namespace NUMINAMATH_GPT_set_C_cannot_form_triangle_l1245_124599

def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Given conditions
def set_A := (3, 6, 8)
def set_B := (3, 8, 9)
def set_C := (3, 6, 9)
def set_D := (6, 8, 9)

theorem set_C_cannot_form_triangle : ¬ is_triangle 3 6 9 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_set_C_cannot_form_triangle_l1245_124599


namespace NUMINAMATH_GPT_determine_guilty_defendant_l1245_124550

-- Define the defendants
inductive Defendant
| A
| B
| C

open Defendant

-- Define the guilty defendant
def guilty_defendant : Defendant := C

-- Define the conditions
def condition1 (d : Defendant) : Prop :=
d ≠ A ∧ d ≠ B ∧ d ≠ C → false  -- "There were three defendants, and only one of them was guilty."

def condition2 (d : Defendant) : Prop :=
d = A → d ≠ B  -- "Defendant A accused defendant B."

def condition3 (d : Defendant) : Prop :=
d = B → d = B  -- "Defendant B admitted to being guilty."

def condition4 (d : Defendant) : Prop :=
d = C → (d = C ∨ d = A)  -- "Defendant C either admitted to being guilty or accused A."

-- The proof problem statement
theorem determine_guilty_defendant :
  (∃ d : Defendant, condition1 d ∧ condition2 d ∧ condition3 d ∧ condition4 d) → guilty_defendant = C :=
by {
  sorry
}

end NUMINAMATH_GPT_determine_guilty_defendant_l1245_124550


namespace NUMINAMATH_GPT_carol_first_to_roll_six_l1245_124517

def probability_roll (x : ℕ) (success : ℕ) : ℚ := success / x

def first_to_roll_six_probability (a b c : ℕ) : ℚ :=
  let p_six : ℚ := probability_roll 6 1
  let p_not_six : ℚ := 1 - p_six
  let cycle_prob : ℚ := p_not_six * p_not_six * p_six
  let continue_prob : ℚ := p_not_six * p_not_six * p_not_six
  let geometric_sum : ℚ := cycle_prob / (1 - continue_prob)
  geometric_sum

theorem carol_first_to_roll_six :
  first_to_roll_six_probability 1 1 1 = 25 / 91 := 
sorry

end NUMINAMATH_GPT_carol_first_to_roll_six_l1245_124517


namespace NUMINAMATH_GPT_distance_to_x_axis_P_l1245_124551

-- The coordinates of point P
def P : ℝ × ℝ := (3, -2)

-- The distance from point P to the x-axis
def distance_to_x_axis (point : ℝ × ℝ) : ℝ :=
  abs (point.snd)

theorem distance_to_x_axis_P : distance_to_x_axis P = 2 :=
by
  -- Use the provided point P and calculate the distance
  sorry

end NUMINAMATH_GPT_distance_to_x_axis_P_l1245_124551


namespace NUMINAMATH_GPT_mom_buys_tshirts_l1245_124578

theorem mom_buys_tshirts 
  (tshirts_per_package : ℕ := 3) 
  (num_packages : ℕ := 17) :
  tshirts_per_package * num_packages = 51 :=
by
  sorry

end NUMINAMATH_GPT_mom_buys_tshirts_l1245_124578


namespace NUMINAMATH_GPT_find_n_value_l1245_124579

theorem find_n_value (n : ℤ) : (5^3 - 7 = 6^2 + n) ↔ (n = 82) :=
by
  sorry

end NUMINAMATH_GPT_find_n_value_l1245_124579


namespace NUMINAMATH_GPT_num_passengers_on_second_plane_l1245_124521

theorem num_passengers_on_second_plane :
  ∃ x : ℕ, 600 - (2 * 50) + 600 - (2 * x) + 600 - (2 * 40) = 1500 →
  x = 60 :=
by
  sorry

end NUMINAMATH_GPT_num_passengers_on_second_plane_l1245_124521


namespace NUMINAMATH_GPT_log_eq_solution_l1245_124563

theorem log_eq_solution (x : ℝ) (h : Real.log 8 / Real.log x = Real.log 5 / Real.log 125) : x = 512 := by
  sorry

end NUMINAMATH_GPT_log_eq_solution_l1245_124563


namespace NUMINAMATH_GPT_national_currency_depreciation_bond_annual_coupon_income_dividend_yield_tax_deduction_l1245_124589

-- Question 5
theorem national_currency_depreciation (term : String) : term = "Devaluation" := 
sorry

-- Question 6
theorem bond_annual_coupon_income 
  (purchase_price face_value annual_yield annual_coupon : ℝ) 
  (h_price : purchase_price = 900)
  (h_face : face_value = 1000)
  (h_yield : annual_yield = 0.15) 
  (h_coupon : annual_coupon = 135) : 
  annual_coupon = annual_yield * purchase_price := 
sorry

-- Question 7
theorem dividend_yield 
  (num_shares price_per_share total_dividends dividend_yield : ℝ)
  (h_shares : num_shares = 1000000)
  (h_price : price_per_share = 400)
  (h_dividends : total_dividends = 60000000)
  (h_yield : dividend_yield = 15) : 
  dividend_yield = (total_dividends / num_shares / price_per_share) * 100 :=
sorry

-- Question 8
theorem tax_deduction 
  (insurance_premium annual_salary tax_return : ℝ)
  (h_premium : insurance_premium = 120000)
  (h_salary : annual_salary = 110000)
  (h_return : tax_return = 14300) : 
  tax_return = 0.13 * min insurance_premium annual_salary := 
sorry

end NUMINAMATH_GPT_national_currency_depreciation_bond_annual_coupon_income_dividend_yield_tax_deduction_l1245_124589


namespace NUMINAMATH_GPT_basketball_players_l1245_124584

theorem basketball_players {total : ℕ} (total_boys : total = 22) 
                           (football_boys : ℕ) (football_boys_count : football_boys = 15) 
                           (neither_boys : ℕ) (neither_boys_count : neither_boys = 3) 
                           (both_boys : ℕ) (both_boys_count : both_boys = 18) : 
                           (total - neither_boys = 19) := 
by
  sorry

end NUMINAMATH_GPT_basketball_players_l1245_124584


namespace NUMINAMATH_GPT_largest_exterior_angle_l1245_124532

theorem largest_exterior_angle (x : ℝ) (h1 : 3 * x + 4 * x + 5 * x = 180) : 180 - 3 * (180 / 12) = 135 :=
by {
  -- Sorry is a placeholder for the actual proof
  sorry
}

end NUMINAMATH_GPT_largest_exterior_angle_l1245_124532


namespace NUMINAMATH_GPT_factorize_expr_l1245_124588

theorem factorize_expr (x : ℝ) : x^3 - 4 * x = x * (x + 2) * (x - 2) :=
  sorry

end NUMINAMATH_GPT_factorize_expr_l1245_124588


namespace NUMINAMATH_GPT_geometric_sequence_ratio_l1245_124557

theorem geometric_sequence_ratio (a1 : ℕ) (S : ℕ → ℕ) (q : ℕ) (h1 : q = 2)
  (h2 : ∀ n, S n = a1 * (1 - q ^ (n + 1)) / (1 - q)) :
  S 4 / S 2 = 5 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_ratio_l1245_124557


namespace NUMINAMATH_GPT_students_passed_both_tests_l1245_124575

theorem students_passed_both_tests
  (n : ℕ) (A : ℕ) (B : ℕ) (C : ℕ)
  (h1 : n = 100) 
  (h2 : A = 60) 
  (h3 : B = 40) 
  (h4 : C = 20) :
  A + B - ((n - C) - (A + B - n)) = 20 :=
by
  sorry

end NUMINAMATH_GPT_students_passed_both_tests_l1245_124575


namespace NUMINAMATH_GPT_find_a_plus_b_minus_c_l1245_124598

theorem find_a_plus_b_minus_c (a b c : ℤ) (h1 : 3 * b = 5 * a) (h2 : 7 * a = 3 * c) (h3 : 3 * a + 2 * b - 4 * c = -9) : a + b - c = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_a_plus_b_minus_c_l1245_124598


namespace NUMINAMATH_GPT_find_a_l1245_124535

variable {x y a : ℝ}

theorem find_a (h1 : 2 * x - y + a ≥ 0) (h2 : 3 * x + y ≤ 3) (h3 : ∀ (x y : ℝ), 4 * x + 3 * y ≤ 8) : a = 2 := 
sorry

end NUMINAMATH_GPT_find_a_l1245_124535


namespace NUMINAMATH_GPT_find_num_male_general_attendees_l1245_124507

def num_attendees := 1000
def num_presenters := 420
def total_general_attendees := num_attendees - num_presenters

variables (M_p F_p M_g F_g : ℕ)

axiom condition1 : M_p = F_p + 20
axiom condition2 : M_p + F_p = 420
axiom condition3 : F_g = M_g + 56
axiom condition4 : M_g + F_g = total_general_attendees

theorem find_num_male_general_attendees :
  M_g = 262 :=
by
  sorry

end NUMINAMATH_GPT_find_num_male_general_attendees_l1245_124507


namespace NUMINAMATH_GPT_mechanical_pencils_fraction_l1245_124592

theorem mechanical_pencils_fraction (total_pencils : ℕ) (frac_mechanical : ℚ)
    (mechanical_pencils : ℕ) (standard_pencils : ℕ) (new_total_pencils : ℕ) 
    (new_standard_pencils : ℕ) (new_frac_mechanical : ℚ):
  total_pencils = 120 →
  frac_mechanical = 1 / 4 →
  mechanical_pencils = frac_mechanical * total_pencils →
  standard_pencils = total_pencils - mechanical_pencils →
  new_standard_pencils = 3 * standard_pencils →
  new_total_pencils = mechanical_pencils + new_standard_pencils →
  new_frac_mechanical = mechanical_pencils / new_total_pencils →
  new_frac_mechanical = 1 / 10 :=
by
  sorry

end NUMINAMATH_GPT_mechanical_pencils_fraction_l1245_124592


namespace NUMINAMATH_GPT_unique_two_digit_solution_l1245_124545

theorem unique_two_digit_solution :
  ∃! (u : ℕ), 9 < u ∧ u < 100 ∧ 13 * u % 100 = 52 := 
sorry

end NUMINAMATH_GPT_unique_two_digit_solution_l1245_124545


namespace NUMINAMATH_GPT_find_principal_l1245_124506

-- Define the conditions
variables (P R : ℝ) -- Define P and R as real numbers
variable (h : (P * 50) / 100 = 300) -- Introduce the equation obtained from the conditions

-- State the theorem
theorem find_principal (P R : ℝ) (h : (P * 50) / 100 = 300) : P = 600 :=
sorry

end NUMINAMATH_GPT_find_principal_l1245_124506


namespace NUMINAMATH_GPT_scientific_notation_of_number_l1245_124585

theorem scientific_notation_of_number (num : ℝ) (a b: ℝ) : 
  num = 0.0000046 ∧ 
  (a = 46 ∧ b = -7 ∨ 
   a = 4.6 ∧ b = -7 ∨ 
   a = 4.6 ∧ b = -6 ∨ 
   a = 0.46 ∧ b = -5) → 
  a = 4.6 ∧ b = -6 :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_of_number_l1245_124585


namespace NUMINAMATH_GPT_smallest_positive_integer_l1245_124553

theorem smallest_positive_integer :
  ∃ x : ℕ,
    x % 5 = 4 ∧
    x % 7 = 5 ∧
    x % 11 = 9 ∧
    x % 13 = 11 ∧
    (∀ y : ℕ, (y % 5 = 4 ∧ y % 7 = 5 ∧ y % 11 = 9 ∧ y % 13 = 11) → y ≥ x) ∧ x = 999 :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_integer_l1245_124553


namespace NUMINAMATH_GPT_terrier_hush_interval_l1245_124587

-- Definitions based on conditions
def poodle_barks_per_terrier_bark : ℕ := 2
def total_poodle_barks : ℕ := 24
def terrier_hushes : ℕ := 6

-- Derived values based on definitions
def total_terrier_barks := total_poodle_barks / poodle_barks_per_terrier_bark
def interval_hush := total_terrier_barks / terrier_hushes

-- The theorem stating the terrier's hush interval
theorem terrier_hush_interval : interval_hush = 2 := by
  have h1 : total_terrier_barks = 12 := by sorry
  have h2 : interval_hush = 2 := by sorry
  exact h2

end NUMINAMATH_GPT_terrier_hush_interval_l1245_124587


namespace NUMINAMATH_GPT_ab_eq_neg_one_l1245_124548

variable (a b : ℝ)

-- Condition for the inequality (x >= 0) -> (0 ≤ x^4 - x^3 + ax + b ≤ (x^2 - 1)^2)
def condition (a b : ℝ) : Prop :=
  ∀ x : ℝ, x ≥ 0 → 
    0 ≤ x^4 - x^3 + a * x + b ∧ 
    x^4 - x^3 + a * x + b ≤ (x^2 - 1)^2

-- Main statement to prove that assuming the condition, a * b = -1
theorem ab_eq_neg_one (h : condition a b) : a * b = -1 := 
  sorry

end NUMINAMATH_GPT_ab_eq_neg_one_l1245_124548


namespace NUMINAMATH_GPT_first_digit_after_decimal_correct_l1245_124562

noncomputable def first_digit_after_decimal (n: ℕ) : ℕ :=
  if n % 2 = 0 then 9 else 4

theorem first_digit_after_decimal_correct (n : ℕ) :
  (first_digit_after_decimal n = 9 ↔ n % 2 = 0) ∧ (first_digit_after_decimal n = 4 ↔ n % 2 = 1) :=
by
  sorry

end NUMINAMATH_GPT_first_digit_after_decimal_correct_l1245_124562


namespace NUMINAMATH_GPT_monotonicity_and_extreme_values_l1245_124597

noncomputable def f (x : ℝ) : ℝ := Real.log x - x

theorem monotonicity_and_extreme_values :
  (∀ x : ℝ, 0 < x ∧ x < 1 → f x < f (1 - x)) ∧
  (∀ x : ℝ, x > 1 → f x < f 1) ∧
  f 1 = -1 :=
by 
  sorry

end NUMINAMATH_GPT_monotonicity_and_extreme_values_l1245_124597


namespace NUMINAMATH_GPT_awareness_not_related_to_education_level_l1245_124520

def low_education : ℕ := 35 + 35 + 80 + 40 + 60 + 150
def high_education : ℕ := 55 + 64 + 6 + 110 + 140 + 25

def a : ℕ := 150
def b : ℕ := 125
def c : ℕ := 250
def d : ℕ := 275
def n : ℕ := 800

-- K^2 calculation
def K2 : ℚ := (n * (a * d - b * c)^2) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Critical value for 95% confidence
def critical_value_95 : ℚ := 3.841

theorem awareness_not_related_to_education_level : K2 < critical_value_95 :=
by
  -- proof to be added here
  sorry

end NUMINAMATH_GPT_awareness_not_related_to_education_level_l1245_124520


namespace NUMINAMATH_GPT_find_coordinates_of_P_l1245_124518

/-- Let the curve C be defined by the equation y = x^3 - 10x + 3 and point P lies on this curve in the second quadrant.
We are given that the slope of the tangent line to the curve at point P is 2. We need to find the coordinates of P.
--/
theorem find_coordinates_of_P :
  ∃ (x y : ℝ), (y = x ^ 3 - 10 * x + 3) ∧ (3 * x ^ 2 - 10 = 2) ∧ (x < 0) ∧ (x = -2) ∧ (y = 15) :=
by
  sorry

end NUMINAMATH_GPT_find_coordinates_of_P_l1245_124518


namespace NUMINAMATH_GPT_Rachel_drinks_correct_glasses_l1245_124549

def glasses_Sunday : ℕ := 2
def glasses_Monday : ℕ := 4
def glasses_TuesdayToFriday : ℕ := 3
def days_TuesdayToFriday : ℕ := 4
def ounces_per_glass : ℕ := 10
def total_goal : ℕ := 220
def glasses_Saturday : ℕ := 4

theorem Rachel_drinks_correct_glasses :
  ounces_per_glass * (glasses_Sunday + glasses_Monday + days_TuesdayToFriday * glasses_TuesdayToFriday + glasses_Saturday) = total_goal :=
sorry

end NUMINAMATH_GPT_Rachel_drinks_correct_glasses_l1245_124549


namespace NUMINAMATH_GPT_depth_of_pond_l1245_124594

theorem depth_of_pond (L W V D : ℝ) (hL : L = 20) (hW : W = 10) (hV : V = 1000) (hV_formula : V = L * W * D) : D = 5 := by
  -- at this point, you could start the proof which involves deriving D from hV and hV_formula using arithmetic rules.
  sorry

end NUMINAMATH_GPT_depth_of_pond_l1245_124594


namespace NUMINAMATH_GPT_fraction_left_handed_non_throwers_is_one_third_l1245_124596

theorem fraction_left_handed_non_throwers_is_one_third :
  let total_players := 70
  let throwers := 31
  let right_handed := 57
  let non_throwers := total_players - throwers
  let right_handed_non_throwers := right_handed - throwers
  let left_handed_non_throwers := non_throwers - right_handed_non_throwers
  (left_handed_non_throwers : ℝ) / non_throwers = 1 / 3 := by
  sorry

end NUMINAMATH_GPT_fraction_left_handed_non_throwers_is_one_third_l1245_124596


namespace NUMINAMATH_GPT_quadratic_completion_l1245_124569

theorem quadratic_completion :
  (∀ x : ℝ, (∃ a h k : ℝ, (x ^ 2 - 2 * x - 1 = a * (x - h) ^ 2 + k) ∧ (a = 1) ∧ (h = 1) ∧ (k = -2))) :=
sorry

end NUMINAMATH_GPT_quadratic_completion_l1245_124569


namespace NUMINAMATH_GPT_find_teacher_age_l1245_124555

noncomputable def age_of_teacher (avg_age_students : ℕ) (num_students : ℕ) 
                                (avg_age_inclusive : ℕ) (num_people_inclusive : ℕ) : ℕ :=
  let total_age_students := num_students * avg_age_students
  let total_age_inclusive := num_people_inclusive * avg_age_inclusive
  total_age_inclusive - total_age_students

theorem find_teacher_age : age_of_teacher 15 10 16 11 = 26 := 
by 
  sorry

end NUMINAMATH_GPT_find_teacher_age_l1245_124555


namespace NUMINAMATH_GPT_cistern_fill_time_l1245_124590

theorem cistern_fill_time (hA : ℝ) (hB : ℝ) (hC : ℝ) : hA = 12 → hB = 18 → hC = 15 → 
  1 / ((1 / hA) + (1 / hB) - (1 / hC)) = 180 / 13 :=
by
  intros hA_eq hB_eq hC_eq
  rw [hA_eq, hB_eq, hC_eq]
  sorry

end NUMINAMATH_GPT_cistern_fill_time_l1245_124590


namespace NUMINAMATH_GPT_roots_of_quadratic_expression_l1245_124552

theorem roots_of_quadratic_expression :
    (∀ x: ℝ, (x^2 + 3 * x - 2 = 0) → ∃ x₁ x₂: ℝ, x = x₁ ∨ x = x₂) ∧ 
    (∀ x₁ x₂ : ℝ, (x₁ + x₂ = -3) ∧ (x₁ * x₂ = -2) → x₁^2 + 2 * x₁ - x₂ = 5) :=
by
  sorry

end NUMINAMATH_GPT_roots_of_quadratic_expression_l1245_124552


namespace NUMINAMATH_GPT_sec_225_eq_neg_sqrt2_csc_225_eq_neg_sqrt2_l1245_124523

noncomputable def sec (θ : ℝ) : ℝ := 1 / Real.cos θ
noncomputable def csc (θ : ℝ) : ℝ := 1 / Real.sin θ

theorem sec_225_eq_neg_sqrt2 :
  sec (225 * Real.pi / 180) = -Real.sqrt 2 := sorry

theorem csc_225_eq_neg_sqrt2 :
  csc (225 * Real.pi / 180) = -Real.sqrt 2 := sorry

end NUMINAMATH_GPT_sec_225_eq_neg_sqrt2_csc_225_eq_neg_sqrt2_l1245_124523


namespace NUMINAMATH_GPT_find_x_values_for_inverse_l1245_124571

def f (x : ℝ) : ℝ := x^2 - 3 * x - 4

theorem find_x_values_for_inverse :
  ∃ (x : ℝ), (f x = 2 + 2 * Real.sqrt 2 ∨ f x = 2 - 2 * Real.sqrt 2) ∧ f x = x :=
sorry

end NUMINAMATH_GPT_find_x_values_for_inverse_l1245_124571


namespace NUMINAMATH_GPT_phil_packs_duration_l1245_124547

noncomputable def total_cards_left_after_fire : ℕ := 520
noncomputable def total_cards_initially : ℕ := total_cards_left_after_fire * 2
noncomputable def cards_per_pack : ℕ := 20
noncomputable def packs_bought_weeks : ℕ := total_cards_initially / cards_per_pack

theorem phil_packs_duration : packs_bought_weeks = 52 := by
  sorry

end NUMINAMATH_GPT_phil_packs_duration_l1245_124547


namespace NUMINAMATH_GPT_six_digits_sum_l1245_124582

theorem six_digits_sum 
  (a b c d e f g : ℕ) 
  (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d) (h4 : a ≠ e) (h5 : a ≠ f) (h6 : a ≠ g)
  (h7 : b ≠ c) (h8 : b ≠ d) (h9 : b ≠ e) (h10 : b ≠ f) (h11 : b ≠ g)
  (h12 : c ≠ d) (h13 : c ≠ e) (h14 : c ≠ f) (h15 : c ≠ g)
  (h16 : d ≠ e) (h17 : d ≠ f) (h18 : d ≠ g)
  (h19 : e ≠ f) (h20 : e ≠ g)
  (h21 : f ≠ g)
  (h22 : 2 ≤ a) (h23 : a ≤ 9) 
  (h24 : 2 ≤ b) (h25 : b ≤ 9) 
  (h26 : 2 ≤ c) (h27 : c ≤ 9)
  (h28 : 2 ≤ d) (h29 : d ≤ 9)
  (h30 : 2 ≤ e) (h31 : e ≤ 9)
  (h32 : 2 ≤ f) (h33 : f ≤ 9)
  (h34 : 2 ≤ g) (h35 : g ≤ 9)
  (h36 : a + b + c = 25)
  (h37 : d + e + f + g = 15)
  (h38 : b = e) :
  a + b + c + d + f + g = 31 := 
sorry

end NUMINAMATH_GPT_six_digits_sum_l1245_124582


namespace NUMINAMATH_GPT_no_primes_in_sequence_l1245_124511

-- Definitions and conditions derived from the problem statement
variable (a : ℕ → ℕ) -- sequence of natural numbers
variable (increasing : ∀ n, a n < a (n + 1)) -- increasing sequence
variable (is_arith_or_geom : ∀ n, (2 * a (n + 1) = a n + a (n + 2)) ∨ (a (n + 1) ^ 2 = a n * a (n + 2))) -- arithmetic or geometric progression condition
variable (divisible_by_four : a 0 % 4 = 0 ∧ a 1 % 4 = 0) -- first two numbers divisible by 4

-- The statement to prove: no prime numbers exist in the sequence
theorem no_primes_in_sequence : ∀ n, ¬ (Nat.Prime (a n)) :=
by 
  sorry

end NUMINAMATH_GPT_no_primes_in_sequence_l1245_124511


namespace NUMINAMATH_GPT_prove_sufficient_and_necessary_l1245_124529

-- The definition of the focus of the parabola y^2 = 4x.
def focus_parabola : (ℝ × ℝ) := (1, 0)

-- The condition that the line passes through a given point.
def line_passes_through (m b : ℝ) (p : ℝ × ℝ) : Prop := 
  p.2 = m * p.1 + b

-- Let y = x + b and the equation of the parabola be y^2 = 4x.
def sufficient_and_necessary (b : ℝ) : Prop :=
  line_passes_through 1 b focus_parabola ↔ b = -1

theorem prove_sufficient_and_necessary : sufficient_and_necessary (-1) :=
by
  sorry

end NUMINAMATH_GPT_prove_sufficient_and_necessary_l1245_124529


namespace NUMINAMATH_GPT_hyperbola_parabola_foci_l1245_124501

-- Definition of the hyperbola
def hyperbola (k : ℝ) (x y : ℝ) : Prop := y^2 / 5 - x^2 / k = 1

-- Definition of the parabola
def parabola (x y : ℝ) : Prop := x^2 = 12 * y

-- Condition that both curves have the same foci
def same_foci (focus : ℝ) (x y : ℝ) : Prop := focus = 3 ∧ (parabola x y → ((0, focus) : ℝ×ℝ) = (0, 3)) ∧ (∃ k : ℝ, hyperbola k x y ∧ ((0, focus) : ℝ×ℝ) = (0, 3))

theorem hyperbola_parabola_foci (k : ℝ) (x y : ℝ) : same_foci 3 x y → k = -4 := 
by {
  sorry
}

end NUMINAMATH_GPT_hyperbola_parabola_foci_l1245_124501


namespace NUMINAMATH_GPT_four_sin_t_plus_cos_2t_bounds_l1245_124593

theorem four_sin_t_plus_cos_2t_bounds (t : ℝ) : -5 ≤ 4 * Real.sin t + Real.cos (2 * t) ∧ 4 * Real.sin t + Real.cos (2 * t) ≤ 3 := by
  sorry

end NUMINAMATH_GPT_four_sin_t_plus_cos_2t_bounds_l1245_124593


namespace NUMINAMATH_GPT_complement_of_M_in_U_l1245_124580

def U := Set.univ (α := ℝ)
def M := {x : ℝ | x < -2 ∨ x > 8}
def compl_M := {x : ℝ | -2 ≤ x ∧ x ≤ 8}

theorem complement_of_M_in_U : compl_M = U \ M :=
by
  sorry

end NUMINAMATH_GPT_complement_of_M_in_U_l1245_124580


namespace NUMINAMATH_GPT_aunt_angela_nieces_l1245_124540

theorem aunt_angela_nieces (total_jellybeans : ℕ)
                           (jellybeans_per_child : ℕ)
                           (num_nephews : ℕ)
                           (num_nieces : ℕ) 
                           (total_children : ℕ) 
                           (h1 : total_jellybeans = 70)
                           (h2 : jellybeans_per_child = 14)
                           (h3 : num_nephews = 3)
                           (h4 : total_children = total_jellybeans / jellybeans_per_child)
                           (h5 : total_children = num_nephews + num_nieces) :
                           num_nieces = 2 :=
by
  sorry

end NUMINAMATH_GPT_aunt_angela_nieces_l1245_124540


namespace NUMINAMATH_GPT_wire_around_field_l1245_124539

theorem wire_around_field 
  (area_square : ℕ)
  (total_length_wire : ℕ)
  (h_area : area_square = 69696)
  (h_total_length : total_length_wire = 15840) :
  (total_length_wire / (4 * Int.natAbs (Int.sqrt area_square))) = 15 :=
  sorry

end NUMINAMATH_GPT_wire_around_field_l1245_124539


namespace NUMINAMATH_GPT_inequality_a_b_c_d_l1245_124564

theorem inequality_a_b_c_d 
  (a b c d : ℝ) 
  (h0 : 0 ≤ a) 
  (h1 : a ≤ b) 
  (h2 : b ≤ c) 
  (h3 : c ≤ d) :
  a^b * b^c * c^d * d^a ≥ b^a * c^b * d^c * a^d := 
by
  sorry

end NUMINAMATH_GPT_inequality_a_b_c_d_l1245_124564


namespace NUMINAMATH_GPT_incorrect_statement_l1245_124513

def data_set : List ℤ := [10, 8, 6, 9, 8, 7, 8]

theorem incorrect_statement : 
  let mode := 8
  let median := 8
  let mean := 8
  let variance := 8
  (∃ x ∈ data_set, x ≠ 8) → -- suppose there is at least one element in the dataset not equal to 8
  (1 / 7 : ℚ) * (4 + 0 + 4 + 1 + 0 + 1 + 0) ≠ 8 := -- calculating real variance from dataset
by
  sorry

end NUMINAMATH_GPT_incorrect_statement_l1245_124513


namespace NUMINAMATH_GPT_sum_of_possible_ks_l1245_124556

theorem sum_of_possible_ks :
  ∃ S : Finset ℕ, (∀ (j k : ℕ), j > 0 ∧ k > 0 → (1 / j + 1 / k = 1 / 4) ↔ k ∈ S) ∧ S.sum id = 51 :=
  sorry

end NUMINAMATH_GPT_sum_of_possible_ks_l1245_124556


namespace NUMINAMATH_GPT_solve_equations_l1245_124519

theorem solve_equations :
  (∀ x : ℝ, (1 / 2) * (2 * x - 5) ^ 2 - 2 = 0 ↔ x = 7 / 2 ∨ x = 3 / 2) ∧
  (∀ x : ℝ, x ^ 2 - 4 * x - 4 = 0 ↔ x = 2 + 2 * Real.sqrt 2 ∨ x = 2 - 2 * Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_GPT_solve_equations_l1245_124519


namespace NUMINAMATH_GPT_paths_from_A_to_B_via_C_l1245_124522

open Classical

-- Definitions based on conditions
variables (lattice : Type) [PartialOrder lattice]
variables (A B C : lattice)
variables (first_red first_blue second_red second_blue first_green second_green orange : lattice)

-- Conditions encoded as hypotheses
def direction_changes : Prop :=
  -- Arrow from first green to orange is now one way from orange to green
  ∀ x : lattice, x = first_green → orange < x ∧ ¬ (x < orange) ∧
  -- Additional stop at point C located directly after the first blue arrows
  (C < first_blue ∨ first_blue < C)

-- Now stating the proof problem
theorem paths_from_A_to_B_via_C :
  direction_changes lattice first_green orange first_blue C →
  -- Total number of paths from A to B via C is 12
  (2 + 2) * 3 * 1 = 12 :=
by
  sorry

end NUMINAMATH_GPT_paths_from_A_to_B_via_C_l1245_124522


namespace NUMINAMATH_GPT_expression_evaluation_l1245_124558

theorem expression_evaluation (x y : ℝ) (h₁ : x > y) (h₂ : y > 0) : 
    (x^(2*y) * y^x) / (y^(2*x) * x^y) = (x / y)^(y - x) :=
by
  sorry

end NUMINAMATH_GPT_expression_evaluation_l1245_124558


namespace NUMINAMATH_GPT_range_of_a_l1245_124574

noncomputable def e := Real.exp 1

theorem range_of_a (a : Real) 
  (h : ∀ x : Real, 1 ≤ x ∧ x ≤ 2 → Real.exp x - a ≥ 0) : 
  a ≤ e :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1245_124574


namespace NUMINAMATH_GPT_find_a_l1245_124570

theorem find_a (a x : ℝ) (h1: a - 2 ≤ x) (h2: x ≤ a + 1) (h3 : -x^2 + 2 * x + 3 = 3) :
  a = 2 := sorry

end NUMINAMATH_GPT_find_a_l1245_124570


namespace NUMINAMATH_GPT_compute_sixth_power_sum_l1245_124581

theorem compute_sixth_power_sum (ζ1 ζ2 ζ3 : ℂ) 
  (h1 : ζ1 + ζ2 + ζ3 = 2)
  (h2 : ζ1^2 + ζ2^2 + ζ3^2 = 5)
  (h3 : ζ1^4 + ζ2^4 + ζ3^4 = 29) :
  ζ1^6 + ζ2^6 + ζ3^6 = 101.40625 := 
by
  sorry

end NUMINAMATH_GPT_compute_sixth_power_sum_l1245_124581


namespace NUMINAMATH_GPT_minimum_boxes_to_eliminate_l1245_124526

theorem minimum_boxes_to_eliminate (total_boxes remaining_boxes : ℕ) 
  (high_value_boxes : ℕ) (h1 : total_boxes = 30) (h2 : high_value_boxes = 10)
  (h3 : remaining_boxes = total_boxes - 20) :
  remaining_boxes ≥ high_value_boxes → remaining_boxes = 10 :=
by 
  sorry

end NUMINAMATH_GPT_minimum_boxes_to_eliminate_l1245_124526


namespace NUMINAMATH_GPT_simplify_expression_l1245_124500

theorem simplify_expression :
  ((0.3 * 0.2) / (0.4 * 0.5)) - (0.1 * 0.6) = 0.24 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1245_124500


namespace NUMINAMATH_GPT_fare_ratio_l1245_124542

theorem fare_ratio (F1 F2 : ℕ) (h1 : F1 = 96000) (h2 : F1 + F2 = 224000) : F1 / (Nat.gcd F1 F2) = 3 ∧ F2 / (Nat.gcd F1 F2) = 4 :=
by
  sorry

end NUMINAMATH_GPT_fare_ratio_l1245_124542


namespace NUMINAMATH_GPT_range_of_x_range_of_a_l1245_124591

-- Definitions of propositions p and q
def p (a x : ℝ) := (x - a) * (x - 3 * a) < 0
def q (x : ℝ) := (x - 3) / (x - 2) ≤ 0

-- Question 1
theorem range_of_x (a x : ℝ) : a = 1 → p a x ∧ q x → 2 < x ∧ x < 3 := by
  sorry

-- Question 2
theorem range_of_a (a : ℝ) : (∀ x, ¬p a x → ¬q x) → (∀ x, q x → p a x) → 1 < a ∧ a ≤ 2 := by
  sorry

end NUMINAMATH_GPT_range_of_x_range_of_a_l1245_124591


namespace NUMINAMATH_GPT_inequality_sinx_plus_y_cosx_plus_y_l1245_124576

open Real

theorem inequality_sinx_plus_y_cosx_plus_y (
  y x : ℝ
) (hx : x ∈ Set.Icc (π / 4) (3 * π / 4)) (hy : y ∈ Set.Icc (π / 4) (3 * π / 4)) :
  sin (x + y) + cos (x + y) ≤ sin x + cos x + sin y + cos y :=
sorry

end NUMINAMATH_GPT_inequality_sinx_plus_y_cosx_plus_y_l1245_124576


namespace NUMINAMATH_GPT_average_student_headcount_l1245_124515

theorem average_student_headcount (headcount_03_04 headcount_04_05 : ℕ) 
  (h1 : headcount_03_04 = 10500) 
  (h2 : headcount_04_05 = 10700) : 
  (headcount_03_04 + headcount_04_05) / 2 = 10600 := 
by
  sorry

end NUMINAMATH_GPT_average_student_headcount_l1245_124515


namespace NUMINAMATH_GPT_count_valid_m_l1245_124541

def is_divisor (a b : ℕ) : Prop := b % a = 0

def valid_m (m : ℕ) : Prop :=
  m > 1 ∧ is_divisor m 480 ∧ (480 / m) > 1

theorem count_valid_m : (∃ m, valid_m m) → Nat.card {m // valid_m m} = 22 :=
by sorry

end NUMINAMATH_GPT_count_valid_m_l1245_124541


namespace NUMINAMATH_GPT_probability_grade_A_l1245_124554

-- Defining probabilities
def P_B : ℝ := 0.05
def P_C : ℝ := 0.03

-- Theorem: proving the probability of Grade A
theorem probability_grade_A : 1 - P_B - P_C = 0.92 :=
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_probability_grade_A_l1245_124554


namespace NUMINAMATH_GPT_anna_total_value_l1245_124595

theorem anna_total_value (total_bills : ℕ) (five_dollar_bills : ℕ) (ten_dollar_bills : ℕ)
  (h1 : total_bills = 12) (h2 : five_dollar_bills = 4) (h3 : ten_dollar_bills = total_bills - five_dollar_bills) :
  5 * five_dollar_bills + 10 * ten_dollar_bills = 100 := by
  sorry

end NUMINAMATH_GPT_anna_total_value_l1245_124595


namespace NUMINAMATH_GPT_work_done_in_days_l1245_124546

theorem work_done_in_days (M B : ℕ) (x : ℕ) 
  (h1 : 12 * 2 * B + 16 * B = 200 * B / 5) 
  (h2 : 13 * 2 * B + 24 * B = 50 * x * B)
  (h3 : M = 2 * B) : 
  x = 4 := 
by
  sorry

end NUMINAMATH_GPT_work_done_in_days_l1245_124546


namespace NUMINAMATH_GPT_seokjin_higher_than_jungkook_l1245_124533

variable (Jungkook_yoojeong_seokjin_stairs : ℕ)

def jungkook_stair := 19
def yoojeong_stair := jungkook_stair + 8
def seokjin_stair := yoojeong_stair - 5

theorem seokjin_higher_than_jungkook : seokjin_stair - jungkook_stair = 3 :=
by sorry

end NUMINAMATH_GPT_seokjin_higher_than_jungkook_l1245_124533


namespace NUMINAMATH_GPT_intersection_of_lines_l1245_124503

-- Define the conditions of the problem
def first_line (x y : ℝ) : Prop := y = -3 * x + 1
def second_line (x y : ℝ) : Prop := y + 1 = 15 * x

-- Prove the intersection point of the two lines
theorem intersection_of_lines : 
  ∃ (x y : ℝ), first_line x y ∧ second_line x y ∧ x = 1 / 9 ∧ y = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_lines_l1245_124503


namespace NUMINAMATH_GPT_garageHasWheels_l1245_124559

-- Define the conditions
def bikeWheelsPerBike : Nat := 2
def bikesInGarage : Nat := 10

-- State the theorem to be proved
theorem garageHasWheels : bikesInGarage * bikeWheelsPerBike = 20 := by
  sorry

end NUMINAMATH_GPT_garageHasWheels_l1245_124559


namespace NUMINAMATH_GPT_inequality_solution_l1245_124509

theorem inequality_solution (x : ℝ) :
  (2 * x - 1 > 0 ∧ x + 1 ≤ 3) ↔ (1 / 2 < x ∧ x ≤ 2) :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l1245_124509


namespace NUMINAMATH_GPT_triangle_area_l1245_124525

theorem triangle_area : 
  ∀ (A B C : ℝ × ℝ), 
  A = (0, 0) → 
  B = (4, 0) → 
  C = (2, 6) → 
  (1 / 2 : ℝ) * (4 : ℝ) * (6 : ℝ) = (12.0 : ℝ) := 
by 
  intros A B C hA hB hC
  simp [hA, hB, hC]
  norm_num

end NUMINAMATH_GPT_triangle_area_l1245_124525


namespace NUMINAMATH_GPT_not_perfect_power_l1245_124537

theorem not_perfect_power (k : ℕ) (h : k ≥ 2) : ∀ m n : ℕ, m > 1 → n > 1 → 10^k - 1 ≠ m ^ n :=
by 
  sorry

end NUMINAMATH_GPT_not_perfect_power_l1245_124537


namespace NUMINAMATH_GPT_solve_for_angle_a_l1245_124524

theorem solve_for_angle_a (a b c d e : ℝ) (h1 : a + b + c + d = 360) (h2 : e = 360 - (a + d)) : a = 360 - e - b - c :=
by
  sorry

end NUMINAMATH_GPT_solve_for_angle_a_l1245_124524


namespace NUMINAMATH_GPT_quadratic_roots_l1245_124505

theorem quadratic_roots (c : ℝ) 
  (h : ∀ x : ℝ, (x^2 - 3*x + c = 0) ↔ (x = (3 + Real.sqrt c) / 2 ∨ x = (3 - Real.sqrt c) / 2)) :
  c = 9 / 5 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_roots_l1245_124505


namespace NUMINAMATH_GPT_range_of_4x_plus_2y_l1245_124543

theorem range_of_4x_plus_2y (x y : ℝ) 
  (h₁ : 1 ≤ x + y ∧ x + y ≤ 3)
  (h₂ : -1 ≤ x - y ∧ x - y ≤ 1) : 
  2 ≤ 4 * x + 2 * y ∧ 4 * x + 2 * y ≤ 10 :=
sorry

end NUMINAMATH_GPT_range_of_4x_plus_2y_l1245_124543


namespace NUMINAMATH_GPT_min_students_l1245_124528

theorem min_students (M D : ℕ) (hD : D = 5) (h_ratio : (M: ℚ) / (M + D) > 0.6) : M + D = 13 :=
by 
  sorry

end NUMINAMATH_GPT_min_students_l1245_124528


namespace NUMINAMATH_GPT_skipping_rope_equation_correct_l1245_124534

-- Definitions of constraints
variable (x : ℕ) -- Number of skips per minute by Xiao Ji
variable (H1 : 0 < x) -- The number of skips per minute by Xiao Ji is positive
variable (H2 : 100 / x * x = 100) -- Xiao Ji skips exactly 100 times

-- Xiao Fan's conditions
variable (H3 : 100 + 20 = 120) -- Xiao Fan skips 20 more times than Xiao Ji
variable (H4 : x + 30 > 0) -- Xiao Fan skips 30 more times per minute than Xiao Ji

-- Prove the equation is correct
theorem skipping_rope_equation_correct :
  100 / x = 120 / (x + 30) :=
by
  sorry

end NUMINAMATH_GPT_skipping_rope_equation_correct_l1245_124534


namespace NUMINAMATH_GPT_M_intersect_N_equals_M_l1245_124512

-- Define the sets M and N
def M := { x : ℝ | x^2 - 3 * x + 2 = 0 }
def N := { x : ℝ | x * (x - 1) * (x - 2) = 0 }

-- The theorem we want to prove
theorem M_intersect_N_equals_M : M ∩ N = M := 
by 
  sorry

end NUMINAMATH_GPT_M_intersect_N_equals_M_l1245_124512


namespace NUMINAMATH_GPT_num_of_positive_divisors_l1245_124516

-- Given conditions
variables {x y z : ℕ}
variables (p1 p2 p3 : ℕ) -- primes
variables (h1 : x = p1 ^ 3) (h2 : y = p2 ^ 3) (h3 : z = p3 ^ 3)
variables (hx : x ≠ y) (hy : y ≠ z) (hz : z ≠ x)

-- Lean statement to prove
theorem num_of_positive_divisors (hx3 : x = p1 ^ 3) (hy3 : y = p2 ^ 3) (hz3 : z = p3 ^ 3) 
    (Hdist : p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3) : 
    ∃ n : ℕ, n = 10 * 13 * 7 ∧ n = (x^3 * y^4 * z^2).factors.length :=
sorry

end NUMINAMATH_GPT_num_of_positive_divisors_l1245_124516


namespace NUMINAMATH_GPT_correct_linear_regression_statement_l1245_124583

-- Definitions based on the conditions:
def linear_regression (b a e : ℝ) (x : ℝ) : ℝ := b * x + a + e

def statement_A (b a e : ℝ) (x : ℝ) : Prop := linear_regression b a e x = b * x + a + e

def statement_B (b a e : ℝ) (x : ℝ) : Prop := ∀ x1 x2, (linear_regression b a e x1 ≠ linear_regression b a e x2) → (x1 ≠ x2)

def statement_C (b a e : ℝ) (x : ℝ) : Prop := ∃ (other_factors : ℝ), linear_regression b a e x = b * x + a + other_factors + e

def statement_D (b a e : ℝ) (x : ℝ) : Prop := (e ≠ 0) → false

-- The proof statement
theorem correct_linear_regression_statement (b a e : ℝ) (x : ℝ) :
  (statement_C b a e x) :=
sorry

end NUMINAMATH_GPT_correct_linear_regression_statement_l1245_124583


namespace NUMINAMATH_GPT_train_cross_time_l1245_124566

def length_of_train : Float := 135.0 -- in meters
def speed_of_train_kmh : Float := 45.0 -- in kilometers per hour
def length_of_bridge : Float := 240.03 -- in meters

def speed_of_train_ms : Float := speed_of_train_kmh * 1000.0 / 3600.0

def total_distance : Float := length_of_train + length_of_bridge

def time_to_cross : Float := total_distance / speed_of_train_ms

theorem train_cross_time : time_to_cross = 30.0024 :=
by
  sorry

end NUMINAMATH_GPT_train_cross_time_l1245_124566


namespace NUMINAMATH_GPT_probability_of_consecutive_triplets_l1245_124577

def total_ways_to_select_3_days (n : ℕ) : ℕ :=
  Nat.choose n 3

def number_of_consecutive_triplets (n : ℕ) : ℕ :=
  n - 2

theorem probability_of_consecutive_triplets :
  let total_ways := total_ways_to_select_3_days 10
  let consecutive_triplets := number_of_consecutive_triplets 10
  (consecutive_triplets : ℚ) / total_ways = 1 / 15 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_consecutive_triplets_l1245_124577


namespace NUMINAMATH_GPT_MattRate_l1245_124586

variable (M : ℝ) (t : ℝ)

def MattRateCondition : Prop := M * t = 220
def TomRateCondition : Prop := (M + 5) * t = 275

theorem MattRate (h1 : MattRateCondition M t) (h2 : TomRateCondition M t) : M = 20 := by
  sorry

end NUMINAMATH_GPT_MattRate_l1245_124586


namespace NUMINAMATH_GPT_number_of_guest_cars_l1245_124502

-- Definitions and conditions
def total_wheels : ℕ := 48
def mother_car_wheels : ℕ := 4
def father_jeep_wheels : ℕ := 4
def wheels_per_car : ℕ := 4

-- Theorem statement
theorem number_of_guest_cars (total_wheels mother_car_wheels father_jeep_wheels wheels_per_car : ℕ) : ℕ :=
  (total_wheels - (mother_car_wheels + father_jeep_wheels)) / wheels_per_car

-- Specific instance for the problem
example : number_of_guest_cars 48 4 4 4 = 10 := 
by
  sorry

end NUMINAMATH_GPT_number_of_guest_cars_l1245_124502


namespace NUMINAMATH_GPT_pages_and_cost_calculation_l1245_124531

noncomputable def copy_pages_cost (cents_per_5_pages : ℕ) (total_cents : ℕ) (discount_threshold : ℕ) (discount_rate : ℝ) : ℝ :=
if total_cents < discount_threshold * (cents_per_5_pages / 5) then
  total_cents / (cents_per_5_pages / 5)
else
  let num_pages_before_discount := discount_threshold
  let remaining_pages := total_cents / (cents_per_5_pages / 5) - num_pages_before_discount
  let cost_before_discount := num_pages_before_discount * (cents_per_5_pages / 5)
  let discounted_cost := remaining_pages * (cents_per_5_pages / 5) * (1 - discount_rate)
  cost_before_discount + discounted_cost

theorem pages_and_cost_calculation :
  let cents_per_5_pages := 10
  let total_cents := 5000
  let discount_threshold := 1000
  let discount_rate := 0.10
  let num_pages := (cents_per_5_pages * 2500) / 5
  let cost := copy_pages_cost cents_per_5_pages total_cents discount_threshold discount_rate
  (num_pages = 2500) ∧ (cost = 4700) :=
by
  sorry

end NUMINAMATH_GPT_pages_and_cost_calculation_l1245_124531


namespace NUMINAMATH_GPT_range_of_angle_B_l1245_124536

theorem range_of_angle_B {A B C : ℝ} (a b c : ℝ) (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a)
  (h_sinB : Real.sin B = Real.sqrt (Real.sin A * Real.sin C)) :
  0 < B ∧ B ≤ Real.pi / 3 :=
sorry

end NUMINAMATH_GPT_range_of_angle_B_l1245_124536


namespace NUMINAMATH_GPT_ratio_female_to_male_l1245_124527

-- Definitions for the conditions
def average_age_female (f : ℕ) : ℕ := 40 * f
def average_age_male (m : ℕ) : ℕ := 25 * m
def average_age_total (f m : ℕ) : ℕ := (30 * (f + m))

-- Statement to prove
theorem ratio_female_to_male (f m : ℕ) 
  (h_avg_f: average_age_female f = 40 * f)
  (h_avg_m: average_age_male m = 25 * m)
  (h_avg_total: average_age_total f m = 30 * (f + m)) : 
  f / m = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_female_to_male_l1245_124527


namespace NUMINAMATH_GPT_total_weight_of_carrots_and_cucumbers_is_875_l1245_124514

theorem total_weight_of_carrots_and_cucumbers_is_875 :
  ∀ (carrots : ℕ) (cucumbers : ℕ),
    carrots = 250 →
    cucumbers = (5 * carrots) / 2 →
    carrots + cucumbers = 875 := 
by
  intros carrots cucumbers h_carrots h_cucumbers
  rw [h_carrots, h_cucumbers]
  sorry

end NUMINAMATH_GPT_total_weight_of_carrots_and_cucumbers_is_875_l1245_124514


namespace NUMINAMATH_GPT_lollipop_count_l1245_124504

theorem lollipop_count (total_cost one_lollipop_cost : ℚ) (h1 : total_cost = 90) (h2 : one_lollipop_cost = 0.75) : total_cost / one_lollipop_cost = 120 :=
by
  sorry

end NUMINAMATH_GPT_lollipop_count_l1245_124504


namespace NUMINAMATH_GPT_remainder_is_20_l1245_124572

theorem remainder_is_20 :
  ∀ (larger smaller quotient remainder : ℕ),
    (larger = 1634) →
    (larger - smaller = 1365) →
    (larger = quotient * smaller + remainder) →
    (quotient = 6) →
    remainder = 20 :=
by
  intros larger smaller quotient remainder h_larger h_difference h_division h_quotient
  sorry

end NUMINAMATH_GPT_remainder_is_20_l1245_124572


namespace NUMINAMATH_GPT_unknown_number_is_three_or_twenty_seven_l1245_124560

theorem unknown_number_is_three_or_twenty_seven
    (x y : ℝ)
    (h1 : y - 3 = x - y)
    (h2 : (y - 6) / 3 = x / (y - 6)) :
    x = 3 ∨ x = 27 :=
by
  sorry

end NUMINAMATH_GPT_unknown_number_is_three_or_twenty_seven_l1245_124560


namespace NUMINAMATH_GPT_complement_intersection_l1245_124568

open Set

variable {R : Type} [LinearOrderedField R]

def P : Set R := {x | x^2 - 2*x ≥ 0}
def Q : Set R := {x | 1 < x ∧ x ≤ 3}

theorem complement_intersection : (compl P ∩ Q) = {x : R | 1 < x ∧ x < 2} := by
  sorry

end NUMINAMATH_GPT_complement_intersection_l1245_124568


namespace NUMINAMATH_GPT_range_of_x_for_odd_function_l1245_124565

theorem range_of_x_for_odd_function (f : ℝ → ℝ) (domain : Set ℝ)
  (h_odd : ∀ x ∈ domain, f (-x) = -f x)
  (h_mono : ∀ x y, 0 < x -> x < y -> f x < f y)
  (h_f3 : f 3 = 0)
  (h_ineq : ∀ x, x ∈ domain -> x * (f x - f (-x)) < 0) : 
  ∀ x, x * f x < 0 ↔ -3 < x ∧ x < 0 ∨ 0 < x ∧ x < 3 :=
by sorry

end NUMINAMATH_GPT_range_of_x_for_odd_function_l1245_124565


namespace NUMINAMATH_GPT_largest_integer_of_four_l1245_124567

theorem largest_integer_of_four (A B C D : ℤ)
  (h_diff: A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (h_order: A < B ∧ B < C ∧ C < D)
  (h_avg: (A + B + C + D) / 4 = 74)
  (h_A_min: A ≥ 29) : D = 206 :=
by
  sorry

end NUMINAMATH_GPT_largest_integer_of_four_l1245_124567


namespace NUMINAMATH_GPT_correct_cases_needed_l1245_124530

noncomputable def cases_needed (boxes_sold : ℕ) (boxes_per_case : ℕ) : ℕ :=
  (boxes_sold + boxes_per_case - 1) / boxes_per_case

theorem correct_cases_needed :
  cases_needed 10 6 = 2 ∧ -- For trefoils
  cases_needed 15 5 = 3 ∧ -- For samoas
  cases_needed 20 10 = 2  -- For thin mints
:= by
  sorry

end NUMINAMATH_GPT_correct_cases_needed_l1245_124530


namespace NUMINAMATH_GPT_scientific_notation_of_130944000000_l1245_124510

theorem scientific_notation_of_130944000000 :
  130944000000 = 1.30944 * 10^11 :=
by sorry

end NUMINAMATH_GPT_scientific_notation_of_130944000000_l1245_124510


namespace NUMINAMATH_GPT_vertex_of_quadratic_function_l1245_124538

-- Define the function and constants
variables (p q : ℝ)
  (hp : p > 0)
  (hq : q > 0)

-- State the theorem
theorem vertex_of_quadratic_function : 
  ∀ p q : ℝ, p > 0 → q > 0 → 
  (∀ x : ℝ, x = - (2 * p) / (2 : ℝ) → x = -p) := 
sorry

end NUMINAMATH_GPT_vertex_of_quadratic_function_l1245_124538


namespace NUMINAMATH_GPT_train_length_is_300_l1245_124508

-- Definitions based on the conditions
def trainCrossesPlatform (L V : ℝ) : Prop :=
  L + 400 = V * 42

def trainCrossesSignalPole (L V : ℝ) : Prop :=
  L = V * 18

-- The main theorem statement
theorem train_length_is_300 (L V : ℝ)
  (h1 : trainCrossesPlatform L V)
  (h2 : trainCrossesSignalPole L V) :
  L = 300 :=
by
  sorry

end NUMINAMATH_GPT_train_length_is_300_l1245_124508
