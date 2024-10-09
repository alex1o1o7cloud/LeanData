import Mathlib

namespace ship_speed_l1306_130631

theorem ship_speed 
  (D : ℝ)
  (h1 : (D/2) - 200 = D/3)
  (S := (D / 2) / 20):
  S = 30 :=
by
  -- proof here
  sorry

end ship_speed_l1306_130631


namespace parabola_focus_distance_l1306_130693

theorem parabola_focus_distance (C : Set (ℝ × ℝ))
  (hC : ∀ x y, (y^2 = x) → (x, y) ∈ C)
  (F : ℝ × ℝ)
  (hF : F = (1/4, 0))
  (A : ℝ × ℝ)
  (hA : A = (x0, y0) ∧ (y0^2 = x0 ∧ (x0, y0) ∈ C))
  (hAF : dist A F = (5/4) * x0) :
  x0 = 1 :=
sorry

end parabola_focus_distance_l1306_130693


namespace jon_initial_fastball_speed_l1306_130691

theorem jon_initial_fastball_speed 
  (S : ℝ) -- Condition: Jon's initial fastball speed \( S \)
  (h1 : ∀ t : ℕ, t = 4 * 4)  -- Condition: Training time is 4 times for 4 weeks each
  (h2 : ∀ w : ℕ, w = 16)  -- Condition: Total weeks of training (4*4=16)
  (h3 : ∀ g : ℝ, g = 1)  -- Condition: Gains 1 mph per week
  (h4 : ∃ S_new : ℝ, S_new = (S + 16) ∧ S_new = 1.2 * S) -- Condition: Speed increases by 20%
  : S = 80 := 
sorry

end jon_initial_fastball_speed_l1306_130691


namespace YoongiHasSevenPets_l1306_130648

def YoongiPets (dogs cats : ℕ) : ℕ := dogs + cats

theorem YoongiHasSevenPets : YoongiPets 5 2 = 7 :=
by
  sorry

end YoongiHasSevenPets_l1306_130648


namespace john_total_spent_l1306_130659

-- Defining the conditions from part a)
def vacuum_cleaner_original_price : ℝ := 250
def vacuum_cleaner_discount_rate : ℝ := 0.20
def dishwasher_price : ℝ := 450
def special_offer_discount : ℝ := 75
def sales_tax_rate : ℝ := 0.07

-- The adesso to formalize part c noncomputably.
noncomputable def total_amount_spent : ℝ :=
  let vacuum_cleaner_discount := vacuum_cleaner_original_price * vacuum_cleaner_discount_rate
  let vacuum_cleaner_final_price := vacuum_cleaner_original_price - vacuum_cleaner_discount
  let total_before_special_offer := vacuum_cleaner_final_price + dishwasher_price
  let total_after_special_offer := total_before_special_offer - special_offer_discount
  let sales_tax := total_after_special_offer * sales_tax_rate
  total_after_special_offer + sales_tax

-- The proof statement
theorem john_total_spent : total_amount_spent = 615.25 := by
  sorry

end john_total_spent_l1306_130659


namespace total_amount_raised_l1306_130666

-- Definitions based on conditions
def PancakeCost : ℕ := 4
def BaconCost : ℕ := 2
def NumPancakesSold : ℕ := 60
def NumBaconSold : ℕ := 90

-- Lean statement proving that the total amount raised is $420
theorem total_amount_raised : (NumPancakesSold * PancakeCost) + (NumBaconSold * BaconCost) = 420 := by
  -- Since we are not required to prove, we use sorry here
  sorry

end total_amount_raised_l1306_130666


namespace problem_l1306_130668

noncomputable def fx (a b c : ℝ) (x : ℝ) : ℝ := a * x + b / x + c

theorem problem 
  (a b c : ℝ) 
  (h_odd : ∀ x, fx a b c x = -fx a b c (-x))
  (h_f1 : fx a b c 1 = 5 / 2)
  (h_f2 : fx a b c 2 = 17 / 4) :
  (a = 2) ∧ (b = 1 / 2) ∧ (c = 0) ∧ (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 / 2 → fx a b c x₁ > fx a b c x₂) := 
sorry

end problem_l1306_130668


namespace arithmetic_sequence_subtract_l1306_130679

theorem arithmetic_sequence_subtract (a : ℕ → ℝ) (d : ℝ) :
  (a 4 + a 6 + a 8 + a 10 + a 12 = 120) →
  (a 9 - (1 / 3) * a 11 = 16) :=
by
  sorry

end arithmetic_sequence_subtract_l1306_130679


namespace probability_of_s_in_statistics_l1306_130616

theorem probability_of_s_in_statistics :
  let totalLetters := 10
  let count_s := 3
  (count_s / totalLetters : ℚ) = 3 / 10 := by
  sorry

end probability_of_s_in_statistics_l1306_130616


namespace triangle_inequality_l1306_130638

theorem triangle_inequality
  (a b c : ℝ)
  (habc : ¬(a + b ≤ c ∨ a + c ≤ b ∨ b + c ≤ a)) :
  a * (b - c)^2 + b * (c - a)^2 + c * (a - b)^2 + 4 * a * b * c > a^3 + b^3 + c^3 := 
by {
  sorry
}

end triangle_inequality_l1306_130638


namespace division_of_floats_l1306_130683

theorem division_of_floats : 4.036 / 0.04 = 100.9 :=
by
  sorry

end division_of_floats_l1306_130683


namespace converse_of_propositions_is_true_l1306_130637

theorem converse_of_propositions_is_true :
  (∀ x : ℝ, (x = 1 ∨ x = 2) ↔ (x^2 - 3 * x + 2 = 0)) ∧
  (∀ x y : ℝ, (x^2 + y^2 = 0) ↔ (x = 0 ∧ y = 0)) := 
by {
  sorry
}

end converse_of_propositions_is_true_l1306_130637


namespace jacob_hours_l1306_130670

theorem jacob_hours (J : ℕ) (H1 : ∃ (G P : ℕ),
    G = J - 6 ∧
    P = 2 * G - 4 ∧
    J + G + P = 50) : J = 18 :=
by
  sorry

end jacob_hours_l1306_130670


namespace find_f_inv_value_l1306_130613

noncomputable def f (x : ℝ) : ℝ := 8^x
noncomputable def f_inv (y : ℝ) : ℝ := Real.logb 8 y

theorem find_f_inv_value (a : ℝ) (h : a = 8^(1/3)) : f_inv (a + 2) = Real.logb 8 (8^(1/3) + 2) := by
  sorry

end find_f_inv_value_l1306_130613


namespace large_box_chocolate_bars_l1306_130618

theorem large_box_chocolate_bars (num_small_boxes : ℕ) (chocolates_per_box : ℕ) 
  (h1 : num_small_boxes = 18) (h2 : chocolates_per_box = 28) : 
  num_small_boxes * chocolates_per_box = 504 := by
  sorry

end large_box_chocolate_bars_l1306_130618


namespace polynomial_integer_values_l1306_130619

theorem polynomial_integer_values (a b c d : ℤ) (h1 : ∃ (n : ℤ), n = (a * (-1)^3 + b * (-1)^2 - c * (-1) - d))
  (h2 : ∃ (n : ℤ), n = (a * 0^3 + b * 0^2 - c * 0 - d))
  (h3 : ∃ (n : ℤ), n = (a * 1^3 + b * 1^2 - c * 1 - d))
  (h4 : ∃ (n : ℤ), n = (a * 2^3 + b * 2^2 - c * 2 - d)) :
  ∀ x : ℤ, ∃ m : ℤ, m = a * x^3 + b * x^2 - c * x - d :=
by {
  -- proof goes here
  sorry
}

end polynomial_integer_values_l1306_130619


namespace nitrogen_mass_percentage_in_ammonium_phosphate_l1306_130655

def nitrogen_mass_percentage
  (molar_mass_N : ℚ)
  (molar_mass_H : ℚ)
  (molar_mass_P : ℚ)
  (molar_mass_O : ℚ)
  : ℚ :=
  let molar_mass_NH4 := molar_mass_N + 4 * molar_mass_H
  let molar_mass_PO4 := molar_mass_P + 4 * molar_mass_O
  let molar_mass_NH4_3_PO4 := 3 * molar_mass_NH4 + molar_mass_PO4
  let mass_N_in_NH4_3_PO4 := 3 * molar_mass_N
  (mass_N_in_NH4_3_PO4 / molar_mass_NH4_3_PO4) * 100

theorem nitrogen_mass_percentage_in_ammonium_phosphate
  (molar_mass_N : ℚ := 14.01)
  (molar_mass_H : ℚ := 1.01)
  (molar_mass_P : ℚ := 30.97)
  (molar_mass_O : ℚ := 16.00)
  : nitrogen_mass_percentage molar_mass_N molar_mass_H molar_mass_P molar_mass_O = 28.19 :=
by
  sorry

end nitrogen_mass_percentage_in_ammonium_phosphate_l1306_130655


namespace conditions_for_unique_solution_l1306_130682

noncomputable def is_solution (n p x y z : ℕ) : Prop :=
x + p * y = n ∧ x + y = p^z

def unique_positive_integer_solution (n p : ℕ) : Prop :=
∃! (x y z : ℕ), 0 < x ∧ 0 < y ∧ 0 < z ∧ is_solution n p x y z

theorem conditions_for_unique_solution {n p : ℕ} :
  (1 < p) ∧ ((n - 1) % (p - 1) = 0) ∧ ∀ k : ℕ, n ≠ p^k ↔ unique_positive_integer_solution n p :=
sorry

end conditions_for_unique_solution_l1306_130682


namespace find_p_q_of_divisibility_l1306_130632

theorem find_p_q_of_divisibility 
  (p q : ℤ) 
  (h1 : (x + 3) ∣ (x^5 - 2*x^4 + 3*x^3 - p*x^2 + q*x - 6)) 
  (h2 : (x - 2) ∣ (x^5 - 2*x^4 + 3*x^3 - p*x^2 + q*x - 6)) 
  : p = -31 ∧ q = -71 :=
by
  sorry

end find_p_q_of_divisibility_l1306_130632


namespace quadratic_coeffs_l1306_130692

theorem quadratic_coeffs (x : ℝ) :
  (x - 1)^2 = 3 * x - 2 → ∃ b c, (x^2 + b * x + c = 0 ∧ b = -5 ∧ c = 3) :=
by
  sorry

end quadratic_coeffs_l1306_130692


namespace value_of_a_l1306_130687

/--
Given that x = 3 is a solution to the equation 3x - 2a = 5,
prove that a = 2.
-/
theorem value_of_a (x a : ℤ) (h : 3 * x - 2 * a = 5) (hx : x = 3) : a = 2 :=
by
  sorry

end value_of_a_l1306_130687


namespace necessary_but_not_sufficient_condition_l1306_130672

open Classical

theorem necessary_but_not_sufficient_condition (a b : ℝ) :
  (a > 1 ∧ b > 3) → (a + b > 4) ∧ ¬((a + b > 4) → (a > 1 ∧ b > 3)) :=
by
  sorry

end necessary_but_not_sufficient_condition_l1306_130672


namespace vasya_birthday_was_thursday_l1306_130686

variable (today : String)
variable (tomorrow : String)
variable (day_after_tomorrow : String)
variable (birthday : String)

-- Conditions given in the problem
axiom birthday_not_sunday : birthday ≠ "Sunday"
axiom sunday_day_after_tomorrow : day_after_tomorrow = "Sunday"

-- From the conditions, we need to prove that Vasya's birthday was on Thursday.
theorem vasya_birthday_was_thursday : birthday = "Thursday" :=
by
  -- Fill in the proof here
  sorry

end vasya_birthday_was_thursday_l1306_130686


namespace arithmetic_sequence_8th_term_l1306_130630

theorem arithmetic_sequence_8th_term 
    (a₁ : ℝ) (a₅ : ℝ) (n : ℕ) (a₈ : ℝ) 
    (h₁ : a₁ = 3) 
    (h₂ : a₅ = 78) 
    (h₃ : n = 25) : 
    a₈ = 24.875 := by
  sorry

end arithmetic_sequence_8th_term_l1306_130630


namespace coloring_problem_l1306_130604

theorem coloring_problem (a : ℕ → ℕ) (n t : ℕ) 
  (h1 : ∀ i j, i < j → a i < a j) 
  (h2 : ∀ x : ℤ, ∃ i, 0 < i ∧ i ≤ n ∧ ((x + a (i - 1)) % t) = 0) : 
  n ∣ t :=
by
  sorry

end coloring_problem_l1306_130604


namespace poly_div_simplification_l1306_130609

-- Assume a and b are real numbers.
variables (a b : ℝ)

-- Theorem to prove the equivalence
theorem poly_div_simplification (a b : ℝ) : (4 * a^2 - b^2) / (b - 2 * a) = -2 * a - b :=
by
  -- The proof will go here
  sorry

end poly_div_simplification_l1306_130609


namespace negate_universal_to_existential_l1306_130625

variable {f : ℝ → ℝ}

theorem negate_universal_to_existential :
  (¬ (∀ x1 x2 : ℝ, (f x2 - f x1) * (x2 - x1) ≥ 0)) ↔
  (∃ x1 x2 : ℝ, (f x2 - f x1) * (x2 - x1) < 0) :=
  sorry

end negate_universal_to_existential_l1306_130625


namespace find_f_inv_8_l1306_130678

variable (f : ℝ → ℝ)

-- Given conditions
axiom h1 : f 5 = 1
axiom h2 : ∀ x, f (2 * x) = 2 * f x

-- Theorem to prove
theorem find_f_inv_8 : f ⁻¹' {8} = {40} :=
by sorry

end find_f_inv_8_l1306_130678


namespace ducks_in_garden_l1306_130673

theorem ducks_in_garden (num_rabbits : ℕ) (num_ducks : ℕ) 
  (total_legs : ℕ)
  (rabbit_legs : ℕ) (duck_legs : ℕ) 
  (H1 : num_rabbits = 9)
  (H2 : rabbit_legs = 4)
  (H3 : duck_legs = 2)
  (H4 : total_legs = 48)
  (H5 : num_rabbits * rabbit_legs + num_ducks * duck_legs = total_legs) :
  num_ducks = 6 := 
by {
  sorry
}

end ducks_in_garden_l1306_130673


namespace stone_radius_l1306_130688

theorem stone_radius (hole_diameter hole_depth : ℝ) (r : ℝ) :
  hole_diameter = 30 → hole_depth = 10 → (r - 10)^2 + 15^2 = r^2 → r = 16.25 :=
by
  intros h_diam h_depth hyp_eq
  sorry

end stone_radius_l1306_130688


namespace fraction_to_terminating_decimal_l1306_130633

theorem fraction_to_terminating_decimal :
  (53 : ℚ)/160 = 0.33125 :=
by sorry

end fraction_to_terminating_decimal_l1306_130633


namespace annual_interest_rate_is_6_percent_l1306_130669

-- Definitions from the conditions
def principal : ℕ := 150
def total_amount_paid : ℕ := 159
def interest := total_amount_paid - principal
def interest_rate := (interest * 100) / principal

-- The theorem to prove
theorem annual_interest_rate_is_6_percent :
  interest_rate = 6 := by sorry

end annual_interest_rate_is_6_percent_l1306_130669


namespace AE_length_l1306_130605

theorem AE_length :
  ∀ (A B C D E : Type) 
    (AB CD AC BD AE EC : ℕ),
  AB = 12 → CD = 15 → AC = 18 → BD = 27 → 
  (AE + EC = AC) → 
  (AE * (18 - AE)) = (4 / 9 * 18 * 8) → 
  9 * AE = 72 → 
  AE = 8 := 
by
  intros A B C D E AB CD AC BD AE EC hAB hCD hAC hBD hSum hEqual hSolve
  sorry

end AE_length_l1306_130605


namespace hyperbola_center_coordinates_l1306_130654

-- Defining the equation of the hyperbola
def hyperbola_eq (x y : ℝ) : Prop :=
  (3 * y + 6)^2 / 16 - (2 * x - 1)^2 / 9 = 1

-- Stating the theorem to verify the center of the hyperbola
theorem hyperbola_center_coordinates :
  ∃ (h k : ℝ), (h = 1/2) ∧ (k = -2) ∧ 
    ∀ x y, hyperbola_eq x y ↔ ((y + 2)^2 / (4 / 3)^2 - (x - 1/2)^2 / (3 / 2)^2 = 1) :=
by sorry

end hyperbola_center_coordinates_l1306_130654


namespace k_range_l1306_130675

noncomputable def f (x : ℝ) (k : ℝ) : ℝ :=
  (Real.log x) - x - x * Real.exp (-x) - k

theorem k_range (k : ℝ) : (∀ x > 0, ∃ x > 0, f x k = 0) ↔ k ≤ -1 - (1 / Real.exp 1) :=
sorry

end k_range_l1306_130675


namespace problem_l1306_130646

-- Definitions for angles A, B, C and sides a, b, c of a triangle.
variables {A B C : ℝ} {a b c : ℝ}
-- Given condition
variables (h : a = b * Real.cos C + c * Real.sin B)

-- Triangle inequality and angle conditions
variables (ha : 0 < A) (hb : 0 < B) (hc : 0 < C)
variables (suma : A + B + C = Real.pi)

-- Goal: to prove that under the given condition, angle B is π/4
theorem problem : B = Real.pi / 4 :=
by {
  sorry
}

end problem_l1306_130646


namespace second_part_lent_years_l1306_130643

theorem second_part_lent_years 
  (P1 P2 T : ℝ)
  (h1 : P1 + P2 = 2743)
  (h2 : P2 = 1688)
  (h3 : P1 * 0.03 * 8 = P2 * 0.05 * T) 
  : T = 3 :=
sorry

end second_part_lent_years_l1306_130643


namespace jerry_cut_maple_trees_l1306_130608

theorem jerry_cut_maple_trees :
  (∀ pine maple walnut : ℕ, 
    pine = 8 * 80 ∧ 
    walnut = 4 * 100 ∧ 
    1220 = pine + walnut + maple * 60) → 
  maple = 3 := 
by 
  sorry

end jerry_cut_maple_trees_l1306_130608


namespace diff_quotient_remainder_n_75_l1306_130601

theorem diff_quotient_remainder_n_75 :
  ∃ n q r p : ℕ,  n = 75 ∧ n = 5 * q ∧ n = 34 * p + r ∧ q > r ∧ (q - r = 8) :=
by
  sorry

end diff_quotient_remainder_n_75_l1306_130601


namespace max_tiles_on_floor_l1306_130640

theorem max_tiles_on_floor
  (tile_w tile_h floor_w floor_h : ℕ)
  (h_tile_w : tile_w = 25)
  (h_tile_h : tile_h = 65)
  (h_floor_w : floor_w = 150)
  (h_floor_h : floor_h = 390) :
  max ((floor_h / tile_h) * (floor_w / tile_w))
      ((floor_h / tile_w) * (floor_w / tile_h)) = 36 :=
by
  -- Given conditions and calculations will be proved in the proof.
  sorry

end max_tiles_on_floor_l1306_130640


namespace ab_value_l1306_130617

theorem ab_value (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 29) : a * b = 10 := 
by 
  sorry

end ab_value_l1306_130617


namespace max_area_of_rectangle_l1306_130689

-- Question: Prove the largest possible area of a rectangle given the conditions
theorem max_area_of_rectangle :
  ∀ (x : ℝ), (2 * x + 2 * (x + 5) = 60) → x * (x + 5) ≤ 218.75 :=
by
  sorry

end max_area_of_rectangle_l1306_130689


namespace find_a_b_a_b_values_l1306_130635

/-
Define the matrix M as given in the problem.
Define the constants a and b, and state the condition that proves their correct values such that M_inv = a * M + b * I.
-/

open Matrix

noncomputable def M : Matrix (Fin 2) (Fin 2) ℚ :=
  !![2, 0;
     1, -3]

noncomputable def M_inv : Matrix (Fin 2) (Fin 2) ℚ :=
  !![1/2, 0;
     1/6, -1/3]

theorem find_a_b :
  ∃ (a b : ℚ), (M⁻¹) = a • M + b • (1 : Matrix (Fin 2) (Fin 2) ℚ) :=
sorry

theorem a_b_values :
  (∃ (a b : ℚ), (M⁻¹) = a • M + b • (1 : Matrix (Fin 2) (Fin 2) ℚ)) ∧
  (∃ a b : ℚ, a = 1/6 ∧ b = 1/6) :=
sorry

end find_a_b_a_b_values_l1306_130635


namespace max_second_smallest_l1306_130626

noncomputable def f (M : ℕ) : ℕ :=
  (M - 1) * (90 - M) * (89 - M) * (88 - M)

theorem max_second_smallest (M : ℕ) (cond : 1 ≤ M ∧ M ≤ 89) : M = 23 ↔ (∀ N : ℕ, f M ≥ f N) :=
by
  sorry

end max_second_smallest_l1306_130626


namespace find_ratio_of_b1_b2_l1306_130667

variable (a b k a1 a2 b1 b2 : ℝ)
variable (h1 : a1 ≠ 0) (h2 : a2 ≠ 0) (hb1 : b1 ≠ 0) (hb2 : b2 ≠ 0)

noncomputable def inversely_proportional_condition := a1 * b1 = a2 * b2
noncomputable def ratio_condition := a1 / a2 = 3 / 4
noncomputable def difference_condition := b1 - b2 = 5

theorem find_ratio_of_b1_b2 
  (h_inv : inversely_proportional_condition a1 a2 b1 b2)
  (h_rat : ratio_condition a1 a2)
  (h_diff : difference_condition b1 b2) :
  b1 / b2 = 4 / 3 :=
sorry

end find_ratio_of_b1_b2_l1306_130667


namespace expression_value_l1306_130694

noncomputable def givenExpression : ℝ :=
  -2^2 + Real.sqrt 8 - 3 + 1/3

theorem expression_value : givenExpression = -20/3 + 2 * Real.sqrt 2 := 
by
  sorry

end expression_value_l1306_130694


namespace avg_weight_b_c_l1306_130658

theorem avg_weight_b_c
  (a b c : ℝ)
  (h1 : (a + b + c) / 3 = 45)
  (h2 : (a + b) / 2 = 40)
  (h3 : b = 31) :
  (b + c) / 2 = 43 := 
by {
  sorry
}

end avg_weight_b_c_l1306_130658


namespace simplify_power_expression_l1306_130661

theorem simplify_power_expression (x : ℝ) : (3 * x^4)^5 = 243 * x^20 :=
by
  sorry

end simplify_power_expression_l1306_130661


namespace measure_of_angle_A_l1306_130696

-- Define the given conditions
variables (A B : ℝ)
axiom supplementary : A + B = 180
axiom measure_rel : A = 7 * B

-- The theorem statement to prove
theorem measure_of_angle_A : A = 157.5 :=
by
  -- proof steps would go here, but are omitted
  sorry

end measure_of_angle_A_l1306_130696


namespace find_fruit_juice_amount_l1306_130665

def total_punch : ℕ := 14 * 10
def mountain_dew : ℕ := 6 * 12
def ice : ℕ := 28
def fruit_juice : ℕ := total_punch - mountain_dew - ice

theorem find_fruit_juice_amount : fruit_juice = 40 := by
  sorry

end find_fruit_juice_amount_l1306_130665


namespace apartment_building_count_l1306_130634

theorem apartment_building_count 
  (floors_per_building : ℕ) 
  (apartments_per_floor : ℕ) 
  (doors_per_apartment : ℕ) 
  (total_doors_needed : ℕ) 
  (doors_per_building : ℕ) 
  (number_of_buildings : ℕ)
  (h1 : floors_per_building = 12)
  (h2 : apartments_per_floor = 6) 
  (h3 : doors_per_apartment = 7) 
  (h4 : total_doors_needed = 1008) 
  (h5 : doors_per_building = apartments_per_floor * doors_per_apartment * floors_per_building)
  (h6 : number_of_buildings = total_doors_needed / doors_per_building) : 
  number_of_buildings = 2 := 
by 
  rw [h1, h2, h3] at h5 
  rw [h5, h4] at h6 
  exact h6

end apartment_building_count_l1306_130634


namespace length_AC_correct_l1306_130614

noncomputable def length_AC (A B C D : Type) : ℝ := 105 / 17

variable {A B C D : Type}
variables (angle_BAC angle_ADB length_AD length_BC : ℝ)

theorem length_AC_correct
  (h1 : angle_BAC = 60)
  (h2 : angle_ADB = 30)
  (h3 : length_AD = 3)
  (h4 : length_BC = 9) :
  length_AC A B C D = 105 / 17 :=
sorry

end length_AC_correct_l1306_130614


namespace line_circle_no_intersection_l1306_130603

theorem line_circle_no_intersection : 
  (∀ x y : ℝ, 3 * x + 4 * y = 12 → (x - 1)^2 + (y + 1)^2 ≠ 1) :=
by
  sorry

end line_circle_no_intersection_l1306_130603


namespace find_n_l1306_130620

theorem find_n :
  ∃ (n : ℤ), (4 ≤ n ∧ n ≤ 8) ∧ (n % 5 = 2) ∧ (n = 7) :=
by
  sorry

end find_n_l1306_130620


namespace cone_volume_l1306_130652

theorem cone_volume (r l h V: ℝ) (h1: 15 * Real.pi = Real.pi * r^2 + Real.pi * r * l)
  (h2: 2 * Real.pi * r = (1 / 3) * Real.pi * l) :
  (V = (1 / 3) * Real.pi * r^2 * h) → h = Real.sqrt (l^2 - r^2) → l = 6 * r → r = Real.sqrt (15 / 7) → 
  V = (25 * Real.sqrt 3 / 7) * Real.pi :=
sorry

end cone_volume_l1306_130652


namespace retirement_savings_l1306_130681

/-- Define the initial deposit amount -/
def P : ℕ := 800000

/-- Define the annual interest rate as a rational number -/
def r : ℚ := 7/100

/-- Define the number of years the money is invested for -/
def t : ℕ := 15

/-- Simple interest formula to calculate the accumulated amount -/
noncomputable def A : ℚ := P * (1 + r * t)

theorem retirement_savings :
  A = 1640000 := 
by
  sorry

end retirement_savings_l1306_130681


namespace linear_function_does_not_pass_through_quadrant_3_l1306_130651

theorem linear_function_does_not_pass_through_quadrant_3
  (f : ℝ → ℝ) (h : ∀ x, f x = -3 * x + 5) :
  ¬ (∃ x y : ℝ, x < 0 ∧ y < 0 ∧ f x = y) :=
by
  sorry

end linear_function_does_not_pass_through_quadrant_3_l1306_130651


namespace rob_has_24_cards_l1306_130677

theorem rob_has_24_cards 
  (r : ℕ) -- total number of baseball cards Rob has
  (dr : ℕ) -- number of doubles Rob has
  (hj: dr = 1 / 3 * r) -- one third of Rob's cards are doubles
  (jess_doubles : ℕ) -- number of doubles Jess has
  (hj_mult : jess_doubles = 5 * dr) -- Jess has 5 times as many doubles as Rob
  (jess_doubles_40 : jess_doubles = 40) -- Jess has 40 doubles baseball cards
: r = 24 :=
by
  sorry

end rob_has_24_cards_l1306_130677


namespace height_of_right_triangle_l1306_130656

theorem height_of_right_triangle (a b c : ℝ) (h : ℝ) (h_right : a^2 + b^2 = c^2) (h_area : h = (a * b) / c) : h = (a * b) / c := 
by
  sorry

end height_of_right_triangle_l1306_130656


namespace find_general_term_a_l1306_130657

-- Define the sequence and conditions
noncomputable def S (n : ℕ) : ℚ :=
  if n = 0 then 0 else (n - 1) / (n * (n + 1))

-- General term to prove
def a (n : ℕ) : ℚ := 1 / (2^n) - 1 / (n * (n + 1))

theorem find_general_term_a :
  ∀ n : ℕ, n > 0 → S n + a n = (n - 1) / (n * (n + 1)) :=
by
  intro n hn
  sorry -- Proof omitted

end find_general_term_a_l1306_130657


namespace ways_to_climb_four_steps_l1306_130636

theorem ways_to_climb_four_steps (ways_to_climb : ℕ → ℕ) 
  (h1 : ways_to_climb 1 = 1) 
  (h2 : ways_to_climb 2 = 2) 
  (h3 : ways_to_climb 3 = 3) 
  (h_step : ∀ n, ways_to_climb n = ways_to_climb (n - 1) + ways_to_climb (n - 2)) : 
  ways_to_climb 4 = 5 := 
sorry

end ways_to_climb_four_steps_l1306_130636


namespace blue_tiles_in_45th_row_l1306_130697

theorem blue_tiles_in_45th_row :
  ∀ (n : ℕ), n = 45 → (∃ r b : ℕ, (r + b = 2 * n - 1) ∧ (r > b) ∧ (r - 1 = b)) → b = 44 :=
by
  -- Skipping the proof with sorry to adhere to instruction
  sorry

end blue_tiles_in_45th_row_l1306_130697


namespace fred_added_nine_l1306_130664

def onions_in_basket (initial_onions : ℕ) (added_by_sara : ℕ) (taken_by_sally : ℕ) (added_by_fred : ℕ) : ℕ :=
  initial_onions + added_by_sara - taken_by_sally + added_by_fred

theorem fred_added_nine : ∀ (S F : ℕ), onions_in_basket S 4 5 F = S + 8 → F = 9 :=
by
  intros S F h
  sorry

end fred_added_nine_l1306_130664


namespace arithmetic_sequence_50th_term_l1306_130699

theorem arithmetic_sequence_50th_term :
  let a_1 := 3
  let d := 5
  let n := 50
  let a_n := a_1 + (n - 1) * d
  a_n = 248 :=
by
  let a_1 := 3
  let d := 5
  let n := 50
  let a_n := a_1 + (n - 1) * d
  sorry

end arithmetic_sequence_50th_term_l1306_130699


namespace arithmetic_progression_implies_equality_l1306_130639

theorem arithmetic_progression_implies_equality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ((a + b) / 2) = ((Real.sqrt (a * b) + Real.sqrt ((a^2 + b^2) / 2)) / 2) → a = b :=
by
  sorry

end arithmetic_progression_implies_equality_l1306_130639


namespace find_d_l1306_130650

-- Define AP terms as S_n = a + (n-1)d, sum of first 10 terms, and difference expression
def arithmetic_progression (S : ℕ → ℕ) (a d : ℕ) : Prop :=
  ∀ n, S n = a + (n - 1) * d

def sum_first_ten (S : ℕ → ℕ) : Prop :=
  (S 1) + (S 2) + (S 3) + (S 4) + (S 5) + (S 6) + (S 7) + (S 8) + (S 9) + (S 10) = 55

def difference_expression (S : ℕ → ℕ) (d : ℕ) : Prop :=
  (S 10 - S 8) + (S 9 - S 7) + (S 8 - S 6) + (S 7 - S 5) + (S 6 - S 4) +
  (S 5 - S 3) + (S 4 - S 2) + (S 3 - S 1) = d

theorem find_d : ∃ (d : ℕ) (S : ℕ → ℕ) (a : ℕ), 
  (∀ n, S n = a + (n - 1) * d) ∧ 
  (S 1) + (S 2) + (S 3) + (S 4) + (S 5) + (S 6) + (S 7) + (S 8) + (S 9) + (S 10) = 55 ∧
  (S 10 - S 8) + (S 9 - S 7) + (S 8 - S 6) + (S 7 - S 5) + (S 6 - S 4) +
  (S 5 - S 3) + (S 4 - S 2) + (S 3 - S 1) = 16 :=
by
  sorry  -- proof is not required

end find_d_l1306_130650


namespace problem_statement_l1306_130642

-- Define the arithmetic sequence conditions
variables (a : ℕ → ℕ) (d : ℕ)
axiom h1 : a 1 = 2
axiom h2 : a 2018 = 2019
axiom arithmetic_seq : ∀ n, a (n + 1) = a n + d

-- Define the sum of the first n terms of the sequence
def sum_seq (n : ℕ) : ℕ := (n * a 1) + (n * (n-1) * d / 2)

theorem problem_statement : sum_seq a 5 + a 2014 = 2035 :=
by sorry

end problem_statement_l1306_130642


namespace greatest_possible_value_x_l1306_130611

theorem greatest_possible_value_x :
  ∀ x : ℚ, (∃ y : ℚ, y = (5 * x - 25) / (4 * x - 5) ∧ y^2 + y = 18) →
  x ≤ 55 / 29 :=
by sorry

end greatest_possible_value_x_l1306_130611


namespace pow_mul_eq_add_l1306_130685

variable (a : ℝ)

theorem pow_mul_eq_add : a^2 * a^3 = a^5 := 
by 
  sorry

end pow_mul_eq_add_l1306_130685


namespace roots_of_transformed_quadratic_l1306_130647

theorem roots_of_transformed_quadratic (a b p q s1 s2 : ℝ)
    (h_quad_eq : s1 ^ 2 + a * s1 + b = 0 ∧ s2 ^ 2 + a * s2 + b = 0)
    (h_sum_roots : s1 + s2 = -a)
    (h_prod_roots : s1 * s2 = b) :
        p = -(a ^ 4 - 4 * a ^ 2 * b + 2 * b ^ 2) ∧ 
        q = b ^ 4 :=
by
  sorry

end roots_of_transformed_quadratic_l1306_130647


namespace cost_per_serving_l1306_130615

-- Define the costs
def pasta_cost : ℝ := 1.00
def sauce_cost : ℝ := 2.00
def meatball_cost : ℝ := 5.00

-- Define the number of servings
def servings : ℝ := 8.0

-- State the theorem
theorem cost_per_serving : (pasta_cost + sauce_cost + meatball_cost) / servings = 1.00 :=
by
  sorry

end cost_per_serving_l1306_130615


namespace polynomial_behavior_l1306_130680

noncomputable def Q (x : ℝ) : ℝ := x^6 - 6 * x^5 + 10 * x^4 - x^3 - x + 12

theorem polynomial_behavior : 
  (∀ x : ℝ, x < 0 → Q x > 0) ∧ (∃ x : ℝ, x > 0 ∧ Q x = 0) := 
by 
  sorry

end polynomial_behavior_l1306_130680


namespace second_place_jump_l1306_130660

theorem second_place_jump : 
  ∀ (Kyungsoo Younghee Jinju Chanho : ℝ), 
    Kyungsoo = 2.3 → 
    Younghee = 0.9 → 
    Jinju = 1.8 → 
    Chanho = 2.5 → 
    ((Kyungsoo < Chanho) ∧ (Kyungsoo > Jinju) ∧ (Kyungsoo > Younghee)) :=
by 
  sorry

end second_place_jump_l1306_130660


namespace mike_bricks_l1306_130623

theorem mike_bricks (total_bricks bricks_A bricks_B bricks_other: ℕ) 
  (h1 : bricks_A = 40) 
  (h2 : bricks_B = bricks_A / 2)
  (h3 : total_bricks = 150) 
  (h4 : total_bricks = bricks_A + bricks_B + bricks_other) : bricks_other = 90 := 
by 
  sorry

end mike_bricks_l1306_130623


namespace sum_of_squares_inequality_l1306_130671

theorem sum_of_squares_inequality (a b c : ℝ) : a^2 + b^2 + c^2 ≥ (1/3)*(a + b + c)^2 := sorry

end sum_of_squares_inequality_l1306_130671


namespace candy_initial_amount_l1306_130602

namespace CandyProblem

variable (initial_candy given_candy left_candy : ℕ)

theorem candy_initial_amount (h1 : given_candy = 10) (h2 : left_candy = 68) (h3 : left_candy = initial_candy - given_candy) : initial_candy = 78 := 
  sorry
end CandyProblem

end candy_initial_amount_l1306_130602


namespace launch_country_is_soviet_union_l1306_130622

-- Definitions of conditions
def launch_date : String := "October 4, 1957"
def satellite_launched_on (date : String) : Prop := date = "October 4, 1957"
def choices : List String := ["A. United States", "B. Soviet Union", "C. European Union", "D. Germany"]

-- Problem statement
theorem launch_country_is_soviet_union : 
  satellite_launched_on launch_date → 
  "B. Soviet Union" ∈ choices := 
by
  sorry

end launch_country_is_soviet_union_l1306_130622


namespace calculate_total_cost_l1306_130684

noncomputable def sandwich_cost : ℕ := 4
noncomputable def soda_cost : ℕ := 3
noncomputable def num_sandwiches : ℕ := 7
noncomputable def num_sodas : ℕ := 8
noncomputable def total_cost : ℕ := sandwich_cost * num_sandwiches + soda_cost * num_sodas

theorem calculate_total_cost : total_cost = 52 := by
  sorry

end calculate_total_cost_l1306_130684


namespace led_message_count_l1306_130607

theorem led_message_count : 
  let n := 7
  let colors := 2
  let lit_leds := 3
  let non_adjacent_combinations := 10
  (non_adjacent_combinations * (colors ^ lit_leds)) = 80 :=
by
  sorry

end led_message_count_l1306_130607


namespace factor_expression_l1306_130695

theorem factor_expression (x y z : ℝ) :
  x^3 * (y^2 - z^2) - y^3 * (z^2 - x^2) + z^3 * (x^2 - y^2) =
  (x - y) * (y - z) * (z - x) * (x * y + z^2 - z * x) :=
by
  sorry

end factor_expression_l1306_130695


namespace find_tangent_c_l1306_130629

theorem find_tangent_c (c : ℝ) : 
  (∀ x y : ℝ, y = 3 * x + c ∧ y^2 = 12 * x → (-12)^2 - 4 * (1) * (12 * c) = 0) → c = 3 :=
sorry

end find_tangent_c_l1306_130629


namespace helen_chocolate_chip_cookies_l1306_130612

def number_of_raisin_cookies := 231
def difference := 25

theorem helen_chocolate_chip_cookies :
  ∃ C, C = number_of_raisin_cookies + difference ∧ C = 256 :=
by
  sorry -- Skipping the proof

end helen_chocolate_chip_cookies_l1306_130612


namespace number_of_blue_balls_l1306_130627

theorem number_of_blue_balls (b : ℕ) 
  (h1 : 0 < b ∧ b ≤ 15)
  (prob : (b / 15) * ((b - 1) / 14) = 1 / 21) :
  b = 5 := sorry

end number_of_blue_balls_l1306_130627


namespace students_with_same_grade_l1306_130644

theorem students_with_same_grade :
  let total_students := 40
  let students_with_same_A := 3
  let students_with_same_B := 2
  let students_with_same_C := 6
  let students_with_same_D := 1
  let total_same_grade_students := students_with_same_A + students_with_same_B + students_with_same_C + students_with_same_D
  total_same_grade_students = 12 →
  (total_same_grade_students / total_students) * 100 = 30 :=
by
  sorry

end students_with_same_grade_l1306_130644


namespace value_of_a_l1306_130653

theorem value_of_a {a : ℝ} (h : ∀ x y : ℝ, (a * x^2 + 2 * x + 1 = 0 ∧ a * y^2 + 2 * y + 1 = 0) → x = y) : a = 0 ∨ a = 1 := 
  sorry

end value_of_a_l1306_130653


namespace circle_trajectory_l1306_130610

theorem circle_trajectory (a b : ℝ) :
  ∃ x y : ℝ, (b - 3)^2 + a^2 = (b + 3)^2 → x^2 = 12 * y := 
sorry

end circle_trajectory_l1306_130610


namespace central_angle_of_sector_l1306_130649

theorem central_angle_of_sector 
  (r : ℝ) (s : ℝ) (c : ℝ)
  (h1 : r = 5)
  (h2 : s = 15)
  (h3 : c = 2 * π * r) :
  ∃ n : ℝ, (n * s * π / 180 = c) ∧ n = 120 :=
by
  use 120
  sorry

end central_angle_of_sector_l1306_130649


namespace altitude_correct_l1306_130600

-- Define the given sides and area of the triangle
def AB : ℝ := 30
def BC : ℝ := 17
def AC : ℝ := 25
def area_ABC : ℝ := 120

-- The length of the altitude from the vertex C to the base AB
def height_C_to_AB : ℝ := 8

-- Problem statement to be proven
theorem altitude_correct : (1 / 2) * AB * height_C_to_AB = area_ABC :=
by
  sorry

end altitude_correct_l1306_130600


namespace car_speed_l1306_130676

theorem car_speed 
  (d : ℝ) (t : ℝ) 
  (hd : d = 520) (ht : t = 8) : 
  d / t = 65 := 
by 
  sorry

end car_speed_l1306_130676


namespace find_m_prove_inequality_l1306_130690

-- Using noncomputable to handle real numbers where needed
noncomputable def f (x m : ℝ) := m - |x - 1|

-- First proof: Find m given conditions on f(x)
theorem find_m (m : ℝ) :
  (∀ x, f (x + 2) m + f (x - 2) m ≥ 0 ↔ -2 ≤ x ∧ x ≤ 4) → m = 3 :=
sorry

-- Second proof: Prove the inequality given m = 3
theorem prove_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (1 / a + 1 / (2 * b) + 1 / (3 * c) = 3) → a + 2 * b + 3 * c ≥ 3 :=
sorry

end find_m_prove_inequality_l1306_130690


namespace sum_eq_two_l1306_130663

theorem sum_eq_two (x y : ℝ) (h : x^2 + y^2 = 10 * x - 6 * y - 34) : x + y = 2 :=
by
  sorry

end sum_eq_two_l1306_130663


namespace lucy_found_shells_l1306_130645

theorem lucy_found_shells (original current : ℕ) (h1 : original = 68) (h2 : current = 89) : current - original = 21 :=
by {
    sorry
}

end lucy_found_shells_l1306_130645


namespace intersection_complement_A_B_subset_A_C_l1306_130641

-- Definition of sets A, B, and complements in terms of conditions
def setA : Set ℝ := { x | 3 ≤ x ∧ x < 7 }
def setB : Set ℝ := { x | 2 < x ∧ x < 10 }
def complement_A : Set ℝ := { x | x < 3 ∨ x ≥ 7 }

-- Proof Problem (1)
theorem intersection_complement_A_B :
  ((complement_A) ∩ setB) = { x | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10) } := 
  sorry

-- Definition of set C 
def setC (a : ℝ) : Set ℝ := { x | x < a }
-- Proof Problem (2)
theorem subset_A_C {a : ℝ} (h : setA ⊆ setC a) : a ≥ 7 :=
  sorry

end intersection_complement_A_B_subset_A_C_l1306_130641


namespace pow_mod_eq_l1306_130662

theorem pow_mod_eq :
  (13 ^ 7) % 11 = 7 :=
by
  sorry

end pow_mod_eq_l1306_130662


namespace jean_business_hours_l1306_130674

-- Definitions of the conditions
def weekday_hours : ℕ := 10 - 16 -- from 4 pm to 10 pm
def weekend_hours : ℕ := 10 - 18 -- from 6 pm to 10 pm
def weekdays : ℕ := 5 -- Monday through Friday
def weekends : ℕ := 2 -- Saturday and Sunday

-- Total weekly hours
def total_weekly_hours : ℕ :=
  (weekday_hours * weekdays) + (weekend_hours * weekends)

-- Proof statement
theorem jean_business_hours : total_weekly_hours = 38 :=
by
  sorry

end jean_business_hours_l1306_130674


namespace unattainable_y_l1306_130606

theorem unattainable_y (x : ℝ) (h : 4 * x + 5 ≠ 0) : 
  (y = (3 - x) / (4 * x + 5)) → (y ≠ -1/4) :=
sorry

end unattainable_y_l1306_130606


namespace Jesse_remaining_money_l1306_130698

-- Define the conditions
def initial_money := 50
def novel_cost := 7
def lunch_cost := 2 * novel_cost
def total_spent := novel_cost + lunch_cost

-- Define the remaining money after spending
def remaining_money := initial_money - total_spent

-- Prove that the remaining money is $29
theorem Jesse_remaining_money : remaining_money = 29 := 
by
  sorry

end Jesse_remaining_money_l1306_130698


namespace mul_99_105_l1306_130624

theorem mul_99_105 : 99 * 105 = 10395 := 
by
  -- Annotations and imports are handled; only the final Lean statement provided as requested.
  sorry

end mul_99_105_l1306_130624


namespace ratio_length_breadth_l1306_130628

-- Define the conditions
def length := 135
def area := 6075

-- Define the breadth in terms of the area and length
def breadth := area / length

-- The problem statement as a Lean 4 theorem to prove the ratio
theorem ratio_length_breadth : length / breadth = 3 := 
by
  -- Proof goes here
  sorry

end ratio_length_breadth_l1306_130628


namespace dacid_average_marks_is_75_l1306_130621

/-- Defining the marks obtained in each subject as constants -/
def english_marks : ℕ := 76
def mathematics_marks : ℕ := 65
def physics_marks : ℕ := 82
def chemistry_marks : ℕ := 67
def biology_marks : ℕ := 85

/-- Total marks calculation -/
def total_marks : ℕ :=
  english_marks + mathematics_marks + physics_marks + chemistry_marks + biology_marks

/-- Number of subjects -/
def number_of_subjects : ℕ := 5

/-- Average marks calculation -/
def average_marks : ℕ :=
  total_marks / number_of_subjects

/-- Theorem proving that Dacid's average marks is 75 -/
theorem dacid_average_marks_is_75 : average_marks = 75 :=
  sorry

end dacid_average_marks_is_75_l1306_130621
