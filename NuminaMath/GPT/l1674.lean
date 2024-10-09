import Mathlib

namespace hours_per_day_l1674_167494

theorem hours_per_day 
  (H : ℕ)
  (h1 : 6 * 8 * H = 48 * H)
  (h2 : 4 * 3 * 8 = 96)
  (h3 : (48 * H) / 75 = 96 / 30) : 
  H = 5 :=
by
  sorry

end hours_per_day_l1674_167494


namespace sum_of_fractions_is_correct_l1674_167499

-- Definitions from the conditions
def half_of_third := (1 : ℚ) / 2 * (1 : ℚ) / 3
def third_of_quarter := (1 : ℚ) / 3 * (1 : ℚ) / 4
def quarter_of_fifth := (1 : ℚ) / 4 * (1 : ℚ) / 5
def sum_fractions := half_of_third + third_of_quarter + quarter_of_fifth

-- The theorem to prove
theorem sum_of_fractions_is_correct : sum_fractions = (3 : ℚ) / 10 := by
  sorry

end sum_of_fractions_is_correct_l1674_167499


namespace literature_books_cost_more_l1674_167496

theorem literature_books_cost_more :
  let num_books := 45
  let literature_cost_per_book := 7
  let technology_cost_per_book := 5
  (num_books * literature_cost_per_book) - (num_books * technology_cost_per_book) = 90 :=
by
  sorry

end literature_books_cost_more_l1674_167496


namespace sums_correct_l1674_167486

theorem sums_correct (x : ℕ) (h : x + 2 * x = 48) : x = 16 :=
by
  sorry

end sums_correct_l1674_167486


namespace f_13_eq_223_l1674_167492

def f (n : ℕ) : ℕ := n^2 + n + 41

theorem f_13_eq_223 : f 13 = 223 :=
by
  sorry

end f_13_eq_223_l1674_167492


namespace sum_put_at_simple_interest_l1674_167411

theorem sum_put_at_simple_interest (P R : ℝ) 
  (h : ((P * (R + 3) * 2) / 100) - ((P * R * 2) / 100) = 300) : 
  P = 5000 :=
by
  sorry

end sum_put_at_simple_interest_l1674_167411


namespace quadratic_equation_is_D_l1674_167460

theorem quadratic_equation_is_D (x a b c : ℝ) : 
  (¬ (∃ b' : ℝ, (x^2 - 2) * x = b' * x + 2)) ∧
  (¬ ((a ≠ 0) ∧ (ax^2 + bx + c = 0))) ∧
  (¬ (x + (1 / x) = 5)) ∧
  ((x^2 = 0) ↔ true) :=
by sorry

end quadratic_equation_is_D_l1674_167460


namespace pheromone_effect_on_population_l1674_167432

-- Definitions of conditions
def disrupt_sex_ratio (uses_pheromones : Bool) : Bool :=
  uses_pheromones = true

def decrease_birth_rate (disrupt_sex_ratio : Bool) : Bool :=
  disrupt_sex_ratio = true

def decrease_population_density (decrease_birth_rate : Bool) : Bool :=
  decrease_birth_rate = true

-- Problem Statement for Lean 4
theorem pheromone_effect_on_population (uses_pheromones : Bool) :
  disrupt_sex_ratio uses_pheromones = true →
  decrease_birth_rate (disrupt_sex_ratio uses_pheromones) = true →
  decrease_population_density (decrease_birth_rate (disrupt_sex_ratio uses_pheromones)) = true :=
sorry

end pheromone_effect_on_population_l1674_167432


namespace solve_quintic_equation_l1674_167493

theorem solve_quintic_equation :
  {x : ℝ | x * (x - 3)^2 * (5 + x) * (x^2 - 1) = 0} = {0, 3, -5, 1, -1} :=
by
  sorry

end solve_quintic_equation_l1674_167493


namespace powerFunctionAtPoint_l1674_167409

def powerFunction (n : ℕ) (x : ℕ) : ℕ := x ^ n

theorem powerFunctionAtPoint (n : ℕ) (h : powerFunction n 2 = 8) : powerFunction n 3 = 27 :=
  by {
    sorry
}

end powerFunctionAtPoint_l1674_167409


namespace gum_left_after_sharing_l1674_167419

-- Define the initial state of Adrianna's gum and the changes to it
def initial_gum : Nat := 10
def additional_gum : Nat := 3
def given_out_gum : Nat := 11

-- Define the final state of Adrianna's gum
def final_gum : Nat := initial_gum + additional_gum - given_out_gum

-- Prove that Adrianna ends up with 2 pieces of gum under the given conditions
theorem gum_left_after_sharing :
  final_gum = 2 :=
by 
  -- Since this is just the statement and not the proof, we end with sorry.
  sorry

end gum_left_after_sharing_l1674_167419


namespace range_f3_l1674_167423

def function_f (a c x : ℝ) : ℝ := a * x^2 - c

theorem range_f3 (a c : ℝ) :
  (-4 ≤ function_f a c 1) ∧ (function_f a c 1 ≤ -1) →
  (-1 ≤ function_f a c 2) ∧ (function_f a c 2 ≤ 5) →
  -12 ≤ function_f a c 3 ∧ function_f a c 3 ≤ 1.75 :=
by
  sorry

end range_f3_l1674_167423


namespace positive_integer_solutions_count_3x_plus_4y_eq_1024_l1674_167462

theorem positive_integer_solutions_count_3x_plus_4y_eq_1024 :
  (∃ x y : ℕ, x > 0 ∧ y > 0 ∧ 3 * x + 4 * y = 1024) ∧ 
  (∀ n, n = 85 → ∃! (s : ℕ × ℕ), s.fst > 0 ∧ s.snd > 0 ∧ 3 * s.fst + 4 * s.snd = 1024 ∧ n = 85) := 
sorry

end positive_integer_solutions_count_3x_plus_4y_eq_1024_l1674_167462


namespace hcf_two_numbers_l1674_167415

theorem hcf_two_numbers
  (x y : ℕ) 
  (h_lcm : Nat.lcm x y = 560)
  (h_prod : x * y = 42000) : Nat.gcd x y = 75 :=
by
  sorry

end hcf_two_numbers_l1674_167415


namespace average_last_three_l1674_167442

theorem average_last_three {a b c d e f g : ℝ} 
  (h_avg_all : (a + b + c + d + e + f + g) / 7 = 60)
  (h_avg_first_four : (a + b + c + d) / 4 = 55) : 
  (e + f + g) / 3 = 200 / 3 :=
by
  sorry

end average_last_three_l1674_167442


namespace Matt_income_from_plantation_l1674_167467

noncomputable def plantation_income :=
  let plantation_area := 500 * 500  -- square feet
  let grams_peanuts_per_sq_ft := 50 -- grams
  let grams_peanut_butter_per_20g_peanuts := 5  -- grams
  let price_per_kg_peanut_butter := 10 -- $

  -- Total revenue calculation
  plantation_area * grams_peanuts_per_sq_ft * grams_peanut_butter_per_20g_peanuts /
  20 / 1000 * price_per_kg_peanut_butter

theorem Matt_income_from_plantation :
  plantation_income = 31250 := sorry

end Matt_income_from_plantation_l1674_167467


namespace negation_of_prop_l1674_167412

theorem negation_of_prop :
  (¬ ∀ x : ℝ, x^2 - x + 2 ≥ 0) ↔ (∃ x : ℝ, x^2 - x + 2 < 0) :=
sorry

end negation_of_prop_l1674_167412


namespace sqrt_mixed_number_simplified_l1674_167427

theorem sqrt_mixed_number_simplified : 
  (Real.sqrt (12 + 1 / 9) = Real.sqrt 109 / 3) := by
  sorry

end sqrt_mixed_number_simplified_l1674_167427


namespace jonessa_take_home_pay_l1674_167458

noncomputable def tax_rate : ℝ := 0.10
noncomputable def pay : ℝ := 500
noncomputable def tax_amount : ℝ := pay * tax_rate
noncomputable def take_home_pay : ℝ := pay - tax_amount

theorem jonessa_take_home_pay : take_home_pay = 450 := by
  have h1 : tax_amount = 50 := by
    sorry
  have h2 : take_home_pay = 450 := by
    sorry
  exact h2

end jonessa_take_home_pay_l1674_167458


namespace sum_not_fourteen_l1674_167470

theorem sum_not_fourteen (a b c d : ℕ) (ha : 1 ≤ a ∧ a ≤ 6) (hb : 1 ≤ b ∧ b ≤ 6) (hc : 1 ≤ c ∧ c ≤ 6) (hd : 1 ≤ d ∧ d ≤ 6) 
  (hprod : a * b * c * d = 120) : a + b + c + d ≠ 14 :=
sorry

end sum_not_fourteen_l1674_167470


namespace cylinder_radius_l1674_167426

theorem cylinder_radius (h r: ℝ) (S: ℝ) (S_eq: S = 130 * Real.pi) (h_eq: h = 8) 
    (surface_area_eq: S = 2 * Real.pi * r^2 + 2 * Real.pi * r * h) : 
    r = 5 :=
by {
  -- Placeholder for proof steps.
  sorry
}

end cylinder_radius_l1674_167426


namespace marble_solid_color_percentage_l1674_167400

theorem marble_solid_color_percentage (a b : ℕ) (h1 : a = 5) (h2 : b = 85) : a + b = 90 := 
by
  sorry

end marble_solid_color_percentage_l1674_167400


namespace find_angle_B_l1674_167454

theorem find_angle_B 
  (A B C : ℝ)
  (h1 : B = A + 10)
  (h2 : C = B + 10)
  (h3 : A + B + C = 180) :
  B = 60 :=
sorry

end find_angle_B_l1674_167454


namespace jessies_original_weight_l1674_167478

theorem jessies_original_weight (current_weight weight_lost original_weight : ℕ) 
  (h_current: current_weight = 27) (h_lost: weight_lost = 101) 
  (h_original: original_weight = current_weight + weight_lost) : 
  original_weight = 128 :=
by
  rw [h_current, h_lost] at h_original
  exact h_original

end jessies_original_weight_l1674_167478


namespace sun_salutations_per_year_l1674_167434

theorem sun_salutations_per_year :
  (∀ S : Nat, S = 5) ∧
  (∀ W : Nat, W = 5) ∧
  (∀ Y : Nat, Y = 52) →
  ∃ T : Nat, T = 1300 :=
by 
  sorry

end sun_salutations_per_year_l1674_167434


namespace first_year_after_2020_with_sum_15_l1674_167453

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem first_year_after_2020_with_sum_15 :
  ∀ n, n > 2020 → (sum_of_digits n = 15 ↔ n = 2058) := by
  sorry

end first_year_after_2020_with_sum_15_l1674_167453


namespace union_A_B_equiv_l1674_167401

def A : Set ℝ := {x : ℝ | x > 2}
def B : Set ℝ := {x : ℝ | 1 ≤ x ∧ x ≤ 3}

theorem union_A_B_equiv : A ∪ B = {x : ℝ | x ≥ 1} :=
by
  sorry

end union_A_B_equiv_l1674_167401


namespace grilled_cheese_sandwiches_l1674_167482

-- Define the number of ham sandwiches Joan makes
def ham_sandwiches := 8

-- Define the cheese requirements for each type of sandwich
def cheddar_for_ham := 1
def swiss_for_ham := 1
def cheddar_for_grilled := 2
def gouda_for_grilled := 1

-- Total cheese used
def total_cheddar := 40
def total_swiss := 20
def total_gouda := 30

-- Prove the number of grilled cheese sandwiches Joan makes
theorem grilled_cheese_sandwiches (ham_sandwiches : ℕ) (cheddar_for_ham : ℕ) (swiss_for_ham : ℕ)
                                  (cheddar_for_grilled : ℕ) (gouda_for_grilled : ℕ)
                                  (total_cheddar : ℕ) (total_swiss : ℕ) (total_gouda : ℕ) :
    (total_cheddar - ham_sandwiches * cheddar_for_ham) / cheddar_for_grilled = 16 :=
by
  sorry

end grilled_cheese_sandwiches_l1674_167482


namespace base_7_minus_base_8_l1674_167472

def convert_base_7 (n : ℕ) : ℕ :=
  match n with
  | 543210 => 5 * 7^5 + 4 * 7^4 + 3 * 7^3 + 2 * 7^2 + 1 * 7^1 + 0 * 7^0
  | _ => 0

def convert_base_8 (n : ℕ) : ℕ :=
  match n with
  | 45321 => 4 * 8^4 + 5 * 8^3 + 3 * 8^2 + 2 * 8^1 + 1 * 8^0
  | _ => 0

theorem base_7_minus_base_8 : convert_base_7 543210 - convert_base_8 45321 = 75620 := by
  sorry

end base_7_minus_base_8_l1674_167472


namespace Tyler_cucumbers_and_grapes_l1674_167408

theorem Tyler_cucumbers_and_grapes (a b c g : ℝ) (h1 : 10 * a = 5 * b) (h2 : 3 * b = 4 * c) (h3 : 4 * c = 6 * g) :
  (20 * a = (40 / 3) * c) ∧ (20 * a = 20 * g) :=
by
  sorry

end Tyler_cucumbers_and_grapes_l1674_167408


namespace parallel_vectors_l1674_167447

variable (x : ℝ)
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (x, -2)

theorem parallel_vectors (h : (1 * (-2) - 2 * x = 0)) : x = -1 :=
by
  sorry

end parallel_vectors_l1674_167447


namespace Jimmy_earns_229_l1674_167438

-- Definitions based on conditions from the problem
def number_of_type_A : ℕ := 5
def number_of_type_B : ℕ := 4
def number_of_type_C : ℕ := 3

def value_of_type_A : ℕ := 20
def value_of_type_B : ℕ := 30
def value_of_type_C : ℕ := 40

def discount_type_A : ℕ := 7
def discount_type_B : ℕ := 10
def discount_type_C : ℕ := 12

-- Calculation of the total amount Jimmy will earn
def total_earnings : ℕ :=
  let price_A := value_of_type_A - discount_type_A
  let price_B := value_of_type_B - discount_type_B
  let price_C := value_of_type_C - discount_type_C
  (number_of_type_A * price_A) +
  (number_of_type_B * price_B) +
  (number_of_type_C * price_C)

-- The statement to be proved
theorem Jimmy_earns_229 : total_earnings = 229 :=
by
  -- Proof omitted
  sorry

end Jimmy_earns_229_l1674_167438


namespace area_of_plot_is_correct_l1674_167487

-- Define the side length of the square plot
def side_length : ℝ := 50.5

-- Define the area of the square plot
def area_of_square (s : ℝ) : ℝ := s * s

-- Theorem stating that the area of a square plot with side length 50.5 m is 2550.25 m²
theorem area_of_plot_is_correct : area_of_square side_length = 2550.25 := by
  sorry

end area_of_plot_is_correct_l1674_167487


namespace complement_of_A_in_U_is_4_l1674_167417

-- Define the universal set U
def U : Set ℕ := { x | 1 < x ∧ x < 5 }

-- Define the set A
def A : Set ℕ := {2, 3}

-- Define the complement of A in U
def complement_U_of_A : Set ℕ := { x ∈ U | x ∉ A }

-- State the theorem
theorem complement_of_A_in_U_is_4 : complement_U_of_A = {4} :=
by
  sorry

end complement_of_A_in_U_is_4_l1674_167417


namespace number_of_integers_having_squares_less_than_10_million_l1674_167451

theorem number_of_integers_having_squares_less_than_10_million : 
  ∃ n : ℕ, (n = 3162) ∧ (∀ k : ℕ, k ≤ 3162 → (k^2 < 10^7)) :=
by 
  sorry

end number_of_integers_having_squares_less_than_10_million_l1674_167451


namespace equal_numbers_product_l1674_167480

theorem equal_numbers_product :
  ∀ (a b c d : ℕ), 
  (a + b + c + d = 80) → 
  (a = 12) → 
  (b = 22) → 
  (c = d) → 
  (c * d = 529) :=
by
  intros a b c d hsum ha hb hcd
  -- proof skipped
  sorry

end equal_numbers_product_l1674_167480


namespace parabolic_arch_properties_l1674_167471

noncomputable def parabolic_arch_height (x : ℝ) : ℝ :=
  let a : ℝ := -4 / 125
  let k : ℝ := 20
  a * x^2 + k

theorem parabolic_arch_properties :
  (parabolic_arch_height 10 = 16.8) ∧ (parabolic_arch_height 10 = parabolic_arch_height 10 → (10 = 10 ∨ 10 = -10)) :=
by
  have h1 : parabolic_arch_height 10 = 16.8 :=
    sorry
  have h2 : parabolic_arch_height 10 = parabolic_arch_height 10 → (10 = 10 ∨ 10 = -10) :=
    sorry
  exact ⟨h1, h2⟩

end parabolic_arch_properties_l1674_167471


namespace int_n_satisfying_conditions_l1674_167469

theorem int_n_satisfying_conditions : 
  (∃! (n : ℤ), ∃ (k : ℤ), (n + 3 = k^2 * (23 - n)) ∧ n ≠ 23) :=
by
  use 2
  -- Provide a proof for this statement here
  sorry

end int_n_satisfying_conditions_l1674_167469


namespace find_y_l1674_167424

theorem find_y (x y : ℝ) (h1 : x - y = 10) (h2 : x + y = 18) : y = 4 := 
by 
  sorry

end find_y_l1674_167424


namespace age_calculation_l1674_167449

/-- Let Thomas be a 6-year-old child, Shay be 13 years older than Thomas, 
and also 5 years younger than James. Let Violet be 3 years younger than 
Thomas, and Emily be the same age as Shay. This theorem proves that when 
Violet reaches the age of Thomas (6 years old), James will be 27 years old 
and Emily will be 22 years old. -/
theorem age_calculation : 
  ∀ (Thomas Shay James Violet Emily : ℕ),
    Thomas = 6 →
    Shay = Thomas + 13 →
    James = Shay + 5 →
    Violet = Thomas - 3 →
    Emily = Shay →
    (Violet + (6 - Violet) = 6) →
    (James + (6 - Violet) = 27 ∧ Emily + (6 - Violet) = 22) :=
by
  intros Thomas Shay James Violet Emily ht hs hj hv he hv_diff
  sorry

end age_calculation_l1674_167449


namespace total_drink_volume_l1674_167431

-- Define the percentages of the various juices
def grapefruit_percentage : ℝ := 0.20
def lemon_percentage : ℝ := 0.25
def pineapple_percentage : ℝ := 0.10
def mango_percentage : ℝ := 0.15

-- Define the volume of orange juice in ounces
def orange_juice_volume : ℝ := 24

-- State the total percentage of all juices other than orange juice
def non_orange_percentage : ℝ := grapefruit_percentage + lemon_percentage + pineapple_percentage + mango_percentage

-- Calculate the percentage of orange juice
def orange_percentage : ℝ := 1 - non_orange_percentage

-- State that the total volume of the drink is such that 30% of it is 24 ounces
theorem total_drink_volume : ∃ (total_volume : ℝ), (orange_percentage * total_volume = orange_juice_volume) ∧ (total_volume = 80) := by
  use 80
  sorry

end total_drink_volume_l1674_167431


namespace smallest_n_terminating_decimal_l1674_167402

theorem smallest_n_terminating_decimal : ∃ n : ℕ, (∀ m : ℕ, m < n → (∀ k : ℕ, (n = 103 + k) → (∃ a b : ℕ, k = 2^a * 5^b)) → (k ≠ 0 → k = 125)) ∧ n = 22 := 
sorry

end smallest_n_terminating_decimal_l1674_167402


namespace hyperbola_asymptote_l1674_167481

theorem hyperbola_asymptote (a : ℝ) (h₀ : a > 0) 
  (h₁ : ∃ (x y : ℝ), (x, y) = (2, 1) ∧ 
       (y = (2 / a) * x ∨ y = -(2 / a) * x)) : a = 4 := by
  sorry

end hyperbola_asymptote_l1674_167481


namespace possible_digits_C_multiple_of_5_l1674_167410

theorem possible_digits_C_multiple_of_5 :
    ∃ (digits : Finset ℕ), (∀ x ∈ digits, x < 10) ∧ digits.card = 10 ∧ (∀ C ∈ digits, ∃ n : ℕ, n = 1000 + C * 100 + 35 ∧ n % 5 = 0) :=
by {
  sorry
}

end possible_digits_C_multiple_of_5_l1674_167410


namespace cubic_polynomial_p_value_l1674_167463

noncomputable def p (x : ℝ) : ℝ := sorry

theorem cubic_polynomial_p_value :
  (∀ n ∈ ({1, 2, 3, 5} : Finset ℝ), p n = 1 / n ^ 2) →
  p 4 = 1 / 150 := 
by
  intros h
  sorry

end cubic_polynomial_p_value_l1674_167463


namespace find_number_l1674_167440

theorem find_number (x : ℝ) (h : x - (3 / 5) * x = 58) : x = 145 := by
  sorry

end find_number_l1674_167440


namespace allocation_of_fabric_l1674_167497

theorem allocation_of_fabric (x : ℝ) (y : ℝ) 
  (fabric_for_top : 3 * x = 2 * x)
  (fabric_for_pants : 3 * y = 3 * (600 - x))
  (total_fabric : x + y = 600)
  (sets_match : (x / 3) * 2 = (y / 3) * 3) : 
  x = 360 ∧ y = 240 := 
by
  sorry

end allocation_of_fabric_l1674_167497


namespace find_b_l1674_167414

theorem find_b (b : ℤ) (h1 : ∀ (a b : ℤ), a * b = (a - 1) * (b - 1)) (h2 : 21 * b = 160) : b = 9 := by
  sorry

end find_b_l1674_167414


namespace correct_inequality_l1674_167455

def a : ℚ := -4 / 5
def b : ℚ := -3 / 4

theorem correct_inequality : a < b := 
by {
  -- Proof here
  sorry
}

end correct_inequality_l1674_167455


namespace negation_proposition_equivalence_l1674_167461

theorem negation_proposition_equivalence : 
  (¬ ∃ x : ℝ, x^2 - 2 * x + 1 < 0) ↔ (∀ x : ℝ, x^2 - 2 * x + 1 ≥ 0) :=
by
  sorry

end negation_proposition_equivalence_l1674_167461


namespace cubics_sum_l1674_167464

theorem cubics_sum (x y : ℝ) (h1 : x + y = 10) (h2 : x * y = 12) : x^3 + y^3 = 640 :=
by
  sorry

end cubics_sum_l1674_167464


namespace minimal_rotations_triangle_l1674_167477

/-- Given a triangle with angles α, β, γ at vertices 1, 2, 3 respectively.
    The triangle returns to its original position after 15 rotations around vertex 1 by α,
    and after 6 rotations around vertex 2 by β.
    We need to show that the minimal positive integer n such that the triangle returns
    to its original position after n rotations around vertex 3 by γ is 5. -/
theorem minimal_rotations_triangle :
  ∃ (α β γ : ℝ) (k m l n : ℤ), 
    (15 * α = 360 * k) ∧ 
    (6 * β = 360 * m) ∧ 
    (α + β + γ = 180) ∧ 
    (n * γ = 360 * l) ∧ 
    (∀ n' : ℤ, n' > 0 → (∃ k' m' l' : ℤ, 
      (15 * α = 360 * k') ∧ 
      (6 * β = 360 * m') ∧ 
      (α + β + γ = 180) ∧ 
      (n' * γ = 360 * l') → n <= n')) ∧ 
    n = 5 := by
  sorry

end minimal_rotations_triangle_l1674_167477


namespace sign_of_ac_l1674_167473

theorem sign_of_ac (a b c d : ℝ) (hb : b ≠ 0) (hd : d ≠ 0) (h : (a / b) + (c / d) = (a + c) / (b + d)) : a * c < 0 :=
by
  sorry

end sign_of_ac_l1674_167473


namespace range_of_b_div_a_l1674_167465

theorem range_of_b_div_a 
  (a b : ℝ)
  (h1 : 0 < a) 
  (h2 : a ≤ 2)
  (h3 : b ≥ 1)
  (h4 : b ≤ a^2) : 
  (1 / 2) ≤ b / a ∧ b / a ≤ 2 := 
sorry

end range_of_b_div_a_l1674_167465


namespace find_jamals_grade_l1674_167489

noncomputable def jamals_grade (n_students : ℕ) (absent_students : ℕ) (test_avg_28_students : ℕ) (new_total_avg_30_students : ℕ) (taqeesha_score : ℕ) : ℕ :=
  let total_28_students := 28 * test_avg_28_students
  let total_30_students := 30 * new_total_avg_30_students
  let combined_score := total_30_students - total_28_students
  combined_score - taqeesha_score

theorem find_jamals_grade :
  jamals_grade 30 2 85 86 92 = 108 :=
by
  sorry

end find_jamals_grade_l1674_167489


namespace round_to_nearest_tenth_l1674_167407

theorem round_to_nearest_tenth : 
  let x := 36.89753 
  let tenth_place := 8
  let hundredth_place := 9
  (hundredth_place > 5) → (Float.round (10 * x) / 10 = 36.9) := 
by
  intros x tenth_place hundredth_place h
  sorry

end round_to_nearest_tenth_l1674_167407


namespace center_of_circle_l1674_167406

-- Define the equation of the circle
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 2*y - 2 = 0

-- Define the condition for the center of the circle
def is_center_of_circle (a b : ℝ) : Prop :=
  ∀ x y : ℝ, circle_equation x y ↔ (x - a)^2 + (y - b)^2 = 4

-- The main theorem to be proved
theorem center_of_circle : is_center_of_circle 1 (-1) :=
by
  sorry

end center_of_circle_l1674_167406


namespace bc_sum_condition_l1674_167490

-- Define the conditions as Lean definitions
def is_positive_integer (n : ℕ) : Prop := n > 0
def not_equal_to (x y : ℕ) : Prop := x ≠ y
def less_than_or_equal_to_nine (n : ℕ) : Prop := n ≤ 9

-- Main proof statement
theorem bc_sum_condition (a b c : ℕ) (h_pos_a : is_positive_integer a) (h_pos_b : is_positive_integer b) (h_pos_c : is_positive_integer c)
  (h_a_not_1 : a ≠ 1) (h_b_not_c : b ≠ c) (h_b_le_9 : less_than_or_equal_to_nine b) (h_c_le_9 : less_than_or_equal_to_nine c)
  (h_eq : (10 * a + b) * (10 * a + c) = 100 * a * a + 110 * a + b * c) :
  b + c = 11 := by
  sorry

end bc_sum_condition_l1674_167490


namespace ordered_pair_represents_5_1_l1674_167422

structure OrderedPair (α : Type) :=
  (fst : α)
  (snd : α)

def represents_rows_cols (pair : OrderedPair ℝ) (rows cols : ℕ) : Prop :=
  pair.fst = rows ∧ pair.snd = cols

theorem ordered_pair_represents_5_1 :
  represents_rows_cols (OrderedPair.mk 2 3) 2 3 →
  represents_rows_cols (OrderedPair.mk 5 1) 5 1 :=
by
  intros h
  sorry

end ordered_pair_represents_5_1_l1674_167422


namespace exists_indices_non_decreasing_l1674_167433

theorem exists_indices_non_decreasing
    (a b c : ℕ → ℕ) :
    ∃ p q : ℕ, p ≠ q ∧ a p ≥ a q ∧ b p ≥ b q ∧ c p ≥ c q :=
  sorry

end exists_indices_non_decreasing_l1674_167433


namespace count_three_letter_sets_l1674_167476

-- Define the set of letters
def letters := Finset.range 10  -- representing letters A (0) to J (9)

-- Define the condition that J (represented by 9) cannot be the first initial
def valid_first_initials := letters.erase 9  -- remove 9 (J) from 0 to 9

-- Calculate the number of valid three-letter sets of initials
theorem count_three_letter_sets : 
  let first_initials := valid_first_initials
  let second_initials := letters
  let third_initials := letters
  first_initials.card * second_initials.card * third_initials.card = 900 := by
  sorry

end count_three_letter_sets_l1674_167476


namespace cylindrical_coordinates_cone_shape_l1674_167468

def cylindrical_coordinates := Type

def shape_description (r θ z : ℝ) : Prop :=
θ = 2 * z

theorem cylindrical_coordinates_cone_shape (r θ z : ℝ) :
  shape_description r θ z → θ = 2 * z → Prop := sorry

end cylindrical_coordinates_cone_shape_l1674_167468


namespace percent_decrease_is_80_l1674_167404

-- Definitions based on the conditions
def original_price := 100
def sale_price := 20

-- Theorem statement to prove the percent decrease
theorem percent_decrease_is_80 :
  ((original_price - sale_price) / original_price * 100) = 80 := 
by
  sorry

end percent_decrease_is_80_l1674_167404


namespace beef_original_weight_l1674_167439

noncomputable def originalWeightBeforeProcessing (weightAfterProcessing : ℝ) (lossPercentage : ℝ) : ℝ :=
  weightAfterProcessing / (1 - lossPercentage / 100)

theorem beef_original_weight : originalWeightBeforeProcessing 570 35 = 876.92 :=
by
  sorry

end beef_original_weight_l1674_167439


namespace green_block_weight_l1674_167485

theorem green_block_weight (y g : ℝ) (h1 : y = 0.6) (h2 : y = g + 0.2) : g = 0.4 :=
by
  sorry

end green_block_weight_l1674_167485


namespace sum_is_correct_l1674_167491

theorem sum_is_correct (a b c d : ℤ) 
  (h : a + 1 = b + 2 ∧ b + 2 = c + 3 ∧ c + 3 = d + 4 ∧ d + 4 = a + b + c + d + 7) : 
  a + b + c + d = -6 := 
by 
  sorry

end sum_is_correct_l1674_167491


namespace largest_divisible_by_3_power_l1674_167474

theorem largest_divisible_by_3_power :
  ∃ n : ℕ, (∀ k : ℕ, 1 ≤ k ∧ k ≤ 100 → ∃ m : ℕ, (3^m ∣ (2*k - 1)) → n = 49) :=
sorry

end largest_divisible_by_3_power_l1674_167474


namespace linear_system_incorrect_statement_l1674_167495

def is_determinant (a b c d : ℝ) := a * d - b * c

def is_solution_system (a1 b1 c1 a2 b2 c2 D Dx Dy : ℝ) :=
  D = is_determinant a1 b1 a2 b2 ∧
  Dx = is_determinant c1 b1 c2 b2 ∧
  Dy = is_determinant a1 c1 a2 c2

def is_solution_linear_system (a1 b1 c1 a2 b2 c2 x y : ℝ) :=
  a1 * x + b1 * y = c1 ∧ a2 * x + b2 * y = c2

theorem linear_system_incorrect_statement :
  ∀ (x y : ℝ),
    is_solution_system 3 (-1) 1 1 3 7 10 10 20 ∧
    is_solution_linear_system 3 (-1) 1 1 3 7 x y →
    x = 1 ∧ y = 2 ∧ ¬(20 = -20) := 
by sorry

end linear_system_incorrect_statement_l1674_167495


namespace sum_of_digits_B_l1674_167429

def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).foldl (· + ·) 0

def A : ℕ := sum_of_digits (4444 ^ 4444)

def B : ℕ := sum_of_digits A

theorem sum_of_digits_B : 
  sum_of_digits B = 7 := by
    sorry

end sum_of_digits_B_l1674_167429


namespace sqrt_eq_two_or_neg_two_l1674_167484

theorem sqrt_eq_two_or_neg_two (x : ℝ) (h : x^2 = 4) : x = 2 ∨ x = -2 :=
sorry

end sqrt_eq_two_or_neg_two_l1674_167484


namespace continuity_f_at_1_l1674_167457

theorem continuity_f_at_1 (f : ℝ → ℝ) (x0 : ℝ)
  (h1 : f x0 = -12)
  (h2 : ∀ x : ℝ, f x = -5 * x^2 - 7)
  (h3 : x0 = 1) :
  ∀ ε : ℝ, ε > 0 → ∃ δ : ℝ, δ > 0 ∧ ∀ x : ℝ, |x - x0| < δ → |f x - f x0| < ε :=
by
  sorry

end continuity_f_at_1_l1674_167457


namespace solution_set_of_inequality_l1674_167450

variable {f : ℝ → ℝ}

noncomputable def F (x : ℝ) : ℝ := x^2 * f x

theorem solution_set_of_inequality
  (h_diff : ∀ x < 0, DifferentiableAt ℝ f x) 
  (h_cond : ∀ x < 0, 2 * f x + x * (deriv f x) > x^2) :
  ∀ x, ((x + 2016)^2 * f (x + 2016) - 9 * f (-3) < 0) ↔ (-2019 < x ∧ x < -2016) :=
by
  sorry

end solution_set_of_inequality_l1674_167450


namespace solve_for_y_l1674_167446

-- Define the functions f and g
def f (x : ℝ) : ℝ := 2 * x - 3
def g (x y : ℝ) : ℝ := 3 * x + y

-- State the theorem to be proven
theorem solve_for_y (x y : ℝ) : 2 * f x - 11 + g x y = f (x - 2) ↔ y = -5 * x + 10 :=
by
  sorry

end solve_for_y_l1674_167446


namespace solve_for_x_l1674_167428

theorem solve_for_x (x : ℝ) (h : x ≠ 0) : (3 * x)^5 = (9 * x)^4 → x = 27 := 
by 
  admit

end solve_for_x_l1674_167428


namespace fred_initial_dimes_l1674_167418

theorem fred_initial_dimes (current_dimes borrowed_dimes initial_dimes : ℕ)
  (hc : current_dimes = 4)
  (hb : borrowed_dimes = 3)
  (hi : current_dimes + borrowed_dimes = initial_dimes) :
  initial_dimes = 7 := 
by
  sorry

end fred_initial_dimes_l1674_167418


namespace chef_earns_less_than_manager_l1674_167445

noncomputable def manager_wage : ℚ := 8.50
noncomputable def dishwasher_wage : ℚ := manager_wage / 2
noncomputable def chef_wage : ℚ := dishwasher_wage + 0.22 * dishwasher_wage

theorem chef_earns_less_than_manager :
  manager_wage - chef_wage = 3.315 := by
  sorry

end chef_earns_less_than_manager_l1674_167445


namespace thickness_of_layer_l1674_167488

noncomputable def volume_of_sphere (r : ℝ) : ℝ := (4/3) * Real.pi * r^3

noncomputable def volume_of_cylinder (r h : ℝ) : ℝ := Real.pi * r^2 * h

theorem thickness_of_layer (radius_sphere radius_cylinder : ℝ) (volume_sphere volume_cylinder : ℝ) (h : ℝ) : 
  radius_sphere = 3 → 
  radius_cylinder = 10 →
  volume_sphere = volume_of_sphere radius_sphere →
  volume_cylinder = volume_of_cylinder radius_cylinder h →
  volume_sphere = volume_cylinder → 
  h = 9 / 25 :=
by
  intros
  sorry

end thickness_of_layer_l1674_167488


namespace true_propositions_count_l1674_167420

-- Original Proposition
def P (x y : ℝ) : Prop := x^2 + y^2 = 0 → x = 0 ∧ y = 0

-- Converse Proposition
def Q (x y : ℝ) : Prop := x = 0 ∧ y = 0 → x^2 + y^2 = 0

-- Contrapositive Proposition
def contrapositive_Q_P (x y : ℝ) : Prop := (x ≠ 0 ∨ y ≠ 0) → (x^2 + y^2 ≠ 0)

-- Inverse Proposition
def inverse_P (x y : ℝ) : Prop := (x^2 + y^2 ≠ 0) → (x ≠ 0 ∨ y ≠ 0)

-- Problem Statement
theorem true_propositions_count : ∀ (x y : ℝ),
  P x y ∧ Q x y ∧ contrapositive_Q_P x y ∧ inverse_P x y → 3 = 3 :=
by
  intros x y h
  sorry

end true_propositions_count_l1674_167420


namespace joan_gave_sam_seashells_l1674_167405

theorem joan_gave_sam_seashells (original_seashells : ℕ) (left_seashells : ℕ) (given_seashells : ℕ) 
  (h1 : original_seashells = 70) (h2 : left_seashells = 27) : given_seashells = 43 :=
by
  have h3 : given_seashells = original_seashells - left_seashells := sorry
  rw [h1, h2] at h3
  exact h3

end joan_gave_sam_seashells_l1674_167405


namespace exists_y_lt_p_div2_py_plus1_not_product_of_greater_y_l1674_167452

theorem exists_y_lt_p_div2_py_plus1_not_product_of_greater_y (p : ℕ) [hp : Fact (Nat.Prime p)] (h3 : 3 < p) :
  ∃ y : ℕ, y < p / 2 ∧ ∀ a b : ℕ, py + 1 ≠ a * b ∨ a ≤ y ∨ b ≤ y :=
by
  sorry

end exists_y_lt_p_div2_py_plus1_not_product_of_greater_y_l1674_167452


namespace nathan_write_in_one_hour_l1674_167448

/-- Jacob can write twice as fast as Nathan. Nathan wrote some letters in one hour. Together, they can write 750 letters in 10 hours. How many letters can Nathan write in one hour? -/
theorem nathan_write_in_one_hour
  (N : ℕ)  -- Assume N is the number of letters Nathan can write in one hour
  (H₁ : ∀ (J : ℕ), J = 2 * N)  -- Jacob writes twice faster, so letters written by Jacob in one hour is 2N
  (H₂ : 10 * (N + 2 * N) = 750)  -- Together they write 750 letters in 10 hours
  : N = 25 := by
  -- Proof will go here
  sorry

end nathan_write_in_one_hour_l1674_167448


namespace katherine_savings_multiple_l1674_167498

variable (A K : ℕ)

theorem katherine_savings_multiple
  (h1 : A + K = 750)
  (h2 : A - 150 = 1 / 3 * K) :
  2 * K / A = 3 :=
sorry

end katherine_savings_multiple_l1674_167498


namespace odd_function_k_eq_neg_one_f_x_greater_2_neg_x_k_gt_zero_l1674_167403

noncomputable def f (x : ℝ) (k : ℝ) := 2^x + k * 2^(-x)

-- Prove that if f(x) is an odd function, then k = -1.
theorem odd_function_k_eq_neg_one {k : ℝ} (h : ∀ x, f x k = -f (-x) k) : k = -1 :=
by sorry

-- Prove that if for all x in [0, +∞), f(x) > 2^(-x), then k > 0.
theorem f_x_greater_2_neg_x_k_gt_zero {k : ℝ} (h : ∀ x, 0 ≤ x → f x k > 2^(-x)) : k > 0 :=
by sorry

end odd_function_k_eq_neg_one_f_x_greater_2_neg_x_k_gt_zero_l1674_167403


namespace circle_circumference_l1674_167441

noncomputable def circumference_of_circle (speed1 speed2 time : ℝ) : ℝ :=
  let distance1 := speed1 * time
  let distance2 := speed2 * time
  distance1 + distance2

theorem circle_circumference
    (speed1 speed2 time : ℝ)
    (h1 : speed1 = 7)
    (h2 : speed2 = 8)
    (h3 : time = 12) :
    circumference_of_circle speed1 speed2 time = 180 := by
  sorry

end circle_circumference_l1674_167441


namespace k_h_of_3_eq_79_l1674_167437

def h (x : ℝ) : ℝ := x^3
def k (x : ℝ) : ℝ := 3 * x - 2

theorem k_h_of_3_eq_79 : k (h 3) = 79 := by
  sorry

end k_h_of_3_eq_79_l1674_167437


namespace rest_area_location_l1674_167425

theorem rest_area_location : 
  ∃ (rest_area_milepost : ℕ), 
    let first_exit := 23
    let seventh_exit := 95
    let distance := seventh_exit - first_exit
    let halfway_distance := distance / 2
    rest_area_milepost = first_exit + halfway_distance :=
by
  sorry

end rest_area_location_l1674_167425


namespace greatest_value_of_x_l1674_167479

theorem greatest_value_of_x
  (x : ℕ)
  (h1 : x % 4 = 0) -- x is a multiple of 4
  (h2 : x > 0) -- x is positive
  (h3 : x^3 < 2000) -- x^3 < 2000
  : x ≤ 12 :=
by
  sorry

end greatest_value_of_x_l1674_167479


namespace min_value_of_expression_l1674_167459

noncomputable def min_val_expr (x y : ℝ) : ℝ :=
  (8 / (x + 1)) + (1 / y)

theorem min_value_of_expression
  (x y : ℝ) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (hcond : 2 * x + y = 1) :
  min_val_expr x y = (25 / 3) :=
sorry

end min_value_of_expression_l1674_167459


namespace find_triangle_side1_l1674_167483

def triangle_side1 (Perimeter Side2 Side3 Side1 : ℕ) : Prop :=
  Perimeter = Side1 + Side2 + Side3

theorem find_triangle_side1 :
  ∀ (Perimeter Side2 Side3 Side1 : ℕ), 
    (Perimeter = 160) → (Side2 = 50) → (Side3 = 70) → triangle_side1 Perimeter Side2 Side3 Side1 → Side1 = 40 :=
by
  intros Perimeter Side2 Side3 Side1 h1 h2 h3 h4
  sorry

end find_triangle_side1_l1674_167483


namespace least_people_cheaper_second_caterer_l1674_167443

noncomputable def cost_first_caterer (x : ℕ) : ℕ := 50 + 18 * x

noncomputable def cost_second_caterer (x : ℕ) : ℕ := 
  if x >= 30 then 150 + 15 * x else 180 + 15 * x

theorem least_people_cheaper_second_caterer : ∃ x : ℕ, x = 34 ∧ x >= 30 ∧ cost_second_caterer x < cost_first_caterer x :=
by
  sorry

end least_people_cheaper_second_caterer_l1674_167443


namespace increase_in_average_weight_l1674_167466

variable {A X : ℝ}

-- Given initial conditions
axiom average_initial_weight_8 : X = (8 * A - 62 + 90) / 8 - A

-- The goal to prove
theorem increase_in_average_weight : X = 3.5 :=
by
  sorry

end increase_in_average_weight_l1674_167466


namespace intersection_P_Q_l1674_167421

def P := {x : ℝ | x^2 - 9 < 0}
def Q := {y : ℤ | ∃ x : ℤ, y = 2*x}

theorem intersection_P_Q :
  {x : ℝ | x ∈ P ∧ (∃ n : ℤ, x = 2*n)} = {-2, 0, 2} :=
by
  sorry

end intersection_P_Q_l1674_167421


namespace jungkook_needs_more_paper_l1674_167413

def bundles : Nat := 5
def pieces_per_bundle : Nat := 8
def rows : Nat := 9
def sheets_per_row : Nat := 6

def total_pieces : Nat := bundles * pieces_per_bundle
def pieces_needed : Nat := rows * sheets_per_row
def pieces_missing : Nat := pieces_needed - total_pieces

theorem jungkook_needs_more_paper : pieces_missing = 14 := by
  sorry

end jungkook_needs_more_paper_l1674_167413


namespace calculate_expression_l1674_167430

theorem calculate_expression :
  (-2)^(4^2) + 2^(3^2) = 66048 := by sorry

end calculate_expression_l1674_167430


namespace max_apartment_size_l1674_167416

theorem max_apartment_size (rate cost per_sqft : ℝ) (budget : ℝ) (h1 : rate = 1.20) (h2 : budget = 864) : cost = 720 :=
by
  sorry

end max_apartment_size_l1674_167416


namespace a_plus_b_equals_4_l1674_167456

theorem a_plus_b_equals_4 (f : ℝ → ℝ) (a b : ℝ) (h_dom : ∀ x, 1 ≤ x ∧ x ≤ b → f x = (1/2) * (x-1)^2 + a)
  (h_range : ∀ y, 1 ≤ y ∧ y ≤ b → ∃ x, 1 ≤ x ∧ x ≤ b ∧ f x = y) (h_b_pos : b > 1) : a + b = 4 :=
sorry

end a_plus_b_equals_4_l1674_167456


namespace maria_workday_end_l1674_167475

def time_in_minutes (h : ℕ) (m : ℕ) : ℕ := h * 60 + m

def start_time : ℕ := time_in_minutes 7 25
def lunch_break : ℕ := 45
def noon : ℕ := time_in_minutes 12 0
def work_hours : ℕ := 8 * 60
def end_time : ℕ := time_in_minutes 16 10

theorem maria_workday_end : start_time + (noon - start_time) + lunch_break + (work_hours - (noon - start_time)) = end_time := by
  sorry

end maria_workday_end_l1674_167475


namespace repeated_number_divisible_by_1001001_l1674_167444

theorem repeated_number_divisible_by_1001001 (a b c : ℕ) (h1 : a < 10) (h2 : b < 10) (h3 : c < 10) : 
  (1000000 * (100 * a + 10 * b + c) + 1000 * (100 * a + 10 * b + c) + (100 * a + 10 * b + c)) % 1001001 = 0 := 
by 
  sorry

end repeated_number_divisible_by_1001001_l1674_167444


namespace finite_non_friends_iff_l1674_167435

def isFriend (u n : ℕ) : Prop :=
  ∃ N : ℕ, N % n = 0 ∧ (N.digits 10).sum = u

theorem finite_non_friends_iff (n : ℕ) : (∃ᶠ u in at_top, ¬ isFriend u n) ↔ ¬ (3 ∣ n) := 
by
  sorry

end finite_non_friends_iff_l1674_167435


namespace calculate_fraction_l1674_167436

theorem calculate_fraction :
  (10^9 / (2 * 10^5) = 5000) :=
  sorry

end calculate_fraction_l1674_167436
