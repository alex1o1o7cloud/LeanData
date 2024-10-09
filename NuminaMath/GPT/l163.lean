import Mathlib

namespace inheritance_amount_l163_16380

theorem inheritance_amount
  (x : ℝ)
  (H1 : 0.25 * x + 0.15 * (x - 0.25 * x) = 15000) : x = 41379 := 
sorry

end inheritance_amount_l163_16380


namespace mass_of_alcl3_formed_l163_16384

noncomputable def molarMass (atomicMasses : List (ℕ × ℕ)) : ℕ :=
atomicMasses.foldl (λ acc elem => acc + elem.1 * elem.2) 0

theorem mass_of_alcl3_formed :
  let atomic_mass_al := 26.98
  let atomic_mass_cl := 35.45
  let molar_mass_alcl3 := 2 * atomic_mass_al + 3 * atomic_mass_cl
  let moles_al2co3 := 10
  let moles_alcl3 := 2 * moles_al2co3
  let mass_alcl3 := moles_alcl3 * molar_mass_alcl3
  mass_alcl3 = 3206.2 := sorry

end mass_of_alcl3_formed_l163_16384


namespace max_alpha_flights_achievable_l163_16393

def max_alpha_flights (n : ℕ) : ℕ :=
  let total_flights := n * (n - 1) / 2
  let max_beta_flights := n / 2
  total_flights - max_beta_flights

theorem max_alpha_flights_achievable (n : ℕ) : 
  ∃ k, k = n * (n - 1) / 2 - n / 2 ∧ k ≤ max_alpha_flights n :=
by
  sorry

end max_alpha_flights_achievable_l163_16393


namespace haley_initial_cupcakes_l163_16353

-- Define the conditions
def todd_eats : ℕ := 11
def packages : ℕ := 3
def cupcakes_per_package : ℕ := 3

-- Initial cupcakes calculation
def initial_cupcakes := packages * cupcakes_per_package + todd_eats

-- The theorem to prove
theorem haley_initial_cupcakes : initial_cupcakes = 20 :=
by
  -- Mathematical proof would go here,
  -- but we leave it as sorry for now.
  sorry

end haley_initial_cupcakes_l163_16353


namespace distinct_integer_roots_l163_16378

-- Definitions of m and the polynomial equation.
def poly (m : ℤ) (x : ℤ) : Prop :=
  x^2 - 2 * (2 * m - 3) * x + 4 * m^2 - 14 * m + 8 = 0

-- Theorem stating that for m = 12 and m = 24, the polynomial has specific roots.
theorem distinct_integer_roots (m x : ℤ) (h1 : 4 < m) (h2 : m < 40) :
  (m = 12 ∨ m = 24) ∧ 
  ((m = 12 ∧ (x = 26 ∨ x = 16) ∧ poly m x) ∨
   (m = 24 ∧ (x = 52 ∨ x = 38) ∧ poly m x)) :=
by
  sorry

end distinct_integer_roots_l163_16378


namespace Marias_score_l163_16394

def total_questions := 30
def points_per_correct_answer := 20
def points_deducted_per_incorrect_answer := 5
def total_answered := total_questions
def correct_answers := 19
def incorrect_answers := total_questions - correct_answers
def score := (correct_answers * points_per_correct_answer) - (incorrect_answers * points_deducted_per_incorrect_answer)

theorem Marias_score : score = 325 := by
  -- proof goes here
  sorry

end Marias_score_l163_16394


namespace pet_store_cages_l163_16346

theorem pet_store_cages 
  (snakes parrots rabbits snake_cage_capacity parrot_cage_capacity rabbit_cage_capacity : ℕ)
  (h_snakes : snakes = 4) 
  (h_parrots : parrots = 6) 
  (h_rabbits : rabbits = 8) 
  (h_snake_cage_capacity : snake_cage_capacity = 2) 
  (h_parrot_cage_capacity : parrot_cage_capacity = 3) 
  (h_rabbit_cage_capacity : rabbit_cage_capacity = 4) 
  : (snakes / snake_cage_capacity) + (parrots / parrot_cage_capacity) + (rabbits / rabbit_cage_capacity) = 6 := 
by 
  sorry

end pet_store_cages_l163_16346


namespace g_of_neg_5_is_4_l163_16324

def f (x : ℝ) : ℝ := 3 * x - 8
def g (y : ℝ) : ℝ := 2 * y^2 + 5 * y - 3

theorem g_of_neg_5_is_4 : g (-5) = 4 :=
by
  sorry

end g_of_neg_5_is_4_l163_16324


namespace possible_values_of_d_l163_16375

theorem possible_values_of_d :
  ∃ (e f d : ℤ), (e + 12) * (f + 12) = 1 ∧
  ∀ x, (x - d) * (x - 12) + 1 = (x + e) * (x + f) ↔ (d = 22 ∨ d = 26) :=
by
  sorry

end possible_values_of_d_l163_16375


namespace delivery_payment_l163_16315

-- Define the problem conditions and the expected outcome
theorem delivery_payment 
    (deliveries_Oula : ℕ) 
    (deliveries_Tona : ℕ) 
    (difference_in_pay : ℝ) 
    (P : ℝ) 
    (H1 : deliveries_Oula = 96) 
    (H2 : deliveries_Tona = 72) 
    (H3 : difference_in_pay = 2400) :
    96 * P - 72 * P = 2400 → P = 100 :=
by
  intro h1
  sorry

end delivery_payment_l163_16315


namespace solve_x_1_solve_x_2_solve_x_3_l163_16319

-- Proof 1: Given 356 * x = 2492, prove that x = 7
theorem solve_x_1 (x : ℕ) (h : 356 * x = 2492) : x = 7 :=
sorry

-- Proof 2: Given x / 39 = 235, prove that x = 9165
theorem solve_x_2 (x : ℕ) (h : x / 39 = 235) : x = 9165 :=
sorry

-- Proof 3: Given 1908 - x = 529, prove that x = 1379
theorem solve_x_3 (x : ℕ) (h : 1908 - x = 529) : x = 1379 :=
sorry

end solve_x_1_solve_x_2_solve_x_3_l163_16319


namespace car_value_correct_l163_16372

-- Define the initial value and the annual decrease percentages
def initial_value : ℝ := 10000
def annual_decreases : List ℝ := [0.20, 0.15, 0.10, 0.08, 0.05]

-- Function to compute the value of the car after n years
def value_after_years (initial_value : ℝ) (annual_decreases : List ℝ) : ℝ :=
  annual_decreases.foldl (λ acc decrease => acc * (1 - decrease)) initial_value

-- The target value after 5 years
def target_value : ℝ := 5348.88

-- Theorem stating that the computed value matches the target value
theorem car_value_correct :
  value_after_years initial_value annual_decreases = target_value := 
sorry

end car_value_correct_l163_16372


namespace tangency_of_parabolas_l163_16343

theorem tangency_of_parabolas :
  ∃ x y : ℝ, y = x^2 + 12*x + 40
  ∧ x = y^2 + 44*y + 400
  ∧ x = -11 / 2
  ∧ y = -43 / 2 := by
sorry

end tangency_of_parabolas_l163_16343


namespace triangle_ratio_l163_16337

theorem triangle_ratio (L W : ℝ) (hL : L > 0) (hW : W > 0) : 
  let p := 12;
  let q := 8;
  let segment_length := L / p;
  let segment_width := W / q;
  let area_X := (segment_length * segment_width) / 2;
  let area_rectangle := L * W;
  (area_X / area_rectangle) = (1 / 192) :=
by 
  sorry

end triangle_ratio_l163_16337


namespace sum_of_first_19_terms_l163_16364

noncomputable def a_n (a1 : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a1 + (n - 1) * d

noncomputable def S_n (a1 : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  n / 2 * (a1 + a_n a1 d n)

theorem sum_of_first_19_terms (a1 d : ℝ) (h : a1 + 9 * d = 1) : S_n a1 d 19 = 19 := by
  sorry

end sum_of_first_19_terms_l163_16364


namespace triangle_construction_conditions_l163_16365

open Classical

noncomputable def construct_triangle (m_a m_b s_c : ℝ) : Prop :=
  m_a ≤ 2 * s_c ∧ m_b ≤ 2 * s_c

theorem triangle_construction_conditions (m_a m_b s_c : ℝ) :
  construct_triangle m_a m_b s_c ↔ (m_a ≤ 2 * s_c ∧ m_b ≤ 2 * s_c) :=
by
  sorry

end triangle_construction_conditions_l163_16365


namespace integer_solutions_l163_16386

theorem integer_solutions (n : ℤ) : (n^2 + 1) ∣ (n^5 + 3) ↔ n = -3 ∨ n = -1 ∨ n = 0 ∨ n = 1 ∨ n = 2 := 
sorry

end integer_solutions_l163_16386


namespace find_pairs_of_numbers_l163_16333

theorem find_pairs_of_numbers (a b : ℝ) :
  (a^2 + b^2 = 15 * (a + b)) ∧ (a^2 - b^2 = 3 * (a - b) ∨ a^2 - b^2 = -3 * (a - b))
  ↔ (a = 6 ∧ b = -3) ∨ (a = -3 ∧ b = 6) ∨ (a = 0 ∧ b = 0) ∨ (a = 15 ∧ b = 15) :=
sorry

end find_pairs_of_numbers_l163_16333


namespace mixedGasTemperature_is_correct_l163_16390

noncomputable def mixedGasTemperature (V₁ V₂ p₁ p₂ T₁ T₂ : ℝ) : ℝ := 
  (p₁ * V₁ + p₂ * V₂) / ((p₁ * V₁) / T₁ + (p₂ * V₂) / T₂)

theorem mixedGasTemperature_is_correct :
  mixedGasTemperature 2 3 3 4 400 500 = 462 := by
    sorry

end mixedGasTemperature_is_correct_l163_16390


namespace find_a_l163_16348

def F (a b c : ℝ) : ℝ := a * (b^2 + c^2) + b * c

theorem find_a (a : ℝ) (h : F a 3 4 = F a 2 5) : a = 1 / 2 :=
by
  sorry

end find_a_l163_16348


namespace four_digit_numbers_sum_even_l163_16339

theorem four_digit_numbers_sum_even : 
  ∃ N : ℕ, 
    (∀ (digits : Finset ℕ) (thousands hundreds tens units : ℕ), 
      digits = {1, 2, 3, 4, 5, 6} ∧ 
      ∀ n ∈ digits, (0 < n ∧ n < 10) ∧ 
      (thousands ∈ digits ∧ hundreds ∈ digits ∧ tens ∈ digits ∧ units ∈ digits) ∧ 
      (thousands ≠ hundreds ∧ thousands ≠ tens ∧ thousands ≠ units ∧ 
       hundreds ≠ tens ∧ hundreds ≠ units ∧ tens ≠ units) ∧ 
      (tens + units) % 2 = 0 → N = 324) :=
sorry

end four_digit_numbers_sum_even_l163_16339


namespace linear_function_through_two_points_l163_16379

theorem linear_function_through_two_points :
  ∃ (k b : ℝ), (∀ x, y = k * x + b) ∧
  (k ≠ 0) ∧
  (3 = 2 * k + b) ∧
  (2 = 3 * k + b) ∧
  (∀ x, y = -x + 5) :=
by
  sorry

end linear_function_through_two_points_l163_16379


namespace person_half_Jordyn_age_is_6_l163_16388

variables (Mehki_age Jordyn_age certain_age : ℕ)
axiom h1 : Mehki_age = Jordyn_age + 10
axiom h2 : Jordyn_age = 2 * certain_age
axiom h3 : Mehki_age = 22

theorem person_half_Jordyn_age_is_6 : certain_age = 6 :=
by sorry

end person_half_Jordyn_age_is_6_l163_16388


namespace simplify_and_evaluate_l163_16377

theorem simplify_and_evaluate (a b : ℤ) (h₁ : a = -1) (h₂ : b = 3) :
  2 * a * b^2 - (3 * a^2 * b - 2 * (3 * a^2 * b - a * b^2 - 1)) = 7 :=
by
  sorry

end simplify_and_evaluate_l163_16377


namespace no_conclusions_deducible_l163_16317

open Set

variable {U : Type}  -- Universe of discourse

-- Conditions
variables (Bars Fins Grips : Set U)

def some_bars_are_not_fins := ∃ x, x ∈ Bars ∧ x ∉ Fins
def no_fins_are_grips := ∀ x, x ∈ Fins → x ∉ Grips

-- Lean statement
theorem no_conclusions_deducible 
  (h1 : some_bars_are_not_fins Bars Fins)
  (h2 : no_fins_are_grips Fins Grips) :
  ¬((∃ x, x ∈ Bars ∧ x ∉ Grips) ∨
    (∃ x, x ∈ Grips ∧ x ∉ Bars) ∨
    (∀ x, x ∈ Bars → x ∉ Grips) ∨
    (∃ x, x ∈ Bars ∧ x ∈ Grips)) :=
sorry

end no_conclusions_deducible_l163_16317


namespace min_students_l163_16357

theorem min_students (S a b c : ℕ) (h1 : 3 * a > S) (h2 : 10 * b > 3 * S) (h3 : 11 * c > 4 * S) (h4 : S = a + b + c) : S ≥ 173 :=
by
  sorry

end min_students_l163_16357


namespace correct_comparison_l163_16358

theorem correct_comparison :
  ( 
    (-1 > -0.1) = false ∧ 
    (-4 / 3 < -5 / 4) = true ∧ 
    (-1 / 2 > -(-1 / 3)) = false ∧ 
    (Real.pi = 3.14) = false 
  ) :=
by
  sorry

end correct_comparison_l163_16358


namespace coffee_grinder_assembly_time_l163_16398

-- Variables for the assembly rates
variables (h r : ℝ)

-- Definitions of conditions
def condition1 : Prop := h / 4 = r
def condition2 : Prop := r / 4 = h
def condition3 : Prop := ∀ start_time end_time net_added, 
  start_time = 9 ∧ end_time = 12 ∧ net_added = 27 → 3 * 3/4 * h = net_added
def condition4 : Prop := ∀ start_time end_time net_added, 
  start_time = 13 ∧ end_time = 19 ∧ net_added = 120 → 6 * 3/4 * r = net_added

-- Theorem statement
theorem coffee_grinder_assembly_time
  (h r : ℝ)
  (c1 : condition1 h r)
  (c2 : condition2 h r)
  (c3 : condition3 h)
  (c4 : condition4 r) :
  h = 12 ∧ r = 80 / 3 :=
sorry

end coffee_grinder_assembly_time_l163_16398


namespace total_hunts_l163_16366

-- Conditions
def Sam_hunts : ℕ := 6
def Rob_hunts := Sam_hunts / 2
def combined_Rob_Sam_hunts := Rob_hunts + Sam_hunts
def Mark_hunts := combined_Rob_Sam_hunts / 3
def Peter_hunts := 3 * Mark_hunts

-- Question and proof statement
theorem total_hunts : Sam_hunts + Rob_hunts + Mark_hunts + Peter_hunts = 21 := by
  sorry

end total_hunts_l163_16366


namespace garden_width_l163_16392

theorem garden_width :
  ∃ w l : ℝ, (2 * l + 2 * w = 60) ∧ (l * w = 200) ∧ (l = 2 * w) ∧ (w = 10) :=
by
  sorry

end garden_width_l163_16392


namespace average_revenue_per_hour_l163_16399

theorem average_revenue_per_hour 
    (sold_A_hour1 : ℕ) (sold_B_hour1 : ℕ) (sold_A_hour2 : ℕ) (sold_B_hour2 : ℕ)
    (price_A_hour1 : ℕ) (price_A_hour2 : ℕ) (price_B_constant : ℕ) : 
    (sold_A_hour1 = 10) ∧ (sold_B_hour1 = 5) ∧ (sold_A_hour2 = 2) ∧ (sold_B_hour2 = 3) ∧
    (price_A_hour1 = 3) ∧ (price_A_hour2 = 4) ∧ (price_B_constant = 2) →
    (54 / 2 = 27) :=
by
  intros
  sorry

end average_revenue_per_hour_l163_16399


namespace martha_total_clothes_l163_16310

def jackets_purchased : ℕ := 4
def tshirts_purchased : ℕ := 9
def jackets_free : ℕ := jackets_purchased / 2
def tshirts_free : ℕ := tshirts_purchased / 3
def total_jackets : ℕ := jackets_purchased + jackets_free
def total_tshirts : ℕ := tshirts_purchased + tshirts_free

theorem martha_total_clothes : total_jackets + total_tshirts = 18 := by
  sorry

end martha_total_clothes_l163_16310


namespace find_n_l163_16368

theorem find_n :
  ∃ n : ℕ, ∀ (a b c : ℕ), a + b + c = 200 ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a ∧
    (n = a + b * c) ∧ (n = b + c * a) ∧ (n = c + a * b) → n = 199 :=
by {
  sorry
}

end find_n_l163_16368


namespace necessary_not_sufficient_condition_l163_16314

noncomputable def S (a₁ q : ℝ) : ℝ := a₁ / (1 - q)

theorem necessary_not_sufficient_condition (a₁ q : ℝ) (h₁ : |q| < 1) :
  (a₁ + q = 1) → (S a₁ q = 1) ∧ ¬((S a₁ q = 1) → (a₁ + q = 1)) :=
by
  sorry

end necessary_not_sufficient_condition_l163_16314


namespace intersecting_circles_l163_16320

noncomputable def distance (z1 z2 : Complex) : ℝ :=
  Complex.abs (z1 - z2)

theorem intersecting_circles (k : ℝ) :
  (∀ (z : Complex), (distance z 4 = 3 * distance z (-4)) → (distance z 0 = k)) →
  (k = 13 + Real.sqrt 153 ∨ k = |13 - Real.sqrt 153|) := 
sorry

end intersecting_circles_l163_16320


namespace polynomial_solution_l163_16335

noncomputable def p (x : ℝ) := 2 * Real.sqrt 3 * x^4 - 6

theorem polynomial_solution (x : ℝ) : 
  (p (x^4) - p (x^4 - 3) = (p x)^3 - 18) :=
by
  sorry

end polynomial_solution_l163_16335


namespace unique_spicy_pair_l163_16338

def is_spicy (n : ℕ) : Prop :=
  let A := (n / 100) % 10
  let B := (n / 10) % 10
  let C := n % 10
  n = A^3 + B^3 + C^3

theorem unique_spicy_pair : ∃! n : ℕ, is_spicy n ∧ is_spicy (n + 1) ∧ 100 ≤ n ∧ n < 1000 ∧ n = 370 := 
sorry

end unique_spicy_pair_l163_16338


namespace not_basic_logic_structure_l163_16332

def SequenceStructure : Prop := true
def ConditionStructure : Prop := true
def LoopStructure : Prop := true
def DecisionStructure : Prop := true

theorem not_basic_logic_structure : ¬ (SequenceStructure ∨ ConditionStructure ∨ LoopStructure) -> DecisionStructure := by
  sorry

end not_basic_logic_structure_l163_16332


namespace find_x_l163_16303

noncomputable def isCorrectValue (x : ℝ) : Prop :=
  ⌊x⌋ + x = 13.4

theorem find_x (x : ℝ) (h : isCorrectValue x) : x = 6.4 :=
  sorry

end find_x_l163_16303


namespace actual_cost_of_article_l163_16347

theorem actual_cost_of_article (x : ℝ) (h : 0.80 * x = 620) : x = 775 :=
sorry

end actual_cost_of_article_l163_16347


namespace initial_group_size_l163_16313

theorem initial_group_size
  (n : ℕ) (W : ℝ)
  (h_avg_increase : ∀ W n, ((W + 12) / n) = (W / n + 3))
  (h_new_person_weight : 82 = 70 + 12) : n = 4 :=
by
  sorry

end initial_group_size_l163_16313


namespace train_length_is_250_l163_16362

-- Define the length of the train
def train_length (L : ℝ) (V : ℝ) :=
  -- Condition 1
  (V = L / 10) → 
  -- Condition 2
  (V = (L + 1250) / 60) → 
  -- Question
  L = 250

-- Here's the statement that we expect to prove
theorem train_length_is_250 (L V : ℝ) : train_length L V :=
by {
  -- sorry is a placeholder to indicate the theorem proof is omitted
  sorry
}

end train_length_is_250_l163_16362


namespace find_prime_power_solutions_l163_16304

theorem find_prime_power_solutions (p n m : ℕ) (hp : Nat.Prime p) (hn : n > 0) (hm : m > 0) 
  (h : p^n + 144 = m^2) :
  (p = 2 ∧ n = 9 ∧ m = 36) ∨ (p = 3 ∧ n = 4 ∧ m = 27) :=
by sorry

end find_prime_power_solutions_l163_16304


namespace average_percentage_of_kernels_popped_l163_16334

theorem average_percentage_of_kernels_popped :
  let bag1_popped := 60
  let bag1_total := 75
  let bag2_popped := 42
  let bag2_total := 50
  let bag3_popped := 82
  let bag3_total := 100
  let percentage (popped total : ℕ) := (popped : ℚ) / total * 100
  let p1 := percentage bag1_popped bag1_total
  let p2 := percentage bag2_popped bag2_total
  let p3 := percentage bag3_popped bag3_total
  let avg := (p1 + p2 + p3) / 3
  avg = 82 :=
by
  sorry

end average_percentage_of_kernels_popped_l163_16334


namespace monthly_manufacturing_expenses_l163_16373

theorem monthly_manufacturing_expenses 
  (num_looms : ℕ) (total_sales_value : ℚ) 
  (monthly_establishment_charges : ℚ) 
  (decrease_in_profit : ℚ) 
  (sales_per_loom : ℚ) 
  (manufacturing_expenses_per_loom : ℚ) 
  (total_manufacturing_expenses : ℚ) : 
  num_looms = 80 → 
  total_sales_value = 500000 → 
  monthly_establishment_charges = 75000 → 
  decrease_in_profit = 4375 → 
  sales_per_loom = total_sales_value / num_looms → 
  manufacturing_expenses_per_loom = sales_per_loom - decrease_in_profit → 
  total_manufacturing_expenses = manufacturing_expenses_per_loom * num_looms →
  total_manufacturing_expenses = 150000 :=
by
  intros h_num_looms h_total_sales h_monthly_est_charges h_decrease_in_profit h_sales_per_loom h_manufacturing_expenses_per_loom h_total_manufacturing_expenses
  sorry

end monthly_manufacturing_expenses_l163_16373


namespace calculate_x_value_l163_16374

theorem calculate_x_value : 
  529 + 2 * 23 * 3 + 9 = 676 := 
by
  sorry

end calculate_x_value_l163_16374


namespace hyperbola_equation_l163_16369

theorem hyperbola_equation 
  (x y : ℝ)
  (h_ellipse : x^2 / 10 + y^2 / 5 = 1)
  (h_asymptote : 3 * x + 4 * y = 0)
  (h_hyperbola : ∃ k ≠ 0, 9 * x^2 - 16 * y^2 = k) :
  ∃ k : ℝ, k = 45 ∧ (x^2 / 5 - 16 * y^2 / 45 = 1) :=
sorry

end hyperbola_equation_l163_16369


namespace smallest_n_perfect_square_and_cube_l163_16355

theorem smallest_n_perfect_square_and_cube (n : ℕ) (h1 : ∃ k : ℕ, 5 * n = k^2) (h2 : ∃ m : ℕ, 4 * n = m^3) :
  n = 1080 :=
  sorry

end smallest_n_perfect_square_and_cube_l163_16355


namespace tickets_per_ride_factor_l163_16312

theorem tickets_per_ride_factor (initial_tickets spent_tickets remaining_tickets : ℕ) 
  (h1 : initial_tickets = 40) 
  (h2 : spent_tickets = 28) 
  (h3 : remaining_tickets = initial_tickets - spent_tickets) : 
  ∃ k : ℕ, remaining_tickets = 12 ∧ (∀ m : ℕ, m ∣ remaining_tickets → m = k) → (k ∣ 12) :=
by
  sorry

end tickets_per_ride_factor_l163_16312


namespace boys_count_l163_16331

variable (B G : ℕ)

theorem boys_count (h1 : B + G = 466) (h2 : G = B + 212) : B = 127 := by
  sorry

end boys_count_l163_16331


namespace distance_to_school_is_correct_l163_16367

-- Define the necessary constants, variables, and conditions
def distance_to_market : ℝ := 2
def total_weekly_mileage : ℝ := 44
def school_trip_miles (x : ℝ) : ℝ := 16 * x
def market_trip_miles : ℝ := 2 * distance_to_market
def total_trip_miles (x : ℝ) : ℝ := school_trip_miles x + market_trip_miles

-- Prove that the distance from Philip's house to the children's school is 2.5 miles
theorem distance_to_school_is_correct (x : ℝ) (h : total_trip_miles x = total_weekly_mileage) :
  x = 2.5 :=
by
  -- Insert necessary proof steps starting with the provided hypothesis
  sorry

end distance_to_school_is_correct_l163_16367


namespace bus_is_there_probability_l163_16371

noncomputable def probability_bus_present : ℚ :=
  let total_area := 90 * 90
  let triangle_area := (75 * 75) / 2
  let parallelogram_area := 75 * 15
  let shaded_area := triangle_area + parallelogram_area
  shaded_area / total_area

theorem bus_is_there_probability :
  probability_bus_present = 7/16 :=
by
  sorry

end bus_is_there_probability_l163_16371


namespace race_distance_l163_16301

theorem race_distance 
  (D : ℝ) 
  (A_time : ℝ) (B_time : ℝ) 
  (A_beats_B_by : ℝ) 
  (A_time_eq : A_time = 36)
  (B_time_eq : B_time = 45)
  (A_beats_B_by_eq : A_beats_B_by = 24) :
  ((D / A_time) * B_time = D + A_beats_B_by) -> D = 24 := 
by 
  sorry

end race_distance_l163_16301


namespace largest_y_coordinate_l163_16397

theorem largest_y_coordinate (x y : ℝ) (h : x^2 / 49 + (y - 3)^2 / 25 = 0) : y = 3 :=
sorry

end largest_y_coordinate_l163_16397


namespace percentage_error_calculation_l163_16395

theorem percentage_error_calculation (x : ℝ) :
  let correct_value := x * (5 / 3)
  let incorrect_value := x * (3 / 5)
  let difference := correct_value - incorrect_value
  let percentage_error := (difference / correct_value) * 100
  percentage_error = 64 := 
by
  let correct_value := x * (5 / 3)
  let incorrect_value := x * (3 / 5)
  let difference := correct_value - incorrect_value
  let percentage_error := (difference / correct_value) * 100
  sorry

end percentage_error_calculation_l163_16395


namespace range_of_f_l163_16350

noncomputable def f (x : ℝ) := Real.log (2 - x^2) / Real.log (1 / 2)

theorem range_of_f : Set.range f = Set.Icc (-1 : ℝ) 0 := by
  sorry

end range_of_f_l163_16350


namespace minimum_value_l163_16311

variable (a b c : ℝ)
variable (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
variable (h_sum : a + b + c = 3)

theorem minimum_value : 
  (1 / (3 * a + 5 * b)) + (1 / (3 * b + 5 * c)) + (1 / (3 * c + 5 * a)) ≥ 9 / 8 :=
by
  sorry

end minimum_value_l163_16311


namespace speeds_of_cars_l163_16330

theorem speeds_of_cars (d_A d_B : ℝ) (v_A v_B : ℝ) (h1 : d_A = 300) (h2 : d_B = 250) (h3 : v_A = v_B + 5) (h4 : d_A / v_A = d_B / v_B) :
  v_B = 25 ∧ v_A = 30 :=
by
  sorry

end speeds_of_cars_l163_16330


namespace matrix_norm_min_l163_16360

-- Definition of the matrix
def matrix_mul (a b c d : ℤ) : Option (ℤ × ℤ × ℤ × ℤ) :=
  if a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 then
    some (a^2 + b * c, a * b + b * d, a * c + c * d, b * c + d^2)
  else
    none

-- Main theorem statement
theorem matrix_norm_min (a b c d : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (hc : c ≠ 0) (hd : d ≠ 0) :
  matrix_mul a b c d = some (8, 0, 0, 5) → 
  |a| + |b| + |c| + |d| = 9 :=
by
  sorry

end matrix_norm_min_l163_16360


namespace sufficient_not_necessary_l163_16323

theorem sufficient_not_necessary (x : ℝ) : (x > 3 → x > 1) ∧ ¬ (x > 1 → x > 3) :=
by 
  sorry

end sufficient_not_necessary_l163_16323


namespace salary_increase_l163_16318

theorem salary_increase (original_salary reduced_salary : ℝ) (hx : reduced_salary = original_salary * 0.5) : 
  (reduced_salary + reduced_salary * 1) = original_salary :=
by
  -- Prove the required increase percent to return to original salary
  sorry

end salary_increase_l163_16318


namespace point_on_graph_l163_16396

theorem point_on_graph (x y : ℝ) (h : y = 3 * x + 1) : (x, y) = (2, 7) :=
sorry

end point_on_graph_l163_16396


namespace mrs_martin_pays_l163_16340

def kiddie_scoop_cost : ℕ := 3
def regular_scoop_cost : ℕ := 4
def double_scoop_cost : ℕ := 6

def mr_martin_scoops : ℕ := 1
def mrs_martin_scoops : ℕ := 1
def children_scoops : ℕ := 2
def teenage_children_scoops : ℕ := 3

def total_cost : ℕ :=
  (mr_martin_scoops + mrs_martin_scoops) * regular_scoop_cost +
  children_scoops * kiddie_scoop_cost +
  teenage_children_scoops * double_scoop_cost

theorem mrs_martin_pays : total_cost = 32 :=
  by sorry

end mrs_martin_pays_l163_16340


namespace remainder_of_sum_l163_16302

theorem remainder_of_sum (a b c : ℕ) (h₁ : a % 15 = 11) (h₂ : b % 15 = 12) (h₃ : c % 15 = 13) : 
  (a + b + c) % 15 = 6 :=
by 
  sorry

end remainder_of_sum_l163_16302


namespace equivalent_fraction_power_multiplication_l163_16387

theorem equivalent_fraction_power_multiplication : 
  (8 / 9) ^ 2 * (1 / 3) ^ 2 * (2 / 5) = (128 / 3645) := 
by 
  sorry

end equivalent_fraction_power_multiplication_l163_16387


namespace simplify_expression_l163_16361

variable (x : ℝ)

theorem simplify_expression (x : ℝ) : (2 * x ^ 3) ^ 3 = 8 * x ^ 9 := by
  sorry

end simplify_expression_l163_16361


namespace comic_book_arrangement_l163_16325

theorem comic_book_arrangement :
  let spiderman_books := 7
  let archie_books := 6
  let garfield_books := 5
  let groups := 3
  Nat.factorial spiderman_books * Nat.factorial archie_books * Nat.factorial garfield_books * Nat.factorial groups = 248005440 :=
by
  sorry

end comic_book_arrangement_l163_16325


namespace no_two_primes_sum_to_10003_l163_16344

-- Define what it means for a number to be a prime
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the specific numbers involved
def even_prime : ℕ := 2
def target_number : ℕ := 10003
def candidate : ℕ := target_number - even_prime

-- State the main proposition in question
theorem no_two_primes_sum_to_10003 :
  ¬ ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p + q = target_number :=
sorry

end no_two_primes_sum_to_10003_l163_16344


namespace factor_expression_l163_16391

theorem factor_expression (x : ℚ) : 12 * x ^ 2 + 8 * x = 4 * x * (3 * x + 2) := sorry

end factor_expression_l163_16391


namespace angle_between_lines_at_most_l163_16383
-- Import the entire Mathlib library for general mathematical definitions

-- Define the problem statement in Lean 4
theorem angle_between_lines_at_most (n : ℕ) (h : n > 0) :
  ∃ (l1 l2 : ℝ), l1 ≠ l2 ∧ (n : ℝ) > 0 → ∃ θ, 0 ≤ θ ∧ θ ≤ 180 / n := by
  sorry

end angle_between_lines_at_most_l163_16383


namespace unique_ordered_triples_count_l163_16382

theorem unique_ordered_triples_count :
  ∃ (n : ℕ), n = 1 ∧ ∀ (a b c : ℕ), 1 ≤ a ∧ a ≤ b ∧ b ≤ c ∧
  abc = 4 * (ab + bc + ca) ∧ a = c / 4 -> False :=
sorry

end unique_ordered_triples_count_l163_16382


namespace cleaned_area_correct_l163_16342

def lizzie_cleaned : ℚ := 3534 + 2/3
def hilltown_team_cleaned : ℚ := 4675 + 5/8
def green_valley_cleaned : ℚ := 2847 + 7/9
def riverbank_cleaned : ℚ := 6301 + 1/3
def meadowlane_cleaned : ℚ := 3467 + 4/5

def total_cleaned : ℚ := lizzie_cleaned + hilltown_team_cleaned + green_valley_cleaned + riverbank_cleaned + meadowlane_cleaned
def total_farmland : ℚ := 28500

def remaining_area_to_clean : ℚ := total_farmland - total_cleaned

theorem cleaned_area_correct : remaining_area_to_clean = 7672.7964 :=
by
  sorry

end cleaned_area_correct_l163_16342


namespace fractional_equation_positive_root_l163_16322

theorem fractional_equation_positive_root (a : ℝ) (ha : ∃ x : ℝ, x > 0 ∧ (6 / (x - 2) - 1 = a * x / (2 - x))) : a = -3 :=
by
  sorry

end fractional_equation_positive_root_l163_16322


namespace rex_has_399_cards_left_l163_16359

def Nicole_cards := 700

def Cindy_cards := 3 * Nicole_cards + (40 / 100) * (3 * Nicole_cards)
def Tim_cards := (4 / 5) * Cindy_cards
def combined_total := Nicole_cards + Cindy_cards + Tim_cards
def Rex_and_Joe_cards := (60 / 100) * combined_total

def cards_per_person := Nat.floor (Rex_and_Joe_cards / 9)

theorem rex_has_399_cards_left : cards_per_person = 399 := by
  sorry

end rex_has_399_cards_left_l163_16359


namespace intersection_M_N_l163_16345

def M := {x : ℝ | x < 1}

def N := {y : ℝ | ∃ x : ℝ, y = Real.exp x}

theorem intersection_M_N : M ∩ N = {x : ℝ | 0 < x ∧ x < 1} :=
  sorry

end intersection_M_N_l163_16345


namespace largest_divisor_of_product_l163_16329

-- Definition of factorial
def factorial : ℕ → ℕ
| 0       => 1
| (n + 1) => (n + 1) * factorial n

-- Definition of P, the product of the visible numbers when an 8-sided die is rolled
def P (excluded: ℕ) : ℕ :=
  factorial 8 / excluded

-- The main theorem to prove
theorem largest_divisor_of_product (excluded: ℕ) (h₁: 1 ≤ excluded) (h₂: excluded ≤ 8): 
  ∃ n, n = 192 ∧ ∀ k, k > 192 → ¬k ∣ P excluded :=
sorry

end largest_divisor_of_product_l163_16329


namespace hyperbola_equation_l163_16349

theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
                           (h3 : b = 2 * a) (h4 : ((4 : ℝ), 1) ∈ {p : ℝ × ℝ | (p.1)^2 / (a^2) - (p.2)^2 / (b^2) = 1}) :
    {p : ℝ × ℝ | (p.1)^2 / 12 - (p.2)^2 / 3 = 1} = {p : ℝ × ℝ | (p.1)^2 / (a^2) - (p.2)^2 / (b^2) = 1} :=
by
  sorry

end hyperbola_equation_l163_16349


namespace remainder_of_72nd_integers_div_by_8_is_5_l163_16309

theorem remainder_of_72nd_integers_div_by_8_is_5 (s : Set ℤ) (h₁ : ∀ x ∈ s, ∃ k : ℤ, x = 8 * k + r) 
  (h₂ : 573 ∈ (s : Set ℤ)) : 
  ∃ (r : ℤ), r = 5 :=
by
  sorry

end remainder_of_72nd_integers_div_by_8_is_5_l163_16309


namespace obtuse_triangle_sum_range_l163_16308

variable (a b c : ℝ)

theorem obtuse_triangle_sum_range (h1 : b^2 + c^2 - a^2 = b * c)
                                   (h2 : a = (Real.sqrt 3) / 2)
                                   (h3 : (b * c) * (Real.cos (Real.pi - Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c)))) < 0) :
    (b + c) ∈ Set.Ioo ((Real.sqrt 3) / 2) (3 / 2) :=
sorry

end obtuse_triangle_sum_range_l163_16308


namespace cats_in_shelter_l163_16305

-- Define the initial conditions
def initial_cats := 20
def monday_addition := 2
def tuesday_addition := 1
def wednesday_subtraction := 3 * 2

-- Problem statement: Prove that the total number of cats after all events is 17
theorem cats_in_shelter : initial_cats + monday_addition + tuesday_addition - wednesday_subtraction = 17 :=
by
  sorry

end cats_in_shelter_l163_16305


namespace initial_food_supplies_l163_16376

theorem initial_food_supplies (x : ℝ) 
  (h1 : (3 / 5) * x - (3 / 5) * ((3 / 5) * x) = 96) : x = 400 :=
by
  sorry

end initial_food_supplies_l163_16376


namespace arithmetic_sequence_21st_term_l163_16352

theorem arithmetic_sequence_21st_term (a : ℕ → ℤ) (h1 : a 1 = 3) (h2 : a 2 = 13) (h3 : a 3 = 23) :
  a 21 = 203 :=
by
  sorry

end arithmetic_sequence_21st_term_l163_16352


namespace f_inv_f_inv_15_l163_16385

def f (x : ℝ) : ℝ := 3 * x + 6

noncomputable def f_inv (x : ℝ) : ℝ := (x - 6) / 3

theorem f_inv_f_inv_15 : f_inv (f_inv 15) = -1 :=
by
  sorry

end f_inv_f_inv_15_l163_16385


namespace find_n_l163_16307

theorem find_n (a : ℝ) (x : ℝ) (y : ℝ) (h1 : 0 < a) (h2 : a * x + 0.6 * a * y = 5 / 10)
(h3 : 1.6 * a * x + 1.2 * a * y = 1 - 1 / 10) : 
∃ n : ℕ, n = 10 :=
by
  sorry

end find_n_l163_16307


namespace length_of_base_l163_16363

-- Define the conditions of the problem
def base_of_triangle (b : ℕ) : Prop :=
  ∃ c : ℕ, b + 3 + c = 12 ∧ 9 + b*b = c*c

-- Statement to prove
theorem length_of_base : base_of_triangle 4 :=
  sorry

end length_of_base_l163_16363


namespace modulus_of_z_eq_sqrt2_l163_16327

noncomputable def complex_z : ℂ := (1 + 3 * Complex.I) / (2 - Complex.I)

theorem modulus_of_z_eq_sqrt2 : Complex.abs complex_z = Real.sqrt 2 := by
  sorry

end modulus_of_z_eq_sqrt2_l163_16327


namespace intersection_of_A_and_B_l163_16306

-- Definitions of sets A and B
def A : Set ℝ := { x | -2 < x ∧ x < 1 }
def B : Set ℝ := { x | 0 < x ∧ x < 2 }

-- Definition of the expected intersection of A and B
def expected_intersection : Set ℝ := { x | 0 < x ∧ x < 1 }

-- The main theorem stating the proof problem
theorem intersection_of_A_and_B :
  ∀ x : ℝ, x ∈ (A ∩ B) ↔ x ∈ expected_intersection :=
by
  intro x
  sorry

end intersection_of_A_and_B_l163_16306


namespace problem_statement_l163_16326

namespace CoinFlipping

/-- 
Define the probability that Alice and Bob both get the same number of heads
when flipping three coins where two are fair and one is biased with a probability
of 3/5 for heads. We aim to calculate p + q where p/q is this probability and 
output the final result - p + q should equal 263.
-/
def same_heads_probability_sum : ℕ :=
  let p := 63
  let q := 200
  p + q

theorem problem_statement : same_heads_probability_sum = 263 :=
  by
  -- proof to be filled in
  sorry

end CoinFlipping

end problem_statement_l163_16326


namespace find_x_l163_16354

theorem find_x (x : ℝ) (h : 3 * x = (20 - x) + 20) : x = 10 :=
sorry

end find_x_l163_16354


namespace max_value_is_63_l163_16370

noncomputable def max_value (x y : ℝ) : ℝ :=
  x^2 + 3*x*y + 4*y^2

theorem max_value_is_63 (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (cond : x^2 - 3*x*y + 4*y^2 = 9) :
  max_value x y ≤ 63 :=
by
  sorry

end max_value_is_63_l163_16370


namespace total_number_of_birds_l163_16351

theorem total_number_of_birds (B C G S W : ℕ) (h1 : C = 2 * B) (h2 : G = 4 * B)
  (h3 : S = (C + G) / 2) (h4 : W = 8) (h5 : B = 2 * W) :
  C + G + S + W + B = 168 :=
  by
  sorry

end total_number_of_birds_l163_16351


namespace number_of_girls_l163_16381

-- Define the problem conditions as constants
def total_saplings : ℕ := 44
def teacher_saplings : ℕ := 6
def boy_saplings : ℕ := 4
def girl_saplings : ℕ := 2
def total_students : ℕ := 12
def students_saplings : ℕ := total_saplings - teacher_saplings

-- The proof problem statement
theorem number_of_girls (x y : ℕ) (h1 : x + y = total_students)
  (h2 : boy_saplings * x + girl_saplings * y = students_saplings) :
  y = 5 :=
by
  sorry

end number_of_girls_l163_16381


namespace inverse_B2_l163_16341

def matrix_B_inv : Matrix (Fin 2) (Fin 2) ℝ := !![3, 7; -2, -4]

def matrix_B2_inv : Matrix (Fin 2) (Fin 2) ℝ := !![-5, -7; 2, 2]

theorem inverse_B2 (B : Matrix (Fin 2) (Fin 2) ℝ) (hB_inv : B⁻¹ = matrix_B_inv) :
  (B^2)⁻¹ = matrix_B2_inv :=
sorry

end inverse_B2_l163_16341


namespace find_xy_l163_16316

theorem find_xy (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 + y^2 = 58) : x * y = 21 :=
sorry

end find_xy_l163_16316


namespace caffeine_over_goal_l163_16356

theorem caffeine_over_goal (cups_per_day : ℕ) (mg_per_cup : ℕ) (caffeine_goal : ℕ) (total_cups : ℕ) :
  total_cups = 3 ->
  cups_per_day = 3 ->
  mg_per_cup = 80 ->
  caffeine_goal = 200 ->
  (cups_per_day * mg_per_cup) - caffeine_goal = 40 := by
  sorry

end caffeine_over_goal_l163_16356


namespace math_problem_l163_16336

theorem math_problem
  (x : ℕ) (y : ℕ)
  (h1 : x = (Finset.range (60 + 1 + 1) \ Finset.range 50).sum id)
  (h2 : y = ((Finset.range (60 + 1) \ Finset.range 50).filter (λ n => n % 2 = 0)).card)
  (h3 : x + y = 611) :
  (Finset.range (60 + 1 + 1) \ Finset.range 50).sum id = 605 ∧
  ((Finset.range (60 + 1) \ Finset.range 50).filter (λ n => n % 2 = 0)).card = 6 := 
by
  sorry

end math_problem_l163_16336


namespace bee_total_correct_l163_16328

def initial_bees : Nat := 16
def incoming_bees : Nat := 10
def total_bees : Nat := initial_bees + incoming_bees

theorem bee_total_correct : total_bees = 26 := by
  sorry

end bee_total_correct_l163_16328


namespace work_ratio_l163_16321

theorem work_ratio (r : ℕ) (w : ℕ) (m₁ m₂ d₁ d₂ : ℕ)
  (h₁ : m₁ = 5) 
  (h₂ : d₁ = 15) 
  (h₃ : m₂ = 3) 
  (h₄ : d₂ = 25)
  (h₅ : w = (m₁ * r * d₁) + (m₂ * r * d₂)) :
  ((m₁ * r * d₁):ℚ) / (w:ℚ) = 1 / 2 := by
  sorry

end work_ratio_l163_16321


namespace expected_babies_is_1008_l163_16300

noncomputable def babies_expected_after_loss
  (num_kettles : ℕ)
  (pregnancies_per_kettle : ℕ)
  (babies_per_pregnancy : ℕ)
  (loss_percentage : ℤ) : ℤ :=
  let total_babies := num_kettles * pregnancies_per_kettle * babies_per_pregnancy
  let survival_rate := (100 - loss_percentage) / 100
  total_babies * survival_rate

theorem expected_babies_is_1008 :
  babies_expected_after_loss 12 20 6 30 = 1008 :=
by
  sorry

end expected_babies_is_1008_l163_16300


namespace max_pawns_19x19_l163_16389

def maxPawnsOnChessboard (n : ℕ) := 
  n * n

theorem max_pawns_19x19 :
  maxPawnsOnChessboard 19 = 361 := 
by
  sorry

end max_pawns_19x19_l163_16389
