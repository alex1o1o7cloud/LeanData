import Mathlib

namespace flight_relation_not_preserved_l2378_237827

noncomputable def swap_city_flights (cities : Finset ℕ) (flights : ℕ → ℕ → Bool) : Prop := sorry

theorem flight_relation_not_preserved (cities : Finset ℕ) (flights : ℕ → ℕ → Bool) (M N : ℕ) (hM : M ∈ cities) (hN : N ∈ cities) : 
  ¬ swap_city_flights cities flights :=
sorry

end flight_relation_not_preserved_l2378_237827


namespace mark_bread_time_l2378_237825

def rise_time1 : Nat := 120
def rise_time2 : Nat := 120
def kneading_time : Nat := 10
def baking_time : Nat := 30

def total_time : Nat := rise_time1 + rise_time2 + kneading_time + baking_time

theorem mark_bread_time : total_time = 280 := by
  sorry

end mark_bread_time_l2378_237825


namespace original_rent_of_increased_friend_l2378_237839

theorem original_rent_of_increased_friend (avg_rent : ℝ) (new_avg_rent : ℝ) (num_friends : ℝ) (rent_increase_pct : ℝ)
  (total_old_rent : ℝ) (total_new_rent : ℝ) (increase_amount : ℝ) (R : ℝ) :
  avg_rent = 800 ∧ new_avg_rent = 850 ∧ num_friends = 4 ∧ rent_increase_pct = 0.16 ∧
  total_old_rent = num_friends * avg_rent ∧ total_new_rent = num_friends * new_avg_rent ∧
  increase_amount = total_new_rent - total_old_rent ∧ increase_amount = rent_increase_pct * R →
  R = 1250 :=
by
  sorry

end original_rent_of_increased_friend_l2378_237839


namespace intersection_A_B_l2378_237864

-- Define the sets A and B
def A : Set ℤ := {1, 3, 5, 7}
def B : Set ℤ := {x | 2 ≤ x ∧ x ≤ 5}

-- The goal is to prove that A ∩ B = {3, 5}
theorem intersection_A_B : A ∩ B = {3, 5} :=
  sorry

end intersection_A_B_l2378_237864


namespace proof_problem_statement_l2378_237870

noncomputable def proof_problem (x y: ℝ) : Prop :=
  x ≥ 1 ∧ y ≥ 1 ∧ (∀ n : ℕ, n > 0 → (⌊x / y⌋ : ℝ) = ⌊↑n * x⌋ / ⌊↑n * y⌋) →
  (x = y ∨ (∃ k : ℤ, k ≠ 0 ∧ (x = k * y ∨ y = k * x)))

-- The formal statement of the problem
theorem proof_problem_statement (x y : ℝ) :
  proof_problem x y := by
  sorry

end proof_problem_statement_l2378_237870


namespace moles_of_C2H6_formed_l2378_237834

-- Define the initial conditions
def initial_moles_H2 : ℕ := 3
def initial_moles_C2H4 : ℕ := 3
def reaction_ratio_C2H4_H2_C2H6 (C2H4 H2 C2H6 : ℕ) : Prop :=
  C2H4 = H2 ∧ C2H4 = C2H6

-- State the theorem to prove
theorem moles_of_C2H6_formed : reaction_ratio_C2H4_H2_C2H6 initial_moles_C2H4 initial_moles_H2 3 :=
by {
  sorry
}

end moles_of_C2H6_formed_l2378_237834


namespace books_shelved_in_fiction_section_l2378_237853

def calculate_books_shelved_in_fiction_section (total_books : ℕ) (remaining_books : ℕ) (books_shelved_in_history : ℕ) (books_shelved_in_children : ℕ) (books_added_back : ℕ) : ℕ :=
  let total_shelved := total_books - remaining_books
  let adjusted_books_shelved_in_children := books_shelved_in_children - books_added_back
  let total_shelved_in_history_and_children := books_shelved_in_history + adjusted_books_shelved_in_children
  total_shelved - total_shelved_in_history_and_children

theorem books_shelved_in_fiction_section:
  calculate_books_shelved_in_fiction_section 51 16 12 8 4 = 19 :=
by 
  -- Definition of the function gives the output directly so proof is trivial.
  rfl

end books_shelved_in_fiction_section_l2378_237853


namespace smallest_cookie_packages_l2378_237841

/-- The smallest number of cookie packages Zoey can buy in order to buy an equal number of cookie
and milk packages. -/
theorem smallest_cookie_packages (n : ℕ) (h1 : ∃ k : ℕ, 5 * k = 7 * n) : n = 7 :=
sorry

end smallest_cookie_packages_l2378_237841


namespace rectangle_diagonal_length_l2378_237821

theorem rectangle_diagonal_length (l : ℝ) (L W d : ℝ) 
  (h_ratio : L = 5 * l ∧ W = 2 * l)
  (h_perimeter : 2 * (L + W) = 100) :
  d = (5 * Real.sqrt 290) / 7 :=
by
  sorry

end rectangle_diagonal_length_l2378_237821


namespace fourth_number_unit_digit_l2378_237835

def unit_digit (n : ℕ) : ℕ := n % 10

theorem fourth_number_unit_digit (a b c d : ℕ) (h₁ : a = 7858) (h₂: b = 1086) (h₃ : c = 4582) (h₄ : unit_digit (a * b * c * d) = 8) :
  unit_digit d = 4 :=
sorry

end fourth_number_unit_digit_l2378_237835


namespace age_of_25th_student_l2378_237877

variable (total_students : ℕ) (total_average : ℕ)
variable (group1_students : ℕ) (group1_average : ℕ)
variable (group2_students : ℕ) (group2_average : ℕ)

theorem age_of_25th_student 
  (h1 : total_students = 25) 
  (h2 : total_average = 25)
  (h3 : group1_students = 10)
  (h4 : group1_average = 22)
  (h5 : group2_students = 14)
  (h6 : group2_average = 28) : 
  (total_students * total_average) =
  (group1_students * group1_average) + (group2_students * group2_average) + 13 :=
by sorry

end age_of_25th_student_l2378_237877


namespace max_g6_l2378_237814

noncomputable def g (x : ℝ) : ℝ :=
sorry

theorem max_g6 :
  (∀ x, (g x = a * x^2 + b * x + c) ∧ (a ≥ 0) ∧ (b ≥ 0) ∧ (c ≥ 0)) →
  (g 3 = 3) →
  (g 9 = 243) →
  (g 6 ≤ 6) :=
sorry

end max_g6_l2378_237814


namespace exists_f_m_eq_n_plus_2017_l2378_237881

theorem exists_f_m_eq_n_plus_2017 (m : ℕ) (h : m > 0) :
  (∃ f : ℤ → ℤ, ∀ n : ℤ, (f^[m] n = n + 2017)) ↔ (m = 1 ∨ m = 2017) :=
by
  sorry

end exists_f_m_eq_n_plus_2017_l2378_237881


namespace rational_solutions_k_l2378_237842

theorem rational_solutions_k (k : ℕ) (h : k > 0) : (∃ x : ℚ, 2 * (k : ℚ) * x^2 + 36 * x + 3 * (k : ℚ) = 0) → k = 6 :=
by
  -- proof to be written
  sorry

end rational_solutions_k_l2378_237842


namespace prove_sum_l2378_237829

theorem prove_sum (a b : ℝ) (h1 : a * (a - 4) = 12) (h2 : b * (b - 4) = 12) (h3 : a ≠ b) : a + b = 4 := by
  sorry

end prove_sum_l2378_237829


namespace q_is_false_given_conditions_l2378_237812

theorem q_is_false_given_conditions
  (h₁: ¬(p ∧ q) = true) 
  (h₂: ¬¬p = true) 
  : q = false := 
sorry

end q_is_false_given_conditions_l2378_237812


namespace number_of_planks_needed_l2378_237800

-- Definitions based on conditions
def bed_height : ℕ := 2
def bed_width : ℕ := 2
def bed_length : ℕ := 8
def plank_width : ℕ := 1
def lumber_length : ℕ := 8
def num_beds : ℕ := 10

-- The theorem statement
theorem number_of_planks_needed : (2 * (bed_length / lumber_length) * bed_height) + (2 * ((bed_width * bed_height) / lumber_length) * lumber_length / 4) * num_beds = 60 :=
  by sorry

end number_of_planks_needed_l2378_237800


namespace B_and_C_together_l2378_237816

-- Defining the variables and conditions
variable (A B C : ℕ)
variable (h1 : A + B + C = 500)
variable (h2 : A + C = 200)
variable (h3 : C = 50)

-- The theorem to prove that B + C = 350
theorem B_and_C_together : B + C = 350 :=
by
  -- Replacing with the actual proof steps
  sorry

end B_and_C_together_l2378_237816


namespace toads_max_l2378_237888

theorem toads_max (n : ℕ) (h₁ : n ≥ 3) : 
  ∃ k : ℕ, k = ⌈ (n : ℝ) / 2 ⌉ ∧ ∀ (labels : Fin n → Fin n) (jumps : Fin n → ℕ), 
  (∀ i, jumps (labels i) = labels i) → ¬ ∃ f : Fin k → Fin n, ∀ i₁ i₂, i₁ ≠ i₂ → f i₁ ≠ f i₂ :=
sorry

end toads_max_l2378_237888


namespace solution_to_functional_equation_l2378_237844

noncomputable def find_functions (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f x = f (x / 2) + (x / 2) * (deriv f x)

theorem solution_to_functional_equation :
  ∀ (f : ℝ → ℝ), (∀ x : ℝ, f x = f (x / 2) + (x / 2) * (deriv f x)) ↔ (∃ c b : ℝ, ∀ x : ℝ, f x = c * x + b) :=
by {
  sorry
}

end solution_to_functional_equation_l2378_237844


namespace total_students_in_class_l2378_237898

theorem total_students_in_class (B G : ℕ) (h1 : G = 160) (h2 : 5 * G = 8 * B) : B + G = 260 :=
by
  -- Proof steps would go here
  sorry

end total_students_in_class_l2378_237898


namespace value_of_expression_l2378_237807

theorem value_of_expression (x : ℝ) (h : 7 * x^2 - 2 * x - 4 = 4 * x + 11) : 
  (5 * x - 7)^2 = 11.63265306 := 
by 
  sorry

end value_of_expression_l2378_237807


namespace framed_painting_ratio_l2378_237886

def painting_width := 20
def painting_height := 30

def smaller_dimension := painting_width + 2 * 5
def larger_dimension := painting_height + 4 * 5

noncomputable def ratio := (smaller_dimension : ℚ) / (larger_dimension : ℚ)

theorem framed_painting_ratio :
  ratio = 3 / 5 :=
by
  sorry

end framed_painting_ratio_l2378_237886


namespace angela_age_in_5_years_l2378_237830

-- Define the variables representing Angela and Beth's ages.
variable (A B : ℕ)

-- State the conditions as hypotheses.
def condition_1 : Prop := A = 4 * B
def condition_2 : Prop := (A - 5) + (B - 5) = 45

-- State the final proposition that Angela will be 49 years old in five years.
theorem angela_age_in_5_years (h1 : condition_1 A B) (h2 : condition_2 A B) : A + 5 = 49 := by
  sorry

end angela_age_in_5_years_l2378_237830


namespace rectangle_circle_area_ratio_l2378_237824

noncomputable def area_ratio (w r : ℝ) (h : 3 * w = Real.pi * r) : ℝ :=
  (2 * w^2) / (Real.pi * r^2)

theorem rectangle_circle_area_ratio (w r : ℝ) (h : 3 * w = Real.pi * r) :
  area_ratio w r h = 18 / (Real.pi * Real.pi) :=
by
  sorry

end rectangle_circle_area_ratio_l2378_237824


namespace senior_students_in_sample_l2378_237861

theorem senior_students_in_sample 
  (total_students : ℕ) (total_seniors : ℕ) (sample_size : ℕ) 
  (h1 : total_students = 2000)
  (h2 : total_seniors = 500)
  (h3 : sample_size = 200) : 
  (total_seniors * sample_size / total_students = 50) :=
by {
  sorry
}

end senior_students_in_sample_l2378_237861


namespace nina_weeks_to_afford_game_l2378_237883

noncomputable def game_cost : ℝ := 50
noncomputable def sales_tax_rate : ℝ := 0.10
noncomputable def weekly_allowance : ℝ := 10
noncomputable def saving_rate : ℝ := 0.5

noncomputable def total_cost : ℝ := game_cost + (game_cost * sales_tax_rate)
noncomputable def savings_per_week : ℝ := weekly_allowance * saving_rate
noncomputable def weeks_needed : ℝ := total_cost / savings_per_week

theorem nina_weeks_to_afford_game : weeks_needed = 11 := by
  sorry

end nina_weeks_to_afford_game_l2378_237883


namespace fraction_zero_solution_l2378_237885

theorem fraction_zero_solution (x : ℝ) (h : (x - 1) / (2 - x) = 0) : x = 1 :=
sorry

end fraction_zero_solution_l2378_237885


namespace product_of_possible_values_of_x_l2378_237811

theorem product_of_possible_values_of_x : 
  (∀ x, |x - 7| - 5 = 4 → x = 16 ∨ x = -2) -> (16 * -2 = -32) :=
by
  intro h
  have := h 16
  have := h (-2)
  sorry

end product_of_possible_values_of_x_l2378_237811


namespace original_sheets_count_is_115_l2378_237805

def find_sheets_count (S P : ℕ) : Prop :=
  -- Ann's condition: all papers are used leaving 100 flyers
  S - P = 100 ∧
  -- Bob's condition: all bindings used leaving 35 sheets of paper
  5 * P = S - 35

theorem original_sheets_count_is_115 (S P : ℕ) (h : find_sheets_count S P) : S = 115 :=
by
  sorry

end original_sheets_count_is_115_l2378_237805


namespace smallest_k_l2378_237833

-- Define the non-decreasing property of digits in a five-digit number
def non_decreasing (n : Fin 5 → ℕ) : Prop :=
  n 0 ≤ n 1 ∧ n 1 ≤ n 2 ∧ n 2 ≤ n 3 ∧ n 3 ≤ n 4

-- Define the overlap property in at least one digit
def overlap (n1 n2 : Fin 5 → ℕ) : Prop :=
  ∃ i : Fin 5, n1 i = n2 i

-- The main theorem stating the problem
theorem smallest_k {N1 Nk : Fin 5 → ℕ} :
  (∀ n : Fin 5 → ℕ, non_decreasing n → overlap N1 n ∨ overlap Nk n) → 
  ∃ (k : Nat), k = 2 :=
sorry

end smallest_k_l2378_237833


namespace find_c_l2378_237804

theorem find_c (c : ℝ) :
  (∃ (infinitely_many_y : ℝ → Prop), (∀ y, infinitely_many_y y ↔ 3 * (5 + 2 * c * y) = 18 * y + 15))
  → c = 3 :=
by
  sorry

end find_c_l2378_237804


namespace jennifer_fruits_left_l2378_237832

open Nat

theorem jennifer_fruits_left :
  (p o a g : ℕ) → p = 10 → o = 20 → a = 2 * p → g = 2 → (p - g) + (o - g) + (a - g) = 44 :=
by
  intros p o a g h_p h_o h_a h_g
  rw [h_p, h_o, h_a, h_g]
  sorry

end jennifer_fruits_left_l2378_237832


namespace sally_has_more_cards_l2378_237813

def SallyInitial : ℕ := 27
def DanTotal : ℕ := 41
def SallyBought : ℕ := 20
def SallyTotal := SallyInitial + SallyBought

theorem sally_has_more_cards : SallyTotal - DanTotal = 6 := by
  sorry

end sally_has_more_cards_l2378_237813


namespace total_plums_l2378_237868

def alyssa_plums : Nat := 17
def jason_plums : Nat := 10

theorem total_plums : alyssa_plums + jason_plums = 27 := 
by
  -- proof goes here
  sorry

end total_plums_l2378_237868


namespace find_length_AD_l2378_237884

-- Given data and conditions
def triangle_ABC (A B C D : Type) : Prop := sorry
def angle_bisector_AD (A B C D : Type) : Prop := sorry
def length_BD : ℝ := 40
def length_BC : ℝ := 45
def length_AC : ℝ := 36

-- Prove that AD = 320 units
theorem find_length_AD (A B C D : Type)
  (h1 : triangle_ABC A B C D)
  (h2 : angle_bisector_AD A B C D)
  (h3 : length_BD = 40)
  (h4 : length_BC = 45)
  (h5 : length_AC = 36) :
  ∃ x : ℝ, x = 320 :=
sorry

end find_length_AD_l2378_237884


namespace monthly_rent_calculation_l2378_237840

noncomputable def monthly_rent (purchase_cost : ℕ) (maintenance_pct : ℝ) (annual_taxes : ℕ) (target_roi : ℝ) : ℝ :=
  let annual_return := target_roi * (purchase_cost : ℝ)
  let total_annual_requirement := annual_return + (annual_taxes : ℝ)
  let monthly_requirement := total_annual_requirement / 12
  let actual_rent := monthly_requirement / (1 - maintenance_pct)
  actual_rent

theorem monthly_rent_calculation :
  monthly_rent 12000 0.15 400 0.06 = 109.80 :=
by
  sorry

end monthly_rent_calculation_l2378_237840


namespace purely_imaginary_value_of_m_third_quadrant_value_of_m_l2378_237859

theorem purely_imaginary_value_of_m (m : ℝ) :
  (2 * m^2 - 3 * m - 2 = 0) ∧ (m^2 - 2 * m ≠ 0) → m = -1/2 :=
by
  sorry

theorem third_quadrant_value_of_m (m : ℝ) :
  (2 * m^2 - 3 * m - 2 < 0) ∧ (m^2 - 2 * m < 0) → 0 < m ∧ m < 2 :=
by
  sorry

end purely_imaginary_value_of_m_third_quadrant_value_of_m_l2378_237859


namespace min_value_g_geq_6_min_value_g_eq_6_l2378_237889

noncomputable def g (x : ℝ) : ℝ :=
  x + (x / (x^2 + 2)) + (x * (x + 5) / (x^2 + 3)) + (3 * (x + 3) / (x * (x^2 + 3)))

theorem min_value_g_geq_6 : ∀ x > 0, g x ≥ 6 :=
by
  sorry

theorem min_value_g_eq_6 : ∃ x > 0, g x = 6 :=
by
  sorry

end min_value_g_geq_6_min_value_g_eq_6_l2378_237889


namespace growth_rate_inequality_l2378_237831

theorem growth_rate_inequality (a b x : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) (h_x_pos : x > 0) :
  x ≤ (a + b) / 2 :=
sorry

end growth_rate_inequality_l2378_237831


namespace circle_equation_l2378_237855

theorem circle_equation :
  ∃ r : ℝ, ∀ x y : ℝ,
  ((x - 2) * (x - 2) + (y - 1) * (y - 1) = r * r) ∧
  ((5 - 2) * (5 - 2) + (-2 - 1) * (-2 - 1) = r * r) ∧
  (5 + 2 * -2 - 5 + r * r = 0) :=
sorry

end circle_equation_l2378_237855


namespace p_true_of_and_not_p_false_l2378_237828

variable {p q : Prop}

theorem p_true_of_and_not_p_false (h1 : ¬(p ∧ q)) (h2 : ¬¬p) : p :=
sorry

end p_true_of_and_not_p_false_l2378_237828


namespace range_of_a_for_p_range_of_a_for_p_and_q_l2378_237880

variable (a : ℝ)

/-- For any x ∈ ℝ, ax^2 - x + 3 > 0 if and only if a > 1/12 -/
def condition_p : Prop := ∀ x : ℝ, a * x^2 - x + 3 > 0

/-- There exists x ∈ [1, 2] such that 2^x * a ≥ 1 -/
def condition_q : Prop := ∃ x : ℝ, 1 ≤ x ∧ x ≤ 2 ∧ a * 2^x ≥ 1

/-- Theorem (1): The range of values for a such that condition_p holds true is (1/12, +∞) -/
theorem range_of_a_for_p (h : condition_p a) : a > 1/12 :=
sorry

/-- Theorem (2): The range of values for a such that condition_p and condition_q have different truth values is (1/12, 1/4) -/
theorem range_of_a_for_p_and_q (h₁ : condition_p a) (h₂ : ¬condition_q a) : 1/12 < a ∧ a < 1/4 :=
sorry

end range_of_a_for_p_range_of_a_for_p_and_q_l2378_237880


namespace base_729_base8_l2378_237801

theorem base_729_base8 (b : ℕ) (X Y : ℕ) (h_distinct : X ≠ Y)
  (h_range : b^3 ≤ 729 ∧ 729 < b^4)
  (h_form : 729 = X * b^3 + Y * b^2 + X * b + Y) : b = 8 :=
sorry

end base_729_base8_l2378_237801


namespace remainder_of_13_plus_x_mod_29_l2378_237815

theorem remainder_of_13_plus_x_mod_29
  (x : ℕ)
  (hx : 8 * x ≡ 1 [MOD 29])
  (hp : 0 < x) : 
  (13 + x) % 29 = 18 :=
sorry

end remainder_of_13_plus_x_mod_29_l2378_237815


namespace symmetric_line_condition_l2378_237893

theorem symmetric_line_condition (x y : ℝ) :
  (∀ x y : ℝ, x - 2 * y - 3 = 0 → -y + 2 * x - 3 = 0) →
  (∀ x y : ℝ, x + y = 0 → ∃ a b c : ℝ, 2 * x - y - 3 = 0) :=
sorry

end symmetric_line_condition_l2378_237893


namespace pickup_carries_10_bags_per_trip_l2378_237846

def total_weight : ℕ := 10000
def weight_one_bag : ℕ := 50
def number_of_trips : ℕ := 20
def total_bags : ℕ := total_weight / weight_one_bag
def bags_per_trip : ℕ := total_bags / number_of_trips

theorem pickup_carries_10_bags_per_trip : bags_per_trip = 10 := by
  sorry

end pickup_carries_10_bags_per_trip_l2378_237846


namespace smallest_positive_integer_a_l2378_237817

noncomputable def isPerfectSquare (x : ℕ) : Prop :=
  ∃ (k : ℕ), k * k = x

theorem smallest_positive_integer_a :
  ∃ (a : ℕ), 0 < a ∧ (isPerfectSquare (10 + a)) ∧ (isPerfectSquare (10 * a)) ∧ 
  ∀ b : ℕ, 0 < b ∧ (isPerfectSquare (10 + b)) ∧ (isPerfectSquare (10 * b)) → a ≤ b :=
sorry

end smallest_positive_integer_a_l2378_237817


namespace weight_of_hollow_golden_sphere_l2378_237862

theorem weight_of_hollow_golden_sphere : 
  let diameter := 12
  let thickness := 0.3
  let pi := (3 : Real)
  let outer_radius := diameter / 2
  let inner_radius := (outer_radius - thickness)
  let outer_volume := (4 / 3) * pi * outer_radius^3
  let inner_volume := (4 / 3) * pi * inner_radius^3
  let gold_volume := outer_volume - inner_volume
  let weight_per_cubic_inch := 1
  let weight := gold_volume * weight_per_cubic_inch
  weight = 123.23 :=
by
  sorry

end weight_of_hollow_golden_sphere_l2378_237862


namespace trigonometric_identity_l2378_237820

theorem trigonometric_identity (α : Real) (h : Real.tan (α / 2) = 4) :
    (6 * Real.sin α - 7 * Real.cos α + 1) / (8 * Real.sin α + 9 * Real.cos α - 1) = -85 / 44 := by
  sorry

end trigonometric_identity_l2378_237820


namespace parity_equivalence_l2378_237875

def p_q_parity_condition (p q : ℕ) : Prop :=
  (p^3 - q^3) % 2 = 0 ↔ (p + q) % 2 = 0

theorem parity_equivalence (p q : ℕ) : p_q_parity_condition p q :=
by sorry

end parity_equivalence_l2378_237875


namespace always_possible_to_rotate_disks_l2378_237843

def labels_are_distinct (a : Fin 20 → ℕ) : Prop :=
  ∀ i j : Fin 20, i ≠ j → a i ≠ a j

def opposite_position (i : Fin 20) (r : Fin 20) : Fin 20 :=
  (i + r) % 20

def no_identical_numbers_opposite (a b : Fin 20 → ℕ) (r : Fin 20) : Prop :=
  ∀ i : Fin 20, a i ≠ b (opposite_position i r)

theorem always_possible_to_rotate_disks (a b : Fin 20 → ℕ) :
  labels_are_distinct a →
  labels_are_distinct b →
  ∃ r : Fin 20, no_identical_numbers_opposite a b r :=
sorry

end always_possible_to_rotate_disks_l2378_237843


namespace area_of_region_bounded_by_lines_and_y_axis_l2378_237837

noncomputable def area_of_triangle_bounded_by_lines : ℝ :=
  let y1 (x : ℝ) := 3 * x - 6
  let y2 (x : ℝ) := -2 * x + 18
  let intersection_x := 24 / 5
  let intersection_y := y1 intersection_x
  let base := 18 + 6
  let height := intersection_x
  1 / 2 * base * height

theorem area_of_region_bounded_by_lines_and_y_axis :
  area_of_triangle_bounded_by_lines = 57.6 :=
by
  sorry

end area_of_region_bounded_by_lines_and_y_axis_l2378_237837


namespace box_internal_volume_in_cubic_feet_l2378_237863

def box_length := 26 -- inches
def box_width := 26 -- inches
def box_height := 14 -- inches
def wall_thickness := 1 -- inch

def external_volume := box_length * box_width * box_height -- cubic inches
def internal_length := box_length - 2 * wall_thickness
def internal_width := box_width - 2 * wall_thickness
def internal_height := box_height - 2 * wall_thickness
def internal_volume := internal_length * internal_width * internal_height -- cubic inches

def cubic_inches_to_cubic_feet (v : ℕ) : ℕ := v / 1728

theorem box_internal_volume_in_cubic_feet : cubic_inches_to_cubic_feet internal_volume = 4 := by
  sorry

end box_internal_volume_in_cubic_feet_l2378_237863


namespace maximum_ratio_is_2_plus_2_sqrt2_l2378_237895

noncomputable def C1_polar_eq (θ : ℝ) : Prop :=
  ∀ ρ : ℝ, ρ * (Real.cos θ + Real.sin θ) = 1

noncomputable def C2_polar_eq (θ : ℝ) : Prop :=
  ∀ ρ : ℝ, ρ = 4 * Real.cos θ

theorem maximum_ratio_is_2_plus_2_sqrt2 (α : ℝ) (hα : 0 ≤ α ∧ α ≤ Real.pi / 2) :
  ∃ ρA ρB : ℝ, (ρA = 1 / (Real.cos α + Real.sin α)) ∧ (ρB = 4 * Real.cos α) ∧ 
  (4 * Real.cos α * (Real.cos α + Real.sin α) = 2 + 2 * Real.sqrt 2) :=
sorry

end maximum_ratio_is_2_plus_2_sqrt2_l2378_237895


namespace total_games_l2378_237851

variable (Ken_games Dave_games Jerry_games : ℕ)

-- The conditions from the problem.
def condition1 : Prop := Ken_games = Dave_games + 5
def condition2 : Prop := Dave_games = Jerry_games + 3
def condition3 : Prop := Jerry_games = 7

-- The final statement to prove
theorem total_games (h1 : condition1 Ken_games Dave_games) 
                    (h2 : condition2 Dave_games Jerry_games) 
                    (h3 : condition3 Jerry_games) : 
  Ken_games + Dave_games + Jerry_games = 32 :=
by
  sorry

end total_games_l2378_237851


namespace solution_to_inequality_system_l2378_237887

theorem solution_to_inequality_system :
  (∀ x : ℝ, 2 * (x - 1) < x + 2 → (x + 1) / 2 < x → 1 < x ∧ x < 4) :=
by
  intros x h1 h2
  sorry

end solution_to_inequality_system_l2378_237887


namespace value_of_x_l2378_237803

theorem value_of_x (x c m n : ℝ) (hne: m≠n) (hneq : c ≠ 0) 
  (h1: c = 3) (h2: m = 2) (h3: n = 5)
  (h4: (x + c * m)^2 - (x + c * n)^2 = (m - n)^2) : 
  x = -11 := by
  sorry

end value_of_x_l2378_237803


namespace cat_food_finished_on_sunday_l2378_237882

def cat_morning_consumption : ℚ := 1 / 2
def cat_evening_consumption : ℚ := 1 / 3
def total_food : ℚ := 10
def daily_consumption : ℚ := cat_morning_consumption + cat_evening_consumption
def days_to_finish_food (total_food daily_consumption : ℚ) : ℚ :=
  total_food / daily_consumption

theorem cat_food_finished_on_sunday :
  days_to_finish_food total_food daily_consumption = 7 := 
sorry

end cat_food_finished_on_sunday_l2378_237882


namespace largest_divisor_of_n_l2378_237899

theorem largest_divisor_of_n (n : ℕ) (h1 : 0 < n) (h2 : 127 ∣ n^3) : 127 ∣ n :=
sorry

end largest_divisor_of_n_l2378_237899


namespace person_B_winning_strategy_l2378_237836

-- Definitions for the problem conditions
def winning_strategy_condition (L a b : ℕ) : Prop := 
  b = 2 * a ∧ ∃ k : ℕ, L = k * a

-- Lean theorem statement for the given problem
theorem person_B_winning_strategy (L a b : ℕ) (hL_pos : 0 < L) (ha_lt_hb : a < b) 
(hpos_a : 0 < a) (hpos_b : 0 < b) : 
  (∃ B_strat : Type, winning_strategy_condition L a b) :=
sorry

end person_B_winning_strategy_l2378_237836


namespace remainder_when_divided_by_6_l2378_237897

theorem remainder_when_divided_by_6 (n : ℕ) (h₁ : n = 482157)
  (odd_n : n % 2 ≠ 0) (div_by_3 : n % 3 = 0) : n % 6 = 3 :=
by
  -- Proof goes here
  sorry

end remainder_when_divided_by_6_l2378_237897


namespace b_joined_after_x_months_l2378_237892

-- Establish the given conditions as hypotheses
theorem b_joined_after_x_months
  (a_start_capital : ℝ)
  (b_start_capital : ℝ)
  (profit_ratio : ℝ)
  (months_in_year : ℝ)
  (a_capital_time : ℝ)
  (b_capital_time : ℝ)
  (a_profit_ratio : ℝ)
  (b_profit_ratio : ℝ)
  (x : ℝ)
  (h1 : a_start_capital = 3500)
  (h2 : b_start_capital = 9000)
  (h3 : profit_ratio = 2 / 3)
  (h4 : months_in_year = 12)
  (h5 : a_capital_time = 12)
  (h6 : b_capital_time = 12 - x)
  (h7 : a_profit_ratio = 2)
  (h8 : b_profit_ratio = 3)
  (h_ratio : (a_start_capital * a_capital_time) / (b_start_capital * b_capital_time) = profit_ratio) :
  x = 5 :=
by
  sorry

end b_joined_after_x_months_l2378_237892


namespace Jean_had_41_candies_at_first_l2378_237852

-- Let total_candies be the initial number of candies Jean had
variable (total_candies : ℕ)
-- Jean gave 18 pieces to a friend
def given_away := 18
-- Jean ate 7 pieces
def eaten := 7
-- Jean has 16 pieces left now
def remaining := 16

-- Calculate the total number of candies initially
def candy_initial (total_candies given_away eaten remaining : ℕ) : Prop :=
  total_candies = remaining + (given_away + eaten)

-- Prove that Jean had 41 pieces of candy initially
theorem Jean_had_41_candies_at_first : candy_initial 41 given_away eaten remaining :=
by
  -- Skipping the proof for now
  sorry

end Jean_had_41_candies_at_first_l2378_237852


namespace chiming_time_is_5_l2378_237823

-- Define the conditions for the clocks
def queen_strikes (h : ℕ) : Prop := (2 * h) % 3 = 0
def king_strikes (h : ℕ) : Prop := (3 * h) % 2 = 0

-- Define the chiming synchronization at the same time condition
def chiming_synchronization (h: ℕ) : Prop :=
  3 * h = 2 * ((2 * h) + 2)

-- The proof statement
theorem chiming_time_is_5 : ∃ h: ℕ, queen_strikes h ∧ king_strikes h ∧ chiming_synchronization h ∧ h = 5 :=
by
  sorry

end chiming_time_is_5_l2378_237823


namespace range_of_k_l2378_237857

theorem range_of_k (k : ℝ) :
  (∀ x : ℤ, ((x^2 - x - 2 > 0) ∧ (2*x^2 + (2*k + 5)*x + 5*k < 0)) ↔ (x = -2)) -> 
  (-3 ≤ k ∧ k < 2) :=
by 
  sorry

end range_of_k_l2378_237857


namespace zoe_pictures_l2378_237819

theorem zoe_pictures (pictures_taken : ℕ) (dolphin_show_pictures : ℕ)
  (h1 : pictures_taken = 28) (h2 : dolphin_show_pictures = 16) :
  pictures_taken + dolphin_show_pictures = 44 :=
sorry

end zoe_pictures_l2378_237819


namespace min_distance_l2378_237802

noncomputable def f (x : ℝ) : ℝ := x^2
noncomputable def g (x : ℝ) : ℝ := Real.log x

theorem min_distance :
  ∃ m : ℝ, (∀ x > 0, x ≠ m → (f m - g m) ≤ (f x - g x)) ∧ m = Real.sqrt 2 / 2 :=
by
  sorry

end min_distance_l2378_237802


namespace sales_discount_percentage_l2378_237894

theorem sales_discount_percentage :
  ∀ (P N : ℝ) (D : ℝ),
  (N * 1.12 * (P * (1 - D / 100)) = P * N * (1 + 0.008)) → D = 10 :=
by
  intros P N D h
  sorry

end sales_discount_percentage_l2378_237894


namespace compute_fraction_l2378_237850

noncomputable def distinct_and_sum_zero (w x y z : ℝ) : Prop :=
w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z ∧ w + x + y + z = 0

theorem compute_fraction (w x y z : ℝ) (h : distinct_and_sum_zero w x y z) :
  (w * y + x * z) / (w^2 + x^2 + y^2 + z^2) = -1 / 2 :=
sorry

end compute_fraction_l2378_237850


namespace find_smaller_integer_l2378_237849

theorem find_smaller_integer : ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ y = x + 8 ∧ x * y = 80 ∧ x = 2 :=
by
  sorry

end find_smaller_integer_l2378_237849


namespace theoretical_yield_H2SO4_l2378_237806

-- Define the theoretical yield calculation problem in terms of moles of reactions and products
theorem theoretical_yield_H2SO4 
  (moles_SO3 : ℝ) (moles_H2O : ℝ) 
  (reaction : moles_SO3 + moles_H2O = 2.0 + 1.5) 
  (limiting_reactant_H2O : moles_H2O = 1.5) : 
  1.5 = moles_H2O * 1 :=
  sorry

end theoretical_yield_H2SO4_l2378_237806


namespace largest_number_l2378_237856

theorem largest_number (A B C D E : ℝ) (hA : A = 0.998) (hB : B = 0.9899) (hC : C = 0.9) (hD : D = 0.9989) (hE : E = 0.8999) :
  D > A ∧ D > B ∧ D > C ∧ D > E :=
by
  sorry

end largest_number_l2378_237856


namespace problem_part1_problem_part2_l2378_237872

open Complex

noncomputable def E1 := ((1 + I)^2 / (1 + 2 * I)) + ((1 - I)^2 / (2 - I))

theorem problem_part1 : E1 = (6 / 5) - (2 / 5) * I :=
by
  sorry

theorem problem_part2 (x y : ℝ) (h1 : (x / 2) + (y / 5) = 1) (h2 : (x / 2) + (2 * y / 5) = 3) : x = -2 ∧ y = 10 :=
by
  sorry

end problem_part1_problem_part2_l2378_237872


namespace find_real_solutions_l2378_237810

theorem find_real_solutions (x : ℝ) :
  x^4 + (3 - x)^4 = 146 ↔ x = 1.5 + Real.sqrt 3.4175 ∨ x = 1.5 - Real.sqrt 3.4175 :=
by
  sorry

end find_real_solutions_l2378_237810


namespace even_function_l2378_237845

theorem even_function (f : ℝ → ℝ) (not_zero : ∃ x, f x ≠ 0) 
  (h : ∀ a b : ℝ, f (a + b) + f (a - b) = 2 * f a + 2 * f b) : 
  ∀ x : ℝ, f (-x) = f x := 
sorry

end even_function_l2378_237845


namespace minimum_groups_l2378_237809

theorem minimum_groups (students : ℕ) (max_group_size : ℕ) (h_students : students = 30) (h_max_group_size : max_group_size = 12) : 
  ∃ least_groups : ℕ, least_groups = 3 :=
by
  sorry

end minimum_groups_l2378_237809


namespace transform_polynomial_l2378_237876

open Real

variable {x y : ℝ}

theorem transform_polynomial 
  (h1 : y = x + 1 / x) 
  (h2 : x^4 + x^3 - 4 * x^2 + x + 1 = 0) : 
  x^2 * (y^2 + y - 6) = 0 := 
sorry

end transform_polynomial_l2378_237876


namespace solve_for_s_l2378_237891

theorem solve_for_s (s : ℝ) (t : ℝ) (h1 : t = 8 * s^2) (h2 : t = 4.8) : s = Real.sqrt 0.6 ∨ s = -Real.sqrt 0.6 := by
  sorry

end solve_for_s_l2378_237891


namespace total_cost_is_correct_l2378_237896

-- Define the price of pizzas
def pizza_price : ℕ := 5

-- Define the count of triple cheese and meat lovers pizzas
def triple_cheese_pizzas : ℕ := 10
def meat_lovers_pizzas : ℕ := 9

-- Define the special offers
def buy1get1free (count : ℕ) : ℕ := count / 2 + count % 2
def buy2get1free (count : ℕ) : ℕ := (count / 3) * 2 + count % 3

-- Define the cost calculations using the special offers
def cost_triple_cheese : ℕ := buy1get1free triple_cheese_pizzas * pizza_price
def cost_meat_lovers : ℕ := buy2get1free meat_lovers_pizzas * pizza_price

-- Define the total cost calculation
def total_cost : ℕ := cost_triple_cheese + cost_meat_lovers

-- The theorem we need to prove
theorem total_cost_is_correct :
  total_cost = 55 := by
  sorry

end total_cost_is_correct_l2378_237896


namespace trihedral_angle_plane_angles_acute_l2378_237838

open Real

-- Define what it means for an angle to be acute
def is_acute (θ : ℝ) : Prop :=
  0 < θ ∧ θ < π / 2

-- Define the given conditions
variable {A B C α β γ : ℝ}
variable (hA : is_acute A)
variable (hB : is_acute B)
variable (hC : is_acute C)

-- State the problem: if dihedral angles are acute, then plane angles are also acute
theorem trihedral_angle_plane_angles_acute :
  is_acute A → is_acute B → is_acute C → is_acute α ∧ is_acute β ∧ is_acute γ :=
sorry

end trihedral_angle_plane_angles_acute_l2378_237838


namespace new_rope_length_l2378_237867

-- Define the given constants and conditions
def rope_length_initial : ℝ := 12
def additional_area : ℝ := 1511.7142857142858
noncomputable def pi_approx : ℝ := Real.pi

-- Define the proof statement
theorem new_rope_length :
  let r2 := Real.sqrt ((additional_area / pi_approx) + rope_length_initial ^ 2)
  r2 = 25 :=
by
  -- Placeholder for the proof
  sorry

end new_rope_length_l2378_237867


namespace sin_cos_bounds_l2378_237873

theorem sin_cos_bounds (w x y z : ℝ)
  (hw : -Real.pi / 2 ≤ w ∧ w ≤ Real.pi / 2)
  (hx : -Real.pi / 2 ≤ x ∧ x ≤ Real.pi / 2)
  (hy : -Real.pi / 2 ≤ y ∧ y ≤ Real.pi / 2)
  (hz : -Real.pi / 2 ≤ z ∧ z ≤ Real.pi / 2)
  (h₁ : Real.sin w + Real.sin x + Real.sin y + Real.sin z = 1)
  (h₂ : Real.cos (2 * w) + Real.cos (2 * x) + Real.cos (2 * y) + Real.cos (2 * z) ≥ 10 / 3) :
  0 ≤ w ∧ w ≤ Real.pi / 6 ∧ 0 ≤ x ∧ x ≤ Real.pi / 6 ∧ 0 ≤ y ∧ y ≤ Real.pi / 6 ∧ 0 ≤ z ∧ z ≤ Real.pi / 6 :=
by
  sorry

end sin_cos_bounds_l2378_237873


namespace loan_amounts_l2378_237865

theorem loan_amounts (x y : ℝ) (h1 : x + y = 50) (h2 : 0.1 * x + 0.08 * y = 4.4) : x = 20 ∧ y = 30 := by
  sorry

end loan_amounts_l2378_237865


namespace B_current_age_l2378_237826

theorem B_current_age (A B : ℕ) (h1 : A = B + 15) (h2 : A - 5 = 2 * (B - 5)) : B = 20 :=
by sorry

end B_current_age_l2378_237826


namespace min_value_of_f_l2378_237869

noncomputable def f (x : ℝ) : ℝ := (4 * x^2 - 2 * x + 16) / (2 * x - 1)

theorem min_value_of_f :
  ∃ x : ℝ, x ≥ 1 ∧ f x = 9 ∧ (∀ y : ℝ, y ≥ 1 → f y ≥ 9) :=
by { sorry }

end min_value_of_f_l2378_237869


namespace distance_between_stripes_l2378_237818

/-- Define the parallel curbs and stripes -/
structure Crosswalk where
  distance_between_curbs : ℝ
  curb_distance_between_stripes : ℝ
  stripe_length : ℝ
  stripe_cross_distance : ℝ
  
open Crosswalk

/-- Conditions given in the problem -/
def crosswalk : Crosswalk where
  distance_between_curbs := 60 -- feet
  curb_distance_between_stripes := 20 -- feet
  stripe_length := 50 -- feet
  stripe_cross_distance := 50 -- feet

/-- Theorem to prove the distance between stripes -/
theorem distance_between_stripes (cw : Crosswalk) :
  2 * (cw.curb_distance_between_stripes * cw.distance_between_curbs) / cw.stripe_length = 24 := sorry

end distance_between_stripes_l2378_237818


namespace arithmetic_series_sum_correct_l2378_237860

-- Define the parameters of the arithmetic series
def a : ℤ := -53
def l : ℤ := 3
def d : ℤ := 2

-- Define the number of terms in the series
def n : ℕ := 29

-- The expected sum of the series
def expected_sum : ℤ := -725

-- Define the nth term formula
noncomputable def nth_term (a d : ℤ) (n : ℕ) : ℤ :=
  a + (n - 1) * d

-- Define the sum of the arithmetic series
noncomputable def arithmetic_series_sum (a l : ℤ) (n : ℕ) : ℤ :=
  (n * (a + l)) / 2

-- Statement of the proof problem
theorem arithmetic_series_sum_correct :
  arithmetic_series_sum a l n = expected_sum := by
  sorry

end arithmetic_series_sum_correct_l2378_237860


namespace area_rectangle_l2378_237866

theorem area_rectangle 
    (x y : ℝ)
    (h1 : 5 * x + 4 * y = 10)
    (h2 : 3 * x = 2 * y) :
    5 * (x * y) = 3000 / 121 :=
by
  sorry

end area_rectangle_l2378_237866


namespace garden_width_min_5_l2378_237822

theorem garden_width_min_5 (width length : ℝ) (h_length : length = width + 20) (h_area : width * length ≥ 150) :
  width ≥ 5 :=
sorry

end garden_width_min_5_l2378_237822


namespace polynomial_satisfies_conditions_l2378_237808

noncomputable def f (x y z : ℝ) : ℝ := (x^2 - y^3) * (y^3 - z^6) * (z^6 - x^2)

theorem polynomial_satisfies_conditions :
  (∀ x y z : ℝ, f x (z^2) y + f x (y^2) z = 0) ∧ 
  (∀ x y z : ℝ, f (z^3) y x + f (x^3) y z = 0) :=
by
  sorry

end polynomial_satisfies_conditions_l2378_237808


namespace initial_number_of_men_l2378_237871

theorem initial_number_of_men (M : ℕ) (h1 : ∃ food : ℕ, food = M * 22) (h2 : ∀ food, food = (M * 20)) (h3 : ∃ food : ℕ, food = ((M + 40) * 19)) : M = 760 := by
  sorry

end initial_number_of_men_l2378_237871


namespace determine_d_l2378_237878

theorem determine_d (a b c d : ℝ) (h : a^2 + b^2 + c^2 + 2 = d + (a + b + c - d)^(1/3)) : d = 1/2 := by
  sorry

end determine_d_l2378_237878


namespace mb_less_than_neg_one_point_five_l2378_237890

theorem mb_less_than_neg_one_point_five (m b : ℚ) (h1 : m = 3/4) (h2 : b = -2) : m * b < -1.5 :=
by {
  -- sorry skips the proof
  sorry
}

end mb_less_than_neg_one_point_five_l2378_237890


namespace profitable_year_exists_option2_more_economical_l2378_237847

noncomputable def total_expenses (x : ℕ) : ℝ := 2 * (x:ℝ)^2 + 10 * x  

noncomputable def annual_income (x : ℕ) : ℝ := 50 * x  

def year_profitable (x : ℕ) : Prop := annual_income x > total_expenses x + 98 / 1000

theorem profitable_year_exists : ∃ x : ℕ, year_profitable x ∧ x = 3 := sorry

noncomputable def total_profit (x : ℕ) : ℝ := 
  50 * x - 2 * (x:ℝ)^2 + 10 * x - 98 / 1000 + if x = 10 then 8 else if x = 7 then 26 else 0

theorem option2_more_economical : 
  total_profit 10 = 110 ∧ total_profit 7 = 110 ∧ 7 < 10 :=
sorry

end profitable_year_exists_option2_more_economical_l2378_237847


namespace GCD_is_six_l2378_237879

-- Define the numbers
def a : ℕ := 36
def b : ℕ := 60
def c : ℕ := 90

-- Define the GCD using Lean's gcd function
def GCD_abc : ℕ := Nat.gcd (Nat.gcd a b) c

-- State the theorem that GCD of 36, 60, and 90 is 6
theorem GCD_is_six : GCD_abc = 6 := by
  sorry -- Proof skipped

end GCD_is_six_l2378_237879


namespace candy_bar_price_l2378_237854

theorem candy_bar_price (total_money bread_cost candy_bar_price remaining_money : ℝ) 
    (h1 : total_money = 32)
    (h2 : bread_cost = 3)
    (h3 : remaining_money = 18)
    (h4 : total_money - bread_cost - candy_bar_price - (1 / 3) * (total_money - bread_cost - candy_bar_price) = remaining_money) :
    candy_bar_price = 1.33 := 
sorry

end candy_bar_price_l2378_237854


namespace equation_of_parametrized_curve_l2378_237874

theorem equation_of_parametrized_curve :
  ∀ t : ℝ, let x := 3 * t + 6 
           let y := 5 * t - 8 
           ∃ (m b : ℝ), y = m * x + b ∧ m = 5 / 3 ∧ b = -18 :=
by
  sorry

end equation_of_parametrized_curve_l2378_237874


namespace vertex_of_quadratic_l2378_237848

theorem vertex_of_quadratic (x : ℝ) : 
  (y : ℝ) = -2 * (x + 1) ^ 2 + 3 →
  (∃ vertex_x vertex_y : ℝ, vertex_x = -1 ∧ vertex_y = 3 ∧ y = -2 * (vertex_x + 1) ^ 2 + vertex_y) :=
by
  intro h
  exists -1, 3
  simp [h]
  sorry

end vertex_of_quadratic_l2378_237848


namespace angle_bisector_b_c_sum_l2378_237858

theorem angle_bisector_b_c_sum (A B C : ℝ × ℝ)
  (hA : A = (4, -3))
  (hB : B = (-6, 21))
  (hC : C = (10, 7)) :
  ∃ b c : ℝ, (3 * x + b * y + c = 0) ∧ (b + c = correct_answer) :=
by
  sorry

end angle_bisector_b_c_sum_l2378_237858
