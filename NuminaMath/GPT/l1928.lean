import Mathlib

namespace rate_per_kg_grapes_l1928_192886

/-- Define the conditions for the problem -/
def rate_per_kg_mangoes : ℕ := 55
def kg_grapes_purchased : ℕ := 3
def kg_mangoes_purchased : ℕ := 9
def total_paid : ℕ := 705

/-- The theorem statement to prove the rate per kg for grapes -/
theorem rate_per_kg_grapes (G : ℕ) :
  kg_grapes_purchased * G + kg_mangoes_purchased * rate_per_kg_mangoes = total_paid →
  G = 70 :=
by
  sorry -- Proof will go here

end rate_per_kg_grapes_l1928_192886


namespace sin_315_equals_minus_sqrt2_div_2_l1928_192865

theorem sin_315_equals_minus_sqrt2_div_2 :
  Real.sin (315 * Real.pi / 180) = - Real.sqrt 2 / 2 := by
  sorry

end sin_315_equals_minus_sqrt2_div_2_l1928_192865


namespace diameter_of_lake_l1928_192883

theorem diameter_of_lake (d : ℝ) (pi : ℝ) (h1 : pi = 3.14) 
  (h2 : 3.14 * d - d = 1.14) : d = 0.5327 :=
by
  sorry

end diameter_of_lake_l1928_192883


namespace lowest_possible_sale_price_percentage_l1928_192814

def list_price : ℝ := 80
def initial_discount : ℝ := 0.5
def additional_discount : ℝ := 0.2

theorem lowest_possible_sale_price_percentage 
  (list_price : ℝ) (initial_discount : ℝ) (additional_discount : ℝ) :
  ( (list_price - (list_price * initial_discount)) - (list_price * additional_discount) ) / list_price * 100 = 30 :=
by
  sorry

end lowest_possible_sale_price_percentage_l1928_192814


namespace sum_of_numbers_l1928_192812

noncomputable def sum_two_numbers (x y : ℝ) : ℝ :=
  x + y

theorem sum_of_numbers (x y : ℝ) (h1 : x * y = 16) (h2 : (1 / x) = 3 * (1 / y)) : 
  sum_two_numbers x y = (16 * Real.sqrt 3) / 3 := 
by 
  sorry

end sum_of_numbers_l1928_192812


namespace ones_digit_largest_power_of_2_divides_32_factorial_ones_digit_largest_power_of_3_divides_32_factorial_l1928_192831

theorem ones_digit_largest_power_of_2_divides_32_factorial : 
  (2^31 % 10) = 8 := 
by
  sorry

theorem ones_digit_largest_power_of_3_divides_32_factorial : 
  (3^14 % 10) = 9 := 
by
  sorry

end ones_digit_largest_power_of_2_divides_32_factorial_ones_digit_largest_power_of_3_divides_32_factorial_l1928_192831


namespace river_depth_l1928_192899

theorem river_depth (V : ℝ) (W : ℝ) (F : ℝ) (D : ℝ) 
  (hV : V = 10666.666666666666) 
  (hW : W = 40) 
  (hF : F = 66.66666666666667) 
  (hV_eq : V = W * D * F) : 
  D = 4 :=
by sorry

end river_depth_l1928_192899


namespace pizza_slices_with_all_three_toppings_l1928_192832

theorem pizza_slices_with_all_three_toppings : 
  ∀ (a b c d e f g : ℕ), 
  a + b + c + d + e + f + g = 24 ∧ 
  a + d + e + g = 12 ∧ 
  b + d + f + g = 15 ∧ 
  c + e + f + g = 10 → 
  g = 5 := 
by {
  sorry
}

end pizza_slices_with_all_three_toppings_l1928_192832


namespace population_in_2060_l1928_192840

noncomputable def population (year : ℕ) : ℕ :=
  if h : (year - 2000) % 20 = 0 then
    250 * 2 ^ ((year - 2000) / 20)
  else
    0 -- This handles non-multiples of 20 cases, which are irrelevant here

theorem population_in_2060 : population 2060 = 2000 := by
  sorry

end population_in_2060_l1928_192840


namespace sum_of_x_coordinates_of_other_vertices_l1928_192898

theorem sum_of_x_coordinates_of_other_vertices {x1 y1 x2 y2 x3 y3 x4 y4: ℝ} 
    (h1 : (x1, y1) = (2, 12))
    (h2 : (x2, y2) = (8, 3))
    (midpoint_eq : (x1 + x2) / 2 = (x3 + x4) / 2) 
    : x3 + x4 = 10 := 
by
  have h4 : (2 + 8) / 2 = 5 := by norm_num
  have h5 : 2 * 5 = 10 := by norm_num
  sorry

end sum_of_x_coordinates_of_other_vertices_l1928_192898


namespace total_pies_l1928_192863

def apple_Pies (totalApples : ℕ) (applesPerPie : ℕ) (piesPerBatch : ℕ) : ℕ :=
  (totalApples / applesPerPie) * piesPerBatch

def pear_Pies (totalPears : ℕ) (pearsPerPie : ℕ) (piesPerBatch : ℕ) : ℕ :=
  (totalPears / pearsPerPie) * piesPerBatch

theorem total_pies :
  let apples : ℕ := 27
  let pears : ℕ := 30
  let applesPerPie : ℕ := 9
  let pearsPerPie : ℕ := 15
  let applePiesPerBatch : ℕ := 2
  let pearPiesPerBatch : ℕ := 3
  apple_Pies apples applesPerPie applePiesPerBatch + pear_Pies pears pearsPerPie pearPiesPerBatch = 12 :=
by
  sorry

end total_pies_l1928_192863


namespace find_P_l1928_192866

theorem find_P 
  (digits : Finset ℕ) 
  (h_digits : digits = {1, 2, 3, 4, 5, 6}) 
  (P Q R S T U : ℕ)
  (h_unique : P ∈ digits ∧ Q ∈ digits ∧ R ∈ digits ∧ S ∈ digits ∧ T ∈ digits ∧ U ∈ digits ∧ 
              P ≠ Q ∧ P ≠ R ∧ P ≠ S ∧ P ≠ T ∧ P ≠ U ∧
              Q ≠ R ∧ Q ≠ S ∧ Q ≠ T ∧ Q ≠ U ∧
              R ≠ S ∧ R ≠ T ∧ R ≠ U ∧ 
              S ≠ T ∧ S ≠ U ∧ 
              T ≠ U) 
  (h_div5 : (100 * P + 10 * Q + R) % 5 = 0)
  (h_div3 : (100 * Q + 10 * R + S) % 3 = 0)
  (h_div2 : (100 * R + 10 * S + T) % 2 = 0) :
  P = 2 :=
sorry

end find_P_l1928_192866


namespace find_a_l1928_192826

noncomputable def f (a x : ℝ) := a * Real.exp x + 2 * x^2

noncomputable def f' (a x : ℝ) := a * Real.exp x + 4 * x

theorem find_a (a : ℝ) (h : f' a 0 = 2) : a = 2 :=
by
  unfold f' at h
  simp at h
  exact h

end find_a_l1928_192826


namespace sequence_ninth_term_l1928_192837

theorem sequence_ninth_term (a b : ℚ) :
  ∀ n : ℕ, n = 9 → (-1 : ℚ) ^ n * (n * b ^ n) / ((n + 1) * a ^ (n + 2)) = -9 * b^9 / (10 * a^11) :=
by
  sorry

end sequence_ninth_term_l1928_192837


namespace calculate_f_zero_l1928_192878

noncomputable def f (ω φ x : ℝ) := Real.sin (ω * x + φ)

theorem calculate_f_zero
  (ω φ : ℝ)
  (h_inc : ∀ x y : ℝ, (π / 6 < x ∧ x < y ∧ y < 2 * π / 3) → f ω φ x < f ω φ y)
  (h_symmetry1 : ∀ x : ℝ, f ω φ (π / 6 - x) = f ω φ (π / 6 + x))
  (h_symmetry2 : ∀ x : ℝ, f ω φ (2 * π / 3 - x) = f ω φ (2 * π / 3 + x)) :
  f ω φ 0 = -1 / 2 :=
sorry

end calculate_f_zero_l1928_192878


namespace glasses_in_smaller_box_l1928_192838

variable (x : ℕ)

theorem glasses_in_smaller_box (h : (x + 16) / 2 = 15) : x = 14 :=
by
  sorry

end glasses_in_smaller_box_l1928_192838


namespace division_quotient_remainder_l1928_192855

theorem division_quotient_remainder :
  ∃ (q r : ℝ), 76.6 = 1.8 * q + r ∧ 0 ≤ r ∧ r < 1.8 ∧ q = 42 ∧ r = 1 := by
  sorry

end division_quotient_remainder_l1928_192855


namespace add_to_fraction_eq_l1928_192801

theorem add_to_fraction_eq (n : ℤ) : (3 + n : ℚ) / (5 + n) = 5 / 6 → n = 7 := 
by
  sorry

end add_to_fraction_eq_l1928_192801


namespace find_number_l1928_192869

theorem find_number (x : ℝ) : 60 + (x * 12) / (180 / 3) = 61 ↔ x = 5 := by
  sorry  -- proof can be filled in here when needed

end find_number_l1928_192869


namespace eleven_y_minus_x_l1928_192857

theorem eleven_y_minus_x (x y : ℤ) (h1 : x = 7 * y + 3) (h2 : 2 * x = 18 * y + 2) : 11 * y - x = 1 := by
  sorry

end eleven_y_minus_x_l1928_192857


namespace find_acute_angle_l1928_192808

theorem find_acute_angle (α : ℝ) (h1 : 0 < α) (h2 : α < π / 2) 
    (h3 : Real.sin α = 1 - Real.sqrt 3 * Real.tan (π / 18) * Real.sin α) : 
    α = π / 3 * 5 / 9 :=
by
  sorry

end find_acute_angle_l1928_192808


namespace bruno_initial_books_l1928_192817

theorem bruno_initial_books :
  ∃ (B : ℕ), B - 4 + 10 = 39 → B = 33 :=
by
  use 33
  intro h
  linarith [h]

end bruno_initial_books_l1928_192817


namespace find_b_from_root_and_constant_l1928_192853

theorem find_b_from_root_and_constant
  (b k : ℝ)
  (h₁ : k = 44)
  (h₂ : ∃ (x : ℝ), x = 4 ∧ 2*x^2 + b*x - k = 0) :
  b = 3 :=
by
  sorry

end find_b_from_root_and_constant_l1928_192853


namespace set_intersection_l1928_192839

open Set

variable (U : Set ℝ)
variable (A B : Set ℝ)

def complement (s : Set ℝ) := {x : ℝ | x ∉ s}

theorem set_intersection (hU : U = univ)
                         (hA : A = {x : ℝ | x > 0})
                         (hB : B = {x : ℝ | x > 1}) :
  A ∩ complement B = {x : ℝ | 0 < x ∧ x ≤ 1} :=
by
  sorry

end set_intersection_l1928_192839


namespace sqrt_expression_l1928_192805

theorem sqrt_expression : 
  (Real.sqrt 75 + Real.sqrt 8 - Real.sqrt 18 - Real.sqrt 6 * Real.sqrt 2 = 3 * Real.sqrt 3 - Real.sqrt 2) := 
by 
  sorry

end sqrt_expression_l1928_192805


namespace lcm_150_456_l1928_192892

theorem lcm_150_456 : Nat.lcm 150 456 = 11400 := by
  sorry

end lcm_150_456_l1928_192892


namespace davids_profit_l1928_192846

-- Definitions of conditions
def weight_of_rice : ℝ := 50
def cost_of_rice : ℝ := 50
def selling_price_per_kg : ℝ := 1.20

-- Theorem stating the expected profit
theorem davids_profit : 
  (selling_price_per_kg * weight_of_rice) - cost_of_rice = 10 := 
by 
  -- Proofs are omitted.
  sorry

end davids_profit_l1928_192846


namespace buffalo_theft_l1928_192836

theorem buffalo_theft (initial_apples falling_apples remaining_apples stolen_apples : ℕ)
  (h1 : initial_apples = 79)
  (h2 : falling_apples = 26)
  (h3 : remaining_apples = 8) :
  initial_apples - falling_apples - stolen_apples = remaining_apples ↔ stolen_apples = 45 :=
by sorry

end buffalo_theft_l1928_192836


namespace geometric_sum_S6_l1928_192828

variable {a : ℕ → ℝ} (S : ℕ → ℝ)

-- Conditions: S_n represents the sum of the first n terms of the geometric sequence {a_n}
-- and we have S_2 = 4 and S_4 = 6
theorem geometric_sum_S6 (S : ℕ → ℝ) (h1 : S 2 = 4) (h2 : S 4 = 6) : S 6 = 7 :=
sorry

end geometric_sum_S6_l1928_192828


namespace intersection_of_M_and_N_l1928_192881

noncomputable def setM : Set ℝ := { x : ℝ | x^2 - 2 * x - 3 < 0 }
noncomputable def setN : Set ℝ := { x : ℝ | Real.log x / Real.log 2 < 1 }

theorem intersection_of_M_and_N : { x : ℝ | x ∈ setM ∧ x ∈ setN } = { x : ℝ | 0 < x ∧ x < 2 } :=
by
  sorry

end intersection_of_M_and_N_l1928_192881


namespace sample_size_calculation_l1928_192809

theorem sample_size_calculation (n : ℕ) (ratio_A_B_C q_A q_B q_C : ℕ) 
  (ratio_condition : ratio_A_B_C = 2 ∧ ratio_A_B_C * q_A = 2 ∧ ratio_A_B_C * q_B = 3 ∧ ratio_A_B_C * q_C = 5)
  (sample_A_units : q_A = 16) : n = 80 :=
sorry

end sample_size_calculation_l1928_192809


namespace unique_solution_l1928_192848

theorem unique_solution (a b x: ℝ) : 
  (4 * x - 7 + a = (b - 1) * x + 2) ↔ (b ≠ 5) := 
by
  sorry -- proof is omitted as per instructions

end unique_solution_l1928_192848


namespace palabras_bookstore_workers_l1928_192887

theorem palabras_bookstore_workers (W : ℕ) (h1 : W / 2 = (W / 2)) (h2 : W / 6 = (W / 6)) (h3 : 12 = 12) (h4 : W - (W / 2 + W / 6 - 12 + 1) = 35) : W = 210 := 
sorry

end palabras_bookstore_workers_l1928_192887


namespace two_times_six_pow_n_plus_one_ne_product_of_consecutive_l1928_192834

theorem two_times_six_pow_n_plus_one_ne_product_of_consecutive (n k : ℕ) :
  2 * (6 ^ n + 1) ≠ k * (k + 1) :=
sorry

end two_times_six_pow_n_plus_one_ne_product_of_consecutive_l1928_192834


namespace sufficient_but_not_necessary_l1928_192880

theorem sufficient_but_not_necessary (x : ℝ) : (x^2 = 9 → x = 3) ∧ (¬(x^2 = 9 → x = 3 ∨ x = -3)) :=
by
  sorry

end sufficient_but_not_necessary_l1928_192880


namespace total_collection_l1928_192852

theorem total_collection (n : ℕ) (c : ℕ) (h_n : n = 88) (h_c : c = 88) : 
  (n * c / 100 : ℚ) = 77.44 :=
by
  sorry

end total_collection_l1928_192852


namespace probability_club_then_queen_l1928_192895

theorem probability_club_then_queen : 
  let total_cards := 52
  let total_clubs := 13
  let total_queens := 4
  let queen_of_clubs := 1
  let non_queen_clubs := total_clubs - queen_of_clubs
  
  let prob_queen_of_clubs_then_other_queen := (queen_of_clubs / total_cards) * ((total_queens - 1) / (total_cards - 1))
  let prob_non_queen_clubs_then_queen := (non_queen_clubs / total_cards) * (total_queens / (total_cards - 1))
  let total_probability := prob_queen_of_clubs_then_other_queen + prob_non_queen_clubs_then_queen
  
  total_probability = 1 / 52 := by
  let total_cards := 52
  let total_clubs := 13
  let total_queens := 4
  let queen_of_clubs := 1
  let non_queen_clubs := total_clubs - queen_of_clubs
  
  let prob_queen_of_clubs_then_other_queen := (queen_of_clubs / total_cards) * ((total_queens - 1) / (total_cards - 1))
  let prob_non_queen_clubs_then_queen := (non_queen_clubs / total_cards) * (total_queens / (total_cards - 1))
  let total_probability := prob_queen_of_clubs_then_other_queen + prob_non_queen_clubs_then_queen
  
  sorry

end probability_club_then_queen_l1928_192895


namespace diane_head_start_l1928_192821

theorem diane_head_start (x : ℝ) :
  (100 - 11.91) / (88.09 + x) = 99.25 / 100 ->
  abs (x - 12.68) < 0.01 := 
by
  sorry

end diane_head_start_l1928_192821


namespace omar_remaining_coffee_l1928_192810

noncomputable def remaining_coffee : ℝ := 
  let initial_coffee := 12
  let after_first_drink := initial_coffee - (initial_coffee * 1/4)
  let after_office_drink := after_first_drink - (after_first_drink * 1/3)
  let espresso_in_ounces := 75 / 29.57
  let after_espresso := after_office_drink + espresso_in_ounces
  let after_lunch_drink := after_espresso - (after_espresso * 0.75)
  let iced_tea_addition := 4 * 1/2
  let after_iced_tea := after_lunch_drink + iced_tea_addition
  let after_cold_drink := after_iced_tea - (after_iced_tea * 0.6)
  after_cold_drink

theorem omar_remaining_coffee : remaining_coffee = 1.654 :=
by 
  sorry

end omar_remaining_coffee_l1928_192810


namespace cube_difference_l1928_192884

variables (a b : ℝ)  -- Specify the variables a and b are real numbers

theorem cube_difference (h1 : a - b = 3) (h2 : a^2 + b^2 = 27) : a^3 - b^3 = 108 :=
by
  -- Skip the proof as requested.
  sorry

end cube_difference_l1928_192884


namespace eccentricity_range_of_ellipse_l1928_192894

theorem eccentricity_range_of_ellipse
  (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b)
  (e : ℝ) (he1 : e > 0) (he2 : e < 1)
  (h_directrix : 2 * (a / e) ≤ 3 * (2 * a)) :
  (1 / 3) ≤ e ∧ e < 1 := 
sorry

end eccentricity_range_of_ellipse_l1928_192894


namespace expected_people_with_condition_l1928_192845

noncomputable def proportion_of_condition := 1 / 3
def total_population := 450

theorem expected_people_with_condition :
  (proportion_of_condition * total_population) = 150 := by
  sorry

end expected_people_with_condition_l1928_192845


namespace unreasonable_inference_l1928_192818

theorem unreasonable_inference:
  (∀ (S T : Type) (P : S → Prop) (Q : T → Prop), (∀ x y, P x → ¬ Q y) → ¬ (∀ x, P x) → (∃ y, ¬ Q y))
  ∧ ¬ (∀ s : ℝ, (s = 100) → ∀ t : ℝ, t = 100) :=
sorry

end unreasonable_inference_l1928_192818


namespace range_of_b_l1928_192893

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := Real.exp x * (x*x - b*x)

theorem range_of_b (b : ℝ) : 
  (∀ x ∈ Set.Icc (1/2 : ℝ) 2, 0 < (Real.exp x * ((x*x + (2 - b) * x - b)))) →
  b < 8/3 := 
sorry

end range_of_b_l1928_192893


namespace students_called_in_sick_l1928_192872

-- Conditions
def total_cupcakes : ℕ := 2 * 12 + 6
def total_people : ℕ := 27 + 1 + 1
def cupcakes_left : ℕ := 4
def cupcakes_given_out : ℕ := total_cupcakes - cupcakes_left

-- Statement to prove
theorem students_called_in_sick : total_people - cupcakes_given_out = 3 := by
  -- The proof steps would be implemented here
  sorry

end students_called_in_sick_l1928_192872


namespace sqrt_7_estimate_l1928_192819

theorem sqrt_7_estimate (h1 : 4 < 7) (h2 : 7 < 9) (h3 : Nat.sqrt 4 = 2) (h4 : Nat.sqrt 9 = 3) : 2 < Real.sqrt 7 ∧ Real.sqrt 7 < 3 :=
  by {
    -- the proof would go here, but use 'sorry' to omit it
    sorry
  }

end sqrt_7_estimate_l1928_192819


namespace general_form_of_line_l_l1928_192800

-- Define the point
def pointA : ℝ × ℝ := (1, 2)

-- Define the normal vector
def normalVector : ℝ × ℝ := (1, -3)

-- Define the general form equation
def generalFormEq (x y : ℝ) : Prop := x - 3 * y + 5 = 0

-- Statement to prove
theorem general_form_of_line_l (x y : ℝ) (h_pointA : pointA = (1, 2)) (h_normalVector : normalVector = (1, -3)) :
  generalFormEq x y :=
sorry

end general_form_of_line_l_l1928_192800


namespace relationship_between_ys_l1928_192823

theorem relationship_between_ys :
  ∀ (y1 y2 y3 : ℝ),
    (y1 = - (6 / (-2))) ∧ (y2 = - (6 / (-1))) ∧ (y3 = - (6 / 3)) →
    y2 > y1 ∧ y1 > y3 :=
by sorry

end relationship_between_ys_l1928_192823


namespace find_ordered_pairs_l1928_192811

theorem find_ordered_pairs :
  {p : ℝ × ℝ | p.1 > p.2 ∧ (p.1 - p.2 = 2 * p.1 / p.2 ∨ p.1 - p.2 = 2 * p.2 / p.1)} = 
  {(8, 4), (9, 3), (2, 1)} :=
sorry

end find_ordered_pairs_l1928_192811


namespace a_minus_b_l1928_192822

noncomputable def find_a_b (a b : ℝ) :=
  ∃ k : ℝ, ∀ (x : ℝ) (y : ℝ), 
    (k = 2 + a) ∧ 
    (y = k * x + 1) ∧ 
    (y = x^2 + a * x + b) ∧ 
    (x = 1) ∧ (y = 3)

theorem a_minus_b (a b : ℝ) (h : find_a_b a b) : a - b = -2 := by 
  sorry

end a_minus_b_l1928_192822


namespace largest_angle_in_triangle_l1928_192835

noncomputable def angle_sum : ℝ := 120 -- $\frac{4}{3}$ of 90 degrees
noncomputable def angle_difference : ℝ := 20

theorem largest_angle_in_triangle :
  ∃ (a b c : ℝ), a + b + c = 180 ∧ a + b = angle_sum ∧ b = a + angle_difference ∧
  max a (max b c) = 70 :=
by
  sorry

end largest_angle_in_triangle_l1928_192835


namespace sin_double_angle_l1928_192833

theorem sin_double_angle (α : ℝ) 
  (h1 : Real.cos (α + Real.pi / 4) = 3 / 5)
  (h2 : Real.pi / 2 ≤ α ∧ α ≤ 3 * Real.pi / 2) : 
  Real.sin (2 * α) = 7 / 25 := 
by sorry

end sin_double_angle_l1928_192833


namespace triangle_ratio_and_angle_l1928_192856

variable (A B C : ℝ)
variable (a b c : ℝ)
variable (sinA sinB sinC : ℝ)

theorem triangle_ratio_and_angle
  (h_triangle : a / sinA = b / sinB ∧ b / sinB = c / sinC)
  (h_sin_ratio : sinA / sinB = 5 / 7 ∧ sinB / sinC = 7 / 8) :
  (a / b = 5 / 7 ∧ b / c = 7 / 8) ∧ B = 60 :=
by
  sorry

end triangle_ratio_and_angle_l1928_192856


namespace neither_necessary_nor_sufficient_condition_l1928_192851

def red_balls := 5
def yellow_balls := 3
def white_balls := 2
def total_balls := red_balls + yellow_balls + white_balls

def event_A_occurs := ∃ (r : ℕ) (y : ℕ), (r ≤ red_balls) ∧ (y ≤ yellow_balls) ∧ (r = 1) ∧ (y = 1)
def event_B_occurs := ∃ (x y : ℕ), (x ≤ total_balls) ∧ (y ≤ total_balls) ∧ (x ≠ y)

theorem neither_necessary_nor_sufficient_condition :
  ¬(¬event_A_occurs → ¬event_B_occurs) ∧ ¬(¬event_B_occurs → ¬event_A_occurs) := 
sorry

end neither_necessary_nor_sufficient_condition_l1928_192851


namespace cost_per_mile_l1928_192820

theorem cost_per_mile 
    (round_trip_distance : ℝ)
    (num_days : ℕ)
    (total_cost : ℝ) 
    (h1 : round_trip_distance = 200 * 2)
    (h2 : num_days = 7)
    (h3 : total_cost = 7000) 
  : (total_cost / (round_trip_distance * num_days) = 2.5) :=
by
  sorry

end cost_per_mile_l1928_192820


namespace painter_time_remaining_l1928_192843

theorem painter_time_remaining (total_rooms : ℕ) (time_per_room : ℕ) (rooms_painted : ℕ) (remaining_hours : ℕ)
  (h1 : total_rooms = 12) (h2 : time_per_room = 7) (h3 : rooms_painted = 5) 
  (h4 : remaining_hours = (total_rooms - rooms_painted) * time_per_room) : 
  remaining_hours = 49 :=
by
  sorry

end painter_time_remaining_l1928_192843


namespace total_cups_needed_l1928_192829

theorem total_cups_needed (cereal_servings : ℝ) (milk_servings : ℝ) (nuts_servings : ℝ) 
  (cereal_cups_per_serving : ℝ) (milk_cups_per_serving : ℝ) (nuts_cups_per_serving : ℝ) : 
  cereal_servings = 18.0 ∧ milk_servings = 12.0 ∧ nuts_servings = 6.0 ∧ 
  cereal_cups_per_serving = 2.0 ∧ milk_cups_per_serving = 1.5 ∧ nuts_cups_per_serving = 0.5 → 
  (cereal_servings * cereal_cups_per_serving + milk_servings * milk_cups_per_serving + 
   nuts_servings * nuts_cups_per_serving) = 57.0 :=
by
  sorry

end total_cups_needed_l1928_192829


namespace range_of_m_l1928_192816

def A (m : ℝ) : Set ℝ := {x | x^2 - 4*m*x + 2*m + 6 = 0}
def B : Set ℝ := {x | x < 0}

theorem range_of_m (m : ℝ) (h : (A m) ∩ B ≠ ∅) : m ≤ -1 :=
sorry

end range_of_m_l1928_192816


namespace remainder_n_squared_plus_3n_plus_5_l1928_192874

theorem remainder_n_squared_plus_3n_plus_5 (n : ℕ) (h : n % 25 = 24) : (n^2 + 3 * n + 5) % 25 = 3 :=
by
  sorry

end remainder_n_squared_plus_3n_plus_5_l1928_192874


namespace cards_exchanged_l1928_192815

theorem cards_exchanged (x : ℕ) (h : x * (x - 1) = 1980) : x * (x - 1) = 1980 :=
by sorry

end cards_exchanged_l1928_192815


namespace max_choir_members_l1928_192825

theorem max_choir_members (n : ℕ) (x y : ℕ) : 
  n = x^2 + 11 ∧ n = y * (y + 3) → n = 54 :=
by
  sorry

end max_choir_members_l1928_192825


namespace find_coefficients_l1928_192868

theorem find_coefficients
  (a b c : ℝ)
  (hA : ∀ x : ℝ, (x = -3 ∨ x = 4) ↔ (x^2 + a * x - 12 = 0))
  (hB : ∀ x : ℝ, (x = -3 ∨ x = 1) ↔ (x^2 + b * x + c = 0))
  (hAnotB : ¬ (∀ x, (x^2 + a * x - 12 = 0) ↔ (x^2 + b * x + c = 0)))
  (hA_inter_B : ∀ x, x = -3 ↔ (x^2 + a * x - 12 = 0) ∧ (x^2 + b * x + c = 0))
  (hA_union_B : ∀ x, (x = -3 ∨ x = 1 ∨ x = 4) ↔ (x^2 + a * x - 12 = 0) ∨ (x^2 + b * x + c = 0)):
  a = -1 ∧ b = 2 ∧ c = -3 :=
sorry

end find_coefficients_l1928_192868


namespace total_chairs_in_canteen_l1928_192867

theorem total_chairs_in_canteen 
    (round_tables : ℕ) 
    (chairs_per_round_table : ℕ) 
    (rectangular_tables : ℕ) 
    (chairs_per_rectangular_table : ℕ) 
    (square_tables : ℕ) 
    (chairs_per_square_table : ℕ) 
    (extra_chairs : ℕ) 
    (h1 : round_tables = 3)
    (h2 : chairs_per_round_table = 6)
    (h3 : rectangular_tables = 4)
    (h4 : chairs_per_rectangular_table = 7)
    (h5 : square_tables = 2)
    (h6 : chairs_per_square_table = 4)
    (h7 : extra_chairs = 5) :
    round_tables * chairs_per_round_table +
    rectangular_tables * chairs_per_rectangular_table +
    square_tables * chairs_per_square_table +
    extra_chairs = 59 := by
  sorry

end total_chairs_in_canteen_l1928_192867


namespace mark_total_cans_l1928_192873

theorem mark_total_cans (p1 p2 p3 p4 p5 p6 : ℕ) (c1 c2 c3 c4 c5 c6 : ℕ)
  (h1 : p1 = 30) (h2 : p2 = 25) (h3 : p3 = 35) (h4 : p4 = 40) 
  (h5 : p5 = 28) (h6 : p6 = 32) (hc1 : c1 = 12) (hc2 : c2 = 10) 
  (hc3 : c3 = 15) (hc4 : c4 = 14) (hc5 : c5 = 11) (hc6 : c6 = 13) :
  p1 * c1 + p2 * c2 + p3 * c3 + p4 * c4 + p5 * c5 + p6 * c6 = 2419 := 
by 
  sorry

end mark_total_cans_l1928_192873


namespace sum_of_acute_angles_l1928_192860

open Real

theorem sum_of_acute_angles (θ₁ θ₂ : ℝ)
  (h1 : 0 < θ₁ ∧ θ₁ < π / 2)
  (h2 : 0 < θ₂ ∧ θ₂ < π / 2)
  (h_eq : (sin θ₁) ^ 2020 / (cos θ₂) ^ 2018 + (cos θ₁) ^ 2020 / (sin θ₂) ^ 2018 = 1) :
  θ₁ + θ₂ = π / 2 := sorry

end sum_of_acute_angles_l1928_192860


namespace confidence_relationship_l1928_192876
noncomputable def K_squared : ℝ := 3.918
noncomputable def critical_value : ℝ := 3.841
noncomputable def p_val : ℝ := 0.05

theorem confidence_relationship (K_squared : ℝ) (critical_value : ℝ) (p_val : ℝ) :
  K_squared ≥ critical_value -> p_val = 0.05 ->
  1 - p_val = 0.95 :=
by
  sorry

end confidence_relationship_l1928_192876


namespace range_of_a_l1928_192806

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 1 < x ∧ x < 4 → (1/3 * x^3 - (a/2) * x^2 + (a-1) * x + 1) < 0) ∧
  (∀ x : ℝ, x > 6 → (1/3 * x^3 - (a/2) * x^2 + (a-1) * x + 1) > 0)
  ↔ (5 < a ∧ a < 7) :=
sorry

end range_of_a_l1928_192806


namespace sequence_general_formula_l1928_192862

-- Define the sequence S_n and the initial conditions
def S (n : ℕ) : ℕ := 3^(n + 1) - 1

-- Define the sequence a_n
def a (n : ℕ) : ℕ :=
  if n = 1 then 8 else 2 * 3^n

-- Theorem statement proving the general formula
theorem sequence_general_formula (n : ℕ) : 
  a n = if n = 1 then 8 else 2 * 3^n := by
  -- This is where the proof would go
  sorry

end sequence_general_formula_l1928_192862


namespace sum_of_thousands_and_units_digit_of_product_l1928_192803

def thousands_digit (n : ℕ) : ℕ := (n / 1000) % 10
def units_digit (n : ℕ) : ℕ := n % 10

-- Define the two 102-digit numbers
def num1 : ℕ := 404040404040404040404040404040404040404040404040404040404040404040404040404040404040404040404
def num2 : ℕ := 707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707

-- Define their product
def product : ℕ := num1 * num2

-- Define the conditions for the problem
def A := thousands_digit product
def B := units_digit product

-- Define the problem statement
theorem sum_of_thousands_and_units_digit_of_product : A + B = 13 := 
by
  sorry

end sum_of_thousands_and_units_digit_of_product_l1928_192803


namespace pencils_distributed_l1928_192813

-- Define the conditions as a Lean statement
theorem pencils_distributed :
  let friends := 4
  let pencils := 8
  let at_least_one := 1
  ∃ (ways : ℕ), ways = 35 := sorry

end pencils_distributed_l1928_192813


namespace jayden_current_age_l1928_192827

def current_age_of_Jayden (e : ℕ) (j_in_3_years : ℕ) : ℕ :=
  j_in_3_years - 3

theorem jayden_current_age (e : ℕ) (h1 : e = 11) (h2 : ∃ j : ℕ, j = ((e + 3) / 2) ∧ j_in_3_years = j) : 
  current_age_of_Jayden e j_in_3_years = 4 :=
by
  sorry

end jayden_current_age_l1928_192827


namespace find_a_l1928_192879

theorem find_a (a : ℝ) :
  (∀ (x y : ℝ), (a * x + 2 * y + 3 * a = 0) → (3 * x + (a - 1) * y = a - 7)) → 
  a = 3 :=
by
  sorry

end find_a_l1928_192879


namespace started_with_l1928_192824

-- Define the conditions
def total_eggs : ℕ := 70
def bought_eggs : ℕ := 62

-- Define the statement to prove
theorem started_with (initial_eggs : ℕ) : initial_eggs = total_eggs - bought_eggs → initial_eggs = 8 := by
  intro h
  sorry

end started_with_l1928_192824


namespace beetle_crawls_100th_segment_in_1300_seconds_l1928_192844

def segment_length (n : ℕ) : ℕ :=
  (n / 4) + 1

def total_length (s : ℕ) : ℕ :=
  (s / 4) * 4 * (segment_length (s - 1)) * (segment_length (s - 1) + 1) / 2

theorem beetle_crawls_100th_segment_in_1300_seconds :
  total_length 100 = 1300 :=
  sorry

end beetle_crawls_100th_segment_in_1300_seconds_l1928_192844


namespace second_root_of_system_l1928_192861

def system_of_equations (x y : ℝ) : Prop :=
  (2 * x^2 + 3 * x * y + y^2 = 70) ∧ (6 * x^2 + x * y - y^2 = 50)

theorem second_root_of_system :
  system_of_equations 3 4 →
  system_of_equations (-3) (-4) :=
by
  sorry

end second_root_of_system_l1928_192861


namespace sequence_polynomial_degree_l1928_192830

theorem sequence_polynomial_degree
  (k : ℕ)
  (v : ℕ → ℤ)
  (u : ℕ → ℤ)
  (h_diff_poly : ∃ p : Polynomial ℤ, ∀ n, v n = Polynomial.eval (n : ℤ) p)
  (h_diff_seq : ∀ n, v n = (u (n + 1) - u n)) :
  ∃ q : Polynomial ℤ, ∀ n, u n = Polynomial.eval (n : ℤ) q := 
sorry

end sequence_polynomial_degree_l1928_192830


namespace number_of_biscuits_l1928_192842

theorem number_of_biscuits (dough_length dough_width biscuit_length biscuit_width : ℕ)
    (h_dough : dough_length = 12) (h_dough_width : dough_width = 12)
    (h_biscuit_length : biscuit_length = 3) (h_biscuit_width : biscuit_width = 3)
    (dough_area : ℕ := dough_length * dough_width)
    (biscuit_area : ℕ := biscuit_length * biscuit_width) :
    dough_area / biscuit_area = 16 :=
by
  -- assume dough_area and biscuit_area are calculated from the given conditions
  -- dough_area = 144 and biscuit_area = 9
  sorry

end number_of_biscuits_l1928_192842


namespace sidney_thursday_jacks_l1928_192841

open Nat

-- Define the number of jumping jacks Sidney did on each day
def monday_jacks := 20
def tuesday_jacks := 36
def wednesday_jacks := 40

-- Define the total number of jumping jacks done by Sidney
-- on Monday, Tuesday, and Wednesday
def sidney_mon_wed_jacks := monday_jacks + tuesday_jacks + wednesday_jacks

-- Define the total number of jumping jacks done by Brooke
def brooke_jacks := 438

-- Define the relationship between Brooke's and Sidney's total jumping jacks
def sidney_total_jacks := brooke_jacks / 3

-- Prove the number of jumping jacks Sidney did on Thursday
theorem sidney_thursday_jacks :
  sidney_total_jacks - sidney_mon_wed_jacks = 50 :=
by
  sorry

end sidney_thursday_jacks_l1928_192841


namespace color_change_probability_is_correct_l1928_192870

-- Given definitions
def cycle_time : ℕ := 45 + 5 + 10 + 40

def favorable_time : ℕ := 5 + 5 + 5

def probability_color_change : ℚ := favorable_time / cycle_time

-- Theorem statement to prove the probability
theorem color_change_probability_is_correct :
  probability_color_change = 0.15 := 
sorry

end color_change_probability_is_correct_l1928_192870


namespace mysterious_division_l1928_192847

theorem mysterious_division (d : ℕ) : (8 * d < 1000) ∧ (7 * d < 900) → d = 124 :=
by
  intro h
  sorry

end mysterious_division_l1928_192847


namespace initial_non_electrified_part_l1928_192889

variables (x y : ℝ)

def electrified_fraction : Prop :=
  x + y = 1 ∧ 2 * x + 0.75 * y = 1

theorem initial_non_electrified_part (h : electrified_fraction x y) : y = 4 / 5 :=
by {
  sorry
}

end initial_non_electrified_part_l1928_192889


namespace george_team_final_round_average_required_less_than_record_l1928_192875

theorem george_team_final_round_average_required_less_than_record :
  ∀ (old_record average_score : ℕ) (players : ℕ) (rounds : ℕ) (current_score : ℕ),
    old_record = 287 →
    players = 4 →
    rounds = 10 →
    current_score = 10440 →
    (old_record - ((rounds * (old_record * players) - current_score) / players)) = 27 :=
by
  -- Given the values and conditions, prove the equality here
  sorry

end george_team_final_round_average_required_less_than_record_l1928_192875


namespace ball_bounce_height_l1928_192802

theorem ball_bounce_height (a : ℝ) (r : ℝ) (threshold : ℝ) (k : ℕ) 
  (h_a : a = 20) (h_r : r = 1/2) (h_threshold : threshold = 0.5) :
  20 * (r^k) < threshold ↔ k = 5 :=
by sorry

end ball_bounce_height_l1928_192802


namespace failing_percentage_exceeds_35_percent_l1928_192804

theorem failing_percentage_exceeds_35_percent:
  ∃ (n D A B failD failA : ℕ), 
  n = 25 ∧
  D + A - B = n ∧
  (failD * 100) / D = 30 ∧
  (failA * 100) / A = 30 ∧
  ((failD + failA) * 100) / n > 35 := 
by
  sorry

end failing_percentage_exceeds_35_percent_l1928_192804


namespace probability_3_queens_or_at_least_2_aces_l1928_192885

-- Definitions of drawing from a standard deck and probabilities involved
def num_cards : ℕ := 52
def num_queens : ℕ := 4
def num_aces : ℕ := 4

def probability_all_queens : ℚ := (4/52) * (3/51) * (2/50)
def probability_2_aces_1_non_ace : ℚ := (4/52) * (3/51) * (48/50)
def probability_3_aces : ℚ := (4/52) * (3/51) * (2/50)
def probability_at_least_2_aces : ℚ := (probability_2_aces_1_non_ace) + (probability_3_aces)

def total_probability : ℚ := probability_all_queens + probability_at_least_2_aces

-- Statement to be proved
theorem probability_3_queens_or_at_least_2_aces :
  total_probability = 220 / 581747 :=
sorry

end probability_3_queens_or_at_least_2_aces_l1928_192885


namespace sample_size_is_fifteen_l1928_192864

variable (total_employees : ℕ) (young_employees : ℕ) (middle_aged_employees : ℕ)
variable (elderly_employees : ℕ) (young_sample_count : ℕ) (sample_size : ℕ)

theorem sample_size_is_fifteen
  (h1 : total_employees = 750)
  (h2 : young_employees = 350)
  (h3 : middle_aged_employees = 250)
  (h4 : elderly_employees = 150)
  (h5 : 7 = young_sample_count)
  : sample_size = 15 := 
sorry

end sample_size_is_fifteen_l1928_192864


namespace number_of_spiders_l1928_192807

theorem number_of_spiders (total_legs birds dogs snakes : ℕ) (legs_per_bird legs_per_dog legs_per_snake legs_per_spider : ℕ) (h1 : total_legs = 34)
  (h2 : birds = 3) (h3 : dogs = 5) (h4 : snakes = 4) (h5 : legs_per_bird = 2) (h6 : legs_per_dog = 4)
  (h7 : legs_per_snake = 0) (h8 : legs_per_spider = 8) : 
  (total_legs - (birds * legs_per_bird + dogs * legs_per_dog + snakes * legs_per_snake)) / legs_per_spider = 1 :=
by sorry

end number_of_spiders_l1928_192807


namespace solve_fraction_equation_l1928_192897

theorem solve_fraction_equation (x : ℝ) (h : x ≠ 1) : (3 * x - 1) / (4 * x - 4) = 2 / 3 → x = -5 :=
by
  intro h_eq
  sorry

end solve_fraction_equation_l1928_192897


namespace right_triangle_area_integer_l1928_192891

theorem right_triangle_area_integer (a b c : ℤ) (h : a * a + b * b = c * c) : ∃ (n : ℤ), (1 / 2 : ℚ) * a * b = ↑n := 
sorry

end right_triangle_area_integer_l1928_192891


namespace debby_deleted_pictures_l1928_192882

theorem debby_deleted_pictures :
  ∀ (zoo_pics museum_pics remaining_pics : ℕ), 
  zoo_pics = 24 →
  museum_pics = 12 →
  remaining_pics = 22 →
  (zoo_pics + museum_pics) - remaining_pics = 14 :=
by
  intros zoo_pics museum_pics remaining_pics hz hm hr
  sorry

end debby_deleted_pictures_l1928_192882


namespace more_cats_than_dogs_l1928_192877

theorem more_cats_than_dogs (total_animals : ℕ) (cats : ℕ) (h1 : total_animals = 60) (h2 : cats = 40) : (cats - (total_animals - cats)) = 20 :=
by 
  sorry

end more_cats_than_dogs_l1928_192877


namespace find_k_l1928_192850

-- Definition of vectors a and b
def vec_a (k : ℝ) : ℝ × ℝ := (-1, k)
def vec_b : ℝ × ℝ := (3, 1)

-- Definition of dot product
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Property of perpendicular vectors (dot product is zero)
def is_perpendicular (u v : ℝ × ℝ) : Prop := dot_product u v = 0

-- Problem statement
theorem find_k (k : ℝ) :
  is_perpendicular (vec_a k) (vec_a k) →
  (k = -2 ∨ k = 1) :=
sorry

end find_k_l1928_192850


namespace find_x_l1928_192890

theorem find_x (x : ℤ) (h : 4 * x - 23 = 33) : x = 14 := 
by 
  sorry

end find_x_l1928_192890


namespace smallest_base_power_l1928_192849

theorem smallest_base_power (x y z : ℝ) (hx : 1 < x) (hy : 1 < y) (hz : 1 < z)
  (h_log_eq : Real.log x / Real.log 2 = Real.log y / Real.log 3 ∧ Real.log y / Real.log 3 = Real.log z / Real.log 5) :
  z ^ (1 / 5) < x ^ (1 / 2) ∧ z ^ (1 / 5) < y ^ (1 / 3) :=
by
  -- required proof here
  sorry

end smallest_base_power_l1928_192849


namespace functional_relationship_maximum_profit_desired_profit_l1928_192858

-- Conditions
def cost_price := 80
def y (x : ℝ) : ℝ := -2 * x + 320
def w (x : ℝ) : ℝ := (x - cost_price) * y x

-- Functional relationship
theorem functional_relationship (x : ℝ) (hx : 80 ≤ x ∧ x ≤ 160) :
  w x = -2 * x^2 + 480 * x - 25600 :=
by sorry

-- Maximizing daily profit
theorem maximum_profit :
  ∃ x, 80 ≤ x ∧ x ≤ 160 ∧ w x = 3200 ∧ (∀ y, 80 ≤ y ∧ y ≤ 160 → w y ≤ 3200) ∧ x = 120 :=
by sorry

-- Desired profit of 2400 dollars
theorem desired_profit (w_desired : ℝ) (hw : w_desired = 2400) :
  ∃ x, 80 ≤ x ∧ x ≤ 160 ∧ w x = w_desired ∧ x = 100 :=
by sorry

end functional_relationship_maximum_profit_desired_profit_l1928_192858


namespace find_product_l1928_192859

-- Define the variables used in the problem statement
variables (A P D B E C F : Type) (AP PD BP PE CP PF : ℝ)

-- The condition given in the problem
def condition (x y z : ℝ) : Prop := 
  x + y + z = 90

-- The theorem to prove
theorem find_product (x y z : ℝ) (h : condition x y z) : 
  x * y * z = 94 :=
sorry

end find_product_l1928_192859


namespace sum_of_three_numbers_l1928_192888

variable {a b c : ℝ}

theorem sum_of_three_numbers :
  a^2 + b^2 + c^2 = 99 ∧ ab + bc + ca = 131 → a + b + c = 19 :=
by
  sorry

end sum_of_three_numbers_l1928_192888


namespace percentage_loss_calculation_l1928_192896

theorem percentage_loss_calculation
  (initial_cost_euro : ℝ)
  (retail_price_dollars : ℝ)
  (exchange_rate_initial : ℝ)
  (discount1 : ℝ)
  (discount2 : ℝ)
  (sales_tax : ℝ)
  (exchange_rate_new : ℝ)
  (final_sale_price_dollars : ℝ) :
  initial_cost_euro = 800 ∧
  retail_price_dollars = 900 ∧
  exchange_rate_initial = 1.1 ∧
  discount1 = 0.10 ∧
  discount2 = 0.15 ∧
  sales_tax = 0.10 ∧
  exchange_rate_new = 1.5 ∧
  final_sale_price_dollars = (retail_price_dollars * (1 - discount1) * (1 - discount2) * (1 + sales_tax)) →
  ((initial_cost_euro - final_sale_price_dollars / exchange_rate_new) / initial_cost_euro) * 100 = 36.89 := by
  sorry

end percentage_loss_calculation_l1928_192896


namespace common_ratio_l1928_192854

noncomputable def geometric_sum (a₁ q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a₁ else a₁ * (1 - q^n) / (1 - q)

theorem common_ratio (a₁ : ℝ) (h : a₁ ≠ 0) : 
  (∀ S4 S5 S6, S5 = geometric_sum a₁ q 5 ∧ S4 = geometric_sum a₁ q 4 ∧ S6 = geometric_sum a₁ q 6 → 
  2 * S4 = S5 + S6) → 
  q = -2 := 
by
  sorry

end common_ratio_l1928_192854


namespace negation_proof_l1928_192871

theorem negation_proof (a b : ℝ) (h : a^2 + b^2 = 0) : ¬(a = 0 ∧ b = 0) :=
sorry

end negation_proof_l1928_192871
