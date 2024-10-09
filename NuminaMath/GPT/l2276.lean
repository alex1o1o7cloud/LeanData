import Mathlib

namespace tires_sale_price_l2276_227686

variable (n : ℕ)
variable (t p_original p_sale : ℝ)

theorem tires_sale_price
  (h₁ : n = 4)
  (h₂ : t = 36)
  (h₃ : p_original = 84)
  (h₄ : p_sale = p_original - t / n) :
  p_sale = 75 := by
  sorry

end tires_sale_price_l2276_227686


namespace find_a_l2276_227645

theorem find_a (a : ℝ) (y : ℝ → ℝ) (y' : ℝ → ℝ) 
    (h_curve : ∀ x, y x = x^4 + a * x^2 + 1)
    (h_derivative : ∀ x, y' x = (4 * x^3 + 2 * a * x))
    (h_tangent_slope : y' (-1) = 8) :
    a = -6 :=
by
  -- To be proven
  sorry

end find_a_l2276_227645


namespace width_of_metallic_sheet_l2276_227601

theorem width_of_metallic_sheet 
  (length : ℕ)
  (new_volume : ℕ) 
  (side_length_of_square : ℕ)
  (height_of_box : ℕ)
  (new_length : ℕ)
  (new_width : ℕ)
  (w : ℕ) : 
  length = 48 → 
  new_volume = 5120 → 
  side_length_of_square = 8 → 
  height_of_box = 8 → 
  new_length = length - 2 * side_length_of_square → 
  new_width = w - 2 * side_length_of_square → 
  new_volume = new_length * new_width * height_of_box → 
  w = 36 := 
by 
  intros _ _ _ _ _ _ _ 
  sorry

end width_of_metallic_sheet_l2276_227601


namespace total_packs_sold_l2276_227672

def packs_sold_village_1 : ℕ := 23
def packs_sold_village_2 : ℕ := 28

theorem total_packs_sold : packs_sold_village_1 + packs_sold_village_2 = 51 :=
by
  -- We acknowledge the correctness of the calculation.
  sorry

end total_packs_sold_l2276_227672


namespace reinforcement_calculation_l2276_227635

theorem reinforcement_calculation
  (initial_men : ℕ := 2000)
  (initial_days : ℕ := 40)
  (days_until_reinforcement : ℕ := 20)
  (additional_days_post_reinforcement : ℕ := 10)
  (total_initial_provisions : ℕ := initial_men * initial_days)
  (remaining_provisions_post_20_days : ℕ := total_initial_provisions / 2)
  : ∃ (reinforcement_men : ℕ), reinforcement_men = 2000 :=
by
  have remaining_provisions := remaining_provisions_post_20_days
  have total_post_reinforcement := initial_men + ((remaining_provisions) / (additional_days_post_reinforcement))

  use (total_post_reinforcement - initial_men)
  sorry

end reinforcement_calculation_l2276_227635


namespace banana_cost_l2276_227669

theorem banana_cost (pounds: ℕ) (rate: ℕ) (per_pounds: ℕ) : 
 (pounds = 18) → (rate = 3) → (per_pounds = 3) → 
  (pounds / per_pounds * rate = 18) := by
  intros
  sorry

end banana_cost_l2276_227669


namespace translation_correctness_l2276_227678

theorem translation_correctness :
  ( ∀ (x : ℝ), ((x + 4)^2 - 5) = ((x + 4)^2 - 5) ) :=
by
  sorry

end translation_correctness_l2276_227678


namespace max_popsicles_l2276_227644

theorem max_popsicles (budget : ℕ) (cost_single : ℕ) (popsicles_single : ℕ) (cost_box3 : ℕ) (popsicles_box3 : ℕ) (cost_box7 : ℕ) (popsicles_box7 : ℕ)
  (h_budget : budget = 10) (h_cost_single : cost_single = 1) (h_popsicles_single : popsicles_single = 1)
  (h_cost_box3 : cost_box3 = 3) (h_popsicles_box3 : popsicles_box3 = 3)
  (h_cost_box7 : cost_box7 = 4) (h_popsicles_box7 : popsicles_box7 = 7) :
  ∃ n, n = 16 :=
by
  sorry

end max_popsicles_l2276_227644


namespace area_of_rectangular_field_l2276_227651

-- Definitions from conditions
def L : ℕ := 20
def total_fencing : ℕ := 32

-- Additional variables inferred from the conditions
def W : ℕ := (total_fencing - L) / 2

-- The theorem statement
theorem area_of_rectangular_field : L * W = 120 :=
by
  -- Definitions and substitutions are included in the theorem proof
  sorry

end area_of_rectangular_field_l2276_227651


namespace inverse_function_evaluation_l2276_227647

theorem inverse_function_evaluation :
  ∀ (f : ℕ → ℕ) (f_inv : ℕ → ℕ),
    (∀ y, f_inv (f y) = y) ∧ (∀ x, f (f_inv x) = x) →
    f 4 = 7 →
    f 6 = 3 →
    f 3 = 6 →
    f_inv (f_inv 6 + f_inv 7) = 4 :=
by
  intros f f_inv hf hf1 hf2 hf3
  sorry

end inverse_function_evaluation_l2276_227647


namespace determine_a_l2276_227664

theorem determine_a (a b c : ℤ) (h_eq : ∀ x : ℤ, (x - a) * (x - 15) + 4 = (x + b) * (x + c)) :
  a = 16 ∨ a = 21 :=
  sorry

end determine_a_l2276_227664


namespace total_pay_l2276_227681

-- Definitions based on the conditions
def y_pay : ℕ := 290
def x_pay : ℕ := (120 * y_pay) / 100

-- The statement to prove that the total pay is Rs. 638
theorem total_pay : x_pay + y_pay = 638 := 
by
  -- skipping the proof for now
  sorry

end total_pay_l2276_227681


namespace parabola_circle_intersection_l2276_227667

theorem parabola_circle_intersection (a : ℝ) : 
  a ≤ Real.sqrt 2 + 1 / 4 → 
  ∃ (b x y : ℝ), y = x^2 + a ∧ x^2 + y^2 + 2 * b^2 = 2 * b * (x - y) + 1 :=
by
  sorry

end parabola_circle_intersection_l2276_227667


namespace product_of_three_numbers_l2276_227680

theorem product_of_three_numbers (a b c : ℝ) 
  (h₁ : a + b + c = 45)
  (h₂ : a = 2 * (b + c))
  (h₃ : c = 4 * b) : 
  a * b * c = 1080 := 
sorry

end product_of_three_numbers_l2276_227680


namespace evaluate_expression_l2276_227673

theorem evaluate_expression (a b : ℕ) (ha : a = 7) (hb : b = 5) : 3 * (a^3 + b^3) / (a^2 - a * b + b^2) = 36 :=
by
  rw [ha, hb]
  sorry

end evaluate_expression_l2276_227673


namespace linear_function_value_l2276_227615

theorem linear_function_value (g : ℝ → ℝ) (h_linear : ∀ x y, g (x + y) = g x + g y)
  (h_scale : ∀ c x, g (c * x) = c * g x) (h : g 10 - g 0 = 20) : g 20 - g 0 = 40 :=
by
  sorry

end linear_function_value_l2276_227615


namespace aviana_brought_pieces_l2276_227684

variable (total_people : ℕ) (fraction_eat_pizza : ℚ) (pieces_per_person : ℕ) (remaining_pieces : ℕ)

theorem aviana_brought_pieces (h1 : total_people = 15) 
                             (h2 : fraction_eat_pizza = 3 / 5) 
                             (h3 : pieces_per_person = 4) 
                             (h4 : remaining_pieces = 14) :
                             ∃ (brought_pieces : ℕ), brought_pieces = 50 :=
by sorry

end aviana_brought_pieces_l2276_227684


namespace school_students_unique_l2276_227612

theorem school_students_unique 
  (n : ℕ)
  (h1 : 70 < n) 
  (h2 : n < 130) 
  (h3 : n % 4 = 2) 
  (h4 : n % 5 = 2)
  (h5 : n % 6 = 2) : 
  (n = 92 ∨ n = 122) :=
  sorry

end school_students_unique_l2276_227612


namespace find_sr_division_l2276_227618

theorem find_sr_division (k : ℚ) (c r s : ℚ)
  (h_c : c = 10)
  (h_r : r = -3 / 10)
  (h_s : s = 191 / 10)
  (h_expr : 10 * k^2 - 6 * k + 20 = c * (k + r)^2 + s) :
  s / r = -191 / 3 :=
by
  sorry

end find_sr_division_l2276_227618


namespace quarters_percentage_value_l2276_227622

theorem quarters_percentage_value (dimes quarters : Nat) (value_dime value_quarter : Nat) (total_value quarter_value : Nat)
(h_dimes : dimes = 30)
(h_quarters : quarters = 40)
(h_value_dime : value_dime = 10)
(h_value_quarter : value_quarter = 25)
(h_total_value : total_value = dimes * value_dime + quarters * value_quarter)
(h_quarter_value : quarter_value = quarters * value_quarter) :
(quarter_value : ℚ) / (total_value : ℚ) * 100 = 76.92 := 
sorry

end quarters_percentage_value_l2276_227622


namespace hyperbola_problem_l2276_227602

noncomputable def is_hyperbola (x y : ℝ) (a b : ℝ) : Prop :=
  (x^2 / a^2) - ((y - 2)^2 / b^2) = 1

variables (s : ℝ)

theorem hyperbola_problem
  (h₁ : is_hyperbola 0 5 a b)
  (h₂ : is_hyperbola (-1) 6 a b)
  (h₃ : is_hyperbola s 3 a b)
  (hb : b^2 = 9)
  (ha : a^2 = 9 / 25) :
  s^2 = 2 / 5 :=
sorry

end hyperbola_problem_l2276_227602


namespace find_angle_l2276_227600

-- Definitions based on conditions
def is_complement (x : ℝ) : ℝ := 90 - x
def is_supplement (x : ℝ) : ℝ := 180 - x

-- Main statement
theorem find_angle (x : ℝ) (h : is_supplement x = 15 + 4 * is_complement x) : x = 65 :=
by
  sorry

end find_angle_l2276_227600


namespace intersection_point_exists_circle_equation_standard_form_l2276_227606

noncomputable def line1 (x y : ℝ) : Prop := 2 * x + y = 0
noncomputable def line2 (x y : ℝ) : Prop := x + y = 2
noncomputable def line3 (x y : ℝ) : Prop := 3 * x + 4 * y + 5 = 0

theorem intersection_point_exists :
  ∃ (C : ℝ × ℝ), (line1 C.1 C.2 ∧ line2 C.1 C.2) ∧ C = (-2, 4) :=
sorry

theorem circle_equation_standard_form :
  ∃ (center : ℝ × ℝ) (radius : ℝ), center = (-2, 4) ∧ radius = 3 ∧
  ∀ x y : ℝ, ((x + 2) ^ 2 + (y - 4) ^ 2 = 9) :=
sorry

end intersection_point_exists_circle_equation_standard_form_l2276_227606


namespace exists_x_such_that_f_x_eq_0_l2276_227616

noncomputable def f (x : ℝ) : ℝ :=
if x < 1 then
  3 * x - 4
else
  -x^2 + 3 * x - 5

theorem exists_x_such_that_f_x_eq_0 :
  ∃ x : ℝ, f x = 0 ∧ x = 1.192 :=
sorry

end exists_x_such_that_f_x_eq_0_l2276_227616


namespace last_five_digits_of_sequence_l2276_227656

theorem last_five_digits_of_sequence (seq : Fin 36 → Fin 2) 
  (h0 : seq 0 = 0) (h1 : seq 1 = 0) (h2 : seq 2 = 0) (h3 : seq 3 = 0) (h4 : seq 4 = 0)
  (unique_combos : ∀ (combo: Fin 32 → Fin 2), 
    ∃ (start_index : Fin 32), ∀ (i : Fin 5),
      combo i = seq ((start_index + i) % 36)) :
  seq 31 = 1 ∧ seq 32 = 1 ∧ seq 33 = 1 ∧ seq 34 = 0 ∧ seq 35 = 1 :=
by
  sorry

end last_five_digits_of_sequence_l2276_227656


namespace cos_equivalent_l2276_227660

open Real

theorem cos_equivalent (alpha : ℝ) (h : sin (π / 3 + alpha) = 1 / 3) : 
  cos (5 * π / 6 + alpha) = -1 / 3 :=
sorry

end cos_equivalent_l2276_227660


namespace octagon_area_is_six_and_m_plus_n_is_seven_l2276_227617

noncomputable def area_of_octagon (side_length : ℕ) (segment_length : ℚ) : ℚ :=
  let triangle_area := 1 / 2 * side_length * segment_length
  let octagon_area := 8 * triangle_area
  octagon_area

theorem octagon_area_is_six_and_m_plus_n_is_seven :
  area_of_octagon 2 (3/4) = 6 ∧ (6 + 1 = 7) :=
by
  sorry

end octagon_area_is_six_and_m_plus_n_is_seven_l2276_227617


namespace bills_difference_l2276_227613

variable (m j : ℝ)

theorem bills_difference :
  (0.10 * m = 2) → (0.20 * j = 2) → (m - j = 10) :=
by
  intros h1 h2
  sorry

end bills_difference_l2276_227613


namespace gasoline_fraction_used_l2276_227609

theorem gasoline_fraction_used
  (speed : ℕ) (gas_usage : ℕ) (initial_gallons : ℕ) (travel_time : ℕ)
  (h_speed : speed = 50) (h_gas_usage : gas_usage = 30) 
  (h_initial_gallons : initial_gallons = 15) (h_travel_time : travel_time = 5) :
  (speed * travel_time) / gas_usage / initial_gallons = 5 / 9 :=
by
  sorry

end gasoline_fraction_used_l2276_227609


namespace operation_value_l2276_227685

def operation1 (y : ℤ) : ℤ := 8 - y
def operation2 (y : ℤ) : ℤ := y - 8

theorem operation_value : operation2 (operation1 15) = -15 := by
  sorry

end operation_value_l2276_227685


namespace smallest_row_sum_greater_than_50_l2276_227663

noncomputable def sum_interior_pascal (n : ℕ) : ℕ :=
  2^(n-1) - 2

theorem smallest_row_sum_greater_than_50 : ∃ n, sum_interior_pascal n > 50 ∧ (∀ m, m < n → sum_interior_pascal m ≤ 50) ∧ sum_interior_pascal 7 = 62 ∧ (sum_interior_pascal 7) % 2 = 0 :=
by
  sorry

end smallest_row_sum_greater_than_50_l2276_227663


namespace vertices_sum_zero_l2276_227633

theorem vertices_sum_zero
  (a b c d e f g h : ℝ)
  (h1 : a = (b + e + d) / 3)
  (h2 : b = (c + f + a) / 3)
  (h3 : c = (d + g + b) / 3)
  (h4 : d = (a + h + e) / 3)
  :
  (a + b + c + d) - (e + f + g + h) = 0 :=
by
  sorry

end vertices_sum_zero_l2276_227633


namespace arcsin_sqrt3_over_2_eq_pi_over_3_l2276_227636

theorem arcsin_sqrt3_over_2_eq_pi_over_3 :
  Real.arcsin (Real.sqrt 3 / 2) = π / 3 :=
by
  have h : Real.sin (π / 3) = Real.sqrt 3 / 2 := by
    -- This is a known trigonometric identity.
    sorry
  -- Use the property of arcsin to get the result.
  sorry

end arcsin_sqrt3_over_2_eq_pi_over_3_l2276_227636


namespace rectangles_with_perimeter_equals_area_l2276_227689

theorem rectangles_with_perimeter_equals_area (a b : ℕ) (h : 2 * (a + b) = a * b) : (a = 3 ∧ b = 6) ∨ (a = 6 ∧ b = 3) ∨ (a = 4 ∧ b = 4) :=
  sorry

end rectangles_with_perimeter_equals_area_l2276_227689


namespace algebra_problem_l2276_227626

theorem algebra_problem 
  (x : ℝ) 
  (h : x^2 - 2 * x = 3) : 
  2 * x^2 - 4 * x + 3 = 9 := 
by 
  sorry

end algebra_problem_l2276_227626


namespace find_a_l2276_227614

def f (a : ℝ) (x : ℝ) := a * x^2 + 3 * x - 2

theorem find_a (a : ℝ) (h : deriv (f a) 2 = 7) : a = 1 :=
by {
  sorry
}

end find_a_l2276_227614


namespace eval_expr_l2276_227691

theorem eval_expr : (1 / (5^2)^4 * 5^11 * 2) = 250 := by
  sorry

end eval_expr_l2276_227691


namespace race_participants_minimum_l2276_227687

theorem race_participants_minimum : ∃ n : ℕ, 
  ((n - 1) % 3 = 0) ∧ 
  ((n - 1) % 4 = 0) ∧ 
  ((n - 1) % 5 = 0) ∧ 
  (∀ m : ℕ, 
    ((m - 1) % 3 = 0) ∧ 
    ((m - 1) % 4 = 0) ∧ 
    ((m - 1) % 5 = 0) → 
    n ≤ m) := 
sorry

end race_participants_minimum_l2276_227687


namespace arithmetic_geometric_l2276_227666

theorem arithmetic_geometric (a_n : ℕ → ℤ) (h1 : ∀ n, a_n n = a_n 0 + n * 2)
  (h2 : ∃ a, a = a_n 0 ∧ (a_n 0 + 4)^2 = a_n 0 * (a_n 0 + 6)) : a_n 0 = -8 := by
  sorry

end arithmetic_geometric_l2276_227666


namespace cost_of_iPhone_l2276_227631

theorem cost_of_iPhone (P : ℝ) 
  (phone_contract_cost : ℝ := 200)
  (case_percent_of_P : ℝ := 0.20)
  (headphones_percent_of_case : ℝ := 0.50)
  (total_yearly_cost : ℝ := 3700) :
  let year_phone_contract_cost := (phone_contract_cost * 12)
  let case_cost := (case_percent_of_P * P)
  let headphones_cost := (headphones_percent_of_case * case_cost)
  P + year_phone_contract_cost + case_cost + headphones_cost = total_yearly_cost → 
  P = 1000 :=
by
  sorry  -- proof not required

end cost_of_iPhone_l2276_227631


namespace intersection_of_A_and_B_l2276_227640

def A : Set ℤ := {-2, -1}
def B : Set ℤ := {-1, 2, 3}

theorem intersection_of_A_and_B :
  A ∩ B = {-1} :=
sorry

end intersection_of_A_and_B_l2276_227640


namespace range_of_a_l2276_227619

variable (a x : ℝ)

theorem range_of_a (h : ax > 2) (h_transform: ax > 2 → x < 2/a) : a < 0 :=
sorry

end range_of_a_l2276_227619


namespace sqrt_x_minus_2_range_l2276_227603

theorem sqrt_x_minus_2_range (x : ℝ) : x - 2 ≥ 0 → x ≥ 2 :=
by sorry

end sqrt_x_minus_2_range_l2276_227603


namespace minimum_value_x_squared_plus_12x_plus_5_l2276_227671

theorem minimum_value_x_squared_plus_12x_plus_5 : ∃ x : ℝ, x^2 + 12 * x + 5 = -31 :=
by sorry

end minimum_value_x_squared_plus_12x_plus_5_l2276_227671


namespace relay_race_team_members_l2276_227639

theorem relay_race_team_members (n : ℕ) (d : ℕ) (h1 : n = 5) (h2 : d = 150) : d / n = 30 := 
by {
  -- Place the conditions here as hypotheses
  sorry
}

end relay_race_team_members_l2276_227639


namespace compare_neg_numbers_l2276_227697

theorem compare_neg_numbers : - 0.6 > - (2 / 3) := 
by sorry

end compare_neg_numbers_l2276_227697


namespace probability_f_leq_zero_l2276_227683

noncomputable def f (k x : ℝ) : ℝ := k * x - 1

theorem probability_f_leq_zero : 
  ∀ (x : ℝ), (0 ≤ x ∧ x ≤ 1) →
  (∀ k ∈ Set.Icc (-2 : ℝ) (2 : ℝ), f k x ≤ 0) →
  (∃ k ∈ Set.Icc (-2 : ℝ) (1 : ℝ), f k x ≤ 0) →
  ((1 - (-2)) / (2 - (-2)) = 3 / 4) :=
by sorry

end probability_f_leq_zero_l2276_227683


namespace linear_function_third_quadrant_and_origin_l2276_227621

theorem linear_function_third_quadrant_and_origin (k b : ℝ) (h1 : ∀ x < 0, k * x + b ≥ 0) (h2 : k * 0 + b ≠ 0) : k < 0 ∧ b > 0 :=
sorry

end linear_function_third_quadrant_and_origin_l2276_227621


namespace find_m_value_l2276_227652

noncomputable def is_direct_proportion_function (m : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x : ℝ, (m - 1) * x ^ (2 - m^2) = k * x

theorem find_m_value (m : ℝ) (hk : ∃ k : ℝ, k ≠ 0 ∧ ∀ x : ℝ, (m - 1) * x ^ (2 - m^2) = k * x) : m = -1 :=
by
  sorry

end find_m_value_l2276_227652


namespace coordinates_A_B_l2276_227657

theorem coordinates_A_B : 
  (∃ x, 7 * x + 2 * 3 = 41) ∧ (∃ y, 7 * (-5) + 2 * y = 41) → 
  ((∃ x, x = 5) ∧ (∃ y, y = 38)) :=
by
  sorry

end coordinates_A_B_l2276_227657


namespace geom_seq_thm_l2276_227661

noncomputable def geom_seq (a : ℕ → ℝ) : Prop :=
a 1 ≠ 0 ∧ ∀ n, a (n + 1) = (a n ^ 2) / (a (n - 1))

theorem geom_seq_thm (a : ℕ → ℝ) (h : geom_seq a) (h_neg : ∀ n, a n < 0) 
  (h_eq : a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 36) : a 3 + a 5 = -6 :=
by
  sorry

end geom_seq_thm_l2276_227661


namespace rationalize_denominator_simplify_l2276_227650

theorem rationalize_denominator_simplify :
  let a : ℝ := 3
  let b : ℝ := 2
  let c : ℝ := 1
  let d : ℝ := 2
  ∀ (x y z : ℝ), 
  (x = 3 * Real.sqrt 2) → 
  (y = 3) → 
  (z = Real.sqrt 3) → 
  (x / (y - z) = (3 * Real.sqrt 2 + Real.sqrt 6) / 2) :=
by
  sorry

end rationalize_denominator_simplify_l2276_227650


namespace find_a_l2276_227625

theorem find_a (a : ℝ) : (1 : ℝ)^2 + 1 + 2 * a = 0 → a = -1 := 
by 
  sorry

end find_a_l2276_227625


namespace parallel_segments_l2276_227698

structure Point2D where
  x : Int
  y : Int

def vector (P Q : Point2D) : Point2D :=
  { x := Q.x - P.x, y := Q.y - P.y }

def is_parallel (v1 v2 : Point2D) : Prop :=
  ∃ k : Int, v2.x = k * v1.x ∧ v2.y = k * v1.y 

theorem parallel_segments :
  let A := { x := 1, y := 3 }
  let B := { x := 2, y := -1 }
  let C := { x := 0, y := 4 }
  let D := { x := 2, y := -4 }
  is_parallel (vector A B) (vector C D) := 
  sorry

end parallel_segments_l2276_227698


namespace total_integers_at_least_eleven_l2276_227641

theorem total_integers_at_least_eleven (n neg_count : ℕ) 
  (h1 : neg_count % 2 = 1)
  (h2 : neg_count ≤ 11) :
  n ≥ 11 := 
sorry

end total_integers_at_least_eleven_l2276_227641


namespace sum_possible_values_l2276_227630

theorem sum_possible_values (N : ℤ) (h : N * (N - 8) = -7) : 
  ∀ (N1 N2 : ℤ), (N1 * (N1 - 8) = -7) ∧ (N2 * (N2 - 8) = -7) → (N1 + N2 = 8) :=
by
  sorry

end sum_possible_values_l2276_227630


namespace mean_problem_l2276_227674

theorem mean_problem (x : ℝ) (h : (12 + x + 42 + 78 + 104) / 5 = 62) :
  (128 + 255 + 511 + 1023 + x) / 5 = 398.2 :=
by
  sorry

end mean_problem_l2276_227674


namespace arithmetic_sequence_length_l2276_227629

theorem arithmetic_sequence_length :
  ∃ n : ℕ, n > 0 ∧ ∀ (a_1 a_2 a_n : ℤ), a_1 = 2 ∧ a_2 = 6 ∧ a_n = 2006 →
  a_n = a_1 + (n - 1) * (a_2 - a_1) → n = 502 := by
  sorry

end arithmetic_sequence_length_l2276_227629


namespace no_solution_l2276_227628

theorem no_solution (x y n : ℕ) (hx : 0 < x) (hy : 0 < y) (hn : 0 < n) : 
  ¬ (x^2 + y^2 + 41 = 2^n) :=
by sorry

end no_solution_l2276_227628


namespace initial_volume_of_mixture_l2276_227608

/-- A mixture contains 10% water. 
5 liters of water should be added to this so that the water becomes 20% in the new mixture.
Prove that the initial volume of the mixture is 40 liters. -/
theorem initial_volume_of_mixture 
  (V : ℚ) -- Define the initial volume of the mixture
  (h1 : 0.10 * V + 5 = 0.20 * (V + 5)) -- Condition on the mixture
  : V = 40 := -- The statement to prove
by
  sorry -- Proof not required

end initial_volume_of_mixture_l2276_227608


namespace slope_range_l2276_227668

theorem slope_range {A : ℝ × ℝ} (k : ℝ) : 
  A = (1, 1) → (0 < 1 - k ∧ 1 - k < 2) → -1 < k ∧ k < 1 :=
by
  sorry

end slope_range_l2276_227668


namespace derek_joe_ratio_l2276_227611

theorem derek_joe_ratio (D J T : ℝ) (h0 : J = 23) (h1 : T = 30) (h2 : T = (1/3 : ℝ) * D + 16) :
  D / J = 42 / 23 :=
by
  sorry

end derek_joe_ratio_l2276_227611


namespace time_to_empty_l2276_227696

-- Definitions for the conditions
def rate_fill_no_leak (R : ℝ) := R = 1 / 2 -- Cistern fills in 2 hours without leak
def effective_fill_rate (R L : ℝ) := R - L = 1 / 4 -- Effective fill rate when leaking
def remember_fill_time_leak (R L : ℝ) := (R - L) * 4 = 1 -- 4 hours to fill with leak

-- Main theorem statement
theorem time_to_empty (R L : ℝ) (h1 : rate_fill_no_leak R) (h2 : effective_fill_rate R L)
  (h3 : remember_fill_time_leak R L) : (1 / L = 4) :=
by
  sorry

end time_to_empty_l2276_227696


namespace find_angles_l2276_227690

theorem find_angles (A B : ℝ) (h1 : A + B = 90) (h2 : A = 4 * B) : A = 72 ∧ B = 18 :=
by {
  sorry
}

end find_angles_l2276_227690


namespace sum_of_sequence_l2276_227607

noncomputable def sequence_sum (a : ℝ) (n : ℕ) : ℝ :=
if a = 1 then sorry else (5 * (1 - a ^ n) / (1 - a) ^ 2) - (4 + (5 * n - 4) * a ^ n) / (1 - a)

theorem sum_of_sequence (S : ℕ → ℝ) (a : ℝ) (h1 : S 1 = 1)
                       (h2 : ∀ n, S (n + 1) - S n = (5 * n + 1) * a ^ n) (h3 : |a| ≠ 1) :
  ∀ n, S n = sequence_sum a n :=
  sorry

end sum_of_sequence_l2276_227607


namespace candies_remaining_after_yellow_eaten_l2276_227688

theorem candies_remaining_after_yellow_eaten :
  let red_candies := 40
  let yellow_candies := 3 * red_candies - 20
  let blue_candies := yellow_candies / 2
  red_candies + blue_candies = 90 :=
by
  sorry

end candies_remaining_after_yellow_eaten_l2276_227688


namespace jason_pokemon_cards_l2276_227676

theorem jason_pokemon_cards :
  ∀ (initial_cards trade_benny_lost trade_benny_gain trade_sean_lost trade_sean_gain give_to_brother : ℕ),
  initial_cards = 5 →
  trade_benny_lost = 2 →
  trade_benny_gain = 3 →
  trade_sean_lost = 3 →
  trade_sean_gain = 4 →
  give_to_brother = 2 →
  initial_cards - trade_benny_lost + trade_benny_gain - trade_sean_lost + trade_sean_gain - give_to_brother = 5 :=
by
  intros
  sorry

end jason_pokemon_cards_l2276_227676


namespace sum_geometric_sequence_divisibility_l2276_227605

theorem sum_geometric_sequence_divisibility (n : ℕ) (h_pos: n > 0) :
  (n % 2 = 1 ↔ (3^(n+1) - 2^(n+1)) % 5 = 0) :=
sorry

end sum_geometric_sequence_divisibility_l2276_227605


namespace total_players_l2276_227665

-- Definitions for conditions
def K : Nat := 10
def KK : Nat := 30
def B : Nat := 5

-- Statement of the proof problem
theorem total_players : K + KK - B = 35 :=
by
  -- Proof not required, just providing the statement
  sorry

end total_players_l2276_227665


namespace last_integer_in_sequence_is_21853_l2276_227638

def is_divisible_by (n m : ℕ) : Prop := 
  ∃ k : ℕ, n = m * k

-- Conditions
def starts_with : ℕ := 590049
def divides_previous (a b : ℕ) : Prop := b = a / 3

-- The target hypothesis to prove
theorem last_integer_in_sequence_is_21853 :
  ∀ (a b c d : ℕ),
    a = starts_with →
    divides_previous a b →
    divides_previous b c →
    divides_previous c d →
    ¬ is_divisible_by d 3 →
    d = 21853 :=
by
  intros a b c d ha hb hc hd hnd
  sorry

end last_integer_in_sequence_is_21853_l2276_227638


namespace simplify_expression_l2276_227653

theorem simplify_expression : 
    1 - 1 / (1 + Real.sqrt (2 + Real.sqrt 3)) + 1 / (1 - Real.sqrt (2 - Real.sqrt 3)) 
    = 1 + (Real.sqrt (2 - Real.sqrt 3) + Real.sqrt (2 + Real.sqrt 3)) / (-1 - Real.sqrt 3) := 
by
  sorry

end simplify_expression_l2276_227653


namespace find_n_plus_c_l2276_227659

variables (n c : ℝ)

-- Conditions from the problem
def line1 := ∀ (x y : ℝ), (x = 4) → (y = 11) → (y = n * x + 3)
def line2 := ∀ (x y : ℝ), (x = 4) → (y = 11) → (y = 5 * x + c)

theorem find_n_plus_c (h1 : line1 n)
                      (h2 : line2 c) :
  n + c = -7 := by
  sorry

end find_n_plus_c_l2276_227659


namespace false_proposition_p_and_q_l2276_227692

open Classical

-- Define the propositions
def p (a b c : ℝ) : Prop := b * b = a * c
def q (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- We provide the conditions specified in the problem
variable (a b c : ℝ)
variable (f : ℝ → ℝ)
axiom hq : ∀ x, f x = f (-x)
axiom hp : ¬ (∀ a b c, p a b c ↔ (b ≠ 0 ∧ c ≠ 0 ∧ b * b = a * c))

-- The false proposition among the given options is "p and q"
theorem false_proposition_p_and_q : ¬ (∀ a b c (f : ℝ → ℝ), p a b c ∧ q f) :=
by
  -- This is where the proof would go, but is marked as a placeholder
  sorry

end false_proposition_p_and_q_l2276_227692


namespace number_of_cows_consume_in_96_days_l2276_227694

-- Given conditions
def grass_growth_rate := 10 / 3
def consumption_by_70_cows_in_24_days := 70 * 24
def consumption_by_30_cows_in_60_days := 30 * 60
def total_grass_in_96_days := consumption_by_30_cows_in_60_days + 120

-- Problem statement
theorem number_of_cows_consume_in_96_days : 
  (x : ℕ) -> 96 * x = total_grass_in_96_days -> x = 20 :=
by
  intros x h
  sorry

end number_of_cows_consume_in_96_days_l2276_227694


namespace total_wicks_20_l2276_227623

theorem total_wicks_20 (string_length_ft : ℕ) (length_wick_1 length_wick_2 : ℕ) (wicks_1 wicks_2 : ℕ) :
  string_length_ft = 15 →
  length_wick_1 = 6 →
  length_wick_2 = 12 →
  wicks_1 = wicks_2 →
  (string_length_ft * 12) = (length_wick_1 * wicks_1 + length_wick_2 * wicks_2) →
  (wicks_1 + wicks_2) = 20 :=
by
  intro h1 h2 h3 h4 h5
  sorry

end total_wicks_20_l2276_227623


namespace larger_number_is_sixty_three_l2276_227649

theorem larger_number_is_sixty_three (x y : ℕ) (h1 : x + y = 84) (h2 : y = 3 * x) : y = 63 :=
  sorry

end larger_number_is_sixty_three_l2276_227649


namespace sum_has_minimum_term_then_d_positive_Sn_positive_then_increasing_sequence_l2276_227646

def is_sum_of_arithmetic_sequence (S : ℕ → ℚ) (a₁ d : ℚ) :=
  ∀ n : ℕ, S n = n * a₁ + (n * (n - 1) / 2) * d

theorem sum_has_minimum_term_then_d_positive
  {S : ℕ → ℚ} {d a₁ : ℚ} (h : d ≠ 0)
  (hS : is_sum_of_arithmetic_sequence S a₁ d)
  (h_min : ∃ n : ℕ, ∀ m : ℕ, S n ≤ S m) :
  d > 0 :=
sorry

theorem Sn_positive_then_increasing_sequence
  {S : ℕ → ℚ} {d a₁ : ℚ} (h : d ≠ 0)
  (hS : is_sum_of_arithmetic_sequence S a₁ d)
  (h_pos : ∀ n : ℕ, S n > 0) :
  (∀ n : ℕ, S n < S (n + 1)) :=
sorry

end sum_has_minimum_term_then_d_positive_Sn_positive_then_increasing_sequence_l2276_227646


namespace construct_origin_from_A_and_B_l2276_227643

structure Point where
  x : ℝ
  y : ℝ

def A : Point := ⟨1, 2⟩
def B : Point := ⟨3, 1⟩
def isAboveAndToLeft (p₁ p₂ : Point) : Prop := p₁.x < p₂.x ∧ p₁.y > p₂.y
def isOriginConstructed (A B : Point) : Prop := ∃ O : Point, O = ⟨0, 0⟩

theorem construct_origin_from_A_and_B : 
  isAboveAndToLeft A B → isOriginConstructed A B :=
by
  sorry

end construct_origin_from_A_and_B_l2276_227643


namespace slope_of_asymptotes_l2276_227655

theorem slope_of_asymptotes (a b : ℝ) (h : a^2 = 144) (k : b^2 = 81) : (b / a = 3 / 4) :=
by
  sorry

end slope_of_asymptotes_l2276_227655


namespace prod_sum_leq_four_l2276_227695

theorem prod_sum_leq_four (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h_sum : a + b + c + d = 4) :
  ab + bc + cd + da ≤ 4 :=
sorry

end prod_sum_leq_four_l2276_227695


namespace max_value_3x_4y_l2276_227637

noncomputable def y_geom_mean (x y : ℝ) : Prop :=
  y^2 = (1 - x) * (1 + x)

theorem max_value_3x_4y (x y : ℝ) (h : y_geom_mean x y) : 3 * x + 4 * y ≤ 5 :=
sorry

end max_value_3x_4y_l2276_227637


namespace other_candidate_votes_l2276_227670

theorem other_candidate_votes (h1 : one_candidate_votes / valid_votes = 0.6)
    (h2 : 0.3 * total_votes = invalid_votes)
    (h3 : total_votes = 9000)
    (h4 : valid_votes + invalid_votes = total_votes) :
    valid_votes - one_candidate_votes = 2520 :=
by
  sorry

end other_candidate_votes_l2276_227670


namespace pigs_count_l2276_227604

-- Definitions from step a)
def pigs_leg_count : ℕ := 4 -- Each pig has 4 legs
def hens_leg_count : ℕ := 2 -- Each hen has 2 legs

variable {P H : ℕ} -- P is the number of pigs, H is the number of hens

-- Condition from step a) as a function
def total_legs (P H : ℕ) : ℕ := pigs_leg_count * P + hens_leg_count * H
def total_heads (P H : ℕ) : ℕ := P + H

-- Theorem to prove the number of pigs given the condition
theorem pigs_count {P H : ℕ} (h : total_legs P H = 2 * total_heads P H + 22) : P = 11 :=
  by 
    sorry

end pigs_count_l2276_227604


namespace quadratic_general_form_l2276_227654

theorem quadratic_general_form :
  ∀ x : ℝ, (x - 2) * (x + 3) = 1 → x^2 + x - 7 = 0 :=
by
  intros x h
  sorry

end quadratic_general_form_l2276_227654


namespace symmetrical_point_l2276_227682

-- Definition of symmetry with respect to the x-axis
def symmetrical (x y: ℝ) : ℝ × ℝ := (x, -y)

-- Coordinates of the original point A
def A : ℝ × ℝ := (-2, 3)

-- Coordinates of the symmetrical point
def symmetrical_A : ℝ × ℝ := symmetrical (-2) 3

-- The theorem we want to prove
theorem symmetrical_point :
  symmetrical_A = (-2, -3) :=
by
  -- Provide the proof here
  sorry

end symmetrical_point_l2276_227682


namespace parabola_x_intercepts_l2276_227627

theorem parabola_x_intercepts :
  ∃! (x : ℝ), ∃ (y : ℝ), y = 0 ∧ x = -2 * y^2 + y + 1 :=
sorry

end parabola_x_intercepts_l2276_227627


namespace typeB_lines_l2276_227679

noncomputable def isTypeBLine (line : Real → Real) : Prop :=
  ∃ P : ℝ × ℝ, line P.1 = P.2 ∧ (Real.sqrt ((P.1 + 5)^2 + P.2^2) - Real.sqrt ((P.1 - 5)^2 + P.2^2) = 6)

theorem typeB_lines :
  isTypeBLine (fun x => x + 1) ∧ isTypeBLine (fun x => 2) :=
by sorry

end typeB_lines_l2276_227679


namespace intersection_M_N_l2276_227642

def M : Set ℝ := { x | x^2 - x - 6 ≤ 0 }
def N : Set ℝ := { x | -2 < x ∧ x ≤ 4 }

theorem intersection_M_N : (M ∩ N) = { x | -2 < x ∧ x ≤ 3 } :=
by
  sorry

end intersection_M_N_l2276_227642


namespace crowdfunding_successful_l2276_227658

variable (highest_level second_level lowest_level total_amount : ℕ)
variable (x y z : ℕ)

noncomputable def crowdfunding_conditions (highest_level second_level lowest_level : ℕ) := 
  second_level = highest_level / 10 ∧ lowest_level = second_level / 10

noncomputable def total_raised (highest_level second_level lowest_level x y z : ℕ) :=
  highest_level * x + second_level * y + lowest_level * z

theorem crowdfunding_successful (h1 : highest_level = 5000) 
                                (h2 : crowdfunding_conditions highest_level second_level lowest_level) 
                                (h3 : total_amount = 12000) 
                                (h4 : y = 3) 
                                (h5 : z = 10) :
  total_raised highest_level second_level lowest_level x y z = total_amount → x = 2 := by
  sorry

end crowdfunding_successful_l2276_227658


namespace distinct_exponentiation_values_l2276_227693

theorem distinct_exponentiation_values : 
  (∃ v1 v2 v3 v4 v5 : ℕ, 
    v1 = (3 : ℕ)^(3 : ℕ)^(3 : ℕ)^(3 : ℕ) ∧
    v2 = (3 : ℕ)^((3 : ℕ)^(3 : ℕ)^(3 : ℕ)) ∧
    v3 = (3 : ℕ)^(((3 : ℕ)^(3 : ℕ))^(3 : ℕ)) ∧
    v4 = ((3 : ℕ)^(3 : ℕ)^3) ∧
    v5 = ((3 : ℕ)^((3 : ℕ)^(3 : ℕ)^3)) ∧
    v1 ≠ v2 ∧ v1 ≠ v3 ∧ v1 ≠ v4 ∧ v1 ≠ v5 ∧ 
    v2 ≠ v3 ∧ v2 ≠ v4 ∧ v2 ≠ v5 ∧ 
    v3 ≠ v4 ∧ v3 ≠ v5 ∧ 
    v4 ≠ v5) := 
sorry

end distinct_exponentiation_values_l2276_227693


namespace songs_distribution_l2276_227648

-- Define the sets involved
structure Girl := (Amy Beth Jo : Prop)
axiom no_song_liked_by_all : ∀ song : Girl, ¬(song.Amy ∧ song.Beth ∧ song.Jo)
axiom no_song_disliked_by_all : ∀ song : Girl, song.Amy ∨ song.Beth ∨ song.Jo
axiom pairwise_liked : ∀ song : Girl,
  (song.Amy ∧ song.Beth ∧ ¬song.Jo) ∨
  (song.Beth ∧ song.Jo ∧ ¬song.Amy) ∨
  (song.Jo ∧ song.Amy ∧ ¬song.Beth)

-- Define the theorem to prove that there are exactly 90 ways to distribute the songs
theorem songs_distribution : ∃ ways : ℕ, ways = 90 := sorry

end songs_distribution_l2276_227648


namespace number_of_large_boxes_l2276_227624

theorem number_of_large_boxes (total_boxes : ℕ) (small_weight large_weight remaining_small remaining_large : ℕ) :
  total_boxes = 62 →
  small_weight = 5 →
  large_weight = 3 →
  remaining_small = 15 →
  remaining_large = 15 →
  ∀ (small_boxes large_boxes : ℕ),
    total_boxes = small_boxes + large_boxes →
    ((large_boxes * large_weight) + (remaining_small * small_weight) = (small_boxes * small_weight) + (remaining_large * large_weight)) →
    large_boxes = 27 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end number_of_large_boxes_l2276_227624


namespace certain_number_k_l2276_227634

theorem certain_number_k (x : ℕ) (k : ℕ) (h1 : x = 14) (h2 : 2^x - 2^(x-2) = k * 2^12) : k = 3 := by
  sorry

end certain_number_k_l2276_227634


namespace firetruck_reachable_area_l2276_227610

theorem firetruck_reachable_area :
  let m := 700
  let n := 31
  let area := m / n -- The area in square miles
  let time := 1 / 10 -- The available time in hours
  let speed_highway := 50 -- Speed on the highway in miles/hour
  let speed_prairie := 14 -- Speed across the prairie in miles/hour
  -- The intersection point of highways is the origin (0, 0)
  -- The firetruck can move within the reachable area
  -- There exist regions formed by the intersection points of movement directions
  m + n = 731 :=
by
  sorry

end firetruck_reachable_area_l2276_227610


namespace solve_for_x_l2276_227632

variable (a b c d x : ℝ)

theorem solve_for_x (h1 : a ≠ b) (h2 : b ≠ 0) 
  (h3 : d ≠ c) (h4 : c % x = 0) (h5 : d % x = 0) 
  (h6 : (2*a + x) / (3*b + x) = c / d) : 
  x = (3*b*c - 2*a*d) / (d - c) := 
sorry

end solve_for_x_l2276_227632


namespace number_of_n_with_odd_tens_digit_in_square_l2276_227662

def ends_in_3_or_7 (n : ℕ) : Prop :=
  n % 10 = 3 ∨ n % 10 = 7

def tens_digit_odd (n : ℕ) : Prop :=
  ((n * n / 10) % 10) % 2 = 1

theorem number_of_n_with_odd_tens_digit_in_square :
  ∀ n ∈ {n : ℕ | n ≤ 50 ∧ ends_in_3_or_7 n}, ¬tens_digit_odd n :=
by 
  sorry

end number_of_n_with_odd_tens_digit_in_square_l2276_227662


namespace cylinder_ellipse_major_axis_l2276_227675

-- Given a right circular cylinder of radius 2
-- and a plane intersecting it forming an ellipse
-- with the major axis being 50% longer than the minor axis,
-- prove that the length of the major axis is 6.

theorem cylinder_ellipse_major_axis :
  ∀ (r : ℝ) (major minor : ℝ),
    r = 2 → major = 1.5 * minor → minor = 2 * r → major = 6 :=
by
  -- Proof step to be filled by the prover.
  sorry

end cylinder_ellipse_major_axis_l2276_227675


namespace solve_for_x_l2276_227620

theorem solve_for_x (x : ℚ) : (1 / 3) + (1 / x) = (3 / 4) → x = 12 / 5 :=
by
  intro h
  -- Proof goes here
  sorry

end solve_for_x_l2276_227620


namespace samantha_spends_on_dog_toys_l2276_227677

theorem samantha_spends_on_dog_toys:
  let toy_price := 12.00
  let discount := 0.5
  let num_toys := 4
  let tax_rate := 0.08
  let full_price_toys := num_toys / 2
  let half_price_toys := num_toys / 2
  let total_cost_before_tax := full_price_toys * toy_price + half_price_toys * (toy_price * discount)
  let sales_tax := total_cost_before_tax * tax_rate
  let total_cost_after_tax := total_cost_before_tax + sales_tax
  total_cost_after_tax = 38.88 :=
by {
  sorry
}

end samantha_spends_on_dog_toys_l2276_227677


namespace select_10_teams_l2276_227699

def football_problem (teams : Finset ℕ) (played_on_day1 : Finset (ℕ × ℕ)) (played_on_day2 : Finset (ℕ × ℕ)) : Prop :=
  ∀ (v : ℕ), v ∈ teams → (∃ u w : ℕ, (u, v) ∈ played_on_day1 ∧ (v, w) ∈ played_on_day2)

theorem select_10_teams {teams : Finset ℕ}
  (h : teams.card = 20)
  {played_on_day1 played_on_day2 : Finset (ℕ × ℕ)}
  (h1 : ∀ ⦃u v : ℕ⦄, (u, v) ∈ played_on_day1 → u ∈ teams ∧ v ∈ teams)
  (h2 : ∀ ⦃u v : ℕ⦄, (u, v) ∈ played_on_day2 → u ∈ teams ∧ v ∈ teams)
  (h3 : ∀ x ∈ teams, ∃ u w, (u, x) ∈ played_on_day1 ∧ (x, w) ∈ played_on_day2) :
  ∃ S : Finset ℕ, S.card = 10 ∧ (∀ ⦃x y⦄, x ∈ S → y ∈ S → x ≠ y → (¬((x, y) ∈ played_on_day1) ∧ ¬((x, y) ∈ played_on_day2))) :=
by
  sorry

end select_10_teams_l2276_227699
