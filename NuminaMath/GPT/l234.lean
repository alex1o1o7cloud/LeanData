import Mathlib

namespace least_number_of_square_tiles_l234_234249

theorem least_number_of_square_tiles
  (length_cm : ℕ) (width_cm : ℕ)
  (h1 : length_cm = 816) (h2 : width_cm = 432) :
  ∃ tile_count : ℕ, tile_count = 153 :=
by
  sorry

end least_number_of_square_tiles_l234_234249


namespace find_c_l234_234005

theorem find_c (c : ℝ) (h : ∃ (f : ℝ → ℝ), (f = λ x => c * x^3 + 23 * x^2 - 5 * c * x + 55) ∧ f (-5) = 0) : c = 6.3 := 
by {
  sorry
}

end find_c_l234_234005


namespace distance_from_Q_to_EG_l234_234621

noncomputable def distance_to_line : ℝ :=
  let E := (0, 5)
  let F := (5, 5)
  let G := (5, 0)
  let H := (0, 0)
  let N := (2.5, 0)
  let Q := (25 / 7, 10 / 7)
  let line_y := 5
  let distance := abs (line_y - Q.2)
  distance

theorem distance_from_Q_to_EG : distance_to_line = 25 / 7 :=
by
  sorry

end distance_from_Q_to_EG_l234_234621


namespace arithmetic_sequence_first_term_l234_234967

theorem arithmetic_sequence_first_term :
  ∃ a₁ a₂ d : ℤ, a₂ = -5 ∧ d = 3 ∧ a₂ = a₁ + d ∧ a₁ = -8 :=
by
  sorry

end arithmetic_sequence_first_term_l234_234967


namespace intersect_A_B_when_a_1_subset_A_B_range_a_l234_234418

def poly_eqn (x : ℝ) : Prop := -x ^ 2 - 2 * x + 8 = 0

def sol_set_A : Set ℝ := {x | poly_eqn x}

def inequality (a x : ℝ) : Prop := a * x - 1 ≤ 0

def sol_set_B (a : ℝ) : Set ℝ := {x | inequality a x}

theorem intersect_A_B_when_a_1 :
  sol_set_A ∩ sol_set_B 1 = { -4 } :=
sorry

theorem subset_A_B_range_a (a : ℝ) :
  sol_set_A ⊆ sol_set_B a ↔ (-1 / 4 : ℝ) ≤ a ∧ a ≤ 1 / 2 :=
sorry
 
end intersect_A_B_when_a_1_subset_A_B_range_a_l234_234418


namespace sequence_formula_l234_234780

noncomputable def a (n : ℕ) : ℕ :=
if n = 0 then 1 else (a (n - 1)) + 2^(n-1)

theorem sequence_formula (n : ℕ) (h : n > 0) : 
    a n = 2^n - 1 := 
sorry

end sequence_formula_l234_234780


namespace cd_cost_l234_234469

theorem cd_cost (mp3_cost savings father_amt lacks cd_cost : ℝ) :
  mp3_cost = 120 ∧ savings = 55 ∧ father_amt = 20 ∧ lacks = 64 →
  120 + cd_cost - (savings + father_amt) = lacks → 
  cd_cost = 19 :=
by
  intros
  sorry

end cd_cost_l234_234469


namespace value_of_x_l234_234990

theorem value_of_x (x : ℝ) : 
  (x ≤ 0 → x^2 + 1 = 5 → x = -2) ∧ 
  (0 < x → -2 * x = 5 → false) := 
sorry

end value_of_x_l234_234990


namespace car_b_speed_l234_234260

/--
A car A going at 30 miles per hour set out on an 80-mile trip at 9:00 a.m.
Exactly 10 minutes later, a car B left from the same place and followed the same route.
Car B caught up with car A at 10:30 a.m.
Prove that the speed of car B is 33.75 miles per hour.
-/
theorem car_b_speed
    (v_a : ℝ) (t_start_a t_start_b t_end : ℝ) (v_b : ℝ)
    (h1 : v_a = 30) 
    (h2 : t_start_a = 9) 
    (h3 : t_start_b = 9 + (10 / 60)) 
    (h4 : t_end = 10.5) 
    (h5 : t_end - t_start_b = (4 / 3))
    (h6 : v_b * (t_end - t_start_b) = v_a * (t_end - t_start_a) + (v_a * (10 / 60))) :
  v_b = 33.75 := 
sorry

end car_b_speed_l234_234260


namespace pool_capacity_l234_234946

theorem pool_capacity (C : ℝ) (h1 : C * 0.70 = C * 0.40 + 300)
  (h2 : 300 = C * 0.30) : C = 1000 :=
sorry

end pool_capacity_l234_234946


namespace marbles_initial_count_l234_234352

theorem marbles_initial_count :
  let total_customers := 20
  let marbles_per_customer := 15
  let marbles_remaining := 100
  ∃ initial_marbles, initial_marbles = total_customers * marbles_per_customer + marbles_remaining :=
by
  let total_customers := 20
  let marbles_per_customer := 15
  let marbles_remaining := 100
  existsi (total_customers * marbles_per_customer + marbles_remaining)
  rfl

end marbles_initial_count_l234_234352


namespace cos_beta_value_l234_234276

theorem cos_beta_value
  (α β : ℝ)
  (hαβ : 0 < α ∧ α < π ∧ 0 < β ∧ β < π)
  (h1 : Real.sin (α + β) = 5 / 13)
  (h2 : Real.tan (α / 2) = 1 / 2) :
  Real.cos β = -16 / 65 := 
by 
  sorry

end cos_beta_value_l234_234276


namespace cos_2_alpha_plus_beta_eq_l234_234114

variable (α β : ℝ)

def tan_roots_of_quadratic (x : ℝ) : Prop := x^2 + 5 * x - 6 = 0

theorem cos_2_alpha_plus_beta_eq :
  ∀ α β : ℝ, tan_roots_of_quadratic (Real.tan α) ∧ tan_roots_of_quadratic (Real.tan β) →
  Real.cos (2 * (α + β)) = 12 / 37 :=
by
  intros
  sorry

end cos_2_alpha_plus_beta_eq_l234_234114


namespace a_n_formula_b_n_formula_S_n_formula_l234_234213

noncomputable def a_n (n : ℕ) : ℕ := 3 * n
noncomputable def b_n (n : ℕ) : ℕ := 2^(n-1) + 3 * n
noncomputable def S_n (n : ℕ) : ℕ := 2^n - 1 + (3 * n^2 + 3 * n) / 2

theorem a_n_formula (n : ℕ) : a_n n = 3 * n := by
  unfold a_n
  rfl

theorem b_n_formula (n : ℕ) : b_n n = 2^(n-1) + 3 * n := by
  unfold b_n
  rfl

theorem S_n_formula (n : ℕ) : S_n n = 2^n - 1 + (3 * n^2 + 3 * n) / 2 := by
  unfold S_n
  rfl

end a_n_formula_b_n_formula_S_n_formula_l234_234213


namespace sum_product_poly_roots_eq_l234_234681

theorem sum_product_poly_roots_eq (b c : ℝ) 
  (h1 : -1 + 2 = -b) 
  (h2 : (-1) * 2 = c) : c + b = -3 := 
by 
  sorry

end sum_product_poly_roots_eq_l234_234681


namespace percent_decrease_first_year_l234_234145

theorem percent_decrease_first_year (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 100) 
  (h_second_year : 0.9 * (100 - x) = 54) : x = 40 :=
by sorry

end percent_decrease_first_year_l234_234145


namespace min_value_expression_l234_234378

noncomputable 
def min_value_condition (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_abc : a * b * c = 1) : ℝ :=
  (a + 1) * (b + 1) * (c + 1)

theorem min_value_expression (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_abc : a * b * c = 1) : 
  min_value_condition a b c h_pos h_abc = 8 :=
sorry

end min_value_expression_l234_234378


namespace probability_of_five_3s_is_099_l234_234483

-- Define conditions
def number_of_dice : ℕ := 15
def rolled_value : ℕ := 3
def probability_of_3 : ℚ := 1 / 8
def number_of_successes : ℕ := 5
def probability_of_not_3 : ℚ := 7 / 8

-- Define the binomial coefficient function
def binomial_coefficient (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Define the probability calculation
def probability_exactly_five_3s : ℚ :=
  binomial_coefficient number_of_dice number_of_successes *
  probability_of_3 ^ number_of_successes *
  probability_of_not_3 ^ (number_of_dice - number_of_successes)

theorem probability_of_five_3s_is_099 :
  probability_exactly_five_3s = 0.099 := by
  sorry -- Proof to be filled in later

end probability_of_five_3s_is_099_l234_234483


namespace charlene_gave_18_necklaces_l234_234581

theorem charlene_gave_18_necklaces
  (initial_necklaces : ℕ) (sold_necklaces : ℕ) (left_necklaces : ℕ)
  (h1 : initial_necklaces = 60)
  (h2 : sold_necklaces = 16)
  (h3 : left_necklaces = 26) :
  initial_necklaces - sold_necklaces - left_necklaces = 18 :=
by
  sorry

end charlene_gave_18_necklaces_l234_234581


namespace perpendicular_case_parallel_case_l234_234327

noncomputable def vector_a : ℝ × ℝ := (1, 2)
noncomputable def vector_b : ℝ × ℝ := (-3, 2)
noncomputable def k_perpendicular : ℝ := 19
noncomputable def k_parallel : ℝ := -1/3

-- Define the operations used:
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

def scalar_mult (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (k * v.1, k * v.2)

def vector_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 + v2.1, v1.2 + v2.2)

def vector_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 - v2.1, v1.2 - v2.2)

-- Perpendicular case: 
theorem perpendicular_case : dot_product (vector_add (scalar_mult k_perpendicular vector_a) vector_b) (vector_sub vector_a (scalar_mult 3 vector_b)) = 0 := sorry

-- Parallel case:
theorem parallel_case : ∃ c : ℝ, vector_add (scalar_mult k_parallel vector_a) vector_b = scalar_mult c (vector_sub vector_a (scalar_mult 3 vector_b)) ∧ c < 0 := sorry

end perpendicular_case_parallel_case_l234_234327


namespace joe_first_lift_weight_l234_234564

variable (x y : ℕ)

def conditions : Prop :=
  (x + y = 1800) ∧ (2 * x = y + 300)

theorem joe_first_lift_weight (h : conditions x y) : x = 700 := by
  sorry

end joe_first_lift_weight_l234_234564


namespace geometric_sequence_a5_eq_2_l234_234084

-- Define geometric sequence and the properties
noncomputable def geometric_seq (a : ℕ → ℝ) (q : ℝ) := ∀ n, a (n + 1) = a n * q

-- Given conditions
variables {a : ℕ → ℝ} {q : ℝ}

-- Roots of given quadratic equation
variables (h1 : a 3 = 1 ∨ a 3 = 4 / 1) (h2 : a 7 = 4 / a 3)
variables (h3 : q > 0) (h4 : geometric_seq a q)

-- Prove that a5 = 2
theorem geometric_sequence_a5_eq_2 : a 5 = 2 :=
sorry

end geometric_sequence_a5_eq_2_l234_234084


namespace isabella_purchases_l234_234188

def isabella_items_total (alexis_pants alexis_dresses isabella_pants isabella_dresses : ℕ) : ℕ :=
  isabella_pants + isabella_dresses

theorem isabella_purchases
  (alexis_pants : ℕ) (alexis_dresses : ℕ)
  (h_pants : alexis_pants = 21)
  (h_dresses : alexis_dresses = 18)
  (h_ratio : ∀ (x : ℕ), alexis_pants = 3 * x → alexis_dresses = 3 * x):
  isabella_items_total (21 / 3) (18 / 3) = 13 :=
by
  sorry

end isabella_purchases_l234_234188


namespace shorter_piece_is_28_l234_234486

noncomputable def shorter_piece_length (x : ℕ) : Prop :=
  x + (x + 12) = 68 → x = 28

theorem shorter_piece_is_28 (x : ℕ) : shorter_piece_length x :=
by
  intro h
  have h1 : 2 * x + 12 = 68 := by linarith
  have h2 : 2 * x = 56 := by linarith
  have h3 : x = 28 := by linarith
  exact h3

end shorter_piece_is_28_l234_234486


namespace cost_per_vent_l234_234175

/--
Given that:
1. The total cost of the HVAC system is $20,000.
2. The system includes 2 conditioning zones.
3. Each zone has 5 vents.

Prove that the cost per vent is $2000.
-/
theorem cost_per_vent (total_cost : ℕ) (zones : ℕ) (vents_per_zone : ℕ) (h1 : total_cost = 20000) (h2 : zones = 2) (h3 : vents_per_zone = 5) :
  total_cost / (zones * vents_per_zone) = 2000 := 
sorry

end cost_per_vent_l234_234175


namespace no_prime_solutions_for_x2_plus_y3_eq_z4_l234_234552

theorem no_prime_solutions_for_x2_plus_y3_eq_z4 :
  ¬ ∃ (x y z : ℕ), Prime x ∧ Prime y ∧ Prime z ∧ x^2 + y^3 = z^4 := sorry

end no_prime_solutions_for_x2_plus_y3_eq_z4_l234_234552


namespace bank_balance_after_two_years_l234_234601

-- Define the original amount deposited
def original_amount : ℝ := 5600

-- Define the interest rate
def interest_rate : ℝ := 0.07

-- Define the interest for each year based on the original amount
def interest_per_year : ℝ := original_amount * interest_rate

-- Define the total amount after two years
def total_amount_after_two_years : ℝ := original_amount + interest_per_year + interest_per_year

-- Define the target value
def target_value : ℝ := 6384

-- The theorem we aim to prove
theorem bank_balance_after_two_years : 
  total_amount_after_two_years = target_value := 
by
  -- Proof goes here
  sorry

end bank_balance_after_two_years_l234_234601


namespace analyze_properties_l234_234209

noncomputable def eq_condition (x a : ℝ) : Prop :=
x ≠ 0 ∧ a = (x - 1) / (x^2)

noncomputable def first_condition (x a : ℝ) : Prop :=
x⁻¹ + a * x = 1

noncomputable def second_condition (x a : ℝ) : Prop :=
x⁻¹ + a * x > 1

noncomputable def third_condition (x a : ℝ) : Prop :=
x⁻¹ + a * x < 1

theorem analyze_properties (x a : ℝ) (h1 : eq_condition x a):
(first_condition x a) ∧ ¬(second_condition x a) ∧ ¬(third_condition x a) :=
by
  sorry

end analyze_properties_l234_234209


namespace greatest_triangle_perimeter_l234_234697

theorem greatest_triangle_perimeter :
  ∃ (x : ℕ), 3 < x ∧ x < 6 ∧ max (x + 4 * x + 17) (5 + 4 * 5 + 17) = 42 :=
by
  sorry

end greatest_triangle_perimeter_l234_234697


namespace smallest_angle_in_trapezoid_l234_234808

theorem smallest_angle_in_trapezoid 
  (a d : ℝ) 
  (h1 : a + 2 * d = 150) 
  (h2 : a + d + a + 2 * d = 180) : 
  a = 90 := 
sorry

end smallest_angle_in_trapezoid_l234_234808


namespace functional_equation_solution_l234_234611

theorem functional_equation_solution (f : ℝ → ℝ) (h : ∀ x y : ℝ, f x * f y = f (x - y)) :
  (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = 1) :=
sorry

end functional_equation_solution_l234_234611


namespace polar_to_cartesian_conversion_l234_234514

noncomputable def polarToCartesian (ρ θ : ℝ) : ℝ × ℝ :=
  let x := ρ * Real.cos θ
  let y := ρ * Real.sin θ
  (x, y)

theorem polar_to_cartesian_conversion :
  polarToCartesian 4 (Real.pi / 3) = (2, 2 * Real.sqrt 3) :=
by
  sorry

end polar_to_cartesian_conversion_l234_234514


namespace find_s_l234_234911

theorem find_s (s t : ℝ) (h1 : 8 * s + 4 * t = 160) (h2 : t = 2 * s - 3) : s = 10.75 :=
by
  sorry

end find_s_l234_234911


namespace Duke_broke_record_by_5_l234_234450

theorem Duke_broke_record_by_5 :
  let free_throws := 5
  let regular_baskets := 4
  let normal_three_pointers := 2
  let extra_three_pointers := 1
  let points_per_free_throw := 1
  let points_per_regular_basket := 2
  let points_per_three_pointer := 3
  let points_to_tie_record := 17

  let total_points_scored := (free_throws * points_per_free_throw) +
                             (regular_baskets * points_per_regular_basket) +
                             ((normal_three_pointers + extra_three_pointers) * points_per_three_pointer)
  total_points_scored = 22 →
  total_points_scored - points_to_tie_record = 5 :=

by
  intros
  sorry

end Duke_broke_record_by_5_l234_234450


namespace fraction_irreducible_l234_234322

theorem fraction_irreducible (n : ℤ) : Int.gcd (39 * n + 4) (26 * n + 3) = 1 :=
sorry

end fraction_irreducible_l234_234322


namespace part1_part2_l234_234870

open Set

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - (a + 1 / a) * x + 1

theorem part1 (x : ℝ) : f 2 (2^x) ≤ 0 ↔ -1 ≤ x ∧ x ≤ 1 :=
by sorry

theorem part2 (a x : ℝ) (h : a > 2) : f a x ≥ 0 ↔ x ∈ (Iic (1/a) ∪ Ici a) :=
by sorry

end part1_part2_l234_234870


namespace distance_AC_l234_234301

theorem distance_AC (t_Eddy t_Freddy : ℕ) (d_AB : ℝ) (speed_ratio : ℝ) : 
  t_Eddy = 3 ∧ t_Freddy = 4 ∧ d_AB = 510 ∧ speed_ratio = 2.2666666666666666 → 
  ∃ d_AC : ℝ, d_AC = 300 :=
by 
  intros h
  obtain ⟨hE, hF, hD, hR⟩ := h
  -- Declare velocities
  let v_Eddy : ℝ := d_AB / t_Eddy
  let v_Freddy : ℝ := v_Eddy / speed_ratio
  let d_AC : ℝ := v_Freddy * t_Freddy
  -- Prove the distance
  use d_AC
  sorry

end distance_AC_l234_234301


namespace find_b_l234_234021

variable (x : ℝ)

theorem find_b (a b: ℝ) (h1 : x + 1/x = a) (h2 : x^3 + 1/x^3 = b) (ha : a = 3): b = 18 :=
by
  sorry

end find_b_l234_234021


namespace overall_profit_refrigerator_mobile_phone_l234_234982

theorem overall_profit_refrigerator_mobile_phone
  (purchase_price_refrigerator : ℕ)
  (purchase_price_mobile_phone : ℕ)
  (loss_percentage_refrigerator : ℕ)
  (profit_percentage_mobile_phone : ℕ)
  (selling_price_refrigerator : ℕ)
  (selling_price_mobile_phone : ℕ)
  (total_cost_price : ℕ)
  (total_selling_price : ℕ)
  (overall_profit : ℕ) :
  purchase_price_refrigerator = 15000 →
  purchase_price_mobile_phone = 8000 →
  loss_percentage_refrigerator = 4 →
  profit_percentage_mobile_phone = 10 →
  selling_price_refrigerator = purchase_price_refrigerator - (purchase_price_refrigerator * loss_percentage_refrigerator / 100) →
  selling_price_mobile_phone = purchase_price_mobile_phone + (purchase_price_mobile_phone * profit_percentage_mobile_phone / 100) →
  total_cost_price = purchase_price_refrigerator + purchase_price_mobile_phone →
  total_selling_price = selling_price_refrigerator + selling_price_mobile_phone →
  overall_profit = total_selling_price - total_cost_price →
  overall_profit = 200 :=
  by sorry

end overall_profit_refrigerator_mobile_phone_l234_234982


namespace cuboid_volume_l234_234361

theorem cuboid_volume (a b c : ℝ) (ha : a = 4) (hb : b = 5) (hc : c = 6) : a * b * c = 120 :=
by
  sorry

end cuboid_volume_l234_234361


namespace find_biology_marks_l234_234202

variable (english mathematics physics chemistry average_marks : ℕ)

theorem find_biology_marks
  (h_english : english = 86)
  (h_mathematics : mathematics = 85)
  (h_physics : physics = 92)
  (h_chemistry : chemistry = 87)
  (h_average_marks : average_marks = 89) : 
  (english + mathematics + physics + chemistry + (445 - (english + mathematics + physics + chemistry))) / 5 = average_marks :=
by
  sorry

end find_biology_marks_l234_234202


namespace least_positive_integer_reducible_fraction_l234_234669

theorem least_positive_integer_reducible_fraction :
  ∃ (n : ℕ), n > 0 ∧ (∀ m : ℕ, m > 0 → (∃ d : ℕ, d > 1 ∧ d ∣ (m - 10) ∧ d ∣ (9 * m + 11)) ↔ m ≥ n) ∧ n = 111 :=
by
  sorry

end least_positive_integer_reducible_fraction_l234_234669


namespace blaine_fish_caught_l234_234677

theorem blaine_fish_caught (B : ℕ) (cond1 : B + 2 * B = 15) : B = 5 := by 
  sorry

end blaine_fish_caught_l234_234677


namespace dress_total_price_correct_l234_234104

-- Define constants and variables
def original_price : ℝ := 120
def discount_rate : ℝ := 0.30
def tax_rate : ℝ := 0.15

-- Function to calculate sale price after discount
def sale_price (op : ℝ) (dr : ℝ) : ℝ := op - (op * dr)

-- Function to calculate total price including tax
def total_selling_price (sp : ℝ) (tr : ℝ) : ℝ := sp + (sp * tr)

-- The proof statement to be proven
theorem dress_total_price_correct :
  total_selling_price (sale_price original_price discount_rate) tax_rate = 96.6 :=
  by sorry

end dress_total_price_correct_l234_234104


namespace largest_integer_of_four_l234_234391

theorem largest_integer_of_four (a b c d : ℤ) 
  (h1 : a + b + c = 160) 
  (h2 : a + b + d = 185) 
  (h3 : a + c + d = 205) 
  (h4 : b + c + d = 230) : 
  max (max a (max b c)) d = 100 := 
by
  sorry

end largest_integer_of_four_l234_234391


namespace quadratic_function_correct_l234_234819

-- Defining the quadratic function a
def quadratic_function (x : ℝ) : ℝ := 2 * x^2 - 14 * x + 20

-- Theorem stating that the quadratic function passes through the points (2, 0) and (5, 0)
theorem quadratic_function_correct : 
  quadratic_function 2 = 0 ∧ quadratic_function 5 = 0 := 
by
  -- these proofs are skipped with sorry for now
  sorry

end quadratic_function_correct_l234_234819


namespace totalGoals_l234_234937

-- Define the conditions
def louieLastMatchGoals : Nat := 4
def louiePreviousGoals : Nat := 40
def gamesPerSeason : Nat := 50
def seasons : Nat := 3
def brotherGoalsPerGame := 2 * louieLastMatchGoals

-- Define the properties derived from the conditions
def totalBrotherGoals : Nat := brotherGoalsPerGame * gamesPerSeason * seasons
def totalLouieGoals : Nat := louiePreviousGoals + louieLastMatchGoals

-- State what needs to be proved
theorem totalGoals : louiePreviousGoals + louieLastMatchGoals + brotherGoalsPerGame * gamesPerSeason * seasons = 1244 := by
  sorry

end totalGoals_l234_234937


namespace sunglasses_price_l234_234190

theorem sunglasses_price (P : ℝ) 
  (buy_cost_per_pair : ℝ := 26) 
  (pairs_sold : ℝ := 10) 
  (sign_cost : ℝ := 20) :
  (pairs_sold * P - pairs_sold * buy_cost_per_pair) / 2 = sign_cost →
  P = 30 := 
by
  sorry

end sunglasses_price_l234_234190


namespace cyclic_quadrilateral_tangency_l234_234382

theorem cyclic_quadrilateral_tangency (a b c d x y : ℝ) (h_cyclic : a = 80 ∧ b = 100 ∧ c = 140 ∧ d = 120) 
  (h_tangency: x + y = 140) : |x - y| = 5 := 
sorry

end cyclic_quadrilateral_tangency_l234_234382


namespace geometric_progression_solution_l234_234598

theorem geometric_progression_solution (b4 b2 b6 : ℚ) (h1 : b4 - b2 = -45 / 32) (h2 : b6 - b4 = -45 / 512) :
  (∃ (b1 q : ℚ), b4 = b1 * q^3 ∧ b2 = b1 * q ∧ b6 = b1 * q^5 ∧ 
    ((b1 = 6 ∧ q = 1 / 4) ∨ (b1 = -6 ∧ q = -1 / 4))) :=
by
  sorry

end geometric_progression_solution_l234_234598


namespace fraction_of_historical_fiction_new_releases_l234_234128

theorem fraction_of_historical_fiction_new_releases (total_books : ℕ) (p1 p2 p3 : ℕ) (frac_hist_fic : Rat) (frac_new_hist_fic : Rat) (frac_new_non_hist_fic : Rat) 
  (h1 : total_books > 0) (h2 : frac_hist_fic = 40 / 100) (h3 : frac_new_hist_fic = 40 / 100) (h4 : frac_new_non_hist_fic = 40 / 100) 
  (h5 : p1 = frac_hist_fic * total_books) (h6 : p2 = frac_new_hist_fic * p1) (h7 : p3 = frac_new_non_hist_fic * (total_books - p1)) :
  p2 / (p2 + p3) = 2 / 5 :=
by
  sorry

end fraction_of_historical_fiction_new_releases_l234_234128


namespace smallest_among_neg2_cube_neg3_square_neg_neg1_l234_234817

def smallest_among (a b c : ℤ) : ℤ :=
if a < b then
  if a < c then a else c
else
  if b < c then b else c

theorem smallest_among_neg2_cube_neg3_square_neg_neg1 :
  smallest_among ((-2)^3) (-(3^2)) (-(-1)) = -(3^2) :=
by
  sorry

end smallest_among_neg2_cube_neg3_square_neg_neg1_l234_234817


namespace area_of_triangle_DEF_l234_234324

theorem area_of_triangle_DEF :
  ∃ (DEF : Type) (area_u1 area_u2 area_u3 area_triangle : ℝ),
  area_u1 = 25 ∧
  area_u2 = 16 ∧
  area_u3 = 64 ∧
  area_triangle = area_u1 + area_u2 + area_u3 ∧
  area_triangle = 289 :=
by
  sorry

end area_of_triangle_DEF_l234_234324


namespace vector_sum_l234_234546

def v1 : ℤ × ℤ := (5, -3)
def v2 : ℤ × ℤ := (-2, 4)
def scalar : ℤ := 3

theorem vector_sum : 
  (v1.1 + scalar * v2.1, v1.2 + scalar * v2.2) = (-1, 9) := 
by 
  sorry

end vector_sum_l234_234546


namespace minimize_cost_l234_234720

noncomputable def total_cost (x : ℝ) : ℝ := (16000000 / x) + 40000 * x

theorem minimize_cost : ∃ (x : ℝ), x > 0 ∧ (∀ y > 0, total_cost x ≤ total_cost y) ∧ x = 20 := 
sorry

end minimize_cost_l234_234720


namespace number_of_dice_l234_234898

theorem number_of_dice (n : ℕ) (h : (1 / 6 : ℝ) ^ (n - 1) = 0.0007716049382716049) : n = 5 :=
sorry

end number_of_dice_l234_234898


namespace shirt_cost_correct_l234_234726

-- Definitions based on the conditions
def initial_amount : ℕ := 109
def pants_cost : ℕ := 13
def remaining_amount : ℕ := 74
def total_spent : ℕ := initial_amount - remaining_amount
def shirts_cost : ℕ := total_spent - pants_cost
def number_of_shirts : ℕ := 2

-- Statement to be proved
theorem shirt_cost_correct : shirts_cost / number_of_shirts = 11 := by
  sorry

end shirt_cost_correct_l234_234726


namespace number_of_two_bedroom_units_l234_234637

-- Definitions based on the conditions
def is_solution (x y : ℕ) : Prop :=
  (x + y = 12) ∧ (360 * x + 450 * y = 4950)

theorem number_of_two_bedroom_units : ∃ y : ℕ, is_solution (12 - y) y ∧ y = 7 :=
by
  sorry

end number_of_two_bedroom_units_l234_234637


namespace total_surface_area_correct_l234_234787

def six_cubes_surface_area : ℕ :=
  let cube_edge := 1
  let cubes := 6
  let initial_surface_area := 6 * cubes -- six faces per cube, total initial surface area
  let hidden_faces := 10 -- determined by counting connections
  initial_surface_area - hidden_faces

theorem total_surface_area_correct : six_cubes_surface_area = 26 := by
  sorry

end total_surface_area_correct_l234_234787


namespace other_root_l234_234692

/-- Given the quadratic equation x^2 - 3x + k = 0 has one root as 1, 
    prove that the other root is 2. -/
theorem other_root (k : ℝ) (h : 1^2 - 3 * 1 + k = 0) : 
  2^2 - 3 * 2 + k = 0 := 
by 
  sorry

end other_root_l234_234692


namespace cannot_form_right_triangle_l234_234381

theorem cannot_form_right_triangle (a b c : ℕ) (h_a : a = 3) (h_b : b = 5) (h_c : c = 7) : 
  a^2 + b^2 ≠ c^2 :=
by 
  rw [h_a, h_b, h_c]
  sorry

end cannot_form_right_triangle_l234_234381


namespace triangle_internal_region_l234_234037

-- Define the three lines forming the triangle
def line1 (x y : ℝ) : Prop := x + 2 * y = 2
def line2 (x y : ℝ) : Prop := 2 * x + y = 2
def line3 (x y : ℝ) : Prop := x - y = 3

-- Define the inequalities representing the internal region of the triangle
def region (x y : ℝ) : Prop :=
  x - y < 3 ∧ x + 2 * y < 2 ∧ 2 * x + y > 2

-- State that the internal region excluding the boundary is given by the inequalities
theorem triangle_internal_region (x y : ℝ) :
  (∃ (x y : ℝ), line1 x y ∧ line2 x y ∧ line3 x y) → region x y :=
  sorry

end triangle_internal_region_l234_234037


namespace part_A_part_C_part_D_l234_234197

noncomputable def f : ℝ → ℝ := sorry -- define f with given properties

-- Given conditions
axiom mono_incr_on_neg1_0 : ∀ x y : ℝ, -1 ≤ x → x ≤ 0 → -1 ≤ y → y ≤ 0 → x < y → f x < f y
axiom symmetry_about_1 : ∀ x : ℝ, f (1 + x) = f (1 - x)
axiom symmetry_about_2_0 : ∀ x : ℝ, f (2 + x) = -f (2 - x)

-- Prove the statements
theorem part_A : f 0 = f (-2) := sorry
theorem part_C : ∀ x y : ℝ, 2 < x → x < 3 → 2 < y → y < 3 → x < y → f x > f y := sorry
theorem part_D : f 2021 > f 2022 ∧ f 2022 > f 2023 := sorry

end part_A_part_C_part_D_l234_234197


namespace sufficient_m_value_l234_234810

theorem sufficient_m_value (m : ℕ) : 
  ((8 = m ∨ 9 = m) → 
  (m^2 + m^4 + m^6 + m^8 ≥ 6^3 + 6^5 + 6^7 + 6^9)) := 
by 
  sorry

end sufficient_m_value_l234_234810


namespace option_b_has_two_distinct_real_roots_l234_234475

def has_two_distinct_real_roots (a b c : ℝ) : Prop :=
  let Δ := b^2 - 4 * a * c
  Δ > 0

theorem option_b_has_two_distinct_real_roots :
  has_two_distinct_real_roots 1 (-2) (-3) :=
by
  sorry

end option_b_has_two_distinct_real_roots_l234_234475


namespace dormitory_problem_l234_234841

theorem dormitory_problem (x : ℕ) :
  9 < x ∧ x < 12
  → (x = 10 ∧ 4 * x + 18 = 58)
  ∨ (x = 11 ∧ 4 * x + 18 = 62) :=
by
  intros h
  sorry

end dormitory_problem_l234_234841


namespace fraction_addition_l234_234379

variable (a : ℝ) -- Introduce a real number 'a'

theorem fraction_addition :
  (3 / a) + (2 / a) = 5 / a :=
sorry -- Skipping the proof steps as instructed

end fraction_addition_l234_234379


namespace find_x_l234_234229

theorem find_x (x : ℝ) : (x / 4 * 5 + 10 - 12 = 48) → (x = 40) :=
by
  sorry

end find_x_l234_234229


namespace calculate_distance_l234_234627

theorem calculate_distance (t : ℕ) (h_t : t = 4) : 5 * t^2 + 2 * t = 88 :=
by
  rw [h_t]
  norm_num

end calculate_distance_l234_234627


namespace compute_expression_l234_234214

theorem compute_expression : 7^2 - 5 * 6 + 6^2 = 55 := by
  sorry

end compute_expression_l234_234214


namespace unique_solution_m_l234_234334

theorem unique_solution_m (m : ℝ) : (∃ x : ℝ, 3 * x^2 - 6 * x + m = 0 ∧ (∀ y₁ y₂ : ℝ, 3 * y₁^2 - 6 * y₂ + m = 0 → y₁ = y₂)) → m = 3 :=
by
  sorry

end unique_solution_m_l234_234334


namespace expression_value_l234_234298

theorem expression_value : (5^2 - 5) * (6^2 - 6) - (7^2 - 7) = 558 := by
  sorry

end expression_value_l234_234298


namespace goose_eggs_calculation_l234_234831

theorem goose_eggs_calculation (E : ℝ) (hatch_fraction : ℝ) (survived_first_month_fraction : ℝ) 
(survived_first_year_fraction : ℝ) (survived_first_year : ℝ) (no_more_than_one_per_egg : Prop) 
(h_hatch : hatch_fraction = 1/3) 
(h_month_survival : survived_first_month_fraction = 3/4)
(h_year_survival : survived_first_year_fraction = 2/5)
(h_survived120 : survived_first_year = 120)
(h_no_more_than_one : no_more_than_one_per_egg) :
  E = 1200 :=
by
  -- Convert the information from conditions to formulate the equation
  sorry


end goose_eggs_calculation_l234_234831


namespace total_selling_price_16800_l234_234848

noncomputable def total_selling_price (CP_per_toy : ℕ) : ℕ :=
  let CP_18 := 18 * CP_per_toy
  let Gain := 3 * CP_per_toy
  CP_18 + Gain

theorem total_selling_price_16800 :
  total_selling_price 800 = 16800 :=
by
  sorry

end total_selling_price_16800_l234_234848


namespace triangle_inequality_l234_234388

theorem triangle_inequality (a b c : ℝ) (h : a + b + c = 1) : a^2 + b^2 + c^2 + 4 * a * b * c < 1 / 2 :=
sorry

end triangle_inequality_l234_234388


namespace colored_sectors_overlap_l234_234198

/--
Given two disks each divided into 1985 equal sectors, with 200 sectors on each disk colored arbitrarily,
and one disk is rotated by angles that are multiples of 360 degrees / 1985, 
prove that there are at least 80 positions where no more than 20 colored sectors coincide.
-/
theorem colored_sectors_overlap :
  ∀ (disks : ℕ → ℕ) (sectors_colored : ℕ),
  disks 1 = 1985 → disks 2 = 1985 →
  sectors_colored = 200 →
  ∃ (p : ℕ), p ≥ 80 ∧ (∀ (i : ℕ), (i < p → sectors_colored ≤ 20)) := 
sorry

end colored_sectors_overlap_l234_234198


namespace alien_home_planet_people_count_l234_234283

noncomputable def alien_earth_abduction (total_abducted returned_percentage taken_to_other_planet : ℕ) : ℕ :=
  let returned := total_abducted * returned_percentage / 100
  let remaining := total_abducted - returned
  remaining - taken_to_other_planet

theorem alien_home_planet_people_count :
  alien_earth_abduction 200 80 10 = 30 :=
by
  sorry

end alien_home_planet_people_count_l234_234283


namespace time_after_9876_seconds_l234_234045

noncomputable def currentTime : Nat := 2 * 3600 + 45 * 60 + 0
noncomputable def futureDuration : Nat := 9876
noncomputable def resultingTime : Nat := 5 * 3600 + 29 * 60 + 36

theorem time_after_9876_seconds : 
  (currentTime + futureDuration) % (24 * 3600) = resultingTime := 
by 
  sorry

end time_after_9876_seconds_l234_234045


namespace identify_quadratic_equation_l234_234932

/-- Proving which equation is a quadratic equation from given options -/
def is_quadratic_equation (eq : String) : Prop :=
  eq = "sqrt(x^2)=2" ∨ eq = "x^2 - x - 2" ∨ eq = "1/x^2 - 2=0" ∨ eq = "x^2=0"

theorem identify_quadratic_equation :
  ∀ (eq : String), is_quadratic_equation eq → eq = "x^2=0" :=
by
  intro eq h
  -- add proof steps here
  sorry

end identify_quadratic_equation_l234_234932


namespace max_take_home_pay_income_l234_234219

theorem max_take_home_pay_income (x : ℤ) : 
  (1000 * 2 * 50) - 20 * 50^2 = 100000 := 
by 
  sorry

end max_take_home_pay_income_l234_234219


namespace population_increase_l234_234755

theorem population_increase (k l m : ℝ) : 
  (1 + k/100) * (1 + l/100) * (1 + m/100) = 
  1 + (k + l + m)/100 + (k*l + k*m + l*m)/10000 + k*l*m/1000000 :=
by sorry

end population_increase_l234_234755


namespace no_n_in_range_l234_234432

def g (n : ℕ) : ℕ := 7 + 4 * n + 6 * n ^ 2 + 3 * n ^ 3 + 4 * n ^ 4 + 3 * n ^ 5

theorem no_n_in_range
  : ¬ ∃ n : ℕ, 2 ≤ n ∧ n ≤ 100 ∧ g n % 11 = 0 := sorry

end no_n_in_range_l234_234432


namespace find_number_l234_234881

def exceeding_condition (x : ℝ) : Prop :=
  x = 0.16 * x + 84

theorem find_number : ∃ x : ℝ, exceeding_condition x ∧ x = 100 :=
by
  -- Proof goes here, currently omitted.
  sorry

end find_number_l234_234881


namespace existence_of_intersection_l234_234293

def setA (m : ℝ) : Set (ℝ × ℝ) := { p | ∃ (x y : ℝ), p = (x, y) ∧ (x^2 + m * x - y + 2 = 0) }
def setB : Set (ℝ × ℝ) := { p | ∃ (x y : ℝ), p = (x, y) ∧ (x - y + 1 = 0) ∧ (0 ≤ x ∧ x ≤ 2) }

theorem existence_of_intersection (m : ℝ) : (∃ (p : ℝ × ℝ), p ∈ (setA m ∩ setB)) ↔ m ≤ -1 := 
sorry

end existence_of_intersection_l234_234293


namespace range_of_a_l234_234336

/-- Proposition p: ∀ x ∈ [1,2], x² - a ≥ 0 -/
def prop_p (a : ℝ) : Prop := 
  ∀ x : ℝ, (1 ≤ x ∧ x ≤ 2) → x^2 - a ≥ 0

/-- Proposition q: ∃ x₀ ∈ ℝ, x + 2ax₀ + 2 - a = 0 -/
def prop_q (a : ℝ) : Prop := 
  ∃ x₀ : ℝ, ∃ x : ℝ, x + 2 * a * x₀ + 2 - a = 0

theorem range_of_a (a : ℝ) (h : prop_p a ∧ prop_q a) : a ≤ -2 ∨ a = 1 :=
sorry

end range_of_a_l234_234336


namespace equal_area_of_second_square_l234_234158

/-- 
In an isosceles right triangle with legs of length 25√2 cm, if a square is inscribed such that two 
of its vertices lie on one leg and one vertex on each of the hypotenuse and the other leg, 
and the area of the square is 625 cm², prove that the area of another inscribed square 
(with one vertex each on the hypotenuse and one leg, and two vertices on the other leg) is also 625 cm².
-/
theorem equal_area_of_second_square 
  (a b : ℝ) (h1 : a = 25 * Real.sqrt 2)  
  (h2 : b = 625) :
  ∃ c : ℝ, c = 625 :=
by
  sorry

end equal_area_of_second_square_l234_234158


namespace total_cost_of_long_distance_bill_l234_234548

theorem total_cost_of_long_distance_bill
  (monthly_fee : ℝ := 5)
  (cost_per_minute : ℝ := 0.25)
  (minutes_billed : ℝ := 28.08) :
  monthly_fee + cost_per_minute * minutes_billed = 12.02 := by
  sorry

end total_cost_of_long_distance_bill_l234_234548


namespace norris_money_left_l234_234634

def sept_savings : ℕ := 29
def oct_savings : ℕ := 25
def nov_savings : ℕ := 31
def hugo_spent  : ℕ := 75
def total_savings : ℕ := sept_savings + oct_savings + nov_savings
def norris_left : ℕ := total_savings - hugo_spent

theorem norris_money_left : norris_left = 10 := by
  unfold norris_left total_savings sept_savings oct_savings nov_savings hugo_spent
  sorry

end norris_money_left_l234_234634


namespace percentage_of_boys_l234_234693

theorem percentage_of_boys (total_students boys girls : ℕ) (h_ratio : boys * 4 = girls * 3) (h_total : boys + girls = total_students) (h_total_students : total_students = 42) : (boys : ℚ) * 100 / total_students = 42.857 :=
by
  sorry

end percentage_of_boys_l234_234693


namespace largest_k_inequality_l234_234707

theorem largest_k_inequality
  (a b c : ℝ)
  (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c)
  (h_pos : (a + b) * (b + c) * (c + a) > 0) :
  a^2 + b^2 + c^2 - a * b - b * c - c * a ≥ 
  (1 / 2) * abs ((a^3 - b^3) / (a + b) + (b^3 - c^3) / (b + c) + (c^3 - a^3) / (c + a)) :=
by
  sorry

end largest_k_inequality_l234_234707


namespace area_of_rectangle_l234_234667

theorem area_of_rectangle (M N P Q R S X Y : Type) 
  (PQ : ℝ) (PX XY YQ : ℝ) (R_perpendicular_to_PQ S_perpendicular_to_PQ : Prop) 
  (R_through_M S_through_Q : Prop) 
  (segment_lengths : PQ = PX + XY + YQ) : PQ = 5 ∧ PX = 1 ∧ XY = 2 ∧ YQ = 2 
  → 2 * (1/2 * PQ * 2) = 10 :=
  sorry

end area_of_rectangle_l234_234667


namespace num_customers_did_not_tip_l234_234883

def total_customers : Nat := 9
def total_earnings : Nat := 32
def tip_per_customer : Nat := 8
def customers_who_tipped := total_earnings / tip_per_customer
def customers_who_did_not_tip := total_customers - customers_who_tipped

theorem num_customers_did_not_tip : customers_who_did_not_tip = 5 := 
by
  -- We use the definitions provided.
  have eq1 : customers_who_tipped = 4 := by
    sorry
  have eq2 : customers_who_did_not_tip = total_customers - customers_who_tipped := by
    sorry
  have eq3 : customers_who_did_not_tip = 9 - 4 := by
    sorry
  exact eq3

end num_customers_did_not_tip_l234_234883


namespace computation_of_difference_of_squares_l234_234056

theorem computation_of_difference_of_squares : (65^2 - 35^2) = 3000 := sorry

end computation_of_difference_of_squares_l234_234056


namespace tan_70_sin_80_eq_neg1_l234_234033

theorem tan_70_sin_80_eq_neg1 :
  (Real.tan 70 * Real.sin 80 * (Real.sqrt 3 * Real.tan 20 - 1) = -1) :=
sorry

end tan_70_sin_80_eq_neg1_l234_234033


namespace simplify_polynomial_l234_234680

theorem simplify_polynomial (y : ℝ) :
    (4 * y^10 + 6 * y^9 + 3 * y^8) + (2 * y^12 + 5 * y^10 + y^9 + y^7 + 4 * y^4 + 7 * y + 9) =
    2 * y^12 + 9 * y^10 + 7 * y^9 + 3 * y^8 + y^7 + 4 * y^4 + 7 * y + 9 := by
  sorry

end simplify_polynomial_l234_234680


namespace trigonometric_expression_value_l234_234360

-- Define the line equation and the conditions about the slope angle
def line_eq (x y : ℝ) : Prop := 6 * x - 2 * y - 5 = 0

-- The slope angle alpha
variable (α : ℝ)

-- Given conditions
axiom slope_tan : Real.tan α = 3

-- The expression we need to prove equals -2
theorem trigonometric_expression_value :
  (Real.sin (Real.pi - α) + Real.cos (-α)) / (Real.sin (-α) - Real.cos (Real.pi + α)) = -2 :=
by
  sorry

end trigonometric_expression_value_l234_234360


namespace intersection_of_S_and_T_l234_234314

-- Define the sets S and T in Lean
def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

-- Prove that S ∩ T = T
theorem intersection_of_S_and_T : S ∩ T = T := 
by {
  sorry -- placeholder for proof
}

end intersection_of_S_and_T_l234_234314


namespace imaginary_part_of_z_l234_234606

open Complex

-- Definition of the complex number as per the problem statement
def z : ℂ := (2 - 3 * Complex.I) * Complex.I

-- The theorem stating that the imaginary part of the given complex number is 2
theorem imaginary_part_of_z : z.im = 2 :=
by
  sorry

end imaginary_part_of_z_l234_234606


namespace cuberoot_eq_l234_234329

open Real

theorem cuberoot_eq (x : ℝ) (h: (5:ℝ) * x + 4 = (5:ℝ) ^ 3 / (2:ℝ) ^ 3) : x = 93 / 40 := by
  sorry

end cuberoot_eq_l234_234329


namespace number_sequence_53rd_l234_234308

theorem number_sequence_53rd (n : ℕ) (h₁ : n = 53) : n = 53 :=
by {
  sorry
}

end number_sequence_53rd_l234_234308


namespace tiffany_math_homework_pages_l234_234654

def math_problems (m : ℕ) : ℕ := 3 * m
def reading_problems : ℕ := 4 * 3
def total_problems (m : ℕ) : ℕ := math_problems m + reading_problems

theorem tiffany_math_homework_pages (m : ℕ) (h : total_problems m = 30) : m = 6 :=
by
  sorry

end tiffany_math_homework_pages_l234_234654


namespace three_digit_numbers_mod_1000_l234_234366

theorem three_digit_numbers_mod_1000 (n : ℕ) (h_lower : 100 ≤ n) (h_upper : n ≤ 999) : 
  (n^2 ≡ n [MOD 1000]) ↔ (n = 376 ∨ n = 625) :=
by sorry

end three_digit_numbers_mod_1000_l234_234366


namespace julia_total_watches_l234_234840

namespace JuliaWatches

-- Given conditions
def silver_watches : ℕ := 20
def bronze_watches : ℕ := 3 * silver_watches
def platinum_watches : ℕ := 2 * bronze_watches
def gold_watches : ℕ := (20 * (silver_watches + platinum_watches)) / 100  -- 20 is 20% and division by 100 to get the percentage

-- Proving the total watches Julia owns after the purchase
theorem julia_total_watches : silver_watches + bronze_watches + platinum_watches + gold_watches = 228 := by
  sorry

end JuliaWatches

end julia_total_watches_l234_234840


namespace sqrt_pi_decimal_expansion_l234_234393

-- Statement of the problem: Compute the first 23 digits of the decimal expansion of sqrt(pi)
theorem sqrt_pi_decimal_expansion : 
  ( ∀ n, n ≤ 22 → 
    (digits : List ℕ) = [d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15, d16, d17, d18, d19, d20, d21, d22, d23] →
      (d1 = 1 ∧ d2 = 7 ∧ d3 = 7 ∧ d4 = 2 ∧ d5 = 4 ∧ d6 = 5 ∧ d7 = 3 ∧ d8 = 8 ∧ d9 = 5 ∧ d10 = 0 ∧ d11 = 9 ∧ d12 = 0 ∧ d13 = 5 ∧ d14 = 5 ∧ d15 = 1 ∧ d16 = 6 ∧ d17 = 0 ∧ d18 = 2 ∧ d19 = 7 ∧ d20 = 2 ∧ d21 = 9 ∧ d22 = 8 ∧ d23 = 1)) → 
  True :=
by
  sorry
  -- Actual proof to be filled, this is just the statement showing that we expected the digits 
  -- of the decimal expansion of sqrt(pi) match the specified values up to the 23rd place.

end sqrt_pi_decimal_expansion_l234_234393


namespace proof_problem_l234_234832

def g : ℕ → ℕ := sorry
def g_inv : ℕ → ℕ := sorry

axiom g_inv_is_inverse : ∀ y, g (g_inv y) = y ∧ g_inv (g y) = y
axiom g_4_eq_6 : g 4 = 6
axiom g_6_eq_2 : g 6 = 2
axiom g_3_eq_7 : g 3 = 7

theorem proof_problem :
  g_inv (g_inv 7 + g_inv 6) = 3 :=
by
  sorry

end proof_problem_l234_234832


namespace find_real_numbers_l234_234604

theorem find_real_numbers (x y : ℝ) (h₁ : x + y = 3) (h₂ : x^5 + y^5 = 33) :
  (x = 1 ∧ y = 2) ∨ (x = 2 ∧ y = 1) := by
  sorry

end find_real_numbers_l234_234604


namespace tenfold_largest_two_digit_number_l234_234803

def largest_two_digit_number : ℕ := 99

theorem tenfold_largest_two_digit_number :
  10 * largest_two_digit_number = 990 :=
by
  sorry

end tenfold_largest_two_digit_number_l234_234803


namespace crow_speed_l234_234798

/-- Definitions from conditions -/
def distance_between_nest_and_ditch : ℝ := 250 -- in meters
def total_trips : ℕ := 15
def total_hours : ℝ := 1.5 -- hours

/-- The statement to be proved -/
theorem crow_speed :
  let distance_per_trip := 2 * distance_between_nest_and_ditch
  let total_distance := (total_trips : ℝ) * distance_per_trip / 1000 -- convert to kilometers
  let speed := total_distance / total_hours
  speed = 5 := by
  let distance_per_trip := 2 * distance_between_nest_and_ditch
  let total_distance := (total_trips : ℝ) * distance_per_trip / 1000
  let speed := total_distance / total_hours
  sorry

end crow_speed_l234_234798


namespace sequence_next_number_l234_234751

def next_number_in_sequence (seq : List ℕ) : ℕ :=
  if seq = [1, 1, 2, 1, 2, 3, 1, 2, 3, 4, 1, 2] then 3 else sorry

theorem sequence_next_number :
  next_number_in_sequence [1, 1, 2, 1, 2, 3, 1, 2, 3, 4, 1, 2] = 3 :=
by
  -- This proof is to ensure the pattern conditions are met
  sorry

end sequence_next_number_l234_234751


namespace depth_of_channel_l234_234071

noncomputable def trapezium_area (a b h : ℝ) : ℝ :=
1/2 * (a + b) * h

theorem depth_of_channel :
  ∃ h : ℝ, trapezium_area 12 8 h = 700 ∧ h = 70 :=
by
  use 70
  unfold trapezium_area
  sorry

end depth_of_channel_l234_234071


namespace general_formula_arithmetic_sequence_sum_of_sequence_b_l234_234731

-- Definitions of arithmetic sequence {a_n} and geometric sequence conditions
def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
 ∀ n : ℕ, a (n + 1) = a n + d

def sum_first_n_terms (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
 ∀ n : ℕ, S n = n * (a 1 + a n) / 2

def geometric_sequence (a : ℕ → ℤ) : Prop :=
  a 3 ^ 2 = a 1 * a 7

def arithmetic_sum_S3 (S : ℕ → ℤ) : Prop :=
  S 3 = 9

def general_formula (a : ℕ → ℤ) : Prop :=
 ∀ n : ℕ, a n = n + 1

def sum_first_n_terms_b (b : ℕ → ℤ) (T : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, T n = (n-1) * 2^(n+1) + 2

-- The Lean theorem statements
theorem general_formula_arithmetic_sequence
  (a : ℕ → ℤ) (d : ℤ) (S : ℕ → ℤ)
  (h1 : arithmetic_sequence a d)
  (h2 : sum_first_n_terms a S)
  (h3 : geometric_sequence a)
  (h4 : arithmetic_sum_S3 S) :
  general_formula a :=
  sorry

theorem sum_of_sequence_b
  (a b : ℕ → ℤ) (T : ℕ → ℤ)
  (h1 : general_formula a)
  (h2 : ∀ n : ℕ, b n = (a n - 1) * 2^n)
  (h3 : sum_first_n_terms_b b T) :
  ∀ n : ℕ, T n = (n-1) * 2^(n+1) + 2 :=
  sorry

end general_formula_arithmetic_sequence_sum_of_sequence_b_l234_234731


namespace vertical_asymptotes_l234_234082

noncomputable def f (x : ℝ) := (x^3 + 3*x^2 + 2*x + 12) / (x^2 - 5*x + 6)

theorem vertical_asymptotes (x : ℝ) : 
  (x^2 - 5*x + 6 = 0) ∧ (x^3 + 3*x^2 + 2*x + 12 ≠ 0) ↔ (x = 2 ∨ x = 3) :=
by
  sorry

end vertical_asymptotes_l234_234082


namespace total_cost_is_correct_l234_234976

noncomputable def total_cost : ℝ :=
  let palm_fern_cost := 15.00
  let creeping_jenny_cost := 4.00
  let geranium_cost := 3.50
  let elephant_ear_cost := 7.00
  let purple_fountain_grass_cost := 6.00
  let pots := 6
  let sales_tax := 0.07
  let cost_one_pot := palm_fern_cost 
                   + 4 * creeping_jenny_cost 
                   + 4 * geranium_cost 
                   + 2 * elephant_ear_cost 
                   + 3 * purple_fountain_grass_cost
  let total_pots_cost := pots * cost_one_pot
  let tax := total_pots_cost * sales_tax
  total_pots_cost + tax

theorem total_cost_is_correct : total_cost = 494.34 :=
by
  -- This is where the proof would go, but we are adding sorry to skip the proof
  sorry

end total_cost_is_correct_l234_234976


namespace particle_position_at_2004_seconds_l234_234093

structure ParticleState where
  position : ℕ × ℕ

def initialState : ParticleState :=
  { position := (0, 0) }

def moveParticle (state : ParticleState) (time : ℕ) : ParticleState :=
  if time = 0 then initialState
  else if (time - 1) % 4 < 2 then
    { state with position := (state.position.fst + 1, state.position.snd) }
  else
    { state with position := (state.position.fst, state.position.snd + 1) }

def particlePositionAfterTime (time : ℕ) : ParticleState :=
  (List.range time).foldl moveParticle initialState

/-- The position of the particle after 2004 seconds is (20, 44) -/
theorem particle_position_at_2004_seconds :
  (particlePositionAfterTime 2004).position = (20, 44) :=
  sorry

end particle_position_at_2004_seconds_l234_234093


namespace sin_theta_value_l234_234910

open Real

theorem sin_theta_value
  (θ : ℝ)
  (h1 : θ ∈ Set.Ioo (3 * π / 4) (5 * π / 4))
  (h2 : sin (θ - π / 4) = 5 / 13) :
  sin θ = - (7 * sqrt 2) / 26 :=
  sorry

end sin_theta_value_l234_234910


namespace gcd_221_195_l234_234415

-- Define the two numbers
def a := 221
def b := 195

-- Statement of the problem: the gcd of a and b is 13
theorem gcd_221_195 : Nat.gcd a b = 13 := 
by
  sorry

end gcd_221_195_l234_234415


namespace anne_speed_ratio_l234_234387

variable (B A A' : ℝ)

theorem anne_speed_ratio (h1 : A = 1 / 12)
                        (h2 : B + A = 1 / 4)
                        (h3 : B + A' = 1 / 3) : 
                        A' / A = 2 := 
by
  -- Proof is omitted
  sorry

end anne_speed_ratio_l234_234387


namespace equation_has_two_distinct_roots_l234_234668

def quadratic (a x : ℝ) : ℝ :=
  a^2 * (x - 2) + a * (39 - 20 * x) + 20 

theorem equation_has_two_distinct_roots (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ quadratic a x1 = 0 ∧ quadratic a x2 = 0) ↔ a = 20 := 
by
  sorry

end equation_has_two_distinct_roots_l234_234668


namespace staircase_steps_l234_234211

theorem staircase_steps (x : ℕ) :
  x % 2 = 1 ∧
  x % 3 = 2 ∧
  x % 4 = 3 ∧
  x % 5 = 4 ∧
  x % 6 = 5 ∧
  x % 7 = 0 → 
  x ≡ 119 [MOD 420] :=
by
  sorry

end staircase_steps_l234_234211


namespace students_per_group_l234_234299

def total_students : ℕ := 30
def number_of_groups : ℕ := 6

theorem students_per_group :
  total_students / number_of_groups = 5 :=
by
  sorry

end students_per_group_l234_234299


namespace tan_sum_angle_l234_234073

theorem tan_sum_angle (α : ℝ) (h : Real.tan α = 2) : Real.tan (π / 4 + α) = -3 := 
by sorry

end tan_sum_angle_l234_234073


namespace linear_function_not_in_second_quadrant_l234_234503

-- Define the linear function y = x - 1.
def linear_function (x : ℝ) : ℝ := x - 1

-- Define the condition for a point to be in the second quadrant.
def in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

-- State that for any point (x, y) in the second quadrant, it does not satisfy y = x - 1.
theorem linear_function_not_in_second_quadrant {x y : ℝ} (h : in_second_quadrant x y) : linear_function x ≠ y :=
sorry

end linear_function_not_in_second_quadrant_l234_234503


namespace tickets_sold_total_l234_234853

-- Define the conditions
variables (A : ℕ) (S : ℕ) (total_amount : ℝ := 222.50) (adult_ticket_price : ℝ := 4) (student_ticket_price : ℝ := 2.50)
variables (student_tickets_sold : ℕ := 9)

-- Define the total money equation and the question
theorem tickets_sold_total :
  4 * (A : ℝ) + 2.5 * (9 : ℝ) = 222.50 → A + 9 = 59 :=
by sorry

end tickets_sold_total_l234_234853


namespace M_gt_N_l234_234429

variable (x : ℝ)

def M := x^2 + 4 * x - 2

def N := 6 * x - 5

theorem M_gt_N : M x > N x := sorry

end M_gt_N_l234_234429


namespace fixed_point_of_parabola_l234_234457

theorem fixed_point_of_parabola :
  ∀ (m : ℝ), ∃ (a b : ℝ), (∀ (x : ℝ), (a = -3 ∧ b = 81) → (y = 9*x^2 + m*x + 3*m) → (y = 81)) :=
by
  sorry

end fixed_point_of_parabola_l234_234457


namespace two_cos_30_eq_sqrt_3_l234_234355

open Real

-- Given condition: cos 30 degrees is sqrt(3)/2
def cos_30_eq : cos (π / 6) = sqrt 3 / 2 := 
sorry

-- Goal: to prove that 2 * cos 30 degrees = sqrt(3)
theorem two_cos_30_eq_sqrt_3 : 2 * cos (π / 6) = sqrt 3 :=
by
  rw [cos_30_eq]
  sorry

end two_cos_30_eq_sqrt_3_l234_234355


namespace find_c_for_circle_radius_five_l234_234116

theorem find_c_for_circle_radius_five
  (c : ℝ)
  (h : ∀ x y : ℝ, x^2 + 8 * x + y^2 + 2 * y + c = 0) :
  c = -8 :=
sorry

end find_c_for_circle_radius_five_l234_234116


namespace bricks_in_wall_l234_234310

-- Definitions for individual working times and breaks
def Bea_build_time := 8  -- hours
def Bea_break_time := 10 / 60  -- hours per hour
def Ben_build_time := 12  -- hours
def Ben_break_time := 15 / 60  -- hours per hour

-- Total effective rates
def Bea_effective_rate (h : ℕ) := h / (Bea_build_time * (1 - Bea_break_time))
def Ben_effective_rate (h : ℕ) := h / (Ben_build_time * (1 - Ben_break_time))

-- Decreased rate due to talking
def total_effective_rate (h : ℕ) := Bea_effective_rate h + Ben_effective_rate h - 12

-- Define the Lean proof statement
theorem bricks_in_wall (h : ℕ) :
  (6 * total_effective_rate h = h) → h = 127 :=
by sorry

end bricks_in_wall_l234_234310


namespace gcd_factorial_7_8_l234_234321

theorem gcd_factorial_7_8 : Nat.gcd (Nat.factorial 7) (Nat.factorial 8) = 5040 := 
by
  sorry

end gcd_factorial_7_8_l234_234321


namespace least_number_to_subtract_997_l234_234918

theorem least_number_to_subtract_997 (x : ℕ) (h : x = 997) 
  : ∃ y : ℕ, ∀ m (h₁ : m = (997 - y)), 
    m % 5 = 3 ∧ m % 9 = 3 ∧ m % 11 = 3 ∧ y = 4 :=
by
  -- Proof omitted
  sorry

end least_number_to_subtract_997_l234_234918


namespace count_five_digit_numbers_ending_in_6_divisible_by_3_l234_234631

theorem count_five_digit_numbers_ending_in_6_divisible_by_3 : 
  (∃ (n : ℕ), n = 3000 ∧
  ∀ (x : ℕ), (x ≥ 10000 ∧ x ≤ 99999) ∧ (x % 10 = 6) ∧ (x % 3 = 0) ↔ 
  (∃ (k : ℕ), x = 10026 + k * 30 ∧ k < 3000)) :=
by
  -- Proof is omitted
  sorry

end count_five_digit_numbers_ending_in_6_divisible_by_3_l234_234631


namespace equilateral_triangle_perimeter_l234_234663

-- Define the condition of an equilateral triangle where each side is 7 cm
def side_length : ℕ := 7

def is_equilateral_triangle (a b c : ℕ) : Prop :=
  a = b ∧ b = c

-- Define the perimeter function for a triangle
def perimeter (a b c : ℕ) : ℕ :=
  a + b + c

-- Statement to prove
theorem equilateral_triangle_perimeter : is_equilateral_triangle side_length side_length side_length → perimeter side_length side_length side_length = 21 :=
sorry

end equilateral_triangle_perimeter_l234_234663


namespace smallest_x_for_multiple_of_720_l234_234765

theorem smallest_x_for_multiple_of_720 (x : ℕ) (h1 : 450 = 2^1 * 3^2 * 5^2) (h2 : 720 = 2^4 * 3^2 * 5^1) : x = 8 ↔ (450 * x) % 720 = 0 :=
by
  sorry

end smallest_x_for_multiple_of_720_l234_234765


namespace value_of_x_l234_234977

-- Define variables and conditions
def consecutive (x y z : ℤ) : Prop := x = z + 2 ∧ y = z + 1

-- Main proposition
theorem value_of_x (x y z : ℤ) (h1 : consecutive x y z) (h2 : z = 2) (h3 : 2 * x + 3 * y + 3 * z = 5 * y + 8) : x = 4 :=
by
  sorry

end value_of_x_l234_234977


namespace mode_of_list_is_five_l234_234802

def list := [3, 4, 5, 5, 5, 5, 7, 11, 21]

def occurrence_count (l : List ℕ) (x : ℕ) : ℕ :=
  l.count x

def is_mode (l : List ℕ) (x : ℕ) : Prop :=
  ∀ y : ℕ, occurrence_count l x ≥ occurrence_count l y

theorem mode_of_list_is_five : is_mode list 5 := by
  sorry

end mode_of_list_is_five_l234_234802


namespace c_a_plus_c_b_geq_a_a_plus_b_b_l234_234830

theorem c_a_plus_c_b_geq_a_a_plus_b_b (a b : ℕ) (ha : 0 < a) (hb : 0 < b)
  (c : ℚ) (h : c = (a^(a+1) + b^(b+1)) / (a^a + b^b)) :
  c^a + c^b ≥ a^a + b^b :=
sorry

end c_a_plus_c_b_geq_a_a_plus_b_b_l234_234830


namespace bg_fg_ratio_l234_234748

open Real

-- Given the lengths AB, BD, AF, DF, BE, CF
def AB : ℝ := 15
def BD : ℝ := 18
def AF : ℝ := 15
def DF : ℝ := 12
def BE : ℝ := 24
def CF : ℝ := 17

-- Prove that the ratio BG : FG = 27 : 17
theorem bg_fg_ratio (BG FG : ℝ)
  (h_BG_FG : BG / FG = 27 / 17) :
  BG / FG = 27 / 17 := by
  sorry

end bg_fg_ratio_l234_234748


namespace square_of_third_side_l234_234989

theorem square_of_third_side (a b : ℕ) (h1 : a = 4) (h2 : b = 5) 
    (h_right_triangle : (a^2 + b^2 = c^2) ∨ (b^2 + c^2 = a^2)) : 
    (c = 9) ∨ (c = 41) :=
sorry

end square_of_third_side_l234_234989


namespace solve_container_capacity_l234_234960

noncomputable def container_capacity (C : ℝ) :=
  (0.75 * C - 0.35 * C = 48)

theorem solve_container_capacity : ∃ C : ℝ, container_capacity C ∧ C = 120 :=
by
  use 120
  constructor
  {
    -- Proof that 0.75 * 120 - 0.35 * 120 = 48
    sorry
  }
  -- Proof that C = 120
  sorry

end solve_container_capacity_l234_234960


namespace count_valid_words_l234_234878

def total_words (n : ℕ) : ℕ := 25 ^ n

def words_with_no_A (n : ℕ) : ℕ := 24 ^ n

def words_with_one_A (n : ℕ) : ℕ := n * 24 ^ (n - 1)

def words_with_less_than_two_As : ℕ :=
  (words_with_no_A 2) + (2 * 24) +
  (words_with_no_A 3) + (3 * 24 ^ 2) +
  (words_with_no_A 4) + (4 * 24 ^ 3) +
  (words_with_no_A 5) + (5 * 24 ^ 4)

def valid_words : ℕ :=
  (total_words 1 + total_words 2 + total_words 3 + total_words 4 + total_words 5) -
  words_with_less_than_two_As

theorem count_valid_words : valid_words = sorry :=
by sorry

end count_valid_words_l234_234878


namespace normal_price_of_article_l234_234741

theorem normal_price_of_article (P : ℝ) (sale_price : ℝ) (discount1 discount2 : ℝ) :
  discount1 = 0.10 → discount2 = 0.20 → sale_price = 108 →
  P * (1 - discount1) * (1 - discount2) = sale_price → P = 150 :=
by
  intro hd1 hd2 hsp hdiscount
  -- skipping the proof for now
  sorry

end normal_price_of_article_l234_234741


namespace gcd_polynomial_even_multiple_of_97_l234_234489

theorem gcd_polynomial_even_multiple_of_97 (b : ℤ) (k : ℤ) (h_b : b = 2 * 97 * k) :
  Int.gcd (3 * b^2 + 41 * b + 74) (b + 19) = 1 :=
by
  sorry

end gcd_polynomial_even_multiple_of_97_l234_234489


namespace compute_expression_l234_234068

theorem compute_expression : 1010^2 - 990^2 - 1005^2 + 995^2 = 20000 := by
  sorry

end compute_expression_l234_234068


namespace min_value_x_y_l234_234053

theorem min_value_x_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 9 / y = 1) : x + y ≥ 16 :=
sorry

end min_value_x_y_l234_234053


namespace rational_coefficients_terms_count_l234_234212

theorem rational_coefficients_terms_count : 
  (∃ s : Finset ℕ, ∀ k ∈ s, k % 20 = 0 ∧ k ≤ 725 ∧ s.card = 37) :=
by
  -- Translates to finding the set of all k satisfying the condition and 
  -- ensuring it has a cardinality of 37.
  sorry

end rational_coefficients_terms_count_l234_234212


namespace range_of_m_l234_234106

theorem range_of_m (m : ℝ) : (∀ x : ℝ, m * x^2 - m * x - 2 < 0) → -8 < m ∧ m ≤ 0 :=
sorry

end range_of_m_l234_234106


namespace population_at_seven_years_l234_234569

theorem population_at_seven_years (a x : ℕ) (y: ℝ) (h₀: a = 100) (h₁: x = 7) (h₂: y = a * Real.logb 2 (x + 1)):
  y = 300 :=
by
  -- We include the conditions in the theorem statement
  sorry

end population_at_seven_years_l234_234569


namespace pencils_multiple_of_10_l234_234571

theorem pencils_multiple_of_10 (pens : ℕ) (students : ℕ) (pencils : ℕ) 
  (h_pens : pens = 1230) 
  (h_students : students = 10) 
  (h_max_distribute : ∀ s, s ≤ students → (∃ pens_per_student, pens = pens_per_student * s ∧ ∃ pencils_per_student, pencils = pencils_per_student * s)) :
  ∃ n, pencils = 10 * n :=
by
  sorry

end pencils_multiple_of_10_l234_234571


namespace bus_fare_max_profit_passenger_count_change_l234_234320

noncomputable def demand (p : ℝ) : ℝ := 3000 - 20 * p
noncomputable def train_fare : ℝ := 10
noncomputable def train_capacity : ℝ := 1000
noncomputable def bus_cost (y : ℝ) : ℝ := y + 5

theorem bus_fare_max_profit : 
  ∃ (p_bus : ℝ), 
  p_bus = 50.5 ∧ 
  p_bus * (demand p_bus - train_capacity) - bus_cost (demand p_bus - train_capacity) = 
  p_bus * (demand p_bus - train_capacity) - (demand p_bus - train_capacity + 5) := 
sorry

theorem passenger_count_change :
  (demand train_fare - train_capacity) + train_capacity - demand 75.5 = 500 :=
sorry

end bus_fare_max_profit_passenger_count_change_l234_234320


namespace paving_stone_width_l234_234223

theorem paving_stone_width :
  let courtyard_length := 70
  let courtyard_width := 16.5
  let num_paving_stones := 231
  let paving_stone_length := 2.5
  let courtyard_area := courtyard_length * courtyard_width
  let total_area_covered := courtyard_area
  let paving_stone_width := total_area_covered / (paving_stone_length * num_paving_stones)
  paving_stone_width = 2 :=
by
  sorry

end paving_stone_width_l234_234223


namespace problem_solution_l234_234945

theorem problem_solution
  (a1 a2 a3: ℝ)
  (a_arith_seq : ∃ d, a1 = 1 + d ∧ a2 = a1 + d ∧ a3 = a2 + d ∧ 9 = a3 + d)
  (b1 b2 b3: ℝ)
  (b_geo_seq : ∃ r, r > 0 ∧ b1 = -9 * r ∧ b2 = b1 * r ∧ b3 = b2 * r ∧ -1 = b3 * r) :
  (b2 / (a1 + a3) = -3 / 10) :=
by
  -- Placeholder for the proof, not required in this context
  sorry

end problem_solution_l234_234945


namespace find_M_pos_int_l234_234089

theorem find_M_pos_int (M : ℕ) (hM : 33^2 * 66^2 = 15^2 * M^2) :
    M = 726 :=
by
  -- Sorry, skipping the proof.
  sorry

end find_M_pos_int_l234_234089


namespace expression_evaluation_l234_234684

theorem expression_evaluation (a b c d : ℝ) 
  (h₁ : a + b = 0) 
  (h₂ : c * d = 1) : 
  (a + b)^2 - 3 * (c * d)^4 = -3 := 
by
  -- Proof steps are omitted, as only the statement is required.
  sorry

end expression_evaluation_l234_234684


namespace ann_trip_longer_than_mary_l234_234263

-- Define constants for conditions
def mary_hill_length : ℕ := 630
def mary_speed : ℕ := 90
def ann_hill_length : ℕ := 800
def ann_speed : ℕ := 40

-- Define a theorem to express the question and correct answer
theorem ann_trip_longer_than_mary : 
  (ann_hill_length / ann_speed - mary_hill_length / mary_speed) = 13 :=
by
  -- Now insert sorry to leave the proof unfinished
  sorry

end ann_trip_longer_than_mary_l234_234263


namespace plane_equation_passing_through_point_and_parallel_l234_234674

-- Define the point and the plane parameters
def point : ℝ × ℝ × ℝ := (2, 3, 1)
def normal_vector : ℝ × ℝ × ℝ := (2, -1, 3)
def plane (A B C D : ℝ) (x y z : ℝ) : Prop := A * x + B * y + C * z + D = 0

-- Main theorem statement
theorem plane_equation_passing_through_point_and_parallel :
  ∃ D : ℝ, plane 2 (-1) 3 D 2 3 1 ∧ plane 2 (-1) 3 D 0 0 0 :=
sorry

end plane_equation_passing_through_point_and_parallel_l234_234674


namespace employee_percentage_six_years_or_more_l234_234656

theorem employee_percentage_six_years_or_more
  (x : ℕ)
  (total_employees : ℕ := 36 * x)
  (employees_6_or_more : ℕ := 8 * x) :
  (employees_6_or_more : ℚ) / (total_employees : ℚ) * 100 = 22.22 := 
sorry

end employee_percentage_six_years_or_more_l234_234656


namespace oldest_child_age_l234_234992

theorem oldest_child_age (x : ℕ) (h_avg : (5 + 7 + 10 + x) / 4 = 8) : x = 10 :=
by
  sorry

end oldest_child_age_l234_234992


namespace six_to_2049_not_square_l234_234539

theorem six_to_2049_not_square
  (h1: ∃ x: ℝ, 1^2048 = x^2)
  (h2: ∃ x: ℝ, 2^2050 = x^2)
  (h3: ¬∃ x: ℝ, 6^2049 = x^2)
  (h4: ∃ x: ℝ, 4^2051 = x^2)
  (h5: ∃ x: ℝ, 5^2052 = x^2):
  ¬∃ y: ℝ, y^2 = 6^2049 := 
by sorry

end six_to_2049_not_square_l234_234539


namespace q_0_plus_q_5_l234_234054

-- Define the properties of the polynomial q(x)
variable (q : ℝ → ℝ)
variable (monic_q : ∀ x, ∃ a b c d e f, a = 1 ∧ q x = a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + f)
variable (deg_q : ∀ x, degree q = 5)
variable (q_1 : q 1 = 26)
variable (q_2 : q 2 = 52)
variable (q_3 : q 3 = 78)

-- State the theorem to find q(0) + q(5)
theorem q_0_plus_q_5 : q 0 + q 5 = 58 :=
sorry

end q_0_plus_q_5_l234_234054


namespace maximum_rectangle_area_l234_234827

-- Define the perimeter condition
def perimeter (rectangle : ℝ × ℝ) : ℝ :=
  2 * rectangle.fst + 2 * rectangle.snd

-- Define the area function
def area (rectangle : ℝ × ℝ) : ℝ :=
  rectangle.fst * rectangle.snd

-- Define the question statement in terms of Lean
theorem maximum_rectangle_area (length_width : ℝ × ℝ) (h : perimeter length_width = 32) : 
  area length_width ≤ 64 :=
sorry

end maximum_rectangle_area_l234_234827


namespace Alice_min_speed_l234_234019

theorem Alice_min_speed
  (distance : Real := 120)
  (bob_speed : Real := 40)
  (alice_delay : Real := 0.5)
  (alice_min_speed : Real := distance / (distance / bob_speed - alice_delay)) :
  alice_min_speed = 48 := 
by
  sorry

end Alice_min_speed_l234_234019


namespace find_numbers_l234_234723

theorem find_numbers (n : ℕ) (h1 : n ≥ 2) (a : ℕ) (ha : a ≠ 1) (ha_min : ∀ d, d ∣ n → d ≠ 1 → a ≤ d) (b : ℕ) (hb : b ∣ n) :
  n = a^2 + b^2 ↔ n = 8 ∨ n = 20 :=
by sorry

end find_numbers_l234_234723


namespace rate_of_first_car_l234_234232

theorem rate_of_first_car
  (r : ℕ) (h1 : 3 * r + 30 = 180) : r = 50 :=
sorry

end rate_of_first_car_l234_234232


namespace avg_abc_43_l234_234452

variables (A B C : ℝ)

def avg_ab (A B : ℝ) : Prop := (A + B) / 2 = 40
def avg_bc (B C : ℝ) : Prop := (B + C) / 2 = 43
def weight_b (B : ℝ) : Prop := B = 37

theorem avg_abc_43 (A B C : ℝ) (h1 : avg_ab A B) (h2 : avg_bc B C) (h3 : weight_b B) :
  (A + B + C) / 3 = 43 :=
by
  sorry

end avg_abc_43_l234_234452


namespace a_and_b_together_complete_in_10_days_l234_234965

noncomputable def a_works_twice_as_fast_as_b (a b : ℝ) : Prop :=
  a = 2 * b

noncomputable def b_can_complete_work_in_30_days (b : ℝ) : Prop :=
  b = 1/30

theorem a_and_b_together_complete_in_10_days (a b : ℝ) 
  (h₁ : a_works_twice_as_fast_as_b a b)
  (h₂ : b_can_complete_work_in_30_days b) : 
  (1 / (a + b)) = 10 := 
sorry

end a_and_b_together_complete_in_10_days_l234_234965


namespace customers_left_correct_l234_234200

-- Define the initial conditions
def initial_customers : ℕ := 8
def remaining_customers : ℕ := 5

-- Define the statement regarding customers left
def customers_left : ℕ := initial_customers - remaining_customers

-- The theorem we need to prove
theorem customers_left_correct : customers_left = 3 := by
    -- Skipping the actual proof
    sorry

end customers_left_correct_l234_234200


namespace cost_price_represents_articles_l234_234271

theorem cost_price_represents_articles (C S : ℝ) (N : ℕ)
  (h1 : N * C = 16 * S)
  (h2 : S = C * 1.125) :
  N = 18 :=
by
  sorry

end cost_price_represents_articles_l234_234271


namespace exists_100_distinct_sums_l234_234988

theorem exists_100_distinct_sums : ∃ (a : Fin 100 → ℕ), (∀ i j k l : Fin 100, i ≠ j → k ≠ l → (i, j) ≠ (k, l) → a i + a j ≠ a k + a l) ∧ (∀ i : Fin 100, 1 ≤ a i ∧ a i ≤ 25000) :=
by
  sorry

end exists_100_distinct_sums_l234_234988


namespace remainder_q_x_plus_2_l234_234882

def q (x : ℝ) (D E F : ℝ) : ℝ := D * x ^ 6 + E * x ^ 4 + F * x ^ 2 + 5

theorem remainder_q_x_plus_2 (D E F : ℝ) (h : q 2 D E F = 13) : q (-2) D E F = 13 :=
by
  sorry

end remainder_q_x_plus_2_l234_234882


namespace product_of_y_values_l234_234086

theorem product_of_y_values (y : ℝ) (h : abs (2 * y * 3) + 5 = 47) :
  ∃ y1 y2, (abs (2 * y1 * 3) + 5 = 47) ∧ (abs (2 * y2 * 3) + 5 = 47) ∧ y1 * y2 = -49 :=
by 
  sorry

end product_of_y_values_l234_234086


namespace evaluate_g_at_6_l234_234703

def g (x : ℝ) := 3 * x^4 - 19 * x^3 + 31 * x^2 - 27 * x - 72

theorem evaluate_g_at_6 : g 6 = 666 := by
  sorry

end evaluate_g_at_6_l234_234703


namespace total_votes_is_240_l234_234615

variable {x : ℕ} -- Total number of votes (natural number)
variable {S : ℤ} -- Score (integer)

-- Given conditions
axiom score_condition : S = 120
axiom votes_condition : 3 * x / 4 - x / 4 = S

theorem total_votes_is_240 : x = 240 :=
by
  -- Proof should go here
  sorry

end total_votes_is_240_l234_234615


namespace abs_h_eq_1_div_2_l234_234417

theorem abs_h_eq_1_div_2 {h : ℝ} 
  (h_sum_sq_roots : ∀ (r s : ℝ), (r + s) = 4 * h ∧ (r * s) = -8 → (r ^ 2 + s ^ 2) = 20) : 
  |h| = 1 / 2 :=
sorry

end abs_h_eq_1_div_2_l234_234417


namespace megan_initial_cupcakes_l234_234210

noncomputable def initial_cupcakes (packages : Nat) (cupcakes_per_package : Nat) (cupcakes_eaten : Nat) : Nat :=
  packages * cupcakes_per_package + cupcakes_eaten

theorem megan_initial_cupcakes (packages : Nat) (cupcakes_per_package : Nat) (cupcakes_eaten : Nat) :
  packages = 4 → cupcakes_per_package = 7 → cupcakes_eaten = 43 →
  initial_cupcakes packages cupcakes_per_package cupcakes_eaten = 71 :=
by
  intros
  simp [initial_cupcakes]
  sorry

end megan_initial_cupcakes_l234_234210


namespace find_other_endpoint_l234_234385

theorem find_other_endpoint (x_m y_m x_1 y_1 x_2 y_2 : ℝ)
  (h_midpoint_x : x_m = (x_1 + x_2) / 2)
  (h_midpoint_y : y_m = (y_1 + y_2) / 2)
  (h_given_midpoint : x_m = 3 ∧ y_m = 0)
  (h_given_endpoint1 : x_1 = 7 ∧ y_1 = -4) :
  x_2 = -1 ∧ y_2 = 4 :=
sorry

end find_other_endpoint_l234_234385


namespace cos_alpha_sqrt_l234_234312

theorem cos_alpha_sqrt {α : ℝ} (h1 : Real.sin (π - α) = 1 / 3) (h2 : π / 2 ≤ α ∧ α ≤ π) : 
  Real.cos α = - (2 * Real.sqrt 2) / 3 := 
by
  sorry

end cos_alpha_sqrt_l234_234312


namespace trader_loss_percent_l234_234344

noncomputable def CP1 : ℝ := 325475 / 1.13
noncomputable def CP2 : ℝ := 325475 / 0.87
noncomputable def TCP : ℝ := CP1 + CP2
noncomputable def TSP : ℝ := 325475 * 2
noncomputable def profit_or_loss : ℝ := TSP - TCP
noncomputable def profit_or_loss_percent : ℝ := (profit_or_loss / TCP) * 100

theorem trader_loss_percent : profit_or_loss_percent = -1.684 := by 
  sorry

end trader_loss_percent_l234_234344


namespace intersection_compl_A_compl_B_l234_234305

open Set

variable (x y : ℝ)

def U : Set ℝ := univ
def A : Set ℝ := {x | -1 < x ∧ x < 4}
def B : Set ℝ := {y | ∃ x, y = x + 1 ∧ -1 < x ∧ x < 4}

theorem intersection_compl_A_compl_B (U A B : Set ℝ) (hU : U = univ) (hA : A = {x | -1 < x ∧ x < 4}) (hB : B = {y | ∃ x, y = x + 1 ∧ -1 < x ∧ x < 4}):
  (Aᶜ ∩ Bᶜ) = (Iic (-1) ∪ Ici 5) :=
by
  sorry

end intersection_compl_A_compl_B_l234_234305


namespace solve_for_x_l234_234135

theorem solve_for_x (x : ℚ) (h : (3 * x + 5) / 7 = 13) : x = 86 / 3 :=
sorry

end solve_for_x_l234_234135


namespace minimum_value_of_k_l234_234580

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x + b
noncomputable def g (a c : ℝ) (x : ℝ) : ℝ := a * x + c
noncomputable def h (a b c : ℝ) (x : ℝ) : ℝ := (f a b x)^2 + 8 * (g a c x)
noncomputable def k (a b c : ℝ) (x : ℝ) : ℝ := (g a c x)^2 + 8 * (f a b x)

theorem minimum_value_of_k:
  ∃ a b c : ℝ, a ≠ 0 ∧ (∀ x : ℝ, h a b c x ≥ -29) → (∃ x : ℝ, k a b c x = -3) := sorry

end minimum_value_of_k_l234_234580


namespace solve_equation_l234_234306

def equation (x : ℝ) := (x / (x - 2)) + (2 / (x^2 - 4)) = 1

theorem solve_equation (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -2) : 
  equation x ↔ x = -3 :=
by
  sorry

end solve_equation_l234_234306


namespace triangle_perimeter_correct_l234_234347

noncomputable def triangle_perimeter (a b x : ℕ) : ℕ := a + b + x

theorem triangle_perimeter_correct :
  ∀ (x : ℕ), (2 + 4 + x = 10) → 2 < x → x < 6 → (∀ k : ℕ, k = x → k % 2 = 0) → triangle_perimeter 2 4 x = 10 :=
by
  intros x h1 h2 h3
  rw [triangle_perimeter, h1]
  sorry

end triangle_perimeter_correct_l234_234347


namespace divisible_by_3_l234_234701

theorem divisible_by_3 (n : ℕ) : (n * 2^n + 1) % 3 = 0 ↔ n % 6 = 1 ∨ n % 6 = 2 := 
sorry

end divisible_by_3_l234_234701


namespace tangent_line_eqn_c_range_l234_234342

noncomputable def f (x : ℝ) := 3 * x * Real.log x + 2

theorem tangent_line_eqn :
  let k := 3 
  let x₀ := 1 
  let y₀ := f x₀ 
  y = k * (x - x₀) + y₀ ↔ 3*x - y - 1 = 0 :=
by sorry

theorem c_range (x : ℝ) (hx : 1 < x) (c : ℝ) :
  f x ≤ x^2 - c * x → c ≤ 1 - 3 * Real.log 2 :=
by sorry

end tangent_line_eqn_c_range_l234_234342


namespace percentage_of_water_in_dried_grapes_l234_234479

theorem percentage_of_water_in_dried_grapes 
  (weight_fresh : ℝ) 
  (weight_dried : ℝ) 
  (percentage_water_fresh : ℝ) 
  (solid_weight : ℝ)
  (water_weight_dried : ℝ) 
  (percentage_water_dried : ℝ) 
  (H1 : weight_fresh = 30) 
  (H2 : weight_dried = 15) 
  (H3 : percentage_water_fresh = 0.60) 
  (H4 : solid_weight = weight_fresh * (1 - percentage_water_fresh)) 
  (H5 : water_weight_dried = weight_dried - solid_weight) 
  (H6 : percentage_water_dried = (water_weight_dried / weight_dried) * 100) 
  : percentage_water_dried = 20 := 
  by { sorry }

end percentage_of_water_in_dried_grapes_l234_234479


namespace least_possible_square_area_l234_234852

theorem least_possible_square_area (measured_length : ℝ) (h : measured_length = 7) : 
  ∃ (actual_length : ℝ), 6.5 ≤ actual_length ∧ actual_length < 7.5 ∧ 
  (∀ (side : ℝ), 6.5 ≤ side ∧ side < 7.5 → side * side ≥ actual_length * actual_length) ∧ 
  actual_length * actual_length = 42.25 :=
by
  sorry

end least_possible_square_area_l234_234852


namespace Kelly_current_baking_powder_l234_234789

-- Definitions based on conditions
def yesterday_amount : ℝ := 0.4
def difference : ℝ := 0.1
def current_amount : ℝ := yesterday_amount - difference

-- Statement to prove the question == answer given the conditions
theorem Kelly_current_baking_powder : current_amount = 0.3 := 
by
  sorry

end Kelly_current_baking_powder_l234_234789


namespace regular_polygon_area_l234_234594
open Real

theorem regular_polygon_area (R : ℝ) (n : ℕ) (hR : 0 < R) (hn : 8 ≤ n) (h_area : (1/2) * n * R^2 * sin (360 / n * (π / 180)) = 4 * R^2) :
  n = 10 := 
sorry

end regular_polygon_area_l234_234594


namespace two_square_numbers_difference_133_l234_234120

theorem two_square_numbers_difference_133 : 
  ∃ (x y : ℤ), x^2 - y^2 = 133 ∧ ((x = 67 ∧ y = 66) ∨ (x = 13 ∧ y = 6)) :=
by {
  sorry
}

end two_square_numbers_difference_133_l234_234120


namespace y_coordinate_of_P_l234_234438

theorem y_coordinate_of_P (x y : ℝ) (h1 : |y| = 1/2 * |x|) (h2 : |x| = 12) :
  y = 6 ∨ y = -6 :=
sorry

end y_coordinate_of_P_l234_234438


namespace problem_statement_l234_234029

def f (x : ℝ) : ℝ := 3 * x^2 - 2
def k (x : ℝ) : ℝ := -2 * x^3 + 2

theorem problem_statement : f (k 2) = 586 := by
  sorry

end problem_statement_l234_234029


namespace trajectory_of_complex_point_l234_234139

open Complex Topology

theorem trajectory_of_complex_point (z : ℂ) (hz : ‖z‖ ≤ 1) : 
  {w : ℂ | ‖w‖ ≤ 1} = {w : ℂ | w.re * w.re + w.im * w.im ≤ 1} :=
sorry

end trajectory_of_complex_point_l234_234139


namespace sqrt_sum_eq_five_sqrt_three_l234_234873

theorem sqrt_sum_eq_five_sqrt_three : Real.sqrt 12 + Real.sqrt 27 = 5 * Real.sqrt 3 := by
  sorry

end sqrt_sum_eq_five_sqrt_three_l234_234873


namespace wendy_makeup_time_l234_234224

theorem wendy_makeup_time :
  ∀ (num_products wait_time total_time makeup_time : ℕ),
    num_products = 5 →
    wait_time = 5 →
    total_time = 55 →
    makeup_time = total_time - (num_products - 1) * wait_time →
    makeup_time = 35 :=
by
  intro num_products wait_time total_time makeup_time h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end wendy_makeup_time_l234_234224


namespace largest_divisor_of_three_consecutive_even_integers_is_sixteen_l234_234806

theorem largest_divisor_of_three_consecutive_even_integers_is_sixteen (n : ℕ) :
  ∃ d : ℕ, d = 16 ∧ 16 ∣ ((2 * n) * (2 * n + 2) * (2 * n + 4)) :=
by
  sorry

end largest_divisor_of_three_consecutive_even_integers_is_sixteen_l234_234806


namespace polar_to_cartesian_l234_234782

theorem polar_to_cartesian (θ ρ x y : ℝ) (h1 : ρ = 2 * Real.sin θ) (h2 : x = ρ * Real.cos θ) (h3 : y = ρ * Real.sin θ) :
  x^2 + (y - 1)^2 = 1 :=
sorry

end polar_to_cartesian_l234_234782


namespace percent_defective_units_l234_234318

variable (D : ℝ) -- Let D represent the percent of units produced that are defective

theorem percent_defective_units
  (h1 : 0.05 * D = 0.4) : 
  D = 8 :=
by sorry

end percent_defective_units_l234_234318


namespace inequality_proof_l234_234791

theorem inequality_proof {x y z : ℝ} (n : ℕ) 
  (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) 
  (h4 : x + y + z = 1)
  : (x^4 / (y * (1 - y^n))) + (y^4 / (z * (1 - z^n))) + (z^4 / (x * (1 - x^n))) 
    ≥ (3^n) / (3^(n - 2) - 9) :=
by
  sorry

end inequality_proof_l234_234791


namespace verify_digits_l234_234942

theorem verify_digits :
  ∀ (a b c d e f g h : ℕ), 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧
  d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧
  e ≠ f ∧ e ≠ g ∧ e ≠ h ∧
  f ≠ g ∧ f ≠ h ∧
  g ≠ h ∧
  a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧ f < 10 ∧ g < 10 ∧ h < 10 →
  (10 * a + b) - (10 * c + d) = 10 * e + d →
  e * f = 10 * d + c →
  (10 * g + d) + (10 * g + b) = 10 * h + c →
  a = 9 ∧ b = 8 ∧ c = 2 ∧ d = 4 ∧ e = 7 ∧ f = 6 ∧ g = 1 ∧ h = 3 :=
by
  intros a b c d e f g h
  intros h1 h2 h3
  sorry

end verify_digits_l234_234942


namespace total_sample_needed_l234_234837

-- Given constants
def elementary_students : ℕ := 270
def junior_high_students : ℕ := 360
def senior_high_students : ℕ := 300
def junior_high_sample : ℕ := 12

-- Calculate the total number of students in the school
def total_students : ℕ := elementary_students + junior_high_students + senior_high_students

-- Define the sampling ratio based on junior high section
def sampling_ratio : ℚ := junior_high_sample / junior_high_students

-- Apply the sampling ratio to the total number of students to get the total sample size
def total_sample : ℚ := sampling_ratio * total_students

-- Prove that the total number of students that need to be sampled is 31
theorem total_sample_needed : total_sample = 31 := sorry

end total_sample_needed_l234_234837


namespace determine_m_range_l234_234296

theorem determine_m_range (m : ℝ) (h : (∃ (x y : ℝ), x^2 + y^2 + 2 * m * x + 2 = 0) ∧ 
                                    (∃ (r : ℝ) (h_r : r^2 = m^2 - 2), π * r^2 ≥ 4 * π)) :
  (m ≤ -Real.sqrt 6 ∨ m ≥ Real.sqrt 6) :=
by
  sorry

end determine_m_range_l234_234296


namespace most_marbles_l234_234028

def total_marbles := 24
def red_marble_fraction := 1 / 4
def red_marbles := red_marble_fraction * total_marbles
def blue_marbles := red_marbles + 6
def yellow_marbles := total_marbles - red_marbles - blue_marbles

theorem most_marbles : blue_marbles > red_marbles ∧ blue_marbles > yellow_marbles :=
by
  sorry

end most_marbles_l234_234028


namespace cos_cofunction_identity_l234_234801

theorem cos_cofunction_identity (α : ℝ) (h : Real.sin (30 * Real.pi / 180 + α) = Real.sqrt 3 / 2) :
  Real.cos (60 * Real.pi / 180 - α) = Real.sqrt 3 / 2 := by
  sorry

end cos_cofunction_identity_l234_234801


namespace cost_price_of_article_l234_234499

theorem cost_price_of_article (C SP1 SP2 G1 G2 : ℝ) 
  (h_SP1 : SP1 = 160) 
  (h_SP2 : SP2 = 220) 
  (h_gain_relation : G2 = 1.05 * G1) 
  (h_G1 : G1 = SP1 - C) 
  (h_G2 : G2 = SP2 - C) : C = 1040 :=
by
  sorry

end cost_price_of_article_l234_234499


namespace fourth_intersection_point_exists_l234_234708

noncomputable def find_fourth_intersection_point : Prop :=
  let points := [(4, 1/2), (-6, -1/3), (1/4, 8), (-2/3, -3)]
  ∃ (h k r : ℝ), 
  ∀ (x y : ℝ), (x, y) ∈ points → (x - h) ^ 2 + (y - k) ^ 2 = r ^ 2

theorem fourth_intersection_point_exists :
  find_fourth_intersection_point :=
by
  sorry

end fourth_intersection_point_exists_l234_234708


namespace probability_single_trial_l234_234397

theorem probability_single_trial 
  (p : ℝ) 
  (h₁ : ∀ n : ℕ, 1 ≤ n → ∃ x : ℝ, x = (1 - (1 - p) ^ n)) 
  (h₂ : 1 - (1 - p) ^ 4 = 65 / 81) : 
  p = 1 / 3 :=
by 
  sorry

end probability_single_trial_l234_234397


namespace aprons_to_sew_tomorrow_l234_234596

def total_aprons : ℕ := 150
def already_sewn : ℕ := 13
def sewn_today (already_sewn : ℕ) : ℕ := 3 * already_sewn
def sewn_tomorrow (total_aprons : ℕ) (already_sewn : ℕ) (sewn_today : ℕ) : ℕ :=
  let remaining := total_aprons - (already_sewn + sewn_today)
  remaining / 2

theorem aprons_to_sew_tomorrow : sewn_tomorrow total_aprons already_sewn (sewn_today already_sewn) = 49 :=
  by 
    sorry

end aprons_to_sew_tomorrow_l234_234596


namespace polynomial_n_values_possible_num_values_of_n_l234_234768

theorem polynomial_n_values_possible :
  ∃ (n : ℤ), 
    (∀ (x : ℝ), x^3 - 4050 * x^2 + (m : ℝ) * x + (n : ℝ) = 0 → x > 0) ∧
    (∃ a : ℤ, a > 0 ∧ ∀ (x : ℝ), x^3 - 4050 * x^2 + (m : ℝ) * x + (n : ℝ) = 0 → 
      x = a ∨ x = a / 4 + r ∨ x = a / 4 - r) ∧
    1 ≤ r^2 ∧ r^2 ≤ 4090499 :=
sorry

theorem num_values_of_n : 
  ∃ (n_values : ℤ), n_values = 4088474 :=
sorry

end polynomial_n_values_possible_num_values_of_n_l234_234768


namespace uncover_area_is_64_l234_234735

-- Conditions as definitions
def length_of_floor := 10
def width_of_floor := 8
def side_of_carpet := 4

-- The statement of the problem
theorem uncover_area_is_64 :
  let area_of_floor := length_of_floor * width_of_floor
  let area_of_carpet := side_of_carpet * side_of_carpet
  let uncovered_area := area_of_floor - area_of_carpet
  uncovered_area = 64 :=
by
  sorry

end uncover_area_is_64_l234_234735


namespace probability_diagonals_intersect_inside_decagon_l234_234526

/-- Two diagonals of a regular decagon are chosen. 
  What is the probability that their intersection lies inside the decagon?
-/
theorem probability_diagonals_intersect_inside_decagon : 
  let num_diagonals := 35
  let num_pairs := num_diagonals * (num_diagonals - 1) / 2
  let num_intersecting_pairs := 210
  let probability := num_intersecting_pairs / num_pairs
  probability = 42 / 119 :=
by
  -- Definitions based on the conditions
  let num_diagonals := (10 * (10 - 3)) / 2
  let num_pairs := num_diagonals * (num_diagonals - 1) / 2
  let num_intersecting_pairs := 210

  -- Simplified probability
  let probability := num_intersecting_pairs / num_pairs

  -- Sorry used to skip the proof
  sorry

end probability_diagonals_intersect_inside_decagon_l234_234526


namespace rhombus_side_length_15_l234_234433

variable {p : ℝ} (h_p : p = 60)
variable {n : ℕ} (h_n : n = 4)

noncomputable def side_length_of_rhombus (p : ℝ) (n : ℕ) : ℝ :=
p / n

theorem rhombus_side_length_15 (h_p : p = 60) (h_n : n = 4) :
  side_length_of_rhombus p n = 15 :=
by
  sorry

end rhombus_side_length_15_l234_234433


namespace largest_divisor_of_composite_l234_234602

theorem largest_divisor_of_composite (n : ℕ) (h : n > 1 ∧ ¬ Nat.Prime n) : 12 ∣ (n^4 - n^2) :=
sorry

end largest_divisor_of_composite_l234_234602


namespace find_ages_l234_234785

theorem find_ages (M F S : ℕ) 
  (h1 : M = 2 * F / 5)
  (h2 : M + 10 = (F + 10) / 2)
  (h3 : S + 10 = 3 * (F + 10) / 4) :
  M = 20 ∧ F = 50 ∧ S = 35 := 
by
  sorry

end find_ages_l234_234785


namespace min_cost_open_top_rectangular_pool_l234_234828

theorem min_cost_open_top_rectangular_pool
  (volume : ℝ)
  (depth : ℝ)
  (cost_bottom_per_sqm : ℝ)
  (cost_walls_per_sqm : ℝ)
  (h1 : volume = 18)
  (h2 : depth = 2)
  (h3 : cost_bottom_per_sqm = 200)
  (h4 : cost_walls_per_sqm = 150) :
  ∃ (min_cost : ℝ), min_cost = 5400 :=
by
  sorry

end min_cost_open_top_rectangular_pool_l234_234828


namespace average_score_for_girls_at_both_schools_combined_l234_234560

/-
  The following conditions are given:
  - Average score for boys at Lincoln HS = 75
  - Average score for boys at Monroe HS = 85
  - Average score for boys at both schools combined = 82
  - Average score for girls at Lincoln HS = 78
  - Average score for girls at Monroe HS = 92
  - Average score for boys and girls combined at Lincoln HS = 76
  - Average score for boys and girls combined at Monroe HS = 88

  The goal is to prove that the average score for the girls at both schools combined is 89.
-/
theorem average_score_for_girls_at_both_schools_combined 
  (L l M m : ℕ)
  (h1 : (75 * L + 78 * l) / (L + l) = 76)
  (h2 : (85 * M + 92 * m) / (M + m) = 88)
  (h3 : (75 * L + 85 * M) / (L + M) = 82)
  : (78 * l + 92 * m) / (l + m) = 89 := 
sorry

end average_score_for_girls_at_both_schools_combined_l234_234560


namespace a_plus_b_eq_11_l234_234079

noncomputable def f (a b x : ℝ) : ℝ := x^3 + 3 * a * x^2 + b * x + a^2

theorem a_plus_b_eq_11 (a b : ℝ) 
  (h1 : ∀ x, f a b x ≤ f a b (-1))
  (h2 : f a b (-1) = 0) 
  : a + b = 11 :=
sorry

end a_plus_b_eq_11_l234_234079


namespace find_x_modulo_l234_234399

theorem find_x_modulo (k : ℤ) : ∃ x : ℤ, x = 18 + 31 * k ∧ ((37 * x) % 31 = 15) := by
  sorry

end find_x_modulo_l234_234399


namespace largest_common_value_l234_234900

theorem largest_common_value (a : ℕ) (h1 : a % 4 = 3) (h2 : a % 9 = 5) (h3 : a < 600) :
  a = 599 :=
sorry

end largest_common_value_l234_234900


namespace circus_dogs_ratio_l234_234494

theorem circus_dogs_ratio :
  ∀ (x y : ℕ), 
  (x + y = 12) → (2 * x + 4 * y = 36) → (x = y) → x / y = 1 :=
by
  intros x y h1 h2 h3
  sorry

end circus_dogs_ratio_l234_234494


namespace functional_equation_continuous_function_l234_234463

noncomputable def f : ℝ → ℝ := sorry

theorem functional_equation_continuous_function (f : ℝ → ℝ) (x₀ : ℝ) (h1 : Continuous f) (h2 : f x₀ ≠ 0) 
  (h3 : ∀ x y : ℝ, f (x + y) = f x * f y) : 
  ∃ a > 0, ∀ x : ℝ, f x = a ^ x := 
by
  sorry

end functional_equation_continuous_function_l234_234463


namespace day_of_20th_is_Thursday_l234_234059

noncomputable def day_of_week (d : ℕ) : String :=
  match d % 7 with
  | 0 => "Saturday"
  | 1 => "Sunday"
  | 2 => "Monday"
  | 3 => "Tuesday"
  | 4 => "Wednesday"
  | 5 => "Thursday"
  | 6 => "Friday"
  | _ => "Unknown"

theorem day_of_20th_is_Thursday (s1 s2 s3: ℕ) (h1: 2 ≤ s1) (h2: s1 ≤ 30) (h3: s2 = s1 + 14) (h4: s3 = s2 + 14) (h5: s3 ≤ 30) (h6: day_of_week s1 = "Sunday") : 
  day_of_week 20 = "Thursday" :=
by
  sorry

end day_of_20th_is_Thursday_l234_234059


namespace twice_brother_age_l234_234142

theorem twice_brother_age (current_my_age : ℕ) (current_brother_age : ℕ) (years : ℕ) :
  current_my_age = 20 →
  (current_my_age + years) + (current_brother_age + years) = 45 →
  current_my_age + years = 2 * (current_brother_age + years) →
  years = 10 :=
by 
  intros h1 h2 h3
  sorry

end twice_brother_age_l234_234142


namespace find_efg_correct_l234_234422

noncomputable def find_efg (M : ℕ) : ℕ :=
  let efgh := M % 10000
  let e := efgh / 1000
  let efg := efgh / 10
  if (M^2 % 10000 = efgh) ∧ (e ≠ 0) ∧ ((M % 32 = 0 ∧ (M - 1) % 125 = 0) ∨ (M % 125 = 0 ∧ (M - 1) % 32 = 0))
  then efg
  else 0
  
theorem find_efg_correct {M : ℕ} (h_conditions: (M^2 % 10000 = M % 10000) ∧ (M % 32 = 0 ∧ (M - 1) % 125 = 0 ∨ M % 125 = 0 ∧ (M-1) % 32 = 0) ∧ ((M % 10000 / 1000) ≠ 0)) :
  find_efg M = 362 :=
by
  sorry

end find_efg_correct_l234_234422


namespace quadratic_roots_conditions_l234_234371

-- Definitions of the given conditions.
variables (a b c : ℝ)  -- Coefficients of the quadratic trinomial
variable (h : b^2 - 4 * a * c ≥ 0)  -- Given condition that the discriminant is non-negative

-- Statement to prove:
theorem quadratic_roots_conditions (a b c : ℝ) (h : b^2 - 4 * a * c ≥ 0) :
  ¬(∀ x : ℝ, a^2 * x^2 + b^2 * x + c^2 = 0) ∧ (∀ x : ℝ, a^3 * x^2 + b^3 * x + c^3 = 0 → b^6 - 4 * a^3 * c^3 ≥ 0) :=
by
  sorry

end quadratic_roots_conditions_l234_234371


namespace silk_diameter_scientific_notation_l234_234396

-- Definition of the given condition
def silk_diameter := 0.000014 

-- The goal to be proved
theorem silk_diameter_scientific_notation : silk_diameter = 1.4 * 10^(-5) := 
by 
  sorry

end silk_diameter_scientific_notation_l234_234396


namespace cube_weight_l234_234632

theorem cube_weight (l1 l2 V1 V2 k : ℝ) (h1: l2 = 2 * l1) (h2: V1 = l1^3) (h3: V2 = (2 * l1)^3) (h4: w2 = 48) (h5: V2 * k = w2) (h6: V1 * k = w1):
  w1 = 6 :=
by
  sorry

end cube_weight_l234_234632


namespace optionD_is_not_linear_system_l234_234189

-- Define the equations for each option
def eqA1 (x y : ℝ) : Prop := 3 * x + 2 * y = 10
def eqA2 (x y : ℝ) : Prop := 2 * x - 3 * y = 5

def eqB1 (x y : ℝ) : Prop := 3 * x + 5 * y = 1
def eqB2 (x y : ℝ) : Prop := 2 * x - y = 4

def eqC1 (x y : ℝ) : Prop := x + 5 * y = 1
def eqC2 (x y : ℝ) : Prop := x - 5 * y = 2

def eqD1 (x y : ℝ) : Prop := x - y = 1
def eqD2 (x y : ℝ) : Prop := y + 1 / x = 3

-- Define the property of a linear equation
def is_linear (eq : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b c : ℝ), ∀ x y, eq x y → a * x + b * y = c

-- State the theorem
theorem optionD_is_not_linear_system : ¬ (is_linear eqD1 ∧ is_linear eqD2) :=
by
  sorry

end optionD_is_not_linear_system_l234_234189


namespace packs_sold_to_uncle_is_correct_l234_234576

-- Define the conditions and constants
def total_packs_needed := 50
def packs_sold_to_grandmother := 12
def packs_sold_to_neighbor := 5
def packs_left_to_sell := 26

-- Calculate total packs sold so far
def total_packs_sold := total_packs_needed - packs_left_to_sell

-- Calculate total packs sold to grandmother and neighbor
def packs_sold_to_grandmother_and_neighbor := packs_sold_to_grandmother + packs_sold_to_neighbor

-- The pack sold to uncle
def packs_sold_to_uncle := total_packs_sold - packs_sold_to_grandmother_and_neighbor

-- Prove the packs sold to uncle
theorem packs_sold_to_uncle_is_correct : packs_sold_to_uncle = 7 := by
  -- The proof steps are omitted
  sorry

end packs_sold_to_uncle_is_correct_l234_234576


namespace purple_chips_selected_is_one_l234_234066

noncomputable def chips_selected (B G P R x : ℕ) : Prop :=
  (1^B) * (5^G) * (x^P) * (11^R) = 140800 ∧ 5 < x ∧ x < 11

theorem purple_chips_selected_is_one :
  ∃ B G P R x, chips_selected B G P R x ∧ P = 1 :=
by {
  sorry
}

end purple_chips_selected_is_one_l234_234066


namespace max_single_player_salary_l234_234767

variable (n : ℕ) (m : ℕ) (p : ℕ) (s : ℕ)

theorem max_single_player_salary
  (h1 : n = 18)
  (h2 : ∀ i : ℕ, i < n → p ≥ 20000)
  (h3 : s = 800000)
  (h4 : n * 20000 ≤ s) :
  ∃ x : ℕ, x = 460000 :=
by
  sorry

end max_single_player_salary_l234_234767


namespace find_x_l234_234925

-- Definitions of the conditions in Lean 4
def angle_sum_180 (A B C : ℝ) : Prop := A + B + C = 180
def angle_BAC_eq_90 (A : ℝ) : Prop := A = 90
def angle_BCA_eq_2x (C x : ℝ) : Prop := C = 2 * x
def angle_ABC_eq_3x (B x : ℝ) : Prop := B = 3 * x

-- The theorem we need to prove
theorem find_x (A B C x : ℝ) 
  (h1 : angle_sum_180 A B C) 
  (h2 : angle_BAC_eq_90 A)
  (h3 : angle_BCA_eq_2x C x) 
  (h4 : angle_ABC_eq_3x B x) : x = 18 :=
by 
  sorry

end find_x_l234_234925


namespace least_number_to_add_l234_234784

theorem least_number_to_add (n : ℕ) : 
  (∀ k : ℕ, n = 1 + k * 425 ↔ n + 1019 % 425 = 0) → n = 256 := 
sorry

end least_number_to_add_l234_234784


namespace ordered_triples_count_l234_234978

theorem ordered_triples_count :
  {n : ℕ // n = 4} :=
sorry

end ordered_triples_count_l234_234978


namespace area_triangle_ABC_l234_234725

noncomputable def point := ℝ × ℝ

structure Parallelogram (A B C D : point) : Prop :=
(parallel_AB_CD : ∃ m1 m2, m1 ≠ m2 ∧ (A.2 - B.2) / (A.1 - B.1) = m1 ∧ (C.2 - D.2) / (C.1 - D.1) = m2)
(equal_heights : ∃ h, (B.2 - A.2 = h) ∧ (C.2 - D.2 = h))
(area_parallelogram : (B.1 - A.1) * (B.2 - A.2) + (C.1 - D.1) * (C.2 - D.2) = 27)
(thrice_length : (C.1 - D.1) = 3 * (B.1 - A.1))

theorem area_triangle_ABC (A B C D : point) (h : Parallelogram A B C D) : 
  ∃ triangle_area : ℝ, triangle_area = 13.5 :=
by
  sorry

end area_triangle_ABC_l234_234725


namespace product_of_a_and_c_l234_234511

theorem product_of_a_and_c (a b c : ℝ) (h1 : a + b + c = 100) (h2 : a - b = 20) (h3 : b - c = 30) : a * c = 378.07 :=
by
  sorry

end product_of_a_and_c_l234_234511


namespace fruit_bowl_remaining_l234_234532

-- Define the initial conditions
def oranges : Nat := 3
def lemons : Nat := 6
def fruits_eaten : Nat := 3

-- Define the total count of fruits initially
def total_fruits : Nat := oranges + lemons

-- The goal is to prove remaining fruits == 6
theorem fruit_bowl_remaining : total_fruits - fruits_eaten = 6 := by
  sorry

end fruit_bowl_remaining_l234_234532


namespace password_encryption_l234_234153

variables (a b x : ℝ)

theorem password_encryption :
  3 * a * (x^2 - 1) - 3 * b * (x^2 - 1) = 3 * (x + 1) * (x - 1) * (a - b) :=
by sorry

end password_encryption_l234_234153


namespace distance_AB_polar_l234_234172

open Real

theorem distance_AB_polar (A B : ℝ × ℝ) (θ₁ θ₂ : ℝ) (hA : A = (4, θ₁)) (hB : B = (12, θ₂))
  (hθ : θ₁ - θ₂ = π / 3) : dist (4 * cos θ₁, 4 * sin θ₁) (12 * cos θ₂, 12 * sin θ₂) = 4 * sqrt 13 :=
by
  sorry

end distance_AB_polar_l234_234172


namespace solution_of_valve_problem_l234_234971

noncomputable def valve_filling_problem : Prop :=
  ∃ (x y z : ℝ), 
    (x + y + z = 1 / 2) ∧    -- Condition when all three valves are open
    (x + z = 1 / 3) ∧        -- Condition when valves X and Z are open
    (y + z = 1 / 4) ∧        -- Condition when valves Y and Z are open
    (1 / (x + y) = 2.4)      -- Required condition for valves X and Y

theorem solution_of_valve_problem : valve_filling_problem :=
sorry

end solution_of_valve_problem_l234_234971


namespace volume_le_one_fourth_of_original_volume_of_sub_tetrahedron_l234_234025

noncomputable def volume_tetrahedron (V A B C : Point) : ℝ := sorry

def is_interior_point (M V A B C : Point) : Prop := sorry -- Definition of an interior point

def is_barycenter (M V A B C : Point) : Prop := sorry -- Definition of a barycenter

def intersects_lines_planes (M V A B C A1 B1 C1 : Point) : Prop := sorry -- Definition of intersection points

def intersects_lines_sides (V A1 B1 C1 A B C A2 B2 C2 : Point) : Prop := sorry -- Definition of intersection points with sides

theorem volume_le_one_fourth_of_original (V A B C: Point) 
  (M : Point) (A1 B1 C1 A2 B2 C2 : Point) 
  (h_interior : is_interior_point M V A B C) 
  (h_intersects_planes : intersects_lines_planes M V A B C A1 B1 C1) 
  (h_intersects_sides : intersects_lines_sides V A1 B1 C1 A B C A2 B2 C2) :
  volume_tetrahedron V A2 B2 C2 ≤ (1/4) * volume_tetrahedron V A B C :=
sorry

theorem volume_of_sub_tetrahedron (V A B C: Point) 
  (M V1 : Point) (A1 B1 C1 : Point)
  (h_barycenter : is_barycenter M V A B C)
  (h_intersects_planes : intersects_lines_planes M V A B C A1 B1 C1)
  (h_point_V1 : intersects_something_to_find_V1) : 
  volume_tetrahedron V1 A1 B1 C1 = (1/4) * volume_tetrahedron V A B C :=
sorry

end volume_le_one_fourth_of_original_volume_of_sub_tetrahedron_l234_234025


namespace digit_sum_eq_21_l234_234266

theorem digit_sum_eq_21 (A B C D: ℕ) (h1: A ≠ 0) 
    (h2: (A * 10 + B) * 100 + (C * 10 + D) = (C * 10 + D)^2 - (A * 10 + B)^2) 
    (hA: A < 10) (hB: B < 10) (hC: C < 10) (hD: D < 10) : 
    A + B + C + D = 21 :=
by 
  sorry

end digit_sum_eq_21_l234_234266


namespace math_problem_l234_234060

noncomputable def compute_value (a b c : ℝ) : ℝ :=
  (b / (a + b)) + (c / (b + c)) + (a / (c + a))

theorem math_problem (a b c : ℝ)
  (h1 : (a * c) / (a + b) + (b * a) / (b + c) + (c * b) / (c + a) = -12)
  (h2 : (b * c) / (a + b) + (c * a) / (b + c) + (a * b) / (c + a) = 15) :
  compute_value a b c = 6 :=
sorry

end math_problem_l234_234060


namespace molecular_weight_of_moles_l234_234538

-- Approximate atomic weights
def atomic_weight_N := 14.01
def atomic_weight_O := 16.00

-- Molecular weight of N2O3
def molecular_weight_N2O3 := (2 * atomic_weight_N) + (3 * atomic_weight_O)

-- Given the total molecular weight of some moles of N2O3
def total_molecular_weight : ℝ := 228

-- We aim to prove that the total molecular weight of some moles of N2O3 equals 228 g
theorem molecular_weight_of_moles (h: molecular_weight_N2O3 ≠ 0) :
  total_molecular_weight = 228 := by
  sorry

end molecular_weight_of_moles_l234_234538


namespace students_speaking_Gujarati_l234_234127

theorem students_speaking_Gujarati 
  (total_students : ℕ)
  (students_Hindi : ℕ)
  (students_Marathi : ℕ)
  (students_two_languages : ℕ)
  (students_all_three_languages : ℕ)
  (students_total_set: 22 = total_students)
  (students_H_set: 15 = students_Hindi)
  (students_M_set: 6 = students_Marathi)
  (students_two_set: 2 = students_two_languages)
  (students_all_three_set: 1 = students_all_three_languages) :
  ∃ (students_Gujarati : ℕ), 
  22 = students_Gujarati + 15 + 6 - 2 + 1 ∧ students_Gujarati = 2 :=
by
  sorry

end students_speaking_Gujarati_l234_234127


namespace infinitely_many_k_numbers_unique_k_4_l234_234031

theorem infinitely_many_k_numbers_unique_k_4 :
  ∀ k : ℕ, (∃ n : ℕ, (∃ r : ℕ, n = r * (r + k)) ∧ (∃ m : ℕ, n = m^2 - k)
          ∧ ∀ N : ℕ, ∃ r : ℕ, ∃ m : ℕ, N < r ∧ (r * (r + k) = m^2 - k)) ↔ k = 4 :=
by
  sorry

end infinitely_many_k_numbers_unique_k_4_l234_234031


namespace nancy_carrots_l234_234081

theorem nancy_carrots (picked_day_1 threw_out total_left total_final picked_next_day : ℕ)
  (h1 : picked_day_1 = 12)
  (h2 : threw_out = 2)
  (h3 : total_final = 31)
  (h4 : total_left = picked_day_1 - threw_out)
  (h5 : total_final = total_left + picked_next_day) :
  picked_next_day = 21 :=
by
  sorry

end nancy_carrots_l234_234081


namespace latus_rectum_of_parabola_l234_234724

theorem latus_rectum_of_parabola (p : ℝ) (hp : 0 < p) (A : ℝ × ℝ) (hA : A = (1, 1/2)) :
  ∃ a : ℝ, y^2 = 4 * a * x → A.2 ^ 2 = 4 * a * A.1 → x = -1 / (4 * a) → x = -1 / 16 :=
by
  sorry

end latus_rectum_of_parabola_l234_234724


namespace eds_weight_l234_234253

variable (Al Ben Carl Ed : ℕ)

def weight_conditions : Prop :=
  Carl = 175 ∧ Ben = Carl - 16 ∧ Al = Ben + 25 ∧ Ed = Al - 38

theorem eds_weight (h : weight_conditions Al Ben Carl Ed) : Ed = 146 :=
by
  -- Conditions
  have h1 : Carl = 175    := h.1
  have h2 : Ben = Carl - 16 := h.2.1
  have h3 : Al = Ben + 25   := h.2.2.1
  have h4 : Ed = Al - 38    := h.2.2.2
  -- Proof itself is omitted, sorry placeholder
  sorry

end eds_weight_l234_234253


namespace average_earning_week_l234_234738

theorem average_earning_week (D1 D2 D3 D4 D5 D6 D7 : ℝ)
  (h1 : (D1 + D2 + D3 + D4) / 4 = 25)
  (h2 : (D4 + D5 + D6 + D7) / 4 = 22)
  (h3 : D4 = 20) : 
  (D1 + D2 + D3 + D4 + D5 + D6 + D7) / 7 = 24 :=
by
  sorry

end average_earning_week_l234_234738


namespace prop_2_l234_234309

variables (m n : Plane → Prop) (α β γ : Plane)

def perpendicular (m : Line) (α : Plane) : Prop :=
  -- define perpendicular relationship between line and plane
  sorry

def parallel (m : Line) (n : Line) : Prop :=
  -- define parallel relationship between two lines
  sorry

-- The proof of proposition (2) converted into Lean 4 statement
theorem prop_2 (hm₁ : perpendicular m α) (hn₁ : perpendicular n α) : parallel m n :=
  sorry

end prop_2_l234_234309


namespace sum_m_n_zero_l234_234555

theorem sum_m_n_zero
  (m n p : ℝ)
  (h1 : mn + p^2 + 4 = 0)
  (h2 : m - n = 4) :
  m + n = 0 :=
sorry

end sum_m_n_zero_l234_234555


namespace arithmetic_mean_difference_l234_234407

theorem arithmetic_mean_difference (p q r : ℝ) 
  (h1 : (p + q) / 2 = 10) 
  (h2 : (q + r) / 2 = 20) : 
  r - p = 20 := 
sorry

end arithmetic_mean_difference_l234_234407


namespace part_one_part_two_part_three_l234_234159

def f(x : ℝ) := x^2 - 1
def g(a x : ℝ) := a * |x - 1|

-- (I)
theorem part_one (a : ℝ) : 
  ((∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ |f x₁| = g a x₁ ∧ |f x₂| = g a x₂) ↔ (a = 0 ∨ a = 2)) :=
sorry

-- (II)
theorem part_two (a : ℝ) : 
  (∀ x : ℝ, f x ≥ g a x) ↔ (a <= -2) :=
sorry

-- (III)
def G(a x : ℝ) := |f x| + g a x

theorem part_three (a : ℝ) (h : a < 0) : 
  (∀ x ∈ [-2, 2], G a x ≤ if a <= -3 then 0 else 3 + a) :=
sorry

end part_one_part_two_part_three_l234_234159


namespace probability_correct_l234_234588

-- Define the total number of bulbs, good quality bulbs, and inferior quality bulbs
def total_bulbs : ℕ := 6
def good_bulbs : ℕ := 4
def inferior_bulbs : ℕ := 2

-- Define the probability of drawing one good bulb and one inferior bulb with replacement
def probability_one_good_one_inferior : ℚ := (good_bulbs * inferior_bulbs * 2) / (total_bulbs ^ 2)

-- Theorem stating that the probability of drawing one good bulb and one inferior bulb is 4/9
theorem probability_correct : probability_one_good_one_inferior = 4 / 9 := 
by
  -- Proof is skipped here
  sorry

end probability_correct_l234_234588


namespace repeating_decimal_sum_l234_234913

open Real

noncomputable def repeating_decimal_to_fraction (d: ℕ) : ℚ :=
  if d = 3 then 1/3 else if d = 7 then 7/99 else if d = 9 then 1/111 else 0 -- specific case of 3, 7, 9.

theorem repeating_decimal_sum:
  let x := repeating_decimal_to_fraction 3
  let y := repeating_decimal_to_fraction 7
  let z := repeating_decimal_to_fraction 9
  x + y + z = 499 / 1189 :=
by
  sorry -- Proof is omitted

end repeating_decimal_sum_l234_234913


namespace average_screen_time_per_player_l234_234481

def video_point_guard : ℕ := 130
def video_shooting_guard : ℕ := 145
def video_small_forward : ℕ := 85
def video_power_forward : ℕ := 60
def video_center : ℕ := 180
def total_video_time : ℕ := 
  video_point_guard + video_shooting_guard + video_small_forward + video_power_forward + video_center
def total_video_time_minutes : ℕ := total_video_time / 60
def number_of_players : ℕ := 5

theorem average_screen_time_per_player : total_video_time_minutes / number_of_players = 2 :=
  sorry

end average_screen_time_per_player_l234_234481


namespace gcd_98_63_l234_234044

-- The statement of the problem in Lean 4
theorem gcd_98_63 : Nat.gcd 98 63 = 7 := by
  sorry

end gcd_98_63_l234_234044


namespace number_of_valid_rods_l234_234465

theorem number_of_valid_rods : ∃ n, n = 22 ∧
  (∀ (d : ℕ), 1 < d ∧ d < 25 ∧ d ≠ 4 ∧ d ≠ 9 ∧ d ≠ 12 → d ∈ {d | d > 0}) :=
by
  use 22
  sorry

end number_of_valid_rods_l234_234465


namespace maximum_value_of_transformed_function_l234_234879

theorem maximum_value_of_transformed_function (a b : ℝ) (h_max : ∀ x : ℝ, a * (Real.cos x) + b ≤ 1)
  (h_min : ∀ x : ℝ, a * (Real.cos x) + b ≥ -7) : 
  ∃ ab : ℝ, (ab = 3 + a * b * (Real.sin x)) ∧ (∀ x : ℝ, ab ≤ 15) :=
by
  sorry

end maximum_value_of_transformed_function_l234_234879


namespace complex_multiplication_l234_234573

def i := Complex.I

theorem complex_multiplication (i := Complex.I) : (-1 + i) * (2 - i) = -1 + 3 * i := 
by 
    -- The actual proof steps would go here.
    sorry

end complex_multiplication_l234_234573


namespace part_a_part_b_part_c_l234_234996

-- Part (a) Lean Statement
theorem part_a (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) : 
  (∃ k : ℝ, k = 2 * p / (p + 1)) :=
by
  -- Definitions and conditions would go here
  sorry

-- Part (b) Lean Statement
theorem part_b (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) :
  (∃ q : ℝ, q = 1 - p ∧ ∃ r : ℝ, r = 2 * p / (2 * p + (1 - p) ^ 2)) :=
by
  -- Definitions and conditions would go here
  sorry

-- Part (c) Lean Statement
theorem part_c (N : ℕ) (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) : 
  (∃ S : ℝ, S = N * p / (p + 1)) :=
by
  -- Definitions and conditions would go here
  sorry

end part_a_part_b_part_c_l234_234996


namespace necessary_but_not_sufficient_condition_l234_234740

theorem necessary_but_not_sufficient_condition (a : ℝ) :
  (a < 2) → (∃ x : ℂ, x^2 + (a : ℂ) * x + 1 = 0 ∧ x.im ≠ 0) :=
by
  sorry

end necessary_but_not_sufficient_condition_l234_234740


namespace find_x_l234_234529

def vector (α : Type*) := α × α

def parallel (a b : vector ℝ) : Prop :=
a.1 * b.2 - a.2 * b.1 = 0

theorem find_x (x : ℝ) (a b : vector ℝ)
  (ha : a = (1, 2))
  (hb : b = (x, 4))
  (h : parallel a b) : x = 2 :=
by sorry

end find_x_l234_234529


namespace students_taking_both_languages_l234_234063

theorem students_taking_both_languages (total_students students_neither students_french students_german : ℕ) (h1 : total_students = 69)
  (h2 : students_neither = 15) (h3 : students_french = 41) (h4 : students_german = 22) :
  (students_french + students_german - (total_students - students_neither) = 9) :=
by
  sorry

end students_taking_both_languages_l234_234063


namespace coefficient_m5_n5_in_expansion_l234_234527

-- Definition of the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Goal: prove the coefficient of m^5 n^5 in the expansion of (m+n)^{10} is 252
theorem coefficient_m5_n5_in_expansion : binomial 10 5 = 252 :=
by
  sorry

end coefficient_m5_n5_in_expansion_l234_234527


namespace new_prism_volume_l234_234963

-- Define the original volume
def original_volume : ℝ := 12

-- Define the dimensions modification factors
def length_factor : ℝ := 2
def width_factor : ℝ := 2
def height_factor : ℝ := 3

-- Define the volume of the new prism
def new_volume := (length_factor * width_factor * height_factor) * original_volume

-- State the theorem to prove
theorem new_prism_volume : new_volume = 144 := 
by sorry

end new_prism_volume_l234_234963


namespace marys_score_l234_234622

def score (c w : ℕ) : ℕ := 30 + 4 * c - w
def valid_score_range (s : ℕ) : Prop := s > 90 ∧ s ≤ 170

theorem marys_score : ∃ c w : ℕ, c + w ≤ 35 ∧ score c w = 170 ∧ 
  ∀ (s : ℕ), (valid_score_range s ∧ ∃ c' w', score c' w' = s ∧ c' + w' ≤ 35) → 
  (s = 170) :=
by
  sorry

end marys_score_l234_234622


namespace g_value_at_2_over_9_l234_234280

theorem g_value_at_2_over_9 (g : ℝ → ℝ) 
  (hg0 : g 0 = 0)
  (hgmono : ∀ ⦃x y⦄, 0 ≤ x → x < y → y ≤ 1 → g x ≤ g y)
  (hg_symm : ∀ x, 0 ≤ x → x ≤ 1 → g (1 - x) = 1 - g x)
  (hg_frac : ∀ x, 0 ≤ x → x ≤ 1 → g (x / 4) = g x / 3) :
  g (2 / 9) = 8 / 27 :=
sorry

end g_value_at_2_over_9_l234_234280


namespace compare_sqrt_terms_l234_234425

/-- Compare the sizes of 5 * sqrt 2 and 3 * sqrt 3 -/
theorem compare_sqrt_terms : 5 * Real.sqrt 2 > 3 * Real.sqrt 3 := 
by sorry

end compare_sqrt_terms_l234_234425


namespace horner_method_v3_correct_l234_234455

-- Define the polynomial function according to Horner's method
def horner (x : ℝ) : ℝ :=
  (((((3 * x - 2) * x + 2) * x - 4) * x) * x - 7)

-- Given the value of x
def x_val : ℝ := 2

-- Define v_3 based on the polynomial evaluated at x = 2 using Horner's method
def v3 : ℝ := horner x_val

-- Theorem stating what we need to prove
theorem horner_method_v3_correct : v3 = 16 :=
  by
    sorry

end horner_method_v3_correct_l234_234455


namespace yard_length_l234_234036

theorem yard_length
  (trees : ℕ) (gaps : ℕ) (distance_between_trees : ℕ) :
  trees = 26 → 
  gaps = trees - 1 → 
  distance_between_trees = 14 → 
  length_of_yard = gaps * distance_between_trees → 
  length_of_yard = 350 :=
by
  intros h_trees h_gaps h_distance h_length
  sorry

end yard_length_l234_234036


namespace length_of_AB_l234_234850

/-- A triangle ABC lies between two parallel lines where AC = 5 cm. Prove that AB = 10 cm. -/
noncomputable def triangle_is_between_two_parallel_lines : Prop := sorry

noncomputable def segmentAC : ℝ := 5

theorem length_of_AB :
  ∃ (AB : ℝ), triangle_is_between_two_parallel_lines ∧ segmentAC = 5 ∧ AB = 10 :=
sorry

end length_of_AB_l234_234850


namespace max_a2_plus_b2_l234_234012

theorem max_a2_plus_b2 (a b : ℝ) (h1 : b = 1) (h2 : 1 ≤ -a + 7) (h3 : 1 ≥ a - 3) : a^2 + b^2 = 37 :=
by {
  sorry
}

end max_a2_plus_b2_l234_234012


namespace possible_number_of_friends_l234_234733

-- Define the conditions and problem statement
def player_structure (total_players : ℕ) (n : ℕ) (m : ℕ) : Prop :=
  total_players = n * m ∧ (n - 1) * m = 15

-- The main theorem to prove the number of friends in the group
theorem possible_number_of_friends : ∃ (N : ℕ), 
  (player_structure N 2 15 ∨ player_structure N 4 5 ∨ player_structure N 6 3 ∨ player_structure N 16 1) ∧
  (N = 16 ∨ N = 18 ∨ N = 20 ∨ N = 30) :=
sorry

end possible_number_of_friends_l234_234733


namespace f1_g1_eq_one_l234_234207

-- Definitions of even and odd functions
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def odd_function (g : ℝ → ℝ) : Prop := ∀ x, g x = -g (-x)

-- Given statement to be proved
theorem f1_g1_eq_one (f g : ℝ → ℝ) (h_even : even_function f) (h_odd : odd_function g)
    (h_diff : ∀ x, f x - g x = x^3 + x^2 + 1) : f 1 + g 1 = 1 :=
  sorry

end f1_g1_eq_one_l234_234207


namespace ratio_a_b_c_l234_234119

-- Given condition 14(a^2 + b^2 + c^2) = (a + 2b + 3c)^2
theorem ratio_a_b_c (a b c : ℝ) (h : 14 * (a^2 + b^2 + c^2) = (a + 2 * b + 3 * c)^2) : 
  a / b = 1 / 2 ∧ b / c = 2 / 3 :=
by 
  sorry

end ratio_a_b_c_l234_234119


namespace prove_q_l234_234893

theorem prove_q 
  (p q : ℝ)
  (h : (∀ x, (x + 3) * (x + p) = x^2 + q * x + 12)) : 
  q = 7 :=
sorry

end prove_q_l234_234893


namespace inequality_condition_l234_234115

theorem inequality_condition (x : ℝ) :
  ((x + 3) * (x - 2) < 0 ↔ -3 < x ∧ x < 2) →
  ((-3 < x ∧ x < 0) → (x + 3) * (x - 2) < 0) →
  ∃ p q : Prop, (p → q) ∧ ¬(q → p) ∧
  p = ((x + 3) * (x - 2) < 0) ∧ q = (-3 < x ∧ x < 0) := by
  sorry

end inequality_condition_l234_234115


namespace profit_rate_l234_234170

variables (list_price : ℝ)
          (discount : ℝ := 0.95)
          (selling_increase : ℝ := 1.6)
          (inflation_rate : ℝ := 1.4)

theorem profit_rate (list_price : ℝ) : 
  (selling_increase / (discount * inflation_rate)) - 1 = 0.203 :=
by 
  sorry

end profit_rate_l234_234170


namespace range_of_m_l234_234888

theorem range_of_m (m x : ℝ) (h₁ : (x / (x - 3) - 2 = m / (x - 3))) (h₂ : x ≠ 3) : x > 0 ↔ m < 6 ∧ m ≠ 3 :=
by
  sorry

end range_of_m_l234_234888


namespace range_of_a_l234_234154

variable (a : ℝ)
variable (x y : ℝ)

def system_of_equations := 
  (5 * x + 2 * y = 11 * a + 18) ∧ 
  (2 * x - 3 * y = 12 * a - 8) ∧
  (x > 0) ∧ 
  (y > 0)

theorem range_of_a (h : system_of_equations a x y) : 
  - (2:ℝ) / 3 < a ∧ a < 2 :=
sorry

end range_of_a_l234_234154


namespace intersection_A_B_l234_234340

-- Define the sets A and B
def A : Set ℤ := {0, 1, 2}
def B : Set ℤ := {x : ℤ | x^2 < 3}

-- Prove that A ∩ B = {0, 1}
theorem intersection_A_B :
  A ∩ B = {0, 1} :=
by
  -- Proof goes here
  sorry

end intersection_A_B_l234_234340


namespace inequality_range_l234_234885

theorem inequality_range (a : ℝ) : (∀ x : ℝ, |x + 2| + |x - 3| > a) → a < 5 :=
  sorry

end inequality_range_l234_234885


namespace intersection_M_N_l234_234763

-- Define set M
def set_M : Set ℤ := {x | -1 ≤ x ∧ x ≤ 3}

-- Define set N
def set_N : Set ℤ := {x | ∃ k : ℕ, k > 0 ∧ x = 2 * k - 1}

-- Define the intersection of M and N
def M_intersect_N : Set ℤ := {1, 3}

-- The theorem to prove
theorem intersection_M_N : set_M ∩ set_N = M_intersect_N :=
by sorry

end intersection_M_N_l234_234763


namespace determine_pairs_l234_234052

open Int

-- Definitions corresponding to the conditions of the problem:
def is_prime (p : ℕ) : Prop := Nat.Prime p
def condition1 (p n : ℕ) : Prop := is_prime p
def condition2 (p n : ℕ) : Prop := n ≤ 2 * p
def condition3 (p n : ℕ) : Prop := (n^(p-1)) ∣ ((p-1)^n + 1)

-- Main theorem statement:
theorem determine_pairs (n p : ℕ) (h1 : condition1 p n) (h2 : condition2 p n) (h3 : condition3 p n) :
  (n = 1 ∧ is_prime p) ∨ (n = 2 ∧ p = 2) ∨ (n = 3 ∧ p = 3) :=
sorry

end determine_pairs_l234_234052


namespace log2_bounds_sum_l234_234473

theorem log2_bounds_sum (a b : ℤ) (h1 : a < b) (h2 : b = a + 1) (h3 : (a : ℝ) < Real.log 50 / Real.log 2) (h4 : Real.log 50 / Real.log 2 < (b : ℝ)) :
  a + b = 11 :=
sorry

end log2_bounds_sum_l234_234473


namespace base_b_arithmetic_l234_234954

theorem base_b_arithmetic (b : ℕ) (h1 : 4 + 3 = 7) (h2 : 6 + 2 = 8) (h3 : 4 + 6 = 10) (h4 : 3 + 4 + 1 = 8) : b = 9 :=
  sorry

end base_b_arithmetic_l234_234954


namespace intersection_of_sets_l234_234875

def A : Set ℝ := { x | 1 ≤ x ∧ x ≤ 3 }
def B : Set ℝ := { x | 2 < x ∧ x < 4 }

theorem intersection_of_sets : A ∩ B = { x | 2 < x ∧ x ≤ 3 } := 
by 
  sorry

end intersection_of_sets_l234_234875


namespace find_a_l234_234010

-- Defining the curve y in terms of x and a
def curve (x : ℝ) (a : ℝ) : ℝ := x^4 + a*x^2 + 1

-- Defining the derivative of the curve
def derivative (x : ℝ) (a : ℝ) : ℝ := 4*x^3 + 2*a*x

-- The proof statement asserting the value of a
theorem find_a (a : ℝ) (h1 : derivative (-1) a = 8): a = -6 :=
by
  -- we assume here the necessary calculations and logical steps to prove the theorem
  sorry

end find_a_l234_234010


namespace larger_integer_is_24_l234_234430

theorem larger_integer_is_24 {x : ℤ} (h1 : ∃ x, 4 * x = x + 6) :
  ∃ y, y = 4 * x ∧ y = 24 := by
  sorry

end larger_integer_is_24_l234_234430


namespace find_m_l234_234487

def is_good (n : ℤ) : Prop :=
  ¬ (∃ k : ℤ, |n| = k^2)

theorem find_m (m : ℤ) : (m % 4 = 3) → 
  (∃ a b c : ℤ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ is_good a ∧ is_good b ∧ is_good c ∧ (a * b * c) % 2 = 1 ∧ a + b + c = m) :=
sorry

end find_m_l234_234487


namespace incorrect_relationship_f_pi4_f_pi_l234_234222

open Real

noncomputable def f : ℝ → ℝ := sorry

-- Conditions
axiom f_derivative_exists : ∀ x : ℝ, DifferentiableAt ℝ f x
axiom f_derivative_lt_sin2x : ∀ x : ℝ, 0 < x → deriv f x < (sin x) ^ 2
axiom f_symmetric_property : ∀ x : ℝ, f (-x) + f x = 2 * (sin x) ^ 2

theorem incorrect_relationship_f_pi4_f_pi : ¬ (f (π / 4) < f π) :=
by sorry

end incorrect_relationship_f_pi4_f_pi_l234_234222


namespace wrapping_paper_cost_l234_234578
noncomputable def cost_per_roll (shirt_boxes XL_boxes: ℕ) (cost_total: ℝ) : ℝ :=
  let rolls_for_shirts := shirt_boxes / 5
  let rolls_for_xls := XL_boxes / 3
  let total_rolls := rolls_for_shirts + rolls_for_xls
  cost_total / total_rolls

theorem wrapping_paper_cost : cost_per_roll 20 12 32 = 4 :=
by
  sorry

end wrapping_paper_cost_l234_234578


namespace cube_coloring_schemes_l234_234916

theorem cube_coloring_schemes (colors : Finset ℕ) (h : colors.card = 6) :
  ∃ schemes : Nat, schemes = 230 :=
by
  sorry

end cube_coloring_schemes_l234_234916


namespace find_diminished_value_l234_234623

theorem find_diminished_value :
  ∃ (x : ℕ), 1015 - x = Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 12 16) 18) 21) 28 :=
by
  use 7
  simp
  unfold Nat.lcm
  sorry

end find_diminished_value_l234_234623


namespace molecular_weight_constant_l234_234630

-- Given the molecular weight of a compound
def molecular_weight (w : ℕ) := w = 1188

-- Statement about molecular weight of n moles
def weight_of_n_moles (n : ℕ) := n * 1188

theorem molecular_weight_constant (moles : ℕ) : 
  ∀ (w : ℕ), molecular_weight w → ∀ (n : ℕ), weight_of_n_moles n = n * w :=
by
  intro w h n
  sorry

end molecular_weight_constant_l234_234630


namespace min_value_of_a_l234_234363

theorem min_value_of_a
  (f : ℝ → ℝ)
  (h_even : ∀ x : ℝ, f x = f (-x))
  (h_mono : ∀ x y : ℝ, 0 ≤ x → x ≤ y → f x ≤ f y)
  (a : ℝ)
  (h_cond : f (Real.logb 2 a) + f (Real.logb (1/2) a) ≤ 2 * f 1) :
  a = 1/2 := sorry

end min_value_of_a_l234_234363


namespace proper_subset_singleton_l234_234072

theorem proper_subset_singleton : ∀ (P : Set ℕ), P = {0} → (∃ S, S ⊂ P ∧ S = ∅) :=
by
  sorry

end proper_subset_singleton_l234_234072


namespace albert_needs_more_money_l234_234820

def cost_paintbrush : Real := 1.50
def cost_paints : Real := 4.35
def cost_easel : Real := 12.65
def cost_canvas : Real := 7.95
def cost_palette : Real := 3.75
def money_albert_has : Real := 10.60
def total_cost : Real := cost_paintbrush + cost_paints + cost_easel + cost_canvas + cost_palette
def money_needed : Real := total_cost - money_albert_has

theorem albert_needs_more_money : money_needed = 19.60 := by
  sorry

end albert_needs_more_money_l234_234820


namespace brown_eggs_survived_l234_234788

-- Conditions
variables (B : ℕ)  -- Number of brown eggs that survived

-- States that Linda had three times as many white eggs as brown eggs before the fall
def white_eggs_eq_3_times_brown : Prop := 3 * B + B = 12

-- Theorem statement
theorem brown_eggs_survived (h : white_eggs_eq_3_times_brown B) : B = 3 :=
sorry

end brown_eggs_survived_l234_234788


namespace cuboid_total_edge_length_cuboid_surface_area_l234_234739

variables (a b c : ℝ)

theorem cuboid_total_edge_length : 4 * (a + b + c) = 4 * (a + b + c) := 
by
  sorry

theorem cuboid_surface_area : 2 * (a * b + b * c + a * c) = 2 * (a * b + b * c + a * c) := 
by
  sorry

end cuboid_total_edge_length_cuboid_surface_area_l234_234739


namespace domain_of_f_decreasing_on_interval_range_of_f_l234_234192

noncomputable def f (x : ℝ) : ℝ := Real.log (3 + 2 * x - x^2) / Real.log 2

theorem domain_of_f :
  ∀ x : ℝ, (3 + 2 * x - x^2 > 0) ↔ (-1 < x ∧ x < 3) :=
by
  sorry

theorem decreasing_on_interval :
  ∀ (x₁ x₂ : ℝ), (1 < x₁ ∧ x₁ < x₂ ∧ x₂ < 3) →
  f x₂ < f x₁ :=
by
  sorry

theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, -1 < x ∧ x < 3 ∧ y = f x) ↔ y ≤ 2 :=
by
  sorry

end domain_of_f_decreasing_on_interval_range_of_f_l234_234192


namespace total_computers_needed_l234_234395

theorem total_computers_needed
    (initial_students : ℕ)
    (students_per_computer : ℕ)
    (additional_students : ℕ)
    (initial_computers : ℕ := initial_students / students_per_computer)
    (total_computers : ℕ := initial_computers + (additional_students / students_per_computer))
    (h1 : initial_students = 82)
    (h2 : students_per_computer = 2)
    (h3 : additional_students = 16) :
    total_computers = 49 :=
by
  -- The proof would normally go here
  sorry

end total_computers_needed_l234_234395


namespace tan_alpha_minus_beta_alpha_plus_beta_l234_234001

variable (α β : ℝ)

-- Conditions as hypotheses
axiom tan_alpha : Real.tan α = 2
axiom tan_beta : Real.tan β = -1 / 3
axiom alpha_range : 0 < α ∧ α < Real.pi / 2
axiom beta_range : Real.pi / 2 < β ∧ β < Real.pi

-- Proof statements
theorem tan_alpha_minus_beta : Real.tan (α - β) = 7 := by
  sorry

theorem alpha_plus_beta : α + β = 5 * Real.pi / 4 := by
  sorry

end tan_alpha_minus_beta_alpha_plus_beta_l234_234001


namespace dr_jones_remaining_salary_l234_234756

theorem dr_jones_remaining_salary:
  let salary := 6000
  let house_rental := 640
  let food_expense := 380
  let electric_water_bill := (1/4) * salary
  let insurances := (1/5) * salary
  let taxes := (10/100) * salary
  let transportation := (3/100) * salary
  let emergency_costs := (2/100) * salary
  let total_expenses := house_rental + food_expense + electric_water_bill + insurances + taxes + transportation + emergency_costs
  let remaining_salary := salary - total_expenses
  remaining_salary = 1380 :=
by
  sorry

end dr_jones_remaining_salary_l234_234756


namespace sum_of_special_integers_l234_234889

theorem sum_of_special_integers :
  let a := 0
  let b := 1
  let c := -1
  a + b + c = 0 := by
  sorry

end sum_of_special_integers_l234_234889


namespace sandy_shopping_l234_234151

variable (X : ℝ)

theorem sandy_shopping (h : 0.70 * X = 210) : X = 300 := by
  sorry

end sandy_shopping_l234_234151


namespace no_n_geq_2_makes_10101n_prime_l234_234482

theorem no_n_geq_2_makes_10101n_prime : ∀ n : ℕ, n ≥ 2 → ¬ Prime (n^4 + n^2 + 1) :=
by
  sorry

end no_n_geq_2_makes_10101n_prime_l234_234482


namespace division_problem_l234_234246

theorem division_problem (n : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) 
  (h_div : divisor = 12) (h_quo : quotient = 9) (h_rem : remainder = 1) 
  (h_eq: n = divisor * quotient + remainder) : n = 109 :=
by
  sorry

end division_problem_l234_234246


namespace geometric_sequence_sum_l234_234150

theorem geometric_sequence_sum (S : ℕ → ℚ) (a : ℕ → ℚ)
  (h1 : S 4 = 1)
  (h2 : S 8 = 3)
  (h3 : ∀ n, S (n + 4) - S n = a (n + 1) + a (n + 2) + a (n + 3) + a (n + 4)) :
  a 17 + a 18 + a 19 + a 20 = 16 :=
by
  -- Insert your proof here.
  sorry

end geometric_sequence_sum_l234_234150


namespace fettuccine_to_penne_ratio_l234_234239

theorem fettuccine_to_penne_ratio
  (num_surveyed : ℕ)
  (num_spaghetti : ℕ)
  (num_ravioli : ℕ)
  (num_fettuccine : ℕ)
  (num_penne : ℕ)
  (h_surveyed : num_surveyed = 800)
  (h_spaghetti : num_spaghetti = 300)
  (h_ravioli : num_ravioli = 200)
  (h_fettuccine : num_fettuccine = 150)
  (h_penne : num_penne = 150) :
  num_fettuccine / num_penne = 1 :=
by
  sorry

end fettuccine_to_penne_ratio_l234_234239


namespace points_player_1_after_13_rotations_l234_234944

variable (table : List ℕ) (players : Fin 16 → ℕ)

axiom round_rotating_table : table = [0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1]
axiom points_player_5 : players 5 = 72
axiom points_player_9 : players 9 = 84

theorem points_player_1_after_13_rotations : players 1 = 20 := 
  sorry

end points_player_1_after_13_rotations_l234_234944


namespace p_plus_q_eq_10_l234_234915

theorem p_plus_q_eq_10 (p q : ℕ) (hp : p > q) (hpq1 : p < 10) (hpq2 : q < 10)
  (h : p.factorial / q.factorial = 840) : p + q = 10 :=
by
  sorry

end p_plus_q_eq_10_l234_234915


namespace unique_solution_l234_234208

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n
def is_odd (n : ℕ) : Prop := n % 2 = 1

theorem unique_solution (x y : ℕ) :
  is_prime x →
  is_odd y →
  x^2 + y = 2007 →
  (x = 2 ∧ y = 2003) :=
by
  sorry

end unique_solution_l234_234208


namespace smallest_positive_integer_divisible_12_15_16_exists_l234_234410

theorem smallest_positive_integer_divisible_12_15_16_exists :
  ∃ x : ℕ, x > 0 ∧ 12 ∣ x ∧ 15 ∣ x ∧ 16 ∣ x ∧ x = 240 :=
by sorry

end smallest_positive_integer_divisible_12_15_16_exists_l234_234410


namespace rectangle_circles_l234_234904

theorem rectangle_circles (p q : Prop) (hp : p) (hq : ¬ q) : p ∨ q :=
by sorry

end rectangle_circles_l234_234904


namespace probability_red_card_top_l234_234097

def num_red_cards : ℕ := 26
def total_cards : ℕ := 52
def prob_red_card_top : ℚ := num_red_cards / total_cards

theorem probability_red_card_top : prob_red_card_top = (1 / 2) := by
  sorry

end probability_red_card_top_l234_234097


namespace projectile_reaches_height_at_first_l234_234372

noncomputable def reach_height (t : ℝ) : ℝ :=
-16 * t^2 + 80 * t

theorem projectile_reaches_height_at_first (t : ℝ) :
  reach_height t = 36 → t = 0.5 :=
by
  -- The proof can be provided here
  sorry

end projectile_reaches_height_at_first_l234_234372


namespace rectangle_width_l234_234909

theorem rectangle_width (L W : ℝ) 
  (h1 : L * W = 300)
  (h2 : 2 * L + 2 * W = 70) : 
  W = 15 :=
by 
  -- We prove the width W of the rectangle is 15 meters.
  sorry

end rectangle_width_l234_234909


namespace common_term_sequence_7n_l234_234894

theorem common_term_sequence_7n (n : ℕ) : 
  ∃ a_n : ℕ, a_n = (7 / 9) * (10^n - 1) :=
by
  sorry

end common_term_sequence_7n_l234_234894


namespace find_number_l234_234095

def x : ℝ := 33.75

theorem find_number (x: ℝ) :
  (0.30 * x = 0.25 * 45) → x = 33.75 :=
by
  sorry

end find_number_l234_234095


namespace number_of_owls_joined_l234_234445

-- Define the initial condition
def initial_owls : ℕ := 3

-- Define the current condition
def current_owls : ℕ := 5

-- Define the problem statement as a theorem
theorem number_of_owls_joined : (current_owls - initial_owls) = 2 :=
by
  sorry

end number_of_owls_joined_l234_234445


namespace cara_meets_don_distance_l234_234092

theorem cara_meets_don_distance (distance total_distance : ℝ) (cara_speed don_speed : ℝ) (delay : ℝ) 
  (h_total_distance : total_distance = 45)
  (h_cara_speed : cara_speed = 6)
  (h_don_speed : don_speed = 5)
  (h_delay : delay = 2) :
  distance = 30 :=
by
  have h := 1 / total_distance
  have : cara_speed * (distance / cara_speed) + don_speed * (distance / cara_speed - delay) = 45 := sorry
  exact sorry

end cara_meets_don_distance_l234_234092


namespace ratio_of_triangles_in_octagon_l234_234774

-- Conditions
def regular_octagon_division : Prop := 
  let L := 1 -- Area of each small congruent right triangle
  let ABJ := 2 * L -- Area of triangle ABJ
  let ADE := 6 * L -- Area of triangle ADE
  (ABJ / ADE = (1:ℝ) / 3)

-- Statement
theorem ratio_of_triangles_in_octagon : regular_octagon_division := by
  sorry

end ratio_of_triangles_in_octagon_l234_234774


namespace dan_remaining_marbles_l234_234303

-- Define the initial number of marbles Dan has
def initial_marbles : ℕ := 64

-- Define the number of marbles Dan gave to Mary
def marbles_given : ℕ := 14

-- Define the number of remaining marbles
def remaining_marbles : ℕ := initial_marbles - marbles_given

-- State the theorem
theorem dan_remaining_marbles : remaining_marbles = 50 := by
  -- Placeholder for the proof
  sorry

end dan_remaining_marbles_l234_234303


namespace sum_of_values_satisfying_equation_l234_234405

noncomputable def sum_of_roots_of_quadratic (a b c : ℝ) : ℝ := -b / a

theorem sum_of_values_satisfying_equation :
  (∃ x : ℝ, (x^2 - 5 * x + 7 = 9)) →
  sum_of_roots_of_quadratic 1 (-5) (-2) = 5 :=
by
  sorry

end sum_of_values_satisfying_equation_l234_234405


namespace average_math_test_score_l234_234039

theorem average_math_test_score :
    let june_score := 97
    let patty_score := 85
    let josh_score := 100
    let henry_score := 94
    let num_children := 4
    let total_score := june_score + patty_score + josh_score + henry_score
    total_score / num_children = 94 := by
  sorry

end average_math_test_score_l234_234039


namespace filling_time_with_ab_l234_234865

theorem filling_time_with_ab (a b c l : ℝ) (h1 : a + b + c - l = 5 / 6) (h2 : a + c - l = 1 / 2) (h3 : b + c - l = 1 / 3) : 
  1 / (a + b) = 1.2 :=
by
  sorry

end filling_time_with_ab_l234_234865


namespace square_field_area_l234_234746

theorem square_field_area (x : ℕ) 
    (hx : 4 * x - 2 = 666) : x^2 = 27889 := by
  -- We would solve for x using the given equation.
  sorry

end square_field_area_l234_234746


namespace sequence_converges_to_zero_and_N_for_epsilon_l234_234477

theorem sequence_converges_to_zero_and_N_for_epsilon :
  (∀ ε > 0, ∃ N : ℕ, ∀ n > N, |1 / (n : ℝ) - 0| < ε) ∧ 
  (∃ N : ℕ, ∀ n > N, |1 / (n : ℝ)| < 0.001) :=
by
  sorry

end sequence_converges_to_zero_and_N_for_epsilon_l234_234477


namespace brians_gas_usage_l234_234544

theorem brians_gas_usage (miles_per_gallon : ℕ) (miles_traveled : ℕ) (gallons_used : ℕ) 
  (h1 : miles_per_gallon = 20) 
  (h2 : miles_traveled = 60) 
  (h3 : gallons_used = miles_traveled / miles_per_gallon) : 
  gallons_used = 3 := 
by 
  rw [h1, h2] at h3 
  exact h3

end brians_gas_usage_l234_234544


namespace area_of_triangle_KBC_l234_234130

noncomputable def length_FE := 7
noncomputable def length_BC := 7
noncomputable def length_JB := 5
noncomputable def length_BK := 5

theorem area_of_triangle_KBC : (1 / 2 : ℝ) * length_BC * length_BK = 17.5 := by
  -- conditions: 
  -- 1. Hexagon ABCDEF is equilateral with each side of length s.
  -- 2. Squares ABJI and FEHG are formed outside the hexagon with areas 25 and 49 respectively.
  -- 3. Triangle JBK is equilateral.
  -- 4. FE = BC.
  sorry

end area_of_triangle_KBC_l234_234130


namespace fewer_gallons_for_plants_correct_l234_234699

-- Define the initial conditions
def initial_water : ℕ := 65
def water_per_car : ℕ := 7
def total_cars : ℕ := 2
def water_for_cars : ℕ := water_per_car * total_cars
def water_remaining_after_cars : ℕ := initial_water - water_for_cars
def water_for_plates_clothes : ℕ := 24
def water_remaining_before_plates_clothes : ℕ := water_for_plates_clothes * 2
def water_for_plants : ℕ := water_remaining_after_cars - water_remaining_before_plates_clothes

-- Define the query statement
def fewer_gallons_for_plants : Prop := water_per_car - water_for_plants = 4

-- Proof skeleton
theorem fewer_gallons_for_plants_correct : fewer_gallons_for_plants :=
by sorry

end fewer_gallons_for_plants_correct_l234_234699


namespace triangle_inequality_proof_l234_234265

theorem triangle_inequality_proof (a b c : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) :
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 :=
sorry

end triangle_inequality_proof_l234_234265


namespace smaller_square_area_percentage_is_zero_l234_234776

noncomputable def area_smaller_square_percentage (r : ℝ) : ℝ :=
  let side_length_larger_square := 2 * r
  let x := 0  -- Solution from the Pythagorean step
  let area_larger_square := side_length_larger_square ^ 2
  let area_smaller_square := x ^ 2
  100 * area_smaller_square / area_larger_square

theorem smaller_square_area_percentage_is_zero (r : ℝ) :
    area_smaller_square_percentage r = 0 :=
  sorry

end smaller_square_area_percentage_is_zero_l234_234776


namespace find_m_p_pairs_l234_234368

theorem find_m_p_pairs (m p : ℕ) (h_prime : Nat.Prime p) (h_eq : ∃ (x : ℕ), 2^m * p^2 + 27 = x^3) :
  (m, p) = (1, 7) :=
sorry

end find_m_p_pairs_l234_234368


namespace sequence_property_l234_234471

theorem sequence_property (a : ℕ → ℝ) (h1 : ∀ n : ℕ, 1 ≤ n ∧ n ≤ 98 → a n - 2022 * a (n+1) + 2021 * a (n+2) ≥ 0)
  (h2 : a 99 - 2022 * a 100 + 2021 * a 1 ≥ 0)
  (h3 : a 100 - 2022 * a 1 + 2021 * a 2 ≥ 0)
  (h4 : a 10 = 10) :
  a 22 = 10 := 
sorry

end sequence_property_l234_234471


namespace negation_proposition_false_l234_234057

variable {R : Type} [LinearOrderedField R]

theorem negation_proposition_false (x y : R) :
  ¬ (x > 2 ∧ y > 3 → x + y > 5) = false := by
sorry

end negation_proposition_false_l234_234057


namespace minimum_toothpicks_for_5_squares_l234_234920

theorem minimum_toothpicks_for_5_squares :
  let single_square_toothpicks := 4
  let additional_shared_side_toothpicks := 3
  ∃ n, n = single_square_toothpicks + 4 * additional_shared_side_toothpicks ∧ n = 15 :=
by
  sorry

end minimum_toothpicks_for_5_squares_l234_234920


namespace inequality_a_b_l234_234744

theorem inequality_a_b (a b : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) :
    a / (b + 1) + b / (a + 1) ≤ 1 :=
  sorry

end inequality_a_b_l234_234744


namespace megatek_manufacturing_percentage_l234_234004

theorem megatek_manufacturing_percentage 
  (total_degrees : ℝ := 360)
  (manufacturing_degrees : ℝ := 18)
  (is_proportional : (manufacturing_degrees / total_degrees) * 100 = 5) :
  (manufacturing_degrees / total_degrees) * 100 = 5 := 
  by
  exact is_proportional

end megatek_manufacturing_percentage_l234_234004


namespace triangles_formed_l234_234566

-- Define the combinatorial function for binomial coefficients.
def binom (n k : ℕ) : ℕ :=
  if h : k ≤ n then Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k)) else 0

-- Given conditions
def points_on_first_line := 6
def points_on_second_line := 8

-- Number of triangles calculation
def total_triangles :=
  binom points_on_first_line 2 * binom points_on_second_line 1 +
  binom points_on_first_line 1 * binom points_on_second_line 2

-- The final theorem to prove
theorem triangles_formed : total_triangles = 288 :=
by
  sorry

end triangles_formed_l234_234566


namespace store_loss_l234_234896

noncomputable def calculation (x y : ℕ) : ℤ :=
  let revenue : ℕ := 60 * 2
  let cost : ℕ := x + y
  revenue - cost

theorem store_loss (x y : ℕ) (hx : (60 - x) * 2 = x) (hy : (y - 60) * 2 = y) :
  calculation x y = -40 := by
    sorry

end store_loss_l234_234896


namespace sufficient_but_not_necessary_l234_234281

theorem sufficient_but_not_necessary (a : ℝ) : a = 1 → |a| = 1 ∧ (|a| = 1 → a = 1 → false) :=
by
  sorry

end sufficient_but_not_necessary_l234_234281


namespace dividend_50100_l234_234536

theorem dividend_50100 (D Q R : ℕ) (h1 : D = 20 * Q) (h2 : D = 10 * R) (h3 : R = 100) : 
    D * Q + R = 50100 := by
  sorry

end dividend_50100_l234_234536


namespace rachel_picture_books_shelves_l234_234112

theorem rachel_picture_books_shelves (mystery_shelves : ℕ) (books_per_shelf : ℕ) (total_books : ℕ) 
  (h1 : mystery_shelves = 6) 
  (h2 : books_per_shelf = 9) 
  (h3 : total_books = 72) : 
  (total_books - mystery_shelves * books_per_shelf) / books_per_shelf = 2 :=
by sorry

end rachel_picture_books_shelves_l234_234112


namespace contradiction_proof_l234_234267

theorem contradiction_proof (a b : ℕ) (h : a + b ≥ 3) : (a ≥ 2) ∨ (b ≥ 2) :=
sorry

end contradiction_proof_l234_234267


namespace Emily_used_10_dimes_l234_234509

theorem Emily_used_10_dimes
  (p n d : ℕ)
  (h1 : p + n + d = 50)
  (h2 : p + 5 * n + 10 * d = 200) :
  d = 10 := by
  sorry

end Emily_used_10_dimes_l234_234509


namespace connie_initial_marbles_l234_234176

theorem connie_initial_marbles (marbles_given : ℕ) (marbles_left : ℕ) (initial_marbles : ℕ) 
    (h1 : marbles_given = 183) (h2 : marbles_left = 593) : initial_marbles = 776 :=
by
  sorry

end connie_initial_marbles_l234_234176


namespace shares_of_valuable_stock_l234_234274

theorem shares_of_valuable_stock 
  (price_val : ℕ := 78)
  (price_oth : ℕ := 39)
  (shares_oth : ℕ := 26)
  (total_asset : ℕ := 2106)
  (x : ℕ) 
  (h_val_stock : total_asset = 78 * x + 39 * 26) : 
  x = 14 :=
by
  sorry

end shares_of_valuable_stock_l234_234274


namespace candy_in_each_bag_l234_234956

theorem candy_in_each_bag (total_candy : ℕ) (bags : ℕ) (h1 : total_candy = 16) (h2 : bags = 2) : total_candy / bags = 8 :=
by {
    sorry
}

end candy_in_each_bag_l234_234956


namespace change_in_expression_is_correct_l234_234974

def change_in_expression (x a : ℝ) : ℝ :=
  if increases : true then (x + a)^2 - 3 - (x^2 - 3)
  else (x - a)^2 - 3 - (x^2 - 3)

theorem change_in_expression_is_correct (x a : ℝ) :
  a > 0 → change_in_expression x a = 2 * a * x + a^2 ∨ change_in_expression x a = -(2 * a * x) + a^2 :=
by
  sorry

end change_in_expression_is_correct_l234_234974


namespace apples_total_l234_234952

theorem apples_total
    (cecile_apples : ℕ := 15)
    (diane_apples_more : ℕ := 20) :
    (cecile_apples + (cecile_apples + diane_apples_more)) = 50 :=
by
  sorry

end apples_total_l234_234952


namespace money_leftover_is_90_l234_234362

-- Define constants and given conditions.
def jars_quarters : ℕ := 4
def quarters_per_jar : ℕ := 160
def jars_dimes : ℕ := 4
def dimes_per_jar : ℕ := 300
def jars_nickels : ℕ := 2
def nickels_per_jar : ℕ := 500

def value_per_quarter : ℝ := 0.25
def value_per_dime : ℝ := 0.10
def value_per_nickel : ℝ := 0.05

def bike_cost : ℝ := 240
def total_quarters := jars_quarters * quarters_per_jar
def total_dimes := jars_dimes * dimes_per_jar
def total_nickels := jars_nickels * nickels_per_jar

-- Calculate the total money Jenn has in quarters, dimes, and nickels.
def total_value_quarters : ℝ := total_quarters * value_per_quarter
def total_value_dimes : ℝ := total_dimes * value_per_dime
def total_value_nickels : ℝ := total_nickels * value_per_nickel

def total_money : ℝ := total_value_quarters + total_value_dimes + total_value_nickels

-- Calculate the money left after buying the bike.
def money_left : ℝ := total_money - bike_cost

-- Prove that the amount of money left is precisely $90.
theorem money_leftover_is_90 : money_left = 90 :=
by
  -- Placeholder for the proof
  sorry

end money_leftover_is_90_l234_234362


namespace final_l234_234435

noncomputable def f (x : ℝ) : ℝ :=
  if h : x ∈ [-3, -2] then 4 * x
  else sorry

lemma f_periodic (h : ∀ x : ℝ, f (x + 3) = - (1 / f x)) :
 ∀ x : ℝ, f (x + 6) = f x :=
sorry

lemma f_even (h : ∀ x : ℝ, f x = f (-x)) : ℕ := sorry

theorem final (h1 : ∀ x : ℝ, f (x + 3) = - (1 / f x))
  (h2 : ∀ x : ℝ, f x = f (-x))
  (h3 : ∀ x : ℝ, x ∈ [-3, -2] → f x = 4 * x) :
  f 107.5 = 1 / 10 :=
sorry

end final_l234_234435


namespace sum_of_squares_of_roots_l234_234778

theorem sum_of_squares_of_roots :
  let a := 1
  let b := 8
  let c := -12
  let r1_r2_sum := -(b:ℝ) / a
  let r1_r2_product := (c:ℝ) / a
  (r1_r2_sum) ^ 2 - 2 * r1_r2_product = 88 :=
by
  sorry

end sum_of_squares_of_roots_l234_234778


namespace machine_working_time_l234_234980

theorem machine_working_time (total_shirts_made : ℕ) (shirts_per_minute : ℕ)
  (h1 : total_shirts_made = 196) (h2 : shirts_per_minute = 7) :
  (total_shirts_made / shirts_per_minute = 28) :=
by
  sorry

end machine_working_time_l234_234980


namespace all_equal_l234_234218

theorem all_equal (a : Fin 100 → ℝ) 
  (h1 : a 0 - 3 * a 1 + 2 * a 2 ≥ 0)
  (h2 : a 1 - 3 * a 2 + 2 * a 3 ≥ 0)
  (h3 : a 2 - 3 * a 3 + 2 * a 4 ≥ 0)
  -- ...
  (h99: a 98 - 3 * a 99 + 2 * a 0 ≥ 0)
  (h100: a 99 - 3 * a 0 + 2 * a 1 ≥ 0) : 
    ∀ i : Fin 100, a i = a 0 := 
by 
  sorry

end all_equal_l234_234218


namespace seating_arrangements_exactly_two_adjacent_empty_l234_234872

theorem seating_arrangements_exactly_two_adjacent_empty :
  let seats := 6
  let people := 3
  let arrangements := (seats.factorial / (seats - people).factorial)
  let non_adj_non_empty := ((seats - people).choose people * people.factorial)
  let all_adj_empty := ((seats - (people + 1)).choose 1 * people.factorial)
  arrangements - non_adj_non_empty - all_adj_empty = 72 := by
  sorry

end seating_arrangements_exactly_two_adjacent_empty_l234_234872


namespace find_a_plus_b_l234_234858

theorem find_a_plus_b (x a b : ℝ) (ha : a > 0) (hb : b > 0) (h : x = a + Real.sqrt b) 
  (hx : x^2 + 3 * x + ↑(3) / x + 1 / x^2 = 30) : 
  a + b = 5 := 
sorry

end find_a_plus_b_l234_234858


namespace work_rate_b_l234_234400

theorem work_rate_b (W : ℝ) (A B C : ℝ) :
  (A = W / 11) → 
  (C = W / 55) →
  (8 * A + 4 * B + 4 * C = W) →
  B = W / (2420 / 341) :=
by
  intros hA hC hWork
  -- We start with the given assumptions and work towards showing B = W / (2420 / 341)
  sorry

end work_rate_b_l234_234400


namespace find_number_l234_234047

theorem find_number (x : ℝ) (h : x + 33 + 333 + 33.3 = 399.6) : x = 0.3 :=
by
  sorry

end find_number_l234_234047


namespace g_neg_one_l234_234856

variables (f : ℝ → ℝ) (g : ℝ → ℝ)
variables (h₀ : ∀ x : ℝ, f (-x) + x^2 = -(f x + x^2))
variables (h₁ : f 1 = 1)
variables (h₂ : ∀ x : ℝ, g x = f x + 2)

theorem g_neg_one : g (-1) = -1 :=
by
  sorry

end g_neg_one_l234_234856


namespace place_value_accuracy_l234_234547

theorem place_value_accuracy (x : ℝ) (h : x = 3.20 * 10000) :
  ∃ p : ℕ, p = 100 ∧ (∃ k : ℤ, x / p = k) := by
  sorry

end place_value_accuracy_l234_234547


namespace asparagus_cost_correct_l234_234242

def cost_asparagus (total_start: Int) (total_left: Int) (cost_bananas: Int) (cost_pears: Int) (cost_chicken: Int) : Int := 
  total_start - total_left - cost_bananas - cost_pears - cost_chicken

theorem asparagus_cost_correct :
  cost_asparagus 55 28 8 2 11 = 6 :=
by
  sorry

end asparagus_cost_correct_l234_234242


namespace numerator_of_fraction_l234_234419

theorem numerator_of_fraction (x : ℤ) (h : (x : ℚ) / (4 * x - 5) = 3 / 7) : x = 3 := 
sorry

end numerator_of_fraction_l234_234419


namespace volume_proportionality_l234_234341

variable (W V : ℕ)
variable (k : ℚ)

-- Given conditions
theorem volume_proportionality (h1 : V = k * W) (h2 : W = 112) (h3 : k = 3 / 7) :
  V = 48 := by
  sorry

end volume_proportionality_l234_234341


namespace remainder_8_times_10_pow_18_plus_1_pow_18_div_9_l234_234587

theorem remainder_8_times_10_pow_18_plus_1_pow_18_div_9 :
  (8 * 10^18 + 1^18) % 9 = 0 := 
by 
  sorry

end remainder_8_times_10_pow_18_plus_1_pow_18_div_9_l234_234587


namespace sam_travel_time_l234_234080

theorem sam_travel_time (d_AC d_CB : ℕ) (v_sam : ℕ) 
  (h1 : d_AC = 600) (h2 : d_CB = 400) (h3 : v_sam = 50) : 
  (d_AC + d_CB) / v_sam = 20 := 
by
  sorry

end sam_travel_time_l234_234080


namespace sum_of_four_integers_l234_234642

noncomputable def originalSum (a b c d : ℤ) :=
  (a + b + c + d)

theorem sum_of_four_integers
  (a b c d : ℤ)
  (h1 : (a + b + c) / 3 + d = 8)
  (h2 : (a + b + d) / 3 + c = 12)
  (h3 : (a + c + d) / 3 + b = 32 / 3)
  (h4 : (b + c + d) / 3 + a = 28 / 3) :
  originalSum a b c d = 30 :=
sorry

end sum_of_four_integers_l234_234642


namespace find_angle_A_l234_234216

theorem find_angle_A (a b c : ℝ) (A B C : ℝ) 
  (h1 : (Real.sin A + Real.sin B) * (a - b) = (Real.sin C - Real.sin B) * c) :
  A = Real.pi / 3 :=
sorry

end find_angle_A_l234_234216


namespace limit_series_product_eq_l234_234757

variable (a r s : ℝ)

noncomputable def series_product_sum_limit : ℝ :=
∑' n : ℕ, (a * r^n) * (a * s^n)

theorem limit_series_product_eq :
  |r| < 1 → |s| < 1 → series_product_sum_limit a r s = a^2 / (1 - r * s) :=
by
  intro hr hs
  sorry

end limit_series_product_eq_l234_234757


namespace percentage_cut_away_in_second_week_l234_234775

theorem percentage_cut_away_in_second_week :
  ∃(x : ℝ), (x / 100) * 142.5 * 0.9 = 109.0125 ∧ x = 15 :=
by
  sorry

end percentage_cut_away_in_second_week_l234_234775


namespace distribute_places_l234_234655

open Nat

theorem distribute_places (places schools : ℕ) (h_places : places = 7) (h_schools : schools = 3) : 
  ∃ n : ℕ, n = (Nat.choose (places - 1) (schools - 1)) ∧ n = 15 :=
by
  rw [h_places, h_schools]
  use 15
  , sorry

end distribute_places_l234_234655


namespace partition_count_l234_234818

noncomputable def count_partition (n : ℕ) : ℕ :=
  -- Function that counts the number of ways to partition n as per the given conditions
  n

theorem partition_count (n : ℕ) (h : n > 0) :
  count_partition n = n :=
sorry

end partition_count_l234_234818


namespace det_example_l234_234058

theorem det_example : (1 * 4 - 2 * 3) = -2 :=
by
  -- Skip the proof with sorry
  sorry

end det_example_l234_234058


namespace tangent_line_circle_l234_234921

theorem tangent_line_circle (m : ℝ) (h : m > 0) : 
  (∀ x y : ℝ, x + y = 2 ↔ x^2 + y^2 = m) → m = 2 :=
by
  intro h_tangent
  sorry

end tangent_line_circle_l234_234921


namespace cube_volume_is_27_l234_234727

theorem cube_volume_is_27 
    (a : ℕ) 
    (Vol_cube : ℕ := a^3)
    (Vol_new : ℕ := (a - 2) * a * (a + 2))
    (h : Vol_new + 12 = Vol_cube) : Vol_cube = 27 :=
by
    sorry

end cube_volume_is_27_l234_234727


namespace lemonade_second_intermission_l234_234090

theorem lemonade_second_intermission (first_intermission third_intermission total_lemonade second_intermission : ℝ) 
  (h1 : first_intermission = 0.25) 
  (h2 : third_intermission = 0.25) 
  (h3 : total_lemonade = 0.92) 
  (h4 : second_intermission = total_lemonade - (first_intermission + third_intermission)) : 
  second_intermission = 0.42 := 
by 
  sorry

end lemonade_second_intermission_l234_234090


namespace compound_weight_l234_234351

noncomputable def weightB : ℝ := 275
noncomputable def ratioAtoB : ℝ := 2 / 10

theorem compound_weight (weightA weightB total_weight : ℝ) 
  (h1 : ratioAtoB = 2 / 10) 
  (h2 : weightB = 275) 
  (h3 : weightA = weightB * (2 / 10)) 
  (h4 : total_weight = weightA + weightB) : 
  total_weight = 330 := 
by sorry

end compound_weight_l234_234351


namespace nonneg_integer_representation_l234_234173

theorem nonneg_integer_representation (n : ℕ) : 
  ∃ x y : ℕ, n = (x + y) * (x + y) + 3 * x + y / 2 := 
sorry

end nonneg_integer_representation_l234_234173


namespace ferris_break_length_l234_234302

-- Definitions of the given conditions
def audrey_work_rate := 1 / 4  -- Audrey completes 1/4 of the job per hour
def ferris_work_rate := 1 / 3  -- Ferris completes 1/3 of the job per hour
def total_work_time := 2       -- They worked together for 2 hours
def num_breaks := 6            -- Ferris took 6 breaks during the work period

-- The theorem to prove the length of each break Ferris took
theorem ferris_break_length (break_length : ℝ) :
  (audrey_work_rate * total_work_time) + 
  (ferris_work_rate * (total_work_time - (break_length / 60) * num_breaks)) = 1 →
  break_length = 2.5 :=
by
  sorry

end ferris_break_length_l234_234302


namespace pieces_given_by_brother_l234_234985

-- Given conditions
def original_pieces : ℕ := 18
def total_pieces_now : ℕ := 62

-- The statement to prove
theorem pieces_given_by_brother : total_pieces_now - original_pieces = 44 := by
  -- Starting with the given conditions
  unfold original_pieces total_pieces_now
  -- Place to insert the proof
  sorry

end pieces_given_by_brother_l234_234985


namespace intersection_point_l234_234099

noncomputable def line1 (x : ℚ) : ℚ := 3 * x
noncomputable def line2 (x : ℚ) : ℚ := -9 * x - 6

theorem intersection_point : ∃ (x y : ℚ), line1 x = y ∧ line2 x = y ∧ x = -1/2 ∧ y = -3/2 :=
by
  -- skipping the actual proof steps
  sorry

end intersection_point_l234_234099


namespace kittens_remaining_l234_234157

theorem kittens_remaining (original_kittens : ℕ) (kittens_given_away : ℕ) 
  (h1 : original_kittens = 8) (h2 : kittens_given_away = 4) : 
  original_kittens - kittens_given_away = 4 := by
  sorry

end kittens_remaining_l234_234157


namespace time_saved_1200_miles_l234_234409

theorem time_saved_1200_miles
  (distance : ℕ)
  (speed1 speed2 : ℕ)
  (h_distance : distance = 1200)
  (h_speed1 : speed1 = 60)
  (h_speed2 : speed2 = 50) :
  (distance / speed2) - (distance / speed1) = 4 :=
by
  sorry

end time_saved_1200_miles_l234_234409


namespace prob_neq_zero_l234_234383

noncomputable def probability_no_one (a b c d : ℕ) : ℚ :=
  if 1 ≤ a ∧ a ≤ 6 ∧ 1 ≤ b ∧ b ≤ 6 ∧ 1 ≤ c ∧ c ≤ 6 ∧ 1 ≤ d ∧ d ≤ 6 
  then (5/6)^4 
  else 0

theorem prob_neq_zero (a b c d : ℕ) :
  (1 ≤ a) ∧ (a ≤ 6) ∧ (1 ≤ b) ∧ (b ≤ 6) ∧ (1 ≤ c) ∧ (c ≤ 6) ∧ (1 ≤ d) ∧ (d ≤ 6) →
  (a - 1) * (b - 1) * (c - 1) * (d - 1) ≠ 0 ↔ 
  probability_no_one a b c d = 625/1296 :=
by
  sorry

end prob_neq_zero_l234_234383


namespace max_whole_number_n_l234_234339

theorem max_whole_number_n (n : ℕ) : (1/2 + n/9 < 1) → n ≤ 4 :=
by
  sorry

end max_whole_number_n_l234_234339


namespace unique_non_zero_in_rows_and_cols_l234_234986

variable (n : ℕ) (A : Matrix (Fin n) (Fin n) ℝ)

theorem unique_non_zero_in_rows_and_cols
  (non_neg_A : ∀ i j, 0 ≤ A i j)
  (non_sing_A : Invertible A)
  (non_neg_A_inv : ∀ i j, 0 ≤ (A⁻¹) i j) :
  (∀ i, ∃! j, A i j ≠ 0) ∧ (∀ j, ∃! i, A i j ≠ 0) := by
  sorry

end unique_non_zero_in_rows_and_cols_l234_234986


namespace smallest_number_of_pets_l234_234769

noncomputable def smallest_common_multiple (a b c : Nat) : Nat :=
  Nat.lcm a (Nat.lcm b c)

theorem smallest_number_of_pets : smallest_common_multiple 3 15 9 = 45 :=
by
  sorry

end smallest_number_of_pets_l234_234769


namespace find_n_l234_234234

theorem find_n (n : ℕ) (h : n * n.factorial + 2 * n.factorial = 5040) : n = 5 :=
by {
  sorry
}

end find_n_l234_234234


namespace cosine_difference_l234_234480

theorem cosine_difference (A B : ℝ) (h1 : Real.sin A + Real.sin B = 3/2) (h2 : Real.cos A + Real.cos B = 2) :
  Real.cos (A - B) = 17 / 8 :=
by
  sorry

end cosine_difference_l234_234480


namespace sum_of_uv_l234_234781

theorem sum_of_uv (u v : ℕ) (hu : 0 < u) (hv : 0 < v) (hv_lt_hu : v < u)
  (area_pent : 6 * u * v = 500) : u + v = 19 :=
by
  sorry

end sum_of_uv_l234_234781


namespace land_area_in_acres_l234_234453

-- Define the conditions given in the problem.
def length_cm : ℕ := 30
def width_cm : ℕ := 20
def scale_cm_to_mile : ℕ := 1  -- 1 cm corresponds to 1 mile.
def sq_mile_to_acres : ℕ := 640  -- 1 square mile corresponds to 640 acres.

-- Define the statement to be proved.
theorem land_area_in_acres :
  (length_cm * width_cm * sq_mile_to_acres) = 384000 := 
  by sorry

end land_area_in_acres_l234_234453


namespace Corey_found_golf_balls_on_Saturday_l234_234098

def goal : ℕ := 48
def golf_balls_found_on_sunday : ℕ := 18
def golf_balls_needed : ℕ := 14
def golf_balls_found_on_saturday : ℕ := 16

theorem Corey_found_golf_balls_on_Saturday :
  (goal - golf_balls_found_on_sunday - golf_balls_needed) = golf_balls_found_on_saturday := 
by
  sorry

end Corey_found_golf_balls_on_Saturday_l234_234098


namespace find_a_l234_234717

noncomputable def curve (a : ℝ) (x : ℝ) : ℝ :=
  (x + a) * Real.log x

noncomputable def curve_deriv (a : ℝ) (x : ℝ) : ℝ :=
  Real.log x + (x + a) / x

theorem find_a (a : ℝ) (h : curve (x := 1) a = 2) : a = 1 :=
by
  have eq1 : curve 1 0 = (1 + a) * 0 := by sorry
  have eq2 : curve 1 1 = (1 + a) * Real.log 1 := by sorry
  have eq3 : curve_deriv a 1 = Real.log 1 + (1 + a) / 1 := by sorry
  have eq4 : 2 = 1 + a := by sorry
  sorry -- Complete proof would follow here

end find_a_l234_234717


namespace a8_equals_two_or_minus_two_l234_234545

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n m : ℕ, ∃ r : ℝ, a (n + m) = a n * a m / a 0

theorem a8_equals_two_or_minus_two (a : ℕ → ℝ) 
    (h_geom : geometric_sequence a)
    (h_roots : ∃ x y : ℝ, x^2 - 8 * x + 4 = 0 ∧ y^2 - 8 * y + 4 = 0 ∧ a 6 = x ∧ a 10 = y) :
  a 8 = 2 ∨ a 8 = -2 :=
by
  sorry

end a8_equals_two_or_minus_two_l234_234545


namespace max_value_is_27_l234_234169

noncomputable def max_value_of_expression (a b c : ℝ) : ℝ :=
  (a - b)^2 + (b - c)^2 + (c - a)^2

theorem max_value_is_27 (a b c : ℝ)
  (h : a^2 + b^2 + c^2 = 9) : max_value_of_expression a b c = 27 :=
by
  sorry

end max_value_is_27_l234_234169


namespace maria_green_beans_l234_234528

theorem maria_green_beans
    (potatoes : ℕ)
    (carrots : ℕ)
    (onions : ℕ)
    (green_beans : ℕ)
    (h1 : potatoes = 2)
    (h2 : carrots = 6 * potatoes)
    (h3 : onions = 2 * carrots)
    (h4 : green_beans = onions / 3) :
  green_beans = 8 := 
sorry

end maria_green_beans_l234_234528


namespace six_digit_number_divisible_by_37_l234_234446

theorem six_digit_number_divisible_by_37 (a b : ℕ) (h1 : 100 ≤ a ∧ a < 1000) (h2 : 100 ≤ b ∧ b < 1000) (h3 : 37 ∣ (a + b)) : 37 ∣ (1000 * a + b) :=
sorry

end six_digit_number_divisible_by_37_l234_234446


namespace kilometers_to_chains_l234_234523

theorem kilometers_to_chains :
  (1 * 10 * 50 = 500) :=
by
  sorry

end kilometers_to_chains_l234_234523


namespace logan_list_count_l234_234579

theorem logan_list_count : 
    let smallest_square_multiple := 900
    let smallest_cube_multiple := 27000
    ∃ n, n = 871 ∧ 
        ∀ k, (k * 30 ≥ smallest_square_multiple ∧ k * 30 ≤ smallest_cube_multiple) ↔ (30 ≤ k ∧ k ≤ 900) :=
by
    let smallest_square_multiple := 900
    let smallest_cube_multiple := 27000
    use 871
    sorry

end logan_list_count_l234_234579


namespace total_students_l234_234895

/-- Definition of the problem's conditions as Lean statements -/
def left_col := 8
def right_col := 14
def front_row := 7
def back_row := 15

/-- The total number of columns calculated from Eunji's column positions -/
def total_columns := left_col + right_col - 1
/-- The total number of rows calculated from Eunji's row positions -/
def total_rows := front_row + back_row - 1

/-- Lean statement showing the total number of students given the conditions -/
theorem total_students : total_columns * total_rows = 441 := by
  sorry

end total_students_l234_234895


namespace variance_ξ_l234_234444

variable (P : ℕ → ℝ) (ξ : ℕ)

-- conditions
axiom P_0 : P 0 = 1 / 5
axiom P_1 : P 1 + P 2 = 4 / 5
axiom E_ξ : (0 * P 0 + 1 * P 1 + 2 * P 2) = 1

-- proof statement
theorem variance_ξ : (0 - 1)^2 * P 0 + (1 - 1)^2 * P 1 + (2 - 1)^2 * P 2 = 2 / 5 :=
by sorry

end variance_ξ_l234_234444


namespace pq_sum_l234_234812

open Real

section Problem
variables (p q : ℝ)
  (hp : p^3 - 21 * p^2 + 35 * p - 105 = 0)
  (hq : 5 * q^3 - 35 * q^2 - 175 * q + 1225 = 0)

theorem pq_sum : p + q = 21 / 2 :=
sorry
end Problem

end pq_sum_l234_234812


namespace highest_place_value_quotient_and_remainder_l234_234050

-- Conditions
def dividend := 438
def divisor := 4

-- Theorem stating that the highest place value of the quotient is the hundreds place, and the remainder is 2
theorem highest_place_value_quotient_and_remainder : 
  (dividend = divisor * (dividend / divisor) + (dividend % divisor)) ∧ 
  ((dividend / divisor) >= 100) ∧ 
  ((dividend % divisor) = 2) :=
by
  sorry

end highest_place_value_quotient_and_remainder_l234_234050


namespace robotics_club_non_participants_l234_234404

theorem robotics_club_non_participants (club_students electronics_students programming_students both_students : ℕ) 
  (h1 : club_students = 80) 
  (h2 : electronics_students = 45) 
  (h3 : programming_students = 50) 
  (h4 : both_students = 30) : 
  club_students - (electronics_students - both_students + programming_students - both_students + both_students) = 15 :=
by
  -- The proof would be here
  sorry

end robotics_club_non_participants_l234_234404


namespace factorization_x12_minus_729_l234_234353

theorem factorization_x12_minus_729 (x : ℝ) : 
  x^12 - 729 = (x^2 + 3) * (x^4 - 3 * x^2 + 9) * (x^3 - 3) * (x^3 + 3) :=
by sorry

end factorization_x12_minus_729_l234_234353


namespace bus_driver_total_hours_l234_234825

theorem bus_driver_total_hours
  (reg_rate : ℝ := 16)
  (ot_rate : ℝ := 28)
  (total_hours : ℝ)
  (total_compensation : ℝ := 920)
  (h : total_compensation = reg_rate * 40 + ot_rate * (total_hours - 40)) :
  total_hours = 50 := 
by 
  sorry

end bus_driver_total_hours_l234_234825


namespace value_of_f_neg_a_l234_234640

noncomputable def f (x : ℝ) : ℝ := x^3 + Real.sin x + 1

theorem value_of_f_neg_a (a : ℝ) (h : f a = 2) : f (-a) = 0 := by
  sorry

end value_of_f_neg_a_l234_234640


namespace quadrilateral_diagonal_areas_relation_l234_234375

-- Defining the areas of the four triangles and the quadrilateral
variables (A B C D Q : ℝ)

-- Stating the property to be proven
theorem quadrilateral_diagonal_areas_relation 
  (H1 : Q = A + B + C + D) :
  A * B * C * D = ((A + B) * (B + C) * (C + D) * (D + A))^2 / Q^4 :=
by sorry

end quadrilateral_diagonal_areas_relation_l234_234375


namespace tan_pi_seven_product_eq_sqrt_seven_l234_234972

theorem tan_pi_seven_product_eq_sqrt_seven :
  (Real.tan (Real.pi / 7)) * (Real.tan (2 * Real.pi / 7)) * (Real.tan (3 * Real.pi / 7)) = Real.sqrt 7 :=
by
  sorry

end tan_pi_seven_product_eq_sqrt_seven_l234_234972


namespace evaluate_expression_l234_234777

theorem evaluate_expression (x : ℝ) (h1 : x^3 + 2 ≠ 0) (h2 : x^3 - 2 ≠ 0) :
  (( (x+2)^3 * (x^2-x+2)^3 / (x^3+2)^3 )^3 * ( (x-2)^3 * (x^2+x+2)^3 / (x^3-2)^3 )^3 ) = 1 :=
by
  sorry

end evaluate_expression_l234_234777


namespace problem_l234_234364

-- Definitions
variables {a b : ℝ}
def is_root (p : ℝ → ℝ) (x : ℝ) : Prop := p x = 0

-- Root condition using the given equation
def quadratic_eq (x : ℝ) : ℝ := (x - 3) * (2 * x + 7) - (x^2 - 11 * x + 28)

-- Statement to prove
theorem problem (ha : is_root quadratic_eq a) (hb : is_root quadratic_eq b) (h_distinct : a ≠ b):
  (a + 2) * (b + 2) = -66 :=
sorry

end problem_l234_234364


namespace mr_castiel_sausages_l234_234015

theorem mr_castiel_sausages (S : ℕ) :
  S * (3 / 5) * (1 / 2) * (1 / 4) * (3 / 4) = 45 → S = 600 :=
by
  sorry

end mr_castiel_sausages_l234_234015


namespace intersection_A_B_l234_234243

open Set

def setA : Set ℕ := {x | x - 4 < 0}
def setB : Set ℕ := {0, 1, 3, 4}

theorem intersection_A_B : setA ∩ setB = {0, 1, 3} := by
  sorry

end intersection_A_B_l234_234243


namespace triangle_sides_inequality_l234_234278

theorem triangle_sides_inequality (a b c : ℝ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) :
  a^2 - 2 * a * b + b^2 - c^2 < 0 :=
by
  sorry

end triangle_sides_inequality_l234_234278


namespace marcy_drinks_in_250_minutes_l234_234269

-- Define a function to represent that Marcy takes n minutes to drink x liters of water.
def time_to_drink (minutes_per_sip : ℕ) (sip_volume_ml : ℕ) (total_volume_liters : ℕ) : ℕ :=
  let total_volume_ml := total_volume_liters * 1000
  let sips := total_volume_ml / sip_volume_ml
  sips * minutes_per_sip

theorem marcy_drinks_in_250_minutes :
  time_to_drink 5 40 2 = 250 :=
  by
    -- The function definition and its application will show this value holds.
    sorry

end marcy_drinks_in_250_minutes_l234_234269


namespace stratified_sampling_red_balls_l234_234711

theorem stratified_sampling_red_balls (total_balls red_balls sample_size : ℕ) (h_total : total_balls = 100) (h_red : red_balls = 20) (h_sample : sample_size = 10) :
  (sample_size * (red_balls / total_balls)) = 2 := by
  sorry

end stratified_sampling_red_balls_l234_234711


namespace estimate_sqrt_expression_l234_234226

theorem estimate_sqrt_expression :
  5 < 3 * Real.sqrt 5 - 1 ∧ 3 * Real.sqrt 5 - 1 < 6 :=
by
  sorry

end estimate_sqrt_expression_l234_234226


namespace fourth_power_mod_7_is_0_l234_234672

def fourth_smallest_prime := 7
def square_of_fourth_smallest_prime := fourth_smallest_prime ^ 2
def fourth_power_of_square := square_of_fourth_smallest_prime ^ 4

theorem fourth_power_mod_7_is_0 : 
  (fourth_power_of_square % 7) = 0 :=
by sorry

end fourth_power_mod_7_is_0_l234_234672


namespace find_p_l234_234137

theorem find_p (f p : ℂ) (w : ℂ) (h1 : f * p - w = 15000) (h2 : f = 8) (h3 : w = 10 + 200 * Complex.I) : 
  p = 1876.25 + 25 * Complex.I := 
sorry

end find_p_l234_234137


namespace half_hour_half_circle_half_hour_statement_is_true_l234_234195

-- Definitions based on conditions
def half_circle_divisions : ℕ := 30
def small_divisions_per_minute : ℕ := 1
def total_small_divisions : ℕ := 60
def minutes_per_circle : ℕ := 60

-- Relation of small divisions and time taken
def time_taken_for_small_divisions (divs : ℕ) : ℕ := divs * small_divisions_per_minute

-- Theorem to prove the statement
theorem half_hour_half_circle : time_taken_for_small_divisions half_circle_divisions = 30 :=
by
  -- Given half circle covers 30 small divisions
  -- Each small division represents 1 minute
  -- Therefore, time taken for 30 divisions should be 30 minutes
  exact rfl

-- The final statement proving the truth of the condition
theorem half_hour_statement_is_true : 
  (time_taken_for_small_divisions half_circle_divisions = 30) → True :=
by
  intro h
  trivial

end half_hour_half_circle_half_hour_statement_is_true_l234_234195


namespace math_problem_l234_234237

theorem math_problem : (3 ^ 456) + (9 ^ 5 / 9 ^ 3) = 82 := 
by 
  sorry

end math_problem_l234_234237


namespace students_in_circle_l234_234185

theorem students_in_circle (n : ℕ) (h1 : n > 6) (h2 : n > 16) (h3 : n / 2 = 10) : n + 2 = 22 := by
  sorry

end students_in_circle_l234_234185


namespace smallest_n_13n_congruent_456_mod_5_l234_234252

theorem smallest_n_13n_congruent_456_mod_5 : ∃ n : ℕ, (n > 0) ∧ (13 * n ≡ 456 [MOD 5]) ∧ (∀ m : ℕ, (m > 0 ∧ 13 * m ≡ 456 [MOD 5]) → n ≤ m) :=
by
  sorry

end smallest_n_13n_congruent_456_mod_5_l234_234252


namespace icosahedron_probability_div_by_three_at_least_one_fourth_l234_234857
open ProbabilityTheory

theorem icosahedron_probability_div_by_three_at_least_one_fourth (a b c : ℕ) (h : a + b + c = 20) :
  (a^3 + b^3 + c^3 + 6 * a * b * c : ℚ) / (a + b + c)^3 ≥ 1 / 4 :=
sorry

end icosahedron_probability_div_by_three_at_least_one_fourth_l234_234857


namespace value_of_N_l234_234370

theorem value_of_N : ∃ N : ℕ, (32^5 * 16^4 / 8^7) = 2^N ∧ N = 20 := by
  use 20
  sorry

end value_of_N_l234_234370


namespace perimeter_is_140_l234_234459

-- Definitions for conditions
def width (w : ℝ) := w
def length (w : ℝ) := width w + 10
def perimeter (w : ℝ) := 2 * (length w + width w)

-- Cost condition
def cost_condition (w : ℝ) : Prop := (perimeter w) * 6.5 = 910

-- Proving that if cost_condition holds, the perimeter is 140
theorem perimeter_is_140 (w : ℝ) (h : cost_condition w) : perimeter w = 140 :=
by sorry

end perimeter_is_140_l234_234459


namespace y_gets_per_rupee_l234_234800

theorem y_gets_per_rupee (a p : ℝ) (ha : a * p = 63) (htotal : p + a * p + 0.3 * p = 245) : a = 0.63 :=
by
  sorry

end y_gets_per_rupee_l234_234800


namespace initial_birds_correct_l234_234582

def flown_away : ℝ := 8.0
def left_on_fence : ℝ := 4.0
def initial_birds : ℝ := flown_away + left_on_fence

theorem initial_birds_correct : initial_birds = 12.0 := by
  sorry

end initial_birds_correct_l234_234582


namespace triangle_perimeter_l234_234103

theorem triangle_perimeter (MN NP MP : ℝ)
  (h1 : MN - NP = 18)
  (h2 : MP = 40)
  (h3 : MN / NP = 28 / 12) : 
  MN + NP + MP = 85 :=
by
  -- Proof is omitted
  sorry

end triangle_perimeter_l234_234103


namespace fraction_of_yellow_balls_l234_234179

theorem fraction_of_yellow_balls
  (total_balls : ℕ)
  (fraction_green : ℚ)
  (fraction_blue : ℚ)
  (number_blue : ℕ)
  (number_white : ℕ)
  (total_balls_eq : total_balls = number_blue * (1 / fraction_blue))
  (fraction_green_eq : fraction_green = 1 / 4)
  (fraction_blue_eq : fraction_blue = 1 / 8)
  (number_white_eq : number_white = 26)
  (number_blue_eq : number_blue = 6) :
  (total_balls - (total_balls * fraction_green + number_blue + number_white)) / total_balls = 1 / 12 :=
by
  sorry

end fraction_of_yellow_balls_l234_234179


namespace geometric_sequence_sufficient_not_necessary_l234_234616

theorem geometric_sequence_sufficient_not_necessary (a b c : ℝ) :
  (∃ r : ℝ, a = b * r ∧ b = c * r) → (b^2 = a * c) ∧ ¬ ( (b^2 = a * c) → (∃ r : ℝ, a = b * r ∧ b = c * r) ) :=
by
  sorry

end geometric_sequence_sufficient_not_necessary_l234_234616


namespace correct_option_D_l234_234261

variables {a b m : Type}
variables {α β : Type}

axiom parallel (x y : Type) : Prop
axiom perpendicular (x y : Type) : Prop

variables (a_parallel_b : parallel a b)
variables (a_parallel_alpha : parallel a α)

variables (alpha_perpendicular_beta : perpendicular α β)
variables (a_parallel_alpha : parallel a α)

variables (alpha_parallel_beta : parallel α β)
variables (m_perpendicular_alpha : perpendicular m α)

theorem correct_option_D : parallel α β ∧ perpendicular m α → perpendicular m β := sorry

end correct_option_D_l234_234261


namespace square_perimeter_eq_16_l234_234877

theorem square_perimeter_eq_16 (s : ℕ) (h : s^2 = 4 * s) : 4 * s = 16 :=
by {
  sorry
}

end square_perimeter_eq_16_l234_234877


namespace final_price_is_correct_l234_234804

def cost_cucumber : ℝ := 5
def cost_tomato : ℝ := cost_cucumber - 0.2 * cost_cucumber
def cost_bell_pepper : ℝ := cost_cucumber + 0.5 * cost_cucumber
def total_cost_before_discount : ℝ := 2 * cost_tomato + 3 * cost_cucumber + 4 * cost_bell_pepper
def final_price : ℝ := total_cost_before_discount - 0.1 * total_cost_before_discount

theorem final_price_is_correct : final_price = 47.7 := sorry

end final_price_is_correct_l234_234804


namespace books_in_special_collection_l234_234936

theorem books_in_special_collection (B : ℕ) :
  (∃ returned not_returned loaned_out_end  : ℝ, 
    loaned_out_end = 54 ∧ 
    returned = 0.65 * 60.00000000000001 ∧ 
    not_returned = 60.00000000000001 - returned ∧ 
    B = loaned_out_end + not_returned) → 
  B = 75 :=
by 
  intro h
  sorry

end books_in_special_collection_l234_234936


namespace tree_initial_height_l234_234935

theorem tree_initial_height (H : ℝ) (C : ℝ) (P : H + 6 = (H + 4) + 1/4 * (H + 4) ∧ C = 1) : H = 4 :=
by
  let H := 4
  sorry

end tree_initial_height_l234_234935


namespace closest_point_to_line_l234_234961

theorem closest_point_to_line {x y : ℝ} (h : y = 2 * x - 4) :
  ∃ (closest_x closest_y : ℝ),
    closest_x = 9 / 5 ∧ closest_y = -2 / 5 ∧ closest_y = 2 * closest_x - 4 ∧
    ∀ (x' y' : ℝ), y' = 2 * x' - 4 → (closest_x - 3)^2 + (closest_y + 1)^2 ≤ (x' - 3)^2 + (y' + 1)^2 :=
by
  sorry

end closest_point_to_line_l234_234961


namespace Sidney_JumpJacks_Tuesday_l234_234485

variable (JumpJacksMonday JumpJacksTuesday JumpJacksWednesday JumpJacksThursday : ℕ)
variable (SidneyTotalJumpJacks BrookeTotalJumpJacks : ℕ)

-- Given conditions
axiom H1 : JumpJacksMonday = 20
axiom H2 : JumpJacksWednesday = 40
axiom H3 : JumpJacksThursday = 50
axiom H4 : BrookeTotalJumpJacks = 3 * SidneyTotalJumpJacks
axiom H5 : BrookeTotalJumpJacks = 438

-- Prove Sidney's JumpJacks on Tuesday
theorem Sidney_JumpJacks_Tuesday : JumpJacksTuesday = 36 :=
by
  sorry

end Sidney_JumpJacks_Tuesday_l234_234485


namespace find_x_l234_234715

theorem find_x (x y z : ℝ) (h1 : x ≠ 0) 
  (h2 : x / 3 = z + 2 * y ^ 2) 
  (h3 : x / 6 = 3 * z - y) : 
  x = 168 :=
by
  sorry

end find_x_l234_234715


namespace remainder_2015_div_28_l234_234102

theorem remainder_2015_div_28 : 2015 % 28 = 17 :=
by
  sorry

end remainder_2015_div_28_l234_234102


namespace sum_of_exponents_sqrt_l234_234181

theorem sum_of_exponents_sqrt (a b c : ℕ) : 2 + 4 + 6 = 12 := by
  sorry

end sum_of_exponents_sqrt_l234_234181


namespace induction_example_l234_234876

theorem induction_example (n : ℕ) (h : n ≥ 5) : 2^n > n^2 + 1 :=
sorry

end induction_example_l234_234876


namespace minimum_reciprocal_sum_l234_234721

noncomputable def log_function_a (a : ℝ) (x : ℝ) : ℝ := 
  Real.log x / Real.log a

theorem minimum_reciprocal_sum (a m n : ℝ) 
  (ha1 : 0 < a) (ha2 : a ≠ 1) 
  (hmn : 0 < m ∧ 0 < n ∧ 2 * m + n = 2) 
  (hA : log_function_a a (1 : ℝ) + -1 = -1) 
  : 1 / m + 2 / n = 4 := 
by
  sorry

end minimum_reciprocal_sum_l234_234721


namespace roy_older_than_julia_l234_234619

variable {R J K x : ℝ}

theorem roy_older_than_julia (h1 : R = J + x)
                            (h2 : R = K + x / 2)
                            (h3 : R + 2 = 2 * (J + 2))
                            (h4 : (R + 2) * (K + 2) = 192) :
                            x = 2 :=
by
  sorry

end roy_older_than_julia_l234_234619


namespace min_value_of_E_l234_234386

noncomputable def E : ℝ := sorry

theorem min_value_of_E :
  (∀ x : ℝ, |E| + |x + 7| + |x - 5| ≥ 12) →
  (∃ x : ℝ, |x + 7| + |x - 5| = 12 → |E| = 0) :=
sorry

end min_value_of_E_l234_234386


namespace olympiad_problem_l234_234244

variable (a b c d : ℕ)
variable (N : ℕ := a + b + c + d)

theorem olympiad_problem
  (h1 : (a + d) / (N:ℚ) = 0.5)
  (h2 : (b + d) / (N:ℚ) = 0.6)
  (h3 : (c + d) / (N:ℚ) = 0.7)
  : (d : ℚ) / N * 100 = 40 := by
  sorry

end olympiad_problem_l234_234244


namespace minimum_teachers_needed_l234_234520

theorem minimum_teachers_needed
  (math_teachers : ℕ) (physics_teachers : ℕ) (chemistry_teachers : ℕ)
  (max_subjects_per_teacher : ℕ) :
  math_teachers = 7 →
  physics_teachers = 6 →
  chemistry_teachers = 5 →
  max_subjects_per_teacher = 3 →
  ∃ t : ℕ, t = 5 ∧ (t * max_subjects_per_teacher ≥ math_teachers + physics_teachers + chemistry_teachers) :=
by
  repeat { sorry }

end minimum_teachers_needed_l234_234520


namespace value_of_x_l234_234411

theorem value_of_x (p q r x : ℝ)
  (h1 : p = 72)
  (h2 : q = 18)
  (h3 : r = 108)
  (h4 : x = 180 - (q + r)) : 
  x = 54 := by
  sorry

end value_of_x_l234_234411


namespace distance_between_trees_correct_l234_234567

-- Define the given conditions
def yard_length : ℕ := 300
def tree_count : ℕ := 26
def interval_count : ℕ := tree_count - 1

-- Define the target distance between two consecutive trees
def target_distance : ℕ := 12

-- Prove that the distance between two consecutive trees is correct
theorem distance_between_trees_correct :
  yard_length / interval_count = target_distance := 
by
  sorry

end distance_between_trees_correct_l234_234567


namespace function_problem_l234_234087

theorem function_problem (f : ℕ → ℝ) (h1 : ∀ p q : ℕ, f (p + q) = f p * f q) (h2 : f 1 = 3) :
  (f (1) ^ 2 + f (2)) / f (1) + (f (2) ^ 2 + f (4)) / f (3) + (f (3) ^ 2 + f (6)) / f (5) + 
  (f (4) ^ 2 + f (8)) / f (7) + (f (5) ^ 2 + f (10)) / f (9) = 30 := by
  sorry

end function_problem_l234_234087


namespace problem_l234_234178

variable (x y : ℝ)

-- Define the given condition
def condition : Prop := |x + 5| + (y - 4)^2 = 0

-- State the theorem we need to prove
theorem problem (h : condition x y) : (x + y)^99 = -1 := sorry

end problem_l234_234178


namespace arithmetic_geometric_sequence_problem_l234_234186

theorem arithmetic_geometric_sequence_problem 
  (a : ℕ → ℚ)
  (b : ℕ → ℚ)
  (q : ℚ)
  (h1 : ∀ n m : ℕ, a (n + m) = a n * (q ^ m))
  (h2 : a 2 * a 3 * a 4 = 27 / 64)
  (h3 : q = 2)
  (h4 : ∃ d : ℚ, ∀ n : ℕ, b (n + 1) = b n + d)
  (h5 : b 7 = a 5) : 
  b 3 + b 11 = 6 := 
sorry

end arithmetic_geometric_sequence_problem_l234_234186


namespace arithmetic_sequence_formula_l234_234966

theorem arithmetic_sequence_formula :
  ∀ (a : ℕ → ℕ), (a 1 = 2) → (∀ n, a (n + 1) = a n + 2) → ∀ n, a n = 2 * n :=
by
  intro a
  intro h1
  intro hdiff
  sorry

end arithmetic_sequence_formula_l234_234966


namespace opposite_of_neg3_is_3_l234_234286

theorem opposite_of_neg3_is_3 : -(-3) = 3 := by
  sorry

end opposite_of_neg3_is_3_l234_234286


namespace John_l234_234543

/-- Assume Grant scored 10 points higher on his math test than John.
John received a certain ratio of points as Hunter who scored 45 points on his math test.
Grant's test score was 100. -/
theorem John's_points_to_Hunter's_points_ratio 
  (Grant John Hunter : ℕ) 
  (h1 : Grant = John + 10)
  (h2 : Hunter = 45)
  (h_grant_score : Grant = 100) : 
  (John : ℚ) / (Hunter : ℚ) = 2 / 1 :=
sorry

end John_l234_234543


namespace inv_of_15_mod_1003_l234_234038

theorem inv_of_15_mod_1003 : ∃ x : ℕ, x ≤ 1002 ∧ 15 * x ≡ 1 [MOD 1003] ∧ x = 937 :=
by sorry

end inv_of_15_mod_1003_l234_234038


namespace y_coord_diff_eq_nine_l234_234736

-- Declaring the variables and conditions
variables (m n : ℝ) (p : ℝ) (h1 : p = 3)
variable (L1 : m = (n / 3) - (2 / 5))
variable (L2 : m + p = ((n + 9) / 3) - (2 / 5))

-- The theorem statement
theorem y_coord_diff_eq_nine : (n + 9) - n = 9 :=
by
  sorry

end y_coord_diff_eq_nine_l234_234736


namespace negation_of_exists_l234_234678

theorem negation_of_exists : (¬ ∃ x_0 : ℝ, x_0 < 0 ∧ x_0^2 > 0) ↔ ∀ x : ℝ, x < 0 → x^2 ≤ 0 :=
sorry

end negation_of_exists_l234_234678


namespace tony_water_intake_l234_234661

-- Define the constants and conditions
def water_yesterday : ℝ := 48
def percentage_less_yesterday : ℝ := 0.04
def percentage_more_day_before_yesterday : ℝ := 0.05

-- Define the key quantity to find
noncomputable def water_two_days_ago : ℝ := water_yesterday / (1.05 * (1 - percentage_less_yesterday))

-- The proof statement
theorem tony_water_intake :
  water_two_days_ago = 47.62 :=
by
  sorry

end tony_water_intake_l234_234661


namespace find_number_l234_234294

theorem find_number (x : ℝ) (h : 61 + 5 * 12 / (180 / x) = 62): x = 3 :=
by
  sorry

end find_number_l234_234294


namespace min_value_f_solve_inequality_f_l234_234434

noncomputable def f (x : ℝ) : ℝ := abs (x - 1) + abs (2 * x + 4)

-- Proof Problem 1
theorem min_value_f : ∃ x : ℝ, f x = 3 :=
by { sorry }

-- Proof Problem 2
theorem solve_inequality_f : {x : ℝ | abs (f x - 6) ≤ 1} = 
    ({x : ℝ | -10/3 ≤ x ∧ x ≤ -8/3} ∪ 
    {x : ℝ | 0 ≤ x ∧ x ≤ 1} ∪ 
    {x : ℝ | 1 < x ∧ x ≤ 4/3}) :=
by { sorry }

end min_value_f_solve_inequality_f_l234_234434


namespace performance_attendance_l234_234389

theorem performance_attendance (A C : ℕ) (hC : C = 18) (hTickets : 16 * A + 9 * C = 258) : A + C = 24 :=
by
  sorry

end performance_attendance_l234_234389


namespace unique_real_root_iff_a_eq_3_l234_234500

noncomputable def f (x a : ℝ) : ℝ := x^2 + a * abs x + a^2 - 9

theorem unique_real_root_iff_a_eq_3 {a : ℝ} (hu : ∃! x : ℝ, f x a = 0) : a = 3 :=
sorry

end unique_real_root_iff_a_eq_3_l234_234500


namespace S9_equals_27_l234_234648

variables {a : ℕ → ℤ} {S : ℕ → ℤ} {d : ℤ}

-- (Condition 1) The sequence is an arithmetic sequence: a_{n+1} = a_n + d
axiom arithmetic_seq : ∀ n : ℕ, a (n + 1) = a n + d

-- (Condition 2) The sum S_n is the sum of the first n terms of the sequence
axiom sum_first_n_terms : ∀ n : ℕ, S n = n * (a 1 + a n) / 2

-- (Condition 3) Given a_1 = 2 * a_3 - 3
axiom given_condition : a 1 = 2 * a 3 - 3

-- Prove that S_9 = 27
theorem S9_equals_27 : S 9 = 27 :=
by
  sorry

end S9_equals_27_l234_234648


namespace inequality_example_l234_234568

theorem inequality_example (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / Real.sqrt (a ^ 2 + 8 * b * c)) + (b / Real.sqrt (b ^ 2 + 8 * c * a)) + (c / Real.sqrt (c ^ 2 + 8 * a * b)) ≥ 1 :=
by
  sorry

end inequality_example_l234_234568


namespace angle_B_in_parallelogram_l234_234854

variable (A B : ℝ)

theorem angle_B_in_parallelogram (h_parallelogram : ∀ {A B C D : ℝ}, A + B = 180 ↔ A = B) 
  (h_A : A = 50) : B = 130 := by
  sorry

end angle_B_in_parallelogram_l234_234854


namespace fibonacci_contains_21_l234_234030

-- Definition of the Fibonacci sequence
def fibonacci : ℕ → ℕ 
| 0 => 1
| 1 => 1
| (n+2) => fibonacci n + fibonacci (n+1)

-- Theorem statement: Proving that 21 is in the Fibonacci sequence
theorem fibonacci_contains_21 : ∃ n, fibonacci n = 21 :=
by
  sorry

end fibonacci_contains_21_l234_234030


namespace allison_marbles_l234_234123

theorem allison_marbles (A B C : ℕ) (h1 : B = A + 8) (h2 : C = 3 * B) (h3 : C + A = 136) : 
  A = 28 :=
by
  sorry

end allison_marbles_l234_234123


namespace find_a_evaluate_expr_l234_234194

-- Given polynomials A and B
def A (a x y : ℝ) : ℝ := a * x^2 + 3 * x * y + 2 * |a| * x
def B (x y : ℝ) : ℝ := 2 * x^2 + 6 * x * y + 4 * x + y + 1

-- Statement part (1)
theorem find_a (a : ℝ) (x y : ℝ) (h : (2 * A a x y - B x y) = (2 * a - 2) * x^2 + (4 * |a| - 4) * x - y - 1) : a = -1 := 
  sorry

-- Expression for part (2)
def expr (a : ℝ) : ℝ := 3 * (-3 * a^2 - 2 * a) - (a^2 - 2 * (5 * a - 4 * a^2 + 1) - 2 * a)

-- Statement part (2)
theorem evaluate_expr : expr (-1) = -22 := 
  sorry

end find_a_evaluate_expr_l234_234194


namespace minimum_value_condition_l234_234565

theorem minimum_value_condition (x a : ℝ) (h1 : x > a) (h2 : ∀ y, y > a → x + 4 / (y - a) > 9) : a = 6 :=
sorry

end minimum_value_condition_l234_234565


namespace nine_fact_div_four_fact_eq_15120_l234_234561

theorem nine_fact_div_four_fact_eq_15120 :
  (362880 / 24) = 15120 :=
by
  sorry

end nine_fact_div_four_fact_eq_15120_l234_234561


namespace hex_B3F_to_decimal_l234_234228

-- Define the hexadecimal values of B, 3, F
def hex_B : ℕ := 11
def hex_3 : ℕ := 3
def hex_F : ℕ := 15

-- Prove the conversion of B3F_{16} to a base 10 integer equals 2879
theorem hex_B3F_to_decimal : (hex_B * 16^2 + hex_3 * 16^1 + hex_F * 16^0) = 2879 := 
by 
  -- calculation details skipped
  sorry

end hex_B3F_to_decimal_l234_234228


namespace chess_tournament_l234_234338

theorem chess_tournament (n : ℕ) (h1 : 10 * 9 * n / 2 = 90) : n = 2 :=
by
  sorry

end chess_tournament_l234_234338


namespace acrobat_eq_two_lambs_l234_234905

variables (ACROBAT DOG BARREL SPOOL LAMB : ℝ)

axiom acrobat_dog_eq_two_barrels : ACROBAT + DOG = 2 * BARREL
axiom dog_eq_two_spools : DOG = 2 * SPOOL
axiom lamb_spool_eq_barrel : LAMB + SPOOL = BARREL

theorem acrobat_eq_two_lambs : ACROBAT = 2 * LAMB :=
by
  sorry

end acrobat_eq_two_lambs_l234_234905


namespace compute_fraction_l234_234605

theorem compute_fraction :
  ( (12^4 + 500) * (24^4 + 500) * (36^4 + 500) * (48^4 + 500) * (60^4 + 500) ) /
  ( (6^4 + 500) * (18^4 + 500) * (30^4 + 500) * (42^4 + 500) * (54^4 + 500) ) = -182 :=
by
  sorry

end compute_fraction_l234_234605


namespace john_remaining_money_l234_234160

variable (q : ℝ)
variable (number_of_small_pizzas number_of_large_pizzas number_of_drinks : ℕ)
variable (cost_of_drink cost_of_small_pizza cost_of_large_pizza dollars_left : ℝ)

def john_purchases := number_of_small_pizzas = 2 ∧
                      number_of_large_pizzas = 1 ∧
                      number_of_drinks = 4 ∧
                      cost_of_drink = q ∧
                      cost_of_small_pizza = q ∧
                      cost_of_large_pizza = 4 * q ∧
                      dollars_left = 50 - (4 * q + 2 * q + 4 * q)

theorem john_remaining_money : john_purchases q 2 1 4 q q (4 * q) (50 - 10 * q) :=
by
  sorry

end john_remaining_money_l234_234160


namespace apples_count_l234_234737

theorem apples_count (n : ℕ) (h₁ : n > 2)
  (h₂ : 144 / n - 144 / (n + 2) = 1) :
  n + 2 = 18 :=
by
  sorry

end apples_count_l234_234737


namespace find_m_plus_n_l234_234855

theorem find_m_plus_n
  (m n : ℝ)
  (l1 : ∀ x y : ℝ, 2 * x + m * y + 2 = 0)
  (l2 : ∀ x y : ℝ, 2 * x + y - 1 = 0)
  (l3 : ∀ x y : ℝ, x + n * y + 1 = 0)
  (parallel_l1_l2 : ∀ x y : ℝ, (2 * x + m * y + 2 = 0) → (2 * x + y - 1 = 0))
  (perpendicular_l1_l3 : ∀ x y : ℝ, (2 * x + m * y + 2 = 0) ∧ (x + n * y + 1 = 0) → true) :
  m + n = -1 :=
by
  sorry

end find_m_plus_n_l234_234855


namespace flagstaff_height_is_correct_l234_234629

noncomputable def flagstaff_height : ℝ := 40.25 * 12.5 / 28.75

theorem flagstaff_height_is_correct :
  flagstaff_height = 17.5 :=
by 
  -- These conditions are implicit in the previous definition
  sorry

end flagstaff_height_is_correct_l234_234629


namespace pharmacist_weights_exist_l234_234595

theorem pharmacist_weights_exist :
  ∃ (a b c : ℝ), a + b = 100 ∧ a + c = 101 ∧ b + c = 102 ∧ a < 90 ∧ b < 90 ∧ c < 90 :=
by
  sorry

end pharmacist_weights_exist_l234_234595


namespace simplify_and_evaluate_l234_234168

noncomputable def expr (x : ℝ) : ℝ :=
  (x + 3) * (x - 2) + x * (4 - x)

theorem simplify_and_evaluate (x : ℝ) (hx : x = 2) : expr x = 4 :=
by
  rw [hx]
  show expr 2 = 4
  sorry

end simplify_and_evaluate_l234_234168


namespace probability_of_unique_color_and_number_l234_234929

-- Defining the sets of colors and numbers
inductive Color
| red
| yellow
| blue

inductive Number
| one
| two
| three

-- Defining a ball as a combination of a Color and a Number
structure Ball :=
(color : Color)
(number : Number)

-- Setting up the list of 9 balls
def allBalls : List Ball :=
  [⟨Color.red, Number.one⟩, ⟨Color.red, Number.two⟩, ⟨Color.red, Number.three⟩,
   ⟨Color.yellow, Number.one⟩, ⟨Color.yellow, Number.two⟩, ⟨Color.yellow, Number.three⟩,
   ⟨Color.blue, Number.one⟩, ⟨Color.blue, Number.two⟩, ⟨Color.blue, Number.three⟩]

-- Proving the probability calculation as a theorem
noncomputable def probability_neither_same_color_nor_number : ℕ → ℕ → ℚ :=
  λ favorable total => favorable / total

theorem probability_of_unique_color_and_number :
  probability_neither_same_color_nor_number
    (6) -- favorable outcomes
    (84) -- total outcomes
  = 1 / 14 := by
  sorry

end probability_of_unique_color_and_number_l234_234929


namespace gcd_18_30_l234_234069

theorem gcd_18_30: Int.gcd 18 30 = 6 := by
  sorry

end gcd_18_30_l234_234069


namespace hyperbola_focus_distance_l234_234290

theorem hyperbola_focus_distance :
  ∀ (x y : ℝ), (x^2 / 4 - y^2 / 3 = 1) → ∀ (F₁ F₂ : ℝ × ℝ), ∃ P : ℝ × ℝ, dist P F₁ = 3 → dist P F₂ = 7 :=
by
  sorry

end hyperbola_focus_distance_l234_234290


namespace sufficient_but_not_necessary_l234_234948

theorem sufficient_but_not_necessary (x : ℝ) :
  (x^2 > 1) → (1 / x < 1) ∧ ¬(1 / x < 1 → x^2 > 1) :=
by
  sorry

end sufficient_but_not_necessary_l234_234948


namespace gcd_gx_x_l234_234829

noncomputable def g (x : ℤ) : ℤ :=
  (3 * x + 5) * (9 * x + 4) * (11 * x + 8) * (x + 11)

theorem gcd_gx_x (x : ℤ) (h : 34914 ∣ x) : Int.gcd (g x) x = 1760 :=
by
  sorry

end gcd_gx_x_l234_234829


namespace number_of_pieces_correct_l234_234488

-- Define the dimensions of the pan
def pan_length : ℕ := 30
def pan_width : ℕ := 24

-- Define the dimensions of each piece of brownie
def piece_length : ℕ := 3
def piece_width : ℕ := 2

-- Calculate the area of the pan
def pan_area : ℕ := pan_length * pan_width

-- Calculate the area of each piece of brownie
def piece_area : ℕ := piece_length * piece_width

-- The proof problem statement
theorem number_of_pieces_correct : (pan_area / piece_area) = 120 :=
by sorry

end number_of_pieces_correct_l234_234488


namespace max_value_of_quadratic_l234_234180

theorem max_value_of_quadratic :
  ∀ z : ℝ, -6*z^2 + 24*z - 12 ≤ 12 :=
by
  sorry

end max_value_of_quadratic_l234_234180


namespace top_leftmost_rectangle_is_B_l234_234866

-- Define the sides of the rectangles
structure Rectangle :=
  (w : ℕ)
  (x : ℕ)
  (y : ℕ)
  (z : ℕ)

-- Define the specific rectangles with their side values
noncomputable def rectA : Rectangle := ⟨2, 7, 4, 7⟩
noncomputable def rectB : Rectangle := ⟨0, 6, 8, 5⟩
noncomputable def rectC : Rectangle := ⟨6, 3, 1, 1⟩
noncomputable def rectD : Rectangle := ⟨8, 4, 0, 2⟩
noncomputable def rectE : Rectangle := ⟨5, 9, 3, 6⟩
noncomputable def rectF : Rectangle := ⟨7, 5, 9, 0⟩

-- Prove that Rectangle B is the top leftmost rectangle
theorem top_leftmost_rectangle_is_B :
  (rectB.w = 0 ∧ rectB.x = 6 ∧ rectB.y = 8 ∧ rectB.z = 5) :=
by {
  sorry
}

end top_leftmost_rectangle_is_B_l234_234866


namespace total_students_l234_234562

theorem total_students (N : ℕ)
    (h1 : (15 * 75) + (10 * 90) = N * 81) :
    N = 25 :=
by
  sorry

end total_students_l234_234562


namespace find_ellipse_l234_234771

noncomputable def standard_equation_ellipse (x y : ℝ) : Prop :=
  (x^2 / 9 + y^2 / 3 = 1)
  ∨ (x^2 / 18 + y^2 / 9 = 1)
  ∨ (y^2 / (45 / 2) + x^2 / (45 / 4) = 1)

variables 
  (P1 P2 : ℝ × ℝ)
  (focus : ℝ × ℝ)
  (a b : ℝ)

def passes_through_points (P1 P2 : ℝ × ℝ) : Prop :=
  ∀ equation : (ℝ → ℝ → Prop), 
    equation P1.1 P1.2 ∧ equation P2.1 P2.2

def focus_conditions (focus : ℝ × ℝ) : Prop :=
  -- Condition indicating focus, relationship with the minor axis etc., will be precisely defined here
  true -- Placeholder, needs correct mathematical condition

theorem find_ellipse : 
  passes_through_points P1 P2 
  → focus_conditions focus 
  → standard_equation_ellipse x y :=
sorry

end find_ellipse_l234_234771


namespace a_seq_gt_one_l234_234118

noncomputable def a_seq (a : ℝ) (n : ℕ) : ℝ :=
  if n = 0 then 0
  else if n = 1 then 1 + a
  else (1 / a_seq a (n - 1)) + a

theorem a_seq_gt_one (a : ℝ) (h : 0 < a ∧ a < 1) : ∀ n : ℕ, 1 < a_seq a n :=
by {
  sorry
}

end a_seq_gt_one_l234_234118


namespace ratio_expression_value_l234_234874

theorem ratio_expression_value (a b : ℝ) (h : a / b = 4 / 1) : 
  (a - 3 * b) / (2 * a - b) = 1 / 7 := 
by 
  sorry

end ratio_expression_value_l234_234874


namespace simplify_fraction_l234_234131

theorem simplify_fraction :
  (144 : ℤ) / (1296 : ℤ) = 1 / 9 := 
by sorry

end simplify_fraction_l234_234131


namespace additional_oil_needed_l234_234758

def oil_needed_each_cylinder : ℕ := 8
def number_of_cylinders : ℕ := 6
def oil_already_added : ℕ := 16

theorem additional_oil_needed : 
  (oil_needed_each_cylinder * number_of_cylinders) - oil_already_added = 32 := by
  sorry

end additional_oil_needed_l234_234758


namespace triangle_possible_sides_l234_234635

theorem triangle_possible_sides (a b c : ℕ) (h₁ : a + b + c = 7) (h₂ : a + b > c) (h₃ : a + c > b) (h₄ : b + c > a) :
  a = 1 ∨ a = 2 ∨ a = 3 :=
by {
  sorry
}

end triangle_possible_sides_l234_234635


namespace parrots_per_cage_l234_234423

-- Definitions of the given conditions
def num_cages : ℕ := 6
def num_parakeets_per_cage : ℕ := 7
def total_birds : ℕ := 54

-- Proposition stating the question and the correct answer
theorem parrots_per_cage : (total_birds - num_cages * num_parakeets_per_cage) / num_cages = 2 := 
by
  sorry

end parrots_per_cage_l234_234423


namespace positive_difference_arithmetic_sequence_l234_234201

theorem positive_difference_arithmetic_sequence :
  let a := 3
  let d := 5
  let a₁₀₀ := a + (100 - 1) * d
  let a₁₁₀ := a + (110 - 1) * d
  a₁₁₀ - a₁₀₀ = 50 :=
by
  sorry

end positive_difference_arithmetic_sequence_l234_234201


namespace product_increase_false_l234_234369

theorem product_increase_false (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) : 
  ¬ (a * b = a * (10 * b) / 10 ∧ a * (10 * b) / 10 = 10 * (a * b)) :=
by 
  sorry

end product_increase_false_l234_234369


namespace rotation_transform_l234_234016

theorem rotation_transform (x y α : ℝ) :
    let x' := x * Real.cos α - y * Real.sin α
    let y' := x * Real.sin α + y * Real.cos α
    (x', y') = (x * Real.cos α - y * Real.sin α, x * Real.sin α + y * Real.cos α) := by
  sorry

end rotation_transform_l234_234016


namespace math_proof_problem_l234_234138

variable (a_n : ℕ → ℤ) (S_n : ℕ → ℤ)
variable (a_1 d : ℤ)
variable (n : ℕ)

def arith_seq : Prop := ∀ n, a_n n = a_1 + (n - 1) * d

def sum_arith_seq : Prop := ∀ n, S_n n = n * (a_1 + (n - 1) * d / 2)

def condition1 : Prop := a_n 5 + a_n 9 = -2

def condition2 : Prop := S_n 3 = 57

noncomputable def general_formula : Prop := ∀ n, a_n n = 27 - 4 * n

noncomputable def max_S_n : Prop := ∀ n, S_n n ≤ 78 ∧ ∃ n, S_n n = 78

theorem math_proof_problem : 
  arith_seq a_n a_1 d ∧ sum_arith_seq S_n a_1 d ∧ condition1 a_n ∧ condition2 S_n 
  → general_formula a_n ∧ max_S_n S_n := 
sorry

end math_proof_problem_l234_234138


namespace rectangle_area_exceeds_m_l234_234728

theorem rectangle_area_exceeds_m (m : ℤ) (h_m : m > 12) :
  ∃ x y : ℤ, x * y > m ∧ (x - 1) * y < m ∧ x * (y - 1) < m :=
by
  sorry

end rectangle_area_exceeds_m_l234_234728


namespace age_ratio_l234_234709

theorem age_ratio (R D : ℕ) (h1 : D = 15) (h2 : R + 6 = 26) : R / D = 4 / 3 := by
  sorry

end age_ratio_l234_234709


namespace decrease_percent_in_revenue_l234_234597

theorem decrease_percent_in_revenue
  (T C : ℝ) -- T = original tax, C = original consumption
  (h1 : 0 < T) -- ensuring that T is positive
  (h2 : 0 < C) -- ensuring that C is positive
  (new_tax : ℝ := 0.75 * T) -- new tax is 75% of original tax
  (new_consumption : ℝ := 1.10 * C) -- new consumption is 110% of original consumption
  (original_revenue : ℝ := T * C) -- original revenue
  (new_revenue : ℝ := (0.75 * T) * (1.10 * C)) -- new revenue
  (decrease_percent : ℝ := ((T * C - (0.75 * T) * (1.10 * C)) / (T * C)) * 100) -- decrease percent
  : decrease_percent = 17.5 :=
by
  sorry

end decrease_percent_in_revenue_l234_234597


namespace total_cost_of_fencing_l234_234105

def P : ℤ := 42 + 35 + 52 + 66 + 40
def cost_per_meter : ℤ := 3
def total_cost : ℤ := P * cost_per_meter

theorem total_cost_of_fencing : total_cost = 705 := by
  sorry

end total_cost_of_fencing_l234_234105


namespace arrangements_of_45520_l234_234712

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

def arrangements (n : Nat) (k : Nat) : Nat :=
  factorial n / factorial k

theorem arrangements_of_45520 : 
  let n0_pos := 4
  let remaining_digits := 4 * arrangements 4 2
  n0_pos * remaining_digits = 48 :=
by
  -- Definitions and lemmas can be introduced here
  sorry

end arrangements_of_45520_l234_234712


namespace max_height_reached_threat_to_object_at_70km_l234_234809

noncomputable def initial_acceleration : ℝ := 20 -- m/s^2
noncomputable def duration : ℝ := 50 -- seconds
noncomputable def gravity : ℝ := 10 -- m/s^2
noncomputable def height_at_max_time : ℝ := 75000 -- meters (75km)

-- Proof that the maximum height reached is 75 km
theorem max_height_reached (a τ g : ℝ) (H : ℝ) (h₀: a = initial_acceleration) (h₁: τ = duration) (h₂: g = gravity) (h₃: H = height_at_max_time) :
  H = 75 * 1000 := 
sorry

-- Proof that the rocket poses a threat to an object located at 70 km
theorem threat_to_object_at_70km (a τ g : ℝ) (H : ℝ) (h₀: a = initial_acceleration) (h₁: τ = duration) (h₂: g = gravity) (h₃: H = height_at_max_time) :
  H > 70 * 1000 :=
sorry

end max_height_reached_threat_to_object_at_70km_l234_234809


namespace Zilla_savings_l234_234914

/-- Zilla's monthly savings based on her spending distributions -/
theorem Zilla_savings
  (rent : ℚ) (monthly_earnings_percentage : ℚ)
  (other_expenses_fraction : ℚ) (monthly_rent : ℚ)
  (monthly_expenses : ℚ) (total_monthly_earnings : ℚ)
  (half_monthly_earnings : ℚ) (savings : ℚ)
  (h1 : rent = 133)
  (h2 : monthly_earnings_percentage = 0.07)
  (h3 : other_expenses_fraction = 0.5)
  (h4 : total_monthly_earnings = monthly_rent / monthly_earnings_percentage)
  (h5 : half_monthly_earnings = total_monthly_earnings * other_expenses_fraction)
  (h6 : savings = total_monthly_earnings - (monthly_rent + half_monthly_earnings))
  : savings = 817 :=
sorry

end Zilla_savings_l234_234914


namespace rectangular_prism_faces_l234_234111

theorem rectangular_prism_faces (n : ℕ) (h1 : ∀ z : ℕ, z > 0 → z^3 = 2 * n^3) 
  (h2 : n > 0) :
  (∃ f : ℕ, f = (1 / 6 : ℚ) * (6 * 2 * n^3) ∧ 
    f = 10 * n^2) ↔ n = 5 := by
sorry

end rectangular_prism_faces_l234_234111


namespace bryden_receives_10_dollars_l234_234845

theorem bryden_receives_10_dollars 
  (collector_rate : ℝ := 5)
  (num_quarters : ℝ := 4)
  (face_value_per_quarter : ℝ := 0.50) :
  collector_rate * num_quarters * face_value_per_quarter = 10 :=
by
  sorry

end bryden_receives_10_dollars_l234_234845


namespace middle_school_students_count_l234_234625

variable (M H m h : ℕ)
variable (total_students : ℕ := 36)
variable (percentage_middle : ℕ := 20)
variable (percentage_high : ℕ := 25)

theorem middle_school_students_count :
  total_students = 36 ∧ (m = h) →
  (percentage_middle / 100 * M = m) ∧
  (percentage_high / 100 * H = h) →
  M + H = total_students →
  M = 16 :=
by sorry

end middle_school_students_count_l234_234625


namespace num_zeros_in_binary_l234_234940

namespace BinaryZeros

def expression : ℕ := ((18 * 8192 + 8 * 128 - 12 * 16) / 6) + (4 * 64) + (3 ^ 5) - (25 * 2)

def binary_zeros (n : ℕ) : ℕ :=
  (Nat.digits 2 n).count 0

theorem num_zeros_in_binary :
  binary_zeros expression = 6 :=
by
  sorry

end BinaryZeros

end num_zeros_in_binary_l234_234940


namespace trapezoid_perimeter_l234_234599

noncomputable def length_AD : ℝ := 8
noncomputable def length_BC : ℝ := 18
noncomputable def length_AB : ℝ := 12 -- Derived from tangency and symmetry considerations
noncomputable def length_CD : ℝ := 18

theorem trapezoid_perimeter (ABCD : Π (a b c d : Type), a → b → c → d → Prop)
  (AD BC AB CD : ℝ)
  (h1 : AD = 8) (h2 : BC = 18) (h3 : AB = 12) (h4 : CD = 18)
  : AD + BC + AB + CD = 56 :=
by
  rw [h1, h2, h3, h4]
  norm_num

end trapezoid_perimeter_l234_234599


namespace correct_option_l234_234541

-- Define the operations as functions to be used in the Lean statement.
def optA : ℕ := 3 + 5 * 7 + 9
def optB : ℕ := 3 + 5 + 7 * 9
def optC : ℕ := 3 * 5 * 7 - 9
def optD : ℕ := 3 * 5 * 7 + 9
def optE : ℕ := 3 * 5 + 7 * 9

-- The theorem to prove that the correct option is (E).
theorem correct_option : optE = 78 ∧ optA ≠ 78 ∧ optB ≠ 78 ∧ optC ≠ 78 ∧ optD ≠ 78 := by {
  sorry
}

end correct_option_l234_234541


namespace circle_representation_l234_234132

theorem circle_representation (m : ℝ) : (∃ x y : ℝ, x^2 + y^2 - 2 * m * x + 2 * m^2 + 2 * m - 3 = 0) ↔ m ∈ Set.Ioo (-3 : ℝ) (1 / 2 : ℝ) :=
by
  sorry

end circle_representation_l234_234132


namespace probability_individual_selected_l234_234592

theorem probability_individual_selected :
  ∀ (N M : ℕ) (m : ℕ), N = 100 → M = 5 → (m < N) →
  (probability_of_selecting_m : ℝ) =
  (1 / N * M) :=
by
  intros N M m hN hM hm
  sorry

end probability_individual_selected_l234_234592


namespace range_of_a_l234_234732

variable (f : ℝ → ℝ)
variable (a : ℝ)

-- Condition: f(x) is an increasing function on ℝ.
def is_increasing_on_ℝ (f : ℝ → ℝ) := ∀ x y : ℝ, x < y → f x < f y

-- Equivalent proof problem in Lean 4:
theorem range_of_a (h : is_increasing_on_ℝ f) : 1 < a ∧ a < 6 := by
  sorry

end range_of_a_l234_234732


namespace min_rectangle_perimeter_l234_234842

theorem min_rectangle_perimeter (x y : ℤ) (h1 : x * y = 50) (hx : 0 < x) (hy : 0 < y) : 
  (∀ x y, x * y = 50 → 2 * (x + y) ≥ 30) ∧ 
  ∃ x y, x * y = 50 ∧ 2 * (x + y) = 30 := 
by sorry

end min_rectangle_perimeter_l234_234842


namespace train_speed_km_hr_l234_234747

def train_length : ℝ := 130  -- Length of the train in meters
def bridge_and_train_length : ℝ := 245  -- Total length of the bridge and the train in meters
def crossing_time : ℝ := 30  -- Time to cross the bridge in seconds

theorem train_speed_km_hr : (train_length + bridge_and_train_length) / crossing_time * 3.6 = 45 := by
  sorry

end train_speed_km_hr_l234_234747


namespace positive_integers_sum_of_squares_l234_234950

theorem positive_integers_sum_of_squares
  (a b c d : ℤ)
  (h1 : a^2 + b^2 + c^2 + d^2 = 90)
  (h2 : a + b + c + d = 16) :
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d := 
by
  sorry

end positive_integers_sum_of_squares_l234_234950


namespace ann_boxes_less_than_n_l234_234713

-- Define the total number of boxes n
def n : ℕ := 12

-- Define the number of boxes Mark sold
def mark_sold : ℕ := n - 11

-- Define a condition on the number of boxes Ann sold
def ann_sold (A : ℕ) : Prop := 1 ≤ A ∧ A < n - mark_sold

-- The statement to prove
theorem ann_boxes_less_than_n : ∃ A : ℕ, ann_sold A ∧ n - A = 2 :=
by
  sorry

end ann_boxes_less_than_n_l234_234713


namespace positive_real_triangle_inequality_l234_234346

theorem positive_real_triangle_inequality
    (a b c : ℝ)
    (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
    (h : 5 * a * b * c > a^3 + b^3 + c^3) :
    a + b > c ∧ a + c > b ∧ b + c > a :=
by
  sorry

end positive_real_triangle_inequality_l234_234346


namespace boat_capacity_per_trip_l234_234456

theorem boat_capacity_per_trip (trips_per_day : ℕ) (total_people : ℕ) (days : ℕ) :
  trips_per_day = 4 → total_people = 96 → days = 2 → (total_people / (trips_per_day * days)) = 12 :=
by
  intros
  sorry

end boat_capacity_per_trip_l234_234456


namespace systematic_sampling_first_two_numbers_l234_234931

theorem systematic_sampling_first_two_numbers
  (sample_size : ℕ) (population_size : ℕ) (last_sample_number : ℕ)
  (h1 : sample_size = 50) (h2 : population_size = 8000) (h3 : last_sample_number = 7900) :
  ∃ first second : ℕ, first = 60 ∧ second = 220 :=
by
  -- Proof to be provided.
  sorry

end systematic_sampling_first_two_numbers_l234_234931


namespace book_page_count_l234_234772

theorem book_page_count (pages_per_night : ℝ) (nights : ℝ) : pages_per_night = 120.0 → nights = 10.0 → pages_per_night * nights = 1200.0 :=
by
  sorry

end book_page_count_l234_234772


namespace rainfall_difference_l234_234126

noncomputable def r₁ : ℝ := 26
noncomputable def r₂ : ℝ := 34
noncomputable def r₃ : ℝ := r₂ - 12
noncomputable def avg : ℝ := 140

theorem rainfall_difference : (avg - (r₁ + r₂ + r₃)) = 58 := 
by
  sorry

end rainfall_difference_l234_234126


namespace cost_of_milk_l234_234764

-- Given conditions
def total_cost_of_groceries : ℕ := 42
def cost_of_bananas : ℕ := 12
def cost_of_bread : ℕ := 9
def cost_of_apples : ℕ := 14

-- Prove that the cost of milk is $7
theorem cost_of_milk : total_cost_of_groceries - (cost_of_bananas + cost_of_bread + cost_of_apples) = 7 := 
by 
  sorry

end cost_of_milk_l234_234764


namespace no_solns_to_equation_l234_234626

noncomputable def no_solution : Prop :=
  ∀ (n m r : ℕ), (1 ≤ n) → (1 ≤ m) → (1 ≤ r) → n^5 + 49^m ≠ 1221^r

theorem no_solns_to_equation : no_solution :=
sorry

end no_solns_to_equation_l234_234626


namespace log12_div_log15_eq_2m_n_div_1_m_n_l234_234636

variable (m n : Real)

theorem log12_div_log15_eq_2m_n_div_1_m_n 
  (h1 : Real.log 2 = m) 
  (h2 : Real.log 3 = n) : 
  Real.log 12 / Real.log 15 = (2 * m + n) / (1 - m + n) :=
by sorry

end log12_div_log15_eq_2m_n_div_1_m_n_l234_234636


namespace find_f_7_l234_234943

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function : ∀ x : ℝ, f (-x) = -f x
axiom function_period : ∀ x : ℝ, f (x + 2) = -f x
axiom function_value_range : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = x

theorem find_f_7 : f 7 = -1 := by
  sorry

end find_f_7_l234_234943


namespace no_such_number_exists_l234_234078

theorem no_such_number_exists :
  ¬ ∃ n : ℕ, 529 < n ∧ n < 538 ∧ 16 ∣ n :=
by sorry

end no_such_number_exists_l234_234078


namespace binders_required_l234_234891

variables (b1 b2 B1 B2 d1 d2 b3 : ℕ)

def binding_rate_per_binder_per_day : ℚ := B1 / (↑b1 * d1)

def books_per_binder_in_d2_days : ℚ := binding_rate_per_binder_per_day b1 B1 d1 * ↑d2

def binding_rate_for_b2_binders : ℚ := B2 / ↑b2

theorem binders_required (b1 b2 B1 B2 d1 d2 b3 : ℕ)
  (h1 : binding_rate_per_binder_per_day b1 B1 d1 = binding_rate_for_b2_binders b2 B2)
  (h2 : books_per_binder_in_d2_days b1 B1 d1 d2 = binding_rate_for_b2_binders b2 B2) :
  b3 = b2 :=
sorry

end binders_required_l234_234891


namespace speed_of_current_l234_234235

variable (m c : ℝ)

theorem speed_of_current (h1 : m + c = 15) (h2 : m - c = 10) : c = 2.5 :=
sorry

end speed_of_current_l234_234235


namespace solve_eq1_solve_eq2_l234_234839

variable (x : ℝ)

theorem solve_eq1 : (2 * x - 3 * (2 * x - 3) = x + 4) → (x = 1) :=
by
  intro h
  sorry

theorem solve_eq2 : ((3 / 4 * x - 1 / 4) - 1 = (5 / 6 * x - 7 / 6)) → (x = -1) :=
by
  intro h
  sorry

end solve_eq1_solve_eq2_l234_234839


namespace subtraction_equality_l234_234091

theorem subtraction_equality : 3.56 - 2.15 = 1.41 :=
by
  sorry

end subtraction_equality_l234_234091


namespace max_jogs_l234_234325

theorem max_jogs (jags jigs jogs jugs : ℕ) : 2 * jags + 3 * jigs + 8 * jogs + 5 * jugs = 72 → jags ≥ 1 → jigs ≥ 1 → jugs ≥ 1 → jogs ≤ 7 :=
by
  sorry

end max_jogs_l234_234325


namespace composite_10201_in_all_bases_greater_than_two_composite_10101_in_all_bases_l234_234297

-- Definition for part (a)
def composite_base_greater_than_two (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ (n^4 + 2*n^2 + 1) = a * b

-- Proof statement for part (a)
theorem composite_10201_in_all_bases_greater_than_two (n : ℕ) (h : n > 2) : composite_base_greater_than_two n :=
by sorry

-- Definition for part (b)
def composite_in_all_bases (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ (n^4 + n^2 + 1) = a * b

-- Proof statement for part (b)
theorem composite_10101_in_all_bases (n : ℕ) : composite_in_all_bases n :=
by sorry

end composite_10201_in_all_bases_greater_than_two_composite_10101_in_all_bases_l234_234297


namespace interest_rate_second_part_l234_234449

theorem interest_rate_second_part (P1 P2: ℝ) (total_sum : ℝ) (rate1 : ℝ) (time1 : ℝ) (time2 : ℝ) (interest_second_part: ℝ ) : 
  total_sum = 2717 → P2 = 1672 → time1 = 8 → rate1 = 3 → time2 = 3 →
  P1 + P2 = total_sum →
  P1 * rate1 * time1 / 100 = P2 * interest_second_part * time2 / 100 →
  interest_second_part = 5 :=
by
  sorry

end interest_rate_second_part_l234_234449


namespace best_coupon1_price_l234_234859

theorem best_coupon1_price (x : ℝ) 
    (h1 : 60 ≤ x ∨ x = 60)
    (h2_1 : 25 < 0.12 * x) 
    (h2_2 : 0.12 * x > 0.2 * x - 30) :
    x = 209.95 ∨ x = 229.95 ∨ x = 249.95 :=
by sorry

end best_coupon1_price_l234_234859


namespace solve_abs_inequality_l234_234653

theorem solve_abs_inequality (x : ℝ) : 
    (2 ≤ |x - 1| ∧ |x - 1| ≤ 5) ↔ ( -4 ≤ x ∧ x ≤ -1 ∨ 3 ≤ x ∧ x ≤ 6) := 
by
    sorry

end solve_abs_inequality_l234_234653


namespace present_worth_proof_l234_234258

-- Define the conditions
def banker's_gain (BG : ℝ) : Prop := BG = 16
def true_discount (TD : ℝ) : Prop := TD = 96

-- Define the relationship from the problem
def relationship (BG TD PW : ℝ) : Prop := BG = TD - PW

-- Define the present worth of the sum
def present_worth : ℝ := 80

-- Theorem stating that the present worth of the sum is Rs. 80 given the conditions
theorem present_worth_proof (BG TD PW : ℝ)
  (hBG : banker's_gain BG)
  (hTD : true_discount TD)
  (hRelation : relationship BG TD PW) :
  PW = present_worth := by
  sorry

end present_worth_proof_l234_234258


namespace base_conversion_problem_l234_234786

theorem base_conversion_problem 
  (b x y z : ℕ)
  (h1 : 1987 = x * b^2 + y * b + z)
  (h2 : x + y + z = 25) :
  b = 19 ∧ x = 5 ∧ y = 9 ∧ z = 11 := 
by
  sorry

end base_conversion_problem_l234_234786


namespace maximilian_wealth_greater_than_national_wealth_l234_234973

theorem maximilian_wealth_greater_than_national_wealth (x y z : ℝ) (h1 : 2 * x > z) (h2 : y < z) :
    x > (2 * x + y) - (x + z) :=
by
  sorry

end maximilian_wealth_greater_than_national_wealth_l234_234973


namespace range_of_smallest_side_l234_234495

theorem range_of_smallest_side 
  (c : ℝ) -- the perimeter of the triangle
  (a : ℝ) (b : ℝ) (A : ℝ)  -- three sides of the triangle
  (ha : 0 < a) 
  (hb : b = 2 * a) 
  (hc : a + b + A = c)
  (htriangle : a + b > A ∧ a + A > b ∧ b + A > a) 
  : 
  ∃ (l u : ℝ), l = c / 6 ∧ u = c / 4 ∧ l < a ∧ a < u 
:= sorry

end range_of_smallest_side_l234_234495


namespace AD_mutually_exclusive_not_complementary_l234_234964

-- Define the sets representing the outcomes of the events
def A : Set ℕ := {1, 3, 5}
def B : Set ℕ := {2, 4, 6}
def C : Set ℕ := {2, 4, 6}
def D : Set ℕ := {2, 4}

-- Define mutually exclusive
def mutually_exclusive (X Y : Set ℕ) : Prop := X ∩ Y = ∅

-- Define complementary
def complementary (X Y : Set ℕ) : Prop := X ∪ Y = {1, 2, 3, 4, 5, 6}

-- The statement to prove that events A and D are mutually exclusive but not complementary
theorem AD_mutually_exclusive_not_complementary :
  mutually_exclusive A D ∧ ¬ complementary A D :=
by
  sorry

end AD_mutually_exclusive_not_complementary_l234_234964


namespace age_equivalence_l234_234390

variable (x : ℕ)

theorem age_equivalence : ∃ x : ℕ, 60 + x = 35 + x + 11 + x ∧ x = 14 :=
by
  sorry

end age_equivalence_l234_234390


namespace zero_point_neg_x₀_l234_234633

-- Define odd function property
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define zero point condition for the function
def is_zero_point (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  f x₀ = Real.exp x₀

-- The main theorem to be proved
theorem zero_point_neg_x₀ (f : ℝ → ℝ) (x₀ : ℝ)
  (h_odd : is_odd_function f)
  (h_zero : is_zero_point f x₀) :
  f (-x₀) * Real.exp x₀ + 1 = 0 :=
sorry

end zero_point_neg_x₀_l234_234633


namespace coefficient_a2b2_in_expansion_l234_234590

theorem coefficient_a2b2_in_expansion :
  -- Combining the coefficients: \binom{4}{2} and \binom{6}{3}
  (Nat.choose 4 2) * (Nat.choose 6 3) = 120 :=
by
  -- No proof required, using sorry to indicate that.
  sorry

end coefficient_a2b2_in_expansion_l234_234590


namespace fourth_quadrant_point_l234_234793

theorem fourth_quadrant_point (a : ℤ) (h1 : 2 * a + 6 > 0) (h2 : 3 * a + 3 < 0) :
  (2 * a + 6, 3 * a + 3) = (2, -3) :=
sorry

end fourth_quadrant_point_l234_234793


namespace div_by_5_mul_diff_l234_234811

theorem div_by_5_mul_diff (x y z : ℤ) (hxy : x ≠ y) (hyz : y ≠ z) (hzx : z ≠ x) :
  5 ∣ ((x - y)^5 + (y - z)^5 + (z - x)^5) :=
by
  sorry

end div_by_5_mul_diff_l234_234811


namespace fraction_zero_solution_l234_234657

theorem fraction_zero_solution (x : ℝ) (h : (|x| - 2) / (x - 2) = 0) : x = -2 :=
sorry

end fraction_zero_solution_l234_234657


namespace arithmetic_problem_l234_234658

theorem arithmetic_problem : (56^2 + 56^2) / 28^2 = 8 := by
  sorry

end arithmetic_problem_l234_234658


namespace perfect_square_trinomial_implies_possible_m_values_l234_234507

theorem perfect_square_trinomial_implies_possible_m_values (m : ℝ) :
  (∃ a : ℝ, ∀ x : ℝ, (x - a)^2 = x^2 - 2*m*x + 16) → (m = 4 ∨ m = -4) :=
by
  sorry

end perfect_square_trinomial_implies_possible_m_values_l234_234507


namespace circle_condition_l234_234100

theorem circle_condition (f : ℝ) : (∃ x y : ℝ, x^2 + y^2 - 4*x + 6*y + f = 0) ↔ f < 13 :=
by
  sorry

end circle_condition_l234_234100


namespace train_stoppage_time_l234_234664

theorem train_stoppage_time
  (D : ℝ) -- Distance in kilometers
  (T_no_stop : ℝ := D / 300) -- Time without stoppages in hours
  (T_with_stop : ℝ := D / 200) -- Time with stoppages in hours
  (T_stop : ℝ := T_with_stop - T_no_stop) -- Time lost due to stoppages in hours
  (T_stop_minutes : ℝ := T_stop * 60) -- Time lost due to stoppages in minutes
  (stoppage_per_hour : ℝ := T_stop_minutes / (D / 300)) -- Time stopped per hour of travel
  : stoppage_per_hour = 30 := sorry

end train_stoppage_time_l234_234664


namespace gcd_175_100_65_l234_234685

theorem gcd_175_100_65 : Nat.gcd (Nat.gcd 175 100) 65 = 5 :=
by
  sorry

end gcd_175_100_65_l234_234685


namespace discount_percentage_is_25_l234_234908

def piano_cost := 500
def lessons_count := 20
def lesson_price := 40
def total_paid := 1100

def lessons_cost := lessons_count * lesson_price
def total_cost := piano_cost + lessons_cost
def discount_amount := total_cost - total_paid
def discount_percentage := (discount_amount / lessons_cost) * 100

theorem discount_percentage_is_25 : discount_percentage = 25 := by
  sorry

end discount_percentage_is_25_l234_234908


namespace prism_faces_l234_234917

theorem prism_faces (E V F : ℕ) (n : ℕ) 
  (h1 : E + V = 40) 
  (h2 : E = 3 * F - 6) 
  (h3 : V - E + F = 2)
  (h4 : V = 2 * n)
  : F = 10 := 
by
  sorry

end prism_faces_l234_234917


namespace value_of_PQRS_l234_234484

theorem value_of_PQRS : 
  let P := 2 * (Real.sqrt 2010 + Real.sqrt 2011)
  let Q := 3 * (-Real.sqrt 2010 - Real.sqrt 2011)
  let R := 2 * (Real.sqrt 2010 - Real.sqrt 2011)
  let S := 3 * (Real.sqrt 2011 - Real.sqrt 2010)
  P * Q * R * S = -36 :=
by
  sorry

end value_of_PQRS_l234_234484


namespace value_of_frac_l234_234549

theorem value_of_frac (a b c d : ℕ) (h1 : a = 4 * b) (h2 : b = 3 * c) (h3 : c = 5 * d) :
  (a * c) / (b * d) = 20 :=
by
  sorry

end value_of_frac_l234_234549


namespace boy_real_name_is_kolya_l234_234660

variable (days_answers : Fin 6 → String)
variable (lies_on : Fin 6 → Bool)
variable (truth_days : List (Fin 6))

-- Define the conditions
def condition_truth_days : List (Fin 6) := [0, 1] -- Suppose Thursday is 0, Friday is 1.
def condition_lies_on (d : Fin 6) : Bool := d = 2 -- Suppose Tuesday is 2.

-- The sequence of answers
def condition_days_answers : Fin 6 → String := 
  fun d => match d with
    | 0 => "Kolya"
    | 1 => "Petya"
    | 2 => "Kolya"
    | 3 => "Petya"
    | 4 => "Vasya"
    | 5 => "Petya"
    | _ => "Unknown"

-- The proof problem statement
theorem boy_real_name_is_kolya : 
  ∀ (d : Fin 6), 
  (d ∈ condition_truth_days → condition_days_answers d = "Kolya") ∧
  (condition_lies_on d → condition_days_answers d ≠ "Vasya") ∧ 
  (¬(d ∈ condition_truth_days ∨ condition_lies_on d) → True) →
  "Kolya" = "Kolya" :=
by
  sorry

end boy_real_name_is_kolya_l234_234660


namespace words_lost_equal_137_l234_234505

-- Definitions based on conditions
def letters_in_oz : ℕ := 68
def forbidden_letter_index : ℕ := 7

def words_lost_due_to_forbidden_letter : ℕ :=
  let one_letter_words_lost : ℕ := 1
  let two_letter_words_lost : ℕ := 2 * (letters_in_oz - 1)
  one_letter_words_lost + two_letter_words_lost

-- Theorem stating that the words lost due to prohibition is 137
theorem words_lost_equal_137 :
  words_lost_due_to_forbidden_letter = 137 :=
sorry

end words_lost_equal_137_l234_234505


namespace slope_angle_of_line_l234_234675

theorem slope_angle_of_line (θ : ℝ) : 
  (∃ m : ℝ, ∀ x y : ℝ, 4 * x + y - 1 = 0 ↔ y = m * x + 1) ∧ (m = -4) → 
  θ = Real.pi - Real.arctan 4 :=
by
  sorry

end slope_angle_of_line_l234_234675


namespace cleaning_time_together_l234_234835

theorem cleaning_time_together (lisa_time kay_time ben_time sarah_time : ℕ)
  (h_lisa : lisa_time = 8) (h_kay : kay_time = 12) 
  (h_ben : ben_time = 16) (h_sarah : sarah_time = 24) :
  1 / ((1 / (lisa_time:ℚ)) + (1 / (kay_time:ℚ)) + (1 / (ben_time:ℚ)) + (1 / (sarah_time:ℚ))) = (16 / 5 : ℚ) :=
by
  sorry

end cleaning_time_together_l234_234835


namespace find_value_of_reciprocal_cubic_sum_l234_234867

theorem find_value_of_reciprocal_cubic_sum
  (a b c r s : ℝ)
  (h₁ : a + b + c = 0)
  (h₂ : a ≠ 0)
  (h₃ : b^2 - 4 * a * c ≥ 0)
  (h₄ : r ≠ 0)
  (h₅ : s ≠ 0)
  (h₆ : a * r^2 + b * r + c = 0)
  (h₇ : a * s^2 + b * s + c = 0)
  (h₈ : r + s = -b / a)
  (h₉ : r * s = -c / a) :
  1 / r^3 + 1 / s^3 = -b * (b^2 + 3 * a^2 + 3 * a * b) / (a + b)^3 :=
by
  sorry

end find_value_of_reciprocal_cubic_sum_l234_234867


namespace possible_values_l234_234328

noncomputable def matrixN (x y z : ℂ) : Matrix (Fin 3) (Fin 3) ℂ :=
  ![![x, y, z], ![z, x, y], ![y, z, x]]

theorem possible_values (x y z : ℂ) (h1 : (matrixN x y z)^3 = 1)
  (h2 : x * y * z = 1) : x^3 + y^3 + z^3 = 4 ∨ x^3 + y^3 + z^3 = -2 :=
  sorry

end possible_values_l234_234328


namespace probability_factor_90_less_than_10_l234_234609

-- Definitions from conditions
def number_factors_90 : ℕ := 12
def factors_90_less_than_10 : ℕ := 6

-- The corresponding proof problem
theorem probability_factor_90_less_than_10 : 
  (factors_90_less_than_10 / number_factors_90 : ℚ) = 1 / 2 :=
by
  sorry  -- proof to be filled in

end probability_factor_90_less_than_10_l234_234609


namespace find_x_l234_234551

theorem find_x (x : ℝ) : (0.75 / x = 10 / 8) → (x = 0.6) := by
  sorry

end find_x_l234_234551


namespace solve_for_y_l234_234221

variable (x y z : ℝ)

theorem solve_for_y (h : 3 * x + 3 * y + 3 * z + 11 = 143) : y = 44 - x - z :=
by 
  sorry

end solve_for_y_l234_234221


namespace total_candies_needed_l234_234083

def candies_per_box : ℕ := 156
def number_of_children : ℕ := 20

theorem total_candies_needed : candies_per_box * number_of_children = 3120 := by
  sorry

end total_candies_needed_l234_234083


namespace items_in_storeroom_l234_234121

-- Conditions definitions
def restocked_items : ℕ := 4458
def sold_items : ℕ := 1561
def total_items_left : ℕ := 3472

-- Statement of the proof
theorem items_in_storeroom : (total_items_left - (restocked_items - sold_items)) = 575 := 
by
  sorry

end items_in_storeroom_l234_234121


namespace five_times_x_plus_four_l234_234413

theorem five_times_x_plus_four (x : ℝ) (h : 4 * x - 3 = 13 * x + 12) : 5 * (x + 4) = 35 / 3 := 
by
  sorry

end five_times_x_plus_four_l234_234413


namespace undefined_values_of_expression_l234_234994

theorem undefined_values_of_expression (a : ℝ) :
  a^2 - 9 = 0 ↔ a = -3 ∨ a = 3 := 
sorry

end undefined_values_of_expression_l234_234994


namespace water_charging_standard_l234_234064

theorem water_charging_standard
  (x y : ℝ)
  (h1 : 10 * x + 5 * y = 35)
  (h2 : 10 * x + 8 * y = 44) : 
  x = 2 ∧ y = 3 :=
by
  sorry

end water_charging_standard_l234_234064


namespace EFGH_perimeter_l234_234897

noncomputable def perimeter_rectangle_EFGH (WE EX WY XZ : ℕ) : Rat :=
  let WX := Real.sqrt (WE ^ 2 + EX ^ 2)
  let p := 15232
  let q := 100
  p / q

theorem EFGH_perimeter :
  let WE := 12
  let EX := 16
  let WY := 24
  let XZ := 32
  perimeter_rectangle_EFGH WE EX WY XZ = 15232 / 100 :=
by
  sorry

end EFGH_perimeter_l234_234897


namespace triangle_right_triangle_l234_234608

-- Defining the sides of the triangle
variables (a b c : ℝ)

-- Theorem statement
theorem triangle_right_triangle (h : (a + b)^2 = c^2 + 2 * a * b) : a^2 + b^2 = c^2 :=
by {
  sorry
}

end triangle_right_triangle_l234_234608


namespace kristine_travel_distance_l234_234618

theorem kristine_travel_distance :
  ∃ T : ℝ, T + T / 2 + T / 6 = 500 ∧ T = 300 := by
  sorry

end kristine_travel_distance_l234_234618


namespace Somu_years_back_l234_234603

-- Define the current ages of Somu and his father, and the relationship between them
variables (S F : ℕ)
variable (Y : ℕ)

-- Hypotheses based on the problem conditions
axiom age_of_Somu : S = 14
axiom age_relation : S = F / 3

-- Define the condition for years back when Somu was one-fifth his father's age
axiom years_back_condition : S - Y = (F - Y) / 5

-- Problem statement: Prove that 7 years back, Somu was one-fifth of his father's age
theorem Somu_years_back : Y = 7 :=
by
  sorry

end Somu_years_back_l234_234603


namespace number_of_dvds_remaining_l234_234196

def initial_dvds : ℕ := 850

def week1_rented : ℕ := (initial_dvds * 25) / 100
def week1_sold : ℕ := 15
def remaining_after_week1 : ℕ := initial_dvds - week1_rented - week1_sold

def week2_rented : ℕ := (remaining_after_week1 * 35) / 100
def week2_sold : ℕ := 25
def remaining_after_week2 : ℕ := remaining_after_week1 - week2_rented - week2_sold

def week3_rented : ℕ := (remaining_after_week2 * 50) / 100
def week3_sold : ℕ := (remaining_after_week2 - week3_rented) * 5 / 100
def remaining_after_week3 : ℕ := remaining_after_week2 - week3_rented - week3_sold

theorem number_of_dvds_remaining : remaining_after_week3 = 181 :=
by
  -- proof goes here
  sorry

end number_of_dvds_remaining_l234_234196


namespace solve_for_x_l234_234868

theorem solve_for_x (x y : ℤ) (h1 : x + y = 10) (h2 : x - y = 18) : x = 14 :=
by
  -- Proof will go here
  sorry

end solve_for_x_l234_234868


namespace donny_money_left_l234_234846

-- Definitions based on Conditions
def initial_amount : ℝ := 78
def cost_kite : ℝ := 8
def cost_frisbee : ℝ := 9

-- Discounted cost of roller skates
def original_cost_roller_skates : ℝ := 15
def discount_rate_roller_skates : ℝ := 0.10
def discounted_cost_roller_skates : ℝ :=
  original_cost_roller_skates * (1 - discount_rate_roller_skates)

-- Cost of LEGO set with coupon
def original_cost_lego_set : ℝ := 25
def coupon_lego_set : ℝ := 5
def discounted_cost_lego_set : ℝ :=
  original_cost_lego_set - coupon_lego_set

-- Cost of puzzle with tax
def original_cost_puzzle : ℝ := 12
def tax_rate_puzzle : ℝ := 0.05
def taxed_cost_puzzle : ℝ :=
  original_cost_puzzle * (1 + tax_rate_puzzle)

-- Total cost calculated from item costs
def total_cost : ℝ :=
  cost_kite + cost_frisbee + discounted_cost_roller_skates + discounted_cost_lego_set + taxed_cost_puzzle

def money_left_after_shopping : ℝ :=
  initial_amount - total_cost

-- Prove the main statement
theorem donny_money_left : money_left_after_shopping = 14.90 := by
  sorry

end donny_money_left_l234_234846


namespace certain_number_correct_l234_234348

theorem certain_number_correct (x : ℝ) (h1 : 213 * 16 = 3408) (h2 : 213 * x = 340.8) : x = 1.6 := by
  sorry

end certain_number_correct_l234_234348


namespace find_other_number_l234_234313

theorem find_other_number 
  (A B : ℕ) 
  (h1 : A = 385) 
  (h2 : Nat.lcm A B = 2310) 
  (h3 : Nat.gcd A B = 30) : 
  B = 180 := 
by
  sorry

end find_other_number_l234_234313


namespace infinitely_many_not_representable_l234_234834

def can_be_represented_as_p_n_2k (c : ℕ) : Prop :=
  ∃ (p n k : ℕ), Prime p ∧ c = p + n^(2 * k)

theorem infinitely_many_not_representable :
  ∃ᶠ m in at_top, ¬ can_be_represented_as_p_n_2k (2^m + 1) := 
sorry

end infinitely_many_not_representable_l234_234834


namespace dhoni_dishwasher_spending_l234_234924

noncomputable def percentage_difference : ℝ := 0.25 - 0.225
noncomputable def percentage_less_than : ℝ := (percentage_difference / 0.25) * 100

theorem dhoni_dishwasher_spending :
  (percentage_difference / 0.25) * 100 = 10 :=
by sorry

end dhoni_dishwasher_spending_l234_234924


namespace betty_cupcakes_per_hour_l234_234647

theorem betty_cupcakes_per_hour (B : ℕ) (Dora_rate : ℕ) (betty_break_hours : ℕ) (total_hours : ℕ) (cupcake_diff : ℕ) :
  Dora_rate = 8 →
  betty_break_hours = 2 →
  total_hours = 5 →
  cupcake_diff = 10 →
  (total_hours - betty_break_hours) * B = Dora_rate * total_hours - cupcake_diff →
  B = 10 :=
by
  intros hDora_rate hbreak_hours htotal_hours hcupcake_diff hcupcake_eq
  sorry

end betty_cupcakes_per_hour_l234_234647


namespace part1_part2_range_of_a_l234_234108

noncomputable def f1 (x : ℝ) : ℝ := Real.sin x - Real.log (x + 1)

theorem part1 (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 1) : f1 x ≥ 0 := sorry

noncomputable def f2 (x a : ℝ) : ℝ := Real.sin x - a * Real.log (x + 1)

theorem part2 {a : ℝ} (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ Real.pi) : f2 x a ≤ 2 * Real.exp x - 2 := sorry

theorem range_of_a : {a : ℝ | ∀ x : ℝ, 0 ≤ x → x ≤ Real.pi → f2 x a ≤ 2 * Real.exp x - 2} = {a : ℝ | a ≥ -1} := sorry

end part1_part2_range_of_a_l234_234108


namespace kamal_twice_age_in_future_l234_234376

theorem kamal_twice_age_in_future :
  ∃ x : ℕ, (K = 40) ∧ (K - 8 = 4 * (S - 8)) ∧ (K + x = 2 * (S + x)) :=
by {
  sorry 
}

end kamal_twice_age_in_future_l234_234376


namespace arcsin_sqrt3_div_2_eq_pi_div_3_l234_234849

theorem arcsin_sqrt3_div_2_eq_pi_div_3 : 
  Real.arcsin (Real.sqrt 3 / 2) = Real.pi / 3 := 
by 
    sorry

end arcsin_sqrt3_div_2_eq_pi_div_3_l234_234849


namespace evaluate_expression_at_zero_l234_234008

theorem evaluate_expression_at_zero :
  (0^2 + 5 * 0 - 10) = -10 :=
by
  sorry

end evaluate_expression_at_zero_l234_234008


namespace proof_problem_l234_234683

-- Definition for the binomial coefficient
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Given conditions
def probability (b : ℕ) : ℚ :=
  (binom (40 - b) 2 + binom (b - 1) 2 : ℚ) / 1225

def is_coprime (a b : ℕ) : Prop :=
  Nat.gcd a b = 1

def minimum_b (b : ℕ) : Prop :=
  b = 11 ∧ probability 11 = 857 / 1225 ∧ is_coprime 857 1225 ∧ 857 + 1225 = 2082

-- Statement to prove
theorem proof_problem : ∃ b, minimum_b b := 
by
  -- Lean statement goes here
  sorry

end proof_problem_l234_234683


namespace train_speed_40_l234_234146

-- Definitions for the conditions
def passes_pole (L V : ℝ) := V = L / 8
def passes_stationary_train (L V : ℝ) := V = (L + 400) / 18

-- The theorem we want to prove
theorem train_speed_40 (L V : ℝ) (h1 : passes_pole L V) (h2 : passes_stationary_train L V) : V = 40 := 
sorry

end train_speed_40_l234_234146


namespace ned_did_not_wash_10_items_l234_234512

theorem ned_did_not_wash_10_items :
  let short_sleeve_shirts := 9
  let long_sleeve_shirts := 21
  let pairs_of_pants := 15
  let jackets := 8
  let total_items := short_sleeve_shirts + long_sleeve_shirts + pairs_of_pants + jackets
  let washed_items := 43
  let not_washed_Items := total_items - washed_items
  not_washed_Items = 10 := by
sorry

end ned_did_not_wash_10_items_l234_234512


namespace roots_imply_sum_l234_234926

theorem roots_imply_sum (a b c x1 x2 : ℝ) (hneq : a ≠ 0) (hroots : a * x1 ^ 2 + b * x1 + c = 0 ∧ a * x2 ^ 2 + b * x2 + c = 0) :
  x1 + x2 = -b / a :=
sorry

end roots_imply_sum_l234_234926


namespace not_prime_4k4_plus_1_not_prime_k4_plus_4_l234_234254

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem not_prime_4k4_plus_1 (k : ℕ) (hk : k > 0) : ¬ is_prime (4 * k^4 + 1) :=
by sorry

theorem not_prime_k4_plus_4 (k : ℕ) (hk : k > 0) : ¬ is_prime (k^4 + 4) :=
by sorry

end not_prime_4k4_plus_1_not_prime_k4_plus_4_l234_234254


namespace train_speed_l234_234122

theorem train_speed (v : ℕ) :
    let distance_between_stations := 155
    let speed_of_train_from_A := 20
    let start_time_train_A := 7
    let start_time_train_B := 8
    let meet_time := 11
    let distance_traveled_by_A := speed_of_train_from_A * (meet_time - start_time_train_A)
    let remaining_distance := distance_between_stations - distance_traveled_by_A
    let traveling_time_train_B := meet_time - start_time_train_B
    v * traveling_time_train_B = remaining_distance → v = 25 :=
by
  intros
  sorry

end train_speed_l234_234122


namespace find_constants_l234_234927

theorem find_constants
  (k m n : ℝ)
  (h : -x^3 + (k + 7) * x^2 + m * x - 8 = -(x - 2) * (x - 4) * (x - n)) :
  k = 7 ∧ m = 2 ∧ n = 1 :=
sorry

end find_constants_l234_234927


namespace inconsistent_fractions_l234_234861

theorem inconsistent_fractions : (3 / 5 : ℚ) + (17 / 20 : ℚ) > 1 := by
  sorry

end inconsistent_fractions_l234_234861


namespace tshirt_cost_l234_234534

theorem tshirt_cost (initial_amount sweater_cost shoes_cost amount_left spent_on_tshirt : ℕ) 
  (h_initial : initial_amount = 91) 
  (h_sweater : sweater_cost = 24) 
  (h_shoes : shoes_cost = 11) 
  (h_left : amount_left = 50)
  (h_spent : spent_on_tshirt = initial_amount - amount_left - sweater_cost - shoes_cost) :
  spent_on_tshirt = 6 :=
sorry

end tshirt_cost_l234_234534


namespace cone_volume_not_product_base_height_l234_234109

noncomputable def cone_volume (S h : ℝ) := (1/3) * S * h

theorem cone_volume_not_product_base_height (S h : ℝ) :
  cone_volume S h ≠ S * h :=
by sorry

end cone_volume_not_product_base_height_l234_234109


namespace car_b_speed_l234_234638

def speed_of_car_b (Vb Va : ℝ) (tA tB : ℝ) (dist total_dist : ℝ) : Prop :=
  Va = 3 * Vb ∧ tA = 6 ∧ tB = 2 ∧ dist = 1000 ∧ total_dist = Va * tA + Vb * tB

theorem car_b_speed : ∃ Vb Va tA tB dist total_dist, speed_of_car_b Vb Va tA tB dist total_dist ∧ Vb = 50 :=
by
  sorry

end car_b_speed_l234_234638


namespace total_songs_isabel_bought_l234_234141

theorem total_songs_isabel_bought
  (country_albums pop_albums : ℕ)
  (songs_per_album : ℕ)
  (h1 : country_albums = 6)
  (h2 : pop_albums = 2)
  (h3 : songs_per_album = 9) : 
  (country_albums + pop_albums) * songs_per_album = 72 :=
by
  -- We provide only the statement, no proof as per the instruction
  sorry

end total_songs_isabel_bought_l234_234141


namespace lunks_for_apples_l234_234257

theorem lunks_for_apples : 
  (∀ (a : ℕ) (b : ℕ) (k : ℕ), 3 * b * k = 5 * a → 15 * k = 9 * a ∧ 2 * a * 9 = 4 * b * 9 → 15 * 2 * a / 4 = 18) :=
by
  intro a b k h1 h2
  sorry

end lunks_for_apples_l234_234257


namespace solve_for_x_l234_234448

theorem solve_for_x :
  { x : Real | ⌊ 2 * x * ⌊ x ⌋ ⌋ = 58 } = {x : Real | 5.8 ≤ x ∧ x < 5.9} :=
sorry

end solve_for_x_l234_234448


namespace max_a3_in_arith_geo_sequences_l234_234525

theorem max_a3_in_arith_geo_sequences
  (a1 a2 a3 : ℝ) (b1 b2 b3 : ℝ)
  (h1 : a1 + a2 + a3 = 15)
  (h2 : a2 = ((a1 + a3) / 2))
  (h3 : b1 * b2 * b3 = 27)
  (h4 : (a1 + b1) * (a3 + b3) = (a2 + b2) ^ 2)
  (h5 : a1 + b1 > 0)
  (h6 : a2 + b2 > 0)
  (h7 : a3 + b3 > 0) :
  a3 ≤ 59 := sorry

end max_a3_in_arith_geo_sequences_l234_234525


namespace base7_perfect_square_values_l234_234439

theorem base7_perfect_square_values (a b c : ℕ) (h1 : a ≠ 0) (h2 : b < 7) :
  ∃ (n : ℕ), (343 * a + 49 * c + 28 + b = n * n) → (b = 0 ∨ b = 1 ∨ b = 4) :=
by
  sorry

end base7_perfect_square_values_l234_234439


namespace gcd_f_100_f_101_l234_234934

def f (x : ℕ) : ℕ := x^2 - 2*x + 2023

theorem gcd_f_100_f_101 : Nat.gcd (f 100) (f 101) = 1 := by
  sorry

end gcd_f_100_f_101_l234_234934


namespace inequality_proof_l234_234665

theorem inequality_proof 
  (x y z : ℝ) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (hz : 0 < z) 
  (hxyz : x + y + z = 1) : 
  (x^2 + y^2) / z + (y^2 + z^2) / x + (z^2 + x^2) / y ≥ 2 :=
by
  sorry

end inequality_proof_l234_234665


namespace compare_neg_frac1_l234_234406

theorem compare_neg_frac1 : (-3 / 7 : ℝ) < (-8 / 21 : ℝ) :=
sorry

end compare_neg_frac1_l234_234406


namespace range_of_a_l234_234248

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, x^2 + (1 - a) * x + 1 < 0) → a < -1 ∨ a > 3 :=
by
  sorry

end range_of_a_l234_234248


namespace Jackie_has_more_apples_l234_234365

def Adam_apples : Nat := 9
def Jackie_apples : Nat := 10

theorem Jackie_has_more_apples : Jackie_apples - Adam_apples = 1 := by
  sorry

end Jackie_has_more_apples_l234_234365


namespace arithmetic_sequence_sum_l234_234279

theorem arithmetic_sequence_sum (a : ℕ → ℕ) (S : ℕ → ℕ) (b : ℕ → ℚ) (T : ℕ → ℚ) 
  (h1 : a 3 = 7) (h2 : a 5 + a 7 = 26) :
  (∀ n, a n = 2 * n + 1) ∧
  (∀ n, S n = n^2 + 2 * n) ∧
  (∀ n, b n = 1 / ((2 * n + 1)^2 - 1)) ∧
  (∀ n, T n = n / (4 * (n + 1))) :=
by
  sorry

end arithmetic_sequence_sum_l234_234279


namespace angle_measure_l234_234645

theorem angle_measure (α : ℝ) 
  (h1 : 90 - α + (180 - α) = 180) : 
  α = 45 := 
by 
  sorry

end angle_measure_l234_234645


namespace circle_area_from_circumference_l234_234193

theorem circle_area_from_circumference (C : ℝ) (A : ℝ) (hC : C = 36) (hCircumference : ∀ r, C = 2 * Real.pi * r) (hAreaFormula : ∀ r, A = Real.pi * r^2) :
  A = 324 / Real.pi :=
by
  sorry

end circle_area_from_circumference_l234_234193


namespace cubic_roots_expression_l234_234847

noncomputable def polynomial : Polynomial ℂ :=
  Polynomial.X^3 - 3 * Polynomial.X - 2

theorem cubic_roots_expression (α β γ : ℂ)
  (h1 : (Polynomial.X - Polynomial.C α) * 
        (Polynomial.X - Polynomial.C β) * 
        (Polynomial.X - Polynomial.C γ) = polynomial) :
  α * (β - γ)^2 + β * (γ - α)^2 + γ * (α - β)^2 = -18 :=
by
  sorry

end cubic_roots_expression_l234_234847


namespace sum_divisible_by_5_and_7_remainder_12_l234_234957

theorem sum_divisible_by_5_and_7_remainder_12 :
  let a := 105
  let d := 35
  let n := 2013
  let S := (n * (2 * a + (n - 1) * d)) / 2
  S % 12 = 3 :=
by
  sorry

end sum_divisible_by_5_and_7_remainder_12_l234_234957


namespace find_t_l234_234970

theorem find_t (t : ℝ) :
  let P := (t - 5, -2)
  let Q := (-3, t + 4)
  let M := ((t - 8) / 2, (t + 2) / 2)
  (dist M P) ^ 2 = t^2 / 3 →
  t = -12 + 2 * Real.sqrt 21 ∨ t = -12 - 2 * Real.sqrt 21 := sorry

end find_t_l234_234970


namespace base_any_number_l234_234077

theorem base_any_number (base : ℝ) (x y : ℝ) (h1 : 3^x * base^y = 19683) (h2 : x - y = 9) (h3 : x = 9) : true :=
by
  sorry

end base_any_number_l234_234077


namespace smallest_d_l234_234467

noncomputable def smallestPositiveD : ℝ := 1

theorem smallest_d (d : ℝ) : 
  (0 < d) →
  (∀ x y : ℝ, 0 ≤ x → 0 ≤ y → 
    (Real.sqrt (x * y) + d * (x^2 - y^2)^2 ≥ x + y)) →
  d ≥ smallestPositiveD :=
by
  intros h1 h2
  sorry

end smallest_d_l234_234467


namespace value_of_p10_l234_234524

def p (d e f x : ℝ) : ℝ := d * x^2 + e * x + f

theorem value_of_p10 (d e f : ℝ) 
  (h1 : p d e f 3 = p d e f 4)
  (h2 : p d e f 2 = p d e f 5)
  (h3 : p d e f 0 = 2) :
  p d e f 10 = 2 :=
by
  sorry

end value_of_p10_l234_234524


namespace polynomial_identity_l234_234129

theorem polynomial_identity (a : ℝ) (h₁ : a^5 + 5 * a^4 + 10 * a^3 + 3 * a^2 - 9 * a - 6 = 0) (h₂ : a ≠ -1) : (a + 1)^3 = 7 :=
sorry

end polynomial_identity_l234_234129


namespace cole_drive_time_correct_l234_234000

noncomputable def cole_drive_time : ℕ :=
  let distance_to_work := 45 -- derived from the given problem   
  let speed_to_work := 30
  let time_to_work := distance_to_work / speed_to_work -- in hours
  (time_to_work * 60 : ℕ) -- converting hours to minutes

theorem cole_drive_time_correct
  (speed_to_work speed_return: ℕ)
  (total_time: ℕ)
  (H1: speed_to_work = 30)
  (H2: speed_return = 90)
  (H3: total_time = 2):
  cole_drive_time = 90 := by
  -- Proof omitted
  sorry

end cole_drive_time_correct_l234_234000


namespace floor_of_neg_seven_fourths_l234_234143

theorem floor_of_neg_seven_fourths : (Int.floor (-7/4 : ℚ)) = -2 :=
by
  sorry

end floor_of_neg_seven_fourths_l234_234143


namespace evaluate_expression_l234_234975

theorem evaluate_expression (x y : ℕ) (hx : x = 4) (hy : y = 9) : 
  2 * x^(y / 2 : ℕ) + 5 * y^(x / 2 : ℕ) = 1429 := by
  sorry

end evaluate_expression_l234_234975


namespace value_of_a_l234_234231

noncomputable def f (a : ℝ) (x : ℝ) := (x-1)*(x^2 - 3*x + a)

-- Define the condition that 1 is not a critical point
def not_critical (a : ℝ) : Prop := f a 1 ≠ 0

theorem value_of_a (a : ℝ) (h : not_critical a) : a = 2 := 
sorry

end value_of_a_l234_234231


namespace simplify_complex_fraction_l234_234864

theorem simplify_complex_fraction :
  let numerator := (5 : ℂ) + 7 * I
  let denominator := (2 : ℂ) + 3 * I
  numerator / denominator = (31 / 13 : ℂ) - (1 / 13) * I :=
by
  let numerator := (5 : ℂ) + 7 * I
  let denominator := (2 : ℂ) + 3 * I
  sorry

end simplify_complex_fraction_l234_234864


namespace no_solutions_l234_234270

theorem no_solutions
  (x y z : ℤ)
  (h : x^2 + y^2 = 4 * z - 1) : False :=
sorry

end no_solutions_l234_234270


namespace solve_inequality_l234_234408

-- Define the inequality problem.
noncomputable def inequality_problem (x : ℝ) : Prop :=
(x^2 + 2 * x - 15) / (x + 5) < 0

-- Define the solution set.
def solution_set (x : ℝ) : Prop :=
-5 < x ∧ x < 3

-- State the equivalence theorem.
theorem solve_inequality (x : ℝ) (h : x ≠ -5) : 
  inequality_problem x ↔ solution_set x :=
sorry

end solve_inequality_l234_234408


namespace beth_red_pill_cost_l234_234367

noncomputable def red_pill_cost (blue_pill_cost : ℝ) : ℝ := blue_pill_cost + 3

theorem beth_red_pill_cost :
  ∃ (blue_pill_cost : ℝ), 
  (21 * (red_pill_cost blue_pill_cost + blue_pill_cost) = 966) 
  → 
  red_pill_cost blue_pill_cost = 24.5 :=
by
  sorry

end beth_red_pill_cost_l234_234367


namespace julia_game_difference_l234_234694

theorem julia_game_difference :
  let tag_monday := 28
  let hide_seek_monday := 15
  let tag_tuesday := 33
  let hide_seek_tuesday := 21
  let total_monday := tag_monday + hide_seek_monday
  let total_tuesday := tag_tuesday + hide_seek_tuesday
  let difference := total_tuesday - total_monday
  difference = 11 := by
  sorry

end julia_game_difference_l234_234694


namespace maximum_value_attains_maximum_value_l234_234790

theorem maximum_value
  (a b c : ℝ)
  (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c)
  (h₃ : a + b + c = 1) :
  (ab / (a + b)) + (ac / (a + c)) + (bc / (b + c)) ≤ 1 / 2 :=
sorry

theorem attains_maximum_value :
  ∃ (a b c : ℝ), 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ a + b + c = 1 ∧
  (ab / (a + b)) + (ac / (a + c)) + (bc / (b + c)) = 1 / 2 :=
sorry

end maximum_value_attains_maximum_value_l234_234790


namespace largest_three_digit_n_l234_234034

theorem largest_three_digit_n (n : ℕ) : 
  (70 * n ≡ 210 [MOD 350]) ∧ (n < 1000) → 
  n = 998 := by
  sorry

end largest_three_digit_n_l234_234034


namespace geric_bills_l234_234040

variable (G K J : ℕ)

theorem geric_bills (h1 : G = 2 * K) 
                    (h2 : K = J - 2) 
                    (h3 : J = 7 + 3) : 
    G = 16 := by
  sorry

end geric_bills_l234_234040


namespace move_decimal_point_one_place_right_l234_234743

theorem move_decimal_point_one_place_right (x : ℝ) (h : x = 76.08) : x * 10 = 760.8 :=
by
  rw [h]
  -- Here, you would provide proof steps, but we'll use sorry to indicate the proof is omitted.
  sorry

end move_decimal_point_one_place_right_l234_234743


namespace articles_selling_price_to_cost_price_eq_l234_234147

theorem articles_selling_price_to_cost_price_eq (C N : ℝ) (h_gain : 2 * C * N = 20 * C) : N = 10 :=
by
  sorry

end articles_selling_price_to_cost_price_eq_l234_234147


namespace trains_cross_time_l234_234871

noncomputable def time_to_cross : ℝ := 
  let length_train1 := 110 -- length of the first train in meters
  let length_train2 := 150 -- length of the second train in meters
  let speed_train1 := 60 * 1000 / 3600 -- speed of the first train in meters per second
  let speed_train2 := 45 * 1000 / 3600 -- speed of the second train in meters per second
  let bridge_length := 340 -- length of the bridge in meters
  let total_distance := length_train1 + length_train2 + bridge_length -- total distance to be covered
  let relative_speed := speed_train1 + speed_train2 -- relative speed in meters per second
  total_distance / relative_speed

theorem trains_cross_time :
  abs (time_to_cross - 20.57) < 0.01 :=
sorry

end trains_cross_time_l234_234871


namespace general_term_formula_l234_234191

variable {a : ℕ → ℝ} -- Define the sequence as a function ℕ → ℝ

-- Conditions
axiom geom_seq (n : ℕ) (h : n ≥ 2): a (n + 1) = a 2 * (2 : ℝ) ^ (n - 1)
axiom a2_eq_2 : a 2 = 2
axiom a3_a4_cond : 2 * a 3 + a 4 = 16

theorem general_term_formula (n : ℕ) : a n = 2 ^ (n - 1) := by
  sorry -- Proof is not required

end general_term_formula_l234_234191


namespace population_at_300pm_l234_234149

namespace BacteriaProblem

def initial_population : ℕ := 50
def time_increments_to_220pm : ℕ := 4   -- 4 increments of 5 mins each till 2:20 p.m.
def time_increments_to_300pm : ℕ := 2   -- 2 increments of 10 mins each till 3:00 p.m.

def growth_factor_before_220pm : ℕ := 3
def growth_factor_after_220pm : ℕ := 2

theorem population_at_300pm :
  initial_population * growth_factor_before_220pm^time_increments_to_220pm *
  growth_factor_after_220pm^time_increments_to_300pm = 16200 :=
by
  sorry

end BacteriaProblem

end population_at_300pm_l234_234149


namespace quadratic_function_range_l234_234236

-- Define the quadratic function and the domain
def quadratic_function (x : ℝ) : ℝ := -(x - 2)^2 + 1

-- State the proof problem
theorem quadratic_function_range : ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 5 → -8 ≤ quadratic_function x ∧ quadratic_function x ≤ 1 := 
by 
  intro x
  intro h
  sorry

end quadratic_function_range_l234_234236


namespace museum_admission_ratio_l234_234337

theorem museum_admission_ratio (a c : ℕ) (h1 : 30 * a + 15 * c = 2700) (h2 : 2 ≤ a) (h3 : 2 ≤ c) :
  a / (180 - 2 * a) = 2 :=
by
  sorry

end museum_admission_ratio_l234_234337


namespace probability_prime_factor_of_120_l234_234959

open Nat

theorem probability_prime_factor_of_120 : 
  let s := Finset.range 61
  let primes := {2, 3, 5}
  let prime_factors_of_5_fact := primes ∩ s
  (prime_factors_of_5_fact.card : ℚ) / s.card = 1 / 20 :=
by
  sorry

end probability_prime_factor_of_120_l234_234959


namespace students_with_all_three_pets_l234_234907

variable (x y z : ℕ)
variable (total_students : ℕ := 40)
variable (dog_students : ℕ := total_students * 5 / 8)
variable (cat_students : ℕ := total_students * 1 / 4)
variable (other_students : ℕ := 8)
variable (no_pet_students : ℕ := 6)
variable (only_dog_students : ℕ := 12)
variable (only_other_students : ℕ := 3)
variable (cat_other_no_dog_students : ℕ := 10)

theorem students_with_all_three_pets :
  (x + y + z + 10 + 3 + 12 = total_students - no_pet_students) →
  (x + z + 10 = dog_students) →
  (10 + z = cat_students) →
  (y + z + 10 = other_students) →
  z = 0 :=
by
  -- Provide proof here
  sorry

end students_with_all_three_pets_l234_234907


namespace buns_per_pack_is_eight_l234_234659

-- Declaring the conditions
def burgers_per_guest : ℕ := 3
def total_friends : ℕ := 10
def friends_no_meat : ℕ := 1
def friends_no_bread : ℕ := 1
def packs_of_buns : ℕ := 3

-- Derived values from the conditions
def effective_friends_for_burgers : ℕ := total_friends - friends_no_meat
def effective_friends_for_buns : ℕ := total_friends - friends_no_bread

-- Final computation to prove
def buns_per_pack : ℕ := 24 / packs_of_buns

-- Theorem statement
theorem buns_per_pack_is_eight : buns_per_pack = 8 := by
  -- use sorry as we are not providing the proof steps 
  sorry

end buns_per_pack_is_eight_l234_234659


namespace first_player_can_ensure_distinct_rational_roots_l234_234821

theorem first_player_can_ensure_distinct_rational_roots :
  ∃ (a b c : ℚ), a + b + c = 0 ∧ (∀ x : ℚ, x^2 + (b/a) * x + (c/a) = 0 → False) :=
by
  sorry

end first_player_can_ensure_distinct_rational_roots_l234_234821


namespace coffee_ounces_per_cup_l234_234662

theorem coffee_ounces_per_cup
  (persons : ℕ)
  (cups_per_person_per_day : ℕ)
  (cost_per_ounce : ℝ)
  (total_spent_per_week : ℝ)
  (total_cups_per_day : ℕ)
  (total_cups_per_week : ℕ)
  (total_ounces : ℝ)
  (ounces_per_cup : ℝ) :
  persons = 4 →
  cups_per_person_per_day = 2 →
  cost_per_ounce = 1.25 →
  total_spent_per_week = 35 →
  total_cups_per_day = persons * cups_per_person_per_day →
  total_cups_per_week = total_cups_per_day * 7 →
  total_ounces = total_spent_per_week / cost_per_ounce →
  ounces_per_cup = total_ounces / total_cups_per_week →
  ounces_per_cup = 0.5 :=
by
  sorry

end coffee_ounces_per_cup_l234_234662


namespace weights_equal_weights_equal_ints_weights_equal_rationals_l234_234556

theorem weights_equal (w : Fin 13 → ℝ) (swap_n_weighs_balance : ∀ (s : Finset (Fin 13)), s.card = 12 → 
  ∃ (t u : Finset (Fin 13)), t.card = 6 ∧ u.card = 6 ∧ t ∪ u = s ∧ t ∩ u = ∅ ∧ Finset.sum t w = Finset.sum u w) :
  ∃ (m : ℝ), ∀ (i : Fin 13), w i = m :=
by
  sorry

theorem weights_equal_ints (w : Fin 13 → ℤ) (swap_n_weighs_balance_ints : ∀ (s : Finset (Fin 13)), s.card = 12 → 
  ∃ (t u : Finset (Fin 13)), t.card = 6 ∧ u.card = 6 ∧ t ∪ u = s ∧ t ∩ u = ∅ ∧ Finset.sum t w = Finset.sum u w) :
  ∃ (m : ℤ), ∀ (i : Fin 13), w i = m :=
by
  sorry

theorem weights_equal_rationals (w : Fin 13 → ℚ) (swap_n_weighs_balance_rationals : ∀ (s : Finset (Fin 13)), s.card = 12 → 
  ∃ (t u : Finset (Fin 13)), t.card = 6 ∧ u.card = 6 ∧ t ∪ u = s ∧ t ∩ u = ∅ ∧ Finset.sum t w = Finset.sum u w) :
  ∃ (m : ℚ), ∀ (i : Fin 13), w i = m :=
by
  sorry

end weights_equal_weights_equal_ints_weights_equal_rationals_l234_234556


namespace solve_for_a_l234_234880

def g (x : ℝ) : ℝ := 5 * x - 6

theorem solve_for_a (a : ℝ) : g a = 4 → a = 2 := by
  sorry

end solve_for_a_l234_234880


namespace complex_ab_value_l234_234401

theorem complex_ab_value (a b : ℝ) (i : ℂ) (h_i : i = Complex.I) (h_z : a + b * i = (4 + 3 * i) * i) : a * b = -12 :=
by {
  sorry
}

end complex_ab_value_l234_234401


namespace integer_solutions_abs_inequality_l234_234205

-- Define the condition as a predicate
def abs_inequality_condition (x : ℝ) : Prop := |x - 4| ≤ 3

-- State the proposition
theorem integer_solutions_abs_inequality : ∃ (n : ℕ), n = 7 ∧ ∀ (x : ℤ), abs_inequality_condition x → (x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 5 ∨ x = 6 ∨ x = 7) :=
sorry

end integer_solutions_abs_inequality_l234_234205


namespace central_angle_probability_l234_234357

theorem central_angle_probability (A : ℝ) (x : ℝ)
  (h1 : A > 0)
  (h2 : (x / 360) * A / A = 1 / 8) : 
  x = 45 := 
by
  sorry

end central_angle_probability_l234_234357


namespace number_of_triangles_for_second_star_l234_234133

theorem number_of_triangles_for_second_star (a b : ℝ) (h₁ : a + b + 90 = 180) (h₂ : 5 * (360 / 5) = 360) :
  360 / (180 - 90 - (360 / 5)) = 20 :=
by
  sorry

end number_of_triangles_for_second_star_l234_234133


namespace problem_1_problem_2_problem_3_l234_234519

noncomputable def f (a x : ℝ) : ℝ := a^(x-1)

theorem problem_1 (a : ℝ) (h_pos : a > 0) (h_neq : a ≠ 1) :
  f a 3 = 4 → a = 2 :=
sorry

theorem problem_2 (a : ℝ) (h_pos : a > 0) (h_neq : a ≠ 1) :
  f a (Real.log a) = 100 → (a = 100 ∨ a = 1 / 10) :=
sorry

theorem problem_3 (a : ℝ) (h_pos : a > 0) (h_neq : a ≠ 1) :
  (a > 1 → f a (Real.log (1 / 100)) > f a (-2.1)) ∧
  (0 < a ∧ a < 1 → f a (Real.log (1 / 100)) < f a (-2.1)) :=
sorry

end problem_1_problem_2_problem_3_l234_234519


namespace daniela_total_spent_l234_234416

-- Step d) Rewrite the math proof problem
theorem daniela_total_spent
    (shoe_price : ℤ) (dress_price : ℤ) (shoe_discount : ℤ) (dress_discount : ℤ)
    (shoe_count : ℤ)
    (shoe_original_price : shoe_price = 50)
    (dress_original_price : dress_price = 100)
    (shoe_discount_rate : shoe_discount = 40)
    (dress_discount_rate : dress_discount = 20)
    (shoe_total_count : shoe_count = 2)
    : shoe_count * (shoe_price - (shoe_price * shoe_discount / 100)) + (dress_price - (dress_price * dress_discount / 100)) = 140 := by 
    sorry

end daniela_total_spent_l234_234416


namespace man_speed_in_still_water_l234_234890

theorem man_speed_in_still_water 
  (V_u : ℕ) (V_d : ℕ) 
  (hu : V_u = 34) 
  (hd : V_d = 48) : 
  V_s = (V_u + V_d) / 2 :=
by
  sorry

end man_speed_in_still_water_l234_234890


namespace median_eq_range_le_l234_234451

def sample_data (x : ℕ → ℝ) :=
  x 1 ≤ x 2 ∧ x 2 ≤ x 3 ∧ x 3 ≤ x 4 ∧ x 4 ≤ x 5 ∧ x 5 ≤ x 6

theorem median_eq_range_le
  (x : ℕ → ℝ) 
  (h_sample_data : sample_data x) :
  ((x 3 + x 4) / 2 = (x 3 + x 4) / 2) ∧ (x 5 - x 2 ≤ x 6 - x 1) :=
by
  sorry

end median_eq_range_le_l234_234451


namespace scientific_notation_0_056_l234_234291

theorem scientific_notation_0_056 :
  (0.056 = 5.6 * 10^(-2)) :=
by
  sorry

end scientific_notation_0_056_l234_234291


namespace amount_spent_on_milk_is_1500_l234_234014

def total_salary (saved : ℕ) (saving_percent : ℕ) : ℕ := 
  saved / (saving_percent / 100)

def total_spent_excluding_milk (rent groceries education petrol misc : ℕ) : ℕ := 
  rent + groceries + education + petrol + misc

def amount_spent_on_milk (total_salary total_spent savings : ℕ) : ℕ := 
  total_salary - total_spent - savings

theorem amount_spent_on_milk_is_1500 :
  let rent := 5000
  let groceries := 4500
  let education := 2500
  let petrol := 2000
  let misc := 2500
  let savings := 2000
  let saving_percent := 10
  let salary := total_salary savings saving_percent
  let spent_excluding_milk := total_spent_excluding_milk rent groceries education petrol misc
  amount_spent_on_milk salary spent_excluding_milk savings = 1500 :=
by {
  sorry
}

end amount_spent_on_milk_is_1500_l234_234014


namespace people_got_off_at_first_stop_l234_234687

theorem people_got_off_at_first_stop 
  (X : ℕ)
  (h1 : 50 - X - 6 - 1 = 28) :
  X = 15 :=
by
  sorry

end people_got_off_at_first_stop_l234_234687


namespace initial_mixture_volume_l234_234906

variable (p q : ℕ) (x : ℕ)

theorem initial_mixture_volume :
  (3 * x) + (2 * x) = 5 * x →
  (3 * x) / (2 * x + 12) = 3 / 4 →
  5 * x = 30 :=
by
  sorry

end initial_mixture_volume_l234_234906


namespace inequality_false_l234_234586

variable {x y w : ℝ}

theorem inequality_false (hx : x > y) (hy : y > 0) (hw : w ≠ 0) : ¬(x^2 * w > y^2 * w) :=
by {
  sorry -- You could replace this "sorry" with a proper proof.
}

end inequality_false_l234_234586


namespace tetrahedron_volume_from_pentagon_l234_234695

noncomputable def volume_of_tetrahedron (side_length : ℝ) (diagonal_length : ℝ) (base_area : ℝ) (height : ℝ) : ℝ :=
  (1 / 3) * base_area * height

theorem tetrahedron_volume_from_pentagon :
  ∀ (s : ℝ), s = 1 →
  volume_of_tetrahedron s ((1 + Real.sqrt 5) / 2) ((Real.sqrt 3) / 4) (Real.sqrt ((5 + 2 * Real.sqrt 5) / 4)) =
  (1 + Real.sqrt 5) / 24 :=
by
  intros s hs
  rw [hs]
  sorry

end tetrahedron_volume_from_pentagon_l234_234695


namespace radio_lowest_price_rank_l234_234745

-- Definitions based on the conditions
def total_items : ℕ := 38
def radio_highest_rank : ℕ := 16

-- The theorem statement
theorem radio_lowest_price_rank : (total_items - (radio_highest_rank - 1)) = 24 := by
  sorry

end radio_lowest_price_rank_l234_234745


namespace faucet_draining_time_l234_234718

theorem faucet_draining_time 
  (all_faucets_drain_time : ℝ)
  (n : ℝ) 
  (first_faucet_time : ℝ) 
  (last_faucet_time : ℝ) 
  (avg_drain_time : ℝ)
  (condition_1 : all_faucets_drain_time = 24)
  (condition_2 : last_faucet_time = first_faucet_time / 7)
  (condition_3 : avg_drain_time = (first_faucet_time + last_faucet_time) / 2)
  (condition_4 : avg_drain_time = 24) : 
  first_faucet_time = 42 := 
by
  sorry

end faucet_draining_time_l234_234718


namespace lottery_winning_situations_l234_234374

theorem lottery_winning_situations :
  let num_tickets := 8
  let first_prize := 1
  let second_prize := 1
  let third_prize := 1
  let non_winning := 5
  let customers := 4
  let tickets_per_customer := 2
  let total_ways := 24 + 36
  total_ways = 60 :=
by
  let num_tickets := 8
  let first_prize := 1
  let second_prize := 1
  let third_prize := 1
  let non_winning := 5
  let customers := 4
  let tickets_per_customer := 2
  let total_ways := 24 + 36

  -- Skipping proof steps
  sorry

end lottery_winning_situations_l234_234374


namespace solve_for_y_l234_234431

theorem solve_for_y (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : y = 1 :=
by
  sorry

end solve_for_y_l234_234431


namespace hyperbola_k_range_l234_234436

theorem hyperbola_k_range {k : ℝ} 
  (h : ∀ x y : ℝ, x^2 + (k-1)*y^2 = k+1 → (k > -1 ∧ k < 1)) : 
  -1 < k ∧ k < 1 :=
by 
  sorry

end hyperbola_k_range_l234_234436


namespace book_total_pages_l234_234947

theorem book_total_pages (x : ℝ) 
  (h1 : ∀ d1 : ℝ, d1 = x * (1/6) + 10)
  (h2 : ∀ remaining1 : ℝ, remaining1 = x - d1)
  (h3 : ∀ d2 : ℝ, d2 = remaining1 * (1/5) + 12)
  (h4 : ∀ remaining2 : ℝ, remaining2 = remaining1 - d2)
  (h5 : ∀ d3 : ℝ, d3 = remaining2 * (1/4) + 14)
  (h6 : ∀ remaining3 : ℝ, remaining3 = remaining2 - d3)
  (h7 : remaining3 = 52) : x = 169 := sorry

end book_total_pages_l234_234947


namespace smallest_integer_cube_ends_in_576_l234_234939

theorem smallest_integer_cube_ends_in_576 : ∃ n : ℕ, n > 0 ∧ n^3 % 1000 = 576 ∧ ∀ m : ℕ, m > 0 → m^3 % 1000 = 576 → m ≥ n := 
by
  sorry

end smallest_integer_cube_ends_in_576_l234_234939


namespace lattice_point_condition_l234_234156

theorem lattice_point_condition (b : ℚ) :
  (∀ (m : ℚ), (1 / 3 < m ∧ m < b) →
    ∀ x : ℤ, (0 < x ∧ x ≤ 200) →
      ¬ ∃ y : ℤ, y = m * x + 3) →
  b = 68 / 203 := 
sorry

end lattice_point_condition_l234_234156


namespace sum_of_roots_l234_234919

-- Define the quadratic equation
def quadratic_eq (a b c x : ℝ) : Prop :=
  a * x^2 + b * x + c = 0

-- Prove that the sum of the roots of the given quadratic equation is 6
theorem sum_of_roots :
  (quadratic_eq 1 (-6) 9) x → (quadratic_eq 1 (-6) 9) y → x ≠ y → x + y = 6 :=
by
  sorry

end sum_of_roots_l234_234919


namespace relationship_between_y_l234_234085

theorem relationship_between_y
  (m y₁ y₂ y₃ : ℝ)
  (hA : y₁ = -(-1)^2 + 2 * -1 + m)
  (hB : y₂ = -(1)^2 + 2 * 1 + m)
  (hC : y₃ = -(2)^2 + 2 * 2 + m) :
  y₁ < y₃ ∧ y₃ < y₂ :=
sorry

end relationship_between_y_l234_234085


namespace lumber_cut_length_l234_234273

-- Define lengths of the pieces
def length_W : ℝ := 5
def length_X : ℝ := 3
def length_Y : ℝ := 5
def length_Z : ℝ := 4

-- Define distances from line M to the left end of the pieces
def distance_X : ℝ := 3
def distance_Y : ℝ := 2
def distance_Z : ℝ := 1.5

-- Define the total length of the pieces
def total_length : ℝ := 17

-- Define the length per side when cut by L
def length_per_side : ℝ := 8.5

theorem lumber_cut_length :
    (∃ (d : ℝ), 4 * d - 6.5 = 8.5 ∧ d = 3.75) :=
by
  sorry

end lumber_cut_length_l234_234273


namespace sequence_correct_l234_234559

def seq_formula (n : ℕ) : ℚ := 3/2 + (-1)^n * 11/2

theorem sequence_correct (n : ℕ) :
  (n % 2 = 0 ∧ seq_formula n = 7) ∨ (n % 2 = 1 ∧ seq_formula n = -4) :=
by
  sorry

end sequence_correct_l234_234559


namespace contact_prob_correct_l234_234493

-- Define the conditions.
def m : ℕ := 6
def n : ℕ := 7
variable (p : ℝ)

-- Define the probability computation.
def prob_contact : ℝ := 1 - (1 - p)^(m * n)

-- Formal statement of the problem.
theorem contact_prob_correct : prob_contact p = 1 - (1 - p)^42 := by
  sorry

end contact_prob_correct_l234_234493


namespace width_of_box_l234_234117

theorem width_of_box 
(length depth num_cubes : ℕ)
(h_length : length = 49)
(h_depth : depth = 14)
(h_num_cubes : num_cubes = 84)
: ∃ width : ℕ, width = 42 := 
sorry

end width_of_box_l234_234117


namespace simplify_expression_l234_234612

variables {a b c : ℝ}
variable (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0)
variable (h₃ : b - 2 / c ≠ 0)

theorem simplify_expression :
  (a - 2 / b) / (b - 2 / c) = c / b :=
sorry

end simplify_expression_l234_234612


namespace circle_diameter_equality_l234_234476

theorem circle_diameter_equality (r d : ℝ) (h₁ : d = 2 * r) (h₂ : π * d = π * r^2) : d = 4 :=
by
  sorry

end circle_diameter_equality_l234_234476


namespace quantiville_jacket_junction_l234_234262

theorem quantiville_jacket_junction :
  let sales_tax_rate := 0.07
  let original_price := 120.0
  let discount := 0.25
  let amy_total := (original_price * (1 + sales_tax_rate)) * (1 - discount)
  let bob_total := (original_price * (1 - discount)) * (1 + sales_tax_rate)
  let carla_total := ((original_price * (1 + sales_tax_rate)) * (1 - discount)) * (1 + sales_tax_rate)
  (carla_total - amy_total) = 6.744 :=
by
  sorry

end quantiville_jacket_junction_l234_234262


namespace rich_walks_ratio_is_2_l234_234048

-- Define the conditions in the problem
def house_to_sidewalk : ℕ := 20
def sidewalk_to_end : ℕ := 200
def total_distance_walked : ℕ := 1980
def ratio_after_left_to_so_far (x : ℕ) : ℕ := (house_to_sidewalk + sidewalk_to_end) * x / (house_to_sidewalk + sidewalk_to_end)

-- Main theorem to prove the ratio is 2:1
theorem rich_walks_ratio_is_2 (x : ℕ) (h : 2 * ((house_to_sidewalk + sidewalk_to_end) * 2 + house_to_sidewalk + sidewalk_to_end / 2 * 3 ) = total_distance_walked) :
  ratio_after_left_to_so_far x = 2 :=
by
  sorry

end rich_walks_ratio_is_2_l234_234048


namespace length_more_than_breadth_l234_234101

theorem length_more_than_breadth (b x : ℝ) (h1 : b + x = 61) (h2 : 26.50 * (4 * b + 2 * x) = 5300) : x = 22 :=
by
  sorry

end length_more_than_breadth_l234_234101


namespace tailor_cut_difference_l234_234163

theorem tailor_cut_difference :
  (7 / 8 + 11 / 12) - (5 / 6 + 3 / 4) = 5 / 24 :=
by
  sorry

end tailor_cut_difference_l234_234163


namespace money_spent_on_ferris_wheel_l234_234070

-- Conditions
def initial_tickets : ℕ := 6
def remaining_tickets : ℕ := 3
def ticket_cost : ℕ := 9

-- Prove that the money spent during the ferris wheel ride is 27 dollars
theorem money_spent_on_ferris_wheel : (initial_tickets - remaining_tickets) * ticket_cost = 27 := by
  sorry

end money_spent_on_ferris_wheel_l234_234070


namespace largest_x_satisfying_abs_eq_largest_x_is_correct_l234_234468

theorem largest_x_satisfying_abs_eq (x : ℝ) (h : |x - 5| = 12) : x ≤ 17 :=
by
  sorry

noncomputable def largest_x : ℝ := 17

theorem largest_x_is_correct (x : ℝ) (h : |x - 5| = 12) : x ≤ largest_x :=
largest_x_satisfying_abs_eq x h

end largest_x_satisfying_abs_eq_largest_x_is_correct_l234_234468


namespace simplify_expression_l234_234032

variable (x : ℝ)

theorem simplify_expression (h : x ≠ 1) : (x^2 + 1) / (x - 1) - 2 * x / (x - 1) = x - 1 :=
by sorry

end simplify_expression_l234_234032


namespace odd_function_inequality_solution_l234_234836

noncomputable def f (x : ℝ) : ℝ := if x > 0 then x - 2 else -(x - 2)

theorem odd_function_inequality_solution :
  {x : ℝ | f x < 0} = {x : ℝ | x < -2} ∪ {x : ℝ | 0 < x ∧ x < 2} :=
by
  -- A placeholder for the actual proof
  sorry

end odd_function_inequality_solution_l234_234836


namespace pure_imaginary_a_zero_l234_234398

theorem pure_imaginary_a_zero (a : ℝ) (i : ℂ) (hi : i^2 = -1) :
  (z = (1 - (a:ℝ)^2 * i) / i) ∧ (∀ (z : ℂ), z.re = 0 → z = (0 : ℂ)) → a = 0 :=
by
  sorry

end pure_imaginary_a_zero_l234_234398


namespace number_of_plastic_bottles_l234_234696

-- Define the weights of glass and plastic bottles
variables (G P : ℕ)

-- Define the number of plastic bottles in the second scenario
variable (x : ℕ)

-- Define the conditions
def condition_1 := 3 * G = 600
def condition_2 := G = P + 150
def condition_3 := 4 * G + x * P = 1050

-- Proof that x is equal to 5 given the conditions
theorem number_of_plastic_bottles (h1 : condition_1 G) (h2 : condition_2 G P) (h3 : condition_3 G P x) : x = 5 :=
sorry

end number_of_plastic_bottles_l234_234696


namespace sum_of_possible_values_of_k_l234_234807

theorem sum_of_possible_values_of_k :
  (∀ j k : ℕ, 0 < j ∧ 0 < k → (1 / (j:ℚ)) + (1 / (k:ℚ)) = (1 / 5) → k = 6 ∨ k = 10 ∨ k = 30) ∧ 
  (46 = 6 + 10 + 30) :=
by
  sorry

end sum_of_possible_values_of_k_l234_234807


namespace rectangle_area_change_l234_234458

theorem rectangle_area_change (x : ℝ) :
  let L := 1 -- arbitrary non-zero value for length
  let W := 1 -- arbitrary non-zero value for width
  (1 + x / 100) * (1 - x / 100) = 1.01 -> x = 10 := 
by
  sorry

end rectangle_area_change_l234_234458


namespace find_smallest_m_l234_234513

theorem find_smallest_m : ∃ m : ℕ, m > 0 ∧ (790 * m ≡ 1430 * m [MOD 30]) ∧ ∀ n : ℕ, n > 0 ∧ (790 * n ≡ 1430 * n [MOD 30]) → m ≤ n :=
by
  sorry

end find_smallest_m_l234_234513


namespace area_of_L_shape_is_58_l234_234923

-- Define the dimensions of the large rectangle
def large_rectangle_length : ℕ := 10
def large_rectangle_width : ℕ := 7

-- Define the dimensions of the smaller rectangle to be removed
def small_rectangle_length : ℕ := 4
def small_rectangle_width : ℕ := 3

-- Define the area of the large rectangle
def area_large_rectangle : ℕ := large_rectangle_length * large_rectangle_width

-- Define the area of the small rectangle
def area_small_rectangle : ℕ := small_rectangle_length * small_rectangle_width

-- Define the area of the "L" shaped region
def area_L_shape : ℕ := area_large_rectangle - area_small_rectangle

-- Prove that the area of the "L" shaped region is 58 square units
theorem area_of_L_shape_is_58 : area_L_shape = 58 := by
  sorry

end area_of_L_shape_is_58_l234_234923


namespace quadratic_completion_l234_234838

theorem quadratic_completion (x : ℝ) :
  2 * x^2 + 3 * x + 1 = 0 ↔ 2 * (x + 3 / 4)^2 - 1 / 8 = 0 :=
by
  sorry

end quadratic_completion_l234_234838


namespace ending_number_of_SetB_l234_234676

-- Definition of Set A
def SetA : Set ℕ := {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}

-- Definition of Set B
def SetB_ends_at (n : ℕ) : Set ℕ := {i | 6 ≤ i ∧ i ≤ n}

-- The main theorem statement
theorem ending_number_of_SetB : ∃ n, SetA ∩ SetB_ends_at n = {6, 7, 8, 9, 10, 11, 12, 13, 14, 15} ∧ 10 ∈ SetA ∩ SetB_ends_at n := 
sorry

end ending_number_of_SetB_l234_234676


namespace percentage_more_than_l234_234136

variable (P Q : ℝ)

-- P gets 20% more than Q
def getsMoreThan (P Q : ℝ) : Prop :=
  P = 1.20 * Q

-- Q gets 20% less than P
def getsLessThan (Q P : ℝ) : Prop :=
  Q = 0.80 * P

theorem percentage_more_than :
  getsLessThan Q P → getsMoreThan P Q := 
sorry

end percentage_more_than_l234_234136


namespace find_x_l234_234833

theorem find_x (t : ℤ) : 
∃ x : ℤ, (x % 7 = 3) ∧ (x^2 % 49 = 44) ∧ (x^3 % 343 = 111) ∧ (x = 343 * t + 17) :=
sorry

end find_x_l234_234833


namespace find_middle_part_value_l234_234682

-- Define the ratios
def ratio1 := 1 / 2
def ratio2 := 1 / 4
def ratio3 := 1 / 8

-- Total sum
def total_sum := 120

-- Parts proportional to ratios
def part1 (x : ℝ) := x
def part2 (x : ℝ) := ratio1 * x
def part3 (x : ℝ) := ratio2 * x

-- Equation representing the sum of the parts equals to the total sum
def equation (x : ℝ) : Prop :=
  part1 x + part2 x / 2 + part2 x = x * (1 + ratio1 + ratio2)

-- Defining the middle part
def middle_part (x : ℝ) := ratio1 * x

theorem find_middle_part_value :
  ∃ x : ℝ, equation x ∧ middle_part x = 34.2857 := sorry

end find_middle_part_value_l234_234682


namespace evaluate_f_g_f_l234_234705

-- Define f(x)
def f (x : ℝ) : ℝ := 4 * x + 4

-- Define g(x)
def g (x : ℝ) : ℝ := x^2 + 5 * x + 3

-- State the theorem we're proving
theorem evaluate_f_g_f : f (g (f 3)) = 1360 := by
  sorry

end evaluate_f_g_f_l234_234705


namespace exists_smallest_positive_period_even_function_l234_234689

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def period (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

noncomputable def functions : List (ℝ → ℝ) :=
  [
    (λ x => Real.sin (2 * x + Real.pi / 2)),
    (λ x => Real.cos (2 * x + Real.pi / 2)),
    (λ x => Real.sin (2 * x) + Real.cos (2 * x)),
    (λ x => Real.sin x + Real.cos x)
  ]

def smallest_positive_period_even_function : ℝ → Prop :=
  λ T => ∃ f ∈ functions, is_even_function f ∧ period f T ∧ T > 0

theorem exists_smallest_positive_period_even_function :
  smallest_positive_period_even_function Real.pi :=
sorry

end exists_smallest_positive_period_even_function_l234_234689


namespace circle_radius_l234_234554

theorem circle_radius (r : ℝ) (x y : ℝ) (h₁ : x = π * r ^ 2) (h₂ : y = 2 * π * r - 6) (h₃ : x + y = 94 * π) : 
  r = 10 :=
sorry

end circle_radius_l234_234554


namespace geometric_sequence_four_seven_prod_l234_234688

def is_geometric_sequence (a : ℕ → ℝ) :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_four_seven_prod
    (a : ℕ → ℝ)
    (h_geom : is_geometric_sequence a)
    (h_roots : ∀ x, 3 * x^2 - 2 * x - 6 = 0 → (x = a 1 ∨ x = a 10)) :
  a 4 * a 7 = -2 := 
sorry

end geometric_sequence_four_seven_prod_l234_234688


namespace find_a_l234_234613

noncomputable def parabola_eq (a b c : ℤ) (x : ℤ) : ℤ :=
  a * x^2 + b * x + c

theorem find_a (a b c : ℤ)
  (h_vertex : ∀ x, parabola_eq a b c x = a * (x - 2)^2 + 5) 
  (h_point : parabola_eq a b c 1 = 6) :
  a = 1 := 
by 
  sorry

end find_a_l234_234613


namespace distance_from_hut_to_station_l234_234220

variable (t s : ℝ)

theorem distance_from_hut_to_station
  (h1 : s / 4 = t + 3 / 4)
  (h2 : s / 6 = t - 1 / 2) :
  s = 15 := by
  sorry

end distance_from_hut_to_station_l234_234220


namespace sum_of_first_ten_terms_seq_l234_234823

def a₁ : ℤ := -5
def d : ℤ := 6
def n : ℕ := 10

theorem sum_of_first_ten_terms_seq : (n * (a₁ + a₁ + (n - 1) * d)) / 2 = 220 :=
by
  sorry

end sum_of_first_ten_terms_seq_l234_234823


namespace coordinates_of_B_l234_234247

structure Point where
  x : Float
  y : Float

def symmetricWithRespectToY (A B : Point) : Prop :=
  B.x = -A.x ∧ B.y = A.y

theorem coordinates_of_B (A B : Point) 
  (hA : A.x = 2 ∧ A.y = -5)
  (h_sym : symmetricWithRespectToY A B) :
  B.x = -2 ∧ B.y = -5 :=
by
  sorry

end coordinates_of_B_l234_234247


namespace triangle_perimeter_not_78_l234_234704

theorem triangle_perimeter_not_78 (x : ℝ) (h1 : 11 < x) (h2 : x < 37) : 13 + 24 + x ≠ 78 :=
by
  -- Using the given conditions to show the perimeter is not 78
  intro h
  have h3 : 48 < 13 + 24 + x := by linarith
  have h4 : 13 + 24 + x < 74 := by linarith
  linarith

end triangle_perimeter_not_78_l234_234704


namespace sum_of_midpoints_l234_234206

theorem sum_of_midpoints (d e f : ℝ) (h : d + e + f = 15) :
  (d + e) / 2 + (d + f) / 2 + (e + f) / 2 = 15 :=
by sorry

end sum_of_midpoints_l234_234206


namespace factor_expression_l234_234492

theorem factor_expression (y : ℝ) : 84 * y ^ 13 + 210 * y ^ 26 = 42 * y ^ 13 * (2 + 5 * y ^ 13) :=
by sorry

end factor_expression_l234_234492


namespace closest_multiple_of_17_to_2502_is_2499_l234_234984

def isNearestMultipleOf17 (m n : ℤ) : Prop :=
  ∃ k : ℤ, 17 * k = n ∧ abs (m - n) ≤ abs (m - 17 * (k + 1)) ∧ abs (m - n) ≤ abs (m - 17 * (k - 1))

theorem closest_multiple_of_17_to_2502_is_2499 :
  isNearestMultipleOf17 2502 2499 :=
sorry

end closest_multiple_of_17_to_2502_is_2499_l234_234984


namespace adjacent_number_in_grid_l234_234392

def adjacent_triangle_number (k n: ℕ) : ℕ :=
  if k % 2 = 1 then n - k else n + k

theorem adjacent_number_in_grid (n : ℕ) (bound: n = 350) :
  let k := Nat.ceil (Real.sqrt n)
  let m := (k * k) - n
  k = 19 ∧ m = 19 →
  adjacent_triangle_number k n = 314 :=
by
  sorry

end adjacent_number_in_grid_l234_234392


namespace commission_rate_correct_l234_234863

variables (weekly_earnings : ℕ) (commission : ℕ) (total_earnings : ℕ) (sales : ℕ) (commission_rate : ℕ)

-- Base earnings per week without commission
def base_earnings : ℕ := 190

-- Total earnings target
def earnings_goal : ℕ := 500

-- Minimum sales required to meet the earnings goal
def sales_needed : ℕ := 7750

-- Definition of the commission as needed to meet the goal
def needed_commission : ℕ := earnings_goal - base_earnings

-- Definition of the actual commission rate
def commission_rate_per_sale : ℕ := (needed_commission * 100) / sales_needed

-- Proof goal: Show that commission_rate_per_sale is 4
theorem commission_rate_correct : commission_rate_per_sale = 4 :=
by
  sorry

end commission_rate_correct_l234_234863


namespace combined_total_l234_234161

-- Definitions for the problem conditions
def marks_sandcastles : ℕ := 20
def towers_per_marks_sandcastle : ℕ := 10

def jeffs_multiplier : ℕ := 3
def towers_per_jeffs_sandcastle : ℕ := 5

-- Definitions derived from conditions
def jeffs_sandcastles : ℕ := jeffs_multiplier * marks_sandcastles
def marks_towers : ℕ := marks_sandcastles * towers_per_marks_sandcastle
def jeffs_towers : ℕ := jeffs_sandcastles * towers_per_jeffs_sandcastle

-- Question translated to a Lean theorem
theorem combined_total : 
  (marks_sandcastles + jeffs_sandcastles) + (marks_towers + jeffs_towers) = 580 :=
by
  -- The proof would go here
  sorry

end combined_total_l234_234161


namespace sum_of_exponents_l234_234570

-- Definition of Like Terms
def like_terms (m n : ℕ) : Prop :=
  m = 3 ∧ n = 2

-- Theorem statement
theorem sum_of_exponents (m n : ℕ) (h : like_terms m n) : m + n = 5 :=
sorry

end sum_of_exponents_l234_234570


namespace average_of_consecutive_odds_is_24_l234_234535

theorem average_of_consecutive_odds_is_24 (a b c d : ℤ) 
  (h1 : a < b) (h2 : b < c) (h3 : c < d) 
  (h4 : d = 27) 
  (h5 : b = d - 2) (h6 : c = d - 4) (h7 : a = d - 6) 
  (h8 : ∀ x : ℤ, x % 2 = 1) :
  ((a + b + c + d) / 4) = 24 :=
by {
  sorry
}

end average_of_consecutive_odds_is_24_l234_234535


namespace domain_of_f_l234_234749

open Real

noncomputable def f (x : ℝ) : ℝ := (log (2 * x - x^2)) / (x - 1)

theorem domain_of_f (x : ℝ) : (0 < x ∧ x < 1) ∨ (1 < x ∧ x < 2) ↔ (2 * x - x^2 > 0 ∧ x ≠ 1) := by
  sorry

end domain_of_f_l234_234749


namespace solve_equation_l234_234466

theorem solve_equation :
  ∃ y : ℚ, 2 * (y - 3) - 6 * (2 * y - 1) = -3 * (2 - 5 * y) ↔ y = 6 / 25 :=
by
  sorry

end solve_equation_l234_234466


namespace tangent_line_through_P_is_correct_l234_234013

-- Define the point P
def P : ℝ × ℝ := (2, 4)

-- Define the circle C
def C (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 5

-- Define the tangent line equation to prove
def tangent_line (x y : ℝ) : Prop := x + 2 * y - 10 = 0

-- Problem statement in Lean 4
theorem tangent_line_through_P_is_correct :
  C P.1 P.2 → tangent_line P.1 P.2 :=
by
  intros hC
  sorry

end tangent_line_through_P_is_correct_l234_234013


namespace total_fruit_count_l234_234999

-- Define the number of oranges
def oranges : ℕ := 6

-- Define the number of apples based on the number of oranges
def apples : ℕ := oranges - 2

-- Define the number of bananas based on the number of apples
def bananas : ℕ := 3 * apples

-- Define the number of peaches based on the number of bananas
def peaches : ℕ := bananas / 2

-- Define the total number of fruits in the basket
def total_fruits : ℕ := oranges + apples + bananas + peaches

-- Prove that the total number of pieces of fruit in the basket is 28
theorem total_fruit_count : total_fruits = 28 := by
  sorry

end total_fruit_count_l234_234999


namespace angle_between_lines_l234_234349

theorem angle_between_lines :
  let L1 := {p : ℝ × ℝ | p.1 = -3}  -- Line x+3=0
  let L2 := {p: ℝ × ℝ | p.1 + p.2 - 3 = 0}  -- Line x+y-3=0
  ∃ θ : ℝ, 0 < θ ∧ θ < 180 ∧ θ = 45 :=
sorry

end angle_between_lines_l234_234349


namespace remainder_of_division_l234_234760

def p (x : ℝ) : ℝ := 8*x^4 - 10*x^3 + 16*x^2 - 18*x + 5
def d (x : ℝ) : ℝ := 4*x - 8

theorem remainder_of_division :
  (p 2) = 81 :=
by
  sorry

end remainder_of_division_l234_234760


namespace smallest_sector_angle_divided_circle_l234_234113

theorem smallest_sector_angle_divided_circle : ∃ a d : ℕ, 
  (2 * a + 7 * d = 90) ∧ 
  (8 * (a + (a + 7 * d)) / 2 = 360) ∧ 
  a = 38 := 
by
  sorry

end smallest_sector_angle_divided_circle_l234_234113


namespace proof_problem_l234_234805

theorem proof_problem 
  (a b c : ℝ) 
  (h1 : ∀ x, (x < -4 ∨ (23 ≤ x ∧ x ≤ 27)) ↔ ((x - a) * (x - b) / (x - c) ≤ 0))
  (h2 : a < b) : 
  a + 2 * b + 3 * c = 65 :=
sorry

end proof_problem_l234_234805


namespace symmetric_line_eq_l234_234335

theorem symmetric_line_eq (x y : ℝ) (h : 3 * x + 4 * y + 5 = 0) : 3 * x - 4 * y + 5 = 0 :=
sorry

end symmetric_line_eq_l234_234335


namespace initial_distances_l234_234240

theorem initial_distances (x y : ℝ) 
  (h1: x^2 + y^2 = 400)
  (h2: (x - 6)^2 + (y - 8)^2 = 100) : 
  x = 12 ∧ y = 16 := 
by 
  sorry

end initial_distances_l234_234240


namespace sum_of_all_x_l234_234902

theorem sum_of_all_x (x1 x2 : ℝ) (h1 : (x1 + 5)^2 = 81) (h2 : (x2 + 5)^2 = 81) : x1 + x2 = -10 :=
by
  sorry

end sum_of_all_x_l234_234902


namespace range_of_a_l234_234901

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - a * x + 2 * a > 0) ↔ (0 < a ∧ a < 8) :=
by
  sorry

end range_of_a_l234_234901


namespace expression_evaluation_l234_234730

-- Define expression variable to ensure emphasis on conditions and calculations
def expression : ℤ := 9 - (8 + 7) * 6 + 5^2 - (4 * 3) + 2 - 1

theorem expression_evaluation : expression = -67 :=
by
  -- Use assumptions about the order of operations to conclude
  sorry

end expression_evaluation_l234_234730


namespace bridge_length_sufficient_l234_234230

structure Train :=
  (length : ℕ) -- length of the train in meters
  (speed : ℚ) -- speed of the train in km/hr

def speed_in_m_per_s (speed_in_km_per_hr : ℚ) : ℚ :=
  speed_in_km_per_hr * 1000 / 3600

noncomputable def length_of_bridge (train1 train2 : Train) : ℚ :=
  let train1_speed_m_per_s := speed_in_m_per_s train1.speed
  let train2_speed_m_per_s := speed_in_m_per_s train2.speed
  let relative_speed := train1_speed_m_per_s + train2_speed_m_per_s
  let total_length := train1.length + train2.length
  let time_to_pass := total_length / relative_speed
  let distance_train1 := train1_speed_m_per_s * time_to_pass
  let distance_train2 := train2_speed_m_per_s * time_to_pass
  distance_train1 + distance_train2

theorem bridge_length_sufficient (train1 train2 : Train) (h1 : train1.length = 200) (h2 : train1.speed = 60) (h3 : train2.length = 150) (h4 : train2.speed = 45) :
  length_of_bridge train1 train2 ≥ 350.04 :=
  by
  sorry

end bridge_length_sufficient_l234_234230


namespace smallest_digit_divisible_by_11_l234_234345

theorem smallest_digit_divisible_by_11 :
  ∃ (d : ℕ), d < 10 ∧ ∀ n : ℕ, (n + 45000 + 1000 + 457 + d) % 11 = 0 → d = 5 :=
by {
  sorry
}

end smallest_digit_divisible_by_11_l234_234345


namespace Brian_watch_animal_videos_l234_234558

theorem Brian_watch_animal_videos :
  let cat_video := 4
  let dog_video := 2 * cat_video
  let gorilla_video := 2 * (cat_video + dog_video)
  let elephant_video := cat_video + dog_video + gorilla_video
  let dolphin_video := cat_video + dog_video + gorilla_video + elephant_video
  let total_time := cat_video + dog_video + gorilla_video + elephant_video + dolphin_video
  total_time = 144 := by
{
  let cat_video := 4
  let dog_video := 2 * cat_video
  let gorilla_video := 2 * (cat_video + dog_video)
  let elephant_video := cat_video + dog_video + gorilla_video
  let dolphin_video := cat_video + dog_video + gorilla_video + elephant_video
  let total_time := cat_video + dog_video + gorilla_video + elephant_video + dolphin_video
  have h1 : total_time = (4 + 8 + 24 + 36 + 72) := sorry
  exact h1
}

end Brian_watch_animal_videos_l234_234558


namespace ara_final_height_is_59_l234_234651

noncomputable def initial_shea_height : ℝ := 51.2
noncomputable def initial_ara_height : ℝ := initial_shea_height + 4
noncomputable def final_shea_height : ℝ := 64
noncomputable def shea_growth : ℝ := final_shea_height - initial_shea_height
noncomputable def ara_growth : ℝ := shea_growth / 3
noncomputable def final_ara_height : ℝ := initial_ara_height + ara_growth

theorem ara_final_height_is_59 :
  final_ara_height = 59 := by
  sorry

end ara_final_height_is_59_l234_234651


namespace prime_integer_roots_l234_234796

theorem prime_integer_roots (p : ℕ) (hp : Prime p) 
  (hroots : ∀ (x1 x2 : ℤ), x1 * x2 = -512 * p ∧ x1 + x2 = -p) : p = 2 :=
by
  -- Proof omitted
  sorry

end prime_integer_roots_l234_234796


namespace cone_volume_l234_234110

theorem cone_volume (r l: ℝ) (r_eq : r = 2) (l_eq : l = 4) (h : ℝ) (h_eq : h = 2 * Real.sqrt 3) :
  (1 / 3) * π * r^2 * h = (8 * Real.sqrt 3 * π) / 3 :=
by
  -- Sorry to skip the proof
  sorry

end cone_volume_l234_234110


namespace total_pay_is_880_l234_234903

theorem total_pay_is_880 (X_pay Y_pay : ℝ) 
  (hY : Y_pay = 400)
  (hX : X_pay = 1.2 * Y_pay):
  X_pay + Y_pay = 880 :=
by
  sorry

end total_pay_is_880_l234_234903


namespace find_replaced_man_weight_l234_234319

variable (n : ℕ) (new_weight old_avg_weight : ℝ) (weight_inc : ℝ) (W : ℝ)

theorem find_replaced_man_weight 
  (h1 : n = 8) 
  (h2 : new_weight = 68) 
  (h3 : weight_inc = 1) 
  (h4 : 8 * (old_avg_weight + 1) = 8 * old_avg_weight + (new_weight - W)) 
  : W = 60 :=
by
  sorry

end find_replaced_man_weight_l234_234319


namespace number_of_five_digit_numbers_without_repeating_digits_with_two_adjacent_odds_is_72_l234_234795

def five_digit_number_count : Nat :=
  -- Number of ways to select and arrange odd digits in two groups
  let group_odd_digits := (Nat.choose 3 2) * (Nat.factorial 2)
  -- Number of ways to arrange the even digits
  let arrange_even_digits := Nat.factorial 2
  -- Number of ways to insert two groups of odd digits into the gaps among even digits
  let insert_odd_groups := (Nat.factorial 3)
  -- Total ways
  group_odd_digits * arrange_even_digits * arrange_even_digits * insert_odd_groups

theorem number_of_five_digit_numbers_without_repeating_digits_with_two_adjacent_odds_is_72 :
  five_digit_number_count = 72 :=
by
  -- Placeholder for proof
  sorry

end number_of_five_digit_numbers_without_repeating_digits_with_two_adjacent_odds_is_72_l234_234795


namespace sum_of_three_numbers_l234_234017

theorem sum_of_three_numbers (A B C : ℕ) 
  (h1 : B = 30)
  (h2 : A * 3 = 2 * B)
  (h3 : C * 5 = 8 * B) : 
  A + B + C = 98 :=
by
  sorry

end sum_of_three_numbers_l234_234017


namespace planting_scheme_correct_l234_234766

-- Setting up the problem as the conditions given
def types_of_seeds := ["peanuts", "Chinese cabbage", "potatoes", "corn", "wheat", "apples"]

def first_plot_seeds := ["corn", "apples"]

def planting_schemes_count : ℕ :=
  let choose_first_plot := 2  -- C(2, 1), choosing either "corn" or "apples" for the first plot
  let remaining_seeds := 5  -- 6 - 1 = 5 remaining seeds after choosing for the first plot
  let arrangements_remaining := 5 * 4 * 3  -- A(5, 3), arrangements of 3 plots from 5 remaining seeds
  choose_first_plot * arrangements_remaining

theorem planting_scheme_correct : planting_schemes_count = 120 := by
  sorry

end planting_scheme_correct_l234_234766


namespace find_n_22_or_23_l234_234537

theorem find_n_22_or_23 (n : ℕ) : 
  (∃ (sol_count : ℕ), sol_count = 30 ∧ (∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ 3 * x + 2 * y + 4 * z = n)) → 
  (n = 22 ∨ n = 23) := 
sorry

end find_n_22_or_23_l234_234537


namespace Seohyeon_l234_234165

-- Define the distances in their respective units
def d_Kunwoo_km : ℝ := 3.97
def d_Seohyeon_m : ℝ := 4028

-- Convert Kunwoo's distance to meters
def d_Kunwoo_m : ℝ := d_Kunwoo_km * 1000

-- The main theorem we need to prove
theorem Seohyeon's_distance_longer_than_Kunwoo's :
  d_Seohyeon_m > d_Kunwoo_m :=
by
  sorry

end Seohyeon_l234_234165


namespace verify_total_amount_spent_by_mary_l234_234332

def shirt_price : Float := 13.04
def shirt_sales_tax_rate : Float := 0.07

def jacket_original_price_gbp : Float := 15.34
def jacket_discount_rate : Float := 0.20
def jacket_sales_tax_rate : Float := 0.085
def conversion_rate_usd_per_gbp : Float := 1.28

def scarf_price : Float := 7.90
def hat_price : Float := 9.13
def hat_scarf_sales_tax_rate : Float := 0.065

def total_amount_spent_by_mary : Float :=
  let shirt_total := shirt_price * (1 + shirt_sales_tax_rate)
  let jacket_discounted := jacket_original_price_gbp * (1 - jacket_discount_rate)
  let jacket_total_gbp := jacket_discounted * (1 + jacket_sales_tax_rate)
  let jacket_total_usd := jacket_total_gbp * conversion_rate_usd_per_gbp
  let hat_scarf_combined_price := scarf_price + hat_price
  let hat_scarf_total := hat_scarf_combined_price * (1 + hat_scarf_sales_tax_rate)
  shirt_total + jacket_total_usd + hat_scarf_total

theorem verify_total_amount_spent_by_mary : total_amount_spent_by_mary = 49.13 :=
by sorry

end verify_total_amount_spent_by_mary_l234_234332


namespace xyz_ineq_l234_234646

theorem xyz_ineq (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x * y + y * z + z * x = 1) : 
  x * y * z * (x + y + z) ≤ 1 / 3 := 
sorry

end xyz_ineq_l234_234646


namespace smallest_k_for_64k_greater_than_6_l234_234792

theorem smallest_k_for_64k_greater_than_6 : ∃ (k : ℕ), 64 ^ k > 6 ∧ ∀ m : ℕ, m < k → 64 ^ m ≤ 6 :=
by
  use 1
  sorry

end smallest_k_for_64k_greater_than_6_l234_234792


namespace unoccupied_volume_of_tank_l234_234183

theorem unoccupied_volume_of_tank (length width height : ℝ) (num_marbles : ℕ) (marble_radius : ℝ) (fill_fraction : ℝ) :
    length = 12 → width = 12 → height = 15 → num_marbles = 5 → marble_radius = 1.5 → fill_fraction = 1/3 →
    (length * width * height * (1 - fill_fraction) - num_marbles * (4 / 3 * Real.pi * marble_radius^3) = 1440 - 22.5 * Real.pi) :=
by
  intros
  sorry

end unoccupied_volume_of_tank_l234_234183


namespace frost_time_with_sprained_wrist_l234_234610

-- Definitions
def normal_time_per_cake : ℕ := 5
def additional_time_for_10_cakes : ℕ := 30
def normal_time_for_10_cakes : ℕ := 10 * normal_time_per_cake
def sprained_time_for_10_cakes : ℕ := normal_time_for_10_cakes + additional_time_for_10_cakes

-- Theorems
theorem frost_time_with_sprained_wrist : ∀ x : ℕ, 
  (10 * x = sprained_time_for_10_cakes) ↔ (x = 8) := 
sorry

end frost_time_with_sprained_wrist_l234_234610


namespace repeating_decimal_to_fraction_l234_234125

theorem repeating_decimal_to_fraction : (0.36 : ℝ) = (11 / 30 : ℝ) :=
sorry

end repeating_decimal_to_fraction_l234_234125


namespace total_sum_spent_l234_234851

theorem total_sum_spent (b gift : ℝ) (friends tanya : ℕ) (extra_payment : ℝ)
  (h1 : friends = 10)
  (h2 : tanya = 1)
  (h3 : extra_payment = 3)
  (h4 : gift = 15)
  (h5 : b = 270)
  : (b + gift) = 285 :=
by {
  -- Given:
  -- friends = 10 (number of dinner friends),
  -- tanya = 1 (Tanya who forgot to pay),
  -- extra_payment = 3 (extra payment by each of the remaining 9 friends),
  -- gift = 15 (cost of the gift),
  -- b = 270 (total bill for the dinner excluding the gift),

  -- We need to prove:
  -- total sum spent by the group is $285, i.e., (b + gift) = 285

  sorry 
}

end total_sum_spent_l234_234851


namespace indigo_restaurant_average_rating_l234_234496

theorem indigo_restaurant_average_rating :
  let n_5stars := 6
  let n_4stars := 7
  let n_3stars := 4
  let n_2stars := 1
  let total_reviews := 18
  let total_stars := n_5stars * 5 + n_4stars * 4 + n_3stars * 3 + n_2stars * 2
  (total_stars / total_reviews : ℝ) = 4 :=
by
  sorry

end indigo_restaurant_average_rating_l234_234496


namespace convex_polygon_triangles_impossible_l234_234716

theorem convex_polygon_triangles_impossible :
  ∀ (a b c : ℕ), 2016 + 2 * b + c - 2014 = 0 → a + b + c = 2014 → a = 1007 → false :=
sorry

end convex_polygon_triangles_impossible_l234_234716


namespace quadratic_solution_l234_234930

theorem quadratic_solution (x : ℝ) (h : 2 * x ^ 2 - 2 = 0) : x = 1 ∨ x = -1 :=
sorry

end quadratic_solution_l234_234930


namespace max_temp_range_l234_234424

-- Definitions based on given conditions
def average_temp : ℤ := 40
def lowest_temp : ℤ := 30

-- Total number of days
def days : ℕ := 5

-- Given that the average temperature and lowest temperature are provided, prove the maximum range.
theorem max_temp_range 
  (avg_temp_eq : (average_temp * days) = 200)
  (temp_min : lowest_temp = 30) : 
  ∃ max_temp : ℤ, max_temp - lowest_temp = 50 :=
by
  -- Assume maximum temperature
  let max_temp := 80
  have total_sum := (average_temp * days)
  have min_occurrences := 3 * lowest_temp
  have highest_temp := total_sum - min_occurrences - lowest_temp
  have range := highest_temp - lowest_temp
  use max_temp
  sorry

end max_temp_range_l234_234424


namespace ticket_cost_difference_l234_234042

noncomputable def total_cost_adults (tickets : ℕ) (price : ℝ) : ℝ := tickets * price
noncomputable def total_cost_children (tickets : ℕ) (price : ℝ) : ℝ := tickets * price
noncomputable def total_tickets (adults : ℕ) (children : ℕ) : ℕ := adults + children
noncomputable def discount (threshold : ℕ) (discount_rate : ℝ) (cost : ℝ) (tickets : ℕ) : ℝ :=
  if tickets > threshold then cost * discount_rate else 0
noncomputable def final_cost (initial_cost : ℝ) (discount : ℝ) : ℝ := initial_cost - discount
noncomputable def proportional_discount (partial_cost : ℝ) (total_cost : ℝ) (total_discount : ℝ) : ℝ :=
  (partial_cost / total_cost) * total_discount
noncomputable def difference (cost1 : ℝ) (cost2 : ℝ) : ℝ := cost1 - cost2

theorem ticket_cost_difference :
  let adult_tickets := 9
  let children_tickets := 7
  let adult_price := 11
  let children_price := 7
  let discount_rate := 0.15
  let discount_threshold := 10
  let total_adult_cost := total_cost_adults adult_tickets adult_price
  let total_children_cost := total_cost_children children_tickets children_price
  let all_tickets := total_tickets adult_tickets children_tickets
  let initial_total_cost := total_adult_cost + total_children_cost
  let total_discount := discount discount_threshold discount_rate initial_total_cost all_tickets
  let final_total_cost := final_cost initial_total_cost total_discount
  let adult_discount := proportional_discount total_adult_cost initial_total_cost total_discount
  let children_discount := proportional_discount total_children_cost initial_total_cost total_discount
  let final_adult_cost := final_cost total_adult_cost adult_discount
  let final_children_cost := final_cost total_children_cost children_discount
  difference final_adult_cost final_children_cost = 42.52 := by
  sorry

end ticket_cost_difference_l234_234042


namespace positive_integer_x_l234_234933

theorem positive_integer_x (x : ℕ) (hx : 15 * x = x^2 + 56) : x = 8 := by
  sorry

end positive_integer_x_l234_234933


namespace arjun_becca_3_different_colors_l234_234027

open Classical

noncomputable def arjun_becca_probability : ℚ := 
  let arjun_initial := [2, 1, 1, 1] -- 2 red, 1 green, 1 yellow, 1 violet
  let becca_initial := [2, 1] -- 2 black, 1 orange
  
  -- possible cases represented as a list of probabilities
  let cases := [
    (2/5) * (1/4) * (3/5),    -- Case 1: Arjun does move a red ball to Becca, and then processes accordingly
    (3/5) * (1/2) * (1/5),    -- Case 2a: Arjun moves a non-red ball, followed by Becca moving a black ball, concluding in the defined manner
    (3/5) * (1/2) * (3/5)     -- Case 2b: Arjun moves a non-red ball, followed by Becca moving a non-black ball, again concluding appropriately
  ]
  
  -- sum of cases representing the total probability
  let total_probability := List.sum cases
  
  total_probability

theorem arjun_becca_3_different_colors : arjun_becca_probability = 3/10 := 
  by
    simp [arjun_becca_probability]
    sorry

end arjun_becca_3_different_colors_l234_234027


namespace quadratic_roots_value_l234_234245

theorem quadratic_roots_value (d : ℝ) 
  (h : ∀ x : ℝ, x^2 + 7 * x + d = 0 → x = (-7 + Real.sqrt d) / 2 ∨ x = (-7 - Real.sqrt d) / 2) : 
  d = 9.8 :=
by 
  sorry

end quadratic_roots_value_l234_234245


namespace find_d_value_l234_234420

theorem find_d_value 
  (x y d : ℝ)
  (h1 : 7^(3 * x - 1) * 3^(4 * y - 3) = 49^x * d^y)
  (h2 : x + y = 4) :
  d = 27 :=
by 
  sorry

end find_d_value_l234_234420


namespace arithmetic_sequence_S10_l234_234300

def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ :=
  a + n * d

def Sn (a d : ℤ) (n : ℕ) : ℤ :=
  n * a + (n * (n - 1)) / 2 * d

theorem arithmetic_sequence_S10 :
  ∃ (a d : ℤ), d ≠ 0 ∧ Sn a d 8 = 16 ∧
  (arithmetic_sequence a d 3)^2 = (arithmetic_sequence a d 2) * (arithmetic_sequence a d 6) ∧
  Sn a d 10 = 30 :=
by
  sorry

end arithmetic_sequence_S10_l234_234300


namespace std_dev_samples_l234_234516

def sample_A := [82, 84, 84, 86, 86, 86, 88, 88, 88, 88]
def sample_B := [84, 86, 86, 88, 88, 88, 90, 90, 90, 90]

noncomputable def std_dev (l : List ℕ) :=
  let n := l.length
  let mean := (l.sum : ℚ) / n
  let variance := (l.map (λ x => (x - mean) * (x - mean))).sum / n
  variance.sqrt

theorem std_dev_samples :
  std_dev sample_A = std_dev sample_B := 
sorry

end std_dev_samples_l234_234516


namespace batsman_average_runs_l234_234011

theorem batsman_average_runs
  (average_20_matches : ℕ → ℕ)
  (average_10_matches : ℕ → ℕ)
  (h1 : average_20_matches = 20 * 40)
  (h2 : average_10_matches = 10 * 13) :
  (average_20_matches + average_10_matches) / 30 = 31 := 
by 
  sorry

end batsman_average_runs_l234_234011


namespace gcd_840_1764_l234_234343

theorem gcd_840_1764 : gcd 840 1764 = 84 := 
by
  -- proof steps will go here
  sorry

end gcd_840_1764_l234_234343


namespace solve_logarithmic_system_l234_234572

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem solve_logarithmic_system :
  ∃ x y : ℝ, log_base 2 x + log_base 4 y = 4 ∧ log_base 4 x + log_base 2 y = 5 ∧ x = 4 ∧ y = 16 :=
by
  sorry

end solve_logarithmic_system_l234_234572


namespace fox_appropriation_l234_234722

variable (a m : ℕ) (n : ℕ) (y x : ℕ)

-- Definitions based on conditions
def fox_funds : Prop :=
  (m-1)*a + x = m*y ∧ 2*(m-1)*a + x = (m+1)*y ∧ 
  3*(m-1)*a + x = (m+2)*y ∧ n*(m-1)*a + x = (m+n-1)*y

-- Theorems to prove the final conclusions
theorem fox_appropriation (h : fox_funds a m n y x) : 
  y = (m-1)*a ∧ x = (m-1)^2*a :=
by
  sorry

end fox_appropriation_l234_234722


namespace reciprocal_of_neg_5_l234_234814

theorem reciprocal_of_neg_5 : (∃ r : ℚ, -5 * r = 1) ∧ r = -1 / 5 :=
by sorry

end reciprocal_of_neg_5_l234_234814


namespace largest_n_exists_l234_234969

theorem largest_n_exists (n x y z : ℕ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) : 
  n^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 3*x + 3*y + 3*z - 6 → 
  n ≤ 8 :=
sorry

end largest_n_exists_l234_234969


namespace isabel_pictures_l234_234553

theorem isabel_pictures
  (phone_pics : ℕ)
  (camera_pics : ℕ)
  (total_albums : ℕ)
  (h_phone_pics : phone_pics = 2)
  (h_camera_pics : camera_pics = 4)
  (h_total_albums : total_albums = 3) :
  (phone_pics + camera_pics) / total_albums = 2 :=
by
  sorry

end isabel_pictures_l234_234553


namespace pony_average_speed_l234_234043

theorem pony_average_speed
  (time_head_start : ℝ)
  (time_catch : ℝ)
  (horse_speed : ℝ)
  (distance_covered_by_horse : ℝ)
  (distance_covered_by_pony : ℝ)
  (pony's_head_start : ℝ)
  : (time_head_start = 3) → (time_catch = 4) → (horse_speed = 35) → 
    (distance_covered_by_horse = horse_speed * time_catch) → 
    (pony's_head_start = time_head_start * v) → 
    (distance_covered_by_pony = pony's_head_start + (v * time_catch)) → 
    (distance_covered_by_horse = distance_covered_by_pony) → v = 20 :=
  by 
    intros h1 h2 h3 h4 h5 h6 h7
    sorry

end pony_average_speed_l234_234043


namespace triangle_inequality_proof_l234_234333

variable (a b c : ℝ)

-- Condition that a, b, c are side lengths of a triangle
axiom triangle_inequality1 : a + b > c
axiom triangle_inequality2 : b + c > a
axiom triangle_inequality3 : c + a > b

-- Theorem stating the required inequality and the condition for equality
theorem triangle_inequality_proof :
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 ∧
  (a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) = 0 ↔ a = b ∧ b = c ∧ c = a) :=
sorry

end triangle_inequality_proof_l234_234333


namespace geom_seq_a6_value_l234_234187

variable {α : Type _} [LinearOrderedField α]

theorem geom_seq_a6_value (a : ℕ → α) (q : α) 
(h_geom : ∀ n, a (n + 1) = a n * q)
(h_cond : a 4 + a 8 = π) : 
a 6 * (a 2 + 2 * a 6 + a 10) = π^2 := by
  sorry

end geom_seq_a6_value_l234_234187


namespace a_must_not_be_zero_l234_234167

theorem a_must_not_be_zero (a b c d : ℝ) (h₁ : a / b < -3 * (c / d)) (h₂ : b ≠ 0) (h₃ : d ≠ 0) (h₄ : c = 2 * a) : a ≠ 0 :=
sorry

end a_must_not_be_zero_l234_234167


namespace range_of_m_l234_234600

noncomputable def prop_p (m : ℝ) : Prop :=
0 < m ∧ m < 1 / 3

noncomputable def prop_q (m : ℝ) : Prop :=
0 < m ∧ m < 15

theorem range_of_m (m : ℝ) : (prop_p m ∧ ¬ prop_q m) ∨ (¬ prop_p m ∧ prop_q m) ↔ 1 / 3 ≤ m ∧ m < 15 :=
sorry

end range_of_m_l234_234600


namespace min_value_of_x_l234_234067

theorem min_value_of_x (x : ℝ) (h : 2 * (x + 1) ≥ x + 1) : x ≥ -1 := sorry

end min_value_of_x_l234_234067


namespace largest_multiple_of_7_less_than_neg50_l234_234284

theorem largest_multiple_of_7_less_than_neg50 : ∃ x, (∃ k : ℤ, x = 7 * k) ∧ x < -50 ∧ ∀ y, (∃ m : ℤ, y = 7 * m) → y < -50 → y ≤ x :=
sorry

end largest_multiple_of_7_less_than_neg50_l234_234284


namespace participants_begin_competition_l234_234403

theorem participants_begin_competition (x : ℝ) 
  (h1 : 0.4 * x * (1 / 4) = 16) : 
  x = 160 := 
by
  sorry

end participants_begin_competition_l234_234403


namespace f_even_l234_234177

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def f (x : ℝ) : ℝ := x^2 + 1

theorem f_even (a : ℝ) (h1 : is_even f) (h2 : ∀ x, -1 ≤ x ∧ x ≤ a) : f a = 2 :=
  sorry

end f_even_l234_234177


namespace simplify_and_rationalize_l234_234377

theorem simplify_and_rationalize :
  let x := (Real.sqrt 5 / Real.sqrt 7) * (Real.sqrt 9 / Real.sqrt 11) * (Real.sqrt 13 / Real.sqrt 17)
  x = 3 * Real.sqrt 84885 / 1309 := sorry

end simplify_and_rationalize_l234_234377


namespace transformed_quadratic_roots_l234_234215

-- Definitions of the conditions
def quadratic_roots (a b : ℝ) : Prop :=
  ∀ x : ℝ, a * x^2 + b * x + 3 = 0 → (x = -2) ∨ (x = 3)

-- Statement of the theorem
theorem transformed_quadratic_roots (a b : ℝ) :
  quadratic_roots a b →
  ∀ x : ℝ, a * (x + 2)^2 + b * (x + 2) + 3 = 0 → (x = -4) ∨ (x = 1) :=
sorry

end transformed_quadratic_roots_l234_234215


namespace ernie_can_make_circles_l234_234358

-- Make a statement of the problem in Lean 4
theorem ernie_can_make_circles (total_boxes : ℕ) (ali_boxes_per_circle : ℕ) (ernie_boxes_per_circle : ℕ) (ali_circles : ℕ) 
  (h1 : total_boxes = 80) (h2 : ali_boxes_per_circle = 8) (h3 : ernie_boxes_per_circle = 10) (h4 : ali_circles = 5) :
  (total_boxes - ali_boxes_per_circle * ali_circles) / ernie_boxes_per_circle = 4 := 
by 
  -- Proof of the theorem
  sorry

end ernie_can_make_circles_l234_234358


namespace female_officers_on_police_force_l234_234742

theorem female_officers_on_police_force
  (percent_on_duty : ℝ)
  (total_on_duty : ℕ)
  (half_female_on_duty : ℕ)
  (h1 : percent_on_duty = 0.16)
  (h2 : total_on_duty = 160)
  (h3 : half_female_on_duty = total_on_duty / 2)
  (h4 : half_female_on_duty = 80)
  :
  ∃ (total_female_officers : ℕ), total_female_officers = 500 :=
by
  sorry

end female_officers_on_police_force_l234_234742


namespace mildred_initial_oranges_l234_234461

theorem mildred_initial_oranges (final_oranges : ℕ) (added_oranges : ℕ) 
  (final_oranges_eq : final_oranges = 79) (added_oranges_eq : added_oranges = 2) : 
  final_oranges - added_oranges = 77 :=
by
  -- proof steps would go here
  sorry

end mildred_initial_oranges_l234_234461


namespace find_a1_in_arithmetic_sequence_l234_234575

noncomputable def arithmetic_sequence_sum (a₁ d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem find_a1_in_arithmetic_sequence :
  ∀ (a₁ d : ℤ), d = -2 →
  (arithmetic_sequence_sum a₁ d 11 = arithmetic_sequence_sum a₁ d 10) →
  a₁ = 20 :=
by
  intro a₁ d hd hs
  sorry

end find_a1_in_arithmetic_sequence_l234_234575


namespace correct_factorization_l234_234641

theorem correct_factorization {x y : ℝ} :
  (2 * x ^ 2 - 8 * y ^ 2 = 2 * (x + 2 * y) * (x - 2 * y)) ∧
  ¬(x ^ 2 + 3 * x * y + 9 * y ^ 2 = (x + 3 * y) ^ 2)
    ∧ ¬(2 * x ^ 2 - 4 * x * y + 9 * y ^ 2 = (2 * x - 3 * y) ^ 2)
    ∧ ¬(x * (x - y) + y * (y - x) = (x - y) * (x + y)) := 
by sorry

end correct_factorization_l234_234641


namespace julia_average_speed_l234_234315

-- Define the conditions as constants
def total_distance : ℝ := 28
def total_time : ℝ := 4

-- Define the theorem stating Julia's average speed
theorem julia_average_speed : total_distance / total_time = 7 := by
  sorry

end julia_average_speed_l234_234315


namespace sara_change_l234_234292

def cost_of_first_book : ℝ := 5.5
def cost_of_second_book : ℝ := 6.5
def amount_given : ℝ := 20.0
def total_cost : ℝ := cost_of_first_book + cost_of_second_book
def change : ℝ := amount_given - total_cost

theorem sara_change : change = 8 :=
by
  have total_cost_correct : total_cost = 12.0 := by sorry
  have change_correct : change = amount_given - total_cost := by sorry
  show change = 8
  sorry

end sara_change_l234_234292


namespace find_unknown_number_l234_234442

-- Defining the conditions of the problem
def equation (x : ℝ) : Prop := (45 + x / 89) * 89 = 4028

-- Stating the theorem to be proved
theorem find_unknown_number : equation 23 :=
by
  -- Placeholder for the proof
  sorry

end find_unknown_number_l234_234442


namespace playground_area_l234_234620

noncomputable def calculate_area (w s : ℝ) : ℝ := s * s

theorem playground_area (w s : ℝ) (h1 : s = 3 * w + 10) (h2 : 4 * s = 480) : calculate_area w s = 14400 := by
  sorry

end playground_area_l234_234620


namespace domain_of_func_1_domain_of_func_2_domain_of_func_3_domain_of_func_4_l234_234892
-- Import the necessary library.

-- Define the domains for the given functions.
def domain_func_1 (x : ℝ) : Prop := true

def domain_func_2 (x : ℝ) : Prop := 1 ≤ x ∧ x ≤ 2

def domain_func_3 (x : ℝ) : Prop := x ≥ -3 ∧ x ≠ 1

def domain_func_4 (x : ℝ) : Prop := 2 ≤ x ∧ x ≤ 5 ∧ x ≠ 3

-- Prove the domains of each function.
theorem domain_of_func_1 : ∀ x : ℝ, domain_func_1 x :=
by sorry

theorem domain_of_func_2 : ∀ x : ℝ, domain_func_2 x ↔ (1 ≤ x ∧ x ≤ 2) :=
by sorry

theorem domain_of_func_3 : ∀ x : ℝ, domain_func_3 x ↔ (x ≥ -3 ∧ x ≠ 1) :=
by sorry

theorem domain_of_func_4 : ∀ x : ℝ, domain_func_4 x ↔ (2 ≤ x ∧ x ≤ 5 ∧ x ≠ 3) :=
by sorry

end domain_of_func_1_domain_of_func_2_domain_of_func_3_domain_of_func_4_l234_234892


namespace find_number_of_girls_l234_234474

variable (G : ℕ)

-- Given conditions
def avg_weight_girls (total_weight_girls : ℕ) : Prop := total_weight_girls = 45 * G
def avg_weight_boys (total_weight_boys : ℕ) : Prop := total_weight_boys = 275
def avg_weight_students (total_weight_students : ℕ) : Prop := total_weight_students = 500

-- Proposition to prove
theorem find_number_of_girls 
  (total_weight_girls : ℕ) 
  (total_weight_boys : ℕ) 
  (total_weight_students : ℕ) 
  (h1 : avg_weight_girls G total_weight_girls)
  (h2 : avg_weight_boys total_weight_boys)
  (h3 : avg_weight_students total_weight_students) : 
  G = 5 :=
by sorry

end find_number_of_girls_l234_234474


namespace peter_remaining_walk_time_l234_234752

-- Define the parameters and conditions
def total_distance : ℝ := 2.5
def time_per_mile : ℝ := 20
def distance_walked : ℝ := 1

-- Define the remaining distance
def remaining_distance : ℝ := total_distance - distance_walked

-- Define the remaining time Peter needs to walk
def remaining_time_to_walk (d : ℝ) (t : ℝ) : ℝ := d * t

-- State the problem we want to prove
theorem peter_remaining_walk_time :
  remaining_time_to_walk remaining_distance time_per_mile = 30 :=
by
  -- Placeholder for the proof
  sorry

end peter_remaining_walk_time_l234_234752


namespace length_LL1_l234_234997

theorem length_LL1 (XZ : ℝ) (XY : ℝ) (YZ : ℝ) (X1Y : ℝ) (X1Z : ℝ) (LM : ℝ) (LN : ℝ) (MN : ℝ) (L1N : ℝ) (LL1 : ℝ) : 
  XZ = 13 → XY = 5 → 
  YZ = Real.sqrt (XZ^2 - XY^2) → 
  X1Y = 60 / 17 → 
  X1Z = 84 / 17 → 
  LM = X1Z → LN = X1Y → 
  MN = Real.sqrt (LM^2 - LN^2) → 
  (∀ k, L1N = 5 * k ∧ (7 * k + 5 * k) = MN → LL1 = 5 * k) →
  LL1 = 20 / 17 :=
by sorry

end length_LL1_l234_234997


namespace mike_total_cans_l234_234134

theorem mike_total_cans (monday_cans : ℕ) (tuesday_cans : ℕ) (total_cans : ℕ) : 
  monday_cans = 71 ∧ tuesday_cans = 27 ∧ total_cans = monday_cans + tuesday_cans → total_cans = 98 :=
by
  sorry

end mike_total_cans_l234_234134


namespace homework_total_time_l234_234478

theorem homework_total_time :
  ∀ (j g p : ℕ),
  j = 18 →
  g = j - 6 →
  p = 2 * g - 4 →
  j + g + p = 50 :=
by
  intros j g p h1 h2 h3
  sorry

end homework_total_time_l234_234478


namespace calculate_AE_l234_234003

variable {k : ℝ} (A B C D E : Type*)

namespace Geometry

def shared_angle (A B C : Type*) : Prop := sorry -- assumes triangles share angle A

def prop_constant_proportion (AB AC AD AE : ℝ) (k : ℝ) : Prop :=
  AB * AC = k * AD * AE

theorem calculate_AE
  (A B C D E : Type*) 
  (AB AC AD AE : ℝ)
  (h_shared : shared_angle A B C)
  (h_AB : AB = 5)
  (h_AC : AC = 7)
  (h_AD : AD = 2)
  (h_proportion : prop_constant_proportion AB AC AD AE k)
  (h_k : k = 1) :
  AE = 17.5 := 
sorry

end Geometry

end calculate_AE_l234_234003


namespace compare_expressions_l234_234542

-- Define the theorem statement
theorem compare_expressions (x : ℝ) : (x - 2) * (x + 3) > x^2 + x - 7 := by
  sorry -- The proof is omitted.

end compare_expressions_l234_234542


namespace minimum_value_of_expression_l234_234884

theorem minimum_value_of_expression (x y z : ℝ) (h : 2 * x - 3 * y + z = 3) :
  ∃ (x y z : ℝ), (x^2 + (y - 1)^2 + z^2) = 18 / 7 ∧ y = -2 / 7 :=
sorry

end minimum_value_of_expression_l234_234884


namespace winner_last_year_ounces_l234_234437

/-- Definition of the problem conditions -/
def ouncesPerHamburger : ℕ := 4
def hamburgersTonyaAte : ℕ := 22

/-- Theorem stating the desired result -/
theorem winner_last_year_ounces :
  hamburgersTonyaAte * ouncesPerHamburger = 88 :=
by
  sorry

end winner_last_year_ounces_l234_234437


namespace solve_inequality_l234_234498

theorem solve_inequality (x : ℝ) : 2 * x^2 - x - 1 > 0 ↔ x < -1/2 ∨ x > 1 :=
by
  sorry

end solve_inequality_l234_234498


namespace net_displacement_east_of_A_total_fuel_consumed_l234_234949

def distances : List Int := [22, -3, 4, -2, -8, -17, -2, 12, 7, -5]
def fuel_consumption_per_km : ℝ := 0.07

theorem net_displacement_east_of_A :
  List.sum distances = 8 := by
  sorry

theorem total_fuel_consumed :
  List.sum (distances.map Int.natAbs) * fuel_consumption_per_km = 5.74 := by
  sorry

end net_displacement_east_of_A_total_fuel_consumed_l234_234949


namespace find_number_l234_234584

-- Define the number x and state the condition 55 + x = 88
def x := 33

-- State the theorem to be proven: if 55 + x = 88, then x = 33
theorem find_number (h : 55 + x = 88) : x = 33 :=
by
  sorry

end find_number_l234_234584


namespace intersection_A_B_l234_234698

def set_A : Set ℝ := {x | x > 0}
def set_B : Set ℝ := {x | x < 4}

theorem intersection_A_B :
  set_A ∩ set_B = {x | 0 < x ∧ x < 4} := sorry

end intersection_A_B_l234_234698


namespace evaluate_256_pow_5_div_8_l234_234666

theorem evaluate_256_pow_5_div_8 (h : 256 = 2^8) : 256^(5/8) = 32 :=
by
  sorry

end evaluate_256_pow_5_div_8_l234_234666


namespace value_of_expression_l234_234272

theorem value_of_expression (x y : ℝ) (h1 : x = 12) (h2 : y = 18) : 3 * (x - y) * (x + y) = -540 :=
by
  rw [h1, h2]
  sorry

end value_of_expression_l234_234272


namespace min_bench_sections_l234_234107

theorem min_bench_sections (N : ℕ) :
  ∀ x y : ℕ, (x = y) → (x = 8 * N) → (y = 12 * N) → (24 * N) % 20 = 0 → N = 5 :=
by
  intros
  sorry

end min_bench_sections_l234_234107


namespace perimeter_of_ABFCDE_l234_234255

-- Define the problem parameters
def square_perimeter : ℤ := 60
def side_length (p : ℤ) : ℤ := p / 4
def equilateral_triangle_side (l : ℤ) : ℤ := l
def new_shape_sides : ℕ := 6
def new_perimeter (s : ℤ) : ℤ := new_shape_sides * s

-- Define the theorem to be proved
theorem perimeter_of_ABFCDE (p : ℤ) (s : ℕ) (len : ℤ) : len = side_length p → len = equilateral_triangle_side len →
  new_perimeter len = 90 :=
by
  intros h1 h2
  sorry

end perimeter_of_ABFCDE_l234_234255


namespace coordinates_of_B_l234_234464

-- Define the point A
structure Point :=
  (x : ℝ)
  (y : ℝ)

def A : Point := { x := 2, y := 1 }

-- Define the rotation transformation for pi/2 clockwise
def rotate_clockwise_90 (p : Point) : Point :=
  { x := p.y, y := -p.x }

-- Define the point B after rotation
def B := rotate_clockwise_90 A

-- The theorem stating the coordinates of point B (the correct answer)
theorem coordinates_of_B : B = { x := 1, y := -2 } :=
  sorry

end coordinates_of_B_l234_234464


namespace measure_of_C_and_max_perimeter_l234_234311

noncomputable def triangle_C_and_perimeter (a b c A B C : ℝ) (hABC : (2 * a + b) * Real.sin A + (2 * b + a) * Real.sin B = 2 * c * Real.sin C) (hc : c = Real.sqrt 3) : Prop :=
  (C = 2 * Real.pi / 3) ∧ (2 * Real.sin A + 2 * Real.sin B + c ≤ 2 + Real.sqrt 3)

-- Now the Lean theorem statement
theorem measure_of_C_and_max_perimeter (a b c A B C : ℝ) (hABC : (2 * a + b) * Real.sin A + (2 * b + a) * Real.sin B = 2 * c * Real.sin C) (hc : c = Real.sqrt 3) :
  triangle_C_and_perimeter a b c A B C hABC hc :=
by 
  sorry

end measure_of_C_and_max_perimeter_l234_234311


namespace greatest_possible_employees_take_subway_l234_234020

variable (P F : ℕ)

def part_time_employees_take_subway : ℕ := P / 3
def full_time_employees_take_subway : ℕ := F / 4

theorem greatest_possible_employees_take_subway 
  (h1 : P + F = 48) : part_time_employees_take_subway P + full_time_employees_take_subway F ≤ 15 := 
sorry

end greatest_possible_employees_take_subway_l234_234020


namespace complete_square_result_l234_234517

theorem complete_square_result (x : ℝ) :
  (x^2 - 4 * x - 3 = 0) → ((x - 2) ^ 2 = 7) :=
by sorry

end complete_square_result_l234_234517


namespace john_additional_tax_l234_234673

-- Define the old and new tax rates
def old_tax (income : ℕ) : ℕ :=
  if income ≤ 500000 then income * 20 / 100
  else if income ≤ 1000000 then 100000 + (income - 500000) * 25 / 100
  else 225000 + (income - 1000000) * 30 / 100

def new_tax (income : ℕ) : ℕ :=
  if income ≤ 500000 then income * 30 / 100
  else if income ≤ 1000000 then 150000 + (income - 500000) * 35 / 100
  else 325000 + (income - 1000000) * 40 / 100

-- Calculate the tax for rental income after deduction
def rental_income_tax (rental_income : ℕ) : ℕ :=
  let taxable_rental_income := rental_income - rental_income * 10 / 100
  taxable_rental_income * 40 / 100

-- Calculate the tax for investment income
def investment_income_tax (investment_income : ℕ) : ℕ :=
  investment_income * 25 / 100

-- Calculate the tax for self-employment income
def self_employment_income_tax (self_employment_income : ℕ) : ℕ :=
  self_employment_income * 15 / 100

-- Define the total additional tax John pays
def additional_tax_paid (old_main_income new_main_income rental_income investment_income self_employment_income : ℕ) : ℕ :=
  let old_tax_main := old_tax old_main_income
  let new_tax_main := new_tax new_main_income
  let rental_tax := rental_income_tax rental_income
  let investment_tax := investment_income_tax investment_income
  let self_employment_tax := self_employment_income_tax self_employment_income
  (new_tax_main - old_tax_main) + rental_tax + investment_tax + self_employment_tax

-- Prove John pays $352,250 more in taxes under the new system
theorem john_additional_tax (main_income_old main_income_new rental_income investment_income self_employment_income : ℕ) :
  main_income_old = 1000000 →
  main_income_new = 1500000 →
  rental_income = 100000 →
  investment_income = 50000 →
  self_employment_income = 25000 →
  additional_tax_paid main_income_old main_income_new rental_income investment_income self_employment_income = 352250 :=
by
  intros h_old h_new h_rental h_invest h_self
  rw [h_old, h_new, h_rental, h_invest, h_self]
  -- calculation steps are omitted
  sorry

end john_additional_tax_l234_234673


namespace arithmetic_sequence_sum_l234_234412

theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h : a 3 + a 4 + a 5 = 12) :
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 28 := 
by
  sorry

end arithmetic_sequence_sum_l234_234412


namespace root_quadratic_eq_k_value_l234_234649

theorem root_quadratic_eq_k_value (k : ℤ) :
  (∃ x : ℤ, x = 5 ∧ 2 * x ^ 2 + 3 * x - k = 0) → k = 65 :=
by
  sorry

end root_quadratic_eq_k_value_l234_234649


namespace triangle_least_perimeter_l234_234501

theorem triangle_least_perimeter (x : ℤ) (h1 : x + 27 > 34) (h2 : 34 + 27 > x) (h3 : x + 34 > 27) : 27 + 34 + x ≥ 69 :=
by
  have h1' : x > 7 := by linarith
  sorry

end triangle_least_perimeter_l234_234501


namespace total_profit_calculation_l234_234023

-- Define the parameters of the problem
def rajan_investment : ℕ := 20000
def rakesh_investment : ℕ := 25000
def mukesh_investment : ℕ := 15000
def rajan_investment_time : ℕ := 12 -- in months
def rakesh_investment_time : ℕ := 4 -- in months
def mukesh_investment_time : ℕ := 8 -- in months
def rajan_final_share : ℕ := 2400

-- Calculation for total profit
def total_profit (rajan_investment rakesh_investment mukesh_investment
                  rajan_investment_time rakesh_investment_time mukesh_investment_time
                  rajan_final_share : ℕ) : ℕ :=
  let rajan_share := rajan_investment * rajan_investment_time
  let rakesh_share := rakesh_investment * rakesh_investment_time
  let mukesh_share := mukesh_investment * mukesh_investment_time
  let total_investment := rajan_share + rakesh_share + mukesh_share
  (rajan_final_share * total_investment) / rajan_share

-- Proof problem statement
theorem total_profit_calculation :
  total_profit rajan_investment rakesh_investment mukesh_investment
               rajan_investment_time rakesh_investment_time mukesh_investment_time
               rajan_final_share = 4600 :=
by sorry

end total_profit_calculation_l234_234023


namespace wind_velocity_l234_234323

def pressure (P A V : ℝ) (k : ℝ) : Prop :=
  P = k * A * V^2

theorem wind_velocity (k : ℝ) (h_initial : pressure 4 4 8 k) (h_final : pressure 64 16 v k) : v = 16 := by
  sorry

end wind_velocity_l234_234323


namespace abs_eq_self_iff_nonneg_l234_234022

variable (a : ℝ)

theorem abs_eq_self_iff_nonneg (h : |a| = a) : a ≥ 0 :=
by
  sorry

end abs_eq_self_iff_nonneg_l234_234022


namespace engineer_thought_of_l234_234958

def isProperDivisor (n k : ℕ) : Prop :=
  k ≠ 1 ∧ k ≠ n ∧ k ∣ n

def transformDivisors (n m : ℕ) : Prop :=
  ∀ k, isProperDivisor n k → isProperDivisor m (k + 1)

theorem engineer_thought_of (n : ℕ) :
  (∀ m : ℕ, n = 2^2 ∨ n = 2^3 → transformDivisors n m → (m % 2 = 1)) :=
by
  sorry

end engineer_thought_of_l234_234958


namespace inequality_positive_real_xyz_l234_234753

theorem inequality_positive_real_xyz (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x * y * z = 1) :
  (x^3 / ((1 + y) * (1 + z)) + y^3 / ((1 + z) * (1 + x)) + z^3 / ((1 + x) * (1 + y))) ≥ (3 / 4) := 
by
  -- Proof is to be constructed here
  sorry

end inequality_positive_real_xyz_l234_234753


namespace mowing_lawn_time_l234_234998

theorem mowing_lawn_time (pay_mow : ℝ) (rate_hour : ℝ) (time_plant : ℝ) (charge_flowers : ℝ) :
  pay_mow = 15 → rate_hour = 20 → time_plant = 2 → charge_flowers = 45 → 
  (charge_flowers + pay_mow) / rate_hour - time_plant = 1 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  -- This is an outline, so the actual proof steps are omitted
  sorry

end mowing_lawn_time_l234_234998


namespace solve_quad_linear_system_l234_234251

theorem solve_quad_linear_system :
  (∃ x y : ℝ, x^2 - 6 * x + 8 = 0 ∧ y + 2 * x = 12 ∧ ((x, y) = (4, 4) ∨ (x, y) = (2, 8))) :=
sorry

end solve_quad_linear_system_l234_234251


namespace fraction_dutch_americans_has_window_l234_234384

variable (P D DA : ℕ)
variable (f_P_d d_P_w : ℚ)
variable (DA_w : ℕ)

-- Total number of people on the bus P 
-- Fraction of people who were Dutch f_P_d
-- Fraction of Dutch Americans who got window seats d_P_w
-- Number of Dutch Americans who sat at windows DA_w
-- Define the assumptions
def total_people_on_bus := P = 90
def fraction_dutch := f_P_d = 3 / 5
def fraction_dutch_americans_window := d_P_w = 1 / 3
def dutch_americans_window := DA_w = 9

-- Prove that fraction of Dutch people who were also American is 1/2
theorem fraction_dutch_americans_has_window (P D DA DA_w : ℕ) (f_P_d d_P_w : ℚ) :
  total_people_on_bus P ∧ fraction_dutch f_P_d ∧
  fraction_dutch_americans_window d_P_w ∧ dutch_americans_window DA_w →
  (DA: ℚ) / D = 1 / 2 :=
by
  sorry

end fraction_dutch_americans_has_window_l234_234384


namespace right_triangle_no_k_values_l234_234690

theorem right_triangle_no_k_values (k : ℕ) (h : k > 0) : 
  ¬ (∃ k, k > 0 ∧ ((17 > k ∧ 17^2 = 13^2 + k^2) ∨ (k > 17 ∧ k < 30 ∧ k^2 = 13^2 + 17^2))) :=
sorry

end right_triangle_no_k_values_l234_234690


namespace john_initial_bench_weight_l234_234962

variable (B : ℕ)

theorem john_initial_bench_weight (B : ℕ) (HNewTotal : 1490 = 490 + B + 600) : B = 400 :=
by
  sorry

end john_initial_bench_weight_l234_234962


namespace log_div_log_inv_of_16_l234_234055

theorem log_div_log_inv_of_16 : (Real.log 16) / (Real.log (1 / 16)) = -1 :=
by
  sorry

end log_div_log_inv_of_16_l234_234055


namespace solve_system_of_equations_l234_234761

theorem solve_system_of_equations : 
  ∃ (x y : ℝ), 
  (x / y + y / x) * (x + y) = 15 ∧ 
  (x^2 / y^2 + y^2 / x^2) * (x^2 + y^2) = 85 ∧
  ((x = 2 ∧ y = 4) ∨ (x = 4 ∧ y = 2)) :=
by
  sorry

end solve_system_of_equations_l234_234761


namespace range_of_a_l234_234035

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, ¬ x^2 + (a - 1) * x + 1 < 0) ↔ (-1 < a ∧ a < 3) :=
by
  sorry

end range_of_a_l234_234035


namespace time_to_drain_tank_due_to_leak_l234_234217

noncomputable def timeToDrain (P L : ℝ) : ℝ := (1 : ℝ) / L

theorem time_to_drain_tank_due_to_leak (P L : ℝ)
  (hP : P = 0.5)
  (hL : P - L = 5/11) :
  timeToDrain P L = 22 :=
by
  -- to state what needs to be proved here
  sorry

end time_to_drain_tank_due_to_leak_l234_234217


namespace merchant_boxes_fulfill_order_l234_234639

theorem merchant_boxes_fulfill_order :
  ∃ (a b c d e : ℕ), 16 * a + 17 * b + 23 * c + 39 * d + 40 * e = 100 := sorry

end merchant_boxes_fulfill_order_l234_234639


namespace highest_score_l234_234354

variable (avg runs_excluding: ℕ)
variable (innings remaining_innings total_runs total_runs_excluding H L: ℕ)

axiom batting_average (h_avg: avg = 60) (h_innings: innings = 46) : total_runs = avg * innings
axiom diff_highest_lowest_score (h_diff: H - L = 190) : true
axiom avg_excluding_high_low (h_avg_excluding: runs_excluding = 58) (h_remaining_innings: remaining_innings = 44) : total_runs_excluding = runs_excluding * remaining_innings
axiom sum_high_low : total_runs - total_runs_excluding = 208

theorem highest_score (h_avg: avg = 60) (h_innings: innings = 46) (h_diff: H - L = 190) (h_avg_excluding: runs_excluding = 58) (h_remaining_innings: remaining_innings = 44)
    (calc_total_runs: total_runs = avg * innings) 
    (calc_total_runs_excluding: total_runs_excluding = runs_excluding * remaining_innings)
    (calc_sum_high_low: total_runs - total_runs_excluding = 208) : H = 199 :=
by
  sorry

end highest_score_l234_234354


namespace sequence_negation_l234_234504

theorem sequence_negation (x : ℕ → ℝ) (x1_pos : x 1 > 0) (x1_neq1 : x 1 ≠ 1)
  (rec_seq : ∀ n : ℕ, x (n + 1) = (x n * (x n ^ 2 + 3)) / (3 * x n ^ 2 + 1)) :
  ∃ n : ℕ, x n ≤ x (n + 1) :=
sorry

end sequence_negation_l234_234504


namespace sum_of_variables_l234_234074

theorem sum_of_variables (x y z : ℝ) (hpos_x : 0 < x) (hpos_y : 0 < y) (hpos_z : 0 < z)
  (hxy : x * y = 30) (hxz : x * z = 60) (hyz : y * z = 90) :
  x + y + z = 11 * Real.sqrt 5 :=
by
  sorry

end sum_of_variables_l234_234074


namespace sum_of_Q_and_R_in_base_8_l234_234729

theorem sum_of_Q_and_R_in_base_8 (P Q R : ℕ) (hp : 1 ≤ P ∧ P < 8) (hq : 1 ≤ Q ∧ Q < 8) (hr : 1 ≤ R ∧ R < 8) 
  (hdistinct : P ≠ Q ∧ Q ≠ R ∧ P ≠ R) (H : 8^2 * P + 8 * Q + R + (8^2 * R + 8 * Q + P) + (8^2 * Q + 8 * P + R) 
  = 8^3 * P + 8^2 * P + 8 * P) : Q + R = 7 := 
sorry

end sum_of_Q_and_R_in_base_8_l234_234729


namespace pears_sold_in_afternoon_l234_234779

theorem pears_sold_in_afternoon (m a total : ℕ) (h1 : a = 2 * m) (h2 : m = 120) (h3 : m + a = total) (h4 : total = 360) :
  a = 240 :=
by
  sorry

end pears_sold_in_afternoon_l234_234779


namespace eclipse_time_coincide_eclipse_start_time_eclipse_end_time_l234_234991

noncomputable def relative_speed_moon_sun := (17/16 : ℝ) - (1/12 : ℝ)
noncomputable def initial_distance := (47/10 : ℝ)
noncomputable def time_coincide := initial_distance / relative_speed_moon_sun + (9 + 13/60 : ℝ)

theorem eclipse_time_coincide : 
  (time_coincide - 12 : ℝ) = (2 + 1/60 : ℝ) :=
sorry

noncomputable def start_distance := (37/10 : ℝ)
noncomputable def time_start := start_distance / relative_speed_moon_sun + (9 + 13/60 : ℝ)

theorem eclipse_start_time : 
  (time_start - 12 : ℝ) = (1 + 59/60 : ℝ) :=
sorry

noncomputable def end_distance := (57/10 : ℝ)
noncomputable def time_end := end_distance / relative_speed_moon_sun + (9 + 13/60 : ℝ)

theorem eclipse_end_time : 
  (time_end - 12 : ℝ) = (3 + 2/60 : ℝ) :=
sorry

end eclipse_time_coincide_eclipse_start_time_eclipse_end_time_l234_234991


namespace wall_ratio_l234_234148

theorem wall_ratio (V : ℝ) (B : ℝ) (H : ℝ) (x : ℝ) (L : ℝ) :
  V = 12.8 →
  B = 0.4 →
  H = 5 * B →
  L = x * H →
  V = B * H * L →
  x = 4 ∧ L / H = 4 :=
by
  intros hV hB hH hL hVL
  sorry

end wall_ratio_l234_234148


namespace find_smaller_number_l234_234277

theorem find_smaller_number (x y : ℤ) (h1 : x + y = 15) (h2 : 3 * x = 5 * y - 11) : x = 8 :=
by
  sorry

end find_smaller_number_l234_234277


namespace mats_length_l234_234124

open Real

theorem mats_length (r : ℝ) (n : ℤ) (w : ℝ) (y : ℝ) (h₁ : r = 6) (h₂ : n = 8) (h₃ : w = 1):
  y = 6 * sqrt (2 - sqrt 2) :=
sorry

end mats_length_l234_234124


namespace maddie_watched_8_episodes_l234_234826

def minutes_per_episode : ℕ := 44
def minutes_monday : ℕ := 138
def minutes_tuesday_wednesday : ℕ := 0
def minutes_thursday : ℕ := 21
def episodes_friday : ℕ := 2
def minutes_per_episode_friday := episodes_friday * minutes_per_episode
def minutes_weekend : ℕ := 105
def total_minutes := minutes_monday + minutes_tuesday_wednesday + minutes_thursday + minutes_per_episode_friday + minutes_weekend
def answer := total_minutes / minutes_per_episode

theorem maddie_watched_8_episodes : answer = 8 := by
  sorry

end maddie_watched_8_episodes_l234_234826


namespace willie_gave_emily_7_stickers_l234_234862

theorem willie_gave_emily_7_stickers (initial_stickers : ℕ) (final_stickers : ℕ) (given_stickers : ℕ) 
  (h1 : initial_stickers = 36) (h2 : final_stickers = 29) (h3 : given_stickers = initial_stickers - final_stickers) : 
  given_stickers = 7 :=
by
  rw [h1, h2] at h3 -- Replace initial_stickers with 36 and final_stickers with 29 in h3
  exact h3  -- given_stickers = 36 - 29 which is equal to 7.


end willie_gave_emily_7_stickers_l234_234862


namespace fraction_to_terminating_decimal_l234_234287

theorem fraction_to_terminating_decimal :
  (47 / (2^3 * 5^4) : ℝ) = 0.0094 := by
  sorry

end fraction_to_terminating_decimal_l234_234287


namespace mean_identity_example_l234_234046

theorem mean_identity_example {x y z : ℝ} 
  (h1 : x + y + z = 30)
  (h2 : x * y * z = 343)
  (h3 : x * y + y * z + z * x = 257.25) :
  x^2 + y^2 + z^2 = 385.5 :=
by
  sorry

end mean_identity_example_l234_234046


namespace arnel_number_of_boxes_l234_234472

def arnel_kept_pencils : ℕ := 10
def number_of_friends : ℕ := 5
def pencils_per_friend : ℕ := 8
def pencils_per_box : ℕ := 5

theorem arnel_number_of_boxes : ∃ (num_boxes : ℕ), 
  (number_of_friends * pencils_per_friend) + arnel_kept_pencils = num_boxes * pencils_per_box ∧ 
  num_boxes = 10 := sorry

end arnel_number_of_boxes_l234_234472


namespace find_custom_operation_value_l234_234702

noncomputable def custom_operation (a b : ℤ) : ℚ := (1 : ℚ)/a + (1 : ℚ)/b

theorem find_custom_operation_value (a b : ℤ) (h1 : a + b = 12) (h2 : a * b = 32) :
  custom_operation a b = 3 / 8 := by
  sorry

end find_custom_operation_value_l234_234702


namespace sqrt_72_eq_6_sqrt_2_l234_234166

theorem sqrt_72_eq_6_sqrt_2 : Real.sqrt 72 = 6 * Real.sqrt 2 := by
  sorry

end sqrt_72_eq_6_sqrt_2_l234_234166


namespace find_x_l234_234164

theorem find_x :
  ∃ x : ℝ, (0 < x) ∧ (⌊x⌋ * x + x^2 = 93) ∧ (x = 7.10) :=
by {
   sorry
}

end find_x_l234_234164


namespace bus_trip_children_difference_l234_234359

theorem bus_trip_children_difference :
  let initial := 41
  let final :=
    initial
    - 12 + 5   -- First bus stop
    - 7 + 10   -- Second bus stop
    - 14 + 3   -- Third bus stop
    - 9 + 6    -- Fourth bus stop
  initial - final = 18 :=
by sorry

end bus_trip_children_difference_l234_234359


namespace price_of_lemonade_l234_234981

def costOfIngredients : ℝ := 20
def numberOfCups : ℕ := 50
def desiredProfit : ℝ := 80

theorem price_of_lemonade (price_per_cup : ℝ) :
  (costOfIngredients + desiredProfit) / numberOfCups = price_per_cup → price_per_cup = 2 :=
by
  sorry

end price_of_lemonade_l234_234981


namespace sleep_hours_l234_234938

-- Define the times Isaac wakes up, goes to sleep, and takes naps
def monday : ℝ := 16 - 9
def tuesday_night : ℝ := 12 - 6.5
def tuesday_nap : ℝ := 1
def wednesday : ℝ := 9.75 - 7.75
def thursday_night : ℝ := 15.5 - 8
def thursday_nap : ℝ := 1.5
def friday : ℝ := 12 - 7.25
def saturday : ℝ := 12.75 - 9
def sunday_night : ℝ := 10.5 - 8.5
def sunday_nap : ℝ := 2

noncomputable def total_sleep : ℝ := 
  monday +
  (tuesday_night + tuesday_nap) +
  wednesday +
  (thursday_night + thursday_nap) +
  friday +
  saturday +
  (sunday_night + sunday_nap)

theorem sleep_hours (total_sleep : ℝ) : total_sleep = 36.75 := 
by
  -- Here, you would provide the steps used to add up the hours, but we will skip with sorry
  sorry

end sleep_hours_l234_234938


namespace train_length_l234_234203

theorem train_length (speed : ℝ) (time : ℝ) (h1 : speed = 36 * 1000 / 3600) (h2 : time = 14.998800095992321) :
  speed * time = 149.99 :=
by {
  sorry
}

end train_length_l234_234203


namespace math_problem_l234_234522

open Classical

theorem math_problem (s x y : ℝ) (h₁ : s > 0) (h₂ : x^2 + y^2 ≠ 0) (h₃ : x * s^2 < y * s^2) :
  ¬(-x^2 < -y^2) ∧ ¬(-x^2 < y^2) ∧ ¬(x^2 < -y^2) ∧ ¬(x^2 > y^2) := by
  sorry

end math_problem_l234_234522


namespace money_left_after_transactions_l234_234233

-- Define the coin values and quantities
def dimes := 50
def quarters := 24
def nickels := 40
def pennies := 75

-- Define the item costs
def candy_bar_cost := 6 * 10 + 4 * 5 + 5
def lollipop_cost := 25 + 2 * 10 + 10 - 5 
def bag_of_chips_cost := 2 * 25 + 3 * 10 + 15
def bottle_of_soda_cost := 25 + 6 * 10 + 5 * 5 + 20 - 5

-- Define the number of items bought
def num_candy_bars := 6
def num_lollipops := 3
def num_bags_of_chips := 4
def num_bottles_of_soda := 2

-- Define the initial total money
def total_money := (dimes * 10) + (quarters * 25) + (nickels * 5) + (pennies)

-- Calculate the total cost of items
def total_cost := num_candy_bars * candy_bar_cost + num_lollipops * lollipop_cost + num_bags_of_chips * bag_of_chips_cost + num_bottles_of_soda * bottle_of_soda_cost

-- Calculate the money left after transactions
def money_left := total_money - total_cost

-- Theorem statement to prove
theorem money_left_after_transactions : money_left = 85 := by
  sorry

end money_left_after_transactions_l234_234233


namespace cargo_to_passenger_ratio_l234_234799

def total_cars : Nat := 71
def passenger_cars : Nat := 44
def engine_and_caboose : Nat := 2
def cargo_cars : Nat := total_cars - passenger_cars - engine_and_caboose

theorem cargo_to_passenger_ratio : cargo_cars = 25 ∧ passenger_cars = 44 →
  cargo_cars.toFloat / passenger_cars.toFloat = 25.0 / 44.0 :=
by
  intros h
  rw [h.1]
  rw [h.2]
  sorry

end cargo_to_passenger_ratio_l234_234799


namespace sum_single_digit_base_eq_21_imp_b_eq_7_l234_234679

theorem sum_single_digit_base_eq_21_imp_b_eq_7 (b : ℕ) (h : (b - 1) * b / 2 = 2 * b + 1) : b = 7 :=
sorry

end sum_single_digit_base_eq_21_imp_b_eq_7_l234_234679


namespace num_valid_N_l234_234506

theorem num_valid_N : 
  ∃ n : ℕ, n = 4 ∧ ∀ (N : ℕ), (N > 0) → (∃ k : ℕ, 60 = (N+3) * k ∧ k % 2 = 0) ↔ (N = 1 ∨ N = 9 ∨ N = 17 ∨ N = 57) :=
sorry

end num_valid_N_l234_234506


namespace richard_older_than_david_l234_234427

variable {R D S : ℕ}

theorem richard_older_than_david (h1 : R > D) (h2 : D = S + 8) (h3 : R + 8 = 2 * (S + 8)) (h4 : D = 14) : R - D = 6 := by
  sorry

end richard_older_than_david_l234_234427


namespace minimum_value_expression_l234_234460

theorem minimum_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  8 * a^3 + 12 * b^3 + 27 * c^3 + (3 / (27 * a * b * c)) ≥ 6 :=
by
  sorry

end minimum_value_expression_l234_234460


namespace exists_sequence_for_k_l234_234250

variable (n : ℕ) (k : ℕ)

noncomputable def exists_sequence (n k : ℕ) : Prop :=
  ∃ (x : ℕ → ℕ), ∀ i : ℕ, i < n → x i < x (i + 1)

theorem exists_sequence_for_k (n k : ℕ) (h1 : 1 ≤ k) (h2 : k ≤ n) :
  exists_sequence n k :=
  sorry

end exists_sequence_for_k_l234_234250


namespace num_terms_arithmetic_seq_l234_234259

theorem num_terms_arithmetic_seq (a d l : ℝ) (n : ℕ)
  (h1 : a = 3.25) 
  (h2 : d = 4)
  (h3 : l = 55.25)
  (h4 : l = a + (↑n - 1) * d) :
  n = 14 :=
by
  sorry

end num_terms_arithmetic_seq_l234_234259


namespace cos_sq_minus_sin_sq_l234_234440

variable (α β : ℝ)

theorem cos_sq_minus_sin_sq (h : Real.cos (α + β) * Real.cos (α - β) = 1 / 3) :
  Real.cos α ^ 2 - Real.sin β ^ 2 = 1 / 3 :=
sorry

end cos_sq_minus_sin_sq_l234_234440


namespace opposite_of_neg_half_is_half_l234_234162

theorem opposite_of_neg_half_is_half : -(-1 / 2) = (1 / 2) :=
by
  sorry

end opposite_of_neg_half_is_half_l234_234162


namespace value_of_y_l234_234941

theorem value_of_y (x y : ℝ) (h1 : x * y = 9) (h2 : x / y = 36) : y = 1 / 2 :=
by
  sorry

end value_of_y_l234_234941


namespace function_property_l234_234225

variable (g : ℝ × ℝ → ℝ)
variable (cond : ∀ x y : ℝ, g (x, y) = - g (y, x))

theorem function_property (x : ℝ) : g (x, x) = 0 :=
by
  sorry

end function_property_l234_234225


namespace log_increasing_on_interval_l234_234770

theorem log_increasing_on_interval :
  ∀ x : ℝ, x < 1 → (0.2 : ℝ)^(x^2 - 3*x + 2) > 1 :=
by
  sorry

end log_increasing_on_interval_l234_234770


namespace rachel_plant_placement_l234_234241

def num_ways_to_place_plants : ℕ :=
  let plants := ["basil", "basil", "aloe", "cactus"]
  let lamps := ["white", "white", "red", "red"]
  -- we need to compute the number of ways to place 4 plants under 4 lamps
  22

theorem rachel_plant_placement :
  num_ways_to_place_plants = 22 :=
by
  -- Proof omitted for brevity
  sorry

end rachel_plant_placement_l234_234241


namespace divisible_by_5886_l234_234794

theorem divisible_by_5886 (r b c : ℕ) (h1 : (523000 + r * 1000 + b * 100 + c * 10) % 89 = 0) (h2 : r * b * c = 180) : 
  (523000 + r * 1000 + b * 100 + c * 10) % 5886 = 0 := 
sorry

end divisible_by_5886_l234_234794


namespace CannotDetermineDraculaStatus_l234_234968

variable (Transylvanian_is_human : Prop)
variable (Dracula_is_alive : Prop)
variable (Statement : Transylvanian_is_human → Dracula_is_alive)

theorem CannotDetermineDraculaStatus : ¬ (∃ (H : Prop), H = Dracula_is_alive) :=
by
  sorry

end CannotDetermineDraculaStatus_l234_234968


namespace bathroom_square_footage_l234_234577

theorem bathroom_square_footage 
  (tiles_width : ℕ) (tiles_length : ℕ) (tile_size_inch : ℕ)
  (inch_to_foot : ℕ) 
  (h_width : tiles_width = 10) 
  (h_length : tiles_length = 20)
  (h_tile_size : tile_size_inch = 6)
  (h_inch_to_foot : inch_to_foot = 12) :
  let tile_size_foot : ℚ := tile_size_inch / inch_to_foot
  let width_foot : ℚ := tiles_width * tile_size_foot
  let length_foot : ℚ := tiles_length * tile_size_foot
  let area : ℚ := width_foot * length_foot
  area = 50 := 
by
  sorry

end bathroom_square_footage_l234_234577


namespace bathroom_length_l234_234843

theorem bathroom_length (A L W : ℝ) (h₁ : A = 8) (h₂ : W = 2) (h₃ : A = L * W) : L = 4 :=
by
  -- Skip the proof with sorry
  sorry

end bathroom_length_l234_234843


namespace dormouse_is_thief_l234_234009

-- Definitions of the suspects
inductive Suspect
| MarchHare
| Hatter
| Dormouse

open Suspect

-- Definitions of the statement conditions
def statement (s : Suspect) : Suspect :=
match s with
| MarchHare => Hatter
| Hatter => sorry -- Sonya and Hatter's testimonies are not recorded
| Dormouse => sorry -- Sonya and Hatter's testimonies are not recorded

-- Condition that only the thief tells the truth
def tells_truth (thief : Suspect) (s : Suspect) : Prop :=
s = thief

-- Conditions of the problem
axiom condition1 : statement MarchHare = Hatter
axiom condition2 : ∃ t, tells_truth t MarchHare ∧ ¬ tells_truth t Hatter ∧ ¬ tells_truth t Dormouse

-- Proposition that Dormouse (Sonya) is the thief
theorem dormouse_is_thief : (∃ t, tells_truth t MarchHare ∧ ¬ tells_truth t Hatter ∧ ¬ tells_truth t Dormouse) → t = Dormouse :=
sorry

end dormouse_is_thief_l234_234009


namespace not_divisible_l234_234380

theorem not_divisible {x y : ℕ} (hx : x > 0) (hy : y > 2) : ¬ (2^y - 1) ∣ (2^x + 1) := sorry

end not_divisible_l234_234380


namespace simplify_to_quadratic_form_l234_234824

noncomputable def simplify_expression (p : ℝ) : ℝ :=
  ((6 * p + 2) - 3 * p * 5) ^ 2 + (5 - 2 / 4) * (8 * p - 12)

theorem simplify_to_quadratic_form (p : ℝ) : simplify_expression p = 81 * p ^ 2 - 50 :=
sorry

end simplify_to_quadratic_form_l234_234824


namespace largest_among_a_b_c_d_l234_234691

noncomputable def a : ℝ := Real.sin (Real.cos (2015 * Real.pi / 180))
noncomputable def b : ℝ := Real.sin (Real.sin (2015 * Real.pi / 180))
noncomputable def c : ℝ := Real.cos (Real.sin (2015 * Real.pi / 180))
noncomputable def d : ℝ := Real.cos (Real.cos (2015 * Real.pi / 180))

theorem largest_among_a_b_c_d : c = max a (max b (max c d)) := by
  sorry

end largest_among_a_b_c_d_l234_234691


namespace divide_inequality_by_negative_l234_234550

theorem divide_inequality_by_negative {x : ℝ} (h : -6 * x > 2) : x < -1 / 3 :=
by sorry

end divide_inequality_by_negative_l234_234550


namespace gym_membership_total_cost_l234_234860

-- Definitions for the conditions stated in the problem
def first_gym_monthly_fee : ℕ := 10
def first_gym_signup_fee : ℕ := 50
def first_gym_discount_rate : ℕ := 10
def first_gym_personal_training_cost : ℕ := 25
def first_gym_sessions_per_year : ℕ := 52

def second_gym_multiplier : ℕ := 3
def second_gym_monthly_fee : ℕ := 3 * first_gym_monthly_fee
def second_gym_signup_fee_multiplier : ℕ := 4
def second_gym_discount_rate : ℕ := 10
def second_gym_personal_training_cost : ℕ := 45
def second_gym_sessions_per_year : ℕ := 52

-- Proof of the total amount John paid in the first year
theorem gym_membership_total_cost:
  let first_gym_annual_cost := (first_gym_monthly_fee * 12) +
                                (first_gym_signup_fee * (100 - first_gym_discount_rate) / 100) +
                                (first_gym_personal_training_cost * first_gym_sessions_per_year)
  let second_gym_annual_cost := (second_gym_monthly_fee * 12) +
                                (second_gym_monthly_fee * second_gym_signup_fee_multiplier * (100 - second_gym_discount_rate) / 100) +
                                (second_gym_personal_training_cost * second_gym_sessions_per_year)
  let total_annual_cost := first_gym_annual_cost + second_gym_annual_cost
  total_annual_cost = 4273 := by
  -- Declaration of the variables used in the problem
  let first_gym_annual_cost := 1465
  let second_gym_annual_cost := 2808
  let total_annual_cost := first_gym_annual_cost + second_gym_annual_cost
  -- Simplify and verify the total cost
  sorry

end gym_membership_total_cost_l234_234860


namespace points_opposite_sides_l234_234650

theorem points_opposite_sides (m : ℝ) : (-2 < m ∧ m < -1) ↔ ((2 - 3 * 1 - m) * (1 - 3 * 1 - m) < 0) := by
  sorry

end points_opposite_sides_l234_234650


namespace find_sum_of_abc_l234_234518

noncomputable def m (a b c : ℕ) : ℝ := a - b * Real.sqrt c

theorem find_sum_of_abc (a b c : ℕ) (ha : ¬ (c % 2 = 0) ∧ ∀ p : ℕ, Prime p → ¬ p * p ∣ c) 
  (hprob : ((30 - m a b c) ^ 2 / 30 ^ 2 = 0.75)) : a + b + c = 48 := 
by
  sorry

end find_sum_of_abc_l234_234518


namespace hcf_of_two_numbers_l234_234912

theorem hcf_of_two_numbers (A B : ℕ) (h1 : Nat.lcm A B = 750) (h2 : A * B = 18750) : Nat.gcd A B = 25 :=
by
  sorry

end hcf_of_two_numbers_l234_234912


namespace right_triangle_area_l234_234454

theorem right_triangle_area (a b : ℝ) 
  (h1 : a = 5) 
  (h2 : b = 12) 
  (right_triangle : ∃ c : ℝ, c^2 = a^2 + b^2) : 
  ∃ A : ℝ, A = 1/2 * a * b ∧ A = 30 := 
by
  sorry

end right_triangle_area_l234_234454


namespace solve_a_b_l234_234402

theorem solve_a_b (a b : ℕ) (h₀ : 2 * a^2 = 3 * b^3) : ∃ k : ℕ, a = 18 * k^3 ∧ b = 6 * k^2 := 
sorry

end solve_a_b_l234_234402


namespace arnold_danny_age_l234_234061

theorem arnold_danny_age (x : ℕ) : (x + 1) * (x + 1) = x * x + 9 → x = 4 :=
by
  intro h
  sorry

end arnold_danny_age_l234_234061


namespace eval_expression_l234_234316

theorem eval_expression : 3 ^ 2 - (4 * 2) = 1 :=
by
  sorry

end eval_expression_l234_234316


namespace electric_blankets_sold_l234_234869

theorem electric_blankets_sold (T H E : ℕ)
  (h1 : 2 * T + 6 * H + 10 * E = 1800)
  (h2 : T = 7 * H)
  (h3 : H = 2 * E) : 
  E = 36 :=
by {
  sorry
}

end electric_blankets_sold_l234_234869


namespace paving_stone_length_l234_234088

theorem paving_stone_length (courtyard_length courtyard_width paving_stone_width : ℝ)
  (num_paving_stones : ℕ)
  (courtyard_dims : courtyard_length = 40 ∧ courtyard_width = 20) 
  (paving_stone_dims : paving_stone_width = 2) 
  (num_stones : num_paving_stones = 100) 
  : (courtyard_length * courtyard_width) / (num_paving_stones * paving_stone_width) = 4 :=
by 
  sorry

end paving_stone_length_l234_234088


namespace find_toonies_l234_234356

-- Define the number of coins and their values
variables (L T : ℕ) -- L represents the number of loonies, T represents the number of toonies

-- Define the conditions
def total_coins := L + T = 10
def total_value := 1 * L + 2 * T = 14

-- Define the theorem to be proven
theorem find_toonies (L T : ℕ) (h1 : total_coins L T) (h2 : total_value L T) : T = 4 :=
by
  sorry

end find_toonies_l234_234356


namespace min_value_abs_function_l234_234557

theorem min_value_abs_function : ∀ x : ℝ, 4 ≤ x ∧ x ≤ 6 → (|x - 4| + |x - 6| = 2) :=
by
  sorry


end min_value_abs_function_l234_234557


namespace solve_fractional_equation_for_c_l234_234899

theorem solve_fractional_equation_for_c :
  (∃ c : ℝ, (c - 37) / 3 = (3 * c + 7) / 8) → c = -317 := by
sorry

end solve_fractional_equation_for_c_l234_234899


namespace initial_eggs_count_l234_234773

theorem initial_eggs_count (harry_adds : ℕ) (total_eggs : ℕ) (initial_eggs : ℕ) :
  harry_adds = 5 → total_eggs = 52 → initial_eggs = total_eggs - harry_adds → initial_eggs = 47 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end initial_eggs_count_l234_234773


namespace girl_scouts_short_amount_l234_234049

-- Definitions based on conditions
def amount_earned : ℝ := 30
def pool_entry_cost_per_person : ℝ := 2.50
def num_people : ℕ := 10
def transportation_fee_per_person : ℝ := 1.25
def snack_cost_per_person : ℝ := 3.00

-- Calculate individual costs
def total_pool_entry_cost : ℝ := pool_entry_cost_per_person * num_people
def total_transportation_fee : ℝ := transportation_fee_per_person * num_people
def total_snack_cost : ℝ := snack_cost_per_person * num_people

-- Calculate total expenses
def total_expenses : ℝ := total_pool_entry_cost + total_transportation_fee + total_snack_cost

-- The amount left after expenses
def amount_left : ℝ := amount_earned - total_expenses

-- Proof problem statement
theorem girl_scouts_short_amount : amount_left = -37.50 := by
  sorry

end girl_scouts_short_amount_l234_234049


namespace cannot_achieve_80_cents_with_six_coins_l234_234762

theorem cannot_achieve_80_cents_with_six_coins:
  ¬ (∃ (p n d : ℕ), p + n + d = 6 ∧ p + 5 * n + 10 * d = 80) :=
by
  sorry

end cannot_achieve_80_cents_with_six_coins_l234_234762


namespace percent_parrots_among_non_pelicans_l234_234583

theorem percent_parrots_among_non_pelicans 
  (parrots_percent pelicans_percent owls_percent sparrows_percent : ℝ) 
  (H1 : parrots_percent = 40) 
  (H2 : pelicans_percent = 20) 
  (H3 : owls_percent = 15) 
  (H4 : sparrows_percent = 100 - parrots_percent - pelicans_percent - owls_percent)
  (H5 : pelicans_percent / 100 < 1) :
  parrots_percent / (100 - pelicans_percent) * 100 = 50 :=
by sorry

end percent_parrots_among_non_pelicans_l234_234583


namespace sum_of_other_endpoint_coordinates_l234_234955

/-- 
  Given that (9, -15) is the midpoint of the segment with one endpoint (7, 4),
  find the sum of the coordinates of the other endpoint.
-/
theorem sum_of_other_endpoint_coordinates : 
  ∃ x y : ℤ, ((7 + x) / 2 = 9 ∧ (4 + y) / 2 = -15) ∧ (x + y = -23) :=
by
  sorry

end sum_of_other_endpoint_coordinates_l234_234955


namespace minimum_groups_needed_l234_234617

theorem minimum_groups_needed :
  ∃ (g : ℕ), g = 5 ∧ ∀ n k : ℕ, n = 30 → k ≤ 7 → n / k = g :=
by
  sorry

end minimum_groups_needed_l234_234617


namespace quadratic_root_property_l234_234184

theorem quadratic_root_property (a x1 x2 : ℝ) 
  (h_eq : ∀ x, a * x^2 - (3 * a + 1) * x + 2 * (a + 1) = 0)
  (h_distinct : x1 ≠ x2)
  (h_relation : x1 - x1 * x2 + x2 = 1 - a) : a = -1 :=
sorry

end quadratic_root_property_l234_234184


namespace digits_sum_is_31_l234_234844

noncomputable def digits_sum_proof (A B C D E F G : ℕ) : Prop :=
  (1000 * A + 100 * B + 10 * C + D + 100 * E + 10 * F + G = 2020) ∧ 
  (A ≠ B) ∧ (A ≠ C) ∧ (A ≠ D) ∧ (A ≠ E) ∧ (A ≠ F) ∧ (A ≠ G) ∧
  (B ≠ C) ∧ (B ≠ D) ∧ (B ≠ E) ∧ (B ≠ F) ∧ (B ≠ G) ∧
  (C ≠ D) ∧ (C ≠ E) ∧ (C ≠ F) ∧ (C ≠ G) ∧
  (D ≠ E) ∧ (D ≠ F) ∧ (D ≠ G) ∧
  (E ≠ F) ∧ (E ≠ G) ∧
  (F ≠ G)

theorem digits_sum_is_31 (A B C D E F G : ℕ) (h : digits_sum_proof A B C D E F G) : 
  A + B + C + D + E + F + G = 31 :=
sorry

end digits_sum_is_31_l234_234844


namespace math_problem_l234_234441

theorem math_problem (a b : ℝ) 
  (h1 : a^2 - 3*a*b + 2*b^2 + a - b = 0)
  (h2 : a^2 - 2*a*b + b^2 - 5*a + 7*b = 0) :
  a*b - 12*a + 15*b = 0 :=
by
  sorry

end math_problem_l234_234441


namespace stream_speed_is_one_l234_234887

noncomputable def speed_of_stream (downstream_speed upstream_speed : ℝ) : ℝ :=
  (downstream_speed - upstream_speed) / 2

theorem stream_speed_is_one : speed_of_stream 10 8 = 1 := by
  sorry

end stream_speed_is_one_l234_234887


namespace complement_U_A_l234_234644

def U : Set ℝ := {x | ∃ y, y = Real.sqrt x}
def A : Set ℝ := {x | 3 ≤ 2 * x - 1 ∧ 2 * x - 1 < 5}

theorem complement_U_A : (U \ A) = {x | (0 ≤ x ∧ x < 2) ∨ (3 ≤ x)} := sorry

end complement_U_A_l234_234644


namespace john_weight_loss_percentage_l234_234951

def john_initial_weight := 220
def john_final_weight_after_gain := 200
def weight_gain := 2

theorem john_weight_loss_percentage : 
  ∃ P : ℝ, (john_initial_weight - (P / 100) * john_initial_weight + weight_gain = john_final_weight_after_gain) ∧ P = 10 :=
sorry

end john_weight_loss_percentage_l234_234951


namespace greatest_power_of_3_l234_234813

theorem greatest_power_of_3 (n : ℕ) : 
  (n = 603) → 
  3^603 ∣ (15^n - 6^n + 3^n) ∧ ¬ (3^(603+1) ∣ (15^n - 6^n + 3^n)) :=
by
  intro hn
  cases hn
  sorry

end greatest_power_of_3_l234_234813


namespace vertex_hyperbola_l234_234006

theorem vertex_hyperbola (a b : ℝ) (h_cond : 8 * a^2 + 4 * a * b = b^3) :
    let xv := -b / (2 * a)
    let yv := (4 * a - b^2) / (4 * a)
    (xv * yv = 1) :=
  by
  sorry

end vertex_hyperbola_l234_234006


namespace max_path_length_correct_l234_234515

noncomputable def maxFlyPathLength : ℝ :=
  2 * Real.sqrt 2 + Real.sqrt 6 + 6

theorem max_path_length_correct :
  ∀ (fly_path_length : ℝ), (fly_path_length = maxFlyPathLength) :=
by
  intro fly_path_length
  sorry

end max_path_length_correct_l234_234515


namespace second_neighbor_brought_less_l234_234330

theorem second_neighbor_brought_less (n1 n2 : ℕ) (htotal : ℕ) (h1 : n1 = 75) (h_total : n1 + n2 = 125) :
  n1 - n2 = 25 :=
by
  sorry

end second_neighbor_brought_less_l234_234330


namespace profit_percentage_of_revenues_l234_234421

theorem profit_percentage_of_revenues (R P : ℝ)
  (H1 : R > 0)
  (H2 : P > 0)
  (H3 : P * 0.98 = R * 0.098) :
  (P / R) * 100 = 10 := by
  sorry

end profit_percentage_of_revenues_l234_234421


namespace leo_weight_proof_l234_234922

def Leo_s_current_weight (L K : ℝ) := 
  L + 10 = 1.5 * K ∧ L + K = 170 → L = 98

theorem leo_weight_proof : ∀ (L K : ℝ), L + 10 = 1.5 * K ∧ L + K = 170 → L = 98 := 
by 
  intros L K h
  sorry

end leo_weight_proof_l234_234922


namespace tan_double_angle_l234_234227

variable {α β : ℝ}

theorem tan_double_angle (h1 : Real.tan (α + β) = 3) (h2 : Real.tan (α - β) = 2) : Real.tan (2 * α) = -1 := by
  sorry

end tan_double_angle_l234_234227


namespace hexagon_perimeter_l234_234987

theorem hexagon_perimeter (side_length : ℕ) (num_sides : ℕ) (perimeter : ℕ) 
  (h1 : num_sides = 6)
  (h2 : side_length = 7)
  (h3 : perimeter = side_length * num_sides) : perimeter = 42 := by
  sorry

end hexagon_perimeter_l234_234987


namespace heat_more_games_than_bulls_l234_234652

theorem heat_more_games_than_bulls (H : ℕ) 
(h1 : 70 + H = 145) :
H - 70 = 5 :=
sorry

end heat_more_games_than_bulls_l234_234652


namespace right_triangles_sides_l234_234275

theorem right_triangles_sides (a b c p S r DH FC FH: ℝ)
  (h₁ : a = 10)
  (h₂ : b = 10)
  (h₃ : c = 12)
  (h₄ : p = (a + b + c) / 2)
  (h₅ : S = Real.sqrt (p * (p - a) * (p - b) * (p - c)))
  (h₆ : r = S / p)
  (h₇ : DH = (c / 2) - r)
  (h₈ : FC = (a * r) / DH)
  (h₉ : FH = Real.sqrt (FC^2 - DH^2))
: FC = 3 ∧ DH = 4 ∧ FH = 5 := by
  sorry

end right_triangles_sides_l234_234275


namespace percentage_of_total_population_absent_l234_234591

def total_students : ℕ := 120
def boys : ℕ := 72
def girls : ℕ := 48
def boys_absent_fraction : ℚ := 1/8
def girls_absent_fraction : ℚ := 1/4

theorem percentage_of_total_population_absent : 
  (boys_absent_fraction * boys + girls_absent_fraction * girls) / total_students * 100 = 17.5 :=
by
  sorry

end percentage_of_total_population_absent_l234_234591


namespace find_a_l234_234797

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a * x^2 + x
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := 3 * x^2 + 2 * a * x + 1

theorem find_a (a : ℝ) : f' a 1 = 6 → a = 1 :=
by
  intro h
  have h_f_prime : 3 * (1 : ℝ) ^ 2 + 2 * a * (1 : ℝ) + 1 = 6 := h
  sorry

end find_a_l234_234797


namespace most_stable_athlete_l234_234563

theorem most_stable_athlete (s2_A s2_B s2_C s2_D : ℝ) 
  (hA : s2_A = 0.5) 
  (hB : s2_B = 0.5) 
  (hC : s2_C = 0.6) 
  (hD : s2_D = 0.4) :
  s2_D < s2_A ∧ s2_D < s2_B ∧ s2_D < s2_C :=
by
  sorry

end most_stable_athlete_l234_234563


namespace value_of_6_inch_cube_is_1688_l234_234443

noncomputable def cube_value (side_length : ℝ) : ℝ :=
  let volume := side_length ^ 3
  (volume / 64) * 500

-- Main statement
theorem value_of_6_inch_cube_is_1688 :
  cube_value 6 = 1688 := by
  sorry

end value_of_6_inch_cube_is_1688_l234_234443


namespace consecutive_cubes_perfect_square_l234_234317

theorem consecutive_cubes_perfect_square :
  ∃ n k : ℕ, (n + 1)^3 - n^3 = k^2 ∧ 
             (∀ m l : ℕ, (m + 1)^3 - m^3 = l^2 → n ≤ m) :=
sorry

end consecutive_cubes_perfect_square_l234_234317


namespace Avianna_red_candles_l234_234024

theorem Avianna_red_candles (R : ℕ) : 
  (R / 27 = 5 / 3) → R = 45 := 
by
  sorry

end Avianna_red_candles_l234_234024


namespace robot_trajectory_no_intersection_l234_234140

noncomputable def parabola_equation (x y : ℝ) : Prop := y^2 = 4 * x
noncomputable def line_equation (x y k : ℝ) : Prop := y = k * (x + 1)

theorem robot_trajectory_no_intersection (k : ℝ) :
  (∀ x y : ℝ, parabola_equation x y → ¬ line_equation x y k) →
  (k > 1 ∨ k < -1) :=
by
  sorry

end robot_trajectory_no_intersection_l234_234140


namespace train_length_correct_l234_234671

noncomputable def train_length (speed_kmh : ℝ) (time_s : ℝ) : ℝ :=
  let speed_ms := (speed_kmh * 1000) / 3600
  speed_ms * time_s

theorem train_length_correct 
  (speed_kmh : ℝ := 60) 
  (time_s : ℝ := 9) :
  train_length speed_kmh time_s = 150.03 := by 
  sorry

end train_length_correct_l234_234671


namespace evaluate_f_i_l234_234326

noncomputable def f (x : ℂ) : ℂ :=
  (x^5 + 2 * x^3 + x) / (x + 1)

theorem evaluate_f_i : f (Complex.I) = 0 := 
  sorry

end evaluate_f_i_l234_234326


namespace complement_of_intersection_l234_234706

theorem complement_of_intersection (U M N : Set ℤ) (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2, 4}) (hN : N = {3, 4, 5}) :
   U \ (M ∩ N) = {1, 2, 3, 5} := by
   sorry

end complement_of_intersection_l234_234706


namespace cubic_identity_l234_234574

theorem cubic_identity (x y z : ℝ) 
  (h1 : x + y + z = 12) 
  (h2 : xy + xz + yz = 30) : 
  x^3 + y^3 + z^3 - 3 * x * y * z = 648 :=
sorry

end cubic_identity_l234_234574


namespace smallest_prime_that_is_6_more_than_perfect_square_and_9_less_than_next_perfect_square_l234_234289

theorem smallest_prime_that_is_6_more_than_perfect_square_and_9_less_than_next_perfect_square :
  ∃ p : ℕ, Prime p ∧ (∃ k m : ℤ, k^2 = p - 6 ∧ m^2 = p + 9 ∧ m^2 - k^2 = 15) ∧ p = 127 :=
sorry

end smallest_prime_that_is_6_more_than_perfect_square_and_9_less_than_next_perfect_square_l234_234289


namespace benny_start_cards_l234_234531

--- Benny bought 4 new cards before the dog ate half of his collection.
def new_cards : Int := 4

--- The remaining cards after the dog ate half of the collection is 34.
def remaining_cards : Int := 34

--- The total number of cards Benny had before adding the new cards and the dog ate half.
def total_before_eating := remaining_cards * 2

theorem benny_start_cards : total_before_eating - new_cards = 64 :=
sorry

end benny_start_cards_l234_234531


namespace ancient_chinese_poem_l234_234470

theorem ancient_chinese_poem (x : ℕ) :
  (7 * x + 7 = 9 * (x - 1)) :=
sorry

end ancient_chinese_poem_l234_234470


namespace rational_square_plus_one_positive_l234_234502

theorem rational_square_plus_one_positive (x : ℚ) : x^2 + 1 > 0 :=
sorry

end rational_square_plus_one_positive_l234_234502


namespace stools_count_l234_234268

theorem stools_count : ∃ x y : ℕ, 3 * x + 4 * y = 39 ∧ x = 3 := 
by
  sorry

end stools_count_l234_234268


namespace circle_equation_line_equation_l234_234171

theorem circle_equation (a b r x y : ℝ) (h1 : a + b = 2 * x + y)
  (h2 : (a, 2*a - 2) = ((1, 2) : ℝ × ℝ))
  (h3 : (a, 2*a - 2) = ((2, 1) : ℝ × ℝ)) :
  (x - 2) ^ 2 + (y - 2) ^ 2 = 1 := sorry

theorem line_equation (x y m : ℝ) (h1 : y + 3 = (x - (-3)) * ((-3) - 0) / (m - (-3)))
  (h2 : (x, y) = (m, 0) ∨ (x, y) = (m, 0))
  (h3 : (m = 1 ∨ m = - 3 / 4)) :
  (3 * x + 4 * y - 3 = 0) ∨ (4 * x + 3 * y + 3 = 0) := sorry

end circle_equation_line_equation_l234_234171


namespace find_CD_l234_234754

noncomputable def C : ℝ := 32 / 9
noncomputable def D : ℝ := 4 / 9

theorem find_CD :
  (∀ x, x ≠ 6 ∧ x ≠ -3 → (4 * x + 8) / (x^2 - 3 * x - 18) = 
       C / (x - 6) + D / (x + 3)) →
  C = 32 / 9 ∧ D = 4 / 9 :=
by sorry

end find_CD_l234_234754


namespace tangent_value_range_l234_234686

theorem tangent_value_range : 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ (π / 4) → 0 ≤ (Real.tan x) ∧ (Real.tan x) ≤ 1) :=
by
  sorry

end tangent_value_range_l234_234686


namespace tangent_line_eq_l234_234414

noncomputable def curve (x : ℝ) : ℝ := x^2 + 3 * x + 1

def point : ℝ × ℝ := (2, 5)

theorem tangent_line_eq : ∀ (x y : ℝ), 
  (y = x^2 + 3 * x + 1) ∧ (x = 2 ∧ y = 5) →
  7 * x - y = 9 :=
by
  intros x y h
  sorry

end tangent_line_eq_l234_234414


namespace age_difference_l234_234018

theorem age_difference (A B n : ℕ) (h1 : A = B + n) (h2 : A - 1 = 3 * (B - 1)) (h3 : A = B^2) : n = 2 :=
by
  sorry

end age_difference_l234_234018


namespace joan_dozen_of_eggs_l234_234624

def number_of_eggs : ℕ := 72
def dozen : ℕ := 12

theorem joan_dozen_of_eggs : (number_of_eggs / dozen) = 6 := by
  sorry

end joan_dozen_of_eggs_l234_234624


namespace butterfat_mixture_l234_234983

theorem butterfat_mixture (x : ℝ) :
  (0.10 * x + 0.30 * 8 = 0.20 * (x + 8)) → x = 8 :=
by
  intro h
  sorry

end butterfat_mixture_l234_234983


namespace sam_drove_distance_l234_234264

theorem sam_drove_distance (marguerite_distance : ℝ) (marguerite_time : ℝ) (sam_time : ℝ) :
  marguerite_distance = 150 ∧ marguerite_time = 3 ∧ sam_time = 4 →
  (sam_time * (marguerite_distance / marguerite_time) = 200) :=
by
  sorry

end sam_drove_distance_l234_234264


namespace at_most_one_true_l234_234282

theorem at_most_one_true (p q : Prop) (h : ¬(p ∧ q)) : ¬(p ∧ q ∧ ¬(¬p ∧ ¬q)) :=
by
  sorry

end at_most_one_true_l234_234282


namespace arithmetic_seq_question_l234_234510

theorem arithmetic_seq_question (a : ℕ → ℤ) (d : ℤ) (h_arith : ∀ n, a n = a 1 + (n - 1) * d)
  (h_sum : a 4 + a 6 + a 8 + a 10 + a 12 = 120) :
  2 * a 10 - a 12 = 24 := 
sorry

end arithmetic_seq_question_l234_234510


namespace solve_fraction_equation_l234_234007

theorem solve_fraction_equation (x : ℚ) (h : (x + 7) / (x - 4) = (x - 5) / (x + 3)) : x = -1 / 19 := 
sorry

end solve_fraction_equation_l234_234007


namespace intersection_P_Q_l234_234521

def P (y : ℝ) : Prop :=
  ∃ (x : ℝ), y = -x^2 + 2

def Q (y : ℝ) : Prop :=
  ∃ (x : ℝ), y = x

theorem intersection_P_Q :
  { y : ℝ | P y } ∩ { y : ℝ | Q y } = { y : ℝ | y ≤ 2 } :=
by
  sorry

end intersection_P_Q_l234_234521


namespace distance_center_to_line_circle_l234_234174

noncomputable def circle_center : ℝ × ℝ := (2, Real.pi / 2)

noncomputable def distance_from_center_to_line (radius : ℝ) (center : ℝ × ℝ) : ℝ :=
  radius * Real.sin (center.snd - Real.pi / 3)

theorem distance_center_to_line_circle : distance_from_center_to_line 2 circle_center = 1 := by
  sorry

end distance_center_to_line_circle_l234_234174


namespace alpha_beta_value_l234_234204

variable (α β : ℝ)

def quadratic (x : ℝ) := x^2 + 2 * x - 2005

axiom roots_quadratic_eq : quadratic α = 0 ∧ quadratic β = 0

theorem alpha_beta_value :
  α^2 + 3 * α + β = 2003 :=
by sorry

end alpha_beta_value_l234_234204


namespace find_star_l234_234928

theorem find_star :
  ∃ (star : ℤ), 45 - ( 28 - ( 37 - ( 15 - star ) ) ) = 56 ∧ star = 17 :=
by
  sorry

end find_star_l234_234928


namespace zero_function_l234_234394

noncomputable def f : ℝ → ℝ := sorry -- Let it be a placeholder for now.

theorem zero_function (a b : ℝ) (h_cont : ContinuousOn f (Set.Icc a b))
  (h_int : ∀ n : ℕ, ∫ x in a..b, (x : ℝ)^n * f x = 0) :
  ∀ x ∈ Set.Icc a b, f x = 0 :=
by
  sorry -- placeholder for the proof

end zero_function_l234_234394


namespace tangent_parallel_l234_234075

noncomputable def f (x: ℝ) : ℝ := x^4 - x
noncomputable def f' (x: ℝ) : ℝ := 4 * x^3 - 1

theorem tangent_parallel
  (P : ℝ × ℝ)
  (hp : P = (1, 0))
  (tangent_parallel : ∀ x, f' x = 3 ↔ x = 1)
  : P = (1, 0) := 
by 
  sorry

end tangent_parallel_l234_234075


namespace geometric_sequence_a2_value_l234_234062

theorem geometric_sequence_a2_value
    (a : ℕ → ℝ)
    (h1 : a 1 = 1/5)
    (h3 : a 3 = 5)
    (geometric : ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r) :
    a 2 = 1 ∨ a 2 = -1 := by
  sorry

end geometric_sequence_a2_value_l234_234062


namespace C_and_D_complete_work_together_in_2_86_days_l234_234530

def work_rate (days : ℕ) : ℚ := 1 / days

def A_rate := work_rate 4
def B_rate := work_rate 10
def D_rate := work_rate 5

noncomputable def C_rate : ℚ :=
  let combined_A_B_C_rate := A_rate + B_rate + (1 / (2 : ℚ))
  let C_rate := 1 / (20 / 3 : ℚ)  -- Solved from the equations provided in the solution
  C_rate

noncomputable def combined_C_D_rate := C_rate + D_rate

noncomputable def days_for_C_and_D_to_complete_work : ℚ :=
  1 / combined_C_D_rate

theorem C_and_D_complete_work_together_in_2_86_days :
  abs (days_for_C_and_D_to_complete_work - 2.86) < 0.01 := sorry

end C_and_D_complete_work_together_in_2_86_days_l234_234530


namespace simplify_expression_l234_234428

variable (b : ℝ) (hb : 0 < b)

theorem simplify_expression : 
  ( ( b ^ (16 / 8) ^ (1 / 4) ) ^ 3 * ( b ^ (16 / 4) ^ (1 / 8) ) ^ 3 ) = b ^ 3 := by
  sorry

end simplify_expression_l234_234428


namespace geom_seq_frac_l234_234589

noncomputable def geom_seq_sum (a1 : ℕ) (q : ℕ) (n : ℕ) : ℕ :=
  a1 * (1 - q ^ n) / (1 - q)

theorem geom_seq_frac (a1 q : ℕ) (hq : q > 1) (h_sum : a1 * (q ^ 3 + q ^ 6 + 1 + q + q ^ 2 + q ^ 5) = 20)
  (h_prod : a1 ^ 7 * q ^ (3 + 6) = 64) :
  geom_seq_sum a1 q 6 / geom_seq_sum a1 q 9 = 5 / 21 :=
by
  sorry

end geom_seq_frac_l234_234589


namespace snowman_volume_l234_234822

noncomputable def volume_snowman (r₁ r₂ r₃ r_c h_c : ℝ) : ℝ :=
  (4 / 3 * Real.pi * r₁^3) + (4 / 3 * Real.pi * r₂^3) + (4 / 3 * Real.pi * r₃^3) + (Real.pi * r_c^2 * h_c)

theorem snowman_volume 
  : volume_snowman 4 6 8 3 5 = 1101 * Real.pi := 
by 
  sorry

end snowman_volume_l234_234822


namespace eve_ran_further_l234_234094

variable (ran_distance walked_distance difference_distance : ℝ)

theorem eve_ran_further (h1 : ran_distance = 0.7) (h2 : walked_distance = 0.6) : ran_distance - walked_distance = 0.1 := by
  sorry

end eve_ran_further_l234_234094


namespace odd_function_behavior_l234_234096

theorem odd_function_behavior (f : ℝ → ℝ)
  (h_odd: ∀ x, f (-x) = -f x)
  (h_increasing: ∀ x y, 3 ≤ x → x ≤ 7 → 3 ≤ y → y ≤ 7 → x < y → f x < f y)
  (h_max: ∀ x, 3 ≤ x → x ≤ 7 → f x ≤ 5) :
  (∀ x, -7 ≤ x → x ≤ -3 → f x ≥ -5) ∧ (∀ x y, -7 ≤ x → x ≤ -3 → -7 ≤ y → y ≤ -3 → x < y → f x < f y) :=
sorry

end odd_function_behavior_l234_234096


namespace age_twice_in_Y_years_l234_234700

def present_age_of_son : ℕ := 24
def age_difference := 26
def present_age_of_man : ℕ := present_age_of_son + age_difference

theorem age_twice_in_Y_years : 
  ∃ (Y : ℕ), present_age_of_man + Y = 2 * (present_age_of_son + Y) → Y = 2 :=
by
  sorry

end age_twice_in_Y_years_l234_234700


namespace product_of_integers_l234_234199

-- Define the conditions as variables in Lean
variables {x y : ℤ}

-- State the main theorem/proof
theorem product_of_integers (h1 : x + y = 8) (h2 : x^2 + y^2 = 34) : x * y = 15 := by
  sorry

end product_of_integers_l234_234199


namespace max_number_of_squares_with_twelve_points_l234_234256

-- Define the condition: twelve marked points in a grid
def twelve_points_marked_on_grid : Prop := 
  -- Assuming twelve specific points represented in a grid-like structure
  -- (This will be defined concretely in the proof implementation context)
  sorry

-- Define the problem statement to be proved
theorem max_number_of_squares_with_twelve_points : 
  twelve_points_marked_on_grid → (∃ n, n = 11) :=
by 
  sorry

end max_number_of_squares_with_twelve_points_l234_234256


namespace sin_double_angle_l234_234607

theorem sin_double_angle (α : ℝ)
  (h : Real.cos (Real.pi / 4 - α) = -3 / 5) :
  Real.sin (2 * α) = -7 / 25 := by
sorry

end sin_double_angle_l234_234607


namespace initial_volume_of_solution_l234_234076

variable (V : ℝ)

theorem initial_volume_of_solution :
  (0.05 * V + 5.5 = 0.15 * (V + 10)) → (V = 40) :=
by
  intro h
  sorry

end initial_volume_of_solution_l234_234076


namespace tanker_fill_rate_l234_234993

theorem tanker_fill_rate :
  let barrels_per_min := 2
  let liters_per_barrel := 159
  let cubic_meters_per_liter := 0.001
  let minutes_per_hour := 60
  let liters_per_min := barrels_per_min * liters_per_barrel
  let liters_per_hour := liters_per_min * minutes_per_hour
  let cubic_meters_per_hour := liters_per_hour * cubic_meters_per_liter
  cubic_meters_per_hour = 19.08 :=
  by {
    sorry
  }

end tanker_fill_rate_l234_234993


namespace transmit_data_time_l234_234508

def total_chunks (blocks: ℕ) (chunks_per_block: ℕ) : ℕ := blocks * chunks_per_block

def transmit_time (total_chunks: ℕ) (chunks_per_second: ℕ) : ℕ := total_chunks / chunks_per_second

def time_in_minutes (transmit_time_seconds: ℕ) : ℕ := transmit_time_seconds / 60

theorem transmit_data_time :
  ∀ (blocks chunks_per_block chunks_per_second : ℕ),
    blocks = 150 →
    chunks_per_block = 256 →
    chunks_per_second = 200 →
    time_in_minutes (transmit_time (total_chunks blocks chunks_per_block) chunks_per_second) = 3 := by
  intros
  sorry

end transmit_data_time_l234_234508


namespace f_eq_f_inv_implies_x_eq_0_l234_234490

noncomputable def f (x : ℝ) : ℝ := 3 * x^2 + 2 * x - 1
noncomputable def f_inv (x : ℝ) : ℝ := (-1 + Real.sqrt (3 * x + 4)) / 3

theorem f_eq_f_inv_implies_x_eq_0 (x : ℝ) : f x = f_inv x → x = 0 :=
by
  sorry

end f_eq_f_inv_implies_x_eq_0_l234_234490


namespace min_value_of_a_plus_b_l234_234759

theorem min_value_of_a_plus_b (a b : ℤ) (h1 : Even a) (h2 : Even b) (h3 : a * b = 144) : a + b = -74 :=
sorry

end min_value_of_a_plus_b_l234_234759


namespace tax_per_pound_is_one_l234_234051

-- Define the conditions
def bulk_price_per_pound : ℝ := 5          -- Condition 1
def minimum_spend : ℝ := 40               -- Condition 2
def total_paid : ℝ := 240                 -- Condition 4
def excess_pounds : ℝ := 32               -- Condition 5

-- Define the proof problem statement
theorem tax_per_pound_is_one :
  ∃ (T : ℝ), total_paid = (minimum_spend / bulk_price_per_pound + excess_pounds) * bulk_price_per_pound + 
  (minimum_spend / bulk_price_per_pound + excess_pounds) * T ∧ 
  T = 1 :=
by 
  sorry

end tax_per_pound_is_one_l234_234051


namespace true_discount_is_52_l234_234144

/-- The banker's gain on a bill due 3 years hence at 15% per annum is Rs. 23.4. -/
def BG : ℝ := 23.4

/-- The rate of interest per annum is 15%. -/
def R : ℝ := 15

/-- The time in years is 3. -/
def T : ℝ := 3

/-- The true discount is Rs. 52. -/
theorem true_discount_is_52 : BG * 100 / (R * T) = 52 :=
by
  -- Placeholder for proof. This needs proper calculation.
  sorry

end true_discount_is_52_l234_234144


namespace coffee_prices_purchase_ways_l234_234462

-- Define the cost equations for coffee A and B
def cost_equation1 (x y : ℕ) : Prop := 10 * x + 15 * y = 230
def cost_equation2 (x y : ℕ) : Prop := 25 * x + 25 * y = 450

-- Define what we need to prove for task 1
theorem coffee_prices (x y : ℕ) (h1 : cost_equation1 x y) (h2 : cost_equation2 x y) : x = 8 ∧ y = 10 := 
sorry

-- Define the condition for valid purchases of coffee A and B
def valid_purchase (m n : ℕ) : Prop := 8 * m + 10 * n = 200

-- Prove that there are 4 ways to purchase coffee A and B with 200 yuan
theorem purchase_ways : ∃ several : ℕ, several = 4 ∧ (∃ m n : ℕ, valid_purchase m n) := 
sorry

end coffee_prices_purchase_ways_l234_234462


namespace original_number_l234_234285

/-- Proof that the original three-digit number abc equals 118 under the given conditions. -/
theorem original_number (N : ℕ) (hN : N = 4332) (a b c : ℕ)
  (h : 100 * a + 10 * b + c = 118) :
  100 * a + 10 * b + c = 118 :=
by
  sorry

end original_number_l234_234285


namespace find_number_l234_234295

theorem find_number (x : ℝ) (h : 100 - x = x + 40) : x = 30 :=
sorry

end find_number_l234_234295


namespace sufficient_but_not_necessary_condition_for_x_lt_3_not_necessary_condition_for_x_lt_3_l234_234182

theorem sufficient_but_not_necessary_condition_for_x_lt_3 (x : ℝ) : |x - 1| < 2 → x < 3 :=
by {
  sorry
}

theorem not_necessary_condition_for_x_lt_3 (x : ℝ) : (x < 3) → ¬(-1 < x ∧ x < 3) :=
by {
  sorry
}

end sufficient_but_not_necessary_condition_for_x_lt_3_not_necessary_condition_for_x_lt_3_l234_234182


namespace arithmetic_sequence_sum_l234_234152

variable (a : ℕ → ℝ)
variable (d : ℝ)

-- The sequence is arithmetic
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) := ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (h_arith : is_arithmetic_sequence a d) (h_condition : a 2 + a 10 = 16) :
  a 4 + a 8 = 16 :=
sorry

end arithmetic_sequence_sum_l234_234152


namespace quadratic_expression_value_l234_234002

theorem quadratic_expression_value (a : ℝ) (h : a^2 - 2 * a - 3 = 0) : a^2 - 2 * a + 1 = 4 :=
by 
  -- Proof omitted for clarity in this part
  sorry 

end quadratic_expression_value_l234_234002


namespace ratio_proof_l234_234815

-- Define x and y as real numbers
variables (x y : ℝ)
-- Define the given condition
def given_condition : Prop := (3 * x - 2 * y) / (2 * x + y) = 3 / 4
-- Define the result to prove
def result : Prop := x / y = 11 / 6

-- State the theorem
theorem ratio_proof (h : given_condition x y) : result x y :=
by 
  sorry

end ratio_proof_l234_234815


namespace smallest_k_for_sixty_four_gt_four_nineteen_l234_234350

-- Definitions of the conditions
def sixty_four (k : ℕ) : ℕ := 64^k
def four_nineteen : ℕ := 4^19

-- The theorem to prove
theorem smallest_k_for_sixty_four_gt_four_nineteen (k : ℕ) : sixty_four k > four_nineteen ↔ k ≥ 7 := 
by
  sorry

end smallest_k_for_sixty_four_gt_four_nineteen_l234_234350


namespace taller_tree_height_is_108_l234_234643

variables (H : ℝ)

-- Conditions
def taller_tree_height := H
def shorter_tree_height := H - 18
def ratio_condition := (H - 18) / H = 5 / 6

-- Theorem to prove
theorem taller_tree_height_is_108 (hH : 0 < H) (h_ratio : ratio_condition H) : taller_tree_height H = 108 :=
sorry

end taller_tree_height_is_108_l234_234643


namespace abs_x_minus_one_sufficient_but_not_necessary_for_quadratic_l234_234995

theorem abs_x_minus_one_sufficient_but_not_necessary_for_quadratic (x : ℝ) :
  (|x - 1| < 2) → (x^2 - 4 * x - 5 < 0) ∧ ¬(x^2 - 4 * x - 5 < 0 → |x - 1| < 2) :=
by
  sorry

end abs_x_minus_one_sufficient_but_not_necessary_for_quadratic_l234_234995


namespace sum_c_d_l234_234714

theorem sum_c_d (c d : ℝ) (h : ∀ x, (x - 2) * (x + 3) = x^2 + c * x + d) :
  c + d = -5 :=
sorry

end sum_c_d_l234_234714


namespace fraction_eaten_correct_l234_234540

def initial_nuts : Nat := 30
def nuts_left : Nat := 5
def eaten_nuts : Nat := initial_nuts - nuts_left
def fraction_eaten : Rat := eaten_nuts / initial_nuts

theorem fraction_eaten_correct : fraction_eaten = 5 / 6 := by
  sorry

end fraction_eaten_correct_l234_234540


namespace semicircle_radius_l234_234614

-- Definition of the problem conditions
variables (a h : ℝ) -- base and height of the triangle
variable (R : ℝ)    -- radius of the semicircle

-- Statement of the proof problem
theorem semicircle_radius (h_pos : 0 < h) (a_pos : 0 < a) 
(semicircle_condition : ∀ R > 0, a * (h - R) = 2 * R * h) : R = a * h / (a + 2 * h) :=
sorry

end semicircle_radius_l234_234614


namespace value_of_a_l234_234307

-- Define the sets A and B and the intersection condition
def A (a : ℝ) : Set ℝ := {a ^ 2, a + 1, -3}
def B (a : ℝ) : Set ℝ := {a - 3, 2 * a - 1, a ^ 2 + 1}

theorem value_of_a (a : ℝ) (h : A a ∩ B a = {-3}) : a = -1 :=
by {
  -- Insert proof here when ready, using h to show a = -1
  sorry
}

end value_of_a_l234_234307


namespace original_number_of_workers_l234_234816

theorem original_number_of_workers (W A : ℕ)
  (h1 : W * 75 = A)
  (h2 : (W + 10) * 65 = A) :
  W = 65 :=
by
  sorry

end original_number_of_workers_l234_234816


namespace rowing_speed_in_still_water_l234_234065

noncomputable def speedInStillWater (distance_m : ℝ) (time_s : ℝ) (speed_current : ℝ) : ℝ :=
  let distance_km := distance_m / 1000
  let time_h := time_s / 3600
  let speed_downstream := distance_km / time_h
  speed_downstream - speed_current

theorem rowing_speed_in_still_water :
  speedInStillWater 45.5 9.099272058235341 8.5 = 9.5 :=
by
  sorry

end rowing_speed_in_still_water_l234_234065


namespace intersection_of_A_and_B_l234_234238

def setA : Set ℝ := {x : ℝ | |x| > 1}
def setB : Set ℝ := {x : ℝ | 0 < x ∧ x < 2}

theorem intersection_of_A_and_B : setA ∩ setB = {x : ℝ | 1 < x ∧ x < 2} :=
by
  sorry

end intersection_of_A_and_B_l234_234238


namespace intersection_M_N_l234_234979

def M : Set ℝ := {x | x^2 + x - 6 < 0}
def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

theorem intersection_M_N : M ∩ N = {x | 1 ≤ x ∧ x < 2} := by
  sorry

end intersection_M_N_l234_234979


namespace min_value_expression_l234_234304

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
    (x + 1 / y) ^ 2 + (y + 1 / (2 * x)) ^ 2 ≥ 3 + 2 * Real.sqrt 2 := 
sorry

end min_value_expression_l234_234304


namespace diff_hours_l234_234331

def hours_English : ℕ := 7
def hours_Spanish : ℕ := 4

theorem diff_hours : hours_English - hours_Spanish = 3 :=
by
  sorry

end diff_hours_l234_234331


namespace player_A_always_wins_l234_234886

theorem player_A_always_wins (a b c : ℤ) :
  ∃ (x1 x2 x3 : ℤ), x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ (x - x1) * (x - x2) * (x - x3) = x^3 + a*x^2 + b*x + c :=
sorry

end player_A_always_wins_l234_234886


namespace smallest_value_of_k_l234_234533

theorem smallest_value_of_k (k : ℝ) :
  (∃ x : ℝ, x^2 - 4 * x + k = 5) ↔ k >= 9 := 
sorry

end smallest_value_of_k_l234_234533


namespace number_of_people_who_bought_1_balloon_l234_234585

-- Define the variables and the main theorem statement
variables (x1 x2 x3 x4 : ℕ)

theorem number_of_people_who_bought_1_balloon : 
  (x1 + x2 + x3 + x4 = 101) → 
  (x1 + 2 * x2 + 3 * x3 + 4 * x4 = 212) →
  (x4 = x2 + 13) → 
  x1 = 52 :=
by
  intros h1 h2 h3
  sorry

end number_of_people_who_bought_1_balloon_l234_234585


namespace couch_cost_l234_234670

theorem couch_cost
  (C : ℕ)  -- Cost of the couch
  (table_cost : ℕ := 100)
  (lamp_cost : ℕ := 50)
  (amount_paid : ℕ := 500)
  (amount_owed : ℕ := 400)
  (total_furniture_cost : ℕ := C + table_cost + lamp_cost)
  (remaining_amount_owed : total_furniture_cost - amount_paid = amount_owed) :
   C = 750 := 
sorry

end couch_cost_l234_234670


namespace domain_of_function_l234_234628

theorem domain_of_function (x : ℝ) : 
  {x | ∃ k : ℤ, - (Real.pi / 3) + (2 : ℝ) * k * Real.pi ≤ x ∧ x ≤ (Real.pi / 3) + (2 : ℝ) * k * Real.pi} :=
by
  -- Proof omitted
  sorry

end domain_of_function_l234_234628


namespace x_equals_y_squared_plus_2y_minus_1_l234_234288

theorem x_equals_y_squared_plus_2y_minus_1 (x y : ℝ) (h : x / (x - 1) = (y^2 + 2 * y - 1) / (y^2 + 2 * y - 2)) : 
  x = y^2 + 2 * y - 1 :=
sorry

end x_equals_y_squared_plus_2y_minus_1_l234_234288


namespace tan_half_angle_l234_234734

theorem tan_half_angle (α : ℝ) (h1 : Real.sin α + Real.cos α = 1 / 5)
  (h2 : 3 * π / 2 < α ∧ α < 2 * π) : 
  Real.tan (α / 2) = -1 / 3 :=
sorry

end tan_half_angle_l234_234734


namespace sum_odd_even_integers_l234_234719

theorem sum_odd_even_integers :
  let odd_terms_sum := (15 / 2) * (1 + 29)
  let even_terms_sum := (10 / 2) * (2 + 20)
  odd_terms_sum + even_terms_sum = 335 :=
by
  let odd_terms_sum := (15 / 2) * (1 + 29)
  let even_terms_sum := (10 / 2) * (2 + 20)
  show odd_terms_sum + even_terms_sum = 335
  sorry

end sum_odd_even_integers_l234_234719


namespace valentines_left_l234_234491

theorem valentines_left (initial_valentines given_away : ℕ) (h_initial : initial_valentines = 30) (h_given : given_away = 8) :
  initial_valentines - given_away = 22 :=
by {
  sorry
}

end valentines_left_l234_234491


namespace bacteria_count_correct_l234_234155

-- Define the initial number of bacteria
def initial_bacteria : ℕ := 800

-- Define the doubling time in hours
def doubling_time : ℕ := 3

-- Define the function that calculates the number of bacteria after t hours
noncomputable def bacteria_after (t : ℕ) : ℕ :=
  initial_bacteria * 2 ^ (t / doubling_time)

-- Define the target number of bacteria
def target_bacteria : ℕ := 51200

-- Define the specific time we want to prove the bacteria count equals the target
def specific_time : ℕ := 18

-- Prove that after 18 hours, there will be exactly 51,200 bacteria
theorem bacteria_count_correct : bacteria_after specific_time = target_bacteria :=
  sorry

end bacteria_count_correct_l234_234155


namespace logarithm_function_decreasing_l234_234026

theorem logarithm_function_decreasing (a : ℝ) : 
  (∀ x ∈ Set.Ici (-1), (3 * x^2 - a * x + 5) ≤ (3 * x^2 - a * (x + 1) + 5)) ↔ (-8 < a ∧ a ≤ -6) :=
by
  sorry

end logarithm_function_decreasing_l234_234026


namespace net_price_change_is_twelve_percent_l234_234426

variable (P : ℝ)

def net_price_change (P : ℝ) : ℝ := 
  let decreased_price := 0.8 * P
  let increased_price := 1.4 * decreased_price
  increased_price - P

theorem net_price_change_is_twelve_percent (P : ℝ) : net_price_change P = 0.12 * P := by
  sorry

end net_price_change_is_twelve_percent_l234_234426


namespace pat_kate_ratio_l234_234710

theorem pat_kate_ratio 
  (P K M : ℕ)
  (h1 : P + K + M = 117)
  (h2 : ∃ r : ℕ, P = r * K)
  (h3 : P = M / 3)
  (h4 : M = K + 65) : 
  P / K = 2 :=
by
  sorry

end pat_kate_ratio_l234_234710


namespace last_digit_base5_of_M_l234_234373

theorem last_digit_base5_of_M (d e f : ℕ) (hd : d < 5) (he : e < 5) (hf : f < 5)
  (h : 25 * d + 5 * e + f = 64 * f + 8 * e + d) : f = 0 :=
by
  sorry

end last_digit_base5_of_M_l234_234373


namespace period_of_f_max_value_of_f_and_values_l234_234593

noncomputable def f (x : ℝ) : ℝ := (1 - Real.sin (2 * x)) / (Real.sin x - Real.cos x)

-- Statement 1: The period of f(x) is 2π
theorem period_of_f : ∀ x, f (x + 2 * Real.pi) = f x := by
  sorry

-- Statement 2: The maximum value of f(x) is √2 and it is attained at x = 2kπ + 3π/4, k ∈ ℤ
theorem max_value_of_f_and_values :
  (∀ x, f x ≤ Real.sqrt 2) ∧
  (∃ k : ℤ, f (2 * k * Real.pi + 3 * Real.pi / 4) = Real.sqrt 2) := by
  sorry

end period_of_f_max_value_of_f_and_values_l234_234593


namespace estimated_red_balls_l234_234953

theorem estimated_red_balls
  (total_balls : ℕ)
  (total_draws : ℕ)
  (red_draws : ℕ)
  (h_total_balls : total_balls = 12)
  (h_total_draws : total_draws = 200)
  (h_red_draws : red_draws = 50) :
  red_draws * total_balls = total_draws * 3 :=
by
  sorry

end estimated_red_balls_l234_234953


namespace men_in_first_group_l234_234750

noncomputable def first_group_men (x m b W : ℕ) : Prop :=
  let eq1 := 10 * x * m + 80 * b = W
  let eq2 := 2 * (26 * m + 48 * b) = W
  let eq3 := 4 * (15 * m + 20 * b) = W
  eq1 ∧ eq2 ∧ eq3

theorem men_in_first_group (m b W : ℕ) (h_condition : first_group_men 6 m b W) : 
  ∃ x, x = 6 :=
by
  sorry

end men_in_first_group_l234_234750


namespace number_of_students_l234_234783

theorem number_of_students (n : ℕ) (h1 : 90 - n = n / 2) : n = 60 :=
by
  sorry

end number_of_students_l234_234783


namespace prime_solution_exists_l234_234497

theorem prime_solution_exists (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) :
  p^2 + 1 = 74 * (q^2 + r^2) → (p = 31 ∧ q = 2 ∧ r = 3) :=
by
  sorry

end prime_solution_exists_l234_234497


namespace parallel_line_perpendicular_line_l234_234447

theorem parallel_line (x y : ℝ) (h : y = 2 * x + 3) : ∃ a : ℝ, 3 * x - 2 * y + a = 0 :=
by
  use 1
  sorry

theorem perpendicular_line  (x y : ℝ) (h : y = -x / 2) : ∃ c : ℝ, 3 * x - 2 * y + c = 0 :=
by
  use -5
  sorry

end parallel_line_perpendicular_line_l234_234447


namespace find_y_l234_234041

theorem find_y (y : ℝ) : (∃ y : ℝ, (4, y) ≠ (2, -3) ∧ ((-3 - y) / (2 - 4) = 1)) → y = -1 :=
by
  sorry

end find_y_l234_234041
