import Mathlib

namespace triangle_inequality_l1233_123362

-- Define the conditions as Lean hypotheses
variables {a b c : ℝ}

-- Lean statement for the problem
theorem triangle_inequality (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 1) : 
  a^2 + b^2 + c^2 + 4*a*b*c < 1/2 :=
sorry

end triangle_inequality_l1233_123362


namespace sum_of_four_consecutive_integers_divisible_by_two_l1233_123382

theorem sum_of_four_consecutive_integers_divisible_by_two (n : ℤ) : 
  2 ∣ ((n-1) + n + (n+1) + (n+2)) :=
by
  sorry

end sum_of_four_consecutive_integers_divisible_by_two_l1233_123382


namespace planks_ratio_l1233_123343

theorem planks_ratio (P S : ℕ) (H : S + 100 + 20 + 30 = 200) (T : P = 200) (R : S = 200 / 2) : 
(S : ℚ) / P = 1 / 2 :=
by
  sorry

end planks_ratio_l1233_123343


namespace max_value_is_one_eighth_l1233_123360

noncomputable def find_max_value (a b c : ℝ) : ℝ :=
  a^2 * b^2 * c^2 * (a + b + c) / ((a + b)^3 * (b + c)^3)

theorem max_value_is_one_eighth (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c):
  find_max_value a b c ≤ 1 / 8 :=
by
  sorry

end max_value_is_one_eighth_l1233_123360


namespace fraction_to_decimal_l1233_123361

theorem fraction_to_decimal : (3 : ℝ) / 50 = 0.06 := by
  sorry

end fraction_to_decimal_l1233_123361


namespace equidistant_point_l1233_123340

/-- 
  Find the point in the xz-plane that is equidistant from the points (1, 0, 0), 
  (0, -2, 3), and (4, 2, -2). The point in question is \left( \frac{41}{7}, 0, -\frac{19}{14} \right).
-/
theorem equidistant_point :
  ∃ (x z : ℚ), 
    (x - 1)^2 + z^2 = x^2 + 4 + (z - 3)^2 ∧
    (x - 1)^2 + z^2 = (x - 4)^2 + 4 + (z + 2)^2 ∧
    x = 41 / 7 ∧ z = -19 / 14 :=
by
  sorry

end equidistant_point_l1233_123340


namespace convert_to_polar_l1233_123394

noncomputable def polar_coordinates (x y : ℝ) : ℝ × ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let θ := Real.arctan (y / x)
  (r, θ)

theorem convert_to_polar (x y : ℝ) (hx : x = 8) (hy : y = 3 * Real.sqrt 3) :
  polar_coordinates x y = (Real.sqrt 91, Real.arctan (3 * Real.sqrt 3 / 8)) :=
by
  rw [hx, hy]
  simp [polar_coordinates]
  -- place to handle conversions and simplifications if necessary
  sorry

end convert_to_polar_l1233_123394


namespace perfect_squares_in_range_100_400_l1233_123349

theorem perfect_squares_in_range_100_400 : ∃ n : ℕ, (∀ m, 100 ≤ m^2 → m^2 ≤ 400 → m^2 = (m - 10 + 1)^2) ∧ n = 9 := 
by
  sorry

end perfect_squares_in_range_100_400_l1233_123349


namespace chords_intersecting_theorem_l1233_123352

noncomputable def intersecting_chords_theorem (P A B C D : ℝ) (h_circle : P ≠ A) (h_ab : A ≠ B) (h_cd : C ≠ D) : ℝ :=
  sorry

theorem chords_intersecting_theorem (P A B C D : ℝ) (h_circle : P ≠ A) (h_ab : A ≠ B) (h_cd : C ≠ D) :
  (P - A) * (P - B) = (P - C) * (P - D) :=
by sorry

end chords_intersecting_theorem_l1233_123352


namespace initial_volume_of_solution_is_six_l1233_123366

theorem initial_volume_of_solution_is_six
  (V : ℝ)
  (h1 : 0.30 * V + 2.4 = 0.50 * (V + 2.4)) :
  V = 6 :=
by
  sorry

end initial_volume_of_solution_is_six_l1233_123366


namespace smallest_divisor_l1233_123375

noncomputable def even_four_digit_number (m : ℕ) : Prop :=
  1000 ≤ m ∧ m < 10000 ∧ m % 2 = 0

def divisor_ordered (m : ℕ) (d : ℕ) : Prop :=
  d ∣ m

theorem smallest_divisor (m : ℕ) (h1 : even_four_digit_number m) (h2 : divisor_ordered m 437) :
  ∃ d,  d > 437 ∧ divisor_ordered m d ∧ (∀ e, e > 437 → divisor_ordered m e → d ≤ e) ∧ d = 874 :=
sorry

end smallest_divisor_l1233_123375


namespace sum_of_decimals_as_fraction_l1233_123372

/-- Define the problem inputs as constants -/
def d1 : ℚ := 2 / 10
def d2 : ℚ := 4 / 100
def d3 : ℚ := 6 / 1000
def d4 : ℚ := 8 / 10000
def d5 : ℚ := 1 / 100000

/-- The main theorem statement -/
theorem sum_of_decimals_as_fraction : 
  d1 + d2 + d3 + d4 + d5 = 24681 / 100000 := 
by 
  sorry

end sum_of_decimals_as_fraction_l1233_123372


namespace inequality_solution_l1233_123337

theorem inequality_solution (x : ℝ) : (x ≠ -2) ↔ (0 ≤ x^2 / (x + 2)^2) := by
  sorry

end inequality_solution_l1233_123337


namespace functional_equation_solution_l1233_123309

theorem functional_equation_solution (f : ℚ → ℚ)
  (H : ∀ x y : ℚ, f (x + y) + f (x - y) = 2 * f x + 2 * f y) :
  ∃ a : ℚ, ∀ x : ℚ, f x = a * x^2 :=
by
  sorry

end functional_equation_solution_l1233_123309


namespace ball_distribution_l1233_123399

theorem ball_distribution (n : ℕ) (P_white P_red P_yellow : ℚ) (num_white num_red num_yellow : ℕ) 
  (total_balls : n = 6)
  (prob_white : P_white = 1/2)
  (prob_red : P_red = 1/3)
  (prob_yellow : P_yellow = 1/6) :
  num_white = 3 ∧ num_red = 2 ∧ num_yellow = 1 := 
sorry

end ball_distribution_l1233_123399


namespace proof_g_2_l1233_123312

def g (x : ℝ) : ℝ := 3 * x ^ 8 - 4 * x ^ 4 + 2 * x ^ 2 - 6

theorem proof_g_2 :
  g (-2) = 10 → g (2) = 1402 := by
  sorry

end proof_g_2_l1233_123312


namespace rahul_deepak_age_ratio_l1233_123369

-- Define the conditions
variables (R D : ℕ)
axiom deepak_age : D = 33
axiom rahul_future_age : R + 6 = 50

-- Define the theorem to prove the ratio
theorem rahul_deepak_age_ratio : R / D = 4 / 3 :=
by
  -- Placeholder for proof
  sorry

end rahul_deepak_age_ratio_l1233_123369


namespace sticks_difference_l1233_123308

def sticks_picked_up : ℕ := 14
def sticks_left : ℕ := 4

theorem sticks_difference : (sticks_picked_up - sticks_left) = 10 := by
  sorry

end sticks_difference_l1233_123308


namespace C_younger_than_A_l1233_123389

variables (A B C : ℕ)

-- Original Condition
axiom age_condition : A + B = B + C + 17

-- Lean Statement to Prove
theorem C_younger_than_A (A B C : ℕ) (h : A + B = B + C + 17) : C + 17 = A :=
by {
  -- Proof would go here but is omitted.
  sorry
}

end C_younger_than_A_l1233_123389


namespace meaningful_fraction_l1233_123324

theorem meaningful_fraction (x : ℝ) : (∃ y : ℝ, y = 5 / (x - 3)) ↔ x ≠ 3 :=
by
  sorry

end meaningful_fraction_l1233_123324


namespace watermelons_eaten_l1233_123358

theorem watermelons_eaten (original left : ℕ) (h1 : original = 4) (h2 : left = 1) :
  original - left = 3 :=
by {
  -- Providing the proof steps is not necessary as per the instructions
  sorry
}

end watermelons_eaten_l1233_123358


namespace rectangle_area_l1233_123391

-- Define the length and width of the rectangle based on given ratio
def length (k: ℝ) := 5 * k
def width (k: ℝ) := 2 * k

-- The perimeter condition
def perimeter (k: ℝ) := 2 * (length k) + 2 * (width k) = 280

-- The diagonal condition
def diagonal_condition (k: ℝ) := (width k) * Real.sqrt 2 = (length k) / 2

-- The area of the rectangle
def area (k: ℝ) := (length k) * (width k)

-- The main theorem to be proven
theorem rectangle_area : ∃ k: ℝ, perimeter k ∧ diagonal_condition k ∧ area k = 4000 :=
by
  sorry

end rectangle_area_l1233_123391


namespace shampoo_duration_l1233_123376

-- Conditions
def rose_shampoo : ℚ := 1/3
def jasmine_shampoo : ℚ := 1/4
def daily_usage : ℚ := 1/12

-- Question
theorem shampoo_duration : (rose_shampoo + jasmine_shampoo) / daily_usage = 7 := by
  sorry

end shampoo_duration_l1233_123376


namespace smallest_base_b_l1233_123373

theorem smallest_base_b (b : ℕ) : (b ≥ 1) → (b^2 ≤ 82) → (82 < b^3) → b = 5 := by
  sorry

end smallest_base_b_l1233_123373


namespace group1_calculation_group2_calculation_l1233_123313

theorem group1_calculation : 9 / 3 * (9 - 1) = 24 := by
  sorry

theorem group2_calculation : 7 * (3 + 3 / 7) = 24 := by
  sorry

end group1_calculation_group2_calculation_l1233_123313


namespace sum_of_reciprocals_eq_six_l1233_123354

theorem sum_of_reciprocals_eq_six (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x + y = 6 * x * y) (h2 : y = 2 * x) :
  (1 / x) + (1 / y) = 6 := by
  sorry

end sum_of_reciprocals_eq_six_l1233_123354


namespace average_monthly_income_P_and_R_l1233_123350

theorem average_monthly_income_P_and_R 
  (P Q R : ℝ)
  (h1 : (P + Q) / 2 = 5050)
  (h2 : (Q + R) / 2 = 6250)
  (h3 : P = 4000) :
  (P + R) / 2 = 5200 :=
sorry

end average_monthly_income_P_and_R_l1233_123350


namespace number_of_tons_is_3_l1233_123339

noncomputable def calculate_tons_of_mulch {total_cost price_per_pound pounds_per_ton : ℝ} 
  (h_total_cost : total_cost = 15000) 
  (h_price_per_pound : price_per_pound = 2.5) 
  (h_pounds_per_ton : pounds_per_ton = 2000) : ℝ := 
  total_cost / price_per_pound / pounds_per_ton

theorem number_of_tons_is_3 
  (total_cost price_per_pound pounds_per_ton : ℝ) 
  (h_total_cost : total_cost = 15000) 
  (h_price_per_pound : price_per_pound = 2.5) 
  (h_pounds_per_ton : pounds_per_ton = 2000) : 
  calculate_tons_of_mulch h_total_cost h_price_per_pound h_pounds_per_ton = 3 := 
by
  sorry

end number_of_tons_is_3_l1233_123339


namespace parabola_directrix_l1233_123348

-- Defining the given condition
def given_parabola_equation (x y : ℝ) : Prop := y = 2 * x^2

-- Defining the expected directrix equation for the parabola
def directrix_equation (y : ℝ) : Prop := y = -1 / 8

-- The theorem we aim to prove
theorem parabola_directrix :
  (∀ x y : ℝ, given_parabola_equation x y) → (directrix_equation (-1 / 8)) :=
by
  -- Using 'sorry' here since the proof is not required
  sorry

end parabola_directrix_l1233_123348


namespace volume_frustum_correct_l1233_123329

noncomputable def volume_of_frustum 
  (base_edge_orig : ℝ) 
  (altitude_orig : ℝ) 
  (base_edge_small : ℝ) 
  (altitude_small : ℝ) : ℝ :=
  let volume_ratio := (base_edge_small / base_edge_orig) ^ 3
  let base_area_orig := (Real.sqrt 3 / 4) * base_edge_orig ^ 2
  let volume_orig := (1 / 3) * base_area_orig * altitude_orig
  let volume_small := volume_ratio * volume_orig
  let volume_frustum := volume_orig - volume_small
  volume_frustum

theorem volume_frustum_correct :
  volume_of_frustum 18 9 9 3 = 212.625 * Real.sqrt 3 :=
sorry

end volume_frustum_correct_l1233_123329


namespace find_five_value_l1233_123374

def f (x : ℝ) : ℝ := x^2 - x

theorem find_five_value : f 5 = 20 := by
  sorry

end find_five_value_l1233_123374


namespace max_red_balls_l1233_123335

theorem max_red_balls (R B G : ℕ) (h1 : G = 12) (h2 : R + B + G = 28) (h3 : R + G < 24) : R ≤ 11 := 
by
  sorry

end max_red_balls_l1233_123335


namespace men_with_ac_at_least_12_l1233_123345

-- Define the variables and conditions
variable (total_men : ℕ) (married_men : ℕ) (tv_men : ℕ) (radio_men : ℕ) (men_with_all_four : ℕ)

-- Assume the given conditions
axiom h1 : total_men = 100
axiom h2 : married_men = 82
axiom h3 : tv_men = 75
axiom h4 : radio_men = 85
axiom h5 : men_with_all_four = 12

-- Define the number of men with AC
variable (ac_men : ℕ)

-- State the proposition that the number of men with AC is at least 12
theorem men_with_ac_at_least_12 : ac_men ≥ 12 := sorry

end men_with_ac_at_least_12_l1233_123345


namespace revenue_from_full_price_tickets_l1233_123307

-- Let's define our variables and assumptions
variables (f h p: ℕ)

-- Total number of tickets sold
def total_tickets (f h: ℕ) : Prop := f + h = 200

-- Total revenue from tickets
def total_revenue (f h p: ℕ) : Prop := f * p + h * (p / 3) = 2500

-- Statement to prove the revenue from full-price tickets
theorem revenue_from_full_price_tickets (f h p: ℕ) (hf: total_tickets f h) 
  (hr: total_revenue f h p): f * p = 1250 :=
sorry

end revenue_from_full_price_tickets_l1233_123307


namespace find_a_l1233_123342

noncomputable def pure_imaginary_simplification (a : ℝ) (i : ℂ) (hi : i * i = -1) : Prop :=
  let denom := (3 : ℂ) - (4 : ℂ) * i
  let numer := (15 : ℂ)
  let complex_num := a + numer / denom
  let simplified_real := a + (9 : ℝ) / (5 : ℝ)
  simplified_real = 0

theorem find_a (i : ℂ) (hi : i * i = -1) : pure_imaginary_simplification (- 9 / 5 : ℝ) i hi :=
by
  sorry

end find_a_l1233_123342


namespace min_value_of_f_l1233_123386

noncomputable def f (a b x : ℝ) : ℝ :=
  (a / (Real.sin x) ^ 2) + b * (Real.sin x) ^ 2

theorem min_value_of_f (a b : ℝ) (h1 : a = 2) (h2 : b = 1) (h3 : a > b) (h4 : b > 0) :
  ∃ x, f a b x = 3 := 
sorry

end min_value_of_f_l1233_123386


namespace max_fruit_to_teacher_l1233_123377

theorem max_fruit_to_teacher (A G : ℕ) : (A % 7 ≤ 6) ∧ (G % 7 ≤ 6) :=
by
  sorry

end max_fruit_to_teacher_l1233_123377


namespace binomial_np_sum_l1233_123364

-- Definitions of variance and expectation for a binomial distribution
def binomial_variance (n : ℕ) (p : ℚ) : ℚ := n * p * (1 - p)
def binomial_expectation (n : ℕ) (p : ℚ) : ℚ := n * p

-- Statement of the problem
theorem binomial_np_sum (n : ℕ) (p : ℚ) (h_var : binomial_variance n p = 4) (h_exp : binomial_expectation n p = 12) :
    n + p = 56 / 3 := by
  sorry

end binomial_np_sum_l1233_123364


namespace pow_mod_remainder_l1233_123321

theorem pow_mod_remainder :
  (3 ^ 2023) % 5 = 2 :=
by sorry

end pow_mod_remainder_l1233_123321


namespace Pyarelal_loss_l1233_123383

variables (capital_of_pyarelal capital_of_ashok : ℝ) (total_loss : ℝ)

def is_ninth (a b : ℝ) : Prop := a = b / 9

def applied_loss (loss : ℝ) (ratio : ℝ) : ℝ := ratio * loss

theorem Pyarelal_loss (h1: is_ninth capital_of_ashok capital_of_pyarelal) 
                        (h2: total_loss = 1600) : 
                        applied_loss total_loss (9/10) = 1440 :=
by 
  unfold is_ninth at h1
  sorry

end Pyarelal_loss_l1233_123383


namespace sqrt_value_l1233_123315

theorem sqrt_value (h : Real.sqrt 100.4004 = 10.02) : Real.sqrt 1.004004 = 1.002 := 
by
  sorry

end sqrt_value_l1233_123315


namespace find_multiple_of_brothers_l1233_123384

theorem find_multiple_of_brothers : 
  ∃ x : ℕ, (x * 4) - 2 = 6 :=
by
  -- Provide the correct Lean statement for the problem
  sorry

end find_multiple_of_brothers_l1233_123384


namespace jo_climb_stairs_ways_l1233_123318

def f : ℕ → ℕ
| 0 => 1
| 1 => 1
| 2 => 2
| (n + 3) => f (n + 2) + f (n + 1) + f n

theorem jo_climb_stairs_ways : f 8 = 81 :=
by
    sorry

end jo_climb_stairs_ways_l1233_123318


namespace subject_difference_l1233_123359

-- Define the problem in terms of conditions and question
theorem subject_difference (C R M : ℕ) (hC : C = 10) (hR : R = C + 4) (hM : M + R + C = 41) : M - R = 3 :=
by
  -- Lean expects a proof here, we skip it with sorry
  sorry

end subject_difference_l1233_123359


namespace assign_students_to_villages_l1233_123326

theorem assign_students_to_villages (n m : ℕ) (hn : n = 5) (hm : m = 3) :
  ∃ N : ℕ, N = 70 ∧ 
  (∃ (f : Fin n → Fin m), (∀ i j, f i = f j ↔ i = j) ∧ 
  (∀ x : Fin m, ∃ y : Fin n, f y = x)) :=
by
  sorry

end assign_students_to_villages_l1233_123326


namespace find_x_for_opposite_expressions_l1233_123355

theorem find_x_for_opposite_expressions :
  ∃ x : ℝ, (x + 1) + (3 * x - 5) = 0 ↔ x = 1 :=
by
  sorry

end find_x_for_opposite_expressions_l1233_123355


namespace sum_a_b_when_pow_is_max_l1233_123367

theorem sum_a_b_when_pow_is_max (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 1) (h_pow : a^b < 500) 
(h_max : ∀ (a' b' : ℕ), (a' > 0) -> (b' > 1) -> (a'^b' < 500) -> a^b >= a'^b') : a + b = 24 := by
  sorry

end sum_a_b_when_pow_is_max_l1233_123367


namespace total_bottles_in_box_l1233_123351

def dozens (n : ℕ) := 12 * n

def water_bottles : ℕ := dozens 2

def apple_bottles : ℕ := water_bottles + 6

def total_bottles : ℕ := water_bottles + apple_bottles

theorem total_bottles_in_box : total_bottles = 54 := 
by
  sorry

end total_bottles_in_box_l1233_123351


namespace has_exactly_one_zero_interval_l1233_123338

noncomputable def f (a x : ℝ) : ℝ := x^2 - a*x + 1

theorem has_exactly_one_zero_interval (a : ℝ) (h : a > 3) : ∃! x, 0 < x ∧ x < 2 ∧ f a x = 0 :=
sorry

end has_exactly_one_zero_interval_l1233_123338


namespace polynomial_evaluation_l1233_123301

-- Define operations using Lean syntax
def star (a b : ℚ) := a + b
def otimes (a b : ℚ) := a - b

-- Define a function to represent the polynomial expression
def expression (a b : ℚ) := star (a^2 * b) (3 * a * b) + otimes (5 * a^2 * b) (4 * a * b)

theorem polynomial_evaluation (a b : ℚ) (ha : a = 5) (hb : b = 3) : expression a b = 435 := by
  sorry

end polynomial_evaluation_l1233_123301


namespace evaluate_expression_simplified_l1233_123319

theorem evaluate_expression_simplified (x : ℝ) (h : x = Real.sqrt 2) : 
  (x + 3) ^ 2 + (x + 2) * (x - 2) - x * (x + 6) = 7 := by
  rw [h]
  sorry

end evaluate_expression_simplified_l1233_123319


namespace investment_duration_l1233_123356

theorem investment_duration 
  (P : ℝ) (A : ℝ) (r : ℝ) (t : ℝ)
  (h1 : P = 939.60)
  (h2 : A = 1120)
  (h3 : r = 8) :
  t = 2.4 :=
by
  sorry

end investment_duration_l1233_123356


namespace gift_box_spinning_tops_l1233_123381

theorem gift_box_spinning_tops
  (red_box_cost : ℕ) (red_box_tops : ℕ)
  (yellow_box_cost : ℕ) (yellow_box_tops : ℕ)
  (total_spent : ℕ) (total_boxes : ℕ)
  (h_red_box_cost : red_box_cost = 5)
  (h_red_box_tops : red_box_tops = 3)
  (h_yellow_box_cost : yellow_box_cost = 9)
  (h_yellow_box_tops : yellow_box_tops = 5)
  (h_total_spent : total_spent = 600)
  (h_total_boxes : total_boxes = 72) :
  ∃ (red_boxes : ℕ) (yellow_boxes : ℕ), (red_boxes + yellow_boxes = total_boxes) ∧
  (red_box_cost * red_boxes + yellow_box_cost * yellow_boxes = total_spent) ∧
  (red_box_tops * red_boxes + yellow_box_tops * yellow_boxes = 336) :=
by
  sorry

end gift_box_spinning_tops_l1233_123381


namespace first_interest_rate_l1233_123320

theorem first_interest_rate (r : ℝ) : 
  (70000:ℝ) = (60000:ℝ) + (10000:ℝ) →
  (8000:ℝ) = (60000 * r / 100) + (10000 * 20 / 100) →
  r = 10 :=
by
  intros h1 h2
  sorry

end first_interest_rate_l1233_123320


namespace cos_difference_identity_cos_phi_value_l1233_123396

variables (α β θ φ : ℝ)
variables (a b : ℝ × ℝ)

-- Part I
theorem cos_difference_identity (hα : 0 ≤ α ∧ α ≤ 2 * Real.pi) (hβ : 0 ≤ β ∧ β ≤ 2 * Real.pi) : 
  Real.cos (α - β) = Real.cos α * Real.cos β + Real.sin α * Real.sin β :=
sorry

-- Part II
theorem cos_phi_value (hθ : 0 < θ ∧ θ < Real.pi / 2) (hφ : 0 < φ ∧ φ < Real.pi / 2)
  (ha : a = (Real.sin θ, -2)) (hb : b = (1, Real.cos θ)) (dot_ab_zero : a.1 * b.1 + a.2 * b.2 = 0)
  (h_sin_diff : Real.sin (theta - phi) = Real.sqrt 10 / 10) :
  Real.cos φ = Real.sqrt 2 / 2 :=
sorry

end cos_difference_identity_cos_phi_value_l1233_123396


namespace int_valued_fractions_l1233_123316

theorem int_valued_fractions (a : ℤ) :
  ∃ k : ℤ, (a^2 - 21 * a + 17) = k * a ↔ a = 1 ∨ a = -1 ∨ a = 17 ∨ a = -17 :=
by {
  sorry
}

end int_valued_fractions_l1233_123316


namespace sum_digits_B_of_4444_4444_l1233_123317

noncomputable def sum_digits (n : ℕ) : ℕ :=
  n.digits 10 |> List.sum

theorem sum_digits_B_of_4444_4444 :
  let A : ℕ := sum_digits (4444 ^ 4444)
  let B : ℕ := sum_digits A
  sum_digits B = 7 :=
by
  sorry

end sum_digits_B_of_4444_4444_l1233_123317


namespace number_of_integers_satisfying_condition_l1233_123385

def satisfies_condition (n : ℤ) : Prop :=
  1 + Int.floor (101 * n / 102) = Int.ceil (98 * n / 99)

noncomputable def number_of_solutions : ℤ :=
  10198

theorem number_of_integers_satisfying_condition :
  (∃ n : ℤ, satisfies_condition n) ↔ number_of_solutions = 10198 :=
sorry

end number_of_integers_satisfying_condition_l1233_123385


namespace Hillary_activities_LCM_l1233_123322

theorem Hillary_activities_LCM :
  let swim := 6
  let run := 4
  let cycle := 16
  Nat.lcm (Nat.lcm swim run) cycle = 48 :=
by
  sorry

end Hillary_activities_LCM_l1233_123322


namespace proof_problem_l1233_123357

noncomputable def M : Set ℝ := { x | x ≥ 2 }
noncomputable def a : ℝ := Real.pi

theorem proof_problem : a ∈ M ∧ {a} ⊂ M :=
by
  sorry

end proof_problem_l1233_123357


namespace closest_ratio_adults_children_l1233_123311

theorem closest_ratio_adults_children (a c : ℕ) 
  (h1 : 30 * a + 15 * c = 2250) 
  (h2 : a ≥ 50) 
  (h3 : c ≥ 20) : a = 50 ∧ c = 50 :=
by {
  sorry
}

end closest_ratio_adults_children_l1233_123311


namespace slope_of_intersection_points_l1233_123398

theorem slope_of_intersection_points :
  ∀ s : ℝ, ∃ k b : ℝ, (∀ (x y : ℝ), (2 * x - 3 * y = 4 * s + 6) ∧ (2 * x + y = 3 * s + 1) → y = k * x + b) ∧ k = -2/13 := 
by
  intros s
  -- Proof to be provided here
  sorry

end slope_of_intersection_points_l1233_123398


namespace quadrilateral_area_l1233_123303

-- Define the dimensions of the rectangles
variables (AB BC EF FG : ℝ)
variables (AFCH_area : ℝ)

-- State the conditions explicitly
def conditions : Prop :=
  (AB = 9) ∧ 
  (BC = 5) ∧ 
  (EF = 3) ∧ 
  (FG = 10)

-- State the theorem to prove
theorem quadrilateral_area (h: conditions AB BC EF FG) : 
  AFCH_area = 52.5 := 
sorry

end quadrilateral_area_l1233_123303


namespace mistaken_divisor_is_12_l1233_123327

-- Definitions based on conditions
def correct_divisor : ℕ := 21
def correct_quotient : ℕ := 36
def mistaken_quotient : ℕ := 63

-- The mistaken divisor  is computed as:
def mistaken_divisor : ℕ := correct_quotient * correct_divisor / mistaken_quotient

-- The theorem to prove the mistaken divisor is 12
theorem mistaken_divisor_is_12 : mistaken_divisor = 12 := by
  sorry

end mistaken_divisor_is_12_l1233_123327


namespace leak_drain_time_l1233_123390

theorem leak_drain_time (P L : ℝ) (hP : P = 1/2) (h_combined : P - L = 3/7) : 1 / L = 14 :=
by
  -- Definitions of the conditions
  -- The rate of the pump filling the tank
  have hP : P = 1 / 2 := hP
  -- The combined rate of the pump (filling) and leak (draining)
  have h_combined : P - L = 3 / 7 := h_combined
  -- From these definitions, continue the proof
  sorry

end leak_drain_time_l1233_123390


namespace evaluate_series_l1233_123323

noncomputable def infinite_series :=
  ∑' n, (n^3 + 2*n^2 - 3) / (n+3).factorial

theorem evaluate_series : infinite_series = 1 / 4 :=
by
  sorry

end evaluate_series_l1233_123323


namespace periodic_sum_constant_l1233_123353

noncomputable def is_periodic (f : ℝ → ℝ) (a : ℝ) : Prop :=
a ≠ 0 ∧ ∀ x : ℝ, f (a + x) = f x

theorem periodic_sum_constant (f g : ℝ → ℝ) (a b : ℝ)
  (ha : a ≠ 0) (hb : b ≠ 0) (hfa : is_periodic f a) (hgb : is_periodic g b)
  (harational : ∃ m n : ℤ, (a : ℝ) = m / n) (hbirrational : ¬ ∃ m n : ℤ, (b : ℝ) = m / n) :
  (∃ c : ℝ, c ≠ 0 ∧ ∀ x : ℝ, (f + g) (c + x) = (f + g) x) →
  (∀ x : ℝ, f x = f 0) ∨ (∀ x : ℝ, g x = g 0) :=
sorry

end periodic_sum_constant_l1233_123353


namespace find_a_l1233_123325

noncomputable def f (x : ℝ) : ℝ := 3 * ((x - 1) / 2) - 2

theorem find_a (x a : ℝ) (hx : f a = 4) (ha : a = 2 * x + 1) : a = 5 :=
by
  sorry

end find_a_l1233_123325


namespace twelve_pow_six_mod_nine_eq_zero_l1233_123328

theorem twelve_pow_six_mod_nine_eq_zero : (∃ n : ℕ, 0 ≤ n ∧ n < 9 ∧ 12^6 ≡ n [MOD 9]) → 12^6 ≡ 0 [MOD 9] :=
by
  sorry

end twelve_pow_six_mod_nine_eq_zero_l1233_123328


namespace math_problem_l1233_123333

noncomputable def f (x a : ℝ) : ℝ := -4 * (Real.cos x) ^ 2 + 4 * Real.sqrt 3 * a * (Real.sin x) * (Real.cos x) + 2

theorem math_problem (a : ℝ) :
  (∃ a, ∀ x, f x a = f (π/6 - x) a) →    -- Symmetry condition
  (a = 1 ∧
  ∀ k : ℤ, ∀ x, (x ∈ Set.Icc (π/3 + k * π) (5 * π / 6 + k * π) → 
    x ∈ Set.Icc (π/3 + k * π) (5 * π / 6 + k * π)) ∧  -- Decreasing intervals
  (∀ x, 2 * x - π / 6 ∈ Set.Icc (-2 * π / 3) (π / 6) → 
    f x a ∈ Set.Icc (-4 : ℝ) 2)) := -- Range on given interval
sorry

end math_problem_l1233_123333


namespace sneakers_sold_l1233_123393

theorem sneakers_sold (total_shoes sandals boots : ℕ) (h1 : total_shoes = 17) (h2 : sandals = 4) (h3 : boots = 11) :
  total_shoes - (sandals + boots) = 2 :=
by
  -- proof steps will be included here
  sorry

end sneakers_sold_l1233_123393


namespace no_real_roots_equationD_l1233_123341

def discriminant (a b c : ℕ) : ℤ := b^2 - 4 * a * c

def equationA := (1, -2, -4)
def equationB := (1, -4, 4)
def equationC := (1, -2, -5)
def equationD := (1, 3, 5)

theorem no_real_roots_equationD :
  discriminant (1 : ℕ) 3 5 < 0 :=
by
  show discriminant 1 3 5 < 0
  sorry

end no_real_roots_equationD_l1233_123341


namespace algebraic_expression_value_l1233_123334

theorem algebraic_expression_value (x : ℝ) (h : x = 4 * Real.sin (Real.pi / 4) - 2) :
  (1 / (x - 1) / (x + 2) / (x ^ 2 - 2 * x + 1) - x / (x + 2)) = - (Real.sqrt 2 / 4) :=
by
  sorry

end algebraic_expression_value_l1233_123334


namespace inequality_solution_1_inequality_solution_2_l1233_123388

-- Definition for part 1
theorem inequality_solution_1 (x : ℝ) : x^2 + 3*x - 4 > 0 ↔ x > 1 ∨ x < -4 :=
sorry

-- Definition for part 2
theorem inequality_solution_2 (x : ℝ) : (1 - x) / (x - 5) ≥ 1 ↔ 3 ≤ x ∧ x < 5 :=
sorry

end inequality_solution_1_inequality_solution_2_l1233_123388


namespace youngest_is_dan_l1233_123370

notation "alice" => 21
notation "bob" => 18
notation "clare" => 22
notation "dan" => 16
notation "eve" => 28

theorem youngest_is_dan :
  let a := alice
  let b := bob
  let c := clare
  let d := dan
  let e := eve
  a + b = 39 ∧
  b + c = 40 ∧
  c + d = 38 ∧
  d + e = 44 ∧
  a + b + c + d + e = 105 →
  min (min (min (min a b) c) d) e = d :=
by {
  sorry
}

end youngest_is_dan_l1233_123370


namespace Kyle_is_25_l1233_123344

-- Definitions based on the conditions
def Tyson_age : Nat := 20
def Frederick_age : Nat := 2 * Tyson_age
def Julian_age : Nat := Frederick_age - 20
def Kyle_age : Nat := Julian_age + 5

-- The theorem to prove
theorem Kyle_is_25 : Kyle_age = 25 := by
  sorry

end Kyle_is_25_l1233_123344


namespace sub_decimal_proof_l1233_123365

theorem sub_decimal_proof : 2.5 - 0.32 = 2.18 :=
  by sorry

end sub_decimal_proof_l1233_123365


namespace find_y_l1233_123346

variable (h : ℕ) -- integral number of hours

-- Distance between A and B
def distance_AB : ℕ := 60

-- Speed and distance walked by woman starting at A
def speed_A : ℕ := 3
def distance_A (h : ℕ) : ℕ := speed_A * h

-- Speed and distance walked by woman starting at B
def speed_B_1st_hour : ℕ := 2
def distance_B (h : ℕ) : ℕ := (h * (h + 3)) / 2

-- Meeting point equation
def meeting_point_eqn (h : ℕ) : Prop := (distance_A h) + (distance_B h) = distance_AB

-- Requirement: y miles nearer to A whereas y = distance_AB - 2 * distance_B (since B meets closer to A by y miles)
def y_nearer_A (h : ℕ) : ℕ := distance_AB - 2 * (distance_A h)

-- Prove y = 6 for the specific value of h
theorem find_y : ∃ (h : ℕ), meeting_point_eqn h ∧ y_nearer_A h = 6 := by
  sorry

end find_y_l1233_123346


namespace max_servings_l1233_123392

-- Definitions based on the conditions
def servings_recipe := 3
def bananas_per_serving := 2 / servings_recipe
def strawberries_per_serving := 1 / servings_recipe
def yogurt_per_serving := 2 / servings_recipe

def emily_bananas := 4
def emily_strawberries := 3
def emily_yogurt := 6

-- Prove that Emily can make at most 6 servings while keeping the proportions the same
theorem max_servings :
  min (emily_bananas / bananas_per_serving) 
      (min (emily_strawberries / strawberries_per_serving) 
           (emily_yogurt / yogurt_per_serving)) = 6 := sorry

end max_servings_l1233_123392


namespace two_digit_number_formed_l1233_123368

theorem two_digit_number_formed (A B C D E F : ℕ) 
  (A_C_D_const : A + C + D = constant)
  (A_B_const : A + B = constant)
  (B_D_F_const : B + D + F = constant)
  (E_F_const : E + F = constant)
  (E_B_C_const : E + B + C = constant)
  (B_eq_C_D : B = C + D)
  (B_D_eq_E : B + D = E)
  (E_C_eq_A : E + C = A) 
  (hA : A = 6) 
  (hB : B = 3)
  : 10 * A + B = 63 :=
by sorry

end two_digit_number_formed_l1233_123368


namespace first_month_sale_l1233_123371

theorem first_month_sale 
(sale_2 sale_3 sale_4 sale_5 sale_6 : ℕ)
(avg_sale : ℕ) 
(h_avg: avg_sale = 6500)
(h_sale2: sale_2 = 6927)
(h_sale3: sale_3 = 6855)
(h_sale4: sale_4 = 7230)
(h_sale5: sale_5 = 6562)
(h_sale6: sale_6 = 4791)
: sale_1 = 6635 := by
  sorry

end first_month_sale_l1233_123371


namespace total_bales_stored_l1233_123306

theorem total_bales_stored 
  (initial_bales : ℕ := 540) 
  (new_bales : ℕ := 2) : 
  initial_bales + new_bales = 542 :=
by
  sorry

end total_bales_stored_l1233_123306


namespace James_total_water_capacity_l1233_123378

theorem James_total_water_capacity : 
  let cask_capacity := 20 -- capacity of a cask in gallons
  let barrel_capacity := 2 * cask_capacity + 3 -- capacity of a barrel in gallons
  let total_capacity := 4 * barrel_capacity + cask_capacity -- total water storage capacity
  total_capacity = 192 := by
    let cask_capacity := 20
    let barrel_capacity := 2 * cask_capacity + 3
    let total_capacity := 4 * barrel_capacity + cask_capacity
    have h : total_capacity = 192 := by sorry
    exact h

end James_total_water_capacity_l1233_123378


namespace statement_bug_travel_direction_l1233_123314

/-
  Theorem statement: On a plane with a grid formed by regular hexagons of side length 1,
  if a bug traveled from node A to node B along the shortest path of 100 units,
  then the bug traveled exactly 50 units in one direction.
-/
theorem bug_travel_direction (side_length : ℝ) (total_distance : ℝ) 
  (hexagonal_grid : Π (x y : ℝ), Prop) (A B : ℝ × ℝ) 
  (shortest_path : ℝ) :
  side_length = 1 ∧ shortest_path = 100 →
  ∃ (directional_travel : ℝ), directional_travel = 50 :=
by
  sorry

end statement_bug_travel_direction_l1233_123314


namespace union_of_A_and_B_l1233_123397

def A := {x : ℝ | ∃ y : ℝ, y = Real.log (x - 3)}
def B := {y : ℝ | ∃ (x : ℝ), y = Real.exp x}

theorem union_of_A_and_B : A ∪ B = {x : ℝ | x > 0} := by
sorry

end union_of_A_and_B_l1233_123397


namespace ten_numbers_exists_l1233_123387

theorem ten_numbers_exists :
  ∃ (a : Fin 10 → ℕ), 
    (∀ i j : Fin 10, i ≠ j → ¬ (a i ∣ a j))
    ∧ (∀ i j : Fin 10, i ≠ j → a i ^ 2 ∣ a j * a j) :=
sorry

end ten_numbers_exists_l1233_123387


namespace problem_inequality_problem_equality_condition_l1233_123363

theorem problem_inequality (a b c : ℕ) (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) :
  (a^3 + b^3 + c^3) / 3 ≥ a * b * c + a + b + c :=
sorry

theorem problem_equality_condition (a b c : ℕ) :
  (a^3 + b^3 + c^3) / 3 = a * b * c + a + b + c ↔ a + 1 = b ∧ b + 1 = c :=
sorry

end problem_inequality_problem_equality_condition_l1233_123363


namespace parabola_behavior_l1233_123336

theorem parabola_behavior (x : ℝ) (h : x < 0) : ∃ y, y = 2*x^2 - 1 ∧ ∀ x1 x2 : ℝ, x1 < x2 ∧ x1 < 0 ∧ x2 < 0 → (2*x1^2 - 1) > (2*x2^2 - 1) :=
by
  sorry

end parabola_behavior_l1233_123336


namespace carrots_total_l1233_123395
-- import the necessary library

-- define the conditions as given
def sandy_carrots : Nat := 6
def sam_carrots : Nat := 3

-- state the problem as a theorem to be proven
theorem carrots_total : sandy_carrots + sam_carrots = 9 := by
  sorry

end carrots_total_l1233_123395


namespace smallest_number_of_people_l1233_123302

theorem smallest_number_of_people (N : ℕ) :
  (∃ (N : ℕ), ∀ seats : ℕ, seats = 80 → N ≤ 80 → ∀ n : ℕ, n > N → (∃ m : ℕ, (m < N) ∧ ((seats + m) % 80 < seats))) → N = 20 :=
by
  sorry

end smallest_number_of_people_l1233_123302


namespace exists_g_l1233_123380

variable {R : Type} [Field R]

-- Define the function f with the given condition
def f (x y : R) : R := sorry

-- The main theorem to prove the existence of g
theorem exists_g (f_condition: ∀ x y z : R, f x y + f y z + f z x = 0) : ∃ g : R → R, ∀ x y : R, f x y = g x - g y := 
by 
  sorry

end exists_g_l1233_123380


namespace remainder_14_plus_x_mod_31_l1233_123304

theorem remainder_14_plus_x_mod_31 (x : ℕ) (hx : 7 * x ≡ 1 [MOD 31]) : (14 + x) % 31 = 23 := 
sorry

end remainder_14_plus_x_mod_31_l1233_123304


namespace train_cross_platform_time_l1233_123330

def train_length : ℝ := 300
def platform_length : ℝ := 550
def signal_pole_time : ℝ := 18

theorem train_cross_platform_time :
  let speed : ℝ := train_length / signal_pole_time
  let total_distance : ℝ := train_length + platform_length
  let crossing_time : ℝ := total_distance / speed
  crossing_time = 51 :=
by
  sorry

end train_cross_platform_time_l1233_123330


namespace arithmetic_sequence_a2_value_l1233_123300

theorem arithmetic_sequence_a2_value 
  (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h1 : ∀ n, a (n + 1) = a n + 3)
  (h2 : S n = n * (a 1 + a n) / 2)
  (hS13 : S 13 = 156) :
  a 2 = -3 := 
    sorry

end arithmetic_sequence_a2_value_l1233_123300


namespace smallest_num_is_1113805958_l1233_123331

def smallest_num (n : ℕ) : Prop :=
  (n + 5) % 19 = 0 ∧ (n + 5) % 73 = 0 ∧ (n + 5) % 101 = 0 ∧ (n + 5) % 89 = 0

theorem smallest_num_is_1113805958 : ∃ n, smallest_num n ∧ n = 1113805958 :=
by
  use 1113805958
  unfold smallest_num
  simp
  sorry

end smallest_num_is_1113805958_l1233_123331


namespace total_renovation_cost_eq_l1233_123310

-- Define the conditions
def hourly_rate_1 := 15
def hourly_rate_2 := 20
def hourly_rate_3 := 18
def hourly_rate_4 := 22
def hours_per_day := 8
def days := 10
def meal_cost_per_professional_per_day := 10
def material_cost := 2500
def plumbing_issue_cost := 750
def electrical_issue_cost := 500
def faulty_appliance_cost := 400

-- Define the calculated values based on the conditions
def daily_labor_cost_condition := 
  hourly_rate_1 * hours_per_day + 
  hourly_rate_2 * hours_per_day + 
  hourly_rate_3 * hours_per_day + 
  hourly_rate_4 * hours_per_day
def total_labor_cost := daily_labor_cost_condition * days

def daily_meal_cost := meal_cost_per_professional_per_day * 4
def total_meal_cost := daily_meal_cost * days

def unexpected_repair_costs := plumbing_issue_cost + electrical_issue_cost + faulty_appliance_cost

def total_cost := total_labor_cost + total_meal_cost + material_cost + unexpected_repair_costs

-- The theorem to prove that the total cost of the renovation is $10,550
theorem total_renovation_cost_eq : total_cost = 10550 := by
  sorry

end total_renovation_cost_eq_l1233_123310


namespace sum_max_min_value_f_l1233_123305

noncomputable def f (x : ℝ) : ℝ := ((x + 1) ^ 2 + x) / (x ^ 2 + 1)

theorem sum_max_min_value_f : 
  let M := (⨆ x : ℝ, f x)
  let m := (⨅ x : ℝ, f x)
  M + m = 2 :=
by
-- Proof to be filled in
  sorry

end sum_max_min_value_f_l1233_123305


namespace production_days_l1233_123347

theorem production_days (n : ℕ) (P : ℕ)
  (h1 : P = 40 * n)
  (h2 : (P + 90) / (n + 1) = 45) :
  n = 9 :=
by
  sorry

end production_days_l1233_123347


namespace beneficial_for_kati_l1233_123379

variables (n : ℕ) (x y : ℝ)

theorem beneficial_for_kati (hn : n > 0) (hx : x ≥ 0) (hy : y ≥ 0) :
  (x + y) / (n + 2) > (x + y / 2) / (n + 1) :=
sorry

end beneficial_for_kati_l1233_123379


namespace triangle_area_ratio_l1233_123332

theorem triangle_area_ratio
  (a b c : ℕ) (S_triangle : ℕ) -- assuming S_triangle represents the area of the original triangle
  (S_bisected_triangle : ℕ) -- assuming S_bisected_triangle represents the area of the bisected triangle
  (is_angle_bisector : ∀ x y z : ℕ, ∃ k, k = (2 * a * b * c * x) / ((a + b) * (a + c) * (b + c))) :
  S_bisected_triangle = (2 * a * b * c) / ((a + b) * (a + c) * (b + c)) * S_triangle :=
sorry

end triangle_area_ratio_l1233_123332
