import Mathlib

namespace share_of_each_person_l2102_210289

theorem share_of_each_person (total_length : ℕ) (h1 : total_length = 12) (h2 : total_length % 2 = 0)
  : total_length / 2 = 6 :=
by
  sorry

end share_of_each_person_l2102_210289


namespace train_crosses_post_in_25_2_seconds_l2102_210280

noncomputable def train_crossing_time (speed_kmph : ℝ) (length_m : ℝ) : ℝ :=
  length_m / (speed_kmph * 1000 / 3600)

theorem train_crosses_post_in_25_2_seconds :
  train_crossing_time 40 280.0224 = 25.2 :=
by 
  sorry

end train_crosses_post_in_25_2_seconds_l2102_210280


namespace value_of_expression_l2102_210222

theorem value_of_expression (x y : ℝ) (h₀ : x = Real.sqrt 2 + 1) (h₁ : y = Real.sqrt 2 - 1) : 
  (x + y) * (x - y) = 4 * Real.sqrt 2 :=
by
  sorry

end value_of_expression_l2102_210222


namespace reduce_fraction_l2102_210225

-- Defining a structure for a fraction
structure Fraction where
  num : ℕ
  denom : ℕ
  deriving Repr

-- The original fraction
def originalFraction : Fraction :=
  { num := 368, denom := 598 }

-- The reduced fraction
def reducedFraction : Fraction :=
  { num := 184, denom := 299 }

-- The statement of our theorem
theorem reduce_fraction :
  ∃ (d : ℕ), d > 0 ∧ (originalFraction.num / d = reducedFraction.num) ∧ (originalFraction.denom / d = reducedFraction.denom) := by
  sorry

end reduce_fraction_l2102_210225


namespace smallest_pos_int_y_satisfies_congruence_l2102_210257

theorem smallest_pos_int_y_satisfies_congruence :
  ∃ y : ℕ, (y > 0) ∧ (26 * y + 8) % 16 = 4 ∧ ∀ z : ℕ, (z > 0) ∧ (26 * z + 8) % 16 = 4 → y ≤ z :=
sorry

end smallest_pos_int_y_satisfies_congruence_l2102_210257


namespace solution_set_of_inequality_l2102_210207

theorem solution_set_of_inequality :
  { x : ℝ | (x - 2) / (x + 3) ≥ 0 } = { x : ℝ | x < -3 } ∪ { x : ℝ | x ≥ 2 } := 
sorry

end solution_set_of_inequality_l2102_210207


namespace print_time_l2102_210281

-- Conditions
def printer_pages_per_minute : ℕ := 25
def total_pages : ℕ := 350

-- Theorem
theorem print_time :
  (total_pages / printer_pages_per_minute : ℕ) = 14 :=
by sorry

end print_time_l2102_210281


namespace rationalize_sqrt_l2102_210251

theorem rationalize_sqrt (h : Real.sqrt 35 ≠ 0) : 35 / Real.sqrt 35 = Real.sqrt 35 := 
by 
sorry

end rationalize_sqrt_l2102_210251


namespace arithmetic_sequence_problem_l2102_210205

noncomputable def a_n (n : ℕ) : ℚ := 1 + (n - 1) / 2

noncomputable def S_n (n : ℕ) : ℚ := n * (n + 3) / 4

theorem arithmetic_sequence_problem :
  -- Given
  (∀ n, ∃ d, a_n n = a_1 + (n - 1) * d) →
  (a_n 7 = 4) →
  (a_n 19 = 2 * a_n 9) →
  -- Prove
  (∀ n, a_n n = (n + 1) / 2) ∧ (∀ n, S_n n = n * (n + 3) / 4) :=
by
  sorry

end arithmetic_sequence_problem_l2102_210205


namespace first_cat_blue_eyed_kittens_l2102_210291

variable (B : ℕ)
variable (C1 : 35 * (B + 17) = 100 * (B + 4))

theorem first_cat_blue_eyed_kittens : B = 3 :=
by
  -- proof
  sorry

end first_cat_blue_eyed_kittens_l2102_210291


namespace find_fraction_l2102_210243

-- Define the initial amount, the amount spent on pads, and the remaining amount
def initial_amount := 150
def spent_on_pads := 50
def remaining := 25

-- Define the fraction she spent on hockey skates
def fraction_spent_on_skates (f : ℚ) : Prop :=
  let spent_on_skates := initial_amount - remaining - spent_on_pads
  (spent_on_skates / initial_amount) = f

theorem find_fraction : fraction_spent_on_skates (1 / 2) :=
by
  -- Proof steps go here
  sorry

end find_fraction_l2102_210243


namespace find_sum_of_angles_l2102_210295

-- Given conditions
def angleP := 34
def angleQ := 76
def angleR := 28

-- Proposition to prove
theorem find_sum_of_angles (x z : ℝ) (h1 : x + z = 138) : x + z = 138 :=
by
  have angleP := 34
  have angleQ := 76
  have angleR := 28
  exact h1

end find_sum_of_angles_l2102_210295


namespace eq1_eq2_eq3_eq4_l2102_210233

theorem eq1 (x : ℚ) : 3 * x^2 - 32 * x - 48 = 0 ↔ (x = 12 ∨ x = -4/3) := sorry

theorem eq2 (x : ℚ) : 4 * x^2 + x - 3 = 0 ↔ (x = 3/4 ∨ x = -1) := sorry

theorem eq3 (x : ℚ) : (3 * x + 1)^2 - 4 = 0 ↔ (x = 1/3 ∨ x = -1) := sorry

theorem eq4 (x : ℚ) : 9 * (x - 2)^2 = 4 * (x + 1)^2 ↔ (x = 8 ∨ x = 4/5) := sorry

end eq1_eq2_eq3_eq4_l2102_210233


namespace ELMO_value_l2102_210250

def digits := {n : ℕ // n < 10}

variables (L E T M O : digits)

-- Conditions
axiom h1 : L.val ≠ 0
axiom h2 : O.val = 0
axiom h3 : (1000 * L.val + 100 * E.val + 10 * E.val + T.val) + (100 * L.val + 10 * M.val + T.val) = 1000 * T.val + L.val

-- Conclusion
theorem ELMO_value : E.val * 1000 + L.val * 100 + M.val * 10 + O.val = 1880 :=
sorry

end ELMO_value_l2102_210250


namespace total_payment_correct_l2102_210299

def rate_per_kg_grapes := 68
def quantity_grapes := 7
def rate_per_kg_mangoes := 48
def quantity_mangoes := 9

def cost_grapes := rate_per_kg_grapes * quantity_grapes
def cost_mangoes := rate_per_kg_mangoes * quantity_mangoes

def total_amount_paid := cost_grapes + cost_mangoes

theorem total_payment_correct :
  total_amount_paid = 908 := by
  sorry

end total_payment_correct_l2102_210299


namespace eval_composed_function_l2102_210284

noncomputable def f (x : ℝ) := 3 * x^2 - 4
noncomputable def k (x : ℝ) := 5 * x^3 + 2

theorem eval_composed_function :
  f (k 2) = 5288 := 
by
  sorry

end eval_composed_function_l2102_210284


namespace sin_2theta_in_third_quadrant_l2102_210209

open Real

variables (θ : ℝ)

/-- \theta is an angle in the third quadrant.
Given that \(\sin^{4}\theta + \cos^{4}\theta = \frac{5}{9}\), 
prove that \(\sin 2\theta = \frac{2\sqrt{2}}{3}\). --/
theorem sin_2theta_in_third_quadrant (h_theta_third_quadrant : π < θ ∧ θ < 3 * π / 2)
(h_cond : sin θ ^ 4 + cos θ ^ 4 = 5 / 9) : sin (2 * θ) = 2 * sqrt 2 / 3 :=
sorry

end sin_2theta_in_third_quadrant_l2102_210209


namespace notebook_cost_l2102_210261

theorem notebook_cost :
  let mean_expenditure := 500
  let daily_expenditures := [450, 600, 400, 500, 550, 300]
  let cost_earphone := 620
  let cost_pen := 30
  let total_days := 7
  let total_expenditure := mean_expenditure * total_days
  let sum_other_days := daily_expenditures.sum
  let expenditure_friday := total_expenditure - sum_other_days
  let cost_notebook := expenditure_friday - (cost_earphone + cost_pen)
  cost_notebook = 50 := by
  sorry

end notebook_cost_l2102_210261


namespace fill_tank_time_l2102_210226

-- Define the rates of filling and draining
def rateA : ℕ := 200 -- Pipe A fills at 200 liters per minute
def rateB : ℕ := 50  -- Pipe B fills at 50 liters per minute
def rateC : ℕ := 25  -- Pipe C drains at 25 liters per minute

-- Define the times each pipe is open
def timeA : ℕ := 1   -- Pipe A is open for 1 minute
def timeB : ℕ := 2   -- Pipe B is open for 2 minutes
def timeC : ℕ := 2   -- Pipe C is open for 2 minutes

-- Define the capacity of the tank
def tankCapacity : ℕ := 1000

-- Prove the total time to fill the tank is 20 minutes
theorem fill_tank_time : 
  (tankCapacity * ((timeA * rateA + timeB * rateB) - (timeC * rateC)) * 5) = 20 :=
sorry

end fill_tank_time_l2102_210226


namespace sequence_values_l2102_210208

theorem sequence_values (x y z : ℚ) :
  (∀ n : ℕ, x = 1 ∧ y = 9 / 8 ∧ z = 5 / 4) :=
by
  sorry

end sequence_values_l2102_210208


namespace original_wattage_l2102_210283

theorem original_wattage (W : ℝ) (h1 : 143 = 1.30 * W) : W = 110 := 
by
  sorry

end original_wattage_l2102_210283


namespace felipe_building_time_l2102_210285

theorem felipe_building_time
  (F E : ℕ)
  (combined_time_without_breaks : ℕ)
  (felipe_time_fraction : F = E / 2)
  (combined_time_condition : F + E = 90)
  (felipe_break : ℕ)
  (emilio_break : ℕ)
  (felipe_break_is_6_months : felipe_break = 6)
  (emilio_break_is_double_felipe : emilio_break = 2 * felipe_break) :
  F + felipe_break = 36 := by
  sorry

end felipe_building_time_l2102_210285


namespace total_pens_l2102_210278

theorem total_pens (r : ℕ) (h1 : r > 10) (h2 : 357 % r = 0) (h3 : 441 % r = 0) :
  (357 / r) + (441 / r) = 38 :=
sorry

end total_pens_l2102_210278


namespace coats_collected_elem_schools_correct_l2102_210287

-- Conditions
def total_coats_collected : ℕ := 9437
def coats_collected_high_schools : ℕ := 6922

-- Definition to find coats collected from elementary schools
def coats_collected_elementary_schools : ℕ := total_coats_collected - coats_collected_high_schools

-- Theorem statement
theorem coats_collected_elem_schools_correct : 
  coats_collected_elementary_schools = 2515 := sorry

end coats_collected_elem_schools_correct_l2102_210287


namespace total_interest_received_l2102_210255

def principal_B := 5000
def principal_C := 3000
def rate := 9
def time_B := 2
def time_C := 4
def simple_interest (P : ℕ) (R : ℕ) (T : ℕ) : ℕ := P * R * T / 100

theorem total_interest_received :
  let SI_B := simple_interest principal_B rate time_B
  let SI_C := simple_interest principal_C rate time_C
  SI_B + SI_C = 1980 := 
by
  sorry

end total_interest_received_l2102_210255


namespace find_pairs_of_nonneg_ints_l2102_210272

theorem find_pairs_of_nonneg_ints (m n : ℕ) :
  m^2 + 2 * 3^n = m * (2^(n + 1) - 1) ↔ (m, n) = (9, 3) ∨ (m, n) = (6, 3) ∨ (m, n) = (9, 5) ∨ (m, n) = (54, 5) :=
by
  sorry

end find_pairs_of_nonneg_ints_l2102_210272


namespace pure_imaginary_real_part_zero_l2102_210241

-- Define the condition that the complex number a + i is a pure imaginary number.
def isPureImaginary (z : ℂ) : Prop :=
  ∃ b : ℝ, z = Complex.I * b

-- Define the complex number a + i.
def z (a : ℝ) : ℂ := a + Complex.I

-- The theorem states that if z is pure imaginary, then a = 0.
theorem pure_imaginary_real_part_zero (a : ℝ) (h : isPureImaginary (z a)) : a = 0 :=
by
  sorry

end pure_imaginary_real_part_zero_l2102_210241


namespace quadratic_ineq_solution_l2102_210268

theorem quadratic_ineq_solution (a b : ℝ) 
  (h_solution_set : ∀ x, (ax^2 + bx - 1 > 0) ↔ (1 / 3 < x ∧ x < 1))
  (h_roots : (a / 3 + b = -1 / a) ∧ (a / 3 = -1 / a)) 
  (h_a_neg : a < 0) : a + b = 1 := 
sorry 

end quadratic_ineq_solution_l2102_210268


namespace range_of_m_for_nonempty_solution_set_l2102_210244

theorem range_of_m_for_nonempty_solution_set :
  {m : ℝ | ∃ x : ℝ, m * x^2 - m * x + 1 < 0} = {m : ℝ | m < 0} ∪ {m : ℝ | m > 4} :=
by sorry

end range_of_m_for_nonempty_solution_set_l2102_210244


namespace village_population_decrease_rate_l2102_210248

theorem village_population_decrease_rate :
  ∃ (R : ℝ), 15 * R = 18000 :=
by
  sorry

end village_population_decrease_rate_l2102_210248


namespace loss_percentage_grinder_l2102_210273

-- Conditions
def CP_grinder : ℝ := 15000
def CP_mobile : ℝ := 8000
def profit_mobile : ℝ := 0.10
def total_profit : ℝ := 200

-- Theorem to prove the loss percentage on the grinder
theorem loss_percentage_grinder : 
  ( (CP_grinder - (23200 - (CP_mobile * (1 + profit_mobile)))) / CP_grinder ) * 100 = 4 :=
by
  sorry

end loss_percentage_grinder_l2102_210273


namespace sleeves_add_correct_weight_l2102_210200

variable (R W_r W_s S : ℝ)

-- Conditions
def raw_squat : Prop := R = 600
def wraps_add_25_percent : Prop := W_r = R + 0.25 * R
def wraps_vs_sleeves_difference : Prop := W_r = W_s + 120

-- To Prove
theorem sleeves_add_correct_weight (h1 : raw_squat R) (h2 : wraps_add_25_percent R W_r) (h3 : wraps_vs_sleeves_difference W_r W_s) : S = 30 :=
by
  sorry

end sleeves_add_correct_weight_l2102_210200


namespace find_x_l2102_210265

def diamond (x y : ℤ) : ℤ := 3 * x - y^2

theorem find_x (x : ℤ) (h : diamond x 7 = 20) : x = 23 :=
sorry

end find_x_l2102_210265


namespace max_a_for_no_lattice_point_l2102_210239

theorem max_a_for_no_lattice_point (a : ℝ) (hm : ∀ m : ℝ, 1 / 2 < m ∧ m < a → ¬ ∃ x y : ℤ, 0 < x ∧ x ≤ 200 ∧ y = m * x + 3) : 
  a = 101 / 201 :=
sorry

end max_a_for_no_lattice_point_l2102_210239


namespace find_somu_age_l2102_210204

noncomputable def somu_age (S F : ℕ) : Prop :=
  S = (1/3 : ℝ) * F ∧ S - 6 = (1/5 : ℝ) * (F - 6)

theorem find_somu_age {S F : ℕ} (h : somu_age S F) : S = 12 :=
by sorry

end find_somu_age_l2102_210204


namespace find_x_plus_y_l2102_210235

theorem find_x_plus_y (x y : Real) (h1 : x + Real.sin y = 2010) (h2 : x + 2010 * Real.cos y = 2009) (h3 : 0 ≤ y ∧ y ≤ Real.pi / 2) : x + y = 2009 + Real.pi / 2 :=
by
  sorry

end find_x_plus_y_l2102_210235


namespace kati_age_l2102_210237

/-- Define the age of Kati using the given conditions -/
theorem kati_age (kati_age : ℕ) (brother_age kati_birthdays : ℕ) 
  (h1 : kati_age = kati_birthdays) 
  (h2 : kati_age + brother_age = 111) 
  (h3 : kati_birthdays = kati_age) : 
  kati_age = 18 :=
by
  sorry

end kati_age_l2102_210237


namespace n_sum_of_two_squares_l2102_210253

theorem n_sum_of_two_squares (n : ℤ) (m : ℤ) (hn_gt_2 : n > 2) (hn2_eq_diff_cubes : n^2 = (m+1)^3 - m^3) : 
  ∃ a b : ℤ, n = a^2 + b^2 :=
sorry

end n_sum_of_two_squares_l2102_210253


namespace triangle_inequality_area_equality_condition_l2102_210220

theorem triangle_inequality_area (a b c S : ℝ) (h_area : S = (a * b * Real.sin (Real.arccos ((a*a + b*b - c*c) / (2*a*b)))) / 2) :
  a^2 + b^2 + c^2 ≥ 4 * Real.sqrt 3 * S :=
by
  sorry

theorem equality_condition (a b c : ℝ) (h_eq : a = b ∧ b = c) : 
  a^2 + b^2 + c^2 = 4 * Real.sqrt 3 * (a^2 * (Real.sqrt 3 / 4)) :=
by
  sorry

end triangle_inequality_area_equality_condition_l2102_210220


namespace eccentricity_of_given_hyperbola_l2102_210214

noncomputable def hyperbola_eccentricity (a b : ℝ) (h : b = 2 * a) : ℝ :=
  Real.sqrt (1 + (b * b) / (a * a))

theorem eccentricity_of_given_hyperbola (a b : ℝ) 
  (h_hyperbola : b = 2 * a)
  (h_asymptote : ∃ k, k = 2 ∧ ∀ x, y = k * x → ((y * a) = (b * x))) :
  hyperbola_eccentricity a b h_hyperbola = Real.sqrt 5 :=
by
  sorry

end eccentricity_of_given_hyperbola_l2102_210214


namespace maria_ends_up_with_22_towels_l2102_210247

-- Define the number of green towels Maria bought
def green_towels : Nat := 35

-- Define the number of white towels Maria bought
def white_towels : Nat := 21

-- Define the number of towels Maria gave to her mother
def given_towels : Nat := 34

-- Total towels Maria initially bought
def total_towels := green_towels + white_towels

-- Towels Maria ended up with
def remaining_towels := total_towels - given_towels

theorem maria_ends_up_with_22_towels :
  remaining_towels = 22 :=
by
  sorry

end maria_ends_up_with_22_towels_l2102_210247


namespace henrietta_paint_needed_l2102_210286

theorem henrietta_paint_needed :
  let living_room_area := 600
  let num_bedrooms := 3
  let bedroom_area := 400
  let paint_coverage_per_gallon := 600
  let total_area := living_room_area + (num_bedrooms * bedroom_area)
  total_area / paint_coverage_per_gallon = 3 :=
by
  -- Proof should be completed here.
  sorry

end henrietta_paint_needed_l2102_210286


namespace jason_average_messages_l2102_210258

theorem jason_average_messages : 
    let monday := 220
    let tuesday := monday / 2
    let wednesday := 50
    let thursday := 50
    let friday := 50
    let total_messages := monday + tuesday + wednesday + thursday + friday
    let average_messages := total_messages / 5
    average_messages = 96 :=
by
  let monday := 220
  let tuesday := monday / 2
  let wednesday := 50
  let thursday := 50
  let friday := 50
  let total_messages := monday + tuesday + wednesday + thursday + friday
  let average_messages := total_messages / 5
  have h : average_messages = 96 := sorry
  exact h

end jason_average_messages_l2102_210258


namespace max_min_value_l2102_210276

def f (x t : ℝ) : ℝ := x^2 - 2 * t * x + t

theorem max_min_value : 
  ∀ t : ℝ, (-1 ≤ t ∧ t ≤ 1) →
  (∀ x : ℝ, (-1 ≤ x ∧ x ≤ 1) → f x t ≥ -t^2 + t) →
  (∃ t : ℝ, (-1 ≤ t ∧ t ≤ 1) ∧ ∀ x : ℝ, (-1 ≤ x ∧ x ≤ 1) → f x t ≥ -t^2 + t ∧ -t^2 + t = 1/4) :=
sorry

end max_min_value_l2102_210276


namespace find_n_in_range_and_modulus_l2102_210240

theorem find_n_in_range_and_modulus :
  ∃ n : ℤ, 0 ≤ n ∧ n < 21 ∧ (-200) % 21 = n % 21 → n = 10 := by
  sorry

end find_n_in_range_and_modulus_l2102_210240


namespace arithmetic_series_sum_l2102_210224

theorem arithmetic_series_sum : 
  ∀ (a d a_n : ℤ), 
  a = -48 → d = 2 → a_n = 0 → 
  ∃ n S : ℤ, 
  a + (n - 1) * d = a_n ∧ 
  S = n * (a + a_n) / 2 ∧ 
  S = -600 :=
by
  intros a d a_n ha hd han
  have h₁ : a = -48 := ha
  have h₂ : d = 2 := hd
  have h₃ : a_n = 0 := han
  sorry

end arithmetic_series_sum_l2102_210224


namespace ratio_of_doctors_lawyers_engineers_l2102_210266

variables (d l e : ℕ)

-- Conditions
def average_age_per_group (d l e : ℕ) : Prop :=
  (40 * d + 55 * l + 35 * e) = 45 * (d + l + e)

-- Theorem
theorem ratio_of_doctors_lawyers_engineers
  (h : average_age_per_group d l e) :
  l = d + 2 * e :=
by sorry

end ratio_of_doctors_lawyers_engineers_l2102_210266


namespace correct_conditions_for_cubic_eq_single_root_l2102_210294

noncomputable def hasSingleRealRoot (a b : ℝ) : Prop :=
  let f := λ x : ℝ => x^3 - a * x + b
  let f' := λ x : ℝ => 3 * x^2 - a
  ∀ (x y : ℝ), f' x = 0 → f' y = 0 → x = y

theorem correct_conditions_for_cubic_eq_single_root :
  (hasSingleRealRoot 0 2) ∧ 
  (hasSingleRealRoot (-3) 2) ∧ 
  (hasSingleRealRoot 3 (-3)) :=
  by 
    sorry

end correct_conditions_for_cubic_eq_single_root_l2102_210294


namespace maria_initial_carrots_l2102_210230

theorem maria_initial_carrots (C : ℕ) (h : C - 11 + 15 = 52) : C = 48 :=
by
  sorry

end maria_initial_carrots_l2102_210230


namespace systematic_sampling_first_segment_l2102_210245

theorem systematic_sampling_first_segment:
  ∀ (total_students sample_size segment_size 
     drawn_16th drawn_first : ℕ),
  total_students = 160 →
  sample_size = 20 →
  segment_size = 8 →
  drawn_16th = 125 →
  drawn_16th = drawn_first + segment_size * (16 - 1) →
  drawn_first = 5 :=
by
  intros total_students sample_size segment_size drawn_16th drawn_first
         htots hsamp hseg hdrw16 heq
  sorry

end systematic_sampling_first_segment_l2102_210245


namespace area_bounded_region_l2102_210254

theorem area_bounded_region (x y : ℝ) (h : y^2 + 2*x*y + 30*|x| = 300) : 
  ∃ A, A = 900 := 
sorry

end area_bounded_region_l2102_210254


namespace quotient_of_division_l2102_210217

theorem quotient_of_division (Q : ℤ) (h1 : 172 = (17 * Q) + 2) : Q = 10 :=
sorry

end quotient_of_division_l2102_210217


namespace range_of_m_l2102_210293

variable {m x x1 x2 y1 y2 : ℝ}

noncomputable def linear_function (m x : ℝ) : ℝ := (m - 2) * x + (2 + m)

theorem range_of_m (h1 : x1 < x2) (h2 : y1 = linear_function m x1) (h3 : y2 = linear_function m x2) (h4 : y1 > y2) : m < 2 :=
by
  sorry

end range_of_m_l2102_210293


namespace negation_of_proposition_exists_negation_of_proposition_l2102_210296

theorem negation_of_proposition : 
  (∀ x : ℝ, 2^x - 2*x - 2 ≥ 0) ↔ ¬(∀ x : ℝ, 2^x - 2*x - 2 ≥ 0) :=
by
  sorry

theorem exists_negation_of_proposition : 
  (¬(∀ x : ℝ, 2^x - 2*x - 2 ≥ 0)) ↔ ∃ x : ℝ, 2^x - 2*x - 2 < 0 :=
by
  sorry

end negation_of_proposition_exists_negation_of_proposition_l2102_210296


namespace fluffy_striped_or_spotted_cats_l2102_210277

theorem fluffy_striped_or_spotted_cats (total_cats : ℕ) (striped_fraction : ℚ) (spotted_fraction : ℚ)
    (fluffy_striped_fraction : ℚ) (fluffy_spotted_fraction : ℚ) (striped_spotted_fraction : ℚ) :
    total_cats = 180 ∧ striped_fraction = 1/2 ∧ spotted_fraction = 1/3 ∧
    fluffy_striped_fraction = 1/8 ∧ fluffy_spotted_fraction = 3/7 →
    striped_spotted_fraction = 36 :=
by
    sorry

end fluffy_striped_or_spotted_cats_l2102_210277


namespace minimize_y_l2102_210218

variables (a b k : ℝ)

def y (x : ℝ) : ℝ := 3 * (x - a) ^ 2 + (x - b) ^ 2 + k * x

theorem minimize_y : ∃ x : ℝ, y a b k x = y a b k ( (6 * a + 2 * b - k) / 8 ) :=
  sorry

end minimize_y_l2102_210218


namespace small_branches_count_l2102_210263

theorem small_branches_count (x : ℕ) (h : x^2 + x + 1 = 91) : x = 9 := 
  sorry

end small_branches_count_l2102_210263


namespace solve_for_question_mark_l2102_210288

/-- Prove that the number that should replace "?" in the equation 
    300 * 2 + (12 + ?) * (1 / 8) = 602 is equal to 4. -/
theorem solve_for_question_mark : 
  ∃ (x : ℕ), 300 * 2 + (12 + x) * (1 / 8) = 602 ∧ x = 4 := 
by
  sorry

end solve_for_question_mark_l2102_210288


namespace probability_white_first_red_second_l2102_210256

theorem probability_white_first_red_second :
  let total_marbles := 10
  let white_marbles := 6
  let red_marbles := 4
  let prob_white_first := white_marbles / total_marbles
  let prob_red_second_given_white_first := red_marbles / (total_marbles - 1)
  let prob_combined := prob_white_first * prob_red_second_given_white_first
  prob_combined = 4 / 15 :=
by
  sorry

end probability_white_first_red_second_l2102_210256


namespace find_f2_l2102_210259

namespace ProofProblem

-- Define the polynomial function f
def f (x a b : ℤ) : ℤ := x^5 + a * x^3 + b * x - 8

-- Conditions given in the problem
axiom f_neg2 : ∃ a b : ℤ, f (-2) a b = 10

-- Define the theorem statement
theorem find_f2 : ∃ a b : ℤ, f 2 a b = -26 :=
by
  sorry

end ProofProblem

end find_f2_l2102_210259


namespace circle_equation_bisects_l2102_210229

-- Define the given conditions
def circle1_eq (x y : ℝ) : Prop := (x - 4)^2 + (y - 8)^2 = 1
def circle2_eq (x y : ℝ) : Prop := (x - 6)^2 + (y + 6)^2 = 9

-- Define the goal equation
def circleC_eq (x y : ℝ) : Prop := x^2 + y^2 = 81

-- The statement of the problem
theorem circle_equation_bisects (a r : ℝ) (h1 : ∀ x y, circle1_eq x y → circleC_eq x y) (h2 : ∀ x y, circle2_eq x y → circleC_eq x y):
  circleC_eq (a * r) 0 := sorry

end circle_equation_bisects_l2102_210229


namespace fraction_sum_ratio_l2102_210271

theorem fraction_sum_ratio :
  let A := (Finset.range 1002).sum (λ k => 1 / ((2 * k + 1) * (2 * k + 2)))
  let B := (Finset.range 1002).sum (λ k => 1 / ((1003 + k) * (2004 - k)))
  (A / B) = (3007 / 2) :=
by
  sorry

end fraction_sum_ratio_l2102_210271


namespace reach_one_from_any_non_zero_l2102_210232

-- Define the game rules as functions
def remove_units_digit (n : ℕ) : ℕ :=
  n / 10

def multiply_by_two (n : ℕ) : ℕ :=
  n * 2

-- Lemma: Prove that starting from 45, we can reach 1 using the game rules.
lemma reach_one_from_45 : ∃ f : ℕ → ℕ, f 45 = 1 := 
by {
  -- You can define the sequence explicitly or use the function definitions.
  sorry
}

-- Lemma: Prove that starting from 345, we can reach 1 using the game rules.
lemma reach_one_from_345 : ∃ f : ℕ → ℕ, f 345 = 1 := 
by {
  -- You can define the sequence explicitly or use the function definitions.
  sorry
}

-- Theorem: Prove that any non-zero natural number can be reduced to 1 using the game rules.
theorem reach_one_from_any_non_zero (n : ℕ) (h : n ≠ 0) : ∃ f : ℕ → ℕ, f n = 1 :=
by {
  sorry
}

end reach_one_from_any_non_zero_l2102_210232


namespace volume_of_one_slice_l2102_210262

theorem volume_of_one_slice
  (circumference : ℝ)
  (c : circumference = 18 * Real.pi):
  ∃ V, V = 162 * Real.pi :=
by sorry

end volume_of_one_slice_l2102_210262


namespace base_6_four_digit_odd_final_digit_l2102_210274

-- Definition of the conditions
def four_digit_number (n b : ℕ) : Prop :=
  b^3 ≤ n ∧ n < b^4

def odd_digit (n b : ℕ) : Prop :=
  (n % b) % 2 = 1

-- Problem statement
theorem base_6_four_digit_odd_final_digit :
  four_digit_number 350 6 ∧ odd_digit 350 6 := by
  sorry

end base_6_four_digit_odd_final_digit_l2102_210274


namespace find_y_l2102_210260

-- Define the atomic weights
def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.01
def atomic_weight_O : ℝ := 16.00

-- Define the molecular weight of the compound C6HyO7
def molecular_weight : ℝ := 192

-- Define the contribution of Carbon and Oxygen
def contribution_C : ℝ := 6 * atomic_weight_C
def contribution_O : ℝ := 7 * atomic_weight_O

-- The proof statement
theorem find_y (y : ℕ) :
  molecular_weight = contribution_C + y * atomic_weight_H + contribution_O → y = 8 :=
by
  sorry

end find_y_l2102_210260


namespace number_of_new_books_l2102_210227

-- Defining the given conditions
def adventure_books : ℕ := 24
def mystery_books : ℕ := 37
def used_books : ℕ := 18

-- Defining the total books and new books
def total_books : ℕ := adventure_books + mystery_books
def new_books : ℕ := total_books - used_books

-- Proving the number of new books
theorem number_of_new_books : new_books = 43 := by
  -- Here we need to show that the calculated number of new books equals 43
  sorry

end number_of_new_books_l2102_210227


namespace cistern_length_l2102_210275

theorem cistern_length
  (L W D A : ℝ)
  (hW : W = 4)
  (hD : D = 1.25)
  (hA : A = 49)
  (hWetSurface : A = L * W + 2 * L * D) :
  L = 7.54 := by
  sorry

end cistern_length_l2102_210275


namespace minimum_detectors_required_l2102_210282

/-- There is a cube with each face divided into 4 identical square cells, making a total of 24 cells.
Oleg wants to mark 8 cells with invisible ink such that no two marked cells share a side.
Rustem wants to place detectors in the cells so that all marked cells can be identified. -/
def minimum_detectors_to_identify_all_marked_cells (total_cells: ℕ) (marked_cells: ℕ) 
  (cells_per_face: ℕ) (faces: ℕ) : ℕ :=
  if total_cells = faces * cells_per_face ∧ marked_cells = 8 then 16 else 0

theorem minimum_detectors_required :
  minimum_detectors_to_identify_all_marked_cells 24 8 4 6 = 16 :=
by
  sorry

end minimum_detectors_required_l2102_210282


namespace sqrt_computation_l2102_210221

theorem sqrt_computation : 
  Real.sqrt ((35 * 34 * 33 * 32) + Nat.factorial 4) = 1114 := by
sorry

end sqrt_computation_l2102_210221


namespace geom_series_sum_l2102_210215

def a : ℚ := 1 / 3
def r : ℚ := 2 / 3
def n : ℕ := 9

def S_n (a r : ℚ) (n : ℕ) := a * (1 - r^n) / (1 - r)

theorem geom_series_sum :
  S_n a r n = 19171 / 19683 := by
    sorry

end geom_series_sum_l2102_210215


namespace tan_cos_sin_fraction_l2102_210201

theorem tan_cos_sin_fraction (α : ℝ) (h : Real.tan α = -3) : 
  (Real.cos α + 2 * Real.sin α) / (Real.cos α - 3 * Real.sin α) = -1 / 2 := 
by
  sorry

end tan_cos_sin_fraction_l2102_210201


namespace opposite_numbers_abs_eq_l2102_210202

theorem opposite_numbers_abs_eq (a : ℚ) : abs a = abs (-a) :=
by
  sorry

end opposite_numbers_abs_eq_l2102_210202


namespace joan_and_karl_sofas_l2102_210297

variable (J K : ℝ)

theorem joan_and_karl_sofas (hJ : J = 230) (hSum : J + K = 600) :
  2 * J - K = 90 :=
by
  sorry

end joan_and_karl_sofas_l2102_210297


namespace interior_and_exterior_angles_of_regular_dodecagon_l2102_210269

-- Definition of a regular dodecagon
def regular_dodecagon_sides : ℕ := 12

-- The sum of the interior angles of a regular polygon
def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

-- Measure of one interior angle of a regular polygon
def one_interior_angle (n : ℕ) : ℕ := sum_of_interior_angles n / n

-- Measure of one exterior angle of a regular polygon (180 degrees supplementary to interior angle)
def one_exterior_angle (n : ℕ) : ℕ := 180 - one_interior_angle n

-- The theorem to prove
theorem interior_and_exterior_angles_of_regular_dodecagon :
  one_interior_angle regular_dodecagon_sides = 150 ∧ one_exterior_angle regular_dodecagon_sides = 30 :=
by
  sorry

end interior_and_exterior_angles_of_regular_dodecagon_l2102_210269


namespace infinite_series_sum_l2102_210252

noncomputable def inf_series (a b : ℝ) : ℝ :=
  ∑' (n : ℕ), if n = 1 then 1 / (b * a)
  else if n % 2 = 0 then 1 / ((↑(n - 1) * a - b) * (↑n * a - b))
  else 1 / ((↑(n - 1) * a + b) * (↑n * a - b))

theorem infinite_series_sum (a b : ℝ) 
  (h₁ : a > 0) (h₂ : b > 0) (h₃ : a > b) :
  inf_series a b = 1 / (a * b) :=
sorry

end infinite_series_sum_l2102_210252


namespace area_of_figure_l2102_210231

noncomputable def area_enclosed : ℝ :=
  ∫ x in (0 : ℝ)..(2 * Real.pi / 3), 2 * Real.sin x

theorem area_of_figure :
  area_enclosed = 3 := by
  sorry

end area_of_figure_l2102_210231


namespace Ruth_math_class_percentage_l2102_210242

theorem Ruth_math_class_percentage :
  let hours_school_day := 8
  let days_school_week := 5
  let hours_math_week := 10
  let total_school_hours_week := hours_school_day * days_school_week
  (hours_math_week / total_school_hours_week) * 100 = 25 := 
by 
  let hours_school_day := 8
  let days_school_week := 5
  let hours_math_week := 10
  let total_school_hours_week := hours_school_day * days_school_week
  -- skip the proof here
  sorry

end Ruth_math_class_percentage_l2102_210242


namespace find_P_Q_R_l2102_210223

theorem find_P_Q_R :
  ∃ P Q R : ℝ, (∀ x : ℝ, x ≠ 2 ∧ x ≠ 4 → 
    (5 * x / ((x - 4) * (x - 2)^2) = P / (x - 4) + Q / (x - 2) + R / (x - 2)^2)) 
    ∧ P = 5 ∧ Q = -5 ∧ R = -5 :=
by
  sorry

end find_P_Q_R_l2102_210223


namespace area_triangle_sum_l2102_210206

theorem area_triangle_sum (AB : ℝ) (angle_BAC angle_ABC angle_ACB angle_EDC : ℝ) 
  (h_AB : AB = 1) (h_angle_BAC : angle_BAC = 70) (h_angle_ABC : angle_ABC = 50) 
  (h_angle_ACB : angle_ACB = 60) (h_angle_EDC : angle_EDC = 80) :
  let area_triangle := (1/2) * AB * (Real.sin angle_70 / Real.sin angle_60) * (Real.sin angle_60) 
  let area_CDE := (1/2) * (Real.sin angle_80)
  area_triangle + 2 * area_CDE = (Real.sin angle_70 + Real.sin angle_80) / 2 :=
sorry

end area_triangle_sum_l2102_210206


namespace min_value_b1_b2_l2102_210267

noncomputable def seq (b : ℕ → ℕ) : Prop :=
  ∀ n ≥ 1, b (n + 2) = (b n + 2017) / (1 + b (n + 1))

theorem min_value_b1_b2 (b : ℕ → ℕ)
  (h_pos : ∀ n, b n > 0)
  (h_seq : seq b) :
  b 1 + b 2 = 2018 := sorry

end min_value_b1_b2_l2102_210267


namespace find_natural_number_with_common_divisor_l2102_210234

def commonDivisor (a b : ℕ) (d : ℕ) : Prop :=
  d > 1 ∧ d ∣ a ∧ d ∣ b

theorem find_natural_number_with_common_divisor :
  ∃ n : ℕ, (∀ k : ℕ, 0 ≤ k ∧ k ≤ 20 →
    ∃ d : ℕ, commonDivisor (n + k) 30030 d) ∧ n = 9440 :=
by
  sorry

end find_natural_number_with_common_divisor_l2102_210234


namespace base_length_first_tri_sail_l2102_210212

-- Define the areas of the sails
def area_rect_sail : ℕ := 5 * 8
def area_second_tri_sail : ℕ := (4 * 6) / 2

-- Total canvas area needed
def total_canvas_area_needed : ℕ := 58

-- Calculate the total area so far (rectangular sail + second triangular sail)
def total_area_so_far : ℕ := area_rect_sail + area_second_tri_sail

-- Define the height of the first triangular sail
def height_first_tri_sail : ℕ := 4

-- Define the area needed for the first triangular sail
def area_first_tri_sail : ℕ := total_canvas_area_needed - total_area_so_far

-- Prove that the base length of the first triangular sail is 3 inches
theorem base_length_first_tri_sail : ∃ base : ℕ, base = 3 ∧ (base * height_first_tri_sail) / 2 = area_first_tri_sail := by
  use 3
  have h1 : (3 * 4) / 2 = 6 := by sorry -- This is a placeholder for actual calculation
  exact ⟨rfl, h1⟩

end base_length_first_tri_sail_l2102_210212


namespace vector_calculation_l2102_210270

def vec_a : ℝ × ℝ := (1, 1)
def vec_b : ℝ × ℝ := (1, -1)
def vec_result : ℝ × ℝ := (3 * vec_a.fst - 2 * vec_b.fst, 3 * vec_a.snd - 2 * vec_b.snd)
def target_vec : ℝ × ℝ := (1, 5)

theorem vector_calculation :
  vec_result = target_vec :=
sorry

end vector_calculation_l2102_210270


namespace age_difference_l2102_210219

variable (a b c d : ℕ)
variable (h1 : a + b = b + c + 11)
variable (h2 : a + c = c + d + 15)
variable (h3 : b + d = 36)
variable (h4 : a * 2 = 3 * d)

theorem age_difference :
  a - b = 39 :=
by
  sorry

end age_difference_l2102_210219


namespace joy_sixth_time_is_87_seconds_l2102_210246

def sixth_time (times : List ℝ) (new_median : ℝ) : ℝ :=
  let sorted_times := times |>.insertNth 2 (2 * new_median - times.nthLe 2 sorry)
  2 * new_median - times.nthLe 2 sorry

theorem joy_sixth_time_is_87_seconds (times : List ℝ) (new_median : ℝ) :
  times = [82, 85, 93, 95, 99] → new_median = 90 →
  sixth_time times new_median = 87 :=
by
  intros h_times h_median
  rw [h_times]
  rw [h_median]
  sorry

end joy_sixth_time_is_87_seconds_l2102_210246


namespace range_of_k_l2102_210236

theorem range_of_k {k : ℝ} : (∀ x : ℝ, x < 0 → (k - 2)/x > 0) ∧ (∀ x : ℝ, x > 0 → (k - 2)/x < 0) → k < 2 := 
by
  sorry

end range_of_k_l2102_210236


namespace sin_A_over_1_minus_cos_A_l2102_210279

variable {a b c : ℝ} -- Side lengths of the triangle
variable {A B C : ℝ} -- Angles opposite to the sides

theorem sin_A_over_1_minus_cos_A 
  (h_area : 0.5 * b * c * Real.sin A = a^2 - (b - c)^2) :
  Real.sin A / (1 - Real.cos A) = 3 :=
sorry

end sin_A_over_1_minus_cos_A_l2102_210279


namespace x_power6_y_power6_l2102_210290

theorem x_power6_y_power6 (x y a b : ℝ) (h1 : x + y = a) (h2 : x * y = b) :
  x^6 + y^6 = a^6 - 6 * a^4 * b + 9 * a^2 * b^2 - 2 * b^3 :=
sorry

end x_power6_y_power6_l2102_210290


namespace find_largest_integer_l2102_210213

theorem find_largest_integer : ∃ (x : ℤ), x < 120 ∧ x % 8 = 7 ∧ x = 119 := 
by
  use 119
  sorry

end find_largest_integer_l2102_210213


namespace curve_crossing_point_l2102_210216

theorem curve_crossing_point :
  (∃ t : ℝ, (t^2 - 4 = 2) ∧ (t^3 - 6 * t + 4 = 4)) ∧
  (∃ t' : ℝ, t ≠ t' ∧ (t'^2 - 4 = 2) ∧ (t'^3 - 6 * t' + 4 = 4)) :=
sorry

end curve_crossing_point_l2102_210216


namespace inversely_proportional_value_l2102_210292

theorem inversely_proportional_value (a b k : ℝ) (h1 : a * b = k) (h2 : a = 40) (h3 : b = 8) :
  ∃ a' : ℝ, a' * 10 = k ∧ a' = 32 :=
by {
  use 32,
  sorry
}

end inversely_proportional_value_l2102_210292


namespace solution_inequality_l2102_210298

variable (f : ℝ → ℝ)

-- Conditions
def is_even_function_at (f : ℝ → ℝ) (x : ℝ) : Prop := f (2 + x) = f (2 - x)
def is_increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop := ∀ ⦃x y⦄, x < y → x ∈ s → y ∈ s → f x < f y

-- Main statement
theorem solution_inequality 
  (h1 : ∀ x, is_even_function_at f x)
  (h2 : is_increasing_on f {x : ℝ | x ≤ 2}) :
  (∀ a : ℝ, (a > -1) ∧ (a ≠ 0) ↔ f (a^2 + 3*a + 2) < f (a^2 - a + 2)) :=
by {
  sorry
}

end solution_inequality_l2102_210298


namespace length_width_difference_l2102_210203

theorem length_width_difference
  (w l : ℝ)
  (h1 : l = 4 * w)
  (h2 : l * w = 768) :
  l - w = 24 * Real.sqrt 3 :=
by
  sorry

end length_width_difference_l2102_210203


namespace total_problems_l2102_210264

theorem total_problems (C W : ℕ) (h1 : 3 * C + 5 * W = 110) (h2 : C = 20) : C + W = 30 :=
by {
  sorry
}

end total_problems_l2102_210264


namespace solve_equation_l2102_210249

variable (a b c : ℝ)

theorem solve_equation (h : (a / Real.sqrt (18 * b)) * (c / Real.sqrt (72 * b)) = 1) : 
  a * c = 36 * b :=
by 
  -- Proof goes here
  sorry

end solve_equation_l2102_210249


namespace find_shirt_cost_l2102_210228

def cost_each_shirt (x : ℝ) : Prop :=
  let total_purchase_price := x + 5 + 30 + 14
  let shipping_cost := if total_purchase_price > 50 then 0.2 * total_purchase_price else 5
  let total_bill := total_purchase_price + shipping_cost
  total_bill = 102

theorem find_shirt_cost (x : ℝ) (h : cost_each_shirt x) : x = 36 :=
sorry

end find_shirt_cost_l2102_210228


namespace find_triangle_height_l2102_210211

-- Given conditions
def triangle_area : ℝ := 960
def base : ℝ := 48

-- The problem is to find the height such that 960 = (1/2) * 48 * height
theorem find_triangle_height (height : ℝ) 
  (h_area : triangle_area = (1/2) * base * height) : height = 40 := by
  sorry

end find_triangle_height_l2102_210211


namespace proof_5x_plus_4_l2102_210210

variable (x : ℝ)

-- Given condition
def condition := 5 * x - 8 = 15 * x + 12

-- Required proof
theorem proof_5x_plus_4 (h : condition x) : 5 * (x + 4) = 10 :=
by {
  sorry
}

end proof_5x_plus_4_l2102_210210


namespace increasing_on_interval_l2102_210238

open Real

noncomputable def f (x a b : ℝ) := abs (x^2 - 2*a*x + b)

theorem increasing_on_interval {a b : ℝ} (h : a^2 - b ≤ 0) :
  ∀ ⦃x1 x2⦄, a ≤ x1 → x1 ≤ x2 → f x1 a b ≤ f x2 a b := sorry

end increasing_on_interval_l2102_210238
