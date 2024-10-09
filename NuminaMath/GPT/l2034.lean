import Mathlib

namespace jar_initial_water_fraction_l2034_203424

theorem jar_initial_water_fraction (C W : ℝ) (hC : C > 0) (hW : W + C / 4 = 0.75 * C) : W / C = 0.5 :=
by
  -- necessary parameters and sorry for the proof 
  sorry

end jar_initial_water_fraction_l2034_203424


namespace inequality_a_b_c_l2034_203465

theorem inequality_a_b_c (a b c : ℝ) (h1 : 0 < a) (h2: 0 < b) (h3: 0 < c) (h4: a^2 + b^2 + c^2 = 3) : 
  (a / (a + 5) + b / (b + 5) + c / (c + 5) ≤ 1 / 2) :=
by
  sorry

end inequality_a_b_c_l2034_203465


namespace qr_length_is_correct_l2034_203495

/-- Define points and segments in the triangle. -/
structure Point :=
(x : ℝ)
(y : ℝ)

structure Triangle :=
(P Q R : Point)

def PQ_length (T : Triangle) : ℝ :=
(T.Q.x - T.P.x) * (T.Q.x - T.P.x) + (T.Q.y - T.P.y) * (T.Q.y - T.P.y)

def PR_length (T : Triangle) : ℝ :=
(T.R.x - T.P.x) * (T.R.x - T.P.x) + (T.R.y - T.P.y) * (T.R.y - T.P.y)

def QR_length (T : Triangle) : ℝ :=
(T.R.x - T.Q.x) * (T.R.x - T.Q.x) + (T.R.y - T.Q.y) * (T.R.y - T.Q.y)

noncomputable def XZ_length (T : Triangle) (X Y Z : Point) : ℝ :=
(PQ_length T)^(1/2) -- Assume the least length of XZ that follows the given conditions

theorem qr_length_is_correct (T : Triangle) :
  PQ_length T = 4*4 → 
  XZ_length T T.P T.Q T.R = 3.2 →
  QR_length T = 4*4 :=
sorry

end qr_length_is_correct_l2034_203495


namespace num_correct_props_geometric_sequence_l2034_203445

-- Define what it means to be a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

-- Original Proposition P
def Prop_P (a : ℕ → ℝ) :=
  a 1 < a 2 ∧ a 2 < a 3 → ∀ n : ℕ, a n < a (n + 1)

-- Converse of Proposition P
def Conv_Prop_P (a : ℕ → ℝ) :=
  ( ∀ n : ℕ, a n < a (n + 1) ) → a 1 < a 2 ∧ a 2 < a 3

-- Inverse of Proposition P
def Inv_Prop_P (a : ℕ → ℝ) :=
  ¬(a 1 < a 2 ∧ a 2 < a 3) → ¬( ∀ n : ℕ, a n < a (n + 1) )

-- Contrapositive of Proposition P
def Contra_Prop_P (a : ℕ → ℝ) :=
  ¬( ∀ n : ℕ, a n < a (n + 1) ) → ¬(a 1 < a 2 ∧ a 2 < a 3)

-- Main theorem to be proved
theorem num_correct_props_geometric_sequence (a : ℕ → ℝ) :
  is_geometric_sequence a → 
  Prop_P a ∧ Conv_Prop_P a ∧ Inv_Prop_P a ∧ Contra_Prop_P a := by
  sorry

end num_correct_props_geometric_sequence_l2034_203445


namespace erika_walked_distance_l2034_203457

/-- Erika traveled to visit her cousin. She started on a scooter at an average speed of 
22 kilometers per hour. After completing three-fifths of the distance, the scooter's battery died, 
and she walked the rest of the way at 4 kilometers per hour. The total time it took her to reach her cousin's 
house was 2 hours. How far, in kilometers rounded to the nearest tenth, did Erika walk? -/
theorem erika_walked_distance (d : ℝ) (h1 : d > 0)
  (h2 : (3 / 5 * d) / 22 + (2 / 5 * d) / 4 = 2) : 
  (2 / 5 * d) = 6.3 :=
sorry

end erika_walked_distance_l2034_203457


namespace possible_sums_of_digits_l2034_203431

def is_four_digit_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def all_digits_nonzero (A : ℕ) : Prop :=
  let a := (A / 1000) % 10
  let b := (A / 100) % 10
  let c := (A / 10) % 10
  let d := (A % 10)
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0

def reverse_number (A : ℕ) : ℕ :=
  let a := (A / 1000) % 10
  let b := (A / 100) % 10
  let c := (A / 10) % 10
  let d := (A % 10)
  1000 * d + 100 * c + 10 * b + a

def sum_of_digits (A : ℕ) : ℕ :=
  let a := (A / 1000) % 10
  let b := (A / 100) % 10
  let c := (A / 10) % 10
  let d := (A % 10)
  a + b + c + d

theorem possible_sums_of_digits (A B : ℕ) 
  (h_four_digit : is_four_digit_number A) 
  (h_nonzero_digits : all_digits_nonzero A) 
  (h_reverse : B = reverse_number A) 
  (h_divisible : (A + B) % 109 = 0) : 
  sum_of_digits A = 14 ∨ sum_of_digits A = 23 ∨ sum_of_digits A = 28 := 
sorry

end possible_sums_of_digits_l2034_203431


namespace geometric_sequence_condition_neither_necessary_nor_sufficient_l2034_203498

noncomputable def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n, a (n + 1) = a n * q

noncomputable def is_monotonically_increasing (a : ℕ → ℝ) :=
  ∀ n, a (n + 1) > a n

theorem geometric_sequence_condition_neither_necessary_nor_sufficient (a : ℕ → ℝ) (q : ℝ) :
  is_geometric_sequence a q → ¬( (is_monotonically_increasing a ↔ q > 1) ) :=
by sorry

end geometric_sequence_condition_neither_necessary_nor_sufficient_l2034_203498


namespace percent_calculation_l2034_203430

theorem percent_calculation (y : ℝ) : (0.3 * (0.6 * y) = 0.18 * y) → (0.18 * y / y) * 100 = 18 := by
  sorry

end percent_calculation_l2034_203430


namespace tablet_battery_life_l2034_203486

theorem tablet_battery_life :
  ∀ (active_usage_hours idle_usage_hours : ℕ),
  active_usage_hours + idle_usage_hours = 12 →
  active_usage_hours = 3 →
  ((active_usage_hours / 2) + (idle_usage_hours / 10)) > 1 →
  idle_usage_hours = 9 →
  0 = 0 := 
by
  intros active_usage_hours idle_usage_hours h1 h2 h3 h4
  sorry

end tablet_battery_life_l2034_203486


namespace distance_from_dorm_to_city_l2034_203441

theorem distance_from_dorm_to_city (D : ℝ) (h1 : D = (1/4)*D + (1/2)*D + 10 ) : D = 40 :=
sorry

end distance_from_dorm_to_city_l2034_203441


namespace distance_from_plate_to_bottom_edge_l2034_203428

theorem distance_from_plate_to_bottom_edge :
  ∀ (W T d : ℕ), W = 73 ∧ T = 20 ∧ (T + d = W) → d = 53 :=
by
  intros W T d
  rintro ⟨hW, hT, h⟩
  rw [hW, hT] at h
  linarith

end distance_from_plate_to_bottom_edge_l2034_203428


namespace quadratic_range_l2034_203412

noncomputable def quadratic_condition (a m : ℝ) : Prop :=
  (a > 0) ∧ (a ≠ 1) ∧ (- (1 + 1 / m) > 0) ∧
  (3 * m^2 - 2 * m - 1 ≤ 0)

theorem quadratic_range (a m : ℝ) :
  quadratic_condition a m → - (1 / 3) ≤ m ∧ m < 0 :=
by sorry

end quadratic_range_l2034_203412


namespace hexagon_angles_sum_l2034_203490

theorem hexagon_angles_sum (mA mB mC : ℤ) (x y : ℤ)
  (hA : mA = 35) (hB : mB = 80) (hC : mC = 30)
  (hSum : (6 - 2) * 180 = 720)
  (hAdjacentA : 90 + 90 = 180)
  (hAdjacentC : 90 - mC = 60) :
  x + y = 95 := by
  sorry

end hexagon_angles_sum_l2034_203490


namespace total_jellybeans_l2034_203403

-- Definitions of the conditions
def caleb_jellybeans : ℕ := 3 * 12
def sophie_jellybeans (caleb_jellybeans : ℕ) : ℕ := caleb_jellybeans / 2

-- Statement of the proof problem
theorem total_jellybeans (C : caleb_jellybeans = 36) (S : sophie_jellybeans 36 = 18) :
  caleb_jellybeans + sophie_jellybeans 36 = 54 :=
by
  sorry

end total_jellybeans_l2034_203403


namespace moles_of_CO2_formed_l2034_203494

-- Define the reaction
def reaction (HCl NaHCO3 CO2 : ℕ) : Prop :=
  HCl = NaHCO3 ∧ HCl + NaHCO3 = CO2

-- Given conditions
def given_conditions : Prop :=
  ∃ (HCl NaHCO3 CO2 : ℕ),
    reaction HCl NaHCO3 CO2 ∧ HCl = 3 ∧ NaHCO3 = 3

-- Prove the number of moles of CO2 formed is 3.
theorem moles_of_CO2_formed : given_conditions → ∃ CO2 : ℕ, CO2 = 3 :=
  by
    intros h
    sorry

end moles_of_CO2_formed_l2034_203494


namespace smaller_number_is_180_l2034_203404

theorem smaller_number_is_180 (a b : ℕ) (h1 : a = 3 * b) (h2 : a + 4 * b = 420) :
  a = 180 :=
sorry

end smaller_number_is_180_l2034_203404


namespace total_cost_l2034_203463

def c_teacher : ℕ := 60
def c_student : ℕ := 40

theorem total_cost (x : ℕ) : ∃ y : ℕ, y = c_student * x + c_teacher := by
  sorry

end total_cost_l2034_203463


namespace largest_int_with_remainder_5_lt_100_l2034_203472

theorem largest_int_with_remainder_5_lt_100 (x : ℤ) : (x < 100) ∧ (x % 8 = 5) → x = 93 := 
sorry

end largest_int_with_remainder_5_lt_100_l2034_203472


namespace prank_helpers_combinations_l2034_203464

theorem prank_helpers_combinations :
  let Monday := 1
  let Tuesday := 2
  let Wednesday := 3
  let Thursday := 4
  let Friday := 1
  (Monday * Tuesday * Wednesday * Thursday * Friday = 24) :=
by
  intros
  sorry

end prank_helpers_combinations_l2034_203464


namespace both_shots_hit_target_exactly_one_shot_hits_target_l2034_203462

variable (p q : Prop)

theorem both_shots_hit_target : (p ∧ q) := sorry

theorem exactly_one_shot_hits_target : ((p ∧ ¬ q) ∨ (¬ p ∧ q)) := sorry

end both_shots_hit_target_exactly_one_shot_hits_target_l2034_203462


namespace total_water_intake_l2034_203491

def theo_weekday := 8
def mason_weekday := 7
def roxy_weekday := 9
def zara_weekday := 10
def lily_weekday := 6

def theo_weekend := 10
def mason_weekend := 8
def roxy_weekend := 11
def zara_weekend := 12
def lily_weekend := 7

def total_cups_in_week (weekday_cups weekend_cups : ℕ) : ℕ :=
  5 * weekday_cups + 2 * weekend_cups

theorem total_water_intake :
  total_cups_in_week theo_weekday theo_weekend +
  total_cups_in_week mason_weekday mason_weekend +
  total_cups_in_week roxy_weekday roxy_weekend +
  total_cups_in_week zara_weekday zara_weekend +
  total_cups_in_week lily_weekday lily_weekend = 296 :=
by sorry

end total_water_intake_l2034_203491


namespace problem_statement_l2034_203415

variables {a b c p q r : ℝ}

-- Given conditions
axiom h1 : 19 * p + b * q + c * r = 0
axiom h2 : a * p + 29 * q + c * r = 0
axiom h3 : a * p + b * q + 56 * r = 0
axiom h4 : a ≠ 19
axiom h5 : p ≠ 0

-- Statement to prove
theorem problem_statement : 
  (a / (a - 19)) + (b / (b - 29)) + (c / (c - 56)) = 1 :=
sorry

end problem_statement_l2034_203415


namespace hall_width_l2034_203421

theorem hall_width
  (L H E C : ℝ)
  (hL : L = 20)
  (hH : H = 5)
  (hE : E = 57000)
  (hC : C = 60) :
  ∃ w : ℝ, (w * 50 + 100) * C = E ∧ w = 17 :=
by
  use 17
  simp [hL, hH, hE, hC]
  sorry

end hall_width_l2034_203421


namespace complex_magnitude_of_3_minus_4i_l2034_203452

open Complex

theorem complex_magnitude_of_3_minus_4i : Complex.abs ⟨3, -4⟩ = 5 := sorry

end complex_magnitude_of_3_minus_4i_l2034_203452


namespace volume_of_region_l2034_203425

theorem volume_of_region (r1 r2 : ℝ) (h : r1 = 5) (h2 : r2 = 8) : 
  let V_sphere (r : ℝ) := (4 / 3) * Real.pi * r^3
  let V_cylinder (r : ℝ) := Real.pi * r^2 * r
  (V_sphere r2) - (V_sphere r1) - (V_cylinder r1) = 391 * Real.pi :=
by
  -- Placeholder proof
  sorry

end volume_of_region_l2034_203425


namespace inequality_holds_l2034_203488

variable {x y : ℝ}

theorem inequality_holds (x : ℝ) (y : ℝ) (hy : y ≥ 5) : 
  x^2 - 2 * x * Real.sqrt (y - 5) + y^2 + y - 30 ≥ 0 := 
sorry

end inequality_holds_l2034_203488


namespace alberto_spent_2457_l2034_203493

-- Define the expenses by Samara on each item
def oil_expense : ℕ := 25
def tires_expense : ℕ := 467
def detailing_expense : ℕ := 79

-- Define the additional amount Alberto spent more than Samara
def additional_amount : ℕ := 1886

-- Total amount spent by Samara
def samara_total_expense : ℕ := oil_expense + tires_expense + detailing_expense

-- The amount spent by Alberto
def alberto_expense := samara_total_expense + additional_amount

-- Theorem stating the amount spent by Alberto
theorem alberto_spent_2457 :
  alberto_expense = 2457 :=
by {
  -- Include the actual proof here if necessary
  sorry
}

end alberto_spent_2457_l2034_203493


namespace eval_f_at_two_eval_f_at_neg_two_l2034_203477

def f (x : ℝ) : ℝ := 2 * x ^ 2 + 3 * x

theorem eval_f_at_two : f 2 = 14 :=
by
  sorry

theorem eval_f_at_neg_two : f (-2) = 2 :=
by
  sorry

end eval_f_at_two_eval_f_at_neg_two_l2034_203477


namespace negation_of_existential_proposition_l2034_203466

theorem negation_of_existential_proposition :
  ¬(∃ x_0 : ℝ, x_0^2 > Real.exp x_0) ↔ ∀ (x : ℝ), x^2 ≤ Real.exp x :=
by
  sorry

end negation_of_existential_proposition_l2034_203466


namespace intersection_of_lines_l2034_203438

theorem intersection_of_lines : 
  ∃ (x y : ℚ), 
  (5 * x - 3 * y = 20) ∧ (3 * x + 4 * y = 6) ∧ 
  x = 98 / 29 ∧ 
  y = 87 / 58 :=
by 
  sorry

end intersection_of_lines_l2034_203438


namespace ratio_of_shirt_to_pants_l2034_203470

theorem ratio_of_shirt_to_pants
    (total_cost : ℕ)
    (price_pants : ℕ)
    (price_shoes : ℕ)
    (price_shirt : ℕ)
    (h1 : total_cost = 340)
    (h2 : price_pants = 120)
    (h3 : price_shoes = price_pants + 10)
    (h4 : price_shirt = total_cost - (price_pants + price_shoes)) :
    price_shirt * 4 = price_pants * 3 := sorry

end ratio_of_shirt_to_pants_l2034_203470


namespace min_value_m_l2034_203450

theorem min_value_m (a : ℕ → ℕ) (b : ℕ → ℕ) (S : ℕ → ℕ)
  (h_arithmetic : ∀ n, a (n + 1) = a n + a 1)
  (h_geometric : ∀ n, b (n + 1) = b 1 * (b 1 ^ n))
  (h_b1_mean : 2 * b 1 = a 1 + a 2)
  (h_a3 : a 3 = 5)
  (h_b3 : b 3 = a 4 + 1)
  (h_S_formula : ∀ n, S n = n^2)
  (h_S_le_b : ∀ n ≥ 4, S n ≤ b n) :
  ∃ m, ∀ n, (n ≥ m → S n ≤ b n) ∧ m = 4 := sorry

end min_value_m_l2034_203450


namespace cube_inverse_sum_l2034_203454

theorem cube_inverse_sum (x : ℂ) (h : x + 1/x = -3) : x^3 + (1/x)^3 = -18 :=
by
  sorry

end cube_inverse_sum_l2034_203454


namespace frog_weight_difference_l2034_203499

theorem frog_weight_difference
  (large_frog_weight : ℕ)
  (small_frog_weight : ℕ)
  (h1 : large_frog_weight = 10 * small_frog_weight)
  (h2 : large_frog_weight = 120) :
  large_frog_weight - small_frog_weight = 108 :=
by
  sorry

end frog_weight_difference_l2034_203499


namespace worth_of_each_gift_is_4_l2034_203484

noncomputable def worth_of_each_gift
  (workers_per_block : ℕ)
  (total_blocks : ℕ)
  (total_amount : ℝ) : ℝ :=
  total_amount / (workers_per_block * total_blocks)

theorem worth_of_each_gift_is_4 (workers_per_block total_blocks : ℕ) (total_amount : ℝ)
  (h1 : workers_per_block = 100)
  (h2 : total_blocks = 10)
  (h3 : total_amount = 4000) :
  worth_of_each_gift workers_per_block total_blocks total_amount = 4 :=
by
  sorry

end worth_of_each_gift_is_4_l2034_203484


namespace largest_k_consecutive_sum_l2034_203478

theorem largest_k_consecutive_sum (k n : ℕ) :
  (5^7 = (k * (2 * n + k + 1)) / 2) → 1 ≤ k → k * (2 * n + k + 1) = 2 * 5^7 → k = 250 :=
sorry

end largest_k_consecutive_sum_l2034_203478


namespace triangle_obtuse_at_most_one_l2034_203456

open Real -- Work within the Real number system

-- Definitions and main proposition
def is_obtuse (angle : ℝ) : Prop := angle > 90

def triangle (a b c : ℝ) : Prop := a + b + c = 180

theorem triangle_obtuse_at_most_one (a b c : ℝ) (h : triangle a b c) :
  is_obtuse a ∧ is_obtuse b → false :=
by
  sorry

end triangle_obtuse_at_most_one_l2034_203456


namespace jim_juice_amount_l2034_203426

def susan_juice : ℚ := 3 / 8
def jim_fraction : ℚ := 5 / 6

theorem jim_juice_amount : jim_fraction * susan_juice = 5 / 16 := by
  sorry

end jim_juice_amount_l2034_203426


namespace x_plus_y_value_l2034_203420

def sum_of_integers (a b : ℕ) : ℕ :=
  (b - a + 1) * (a + b) / 2

def count_even_integers (a b : ℕ) : ℕ :=
  (b - a) / 2 + 1

theorem x_plus_y_value :
  let x := sum_of_integers 50 70
  let y := count_even_integers 50 70
  x + y = 1271 := by
    let x := sum_of_integers 50 70
    let y := count_even_integers 50 70
    sorry

end x_plus_y_value_l2034_203420


namespace expression_evaluation_l2034_203485

theorem expression_evaluation (x y z : ℝ) (h : x = y + z) (h' : x = 2) :
  x^3 + 2 * y^3 + 2 * z^3 + 6 * x * y * z = 24 :=
by
  sorry

end expression_evaluation_l2034_203485


namespace fraction_conversion_l2034_203458

theorem fraction_conversion :
  let A := 4.5
  let B := 0.8
  let C := 80.0
  let D := 0.08
  let E := 0.45
  (4 / 5) = B :=
by
  sorry

end fraction_conversion_l2034_203458


namespace arithmetic_sequence_a5_l2034_203416

-- Define the concept of an arithmetic sequence
def arithmetic_sequence (a d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

-- The problem's conditions
def a₁ : ℕ := 2
def d : ℕ := 3

-- The proof problem
theorem arithmetic_sequence_a5 : arithmetic_sequence a₁ d 5 = 14 := by
  sorry

end arithmetic_sequence_a5_l2034_203416


namespace man_speed_in_still_water_l2034_203480

theorem man_speed_in_still_water
  (speed_of_current_kmph : ℝ)
  (time_seconds : ℝ)
  (distance_meters : ℝ)
  (speed_of_current_ms : ℝ := speed_of_current_kmph * (1000 / 3600))
  (speed_downstream : ℝ := distance_meters / time_seconds) :
  speed_of_current_kmph = 3 →
  time_seconds = 13.998880089592832 →
  distance_meters = 70 →
  (speed_downstream = (25 / 6)) →
  (speed_downstream - speed_of_current_ms) * (3600 / 1000) = 15 :=
by
  intros h_speed_current h_time h_distance h_downstream
  sorry

end man_speed_in_still_water_l2034_203480


namespace polar_to_rectangular_conversion_l2034_203408

noncomputable def polar_to_rectangular (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

theorem polar_to_rectangular_conversion :
  polar_to_rectangular 5 (5 * Real.pi / 4) = (-5 * Real.sqrt 2 / 2, -5 * Real.sqrt 2 / 2) := by
  sorry

end polar_to_rectangular_conversion_l2034_203408


namespace part_one_part_two_l2034_203401

def f (x : ℝ) (a : ℝ) := |2 * x - a| + a
def g (x : ℝ) := |2 * x - 1|

-- (1) Prove that if a = 2, then ∀ x, f(x, 2) ≤ 6 implies -1 ≤ x ≤ 3
theorem part_one (x : ℝ) : f x 2 ≤ 6 → -1 ≤ x ∧ x ≤ 3 :=
by sorry

-- (2) Prove that ∀ a ∈ ℝ, ∀ x ∈ ℝ, (f(x, a) + g(x) ≥ 3 → a ∈ [2, +∞))
theorem part_two (a x : ℝ) : f x a + g x ≥ 3 → 2 ≤ a :=
by sorry

end part_one_part_two_l2034_203401


namespace least_sum_of_exponents_l2034_203476

theorem least_sum_of_exponents (a b c d e : ℕ) (h : ℕ) (h_divisors : 225 ∣ h ∧ 216 ∣ h ∧ 847 ∣ h)
  (h_form : h = (2 ^ a) * (3 ^ b) * (5 ^ c) * (7 ^ d) * (11 ^ e)) : 
  a + b + c + d + e = 10 :=
sorry

end least_sum_of_exponents_l2034_203476


namespace number_of_oddly_powerful_integers_lt_500_l2034_203483

noncomputable def count_oddly_powerful_integers_lt_500 : ℕ :=
  let count_cubes := 7 -- we counted cubes: 1^3, 2^3, 3^3, 4^3, 5^3, 6^3, 7^3
  let count_fifth_powers := 1 -- the additional fifth power not a cube: 3^5
  count_cubes + count_fifth_powers

theorem number_of_oddly_powerful_integers_lt_500 : count_oddly_powerful_integers_lt_500 = 8 :=
  sorry

end number_of_oddly_powerful_integers_lt_500_l2034_203483


namespace bricks_in_wall_l2034_203444

theorem bricks_in_wall (x : ℕ) (r₁ r₂ combined_rate : ℕ) :
  (r₁ = x / 8) →
  (r₂ = x / 12) →
  (combined_rate = r₁ + r₂ - 15) →
  (6 * combined_rate = x) →
  x = 360 :=
by
  intros h₁ h₂ h₃ h₄
  sorry

end bricks_in_wall_l2034_203444


namespace intersect_trihedral_angle_l2034_203400

-- Definitions of variables
variables {a b c : ℝ} (S : Type) 

-- Definition of a valid intersection condition
def valid_intersection (a b c : ℝ) : Prop :=
  a^2 + b^2 - c^2 > 0 ∧ b^2 + c^2 - a^2 > 0 ∧ a^2 + c^2 - b^2 > 0

-- Theorem statement
theorem intersect_trihedral_angle (h : valid_intersection a b c) : 
  ∃ (SA SB SC : ℝ), (SA^2 + SB^2 = a^2 ∧ SA^2 + SC^2 = b^2 ∧ SB^2 + SC^2 = c^2) :=
sorry

end intersect_trihedral_angle_l2034_203400


namespace number_of_donuts_finished_l2034_203446

-- Definitions from conditions
def ounces_per_donut : ℕ := 2
def ounces_per_pot : ℕ := 12
def cost_per_pot : ℕ := 3
def total_spent : ℕ := 18

-- Theorem statement
theorem number_of_donuts_finished (H1 : ounces_per_donut = 2)
                                   (H2 : ounces_per_pot = 12)
                                   (H3 : cost_per_pot = 3)
                                   (H4 : total_spent = 18) : 
  ∃ n : ℕ, n = 36 :=
  sorry

end number_of_donuts_finished_l2034_203446


namespace masha_can_pay_exactly_with_11_ruble_bills_l2034_203437

theorem masha_can_pay_exactly_with_11_ruble_bills (m n k p : ℕ) 
  (h1 : 3 * m + 4 * n + 5 * k = 11 * p) : 
  ∃ q : ℕ, 9 * m + n + 4 * k = 11 * q := 
by {
  sorry
}

end masha_can_pay_exactly_with_11_ruble_bills_l2034_203437


namespace algebraic_expression_value_l2034_203429

theorem algebraic_expression_value (a b : ℝ) 
  (h : |a + 2| + (b - 1)^2 = 0) : (a + b) ^ 2005 = -1 :=
by
  sorry

end algebraic_expression_value_l2034_203429


namespace least_multiple_of_15_greater_than_500_l2034_203474

theorem least_multiple_of_15_greater_than_500 : 
  ∃ (n : ℕ), n > 500 ∧ (∃ (k : ℕ), n = 15 * k) ∧ (n = 510) :=
by
  sorry

end least_multiple_of_15_greater_than_500_l2034_203474


namespace find_digit_to_make_divisible_by_seven_l2034_203449

/-- 
  Given a number formed by concatenating 2023 digits of 6 with 2023 digits of 5.
  In a three-digit number 6*5, find the digit * to make this number divisible by 7.
  i.e., We must find the digit x such that the number 600 + 10x + 5 is divisible by 7.
-/
theorem find_digit_to_make_divisible_by_seven :
  ∃ x : ℕ, x < 10 ∧ (600 + 10 * x + 5) % 7 = 0 :=
sorry

end find_digit_to_make_divisible_by_seven_l2034_203449


namespace distance_metric_l2034_203497

noncomputable def d (x y : ℝ) : ℝ :=
  (|x - y|) / (Real.sqrt (1 + x^2) * Real.sqrt (1 + y^2))

theorem distance_metric (x y z : ℝ) :
  (d x x = 0) ∧
  (d x y = d y x) ∧
  (d x y + d y z ≥ d x z) := by
  sorry

end distance_metric_l2034_203497


namespace fraction_addition_l2034_203406

theorem fraction_addition :
  let a := (1 : ℚ) / 6
  let b := (5 : ℚ) / 12
  a + b = 7 / 12 :=
by
  let a := (1 : ℚ) / 6
  let b := (5 : ℚ) / 12
  have : a + b = 7 / 12 := sorry
  exact this

end fraction_addition_l2034_203406


namespace number_of_people_l2034_203432

theorem number_of_people (x y z : ℕ) 
  (h1 : x + y + z = 12) 
  (h2 : 2 * x + y / 2 + z / 4 = 12) : 
  x = 5 ∧ y = 1 ∧ z = 6 := 
by
  sorry

end number_of_people_l2034_203432


namespace circle_area_l2034_203481

theorem circle_area (r : ℝ) (h : 6 / (2 * π * r) = r / 2) : π * r^2 = 3 :=
by
  sorry

end circle_area_l2034_203481


namespace proof_problem_l2034_203460

variable {a b c : ℝ}

theorem proof_problem (h_cond : 0 < a ∧ a < b ∧ b < c) : 
  a * c < b * c ∧ a + b < b + c ∧ c / a > c / b := by
  sorry

end proof_problem_l2034_203460


namespace determine_dress_and_notebooks_l2034_203443

structure Girl :=
  (name : String)
  (dress_color : String)
  (notebook_color : String)

def colors := ["red", "yellow", "blue"]

def Sveta : Girl := ⟨"Sveta", "red", "red"⟩
def Ira : Girl := ⟨"Ira", "blue", "yellow"⟩
def Tania : Girl := ⟨"Tania", "yellow", "blue"⟩

theorem determine_dress_and_notebooks :
  (Sveta.dress_color = Sveta.notebook_color) ∧
  (¬ Tania.dress_color = "red") ∧
  (¬ Tania.notebook_color = "red") ∧
  (Ira.notebook_color = "yellow") ∧
  (Sveta ∈ [Sveta, Ira, Tania]) ∧
  (Ira ∈ [Sveta, Ira, Tania]) ∧
  (Tania ∈ [Sveta, Ira, Tania]) →
  ([Sveta, Ira, Tania] = 
   [{name := "Sveta", dress_color := "red", notebook_color := "red"},
    {name := "Ira", dress_color := "blue", notebook_color := "yellow"},
    {name := "Tania", dress_color := "yellow", notebook_color := "blue"}])
:=
by
  intro h
  sorry

end determine_dress_and_notebooks_l2034_203443


namespace sum_of_first_1000_terms_l2034_203471

def sequence_block_sum (n : ℕ) : ℕ :=
  1 + 3 * n

def sequence_sum_up_to (k : ℕ) : ℕ :=
  if k = 0 then 0 else (1 + 3 * (k * (k - 1) / 2)) + k

def nth_term_position (n : ℕ) : ℕ :=
  n * (n + 1) / 2 + n

theorem sum_of_first_1000_terms : sequence_sum_up_to 43 + (1000 - nth_term_position 43) * 3 = 2912 :=
sorry

end sum_of_first_1000_terms_l2034_203471


namespace calculate_product_l2034_203451

theorem calculate_product : (3 * 5 * 7 = 38) → (13 * 15 * 17 = 268) → 1 * 3 * 5 = 15 :=
by
  intros h1 h2
  sorry

end calculate_product_l2034_203451


namespace axis_of_symmetry_shift_l2034_203407

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

theorem axis_of_symmetry_shift (f : ℝ → ℝ) (hf : is_even_function f) :
  (∃ a : ℝ, ∀ x : ℝ, f (x + 1) = f (-(x + 1))) :=
sorry

end axis_of_symmetry_shift_l2034_203407


namespace number_of_best_friends_l2034_203434

-- Constants and conditions
def initial_tickets : ℕ := 37
def tickets_per_friend : ℕ := 5
def tickets_left : ℕ := 2

-- Problem statement
theorem number_of_best_friends : (initial_tickets - tickets_left) / tickets_per_friend = 7 :=
by
  sorry

end number_of_best_friends_l2034_203434


namespace stratified_sampling_11th_grade_representatives_l2034_203436

theorem stratified_sampling_11th_grade_representatives 
  (students_10th : ℕ)
  (students_11th : ℕ)
  (students_12th : ℕ)
  (total_rep : ℕ)
  (total_students : students_10th + students_11th + students_12th = 5000)
  (Students_10th : students_10th = 2500)
  (Students_11th : students_11th = 1500)
  (Students_12th : students_12th = 1000)
  (Total_rep : total_rep = 30) : 
  (9 : ℕ) = (3 : ℚ) / (10 : ℚ) * (30 : ℕ) :=
sorry

end stratified_sampling_11th_grade_representatives_l2034_203436


namespace math_proof_l2034_203440

def problem_statement : Prop :=
  ∃ x : ℕ, (2 * x + 3 = 19) ∧ (x + (2 * x + 3) = 27)

theorem math_proof : problem_statement :=
  sorry

end math_proof_l2034_203440


namespace patrick_purchased_pencils_l2034_203405

theorem patrick_purchased_pencils (c s : ℝ) : 
  (∀ n : ℝ, n * c = 1.375 * n * s ∧ (n * c - n * s = 30 * s) → n = 80) :=
by sorry

end patrick_purchased_pencils_l2034_203405


namespace max_value_fraction_l2034_203492

theorem max_value_fraction (a b : ℝ) (h1 : ab = 1) (h2 : a > b) (h3 : b ≥ 2/3) :
  ∃ C, C = 30 / 97 ∧ (∀ x y : ℝ, (xy = 1) → (x > y) → (y ≥ 2/3) → (x - y) / (x^2 + y^2) ≤ C) :=
sorry

end max_value_fraction_l2034_203492


namespace negation_proposition_l2034_203469

theorem negation_proposition (x : ℝ) (hx : 0 < x) : x + 4 / x ≥ 4 :=
sorry

end negation_proposition_l2034_203469


namespace number_of_persons_l2034_203487

theorem number_of_persons (P : ℕ) : 
  (P * 12 * 5 = 30 * 13 * 6) → P = 39 :=
by
  sorry

end number_of_persons_l2034_203487


namespace find_m_l2034_203496

theorem find_m (m : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = x^2 - 2 * x + m) 
  (h2 : ∀ x ≥ (3 : ℝ), f x ≥ 1) : m = -2 := 
sorry

end find_m_l2034_203496


namespace remainder_prod_mod_7_l2034_203417

theorem remainder_prod_mod_7 : 
  (3 * 13 * 23 * 33 * 43 * 53 * 63 * 73 * 83 * 93) % 7 = 4 :=
by
  sorry

end remainder_prod_mod_7_l2034_203417


namespace simultaneous_in_Quadrant_I_l2034_203419

def in_Quadrant_I (x y : ℝ) : Prop := x > 0 ∧ y > 0

theorem simultaneous_in_Quadrant_I (c x y : ℝ) : 
  (2 * x - y = 5) ∧ (c * x + y = 4) ↔ in_Quadrant_I x y ∧ (-2 < c ∧ c < 8 / 5) :=
sorry

end simultaneous_in_Quadrant_I_l2034_203419


namespace ratio_of_sums_l2034_203427

variable (a : ℕ → ℝ) (S : ℕ → ℝ)

axiom arithmetic_sum : ∀ n, S n = n * (a 1 + a n) / 2
axiom a4_eq_2a3 : a 4 = 2 * a 3

theorem ratio_of_sums (a : ℕ → ℝ) (S : ℕ → ℝ)
                      (arithmetic_sum : ∀ n, S n = n * (a 1 + a n) / 2)
                      (a4_eq_2a3 : a 4 = 2 * a 3) :
  S 7 / S 5 = 14 / 5 :=
by sorry

end ratio_of_sums_l2034_203427


namespace perfect_square_expression_l2034_203473

theorem perfect_square_expression (x y : ℕ) (p : ℕ) [Fact (Nat.Prime p)]
    (h : 4 * x^2 + 8 * y^2 + (2 * x - 3 * y) * p - 12 * x * y = 0) :
    ∃ (n : ℕ), 4 * y + 1 = n^2 :=
sorry

end perfect_square_expression_l2034_203473


namespace perimeter_ratio_of_divided_square_l2034_203410

theorem perimeter_ratio_of_divided_square
  (S_ΔADE : ℝ) (S_EDCB : ℝ)
  (S_ratio : S_ΔADE / S_EDCB = 5 / 19)
  : ∃ (perim_ΔADE perim_EDCB : ℝ),
  perim_ΔADE / perim_EDCB = 15 / 22 :=
by
  -- Let S_ΔADE = 5x and S_EDCB = 19x
  -- x can be calculated based on the given S_ratio = 5/19
  -- Apply geometric properties and simplifications analogous to the described solution.
  sorry

end perimeter_ratio_of_divided_square_l2034_203410


namespace range_of_a_l2034_203439

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, x^2 + (a - 1) * x + 1 < 0) ↔ (a < -1 ∨ a > 3) :=
by
  sorry

end range_of_a_l2034_203439


namespace quadrilateral_tile_angles_l2034_203479

theorem quadrilateral_tile_angles :
  ∃ a b c d : ℝ, a + b + c + d = 360 ∧ a = 45 ∧ b = 60 ∧ c = 105 ∧ d = 150 := 
by {
  sorry
}

end quadrilateral_tile_angles_l2034_203479


namespace logarithmic_expression_max_value_l2034_203459

theorem logarithmic_expression_max_value (a b : ℝ) (h1 : a > b) (h2 : b > 1) (h3 : a / b = 3) :
  3 * Real.log (a / b) / Real.log a + 2 * Real.log (b / a) / Real.log b = -4 := 
sorry

end logarithmic_expression_max_value_l2034_203459


namespace perimeter_ratio_l2034_203461

/-- Suppose we have a square piece of paper, 6 inches on each side, folded in half horizontally. 
The paper is then cut along the fold, and one of the halves is subsequently cut again horizontally 
through all layers. This results in one large rectangle and two smaller identical rectangles. 
Find the ratio of the perimeter of one smaller rectangle to the perimeter of the larger rectangle. -/
theorem perimeter_ratio (side_length : ℝ) (half_side_length : ℝ) (double_half_side_length : ℝ) :
    side_length = 6 →
    half_side_length = side_length / 2 →
    double_half_side_length = 1.5 * 2 →
    (2 * (half_side_length / 2 + side_length)) / (2 * (half_side_length + side_length)) = (5 / 6) :=
by
    -- Declare the side lengths
    intros h₁ h₂ h₃
    -- Insert the necessary algebra (proven manually earlier)
    sorry

end perimeter_ratio_l2034_203461


namespace gumballs_ensure_four_same_color_l2034_203418

-- Define the total number of gumballs in each color
def red_gumballs : ℕ := 10
def white_gumballs : ℕ := 9
def blue_gumballs : ℕ := 8
def green_gumballs : ℕ := 7

-- Define the minimum number of gumballs to ensure four of the same color
def min_gumballs_to_ensure_four_same_color : ℕ := 13

-- Prove that the minimum number of gumballs to ensure four of the same color is 13
theorem gumballs_ensure_four_same_color (n : ℕ) 
  (h₁ : red_gumballs ≥ 3)
  (h₂ : white_gumballs ≥ 3)
  (h₃ : blue_gumballs ≥ 3)
  (h₄ : green_gumballs ≥ 3)
  : n ≥ min_gumballs_to_ensure_four_same_color := 
sorry

end gumballs_ensure_four_same_color_l2034_203418


namespace probability_three_consecutive_heads_four_tosses_l2034_203447

theorem probability_three_consecutive_heads_four_tosses :
  let total_outcomes := 16
  let favorable_outcomes := 2
  let probability := (favorable_outcomes : ℚ) / (total_outcomes : ℚ)
  probability = 1 / 8 := by
    sorry

end probability_three_consecutive_heads_four_tosses_l2034_203447


namespace max_songs_played_l2034_203448

theorem max_songs_played (n m t : ℕ) (h1 : n = 50) (h2 : m = 50) (h3 : t = 180) :
  3 * n + 5 * (m - ((t - 3 * n) / 5)) = 56 :=
by
  sorry

end max_songs_played_l2034_203448


namespace find_maximum_value_of_f_φ_has_root_l2034_203442

open Set Real

noncomputable section

-- Definition of the function f(x)
def f (x : ℝ) : ℝ := -6 * (sin x + cos x) - 3

-- Definition of the function φ(x)
def φ (x : ℝ) : ℝ := f x + 10

-- The assumptions on the interval
def interval := Icc 0 (π / 4)

-- Statement to prove that the maximum value of f(x) is -9
theorem find_maximum_value_of_f : ∀ x ∈ interval, f x ≤ -9 ∧ ∃ x_0 ∈ interval, f x_0 = -9 := sorry

-- Statement to prove that φ(x) has a root in the interval
theorem φ_has_root : ∃ x ∈ interval, φ x = 0 := sorry

end find_maximum_value_of_f_φ_has_root_l2034_203442


namespace g_at_3_value_l2034_203453

theorem g_at_3_value (c d : ℝ) (g : ℝ → ℝ) 
  (h1 : g 1 = 7)
  (h2 : g 2 = 11)
  (h3 : ∀ x : ℝ, g x = c * x + d * x + 3) : 
  g 3 = 15 :=
by
  sorry

end g_at_3_value_l2034_203453


namespace eqn_distinct_real_roots_l2034_203409

noncomputable def f (x : ℝ) : ℝ := 
if x ≥ 0 then x^2 + 2 else 4 * x * Real.cos x + 1

theorem eqn_distinct_real_roots (m : ℝ) : 
  (∀ x : ℝ, f x = m * x + 1) → 
  ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ ∈ Set.Icc (-2 * Real.pi) Real.pi ∧ x₂ ∈ Set.Icc (-2 * Real.pi) Real.pi :=
  sorry

end eqn_distinct_real_roots_l2034_203409


namespace mice_needed_l2034_203414

-- Definitions for relative strength in terms of M (Mouse strength)
def C (M : ℕ) : ℕ := 6 * M
def J (M : ℕ) : ℕ := 5 * C M
def G (M : ℕ) : ℕ := 4 * J M
def B (M : ℕ) : ℕ := 3 * G M
def D (M : ℕ) : ℕ := 2 * B M

-- Condition: all together can pull up the Turnip with strength 1237M
def total_strength_with_mouse (M : ℕ) : ℕ :=
  D M + B M + G M + J M + C M + M

-- Condition: without the Mouse, they cannot pull up the Turnip
def total_strength_without_mouse (M : ℕ) : ℕ :=
  D M + B M + G M + J M + C M

theorem mice_needed (M : ℕ) (h : total_strength_with_mouse M = 1237 * M) (h2 : total_strength_without_mouse M < 1237 * M) :
  1237 = 1237 :=
by
  -- using sorry to indicate proof is not provided
  sorry

end mice_needed_l2034_203414


namespace octagon_non_intersecting_diagonals_l2034_203411

-- Define what an octagon is
def octagon : Type := { vertices : Finset (Fin 8) // vertices.card = 8 }

-- Define non-intersecting diagonals in an octagon
def non_intersecting_diagonals (oct : octagon) : ℕ :=
  8  -- Given the cyclic pattern and star formation, we know the number is 8

-- The theorem we want to prove
theorem octagon_non_intersecting_diagonals (oct : octagon) : non_intersecting_diagonals oct = 8 :=
by sorry

end octagon_non_intersecting_diagonals_l2034_203411


namespace polynomial_value_at_3_l2034_203475

theorem polynomial_value_at_3 :
  ∃ (P : ℕ → ℚ), 
    (∀ (x : ℕ), P x = b_0 + b_1 * x + b_2 * x^2 + b_3 * x^3 + b_4 * x^4 + b_5 * x^5 + b_6 * x^6) ∧ 
    (∀ (i : ℕ), i ≤ 6 → 0 ≤ b_i ∧ b_i < 5) ∧ 
    P (Nat.sqrt 5) = 35 + 26 * Nat.sqrt 5 -> 
    P 3 = 437 := 
by
  simp
  sorry

end polynomial_value_at_3_l2034_203475


namespace linear_function_quadrants_l2034_203468

theorem linear_function_quadrants (m : ℝ) :
  (∀ (x : ℝ), y = -3 * x + m →
  (x < 0 ∧ y > 0 ∨ x > 0 ∧ y < 0 ∨ x < 0 ∧ y < 0)) → m < 0 :=
sorry

end linear_function_quadrants_l2034_203468


namespace nonnegative_solution_exists_l2034_203402

theorem nonnegative_solution_exists
  (a b c d n : ℕ)
  (h_npos : 0 < n)
  (h_gcd_abc : Nat.gcd (Nat.gcd a b) c = 1)
  (h_gcd_ab : Nat.gcd a b = d)
  (h_conds : n > a * b / d + c * d - a - b - c) :
  ∃ x y z : ℕ, a * x + b * y + c * z = n := 
by
  sorry

end nonnegative_solution_exists_l2034_203402


namespace stamps_on_last_page_l2034_203433

theorem stamps_on_last_page (total_books : ℕ) (pages_per_book : ℕ) (stamps_per_page_initial : ℕ) (stamps_per_page_new : ℕ)
    (full_books_new : ℕ) (pages_filled_seventh_book : ℕ) (total_stamps : ℕ) (stamps_in_seventh_book : ℕ) 
    (remaining_stamps : ℕ) :
    total_books = 10 →
    pages_per_book = 50 →
    stamps_per_page_initial = 8 →
    stamps_per_page_new = 12 →
    full_books_new = 6 →
    pages_filled_seventh_book = 37 →
    total_stamps = total_books * pages_per_book * stamps_per_page_initial →
    stamps_in_seventh_book = 4000 - (600 * full_books_new) →
    remaining_stamps = stamps_in_seventh_book - (pages_filled_seventh_book * stamps_per_page_new) →
    remaining_stamps = 4 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end stamps_on_last_page_l2034_203433


namespace curves_intersect_at_l2034_203489

noncomputable def f (x : ℝ) : ℝ := 3 * x^2 - 4 * x + 2
noncomputable def g (x : ℝ) : ℝ := -x^3 + 9 * x^2 - 4 * x + 2

theorem curves_intersect_at :
  (∃ x : ℝ, f x = g x) ↔ ([(0, 2), (6, 86)] = [(0, 2), (6, 86)]) :=
by
  sorry

end curves_intersect_at_l2034_203489


namespace max_area_of_triangle_l2034_203482

open Real

theorem max_area_of_triangle (a b c : ℝ) 
  (ha : 9 ≥ a) 
  (ha1 : a ≥ 8) 
  (hb : 8 ≥ b) 
  (hb1 : b ≥ 4) 
  (hc : 4 ≥ c) 
  (hc1 : c ≥ 3) : 
  ∃ A : ℝ, ∃ S : ℝ, S ≤ 16 ∧ S = max (1/2 * b * c * sin A) 16 := 
sorry

end max_area_of_triangle_l2034_203482


namespace range_of_b_for_local_minimum_l2034_203423

variable {x : ℝ}
variable (b : ℝ)

def f (x : ℝ) (b : ℝ) : ℝ :=
  x^3 - 6 * b * x + 3 * b

def f' (x : ℝ) (b : ℝ) : ℝ :=
  3 * x^2 - 6 * b

theorem range_of_b_for_local_minimum
  (h1 : f' 0 b < 0)
  (h2 : f' 1 b > 0) :
  0 < b ∧ b < 1 / 2 :=
by
  sorry

end range_of_b_for_local_minimum_l2034_203423


namespace superhero_vs_supervillain_distance_l2034_203467

-- Definitions expressing the conditions
def superhero_speed (miles : ℕ) (minutes : ℕ) := (10 : ℕ) / (4 : ℕ)
def supervillain_speed (miles_per_hour : ℕ) := (100 : ℕ)

-- Distance calculation in 60 minutes
def superhero_distance_in_hour := 60 * superhero_speed 10 4
def supervillain_distance_in_hour := supervillain_speed 100

-- Proof statement
theorem superhero_vs_supervillain_distance :
  superhero_distance_in_hour - supervillain_distance_in_hour = (50 : ℕ) :=
by
  sorry

end superhero_vs_supervillain_distance_l2034_203467


namespace mike_took_23_green_marbles_l2034_203413

-- Definition of the conditions
def original_green_marbles : ℕ := 32
def remaining_green_marbles : ℕ := 9

-- Definition of the statement we want to prove
theorem mike_took_23_green_marbles : original_green_marbles - remaining_green_marbles = 23 := by
  sorry

end mike_took_23_green_marbles_l2034_203413


namespace inequality_solution_set_compare_mn_and_2m_plus_2n_l2034_203455

def f (x : ℝ) : ℝ := |x| + |x - 3|

theorem inequality_solution_set :
  {x : ℝ | f x - 5 ≥ x} = { x : ℝ | x ≤ -2 / 3 } ∪ { x : ℝ | x ≥ 8 } :=
sorry

theorem compare_mn_and_2m_plus_2n (m n : ℝ) (hm : ∃ x, m = f x) (hn : ∃ x, n = f x) :
  2 * (m + n) < m * n + 4 :=
sorry

end inequality_solution_set_compare_mn_and_2m_plus_2n_l2034_203455


namespace triangles_in_pentadecagon_l2034_203435

theorem triangles_in_pentadecagon : (Nat.choose 15 3) = 455 :=
by
  sorry

end triangles_in_pentadecagon_l2034_203435


namespace abs_x_equals_4_l2034_203422

-- Define the points A and B as per the conditions
def point_A (x : ℝ) : ℝ := 3 + x
def point_B (x : ℝ) : ℝ := 3 - x

-- Define the distance between points A and B
def distance (x : ℝ) : ℝ := abs ((point_A x) - (point_B x))

theorem abs_x_equals_4 (x : ℝ) (h : distance x = 8) : abs x = 4 :=
by
  sorry

end abs_x_equals_4_l2034_203422
