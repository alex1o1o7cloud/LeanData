import Mathlib

namespace product_of_two_numbers_is_320_l60_60139

theorem product_of_two_numbers_is_320 (x y : ℕ) (h1 : x + y = 36) (h2 : x - y = 4) (h3 : x = 5 * (y / 4)) : x * y = 320 :=
by {
  sorry
}

end product_of_two_numbers_is_320_l60_60139


namespace min_value_x_plus_2y_l60_60766

theorem min_value_x_plus_2y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = x * y) : x + 2 * y ≥ 3 + 2 * Real.sqrt 2 :=
sorry

end min_value_x_plus_2y_l60_60766


namespace domain_of_log_l60_60882

def log_domain := {x : ℝ | x > 1}

theorem domain_of_log : {x : ℝ | ∃ y, y = log_domain} = {x : ℝ | x > 1} :=
by
  sorry

end domain_of_log_l60_60882


namespace balls_in_boxes_l60_60945

-- Define the conditions
def num_balls : ℕ := 3
def num_boxes : ℕ := 4

-- Define the problem
theorem balls_in_boxes : (num_boxes ^ num_balls) = 64 :=
by
  -- We acknowledge that we are skipping the proof details here
  sorry

end balls_in_boxes_l60_60945


namespace find_f_2022_l60_60070

def satisfies_condition (f : ℝ → ℝ) : Prop :=
  ∀ a b : ℝ, f ((a + 2 * b) / 3) = (f a + 2 * f b) / 3

variables (f : ℝ → ℝ)
  (h_condition : satisfies_condition f)
  (h_f1 : f 1 = 1)
  (h_f4 : f 4 = 7)

theorem find_f_2022 : f 2022 = 4043 :=
  sorry

end find_f_2022_l60_60070


namespace volume_of_regular_triangular_pyramid_l60_60794

noncomputable def pyramid_volume (a b γ : ℝ) : ℝ :=
  (1 / 3) * (a^2 * Real.sqrt 3 / 4) * Real.sqrt (b^2 - (a * Real.sqrt 3 / (2 * Real.cos (γ / 2)))^2)

theorem volume_of_regular_triangular_pyramid (a b γ : ℝ) :
  pyramid_volume a b γ = (1 / 3) * (a^2 * Real.sqrt 3 / 4) * Real.sqrt (b^2 - (a * Real.sqrt 3 / (2 * Real.cos (γ / 2)))^2) :=
by
  sorry

end volume_of_regular_triangular_pyramid_l60_60794


namespace find_s_l60_60942

theorem find_s (k s : ℝ) (h1 : 5 = k * 2^s) (h2 : 45 = k * 8^s) : s = (Real.log 9) / (2 * Real.log 2) :=
by
  sorry

end find_s_l60_60942


namespace weight_of_person_being_replaced_l60_60176

variable (W_old : ℝ)

theorem weight_of_person_being_replaced :
  (W_old : ℝ) = 35 :=
by
  -- Given: The average weight of 8 persons increases by 5 kg.
  -- The weight of the new person is 75 kg.
  -- The total weight increase is 40 kg.
  -- Prove that W_old = 35 kg.
  sorry

end weight_of_person_being_replaced_l60_60176


namespace find_m_l60_60418

theorem find_m {m : ℝ} (a b : ℝ × ℝ) (H : a = (3, m) ∧ b = (2, -1)) (H_dot : a.1 * b.1 + a.2 * b.2 = 0) : m = 6 := 
by
  sorry

end find_m_l60_60418


namespace false_implies_not_all_ripe_l60_60008

def all_ripe (basket : Type) [Nonempty basket] (P : basket → Prop) : Prop :=
  ∀ x : basket, P x

theorem false_implies_not_all_ripe
  (basket : Type)
  [Nonempty basket]
  (P : basket → Prop)
  (h : ¬ all_ripe basket P) :
  (∃ x, ¬ P x) ∧ ¬ all_ripe basket P :=
by
  sorry

end false_implies_not_all_ripe_l60_60008


namespace find_number_l60_60648

theorem find_number (N p q : ℝ) 
  (h1 : N / p = 6) 
  (h2 : N / q = 18) 
  (h3 : p - q = 1 / 3) : 
  N = 3 := 
by 
  sorry

end find_number_l60_60648


namespace small_pump_fill_time_l60_60761

noncomputable def small_pump_time (large_pump_time combined_time : ℝ) : ℝ :=
  let large_pump_rate := 1 / large_pump_time
  let combined_rate := 1 / combined_time
  let small_pump_rate := combined_rate - large_pump_rate
  1 / small_pump_rate

theorem small_pump_fill_time :
  small_pump_time (1 / 3) 0.2857142857142857 = 2 :=
by
  sorry

end small_pump_fill_time_l60_60761


namespace negation_of_existence_l60_60211

variable (x : ℝ)

theorem negation_of_existence :
  (¬ ∃ x₀ : ℝ, x₀^2 + 2 * x₀ - 3 > 0) ↔ (∀ x : ℝ, x^2 + 2*x - 3 ≤ 0) :=
by sorry

end negation_of_existence_l60_60211


namespace select_two_subsets_union_six_elements_l60_60087

def f (n : ℕ) : ℕ :=
  if n = 0 then 1 else 3 * f (n - 1) - 1

theorem select_two_subsets_union_six_elements :
  f 6 = 365 :=
by
  sorry

end select_two_subsets_union_six_elements_l60_60087


namespace sum_of_series_l60_60057

theorem sum_of_series : (1 / (1 * 2 * 3) + 1 / (2 * 3 * 4) + 1 / (3 * 4 * 5) + 1 / (4 * 5 * 6)) = 7 / 30 :=
by
  sorry

end sum_of_series_l60_60057


namespace contrapositive_geometric_sequence_l60_60108

def geometric_sequence (a b c : ℝ) : Prop :=
  b^2 = a * c

theorem contrapositive_geometric_sequence (a b c : ℝ) :
  (b^2 ≠ a * c) → ¬geometric_sequence a b c :=
by
  intros h
  unfold geometric_sequence
  assumption

end contrapositive_geometric_sequence_l60_60108


namespace triangle_side_lengths_log_l60_60017

theorem triangle_side_lengths_log (m : ℕ) (log15 log81 logm : ℝ)
  (h1 : log15 = Real.log 15 / Real.log 10)
  (h2 : log81 = Real.log 81 / Real.log 10)
  (h3 : logm = Real.log m / Real.log 10)
  (h4 : 0 < log15 ∧ 0 < log81 ∧ 0 < logm)
  (h5 : log15 + log81 > logm)
  (h6 : log15 + logm > log81)
  (h7 : log81 + logm > log15)
  (h8 : m > 0) :
  6 ≤ m ∧ m < 1215 → 
  ∃ n : ℕ, n = 1215 - 6 ∧ n = 1209 :=
by
  sorry

end triangle_side_lengths_log_l60_60017


namespace find_p_q_l60_60315

theorem find_p_q (p q : ℚ) : 
    (∀ x, x^5 - x^4 + x^3 - p*x^2 + q*x + 9 = 0 → (x = -3 ∨ x = 2)) →
    (p, q) = (-19.5, -55.5) :=
by {
  sorry
}

end find_p_q_l60_60315


namespace transform_quadratic_to_squared_form_l60_60460

theorem transform_quadratic_to_squared_form :
  ∀ x : ℝ, 2 * x^2 - 3 * x + 1 = 0 → (x - 3 / 4)^2 = 1 / 16 :=
by
  intro x h
  sorry

end transform_quadratic_to_squared_form_l60_60460


namespace range_of_a_l60_60089

noncomputable def f (x a : ℝ) : ℝ := x * Real.log x + a * x^2 - (2 * a + 1) * x + 1

theorem range_of_a (a : ℝ) (h_a : 0 < a ∧ a ≤ 1/2) : 
  ∀ x : ℝ, x ∈ Set.Ici a → f x a ≥ a^3 - a - 1/8 :=
by
  sorry

end range_of_a_l60_60089


namespace range_of_x_squared_plus_y_squared_l60_60907

def increasing (f : ℝ → ℝ) := ∀ x y, x < y → f x < f y
def symmetric_about_origin (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

theorem range_of_x_squared_plus_y_squared 
  (f : ℝ → ℝ) 
  (h_incr : increasing f) 
  (h_symm : symmetric_about_origin f) 
  (h_ineq : ∀ x y, f (x^2 - 6 * x) + f (y^2 - 8 * y + 24) < 0) : 
  ∀ x y, 16 < x^2 + y^2 ∧ x^2 + y^2 < 36 := 
sorry

end range_of_x_squared_plus_y_squared_l60_60907


namespace solution_set_of_inequality_l60_60561

theorem solution_set_of_inequality :
  {x : ℝ | (3 * x - 1) / (2 - x) ≥ 0} = {x : ℝ | 1/3 ≤ x ∧ x < 2} :=
by
  sorry

end solution_set_of_inequality_l60_60561


namespace equilateral_triangle_M_properties_l60_60207

-- Define the points involved
variables (A B C M P Q R : ℝ)
-- Define distances from M to the sides as given by perpendiculars
variables (d_AP d_BQ d_CR d_PB d_QC d_RA : ℝ)

-- Equilateral triangle assumption and perpendiculars from M to sides
def equilateral_triangle (A B C : ℝ) : Prop := sorry
def perpendicular_from_point (M P R : ℝ) (line : ℝ) : Prop := sorry

-- Problem statement encapsulating the given conditions and what needs to be proved:
theorem equilateral_triangle_M_properties
  (h_triangle: equilateral_triangle A B C)
  (h_perp_AP: perpendicular_from_point M P A B)
  (h_perp_BQ: perpendicular_from_point M Q B C)
  (h_perp_CR: perpendicular_from_point M R C A) :
  (d_AP^2 + d_BQ^2 + d_CR^2 = d_PB^2 + d_QC^2 + d_RA^2) ∧ 
  (d_AP + d_BQ + d_CR = d_PB + d_QC + d_RA) := sorry

end equilateral_triangle_M_properties_l60_60207


namespace no_x4_term_implies_a_zero_l60_60840

theorem no_x4_term_implies_a_zero (a : ℝ) :
  ¬ (∃ (x : ℝ), -5 * x^3 * (x^2 + a * x + 5) = -5 * x^5 - 5 * a * x^4 - 25 * x^3 + 5 * a * x^4) →
  a = 0 :=
by
  -- Step through the proof process to derive this conclusion
  sorry

end no_x4_term_implies_a_zero_l60_60840


namespace faucet_leakage_volume_l60_60385

def leakage_rate : ℝ := 0.1
def time_seconds : ℝ := 14400
def expected_volume : ℝ := 1.4 * 10^3

theorem faucet_leakage_volume : 
  leakage_rate * time_seconds = expected_volume := 
by
  -- proof
  sorry

end faucet_leakage_volume_l60_60385


namespace odd_prime_divisibility_two_prime_divisibility_l60_60026

theorem odd_prime_divisibility (p a n : ℕ) (hp : p % 2 = 1) (hp_prime : Nat.Prime p)
  (ha : a > 0) (hn : n > 0) (div_cond : p^n ∣ a^p - 1) : p^(n-1) ∣ a - 1 :=
sorry

theorem two_prime_divisibility (a n : ℕ) (ha : a > 0) (hn : n > 0) (div_cond : 2^n ∣ a^2 - 1) : ¬ 2^(n-1) ∣ a - 1 :=
sorry

end odd_prime_divisibility_two_prime_divisibility_l60_60026


namespace cost_price_computer_table_l60_60516

theorem cost_price_computer_table 
  (CP SP : ℝ)
  (h1 : SP = CP * 1.20)
  (h2 : SP = 8400) :
  CP = 7000 :=
by
  sorry

end cost_price_computer_table_l60_60516


namespace max_questions_wrong_to_succeed_l60_60004

theorem max_questions_wrong_to_succeed :
  ∀ (total_questions : ℕ) (passing_percentage : ℚ),
  total_questions = 50 →
  passing_percentage = 0.75 →
  ∃ (max_wrong : ℕ), max_wrong = 12 ∧
    (total_questions - max_wrong) ≥ passing_percentage * total_questions := by
  intro total_questions passing_percentage h1 h2
  use 12
  constructor
  . rfl
  . sorry  -- Proof omitted

end max_questions_wrong_to_succeed_l60_60004


namespace simplify_frac_l60_60229

theorem simplify_frac : (5^4 + 5^2) / (5^3 - 5) = 65 / 12 :=
by 
  sorry

end simplify_frac_l60_60229


namespace factorial_power_of_two_iff_power_of_two_l60_60347

-- Assuming n is a positive integer
variable {n : ℕ} (h : n > 0)

theorem factorial_power_of_two_iff_power_of_two :
  (∃ k : ℕ, n = 2^k ) ↔ ∃ m : ℕ, 2^(n-1) ∣ n! :=
by {
  sorry
}

end factorial_power_of_two_iff_power_of_two_l60_60347


namespace probability_interval_l60_60800

theorem probability_interval (P_A P_B P_A_inter_P_B : ℝ) (h1 : P_A = 3 / 4) (h2 : P_B = 2 / 3) : 
  5/12 ≤ P_A_inter_P_B ∧ P_A_inter_P_B ≤ 2/3 :=
sorry

end probability_interval_l60_60800


namespace hotel_rooms_l60_60594

theorem hotel_rooms (h₁ : ∀ R : ℕ, (∃ n : ℕ, n = R * 3) → (∃ m : ℕ, m = 2 * R * 3) → m = 60) : (∃ R : ℕ, R = 10) :=
by
  sorry

end hotel_rooms_l60_60594


namespace arthur_walking_distance_l60_60152

/-- Arthur walks 8 blocks west and 10 blocks south, 
    each block being 1/4 mile -/
theorem arthur_walking_distance 
  (blocks_west : ℕ) (blocks_south : ℕ) (block_distance : ℚ)
  (h1 : blocks_west = 8) (h2 : blocks_south = 10) (h3 : block_distance = 1/4) :
  (blocks_west + blocks_south) * block_distance = 4.5 := 
by
  sorry

end arthur_walking_distance_l60_60152


namespace solve_for_y_l60_60716

theorem solve_for_y (x y : ℝ) (h₁ : x^(2 * y) = 64) (h₂ : x = 8) : y = 1 :=
by
  sorry

end solve_for_y_l60_60716


namespace Louisa_traveled_240_miles_first_day_l60_60668

noncomputable def distance_first_day (h : ℕ) := 60 * (h - 3)

theorem Louisa_traveled_240_miles_first_day :
  ∃ h : ℕ, 420 = 60 * h ∧ distance_first_day h = 240 :=
by
  sorry

end Louisa_traveled_240_miles_first_day_l60_60668


namespace absolute_value_zero_l60_60164

theorem absolute_value_zero (x : ℝ) (h : |4 * x + 6| = 0) : x = -3 / 2 :=
sorry

end absolute_value_zero_l60_60164


namespace function_above_x_axis_l60_60404

theorem function_above_x_axis (m : ℝ) : 
  (∀ x : ℝ, x > 0 → 9^x - m * 3^x + m + 1 > 0) ↔ m < 2 + 2 * Real.sqrt 2 :=
sorry

end function_above_x_axis_l60_60404


namespace equilibrium_mass_l60_60974

variable (l m2 S g : ℝ) (m1 : ℝ)

-- Given conditions
def length_of_rod : ℝ := 0.5 -- length l in meters
def mass_of_rod : ℝ := 2 -- mass m2 in kg
def distance_S : ℝ := 0.1 -- distance S in meters
def gravity : ℝ := 9.8 -- gravitational acceleration in m/s^2

-- Equivalence statement
theorem equilibrium_mass (h1 : l = length_of_rod)
                         (h2 : m2 = mass_of_rod)
                         (h3 : S = distance_S)
                         (h4 : g = gravity) :
  m1 = 10 := sorry

end equilibrium_mass_l60_60974


namespace bagel_pieces_after_10_cuts_l60_60436

def bagel_pieces_after_cuts (initial_pieces : ℕ) (cuts : ℕ) : ℕ :=
  initial_pieces + cuts

theorem bagel_pieces_after_10_cuts : bagel_pieces_after_cuts 1 10 = 11 := by
  sorry

end bagel_pieces_after_10_cuts_l60_60436


namespace find_q_l60_60329

def Q (x : ℝ) (p q r : ℝ) : ℝ := x^3 + p * x^2 + q * x + r

theorem find_q (p q r : ℝ) (h1 : -p = 2 * (-r)) (h2 : -p = 1 + p + q + r) (hy_intercept : r = 5) : q = -24 :=
by
  sorry

end find_q_l60_60329


namespace least_positive_value_l60_60350

theorem least_positive_value (x y z : ℤ) : ∃ x y z : ℤ, 0 < 72 * x + 54 * y + 36 * z ∧ ∀ (a b c : ℤ), 0 < 72 * a + 54 * b + 36 * c → 72 * x + 54 * y + 36 * z ≤ 72 * a + 54 * b + 36 * c :=
sorry

end least_positive_value_l60_60350


namespace right_angled_triangle_area_l60_60171

theorem right_angled_triangle_area 
  (a b c : ℝ) (h1 : a^2 + b^2 = c^2) (h2 : a + b + c = 18) (h3 : a^2 + b^2 + c^2 = 128) : 
  (1/2) * a * b = 9 :=
by
  -- Proof will be added here
  sorry

end right_angled_triangle_area_l60_60171


namespace range_of_f_l60_60114

noncomputable def f (x : ℝ) : ℝ := 
  Real.cos (2 * x - Real.pi / 3) + 2 * Real.sin (x - Real.pi / 4) * Real.sin (x + Real.pi / 4)

theorem range_of_f : ∀ x ∈ Set.Icc (-Real.pi / 12) (Real.pi / 2), 
  -Real.sqrt 3 / 2 ≤ f x ∧ f x ≤ 1 := by
  sorry

end range_of_f_l60_60114


namespace quadratic_intersects_xaxis_once_l60_60048

theorem quadratic_intersects_xaxis_once (k : ℝ) : 
  (∃ x : ℝ, x^2 - 2*x + k = 0) ↔ k = 1 :=
by
  sorry

end quadratic_intersects_xaxis_once_l60_60048


namespace rectangle_area_l60_60348

theorem rectangle_area (d : ℝ) (w : ℝ) (h : (3 * w)^2 + w^2 = d^2) : 
  3 * w^2 = d^2 / 10 :=
by
  sorry

end rectangle_area_l60_60348


namespace wax_current_eq_l60_60895

-- Define the constants for the wax required and additional wax needed
def w_required : ℕ := 166
def w_more : ℕ := 146

-- Define the term to represent the current wax he has
def w_current : ℕ := w_required - w_more

-- Theorem statement to prove the current wax quantity
theorem wax_current_eq : w_current = 20 := by
  -- Proof outline would go here, but per instructions, we skip with sorry
  sorry

end wax_current_eq_l60_60895


namespace increase_in_expenses_is_20_percent_l60_60559

noncomputable def man's_salary : ℝ := 6500
noncomputable def initial_savings : ℝ := 0.20 * man's_salary
noncomputable def new_savings : ℝ := 260
noncomputable def reduction_in_savings : ℝ := initial_savings - new_savings
noncomputable def initial_expenses : ℝ := 0.80 * man's_salary
noncomputable def increase_in_expenses_percentage : ℝ := (reduction_in_savings / initial_expenses) * 100

theorem increase_in_expenses_is_20_percent :
  increase_in_expenses_percentage = 20 := by
  sorry

end increase_in_expenses_is_20_percent_l60_60559


namespace height_percentage_increase_l60_60781

theorem height_percentage_increase (B A : ℝ) 
  (hA : A = B * 0.8) : ((B - A) / A) * 100 = 25 := by
--   Given the condition that A's height is 20% less than B's height
--   translate into A = B * 0.8
--   We need to show ((B - A) / A) * 100 = 25
sorry

end height_percentage_increase_l60_60781


namespace find_x_l60_60160

theorem find_x (x : ℚ) (h : (3 - x) / (2 - x) - 1 / (x - 2) = 3) : x = 1 := 
  sorry

end find_x_l60_60160


namespace drops_of_glue_needed_l60_60550

def number_of_clippings (friend : ℕ) : ℕ :=
  match friend with
  | 1 => 4
  | 2 => 7
  | 3 => 5
  | 4 => 3
  | 5 => 5
  | 6 => 8
  | 7 => 2
  | 8 => 6
  | _ => 0

def total_drops_of_glue : ℕ :=
  (number_of_clippings 1 +
   number_of_clippings 2 +
   number_of_clippings 3 +
   number_of_clippings 4 +
   number_of_clippings 5 +
   number_of_clippings 6 +
   number_of_clippings 7 +
   number_of_clippings 8) * 6

theorem drops_of_glue_needed : total_drops_of_glue = 240 :=
by
  sorry

end drops_of_glue_needed_l60_60550


namespace greg_age_is_16_l60_60924

-- Definitions based on given conditions
def cindy_age : ℕ := 5
def jan_age : ℕ := cindy_age + 2
def marcia_age : ℕ := 2 * jan_age
def greg_age : ℕ := marcia_age + 2

-- Theorem stating that Greg's age is 16 years given the above conditions
theorem greg_age_is_16 : greg_age = 16 := by
  sorry

end greg_age_is_16_l60_60924


namespace find_common_difference_find_max_sum_find_max_n_l60_60828

-- Condition for the sequence
def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d

-- Problem statement (1): Find the common difference
theorem find_common_difference (a : ℕ → ℤ) (d : ℤ) (h1 : a 1 = 23)
  (h2 : is_arithmetic_sequence a d)
  (h6 : a 6 > 0)
  (h7 : a 7 < 0) : d = -4 :=
sorry

-- Problem statement (2): Find the maximum value of the sum S₆
theorem find_max_sum (d : ℤ) (h : d = -4) : 6 * 23 + (6 * 5 / 2) * d = 78 :=
sorry

-- Problem statement (3): Find the maximum value of n when S_n > 0
theorem find_max_n (d : ℤ) (h : d = -4) : ∀ n : ℕ, (n > 0 ∧ (23 * n + (n * (n - 1) / 2) * d > 0)) → n ≤ 12 :=
sorry

end find_common_difference_find_max_sum_find_max_n_l60_60828


namespace point_distance_5_5_l60_60411

-- Define the distance function in the context of the problem
def distance_from_origin (x : ℝ) : ℝ := abs x

-- Formalize the proposition
theorem point_distance_5_5 (x : ℝ) : distance_from_origin x = 5.5 → (x = -5.5 ∨ x = 5.5) :=
by
  intro h
  simp [distance_from_origin] at h
  sorry

end point_distance_5_5_l60_60411


namespace largest_lucky_number_l60_60588

theorem largest_lucky_number : 
  let a := 1
  let b := 4
  let lucky_number (x y : ℕ) := x + y + x * y
  let c1 := lucky_number a b
  let c2 := lucky_number b c1
  let c3 := lucky_number c1 c2
  c3 = 499 :=
by
  sorry

end largest_lucky_number_l60_60588


namespace sum_of_digits_Joey_age_twice_Max_next_l60_60855

noncomputable def Joey_is_two_years_older (C : ℕ) : ℕ := C + 2

noncomputable def Max_age_today := 2

noncomputable def Eight_multiples_of_Max (C : ℕ) := 
  ∃ n : ℕ, C = 24 + n

noncomputable def Next_Joey_age_twice_Max (C J M n : ℕ): Prop := J + n = 2 * (M + n)

theorem sum_of_digits_Joey_age_twice_Max_next (C J M n : ℕ) 
  (h1: J = Joey_is_two_years_older C)
  (h2: M = Max_age_today)
  (h3: Eight_multiples_of_Max C)
  (h4: Next_Joey_age_twice_Max C J M n) 
  : ∃ s, s = 7 :=
sorry

end sum_of_digits_Joey_age_twice_Max_next_l60_60855


namespace ratio_of_u_to_v_l60_60009

theorem ratio_of_u_to_v {b u v : ℝ} 
  (h1 : b ≠ 0)
  (h2 : 0 = 12 * u + b)
  (h3 : 0 = 8 * v + b) : 
  u / v = 2 / 3 := 
by
  sorry

end ratio_of_u_to_v_l60_60009


namespace consecutive_days_sum_l60_60101

theorem consecutive_days_sum (x : ℕ) (h : 3 * x + 3 = 33) : x = 10 ∧ x + 1 = 11 ∧ x + 2 = 12 :=
by {
  sorry
}

end consecutive_days_sum_l60_60101


namespace polygon_sides_l60_60497

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 + 360 = 1800) : n = 10 := by
  sorry

end polygon_sides_l60_60497


namespace draw_at_least_two_first_grade_products_l60_60243

theorem draw_at_least_two_first_grade_products :
  let total_products := 9
  let first_grade := 4
  let second_grade := 3
  let third_grade := 2
  let total_draws := 4
  let ways_to_draw := Nat.choose total_products total_draws
  let ways_no_first_grade := Nat.choose (second_grade + third_grade) total_draws
  let ways_one_first_grade := Nat.choose first_grade 1 * Nat.choose (second_grade + third_grade) (total_draws - 1)
  ways_to_draw - ways_no_first_grade - ways_one_first_grade = 81 := sorry

end draw_at_least_two_first_grade_products_l60_60243


namespace find_bc_div_a_l60_60926

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin x + 2 * Real.cos x + 1

variable (a b c : ℝ)

def satisfied (x : ℝ) : Prop := a * f x + b * f (x - c) = 1

theorem find_bc_div_a (ha : ∀ x, satisfied a b c x) : (b * Real.cos c / a) = -1 := 
by sorry

end find_bc_div_a_l60_60926


namespace book_price_l60_60356

theorem book_price (B P : ℝ) 
  (h1 : (1 / 3) * B = 36) 
  (h2 : (2 / 3) * B * P = 252) : 
  P = 3.5 :=
by {
  sorry
}

end book_price_l60_60356


namespace cube_volume_l60_60043

theorem cube_volume (A : ℝ) (hA : A = 96) (s : ℝ) (hS : A = 6 * s^2) : s^3 = 64 := by
  sorry

end cube_volume_l60_60043


namespace product_of_three_numbers_l60_60652

theorem product_of_three_numbers
  (x y z n : ℤ)
  (h1 : x + y + z = 165)
  (h2 : n = 7 * x)
  (h3 : n = y - 9)
  (h4 : n = z + 9) :
  x * y * z = 64328 := 
by
  sorry

end product_of_three_numbers_l60_60652


namespace max_value_abs_diff_PQ_PR_l60_60179

-- Definitions for the points on the given curves
def hyperbola (x y : ℝ) : Prop := (x^2 / 16) - (y^2 / 9) = 1
def circle1 (x y : ℝ) : Prop := (x - 5)^2 + y^2 = 1
def circle2 (x y : ℝ) : Prop := (x + 5)^2 + y^2 = 1

-- Statement of the problem as a theorem
theorem max_value_abs_diff_PQ_PR (P Q R : ℝ × ℝ)
(hyp_P : hyperbola P.1 P.2)
(hyp_Q : circle1 Q.1 Q.2)
(hyp_R : circle2 R.1 R.2) :
  max (abs (dist P Q - dist P R)) = 10 :=
sorry

end max_value_abs_diff_PQ_PR_l60_60179


namespace sculpture_height_is_34_inches_l60_60448

-- Define the height of the base in inches
def height_of_base_in_inches : ℕ := 2

-- Define the total height in feet
def total_height_in_feet : ℕ := 3

-- Convert feet to inches (1 foot = 12 inches)
def total_height_in_inches (feet : ℕ) : ℕ := feet * 12

-- The height of the sculpture, given the base and total height
def height_of_sculpture (total_height base_height : ℕ) : ℕ := total_height - base_height

-- State the theorem that the height of the sculpture is 34 inches
theorem sculpture_height_is_34_inches :
  height_of_sculpture (total_height_in_inches total_height_in_feet) height_of_base_in_inches = 34 := by
  sorry

end sculpture_height_is_34_inches_l60_60448


namespace simple_interest_l60_60174

theorem simple_interest (P R T : ℝ) (hP : P = 8965) (hR : R = 9) (hT : T = 5) : 
    (P * R * T) / 100 = 806.85 := 
by 
  sorry

end simple_interest_l60_60174


namespace smallest_non_factor_product_l60_60188

theorem smallest_non_factor_product (a b : ℕ) (h1 : a ≠ b) (h2 : a ∣ 48) (h3 : b ∣ 48) (h4 : ¬ (a * b ∣ 48)) : a * b = 18 :=
by
  -- proof intentionally omitted
  sorry

end smallest_non_factor_product_l60_60188


namespace greg_books_difference_l60_60066

theorem greg_books_difference (M K G X : ℕ)
  (hM : M = 32)
  (hK : K = M / 4)
  (hG : G = 2 * K + X)
  (htotal : M + K + G = 65) :
  X = 9 :=
by
  sorry

end greg_books_difference_l60_60066


namespace integer_sided_triangle_with_60_degree_angle_exists_l60_60512

theorem integer_sided_triangle_with_60_degree_angle_exists 
  (m n t : ℤ) : 
  ∃ (x y z : ℤ), (x = (m^2 - n^2) * t) ∧ 
                  (y = m * (m - 2 * n) * t) ∧ 
                  (z = (m^2 - m * n + n^2) * t) := by
  sorry

end integer_sided_triangle_with_60_degree_angle_exists_l60_60512


namespace contribution_amount_l60_60830

-- Definitions based on conditions
variable (x : ℝ)

-- Total amount needed
def total_needed := 200

-- Contributions from different families
def contribution_two_families := 2 * x
def contribution_eight_families := 8 * 10 -- 80
def contribution_ten_families := 10 * 5 -- 50
def total_contribution := contribution_two_families + contribution_eight_families + contribution_ten_families

-- Amount raised so far given they need 30 more to reach the target
def raised_so_far := total_needed - 30 -- 170

-- Statement to prove
theorem contribution_amount :
  total_contribution x = raised_so_far →
  x = 20 := by 
  sorry

end contribution_amount_l60_60830


namespace second_machine_completion_time_l60_60186

variable (time_first_machine : ℝ) (rate_first_machine : ℝ) (rate_combined : ℝ)
variable (rate_second_machine: ℝ) (y : ℝ)

def processing_rate_first_machine := rate_first_machine = 100
def processing_rate_combined := rate_combined = 1000 / 3
def processing_rate_second_machine := rate_second_machine = rate_combined - rate_first_machine
def completion_time_second_machine := y = 1000 / rate_second_machine

theorem second_machine_completion_time
  (h1: processing_rate_first_machine rate_first_machine)
  (h2: processing_rate_combined rate_combined)
  (h3: processing_rate_second_machine rate_combined rate_first_machine rate_second_machine)
  (h4: completion_time_second_machine rate_second_machine y) :
  y = 30 / 7 :=
sorry

end second_machine_completion_time_l60_60186


namespace complement_intersection_eq_l60_60893

variable (U P Q : Set ℕ)
variable (hU : U = {1, 2, 3})
variable (hP : P = {1, 2})
variable (hQ : Q = {2, 3})

theorem complement_intersection_eq : 
  (U \ (P ∩ Q)) = {1, 3} := by
  sorry

end complement_intersection_eq_l60_60893


namespace necessary_but_not_sufficient_l60_60444

variables (α β : Plane) (m : Line)

-- Define what it means for planes and lines to be perpendicular
def plane_perpendicular (p1 p2 : Plane) : Prop := sorry
def line_perpendicular_plane (l : Line) (p : Plane) : Prop := sorry

-- The main theorem to be established
theorem necessary_but_not_sufficient :
  (plane_perpendicular α β) → (line_perpendicular_plane m β) ∧ ¬ ((plane_perpendicular α β) ↔ (line_perpendicular_plane m β)) :=
sorry

end necessary_but_not_sufficient_l60_60444


namespace saree_final_price_l60_60253

noncomputable def saree_original_price : ℝ := 5000
noncomputable def first_discount_rate : ℝ := 0.20
noncomputable def second_discount_rate : ℝ := 0.15
noncomputable def third_discount_rate : ℝ := 0.10
noncomputable def fourth_discount_rate : ℝ := 0.05
noncomputable def tax_rate : ℝ := 0.12
noncomputable def luxury_tax_rate : ℝ := 0.05
noncomputable def custom_fee : ℝ := 200
noncomputable def exchange_rate_to_usd : ℝ := 0.013

theorem saree_final_price :
  let price_after_first_discount := saree_original_price * (1 - first_discount_rate)
  let price_after_second_discount := price_after_first_discount * (1 - second_discount_rate)
  let price_after_third_discount := price_after_second_discount * (1 - third_discount_rate)
  let price_after_fourth_discount := price_after_third_discount * (1 - fourth_discount_rate)
  let tax := price_after_fourth_discount * tax_rate
  let luxury_tax := price_after_fourth_discount * luxury_tax_rate
  let total_charges := tax + luxury_tax + custom_fee
  let total_price_rs := price_after_fourth_discount + total_charges
  let final_price_usd := total_price_rs * exchange_rate_to_usd
  abs (final_price_usd - 46.82) < 0.01 :=
by sorry

end saree_final_price_l60_60253


namespace find_b_value_l60_60890

-- Let's define the given conditions as hypotheses in Lean

theorem find_b_value 
  (x1 y1 x2 y2 : ℤ) 
  (h1 : (x1, y1) = (2, 2)) 
  (h2 : (x2, y2) = (8, 14)) 
  (midpoint : ∃ (m1 m2 : ℤ), m1 = (x1 + x2) / 2 ∧ m2 = (y1 + y2) / 2 ∧ (m1, m2) = (5, 8))
  (perpendicular_bisector : ∀ (x y : ℤ), x + y = b → (x, y) = (5, 8)) :
  b = 13 := 
by {
  sorry
}

end find_b_value_l60_60890


namespace truncated_cone_contact_radius_l60_60417

theorem truncated_cone_contact_radius (R r r' ζ : ℝ)
  (h volume_condition : ℝ)
  (R_pos : 0 < R)
  (r_pos : 0 < r)
  (r'_pos : 0 < r')
  (ζ_pos : 0 < ζ)
  (h_eq : h = 2 * R)
  (volume_condition_eq :
    (2 : ℝ) * ((4 / 3) * Real.pi * R^3) = 
    (2 / 3) * Real.pi * h * (r^2 + r * r' + r'^2)) :
  ζ = (2 * R * Real.sqrt 5) / 5 :=
by
  sorry

end truncated_cone_contact_radius_l60_60417


namespace sum_m_n_l60_60905

open Real

noncomputable def f (x : ℝ) : ℝ := |log x / log 2|

theorem sum_m_n (m n : ℝ) (hm_pos : 0 < m) (hn_pos : 0 < n) (h_mn : m < n) 
  (h_f_eq : f m = f n) (h_max_f : ∀ x : ℝ, m^2 ≤ x ∧ x ≤ n → f x ≤ 2) :
  m + n = 5 / 2 :=
sorry

end sum_m_n_l60_60905


namespace problem_l60_60044

-- Define first terms
def a_1 : ℕ := 12
def b_1 : ℕ := 48

-- Define the 100th term condition
def a_100 (d_a : ℚ) := 12 + 99 * d_a
def b_100 (d_b : ℚ) := 48 + 99 * d_b

-- Condition that the sum of the 100th terms is 200
def condition (d_a d_b : ℚ) := a_100 d_a + b_100 d_b = 200

-- Define the value of the sum of the first 100 terms
def sequence_sum (d_a d_b : ℚ) := 100 * 60 + (140 / 99) * ((99 * 100) / 2)

-- The proof theorem
theorem problem : ∀ d_a d_b : ℚ, condition d_a d_b → sequence_sum d_a d_b = 13000 :=
by
  intros d_a d_b h_cond
  sorry

end problem_l60_60044


namespace solve_linear_eq_l60_60692

theorem solve_linear_eq (x : ℝ) : (x + 1) / 3 = 0 → x = -1 := 
by 
  sorry

end solve_linear_eq_l60_60692


namespace probability_of_type_A_probability_of_different_type_l60_60200

def total_questions : ℕ := 6
def type_A_questions : ℕ := 4
def type_B_questions : ℕ := 2
def select_questions : ℕ := 2

def total_combinations := Nat.choose total_questions select_questions
def type_A_combinations := Nat.choose type_A_questions select_questions
def different_type_combinations := Nat.choose type_A_questions 1 * Nat.choose type_B_questions 1

theorem probability_of_type_A : (type_A_combinations : ℚ) / total_combinations = 2 / 5 := by
  sorry

theorem probability_of_different_type : (different_type_combinations : ℚ) / total_combinations = 8 / 15 := by
  sorry

end probability_of_type_A_probability_of_different_type_l60_60200


namespace direct_proportion_b_zero_l60_60475

theorem direct_proportion_b_zero (b : ℝ) (x y : ℝ) 
  (h : ∀ x, y = x + b → ∃ k, y = k * x) : b = 0 :=
sorry

end direct_proportion_b_zero_l60_60475


namespace number_of_sodas_bought_l60_60256

theorem number_of_sodas_bought
  (sandwich_cost : ℝ)
  (num_sandwiches : ℝ)
  (soda_cost : ℝ)
  (total_cost : ℝ)
  (h1 : sandwich_cost = 3.49)
  (h2 : num_sandwiches = 2)
  (h3 : soda_cost = 0.87)
  (h4 : total_cost = 10.46) :
  (total_cost - num_sandwiches * sandwich_cost) / soda_cost = 4 := 
sorry

end number_of_sodas_bought_l60_60256


namespace simple_interest_years_l60_60061

theorem simple_interest_years (P : ℝ) (R : ℝ) (T : ℝ) 
  (h1 : P = 300) 
  (h2 : (P * ((R + 6) / 100) * T) = (P * (R / 100) * T + 90)) : 
  T = 5 := 
by 
  -- Necessary proof steps go here
  sorry

end simple_interest_years_l60_60061


namespace cos_B_equals_3_over_4_l60_60538

variables {A B C : ℝ} {a b c R : ℝ} (h₁ : b * Real.sin B - a * Real.sin A = (1/2) * a * Real.sin C)
  (h₂ :  2 * R ^ 2 * Real.sin B * (1 - Real.cos (2 * A)) = (1 / 2) * a * b * Real.sin C)

theorem cos_B_equals_3_over_4 : Real.cos B = 3 / 4 := by
  sorry

end cos_B_equals_3_over_4_l60_60538


namespace sector_arc_length_l60_60025

theorem sector_arc_length (r : ℝ) (θ : ℝ) (L : ℝ) (h₁ : r = 1) (h₂ : θ = 60 * π / 180) : L = π / 3 :=
by
  sorry

end sector_arc_length_l60_60025


namespace sufficient_but_not_necessary_l60_60687

variable (x : ℝ)

def condition_p := -1 ≤ x ∧ x ≤ 1
def condition_q := x ≥ -2

theorem sufficient_but_not_necessary :
  (condition_p x → condition_q x) ∧ ¬(condition_q x → condition_p x) :=
by 
  sorry

end sufficient_but_not_necessary_l60_60687


namespace polynomial_non_negative_l60_60862

theorem polynomial_non_negative (a : ℝ) : a^2 * (a^2 - 1) - a^2 + 1 ≥ 0 := by
  -- we would include the proof steps here
  sorry

end polynomial_non_negative_l60_60862


namespace compare_neg_one_neg_sqrt_two_l60_60302

theorem compare_neg_one_neg_sqrt_two : -1 > -Real.sqrt 2 :=
  by
    sorry

end compare_neg_one_neg_sqrt_two_l60_60302


namespace functional_eq_f800_l60_60850

theorem functional_eq_f800
  (f : ℝ → ℝ)
  (h1 : ∀ x y : ℝ, 0 < x → 0 < y → f (x * y) = f x / y)
  (h2 : f 1000 = 6)
  : f 800 = 7.5 := by
  sorry

end functional_eq_f800_l60_60850


namespace smallest_integer_relative_prime_to_2310_l60_60484

theorem smallest_integer_relative_prime_to_2310 (n : ℕ) : (2 < n → n ≤ 13 → ¬ (n ∣ 2310)) → n = 13 := by
  sorry

end smallest_integer_relative_prime_to_2310_l60_60484


namespace Clara_sells_third_type_boxes_l60_60519

variable (total_cookies boxes_first boxes_second boxes_third : ℕ)
variable (cookies_per_first cookies_per_second cookies_per_third : ℕ)

theorem Clara_sells_third_type_boxes (h1 : cookies_per_first = 12)
                                    (h2 : boxes_first = 50)
                                    (h3 : cookies_per_second = 20)
                                    (h4 : boxes_second = 80)
                                    (h5 : cookies_per_third = 16)
                                    (h6 : total_cookies = 3320) :
                                    boxes_third = 70 :=
by
  sorry

end Clara_sells_third_type_boxes_l60_60519


namespace parallel_lines_implies_m_no_perpendicular_lines_solution_l60_60702

noncomputable def parallel_slopes (m : ℝ) : Prop :=
  let y₁ := -m
  let y₂ := -2 / m
  y₁ = y₂

noncomputable def perpendicular_slopes (m : ℝ) : Prop :=
  let y₁ := -m
  let y₂ := -2 / m
  y₁ * y₂ = -1

theorem parallel_lines_implies_m (m : ℝ) : parallel_slopes m ↔ m = Real.sqrt 2 ∨ m = -Real.sqrt 2 :=
by
  sorry

theorem no_perpendicular_lines_solution (m : ℝ) : perpendicular_slopes m → false :=
by
  sorry

end parallel_lines_implies_m_no_perpendicular_lines_solution_l60_60702


namespace infinite_series_converges_l60_60987

open BigOperators

noncomputable def problem : ℝ :=
  ∑' n : ℕ, if n > 0 then (3 * n - 2) / (n * (n + 1) * (n + 3)) else 0

theorem infinite_series_converges : problem = 61 / 24 :=
sorry

end infinite_series_converges_l60_60987


namespace area_T_is_34_l60_60344

/-- Define the dimensions of the large rectangle -/
def width_rect : ℕ := 10
def height_rect : ℕ := 4
/-- Define the dimensions of the removed section -/
def width_removed : ℕ := 6
def height_removed : ℕ := 1

/-- Calculate the area of the large rectangle -/
def area_rect : ℕ := width_rect * height_rect

/-- Calculate the area of the removed section -/
def area_removed : ℕ := width_removed * height_removed

/-- Calculate the area of the "T" shape -/
def area_T : ℕ := area_rect - area_removed

/-- To prove that the area of the T-shape is 34 square units -/
theorem area_T_is_34 : area_T = 34 := 
by {
  sorry
}

end area_T_is_34_l60_60344


namespace g_x_minus_3_l60_60620

def g (x : ℝ) : ℝ := x^2

theorem g_x_minus_3 (x : ℝ) : g (x - 3) = x^2 - 6 * x + 9 :=
by
  -- This is where the proof would go
  sorry

end g_x_minus_3_l60_60620


namespace measure_angle_C_l60_60238

theorem measure_angle_C (A B C : ℝ) (h1 : A = 60) (h2 : B = 60) (h3 : C = 60 - 10) (sum_angles : A + B + C = 180) : C = 53.33 :=
by
  sorry

end measure_angle_C_l60_60238


namespace number_of_red_pencils_l60_60338

theorem number_of_red_pencils (B R G : ℕ) (h1 : B + R + G = 20) (h2 : B = 6 * G) (h3 : R < B) : R = 6 :=
by
  sorry

end number_of_red_pencils_l60_60338


namespace avg_distance_is_600_l60_60275

-- Assuming Mickey runs half as many times as Johnny
def num_laps_johnny := 4
def num_laps_mickey := num_laps_johnny / 2
def lap_distance := 200

-- Calculating distances
def distance_johnny := num_laps_johnny * lap_distance
def distance_mickey := num_laps_mickey * lap_distance

-- Total distance run by both Mickey and Johnny
def total_distance := distance_johnny + distance_mickey

-- Average distance run by Johnny and Mickey
def avg_distance := total_distance / 2

-- Prove that the average distance run by Johnny and Mickey is 600 meters
theorem avg_distance_is_600 : avg_distance = 600 :=
by
  sorry

end avg_distance_is_600_l60_60275


namespace bridge_length_is_correct_l60_60249

def speed_km_hr : ℝ := 45
def train_length_m : ℝ := 120
def crossing_time_s : ℝ := 30

noncomputable def speed_m_s : ℝ := speed_km_hr * 1000 / 3600
noncomputable def total_distance_m : ℝ := speed_m_s * crossing_time_s
noncomputable def bridge_length_m : ℝ := total_distance_m - train_length_m

theorem bridge_length_is_correct : bridge_length_m = 255 := by
  sorry

end bridge_length_is_correct_l60_60249


namespace possible_values_of_x_l60_60976

theorem possible_values_of_x (x : ℕ) (h1 : ∃ k : ℕ, k * k = 8 - x) (h2 : 1 ≤ x ∧ x ≤ 8) :
  x = 4 ∨ x = 7 ∨ x = 8 :=
by
  sorry

end possible_values_of_x_l60_60976


namespace combined_avg_score_l60_60426

theorem combined_avg_score (x : ℕ) : 
  let avgA := 65
  let avgB := 90 
  let avgC := 77 
  let ratioA := 4 
  let ratioB := 6 
  let ratioC := 5 
  let total_students := 15 * x 
  let total_score := (ratioA * avgA + ratioB * avgB + ratioC * avgC) * x
  (total_score / total_students) = 79 := 
by
  sorry

end combined_avg_score_l60_60426


namespace last_two_videos_length_l60_60027

noncomputable def ad1 : ℕ := 45
noncomputable def ad2 : ℕ := 30
noncomputable def pause1 : ℕ := 45
noncomputable def pause2 : ℕ := 30
noncomputable def video1 : ℕ := 120
noncomputable def video2 : ℕ := 270
noncomputable def total_time : ℕ := 960

theorem last_two_videos_length : 
    ∃ v : ℕ, 
    v = 210 ∧ 
    total_time = ad1 + ad2 + video1 + video2 + pause1 + pause2 + 2 * v :=
by
  sorry

end last_two_videos_length_l60_60027


namespace find_number_l60_60392

theorem find_number (x : ℤ) (h : 72516 * x = 724797420) : x = 10001 :=
by
  sorry

end find_number_l60_60392


namespace sufficient_conditions_for_sum_positive_l60_60125

variable {a b : ℝ}

theorem sufficient_conditions_for_sum_positive (h₃ : a + b > 2) (h₄ : a > 0 ∧ b > 0) : a + b > 0 :=
by {
  sorry
}

end sufficient_conditions_for_sum_positive_l60_60125


namespace negation_of_existential_statement_l60_60946

theorem negation_of_existential_statement {f : ℝ → ℝ} :
  (¬ ∃ x₀ : ℝ, f x₀ < 0) ↔ (∀ x : ℝ, f x ≥ 0) :=
by
  sorry

end negation_of_existential_statement_l60_60946


namespace count_teams_of_6_l60_60708

theorem count_teams_of_6 
  (students : Fin 12 → Type)
  (played_together_once : ∀ (s : Finset (Fin 12)) (h : s.card = 5), ∃! t : Finset (Fin 12), t.card = 6 ∧ s ⊆ t) :
  (∃ team_count : ℕ, team_count = 132) :=
by
  -- Proof omitted
  sorry

end count_teams_of_6_l60_60708


namespace least_number_to_subtract_l60_60064

theorem least_number_to_subtract (x : ℕ) (h : x = 7538 % 14) : (7538 - x) % 14 = 0 :=
by
  -- Proof goes here
  sorry

end least_number_to_subtract_l60_60064


namespace math_problem_l60_60175

theorem math_problem {x y : ℕ} (h1 : 1059 % x = y) (h2 : 1417 % x = y) (h3 : 2312 % x = y) : x - y = 15 := by
  sorry

end math_problem_l60_60175


namespace solution_set_of_inequality_l60_60361

theorem solution_set_of_inequality :
  {x : ℝ | 2 ≥ 1 / (x - 1)} = {x : ℝ | x < 1} ∪ {x : ℝ | x ≥ 3 / 2} :=
by
  sorry

end solution_set_of_inequality_l60_60361


namespace range_of_a_l60_60318

theorem range_of_a (a : ℝ) : 
  (∃ x y : ℝ, x^2 + y^2 + 2 * x - 4 * y + a = 0) → a < 5 := 
by sorry

end range_of_a_l60_60318


namespace min_questions_to_find_phone_number_min_questions_to_find_phone_number_is_17_l60_60909

theorem min_questions_to_find_phone_number : 
  ∃ n : ℕ, ∀ (N : ℕ), (N = 100000 → 2 ^ n ≥ N) ∧ (2 ^ (n - 1) < N) := sorry

-- In simpler form, since log_2(100000) ≈ 16.60965, we have:
theorem min_questions_to_find_phone_number_is_17 : 
  ∀ (N : ℕ), (N = 100000 → 17 = Nat.ceil (Real.logb 2 100000)) := sorry

end min_questions_to_find_phone_number_min_questions_to_find_phone_number_is_17_l60_60909


namespace num_people_at_gathering_l60_60098

noncomputable def total_people_at_gathering : ℕ :=
  let wine_soda := 12
  let wine_juice := 10
  let wine_coffee := 6
  let wine_tea := 4
  let soda_juice := 8
  let soda_coffee := 5
  let soda_tea := 3
  let juice_coffee := 7
  let juice_tea := 2
  let coffee_tea := 4
  let wine_soda_juice := 3
  let wine_soda_coffee := 1
  let wine_soda_tea := 2
  let wine_juice_coffee := 3
  let wine_juice_tea := 1
  let wine_coffee_tea := 2
  let soda_juice_coffee := 3
  let soda_juice_tea := 1
  let soda_coffee_tea := 2
  let juice_coffee_tea := 3
  let all_five := 1
  wine_soda + wine_juice + wine_coffee + wine_tea +
  soda_juice + soda_coffee + soda_tea + juice_coffee +
  juice_tea + coffee_tea + wine_soda_juice + wine_soda_coffee +
  wine_soda_tea + wine_juice_coffee + wine_juice_tea +
  wine_coffee_tea + soda_juice_coffee + soda_juice_tea +
  soda_coffee_tea + juice_coffee_tea + all_five

theorem num_people_at_gathering : total_people_at_gathering = 89 := by
  sorry

end num_people_at_gathering_l60_60098


namespace jamshid_taimour_painting_problem_l60_60458

/-- Jamshid and Taimour Painting Problem -/
theorem jamshid_taimour_painting_problem (T : ℝ) (h1 : T > 0)
  (h2 : 1 / T + 2 / T = 1 / 5) : T = 15 :=
by
  -- solving the theorem
  sorry

end jamshid_taimour_painting_problem_l60_60458


namespace find_angle_A_and_triangle_perimeter_l60_60204

-- Declare the main theorem using the provided conditions and the desired results
theorem find_angle_A_and_triangle_perimeter
  (a b c : ℝ) (A B : ℝ)
  (h1 : 0 < A ∧ A < Real.pi)
  (h2 : (Real.sqrt 3) * b * c * (Real.cos A) = a * (Real.sin B))
  (h3 : a = Real.sqrt 2)
  (h4 : (c / a) = (Real.sin A / Real.sin B)) :
  (A = Real.pi / 3) ∧ (a + b + c = 3 * Real.sqrt 2) :=
  sorry -- Proof is left as an exercise

end find_angle_A_and_triangle_perimeter_l60_60204


namespace methane_reaction_l60_60609

noncomputable def methane_reacts_with_chlorine
  (moles_CH₄ : ℕ)
  (moles_Cl₂ : ℕ)
  (moles_CCl₄ : ℕ)
  (moles_HCl_produced : ℕ) : Prop :=
  moles_CH₄ = 3 ∧ 
  moles_Cl₂ = 12 ∧ 
  moles_CCl₄ = 3 ∧ 
  moles_HCl_produced = 12

theorem methane_reaction : 
  methane_reacts_with_chlorine 3 12 3 12 :=
by sorry

end methane_reaction_l60_60609


namespace smallest_E_of_positive_reals_l60_60115

noncomputable def E (a b c : ℝ) : ℝ :=
  (a^3) / (1 - a^2) + (b^3) / (1 - b^2) + (c^3) / (1 - c^2)

theorem smallest_E_of_positive_reals (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_sum : a + b + c = 1) : 
  E a b c = 1 / 8 := 
sorry

end smallest_E_of_positive_reals_l60_60115


namespace solve_for_x_l60_60545

theorem solve_for_x (x : ℝ) 
  (h : (2 / (x + 3)) + (3 * x / (x + 3)) - (5 / (x + 3)) = 2) : 
  x = 9 := 
by
  sorry

end solve_for_x_l60_60545


namespace exists_four_integers_multiple_1984_l60_60534

theorem exists_four_integers_multiple_1984 (a : Fin 97 → ℕ) (h_distinct : Function.Injective a) :
  ∃ i j k l : Fin 97, i ≠ j ∧ k ≠ l ∧ 1984 ∣ (a i - a j) * (a k - a l) :=
sorry

end exists_four_integers_multiple_1984_l60_60534


namespace modulus_of_z_l60_60384

open Complex

theorem modulus_of_z (z : ℂ) (h : z^2 = (3/4 : ℝ) - I) : abs z = Real.sqrt 5 / 2 := 
  sorry

end modulus_of_z_l60_60384


namespace sum_areas_of_square_and_rectangle_l60_60377

theorem sum_areas_of_square_and_rectangle (s w l : ℝ) 
  (h1 : s^2 + w * l = 130)
  (h2 : 4 * s - 2 * (w + l) = 20)
  (h3 : l = 2 * w) : 
  s^2 + 2 * w^2 = 118 :=
by
  -- Provide space for proof
  sorry

end sum_areas_of_square_and_rectangle_l60_60377


namespace roque_commute_time_l60_60589

theorem roque_commute_time :
  let walk_time := 2
  let bike_time := 1
  let walks_per_week := 3
  let bike_rides_per_week := 2
  let total_walk_time := 2 * walks_per_week * walk_time
  let total_bike_time := 2 * bike_rides_per_week * bike_time
  total_walk_time + total_bike_time = 16 :=
by sorry

end roque_commute_time_l60_60589


namespace total_distance_flash_runs_l60_60440

-- Define the problem with given conditions
theorem total_distance_flash_runs (v k d a : ℝ) (hk : k > 1) : 
  let t := d / (v * (k - 1))
  let distance_to_catch_ace := k * v * t
  let total_distance := distance_to_catch_ace + a
  total_distance = (k * d) / (k - 1) + a := 
by
  sorry

end total_distance_flash_runs_l60_60440


namespace conference_handshakes_l60_60650

theorem conference_handshakes (total_people : ℕ) (group1_people : ℕ) (group2_people : ℕ)
  (group1_knows_each_other : group1_people = 25)
  (group2_knows_no_one_in_group1 : group2_people = 15)
  (total_group : total_people = group1_people + group2_people)
  (total_handshakes : ℕ := group2_people * (group1_people + group2_people - 1) - group2_people * (group2_people - 1) / 2) :
  total_handshakes = 480 := by
  -- Placeholder for proof
  sorry

end conference_handshakes_l60_60650


namespace tan_beta_value_l60_60181

theorem tan_beta_value (α β : ℝ) (h1 : Real.tan α = 1 / 3) (h2 : Real.tan (α + β) = 1 / 2) : Real.tan β = 1 / 7 :=
by
  sorry

end tan_beta_value_l60_60181


namespace ratio_triangle_BFD_to_square_ABCE_l60_60775

-- Defining necessary components for the mathematical problem
def square_ABCE (x : ℝ) : ℝ := 16 * x^2
def triangle_BFD_area (x : ℝ) : ℝ := 7 * x^2

-- The theorem that needs to be proven, stating the ratio of the areas
theorem ratio_triangle_BFD_to_square_ABCE (x : ℝ) (hx : x > 0) :
  (triangle_BFD_area x) / (square_ABCE x) = 7 / 16 :=
by
  sorry

end ratio_triangle_BFD_to_square_ABCE_l60_60775


namespace jeans_and_shirts_l60_60636

-- Let's define the necessary variables and conditions.
variables (J S X : ℝ)

-- Given conditions
def condition1 := 3 * J + 2 * S = X
def condition2 := 2 * J + 3 * S = 61

-- Given the price of one shirt
def price_of_shirt := S = 9

-- The problem we need to prove
theorem jeans_and_shirts : condition1 J S X ∧ condition2 J S ∧ price_of_shirt S →
  X = 69 :=
by
  sorry

end jeans_and_shirts_l60_60636


namespace incorrect_membership_l60_60170

-- Let's define the sets involved.
def a : Set ℕ := {1}             -- singleton set {a}
def ab : Set (Set ℕ) := {{1}, {2}}  -- set {a, b}

-- Now, the proof statement.
theorem incorrect_membership : ¬ (a ∈ ab) := 
by { sorry }

end incorrect_membership_l60_60170


namespace range_of_a_l60_60304

-- Assuming all necessary imports and definitions are included

variable {R : Type} [LinearOrderedField R]

def satisfies_conditions (f : R → R) (a : R) : Prop :=
  (∀ x, f (1 + x) = f (1 - x)) ∧
  (∀ x y, 1 ≤ x → x < y → f x < f y) ∧
  (∀ x, (1/2 : R) ≤ x ∧ x ≤ 1 → f (a * x) < f (x - 1))

theorem range_of_a (f : R → R) (a : R) :
  satisfies_conditions f a → 0 < a ∧ a < 2 :=
by
  sorry

end range_of_a_l60_60304


namespace eulers_polyhedron_theorem_l60_60713

theorem eulers_polyhedron_theorem 
  (V E F t h : ℕ) (T H : ℕ) :
  (F = 30) →
  (t = 20) →
  (h = 10) →
  (T = 3) →
  (H = 2) →
  (E = (3 * t + 6 * h) / 2) →
  (V - E + F = 2) →
  100 * H + 10 * T + V = 262 :=
by
  intros F_eq t_eq h_eq T_eq H_eq E_eq euler_eq
  rw [F_eq, t_eq, h_eq, T_eq, H_eq, E_eq] at *
  sorry

end eulers_polyhedron_theorem_l60_60713


namespace smallest_common_multiple_of_8_and_6_l60_60381

theorem smallest_common_multiple_of_8_and_6 : ∃ n : ℕ, n > 0 ∧ (8 ∣ n) ∧ (6 ∣ n) ∧ ∀ m : ℕ, (m > 0 ∧ (8 ∣ m) ∧ (6 ∣ m)) → n ≤ m :=
by
  sorry

end smallest_common_multiple_of_8_and_6_l60_60381


namespace factor_correct_l60_60010

-- Define the polynomial p(x)
def p (x : ℤ) : ℤ := 6 * (x + 4) * (x + 7) * (x + 9) * (x + 11) - 5 * x^2

-- Define the potential factors of p(x)
def f1 (x : ℤ) : ℤ := 3 * x^2 + 93 * x
def f2 (x : ℤ) : ℤ := 2 * x^2 + 178 * x + 5432

theorem factor_correct : ∀ x : ℤ, p x = f1 x * f2 x := by
  sorry

end factor_correct_l60_60010


namespace fill_time_eight_faucets_l60_60886

theorem fill_time_eight_faucets (r : ℝ) (h1 : 4 * r * 8 = 150) :
  8 * r * (50 / (8 * r)) * 60 = 80 := by
  sorry

end fill_time_eight_faucets_l60_60886


namespace emma_investment_l60_60242

-- Define the basic problem parameters
def P : ℝ := 2500
def r : ℝ := 0.04
def n : ℕ := 21
def expected_amount : ℝ := 6101.50

-- Define the compound interest formula result
def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r) ^ n

-- State the theorem
theorem emma_investment : 
  compound_interest P r n = expected_amount := 
  sorry

end emma_investment_l60_60242


namespace alcohol_water_ratio_l60_60309

theorem alcohol_water_ratio
  (V p q : ℝ)
  (hV : V > 0)
  (hp : p > 0)
  (hq : q > 0) :
  let total_alcohol := 3 * V * (p / (p + 1)) + V * (q / (q + 1))
  let total_water := 3 * V * (1 / (p + 1)) + V * (1 / (q + 1))
  total_alcohol / total_water = (3 * p * (q + 1) + q * (p + 1)) / (3 * (q + 1) + (p + 1)) :=
sorry

end alcohol_water_ratio_l60_60309


namespace Joey_downhill_speed_l60_60834

theorem Joey_downhill_speed
  (Route_length : ℝ) (Time_uphill : ℝ) (Speed_uphill : ℝ) (Overall_average_speed : ℝ) (Extra_time_due_to_rain : ℝ) :
  Route_length = 5 →
  Time_uphill = 1.25 →
  Speed_uphill = 4 →
  Overall_average_speed = 6 →
  Extra_time_due_to_rain = 0.25 →
  ((2 * Route_length) / Overall_average_speed - Time_uphill - Extra_time_due_to_rain) * (Route_length / (2 * Route_length / Overall_average_speed - Time_uphill - Extra_time_due_to_rain)) = 30 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end Joey_downhill_speed_l60_60834


namespace no_real_m_perpendicular_l60_60506

theorem no_real_m_perpendicular (m : ℝ) : 
  ¬ ∃ m, ((m - 2) * m = -3) := 
sorry

end no_real_m_perpendicular_l60_60506


namespace sqrt_expression_value_l60_60649

variable (a b : ℝ) 

theorem sqrt_expression_value (ha : a ≠ 0) (hb : b ≠ 0) (ha_neg : a < 0) :
  Real.sqrt (-a^3) * Real.sqrt ((-b)^4) = -a * |b| * Real.sqrt (-a) := by
  sorry

end sqrt_expression_value_l60_60649


namespace alicia_satisfaction_l60_60992

theorem alicia_satisfaction (t : ℚ) (h_sat : t * (12 - t) = (4 - t) * (2 * t + 2)) : t = 2 :=
by
  sorry

end alicia_satisfaction_l60_60992


namespace sunlovers_happy_days_l60_60736

open Nat

theorem sunlovers_happy_days (D R : ℕ) : 
  (D^2 + 4) * (R^2 + 4) - 2 * D * (R^2 + 4) - 2 * R * (D^2 + 4) ≥ 0 := 
by 
  sorry

end sunlovers_happy_days_l60_60736


namespace angles_arith_prog_tangent_tangent_parallel_euler_line_l60_60753

-- Define a non-equilateral triangle with angles in arithmetic progression
structure Triangle :=
  (A B C : ℝ) -- Angles in a non-equilateral triangle
  (non_equilateral : A ≠ B ∨ B ≠ C ∨ A ≠ C)
  (angles_arith_progression : (2 * B = A + C))

-- Additional geometry concepts will be assumptions as their definition 
-- would involve extensive axiomatic setups

-- The main theorem to state the equivalence
theorem angles_arith_prog_tangent_tangent_parallel_euler_line (Δ : Triangle)
  (common_tangent_parallel_euler : sorry) : 
  ((Δ.A = 60) ∨ (Δ.B = 60) ∨ (Δ.C = 60)) :=
sorry

end angles_arith_prog_tangent_tangent_parallel_euler_line_l60_60753


namespace find_current_listens_l60_60234

theorem find_current_listens (x : ℕ) (h : 15 * x = 900000) : x = 60000 :=
by
  sorry

end find_current_listens_l60_60234


namespace fraction_equiv_l60_60894

theorem fraction_equiv (x y : ℚ) (h : (5/6) * 192 = (x/y) * 192 + 100) : x/y = 5/16 :=
sorry

end fraction_equiv_l60_60894


namespace length_of_train_l60_60555

noncomputable def train_length : ℕ := 1200

theorem length_of_train 
  (L : ℝ) 
  (speed_km_per_hr : ℝ) 
  (time_min : ℕ) 
  (speed_m_per_s : ℝ) 
  (time_sec : ℕ) 
  (distance : ℝ) 
  (cond1 : L = L)
  (cond2 : speed_km_per_hr = 144) 
  (cond3 : time_min = 1)
  (cond4 : speed_m_per_s = speed_km_per_hr * 1000 / 3600)
  (cond5 : time_sec = time_min * 60)
  (cond6 : distance = speed_m_per_s * time_sec)
  (cond7 : 2 * L = distance)
  : L = train_length := 
sorry

end length_of_train_l60_60555


namespace probability_calculation_l60_60577

noncomputable def probability_of_event_A : ℚ := 
  let total_ways := 35 
  let favorable_ways := 6 
  favorable_ways / total_ways

theorem probability_calculation (A_team B_team : Type) [Fintype A_team] [Fintype B_team] [DecidableEq A_team] [DecidableEq B_team] :
  let total_players := 7 
  let selected_players := 4 
  let seeded_A := 2 
  let nonseeded_A := 1 
  let seeded_B := 2 
  let nonseeded_B := 2 
  let event_total_ways := Nat.choose total_players selected_players 
  let event_A_ways := Nat.choose seeded_A 2 * Nat.choose nonseeded_A 2 + Nat.choose seeded_B 2 * Nat.choose nonseeded_B 2 
  probability_of_event_A = 6 / 35 := 
sorry

end probability_calculation_l60_60577


namespace factorization_4x2_minus_144_l60_60798

theorem factorization_4x2_minus_144 (x : ℝ) : 4 * x^2 - 144 = 4 * (x - 6) * (x + 6) := 
  sorry

end factorization_4x2_minus_144_l60_60798


namespace calc_expression_l60_60273

theorem calc_expression : 2^1 + 1^0 - 3^2 = -6 := by
  sorry

end calc_expression_l60_60273


namespace new_average_after_multiplication_l60_60908

theorem new_average_after_multiplication
  (n : ℕ) (a : ℕ) (m : ℕ)
  (h1 : n = 7)
  (h2 : a = 25)
  (h3 : m = 5):
  (n * a * m / n) = 125 :=
by
  sorry


end new_average_after_multiplication_l60_60908


namespace no_solution_inequality_l60_60661

theorem no_solution_inequality (a : ℝ) : (¬ ∃ x : ℝ, x > 1 ∧ x < a - 1) → a ≤ 2 :=
by
  sorry

end no_solution_inequality_l60_60661


namespace top_card_is_club_probability_l60_60751

-- Definitions based on the conditions
def deck_size := 52
def suit_count := 4
def cards_per_suit := deck_size / suit_count

-- The question we want to prove
theorem top_card_is_club_probability :
  (13 : ℝ) / (52 : ℝ) = 1 / 4 :=
by 
  sorry

end top_card_is_club_probability_l60_60751


namespace clare_remaining_money_l60_60224

-- Definitions based on conditions
def clare_initial_money : ℕ := 47
def bread_quantity : ℕ := 4
def milk_quantity : ℕ := 2
def bread_cost : ℕ := 2
def milk_cost : ℕ := 2

-- The goal is to prove that Clare has $35 left after her purchases.
theorem clare_remaining_money : 
  clare_initial_money - (bread_quantity * bread_cost + milk_quantity * milk_cost) = 35 := 
sorry

end clare_remaining_money_l60_60224


namespace sequence_nth_term_16_l60_60759

theorem sequence_nth_term_16 (n : ℕ) (sqrt2 : ℝ) (h_sqrt2 : sqrt2 = Real.sqrt 2) (a_n : ℕ → ℝ) 
  (h_seq : ∀ n, a_n n = sqrt2 ^ (n - 1)) :
  a_n n = 16 → n = 9 := by
  sorry

end sequence_nth_term_16_l60_60759


namespace value_of_S_l60_60030

def pseudocode_value : ℕ := 1
def increment (S I : ℕ) : ℕ := S + I

def loop_steps : ℕ :=
  let S := pseudocode_value
  let S := increment S 1
  let S := increment S 3
  let S := increment S 5
  let S := increment S 7
  S

theorem value_of_S : loop_steps = 17 :=
  by sorry

end value_of_S_l60_60030


namespace b_10_eq_64_l60_60744

noncomputable def a (n : ℕ) : ℕ := -- Definition of the sequence a_n
  sorry

noncomputable def b (n : ℕ) : ℕ := -- Definition of the sequence b_n
  a n + a (n + 1)

theorem b_10_eq_64 (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, a n * a (n + 1) = 2^n) :
  b 10 = 64 :=
sorry

end b_10_eq_64_l60_60744


namespace correct_factorization_A_l60_60851

-- Define the polynomial expressions
def expression_A : Prop :=
  (x : ℝ) → x^2 - x - 6 = (x + 2) * (x - 3)

def expression_B : Prop :=
  (x : ℝ) → x^2 - 1 = x * (x - 1 / x)

def expression_C : Prop :=
  (x y : ℝ) → 7 * x^2 * y^5 = x * y * 7 * x * y^4

def expression_D : Prop :=
  (x : ℝ) → x^2 + 4 * x + 4 = x * (x + 4) + 4

-- The correct factorization from left to right
theorem correct_factorization_A : expression_A := 
by 
  -- Proof omitted
  sorry

end correct_factorization_A_l60_60851


namespace sufficient_but_not_necessary_not_necessary_condition_l60_60059

theorem sufficient_but_not_necessary 
  (α : ℝ) (h : Real.sin α = Real.cos α) :
  Real.cos (2 * α) = 0 :=
by sorry

theorem not_necessary_condition 
  (α : ℝ) (h : Real.cos (2 * α) = 0) :
  ∃ β : ℝ, Real.sin β ≠ Real.cos β :=
by sorry

end sufficient_but_not_necessary_not_necessary_condition_l60_60059


namespace corset_total_cost_l60_60699

def purple_bead_cost : ℝ := 50 * 20 * 0.12
def blue_bead_cost : ℝ := 40 * 18 * 0.10
def gold_bead_cost : ℝ := 80 * 0.08
def red_bead_cost : ℝ := 30 * 15 * 0.09
def silver_bead_cost : ℝ := 100 * 0.07

def total_cost : ℝ := purple_bead_cost + blue_bead_cost + gold_bead_cost + red_bead_cost + silver_bead_cost

theorem corset_total_cost : total_cost = 245.90 := by
  sorry

end corset_total_cost_l60_60699


namespace number_of_packets_l60_60093

def ounces_in_packet : ℕ := 16 * 16 + 4
def ounces_in_ton : ℕ := 2500 * 16
def gunny_bag_capacity_in_ounces : ℕ := 13 * ounces_in_ton

theorem number_of_packets : gunny_bag_capacity_in_ounces / ounces_in_packet = 2000 :=
by
  sorry

end number_of_packets_l60_60093


namespace interest_rate_difference_l60_60726

theorem interest_rate_difference (P T : ℝ) (R1 R2 : ℝ) (I_diff : ℝ) (hP : P = 2100) 
  (hT : T = 3) (hI : I_diff = 63) :
  R2 - R1 = 0.01 :=
by
  sorry

end interest_rate_difference_l60_60726


namespace radius_of_roots_circle_l60_60000

theorem radius_of_roots_circle (z : ℂ) (hz : (z - 2)^6 = 64 * z^6) : ∃ r : ℝ, r = 2 / 3 :=
by
  sorry

end radius_of_roots_circle_l60_60000


namespace min_value_of_a_sq_plus_b_sq_l60_60614

theorem min_value_of_a_sq_plus_b_sq {a b t : ℝ} (h : 2 * a + 3 * b = t) :
  ∃ a b : ℝ, (2 * a + 3 * b = t) ∧ (a^2 + b^2 = (13 * t^2) / 169) :=
by
  sorry

end min_value_of_a_sq_plus_b_sq_l60_60614


namespace rectangle_in_triangle_area_l60_60937

theorem rectangle_in_triangle_area
  (PR : ℝ) (h_PR : PR = 15)
  (Q_altitude : ℝ) (h_Q_altitude : Q_altitude = 9)
  (x : ℝ)
  (AD : ℝ) (h_AD : AD = x)
  (AB : ℝ) (h_AB : AB = x / 3) :
  (AB * AD = 675 / 64) :=
by
  sorry

end rectangle_in_triangle_area_l60_60937


namespace expenditure_ratio_l60_60419

theorem expenditure_ratio 
  (I1 : ℝ) (I2 : ℝ) (E1 : ℝ) (E2 : ℝ) (S1 : ℝ) (S2 : ℝ)
  (h1 : I1 = 3500)
  (h2 : I2 = (4 / 5) * I1)
  (h3 : S1 = I1 - E1)
  (h4 : S2 = I2 - E2)
  (h5 : S1 = 1400)
  (h6 : S2 = 1400) : 
  E1 / E2 = 3 / 2 :=
by
  -- Steps of the proof will go here
  sorry

end expenditure_ratio_l60_60419


namespace last_three_digits_2005_pow_2005_l60_60127

def last_three_digits (n : ℕ) : ℕ :=
  n % 1000

theorem last_three_digits_2005_pow_2005 :
  last_three_digits (2005 ^ 2005) = 125 :=
sorry

end last_three_digits_2005_pow_2005_l60_60127


namespace problem_seven_integers_l60_60060

theorem problem_seven_integers (a b c d e f g : ℕ) 
  (h1 : b = a + 1) 
  (h2 : c = b + 1) 
  (h3 : d = c + 1) 
  (h4 : e = d + 1) 
  (h5 : f = e + 1) 
  (h6 : g = f + 1) 
  (h_sum : a + b + c + d + e + f + g = 2017) : 
  a = 286 ∨ g = 286 :=
sorry

end problem_seven_integers_l60_60060


namespace c_amount_correct_b_share_correct_l60_60747

-- Conditions
def total_sum : ℝ := 246    -- Total sum of money
def c_share : ℝ := 48      -- C's share in Rs
def c_per_rs : ℝ := 0.40   -- C's amount per Rs

-- Expressing the given condition c_share = total sum * c_per_rs
theorem c_amount_correct : c_share = total_sum * c_per_rs := 
  by
  -- Substitute that can be more elaboration of the calculations done
  sorry

-- Additional condition for the total per Rs distribution
axiom a_b_c_total : ∀ (a b : ℝ), a + b + c_per_rs = 1

-- Proving B's share per Rs is approximately 0.4049
theorem b_share_correct : ∃ a b : ℝ, c_share = 246 * 0.40 ∧ a + b + 0.40 = 1 ∧ b = 1 - (48 / 246) - 0.40 := 
  by
  -- Substitute that can be elaboration of the proof arguments done in the translated form
  sorry

end c_amount_correct_b_share_correct_l60_60747


namespace area_of_wrapping_paper_l60_60969

theorem area_of_wrapping_paper (l w h: ℝ) (l_pos: 0 < l) (w_pos: 0 < w) (h_pos: 0 < h) :
  ∃ s: ℝ, s = l + w ∧ s^2 = (l + w)^2 :=
by 
  sorry

end area_of_wrapping_paper_l60_60969


namespace three_mathematicians_same_language_l60_60998

theorem three_mathematicians_same_language
  (M : Fin 9 → Finset string)
  (h1 : ∀ i j k : Fin 9, ∃ lang, i ≠ j → i ≠ k → j ≠ k → lang ∈ M i ∧ lang ∈ M j)
  (h2 : ∀ i : Fin 9, (M i).card ≤ 3)
  : ∃ lang ∈ ⋃ i, M i, ∃ (A B C : Fin 9), A ≠ B → A ≠ C → B ≠ C → lang ∈ M A ∧ lang ∈ M B ∧ lang ∈ M C :=
sorry

end three_mathematicians_same_language_l60_60998


namespace circle_division_parts_l60_60103

-- Define the number of parts a circle is divided into by the chords.
noncomputable def numberOfParts (n : ℕ) : ℚ :=
  (n^4 - 6*n^3 + 23*n^2 - 18*n + 24) / 24

-- Prove that the number of parts is given by the defined function.
theorem circle_division_parts (n : ℕ) : numberOfParts n = (n^4 - 6*n^3 + 23*n^2 - 18*n + 24) / 24 := by
  sorry

end circle_division_parts_l60_60103


namespace angela_insects_l60_60848

theorem angela_insects (A J D : ℕ) (h1 : A = J / 2) (h2 : J = 5 * D) (h3 : D = 30) : A = 75 :=
by
  sorry

end angela_insects_l60_60848


namespace greatest_value_l60_60438

theorem greatest_value (x : ℝ) : -x^2 + 9 * x - 18 ≥ 0 → x ≤ 6 :=
by
  sorry

end greatest_value_l60_60438


namespace spherical_to_rectangular_conversion_l60_60730

theorem spherical_to_rectangular_conversion :
  ∃ x y z : ℝ, 
    x = -Real.sqrt 2 ∧ 
    y = 0 ∧ 
    z = Real.sqrt 2 ∧ 
    (∃ rho theta phi : ℝ, 
      rho = 2 ∧
      theta = π ∧
      phi = π/4 ∧
      x = rho * Real.sin phi * Real.cos theta ∧
      y = rho * Real.sin phi * Real.sin theta ∧
      z = rho * Real.cos phi) :=
by
  sorry

end spherical_to_rectangular_conversion_l60_60730


namespace find_roots_of_star_eq_l60_60953

def star (a b : ℝ) : ℝ := a^2 - b^2

theorem find_roots_of_star_eq :
  (star (star 2 3) x = 9) ↔ (x = 4 ∨ x = -4) :=
by
  sorry

end find_roots_of_star_eq_l60_60953


namespace holiday_not_on_22nd_l60_60223

def isThirdWednesday (d : ℕ) : Prop :=
  d = 15 ∨ d = 16 ∨ d = 17 ∨ d = 18 ∨ d = 19 ∨ d = 20 ∨ d = 21

theorem holiday_not_on_22nd :
  ¬ isThirdWednesday 22 :=
by
  intro h
  cases h
  repeat { contradiction }

end holiday_not_on_22nd_l60_60223


namespace ellipse_foci_y_axis_l60_60159

-- Given the equation of the ellipse x^2 + k * y^2 = 2 with foci on the y-axis,
-- prove that the range of k such that the ellipse is oriented with foci on the y-axis is (0, 1).
theorem ellipse_foci_y_axis (k : ℝ) (h1 : 0 < k) (h2 : k < 1) : 
  ∃ (a b : ℝ), a^2 + b^2 = 2 ∧ a > 0 ∧ b > 0 ∧ b / a = k ∧ x^2 + k * y^2 = 2 :=
sorry

end ellipse_foci_y_axis_l60_60159


namespace count_positive_integers_l60_60092

theorem count_positive_integers (x : ℤ) : 
  (25 < x^2 + 6 * x + 8) → (x^2 + 6 * x + 8 < 50) → (x > 0) → (x = 3 ∨ x = 4) :=
by sorry

end count_positive_integers_l60_60092


namespace vertex_of_parabola_l60_60632

def f (x : ℝ) : ℝ := 2 - (2*x + 1)^2

theorem vertex_of_parabola :
  (∀ x : ℝ, f x ≤ 2) ∧ (f (-1/2) = 2) :=
by
  sorry

end vertex_of_parabola_l60_60632


namespace speed_of_j_l60_60857

theorem speed_of_j (j p : ℝ) 
  (h_faster : j > p)
  (h_distance_j : 24 / j = 24 / j)
  (h_distance_p : 24 / p = 24 / p)
  (h_sum_speeds : j + p = 7)
  (h_sum_times : 24 / j + 24 / p = 14) : j = 4 := 
sorry

end speed_of_j_l60_60857


namespace John_paid_total_l60_60336

def vet_cost : ℝ := 400
def num_appointments : ℕ := 3
def insurance_cost : ℝ := 100
def coverage_rate : ℝ := 0.8

def discount : ℝ := vet_cost * coverage_rate
def discounted_visits : ℕ := num_appointments - 1
def discounted_cost : ℝ := vet_cost - discount
def total_discounted_cost : ℝ := discounted_visits * discounted_cost
def J_total : ℝ := vet_cost + total_discounted_cost + insurance_cost

theorem John_paid_total : J_total = 660 := by
  sorry

end John_paid_total_l60_60336


namespace min_value_of_quadratic_l60_60222

-- Define the given quadratic function
def quadratic (x : ℝ) : ℝ := 3 * x^2 + 8 * x + 15

-- Define the assertion that the minimum value of the quadratic function is 29/3
theorem min_value_of_quadratic : ∃ x : ℝ, quadratic x = 29/3 ∧ ∀ y : ℝ, quadratic y ≥ 29/3 :=
by
  sorry

end min_value_of_quadratic_l60_60222


namespace scarves_per_box_l60_60366

theorem scarves_per_box (S M : ℕ) (h1 : S = M) (h2 : 6 * (S + M) = 60) : S = 5 :=
by
  sorry

end scarves_per_box_l60_60366


namespace original_fraction_is_one_third_l60_60313

theorem original_fraction_is_one_third (a b : ℕ) 
  (coprime_ab : Nat.gcd a b = 1) 
  (h : (a + 2) * b = 3 * a * b^2) : 
  (a = 1 ∧ b = 3) := 
by 
  sorry

end original_fraction_is_one_third_l60_60313


namespace pencils_bought_l60_60963

theorem pencils_bought (payment change pencil_cost glue_cost : ℕ)
  (h_payment : payment = 1000)
  (h_change : change = 100)
  (h_pencil_cost : pencil_cost = 210)
  (h_glue_cost : glue_cost = 270) :
  (payment - change - glue_cost) / pencil_cost = 3 :=
by sorry

end pencils_bought_l60_60963


namespace like_terms_solutions_l60_60574

theorem like_terms_solutions (x y : ℤ) (h1 : 5 = 4 * x + 1) (h2 : 3 * y = 6) :
  x = 1 ∧ y = 2 := 
by 
  -- proof goes here
  sorry

end like_terms_solutions_l60_60574


namespace percent_of_b_l60_60977

theorem percent_of_b (a b c : ℝ) (h1 : c = 0.25 * a) (h2 : b = 2.5 * a) : c = 0.1 * b := 
by
  sorry

end percent_of_b_l60_60977


namespace units_digit_of_eight_consecutive_odd_numbers_is_zero_l60_60927

def is_odd (n : ℤ) : Prop :=
  ∃ k : ℤ, n = 2 * k + 1

theorem units_digit_of_eight_consecutive_odd_numbers_is_zero (n : ℤ)
  (h₀ : is_odd n) :
  ((n * (n + 2) * (n + 4) * (n + 6) * (n + 8) * (n + 10) * (n + 12) * (n + 14)) % 10 = 0) :=
sorry

end units_digit_of_eight_consecutive_odd_numbers_is_zero_l60_60927


namespace minimum_value_an_eq_neg28_at_n_eq_3_l60_60586

noncomputable def seq_an (n : ℕ) : ℝ :=
  if n > 0 then (5 / 2) * n^2 - (13 / 2) * n
  else 0

noncomputable def delta_seq_an (n : ℕ) : ℝ := seq_an (n + 1) - seq_an n

noncomputable def delta2_seq_an (n : ℕ) : ℝ := delta_seq_an (n + 1) - delta_seq_an n

theorem minimum_value_an_eq_neg28_at_n_eq_3 : 
  ∃ (n : ℕ), n = 3 ∧ seq_an n = -28 :=
by
  sorry

end minimum_value_an_eq_neg28_at_n_eq_3_l60_60586


namespace polynomial_not_divisible_by_x_minus_5_l60_60880

theorem polynomial_not_divisible_by_x_minus_5 (m : ℝ) :
  (∀ x, x = 4 → (4 * x^3 - 16 * x^2 + m * x - 20) = 0) →
  ¬(∀ x, x = 5 → (4 * x^3 - 16 * x^2 + m * x - 20) = 0) :=
by
  sorry

end polynomial_not_divisible_by_x_minus_5_l60_60880


namespace problem_l60_60581

-- Definitions and conditions
variable {a : ℕ → ℝ} -- sequence definition
variable {S : ℕ → ℝ} -- sum of first n terms

-- Condition: a_n ≠ 0 for all n ∈ ℕ^*
axiom h1 : ∀ n : ℕ, n > 0 → a n ≠ 0

-- Condition: a_n * a_{n+1} = S_n
axiom h2 : ∀ n : ℕ, n > 0 → a n * a (n + 1) = S n

-- Given: S_1 = a_1
axiom h3 : S 1 = a 1

-- Given: S_2 = a_1 + a_2
axiom h4 : S 2 = a 1 + a 2

-- Prove: a_3 - a_1 = 1
theorem problem : a 3 - a 1 = 1 := by
  sorry

end problem_l60_60581


namespace jennifer_sweets_l60_60629

theorem jennifer_sweets :
  let green_sweets := 212
  let blue_sweets := 310
  let yellow_sweets := 502
  let total_sweets := green_sweets + blue_sweets + yellow_sweets
  let number_of_people := 4
  total_sweets / number_of_people = 256 := 
by
  sorry

end jennifer_sweets_l60_60629


namespace find_value_of_a_l60_60627

theorem find_value_of_a (a : ℝ) 
  (h : (2 * a + 16 + 3 * a - 8) / 2 = 69) : a = 26 := 
by
  sorry

end find_value_of_a_l60_60627


namespace meet_time_approx_l60_60078

noncomputable def length_of_track : ℝ := 1800 -- in meters
noncomputable def speed_first_woman : ℝ := 10 * 1000 / 3600 -- in meters per second
noncomputable def speed_second_woman : ℝ := 20 * 1000 / 3600 -- in meters per second
noncomputable def relative_speed : ℝ := speed_first_woman + speed_second_woman

theorem meet_time_approx (ε : ℝ) (hε : ε = 216.048) :
  ∃ t : ℝ, t = length_of_track / relative_speed ∧ abs (t - ε) < 0.001 :=
by
  sorry

end meet_time_approx_l60_60078


namespace solve_z_for_complex_eq_l60_60326

theorem solve_z_for_complex_eq (i : ℂ) (h : i^2 = -1) : ∀ (z : ℂ), 3 - 2 * i * z = -4 + 5 * i * z → z = -i :=
by
  intro z
  intro eqn
  -- The proof would go here
  sorry

end solve_z_for_complex_eq_l60_60326


namespace find_a_for_square_of_binomial_l60_60198

theorem find_a_for_square_of_binomial (a : ℝ) :
  (∃ r s : ℝ, (r * x + s)^2 = a * x^2 + 18 * x + 9) ↔ a = 9 := 
sorry

end find_a_for_square_of_binomial_l60_60198


namespace glen_animals_total_impossible_l60_60903

theorem glen_animals_total_impossible (t : ℕ) :
  ¬ (∃ t : ℕ, 41 * t = 108) := sorry

end glen_animals_total_impossible_l60_60903


namespace bouquet_count_l60_60117

theorem bouquet_count : ∃ n : ℕ, n = 9 ∧ ∀ (r c : ℕ), 3 * r + 2 * c = 50 → n = 9 :=
by
  sorry

end bouquet_count_l60_60117


namespace find_prime_pair_l60_60408

noncomputable def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem find_prime_pair (p q : ℕ) (hp : is_prime p) (hq : is_prime q) (h : p > q) (h_prime : is_prime (p^5 - q^5)) : (p, q) = (3, 2) := 
  sorry

end find_prime_pair_l60_60408


namespace option_D_not_right_angled_l60_60215

def is_right_angled_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

def option_A (a b c : ℝ) : Prop :=
  b^2 = a^2 - c^2

def option_B (a b c : ℝ) : Prop :=
  a = 3 * c / 5 ∧ b = 4 * c / 5

def option_C (A B C : ℝ) : Prop :=
  C = A - B ∧ A + B + C = 180

def option_D (A B C : ℝ) : Prop :=
  A / 3 = B / 4 ∧ B / 4 = C / 5

theorem option_D_not_right_angled (a b c A B C : ℝ) :
  ¬ is_right_angled_triangle a b c ↔ option_D A B C :=
  sorry

end option_D_not_right_angled_l60_60215


namespace center_of_tangent_circle_l60_60797

theorem center_of_tangent_circle (x y : ℝ) 
  (h1 : 3*x - 4*y = 12) 
  (h2 : 3*x - 4*y = -24)
  (h3 : x - 2*y = 0) : 
  (x, y) = (-6, -3) :=
by
  sorry

end center_of_tangent_circle_l60_60797


namespace percentage_of_number_l60_60161

theorem percentage_of_number (N P : ℕ) (h₁ : N = 50) (h₂ : N = (P * N / 100) + 42) : P = 16 :=
by
  sorry

end percentage_of_number_l60_60161


namespace expand_expression_l60_60618

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := 
by
  sorry

end expand_expression_l60_60618


namespace isosceles_triangle_smallest_angle_l60_60665

-- Given conditions:
-- 1. The triangle is isosceles
-- 2. One angle is 40% larger than the measure of a right angle

theorem isosceles_triangle_smallest_angle :
  ∃ (A B C : ℝ), 
  A + B + C = 180 ∧ 
  (A = B ∨ A = C ∨ B = C) ∧ 
  (∃ (large_angle : ℝ), large_angle = 90 + 0.4 * 90 ∧ (A = large_angle ∨ B = large_angle ∨ C = large_angle)) →
  (A = 27 ∨ B = 27 ∨ C = 27) := sorry

end isosceles_triangle_smallest_angle_l60_60665


namespace sin_four_arcsin_eq_l60_60930

theorem sin_four_arcsin_eq (x : ℝ) (h : -1 ≤ x ∧ x ≤ 1) : 
  Real.sin (4 * Real.arcsin x) = 4 * x * (1 - 2 * x^2) * Real.sqrt (1 - x^2) :=
by
  sorry

end sin_four_arcsin_eq_l60_60930


namespace measure_of_MNP_l60_60232

-- Define the conditions of the pentagon
variables {M N P Q S : Type} -- Define the vertices of the pentagon
variables {MN NP PQ QS SM : ℝ} -- Define the lengths of the sides
variables (MNP QNS : ℝ) -- Define the measures of the involved angles

-- State the conditions
-- Pentagon sides are equal
axiom equal_sides : MN = NP ∧ NP = PQ ∧ PQ = QS ∧ QS = SM ∧ SM = MN 
-- Angle relation
axiom angle_relation : MNP = 2 * QNS

-- The goal is to prove that measure of angle MNP is 60 degrees
theorem measure_of_MNP : MNP = 60 :=
by {
  sorry -- The proof goes here
}

end measure_of_MNP_l60_60232


namespace range_of_k_l60_60005

theorem range_of_k (k : ℝ) : (∃ x y : ℝ, x^2 + k * y^2 = 2) ∧ (∀ x y : ℝ, y ≠ 0 → x^2 + k * y^2 = 2 → (x = 0 ∧ (∃ a : ℝ, a > 1 ∧ y = a))) → 0 < k ∧ k < 1 :=
sorry

end range_of_k_l60_60005


namespace xiaoxiao_age_in_2015_l60_60217

-- Definitions for conditions
variables (x : ℕ) (T : ℕ)

-- The total age of the family in 2015 was 7 times Xiaoxiao's age
axiom h1 : T = 7 * x

-- The total age of the family in 2020 after the sibling is 6 times Xiaoxiao's age in 2020
axiom h2 : T + 19 = 6 * (x + 5)

-- Proof goal: Xiaoxiao’s age in 2015 is 11
theorem xiaoxiao_age_in_2015 : x = 11 :=
by
  sorry

end xiaoxiao_age_in_2015_l60_60217


namespace maximize_prob_l60_60185

-- Define the probability of correctly answering each question
def prob_A : ℝ := 0.6
def prob_B : ℝ := 0.8
def prob_C : ℝ := 0.5

-- Define the probability of getting two questions correct in a row for each order
def prob_A_first : ℝ := (prob_A * prob_B * (1 - prob_C) + (1 - prob_A) * prob_B * prob_C) +
                        (prob_A * prob_C * (1 - prob_B) + (1 - prob_A) * prob_C * prob_B)
def prob_B_first : ℝ := (prob_B * prob_A * (1 - prob_C) + (1 - prob_B) * prob_A * prob_C) +
                        (prob_B * prob_C * (1 - prob_A) + (1 - prob_B) * prob_C * prob_A)
def prob_C_first : ℝ := (prob_C * prob_A * (1 - prob_B) + (1 - prob_C) * prob_A * prob_B) +
                        (prob_C * prob_B * (1 - prob_A) + (1 - prob_C) * prob_B * prob_A)

-- Prove that the maximum probability is obtained when question C is answered first
theorem maximize_prob : prob_C_first > prob_A_first ∧ prob_C_first > prob_B_first :=
by
  -- Add the proof details here
  sorry

end maximize_prob_l60_60185


namespace max_marks_l60_60183

theorem max_marks (M : ℕ) (h_pass : 55 / 100 * M = 510) : M = 928 :=
sorry

end max_marks_l60_60183


namespace distance_between_points_l60_60883

theorem distance_between_points {A B : ℝ}
  (hA : abs A = 3)
  (hB : abs B = 9) :
  abs (A - B) = 6 ∨ abs (A - B) = 12 :=
sorry

end distance_between_points_l60_60883


namespace proof_problem_l60_60346

variable {a b c : ℝ}
variable (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
variable (h4 : (a+1) * (b+1) * (c+1) = 8)

theorem proof_problem :
  a + b + c ≥ 3 ∧ abc ≤ 1 :=
by
  sorry

end proof_problem_l60_60346


namespace homework_problems_left_l60_60464

def math_problems : ℕ := 43
def science_problems : ℕ := 12
def finished_problems : ℕ := 44

theorem homework_problems_left :
  (math_problems + science_problems - finished_problems) = 11 :=
by
  sorry

end homework_problems_left_l60_60464


namespace detergent_per_pound_l60_60456

-- Define the conditions
def total_ounces_detergent := 18
def total_pounds_clothes := 9

-- Define the question to prove the amount of detergent per pound of clothes
theorem detergent_per_pound : total_ounces_detergent / total_pounds_clothes = 2 := by
  sorry

end detergent_per_pound_l60_60456


namespace cookie_distribution_l60_60901

def trays := 4
def cookies_per_tray := 24
def total_cookies := trays * cookies_per_tray
def packs := 8
def cookies_per_pack := total_cookies / packs

theorem cookie_distribution : cookies_per_pack = 12 := by
  sorry

end cookie_distribution_l60_60901


namespace negative_only_option_B_l60_60776

theorem negative_only_option_B :
  (0 > -3) ∧ 
  (|-3| = 3) ∧ 
  (0 < 3) ∧
  (0 < (1/3)) ∧
  ∀ x, x = -3 → x < 0 :=
by
  sorry

end negative_only_option_B_l60_60776


namespace integers_with_abs_less_than_four_l60_60765

theorem integers_with_abs_less_than_four :
  {x : ℤ | |x| < 4} = {-3, -2, -1, 0, 1, 2, 3} :=
sorry

end integers_with_abs_less_than_four_l60_60765


namespace regression_equation_represents_real_relationship_maximized_l60_60811

-- Definitions from the conditions
def regression_equation (y x : ℝ) := ∃ (a b : ℝ), y = a * x + b

def represents_real_relationship_maximized (y x : ℝ) := regression_equation y x

-- The proof problem statement
theorem regression_equation_represents_real_relationship_maximized 
: ∀ (y x : ℝ), regression_equation y x → represents_real_relationship_maximized y x :=
by
  sorry

end regression_equation_represents_real_relationship_maximized_l60_60811


namespace log_ordering_l60_60854

noncomputable def a : ℝ := Real.log 6 / Real.log 3
noncomputable def b : ℝ := Real.log 8 / Real.log 4
noncomputable def c : ℝ := Real.log 10 / Real.log 5

theorem log_ordering : a > b ∧ b > c :=
by {
  sorry
}

end log_ordering_l60_60854


namespace geometric_sequence_a4_l60_60036

variable (a : ℕ → ℝ) (q : ℝ)

-- Conditions
def condition1 : Prop := 3 * a 5 = a 6
def condition2 : Prop := a 2 = 1

-- Question
def question : Prop := a 4 = 9

theorem geometric_sequence_a4 (h1 : condition1 a) (h2 : condition2 a) : question a :=
sorry

end geometric_sequence_a4_l60_60036


namespace symmetric_circle_equation_l60_60031

-- Define the original circle and the line of symmetry
def original_circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 2
def line_of_symmetry (x y : ℝ) : Prop := 2 * x - y + 3 = 0

-- Proving the equation of the symmetric circle
theorem symmetric_circle_equation :
  (∀ x y : ℝ, original_circle x y ↔ (x + 3)^2 + (y - 2)^2 = 2) :=
by
  sorry

end symmetric_circle_equation_l60_60031


namespace james_monthly_earnings_l60_60660

theorem james_monthly_earnings (initial_subscribers gifted_subscribers earnings_per_subscriber : ℕ)
  (initial_subscribers_eq : initial_subscribers = 150)
  (gifted_subscribers_eq : gifted_subscribers = 50)
  (earnings_per_subscriber_eq : earnings_per_subscriber = 9) :
  (initial_subscribers + gifted_subscribers) * earnings_per_subscriber = 1800 := by
  sorry

end james_monthly_earnings_l60_60660


namespace area_is_12_5_l60_60877

-- Define the triangle XYZ
structure Triangle := 
  (X Y Z : Type) 
  (XZ YZ : ℝ) 
  (angleX angleY angleZ : ℝ)

-- Provided conditions in the problem
def triangleXYZ : Triangle := {
  X := ℝ, 
  Y := ℝ, 
  Z := ℝ, 
  XZ := 5,
  YZ := 5,
  angleX := 45,
  angleY := 45,
  angleZ := 90
}

-- Lean statement to prove the area of triangle XYZ
theorem area_is_12_5 (t : Triangle) 
  (h1 : t.angleZ = 90)
  (h2 : t.angleX = 45)
  (h3 : t.angleY = 45)
  (h4 : t.XZ = 5)
  (h5 : t.YZ = 5) : 
  (1/2 * t.XZ * t.YZ) = 12.5 :=
sorry

end area_is_12_5_l60_60877


namespace min_value_fraction_l60_60576

theorem min_value_fraction (p q r : ℝ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) :
  (∃ T : ℝ, T = (5 * r / (3 * p + 2 * q) + 5 * p / (2 * q + 3 * r) + 2 * q / (p + r)) ∧ T = 19 / 4) :=
sorry

end min_value_fraction_l60_60576


namespace tycho_jogging_schedule_count_l60_60375

-- Definition of the conditions
def non_consecutive_shot_schedule (days : Finset ℕ) : Prop :=
  ∀ day ∈ days, ∀ next_day ∈ days, day < next_day → next_day - day > 1

-- Definition stating there are exactly seven valid schedules
theorem tycho_jogging_schedule_count :
  ∃ (S : Finset (Finset ℕ)), (∀ s ∈ S, s.card = 3 ∧ non_consecutive_shot_schedule s) ∧ S.card = 7 := 
sorry

end tycho_jogging_schedule_count_l60_60375


namespace problem1_eval_problem2_eval_l60_60721

theorem problem1_eval : (1 * (Real.pi - 3.14)^0 - |2 - Real.sqrt 3| + (-1 / 2)^2) = Real.sqrt 3 - 3 / 4 :=
  sorry

theorem problem2_eval : (Real.sqrt (1 / 3) + Real.sqrt 6 * (1 / Real.sqrt 2 + Real.sqrt 8)) = 16 * Real.sqrt 3 / 3 :=
  sorry

end problem1_eval_problem2_eval_l60_60721


namespace fractions_are_integers_l60_60362

theorem fractions_are_integers (a b c : ℤ) (h : ∃ k : ℤ, (a * b / c) + (a * c / b) + (b * c / a) = k) :
  ∃ k1 k2 k3 : ℤ, (a * b / c) = k1 ∧ (a * c / b) = k2 ∧ (b * c / a) = k3 :=
by
  sorry

end fractions_are_integers_l60_60362


namespace factorial_subtraction_l60_60327

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem factorial_subtraction : factorial 10 - factorial 9 = 3265920 := by
  sorry

end factorial_subtraction_l60_60327


namespace sum_of_first_9_terms_45_l60_60571

-- Define the arithmetic sequence and sum of terms in the sequence
def S (n : ℕ) : ℕ := sorry  -- Placeholder for the sum of the first n terms of the sequence
def a (n : ℕ) : ℕ := sorry  -- Placeholder for the n-th term of the sequence

-- Given conditions
axiom condition1 : a 3 + a 5 + a 7 = 15

-- Proof goal
theorem sum_of_first_9_terms_45 : S 9 = 45 :=
by
  sorry

end sum_of_first_9_terms_45_l60_60571


namespace billy_age_l60_60120

variable (B J : ℕ)

theorem billy_age (h1 : B = 3 * J) (h2 : B + J = 60) : B = 45 :=
by
  sorry

end billy_age_l60_60120


namespace find_y_values_l60_60471

theorem find_y_values
  (y₁ y₂ y₃ y₄ y₅ : ℝ)
  (h₁ : y₁ + 3 * y₂ + 6 * y₃ + 10 * y₄ + 15 * y₅ = 3)
  (h₂ : 3 * y₁ + 6 * y₂ + 10 * y₃ + 15 * y₄ + 21 * y₅ = 20)
  (h₃ : 6 * y₁ + 10 * y₂ + 15 * y₃ + 21 * y₄ + 28 * y₅ = 86)
  (h₄ : 10 * y₁ + 15 * y₂ + 21 * y₃ + 28 * y₄ + 36 * y₅ = 225) :
  15 * y₁ + 21 * y₂ + 28 * y₃ + 36 * y₄ + 45 * y₅ = 395 :=
by {
  sorry
}

end find_y_values_l60_60471


namespace degree_of_resulting_poly_l60_60012

-- Define the polynomials involved in the problem
noncomputable def poly_1 : Polynomial ℝ := 3 * Polynomial.X ^ 5 + 2 * Polynomial.X ^ 3 - Polynomial.X - 16
noncomputable def poly_2 : Polynomial ℝ := 4 * Polynomial.X ^ 11 - 8 * Polynomial.X ^ 8 + 6 * Polynomial.X ^ 5 + 35
noncomputable def poly_3 : Polynomial ℝ := (Polynomial.X ^ 2 + 4) ^ 8

-- Define the resulting polynomial
noncomputable def resulting_poly : Polynomial ℝ :=
  poly_1 * poly_2 - poly_3

-- The goal is to prove that the degree of the resulting polynomial is 16
theorem degree_of_resulting_poly : resulting_poly.degree = 16 := 
sorry

end degree_of_resulting_poly_l60_60012


namespace sum_of_first_40_terms_l60_60676

def a : ℕ → ℤ := sorry

def S (n : ℕ) : ℤ := (Finset.range n).sum a

theorem sum_of_first_40_terms :
  (∀ n : ℕ, a (n + 1) + (-1) ^ n * a n = n) →
  S 40 = 420 := 
sorry

end sum_of_first_40_terms_l60_60676


namespace mass_percentage_I_in_CaI2_l60_60514

theorem mass_percentage_I_in_CaI2 :
  let molar_mass_Ca : ℝ := 40.08
  let molar_mass_I : ℝ := 126.90
  let molar_mass_CaI2 : ℝ := molar_mass_Ca + 2 * molar_mass_I
  let mass_percentage_I : ℝ := (2 * molar_mass_I / molar_mass_CaI2) * 100
  mass_percentage_I = 86.36 := by
  sorry

end mass_percentage_I_in_CaI2_l60_60514


namespace abs_value_sum_l60_60469

noncomputable def sin_theta_in_bounds (θ : ℝ) : Prop :=
  -1 ≤ Real.sin θ ∧ Real.sin θ ≤ 1

noncomputable def x_satisfies_log_eq (θ x : ℝ) : Prop :=
  Real.log x / Real.log 3 = 1 + Real.sin θ

theorem abs_value_sum (θ x : ℝ) (h1 : x_satisfies_log_eq θ x) (h2 : sin_theta_in_bounds θ) :
  |x - 1| + |x - 9| = 8 :=
sorry

end abs_value_sum_l60_60469


namespace equation_1_equation_2_l60_60162

theorem equation_1 (x : ℝ) : x^2 - 1 = 8 ↔ x = 3 ∨ x = -3 :=
by sorry

theorem equation_2 (x : ℝ) : (x + 4)^3 = -64 ↔ x = -8 :=
by sorry

end equation_1_equation_2_l60_60162


namespace lcm_28_72_l60_60910

theorem lcm_28_72 : Nat.lcm 28 72 = 504 := by
  sorry

end lcm_28_72_l60_60910


namespace find_a_l60_60541

theorem find_a (a b c : ℕ) (h1 : a + b = c) (h2 : b + c = 6) (h3 : c = 4) : a = 2 :=
by
  sorry

end find_a_l60_60541


namespace fish_minimum_catch_l60_60900

theorem fish_minimum_catch (a1 a2 a3 a4 a5 : ℕ) (h_sum : a1 + a2 + a3 + a4 + a5 = 100)
  (h_non_increasing : a1 ≥ a2 ∧ a2 ≥ a3 ∧ a3 ≥ a4 ∧ a4 ≥ a5) : 
  a1 + a3 + a5 ≥ 50 :=
sorry

end fish_minimum_catch_l60_60900


namespace car_speed_correct_l60_60715

noncomputable def car_speed (d v_bike t_delay : ℝ) (h1 : v_bike > 0) (h2 : t_delay > 0): ℝ := 2 * v_bike

theorem car_speed_correct:
  ∀ (d v_bike : ℝ) (t_delay : ℝ) (h1 : v_bike > 0) (h2 : t_delay > 0),
    (d / v_bike - t_delay = d / (car_speed d v_bike t_delay h1 h2)) → 
    car_speed d v_bike t_delay h1 h2 = 0.6 :=
by
  intros
  -- The proof would go here
  sorry

end car_speed_correct_l60_60715


namespace back_wheel_revolutions_l60_60358

theorem back_wheel_revolutions
  (front_diameter : ℝ) (back_diameter : ℝ) (front_revolutions : ℝ) (back_revolutions : ℝ)
  (front_diameter_eq : front_diameter = 28)
  (back_diameter_eq : back_diameter = 20)
  (front_revolutions_eq : front_revolutions = 50)
  (distance_eq : ∀ {d₁ d₂}, 2 * Real.pi * d₁ / 2 * front_revolutions = back_revolutions * (2 * Real.pi * d₂ / 2)) :
  back_revolutions = 70 :=
by
  have front_circumference : ℝ := 2 * Real.pi * front_diameter / 2
  have back_circumference : ℝ := 2 * Real.pi * back_diameter / 2
  have total_distance : ℝ := front_circumference * front_revolutions
  have revolutions : ℝ := total_distance / back_circumference 
  sorry

end back_wheel_revolutions_l60_60358


namespace part1_part2_l60_60264

-- Let m be the cost price this year
-- Let x be the selling price per bottle
-- Assuming:
-- 1. The cost price per bottle increased by 4 yuan this year compared to last year.
-- 2. The quantity of detergent purchased for 1440 yuan this year equals to the quantity purchased for 1200 yuan last year.
-- 3. The selling price per bottle is 36 yuan with 600 bottles sold per week.
-- 4. Weekly sales increase by 100 bottles for every 1 yuan reduction in price.
-- 5. The selling price cannot be lower than the cost price.

-- Definition for improved readability:
def costPriceLastYear (m : ℕ) : ℕ := m - 4

-- Quantity equations
def quantityPurchasedThisYear (m : ℕ) : ℕ := 1440 / m
def quantityPurchasedLastYear (m : ℕ) : ℕ := 1200 / (costPriceLastYear m)

-- Profit Function
def profitFunction (m x : ℝ) : ℝ :=
  (x - m) * (600 + 100 * (36 - x))

-- Maximum Profit and Best Selling Price
def maxProfit : ℝ := 8100
def bestSellingPrice : ℝ := 33

theorem part1 (m : ℕ) (h₁ : 1440 / m = 1200 / costPriceLastYear m) : m = 24 := by
  sorry  -- Will be proved later

theorem part2 (m : ℝ) (x : ℝ)
    (h₀ : m = 24)
    (hx : 600 + 100 * (36 - x) > 0)
    (hx₁ : x ≥ m)
    : profitFunction m x ≤ maxProfit ∧ (∃! (y : ℝ), y = bestSellingPrice ∧ profitFunction m y = maxProfit) := by
  sorry  -- Will be proved later

end part1_part2_l60_60264


namespace necessary_but_not_sufficient_condition_l60_60876

theorem necessary_but_not_sufficient_condition (a c : ℝ) (h : c ≠ 0) : ¬ ((∀ (a : ℝ) (h : c ≠ 0), (ax^2 + y^2 = c) → ((ax^2 + y^2 = c) → ( (c ≠ 0) ))) ∧ ¬ ((∀ (a : ℝ), ¬ (ax^2 + y^2 ≠ c) → ( (ax^2 + y^2 = c) → ((c = 0) ))) )) :=
sorry

end necessary_but_not_sufficient_condition_l60_60876


namespace parabola_translation_left_by_two_units_l60_60990

/-- 
The parabola y = x^2 + 4x + 5 is obtained by translating the parabola y = x^2 + 1. 
Prove that this translation is 2 units to the left.
-/
theorem parabola_translation_left_by_two_units :
  ∀ x : ℝ, (x^2 + 4*x + 5) = ((x+2)^2 + 1) :=
by
  intro x
  sorry

end parabola_translation_left_by_two_units_l60_60990


namespace value_of_a_squared_plus_2a_l60_60477

theorem value_of_a_squared_plus_2a (a x : ℝ) (h1 : x = -5) (h2 : 2 * x + 8 = x / 5 - a) : a^2 + 2 * a = 3 :=
by {
  sorry
}

end value_of_a_squared_plus_2a_l60_60477


namespace largest_angle_in_triangle_l60_60712

theorem largest_angle_in_triangle
    (a b c : ℝ)
    (h_sum_two_angles : a + b = (7 / 5) * 90)
    (h_angle_difference : b = a + 40) :
    max a (max b c) = 83 :=
by
  sorry

end largest_angle_in_triangle_l60_60712


namespace min_value_of_one_over_a_plus_one_over_b_l60_60914

theorem min_value_of_one_over_a_plus_one_over_b (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a + b = 12) :
  (∃ c : ℝ, (c = 1/ a + 1 / b) ∧ c = 1 / 3) :=
sorry

end min_value_of_one_over_a_plus_one_over_b_l60_60914


namespace triangle_heights_inequality_l60_60147

variable {R : Type} [OrderedRing R]

theorem triangle_heights_inequality (m_a m_b m_c s : R) 
  (h_m_a_nonneg : 0 ≤ m_a) (h_m_b_nonneg : 0 ≤ m_b) (h_m_c_nonneg : 0 ≤ m_c)
  (h_s_nonneg : 0 ≤ s) : 
  m_a^2 + m_b^2 + m_c^2 ≤ s^2 := 
by
  sorry

end triangle_heights_inequality_l60_60147


namespace total_seashells_found_l60_60439

-- Defining the conditions
def joan_daily_seashells : ℕ := 6
def jessica_daily_seashells : ℕ := 8
def length_of_vacation : ℕ := 7

-- Stating the theorem
theorem total_seashells_found : 
  (joan_daily_seashells + jessica_daily_seashells) * length_of_vacation = 98 :=
by
  sorry

end total_seashells_found_l60_60439


namespace problem1_problem2_l60_60085

variable (x a : ℝ)

def P := x^2 - 5*a*x + 4*a^2 < 0
def Q := (x^2 - 2*x - 8 <= 0) ∧ (x^2 + 3*x - 10 > 0)

theorem problem1 (h : 1 = a) (hP : P x a) (hQ : Q x) : 2 < x ∧ x ≤ 4 :=
sorry

theorem problem2 (h1 : ∀ x, ¬P x a → ¬Q x) (h2 : ∃ x, P x a ∧ ¬Q x) : 1 < a ∧ a ≤ 2 :=
sorry

end problem1_problem2_l60_60085


namespace total_price_is_correct_l60_60452

def total_price_of_hats (total_hats : ℕ) (blue_hat_cost green_hat_cost : ℕ) (num_green_hats : ℕ) : ℕ :=
  let num_blue_hats := total_hats - num_green_hats
  let cost_green_hats := num_green_hats * green_hat_cost
  let cost_blue_hats := num_blue_hats * blue_hat_cost
  cost_green_hats + cost_blue_hats

theorem total_price_is_correct : total_price_of_hats 85 6 7 40 = 550 := 
  sorry

end total_price_is_correct_l60_60452


namespace journey_time_l60_60916

-- Conditions
def initial_speed : ℝ := 80  -- miles per hour
def initial_time : ℝ := 5    -- hours
def new_speed : ℝ := 50      -- miles per hour
def distance : ℝ := initial_speed * initial_time

-- Statement
theorem journey_time :
  distance / new_speed = 8.00 :=
by
  sorry

end journey_time_l60_60916


namespace two_digit_plus_one_multiple_of_3_4_5_6_7_l60_60962

theorem two_digit_plus_one_multiple_of_3_4_5_6_7 (n : ℕ) (h1 : 10 ≤ n) (h2 : n < 100) :
  (∃ m : ℕ, (m = n - 1 ∧ m % 3 = 0 ∧ m % 4 = 0 ∧ m % 5 = 0 ∧ m % 6 = 0 ∧ m % 7 = 0)) → False :=
sorry

end two_digit_plus_one_multiple_of_3_4_5_6_7_l60_60962


namespace problem_solution_set_l60_60287

open Nat

def combination (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)
def permutation (n k : ℕ) : ℕ := n.factorial / (n - k).factorial

theorem problem_solution_set :
  {n : ℕ | 2 * combination n 3 ≤ permutation n 2} = {n | n = 3 ∨ n = 4 ∨ n = 5} :=
by
  sorry

end problem_solution_set_l60_60287


namespace part1_part2_l60_60401

-- Define set A
def set_A : Set ℝ := { x | -3 ≤ x ∧ x ≤ 4 }

-- Define set B depending on m
def set_B (m : ℝ) : Set ℝ := { x | 2 * m - 1 ≤ x ∧ x ≤ m + 1 }

-- Part 1: When m = -3, find A ∩ B
theorem part1 : set_B (-3) ∩ set_A = { x | -3 ≤ x ∧ x ≤ -2 } := 
sorry

-- Part 2: Find the range of m such that B ⊆ A
theorem part2 (m : ℝ) : set_B m ⊆ set_A ↔ m ≥ -1 :=
sorry

end part1_part2_l60_60401


namespace remainder_sequences_mod_1000_l60_60168

theorem remainder_sequences_mod_1000 :
  ∃ m, (m = 752) ∧ (m % 1000 = 752) ∧ 
  (∃ (a : ℕ → ℕ) (h : ∀ i, 1 ≤ i ∧ i ≤ 6 → (a i) - i % 2 = 1), 
    (∀ i j, 1 ≤ i ∧ i ≤ j ∧ j ≤ 6 → a i ≤ a j) ∧ 
    (∀ i, 1 ≤ i ∧ i ≤ 6 → 1 ≤ a i ∧ a i ≤ 1500)
  ) := by
    -- proof would go here
    sorry

end remainder_sequences_mod_1000_l60_60168


namespace tangent_line_at_pi_one_l60_60949

noncomputable def function (x : ℝ) : ℝ := Real.exp x * Real.sin x + 1
noncomputable def tangent_line (x : ℝ) (y : ℝ) : ℝ := x * Real.exp Real.pi + y - 1 - Real.pi * Real.exp Real.pi

theorem tangent_line_at_pi_one :
  tangent_line x y = 0 ↔ y = function x → x = Real.pi ∧ y = 1 :=
by
  sorry

end tangent_line_at_pi_one_l60_60949


namespace choose_student_B_l60_60796

-- Define the scores for students A and B
def scores_A : List ℕ := [72, 85, 86, 90, 92]
def scores_B : List ℕ := [76, 83, 85, 87, 94]

-- Function to calculate the average of scores
def average (scores : List ℕ) : ℚ :=
  scores.sum / scores.length

-- Function to calculate the variance of scores
def variance (scores : List ℕ) : ℚ :=
  let mean := average scores
  (scores.map (λ x => (x - mean) * (x - mean))).sum / scores.length

-- Calculate the average scores for A and B
def avg_A : ℚ := average scores_A
def avg_B : ℚ := average scores_B

-- Calculate the variances for A and B
def var_A : ℚ := variance scores_A
def var_B : ℚ := variance scores_B

-- The theorem to be proved
theorem choose_student_B : var_B < var_A :=
  by sorry

end choose_student_B_l60_60796


namespace total_number_of_workers_l60_60835

theorem total_number_of_workers 
  (W : ℕ) 
  (h_all_avg : W * 8000 = 10 * 12000 + (W - 10) * 6000) : 
  W = 30 := 
by
  sorry

end total_number_of_workers_l60_60835


namespace value_of_a_minus_b_l60_60133

theorem value_of_a_minus_b (a b : ℝ) (h₁ : |a| = 2) (h₂ : |b| = 5) (h₃ : a < b) :
  a - b = -3 ∨ a - b = -7 := 
sorry

end value_of_a_minus_b_l60_60133


namespace eccentricity_of_ellipse_l60_60379

variable (a b c d1 d2 : ℝ)
variable (h1 : a > 0) (h2 : b > 0) (h3 : a > b)
variable (h4 : 2 * c = (d1 + d2) / 2)
variable (h5 : d1 + d2 = 2 * a)

theorem eccentricity_of_ellipse : (c / a) = 1 / 2 :=
by
  sorry

end eccentricity_of_ellipse_l60_60379


namespace cory_fruits_arrangement_l60_60911

-- Conditions
def apples : ℕ := 4
def oranges : ℕ := 2
def lemon : ℕ := 1
def total_fruits : ℕ := apples + oranges + lemon

-- Formula to calculate the number of distinct ways
def factorial (n : ℕ) : ℕ := 
  if n = 0 then 1 else n * factorial (n - 1)

def arrangement_count : ℕ :=
  factorial total_fruits / (factorial apples * factorial oranges * factorial lemon)

theorem cory_fruits_arrangement : arrangement_count = 105 := by
  -- Sorry is placed here to skip the actual proof
  sorry

end cory_fruits_arrangement_l60_60911


namespace user_count_exceed_50000_l60_60710

noncomputable def A (t : ℝ) (k : ℝ) := 500 * Real.exp (k * t)

theorem user_count_exceed_50000 :
  (∃ k : ℝ, A 10 k = 2000) →
  (∀ t : ℝ, A t k > 50000) →
  ∃ t : ℝ, t >= 34 :=
by
  sorry

end user_count_exceed_50000_l60_60710


namespace solve_for_a_l60_60206

theorem solve_for_a (a : ℝ) (h : 4 * a + 9 + (3 * a + 5) = 0) : a = -2 :=
by
  sorry

end solve_for_a_l60_60206


namespace line_equation_l60_60429

theorem line_equation (θ : Real) (b : Real) (h1 : θ = 45) (h2 : b = 2) : (y = x + b) :=
by
  -- Assume θ = 45°. The corresponding slope is k = tan(θ) = 1.
  -- Since the y-intercept b = 2, the equation of the line y = mx + b = x + 2.
  sorry

end line_equation_l60_60429


namespace problem_l60_60790

   def f (n : ℕ) : ℕ := sorry

   theorem problem (f : ℕ → ℕ) (h1 : ∀ n, f (f n) + f n = 2 * n + 3) (h2 : f 0 = 1) :
     f 2013 = 2014 :=
   sorry
   
end problem_l60_60790


namespace imaginary_part_is_empty_l60_60641

def imaginary_part_empty (z : ℂ) : Prop :=
  z.im = 0

theorem imaginary_part_is_empty (z : ℂ) (h : z.im = 0) : imaginary_part_empty z :=
by
  -- proof skipped
  sorry

end imaginary_part_is_empty_l60_60641


namespace find_first_number_l60_60861

theorem find_first_number (x : ℝ) (h1 : 2994 / x = 175) (h2 : 29.94 / 1.45 = 17.5) : x = 17.1 :=
by
  sorry

end find_first_number_l60_60861


namespace worker_allocation_correct_l60_60268

variable (x y : ℕ)
variable (H1 : x + y = 50)
variable (H2 : x = 30)
variable (H3 : y = 20)
variable (H4 : 120 * (50 - x) = 2 * 40 * x)

theorem worker_allocation_correct 
  (h₁ : x = 30) 
  (h₂ : y = 20) 
  (h₃ : x + y = 50) 
  (h₄ : 120 * (50 - x) = 2 * 40 * x) 
  : true := 
by
  sorry

end worker_allocation_correct_l60_60268


namespace central_angle_of_unfolded_side_surface_l60_60690

theorem central_angle_of_unfolded_side_surface
  (radius : ℝ) (slant_height : ℝ) (arc_length : ℝ) (central_angle_deg : ℝ)
  (h_radius : radius = 1)
  (h_slant_height : slant_height = 3)
  (h_arc_length : arc_length = 2 * Real.pi) :
  central_angle_deg = 120 :=
by
  sorry

end central_angle_of_unfolded_side_surface_l60_60690


namespace math_problem_proof_l60_60626

variable (Zhang Li Wang Zhao Liu : Prop)
variable (n : ℕ)
variable (reviewed_truth : Zhang → n = 0 ∧ Li → n = 1 ∧ Wang → n = 2 ∧ Zhao → n = 3 ∧ Liu → n = 4)
variable (reviewed_lie : ¬Zhang → ¬(n = 0) ∧ ¬Li → ¬(n = 1) ∧ ¬Wang → ¬(n = 2) ∧ ¬Zhao → ¬(n = 3) ∧ ¬Liu → ¬(n = 4))
variable (some_reviewed : ∃ x, x ∧ ¬x)

theorem math_problem_proof: n = 1 :=
by
  -- Proof omitted, insert logic here
  sorry

end math_problem_proof_l60_60626


namespace simplify_expression_l60_60892

theorem simplify_expression (w : ℝ) :
  2 * w^2 + 3 - 4 * w^2 + 2 * w - 6 * w + 4 = -2 * w^2 - 4 * w + 7 :=
by
  sorry

end simplify_expression_l60_60892


namespace students_in_same_month_l60_60939

theorem students_in_same_month (students : ℕ) (months : ℕ) 
  (h : students = 50) (h_months : months = 12) : 
  ∃ k ≥ 5, ∃ i, i < months ∧ ∃ f : ℕ → ℕ, (∀ j < students, f j < months) 
  ∧ ∃ n ≥ 5, ∃ j < students, f j = i :=
by 
  sorry

end students_in_same_month_l60_60939


namespace simplify_expression_l60_60973

theorem simplify_expression :
  ((3 + 4 + 5 + 6) ^ 2 / 4) + ((3 * 6 + 9) ^ 2 / 3) = 324 := 
  sorry

end simplify_expression_l60_60973


namespace smallest_n_l60_60021

theorem smallest_n (n : ℕ) (h : 5 * n % 26 = 220 % 26) : n = 18 :=
by
  -- Initial congruence simplification
  have h1 : 220 % 26 = 12 := by norm_num
  rw [h1] at h
  -- Reformulation of the problem
  have h2 : 5 * n % 26 = 12 := h
  -- Conclude the smallest n
  sorry

end smallest_n_l60_60021


namespace problem_statement_l60_60437

def has_arithmetic_square_root (x : ℝ) : Prop :=
  ∃ y : ℝ, y * y = x

theorem problem_statement :
  (¬ has_arithmetic_square_root (-abs 9)) ∧
  (has_arithmetic_square_root ((-1/4)^2)) ∧
  (has_arithmetic_square_root 0) ∧
  (has_arithmetic_square_root (10^2)) := 
sorry

end problem_statement_l60_60437


namespace average_steps_per_day_l60_60063

theorem average_steps_per_day (total_steps : ℕ) (h : total_steps = 56392) : 
  (total_steps / 7 : ℚ) = 8056.00 :=
by
  sorry

end average_steps_per_day_l60_60063


namespace calculation_l60_60598

theorem calculation :
  (-1:ℤ)^(2022) + (Real.sqrt 9) - 2 * (Real.sin (Real.pi / 6)) = 3 := by
  -- According to the mathematical problem and the given solution.
  -- Here we use essential definitions and facts provided in the problem.
  sorry

end calculation_l60_60598


namespace problem1_problem2_problem3_problem4_l60_60450

theorem problem1 : 24 - (-16) + (-25) - 32 = -17 := by
  sorry

theorem problem2 : (-1 / 2) * 2 / 2 * (-1 / 2) = 1 / 4 := by
  sorry

theorem problem3 : -2^2 * 5 - (-2)^3 * (1 / 8) + 1 = -18 := by
  sorry

theorem problem4 : ((-1 / 4) - (5 / 6) + (8 / 9)) / (-1 / 6)^2 + (-2)^2 * (-6)= -31 := by
  sorry

end problem1_problem2_problem3_problem4_l60_60450


namespace merry_go_round_cost_per_child_l60_60997

-- Definitions
def num_children := 5
def ferris_wheel_cost_per_child := 5
def num_children_on_ferris_wheel := 3
def ice_cream_cost_per_cone := 8
def ice_cream_cones_per_child := 2
def total_spent := 110

-- Totals
def ferris_wheel_total_cost := num_children_on_ferris_wheel * ferris_wheel_cost_per_child
def ice_cream_total_cost := num_children * ice_cream_cones_per_child * ice_cream_cost_per_cone
def merry_go_round_total_cost := total_spent - ferris_wheel_total_cost - ice_cream_total_cost

-- Final proof statement
theorem merry_go_round_cost_per_child : 
  merry_go_round_total_cost / num_children = 3 :=
by
  -- We skip the actual proof here
  sorry

end merry_go_round_cost_per_child_l60_60997


namespace value_of_expression_l60_60122

theorem value_of_expression {p q : ℝ} (hp : 3 * p^2 + 9 * p - 21 = 0) (hq : 3 * q^2 + 9 * q - 21 = 0) : 
  (3 * p - 4) * (6 * q - 8) = 122 :=
by
  sorry

end value_of_expression_l60_60122


namespace find_k_value_for_unique_real_solution_l60_60643

noncomputable def cubic_has_exactly_one_real_solution (k : ℝ) : Prop :=
    ∃! x : ℝ, 4*x^3 + 9*x^2 + k*x + 4 = 0

theorem find_k_value_for_unique_real_solution :
  ∃ (k : ℝ), k > 0 ∧ cubic_has_exactly_one_real_solution k ∧ k = 6.75 :=
sorry

end find_k_value_for_unique_real_solution_l60_60643


namespace necessary_but_not_sufficient_l60_60972

theorem necessary_but_not_sufficient (m : ℝ) :
  (∀ x : ℝ, x^2 + m * x + 1 > 0) ↔ -2 < m ∧ m < 2 → m < 2 :=
by
  sorry

end necessary_but_not_sufficient_l60_60972


namespace weeks_to_meet_goal_l60_60578

def hourly_rate : ℕ := 6
def hours_monday : ℕ := 2
def hours_tuesday : ℕ := 3
def hours_wednesday : ℕ := 4
def hours_thursday : ℕ := 2
def hours_friday : ℕ := 3
def helmet_cost : ℕ := 340
def gloves_cost : ℕ := 45
def initial_savings : ℕ := 40
def misc_expenses : ℕ := 20

theorem weeks_to_meet_goal : 
  let total_needed := helmet_cost + gloves_cost + misc_expenses
  let total_deficit := total_needed - initial_savings
  let total_weekly_hours := hours_monday + hours_tuesday + hours_wednesday + hours_thursday + hours_friday
  let weekly_earnings := total_weekly_hours * hourly_rate
  let weeks_required := Nat.ceil (total_deficit / weekly_earnings)
  weeks_required = 5 := sorry

end weeks_to_meet_goal_l60_60578


namespace max_students_late_all_three_days_l60_60107

theorem max_students_late_all_three_days (A B C total l: ℕ) 
  (hA: A = 20) 
  (hB: B = 13) 
  (hC: C = 7) 
  (htotal: total = 30) 
  (hposA: 0 ≤ A) (hposB: 0 ≤ B) (hposC: 0 ≤ C) 
  (hpostotal: 0 ≤ total) 
  : l = 5 := by
  sorry

end max_students_late_all_three_days_l60_60107


namespace jimmy_more_sheets_than_tommy_l60_60483

theorem jimmy_more_sheets_than_tommy 
  (jimmy_initial_sheets : ℕ)
  (tommy_initial_sheets : ℕ)
  (additional_sheets : ℕ)
  (h1 : tommy_initial_sheets = jimmy_initial_sheets + 25)
  (h2 : jimmy_initial_sheets = 58)
  (h3 : additional_sheets = 85) :
  (jimmy_initial_sheets + additional_sheets) - tommy_initial_sheets = 60 := 
by
  sorry

end jimmy_more_sheets_than_tommy_l60_60483


namespace half_abs_diff_squares_l60_60836

theorem half_abs_diff_squares (a b : ℝ) (h₁ : a = 25) (h₂ : b = 20) :
  (1 / 2) * |a^2 - b^2| = 112.5 :=
sorry

end half_abs_diff_squares_l60_60836


namespace polynomial_simplification_l60_60918

noncomputable def given_polynomial (x : ℝ) : ℝ :=
  3 - 5 * x - 7 * x^2 + 9 + 11 * x - 13 * x^2 + 15 - 17 * x + 19 * x^2 + 2 * x^3

theorem polynomial_simplification (x : ℝ) :
  given_polynomial x = 2 * x^3 - x^2 - 11 * x + 27 :=
by
  -- The proof is skipped
  sorry

end polynomial_simplification_l60_60918


namespace difference_of_squares_l60_60367

theorem difference_of_squares : (540^2 - 460^2 = 80000) :=
by
  have a := 540
  have b := 460
  have identity := (a + b) * (a - b)
  sorry

end difference_of_squares_l60_60367


namespace monthly_sales_fraction_l60_60552

theorem monthly_sales_fraction (V S_D T : ℝ) 
  (h1 : S_D = 6 * V) 
  (h2 : S_D = 0.35294117647058826 * T) 
  : V = (1 / 17) * T :=
sorry

end monthly_sales_fraction_l60_60552


namespace average_monthly_balance_l60_60887

theorem average_monthly_balance :
  let balances := [100, 200, 250, 50, 300, 300]
  (balances.sum / balances.length : ℕ) = 200 :=
by
  sorry

end average_monthly_balance_l60_60887


namespace expected_yield_correct_l60_60819

/-- Define the problem variables and conditions -/
def steps_x : ℕ := 25
def steps_y : ℕ := 20
def step_length : ℝ := 2.5
def yield_per_sqft : ℝ := 0.75

/-- Calculate the dimensions in feet -/
def length_x := steps_x * step_length
def length_y := steps_y * step_length

/-- Calculate the area of the orchard -/
def area := length_x * length_y

/-- Calculate the expected yield of apples -/
def expected_yield := area * yield_per_sqft

/-- Prove the expected yield of apples is 2343.75 pounds -/
theorem expected_yield_correct : expected_yield = 2343.75 := sorry

end expected_yield_correct_l60_60819


namespace solution_exists_l60_60931

theorem solution_exists (x y z u v : ℕ) (hx : x > 2000) (hy : y > 2000) (hz : z > 2000) (hu : u > 2000) (hv : v > 2000) : 
  x^2 + y^2 + z^2 + u^2 + v^2 = x * y * z * u * v - 65 :=
sorry

end solution_exists_l60_60931


namespace perfect_squares_between_50_and_1000_l60_60231

theorem perfect_squares_between_50_and_1000 :
  ∃ (count : ℕ), count = 24 ∧ ∀ (n : ℕ), 50 < n * n ∧ n * n < 1000 ↔ 8 ≤ n ∧ n ≤ 31 :=
by {
  -- proof goes here
  sorry
}

end perfect_squares_between_50_and_1000_l60_60231


namespace max_value_of_sum_max_value_achievable_l60_60039

theorem max_value_of_sum (a b c d : ℝ) 
  (h : a^6 + b^6 + c^6 + d^6 = 64) : a^7 + b^7 + c^7 + d^7 ≤ 128 :=
sorry

theorem max_value_achievable : ∃ a b c d : ℝ,
  a^6 + b^6 + c^6 + d^6 = 64 ∧ a^7 + b^7 + c^7 + d^7 = 128 :=
sorry

end max_value_of_sum_max_value_achievable_l60_60039


namespace domain_of_rational_function_l60_60656

theorem domain_of_rational_function 
  (c : ℝ) 
  (h : -7 * (6 ^ 2) + 28 * c < 0) : 
  c < -9 / 7 :=
by sorry

end domain_of_rational_function_l60_60656


namespace max_minute_hands_l60_60881

theorem max_minute_hands (m n : ℕ) (h : m * n = 27) : m + n ≤ 28 :=
  sorry

end max_minute_hands_l60_60881


namespace probability_equals_two_thirds_l60_60920

-- Definitions for total arrangements and favorable arrangements
def total_arrangements : ℕ := Nat.choose 6 2
def favorable_arrangements : ℕ := Nat.choose 5 2

-- Probability that 2 zeros are not adjacent
def probability_not_adjacent : ℚ := favorable_arrangements / total_arrangements

theorem probability_equals_two_thirds : probability_not_adjacent = 2 / 3 := 
by 
  let total_arrangements := 15
  let favorable_arrangements := 10
  have h1 : probability_not_adjacent = (10 : ℚ) / (15 : ℚ) := rfl
  have h2 : (10 : ℚ) / (15 : ℚ) = 2 / 3 := by norm_num
  exact Eq.trans h1 h2 

end probability_equals_two_thirds_l60_60920


namespace largest_beverage_amount_l60_60841

theorem largest_beverage_amount :
  let Milk := (3 / 8 : ℚ)
  let Cider := (7 / 10 : ℚ)
  let OrangeJuice := (11 / 15 : ℚ)
  OrangeJuice > Milk ∧ OrangeJuice > Cider :=
by
  have Milk := (3 / 8 : ℚ)
  have Cider := (7 / 10 : ℚ)
  have OrangeJuice := (11 / 15 : ℚ)
  sorry

end largest_beverage_amount_l60_60841


namespace john_unanswered_problems_is_9_l60_60902

variables (x y z : ℕ)

theorem john_unanswered_problems_is_9 (h1 : 5 * x + 2 * z = 93)
                                      (h2 : 4 * x - y = 54)
                                      (h3 : x + y + z = 30) : 
  z = 9 :=
by 
  sorry

end john_unanswered_problems_is_9_l60_60902


namespace range_of_a_l60_60011

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x^2 + (a + 1) * x + 1 < 0) → (a < -3 ∨ a > 1) :=
by
  sorry

end range_of_a_l60_60011


namespace sum_of_three_numbers_l60_60007

theorem sum_of_three_numbers (a b c : ℝ) (h1 : a + b = 35) (h2 : b + c = 42) (h3 : c + a = 58) :
  a + b + c = 67.5 :=
by
  sorry

end sum_of_three_numbers_l60_60007


namespace inequality_proof_l60_60124

theorem inequality_proof (a b c : ℝ) (h : a * b * c = 1) : 
  1 / (2 * a^2 + b^2 + 3) + 1 / (2 * b^2 + c^2 + 3) + 1 / (2 * c^2 + a^2 + 3) ≤ 1 / 2 := 
by
  sorry

end inequality_proof_l60_60124


namespace jonah_total_lemonade_l60_60386

theorem jonah_total_lemonade : 
  0.25 + 0.4166666666666667 + 0.25 + 0.5833333333333334 = 1.5 :=
by
  sorry

end jonah_total_lemonade_l60_60386


namespace avery_donation_l60_60539

theorem avery_donation (shirts pants shorts : ℕ)
  (h_shirts : shirts = 4)
  (h_pants : pants = 2 * shirts)
  (h_shorts : shorts = pants / 2) :
  shirts + pants + shorts = 16 := by
  sorry

end avery_donation_l60_60539


namespace A_20_equals_17711_l60_60595

def A : ℕ → ℕ
| 0     => 1  -- by definition, an alternating sequence on an empty set, counting empty sequence
| 1     => 2  -- base case
| 2     => 3  -- base case
| (n+3) => A (n+2) + A (n+1)

theorem A_20_equals_17711 : A 20 = 17711 := 
sorry

end A_20_equals_17711_l60_60595


namespace star_polygon_x_value_l60_60779

theorem star_polygon_x_value
  (a b c d e p q r s t : ℝ)
  (h1 : p + q + r + s + t = 500)
  (h2 : a + b + c + d + e = x)
  :
  x = 140 :=
sorry

end star_polygon_x_value_l60_60779


namespace find_S11_l60_60542

variable {a : ℕ → ℚ} -- Define the arithmetic sequence as a function

-- Define conditions
def arithmetic_sequence (a : ℕ → ℚ) :=
∀ n m, a (n + m) = a n + a m

def S (n : ℕ) (a : ℕ → ℚ) : ℚ := (n / 2 : ℚ) * (a 1 + a n)

-- Define the problem statement to be proved
theorem find_S11 (h_arith : arithmetic_sequence a) (h_eq : a 3 + a 6 + a 9 = 54) : 
  S 11 a = 198 :=
sorry

end find_S11_l60_60542


namespace arithmetic_sequence_probability_correct_l60_60288

noncomputable def arithmetic_sequence_probability : ℚ := 
  let total_ways := Nat.choose 5 3
  let arithmetic_sequences := 4
  (arithmetic_sequences : ℚ) / (total_ways : ℚ)

theorem arithmetic_sequence_probability_correct :
  arithmetic_sequence_probability = 0.4 := by
  unfold arithmetic_sequence_probability
  sorry

end arithmetic_sequence_probability_correct_l60_60288


namespace average_height_l60_60544

theorem average_height (avg1 avg2 : ℕ) (n1 n2 : ℕ) (total_students : ℕ)
  (h1 : avg1 = 20) (h2 : avg2 = 20) (h3 : n1 = 20) (h4 : n2 = 11) (h5 : total_students = 31) :
  (n1 * avg1 + n2 * avg2) / total_students = 20 :=
by
  -- Placeholder for the proof
  sorry

end average_height_l60_60544


namespace car_speed_l60_60689

theorem car_speed (v : ℝ) (h : (1/v) * 3600 = ((1/48) * 3600) + 15) : v = 40 := 
by 
  sorry

end car_speed_l60_60689


namespace roots_poly_eval_l60_60335

theorem roots_poly_eval : ∀ (c d : ℝ), (c + d = 6 ∧ c * d = 8) → c^4 + c^3 * d + d^3 * c + d^4 = 432 :=
by
  intros c d h
  sorry

end roots_poly_eval_l60_60335


namespace number_of_rocks_tossed_l60_60489

-- Conditions
def pebbles : ℕ := 6
def rocks : ℕ := 3
def boulders : ℕ := 2
def pebble_splash : ℚ := 1 / 4
def rock_splash : ℚ := 1 / 2
def boulder_splash : ℚ := 2

-- Total width of the splashes
def total_splash (R : ℕ) : ℚ := 
  pebbles * pebble_splash + R * rock_splash + boulders * boulder_splash

-- Given condition
def total_splash_condition : ℚ := 7

theorem number_of_rocks_tossed : 
  total_splash rocks = total_splash_condition → rocks = 3 :=
by
  intro h
  sorry

end number_of_rocks_tossed_l60_60489


namespace factor_correct_l60_60251

noncomputable def factor_expression (x : ℝ) : ℝ :=
  66 * x^6 - 231 * x^12

theorem factor_correct (x : ℝ) :
  factor_expression x = 33 * x^6 * (2 - 7 * x^6) :=
by 
  sorry

end factor_correct_l60_60251


namespace problem_statement_l60_60567

noncomputable def is_integer (x : ℚ) : Prop := ∃ (n : ℤ), x = n

theorem problem_statement (m n p q : ℕ) (h₁ : m ≠ p) (h₂ : is_integer ((mn + pq : ℚ) / (m - p))) :
  is_integer ((mq + np : ℚ) / (m - p)) :=
sorry

end problem_statement_l60_60567


namespace value_of_f_2017_l60_60917

def f (x : ℕ) : ℕ := x^2 - x * (0 : ℕ) - 1

theorem value_of_f_2017 : f 2017 = 2016 * 2018 := by
  sorry

end value_of_f_2017_l60_60917


namespace at_most_one_cube_l60_60400

theorem at_most_one_cube (a : ℕ → ℕ) (h₁ : ∀ n, a (n + 1) = a n ^ 2 + 2018) :
  ∃! n, ∃ m : ℕ, a n = m ^ 3 := sorry

end at_most_one_cube_l60_60400


namespace only_set_C_forms_triangle_l60_60294

def triangle_inequality (a b c : ℝ) : Prop := 
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem only_set_C_forms_triangle : 
  (¬ triangle_inequality 1 2 3) ∧ 
  (¬ triangle_inequality 2 3 6) ∧ 
  triangle_inequality 4 6 8 ∧ 
  (¬ triangle_inequality 5 6 12) := 
by 
  sorry

end only_set_C_forms_triangle_l60_60294


namespace sequence_formula_l60_60476

theorem sequence_formula (a : ℕ → ℝ) (h₀ : a 1 = 1) (h₁ : ∀ n : ℕ, 0 < n → a (n + 1) = a n / (1 + a n)) : 
  ∀ n : ℕ, 0 < n → a n = 1 / n := 
by 
  sorry

end sequence_formula_l60_60476


namespace area_in_square_yards_l60_60308

/-
  Given:
  - length of the classroom in feet
  - width of the classroom in feet

  Prove that the area required to cover the classroom in square yards is 30. 
-/

def classroom_length_feet : ℕ := 15
def classroom_width_feet : ℕ := 18
def feet_to_yard (feet : ℕ) : ℕ := feet / 3

theorem area_in_square_yards :
  let length_yards := feet_to_yard classroom_length_feet
  let width_yards := feet_to_yard classroom_width_feet
  length_yards * width_yards = 30 :=
by
  sorry

end area_in_square_yards_l60_60308


namespace Ram_money_l60_60073

theorem Ram_money (R G K : ℕ) (h1 : R = 7 * G / 17) (h2 : G = 7 * K / 17) (h3 : K = 4046) : R = 686 := by
  sorry

end Ram_money_l60_60073


namespace cubic_sum_identity_l60_60306

theorem cubic_sum_identity (x y z : ℝ) (h1 : x + y + z = 15) (h2 : xy + yz + zx = 34) :
  x^3 + y^3 + z^3 - 3 * x * y * z = 1845 :=
by
  sorry

end cubic_sum_identity_l60_60306


namespace total_pages_is_905_l60_60523

def history_pages : ℕ := 160
def geography_pages : ℕ := history_pages + 70
def math_pages : ℕ := (history_pages + geography_pages) / 2
def science_pages : ℕ := 2 * history_pages
def total_pages : ℕ := history_pages + geography_pages + math_pages + science_pages

theorem total_pages_is_905 : total_pages = 905 := by
  sorry

end total_pages_is_905_l60_60523


namespace final_segment_position_correct_l60_60314

def initial_segment : ℝ × ℝ := (1, 6)
def rotate_180_about (p : ℝ) (x : ℝ) : ℝ := p - (x - p)
def first_rotation_segment : ℝ × ℝ := (rotate_180_about 2 6, rotate_180_about 2 1)
def second_rotation_segment : ℝ × ℝ := (rotate_180_about 1 3, rotate_180_about 1 (-2))

theorem final_segment_position_correct :
  second_rotation_segment = (-1, 4) :=
by
  -- This is a placeholder for the actual proof.
  sorry

end final_segment_position_correct_l60_60314


namespace minimum_ab_value_is_two_l60_60442

noncomputable def minimum_value_ab (a b : ℝ) (h1 : a^2 ≠ 0) (h2 : b ≠ 0)
  (h3 : a^2 * b = a^2 + 1) : ℝ :=
|a * b|

theorem minimum_ab_value_is_two (a b : ℝ) (h1 : a^2 ≠ 0) (h2 : b ≠ 0)
  (h3 : a^2 * b = a^2 + 1) : minimum_value_ab a b h1 h2 h3 = 2 := by
  sorry

end minimum_ab_value_is_two_l60_60442


namespace milk_transfer_equal_l60_60498

theorem milk_transfer_equal (A B C x : ℕ) (hA : A = 1200) (hB : B = A - 750) (hC : C = A - B) (h_eq : B + x = C - x) :
  x = 150 :=
by
  sorry

end milk_transfer_equal_l60_60498


namespace find_red_coin_l60_60020

/- Define the function f(n) as the minimum number of scans required to determine the red coin
   - out of n coins with the given conditions.
   - Seyed has 998 white coins, 1 red coin, and 1 red-white coin.
-/

def f (n : Nat) : Nat := sorry

/- The main theorem to be proved: There exists an algorithm that can find the red coin using 
   the scanner at most 17 times for 1000 coins.
-/

theorem find_red_coin (n : Nat) (h : n = 1000) : f n ≤ 17 := sorry

end find_red_coin_l60_60020


namespace evaluate_expression_l60_60016

theorem evaluate_expression : 2^3 + 2^3 + 2^3 + 2^3 = 2^5 := by
  sorry

end evaluate_expression_l60_60016


namespace cookies_left_l60_60270

theorem cookies_left (days_baking : ℕ) (trays_per_day : ℕ) (cookies_per_tray : ℕ) (frank_eats_per_day : ℕ) (ted_eats_on_sixth_day : ℕ) :
  trays_per_day * cookies_per_tray * days_baking - frank_eats_per_day * days_baking - ted_eats_on_sixth_day = 134 :=
by
  have days_baking := 6
  have trays_per_day := 2
  have cookies_per_tray := 12
  have frank_eats_per_day := 1
  have ted_eats_on_sixth_day := 4
  sorry

end cookies_left_l60_60270


namespace contestants_order_l60_60001

variables (G E H F : ℕ) -- Scores of the participants, given that they are nonnegative

theorem contestants_order (h1 : E + G = F + H) (h2 : F + E = H + G) (h3 : G > E + F) : 
  G ≥ E ∧ G ≥ H ∧ G ≥ F ∧ E = H ∧ E ≥ F :=
by {
  sorry
}

end contestants_order_l60_60001


namespace group_A_percentage_l60_60339

/-!
In an examination, there are 100 questions divided into 3 groups A, B, and C such that each group contains at least one question. 
Each question in group A carries 1 mark, each question in group B carries 2 marks, and each question in group C carries 3 marks. 
It is known that:
- Group B contains 23 questions
- Group C contains 1 question.
Prove that the percentage of the total marks that the questions in group A carry is 60.8%.
-/

theorem group_A_percentage :
  ∃ (a b c : ℕ), b = 23 ∧ c = 1 ∧ (a + b + c = 100) ∧ ((a * 1) + (b * 2) + (c * 3) = 125) ∧ ((a : ℝ) / 125 * 100 = 60.8) :=
by
  sorry

end group_A_percentage_l60_60339


namespace find_radius_of_circle_l60_60731

theorem find_radius_of_circle :
  ∀ (r : ℝ) (α : ℝ) (ρ : ℝ) (θ : ℝ), r > 0 →
  (∀ (x y : ℝ), x = r * Real.cos α ∧ y = r * Real.sin α → x^2 + y^2 = r^2) →
  (∃ (x y: ℝ), x - y + 2 = 0 ∧ 2 * Real.sqrt (r^2 - 2) = 2 * Real.sqrt 2) →
  r = 2 :=
by
  intro r α ρ θ r_pos curve_eq polar_eq
  sorry

end find_radius_of_circle_l60_60731


namespace pow_div_l60_60646

theorem pow_div (x : ℕ) (a b c d : ℕ) (h1 : x^b = d) (h2 : x^(a*d) = c) : c / (d^b) = 512 := by
  sorry

end pow_div_l60_60646


namespace ball_hits_ground_l60_60274

theorem ball_hits_ground :
  ∃ (t : ℝ), (t = 2) ∧ (-4.9 * t^2 + 5.7 * t + 7 = 0) :=
sorry

end ball_hits_ground_l60_60274


namespace sky_color_change_l60_60140

theorem sky_color_change (hours: ℕ) (colors: ℕ) (minutes_per_hour: ℕ) 
                          (H1: hours = 2) 
                          (H2: colors = 12) 
                          (H3: minutes_per_hour = 60) : 
                          (hours * minutes_per_hour) / colors = 10 := 
by
  sorry

end sky_color_change_l60_60140


namespace exist_positive_integers_summing_to_one_l60_60468

theorem exist_positive_integers_summing_to_one :
  ∃ (x y z : ℕ), 0 < x ∧ 0 < y ∧ 0 < z ∧ (1 / (x:ℚ) + 1 / (y:ℚ) + 1 / (z:ℚ) = 1)
    ∧ ((x = 2 ∧ y = 4 ∧ z = 4) ∨ (x = 2 ∧ y = 3 ∧ z = 6) ∨ (x = 3 ∧ y = 3 ∧ z = 3)) :=
by
  sorry

end exist_positive_integers_summing_to_one_l60_60468


namespace total_flour_needed_l60_60773

-- Definitions of flour needed by Katie and Sheila
def katie_flour : ℕ := 3
def sheila_flour : ℕ := katie_flour + 2

-- Statement of the theorem
theorem total_flour_needed : katie_flour + sheila_flour = 8 := by
  -- The proof would go here
  sorry

end total_flour_needed_l60_60773


namespace part1_part2_l60_60971

variables (a b c d m : Real) 

-- Condition: a and b are opposite numbers
def opposite_numbers (a b : Real) : Prop := a = -b

-- Condition: c and d are reciprocals
def reciprocals (c d : Real) : Prop := c = 1 / d

-- Condition: |m| = 3
def absolute_value_three (m : Real) : Prop := abs m = 3

-- Statement for part 1
theorem part1 (h1 : opposite_numbers a b) (h2 : reciprocals c d) (h3 : absolute_value_three m) :
  a + b = 0 ∧ c * d = 1 ∧ (m = 3 ∨ m = -3) :=
by
  sorry

-- Statement for part 2
theorem part2 (h1 : opposite_numbers a b) (h2 : reciprocals c d) (h3 : absolute_value_three m) (h4 : m < 0) :
  m^3 + c * d + (a + b) / m = -26 :=
by
  sorry

end part1_part2_l60_60971


namespace total_sum_money_l60_60359

theorem total_sum_money (a b c : ℝ) (h1 : b = 0.65 * a) (h2 : c = 0.40 * a) (h3 : c = 64) :
  a + b + c = 328 :=
by
  sorry

end total_sum_money_l60_60359


namespace min_S_l60_60261

variable {x y : ℝ}
def condition (x y : ℝ) : Prop := (4 * x^2 + 5 * x * y + 4 * y^2 = 5)
def S (x y : ℝ) : ℝ := x^2 + y^2
theorem min_S (hx : condition x y) : S x y = (10 / 13) :=
sorry

end min_S_l60_60261


namespace kabadi_players_l60_60252

def people_play_kabadi (Kho_only Both Total : ℕ) : Prop :=
  ∃ K : ℕ, Kho_only = 20 ∧ Both = 5 ∧ Total = 30 ∧ K = Total - Kho_only ∧ (K + Both) = 15

theorem kabadi_players :
  people_play_kabadi 20 5 30 :=
by
  sorry

end kabadi_players_l60_60252


namespace base_6_to_base_10_exact_value_l60_60889

def base_6_to_base_10 (n : ℕ) : ℕ :=
  1 * 6^2 + 5 * 6^1 + 4 * 6^0

theorem base_6_to_base_10_exact_value : base_6_to_base_10 154 = 70 := by
  rfl

end base_6_to_base_10_exact_value_l60_60889


namespace James_pays_35_l60_60667

theorem James_pays_35 (first_lesson_free : Bool) (total_lessons : Nat) (cost_per_lesson : Nat) 
  (first_x_paid_lessons_free : Nat) (every_other_remainings_free : Nat) (uncle_pays_half : Bool) :
  total_lessons = 20 → 
  first_lesson_free = true → 
  cost_per_lesson = 5 →
  first_x_paid_lessons_free = 10 →
  every_other_remainings_free = 1 → 
  uncle_pays_half = true →
  (10 * cost_per_lesson + 4 * cost_per_lesson) / 2 = 35 :=
by
  sorry

end James_pays_35_l60_60667


namespace simplify_and_evaluate_expr_l60_60357

variables (a b : Int)

theorem simplify_and_evaluate_expr (ha : a = 1) (hb : b = -2) : 
  2 * (3 * a^2 * b - a * b^2) - 3 * (-a * b^2 + a^2 * b - 1) = 1 :=
by
  sorry

end simplify_and_evaluate_expr_l60_60357


namespace set_union_intersection_example_l60_60849

open Set

theorem set_union_intersection_example :
  let A := {1, 3, 4, 5}
  let B := {2, 4, 6}
  let C := {0, 1, 2, 3, 4}
  (A ∪ B) ∩ C = ({1, 2, 3, 4} : Set ℕ) :=
by
  sorry

end set_union_intersection_example_l60_60849


namespace quadratic_equation_unique_l60_60069

/-- Prove that among the given options, the only quadratic equation in \( x \) is \( x^2 - 3x = 0 \). -/
theorem quadratic_equation_unique (A B C D : ℝ → ℝ) :
  A = (3 * x + 2) →
  B = (x^2 - 3 * x) →
  C = (x + 3 * x * y - 1) →
  D = (1 / x - 4) →
  ∃! (eq : ℝ → ℝ), eq = B := by
  sorry

end quadratic_equation_unique_l60_60069


namespace valentine_giveaway_l60_60262

theorem valentine_giveaway (initial : ℕ) (left : ℕ) (given : ℕ) (h1 : initial = 30) (h2 : left = 22) : given = initial - left → given = 8 :=
by
  sorry

end valentine_giveaway_l60_60262


namespace sample_size_is_10_l60_60143

def product := Type

noncomputable def number_of_products : ℕ := 80
noncomputable def selected_products_for_quality_inspection : ℕ := 10

theorem sample_size_is_10 
  (N : ℕ) (sample_size : ℕ) 
  (hN : N = 80) 
  (h_sample_size : sample_size = 10) : 
  sample_size = 10 :=
by 
  sorry

end sample_size_is_10_l60_60143


namespace square_area_in_ellipse_l60_60472

theorem square_area_in_ellipse :
  (∃ t : ℝ, 
    (∀ x y : ℝ, ((x = t ∨ x = -t) ∧ (y = t ∨ y = -t)) → (x^2 / 4 + y^2 / 8 = 1)) 
    ∧ t > 0 
    ∧ ((2 * t)^2 = 32 / 3)) :=
sorry

end square_area_in_ellipse_l60_60472


namespace chip_cost_l60_60322

theorem chip_cost 
  (calories_per_chip : ℕ)
  (chips_per_bag : ℕ)
  (cost_per_bag : ℕ)
  (desired_calories : ℕ)
  (h1 : calories_per_chip = 10)
  (h2 : chips_per_bag = 24)
  (h3 : cost_per_bag = 2)
  (h4 : desired_calories = 480) : 
  cost_per_bag * (desired_calories / (calories_per_chip * chips_per_bag)) = 4 := 
by 
  sorry

end chip_cost_l60_60322


namespace collinear_iff_linear_combination_l60_60651

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (O A B C : V) (k : ℝ)

theorem collinear_iff_linear_combination (O A B C : V) (k : ℝ) :
  (C = k • A + (1 - k) • B) ↔ ∃ (k' : ℝ), C - B = k' • (A - B) :=
sorry

end collinear_iff_linear_combination_l60_60651


namespace find_higher_percentage_l60_60500

-- Definitions based on conditions
def principal : ℕ := 8400
def time : ℕ := 2
def rate_0 : ℕ := 10
def delta_interest : ℕ := 840

-- The proof statement
theorem find_higher_percentage (r : ℕ) :
  (principal * rate_0 * time / 100 + delta_interest = principal * r * time / 100) →
  r = 15 :=
by sorry

end find_higher_percentage_l60_60500


namespace ratio_movies_allowance_l60_60282

variable (M A : ℕ)
variable (weeklyAllowance moneyEarned endMoney : ℕ)
variable (H1 : weeklyAllowance = 8)
variable (H2 : moneyEarned = 8)
variable (H3 : endMoney = 12)
variable (H4 : weeklyAllowance + moneyEarned - M = endMoney)
variable (H5 : A = 8)
variable (H6 : M = weeklyAllowance + moneyEarned - endMoney / 1)

theorem ratio_movies_allowance (M A : ℕ) 
  (weeklyAllowance moneyEarned endMoney : ℕ)
  (H1 : weeklyAllowance = 8)
  (H2 : moneyEarned = 8)
  (H3 : endMoney = 12)
  (H4 : weeklyAllowance + moneyEarned - M = endMoney)
  (H5 : A = 8)
  (H6 : M = weeklyAllowance + moneyEarned - endMoney / 1) :
  M / A = 1 / 2 :=
sorry

end ratio_movies_allowance_l60_60282


namespace parity_of_f_l60_60728

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

def f (x : ℝ) : ℝ :=
  x * (x - 2) * (x - 1) * x * (x + 1) * (x + 2)

theorem parity_of_f :
  is_even_function f ∧ ¬ (∃ g : ℝ → ℝ, g = f ∧ (∀ x : ℝ, g (-x) = -g x)) :=
by
  sorry

end parity_of_f_l60_60728


namespace retailer_discount_problem_l60_60023

theorem retailer_discount_problem
  (CP MP SP : ℝ) 
  (h1 : CP = 100)
  (h2 : MP = CP + (0.65 * CP))
  (h3 : SP = CP + (0.2375 * CP)) :
  (MP - SP) / MP * 100 = 25 :=
by
  sorry

end retailer_discount_problem_l60_60023


namespace text_messages_in_march_l60_60380

theorem text_messages_in_march
  (nov_texts : ℕ)
  (dec_texts : ℕ)
  (jan_texts : ℕ)
  (feb_texts : ℕ)
  (double_pattern : ∀ n m : ℕ, m = 2 * n)
  (h_nov : nov_texts = 1)
  (h_dec : dec_texts = 2 * nov_texts)
  (h_jan : jan_texts = 2 * dec_texts)
  (h_feb : feb_texts = 2 * jan_texts) : 
  ∃ mar_texts : ℕ, mar_texts = 2 * feb_texts ∧ mar_texts = 16 := 
by
  sorry

end text_messages_in_march_l60_60380


namespace gcd_180_308_l60_60150

theorem gcd_180_308 : Nat.gcd 180 308 = 4 :=
by
  sorry

end gcd_180_308_l60_60150


namespace train_late_average_speed_l60_60461

theorem train_late_average_speed 
  (distance : ℝ) (on_time_speed : ℝ) (late_time_additional : ℝ) 
  (on_time : distance / on_time_speed = 1.75) 
  (late : distance / (on_time_speed * 2/2.5) = 2) :
  distance / 2 = 35 :=
by
  sorry

end train_late_average_speed_l60_60461


namespace exactly_three_assertions_l60_60694

theorem exactly_three_assertions (x : ℕ) : 
  10 ≤ x ∧ x < 100 ∧
  ((x % 3 = 0) ∧ (x % 5 = 0) ∧ (x % 9 ≠ 0) ∧ (x % 15 = 0) ∧ (x % 25 ≠ 0) ∧ (x % 45 ≠ 0)) ↔
  (x = 15 ∨ x = 30 ∨ x = 60) :=
by
  sorry

end exactly_three_assertions_l60_60694


namespace line_circle_no_intersection_l60_60993

theorem line_circle_no_intersection :
  ∀ (x y : ℝ), (5 * x + 8 * y = 10) → ¬ (x^2 + y^2 = 1) :=
by
  intro x y hline hcirc
  -- Proof omitted
  sorry

end line_circle_no_intersection_l60_60993


namespace root_in_interval_l60_60292

noncomputable def f (x : ℝ) : ℝ := x + Real.log x - 3

theorem root_in_interval : ∃ m, f m = 0 ∧ 2 < m ∧ m < 3 :=
by
  sorry

end root_in_interval_l60_60292


namespace h_eq_x_solution_l60_60587

noncomputable def h (x : ℝ) : ℝ := (3 * ((x + 3) / 5) + 10)

theorem h_eq_x_solution (x : ℝ) (h_cond : ∀ y, h (5 * y - 3) = 3 * y + 10) : h x = x → x = 29.5 :=
by
  sorry

end h_eq_x_solution_l60_60587


namespace tan_alpha_plus_beta_tan_beta_l60_60628

variable (α β : ℝ)

-- Given conditions
def tan_condition_1 : Prop := Real.tan (Real.pi + α) = -1 / 3
def tan_condition_2 : Prop := Real.tan (α + β) = (Real.sin α + 2 * Real.cos α) / (5 * Real.cos α - Real.sin α)

-- Proving the results
theorem tan_alpha_plus_beta (h1 : tan_condition_1 α) (h2 : tan_condition_2 α β) : 
  Real.tan (α + β) = 5 / 16 :=
sorry

theorem tan_beta (h1 : tan_condition_1 α) (h2 : tan_condition_2 α β) :
  Real.tan β = 31 / 43 :=
sorry

end tan_alpha_plus_beta_tan_beta_l60_60628


namespace bucket_initial_amount_l60_60230

theorem bucket_initial_amount (A B : ℝ) 
  (h1 : A - 6 = (1 / 3) * (B + 6)) 
  (h2 : B - 6 = (1 / 2) * (A + 6)) : 
  A = 13.2 := 
sorry

end bucket_initial_amount_l60_60230


namespace correct_removal_of_parentheses_l60_60966

theorem correct_removal_of_parentheses (x : ℝ) : (1/3) * (6 * x - 3) = 2 * x - 1 :=
by sorry

end correct_removal_of_parentheses_l60_60966


namespace marts_income_percentage_of_juans_l60_60742

variable (T J M : Real)
variable (h1 : M = 1.60 * T)
variable (h2 : T = 0.40 * J)

theorem marts_income_percentage_of_juans : M = 0.64 * J :=
by
  sorry

end marts_income_percentage_of_juans_l60_60742


namespace wrapping_paper_solution_l60_60520

variable (P1 P2 P3 : ℝ)

def wrapping_paper_problem : Prop :=
  P1 = 2 ∧
  P3 = P1 + P2 ∧
  P1 + P2 + P3 = 7 →
  (P2 / P1) = 3 / 4

theorem wrapping_paper_solution : wrapping_paper_problem P1 P2 P3 :=
by
  sorry

end wrapping_paper_solution_l60_60520


namespace baron_munchausen_correct_l60_60518

noncomputable def P (x : ℕ) : ℕ := sorry -- Assume non-constant polynomial with non-negative integer coefficients
noncomputable def Q (x : ℕ) : ℕ := sorry -- Assume non-constant polynomial with non-negative integer coefficients

theorem baron_munchausen_correct (b p0 : ℕ) 
  (hP2 : P 2 = b) 
  (hPp2 : P b = p0) 
  (hQ2 : Q 2 = b) 
  (hQp2 : Q b = p0) : 
  P = Q := sorry

end baron_munchausen_correct_l60_60518


namespace sin_alpha_cos_alpha_l60_60479

theorem sin_alpha_cos_alpha {α : ℝ} (h : Real.sin (3 * Real.pi - α) = -2 * Real.sin (Real.pi / 2 + α)) :
  Real.sin α * Real.cos α = -2 / 5 :=
by
  sorry

end sin_alpha_cos_alpha_l60_60479


namespace removing_zeros_changes_value_l60_60684

noncomputable def a : ℝ := 7.0800
noncomputable def b : ℝ := 7.8

theorem removing_zeros_changes_value : a ≠ b :=
by
  -- proof goes here
  sorry

end removing_zeros_changes_value_l60_60684


namespace compare_logs_l60_60434

noncomputable def a := Real.log 2 / Real.log 3
noncomputable def b := Real.log 3 / Real.log 5
noncomputable def c := Real.log 5 / Real.log 8

theorem compare_logs : a < b ∧ b < c := by
  sorry

end compare_logs_l60_60434


namespace Nord_Stream_pipeline_payment_l60_60852

/-- Suppose Russia, Germany, and France decided to build the "Nord Stream 2" pipeline,
     which is 1200 km long, agreeing to finance this project equally.
     Russia built 650 kilometers of the pipeline.
     Germany built 550 kilometers of the pipeline.
     France contributed its share in money and did not build any kilometers.
     Germany received 1.2 billion euros from France.
     Prove that Russia should receive 2 billion euros from France.
--/
theorem Nord_Stream_pipeline_payment
  (total_km : ℝ)
  (russia_km : ℝ)
  (germany_km : ℝ)
  (total_countries : ℝ)
  (payment_to_germany : ℝ)
  (germany_additional_payment : ℝ)
  (france_km : ℝ)
  (france_payment_ratio : ℝ)
  (russia_payment : ℝ) :
  total_km = 1200 ∧
  russia_km = 650 ∧
  germany_km = 550 ∧
  total_countries = 3 ∧
  payment_to_germany = 1.2 ∧
  france_km = 0 ∧
  germany_additional_payment = germany_km - (total_km / total_countries) ∧
  france_payment_ratio = 5 / 3 ∧
  russia_payment = payment_to_germany * (5 / 3) →
  russia_payment = 2 := by sorry

end Nord_Stream_pipeline_payment_l60_60852


namespace vertex_of_parabola_l60_60473

theorem vertex_of_parabola : 
  ∀ (x y : ℝ), (y = -x^2 + 3) → (0, 3) ∈ {(h, k) | ∃ (a : ℝ), y = a * (x - h)^2 + k} :=
by
  sorry

end vertex_of_parabola_l60_60473


namespace Greg_gold_amount_l60_60940

noncomputable def gold_amounts (G K : ℕ) : Prop :=
  G = K / 4 ∧ G + K = 100

theorem Greg_gold_amount (G K : ℕ) (h : gold_amounts G K) : G = 20 := 
by
  sorry

end Greg_gold_amount_l60_60940


namespace largest_integer_divides_product_l60_60935

theorem largest_integer_divides_product (n : ℕ) : 
  ∃ m, ∀ k : ℕ, k = (2*n-1)*(2*n)*(2*n+2) → m ≥ 1 ∧ m = 8 ∧ m ∣ k :=
by
  sorry

end largest_integer_divides_product_l60_60935


namespace car_distance_problem_l60_60407

-- A definition for the initial conditions.
def initial_conditions (D : ℝ) (S : ℝ) (T : ℝ) : Prop :=
  T = 6 ∧ S = 50 ∧ (3/2 * T = 9)

-- The statement corresponding to the given problem.
theorem car_distance_problem (D : ℝ) (S : ℝ) (T : ℝ) :
  initial_conditions D S T → D = 450 :=
by
  -- leave the proof as an exercise.
  sorry

end car_distance_problem_l60_60407


namespace find_vector_at_t4_l60_60435

def vector_at (t : ℝ) (a d : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := a
  let (dx, dy, dz) := d
  (x + t * dx, y + t * dy, z + t * dz)

theorem find_vector_at_t4 :
  ∀ (a d : ℝ × ℝ × ℝ),
    vector_at (-2) a d = (2, 6, 16) →
    vector_at 1 a d = (-1, -5, -10) →
    vector_at 4 a d = (-16, -60, -140) :=
by
  intros a d h1 h2
  sorry

end find_vector_at_t4_l60_60435


namespace area_of_triangle_ADE_l60_60013

noncomputable def triangle_area (A B C: ℝ × ℝ) : ℝ :=
  1 / 2 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

theorem area_of_triangle_ADE (A B D E F : ℝ × ℝ) (h₁ : A.1 = 0 ∧ A.2 = 0) (h₂ : B.1 = 8 ∧ B.2 = 0)
  (h₃ : D.1 = 8 ∧ D.2= 8) (h₄ : E.1 = 4 * 3 / 5 ∧ E.2 = 0) 
  (h₅ : F.1 = 0 ∧ F.2 = 12) :
  triangle_area A D E = 288 / 25 := 
sorry

end area_of_triangle_ADE_l60_60013


namespace fundraiser_goal_l60_60688

theorem fundraiser_goal (bronze_donation silver_donation gold_donation goal : ℕ)
  (bronze_families silver_families gold_family : ℕ)
  (H_bronze_amount : bronze_families * bronze_donation = 250)
  (H_silver_amount : silver_families * silver_donation = 350)
  (H_gold_amount : gold_family * gold_donation = 100)
  (H_goal : goal = 750) :
  goal - (bronze_families * bronze_donation + silver_families * silver_donation + gold_family * gold_donation) = 50 :=
by
  sorry

end fundraiser_goal_l60_60688


namespace intersection_of_A_and_B_l60_60792

open Set

def A : Set ℝ := {-1, 0, 1}
def B : Set ℝ := {x | x < 0}

theorem intersection_of_A_and_B : A ∩ B = {-1} :=
  sorry

end intersection_of_A_and_B_l60_60792


namespace original_expenditure_beginning_month_l60_60984

theorem original_expenditure_beginning_month (A E : ℝ)
  (h1 : E = 35 * A)
  (h2 : E + 84 = 42 * (A - 1))
  (h3 : E + 124 = 37 * (A + 1))
  (h4 : E + 154 = 40 * (A + 1)) :
  E = 630 := 
sorry

end original_expenditure_beginning_month_l60_60984


namespace value_of_a_plus_b_minus_c_l60_60693

theorem value_of_a_plus_b_minus_c (a b c : ℝ) 
  (h1 : abs a = 1) 
  (h2 : abs b = 2) 
  (h3 : abs c = 3) 
  (h4 : a > b) 
  (h5 : b > c) : 
  a + b - c = 2 := 
sorry

end value_of_a_plus_b_minus_c_l60_60693


namespace tan_neg_five_pi_over_three_l60_60682

theorem tan_neg_five_pi_over_three : Real.tan (-5 * Real.pi / 3) = Real.sqrt 3 := 
by 
  sorry

end tan_neg_five_pi_over_three_l60_60682


namespace factorization_identity_l60_60582

theorem factorization_identity (x : ℝ) : 
  16 * x^2 - 40 * x + 25 = (4 * x - 5)^2 :=
  sorry

end factorization_identity_l60_60582


namespace initial_amount_spent_l60_60370

theorem initial_amount_spent (X : ℝ) 
    (h_bread : X - 3 ≥ 0) 
    (h_candy : X - 3 - 2 ≥ 0) 
    (h_turkey : X - 3 - 2 - (1/3) * (X - 3 - 2) ≥ 0) 
    (h_remaining : X - 3 - 2 - (1/3) * (X - 3 - 2) = 18) : X = 32 := 
sorry

end initial_amount_spent_l60_60370


namespace Pria_drove_372_miles_l60_60038

theorem Pria_drove_372_miles (advertisement_mileage : ℕ) (tank_capacity : ℕ) (mileage_difference : ℕ) 
(h1 : advertisement_mileage = 35) 
(h2 : tank_capacity = 12) 
(h3 : mileage_difference = 4) : 
(advertisement_mileage - mileage_difference) * tank_capacity = 372 :=
by sorry

end Pria_drove_372_miles_l60_60038


namespace smallest_perimeter_iso_triangle_l60_60723

theorem smallest_perimeter_iso_triangle :
  ∃ (x y : ℕ), (PQ = PR ∧ PQ = x ∧ PR = x ∧ QR = y ∧ QJ = 10 ∧ PQ + PR + QR = 416 ∧ 
  PQ = PR ∧ y = 8 ∧ 2 * (x + y) = 416 ∧ y^2 - 50 > 0 ∧ y < 10) :=
sorry

end smallest_perimeter_iso_triangle_l60_60723


namespace sum_of_fraction_numerator_and_denominator_l60_60530

theorem sum_of_fraction_numerator_and_denominator (x : ℚ) (a b : ℤ) :
  x = (45 / 99 : ℚ) ∧ (a = 5) ∧ (b = 11) → (a + b = 16) :=
by
  sorry

end sum_of_fraction_numerator_and_denominator_l60_60530


namespace triathlon_minimum_speeds_l60_60399

theorem triathlon_minimum_speeds (x : ℝ) (T : ℝ := 80) (total_time : ℝ := (800 / x + 20000 / (7.5 * x) + 4000 / (3 * x))) :
  total_time ≤ T → x ≥ 60 ∧ 3 * x = 180 ∧ 7.5 * x = 450 :=
by
  sorry

end triathlon_minimum_speeds_l60_60399


namespace company_a_taxis_l60_60054

variable (a b : ℕ)

theorem company_a_taxis
  (h1 : 5 * a < 56)
  (h2 : 6 * a > 56)
  (h3 : 4 * b < 56)
  (h4 : 5 * b > 56)
  (h5 : b = a + 3) :
  a = 10 := by
  sorry

end company_a_taxis_l60_60054


namespace regular_eqn_exists_l60_60596

noncomputable def parametric_eqs (k : ℝ) : ℝ × ℝ :=
  (4 * k / (1 - k^2), 4 * k^2 / (1 - k^2))

theorem regular_eqn_exists (k : ℝ) (x y : ℝ) (h1 : x = 4 * k / (1 - k^2)) 
(h2 : y = 4 * k^2 / (1 - k^2)) : x^2 - y^2 - 4 * y = 0 :=
sorry

end regular_eqn_exists_l60_60596


namespace avg_weight_B_correct_l60_60805

-- Definitions of the conditions
def students_A : ℕ := 24
def students_B : ℕ := 16
def avg_weight_A : ℝ := 40
def avg_weight_class : ℝ := 38

-- Definition of the total weight calculation for sections A and B
def total_weight_A : ℝ := students_A * avg_weight_A
def total_weight_class : ℝ := (students_A + students_B) * avg_weight_class

-- Defining the average weight of section B as the unknown to be proven
noncomputable def avg_weight_B : ℝ := 35

-- The theorem to prove that the average weight of section B is 35 kg
theorem avg_weight_B_correct : 
  total_weight_A + students_B * avg_weight_B = total_weight_class :=
by
  sorry

end avg_weight_B_correct_l60_60805


namespace doubles_tournament_handshakes_l60_60421

theorem doubles_tournament_handshakes :
  let num_teams := 3
  let players_per_team := 2
  let total_players := num_teams * players_per_team
  let handshakes_per_player := total_players - 2
  let total_handshakes := total_players * handshakes_per_player / 2
  total_handshakes = 12 :=
by
  sorry

end doubles_tournament_handshakes_l60_60421


namespace find_k_l60_60631

-- Define the sequence and its sum
def Sn (k : ℝ) (n : ℕ) : ℝ := k + 3^n
def an (k : ℝ) (n : ℕ) : ℝ := Sn k n - (if n = 0 then 0 else Sn k (n - 1))

-- Define the condition that a sequence is geometric
def is_geometric (a : ℕ → ℝ) (r : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = r * a n

theorem find_k (k : ℝ) :
  is_geometric (an k) (an k 1 / an k 0) → k = -1 := 
by sorry

end find_k_l60_60631


namespace fractions_equivalent_under_scaling_l60_60691

theorem fractions_equivalent_under_scaling (a b d k x : ℝ) (h₀ : d ≠ 0) (h₁ : k ≠ 0) :
  (a * (k * x) + b) / (a * (k * x) + d) = (b * (k * x)) / (d * (k * x)) ↔ b = d :=
by sorry

end fractions_equivalent_under_scaling_l60_60691


namespace general_term_sequence_l60_60774

theorem general_term_sequence (a : ℕ → ℕ) (h₁ : a 1 = 1) (h₂ : ∀ n, a (n + 1) = a n + 3^n) :
  ∀ n, a n = (3^n - 1) / 2 := 
by
  sorry

end general_term_sequence_l60_60774


namespace roger_allowance_fraction_l60_60804

noncomputable def allowance_fraction (A m s p : ℝ) : ℝ :=
  m + s + p

theorem roger_allowance_fraction (A : ℝ) (m s p : ℝ) 
  (h_movie : m = 0.25 * (A - s - p))
  (h_soda : s = 0.10 * (A - m - p))
  (h_popcorn : p = 0.05 * (A - m - s)) :
  allowance_fraction A m s p = 0.32 * A :=
by
  sorry

end roger_allowance_fraction_l60_60804


namespace no_such_positive_integer_l60_60298

theorem no_such_positive_integer (n : ℕ) (d : ℕ → ℕ)
  (h₁ : ∃ d1 d2 d3 d4 d5, d 1 = d1 ∧ d 2 = d2 ∧ d 3 = d3 ∧ d 4 = d4 ∧ d 5 = d5) 
  (h₂ : 1 ≤ d 1 ∧ d 1 < d 2 ∧ d 2 < d 3 ∧ d 3 < d 4 ∧ d 4 < d 5)
  (h₃ : ∀ i, 1 ≤ i → i ≤ 5 → d i ∣ n)
  (h₄ : ∀ i, 1 ≤ i → i ≤ 5 → ∀ j, i ≠ j → d i ≠ d j)
  (h₅ : ∃ x, 1 + (d 2)^2 + (d 3)^2 + (d 4)^2 + (d 5)^2 = x^2) :
  false :=
sorry

end no_such_positive_integer_l60_60298


namespace red_mushrooms_bill_l60_60102

theorem red_mushrooms_bill (R : ℝ) : 
  (2/3) * R + 6 + 3 = 17 → R = 12 :=
by
  intro h
  sorry

end red_mushrooms_bill_l60_60102


namespace albums_created_l60_60332

def phone_pics : ℕ := 2
def camera_pics : ℕ := 4
def pics_per_album : ℕ := 2
def total_pics : ℕ := phone_pics + camera_pics

theorem albums_created : total_pics / pics_per_album = 3 := by
  sorry

end albums_created_l60_60332


namespace fractional_inequality_solution_l60_60149

theorem fractional_inequality_solution :
  ∃ (m n : ℕ), n = m^2 - 1 ∧ 
               (m + 2) / (n + 2 : ℝ) > 1 / 3 ∧ 
               (m - 3) / (n - 3 : ℝ) < 1 / 10 ∧ 
               1 ≤ m ∧ m ≤ 9 ∧ 1 ≤ n ∧ n ≤ 9 ∧ 
               (m = 3) ∧ (n = 8) := 
by
  sorry

end fractional_inequality_solution_l60_60149


namespace correct_calculated_value_l60_60343

theorem correct_calculated_value (x : ℤ) 
  (h : x / 16 = 8 ∧ x % 16 = 4) : (x * 16 + 8 = 2120) := by
  sorry

end correct_calculated_value_l60_60343


namespace max_projection_area_of_tetrahedron_l60_60210

theorem max_projection_area_of_tetrahedron (a : ℝ) (h1 : a > 0) :
  ∃ (A : ℝ), (A = a^2 / 2) :=
by
  sorry

end max_projection_area_of_tetrahedron_l60_60210


namespace calculate_expression_l60_60968

theorem calculate_expression :
  150 * (150 - 4) - (150 * 150 - 8 + 2^3) = -600 :=
by
  sorry

end calculate_expression_l60_60968


namespace dan_has_remaining_cards_l60_60129

-- Define the initial conditions
def initial_cards : ℕ := 97
def cards_sold_to_sam : ℕ := 15

-- Define the expected result
def remaining_cards (initial : ℕ) (sold : ℕ) : ℕ := initial - sold

-- State the theorem to prove
theorem dan_has_remaining_cards : remaining_cards initial_cards cards_sold_to_sam = 82 :=
by
  -- This insertion is a placeholder for the proof
  sorry

end dan_has_remaining_cards_l60_60129


namespace simplify_sqrt_expression_correct_l60_60307

noncomputable def simplify_sqrt_expression (m : ℝ) (h_triangle : (2 < m + 5) ∧ (m < 2 + 5) ∧ (5 < 2 + m)) : ℝ :=
  (Real.sqrt (9 - 6 * m + m^2)) - (Real.sqrt (m^2 - 14 * m + 49))

theorem simplify_sqrt_expression_correct (m : ℝ) (h_triangle : (2 < m + 5) ∧ (m < 2 + 5) ∧ (5 < 2 + m)) :
  simplify_sqrt_expression m h_triangle = 2 * m - 10 :=
sorry

end simplify_sqrt_expression_correct_l60_60307


namespace quadratic_identity_l60_60789

theorem quadratic_identity
  (a b c x : ℝ) (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) :
  (a^2 * (x - b) * (x - c) / ((a - b) * (a - c))) +
  (b^2 * (x - a) * (x - c) / ((b - a) * (b - c))) +
  (c^2 * (x - a) * (x - b) / ((c - a) * (c - b))) =
  x^2 :=
sorry

end quadratic_identity_l60_60789


namespace remainder_when_divided_by_13_l60_60100

theorem remainder_when_divided_by_13 (N : ℕ) (k : ℕ) (hk : N = 39 * k + 15) : N % 13 = 2 :=
sorry

end remainder_when_divided_by_13_l60_60100


namespace max_divisor_f_l60_60732

def f (n : ℕ) : ℕ := (2 * n + 7) * 3^n + 9

theorem max_divisor_f (m : ℕ) : (∀ n : ℕ, m ∣ f n) → m = 36 :=
sorry

end max_divisor_f_l60_60732


namespace votes_cast_is_750_l60_60803

-- Define the conditions as Lean statements
def initial_score : ℤ := 0
def score_increase (likes : ℕ) : ℤ := likes
def score_decrease (dislikes : ℕ) : ℤ := -dislikes
def observed_score : ℤ := 150
def percent_likes : ℚ := 0.60

-- Express the proof
theorem votes_cast_is_750 (total_votes : ℕ) (likes : ℕ) (dislikes : ℕ) 
  (h1 : total_votes = likes + dislikes) 
  (h2 : percent_likes * total_votes = likes) 
  (h3 : dislikes = (1 - percent_likes) * total_votes)
  (h4 : observed_score = score_increase likes + score_decrease dislikes) :
  total_votes = 750 := 
sorry

end votes_cast_is_750_l60_60803


namespace find_c_l60_60568

theorem find_c (c : ℝ) (h : ∃ β : ℝ, (5 + β = -c) ∧ (5 * β = 45)) : c = -14 := 
  sorry

end find_c_l60_60568


namespace g_600_l60_60112

def g : ℕ → ℕ := sorry

axiom g_mul (x y : ℕ) (hx : x > 0) (hy : y > 0) : g (x * y) = g x + g y
axiom g_12 : g 12 = 18
axiom g_48 : g 48 = 26

theorem g_600 : g 600 = 36 :=
by 
  sorry

end g_600_l60_60112


namespace probability_properties_l60_60050

noncomputable def P1 : ℝ := 1 / 4
noncomputable def P2 : ℝ := 1 / 4
noncomputable def P3 : ℝ := 1 / 2

theorem probability_properties :
  (P1 ≠ P3) ∧
  (P1 + P2 = P3) ∧
  (P1 + P2 + P3 = 1) ∧
  (P3 = 2 * P1) ∧
  (P3 = 2 * P2) :=
by
  sorry

end probability_properties_l60_60050


namespace product_base9_l60_60533

open Nat

noncomputable def base9_product (a b : ℕ) : ℕ := 
  let a_base10 := 3*9^2 + 6*9^1 + 2*9^0
  let b_base10 := 7
  let product_base10 := a_base10 * b_base10
  -- converting product_base10 from base 10 to base 9
  2 * 9^3 + 8 * 9^2 + 7 * 9^1 + 5 * 9^0 -- which simplifies to 2875 in base 9

theorem product_base9: base9_product 362 7 = 2875 :=
by
  -- Here should be the proof or a computational check
  sorry

end product_base9_l60_60533


namespace find_k_for_circle_radius_5_l60_60413

theorem find_k_for_circle_radius_5 (k : ℝ) :
  (∃ x y : ℝ, (x^2 + 12 * x + y^2 + 8 * y - k = 0)) → k = -27 :=
by
  sorry

end find_k_for_circle_radius_5_l60_60413


namespace rotations_needed_to_reach_goal_l60_60511

-- Define the given conditions
def rotations_per_block : ℕ := 200
def blocks_goal : ℕ := 8
def current_rotations : ℕ := 600

-- Define total_rotations_needed and more_rotations_needed
def total_rotations_needed : ℕ := blocks_goal * rotations_per_block
def more_rotations_needed : ℕ := total_rotations_needed - current_rotations

-- Theorem stating the solution
theorem rotations_needed_to_reach_goal : more_rotations_needed = 1000 := by
  -- proof steps are omitted
  sorry

end rotations_needed_to_reach_goal_l60_60511


namespace find_x_l60_60727

theorem find_x (x : ℕ) (h1 : x % 6 = 0) (h2 : x^2 > 144) (h3 : x < 30) : x = 18 ∨ x = 24 :=
sorry

end find_x_l60_60727


namespace wendys_brother_pieces_l60_60897

-- Definitions based on conditions
def number_of_boxes : ℕ := 2
def pieces_per_box : ℕ := 3
def total_pieces : ℕ := 12

-- Summarization of Wendy's pieces of candy
def wendys_pieces : ℕ := number_of_boxes * pieces_per_box

-- Lean statement: Prove the number of pieces Wendy's brother had
theorem wendys_brother_pieces : total_pieces - wendys_pieces = 6 :=
by
  sorry

end wendys_brother_pieces_l60_60897


namespace series_result_l60_60208

noncomputable def series_sum (u : ℕ → ℚ) (s : ℚ) : Prop :=
  ∑' n, u n = s

def nth_term (n : ℕ) : ℚ := (4 * n + 3) / ((4 * n + 1)^2 * (4 * n + 5)^2)

theorem series_result : series_sum nth_term (1 / 200) := by
  sorry

end series_result_l60_60208


namespace percentage_of_employees_driving_l60_60635

theorem percentage_of_employees_driving
  (total_employees : ℕ)
  (drivers : ℕ)
  (public_transport : ℕ)
  (H1 : total_employees = 200)
  (H2 : drivers = public_transport + 40)
  (H3 : public_transport = (total_employees - drivers) / 2) :
  (drivers:ℝ) / (total_employees:ℝ) * 100 = 46.5 :=
by {
  sorry
}

end percentage_of_employees_driving_l60_60635


namespace scientific_notation_of_tourists_l60_60131

theorem scientific_notation_of_tourists : 
  (23766400 : ℝ) = 2.37664 * 10^7 :=
by 
  sorry

end scientific_notation_of_tourists_l60_60131


namespace M_eq_N_l60_60083

def M : Set ℝ := {x | ∃ m : ℤ, x = Real.sin ((2 * m - 3) * Real.pi / 6)}
def N : Set ℝ := {y | ∃ n : ℤ, y = Real.cos (n * Real.pi / 3)}

theorem M_eq_N : M = N := 
by 
  sorry

end M_eq_N_l60_60083


namespace handshakes_count_l60_60999

-- Define the parameters
def teams : ℕ := 3
def players_per_team : ℕ := 7
def referees : ℕ := 3

-- Calculate handshakes among team members
def handshakes_among_teams :=
  let unique_handshakes_per_team := players_per_team * 2 * players_per_team / 2
  unique_handshakes_per_team * teams

-- Calculate handshakes between players and referees
def players_shake_hands_with_referees :=
  teams * players_per_team * referees

-- Calculate total handshakes
def total_handshakes :=
  handshakes_among_teams + players_shake_hands_with_referees

-- Proof statement
theorem handshakes_count : total_handshakes = 210 := by
  sorry

end handshakes_count_l60_60999


namespace abs_fraction_eq_sqrt_seven_thirds_l60_60827

open Real

theorem abs_fraction_eq_sqrt_seven_thirds {a b : ℝ} 
  (a_nonzero : a ≠ 0) (b_nonzero : b ≠ 0) (h : a^2 + b^2 = 5 * a * b) : 
  abs ((a + b) / (a - b)) = sqrt (7 / 3) :=
by
  sorry

end abs_fraction_eq_sqrt_seven_thirds_l60_60827


namespace algebraic_expression_value_l60_60680

open Real

theorem algebraic_expression_value
  (θ : ℝ)
  (a := (cos θ, sin θ))
  (b := (1, -2))
  (h_parallel : a.1 * b.2 = a.2 * b.1) :
  (2 * sin θ - cos θ) / (sin θ + cos θ) = 5 :=
by
  sorry

end algebraic_expression_value_l60_60680


namespace alpha_numeric_puzzle_l60_60873

theorem alpha_numeric_puzzle : 
  ∀ (a b c d e f g h i : ℕ),
  (∀ x y : ℕ, x ≠ 0 → y ≠ 0 → x ≠ y) →
  100 * a + 10 * b + c + 100 * d + 10 * e + f + 100 * g + 10 * h + i = 1665 → 
  c + f + i = 15 →
  b + e + h = 15 :=
by
  intros a b c d e f g h i distinct nonzero_sum unit_digits_sum
  sorry

end alpha_numeric_puzzle_l60_60873


namespace no_cracked_seashells_l60_60485

theorem no_cracked_seashells (tom_seashells : ℕ) (fred_seashells : ℕ) (total_seashells : ℕ)
  (h1 : tom_seashells = 15) (h2 : fred_seashells = 43) (h3 : total_seashells = 58)
  (h4 : tom_seashells + fred_seashells = total_seashells) : 
  (total_seashells - (tom_seashells + fred_seashells) = 0) :=
by
  sorry

end no_cracked_seashells_l60_60485


namespace count_valid_n_le_30_l60_60994

theorem count_valid_n_le_30 :
  ∀ n : ℕ, (0 < n ∧ n ≤ 30) → (n! * 2) % (n * (n + 1)) = 0 := by
  sorry

end count_valid_n_le_30_l60_60994


namespace set_intersection_l60_60330

open Set

variable (x : ℝ)

def U : Set ℝ := univ
def A : Set ℝ := { x | |x - 1| > 2 }
def B : Set ℝ := { x | x^2 - 6 * x + 8 < 0 }

theorem set_intersection (x : ℝ) : x ∈ (U \ A) ∩ B ↔ 2 < x ∧ x ≤ 3 := sorry

end set_intersection_l60_60330


namespace expression_divisible_by_84_l60_60281

theorem expression_divisible_by_84 (p : ℕ) (hp : p > 0) : (4 ^ (2 * p) - 3 ^ (2 * p) - 7) % 84 = 0 :=
by
  sorry

end expression_divisible_by_84_l60_60281


namespace length_of_train_l60_60255

-- We define the conditions
def crosses_platform_1 (L : ℝ) : Prop := 
  let v := (L + 100) / 15
  v = (L + 100) / 15

def crosses_platform_2 (L : ℝ) : Prop := 
  let v := (L + 250) / 20
  v = (L + 250) / 20

-- We state the main theorem we need to prove
theorem length_of_train :
  ∃ L : ℝ, crosses_platform_1 L ∧ crosses_platform_2 L ∧ (L = 350) :=
sorry

end length_of_train_l60_60255


namespace area_of_estate_l60_60065

theorem area_of_estate (side_length_in_inches : ℝ) (scale : ℝ) (real_side_length : ℝ) (area : ℝ) :
  side_length_in_inches = 12 →
  scale = 100 →
  real_side_length = side_length_in_inches * scale →
  area = real_side_length ^ 2 →
  area = 1440000 :=
by
  sorry

end area_of_estate_l60_60065


namespace square_area_l60_60235

theorem square_area (x : ℝ) (side_length : ℝ) 
  (h1_side_length : side_length = 5 * x - 10)
  (h2_side_length : side_length = 3 * (x + 4)) :
  side_length ^ 2 = 2025 :=
by
  sorry

end square_area_l60_60235


namespace compare_sums_l60_60579

theorem compare_sums (a b c : ℝ) (h : a > b ∧ b > c) : a^2 * b + b^2 * c + c^2 * a > a * b^2 + b * c^2 + c * a^2 := by
  sorry

end compare_sums_l60_60579


namespace vasya_numbers_l60_60832

theorem vasya_numbers : ∀ (x y : ℝ), 
  (x + y = x * y) ∧ (x * y = x / y) → (x = 1/2 ∧ y = -1) :=
by
  intros x y h
  sorry

end vasya_numbers_l60_60832


namespace inverse_proportionality_example_l60_60396

theorem inverse_proportionality_example (k : ℝ) (x : ℝ) (y : ℝ) (h1 : 5 * 10 = k) (h2 : x * 40 = k) : x = 5 / 4 :=
by
  -- sorry is used to skip the proof.
  sorry

end inverse_proportionality_example_l60_60396


namespace tangent_line_eq_l60_60283

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + 1

theorem tangent_line_eq
  (a b : ℝ)
  (h1 : 3 + 2*a + b = 2*a)
  (h2 : 12 + 4*a + b = -b)
  : ∀ x y : ℝ , (f a b 1 = -5/2 ∧
  y - (f a b 1) = -3 * (x - 1))
  → (6*x + 2*y - 1 = 0) :=
by
  sorry

end tangent_line_eq_l60_60283


namespace books_cost_l60_60028

theorem books_cost (total_cost_three_books cost_seven_books : ℕ) 
  (h₁ : total_cost_three_books = 45)
  (h₂ : cost_seven_books = 7 * (total_cost_three_books / 3)) : 
  cost_seven_books = 105 :=
  sorry

end books_cost_l60_60028


namespace find_b_in_triangle_l60_60187

theorem find_b_in_triangle (c : ℝ) (B C : ℝ) (h1 : c = Real.sqrt 3)
  (h2 : B = Real.pi / 4) (h3 : C = Real.pi / 3) : ∃ b : ℝ, b = Real.sqrt 2 :=
by
  sorry

end find_b_in_triangle_l60_60187


namespace sum_x_y_z_l60_60481

theorem sum_x_y_z (a b : ℝ) (x y z : ℕ) 
  (h_a : a^2 = 16 / 44) 
  (h_b : b^2 = (2 + Real.sqrt 5)^2 / 11) 
  (h_a_neg : a < 0) 
  (h_b_pos : b > 0) 
  (h_expr : (a + b)^3 = x * Real.sqrt y / z) : 
  x + y + z = 181 := 
sorry

end sum_x_y_z_l60_60481


namespace Diana_total_earnings_l60_60844

def July : ℝ := 150
def August : ℝ := 3 * July
def September : ℝ := 2 * August
def October : ℝ := September + 0.1 * September
def November : ℝ := 0.95 * October
def Total_earnings : ℝ := July + August + September + October + November

theorem Diana_total_earnings : Total_earnings = 3430.50 := by
  sorry

end Diana_total_earnings_l60_60844


namespace largest_prime_divisor_of_1202102_5_l60_60466

def base_5_to_decimal (n : String) : ℕ := 
  let digits := n.toList.map (λ c => c.toNat - '0'.toNat)
  digits.foldr (λ (digit acc : ℕ) => acc * 5 + digit) 0

def largest_prime_factor (n : ℕ) : ℕ := sorry -- Placeholder for the actual factorization logic.

theorem largest_prime_divisor_of_1202102_5 : 
  largest_prime_factor (base_5_to_decimal "1202102") = 307 := 
sorry

end largest_prime_divisor_of_1202102_5_l60_60466


namespace find_x_l60_60180

def vector_add (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 + v.1, u.2 + v.2)
def vectors_parallel (u v : ℝ × ℝ) : Prop := ∃ k : ℝ, (u.1 * k = v.1) ∧ (u.2 * k = v.2)

theorem find_x :
  let a := (1, -2)
  let b := (3, -1)
  let c := (x, 4)
  vectors_parallel (vector_add a c) (vector_add b c) → x = 3 :=
by intros; sorry

end find_x_l60_60180


namespace min_colors_5x5_grid_l60_60260

def is_valid_coloring (grid : Fin 5 × Fin 5 → ℕ) (k : ℕ) : Prop :=
  ∀ i j : Fin 5, ∀ di dj : Fin 2, ∀ c : ℕ,
    (di ≠ 0 ∨ dj ≠ 0) →
    (grid (i, j) = c ∧ grid (i + di, j + dj) = c ∧ grid (i + 2 * di, j + 2 * dj) = c) → 
    False

theorem min_colors_5x5_grid : 
  ∀ (grid : Fin 5 × Fin 5 → ℕ), (∀ i j, grid (i, j) < 3) → is_valid_coloring grid 3 := 
by
  sorry

end min_colors_5x5_grid_l60_60260


namespace average_runs_in_second_set_l60_60938

theorem average_runs_in_second_set
  (avg_first_set : ℕ → ℕ → ℕ)
  (avg_all_matches : ℕ → ℕ → ℕ)
  (avg1 : ℕ := avg_first_set 20 30)
  (avg2 : ℕ := avg_all_matches 30 25) :
  ∃ (A : ℕ), A = 15 := by
  sorry

end average_runs_in_second_set_l60_60938


namespace jerry_total_bill_l60_60749

-- Definitions for the initial bill and late fees
def initial_bill : ℝ := 250
def first_fee_rate : ℝ := 0.02
def second_fee_rate : ℝ := 0.03

-- Function to calculate the total bill after applying the fees
def total_bill (init : ℝ) (rate1 : ℝ) (rate2 : ℝ) : ℝ :=
  let first_total := init * (1 + rate1)
  first_total * (1 + rate2)

-- Theorem statement
theorem jerry_total_bill : total_bill initial_bill first_fee_rate second_fee_rate = 262.65 := by
  sorry

end jerry_total_bill_l60_60749


namespace part_a_exists_part_b_impossible_l60_60376

def gridSize : Nat := 7 * 14
def cellCount (x y : Nat) : Nat := 4 * x + 3 * y
def x_equals_y_condition (x y : Nat) : Prop := x = y
def x_greater_y_condition (x y : Nat) : Prop := x > y

theorem part_a_exists (x y : Nat) (h : cellCount x y = gridSize) : ∃ (x y : Nat), x_equals_y_condition x y ∧ cellCount x y = gridSize :=
by
  sorry

theorem part_b_impossible (x y : Nat) (h : cellCount x y = gridSize) : ¬ ∃ (x y : Nat), x_greater_y_condition x y ∧ cellCount x y = gridSize :=
by
  sorry


end part_a_exists_part_b_impossible_l60_60376


namespace ab_range_l60_60625

variable (a b : ℝ)
variable (h_pos_a : 0 < a) (h_pos_b : 0 < b)
variable (h_eq : a * b = a + b + 8)

theorem ab_range (h : a * b = a + b + 8) : 16 ≤ a * b :=
by sorry

end ab_range_l60_60625


namespace zero_in_P_two_not_in_P_l60_60365

variables (P : Set Int)

-- Conditions
def condition_1 := ∃ x ∈ P, x > 0 ∧ ∃ y ∈ P, y < 0
def condition_2 := ∃ x ∈ P, x % 2 = 0 ∧ ∃ y ∈ P, y % 2 ≠ 0 
def condition_3 := 1 ∉ P
def condition_4 := ∀ x y, x ∈ P → y ∈ P → x + y ∈ P

-- Proving 0 ∈ P
theorem zero_in_P (h1 : condition_1 P) (h2 : condition_2 P) (h3 : condition_3 P) (h4 : condition_4 P) : 0 ∈ P := 
sorry

-- Proving 2 ∉ P
theorem two_not_in_P (h1 : condition_1 P) (h2 : condition_2 P) (h3 : condition_3 P) (h4 : condition_4 P) : 2 ∉ P := 
sorry

end zero_in_P_two_not_in_P_l60_60365


namespace original_costs_l60_60325

theorem original_costs (P_old P_second_oldest : ℝ) (h1 : 0.9 * P_old = 1800) (h2 : 0.85 * P_second_oldest = 900) :
  P_old + P_second_oldest = 3058.82 :=
by sorry

end original_costs_l60_60325


namespace Carol_mother_carrots_l60_60750

theorem Carol_mother_carrots (carol_picked : ℕ) (total_good : ℕ) (total_bad : ℕ) (total_carrots : ℕ) (mother_picked : ℕ) :
  carol_picked = 29 → total_good = 38 → total_bad = 7 → total_carrots = total_good + total_bad → mother_picked = total_carrots - carol_picked → mother_picked = 16 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3] at *
  sorry

end Carol_mother_carrots_l60_60750


namespace average_salary_decrease_l60_60047

theorem average_salary_decrease 
    (avg_wage_illiterate_initial : ℝ)
    (avg_wage_illiterate_new : ℝ)
    (num_illiterate : ℕ)
    (num_literate : ℕ)
    (num_total : ℕ)
    (total_decrease : ℝ) :
    avg_wage_illiterate_initial = 25 →
    avg_wage_illiterate_new = 10 →
    num_illiterate = 20 →
    num_literate = 10 →
    num_total = num_illiterate + num_literate →
    total_decrease = (avg_wage_illiterate_initial - avg_wage_illiterate_new) * num_illiterate →
    total_decrease / num_total = 10 :=
by
  intros avg_wage_illiterate_initial_eq avg_wage_illiterate_new_eq num_illiterate_eq num_literate_eq num_total_eq total_decrease_eq
  sorry

end average_salary_decrease_l60_60047


namespace total_number_of_flowers_l60_60653

theorem total_number_of_flowers (pots : ℕ) (flowers_per_pot : ℕ) (h_pots : pots = 544) (h_flowers_per_pot : flowers_per_pot = 32) : 
  pots * flowers_per_pot = 17408 := by
  sorry

end total_number_of_flowers_l60_60653


namespace trapezoid_area_l60_60791

-- Geometry setup
variable (outer_area : ℝ) (inner_height_ratio : ℝ)

-- Conditions
def outer_triangle_area := outer_area = 36
def inner_height_to_outer_height := inner_height_ratio = 2 / 3

-- Conclusion: Area of one trapezoid
theorem trapezoid_area (outer_area inner_height_ratio : ℝ) 
  (h_outer : outer_triangle_area outer_area) 
  (h_inner : inner_height_to_outer_height inner_height_ratio) : 
  (outer_area - 16 * Real.sqrt 3) / 3 = (36 - 16 * Real.sqrt 3) / 3 := 
sorry

end trapezoid_area_l60_60791


namespace range_of_m_if_solution_set_empty_solve_inequality_y_geq_m_l60_60074

noncomputable def quadratic_function (m x : ℝ) : ℝ :=
  (m + 1) * x^2 - m * x + m - 1

-- Part 1
theorem range_of_m_if_solution_set_empty (m : ℝ) :
  (∀ x : ℝ, quadratic_function m x < 0 → false) ↔ m ≥ 2 * Real.sqrt 3 / 3 := sorry

-- Part 2
theorem solve_inequality_y_geq_m (m x : ℝ) (h : m > -2) :
  (quadratic_function m x ≥ m) ↔ 
  (m = -1 → x ≥ 1) ∧
  (m > -1 → x ≤ -1/(m+1) ∨ x ≥ 1) ∧
  (m > -2 ∧ m < -1 → 1 ≤ x ∧ x ≤ -1/(m+1)) := sorry

end range_of_m_if_solution_set_empty_solve_inequality_y_geq_m_l60_60074


namespace problem_statement_l60_60686

theorem problem_statement (x : ℝ) (h : x + 1/x = 3) : x^2 + 1/x^2 = 7 :=
by
  sorry

end problem_statement_l60_60686


namespace most_probable_hits_l60_60734

theorem most_probable_hits (p : ℝ) (q : ℝ) (k0 : ℕ) (n : ℤ) 
  (h1 : p = 0.7) (h2 : q = 1 - p) (h3 : k0 = 16) 
  (h4 : 21 < (n : ℝ) * 0.7) (h5 : (n : ℝ) * 0.7 < 23.3) : 
  n = 22 ∨ n = 23 :=
sorry

end most_probable_hits_l60_60734


namespace marquita_garden_width_l60_60081

theorem marquita_garden_width
  (mancino_gardens : ℕ) (marquita_gardens : ℕ)
  (mancino_length mancnio_width marquita_length total_area : ℕ)
  (h1 : mancino_gardens = 3)
  (h2 : mancino_length = 16)
  (h3 : mancnio_width = 5)
  (h4 : marquita_gardens = 2)
  (h5 : marquita_length = 8)
  (h6 : total_area = 304) :
  ∃ (marquita_width : ℕ), marquita_width = 4 :=
by
  sorry

end marquita_garden_width_l60_60081


namespace anie_days_to_finish_task_l60_60540

def extra_hours : ℕ := 5
def normal_work_hours : ℕ := 10
def total_project_hours : ℕ := 1500

theorem anie_days_to_finish_task : (total_project_hours / (normal_work_hours + extra_hours)) = 100 :=
by
  sorry

end anie_days_to_finish_task_l60_60540


namespace diameter_of_circular_field_l60_60816

theorem diameter_of_circular_field :
  ∀ (π : ℝ) (cost_per_meter total_cost circumference diameter : ℝ),
    π = Real.pi → 
    cost_per_meter = 1.50 → 
    total_cost = 94.24777960769379 → 
    circumference = total_cost / cost_per_meter →
    circumference = π * diameter →
    diameter = 20 := 
by
  intros π cost_per_meter total_cost circumference diameter hπ hcp ht cutoff_circ hcirc
  sorry

end diameter_of_circular_field_l60_60816


namespace lara_cookies_l60_60842

theorem lara_cookies (total_cookies trays rows_per_row : ℕ)
  (h_total : total_cookies = 120)
  (h_trays : trays = 4)
  (h_rows_per_row : rows_per_row = 6) :
  total_cookies / rows_per_row / trays = 5 :=
by
  sorry

end lara_cookies_l60_60842


namespace algebraic_expression_value_l60_60575

theorem algebraic_expression_value (x : ℝ) (h : (x^2 - x)^2 - 4 * (x^2 - x) - 12 = 0) : x^2 - x + 1 = 7 :=
sorry

end algebraic_expression_value_l60_60575


namespace find_m_when_lines_parallel_l60_60821

theorem find_m_when_lines_parallel (m : ℝ) :
  (∀ x y : ℝ, x + (1 + m) * y = 2 - m) ∧ (∀ x y : ℝ, 2 * m * x + 4 * y = -16) →
  ∃ m : ℝ, m = 1 :=
sorry

end find_m_when_lines_parallel_l60_60821


namespace parabolas_intersect_with_high_probability_l60_60601

noncomputable def high_probability_of_intersection : Prop :=
  ∀ (a b c d : ℕ), 1 ≤ a ∧ a ≤ 6 ∧ 1 ≤ b ∧ b ≤ 6 ∧ 1 ≤ c ∧ c ≤ 6 ∧ 1 ≤ d ∧ d ≤ 6 →
  (a - c) ^ 2 + 4 * (b - d) >= 0

theorem parabolas_intersect_with_high_probability : high_probability_of_intersection := sorry

end parabolas_intersect_with_high_probability_l60_60601


namespace square_of_binomial_is_25_l60_60034

theorem square_of_binomial_is_25 (a : ℝ)
  (h : ∃ b : ℝ, (4 * (x : ℝ) + b)^2 = 16 * x^2 + 40 * x + a) : a = 25 :=
sorry

end square_of_binomial_is_25_l60_60034


namespace fencing_required_l60_60617

theorem fencing_required (L W : ℝ) (h1 : L = 40) (h2 : L * W = 680) : 2 * W + L = 74 :=
by
  sorry

end fencing_required_l60_60617


namespace min_value_of_T_l60_60763

noncomputable def T_min_value (a b c : ℝ) : ℝ :=
  (5 + 2*a*b + 4*a*c) / (a*b + 1)

theorem min_value_of_T :
  ∀ (a b c : ℝ),
  a < 0 →
  b > 0 →
  b^2 ≤ (4 * c) / a →
  c ≤ (1/4) * a * b^2 →
  T_min_value a b c ≥ 4 ∧ (T_min_value a b c = 4 ↔ a * b = -3) :=
by
  intros
  sorry

end min_value_of_T_l60_60763


namespace solve_inequalities_l60_60988

theorem solve_inequalities (a b : ℝ) :
  (∀ x : ℝ, -1 < x ∧ x < 3 → x - a < 1 ∧ x - 2 * b > 3) ↔ (a = 2 ∧ b = -2) := 
  by 
    sorry

end solve_inequalities_l60_60988


namespace tangerines_more_than_oranges_l60_60173

def initial_oranges := 5
def initial_tangerines := 17
def oranges_taken := 2
def tangerines_taken := 10

theorem tangerines_more_than_oranges
  (initial_oranges: ℕ) -- Tina starts with 5 oranges
  (initial_tangerines: ℕ) -- Tina starts with 17 tangerines
  (oranges_taken: ℕ) -- Tina takes away 2 oranges
  (tangerines_taken: ℕ) -- Tina takes away 10 tangerines
  : (initial_tangerines - tangerines_taken) - (initial_oranges - oranges_taken) = 4 := 
by
  sorry

end tangerines_more_than_oranges_l60_60173


namespace binary_addition_l60_60706

theorem binary_addition :
  (0b1101 : Nat) + 0b101 + 0b1110 + 0b111 + 0b1010 = 0b10101 := by
  sorry

end binary_addition_l60_60706


namespace new_cube_weight_l60_60737

-- Define the weight function for a cube given side length and density.
def weight (ρ : ℝ) (s : ℝ) : ℝ := ρ * s^3

-- Given conditions: the weight of the original cube.
axiom original_weight : ∃ ρ s : ℝ, weight ρ s = 7

-- The goal is to prove that a new cube with sides twice as long weighs 56 pounds.
theorem new_cube_weight : 
  (∃ ρ s : ℝ, weight ρ (2 * s) = 56) := by
  sorry

end new_cube_weight_l60_60737


namespace find_a_for_opposite_roots_l60_60455

-- Define the equation and condition using the given problem details
theorem find_a_for_opposite_roots (a : ℝ) 
  (h : ∀ (x : ℝ), x^2 - (a^2 - 2 * a - 15) * x + a - 1 = 0 
    → (∃! (x1 x2 : ℝ), x1 + x2 = 0)) :
  a = -3 := 
sorry

end find_a_for_opposite_roots_l60_60455


namespace parametric_eq_of_curve_C_max_x_plus_y_on_curve_C_l60_60454

noncomputable def curve_C (θ : ℝ) : ℝ × ℝ := 
(2 + Real.sqrt 2 * Real.cos θ, 
 2 + Real.sqrt 2 * Real.sin θ)

theorem parametric_eq_of_curve_C (θ : ℝ) : 
    ∃ x y, 
    (x, y) = curve_C θ ∧ 
    (x - 2)^2 + (y - 2)^2 = 2 := by sorry

theorem max_x_plus_y_on_curve_C :
    ∃ x y θ, 
    (x, y) = curve_C θ ∧ 
    (∀ p : ℝ × ℝ, (p.1, p.2) = curve_C θ → 
    p.1 + p.2 ≤ 6) ∧
    x + y = 6 ∧
    x = 3 ∧ 
    y = 3 := by sorry

end parametric_eq_of_curve_C_max_x_plus_y_on_curve_C_l60_60454


namespace tens_digit_2023_pow_2024_minus_2025_l60_60492

theorem tens_digit_2023_pow_2024_minus_2025 :
  (2023 ^ 2024 - 2025) % 100 / 10 % 10 = 5 :=
sorry

end tens_digit_2023_pow_2024_minus_2025_l60_60492


namespace inequality_problem_l60_60532

theorem inequality_problem (a b c : ℝ) (h : a < b ∧ b < 0) : a^2 > a * b ∧ a * b > b^2 :=
by
  -- The proof is supposed to be here
  sorry

end inequality_problem_l60_60532


namespace average_speed_l60_60128

theorem average_speed (uphill_speed downhill_speed : ℚ) (t : ℚ) (v : ℚ) :
  uphill_speed = 4 →
  downhill_speed = 6 →
  (1 / uphill_speed + 1 / downhill_speed = t) →
  (v * t = 2) →
  v = 4.8 :=
by
  intros
  sorry

end average_speed_l60_60128


namespace rate_of_mixed_oil_l60_60300

/-- If 10 litres of an oil at Rs. 50 per litre is mixed with 5 litres of another oil at Rs. 67 per litre,
    then the rate of the mixed oil per litre is Rs. 55.67. --/
theorem rate_of_mixed_oil : 
  let volume1 := 10
  let price1 := 50
  let volume2 := 5
  let price2 := 67
  let total_cost := (volume1 * price1) + (volume2 * price2)
  let total_volume := volume1 + volume2
  (total_cost / total_volume : ℝ) = 55.67 :=
by
  sorry

end rate_of_mixed_oil_l60_60300


namespace min_value_expr_min_max_value_expr_max_l60_60637

noncomputable def min_value_expr (a b : ℝ) : ℝ := 
  1 / (a - b) + 4 / (b - 1)

noncomputable def max_value_expr (a b : ℝ) : ℝ :=
  a * b - b^2 - a + b

theorem min_value_expr_min (a b : ℝ) (h1 : a > b) (h2 : b > 1) (h3 : a + 3 * b = 5) : 
  min_value_expr a b = 25 :=
sorry

theorem max_value_expr_max (a b : ℝ) (h1 : a > b) (h2 : b > 1) (h3 : a + 3 * b = 5) :
  max_value_expr a b = 1 / 16 :=
sorry

end min_value_expr_min_max_value_expr_max_l60_60637


namespace difference_between_mean_and_median_l60_60284

namespace MathProof

noncomputable def percentage_72 := 0.12
noncomputable def percentage_82 := 0.30
noncomputable def percentage_87 := 0.18
noncomputable def percentage_91 := 0.10
noncomputable def percentage_96 := 1 - (percentage_72 + percentage_82 + percentage_87 + percentage_91)

noncomputable def num_students := 20
noncomputable def scores := [72, 72, 82, 82, 82, 82, 82, 82, 87, 87, 87, 87, 91, 91, 96, 96, 96, 96, 96, 96]

noncomputable def mean_score : ℚ := (72 * 2 + 82 * 6 + 87 * 4 + 91 * 2 + 96 * 6) / num_students
noncomputable def median_score : ℚ := 87

theorem difference_between_mean_and_median :
  mean_score - median_score = 0.1 := by
  sorry

end MathProof

end difference_between_mean_and_median_l60_60284


namespace total_litter_weight_l60_60770

-- Definitions of the conditions
def gina_bags : ℕ := 2
def neighborhood_multiplier : ℕ := 82
def bag_weight : ℕ := 4

-- Representing the total calculation
def neighborhood_bags : ℕ := neighborhood_multiplier * gina_bags
def total_bags : ℕ := neighborhood_bags + gina_bags

def total_weight : ℕ := total_bags * bag_weight

-- Statement of the problem
theorem total_litter_weight : total_weight = 664 :=
by
  sorry

end total_litter_weight_l60_60770


namespace angle_of_inclination_l60_60719

theorem angle_of_inclination (A B : ℝ × ℝ) (hA : A = (2, 5)) (hB : B = (4, 3)) : 
  ∃ θ : ℝ, θ = (3 * Real.pi) / 4 ∧ (∃ k : ℝ, k = (A.2 - B.2) / (A.1 - B.1) ∧ Real.tan θ = k) :=
by
  sorry

end angle_of_inclination_l60_60719


namespace negate_exists_statement_l60_60768

theorem negate_exists_statement : 
  (∃ x : ℝ, x^2 + x - 2 < 0) ↔ ¬ (∀ x : ℝ, x^2 + x - 2 ≥ 0) :=
by sorry

end negate_exists_statement_l60_60768


namespace solution_comparison_l60_60952

theorem solution_comparison (a a' b b' k : ℝ) (h1 : a ≠ 0) (h2 : a' ≠ 0) (h3 : 0 < k) :
  (k * b * a') > (a * b') :=
sorry

end solution_comparison_l60_60952


namespace circles_common_point_l60_60046

theorem circles_common_point {n : ℕ} (hn : n ≥ 5) (circles : Fin n → Set Point)
  (hcommon : ∀ (a b c : Fin n), (circles a ∩ circles b ∩ circles c).Nonempty) :
  ∃ p : Point, ∀ i : Fin n, p ∈ circles i :=
sorry

end circles_common_point_l60_60046


namespace initial_number_of_girls_is_31_l60_60082

-- Define initial number of boys and girls
variables (b g : ℕ)

-- Conditions
def first_condition (g b : ℕ) : Prop := b = 3 * (g - 18)
def second_condition (g b : ℕ) : Prop := 4 * (b - 36) = g - 18

-- Theorem statement
theorem initial_number_of_girls_is_31 (b g : ℕ) (h1 : first_condition g b) (h2 : second_condition g b) : g = 31 :=
by
  sorry

end initial_number_of_girls_is_31_l60_60082


namespace ab_eq_six_l60_60975

theorem ab_eq_six (a b : ℚ) (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end ab_eq_six_l60_60975


namespace prob_twins_street_l60_60863

variable (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1)

theorem prob_twins_street : p ≠ 1 → real := sorry

end prob_twins_street_l60_60863


namespace day_of_50th_day_l60_60537

theorem day_of_50th_day (days_250_N days_150_N1 : ℕ) 
  (h₁ : days_250_N % 7 = 5) (h₂ : days_150_N1 % 7 = 5) : 
  ((50 + 315 - 150 + 365 * 2) % 7) = 4 := 
  sorry

end day_of_50th_day_l60_60537


namespace painting_combinations_l60_60505

-- Define the conditions and the problem statement
def top_row_paint_count := 2
def total_lockers_per_row := 4
def valid_paintings := Nat.choose total_lockers_per_row top_row_paint_count

theorem painting_combinations : valid_paintings = 6 := by
  -- Use the derived conditions to provide the proof
  sorry

end painting_combinations_l60_60505


namespace x_squared_minus_y_squared_l60_60865

theorem x_squared_minus_y_squared (x y : ℝ) (h1 : x + y = 9) (h2 : x - y = 3) : x^2 - y^2 = 27 := by
  sorry

end x_squared_minus_y_squared_l60_60865


namespace relationship_xyz_l60_60068

theorem relationship_xyz (x y z : ℝ) (h1 : x = Real.log x) (h2 : y = Real.logb 5 2) (h3 : z = Real.exp (-0.5)) : x > z ∧ z > y :=
by
  sorry

end relationship_xyz_l60_60068


namespace intersection_of_A_and_B_l60_60778

-- Define the set A
def A : Set ℝ := {-1, 0, 1}

-- Define the set B based on the given conditions
def B : Set ℝ := {y | ∃ x ∈ A, y = Real.cos (Real.pi * x)}

-- The main theorem to prove that A ∩ B is {-1, 1}
theorem intersection_of_A_and_B : A ∩ B = {-1, 1} := by
  sorry

end intersection_of_A_and_B_l60_60778


namespace largest_band_members_l60_60094

def band_formation (m r x : ℕ) : Prop :=
  m < 100 ∧ m = r * x + 2 ∧ (r - 2) * (x + 1) = m ∧ r - 2 * x = 4

theorem largest_band_members : ∃ (r x m : ℕ), band_formation m r x ∧ m = 98 := 
  sorry

end largest_band_members_l60_60094


namespace min_value_frac_x_y_l60_60603

theorem min_value_frac_x_y (x y : ℝ) (hx : x > 0) (hy : y > -1) (hxy : x + y = 1) :
  ∃ m, m = 2 + Real.sqrt 3 ∧ ∀ x y, x > 0 → y > -1 → x + y = 1 → (x^2 + 3) / x + y^2 / (y + 1) ≥ m :=
sorry

end min_value_frac_x_y_l60_60603


namespace volume_ratio_l60_60430

theorem volume_ratio (x : ℝ) (h : x > 0) : 
  let V_Q := x^3
  let V_P := (3 * x)^3
  (V_Q / V_P) = (1 / 27) :=
by
  sorry

end volume_ratio_l60_60430


namespace Riley_fewer_pairs_l60_60480

-- Define the conditions
def Ellie_pairs : ℕ := 8
def Total_pairs : ℕ := 13

-- Prove the statement
theorem Riley_fewer_pairs : (Total_pairs - Ellie_pairs) - Ellie_pairs = 3 :=
by
  -- Skip the proof
  sorry

end Riley_fewer_pairs_l60_60480


namespace chess_team_boys_l60_60915

variable (B G : ℕ)

theorem chess_team_boys (h1 : B + G = 30) (h2 : (1 / 3 : ℝ) * G + B = 20) : B = 15 := by
  sorry

end chess_team_boys_l60_60915


namespace value_of_x_l60_60349

theorem value_of_x (x y z : ℕ) (h1 : x = y / 3) (h2 : y = z / 4) (h3 : z = 96) : x = 8 :=
by
  sorry

end value_of_x_l60_60349


namespace least_number_when_increased_by_6_is_divisible_l60_60570

theorem least_number_when_increased_by_6_is_divisible :
  ∃ n : ℕ, 
    (n + 6) % 24 = 0 ∧ 
    (n + 6) % 32 = 0 ∧ 
    (n + 6) % 36 = 0 ∧ 
    (n + 6) % 54 = 0 ∧ 
    n = 858 :=
by
  sorry

end least_number_when_increased_by_6_is_divisible_l60_60570


namespace km_per_gallon_proof_l60_60095

-- Define the given conditions
def distance := 100
def gallons := 10

-- Define what we need to prove the correct answer
def kilometers_per_gallon := distance / gallons

-- Prove that the calculated kilometers per gallon is equal to 10
theorem km_per_gallon_proof : kilometers_per_gallon = 10 := by
  sorry

end km_per_gallon_proof_l60_60095


namespace number_of_cows_l60_60934

theorem number_of_cows (H : ℕ) (C : ℕ) (h1 : H = 6) (h2 : C / H = 7 / 2) : C = 21 :=
by
  sorry

end number_of_cows_l60_60934


namespace basketball_scores_l60_60554

theorem basketball_scores :
  ∃ P: Finset ℕ, (∀ x y: ℕ, (x + y = 7 → P = {p | ∃ x y: ℕ, p = 3 * x + 2 * y})) ∧ (P.card = 8) :=
sorry

end basketball_scores_l60_60554


namespace solve_equation_l60_60824

theorem solve_equation (x : ℚ) (h : x ≠ 3) : (x + 5) / (x - 3) = 4 ↔ x = 17 / 3 :=
sorry

end solve_equation_l60_60824


namespace min_value_a4b3c2_l60_60703

theorem min_value_a4b3c2 (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : 1/a + 1/b + 1/c = 9) : a^4 * b^3 * c^2 ≥ 1/1152 := 
sorry

end min_value_a4b3c2_l60_60703


namespace compute_expression_l60_60431

theorem compute_expression :
  3 * 3^4 - 9^60 / 9^57 = -486 :=
by
  sorry

end compute_expression_l60_60431


namespace probability_two_red_balls_randomly_picked_l60_60806

theorem probability_two_red_balls_randomly_picked :
  (3/9) * (2/8) = 1/12 :=
by sorry

end probability_two_red_balls_randomly_picked_l60_60806


namespace total_students_count_l60_60118

theorem total_students_count (n1 n2 n: ℕ) (avg1 avg2 avg_tot: ℝ)
  (h1: n1 = 15) (h2: avg1 = 70) (h3: n2 = 10) (h4: avg2 = 90) (h5: avg_tot = 78)
  (h6: (n1 * avg1 + n2 * avg2) / (n1 + n2) = avg_tot) :
  n = 25 :=
by
  sorry

end total_students_count_l60_60118


namespace zachary_crunches_more_than_pushups_l60_60169

def push_ups_zachary : ℕ := 46
def crunches_zachary : ℕ := 58

theorem zachary_crunches_more_than_pushups : crunches_zachary - push_ups_zachary = 12 := by
  sorry

end zachary_crunches_more_than_pushups_l60_60169


namespace range_of_m_l60_60321

open Classical

variable {m : ℝ}

def p (m : ℝ) : Prop := ∃ x : ℝ, x^2 + 2 * x + m ≤ 0
def q (m : ℝ) : Prop := ∀ x : ℝ, (3 - m) > 1 → ((3 - m) ^ x > 0)

theorem range_of_m (hm : (p m ∨ q m) ∧ ¬(p m ∧ q m)) : 1 < m ∧ m < 2 :=
  sorry

end range_of_m_l60_60321


namespace coordinates_of_C_l60_60884

theorem coordinates_of_C (A B : ℝ × ℝ) (C : ℝ × ℝ) 
    (hA : A = (1, 3)) (hB : B = (9, -3)) (hBC_AB : dist B C = 1/2 * dist A B) : 
    C = (13, -6) :=
sorry

end coordinates_of_C_l60_60884


namespace roots_of_unity_sum_l60_60209

theorem roots_of_unity_sum (x y z : ℂ) (n m p : ℕ)
  (hx : x^n = 1) (hy : y^m = 1) (hz : z^p = 1) :
  (∃ k : ℕ, (x + y + z)^k = 1) ↔ (x + y = 0 ∨ y + z = 0 ∨ z + x = 0) :=
sorry

end roots_of_unity_sum_l60_60209


namespace fifty_percent_of_number_l60_60352

-- Define the given condition
def given_condition (x : ℝ) : Prop :=
  0.6 * x = 42

-- Define the statement we need to prove
theorem fifty_percent_of_number (x : ℝ) (h : given_condition x) : 0.5 * x = 35 := by
  sorry

end fifty_percent_of_number_l60_60352


namespace set_C_is_correct_l60_60391

open Set

noncomputable def set_A : Set ℝ := {x | x ^ 2 - x - 12 ≤ 0}
noncomputable def set_B : Set ℝ := {x | (x + 1) / (x - 1) < 0}
noncomputable def set_C : Set ℝ := {x | x ∈ set_A ∧ x ∉ set_B}

theorem set_C_is_correct : set_C = {x | -3 ≤ x ∧ x ≤ -1} ∪ {x | 1 ≤ x ∧ x ≤ 4} :=
by
  sorry

end set_C_is_correct_l60_60391


namespace length_ac_l60_60677

theorem length_ac (a b c d e : ℝ) (h1 : bc = 3 * cd) (h2 : de = 7) (h3 : ab = 5) (h4 : ae = 20) :
    ac = 11 :=
by
  sorry

end length_ac_l60_60677


namespace sum_at_simple_interest_l60_60213

theorem sum_at_simple_interest 
  (P R : ℕ)
  (h : ((P * (R + 1) * 3) / 100) - ((P * R * 3) / 100) = 69) : 
  P = 2300 :=
by sorry

end sum_at_simple_interest_l60_60213


namespace problem_statement_l60_60166

def system_eq1 (x y : ℝ) := x^3 - 5 * x * y^2 = 21
def system_eq2 (y x : ℝ) := y^3 - 5 * x^2 * y = 28

theorem problem_statement
(x1 y1 x2 y2 x3 y3 : ℝ)
(h1 : system_eq1 x1 y1)
(h2 : system_eq2 y1 x1)
(h3 : system_eq1 x2 y2)
(h4 : system_eq2 y2 x2)
(h5 : system_eq1 x3 y3)
(h6 : system_eq2 y3 x3)
(h_distinct : (x1, y1) ≠ (x2, y2) ∧ (x1, y1) ≠ (x3, y3) ∧ (x2, y2) ≠ (x3, y3)) :
  (11 - x1 / y1) * (11 - x2 / y2) * (11 - x3 / y3) = 1729 :=
sorry

end problem_statement_l60_60166


namespace subsequent_flights_requirements_l60_60003

-- Define the initial conditions
def late_flights : ℕ := 1
def on_time_flights : ℕ := 3
def total_initial_flights : ℕ := late_flights + on_time_flights

-- Define the number of subsequent flights needed
def subsequent_flights_needed (x : ℕ) : Prop :=
  let total_flights := total_initial_flights + x
  let on_time_total := on_time_flights + x
  (on_time_total : ℚ) / (total_flights : ℚ) > 0.40

-- State the theorem to prove
theorem subsequent_flights_requirements:
  ∃ x : ℕ, subsequent_flights_needed x := sorry

end subsequent_flights_requirements_l60_60003


namespace Faye_created_rows_l60_60513

theorem Faye_created_rows (total_crayons : ℕ) (crayons_per_row : ℕ) (rows : ℕ) 
  (h1 : total_crayons = 210) (h2 : crayons_per_row = 30) : rows = 7 :=
by
  sorry

end Faye_created_rows_l60_60513


namespace inequality_example_l60_60334

theorem inequality_example (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a^2 + b^2 + c^2 = 3) : 
    1 / (1 + a * b) + 1 / (1 + b * c) + 1 / (1 + a * c) ≥ 3 / 2 :=
by
  sorry

end inequality_example_l60_60334


namespace chandler_saves_weeks_l60_60685

theorem chandler_saves_weeks 
  (cost_of_bike : ℕ) 
  (grandparents_money : ℕ) 
  (aunt_money : ℕ) 
  (cousin_money : ℕ) 
  (weekly_earnings : ℕ)
  (total_birthday_money : ℕ := grandparents_money + aunt_money + cousin_money) 
  (total_money : ℕ := total_birthday_money + weekly_earnings * 24):
  (cost_of_bike = 600) → 
  (grandparents_money = 60) → 
  (aunt_money = 40) → 
  (cousin_money = 20) → 
  (weekly_earnings = 20) → 
  (total_money = cost_of_bike) → 
  24 = ((cost_of_bike - total_birthday_money) / weekly_earnings) := 
by 
  intros; 
  sorry

end chandler_saves_weeks_l60_60685


namespace FourConsecIntsSum34Unique_l60_60163

theorem FourConsecIntsSum34Unique :
  ∃! (a b c d : ℕ), (a < b) ∧ (b < c) ∧ (c < d) ∧ (a + b + c + d = 34) ∧ (d = a + 3) :=
by
  -- The proof will be placed here
  sorry

end FourConsecIntsSum34Unique_l60_60163


namespace no_solution_for_equation_l60_60758

theorem no_solution_for_equation : 
  ∀ x : ℝ, (x ≠ 3) → (x-1)/(x-3) = 2 - 2/(3-x) → False :=
by
  intro x hx heq
  sorry

end no_solution_for_equation_l60_60758


namespace flag_count_l60_60633

-- Definitions of colors as a datatype
inductive Color
| red : Color
| white : Color
| blue : Color
| green : Color
| yellow : Color

open Color

-- Total number of distinct flags possible
theorem flag_count : 
  (∃ m : Color, 
   (∃ t : Color, 
    (t ≠ m ∧ 
     ∃ b : Color, 
     (b ≠ m ∧ b ≠ red ∧ b ≠ blue)))) ∧ 
  (5 * 4 * 2 = 40) := 
  sorry

end flag_count_l60_60633


namespace percentage_of_number_l60_60214

theorem percentage_of_number (N P : ℝ) (h1 : 0.60 * N = 240) (h2 : (P / 100) * N = 160) : P = 40 :=
by
  sorry

end percentage_of_number_l60_60214


namespace max_value_of_f_l60_60696

open Real

noncomputable def f (θ : ℝ) : ℝ :=
  sin (θ / 2) * (1 + cos θ)

theorem max_value_of_f : 
  (∃ θ : ℝ, 0 < θ ∧ θ < π ∧ (∀ θ' : ℝ, 0 < θ' ∧ θ' < π → f θ' ≤ f θ) ∧ f θ = 4 * sqrt 3 / 9) := 
by
  sorry

end max_value_of_f_l60_60696


namespace find_line_equation_l60_60565

-- Define point A
def point_A : ℝ × ℝ := (-3, 4)

-- Define the conditions
def passes_through_point_A (line_eq : ℝ → ℝ → ℝ) : Prop :=
  line_eq (-3) 4 = 0

def intercept_condition (line_eq : ℝ → ℝ → ℝ) : Prop :=
  ∃ a : ℝ, a ≠ 0 ∧ line_eq (2 * a) 0 = 0 ∧ line_eq 0 a = 0

-- Define the equations of the line
def line1 (x y : ℝ) : ℝ := 3 * y + 4 * x
def line2 (x y : ℝ) : ℝ := 2 * x - y - 5

-- Statement of the problem
theorem find_line_equation : 
  (passes_through_point_A line1 ∧ intercept_condition line1) ∨
  (passes_through_point_A line2 ∧ intercept_condition line2) :=
sorry

end find_line_equation_l60_60565


namespace find_number_chosen_l60_60700

theorem find_number_chosen (x : ℤ) (h : 4 * x - 138 = 102) : x = 60 := sorry

end find_number_chosen_l60_60700


namespace functions_unique_l60_60113

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def g (x : ℝ) : ℝ := sorry

theorem functions_unique (f g: ℝ → ℝ) :
  (∀ x : ℝ, x < 0 → (f (g x) = x / (x * f x - 2)) ∧ (g (f x) = x / (x * g x - 2))) →
  (∀ x : ℝ, 0 < x → (f x = 3 / x ∧ g x = 3 / x)) :=
by
  sorry

end functions_unique_l60_60113


namespace unique_solution_in_z3_l60_60424

theorem unique_solution_in_z3 (x y z : ℤ) (h : x^3 + 2 * y^3 = 4 * z^3) : 
  x = 0 ∧ y = 0 ∧ z = 0 :=
sorry

end unique_solution_in_z3_l60_60424


namespace profit_percentage_with_discount_l60_60837

theorem profit_percentage_with_discount
    (P M : ℝ)
    (h1 : M = 1.27 * P)
    (h2 : 0 < P) :
    ((0.95 * M - P) / P) * 100 = 20.65 :=
by
  sorry

end profit_percentage_with_discount_l60_60837


namespace mean_weight_of_cats_l60_60109

def weight_list : List ℝ :=
  [87, 90, 93, 95, 95, 98, 104, 106, 106, 107, 109, 110, 111, 112]

noncomputable def total_weight : ℝ := weight_list.sum

noncomputable def mean_weight : ℝ := total_weight / weight_list.length

theorem mean_weight_of_cats : mean_weight = 101.64 := by
  sorry

end mean_weight_of_cats_l60_60109


namespace perfect_square_l60_60899

theorem perfect_square (a b : ℝ) : a^2 + 2 * a * b + b^2 = (a + b)^2 := by
  sorry

end perfect_square_l60_60899


namespace unique_pos_neg_roots_of_poly_l60_60219

noncomputable def poly : Polynomial ℝ := Polynomial.C 1 * Polynomial.X^4 + Polynomial.C 5 * Polynomial.X^3 + Polynomial.C 15 * Polynomial.X - Polynomial.C 9

theorem unique_pos_neg_roots_of_poly : 
  (∃! x : ℝ, (0 < x) ∧ poly.eval x = 0) ∧ (∃! x : ℝ, (x < 0) ∧ poly.eval x = 0) :=
  sorry

end unique_pos_neg_roots_of_poly_l60_60219


namespace eval_expr_at_2_l60_60427

def expr (x : ℝ) : ℝ := (3 * x + 4)^2

theorem eval_expr_at_2 : expr 2 = 100 :=
by sorry

end eval_expr_at_2_l60_60427


namespace find_x_l60_60858

variable (a b c d e f g h x : ℤ)

def cell_relationships (a b c d e f g h x : ℤ) : Prop :=
  (a = 10) ∧
  (h = 3) ∧
  (a = 10 + b) ∧
  (b = c + a) ∧
  (c = b + d) ∧
  (d = c + h) ∧
  (e = 10 + f) ∧
  (f = e + g) ∧
  (g = d + h) ∧
  (h = g + x)

theorem find_x : cell_relationships a b c d e f g h x → x = 7 :=
sorry

end find_x_l60_60858


namespace sin_180_eq_0_l60_60341

theorem sin_180_eq_0 : Real.sin (180 * Real.pi / 180) = 0 :=
by
  sorry

end sin_180_eq_0_l60_60341


namespace num_monic_quadratic_trinomials_l60_60531

noncomputable def count_monic_quadratic_trinomials : ℕ :=
  4489

theorem num_monic_quadratic_trinomials :
  count_monic_quadratic_trinomials = 4489 :=
by
  sorry

end num_monic_quadratic_trinomials_l60_60531


namespace quadratic_inequality_solution_l60_60961

variables {x p q : ℝ}

theorem quadratic_inequality_solution
  (h1 : ∀ x, x^2 + p * x + q < 0 ↔ -1/2 < x ∧ x < 1/3) : 
  ∀ x, q * x^2 + p * x + 1 > 0 ↔ -2 < x ∧ x < 3 :=
by sorry

end quadratic_inequality_solution_l60_60961


namespace arithmetic_sequence_product_l60_60922

theorem arithmetic_sequence_product (b : ℕ → ℤ) (h1 : ∀ n, b (n + 1) = b n + d) 
  (h2 : b 5 * b 6 = 35) : b 4 * b 7 = 27 :=
sorry

end arithmetic_sequence_product_l60_60922


namespace find_coefficients_l60_60297

noncomputable def polynomial_h (x : ℚ) : ℚ := x^3 + 2 * x^2 + 3 * x + 4

noncomputable def polynomial_j (b c d x : ℚ) : ℚ := x^3 + b * x^2 + c * x + d

theorem find_coefficients :
  (∃ b c d : ℚ,
     (∀ s : ℚ, polynomial_h s = 0 → polynomial_j b c d (s^3) = 0) ∧
     (b, c, d) = (6, 12, 8)) :=
sorry

end find_coefficients_l60_60297


namespace initial_fee_correct_l60_60510

-- Define the relevant values
def initialFee := 2.25
def chargePerSegment := 0.4
def totalDistance := 3.6
def totalCharge := 5.85
noncomputable def segments := (totalDistance * (5 / 2))
noncomputable def costForDistance := segments * chargePerSegment

-- Define the theorem
theorem initial_fee_correct :
  totalCharge = initialFee + costForDistance :=
by
  -- Proof is omitted.
  sorry

end initial_fee_correct_l60_60510


namespace rectangle_area_l60_60756

theorem rectangle_area (l w : ℝ) (h1 : l = 4 * w) (h2 : 2 * l + 2 * w = 200) : l * w = 1600 := by
  sorry

end rectangle_area_l60_60756


namespace find_y_l60_60278

theorem find_y (DEG EFG y : ℝ) 
  (h1 : DEG = 150)
  (h2 : EFG = 40)
  (h3 : DEG = EFG + y) :
  y = 110 :=
by
  sorry

end find_y_l60_60278


namespace value_of_sum_l60_60134

theorem value_of_sum (a b c d : ℤ) 
  (h1 : a - b + c = 7)
  (h2 : b - c + d = 8)
  (h3 : c - d + a = 5)
  (h4 : d - a + b = 4) : a + b + c + d = 12 := 
  sorry

end value_of_sum_l60_60134


namespace work_schedules_lcm_l60_60831

theorem work_schedules_lcm : Nat.lcm (Nat.lcm 5 6) (Nat.lcm 8 9) = 360 := 
by 
  sorry

end work_schedules_lcm_l60_60831


namespace total_beads_correct_l60_60718

-- Definitions of the problem conditions
def blue_beads : ℕ := 5
def red_beads : ℕ := 2 * blue_beads
def white_beads : ℕ := blue_beads + red_beads
def silver_beads : ℕ := 10

-- Definition of the total number of beads
def total_beads : ℕ := blue_beads + red_beads + white_beads + silver_beads

-- The main theorem statement
theorem total_beads_correct : total_beads = 40 :=
by 
  sorry

end total_beads_correct_l60_60718


namespace minimum_number_of_peanuts_l60_60572

/--
Five monkeys share a pile of peanuts.
Each monkey divides the peanuts into five piles, leaves one peanut which it eats, and takes away one pile.
This process continues in the same manner until the fifth monkey, who also evenly divides the remaining peanuts into five piles and has one peanut left over.
Prove that the minimum number of peanuts in the pile originally is 3121.
-/
theorem minimum_number_of_peanuts : ∃ N : ℕ, N = 3121 ∧
  (N - 1) % 5 = 0 ∧
  ((4 * ((N - 1) / 5) - 1) % 5 = 0) ∧
  ((4 * ((4 * ((N - 1) / 5) - 1) / 5) - 1) % 5 = 0) ∧
  ((4 * ((4 * ((4 * ((N - 1) / 5) - 1) / 5) - 1) / 5) - 1) % 5 = 0) ∧
  ((4 * ((4 * ((4 * ((4 * ((N - 1) / 5) - 1) / 5) - 1) / 5) - 1) / 5) - 1) / 4) % 5 = 0 :=
by
  sorry

end minimum_number_of_peanuts_l60_60572


namespace find_y_l60_60621

noncomputable def x : ℝ := 0.7142857142857143

def equation (y : ℝ) : Prop :=
  (x * y) / 7 = x^2

theorem find_y : ∃ y : ℝ, equation y ∧ y = 5 :=
by
  use 5
  have h1 : x != 0 := by sorry
  have h2 : (x * 5) / 7 = x^2 := by sorry
  exact ⟨h2, rfl⟩

end find_y_l60_60621


namespace eq_value_l60_60623

theorem eq_value (x y : ℕ) (h1 : x - y = 9) (h2 : x = 9) : 3 ^ x * 4 ^ y = 19683 := by
  sorry

end eq_value_l60_60623


namespace inverse_proposition_true_l60_60045

-- Define a rectangle and a square
structure Rectangle where
  length : ℝ
  width  : ℝ

def is_square (r : Rectangle) : Prop :=
  r.length = r.width ∧ r.length > 0 ∧ r.width > 0

-- Define the condition that a rectangle with equal adjacent sides is a square
def rectangle_with_equal_adjacent_sides_is_square : Prop :=
  ∀ r : Rectangle, r.length = r.width → is_square r

-- Define the inverse proposition that a square is a rectangle with equal adjacent sides
def square_is_rectangle_with_equal_adjacent_sides : Prop :=
  ∀ r : Rectangle, is_square r → r.length = r.width

-- The proof statement
theorem inverse_proposition_true :
  rectangle_with_equal_adjacent_sides_is_square → square_is_rectangle_with_equal_adjacent_sides :=
by
  sorry

end inverse_proposition_true_l60_60045


namespace problem_statement_l60_60847

theorem problem_statement : (6^3 + 4^2) * 7^5 = 3897624 := by
  sorry

end problem_statement_l60_60847


namespace droneSystemEquations_l60_60947

-- Definitions based on conditions
def typeADrones (x y : ℕ) : Prop := x = (1/2 : ℝ) * (x + y) + 11
def typeBDrones (x y : ℕ) : Prop := y = (1/3 : ℝ) * (x + y) - 2

-- Theorem statement
theorem droneSystemEquations (x y : ℕ) :
  typeADrones x y ∧ typeBDrones x y ↔
  (x = (1/2 : ℝ) * (x + y) + 11 ∧ y = (1/3 : ℝ) * (x + y) - 2) :=
by sorry

end droneSystemEquations_l60_60947


namespace time_to_cover_escalator_l60_60679

-- Definitions for the provided conditions.
def escalator_speed : ℝ := 7
def escalator_length : ℝ := 180
def person_speed : ℝ := 2

-- Goal to prove the time taken to cover the escalator length.
theorem time_to_cover_escalator : (escalator_length / (escalator_speed + person_speed)) = 20 := by
  sorry

end time_to_cover_escalator_l60_60679


namespace sum_two_triangular_numbers_iff_l60_60745

theorem sum_two_triangular_numbers_iff (m : ℕ) : 
  (∃ a b : ℕ, m = (a * (a + 1)) / 2 + (b * (b + 1)) / 2) ↔ 
  (∃ x y : ℕ, 4 * m + 1 = x * x + y * y) :=
by sorry

end sum_two_triangular_numbers_iff_l60_60745


namespace problem_diamond_value_l60_60528

def diamond (x y : ℕ) : ℕ := 4 * x + 6 * y

theorem problem_diamond_value :
  diamond 3 4 = 36 := 
by
  sorry

end problem_diamond_value_l60_60528


namespace simplify_expression_l60_60741

variables (a b : ℝ)

theorem simplify_expression : 
  a^(2/3) * b^(1/2) * (-3 * a^(1/2) * b^(1/3)) / (1/3 * a^(1/6) * b^(5/6)) = -9 * a := by
  -- proof here
  sorry

end simplify_expression_l60_60741


namespace value_of_a_l60_60290

theorem value_of_a (a b k : ℝ) (h1 : a = k / b^2) (h2 : a = 40) (h3 : b = 12) (h4 : b = 24) : a = 10 := 
by
  sorry

end value_of_a_l60_60290


namespace cj_more_stamps_than_twice_kj_l60_60944

variable (C K A : ℕ) (x : ℕ)

theorem cj_more_stamps_than_twice_kj :
  (C = 2 * K + x) →
  (K = A / 2) →
  (C + K + A = 930) →
  (A = 370) →
  (x = 25) →
  (C - 2 * K = 5) :=
by
  intros h1 h2 h3 h4 h5
  sorry

end cj_more_stamps_than_twice_kj_l60_60944


namespace max_obtuse_angles_in_quadrilateral_l60_60925

theorem max_obtuse_angles_in_quadrilateral (a b c d : ℝ) 
  (h₁ : a + b + c + d = 360)
  (h₂ : 90 < a)
  (h₃ : 90 < b)
  (h₄ : 90 < c) :
  90 > d :=
sorry

end max_obtuse_angles_in_quadrilateral_l60_60925


namespace sin_870_eq_half_l60_60707

theorem sin_870_eq_half : Real.sin (870 * Real.pi / 180) = 1 / 2 := by
  sorry

end sin_870_eq_half_l60_60707


namespace proof_problem_l60_60599

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := 
  (Real.sin (2 * x), 2 * Real.cos x ^ 2 - 1)

noncomputable def vector_b (θ : ℝ) : ℝ × ℝ := 
  (Real.sin θ, Real.cos θ)

noncomputable def f (x θ : ℝ) : ℝ := 
  (vector_a x).1 * (vector_b θ).1 + (vector_a x).2 * (vector_b θ).2

theorem proof_problem 
  (θ : ℝ) 
  (hθ : 0 < θ ∧ θ < π) 
  (h1 : f (π / 6) θ = 1) 
  (x : ℝ) 
  (hx : -π / 6 ≤ x ∧ x ≤ π / 4) :
  θ = π / 3 ∧
  (∀ x, f x θ = f (x + π) θ) ∧
  (∀ x, -π / 6 ≤ x ∧ x ≤ π / 4 → f x θ ≤ 1) ∧
  (∀ x, -π / 6 ≤ x ∧ x ≤ π / 4 → f x θ ≥ -0.5) :=
by
  sorry

end proof_problem_l60_60599


namespace solve_inequality_system_l60_60795

theorem solve_inequality_system (x : ℝ) (h1 : 3 * x - 2 < x) (h2 : (1 / 3) * x < -2) : x < -6 :=
sorry

end solve_inequality_system_l60_60795


namespace range_of_a_l60_60496

theorem range_of_a :
  (¬ ∃ x : ℝ, x^2 + 2 * a * x + a ≤ 0) → (0 < a ∧ a < 1) :=
by
  intros
  sorry

end range_of_a_l60_60496


namespace probability_student_less_than_25_l60_60312

def total_students : ℕ := 100

-- Percentage conditions translated to proportions
def proportion_male : ℚ := 0.48
def proportion_female : ℚ := 0.52

def proportion_male_25_or_older : ℚ := 0.50
def proportion_female_25_or_older : ℚ := 0.20

-- Definition of probability that a randomly selected student is less than 25 years old.
def probability_less_than_25 : ℚ :=
  (proportion_male * (1 - proportion_male_25_or_older)) +
  (proportion_female * (1 - proportion_female_25_or_older))

theorem probability_student_less_than_25 :
  probability_less_than_25 = 0.656 := by
  sorry

end probability_student_less_than_25_l60_60312


namespace diameter_of_circular_field_l60_60546

noncomputable def diameter (C : ℝ) : ℝ := C / Real.pi

theorem diameter_of_circular_field :
  let cost_per_meter := 3
  let total_cost := 376.99
  let circumference := total_cost / cost_per_meter
  diameter circumference = 40 :=
by
  let cost_per_meter : ℝ := 3
  let total_cost : ℝ := 376.99
  let circumference : ℝ := total_cost / cost_per_meter
  have : circumference = 125.66333333333334 := by sorry
  have : diameter circumference = 40 := by sorry
  sorry

end diameter_of_circular_field_l60_60546


namespace initial_pokemon_cards_l60_60267

variables (x : ℕ)

theorem initial_pokemon_cards (h : x - 2 = 1) : x = 3 := 
sorry

end initial_pokemon_cards_l60_60267


namespace no_integer_roots_l60_60146

theorem no_integer_roots (a b c : ℤ) (ha : Odd a) (hb : Odd b) (hc : Odd c) : ∀ x : ℤ, a * x^2 + b * x + c ≠ 0 := by
  sorry

end no_integer_roots_l60_60146


namespace Hillary_reading_time_on_sunday_l60_60155

-- Define the assigned reading times for both books
def assigned_time_book_a : ℕ := 60 -- minutes
def assigned_time_book_b : ℕ := 45 -- minutes

-- Define the reading times already spent on each book
def time_spent_friday_book_a : ℕ := 16 -- minutes
def time_spent_saturday_book_a : ℕ := 28 -- minutes
def time_spent_saturday_book_b : ℕ := 15 -- minutes

-- Calculate the total time already read for each book
def total_time_read_book_a : ℕ := time_spent_friday_book_a + time_spent_saturday_book_a
def total_time_read_book_b : ℕ := time_spent_saturday_book_b

-- Calculate the remaining time needed for each book
def remaining_time_book_a : ℕ := assigned_time_book_a - total_time_read_book_a
def remaining_time_book_b : ℕ := assigned_time_book_b - total_time_read_book_b

-- Calculate the total remaining time and the equal time division
def total_remaining_time : ℕ := remaining_time_book_a + remaining_time_book_b
def equal_time_division : ℕ := total_remaining_time / 2

-- Theorem statement to prove Hillary's reading time for each book on Sunday
theorem Hillary_reading_time_on_sunday : equal_time_division = 23 := by
  sorry

end Hillary_reading_time_on_sunday_l60_60155


namespace ratio_of_capitals_l60_60535

noncomputable def Ashok_loss (total_loss : ℝ) (Pyarelal_loss : ℝ) : ℝ := total_loss - Pyarelal_loss

theorem ratio_of_capitals (total_loss : ℝ) (Pyarelal_loss : ℝ) (Ashok_capital Pyarelal_capital : ℝ) 
    (h_total_loss : total_loss = 1200)
    (h_Pyarelal_loss : Pyarelal_loss = 1080)
    (h_Ashok_capital : Ashok_capital = 120)
    (h_Pyarelal_capital : Pyarelal_capital = 1080) :
    Ashok_capital / Pyarelal_capital = 1 / 9 :=
by
  sorry

end ratio_of_capitals_l60_60535


namespace smallest_positive_period_one_increasing_interval_l60_60950

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 4)

def is_periodic_with_period (f : ℝ → ℝ) (T : ℝ) :=
  ∀ x, f (x + T) = f x

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x ≤ f y

theorem smallest_positive_period :
  is_periodic_with_period f Real.pi :=
sorry

theorem one_increasing_interval :
  is_increasing_on f (-(Real.pi / 8)) (3 * Real.pi / 8) :=
sorry

end smallest_positive_period_one_increasing_interval_l60_60950


namespace value_of_sum_l60_60771

theorem value_of_sum (a b : ℝ) (h : a^2 * b^2 + a^2 + b^2 + 1 - 2 * a * b = 2 * a * b) : a + b = 2 ∨ a + b = -2 :=
sorry

end value_of_sum_l60_60771


namespace smallest_positive_period_symmetry_axis_not_even_function_decreasing_interval_l60_60342

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + (4 * Real.pi / 3))

theorem smallest_positive_period (T : ℝ) : T = Real.pi ↔ (∀ x : ℝ, f (x + T) = f x) := by
  sorry

theorem symmetry_axis (x : ℝ) : x = (7 * Real.pi / 12) ↔ (∀ y : ℝ, f (2 * x - y) = f y) := by
  sorry

theorem not_even_function : ¬ (∀ x : ℝ, f (x + (Real.pi / 3)) = f (-x - (Real.pi / 3))) := by
  sorry

theorem decreasing_interval (k : ℤ) (x : ℝ) : (k * Real.pi - (5 * Real.pi / 12) ≤ x ∧ x ≤ k * Real.pi + (Real.pi / 12)) ↔ (∀ x1 x2 : ℝ, x1 < x2 → f x1 ≥ f x2) := by
  sorry

end smallest_positive_period_symmetry_axis_not_even_function_decreasing_interval_l60_60342


namespace tan_alpha_plus_pi_over_4_rational_expression_of_trig_l60_60154

theorem tan_alpha_plus_pi_over_4 (α : ℝ) (h : Real.tan (α / 2) = 2) : 
  Real.tan (α + Real.pi / 4) = -1 / 7 := 
by 
  sorry

theorem rational_expression_of_trig (α : ℝ) (h : Real.tan (α / 2) = 2) : 
  (6 * Real.sin α + Real.cos α) / (3 * Real.sin α - 2 * Real.cos α) = 7 / 6 := 
by 
  sorry

end tan_alpha_plus_pi_over_4_rational_expression_of_trig_l60_60154


namespace count_p_shape_points_l60_60317

-- Define the problem conditions
def side_length : ℕ := 10
def point_interval : ℕ := 1
def num_sides : ℕ := 3
def correction_corners : ℕ := 2

-- Define the total expected points
def total_expected_points : ℕ := 31

-- Proof statement
theorem count_p_shape_points :
  ((side_length / point_interval + 1) * num_sides - correction_corners) = total_expected_points := by
  sorry

end count_p_shape_points_l60_60317


namespace area_gray_region_correct_l60_60867

def center_C : ℝ × ℝ := (3, 5)
def radius_C : ℝ := 3
def center_D : ℝ × ℝ := (9, 5)
def radius_D : ℝ := 3

noncomputable def area_gray_region : ℝ :=
  let rectangle_area := (center_D.1 - center_C.1) * (center_C.2 - (center_C.2 - radius_C))
  let sector_area := (1 / 4) * radius_C ^ 2 * Real.pi
  rectangle_area - 2 * sector_area

theorem area_gray_region_correct :
  area_gray_region = 18 - 9 / 2 * Real.pi :=
by
  sorry

end area_gray_region_correct_l60_60867


namespace license_plate_count_l60_60266

-- Formalize the conditions
def is_letter (c : Char) : Prop := 'a' ≤ c ∧ c ≤ 'z'
def is_digit (c : Char) : Prop := '0' ≤ c ∧ c ≤ '9'

-- Define the main proof problem
theorem license_plate_count :
  (26 * (25 + 9) * 26 * 10 = 236600) :=
by sorry

end license_plate_count_l60_60266


namespace nap_time_l60_60088

-- Definitions of given conditions
def flight_duration : ℕ := 680
def reading_time : ℕ := 120
def movie_time : ℕ := 240
def dinner_time : ℕ := 30
def radio_time : ℕ := 40
def game_time : ℕ := 70

def total_activity_time : ℕ := reading_time + movie_time + dinner_time + radio_time + game_time

-- Theorem statement
theorem nap_time : (flight_duration - total_activity_time) / 60 = 3 := by
  -- Here would go the proof steps verifying the equality
  sorry

end nap_time_l60_60088


namespace relationship_P_Q_l60_60672

theorem relationship_P_Q (x : ℝ) (P : ℝ) (Q : ℝ) 
  (hP : P = Real.exp x + Real.exp (-x)) 
  (hQ : Q = (Real.sin x + Real.cos x) ^ 2) : 
  P ≥ Q := 
sorry

end relationship_P_Q_l60_60672


namespace ratio_of_ages_l60_60221

theorem ratio_of_ages (M : ℕ) (S : ℕ) (h1 : M = 24) (h2 : S + 6 = 38) : 
  (S / M : ℚ) = 4 / 3 :=
by
  sorry

end ratio_of_ages_l60_60221


namespace team_selection_l60_60584

open Nat

theorem team_selection :
  let boys := 10
  let girls := 12
  let team_size := 8
  let boys_to_choose := 5
  let girls_to_choose := 3
  choose boys boys_to_choose * choose girls girls_to_choose = 55440 :=
by
  sorry

end team_selection_l60_60584


namespace benny_eggs_l60_60425

theorem benny_eggs (dozen_count : ℕ) (eggs_per_dozen : ℕ) (total_eggs : ℕ) 
  (h1 : dozen_count = 7) 
  (h2 : eggs_per_dozen = 12) 
  (h3 : total_eggs = dozen_count * eggs_per_dozen) : 
  total_eggs = 84 := 
by 
  sorry

end benny_eggs_l60_60425


namespace total_salmon_l60_60374

def male_salmon : Nat := 712261
def female_salmon : Nat := 259378

theorem total_salmon :
  male_salmon + female_salmon = 971639 := by
  sorry

end total_salmon_l60_60374


namespace min_value_functions_l60_60767

noncomputable def f_A (x : ℝ) : ℝ := x^2 + 1 / x^2
noncomputable def f_B (x : ℝ) : ℝ := 2 * x + 2 / x
noncomputable def f_C (x : ℝ) : ℝ := (x - 1) / (x + 1)
noncomputable def f_D (x : ℝ) : ℝ := Real.log (Real.sqrt x + 1)

theorem min_value_functions :
  (∃ x : ℝ, ∀ y : ℝ, f_A x ≤ f_A y) ∧
  (∃ x : ℝ, ∀ y : ℝ, f_D x ≤ f_D y) ∧
  ¬ (∃ x : ℝ, ∀ y : ℝ, f_B x ≤ f_B y) ∧
  ¬ (∃ x : ℝ, ∀ y : ℝ, f_C x ≤ f_C y) :=
by
  sorry

end min_value_functions_l60_60767


namespace circle_symmetric_line_l60_60879
-- Importing the entire Math library

-- Define the statement
theorem circle_symmetric_line (a : ℝ) :
  (∀ (A B : ℝ × ℝ), 
    (A.1)^2 + (A.2)^2 = 2 * a * (A.1) 
    ∧ (B.1)^2 + (B.2)^2 = 2 * a * (B.1) 
    ∧ A.2 = 2 * A.1 + 1 
    ∧ B.2 = 2 * B.1 + 1 
    ∧ A.2 = B.2) 
  → a = -1/2 :=
by
  sorry

end circle_symmetric_line_l60_60879


namespace solve_z_squared_eq_l60_60807

open Complex

theorem solve_z_squared_eq : 
  ∀ z : ℂ, z^2 = -100 - 64 * I → (z = 4 - 8 * I ∨ z = -4 + 8 * I) :=
by
  sorry

end solve_z_squared_eq_l60_60807


namespace largest_number_is_40_l60_60499

theorem largest_number_is_40 
    (a b c : ℕ) 
    (h1 : a ≠ b)
    (h2 : b ≠ c)
    (h3 : a ≠ c)
    (h4 : a + b + c = 100)
    (h5 : c - b = 8)
    (h6 : b - a = 4) : c = 40 :=
sorry

end largest_number_is_40_l60_60499


namespace range_of_a_l60_60504

open Real

theorem range_of_a (a : ℝ) 
  (h : ¬ ∃ x₀ : ℝ, 2 ^ x₀ - 2 ≤ a ^ 2 - 3 * a) : 1 ≤ a ∧ a ≤ 2 := 
by
  sorry

end range_of_a_l60_60504


namespace inequality_abc_l60_60866

theorem inequality_abc {a b c : ℝ} {n : ℕ} 
  (ha : 0 < a ∧ a ≤ 1) (hb : 0 < b ∧ b ≤ 1) (hc : 0 < c ∧ c ≤ 1) (hn : 0 < n) :
  (1 / (1 + a)^(1 / n : ℝ)) + (1 / (1 + b)^(1 / n : ℝ)) + (1 / (1 + c)^(1 / n : ℝ)) 
  ≤ 3 / (1 + (a * b * c)^(1 / 3 : ℝ))^(1 / n : ℝ) := sorry

end inequality_abc_l60_60866


namespace range_of_f_l60_60868

noncomputable def f (t : ℝ) : ℝ := (t^2 + 2 * t) / (t^2 + 2)

theorem range_of_f : Set.range f = Set.Icc (-1 : ℝ) 2 :=
sorry

end range_of_f_l60_60868


namespace circumference_of_wheels_l60_60647

-- Define the variables and conditions
variables (x y : ℝ)

def condition1 (x y : ℝ) : Prop := (120 / x) - (120 / y) = 6
def condition2 (x y : ℝ) : Prop := (4 / 5) * (120 / x) - (5 / 6) * (120 / y) = 4

-- The main theorem to prove
theorem circumference_of_wheels (x y : ℝ) (h1 : condition1 x y) (h2 : condition2 x y) : x = 4 ∧ y = 5 :=
  sorry  -- Proof is omitted

end circumference_of_wheels_l60_60647


namespace find_m_and_max_profit_l60_60482

theorem find_m_and_max_profit (m : ℝ) (y : ℝ) (x : ℝ) (ln : ℝ → ℝ) 
    (h1 : y = m * ln x - 1 / 100 * x ^ 2 + 101 / 50 * x + ln 10)
    (h2 : 10 < x) 
    (h3 : y = 35.7) 
    (h4 : x = 20)
    (ln_2 : ln 2 = 0.7) 
    (ln_5 : ln 5 = 1.6) :
    m = -1 ∧ ∃ x, (x = 50 ∧ (-ln x - 1 / 100 * x ^ 2 + 51 / 50 * x + ln 10 - x) = 24.4) := by
  sorry

end find_m_and_max_profit_l60_60482


namespace sides_of_length_five_l60_60432

theorem sides_of_length_five (GH HI : ℝ) (L : ℝ) (total_perimeter : ℝ) :
  GH = 7 → HI = 5 → total_perimeter = 38 → (∃ n m : ℕ, n + m = 6 ∧ n * 7 + m * 5 = 38 ∧ m = 2) := by
  intros hGH hHI hPerimeter
  sorry

end sides_of_length_five_l60_60432


namespace comparison_l60_60564

noncomputable def a := Real.log 3000 / Real.log 9
noncomputable def b := Real.log 2023 / Real.log 4
noncomputable def c := (11 * Real.exp (0.01 * Real.log 1.001)) / 2

theorem comparison : a < b ∧ b < c :=
by
  sorry

end comparison_l60_60564


namespace product_plus_one_is_square_l60_60573

theorem product_plus_one_is_square (x y : ℕ) (h : x * y = (x + 2) * (y - 2)) : ∃ k : ℕ, x * y + 1 = k * k :=
by
  sorry

end product_plus_one_is_square_l60_60573


namespace age_difference_l60_60664

theorem age_difference 
  (A B : ℤ) 
  (h1 : B = 39) 
  (h2 : A + 10 = 2 * (B - 10)) :
  A - B = 9 := 
by 
  sorry

end age_difference_l60_60664


namespace towels_per_load_l60_60457

-- Defining the given conditions
def total_towels : ℕ := 42
def number_of_loads : ℕ := 6

-- Defining the problem statement: Prove the number of towels per load
theorem towels_per_load : total_towels / number_of_loads = 7 := by 
  sorry

end towels_per_load_l60_60457


namespace even_sum_sufficient_not_necessary_l60_60695

theorem even_sum_sufficient_not_necessary (m n : ℤ) : 
  (∀ m n : ℤ, (Even m ∧ Even n) → Even (m + n)) 
  ∧ (∀ a b : ℤ, Even (a + b) → ¬ (Odd a ∧ Odd b)) :=
by
  sorry

end even_sum_sufficient_not_necessary_l60_60695


namespace initialNumberMembers_l60_60912

-- Define the initial number of members in the group
def initialMembers (n : ℕ) : Prop :=
  let W := n * 48 -- Initial total weight
  let newWeight := W + 78 + 93 -- New total weight after two members join
  let newAverageWeight := (n + 2) * 51 -- New total weight based on the new average weight
  newWeight = newAverageWeight -- The condition that the new total weights are equal

-- Theorem stating that the initial number of members is 23
theorem initialNumberMembers : initialMembers 23 :=
by
  -- Placeholder for proof steps
  sorry

end initialNumberMembers_l60_60912


namespace original_fraction_eq_two_thirds_l60_60823

theorem original_fraction_eq_two_thirds (a b : ℕ) (h : (a^3 : ℚ) / (b + 3) = 2 * (a / b)) : a = 2 ∧ b = 3 :=
by {
  sorry
}

end original_fraction_eq_two_thirds_l60_60823


namespace expression_equals_minus_0p125_l60_60123

-- Define the expression
def compute_expression : ℝ := 0.125^8 * (-8)^7

-- State the theorem to prove
theorem expression_equals_minus_0p125 : compute_expression = -0.125 :=
by {
  sorry
}

end expression_equals_minus_0p125_l60_60123


namespace calculate_expr_l60_60395

variable (x y : ℝ)
variable (hx : x ≠ 0)
variable (hy : y ≠ 0)

theorem calculate_expr : ((x^3 * y^2)^2 * (x / y^3)) = x^7 * y :=
by sorry

end calculate_expr_l60_60395


namespace system_no_solution_iff_n_eq_neg_half_l60_60040

theorem system_no_solution_iff_n_eq_neg_half (x y z n : ℝ) :
  (¬ ∃ x y z, 2 * n * x + y = 2 ∧ n * y + 2 * z = 2 ∧ x + 2 * n * z = 2) ↔ n = -1/2 := by
  sorry

end system_no_solution_iff_n_eq_neg_half_l60_60040


namespace intersection_point_interval_l60_60141

theorem intersection_point_interval (x₀ : ℝ) (h : x₀^3 = 2^x₀ + 1) : 
  1 < x₀ ∧ x₀ < 2 :=
by
  sorry

end intersection_point_interval_l60_60141


namespace charts_per_associate_professor_l60_60965

-- Definitions
def A : ℕ := 3
def B : ℕ := 4
def C : ℕ := 1

-- Conditions based on the given problem
axiom h1 : 2 * A + B = 10
axiom h2 : A * C + 2 * B = 11
axiom h3 : A + B = 7

-- The theorem to be proven
theorem charts_per_associate_professor : C = 1 := by
  sorry

end charts_per_associate_professor_l60_60965


namespace num_positive_integer_N_l60_60746

def num_valid_N : Nat := 7

theorem num_positive_integer_N (N : Nat) (h_pos : N > 0) :
  (∃ k : Nat, k > 3 ∧ N = k - 3 ∧ 48 % k = 0) ↔ (N < 45) ∧ (num_valid_N = 7) := 
by
sorry

end num_positive_integer_N_l60_60746


namespace Q_evaluation_at_2_l60_60191

noncomputable def Q : Polynomial ℚ := 
  (Polynomial.X^2 + Polynomial.C 4)^2

theorem Q_evaluation_at_2 : 
  Q.eval 2 = 64 :=
by 
  -- We'll skip the proof as per the instructions.
  sorry

end Q_evaluation_at_2_l60_60191


namespace inequality_solution_l60_60024

-- Definitions
variables {a b : ℝ}

-- Hypothesis
variable (h : a > b)

-- Theorem
theorem inequality_solution : -2 * a < -2 * b :=
sorry

end inequality_solution_l60_60024


namespace spoiled_milk_percentage_l60_60051

theorem spoiled_milk_percentage (p_egg p_flour p_all_good : ℝ) (h_egg : p_egg = 0.40) (h_flour : p_flour = 0.75) (h_all_good : p_all_good = 0.24) : 
  (1 - (p_all_good / (p_egg * p_flour))) = 0.20 :=
by
  sorry

end spoiled_milk_percentage_l60_60051


namespace range_of_a_l60_60639

theorem range_of_a 
  (f : ℝ → ℝ)
  (h_even : ∀ x, -5 ≤ x ∧ x ≤ 5 → f x = f (-x))
  (h_decreasing : ∀ a b, 0 ≤ a ∧ a < b ∧ b ≤ 5 → f b < f a)
  (h_inequality : ∀ a, f (2 * a + 3) < f a) :
  ∀ a, -5 ≤ a ∧ a ≤ 5 → a ∈ (Set.Icc (-4) (-3) ∪ Set.Ioc (-1) 1) := 
by
  sorry

end range_of_a_l60_60639


namespace train_speed_is_300_kmph_l60_60985

noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  (length / 1000) / (time / 3600)

theorem train_speed_is_300_kmph :
  train_speed 1250 15 = 300 := by
  sorry

end train_speed_is_300_kmph_l60_60985


namespace mean_median_difference_l60_60014

open Real

/-- In a class of 100 students, these are the distributions of scores:
  - 10% scored 60 points
  - 30% scored 75 points
  - 25% scored 80 points
  - 20% scored 90 points
  - 15% scored 100 points

Prove that the difference between the mean and the median scores is 1.5. -/
theorem mean_median_difference :
  let total_students := 100 
  let score_60 := 0.10 * total_students
  let score_75 := 0.30 * total_students
  let score_80 := 0.25 * total_students
  let score_90 := 0.20 * total_students
  let score_100 := (100 - (score_60 + score_75 + score_80 + score_90))
  let median := 80
  let mean := (60 * score_60 + 75 * score_75 + 80 * score_80 + 90 * score_90 + 100 * score_100) / total_students
  mean - median = 1.5 :=
by
  sorry

end mean_median_difference_l60_60014


namespace f_g_of_4_l60_60280

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt x + 12 / Real.sqrt x
noncomputable def g (x : ℝ) : ℝ := 3 * x^2 - x - 4

theorem f_g_of_4 : f (g 4) = 23 * Real.sqrt 10 / 5 := by
  sorry

end f_g_of_4_l60_60280


namespace find_b_l60_60921

-- Conditions
variables (a b c : ℝ) (A B C : ℝ)
variables (h_area : (1/2) * a * c * (Real.sin B) = sqrt 3)
variables (h_B : B = Real.pi / 3)
variables (h_relation : a^2 + c^2 = 3 * a * c)

-- Claim
theorem find_b :
    b = 2 * Real.sqrt 2 :=
  sorry

end find_b_l60_60921


namespace probability_ace_spades_then_king_spades_l60_60423

theorem probability_ace_spades_then_king_spades :
  ∃ (p : ℚ), (p = 1/52 * 1/51) := sorry

end probability_ace_spades_then_king_spades_l60_60423


namespace sequence_AMS_ends_in_14_l60_60967

def start := 3
def add_two (x : ℕ) := x + 2
def multiply_three (x : ℕ) := x * 3
def subtract_one (x : ℕ) := x - 1

theorem sequence_AMS_ends_in_14 : 
  subtract_one (multiply_three (add_two start)) = 14 :=
by
  -- The proof would go here if required.
  sorry

end sequence_AMS_ends_in_14_l60_60967


namespace initial_pinecones_l60_60265

theorem initial_pinecones (P : ℝ) :
  (0.20 * P + 2 * 0.20 * P + 0.25 * (0.40 * P) = 0.70 * P - 0.10 * P) ∧ (0.30 * P = 600) → P = 2000 :=
by
  intro h
  sorry

end initial_pinecones_l60_60265


namespace leakage_empty_time_l60_60488

variables (a : ℝ) (h1 : a > 0) -- Assuming a is positive for the purposes of the problem

theorem leakage_empty_time (h : 7 * a > 0) : (7 * a) / 6 = 7 * a / 6 :=
by
  sorry

end leakage_empty_time_l60_60488


namespace find_c_l60_60121

theorem find_c (b c : ℤ) (H : (b - 4) / (2 * b + 42) = c / 6) : c = 2 := 
sorry

end find_c_l60_60121


namespace flooring_area_already_installed_l60_60896

variable (living_room_length : ℕ) (living_room_width : ℕ) 
variable (flooring_sqft_per_box : ℕ)
variable (remaining_boxes_needed : ℕ)
variable (already_installed : ℕ)

theorem flooring_area_already_installed 
  (h1 : living_room_length = 16)
  (h2 : living_room_width = 20)
  (h3 : flooring_sqft_per_box = 10)
  (h4 : remaining_boxes_needed = 7)
  (h5 : living_room_length * living_room_width = 320)
  (h6 : already_installed = 320 - remaining_boxes_needed * flooring_sqft_per_box) : 
  already_installed = 250 :=
by
  sorry

end flooring_area_already_installed_l60_60896


namespace geometric_sequence_b_l60_60733

theorem geometric_sequence_b (b : ℝ) (h1 : b > 0) (h2 : 30 * (b / 30) = b) (h3 : b * (b / 30) = 9 / 4) :
  b = 3 * Real.sqrt 30 / 2 :=
by
  sorry

end geometric_sequence_b_l60_60733


namespace total_hours_worked_l60_60622

-- Definitions based on the conditions
def hours_per_day : ℕ := 3
def days_worked : ℕ := 6

-- Statement of the problem
theorem total_hours_worked : hours_per_day * days_worked = 18 := by
  sorry

end total_hours_worked_l60_60622


namespace side_length_S2_l60_60201

-- Define the variables
variables (r s : ℕ)

-- Given conditions
def condition1 : Prop := 2 * r + s = 2300
def condition2 : Prop := 2 * r + 3 * s = 4000

-- The main statement to be proven
theorem side_length_S2 (h1 : condition1 r s) (h2 : condition2 r s) : s = 850 := sorry

end side_length_S2_l60_60201


namespace min_k_value_l60_60058

noncomputable def minimum_k_condition (x y z k : ℝ) : Prop :=
  k * (x^2 - x + 1) * (y^2 - y + 1) * (z^2 - z + 1) ≥ (x * y * z)^2 - (x * y * z) + 1

theorem min_k_value :
  ∀ x y z : ℝ, x ≤ 0 → y ≤ 0 → z ≤ 0 → minimum_k_condition x y z (16 / 9) :=
by
  sorry

end min_k_value_l60_60058


namespace calculate_total_tulips_l60_60126

def number_of_red_tulips_for_eyes := 8 * 2
def number_of_purple_tulips_for_eyebrows := 5 * 2
def number_of_red_tulips_for_nose := 12
def number_of_red_tulips_for_smile := 18
def number_of_yellow_tulips_for_background := 9 * number_of_red_tulips_for_smile

def total_number_of_tulips : ℕ :=
  number_of_red_tulips_for_eyes + 
  number_of_red_tulips_for_nose + 
  number_of_red_tulips_for_smile + 
  number_of_purple_tulips_for_eyebrows + 
  number_of_yellow_tulips_for_background

theorem calculate_total_tulips : total_number_of_tulips = 218 := by
  sorry

end calculate_total_tulips_l60_60126


namespace exists_square_with_digit_sum_2002_l60_60956

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem exists_square_with_digit_sum_2002 :
  ∃ (n : ℕ), sum_of_digits (n^2) = 2002 :=
sorry

end exists_square_with_digit_sum_2002_l60_60956


namespace total_surface_area_of_cylinder_l60_60906

noncomputable def rectangle_length : ℝ := 4 * Real.pi
noncomputable def rectangle_width : ℝ := 2

noncomputable def cylinder_radius (length : ℝ) : ℝ := length / (2 * Real.pi)
noncomputable def cylinder_height (width : ℝ) : ℝ := width

noncomputable def cylinder_surface_area (radius height : ℝ) : ℝ :=
  2 * Real.pi * radius^2 + 2 * Real.pi * radius * height

theorem total_surface_area_of_cylinder :
  cylinder_surface_area (cylinder_radius rectangle_length) (cylinder_height rectangle_width) = 16 * Real.pi :=
by
  sorry

end total_surface_area_of_cylinder_l60_60906


namespace find_x_set_l60_60390

theorem find_x_set (x : ℝ) : ((x - 2) ^ 2 < 3 * x + 4) ↔ (0 ≤ x ∧ x < 7) := 
sorry

end find_x_set_l60_60390


namespace farmer_earns_from_runt_pig_l60_60263

def average_bacon_per_pig : ℕ := 20
def price_per_pound : ℕ := 6
def runt_pig_bacon : ℕ := average_bacon_per_pig / 2
def total_money_made (bacon_pounds : ℕ) (price_per_pound : ℕ) : ℕ := bacon_pounds * price_per_pound

theorem farmer_earns_from_runt_pig :
  total_money_made runt_pig_bacon price_per_pound = 60 :=
sorry

end farmer_earns_from_runt_pig_l60_60263


namespace select_integers_divisible_l60_60447

theorem select_integers_divisible (k : ℕ) (s : Finset ℤ) (h₁ : s.card = 2 * 2^k - 1) :
  ∃ t : Finset ℤ, t ⊆ s ∧ t.card = 2^k ∧ (t.sum id) % 2^k = 0 :=
sorry

end select_integers_divisible_l60_60447


namespace iced_coffee_days_per_week_l60_60810

theorem iced_coffee_days_per_week (x : ℕ) (h1 : 5 * 4 = 20)
  (h2 : 20 * 52 = 1040)
  (h3 : 2 * x = 2 * x)
  (h4 : 52 * (2 * x) = 104 * x)
  (h5 : 1040 + 104 * x = 1040 + 104 * x)
  (h6 : 1040 + 104 * x - 338 = 1040 + 104 * x - 338)
  (h7 : (0.75 : ℝ) * (1040 + 104 * x) = 780 + 78 * x) :
  x = 3 :=
by
  sorry

end iced_coffee_days_per_week_l60_60810


namespace andrew_stamps_permits_l60_60638

theorem andrew_stamps_permits (n a T r permits : ℕ)
  (h1 : n = 2)
  (h2 : a = 3)
  (h3 : T = 8)
  (h4 : r = 50)
  (h5 : permits = (T - n * a) * r) :
  permits = 100 :=
by
  rw [h1, h2, h3, h4] at h5
  norm_num at h5
  exact h5

end andrew_stamps_permits_l60_60638


namespace calories_per_burger_l60_60657

-- Conditions given in the problem
def burgers_per_day : Nat := 3
def days : Nat := 2
def total_calories : Nat := 120

-- Total burgers Dimitri will eat in the given period
def total_burgers := burgers_per_day * days

-- Prove that the number of calories per burger is 20
theorem calories_per_burger : total_calories / total_burgers = 20 := 
by 
  -- Skipping the proof with 'sorry' as instructed
  sorry

end calories_per_burger_l60_60657


namespace time_taken_by_A_l60_60612

theorem time_taken_by_A (v_A v_B D t_A t_B : ℚ) (h1 : v_A / v_B = 3 / 4) 
  (h2 : t_A = t_B + 30) (h3 : t_A = D / v_A) (h4 : t_B = D / v_B) 
  : t_A = 120 := 
by 
  sorry

end time_taken_by_A_l60_60612


namespace A_can_give_C_start_l60_60666

noncomputable def start_A_can_give_C : ℝ :=
  let start_AB := 50
  let start_BC := 157.89473684210532
  start_AB + start_BC

theorem A_can_give_C_start :
  start_A_can_give_C = 207.89473684210532 :=
by
  sorry

end A_can_give_C_start_l60_60666


namespace cost_per_game_l60_60311

theorem cost_per_game 
  (x : ℝ)
  (shoe_rent : ℝ := 0.50)
  (total_money : ℝ := 12.80)
  (games : ℕ := 7)
  (h1 : total_money - shoe_rent = 12.30)
  (h2 : 7 * x = 12.30) :
  x = 1.76 := 
sorry

end cost_per_game_l60_60311


namespace daily_savings_amount_l60_60233

theorem daily_savings_amount (total_savings : ℕ) (days : ℕ) (daily_savings : ℕ)
  (h1 : total_savings = 12410)
  (h2 : days = 365)
  (h3 : total_savings = daily_savings * days) :
  daily_savings = 34 :=
sorry

end daily_savings_amount_l60_60233


namespace train_pass_bridge_time_l60_60052

noncomputable def length_of_train : ℝ := 485
noncomputable def length_of_bridge : ℝ := 140
noncomputable def speed_of_train_kmph : ℝ := 45 
noncomputable def speed_of_train_mps : ℝ := speed_of_train_kmph * (1000 / 3600)

theorem train_pass_bridge_time :
  (length_of_train + length_of_bridge) / speed_of_train_mps = 50 :=
by
  sorry

end train_pass_bridge_time_l60_60052


namespace coeff_x2_term_l60_60604

theorem coeff_x2_term (a b c d e f : ℕ) (h1 : a = 3) (h2 : b = 4) (h3 : c = 5) (h4 : d = 6) (h5 : e = 7) (h6 : f = 8) :
    (a * f + b * e * 1 + c * d) = 82 := 
by
    sorry

end coeff_x2_term_l60_60604


namespace periodic_even_function_l60_60502

open Real

noncomputable def f : ℝ → ℝ := sorry

theorem periodic_even_function (f : ℝ → ℝ)
  (h1 : ∀ x, f (x + 2) = f x)
  (h2 : ∀ x, f (-x) = f x)
  (h3 : ∀ x, 2 ≤ x ∧ x ≤ 3 → f x = x) :
  ∀ x, -2 ≤ x ∧ x ≤ 0 → f x = 3 - abs (x + 1) :=
sorry

end periodic_even_function_l60_60502


namespace trapezoid_area_l60_60393

-- Define the properties of the isosceles trapezoid
structure IsoscelesTrapezoid where
  leg : ℝ
  diagonal : ℝ
  longer_base : ℝ
  is_isosceles : True
  legs_equal : True

-- Provide the specific conditions of the problem
def trapezoid : IsoscelesTrapezoid := {
  leg := 40,
  diagonal := 50,
  longer_base := 60,
  is_isosceles := True.intro,
  legs_equal := True.intro
}

-- State the main theorem to translate the proof problem into Lean
theorem trapezoid_area (T : IsoscelesTrapezoid) : T = trapezoid →
  (∃ A : ℝ, A = (15000 - 2000 * Real.sqrt 11) / 9) :=
by
  intros h
  sorry

end trapezoid_area_l60_60393


namespace jane_chickens_l60_60110

-- Conditions
def eggs_per_chicken_per_week : ℕ := 6
def egg_price_per_dozen : ℕ := 2
def total_income_in_2_weeks : ℕ := 20

-- Mathematical problem
theorem jane_chickens : (total_income_in_2_weeks / egg_price_per_dozen) * 12 / (eggs_per_chicken_per_week * 2) = 10 :=
by
  sorry

end jane_chickens_l60_60110


namespace import_tax_applied_amount_l60_60521

theorem import_tax_applied_amount 
    (total_value : ℝ) 
    (import_tax_paid : ℝ)
    (tax_rate : ℝ) 
    (excess_amount : ℝ) 
    (condition1 : total_value = 2580) 
    (condition2 : import_tax_paid = 110.60) 
    (condition3 : tax_rate = 0.07) 
    (condition4 : import_tax_paid = tax_rate * (total_value - excess_amount)) : 
    excess_amount = 1000 :=
by
  sorry

end import_tax_applied_amount_l60_60521


namespace curve_of_polar_equation_is_line_l60_60084

theorem curve_of_polar_equation_is_line (r θ : ℝ) :
  (r = 1 / (Real.sin θ - Real.cos θ)) →
  ∃ (a b c : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧ ∀ (x y : ℝ), r * (Real.sin θ) = y ∧ r * (Real.cos θ) = x → a * x + b * y = c :=
by
  sorry

end curve_of_polar_equation_is_line_l60_60084


namespace min_value_of_reciprocals_l60_60072

theorem min_value_of_reciprocals {x y a b : ℝ} 
  (h1 : 8 * x - y - 4 ≤ 0)
  (h2 : x + y + 1 ≥ 0)
  (h3 : y - 4 * x ≤ 0)
  (h4 : 2 = a * (1 / 2) + b * 1)
  (ha : a > 0)
  (hb : b > 0) :
  (1 / a) + (1 / b) = 9 / 2 :=
sorry

end min_value_of_reciprocals_l60_60072


namespace area_of_square_is_correct_l60_60634

-- Define the nature of the problem setup and parameters
def radius_of_circle : ℝ := 7
def diameter_of_circle : ℝ := 2 * radius_of_circle
def side_length_of_square : ℝ := 2 * diameter_of_circle
def area_of_square : ℝ := side_length_of_square ^ 2

-- Statement of the problem to prove
theorem area_of_square_is_correct : area_of_square = 784 := by
  sorry

end area_of_square_is_correct_l60_60634


namespace airplane_total_luggage_weight_l60_60809

def num_people := 6
def bags_per_person := 5
def weight_per_bag := 50
def additional_bags := 90

def total_weight_people := num_people * bags_per_person * weight_per_bag
def total_weight_additional_bags := additional_bags * weight_per_bag

def total_luggage_weight := total_weight_people + total_weight_additional_bags

theorem airplane_total_luggage_weight : total_luggage_weight = 6000 :=
by
  sorry

end airplane_total_luggage_weight_l60_60809


namespace complex_powers_sum_zero_l60_60409

theorem complex_powers_sum_zero (i : ℂ) (h : i^2 = -1) : i^2023 + i^2024 + i^2025 + i^2026 = 0 :=
by
  sorry

end complex_powers_sum_zero_l60_60409


namespace total_pages_of_book_l60_60658

theorem total_pages_of_book (P : ℝ) (h : 0.4 * P = 16) : P = 40 :=
sorry

end total_pages_of_book_l60_60658


namespace smallest_rel_prime_to_180_l60_60808

theorem smallest_rel_prime_to_180 : ∃ x : ℕ, x > 1 ∧ Int.gcd x 180 = 1 ∧ ∀ y : ℕ, y > 1 ∧ Int.gcd y 180 = 1 → x ≤ y := 
sorry

end smallest_rel_prime_to_180_l60_60808


namespace mumu_identity_l60_60762

def f (m u : ℕ) : ℕ := 
  -- Assume f is correctly defined to match the number of valid Mumu words 
  -- involving m M's and u U's according to the problem's definition.
  sorry 

theorem mumu_identity (u m : ℕ) (h₁ : u ≥ 2) (h₂ : 3 ≤ m) (h₃ : m ≤ 2 * u) :
  f m u = f (2 * u - m + 1) u ↔ f m (u - 1) = f (2 * u - m + 1) (u - 1) :=
by
  sorry

end mumu_identity_l60_60762


namespace distance_from_A_to_C_l60_60269

theorem distance_from_A_to_C (x y : ℕ) (d : ℚ)
  (h1 : d = x / 3) 
  (h2 : 13 + (d * 15) / (y - 13) = 2 * x)
  (h3 : y = 2 * x + 13) 
  : x + y = 26 := 
  sorry

end distance_from_A_to_C_l60_60269


namespace A_union_B_eq_B_l60_60991

-- Define set A
def A : Set ℝ := {-1, 0, 1}

-- Define set B
def B : Set ℝ := {y | ∃ x : ℝ, y = Real.sin x}

-- The proof problem
theorem A_union_B_eq_B : A ∪ B = B := 
  sorry

end A_union_B_eq_B_l60_60991


namespace eraser_cost_l60_60490

theorem eraser_cost (initial_money : ℕ) (scissors_count : ℕ) (scissors_price : ℕ) (erasers_count : ℕ) (remaining_money : ℕ) :
    initial_money = 100 →
    scissors_count = 8 →
    scissors_price = 5 →
    erasers_count = 10 →
    remaining_money = 20 →
    (initial_money - scissors_count * scissors_price - remaining_money) / erasers_count = 4 :=
by
  intro h1 h2 h3 h4 h5
  sorry

end eraser_cost_l60_60490


namespace middle_number_of_five_consecutive_numbers_l60_60709

theorem middle_number_of_five_consecutive_numbers (n : ℕ) 
  (h : (n - 2) + (n - 1) + n + (n + 1) + (n + 2) = 60) : n = 12 :=
by
  sorry

end middle_number_of_five_consecutive_numbers_l60_60709


namespace identity_proof_l60_60780

theorem identity_proof : 
  ∀ x : ℝ, 
    (x^2 + 3*x + 2) * (x + 3) = (x + 1) * (x^2 + 5*x + 6) := 
by 
  sorry

end identity_proof_l60_60780


namespace percentage_female_on_duty_l60_60624

-- Definitions as per conditions in the problem:
def total_on_duty : ℕ := 240
def female_on_duty := total_on_duty / 2 -- Half of those on duty are female
def total_female_officers : ℕ := 300
def percentage_of_something (part : ℕ) (whole : ℕ) : ℕ := (part * 100) / whole

-- Statement of the problem to prove
theorem percentage_female_on_duty : percentage_of_something female_on_duty total_female_officers = 40 :=
by
  sorry

end percentage_female_on_duty_l60_60624


namespace musketeers_strength_order_l60_60272

variables {A P R D : ℝ}

theorem musketeers_strength_order 
  (h1 : P + D > A + R)
  (h2 : P + A > R + D)
  (h3 : P + R = A + D) : 
  P > D ∧ D > A ∧ A > R :=
by
  sorry

end musketeers_strength_order_l60_60272


namespace func_inequality_l60_60507

noncomputable def f (a b c x : ℝ) : ℝ := a * x ^ 2 + b * x + c

-- Given function properties
variables {a b c : ℝ} (h_a : a > 0) (symmetry : ∀ x : ℝ, f a b c (2 + x) = f a b c (2 - x))

theorem func_inequality : f a b c 2 < f a b c 1 ∧ f a b c 1 < f a b c 4 :=
by
  sorry

end func_inequality_l60_60507


namespace books_count_l60_60414

theorem books_count (books_per_box : ℕ) (boxes : ℕ) (total_books : ℕ) 
  (h1 : books_per_box = 3)
  (h2 : boxes = 8)
  (h3 : total_books = books_per_box * boxes) : 
  total_books = 24 := 
by 
  rw [h1, h2] at h3
  exact h3

end books_count_l60_60414


namespace years_in_future_l60_60079

theorem years_in_future (Shekhar Shobha : ℕ) (h1 : Shekhar / Shobha = 4 / 3) (h2 : Shobha = 15) (h3 : Shekhar + t = 26)
  : t = 6 :=
by
  sorry

end years_in_future_l60_60079


namespace area_of_10th_square_l60_60227

noncomputable def area_of_square (n: ℕ) : ℚ :=
  if n = 1 then 4
  else 2 * (1 / 2)^(n - 1)

theorem area_of_10th_square : area_of_square 10 = 1 / 256 := 
  sorry

end area_of_10th_square_l60_60227


namespace volume_of_solid_l60_60860

noncomputable def s : ℝ := 2 * Real.sqrt 2

noncomputable def h : ℝ := 3 * s

noncomputable def base_area (a b : ℝ) : ℝ := 1 / 2 * a * b

noncomputable def volume (base_area height : ℝ) : ℝ := base_area * height

theorem volume_of_solid : volume (base_area s s) h = 24 * Real.sqrt 2 :=
by
  -- The proof will go here
  sorry

end volume_of_solid_l60_60860


namespace find_b_minus_a_l60_60959

theorem find_b_minus_a (a b : ℝ) (h : ∀ x : ℝ, 0 ≤ x → 
  0 ≤ x^4 - x^3 + a * x + b ∧ x^4 - x^3 + a * x + b ≤ (x^2 - 1)^2) : 
  b - a = 2 :=
sorry

end find_b_minus_a_l60_60959


namespace overall_percentage_decrease_l60_60536

theorem overall_percentage_decrease (P x y : ℝ) (hP : P = 100) 
  (h : (P - (x / 100) * P) - (y / 100) * (P - (x / 100) * P) = 55) : 
  ((P - 55) / P) * 100 = 45 := 
by 
  sorry

end overall_percentage_decrease_l60_60536


namespace solve_for_x_l60_60371

theorem solve_for_x (x : ℚ) : (x + 4) / (x - 3) = (x - 2) / (x + 2) -> x = -2 / 11 := by
  sorry

end solve_for_x_l60_60371


namespace trigonometric_identity_l60_60524

open Real

theorem trigonometric_identity
  (α β γ φ : ℝ)
  (h1 : sin α + 7 * sin β = 4 * (sin γ + 2 * sin φ))
  (h2 : cos α + 7 * cos β = 4 * (cos γ + 2 * cos φ)) :
  2 * cos (α - φ) = 7 * cos (β - γ) :=
by sorry

end trigonometric_identity_l60_60524


namespace no_partition_exists_l60_60757

theorem no_partition_exists : ¬ ∃ (x y : ℕ), 
    (1 ≤ x ∧ x ≤ 15) ∧ 
    (1 ≤ y ∧ y ≤ 15) ∧ 
    (x * y = 120 - x - y) :=
by
  sorry

end no_partition_exists_l60_60757


namespace expected_dietary_restriction_l60_60077

theorem expected_dietary_restriction (n : ℕ) (p : ℚ) (sample_size : ℕ) (expected : ℕ) :
  p = 1 / 4 ∧ sample_size = 300 ∧ expected = sample_size * p → expected = 75 := by
  sorry

end expected_dietary_restriction_l60_60077


namespace second_term_is_4_l60_60898

-- Define the arithmetic sequence conditions
variables (a d : ℝ) -- first term a, common difference d

-- The condition given in the problem
def sum_first_and_third_term (a d : ℝ) : Prop :=
  a + (a + 2 * d) = 8

-- What we need to prove
theorem second_term_is_4 (a d : ℝ) (h : sum_first_and_third_term a d) : a + d = 4 :=
sorry

end second_term_is_4_l60_60898


namespace uncovered_area_is_52_l60_60189

-- Define the dimensions of the rectangles
def smaller_rectangle_length : ℕ := 4
def smaller_rectangle_width : ℕ := 2
def larger_rectangle_length : ℕ := 10
def larger_rectangle_width : ℕ := 6

-- Define the areas of both rectangles
def area_larger_rectangle : ℕ := larger_rectangle_length * larger_rectangle_width
def area_smaller_rectangle : ℕ := smaller_rectangle_length * smaller_rectangle_width

-- Define the area of the uncovered region
def area_uncovered_region : ℕ := area_larger_rectangle - area_smaller_rectangle

-- State the theorem
theorem uncovered_area_is_52 : area_uncovered_region = 52 := by sorry

end uncovered_area_is_52_l60_60189


namespace speed_of_B_l60_60813

theorem speed_of_B 
  (A_speed : ℝ)
  (t1 : ℝ)
  (t2 : ℝ)
  (d1 := A_speed * t1)
  (d2 := A_speed * t2)
  (total_distance := d1 + d2)
  (B_speed := total_distance / t2) :
  A_speed = 7 → 
  t1 = 0.5 → 
  t2 = 1.8 →
  B_speed = 8.944 :=
by
  intros h1 h2 h3
  simp [h1, h2, h3]
  exact sorry

end speed_of_B_l60_60813


namespace original_number_of_candies_l60_60970

theorem original_number_of_candies (x : ℝ) (h₀ : x * (0.7 ^ 3) = 40) : x = 117 :=
by 
  sorry

end original_number_of_candies_l60_60970


namespace common_difference_arithmetic_sequence_l60_60086

theorem common_difference_arithmetic_sequence (d : ℝ) :
  (∀ (n : ℝ) (a_1 : ℝ), a_1 = 9 ∧
  (∃ a₄ a₈ : ℝ, a₄ = a_1 + 3 * d ∧ a₈ = a_1 + 7 * d ∧ a₄ = (a_1 * a₈)^(1/2)) →
  d = 1) :=
sorry

end common_difference_arithmetic_sequence_l60_60086


namespace sum_of_15th_set_l60_60817

def first_element_of_set (n : ℕ) : ℕ :=
  3 + (n * (n - 1)) / 2

def sum_of_elements_in_set (n : ℕ) : ℕ :=
  let a_n := first_element_of_set n
  let l_n := a_n + n - 1
  n * (a_n + l_n) / 2

theorem sum_of_15th_set :
  sum_of_elements_in_set 15 = 1725 :=
by
  sorry

end sum_of_15th_set_l60_60817


namespace senior_ticket_cost_l60_60303

theorem senior_ticket_cost (total_tickets : ℕ) (adult_ticket_cost : ℕ) (total_receipts : ℕ) (senior_tickets : ℕ) (senior_ticket_cost : ℕ) :
  total_tickets = 510 →
  adult_ticket_cost = 21 →
  total_receipts = 8748 →
  senior_tickets = 327 →
  senior_ticket_cost = 15 :=
by
  sorry

end senior_ticket_cost_l60_60303


namespace find_k_l60_60948

theorem find_k (x k : ℝ) (h1 : (x^2 - k) * (x + k) = x^3 + k * (x^2 - x - 2)) (h2 : k ≠ 0) :
  k = 2 :=
sorry

end find_k_l60_60948


namespace fraction_to_decimal_l60_60487

theorem fraction_to_decimal :
  (11:ℚ) / 16 = 0.6875 :=
by
  sorry

end fraction_to_decimal_l60_60487


namespace Jessica_has_3_dozens_l60_60148

variable (j : ℕ)

def Sandy_red_marbles (j : ℕ) : ℕ := 4 * j * 12  

theorem Jessica_has_3_dozens {j : ℕ} : Sandy_red_marbles j = 144 → j = 3 := by
  intros h
  sorry

end Jessica_has_3_dozens_l60_60148


namespace range_of_xy_l60_60619

theorem range_of_xy {x y : ℝ} (h₁ : 0 < x) (h₂ : 0 < y)
    (h₃ : x + 2/x + 3*y + 4/y = 10) : 
    1 ≤ x * y ∧ x * y ≤ 8 / 3 :=
by
  sorry

end range_of_xy_l60_60619


namespace equal_acutes_l60_60517

open Real

theorem equal_acutes (a b c : ℝ) (ha : 0 < a ∧ a < π / 2) (hb : 0 < b ∧ b < π / 2) (hc : 0 < c ∧ c < π / 2)
  (h1 : sin b = (sin a + sin c) / 2) (h2 : cos b ^ 2 = cos a * cos c) : a = b ∧ b = c := 
by
  -- We have to fill the proof steps here.
  sorry

end equal_acutes_l60_60517


namespace sin_angle_identity_l60_60838

theorem sin_angle_identity : 
  (Real.sin (Real.pi / 4) * Real.sin (7 * Real.pi / 12) + Real.sin (Real.pi / 4) * Real.sin (Real.pi / 12)) = Real.sqrt 3 / 2 := 
by 
  sorry

end sin_angle_identity_l60_60838


namespace triangle_square_side_ratio_l60_60822

theorem triangle_square_side_ratio :
  (∀ (a : ℝ), (a * 3 = 60) → (∀ (b : ℝ), (b * 4 = 60) → (a / b = 4 / 3))) :=
by
  intros a h1 b h2
  sorry

end triangle_square_side_ratio_l60_60822


namespace inches_per_foot_l60_60493

-- Definition of the conditions in the problem.
def feet_last_week := 6
def feet_less_this_week := 4
def total_inches := 96

-- Lean statement that proves the number of inches in a foot
theorem inches_per_foot : 
    (total_inches / (feet_last_week + (feet_last_week - feet_less_this_week))) = 12 := 
by sorry

end inches_per_foot_l60_60493


namespace remainder_equal_to_zero_l60_60600

def A : ℕ := 270
def B : ℕ := 180
def M : ℕ := 25
def R_A : ℕ := A % M
def R_B : ℕ := B % M
def A_squared_B : ℕ := (A ^ 2 * B) % M
def R_A_R_B : ℕ := (R_A * R_B) % M

theorem remainder_equal_to_zero (h1 : A = 270) (h2 : B = 180) (h3 : M = 25) 
    (h4 : R_A = 20) (h5 : R_B = 5) : 
    A_squared_B = 0 ∧ R_A_R_B = 0 := 
by {
    sorry
}

end remainder_equal_to_zero_l60_60600


namespace max_a_is_2_l60_60644

noncomputable def max_value_of_a (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 = 6) : ℝ :=
  2

theorem max_a_is_2 (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 = 6) :
  max_value_of_a a b c h1 h2 = 2 :=
sorry

end max_a_is_2_l60_60644


namespace polynomial_has_roots_l60_60363

-- Define the polynomial
def polynomial (x : ℂ) : ℂ := 7 * x^4 - 48 * x^3 + 93 * x^2 - 48 * x + 7

-- Theorem to prove the existence of roots for the polynomial equation
theorem polynomial_has_roots : ∃ x : ℂ, polynomial x = 0 := by
  sorry

end polynomial_has_roots_l60_60363


namespace mother_present_age_l60_60042

def person_present_age (P M : ℕ) : Prop :=
  P = (2 / 5) * M

def person_age_in_10_years (P M : ℕ) : Prop :=
  P + 10 = (1 / 2) * (M + 10)

theorem mother_present_age (P M : ℕ) (h1 : person_present_age P M) (h2 : person_age_in_10_years P M) : M = 50 :=
sorry

end mother_present_age_l60_60042


namespace expression_evaluation_l60_60671

theorem expression_evaluation :
  (0.8 ^ 3) - ((0.5 ^ 3) / (0.8 ^ 2)) + 0.40 + (0.5 ^ 2) = 0.9666875 := 
by 
  sorry

end expression_evaluation_l60_60671


namespace P_never_77_l60_60754

def P (x y : ℤ) : ℤ := x^5 - 4 * x^4 * y - 5 * y^2 * x^3 + 20 * y^3 * x^2 + 4 * y^4 * x - 16 * y^5

theorem P_never_77 (x y : ℤ) : P x y ≠ 77 := sorry

end P_never_77_l60_60754


namespace smallest_positive_integer_n_l60_60319

theorem smallest_positive_integer_n :
  ∃ (n: ℕ), n = 4 ∧ (∀ x: ℝ, (Real.sin x)^n + (Real.cos x)^n ≤ 2 / n) :=
sorry

end smallest_positive_integer_n_l60_60319


namespace cards_from_country_correct_l60_60015

def total_cards : ℝ := 403.0
def cards_from_home : ℝ := 287.0
def cards_from_country : ℝ := total_cards - cards_from_home

theorem cards_from_country_correct : cards_from_country = 116.0 := by
  -- proof to be added
  sorry

end cards_from_country_correct_l60_60015


namespace eggs_in_box_l60_60553

theorem eggs_in_box (initial_count : ℝ) (added_count : ℝ) (total_count : ℝ) 
  (h_initial : initial_count = 47.0) 
  (h_added : added_count = 5.0) : total_count = 52.0 :=
by 
  sorry

end eggs_in_box_l60_60553


namespace find_x_for_parallel_vectors_l60_60360

-- Definitions for the given conditions
def a : ℝ × ℝ := (4, 2)
def b (x : ℝ) : ℝ × ℝ := (x, 3)
def parallel (u v : ℝ × ℝ) : Prop := u.1 * v.2 = u.2 * v.1

-- The proof statement
theorem find_x_for_parallel_vectors (x : ℝ) (h : parallel a (b x)) : x = 6 :=
  sorry

end find_x_for_parallel_vectors_l60_60360


namespace eight_digit_number_divisibility_l60_60585

theorem eight_digit_number_divisibility (a b c d : ℕ) (Z : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ 9) 
(h3 : b ≤ 9) (h4 : c ≤ 9) (h5 : d ≤ 9) (hZ : Z = 1001 * (1000 * a + 100 * b + 10 * c + d)) : 
  10001 ∣ Z := 
  by sorry

end eight_digit_number_divisibility_l60_60585


namespace problem1_problem2_l60_60165

open Set

variable {x y z a b : ℝ}

-- Problem 1: Prove the inequality
theorem problem1 (x y z : ℝ) : 
  5 * x^2 + y^2 + z^2 ≥ 2 * x * y + 4 * x + 2 * z - 2 :=
by
  sorry

-- Problem 2: Prove the range of 10a - 5b is [−1, 20]
theorem problem2 (a b : ℝ) 
  (h1 : 1 ≤ 2 * a + b ∧ 2 * a + b ≤ 4)
  (h2 : -1 ≤ a - 2 * b ∧ a - 2 * b ≤ 2) : 
  -1 ≤ 10 * a - 5 * b ∧ 10 * a - 5 * b ≤ 20 :=
by
  sorry

end problem1_problem2_l60_60165


namespace combined_mpg_l60_60190

theorem combined_mpg :
  let mR := 150 -- miles Ray drives
  let mT := 300 -- miles Tom drives
  let mpgR := 50 -- miles per gallon for Ray's car
  let mpgT := 20 -- miles per gallon for Tom's car
  -- Total gasoline used by Ray and Tom
  let gR := mR / mpgR
  let gT := mT / mpgT
  -- Total distance driven
  let total_distance := mR + mT
  -- Total gasoline used
  let total_gasoline := gR + gT
  -- Combined miles per gallon
  let combined_mpg := total_distance / total_gasoline
  combined_mpg = 25 := by
    sorry

end combined_mpg_l60_60190


namespace leak_empty_time_l60_60144

theorem leak_empty_time
  (pump_fill_time : ℝ)
  (leak_fill_time : ℝ)
  (pump_fill_rate : pump_fill_time = 5)
  (leak_fill_rate : leak_fill_time = 10)
  : (1 / 5 - 1 / leak_fill_time)⁻¹ = 10 :=
by
  -- you can fill in the proof here
  sorry

end leak_empty_time_l60_60144


namespace division_multiplication_identity_l60_60138

theorem division_multiplication_identity :
  24 / (-6) * (3 / 2) / (- (4 / 3)) = 9 / 2 := 
by 
  sorry

end division_multiplication_identity_l60_60138


namespace number_of_vegetarians_l60_60932

-- Define the conditions
def only_veg : ℕ := 11
def only_nonveg : ℕ := 6
def both_veg_and_nonveg : ℕ := 9

-- Define the total number of vegetarians
def total_veg : ℕ := only_veg + both_veg_and_nonveg

-- The statement to be proved
theorem number_of_vegetarians : total_veg = 20 := 
by
  sorry

end number_of_vegetarians_l60_60932


namespace relative_prime_in_consecutive_integers_l60_60402

theorem relative_prime_in_consecutive_integers (n : ℤ) : 
  ∃ k, n ≤ k ∧ k ≤ n + 5 ∧ ∀ m, n ≤ m ∧ m ≤ n + 5 ∧ m ≠ k → Int.gcd k m = 1 :=
sorry

end relative_prime_in_consecutive_integers_l60_60402


namespace rectangle_area_l60_60888

theorem rectangle_area (r : ℝ) (w l : ℝ) (h_radius : r = 7) 
  (h_ratio : l = 3 * w) (h_width : w = 2 * r) : l * w = 588 :=
by
  sorry

end rectangle_area_l60_60888


namespace correct_option_is_D_l60_60184

noncomputable def expression1 (a b : ℝ) : Prop := a + b > 2 * b^2
noncomputable def expression2 (a b : ℝ) : Prop := a^5 + b^5 > a^3 * b^2 + a^2 * b^3
noncomputable def expression3 (a b : ℝ) : Prop := a^2 + b^2 ≥ 2 * (a - b - 1)
noncomputable def expression4 (a b : ℝ) : Prop := (b / a) + (a / b) > 2

theorem correct_option_is_D (a b : ℝ) (h : a ≠ b) : 
  (expression3 a b ∧ ¬expression1 a b ∧ ¬expression2 a b ∧ ¬expression4 a b) :=
by
  sorry

end correct_option_is_D_l60_60184


namespace total_selling_price_l60_60228

theorem total_selling_price (profit_per_meter cost_price_per_meter meters : ℕ)
  (h_profit : profit_per_meter = 20)
  (h_cost : cost_price_per_meter = 85)
  (h_meters : meters = 85) :
  (cost_price_per_meter + profit_per_meter) * meters = 8925 :=
by
  sorry

end total_selling_price_l60_60228


namespace ones_digit_of_9_pow_27_l60_60018

-- Definitions representing the cyclical pattern
def ones_digit_of_9_power (n : ℕ) : ℕ :=
  if n % 2 = 1 then 9 else 1

-- The problem statement to be proven
theorem ones_digit_of_9_pow_27 : ones_digit_of_9_power 27 = 9 := 
by
  -- the detailed proof steps are omitted
  sorry

end ones_digit_of_9_pow_27_l60_60018


namespace replaced_person_age_is_40_l60_60177

def average_age_decrease_replacement (T age_of_replaced: ℕ) : Prop :=
  let original_average := T / 10
  let new_total_age := T - age_of_replaced + 10
  let new_average := new_total_age / 10
  original_average - 3 = new_average

theorem replaced_person_age_is_40 (T : ℕ) (h : average_age_decrease_replacement T 40) : Prop :=
  ∀ age_of_replaced, age_of_replaced = 40 → average_age_decrease_replacement T age_of_replaced

-- To actually formalize the proof, you can use the following structure:
-- proof by calculation omitted
lemma replaced_person_age_is_40_proof (T : ℕ) (h : average_age_decrease_replacement T 40) : 
  replaced_person_age_is_40 T h :=
by
  sorry

end replaced_person_age_is_40_l60_60177


namespace initial_alloy_weight_l60_60062

theorem initial_alloy_weight
  (x : ℝ)  -- Weight of the initial alloy in ounces
  (h1 : 0.80 * (x + 24) = 0.50 * x + 24)  -- Equation derived from conditions
: x = 16 := 
sorry

end initial_alloy_weight_l60_60062


namespace mary_change_l60_60784

/-- 
Calculate the change Mary will receive after buying tickets for herself and her 3 children 
at the circus, given the ticket prices and special group rate discount.
-/
theorem mary_change :
  let adult_ticket := 2
  let child_ticket := 1
  let discounted_child_ticket := 0.5 * child_ticket
  let total_cost_with_discount := adult_ticket + 2 * child_ticket + discounted_child_ticket
  let payment := 20
  payment - total_cost_with_discount = 15.50 :=
by
  sorry

end mary_change_l60_60784


namespace find_vertex_parabola_l60_60590

-- Define the quadratic equation of the parabola
def parabola_eq (x y : ℝ) : Prop := x^2 - 4 * x + 3 * y + 10 = 0

-- Definition of the vertex of the parabola
def is_vertex (v : ℝ × ℝ) : Prop :=
  ∀ (x y : ℝ), parabola_eq x y → v = (2, -2)

-- The main statement we want to prove
theorem find_vertex_parabola : 
  ∃ v : ℝ × ℝ, is_vertex v :=
by
  use (2, -2)
  intros x y hyp
  sorry

end find_vertex_parabola_l60_60590


namespace min_area_of_triangle_l60_60212

noncomputable def area_of_triangle (p q : ℤ) : ℚ :=
  (1 / 2 : ℚ) * abs (3 * p - 5 * q)

theorem min_area_of_triangle :
  (∀ p q : ℤ, p ≠ 0 ∨ q ≠ 0 → area_of_triangle p q ≥ (1 / 2 : ℚ)) ∧
  (∃ p q : ℤ, p ≠ 0 ∨ q ≠ 0 ∧ area_of_triangle p q = (1 / 2 : ℚ)) := 
by { 
  sorry 
}

end min_area_of_triangle_l60_60212


namespace students_exceed_rabbits_l60_60558

theorem students_exceed_rabbits (students_per_classroom rabbits_per_classroom number_of_classrooms : ℕ) 
  (h_students : students_per_classroom = 18)
  (h_rabbits : rabbits_per_classroom = 2)
  (h_classrooms : number_of_classrooms = 4) : 
  (students_per_classroom * number_of_classrooms) - (rabbits_per_classroom * number_of_classrooms) = 64 :=
by {
  sorry
}

end students_exceed_rabbits_l60_60558


namespace cost_of_coat_eq_l60_60871

-- Define the given conditions
def total_cost : ℕ := 110
def cost_of_shoes : ℕ := 30
def cost_per_jeans : ℕ := 20
def num_of_jeans : ℕ := 2

-- Define the cost calculation for the jeans
def cost_of_jeans : ℕ := num_of_jeans * cost_per_jeans

-- Define the known total cost (shoes and jeans)
def known_total_cost : ℕ := cost_of_shoes + cost_of_jeans

-- Prove James' coat cost
theorem cost_of_coat_eq :
  (total_cost - known_total_cost) = 40 :=
by
  sorry

end cost_of_coat_eq_l60_60871


namespace factorial_division_identity_l60_60986

theorem factorial_division_identity: (Nat.factorial 10) / ((Nat.factorial 7) * (Nat.factorial 3)) = 120 := by
  sorry

end factorial_division_identity_l60_60986


namespace goose_eggs_count_l60_60153

theorem goose_eggs_count (E : ℕ)
  (h1 : (2/3 : ℚ) * E ≥ 0)
  (h2 : (3/4 : ℚ) * (2/3 : ℚ) * E ≥ 0)
  (h3 : 100 = (2/5 : ℚ) * (3/4 : ℚ) * (2/3 : ℚ) * E) :
  E = 500 := by
  sorry

end goose_eggs_count_l60_60153


namespace funfair_initial_visitors_l60_60642

theorem funfair_initial_visitors {a : ℕ} (ha1 : 50 * a - 40 > 0) (ha2 : 90 - 20 * a > 0) (ha3 : 50 * a - 40 > 90 - 20 * a) :
  (50 * a - 40 = 60) ∨ (50 * a - 40 = 110) ∨ (50 * a - 40 = 160) :=
sorry

end funfair_initial_visitors_l60_60642


namespace no_integer_solution_l60_60683

theorem no_integer_solution : ¬ ∃ (x y : ℤ), x^2 - 7 * y = 10 :=
by
  sorry

end no_integer_solution_l60_60683


namespace peter_pizza_total_l60_60305

theorem peter_pizza_total (total_slices : ℕ) (whole_slice : ℕ) (shared_slice : ℚ) (shared_parts : ℕ) :
  total_slices = 16 ∧ whole_slice = 1 ∧ shared_parts = 3 ∧ shared_slice = 1 / (total_slices * shared_parts) →
  whole_slice / total_slices + shared_slice = 1 / 12 :=
by
  sorry

end peter_pizza_total_l60_60305


namespace centroid_calculation_correct_l60_60467

-- Define the vertices of the triangle
def P : ℝ × ℝ := (2, 3)
def Q : ℝ × ℝ := (-1, 4)
def R : ℝ × ℝ := (4, -2)

-- Define the coordinates of the centroid
noncomputable def S : ℝ × ℝ := ((P.1 + Q.1 + R.1) / 3, (P.2 + Q.2 + R.2) / 3)

-- Prove that 7x + 2y = 15 for the centroid
theorem centroid_calculation_correct : 7 * S.1 + 2 * S.2 = 15 :=
by 
  -- Placeholder for the proof steps
  sorry

end centroid_calculation_correct_l60_60467


namespace max_curved_sides_l60_60463

theorem max_curved_sides (n : ℕ) (h : 2 ≤ n) : 
  ∃ m, m = 2 * n - 2 :=
sorry

end max_curved_sides_l60_60463


namespace shop_owner_cheat_percentage_l60_60151

def CP : ℝ := 100
def cheating_buying : ℝ := 0.15  -- 15% cheating
def actual_cost_price : ℝ := CP * (1 + cheating_buying)  -- $115
def profit_percentage : ℝ := 43.75

theorem shop_owner_cheat_percentage :
  ∃ x : ℝ, profit_percentage = ((CP - x * CP / 100 - actual_cost_price) / actual_cost_price * 100) ∧ x = 65.26 :=
by
  sorry

end shop_owner_cheat_percentage_l60_60151


namespace train_speed_l60_60199

theorem train_speed (length_train length_bridge time_crossing speed : ℝ)
  (h1 : length_train = 100)
  (h2 : length_bridge = 300)
  (h3 : time_crossing = 24)
  (h4 : speed = (length_train + length_bridge) / time_crossing) :
  speed = 16.67 := 
sorry

end train_speed_l60_60199


namespace solve_quadratic_eq_solve_linear_system_l60_60279

theorem solve_quadratic_eq (x : ℚ) : 4 * (x - 1) ^ 2 - 25 = 0 ↔ x = 7 / 2 ∨ x = -3 / 2 := 
by sorry

theorem solve_linear_system (x y : ℚ) : (2 * x - y = 4) ∧ (3 * x + 2 * y = 1) ↔ (x = 9 / 7 ∧ y = -10 / 7) :=
by sorry

end solve_quadratic_eq_solve_linear_system_l60_60279


namespace log_eq_solution_l60_60678

open Real

noncomputable def solve_log_eq : Real :=
  let x := 62.5^(1/3)
  x

theorem log_eq_solution (x : Real) (hx : 3 * log x - 4 * log 5 = -1) :
  x = solve_log_eq :=
by
  sorry

end log_eq_solution_l60_60678


namespace convert_seven_cubic_yards_l60_60449

-- Define the conversion factor from yards to feet
def yardToFeet : ℝ := 3
-- Define the conversion factor from cubic yards to cubic feet
def cubicYardToCubicFeet : ℝ := yardToFeet ^ 3
-- Define the conversion function from cubic yards to cubic feet
noncomputable def convertVolume (volumeInCubicYards : ℝ) : ℝ :=
  volumeInCubicYards * cubicYardToCubicFeet

-- Statement to prove: 7 cubic yards is equivalent to 189 cubic feet
theorem convert_seven_cubic_yards : convertVolume 7 = 189 := by
  sorry

end convert_seven_cubic_yards_l60_60449


namespace no_perf_square_of_prime_three_digit_l60_60958

theorem no_perf_square_of_prime_three_digit {A B C : ℕ} (h_prime: Prime (100 * A + 10 * B + C)) : ¬ ∃ n : ℕ, B^2 - 4 * A * C = n^2 :=
by
  sorry

end no_perf_square_of_prime_three_digit_l60_60958


namespace angle_y_in_triangle_l60_60132

theorem angle_y_in_triangle (y : ℝ) (h1 : ∀ a b c : ℝ, a + b + c = 180) (h2 : 3 * y + y + 40 = 180) : y = 35 :=
sorry

end angle_y_in_triangle_l60_60132


namespace a1_plus_a9_l60_60610

def S (n : ℕ) : ℕ := n^2 + 1

def a (n : ℕ) : ℕ := S n - S (n - 1)

theorem a1_plus_a9 : (a 1) + (a 9) = 19 := by
  sorry

end a1_plus_a9_l60_60610


namespace necessary_but_not_sufficient_condition_l60_60271

theorem necessary_but_not_sufficient_condition (a : ℝ) :
  (0 < a ∧ a ≤ 1) → (∀ x : ℝ, x^2 - 2*a*x + a > 0) :=
by
  sorry

end necessary_but_not_sufficient_condition_l60_60271


namespace new_savings_after_expense_increase_l60_60562

theorem new_savings_after_expense_increase
    (monthly_salary : ℝ)
    (initial_saving_percent : ℝ)
    (expense_increase_percent : ℝ)
    (initial_salary : monthly_salary = 20000)
    (saving_rate : initial_saving_percent = 0.1)
    (increase_rate : expense_increase_percent = 0.1) :
    monthly_salary - (monthly_salary * (1 - initial_saving_percent + (1 - initial_saving_percent) * expense_increase_percent)) = 200 :=
by
  sorry

end new_savings_after_expense_increase_l60_60562


namespace count_positive_even_multiples_of_3_less_than_5000_perfect_squares_l60_60035

theorem count_positive_even_multiples_of_3_less_than_5000_perfect_squares :
  ∃ n : ℕ, (n = 11) ∧ ∀ k : ℕ, (k < 5000) → (k % 2 = 0) → (k % 3 = 0) → (∃ m : ℕ, k = m * m) → k ≤ 36 * 11 * 11 :=
by {
  sorry
}

end count_positive_even_multiples_of_3_less_than_5000_perfect_squares_l60_60035


namespace sin_geq_tan_minus_half_tan_cubed_l60_60755

theorem sin_geq_tan_minus_half_tan_cubed (x : ℝ) (hx : 0 ≤ x ∧ x < π / 2) :
  Real.sin x ≥ Real.tan x - 1/2 * (Real.tan x) ^ 3 := 
sorry

end sin_geq_tan_minus_half_tan_cubed_l60_60755


namespace find_stream_speed_l60_60989

-- Define the problem based on the provided conditions
theorem find_stream_speed (b s : ℝ) (h1 : b + s = 250 / 7) (h2 : b - s = 150 / 21) : s = 14.28 :=
by
  sorry

end find_stream_speed_l60_60989


namespace initial_loss_percentage_l60_60673

theorem initial_loss_percentage 
  (CP : ℝ := 250) 
  (SP : ℝ) 
  (h1 : SP + 50 = 1.10 * CP) : 
  (CP - SP) / CP * 100 = 10 := 
sorry

end initial_loss_percentage_l60_60673


namespace num_people_visited_iceland_l60_60241

noncomputable def total := 100
noncomputable def N := 43  -- Number of people who visited Norway
noncomputable def B := 61  -- Number of people who visited both Iceland and Norway
noncomputable def Neither := 63  -- Number of people who visited neither country
noncomputable def I : ℕ := 55  -- Number of people who visited Iceland (need to prove)

-- Lean statement to prove
theorem num_people_visited_iceland : I = total - Neither + B - N := by
  sorry

end num_people_visited_iceland_l60_60241


namespace cube_volume_from_surface_area_l60_60111

theorem cube_volume_from_surface_area (S : ℝ) (h : S = 150) : (S / 6) ^ (3 / 2) = 125 := by
  sorry

end cube_volume_from_surface_area_l60_60111


namespace min_value_of_function_l60_60929

theorem min_value_of_function : 
  ∃ x > 2, ∀ y > 2, (y + 1 / (y - 2)) ≥ 4 ∧ (x + 1 / (x - 2)) = 4 := 
by sorry

end min_value_of_function_l60_60929


namespace regular_polygon_sides_l60_60293

theorem regular_polygon_sides (P s : ℕ) (hP : P = 180) (hs : s = 15) : P / s = 12 := by
  -- Given
  -- P = 180  -- the perimeter in cm
  -- s = 15   -- the side length in cm
  sorry

end regular_polygon_sides_l60_60293


namespace range_of_a_l60_60957

theorem range_of_a (a : ℝ) :
  (∀ x : ℕ, 0 < x ∧ 3*x + a ≤ 2 → x = 1 ∨ x = 2) ↔ (-7 < a ∧ a ≤ -4) :=
sorry

end range_of_a_l60_60957


namespace range_of_a_l60_60220

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, 0 < x → x + (4 / x) - 1 - a^2 + 2 * a > 0) : -1 < a ∧ a < 3 :=
sorry

end range_of_a_l60_60220


namespace algebraic_expr_value_at_neg_one_l60_60453

-- Define the expression "3 times the square of x minus 5"
def algebraic_expr (x : ℝ) : ℝ := 3 * x^2 + 5

-- Theorem to state the value when x = -1 is 8
theorem algebraic_expr_value_at_neg_one : algebraic_expr (-1) = 8 := 
by
  -- The steps to prove are skipped with 'sorry'
  sorry

end algebraic_expr_value_at_neg_one_l60_60453


namespace amber_josh_departure_time_l60_60764

def latest_departure_time (flight_time : ℕ) (check_in_time : ℕ) (drive_time : ℕ) (parking_time : ℕ) :=
  flight_time - check_in_time - drive_time - parking_time

theorem amber_josh_departure_time :
  latest_departure_time 20 2 (45 / 60) (15 / 60) = 17 :=
by
  -- Placeholder for actual proof
  sorry

end amber_josh_departure_time_l60_60764


namespace mul_inv_800_mod_7801_l60_60331

theorem mul_inv_800_mod_7801 :
  ∃ x : ℕ, 0 ≤ x ∧ x < 7801 ∧ (800 * x) % 7801 = 1 := by
  use 3125
  dsimp
  norm_num1
  sorry

end mul_inv_800_mod_7801_l60_60331


namespace omega_not_possible_l60_60662

noncomputable def f (ω x φ : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem omega_not_possible (ω φ : ℝ) (h1 : ∀ x y, -π/3 ≤ x → x < y → y ≤ π/6 → f ω x φ ≤ f ω y φ)
  (h2 : f ω (π / 6) φ = f ω (4 * π / 3) φ)
  (h3 : f ω (π / 6) φ = -f ω (-π / 3) φ) :
  ω ≠ 7 / 5 :=
sorry

end omega_not_possible_l60_60662


namespace kevin_feeds_each_toad_3_worms_l60_60923

theorem kevin_feeds_each_toad_3_worms
  (num_toads : ℕ) (minutes_per_worm : ℕ) (hours_to_minutes : ℕ) (total_minutes : ℕ)
  (H1 : num_toads = 8)
  (H2 : minutes_per_worm = 15)
  (H3 : hours_to_minutes = 60)
  (H4 : total_minutes = 6 * hours_to_minutes)
  :
  total_minutes / minutes_per_worm / num_toads = 3 :=
sorry

end kevin_feeds_each_toad_3_worms_l60_60923


namespace rate_per_kg_of_grapes_l60_60196

-- Define the conditions 
namespace Problem

-- Given conditions
variables (G : ℝ) (rate_mangoes : ℝ := 55) (cost_paid : ℝ := 1055)
variables (kg_grapes : ℝ := 8) (kg_mangoes : ℝ := 9)

-- Statement to prove
theorem rate_per_kg_of_grapes : 8 * G + 9 * rate_mangoes = cost_paid → G = 70 := 
by
  intro h
  sorry -- proof goes here

end Problem

end rate_per_kg_of_grapes_l60_60196


namespace notebooks_distributed_l60_60032

theorem notebooks_distributed  (C : ℕ) (N : ℕ) 
  (h1 : N = C^2 / 8) 
  (h2 : N = 8 * C) : 
  N = 512 :=
by 
  sorry

end notebooks_distributed_l60_60032


namespace not_necessarily_true_l60_60878

theorem not_necessarily_true (x y : ℝ) (h : x > y) : ¬ (x^2 > y^2) :=
sorry

end not_necessarily_true_l60_60878


namespace rectangular_prism_length_l60_60216

theorem rectangular_prism_length (w l h : ℝ) 
  (h1 : l = 4 * w) 
  (h2 : h = 3 * w) 
  (h3 : 4 * l + 4 * w + 4 * h = 256) : 
  l = 32 :=
by
  sorry

end rectangular_prism_length_l60_60216


namespace arithmetic_sequence_sum_eight_l60_60194

variable {α : Type*} [LinearOrderedField α]

def arithmetic_sum (a₁ a₈ : α) (n : α) : α := (n * (a₁ + a₈)) / 2

theorem arithmetic_sequence_sum_eight {a₄ a₅ : α} (h₄₅ : a₄ + a₅ = 10) :
  let a₁ := a₄ - 3 * ((a₅ - a₄) / 1) -- a₁ in terms of a₄ and a₅
  let a₈ := a₄ + 4 * ((a₅ - a₄) / 1) -- a₈ in terms of a₄ and a₅
  arithmetic_sum a₁ a₈ 8 = 40 :=
by
  sorry

end arithmetic_sequence_sum_eight_l60_60194


namespace number_of_x_intercepts_l60_60135

def parabola (y : ℝ) : ℝ := -3 * y^2 + 2 * y + 3

theorem number_of_x_intercepts : ∃! (x : ℝ), ∃ (y : ℝ), parabola y = x ∧ y = 0 :=
by
  sorry

end number_of_x_intercepts_l60_60135


namespace solve_eq1_solve_eq2_solve_eq3_solve_eq4_l60_60613

theorem solve_eq1 : ∀ (x : ℝ), x^2 - 5 * x = 0 ↔ x = 0 ∨ x = 5 :=
by sorry

theorem solve_eq2 : ∀ (x : ℝ), (2 * x + 1)^2 = 4 ↔ x = -3 / 2 ∨ x = 1 / 2 :=
by sorry

theorem solve_eq3 : ∀ (x : ℝ), x * (x - 1) + 3 * (x - 1) = 0 ↔ x = 1 ∨ x = -3 :=
by sorry

theorem solve_eq4 : ∀ (x : ℝ), x^2 - 2 * x - 8 = 0 ↔ x = -2 ∨ x = 4 :=
by sorry

end solve_eq1_solve_eq2_solve_eq3_solve_eq4_l60_60613


namespace difference_between_possible_x_values_l60_60933

theorem difference_between_possible_x_values :
  ∀ (x : ℝ), (x + 3) ^ 2 / (2 * x + 15) = 3 → (x = 6 ∨ x = -6) →
  (abs (6 - (-6)) = 12) :=
by
  intro x h1 h2
  sorry

end difference_between_possible_x_values_l60_60933


namespace cats_on_ship_l60_60787

theorem cats_on_ship :
  ∃ (C S : ℕ), 
  (C + S + 1 + 1 = 16) ∧
  (4 * C + 2 * S + 2 * 1 + 1 * 1 = 41) ∧ 
  C = 5 :=
by
  sorry

end cats_on_ship_l60_60787


namespace find_x_l60_60167

noncomputable def geometric_series_sum (x: ℝ) : ℝ := 
  1 + x + 2 * x^2 + 3 * x^3 + 4 * x^4 + ∑' n: ℕ, (n + 1) * x^(n + 1)

theorem find_x (x: ℝ) (hx : geometric_series_sum x = 16) : x = 15 / 16 := 
by
  sorry

end find_x_l60_60167


namespace find_y_l60_60049

theorem find_y (x y : ℤ) (h1 : x = -4) (h2 : x^2 + 3 * x + 7 = y - 5) : y = 16 := 
by
  sorry

end find_y_l60_60049


namespace tomatoes_on_each_plant_l60_60354

/-- Andy harvests all the tomatoes from 18 plants that have a certain number of tomatoes each.
    He dries half the tomatoes and turns a third of the remainder into marinara sauce. He has
    42 tomatoes left. Prove that the number of tomatoes on each plant is 7.  -/
theorem tomatoes_on_each_plant (T : ℕ) (h1 : ∀ n, n = 18 * T)
  (h2 : ∀ m, m = (18 * T) / 2)
  (h3 : ∀ k, k = m / 3)
  (h4 : ∀ final, final = m - k ∧ final = 42) : T = 7 :=
by
  sorry

end tomatoes_on_each_plant_l60_60354


namespace flowers_lost_l60_60099

theorem flowers_lost 
  (time_per_flower : ℕ)
  (gathered_time : ℕ) 
  (additional_time : ℕ) 
  (classmates : ℕ) 
  (collected_flowers : ℕ) 
  (total_needed : ℕ)
  (lost_flowers : ℕ) 
  (H1 : time_per_flower = 10)
  (H2 : gathered_time = 120)
  (H3 : additional_time = 210)
  (H4 : classmates = 30)
  (H5 : collected_flowers = gathered_time / time_per_flower)
  (H6 : total_needed = classmates + (additional_time / time_per_flower))
  (H7 : lost_flowers = total_needed - classmates) :
lost_flowers = 3 := 
sorry

end flowers_lost_l60_60099


namespace number_of_tests_in_series_l60_60145

theorem number_of_tests_in_series (S : ℝ) (n : ℝ) :
  (S + 97) / n = 90 →
  (S + 73) / n = 87 →
  n = 8 :=
by 
  sorry

end number_of_tests_in_series_l60_60145


namespace graphs_intersection_points_l60_60405

theorem graphs_intersection_points {g : ℝ → ℝ} (h_injective : Function.Injective g) :
  ∃ (x1 x2 x3 : ℝ), (g (x1^3) = g (x1^5)) ∧ (g (x2^3) = g (x2^5)) ∧ (g (x3^3) = g (x3^5)) ∧ 
  x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ ∀ (x : ℝ), (g (x^3) = g (x^5)) → (x = x1 ∨ x = x2 ∨ x = x3) := 
by
  sorry

end graphs_intersection_points_l60_60405


namespace complete_work_together_in_days_l60_60743

noncomputable def a_days := 16
noncomputable def b_days := 6
noncomputable def c_days := 12

noncomputable def work_rate (days: ℕ) : ℚ := 1 / days

theorem complete_work_together_in_days :
  let combined_rate := (work_rate a_days) + (work_rate b_days) + (work_rate c_days)
  let days_to_complete := 1 / combined_rate
  days_to_complete = 3.2 :=
  sorry

end complete_work_together_in_days_l60_60743


namespace calculate_total_income_l60_60801

/-- Total income calculation proof for a person with given distributions and remaining amount -/
theorem calculate_total_income
  (I : ℝ) -- total income
  (leftover : ℝ := 40000) -- leftover amount after distribution and donation
  (c1_percentage : ℝ := 3 * 0.15) -- percentage given to children
  (c2_percentage : ℝ := 0.30) -- percentage given to wife
  (c3_percentage : ℝ := 0.05) -- percentage donated to orphan house
  (remaining_percentage : ℝ := 1 - (c1_percentage + c2_percentage)) -- remaining percentage after children and wife
  (R : ℝ := remaining_percentage * I) -- remaining amount after children and wife
  (donation : ℝ := c3_percentage * R) -- amount donated to orphan house)
  (left_amount : ℝ := R - donation) -- final remaining amount
  (income : ℝ := (leftover / (1 - remaining_percentage * (1 - c3_percentage)))) -- calculation of the actual income
  : I = income := sorry

end calculate_total_income_l60_60801


namespace ball_falls_total_distance_l60_60225

noncomputable def total_distance : ℕ → ℤ → ℤ → ℤ
| 0, a, _ => 0
| (n+1), a, d => a + total_distance n (a + d) d

theorem ball_falls_total_distance :
  total_distance 5 30 (-6) = 90 :=
by
  sorry

end ball_falls_total_distance_l60_60225


namespace maximum_x_y_value_l60_60247

theorem maximum_x_y_value (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (h1 : x + 2 * y ≤ 6) (h2 : 2 * x + y ≤ 6) : x + y ≤ 4 := 
sorry

end maximum_x_y_value_l60_60247


namespace find_a_of_square_roots_l60_60978

theorem find_a_of_square_roots (a : ℤ) (n : ℤ) (h₁ : 2 * a + 1 = n) (h₂ : a + 5 = n) : a = 4 :=
by
  -- proof goes here
  sorry

end find_a_of_square_roots_l60_60978


namespace k_range_l60_60616

noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 1 then -x^3 + 2*x^2 - x
  else if 1 ≤ x then Real.log x
  else 0 -- Technically, we don't care outside (0, +∞), so this else case doesn't matter.

theorem k_range (k : ℝ) :
  (∀ t : ℝ, 0 < t → f t < k * t) ↔ k ∈ (Set.Ioi (1 / Real.exp 1)) :=
by
  sorry

end k_range_l60_60616


namespace hyperbola_condition_l60_60080

theorem hyperbola_condition (m : ℝ) : (m > 0) ↔ (2 + m > 0 ∧ 1 + m > 0) :=
by sorry

end hyperbola_condition_l60_60080


namespace sin_pi_over_6_plus_α_cos_pi_over_3_plus_2α_l60_60202

variable (α : ℝ)

-- Given conditions
def α_condition (hα : 0 < α ∧ α < π / 2) (hcos : Real.cos α = 3 / 5) : Prop := 
  true

-- Prove the first part: sin(π / 6 + α) = (3 + 4 * real.sqrt 3) / 10
theorem sin_pi_over_6_plus_α (hα : 0 < α ∧ α < π / 2) (hcos : Real.cos α = 3 / 5) :
  Real.sin (π / 6 + α) = (3 + 4 * Real.sqrt 3) / 10 :=
by
  sorry

-- Prove the second part: cos(π / 3 + 2 * α) = -(7 + 24 * real.sqrt 3) / 50
theorem cos_pi_over_3_plus_2α (hα : 0 < α ∧ α < π / 2) (hcos : Real.cos α = 3 / 5) :
  Real.cos (π / 3 + 2 * α) = -(7 + 24 * Real.sqrt 3) / 50 :=
by
  sorry

end sin_pi_over_6_plus_α_cos_pi_over_3_plus_2α_l60_60202


namespace Sarah_ate_one_apple_l60_60237

theorem Sarah_ate_one_apple:
  ∀ (total_apples apples_given_to_teachers apples_given_to_friends apples_left: ℕ), 
  total_apples = 25 →
  apples_given_to_teachers = 16 →
  apples_given_to_friends = 5 →
  apples_left = 3 →
  total_apples - (apples_given_to_teachers + apples_given_to_friends + apples_left) = 1 :=
by
  intros total_apples apples_given_to_teachers apples_given_to_friends apples_left
  intro ht ht gt hf
  sorry

end Sarah_ate_one_apple_l60_60237


namespace thirty_two_not_sum_consecutive_natural_l60_60839

theorem thirty_two_not_sum_consecutive_natural (n k : ℕ) : 
  (n > 0) → (32 ≠ (n * (2 * k + n - 1)) / 2) :=
by
  sorry

end thirty_two_not_sum_consecutive_natural_l60_60839


namespace ratio_paid_back_to_initial_debt_l60_60041

def initial_debt : ℕ := 40
def still_owed : ℕ := 30
def paid_back (initial_debt still_owed : ℕ) : ℕ := initial_debt - still_owed

theorem ratio_paid_back_to_initial_debt
  (initial_debt still_owed : ℕ) :
  (paid_back initial_debt still_owed : ℚ) / initial_debt = 1 / 4 :=
by 
  sorry

end ratio_paid_back_to_initial_debt_l60_60041


namespace range_of_d_l60_60954

theorem range_of_d (a_1 d : ℝ) (h : (a_1 + 2 * d) * (a_1 + 3 * d) + 1 = 0) :
  d ∈ Set.Iic (-2) ∪ Set.Ici 2 :=
sorry

end range_of_d_l60_60954


namespace correct_equation_l60_60669

-- Define the daily paving distances for Team A and Team B
variables (x : ℝ) (h₀ : x > 10)

-- Assuming Team A takes the same number of days to pave 150m as Team B takes to pave 120m
def same_days_to_pave (h₁ : x - 10 > 0) : Prop :=
  (150 / x = 120 / (x - 10))

-- The theorem to be proven
theorem correct_equation (h₁ : x - 10 > 0) : 150 / x = 120 / (x - 10) :=
by
  sorry

end correct_equation_l60_60669


namespace frac_x_y_eq_neg2_l60_60602

open Real

theorem frac_x_y_eq_neg2 (x y : ℝ) (h1 : 1 < (x - y) / (x + y)) (h2 : (x - y) / (x + y) < 4) (h3 : (x + y) / (x - y) ≠ 1) :
  ∃ t : ℤ, (x / y = t) ∧ (t = -2) :=
by sorry

end frac_x_y_eq_neg2_l60_60602


namespace aluminum_percentage_in_new_alloy_l60_60192

theorem aluminum_percentage_in_new_alloy :
  ∀ (x1 x2 x3 : ℝ),
  0 ≤ x1 ∧ x1 ≤ 1 ∧
  0 ≤ x2 ∧ x2 ≤ 1 ∧
  0 ≤ x3 ∧ x3 ≤ 1 ∧
  x1 + x2 + x3 = 1 ∧
  0.15 * x1 + 0.3 * x2 = 0.2 →
  0.15 ≤ 0.6 * x1 + 0.45 * x3 ∧ 0.6 * x1 + 0.45 * x3 ≤ 0.40 :=
by
  -- The proof will be inserted here
  sorry

end aluminum_percentage_in_new_alloy_l60_60192


namespace prevent_white_cube_n2_prevent_white_cube_n3_l60_60799

def min_faces_to_paint (n : ℕ) : ℕ :=
  if n = 2 then 2 else if n = 3 then 12 else sorry

theorem prevent_white_cube_n2 : min_faces_to_paint 2 = 2 := by
  sorry

theorem prevent_white_cube_n3 : min_faces_to_paint 3 = 12 := by
  sorry

end prevent_white_cube_n2_prevent_white_cube_n3_l60_60799


namespace total_pages_in_book_l60_60106

variable (p1 p2 p_total : ℕ)
variable (read_first_four_days : p1 = 4 * 45)
variable (read_next_three_days : p2 = 3 * 52)
variable (total_until_last_day : p_total = p1 + p2 + 15)

theorem total_pages_in_book : p_total = 351 :=
by
  -- Introduce the conditions
  rw [read_first_four_days, read_next_three_days] at total_until_last_day
  sorry

end total_pages_in_book_l60_60106


namespace y_intercept_of_line_l60_60885

theorem y_intercept_of_line (x y : ℝ) (h : 4 * x + 7 * y = 28) (hx : x = 0) : y = 4 :=
by
  -- The proof goes here
  sorry

end y_intercept_of_line_l60_60885


namespace intersection_in_quadrant_II_l60_60996

theorem intersection_in_quadrant_II (x y : ℝ) 
  (h1: y ≥ -2 * x + 3) 
  (h2: y ≤ 3 * x + 6) 
  (h_intersection: x = -3 / 5 ∧ y = 21 / 5) :
  x < 0 ∧ y > 0 := 
sorry

end intersection_in_quadrant_II_l60_60996


namespace max_complexity_51_l60_60508

-- Define the complexity of a number 
def complexity (x : ℚ) : ℕ := sorry -- Placeholder for the actual complexity function definition

-- Define the sequence for m values
def m_sequence (k : ℕ) : List ℕ :=
  List.range' 1 (2^(k-1)) |>.filter (λ n => n % 2 = 1)

-- Define the candidate number
def candidate_number (k : ℕ) : ℚ :=
  (2^(k + 1) + (-1)^k) / (3 * 2^k)

theorem max_complexity_51 : 
  ∃ m, m ∈ m_sequence 50 ∧ 
  (∀ n, n ∈ m_sequence 50 → complexity (n / 2^50) ≤ complexity (candidate_number 50 / 2^50)) :=
sorry

end max_complexity_51_l60_60508


namespace sum_of_interior_angles_of_pentagon_l60_60904

theorem sum_of_interior_angles_of_pentagon : (5 - 2) * 180 = 540 := 
by
  sorry

end sum_of_interior_angles_of_pentagon_l60_60904


namespace joan_first_payment_l60_60333

theorem joan_first_payment (P : ℝ) 
  (total_amount : ℝ) 
  (r : ℝ) 
  (n : ℕ) 
  (h_total : total_amount = 109300)
  (h_r : r = 3)
  (h_n : n = 7)
  (h_sum : total_amount = P * (1 - r^n) / (1 - r)) : 
  P = 100 :=
by
  -- proof goes here
  sorry

end joan_first_payment_l60_60333


namespace simplify_and_evaluate_expression_l60_60443

theorem simplify_and_evaluate_expression (x : ℝ) (h : x = -3) : (1 + 1/(x+1)) / ((x^2 + 4*x + 4) / (x+1)) = -1 :=
by
  sorry

end simplify_and_evaluate_expression_l60_60443


namespace Riverdale_High_students_l60_60415

theorem Riverdale_High_students
  (f j : ℕ)
  (h1 : (3 / 7) * f + (3 / 4) * j = 234)
  (h2 : f + j = 420) :
  f = 64 ∧ j = 356 := by
  sorry

end Riverdale_High_students_l60_60415


namespace quadratic_inequality_solution_set_l60_60226

theorem quadratic_inequality_solution_set :
  {x : ℝ | - x ^ 2 + 4 * x + 12 > 0} = {x : ℝ | -2 < x ∧ x < 6} :=
sorry

end quadratic_inequality_solution_set_l60_60226


namespace number_of_boys_l60_60097

theorem number_of_boys (M W B : ℕ) (X : ℕ) 
  (h1 : 5 * M = W) 
  (h2 : W = B) 
  (h3 : 5 * M * 12 + W * X + B * X = 180) 
  : B = 15 := 
by sorry

end number_of_boys_l60_60097


namespace order_of_a_b_c_l60_60818

noncomputable def a : ℝ := (Real.log 5) / 5
noncomputable def b : ℝ := 1 / Real.exp 1
noncomputable def c : ℝ := (Real.log 4) / 4

theorem order_of_a_b_c : a < c ∧ c < b := by
  sorry

end order_of_a_b_c_l60_60818


namespace closest_multiple_of_15_to_2023_is_2025_l60_60752

theorem closest_multiple_of_15_to_2023_is_2025 (n : ℤ) (h : 15 * n = 2025) : 
  ∀ m : ℤ, abs (2023 - 2025) ≤ abs (2023 - 15 * m) :=
by
  exact sorry

end closest_multiple_of_15_to_2023_is_2025_l60_60752


namespace units_digit_product_even_composite_l60_60218

/-- The units digit of the product of the first three even composite numbers greater than 10 is 8. -/
theorem units_digit_product_even_composite :
  let a := 12
  let b := 14
  let c := 16
  (a * b * c) % 10 = 8 :=
by
  let a := 12
  let b := 14
  let c := 16
  have h : (a * b * c) % 10 = 8
  { sorry }
  exact h

end units_digit_product_even_composite_l60_60218


namespace total_volume_correct_l60_60299

-- Defining the initial conditions
def carl_cubes : ℕ := 4
def carl_side_length : ℕ := 3
def kate_cubes : ℕ := 6
def kate_side_length : ℕ := 1

-- Given the above conditions, define the total volume of all cubes.
def total_volume_of_all_cubes : ℕ := (carl_cubes * carl_side_length ^ 3) + (kate_cubes * kate_side_length ^ 3)

-- The statement we need to prove
theorem total_volume_correct :
  total_volume_of_all_cubes = 114 :=
by
  -- Skipping the proof with sorry as per the instruction
  sorry

end total_volume_correct_l60_60299


namespace ratio_of_d_to_s_l60_60675

theorem ratio_of_d_to_s (s d : ℝ) (n : ℕ) (h1 : n = 15) (h2 : (n^2 * s^2) / ((n * s + 2 * n * d)^2) = 0.75) :
  d / s = 1 / 13 :=
by
  sorry

end ratio_of_d_to_s_l60_60675


namespace total_cost_proof_l60_60378

-- Define the conditions
def length_grass_field : ℝ := 75
def width_grass_field : ℝ := 55
def width_path : ℝ := 2.5
def area_path : ℝ := 6750
def cost_per_sq_m : ℝ := 10

-- Calculate the outer dimensions
def outer_length : ℝ := length_grass_field + 2 * width_path
def outer_width : ℝ := width_grass_field + 2 * width_path

-- Calculate the area of the entire field including the path
def area_entire_field : ℝ := outer_length * outer_width

-- Calculate the area of the grass field without the path
def area_grass_field : ℝ := length_grass_field * width_grass_field

-- Calculate the area of the path
def area_calculated_path : ℝ := area_entire_field - area_grass_field

-- Calculate the total cost of constructing the path
noncomputable def total_cost : ℝ := area_calculated_path * cost_per_sq_m

-- The theorem to prove
theorem total_cost_proof :
  area_calculated_path = area_path ∧ total_cost = 6750 :=
by
  sorry

end total_cost_proof_l60_60378


namespace find_c_and_d_l60_60704

theorem find_c_and_d (c d : ℝ) (h : ℝ → ℝ) (f : ℝ → ℝ) (finv : ℝ → ℝ) 
  (h_def : ∀ x, h x = 6 * x - 5)
  (finv_eq : ∀ x, finv x = 6 * x - 3)
  (f_def : ∀ x, f x = c * x + d)
  (inv_prop : ∀ x, f (finv x) = x ∧ finv (f x) = x) :
  4 * c + 6 * d = 11 / 3 :=
by
  sorry

end find_c_and_d_l60_60704


namespace smallest_positive_solution_eq_sqrt_29_l60_60919

theorem smallest_positive_solution_eq_sqrt_29 :
  ∃ x : ℝ, 0 < x ∧ x^4 - 58 * x^2 + 841 = 0 ∧ x = Real.sqrt 29 :=
by
  sorry

end smallest_positive_solution_eq_sqrt_29_l60_60919


namespace speed_of_stream_l60_60856

theorem speed_of_stream (b s : ℝ) 
  (H1 : b + s = 10)
  (H2 : b - s = 4) : 
  s = 3 :=
sorry

end speed_of_stream_l60_60856


namespace paint_rate_l60_60785

theorem paint_rate (l b : ℝ) (cost : ℕ) (rate_per_sq_m : ℝ) 
  (h1 : l = 3 * b) 
  (h2 : cost = 300) 
  (h3 : l = 13.416407864998739) 
  (area : ℝ := l * b) : 
  rate_per_sq_m = 5 :=
by
  sorry

end paint_rate_l60_60785


namespace least_number_of_marbles_l60_60130

theorem least_number_of_marbles :
  ∃ n, (∀ d ∈ ({3, 4, 5, 7, 8} : Set ℕ), d ∣ n) ∧ n = 840 :=
by
  sorry

end least_number_of_marbles_l60_60130


namespace correct_statement_l60_60301

-- Definitions
def certain_event (P : ℝ → Prop) : Prop := P 1
def impossible_event (P : ℝ → Prop) : Prop := P 0
def uncertain_event (P : ℝ → Prop) : Prop := ∀ p, 0 < p ∧ p < 1 → P p

-- Theorem to prove
theorem correct_statement (P : ℝ → Prop) :
  (certain_event P ∧ impossible_event P ∧ uncertain_event P) →
  (∀ p, P p → p = 1)
:= by
  sorry

end correct_statement_l60_60301


namespace tyre_flattening_time_l60_60071

theorem tyre_flattening_time (R1 R2 : ℝ) (hR1 : R1 = 1 / 9) (hR2 : R2 = 1 / 6) : 
  1 / (R1 + R2) = 3.6 :=
by 
  sorry

end tyre_flattening_time_l60_60071


namespace percentage_profits_revenues_previous_year_l60_60783

noncomputable def companyProfits (R P R2009 P2009 : ℝ) : Prop :=
  (R2009 = 0.8 * R) ∧ (P2009 = 0.15 * R2009) ∧ (P2009 = 1.5 * P)

theorem percentage_profits_revenues_previous_year (R P : ℝ) (h : companyProfits R P (0.8 * R) (0.12 * R)) : 
  (P / R * 100) = 8 :=
by 
  sorry

end percentage_profits_revenues_previous_year_l60_60783


namespace necessary_and_sufficient_condition_l60_60820

-- Variables and conditions
variables (a : ℕ) (A B : ℝ)
variable (positive_a : 0 < a)

-- System of equations
def system_has_positive_integer_solutions (x y z : ℕ) : Prop :=
  (x^2 + y^2 + z^2 = (13 * a)^2) ∧ 
  (x^2 * (A * x^2 + B * y^2) + y^2 * (A * y^2 + B * z^2) + z^2 * (A * z^2 + B * x^2) = 
    (1 / 4) * (2 * A + B) * (13 * a)^4)

-- Statement of the theorem
theorem necessary_and_sufficient_condition:
  (∃ (x y z : ℕ), system_has_positive_integer_solutions a A B x y z) ↔ B = 2 * A :=
sorry

end necessary_and_sufficient_condition_l60_60820


namespace sum_second_largest_and_smallest_l60_60388

theorem sum_second_largest_and_smallest :
  let numbers := [10, 11, 12, 13, 14]
  ∃ second_largest second_smallest, (List.nthLe numbers 3 sorry = second_largest ∧ List.nthLe numbers 1 sorry = second_smallest ∧ second_largest + second_smallest = 24) :=
sorry

end sum_second_largest_and_smallest_l60_60388


namespace winning_percentage_l60_60090

theorem winning_percentage (total_games first_games remaining_games : ℕ) 
                           (first_win_percent remaining_win_percent : ℝ)
                           (total_games_eq : total_games = 60)
                           (first_games_eq : first_games = 30)
                           (remaining_games_eq : remaining_games = 30)
                           (first_win_percent_eq : first_win_percent = 0.40)
                           (remaining_win_percent_eq : remaining_win_percent = 0.80) :
                           (first_win_percent * (first_games : ℝ) +
                            remaining_win_percent * (remaining_games : ℝ)) /
                           (total_games : ℝ) * 100 = 60 := sorry

end winning_percentage_l60_60090


namespace compare_neg_fractions_l60_60136

theorem compare_neg_fractions : - (2 / 3 : ℝ) > - (5 / 7 : ℝ) := 
by 
  sorry

end compare_neg_fractions_l60_60136


namespace product_of_dice_divisible_by_9_l60_60592

-- Define the probability of rolling a number divisible by 3
def prob_roll_div_by_3 : ℚ := 1/6

-- Define the probability of rolling a number not divisible by 3
def prob_roll_not_div_by_3 : ℚ := 2/3

-- Define the probability that the product of numbers rolled on 6 dice is divisible by 9
def prob_product_div_by_9 : ℚ := 449/729

-- Main statement of the problem
theorem product_of_dice_divisible_by_9 :
  (1 - ((prob_roll_not_div_by_3^6) + 
        (6 * prob_roll_div_by_3 * (prob_roll_not_div_by_3^5)) + 
        (15 * (prob_roll_div_by_3^2) * (prob_roll_not_div_by_3^4)))) = prob_product_div_by_9 :=
by {
  sorry
}

end product_of_dice_divisible_by_9_l60_60592


namespace problem_sign_of_trig_product_l60_60137

open Real

theorem problem_sign_of_trig_product (θ : ℝ) (hθ : π / 2 < θ ∧ θ < π) :
  sin (cos θ) * cos (sin (2 * θ)) < 0 :=
sorry

end problem_sign_of_trig_product_l60_60137


namespace find_f_24_25_26_l60_60276

-- Given conditions
def homogeneous (f : ℤ → ℤ → ℤ → ℝ) : Prop :=
  ∀ (n a b c : ℤ), f (n * a) (n * b) (n * c) = n * f a b c

def shift_invariance (f : ℤ → ℤ → ℤ → ℝ) : Prop :=
  ∀ (a b c n : ℤ), f (a + n) (b + n) (c + n) = f a b c + n

def symmetry (f : ℤ → ℤ → ℤ → ℝ) : Prop :=
  ∀ (a b c : ℤ), f a b c = f c b a

-- Proving the required value under the conditions
theorem find_f_24_25_26 (f : ℤ → ℤ → ℤ → ℝ)
  (homo : homogeneous f) 
  (shift : shift_invariance f) 
  (symm : symmetry f) : 
  f 24 25 26 = 25 := 
sorry

end find_f_24_25_26_l60_60276


namespace total_apples_l60_60119

-- Definitions and Conditions
variable (a : ℕ) -- original number of apples in the first pile (scaled integer type)
variable (n m : ℕ) -- arbitrary positions in the sequence

-- Arithmetic sequence of initial piles
def initial_piles := [a, 2*a, 3*a, 4*a, 5*a, 6*a]

-- Given condition transformations
def after_removal_distribution (initial_piles : List ℕ) (k : ℕ) : List ℕ :=
  match k with
  | 0 => [0, 2*a + 10, 3*a + 20, 4*a + 30, 5*a + 40, 6*a + 50]
  | 1 => [a + 10, 0, 3*a + 20, 4*a + 30, 5*a + 40, 6*a + 50]
  | 2 => [a + 10, 2*a + 20, 0, 4*a + 30, 5*a + 40, 6*a + 50]
  | 3 => [a + 10, 2*a + 20, 3*a + 30, 0, 5*a + 40, 6*a + 50]
  | 4 => [a + 10, 2*a + 20, 3*a + 30, 4*a + 40, 0, 6*a + 50]
  | _ => [a + 10, 2*a + 20, 3*a + 30, 4*a + 40, 5*a + 50, 0]

-- Prove the total number of apples
theorem total_apples : (a = 35) → (a + 2 * a + 3 * a + 4 * a + 5 * a + 6 * a = 735) :=
by
  intros h1
  sorry

end total_apples_l60_60119


namespace prove_a_range_l60_60548

-- Defining the propositions p and q
def p (a : ℝ) : Prop := ∃ x ∈ Set.Icc (-1 : ℝ) 1, a^2 * x^2 + a * x - 2 = 0
def q (a : ℝ) : Prop := ∃! x : ℝ, x^2 + 2 * a * x + 2 * a ≤ 0

-- The proposition to prove
theorem prove_a_range (a : ℝ) (hpq : ¬(p a ∨ q a)) : a ∈ Set.Ioo (-1 : ℝ) 0 ∪ Set.Ioo 0 1 :=
by
  sorry

end prove_a_range_l60_60548


namespace range_of_m_l60_60788

theorem range_of_m (m : ℝ) (h : 1 < (8 - m) / (m - 5)) : 5 < m ∧ m < 13 / 2 :=
sorry

end range_of_m_l60_60788


namespace find_constant_term_l60_60941

theorem find_constant_term (c : ℤ) (y : ℤ) (h1 : y = 2) (h2 : 5 * y^2 - 8 * y + c = 59) : c = 55 :=
by
  sorry

end find_constant_term_l60_60941


namespace second_largest_between_28_and_31_l60_60760

theorem second_largest_between_28_and_31 : 
  ∃ (n : ℕ), n > 28 ∧ n ≤ 31 ∧ (∀ m, (m > 28 ∧ m ≤ 31 ∧ m < 31) ->  m ≤ 30) :=
sorry

end second_largest_between_28_and_31_l60_60760


namespace sum_of_four_digit_multiples_of_5_l60_60451

theorem sum_of_four_digit_multiples_of_5 :
  let a := 1000
  let l := 9995
  let d := 5
  let n := ((l - a) / d) + 1
  let S := n * (a + l) / 2
  S = 9895500 :=
by
  let a := 1000
  let l := 9995
  let d := 5
  let n := ((l - a) / d) + 1
  let S := n * (a + l) / 2
  sorry

end sum_of_four_digit_multiples_of_5_l60_60451


namespace calculate_star_value_l60_60995

def custom_operation (a b : ℕ) : ℕ :=
  (a + b)^3

theorem calculate_star_value : custom_operation 3 5 = 512 :=
by
  sorry

end calculate_star_value_l60_60995


namespace percent_increase_surface_area_l60_60725

theorem percent_increase_surface_area (a b c : ℝ) :
  let S := 2 * (a * b + b * c + a * c)
  let S' := 2 * (1.8 * a * 1.8 * b + 1.8 * b * 1.8 * c + 1.8 * c * 1.8 * a)
  (S' - S) / S * 100 = 224 := by
  sorry

end percent_increase_surface_area_l60_60725


namespace distinct_non_zero_reals_square_rational_l60_60328

theorem distinct_non_zero_reals_square_rational
  {a : Fin 10 → ℝ}
  (distinct : ∀ i j, i ≠ j → a i ≠ a j)
  (non_zero : ∀ i, a i ≠ 0)
  (rational_condition : ∀ i j, ∃ (q : ℚ), a i + a j = q ∨ a i * a j = q) :
  ∀ i, ∃ (q : ℚ), (a i)^2 = q :=
by
  sorry

end distinct_non_zero_reals_square_rational_l60_60328


namespace find_n_in_geometric_series_l60_60239

theorem find_n_in_geometric_series :
  let a1 : ℕ := 15
  let a2 : ℕ := 5
  let r1 := a2 / a1
  let S1 := a1 / (1 - r1: ℝ)
  let S2 := 3 * S1
  let r2 := (5 + n) / a1
  S2 = 15 / (1 - r2) →
  n = 20 / 3 :=
by
  sorry

end find_n_in_geometric_series_l60_60239


namespace line_fixed_point_l60_60735

theorem line_fixed_point (m : ℝ) : ∃ x y, (∀ m, y = m * x + (2 * m + 1)) ↔ (x = -2 ∧ y = 1) :=
by
  sorry

end line_fixed_point_l60_60735


namespace Scarlett_adds_correct_amount_l60_60814

-- Define the problem with given conditions
def currentOilAmount : ℝ := 0.17
def desiredOilAmount : ℝ := 0.84

-- Prove that the amount of oil Scarlett needs to add is 0.67 cup
theorem Scarlett_adds_correct_amount : (desiredOilAmount - currentOilAmount) = 0.67 := by
  sorry

end Scarlett_adds_correct_amount_l60_60814


namespace cannot_determine_b_l60_60416

theorem cannot_determine_b 
  (a b c d : ℝ) 
  (h_avg : (a + b + c + d) / 4 = 12.345) 
  (h_ineq : a > b ∧ b > c ∧ c > d) : 
  ¬((b = 12.345) ∨ (b > 12.345) ∨ (b < 12.345)) :=
sorry

end cannot_determine_b_l60_60416


namespace cyclists_cannot_reach_point_B_l60_60630

def v1 := 35 -- Speed of the first cyclist in km/h
def v2 := 25 -- Speed of the second cyclist in km/h
def t := 2   -- Total time in hours
def d  := 30 -- Distance from A to B in km

-- Each cyclist does not rest simultaneously
-- Time equations based on their speed proportions

theorem cyclists_cannot_reach_point_B 
  (v1 := 35) (v2 := 25) (t := 2) (d := 30) 
  (h1 : t * (v1 * (5 / (5 + 7)) / 60) + t * (v2 * (7 / (5 + 7)) / 60) < d) : 
  False := 
sorry

end cyclists_cannot_reach_point_B_l60_60630


namespace remainder_1425_1427_1429_mod_12_l60_60729

theorem remainder_1425_1427_1429_mod_12 :
  (1425 * 1427 * 1429) % 12 = 11 :=
by
  sorry

end remainder_1425_1427_1429_mod_12_l60_60729


namespace max_shapes_in_8x14_grid_l60_60607

def unit_squares := 3
def grid_8x14 := 8 * 14
def grid_points (m n : ℕ) := (m + 1) * (n + 1)
def shapes_grid_points := 8
def max_shapes (total_points shape_points : ℕ) := total_points / shape_points

theorem max_shapes_in_8x14_grid 
  (m n : ℕ) (shape_points : ℕ) 
  (h1 : m = 8) (h2 : n = 14)
  (h3 : shape_points = 8) :
  max_shapes (grid_points m n) shape_points = 16 := by
  sorry

end max_shapes_in_8x14_grid_l60_60607


namespace smallest_n_power_mod_5_l60_60843

theorem smallest_n_power_mod_5 :
  ∃ N : ℕ, 100 ≤ N ∧ N ≤ 999 ∧ (2^N + 1) % 5 = 0 ∧ ∀ M : ℕ, 100 ≤ M ∧ M ≤ 999 ∧ (2^M + 1) % 5 = 0 → N ≤ M := 
sorry

end smallest_n_power_mod_5_l60_60843


namespace largest_sequence_sum_45_l60_60655

theorem largest_sequence_sum_45 
  (S: ℕ → ℕ)
  (h_S: ∀ n, S n = n * (n + 1) / 2)
  (h_sum: ∃ m: ℕ, S m = 45):
  (∃ k: ℕ, k ≤ 9 ∧ S k = 45) ∧ (∀ m: ℕ, S m ≤ 45 → m ≤ 9) :=
by
  sorry

end largest_sequence_sum_45_l60_60655


namespace solve_for_x_l60_60981

theorem solve_for_x (x : ℕ) : 8 * 4^x = 2048 → x = 4 := by
  sorry

end solve_for_x_l60_60981


namespace find_coef_of_quadratic_l60_60869

-- Define the problem conditions
def solutions_of_abs_eq : Set ℤ := {x | abs (x - 3) = 4}

-- Given that the solutions are 7 and -1
def paul_solutions : Set ℤ := {7, -1}

-- The problem translates to proving the equivalence of two sets
def equivalent_equation_solutions (d e : ℤ) : Prop :=
  ∀ x, x ∈ solutions_of_abs_eq ↔ x^2 + d * x + e = 0

theorem find_coef_of_quadratic :
  equivalent_equation_solutions (-6) (-7) :=
by
  sorry

end find_coef_of_quadratic_l60_60869


namespace number_of_ways_to_choose_museums_l60_60258

-- Define the conditions
def number_of_grades : Nat := 6
def number_of_museums : Nat := 6
def number_of_grades_Museum_A : Nat := 2

-- Prove the number of ways to choose museums such that exactly two grades visit Museum A
theorem number_of_ways_to_choose_museums :
  (Nat.choose number_of_grades number_of_grades_Museum_A) * (5 ^ (number_of_grades - number_of_grades_Museum_A)) = Nat.choose 6 2 * 5 ^ 4 :=
by
  sorry

end number_of_ways_to_choose_museums_l60_60258


namespace prime_square_implies_equal_l60_60606

theorem prime_square_implies_equal (p : ℕ) (hp : Nat.Prime p) (hp_gt_2 : p > 2)
  (x y : ℕ) (hx : 1 ≤ x ∧ x ≤ (p-1)/2) (hy : 1 ≤ y ∧ y ≤ (p-1)/2)
  (h_square: ∃ k : ℕ, x * (p - x) * y * (p - y) = k ^ 2) : x = y :=
sorry

end prime_square_implies_equal_l60_60606


namespace winning_candidate_percentage_l60_60422

theorem winning_candidate_percentage
  (total_votes : ℕ)
  (vote_majority : ℕ)
  (winning_candidate_votes : ℕ)
  (losing_candidate_votes : ℕ) :
  total_votes = 400 →
  vote_majority = 160 →
  winning_candidate_votes = total_votes * 70 / 100 →
  losing_candidate_votes = total_votes - winning_candidate_votes →
  winning_candidate_votes - losing_candidate_votes = vote_majority →
  winning_candidate_votes = 280 :=
by 
  intros h1 h2 h3 h4 h5
  sorry

end winning_candidate_percentage_l60_60422


namespace find_x_l60_60670

theorem find_x (x : ℚ) (h : |x - 1| = |x - 2|) : x = 3 / 2 :=
sorry

end find_x_l60_60670


namespace units_produced_by_line_B_l60_60654

-- State the problem with the given conditions and prove the question equals the answer.
theorem units_produced_by_line_B (total_units : ℕ) (B : ℕ) (A C : ℕ) 
    (h1 : total_units = 13200)
    (h2 : A + B + C = total_units)
    (h3 : ∃ d : ℕ, A = B - d ∧ C = B + d) :
    B = 4400 :=
by
  sorry

end units_produced_by_line_B_l60_60654


namespace max_value_sine_cosine_l60_60236

/-- If the maximum value of the function f(x) = 4 * sin x + a * cos x is 5, then a = ±3. -/
theorem max_value_sine_cosine (a : ℝ) :
  (∀ x : ℝ, 4 * Real.sin x + a * Real.cos x ≤ 5) →
  (∃ x : ℝ, 4 * Real.sin x + a * Real.cos x = 5) →
  a = 3 ∨ a = -3 :=
by
  sorry

end max_value_sine_cosine_l60_60236


namespace john_duck_price_l60_60556

theorem john_duck_price
  (n_ducks : ℕ)
  (cost_per_duck : ℕ)
  (weight_per_duck : ℕ)
  (total_profit : ℕ)
  (total_cost : ℕ)
  (total_weight : ℕ)
  (total_revenue : ℕ)
  (price_per_pound : ℕ)
  (h1 : n_ducks = 30)
  (h2 : cost_per_duck = 10)
  (h3 : weight_per_duck = 4)
  (h4 : total_profit = 300)
  (h5 : total_cost = n_ducks * cost_per_duck)
  (h6 : total_weight = n_ducks * weight_per_duck)
  (h7 : total_revenue = total_cost + total_profit)
  (h8 : price_per_pound = total_revenue / total_weight) :
  price_per_pound = 5 := 
sorry

end john_duck_price_l60_60556


namespace four_units_away_l60_60825

theorem four_units_away (x : ℤ) (h : abs (x + 2) = 4) : x = 2 ∨ x = -6 :=
by
  sorry

end four_units_away_l60_60825


namespace find_number_l60_60769

theorem find_number (m : ℤ) (h1 : ∃ k1 : ℤ, k1 * k1 = m + 100) (h2 : ∃ k2 : ℤ, k2 * k2 = m + 168) : m = 156 :=
sorry

end find_number_l60_60769


namespace waiter_earnings_l60_60158

theorem waiter_earnings
  (total_customers : ℕ)
  (no_tip_customers : ℕ)
  (tip_amount : ℕ)
  (customers_tipped : total_customers - no_tip_customers = 3)
  (tips_per_customer : tip_amount = 9) :
  (total_customers - no_tip_customers) * tip_amount = 27 := by
  sorry

end waiter_earnings_l60_60158


namespace solve_linear_eq_l60_60659

theorem solve_linear_eq (x : ℝ) : 3 * x - 6 = 0 ↔ x = 2 :=
sorry

end solve_linear_eq_l60_60659


namespace missed_bus_time_l60_60205

theorem missed_bus_time (T: ℕ) (speed_ratio: ℚ) (T_slow: ℕ) (missed_time: ℕ) : 
  T = 16 → speed_ratio = 4/5 → T_slow = (5/4) * T → missed_time = T_slow - T → missed_time = 4 :=
by
  sorry

end missed_bus_time_l60_60205


namespace perimeter_of_one_rectangle_l60_60486

theorem perimeter_of_one_rectangle (s : ℝ) (rectangle_perimeter rectangle_length rectangle_width : ℝ) (h1 : 4 * s = 240) (h2 : rectangle_width = (1/2) * s) (h3 : rectangle_length = s) (h4 : rectangle_perimeter = 2 * (rectangle_length + rectangle_width)) :
  rectangle_perimeter = 180 := 
sorry

end perimeter_of_one_rectangle_l60_60486


namespace a_7_value_l60_60398

noncomputable def geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
∀ n, a (n + 1) = r * a n

-- Given conditions
def geometric_sequence_positive_terms (a : ℕ → ℝ) : Prop :=
∀ n, a n > 0

def geometric_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
∀ n, S n = (a 0 * (1 - ((a (1 + n)) / a 0))) / (1 - (a 1 / a 0))

def S_4_eq_3S_2 (S : ℕ → ℝ) : Prop :=
S 4 = 3 * S 2

def a_3_eq_2 (a : ℕ → ℝ) : Prop :=
a 3 = 2

-- The statement to prove
theorem a_7_value (a : ℕ → ℝ) (S : ℕ → ℝ) (r : ℝ) :
  geometric_sequence a r →
  geometric_sequence_positive_terms a →
  geometric_sequence_sum a S →
  S_4_eq_3S_2 S →
  a_3_eq_2 a →
  a 7 = 8 :=
by
  sorry

end a_7_value_l60_60398


namespace remainder_of_square_l60_60412

theorem remainder_of_square (n : ℤ) (h : n % 5 = 3) : n^2 % 5 = 4 := 
by 
  sorry

end remainder_of_square_l60_60412


namespace exists_two_natural_pairs_satisfying_equation_l60_60549

theorem exists_two_natural_pairs_satisfying_equation :
  ∃ (x1 y1 x2 y2 : ℕ), (2 * x1^3 = y1^4) ∧ (2 * x2^3 = y2^4) ∧ ¬(x1 = x2 ∧ y1 = y2) :=
sorry

end exists_two_natural_pairs_satisfying_equation_l60_60549


namespace eat_both_veg_nonveg_l60_60055

theorem eat_both_veg_nonveg (total_veg only_veg : ℕ) (h1 : total_veg = 31) (h2 : only_veg = 19) :
  (total_veg - only_veg) = 12 :=
by
  have h3 : total_veg - only_veg = 31 - 19 := by rw [h1, h2]
  exact h3

end eat_both_veg_nonveg_l60_60055


namespace rebecca_gemstones_needed_l60_60928

-- Definitions for the conditions
def magnets_per_earring : Nat := 2
def buttons_per_magnet : Nat := 1 / 2
def gemstones_per_button : Nat := 3
def earrings_per_set : Nat := 2
def sets : Nat := 4

-- Statement to be proved
theorem rebecca_gemstones_needed : 
  gemstones_per_button * (buttons_per_magnet * (magnets_per_earring * (earrings_per_set * sets))) = 24 :=
by
  sorry

end rebecca_gemstones_needed_l60_60928


namespace max_cubes_fit_l60_60006

-- Define the conditions
def box_volume (length : ℕ) (width : ℕ) (height : ℕ) : ℕ := length * width * height
def cube_volume : ℕ := 27
def total_cubes (V_box : ℕ) (V_cube : ℕ) : ℕ := V_box / V_cube

-- Statement of the problem
theorem max_cubes_fit (length width height : ℕ) (V_box : ℕ) (V_cube q : ℕ) :
  length = 8 → width = 9 → height = 12 → V_box = box_volume length width height →
  V_cube = cube_volume → q = total_cubes V_box V_cube → q = 32 :=
by sorry

end max_cubes_fit_l60_60006


namespace cos_alpha_eq_2cos_alpha_plus_pi_div_4_implies_tan_alpha_plus_pi_div_8_l60_60316

theorem cos_alpha_eq_2cos_alpha_plus_pi_div_4_implies_tan_alpha_plus_pi_div_8
  (α : ℝ) (h : Real.cos α = 2 * Real.cos (α + Real.pi / 4)) :
  Real.tan (α + Real.pi / 8) = 3 * (Real.sqrt 2 + 1) := 
sorry

end cos_alpha_eq_2cos_alpha_plus_pi_div_4_implies_tan_alpha_plus_pi_div_8_l60_60316


namespace liza_final_balance_l60_60503

def initial_balance : ℕ := 800
def rent : ℕ := 450
def paycheck : ℕ := 1500
def electricity_bill : ℕ := 117
def internet_bill : ℕ := 100
def phone_bill : ℕ := 70

theorem liza_final_balance :
  initial_balance - rent + paycheck - (electricity_bill + internet_bill) - phone_bill = 1563 := by
  sorry

end liza_final_balance_l60_60503


namespace sum_of_powers_of_i_l60_60829

theorem sum_of_powers_of_i : 
  ∀ (i : ℂ), i^2 = -1 → 1 + i + i^2 + i^3 + i^4 + i^5 + i^6 = i :=
by
  intro i h
  sorry

end sum_of_powers_of_i_l60_60829


namespace triangle_angles_inequality_l60_60296

theorem triangle_angles_inequality (A B C : ℝ) (h1 : 0 < A) (h2 : 0 < B) (h3 : 0 < C) (h_sum : A + B + C = Real.pi) :
  1 / Real.sin (A / 2) + 1 / Real.sin (B / 2) + 1 / Real.sin (C / 2) ≥ 6 := 
sorry

end triangle_angles_inequality_l60_60296


namespace quadratic_solution_range_l60_60593

theorem quadratic_solution_range {x : ℝ} 
  (h : x^2 - 6 * x + 8 < 0) : 
  25 < x^2 + 6 * x + 9 ∧ x^2 + 6 * x + 9 < 49 :=
sorry

end quadratic_solution_range_l60_60593


namespace inequality_abc_l60_60459

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a + b - c) * (b + c - a) * (c + a - b) ≤ a * b * c := 
sorry

end inequality_abc_l60_60459


namespace side_length_of_square_l60_60364

theorem side_length_of_square :
  ∃ n : ℝ, n^2 = 9/16 ∧ n = 3/4 :=
sorry

end side_length_of_square_l60_60364


namespace range_of_values_l60_60815

variable (a : ℝ)

-- State the conditions
def prop.false (a : ℝ) : Prop := ¬ ∃ x : ℝ, a * x^2 + 4 * x + a ≤ 0

-- Prove that the range of values for a where the proposition is false is (2, +∞)
theorem range_of_values (ha : prop.false a) : 2 < a :=
sorry

end range_of_values_l60_60815


namespace total_frames_l60_60859

def frames_per_page : ℝ := 143.0

def pages : ℝ := 11.0

theorem total_frames : frames_per_page * pages = 1573.0 :=
by
  sorry

end total_frames_l60_60859


namespace problem1_problem2_problem3_problem4_l60_60717

theorem problem1 (α : ℝ) (h₁ : Real.sin α > 0) (h₂ : Real.tan α > 0) :
  α ∈ { x : ℝ | x >= 0 ∧ x < π/2 } := sorry

theorem problem2 (α : ℝ) (h₁ : Real.tan α * Real.sin α < 0) :
  α ∈ { x : ℝ | (x > π/2 ∧ x < π) ∨ (x > π ∧ x < 3 * π / 2) } := sorry

theorem problem3 (α : ℝ) (h₁ : Real.sin α * Real.cos α < 0) :
  α ∈ { x : ℝ | (x > π/2 ∧ x < π) ∨ (x > 3 * π / 2 ∧ x < 2 * π) } := sorry

theorem problem4 (α : ℝ) (h₁ : Real.cos α * Real.tan α > 0) :
  α ∈ { x : ℝ | x >= 0 ∧ x < π ∨ x > π ∧ x < 3 * π / 2 } := sorry

end problem1_problem2_problem3_problem4_l60_60717


namespace consecutive_numbers_probability_l60_60445

theorem consecutive_numbers_probability :
  let total_ways := Nat.choose 20 5
  let non_consecutive_ways := Nat.choose 16 5
  let probability_of_non_consecutive := (non_consecutive_ways : ℚ) / (total_ways : ℚ)
  let probability_of_consecutive := 1 - probability_of_non_consecutive
  probability_of_consecutive = 232 / 323 :=
by
  sorry

end consecutive_numbers_probability_l60_60445


namespace fill_time_correct_l60_60527

-- Define the conditions
def rightEyeTime := 2 * 24 -- hours
def leftEyeTime := 3 * 24 -- hours
def rightFootTime := 4 * 24 -- hours
def throatTime := 6       -- hours

def rightEyeRate := 1 / rightEyeTime
def leftEyeRate := 1 / leftEyeTime
def rightFootRate := 1 / rightFootTime
def throatRate := 1 / throatTime

-- Combined rate calculation
def combinedRate := rightEyeRate + leftEyeRate + rightFootRate + throatRate

-- Goal definition
def fillTime := 288 / 61 -- hours

-- Prove that the calculated time to fill the pool matches the given answer
theorem fill_time_correct : (1 / combinedRate) = fillTime :=
by {
  sorry
}

end fill_time_correct_l60_60527


namespace no_integer_solution_exists_l60_60433

theorem no_integer_solution_exists :
  ¬ ∃ m n : ℤ, m^3 = 3 * n^2 + 3 * n + 7 := by
  sorry

end no_integer_solution_exists_l60_60433


namespace smallest_n_common_factor_l60_60323

theorem smallest_n_common_factor :
  ∃ n : ℤ, n > 0 ∧ (gcd (8 * n - 3) (5 * n + 4) > 1) ∧ n = 10 :=
by
  sorry

end smallest_n_common_factor_l60_60323


namespace fraction_value_l60_60705

theorem fraction_value (p q x : ℚ) (h₁ : p / q = 4 / 5) (h₂ : 2 * q + p ≠ 0) (h₃ : 2 * q - p ≠ 0) :
  x + (2 * q - p) / (2 * q + p) = 2 → x = 11 / 7 :=
by
  sorry

end fraction_value_l60_60705


namespace initial_apples_l60_60420

theorem initial_apples (C : ℝ) (h : C + 7.0 = 27) : C = 20.0 := by
  sorry

end initial_apples_l60_60420


namespace bake_sale_comparison_l60_60403

theorem bake_sale_comparison :
  let tamara_small_brownies := 4 * 2
  let tamara_large_brownies := 12 * 3
  let tamara_cookies := 36 * 1.5
  let tamara_total := tamara_small_brownies + tamara_large_brownies + tamara_cookies

  let sarah_muffins := 24 * 1.75
  let sarah_choco_cupcakes := 7 * 2.5
  let sarah_vanilla_cupcakes := 8 * 2
  let sarah_strawberry_cupcakes := 15 * 2.75
  let sarah_total := sarah_muffins + sarah_choco_cupcakes + sarah_vanilla_cupcakes + sarah_strawberry_cupcakes

  sarah_total - tamara_total = 18.75 := by
  sorry

end bake_sale_comparison_l60_60403


namespace find_quadratic_polynomial_l60_60526

def quadratic_polynomial (a b c x : ℝ) : ℝ :=
  a * x^2 + b * x + c

theorem find_quadratic_polynomial : 
  ∃ a b c: ℝ, (∀ x : ℂ, quadratic_polynomial a b c x.re = 0 → (x = 3 + 4*I) ∨ (x = 3 - 4*I)) 
  ∧ (b = 8) 
  ∧ (a = -4/3) 
  ∧ (c = -50/3) :=
by
  sorry

end find_quadratic_polynomial_l60_60526


namespace values_are_equal_and_differ_in_precision_l60_60053

-- We define the decimal values
def val1 : ℝ := 4.5
def val2 : ℝ := 4.50

-- We define the counting units
def unit1 : ℝ := 0.1
def unit2 : ℝ := 0.01

-- Now, we state our theorem
theorem values_are_equal_and_differ_in_precision : 
  val1 = val2 ∧ unit1 ≠ unit2 :=
by
  -- Placeholder for the proof
  sorry

end values_are_equal_and_differ_in_precision_l60_60053


namespace coordinates_of_focus_with_greater_x_coordinate_l60_60979

noncomputable def focus_of_ellipse_with_greater_x_coordinate : (ℝ × ℝ) :=
  let center : ℝ × ℝ := (3, -2)
  let a : ℝ := 3 -- semi-major axis length
  let b : ℝ := 2 -- semi-minor axis length
  let c : ℝ := Real.sqrt (a^2 - b^2)
  let focus_x : ℝ := 3 + c
  (focus_x, -2)

theorem coordinates_of_focus_with_greater_x_coordinate :
  focus_of_ellipse_with_greater_x_coordinate = (3 + Real.sqrt 5, -2) := 
sorry

end coordinates_of_focus_with_greater_x_coordinate_l60_60979


namespace solve_for_x_l60_60608

-- Define the given equation as a predicate
def equation (x: ℚ) : Prop := (x + 4) / (x - 3) = (x - 2) / (x + 2)

-- State the problem in a Lean theorem
theorem solve_for_x : ∃ x : ℚ, equation x ∧ x = -2 / 11 :=
by
  existsi -2 / 11
  constructor
  repeat { sorry }

end solve_for_x_l60_60608


namespace radius_of_larger_circle_l60_60067

theorem radius_of_larger_circle (R1 R2 : ℝ) (α : ℝ) (h1 : α = 60) (h2 : R1 = 24) (h3 : R2 = 3 * R1) : 
  R2 = 72 := 
by
  sorry

end radius_of_larger_circle_l60_60067


namespace paint_room_together_l60_60033

variable (t : ℚ)
variable (Doug_rate : ℚ := 1/5)
variable (Dave_rate : ℚ := 1/7)
variable (Diana_rate : ℚ := 1/6)
variable (Combined_rate : ℚ := Doug_rate + Dave_rate + Diana_rate)
variable (break_time : ℚ := 2)

theorem paint_room_together:
  Combined_rate * (t - break_time) = 1 :=
sorry

end paint_room_together_l60_60033


namespace sabrina_basil_leaves_l60_60105

-- Definitions of variables
variables (S B V : ℕ)

-- Conditions as definitions in Lean
def condition1 : Prop := B = 2 * S
def condition2 : Prop := S = V - 5
def condition3 : Prop := B + S + V = 29

-- Problem statement
theorem sabrina_basil_leaves (h1 : condition1 S B) (h2 : condition2 S V) (h3 : condition3 S B V) : B = 12 :=
by {
  sorry
}

end sabrina_basil_leaves_l60_60105


namespace words_with_at_least_one_consonant_l60_60777

-- Define the letters available and classify them as vowels and consonants
def letters : List Char := ['A', 'B', 'C', 'D', 'E', 'F']
def vowels : List Char := ['A', 'E']
def consonants : List Char := ['B', 'C', 'D', 'F']

-- Define the total number of 5-letter words using the given letters
def total_words : ℕ := 6^5

-- Define the total number of 5-letter words composed exclusively of vowels
def vowel_words : ℕ := 2^5

-- Define the number of 5-letter words that contain at least one consonant
noncomputable def words_with_consonant : ℕ := total_words - vowel_words

-- The theorem to prove
theorem words_with_at_least_one_consonant : words_with_consonant = 7744 := by
  sorry

end words_with_at_least_one_consonant_l60_60777


namespace chord_line_parabola_l60_60681

theorem chord_line_parabola (x1 x2 y1 y2 : ℝ) (hx1 : y1^2 = 8*x1) (hx2 : y2^2 = 8*x2)
  (hmid : (x1 + x2) / 2 = 1 ∧ (y1 + y2) / 2 = -1) : 4*(1/2*(x1 + x2)) + (1/2*(y1 + y2)) - 3 = 0 :=
by
  sorry

end chord_line_parabola_l60_60681


namespace solve_x_l60_60605

theorem solve_x (x : ℝ) (h : (x - 1)^2 = 4) : x = 3 ∨ x = -1 :=
by
  -- proof goes here
  sorry

end solve_x_l60_60605


namespace squirrel_count_l60_60037

theorem squirrel_count (n m : ℕ) (h1 : n = 12) (h2 : m = 12 + 12 / 3) : n + m = 28 := by
  sorry

end squirrel_count_l60_60037


namespace find_certain_number_l60_60793

-- Define the conditions as constants
def n1 : ℕ := 9
def n2 : ℕ := 70
def n3 : ℕ := 25
def n4 : ℕ := 21
def smallest_given_number : ℕ := 3153
def certain_number : ℕ := 3147

-- Lean theorem statement
theorem find_certain_number (n1 n2 n3 n4 smallest_given_number certain_number: ℕ) :
  (∀ x, (∀ y ∈ [n1, n2, n3, n4], y ∣ x) → x ≥ smallest_given_number → x = smallest_given_number + certain_number) :=
sorry -- Skips the proof

end find_certain_number_l60_60793


namespace length_of_AB_l60_60569

theorem length_of_AB :
  ∃ (a b c d e : ℝ), (a < b) ∧ (b < c) ∧ (c < d) ∧ (d < e) ∧
  (b - a = 5) ∧ -- AB = 5
  ((c - b) = 2 * (d - c)) ∧ -- bc = 2 * cd
  (d - e) = 4 ∧ -- de = 4
  (c - a) = 11 ∧ -- ac = 11
  (e - a) = 18 := -- ae = 18
by 
  sorry

end length_of_AB_l60_60569


namespace insulation_cost_of_rectangular_tank_l60_60711

theorem insulation_cost_of_rectangular_tank
  (l w h cost_per_sq_ft : ℕ)
  (hl : l = 4) (hw : w = 5) (hh : h = 3) (hc : cost_per_sq_ft = 20) :
  2 * l * w + 2 * l * h + 2 * w * h * 20 = 1880 :=
by
  sorry

end insulation_cost_of_rectangular_tank_l60_60711


namespace range_of_a_l60_60697

theorem range_of_a 
    (x y a : ℝ) 
    (hx_pos : 0 < x) 
    (hy_pos : 0 < y) 
    (hxy : x + y = 1) 
    (hineq : ∀ (x y : ℝ), 0 < x → 0 < y → x + y = 1 → (1 / x + a / y) ≥ 4) :
    a ≥ 1 := 
by sorry

end range_of_a_l60_60697


namespace most_suitable_method_l60_60491

theorem most_suitable_method {x : ℝ} (h : (x - 1) ^ 2 = 4) :
  "Direct method of taking square root" = "Direct method of taking square root" :=
by
  -- We observe that the equation is already in a form 
  -- that is conducive to applying the direct method of taking the square root,
  -- because the equation is already a perfect square on one side and a constant on the other side.
  sorry

end most_suitable_method_l60_60491


namespace geom_seq_sum_first_10_terms_l60_60478

variable (a : ℕ → ℝ) (a₁ : ℝ) (q : ℝ)
variable (h₀ : a₁ = 1/4)
variable (h₁ : ∀ n, a (n + 1) = a₁ * q ^ n)
variable (S : ℕ → ℝ)
variable (h₂ : S n = a₁ * (1 - q ^ n) / (1 - q))

theorem geom_seq_sum_first_10_terms :
  a 1 = 1 / 4 →
  (a 3) * (a 5) = 4 * ((a 4) - 1) →
  S 10 = 1023 / 4 :=
by
  sorry

end geom_seq_sum_first_10_terms_l60_60478


namespace impossible_triangle_angle_sum_l60_60786

theorem impossible_triangle_angle_sum (x y z : ℝ) (h : x + y + z = 180) : x + y + z ≠ 360 :=
by
sorry

end impossible_triangle_angle_sum_l60_60786


namespace det_scaled_matrices_l60_60195

variable (a b c d : ℝ)

-- Given condition: determinant of the original matrix
def det_A : ℝ := Matrix.det ![![a, b], ![c, d]]

-- Problem statement: determinants of the scaled matrices
theorem det_scaled_matrices
    (h: det_A a b c d = 3) :
  Matrix.det ![![3 * a, 3 * b], ![3 * c, 3 * d]] = 27 ∧
  Matrix.det ![![4 * a, 2 * b], ![4 * c, 2 * d]] = 24 :=
by
  sorry

end det_scaled_matrices_l60_60195


namespace simple_interest_rate_l60_60178

theorem simple_interest_rate (P : ℝ) (T : ℝ) (r : ℝ) (h1 : T = 10) (h2 : (3 / 5) * P = (P * r * T) / 100) : r = 6 := by
  sorry

end simple_interest_rate_l60_60178


namespace coffee_price_increase_l60_60394

variable (C : ℝ) -- cost per pound of green tea and coffee in June
variable (P_green_tea_july : ℝ := 0.1) -- price of green tea per pound in July
variable (mixture_cost : ℝ := 3.15) -- cost of mixture of equal quantities of green tea and coffee for 3 lbs
variable (green_tea_cost_per_lb_july : ℝ := 0.1) -- cost per pound of green tea in July
variable (green_tea_weight : ℝ := 1.5) -- weight of green tea in the mixture in lbs
variable (coffee_weight : ℝ := 1.5) -- weight of coffee in the mixture in lbs
variable (coffee_cost_per_lb_july : ℝ := 2.0) -- cost per pound of coffee in July

theorem coffee_price_increase :
  C = 1 → mixture_cost = 3.15 →
  P_green_tea_july * C = green_tea_cost_per_lb_july →
  green_tea_weight * green_tea_cost_per_lb_july + coffee_weight * coffee_cost_per_lb_july = mixture_cost →
  (coffee_cost_per_lb_july - C) / C * 100 = 100 :=
by
  intros
  sorry

end coffee_price_increase_l60_60394


namespace find_total_amount_l60_60566

-- Definitions according to the conditions
def is_proportion (a b c : ℚ) (p q r : ℚ) : Prop :=
  (a * q = b * p) ∧ (a * r = c * p) ∧ (b * r = c * q)

def total_amount (second_part : ℚ) (prop_total : ℚ) : ℚ :=
  second_part / (1/3) * prop_total

-- Main statement to be proved
theorem find_total_amount (second_part : ℚ) (p1 p2 p3 : ℚ)
  (h : is_proportion p1 p2 p3 (1/2 : ℚ) (1/3 : ℚ) (3/4 : ℚ))
  : second_part = 164.6315789473684 → total_amount second_part (19/12 : ℚ) = 65.16 :=
by
  sorry

end find_total_amount_l60_60566


namespace six_lines_regions_l60_60353

def number_of_regions (n : ℕ) : ℕ := 1 + n + (n * (n - 1) / 2)

theorem six_lines_regions (h1 : 6 > 0) : 
    number_of_regions 6 = 22 :=
by 
  -- Use the formula for calculating number of regions:
  -- number_of_regions n = 1 + n + (n * (n - 1) / 2)
  sorry

end six_lines_regions_l60_60353


namespace geometric_sequence_fifth_term_l60_60943

theorem geometric_sequence_fifth_term
  (a : ℕ) (r : ℕ)
  (h₁ : a = 3)
  (h₂ : a * r^3 = 243) :
  a * r^4 = 243 :=
by
  sorry

end geometric_sequence_fifth_term_l60_60943


namespace part_one_part_two_l60_60960

variable {x : ℝ}

def setA (a : ℝ) : Set ℝ := {x | 0 < a * x + 1 ∧ a * x + 1 ≤ 5}
def setB : Set ℝ := {x | -1 / 2 < x ∧ x ≤ 2}

theorem part_one (a : ℝ) (h : a = 1) : setB ⊆ setA a :=
by
  sorry

theorem part_two (a : ℝ) : (setA a ⊆ setB) ↔ (a < -8 ∨ a ≥ 2) :=
by
  sorry

end part_one_part_two_l60_60960


namespace stanley_run_walk_difference_l60_60056

theorem stanley_run_walk_difference :
  ∀ (ran walked : ℝ), ran = 0.4 → walked = 0.2 → ran - walked = 0.2 :=
by
  intros ran walked h_ran h_walk
  rw [h_ran, h_walk]
  norm_num

end stanley_run_walk_difference_l60_60056


namespace david_older_than_rosy_l60_60826

theorem david_older_than_rosy
  (R D : ℕ) 
  (h1 : R = 12) 
  (h2 : D + 6 = 2 * (R + 6)) : 
  D - R = 18 := 
by
  sorry

end david_older_than_rosy_l60_60826


namespace abs_difference_of_mn_6_and_sum_7_l60_60254

theorem abs_difference_of_mn_6_and_sum_7 (m n : ℝ) (h₁ : m * n = 6) (h₂ : m + n = 7) : |m - n| = 5 := 
sorry

end abs_difference_of_mn_6_and_sum_7_l60_60254


namespace sum_A_k_div_k_l60_60872

noncomputable def A (k : ℕ) : ℕ :=
  (Finset.filter (fun d => d % 2 = 1 ∧ d ≤ Nat.sqrt (2 * k - 1)) (Finset.range k)).card

noncomputable def sumExpression : ℝ :=
  ∑' k, (-1)^(k-1) * (A k / k : ℝ)

theorem sum_A_k_div_k : sumExpression = Real.pi^2 / 8 :=
  sorry

end sum_A_k_div_k_l60_60872


namespace shorter_piece_length_l60_60246

theorem shorter_piece_length (x : ℕ) (h1 : ∃ l : ℕ, x + l = 120 ∧ l = 2 * x + 15) : x = 35 :=
sorry

end shorter_piece_length_l60_60246


namespace smallest_x_l60_60547

-- Define 450 and provide its factorization.
def n1 := 450
def n1_factors := 2^1 * 3^2 * 5^2

-- Define 675 and provide its factorization.
def n2 := 675
def n2_factors := 3^3 * 5^2

-- State the theorem that proves the smallest x for the condition
theorem smallest_x (x : ℕ) (hx : 450 * x % 675 = 0) : x = 3 := sorry

end smallest_x_l60_60547


namespace train_length_calculation_l60_60846

noncomputable def length_of_train (speed : ℝ) (time_in_sec : ℝ) : ℝ :=
  let time_in_hr := time_in_sec / 3600
  let distance_in_km := speed * time_in_hr
  distance_in_km * 1000

theorem train_length_calculation : 
  length_of_train 60 30 = 500 :=
by
  -- The proof would go here, but we provide a stub with sorry.
  sorry

end train_length_calculation_l60_60846


namespace bisection_method_example_l60_60116

noncomputable def f (x : ℝ) : ℝ := x^3 - 6 * x^2 + 4

theorem bisection_method_example :
  (∃ x : ℝ, 0 < x ∧ x < 1 ∧ f x = 0) →
  (∃ x : ℝ, (1 / 2) < x ∧ x < 1 ∧ f x = 0) :=
by
  sorry

end bisection_method_example_l60_60116


namespace inequality_proof_l60_60874

noncomputable def a : ℝ := Real.log 0.3 / Real.log 0.2
noncomputable def b : ℝ := Real.log 0.3 / Real.log 2

theorem inequality_proof : (a * b < a + b ∧ a + b < 0) :=
by
  sorry

end inequality_proof_l60_60874


namespace average_licks_to_center_l60_60833

theorem average_licks_to_center (Dan_lcks Michael_lcks Sam_lcks David_lcks Lance_lcks : ℕ)
  (h1 : Dan_lcks = 58) 
  (h2 : Michael_lcks = 63) 
  (h3 : Sam_lcks = 70) 
  (h4 : David_lcks = 70) 
  (h5 : Lance_lcks = 39) :
  (Dan_lcks + Michael_lcks + Sam_lcks + David_lcks + Lance_lcks) / 5 = 60 :=
by {
  sorry
}

end average_licks_to_center_l60_60833


namespace intersection_A_complement_B_l60_60748

def set_A : Set ℝ := {x | 1 < x ∧ x < 4}
def set_B : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def set_Complement_B : Set ℝ := {x | x < -1 ∨ x > 3}
def set_Intersection : Set ℝ := {x | set_A x ∧ set_Complement_B x}

theorem intersection_A_complement_B : set_Intersection = {x | 3 < x ∧ x < 4} := by
  sorry

end intersection_A_complement_B_l60_60748


namespace initial_southwards_distance_l60_60337

-- Define a structure that outlines the journey details
structure Journey :=
  (southwards : ℕ) 
  (westwards1 : ℕ := 10)
  (northwards : ℕ := 20)
  (westwards2 : ℕ := 20) 
  (home_distance : ℕ := 30)

-- Main theorem statement without proof
theorem initial_southwards_distance (j : Journey) : j.southwards + j.northwards = j.home_distance → j.southwards = 10 := by
  intro h
  sorry

end initial_southwards_distance_l60_60337


namespace Cornelia_three_times_Kilee_l60_60714

variable (x : ℕ)

def Kilee_current_age : ℕ := 20
def Cornelia_current_age : ℕ := 80

theorem Cornelia_three_times_Kilee (x : ℕ) :
  Cornelia_current_age + x = 3 * (Kilee_current_age + x) ↔ x = 10 :=
by
  sorry

end Cornelia_three_times_Kilee_l60_60714


namespace prime_problem_l60_60739

open Nat

-- Definition of primes and conditions based on the problem
def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- The formalized problem and conditions
theorem prime_problem (p q s : ℕ) 
  (p_prime : is_prime p) 
  (q_prime : is_prime q) 
  (s_prime : is_prime s) 
  (h1 : p + q = s + 4) 
  (h2 : 1 < p) 
  (h3 : p < q) : 
  p = 2 :=
sorry

end prime_problem_l60_60739


namespace probability_two_even_balls_l60_60441

theorem probability_two_even_balls
  (total_balls : ℕ)
  (even_balls : ℕ)
  (h_total : total_balls = 16)
  (h_even : even_balls = 8)
  (first_draw : ℕ → ℚ)
  (second_draw : ℕ → ℚ)
  (h_first : first_draw even_balls = even_balls / total_balls)
  (h_second : second_draw (even_balls - 1) = (even_balls - 1) / (total_balls - 1)) :
  (first_draw even_balls) * (second_draw (even_balls - 1)) = 7 / 30 := 
sorry

end probability_two_even_balls_l60_60441


namespace Timmy_needs_to_go_faster_l60_60529

-- Define the trial speeds and the required speed
def s1 : ℕ := 36
def s2 : ℕ := 34
def s3 : ℕ := 38
def s_req : ℕ := 40

-- Statement of the theorem
theorem Timmy_needs_to_go_faster :
  s_req - (s1 + s2 + s3) / 3 = 4 :=
by
  sorry

end Timmy_needs_to_go_faster_l60_60529


namespace tax_diminished_percentage_l60_60611

theorem tax_diminished_percentage (T C : ℝ) (hT : T > 0) (hC : C > 0) (X : ℝ) 
  (h : T * (1 - X / 100) * C * 1.15 = T * C * 0.9315) : X = 19 :=
by 
  sorry

end tax_diminished_percentage_l60_60611


namespace table_filling_impossible_l60_60244

theorem table_filling_impossible :
  ∀ (table : Fin 5 → Fin 8 → Fin 10),
  (∀ digit : Fin 10, ∃ row_set : Finset (Fin 5), row_set.card = 4 ∧
    (∀ row : Fin 5, row ∈ row_set → ∃ col_set : Finset (Fin 8), col_set.card = 4 ∧
      (∀ col : Fin 8, col ∈ col_set → table row col = digit))) →
  False :=
by
  sorry

end table_filling_impossible_l60_60244


namespace germany_fraction_closest_japan_fraction_closest_l60_60738

noncomputable def fraction_approx (a b : ℕ) : ℚ := a / b

theorem germany_fraction_closest :
  abs (fraction_approx 23 150 - fraction_approx 1 7) < 
  min (abs (fraction_approx 23 150 - fraction_approx 1 5))
      (min (abs (fraction_approx 23 150 - fraction_approx 1 6))
           (min (abs (fraction_approx 23 150 - fraction_approx 1 8))
                (abs (fraction_approx 23 150 - fraction_approx 1 9)))) :=
by sorry

theorem japan_fraction_closest :
  abs (fraction_approx 27 150 - fraction_approx 1 6) < 
  min (abs (fraction_approx 27 150 - fraction_approx 1 5))
      (min (abs (fraction_approx 27 150 - fraction_approx 1 7))
           (min (abs (fraction_approx 27 150 - fraction_approx 1 8))
                (abs (fraction_approx 27 150 - fraction_approx 1 9)))) :=
by sorry

end germany_fraction_closest_japan_fraction_closest_l60_60738


namespace average_number_of_visitors_is_25_l60_60560

-- Define the sequence parameters
def a : ℕ := 10  -- First term
def d : ℕ := 5   -- Common difference
def n : ℕ := 7   -- Number of days

-- Define the sequence for the number of visitors on each day
def visitors (i : ℕ) : ℕ := a + (i - 1) * d

-- Define the average number of visitors
def avg_visitors : ℕ := (List.sum (List.map visitors [1, 2, 3, 4, 5, 6, 7])) / n

-- Prove the average
theorem average_number_of_visitors_is_25 : avg_visitors = 25 :=
by
  -- Placeholder for the actual proof
  sorry

end average_number_of_visitors_is_25_l60_60560


namespace part_I_part_II_l60_60285

-- Define the function f
def f (x a : ℝ) := |x - a| + |x - 2|

-- Statement for part (I)
theorem part_I (a : ℝ) (h : ∃ x : ℝ, f x a ≤ a) : a ≥ 1 := sorry

-- Statement for part (II)
theorem part_II (m n p : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : p > 0) (h4 : m + 2 * n + 3 * p = 1) : 
  (3 / m) + (2 / n) + (1 / p) ≥ 6 + 2 * Real.sqrt 6 + 2 * Real.sqrt 2 := sorry

end part_I_part_II_l60_60285


namespace total_food_eaten_l60_60259

theorem total_food_eaten (num_puppies num_dogs : ℕ)
    (dog_food_per_meal dog_meals_per_day puppy_food_per_day : ℕ)
    (dog_food_mult puppy_meal_mult : ℕ)
    (h1 : num_puppies = 6)
    (h2 : num_dogs = 5)
    (h3 : dog_food_per_meal = 6)
    (h4 : dog_meals_per_day = 2)
    (h5 : dog_food_mult = 3)
    (h6 : puppy_meal_mult = 4)
    (h7 : puppy_food_per_day = (dog_food_per_meal / dog_food_mult) * puppy_meal_mult * dog_meals_per_day) :
    (num_dogs * dog_food_per_meal * dog_meals_per_day + num_puppies * puppy_food_per_day) = 108 := by
  -- conclude the theorem
  sorry

end total_food_eaten_l60_60259


namespace equalize_cheese_pieces_l60_60257

-- Defining the initial masses of the three pieces of cheese
def cheese1 : ℕ := 5
def cheese2 : ℕ := 8
def cheese3 : ℕ := 11

-- State that the fox can cut 1g simultaneously from any two pieces
def can_equalize_masses (cut_action : ℕ → ℕ → ℕ → Prop) : Prop :=
  ∃ n1 n2 n3 _ : ℕ,
    cut_action cheese1 cheese2 cheese3 ∧
    (n1 = 0 ∧ n2 = 0 ∧ n3 = 0)

-- Introducing the fox's cut action
def cut_action (a b c : ℕ) : Prop :=
  (∃ x : ℕ, x ≥ 0 ∧ a - x ≥ 0 ∧ b - x ≥ 0 ∧ c ≤ cheese3) ∧
  (∃ y : ℕ, y ≥ 0 ∧ a - y ≥ 0 ∧ b ≤ cheese2 ∧ c - y ≥ 0) ∧
  (∃ z : ℕ, z ≥ 0 ∧ a ≤ cheese1 ∧ b - z ≥ 0 ∧ c - z ≥ 0) 

-- The theorem that proves it's possible to equalize the masses
theorem equalize_cheese_pieces : can_equalize_masses cut_action :=
by
  sorry

end equalize_cheese_pieces_l60_60257


namespace average_of_three_l60_60891

theorem average_of_three (y : ℝ) (h : (15 + 24 + y) / 3 = 20) : y = 21 :=
by
  sorry

end average_of_three_l60_60891


namespace a_and_b_are_kth_powers_l60_60501

theorem a_and_b_are_kth_powers (k : ℕ) (h_k : 1 < k) (a b : ℤ) (h_rel_prime : Int.gcd a b = 1)
  (c : ℤ) (h_ab_power : a * b = c^k) : ∃ (m n : ℤ), a = m^k ∧ b = n^k :=
by
  sorry

end a_and_b_are_kth_powers_l60_60501


namespace inequality_system_solution_l60_60812

theorem inequality_system_solution (x : ℝ) :
  (3 * x > x + 6) ∧ ((1 / 2) * x < -x + 5) ↔ (3 < x) ∧ (x < 10 / 3) :=
by
  sorry

end inequality_system_solution_l60_60812


namespace num_valid_a_values_l60_60870

theorem num_valid_a_values : 
  ∃ S : Finset ℕ, (∀ a ∈ S, a < 100 ∧ (a^3 + 23) % 24 = 0) ∧ S.card = 5 :=
sorry

end num_valid_a_values_l60_60870


namespace two_b_is_16667_percent_of_a_l60_60551

theorem two_b_is_16667_percent_of_a {a b : ℝ} (h : a = 1.2 * b) : (2 * b / a) = 5 / 3 := by
  sorry

end two_b_is_16667_percent_of_a_l60_60551


namespace first_prize_ticket_numbers_l60_60980

theorem first_prize_ticket_numbers :
  {n : ℕ | n < 10000 ∧ (n % 1000 = 418)} = {418, 1418, 2418, 3418, 4418, 5418, 6418, 7418, 8418, 9418} :=
by
  sorry

end first_prize_ticket_numbers_l60_60980


namespace sufficient_not_necessary_condition_l60_60462

theorem sufficient_not_necessary_condition (x : ℝ) (a : ℝ × ℝ) (b : ℝ × ℝ) (hx : x = 2)
    (ha : a = (x, 1)) (hb : b = (4, x)) : 
    (∃ k : ℝ, a = (k * b.1, k * b.2)) ∧ (¬ (∀ k : ℝ, a = (k * b.1, k * b.2))) :=
by 
  sorry

end sufficient_not_necessary_condition_l60_60462


namespace rebecca_haircut_charge_l60_60172

-- Define the conditions
variable (H : ℕ) -- Charge for a haircut
def perm_charge : ℕ := 40
def dye_charge : ℕ := 60
def dye_cost : ℕ := 10
def haircuts_today : ℕ := 4
def perms_today : ℕ := 1
def dye_jobs_today : ℕ := 2
def tips_today : ℕ := 50
def total_amount_end_day : ℕ := 310

-- State the proof problem
theorem rebecca_haircut_charge :
  4 * H + perms_today * perm_charge + dye_jobs_today * dye_charge + tips_today - dye_jobs_today * dye_cost = total_amount_end_day →
  H = 30 :=
by
  sorry

end rebecca_haircut_charge_l60_60172


namespace sum_arithmetic_sequence_has_max_value_l60_60372

noncomputable section
open Classical

-- Defining an arithmetic sequence with first term a1 and common difference d
def arithmetic_sequence (a1 d : ℝ) (n : ℕ) : ℝ :=
  a1 + d * (n - 1)

-- Defining the sum of the first n terms of the arithmetic sequence
def sum_arithmetic_sequence (a1 d : ℝ) (n : ℕ) : ℝ :=
  n / 2 * (2 * a1 + (n - 1) * d)

-- The main statement to prove: Sn has a maximum value given conditions a1 > 0 and d < 0
theorem sum_arithmetic_sequence_has_max_value (a1 d : ℝ) (h1 : a1 > 0) (h2 : d < 0) :
  ∃ M, ∀ n, sum_arithmetic_sequence a1 d n ≤ M :=
by
  sorry

end sum_arithmetic_sequence_has_max_value_l60_60372


namespace balance_balls_l60_60193

theorem balance_balls (G Y B W : ℝ) (h₁ : 4 * G = 10 * B) (h₂ : 3 * Y = 8 * B) (h₃ : 8 * B = 6 * W) :
  5 * G + 5 * Y + 4 * W = 31.1 * B :=
by
  sorry

end balance_balls_l60_60193


namespace evaluate_expr_l60_60515

def x := 2
def y := -1
def z := 3
def expr := 2 * x^2 + y^2 - z^2 + 3 * x * y

theorem evaluate_expr : expr = -6 :=
by sorry

end evaluate_expr_l60_60515


namespace prisha_other_number_l60_60583

def prisha_numbers (a b : ℤ) : Prop :=
  3 * a + 2 * b = 105 ∧ (a = 15 ∨ b = 15)

theorem prisha_other_number (a b : ℤ) (h : prisha_numbers a b) : b = 30 :=
sorry

end prisha_other_number_l60_60583


namespace absolute_value_condition_necessary_non_sufficient_l60_60289

theorem absolute_value_condition_necessary_non_sufficient (x : ℝ) :
  (abs (x - 1) < 2 → x^2 < x) ∧ ¬ (x^2 < x → abs (x - 1) < 2) := sorry

end absolute_value_condition_necessary_non_sufficient_l60_60289


namespace maximise_expression_l60_60465

theorem maximise_expression {x : ℝ} (hx : 0 < x ∧ x < 1) : 
  ∃ (x_max : ℝ), x_max = 1/2 ∧ 
  (∀ y : ℝ, (0 < y ∧ y < 1) → 3 * y * (1 - y) ≤ 3 * x_max * (1 - x_max)) :=
sorry

end maximise_expression_l60_60465


namespace infinite_primes_solutions_l60_60640

theorem infinite_primes_solutions :
  ∀ (P : Finset ℕ), (∀ p ∈ P, Prime p) →
  ∃ q, Prime q ∧ q ∉ P ∧ ∃ x y : ℤ, x^2 + x + 1 = q * y :=
by sorry

end infinite_primes_solutions_l60_60640


namespace find_value_of_a_l60_60864

theorem find_value_of_a (a : ℝ) (h : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → 0 < a → a^x ≥ 1)
  (h_sum : (a^1) + (a^0) = 3) : a = 2 :=
sorry

end find_value_of_a_l60_60864


namespace find_a_l60_60373

noncomputable def f (x : ℝ) : ℝ :=
if x >= 0 then
  x * (x + 1)
else
  -((-x) * ((-x) + 1))

theorem find_a (a : ℝ) (h_odd : ∀ x : ℝ, f (-x) = -f x) (h_pos : ∀ x : ℝ, x >= 0 → f x = x * (x + 1)) (h_a: f a = -2) : a = -1 :=
sorry

end find_a_l60_60373


namespace original_denominator_value_l60_60446

theorem original_denominator_value (d : ℕ) (h1 : 3 + 3 = 6) (h2 : ((6 : ℕ) / (d + 3 : ℕ) = (1 / 3 : ℚ))) : d = 15 :=
sorry

end original_denominator_value_l60_60446


namespace range_of_m_l60_60580

theorem range_of_m (m : ℝ) (h : ∃ (x : ℝ), x > 0 ∧ 2 * x + m - 3 = 0) : m < 3 :=
sorry

end range_of_m_l60_60580


namespace exists_n_geq_k_l60_60522

theorem exists_n_geq_k (a : ℕ → ℕ) (h_distinct : ∀ i j : ℕ, i ≠ j → a i ≠ a j) 
    (h_positive : ∀ i : ℕ, a i > 0) :
    ∀ k : ℕ, ∃ n : ℕ, n > k ∧ a n ≥ n :=
by
  intros k
  sorry

end exists_n_geq_k_l60_60522


namespace count_even_three_digit_numbers_less_than_600_l60_60096

-- Define the digits
def digits : List ℕ := [1, 2, 3, 4, 5, 6]

-- Condition: the number must be less than 600, i.e., hundreds digit in {1, 2, 3, 4, 5}
def valid_hundreds (d : ℕ) : Prop := d ∈ [1, 2, 3, 4, 5]

-- Condition: the units (ones) digit must be even
def valid_units (d : ℕ) : Prop := d ∈ [2, 4, 6]

-- Problem: total number of valid three-digit numbers
def total_valid_numbers : ℕ :=
  List.product (List.product [1, 2, 3, 4, 5] digits) [2, 4, 6] |>.length

-- Proof statement
theorem count_even_three_digit_numbers_less_than_600 :
  total_valid_numbers = 90 := by
  sorry

end count_even_three_digit_numbers_less_than_600_l60_60096


namespace find_a1_l60_60615

theorem find_a1 (S : ℕ → ℝ) (a : ℕ → ℝ) (a1 : ℝ) :
  (∀ n : ℕ, S n = a1 * (2^n - 1)) → a 4 = 24 → 
  a 4 = S 4 - S 3 → 
  a1 = 3 :=
by
  sorry

end find_a1_l60_60615


namespace bus_speed_l60_60351

theorem bus_speed (S : ℝ) (h1 : 36 = S * (2 / 3)) : S = 54 :=
by
sorry

end bus_speed_l60_60351


namespace man_upstream_rate_l60_60724

theorem man_upstream_rate (rate_downstream : ℝ) (rate_still_water : ℝ) (rate_current : ℝ) 
    (h1 : rate_downstream = 32) (h2 : rate_still_water = 24.5) (h3 : rate_current = 7.5) : 
    rate_still_water - rate_current = 17 := 
by 
  sorry

end man_upstream_rate_l60_60724


namespace find_f_ln6_l60_60076

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def condition1 (f : ℝ → ℝ) : Prop :=
  ∀ x, x < 0 → f x = x - Real.exp (-x)

noncomputable def given_function_value : ℝ := Real.log 6

theorem find_f_ln6 (f : ℝ → ℝ)
  (h1 : odd_function f)
  (h2 : condition1 f) :
  f given_function_value = given_function_value + 6 :=
by
  sorry

end find_f_ln6_l60_60076


namespace books_left_in_library_l60_60913

theorem books_left_in_library (initial_books : ℕ) (borrowed_books : ℕ) (left_books : ℕ) 
  (h1 : initial_books = 75) (h2 : borrowed_books = 18) : left_books = 57 :=
by
  sorry

end books_left_in_library_l60_60913


namespace painter_rooms_painted_l60_60406

theorem painter_rooms_painted (total_rooms : ℕ) (hours_per_room : ℕ) (remaining_hours : ℕ) 
    (h1 : total_rooms = 9) (h2 : hours_per_room = 8) (h3 : remaining_hours = 32) : 
    total_rooms - (remaining_hours / hours_per_room) = 5 :=
by
  sorry

end painter_rooms_painted_l60_60406


namespace bob_calories_consumed_l60_60470

/-- Bob eats half of the pizza with 8 slices, each slice being 300 calories.
   Prove that Bob eats 1200 calories. -/
theorem bob_calories_consumed (total_slices : ℕ) (calories_per_slice : ℕ) (half_slices : ℕ) (calories_consumed : ℕ) 
  (h1 : total_slices = 8)
  (h2 : calories_per_slice = 300)
  (h3 : half_slices = total_slices / 2)
  (h4 : calories_consumed = half_slices * calories_per_slice) 
  : calories_consumed = 1200 := 
sorry

end bob_calories_consumed_l60_60470


namespace last_integer_in_sequence_is_one_l60_60397

theorem last_integer_in_sequence_is_one :
  ∀ seq : ℕ → ℕ, (seq 0 = 37) ∧ (∀ n, seq (n + 1) = seq n / 2) → (∃ n, seq (n + 1) = 0 ∧ seq n = 1) :=
by
  sorry

end last_integer_in_sequence_is_one_l60_60397


namespace vacation_cost_l60_60740

theorem vacation_cost (C : ℝ) (h1 : C / 3 - C / 4 = 30) : C = 360 :=
by
  sorry

end vacation_cost_l60_60740


namespace jack_reads_books_in_a_year_l60_60320

/-- If Jack reads 9 books per day, how many books can he read in a year (365 days)? -/
theorem jack_reads_books_in_a_year (books_per_day : ℕ) (days_per_year : ℕ) (books_per_year : ℕ) (h1 : books_per_day = 9) (h2 : days_per_year = 365) : books_per_year = 3285 :=
by
  sorry

end jack_reads_books_in_a_year_l60_60320


namespace incorrect_option_l60_60389

-- Definitions and conditions from the problem
def p (x : ℝ) : Prop := (x - 2) * Real.sqrt (x^2 - 3*x + 2) ≥ 0
def q (k : ℝ) : Prop := ∀ x : ℝ, k * x^2 - k * x - 1 < 0

-- The Lean 4 statement to verify the problem
theorem incorrect_option :
  (¬ ∃ x, p x) ∧ (∃ k, q k) ∧
  (∀ k, -4 < k ∧ k ≤ 0 → q k) →
  (∃ x, ¬p x) :=
  by
  sorry

end incorrect_option_l60_60389


namespace equivalent_expression_l60_60428

noncomputable def problem_statement (α β γ δ p q : ℝ) :=
  (α - γ) * (β - γ) * (α + δ) * (β + δ) = -2 * (p^2 - q^2) + 4

theorem equivalent_expression
  (α β γ δ p q : ℝ)
  (h1 : ∀ x, x^2 + p * x + 2 = 0 → (x = α ∨ x = β))
  (h2 : ∀ x, x^2 + q * x + 2 = 0 → (x = γ ∨ x = δ)) :
  problem_statement α β γ δ p q :=
by sorry

end equivalent_expression_l60_60428


namespace ab_necessary_but_not_sufficient_l60_60936

theorem ab_necessary_but_not_sufficient (a b : ℝ) (i : ℂ) (hi : i^2 = -1) : 
  ab < 0 → ¬ (ab >= 0) ∧ (¬ (ab <= 0)) → (z = i * (a + b * i)) ∧ a > 0 ∧ -b > 0 := 
  sorry

end ab_necessary_but_not_sufficient_l60_60936


namespace Kimiko_age_proof_l60_60142

variables (Kimiko_age Kayla_age : ℕ)
variables (min_driving_age wait_years : ℕ)

def is_half_age (a b : ℕ) : Prop := a = b / 2
def minimum_driving_age (a b : ℕ) : Prop := a + b = 18

theorem Kimiko_age_proof
  (h1 : is_half_age Kayla_age Kimiko_age)
  (h2 : wait_years = 5)
  (h3 : minimum_driving_age Kayla_age wait_years) :
  Kimiko_age = 26 :=
sorry

end Kimiko_age_proof_l60_60142


namespace kids_in_group_l60_60203

theorem kids_in_group :
  ∃ (K : ℕ), (∃ (A : ℕ), A + K = 9 ∧ 2 * A = 14) ∧ K = 2 :=
by
  sorry

end kids_in_group_l60_60203


namespace original_apples_l60_60355

-- Define the conditions using the given data
def sells_fraction : ℝ := 0.40 -- Fraction of apples sold
def remaining_apples : ℝ := 420 -- Apples remaining after selling

-- Theorem statement for proving the original number of apples given the conditions
theorem original_apples (x : ℝ) (sells_fraction : ℝ := 0.40) (remaining_apples : ℝ := 420) : 
  420 / (1 - sells_fraction) = x :=
sorry

end original_apples_l60_60355


namespace f_has_two_zeros_l60_60964

def f (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 1

theorem f_has_two_zeros : ∃ (x1 x2 : ℝ), f x1 = 0 ∧ f x2 = 0 ∧ x1 ≠ x2 := 
by
  sorry

end f_has_two_zeros_l60_60964


namespace initial_investment_B_l60_60563

theorem initial_investment_B (A_initial : ℝ) (B : ℝ) (total_profit : ℝ) (A_profit : ℝ) 
(A_withdraw : ℝ) (B_advance : ℝ) : 
  A_initial = 3000 → B_advance = 1000 → A_withdraw = 1000 → total_profit = 756 → A_profit = 288 → 
  (8 * A_initial + 4 * (A_initial - A_withdraw)) / (8 * B + 4 * (B + B_advance)) = A_profit / (total_profit - A_profit) → 
  B = 4000 := 
by 
  intros h1 h2 h3 h4 h5 h6 
  sorry

end initial_investment_B_l60_60563


namespace find_b_l60_60772

theorem find_b 
  (a b c d : ℚ) 
  (h1 : a = 2 * b + c) 
  (h2 : b = 2 * c + d) 
  (h3 : 2 * c = d + a - 1) 
  (h4 : d = a - c) : 
  b = 2 / 9 :=
by
  -- Proof is omitted (the proof steps would be inserted here)
  sorry

end find_b_l60_60772


namespace smallest_cube_dividing_pq2r4_l60_60722

-- Definitions of conditions
variables {p q r : ℕ} [Fact (Nat.Prime p)] [Fact (Nat.Prime q)] [Fact (Nat.Prime r)]
variables (h_distinct : p ≠ q ∧ p ≠ r ∧ q ≠ r)

-- Definitions used in the proof
def is_perfect_cube (n : ℕ) : Prop :=
  ∃ m : ℕ, m^3 = n

def smallest_perfect_cube_dividing (n k : ℕ) : Prop :=
  is_perfect_cube k ∧ n ∣ k ∧ ∀ k', is_perfect_cube k' ∧ n ∣ k' → k ≤ k'

-- The proof problem
theorem smallest_cube_dividing_pq2r4 (h_distinct : p ≠ q ∧ p ≠ r ∧ q ≠ r) :
  smallest_perfect_cube_dividing (p * q^2 * r^4) ((p * q * r^2)^3) :=
sorry

end smallest_cube_dividing_pq2r4_l60_60722


namespace tetrahedron_paintings_l60_60591

theorem tetrahedron_paintings (n : ℕ) (h : n ≥ 4) : 
  let term1 := (n - 1) * (n - 2) * (n - 3) / 12
  let term2 := (n - 1) * (n - 2) / 3
  let term3 := n - 1
  let term4 := 1
  2 * (term1 + term2 + term3) + n = 
  n * (term1 + term2 + term3 + term4) := by
{
  sorry
}

end tetrahedron_paintings_l60_60591


namespace order_wxyz_l60_60245

def w : ℕ := 2^129 * 3^81 * 5^128
def x : ℕ := 2^127 * 3^81 * 5^128
def y : ℕ := 2^126 * 3^82 * 5^128
def z : ℕ := 2^125 * 3^82 * 5^129

theorem order_wxyz : x < y ∧ y < z ∧ z < w := by
  sorry

end order_wxyz_l60_60245


namespace avg_salary_rest_of_workers_l60_60295

theorem avg_salary_rest_of_workers (avg_all : ℝ) (avg_tech : ℝ) (total_workers : ℕ)
  (total_avg_salary : avg_all = 8000) (tech_avg_salary : avg_tech = 12000) (workers_count : total_workers = 30) :
  (20 * (total_workers * avg_all - 10 * avg_tech) / 20) = 6000 :=
by
  sorry

end avg_salary_rest_of_workers_l60_60295


namespace larger_box_can_carry_more_clay_l60_60674

variable {V₁ : ℝ} -- Volume of the first box
variable {V₂ : ℝ} -- Volume of the second box
variable {m₁ : ℝ} -- Mass the first box can carry
variable {m₂ : ℝ} -- Mass the second box can carry

-- Defining the dimensions of the first box.
def height₁ : ℝ := 1
def width₁ : ℝ := 2
def length₁ : ℝ := 4

-- Defining the dimensions of the second box.
def height₂ : ℝ := 3 * height₁
def width₂ : ℝ := 2 * width₁
def length₂ : ℝ := 2 * length₁

-- Volume calculation for the first box.
def volume₁ : ℝ := height₁ * width₁ * length₁

-- Volume calculation for the second box.
def volume₂ : ℝ := height₂ * width₂ * length₂

-- Condition: The first box can carry 30 grams of clay
def mass₁ : ℝ := 30

-- Given the above conditions, prove the second box can carry 360 grams of clay.
theorem larger_box_can_carry_more_clay (h₁ : volume₁ = height₁ * width₁ * length₁)
                                      (h₂ : volume₂ = height₂ * width₂ * length₂)
                                      (h₃ : mass₁ = 30)
                                      (h₄ : V₁ = volume₁)
                                      (h₅ : V₂ = volume₂) :
  m₂ = 12 * mass₁ := by
  -- Skipping the detailed proof.
  sorry

end larger_box_can_carry_more_clay_l60_60674


namespace general_term_of_an_l60_60382

theorem general_term_of_an (a : ℕ → ℕ) (h1 : a 1 = 1)
    (h_rec : ∀ n : ℕ, a (n + 1) = 2 * a n + 1) :
    ∀ n : ℕ, a n = 2^n - 1 :=
sorry

end general_term_of_an_l60_60382


namespace dot_product_bounds_l60_60701

theorem dot_product_bounds
  (A : ℝ × ℝ)
  (hA : A.1 ^ 2 + (A.2 - 1) ^ 2 = 1) :
  -2 ≤ A.1 * 2 ∧ A.1 * 2 ≤ 2 := 
sorry

end dot_product_bounds_l60_60701


namespace max_quotient_l60_60802

theorem max_quotient (a b : ℝ) (ha : 300 ≤ a ∧ a ≤ 500) (hb : 800 ≤ b ∧ b ≤ 1600) : ∃ q, q = b / a ∧ q ≤ 16 / 3 :=
by 
  sorry

end max_quotient_l60_60802


namespace gcd_A_B_l60_60345

def A : ℤ := 1989^1990 - 1988^1990
def B : ℤ := 1989^1989 - 1988^1989

theorem gcd_A_B : Int.gcd A B = 1 := 
by
  -- Conditions
  have h1 : A = 1989^1990 - 1988^1990 := rfl
  have h2 : B = 1989^1989 - 1988^1989 := rfl
  -- Conclusion
  sorry

end gcd_A_B_l60_60345


namespace tan_C_over_tan_A_max_tan_B_l60_60248

theorem tan_C_over_tan_A {A B C : ℝ} {a b c : ℝ} (h : a^2 + 2 * b^2 = c^2) :
  let tan_A := Real.tan A
  let tan_C := Real.tan C
  (Real.tan C / Real.tan A) = -3 :=
sorry

theorem max_tan_B {A B C : ℝ} {a b c : ℝ} (h : a^2 + 2 * b^2 = c^2) :
  let B := Real.arctan (Real.tan B)
  ∃ (x : ℝ), x = Real.tan B ∧ ∀ y, y = Real.tan B → y ≤ (Real.sqrt 3) / 3 :=
sorry

end tan_C_over_tan_A_max_tan_B_l60_60248


namespace reading_ratio_l60_60720

theorem reading_ratio (x : ℕ) (h1 : 10 * x + 5 * (75 - x) = 500) : 
  (10 * x) / 500 = 1 / 2 :=
by sorry

end reading_ratio_l60_60720


namespace product_last_digit_l60_60340

def last_digit (n : ℕ) : ℕ := n % 10

theorem product_last_digit :
  last_digit (3^65 * 6^59 * 7^71) = 4 :=
by
  sorry

end product_last_digit_l60_60340


namespace total_players_is_60_l60_60383

-- Define the conditions
def Cricket_players : ℕ := 25
def Hockey_players : ℕ := 20
def Football_players : ℕ := 30
def Softball_players : ℕ := 18

def Cricket_and_Hockey : ℕ := 5
def Cricket_and_Football : ℕ := 8
def Cricket_and_Softball : ℕ := 3
def Hockey_and_Football : ℕ := 4
def Hockey_and_Softball : ℕ := 6
def Football_and_Softball : ℕ := 9

def Cricket_Hockey_and_Football_not_Softball : ℕ := 2

-- Define total unique players present on the ground
def total_unique_players : ℕ :=
  Cricket_players + Hockey_players + Football_players + Softball_players -
  (Cricket_and_Hockey + Cricket_and_Football + Cricket_and_Softball +
   Hockey_and_Football + Hockey_and_Softball + Football_and_Softball) +
  Cricket_Hockey_and_Football_not_Softball

-- Statement
theorem total_players_is_60:
  total_unique_players = 60 :=
by
  sorry

end total_players_is_60_l60_60383


namespace population_in_terms_of_t_l60_60156

noncomputable def boys_girls_teachers_total (b g t : ℕ) : ℕ :=
  b + g + t

theorem population_in_terms_of_t (b g t : ℕ) (h1 : b = 4 * g) (h2 : g = 5 * t) :
  boys_girls_teachers_total b g t = 26 * t :=
by
  sorry

end population_in_terms_of_t_l60_60156


namespace polyhedron_volume_l60_60104

-- Define the properties of the polygons
def isosceles_right_triangle (a : ℝ) := a ≠ 0 ∧ ∀ (x y : ℝ), x = y

def square (side : ℝ) := side = 2

def equilateral_triangle (side : ℝ) := side = 2 * Real.sqrt 2

-- Define the conditions
def condition_AE : Prop := isosceles_right_triangle 2
def condition_B : Prop := square 2
def condition_C : Prop := square 2
def condition_D : Prop := square 2
def condition_G : Prop := equilateral_triangle (2 * Real.sqrt 2)

-- Define the polyhedron volume calculation problem
theorem polyhedron_volume (hA : condition_AE) (hE : condition_AE) (hF : condition_AE) (hB : condition_B) (hC : condition_C) (hD : condition_D) (hG : condition_G) : 
  ∃ V : ℝ, V = 16 := 
sorry

end polyhedron_volume_l60_60104


namespace odd_function_value_at_2_l60_60955

variable (f : ℝ → ℝ)
variable (h_odd : ∀ x : ℝ, f (-x) = -f x)

theorem odd_function_value_at_2 : f (-2) + f (2) = 0 :=
by
  sorry

end odd_function_value_at_2_l60_60955


namespace math_problem_l60_60525

theorem math_problem
  (x y z : ℕ)
  (h1 : z = 4)
  (h2 : x + y = 7)
  (h3 : x + z = 8) :
  x + y + z = 11 := 
by
  sorry

end math_problem_l60_60525


namespace cost_price_per_meter_l60_60022

theorem cost_price_per_meter (selling_price : ℝ) (total_meters : ℕ) (profit_per_meter : ℝ)
  (h1 : selling_price = 8925)
  (h2 : total_meters = 85)
  (h3 : profit_per_meter = 5) :
  (selling_price - total_meters * profit_per_meter) / total_meters = 100 := by
  sorry

end cost_price_per_meter_l60_60022


namespace find_x_for_equation_l60_60369

theorem find_x_for_equation : ∃ x : ℝ, (1 / 2) + ((2 / 3) * x + 4) - (8 / 16) = 4.25 ↔ x = 0.375 := 
by
  sorry

end find_x_for_equation_l60_60369


namespace find_numbers_l60_60002

theorem find_numbers
  (X Y : ℕ)
  (h1 : 10 ≤ X ∧ X < 100)
  (h2 : 10 ≤ Y ∧ Y < 100)
  (h3 : X = 2 * Y)
  (h4 : ∃ a b c d, X = 10 * a + b ∧ Y = 10 * c + d ∧ (c + d = a + b) ∧ (c = a - b ∨ d = a - b)) :
  X = 34 ∧ Y = 17 :=
sorry

end find_numbers_l60_60002


namespace people_and_carriages_condition_l60_60845

-- Definitions corresponding to the conditions
def num_people_using_carriages (x : ℕ) : ℕ := 3 * (x - 2)
def num_people_sharing_carriages (x : ℕ) : ℕ := 2 * x + 9

-- The theorem statement we need to prove
theorem people_and_carriages_condition (x : ℕ) : 
  num_people_using_carriages x = num_people_sharing_carriages x ↔ 3 * (x - 2) = 2 * x + 9 :=
by sorry

end people_and_carriages_condition_l60_60845


namespace total_number_of_legs_l60_60368

def kangaroos : ℕ := 23
def goats : ℕ := 3 * kangaroos
def legs_of_kangaroo : ℕ := 2
def legs_of_goat : ℕ := 4

theorem total_number_of_legs : 
  (kangaroos * legs_of_kangaroo + goats * legs_of_goat) = 322 := by
  sorry

end total_number_of_legs_l60_60368


namespace exponential_growth_equation_l60_60310

-- Define the initial and final greening areas and the years in consideration.
def initial_area : ℝ := 1000
def final_area : ℝ := 1440
def years : ℝ := 2

-- Define the average annual growth rate.
variable (x : ℝ)

-- State the theorem about the exponential growth equation.
theorem exponential_growth_equation :
  initial_area * (1 + x) ^ years = final_area :=
sorry

end exponential_growth_equation_l60_60310


namespace range_of_y_l60_60182

theorem range_of_y (y: ℝ) (hy: y > 0) (h_eq: ⌈y⌉ * ⌊y⌋ = 72) : 8 < y ∧ y < 9 :=
by
  sorry

end range_of_y_l60_60182


namespace find_dividend_l60_60029

theorem find_dividend (q : ℕ) (d : ℕ) (r : ℕ) (D : ℕ) 
  (h_q : q = 15000)
  (h_d : d = 82675)
  (h_r : r = 57801)
  (h_D : D = 1240182801) :
  D = d * q + r := by 
  sorry

end find_dividend_l60_60029


namespace product_of_g_of_roots_l60_60557

noncomputable def f (x : ℝ) : ℝ := x^5 - 2*x^3 + x + 1
noncomputable def g (x : ℝ) : ℝ := x^3 - 3*x + 1

theorem product_of_g_of_roots (x₁ x₂ x₃ x₄ x₅ : ℝ)
  (h₁ : f x₁ = 0) (h₂ : f x₂ = 0) (h₃ : f x₃ = 0)
  (h₄ : f x₄ = 0) (h₅ : f x₅ = 0) :
  g x₁ * g x₂ * g x₃ * g x₄ * g x₅ = f (-1 + Real.sqrt 2) * f (-1 - Real.sqrt 2) :=
by
  sorry

end product_of_g_of_roots_l60_60557


namespace john_total_climb_height_l60_60019

-- Define the heights and conditions
def num_flights : ℕ := 3
def height_per_flight : ℕ := 10
def total_stairs_height : ℕ := num_flights * height_per_flight
def rope_height : ℕ := total_stairs_height / 2
def ladder_height : ℕ := rope_height + 10

-- Prove that the total height John climbed is 70 feet
theorem john_total_climb_height : 
  total_stairs_height + rope_height + ladder_height = 70 := by
  sorry

end john_total_climb_height_l60_60019


namespace exists_coeff_less_than_neg_one_l60_60982

theorem exists_coeff_less_than_neg_one 
  (P : Polynomial ℤ)
  (h1 : P.eval 1 = 0)
  (h2 : P.eval 2 = 0) :
  ∃ i, P.coeff i < -1 := sorry

end exists_coeff_less_than_neg_one_l60_60982


namespace units_digit_m_squared_plus_3_pow_m_l60_60091

def m := 2023^2 + 3^2023

theorem units_digit_m_squared_plus_3_pow_m : 
  (m^2 + 3^m) % 10 = 5 := sorry

end units_digit_m_squared_plus_3_pow_m_l60_60091


namespace vector_parallel_l60_60291

theorem vector_parallel {x : ℝ} (h : (4 / x) = (-2 / 5)) : x = -10 :=
  by
  sorry

end vector_parallel_l60_60291


namespace no_solution_condition_l60_60951

theorem no_solution_condition (b : ℝ) : (∀ x : ℝ, 4 * (3 * x - b) ≠ 3 * (4 * x + 16)) ↔ b = -12 := 
by
  sorry

end no_solution_condition_l60_60951


namespace school_cases_of_water_l60_60983

theorem school_cases_of_water (bottles_per_case bottles_used_first_game bottles_left_after_second_game bottles_used_second_game : ℕ)
  (h1 : bottles_per_case = 20)
  (h2 : bottles_used_first_game = 70)
  (h3 : bottles_left_after_second_game = 20)
  (h4 : bottles_used_second_game = 110) :
  let total_bottles_used := bottles_used_first_game + bottles_used_second_game
  let total_bottles_initial := total_bottles_used + bottles_left_after_second_game
  let number_of_cases := total_bottles_initial / bottles_per_case
  number_of_cases = 10 :=
by
  -- The proof goes here
  sorry

end school_cases_of_water_l60_60983


namespace parallelepiped_eq_l60_60597

-- Definitions of the variables and conditions
variables (a b c u v w : ℝ)

-- Prove the identity given the conditions:
theorem parallelepiped_eq :
  u * v * w = a * v * w + b * u * w + c * u * v :=
sorry

end parallelepiped_eq_l60_60597


namespace A_days_l60_60197

theorem A_days (B_days : ℕ) (total_wage A_wage : ℕ) (h_B_days : B_days = 15) (h_total_wage : total_wage = 3000) (h_A_wage : A_wage = 1800) :
  ∃ A_days : ℕ, A_days = 10 := by
  sorry

end A_days_l60_60197


namespace ratio_of_marbles_l60_60250

noncomputable def marble_ratio : ℕ :=
  let initial_marbles := 40
  let marbles_after_breakfast := initial_marbles - 3
  let marbles_after_lunch := marbles_after_breakfast - 5
  let marbles_after_moms_gift := marbles_after_lunch + 12
  let final_marbles := 54
  let marbles_given_back_by_Susie := final_marbles - marbles_after_moms_gift
  marbles_given_back_by_Susie / 5

theorem ratio_of_marbles : marble_ratio = 2 := by
  -- proof steps would go here
  sorry

end ratio_of_marbles_l60_60250


namespace percentage_of_alcohol_in_vessel_Q_l60_60543

theorem percentage_of_alcohol_in_vessel_Q
  (x : ℝ)
  (h_mix : 2.5 + 0.04 * x = 6) :
  x = 87.5 :=
by
  sorry

end percentage_of_alcohol_in_vessel_Q_l60_60543


namespace same_face_probability_correct_l60_60474

-- Define the number of sides on the dice
def sides_20 := 20
def sides_16 := 16

-- Define the number of colored sides for each dice category
def maroon_20 := 5
def teal_20 := 8
def cyan_20 := 6
def sparkly_20 := 1

def maroon_16 := 4
def teal_16 := 6
def cyan_16 := 5
def sparkly_16 := 1

-- Define the probabilities of each color matching
def prob_maroon : ℚ := (maroon_20 / sides_20) * (maroon_16 / sides_16)
def prob_teal : ℚ := (teal_20 / sides_20) * (teal_16 / sides_16)
def prob_cyan : ℚ := (cyan_20 / sides_20) * (cyan_16 / sides_16)
def prob_sparkly : ℚ := (sparkly_20 / sides_20) * (sparkly_16 / sides_16)

-- Define the total probability of same face
def prob_same_face := prob_maroon + prob_teal + prob_cyan + prob_sparkly

-- The theorem we need to prove
theorem same_face_probability_correct : 
  prob_same_face = 99 / 320 :=
by
  sorry

end same_face_probability_correct_l60_60474


namespace not_perfect_square_of_sum_300_l60_60277

theorem not_perfect_square_of_sum_300 : ¬(∃ n : ℕ, n = 10^300 - 1 ∧ (∃ m : ℕ, n = m^2)) :=
by
  sorry

end not_perfect_square_of_sum_300_l60_60277


namespace a_b_c_at_least_one_not_less_than_one_third_l60_60782

theorem a_b_c_at_least_one_not_less_than_one_third (a b c : ℝ) (h : a + b + c = 1) :
  ¬ (a < 1/3 ∧ b < 1/3 ∧ c < 1/3) :=
by
  sorry

end a_b_c_at_least_one_not_less_than_one_third_l60_60782


namespace number_of_girls_in_basketball_club_l60_60387

-- Define the number of members in the basketball club
def total_members : ℕ := 30

-- Define the number of members who attended the practice session
def attended : ℕ := 18

-- Define the unknowns: number of boys (B) and number of girls (G)
variables (B G : ℕ)

-- Define the conditions provided in the problem
def condition1 : Prop := B + G = total_members
def condition2 : Prop := B + (1 / 3) * G = attended

-- Define the theorem to prove
theorem number_of_girls_in_basketball_club (B G : ℕ) (h1 : condition1 B G) (h2 : condition2 B G) : G = 18 :=
sorry

end number_of_girls_in_basketball_club_l60_60387


namespace smallest_n_inequality_l60_60509

variable {x y z : ℝ}

theorem smallest_n_inequality :
  ∃ (n : ℕ), (∀ (x y z : ℝ), (x^2 + y^2 + z^2)^2 ≤ n * (x^4 + y^4 + z^4)) ∧
    (∀ m : ℕ, (∀ (x y z : ℝ), (x^2 + y^2 + z^2)^2 ≤ m * (x^4 + y^4 + z^4)) → n ≤ m) :=
sorry

end smallest_n_inequality_l60_60509


namespace pigs_and_dogs_more_than_sheep_l60_60494

-- Define the number of pigs and sheep
def numberOfPigs : ℕ := 42
def numberOfSheep : ℕ := 48

-- Define the number of dogs such that it is the same as the number of pigs
def numberOfDogs : ℕ := numberOfPigs

-- Define the total number of pigs and dogs
def totalPigsAndDogs : ℕ := numberOfPigs + numberOfDogs

-- State the theorem about the difference between pigs and dogs and the number of sheep
theorem pigs_and_dogs_more_than_sheep :
  totalPigsAndDogs - numberOfSheep = 36 := 
sorry

end pigs_and_dogs_more_than_sheep_l60_60494


namespace distinct_numbers_mean_inequality_l60_60698

open Nat

theorem distinct_numbers_mean_inequality (n m : ℕ) (h_n_m : m ≤ n)
  (a : Fin m → ℕ) (ha_distinct : Function.Injective a)
  (h_cond : ∀ (i j : Fin m), i ≠ j → i.val + j.val ≤ n → ∃ (k : Fin m), a i + a j = a k) :
  (1 : ℝ) / m * (Finset.univ.sum (fun i => a i)) ≥  (n + 1) / 2 :=
by
  sorry

end distinct_numbers_mean_inequality_l60_60698


namespace fully_charge_tablet_time_l60_60286

def time_to_fully_charge_smartphone := 26 -- 26 minutes to fully charge a smartphone
def total_charge_time := 66 -- 66 minutes to charge tablet fully and phone halfway
def halfway_charge_time := time_to_fully_charge_smartphone / 2 -- 13 minutes to charge phone halfway

theorem fully_charge_tablet_time : 
  ∃ T : ℕ, T + halfway_charge_time = total_charge_time ∧ T = 53 := 
by
  sorry

end fully_charge_tablet_time_l60_60286


namespace single_reduction_equivalent_l60_60410

theorem single_reduction_equivalent (P : ℝ) (P_pos : 0 < P) : 
  (P - (P - 0.30 * P)) / P = 0.70 := 
by
  -- Let's denote the original price by P, 
  -- apply first 25% and then 60% reduction 
  -- and show that it's equivalent to a single 70% reduction
  sorry

end single_reduction_equivalent_l60_60410


namespace parallel_lines_intersect_parabola_l60_60645

theorem parallel_lines_intersect_parabola {a k b c x1 x2 x3 x4 : ℝ} 
    (h₁ : x1 < x2) 
    (h₂ : x3 < x4) 
    (intersect1 : ∀ y : ℝ, y = k * x1 + b ∧ y = a * x1^2 ∧ y = k * x2 + b ∧ y = a * x2^2) 
    (intersect2 : ∀ y : ℝ, y = k * x3 + c ∧ y = a * x3^2 ∧ y = k * x4 + c ∧ y = a * x4^2) :
    (x3 - x1) = (x2 - x4) := 
by 
    sorry

end parallel_lines_intersect_parabola_l60_60645


namespace smallest_N_l60_60075

theorem smallest_N (N : ℕ) : 
  (N = 484) ∧ 
  (∃ k : ℕ, 484 = 4 * k) ∧
  (∃ k : ℕ, 485 = 25 * k) ∧
  (∃ k : ℕ, 486 = 9 * k) ∧
  (∃ k : ℕ, 487 = 121 * k) :=
by
  -- Proof omitted (replaced by sorry)
  sorry

end smallest_N_l60_60075


namespace altered_solution_ratio_l60_60324

theorem altered_solution_ratio (initial_bleach : ℕ) (initial_detergent : ℕ) (initial_water : ℕ) :
  initial_bleach / initial_detergent = 2 / 25 ∧
  initial_detergent / initial_water = 25 / 100 →
  (initial_detergent / initial_water) / 2 = 1 / 8 →
  initial_water = 300 →
  (300 / 8) = 37.5 := 
by 
  sorry

end altered_solution_ratio_l60_60324


namespace solve_inequality_l60_60663

-- Definitions based on conditions
def f (x a : ℝ) : ℝ := (x - 2) * (a * x + 2 * a)

-- Theorem Statement
theorem solve_inequality (f_even : ∀ x a, f x a = f (-x) a) (f_inc : ∀ x y a, 0 < x → x < y → f x a ≤ f y a) :
    ∀ a > 0, { x : ℝ | f (2 - x) a > 0 } = { x | x < 0 ∨ x > 4 } :=
by
  -- Sorry to skip the proof
  sorry

end solve_inequality_l60_60663


namespace game_points_product_l60_60495

def g (n : ℕ) : ℕ :=
  if n % 3 = 0 then 12
  else if n % 2 = 0 then 3
  else 0

def allie_rolls : List ℕ := [5, 4, 1, 2, 6]
def betty_rolls : List ℕ := [6, 3, 3, 2, 1]

def calculate_points (rolls : List ℕ) : ℕ :=
  rolls.map g |>.sum

theorem game_points_product :
  calculate_points allie_rolls * calculate_points betty_rolls = 702 :=
by
  sorry

end game_points_product_l60_60495


namespace problem_l60_60240

noncomputable def k : ℝ := 2.9

theorem problem (k : ℝ) (hₖ : k > 1) 
    (h_sum : ∑' n, (7 * n + 2) / k^n = 20 / 3) : 
    k = 2.9 := 
sorry

end problem_l60_60240


namespace brownies_shared_l60_60853

theorem brownies_shared
  (total_brownies : ℕ)
  (tina_brownies : ℕ)
  (husband_brownies : ℕ)
  (remaining_brownies : ℕ)
  (shared_brownies : ℕ)
  (h1 : total_brownies = 24)
  (h2 : tina_brownies = 10)
  (h3 : husband_brownies = 5)
  (h4 : remaining_brownies = 5) :
  shared_brownies = total_brownies - (tina_brownies + husband_brownies + remaining_brownies) → shared_brownies = 4 :=
by
  sorry

end brownies_shared_l60_60853


namespace factorize_1_factorize_2_factorize_3_l60_60875

-- Problem 1: Factorize 3a^3 - 6a^2 + 3a
theorem factorize_1 (a : ℝ) : 3 * a^3 - 6 * a^2 + 3 * a = 3 * a * (a - 1)^2 :=
sorry

-- Problem 2: Factorize a^2(x - y) + b^2(y - x)
theorem factorize_2 (a b x y : ℝ) : a^2 * (x - y) + b^2 * (y - x) = (x - y) * (a - b) * (a + b) :=
sorry

-- Problem 3: Factorize 16(a + b)^2 - 9(a - b)^2
theorem factorize_3 (a b : ℝ) : 16 * (a + b)^2 - 9 * (a - b)^2 = (a + 7 * b) * (7 * a + b) :=
sorry

end factorize_1_factorize_2_factorize_3_l60_60875


namespace large_block_volume_l60_60157

theorem large_block_volume (W D L : ℝ) (h : W * D * L = 4) :
    (2 * W) * (2 * D) * (2 * L) = 32 :=
by
  sorry

end large_block_volume_l60_60157
