import Mathlib

namespace total_job_applications_l303_30311

theorem total_job_applications (apps_in_state : ℕ) (apps_other_states : ℕ) 
  (h1 : apps_in_state = 200)
  (h2 : apps_other_states = 2 * apps_in_state) :
  apps_in_state + apps_other_states = 600 :=
by
  sorry

end total_job_applications_l303_30311


namespace find_positive_integer_M_l303_30392

theorem find_positive_integer_M (M : ℕ) (h : 36^2 * 81^2 = 18^2 * M^2) : M = 162 := by
  sorry

end find_positive_integer_M_l303_30392


namespace like_terms_exponent_l303_30332

theorem like_terms_exponent (x y : ℝ) (n : ℕ) : 
  (∀ (a b : ℝ), a * x ^ 3 * y ^ (n - 1) = b * x ^ 3 * y ^ 1 → n = 2) :=
by
  sorry

end like_terms_exponent_l303_30332


namespace reduced_price_per_dozen_l303_30338

variables {P R : ℝ}

theorem reduced_price_per_dozen
  (H1 : R = 0.6 * P)
  (H2 : 40 / P - 40 / R = 64) :
  R = 3 := 
sorry

end reduced_price_per_dozen_l303_30338


namespace hcf_of_two_numbers_l303_30312

theorem hcf_of_two_numbers (H : ℕ) 
(lcm_def : lcm a b = H * 13 * 14) 
(h : a = 280 ∨ b = 280) 
(is_factor_h : H ∣ 280) : 
H = 5 :=
sorry

end hcf_of_two_numbers_l303_30312


namespace angle_B_is_180_l303_30361

variables {l k : Line} {A B C: Point}

def parallel (l k : Line) : Prop := sorry 
def angle (A B C : Point) : ℝ := sorry

theorem angle_B_is_180 (h1 : parallel l k) (h2 : angle A = 110) (h3 : angle C = 70) :
  angle B = 180 := 
by
  sorry

end angle_B_is_180_l303_30361


namespace arcsin_one_half_eq_pi_six_l303_30364

theorem arcsin_one_half_eq_pi_six : Real.arcsin (1 / 2) = Real.pi / 6 :=
by
  sorry -- Proof omitted

end arcsin_one_half_eq_pi_six_l303_30364


namespace woman_worked_days_l303_30369

-- Define variables and conditions
variables (W I : ℕ)

-- Conditions
def total_days : Prop := W + I = 25
def net_earnings : Prop := 20 * W - 5 * I = 450

-- Main theorem statement
theorem woman_worked_days (h1 : total_days W I) (h2 : net_earnings W I) : W = 23 :=
sorry

end woman_worked_days_l303_30369


namespace complex_number_solution_l303_30334

def imaginary_unit : ℂ := Complex.I -- defining the imaginary unit

theorem complex_number_solution (z : ℂ) (h : z / (z - imaginary_unit) = imaginary_unit) :
  z = (1 / 2 : ℂ) + (1 / 2 : ℂ) * imaginary_unit :=
sorry

end complex_number_solution_l303_30334


namespace count_multiples_l303_30337

theorem count_multiples (n : ℕ) : 
  n = 1 ↔ ∃ k : ℕ, k < 500 ∧ k > 0 ∧ k % 4 = 0 ∧ k % 5 = 0 ∧ k % 6 = 0 ∧ k % 7 = 0 :=
by
  sorry

end count_multiples_l303_30337


namespace S10_value_l303_30319

def sequence_sum (n : ℕ) : ℕ :=
  (2^(n+1)) - 2 - n

theorem S10_value : sequence_sum 10 = 2036 := by
  sorry

end S10_value_l303_30319


namespace number_of_comedies_rented_l303_30336

noncomputable def comedies_rented (r : ℕ) (a : ℕ) : ℕ := 3 * a

theorem number_of_comedies_rented (a : ℕ) (h : a = 5) : comedies_rented 3 a = 15 := by
  rw [h]
  exact rfl

end number_of_comedies_rented_l303_30336


namespace cubic_diff_l303_30329

theorem cubic_diff (a b : ℝ) (h1 : a - b = 4) (h2 : a^2 + b^2 = 40) : a^3 - b^3 = 208 :=
by
  sorry

end cubic_diff_l303_30329


namespace min_a1_value_l303_30331

theorem min_a1_value (a : ℕ → ℝ) :
  (∀ n > 1, a n = 9 * a (n-1) - 2 * n) →
  (∀ n, a n > 0) →
  (∀ x, (∀ n > 1, a n = 9 * a (n-1) - 2 * n) → (∀ n, a n > 0) → x ≥ a 1) →
  a 1 = 499.25 / 648 :=
sorry

end min_a1_value_l303_30331


namespace kyoko_payment_l303_30367

noncomputable def total_cost (balls skipropes frisbees : ℕ) (ball_cost rope_cost frisbee_cost : ℝ) : ℝ :=
  (balls * ball_cost) + (skipropes * rope_cost) + (frisbees * frisbee_cost)

noncomputable def final_amount (total_cost discount_rate : ℝ) : ℝ :=
  total_cost - (discount_rate * total_cost)

theorem kyoko_payment :
  let balls := 3
  let skipropes := 2
  let frisbees := 4
  let ball_cost := 1.54
  let rope_cost := 3.78
  let frisbee_cost := 2.63
  let discount_rate := 0.07
  final_amount (total_cost balls skipropes frisbees ball_cost rope_cost frisbee_cost) discount_rate = 21.11 :=
by
  sorry

end kyoko_payment_l303_30367


namespace sin_330_eq_neg_one_half_l303_30322

theorem sin_330_eq_neg_one_half : Real.sin (330 * Real.pi / 180) = -1 / 2 := by
  sorry

end sin_330_eq_neg_one_half_l303_30322


namespace dice_product_divisibility_probability_l303_30395

theorem dice_product_divisibility_probability :
  let p := 1 - ((5 / 18)^6 : ℚ)
  p = (33996599 / 34012224 : ℚ) :=
by
  -- This is the condition where the probability p is computed as the complementary probability.
  sorry

end dice_product_divisibility_probability_l303_30395


namespace train_length_l303_30303

theorem train_length (time : ℝ) (speed_in_kmph : ℝ) (speed_in_mps : ℝ) (length_of_train : ℝ) :
  (time = 6) →
  (speed_in_kmph = 96) →
  (speed_in_mps = speed_in_kmph * (5 / 18)) →
  length_of_train = speed_in_mps * time →
  length_of_train = 480 := by
  sorry

end train_length_l303_30303


namespace angle_sum_in_hexagon_l303_30324

theorem angle_sum_in_hexagon (P Q R s t : ℝ) 
    (hP: P = 40) (hQ: Q = 88) (hR: R = 30)
    (hex_sum: 6 * 180 - 720 = 0): 
    s + t = 312 :=
by
  have hex_interior_sum: 6 * 180 - 720 = 0 := hex_sum
  sorry

end angle_sum_in_hexagon_l303_30324


namespace most_stable_scores_l303_30320

structure StudentScores :=
  (average : ℝ)
  (variance : ℝ)

def studentA : StudentScores := { average := 132, variance := 38 }
def studentB : StudentScores := { average := 132, variance := 10 }
def studentC : StudentScores := { average := 132, variance := 26 }

theorem most_stable_scores :
  studentB.variance < studentA.variance ∧ studentB.variance < studentC.variance :=
by 
  sorry

end most_stable_scores_l303_30320


namespace tunnel_digging_duration_l303_30380

theorem tunnel_digging_duration (daily_progress : ℕ) (total_length_km : ℕ) 
    (meters_per_km : ℕ) (days_per_year : ℕ) : 
    daily_progress = 5 → total_length_km = 2 → meters_per_km = 1000 → days_per_year = 365 → 
    total_length_km * meters_per_km / daily_progress > 365 :=
by
  intros hprog htunnel hmeters hdays
  /- ... proof steps will go here -/
  sorry

end tunnel_digging_duration_l303_30380


namespace sequence_sum_l303_30365

theorem sequence_sum (P Q R S T U V : ℕ) (h1 : S = 7)
  (h2 : P + Q + R = 21) (h3 : Q + R + S = 21)
  (h4 : R + S + T = 21) (h5 : S + T + U = 21)
  (h6 : T + U + V = 21) : P + V = 14 :=
by
  sorry

end sequence_sum_l303_30365


namespace coopers_daily_pie_count_l303_30359

-- Definitions of conditions
def total_pies_made_per_day (x : ℕ) : ℕ := x
def days := 12
def pies_eaten_by_ashley := 50
def remaining_pies := 34

-- Lean 4 statement of the problem to prove
theorem coopers_daily_pie_count (x : ℕ) : 
  12 * total_pies_made_per_day x - pies_eaten_by_ashley = remaining_pies → 
  x = 7 := 
by
  intro h
  -- Solution steps (not included in the theorem)
  -- Given proof follows from the Lean 4 statement
  sorry

end coopers_daily_pie_count_l303_30359


namespace worker_late_time_l303_30349

noncomputable def usual_time : ℕ := 60
noncomputable def speed_factor : ℚ := 4 / 5

theorem worker_late_time (T T_new : ℕ) (S : ℚ) :
  T = usual_time →
  T = 60 →
  T_new = (5 / 4) * T →
  T_new - T = 15 :=
by
  intros
  subst T
  sorry

end worker_late_time_l303_30349


namespace average_of_three_numbers_l303_30374

theorem average_of_three_numbers (a b c : ℝ)
  (h1 : a + (b + c) / 2 = 65)
  (h2 : b + (a + c) / 2 = 69)
  (h3 : c + (a + b) / 2 = 76) :
  (a + b + c) / 3 = 35 := 
sorry

end average_of_three_numbers_l303_30374


namespace initial_average_age_l303_30362

theorem initial_average_age (A : ℝ) (n : ℕ) (h1 : n = 9) (h2 : (n * A + 35) / (n + 1) = 17) :
  A = 15 :=
by
  sorry

end initial_average_age_l303_30362


namespace set_C_cannot_form_right_triangle_l303_30358

theorem set_C_cannot_form_right_triangle :
  ¬(5^2 + 2^2 = 5^2) :=
by
  sorry

end set_C_cannot_form_right_triangle_l303_30358


namespace possible_to_form_square_l303_30357

noncomputable def shape : Type := sorry
noncomputable def is_square (s : shape) : Prop := sorry
noncomputable def divide_into_parts (s : shape) (n : ℕ) : Prop := sorry
noncomputable def all_triangles (s : shape) : Prop := sorry

theorem possible_to_form_square (s : shape) :
  (∃ (parts : ℕ), parts ≤ 4 ∧ divide_into_parts s parts ∧ is_square s) ∧
  (∃ (parts : ℕ), parts ≤ 5 ∧ divide_into_parts s parts ∧ all_triangles s ∧ is_square s) :=
sorry

end possible_to_form_square_l303_30357


namespace product_of_integers_l303_30399

theorem product_of_integers :
  ∃ (A B C : ℤ), A + B + C = 33 ∧ C = 3 * B ∧ A = C - 23 ∧ A * B * C = 192 :=
by
  sorry

end product_of_integers_l303_30399


namespace max_value_expression_l303_30366

noncomputable def expression (x : ℝ) : ℝ := 5^x - 25^x

theorem max_value_expression : 
  (∀ x : ℝ, expression x ≤ 1/4) ∧ (∃ x : ℝ, expression x = 1/4) := 
by 
  sorry

end max_value_expression_l303_30366


namespace m_squared_minus_n_squared_plus_one_is_perfect_square_l303_30301

theorem m_squared_minus_n_squared_plus_one_is_perfect_square (m n : ℤ)
  (hm : m % 2 = 1) (hn : n % 2 = 1)
  (h : m^2 - n^2 + 1 ∣ n^2 - 1) :
  ∃ k : ℤ, k^2 = m^2 - n^2 + 1 :=
sorry

end m_squared_minus_n_squared_plus_one_is_perfect_square_l303_30301


namespace range_of_f_l303_30302

noncomputable def f (x : ℝ) : ℝ := 2^(2*x) + 2^(x+1) + 3

theorem range_of_f : ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ y > 3 :=
by
  sorry

end range_of_f_l303_30302


namespace speed_of_water_current_l303_30305

theorem speed_of_water_current (v : ℝ) 
  (swimmer_speed_still_water : ℝ := 4) 
  (distance : ℝ := 3) 
  (time : ℝ := 1.5)
  (effective_speed_against_current : ℝ := swimmer_speed_still_water - v) :
  effective_speed_against_current = distance / time → v = 2 := 
by
  -- Proof
  sorry

end speed_of_water_current_l303_30305


namespace function_monotonically_increasing_on_interval_l303_30321

noncomputable def f (x : ℝ) : ℝ := x / Real.log x

theorem function_monotonically_increasing_on_interval (e : ℝ) (h_e_pos : 0 < e) (h_ln_e_pos : 0 < Real.log e) :
  ∀ x : ℝ, e < x → 0 < Real.log x - 1 := 
sorry

end function_monotonically_increasing_on_interval_l303_30321


namespace problem_solution_l303_30391

def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

def multiplicative_inverse (a m : ℕ) (inv : ℕ) : Prop := 
  (a * inv) % m = 1

theorem problem_solution :
  is_right_triangle 60 144 156 ∧ multiplicative_inverse 300 3751 3618 :=
by
  sorry

end problem_solution_l303_30391


namespace james_jail_time_l303_30310

-- Definitions based on the conditions
def arson_sentence := 6
def arson_count := 2
def total_arson_sentence := arson_sentence * arson_count

def explosives_sentence := 2 * total_arson_sentence
def terrorism_sentence := 20

-- Total sentence calculation
def total_jail_time := total_arson_sentence + explosives_sentence + terrorism_sentence

-- The theorem we want to prove
theorem james_jail_time : total_jail_time = 56 := by
  sorry

end james_jail_time_l303_30310


namespace min_distance_eq_3_l303_30318

noncomputable def f (x : ℝ) : ℝ := 4 * Real.sin (Real.pi / 3 * x + Real.pi / 4)

theorem min_distance_eq_3 (x₁ x₂ : ℝ) 
  (h₁ : f x₁ ≤ f x) (h₂ : f x ≤ f x₂) 
  (x : ℝ) :
  |x₁ - x₂| = 3 :=
by
  -- Sorry placeholder for proof.
  sorry

end min_distance_eq_3_l303_30318


namespace number_of_new_players_l303_30370

-- Definitions based on conditions
def total_groups : Nat := 2
def players_per_group : Nat := 5
def returning_players : Nat := 6

-- Convert conditions to definition
def total_players : Nat := total_groups * players_per_group

-- Define what we want to prove
def new_players : Nat := total_players - returning_players

-- The proof problem statement
theorem number_of_new_players :
  new_players = 4 :=
by
  sorry

end number_of_new_players_l303_30370


namespace smallest_solution_l303_30344

theorem smallest_solution (x : ℝ) : (1 / (x - 3) + 1 / (x - 5) = 5 / (x - 4)) → x = 4 - (Real.sqrt 15) / 3 :=
by
  sorry

end smallest_solution_l303_30344


namespace min_cans_needed_l303_30376

theorem min_cans_needed (oz_per_can : ℕ) (total_oz_needed : ℕ) (H1 : oz_per_can = 15) (H2 : total_oz_needed = 150) :
  ∃ n : ℕ, 15 * n ≥ 150 ∧ ∀ m : ℕ, 15 * m ≥ 150 → n ≤ m :=
by
  sorry

end min_cans_needed_l303_30376


namespace total_wheels_in_both_garages_l303_30390

/-- Each cycle type has a different number of wheels. --/
def wheels_per_cycle (cycle_type: String) : ℕ :=
  if cycle_type = "bicycle" then 2
  else if cycle_type = "tricycle" then 3
  else if cycle_type = "unicycle" then 1
  else if cycle_type = "quadracycle" then 4
  else 0

/-- Define the counts of each type of cycle in each garage. --/
def garage1_counts := [("bicycle", 5), ("tricycle", 6), ("unicycle", 9), ("quadracycle", 3)]
def garage2_counts := [("bicycle", 2), ("tricycle", 1), ("unicycle", 3), ("quadracycle", 4)]

/-- Total steps for the calculation --/
def wheels_in_garage (garage_counts: List (String × ℕ)) (missing_wheels_unicycles: ℕ) : ℕ :=
  List.foldl (λ acc (cycle_count: String × ℕ) => 
              acc + (if cycle_count.1 = "unicycle" then (cycle_count.2 * wheels_per_cycle cycle_count.1 - missing_wheels_unicycles) 
                     else (cycle_count.2 * wheels_per_cycle cycle_count.1))) 0 garage_counts

/-- The total number of wheels in both garages. --/
def total_wheels : ℕ := wheels_in_garage garage1_counts 0 + wheels_in_garage garage2_counts 3

/-- Prove that the total number of wheels in both garages is 72. --/
theorem total_wheels_in_both_garages : total_wheels = 72 :=
  by sorry

end total_wheels_in_both_garages_l303_30390


namespace compare_magnitudes_l303_30306

noncomputable
def f (x : ℝ) : ℝ := Real.cos (Real.cos x)

noncomputable
def g (x : ℝ) : ℝ := Real.sin (Real.sin x)

theorem compare_magnitudes : ∀ x : ℝ, f x > g x :=
by
  sorry

end compare_magnitudes_l303_30306


namespace scale_model_height_l303_30351

theorem scale_model_height :
  let scale_ratio : ℚ := 1 / 25
  let actual_height : ℚ := 151
  let model_height : ℚ := actual_height * scale_ratio
  round model_height = 6 :=
by
  sorry

end scale_model_height_l303_30351


namespace pens_exceed_500_on_saturday_l303_30307

theorem pens_exceed_500_on_saturday :
  ∃ k : ℕ, (5 * 3 ^ k > 500) ∧ k = 6 :=
by 
  sorry   -- Skipping the actual proof here

end pens_exceed_500_on_saturday_l303_30307


namespace geometric_sequence_ratio_l303_30377

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) (h_q : q = -1 / 2) :
  (a 1 + a 3 + a 5) / (a 2 + a 4 + a 6) = -2 :=
sorry

end geometric_sequence_ratio_l303_30377


namespace reciprocal_of_sum_l303_30378

theorem reciprocal_of_sum : (1 / (1 / 3 + 1 / 4)) = 12 / 7 := 
by sorry

end reciprocal_of_sum_l303_30378


namespace largest_package_markers_l303_30356

def Alex_markers : ℕ := 36
def Becca_markers : ℕ := 45
def Charlie_markers : ℕ := 60

theorem largest_package_markers (d : ℕ) :
  d ∣ Alex_markers ∧ d ∣ Becca_markers ∧ d ∣ Charlie_markers → d ≤ 3 :=
by
  sorry

end largest_package_markers_l303_30356


namespace find_certain_number_l303_30345

theorem find_certain_number (x : ℝ) (h : 25 * x = 675) : x = 27 :=
by {
  sorry
}

end find_certain_number_l303_30345


namespace quadratic_roots_range_l303_30389

theorem quadratic_roots_range (a : ℝ) :
  (a-1) * x^2 - 2*x + 1 = 0 → (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (a-1) * x1^2 - 2*x1 + 1 = 0 ∧ (a-1) * x2^2 - 2*x2 + 1 = 0) → (a < 2 ∧ a ≠ 1) :=
sorry

end quadratic_roots_range_l303_30389


namespace sqrt_recursive_value_l303_30360

noncomputable def recursive_sqrt (x : ℝ) : ℝ := Real.sqrt (3 - x)

theorem sqrt_recursive_value : 
  ∃ x : ℝ, (x = recursive_sqrt x) ∧ x = ( -1 + Real.sqrt 13 ) / 2 :=
by 
  -- ∃ x, solution assertion to define the value of x 
  use ( -1 + Real.sqrt 13 ) / 2
  sorry 

end sqrt_recursive_value_l303_30360


namespace remainder_division_l303_30397
-- Import the necessary library

-- Define the number and the divisor
def number : ℕ := 2345678901
def divisor : ℕ := 101

-- State the theorem
theorem remainder_division : number % divisor = 23 :=
by sorry

end remainder_division_l303_30397


namespace triangle_is_isosceles_range_of_expression_l303_30315

variable {a b c A B C : ℝ}
variable (triangle_ABC : 0 < A ∧ A < π ∧ 0 < B ∧ B < π)
variable (opposite_sides : a = 1 ∧ b = 1 ∧ c = 1)
variable (cos_condition : a * Real.cos B = b * Real.cos A)

theorem triangle_is_isosceles (h : a * Real.cos B = b * Real.cos A) : A = B := sorry

theorem range_of_expression 
  (h1 : 0 < A ∧ A < π/2) 
  (h2 : a * Real.cos B = b * Real.cos A) : 
  -3/2 < Real.sin (2 * A + π / 6) - 2 * Real.cos B ^ 2 ∧ Real.sin (2 * A + π / 6) - 2 * Real.cos B ^ 2 < 0 := 
sorry

end triangle_is_isosceles_range_of_expression_l303_30315


namespace password_probability_l303_30330

theorem password_probability 
  (password : Fin 6 → Fin 10) 
  (attempts : ℕ) 
  (correct_digit : Fin 10) 
  (probability_first_try : ℚ := 1 / 10)
  (probability_second_try : ℚ := (9 / 10) * (1 / 9)) : 
  ((password 5 = correct_digit) ∧ attempts ≤ 2) →
  (probability_first_try + probability_second_try = 1 / 5) :=
sorry

end password_probability_l303_30330


namespace maximum_side_length_range_l303_30384

variable (P : ℝ)
variable (a b c : ℝ)
variable (h1 : a + b + c = P)
variable (h2 : a ≤ b)
variable (h3 : b ≤ c)
variable (h4 : a + b > c)

theorem maximum_side_length_range : 
  (P / 3) ≤ c ∧ c < (P / 2) :=
by
  sorry

end maximum_side_length_range_l303_30384


namespace xy_solution_l303_30382

theorem xy_solution (x y : ℝ) (h1 : x - y = 6) (h2 : x^3 - y^3 = 72) : x * y = -8 := by
  sorry

end xy_solution_l303_30382


namespace find_g_neg_6_l303_30373

def f (x : ℚ) : ℚ := 4 * x - 9
def g (y : ℚ) : ℚ := 3 * (y * y) + 4 * y - 2

theorem find_g_neg_6 : g (-6) = 43 / 16 := by
  sorry

end find_g_neg_6_l303_30373


namespace union_of_A_and_B_l303_30328

def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}

theorem union_of_A_and_B : A ∪ B = {x | 2 < x ∧ x < 10} :=
by
  sorry

end union_of_A_and_B_l303_30328


namespace fourth_and_fifth_suppliers_cars_equal_l303_30316

-- Define the conditions
def total_cars : ℕ := 5650000
def cars_supplier_1 : ℕ := 1000000
def cars_supplier_2 : ℕ := cars_supplier_1 + 500000
def cars_supplier_3 : ℕ := cars_supplier_1 + cars_supplier_2
def cars_distributed_first_three : ℕ := cars_supplier_1 + cars_supplier_2 + cars_supplier_3
def cars_remaining : ℕ := total_cars - cars_distributed_first_three

-- Theorem stating the question and answer
theorem fourth_and_fifth_suppliers_cars_equal 
  : (cars_remaining / 2) = 325000 := by
  sorry

end fourth_and_fifth_suppliers_cars_equal_l303_30316


namespace kaleb_can_buy_toys_l303_30325

def kaleb_initial_money : ℕ := 12
def money_spent_on_game : ℕ := 8
def money_saved : ℕ := 2
def toy_cost : ℕ := 2

theorem kaleb_can_buy_toys :
  (kaleb_initial_money - money_spent_on_game - money_saved) / toy_cost = 1 :=
by
  sorry

end kaleb_can_buy_toys_l303_30325


namespace average_people_per_row_l303_30383

theorem average_people_per_row (boys girls rows : ℕ) (h_boys : boys = 24) (h_girls : girls = 24) (h_rows : rows = 6) : 
  (boys + girls) / rows = 8 :=
by
  sorry

end average_people_per_row_l303_30383


namespace square_of_neg_three_l303_30394

theorem square_of_neg_three : (-3 : ℤ)^2 = 9 := by
  sorry

end square_of_neg_three_l303_30394


namespace sum_of_coefficients_l303_30375

theorem sum_of_coefficients (A B C : ℤ) 
  (h_factorization : ∀ x, x^3 + A * x^2 + B * x + C = (x + 2) * (x - 2) * (x - 1)) :
  A + B + C = -1 :=
by sorry

end sum_of_coefficients_l303_30375


namespace no_such_rectangle_l303_30385

theorem no_such_rectangle (a b x y : ℝ) (ha : a < b)
  (hx : x < a / 2) (hy : y < a / 2)
  (h_perimeter : 2 * (x + y) = a + b)
  (h_area : x * y = (a * b) / 2) :
  false :=
sorry

end no_such_rectangle_l303_30385


namespace problem_statement_l303_30350

-- Definitions of parallel and perpendicular predicates (should be axioms or definitions in the context)
-- For simplification, assume we have a space with lines and planes, with corresponding relations.

axiom Line : Type
axiom Plane : Type
axiom parallel : Line → Line → Prop
axiom perpendicular : Line → Plane → Prop
axiom subset : Line → Plane → Prop

-- Assume the necessary conditions: m and n are lines, a and b are planes, with given relationships.
variables (m n : Line) (a b : Plane)

-- The conditions given.
variables (m_parallel_n : parallel m n)
variables (m_perpendicular_a : perpendicular m a)

-- The proposition to prove: If m parallel n and m perpendicular to a, then n is perpendicular to a.
theorem problem_statement : perpendicular n a :=
sorry

end problem_statement_l303_30350


namespace candidate_final_score_l303_30314

/- Given conditions -/
def interview_score : ℤ := 80
def written_test_score : ℤ := 90
def interview_weight : ℤ := 3
def written_test_weight : ℤ := 2

/- Final score computation -/
noncomputable def final_score : ℤ :=
  (interview_score * interview_weight + written_test_score * written_test_weight) / (interview_weight + written_test_weight)

theorem candidate_final_score : final_score = 84 := 
by
  sorry

end candidate_final_score_l303_30314


namespace length_of_first_two_CDs_l303_30335

theorem length_of_first_two_CDs
  (x : ℝ)
  (h1 : x + x + 2 * x = 6) :
  x = 1.5 := 
sorry

end length_of_first_two_CDs_l303_30335


namespace is_condition_B_an_algorithm_l303_30348

-- Definitions of conditions A, B, C, D
def condition_A := "At home, it is generally the mother who cooks"
def condition_B := "The steps to cook rice include washing the pot, rinsing the rice, adding water, and heating"
def condition_C := "Cooking outdoors is called camping cooking"
def condition_D := "Rice is necessary for cooking"

-- Definition of being considered an algorithm
def is_algorithm (s : String) : Prop :=
  s = condition_B  -- Based on the analysis that condition_B meets the criteria of an algorithm

-- The proof statement to show that condition_B can be considered an algorithm
theorem is_condition_B_an_algorithm : is_algorithm condition_B :=
by
  sorry

end is_condition_B_an_algorithm_l303_30348


namespace intersection_M_N_l303_30340

-- Define set M
def M : Set Int := {-2, -1, 0, 1}

-- Define set N using the given condition
def N : Set Int := {n : Int | -1 <= n ∧ n <= 3}

-- State that the intersection of M and N is the set {-1, 0, 1}
theorem intersection_M_N :
  M ∩ N = {-1, 0, 1} := by
  sorry

end intersection_M_N_l303_30340


namespace factorization1_factorization2_l303_30347

-- Definitions for the first problem
def expr1 (x : ℝ) := 3 * x^2 - 12
def factorized_form1 (x : ℝ) := 3 * (x + 2) * (x - 2)

-- Theorem for the first problem
theorem factorization1 (x : ℝ) : expr1 x = factorized_form1 x :=
  sorry

-- Definitions for the second problem
def expr2 (a x y : ℝ) := a * x^2 - 4 * a * x * y + 4 * a * y^2
def factorized_form2 (a x y : ℝ) := a * (x - 2 * y) * (x - 2 * y)

-- Theorem for the second problem
theorem factorization2 (a x y : ℝ) : expr2 a x y = factorized_form2 a x y :=
  sorry

end factorization1_factorization2_l303_30347


namespace BoatsRUs_canoes_l303_30342

theorem BoatsRUs_canoes :
  let a := 6
  let r := 3
  let n := 5
  let S := a * (r^n - 1) / (r - 1)
  S = 726 := by
  -- Proof
  sorry

end BoatsRUs_canoes_l303_30342


namespace initial_percentage_of_chemical_x_l303_30341

theorem initial_percentage_of_chemical_x (P : ℝ) (h1 : 20 + 80 * P = 44) : P = 0.3 :=
by sorry

end initial_percentage_of_chemical_x_l303_30341


namespace polynomial_remainder_l303_30327

theorem polynomial_remainder (x : ℂ) : (x^1500) % (x^3 - 1) = 1 := 
sorry

end polynomial_remainder_l303_30327


namespace range_of_f_l303_30326

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^4 + 6 * x^2 + 9

-- Define the domain as [0, ∞)
def domain (x : ℝ) : Prop := x ≥ 0

-- State the theorem which asserts the range of f(x) is [9, ∞)
theorem range_of_f : ∀ y : ℝ, (∃ x : ℝ, domain x ∧ f x = y) ↔ y ≥ 9 := by
  sorry

end range_of_f_l303_30326


namespace at_least_3_defective_correct_l303_30398

/-- Number of products in batch -/
def total_products : ℕ := 50

/-- Number of defective products -/
def defective_products : ℕ := 4

/-- Number of products drawn -/
def drawn_products : ℕ := 5

/-- Number of ways to draw at least 3 defective products out of 5 -/
def num_ways_at_least_3_defective : ℕ :=
  (Nat.choose defective_products 4) * (Nat.choose (total_products - defective_products) 1) +
  (Nat.choose defective_products 3) * (Nat.choose (total_products - defective_products) 2)

theorem at_least_3_defective_correct : num_ways_at_least_3_defective = 4186 := by
  sorry

end at_least_3_defective_correct_l303_30398


namespace polynomial_evaluation_l303_30393

theorem polynomial_evaluation 
  (x : ℝ) 
  (h1 : x^2 - 3 * x - 10 = 0) 
  (h2 : x > 0) : 
  (x^4 - 3 * x^3 + 2 * x^2 + 5 * x - 7) = 318 :=
by
  sorry

end polynomial_evaluation_l303_30393


namespace smallest_integer_in_set_l303_30309

theorem smallest_integer_in_set : 
  ∀ (n : ℤ), (n + 6 < 2 * (n + 3)) → n ≥ 1 :=
by 
  sorry

end smallest_integer_in_set_l303_30309


namespace quadrilateral_segments_l303_30317

theorem quadrilateral_segments {a b c d : ℝ} (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d)
  (h5 : a + b + c + d = 2) (h6 : 1/4 < a) (h7 : a < 1) (h8 : 1/4 < b) (h9 : b < 1)
  (h10 : 1/4 < c) (h11 : c < 1) (h12 : 1/4 < d) (h13 : d < 1) : 
  (a + b > d) ∧ (a + c > d) ∧ (a + d > c) ∧ (b + c > d) ∧ 
  (b + d > c) ∧ (c + d > a) ∧ (a + b + c > d) ∧ (a + b + d > c) ∧
  (a + c + d > b) ∧ (b + c + d > a) :=
sorry

end quadrilateral_segments_l303_30317


namespace pow_15_1234_mod_19_l303_30300

theorem pow_15_1234_mod_19 : (15^1234) % 19 = 6 := 
by sorry

end pow_15_1234_mod_19_l303_30300


namespace divisibility_theorem_l303_30346

theorem divisibility_theorem (n : ℕ) (h1 : n > 0) (h2 : ¬(2 ∣ n)) (h3 : ¬(3 ∣ n)) (k : ℤ) :
  (k + 1) ^ n - k ^ n - 1 ∣ k ^ 2 + k + 1 :=
sorry

end divisibility_theorem_l303_30346


namespace concatenated_number_not_power_of_two_l303_30355

theorem concatenated_number_not_power_of_two :
  ∀ (N : ℕ), (∀ i, 11111 ≤ i ∧ i ≤ 99999) →
  (N ≡ 0 [MOD 11111]) → ¬ ∃ k, N = 2^k :=
by
  sorry

end concatenated_number_not_power_of_two_l303_30355


namespace tank_volume_ratio_l303_30304

theorem tank_volume_ratio (A B : ℝ) 
    (h : (3 / 4) * A = (5 / 8) * B) : A / B = 6 / 5 := 
by 
  sorry

end tank_volume_ratio_l303_30304


namespace number_of_students_l303_30388

theorem number_of_students (groups : ℕ) (students_per_group : ℕ) (minutes_per_student : ℕ) (minutes_per_group : ℕ) :
    groups = 3 →
    minutes_per_student = 4 →
    minutes_per_group = 24 →
    minutes_per_group = students_per_group * minutes_per_student →
    18 = groups * students_per_group :=
by
  intros h_groups h_minutes_per_student h_minutes_per_group h_relation
  sorry

end number_of_students_l303_30388


namespace container_capacity_l303_30354

-- Define the given conditions
def initially_full (x : ℝ) : Prop := (1 / 4) * x + 300 = (3 / 4) * x

-- Define the proof problem to show that the total capacity is 600 liters
theorem container_capacity : ∃ x : ℝ, initially_full x → x = 600 := sorry

end container_capacity_l303_30354


namespace total_volume_tetrahedra_l303_30387

theorem total_volume_tetrahedra (side_length : ℝ) (x : ℝ) (sqrt_2 : ℝ := Real.sqrt 2) 
  (cube_to_octa_length : x = 2 * (sqrt_2 - 1)) 
  (volume_of_one_tetra : ℝ := ((6 - 4 * sqrt_2) * (3 - sqrt_2)) / 6) :
  side_length = 2 → 
  8 * volume_of_one_tetra = (104 - 72 * sqrt_2) / 3 :=
by
  intros
  sorry

end total_volume_tetrahedra_l303_30387


namespace isosceles_right_triangle_area_l303_30308

theorem isosceles_right_triangle_area (h : ℝ) (A : ℝ) :
  (h = 5 * Real.sqrt 2) →
  (A = 12.5) →
  ∃ (leg : ℝ), (leg = 5) ∧ (A = 1 / 2 * leg^2) := by
  sorry

end isosceles_right_triangle_area_l303_30308


namespace distance_between_points_on_line_l303_30363

theorem distance_between_points_on_line (a b c d m k : ℝ) 
  (hab : b = m * a + k) (hcd : d = m * c + k) :
  dist (a, b) (c, d) = |a - c| * Real.sqrt (1 + m^2) :=
by
  sorry

end distance_between_points_on_line_l303_30363


namespace a_in_s_l303_30371

-- Defining the sets and the condition
def S : Set ℕ := {1, 2}
def T (a : ℕ) : Set ℕ := {a}

-- The Lean theorem statement
theorem a_in_s (a : ℕ) (h : S ∪ T a = S) : a = 1 ∨ a = 2 := 
by 
  sorry

end a_in_s_l303_30371


namespace vertical_angles_are_congruent_l303_30352

def supplementary_angles (a b : ℝ) : Prop := a + b = 180
def corresponding_angles (l1 l2 t : ℝ) : Prop := l1 = l2
def exterior_angle_greater (ext int1 int2 : ℝ) : Prop := ext = int1 + int2
def vertical_angles_congruent (a b : ℝ) : Prop := a = b

theorem vertical_angles_are_congruent (a b : ℝ) (h : vertical_angles_congruent a b) : a = b := by
  sorry

end vertical_angles_are_congruent_l303_30352


namespace discriminant_zero_l303_30353

theorem discriminant_zero (a b c : ℝ) (h₁ : a = 1) (h₂ : b = -2) (h₃ : c = 1) :
  (b^2 - 4 * a * c) = 0 :=
by
  sorry

end discriminant_zero_l303_30353


namespace apples_shared_equally_l303_30379

-- Definitions of the given conditions
def num_apples : ℕ := 9
def num_friends : ℕ := 3

-- Statement of the problem
theorem apples_shared_equally : num_apples / num_friends = 3 := by
  sorry

end apples_shared_equally_l303_30379


namespace find_a_in_third_quadrant_l303_30381

theorem find_a_in_third_quadrant :
  ∃ a : ℝ, a < 0 ∧ 3 * a^2 + 4 * a^2 = 28 ∧ a = -2 :=
by
  sorry

end find_a_in_third_quadrant_l303_30381


namespace final_price_after_adjustments_l303_30313

theorem final_price_after_adjustments (p : ℝ) :
  let increased_price := p * 1.30
  let discounted_price := increased_price * 0.75
  let final_price := discounted_price * 1.10
  final_price = 1.0725 * p :=
by
  sorry

end final_price_after_adjustments_l303_30313


namespace hannah_age_is_48_l303_30368

-- Define the ages of the brothers
def num_brothers : ℕ := 3
def age_each_brother : ℕ := 8

-- Define the sum of brothers' ages
def sum_brothers_ages : ℕ := num_brothers * age_each_brother

-- Define the age of Hannah
def hannah_age : ℕ := 2 * sum_brothers_ages

-- The theorem to prove Hannah's age is 48 years
theorem hannah_age_is_48 : hannah_age = 48 := by
  sorry

end hannah_age_is_48_l303_30368


namespace skittles_left_l303_30372

theorem skittles_left (initial_skittles : ℕ) (skittles_given : ℕ) (final_skittles : ℕ) :
  initial_skittles = 50 → skittles_given = 7 → final_skittles = initial_skittles - skittles_given → final_skittles = 43 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end skittles_left_l303_30372


namespace greatest_integer_x_l303_30323

theorem greatest_integer_x (x : ℤ) (h : (5 : ℚ) / 8 > (x : ℚ) / 17) : x ≤ 10 :=
by
  sorry

end greatest_integer_x_l303_30323


namespace part1_part2_l303_30339

-- Statements derived from Step c)
theorem part1 {m : ℝ} (h : ∃ x : ℝ, m - |5 - 2 * x| - |2 * x - 1| = 0) : 4 ≤ m := by
  sorry

theorem part2 {x : ℝ} (hx : |x - 3| + |x + 4| ≤ 8) : -9 / 2 ≤ x ∧ x ≤ 7 / 2 := by
  sorry

end part1_part2_l303_30339


namespace work_together_days_l303_30333

theorem work_together_days
  (a_days : ℝ) (ha : a_days = 18)
  (b_days : ℝ) (hb : b_days = 30)
  (c_days : ℝ) (hc : c_days = 45)
  (combined_days : ℝ) :
  (combined_days = 1 / ((1 / a_days) + (1 / b_days) + (1 / c_days))) → combined_days = 9 := 
by
  sorry

end work_together_days_l303_30333


namespace arun_weight_l303_30386

theorem arun_weight (W B : ℝ) (h1 : 65 < W ∧ W < 72) (h2 : B < W ∧ W < 70) (h3 : W ≤ 68) (h4 : (B + 68) / 2 = 67) : B = 66 :=
sorry

end arun_weight_l303_30386


namespace female_adults_present_l303_30343

variable (children : ℕ) (male_adults : ℕ) (total_people : ℕ)
variable (children_count : children = 80) (male_adults_count : male_adults = 60) (total_people_count : total_people = 200)

theorem female_adults_present : ∃ (female_adults : ℕ), 
  female_adults = total_people - (children + male_adults) ∧ 
  female_adults = 60 :=
by
  sorry

end female_adults_present_l303_30343


namespace sum_of_reciprocals_of_squares_l303_30396

open Real

theorem sum_of_reciprocals_of_squares {a b c : ℝ} (h1 : a + b + c = 6) (h2 : a * b + b * c + c * a = -7) (h3 : a * b * c = -2) :
  1 / a^2 + 1 / b^2 + 1 / c^2 = 73 / 4 :=
by
  sorry

end sum_of_reciprocals_of_squares_l303_30396
