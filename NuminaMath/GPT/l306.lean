import Mathlib

namespace perfect_square_value_of_b_l306_30631

theorem perfect_square_value_of_b :
  (∃ b : ℝ, (11.98 * 11.98 + 11.98 * 0.04 + b * b) = (11.98 + b)^2) →
  (∃ b : ℝ, b = 0.02) :=
sorry

end perfect_square_value_of_b_l306_30631


namespace total_amount_paid_l306_30664

-- Define the parameters
def cost_per_night_per_person : ℕ := 40
def number_of_people : ℕ := 3
def number_of_nights : ℕ := 3

-- Define the total cost calculation
def total_cost := cost_per_night_per_person * number_of_people * number_of_nights

-- The statement of the proof problem
theorem total_amount_paid :
  total_cost = 360 :=
by
  -- Placeholder for the proof
  sorry

end total_amount_paid_l306_30664


namespace ratio_square_pentagon_l306_30639

theorem ratio_square_pentagon (P_sq P_pent : ℕ) 
  (h_sq : P_sq = 60) (h_pent : P_pent = 60) :
  (P_sq / 4) / (P_pent / 5) = 5 / 4 :=
by 
  sorry

end ratio_square_pentagon_l306_30639


namespace algebraic_expression_value_l306_30641

theorem algebraic_expression_value (b a c : ℝ) (h₁ : b < a) (h₂ : a < 0) (h₃ : 0 < c) :
  |b| - |b - a| + |c - a| - |a + b| = b + c - a :=
by
  sorry

end algebraic_expression_value_l306_30641


namespace correct_fraction_order_l306_30680

noncomputable def fraction_ordering : Prop := 
  (16 / 12 < 18 / 13) ∧ (18 / 13 < 21 / 14) ∧ (21 / 14 < 20 / 15)

theorem correct_fraction_order : fraction_ordering := 
by {
  repeat { sorry }
}

end correct_fraction_order_l306_30680


namespace new_socks_bought_l306_30634

theorem new_socks_bought :
  ∀ (original_socks throw_away new_socks total_socks : ℕ),
    original_socks = 28 →
    throw_away = 4 →
    total_socks = 60 →
    total_socks = original_socks - throw_away + new_socks →
    new_socks = 36 :=
by
  intros original_socks throw_away new_socks total_socks h_original h_throw h_total h_eq
  sorry

end new_socks_bought_l306_30634


namespace find_certain_number_l306_30674

theorem find_certain_number (x : ℕ) (h : 220025 = (x + 445) * (2 * (x - 445)) + 25) : x = 555 :=
sorry

end find_certain_number_l306_30674


namespace sequence_less_than_inverse_l306_30630

-- Define the sequence and conditions given in the problem
variables {a : ℕ → ℝ}
axiom positive_sequence (n : ℕ) : 0 < a n
axiom sequence_inequality (n : ℕ) : a n ^ 2 ≤ a n - a (n + 1)

theorem sequence_less_than_inverse (n : ℕ) : a n < 1 / n := 
sorry

end sequence_less_than_inverse_l306_30630


namespace additional_distance_l306_30669

theorem additional_distance (distance_speed_10 : ℝ) (speed1 speed2 time1 time2 distance actual_distance additional_distance : ℝ)
  (h1 : actual_distance = distance_speed_10)
  (h2 : time1 = distance_speed_10 / speed1)
  (h3 : time1 = 5)
  (h4 : speed1 = 10)
  (h5 : time2 = actual_distance / speed2)
  (h6 : speed2 = 14)
  (h7 : distance = speed2 * time1)
  (h8 : distance = 70)
  : additional_distance = distance - actual_distance
  := by
  sorry

end additional_distance_l306_30669


namespace ratio_of_logs_l306_30624

theorem ratio_of_logs (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h : Real.log a / Real.log 4 = Real.log b / Real.log 18 ∧ Real.log b / Real.log 18 = Real.log (a + b) / Real.log 32) :
  b / a = (1 + Real.sqrt 5) / 2 :=
by
  sorry

end ratio_of_logs_l306_30624


namespace winning_candidate_percentage_l306_30618

def percentage_votes (votes1 votes2 votes3 : ℕ) : ℚ := 
  let total_votes := votes1 + votes2 + votes3
  let winning_votes := max (max votes1 votes2) votes3
  (winning_votes * 100) / total_votes

theorem winning_candidate_percentage :
  percentage_votes 3000 5000 15000 = (15000 * 100) / (3000 + 5000 + 15000) :=
by 
  -- This computation should give us the exact percentage fraction.
  -- Simplifying it would yield the result approximately 65.22%
  -- Proof steps can be provided here.
  sorry

end winning_candidate_percentage_l306_30618


namespace fraction_solution_l306_30647

theorem fraction_solution (a : ℤ) (h : 0 < a ∧ (a : ℚ) / (a + 36) = 775 / 1000) : a = 124 := 
by
  sorry

end fraction_solution_l306_30647


namespace farmland_acres_l306_30621

theorem farmland_acres (x y : ℝ) 
  (h1 : x + y = 100) 
  (h2 : 300 * x + (500 / 7) * y = 10000) : 
  true :=
sorry

end farmland_acres_l306_30621


namespace problem_l306_30688

-- Define the polynomial g(x) with given coefficients
def g (x : ℝ) (a : ℝ) : ℝ :=
  x^3 + a * x^2 + x + 8

-- Define the polynomial f(x) with given coefficients
def f (x : ℝ) (a b c : ℝ) : ℝ :=
  x^4 + x^3 + b * x^2 + 50 * x + c

-- Define the conditions
def conditions (a b c r : ℝ) : Prop :=
  ∃ roots : Finset ℝ, (∀ x ∈ roots, g x a = 0) ∧ (∀ x ∈ roots, f x a b c = 0) ∧ (roots.card = 3) ∧
  (8 - r = 50) ∧ (a - r = 1) ∧ (1 - a * r = b) ∧ (-8 * r = c)

-- Define the theorem to be proved
theorem problem (a b c r : ℝ) (h : conditions a b c r) : f 1 a b c = -1333 :=
by sorry

end problem_l306_30688


namespace domain_of_function_l306_30653

theorem domain_of_function :
  { x : ℝ | -2 ≤ x ∧ x < 4 } = { x : ℝ | (x + 2 ≥ 0) ∧ (4 - x > 0) } :=
by
  sorry

end domain_of_function_l306_30653


namespace m_range_l306_30682

open Real

-- Define the points
def A : ℝ × ℝ := (-1, 2)
def B : ℝ × ℝ := (2, -1)

-- Define the line equation
def line_eq (x y m : ℝ) : Prop := x - 2*y + m = 0

-- Theorem: m must belong to the interval [-4, 5]
theorem m_range (m : ℝ) : (line_eq A.1 A.2 m) → (line_eq B.1 B.2 m) → -4 ≤ m ∧ m ≤ 5 := 
sorry

end m_range_l306_30682


namespace right_triangle_exists_and_r_inscribed_circle_l306_30643

theorem right_triangle_exists_and_r_inscribed_circle (d : ℝ) (hd : d > 0) :
  ∃ (a b c : ℝ), 
    a < b ∧ 
    a^2 + b^2 = c^2 ∧
    b = a + d ∧ 
    c = b + d ∧ 
    (a + b - c) / 2 = d :=
by
  sorry

end right_triangle_exists_and_r_inscribed_circle_l306_30643


namespace football_games_per_month_l306_30649

theorem football_games_per_month :
  let total_games := 5491
  let months := 17.0
  total_games / months = 323 := 
by
  let total_games := 5491
  let months := 17.0
  -- This is where the actual computation would happen if we were to provide a proof
  sorry

end football_games_per_month_l306_30649


namespace josiah_total_expenditure_l306_30629

noncomputable def cookies_per_day := 2
noncomputable def cost_per_cookie := 16
noncomputable def days_in_march := 31

theorem josiah_total_expenditure :
  (cookies_per_day * days_in_march * cost_per_cookie) = 992 :=
by sorry

end josiah_total_expenditure_l306_30629


namespace B_2_2_eq_16_l306_30671

def B : ℕ → ℕ → ℕ
| 0, n       => n + 2
| (m+1), 0   => B m 2
| (m+1), (n+1) => B m (B (m+1) n)

theorem B_2_2_eq_16 : B 2 2 = 16 := by
  sorry

end B_2_2_eq_16_l306_30671


namespace radius_of_circle_l306_30673

-- Define the problem condition
def diameter_of_circle : ℕ := 14

-- State the problem as a theorem
theorem radius_of_circle (d : ℕ) (hd : d = diameter_of_circle) : d / 2 = 7 := by 
  sorry

end radius_of_circle_l306_30673


namespace pos_int_solutions_l306_30636

theorem pos_int_solutions (x : ℤ) : (3 * x - 4 < 2 * x) → (0 < x) → (x = 1 ∨ x = 2 ∨ x = 3) :=
by
  intro h1 h2
  have h3 : x - 4 < 0 := by sorry  -- Step derived from inequality simplification
  have h4 : x < 4 := by sorry     -- Adding 4 to both sides
  sorry                           -- Combine conditions to get the specific solutions

end pos_int_solutions_l306_30636


namespace mul_mod_eq_l306_30654

theorem mul_mod_eq :
  (66 * 77 * 88) % 25 = 16 :=
by 
  sorry

end mul_mod_eq_l306_30654


namespace car_drive_distance_l306_30650

-- Define the conditions as constants
def driving_speed : ℕ := 8 -- miles per hour
def driving_hours_before_cool : ℕ := 5 -- hours of constant driving
def cooling_hours : ℕ := 1 -- hours needed for cooling down
def total_time : ℕ := 13 -- hours available

-- Define the calculation for distance driven in cycles
def distance_per_cycle : ℕ := driving_speed * driving_hours_before_cool

-- Calculate the duration of one complete cycle
def cycle_duration : ℕ := driving_hours_before_cool + cooling_hours

-- Theorem statement: the car can drive 88 miles in 13 hours
theorem car_drive_distance : distance_per_cycle * (total_time / cycle_duration) + driving_speed * (total_time % cycle_duration) = 88 :=
by
  sorry

end car_drive_distance_l306_30650


namespace find_common_difference_l306_30686

variable {a_n : ℕ → ℕ}
variable {d : ℕ}

-- Conditions
def first_term (a_n : ℕ → ℕ) := a_n 1 = 1
def common_difference (d : ℕ) := d ≠ 0
def arithmetic_def (a_n : ℕ → ℕ) (d : ℕ) := ∀ n, a_n (n+1) = a_n n + d
def geom_mean_condition (a_n : ℕ → ℕ) := a_n 2 ^ 2 = a_n 1 * a_n 4

-- Proof statement
theorem find_common_difference
  (fa : first_term a_n)
  (cd : common_difference d)
  (ad : arithmetic_def a_n d)
  (gmc : geom_mean_condition a_n) :
  d = 1 := by
  sorry

end find_common_difference_l306_30686


namespace average_speed_l306_30612

theorem average_speed (initial final time : ℕ) (h_initial : initial = 2002) (h_final : final = 2332) (h_time : time = 11) : 
  (final - initial) / time = 30 := by
  sorry

end average_speed_l306_30612


namespace population_factor_proof_l306_30683

-- Define the conditions given in the problem
variables (N x y z : ℕ)

theorem population_factor_proof :
  (N = x^2) ∧ (N + 100 = y^2 + 1) ∧ (N + 200 = z^2) → (7 ∣ N) :=
by sorry

end population_factor_proof_l306_30683


namespace total_steps_five_days_l306_30642

def steps_monday : ℕ := 150 + 170
def steps_tuesday : ℕ := 140 + 170
def steps_wednesday : ℕ := 160 + 210 + 25
def steps_thursday : ℕ := 150 + 140 + 30 + 15
def steps_friday : ℕ := 180 + 200 + 20

theorem total_steps_five_days :
  steps_monday + steps_tuesday + steps_wednesday + steps_thursday + steps_friday = 1760 :=
by
  have h1 : steps_monday = 320 := rfl
  have h2 : steps_tuesday = 310 := rfl
  have h3 : steps_wednesday = 395 := rfl
  have h4 : steps_thursday = 335 := rfl
  have h5 : steps_friday = 400 := rfl
  show 320 + 310 + 395 + 335 + 400 = 1760
  sorry

end total_steps_five_days_l306_30642


namespace quadratic_inequality_solution_set_l306_30625

theorem quadratic_inequality_solution_set {x : ℝ} :
  (x^2 + x - 2 ≤ 0) ↔ (-2 ≤ x ∧ x ≤ 1) :=
by
  sorry

end quadratic_inequality_solution_set_l306_30625


namespace sum_of_remainders_l306_30659

theorem sum_of_remainders {a b c d e : ℤ} (h1 : a % 13 = 3) (h2 : b % 13 = 5) (h3 : c % 13 = 7) (h4 : d % 13 = 9) (h5 : e % 13 = 11) : 
  ((a + b + c + d + e) % 13) = 9 :=
by
  sorry

end sum_of_remainders_l306_30659


namespace machine_minutes_worked_l306_30691

-- Definitions based on conditions
def shirts_made_yesterday : ℕ := 9
def shirts_per_minute : ℕ := 3

-- The proof problem statement
theorem machine_minutes_worked (shirts_made_yesterday shirts_per_minute : ℕ) : 
  shirts_made_yesterday / shirts_per_minute = 3 := 
by
  sorry

end machine_minutes_worked_l306_30691


namespace stratified_sampling_correct_l306_30646

-- Define the total number of students and the ratio of students in grades 10, 11, and 12
def total_students : ℕ := 4000
def ratio_grade10 : ℕ := 32
def ratio_grade11 : ℕ := 33
def ratio_grade12 : ℕ := 35

-- The total sample size
def sample_size : ℕ := 200

-- Define the expected numbers of students drawn from each grade in the sample
def sample_grade10 : ℕ := 64
def sample_grade11 : ℕ := 66
def sample_grade12 : ℕ := 70

-- The theorem to be proved
theorem stratified_sampling_correct :
  (sample_grade10 + sample_grade11 + sample_grade12 = sample_size) ∧
  (sample_grade10 = (ratio_grade10 * sample_size) / (ratio_grade10 + ratio_grade11 + ratio_grade12)) ∧
  (sample_grade11 = (ratio_grade11 * sample_size) / (ratio_grade10 + ratio_grade11 + ratio_grade12)) ∧
  (sample_grade12 = (ratio_grade12 * sample_size) / (ratio_grade10 + ratio_grade11 + ratio_grade12)) :=
by
  sorry

end stratified_sampling_correct_l306_30646


namespace columbus_discovered_america_in_1492_l306_30609

theorem columbus_discovered_america_in_1492 :
  ∃ (x y z : ℕ), x ≠ y ∧ x ≠ z ∧ y ≠ z ∧ x ≠ 1 ∧ y ≠ 1 ∧ z ≠ 1 ∧
  1 + x + y + z = 16 ∧ y + 1 = 5 * z ∧
  1000 + 100 * x + 10 * y + z = 1492 :=
by
  sorry

end columbus_discovered_america_in_1492_l306_30609


namespace weight_of_daughter_l306_30652

variable (M D G S : ℝ)

theorem weight_of_daughter :
  M + D + G + S = 200 →
  D + G = 60 →
  G = M / 5 →
  S = 2 * D →
  D = 800 / 15 :=
by
  intros h1 h2 h3 h4
  sorry

end weight_of_daughter_l306_30652


namespace max_sum_arithmetic_sequence_l306_30687

theorem max_sum_arithmetic_sequence (a : ℝ) (d : ℝ) (n : ℕ) (S : ℕ → ℝ) (h1 : (a + 2) ^ 2 = (a + 8) * (a - 2))
  (h2 : ∀ k, S k = (k * (2 * a + (k - 1) * d)) / 2)
  (h3 : 10 = a) (h4 : -2 = d) :
  S 10 = 90 :=
sorry

end max_sum_arithmetic_sequence_l306_30687


namespace find_starting_number_of_range_l306_30685

theorem find_starting_number_of_range : 
  ∃ (n : ℤ), 
    (∀ k, (0 ≤ k ∧ k < 7) → (n + k * 3 ≤ 31 ∧ n + k * 3 % 3 = 0)) ∧ 
    n + 6 * 3 = 30 - 6 * 3 :=
by
  sorry

end find_starting_number_of_range_l306_30685


namespace probability_getting_wet_l306_30698

theorem probability_getting_wet 
  (P_R : ℝ := 1/2)
  (P_notT : ℝ := 1/2)
  (h1 : 0 ≤ P_R ∧ P_R ≤ 1)
  (h2 : 0 ≤ P_notT ∧ P_notT ≤ 1) 
  : P_R * P_notT = 1/4 := 
by
  -- Proof that the probability of getting wet equals 1/4
  sorry

end probability_getting_wet_l306_30698


namespace find_symmetric_point_l306_30640

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def plane (x y z : ℝ) : ℝ := 
  4 * x + 6 * y + 4 * z - 25

def symmetric_point (M M_prime : Point3D) (plane_eq : ℝ → ℝ → ℝ → ℝ) : Prop :=
  let t : ℝ := (1 / 4)
  let M0 : Point3D := { x := (1 + 4 * t), y := (6 * t), z := (1 + 4 * t) }
  let midpoint_x := (M.x + M_prime.x) / 2
  let midpoint_y := (M.y + M_prime.y) / 2
  let midpoint_z := (M.z + M_prime.z) / 2
  M0.x = midpoint_x ∧ M0.y = midpoint_y ∧ M0.z = midpoint_z ∧
  plane_eq M0.x M0.y M0.z = 0

def M : Point3D := { x := 1, y := 0, z := 1 }

def M_prime : Point3D := { x := 3, y := 3, z := 3 }

theorem find_symmetric_point : symmetric_point M M_prime plane := by
  -- the proof is omitted here
  sorry

end find_symmetric_point_l306_30640


namespace eval_expr_eq_zero_l306_30660

def ceiling_floor_sum (x : ℚ) : ℤ :=
  Int.ceil (x) + Int.floor (-x)

theorem eval_expr_eq_zero : ceiling_floor_sum (7/3) = 0 := by
  sorry

end eval_expr_eq_zero_l306_30660


namespace quadratic_has_two_roots_l306_30681

theorem quadratic_has_two_roots (a b c : ℝ) (h₁ : a ≠ 0) (h₂ : 5 * a + b + 2 * c = 0) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0 := 
  sorry

end quadratic_has_two_roots_l306_30681


namespace coin_landing_heads_prob_l306_30696

theorem coin_landing_heads_prob (p : ℝ) (h : p^2 * (1 - p)^3 = 0.03125) : p = 0.5 :=
by
sorry

end coin_landing_heads_prob_l306_30696


namespace minimum_value_of_a_l306_30684

theorem minimum_value_of_a (a : ℝ) : 
  (∀ x : ℝ, 0 < x ∧ x ≤ 1 / 2 → x^2 + a * x + 1 ≥ 0) → a ≥ -5 / 2 :=
sorry

end minimum_value_of_a_l306_30684


namespace tim_total_payment_correct_l306_30694

-- Define the conditions stated in the problem
def doc_visit_cost : ℝ := 300
def insurance_coverage_percent : ℝ := 0.75
def cat_visit_cost : ℝ := 120
def pet_insurance_coverage : ℝ := 60

-- Define the amounts covered by insurance 
def insurance_coverage_amount : ℝ := doc_visit_cost * insurance_coverage_percent
def tim_payment_for_doc_visit : ℝ := doc_visit_cost - insurance_coverage_amount
def tim_payment_for_cat_visit : ℝ := cat_visit_cost - pet_insurance_coverage

-- Define the total payment Tim needs to make
def tim_total_payment : ℝ := tim_payment_for_doc_visit + tim_payment_for_cat_visit

-- State the main theorem
theorem tim_total_payment_correct : tim_total_payment = 135 := by
  sorry

end tim_total_payment_correct_l306_30694


namespace arman_age_in_years_l306_30600

theorem arman_age_in_years (A S y : ℕ) (h1: A = 6 * S) (h2: S = 2 + 4) (h3: A + y = 40) : y = 4 :=
sorry

end arman_age_in_years_l306_30600


namespace find_m_f_monotonicity_l306_30645

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := 2 / x - x ^ m

theorem find_m : ∃ (m : ℝ), f 4 m = -7 / 2 := sorry

noncomputable def g (x : ℝ) : ℝ := 2 / x - x

theorem f_monotonicity : ∀ x1 x2 : ℝ, (0 < x2 ∧ x2 < x1) → f x1 1 < f x2 1 := sorry

end find_m_f_monotonicity_l306_30645


namespace tangent_line_equation_l306_30614

/-- Prove that the equation of the tangent line to the curve y = x^3 - 4x^2 + 4 at the point (1,1) is y = -5x + 6 -/
theorem tangent_line_equation (x y : ℝ)
  (h_curve : y = x^3 - 4 * x^2 + 4)
  (h_point : x = 1 ∧ y = 1) :
  y = -5 * x + 6 := by
  sorry

end tangent_line_equation_l306_30614


namespace tiles_needed_to_cover_floor_l306_30601

/-- 
A floor 10 feet by 15 feet is to be tiled with 3-inch-by-9-inch tiles. 
This theorem verifies that the necessary number of tiles is 800. 
-/
theorem tiles_needed_to_cover_floor
  (floor_length : ℝ)
  (floor_width : ℝ)
  (tile_length_inch : ℝ)
  (tile_width_inch : ℝ)
  (conversion_factor : ℝ)
  (num_tiles : ℕ) 
  (h_floor_length : floor_length = 10)
  (h_floor_width : floor_width = 15)
  (h_tile_length_inch : tile_length_inch = 3)
  (h_tile_width_inch : tile_width_inch = 9)
  (h_conversion_factor : conversion_factor = 12)
  (h_num_tiles : num_tiles = 800) :
  (floor_length * floor_width) / ((tile_length_inch / conversion_factor) * (tile_width_inch / conversion_factor)) = num_tiles :=
by
  -- The proof is not included, using sorry to mark this part
  sorry

end tiles_needed_to_cover_floor_l306_30601


namespace any_power_ends_in_12890625_l306_30690

theorem any_power_ends_in_12890625 (a : ℕ) (m k : ℕ) (h : a = 10^m * k + 12890625) : ∀ (n : ℕ), 0 < n → ((a ^ n) % 10^8 = 12890625 % 10^8) :=
by
  intros
  sorry

end any_power_ends_in_12890625_l306_30690


namespace ladder_distance_from_wall_l306_30608

theorem ladder_distance_from_wall (θ : ℝ) (L : ℝ) (d : ℝ) 
  (h_angle : θ = 60) (h_length : L = 19) (h_cos : Real.cos (θ * Real.pi / 180) = 0.5) : 
  d = 9.5 :=
by
  sorry

end ladder_distance_from_wall_l306_30608


namespace vertical_asymptote_at_neg_two_over_three_l306_30666

theorem vertical_asymptote_at_neg_two_over_three : 
  ∃ x : ℝ, 6 * x + 4 = 0 ∧ x = -2 / 3 := 
by
  use -2 / 3
  sorry

end vertical_asymptote_at_neg_two_over_three_l306_30666


namespace tangent_parallel_line_l306_30699

open Function

def f (x : ℝ) : ℝ := x^4 - x

def f' (x : ℝ) : ℝ := 4 * x^3 - 1

theorem tangent_parallel_line {P : ℝ × ℝ} (hP : ∃ x y, P = (x, y) ∧ f' x = 3) :
  P = (1, 0) := by
  sorry

end tangent_parallel_line_l306_30699


namespace intervals_of_monotonicity_l306_30670

noncomputable def y (x : ℝ) : ℝ := 2 ^ (x^2 - 2*x + 4)

theorem intervals_of_monotonicity :
  (∀ x : ℝ, x > 1 → (∀ y₁ y₂ : ℝ, x₁ < x₂ → y x₁ < y x₂)) ∧
  (∀ x : ℝ, x < 1 → (∀ y₁ y₂ : ℝ, x₁ < x₂ → y x₁ > y x₂)) :=
by
  sorry

end intervals_of_monotonicity_l306_30670


namespace polygonal_line_exists_l306_30628

theorem polygonal_line_exists (A : Type) (n q : ℕ) (lengths : Fin q → ℝ)
  (yellow_segments : Fin q → (A × A))
  (h_lengths : ∀ i j : Fin q, i < j → lengths i < lengths j)
  (h_yellow_segments_unique : ∀ i j : Fin q, i ≠ j → yellow_segments i ≠ yellow_segments j) :
  ∃ (m : ℕ), m ≥ 2 * q / n :=
sorry

end polygonal_line_exists_l306_30628


namespace quadrilateral_area_is_11_l306_30622

def point := (ℤ × ℤ)

def A : point := (0, 0)
def B : point := (1, 4)
def C : point := (4, 3)
def D : point := (3, 0)

def area_of_quadrilateral (p1 p2 p3 p4 : point) : ℤ :=
  let ⟨x1, y1⟩ := p1
  let ⟨x2, y2⟩ := p2
  let ⟨x3, y3⟩ := p3
  let ⟨x4, y4⟩ := p4
  (|x1*y2 - y1*x2 + x2*y3 - y2*x3 + x3*y4 - y3*x4 + x4*y1 - y4*x1|) / 2

theorem quadrilateral_area_is_11 : area_of_quadrilateral A B C D = 11 := by 
  sorry

end quadrilateral_area_is_11_l306_30622


namespace banana_distinct_arrangements_l306_30672

theorem banana_distinct_arrangements :
  let n := 6
  let f_B := 1
  let f_N := 2
  let f_A := 3
  (n.factorial) / (f_B.factorial * f_N.factorial * f_A.factorial) = 60 := by
sorry

end banana_distinct_arrangements_l306_30672


namespace xy_sum_equal_two_or_minus_two_l306_30626

/-- 
Given the conditions |x| = 3, |y| = 5, and xy < 0, prove that x + y = 2 or x + y = -2. 
-/
theorem xy_sum_equal_two_or_minus_two (x y : ℝ) (hx : |x| = 3) (hy : |y| = 5) (hxy : x * y < 0) : x + y = 2 ∨ x + y = -2 := 
  sorry

end xy_sum_equal_two_or_minus_two_l306_30626


namespace largest_consecutive_odd_number_is_27_l306_30689

theorem largest_consecutive_odd_number_is_27 (a b c : ℤ) 
  (h1: a + b + c = 75)
  (h2: c - a = 6)
  (h3: b = a + 2)
  (h4: c = a + 4) :
  c = 27 := 
sorry

end largest_consecutive_odd_number_is_27_l306_30689


namespace polygon_sides_exterior_interior_sum_l306_30693

theorem polygon_sides_exterior_interior_sum (n : ℕ) (h : ((n - 2) * 180 = 360)) : n = 4 :=
by sorry

end polygon_sides_exterior_interior_sum_l306_30693


namespace exists_xy_such_that_x2_add_y2_eq_n_mod_p_p_mod_4_eq_1_implies_n_can_be_0_p_mod_4_eq_3_implies_n_cannot_be_0_l306_30695

theorem exists_xy_such_that_x2_add_y2_eq_n_mod_p
  (p : ℕ) [Fact (Nat.Prime p)] (n : ℤ)
  (hp1 : p > 5) :
  (∃ x y : ℤ, x ≠ 0 ∧ y ≠ 0 ∧ (x^2 + y^2) % p = n % p) :=
sorry

theorem p_mod_4_eq_1_implies_n_can_be_0
  (p : ℕ) [Fact (Nat.Prime p)] (hp1 : p % 4 = 1) : 
  (∃ x y : ℤ, x ≠ 0 ∧ y ≠ 0 ∧ (x^2 + y^2) % p = 0) :=
sorry

theorem p_mod_4_eq_3_implies_n_cannot_be_0
  (p : ℕ) [Fact (Nat.Prime p)] (hp : p % 4 = 3) :
  ¬(∃ x y : ℤ, x ≠ 0 ∧ y ≠ 0 ∧ (x^2 + y^2) % p = 0) :=
sorry

end exists_xy_such_that_x2_add_y2_eq_n_mod_p_p_mod_4_eq_1_implies_n_can_be_0_p_mod_4_eq_3_implies_n_cannot_be_0_l306_30695


namespace number_of_blobs_of_glue_is_96_l306_30616

def pyramid_blobs_of_glue : Nat :=
  let layer1 := 4 * (4 - 1) * 2
  let layer2 := 3 * (3 - 1) * 2
  let layer3 := 2 * (2 - 1) * 2
  let between1_and_2 := 3 * 3 * 4
  let between2_and_3 := 2 * 2 * 4
  let between3_and_4 := 4
  layer1 + layer2 + layer3 + between1_and_2 + between2_and_3 + between3_and_4

theorem number_of_blobs_of_glue_is_96 :
  pyramid_blobs_of_glue = 96 :=
by
  sorry

end number_of_blobs_of_glue_is_96_l306_30616


namespace square_area_divided_into_rectangles_l306_30637

theorem square_area_divided_into_rectangles (l w : ℝ) 
  (h1 : 2 * (l + w) = 120)
  (h2 : l = 5 * w) :
  (5 * w * w)^2 = 2500 := 
by {
  -- Sorry placeholder for proof
  sorry
}

end square_area_divided_into_rectangles_l306_30637


namespace domain_of_f_l306_30623

noncomputable def f (x : ℝ) := 1 / Real.log (x + 1) + Real.sqrt (9 - x^2)

theorem domain_of_f : {x : ℝ | (x > -1) ∧ (x ≠ 0) ∧ (x ∈ [-3, 3])} = 
  {x : ℝ | -1 < x ∧ x < 0} ∪ {x : ℝ | 0 < x ∧ x ≤ 3} :=
by
  sorry

end domain_of_f_l306_30623


namespace cost_first_third_hour_l306_30602

theorem cost_first_third_hour 
  (c : ℝ) 
  (h1 : 0 < c) 
  (h2 : ∀ t : ℝ, t > 1/4 → (t - 1/4) * 12 + c = 31)
  : c = 5 :=
by
  sorry

end cost_first_third_hour_l306_30602


namespace original_radius_of_cylinder_in_inches_l306_30633

theorem original_radius_of_cylinder_in_inches
  (r : ℝ) (h : ℝ) (V : ℝ → ℝ → ℝ → ℝ) 
  (h_increased_radius : V (r + 4) h π = V r (h + 4) π) 
  (h_original_height : h = 3) :
  r = 8 :=
by
  sorry

end original_radius_of_cylinder_in_inches_l306_30633


namespace max_people_transition_l306_30661

theorem max_people_transition (a : ℕ) (b : ℕ) (c : ℕ) 
  (hA : a = 850 * 6 / 100) (hB : b = 1500 * 42 / 1000) (hC : c = 4536 / 72) :
  max a (max b c) = 63 := 
sorry

end max_people_transition_l306_30661


namespace eccentricity_of_ellipse_l306_30632

theorem eccentricity_of_ellipse (a c : ℝ) (h1 : 2 * c = a) : (c / a) = (1 / 2) :=
by
  -- This is where we would write the proof, but we're using sorry to skip the proof steps.
  sorry

end eccentricity_of_ellipse_l306_30632


namespace total_dogs_l306_30648

def number_of_boxes : ℕ := 15
def dogs_per_box : ℕ := 8

theorem total_dogs : number_of_boxes * dogs_per_box = 120 := by
  sorry

end total_dogs_l306_30648


namespace average_probable_weight_l306_30668

-- Definitions based on the conditions
def ArunOpinion (w : ℝ) : Prop := 65 < w ∧ w < 72
def BrotherOpinion (w : ℝ) : Prop := 60 < w ∧ w < 70
def MotherOpinion (w : ℝ) : Prop := w ≤ 68

-- The actual statement we want to prove
theorem average_probable_weight : 
  (∀ (w : ℝ), ArunOpinion w → BrotherOpinion w → MotherOpinion w → 65 < w ∧ w ≤ 68) →
  (65 + 68) / 2 = 66.5 :=
by 
  intros h1
  sorry

end average_probable_weight_l306_30668


namespace calculate_original_lemon_price_l306_30615

variable (p_lemon_old p_lemon_new p_grape_old p_grape_new : ℝ)
variable (num_lemons num_grapes revenue : ℝ)

theorem calculate_original_lemon_price :
  ∀ (L : ℝ),
  -- conditions
  p_lemon_old = L ∧
  p_lemon_new = L + 4 ∧
  p_grape_old = 7 ∧
  p_grape_new = 9 ∧
  num_lemons = 80 ∧
  num_grapes = 140 ∧
  revenue = 2220 ->
  -- proof that the original price is 8
  p_lemon_old = 8 :=
by
  intros L h
  have h1 : p_lemon_new = L + 4 := h.2.1
  have h2 : p_grape_old = 7 := h.2.2.1
  have h3 : p_grape_new = 9 := h.2.2.2.1
  have h4 : num_lemons = 80 := h.2.2.2.2.1
  have h5 : num_grapes = 140 := h.2.2.2.2.2.1
  have h6 : revenue = 2220 := h.2.2.2.2.2.2
  sorry

end calculate_original_lemon_price_l306_30615


namespace remainders_equality_l306_30679

open Nat

theorem remainders_equality (P P' D R R' r r': ℕ) 
  (hP : P > P')
  (hP_R : P % D = R)
  (hP'_R' : P' % D = R')
  (hPP' : (P * P') % D = r)
  (hRR' : (R * R') % D = r') : r = r' := 
sorry

end remainders_equality_l306_30679


namespace range_of_a_plus_b_l306_30651

variable (a b : ℝ)
variable (pos_a : 0 < a)
variable (pos_b : 0 < b)
variable (h : a + b + 1/a + 1/b = 5)

theorem range_of_a_plus_b : 1 ≤ a + b ∧ a + b ≤ 4 := by
  sorry

end range_of_a_plus_b_l306_30651


namespace equilateral_triangle_in_ellipse_l306_30656

def ellipse_equation (x y a b : ℝ) : Prop := 
  ((x - y)^2 / a^2) + ((x + y)^2 / b^2) = 1

theorem equilateral_triangle_in_ellipse 
  {a b x y : ℝ}
  (A B C : ℝ × ℝ)
  (hA : A.1 = 0 ∧ A.2 = b)
  (hBC_parallel : ∃ k : ℝ, B.2 = k * B.1 ∧ C.2 = k * C.1 ∧ k = 1)
  (hF : ∃ F : ℝ × ℝ, F = C)
  (hEllipseA : ellipse_equation A.1 A.2 a b) 
  (hEllipseB : ellipse_equation B.1 B.2 a b)
  (hEllipseC : ellipse_equation C.1 C.2 a b) 
  (equilateral : dist A B = dist B C ∧ dist B C = dist C A) :
  AB / b = 8 / 5 :=
sorry

end equilateral_triangle_in_ellipse_l306_30656


namespace prime_divisor_form_l306_30611

theorem prime_divisor_form (n : ℕ) (q : ℕ) (hq : (2^(2^n) + 1) % q = 0) (prime_q : Nat.Prime q) :
  ∃ k : ℕ, q = 2^(n+1) * k + 1 :=
sorry

end prime_divisor_form_l306_30611


namespace train_passes_jogger_in_37_seconds_l306_30697

noncomputable def jogger_speed_kmph : ℝ := 9
noncomputable def train_speed_kmph : ℝ := 45
noncomputable def jogger_lead_m : ℝ := 250
noncomputable def train_length_m : ℝ := 120

noncomputable def jogger_speed_mps : ℝ := jogger_speed_kmph * 1000 / 3600
noncomputable def train_speed_mps : ℝ := train_speed_kmph * 1000 / 3600
noncomputable def relative_speed_mps : ℝ := train_speed_mps - jogger_speed_mps
noncomputable def total_distance_m : ℝ := jogger_lead_m + train_length_m

theorem train_passes_jogger_in_37_seconds :
  total_distance_m / relative_speed_mps = 37 := by
  sorry

end train_passes_jogger_in_37_seconds_l306_30697


namespace sum_infinite_geometric_series_l306_30610

theorem sum_infinite_geometric_series : 
  let a : ℝ := 2
  let r : ℝ := -5/8
  a / (1 - r) = 16/13 :=
by
  sorry

end sum_infinite_geometric_series_l306_30610


namespace next_year_multiple_of_6_8_9_l306_30603

theorem next_year_multiple_of_6_8_9 (n : ℕ) (h₀ : n = 2016) (h₁ : n % 6 = 0) (h₂ : n % 8 = 0) (h₃ : n % 9 = 0) : ∃ m > n, m % 6 = 0 ∧ m % 8 = 0 ∧ m % 9 = 0 ∧ m = 2088 :=
by
  sorry

end next_year_multiple_of_6_8_9_l306_30603


namespace total_pepper_weight_l306_30607

theorem total_pepper_weight :
  let green_peppers := 2.8333333333333335
  let red_peppers := 3.254
  let yellow_peppers := 1.375
  let orange_peppers := 0.567
  (green_peppers + red_peppers + yellow_peppers + orange_peppers) = 8.029333333333333 := 
by
  sorry

end total_pepper_weight_l306_30607


namespace num_sets_B_l306_30635

open Set

def A : Set ℕ := {1, 3}

theorem num_sets_B :
  ∃ (B : ℕ → Set ℕ), (∀ b, B b ∪ A = {1, 3, 5}) ∧ (∃ s t u v, B s = {5} ∧
                                                   B t = {1, 5} ∧
                                                   B u = {3, 5} ∧
                                                   B v = {1, 3, 5} ∧ 
                                                   s ≠ t ∧ s ≠ u ∧ s ≠ v ∧
                                                   t ≠ u ∧ t ≠ v ∧
                                                   u ≠ v) :=
sorry

end num_sets_B_l306_30635


namespace fish_ratio_l306_30617

theorem fish_ratio (B T S Bo : ℕ) 
  (hBilly : B = 10) 
  (hTonyBilly : T = 3 * B) 
  (hSarahTony : S = T + 5) 
  (hBobbySarah : Bo = 2 * S) 
  (hTotalFish : Bo + S + T + B = 145) : 
  T / B = 3 :=
by sorry

end fish_ratio_l306_30617


namespace triangle_property_l306_30662

theorem triangle_property
  (A B C : ℝ)
  (a b c : ℝ)
  (R : ℝ)
  (hR : R = Real.sqrt 3)
  (h1 : a * Real.sin C + Real.sqrt 3 * c * Real.cos A = 0)
  (h2 : b + c = Real.sqrt 11)
  (htri : a / Real.sin A = 2 * R ∧ b / Real.sin B = 2 * R ∧ c / Real.sin C = 2 * R):
  a = 3 ∧ (1 / 2 * b * c * Real.sin A = Real.sqrt 3 / 2) := 
sorry

end triangle_property_l306_30662


namespace sum_remainder_l306_30605

theorem sum_remainder (a b c : ℕ) 
  (h1 : a % 15 = 11) 
  (h2 : b % 15 = 13) 
  (h3 : c % 15 = 9) :
  (a + b + c) % 15 = 3 := 
by
  sorry

end sum_remainder_l306_30605


namespace evaluate_expression_l306_30658

theorem evaluate_expression : (5 + 2) + (8 + 6) + (4 + 7) + (3 + 2) = 37 := 
sorry

end evaluate_expression_l306_30658


namespace almond_butter_servings_l306_30692

def convert_mixed_to_fraction (a b : ℤ) (n : ℕ) : ℚ :=
  (a * n + b) / n

def servings (total servings_fraction : ℚ) : ℚ :=
  total / servings_fraction

theorem almond_butter_servings :
  servings (convert_mixed_to_fraction 35 2 3) (convert_mixed_to_fraction 2 1 2) = 14 + 4 / 15 :=
by
  sorry

end almond_butter_servings_l306_30692


namespace sobhas_parents_age_difference_l306_30678

def difference_in_ages (F M : ℕ) : ℕ := F - M

theorem sobhas_parents_age_difference
  (S F M : ℕ)
  (h1 : F = S + 38)
  (h2 : M = S + 32) :
  difference_in_ages F M = 6 := by
  sorry

end sobhas_parents_age_difference_l306_30678


namespace distinct_meals_count_l306_30676

def entries : ℕ := 3
def drinks : ℕ := 3
def desserts : ℕ := 3

theorem distinct_meals_count : entries * drinks * desserts = 27 :=
by
  -- sorry for skipping the proof
  sorry

end distinct_meals_count_l306_30676


namespace donation_student_amount_l306_30657

theorem donation_student_amount (a : ℕ) : 
  let total_amount := 3150
  let teachers_count := 5
  let donation_teachers := teachers_count * a 
  let donation_students := total_amount - donation_teachers
  donation_students = 3150 - 5 * a :=
by
  sorry

end donation_student_amount_l306_30657


namespace min_perimeter_is_676_l306_30620

-- Definitions and conditions based on the problem statement
def equal_perimeter (a b c : ℕ) : Prop :=
  2 * a + 14 * c = 2 * b + 16 * c

def equal_area (a b c : ℕ) : Prop :=
  7 * Real.sqrt (a^2 - 49 * c^2) = 8 * Real.sqrt (b^2 - 64 * c^2)

def base_ratio (b : ℕ) : ℕ := b * 8 / 7

theorem min_perimeter_is_676 :
  ∃ a b c : ℕ, equal_perimeter a b c ∧ equal_area a b c ∧ base_ratio b = a - b ∧ 
  2 * a + 14 * c = 676 :=
sorry

end min_perimeter_is_676_l306_30620


namespace circle_radius_l306_30613

theorem circle_radius (x y : ℝ) : (x^2 + y^2 + 2*x = 0) → ∃ r, r = 1 :=
by sorry

end circle_radius_l306_30613


namespace proof_max_difference_l306_30627

/-- Digits as displayed on the engineering calculator -/
structure Digits :=
  (a b c d e f g h i : ℕ)

-- Possible digits based on broken displays
axiom a_values : {x // x = 3 ∨ x = 5 ∨ x = 9}
axiom b_values : {x // x = 2 ∨ x = 3 ∨ x = 7}
axiom c_values : {x // x = 3 ∨ x = 4 ∨ x = 8 ∨ x = 9}
axiom d_values : {x // x = 2 ∨ x = 3 ∨ x = 7}
axiom e_values : {x // x = 3 ∨ x = 5 ∨ x = 9}
axiom f_values : {x // x = 1 ∨ x = 4 ∨ x = 7}
axiom g_values : {x // x = 4 ∨ x = 5 ∨ x = 9}
axiom h_values : {x // x = 2}
axiom i_values : {x // x = 4 ∨ x = 5 ∨ x = 9}

-- Minuend and subtrahend values
def minuend := 923
def subtrahend := 394

-- Maximum possible value of the difference
def max_difference := 529

theorem proof_max_difference : 
  ∃ (digits : Digits),
    digits.a = 9 ∧ digits.b = 2 ∧ digits.c = 3 ∧
    digits.d = 3 ∧ digits.e = 9 ∧ digits.f = 4 ∧
    digits.g = 5 ∧ digits.h = 2 ∧ digits.i = 9 ∧
    minuend - subtrahend = max_difference :=
by
  sorry

end proof_max_difference_l306_30627


namespace probability_jqka_is_correct_l306_30638

noncomputable def probability_sequence_is_jqka : ℚ :=
  (4 / 52) * (4 / 51) * (4 / 50) * (4 / 49)

theorem probability_jqka_is_correct :
  probability_sequence_is_jqka = (16 / 4048375) :=
by
  sorry

end probability_jqka_is_correct_l306_30638


namespace ratio_volumes_l306_30606

variables (V1 V2 : ℝ)
axiom h1 : (3 / 5) * V1 = (2 / 3) * V2

theorem ratio_volumes : V1 / V2 = 10 / 9 := by
  sorry

end ratio_volumes_l306_30606


namespace problem_l306_30655

open Set

/-- Declaration of the universal set and other sets as per the problem -/
def U : Set ℕ := {0, 1, 2, 4, 6, 8}
def M : Set ℕ := {0, 4, 6}
def N : Set ℕ := {0, 1, 6}

/-- Problem: Prove that M ∪ (U \ N) = {0, 2, 4, 6, 8} -/
theorem problem : M ∪ (U \ N) = {0, 2, 4, 6, 8} := 
by
  sorry

end problem_l306_30655


namespace athena_total_spent_l306_30667

def cost_of_sandwiches (num_sandwiches : ℕ) (cost_per_sandwich : ℝ) : ℝ :=
  num_sandwiches * cost_per_sandwich

def cost_of_drinks (num_drinks : ℕ) (cost_per_drink : ℝ) : ℝ :=
  num_drinks * cost_per_drink

def total_cost (num_sandwiches : ℕ) (cost_per_sandwich : ℝ) (num_drinks : ℕ) (cost_per_drink : ℝ) : ℝ :=
  cost_of_sandwiches num_sandwiches cost_per_sandwich + cost_of_drinks num_drinks cost_per_drink

theorem athena_total_spent :
  total_cost 3 3 2 2.5 = 14 :=
by 
  sorry

end athena_total_spent_l306_30667


namespace total_eggs_l306_30677

theorem total_eggs (students : ℕ) (eggs_per_student : ℕ) (h1 : students = 7) (h2 : eggs_per_student = 8) :
  students * eggs_per_student = 56 :=
by
  sorry

end total_eggs_l306_30677


namespace probability_blue_or_green_face_l306_30644

def cube_faces: ℕ := 6
def blue_faces: ℕ := 3
def red_faces: ℕ := 2
def green_faces: ℕ := 1

theorem probability_blue_or_green_face (h1: blue_faces + red_faces + green_faces = cube_faces):
  (3 + 1) / 6 = 2 / 3 :=
by
  sorry

end probability_blue_or_green_face_l306_30644


namespace max_value_inequality_l306_30675

theorem max_value_inequality (a x₁ x₂ : ℝ) (h_a : a < 0)
  (h_sol : ∀ x, x^2 - 4 * a * x + 3 * a^2 < 0 ↔ x₁ < x ∧ x < x₂) :
    x₁ + x₂ + a / (x₁ * x₂) ≤ - 4 * Real.sqrt 3 / 3 := by
  sorry

end max_value_inequality_l306_30675


namespace find_triples_l306_30619

theorem find_triples :
  { (a, b, c) : ℕ × ℕ × ℕ | (c-1) * (a * b - b - a) = a + b - 2 } =
  { (2, 1, 0), (1, 2, 0), (3, 4, 2), (4, 3, 2), (1, 0, 2), (0, 1, 2), (2, 4, 3), (4, 2, 3) } :=
by
  sorry

end find_triples_l306_30619


namespace perimeter_of_rectangle_l306_30665

theorem perimeter_of_rectangle (b l : ℝ) (h1 : l = 3 * b) (h2 : b * l = 75) : 2 * l + 2 * b = 40 := 
by 
  sorry

end perimeter_of_rectangle_l306_30665


namespace problem_statement_l306_30604

def f (x : Int) : Int :=
  if x > 6 then x^2 - 4
  else if -6 <= x && x <= 6 then 3*x + 2
  else 5

def adjusted_f (x : Int) : Int :=
  let fx := f x
  if x % 3 == 0 then fx + 5 else fx

theorem problem_statement : 
  adjusted_f (-8) + adjusted_f 0 + adjusted_f 9 = 94 :=
by 
  sorry

end problem_statement_l306_30604


namespace church_path_count_is_321_l306_30663

/-- A person starts at the bottom-left corner of an m x n grid and can only move north, east, or 
    northeast. Prove that the number of distinct paths to the top-right corner is 321 
    for a specific grid size (abstracted parameters included). -/
def distinct_paths_to_church (m n : ℕ) : ℕ :=
  let rec P : ℕ → ℕ → ℕ
    | 0, 0 => 1
    | i + 1, 0 => 1
    | 0, j + 1 => 1
    | i + 1, j + 1 => P i (j + 1) + P (i + 1) j + P i j
  P m n

theorem church_path_count_is_321 : distinct_paths_to_church m n = 321 :=
sorry

end church_path_count_is_321_l306_30663
