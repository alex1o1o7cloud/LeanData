import Mathlib

namespace inequality_solution_l2260_226084

theorem inequality_solution (x : ℝ) (h1 : x ≠ -2) (h2 : x ≠ 2) :
  (x ∈ Set.Ioo (-2 : ℝ) (-1) ∨ x ∈ Set.Ioi 2) ↔ 
  (∃ x : ℝ, (x^2 + x - 2) / (x + 2) ≥ (3 / (x - 2)) + (3 / 2)) := by
  sorry

end inequality_solution_l2260_226084


namespace correct_propositions_l2260_226008

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 2 * (Real.sin x + Real.cos x)

-- Proposition 2: Symmetry about the line x = -3π/4
def proposition_2 : Prop := ∀ x, f (x + 3 * Real.pi / 4) = f (-x)

-- Proposition 3: There exists φ ∈ ℝ, such that the graph of the function f(x + φ) is centrally symmetric about the origin
def proposition_3 : Prop := ∃ φ : ℝ, ∀ x, f (x + φ) = -f (-x)

theorem correct_propositions :
  (proposition_2 ∧ proposition_3) := by
  sorry

end correct_propositions_l2260_226008


namespace evelyn_lost_bottle_caps_l2260_226037

-- Definitions from the conditions
def initial_amount : ℝ := 63.0
def final_amount : ℝ := 45.0
def lost_amount : ℝ := 18.0

-- Statement to be proved
theorem evelyn_lost_bottle_caps : initial_amount - final_amount = lost_amount := 
by 
  sorry

end evelyn_lost_bottle_caps_l2260_226037


namespace autumn_pencils_l2260_226011

-- Define the conditions of the problem.
def initial_pencils := 20
def misplaced_pencils := 7
def broken_pencils := 3
def found_pencils := 4
def bought_pencils := 2

-- Define the number of pencils lost and gained.
def pencils_lost := misplaced_pencils + broken_pencils
def pencils_gained := found_pencils + bought_pencils

-- Define the final number of pencils.
def final_pencils := initial_pencils - pencils_lost + pencils_gained

-- The theorem we want to prove.
theorem autumn_pencils : final_pencils = 16 := by
  sorry

end autumn_pencils_l2260_226011


namespace ratio_of_ages_l2260_226032

variables (R J K : ℕ)

axiom h1 : R = J + 8
axiom h2 : R + 4 = 2 * (J + 4)
axiom h3 : (R + 4) * (K + 4) = 192

theorem ratio_of_ages : (R - J) / (R - K) = 2 :=
by sorry

end ratio_of_ages_l2260_226032


namespace sin_sum_arcsin_arctan_l2260_226035

theorem sin_sum_arcsin_arctan :
  let a := Real.arcsin (4 / 5)
  let b := Real.arctan (1 / 2)
  Real.sin (a + b) = (11 * Real.sqrt 5) / 25 :=
by
  sorry

end sin_sum_arcsin_arctan_l2260_226035


namespace ratio_of_age_difference_l2260_226024

-- Define the ages of the scrolls and the ratio R
variables (S1 S2 S3 S4 S5 : ℕ)
variables (R : ℚ)

-- Conditions
axiom h1 : S1 = 4080
axiom h5 : S5 = 20655
axiom h2 : S2 - S1 = R * S5
axiom h3 : S3 - S2 = R * S5
axiom h4 : S4 - S3 = R * S5
axiom h6 : S5 - S4 = R * S5

-- The theorem to prove
theorem ratio_of_age_difference : R = 16575 / 82620 :=
by 
  sorry

end ratio_of_age_difference_l2260_226024


namespace t_range_l2260_226042

noncomputable def exists_nonneg_real_numbers_satisfying_conditions (t : ℝ) : Prop :=
  ∃ (x y z : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ 
  (3 * x^2 + 3 * z * x + z^2 = 1) ∧ 
  (3 * y^2 + 3 * y * z + z^2 = 4) ∧ 
  (x^2 - x * y + y^2 = t)

theorem t_range : ∀ t : ℝ, exists_nonneg_real_numbers_satisfying_conditions t → 
  (t ≥ (3 - Real.sqrt 5) / 2 ∧ t ≤ 1) :=
sorry

end t_range_l2260_226042


namespace total_weight_of_balls_l2260_226016

theorem total_weight_of_balls :
  let weight_blue := 6
  let weight_brown := 3.12
  weight_blue + weight_brown = 9.12 :=
by
  sorry

end total_weight_of_balls_l2260_226016


namespace product_of_consecutive_even_numbers_divisible_by_8_l2260_226010

theorem product_of_consecutive_even_numbers_divisible_by_8 (n : ℤ) : 8 ∣ (2 * n * (2 * n + 2)) :=
by sorry

end product_of_consecutive_even_numbers_divisible_by_8_l2260_226010


namespace base_number_is_two_l2260_226087

theorem base_number_is_two (x : ℝ) (n : ℕ) (h1 : x^(2*n) + x^(2*n) + x^(2*n) + x^(2*n) = 4^22)
  (h2 : n = 21) : x = 2 :=
sorry

end base_number_is_two_l2260_226087


namespace knocks_to_knicks_l2260_226004

variable (knicks knacks knocks : ℝ)

def knicks_eq_knacks : Prop := 
  8 * knicks = 3 * knacks

def knacks_eq_knocks : Prop := 
  4 * knacks = 5 * knocks

theorem knocks_to_knicks
  (h1 : knicks_eq_knacks knicks knacks)
  (h2 : knacks_eq_knocks knacks knocks) :
  20 * knocks = 320 / 15 * knicks :=
  sorry

end knocks_to_knicks_l2260_226004


namespace intersection_of_sets_l2260_226058

open Set

theorem intersection_of_sets : 
  let M : Set ℕ := {0, 2, 4, 8}
  let N : Set ℕ := { x | ∃ a, a ∈ M ∧ x = 2 * a }
  M ∩ N = {0, 4, 8} := 
by
  let M : Set ℕ := {0, 2, 4, 8}
  let N : Set ℕ := { x | ∃ a, a ∈ M ∧ x = 2 * a }
  sorry

end intersection_of_sets_l2260_226058


namespace pentagon_area_l2260_226012

noncomputable def square_area (side_length : ℤ) : ℤ :=
  side_length * side_length

theorem pentagon_area (CF : ℤ) (a b : ℤ) (CE : ℤ) (ED : ℤ) (EF : ℤ) :
  (CF = 5) →
  (a = CE + ED) →
  (b = EF) →
  (CE < ED) →
  CF * CF = CE * CE + EF * EF →
  square_area a + square_area b - (CE * EF / 2) = 71 :=
by
  intros hCF ha hb hCE_lt_ED hPythagorean
  sorry

end pentagon_area_l2260_226012


namespace valid_triplets_l2260_226096

theorem valid_triplets (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_leq1 : a ≤ b) (h_leq2 : b ≤ c)
  (h_div1 : a ∣ (b + c)) (h_div2 : b ∣ (a + c)) (h_div3 : c ∣ (a + b)) :
  (a = b ∧ b = c) ∨ (a = b ∧ c = 2 * a) ∨ (b = 2 * a ∧ c = 3 * a) :=
sorry

end valid_triplets_l2260_226096


namespace opposite_of_neg_five_l2260_226079

theorem opposite_of_neg_five : -(-5) = 5 := by
  sorry

end opposite_of_neg_five_l2260_226079


namespace problem_statement_l2260_226017

theorem problem_statement (a b : ℝ) (h_domain : ∀ x, 1 ≤ x ∧ x ≤ b)
  (h_range : ∀ y, 1 ≤ y ∧ y ≤ b) (h_b_gt_1 : b > 1)
  (h1 : a = 1) (h2 : 1/2 * (b - 1)^2 + 1 = b) : a + b = 4 :=
sorry

end problem_statement_l2260_226017


namespace michael_truck_meet_once_l2260_226026

/-- Michael walks at 6 feet per second -/
def michael_speed := 6

/-- Trash pails are located every 300 feet along the path -/
def pail_distance := 300

/-- A garbage truck travels at 15 feet per second -/
def truck_speed := 15

/-- The garbage truck stops for 45 seconds at each pail -/
def truck_stop_time := 45

/-- Michael passes a pail just as the truck leaves the next pail -/
def initial_distance := 300

/-- Prove that Michael and the truck meet exactly 1 time -/
theorem michael_truck_meet_once :
  ∀ (meeting_times : ℕ), meeting_times = 1 := by
  sorry

end michael_truck_meet_once_l2260_226026


namespace is_incorrect_B_l2260_226059

variable {a b c : ℝ}

theorem is_incorrect_B :
  ¬ ((a > b ∧ b > c) → (1 / (b - c)) < (1 / (a - c))) :=
sorry

end is_incorrect_B_l2260_226059


namespace delivery_truck_speed_l2260_226074

theorem delivery_truck_speed :
  ∀ d t₁ t₂: ℝ,
    (t₁ = 15 / 60) ∧ (t₂ = -15 / 60) ∧ 
    (t₁ = d / 20 - 1 / 4) ∧ (t₂ = d / 60 + 1 / 4) →
    (d = 15) →
    (t = 1 / 2) →
    ( ∃ v: ℝ, t = d / v ∧ v = 30 ) :=
by sorry

end delivery_truck_speed_l2260_226074


namespace tangent_lines_to_circle_through_point_l2260_226091

noncomputable def circle_center : ℝ × ℝ := (1, 2)
noncomputable def circle_radius : ℝ := 2
noncomputable def point_P : ℝ × ℝ := (-1, 5)

theorem tangent_lines_to_circle_through_point :
  ∃ m c : ℝ, (∀ x y : ℝ, (x - 1) ^ 2 + (y - 2) ^ 2 = 4 → (m * x + y + c = 0 → (y = -m * x - c))) ∧
  (m = 5/12 ∧ c = -55/12) ∨ (m = 0 ∧ ∀ x : ℝ, x = -1) :=
sorry

end tangent_lines_to_circle_through_point_l2260_226091


namespace price_reduction_l2260_226073

theorem price_reduction (x : ℝ) (h : 560 * (1 - x) * (1 - x) = 315) : 
  560 * (1 - x)^2 = 315 := 
by
  sorry

end price_reduction_l2260_226073


namespace find_q_l2260_226057

variable {m n q : ℝ}

theorem find_q (h1 : m = 3 * n + 5) (h2 : m + 2 = 3 * (n + q) + 5) : q = 2 / 3 := by
  sorry

end find_q_l2260_226057


namespace find_n_l2260_226007

theorem find_n (n : ℕ) (h : (n + 1) * n.factorial = 5040) : n = 6 := 
by sorry

end find_n_l2260_226007


namespace merchant_marked_price_l2260_226038

-- Given conditions: 30% discount on list price, 10% discount on marked price, 25% profit on selling price
variable (L : ℝ) -- List price
variable (C : ℝ) -- Cost price after discount: C = 0.7 * L
variable (M : ℝ) -- Marked price
variable (S : ℝ) -- Selling price after discount on marked price: S = 0.9 * M

noncomputable def proof_problem : Prop :=
  C = 0.7 * L ∧
  C = 0.75 * S ∧
  S = 0.9 * M ∧
  M = 103.7 / 100 * L

theorem merchant_marked_price (L : ℝ) (C : ℝ) (S : ℝ) (M : ℝ) :
  (C = 0.7 * L) → 
  (C = 0.75 * S) → 
  (S = 0.9 * M) → 
  M = 103.7 / 100 * L :=
by
  sorry

end merchant_marked_price_l2260_226038


namespace mutually_exclusive_necessary_not_sufficient_complementary_l2260_226064

variables {Ω : Type} {A1 A2 : Set Ω}

/-- Definition of mutually exclusive events -/
def mutually_exclusive (A1 A2 : Set Ω) : Prop :=
  A1 ∩ A2 = ∅

/-- Definition of complementary events -/
def complementary (A1 A2 : Set Ω) : Prop :=
  A1 ∪ A2 = Set.univ ∧ mutually_exclusive A1 A2

/-- The proposition that mutually exclusive events are necessary but not sufficient for being complementary -/
theorem mutually_exclusive_necessary_not_sufficient_complementary :
  (mutually_exclusive A1 A2 → complementary A1 A2) = false 
  ∧ (complementary A1 A2 → mutually_exclusive A1 A2) = true :=
sorry

end mutually_exclusive_necessary_not_sufficient_complementary_l2260_226064


namespace sector_triangle_radii_l2260_226053

theorem sector_triangle_radii 
  (r : ℝ) (theta : ℝ) (radius : ℝ) 
  (h_theta_eq: theta = 60)
  (h_radius_eq: radius = 10) :
  let R := (radius * Real.sqrt 3) / 3
  let r_in := (radius * Real.sqrt 3) / 6
  R = 10 * (Real.sqrt 3) / 3 ∧ r_in = 10 * (Real.sqrt 3) / 6 := 
by
  sorry

end sector_triangle_radii_l2260_226053


namespace polynomial_divisibility_l2260_226020

theorem polynomial_divisibility (n : ℕ) (h : n > 2) : 
    (∀ k : ℕ, n = 3 * k + 1) ↔ ∃ (k : ℕ), n = 3 * k + 1 := 
sorry

end polynomial_divisibility_l2260_226020


namespace damian_serena_passing_times_l2260_226043

/-- 
  Damian and Serena are running on a circular track for 40 minutes.
  Damian runs clockwise at 220 m/min on the inner lane with a radius of 45 meters.
  Serena runs counterclockwise at 260 m/min on the outer lane with a radius of 55 meters.
  They start on the same radial line.
  Prove that they pass each other exactly 184 times in 40 minutes. 
-/
theorem damian_serena_passing_times
  (time_run : ℕ)
  (damian_speed : ℕ)
  (serena_speed : ℕ)
  (damian_radius : ℝ)
  (serena_radius : ℝ)
  (start_same_line : Prop) :
  time_run = 40 →
  damian_speed = 220 →
  serena_speed = 260 →
  damian_radius = 45 →
  serena_radius = 55 →
  start_same_line →
  ∃ n : ℕ, n = 184 :=
by
  sorry

end damian_serena_passing_times_l2260_226043


namespace sophie_saves_money_l2260_226048

variable (loads_per_week : ℕ) (dryer_sheets_per_load : ℕ) (weeks_per_year : ℕ) (cost_per_box : ℝ) (sheets_per_box : ℕ)
variable (given_on_birthday : Bool)

noncomputable def money_saved_per_year (loads_per_week : ℕ) (dryer_sheets_per_load : ℕ) (weeks_per_year : ℕ) (cost_per_box : ℝ) (sheets_per_box : ℕ) : ℝ :=
  (loads_per_week * dryer_sheets_per_load * weeks_per_year / sheets_per_box) * cost_per_box

theorem sophie_saves_money (h_loads_per_week : loads_per_week = 4) (h_dryer_sheets_per_load : dryer_sheets_per_load = 1)
                           (h_weeks_per_year : weeks_per_year = 52) (h_cost_per_box : cost_per_box = 5.50)
                           (h_sheets_per_box : sheets_per_box = 104) (h_given_on_birthday : given_on_birthday = true) :
  money_saved_per_year 4 1 52 5.50 104 = 11 :=
by
  have h1 : loads_per_week = 4 := h_loads_per_week
  have h2 : dryer_sheets_per_load = 1 := h_dryer_sheets_per_load
  have h3 : weeks_per_year = 52 := h_weeks_per_year
  have h4 : cost_per_box = 5.50 := h_cost_per_box
  have h5 : sheets_per_box = 104 := h_sheets_per_box
  have h6 : given_on_birthday = true := h_given_on_birthday
  sorry

end sophie_saves_money_l2260_226048


namespace alex_blueberry_pies_l2260_226050

-- Definitions based on given conditions:
def total_pies : ℕ := 30
def ratio (a b c : ℕ) : Prop := (a : ℚ) / b = 2 / 3 ∧ (b : ℚ) / c = 3 / 5

-- Statement to prove the number of blueberry pies
theorem alex_blueberry_pies :
  ∃ (a b c : ℕ), ratio a b c ∧ a + b + c = total_pies ∧ b = 9 :=
by
  sorry

end alex_blueberry_pies_l2260_226050


namespace part1_part2_l2260_226072

def pos_real := {x : ℝ // 0 < x}

variables (a b c : pos_real)

theorem part1 (h : a.val ^ (3/2) + b.val ^ (3/2) + c.val ^ (3/2) = 1) : a.val * b.val * c.val ≤ 1 / 9 :=
sorry

theorem part2 (h : a.val ^ (3/2) + b.val ^ (3/2) + c.val ^ (3/2) = 1) : 
  a.val / (b.val + c.val) + b.val / (a.val + c.val) + c.val / (a.val + b.val) ≤ 1 / (2 * Real.sqrt (a.val * b.val * c.val)) :=
sorry

end part1_part2_l2260_226072


namespace system_solution_ratio_l2260_226030

theorem system_solution_ratio (x y z : ℝ) (h_xyz_nonzero: x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0)
  (h1 : x + (95/9)*y + 4*z = 0) (h2 : 4*x + (95/9)*y - 3*z = 0) (h3 : 3*x + 5*y - 4*z = 0) :
  (x * z) / (y ^ 2) = 175 / 81 := 
by sorry

end system_solution_ratio_l2260_226030


namespace sum_of_variables_l2260_226025

theorem sum_of_variables (a b c d : ℕ) (h1 : ac + bd + ad + bc = 1997) : a + b + c + d = 1998 :=
sorry

end sum_of_variables_l2260_226025


namespace range_of_a_l2260_226086

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x ≤ 0 → a * 4^x - 2^x + 2 > 0) → a > -1 :=
by sorry

end range_of_a_l2260_226086


namespace postman_pete_mileage_l2260_226028

theorem postman_pete_mileage :
  let initial_steps := 30000
  let resets := 72
  let final_steps := 45000
  let steps_per_mile := 1500
  let steps_per_full_cycle := 99999 + 1
  let total_steps := initial_steps + resets * steps_per_full_cycle + final_steps
  total_steps / steps_per_mile = 4850 := 
by 
  sorry

end postman_pete_mileage_l2260_226028


namespace mary_cards_left_l2260_226097

noncomputable def mary_initial_cards : ℝ := 18.0
noncomputable def cards_to_fred : ℝ := 26.0
noncomputable def cards_bought : ℝ := 40.0
noncomputable def mary_final_cards : ℝ := 32.0

theorem mary_cards_left :
  (mary_initial_cards + cards_bought) - cards_to_fred = mary_final_cards := 
by 
  sorry

end mary_cards_left_l2260_226097


namespace discount_percentage_is_25_l2260_226022

-- Define the conditions
def cost_of_coffee : ℕ := 6
def cost_of_cheesecake : ℕ := 10
def final_price_with_discount : ℕ := 12

-- Define the total cost without discount
def total_cost_without_discount : ℕ := cost_of_coffee + cost_of_cheesecake

-- Define the discount amount
def discount_amount : ℕ := total_cost_without_discount - final_price_with_discount

-- Define the percentage discount
def percentage_discount : ℕ := (discount_amount * 100) / total_cost_without_discount

-- Proof Statement
theorem discount_percentage_is_25 : percentage_discount = 25 := by
  sorry

end discount_percentage_is_25_l2260_226022


namespace find_n_l2260_226098

theorem find_n 
  (molecular_weight : ℕ)
  (atomic_weight_Al : ℕ)
  (weight_OH : ℕ)
  (n : ℕ) 
  (h₀ : molecular_weight = 78)
  (h₁ : atomic_weight_Al = 27) 
  (h₂ : weight_OH = 17)
  (h₃ : molecular_weight = atomic_weight_Al + n * weight_OH) : 
  n = 3 := 
by 
  -- the proof is omitted
  sorry

end find_n_l2260_226098


namespace quotient_correct_l2260_226085

noncomputable def find_quotient (z : ℚ) : ℚ :=
  let dividend := (5 * z ^ 5 - 3 * z ^ 4 + 6 * z ^ 3 - 8 * z ^ 2 + 9 * z - 4)
  let divisor := (4 * z ^ 2 + 5 * z + 3)
  let quotient := ((5 / 4) * z ^ 3 - (47 / 16) * z ^ 2 + (257 / 64) * z - (1547 / 256))
  quotient

theorem quotient_correct (z : ℚ) :
  find_quotient z = ((5 / 4) * z ^ 3 - (47 / 16) * z ^ 2 + (257 / 64) * z - (1547 / 256)) :=
by
  sorry

end quotient_correct_l2260_226085


namespace solve_equation_l2260_226082

theorem solve_equation (x : ℝ) :
  (2 * x - 1)^2 - 25 = 0 ↔ (x = 3 ∨ x = -2) :=
by
  sorry

end solve_equation_l2260_226082


namespace PQR_positive_iff_P_Q_R_positive_l2260_226003

noncomputable def P (a b c : ℝ) : ℝ := a + b - c
noncomputable def Q (a b c : ℝ) : ℝ := b + c - a
noncomputable def R (a b c : ℝ) : ℝ := c + a - b

theorem PQR_positive_iff_P_Q_R_positive (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (P a b c * Q a b c * R a b c > 0) ↔ (P a b c > 0 ∧ Q a b c > 0 ∧ R a b c > 0) :=
sorry

end PQR_positive_iff_P_Q_R_positive_l2260_226003


namespace number_put_in_machine_l2260_226049

theorem number_put_in_machine (x : ℕ) (y : ℕ) (h1 : y = x + 15 - 6) (h2 : y = 77) : x = 68 :=
by
  sorry

end number_put_in_machine_l2260_226049


namespace intersection_complement_correct_l2260_226056

def I : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {1, 3, 4, 5}
def B : Set ℕ := {1, 4}
def C_I (s : Set ℕ) := I \ s  -- set complement

theorem intersection_complement_correct: A ∩ C_I B = {3, 5} := by
  -- proof steps go here
  sorry

end intersection_complement_correct_l2260_226056


namespace inequality_am_gm_l2260_226066

theorem inequality_am_gm (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x / (y * z) + y / (z * x) + z / (x * y)) ≥ (1 / x + 1 / y + 1 / z) := 
by
  sorry

end inequality_am_gm_l2260_226066


namespace meaningful_expression_range_l2260_226061

theorem meaningful_expression_range (x : ℝ) : 
  (x - 1 ≥ 0) ∧ (x ≠ 3) ↔ (x ≥ 1 ∧ x ≠ 3) := 
by
  sorry

end meaningful_expression_range_l2260_226061


namespace unique_ab_not_determined_l2260_226067

noncomputable def f (a b : ℝ) (x : ℝ) := a * x^2 + b * x - Real.sqrt 2

theorem unique_ab_not_determined :
  ∀ (a b : ℝ), a > 0 → b > 0 → 
  f a b (f a b (Real.sqrt 2)) = 1 → False := 
by
  sorry

end unique_ab_not_determined_l2260_226067


namespace decimal_111_to_base_5_l2260_226054

def decimal_to_base_5 (n : ℕ) : ℕ :=
  let rec loop (n : ℕ) (acc : ℕ) (place : ℕ) :=
    if n = 0 then acc
    else 
      let rem := n % 5
      let q := n / 5
      loop q (acc + rem * place) (place * 10)
  loop n 0 1

theorem decimal_111_to_base_5 : decimal_to_base_5 111 = 421 :=
  sorry

end decimal_111_to_base_5_l2260_226054


namespace cost_price_of_article_l2260_226065

-- Definitions based on the conditions
def sellingPrice : ℝ := 800
def profitPercentage : ℝ := 25

-- Statement to prove the cost price
theorem cost_price_of_article :
  ∃ cp : ℝ, profitPercentage = ((sellingPrice - cp) / cp) * 100 ∧ cp = 640 :=
by
  sorry

end cost_price_of_article_l2260_226065


namespace min_value_sin_cos_l2260_226039

open Real

theorem min_value_sin_cos : ∀ x : ℝ, 
  ∃ (y : ℝ), (∀ x, y ≤ sin x ^ 6 + (5 / 3) * cos x ^ 6) ∧ y = 5 / 8 :=
by
  sorry

end min_value_sin_cos_l2260_226039


namespace katie_earnings_l2260_226023

theorem katie_earnings :
  4 * 3 + 3 * 7 + 2 * 5 + 5 * 2 = 53 := 
by 
  sorry

end katie_earnings_l2260_226023


namespace probability_odd_sum_probability_even_product_l2260_226051
open Classical

noncomputable def number_of_possible_outcomes : ℕ := 36
noncomputable def number_of_odd_sum_outcomes : ℕ := 18
noncomputable def number_of_even_product_outcomes : ℕ := 27

theorem probability_odd_sum (n : ℕ) (m_1 : ℕ) (h1 : n = number_of_possible_outcomes)
  (h2 : m_1 = number_of_odd_sum_outcomes) : (m_1 : ℝ) / n = 1 / 2 :=
by
  sorry

theorem probability_even_product (n : ℕ) (m_2 : ℕ) (h1 : n = number_of_possible_outcomes)
  (h2 : m_2 = number_of_even_product_outcomes) : (m_2 : ℝ) / n = 3 / 4 :=
by
  sorry

end probability_odd_sum_probability_even_product_l2260_226051


namespace number_of_boxes_initially_l2260_226093

theorem number_of_boxes_initially (B : ℕ) (h1 : ∃ B, 8 * B - 17 = 15) : B = 4 :=
  by
  sorry

end number_of_boxes_initially_l2260_226093


namespace water_breaks_frequency_l2260_226075

theorem water_breaks_frequency :
  ∃ W : ℕ, (240 / 120 + 10) = 240 / W :=
by
  existsi (20 : ℕ)
  sorry

end water_breaks_frequency_l2260_226075


namespace share_of_A_correct_l2260_226046

theorem share_of_A_correct :
  let investment_A1 := 20000
  let investment_A2 := 15000
  let investment_B1 := 20000
  let investment_B2 := 16000
  let investment_C1 := 20000
  let investment_C2 := 26000
  let total_months1 := 5
  let total_months2 := 7
  let total_profit := 69900

  let total_investment_A := (investment_A1 * total_months1) + (investment_A2 * total_months2)
  let total_investment_B := (investment_B1 * total_months1) + (investment_B2 * total_months2)
  let total_investment_C := (investment_C1 * total_months1) + (investment_C2 * total_months2)
  let total_investment := total_investment_A + total_investment_B + total_investment_C

  let share_A := (total_investment_A : ℝ) / (total_investment : ℝ)
  let profit_A := share_A * (total_profit : ℝ)

  profit_A = 20500.99 :=
by
  sorry

end share_of_A_correct_l2260_226046


namespace gift_contributors_l2260_226078

theorem gift_contributors :
  (∃ (n : ℕ), n ≥ 1 ∧ n ≤ 20 ∧ ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → (9 : ℕ) ≤ 20) →
  (∃ (n : ℕ), n = 12) :=
by
  sorry

end gift_contributors_l2260_226078


namespace find_x_y_z_sum_l2260_226027

theorem find_x_y_z_sum :
  ∃ (x y z : ℝ), 
    x^2 + 27 = -8 * y + 10 * z ∧
    y^2 + 196 = 18 * z + 13 * x ∧
    z^2 + 119 = -3 * x + 30 * y ∧
    x + 3 * y + 5 * z = 127.5 :=
sorry

end find_x_y_z_sum_l2260_226027


namespace sprinter_speed_l2260_226001

theorem sprinter_speed
  (distance : ℝ)
  (time : ℝ)
  (H1 : distance = 100)
  (H2 : time = 10) :
    (distance / time = 10) ∧
    ((distance / time) * 60 = 600) ∧
    (((distance / time) * 60 * 60) / 1000 = 36) :=
by
  sorry

end sprinter_speed_l2260_226001


namespace greatest_three_digit_multiple_of_17_l2260_226029

theorem greatest_three_digit_multiple_of_17 :
  ∃ n, n * 17 < 1000 ∧ ∀ m, m * 17 < 1000 → m ≤ n := by
  sorry

end greatest_three_digit_multiple_of_17_l2260_226029


namespace find_s_when_t_eq_5_l2260_226018

theorem find_s_when_t_eq_5 (s : ℝ) (h : 5 = 8 * s^2 + 2 * s) :
  s = (-1 + Real.sqrt 41) / 8 ∨ s = (-1 - Real.sqrt 41) / 8 :=
by sorry

end find_s_when_t_eq_5_l2260_226018


namespace geometric_sequence_properties_l2260_226005

noncomputable def geometric_sequence_sum (r a1 : ℝ) : Prop :=
  a1 * (r^3 + r^4) = 27 ∨ a1 * (r^3 + r^4) = -27

theorem geometric_sequence_properties (a1 r : ℝ) (h1 : a1 + a1 * r = 1) (h2 : a1 * r^2 + a1 * r^3 = 9) :
  geometric_sequence_sum r a1 :=
sorry

end geometric_sequence_properties_l2260_226005


namespace find_first_number_l2260_226041

theorem find_first_number (HCF LCM num2 num1 : ℕ) (hcf_cond : HCF = 20) (lcm_cond : LCM = 396) (num2_cond : num2 = 220) 
    (relation_cond : HCF * LCM = num1 * num2) : num1 = 36 :=
by
  sorry

end find_first_number_l2260_226041


namespace evaluate_expression_l2260_226080

theorem evaluate_expression : 1234562 - (12 * 3 * (2 + 7)) = 1234238 :=
by 
  sorry

end evaluate_expression_l2260_226080


namespace cube_removal_minimum_l2260_226055

theorem cube_removal_minimum (l w h : ℕ) (hu : l = 4) (hv : w = 5) (hw : h = 6) :
  ∃ num_cubes_removed : ℕ, 
    (l * w * h - num_cubes_removed = 4 * 4 * 4) ∧ 
    num_cubes_removed = 56 := 
by
  sorry

end cube_removal_minimum_l2260_226055


namespace mixture_percentage_l2260_226031

variable (P : ℝ)
variable (x_ryegrass_percent : ℝ := 0.40)
variable (y_ryegrass_percent : ℝ := 0.25)
variable (final_mixture_ryegrass_percent : ℝ := 0.32)

theorem mixture_percentage (h : 0.40 * P + 0.25 * (1 - P) = 0.32) : P = 0.07 / 0.15 := by
  sorry

end mixture_percentage_l2260_226031


namespace incorrect_statement_c_l2260_226063

-- Definitions based on conditions
variable (p q : Prop)

-- Lean 4 statement to check the logical proposition
theorem incorrect_statement_c (h : ¬(p ∧ q)) : ¬p ∨ ¬q :=
by
  sorry

end incorrect_statement_c_l2260_226063


namespace car_cost_difference_l2260_226070

-- Definitions based on the problem's conditions
def car_cost_ratio (C A : ℝ) := C / A = 3 / 2
def ac_cost := 1500

-- Theorem statement that needs proving
theorem car_cost_difference (C A : ℝ) (h1 : car_cost_ratio C A) (h2 : A = ac_cost) : C - A = 750 := 
by sorry

end car_cost_difference_l2260_226070


namespace ellipse_foci_distance_sum_l2260_226044

theorem ellipse_foci_distance_sum
    (x y : ℝ)
    (PF1 PF2 : ℝ)
    (a : ℝ)
    (h_ellipse : (x^2 / 36) + (y^2 / 16) = 1)
    (h_foci : ∀F1 F2, ∃e > 0, F1 = (e, 0) ∧ F2 = (-e, 0))
    (h_point_on_ellipse : ∀x y, (x^2 / 36) + (y^2 / 16) = 1 → (x, y) = (PF1, PF2))
    (h_semi_major_axis : a = 6):
    |PF1| + |PF2| = 12 := 
by
  sorry

end ellipse_foci_distance_sum_l2260_226044


namespace partA_l2260_226090

theorem partA (n : ℕ) : 
  1 < (n + 1 / 2) * Real.log (1 + 1 / n) ∧ (n + 1 / 2) * Real.log (1 + 1 / n) < 1 + 1 / (12 * n * (n + 1)) := 
sorry

end partA_l2260_226090


namespace determine_A_l2260_226068

theorem determine_A (A M C : ℕ) (h1 : A < 10) (h2 : M < 10) (h3 : C < 10) 
(h4 : (100 * A + 10 * M + C) * (A + M + C) = 2244) : A = 3 :=
sorry

end determine_A_l2260_226068


namespace average_weight_proof_l2260_226089

variables (W_A W_B W_C W_D W_E : ℝ)

noncomputable def final_average_weight (W_A W_B W_C W_D W_E : ℝ) : ℝ := (W_B + W_C + W_D + W_E) / 4

theorem average_weight_proof
  (h1 : (W_A + W_B + W_C) / 3 = 84)
  (h2 : W_A = 77)
  (h3 : (W_A + W_B + W_C + W_D) / 4 = 80)
  (h4 : W_E = W_D + 5) :
  final_average_weight W_A W_B W_C W_D W_E = 97.25 :=
by
  sorry

end average_weight_proof_l2260_226089


namespace factory_employees_l2260_226033

def num_employees (n12 n14 n17 : ℕ) : ℕ := n12 + n14 + n17

def total_cost (n12 n14 n17 : ℕ) : ℕ := 
    (200 * 12 * 8) + (40 * 14 * 8) + (n17 * 17 * 8)

theorem factory_employees (n17 : ℕ) 
    (h_cost : total_cost 200 40 n17 = 31840) : 
    num_employees 200 40 n17 = 300 := 
by 
    sorry

end factory_employees_l2260_226033


namespace ratio_of_x_and_y_l2260_226083

theorem ratio_of_x_and_y {x y a b : ℝ} (h1 : (2 * a - x) / (3 * b - y) = 3) (h2 : a / b = 4.5) : x / y = 3 :=
sorry

end ratio_of_x_and_y_l2260_226083


namespace find_fourth_term_geometric_progression_l2260_226062

theorem find_fourth_term_geometric_progression (x : ℝ) (a1 a2 a3 : ℝ) (r : ℝ)
  (h1 : a1 = x)
  (h2 : a2 = 3 * x + 6)
  (h3 : a3 = 7 * x + 21)
  (h4 : ∃ r, a2 / a1 = r ∧ a3 / a2 = r)
  (hx : x = 3 / 2) :
  7 * (7 * x + 21) = 220.5 :=
by
  sorry

end find_fourth_term_geometric_progression_l2260_226062


namespace distance_sum_identity_l2260_226094

noncomputable def squared_distance (p1 p2 : (ℝ × ℝ)) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

theorem distance_sum_identity
  (a b c x y : ℝ)
  (A B C P G : ℝ × ℝ)
  (hA : A = (a, b))
  (hB : B = (-c, 0))
  (hC : C = (c, 0))
  (hG : G = (a / 3, b / 3))
  (hP : P = (x, y))
  (hG_centroid : G = ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)) :
  squared_distance A P + squared_distance B P + squared_distance C P =
  squared_distance A G + squared_distance B G + squared_distance C G + 3 * squared_distance G P :=
by sorry

end distance_sum_identity_l2260_226094


namespace area_triangle_DEF_l2260_226095

theorem area_triangle_DEF 
  (DE EL EF : ℝ) (H1 : DE = 15) (H2 : EL = 12) (H3 : EF = 20) 
  (DL : ℝ) (H4 : DE^2 = EL^2 + DL^2) (H5 : DL * EF = DL * 20) :
  1/2 * EF * DL = 90 :=
by
  -- Use the assumptions and conditions to state the theorem.
  sorry

end area_triangle_DEF_l2260_226095


namespace inverse_function_l2260_226081

theorem inverse_function (x : ℝ) (hx : x > 1) : ∃ y : ℝ, x = 2^y + 1 ∧ y = Real.logb 2 (x - 1) :=
sorry

end inverse_function_l2260_226081


namespace tetrahedron_volume_le_one_eight_l2260_226034

theorem tetrahedron_volume_le_one_eight {A B C D : Type} 
  (e₁_AB e₂_AC e₃_AD e₄_BC e₅_BD : ℝ) (h₁ : e₁_AB ≤ 1) (h₂ : e₂_AC ≤ 1) (h₃ : e₃_AD ≤ 1)
  (h₄ : e₄_BC ≤ 1) (h₅ : e₅_BD ≤ 1) : 
  ∃ (vol : ℝ), vol ≤ 1 / 8 :=
sorry

end tetrahedron_volume_le_one_eight_l2260_226034


namespace negation_example_l2260_226019

theorem negation_example : (¬ (∀ x : ℝ, x^2 ≥ 0)) ↔ (∃ x : ℝ, x^2 < 0) :=
by
  sorry

end negation_example_l2260_226019


namespace min_value_l2260_226045

theorem min_value (m n : ℝ) (h1 : 2 * m + n = 1) (h2 : m > 0) (h3 : n > 0) :
  ∃ x, x = 3 + 2 * Real.sqrt 2 ∧ (∀ y, (2 * m + n = 1 → m > 0 → n > 0 → y = (1 / m) + (1 / n) → y ≥ x)) :=
by
  sorry

end min_value_l2260_226045


namespace trigonometric_identity_l2260_226060

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = 3) :
  (1 - Real.cos θ) / Real.sin θ - Real.sin θ / (1 + Real.cos θ) = 0 := 
by 
  sorry

end trigonometric_identity_l2260_226060


namespace line_canonical_form_l2260_226040

theorem line_canonical_form :
  ∃ (x y z : ℝ),
  x + y + z - 2 = 0 ∧
  x - y - 2 * z + 2 = 0 →
  ∃ (k : ℝ),
  x / k = -1 ∧
  (y - 2) / (3 * k) = 1 ∧
  z / (-2 * k) = 1 :=
sorry

end line_canonical_form_l2260_226040


namespace first_mission_days_l2260_226000

-- Definitions
variable (x : ℝ) (extended_first_mission : ℝ) (second_mission : ℝ) (total_mission_time : ℝ)

axiom h1 : extended_first_mission = 1.60 * x
axiom h2 : second_mission = 3
axiom h3 : total_mission_time = 11
axiom h4 : extended_first_mission + second_mission = total_mission_time

-- Theorem to prove
theorem first_mission_days : x = 5 :=
by
  sorry

end first_mission_days_l2260_226000


namespace ring_toss_total_l2260_226009

theorem ring_toss_total (money_per_day : ℕ) (days : ℕ) (total_money : ℕ) 
(h1 : money_per_day = 140) (h2 : days = 3) : total_money = 420 :=
by
  sorry

end ring_toss_total_l2260_226009


namespace probability_different_colors_l2260_226071

-- Define the total number of blue and yellow chips
def blue_chips : ℕ := 5
def yellow_chips : ℕ := 7
def total_chips : ℕ := blue_chips + yellow_chips

-- Define the probability of drawing a blue chip and a yellow chip
def prob_blue : ℚ := blue_chips / total_chips
def prob_yellow : ℚ := yellow_chips / total_chips

-- Define the probability of drawing two chips of different colors
def prob_different_colors := 2 * (prob_blue * prob_yellow)

theorem probability_different_colors :
  prob_different_colors = (35 / 72) := by
  sorry

end probability_different_colors_l2260_226071


namespace sum_cubes_coeffs_l2260_226002

theorem sum_cubes_coeffs :
  ∃ a b c d e : ℤ, 
  (1000 * x^3 + 27 = (a * x + b) * (c * x^2 + d * x + e)) ∧ 
  (a + b + c + d + e = 92) :=
sorry

end sum_cubes_coeffs_l2260_226002


namespace leah_coins_value_l2260_226092

theorem leah_coins_value
  (p n : ℕ)
  (h₁ : n + p = 15)
  (h₂ : n + 2 = p) : p + 5 * n = 38 :=
by
  -- definitions used in converting conditions
  sorry

end leah_coins_value_l2260_226092


namespace sqrt_sum_simplify_l2260_226077

theorem sqrt_sum_simplify : (Real.sqrt 72 + Real.sqrt 32) = 10 * Real.sqrt 2 :=
by sorry

end sqrt_sum_simplify_l2260_226077


namespace find_x_y_sum_l2260_226015

variable {x y : ℝ}

theorem find_x_y_sum (h₁ : (x-1)^3 + 1997 * (x-1) = -1) (h₂ : (y-1)^3 + 1997 * (y-1) = 1) : 
  x + y = 2 := 
by
  sorry

end find_x_y_sum_l2260_226015


namespace find_number_l2260_226021

theorem find_number 
  (x : ℝ) 
  (h1 : 3 * (2 * x + 9) = 69) : x = 7 := by
  sorry

end find_number_l2260_226021


namespace hyperbola_eccentricity_l2260_226047

def isHyperbolaWithEccentricity (e : ℝ) : Prop :=
  ∃ (a b : ℝ), a = 4 * b ∧ e = (Real.sqrt (a^2 + b^2)) / a

theorem hyperbola_eccentricity : isHyperbolaWithEccentricity (Real.sqrt 17 / 4) :=
sorry

end hyperbola_eccentricity_l2260_226047


namespace cone_fits_in_cube_l2260_226006

noncomputable def height_cone : ℝ := 15
noncomputable def diameter_cone_base : ℝ := 8
noncomputable def side_length_cube : ℝ := 15
noncomputable def volume_cube : ℝ := side_length_cube ^ 3

theorem cone_fits_in_cube :
  (height_cone = 15) →
  (diameter_cone_base = 8) →
  (height_cone ≤ side_length_cube ∧ diameter_cone_base ≤ side_length_cube) →
  volume_cube = 3375 := by
  intros h_cone d_base fits
  sorry

end cone_fits_in_cube_l2260_226006


namespace intersection_is_isosceles_right_angled_l2260_226069

def is_isosceles_triangle (x : Type) : Prop := sorry -- Definition of isosceles triangle
def is_right_angled_triangle (x : Type) : Prop := sorry -- Definition of right-angled triangle

def M : Set Type := {x | is_isosceles_triangle x}
def N : Set Type := {x | is_right_angled_triangle x}

theorem intersection_is_isosceles_right_angled :
  (M ∩ N) = {x | is_isosceles_triangle x ∧ is_right_angled_triangle x} := by
  sorry

end intersection_is_isosceles_right_angled_l2260_226069


namespace option_d_may_not_hold_l2260_226036

theorem option_d_may_not_hold (a b : ℝ) (m : ℝ) (h : a < b) : ¬ (m^2 * a > m^2 * b) :=
sorry

end option_d_may_not_hold_l2260_226036


namespace rook_attack_expectation_correct_l2260_226088

open ProbabilityTheory

noncomputable def rook_attack_expectation : ℝ := sorry

theorem rook_attack_expectation_correct :
  rook_attack_expectation = 35.33 := sorry

end rook_attack_expectation_correct_l2260_226088


namespace quadratic_inequality_range_l2260_226052

variable (x : ℝ)

-- Statement of the mathematical problem
theorem quadratic_inequality_range (h : ¬ (x^2 - 5 * x + 4 > 0)) : 1 ≤ x ∧ x ≤ 4 :=
by
  sorry

end quadratic_inequality_range_l2260_226052


namespace distinct_ways_to_divide_books_l2260_226014

theorem distinct_ways_to_divide_books : 
  ∃ (ways : ℕ), ways = 5 := sorry

end distinct_ways_to_divide_books_l2260_226014


namespace largest_four_digit_perfect_cube_is_9261_l2260_226099

-- Define the notion of a four-digit number and perfect cube
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000
def is_perfect_cube (n : ℕ) : Prop := ∃ (k : ℕ), k^3 = n

-- The main theorem statement
theorem largest_four_digit_perfect_cube_is_9261 :
  ∃ n, is_four_digit n ∧ is_perfect_cube n ∧ (∀ m, is_four_digit m ∧ is_perfect_cube m → m ≤ n) ∧ n = 9261 :=
sorry -- Proof is omitted

end largest_four_digit_perfect_cube_is_9261_l2260_226099


namespace exists_projectile_time_l2260_226013

noncomputable def projectile_time := 
  ∃ t1 t2 : ℝ, (-4.9 * t1^2 + 31 * t1 - 40 = 0) ∧ ((abs (t1 - 1.8051) < 0.001) ∨ (abs (t2 - 4.5319) < 0.001))

theorem exists_projectile_time : projectile_time := 
sorry

end exists_projectile_time_l2260_226013


namespace number_of_ways_to_sum_to_4_l2260_226076

-- Definitions deriving from conditions
def cards : List ℕ := [0, 1, 2, 3, 4]

-- Goal to prove
theorem number_of_ways_to_sum_to_4 : 
  let pairs := List.product cards cards
  let valid_pairs := pairs.filter (λ (x, y) => x + y = 4)
  List.length valid_pairs = 5 := 
by
  sorry

end number_of_ways_to_sum_to_4_l2260_226076
