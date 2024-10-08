import Mathlib

namespace max_mixed_gender_groups_l23_23314

theorem max_mixed_gender_groups (b g : ℕ) (h_b : b = 31) (h_g : g = 32) : 
  ∃ max_groups, max_groups = min (b / 2) (g / 3) :=
by
  use 10
  sorry

end max_mixed_gender_groups_l23_23314


namespace dice_probability_l23_23917

noncomputable def probability_event (event_count : ℕ) (total_count : ℕ) : ℚ := 
  event_count / total_count

theorem dice_probability :
  let event_first_die := 3
  let event_second_die := 3
  let total_outcomes_first := 8
  let total_outcomes_second := 8
  probability_event event_first_die total_outcomes_first * probability_event event_second_die total_outcomes_second = 9 / 64 :=
by
  sorry

end dice_probability_l23_23917


namespace picture_area_l23_23294

-- Given dimensions of the paper
def paper_width : ℝ := 8.5
def paper_length : ℝ := 10

-- Given margins
def margin : ℝ := 1.5

-- Calculated dimensions of the picture
def picture_width := paper_width - 2 * margin
def picture_length := paper_length - 2 * margin

-- Statement to prove
theorem picture_area : picture_width * picture_length = 38.5 := by
  -- skipped the proof
  sorry

end picture_area_l23_23294


namespace transylvanian_is_sane_human_l23_23354

def Transylvanian : Type := sorry -- Placeholder type for Transylvanian
def Human : Transylvanian → Prop := sorry
def Sane : Transylvanian → Prop := sorry
def InsaneVampire : Transylvanian → Prop := sorry

/-- The Transylvanian stated: "Either I am a human, or I am sane." -/
axiom statement (T : Transylvanian) : Human T ∨ Sane T

/-- Insane vampires only make true statements. -/
axiom insane_vampire_truth (T : Transylvanian) : InsaneVampire T → (Human T ∨ Sane T)

/-- Insane vampires cannot be sane or human. -/
axiom insane_vampire_condition (T : Transylvanian) : InsaneVampire T → ¬ Human T ∧ ¬ Sane T

theorem transylvanian_is_sane_human (T : Transylvanian) :
  ¬ (InsaneVampire T) → (Human T ∧ Sane T) := sorry

end transylvanian_is_sane_human_l23_23354


namespace jose_investment_proof_l23_23352

noncomputable def jose_investment (total_profit jose_share : ℕ) (tom_investment : ℕ) (months_tom months_jose : ℕ) : ℕ :=
  let tom_share := total_profit - jose_share
  let tom_investment_mr := tom_investment * months_tom
  let ratio := tom_share * months_jose
  tom_investment_mr * jose_share / ratio

theorem jose_investment_proof : 
  ∃ (jose_invested : ℕ), 
    let total_profit := 5400
    let jose_share := 3000
    let tom_invested := 3000
    let months_tom := 12
    let months_jose := 10
    jose_investment total_profit jose_share tom_invested months_tom months_jose = 4500 :=
by
  use 4500
  sorry

end jose_investment_proof_l23_23352


namespace shaded_region_perimeter_l23_23053

theorem shaded_region_perimeter (C : Real) (r : Real) (L : Real) (P : Real)
  (h0 : C = 48)
  (h1 : r = C / (2 * Real.pi))
  (h2 : L = (90 / 360) * C)
  (h3 : P = 3 * L) :
  P = 36 := by
  sorry

end shaded_region_perimeter_l23_23053


namespace polygon_sidedness_l23_23671

-- Define the condition: the sum of the interior angles of the polygon
def sum_of_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

-- Given condition
def given_condition : ℝ := 1260

-- Target proposition to prove
theorem polygon_sidedness (n : ℕ) (h : sum_of_interior_angles n = given_condition) : n = 9 :=
by
  sorry

end polygon_sidedness_l23_23671


namespace general_term_sequence_x_l23_23350

-- Definitions used in Lean statement corresponding to the conditions.
noncomputable def sequence_a (n : ℕ) : ℝ := sorry

noncomputable def sequence_x (n : ℕ) : ℝ := sorry

axiom condition_1 : ∀ n : ℕ, 
  ((sequence_a (n + 2))⁻¹ = ((sequence_a n)⁻¹ + (sequence_a (n + 1))⁻¹) / 2)

axiom condition_2 {n : ℕ} : sequence_x n > 0

axiom condition_3 : sequence_x 1 = 3

axiom condition_4 : sequence_x 1 + sequence_x 2 + sequence_x 3 = 39

axiom condition_5 (n : ℕ) : (sequence_x n)^(sequence_a n) = 
  (sequence_x (n + 1))^(sequence_a (n + 1)) ∧ 
  (sequence_x (n + 1))^(sequence_a (n + 1)) = 
  (sequence_x (n + 2))^(sequence_a (n + 2))

-- Theorem stating that the general term of sequence {x_n} is 3^n.
theorem general_term_sequence_x : ∀ n : ℕ, sequence_x n = 3^n :=
by
  sorry

end general_term_sequence_x_l23_23350


namespace value_of_expression_l23_23736

theorem value_of_expression (a b : ℝ) (h : a + b = 3) : 2 * a^2 + 4 * a * b + 2 * b^2 - 6 = 12 :=
by
  sorry

end value_of_expression_l23_23736


namespace pens_multiple_91_l23_23222

theorem pens_multiple_91 (S : ℕ) (P : ℕ) (total_pencils : ℕ) 
  (h1 : S = 91) (h2 : total_pencils = 910) (h3 : total_pencils % S = 0) :
  ∃ (x : ℕ), P = S * x :=
by 
  sorry

end pens_multiple_91_l23_23222


namespace alice_sold_20_pears_l23_23814

variables (S P C : ℝ)

theorem alice_sold_20_pears (h1 : C = 1.20 * P)
  (h2 : P = 0.50 * S)
  (h3 : S + P + C = 42) : S = 20 :=
by {
  -- mark the proof as incomplete with sorry
  sorry
}

end alice_sold_20_pears_l23_23814


namespace price_of_olives_l23_23108

theorem price_of_olives 
  (cherries_price : ℝ)
  (total_cost_with_discount : ℝ)
  (num_bags : ℕ)
  (discount : ℝ)
  (olives_price : ℝ) :
  cherries_price = 5 →
  total_cost_with_discount = 540 →
  num_bags = 50 →
  discount = 0.10 →
  (0.9 * (num_bags * cherries_price + num_bags * olives_price) = total_cost_with_discount) →
  olives_price = 7 :=
by
  intros h_cherries_price h_total_cost h_num_bags h_discount h_equation
  sorry

end price_of_olives_l23_23108


namespace women_ratio_l23_23664

theorem women_ratio (pop : ℕ) (w_retail : ℕ) (w_fraction : ℚ) (h_pop : pop = 6000000) (h_w_retail : w_retail = 1000000) (h_w_fraction : w_fraction = 1 / 3) : 
  (3000000 : ℚ) / (6000000 : ℚ) = 1 / 2 :=
by sorry

end women_ratio_l23_23664


namespace largest_integer_x_l23_23396

theorem largest_integer_x (x : ℤ) : (8:ℚ)/11 > (x:ℚ)/15 → x ≤ 10 :=
by
  intro h
  sorry

end largest_integer_x_l23_23396


namespace combined_perimeter_of_squares_l23_23835

theorem combined_perimeter_of_squares (p1 p2 : ℝ) (s1 s2 : ℝ) :
  p1 = 40 → p2 = 100 → 4 * s1 = p1 → 4 * s2 = p2 →
  (p1 + p2 - 2 * s1) = 120 :=
by
  intros h1 h2 h3 h4
  sorry

end combined_perimeter_of_squares_l23_23835


namespace correct_calculated_value_l23_23856

theorem correct_calculated_value (N : ℕ) (h : N ≠ 0) :
  N * 16 = 2048 * (N / 128) := by 
  sorry

end correct_calculated_value_l23_23856


namespace quadratic_passing_origin_l23_23746

theorem quadratic_passing_origin (a b c : ℝ) (h : a ≠ 0) :
  ((∀ x y : ℝ, x = 0 → y = 0 → y = a * x^2 + b * x + c) ↔ c = 0) := 
by
  sorry

end quadratic_passing_origin_l23_23746


namespace trains_cross_time_l23_23156

def speed_in_m_per_s (speed_in_km_per_hr : Float) : Float :=
  (speed_in_km_per_hr * 1000) / 3600

def relative_speed (speed1 : Float) (speed2 : Float) : Float :=
  speed1 + speed2

def total_distance (length1 : Float) (length2 : Float) : Float :=
  length1 + length2

def time_to_cross (total_dist : Float) (relative_spd : Float) : Float :=
  total_dist / relative_spd

theorem trains_cross_time 
  (length_train1 : Float := 270)
  (speed_train1 : Float := 120)
  (length_train2 : Float := 230.04)
  (speed_train2 : Float := 80) :
  time_to_cross (total_distance length_train1 length_train2) 
                (relative_speed (speed_in_m_per_s speed_train1) 
                                (speed_in_m_per_s speed_train2)) = 9 := 
by
  sorry

end trains_cross_time_l23_23156


namespace probability_of_purple_marble_l23_23277

theorem probability_of_purple_marble 
  (P_blue : ℝ) 
  (P_green : ℝ) 
  (P_purple : ℝ) 
  (h1 : P_blue = 0.25) 
  (h2 : P_green = 0.55) 
  (h3 : P_blue + P_green + P_purple = 1) 
  : P_purple = 0.20 := 
by 
  sorry

end probability_of_purple_marble_l23_23277


namespace water_evaporation_problem_l23_23486

theorem water_evaporation_problem 
  (W : ℝ) 
  (evaporation_rate : ℝ := 0.01) 
  (evaporation_days : ℝ := 20) 
  (total_evaporation : ℝ := evaporation_rate * evaporation_days) 
  (evaporation_percentage : ℝ := 0.02) 
  (evaporation_amount : ℝ := evaporation_percentage * W) :
  evaporation_amount = total_evaporation → W = 10 :=
by
  sorry

end water_evaporation_problem_l23_23486


namespace sixteen_is_sixtyfour_percent_l23_23338

theorem sixteen_is_sixtyfour_percent (x : ℝ) (h : 16 / x = 64 / 100) : x = 25 :=
by sorry

end sixteen_is_sixtyfour_percent_l23_23338


namespace value_of_six_prime_prime_l23_23693

-- Define the function q' 
def prime (q : ℝ) : ℝ := 3 * q - 3

-- Stating the main theorem we want to prove
theorem value_of_six_prime_prime : prime (prime 6) = 42 :=
by
  sorry

end value_of_six_prime_prime_l23_23693


namespace find_n_l23_23576

open Nat

-- Defining the production rates for conditions.
structure Production := 
  (workers : ℕ)
  (gadgets : ℕ)
  (gizmos : ℕ)
  (hours : ℕ)

def condition1 : Production := { workers := 150, gadgets := 450, gizmos := 300, hours := 1 }
def condition2 : Production := { workers := 100, gadgets := 400, gizmos := 500, hours := 2 }
def condition3 : Production := { workers := 75, gadgets := 900, gizmos := 900, hours := 4 }

-- Statement: Finding the value of n.
theorem find_n :
  (75 * ((condition2.gadgets / condition2.workers) * (condition3.hours / condition2.hours))) = 600 := by
  sorry

end find_n_l23_23576


namespace construct_angle_from_19_l23_23934

theorem construct_angle_from_19 (θ : ℝ) (h : θ = 19) : ∃ n : ℕ, (n * θ) % 360 = 75 :=
by
  -- Placeholder for the proof
  sorry

end construct_angle_from_19_l23_23934


namespace largest_possible_b_l23_23249

theorem largest_possible_b (a b c : ℕ) (h1 : 1 < c) (h2 : c ≤ b) (h3 : b < a) (h4 : a * b * c = 360) : b = 10 :=
sorry

end largest_possible_b_l23_23249


namespace expression_bounds_l23_23187

noncomputable def expression (p q r s : ℝ) : ℝ :=
  Real.sqrt (p^2 + (2 - q)^2) + Real.sqrt (q^2 + (2 - r)^2) +
  Real.sqrt (r^2 + (2 - s)^2) + Real.sqrt (s^2 + (2 - p)^2)

theorem expression_bounds (p q r s : ℝ) (hp : 0 ≤ p ∧ p ≤ 2) (hq : 0 ≤ q ∧ q ≤ 2)
  (hr : 0 ≤ r ∧ r ≤ 2) (hs : 0 ≤ s ∧ s ≤ 2) : 
  4 * Real.sqrt 2 ≤ expression p q r s ∧ expression p q r s ≤ 8 :=
by
  sorry

end expression_bounds_l23_23187


namespace jacob_current_age_l23_23353

theorem jacob_current_age 
  (M : ℕ) 
  (Drew_age : ℕ := M + 5) 
  (Peter_age : ℕ := Drew_age + 4) 
  (John_age : ℕ := 30) 
  (maya_age_eq : 2 * M = John_age) 
  (jacob_future_age : ℕ := Peter_age / 2) 
  (jacob_current_age_eq : ℕ := jacob_future_age - 2) : 
  jacob_current_age_eq = 11 := 
sorry

end jacob_current_age_l23_23353


namespace sin_add_arcsin_arctan_l23_23683

theorem sin_add_arcsin_arctan :
  let a := Real.arcsin (4 / 5)
  let b := Real.arctan (Real.sqrt 3)
  Real.sin (a + b) = (2 + 3 * Real.sqrt 3) / 10 :=
by
  let a := Real.arcsin (4 / 5)
  let b := Real.arctan (Real.sqrt 3)
  sorry

end sin_add_arcsin_arctan_l23_23683


namespace correct_group_l23_23099

def atomic_number (element : String) : Nat :=
  match element with
  | "Be" => 4
  | "C" => 6
  | "B" => 5
  | "Cl" => 17
  | "O" => 8
  | "Li" => 3
  | "Al" => 13
  | "S" => 16
  | "Si" => 14
  | "Mg" => 12
  | _ => 0

def is_descending (lst : List Nat) : Bool :=
  match lst with
  | [] => true
  | [x] => true
  | x :: y :: xs => if x > y then is_descending (y :: xs) else false

theorem correct_group : is_descending [atomic_number "Cl", atomic_number "O", atomic_number "Li"] = true ∧
                        is_descending [atomic_number "Be", atomic_number "C", atomic_number "B"] = false ∧
                        is_descending [atomic_number "Al", atomic_number "S", atomic_number "Si"] = false ∧
                        is_descending [atomic_number "C", atomic_number "S", atomic_number "Mg"] = false :=
by
  -- Prove the given theorem based on the atomic number function and is_descending condition
  sorry

end correct_group_l23_23099


namespace trigonometric_identity_l23_23799

theorem trigonometric_identity
  (x : ℝ)
  (h_tan : Real.tan x = -1/2) :
  Real.sin x ^ 2 + 3 * Real.sin x * Real.cos x - 1 = -2 :=
sorry

end trigonometric_identity_l23_23799


namespace part1_part2_l23_23246

open Real

def f (x : ℝ) : ℝ := |x - 5| - |x - 2|

theorem part1 (m : ℝ) : (∃ x : ℝ, f x ≤ m) ↔ m ≥ -3 := 
sorry

theorem part2 : {x : ℝ | x^2 - 8 * x + 15 + f x ≤ 0} = {x : ℝ | 5 - sqrt 3 ≤ x ∧ x ≤ 6} :=
sorry

end part1_part2_l23_23246


namespace proof_ab_greater_ac_l23_23473

theorem proof_ab_greater_ac (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) : 
  a * b > a * c :=
by sorry

end proof_ab_greater_ac_l23_23473


namespace new_average_weight_calculation_l23_23761

noncomputable def new_average_weight (total_weight : ℝ) (number_of_people : ℝ) : ℝ :=
  total_weight / number_of_people

theorem new_average_weight_calculation :
  let initial_people := 6
  let initial_avg_weight := 156
  let new_person_weight := 121
  (initial_people * initial_avg_weight + new_person_weight) / (initial_people + 1) = 151 := by
  sorry

end new_average_weight_calculation_l23_23761


namespace inscribed_circle_radius_l23_23176

noncomputable def side1 := 13
noncomputable def side2 := 13
noncomputable def side3 := 10
noncomputable def s := (side1 + side2 + side3) / 2
noncomputable def area := Real.sqrt (s * (s - side1) * (s - side2) * (s - side3))
noncomputable def inradius := area / s

theorem inscribed_circle_radius :
  inradius = 10 / 3 :=
by
  sorry

end inscribed_circle_radius_l23_23176


namespace cauchy_schwarz_equivalent_iag_l23_23204

theorem cauchy_schwarz_equivalent_iag (a b c d : ℝ) :
  (∀ x y : ℝ, 0 ≤ x → 0 ≤ y → (Real.sqrt x * Real.sqrt y) ≤ (x + y) / 2) ↔
  ((a * c + b * d) ^ 2 ≤ (a ^ 2 + b ^ 2) * (c ^ 2 + d ^ 2)) := by
  sorry

end cauchy_schwarz_equivalent_iag_l23_23204


namespace find_2023rd_letter_l23_23497

def seq : List Char := ['A', 'B', 'C', 'D', 'D', 'C', 'B', 'A']

theorem find_2023rd_letter : seq.get! ((2023 % seq.length) - 1) = 'B' :=
by
  sorry

end find_2023rd_letter_l23_23497


namespace tractors_moved_l23_23758

-- Define initial conditions
def total_area (tractors: ℕ) (days: ℕ) (hectares_per_day: ℕ) := tractors * days * hectares_per_day

theorem tractors_moved (original_tractors remaining_tractors: ℕ)
  (days_original: ℕ) (hectares_per_day_original: ℕ)
  (days_remaining: ℕ) (hectares_per_day_remaining: ℕ)
  (total_area_original: ℕ) 
  (h1: total_area original_tractors days_original hectares_per_day_original = total_area_original)
  (h2: total_area remaining_tractors days_remaining hectares_per_day_remaining = total_area_original) :
  original_tractors - remaining_tractors = 2 :=
by
  sorry

end tractors_moved_l23_23758


namespace perimeter_of_square_land_is_36_diagonal_of_square_land_is_27_33_l23_23500

def square_land (A P D : ℝ) :=
  (5 * A = 10 * P + 45) ∧
  (3 * D = 2 * P + 10)

theorem perimeter_of_square_land_is_36 (A P D : ℝ) (h1 : 5 * A = 10 * P + 45) (h2 : 3 * D = 2 * P + 10) :
  P = 36 :=
sorry

theorem diagonal_of_square_land_is_27_33 (A P D : ℝ) (h1 : P = 36) (h2 : 3 * D = 2 * P + 10) :
  D = 82 / 3 :=
sorry

end perimeter_of_square_land_is_36_diagonal_of_square_land_is_27_33_l23_23500


namespace present_cost_after_two_years_l23_23619

-- Defining variables and constants
def initial_cost : ℝ := 75
def inflation_rate : ℝ := 0.05
def first_year_increase1 : ℝ := 0.20
def first_year_decrease1 : ℝ := 0.20
def second_year_increase2 : ℝ := 0.30
def second_year_decrease2 : ℝ := 0.25

theorem present_cost_after_two_years : presents_cost = 77.40 :=
by
  let adjusted_initial_cost := initial_cost + (initial_cost * inflation_rate)
  let increased_cost_year1 := adjusted_initial_cost + (adjusted_initial_cost * first_year_increase1)
  let decreased_cost_year1 := increased_cost_year1 - (increased_cost_year1 * first_year_decrease1)
  let adjusted_cost_year1 := decreased_cost_year1 + (decreased_cost_year1 * inflation_rate)
  let increased_cost_year2 := adjusted_cost_year1 + (adjusted_cost_year1 * second_year_increase2)
  let decreased_cost_year2 := increased_cost_year2 - (increased_cost_year2 * second_year_decrease2)
  let presents_cost := decreased_cost_year2
  have h := (presents_cost : ℝ)
  have h := presents_cost
  sorry

end present_cost_after_two_years_l23_23619


namespace car_trader_profit_l23_23616

theorem car_trader_profit (P : ℝ) : 
  let purchase_price := 0.80 * P
  let selling_price := 1.28000000000000004 * P
  let profit := selling_price - purchase_price
  let percentage_increase := (profit / purchase_price) * 100
  percentage_increase = 60 := 
by
  sorry

end car_trader_profit_l23_23616


namespace profit_percent_l23_23228

-- Definitions for the given conditions
variables (P C : ℝ)
-- Condition given: selling at (2/3) of P results in a loss of 5%, i.e., (2/3) * P = 0.95 * C
def condition : Prop := (2 / 3) * P = 0.95 * C

-- Theorem statement: Given the condition, the profit percent when selling at price P is 42.5%
theorem profit_percent (h : condition P C) : ((P - C) / C) * 100 = 42.5 :=
sorry

end profit_percent_l23_23228


namespace monotonic_decreasing_fx_l23_23800

open Real

noncomputable def f (x : ℝ) : ℝ := x * log x

theorem monotonic_decreasing_fx : ∀ (x : ℝ), (0 < x) ∧ (x < (1 / exp 1)) → deriv f x < 0 := 
by
  sorry

end monotonic_decreasing_fx_l23_23800


namespace applicants_majored_in_political_science_l23_23877

theorem applicants_majored_in_political_science
  (total_applicants : ℕ)
  (gpa_above_3 : ℕ)
  (non_political_science_and_gpa_leq_3 : ℕ)
  (political_science_and_gpa_above_3 : ℕ) :
  total_applicants = 40 →
  gpa_above_3 = 20 →
  non_political_science_and_gpa_leq_3 = 10 →
  political_science_and_gpa_above_3 = 5 →
  ∃ P : ℕ, P = 15 :=
by
  intros
  sorry

end applicants_majored_in_political_science_l23_23877


namespace zhang_san_not_losing_probability_l23_23191

theorem zhang_san_not_losing_probability (p_win p_draw : ℚ) (h_win : p_win = 1 / 3) (h_draw : p_draw = 1 / 4) : 
  p_win + p_draw = 7 / 12 := by
  sorry

end zhang_san_not_losing_probability_l23_23191


namespace compute_value_of_fractions_l23_23695

theorem compute_value_of_fractions (a b c : ℝ) 
  (h1 : (ac / (a + b)) + (ba / (b + c)) + (cb / (c + a)) = 0)
  (h2 : (bc / (a + b)) + (ca / (b + c)) + (ab / (c + a)) = 1) :
  (b / (a + b)) + (c / (b + c)) + (a / (c + a)) = 5 / 2 :=
sorry

end compute_value_of_fractions_l23_23695


namespace inequality_solution_l23_23822

theorem inequality_solution (x : ℝ) :
  (∃ x, 2 < x ∧ x < 3) ↔ ∃ x, (x-2)*(x-3)/(x^2 + 1) < 0 := by
  sorry

end inequality_solution_l23_23822


namespace M_inter_N_eq_l23_23942

-- Definitions based on the problem conditions
def M : Set ℝ := { x | abs x ≥ 3 }
def N : Set ℝ := { y | ∃ x ∈ M, y = x^2 }

-- The statement we want to prove
theorem M_inter_N_eq : M ∩ N = { x : ℝ | x ≥ 3 } :=
by
  sorry

end M_inter_N_eq_l23_23942


namespace border_area_is_72_l23_23592

def livingRoomLength : ℝ := 12
def livingRoomWidth : ℝ := 10
def borderWidth : ℝ := 2

def livingRoomArea : ℝ := livingRoomLength * livingRoomWidth
def carpetLength : ℝ := livingRoomLength - 2 * borderWidth
def carpetWidth : ℝ := livingRoomWidth - 2 * borderWidth
def carpetArea : ℝ := carpetLength * carpetWidth
def borderArea : ℝ := livingRoomArea - carpetArea

theorem border_area_is_72 : borderArea = 72 := 
by
  sorry

end border_area_is_72_l23_23592


namespace find_k_l23_23579

-- Define the problem parameters
variables {x y k : ℝ}

-- The conditions given in the problem
def system_of_equations (x y k : ℝ) : Prop :=
  (x + 2 * y = k - 1) ∧ (2 * x + y = 5 * k + 4)

def solution_condition (x y : ℝ) : Prop :=
  x + y = 5

-- The proof statement
theorem find_k (x y k : ℝ) (h1 : system_of_equations x y k) (h2 : solution_condition x y) :
  k = 2 :=
sorry

end find_k_l23_23579


namespace typing_time_l23_23715

def original_speed : ℕ := 212
def reduction : ℕ := 40
def new_speed : ℕ := original_speed - reduction
def document_length : ℕ := 3440
def required_time : ℕ := 20

theorem typing_time :
  document_length / new_speed = required_time :=
by
  sorry

end typing_time_l23_23715


namespace find_f_2018_l23_23915

-- Define the function f, its periodicity and even property
variable (f : ℝ → ℝ)

-- Conditions
axiom f_periodicity : ∀ x : ℝ, f (x + 4) = -f x
axiom f_symmetric : ∀ x : ℝ, f x = f (-x)
axiom f_at_two : f 2 = 2

-- Theorem stating the desired property
theorem find_f_2018 : f 2018 = 2 :=
  sorry

end find_f_2018_l23_23915


namespace algebraic_expression_value_l23_23454

variable {a b c : ℝ}

theorem algebraic_expression_value
  (h1 : (a + b) * (b + c) * (c + a) = 0)
  (h2 : a * b * c < 0) :
  (a / |a|) + (b / |b|) + (c / |c|) = 1 := by
  sorry

end algebraic_expression_value_l23_23454


namespace inequality_b_does_not_hold_l23_23737

theorem inequality_b_does_not_hold (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : ¬(a + d > b + c) ↔ a + d ≤ b + c :=
by
  -- We only need the statement, so we add sorry at the end
  sorry

end inequality_b_does_not_hold_l23_23737


namespace minimum_wins_l23_23850

theorem minimum_wins (x y : ℕ) (h_score : 3 * x + y = 10) (h_games : x + y ≤ 7) (h_bounds : 0 < x ∧ x < 4) : x = 2 :=
by
  sorry

end minimum_wins_l23_23850


namespace average_marks_l23_23254

theorem average_marks :
  let a1 := 76
  let a2 := 65
  let a3 := 82
  let a4 := 67
  let a5 := 75
  let n := 5
  let total_marks := a1 + a2 + a3 + a4 + a5
  let avg_marks := total_marks / n
  avg_marks = 73 :=
by
  sorry

end average_marks_l23_23254


namespace travis_flight_cost_l23_23356

theorem travis_flight_cost 
  (cost_leg1 : ℕ := 1500) 
  (cost_leg2 : ℕ := 1000) 
  (discount_leg1 : ℕ := 25) 
  (discount_leg2 : ℕ := 35) : 
  cost_leg1 - (discount_leg1 * cost_leg1 / 100) + cost_leg2 - (discount_leg2 * cost_leg2 / 100) = 1775 :=
by
  sorry

end travis_flight_cost_l23_23356


namespace sum_of_a_b_l23_23219

theorem sum_of_a_b (a b : ℝ) (h1 : ∀ x : ℝ, (a * (b * x + a) + b = x))
  (h2 : ∀ y : ℝ, (b * (a * y + b) + a = y)) : a + b = -2 := 
sorry

end sum_of_a_b_l23_23219


namespace first_cyclist_speed_l23_23583

theorem first_cyclist_speed (v₁ v₂ : ℕ) (c t : ℕ) 
  (h1 : v₂ = 8) 
  (h2 : c = 675) 
  (h3 : t = 45) 
  (h4 : v₁ * t + v₂ * t = c) : 
  v₁ = 7 :=
by {
  sorry
}

end first_cyclist_speed_l23_23583


namespace find_train_speed_l23_23784

variable (L V : ℝ)

-- Conditions
def condition1 := V = L / 10
def condition2 := V = (L + 600) / 30

-- Theorem statement
theorem find_train_speed (h1 : condition1 L V) (h2 : condition2 L V) : V = 30 :=
by
  sorry

end find_train_speed_l23_23784


namespace paving_stones_needed_l23_23932

def length_courtyard : ℝ := 60
def width_courtyard : ℝ := 14
def width_stone : ℝ := 2
def paving_stones_required : ℕ := 140

theorem paving_stones_needed (L : ℝ) 
  (h1 : length_courtyard * width_courtyard = 840) 
  (h2 : paving_stones_required = 140)
  (h3 : (140 * (L * 2)) = 840) : 
  (length_courtyard * width_courtyard) / (L * width_stone) = 140 := 
by sorry

end paving_stones_needed_l23_23932


namespace number_of_men_in_first_group_l23_23024

-- Define the conditions and the proof problem
theorem number_of_men_in_first_group (M : ℕ) 
  (h1 : ∀ t : ℝ, 22 * t = M) 
  (h2 : ∀ t' : ℝ, 18 * 17.11111111111111 = t') :
  M = 14 := 
by
  sorry

end number_of_men_in_first_group_l23_23024


namespace students_passed_both_tests_l23_23476

theorem students_passed_both_tests
    (total_students : ℕ)
    (passed_long_jump : ℕ)
    (passed_shot_put : ℕ)
    (failed_both : ℕ)
    (h_total : total_students = 50)
    (h_long_jump : passed_long_jump = 40)
    (h_shot_put : passed_shot_put = 31)
    (h_failed_both : failed_both = 4) : 
    (total_students - failed_both = passed_long_jump + passed_shot_put - 25) :=
by 
  sorry

end students_passed_both_tests_l23_23476


namespace louie_share_of_pie_l23_23555

def fraction_of_pie_taken_home (total_pie : ℚ) (shares : ℚ) : ℚ :=
  2 * (total_pie / shares)

theorem louie_share_of_pie : fraction_of_pie_taken_home (8 / 9) 4 = 4 / 9 := 
by 
  sorry

end louie_share_of_pie_l23_23555


namespace blue_bird_high_school_team_arrangement_l23_23312

theorem blue_bird_high_school_team_arrangement : 
  let girls := 2
  let boys := 3
  let girls_permutations := Nat.factorial girls
  let boys_permutations := Nat.factorial boys
  girls_permutations * boys_permutations = 12 := by
  sorry

end blue_bird_high_school_team_arrangement_l23_23312


namespace solution_interval_l23_23477

def check_solution (b : ℝ) (x : ℝ) : ℝ :=
  x^2 - b * x - 5

theorem solution_interval (b x : ℝ) :
  (check_solution b (-2) = 5) ∧
  (check_solution b (-1) = -1) ∧
  (check_solution b (4) = -1) ∧
  (check_solution b (5) = 5) →
  (∃ x, -2 < x ∧ x < -1 ∧ check_solution b x = 0) ∨
  (∃ x, 4 < x ∧ x < 5 ∧ check_solution b x = 0) :=
by
  sorry

end solution_interval_l23_23477


namespace A_alone_days_l23_23528

noncomputable def days_for_A (r_A r_B r_C : ℝ) : ℝ :=
  1 / r_A

theorem A_alone_days
  (r_A r_B r_C : ℝ) 
  (h1 : r_A + r_B = 1 / 3)
  (h2 : r_B + r_C = 1 / 6)
  (h3 : r_A + r_C = 1 / 4) :
  days_for_A r_A r_B r_C = 4.8 := by
  sorry

end A_alone_days_l23_23528


namespace slope_range_l23_23672

theorem slope_range (a b : ℝ) (h₁ : a ≠ -2) (h₂ : a ≠ 2) 
  (h₃ : a^2 / 4 + b^2 / 3 = 1) (h₄ : -2 ≤ b / (a - 2) ∧ b / (a - 2) ≤ -1) :
  (3 / 8 ≤ b / (a + 2) ∧ b / (a + 2) ≤ 3 / 4) :=
sorry

end slope_range_l23_23672


namespace greatest_three_digit_number_l23_23144

theorem greatest_three_digit_number : ∃ n : ℕ, n < 1000 ∧ n >= 100 ∧ (n + 1) % 8 = 0 ∧ (n - 4) % 7 = 0 ∧ n = 967 :=
by
  sorry

end greatest_three_digit_number_l23_23144


namespace cos_double_angle_l23_23568

theorem cos_double_angle (θ : ℝ) (h : Real.cos θ = 1 / 3) : Real.cos (2 * θ) = -7 / 9 :=
by
  sorry

end cos_double_angle_l23_23568


namespace determine_a_l23_23243

theorem determine_a (a p q : ℚ) (h1 : p^2 = a) (h2 : 2 * p * q = 28) (h3 : q^2 = 9) : a = 196 / 9 :=
by
  sorry

end determine_a_l23_23243


namespace probability_of_meeting_l23_23101

noncomputable def meeting_probability : ℝ :=
  let total_area := 10 * 10
  let favorable_area := 51
  favorable_area / total_area

theorem probability_of_meeting : meeting_probability = 51 / 100 :=
by
  sorry

end probability_of_meeting_l23_23101


namespace perfect_square_trinomial_l23_23517

theorem perfect_square_trinomial (a b c : ℤ) (f : ℤ → ℤ) (h : ∀ x : ℤ, f x = a * x^2 + b * x + c) :
  ∃ d e : ℤ, ∀ x : ℤ, f x = (d * x + e) ^ 2 :=
sorry

end perfect_square_trinomial_l23_23517


namespace geometric_sequence_a_l23_23308

open Real

theorem geometric_sequence_a (a : ℝ) (r : ℝ) (h1 : 20 * r = a) (h2 : a * r = 5/4) (h3 : 0 < a) : a = 5 :=
by
  -- The proof would go here
  sorry

end geometric_sequence_a_l23_23308


namespace range_of_m_l23_23611

-- Define the quadratic function f
def f (a c x : ℝ) := a * x^2 - 2 * a * x + c

-- State the theorem
theorem range_of_m (a c : ℝ) (h : f a c 2017 < f a c (-2016)) (m : ℝ) 
  : f a c m ≤ f a c 0 → 0 ≤ m ∧ m ≤ 2 := sorry

end range_of_m_l23_23611


namespace rectangle_not_equal_118_l23_23192

theorem rectangle_not_equal_118 
  (a b : ℕ) (h₀ : a > 0) (h₁ : b > 0) (A : ℕ) (P : ℕ)
  (h₂ : A = a * b) (h₃ : P = 2 * (a + b)) :
  (a + 2) * (b + 2) - 2 ≠ 118 :=
sorry

end rectangle_not_equal_118_l23_23192


namespace clear_board_possible_l23_23979

def operation (board : Array (Array Nat)) (op_type : String) (index : Fin 8) : Array (Array Nat) :=
  match op_type with
  | "column" => board.map (λ row => row.modify index fun x => x - 1)
  | "row" => board.modify index fun row => row.map (λ x => 2 * x)
  | _ => board

def isZeroBoard (board : Array (Array Nat)) : Prop :=
  board.all (λ row => row.all (λ x => x = 0))

theorem clear_board_possible (initial_board : Array (Array Nat)) : 
  ∃ (ops : List (String × Fin 8)), 
    isZeroBoard (ops.foldl (λ b ⟨t, i⟩ => operation b t i) initial_board) :=
sorry

end clear_board_possible_l23_23979


namespace trig_identity_l23_23805

theorem trig_identity (α : ℝ) :
  4.10 * (Real.cos (45 * Real.pi / 180 - α)) ^ 2 
  - (Real.cos (60 * Real.pi / 180 + α)) ^ 2 
  - Real.cos (75 * Real.pi / 180) * Real.sin (75 * Real.pi / 180 - 2 * α) 
  = Real.sin (2 * α) := 
sorry

end trig_identity_l23_23805


namespace candies_left_to_share_l23_23792

def initial_candies : Nat := 100
def sibling_count : Nat := 3
def candies_per_sibling : Nat := 10
def candies_Josh_eats : Nat := 16

theorem candies_left_to_share :
  let candies_given_to_siblings := sibling_count * candies_per_sibling;
  let candies_after_siblings := initial_candies - candies_given_to_siblings;
  let candies_given_to_friend := candies_after_siblings / 2;
  let candies_after_friend := candies_after_siblings - candies_given_to_friend;
  let candies_after_Josh := candies_after_friend - candies_Josh_eats;
  candies_after_Josh = 19 :=
by
  sorry

end candies_left_to_share_l23_23792


namespace acute_angles_45_degrees_l23_23845

-- Assuming quadrilaterals ABCD and A'B'C'D' such that sides of each lie on 
-- the perpendicular bisectors of the sides of the other. We want to prove that
-- the acute angles of A'B'C'D' are 45 degrees.

def convex_quadrilateral (Q : Type) := 
  ∃ (A B C D : Q), True -- Placeholder for a more detailed convex quadrilateral structure

def perpendicular_bisector (S1 S2 T1 T2: Type) := 
  ∃ (M : Type), True -- Placeholder for a more detailed perpendicular bisector structure

theorem acute_angles_45_degrees
  (Q1 Q2 : Type)
  (h1 : convex_quadrilateral Q1)
  (h2 : convex_quadrilateral Q2)
  (perp1 : perpendicular_bisector Q1 Q1 Q2 Q2)
  (perp2 : perpendicular_bisector Q2 Q2 Q1 Q1) :
  ∀ (θ : ℝ), θ = 45 := 
by
  sorry

end acute_angles_45_degrees_l23_23845


namespace percentage_increase_from_1200_to_1680_is_40_l23_23348

theorem percentage_increase_from_1200_to_1680_is_40 :
  let initial_value := 1200
  let final_value := 1680
  let percentage_increase := ((final_value - initial_value) / initial_value) * 100
  percentage_increase = 40 := by
  let initial_value := 1200
  let final_value := 1680
  let percentage_increase := ((final_value - initial_value) / initial_value) * 100
  sorry

end percentage_increase_from_1200_to_1680_is_40_l23_23348


namespace solve_exponential_problem_l23_23415

noncomputable def satisfies_condition (a : ℝ) : Prop :=
  let max_value := if a > 1 then a^2 else a
  let min_value := if a > 1 then a else a^2
  max_value - min_value = a / 2

theorem solve_exponential_problem (a : ℝ) (hpos : a > 0) (hne1 : a ≠ 1) :
  satisfies_condition a ↔ (a = 1 / 2 ∨ a = 3 / 2) :=
sorry

end solve_exponential_problem_l23_23415


namespace current_books_l23_23080

def initial_books : ℕ := 743
def sold_instore_saturday : ℕ := 37
def sold_online_saturday : ℕ := 128
def sold_instore_sunday : ℕ := 2 * sold_instore_saturday
def sold_online_sunday : ℕ := sold_online_saturday + 34
def total_books_sold_saturday : ℕ := sold_instore_saturday + sold_online_saturday
def total_books_sold_sunday : ℕ := sold_instore_sunday + sold_online_sunday
def total_books_sold_weekend : ℕ := total_books_sold_saturday + total_books_sold_sunday
def books_received_shipment : ℕ := 160
def net_change_books : ℤ := books_received_shipment - total_books_sold_weekend

theorem current_books
  (initial_books : ℕ) 
  (sold_instore_saturday : ℕ) 
  (sold_online_saturday : ℕ) 
  (sold_instore_sunday : ℕ)
  (sold_online_sunday : ℕ)
  (total_books_sold_saturday : ℕ)
  (total_books_sold_sunday : ℕ)
  (total_books_sold_weekend : ℕ)
  (books_received_shipment : ℕ)
  (net_change_books : ℤ) : (initial_books - net_change_books) = 502 := 
by {
  sorry
}

end current_books_l23_23080


namespace cleaning_time_ratio_l23_23317

/-- 
Given that Lilly and Fiona together take a total of 480 minutes to clean a room and Fiona
was cleaning for 360 minutes, prove that the ratio of the time Lilly spent cleaning 
to the total time spent cleaning the room is 1:4.
-/
theorem cleaning_time_ratio (total_time minutes Fiona_time : ℕ) 
  (h1 : total_time = 480)
  (h2 : Fiona_time = 360) : 
  (total_time - Fiona_time) / total_time = 1 / 4 :=
by
  sorry

end cleaning_time_ratio_l23_23317


namespace isosceles_triangle_perimeter_l23_23283

theorem isosceles_triangle_perimeter (perimeter_eq_tri : ℕ) (side_eq_tri : ℕ) (base_iso_tri : ℕ) (perimeter_iso_tri : ℕ) 
  (h1 : perimeter_eq_tri = 60) 
  (h2 : side_eq_tri = perimeter_eq_tri / 3) 
  (h3 : base_iso_tri = 5)
  (h4 : perimeter_iso_tri = 2 * side_eq_tri + base_iso_tri) : 
  perimeter_iso_tri = 45 := by
  sorry

end isosceles_triangle_perimeter_l23_23283


namespace negation_proof_l23_23258

-- Definitions based on conditions
def atMostTwoSolutions (solutions : ℕ) : Prop := solutions ≤ 2
def atLeastThreeSolutions (solutions : ℕ) : Prop := solutions ≥ 3

-- Statement of the theorem
theorem negation_proof (solutions : ℕ) : atMostTwoSolutions solutions ↔ ¬ atLeastThreeSolutions solutions :=
by
  sorry

end negation_proof_l23_23258


namespace find_certain_number_l23_23787

theorem find_certain_number (x : ℝ) (h : ((x^4) * 3.456789)^10 = 10^20) : x = 10 :=
sorry

end find_certain_number_l23_23787


namespace carlos_marbles_l23_23304

theorem carlos_marbles :
  ∃ N : ℕ, N > 2 ∧
  (N % 6 = 2) ∧
  (N % 7 = 2) ∧
  (N % 8 = 2) ∧
  (N % 11 = 2) ∧
  N = 3698 :=
by
  sorry

end carlos_marbles_l23_23304


namespace total_rooms_to_paint_l23_23072

-- Definitions based on conditions
def hours_per_room : ℕ := 8
def rooms_already_painted : ℕ := 8
def hours_to_paint_rest : ℕ := 16

-- Theorem statement
theorem total_rooms_to_paint :
  rooms_already_painted + hours_to_paint_rest / hours_per_room = 10 :=
  sorry

end total_rooms_to_paint_l23_23072


namespace no_natural_number_solution_for_divisibility_by_2020_l23_23025

theorem no_natural_number_solution_for_divisibility_by_2020 :
  ¬ ∃ k : ℕ, (k^3 - 3 * k^2 + 2 * k + 2) % 2020 = 0 :=
sorry

end no_natural_number_solution_for_divisibility_by_2020_l23_23025


namespace switches_assembled_are_correct_l23_23776

-- Definitions based on conditions
def total_payment : ℕ := 4700
def first_worker_payment : ℕ := 2000
def second_worker_per_switch_time_min : ℕ := 4
def third_worker_less_payment : ℕ := 300
def overtime_hours : ℕ := 5
def total_minutes (hours : ℕ) : ℕ := hours * 60

-- Function to calculate total switches assembled
noncomputable def total_switches_assembled :=
  let second_worker_payment := (total_payment - first_worker_payment + third_worker_less_payment) / 2
  let third_worker_payment := second_worker_payment - third_worker_less_payment
  let rate_per_switch := second_worker_payment / (total_minutes overtime_hours / second_worker_per_switch_time_min)
  let first_worker_switches := first_worker_payment / rate_per_switch
  let second_worker_switches := total_minutes overtime_hours / second_worker_per_switch_time_min
  let third_worker_switches := third_worker_payment / rate_per_switch
  first_worker_switches + second_worker_switches + third_worker_switches

-- Lean 4 statement to prove the problem
theorem switches_assembled_are_correct : 
  total_switches_assembled = 235 := by
  sorry

end switches_assembled_are_correct_l23_23776


namespace selection_options_l23_23244

theorem selection_options (group1 : Fin 5) (group2 : Fin 4) : (group1.1 + group2.1 + 1 = 9) :=
sorry

end selection_options_l23_23244


namespace total_cost_l23_23829

def daily_rental_cost : ℝ := 25
def cost_per_mile : ℝ := 0.20
def duration_days : ℕ := 4
def distance_miles : ℕ := 400

theorem total_cost 
: (daily_rental_cost * duration_days + cost_per_mile * distance_miles) = 180 := 
by
  sorry

end total_cost_l23_23829


namespace not_enough_funds_to_buy_two_books_l23_23110

def storybook_cost : ℝ := 25.5
def sufficient_funds (amount : ℝ) : Prop := amount >= 50

theorem not_enough_funds_to_buy_two_books : ¬ sufficient_funds (2 * storybook_cost) :=
by
  sorry

end not_enough_funds_to_buy_two_books_l23_23110


namespace average_speed_l23_23546

theorem average_speed (D : ℝ) (hD : D > 0) :
  let t1 := (D / 3) / 80
  let t2 := (D / 3) / 15
  let t3 := (D / 3) / 48
  let total_time := t1 + t2 + t3
  let avg_speed := D / total_time
  avg_speed = 30 :=
by
  sorry

end average_speed_l23_23546


namespace tan_identity_l23_23226

theorem tan_identity (A B : ℝ) (hA : A = 30) (hB : B = 30) :
  (1 + Real.tan (A * Real.pi / 180)) * (1 + Real.tan (B * Real.pi / 180)) = (4 + 2 * Real.sqrt 3)/3 := by
  sorry

end tan_identity_l23_23226


namespace slower_pipe_filling_time_l23_23323

-- Definitions based on conditions
def faster_pipe_rate (S : ℝ) : ℝ := 3 * S
def combined_rate (S : ℝ) : ℝ := (faster_pipe_rate S) + S

-- Statement of what needs to be proved 
theorem slower_pipe_filling_time :
  (∀ S : ℝ, combined_rate S * 40 = 1) →
  ∃ t : ℝ, t = 160 :=
by
  intro h
  sorry

end slower_pipe_filling_time_l23_23323


namespace length_of_QR_l23_23438

theorem length_of_QR {P Q R N : Type} 
  (PQ PR QR : ℝ) (QN NR PN : ℝ)
  (h1 : PQ = 5)
  (h2 : PR = 10)
  (h3 : QN = 3 * NR)
  (h4 : PN = 6)
  (h5 : QR = QN + NR) :
  QR = 724 / 3 :=
by sorry

end length_of_QR_l23_23438


namespace boxes_contain_same_number_of_apples_l23_23163

theorem boxes_contain_same_number_of_apples (total_apples boxes : ℕ) (h1 : total_apples = 49) (h2 : boxes = 7) : 
  total_apples / boxes = 7 :=
by
  sorry

end boxes_contain_same_number_of_apples_l23_23163


namespace girls_more_than_boys_l23_23319

theorem girls_more_than_boys (total_students boys : ℕ) (h : total_students = 466) (b : boys = 127) (gt : total_students - boys > boys) :
  total_students - 2 * boys = 212 := by
  sorry

end girls_more_than_boys_l23_23319


namespace profit_margin_comparison_l23_23554

theorem profit_margin_comparison
    (cost_price_A : ℝ) (selling_price_A : ℝ)
    (cost_price_B : ℝ) (selling_price_B : ℝ)
    (h1 : cost_price_A = 1600)
    (h2 : selling_price_A = 0.9 * 2000)
    (h3 : cost_price_B = 320)
    (h4 : selling_price_B = 0.8 * 460) :
    ((selling_price_B - cost_price_B) / cost_price_B) > ((selling_price_A - cost_price_A) / cost_price_A) := 
by
    sorry

end profit_margin_comparison_l23_23554


namespace percentage_reduction_l23_23447

theorem percentage_reduction (original reduced : ℝ) (h_original : original = 253.25) (h_reduced : reduced = 195) : 
  ((original - reduced) / original) * 100 = 22.99 :=
by
  sorry

end percentage_reduction_l23_23447


namespace line_equation_l23_23113

theorem line_equation {x y : ℝ} (m b : ℝ) (h1 : m = 2) (h2 : b = -3) :
    (∃ (f : ℝ → ℝ), (∀ x, f x = m * x + b) ∧ (∀ x, 2 * x - f x - 3 = 0)) :=
by
  sorry

end line_equation_l23_23113


namespace equation_of_circle_passing_through_points_l23_23548

theorem equation_of_circle_passing_through_points :
  (∃ D E F : ℝ, 
    (∀ x y : ℝ, x = 0 → y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ x y : ℝ, x = 4 → y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ x y : ℝ, x = -1 → y = 1 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (x^2 + y^2 + D * x + E * y + F = 0 ↔ x^2 + y^2 - 4 * x - 6 * y = 0)) :=
sorry

end equation_of_circle_passing_through_points_l23_23548


namespace moses_percentage_l23_23371

theorem moses_percentage (P : ℝ) (T : ℝ) (E : ℝ) (total_amount : ℝ) (moses_more : ℝ)
  (h1 : total_amount = 50)
  (h2 : moses_more = 5)
  (h3 : T = E)
  (h4 : P / 100 * total_amount = E + moses_more)
  (h5 : 2 * E = (1 - P / 100) * total_amount) :
  P = 40 :=
by
  sorry

end moses_percentage_l23_23371


namespace moles_of_ammonia_formed_l23_23339

def reaction (n_koh n_nh4i n_nh3 : ℕ) := 
  n_koh + n_nh4i + n_nh3 

theorem moles_of_ammonia_formed (n_koh : ℕ) :
  reaction n_koh 3 3 = n_koh + 3 + 3 := 
sorry

end moles_of_ammonia_formed_l23_23339


namespace inequality_holds_infinitely_many_times_l23_23708

variable {a : ℕ → ℝ}

theorem inequality_holds_infinitely_many_times
    (h_pos : ∀ n, 0 < a n) :
    ∃ᶠ n in at_top, 1 + a n > a (n - 1) * 2^(1 / n) :=
sorry

end inequality_holds_infinitely_many_times_l23_23708


namespace atomic_weight_of_iodine_is_correct_l23_23552

noncomputable def atomic_weight_iodine (atomic_weight_nitrogen : ℝ) (atomic_weight_hydrogen : ℝ) (molecular_weight_compound : ℝ) : ℝ :=
  molecular_weight_compound - (atomic_weight_nitrogen + 4 * atomic_weight_hydrogen)

theorem atomic_weight_of_iodine_is_correct :
  atomic_weight_iodine 14.01 1.008 145 = 126.958 :=
by
  unfold atomic_weight_iodine
  norm_num

end atomic_weight_of_iodine_is_correct_l23_23552


namespace problem_l23_23897

structure Vector3D where
  x : ℝ
  y : ℝ
  z : ℝ

def NotParallel (v1 v2 : Vector3D) : Prop := ¬ ∃ k : ℝ, v2 = ⟨k * v1.x, k * v1.y, k * v1.z⟩

def a : Vector3D := ⟨1, 2, -2⟩
def b : Vector3D := ⟨-2, -4, 4⟩
def c : Vector3D := ⟨1, 0, 0⟩
def d : Vector3D := ⟨-3, 0, 0⟩
def g : Vector3D := ⟨-2, 3, 5⟩
def h : Vector3D := ⟨16, 24, 40⟩
def e : Vector3D := ⟨2, 3, 0⟩
def f : Vector3D := ⟨0, 0, 0⟩

theorem problem : NotParallel g h := by
  sorry

end problem_l23_23897


namespace initial_soccer_balls_l23_23598

theorem initial_soccer_balls (x : ℝ) (h1 : 0.40 * x = y) (h2 : 0.20 * (0.60 * x) = z) (h3 : 0.80 * (0.60 * x) = 48) : x = 100 := by
  sorry

end initial_soccer_balls_l23_23598


namespace lcm_of_numbers_l23_23021

/-- Define the numbers involved -/
def a := 456
def b := 783
def c := 935
def d := 1024
def e := 1297

/-- Prove the LCM of these numbers is 2308474368000 -/
theorem lcm_of_numbers :
  Int.lcm (Int.lcm (Int.lcm (Int.lcm a b) c) d) e = 2308474368000 :=
by
  sorry

end lcm_of_numbers_l23_23021


namespace optionB_is_difference_of_squares_l23_23313

-- Definitions from conditions
def A_expr (x : ℝ) : ℝ := (x - 2) * (x + 1)
def B_expr (x y : ℝ) : ℝ := (x + 2 * y) * (x - 2 * y)
def C_expr (x y : ℝ) : ℝ := (x + y) * (-x - y)
def D_expr (x : ℝ) : ℝ := (-x + 1) * (x - 1)

theorem optionB_is_difference_of_squares (x y : ℝ) : B_expr x y = x^2 - 4 * y^2 :=
by
  -- Proof is intentionally left out as per instructions
  sorry

end optionB_is_difference_of_squares_l23_23313


namespace expected_waiting_time_l23_23019

/-- Consider a 5-minute interval. There are 5 bites on the first rod 
and 1 bite on the second rod in this interval. Therefore, the total average 
number of bites on both rods during these 5 minutes is 6. The expected waiting 
time for the first bite is 50 seconds. -/
theorem expected_waiting_time
    (average_bites_first_rod : ℝ)
    (average_bites_second_rod : ℝ)
    (total_interval_minutes : ℝ)
    (expected_waiting_time_seconds : ℝ) :
    average_bites_first_rod = 5 ∧
    average_bites_second_rod = 1 ∧
    total_interval_minutes = 5 →
    expected_waiting_time_seconds = 50 :=
by
  sorry

end expected_waiting_time_l23_23019


namespace number_of_toys_bought_l23_23265

def toy_cost (T : ℕ) : ℕ := 10 * T
def card_cost : ℕ := 2 * 5
def shirt_cost : ℕ := 5 * 6
def total_cost (T : ℕ) : ℕ := toy_cost T + card_cost + shirt_cost

theorem number_of_toys_bought (T : ℕ) : total_cost T = 70 → T = 3 :=
by
  intro h
  sorry

end number_of_toys_bought_l23_23265


namespace square_area_from_circle_area_l23_23465

variable (square_area : ℝ) (circle_area : ℝ)

theorem square_area_from_circle_area 
  (h1 : circle_area = 9 * Real.pi) 
  (h2 : square_area = (2 * Real.sqrt (circle_area / Real.pi))^2) : 
  square_area = 36 := 
by
  sorry

end square_area_from_circle_area_l23_23465


namespace minimum_m_value_l23_23830

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 + a * Real.log x + 1

theorem minimum_m_value :
  (∀ x1 x2 : ℝ, x1 ∈ Set.Ici (3 : ℝ) → x2 ∈ Set.Ici (3 : ℝ) → x1 ≠ x2 →
     ∃ a : ℝ, a ∈ Set.Icc (1 : ℝ) (2 : ℝ) ∧
     (f x1 a - f x2 a) / (x2 - x1) < m) →
  m ≥ -20 / 3 := sorry

end minimum_m_value_l23_23830


namespace intersection_distance_to_pole_l23_23968

theorem intersection_distance_to_pole (rho theta : ℝ) (h1 : rho > 0) (h2 : rho = 2 * theta + 1) (h3 : rho * theta = 1) : rho = 2 :=
by
  -- We replace "sorry" with actual proof steps, if necessary.
  sorry

end intersection_distance_to_pole_l23_23968


namespace solve_x_for_equation_l23_23213

theorem solve_x_for_equation :
  ∃ (x : ℚ), 3 * x - 5 = abs (-20 + 6) ∧ x = 19 / 3 :=
by
  sorry

end solve_x_for_equation_l23_23213


namespace Alyssa_number_of_quarters_l23_23227

def value_penny : ℝ := 0.01
def value_quarter : ℝ := 0.25
def num_pennies : ℕ := 7
def total_money : ℝ := 3.07

def num_quarters (q : ℕ) : Prop :=
  total_money - (num_pennies * value_penny) = q * value_quarter

theorem Alyssa_number_of_quarters : ∃ q : ℕ, num_quarters q ∧ q = 12 :=
by
  sorry

end Alyssa_number_of_quarters_l23_23227


namespace rotary_club_extra_omelets_l23_23456

theorem rotary_club_extra_omelets
  (small_children_tickets : ℕ)
  (older_children_tickets : ℕ)
  (adult_tickets : ℕ)
  (senior_tickets : ℕ)
  (eggs_total : ℕ)
  (omelet_for_small_child : ℝ)
  (omelet_for_older_child : ℝ)
  (omelet_for_adult : ℝ)
  (omelet_for_senior : ℝ)
  (eggs_per_omelet : ℕ)
  (extra_omelets : ℕ) :
  small_children_tickets = 53 →
  older_children_tickets = 35 →
  adult_tickets = 75 →
  senior_tickets = 37 →
  eggs_total = 584 →
  omelet_for_small_child = 0.5 →
  omelet_for_older_child = 1 →
  omelet_for_adult = 2 →
  omelet_for_senior = 1.5 →
  eggs_per_omelet = 2 →
  extra_omelets = (eggs_total - (2 * (small_children_tickets * omelet_for_small_child +
                                      older_children_tickets * omelet_for_older_child +
                                      adult_tickets * omelet_for_adult +
                                      senior_tickets * omelet_for_senior))) / eggs_per_omelet →
  extra_omelets = 25 :=
by
  intros hsmo_hold hsoc_hold hat_hold hsnt_hold htot_hold
        hosm_hold hocc_hold hact_hold hsen_hold hepom_hold hres_hold
  sorry

end rotary_club_extra_omelets_l23_23456


namespace unique_function_satisfying_condition_l23_23662

theorem unique_function_satisfying_condition :
  ∀ f : ℝ → ℝ, (∀ x y : ℝ, f (x^2 + f y) = y + (f x)^2) ↔ f = id :=
sorry

end unique_function_satisfying_condition_l23_23662


namespace custom_op_4_8_l23_23418

-- Definition of the custom operation
def custom_op (a b : ℕ) : ℕ := b + b / a

-- Theorem stating the desired equality
theorem custom_op_4_8 : custom_op 4 8 = 10 :=
by
  -- Proof is omitted
  sorry

end custom_op_4_8_l23_23418


namespace total_savings_l23_23607

theorem total_savings :
  let josiah_daily := 0.25 
  let josiah_days := 24 
  let leah_daily := 0.50 
  let leah_days := 20 
  let megan_multiplier := 2
  let megan_days := 12 
  let josiah_savings := josiah_daily * josiah_days 
  let leah_savings := leah_daily * leah_days 
  let megan_daily := megan_multiplier * leah_daily 
  let megan_savings := megan_daily * megan_days 
  let total_savings := josiah_savings + leah_savings + megan_savings 
  total_savings = 28 :=
by
  sorry

end total_savings_l23_23607


namespace problem1_problem2_l23_23211

-- Define the quadratic equation and condition for real roots
def quadratic_eq (a b c x : ℝ) := a * x^2 + b * x + c = 0

-- Problem 1
theorem problem1 (m : ℝ) : ((m - 2) * (m - 2) * (m - 2) + 2 * 2 * (2 - m) * 2 * (-1) ≥ 0) → (m ≤ 3 ∧ m ≠ 2) := sorry

-- Problem 2
theorem problem2 (m : ℝ) : 
  (∀ x, (x = 1 ∨ x = 2) → (m - 2) * x^2 + 2 * x + 1 = 0) → (-1 ≤ m ∧ m < (3 / 4)) := 
sorry

end problem1_problem2_l23_23211


namespace equivalent_expression_l23_23288

variable (x y : ℝ)

def is_positive_real (r : ℝ) : Prop := r > 0

theorem equivalent_expression 
  (hx : is_positive_real x) 
  (hy : is_positive_real y) : 
  (Real.sqrt (Real.sqrt (x ^ 2 * Real.sqrt (y ^ 3)))) = x ^ (1 / 2) * y ^ (1 / 12) :=
by
  sorry

end equivalent_expression_l23_23288


namespace quadratic_roots_l23_23330

-- Definitions based on problem conditions
def sum_of_roots (p q : ℝ) : Prop := p + q = 12
def abs_diff_of_roots (p q : ℝ) : Prop := |p - q| = 4

-- The theorem we want to prove
theorem quadratic_roots : ∃ (a b c : ℝ), a ≠ 0 ∧ (∀ p q, sum_of_roots p q ∧ abs_diff_of_roots p q → a * (x - p) * (x - q) = x^2 - 12 * x + 32) := sorry

end quadratic_roots_l23_23330


namespace initial_pens_l23_23302

theorem initial_pens (P : ℤ) (INIT : 2 * (P + 22) - 19 = 39) : P = 7 :=
by
  sorry

end initial_pens_l23_23302


namespace min_value_fraction_l23_23496

theorem min_value_fraction (x : ℝ) (h : x > 0) : ∃ y, y = 4 ∧ (∀ z, z = (x + 5) / Real.sqrt (x + 1) → y ≤ z) := sorry

end min_value_fraction_l23_23496


namespace equation_of_perpendicular_line_l23_23663

theorem equation_of_perpendicular_line (x y c : ℝ) (h₁ : x = -1) (h₂ : y = 2)
  (h₃ : 2 * x - 3 * y = -c) (h₄ : 3 * x + 2 * y - 7 = 0) :
  2 * x - 3 * y + 8 = 0 :=
sorry

end equation_of_perpendicular_line_l23_23663


namespace tabitha_color_start_l23_23989

def add_color_each_year (n : ℕ) : ℕ := n + 1

theorem tabitha_color_start 
  (age_start age_now future_colors years_future current_colors : ℕ)
  (h1 : age_start = 15)
  (h2 : age_now = 18)
  (h3 : years_future = 3)
  (h4 : age_now + years_future = 21)
  (h5 : future_colors = 8)
  (h6 : future_colors - years_future = current_colors + 3)
  (h7 : current_colors = 5)
  : age_start + (current_colors - (age_now - age_start)) = 3 := 
by
  sorry

end tabitha_color_start_l23_23989


namespace simplify_radical_expression_l23_23779

noncomputable def simpl_radical_form (q : ℝ) : ℝ :=
  Real.sqrt (15 * q) * Real.sqrt (3 * q^2) * Real.sqrt (2 * q^3)

theorem simplify_radical_expression (q : ℝ) :
  simpl_radical_form q = 3 * q^3 * Real.sqrt 10 :=
by
  sorry

end simplify_radical_expression_l23_23779


namespace polygon_sides_twice_diagonals_l23_23166

theorem polygon_sides_twice_diagonals (n : ℕ) (h1 : n ≥ 3) (h2 : n * (n - 3) / 2 = 2 * n) : n = 7 :=
sorry

end polygon_sides_twice_diagonals_l23_23166


namespace probability_at_8_10_probability_at_8_10_through_5_6_probability_at_8_10_within_circle_l23_23562

-- Definitions based on the conditions laid out in the problem
def fly_paths (n_right n_up : ℕ) : ℕ :=
  (Nat.factorial (n_right + n_up)) / ((Nat.factorial n_right) * (Nat.factorial n_up))

-- Probability for part a
theorem probability_at_8_10 : 
  (fly_paths 8 10) / (2 ^ 18) = (Nat.choose 18 8 : ℚ) / 2 ^ 18 := 
sorry

-- Probability for part b
theorem probability_at_8_10_through_5_6 :
  ((fly_paths 5 6) * (fly_paths 1 0) * (fly_paths 2 4)) / (2 ^ 18) = (6930 : ℚ) / 2 ^ 18 :=
sorry

-- Probability for part c
theorem probability_at_8_10_within_circle :
  (2 * fly_paths 2 7 * fly_paths 6 3 + 2 * fly_paths 3 6 * fly_paths 5 3 + (fly_paths 4 6) ^ 2) / (2 ^ 18) = 
  (2 * Nat.choose 9 2 * Nat.choose 9 6 + 2 * Nat.choose 9 3 * Nat.choose 9 5 + (Nat.choose 9 4) ^ 2 : ℚ) / 2 ^ 18 :=
sorry

end probability_at_8_10_probability_at_8_10_through_5_6_probability_at_8_10_within_circle_l23_23562


namespace problem_statement_l23_23505

noncomputable def f : ℝ → ℝ := sorry

theorem problem_statement (h1 : ∀ x : ℝ, f (x + 2016) = f (-x + 2016))
    (h2 : ∀ x1 x2 : ℝ, 2016 ≤ x1 ∧ 2016 ≤ x2 ∧ x1 ≠ x2 → (f x2 - f x1) / (x2 - x1) < 0) :
    f 2019 < f 2014 ∧ f 2014 < f 2017 :=
sorry

end problem_statement_l23_23505


namespace some_number_value_l23_23079

theorem some_number_value (x : ℕ) (some_number : ℕ) : x = 5 → ((x / 5) + some_number = 4) → some_number = 3 :=
by
  intros h1 h2
  sorry

end some_number_value_l23_23079


namespace third_team_pies_l23_23593

theorem third_team_pies (total first_team second_team : ℕ) (h_total : total = 750) (h_first : first_team = 235) (h_second : second_team = 275) :
  total - (first_team + second_team) = 240 := by
  sorry

end third_team_pies_l23_23593


namespace sum_of_perpendiculars_l23_23220

-- define the points on the rectangle
variables {A B C D P S R Q F : Type}

-- define rectangle ABCD and points P, S, R, Q, F
def is_rectangle (A B C D : Type) : Prop := sorry -- conditions for ABCD to be a rectangle
def point_on_segment (P A B: Type) : Prop := sorry -- P is a point on segment AB
def perpendicular (X Y Z : Type) : Prop := sorry -- definition for perpendicular between two segments
def length (X Y : Type) : ℝ := sorry -- definition for the length of a segment

-- Given conditions
axiom rect : is_rectangle A B C D
axiom p_on_ab : point_on_segment P A B
axiom ps_perp_bd : perpendicular P S D
axiom pr_perp_ac : perpendicular P R C
axiom af_perp_bd : perpendicular A F D
axiom pq_perp_af : perpendicular P Q F

-- Prove that PR + PS = AF
theorem sum_of_perpendiculars :
  length P R + length P S = length A F :=
sorry

end sum_of_perpendiculars_l23_23220


namespace num_distinct_convex_polygons_on_12_points_l23_23355

theorem num_distinct_convex_polygons_on_12_points : 
  let num_subsets := 2 ^ 12
  let num_subsets_with_0_members := Nat.choose 12 0
  let num_subsets_with_1_member := Nat.choose 12 1
  let num_subsets_with_2_members := Nat.choose 12 2
  num_subsets - num_subsets_with_0_members - num_subsets_with_1_member - num_subsets_with_2_members = 4017 :=
by
  let num_subsets := 2 ^ 12
  let num_subsets_with_0_members := Nat.choose 12 0
  let num_subsets_with_1_member := Nat.choose 12 1
  let num_subsets_with_2_members := Nat.choose 12 2
  have h : num_subsets - num_subsets_with_0_members - num_subsets_with_1_member - num_subsets_with_2_members = 4017 := by sorry
  exact h

end num_distinct_convex_polygons_on_12_points_l23_23355


namespace diane_money_l23_23325

-- Define the conditions
def total_cost : ℤ := 65
def additional_needed : ℤ := 38
def initial_amount : ℤ := total_cost - additional_needed

-- Theorem statement
theorem diane_money : initial_amount = 27 := by
  sorry

end diane_money_l23_23325


namespace math_problem_l23_23103

noncomputable def problem_statement (x y z : ℝ) (hx : 1 ≤ x) (hy : 1 ≤ y) (hz : 1 ≤ z) (hxyz : x * y * z = 1) : Prop :=
  (x^3 / ((1 + y) * (1 + z)) + y^3 / ((1 + x) * (1 + z)) + z^3 / ((1 + x) * (1 + y))) ≥ 3 / 4

theorem math_problem (x y z : ℝ) (hx : 1 ≤ x) (hy : 1 ≤ y) (hz : 1 ≤ z) (hxyz : x * y * z = 1) :
  problem_statement x y z hx hy hz hxyz :=
sorry

end math_problem_l23_23103


namespace find_k_l23_23044

theorem find_k (k : ℝ) (h : 0.5 * |-2 * k| * |k| = 1) : k = 1 ∨ k = -1 :=
sorry

end find_k_l23_23044


namespace find_missing_number_l23_23823

theorem find_missing_number (x : ℕ) (h1 : (1 + 22 + 23 + 24 + x + 26 + 27 + 2) = 8 * 20) : x = 35 :=
  sorry

end find_missing_number_l23_23823


namespace joshua_share_is_30_l23_23453

-- Definitions based on the conditions
def total_amount_shared : ℝ := 40
def ratio_joshua_justin : ℝ := 3

-- Proposition to prove
theorem joshua_share_is_30 (J : ℝ) (Joshua_share : ℝ) :
  J + ratio_joshua_justin * J = total_amount_shared → 
  Joshua_share = ratio_joshua_justin * J → 
  Joshua_share = 30 :=
sorry

end joshua_share_is_30_l23_23453


namespace option_A_correct_l23_23889

theorem option_A_correct (p : ℕ) (h1 : p > 1) (h2 : p % 2 = 1) : 
  (p - 1)^(p/2 - 1) - 1 ≡ 0 [MOD (p - 2)] :=
sorry

end option_A_correct_l23_23889


namespace boy_overall_average_speed_l23_23529

noncomputable def total_distance : ℝ := 100
noncomputable def distance1 : ℝ := 15
noncomputable def speed1 : ℝ := 12

noncomputable def distance2 : ℝ := 20
noncomputable def speed2 : ℝ := 8

noncomputable def distance3 : ℝ := 10
noncomputable def speed3 : ℝ := 25

noncomputable def distance4 : ℝ := 15
noncomputable def speed4 : ℝ := 18

noncomputable def distance5 : ℝ := 20
noncomputable def speed5 : ℝ := 10

noncomputable def distance6 : ℝ := 20
noncomputable def speed6 : ℝ := 22

noncomputable def time1 : ℝ := distance1 / speed1
noncomputable def time2 : ℝ := distance2 / speed2
noncomputable def time3 : ℝ := distance3 / speed3
noncomputable def time4 : ℝ := distance4 / speed4
noncomputable def time5 : ℝ := distance5 / speed5
noncomputable def time6 : ℝ := distance6 / speed6

noncomputable def total_time : ℝ := time1 + time2 + time3 + time4 + time5 + time6

noncomputable def overall_average_speed : ℝ := total_distance / total_time

theorem boy_overall_average_speed : overall_average_speed = 100 / (15 / 12 + 20 / 8 + 10 / 25 + 15 / 18 + 20 / 10 + 20 / 22) :=
by
  sorry

end boy_overall_average_speed_l23_23529


namespace circle_center_and_radius_l23_23478

theorem circle_center_and_radius:
  ∀ x y : ℝ, 
  (x + 1) ^ 2 + (y - 3) ^ 2 = 36 
  → ∃ C : (ℝ × ℝ), C = (-1, 3) ∧ ∃ r : ℝ, r = 6 := sorry

end circle_center_and_radius_l23_23478


namespace olympic_triathlon_total_distance_l23_23409

theorem olympic_triathlon_total_distance (x : ℝ) (L S : ℝ)
  (hL : L = 4 * x)
  (hS : S = (3 / 80) * x)
  (h_diff : L - S = 8.5) :
  x + L + S = 51.5 := by
  sorry

end olympic_triathlon_total_distance_l23_23409


namespace neg_root_sufficient_not_necessary_l23_23137

theorem neg_root_sufficient_not_necessary (a : ℝ) :
  (∃ x : ℝ, a * x^2 + 2 * x + 1 = 0 ∧ x < 0) ↔ (a < 0) :=
sorry

end neg_root_sufficient_not_necessary_l23_23137


namespace axis_of_symmetry_values_ge_one_range_m_l23_23762

open Real

-- Definitions for vectors and the function f(x)
noncomputable def a (x : ℝ) : ℝ × ℝ := (sin x, cos x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (sin x, sin x)
noncomputable def f (x : ℝ) : ℝ := (a x).fst * (b x).fst + (a x).snd * (b x).snd

-- Part I: Prove the equation of the axis of symmetry of f(x)
theorem axis_of_symmetry {k : ℤ} : f x = (sqrt 2 / 2) * sin (2 * x - π / 4) + 1 / 2 → 
                                    x = k * π / 2 + 3 * π / 8 := 
sorry

-- Part II: Prove the set of values x for which f(x) ≥ 1
theorem values_ge_one : (f x ≥ 1) ↔ (∃ (k : ℤ), π / 4 + k * π ≤ x ∧ x ≤ π / 2 + k * π) := 
sorry

-- Part III: Prove the range of m given the inequality
theorem range_m (m : ℝ) : (∀ x, π / 6 ≤ x ∧ x ≤ π / 3 → f x - m < 2) → 
                            m > (sqrt 3 - 5) / 4 := 
sorry

end axis_of_symmetry_values_ge_one_range_m_l23_23762


namespace grant_school_students_l23_23070

theorem grant_school_students (S : ℕ) 
  (h1 : S / 3 = x) 
  (h2 : x / 4 = 15) : 
  S = 180 := 
sorry

end grant_school_students_l23_23070


namespace sequence_property_l23_23060

noncomputable def U : ℕ → ℕ
| 0       => 0  -- This definition is added to ensure U 1 corresponds to U_1 = 1
| (n + 1) => U n + (n + 1)

theorem sequence_property (n : ℕ) : U n + U (n + 1) = (n + 1) * (n + 1) :=
  sorry

end sequence_property_l23_23060


namespace mrs_heine_dogs_l23_23573

-- Define the number of biscuits per dog
def biscuits_per_dog : ℕ := 3

-- Define the total number of biscuits
def total_biscuits : ℕ := 6

-- Define the number of dogs
def number_of_dogs : ℕ := 2

-- Define the proof statement
theorem mrs_heine_dogs : total_biscuits / biscuits_per_dog = number_of_dogs :=
by
  sorry

end mrs_heine_dogs_l23_23573


namespace find_x_l23_23262

open Real

theorem find_x 
  (x y : ℝ) 
  (hx_pos : 0 < x)
  (hy_pos : 0 < y) 
  (h_eq : 7 * x^2 + 21 * x * y = 2 * x^3 + 3 * x^2 * y) 
  : x = 7 := 
sorry

end find_x_l23_23262


namespace radius_of_circular_film_l23_23093

theorem radius_of_circular_film (r_canister h_canister t_film R: ℝ) 
  (V: ℝ) (h1: r_canister = 5) (h2: h_canister = 10) 
  (h3: t_film = 0.2) (h4: V = 250 * Real.pi): R = 25 * Real.sqrt 2 :=
by
  sorry

end radius_of_circular_film_l23_23093


namespace log_x2y2_l23_23947

theorem log_x2y2 (x y : ℝ) (h1 : Real.log (x^2 * y^5) = 2) (h2 : Real.log (x^3 * y^2) = 2) :
  Real.log (x^2 * y^2) = 16 / 11 :=
by
  sorry

end log_x2y2_l23_23947


namespace ratio_of_w_to_y_l23_23475

theorem ratio_of_w_to_y (w x y z : ℚ)
  (h1 : w / x = 5 / 4)
  (h2 : y / z = 3 / 2)
  (h3 : z / x = 1 / 4) :
  w / y = 10 / 3 :=
sorry

end ratio_of_w_to_y_l23_23475


namespace triangle_side_b_l23_23357

theorem triangle_side_b (A B C a b c : ℝ)
  (hA : A = 135)
  (hc : c = 1)
  (hSinB_SinC : Real.sin B * Real.sin C = Real.sqrt 2 / 10) :
  b = Real.sqrt 2 ∨ b = Real.sqrt 2 / 2 :=
by
  sorry

end triangle_side_b_l23_23357


namespace first_number_in_proportion_l23_23261

variable (x y : ℝ)

theorem first_number_in_proportion
  (h1 : x = 0.9)
  (h2 : y / x = 5 / 6) : 
  y = 0.75 := 
  by 
    sorry

end first_number_in_proportion_l23_23261


namespace cuberoot_sum_l23_23539

-- Prove that the sum c + d = 60 for the simplified form of the given expression.
theorem cuberoot_sum :
  let c := 15
  let d := 45
  c + d = 60 :=
by
  sorry

end cuberoot_sum_l23_23539


namespace no_real_roots_iff_no_positive_discriminant_l23_23455

noncomputable def discriminant (a b c : ℝ) : ℝ := b * b - 4 * a * c

theorem no_real_roots_iff_no_positive_discriminant (m : ℝ) 
  (h : discriminant m (-2*(m+2)) (m+5) < 0) : 
  (discriminant (m-5) (-2*(m+2)) m < 0 ∨ discriminant (m-5) (-2*(m+2)) m > 0 ∨ m - 5 = 0) :=
by 
  sorry

end no_real_roots_iff_no_positive_discriminant_l23_23455


namespace gain_percentage_l23_23145

variables (C S : ℝ) (hC : C > 0)
variables (hS : S > 0)

def cost_price := 25 * C
def selling_price := 25 * S
def gain := 10 * S 

theorem gain_percentage (h_eq : 25 * S = 25 * C + 10 * S):
  (S = C) → 
  ((gain / cost_price) * 100 = 40) :=
by
  sorry

end gain_percentage_l23_23145


namespace number_of_squares_l23_23735

open Int

theorem number_of_squares (n : ℕ) (h : n < 10^7) : 
  (∃ n, 36 ∣ n ∧ n^2 < 10^07) ↔ (n = 87) :=
by sorry

end number_of_squares_l23_23735


namespace violet_children_count_l23_23542

theorem violet_children_count 
  (family_pass_cost : ℕ := 120)
  (adult_ticket_cost : ℕ := 35)
  (child_ticket_cost : ℕ := 20)
  (separate_ticket_total_cost : ℕ := 155)
  (adult_count : ℕ := 1) : 
  ∃ c : ℕ, 35 + 20 * c = 155 ∧ c = 6 :=
by
  sorry

end violet_children_count_l23_23542


namespace student_test_score_l23_23786

variable (C I : ℕ)

theorem student_test_score  
  (h1 : C + I = 100)
  (h2 : C - 2 * I = 64) :
  C = 88 :=
by
  -- Proof steps should go here
  sorry

end student_test_score_l23_23786


namespace vegetable_plot_area_l23_23525

variable (V W : ℝ)

theorem vegetable_plot_area (h1 : (1/2) * V + (1/3) * W = 13) (h2 : (1/2) * W + (1/3) * V = 12) : V = 18 :=
by
  sorry

end vegetable_plot_area_l23_23525


namespace cake_sector_chord_length_l23_23948

noncomputable def sector_longest_chord_square (d : ℝ) (n : ℕ) : ℝ :=
  let r := d / 2
  let theta := (360 : ℝ) / n
  let chord_length := 2 * r * Real.sin (theta / 2 * Real.pi / 180)
  chord_length ^ 2

theorem cake_sector_chord_length :
  sector_longest_chord_square 18 5 = 111.9473 := by
  sorry

end cake_sector_chord_length_l23_23948


namespace cost_of_one_dozen_pens_l23_23424

noncomputable def cost_of_one_pen_and_one_pencil_ratio := 5

theorem cost_of_one_dozen_pens
  (cost_pencil : ℝ)
  (cost_3_pens_5_pencils : 3 * (cost_of_one_pen_and_one_pencil_ratio * cost_pencil) + 5 * cost_pencil = 200) :
  12 * (cost_of_one_pen_and_one_pencil_ratio * cost_pencil) = 600 :=
by
  sorry

end cost_of_one_dozen_pens_l23_23424


namespace a_n_formula_T_n_formula_l23_23557

variable (a : Nat → Int) (b : Nat → Int)
variable (S : Nat → Int) (T : Nat → Int)
variable (d a_1 : Int)

-- Conditions:
axiom a_seq_arith : ∀ n, a (n + 1) = a n + d
axiom S_arith : ∀ n, S n = n * (a 1 + a n) / 2
axiom S_10 : S 10 = 110
axiom geo_seq : (a 2) ^ 2 = a 1 * a 4
axiom b_def : ∀ n, b n = 1 / ((a n - 1) * (a n + 1))

-- Goals: 
-- 1. Find the general formula for the terms of sequence {a_n}
theorem a_n_formula : ∀ n, a n = 2 * n := sorry

-- 2. Find the sum of the first n terms T_n of the sequence {b_n} given b_n
theorem T_n_formula : ∀ n, T n = 1 / 2 - 1 / (4 * n + 2) := sorry

end a_n_formula_T_n_formula_l23_23557


namespace toms_dog_age_in_six_years_l23_23599

-- Define the conditions as hypotheses
variables (B T D : ℕ)

-- Conditions
axiom h1 : B = 4 * D
axiom h2 : T = B - 3
axiom h3 : B + 6 = 30

-- The proof goal: Tom's dog's age in six years
theorem toms_dog_age_in_six_years : D + 6 = 12 :=
  sorry -- Proof is omitted based on the instructions

end toms_dog_age_in_six_years_l23_23599


namespace brenda_age_problem_l23_23775

variable (A B J : Nat)

theorem brenda_age_problem
  (h1 : A = 4 * B) 
  (h2 : J = B + 9) 
  (h3 : A = J) : 
  B = 3 := 
by 
  sorry

end brenda_age_problem_l23_23775


namespace algae_difference_l23_23181

-- Define the original number of algae plants.
def original_algae := 809

-- Define the current number of algae plants.
def current_algae := 3263

-- Statement to prove: The difference between the current number of algae plants and the original number of algae plants is 2454.
theorem algae_difference : current_algae - original_algae = 2454 := by
  sorry

end algae_difference_l23_23181


namespace intersection_point_l23_23458

theorem intersection_point (a b d x y : ℝ) (h1 : a = b + d) (h2 : a * x + b * y = b + 2 * d) :
    (x, y) = (-1, 1) :=
by
  sorry

end intersection_point_l23_23458


namespace arithmetic_sequence_eighth_term_l23_23714

theorem arithmetic_sequence_eighth_term (a d : ℚ) 
  (h1 : 6 * a + 15 * d = 21) 
  (h2 : a + 6 * d = 8) : 
  a + 7 * d = 9 + 2/7 :=
by
  sorry

end arithmetic_sequence_eighth_term_l23_23714


namespace max_distance_on_highway_l23_23547

-- Assume there are definitions for the context of this problem
def mpg_highway : ℝ := 12.2
def gallons : ℝ := 24
def max_distance (mpg : ℝ) (gal : ℝ) : ℝ := mpg * gal

theorem max_distance_on_highway :
  max_distance mpg_highway gallons = 292.8 :=
sorry

end max_distance_on_highway_l23_23547


namespace positive_integers_sequence_l23_23031

theorem positive_integers_sequence (a b c d : ℕ) (h1 : a < b) (h2 : b < c) (h3 : c < d) 
  (h4 : a ∣ (b + c + d)) (h5 : b ∣ (a + c + d)) 
  (h6 : c ∣ (a + b + d)) (h7 : d ∣ (a + b + c)) : 
  (a = 1 ∧ b = 2 ∧ c = 3 ∧ d = 6) ∨ 
  (a = 1 ∧ b = 2 ∧ c = 6 ∧ d = 9) ∨ 
  (a = 1 ∧ b = 3 ∧ c = 8 ∧ d = 12) ∨ 
  (a = 1 ∧ b = 4 ∧ c = 5 ∧ d = 10) ∨ 
  (a = 1 ∧ b = 6 ∧ c = 14 ∧ d = 21) ∨ 
  (a = 2 ∧ b = 3 ∧ c = 10 ∧ d = 15) :=
sorry

end positive_integers_sequence_l23_23031


namespace square_area_l23_23198

theorem square_area (side_length : ℕ) (h : side_length = 17) : side_length * side_length = 289 :=
by sorry

end square_area_l23_23198


namespace krakozyabr_count_l23_23263

variable (n H W T : ℕ)
variable (h1 : H = 5 * n) -- 20% of the 'krakozyabrs' with horns also have wings
variable (h2 : W = 4 * n) -- 25% of the 'krakozyabrs' with wings also have horns
variable (h3 : T = H + W - n) -- Total number of 'krakozyabrs' using inclusion-exclusion
variable (h4 : 25 < T)
variable (h5 : T < 35)

theorem krakozyabr_count : T = 32 := by
  sorry

end krakozyabr_count_l23_23263


namespace history_book_cost_is_correct_l23_23689

-- Define the conditions
def total_books : ℕ := 80
def math_book_cost : ℕ := 4
def total_price : ℕ := 390
def math_books_purchased : ℕ := 10

-- The number of history books
def history_books_purchased : ℕ := total_books - math_books_purchased

-- The total cost of math books
def total_cost_math_books : ℕ := math_books_purchased * math_book_cost

-- The total cost of history books
def total_cost_history_books : ℕ := total_price - total_cost_math_books

-- Define the cost of each history book
def history_book_cost : ℕ := total_cost_history_books / history_books_purchased

-- The theorem to be proven
theorem history_book_cost_is_correct : history_book_cost = 5 := 
by
  sorry

end history_book_cost_is_correct_l23_23689


namespace point_in_fourth_quadrant_l23_23463

theorem point_in_fourth_quadrant (m : ℝ) (h : m < 0) : (-m + 1 > 0 ∧ -1 < 0) :=
by
  sorry

end point_in_fourth_quadrant_l23_23463


namespace crayons_count_l23_23071

-- Definitions based on the conditions given in the problem
def total_crayons : Nat := 96
def benny_crayons : Nat := 12
def fred_crayons : Nat := 2 * benny_crayons
def jason_crayons (sarah_crayons : Nat) : Nat := 3 * sarah_crayons

-- Stating the proof goal
theorem crayons_count (sarah_crayons : Nat) :
  fred_crayons + benny_crayons + jason_crayons sarah_crayons + sarah_crayons = total_crayons →
  sarah_crayons = 15 ∧
  fred_crayons = 24 ∧
  jason_crayons sarah_crayons = 45 ∧
  benny_crayons = 12 :=
by
  sorry

end crayons_count_l23_23071


namespace find_a_l23_23327

variable {x n : ℝ}

theorem find_a (hx : x > 0) (hn : n > 0) :
    (∀ n > 0, x + n^n / x^n ≥ n + 1) ↔ (∀ n > 0, a = n^n) :=
sorry

end find_a_l23_23327


namespace peach_pies_l23_23377

theorem peach_pies (total_pies : ℕ) (apple_ratio blueberry_ratio peach_ratio : ℕ)
  (h_ratio : apple_ratio + blueberry_ratio + peach_ratio = 10)
  (h_total : total_pies = 30)
  (h_ratios : apple_ratio = 3 ∧ blueberry_ratio = 2 ∧ peach_ratio = 5) :
  total_pies / (apple_ratio + blueberry_ratio + peach_ratio) * peach_ratio = 15 :=
by
  sorry

end peach_pies_l23_23377


namespace tire_price_l23_23287

theorem tire_price {p : ℤ} (h : 4 * p + 1 = 421) : p = 105 :=
sorry

end tire_price_l23_23287


namespace option_b_does_not_represent_5x_l23_23441

theorem option_b_does_not_represent_5x (x : ℝ) : 
  (∀ a, a = 5 * x ↔ a = x + x + x + x + x) →
  (¬ (5 * x = x * x * x * x * x)) :=
by
  intro h
  -- Using sorry to skip the proof.
  sorry

end option_b_does_not_represent_5x_l23_23441


namespace Jakes_brother_has_more_l23_23625

-- Define the number of comic books Jake has
def Jake_comics : ℕ := 36

-- Define the total number of comic books Jake and his brother have together
def total_comics : ℕ := 87

-- Prove Jake's brother has 15 more comic books than Jake
theorem Jakes_brother_has_more : ∃ B, B > Jake_comics ∧ B + Jake_comics = total_comics ∧ B - Jake_comics = 15 :=
by
  sorry

end Jakes_brother_has_more_l23_23625


namespace walk_to_school_l23_23120

theorem walk_to_school (W P : ℕ) (h1 : W + P = 41) (h2 : W = P + 3) : W = 22 :=
by 
  sorry

end walk_to_school_l23_23120


namespace factor_polynomial_l23_23223

-- Define the necessary polynomials
def p (x : ℝ) : ℝ := x^2 + 4*x + 3
def q (x : ℝ) : ℝ := x^2 + 8*x + 15
def r (x : ℝ) : ℝ := x^2 + 6*x - 8

-- State the main theorem
theorem factor_polynomial (x : ℝ) : 
  (p x * q x) + r x = (x^2 + 7*x + 1) * (x^2 + 4*x + 7) :=
by
  sorry

end factor_polynomial_l23_23223


namespace josie_initial_amount_is_correct_l23_23432

def cost_of_milk := 4.00 / 2
def cost_of_bread := 3.50
def cost_of_detergent_after_coupon := 10.25 - 1.25
def cost_of_bananas := 2 * 0.75
def total_cost := cost_of_milk + cost_of_bread + cost_of_detergent_after_coupon + cost_of_bananas
def leftover := 4.00
def initial_amount := total_cost + leftover

theorem josie_initial_amount_is_correct :
  initial_amount = 20.00 := by
  sorry

end josie_initial_amount_is_correct_l23_23432


namespace area_transformation_l23_23622

variables {g : ℝ → ℝ}

theorem area_transformation (h : ∫ x in a..b, g x = 12) :
  ∫ x in c..d, 4 * g (2 * x + 3) = 48 :=
by
  sorry

end area_transformation_l23_23622


namespace sum_of_A_and_B_zero_l23_23114

theorem sum_of_A_and_B_zero
  (A B C : ℝ)
  (h1 : A ≠ B)
  (h2 : C ≠ 0)
  (f g : ℝ → ℝ)
  (h3 : ∀ x, f x = A * x + B + C)
  (h4 : ∀ x, g x = B * x + A - C)
  (h5 : ∀ x, f (g x) - g (f x) = 2 * C) : A + B = 0 :=
sorry

end sum_of_A_and_B_zero_l23_23114


namespace sum_of_a_equals_five_l23_23197

theorem sum_of_a_equals_five
  (f : ℕ → ℕ → ℕ)  -- Represents the function f defined by Table 1
  (a : ℕ → ℕ)  -- Represents the occurrences a₀, a₁, ..., a₄
  (h1 : a 0 + a 1 + a 2 + a 3 + a 4 = 5)  -- Condition 1
  (h2 : 0 * a 0 + 1 * a 1 + 2 * a 2 + 3 * a 3 + 4 * a 4 = 5)  -- Condition 2
  : a 0 + a 1 + a 2 + a 3 = 5 :=
sorry

end sum_of_a_equals_five_l23_23197


namespace sum_of_consecutive_negative_integers_with_product_3080_l23_23171

theorem sum_of_consecutive_negative_integers_with_product_3080 :
  ∃ (n : ℤ), n < 0 ∧ (n * (n + 1) = 3080) ∧ (n + (n + 1) = -111) :=
sorry

end sum_of_consecutive_negative_integers_with_product_3080_l23_23171


namespace factorize_2x2_minus_4x_factorize_xy2_minus_2xy_plus_x_l23_23209

-- Problem 1
theorem factorize_2x2_minus_4x (x : ℝ) : 
  2 * x^2 - 4 * x = 2 * x * (x - 2) := 
by 
  sorry

-- Problem 2
theorem factorize_xy2_minus_2xy_plus_x (x y : ℝ) :
  x * y^2 - 2 * x * y + x = x * (y - 1)^2 :=
by 
  sorry

end factorize_2x2_minus_4x_factorize_xy2_minus_2xy_plus_x_l23_23209


namespace water_intake_proof_l23_23545

variable {quarts_per_bottle : ℕ} {bottles_per_day : ℕ} {extra_ounces_per_day : ℕ} 
variable {days_per_week : ℕ} {ounces_per_quart : ℕ} 

def total_weekly_water_intake 
    (quarts_per_bottle : ℕ) 
    (bottles_per_day : ℕ) 
    (extra_ounces_per_day : ℕ) 
    (ounces_per_quart : ℕ) 
    (days_per_week : ℕ) 
    (correct_answer : ℕ) : Prop :=
    (quarts_per_bottle * ounces_per_quart * bottles_per_day + extra_ounces_per_day) * days_per_week = correct_answer

theorem water_intake_proof : 
    total_weekly_water_intake 3 2 20 32 7 812 := 
by
    sorry

end water_intake_proof_l23_23545


namespace Nishita_preferred_shares_l23_23366

variable (P : ℕ)

def preferred_share_dividend : ℕ := 5 * P
def common_share_dividend : ℕ := 3500 * 3  -- 3.5 * 1000

theorem Nishita_preferred_shares :
  preferred_share_dividend P + common_share_dividend = 16500 → P = 1200 :=
by
  unfold preferred_share_dividend common_share_dividend
  intro h
  sorry

end Nishita_preferred_shares_l23_23366


namespace arithmetic_sequence_sum_l23_23264

theorem arithmetic_sequence_sum :
  ∃ (a l d n : ℕ), a = 71 ∧ l = 109 ∧ d = 2 ∧ n = ((l - a) / d) + 1 ∧ 
    (3 * (n * (a + l) / 2) = 5400) := sorry

end arithmetic_sequence_sum_l23_23264


namespace necklace_cost_l23_23728

theorem necklace_cost (total_savings earrings_cost remaining_savings: ℕ) 
                      (h1: total_savings = 80) 
                      (h2: earrings_cost = 23) 
                      (h3: remaining_savings = 9) : 
   total_savings - earrings_cost - remaining_savings = 48 :=
by
  sorry

end necklace_cost_l23_23728


namespace range_of_a_real_root_l23_23982

theorem range_of_a_real_root :
  (∀ x : ℝ, x^2 - a * x + 4 = 0 → ∃ x : ℝ, (x^2 - a * x + 4 = 0 ∧ (a ≥ 4 ∨ a ≤ -4))) ∨
  (∀ x : ℝ, x^2 + (a-2) * x + 4 = 0 → ∃ x : ℝ, (x^2 + (a-2) * x + 4 = 0 ∧ (a ≥ 6 ∨ a ≤ -2))) ∨
  (∀ x : ℝ, x^2 + 2 * a * x + a^2 + 1 = 0 → False) →
  (a ≥ 4 ∨ a ≤ -2) :=
by
  sorry

end range_of_a_real_root_l23_23982


namespace Wayne_blocks_l23_23134

theorem Wayne_blocks (initial_blocks : ℕ) (additional_blocks : ℕ) (total_blocks : ℕ) 
  (h1 : initial_blocks = 9) (h2 : additional_blocks = 6) 
  (h3 : total_blocks = initial_blocks + additional_blocks) : 
  total_blocks = 15 :=
by {
  -- h1: initial_blocks = 9
  -- h2: additional_blocks = 6
  -- h3: total_blocks = initial_blocks + additional_blocks
  sorry
}

end Wayne_blocks_l23_23134


namespace equal_if_fraction_is_positive_integer_l23_23233

theorem equal_if_fraction_is_positive_integer
  (a b : ℕ)
  (h_pos_a : a > 0)
  (h_pos_b : b > 0)
  (K : ℝ := Real.sqrt ((a^2 + b^2:ℕ)/2))
  (A : ℝ := (a + b:ℕ)/2)
  (h_int_pos : ∃ (n : ℕ), n > 0 ∧ K / A = n) :
  a = b := sorry

end equal_if_fraction_is_positive_integer_l23_23233


namespace caleb_ice_cream_vs_frozen_yoghurt_l23_23649

theorem caleb_ice_cream_vs_frozen_yoghurt :
  let cost_chocolate_ice_cream := 6 * 5
  let discount_chocolate := 0.10 * cost_chocolate_ice_cream
  let total_chocolate_ice_cream := cost_chocolate_ice_cream - discount_chocolate

  let cost_vanilla_ice_cream := 4 * 4
  let discount_vanilla := 0.07 * cost_vanilla_ice_cream
  let total_vanilla_ice_cream := cost_vanilla_ice_cream - discount_vanilla

  let total_ice_cream := total_chocolate_ice_cream + total_vanilla_ice_cream

  let cost_strawberry_yoghurt := 3 * 3
  let tax_strawberry := 0.05 * cost_strawberry_yoghurt
  let total_strawberry_yoghurt := cost_strawberry_yoghurt + tax_strawberry

  let cost_mango_yoghurt := 2 * 2
  let tax_mango := 0.03 * cost_mango_yoghurt
  let total_mango_yoghurt := cost_mango_yoghurt + tax_mango

  let total_yoghurt := total_strawberry_yoghurt + total_mango_yoghurt

  (total_ice_cream - total_yoghurt = 28.31) := by
  sorry

end caleb_ice_cream_vs_frozen_yoghurt_l23_23649


namespace averageFishIs75_l23_23926

-- Introduce the number of fish in Boast Pool
def BoastPool : ℕ := 75

-- Introduce the number of fish in Onum Lake
def OnumLake : ℕ := BoastPool + 25

-- Introduce the number of fish in Riddle Pond
def RiddlePond : ℕ := OnumLake / 2

-- Define the total number of fish in all three bodies of water
def totalFish : ℕ := BoastPool + OnumLake + RiddlePond

-- Define the average number of fish in all three bodies of water
def averageFish : ℕ := totalFish / 3

-- Prove that the average number of fish is 75
theorem averageFishIs75 : averageFish = 75 :=
by
  -- We need to provide the proof steps here but using sorry to skip
  sorry

end averageFishIs75_l23_23926


namespace derivative_at_1_l23_23876

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.log x

theorem derivative_at_1 : deriv f 1 = 1 + Real.cos 1 :=
by
  sorry

end derivative_at_1_l23_23876


namespace overall_percent_decrease_l23_23503

theorem overall_percent_decrease (trouser_price_italy : ℝ) (jacket_price_italy : ℝ) 
(trouser_price_uk : ℝ) (trouser_discount_uk : ℝ) (jacket_price_uk : ℝ) 
(jacket_discount_uk : ℝ) (exchange_rate : ℝ) 
(h1 : trouser_price_italy = 200) (h2 : jacket_price_italy = 150) 
(h3 : trouser_price_uk = 150) (h4 : trouser_discount_uk = 0.20) 
(h5 : jacket_price_uk = 120) (h6 : jacket_discount_uk = 0.30) 
(h7 : exchange_rate = 0.85) : 
((trouser_price_italy + jacket_price_italy) - 
 ((trouser_price_uk * (1 - trouser_discount_uk) / exchange_rate) + 
 (jacket_price_uk * (1 - jacket_discount_uk) / exchange_rate))) / 
 (trouser_price_italy + jacket_price_italy) * 100 = 31.43 := 
by 
  sorry

end overall_percent_decrease_l23_23503


namespace sequence_monotonic_b_gt_neg3_l23_23004

theorem sequence_monotonic_b_gt_neg3 (b : ℝ) :
  (∀ n : ℕ, n > 0 → (n+1)^2 + b*(n+1) > n^2 + b*n) ↔ b > -3 :=
by sorry

end sequence_monotonic_b_gt_neg3_l23_23004


namespace find_right_triangle_sides_l23_23153

def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

def area_condition (a b c : ℕ) : Prop :=
  a * b = 3 * (a + b + c)

theorem find_right_triangle_sides :
  ∃ (a b c : ℕ),
    is_right_triangle a b c ∧ area_condition a b c ∧
    ((a = 7 ∧ b = 24 ∧ c = 25) ∨
     (a = 8 ∧ b = 15 ∧ c = 17) ∨
     (a = 9 ∧ b = 12 ∧ c = 15)) :=
sorry

end find_right_triangle_sides_l23_23153


namespace total_lives_l23_23484

theorem total_lives (initial_friends : ℕ) (initial_lives_per_friend : ℕ) (additional_players : ℕ) (lives_per_new_player : ℕ) :
  initial_friends = 7 →
  initial_lives_per_friend = 7 →
  additional_players = 2 →
  lives_per_new_player = 7 →
  (initial_friends * initial_lives_per_friend + additional_players * lives_per_new_player) = 63 :=
by
  intros
  sorry

end total_lives_l23_23484


namespace mean_of_xyz_l23_23221

theorem mean_of_xyz (mean7 : ℕ) (mean10 : ℕ) (x y z : ℕ) (h1 : mean7 = 40) (h2 : mean10 = 50) : (x + y + z) / 3 = 220 / 3 :=
by
  have sum7 := 7 * mean7
  have sum10 := 10 * mean10
  have sum_xyz := sum10 - sum7
  have mean_xyz := sum_xyz / 3
  sorry

end mean_of_xyz_l23_23221


namespace friends_cant_go_to_movies_l23_23397

theorem friends_cant_go_to_movies (total_friends : ℕ) (friends_can_go : ℕ) (H1 : total_friends = 15) (H2 : friends_can_go = 8) : (total_friends - friends_can_go) = 7 :=
by
  sorry

end friends_cant_go_to_movies_l23_23397


namespace intersections_of_absolute_value_functions_l23_23882

theorem intersections_of_absolute_value_functions : 
  (∃ x y : ℝ, y = |3 * x + 4| ∧ y = -|4 * x + 3|) → ∃ (x y : ℝ), (x = -1 ∧ y = 1) ∧ ¬(∃ (x' y' : ℝ), y' = |3 * x' + 4| ∧ y' = -|4 * x' + 3| ∧ (x' ≠ -1 ∨ y' ≠ 1)) :=
by
  sorry

end intersections_of_absolute_value_functions_l23_23882


namespace anna_least_days_l23_23085

theorem anna_least_days (borrow : ℝ) (interest_rate : ℝ) (days : ℕ) :
  (borrow = 20) → (interest_rate = 0.10) → borrow + (borrow * interest_rate * days) ≥ 2 * borrow → days ≥ 10 :=
by
  intros h1 h2 h3
  sorry

end anna_least_days_l23_23085


namespace kho_kho_only_l23_23870

variable (K H B : ℕ)

theorem kho_kho_only :
  (K + B = 10) ∧ (H + 5 = H + B) ∧ (B = 5) ∧ (K + H + B = 45) → H = 35 :=
by
  intros h
  sorry

end kho_kho_only_l23_23870


namespace units_digit_35_87_plus_93_49_l23_23109

theorem units_digit_35_87_plus_93_49 : (35^87 + 93^49) % 10 = 8 := by
  sorry

end units_digit_35_87_plus_93_49_l23_23109


namespace largest_y_value_l23_23008

theorem largest_y_value (y : ℝ) : (6 * y^2 - 31 * y + 35 = 0) → (y ≤ 2.5) :=
by
  intro h
  sorry

end largest_y_value_l23_23008


namespace smallest_perimeter_l23_23771

noncomputable def smallest_possible_perimeter : ℕ :=
  let n := 3
  n + (n + 1) + (n + 2)

theorem smallest_perimeter (n : ℕ) (h : n > 2) (ineq1 : n + (n + 1) > (n + 2)) 
  (ineq2 : n + (n + 2) > (n + 1)) (ineq3 : (n + 1) + (n + 2) > n) : 
  smallest_possible_perimeter = 12 :=
by
  sorry

end smallest_perimeter_l23_23771


namespace complement_of_set_A_is_34_l23_23605

open Set

noncomputable def U : Set ℕ := {n : ℕ | True}

noncomputable def A : Set ℕ := {x : ℕ | x^2 - 7*x + 10 ≥ 0}

-- Complement of A in U
noncomputable def C_U_A : Set ℕ := U \ A

theorem complement_of_set_A_is_34 : C_U_A = {3, 4} :=
by sorry

end complement_of_set_A_is_34_l23_23605


namespace rocket_parachute_opens_l23_23610

theorem rocket_parachute_opens (h t : ℝ) : h = -t^2 + 12 * t + 1 ∧ h = 37 -> t = 6 :=
by sorry

end rocket_parachute_opens_l23_23610


namespace arithmetic_sequence_sum_l23_23127

theorem arithmetic_sequence_sum :
  ∀ (a_n : ℕ → ℝ) (S_n : ℕ → ℝ) (d : ℝ),
    (∀ (n : ℕ), a_n n = 1 + (n - 1) * d) →  -- first condition
    d ≠ 0 →  -- second condition
    (∀ (n : ℕ), S_n n = n / 2 * (2 * 1 + (n - 1) * d)) →  -- third condition
    (1 * (1 + 4 * d) = (1 + d) ^ 2) →  -- fourth condition
    S_n 8 = 64 :=  -- conclusion
by {
  sorry
}

end arithmetic_sequence_sum_l23_23127


namespace solve_inequality_l23_23078

-- Define conditions
def valid_x (x : ℝ) : Prop := x ≠ -3 ∧ x ≠ -8/3

-- Define the inequality
def inequality (x : ℝ) : Prop := (x - 2) / (x + 3) > (4 * x + 5) / (3 * x + 8)

-- Define the solution set
def solution_set (x : ℝ) : Prop :=
  (-3 < x ∧ x < -8/3) ∨ ((1 - Real.sqrt 89) / 4 < x ∧ x < (1 + Real.sqrt 89) / 4)

-- Prove the equivalence
theorem solve_inequality (x : ℝ) (h : valid_x x) : inequality x ↔ solution_set x :=
by
  sorry

end solve_inequality_l23_23078


namespace number_of_mixed_vegetable_plates_l23_23874

def cost_of_chapati := 6
def cost_of_rice := 45
def cost_of_mixed_vegetable := 70
def chapatis_ordered := 16
def rice_ordered := 5
def ice_cream_cups := 6 -- though not used, included for completeness
def total_amount_paid := 1111

def total_cost_of_known_items := (chapatis_ordered * cost_of_chapati) + (rice_ordered * cost_of_rice)
def amount_spent_on_mixed_vegetable := total_amount_paid - total_cost_of_known_items

theorem number_of_mixed_vegetable_plates : 
  amount_spent_on_mixed_vegetable / cost_of_mixed_vegetable = 11 := 
by sorry

end number_of_mixed_vegetable_plates_l23_23874


namespace ryan_learning_hours_l23_23614

theorem ryan_learning_hours :
  ∀ (e c s : ℕ) , (e = 6) → (s = 58) → (e = c + 3) → (c = 3) :=
by
  intros e c s he hs hc
  sorry

end ryan_learning_hours_l23_23614


namespace not_proportional_eqn_exists_l23_23534

theorem not_proportional_eqn_exists :
  ∀ (x y : ℝ), (4 * x + 2 * y = 8) → ¬ ((∃ k : ℝ, x = k * y) ∨ (∃ k : ℝ, x * y = k)) :=
by
  intros x y h
  sorry

end not_proportional_eqn_exists_l23_23534


namespace clock_angle_at_3_30_l23_23316

theorem clock_angle_at_3_30 
    (deg_per_hour: Real := 30)
    (full_circle_deg: Real := 360)
    (hours_on_clock: Real := 12)
    (hour_hand_extra_deg: Real := 30 / 2)
    (hour_hand_deg: Real := 3 * deg_per_hour + hour_hand_extra_deg)
    (minute_hand_deg: Real := 6 * deg_per_hour) : 
    hour_hand_deg = 105 ∧ minute_hand_deg = 180 ∧ (minute_hand_deg - hour_hand_deg) = 75 := 
sorry

-- The problem specifies to write the theorem statement only, without the proof steps.

end clock_angle_at_3_30_l23_23316


namespace inequality_proof_l23_23027

theorem inequality_proof (a b c d : ℕ) (h₀: a + c ≤ 1982) (h₁: (0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d)) (h₂: (a:ℚ)/b + (c:ℚ)/d < 1) :
  1 - (a:ℚ)/b - (c:ℚ)/d > 1 / (1983 ^ 3) :=
sorry

end inequality_proof_l23_23027


namespace problem_statement_l23_23826

variables (a b : ℝ)

-- Conditions: The lines \(x = \frac{1}{3}y + a\) and \(y = \frac{1}{3}x + b\) intersect at \((3, 1)\).
def lines_intersect_at (a b : ℝ) : Prop :=
  (3 = (1/3) * 1 + a) ∧ (1 = (1/3) * 3 + b)

-- Goal: Prove that \(a + b = \frac{8}{3}\)
theorem problem_statement (H : lines_intersect_at a b) : a + b = 8 / 3 :=
by
  sorry

end problem_statement_l23_23826


namespace algebra_ineq_a2_b2_geq_2_l23_23865

theorem algebra_ineq_a2_b2_geq_2
  (a b : ℝ)
  (h1 : a^3 - b^3 = 2)
  (h2 : a^5 - b^5 ≥ 4) :
  a^2 + b^2 ≥ 2 :=
by
  sorry

end algebra_ineq_a2_b2_geq_2_l23_23865


namespace basketball_not_table_tennis_l23_23629

theorem basketball_not_table_tennis (total_students likes_basketball likes_table_tennis dislikes_all : ℕ) (likes_basketball_not_tt : ℕ) :
  total_students = 30 →
  likes_basketball = 15 →
  likes_table_tennis = 10 →
  dislikes_all = 8 →
  (likes_basketball - 3 = likes_basketball_not_tt) →
  likes_basketball_not_tt = 12 := by
  intros h_total h_basketball h_table_tennis h_dislikes h_eq
  sorry

end basketball_not_table_tennis_l23_23629


namespace mary_initial_borrowed_books_l23_23408

-- We first define the initial number of books B.
variable (B : ℕ)

-- Next, we encode the conditions into a final condition of having 12 books.
def final_books (B : ℕ) : ℕ := (B - 3 + 5) - 2 + 7

-- The proof problem is to show that B must be 5.
theorem mary_initial_borrowed_books (B : ℕ) (h : final_books B = 12) : B = 5 :=
by
  sorry

end mary_initial_borrowed_books_l23_23408


namespace polar_to_rect_l23_23165

theorem polar_to_rect (r θ : ℝ) (hr : r = 5) (hθ : θ = 5 * Real.pi / 3) :
  (r * Real.cos θ, r * Real.sin θ) = (2.5, 5 * Real.sqrt 3 / 2) :=
by
  rw [hr, hθ]
  sorry

end polar_to_rect_l23_23165


namespace even_product_implies_even_factor_l23_23042

theorem even_product_implies_even_factor (a b : ℕ) (h : Even (a * b)) : Even a ∨ Even b :=
by
  sorry

end even_product_implies_even_factor_l23_23042


namespace inequality_cube_l23_23992

theorem inequality_cube (a b : ℝ) (h : a > b) : a^3 > b^3 :=
sorry

end inequality_cube_l23_23992


namespace problem_1_problem_2_l23_23006

theorem problem_1 (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a^2 + b^2 + c^2 = 9) : abc ≤ 3 * Real.sqrt 3 := 
sorry

theorem problem_2 (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a^2 + b^2 + c^2 = 9) : 
  (a^2 / (b + c)) + (b^2 / (c + a)) + (c^2 / (a + b)) > (a + b + c) / 3 := 
sorry

end problem_1_problem_2_l23_23006


namespace calculate_expression_l23_23038

theorem calculate_expression : (0.25)^(-0.5) + (1/27)^(-1/3) - 625^(0.25) = 0 := 
by 
  sorry

end calculate_expression_l23_23038


namespace time_taken_by_Arun_to_cross_train_B_l23_23402

structure Train :=
  (length : ℕ)
  (speed_kmh : ℕ)

def to_m_per_s (speed_kmh : ℕ) : ℕ :=
  (speed_kmh * 1000) / 3600

def relative_speed (trainA trainB : Train) : ℕ :=
  to_m_per_s trainA.speed_kmh + to_m_per_s trainB.speed_kmh

def total_length (trainA trainB : Train) : ℕ :=
  trainA.length + trainB.length

def time_to_cross (trainA trainB : Train) : ℕ :=
  total_length trainA trainB / relative_speed trainA trainB

theorem time_taken_by_Arun_to_cross_train_B :
  time_to_cross (Train.mk 175 54) (Train.mk 150 36) = 13 :=
by
  sorry

end time_taken_by_Arun_to_cross_train_B_l23_23402


namespace ducks_in_the_marsh_l23_23457

-- Define the conditions
def number_of_geese : ℕ := 58
def total_number_of_birds : ℕ := 95
def number_of_ducks : ℕ := total_number_of_birds - number_of_geese

-- Prove the conclusion
theorem ducks_in_the_marsh : number_of_ducks = 37 := by
  -- subtraction to find number_of_ducks
  sorry

end ducks_in_the_marsh_l23_23457


namespace total_profit_l23_23690

-- Definitions based on the conditions
variables (A B C : ℝ) (P : ℝ)
variables (hA : A = 3 * B) (hB : B = (2 / 3) * C) (hB_share : ((2 / 11) * P) = 1400)

-- The theorem we are going to prove
theorem total_profit (A B C P : ℝ) (hA : A = 3 * B) (hB : B = (2 / 3) * C) (hB_share : ((2 / 11) * P) = 1400) : 
  P = 7700 :=
by
  sorry

end total_profit_l23_23690


namespace tan_alpha_neg_four_thirds_cos2alpha_plus_cos_alpha_add_pi_over_2_l23_23720

variable (α : ℝ)
variable (h1 : π / 2 < α)
variable (h2 : α < π)
variable (h3 : Real.sin α = 4 / 5)

theorem tan_alpha_neg_four_thirds (h1 : π / 2 < α) (h2 : α < π) (h3 : Real.sin α = 4 / 5) : Real.tan α = -4 / 3 := 
by sorry

theorem cos2alpha_plus_cos_alpha_add_pi_over_2 (h1 : π / 2 < α) (h2 : α < π) (h3 : Real.sin α = 4 / 5) : 
  Real.cos (2 * α) + Real.cos (α + π / 2) = -27 / 25 := 
by sorry

end tan_alpha_neg_four_thirds_cos2alpha_plus_cos_alpha_add_pi_over_2_l23_23720


namespace calculation_result_l23_23121

theorem calculation_result :
  5 * 7 - 6 * 8 + 9 * 2 + 7 * 3 = 26 :=
by sorry

end calculation_result_l23_23121


namespace fraction_simplification_l23_23445

theorem fraction_simplification :
  (1/2 * 1/3 * 1/4 * 1/5 + 3/2 * 3/4 * 3/5) / (1/2 * 2/3 * 2/5) = 41/8 :=
by
  sorry

end fraction_simplification_l23_23445


namespace smallest_x_for_three_digit_product_l23_23073

theorem smallest_x_for_three_digit_product : ∃ x : ℕ, (27 * x >= 100) ∧ (∀ y < x, 27 * y < 100) :=
by
  sorry

end smallest_x_for_three_digit_product_l23_23073


namespace intersection_of_sets_l23_23026

theorem intersection_of_sets {A B : Set Nat} (hA : A = {1, 3, 9}) (hB : B = {1, 5, 9}) :
  A ∩ B = {1, 9} :=
sorry

end intersection_of_sets_l23_23026


namespace total_horse_food_l23_23945

theorem total_horse_food (ratio_sh_to_h : ℕ → ℕ → Prop) 
    (sheep : ℕ) 
    (ounce_per_horse : ℕ) 
    (total_ounces_per_day : ℕ) : 
    ratio_sh_to_h 5 7 → sheep = 40 → ounce_per_horse = 230 → total_ounces_per_day = 12880 :=
by
  intros h_ratio h_sheep h_ounce
  sorry

end total_horse_food_l23_23945


namespace sub_number_l23_23962

theorem sub_number : 600 - 333 = 267 := by
  sorry

end sub_number_l23_23962


namespace current_year_2021_l23_23842

variables (Y : ℤ)

def parents_moved_to_America := 1982
def Aziz_age := 36
def years_before_born := 3

theorem current_year_2021
  (h1 : parents_moved_to_America = 1982)
  (h2 : Aziz_age = 36)
  (h3 : years_before_born = 3)
  (h4 : Y - (Aziz_age) - (years_before_born) = 1982) : 
  Y = 2021 :=
by {
  sorry
}

end current_year_2021_l23_23842


namespace smallest_x_satisfying_expression_l23_23612

theorem smallest_x_satisfying_expression :
  ∃ x : ℤ, (∃ k : ℤ, x^2 + x + 7 = k * (x - 2)) ∧ (∀ y : ℤ, (∃ k' : ℤ, y^2 + y + 7 = k' * (y - 2)) → y ≥ x) ∧ x = -11 :=
by
  sorry

end smallest_x_satisfying_expression_l23_23612


namespace second_crew_tractors_l23_23857

theorem second_crew_tractors
    (total_acres : ℕ)
    (days : ℕ)
    (first_crew_days : ℕ)
    (first_crew_tractors : ℕ)
    (acres_per_tractor_per_day : ℕ)
    (remaining_days : ℕ)
    (remaining_acres_after_first_crew : ℕ)
    (second_crew_acres_per_tractor : ℕ) :
    total_acres = 1700 → days = 5 → first_crew_days = 2 → first_crew_tractors = 2 → 
    acres_per_tractor_per_day = 68 → remaining_days = 3 → 
    remaining_acres_after_first_crew = total_acres - (first_crew_tractors * acres_per_tractor_per_day * first_crew_days) → 
    second_crew_acres_per_tractor = acres_per_tractor_per_day * remaining_days → 
    (remaining_acres_after_first_crew / second_crew_acres_per_tractor = 7) := 
by
  sorry

end second_crew_tractors_l23_23857


namespace box_dimensions_l23_23869

-- Given conditions
variables (a b c : ℕ)
axiom h1 : a + c = 17
axiom h2 : a + b = 13
axiom h3 : b + c = 20

theorem box_dimensions : a = 5 ∧ b = 8 ∧ c = 12 :=
by {
  -- These parts will contain the actual proof, which we omit for now
  sorry
}

end box_dimensions_l23_23869


namespace min_trials_correct_l23_23184

noncomputable def minimum_trials (α p : ℝ) (hα : 0 < α ∧ α < 1) (hp : 0 < p ∧ p < 1) : ℕ :=
  Nat.floor ((Real.log (1 - α)) / (Real.log (1 - p))) + 1

-- The theorem to prove the correctness of minimum_trials
theorem min_trials_correct (α p : ℝ) (hα : 0 < α ∧ α < 1) (hp : 0 < p ∧ p < 1) :
  ∃ n : ℕ, minimum_trials α p hα hp = n ∧ (1 - (1 - p)^n ≥ α) :=
by
  sorry

end min_trials_correct_l23_23184


namespace longest_diagonal_length_l23_23794

-- Define the conditions
variables {a b : ℝ} (h_area : 135 = 1/2 * a * b) (h_ratio : a / b = 5 / 3)

-- Define the target to prove
theorem longest_diagonal_length (a b : ℝ) (h_area : 135 = 1/2 * a * b) (h_ratio : a / b = 5 / 3) :
    a = 15 * Real.sqrt 2 :=
sorry

end longest_diagonal_length_l23_23794


namespace pet_store_animals_l23_23985

theorem pet_store_animals (cats dogs birds : ℕ) 
    (ratio_cats_dogs_birds : 2 * birds = 4 * cats ∧ 3 * cats = 2 * dogs) 
    (num_cats : cats = 20) : dogs = 30 ∧ birds = 40 :=
by 
  -- This is where the proof would go, but we can skip it for this problem statement.
  sorry

end pet_store_animals_l23_23985


namespace find_xyz_l23_23214

-- Let a, b, c, x, y, z be nonzero complex numbers
variables (a b c x y z : ℂ)
-- Given conditions
variables (h1 : a = (b + c) / (x - 2))
variables (h2 : b = (a + c) / (y - 2))
variables (h3 : c = (a + b) / (z - 2))
variables (h4 : x * y + x * z + y * z = 5)
variables (h5 : x + y + z = 3)

-- Prove that xyz = 5
theorem find_xyz : x * y * z = 5 :=
by
  sorry

end find_xyz_l23_23214


namespace storks_more_than_birds_l23_23650

-- Definitions based on given conditions
def initial_birds : ℕ := 3
def added_birds : ℕ := 2
def total_birds : ℕ := initial_birds + added_birds
def storks : ℕ := 6

-- Statement to prove the correct answer
theorem storks_more_than_birds : (storks - total_birds = 1) :=
by
  sorry

end storks_more_than_birds_l23_23650


namespace time_to_cross_signal_post_l23_23295

def train_length := 600 -- in meters
def bridge_length := 5400 -- in meters (5.4 kilometers)
def crossing_time_bridge := 6 * 60 -- in seconds (6 minutes)
def speed := bridge_length / crossing_time_bridge -- in meters per second

theorem time_to_cross_signal_post : 
  (600 / speed) = 40 :=
by
  sorry

end time_to_cross_signal_post_l23_23295


namespace smallest_n_exists_l23_23143

theorem smallest_n_exists (n k : ℕ) (h1 : 0 < n) (h2 : 0 < k) (h3 : 8 / 15 < n / (n + k)) (h4 : n / (n + k) < 7 / 13) : 
  n = 15 :=
  sorry

end smallest_n_exists_l23_23143


namespace distinct_positive_integer_roots_l23_23234

theorem distinct_positive_integer_roots (m a b : ℤ) (h1 : a > 0) (h2 : b > 0) (h3 : a ≠ b) (h4 : a + b = -m) (h5 : a * b = -m + 1) : m = -5 := 
by
  sorry

end distinct_positive_integer_roots_l23_23234


namespace intersect_eq_l23_23014

variable (M N : Set Int)
def M_def : Set Int := { m | -3 < m ∧ m < 2 }
def N_def : Set Int := { n | -1 ≤ n ∧ n ≤ 3 }

theorem intersect_eq : M_def ∩ N_def = { -1, 0, 1 } := by
  sorry

end intersect_eq_l23_23014


namespace tiffany_bags_found_day_after_next_day_l23_23470

noncomputable def tiffany_start : Nat := 10
noncomputable def tiffany_next_day : Nat := 3
noncomputable def tiffany_total : Nat := 20
noncomputable def tiffany_day_after_next_day : Nat := 20 - (tiffany_start + tiffany_next_day)

theorem tiffany_bags_found_day_after_next_day : tiffany_day_after_next_day = 7 := by
  sorry

end tiffany_bags_found_day_after_next_day_l23_23470


namespace seahawks_touchdowns_l23_23237

theorem seahawks_touchdowns (total_points : ℕ) (points_per_touchdown : ℕ) (points_per_field_goal : ℕ) (field_goals : ℕ) (touchdowns : ℕ) :
  total_points = 37 →
  points_per_touchdown = 7 →
  points_per_field_goal = 3 →
  field_goals = 3 →
  total_points = (touchdowns * points_per_touchdown) + (field_goals * points_per_field_goal) →
  touchdowns = 4 :=
by
  intros h_total_points h_points_per_touchdown h_points_per_field_goal h_field_goals h_equation
  sorry

end seahawks_touchdowns_l23_23237


namespace min_value_fraction_l23_23515

theorem min_value_fraction (x : ℝ) (h : x > 6) : 
  (∃ x_min, x_min = 12 ∧ (∀ x > 6, (x * x) / (x - 6) ≥ 18) ∧ (x * x) / (x - 6) = 18) :=
sorry

end min_value_fraction_l23_23515


namespace sequence_formula_l23_23675

theorem sequence_formula (u : ℕ → ℤ) (h0 : u 0 = 1) (h1 : u 1 = 4)
  (h_rec : ∀ n : ℕ, u (n + 2) = 5 * u (n + 1) - 6 * u n) :
  ∀ n : ℕ, u n = 2 * 3^n - 2^n :=
by 
  sorry

end sequence_formula_l23_23675


namespace n_times_2pow_nplus1_plus_1_is_square_l23_23487

theorem n_times_2pow_nplus1_plus_1_is_square (n : ℕ) (h : 0 < n) :
  ∃ m : ℤ, n * 2 ^ (n + 1) + 1 = m * m ↔ n = 3 := 
by
  sorry

end n_times_2pow_nplus1_plus_1_is_square_l23_23487


namespace smallest_n_with_divisors_l23_23521

-- Definitions of the divisors
def d_total (a b c : ℕ) : ℕ := (a + 1) * (b + 1) * (c + 1)
def d_even (a b c : ℕ) : ℕ := a * (b + 1) * (c + 1)
def d_odd (b c : ℕ) : ℕ := (b + 1) * (c + 1)

-- Math problem and proving smallest n
theorem smallest_n_with_divisors (a b c : ℕ) (n : ℕ) (h_1 : d_odd b c = 8) (h_2 : d_even a b c = 16) : n = 60 :=
  sorry

end smallest_n_with_divisors_l23_23521


namespace right_triangle_acute_angle_ratio_l23_23884

theorem right_triangle_acute_angle_ratio (A B : ℝ) (h_ratio : A / B = 5 / 4) (h_sum : A + B = 90) :
  min A B = 40 :=
by
  -- Conditions are provided
  sorry

end right_triangle_acute_angle_ratio_l23_23884


namespace decimal_to_vulgar_fraction_l23_23819

theorem decimal_to_vulgar_fraction :
  ∃ (n d : ℕ), (0.34 : ℝ) = (n : ℝ) / (d : ℝ) ∧ n = 17 :=
by
  sorry

end decimal_to_vulgar_fraction_l23_23819


namespace range_of_x8_l23_23851

theorem range_of_x8 (x : ℕ → ℝ) (h1 : 0 ≤ x 1 ∧ x 1 ≤ x 2)
  (h_recurrence : ∀ n ≥ 1, x (n+2) = x (n+1) + x n)
  (h_x7 : 1 ≤ x 7 ∧ x 7 ≤ 2) : 
  (21/13 : ℝ) ≤ x 8 ∧ x 8 ≤ (13/4) :=
sorry

end range_of_x8_l23_23851


namespace math_problem_l23_23797

theorem math_problem (a b : ℝ) (h1 : 4 + a = 5 - b) (h2 : 5 + b = 8 + a) : 4 - a = 3 :=
by
  sorry

end math_problem_l23_23797


namespace regular_polygon_sides_l23_23190

theorem regular_polygon_sides (n : ℕ) (h : 360 = 18 * n) : n = 20 := 
by 
  sorry

end regular_polygon_sides_l23_23190


namespace sum_of_minimums_is_zero_l23_23253

noncomputable def P : Polynomial ℝ := sorry
noncomputable def Q : Polynomial ℝ := sorry

-- Conditions: P(Q(x)) has zeros at -5, -3, -1, 1
lemma zeroes_PQ : 
  P.eval (Q.eval (-5)) = 0 ∧ 
  P.eval (Q.eval (-3)) = 0 ∧ 
  P.eval (Q.eval (-1)) = 0 ∧ 
  P.eval (Q.eval (1)) = 0 := 
  sorry

-- Conditions: Q(P(x)) has zeros at -7, -5, -1, 3
lemma zeroes_QP : 
  Q.eval (P.eval (-7)) = 0 ∧ 
  Q.eval (P.eval (-5)) = 0 ∧ 
  Q.eval (P.eval (-1)) = 0 ∧ 
  Q.eval (P.eval (3)) = 0 := 
  sorry

-- Definition to find the minimum value of a polynomial
noncomputable def min_value (P : Polynomial ℝ) : ℝ := sorry

-- Main theorem
theorem sum_of_minimums_is_zero :
  min_value P + min_value Q = 0 := 
  sorry

end sum_of_minimums_is_zero_l23_23253


namespace soft_drink_cost_l23_23642

/-- Benny bought 2 soft drinks for a certain price each and 5 candy bars.
    He spent a total of $28. Each candy bar cost $4. 
    Prove that the cost of each soft drink was $4.
--/
theorem soft_drink_cost (S : ℝ) (h1 : 2 * S + 5 * 4 = 28) : S = 4 := 
by
  sorry

end soft_drink_cost_l23_23642


namespace proof_problem_l23_23035

theorem proof_problem
  (x y : ℚ)
  (h1 : 4 * x + 2 * y = 12)
  (h2 : 2 * x + 4 * y = 16) :
  20 * x^2 + 24 * x * y + 20 * y^2 = 3280 / 9 :=
sorry

end proof_problem_l23_23035


namespace least_number_to_add_l23_23081

theorem least_number_to_add (x : ℕ) (h : 1055 % 23 = 20) : x = 3 :=
by
  -- Proof goes here.
  sorry

end least_number_to_add_l23_23081


namespace maxwell_meets_brad_l23_23082

variable (t : ℝ) -- time in hours
variable (distance_between_homes : ℝ) -- total distance
variable (maxwell_speed : ℝ) -- Maxwell's walking speed
variable (brad_speed : ℝ) -- Brad's running speed
variable (brad_delay : ℝ) -- Brad's start time delay

theorem maxwell_meets_brad 
  (hb: brad_delay = 1)
  (d: distance_between_homes = 34)
  (v_m: maxwell_speed = 4)
  (v_b: brad_speed = 6)
  (h : 4 * t + 6 * (t - 1) = distance_between_homes) :
  t = 4 := 
  sorry

end maxwell_meets_brad_l23_23082


namespace solve_system_of_equations_l23_23310

theorem solve_system_of_equations :
  ∃ x : ℕ → ℝ,
  (∀ i : ℕ, i < 100 → x i > 0) ∧
  (x 0 + 1 / x 1 = 4) ∧
  (x 1 + 1 / x 2 = 1) ∧
  (x 2 + 1 / x 0 = 4) ∧
  (∀ i : ℕ, 1 ≤ i ∧ i < 99 → x (2 * i + 1) + 1 / x (2 * i + 2) = 1) ∧
  (∀ i : ℕ, 1 ≤ i ∧ i < 99 → x (2 * i + 2) + 1 / x (2 * i + 3) = 4) ∧
  (x 99 + 1 / x 0 = 1) ∧
  (∀ i : ℕ, i < 50 → x (2 * i) = 2) ∧
  (∀ i : ℕ, i < 50 → x (2 * i + 1) = 1 / 2) :=
sorry

end solve_system_of_equations_l23_23310


namespace find_angle_between_vectors_l23_23890

open Real

noncomputable def vector_norm (v : ℝ × ℝ) : ℝ := sqrt (v.1 ^ 2 + v.2 ^ 2)
noncomputable def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem find_angle_between_vectors 
  (a b : ℝ × ℝ)
  (a_nonzero : a ≠ (0, 0))
  (b_nonzero : b ≠ (0, 0))
  (ha : vector_norm a = 2)
  (hb : vector_norm b = 3)
  (h_sum : vector_norm (a.1 + b.1, a.2 + b.2) = 1)
  : arccos (dot_product a b / (vector_norm a * vector_norm b)) = π :=
sorry

end find_angle_between_vectors_l23_23890


namespace goods_train_speed_l23_23684

theorem goods_train_speed :
  ∀ (length_train length_platform time : ℝ),
    length_train = 250.0416 →
    length_platform = 270 →
    time = 26 →
    (length_train + length_platform) / time = 20 :=
by
  intros length_train length_platform time H_train H_platform H_time
  rw [H_train, H_platform, H_time]
  norm_num
  sorry

end goods_train_speed_l23_23684


namespace Warriors_won_25_games_l23_23648

def CricketResults (Sharks Falcons Warriors Foxes Knights : ℕ) :=
  Sharks > Falcons ∧
  (Warriors > Foxes ∧ Warriors < Knights) ∧
  Foxes > 15 ∧
  (Foxes = 20 ∨ Foxes = 25 ∨ Foxes = 30) ∧
  (Warriors = 20 ∨ Warriors = 25 ∨ Warriors = 30) ∧
  (Knights = 20 ∨ Knights = 25 ∨ Knights = 30)

theorem Warriors_won_25_games (Sharks Falcons Warriors Foxes Knights : ℕ) 
  (h : CricketResults Sharks Falcons Warriors Foxes Knights) :
  Warriors = 25 :=
by
  sorry

end Warriors_won_25_games_l23_23648


namespace shifts_needed_l23_23284

-- Given definitions
def total_workers : ℕ := 12
def workers_per_shift : ℕ := 2
def total_ways_to_assign : ℕ := 23760

-- Prove the number of shifts needed
theorem shifts_needed : total_workers / workers_per_shift = 6 := by
  sorry

end shifts_needed_l23_23284


namespace smallest_possible_fourth_number_l23_23236

theorem smallest_possible_fourth_number 
  (a b : ℕ) 
  (h1 : 21 + 34 + 65 = 120)
  (h2 : 1 * (21 + 34 + 65 + 10 * a + b) = 4 * (2 + 1 + 3 + 4 + 6 + 5 + a + b)) :
  10 * a + b = 12 := 
sorry

end smallest_possible_fourth_number_l23_23236


namespace smallest_integer_greater_than_100_with_gcd_24_eq_4_l23_23824

theorem smallest_integer_greater_than_100_with_gcd_24_eq_4 :
  ∃ x : ℤ, x > 100 ∧ x % 24 = 4 ∧ (∀ y : ℤ, y > 100 ∧ y % 24 = 4 → x ≤ y) :=
sorry

end smallest_integer_greater_than_100_with_gcd_24_eq_4_l23_23824


namespace find_a_range_l23_23403

noncomputable def f (x : ℝ) := (x - 1) / Real.exp x

noncomputable def condition_holds (a : ℝ) : Prop :=
∀ t ∈ (Set.Icc (1/2 : ℝ) 2), f t > t

theorem find_a_range (a : ℝ) (h : condition_holds a) : a > Real.exp 2 + 1/2 := sorry

end find_a_range_l23_23403


namespace least_positive_integer_property_l23_23381

theorem least_positive_integer_property : 
  ∃ (n d : ℕ) (p : ℕ) (h₁ : 1 ≤ d) (h₂ : d ≤ 9) (h₃ : p ≥ 2), 
  (10^p * d = 24 * n) ∧ (∃ k : ℕ, (n = 100 * 10^(p-2) / 3) ∧ (900 = 8 * 10^p + 100 / 3 * 10^(p-2))) := sorry

end least_positive_integer_property_l23_23381


namespace sqrt_floor_eq_sqrt_sqrt_floor_l23_23711

theorem sqrt_floor_eq_sqrt_sqrt_floor (a : ℝ) (h : a > 1) :
  Int.floor (Real.sqrt (Int.floor (Real.sqrt a))) = Int.floor (Real.sqrt (Real.sqrt a)) :=
sorry

end sqrt_floor_eq_sqrt_sqrt_floor_l23_23711


namespace centripetal_accel_v_r_centripetal_accel_omega_r_centripetal_accel_T_r_l23_23964

-- Defining the variables involved
variables {a v r ω T : ℝ}

-- Main theorem statements representing the problem
theorem centripetal_accel_v_r (v r : ℝ) (h₁ : 0 < r) : a = v^2 / r :=
sorry

theorem centripetal_accel_omega_r (ω r : ℝ) (h₁ : 0 < r) : a = r * ω^2 :=
sorry

theorem centripetal_accel_T_r (T r : ℝ) (h₁ : 0 < r) (h₂ : 0 < T) : a = 4 * π^2 * r / T^2 :=
sorry

end centripetal_accel_v_r_centripetal_accel_omega_r_centripetal_accel_T_r_l23_23964


namespace floor_sqrt_77_l23_23370

theorem floor_sqrt_77 : 8 < Real.sqrt 77 ∧ Real.sqrt 77 < 9 → Int.floor (Real.sqrt 77) = 8 :=
by
  sorry

end floor_sqrt_77_l23_23370


namespace probability_of_region_l23_23637

-- Definition of the bounds
def bounds (x y : ℝ) : Prop := 0 ≤ x ∧ x ≤ 4 ∧ 0 ≤ y ∧ y ≤ 8

-- Definition of the region where x + y <= 5
def region (x y : ℝ) : Prop := x + y ≤ 5

-- The proof statement
theorem probability_of_region : 
  (∃ (x y : ℝ), bounds x y ∧ region x y) →
  ∃ (p : ℚ), p = 3/8 :=
by sorry

end probability_of_region_l23_23637


namespace negation_of_p_is_neg_p_l23_23986

-- Define the proposition p
def p : Prop := ∀ n : ℕ, 3^n ≥ n^2 + 1

-- Define the negation of p
def neg_p : Prop := ∃ n_0 : ℕ, 3^n_0 < n_0^2 + 1

-- The proof statement
theorem negation_of_p_is_neg_p : ¬p ↔ neg_p :=
by sorry

end negation_of_p_is_neg_p_l23_23986


namespace minimum_value_of_3a_plus_b_l23_23305

theorem minimum_value_of_3a_plus_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 / a + 1 / b = 2) : 
  3 * a + b ≥ (7 + 2 * Real.sqrt 6) / 2 :=
sorry

end minimum_value_of_3a_plus_b_l23_23305


namespace vieta_formula_l23_23953

-- Define what it means to be a root of a polynomial
noncomputable def is_root (p : ℝ) (a b c d : ℝ) : Prop :=
  a * p^3 + b * p^2 + c * p + d = 0

-- Setting up the variables and conditions for the polynomial
variables (p q r : ℝ)
variable (a b c d : ℝ)
variable (ha : a = 5)
variable (hb : b = -10)
variable (hc : c = 17)
variable (hd : d = -7)
variable (hp : is_root p a b c d)
variable (hq : is_root q a b c d)
variable (hr : is_root r a b c d)

-- Lean statement to prove the desired equality using Vieta's formulas
theorem vieta_formula : 
  pq + qr + rp = c / a :=
by
  -- Translate the problem into Lean structure
  sorry

end vieta_formula_l23_23953


namespace quadratic_increasing_l23_23718

theorem quadratic_increasing (x : ℝ) (hx : x > 1) : ∃ y : ℝ, y = (x-1)^2 + 1 ∧ ∀ (x₁ x₂ : ℝ), x₁ > x ∧ x₂ > x₁ → (x₁ - 1)^2 + 1 < (x₂ - 1)^2 + 1 := by
  sorry

end quadratic_increasing_l23_23718


namespace root_sum_greater_than_one_l23_23049

noncomputable def f (x a : ℝ) : ℝ := (x * Real.log x) / (x - 1) - a

noncomputable def h (x a : ℝ) : ℝ := (x^2 - x) * f x a

theorem root_sum_greater_than_one {a m x1 x2 : ℝ} (ha : a < 0)
  (h_eq_m : ∀ x, h x a = m) (hx1_root : h x1 a = m) (hx2_root : h x2 a = m)
  (hx1x2_distinct : x1 ≠ x2) :
  x1 + x2 > 1 := 
sorry

end root_sum_greater_than_one_l23_23049


namespace payment_option1_payment_option2_cost_effective_option_most_cost_effective_plan_l23_23157

variable (x : ℕ)
variable (hx : x > 10)

noncomputable def option1_payment (x : ℕ) : ℕ := 200 * x + 8000
noncomputable def option2_payment (x : ℕ) : ℕ := 180 * x + 9000

theorem payment_option1 (x : ℕ) (hx : x > 10) : option1_payment x = 200 * x + 8000 :=
by sorry

theorem payment_option2 (x : ℕ) (hx : x > 10) : option2_payment x = 180 * x + 9000 :=
by sorry

theorem cost_effective_option (x : ℕ) (hx : x > 10) (h30 : x = 30) : option1_payment 30 < option2_payment 30 :=
by sorry

theorem most_cost_effective_plan (h30 : x = 30) : (10000 + 3600 = 13600) :=
by sorry

end payment_option1_payment_option2_cost_effective_option_most_cost_effective_plan_l23_23157


namespace values_of_m_l23_23987

theorem values_of_m (m : ℝ) : 
  (∀ x : ℝ, (3 * x^2 + (2 - m) * x + 12 = 0)) ↔ (m = -10 ∨ m = 14) := 
by
  sorry

end values_of_m_l23_23987


namespace percentage_of_x_l23_23469

theorem percentage_of_x (x : ℝ) (h : x > 0) : ((x / 5 + x / 25) / x) * 100 = 24 := 
by 
  sorry

end percentage_of_x_l23_23469


namespace calculate_expression_l23_23647

theorem calculate_expression (m n : ℝ) : 9 * m^2 - (m - 2 * n)^2 = 4 * (2 * m - n) * (m + n) :=
by
  sorry

end calculate_expression_l23_23647


namespace binom_1450_2_eq_1050205_l23_23340

def binom_coefficient (n k : ℕ) : ℕ :=
  n.choose k

theorem binom_1450_2_eq_1050205 : binom_coefficient 1450 2 = 1050205 :=
by {
  sorry
}

end binom_1450_2_eq_1050205_l23_23340


namespace farmer_plow_l23_23183

theorem farmer_plow (P : ℕ) (M : ℕ) (H1 : M = 12) (H2 : 8 * P + M * (8 - (55 / P)) = 30) (H3 : 55 % P = 0) : P = 10 :=
by
  sorry

end farmer_plow_l23_23183


namespace solve_quadratic_l23_23331

theorem solve_quadratic (x : ℝ) : 2 * x^2 - x = 2 ↔ x = (1 + Real.sqrt 17) / 4 ∨ x = (1 - Real.sqrt 17) / 4 := by
  sorry

end solve_quadratic_l23_23331


namespace miles_driven_l23_23336

-- Definitions based on the conditions
def years : ℕ := 9
def months_in_a_year : ℕ := 12
def months_in_a_period : ℕ := 4
def miles_per_period : ℕ := 37000

-- The proof statement
theorem miles_driven : years * months_in_a_year / months_in_a_period * miles_per_period = 999000 := 
sorry

end miles_driven_l23_23336


namespace fox_initial_coins_l23_23633

theorem fox_initial_coins :
  ∃ x : ℤ, x - 10 = 0 ∧ 2 * (x - 10) - 50 = 0 ∧ 2 * (2 * (x - 10) - 50) - 50 = 0 ∧
  2 * (2 * (2 * (x - 10) - 50) - 50) - 50 = 0 ∧ 2 * (2 * (2 * (2 * (x - 10) - 50) - 50) - 50) - 50 = 0 ∧
  x = 56 := 
by
  -- we skip the proof here
  sorry

end fox_initial_coins_l23_23633


namespace answered_both_l23_23807

variables (A B : Type)
variables {test_takers : Type}

-- Defining the conditions
def pa : ℝ := 0.80  -- 80% answered first question correctly
def pb : ℝ := 0.75  -- 75% answered second question correctly
def pnone : ℝ := 0.05 -- 5% answered neither question correctly

-- Formal problem statement
theorem answered_both (test_takers: Type) : 
  (pa + pb - (1 - pnone)) = 0.60 :=
by
  sorry

end answered_both_l23_23807


namespace part1_part2_l23_23613

def A : Set ℝ := {x | (x - 5) / (x + 1) ≤ 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*x - m < 0}
def C : Set ℝ := {x | -1 < x ∧ x < 4}

theorem part1 : A ∩ (B 3)ᶜ = Set.Icc 3 5 := by
  sorry

theorem part2 : A ∩ B m = C → m = 8 := by
  sorry

end part1_part2_l23_23613


namespace upper_limit_for_y_l23_23688

theorem upper_limit_for_y (x y : ℝ) (hx : 5 < x) (hx' : x < 8) (hy : 8 < y) (h_diff : y - x = 7) : y ≤ 14 :=
by
  sorry

end upper_limit_for_y_l23_23688


namespace how_many_tuna_l23_23863

-- Definitions for conditions
variables (customers : ℕ) (weightPerTuna : ℕ) (weightPerCustomer : ℕ)
variables (unsatisfiedCustomers : ℕ)

-- Hypotheses based on the problem conditions
def conditions :=
  customers = 100 ∧
  weightPerTuna = 200 ∧
  weightPerCustomer = 25 ∧
  unsatisfiedCustomers = 20

-- Statement to prove how many tuna Mr. Ray needs
theorem how_many_tuna (h : conditions customers weightPerTuna weightPerCustomer unsatisfiedCustomers) : 
  ∃ n, n = 10 :=
by
  sorry

end how_many_tuna_l23_23863


namespace focal_distance_of_ellipse_l23_23896

theorem focal_distance_of_ellipse : 
  ∀ (θ : ℝ), (∃ (c : ℝ), (x = 5 * Real.cos θ ∧ y = 4 * Real.sin θ) → 2 * c = 6) :=
by
  sorry

end focal_distance_of_ellipse_l23_23896


namespace largest_A_smallest_A_l23_23833

noncomputable def is_coprime_with_12 (n : Nat) : Prop :=
  Nat.gcd n 12 = 1

noncomputable def rotated_number (n : Nat) : Option Nat :=
  if n < 10^7 then none else
  let b := n % 10
  let k := n / 10
  some (b * 10^7 + k)

noncomputable def satisfies_conditions (B : Nat) : Prop :=
  B > 44444444 ∧ is_coprime_with_12 B

theorem largest_A :
  ∃ (B : Nat), satisfies_conditions B ∧ rotated_number B = some 99999998 :=
sorry

theorem smallest_A :
  ∃ (B : Nat), satisfies_conditions B ∧ rotated_number B = some 14444446 :=
sorry

end largest_A_smallest_A_l23_23833


namespace students_in_grades_2_and_3_l23_23972

theorem students_in_grades_2_and_3 (boys_2nd : ℕ) (girls_2nd : ℕ) (third_grade_factor : ℕ) 
  (h_boys_2nd : boys_2nd = 20) (h_girls_2nd : girls_2nd = 11) (h_third_grade_factor : third_grade_factor = 2) :
  (boys_2nd + girls_2nd) + ((boys_2nd + girls_2nd) * third_grade_factor) = 93 := by
  sorry

end students_in_grades_2_and_3_l23_23972


namespace arccos_cos_11_eq_l23_23189

theorem arccos_cos_11_eq: Real.arccos (Real.cos 11) = 11 - 3 * Real.pi := by
  sorry

end arccos_cos_11_eq_l23_23189


namespace pure_imaginary_condition_l23_23815

-- Define the problem
theorem pure_imaginary_condition (θ : ℝ) :
  (∀ k : ℤ, θ = (3 * Real.pi / 4) + k * Real.pi) →
  ∀ z : ℂ, z = (Complex.cos θ - Complex.sin θ * Complex.I) * (1 + Complex.I) →
  ∃ k : ℤ, θ = (3 * Real.pi / 4) + k * Real.pi → 
  (Complex.re z = 0 ∧ Complex.im z ≠ 0) :=
  sorry

end pure_imaginary_condition_l23_23815


namespace discount_limit_l23_23068

theorem discount_limit {cost_price selling_price : ℕ} (x : ℚ)
  (h1: cost_price = 100)
  (h2: selling_price = 150)
  (h3: ∃ p : ℚ, p = 1.2 * cost_price) : selling_price * (x / 10) - cost_price ≥ 0.2 * cost_price ↔ x ≤ 8 :=
by {
  sorry
}

end discount_limit_l23_23068


namespace problem_1_solution_problem_2_solution_problem_3_solution_problem_4_solution_l23_23449

noncomputable def problem_1 : Int :=
  (-3) + 5 - (-3)

theorem problem_1_solution : problem_1 = 5 := by
  sorry

noncomputable def problem_2 : ℚ :=
  (-1/3 - 3/4 + 5/6) * (-24)

theorem problem_2_solution : problem_2 = 6 := by
  sorry

noncomputable def problem_3 : ℚ :=
  1 - (1/9) * (-1/2 - 2^2)

theorem problem_3_solution : problem_3 = 3/2 := by
  sorry

noncomputable def problem_4 : ℚ :=
  ((-1)^2023) * (18 - (-2) * 3) / (15 - 3^3)

theorem problem_4_solution : problem_4 = 2 := by
  sorry

end problem_1_solution_problem_2_solution_problem_3_solution_problem_4_solution_l23_23449


namespace sequence_term_2023_l23_23201

theorem sequence_term_2023 (a : ℕ → ℚ) (h₁ : a 1 = 2) 
  (h₂ : ∀ n, 1 / a n - 1 / a (n + 1) - 1 / (a n * a (n + 1)) = 1) : 
  a 2023 = -1 / 2 := 
sorry

end sequence_term_2023_l23_23201


namespace worksheets_graded_l23_23821

theorem worksheets_graded (w : ℕ) (h1 : ∀ (n : ℕ), n = 3) (h2 : ∀ (n : ℕ), n = 15) (h3 : ∀ (p : ℕ), p = 24)  :
  w = 7 :=
sorry

end worksheets_graded_l23_23821


namespace total_surface_area_correct_l23_23759

-- Defining the dimensions of the rectangular solid
def length := 10
def width := 9
def depth := 6

-- Definition of the total surface area of a rectangular solid
def surface_area (l w d : ℕ) := 2 * (l * w + w * d + l * d)

-- Proposition that the total surface area for the given dimensions is 408 square meters
theorem total_surface_area_correct : surface_area length width depth = 408 := 
by
  sorry

end total_surface_area_correct_l23_23759


namespace inequality_abcd_l23_23604

theorem inequality_abcd (a b c d : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) (h₄ : 0 < d) :
    (a / (b + 2*c + 3*d) + b / (c + 2*d + 3*a) + c / (d + 2*a + 3*b) + d / (a + 2*b + 3*c) >= 2 / 3) :=
by
  sorry

end inequality_abcd_l23_23604


namespace sum_of_squares_not_perfect_square_l23_23274

theorem sum_of_squares_not_perfect_square (n : ℕ) (h : n > 4) :
  ¬ (∃ k : ℕ, 10 * n^2 + 10 * n + 85 = k^2) :=
sorry

end sum_of_squares_not_perfect_square_l23_23274


namespace find_integer_pairs_l23_23245

-- Define the plane and lines properties
def horizontal_lines (h : ℕ) : Prop := h > 0
def non_horizontal_lines (s : ℕ) : Prop := s > 0
def non_parallel (s : ℕ) : Prop := s > 0
def no_three_intersect (total_lines : ℕ) : Prop := total_lines > 0

-- Function to calculate regions from the given formula
def calculate_regions (h s : ℕ) : ℕ :=
  h * (s + 1) + 1 + (s * (s + 1)) / 2

-- Prove that the given (h, s) pairs divide the plane into 1992 regions
theorem find_integer_pairs :
  (horizontal_lines 995 ∧ non_horizontal_lines 1 ∧ non_parallel 1 ∧ no_three_intersect (995 + 1) ∧ calculate_regions 995 1 = 1992)
  ∨ (horizontal_lines 176 ∧ non_horizontal_lines 10 ∧ non_parallel 10 ∧ no_three_intersect (176 + 10) ∧ calculate_regions 176 10 = 1992)
  ∨ (horizontal_lines 80 ∧ non_horizontal_lines 21 ∧ non_parallel 21 ∧ no_three_intersect (80 + 21) ∧ calculate_regions 80 21 = 1992) :=
by
  -- Include individual cases to verify correctness of regions calculation
  sorry

end find_integer_pairs_l23_23245


namespace area_of_ABC_l23_23229

def point : Type := ℝ × ℝ

def area_of_triangle (A B C : point) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem area_of_ABC : area_of_triangle (0, 0) (1, 0) (0, 1) = 0.5 :=
by
  sorry

end area_of_ABC_l23_23229


namespace sqrt_mult_pow_l23_23796

theorem sqrt_mult_pow (a : ℝ) (h_nonneg : 0 ≤ a) : (a^(2/3) * a^(1/5)) = a^(13/15) := by
  sorry

end sqrt_mult_pow_l23_23796


namespace red_beads_cost_l23_23193

theorem red_beads_cost (R : ℝ) (H : 4 * R + 4 * 2 = 10 * 1.72) : R = 2.30 :=
by
  sorry

end red_beads_cost_l23_23193


namespace range_of_a_l23_23763

theorem range_of_a (a : ℝ) :
  (∃ x y : ℝ, x^2 - x + (a - 4) = 0 ∧ y^2 - y + (a - 4) = 0 ∧ x > 0 ∧ y < 0) → a < 4 :=
by
  sorry

end range_of_a_l23_23763


namespace region_area_correct_l23_23712

noncomputable def region_area : ℝ :=
  let region := {p : ℝ × ℝ | |p.1 + p.2| + |p.1 - p.2| ≤ 6}
  let area := (3 - -3) * (3 - -3)
  area

theorem region_area_correct : region_area = 36 :=
by sorry

end region_area_correct_l23_23712


namespace reeya_fourth_subject_score_l23_23332

theorem reeya_fourth_subject_score (s1 s2 s3 s4 : ℕ) (avg : ℕ) (n : ℕ)
  (h_avg : avg = 75) (h_n : n = 4) (h_s1 : s1 = 65) (h_s2 : s2 = 67) (h_s3 : s3 = 76)
  (h_total_sum : avg * n = s1 + s2 + s3 + s4) : s4 = 92 := by
  sorry

end reeya_fourth_subject_score_l23_23332


namespace inequality_proof_l23_23907

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  ab * (a + b) + bc * (b + c) + ac * (a + c) ≥ 6 * abc := 
sorry

end inequality_proof_l23_23907


namespace adam_change_is_correct_l23_23149

-- Define the conditions
def adam_money : ℝ := 5.00
def airplane_cost : ℝ := 4.28
def change : ℝ := adam_money - airplane_cost

-- State the theorem
theorem adam_change_is_correct : change = 0.72 := 
by {
  -- Proof can be added later
  sorry
}

end adam_change_is_correct_l23_23149


namespace total_cost_proof_l23_23774

def uber_cost : ℤ := 22
def lyft_cost : ℤ := uber_cost - 3
def taxi_cost : ℤ := lyft_cost - 4
def tip : ℤ := (taxi_cost * 20) / 100
def total_cost : ℤ := taxi_cost + tip

theorem total_cost_proof :
  total_cost = 18 :=
by
  sorry

end total_cost_proof_l23_23774


namespace pencil_pen_costs_l23_23337

noncomputable def cost_of_items (p q : ℝ) : ℝ := 4 * p + 4 * q

theorem pencil_pen_costs (p q : ℝ) (h1 : 6 * p + 3 * q = 5.40) (h2 : 3 * p + 5 * q = 4.80) : cost_of_items p q = 4.80 :=
by
  sorry

end pencil_pen_costs_l23_23337


namespace quadrilateral_AD_length_l23_23788

noncomputable def length_AD (AB BC CD : ℝ) (angleB angleC : ℝ) : ℝ :=
  let AE := AB + BC * Real.cos angleC
  let CE := BC * Real.sin angleC
  let DE := CD - CE
  Real.sqrt (AE^2 + DE^2)

theorem quadrilateral_AD_length :
  let AB := 7
  let BC := 10
  let CD := 24
  let angleB := Real.pi / 2 -- 90 degrees in radians
  let angleC := Real.pi / 3 -- 60 degrees in radians
  length_AD AB BC CD angleB angleC = Real.sqrt (795 - 240 * Real.sqrt 3) :=
by
  sorry

end quadrilateral_AD_length_l23_23788


namespace factors_of_48_are_multiples_of_6_l23_23899

theorem factors_of_48_are_multiples_of_6 : 
  ∃ (n : ℕ), n = 4 ∧ ∀ d, d ∣ 48 → (6 ∣ d ↔ d = 6 ∨ d = 12 ∨ d = 24 ∨ d = 48) := 
by { sorry }

end factors_of_48_are_multiples_of_6_l23_23899


namespace remainder_modulo_seven_l23_23509

theorem remainder_modulo_seven (n : ℕ)
  (h₁ : n^2 % 7 = 1)
  (h₂ : n^3 % 7 = 6) :
  n % 7 = 6 :=
sorry

end remainder_modulo_seven_l23_23509


namespace complement_intersection_l23_23518

/-- Given the universal set U={1,2,3,4,5},
    A={2,3,4}, and B={1,2,3}, 
    Prove the complement of (A ∩ B) in U is {1,4,5}. -/
theorem complement_intersection 
    (U : Set ℕ) (A : Set ℕ) (B : Set ℕ) 
    (hU : U = {1, 2, 3, 4, 5})
    (hA : A = {2, 3, 4})
    (hB : B = {1, 2, 3}) :
    U \ (A ∩ B) = {1, 4, 5} :=
by
  -- proof goes here
  sorry

end complement_intersection_l23_23518


namespace root_sum_value_l23_23701

theorem root_sum_value (r s t : ℝ) (h1: r + s + t = 24) (h2: r * s + s * t + t * r = 50) (h3: r * s * t = 24) :
  r / (1/r + s * t) + s / (1/s + t * r) + t / (1/t + r * s) = 19.04 :=
sorry

end root_sum_value_l23_23701


namespace triangles_in_divided_square_l23_23620

theorem triangles_in_divided_square (V : ℕ) (marked_points : ℕ) (triangles : ℕ) 
  (h1 : V = 24) -- Vertices - 20 marked points and 4 vertices 
  (h2 : marked_points = 20) -- Marked points
  (h3 : triangles = F - 1) -- Each face (F) except the outer one is a triangle
  (h4 : V - E + F = 2) -- Euler's formula for planar graphs
  (h5 : E = (3*F + 1) / 2) -- Relationship between edges and faces
  (F : ℕ) -- Number of faces including the external face
  (E : ℕ) -- Number of edges
  : triangles = 42 := 
by 
  sorry

end triangles_in_divided_square_l23_23620


namespace f_one_minus_a_l23_23050

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function (x : ℝ) : f (-x) = -f x
axiom periodic_function (x : ℝ) : f (x + 2) = f x
axiom f_one_plus_a {a : ℝ} : f (1 + a) = 1

theorem f_one_minus_a (a : ℝ) : f (1 - a) = -1 :=
by
  sorry

end f_one_minus_a_l23_23050


namespace range_of_a_l23_23117

open Set

variable {a : ℝ}
def M (a : ℝ) : Set ℝ := { x : ℝ | (2 * a - 1) < x ∧ x < (4 * a) }
def N : Set ℝ := { x : ℝ | 1 < x ∧ x < 2 }

theorem range_of_a (h : N ⊆ M a) : 1 / 2 ≤ a ∧ a ≤ 2 := sorry

end range_of_a_l23_23117


namespace sum_of_cubes_of_real_roots_eq_11_l23_23767

-- Define the polynomial f(x) = x^3 - 2x^2 - x + 1
def poly (x : ℝ) : ℝ := x^3 - 2 * x^2 - x + 1

-- State that the polynomial has exactly three real roots
axiom three_real_roots : ∃ (x1 x2 x3 : ℝ), poly x1 = 0 ∧ poly x2 = 0 ∧ poly x3 = 0

-- Prove that the sum of the cubes of the real roots is 11
theorem sum_of_cubes_of_real_roots_eq_11 (x1 x2 x3 : ℝ)
  (hx1 : poly x1 = 0) (hx2 : poly x2 = 0) (hx3 : poly x3 = 0) : 
  x1^3 + x2^3 + x3^3 = 11 :=
by
  sorry

end sum_of_cubes_of_real_roots_eq_11_l23_23767


namespace ascending_function_k_ge_2_l23_23491

open Real

def is_ascending (f : ℝ → ℝ) (k : ℝ) (M : Set ℝ) : Prop :=
  ∀ x ∈ M, f (x + k) ≥ f x

theorem ascending_function_k_ge_2 :
  ∀ (k : ℝ), (∀ x : ℝ, x ≥ -1 → (x + k) ^ 2 ≥ x ^ 2) → k ≥ 2 :=
by
  intros k h
  sorry

end ascending_function_k_ge_2_l23_23491


namespace w_share_l23_23582

theorem w_share (k : ℝ) (w x y z : ℝ) (h1 : w = k) (h2 : x = 6 * k) (h3 : y = 2 * k) (h4 : z = 4 * k) (h5 : x - y = 1500):
  w = 375 := by
  /- Lean code to show w = 375 -/
  sorry

end w_share_l23_23582


namespace animal_stickers_l23_23733

theorem animal_stickers {flower stickers total_stickers animal_stickers : ℕ} 
  (h_flower_stickers : flower = 8) 
  (h_total_stickers : total_stickers = 14)
  (h_total_eq : total_stickers = flower + animal_stickers) : 
  animal_stickers = 6 :=
by
  sorry

end animal_stickers_l23_23733


namespace tan_sum_property_l23_23682

theorem tan_sum_property (t23 t37 : ℝ) (h1 : 23 + 37 = 60) (h2 : Real.tan (60 * Real.pi / 180) = Real.sqrt 3) :
  Real.tan (23 * Real.pi / 180) + Real.tan (37 * Real.pi / 180) + Real.sqrt 3 * Real.tan (23 * Real.pi / 180) * Real.tan (37 * Real.pi / 180) = Real.sqrt 3 :=
sorry

end tan_sum_property_l23_23682


namespace a_must_be_negative_l23_23066

variable (a b c d e : ℝ)

theorem a_must_be_negative
  (h1 : a / b < -c / d)
  (hb : b > 0)
  (hd : d > 0)
  (he : e > 0)
  (h2 : a + e > 0) : a < 0 := by
  sorry

end a_must_be_negative_l23_23066


namespace chocolate_bars_per_box_l23_23860

theorem chocolate_bars_per_box (total_chocolate_bars num_small_boxes : ℕ) (h1 : total_chocolate_bars = 300) (h2 : num_small_boxes = 15) : 
  total_chocolate_bars / num_small_boxes = 20 :=
by 
  sorry

end chocolate_bars_per_box_l23_23860


namespace complex_multiplication_l23_23738

-- Define i such that i^2 = -1
def i : ℂ := Complex.I

theorem complex_multiplication : (3 - 4 * i) * (-7 + 6 * i) = 3 + 46 * i := by
  sorry

end complex_multiplication_l23_23738


namespace monotone_range_of_f_l23_23485

theorem monotone_range_of_f {f : ℝ → ℝ} (a : ℝ) 
  (h : ∀ x y : ℝ, 0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1 ∧ x ≤ y → f x ≤ f y) : a ≤ 0 :=
sorry

end monotone_range_of_f_l23_23485


namespace simplify_fraction_l23_23054

variable {a b c k : ℝ}
variable (h : a * b = c * k ∧ a * b ≠ 0)

theorem simplify_fraction (h : a * b = c * k ∧ a * b ≠ 0) : 
  (a - b - c + k) / (a + b + c + k) = (a - c) / (a + c) :=
by
  sorry

end simplify_fraction_l23_23054


namespace interest_rate_b_to_c_l23_23150

open Real

noncomputable def calculate_rate_b_to_c (P : ℝ) (r1 : ℝ) (t : ℝ) (G : ℝ) : ℝ :=
  let I_a_b := P * (r1 / 100) * t
  let I_b_c := I_a_b + G
  (100 * I_b_c) / (P * t)

theorem interest_rate_b_to_c :
  calculate_rate_b_to_c 3200 12 5 400 = 14.5 := by
  sorry

end interest_rate_b_to_c_l23_23150


namespace eval_expr_l23_23703

theorem eval_expr : 4 * (-3) + 60 / (-15) = -16 := by
  sorry

end eval_expr_l23_23703


namespace integer_roots_and_composite_l23_23544

theorem integer_roots_and_composite (a b : ℤ) (h1 : ∃ x1 x2 : ℤ, x1 * x2 = 1 - b ∧ x1 + x2 = -a) (h2 : b ≠ 1) : 
  ∃ (m n : ℕ), m > 1 ∧ n > 1 ∧ m * n = (a^2 + b^2) := 
sorry

end integer_roots_and_composite_l23_23544


namespace find_length_of_second_movie_l23_23404

noncomputable def length_of_second_movie := 1.5

theorem find_length_of_second_movie
  (total_free_time : ℝ)
  (first_movie_duration : ℝ)
  (words_read : ℝ)
  (reading_rate : ℝ) : 
  first_movie_duration = 3.5 → 
  total_free_time = 8 → 
  words_read = 1800 → 
  reading_rate = 10 → 
  length_of_second_movie = 1.5 := 
by
  intros h1 h2 h3 h4
  -- Here should be the proof steps, which are abstracted away.
  sorry

end find_length_of_second_movie_l23_23404


namespace ratio_of_circumscribed_areas_l23_23104

noncomputable def rect_pentagon_circ_ratio (P : ℝ) : ℝ :=
  let s : ℝ := P / 8
  let r_circle : ℝ := (P * Real.sqrt 10) / 16
  let A : ℝ := Real.pi * (r_circle ^ 2)
  let pentagon_side : ℝ := P / 5
  let R_pentagon : ℝ := P / (10 * Real.sin (Real.pi / 5))
  let B : ℝ := Real.pi * (R_pentagon ^ 2)
  A / B

theorem ratio_of_circumscribed_areas (P : ℝ) : rect_pentagon_circ_ratio P = (5 * (5 - Real.sqrt 5)) / 64 :=
by sorry

end ratio_of_circumscribed_areas_l23_23104


namespace radiator_water_fraction_l23_23739

noncomputable def fraction_of_water_after_replacements (initial_water : ℚ) (initial_antifreeze : ℚ) (removal_fraction : ℚ)
  (num_replacements : ℕ) : ℚ :=
  initial_water * (removal_fraction ^ num_replacements)

theorem radiator_water_fraction :
  let initial_water := 10
  let initial_antifreeze := 10
  let total_volume := 20
  let removal_volume := 5
  let removal_fraction := 3 / 4
  let num_replacements := 4
  fraction_of_water_after_replacements initial_water initial_antifreeze removal_fraction num_replacements / total_volume = 0.158 := 
sorry

end radiator_water_fraction_l23_23739


namespace number_of_birds_l23_23242

-- Conditions
def geese : ℕ := 58
def ducks : ℕ := 37

-- Proof problem statement
theorem number_of_birds : geese + ducks = 95 :=
by
  -- The actual proof is to be provided
  sorry

end number_of_birds_l23_23242


namespace range_of_x_for_odd_monotonic_function_l23_23138

theorem range_of_x_for_odd_monotonic_function 
  (f : ℝ → ℝ)
  (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_monotonic : ∀ x y : ℝ, 0 ≤ x → x ≤ y → f x ≤ f y)
  (h_increasing_on_R : ∀ x y : ℝ, x ≤ y → f x ≤ f y) :
  ∀ x : ℝ, (0 < x) → ( (|f (Real.log x) - f (Real.log (1 / x))| / 2) < f 1 ) → (Real.exp (-1) < x ∧ x < Real.exp 1) := 
by
  sorry

end range_of_x_for_odd_monotonic_function_l23_23138


namespace greatest_n_and_k_l23_23102

-- (condition): k is a positive integer
def isPositive (k : Nat) : Prop :=
  k > 0

-- (condition): k < n
def lessThan (k n : Nat) : Prop :=
  k < n

/-- Let m = 3^n and k be a positive integer such that k < n.
     Determine the greatest value of n for which 3^n divides 25!,
     and the greatest value of k such that 3^k divides (25! - 3^n). -/
theorem greatest_n_and_k :
  ∃ (n k : Nat), (3^n ∣ Nat.factorial 25) ∧ (isPositive k) ∧ (lessThan k n) ∧ (3^k ∣ (Nat.factorial 25 - 3^n)) ∧ n = 10 ∧ k = 9 := by
    sorry

end greatest_n_and_k_l23_23102


namespace sum_first_110_terms_l23_23135

noncomputable def sum_arithmetic (a1 d : ℚ) (n : ℕ) : ℚ :=
  n * a1 + (n * (n - 1) / 2) * d

theorem sum_first_110_terms (a1 d : ℚ) (h1 : sum_arithmetic a1 d 10 = 100)
  (h2 : sum_arithmetic a1 d 100 = 10) : sum_arithmetic a1 d 110 = -110 := by
  sorry

end sum_first_110_terms_l23_23135


namespace triangle_inscribed_in_semicircle_l23_23206

variables {R : ℝ} (P Q R' : ℝ) (PR QR : ℝ)
variables (hR : 0 < R) (h_pq_diameter: P = -R ∧ Q = R)
variables (h_pr_square_qr_square : PR^2 + QR^2 = 4 * R^2)
variables (t := PR + QR)

theorem triangle_inscribed_in_semicircle (h_pos_pr : 0 < PR) (h_pos_qr : 0 < QR) : 
  t^2 ≤ 8 * R^2 :=
sorry

end triangle_inscribed_in_semicircle_l23_23206


namespace quadratic_inequality_solution_l23_23375

theorem quadratic_inequality_solution (d : ℝ) 
  (h1 : 0 < d) 
  (h2 : d < 16) : 
  ∃ x : ℝ, (x^2 - 8*x + d < 0) :=
  sorry

end quadratic_inequality_solution_l23_23375


namespace solve_equation_l23_23855

theorem solve_equation (x : ℝ) (hx : x ≠ 0) 
  (h : 1 / 4 + 8 / x = 13 / x + 1 / 8) : 
  x = 40 :=
sorry

end solve_equation_l23_23855


namespace pairs_symmetry_l23_23588

theorem pairs_symmetry (N : ℕ) (hN : N > 2) :
  ∃ f : {ab : ℕ × ℕ // ab.1 < ab.2 ∧ ab.2 ≤ N ∧ ab.2 / ab.1 > 2} ≃ 
           {ab : ℕ × ℕ // ab.1 < ab.2 ∧ ab.2 ≤ N ∧ ab.2 / ab.1 < 2}, 
  true :=
sorry

end pairs_symmetry_l23_23588


namespace women_in_department_l23_23311

theorem women_in_department : 
  ∀ (total_students men women : ℕ) (men_percentage women_percentage : ℝ),
  men_percentage = 0.70 →
  women_percentage = 0.30 →
  men = 420 →
  total_students = men / men_percentage →
  women = total_students * women_percentage →
  women = 180 :=
by
  intros total_students men women men_percentage women_percentage
  intros h1 h2 h3 h4 h5
  sorry

end women_in_department_l23_23311


namespace find_fraction_l23_23894

noncomputable def condition_eq : ℝ := 5
noncomputable def condition_gq : ℝ := 7

theorem find_fraction {FQ HQ : ℝ} (h : condition_eq * FQ = condition_gq * HQ) :
  FQ / HQ = 7 / 5 :=
by
  have eq_mul : condition_eq = 5 := by rfl
  have gq_mul : condition_gq = 7 := by rfl
  rw [eq_mul, gq_mul] at h
  have h': 5 * FQ = 7 * HQ := h
  field_simp [←h']
  sorry

end find_fraction_l23_23894


namespace seq_inequality_l23_23472

def seq (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ a 2 = 3 ∧ a 3 = 6 ∧ ∀ n, n > 3 → a n = 3 * a (n - 1) - a (n - 2) - 2 * a (n - 3)

theorem seq_inequality (a : ℕ → ℕ) (h : seq a) : ∀ n, n > 3 → a n > 3 * 2 ^ (n - 2) :=
  sorry

end seq_inequality_l23_23472


namespace prove_a_range_l23_23257

noncomputable def f (x : ℝ) : ℝ := 1 / (2 ^ x + 2)

theorem prove_a_range (a : ℝ) :
  (∀ x, 2 ≤ x ∧ x ≤ 3 → f x + f (a - 2 * x) ≤ 1 / 2) → 5 ≤ a :=
by
  sorry

end prove_a_range_l23_23257


namespace initial_number_correct_l23_23259

def initial_number_problem : Prop :=
  ∃ (x : ℝ), x + 3889 - 47.80600000000004 = 3854.002 ∧
            x = 12.808000000000158

theorem initial_number_correct : initial_number_problem :=
by
  -- proof goes here
  sorry

end initial_number_correct_l23_23259


namespace volume_ratio_of_spheres_l23_23238

theorem volume_ratio_of_spheres
  (r1 r2 r3 : ℝ)
  (A1 A2 A3 : ℝ)
  (V1 V2 V3 : ℝ)
  (hA : A1 / A2 = 1 / 4 ∧ A2 / A3 = 4 / 9)
  (hSurfaceArea : A1 = 4 * π * r1^2 ∧ A2 = 4 * π * r2^2 ∧ A3 = 4 * π * r3^2)
  (hVolume : V1 = (4 / 3) * π * r1^3 ∧ V2 = (4 / 3) * π * r2^3 ∧ V3 = (4 / 3) * π * r3^3) :
  V1 / V2 = 1 / 8 ∧ V2 / V3 = 8 / 27 := by
  sorry

end volume_ratio_of_spheres_l23_23238


namespace part1_part2_l23_23817

-- Define the function, assumptions, and the proof for the first part
theorem part1 (m : ℝ) (x : ℝ) :
  (∀ x > 1, -m * (0 * x + 1) * Real.log x + x - 0 ≥ 0) →
  m ≤ Real.exp 1 := sorry

-- Define the function, assumptions, and the proof for the second part
theorem part2 (x : ℝ) :
  (∀ x > 0, (x - 1) * (-(x + 1) * Real.log x + x - 1) ≤ 0) := sorry

end part1_part2_l23_23817


namespace probability_A_given_B_probability_A_or_B_l23_23570

-- Definitions of the given conditions
def PA : ℝ := 0.2
def PB : ℝ := 0.18
def PAB : ℝ := 0.12

-- Theorem to prove the probability that city A also experiences rain when city B is rainy
theorem probability_A_given_B : PA * PB = PAB -> PA = 2 / 3 := by
  sorry

-- Theorem to prove the probability that at least one of the two cities experiences rain
theorem probability_A_or_B (PA PB PAB : ℝ) : (PA + PB - PAB) = 0.26 := by
  sorry

end probability_A_given_B_probability_A_or_B_l23_23570


namespace candy_store_price_per_pound_fudge_l23_23680

theorem candy_store_price_per_pound_fudge 
  (fudge_pounds : ℕ)
  (truffles_dozen : ℕ)
  (truffles_price_each : ℝ)
  (pretzels_dozen : ℕ)
  (pretzels_price_each : ℝ)
  (total_revenue : ℝ) 
  (truffles_total : ℕ := truffles_dozen * 12)
  (pretzels_total : ℕ := pretzels_dozen * 12)
  (truffles_revenue : ℝ := truffles_total * truffles_price_each)
  (pretzels_revenue : ℝ := pretzels_total * pretzels_price_each)
  (fudge_revenue : ℝ := total_revenue - (truffles_revenue + pretzels_revenue))
  (fudge_price_per_pound : ℝ := fudge_revenue / fudge_pounds) :
  fudge_pounds = 20 →
  truffles_dozen = 5 →
  truffles_price_each = 1.50 →
  pretzels_dozen = 3 →
  pretzels_price_each = 2.00 →
  total_revenue = 212 →
  fudge_price_per_pound = 2.5 :=
by 
  sorry

end candy_store_price_per_pound_fudge_l23_23680


namespace slices_per_person_l23_23553

theorem slices_per_person
  (number_of_coworkers : ℕ)
  (number_of_pizzas : ℕ)
  (number_of_slices_per_pizza : ℕ)
  (total_slices : ℕ)
  (slices_per_person : ℕ) :
  number_of_coworkers = 12 →
  number_of_pizzas = 3 →
  number_of_slices_per_pizza = 8 →
  total_slices = number_of_pizzas * number_of_slices_per_pizza →
  slices_per_person = total_slices / number_of_coworkers →
  slices_per_person = 2 :=
by intros; sorry

end slices_per_person_l23_23553


namespace basketball_weight_l23_23859

variable (b c : ℝ)

theorem basketball_weight (h1 : 9 * b = 5 * c) (h2 : 3 * c = 75) : b = 125 / 9 :=
by
  sorry

end basketball_weight_l23_23859


namespace systematic_sampling_method_l23_23944

-- Define the problem conditions
def total_rows : Nat := 40
def seats_per_row : Nat := 25
def attendees_left (row : Nat) : Nat := if row < total_rows then 18 else 0

-- Problem statement to be proved: The method used is systematic sampling.
theorem systematic_sampling_method :
  (∀ r : Nat, r < total_rows → attendees_left r = 18) →
  (seats_per_row = 25) →
  (∃ k, k > 0 ∧ ∀ r, r < total_rows → attendees_left r = 18 + k * r) →
  True :=
by
  intro h1 h2 h3
  sorry

end systematic_sampling_method_l23_23944


namespace prove_sum_is_12_l23_23489

theorem prove_sum_is_12 (a b c : ℕ) (h : 28 * a + 30 * b + 31 * c = 365) : a + b + c = 12 := 
by 
  sorry

end prove_sum_is_12_l23_23489


namespace preimage_of_4_neg_2_eq_1_3_l23_23749

def mapping (x y : ℝ) : ℝ × ℝ := (x + y, x - y)

theorem preimage_of_4_neg_2_eq_1_3 : ∃ x y : ℝ, mapping x y = (4, -2) ∧ (x = 1) ∧ (y = 3) :=
by 
  sorry

end preimage_of_4_neg_2_eq_1_3_l23_23749


namespace grocery_cost_l23_23862

def rent : ℕ := 1100
def utilities : ℕ := 114
def roommate_payment : ℕ := 757

theorem grocery_cost (total_payment : ℕ) (half_rent_utilities : ℕ) (half_groceries : ℕ) (total_groceries : ℕ) :
  total_payment = 757 →
  half_rent_utilities = (rent + utilities) / 2 →
  half_groceries = total_payment - half_rent_utilities →
  total_groceries = half_groceries * 2 →
  total_groceries = 300 :=
by
  intros
  sorry

end grocery_cost_l23_23862


namespace orange_juice_percentage_l23_23591

theorem orange_juice_percentage 
  (V : ℝ) 
  (W : ℝ) 
  (G : ℝ)
  (hV : V = 300)
  (hW: W = 0.4 * V)
  (hG: G = 105) : 
  (V - W - G) / V * 100 = 25 := 
by 
  -- We will need to use sorry to skip the proof and focus just on the statement
  sorry

end orange_juice_percentage_l23_23591


namespace minimum_value_l23_23754

open Real

theorem minimum_value (x : ℝ) (h : 0 < x) : 
  ∃ y, (∀ z > 0, 3 * sqrt z + 2 / z ≥ y) ∧ y = 5 := by
  sorry

end minimum_value_l23_23754


namespace sum_of_roots_l23_23119

theorem sum_of_roots (x₁ x₂ : ℝ) (h1 : x₁^2 = 2 * x₁ + 1) (h2 : x₂^2 = 2 * x₂ + 1) :
  x₁ + x₂ = 2 :=
sorry

end sum_of_roots_l23_23119


namespace reciprocal_of_neg6_l23_23572

theorem reciprocal_of_neg6 : 1 / (-6 : ℝ) = -1 / 6 := 
sorry

end reciprocal_of_neg6_l23_23572


namespace investment_difference_l23_23831

noncomputable def compound_yearly (P : ℕ) (r : ℚ) (t : ℕ) : ℚ :=
  P * (1 + r)^t

noncomputable def compound_monthly (P : ℕ) (r : ℚ) (months : ℕ) : ℚ :=
  P * (1 + r)^(months)

theorem investment_difference :
  let P := 70000
  let r := 0.05
  let t := 3
  let monthly_r := r / 12
  let months := t * 12
  compound_monthly P monthly_r months - compound_yearly P r t = 263.71 :=
by
  sorry

end investment_difference_l23_23831


namespace second_smallest_relative_prime_210_l23_23858

theorem second_smallest_relative_prime_210 (x : ℕ) (h1 : x > 1) (h2 : Nat.gcd x 210 = 1) : x = 13 :=
sorry

end second_smallest_relative_prime_210_l23_23858


namespace junior_score_is_90_l23_23178

theorem junior_score_is_90 {n : ℕ} (hn : n > 0)
    (j : ℕ := n / 5) (s : ℕ := 4 * n / 5)
    (overall_avg : ℝ := 86)
    (senior_avg : ℝ := 85)
    (junior_score : ℝ)
    (h1 : 20 * j = n)
    (h2 : 80 * s = n * 4)
    (h3 : overall_avg * n = 86 * n)
    (h4 : senior_avg * s = 85 * s)
    (h5 : j * junior_score = overall_avg * n - senior_avg * s) :
    junior_score = 90 :=
by
  sorry

end junior_score_is_90_l23_23178


namespace drum_oil_capacity_l23_23479

theorem drum_oil_capacity (C : ℝ) (Y : ℝ) 
  (hX : DrumX_Oil = 0.5 * C) 
  (hY : DrumY_Cap = 2 * C) 
  (hY_filled : Y + 0.5 * C = 0.65 * (2 * C)) :
  Y = 0.8 * C :=
by
  sorry

end drum_oil_capacity_l23_23479


namespace cos_pi_over_3_plus_2alpha_l23_23034

theorem cos_pi_over_3_plus_2alpha (α : ℝ) (h : Real.sin (π / 3 - α) = 1 / 3) : 
  Real.cos (π / 3 + 2 * α) = -7 / 9 :=
by
  sorry

end cos_pi_over_3_plus_2alpha_l23_23034


namespace f_monotonically_decreasing_in_interval_l23_23062

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 + 3 * x^2 - 12 * x + 3

theorem f_monotonically_decreasing_in_interval :
  ∀ x y : ℝ, -2 < x ∧ x < 1 → -2 < y ∧ y < 1 → (y > x → f y < f x) :=
by
  sorry

end f_monotonically_decreasing_in_interval_l23_23062


namespace evaluate_fraction_l23_23685

theorem evaluate_fraction:
  (125 : ℝ)^(1/3) / (64 : ℝ)^(1/2) * (81 : ℝ)^(1/4) = 15 / 8 := 
by
  sorry

end evaluate_fraction_l23_23685


namespace max_value_inequality_l23_23916

theorem max_value_inequality (x y z : ℝ) (h : 9 * x^2 + 4 * y^2 + 25 * z^2 = 1) :
  3 * x + 4 * y + 6 * z ≤ Real.sqrt 53 := by
  sorry

end max_value_inequality_l23_23916


namespace max_handshakes_l23_23164

-- Definitions based on the given conditions
def num_people := 30
def handshake_formula (n : ℕ) := n * (n - 1) / 2

-- Formal statement of the problem
theorem max_handshakes : handshake_formula num_people = 435 :=
by
  -- Calculation here would be carried out in the proof, but not included in the statement itself.
  sorry

end max_handshakes_l23_23164


namespace abs_diff_of_solutions_eq_5_point_5_l23_23514

theorem abs_diff_of_solutions_eq_5_point_5 (x y : ℝ)
  (h1 : ⌊x⌋ + (y - ⌊y⌋) = 3.7)
  (h2 : (x - ⌊x⌋) + ⌊y⌋ = 8.2) :
  |x - y| = 5.5 :=
sorry

end abs_diff_of_solutions_eq_5_point_5_l23_23514


namespace solve_equation_l23_23420

theorem solve_equation (a b : ℤ) (ha : a ≥ 0) (hb : b ≥ 0) (h : a^2 = b * (b + 7)) : 
  (a = 0 ∧ b = 0) ∨ (a = 12 ∧ b = 9) :=
by sorry

end solve_equation_l23_23420


namespace max_path_length_is_32_l23_23296
-- Import the entire Mathlib library to use its definitions and lemmas

-- Definition of the problem setup
def number_of_edges_4x4_grid : Nat := 
  let total_squares := 4 * 4
  let total_edges_per_square := 4
  total_squares * total_edges_per_square

-- Definitions of internal edges shared by adjacent squares
def distinct_edges_4x4_grid : Nat := 
  let horizontal_lines := 5 * 4
  let vertical_lines := 5 * 4
  horizontal_lines + vertical_lines

-- Calculate the maximum length of the path
def max_length_of_path_4x4_grid : Nat := 
  let degree_3_nodes := 8
  distinct_edges_4x4_grid - degree_3_nodes

-- Main statement: Prove that the maximum length of the path is 32
theorem max_path_length_is_32 : max_length_of_path_4x4_grid = 32 := by
  -- Definitions for clarity and correctness
  have h1 : number_of_edges_4x4_grid = 64 := rfl
  have h2 : distinct_edges_4x4_grid = 40 := rfl
  have h3 : max_length_of_path_4x4_grid = 32 := rfl
  exact h3

end max_path_length_is_32_l23_23296


namespace length_of_stone_slab_l23_23999

theorem length_of_stone_slab 
  (num_slabs : ℕ) 
  (total_area : ℝ) 
  (h_num_slabs : num_slabs = 30) 
  (h_total_area : total_area = 50.7): 
  ∃ l : ℝ, l = 1.3 ∧ l * l * num_slabs = total_area := 
by 
  sorry

end length_of_stone_slab_l23_23999


namespace JerryAge_l23_23056

-- Given definitions
def MickeysAge : ℕ := 20
def AgeRelationship (M J : ℕ) : Prop := M = 2 * J + 10

-- Proof statement
theorem JerryAge : ∃ J : ℕ, AgeRelationship MickeysAge J ∧ J = 5 :=
by
  sorry

end JerryAge_l23_23056


namespace g_at_10_is_neg48_l23_23271

variable (g : ℝ → ℝ)

-- Given condition
axiom functional_eqn : ∀ x y : ℝ, g x + g (3 * x + y) + 7 * x * y = g (4 * x - y) + 3 * x^2 + 2

-- Mathematical proof statement
theorem g_at_10_is_neg48 : g 10 = -48 :=
  sorry

end g_at_10_is_neg48_l23_23271


namespace radius_of_unique_circle_l23_23075

noncomputable def circle_radius (z : ℂ) (h k : ℝ) : ℝ :=
  if z = 2 then 1/4 else 0  -- function that determines the circle

def unique_circle_radius : Prop :=
  let x1 := 2
  let y1 := 0
  
  let x2 := 3 / 2
  let y2 := Real.sqrt 11 / 2

  let h := 7 / 4 -- x-coordinate of the circle's center
  let k := 0    -- y-coordinate of the circle's center

  let r := 1 / 4 -- Radius of the circle
  
  -- equation of the circle passing through (x1, y1) and (x2, y2) should satisfy
  -- the radius of the resulting circle is r

  (x1 - h)^2 + y1^2 = r^2 ∧ (x2 - h)^2 + y2^2 = r^2

theorem radius_of_unique_circle :
  unique_circle_radius :=
sorry

end radius_of_unique_circle_l23_23075


namespace calculate_v_sum_l23_23132

def v (x : ℝ) : ℝ := x^3 - 3*x + 1

theorem calculate_v_sum :
  v (2) + v (-2) + v (1) + v (-1) = 4 :=
by
  sorry

end calculate_v_sum_l23_23132


namespace second_term_arithmetic_seq_l23_23393

variable (a d : ℝ)

theorem second_term_arithmetic_seq (h : a + (a + 2 * d) = 8) : a + d = 4 := by
  sorry

end second_term_arithmetic_seq_l23_23393


namespace unique_integral_solution_l23_23431

noncomputable def positiveInt (x : ℤ) : Prop := x > 0

theorem unique_integral_solution (m n : ℤ) (hm : positiveInt m) (hn : positiveInt n) (unique_sol : ∃! (x y : ℤ), x + y^2 = m ∧ x^2 + y = n) : 
  ∃ (k : ℕ), m - n = 2^k ∨ m - n = -2^k :=
sorry

end unique_integral_solution_l23_23431


namespace n_power_2020_plus_4_composite_l23_23018

theorem n_power_2020_plus_4_composite {n : ℕ} (h : n > 1) : ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n^2020 + 4 = a * b := 
by
  sorry

end n_power_2020_plus_4_composite_l23_23018


namespace Smarties_remainder_l23_23566

theorem Smarties_remainder (m : ℕ) (h : m % 11 = 5) : (4 * m) % 11 = 9 :=
by
  sorry

end Smarties_remainder_l23_23566


namespace solve_for_x_l23_23913

theorem solve_for_x (x : ℝ) : (3 : ℝ)^(4 * x^2 - 3 * x + 5) = (3 : ℝ)^(4 * x^2 + 9 * x - 6) ↔ x = 11 / 12 :=
by sorry

end solve_for_x_l23_23913


namespace c_10_eq_3_pow_89_l23_23481

section sequence
  open Nat

  -- Define the sequence c
  def c : ℕ → ℕ
  | 0     => 3  -- Note: Typically Lean sequences start from 0, not 1
  | 1     => 9
  | (n+2) => c n.succ * c n

  -- Define the auxiliary sequence d
  def d : ℕ → ℕ
  | 0     => 1  -- Note: Typically Lean sequences start from 0, not 1
  | 1     => 2
  | (n+2) => d n.succ + d n

  -- The theorem we need to prove
  theorem c_10_eq_3_pow_89 : c 9 = 3 ^ d 9 :=    -- Note: c_{10} in the original problem is c(9) in Lean
  sorry   -- Proof omitted
end sequence

end c_10_eq_3_pow_89_l23_23481


namespace frank_total_cans_l23_23369

def cansCollectedSaturday : List Nat := [4, 6, 5, 7, 8]
def cansCollectedSunday : List Nat := [6, 5, 9]
def cansCollectedMonday : List Nat := [8, 8]

def totalCansCollected (lst1 lst2 lst3 : List Nat) : Nat :=
  lst1.sum + lst2.sum + lst3.sum

theorem frank_total_cans :
  totalCansCollected cansCollectedSaturday cansCollectedSunday cansCollectedMonday = 66 :=
by
  sorry

end frank_total_cans_l23_23369


namespace f_value_at_5pi_over_6_l23_23170

noncomputable def f (x ω : ℝ) := 2 * Real.sin (ω * x + (Real.pi / 3))

theorem f_value_at_5pi_over_6
  (ω : ℝ) (ω_pos : ω > 0)
  (α β : ℝ)
  (h1 : f α ω = 2)
  (h2 : f β ω = 0)
  (h3 : Real.sqrt ((α - β)^2 + 4) = Real.sqrt (4 + (Real.pi^2 / 4))) :
  f (5 * Real.pi / 6) ω = -1 := 
sorry

end f_value_at_5pi_over_6_l23_23170


namespace sum_of_arithmetic_sequence_is_constant_l23_23901

def is_constant (n : ℕ) (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∃ c : ℝ, S n = c

theorem sum_of_arithmetic_sequence_is_constant
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (h1 : ∀ n, S n = n * (a 1 + a n) / 2)
  (h2 : a 2 + a 6 + a 10 = a 1 + d + a 1 + 5 * d + a 1 + 9 * d)
  (h3 : ∀ n, a n = a 1 + (n - 1) * d) :
  is_constant 11 a S :=
by
  sorry

end sum_of_arithmetic_sequence_is_constant_l23_23901


namespace find_ratio_l23_23730

variable {R : Type} [LinearOrderedField R]

def f (x a b : R) : R := x^3 + a*x^2 + b*x - a^2 - 7*a

def condition1 (a b : R) : Prop := f 1 a b = 10

def condition2 (a b : R) : Prop :=
  let f' := fun x => 3*x^2 + 2*a*x + b
  f' 1 = 0

theorem find_ratio (a b : R) (h1 : condition1 a b) (h2 : condition2 a b) :
  a / b = -2 / 3 :=
  sorry

end find_ratio_l23_23730


namespace min_value_of_translated_function_l23_23586

noncomputable def f (x : ℝ) (ϕ : ℝ) : ℝ := Real.sin (2 * x + ϕ)

theorem min_value_of_translated_function :
  ∀ (x : ℝ), 0 ≤ x ∧ x ≤ (Real.pi / 2) → ∀ (ϕ : ℝ), |ϕ| < (Real.pi / 2) →
  ∀ (k : ℤ), f (x + (Real.pi / 6)) (ϕ + (Real.pi / 3) + k * Real.pi) = f x ϕ →
  ∃ y : ℝ, y = - Real.sqrt 3 / 2 := sorry

end min_value_of_translated_function_l23_23586


namespace light_glow_duration_l23_23564

-- Define the conditions
def total_time_seconds : ℕ := 4969
def glow_times : ℚ := 292.29411764705884

-- Prove the equivalent statement
theorem light_glow_duration :
  (total_time_seconds / glow_times) = 17 := by
  sorry

end light_glow_duration_l23_23564


namespace quadratic_has_two_roots_l23_23623

theorem quadratic_has_two_roots 
  (a b c : ℝ) (h : b > a + c ∧ a + c > 0) : ∃ x₁ x₂ : ℝ, a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0 ∧ x₁ ≠ x₂ :=
by
  sorry

end quadratic_has_two_roots_l23_23623


namespace negation_proof_equivalence_l23_23963

theorem negation_proof_equivalence : 
  ¬(∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ (∃ x : ℝ, x^3 - x^2 + 1 > 0) :=
sorry

end negation_proof_equivalence_l23_23963


namespace temp_drop_of_8_deg_is_neg_8_l23_23142

theorem temp_drop_of_8_deg_is_neg_8 (rise_3_deg : ℤ) (h : rise_3_deg = 3) : ∀ drop_8_deg, drop_8_deg = -8 :=
by
  intros
  sorry

end temp_drop_of_8_deg_is_neg_8_l23_23142


namespace average_math_score_of_class_l23_23725

theorem average_math_score_of_class (n : ℕ) (jimin_score jung_score avg_others : ℕ) 
  (h1 : n = 40) 
  (h2 : jimin_score = 98) 
  (h3 : jung_score = 100) 
  (h4 : avg_others = 79) : 
  (38 * avg_others + jimin_score + jung_score) / n = 80 :=
by sorry

end average_math_score_of_class_l23_23725


namespace number_of_true_propositions_l23_23224

theorem number_of_true_propositions : 
  let original_p := ∀ (a : ℝ), a > -1 → a > -2
  let converse_p := ∀ (a : ℝ), a > -2 → a > -1
  let inverse_p := ∀ (a : ℝ), a ≤ -1 → a ≤ -2
  let contrapositive_p := ∀ (a : ℝ), a ≤ -2 → a ≤ -1
  (original_p ∧ contrapositive_p ∧ ¬converse_p ∧ ¬inverse_p) → (2 = 2) :=
by
  intros
  sorry

end number_of_true_propositions_l23_23224


namespace expenses_recorded_as_negative_l23_23351

/-*
  Given:
  1. The income of 5 yuan is recorded as +5 yuan.
  Prove:
  2. The expenses of 5 yuan are recorded as -5 yuan.
*-/

theorem expenses_recorded_as_negative (income_expenses_opposite_sign : ∀ (a : ℤ), -a = -a)
    (income_five_recorded_as_positive : (5 : ℤ) = 5) :
    (-5 : ℤ) = -5 :=
by sorry

end expenses_recorded_as_negative_l23_23351


namespace B_work_rate_l23_23378

theorem B_work_rate (B : ℕ) (A_rate C_rate : ℚ) 
  (A_work : A_rate = 1 / 6)
  (C_work : C_rate = 1 / 8 * (1 / 6 + 1 / B))
  (combined_work : 1 / 6 + 1 / B + C_rate = 1 / 3) : 
  B = 28 :=
by 
  sorry

end B_work_rate_l23_23378


namespace exists_fg_pairs_l23_23721

theorem exists_fg_pairs (a b : ℤ) :
  (∃ (f g : ℤ → ℤ), (∀ x : ℤ, f (g x) = x + a) ∧ (∀ x : ℤ, g (f x) = x + b)) ↔ (a = b ∨ a = -b) := 
sorry

end exists_fg_pairs_l23_23721


namespace sin_cos_identity_tan_identity_l23_23700

open Real

namespace Trigonometry

variable (α : ℝ)

-- Given conditions
def given_conditions := (sin α + cos α = (1/5)) ∧ (0 < α) ∧ (α < π)

-- Prove that sin(α) * cos(α) = -12/25
theorem sin_cos_identity (h : given_conditions α) : sin α * cos α = -12/25 := 
sorry

-- Prove that tan(α) = -4/3
theorem tan_identity (h : given_conditions α) : tan α = -4/3 :=
sorry

end Trigonometry

end sin_cos_identity_tan_identity_l23_23700


namespace Pooja_speed_3_l23_23914

variable (Roja_speed Pooja_speed : ℝ)
variable (t d : ℝ)

theorem Pooja_speed_3
  (h1 : Roja_speed = 6)
  (h2 : t = 4)
  (h3 : d = 36)
  (h4 : d = t * (Roja_speed + Pooja_speed)) :
  Pooja_speed = 3 :=
by
  sorry

end Pooja_speed_3_l23_23914


namespace sqrt_expression_simplification_l23_23207

theorem sqrt_expression_simplification :
  (Real.sqrt (1 / 16) - Real.sqrt (25 / 4) + |Real.sqrt (3) - 1| + Real.sqrt 3) = -13 / 4 + 2 * Real.sqrt 3 :=
by
  have h1 : Real.sqrt (1 / 16) = 1 / 4 := sorry
  have h2 : Real.sqrt (25 / 4) = 5 / 2 := sorry
  have h3 : |Real.sqrt 3 - 1| = Real.sqrt 3 - 1 := sorry
  linarith [h1, h2, h3]

end sqrt_expression_simplification_l23_23207


namespace problem_solution_l23_23969

noncomputable def find_a3_and_sum (a0 a1 a2 a3 a4 a5 : ℝ) : Prop :=
  (∀ x : ℝ, x^5 = a0 + a1 * (x + 2) + a2 * (x + 2)^2 + a3 * (x + 2)^3 + a4 * (x + 2)^4 + a5 * (x + 2)^5) →
  (a3 = 40 ∧ a0 + a1 + a2 + a4 + a5 = -41)

theorem problem_solution {a0 a1 a2 a3 a4 a5 : ℝ} :
  find_a3_and_sum a0 a1 a2 a3 a4 a5 :=
by
  intros h
  sorry

end problem_solution_l23_23969


namespace max_rank_awarded_l23_23033

theorem max_rank_awarded (num_participants rank_threshold total_possible_points : ℕ)
  (H1 : num_participants = 30)
  (H2 : rank_threshold = (30 * 29 / 2 : ℚ) * 0.60)
  (H3 : total_possible_points = (30 * 29 / 2)) :
  ∃ max_awarded : ℕ, max_awarded ≤ 23 :=
by {
  -- Proof omitted
  sorry
}

end max_rank_awarded_l23_23033


namespace no_real_solutions_l23_23324

theorem no_real_solutions : ∀ x : ℝ, ¬(3 * x - 2 * x + 8) ^ 2 = -|x| - 4 :=
by
  intro x
  sorry

end no_real_solutions_l23_23324


namespace maximize_box_volume_l23_23911

-- Define the volume function
def volume (x : ℝ) := (48 - 2 * x)^2 * x

-- Define the constraint on x
def constraint (x : ℝ) := 0 < x ∧ x < 24

-- The theorem stating the side length of the removed square that maximizes the volume
theorem maximize_box_volume : ∃ x : ℝ, constraint x ∧ (∀ y : ℝ, constraint y → volume y ≤ volume 8) :=
by
  sorry

end maximize_box_volume_l23_23911


namespace p_squared_plus_one_over_p_squared_plus_six_l23_23468

theorem p_squared_plus_one_over_p_squared_plus_six (p : ℝ) (h : p + 1/p = 10) : p^2 + 1/p^2 + 6 = 104 :=
by {
  sorry
}

end p_squared_plus_one_over_p_squared_plus_six_l23_23468


namespace pentagonal_number_formula_l23_23003

def pentagonal_number (n : ℕ) : ℕ :=
  (n * (3 * n + 1)) / 2

theorem pentagonal_number_formula (n : ℕ) :
  pentagonal_number n = (n * (3 * n + 1)) / 2 :=
by
  sorry

end pentagonal_number_formula_l23_23003


namespace max_sum_of_integer_pairs_l23_23128

theorem max_sum_of_integer_pairs (x y : ℤ) (h : (x-1)^2 + (y+2)^2 = 36) : 
  max (x + y) = 5 :=
sorry

end max_sum_of_integer_pairs_l23_23128


namespace find_positive_n_unique_solution_l23_23365

theorem find_positive_n_unique_solution (n : ℝ) :
  (∃ x : ℝ, 4 * x^2 + n * x + 4 = 0) → n = 8 :=
by
  intro h
  sorry

end find_positive_n_unique_solution_l23_23365


namespace no_such_base_exists_l23_23871

theorem no_such_base_exists : ¬ ∃ b : ℕ, (b^3 ≤ 630 ∧ 630 < b^4) ∧ (630 % b) % 2 = 1 := by
  sorry

end no_such_base_exists_l23_23871


namespace diamond_example_l23_23707

def diamond (a b : ℕ) : ℤ := 4 * a + 5 * b - a^2 * b

theorem diamond_example : diamond 3 4 = -4 :=
by
  rw [diamond]
  calc
    4 * 3 + 5 * 4 - 3^2 * 4 = 12 + 20 - 36 := by norm_num
                           _              = -4 := by norm_num

end diamond_example_l23_23707


namespace rachel_picked_apples_l23_23852

-- Define relevant variables based on problem conditions
variable (trees : ℕ) (apples_per_tree : ℕ) (remaining_apples : ℕ)
variable (total_apples_picked : ℕ)

-- Assume the given conditions
axiom num_trees : trees = 4
axiom apples_each_tree : apples_per_tree = 7
axiom apples_left : remaining_apples = 29

-- Define the number of apples picked
def total_apples_picked_def := trees * apples_per_tree

-- State the theorem to prove the total apples picked
theorem rachel_picked_apples :
  total_apples_picked_def trees apples_per_tree = 28 :=
by
  -- Proof omitted
  sorry

end rachel_picked_apples_l23_23852


namespace fraction_div_add_result_l23_23886

theorem fraction_div_add_result : 
  (2 / 3) / (4 / 5) + (1 / 2) = (4 / 3) := 
by 
  sorry

end fraction_div_add_result_l23_23886


namespace abs_difference_of_two_numbers_l23_23380

theorem abs_difference_of_two_numbers (x y : ℝ) (h1 : x + y = 36) (h2 : x * y = 105) :
  |x - y| = 6 * Real.sqrt 24.333 := sorry

end abs_difference_of_two_numbers_l23_23380


namespace arithmetic_sequence_5th_term_l23_23270

theorem arithmetic_sequence_5th_term :
  let a1 := 3
  let d := 4
  a1 + 4 * (5 - 1) = 19 :=
by
  sorry

end arithmetic_sequence_5th_term_l23_23270


namespace exists_not_perfect_square_l23_23590

theorem exists_not_perfect_square (a b c : ℤ) : ∃ (n : ℕ), n > 0 ∧ ¬ ∃ k : ℕ, n^3 + a * n^2 + b * n + c = k^2 :=
by
  sorry

end exists_not_perfect_square_l23_23590


namespace valid_for_expression_c_l23_23976

def expression_a_defined (x : ℝ) : Prop := x ≠ 2
def expression_b_defined (x : ℝ) : Prop := x ≠ 3
def expression_c_defined (x : ℝ) : Prop := x ≥ 2
def expression_d_defined (x : ℝ) : Prop := x ≥ 3

theorem valid_for_expression_c :
  (expression_a_defined 2 = false ∧ expression_a_defined 3 = true) ∧
  (expression_b_defined 2 = true ∧ expression_b_defined 3 = false) ∧
  (expression_c_defined 2 = true ∧ expression_c_defined 3 = true) ∧
  (expression_d_defined 2 = false ∧ expression_d_defined 3 = true) ∧
  (expression_c_defined 2 = true ∧ expression_c_defined 3 = true) := by
  sorry

end valid_for_expression_c_l23_23976


namespace cubics_inequality_l23_23389

theorem cubics_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hneq : a ≠ b) : a^3 + b^3 > a^2 * b + a * b^2 :=
sorry

end cubics_inequality_l23_23389


namespace hours_on_task2_l23_23427

theorem hours_on_task2
    (total_hours_per_week : ℕ) 
    (work_days_per_week : ℕ) 
    (hours_per_day_task1 : ℕ) 
    (hours_reduction_task1 : ℕ)
    (h_total_hours : total_hours_per_week = 40)
    (h_work_days : work_days_per_week = 5)
    (h_hours_task1 : hours_per_day_task1 = 5)
    (h_hours_reduction : hours_reduction_task1 = 5)
    : (total_hours_per_week / 2 / work_days_per_week) = 4 :=
by
  -- Skipping proof with sorry
  sorry

end hours_on_task2_l23_23427


namespace expectation_defective_items_variance_of_defective_items_l23_23364
-- Importing the necessary library from Mathlib

-- Define the conditions
def total_products : ℕ := 100
def defective_products : ℕ := 10
def selected_products : ℕ := 3

-- Define the expected number of defective items
def expected_defective_items : ℝ := 0.3

-- Define the variance of defective items
def variance_defective_items : ℝ := 0.2645

-- Lean statements to verify the conditions and results
theorem expectation_defective_items :
  let p := (defective_products: ℝ) / (total_products: ℝ)
  p * (selected_products: ℝ) = expected_defective_items := by sorry

theorem variance_of_defective_items :
  let p := (defective_products: ℝ) / (total_products: ℝ)
  let n := (selected_products: ℝ)
  n * p * (1 - p) * (total_products - n) / (total_products - 1) = variance_defective_items := by sorry

end expectation_defective_items_variance_of_defective_items_l23_23364


namespace combined_mpg_l23_23186

theorem combined_mpg
  (R_eff : ℝ) (T_eff : ℝ)
  (R_dist : ℝ) (T_dist : ℝ)
  (H_R_eff : R_eff = 35)
  (H_T_eff : T_eff = 15)
  (H_R_dist : R_dist = 420)
  (H_T_dist : T_dist = 300)
  : (R_dist + T_dist) / (R_dist / R_eff + T_dist / T_eff) = 22.5 := 
by
  rw [H_R_eff, H_T_eff, H_R_dist, H_T_dist]
  -- Proof steps would go here, but we'll use sorry to skip it.
  sorry

end combined_mpg_l23_23186


namespace triangle_third_side_length_l23_23112

theorem triangle_third_side_length {x : ℝ}
    (h1 : 3 > 0)
    (h2 : 7 > 0)
    (h3 : 3 + 7 > x)
    (h4 : x + 3 > 7)
    (h5 : x + 7 > 3) :
    4 < x ∧ x < 10 := by
  sorry

end triangle_third_side_length_l23_23112


namespace find_number_l23_23028

theorem find_number (S Q R N : ℕ) (hS : S = 555 + 445) (hQ : Q = 2 * (555 - 445)) (hR : R = 50) (h_eq : N = S * Q + R) :
  N = 220050 :=
by
  rw [hS, hQ, hR] at h_eq
  norm_num at h_eq
  exact h_eq

end find_number_l23_23028


namespace line_BC_eq_circumscribed_circle_eq_l23_23523

noncomputable def A : ℝ × ℝ := (3, 0)
noncomputable def B : ℝ × ℝ := (0, -1)
noncomputable def altitude_line (x y : ℝ) : Prop := x + y + 1 = 0
noncomputable def equation_line_BC (x y : ℝ) : Prop := 3 * x - y - 1 = 0
noncomputable def circumscribed_circle (x y : ℝ) : Prop := (x - 5 / 2)^2 + (y + 7 / 2)^2 = 50 / 4

theorem line_BC_eq :
  ∃ x y : ℝ, altitude_line x y →
             B = (x, y) →
             equation_line_BC x y :=
by sorry

theorem circumscribed_circle_eq :
  ∃ x y : ℝ, altitude_line x y →
             (x - 3)^2 + y^2 = (5 / 2)^2 →
             circumscribed_circle x y :=
by sorry

end line_BC_eq_circumscribed_circle_eq_l23_23523


namespace total_number_of_squares_l23_23268

variable (x y : ℕ) -- Variables for the number of 10 cm and 20 cm squares

theorem total_number_of_squares
  (h1 : 100 * x + 400 * y = 2500) -- Condition for area
  (h2 : 40 * x + 80 * y = 280)    -- Condition for cutting length
  : (x + y = 16) :=
sorry

end total_number_of_squares_l23_23268


namespace simplify_fractions_l23_23975

theorem simplify_fractions :
  (20 / 19) * (15 / 28) * (76 / 45) = 95 / 84 :=
by
  sorry

end simplify_fractions_l23_23975


namespace dave_apps_problem_l23_23810

theorem dave_apps_problem 
  (initial_apps : ℕ)
  (added_apps : ℕ)
  (final_apps : ℕ)
  (total_apps := initial_apps + added_apps)
  (deleted_apps := total_apps - final_apps) :
  initial_apps = 21 →
  added_apps = 89 →
  final_apps = 24 →
  (added_apps - deleted_apps = 3) :=
by
  intros
  sorry

end dave_apps_problem_l23_23810


namespace stone_breadth_l23_23086

theorem stone_breadth 
  (hall_length_m : ℕ) (hall_breadth_m : ℕ)
  (stone_length_dm : ℕ) (num_stones : ℕ)
  (hall_area_dm2 : ℕ) (stone_area_dm2 : ℕ) 
  (hall_length_dm hall_breadth_dm : ℕ) (b : ℕ) :
  hall_length_m = 36 → hall_breadth_m = 15 →
  stone_length_dm = 8 → num_stones = 1350 →
  hall_length_dm = hall_length_m * 10 → hall_breadth_dm = hall_breadth_m * 10 →
  hall_area_dm2 = hall_length_dm * hall_breadth_dm →
  stone_area_dm2 = stone_length_dm * b →
  hall_area_dm2 = num_stones * stone_area_dm2 →
  b = 5 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  -- Proof would go here
  sorry

end stone_breadth_l23_23086


namespace solve_system_l23_23522

theorem solve_system :
  ∃ x y : ℤ, (x - 3 * y = 7) ∧ (5 * x + 2 * y = 1) ∧ (x = 1) ∧ (y = -2) :=
by
  sorry

end solve_system_l23_23522


namespace stellar_hospital_multiple_births_l23_23783

/-- At Stellar Hospital, in a particular year, the multiple-birth statistics were such that sets of twins, triplets, and quintuplets accounted for 1200 of the babies born. 
There were twice as many sets of triplets as sets of quintuplets, and there were twice as many sets of twins as sets of triplets.
Determine how many of these 1200 babies were in sets of quintuplets. -/
theorem stellar_hospital_multiple_births 
    (a b c : ℕ)
    (h1 : b = 2 * c)
    (h2 : a = 2 * b)
    (h3 : 2 * a + 3 * b + 5 * c = 1200) :
    5 * c = 316 :=
by sorry

end stellar_hospital_multiple_births_l23_23783


namespace no_appearance_1234_or_3269_no_reappearance_1975_from_2nd_time_l23_23279

def seq (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 9
  | 2 => 7
  | 3 => 5
  | n + 4 => (seq n + seq (n + 1) + seq (n + 2) + seq (n + 3)) % 10

theorem no_appearance_1234_or_3269 : 
  ¬∃ n, seq n = 1 ∧ seq (n + 1) = 2 ∧ seq (n + 2) = 3 ∧ seq (n + 3) = 4 ∨
  seq n = 3 ∧ seq (n + 1) = 2 ∧ seq (n + 2) = 6 ∧ seq (n + 3) = 9 := 
sorry

theorem no_reappearance_1975_from_2nd_time : 
  ¬∃ n > 0, seq n = 1 ∧ seq (n + 1) = 9 ∧ seq (n + 2) = 7 ∧ seq (n + 3) = 5 :=
sorry

end no_appearance_1234_or_3269_no_reappearance_1975_from_2nd_time_l23_23279


namespace part1_solution_l23_23891

theorem part1_solution : ∀ n : ℕ, ∃ k : ℤ, 2^n + 3 = k^2 ↔ n = 0 :=
by sorry

end part1_solution_l23_23891


namespace initial_production_rate_l23_23321

variable (x : ℕ) (t : ℝ)

-- Conditions
def produces_initial (x : ℕ) (t : ℝ) : Prop := x * t = 60
def produces_subsequent : Prop := 60 * 1 = 60
def overall_average (t : ℝ) : Prop := 72 = 120 / (t + 1)

-- Goal: Prove the initial production rate
theorem initial_production_rate : 
  (∃ t : ℝ, produces_initial x t ∧ produces_subsequent ∧ overall_average t) → x = 90 := 
  by
    sorry

end initial_production_rate_l23_23321


namespace ratio_proof_l23_23232

noncomputable def total_capacity : ℝ := 10 -- million gallons
noncomputable def amount_end_month : ℝ := 6 -- million gallons
noncomputable def normal_level : ℝ := total_capacity - 5 -- million gallons

theorem ratio_proof (h1 : amount_end_month = 0.6 * total_capacity)
                    (h2 : normal_level = total_capacity - 5) :
  (amount_end_month / normal_level) = 1.2 :=
by sorry

end ratio_proof_l23_23232


namespace range_of_m_l23_23322

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ 2 ∧ x^2 + (m - 1) * x + 1 = 0) → m ≤ -1 :=
by
  sorry

end range_of_m_l23_23322


namespace volunteer_count_change_l23_23955

theorem volunteer_count_change :
  let x := 1
  let fall_increase := 1.09
  let winter_increase := 1.15
  let spring_decrease := 0.81
  let summer_increase := 1.12
  let summer_end_decrease := 0.95
  let final_ratio := x * fall_increase * winter_increase * spring_decrease * summer_increase * summer_end_decrease
  (final_ratio - x) / x * 100 = 19.13 :=
by
  sorry

end volunteer_count_change_l23_23955


namespace find_a_l23_23866

theorem find_a (a : ℝ) (h : ((2 * a + 16) + (3 * a - 8)) / 2 = 89) : a = 34 :=
sorry

end find_a_l23_23866


namespace prob_A_wins_4_consecutive_prob_fifth_game_needed_prob_C_ultimate_winner_l23_23290

-- Conditions for the game settings
def initial_conditions (a b c : ℕ) : Prop :=
  a = 0 ∧ b = 0 ∧ c = 0

-- Probability of a player winning any game
def win_probability : ℚ := 1 / 2 

-- Probability calculation for A winning four consecutive games
theorem prob_A_wins_4_consecutive :
  win_probability ^ 4 = 1 / 16 :=
by
  sorry

-- Probability calculation for needing a fifth game to be played
theorem prob_fifth_game_needed :
  1 - 4 * (win_probability ^ 4) = 3 / 4 :=
by
  sorry

-- Probability calculation for C being the ultimate winner
theorem prob_C_ultimate_winner :
  1 - 2 * (9 / 32) = 7 / 16 :=
by
  sorry

end prob_A_wins_4_consecutive_prob_fifth_game_needed_prob_C_ultimate_winner_l23_23290


namespace largest_product_of_three_l23_23329

-- Definitions of the numbers in the set
def numbers : List Int := [-5, 1, -3, 5, -2, 2]

-- Define a function to calculate the product of a list of three integers
def product_of_three (a b c : Int) : Int := a * b * c

-- Define a predicate to state that 75 is the largest product of any three numbers from the given list
theorem largest_product_of_three :
  ∃ (a b c : Int), a ∈ numbers ∧ b ∈ numbers ∧ c ∈ numbers ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ product_of_three a b c = 75 :=
sorry

end largest_product_of_three_l23_23329


namespace diagonal_AC_length_l23_23543

theorem diagonal_AC_length (AB BC CD DA : ℝ) (angle_ADC : ℝ) (h_AB : AB = 12) (h_BC : BC = 12) 
(h_CD : CD = 13) (h_DA : DA = 13) (h_angle_ADC : angle_ADC = 60) : 
  AC = 13 := 
sorry

end diagonal_AC_length_l23_23543


namespace number_of_new_terms_l23_23935

theorem number_of_new_terms (n : ℕ) (h : n > 1) :
  (2^(n+1) - 1) - (2^n - 1) + 1 = 2^n := by
sorry

end number_of_new_terms_l23_23935


namespace james_sushi_rolls_l23_23077

def fish_for_sushi : ℕ := 40
def total_fish : ℕ := 400
def bad_fish_percentage : ℕ := 20

theorem james_sushi_rolls :
  let good_fish := total_fish - (bad_fish_percentage * total_fish / 100)
  good_fish / fish_for_sushi = 8 :=
by
  sorry

end james_sushi_rolls_l23_23077


namespace soja_finished_fraction_l23_23335

def pages_finished (x pages_left total_pages : ℕ) : Prop :=
  x - pages_left = 100 ∧ x + pages_left = total_pages

noncomputable def fraction_finished (x total_pages : ℕ) : ℚ :=
  x / total_pages

theorem soja_finished_fraction (x : ℕ) (h1 : pages_finished x (x - 100) 300) :
  fraction_finished x 300 = 2 / 3 :=
by
  sorry

end soja_finished_fraction_l23_23335


namespace initial_catfish_count_l23_23346

theorem initial_catfish_count (goldfish : ℕ) (remaining_fish : ℕ) (disappeared_fish : ℕ) (catfish : ℕ) :
  goldfish = 7 → 
  remaining_fish = 15 → 
  disappeared_fish = 4 → 
  catfish + goldfish = 19 →
  catfish = 12 :=
by
  intros h1 h2 h3 h4
  sorry

end initial_catfish_count_l23_23346


namespace expression_positive_intervals_l23_23778
open Real

theorem expression_positive_intervals (x : ℝ) :
  ((x + 2) * (x - 3) > 0) ↔ (x < -2 ∨ x > 3) := by
  sorry

end expression_positive_intervals_l23_23778


namespace initial_shells_l23_23474

theorem initial_shells (x : ℕ) (h : x + 23 = 28) : x = 5 :=
by
  sorry

end initial_shells_l23_23474


namespace seunghwa_express_bus_distance_per_min_l23_23450

noncomputable def distance_per_min_on_express_bus (total_distance : ℝ) (total_time : ℝ) (time_on_general : ℝ) (gasoline_general : ℝ) (distance_per_gallon : ℝ) (gasoline_used : ℝ) : ℝ :=
  let distance_general := (gasoline_used * distance_per_gallon) / gasoline_general
  let distance_express := total_distance - distance_general
  let time_express := total_time - time_on_general
  (distance_express / time_express)

theorem seunghwa_express_bus_distance_per_min :
  distance_per_min_on_express_bus 120 110 (70) 6 (40.8) 14 = 0.62 :=
by
  sorry

end seunghwa_express_bus_distance_per_min_l23_23450


namespace problem_2535_l23_23603

theorem problem_2535 (a b : ℝ) (h1 : a + b = 5) (h2 : a * b = 1) :
  a + b + (a^3 / b^2) + (b^3 / a^2) = 2535 := sorry

end problem_2535_l23_23603


namespace time_to_be_100_miles_apart_l23_23848

noncomputable def distance_apart (x : ℝ) : ℝ :=
  Real.sqrt ((12 * x) ^ 2 + (16 * x) ^ 2)

theorem time_to_be_100_miles_apart : ∃ x : ℝ, distance_apart x = 100 ↔ x = 5 :=
by {
  sorry
}

end time_to_be_100_miles_apart_l23_23848


namespace investment_amount_l23_23893

theorem investment_amount (R T V : ℝ) (hT : T = 0.9 * R) (hV : V = 0.99 * R) (total_sum : R + T + V = 6936) : R = 2400 :=
by sorry

end investment_amount_l23_23893


namespace truck_tank_capacity_l23_23512

-- Definitions based on conditions
def truck_tank (T : ℝ) : Prop := true
def car_tank : Prop := true
def truck_half_full (T : ℝ) : Prop := true
def car_third_full : Prop := true
def add_fuel (T : ℝ) : Prop := T / 2 + 8 = 18

-- Theorem statement
theorem truck_tank_capacity (T : ℝ) (ht : truck_tank T) (hc : car_tank) 
  (ht_half : truck_half_full T) (hc_third : car_third_full) (hf_add : add_fuel T) : T = 20 :=
  sorry

end truck_tank_capacity_l23_23512


namespace task1_task2_task3_task4_l23_23665

-- Definitions of the given conditions
def cost_price : ℝ := 16
def selling_price_range (x : ℝ) : Prop := 16 ≤ x ∧ x ≤ 48
def init_selling_price : ℝ := 20
def init_sales_volume : ℝ := 360
def decreasing_sales_rate : ℝ := 10
def daily_sales_vol (x : ℝ) : ℝ := 360 - 10 * (x - 20)
def daily_total_profit (x : ℝ) (y : ℝ) : ℝ := y * (x - cost_price)

-- Proof task (1)
theorem task1 : daily_sales_vol 25 = 310 ∧ daily_total_profit 25 (daily_sales_vol 25) = 2790 := 
by 
    -- Your code here
    sorry

-- Proof task (2)
theorem task2 : ∀ x, daily_sales_vol x = -10 * x + 560 := 
by 
    -- Your code here
    sorry

-- Proof task (3)
theorem task3 : ∀ x, 
    W = (x - 16) * (daily_sales_vol x) 
    ∧ W = -10 * x ^ 2 + 720 * x - 8960 
    ∧ (∃ x, -10 * x ^ 2 + 720 * x - 8960 = 4000 ∧ selling_price_range x) := 
by 
    -- Your code here 
    sorry

-- Proof task (4)
theorem task4 : ∃ x, 
    -10 * (x - 36) ^ 2 + 4000 = 3000 
    ∧ selling_price_range x 
    ∧ (x = 26 ∨ x = 46) := 
by 
    -- Your code here 
    sorry

end task1_task2_task3_task4_l23_23665


namespace num_true_statements_is_two_l23_23022

def reciprocal (n : ℕ) : ℚ := 1 / n

theorem num_true_statements_is_two :
  let s1 := reciprocal 4 + reciprocal 8 = reciprocal 12
  let s2 := reciprocal 8 - reciprocal 5 = reciprocal 3
  let s3 := reciprocal 3 * reciprocal 9 = reciprocal 27
  let s4 := reciprocal 15 / reciprocal 3 = reciprocal 5
  (if s1 then 1 else 0) + (if s2 then 1 else 0) + (if s3 then 1 else 0) + (if s4 then 1 else 0) = 2 :=
by
  sorry

end num_true_statements_is_two_l23_23022


namespace max_movies_watched_l23_23179

-- Conditions given in the problem
def movie_duration : Nat := 90
def tuesday_minutes : Nat := 4 * 60 + 30
def tuesday_movies : Nat := tuesday_minutes / movie_duration
def wednesday_movies : Nat := 2 * tuesday_movies

-- Problem statement: Total movies watched in two days
theorem max_movies_watched : 
  tuesday_movies + wednesday_movies = 9 := 
by
  -- We add the placeholder for the proof here
  sorry

end max_movies_watched_l23_23179


namespace average_speed_l23_23152

theorem average_speed (D : ℝ) :
  let time_by_bus := D / 80
  let time_walking := D / 16
  let time_cycling := D / 120
  let total_time := time_by_bus + time_walking + time_cycling
  let total_distance := 2 * D
  total_distance / total_time = 24 := by
  sorry

end average_speed_l23_23152


namespace prime_gt_three_square_minus_one_divisible_by_twentyfour_l23_23423

theorem prime_gt_three_square_minus_one_divisible_by_twentyfour (p : ℕ) (hp_prime : Nat.Prime p) (hp_gt_three : p > 3) : 24 ∣ (p^2 - 1) :=
sorry

end prime_gt_three_square_minus_one_divisible_by_twentyfour_l23_23423


namespace bicycle_wheels_l23_23724

theorem bicycle_wheels :
  ∀ (b : ℕ),
  let bicycles := 24
  let tricycles := 14
  let wheels_per_tricycle := 3
  let total_wheels := 90
  ((bicycles * b) + (tricycles * wheels_per_tricycle) = total_wheels) → b = 2 :=
by {
  sorry
}

end bicycle_wheels_l23_23724


namespace remainder_31_31_plus_31_mod_32_l23_23000

theorem remainder_31_31_plus_31_mod_32 : (31 ^ 31 + 31) % 32 = 30 := 
by sorry

end remainder_31_31_plus_31_mod_32_l23_23000


namespace log_expression_eq_l23_23904

theorem log_expression_eq (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (Real.log x^2 / Real.log (y^4)) * 
  (Real.log (y^3) / Real.log (x^6)) * 
  (Real.log (x^4) / Real.log (y^3)) * 
  (Real.log (y^4) / Real.log (x^2)) * 
  (Real.log (x^6) / Real.log y) = 
  16 * Real.log x / Real.log y := 
sorry

end log_expression_eq_l23_23904


namespace red_ball_probability_l23_23307

-- Define the conditions
def total_balls : ℕ := 10
def yellow_balls : ℕ := 1
def green_balls : ℕ := 3
def red_balls : ℕ := total_balls - yellow_balls - green_balls

-- Define the probability function
def probability_of_red_ball (total red : ℕ) : ℚ := red / total

-- The main theorem statement to prove
theorem red_ball_probability :
  probability_of_red_ball total_balls red_balls = 3 / 5 :=
by
  sorry

end red_ball_probability_l23_23307


namespace factorize_expression_l23_23413

theorem factorize_expression (x y : ℝ) : x^2 + x * y + x = x * (x + y + 1) := 
by
  sorry

end factorize_expression_l23_23413


namespace total_cost_paid_l23_23498

-- Definition of the given conditions
def number_of_DVDs : ℕ := 4
def cost_per_DVD : ℝ := 1.2

-- The theorem to be proven
theorem total_cost_paid : number_of_DVDs * cost_per_DVD = 4.8 := by
  sorry

end total_cost_paid_l23_23498


namespace breakfast_calories_l23_23055

variable (B : ℝ) 

def lunch_calories := 1.25 * B
def dinner_calories := 2.5 * B
def shakes_calories := 900
def total_calories := 3275

theorem breakfast_calories:
  (B + lunch_calories B + dinner_calories B + shakes_calories = total_calories) → B = 500 :=
by
  sorry

end breakfast_calories_l23_23055


namespace find_w_l23_23773

noncomputable def line_p(t : ℝ) : (ℝ × ℝ) := (2 + 3 * t, 5 + 2 * t)
noncomputable def line_q(u : ℝ) : (ℝ × ℝ) := (-3 + 3 * u, 7 + 2 * u)

def vector_DC(t u : ℝ) : ℝ × ℝ := ((2 + 3 * t) - (-3 + 3 * u), (5 + 2 * t) - (7 + 2 * u))

def w_condition (w1 w2 : ℝ) : Prop := w1 + w2 = 3

theorem find_w (t u : ℝ) :
  ∃ w1 w2 : ℝ, 
    w_condition w1 w2 ∧ 
    (∃ k : ℝ, 
      sorry -- This is a placeholder for the projection calculation
    )
    :=
  sorry -- This is a placeholder for the final proof

end find_w_l23_23773


namespace min_value_a1_l23_23801

noncomputable def is_geometric_sequence (seq : ℕ → ℕ) : Prop :=
  ∃ r : ℕ, ∀ n : ℕ, seq (n + 1) = r * seq n

theorem min_value_a1 (a1 a2 : ℕ) (seq : ℕ → ℕ)
  (h1 : is_geometric_sequence seq)
  (h2 : ∀ n : ℕ, seq n > 0)
  (h3 : seq 20 + seq 21 = 20^21) :
  ∃ a b : ℕ, a1 = 2^a * 5^b ∧ a + b = 24 :=
sorry

end min_value_a1_l23_23801


namespace proof_m_cd_value_l23_23959

theorem proof_m_cd_value (a b c d m : ℝ) 
  (H1 : a + b = 0) (H2 : c * d = 1) (H3 : |m| = 3) : 
  m + c * d - (a + b) / (m ^ 2) = 4 ∨ m + c * d - (a + b) / (m ^ 2) = -2 :=
by
  sorry

end proof_m_cd_value_l23_23959


namespace gcd_power_sub_one_l23_23966

theorem gcd_power_sub_one (a b : ℕ) (h1 : b = a + 30) : 
  Nat.gcd (2^a - 1) (2^b - 1) = 2^30 - 1 := 
by 
  sorry

end gcd_power_sub_one_l23_23966


namespace right_triangle_side_sums_l23_23957

theorem right_triangle_side_sums (a b c : ℕ) (h1 : a + b = c + 6) (h2 : a^2 + b^2 = c^2) :
  (a = 7 ∧ b = 24 ∧ c = 25) ∨ (a = 8 ∧ b = 15 ∧ c = 17) ∨ (a = 9 ∧ b = 12 ∧ c = 15) :=
sorry

end right_triangle_side_sums_l23_23957


namespace value_of_m_l23_23037

theorem value_of_m (z1 z2 m : ℝ) (h1 : (Polynomial.X ^ 2 + 5 * Polynomial.X + Polynomial.C m).eval z1 = 0)
  (h2 : (Polynomial.X ^ 2 + 5 * Polynomial.X + Polynomial.C m).eval z2 = 0)
  (h3 : |z1 - z2| = 3) : m = 4 ∨ m = 17 / 2 := sorry

end value_of_m_l23_23037


namespace circle_center_distance_l23_23030

theorem circle_center_distance (R : ℝ) : 
  ∃ (d : ℝ), 
  (∀ (θ : ℝ), θ = 30 → 
  ∀ (r : ℝ), r = 2.5 →
  ∀ (center_on_other_side : ℝ), center_on_other_side = R + R →
  d = 5) :=
by 
  use 5
  intros θ θ_eq r r_eq center_on_other_side center_eq
  sorry

end circle_center_distance_l23_23030


namespace polar_line_eq_l23_23466

theorem polar_line_eq (ρ θ : ℝ) : (ρ * Real.cos θ = 1) ↔ (ρ = Real.cos θ ∨ ρ = Real.sin θ ∨ 1 / Real.cos θ = ρ) := by
  sorry

end polar_line_eq_l23_23466


namespace balloon_descent_rate_l23_23757

theorem balloon_descent_rate (D : ℕ) 
    (rate_of_ascent : ℕ := 50) 
    (time_chain_pulled_1 : ℕ := 15) 
    (time_chain_pulled_2 : ℕ := 15) 
    (time_chain_released_1 : ℕ := 10) 
    (highest_elevation : ℕ := 1400) :
    (time_chain_pulled_1 + time_chain_pulled_2) * rate_of_ascent - time_chain_released_1 * D = highest_elevation 
    → D = 10 := 
by 
  intro h
  sorry

end balloon_descent_rate_l23_23757


namespace length_AE_l23_23565

-- The given conditions:
def isosceles_triangle (A B C : Type*) (AB BC : ℝ) (h : AB = BC) : Prop := true

def angles_and_lengths (A D C E : Type*) (angle_ADC angle_AEC AD CE DC : ℝ) 
  (h_angles : angle_ADC = 60 ∧ angle_AEC = 60)
  (h_lengths : AD = 13 ∧ CE = 13 ∧ DC = 9) : Prop := true

variables {A B C D E : Type*} (AB BC AD CE DC : ℝ)
  (h_isosceles_triangle : isosceles_triangle A B C AB BC (by sorry))
  (h_angles_and_lengths : angles_and_lengths A D C E 60 60 AD CE DC 
    (by split; norm_num) (by repeat {split}; norm_num))

-- The proof problem:
theorem length_AE : ∃ AE : ℝ, AE = 4 :=
  by sorry

end length_AE_l23_23565


namespace min_value_inequality_l23_23727

theorem min_value_inequality (a b c : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 9)
  : (a^2 + b^2 + c^2)/(a + b + c) + (b^2 + c^2)/(b + c) + (c^2 + a^2)/(c + a) + (a^2 + b^2)/(a + b) ≥ 12 :=
by
  sorry

end min_value_inequality_l23_23727


namespace range_of_m_l23_23828

-- Definitions of propositions p and q
def p (m : ℝ) : Prop := 
  (2 * m - 3)^2 - 4 > 0

def q (m : ℝ) : Prop := 
  2 * m > 3

-- Theorem statement
theorem range_of_m (m : ℝ) : ¬ (p m ∧ q m) ∧ (p m ∨ q m) ↔ (m < 1 / 2 ∨ 3 / 2 < m ∧ m ≤ 5 / 2) :=
  sorry

end range_of_m_l23_23828


namespace boys_meeting_problem_l23_23990

theorem boys_meeting_problem (d : ℝ) (t : ℝ)
  (speed1 speed2 : ℝ)
  (h1 : speed1 = 6) 
  (h2 : speed2 = 8) 
  (h3 : t > 0)
  (h4 : ∀ n : ℤ, n * (speed1 + speed2) * t ≠ d) : 
  0 = 0 :=
by 
  sorry

end boys_meeting_problem_l23_23990


namespace condition1_condition2_l23_23849

-- Define the point P
def P (m : ℝ) : ℝ × ℝ := (m + 1, 2 * m - 4)

-- Define the point A
def A : ℝ × ℝ := (-5, 2)

-- Condition 1: P lies on the x-axis
def on_x_axis (p : ℝ × ℝ) : Prop := p.2 = 0

-- Condition 2: AP is parallel to the y-axis
def parallel_y_axis (a p : ℝ × ℝ) : Prop := a.1 = p.1

-- Prove the conditions
theorem condition1 (m : ℝ) (h : on_x_axis (P m)) : P m = (3, 0) :=
by
  sorry

theorem condition2 (m : ℝ) (h : parallel_y_axis A (P m)) : P m = (-5, -16) :=
by
  sorry

end condition1_condition2_l23_23849


namespace composition_of_homotheties_l23_23123

-- Define points A1 and A2 and the coefficients k1 and k2
variables (A1 A2 : ℂ) (k1 k2 : ℂ)

-- Definition of homothety
def homothety (A : ℂ) (k : ℂ) (z : ℂ) : ℂ := k * (z - A) + A

-- Translation vector in case 1
noncomputable def translation_vector (A1 A2 : ℂ) (k1 k2 : ℂ) : ℂ :=
  if k1 * k2 = 1 then (1 - k1) * A1 + (k1 - 1) * A2 else 0 

-- Center A in case 2
noncomputable def center (A1 A2 : ℂ) (k1 k2 : ℂ) : ℂ :=
  if k1 * k2 ≠ 1 then (k2 * (1 - k1) * A1 + (1 - k2) * A2) / (k1 * k2 - 1) else 0

-- The final composition of two homotheties
noncomputable def composition (A1 A2 : ℂ) (k1 k2 : ℂ) (z : ℂ) : ℂ :=
  if k1 * k2 = 1 then z + translation_vector A1 A2 k1 k2
  else homothety (center A1 A2 k1 k2) (k1 * k2) z

-- The theorem to prove
theorem composition_of_homotheties 
  (A1 A2 : ℂ) (k1 k2 : ℂ) : ∀ z : ℂ,
  composition A1 A2 k1 k2 z = if k1 * k2 = 1 then z + translation_vector A1 A2 k1 k2
                              else homothety (center A1 A2 k1 k2) (k1 * k2) z := 
by sorry

end composition_of_homotheties_l23_23123


namespace exponent_arithmetic_proof_l23_23669

theorem exponent_arithmetic_proof :
  ( (6 ^ 6 / 6 ^ 5) ^ 3 * 8 ^ 3 / 4 ^ 3) = 1728 := by
  sorry

end exponent_arithmetic_proof_l23_23669


namespace elderly_people_not_set_l23_23908

def is_well_defined (S : Set α) : Prop := Nonempty S

def all_positive_numbers : Set ℝ := {x : ℝ | 0 < x}
def real_numbers_non_zero : Set ℝ := {x : ℝ | x ≠ 0}
def four_great_inventions : Set String := {"compass", "gunpowder", "papermaking", "printing"}

def elderly_people_description : String := "elderly people"

theorem elderly_people_not_set :
  ¬ (∃ S : Set α, elderly_people_description = "elderly people" ∧ is_well_defined S) :=
sorry

end elderly_people_not_set_l23_23908


namespace solve_quadratic_1_solve_quadratic_2_solve_quadratic_3_solve_quadratic_4_l23_23895

-- Define the first theorem
theorem solve_quadratic_1 (x : ℝ) : x^2 - 2*x = 0 ↔ x = 0 ∨ x = 2 := 
by {
  -- We assume this proof is provided
  sorry
}

-- Define the second theorem
theorem solve_quadratic_2 (x : ℝ) : 25*x^2 - 36 = 0 ↔ x = 6/5 ∨ x = -6/5 := 
by {
  -- We assume this proof is provided
  sorry
}

-- Define the third theorem
theorem solve_quadratic_3 (x : ℝ) : x^2 + 10*x + 21 = 0 ↔ x = -3 ∨ x = -7 := 
by {
  -- We assume this proof is provided
  sorry
}

-- Define the fourth theorem
theorem solve_quadratic_4 (x : ℝ) : (x-3)^2 + 2*x*(x-3) = 0 ↔ x = 3 ∨ x = 1 := 
by {
  -- We assume this proof is provided
  sorry
}

end solve_quadratic_1_solve_quadratic_2_solve_quadratic_3_solve_quadratic_4_l23_23895


namespace prob_same_color_is_correct_l23_23384

-- Define the sides of one die
def blue_sides := 6
def yellow_sides := 8
def green_sides := 10
def purple_sides := 6
def total_sides := 30

-- Define the probability each die shows a specific color
def prob_blue := blue_sides / total_sides
def prob_yellow := yellow_sides / total_sides
def prob_green := green_sides / total_sides
def prob_purple := purple_sides / total_sides

-- The probability that both dice show the same color
def prob_same_color :=
  (prob_blue * prob_blue) + 
  (prob_yellow * prob_yellow) + 
  (prob_green * prob_green) + 
  (prob_purple * prob_purple)

-- We should prove that the computed probability is equal to the given answer
theorem prob_same_color_is_correct :
  prob_same_color = 59 / 225 := 
sorry

end prob_same_color_is_correct_l23_23384


namespace split_terms_addition_l23_23140

theorem split_terms_addition : 
  (-2017 - (2/3)) + (2016 + (3/4)) + (-2015 - (5/6)) + (16 + (1/2)) = -2000 - (1/4) :=
by
  sorry

end split_terms_addition_l23_23140


namespace canteen_distance_l23_23334

-- Given definitions
def G_to_road : ℝ := 450
def G_to_B : ℝ := 700

-- Proof statement
theorem canteen_distance :
  ∃ x : ℝ, (x ≠ 0) ∧ 
           (G_to_road^2 + (x - G_to_road)^2 = x^2) ∧ 
           (x = 538) := 
by {
  sorry
}

end canteen_distance_l23_23334


namespace solve_for_a_l23_23162

theorem solve_for_a (a : ℝ) (h : 2 * a + (1 - 4 * a) = 0) : a = 1 / 2 :=
sorry

end solve_for_a_l23_23162


namespace range_of_e_l23_23345

theorem range_of_e (a b c d e : ℝ)
  (h1 : a + b + c + d + e = 8)
  (h2 : a^2 + b^2 + c^2 + d^2 + e^2 = 16) :
  0 ≤ e ∧ e ≤ 16 / 5 :=
by
  sorry

end range_of_e_l23_23345


namespace cuboid_volume_is_correct_l23_23129

-- Definition of cuboid edges and volume calculation
def cuboid_volume (a b c : ℕ) : ℕ := a * b * c

-- Given conditions
def edge1 : ℕ := 2
def edge2 : ℕ := 5
def edge3 : ℕ := 3

-- Theorem statement
theorem cuboid_volume_is_correct : cuboid_volume edge1 edge2 edge3 = 30 := 
by sorry

end cuboid_volume_is_correct_l23_23129


namespace find_m_n_l23_23155

theorem find_m_n (m n : ℕ) (h : (1/5 : ℝ)^m * (1/4 : ℝ)^n = 1 / (10 : ℝ)^4) : m = 4 ∧ n = 2 :=
sorry

end find_m_n_l23_23155


namespace smallest_part_of_80_divided_by_proportion_l23_23390

theorem smallest_part_of_80_divided_by_proportion (x : ℕ) (h1 : 1 * x + 3 * x + 5 * x + 7 * x = 80) : x = 5 :=
sorry

end smallest_part_of_80_divided_by_proportion_l23_23390


namespace billy_sisters_count_l23_23074

theorem billy_sisters_count 
  (S B : ℕ) -- S is the number of sisters, B is the number of brothers
  (h1 : B = 2 * S) -- Billy has twice as many brothers as sisters
  (h2 : 2 * (B + S) = 12) -- Billy gives 2 sodas to each sibling to give out the 12 pack
  : S = 2 := 
  by sorry

end billy_sisters_count_l23_23074


namespace combined_length_of_trains_l23_23106

theorem combined_length_of_trains
  (speed_A_kmph : ℕ) (speed_B_kmph : ℕ)
  (platform_length : ℕ) (time_A_sec : ℕ) (time_B_sec : ℕ)
  (h_speed_A : speed_A_kmph = 72) (h_speed_B : speed_B_kmph = 90)
  (h_platform_length : platform_length = 300)
  (h_time_A : time_A_sec = 30) (h_time_B : time_B_sec = 24) :
  let speed_A_ms := speed_A_kmph * 5 / 18
  let speed_B_ms := speed_B_kmph * 5 / 18
  let distance_A := speed_A_ms * time_A_sec
  let distance_B := speed_B_ms * time_B_sec
  let length_A := distance_A - platform_length
  let length_B := distance_B - platform_length
  length_A + length_B = 600 :=
by
  sorry

end combined_length_of_trains_l23_23106


namespace imaginary_unit_sum_l23_23949

-- Define that i is the imaginary unit, which satisfies \(i^2 = -1\)
def is_imaginary_unit (i : ℂ) := i^2 = -1

-- The theorem to be proven: i + i^2 + i^3 + i^4 = 0 given that i is the imaginary unit
theorem imaginary_unit_sum (i : ℂ) (h : is_imaginary_unit i) : 
  i + i^2 + i^3 + i^4 = 0 := 
sorry

end imaginary_unit_sum_l23_23949


namespace find_polynomial_P_l23_23861

noncomputable def P (x : ℝ) : ℝ :=
  - (5/8) * x^3 + (5/2) * x^2 + (1/8) * x - 1

theorem find_polynomial_P 
  (α β γ : ℝ)
  (h_roots : ∀ {x: ℝ}, x^3 - 4 * x^2 + 6 * x + 8 = 0 → x = α ∨ x = β ∨ x = γ)
  (h1 : P α = β + γ)
  (h2 : P β = α + γ)
  (h3 : P γ = α + β)
  (h4 : P (α + β + γ) = -20) :
  P x = - (5/8) * x^3 + (5/2) * x^2 + (1/8) * x - 1 :=
by sorry

end find_polynomial_P_l23_23861


namespace probability_ace_then_king_l23_23706

-- Definitions of the conditions
def custom_deck := 65
def extra_spades := 14
def total_aces := 4
def total_kings := 4

-- Probability calculations
noncomputable def P_ace_first : ℚ := total_aces / custom_deck
noncomputable def P_king_second : ℚ := total_kings / (custom_deck - 1)

theorem probability_ace_then_king :
  (P_ace_first * P_king_second) = 1 / 260 :=
by
  sorry

end probability_ace_then_king_l23_23706


namespace largest_k_no_perpendicular_lines_l23_23638

theorem largest_k_no_perpendicular_lines (n : ℕ) (h : 0 < n) :
  (∃ k, ∀ (l : Fin n → ℝ) (f : Fin n), (∀ i j, i ≠ j → l i ≠ -1 / (l j)) → k = Nat.ceil (n / 2)) :=
sorry

end largest_k_no_perpendicular_lines_l23_23638


namespace f_of_3_l23_23013

def f (x : ℚ) : ℚ := (x + 3) / (x - 6)

theorem f_of_3 : f 3 = -2 := by
  sorry

end f_of_3_l23_23013


namespace factorize_expression_l23_23654

theorem factorize_expression (a : ℝ) : 3 * a^2 + 6 * a + 3 = 3 * (a + 1)^2 := 
by sorry

end factorize_expression_l23_23654


namespace g_value_at_50_l23_23492

noncomputable def g : ℝ → ℝ :=
sorry

theorem g_value_at_50 (g : ℝ → ℝ)
  (h : ∀ x y : ℝ, 0 < x → 0 < y → x * g y - y ^ 2 * g x = g (x / y)) :
  g 50 = 0 :=
by
  sorry

end g_value_at_50_l23_23492


namespace quadratic_h_value_l23_23540

theorem quadratic_h_value (p q r h : ℝ) (hq : p*x^2 + q*x + r = 5*(x - 3)^2 + 15):
  let new_quadratic := 4* (p*x^2 + q*x + r)
  let m := 20
  let k := 60
  new_quadratic = m * (x - h) ^ 2 + k → h = 3 := by
  sorry

end quadratic_h_value_l23_23540


namespace number_decomposition_l23_23686

theorem number_decomposition : 10101 = 10000 + 100 + 1 :=
by
  sorry

end number_decomposition_l23_23686


namespace mixed_fraction_product_l23_23900

theorem mixed_fraction_product (X Y : ℕ) (hX : X ≠ 0) (hY : Y ≠ 0) :
  (5 + (1 / X : ℚ)) * (Y + (1 / 2 : ℚ)) = 43 ↔ X = 17 ∧ Y = 8 := 
by 
  sorry

end mixed_fraction_product_l23_23900


namespace simplify_expression_l23_23702

theorem simplify_expression (y : ℝ) :
  4 * y - 3 * y^3 + 6 - (1 - 4 * y + 3 * y^3) = -6 * y^3 + 8 * y + 5 :=
by
  sorry

end simplify_expression_l23_23702


namespace cosine_of_difference_l23_23273

theorem cosine_of_difference (α : ℝ) (h : Real.sin (α + π / 6) = 1 / 3) :
  Real.cos (α - π / 3) = 1 / 3 :=
by
  sorry

end cosine_of_difference_l23_23273


namespace mul_inv_mod_35_l23_23567

theorem mul_inv_mod_35 : (8 * 22) % 35 = 1 := 
  sorry

end mul_inv_mod_35_l23_23567


namespace ratio_correct_l23_23563

theorem ratio_correct : 
    (2^17 * 3^19) / (6^18) = 3 / 2 :=
by sorry

end ratio_correct_l23_23563


namespace NutsInThirdBox_l23_23602

variable (x y z : ℝ)

theorem NutsInThirdBox (h1 : x = (y + z) - 6) (h2 : y = (x + z) - 10) : z = 16 := 
sorry

end NutsInThirdBox_l23_23602


namespace min_b_factors_l23_23151

theorem min_b_factors (x r s b : ℕ) (h : r * s = 1998) (fact : (x + r) * (x + s) = x^2 + b * x + 1998) : b = 91 :=
sorry

end min_b_factors_l23_23151


namespace max_distance_with_optimal_swapping_l23_23256

-- Define the conditions
def front_tire_lifetime : ℕ := 24000
def rear_tire_lifetime : ℕ := 36000

-- Prove that the maximum distance the car can travel given optimal tire swapping is 48,000 km
theorem max_distance_with_optimal_swapping : 
    ∃ x : ℕ, x < 24000 ∧ x < 36000 ∧ (x + min (24000 - x) (36000 - x) = 48000) :=
by {
  sorry
}

end max_distance_with_optimal_swapping_l23_23256


namespace solve_system_b_zero_solve_system_b_nonzero_solve_second_system_l23_23091

section B_zero

variables {x y z b : ℝ}

-- Given conditions for the first system when b = 0
variables (hb_zero : b = 0)
variables (h1 : x + y + z = 0)
variables (h2 : x^2 + y^2 - z^2 = 0)
variables (h3 : 3 * x * y * z - x^3 - y^3 - z^3 = b^3)

theorem solve_system_b_zero :
  ∃ x y z, 3 * x * y * z - x^3 - y^3 - z^3 = b^3 :=
by { sorry }

end B_zero

section B_nonzero

variables {x y z b : ℝ}

-- Given conditions for the first system when b ≠ 0
variables (hb_nonzero : b ≠ 0)
variables (h1 : x + y + z = 2 * b)
variables (h2 : x^2 + y^2 - z^2 = b^2)
variables (h3 : 3 * x * y * z - x^3 - y^3 - z^3 = b^3)

theorem solve_system_b_nonzero :
  ∃ x y z, 3 * x * y * z - x^3 - y^3 - z^3 = b^3 :=
by { sorry }

end B_nonzero

section Second_System

variables {x y z a : ℝ}

-- Given conditions for the second system
variables (h4 : x^2 + y^2 - 2 * z^2 = 2 * a^2)
variables (h5 : x + y + 2 * z = 4 * (a^2 + 1))
variables (h6 : z^2 - x * y = a^2)

theorem solve_second_system :
  ∃ x y z, z^2 - x * y = a^2 :=
by { sorry }

end Second_System

end solve_system_b_zero_solve_system_b_nonzero_solve_second_system_l23_23091


namespace multiple_of_a_l23_23734

theorem multiple_of_a's_share (A B : ℝ) (x : ℝ) (h₁ : A + B + 260 = 585) (h₂ : x * A = 780) (h₃ : 6 * B = 780) : x = 4 :=
sorry

end multiple_of_a_l23_23734


namespace find_bases_l23_23561

theorem find_bases {F1 F2 : ℝ} (R1 R2 : ℕ) 
                   (hR1 : R1 = 9)
                   (hR2 : R2 = 6)
                   (hF1_R1 : F1 = 0.484848 * 9^2 / (9^2 - 1))
                   (hF2_R1 : F2 = 0.848484 * 9^2 / (9^2 - 1))
                   (hF1_R2 : F1 = 0.353535 * 6^2 / (6^2 - 1))
                   (hF2_R2 : F2 = 0.535353 * 6^2 / (6^2 - 1))
                   : R1 + R2 = 15 :=
by
  sorry

end find_bases_l23_23561


namespace solve_for_Theta_l23_23678

-- Define the two-digit number representation condition
def fourTheta (Θ : ℕ) : ℕ := 40 + Θ

-- Main theorem statement
theorem solve_for_Theta (Θ : ℕ) (h1 : 198 / Θ = fourTheta Θ + Θ) (h2 : 0 < Θ ∧ Θ < 10) : Θ = 4 :=
by
  sorry

end solve_for_Theta_l23_23678


namespace oliver_baths_per_week_l23_23482

-- Define all the conditions given in the problem
def bucket_capacity : ℕ := 120
def num_buckets_to_fill_tub : ℕ := 14
def num_buckets_removed : ℕ := 3
def weekly_water_usage : ℕ := 9240

-- Calculate total water to fill bathtub, water removed, water used per bath, and baths per week
def total_tub_capacity : ℕ := num_buckets_to_fill_tub * bucket_capacity
def water_removed : ℕ := num_buckets_removed * bucket_capacity
def water_per_bath : ℕ := total_tub_capacity - water_removed
def baths_per_week : ℕ := weekly_water_usage / water_per_bath

theorem oliver_baths_per_week : baths_per_week = 7 := by
  sorry

end oliver_baths_per_week_l23_23482


namespace two_digit_even_multiple_of_7_l23_23252

def all_digits_product_square (n : ℕ) : Prop :=
  let d1 := n / 10
  let d2 := n % 10
  (d1 * d2) > 0 ∧ ∃ k, d1 * d2 = k * k

theorem two_digit_even_multiple_of_7 (n : ℕ) :
  10 ≤ n ∧ n < 100 ∧ n % 2 = 0 ∧ n % 7 = 0 ∧ all_digits_product_square n ↔ n = 14 ∨ n = 28 ∨ n = 70 :=
by sorry

end two_digit_even_multiple_of_7_l23_23252


namespace apples_difference_l23_23651

-- Definitions based on conditions
def JackiesApples : Nat := 10
def AdamsApples : Nat := 8

-- Statement
theorem apples_difference : JackiesApples - AdamsApples = 2 := by
  sorry

end apples_difference_l23_23651


namespace inequality_of_ab_l23_23524

theorem inequality_of_ab (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a ≠ b) :
  Real.sqrt (a * b) < (a - b) / (Real.log a - Real.log b) ∧ 
  (a - b) / (Real.log a - Real.log b) < (a + b) / 2 :=
by
  sorry

end inequality_of_ab_l23_23524


namespace telephone_number_fraction_calculation_l23_23326

theorem telephone_number_fraction_calculation :
  let valid_phone_numbers := 7 * 10^6
  let special_phone_numbers := 10^5
  (special_phone_numbers / valid_phone_numbers : ℚ) = 1 / 70 :=
by
  sorry

end telephone_number_fraction_calculation_l23_23326


namespace total_cost_correct_l23_23089

-- Define the cost of each category of items
def cost_of_book : ℕ := 16
def cost_of_binders : ℕ := 3 * 2
def cost_of_notebooks : ℕ := 6 * 1

-- Define the total cost calculation
def total_cost : ℕ := cost_of_book + cost_of_binders + cost_of_notebooks

-- Prove that the total cost of Léa's purchases is 28
theorem total_cost_correct : total_cost = 28 :=
by {
  -- This is where the proof would go, but it's omitted for now.
  sorry
}

end total_cost_correct_l23_23089


namespace area_of_AFCH_l23_23131

-- Define the lengths of the sides of the rectangles
def AB : ℝ := 9
def BC : ℝ := 5
def EF : ℝ := 3
def FG : ℝ := 10

-- Define the problem statement
theorem area_of_AFCH :
  let intersection_area := min BC FG * min EF AB
  let total_area := AB * FG
  let outer_ring_area := total_area - intersection_area
  intersection_area + outer_ring_area / 2 = 52.5 :=
by
  -- Use the values of AB, BC, EF, and FG to compute
  sorry

end area_of_AFCH_l23_23131


namespace angle_B_pi_div_3_triangle_perimeter_l23_23216

-- Problem 1: Prove that B = π / 3 given the condition.
theorem angle_B_pi_div_3 (A B C : ℝ) (hTriangle : A + B + C = Real.pi) 
  (hCos : Real.cos B = Real.cos ((A + C) / 2)) : 
  B = Real.pi / 3 :=
sorry

-- Problem 2: Prove the perimeter given the conditions.
theorem triangle_perimeter (a b c : ℝ) (m : ℝ) 
  (altitude : ℝ) 
  (hSides : 8 * a = 3 * c) 
  (hAltitude : altitude = 12 * Real.sqrt 3 / 7) 
  (hAngleB : ∃ B, B = Real.pi / 3) :
  a + b + c = 18 := 
sorry

end angle_B_pi_div_3_triangle_perimeter_l23_23216


namespace planes_parallel_l23_23385

theorem planes_parallel (n1 n2 : ℝ × ℝ × ℝ)
  (h1 : n1 = (2, -1, 0)) 
  (h2 : n2 = (-4, 2, 0)) :
  ∃ k : ℝ, n2 = k • n1 := by
  -- Proof is beyond the scope of this exercise.
  sorry

end planes_parallel_l23_23385


namespace edward_initial_lives_l23_23993

def initialLives (lives_lost lives_left : Nat) : Nat :=
  lives_lost + lives_left

theorem edward_initial_lives (lost left : Nat) (H_lost : lost = 8) (H_left : left = 7) :
  initialLives lost left = 15 :=
by
  sorry

end edward_initial_lives_l23_23993


namespace expression_value_l23_23780

theorem expression_value (a b : ℕ) (h₁ : a = 37) (h₂ : b = 12) : 
  (a + b)^2 - (a^2 + b^2) = 888 := by
  sorry

end expression_value_l23_23780


namespace total_cost_of_constructing_the_path_l23_23635

open Real

-- Define the conditions
def length_field : ℝ := 75
def width_field : ℝ := 55
def path_width : ℝ := 2.8
def area_path_given : ℝ := 1518.72
def cost_per_sq_m : ℝ := 2

-- Define the total cost to be proven
def total_cost : ℝ := 3037.44

-- The statement to be proven
theorem total_cost_of_constructing_the_path :
  let outer_length := length_field + 2 * path_width
  let outer_width := width_field + 2 * path_width
  let total_area_incl_path := outer_length * outer_width
  let area_field := length_field * width_field
  let computed_area_path := total_area_incl_path - area_field
  let given_cost := area_path_given * cost_per_sq_m
  total_cost = given_cost := by
  sorry

end total_cost_of_constructing_the_path_l23_23635


namespace smallest_number_four_solutions_sum_four_squares_l23_23159

def is_sum_of_four_squares (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), n = a^2 + b^2 + c^2 + d^2

theorem smallest_number_four_solutions_sum_four_squares :
  ∃ n : ℕ,
    is_sum_of_four_squares n ∧
    (∃ (a1 b1 c1 d1 a2 b2 c2 d2 a3 b3 c3 d3 a4 b4 c4 d4 : ℕ),
      n = a1^2 + b1^2 + c1^2 + d1^2 ∧
      n = a2^2 + b2^2 + c2^2 + d2^2 ∧
      n = a3^2 + b3^2 + c3^2 + d3^2 ∧
      n = a4^2 + b4^2 + c4^2 + d4^2 ∧
      (a1, b1, c1, d1) ≠ (a2, b2, c2, d2) ∧
      (a1, b1, c1, d1) ≠ (a3, b3, c3, d3) ∧
      (a1, b1, c1, d1) ≠ (a4, b4, c4, d4) ∧
      (a2, b2, c2, d2) ≠ (a3, b3, c3, d3) ∧
      (a2, b2, c2, d2) ≠ (a4, b4, c4, d4) ∧
      (a3, b3, c3, d3) ≠ (a4, b4, c4, d4)) ∧
    (∀ m : ℕ,
      m < 635318657 →
      ¬ (∃ (a5 b5 c5 d5 a6 b6 c6 d6 a7 b7 c7 d7 a8 b8 c8 d8 : ℕ),
        m = a5^2 + b5^2 + c5^2 + d5^2 ∧
        m = a6^2 + b6^2 + c6^2 + d6^2 ∧
        m = a7^2 + b7^2 + c7^2 + d7^2 ∧
        m = a8^2 + b8^2 + c8^2 + d8^2 ∧
        (a5, b5, c5, d5) ≠ (a6, b6, c6, d6) ∧
        (a5, b5, c5, d5) ≠ (a7, b7, c7, d7) ∧
        (a5, b5, c5, d5) ≠ (a8, b8, c8, d8) ∧
        (a6, b6, c6, d6) ≠ (a7, b7, c7, d7) ∧
        (a6, b6, c6, d6) ≠ (a8, b8, c8, d8) ∧
        (a7, b7, c7, d7) ≠ (a8, b8, c8, d8))) :=
  sorry

end smallest_number_four_solutions_sum_four_squares_l23_23159


namespace distance_point_line_l23_23950

theorem distance_point_line (m : ℝ) : 
  abs (m + 1) = 2 ↔ (m = 1 ∨ m = -3) := by
  sorry

end distance_point_line_l23_23950


namespace region_relation_l23_23927

theorem region_relation (A B C : ℝ)
  (a b c : ℝ) (h1 : a = 15) (h2 : b = 36) (h3 : c = 39)
  (h_triangle : a^2 + b^2 = c^2)
  (h_right_triangle : true) -- Since the triangle is already confirmed as right-angle
  (h_A : A = (π * (c / 2)^2 / 2 - 270) / 2)
  (h_B : B = (π * (c / 2)^2 / 2 - 270) / 2)
  (h_C : C = π * (c / 2)^2 / 2) :
  A + B + 270 = C :=
by
  sorry

end region_relation_l23_23927


namespace identify_quadratic_equation_l23_23812

-- Definitions of the equations
def eqA : Prop := ∀ x : ℝ, x^2 + 1/x^2 = 4
def eqB : Prop := ∀ (a b x : ℝ), a*x^2 + b*x - 3 = 0
def eqC : Prop := ∀ x : ℝ, (x - 1)*(x + 2) = 1
def eqD : Prop := ∀ (x y : ℝ), 3*x^2 - 2*x*y - 5*y^2 = 0

-- Definition that identifies whether a given equation is a quadratic equation in one variable
def isQuadraticInOneVariable (eq : Prop) : Prop := 
  ∃ (a b c : ℝ) (a0 : a ≠ 0), ∀ x : ℝ, eq = (a * x^2 + b * x + c = 0)

theorem identify_quadratic_equation :
  isQuadraticInOneVariable eqC :=
by
  sorry

end identify_quadratic_equation_l23_23812


namespace george_change_sum_l23_23928

theorem george_change_sum :
  ∃ n m : ℕ,
    0 ≤ n ∧ n < 19 ∧
    0 ≤ m ∧ m < 10 ∧
    (7 + 5 * n) = (4 + 10 * m) ∧
    (7 + 5 * 14) + (4 + 10 * 7) = 144 :=
by
  -- We declare the problem stating that there exist natural numbers n and m within
  -- the given ranges such that the sums of valid change amounts add up to 144 cents.
  sorry

end george_change_sum_l23_23928


namespace students_just_passed_l23_23231

theorem students_just_passed (total_students : ℕ) (first_division : ℕ) (second_division : ℕ) (just_passed : ℕ)
  (h1 : total_students = 300)
  (h2 : first_division = 26 * total_students / 100)
  (h3 : second_division = 54 * total_students / 100)
  (h4 : just_passed = total_students - (first_division + second_division)) :
  just_passed = 60 :=
sorry

end students_just_passed_l23_23231


namespace find_b_perpendicular_lines_l23_23559

variable (b : ℝ)

theorem find_b_perpendicular_lines :
  (2 * b + (-4) * 3 + 7 * (-1) = 0) → b = 19 / 2 := 
by
  intro h
  sorry

end find_b_perpendicular_lines_l23_23559


namespace area_of_quadrilateral_ABCD_l23_23676

theorem area_of_quadrilateral_ABCD
  (BD : ℝ) (hA : ℝ) (hC : ℝ) (angle_ABD : ℝ) :
  BD = 28 ∧ hA = 8 ∧ hC = 2 ∧ angle_ABD = 60 →
  ∃ (area_ABCD : ℝ), area_ABCD = 140 :=
by
  sorry

end area_of_quadrilateral_ABCD_l23_23676


namespace derivative_f_cos2x_l23_23495

variable {f : ℝ → ℝ} {x : ℝ}

theorem derivative_f_cos2x :
  f (Real.cos (2 * x)) = 1 - 2 * (Real.sin x) ^ 2 →
  deriv f x = -2 * Real.sin (2 * x) :=
by sorry

end derivative_f_cos2x_l23_23495


namespace campers_in_two_classes_l23_23315

-- Definitions of the sets and conditions
variable (S A R : Finset ℕ)
variable (n : ℕ)
variable (x : ℕ)

-- Given conditions
axiom hyp1 : S.card = 20
axiom hyp2 : A.card = 20
axiom hyp3 : R.card = 20
axiom hyp4 : (S ∩ A ∩ R).card = 4
axiom hyp5 : (S \ (A ∪ R)).card + (A \ (S ∪ R)).card + (R \ (S ∪ A)).card = 24

-- The hypothesis that n = |S ∪ A ∪ R|
axiom hyp6 : n = (S ∪ A ∪ R).card

-- Statement to be proven in Lean
theorem campers_in_two_classes : x = 12 :=
by
  sorry

end campers_in_two_classes_l23_23315


namespace ticket_cost_at_30_years_l23_23010

noncomputable def initial_cost : ℝ := 1000000
noncomputable def halving_period_years : ℕ := 10
noncomputable def halving_factor : ℝ := 0.5

def cost_after_n_years (initial_cost : ℝ) (halving_factor : ℝ) (years : ℕ) (period : ℕ) : ℝ :=
  initial_cost * halving_factor ^ (years / period)

theorem ticket_cost_at_30_years (initial_cost halving_factor : ℝ) (years period: ℕ) 
  (h_initial_cost : initial_cost = 1000000)
  (h_halving_factor : halving_factor = 0.5)
  (h_years : years = 30)
  (h_period : period = halving_period_years) : 
  cost_after_n_years initial_cost halving_factor years period = 125000 :=
by 
  sorry

end ticket_cost_at_30_years_l23_23010


namespace find_antecedent_l23_23587

-- Condition: The ratio is 4:6, simplified to 2:3
def ratio (a b : ℕ) : Prop := (a / gcd a b) = 2 ∧ (b / gcd a b) = 3

-- Condition: The consequent is 30
def consequent (y : ℕ) : Prop := y = 30

-- The problem is to find the antecedent
def antecedent (x : ℕ) (y : ℕ) : Prop := ratio x y

-- The theorem to be proved
theorem find_antecedent:
  ∃ x : ℕ, consequent 30 → antecedent x 30 ∧ x = 20 :=
by
  sorry

end find_antecedent_l23_23587


namespace trigonometric_inequality_C_trigonometric_inequality_D_l23_23029

theorem trigonometric_inequality_C (x : Real) : Real.cos (3*Real.pi/5) > Real.cos (-4*Real.pi/5) :=
by
  sorry

theorem trigonometric_inequality_D (y : Real) : Real.sin (Real.pi/10) < Real.cos (Real.pi/10) :=
by
  sorry

end trigonometric_inequality_C_trigonometric_inequality_D_l23_23029


namespace min_value_of_a_plus_2b_l23_23343

theorem min_value_of_a_plus_2b (a b : ℝ) (h_pos : a > 0 ∧ b > 0) (h_eq : 1 / a + 2 / b = 4) : a + 2 * b = 9 / 4 :=
by
  sorry

end min_value_of_a_plus_2b_l23_23343


namespace range_of_a_plus_b_l23_23374

theorem range_of_a_plus_b (a b : ℝ) (h : |a| + |b| + |a - 1| + |b - 1| ≤ 2) : 
  0 ≤ a + b ∧ a + b ≤ 2 :=
sorry

end range_of_a_plus_b_l23_23374


namespace pete_and_ray_spent_200_cents_l23_23756

-- Define the basic units
def cents_in_a_dollar := 100
def value_of_a_nickel := 5
def value_of_a_dime := 10

-- Define the initial amounts and spending
def pete_initial_amount := 250  -- cents
def ray_initial_amount := 250  -- cents
def pete_spent_nickels := 4 * value_of_a_nickel
def ray_remaining_dimes := 7 * value_of_a_dime

-- Calculate total amounts spent
def total_spent_pete := pete_spent_nickels
def total_spent_ray := ray_initial_amount - ray_remaining_dimes
def total_spent := total_spent_pete + total_spent_ray

-- The proof problem statement
theorem pete_and_ray_spent_200_cents : total_spent = 200 := by {
 sorry
}

end pete_and_ray_spent_200_cents_l23_23756


namespace find_larger_number_l23_23208

theorem find_larger_number (L S : ℕ) (h1 : L - S = 1365) (h2 : L = 7 * S + 15) : L = 1590 := 
sorry

end find_larger_number_l23_23208


namespace curve_transformation_l23_23205

-- Define the scaling transformation
def scaling_transform (x y : ℝ) : ℝ × ℝ :=
  (5 * x, 3 * y)

-- Define the transformed curve
def transformed_curve (x' y' : ℝ) : Prop :=
  2 * x' ^ 2 + 8 * y' ^ 2 = 1

-- Define the curve C's equation after scaling
def curve_C (x y : ℝ) : Prop :=
  50 * x ^ 2 + 72 * y ^ 2 = 1

-- Statement of the proof problem
theorem curve_transformation (x y : ℝ) (h : transformed_curve (5 * x) (3 * y)) : curve_C x y :=
by {
  -- The actual proof would be filled in here
  sorry
}

end curve_transformation_l23_23205


namespace time_to_save_for_downpayment_l23_23867

def annual_salary : ℝ := 120000
def savings_percentage : ℝ := 0.15
def house_cost : ℝ := 550000
def downpayment_percentage : ℝ := 0.25

def annual_savings : ℝ := savings_percentage * annual_salary
def downpayment_needed : ℝ := downpayment_percentage * house_cost

theorem time_to_save_for_downpayment :
  (downpayment_needed / annual_savings) = 7.64 :=
by
  -- Proof to be provided
  sorry

end time_to_save_for_downpayment_l23_23867


namespace simplified_t_l23_23809

noncomputable def cuberoot (x : ℝ) : ℝ := x^(1/3)

theorem simplified_t (t : ℝ) (h : t = 1 / (3 - cuberoot 3)) : t = (3 + cuberoot 3) / 6 :=
by
  sorry

end simplified_t_l23_23809


namespace find_parallelogram_height_l23_23705

def parallelogram_height (base area : ℕ) : ℕ := area / base

theorem find_parallelogram_height :
  parallelogram_height 32 448 = 14 :=
by {
  sorry
}

end find_parallelogram_height_l23_23705


namespace num_license_plates_l23_23507

-- Let's state the number of letters in the alphabet, vowels, consonants, and digits.
def num_letters : ℕ := 26
def num_vowels : ℕ := 5  -- A, E, I, O, U and Y is not a vowel
def num_consonants : ℕ := 21  -- Remaining letters including Y
def num_digits : ℕ := 10  -- 0 through 9

-- Prove the number of five-character license plates
theorem num_license_plates : 
  (num_consonants * num_consonants * num_vowels * num_vowels * num_digits) = 110250 :=
  by 
  sorry

end num_license_plates_l23_23507


namespace geometric_sequence_problem_l23_23903

section 
variables (a : ℕ → ℝ) (r : ℝ) 

-- Condition: {a_n} is a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) := ∀ n : ℕ, a (n + 1) = a n * r

-- Condition: a_4 + a_6 = 8
axiom a4_a6_sum : a 4 + a 6 = 8

-- Mathematical equivalent proof problem
theorem geometric_sequence_problem (h : is_geometric_sequence a r) : 
  a 1 * a 7 + 2 * a 3 * a 7 + a 3 * a 9 = 64 :=
sorry

end

end geometric_sequence_problem_l23_23903


namespace average_weight_a_b_l23_23160

theorem average_weight_a_b (A B C : ℝ) 
    (h1 : (A + B + C) / 3 = 45) 
    (h2 : (B + C) / 2 = 44) 
    (h3 : B = 33) : 
    (A + B) / 2 = 40 := 
by 
  sorry

end average_weight_a_b_l23_23160


namespace smallest_integer_l23_23658

/-- The smallest integer m such that m > 1 and m has a remainder of 1 when divided by any of 5, 7, and 3 is 106. -/
theorem smallest_integer (m : ℕ) : m > 1 ∧ m % 5 = 1 ∧ m % 7 = 1 ∧ m % 3 = 1 ↔ m = 106 :=
by
    sorry

end smallest_integer_l23_23658


namespace length_of_second_offset_l23_23790

theorem length_of_second_offset 
  (d : ℝ) (offset1 : ℝ) (area : ℝ) (offset2 : ℝ) 
  (h1 : d = 40)
  (h2 : offset1 = 9)
  (h3 : area = 300) :
  offset2 = 6 :=
by
  sorry

end length_of_second_offset_l23_23790


namespace sale_price_is_correct_l23_23040

def original_price : ℝ := 100
def percentage_decrease : ℝ := 0.30
def sale_price : ℝ := original_price * (1 - percentage_decrease)

theorem sale_price_is_correct : sale_price = 70 := by
  sorry

end sale_price_is_correct_l23_23040


namespace interest_years_eq_three_l23_23425

theorem interest_years_eq_three :
  ∀ (x y : ℝ),
    (x + 1720 = 2795) →
    (x * (3 / 100) * 8 = 1720 * (5 / 100) * y) →
    y = 3 :=
by
  intros x y hsum heq
  sorry

end interest_years_eq_three_l23_23425


namespace relationship_of_y_l23_23172

theorem relationship_of_y (k y1 y2 y3 : ℝ)
  (hk : k < 0)
  (hy1 : y1 = k / -2)
  (hy2 : y2 = k / 1)
  (hy3 : y3 = k / 2) :
  y2 < y3 ∧ y3 < y1 := by
  -- Proof omitted
  sorry

end relationship_of_y_l23_23172


namespace quadratic_solution_l23_23412

def quadratic_rewrite (x b c : ℝ) : ℝ := (x + b) * (x + b) + c

theorem quadratic_solution (b c : ℝ)
  (h1 : ∀ x, x^2 + 2100 * x + 4200 = quadratic_rewrite x b c)
  (h2 : c = -b^2 + 4200) :
  c / b = -1034 :=
by
  sorry

end quadratic_solution_l23_23412


namespace average_score_l23_23632

theorem average_score 
  (total_students : ℕ)
  (assigned_day_students_pct : ℝ)
  (makeup_day_students_pct : ℝ)
  (assigned_day_avg_score : ℝ)
  (makeup_day_avg_score : ℝ)
  (h1 : total_students = 100)
  (h2 : assigned_day_students_pct = 0.70)
  (h3 : makeup_day_students_pct = 0.30)
  (h4 : assigned_day_avg_score = 0.60)
  (h5 : makeup_day_avg_score = 0.90) :
  (0.70 * 100 * 0.60 + 0.30 * 100 * 0.90) / 100 = 0.69 := 
sorry


end average_score_l23_23632


namespace clowns_per_mobile_28_l23_23097

def clowns_in_each_mobile (total_clowns num_mobiles : Nat) (h : total_clowns = 140 ∧ num_mobiles = 5) : Nat :=
  total_clowns / num_mobiles

theorem clowns_per_mobile_28 (total_clowns num_mobiles : Nat) (h : total_clowns = 140 ∧ num_mobiles = 5) :
  clowns_in_each_mobile total_clowns num_mobiles h = 28 :=
by
  sorry

end clowns_per_mobile_28_l23_23097


namespace andrew_age_l23_23399

variables (a g : ℝ)

theorem andrew_age (h1 : g = 15 * a) (h2 : g - a = 60) : a = 30 / 7 :=
by sorry

end andrew_age_l23_23399


namespace rancher_lasso_probability_l23_23141

theorem rancher_lasso_probability : 
  let p_success := 1 / 2
  let p_failure := 1 - p_success
  (1 - p_failure ^ 3) = (7 / 8) := by
  sorry

end rancher_lasso_probability_l23_23141


namespace total_orchids_l23_23069

-- Conditions
def current_orchids : ℕ := 2
def additional_orchids : ℕ := 4

-- Proof statement
theorem total_orchids : current_orchids + additional_orchids = 6 :=
by
  sorry

end total_orchids_l23_23069


namespace symmetric_point_correct_l23_23051

def symmetric_point (P A : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x₁, y₁, z₁) := P
  let (x₀, y₀, z₀) := A
  (2 * x₀ - x₁, 2 * y₀ - y₁, 2 * z₀ - z₁)

def P : ℝ × ℝ × ℝ := (3, -2, 4)
def A : ℝ × ℝ × ℝ := (0, 1, -2)
def expected_result : ℝ × ℝ × ℝ := (-3, 4, -8)

theorem symmetric_point_correct : symmetric_point P A = expected_result :=
  by
    sorry

end symmetric_point_correct_l23_23051


namespace probability_of_both_selected_l23_23880

theorem probability_of_both_selected (pX pY : ℚ) (hX : pX = 1/7) (hY : pY = 2/5) : 
  pX * pY = 2 / 35 :=
by {
  sorry
}

end probability_of_both_selected_l23_23880


namespace jennifer_money_left_l23_23977

theorem jennifer_money_left (initial_amount sandwich_fraction museum_fraction book_fraction : ℚ)
    (initial_eq : initial_amount = 90) 
    (sandwich_eq : sandwich_fraction = 1/5) 
    (museum_eq : museum_fraction = 1/6) 
    (book_eq : book_fraction = 1/2) :
    initial_amount - (initial_amount * sandwich_fraction + initial_amount * museum_fraction + initial_amount * book_fraction) = 12 := 
by 
  sorry

end jennifer_money_left_l23_23977


namespace arithmetic_progression_no_rth_power_l23_23379

noncomputable def is_arith_sequence (a : ℕ → ℤ) : Prop := 
∀ n : ℕ, a n = 4 * (n : ℤ) - 2

theorem arithmetic_progression_no_rth_power (n : ℕ) :
  ∃ a : ℕ → ℤ, is_arith_sequence a ∧ 
  (∀ r : ℕ, 2 ≤ r ∧ r ≤ n → 
  ¬ (∃ k : ℤ, ∃ m : ℕ, m > 0 ∧ a m = k ^ r)) := 
sorry

end arithmetic_progression_no_rth_power_l23_23379


namespace power_product_is_100_l23_23298

theorem power_product_is_100 :
  (10^0.6) * (10^0.4) * (10^0.3) * (10^0.2) * (10^0.5) = 100 :=
by
  sorry

end power_product_is_100_l23_23298


namespace algebraic_expression_value_l23_23806

theorem algebraic_expression_value (a b : ℝ) (h : ∃ x : ℝ, x = 2 ∧ 3 * (a - x) = 2 * (b * x - 4)) :
  9 * a^2 - 24 * a * b + 16 * b^2 + 25 = 29 :=
by sorry

end algebraic_expression_value_l23_23806


namespace sasha_lives_on_seventh_floor_l23_23061

theorem sasha_lives_on_seventh_floor (N : ℕ) (x : ℕ) 
(h1 : x = (1/3 : ℝ) * N) 
(h2 : N - ((1/3 : ℝ) * N + 1) = (1/2 : ℝ) * N) :
  N + 1 = 7 := 
sorry

end sasha_lives_on_seventh_floor_l23_23061


namespace cyclists_meet_fourth_time_l23_23032

theorem cyclists_meet_fourth_time 
  (speed1 speed2 speed3 speed4 : ℕ)
  (len : ℚ)
  (t_start : ℕ)
  (h_speed1 : speed1 = 6)
  (h_speed2 : speed2 = 9)
  (h_speed3 : speed3 = 12)
  (h_speed4 : speed4 = 15)
  (h_len : len = 1 / 3)
  (h_t_start : t_start = 12 * 60 * 60)
  : 
  (t_start + 4 * (20 * 60 + 40)) = 12 * 60 * 60 + 1600  :=
sorry

end cyclists_meet_fourth_time_l23_23032


namespace fixed_point_exists_l23_23645

theorem fixed_point_exists : ∀ (m : ℝ), (m - 1) * (7 / 2) - (m + 3) * (5 / 2) - (m - 11) = 0 :=
by
  intro m
  have h : (m - 1) * (7 / 2) - (m + 3) * (5 / 2) - (m - 11) = 0 :=
    sorry
  exact h

end fixed_point_exists_l23_23645


namespace eight_div_repeat_three_l23_23058

-- Initial condition of the problem
def q : ℚ := 1/3

-- Main theorem to prove
theorem eight_div_repeat_three : (8 : ℚ) / q = 24 := by
  -- proof is omitted with sorry
  sorry

end eight_div_repeat_three_l23_23058


namespace problem_1_l23_23912

noncomputable def f (a x : ℝ) : ℝ := abs (x + 2) + abs (x - a)

theorem problem_1 (a : ℝ) (h : ∀ x : ℝ, f a x ≥ 3) : a ≤ -5 ∨ a ≥ 1 :=
by
  sorry

end problem_1_l23_23912


namespace parallel_vectors_l23_23376

noncomputable def vector_a : ℝ × ℝ := (2, 1)
noncomputable def vector_b (m : ℝ) : ℝ × ℝ := (m, -1)

theorem parallel_vectors {m : ℝ} (h : (∃ k : ℝ, vector_a = k • vector_b m)) : m = -2 :=
by
  sorry

end parallel_vectors_l23_23376


namespace sum_of_three_squares_l23_23803

variable (t s : ℝ)

-- Given equations
axiom h1 : 3 * t + 2 * s = 27
axiom h2 : 2 * t + 3 * s = 25

-- What we aim to prove
theorem sum_of_three_squares : 3 * s = 63 / 5 :=
by
  sorry

end sum_of_three_squares_l23_23803


namespace remainder_y150_div_yminus2_4_l23_23656

theorem remainder_y150_div_yminus2_4 (y : ℝ) :
  (y ^ 150) % ((y - 2) ^ 4) = 554350 * (y - 2) ^ 3 + 22350 * (y - 2) ^ 2 + 600 * (y - 2) + 8 * 2 ^ 147 :=
by
  sorry

end remainder_y150_div_yminus2_4_l23_23656


namespace verify_value_l23_23627

theorem verify_value (a b c d m : ℝ) 
  (h₁ : a = -b) 
  (h₂ : c * d = 1) 
  (h₃ : |m| = 3) :
  3 * c * d + (a + b) / (c * d) - m = 0 ∨ 
  3 * c * d + (a + b) / (c * d) - m = 6 := 
sorry

end verify_value_l23_23627


namespace center_cell_value_l23_23199

namespace MathProof

variables {a b c d e f g h i : ℝ}

-- Conditions
axiom row_product1 : a * b * c = 1
axiom row_product2 : d * e * f = 1
axiom row_product3 : g * h * i = 1

axiom col_product1 : a * d * g = 1
axiom col_product2 : b * e * h = 1
axiom col_product3 : c * f * i = 1

axiom square_product1 : a * b * d * e = 2
axiom square_product2 : b * c * e * f = 2
axiom square_product3 : d * e * g * h = 2
axiom square_product4 : e * f * h * i = 2

-- Proof problem
theorem center_cell_value : e = 1 :=
sorry

end MathProof

end center_cell_value_l23_23199


namespace base_conversion_difference_l23_23905

-- Definitions
def base9_to_base10 (n : ℕ) : ℕ := 3 * (9^2) + 2 * (9^1) + 7 * (9^0)
def base8_to_base10 (m : ℕ) : ℕ := 2 * (8^2) + 5 * (8^1) + 3 * (8^0)

-- Statement
theorem base_conversion_difference :
  base9_to_base10 327 - base8_to_base10 253 = 97 :=
by sorry

end base_conversion_difference_l23_23905


namespace two_crows_problem_l23_23417

def Bird := { P | P = "parrot" ∨ P = "crow"} -- Define possible bird species.

-- Define birds and their statements
def Adam_statement (Adam Carl : Bird) : Prop := Carl = Adam
def Bob_statement (Adam : Bird) : Prop := Adam = "crow"
def Carl_statement (Dave : Bird) : Prop := Dave = "crow"
def Dave_statement (Adam Bob Carl Dave: Bird) : Prop := 
  (if Adam = "parrot" then 1 else 0) + 
  (if Bob = "parrot" then 1 else 0) + 
  (if Carl = "parrot" then 1 else 0) + 
  (if Dave = "parrot" then 1 else 0) ≥ 3

-- The main proposition to prove
def main_statement : Prop :=
  ∃ (Adam Bob Carl Dave : Bird), 
    (Adam_statement Adam Carl) ∧ 
    (Bob_statement Adam) ∧ 
    (Carl_statement Dave) ∧ 
    (Dave_statement Adam Bob Carl Dave) ∧ 
    (if Adam = "crow" then 1 else 0) + 
    (if Bob = "crow" then 1 else 0) + 
    (if Carl = "crow" then 1 else 0) + 
    (if Dave = "crow" then 1 else 0) = 2

-- Proof statement to be filled
theorem two_crows_problem : main_statement :=
by {
  sorry
}

end two_crows_problem_l23_23417


namespace eliza_height_is_68_l23_23511

-- Define the known heights of the siblings
def height_sibling_1 : ℕ := 66
def height_sibling_2 : ℕ := 66
def height_sibling_3 : ℕ := 60

-- The total height of all 5 siblings combined
def total_height : ℕ := 330

-- Eliza is 2 inches shorter than the last sibling
def height_difference : ℕ := 2

-- Define the heights of the siblings
def height_remaining_siblings := total_height - (height_sibling_1 + height_sibling_2 + height_sibling_3)

-- The height of the last sibling
def height_last_sibling := (height_remaining_siblings + height_difference) / 2

-- Eliza's height
def height_eliza := height_last_sibling - height_difference

-- We need to prove that Eliza's height is 68 inches
theorem eliza_height_is_68 : height_eliza = 68 := by
  sorry

end eliza_height_is_68_l23_23511


namespace inequality_proof_l23_23349

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a * b * c = 1/8) :
  a^2 + b^2 + c^2 + a^2 * b^2 + b^2 * c^2 + c^2 * a^2 ≥ 15 / 16 :=
by sorry

end inequality_proof_l23_23349


namespace jimmy_needs_4_packs_of_bread_l23_23571

theorem jimmy_needs_4_packs_of_bread
  (num_sandwiches : ℕ)
  (slices_per_sandwich : ℕ)
  (slices_per_pack : ℕ)
  (initial_slices : ℕ)
  (h1 : num_sandwiches = 8)
  (h2 : slices_per_sandwich = 2)
  (h3 : slices_per_pack = 4)
  (h4 : initial_slices = 0) :
  (num_sandwiches * slices_per_sandwich) / slices_per_pack = 4 := by
  sorry

end jimmy_needs_4_packs_of_bread_l23_23571


namespace hexagon_angle_D_135_l23_23076

theorem hexagon_angle_D_135 
  (A B C D E F : ℝ)
  (h1 : A = B ∧ B = C)
  (h2 : D = E ∧ E = F)
  (h3 : A = D - 30)
  (h4 : A + B + C + D + E + F = 720) :
  D = 135 :=
by {
  sorry
}

end hexagon_angle_D_135_l23_23076


namespace find_value_in_table_l23_23910

theorem find_value_in_table :
  let W := 'W'
  let L := 'L'
  let Q := 'Q'
  let table := [
    [W, '?', Q],
    [L, Q, W],
    [Q, W, L]
  ]
  table[0][1] = L :=
by
  sorry

end find_value_in_table_l23_23910


namespace prob_two_blue_balls_l23_23421

-- Ball and Urn Definitions
def total_balls : ℕ := 10
def blue_balls_initial : ℕ := 6
def red_balls_initial : ℕ := 4

-- Probabilities
def prob_blue_first_draw : ℚ := blue_balls_initial / total_balls
def prob_blue_second_draw_given_first_blue : ℚ :=
  (blue_balls_initial - 1) / (total_balls - 1)

-- Resulting Probability
def prob_both_blue : ℚ := prob_blue_first_draw * prob_blue_second_draw_given_first_blue

-- Statement to Prove
theorem prob_two_blue_balls :
  prob_both_blue = 1 / 3 :=
by
  sorry

end prob_two_blue_balls_l23_23421


namespace power_of_fraction_to_decimal_l23_23490

theorem power_of_fraction_to_decimal : ∃ x : ℕ, (1 / 9 : ℚ) ^ x = 1 / 81 ∧ x = 2 :=
by
  use 2
  simp
  sorry

end power_of_fraction_to_decimal_l23_23490


namespace plane_equation_through_point_and_line_l23_23922

theorem plane_equation_through_point_and_line :
  ∃ (A B C D : ℤ), A > 0 ∧ Int.gcd A B = 1 ∧ Int.gcd A C = 1 ∧ Int.gcd A D = 1 ∧
  ∀ (x y z : ℝ),
    (A * x + B * y + C * z + D = 0 ↔ 
    (∃ (t : ℝ), x = -3 * t - 1 ∧ y = 2 * t + 3 ∧ z = t - 2) ∨ 
    (x = 0 ∧ y = 7 ∧ z = -7)) :=
by
  -- sorry, implementing proofs is not required.
  sorry

end plane_equation_through_point_and_line_l23_23922


namespace distance_between_A_and_B_l23_23291

theorem distance_between_A_and_B 
    (Time_E : ℝ) (Time_F : ℝ) (D_AC : ℝ) (V_ratio : ℝ)
    (E_time : Time_E = 3) (F_time : Time_F = 4) 
    (AC_distance : D_AC = 300) (speed_ratio : V_ratio = 4) : 
    ∃ D_AB : ℝ, D_AB = 900 :=
by
  sorry

end distance_between_A_and_B_l23_23291


namespace fruit_basket_count_l23_23133

-- Define the number of apples and oranges
def apples := 7
def oranges := 12

-- Condition: A fruit basket must contain at least two pieces of fruit
def min_pieces_of_fruit := 2

-- Problem: Prove that there are 101 different fruit baskets containing at least two pieces of fruit
theorem fruit_basket_count (n_apples n_oranges n_min_pieces : Nat) (h_apples : n_apples = apples) (h_oranges : n_oranges = oranges) (h_min_pieces : n_min_pieces = min_pieces_of_fruit) :
  (n_apples = 7) ∧ (n_oranges = 12) ∧ (n_min_pieces = 2) → (104 - 3 = 101) :=
by
  sorry

end fruit_basket_count_l23_23133


namespace solution_exists_l23_23742

def operation (a b : ℚ) : ℚ :=
if a ≥ b then a^2 * b else a * b^2

theorem solution_exists (m : ℚ) (h : operation 3 m = 48) : m = 4 := by
  sorry

end solution_exists_l23_23742


namespace parabolas_intersect_at_point_l23_23709

theorem parabolas_intersect_at_point :
  ∀ (p q : ℝ), p + q = 2019 → (1 : ℝ)^2 + (p : ℝ) * 1 + q = 2020 :=
by
  intros p q h
  sorry

end parabolas_intersect_at_point_l23_23709


namespace sufficient_condition_l23_23100

theorem sufficient_condition (A B C D : Prop) (h : C → D): C → (A > B) := 
by 
  sorry

end sufficient_condition_l23_23100


namespace cubed_sum_identity_l23_23929

theorem cubed_sum_identity {x y : ℝ} (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 := by
  sorry

end cubed_sum_identity_l23_23929


namespace cell_value_l23_23306

variable (P Q R S : ℕ)

-- Condition definitions
def topLeftCell (P : ℕ) : ℕ := P
def topMiddleCell (P Q : ℕ) : ℕ := P + Q
def centerCell (P Q R S : ℕ) : ℕ := P + Q + R + S
def bottomLeftCell (S : ℕ) : ℕ := S

-- Given Conditions
axiom bottomLeftCell_value : bottomLeftCell S = 13
axiom topMiddleCell_value : topMiddleCell P Q = 18
axiom centerCell_value : centerCell P Q R S = 47

-- To prove: R = 16
theorem cell_value : R = 16 :=
by
  sorry

end cell_value_l23_23306


namespace percentage_of_boys_playing_soccer_l23_23898

theorem percentage_of_boys_playing_soccer
  (total_students : ℕ)
  (boys : ℕ)
  (students_playing_soccer : ℕ)
  (girl_students_not_playing_soccer : ℕ)
  (h1 : total_students = 420)
  (h2 : boys = 296)
  (h3 : students_playing_soccer = 250)
  (h4 : girl_students_not_playing_soccer = 89) :
  (students_playing_soccer - (total_students - boys - girl_students_not_playing_soccer)) * 100 / students_playing_soccer = 86 :=
by
  sorry

end percentage_of_boys_playing_soccer_l23_23898


namespace determine_m_from_quadratic_l23_23906

def is_prime (n : ℕ) := 2 ≤ n ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

theorem determine_m_from_quadratic (x1 x2 m : ℕ) (hx1_prime : is_prime x1) (hx2_prime : is_prime x2) 
    (h_roots : x1 + x2 = 1999) (h_product : x1 * x2 = m) : 
    m = 3994 := 
by 
    sorry

end determine_m_from_quadratic_l23_23906


namespace maximize_profit_l23_23360

noncomputable def production_problem : Prop :=
  ∃ (x y : ℕ), (3 * x + 2 * y ≤ 1200) ∧ (x + 2 * y ≤ 800) ∧ 
               (30 * x + 40 * y) = 18000 ∧ 
               x = 200 ∧ 
               y = 300

theorem maximize_profit : production_problem :=
sorry

end maximize_profit_l23_23360


namespace number_of_segments_l23_23126

theorem number_of_segments (tangent_chords : ℕ) (angle_ABC : ℝ) (h : angle_ABC = 80) :
  tangent_chords = 18 :=
sorry

end number_of_segments_l23_23126


namespace min_gennadys_needed_l23_23640

variables (A B V G : ℕ)

theorem min_gennadys_needed
  (hA : A = 45)
  (hB : B = 122)
  (hV : V = 27)
  (hG : ∀ i, i < 121 → A + V < 121 → G ≥ 49) :
  G = 49 :=
sorry

end min_gennadys_needed_l23_23640


namespace largest_even_whole_number_l23_23504

theorem largest_even_whole_number (x : ℕ) (h1 : 9 * x < 150) (h2 : x % 2 = 0) : x ≤ 16 :=
by
  sorry

end largest_even_whole_number_l23_23504


namespace theta_in_fourth_quadrant_l23_23436

theorem theta_in_fourth_quadrant (θ : ℝ) (h1 : Real.cos θ > 0) (h2 : Real.sin (2 * θ) < 0) : 
  (∃ k : ℤ, θ = 2 * π * k + 7 * π / 4 ∨ θ = 2 * π * k + π / 4) ∧ θ = 2 * π * k + 7 * π / 4 :=
sorry

end theta_in_fourth_quadrant_l23_23436


namespace integer_representation_l23_23940

theorem integer_representation (n : ℤ) : ∃ x y z : ℤ, n = x^2 + y^2 - z^2 :=
by sorry

end integer_representation_l23_23940


namespace trapezium_perimeters_l23_23414

theorem trapezium_perimeters (AB BC AD AF : ℝ)
  (h1 : AB = 30) (h2 : BC = 30) (h3 : AD = 25) (h4 : AF = 24) :
  ∃ p : ℝ, (p = 90 ∨ p = 104) :=
by
  sorry

end trapezium_perimeters_l23_23414


namespace tommy_house_price_l23_23997

variable (P : ℝ)

theorem tommy_house_price 
  (h1 : 1.25 * P = 125000) : 
  P = 100000 :=
by
  sorry

end tommy_house_price_l23_23997


namespace square_sum_zero_real_variables_l23_23740

theorem square_sum_zero_real_variables (a b : ℝ) (h : a^2 + b^2 = 0) : a = 0 ∧ b = 0 :=
sorry

end square_sum_zero_real_variables_l23_23740


namespace abc_positive_l23_23240

theorem abc_positive (a b c : ℝ) (h1 : a + b + c > 0) (h2 : ab + bc + ca > 0) (h3 : abc > 0) :
  a > 0 ∧ b > 0 ∧ c > 0 :=
by
  -- Proof goes here
  sorry

end abc_positive_l23_23240


namespace fraction_notation_correct_reading_decimal_correct_l23_23535

-- Define the given conditions
def fraction_notation (num denom : ℕ) : Prop :=
  num / denom = num / denom  -- Essentially stating that in fraction notation, it holds

def reading_decimal (n : ℚ) (s : String) : Prop :=
  if n = 90.58 then s = "ninety point five eight" else false -- Defining the reading rule for this specific case

-- State the theorem using the defined conditions
theorem fraction_notation_correct : fraction_notation 8 9 := 
by 
  sorry

theorem reading_decimal_correct : reading_decimal 90.58 "ninety point five eight" :=
by 
  sorry

end fraction_notation_correct_reading_decimal_correct_l23_23535


namespace oak_trees_initial_count_l23_23493

theorem oak_trees_initial_count (x : ℕ) (cut_down : ℕ) (remaining : ℕ) (h_cut : cut_down = 2) (h_remaining : remaining = 7)
  (h_equation : (x - cut_down) = remaining) : x = 9 := by
  -- We are given that cut_down = 2
  -- and remaining = 7
  -- and we need to show that the initial count x = 9
  sorry

end oak_trees_initial_count_l23_23493


namespace second_number_is_three_l23_23422

theorem second_number_is_three (x y : ℝ) (h1 : x + y = 10) (h2 : 2 * x = 3 * y + 5) : y = 3 :=
by
  -- To be proved: sorry for now
  sorry

end second_number_is_three_l23_23422


namespace factorize_n_squared_minus_nine_l23_23618

theorem factorize_n_squared_minus_nine (n : ℝ) : n^2 - 9 = (n + 3) * (n - 3) := 
sorry

end factorize_n_squared_minus_nine_l23_23618


namespace f_500_l23_23996

-- Define a function f on positive integers
def f (n : ℕ) : ℕ := sorry

-- Assume the given conditions
axiom f_mul (x y : ℕ) (hx : x > 0) (hy : y > 0) : f (x * y) = f x + f y
axiom f_10 : f 10 = 14
axiom f_40 : f 40 = 20

-- Prove the required result
theorem f_500 : f 500 = 39 := by
  sorry

end f_500_l23_23996


namespace smallest_mu_real_number_l23_23752

theorem smallest_mu_real_number (a b c d : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : 0 ≤ d) :
  a^2 + b^2 + c^2 + d^2 ≤ ab + (3/2) * bc + cd :=
sorry

end smallest_mu_real_number_l23_23752


namespace set_subset_find_m_l23_23995

open Set

def A (m : ℝ) : Set ℝ := {1, 3, 2 * m + 3}
def B (m : ℝ) : Set ℝ := {3, m^2}

theorem set_subset_find_m (m : ℝ) : (B m ⊆ A m) → (m = 1 ∨ m = 3) :=
by 
  intro h
  sorry

end set_subset_find_m_l23_23995


namespace intersection_of_M_and_N_l23_23655

-- Define set M
def M : Set ℝ := {x | Real.log x > 0}

-- Define set N
def N : Set ℝ := {x | x^2 ≤ 4}

-- Define the target set
def target : Set ℝ := {x | 1 < x ∧ x ≤ 2}

theorem intersection_of_M_and_N :
  M ∩ N = target :=
sorry

end intersection_of_M_and_N_l23_23655


namespace smallest_positive_expr_l23_23699

theorem smallest_positive_expr (m n : ℤ) : ∃ (m n : ℤ), 216 * m + 493 * n = 1 := 
sorry

end smallest_positive_expr_l23_23699


namespace number_of_routes_600_l23_23095

-- Define the problem conditions
def number_of_routes (total_cities : Nat) (pick_cities : Nat) (selected_cities : List Nat) : Nat := sorry

-- The number of ways to pick and order 3 cities from remaining 5
def num_ways_pick_three (total_cities : Nat) (pick_cities : Nat) : Nat :=
  Nat.factorial total_cities / Nat.factorial (total_cities - pick_cities)

-- The number of ways to choose positions for M and N
def num_ways_positions (total_positions : Nat) (pick_positions : Nat) : Nat :=
  Nat.choose total_positions pick_positions

-- The main theorem to prove
theorem number_of_routes_600 :
  number_of_routes 7 5 [M, N] = num_ways_pick_three 5 3 * num_ways_positions 4 2 :=
  by sorry

end number_of_routes_600_l23_23095


namespace max_and_min_W_l23_23594

noncomputable def W (x y z : ℝ) : ℝ := 2 * x + 6 * y + 4 * z

theorem max_and_min_W {x y z : ℝ} (h1 : x + y + z = 1) (h2 : 3 * y + z ≥ 2) (h3 : 0 ≤ x ∧ x ≤ 1) (h4 : 0 ≤ y ∧ y ≤ 2) :
  ∃ (W_max W_min : ℝ), W_max = 6 ∧ W_min = 4 :=
by
  sorry

end max_and_min_W_l23_23594


namespace trigonometric_identity_l23_23578

theorem trigonometric_identity :
  1 / Real.sin (70 * Real.pi / 180) - Real.sqrt 2 / Real.cos (70 * Real.pi / 180) = 
  -2 * (Real.sin (25 * Real.pi / 180) / Real.sin (40 * Real.pi / 180)) :=
sorry

end trigonometric_identity_l23_23578


namespace proof_q_is_true_l23_23405

variable (p q : Prop)

-- Assuming the conditions
axiom h1 : p ∨ q   -- p or q is true
axiom h2 : ¬ p     -- not p is true

-- Theorem statement to prove q is true
theorem proof_q_is_true : q :=
by
  sorry

end proof_q_is_true_l23_23405


namespace find_m_l23_23532

noncomputable def m_value (m : ℝ) := 
  ((m ^ 2) - m - 1, (m ^ 2) - 2 * m - 1)

theorem find_m (m : ℝ) (h1 : (m ^ 2) - m - 1 = 1) (h2 : (m ^ 2) - 2 * m - 1 < 0) : 
  m = 2 :=
by sorry

end find_m_l23_23532


namespace least_value_y_l23_23502

theorem least_value_y
  (h : ∀ y : ℝ, 5 * y ^ 2 + 7 * y + 3 = 6 → -3 ≤ y) : 
  ∃ y : ℝ, 5 * y ^ 2 + 7 * y + 3 = 6 ∧ y = -3 :=
by
  sorry

end least_value_y_l23_23502


namespace exists_integers_m_n_l23_23652

theorem exists_integers_m_n (x y : ℝ) (hxy : x ≠ y) : 
  ∃ (m n : ℤ), (m * x + n * y > 0) ∧ (n * x + m * y < 0) :=
sorry

end exists_integers_m_n_l23_23652


namespace problem_1_problem_2_l23_23247

variable (a : ℝ) (x : ℝ)

theorem problem_1 (h : a ≠ 1) : (a^2 / (a - 1)) - (a / (a - 1)) = a := 
sorry

theorem problem_2 (h : x ≠ -1) : (x^2 / (x + 1)) - x + 1 = 1 / (x + 1) := 
sorry

end problem_1_problem_2_l23_23247


namespace equation_holds_l23_23017

-- Positive integers less than 10
def is_lt_10 (x : ℕ) : Prop := x > 0 ∧ x < 10

theorem equation_holds (a b c : ℕ) (ha : is_lt_10 a) (hb : is_lt_10 b) (hc : is_lt_10 c) :
  (10 * a + b) * (10 * a + c) = 100 * a * (a + 1) + b * c ↔ b + c = 10 :=
by
  sorry

end equation_holds_l23_23017


namespace sum_of_angles_is_55_l23_23973

noncomputable def arc_BR : ℝ := 60
noncomputable def arc_RS : ℝ := 50
noncomputable def arc_AC : ℝ := 0
noncomputable def arc_BS := arc_BR + arc_RS
noncomputable def angle_P := (arc_BS - arc_AC) / 2
noncomputable def angle_R := arc_AC / 2
noncomputable def sum_of_angles := angle_P + angle_R

theorem sum_of_angles_is_55 :
  sum_of_angles = 55 :=
by
  sorry

end sum_of_angles_is_55_l23_23973


namespace parabola_directrix_l23_23961

theorem parabola_directrix : 
  ∃ d : ℝ, (∀ (y : ℝ), x = - (1 / 4) * y^2 -> 
  ( - (1 / 4) * y^2 = d)) ∧ d = 1 :=
by
  sorry

end parabola_directrix_l23_23961


namespace evaluate_Q_at_2_and_neg2_l23_23394

-- Define the polynomial Q and the conditions
variable {Q : ℤ → ℤ}
variable {m : ℤ}

-- The given conditions
axiom cond1 : Q 0 = m
axiom cond2 : Q 1 = 3 * m
axiom cond3 : Q (-1) = 4 * m

-- The proof goal
theorem evaluate_Q_at_2_and_neg2 : Q 2 + Q (-2) = 22 * m :=
sorry

end evaluate_Q_at_2_and_neg2_l23_23394


namespace manny_has_more_10_bills_than_mandy_l23_23791

theorem manny_has_more_10_bills_than_mandy :
  let mandy_bills_20 := 3
  let manny_bills_50 := 2
  let mandy_total_money := 20 * mandy_bills_20
  let manny_total_money := 50 * manny_bills_50
  let mandy_10_bills := mandy_total_money / 10
  let manny_10_bills := manny_total_money / 10
  mandy_10_bills < manny_10_bills →
  manny_10_bills - mandy_10_bills = 4 := sorry

end manny_has_more_10_bills_than_mandy_l23_23791


namespace sum_of_integers_l23_23834

theorem sum_of_integers (m n : ℕ) (h1 : m * n = 2 * (m + n)) (h2 : m * n = 6 * (m - n)) :
  m + n = 9 := by
  sorry

end sum_of_integers_l23_23834


namespace range_of_values_for_a_l23_23067

noncomputable def problem_statement (a : ℝ) : Prop :=
  ∀ x : ℝ, ¬ (|x - 2| + |x + 3| < a)

theorem range_of_values_for_a (a : ℝ) :
  problem_statement a → a ≤ 5 :=
  sorry

end range_of_values_for_a_l23_23067


namespace kindergarten_solution_l23_23825

def kindergarten_cards (x y z t : ℕ) : Prop :=
  (x + y = 20) ∧ (z + t = 30) ∧ (y + z = 40) → (x + t = 10)

theorem kindergarten_solution : ∃ (x y z t : ℕ), kindergarten_cards x y z t :=
by {
  sorry
}

end kindergarten_solution_l23_23825


namespace first_year_after_2020_with_digit_sum_18_l23_23483

theorem first_year_after_2020_with_digit_sum_18 : 
  ∃ (y : ℕ), y > 2020 ∧ (∃ a b c : ℕ, (2 + a + b + c = 18 ∧ y = 2000 + 100 * a + 10 * b + c)) ∧ y = 2799 := 
sorry

end first_year_after_2020_with_digit_sum_18_l23_23483


namespace max_c_l23_23471

theorem max_c (c : ℝ) : 
  (∀ x y : ℝ, x > y ∧ y > 0 → x^2 - 2 * y^2 ≤ c * x * (y - x)) 
  → c ≤ 2 * Real.sqrt 2 - 4 := 
by
  sorry

end max_c_l23_23471


namespace total_houses_in_lincoln_county_l23_23387

theorem total_houses_in_lincoln_county 
  (original_houses : ℕ) 
  (houses_built : ℕ) 
  (h_original : original_houses = 20817) 
  (h_built : houses_built = 97741) : 
  original_houses + houses_built = 118558 := 
by 
  -- Proof steps or tactics would go here
  sorry

end total_houses_in_lincoln_county_l23_23387


namespace factorization_1_min_value_l23_23401

-- Problem 1: Prove that m² - 4mn + 3n² = (m - 3n)(m - n)
theorem factorization_1 (m n : ℤ) : m^2 - 4*m*n + 3*n^2 = (m - 3*n)*(m - n) :=
by
  sorry

-- Problem 2: Prove that the minimum value of m² - 3m + 2015 is 2012 3/4
theorem min_value (m : ℝ) : ∃ x : ℝ, x = m^2 - 3*m + 2015 ∧ x = 2012 + 3/4 :=
by
  sorry

end factorization_1_min_value_l23_23401


namespace modified_cube_edges_l23_23098

/--
A solid cube with a side length of 4 has different-sized solid cubes removed from three of its corners:
- one corner loses a cube of side length 1,
- another corner loses a cube of side length 2,
- and a third corner loses a cube of side length 1.

The total number of edges of the modified solid is 22.
-/
theorem modified_cube_edges :
  let original_edges := 12
  let edges_removed_1x1 := 6
  let edges_added_2x2 := 16
  original_edges - 2 * edges_removed_1x1 + edges_added_2x2 = 22 := by
  sorry

end modified_cube_edges_l23_23098


namespace methane_hydrate_scientific_notation_l23_23923

theorem methane_hydrate_scientific_notation :
  (9.2 * 10^(-4)) = 0.00092 :=
by sorry

end methane_hydrate_scientific_notation_l23_23923


namespace percentage_decrease_is_correct_l23_23630

variable (P : ℝ)

-- Condition 1: After the first year, the price increased by 30%
def price_after_first_year : ℝ := 1.30 * P

-- Condition 2: At the end of the 2-year period, the price of the painting is 110.5% of the original price
def price_after_second_year : ℝ := 1.105 * P

-- Condition 3: Let D be the percentage decrease during the second year
def D : ℝ := 0.15

-- Goal: Prove that the percentage decrease during the second year is 15%
theorem percentage_decrease_is_correct : 
  1.30 * P - D * 1.30 * P = 1.105 * P → D = 0.15 :=
by
  sorry

end percentage_decrease_is_correct_l23_23630


namespace count_diff_squares_not_representable_1_to_1000_l23_23970

def num_not_diff_squares (n : ℕ) : ℕ :=
  (n + 1) / 4 * (if (n + 1) % 4 >= 2 then 1 else 0)

theorem count_diff_squares_not_representable_1_to_1000 :
  num_not_diff_squares 999 = 250 := 
sorry

end count_diff_squares_not_representable_1_to_1000_l23_23970


namespace ordering_of_powers_l23_23536

theorem ordering_of_powers :
  2^30 < 10^10 ∧ 10^10 < 5^15 :=
by sorry

end ordering_of_powers_l23_23536


namespace x_coordinate_D_l23_23595

noncomputable def find_x_coordinate_D (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) : ℝ := 
  let l := -a * b
  let x := l / c
  x

theorem x_coordinate_D (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (D_on_parabola : d^2 = (a + b) * (d) + l)
  (lines_intersect_y_axis : ∃ l : ℝ, (a^2 = (b + a) * a + l) ∧ (b^2 = (b + a) * b + l) ∧ (c^2 = (d + c) * c + l)) :
  d = (a * b) / c :=
by sorry

end x_coordinate_D_l23_23595


namespace fred_earnings_over_weekend_l23_23452

-- Fred's earning from delivering newspapers
def earnings_from_newspapers : ℕ := 16

-- Fred's earning from washing cars
def earnings_from_cars : ℕ := 74

-- Fred's total earnings over the weekend
def total_earnings : ℕ := earnings_from_newspapers + earnings_from_cars

-- Proof that total earnings is 90
theorem fred_earnings_over_weekend : total_earnings = 90 :=
by 
  -- sorry statement to skip the proof steps
  sorry

end fred_earnings_over_weekend_l23_23452


namespace quadratic_has_minimum_l23_23167

theorem quadratic_has_minimum (a b : ℝ) (h : a > b^2) :
  ∃ (c : ℝ), c = (4 * b^2 / a) - 3 ∧ (∃ x : ℝ, a * x ^ 2 + 2 * b * x + c < 0) :=
by sorry

end quadratic_has_minimum_l23_23167


namespace total_length_of_river_is_80_l23_23215

-- Definitions based on problem conditions
def straight_part_length := 20
def crooked_part_length := 3 * straight_part_length
def total_length_of_river := straight_part_length + crooked_part_length

-- Theorem stating that the total length of the river is 80 miles
theorem total_length_of_river_is_80 :
  total_length_of_river = 80 := by
    -- The proof is omitted
    sorry

end total_length_of_river_is_80_l23_23215


namespace brendan_remaining_money_l23_23041

-- Definitions based on conditions
def earned_amount : ℕ := 5000
def recharge_rate : ℕ := 1/2
def car_cost : ℕ := 1500

-- Proof Statement
theorem brendan_remaining_money : 
  (earned_amount * recharge_rate) - car_cost = 1000 :=
sorry

end brendan_remaining_money_l23_23041


namespace bell_peppers_needed_l23_23433

-- Definitions based on the conditions
def large_slices_per_bell_pepper : ℕ := 20
def small_pieces_from_half_slices : ℕ := (20 / 2) * 3
def total_slices_and_pieces_per_bell_pepper : ℕ := large_slices_per_bell_pepper / 2 + small_pieces_from_half_slices
def desired_total_slices_and_pieces : ℕ := 200

-- Proving the number of bell peppers needed
theorem bell_peppers_needed : 
  desired_total_slices_and_pieces / total_slices_and_pieces_per_bell_pepper = 5 := 
by 
  -- Add the proof steps here
  sorry

end bell_peppers_needed_l23_23433


namespace trader_profit_l23_23130

noncomputable def original_price (P : ℝ) : ℝ := P
noncomputable def discount_price (P : ℝ) : ℝ := 0.95 * P
noncomputable def selling_price (P : ℝ) : ℝ := 1.52 * P
noncomputable def profit (P : ℝ) : ℝ := selling_price P - original_price P
noncomputable def percent_profit (P : ℝ) : ℝ := (profit P / original_price P) * 100

theorem trader_profit (P : ℝ) (hP : 0 < P) : percent_profit P = 52 := by 
  sorry

end trader_profit_l23_23130


namespace curve_is_circle_l23_23679

-- Definition of the curve in polar coordinates
def curve (r θ : ℝ) : Prop :=
  r = 3 * Real.sin θ

-- The theorem to prove
theorem curve_is_circle : ∀ θ : ℝ, ∃ r : ℝ, curve r θ → (∃ c : ℝ × ℝ, ∃ R : ℝ, ∀ p : ℝ × ℝ, (Real.sqrt ((p.1 - c.1) ^ 2 + (p.2 - c.2) ^ 2) = R)) :=
by
  sorry

end curve_is_circle_l23_23679


namespace volume_of_each_hemisphere_container_is_correct_l23_23442

-- Define the given conditions
def Total_volume : ℕ := 10936
def Number_containers : ℕ := 2734

-- Define the volume of each hemisphere container
def Volume_each_container : ℕ := Total_volume / Number_containers

-- The theorem to prove, asserting the volume is correct
theorem volume_of_each_hemisphere_container_is_correct :
  Volume_each_container  = 4 := by
  -- placeholder for the actual proof
  sorry

end volume_of_each_hemisphere_container_is_correct_l23_23442


namespace trig_expression_eval_l23_23506

open Real

-- Declare the main theorem
theorem trig_expression_eval (θ : ℝ) (k : ℤ) 
  (h : sin (θ + k * π) = -2 * cos (θ + k * π)) :
  (4 * sin θ - 2 * cos θ) / (5 * cos θ + 3 * sin θ) = 10 :=
  sorry

end trig_expression_eval_l23_23506


namespace michael_number_l23_23785

theorem michael_number (m : ℕ) (h1 : m % 75 = 0) (h2 : m % 40 = 0) (h3 : 1000 < m) (h4 : m < 3000) :
  m = 1800 ∨ m = 2400 ∨ m = 3000 :=
sorry

end michael_number_l23_23785


namespace third_beats_seventh_l23_23388

-- Definitions and conditions
variable (points : Fin 8 → ℕ)
variable (distinct_points : Function.Injective points)
variable (sum_last_four : points 1 = points 4 + points 5 + points 6 + points 7)

-- Proof statement
theorem third_beats_seventh 
  (h_distinct : ∀ i j, i ≠ j → points i ≠ points j)
  (h_sum : points 1 = points 4 + points 5 + points 6 + points 7) :
  points 2 > points 6 :=
sorry

end third_beats_seventh_l23_23388


namespace sum_of_series_equals_one_half_l23_23300

theorem sum_of_series_equals_one_half : 
  (∑' k : ℕ, (1 / ((2 * k + 1) * (2 * k + 3)))) = 1 / 2 :=
sorry

end sum_of_series_equals_one_half_l23_23300


namespace jan_uses_24_gallons_for_plates_and_clothes_l23_23230

theorem jan_uses_24_gallons_for_plates_and_clothes :
  (65 - (2 * 7 + (2 * 7 - 11))) / 2 = 24 :=
by sorry

end jan_uses_24_gallons_for_plates_and_clothes_l23_23230


namespace count_valid_three_digit_numbers_l23_23361

theorem count_valid_three_digit_numbers : 
  let total_three_digit_numbers := 900 
  let invalid_AAB_or_ABA := 81 + 81
  total_three_digit_numbers - invalid_AAB_or_ABA = 738 := 
by 
  let total_three_digit_numbers := 900
  let invalid_AAB_or_ABA := 81 + 81
  show total_three_digit_numbers - invalid_AAB_or_ABA = 738 
  sorry

end count_valid_three_digit_numbers_l23_23361


namespace integer_solutions_eq_400_l23_23443

theorem integer_solutions_eq_400 : 
  ∃ (s : Finset (ℤ × ℤ)), (∀ x y, (x, y) ∈ s ↔ |3 * x + 2 * y| + |2 * x + y| = 100) ∧ s.card = 400 :=
sorry

end integer_solutions_eq_400_l23_23443


namespace inequality_proof_l23_23210

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + 3 * c) / (a + 2 * b + c) + (4 * b) / (a + b + 2 * c) - (8 * c) / (a + b + 3 * c) ≥ -17 + 12 * Real.sqrt 2 :=
  sorry

end inequality_proof_l23_23210


namespace jackson_maximum_usd_l23_23659

-- Define the rates for chores in various currencies
def usd_per_hour : ℝ := 5
def gbp_per_hour : ℝ := 3
def jpy_per_hour : ℝ := 400
def eur_per_hour : ℝ := 4

-- Define the hours Jackson worked for each task
def usd_hours_vacuuming : ℝ := 2 * 2
def gbp_hours_washing_dishes : ℝ := 0.5
def jpy_hours_cleaning_bathroom : ℝ := 1.5
def eur_hours_sweeping_yard : ℝ := 1

-- Define the exchange rates over three days
def exchange_rates_day1 := (1.35, 0.009, 1.18)  -- (GBP to USD, JPY to USD, EUR to USD)
def exchange_rates_day2 := (1.38, 0.0085, 1.20)
def exchange_rates_day3 := (1.33, 0.0095, 1.21)

-- Define a function to convert currency to USD based on best exchange rates
noncomputable def max_usd (gbp_to_usd jpy_to_usd eur_to_usd : ℝ) : ℝ :=
  (usd_hours_vacuuming * usd_per_hour) +
  (gbp_hours_washing_dishes * gbp_per_hour * gbp_to_usd) +
  (jpy_hours_cleaning_bathroom * jpy_per_hour * jpy_to_usd) +
  (eur_hours_sweeping_yard * eur_per_hour * eur_to_usd)

-- Prove the maximum USD Jackson can have by choosing optimal rates is $32.61
theorem jackson_maximum_usd : max_usd 1.38 0.0095 1.21 = 32.61 :=
by
  sorry

end jackson_maximum_usd_l23_23659


namespace total_oranges_after_increase_l23_23063

theorem total_oranges_after_increase :
  let Mary := 122
  let Jason := 105
  let Tom := 85
  let Sarah := 134
  let increase_rate := 0.10
  let new_Mary := Mary + Mary * increase_rate
  let new_Jason := Jason + Jason * increase_rate
  let new_Tom := Tom + Tom * increase_rate
  let new_Sarah := Sarah + Sarah * increase_rate
  let total_new_oranges := new_Mary + new_Jason + new_Tom + new_Sarah
  Float.round total_new_oranges = 491 := 
by
  sorry

end total_oranges_after_increase_l23_23063


namespace no_real_roots_l23_23854

theorem no_real_roots 
    (h : ∀ x : ℝ, (3 * x^2 / (x - 2)) - (3 * x + 8) / 2 + (5 - 9 * x) / (x - 2) + 2 = 0) 
    : False := by
  sorry

end no_real_roots_l23_23854


namespace percentage_error_in_area_l23_23480

noncomputable def side_with_error (s : ℝ) : ℝ := 1.04 * s

noncomputable def actual_area (s : ℝ) : ℝ := s ^ 2

noncomputable def calculated_area (s : ℝ) : ℝ := (side_with_error s) ^ 2

noncomputable def percentage_error (actual : ℝ) (calculated : ℝ) : ℝ :=
  ((calculated - actual) / actual) * 100

theorem percentage_error_in_area (s : ℝ) :
  percentage_error (actual_area s) (calculated_area s) = 8.16 := by
  sorry

end percentage_error_in_area_l23_23480


namespace find_b_when_a_is_negative12_l23_23188

theorem find_b_when_a_is_negative12 (a b : ℝ) (h1 : a + b = 60) (h2 : a = 3 * b) (h3 : ∃ k, a * b = k) : b = -56.25 :=
sorry

end find_b_when_a_is_negative12_l23_23188


namespace percentage_concentration_acid_l23_23154

-- Definitions based on the given conditions
def volume_acid : ℝ := 1.6
def total_volume : ℝ := 8.0

-- Lean statement to prove the percentage concentration is 20%
theorem percentage_concentration_acid : (volume_acid / total_volume) * 100 = 20 := by
  sorry

end percentage_concentration_acid_l23_23154


namespace suzanna_history_book_pages_l23_23743

theorem suzanna_history_book_pages (H G M S : ℕ) 
  (h_geography : G = H + 70)
  (h_math : M = (1 / 2) * (H + H + 70))
  (h_science : S = 2 * H)
  (h_total : H + G + M + S = 905) : 
  H = 160 := 
by
  sorry

end suzanna_history_book_pages_l23_23743


namespace find_third_number_l23_23875
open BigOperators

def LCM (a b : ℕ) : ℕ := a * b / (Nat.gcd a b)

def LCM_of_three (a b c : ℕ) : ℕ := LCM (LCM a b) c

theorem find_third_number (n : ℕ) (h₁ : LCM 15 25 = 75) (h₂ : LCM_of_three 15 25 n = 525) : n = 7 :=
by 
  sorry

end find_third_number_l23_23875


namespace remainder_when_x_minus_y_div_18_l23_23005

variable (k m : ℤ)
variable (x y : ℤ)
variable (h1 : x = 72 * k + 65)
variable (h2 : y = 54 * m + 22)

theorem remainder_when_x_minus_y_div_18 :
  (x - y) % 18 = 7 := by
sorry

end remainder_when_x_minus_y_div_18_l23_23005


namespace oranges_cost_l23_23537

def cost_for_multiple_dozens (price_per_dozen: ℝ) (dozens: ℝ) : ℝ := 
    price_per_dozen * dozens

theorem oranges_cost (price_for_4_dozens: ℝ) (price_for_5_dozens: ℝ) :
  price_for_4_dozens = 28.80 →
  price_for_5_dozens = cost_for_multiple_dozens (28.80 / 4) 5 →
  price_for_5_dozens = 36 :=
by
  intros h1 h2
  sorry

end oranges_cost_l23_23537


namespace jason_current_cards_l23_23148

-- Define Jason's initial number of Pokemon cards
def jason_initial_cards : ℕ := 1342

-- Define the number of Pokemon cards Alyssa bought
def alyssa_bought_cards : ℕ := 536

-- Define the number of Pokemon cards Jason has now
def jason_final_cards (initial_cards bought_cards : ℕ) : ℕ :=
  initial_cards - bought_cards

-- Theorem statement verifying the final number of Pokemon cards Jason has
theorem jason_current_cards : jason_final_cards jason_initial_cards alyssa_bought_cards = 806 :=
by
  -- Proof goes here
  sorry

end jason_current_cards_l23_23148


namespace verify_cube_modifications_l23_23280

-- Definitions and conditions from the problem
def side_length : ℝ := 9
def initial_volume : ℝ := side_length^3
def initial_surface_area : ℝ := 6 * side_length^2

def volume_remaining : ℝ := 639
def surface_area_remaining : ℝ := 510

-- The theorem proving the volume and surface area of the remaining part after carving the cross-shaped groove
theorem verify_cube_modifications :
  initial_volume - (initial_volume - volume_remaining) = 639 ∧
  510 = surface_area_remaining :=
by
  sorry

end verify_cube_modifications_l23_23280


namespace triangle_perimeter_l23_23048

theorem triangle_perimeter : 
  ∀ (f : ℝ → ℝ), 
  (∀ x, f x = 4 * (1 - x / 3)) →
  ∃ (A B C : ℝ × ℝ), 
  A = (3, 0) ∧ 
  B = (0, 4) ∧ 
  C = (0, 0) ∧ 
  dist A B + dist B C + dist C A = 12 :=
by
  sorry

end triangle_perimeter_l23_23048


namespace stock_yield_calculation_l23_23023

theorem stock_yield_calculation (par_value market_value annual_dividend : ℝ)
  (h1 : par_value = 100)
  (h2 : market_value = 80)
  (h3 : annual_dividend = 0.04 * par_value) :
  (annual_dividend / market_value) * 100 = 5 :=
by
  sorry

end stock_yield_calculation_l23_23023


namespace probability_of_sum_17_is_correct_l23_23628

def probability_sum_17 : ℚ :=
  let favourable_outcomes := 2
  let total_outcomes := 81
  favourable_outcomes / total_outcomes

theorem probability_of_sum_17_is_correct :
  probability_sum_17 = 2 / 81 :=
by
  -- The proof steps are not required for this task
  sorry

end probability_of_sum_17_is_correct_l23_23628


namespace flour_already_put_in_l23_23864

def total_flour : ℕ := 8
def additional_flour_needed : ℕ := 6

theorem flour_already_put_in : total_flour - additional_flour_needed = 2 := by
  sorry

end flour_already_put_in_l23_23864


namespace pentagon_area_l23_23601

theorem pentagon_area 
  (side1 : ℝ) (side2 : ℝ) (side3 : ℝ) (side4 : ℝ) (side5 : ℝ)
  (h1 : side1 = 12) (h2 : side2 = 20) (h3 : side3 = 30) (h4 : side4 = 15) (h5 : side5 = 25)
  (right_angle : ∃ (a b : ℝ), a = side1 ∧ b = side5 ∧ a^2 + b^2 = (a + b)^2) : 
  ∃ (area : ℝ), area = 600 := 
  sorry

end pentagon_area_l23_23601


namespace find_some_number_l23_23606

theorem find_some_number (n m : ℕ) (h : (n / 20) * (n / m) = 1) (n_eq_40 : n = 40) : m = 2 :=
by
  sorry

end find_some_number_l23_23606


namespace alyssa_final_money_l23_23887

-- Definitions based on conditions
def weekly_allowance : Int := 8
def spent_on_movies : Int := weekly_allowance / 2
def earnings_from_washing_car : Int := 8

-- The statement to prove
def final_amount : Int := (weekly_allowance - spent_on_movies) + earnings_from_washing_car

-- The theorem expressing the problem
theorem alyssa_final_money : final_amount = 12 := by
  sorry

end alyssa_final_money_l23_23887


namespace range_of_c_for_two_distinct_roots_l23_23946

theorem range_of_c_for_two_distinct_roots (c : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 - 3 * x1 + c = x1 + 2) ∧ (x2^2 - 3 * x2 + c = x2 + 2)) ↔ (c < 6) :=
sorry

end range_of_c_for_two_distinct_roots_l23_23946


namespace range_of_m_l23_23574

-- Define the points and hyperbola condition
section ProofProblem

variables (m y₁ y₂ : ℝ)

-- Given conditions
def point_A_hyperbola : Prop := y₁ = -3 - m
def point_B_hyperbola : Prop := y₂ = (3 + m) / 2
def y1_greater_than_y2 : Prop := y₁ > y₂

-- The theorem to prove
theorem range_of_m (h1 : point_A_hyperbola m y₁) (h2 : point_B_hyperbola m y₂) (h3 : y1_greater_than_y2 y₁ y₂) : m < -3 :=
by { sorry }

end ProofProblem

end range_of_m_l23_23574


namespace sam_final_amount_l23_23569

def initial_dimes : ℕ := 9
def initial_quarters : ℕ := 5
def initial_nickels : ℕ := 3

def dad_dimes : ℕ := 7
def dad_quarters : ℕ := 2

def mom_nickels : ℕ := 1
def mom_dimes : ℕ := 2

def dime_value : ℕ := 10
def quarter_value : ℕ := 25
def nickel_value : ℕ := 5

def initial_amount : ℕ := (initial_dimes * dime_value) + (initial_quarters * quarter_value) + (initial_nickels * nickel_value)
def dad_amount : ℕ := (dad_dimes * dime_value) + (dad_quarters * quarter_value)
def mom_amount : ℕ := (mom_nickels * nickel_value) + (mom_dimes * dime_value)

def final_amount : ℕ := initial_amount + dad_amount - mom_amount

theorem sam_final_amount : final_amount = 325 := by
  sorry

end sam_final_amount_l23_23569


namespace complex_division_example_l23_23174

theorem complex_division_example (i : ℂ) (h : i^2 = -1) : 
  (2 - i) / (1 + i) = (1/2 : ℂ) - (3/2 : ℂ) * i :=
by
  -- proof would go here
  sorry

end complex_division_example_l23_23174


namespace solve_equation_l23_23059

theorem solve_equation (m n : ℝ) (h₀ : m ≠ 0) (h₁ : n ≠ 0) (h₂ : m ≠ n) :
  ∀ x : ℝ, ((x + m)^2 - 3 * (x + n)^2 = m^2 - 3 * n^2) ↔ (x = 0 ∨ x = m - 3 * n) :=
by
  sorry

end solve_equation_l23_23059


namespace lattice_points_non_visible_square_l23_23600

theorem lattice_points_non_visible_square (n : ℕ) (h : n > 0) : 
  ∃ (a b : ℤ), ∀ (x y : ℤ), a < x ∧ x < a + n ∧ b < y ∧ y < b + n → Int.gcd x y > 1 :=
sorry

end lattice_points_non_visible_square_l23_23600


namespace intersection_of_sets_l23_23047

def setA : Set ℝ := {x | x^2 < 8}
def setB : Set ℝ := {x | 1 - x ≤ 0}
def setIntersection : Set ℝ := {x | x ∈ setA ∧ x ∈ setB}

theorem intersection_of_sets :
    setIntersection = {x | 1 ≤ x ∧ x < 2 * Real.sqrt 2} :=
by
  sorry

end intersection_of_sets_l23_23047


namespace bottles_produced_l23_23988

def machine_rate (total_machines : ℕ) (total_bottles_per_minute : ℕ) : ℕ :=
  total_bottles_per_minute / total_machines

def total_bottles (total_machines : ℕ) (bottles_per_minute : ℕ) (minutes : ℕ) : ℕ :=
  total_machines * bottles_per_minute * minutes

theorem bottles_produced (machines1 machines2 minutes : ℕ) (bottles1 : ℕ) :
  machine_rate machines1 bottles1 = bottles1 / machines1 →
  total_bottles machines2 (bottles1 / machines1) minutes = 2160 :=
by
  intros machine_rate_eq
  sorry

end bottles_produced_l23_23988


namespace arithmetic_series_sum_base6_l23_23883

-- Define the terms in the arithmetic series in base 6
def a₁ := 1
def a₄₅ := 45
def n := a₄₅

-- Sum of arithmetic series in base 6
def sum_arithmetic_series := (n * (a₁ + a₄₅)) / 2

-- Expected result for the arithmetic series sum
def expected_result := 2003

theorem arithmetic_series_sum_base6 :
  sum_arithmetic_series = expected_result := by
  sorry

end arithmetic_series_sum_base6_l23_23883


namespace towel_percentage_decrease_l23_23818

theorem towel_percentage_decrease (L B : ℝ) (hL: L > 0) (hB: B > 0) :
  let OriginalArea := L * B
  let NewLength := 0.8 * L
  let NewBreadth := 0.8 * B
  let NewArea := NewLength * NewBreadth
  let PercentageDecrease := ((OriginalArea - NewArea) / OriginalArea) * 100
  PercentageDecrease = 36 :=
by
  sorry

end towel_percentage_decrease_l23_23818


namespace arithmetic_series_sum_l23_23175

theorem arithmetic_series_sum (n P q S₃n : ℕ) (h₁ : 2 * S₃n = 3 * P - q) : S₃n = 3 * P - q :=
by
  sorry

end arithmetic_series_sum_l23_23175


namespace number_of_paperback_books_l23_23440

variables (P H : ℕ)

theorem number_of_paperback_books (h1 : H = 4) (h2 : P / 3 + 2 * H = 10) : P = 6 := 
by
  sorry

end number_of_paperback_books_l23_23440


namespace study_time_l23_23239

theorem study_time (n_mcq n_fitb : ℕ) (t_mcq t_fitb : ℕ) (total_minutes_per_hour : ℕ) 
  (h1 : n_mcq = 30) (h2 : n_fitb = 30) (h3 : t_mcq = 15) (h4 : t_fitb = 25) (h5 : total_minutes_per_hour = 60) : 
  n_mcq * t_mcq + n_fitb * t_fitb = 20 * total_minutes_per_hour := 
by 
  -- This is a placeholder for the proof
  sorry

end study_time_l23_23239


namespace average_speed_of_train_l23_23551

theorem average_speed_of_train (x : ℝ) (h₀ : x > 0) :
  let time_1 := x / 40
  let time_2 := 2 * x / 20
  let total_time := time_1 + time_2
  let total_distance := 6 * x
  let avg_speed := total_distance / total_time
  avg_speed = 48 := by
  let time_1 := x / 40
  let time_2 := 2 * x / 20
  let total_time := time_1 + time_2
  let total_distance := 6 * x
  let avg_speed := total_distance / total_time
  sorry

end average_speed_of_train_l23_23551


namespace sum_of_solutions_l23_23122

def equation (x : ℝ) : Prop := (6 * x) / 30 = 8 / x

theorem sum_of_solutions : ∀ x1 x2 : ℝ, equation x1 → equation x2 → x1 + x2 = 0 := by
  sorry

end sum_of_solutions_l23_23122


namespace garden_table_bench_cost_l23_23083

theorem garden_table_bench_cost (B T : ℕ) (h1 : T + B = 750) (h2 : T = 2 * B) : B = 250 :=
by
  sorry

end garden_table_bench_cost_l23_23083


namespace no_minus_three_in_range_l23_23753

theorem no_minus_three_in_range (b : ℝ) :
  (∀ x : ℝ, x^2 + b * x + 3 ≠ -3) ↔ b^2 < 24 :=
by
  sorry

end no_minus_three_in_range_l23_23753


namespace triangle_side_count_l23_23550

theorem triangle_side_count :
  {b c : ℕ} → b ≤ 5 → 5 ≤ c → c - b < 5 → ∃ t : ℕ, t = 15 :=
by
  sorry

end triangle_side_count_l23_23550


namespace average_age_when_youngest_born_l23_23710

theorem average_age_when_youngest_born (n : ℕ) (avg_age current_y : ℕ) (total_yr : ℕ) (reduction_yr yr_older : ℕ) (avg_age_older : ℕ) 
  (h1 : n = 7)
  (h2 : avg_age = 30)
  (h3 : current_y = 7)
  (h4 : total_yr = n * avg_age)
  (h5 : reduction_yr = (n - 1) * current_y)
  (h6 : yr_older = total_yr - reduction_yr)
  (h7 : avg_age_older = yr_older / (n - 1)) :
  avg_age_older = 28 :=
by 
  sorry

end average_age_when_youngest_born_l23_23710


namespace total_businesses_l23_23411

theorem total_businesses (B : ℕ) (h1 : B / 2 + B / 3 + 12 = B) : B = 72 :=
sorry

end total_businesses_l23_23411


namespace find_general_term_find_sum_of_b_l23_23430

variables {n : ℕ} (a : ℕ → ℕ) (S : ℕ → ℕ)

-- Given conditions
axiom a5 : a 5 = 10
axiom S7 : S 7 = 56

-- Definition of S (Sum of first n terms of an arithmetic sequence)
def S_def (a : ℕ → ℕ) (n : ℕ) : ℕ := n * (a 1 + a n) / 2

-- Definition of the arithmetic sequence
def a_arith_seq (n : ℕ) : ℕ := 2 * n

-- Assuming the axiom for the arithmetic sequence sum
axiom S_is_arith : S 7 = S_def a 7

theorem find_general_term : a = a_arith_seq := 
by sorry

-- Sequence b
def b (n : ℕ) : ℕ := 2 + 9 ^ n

-- Sum of first n terms of sequence b
def T (n : ℕ) : ℕ := (Finset.range n).sum b

-- Prove T_n formula
theorem find_sum_of_b : ∀ n, T n = 2 * n + 9 / 8 * (9 ^ n - 1) :=
by sorry

end find_general_term_find_sum_of_b_l23_23430


namespace continuity_at_2_l23_23297

theorem continuity_at_2 : 
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 2| < δ → |(-3 * x^2 - 5) + 17| < ε :=
by
  sorry

end continuity_at_2_l23_23297


namespace find_a_value_l23_23519

theorem find_a_value : (15^2 * 8^3 / 256 = 450) :=
by
  sorry

end find_a_value_l23_23519


namespace ivy_baked_55_cupcakes_l23_23328

-- Definitions based on conditions
def cupcakes_morning : ℕ := 20
def cupcakes_afternoon : ℕ := cupcakes_morning + 15
def total_cupcakes : ℕ := cupcakes_morning + cupcakes_afternoon

-- Theorem statement that needs to be proved
theorem ivy_baked_55_cupcakes : total_cupcakes = 55 := by
    sorry

end ivy_baked_55_cupcakes_l23_23328


namespace max_children_l23_23878

theorem max_children (x : ℕ) (h1 : x * (x - 2) + 2 * 5 = 58) : x = 8 :=
by
  sorry

end max_children_l23_23878


namespace polynomial_roots_l23_23766

-- Problem statement: prove that the roots of the given polynomial are {-1, 3, 3}
theorem polynomial_roots : 
  (λ x => x^3 - 5 * x^2 + 3 * x + 9) = (λ x => (x + 1) * (x - 3) ^ 2) :=
by
  sorry

end polynomial_roots_l23_23766


namespace part_a_part_b_l23_23088

theorem part_a (x y : ℂ) : (3 * y + 5 * x * Complex.I = 15 - 7 * Complex.I) ↔ (x = -7/5 ∧ y = 5) := by
  sorry

theorem part_b (x y : ℝ) : (2 * x + 3 * y + (x - y) * Complex.I = 7 + 6 * Complex.I) ↔ (x = 5 ∧ y = -1) := by
  sorry

end part_a_part_b_l23_23088


namespace equation_1_solution_set_equation_2_solution_set_l23_23748

open Real

theorem equation_1_solution_set (x : ℝ) : x^2 - 4 * x - 8 = 0 ↔ (x = 2 * sqrt 3 + 2 ∨ x = -2 * sqrt 3 + 2) :=
by sorry

theorem equation_2_solution_set (x : ℝ) : 3 * x - 6 = x * (x - 2) ↔ (x = 2 ∨ x = 3) :=
by sorry

end equation_1_solution_set_equation_2_solution_set_l23_23748


namespace sequence_becomes_negative_from_8th_term_l23_23333

def seq (n : ℕ) : ℤ := 21 + 4 * n - n ^ 2

theorem sequence_becomes_negative_from_8th_term :
  ∀ n, n ≥ 8 ↔ seq n < 0 :=
by
  -- proof goes here
  sorry

end sequence_becomes_negative_from_8th_term_l23_23333


namespace number_of_rooms_l23_23299

theorem number_of_rooms (x : ℕ) (h1 : ∀ n, 6 * (n - 1) = 5 * n + 4) : x = 10 :=
sorry

end number_of_rooms_l23_23299


namespace cosine_sum_identity_l23_23925

theorem cosine_sum_identity 
  (α : ℝ) 
  (h_sin : Real.sin α = 3 / 5) 
  (h_alpha_first_quad : 0 < α ∧ α < Real.pi / 2) : 
  Real.cos (Real.pi / 3 + α) = (4 - 3 * Real.sqrt 3) / 10 :=
by
  sorry

end cosine_sum_identity_l23_23925


namespace total_students_l23_23941

theorem total_students (boys girls : ℕ) (h_ratio : boys / girls = 8 / 5) (h_girls : girls = 120) : boys + girls = 312 :=
by
  sorry

end total_students_l23_23941


namespace find_intersection_points_l23_23448

def intersection_points (t α : ℝ) : Prop :=
∃ t α : ℝ,
  (2 + t, -1 - t) = (3 * Real.cos α, 3 * Real.sin α) ∧
  ((2 + t = (1 + Real.sqrt 17) / 2 ∧ -1 - t = (1 - Real.sqrt 17) / 2) ∨
   (2 + t = (1 - Real.sqrt 17) / 2 ∧ -1 - t = (1 + Real.sqrt 17) / 2))

theorem find_intersection_points : intersection_points t α :=
sorry

end find_intersection_points_l23_23448


namespace average_stamps_collected_per_day_l23_23428

open Nat

-- Define an arithmetic sequence
def arithmetic_sequence (a d : ℕ) (n : ℕ) : ℕ :=
  a + d * (n - 1)

-- Define the sum of the first n terms of an arithmetic sequence
def sum_arithmetic_sequence (a d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

-- Given conditions
def a := 10
def d := 10
def n := 7

-- Prove that the average number of stamps collected over 7 days is 40
theorem average_stamps_collected_per_day : 
  sum_arithmetic_sequence a d n / n = 40 := 
by
  sorry

end average_stamps_collected_per_day_l23_23428


namespace alpha_in_fourth_quadrant_l23_23275

def point_in_third_quadrant (α : ℝ) : Prop :=
  (Real.tan α < 0) ∧ (Real.sin α < 0)

theorem alpha_in_fourth_quadrant (α : ℝ) (h : point_in_third_quadrant α) : 
  α ∈ Set.Ioc (3 * Real.pi / 2) (2 * Real.pi) :=
by sorry

end alpha_in_fourth_quadrant_l23_23275


namespace fraction_of_menu_vegan_soy_free_l23_23161

def num_vegan_dishes : Nat := 6
def fraction_menu_vegan : ℚ := 1 / 4
def num_vegan_dishes_with_soy : Nat := 4

def num_vegan_soy_free_dishes : Nat := num_vegan_dishes - num_vegan_dishes_with_soy
def fraction_vegan_soy_free : ℚ := num_vegan_soy_free_dishes / num_vegan_dishes
def fraction_menu_vegan_soy_free : ℚ := fraction_vegan_soy_free * fraction_menu_vegan

theorem fraction_of_menu_vegan_soy_free :
  fraction_menu_vegan_soy_free = 1 / 12 := by
  sorry

end fraction_of_menu_vegan_soy_free_l23_23161


namespace algebraic_expression_value_l23_23836

theorem algebraic_expression_value (x : ℝ) (h : x^2 - 2 * x - 1 = 0) : x^3 - x^2 - 3 * x + 2 = 3 := 
by
  sorry

end algebraic_expression_value_l23_23836


namespace division_problem_l23_23467

theorem division_problem :
  0.045 / 0.0075 = 6 :=
sorry

end division_problem_l23_23467


namespace sin_cos_identity_l23_23644

theorem sin_cos_identity (α β γ : ℝ) (h : α + β + γ = 180) :
    Real.sin α + Real.sin β + Real.sin γ = 
    4 * Real.cos (α / 2) * Real.cos (β / 2) * Real.cos (γ / 2) := 
  sorry

end sin_cos_identity_l23_23644


namespace transfer_equation_correct_l23_23770

theorem transfer_equation_correct (x : ℕ) :
  46 + x = 3 * (30 - x) := 
sorry

end transfer_equation_correct_l23_23770


namespace gold_coins_l23_23212

theorem gold_coins (c n : ℕ) 
  (h₁ : n = 8 * (c - 1))
  (h₂ : n = 5 * c + 4) :
  n = 24 :=
by
  sorry

end gold_coins_l23_23212


namespace students_suggested_pasta_l23_23744

-- Define the conditions as variables in Lean
variable (total_students : ℕ := 470)
variable (suggested_mashed_potatoes : ℕ := 230)
variable (suggested_bacon : ℕ := 140)

-- The problem statement to prove
theorem students_suggested_pasta : 
  total_students - (suggested_mashed_potatoes + suggested_bacon) = 100 := by
  sorry

end students_suggested_pasta_l23_23744


namespace karen_start_time_late_l23_23717

theorem karen_start_time_late
  (karen_speed : ℝ := 60) -- Karen drives at 60 mph
  (tom_speed : ℝ := 45) -- Tom drives at 45 mph
  (tom_distance : ℝ := 24) -- Tom drives 24 miles before Karen wins
  (karen_lead : ℝ := 4) -- Karen needs to beat Tom by 4 miles
  : (60 * (24 / 45) - 60 * (28 / 60)) * 60 = 4 := by
  sorry

end karen_start_time_late_l23_23717


namespace sequence_add_l23_23289

theorem sequence_add (x y : ℝ) (h1 : x = 81 * (1 / 3)) (h2 : y = x * (1 / 3)) : x + y = 36 :=
sorry

end sequence_add_l23_23289


namespace hard_candy_food_coloring_l23_23589

theorem hard_candy_food_coloring
  (lollipop_coloring : ℕ) (hard_candy_coloring : ℕ)
  (num_lollipops : ℕ) (num_hardcandies : ℕ)
  (total_coloring : ℕ)
  (H1 : lollipop_coloring = 8)
  (H2 : num_lollipops = 150)
  (H3 : num_hardcandies = 20)
  (H4 : total_coloring = 1800) :
  (20 * hard_candy_coloring + 150 * lollipop_coloring = total_coloring) → 
  hard_candy_coloring = 30 :=
by
  sorry

end hard_candy_food_coloring_l23_23589


namespace curveG_symmetric_curveG_unique_common_point_curveG_common_points_x_y_curveG_max_distance_l23_23118

-- Define the curve G as a set of points (x, y) satisfying the equation x^3 + y^3 - 6xy = 0
def curveG (x y : ℝ) : Prop :=
  x^3 + y^3 - 6 * x * y = 0

-- Prove symmetry of curveG with respect to the line y = x
theorem curveG_symmetric (x y : ℝ) (h : curveG x y) : curveG y x :=
  sorry

-- Prove unique common point with the line x + y - 6 = 0
theorem curveG_unique_common_point : ∃! p : ℝ × ℝ, curveG p.1 p.2 ∧ p.1 + p.2 = 6 :=
  sorry

-- Prove curveG has at least one common point with the line x - y + 1 = 0
theorem curveG_common_points_x_y : ∃ p : ℝ × ℝ, curveG p.1 p.2 ∧ p.1 - p.2 + 1 = 0 :=
  sorry

-- Prove the maximum distance from any point on the curveG to the origin is 3√2
theorem curveG_max_distance : ∀ p : ℝ × ℝ, curveG p.1 p.2 → p.1 > 0 → p.2 > 0 → (p.1^2 + p.2^2 ≤ 18) :=
  sorry

end curveG_symmetric_curveG_unique_common_point_curveG_common_points_x_y_curveG_max_distance_l23_23118


namespace total_toothpicks_for_grid_l23_23015

-- Defining the conditions
def grid_height := 30
def grid_width := 15

-- Define the function that calculates the total number of toothpicks
def total_toothpicks (height : ℕ) (width : ℕ) : ℕ :=
  let horizontal_toothpicks := (height + 1) * width
  let vertical_toothpicks := (width + 1) * height
  horizontal_toothpicks + vertical_toothpicks

-- The theorem stating the problem and its answer
theorem total_toothpicks_for_grid : total_toothpicks grid_height grid_width = 945 :=
by {
  -- Here we would write the proof steps. Using sorry for now.
  sorry
}

end total_toothpicks_for_grid_l23_23015


namespace probability_event_a_without_replacement_independence_of_events_with_replacement_l23_23508

open ProbabilityTheory MeasureTheory Set

-- Definitions corresponding to the conditions
def BallLabeled (i : ℕ) : Prop := i ∈ Finset.range 10

def EventA (second_ball : ℕ) : Prop := second_ball = 2

def EventB (first_ball second_ball : ℕ) (m : ℕ) : Prop := first_ball + second_ball = m

-- First Part: Probability without replacement
theorem probability_event_a_without_replacement :
  ∃ P_A : ℝ, P_A = 1 / 10 := sorry

-- Second Part: Independence with replacement
theorem independence_of_events_with_replacement (m : ℕ) :
  (EventA 2 → (∀ first_ball : ℕ, BallLabeled first_ball → EventB first_ball 2 m) ↔ m = 9) := sorry

end probability_event_a_without_replacement_independence_of_events_with_replacement_l23_23508


namespace range_of_m_l23_23793

theorem range_of_m (x y m : ℝ) (h1 : x - y = 2 * m + 7) (h2 : x + y = 4 * m - 3) 
  (h3 : x < 0) (h4 : y < 0) : m < -2 / 3 := 
by 
  sorry

end range_of_m_l23_23793


namespace haley_tv_total_hours_l23_23200

theorem haley_tv_total_hours (h_sat : Nat) (h_sun : Nat) (H_sat : h_sat = 6) (H_sun : h_sun = 3) :
  h_sat + h_sun = 9 := by
  sorry

end haley_tv_total_hours_l23_23200


namespace calculate_total_people_l23_23802

-- Definitions given in the problem
def cost_per_adult_meal := 3
def num_kids := 7
def total_cost := 15

-- The target property to prove
theorem calculate_total_people : 
  (total_cost / cost_per_adult_meal) + num_kids = 12 := 
by 
  sorry

end calculate_total_people_l23_23802


namespace solve_equation_l23_23846

theorem solve_equation :
  ∀ x : ℝ, 18 / (x^2 - 9) - 3 / (x - 3) = 2 ↔ (x = 4.5 ∨ x = -3) :=
by
  sorry

end solve_equation_l23_23846


namespace crossing_time_correct_l23_23281

def length_of_train : ℝ := 150 -- Length of the train in meters
def speed_of_train_km_per_hr : ℝ := 72 -- Speed of the train in km/hr
def length_of_bridge : ℝ := 132 -- Length of the bridge in meters

noncomputable def speed_of_train_m_per_s : ℝ := (speed_of_train_km_per_hr * 1000) / 3600 -- Speed of the train in m/s

noncomputable def time_to_cross_bridge : ℝ := (length_of_train + length_of_bridge) / speed_of_train_m_per_s -- Time in seconds

theorem crossing_time_correct : time_to_cross_bridge = 14.1 := by
  sorry

end crossing_time_correct_l23_23281


namespace count_solutions_g_composition_eq_l23_23012

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := 3 * Real.cos (Real.pi * x)

-- Define the main theorem
theorem count_solutions_g_composition_eq :
  ∃ (s : Finset ℝ), s.card = 7 ∧ ∀ x ∈ s, -1.5 ≤ x ∧ x ≤ 1.5 ∧ g (g (g x)) = g x :=
by
  sorry

end count_solutions_g_composition_eq_l23_23012


namespace tan_of_alpha_l23_23691

theorem tan_of_alpha 
  (α : ℝ)
  (h1 : Real.sin α = (3 / 5))
  (h2 : α ∈ Set.Ioo (π / 2) π) : Real.tan α = -3 / 4 :=
sorry

end tan_of_alpha_l23_23691


namespace find_p_q_r_divisibility_l23_23398

theorem find_p_q_r_divisibility 
  (p q r : ℝ)
  (h_div : ∀ x, (x^4 + 4*x^3 + 6*p*x^2 + 4*q*x + r) % (x^3 + 3*x^2 + 9*x + 3) = 0)
  : (p + q) * r = 15 :=
by
  -- Proof steps would go here
  sorry

end find_p_q_r_divisibility_l23_23398


namespace factorize_expression_l23_23136

theorem factorize_expression (y a : ℝ) : 
  3 * y * a ^ 2 - 6 * y * a + 3 * y = 3 * y * (a - 1) ^ 2 :=
by
  sorry

end factorize_expression_l23_23136


namespace billy_has_62_crayons_l23_23096

noncomputable def billy_crayons (total_crayons : ℝ) (jane_crayons : ℝ) : ℝ :=
  total_crayons - jane_crayons

theorem billy_has_62_crayons : billy_crayons 114 52.0 = 62 := by
  sorry

end billy_has_62_crayons_l23_23096


namespace gcd_90_270_l23_23764

theorem gcd_90_270 : Int.gcd 90 270 = 90 :=
by
  sorry

end gcd_90_270_l23_23764


namespace percentage_of_b_l23_23892

variable (a b c p : ℝ)

theorem percentage_of_b :
  (0.04 * a = 8) →
  (p * b = 4) →
  (c = b / a) →
  p = 1 / (50 * c) :=
by
  sorry

end percentage_of_b_l23_23892


namespace percent_defective_units_shipped_l23_23696

theorem percent_defective_units_shipped :
  let total_units_defective := 6 / 100
  let defective_units_shipped := 4 / 100
  let percent_defective_units_shipped := (total_units_defective * defective_units_shipped) * 100
  percent_defective_units_shipped = 0.24 := by
  sorry

end percent_defective_units_shipped_l23_23696


namespace find_k_l23_23965

-- Definitions for the given conditions
def slope_of_first_line : ℝ := 2
def alpha : ℝ := slope_of_first_line
def slope_of_second_line : ℝ := 2 * alpha

-- The proof goal
theorem find_k (k : ℝ) : slope_of_second_line = k ↔ k = 4 := by
  sorry

end find_k_l23_23965


namespace denomination_calculation_l23_23820

variables (total_money rs_50_count total_count rs_50_value remaining_count remaining_amount remaining_denomination_value : ℕ)

theorem denomination_calculation 
  (h1 : total_money = 10350)
  (h2 : rs_50_count = 97)
  (h3 : total_count = 108)
  (h4 : rs_50_value = 50)
  (h5 : remaining_count = total_count - rs_50_count)
  (h6 : remaining_amount = total_money - rs_50_count * rs_50_value)
  (h7 : remaining_denomination_value = remaining_amount / remaining_count) :
  remaining_denomination_value = 500 := 
sorry

end denomination_calculation_l23_23820


namespace binom_2000_3_eq_l23_23111

theorem binom_2000_3_eq : Nat.choose 2000 3 = 1331000333 := by
  sorry

end binom_2000_3_eq_l23_23111


namespace pipe_B_fill_time_l23_23827

variable (A B C : ℝ)
variable (fill_time : ℝ := 16)
variable (total_tank : ℝ := 1)

-- Conditions
axiom condition1 : A + B + C = (1 / fill_time)
axiom condition2 : A = 2 * B
axiom condition3 : B = 2 * C

-- Prove that B alone will take 56 hours to fill the tank
theorem pipe_B_fill_time : B = (1 / 56) :=
by sorry

end pipe_B_fill_time_l23_23827


namespace general_term_sequence_l23_23577

noncomputable def a (t : ℝ) (n : ℕ) : ℝ :=
if h : t ≠ 1 then (2 * (t^n - 1) / n) - 1 else 0

theorem general_term_sequence (t : ℝ) (n : ℕ) (hn : n ≠ 0) (h : t ≠ 1) :
  a t (n+1) = (2 * (t^(n+1) - 1) / (n+1)) - 1 := 
sorry

end general_term_sequence_l23_23577


namespace crayons_divided_equally_l23_23804

theorem crayons_divided_equally (total_crayons : ℕ) (number_of_people : ℕ) (crayons_per_person : ℕ) 
  (h1 : total_crayons = 24) (h2 : number_of_people = 3) : 
  crayons_per_person = total_crayons / number_of_people → crayons_per_person = 8 :=
by
  intro h
  rw [h1, h2] at h
  have : 24 / 3 = 8 := by norm_num
  rw [this] at h
  exact h

end crayons_divided_equally_l23_23804


namespace profit_percentage_is_correct_l23_23195

noncomputable def cost_price (SP : ℝ) : ℝ := 0.81 * SP

noncomputable def profit (SP CP : ℝ) : ℝ := SP - CP

noncomputable def profit_percentage (profit CP : ℝ) : ℝ := (profit / CP) * 100

theorem profit_percentage_is_correct (SP : ℝ) (h : SP = 100) :
  profit_percentage (profit SP (cost_price SP)) (cost_price SP) = 23.46 :=
by
  sorry

end profit_percentage_is_correct_l23_23195


namespace line_slope_angle_y_intercept_l23_23719

theorem line_slope_angle_y_intercept :
  ∀ (x y : ℝ), x - y - 1 = 0 → 
    (∃ k b : ℝ, y = x - 1 ∧ k = 1 ∧ b = -1 ∧ θ = 45 ∧ θ = Real.arctan k) := 
    by
      sorry

end line_slope_angle_y_intercept_l23_23719


namespace minimum_value_of_f_on_interval_l23_23147

noncomputable def f (x : ℝ) : ℝ := - (1 / 2) * x^2 + Real.log x

theorem minimum_value_of_f_on_interval :
  (∀ x ∈ (Set.Icc (1 / Real.exp 1) (Real.exp 1)), f x ≥ f (Real.exp 1)) ∧
  ∃ x ∈ (Set.Icc (1 / Real.exp 1) (Real.exp 1)), f x = f (Real.exp 1) := 
by
  sorry

end minimum_value_of_f_on_interval_l23_23147


namespace page_number_counted_twice_l23_23716

theorem page_number_counted_twice {n x : ℕ} (h₁ : n = 70) (h₂ : x > 0) (h₃ : x ≤ n) (h₄ : 2550 = n * (n + 1) / 2 + x) : x = 65 :=
by {
  sorry
}

end page_number_counted_twice_l23_23716


namespace determine_l_l23_23674

theorem determine_l :
  ∃ l : ℤ, (2^2000 - 2^1999 - 3 * 2^1998 + 2^1997 = l * 2^1997) ∧ l = -1 :=
by
  sorry

end determine_l_l23_23674


namespace bus_stops_for_45_minutes_per_hour_l23_23747

-- Define the conditions
def speed_excluding_stoppages : ℝ := 48 -- in km/hr
def speed_including_stoppages : ℝ := 12 -- in km/hr

-- Define the statement to be proven
theorem bus_stops_for_45_minutes_per_hour :
  let speed_reduction := speed_excluding_stoppages - speed_including_stoppages
  let time_stopped : ℝ := (speed_reduction / speed_excluding_stoppages) * 60
  time_stopped = 45 :=
by
  sorry

end bus_stops_for_45_minutes_per_hour_l23_23747


namespace robins_hair_cut_l23_23924

theorem robins_hair_cut (x : ℕ) : 16 - x + 12 = 17 → x = 11 := by
  sorry

end robins_hair_cut_l23_23924


namespace value_of_fraction_l23_23391

theorem value_of_fraction (x y z w : ℝ) 
  (h1 : x = 4 * y) 
  (h2 : y = 3 * z) 
  (h3 : z = 5 * w) : 
  (x * z) / (y * w) = 20 := 
by
  sorry

end value_of_fraction_l23_23391


namespace kyle_and_miles_total_marble_count_l23_23173

noncomputable def kyle_marble_count (F : ℕ) (K : ℕ) : Prop :=
  F = 4 * K

noncomputable def miles_marble_count (F : ℕ) (M : ℕ) : Prop :=
  F = 9 * M

theorem kyle_and_miles_total_marble_count :
  ∀ (F K M : ℕ), F = 36 → kyle_marble_count F K → miles_marble_count F M → K + M = 13 :=
by
  intros F K M hF hK hM
  sorry

end kyle_and_miles_total_marble_count_l23_23173


namespace hyperbola_range_k_l23_23218

theorem hyperbola_range_k (k : ℝ) : 
  (1 < k ∧ k < 3) ↔ (∃ x y : ℝ, (3 - k > 0) ∧ (k - 1 > 0) ∧ (x * x) / (3 - k) - (y * y) / (k - 1) = 1) :=
by {
  sorry
}

end hyperbola_range_k_l23_23218


namespace hot_dog_cost_l23_23124

variables (h d : ℝ)

theorem hot_dog_cost :
  (3 * h + 4 * d = 10) →
  (2 * h + 3 * d = 7) →
  d = 1 :=
by
  intros h_eq d_eq
  -- Proof skipped
  sorry

end hot_dog_cost_l23_23124


namespace medicine_dosage_per_kg_l23_23847

theorem medicine_dosage_per_kg :
  ∀ (child_weight parts dose_per_part total_dose dose_per_kg : ℕ),
    (child_weight = 30) →
    (parts = 3) →
    (dose_per_part = 50) →
    (total_dose = parts * dose_per_part) →
    (dose_per_kg = total_dose / child_weight) →
    dose_per_kg = 5 :=
by
  intros child_weight parts dose_per_part total_dose dose_per_kg
  intros h1 h2 h3 h4 h5
  sorry

end medicine_dosage_per_kg_l23_23847


namespace daniel_age_l23_23974

def isAgeSet (s : Set ℕ) : Prop :=
  s = {4, 6, 8, 10, 12, 14}

def sumTo18 (s : Set ℕ) : Prop :=
  ∃ (a b : ℕ), a ∈ s ∧ b ∈ s ∧ a + b = 18 ∧ a ≠ b

def youngerThan11 (s : Set ℕ) : Prop :=
  ∀ (a : ℕ), a ∈ s → a < 11

def staysHome (DanielAge : ℕ) (s : Set ℕ) : Prop :=
  6 ∈ s ∧ DanielAge ∈ s

theorem daniel_age :
  ∀ (ages : Set ℕ) (DanielAge : ℕ),
    isAgeSet ages →
    (∃ s, sumTo18 s ∧ s ⊆ ages) →
    (∃ s, youngerThan11 s ∧ s ⊆ ages ∧ 6 ∉ s) →
    staysHome DanielAge ages →
    DanielAge = 12 :=
by
  intros ages DanielAge isAgeSetAges sumTo18Ages youngerThan11Ages staysHomeDaniel
  sorry

end daniel_age_l23_23974


namespace brick_height_calc_l23_23235

theorem brick_height_calc 
  (length_wall : ℝ) (height_wall : ℝ) (width_wall : ℝ) 
  (num_bricks : ℕ) 
  (length_brick : ℝ) (width_brick : ℝ) 
  (H : ℝ) 
  (volume_wall : ℝ) 
  (volume_brick : ℝ)
  (condition1 : length_wall = 800) 
  (condition2 : height_wall = 600) 
  (condition3 : width_wall = 22.5)
  (condition4 : num_bricks = 3200) 
  (condition5 : length_brick = 50) 
  (condition6 : width_brick = 11.25) 
  (condition7 : volume_wall = length_wall * height_wall * width_wall) 
  (condition8 : volume_brick = length_brick * width_brick * H) 
  (condition9 : num_bricks * volume_brick = volume_wall) 
  : H = 6 := 
by
  sorry

end brick_height_calc_l23_23235


namespace smallest_positive_integer_n_l23_23624

theorem smallest_positive_integer_n :
  ∃ n : ℕ, (n > 0 ∧ 17 * n % 7 = 2) ∧ ∀ m : ℕ, (m > 0 ∧ 17 * m % 7 = 2) → n ≤ m := 
sorry

end smallest_positive_integer_n_l23_23624


namespace total_charge_rush_hour_trip_l23_23217

def initial_fee : ℝ := 2.35
def non_rush_hour_cost_per_two_fifths_mile : ℝ := 0.35
def rush_hour_cost_increase_percentage : ℝ := 0.20
def traffic_delay_cost_per_mile : ℝ := 1.50
def distance_travelled : ℝ := 3.6

theorem total_charge_rush_hour_trip (initial_fee : ℝ) 
  (non_rush_hour_cost_per_two_fifths_mile : ℝ) 
  (rush_hour_cost_increase_percentage : ℝ)
  (traffic_delay_cost_per_mile : ℝ)
  (distance_travelled : ℝ) : 
  initial_fee = 2.35 → 
  non_rush_hour_cost_per_two_fifths_mile = 0.35 →
  rush_hour_cost_increase_percentage = 0.20 →
  traffic_delay_cost_per_mile = 1.50 →
  distance_travelled = 3.6 →
  (initial_fee + ((5/2) * (non_rush_hour_cost_per_two_fifths_mile * (1 + rush_hour_cost_increase_percentage))) * distance_travelled + (traffic_delay_cost_per_mile * distance_travelled)) = 11.53 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end total_charge_rush_hour_trip_l23_23217


namespace trigonometric_comparison_l23_23444

noncomputable def a : ℝ := Real.sin (3 * Real.pi / 5)
noncomputable def b : ℝ := Real.cos (2 * Real.pi / 5)
noncomputable def c : ℝ := Real.tan (2 * Real.pi / 5)

theorem trigonometric_comparison :
  b < a ∧ a < c :=
by {
  -- Use necessary steps to demonstrate b < a and a < c
  sorry
}

end trigonometric_comparison_l23_23444


namespace greatest_value_x_l23_23255

theorem greatest_value_x (x : ℕ) (h : lcm (lcm x 12) 18 = 108) : x ≤ 108 := sorry

end greatest_value_x_l23_23255


namespace mike_gave_pens_l23_23292

theorem mike_gave_pens (M : ℕ) 
  (initial_pens : ℕ := 5) 
  (pens_after_mike : ℕ := initial_pens + M)
  (pens_after_cindy : ℕ := 2 * pens_after_mike)
  (pens_after_sharon : ℕ := pens_after_cindy - 10)
  (final_pens : ℕ := 40) : 
  pens_after_sharon = final_pens → M = 20 := 
by 
  sorry

end mike_gave_pens_l23_23292


namespace pave_square_with_tiles_l23_23615

theorem pave_square_with_tiles (b c : ℕ) (h_right_triangle : (b > 0) ∧ (c > 0)) :
  (∃ (k : ℕ), k^2 = b^2 + c^2) ↔ (∃ (m n : ℕ), m * c * b = 2 * n^2 * (b^2 + c^2)) := 
sorry

end pave_square_with_tiles_l23_23615


namespace jello_mix_needed_per_pound_l23_23084

variable (bathtub_volume : ℝ) (gallons_per_cubic_foot : ℝ) 
          (pounds_per_gallon : ℝ) (cost_per_tablespoon : ℝ) 
          (total_cost : ℝ)

theorem jello_mix_needed_per_pound :
  bathtub_volume = 6 ∧
  gallons_per_cubic_foot = 7.5 ∧
  pounds_per_gallon = 8 ∧
  cost_per_tablespoon = 0.50 ∧
  total_cost = 270 →
  (total_cost / cost_per_tablespoon) / 
  (bathtub_volume * gallons_per_cubic_foot * pounds_per_gallon) = 1.5 :=
by
  sorry

end jello_mix_needed_per_pound_l23_23084


namespace geometric_sequence_sum_l23_23043

theorem geometric_sequence_sum 
  (a : ℕ → ℝ) 
  (h1 : a 1 + a 2 + a 3 = 7) 
  (h2 : a 2 + a 3 + a 4 = 14) 
  (geom_seq : ∃ q, ∀ n, a (n + 1) = q * a n ∧ q = 2) :
  a 4 + a 5 + a 6 = 56 := 
by
  sorry

end geometric_sequence_sum_l23_23043


namespace center_of_circle_polar_coords_l23_23435

theorem center_of_circle_polar_coords :
  ∀ (θ : ℝ), ∃ (ρ : ℝ), (ρ, θ) = (2, Real.pi) ∧ ρ = - 4 * Real.cos θ := 
sorry

end center_of_circle_polar_coords_l23_23435


namespace detail_understanding_word_meaning_guessing_logical_reasoning_l23_23383

-- Detail Understanding Question
theorem detail_understanding (sentence: String) (s: ∀ x : String, x ∈ ["He hardly watered his new trees,..."] → x = sentence) :
  sentence = "He hardly watered his new trees,..." :=
sorry

-- Word Meaning Guessing Question
theorem word_meaning_guessing (adversity_meaning: String) (meanings: ∀ y : String, y ∈ ["adversity means misfortune or disaster", "lack of water", "sufficient care/attention", "bad weather"] → y = adversity_meaning) :
  adversity_meaning = "adversity means misfortune or disaster" :=
sorry

-- Logical Reasoning Question
theorem logical_reasoning (hope: String) (sentences: ∀ z : String, z ∈ ["The author hopes his sons can withstand the tests of wind and rain in their life journey"] → z = hope) :
  hope = "The author hopes his sons can withstand the tests of wind and rain in their life journey" :=
sorry

end detail_understanding_word_meaning_guessing_logical_reasoning_l23_23383


namespace base_of_isosceles_triangle_l23_23180

theorem base_of_isosceles_triangle (a b c : ℝ) 
  (h₁ : 3 * a = 45) 
  (h₂ : 2 * b + c = 40) 
  (h₃ : b = a ∨ b = a) : c = 10 := 
sorry

end base_of_isosceles_triangle_l23_23180


namespace Nickel_ate_3_chocolates_l23_23462

-- Definitions of the conditions
def Robert_chocolates : ℕ := 12
def extra_chocolates : ℕ := 9
def Nickel_chocolates (N : ℕ) : Prop := Robert_chocolates = N + extra_chocolates

-- The proof goal
theorem Nickel_ate_3_chocolates : ∃ N : ℕ, Nickel_chocolates N ∧ N = 3 :=
by
  sorry

end Nickel_ate_3_chocolates_l23_23462


namespace part1_part2_l23_23057

def U : Set ℝ := {x : ℝ | True}

def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 5}

def B (m : ℝ) : Set ℝ := {x : ℝ | m + 1 ≤ x ∧ x ≤ 2 * m - 1}

-- Part 1: Prove the range of m when 4 ∈ B(m) is [5/2, 3]
theorem part1 (m : ℝ) : (4 ∈ B m) → (5/2 ≤ m ∧ m ≤ 3) := by
  sorry

-- Part 2: Prove the range of m when x ∈ A is a necessary but not sufficient condition for x ∈ B(m) 
theorem part2 (m : ℝ) : (∀ x, x ∈ B m → x ∈ A) ∧ ¬(∀ x, x ∈ A → x ∈ B m) → (m ≤ 3) := by
  sorry

end part1_part2_l23_23057


namespace loss_percentage_is_75_l23_23429

-- Given conditions
def cost_price_one_book (C : ℝ) : Prop := C > 0
def selling_price_one_book (S : ℝ) : Prop := S > 0
def cost_price_5_equals_selling_price_20 (C S : ℝ) : Prop := 5 * C = 20 * S

-- Proof goal
theorem loss_percentage_is_75 (C S : ℝ) (h1 : cost_price_one_book C) (h2 : selling_price_one_book S) (h3 : cost_price_5_equals_selling_price_20 C S) : 
  ((C - S) / C) * 100 = 75 :=
by
  sorry

end loss_percentage_is_75_l23_23429


namespace equation_represents_circle_of_radius_8_l23_23639

theorem equation_represents_circle_of_radius_8 (k : ℝ) : 
  (x^2 + 14 * x + y^2 + 8 * y - k = 0) → k = -1 ↔ (∃ r, r = 8 ∧ (x + 7)^2 + (y + 4)^2 = r^2) :=
by
  sorry

end equation_represents_circle_of_radius_8_l23_23639


namespace S_equals_x4_l23_23464

-- Define the expression for S
def S (x : ℝ) : ℝ := (x - 1)^4 + 4 * (x - 1)^3 + 6 * (x - 1)^2 + 4 * x - 3

-- State the theorem to be proved
theorem S_equals_x4 (x : ℝ) : S x = x^4 :=
by
  sorry

end S_equals_x4_l23_23464


namespace integer_solutions_l23_23510

theorem integer_solutions :
  { (x, y) : ℤ × ℤ | x^2 = 1 + 4 * y^3 * (y + 2) } = {(1, 0), (1, -2), (-1, 0), (-1, -2)} :=
by
  sorry

end integer_solutions_l23_23510


namespace quadrilateral_is_parallelogram_l23_23585

theorem quadrilateral_is_parallelogram 
  (a b c d : ℝ)
  (h : a^2 + b^2 + c^2 + d^2 - 2*a*c - 2*b*d = 0) 
  : (a = c ∧ b = d) → parallelogram :=
by
  sorry

end quadrilateral_is_parallelogram_l23_23585


namespace percentage_of_women_lawyers_l23_23105

theorem percentage_of_women_lawyers
  (T : ℝ) 
  (h1 : 0.70 * T = W) 
  (h2 : 0.28 * T = WL) : 
  ((WL / W) * 100 = 40) :=
by
  sorry

end percentage_of_women_lawyers_l23_23105


namespace quadratic_has_real_roots_find_pos_m_l23_23768

-- Proof problem 1:
theorem quadratic_has_real_roots (m : ℝ) : ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ x^2 - 4 * m * x + 3 * m^2 = 0 :=
by
  sorry

-- Proof problem 2:
theorem find_pos_m (m x1 x2 : ℝ) (hm : x1 > x2) (h_diff : x1 - x2 = 2)
  (h_roots : ∀ m, (x^2 - 4*m*x + 3*m^2 = 0)) : m = 1 :=
by
  sorry

end quadratic_has_real_roots_find_pos_m_l23_23768


namespace calculate_f_2015_l23_23549

noncomputable def f : ℝ → ℝ := sorry

-- Define the odd function property
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Define the periodic function property with period 4
def periodic_4 (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 4) = f x

-- Define the given condition for the interval (0, 2)
def interval_condition (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, 0 < x ∧ x < 2 → f x = 2 * x ^ 2

theorem calculate_f_2015
  (odd_f : odd_function f)
  (periodic_f : periodic_4 f)
  (interval_f : interval_condition f) :
  f 2015 = -2 :=
sorry

end calculate_f_2015_l23_23549


namespace point_P_on_number_line_l23_23367

variable (A : ℝ) (B : ℝ) (P : ℝ)

theorem point_P_on_number_line (hA : A = -1) (hB : B = 5) (hDist : abs (P - A) = abs (B - P)) : P = 2 := 
sorry

end point_P_on_number_line_l23_23367


namespace distinct_sum_of_five_integers_l23_23885

theorem distinct_sum_of_five_integers 
  (a b c d e : ℤ) 
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e) 
  (h_condition : (7 - a) * (7 - b) * (7 - c) * (7 - d) * (7 - e) = -120) : 
  a + b + c + d + e = 25 :=
sorry

end distinct_sum_of_five_integers_l23_23885


namespace solution_set_inequality_range_of_m_l23_23609

def f (x : ℝ) (m : ℝ) : ℝ := m - |x - 1| - |x + 1|

-- Problem 1
theorem solution_set_inequality (x : ℝ) : 
  (f x 5 > 2) ↔ (-3 / 2 < x ∧ x < 3 / 2) :=
sorry

-- Problem 2
theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, y = (x^2 + 2 * x + 3) ∧ y = f x m) ↔ (m ≥ 4) :=
sorry

end solution_set_inequality_range_of_m_l23_23609


namespace painting_perimeter_l23_23344

-- Definitions for the problem conditions
def frame_thickness : ℕ := 3
def frame_area : ℕ := 108

-- Declaration that expresses the given conditions and the problem's conclusion
theorem painting_perimeter {w h : ℕ} (h_frame : (w + 2 * frame_thickness) * (h + 2 * frame_thickness) - w * h = frame_area) :
  2 * (w + h) = 24 :=
by
  sorry

end painting_perimeter_l23_23344


namespace exponents_problem_l23_23368

theorem exponents_problem :
  5000 * (5000^9) * 2^(1000) = 5000^(10) * 2^(1000) := by sorry

end exponents_problem_l23_23368


namespace payment_to_N_l23_23501

variable (x : ℝ)

/-- Conditions stating the total payment and the relationship between M and N's payment --/
axiom total_payment : x + 1.20 * x = 550

/-- Statement to prove the amount paid to N per week --/
theorem payment_to_N : x = 250 :=
by
  sorry

end payment_to_N_l23_23501


namespace largest_possible_green_socks_l23_23816

/--
A box contains a mixture of green socks and yellow socks, with at most 2023 socks in total.
The probability of randomly pulling out two socks of the same color is exactly 1/3.
What is the largest possible number of green socks in the box? 
-/
theorem largest_possible_green_socks (g y : ℤ) (t : ℕ) (h : t ≤ 2023) 
  (prob_condition : (g * (g - 1) + y * (y - 1) = t * (t - 1) / 3)) : 
  g ≤ 990 :=
sorry

end largest_possible_green_socks_l23_23816


namespace range_of_a_l23_23888

open Real

noncomputable def f (a x : ℝ) : ℝ := log x + (1 / 2) * x ^ 2 + a * x

theorem range_of_a (a : ℝ) : (∃ x > 0, deriv (f a) x = 3) ↔ a < 1 := by
  sorry

end range_of_a_l23_23888


namespace angle_in_first_quadrant_l23_23169

-- Define the condition and equivalence proof problem in Lean 4
theorem angle_in_first_quadrant (deg : ℤ) (h1 : deg = 721) : (deg % 360) > 0 := 
by 
  have : deg % 360 = 1 := sorry
  exact sorry

end angle_in_first_quadrant_l23_23169


namespace interest_rate_first_part_eq_3_l23_23584

variable (T P1 P2 r2 I : ℝ)
variable (hT : T = 3400)
variable (hP1 : P1 = 1300)
variable (hP2 : P2 = 2100)
variable (hr2 : r2 = 5)
variable (hI : I = 144)

theorem interest_rate_first_part_eq_3 (r : ℝ) (h : (P1 * r) / 100 + (P2 * r2) / 100 = I) : r = 3 :=
by
  -- leaning in the proof
  sorry

end interest_rate_first_part_eq_3_l23_23584


namespace product_is_correct_l23_23839

-- Define the numbers a and b
def a : ℕ := 72519
def b : ℕ := 9999

-- Theorem statement that proves the correctness of the product
theorem product_is_correct : a * b = 725117481 :=
by
  sorry

end product_is_correct_l23_23839


namespace factorization1_factorization2_factorization3_factorization4_l23_23943

-- Question 1
theorem factorization1 (a b : ℝ) :
  4 * a^2 * b - 6 * a * b^2 = 2 * a * b * (2 * a - 3 * b) :=
by 
  sorry

-- Question 2
theorem factorization2 (x y : ℝ) :
  25 * x^2 - 9 * y^2 = (5 * x + 3 * y) * (5 * x - 3 * y) :=
by 
  sorry

-- Question 3
theorem factorization3 (a b : ℝ) :
  2 * a^2 * b - 8 * a * b^2 + 8 * b^3 = 2 * b * (a - 2 * b)^2 :=
by 
  sorry

-- Question 4
theorem factorization4 (x : ℝ) :
  (x + 2) * (x - 8) + 25 = (x - 3)^2 :=
by 
  sorry

end factorization1_factorization2_factorization3_factorization4_l23_23943


namespace average_letters_per_day_l23_23937

theorem average_letters_per_day (letters_tuesday : Nat) (letters_wednesday : Nat) (total_days : Nat) 
  (h_tuesday : letters_tuesday = 7) (h_wednesday : letters_wednesday = 3) (h_days : total_days = 2) : 
  (letters_tuesday + letters_wednesday) / total_days = 5 :=
by 
  sorry

end average_letters_per_day_l23_23937


namespace time_to_produce_one_item_l23_23902

-- Definitions based on the conditions
def itemsProduced : Nat := 300
def totalTimeHours : ℝ := 2.0
def minutesPerHour : ℝ := 60.0

-- The statement we need to prove
theorem time_to_produce_one_item : (totalTimeHours / itemsProduced * minutesPerHour) = 0.4 := by
  sorry

end time_to_produce_one_item_l23_23902


namespace intersection_of_A_and_B_l23_23789

-- Given sets A and B
def A : Set ℤ := { -1, 0, 1, 2 }
def B : Set ℤ := { 0, 2, 3 }

-- Prove that the intersection of A and B is {0, 2}
theorem intersection_of_A_and_B : A ∩ B = {0, 2} := 
by
  sorry

end intersection_of_A_and_B_l23_23789


namespace minimum_value_is_16_l23_23777

noncomputable def minimum_value_expression (x y : ℝ) (hx : x > 1) (hy : y > 1) : ℝ :=
  (x^3 / (y - 1) + y^3 / (x - 1))

theorem minimum_value_is_16 (x y : ℝ) (hx : x > 1) (hy : y > 1) :
  minimum_value_expression x y hx hy ≥ 16 :=
sorry

end minimum_value_is_16_l23_23777


namespace pyramid_volume_l23_23841

noncomputable def volume_of_pyramid (a b c : ℝ) : ℝ :=
  (1 / 3) * a * b * c * Real.sqrt 2

theorem pyramid_volume (a b c : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (A1 : ∃ x y, 1 / 2 * x * y = a^2) 
  (A2 : ∃ y z, 1 / 2 * y * z = b^2) 
  (A3 : ∃ x z, 1 / 2 * x * z = c^2)
  (h_perpendicular : True) :
  volume_of_pyramid a b c = (1 / 3) * a * b * c * Real.sqrt 2 :=
sorry

end pyramid_volume_l23_23841


namespace student_selection_l23_23373

theorem student_selection : 
  let first_year := 4
  let second_year := 5
  let third_year := 4
  (first_year * second_year) + (first_year * third_year) + (second_year * third_year) = 56 := by
  let first_year := 4
  let second_year := 5
  let third_year := 4
  sorry

end student_selection_l23_23373


namespace minor_axis_length_of_ellipse_l23_23094

theorem minor_axis_length_of_ellipse :
  ∀ (x y : ℝ), (9 * x^2 + y^2 = 36) → 4 = 4 :=
by
  intros x y h
  -- the proof goes here
  sorry

end minor_axis_length_of_ellipse_l23_23094


namespace inverse_proportion_order_l23_23732

theorem inverse_proportion_order (k : ℝ) (y1 y2 y3 : ℝ) 
  (h1 : k > 0) 
  (ha : y1 = k / (-3)) 
  (hb : y2 = k / (-2)) 
  (hc : y3 = k / 2) : 
  y2 < y1 ∧ y1 < y3 := 
sorry

end inverse_proportion_order_l23_23732


namespace total_money_is_2800_l23_23282

-- Define variables for money
def Cecil_money : ℕ := 600
def Catherine_money : ℕ := 2 * Cecil_money - 250
def Carmela_money : ℕ := 2 * Cecil_money + 50

-- Assertion to prove the total money 
theorem total_money_is_2800 : Cecil_money + Catherine_money + Carmela_money = 2800 :=
by
  -- placeholder proof
  sorry

end total_money_is_2800_l23_23282


namespace price_of_other_stock_l23_23225

theorem price_of_other_stock (total_shares : ℕ) (total_spent : ℝ) (share_1_quantity : ℕ) (share_1_price : ℝ) :
  total_shares = 450 ∧ total_spent = 1950 ∧ share_1_quantity = 400 ∧ share_1_price = 3 →
  (750 / 50 = 15) :=
by sorry

end price_of_other_stock_l23_23225


namespace fraction_decomposition_l23_23426

theorem fraction_decomposition :
  ∀ (A B : ℚ), (∀ x : ℚ, x ≠ -2 → x ≠ 4/3 → 
  (7 * x - 15) / ((3 * x - 4) * (x + 2)) = A / (x + 2) + B / (3 * x - 4)) →
  A = 29 / 10 ∧ B = -17 / 10 :=
by
  sorry

end fraction_decomposition_l23_23426


namespace particle_speed_at_time_t_l23_23347

noncomputable def position (t : ℝ) : ℝ × ℝ :=
  (3 * t^2 + t + 1, 6 * t + 2)

theorem particle_speed_at_time_t (t : ℝ) :
  let dx := (position t).1
  let dy := (position t).2
  let vx := 6 * t + 1
  let vy := 6
  let speed := Real.sqrt (vx^2 + vy^2)
  speed = Real.sqrt (36 * t^2 + 12 * t + 37) :=
by
  sorry

end particle_speed_at_time_t_l23_23347


namespace probability_of_point_on_line_4_l23_23933

-- Definitions as per conditions
def total_outcomes : ℕ := 36
def favorable_points : Finset (ℕ × ℕ) := {(1, 3), (2, 2), (3, 1)}
def probability : ℚ := (favorable_points.card : ℚ) / total_outcomes

-- Problem statement to prove
theorem probability_of_point_on_line_4 :
  probability = 1 / 12 :=
by
  sorry

end probability_of_point_on_line_4_l23_23933


namespace apples_for_pies_l23_23811

-- Define the conditions
def apples_per_pie : ℝ := 4.0
def number_of_pies : ℝ := 126.0

-- Define the expected answer
def number_of_apples : ℝ := number_of_pies * apples_per_pie

-- State the theorem to prove the question == answer given the conditions
theorem apples_for_pies : number_of_apples = 504 :=
by
  -- This is where the proof would go. Currently skipped.
  sorry

end apples_for_pies_l23_23811


namespace computation_l23_23960

theorem computation :
  4.165 * 4.8 + 4.165 * 6.7 - 4.165 / (2 / 3) = 41.65 :=
by
  sorry

end computation_l23_23960


namespace jerry_total_shingles_l23_23526

def roof_length : ℕ := 20
def roof_width : ℕ := 40
def num_roofs : ℕ := 3
def shingles_per_square_foot : ℕ := 8

def area_of_one_side (length width : ℕ) : ℕ :=
  length * width

def total_area_one_roof (area_one_side : ℕ) : ℕ :=
  area_one_side * 2

def total_area_three_roofs (total_area_one_roof : ℕ) : ℕ :=
  total_area_one_roof * num_roofs

def total_shingles_needed (total_area_all_roofs shingles_per_square_foot : ℕ) : ℕ :=
  total_area_all_roofs * shingles_per_square_foot

theorem jerry_total_shingles :
  total_shingles_needed (total_area_three_roofs (total_area_one_roof (area_of_one_side roof_length roof_width))) shingles_per_square_foot = 38400 :=
by
  sorry

end jerry_total_shingles_l23_23526


namespace max_two_alphas_l23_23286

theorem max_two_alphas (k : ℕ) (α : ℕ → ℝ) (hα : ∀ n, ∃! i p : ℕ, n = ⌊p * α i⌋ + 1) : k ≤ 2 := 
sorry

end max_two_alphas_l23_23286


namespace distance_rowed_downstream_l23_23994

def speed_of_boat_still_water : ℝ := 70 -- km/h
def distance_upstream : ℝ := 240 -- km
def time_upstream : ℝ := 6 -- hours
def time_downstream : ℝ := 5 -- hours

theorem distance_rowed_downstream :
  let V_b := speed_of_boat_still_water
  let V_upstream := distance_upstream / time_upstream
  let V_s := V_b - V_upstream
  let V_downstream := V_b + V_s
  V_downstream * time_downstream = 500 :=
by
  sorry

end distance_rowed_downstream_l23_23994


namespace final_price_correct_l23_23499

noncomputable def final_price_per_litre : Real :=
  let cost_1 := 70 * 43 * (1 - 0.15)
  let cost_2 := 50 * 51 * (1 + 0.10)
  let cost_3 := 15 * 60 * (1 - 0.08)
  let cost_4 := 25 * 62 * (1 + 0.12)
  let cost_5 := 40 * 67 * (1 - 0.05)
  let cost_6 := 10 * 75 * (1 - 0.18)
  let total_cost := cost_1 + cost_2 + cost_3 + cost_4 + cost_5 + cost_6
  let total_volume := 70 + 50 + 15 + 25 + 40 + 10
  total_cost / total_volume

theorem final_price_correct : final_price_per_litre = 52.80 := by
  sorry

end final_price_correct_l23_23499


namespace runs_scored_by_c_l23_23879

-- Definitions
variables (A B C : ℕ)

-- Conditions as hypotheses
theorem runs_scored_by_c (h1 : B = 3 * A) (h2 : C = 5 * B) (h3 : A + B + C = 95) : C = 75 :=
by
  -- Proof will be here
  sorry

end runs_scored_by_c_l23_23879


namespace negation_of_proposition_l23_23956

variables (a b : ℕ)

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

def both_even (a b : ℕ) : Prop := is_even a ∧ is_even b

def sum_even (a b : ℕ) : Prop := is_even (a + b)

theorem negation_of_proposition : ¬ (both_even a b → sum_even a b) ↔ ¬both_even a b ∨ ¬sum_even a b :=
by sorry

end negation_of_proposition_l23_23956


namespace triangle_shape_isosceles_or_right_l23_23931

variable {A B C : ℝ} -- Angles of the triangle
variable {a b c : ℝ} -- Sides opposite to the angles

theorem triangle_shape_isosceles_or_right (h1 : a^2 + b^2 ≠ 0) (h2 : 
  (a^2 + b^2) * Real.sin (A - B) 
  = (a^2 - b^2) * Real.sin (A + B))
  (h3 : ∀ (A B C : ℝ), A + B + C = π) :
  ∃ (isosceles : Bool), (isosceles = true) ∨ (isosceles = false ∧ A + B = π / 2) :=
sorry

end triangle_shape_isosceles_or_right_l23_23931


namespace minute_hand_gains_per_hour_l23_23667

theorem minute_hand_gains_per_hour (total_gain : ℕ) (total_hours : ℕ) (gain_by_6pm : total_gain = 63) (hours_from_9_to_6 : total_hours = 9) : (total_gain / total_hours) = 7 :=
by
  -- The proof is not required as per instruction.
  sorry

end minute_hand_gains_per_hour_l23_23667


namespace solve_trig_eq_l23_23520

theorem solve_trig_eq (k : ℤ) : 
  ∃ x, 12 * Real.sin x - 5 * Real.cos x = 13 ∧ x = Real.pi / 2 + Real.arctan (5 / 12) + 2 * k * Real.pi :=
sorry

end solve_trig_eq_l23_23520


namespace number_of_paths_l23_23666

/-
We need to define the conditions and the main theorem
-/

def grid_width : ℕ := 5
def grid_height : ℕ := 4
def total_steps : ℕ := 8
def steps_right : ℕ := 5
def steps_up : ℕ := 3

theorem number_of_paths : (Nat.choose total_steps steps_up) = 56 := by
  sorry

end number_of_paths_l23_23666


namespace not_always_true_inequality_l23_23765

theorem not_always_true_inequality (x : ℝ) (hx : x > 0) : 2^x ≤ x^2 := sorry

end not_always_true_inequality_l23_23765


namespace gcd_of_three_numbers_l23_23939

theorem gcd_of_three_numbers :
  Nat.gcd (Nat.gcd 72 120) 168 = 24 :=
sorry

end gcd_of_three_numbers_l23_23939


namespace all_edges_same_color_l23_23185

-- Define the vertices in the two pentagons and the set of all vertices
inductive vertex
| A1 | A2 | A3 | A4 | A5 | B1 | B2 | B3 | B4 | B5
open vertex

-- Predicate to identify edges between vertices
def edge (v1 v2 : vertex) : Prop :=
  match (v1, v2) with
  | (A1, A2) | (A2, A3) | (A3, A4) | (A4, A5) | (A5, A1) => true
  | (B1, B2) | (B2, B3) | (B3, B4) | (B4, B5) | (B5, B1) => true
  | (A1, B1) | (A1, B2) | (A1, B3) | (A1, B4) | (A1, B5) => true
  | (A2, B1) | (A2, B2) | (A2, B3) | (A2, B4) | (A2, B5) => true
  | (A3, B1) | (A3, B2) | (A3, B3) | (A3, B4) | (A3, B5) => true
  | (A4, B1) | (A4, B2) | (A4, B3) | (A4, B4) | (A4, B5) => true
  | (A5, B1) | (A5, B2) | (A5, B3) | (A5, B4) | (A5, B5) => true
  | _ => false

-- Edge coloring predicate 'black' or 'white'
inductive color
| black | white
open color

def edge_color (v1 v2 : vertex) : color → Prop :=
  sorry -- Coloring function needs to be defined accordingly

-- Predicate to check for monochrome triangles
def no_monochrome_triangle : Prop :=
  ∀ v1 v2 v3 : vertex,
    (edge v1 v2 ∧ edge v2 v3 ∧ edge v3 v1) →
    ¬ (∃ c : color, edge_color v1 v2 c ∧ edge_color v2 v3 c ∧ edge_color v3 v1 c)

-- Main theorem statement
theorem all_edges_same_color (no_mt : no_monochrome_triangle) :
  ∃ c : color, ∀ v1 v2 : vertex,
    (edge v1 v2 ∧ (v1 = A1 ∨ v1 = A2 ∨ v1 = A3 ∨ v1 = A4 ∨ v1 = A5) ∧
                 (v2 = A1 ∨ v2 = A2 ∨ v2 = A3 ∨ v2 = A4 ∨ v2 = A5) ) →
    edge_color v1 v2 c ∧
    (edge v1 v2 ∧ (v1 = B1 ∨ v1 = B2 ∨ v1 = B3 ∨ v1 = B4 ∨ v1 = B5) ∧
                 (v2 = B1 ∨ v2 = B2 ∨ v2 = B3 ∨ v2 = B4 ∨ v2 = B5) ) →
    edge_color v1 v2 c := sorry

end all_edges_same_color_l23_23185


namespace greatest_integer_value_x_l23_23936

theorem greatest_integer_value_x :
  ∃ x : ℤ, (8 - 3 * (2 * x + 1) > 26) ∧ ∀ y : ℤ, (8 - 3 * (2 * y + 1) > 26) → y ≤ x :=
sorry

end greatest_integer_value_x_l23_23936


namespace sum_and_product_of_roots_l23_23832

-- Define the polynomial equation and the conditions on the roots
def cubic_eqn (x : ℝ) : Prop := 3 * x ^ 3 - 18 * x ^ 2 + 27 * x - 6 = 0

-- The Lean statement for the given problem
theorem sum_and_product_of_roots (p q r : ℝ) :
  cubic_eqn p ∧ cubic_eqn q ∧ cubic_eqn r →
  (p + q + r = 6) ∧ (p * q * r = 2) :=
by
  sorry

end sum_and_product_of_roots_l23_23832


namespace basketball_holes_l23_23813

theorem basketball_holes (soccer_balls total_basketballs soccer_balls_with_hole balls_without_holes basketballs_without_holes: ℕ) 
  (h1: soccer_balls = 40) 
  (h2: total_basketballs = 15)
  (h3: soccer_balls_with_hole = 30) 
  (h4: balls_without_holes = 18) 
  (h5: basketballs_without_holes = 8) 
  : (total_basketballs - basketballs_without_holes = 7) := 
by
  sorry

end basketball_holes_l23_23813


namespace find_value_of_expression_l23_23285

theorem find_value_of_expression (x y : ℝ)
  (h1 : 5 * x + y = 19)
  (h2 : x + 3 * y = 1) :
  3 * x + 2 * y = 10 :=
sorry

end find_value_of_expression_l23_23285


namespace prove_inequality_l23_23303

variables {a b c A B C k : ℝ}

-- Define the conditions
def conditions (a b c A B C k : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ A > 0 ∧ B > 0 ∧ C > 0 ∧ k > 0 ∧
  a + A = k ∧ b + B = k ∧ c + C = k

-- Define the theorem to be proven
theorem prove_inequality (a b c A B C k : ℝ) (h : conditions a b c A B C k) :
  a * B + b * C + c * A ≤ k^2 :=
sorry

end prove_inequality_l23_23303


namespace find_multiplier_n_l23_23139

variable (x y n : ℝ)

theorem find_multiplier_n (h1 : 5 * x = n * y) 
  (h2 : x * y ≠ 0) 
  (h3 : (1/3 * x) / (1/5 * y) = 1.9999999999999998) : 
  n = 6 := 
by
  sorry

end find_multiplier_n_l23_23139


namespace work_day_percentage_l23_23530

theorem work_day_percentage 
  (work_day_hours : ℕ) 
  (first_meeting_minutes : ℕ) 
  (second_meeting_factor : ℕ) 
  (h_work_day : work_day_hours = 10) 
  (h_first_meeting : first_meeting_minutes = 60) 
  (h_second_meeting_factor : second_meeting_factor = 2) :
  ((first_meeting_minutes + second_meeting_factor * first_meeting_minutes) / (work_day_hours * 60) : ℚ) * 100 = 30 :=
sorry

end work_day_percentage_l23_23530


namespace exponentiation_of_squares_l23_23382

theorem exponentiation_of_squares :
  ((Real.sqrt 2 + 1)^2000 * (Real.sqrt 2 - 1)^2000 = 1) :=
by
  sorry

end exponentiation_of_squares_l23_23382


namespace smallest_x_l23_23087

theorem smallest_x (x : ℕ) :
  (x % 5 = 4) ∧ (x % 7 = 6) ∧ (x % 8 = 7) → x = 279 :=
by
  sorry

end smallest_x_l23_23087


namespace elsa_ends_with_145_marbles_l23_23597

theorem elsa_ends_with_145_marbles :
  let initial := 150
  let after_breakfast := initial - 7
  let after_lunch := after_breakfast - 57
  let after_afternoon := after_lunch + 25
  let after_evening := after_afternoon + 85
  let after_exchange := after_evening - 9 + 6
  let final := after_exchange - 48
  final = 145 := by
    sorry

end elsa_ends_with_145_marbles_l23_23597


namespace factorize_expression_l23_23930

theorem factorize_expression (x : ℝ) : x^3 - 4 * x^2 + 4 * x = x * (x - 2)^2 :=
by
  sorry

end factorize_expression_l23_23930


namespace seq_a2020_l23_23125

def seq (a : ℕ → ℕ) : Prop :=
(∀ n : ℕ, (a n + a (n+1) ≠ a (n+2) + a (n+3))) ∧
(∀ n : ℕ, (a n + a (n+1) + a (n+2) ≠ a (n+3) + a (n+4) + a (n+5))) ∧
(a 1 = 0)

theorem seq_a2020 (a : ℕ → ℕ) (h : seq a) : a 2020 = 1 :=
sorry

end seq_a2020_l23_23125


namespace reduced_less_than_scaled_l23_23278

-- Define the conditions
def original_flow_rate : ℝ := 5.0
def reduced_flow_rate : ℝ := 2.0
def scaled_flow_rate : ℝ := 0.6 * original_flow_rate

-- State the theorem we need to prove
theorem reduced_less_than_scaled : scaled_flow_rate - reduced_flow_rate = 1.0 := 
by
  -- insert the detailed proof steps here
  sorry

end reduced_less_than_scaled_l23_23278


namespace largest_consecutive_integers_sum_to_45_l23_23643

theorem largest_consecutive_integers_sum_to_45 (x n : ℕ) (h : 45 = n * (2 * x + n - 1) / 2) : n ≤ 9 :=
sorry

end largest_consecutive_integers_sum_to_45_l23_23643


namespace Alma_test_score_l23_23958

-- Define the constants and conditions
variables (Alma_age Melina_age Alma_score : ℕ)

-- Conditions
axiom Melina_is_60 : Melina_age = 60
axiom Melina_3_times_Alma : Melina_age = 3 * Alma_age
axiom sum_ages_twice_score : Melina_age + Alma_age = 2 * Alma_score

-- Goal
theorem Alma_test_score : Alma_score = 40 :=
by
  sorry

end Alma_test_score_l23_23958


namespace final_number_is_50_l23_23608

theorem final_number_is_50 (initial_ones initial_fours : ℕ) (h1 : initial_ones = 900) (h2 : initial_fours = 100) :
  ∃ (z : ℝ), (900 * (1:ℝ)^2 + 100 * (4:ℝ)^2) = z^2 ∧ z = 50 :=
by
  sorry

end final_number_is_50_l23_23608


namespace N_subset_M_values_l23_23755

def M : Set ℝ := { x | 2 * x^2 - 3 * x - 2 = 0 }
def N (a : ℝ) : Set ℝ := { x | a * x = 1 }

theorem N_subset_M_values (a : ℝ) (h : N a ⊆ M) : a = 0 ∨ a = -2 ∨ a = 1/2 := 
by
  sorry

end N_subset_M_values_l23_23755


namespace percentage_transactions_anthony_handled_more_l23_23372

theorem percentage_transactions_anthony_handled_more (M A C J : ℕ) (P : ℚ)
  (hM : M = 90)
  (hJ : J = 83)
  (hCJ : J = C + 17)
  (hCA : C = (2 * A) / 3)
  (hP : P = ((A - M): ℚ) / M * 100) :
  P = 10 := by
  sorry

end percentage_transactions_anthony_handled_more_l23_23372


namespace integer_values_satisfying_sqrt_condition_l23_23386

theorem integer_values_satisfying_sqrt_condition : ∃! n : Nat, 2.5 < Real.sqrt n ∧ Real.sqrt n < 3.5 :=
by {
  sorry -- Proof to be filled in
}

end integer_values_satisfying_sqrt_condition_l23_23386


namespace welders_started_on_other_project_l23_23538

theorem welders_started_on_other_project
  (r : ℝ) (x : ℝ) (W : ℝ)
  (h1 : 16 * r * 8 = W)
  (h2 : (16 - x) * r * 24 = W - 16 * r) :
  x = 11 :=
by
  sorry

end welders_started_on_other_project_l23_23538


namespace complex_mul_l23_23039

theorem complex_mul (i : ℂ) (hi : i * i = -1) : (1 - i) * (3 + i) = 4 - 2 * i :=
by
  sorry

end complex_mul_l23_23039


namespace problem_I_solution_problem_II_solution_l23_23527

noncomputable def f (x : ℝ) : ℝ := |3 * x - 2| + |x - 2|

-- Problem (I): Solve the inequality f(x) <= 8
theorem problem_I_solution (x : ℝ) : 
  f x ≤ 8 ↔ -1 ≤ x ∧ x ≤ 3 :=
sorry

-- Problem (II): Find the range of the real number m
theorem problem_II_solution (x m : ℝ) : 
  f x ≥ (m^2 - m + 2) * |x| ↔ (0 ≤ m ∧ m ≤ 1) :=
sorry

end problem_I_solution_problem_II_solution_l23_23527


namespace common_ratio_l23_23359

def geom_seq (a : ℕ → ℝ) (q : ℝ) := ∀ n, a (n + 1) = a n * q
def arith_seq (a : ℕ → ℝ) (x y z : ℕ) := 2 * a z = a x + a y

theorem common_ratio (a : ℕ → ℝ) (q : ℝ) (h_geom : geom_seq a q) (h_arith : arith_seq a 0 1 2) (h_nonzero : a 0 ≠ 0) : q = 1 ∨ q = -1/2 :=
by
  sorry

end common_ratio_l23_23359


namespace value_of_a_l23_23872

theorem value_of_a (a b : ℝ) (h1 : b = 2 * a) (h2 : b = 15 - 4 * a) : a = 5 / 2 :=
by
  sorry

end value_of_a_l23_23872


namespace car_trip_time_l23_23320

theorem car_trip_time (walking_mixed: 1.5 = 1.25 + x) 
                      (walking_both: 2.5 = 2 * 1.25) : 
  2 * x * 60 = 30 :=
by sorry

end car_trip_time_l23_23320


namespace probability_non_smokers_getting_lung_cancer_l23_23406

theorem probability_non_smokers_getting_lung_cancer 
  (overall_lung_cancer : ℝ)
  (smokers_fraction : ℝ)
  (smokers_lung_cancer : ℝ)
  (non_smokers_lung_cancer : ℝ)
  (H1 : overall_lung_cancer = 0.001)
  (H2 : smokers_fraction = 0.2)
  (H3 : smokers_lung_cancer = 0.004)
  (H4 : overall_lung_cancer = smokers_fraction * smokers_lung_cancer + (1 - smokers_fraction) * non_smokers_lung_cancer) :
  non_smokers_lung_cancer = 0.00025 := by
  sorry

end probability_non_smokers_getting_lung_cancer_l23_23406


namespace sara_total_cents_l23_23751

def number_of_quarters : ℕ := 11
def value_per_quarter : ℕ := 25

theorem sara_total_cents : number_of_quarters * value_per_quarter = 275 := by
  sorry

end sara_total_cents_l23_23751


namespace sum_of_angles_x_y_l23_23460

theorem sum_of_angles_x_y :
  let num_arcs := 15
  let angle_per_arc := 360 / num_arcs
  let central_angle_x := 3 * angle_per_arc
  let central_angle_y := 5 * angle_per_arc
  let inscribed_angle (central_angle : ℝ) := central_angle / 2
  let angle_x := inscribed_angle central_angle_x
  let angle_y := inscribed_angle central_angle_y
  angle_x + angle_y = 96 := 
  sorry

end sum_of_angles_x_y_l23_23460


namespace gina_snake_mice_in_decade_l23_23531

-- Definitions based on the conditions in a)
def weeks_per_mouse : ℕ := 4
def weeks_per_year : ℕ := 52
def years_per_decade : ℕ := 10

-- The final statement to prove
theorem gina_snake_mice_in_decade : 
  (weeks_per_year / weeks_per_mouse) * years_per_decade = 130 :=
by
  sorry

end gina_snake_mice_in_decade_l23_23531


namespace speed_down_l23_23723

theorem speed_down {u avg_speed d v : ℝ} (hu : u = 18) (havg : avg_speed = 20.571428571428573) (hv : 2 * d / ((d / u) + (d / v)) = avg_speed) : v = 24 :=
by
  have h1 : 20.571428571428573 = 20.571428571428573 := rfl
  have h2 : 18 = 18 := rfl
  sorry

end speed_down_l23_23723


namespace problem_solution_l23_23919

variable (x y : ℝ)

theorem problem_solution :
  (x - y + 1) * (x - y - 1) = x^2 - 2 * x * y + y^2 - 1 :=
by
  sorry

end problem_solution_l23_23919


namespace ratio_37m48s_2h13m15s_l23_23203

-- Define the total seconds for 37 minutes and 48 seconds
def t1 := 37 * 60 + 48

-- Define the total seconds for 2 hours, 13 minutes, and 15 seconds
def t2 := 2 * 3600 + 13 * 60 + 15

-- Prove the ratio t1 / t2 = 2268 / 7995
theorem ratio_37m48s_2h13m15s : t1 / t2 = 2268 / 7995 := 
by sorry

end ratio_37m48s_2h13m15s_l23_23203


namespace gcd_equation_solution_l23_23011

theorem gcd_equation_solution (x y : ℕ) (h : Nat.gcd x y + x * y / Nat.gcd x y = x + y) : y ∣ x ∨ x ∣ y :=
 by
 sorry

end gcd_equation_solution_l23_23011


namespace find_g_l23_23196

-- Given conditions
def line_equation (x y : ℝ) : Prop := y = 2 * x - 10
def parameterization (g : ℝ → ℝ) (t : ℝ) : Prop := 20 * t - 8 = 2 * g t - 10

-- Statement to prove
theorem find_g (g : ℝ → ℝ) (t : ℝ) :
  (∀ x y, line_equation x y → parameterization g t) →
  g t = 10 * t + 1 :=
sorry

end find_g_l23_23196


namespace part_a_l23_23653

theorem part_a (n : ℕ) : ((x^2 + x + 1) ∣ (x^(2 * n) + x^n + 1)) ↔ (n % 3 = 0) := sorry

end part_a_l23_23653


namespace intersection_eq_l23_23260

def setA : Set ℝ := { x | abs (x - 3) < 2 }
def setB : Set ℝ := { x | (x - 4) / x ≥ 0 }

theorem intersection_eq : setA ∩ setB = { x | 4 ≤ x ∧ x < 5 } :=
by 
  sorry

end intersection_eq_l23_23260


namespace gcd_problem_l23_23868

theorem gcd_problem (x : ℤ) (h : ∃ k, x = 2 * 2027 * k) :
  Int.gcd (3 * x ^ 2 + 47 * x + 101) (x + 23) = 1 :=
sorry

end gcd_problem_l23_23868


namespace initial_apples_count_l23_23918

theorem initial_apples_count (a b : ℕ) (h₁ : b = 13) (h₂ : b = a + 5) : a = 8 :=
by
  sorry

end initial_apples_count_l23_23918


namespace find_vector_BC_l23_23798

structure Point2D where
  x : ℝ
  y : ℝ

def A : Point2D := ⟨0, 1⟩
def B : Point2D := ⟨3, 2⟩
def AC : Point2D := ⟨-4, -3⟩

def vector_add (p1 p2 : Point2D) : Point2D := ⟨p1.x + p2.x, p1.y + p2.y⟩
def vector_sub (p1 p2 : Point2D) : Point2D := ⟨p1.x - p2.x, p1.y - p2.y⟩

def C : Point2D := vector_add A AC
def BC : Point2D := vector_sub C B

theorem find_vector_BC : BC = ⟨-7, -4⟩ := by
  sorry

end find_vector_BC_l23_23798


namespace common_rational_root_is_neg_one_third_l23_23668

theorem common_rational_root_is_neg_one_third (a b c d e f g : ℚ) :
  ∃ k : ℚ, (75 * k^4 + a * k^3 + b * k^2 + c * k + 12 = 0) ∧
           (12 * k^5 + d * k^4 + e * k^3 + f * k^2 + g * k + 75 = 0) ∧
           (¬ k.isInt) ∧ (k < 0) ∧ (k = -1/3) :=
sorry

end common_rational_root_is_neg_one_third_l23_23668


namespace jose_profit_share_l23_23065

theorem jose_profit_share :
  ∀ (Tom_investment Jose_investment total_profit month_investment_tom month_investment_jose total_month_investment: ℝ),
    Tom_investment = 30000 →
    ∃ (months_tom months_jose : ℝ), months_tom = 12 ∧ months_jose = 10 →
      Jose_investment = 45000 →
      total_profit = 72000 →
      month_investment_tom = Tom_investment * months_tom →
      month_investment_jose = Jose_investment * months_jose →
      total_month_investment = month_investment_tom + month_investment_jose →
      (Jose_investment * months_jose / total_month_investment) * total_profit = 40000 :=
by
  sorry

end jose_profit_share_l23_23065


namespace compute_exp_l23_23001

theorem compute_exp : 3 * 3^4 + 9^30 / 9^28 = 324 := 
by sorry

end compute_exp_l23_23001


namespace min_a_b_sum_l23_23177

theorem min_a_b_sum (a b : ℕ) (x : ℕ → ℕ)
  (h0 : x 1 = a)
  (h1 : x 2 = b)
  (h2 : ∀ n, x (n+2) = x n + x (n+1))
  (h3 : ∃ n, x n = 1000) : a + b = 10 :=
sorry

end min_a_b_sum_l23_23177


namespace parabola_shifts_down_decrease_c_real_roots_l23_23488

-- The parabolic function and conditions
variables {a b c k : ℝ}

-- Assumption that a is positive
axiom ha : a > 0

-- Parabola shifts down when constant term c is decreased
theorem parabola_shifts_down (c : ℝ) (k : ℝ) (hk : k > 0) :
  ∀ x, (a * x^2 + b * x + (c - k)) = (a * x^2 + b * x + c) - k :=
by sorry

-- Discriminant of quadratic equation ax^2 + bx + c = 0
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- If the discriminant is negative, decreasing c can result in real roots
theorem decrease_c_real_roots (b c : ℝ) (hb : b^2 < 4 * a * c) (k : ℝ) (hk : k > 0) :
  discriminant a b (c - k) ≥ 0 :=
by sorry

end parabola_shifts_down_decrease_c_real_roots_l23_23488


namespace center_of_circle_l23_23560

theorem center_of_circle (x y : ℝ) :
  x^2 + y^2 - 2 * x - 6 * y + 1 = 0 →
  (1, 3) = (1, 3) :=
by
  intros h
  sorry

end center_of_circle_l23_23560


namespace fiona_pairs_l23_23293

-- Define the combinatorial calculation using the combination formula
def combination (n k : ℕ) := n.choose k

-- The main theorem stating that the number of pairs from 6 people is 15
theorem fiona_pairs : combination 6 2 = 15 :=
by
  sorry

end fiona_pairs_l23_23293


namespace probability_of_white_first_red_second_l23_23636

noncomputable def probability_white_first_red_second : ℚ :=
let totalBalls := 6
let probWhiteFirst := 1 / totalBalls
let remainingBalls := totalBalls - 1
let probRedSecond := 1 / remainingBalls
probWhiteFirst * probRedSecond

theorem probability_of_white_first_red_second :
  probability_white_first_red_second = 1 / 30 :=
by
  sorry

end probability_of_white_first_red_second_l23_23636


namespace find_value_l23_23002

theorem find_value (a b c : ℝ) (h1 : a + b = 8) (h2 : a * b = c^2 + 16) : a + 2 * b + 3 * c = 12 := by
  sorry

end find_value_l23_23002


namespace number_of_dress_designs_l23_23009

open Nat

theorem number_of_dress_designs : (3 * 4 = 12) :=
by
  rfl

end number_of_dress_designs_l23_23009


namespace value_of_a3_l23_23843

variable {a_n : ℕ → ℤ}
variable {S_n : ℕ → ℤ}

-- Given conditions
def S (n : ℕ) : ℤ := 2 * (n ^ 2) - 1
def a (n : ℕ) : ℤ := S n - S (n - 1)

-- The theorem to prove
theorem value_of_a3 : a 3 = 10 := by
  sorry

end value_of_a3_l23_23843


namespace true_propositions_in_reverse_neg_neg_reverse_l23_23272

theorem true_propositions_in_reverse_neg_neg_reverse (a b : ℕ) : 
  (¬ (a ≠ 0 → a * b ≠ 0) ∧ ∃ (a : ℕ), (a = 0 ∧ a * b ≠ 0) ∨ (a ≠ 0 ∧ a * b = 0) ∧ ¬ (¬ ∃ (a : ℕ), a ≠ 0 ∧ a * b ≠ 0 ∧ ¬ ∃ (a : ℕ), a = 0 ∧ a * b = 0)) ∧ (0 = 1) :=
by {
  sorry
}

end true_propositions_in_reverse_neg_neg_reverse_l23_23272


namespace find_b_l23_23251

noncomputable def a (c : ℚ) : ℚ := 10 * c - 10
noncomputable def b (c : ℚ) : ℚ := 10 * c + 10
noncomputable def c_val := (200 : ℚ) / 21

theorem find_b : 
  let a := a c_val
  let b := b c_val
  let c := c_val
  a + b + c = 200 ∧ 
  a + 10 = b - 10 ∧ 
  a + 10 = 10 * c → 
  b = 2210 / 21 :=
by
  intros
  sorry

end find_b_l23_23251


namespace area_transformed_function_l23_23983

noncomputable def area_g : ℝ := 15

noncomputable def area_4g_shifted : ℝ :=
  4 * area_g

theorem area_transformed_function :
  area_4g_shifted = 60 := by
  sorry

end area_transformed_function_l23_23983


namespace bob_wins_game_l23_23358

theorem bob_wins_game : 
  ∀ n : ℕ, 0 < n → 
  (∃ k ≥ 1, ∀ m : ℕ, 0 < m → (∃ a : ℕ, a ≥ 1 ∧ m - a*a = 0) ∨ 
    (∃ k : ℕ, k ≥ 1 ∧ (m = m^k → ¬ (∃ a : ℕ, a ≥ 1 ∧ m - a*a = 0)))
  ) :=
sorry

end bob_wins_game_l23_23358


namespace solve_for_x_l23_23741

theorem solve_for_x (x : ℝ) (h : 1 / (x + 5) + 1 / (x - 5) = 1 / (x - 5)) : x = 1 / 2 :=
by
  sorry

end solve_for_x_l23_23741


namespace solution_inequality_l23_23954

theorem solution_inequality (p p' q q' : ℝ) (hp : p ≠ 0) (hp' : p' ≠ 0)
    (h : -q / p > -q' / p') : q / p < q' / p' :=
by
  sorry

end solution_inequality_l23_23954


namespace solve_for_x_l23_23266

theorem solve_for_x (x : ℝ) (y : ℝ) (h1 : y = 2) (h2 : y = 1 / (4 * x + 2)) : x = -3/8 :=
by
  -- The proof will go here
  sorry

end solve_for_x_l23_23266


namespace find_prob_p_l23_23202

variable (p : ℚ)

theorem find_prob_p (h : 15 * p^4 * (1 - p)^2 = 500 / 2187) : p = 3 / 7 := 
  sorry

end find_prob_p_l23_23202


namespace average_people_per_hour_rounded_l23_23631

def people_moving_per_hour (total_people : ℕ) (days : ℕ) (hours_per_day : ℕ) : ℕ :=
  let total_hours := days * hours_per_day
  (total_people / total_hours : ℕ)

theorem average_people_per_hour_rounded :
  people_moving_per_hour 4500 5 24 = 38 := 
  sorry

end average_people_per_hour_rounded_l23_23631


namespace mod_inverse_11_mod_1105_l23_23045

theorem mod_inverse_11_mod_1105 : (11 * 201) % 1105 = 1 :=
  by 
    sorry

end mod_inverse_11_mod_1105_l23_23045


namespace common_ratio_geometric_series_l23_23558

theorem common_ratio_geometric_series :
  let a := (7 : ℚ) / 8
  let b := - (5 : ℚ) / 12
  (b / a) = - (10 : ℚ) / 21 :=
by
  -- definitions
  let a := (7 : ℚ) / 8
  let b := - (5 : ℚ) / 12
  -- assertion
  have ratio := b / a
  sorry

end common_ratio_geometric_series_l23_23558


namespace percentage_students_passed_is_35_l23_23007

/-
The problem is to prove the percentage of students who passed the examination, given that 520 out of 800 students failed, is 35%.
-/

def total_students : ℕ := 800
def failed_students : ℕ := 520
def passed_students : ℕ := total_students - failed_students

def percentage_passed : ℕ := (passed_students * 100) / total_students

theorem percentage_students_passed_is_35 : percentage_passed = 35 :=
by
  -- Here the proof will go.
  sorry

end percentage_students_passed_is_35_l23_23007


namespace trout_split_equally_l23_23750

-- Conditions: Nancy and Joan caught 18 trout and split them equally
def total_trout : ℕ := 18
def equal_split (n : ℕ) : ℕ := n / 2

-- Theorem: Prove that if they equally split the trout, each person will get 9 trout.
theorem trout_split_equally : equal_split total_trout = 9 :=
by 
  -- Placeholder for the actual proof
  sorry

end trout_split_equally_l23_23750


namespace min_value_of_expression_l23_23998

theorem min_value_of_expression
  (x y : ℝ) 
  (h : x + y = 1) : 
  ∃ (m : ℝ), m = 2 * x^2 + 3 * y^2 ∧ m = 6 / 5 := 
sorry

end min_value_of_expression_l23_23998


namespace lemonade_water_requirement_l23_23677

variables (W S L H : ℕ)

-- Definitions based on the conditions
def water_equation (W S : ℕ) := W = 5 * S
def sugar_equation (S L : ℕ) := S = 3 * L
def honey_equation (H L : ℕ) := H = L
def lemon_juice_amount (L : ℕ) := L = 2

-- Theorem statement for the proof problem
theorem lemonade_water_requirement :
  ∀ (W S L H : ℕ), 
  (water_equation W S) →
  (sugar_equation S L) →
  (honey_equation H L) →
  (lemon_juice_amount L) →
  W = 30 :=
by
  intros W S L H hW hS hH hL
  sorry

end lemonade_water_requirement_l23_23677


namespace initial_savings_correct_l23_23451

-- Define the constants for ticket prices and number of tickets.
def vip_ticket_price : ℕ := 100
def vip_tickets : ℕ := 2
def regular_ticket_price : ℕ := 50
def regular_tickets : ℕ := 3
def leftover_savings : ℕ := 150

-- Define the total cost of tickets.
def total_cost : ℕ := (vip_ticket_price * vip_tickets) + (regular_ticket_price * regular_tickets)

-- Define the initial savings calculation.
def initial_savings : ℕ := total_cost + leftover_savings

-- Theorem stating the initial savings should be $500.
theorem initial_savings_correct : initial_savings = 500 :=
by
  -- Proof steps can be added here.
  sorry

end initial_savings_correct_l23_23451


namespace time_to_cover_escalator_l23_23713

-- Definitions of the rates and length
def escalator_speed : ℝ := 12
def person_speed : ℝ := 2
def escalator_length : ℝ := 210

-- Theorem statement that we need to prove
theorem time_to_cover_escalator :
  (escalator_length / (escalator_speed + person_speed) = 15) :=
by
  sorry

end time_to_cover_escalator_l23_23713


namespace initial_investors_and_contribution_l23_23660

theorem initial_investors_and_contribution :
  ∃ (x y : ℕ), 
    (x - 10) * (y + 1) = x * y ∧
    (x - 25) * (y + 3) = x * y ∧
    x = 100 ∧ 
    y = 9 :=
by
  sorry

end initial_investors_and_contribution_l23_23660


namespace sum_m_n_is_55_l23_23853

theorem sum_m_n_is_55 (a b c : ℝ) (m n : ℕ) (h_rel_prime : Nat.gcd m n = 1)
  (h1 : 5 / a = b + c) (h2 : 10 / b = c + a) (h3 : 13 / c = a + b) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : (a + b + c) = m / n) : m + n = 55 :=
  sorry

end sum_m_n_is_55_l23_23853


namespace number_of_rods_in_one_mile_l23_23581

theorem number_of_rods_in_one_mile :
  (1 : ℤ) * 6 * 60 = 360 :=
by
  sorry

end number_of_rods_in_one_mile_l23_23581


namespace percentage_of_rotten_bananas_l23_23881

-- Define the initial conditions and the question as a Lean theorem statement
theorem percentage_of_rotten_bananas (oranges bananas : ℕ) (perc_rot_oranges perc_good_fruits : ℝ) 
  (total_fruits good_fruits good_oranges good_bananas rotten_bananas perc_rot_bananas : ℝ) :
  oranges = 600 →
  bananas = 400 →
  perc_rot_oranges = 0.15 →
  perc_good_fruits = 0.886 →
  total_fruits = (oranges + bananas) →
  good_fruits = (perc_good_fruits * total_fruits) →
  good_oranges = ((1 - perc_rot_oranges) * oranges) →
  good_bananas = (good_fruits - good_oranges) →
  rotten_bananas = (bananas - good_bananas) →
  perc_rot_bananas = ((rotten_bananas / bananas) * 100) →
  perc_rot_bananas = 6 :=
by
  intros; sorry

end percentage_of_rotten_bananas_l23_23881


namespace foma_wait_time_probability_l23_23362

noncomputable def probability_no_more_than_four_minutes_wait (x y : ℝ) : ℝ :=
if h : 2 < x ∧ x < y ∧ y < 10 ∧ y - x ≤ 4 then
  (1 / 2)
else 0

theorem foma_wait_time_probability :
  ∀ (x y : ℝ), 2 < x → x < y → y < 10 → 
  (probability_no_more_than_four_minutes_wait x y) = 1 / 2 :=
sorry

end foma_wait_time_probability_l23_23362


namespace fraction_addition_simplified_form_l23_23395

theorem fraction_addition_simplified_form :
  (7 / 8) + (3 / 5) = 59 / 40 := 
by sorry

end fraction_addition_simplified_form_l23_23395


namespace correct_minutes_added_l23_23158

theorem correct_minutes_added :
  let time_lost_per_day : ℚ := 3 + 1/4
  let start_time := 1 -- in P.M. on March 15
  let end_time := 3 -- in P.M. on March 22
  let total_days := 7 -- days from March 15 to March 22
  let extra_hours := 2 -- hours on March 22 from 1 P.M. to 3 P.M.
  let total_hours := (total_days * 24) + extra_hours
  let time_lost_per_minute := time_lost_per_day / (24 * 60)
  let total_time_lost := total_hours * time_lost_per_minute
  let total_time_lost_minutes := total_time_lost * 60
  n = total_time_lost_minutes 
→ n = 221 / 96 := 
sorry

end correct_minutes_added_l23_23158


namespace ratio_part_to_whole_number_l23_23837

theorem ratio_part_to_whole_number (P N : ℚ) 
  (h1 : (1 / 4) * (1 / 3) * P = 25) 
  (h2 : 0.40 * N = 300) : P / N = 2 / 5 :=
by
  sorry

end ratio_part_to_whole_number_l23_23837


namespace inequality_range_l23_23341

theorem inequality_range (a : ℝ) (h : ∀ x : ℝ, |x - 3| + |x + 1| > a) : a < 4 := by
  sorry

end inequality_range_l23_23341


namespace total_wheels_l23_23434

def regular_bikes := 7
def children_bikes := 11
def tandem_bikes_4_wheels := 5
def tandem_bikes_6_wheels := 3
def unicycles := 4
def tricycles := 6
def bikes_with_training_wheels := 8

def wheels_regular := 2
def wheels_children := 4
def wheels_tandem_4 := 4
def wheels_tandem_6 := 6
def wheel_unicycle := 1
def wheels_tricycle := 3
def wheels_training := 4

theorem total_wheels : 
  (regular_bikes * wheels_regular) +
  (children_bikes * wheels_children) + 
  (tandem_bikes_4_wheels * wheels_tandem_4) + 
  (tandem_bikes_6_wheels * wheels_tandem_6) + 
  (unicycles * wheel_unicycle) + 
  (tricycles * wheels_tricycle) + 
  (bikes_with_training_wheels * wheels_training) 
  = 150 := by
  sorry

end total_wheels_l23_23434


namespace triangle_equilateral_l23_23840

noncomputable def is_equilateral_triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
  a = b ∧ b = c ∧ A = B ∧ B = C

theorem triangle_equilateral 
  (a b c A B C : ℝ) 
  (h1 : (a + b + c) * (b + c - a) = 3 * a * b * c) 
  (h2 : Real.sin A = 2 * Real.sin B * Real.cos C) : 
  is_equilateral_triangle a b c A B C :=
sorry

end triangle_equilateral_l23_23840


namespace problem_statement_l23_23556

theorem problem_statement (a n : ℕ) (h_a : a ≥ 1) (h_n : n ≥ 1) :
  (∃ k : ℕ, (a + 1)^n - a^n = k * n) ↔ n = 1 := by
  sorry

end problem_statement_l23_23556


namespace exists_integer_a_l23_23516

theorem exists_integer_a (p : ℕ) (hp : p ≥ 5) [Fact (Nat.Prime p)] : 
  ∃ a : ℕ, 1 ≤ a ∧ a ≤ p - 2 ∧ (¬ p^2 ∣ a^(p-1) - 1) ∧ (¬ p^2 ∣ (a+1)^(p-1) - 1) :=
by
  sorry

end exists_integer_a_l23_23516


namespace find_m_2n_3k_l23_23410

def is_prime (p : ℕ) : Prop := Nat.Prime p

theorem find_m_2n_3k (m n k : ℕ) (h1 : m + n = 2021) (h2 : is_prime (m - 3 * k)) (h3 : is_prime (n + k)) :
  m + 2 * n + 3 * k = 2025 ∨ m + 2 * n + 3 * k = 4040 := by
  sorry

end find_m_2n_3k_l23_23410


namespace find_a_l23_23844

def polys_are_integers (a b : Int) : Prop :=
  a ≠ 0 ∧ b ≠ 0

def factor_condition (a b : Int) : Prop :=
  ∀ x : ℝ, (x^2 - x - 1 = 0) → (a * x^19 + b * x^18 + 1 = 0)

theorem find_a (a b : Int) (h : polys_are_integers a b) (h_fac : factor_condition a b) : a = 1597 :=
by
  sorry

end find_a_l23_23844


namespace cori_age_proof_l23_23772

theorem cori_age_proof:
  ∃ (x : ℕ), (3 + x = (1 / 3) * (19 + x)) ∧ x = 5 :=
by
  sorry

end cori_age_proof_l23_23772


namespace minimum_harmonic_sum_l23_23580

theorem minimum_harmonic_sum
  (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0)
  (h_sum : a + b + c = 2) :
  (1 / a + 1 / b + 1 / c) ≥ 9 / 2 :=
by
  sorry

end minimum_harmonic_sum_l23_23580


namespace average_class_score_l23_23036

theorem average_class_score : 
  ∀ (n total score_per_100 score_per_0 avg_rest : ℕ), 
  n = 20 → 
  total = 800 → 
  score_per_100 = 2 → 
  score_per_0 = 3 → 
  avg_rest = 40 → 
  ((score_per_100 * 100 + score_per_0 * 0 + (n - (score_per_100 + score_per_0)) * avg_rest) / n = 40)
:= by
  intros n total score_per_100 score_per_0 avg_rest h_n h_total h_100 h_0 h_rest
  sorry

end average_class_score_l23_23036


namespace remaining_stock_weighs_120_l23_23046

noncomputable def total_remaining_weight (green_beans_weight rice_weight sugar_weight : ℕ) :=
  let remaining_rice := rice_weight - (rice_weight / 3)
  let remaining_sugar := sugar_weight - (sugar_weight / 5)
  let remaining_stock := remaining_rice + remaining_sugar + green_beans_weight
  remaining_stock

theorem remaining_stock_weighs_120 : total_remaining_weight 60 30 50 = 120 :=
by
  have h1: 60 - 30 = 30 := by norm_num
  have h2: 60 - 10 = 50 := by norm_num
  have h3: 30 - (30 / 3) = 20 := by norm_num
  have h4: 50 - (50 / 5) = 40 := by norm_num
  have h5: 20 + 40 + 60 = 120 := by norm_num
  exact h5

end remaining_stock_weighs_120_l23_23046


namespace quadratic_complete_square_l23_23980

theorem quadratic_complete_square:
  ∃ (a b c : ℝ), (∀ (x : ℝ), 3 * x^2 + 9 * x - 81 = a * (x + b) * (x + b) + c) ∧ a + b + c = -83.25 :=
by {
  sorry
}

end quadratic_complete_square_l23_23980


namespace max_visible_sum_is_128_l23_23541

-- Define the structure of the problem
structure Cube :=
  (faces : Fin 6 → Nat)
  (bottom_face : Nat)
  (all_faces : ∀ i : Fin 6, i ≠ ⟨0, by decide⟩ → faces i = bottom_face → False)

-- Define the problem conditions
noncomputable def problem_conditions : Prop :=
  let cubes := [Cube.mk (fun i => [1, 3, 5, 7, 9, 11].get i) 1 sorry,
                Cube.mk (fun i => [1, 3, 5, 7, 9, 11].get i) 1 sorry,
                Cube.mk (fun i => [1, 3, 5, 7, 9, 11].get i) 1 sorry,
                Cube.mk (fun i => [1, 3, 5, 7, 9, 11].get i) 1 sorry]
  -- Cube stacking in two layers, with two cubes per layer
  
  true

-- Define the theorem to be proved
theorem max_visible_sum_is_128 (h : problem_conditions) : 
  ∃ (total_sum : Nat), total_sum = 128 := 
sorry

end max_visible_sum_is_128_l23_23541


namespace find_difference_of_roots_l23_23938

-- Define the conditions for the given problem
def larger_root_of_eq_1 (a : ℝ) : Prop :=
  (1998 * a) ^ 2 - 1997 * 1999 * a - 1 = 0

def smaller_root_of_eq_2 (b : ℝ) : Prop :=
  b ^ 2 + 1998 * b - 1999 = 0

-- Define the main problem with the proof obligation
theorem find_difference_of_roots (a b : ℝ) (h1: larger_root_of_eq_1 a) (h2: smaller_root_of_eq_2 b) : a - b = 2000 :=
sorry

end find_difference_of_roots_l23_23938


namespace find_ellipse_focus_l23_23439

theorem find_ellipse_focus :
  ∀ (a b : ℝ), a^2 = 5 → b^2 = 4 → 
  (∀ x y, (x^2)/(a^2) + (y^2)/(b^2) = 1) →
  ((∃ c : ℝ, c^2 = a^2 - b^2) ∧ (∃ x y, x = 0 ∧ (y = 1 ∨ y = -1))) :=
by
  sorry

end find_ellipse_focus_l23_23439


namespace quadratic_function_relation_l23_23052

theorem quadratic_function_relation 
  (y : ℝ → ℝ) 
  (y_def : ∀ x : ℝ, y x = x^2 + x + 1) 
  (y1 y2 y3 : ℝ) 
  (hA : y (-3) = y1) 
  (hB : y 2 = y2) 
  (hC : y (1/2) = y3) : 
  y3 < y1 ∧ y1 = y2 := 
sorry

end quadratic_function_relation_l23_23052


namespace bottles_needed_to_fill_large_bottle_l23_23673

def medium_bottle_ml : ℕ := 150
def large_bottle_ml : ℕ := 1200

theorem bottles_needed_to_fill_large_bottle : large_bottle_ml / medium_bottle_ml = 8 :=
by
  sorry

end bottles_needed_to_fill_large_bottle_l23_23673


namespace correct_calculation_l23_23533

variable (n : ℕ)
variable (h1 : 63 + n = 70)

theorem correct_calculation : 36 * n = 252 :=
by
  -- Here we will need the Lean proof, which we skip using sorry
  sorry

end correct_calculation_l23_23533


namespace ax0_eq_b_condition_l23_23301

theorem ax0_eq_b_condition (a b x0 : ℝ) (h : a < 0) : (ax0 = b) ↔ (∀ x : ℝ, (1/2 * a * x^2 - b * x) ≤ (1/2 * a * x0^2 - b * x0)) :=
sorry

end ax0_eq_b_condition_l23_23301


namespace find_vector_p_l23_23116

noncomputable def vector_proj (v u : ℝ × ℝ) : ℝ × ℝ :=
  let dot_uv := v.1 * u.1 + v.2 * u.2
  let dot_u := u.1 * u.1 + u.2 * u.2
  let scale := dot_uv / dot_u
  (scale * u.1, scale * u.2)

theorem find_vector_p :
  ∃ p : ℝ × ℝ,
    vector_proj (5, -2) p = p ∧
    vector_proj (2, 6) p = p ∧
    p = (14 / 73, 214 / 73) :=
by
  sorry

end find_vector_p_l23_23116


namespace larger_solution_of_quadratic_l23_23064

theorem larger_solution_of_quadratic :
  ∀ x y : ℝ, x^2 - 19 * x - 48 = 0 ∧ y^2 - 19 * y - 48 = 0 ∧ x ≠ y →
  max x y = 24 :=
by
  sorry

end larger_solution_of_quadratic_l23_23064


namespace paint_left_for_solar_system_l23_23795

-- Definitions for the paint used
def Mary's_paint := 3
def Mike's_paint := Mary's_paint + 2
def Lucy's_paint := 4

-- Total original amount of paint
def original_paint := 25

-- Total paint used by Mary, Mike, and Lucy
def total_paint_used := Mary's_paint + Mike's_paint + Lucy's_paint

-- Theorem stating the amount of paint left for the solar system
theorem paint_left_for_solar_system : (original_paint - total_paint_used) = 13 :=
by
  sorry

end paint_left_for_solar_system_l23_23795


namespace largest_digit_divisible_by_6_l23_23670

theorem largest_digit_divisible_by_6 :
  ∃ N : ℕ, N ≤ 9 ∧ (56780 + N) % 6 = 0 ∧ (∀ M : ℕ, M ≤ 9 → (M % 2 = 0 ∧ (56780 + M) % 3 = 0) → M ≤ N) :=
by
  sorry

end largest_digit_divisible_by_6_l23_23670


namespace profit_calculation_l23_23769

def totalProfit (totalMoney part1 interest1 interest2 time : ℕ) : ℕ :=
  let part2 := totalMoney - part1
  let interestFromPart1 := part1 * interest1 / 100 * time
  let interestFromPart2 := part2 * interest2 / 100 * time
  interestFromPart1 + interestFromPart2

theorem profit_calculation : 
  totalProfit 80000 70000 10 20 1 = 9000 :=
  by 
    -- Rather than providing a full proof, we insert 'sorry' as per the instruction.
    sorry

end profit_calculation_l23_23769


namespace quadratic_nonneg_for_all_t_l23_23745

theorem quadratic_nonneg_for_all_t (x y : ℝ) : 
  (y ≤ x + 1) → (y ≥ -x - 1) → (x ≥ y^2 / 4) → (∀ (t : ℝ), (|t| ≤ 1) → t^2 + y * t + x ≥ 0) :=
by
  intro h1 h2 h3 t ht
  sorry

end quadratic_nonneg_for_all_t_l23_23745


namespace tickets_sold_at_door_l23_23781

theorem tickets_sold_at_door :
  ∃ D : ℕ, ∃ A : ℕ, A + D = 800 ∧ (1450 * A + 2200 * D = 166400) ∧ D = 672 :=
by
  sorry

end tickets_sold_at_door_l23_23781


namespace no_nat_number_satisfies_l23_23838

theorem no_nat_number_satisfies (n : ℕ) : ¬ ((n^2 + 6 * n + 2019) % 100 = 0) :=
sorry

end no_nat_number_satisfies_l23_23838


namespace new_mean_after_adding_14_to_each_of_15_numbers_l23_23363

theorem new_mean_after_adding_14_to_each_of_15_numbers (avg : ℕ) (n : ℕ) (n_sum : ℕ) (new_sum : ℕ) :
  avg = 40 →
  n = 15 →
  n_sum = n * avg →
  new_sum = n_sum + n * 14 →
  new_sum / n = 54 :=
by
  intros h_avg h_n h_n_sum h_new_sum
  sorry

end new_mean_after_adding_14_to_each_of_15_numbers_l23_23363


namespace hannah_total_cost_l23_23626

def price_per_kg : ℝ := 5
def discount_rate : ℝ := 0.4
def kilograms : ℝ := 10

theorem hannah_total_cost :
  (price_per_kg * (1 - discount_rate)) * kilograms = 30 := 
by
  sorry

end hannah_total_cost_l23_23626


namespace find_missing_id_l23_23575

theorem find_missing_id
  (total_students : ℕ)
  (sample_size : ℕ)
  (known_ids : Finset ℕ)
  (k : ℕ)
  (missing_id : ℕ) : 
  total_students = 52 ∧ 
  sample_size = 4 ∧ 
  known_ids = {3, 29, 42} ∧ 
  k = total_students / sample_size ∧ 
  missing_id = 16 :=
by
  sorry

end find_missing_id_l23_23575


namespace range_of_d_l23_23407

variable {S : ℕ → ℝ} -- S is the sum of the series
variable {a : ℕ → ℝ} -- a is the arithmetic sequence

theorem range_of_d (d : ℝ) (h1 : a 3 = 12) (h2 : S 12 > 0) (h3 : S 13 < 0) :
  -24 / 7 < d ∧ d < -3 := sorry

end range_of_d_l23_23407


namespace cody_money_l23_23971

theorem cody_money (a b c d : ℕ) (h₁ : a = 45) (h₂ : b = 9) (h₃ : c = 19) (h₄ : d = a + b - c) : d = 35 :=
by
  rw [h₁, h₂, h₃] at h₄
  simp at h₄
  exact h₄

end cody_money_l23_23971


namespace minimum_knights_l23_23513

-- Definitions based on the conditions
def total_people := 1001
def is_knight (person : ℕ) : Prop := sorry -- Assume definition of knight
def is_liar (person : ℕ) : Prop := sorry    -- Assume definition of liar

-- Conditions
axiom next_to_each_knight_is_liar : ∀ (p : ℕ), is_knight p → is_liar (p + 1) ∨ is_liar (p - 1)
axiom next_to_each_liar_is_knight : ∀ (p : ℕ), is_liar p → is_knight (p + 1) ∨ is_knight (p - 1)

-- Proving the minimum number of knights
theorem minimum_knights : ∃ (k : ℕ), k ≤ total_people ∧ k ≥ 502 ∧ (∀ (n : ℕ), n ≥ k → is_knight n) :=
  sorry

end minimum_knights_l23_23513


namespace seq_proof_l23_23722
noncomputable def seq1_arithmetic (a1 a2 : ℝ) : Prop :=
  ∃ d : ℝ, a1 = -2 + d ∧ a2 = a1 + d ∧ -8 = a2 + d

noncomputable def seq2_geometric (b1 b2 b3 : ℝ) : Prop :=
  ∃ r : ℝ, b1 = -2 * r ∧ b2 = b1 * r ∧ b3 = b2 * r ∧ -8 = b3 * r

theorem seq_proof (a1 a2 b1 b2 b3: ℝ) (h1 : seq1_arithmetic a1 a2) (h2 : seq2_geometric b1 b2 b3) :
  (a2 - a1) / b2 = 1 / 2 :=
sorry

end seq_proof_l23_23722


namespace gcd_m_n_l23_23092

-- Define the numbers m and n
def m : ℕ := 555555555
def n : ℕ := 1111111111

-- State the problem: Prove that gcd(m, n) = 1
theorem gcd_m_n : Nat.gcd m n = 1 :=
by
  -- Proof goes here
  sorry

end gcd_m_n_l23_23092


namespace correct_answer_l23_23267

def g (x : ℤ) : ℤ := x^3
def f (x : ℤ) : ℤ := 3*x - 2

theorem correct_answer : f (g 3) = 79 := by
  sorry

end correct_answer_l23_23267


namespace cookie_ratio_l23_23698

theorem cookie_ratio (f : ℚ) (h_monday : 32 = 32) (h_tuesday : (f : ℚ) * 32 = 32 * (f : ℚ)) 
    (h_wednesday : 3 * (f : ℚ) * 32 - 4 + 32 + (f : ℚ) * 32 = 92) :
    f = 1/2 :=
by
  sorry

end cookie_ratio_l23_23698


namespace cyclist_speed_l23_23400

theorem cyclist_speed 
  (course_length : ℝ)
  (second_cyclist_speed : ℝ)
  (meeting_time : ℝ)
  (total_distance : ℝ)
  (condition1 : course_length = 45)
  (condition2 : second_cyclist_speed = 16)
  (condition3 : meeting_time = 1.5)
  (condition4 : total_distance = meeting_time * (second_cyclist_speed + 14))
  : (meeting_time * 14 + meeting_time * second_cyclist_speed = course_length) :=
by
  sorry

end cyclist_speed_l23_23400


namespace rhombus_angles_l23_23657

-- Define the conditions for the proof
variables (a e f : ℝ) (α β : ℝ)

-- Using the geometric mean condition
def geometric_mean_condition := a^2 = e * f

-- Using the condition that diagonals of a rhombus intersect at right angles and bisect each other
def diagonals_intersect_perpendicularly := α + β = 180 ∧ α = 30 ∧ β = 150

-- Prove the question assuming the given conditions
theorem rhombus_angles (h1 : geometric_mean_condition a e f) (h2 : diagonals_intersect_perpendicularly α β) : 
  (α = 30) ∧ (β = 150) :=
sorry

end rhombus_angles_l23_23657


namespace missing_number_l23_23873

theorem missing_number (mean : ℝ) (numbers : List ℝ) (x : ℝ) (h_mean : mean = 14.2) (h_numbers : numbers = [13.0, 8.0, 13.0, 21.0, 23.0]) :
  (numbers.sum + x) / (numbers.length + 1) = mean → x = 7.2 :=
by
  -- states the hypothesis about the mean calculation into the theorem structure
  intro h
  sorry

end missing_number_l23_23873


namespace min_a_3b_l23_23991

theorem min_a_3b (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : (1 / (a + 3) + 1 / (b + 3) = 1 / 4)) : 
  a + 3*b ≥ 12 + 16*Real.sqrt 3 :=
by sorry

end min_a_3b_l23_23991


namespace stream_speed_l23_23729

/-- The speed of the stream problem -/
theorem stream_speed 
    (b s : ℝ) 
    (downstream_time : ℝ := 3)
    (upstream_time : ℝ := 3)
    (downstream_distance : ℝ := 60)
    (upstream_distance : ℝ := 30)
    (h1 : downstream_distance = (b + s) * downstream_time)
    (h2 : upstream_distance = (b - s) * upstream_time) : 
    s = 5 := 
by {
  -- The proof can be filled here
  sorry
}

end stream_speed_l23_23729


namespace train_time_to_pass_bridge_l23_23687

theorem train_time_to_pass_bridge
  (length_train : ℝ) (length_bridge : ℝ) (speed_kmph : ℝ)
  (h1 : length_train = 500) (h2 : length_bridge = 200) (h3 : speed_kmph = 72) :
  let speed_mps := speed_kmph * 1000 / 3600
  let total_distance := length_train + length_bridge
  let time := total_distance / speed_mps
  time = 35 :=
by
  sorry

end train_time_to_pass_bridge_l23_23687


namespace percentage_increased_is_correct_l23_23269

-- Define the initial and final numbers
def initial_number : Nat := 150
def final_number : Nat := 210

-- Define the function to compute the percentage increase
def percentage_increase (initial final : Nat) : Float :=
  ((final - initial).toFloat / initial.toFloat) * 100.0

-- The theorem we need to prove
theorem percentage_increased_is_correct :
  percentage_increase initial_number final_number = 40 := 
by
  simp [percentage_increase, initial_number, final_number]
  sorry

end percentage_increased_is_correct_l23_23269


namespace sqrt_D_always_irrational_l23_23808

-- Definitions for consecutive even integers and D
def is_consecutive_even (p q : ℤ) : Prop :=
  ∃ k : ℤ, p = 2 * k ∧ q = 2 * k + 2

def D (p q : ℤ) : ℤ :=
  p^2 + q^2 + p * q^2

-- The main statement to prove
theorem sqrt_D_always_irrational (p q : ℤ) (h : is_consecutive_even p q) :
  ¬ ∃ r : ℤ, r * r = D p q :=
sorry

end sqrt_D_always_irrational_l23_23808


namespace noelle_speed_l23_23661

theorem noelle_speed (v d : ℝ) (h1 : d > 0) (h2 : v > 0) 
  (h3 : (2 * d) / ((d / v) + (d / 15)) = 5) : v = 3 := 
sorry

end noelle_speed_l23_23661


namespace required_brick_volume_l23_23704

theorem required_brick_volume :
  let height := 4 / 12 -- in feet
  let length := 6 -- in feet
  let thickness := 4 / 12 -- in feet
  let volume := height * length * thickness
  let rounded_volume := Nat.ceil volume
  rounded_volume = 1 := 
by
  let height := 1 / 3
  let length := 6
  let thickness := 1 / 3
  let volume := height * length * thickness
  let rounded_volume := Nat.ceil volume
  show rounded_volume = 1
  sorry

end required_brick_volume_l23_23704


namespace chemistry_marks_l23_23168

-- Definitions based on given conditions
def total_marks (P C M : ℕ) : Prop := P + C + M = 210
def avg_physics_math (P M : ℕ) : Prop := (P + M) / 2 = 90
def physics_marks (P : ℕ) : Prop := P = 110
def avg_physics_other_subject (P C : ℕ) : Prop := (P + C) / 2 = 70

-- The proof problem statement
theorem chemistry_marks {P C M : ℕ} (h1 : total_marks P C M) (h2 : avg_physics_math P M) (h3 : physics_marks P) : C = 30 ∧ avg_physics_other_subject P C :=
by 
  -- Proof goes here
  sorry

end chemistry_marks_l23_23168


namespace probability_both_counterfeit_given_one_counterfeit_l23_23250

-- Conditions
def total_bills := 20
def counterfeit_bills := 5
def selected_bills := 2
def at_least_one_counterfeit := true

-- Definition of events
def eventA := "both selected bills are counterfeit"
def eventB := "at least one of the selected bills is counterfeit"

-- The theorem to prove
theorem probability_both_counterfeit_given_one_counterfeit : 
  at_least_one_counterfeit →
  ( (counterfeit_bills * (counterfeit_bills - 1)) / (total_bills * (total_bills - 1)) ) / 
    ( (counterfeit_bills * (counterfeit_bills - 1) + counterfeit_bills * (total_bills - counterfeit_bills)) / (total_bills * (total_bills - 1)) ) = 2/17 :=
by
  sorry

end probability_both_counterfeit_given_one_counterfeit_l23_23250


namespace simplify_expression_l23_23981

-- Define the constants and variables with required conditions
variables {x y z p q r : ℝ}

-- Assume the required distinctness conditions
axiom h1 : x ≠ p 
axiom h2 : y ≠ q 
axiom h3 : z ≠ r 

-- State the theorem to be proven
theorem simplify_expression (h : p ≠ q ∧ q ≠ r ∧ r ≠ p) : 
  (2 * (x - p) / (3 * (r - z))) * (2 * (y - q) / (3 * (p - x))) * (2 * (z - r) / (3 * (q - y))) = -8 / 27 :=
  sorry

end simplify_expression_l23_23981


namespace determine_prices_l23_23634

variable (num_items : ℕ) (cost_keychains cost_plush : ℕ) (x : ℚ) (unit_price_keychains unit_price_plush : ℚ)

noncomputable def price_equation (x : ℚ) : Prop :=
  (cost_keychains / x) + (cost_plush / (1.5 * x)) = num_items

theorem determine_prices 
  (h1 : num_items = 15)
  (h2 : cost_keychains = 240)
  (h3 : cost_plush = 180)
  (h4 : price_equation num_items cost_keychains cost_plush x)
  (hx : x = 24) :
  unit_price_keychains = 24 ∧ unit_price_plush = 36 :=
  by
    sorry

end determine_prices_l23_23634


namespace cat_birds_total_l23_23952

def day_birds : ℕ := 8
def night_birds : ℕ := 2 * day_birds
def total_birds : ℕ := day_birds + night_birds

theorem cat_birds_total : total_birds = 24 :=
by
  -- proof goes here
  sorry

end cat_birds_total_l23_23952


namespace find_point_A_l23_23342

theorem find_point_A (x : ℝ) (h : x + 7 - 4 = 0) : x = -3 :=
sorry

end find_point_A_l23_23342


namespace largest_sum_is_three_fourths_l23_23194

-- Definitions of sums
def sum1 := (1 / 4) + (1 / 2)
def sum2 := (1 / 4) + (1 / 9)
def sum3 := (1 / 4) + (1 / 3)
def sum4 := (1 / 4) + (1 / 10)
def sum5 := (1 / 4) + (1 / 6)

-- The theorem stating that sum1 is the maximum of the sums
theorem largest_sum_is_three_fourths : max (max (max (max sum1 sum2) sum3) sum4) sum5 = 3 / 4 := 
sorry

end largest_sum_is_three_fourths_l23_23194


namespace simplify_trig_expression_tan_alpha_value_l23_23020

-- Proof Problem (1)
theorem simplify_trig_expression :
  (∃ θ : ℝ, θ = (20:ℝ) ∧ 
    (∃ α : ℝ, α = (160:ℝ) ∧ 
      (∃ β : ℝ, β = 1 - 2 * (Real.sin θ) * (Real.cos θ) ∧ 
        (∃ γ : ℝ, γ = 1 - (Real.sin θ)^2 ∧ 
          (Real.sqrt β) / ((Real.sin α) - (Real.sqrt γ)) = -1)))) :=
sorry

-- Proof Problem (2)
theorem tan_alpha_value (α : ℝ) (h : Real.tan α = 1 / 3) :
  1 / (4 * (Real.cos α)^2 - 6 * (Real.sin α) * (Real.cos α)) = 5 / 9 :=
sorry

end simplify_trig_expression_tan_alpha_value_l23_23020


namespace paint_coverage_l23_23248

-- Define the conditions
def cost_per_gallon : ℝ := 45
def total_area : ℝ := 1600
def number_of_coats : ℝ := 2
def total_contribution : ℝ := 180 + 180

-- Define the target statement to prove
theorem paint_coverage (H : total_contribution = 360) : 
  let cost_per_gallon := 45 
  let number_of_gallons := total_contribution / cost_per_gallon
  let total_coverage := total_area * number_of_coats
  let coverage_per_gallon := total_coverage / number_of_gallons
  coverage_per_gallon = 400 :=
by
  sorry

end paint_coverage_l23_23248


namespace length_of_greater_segment_l23_23984

-- Definitions based on conditions
variable (shorter longer : ℝ)
variable (h1 : longer = shorter + 2)
variable (h2 : (longer^2) - (shorter^2) = 32)

-- Proof goal
theorem length_of_greater_segment : longer = 9 :=
by
  sorry

end length_of_greater_segment_l23_23984


namespace rational_sqrt_of_rational_xy_l23_23318

theorem rational_sqrt_of_rational_xy (x y : ℚ) (h : x^5 + y^5 = 2 * x^2 * y^2) :
  ∃ k : ℚ, k^2 = 1 - x * y := 
sorry

end rational_sqrt_of_rational_xy_l23_23318


namespace y_intercept_is_2_l23_23731

def equation_of_circle (x y : ℝ) : Prop := x^2 + y^2 = 2

def point_P : ℝ × ℝ := (-1, 1)

def y_intercept_of_tangent_line (m c x y : ℝ) : Prop :=
  equation_of_circle x y ∧
  ((y = m * x + c) ∧ (point_P.1, point_P.2) ∈ {(x, y) | y = m * x + c})

theorem y_intercept_is_2 :
  ∃ m c : ℝ, y_intercept_of_tangent_line m c 0 2 :=
sorry

end y_intercept_is_2_l23_23731


namespace zoe_bought_bottles_l23_23617

theorem zoe_bought_bottles
  (initial_bottles : ℕ)
  (drank_bottles : ℕ)
  (current_bottles : ℕ)
  (initial_bottles_eq : initial_bottles = 42)
  (drank_bottles_eq : drank_bottles = 25)
  (current_bottles_eq : current_bottles = 47) :
  ∃ bought_bottles : ℕ, bought_bottles = 30 :=
by
  sorry

end zoe_bought_bottles_l23_23617


namespace max_product_two_integers_l23_23437

theorem max_product_two_integers (x y : ℤ) (h : x + y = 300) : x * y ≤ 22500 :=
sorry

end max_product_two_integers_l23_23437


namespace kite_area_l23_23681

theorem kite_area {length height : ℕ} (h_length : length = 8) (h_height : height = 10): 
  2 * (1/2 * (length * 2) * (height * 2 / 2)) = 160 :=
by
  rw [h_length, h_height]
  norm_num
  sorry

end kite_area_l23_23681


namespace max_ab_l23_23461

theorem max_ab (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + 4 * b = 1) : ab ≤ 1 / 16 :=
sorry

end max_ab_l23_23461


namespace common_remainder_is_zero_l23_23697

noncomputable def least_number := 100040

theorem common_remainder_is_zero 
  (n : ℕ) 
  (h1 : n = least_number) 
  (condition1 : 4 ∣ n)
  (condition2 : 610 ∣ n)
  (condition3 : 15 ∣ n)
  (h2 : (n.digits 10).sum = 5)
  : ∃ r : ℕ, ∀ (a : ℕ), (a ∈ [4, 610, 15] → n % a = r) ∧ r = 0 :=
by {
  sorry
}

end common_remainder_is_zero_l23_23697


namespace outfit_choices_l23_23921

noncomputable def calculate_outfits : Nat :=
  let shirts := 6
  let pants := 6
  let hats := 6
  let total_outfits := shirts * pants * hats
  let matching_colors := 4 -- tan, black, blue, gray for matching
  total_outfits - matching_colors

theorem outfit_choices : calculate_outfits = 212 :=
by
  sorry

end outfit_choices_l23_23921


namespace trapezoid_area_l23_23182

theorem trapezoid_area (A_outer A_inner : ℝ) (n : ℕ)
  (h_outer : A_outer = 36)
  (h_inner : A_inner = 4)
  (h_n : n = 4) :
  (A_outer - A_inner) / n = 8 := by
  sorry

end trapezoid_area_l23_23182


namespace number_of_homes_cleaned_l23_23641

-- Define constants for the amount Mary earns per home and the total amount she made.
def amount_per_home := 46
def total_amount_made := 276

-- Prove that the number of homes Mary cleaned is 6 given the conditions.
theorem number_of_homes_cleaned : total_amount_made / amount_per_home = 6 :=
by
  sorry

end number_of_homes_cleaned_l23_23641


namespace amusement_park_admission_l23_23646

def number_of_children (children_fee : ℤ) (adults_fee : ℤ) (total_people : ℤ) (total_fees : ℤ) : ℤ :=
  let y := (total_fees - total_people * children_fee) / (adults_fee - children_fee)
  total_people - y

theorem amusement_park_admission :
  number_of_children 15 40 315 8100 = 180 :=
by
  -- Fees in cents to avoid decimals
  sorry  -- Placeholder for the proof

end amusement_park_admission_l23_23646


namespace polygon_interior_exterior_equal_l23_23694

theorem polygon_interior_exterior_equal (n : ℕ) :
  (n - 2) * 180 = 360 → n = 4 :=
by
  sorry

end polygon_interior_exterior_equal_l23_23694


namespace kayden_total_processed_l23_23596

-- Definition of the given conditions and final proof problem statement in Lean 4
variable (x : ℕ)  -- x is the number of cartons delivered to each customer

theorem kayden_total_processed (h : 4 * (x - 60) = 160) : 4 * x = 400 :=
by
  sorry

end kayden_total_processed_l23_23596


namespace tan_alpha_beta_l23_23978

noncomputable def tan_alpha := -1 / 3
noncomputable def cos_beta := (Real.sqrt 5) / 5
noncomputable def beta := (1:ℝ) -- Dummy representation for being in first quadrant

theorem tan_alpha_beta (h1 : tan_alpha = -1 / 3) 
                       (h2 : cos_beta = (Real.sqrt 5) / 5) 
                       (h3 : 0 < beta ∧ beta < Real.pi / 2) : 
  Real.tan (α + β) = 1 := 
sorry

end tan_alpha_beta_l23_23978


namespace sales_tax_difference_l23_23392

theorem sales_tax_difference : 
  let price : Float := 50
  let tax1 : Float := 0.0725
  let tax2 : Float := 0.07
  let sales_tax1 := price * tax1
  let sales_tax2 := price * tax2
  sales_tax1 - sales_tax2 = 0.125 := 
by
  sorry

end sales_tax_difference_l23_23392


namespace int_even_bijection_l23_23909

theorem int_even_bijection :
  ∃ (f : ℤ → ℤ), (∀ n : ℤ, ∃ m : ℤ, f n = m ∧ m % 2 = 0) ∧
                 (∀ m : ℤ, m % 2 = 0 → ∃ n : ℤ, f n = m) := 
sorry

end int_even_bijection_l23_23909


namespace abigail_time_to_finish_l23_23692

noncomputable def words_total : ℕ := 1000
noncomputable def words_per_30_min : ℕ := 300
noncomputable def words_already_written : ℕ := 200
noncomputable def time_per_word : ℝ := 30 / words_per_30_min

theorem abigail_time_to_finish :
  (words_total - words_already_written) * time_per_word = 80 :=
by
  sorry

end abigail_time_to_finish_l23_23692


namespace max_value_x_minus_2y_l23_23090

open Real

theorem max_value_x_minus_2y (x y : ℝ) (h : x^2 + y^2 - 2*x + 4*y = 0) : 
  x - 2*y ≤ 10 :=
sorry

end max_value_x_minus_2y_l23_23090


namespace point_on_transformed_graph_l23_23107

theorem point_on_transformed_graph 
  (f : ℝ → ℝ)
  (h1 : f 12 = 5)
  (x y : ℝ)
  (h2 : 1.5 * y = (f (3 * x) + 3) / 3)
  (point_x : x = 4)
  (point_y : y = 16 / 9) 
  : x + y = 52 / 9 :=
by
  sorry

end point_on_transformed_graph_l23_23107


namespace correct_operation_l23_23309

theorem correct_operation (a b : ℝ) : (-a * b^2)^2 = a^2 * b^4 := 
by 
  sorry

end correct_operation_l23_23309


namespace triangleProblem_correct_l23_23760

noncomputable def triangleProblem : Prop :=
  ∃ (a b c A B C : ℝ),
    A = 60 * Real.pi / 180 ∧
    b = 1 ∧
    (1 / 2) * b * c * Real.sin A = Real.sqrt 3 ∧
    Real.cos A = 1 / 2 ∧
    a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos A ∧
    (a / Real.sin A) = (b / Real.sin B) ∧ (b / Real.sin B) = (c / Real.sin C) ∧
    (a + b + c) / (Real.sin A + Real.sin B + Real.sin C) = 2 * Real.sqrt 39 / 3

theorem triangleProblem_correct : triangleProblem :=
  sorry

end triangleProblem_correct_l23_23760


namespace part_I_part_II_l23_23446

noncomputable def f (x a : ℝ) : ℝ := |x + 1| - |x - a|

theorem part_I (x : ℝ) : (∃ a : ℝ, a = 1 ∧ f x a < 1) ↔ x < (1/2) :=
sorry

theorem part_II (a : ℝ) : (∀ x : ℝ, f x a ≤ 6) ↔ (a = 5 ∨ a = -7) :=
sorry

end part_I_part_II_l23_23446


namespace total_length_of_rubber_pen_pencil_l23_23146

variable (rubber pen pencil : ℕ)

theorem total_length_of_rubber_pen_pencil 
  (h1 : pen = rubber + 3)
  (h2 : pen = pencil - 2)
  (h3 : pencil = 12) : rubber + pen + pencil = 29 := by
  sorry

end total_length_of_rubber_pen_pencil_l23_23146


namespace parabola_directrix_l23_23416

theorem parabola_directrix (y x : ℝ) (h : y^2 = -4 * x) : x = 1 :=
sorry

end parabola_directrix_l23_23416


namespace books_from_second_shop_l23_23920

theorem books_from_second_shop (x : ℕ) (h₁ : 6500 + 2000 = 8500)
    (h₂ : 85 = 8500 / (65 + x)) : x = 35 :=
by
  -- proof goes here
  sorry

end books_from_second_shop_l23_23920


namespace smallest_n_divisibility_l23_23621

theorem smallest_n_divisibility:
  ∃ (n : ℕ), n > 0 ∧ n^2 % 24 = 0 ∧ n^3 % 540 = 0 ∧ n = 60 :=
by
  sorry

end smallest_n_divisibility_l23_23621


namespace evaluate_expression_l23_23782

theorem evaluate_expression (a b c : ℝ) : 
  (a / (30 - a) + b / (70 - b) + c / (75 - c) = 9) → 
  (6 / (30 - a) + 14 / (70 - b) + 15 / (75 - c) = 2.4) :=
by 
  sorry

end evaluate_expression_l23_23782


namespace can_place_circles_l23_23241

theorem can_place_circles (r: ℝ) (h: r = 2008) :
  ∃ (n: ℕ), (n > 4016) ∧ ((n: ℝ) / 2 > r) :=
by 
  sorry

end can_place_circles_l23_23241


namespace abs_inequality_equiv_l23_23967

theorem abs_inequality_equiv (x : ℝ) : 1 ≤ |x - 2| ∧ |x - 2| ≤ 7 ↔ (-5 ≤ x ∧ x ≤ 1) ∨ (3 ≤ x ∧ x ≤ 9) :=
by
  sorry

end abs_inequality_equiv_l23_23967


namespace bill_profit_difference_l23_23115

theorem bill_profit_difference (P SP NSP NP : ℝ) 
  (h1 : SP = 1.10 * P)
  (h2 : SP = 659.9999999999994)
  (h3 : NP = 0.90 * P)
  (h4 : NSP = 1.30 * NP) :
  NSP - SP = 42 := 
sorry

end bill_profit_difference_l23_23115


namespace min_abs_sum_l23_23459

theorem min_abs_sum : ∃ x : ℝ, (|x + 1| + |x + 2| + |x + 6|) = 5 :=
sorry

end min_abs_sum_l23_23459


namespace problem_remainders_l23_23419

open Int

theorem problem_remainders (x : ℤ) :
  (x + 2) % 45 = 7 →
  ((x + 2) % 20 = 7 ∧ x % 19 = 5) :=
by
  sorry

end problem_remainders_l23_23419


namespace height_of_pyramid_l23_23726

-- Define the volumes
def volume_cube (s : ℕ) : ℕ := s^3
def volume_pyramid (b : ℕ) (h : ℕ) : ℕ := (b^2 * h) / 3

-- Given constants
def s := 6
def b := 12

-- Given volume equality
def volumes_equal (s : ℕ) (b : ℕ) (h : ℕ) : Prop :=
  volume_cube s = volume_pyramid b h

-- The statement to prove
theorem height_of_pyramid (h : ℕ) (h_eq : volumes_equal s b h) :
  h = 9 := sorry

end height_of_pyramid_l23_23726


namespace equal_abc_l23_23276

theorem equal_abc {a b c : ℝ} (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h : a^2 * (b + c - a) = b^2 * (c + a - b) ∧ 
       b^2 * (c + a - b) = c^2 * (a + b - c)) : a = b ∧ b = c :=
by
  sorry

end equal_abc_l23_23276


namespace count_positive_integers_l23_23016

theorem count_positive_integers (n : ℕ) (x : ℝ) (h1 : n ≤ 1500) :
  (∃ x : ℝ, n = ⌊x⌋ + ⌊3*x⌋ + ⌊5*x⌋) ↔ n = 668 :=
by
  sorry

end count_positive_integers_l23_23016


namespace shopkeeper_discount_problem_l23_23494

theorem shopkeeper_discount_problem (CP SP_with_discount SP_without_discount Discount : ℝ)
  (h1 : SP_with_discount = CP + 0.273 * CP)
  (h2 : SP_without_discount = CP + 0.34 * CP) :
  Discount = SP_without_discount - SP_with_discount →
  (Discount / SP_without_discount) * 100 = 5 := 
sorry

end shopkeeper_discount_problem_l23_23494


namespace student_ticket_price_l23_23951

theorem student_ticket_price
  (S : ℕ)
  (num_tickets : ℕ := 2000)
  (num_student_tickets : ℕ := 520)
  (price_non_student : ℕ := 11)
  (total_revenue : ℕ := 20960)
  (h : 520 * S + (2000 - 520) * 11 = 20960) :
  S = 9 :=
sorry

end student_ticket_price_l23_23951
