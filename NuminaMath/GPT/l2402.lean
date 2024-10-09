import Mathlib

namespace find_radius_of_circle_l2402_240257

noncomputable def central_angle := 150
noncomputable def arc_length := 5 * Real.pi
noncomputable def arc_length_formula (θ : ℝ) (r : ℝ) : ℝ :=
  (θ / 180) * Real.pi * r

theorem find_radius_of_circle :
  (∃ r : ℝ, arc_length_formula central_angle r = arc_length) ↔ 6 = 6 :=
by  
  sorry

end find_radius_of_circle_l2402_240257


namespace perpendicular_tangent_line_l2402_240297

theorem perpendicular_tangent_line :
  ∃ m : ℝ, ∃ x₀ : ℝ, y₀ = x₀ ^ 3 + 3 * x₀ ^ 2 - 1 ∧ y₀ = -3 * x₀ + m ∧ 
  (∀ x, x ≠ x₀ → x ^ 3 + 3 * x ^ 2 - 1 ≠ -3 * x + m) ∧ m = -2 := 
sorry

end perpendicular_tangent_line_l2402_240297


namespace tate_total_years_l2402_240278

-- Define the conditions
def high_school_years : Nat := 3
def gap_years : Nat := 2
def bachelor_years : Nat := 2 * high_school_years
def certification_years : Nat := 1
def work_experience_years : Nat := 1
def master_years : Nat := bachelor_years / 2
def phd_years : Nat := 3 * (high_school_years + bachelor_years + master_years)

-- Define the total years Tate spent
def total_years : Nat :=
  high_school_years + gap_years +
  bachelor_years + certification_years +
  work_experience_years + master_years + phd_years

-- State the theorem
theorem tate_total_years : total_years = 52 := by
  sorry

end tate_total_years_l2402_240278


namespace solve_abs_ineq_l2402_240219

theorem solve_abs_ineq (x : ℝ) (h : x > 0) : |4 * x - 5| < 8 ↔ 0 < x ∧ x < 13 / 4 :=
by
  sorry

end solve_abs_ineq_l2402_240219


namespace find_x_base_l2402_240292

open Nat

def is_valid_digit (n : ℕ) : Prop := n < 10

def interpret_base (digits : ℕ → ℕ) (n : ℕ) : ℕ :=
  digits 2 * n^2 + digits 1 * n + digits 0

theorem find_x_base (a b c : ℕ)
  (ha : is_valid_digit a)
  (hb : is_valid_digit b)
  (hc : is_valid_digit c)
  (h : interpret_base (fun i => if i = 0 then c else if i = 1 then b else a) 20 = 2 * interpret_base (fun i => if i = 0 then c else if i = 1 then b else a) 13) :
  100 * a + 10 * b + c = 198 :=
by
  sorry

end find_x_base_l2402_240292


namespace sum_of_three_numbers_l2402_240212

-- Definitions for the conditions
def mean_condition_1 (x y z : ℤ) := (x + y + z) / 3 = x + 20
def mean_condition_2 (x y z : ℤ) := (x + y + z) / 3 = z - 18
def median_condition (y : ℤ) := y = 9

-- The Lean 4 statement to prove the sum of x, y, and z is 21
theorem sum_of_three_numbers (x y z : ℤ) 
  (h1 : mean_condition_1 x y z) 
  (h2 : mean_condition_2 x y z) 
  (h3 : median_condition y) : 
  x + y + z = 21 := 
  by 
    sorry

end sum_of_three_numbers_l2402_240212


namespace second_reduction_percentage_is_4_l2402_240270

def original_price := 500
def first_reduction_percent := 5 / 100
def total_reduction := 44

def first_reduction := first_reduction_percent * original_price
def price_after_first_reduction := original_price - first_reduction
def second_reduction := total_reduction - first_reduction
def second_reduction_percent := (second_reduction / price_after_first_reduction) * 100

theorem second_reduction_percentage_is_4 :
  second_reduction_percent = 4 := by
  sorry

end second_reduction_percentage_is_4_l2402_240270


namespace cost_of_each_item_l2402_240289

theorem cost_of_each_item 
  (x y z : ℝ) 
  (h1 : 3 * x + 5 * y + z = 32)
  (h2 : 4 * x + 7 * y + z = 40) : 
  x + y + z = 16 :=
by 
  sorry

end cost_of_each_item_l2402_240289


namespace total_cost_eq_l2402_240273

noncomputable def total_cost : Real :=
  let os_overhead := 1.07
  let cost_per_millisecond := 0.023
  let tape_mounting_cost := 5.35
  let cost_per_megabyte := 0.15
  let cost_per_kwh := 0.02
  let technician_rate_per_hour := 50.0
  let minutes_to_milliseconds := 60000
  let gb_to_mb := 1024

  -- Define program specifics
  let computer_time_minutes := 45.0
  let memory_gb := 3.5
  let electricity_kwh := 2.0
  let technician_time_minutes := 20.0

  -- Calculate costs
  let computer_time_cost := (computer_time_minutes * minutes_to_milliseconds * cost_per_millisecond)
  let memory_cost := (memory_gb * gb_to_mb * cost_per_megabyte)
  let electricity_cost := (electricity_kwh * cost_per_kwh)
  let technician_time_total_hours := (technician_time_minutes * 2 / 60.0)
  let technician_cost := (technician_time_total_hours * technician_rate_per_hour)

  os_overhead + computer_time_cost + tape_mounting_cost + memory_cost + electricity_cost + technician_cost

theorem total_cost_eq : total_cost = 62677.39 := by
  sorry

end total_cost_eq_l2402_240273


namespace jenny_cases_l2402_240286

theorem jenny_cases (total_boxes cases_per_box : ℕ) (h1 : total_boxes = 24) (h2 : cases_per_box = 8) :
  total_boxes / cases_per_box = 3 := by
  sorry

end jenny_cases_l2402_240286


namespace trigonometric_identity_l2402_240245

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = 1 / 2) :
  Real.sin (2 * θ) - 2 * Real.cos θ ^ 2 = -4 / 5 :=
by
  sorry

end trigonometric_identity_l2402_240245


namespace cube_surface_area_150_of_volume_125_l2402_240215

def volume (s : ℝ) : ℝ := s^3

def surface_area (s : ℝ) : ℝ := 6 * s^2

theorem cube_surface_area_150_of_volume_125 :
  ∀ (s : ℝ), volume s = 125 → surface_area s = 150 :=
by 
  intros s hs
  sorry

end cube_surface_area_150_of_volume_125_l2402_240215


namespace carlo_practice_difference_l2402_240265

-- Definitions for given conditions
def monday_practice (T : ℕ) : ℕ := 2 * T
def tuesday_practice (T : ℕ) : ℕ := T
def wednesday_practice (thursday_minutes : ℕ) : ℕ := thursday_minutes + 5
def thursday_practice : ℕ := 50
def friday_practice : ℕ := 60
def total_weekly_practice : ℕ := 300

theorem carlo_practice_difference 
  (T : ℕ) 
  (Monday Tuesday Wednesday Thursday Friday : ℕ) 
  (H1 : Monday = monday_practice T)
  (H2 : Tuesday = tuesday_practice T)
  (H3 : Wednesday = wednesday_practice Thursday)
  (H4 : Thursday = thursday_practice)
  (H5 : Friday = friday_practice)
  (H6 : Monday + Tuesday + Wednesday + Thursday + Friday = total_weekly_practice) :
  (Wednesday - Tuesday = 10) :=
by 
  -- Use the provided conditions and derive the required result.
  sorry

end carlo_practice_difference_l2402_240265


namespace mary_mileage_l2402_240205

def base9_to_base10 : Nat :=
  let d0 := 6 * 9^0
  let d1 := 5 * 9^1
  let d2 := 9 * 9^2
  let d3 := 3 * 9^3
  d0 + d1 + d2 + d3 

theorem mary_mileage :
  base9_to_base10 = 2967 :=
by 
  -- Calculation steps are skipped using sorry
  sorry

end mary_mileage_l2402_240205


namespace callie_caught_frogs_l2402_240263

theorem callie_caught_frogs (A Q B C : ℝ) 
  (hA : A = 2)
  (hQ : Q = 2 * A)
  (hB : B = 3 * Q)
  (hC : C = (5 / 8) * B) : 
  C = 7.5 := by
  sorry

end callie_caught_frogs_l2402_240263


namespace missing_digit_B_l2402_240259

theorem missing_digit_B (B : ℕ) (h : 0 ≤ B ∧ B ≤ 9) (h_div : (100 + 10 * B + 3) % 13 = 0) : B = 4 := 
by
  sorry

end missing_digit_B_l2402_240259


namespace value_of_b_l2402_240296

theorem value_of_b (a b : ℝ) (h1 : 3 * a + 2 = 2) (h2 : b - a = 1) : b = 1 := by
  sorry

end value_of_b_l2402_240296


namespace wally_not_all_numbers_l2402_240258

def next_wally_number (n : ℕ) : ℕ :=
  if n % 2 = 0 then
    n / 2
  else
    (n + 1001) / 2

def eventually_print(n: ℕ) : Prop :=
  ∃ k: ℕ, (next_wally_number^[k]) 1 = n

theorem wally_not_all_numbers :
  ¬ ∀ n, n ≤ 100 → eventually_print n :=
by
  sorry

end wally_not_all_numbers_l2402_240258


namespace largest_d_l2402_240217

variable (a b c d : ℝ)

theorem largest_d (h : a + 1 = b - 2 ∧ b - 2 = c + 3 ∧ c + 3 = d - 4) : 
  d >= a ∧ d >= b ∧ d >= c :=
by
  sorry

end largest_d_l2402_240217


namespace part_I_part_II_l2402_240221

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log x + a * x^2 - (2 * a + 1) * x

theorem part_I (a : ℝ) (ha : a = -2) : 
  (∃ x : ℝ, f a x = 1) ∧ ∀ x : ℝ, f a x ≤ 1 :=
by sorry

theorem part_II (a : ℝ) (ha : a < 1/2) :
  (∃ x : ℝ, 0 < x ∧ x < exp 1 ∧ f a x < 0) → a < (exp 1 - 1) / (exp 1 * (exp 1 - 2)) :=
by sorry

end part_I_part_II_l2402_240221


namespace ticket_cost_l2402_240279

open Real

-- Variables for ticket prices
variable (A C S : ℝ)

-- Given conditions
def cost_condition : Prop :=
  C = A / 2 ∧ S = A - 1.50 ∧ 6 * A + 5 * C + 3 * S = 40.50

-- The goal is to prove that the total cost for 10 adult tickets, 8 child tickets,
-- and 4 senior tickets is 64.38
theorem ticket_cost (h : cost_condition A C S) : 10 * A + 8 * C + 4 * S = 64.38 :=
by
  -- Implementation of the proof would go here
  sorry

end ticket_cost_l2402_240279


namespace rectangle_area_l2402_240253

variable {x : ℝ} (h : x > 0)

theorem rectangle_area (W : ℝ) (L : ℝ) (hL : L = 3 * W) (h_diag : W^2 + L^2 = x^2) :
  (W * L) = (3 / 10) * x^2 := by
  sorry

end rectangle_area_l2402_240253


namespace geom_seq_min_value_l2402_240250

theorem geom_seq_min_value (a : ℕ → ℝ) (q : ℝ) (h_pos : ∀ n, 0 < a n)
  (h_geom : ∀ n, a n = a 1 * q ^ (n - 1))
  (h_condition : a 7 = a 6 + 2 * a 5)
  (h_mult : ∃ m n, m ≠ n ∧ a m * a n = 16 * (a 1) ^ 2) :
  ∃ (m n : ℕ), m ≠ n ∧ m + n = 6 ∧ (1 / m : ℝ) + (4 / n : ℝ) = 3 / 2 :=
by
  sorry

end geom_seq_min_value_l2402_240250


namespace students_in_favor_ABC_l2402_240247

variables (U A B C : Finset ℕ)

-- Given conditions
axiom total_students : U.card = 300
axiom students_in_favor_A : A.card = 210
axiom students_in_favor_B : B.card = 190
axiom students_in_favor_C : C.card = 160
axiom students_against_all : (U \ (A ∪ B ∪ C)).card = 40

-- Proof goal
theorem students_in_favor_ABC : (A ∩ B ∩ C).card = 80 :=
by {
  sorry
}

end students_in_favor_ABC_l2402_240247


namespace probability_same_class_l2402_240285

-- Define the problem conditions
def num_classes : ℕ := 3
def total_scenarios : ℕ := num_classes * num_classes
def same_class_scenarios : ℕ := num_classes

-- Formulate the proof problem
theorem probability_same_class :
  (same_class_scenarios : ℚ) / total_scenarios = 1 / 3 :=
sorry

end probability_same_class_l2402_240285


namespace exist_indices_inequalities_l2402_240266

open Nat

theorem exist_indices_inequalities (a b c : ℕ → ℕ) :
  ∃ p q : ℕ, a p ≥ a q ∧ b p ≥ b q ∧ c p ≥ c q :=
by
  -- The proof is to be written here
  sorry

end exist_indices_inequalities_l2402_240266


namespace quadratic_expression_min_value_l2402_240227

noncomputable def min_value_quadratic_expression (x y z : ℝ) : ℝ :=
(x + 5) ^ 2 + (y - 1) ^ 2 + (z + 3) ^ 2

theorem quadratic_expression_min_value :
  ∃ x y z : ℝ, x - 2 * y + 2 * z = 5 ∧ min_value_quadratic_expression x y z = 36 :=
sorry

end quadratic_expression_min_value_l2402_240227


namespace how_many_bones_in_adult_woman_l2402_240287

-- Define the conditions
def numSkeletons : ℕ := 20
def halfSkeletons : ℕ := 10
def numAdultWomen : ℕ := 10
def numMenAndChildren : ℕ := 10
def numAdultMen : ℕ := 5
def numChildren : ℕ := 5
def totalBones : ℕ := 375

-- Define the proof statement
theorem how_many_bones_in_adult_woman (W : ℕ) (H : 10 * W + 5 * (W + 5) + 5 * (W / 2) = 375) : W = 20 :=
sorry

end how_many_bones_in_adult_woman_l2402_240287


namespace proposition_A_proposition_B_proposition_C_proposition_D_l2402_240201

-- Definitions and conditions for proposition A
def propA_conditions (a b : ℝ) : Prop :=
  a > b ∧ (1 / a) > (1 / b)

def propA (a b : ℝ) : Prop :=
  a * b < 0

-- Definitions and conditions for proposition B
def propB_conditions (a b : ℝ) : Prop :=
  a < b ∧ b < 0

def propB (a b : ℝ) : Prop :=
  a^2 < a * b ∧ a * b < b^2

-- Definitions and conditions for proposition C
def propC_conditions (c a b : ℝ) : Prop :=
  c > a ∧ a > b ∧ b > 0

def propC (c a b : ℝ) : Prop :=
  (a / (c - a)) < (b / (c - b))

-- Definitions and conditions for proposition D
def propD_conditions (a b c : ℝ) : Prop :=
  a > b ∧ b > c ∧ c > 0

def propD (a b c : ℝ) : Prop :=
  (a / b) > ((a + c) / (b + c))

-- The propositions
theorem proposition_A (a b : ℝ) (h : propA_conditions a b) : propA a b := 
sorry

theorem proposition_B (a b : ℝ) (h : propB_conditions a b) : ¬ propB a b :=
sorry

theorem proposition_C (c a b : ℝ) (h : propC_conditions c a b) : ¬ propC c a b :=
sorry

theorem proposition_D (a b c : ℝ) (h : propD_conditions a b c) : propD a b c :=
sorry

end proposition_A_proposition_B_proposition_C_proposition_D_l2402_240201


namespace smallest_multiple_l2402_240255

theorem smallest_multiple (b : ℕ) (h1 : b % 6 = 0) (h2 : b % 15 = 0) (h3 : ∀ n : ℕ, (n % 6 = 0 ∧ n % 15 = 0) → n ≥ b) : b = 30 :=
sorry

end smallest_multiple_l2402_240255


namespace time_to_fill_cistern_proof_l2402_240261

-- Define the filling rate F and emptying rate E
def filling_rate : ℚ := 1 / 3 -- cisterns per hour
def emptying_rate : ℚ := 1 / 6 -- cisterns per hour

-- Define the net rate as the difference between filling and emptying rates
def net_rate : ℚ := filling_rate - emptying_rate

-- Define the time to fill the cistern given the net rate
def time_to_fill_cistern (net_rate : ℚ) : ℚ := 1 / net_rate

-- The proof statement
theorem time_to_fill_cistern_proof : time_to_fill_cistern net_rate = 6 := 
by sorry

end time_to_fill_cistern_proof_l2402_240261


namespace sqrt_diff_ineq_sum_sq_gt_sum_prod_l2402_240225

-- First proof problem: Prove that sqrt(11) - 2 * sqrt(3) > 3 - sqrt(10)
theorem sqrt_diff_ineq : (Real.sqrt 11 - 2 * Real.sqrt 3) > (3 - Real.sqrt 10) := sorry

-- Second proof problem: Prove that a^2 + b^2 + c^2 > ab + bc + ca given a, b, and c are real numbers that are not all equal
theorem sum_sq_gt_sum_prod (a b c : ℝ) (h : ¬ (a = b ∧ b = c ∧ a = c)) : a^2 + b^2 + c^2 > a * b + b * c + c * a := sorry

end sqrt_diff_ineq_sum_sq_gt_sum_prod_l2402_240225


namespace solve_system_l2402_240236

noncomputable def solution1 (a b : ℝ) : ℝ × ℝ := 
  ((a + Real.sqrt (a^2 + 4 * b)) / 2, (-a + Real.sqrt (a^2 + 4 * b)) / 2)

noncomputable def solution2 (a b : ℝ) : ℝ × ℝ := 
  ((a - Real.sqrt (a^2 + 4 * b)) / 2, (-a - Real.sqrt (a^2 + 4 * b)) / 2)

theorem solve_system (a b x y : ℝ) : 
  (x - y = a ∧ x * y = b) ↔ ((x, y) = solution1 a b ∨ (x, y) = solution2 a b) := 
by sorry

end solve_system_l2402_240236


namespace tom_age_ratio_l2402_240229

-- Define the conditions
variable (T N : ℕ) (ages_of_children_sum : ℕ)

-- Given conditions as definitions
def condition1 : Prop := T = ages_of_children_sum
def condition2 : Prop := (T - N) = 3 * (T - 4 * N)

-- The theorem statement to be proven
theorem tom_age_ratio : condition1 T ages_of_children_sum ∧ condition2 T N → T / N = 11 / 2 :=
by sorry

end tom_age_ratio_l2402_240229


namespace problem1_problem2_l2402_240256

def a (n : ℕ) : ℕ :=
  if n = 0 then 0  -- We add this case for Lean to handle zero index
  else if n = 1 then 2
  else 2^(n-1)

def S (n : ℕ) : ℕ :=
  Finset.sum (Finset.range (n + 1)) a

theorem problem1 (n : ℕ) :
  a n = 
  if n = 1 then 2
  else 2^(n-1) :=
sorry

theorem problem2 (n : ℕ) :
  S n = 2^n :=
sorry

end problem1_problem2_l2402_240256


namespace solve_for_b_l2402_240224

variable (a b c d m : ℝ)

theorem solve_for_b (h : m = cadb / (a - b)) : b = ma / (cad + m) :=
sorry

end solve_for_b_l2402_240224


namespace compound_interest_rate_l2402_240211

open Real

theorem compound_interest_rate
  (P : ℝ) (A : ℝ) (t : ℝ) (r : ℝ)
  (h_inv : P = 8000)
  (h_time : t = 2)
  (h_maturity : A = 8820) :
  r = 0.05 :=
by
  sorry

end compound_interest_rate_l2402_240211


namespace variance_decreases_l2402_240241

def scores_initial := [5, 9, 7, 10, 9] -- Initial 5 shot scores
def additional_shot := 8 -- Additional shot score

-- Given variance of initial scores
def variance_initial : ℝ := 3.2

-- Placeholder function to calculate variance of a list of scores
noncomputable def variance (scores : List ℝ) : ℝ := sorry

-- Definition of the new scores list
def scores_new := scores_initial ++ [additional_shot]

-- Define the proof problem
theorem variance_decreases :
  variance scores_new < variance_initial :=
sorry

end variance_decreases_l2402_240241


namespace amount_of_money_C_l2402_240209

theorem amount_of_money_C (a b c d : ℤ) 
  (h1 : a + b + c + d = 600)
  (h2 : a + c = 200)
  (h3 : b + c = 350)
  (h4 : a + d = 300)
  (h5 : a ≥ 2 * b) : c = 150 := 
by
  sorry

end amount_of_money_C_l2402_240209


namespace find_edge_value_l2402_240216

theorem find_edge_value (a b c d e_1 e_2 e_3 e_4 : ℕ) 
  (h1 : e_1 = a + b)
  (h2 : e_2 = b + c)
  (h3 : e_3 = c + d)
  (h4 : e_4 = d + a)
  (h5 : e_1 = 8)
  (h6 : e_3 = 13)
  (h7 : e_1 + e_3 = a + b + c + d)
  : e_4 = 12 := 
by sorry

end find_edge_value_l2402_240216


namespace range_of_m_l2402_240271

theorem range_of_m (m : ℝ) : 
  (¬ ∃ x : ℝ, x^2 - m * x - m < 0) ↔ (-4 ≤ m ∧ m ≤ 0) := 
by sorry

end range_of_m_l2402_240271


namespace days_y_needs_l2402_240275

theorem days_y_needs
  (d : ℝ)
  (h1 : (1:ℝ) / 21 * 14 = 1 - 5 * (1 / d)) :
  d = 10 :=
sorry

end days_y_needs_l2402_240275


namespace girls_collected_more_mushrooms_l2402_240251

variables (N I A V : ℝ)

theorem girls_collected_more_mushrooms 
    (h1 : N > I) 
    (h2 : N > A) 
    (h3 : N > V) 
    (h4 : I ≤ N) 
    (h5 : I ≤ A) 
    (h6 : I ≤ V) 
    (h7 : A > V) : 
    N + I > A + V := 
by {
    sorry
}

end girls_collected_more_mushrooms_l2402_240251


namespace two_numbers_solution_l2402_240293

noncomputable def a := 8 + Real.sqrt 58
noncomputable def b := 8 - Real.sqrt 58

theorem two_numbers_solution : 
  (Real.sqrt (a * b) = Real.sqrt 6) ∧ ((2 * a * b) / (a + b) = 3 / 4) → 
  (a = 8 + Real.sqrt 58 ∧ b = 8 - Real.sqrt 58) ∨ (a = 8 - Real.sqrt 58 ∧ b = 8 + Real.sqrt 58) := 
by
  sorry

end two_numbers_solution_l2402_240293


namespace min_value_inverse_sum_l2402_240277

variable {x y : ℝ}

theorem min_value_inverse_sum (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 4) : (1/x + 1/y) ≥ 1 :=
  sorry

end min_value_inverse_sum_l2402_240277


namespace isosceles_right_triangle_measure_l2402_240281

theorem isosceles_right_triangle_measure (a XY YZ : ℝ) 
    (h1 : XY > YZ) 
    (h2 : a^2 = 25 / (1/2)) : XY = 10 :=
by
  sorry

end isosceles_right_triangle_measure_l2402_240281


namespace ball_bounce_height_l2402_240207

theorem ball_bounce_height :
  ∃ k : ℕ, 2000 * (2 / 3 : ℝ) ^ k < 2 ∧ ∀ j : ℕ, j < k → 2000 * (2 / 3 : ℝ) ^ j ≥ 2 :=
by {
  sorry
}

end ball_bounce_height_l2402_240207


namespace cross_product_example_l2402_240254

def vector_cross (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := 
  (u.2.1 * v.2.2 - u.2.2 * v.2.1, 
   u.2.2 * v.1 - u.1 * v.2.2, 
   u.1 * v.1 - u.2.1 * v.1)
   
theorem cross_product_example : 
  vector_cross (4, 3, -7) (2, 0, 5) = (15, -34, -6) :=
by
  -- The proof will go here
  sorry

end cross_product_example_l2402_240254


namespace cube_volume_surface_area_l2402_240214

variable (x : ℝ)

theorem cube_volume_surface_area (h1 : s^3 = 8 * x) (h2 : 6 * s^2 = 2 * x) : x = 1728 :=
by
  sorry

end cube_volume_surface_area_l2402_240214


namespace unique_real_solution_l2402_240288

theorem unique_real_solution :
  ∀ (x y z w : ℝ),
    x = z + w + Real.sqrt (z * w * x) ∧
    y = w + x + Real.sqrt (w * x * y) ∧
    z = x + y + Real.sqrt (x * y * z) ∧
    w = y + z + Real.sqrt (y * z * w) →
    (x = 0 ∧ y = 0 ∧ z = 0 ∧ w = 0) :=
by
  intro x y z w
  intros h
  have h1 : x = z + w + Real.sqrt (z * w * x) := h.1
  have h2 : y = w + x + Real.sqrt (w * x * y) := h.2.1
  have h3 : z = x + y + Real.sqrt (x * y * z) := h.2.2.1
  have h4 : w = y + z + Real.sqrt (y * z * w) := h.2.2.2
  sorry

end unique_real_solution_l2402_240288


namespace num_digits_abc_l2402_240283

theorem num_digits_abc (a b c : ℕ) (n : ℕ) (h_a : 10^(n-1) ≤ a ∧ a < 10^n) (h_b : 10^(n-1) ≤ b ∧ b < 10^n) (h_c : 10^(n-1) ≤ c ∧ c < 10^n) :
  ¬ ((Int.natAbs ((10^(n-1) : ℕ) * (10^(n-1) : ℕ) * (10^(n-1) : ℕ)) + 1 = 3*n) ∧
     (Int.natAbs ((10^(n-1) : ℕ) * (10^(n-1) : ℕ) * (10^(n-1) : ℕ)) + 1 = 3*n - 1) ∧
     (Int.natAbs ((10^(n-1) : ℕ) * (10^(n-1) : ℕ) * (10^(n-1) : ℕ)) + 1 = 3*n - 2)) :=
sorry

end num_digits_abc_l2402_240283


namespace final_grade_calculation_l2402_240237

theorem final_grade_calculation
  (exam_score homework_score class_participation_score : ℝ)
  (exam_weight homework_weight participation_weight : ℝ)
  (h_exam_score : exam_score = 90)
  (h_homework_score : homework_score = 85)
  (h_class_participation_score : class_participation_score = 80)
  (h_exam_weight : exam_weight = 3)
  (h_homework_weight : homework_weight = 2)
  (h_participation_weight : participation_weight = 5) :
  (exam_score * exam_weight + homework_score * homework_weight + class_participation_score * participation_weight) /
  (exam_weight + homework_weight + participation_weight) = 84 :=
by
  -- The proof would go here
  sorry

end final_grade_calculation_l2402_240237


namespace total_shoes_count_l2402_240276

-- Define the concepts and variables related to the conditions
def num_people := 10
def num_people_regular_shoes := 4
def num_people_sandals := 3
def num_people_slippers := 3
def num_shoes_regular := 2
def num_shoes_sandals := 1
def num_shoes_slippers := 1

-- Goal: Prove that the total number of shoes kept outside is 20
theorem total_shoes_count :
  (num_people_regular_shoes * num_shoes_regular) +
  (num_people_sandals * num_shoes_sandals * 2) +
  (num_people_slippers * num_shoes_slippers * 2) = 20 :=
by
  sorry

end total_shoes_count_l2402_240276


namespace units_digit_of_power_l2402_240210

theorem units_digit_of_power (base : ℕ) (exp : ℕ) (units_base : ℕ) (units_exp_mod : ℕ) :
  (base % 10 = units_base) → (exp % 2 = units_exp_mod) → (units_base = 9) → (units_exp_mod = 0) →
  (base ^ exp % 10 = 1) :=
by
  intros
  sorry

end units_digit_of_power_l2402_240210


namespace three_integers_sum_of_consecutive_odds_l2402_240295

theorem three_integers_sum_of_consecutive_odds :
  {N : ℕ | N ≤ 500 ∧ (∃ j n, N = j * (2 * n + j) ∧ j ≥ 1) ∧
                   (∃! j1 j2 j3, ∃ n1 n2 n3, N = j1 * (2 * n1 + j1) ∧ N = j2 * (2 * n2 + j2) ∧ N = j3 * (2 * n3 + j3) ∧ j1 ≠ j2 ∧ j2 ≠ j3 ∧ j1 ≠ j3)} = {16, 18, 50} :=
by
  sorry

end three_integers_sum_of_consecutive_odds_l2402_240295


namespace find_number_of_pens_l2402_240249

-- Definitions based on the conditions in the problem
def total_utensils (P L : ℕ) : Prop := P + L = 108
def pencils_formula (P L : ℕ) : Prop := L = 5 * P + 12

-- The theorem we need to prove
theorem find_number_of_pens (P L : ℕ) (h1 : total_utensils P L) (h2 : pencils_formula P L) : P = 16 :=
by sorry

end find_number_of_pens_l2402_240249


namespace angle_in_second_quadrant_l2402_240274

theorem angle_in_second_quadrant (α : ℝ) (h₁ : -2 * Real.pi < α) (h₂ : α < -Real.pi) : 
  α = -4 → (α > -3 * Real.pi / 2 ∧ α < -Real.pi / 2) :=
by
  intros hα
  sorry

end angle_in_second_quadrant_l2402_240274


namespace max_a_b_c_d_l2402_240284

theorem max_a_b_c_d (a c d b : ℤ) (hb : b > 0) (h1 : a + b = c) (h2 : b + c = d) (h3 : c + d = a) 
: a + b + c + d = -5 :=
by
  sorry

end max_a_b_c_d_l2402_240284


namespace Peter_work_rate_l2402_240203

theorem Peter_work_rate:
  ∀ (m p j : ℝ),
    (m + p + j) * 20 = 1 →
    (m + p + j) * 10 = 0.5 →
    (p + j) * 10 = 0.5 →
    j * 15 = 0.5 →
    p * 60 = 1 :=
by
  intros m p j h1 h2 h3 h4
  sorry

end Peter_work_rate_l2402_240203


namespace problem_1_problem_2_problem_3_problem_4_l2402_240200

theorem problem_1 : (1 * -2.48) + 4.33 + (-7.52) + (-4.33) = -10 := by
  sorry

theorem problem_2 : 2 * (23 / 6 : ℚ) + - (36 / 7 : ℚ) + - (13 / 6 : ℚ) + - (230 / 7 : ℚ) = -(36 + 1 / 3 : ℚ) := by
  sorry

theorem problem_3 : (4 / 5 : ℚ) - (5 / 6 : ℚ) - (3 / 5 : ℚ) + (1 / 6 : ℚ) = - (7 / 15 : ℚ) := by
  sorry

theorem problem_4 : (-1 ^ 4 : ℚ) - (1 / 6) * (2 - (-3) ^ 2) = 1 / 6 := by
  sorry

end problem_1_problem_2_problem_3_problem_4_l2402_240200


namespace dan_money_left_l2402_240268

def money_left (initial_amount spent_on_candy spent_on_gum : ℝ) : ℝ :=
  initial_amount - (spent_on_candy + spent_on_gum)

theorem dan_money_left :
  money_left 3.75 1.25 0.80 = 1.70 :=
by
  sorry

end dan_money_left_l2402_240268


namespace sqrt_a_minus_2_meaningful_l2402_240230

theorem sqrt_a_minus_2_meaningful (a : ℝ) (h : 0 ≤ a - 2) : 2 ≤ a :=
by
  sorry

end sqrt_a_minus_2_meaningful_l2402_240230


namespace profit_percentage_l2402_240239

theorem profit_percentage (CP SP : ℝ) (hCP : CP = 150) (hSP : SP = 216.67) :
  SP = 0.9 * LP ∧ LP = SP / 0.9 ∧ Profit = SP - CP ∧ Profit_Percentage = (Profit / CP) * 100 ∧ Profit_Percentage = 44.44 :=
by
  sorry

end profit_percentage_l2402_240239


namespace digit_of_fraction_l2402_240222

theorem digit_of_fraction (n : ℕ) : (15 / 37 : ℝ) = 0.405 ∧ 415 % 3 = 1 → ∃ d : ℕ, d = 4 :=
by
  sorry

end digit_of_fraction_l2402_240222


namespace common_difference_of_arithmetic_sequence_l2402_240206

theorem common_difference_of_arithmetic_sequence
  (a : ℕ → ℝ) (h_arith_seq : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))
  (h1 : a 7 - 2 * a 4 = -1)
  (h2 : a 3 = 0) :
  (a 2 - a 1) = - 1 / 2 :=
by
  sorry

end common_difference_of_arithmetic_sequence_l2402_240206


namespace diameter_in_scientific_notation_l2402_240290

def diameter : ℝ := 0.00000011
def scientific_notation (d : ℝ) : Prop := d = 1.1e-7

theorem diameter_in_scientific_notation : scientific_notation diameter :=
by
  sorry

end diameter_in_scientific_notation_l2402_240290


namespace greatest_value_a2_b2_c2_d2_l2402_240298

theorem greatest_value_a2_b2_c2_d2 :
  ∃ (a b c d : ℝ), a + b = 12 ∧ ab + c + d = 54 ∧ ad + bc = 105 ∧ cd = 50 ∧ a^2 + b^2 + c^2 + d^2 = 124 := by
  sorry

end greatest_value_a2_b2_c2_d2_l2402_240298


namespace area_triangle_QDA_l2402_240294

-- Define the points
def Q : ℝ × ℝ := (0, 15)
def A (q : ℝ) : ℝ × ℝ := (q, 15)
def D (p : ℝ) : ℝ × ℝ := (0, p)

-- Define the conditions
variable (q : ℝ) (p : ℝ)
variable (hq : q > 0) (hp : p < 15)

-- Theorem stating the area of the triangle QDA in terms of q and p
theorem area_triangle_QDA : 
  1 / 2 * q * (15 - p) = 1 / 2 * q * (15 - p) :=
by sorry

end area_triangle_QDA_l2402_240294


namespace range_u_inequality_le_range_k_squared_l2402_240208

def D (k : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 > 0 ∧ p.2 > 0 ∧ p.1 + p.2 = k}

theorem range_u (k : ℝ) (hk : k > 0) :
  ∀ (x1 x2 : ℝ), (x1, x2) ∈ D k → 0 < x1 * x2 ∧ x1 * x2 ≤ k^2 / 4 :=
sorry

theorem inequality_le (k : ℝ) (hk : k ≥ 1) :
  ∀ (x1 x2 : ℝ), (x1, x2) ∈ D k →
  (1 / x1 - x1) * (1 / x2 - x2) ≤ (k / 2 - 2 / k)^2 :=
sorry

theorem range_k_squared (k : ℝ) :
  (0 < k^2 ∧ k^2 ≤ 4 * Real.sqrt 5 - 8) ↔
  ∀ (x1 x2 : ℝ), (x1, x2) ∈ D k →
  (1 / x1 - x1) * (1 / x2 - x2) ≥ (k / 2 - 2 / k)^2 :=
sorry

end range_u_inequality_le_range_k_squared_l2402_240208


namespace canadian_math_olympiad_1992_l2402_240244

theorem canadian_math_olympiad_1992
    (n : ℤ) (a : ℕ → ℤ) (k : ℕ)
    (h1 : n ≥ a 1) 
    (h2 : ∀ i, 1 ≤ i → i ≤ k → a i > 0)
    (h3 : ∀ i j, 1 ≤ i → i ≤ k → 1 ≤ j → j ≤ k → n ≥ Int.lcm (a i) (a j))
    (h4 : ∀ i, 1 ≤ i → i < k → a i > a (i + 1)) :
  ∀ i, 1 ≤ i → i ≤ k → i * a i ≤ n :=
sorry

end canadian_math_olympiad_1992_l2402_240244


namespace cos_sin_ratio_l2402_240252

open Real

-- Given conditions
variables {α β : Real}
axiom tan_alpha_beta : tan (α + β) = 2 / 5
axiom tan_beta_pi_over_4 : tan (β - π / 4) = 1 / 4

-- Theorem to be proven
theorem cos_sin_ratio (hαβ : tan (α + β) = 2 / 5) (hβ : tan (β - π / 4) = 1 / 4) :
  (cos α + sin α) / (cos α - sin α) = 3 / 22 :=
sorry

end cos_sin_ratio_l2402_240252


namespace ratio_copper_to_zinc_l2402_240243

theorem ratio_copper_to_zinc (copper zinc : ℝ) (hc : copper = 24) (hz : zinc = 10.67) : (copper / zinc) = 2.25 :=
by
  rw [hc, hz]
  -- Add the arithmetic operation
  sorry

end ratio_copper_to_zinc_l2402_240243


namespace students_multiple_activities_l2402_240223

theorem students_multiple_activities (total_students only_debate only_singing only_dance no_activities students_more_than_one : ℕ) 
  (h1 : total_students = 55) 
  (h2 : only_debate = 10) 
  (h3 : only_singing = 18) 
  (h4 : only_dance = 8)
  (h5 : no_activities = 5)
  (h6 : students_more_than_one = total_students - (only_debate + only_singing + only_dance + no_activities)) :
  students_more_than_one = 14 := by
  sorry

end students_multiple_activities_l2402_240223


namespace no_ordered_triples_l2402_240260

noncomputable def no_solution (x y z : ℝ) : Prop :=
  x^2 - 3 * x * y + 2 * y^2 - z^2 = 31 ∧
  -x^2 + 6 * y * z + 2 * z^2 = 44 ∧
  x^2 + x * y + 8 * z^2 = 100

theorem no_ordered_triples : ¬ ∃ (x y z : ℝ), no_solution x y z := 
by 
  sorry

end no_ordered_triples_l2402_240260


namespace outer_boundary_diameter_l2402_240299

def width_jogging_path : ℝ := 10
def width_vegetable_garden : ℝ := 12
def diameter_pond : ℝ := 20

theorem outer_boundary_diameter :
  2 * (diameter_pond / 2 + width_vegetable_garden + width_jogging_path) = 64 := by
  sorry

end outer_boundary_diameter_l2402_240299


namespace grade12_students_selected_l2402_240234

theorem grade12_students_selected 
    (N : ℕ) (n10 : ℕ) (n12 : ℕ) (k : ℕ) 
    (h1 : N = 1200)
    (h2 : n10 = 240)
    (h3 : 3 * N / (k + 5 + 3) = n12)
    (h4 : k * N / (k + 5 + 3) = n10) :
    n12 = 360 := 
by sorry

end grade12_students_selected_l2402_240234


namespace initial_violet_balloons_l2402_240248

-- Defining the conditions
def violet_balloons_given_by_tom : ℕ := 16
def violet_balloons_left_with_tom : ℕ := 14

-- The statement to prove
theorem initial_violet_balloons (initial_balloons : ℕ) :
  initial_balloons = violet_balloons_given_by_tom + violet_balloons_left_with_tom :=
sorry

end initial_violet_balloons_l2402_240248


namespace solution_set_of_inequality_l2402_240240

open Real Set

noncomputable def f (x : ℝ) : ℝ := exp (-x) - exp x - 5 * x

theorem solution_set_of_inequality :
  { x : ℝ | f (x ^ 2) + f (-x - 6) < 0 } = Iio (-2) ∪ Ioi 3 :=
by
  sorry

end solution_set_of_inequality_l2402_240240


namespace max_area_cross_section_of_prism_l2402_240228

noncomputable def prism_vertex_A : ℝ × ℝ × ℝ := (3, 0, 0)
noncomputable def prism_vertex_B : ℝ × ℝ × ℝ := (-3, 0, 0)
noncomputable def prism_vertex_C : ℝ × ℝ × ℝ := (0, 3 * Real.sqrt 3, 0)
noncomputable def plane_eq (x y z : ℝ) : ℝ := 2 * x - 3 * y + 6 * z

-- Statement
theorem max_area_cross_section_of_prism (h : ℝ) (A B C : ℝ × ℝ × ℝ)
  (plane : ℝ → ℝ → ℝ → ℝ) (cond_h : h = 5)
  (cond_A : A = prism_vertex_A) (cond_B : B = prism_vertex_B) 
  (cond_C : C = prism_vertex_C) (cond_plane : ∀ x y z, plane x y z = 2 * x - 3 * y + 6 * z - 30) : 
  ∃ cross_section : ℝ, cross_section = 0 :=
by
  sorry

end max_area_cross_section_of_prism_l2402_240228


namespace cost_keyboard_l2402_240235

def num_keyboards : ℕ := 15
def num_printers : ℕ := 25
def total_cost : ℝ := 2050
def cost_printer : ℝ := 70
def total_cost_printers : ℝ := num_printers * cost_printer
def total_cost_keyboards : ℝ := total_cost - total_cost_printers

theorem cost_keyboard : total_cost_keyboards / num_keyboards = 20 := by
  sorry

end cost_keyboard_l2402_240235


namespace B_subset_A_l2402_240282

variable {α : Type*}
variable (A B : Set α)

def A_def : Set ℝ := { x | x ≥ 1 }
def B_def : Set ℝ := { x | x > 2 }

theorem B_subset_A : B_def ⊆ A_def :=
sorry

end B_subset_A_l2402_240282


namespace exists_rational_non_integer_linear_l2402_240264

theorem exists_rational_non_integer_linear (k1 k2 : ℤ) : 
  ∃ (x y : ℚ), x ≠ ⌊x⌋ ∧ y ≠ ⌊y⌋ ∧ 
  19 * x + 8 * y = k1 ∧ 8 * x + 3 * y = k2 :=
sorry

end exists_rational_non_integer_linear_l2402_240264


namespace unit_prices_min_number_of_A_l2402_240238

theorem unit_prices (x y : ℝ)
  (h1 : 3 * x + 4 * y = 580)
  (h2 : 6 * x + 5 * y = 860) :
  x = 60 ∧ y = 100 :=
by
  sorry

theorem min_number_of_A (x y a : ℝ)
  (x_h : x = 60)
  (y_h : y = 100)
  (h1 : 3 * x + 4 * y = 580)
  (h2 : 6 * x + 5 * y = 860)
  (trash_can_condition : a + 200 - a = 200)
  (cost_condition : 60 * a + 100 * (200 - a) ≤ 15000) :
  a ≥ 125 :=
by
  sorry

end unit_prices_min_number_of_A_l2402_240238


namespace solve_for_x_l2402_240226

theorem solve_for_x:
  ∃ x : ℚ, (2 / 3 - 1 / 4 = 1 / x) ∧ (x = 12 / 5) := by
  sorry

end solve_for_x_l2402_240226


namespace defective_items_count_l2402_240242

variables 
  (total_items : ℕ)
  (total_video_games : ℕ)
  (total_DVDs : ℕ)
  (total_books : ℕ)
  (working_video_games : ℕ)
  (working_DVDs : ℕ)

theorem defective_items_count
  (h1 : total_items = 56)
  (h2 : total_video_games = 30)
  (h3 : total_DVDs = 15)
  (h4 : total_books = total_items - total_video_games - total_DVDs)
  (h5 : working_video_games = 20)
  (h6 : working_DVDs = 10)
  : (total_video_games - working_video_games) + (total_DVDs - working_DVDs) = 15 :=
sorry

end defective_items_count_l2402_240242


namespace number_of_workers_l2402_240233

theorem number_of_workers (N C : ℕ) 
  (h1 : N * C = 300000) 
  (h2 : N * (C + 50) = 325000) : 
  N = 500 :=
sorry

end number_of_workers_l2402_240233


namespace determine_angle_C_in_DEF_l2402_240202

def Triangle := Type

structure TriangleProps (T : Triangle) :=
  (right_angle : Prop)
  (angle_A : ℝ)
  (angle_B : ℝ)
  (angle_C : ℝ)

axiom triangle_ABC : Triangle
axiom triangle_DEF : Triangle

axiom ABC_props : TriangleProps triangle_ABC
axiom DEF_props : TriangleProps triangle_DEF

noncomputable def similar (T1 T2 : Triangle) : Prop := sorry

theorem determine_angle_C_in_DEF
  (h1 : ABC_props.right_angle = true)
  (h2 : ABC_props.angle_A = 30)
  (h3 : DEF_props.right_angle = true)
  (h4 : DEF_props.angle_B = 60)
  (h5 : similar triangle_ABC triangle_DEF) :
  DEF_props.angle_C = 30 :=
sorry

end determine_angle_C_in_DEF_l2402_240202


namespace height_difference_l2402_240262

def empireStateBuildingHeight : ℕ := 443
def petronasTowersHeight : ℕ := 452

theorem height_difference :
  petronasTowersHeight - empireStateBuildingHeight = 9 := 
sorry

end height_difference_l2402_240262


namespace last_digit_of_S_l2402_240218

def last_digit (n : ℕ) : ℕ := n % 10

theorem last_digit_of_S : last_digit (54 ^ 2020 + 28 ^ 2022) = 0 :=
by 
  -- The Lean proof steps would go here
  sorry

end last_digit_of_S_l2402_240218


namespace S_eq_T_l2402_240204

-- Define the sets S and T
def S : Set ℤ := {x | ∃ n : ℕ, x = 3 * n + 1}
def T : Set ℤ := {x | ∃ k : ℤ, x = 3 * k - 2}

-- Prove that S = T
theorem S_eq_T : S = T := 
by {
  sorry
}

end S_eq_T_l2402_240204


namespace find_xy_sum_l2402_240213

open Nat

theorem find_xy_sum (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h1 : x + y + x * y = 8) 
  (h2 : y + z + y * z = 15) 
  (h3 : z + x + z * x = 35) : 
  x + y + z + x * y = 15 := 
sorry

end find_xy_sum_l2402_240213


namespace arithmetic_sequence_sum_l2402_240232

open Real

noncomputable def a_n : ℕ → ℝ := sorry -- to represent the arithmetic sequence

theorem arithmetic_sequence_sum :
  (∃ d : ℝ, ∀ (n : ℕ), a_n n = a_n 1 + (n - 1) * d) ∧
  (∃ a1 a2011 : ℝ, (a_n 1 = a1) ∧ (a_n 2011 = a2011) ∧ (a1 ^ 2 - 10 * a1 + 16 = 0) ∧ (a2011 ^ 2 - 10 * a2011 + 16 = 0)) →
  a_n 2 + a_n 1006 + a_n 2010 = 15 :=
by
  sorry

end arithmetic_sequence_sum_l2402_240232


namespace polygon_interior_angle_l2402_240220

theorem polygon_interior_angle (n : ℕ) (h : n ≥ 3) 
  (interior_angle : ∀ i, 1 ≤ i ∧ i ≤ n → interior_angle = 120) :
  n = 6 := by sorry

end polygon_interior_angle_l2402_240220


namespace incorrect_inequality_exists_l2402_240231

theorem incorrect_inequality_exists :
  ∃ (x y : ℝ), x < y ∧ x^2 ≥ y^2 :=
by {
  sorry
}

end incorrect_inequality_exists_l2402_240231


namespace stratified_sampling_l2402_240272

-- Define the known quantities
def total_products := 2000
def sample_size := 200
def workshop_production := 250

-- Define the main theorem to prove
theorem stratified_sampling:
  (workshop_production / total_products) * sample_size = 25 := by
  sorry

end stratified_sampling_l2402_240272


namespace ana_multiplied_numbers_l2402_240280

theorem ana_multiplied_numbers (x : ℕ) (y : ℕ) 
    (h_diff : y = x + 202) 
    (h_mistake : x * y - 1000 = 288 * x + 67) :
    x = 97 ∧ y = 299 :=
sorry

end ana_multiplied_numbers_l2402_240280


namespace arithmetic_sequence_sum_l2402_240246

noncomputable def isArithmeticSeq (a : ℕ → ℤ) := ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

noncomputable def sum_first_n (a : ℕ → ℤ) (n : ℕ) := (n + 1) * (a 0 + a n) / 2

theorem arithmetic_sequence_sum (a : ℕ → ℤ) (h_legal_seq : isArithmeticSeq a) (h_sum : sum_first_n a 9 = 120) : 
  a 1 + a 8 = 24 := by
  sorry

end arithmetic_sequence_sum_l2402_240246


namespace matrix_inverse_l2402_240291

variable (N : Matrix (Fin 2) (Fin 2) ℚ) 
variable (I : Matrix (Fin 2) (Fin 2) ℚ)
variable (c d : ℚ)

def M1 : Matrix (Fin 2) (Fin 2) ℚ := ![![3, 0], ![2, -4]]

def M2 : Matrix (Fin 2) (Fin 2) ℚ := ![![1, 0], ![0, 1]]

theorem matrix_inverse (hN : N = M1) 
                       (hI : I = M2) 
                       (hc : c = 1/12) 
                       (hd : d = 1/12) :
                       N⁻¹ = c • N + d • I := by
  sorry

end matrix_inverse_l2402_240291


namespace distinct_integers_sum_to_32_l2402_240267

theorem distinct_integers_sum_to_32 
  (p q r s t : ℤ)
  (h_diff : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ r ≠ s ∧ r ≠ t ∧ s ≠ t)
  (h_eq : (9 - p) * (9 - q) * (9 - r) * (9 - s) * (9 - t) = -120) : 
  p + q + r + s + t = 32 := 
by 
  sorry

end distinct_integers_sum_to_32_l2402_240267


namespace probability_of_individual_selection_l2402_240269

theorem probability_of_individual_selection (sample_size : ℕ) (population_size : ℕ)
  (h_sample : sample_size = 10) (h_population : population_size = 42) :
  (sample_size : ℚ) / (population_size : ℚ) = 5 / 21 := 
by {
  sorry
}

end probability_of_individual_selection_l2402_240269
