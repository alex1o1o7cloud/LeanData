import Mathlib

namespace NUMINAMATH_GPT_simplify_expression_l1239_123933

theorem simplify_expression (b : ℝ) : 3 * b * (3 * b^2 + 2 * b) - 2 * b^2 = 9 * b^3 + 4 * b^2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1239_123933


namespace NUMINAMATH_GPT_math_problem_solution_l1239_123953

noncomputable def problem_statement : Prop :=
  let AB := 4
  let AC := 6
  let BC := 5
  let area_ABC := 9.9216 -- Using the approximated area directly for simplicity
  let K_div3 := area_ABC / 3
  let GP := (2 * K_div3) / BC
  let GQ := (2 * K_div3) / AC
  let GR := (2 * K_div3) / AB
  GP + GQ + GR = 4.08432

theorem math_problem_solution : problem_statement :=
by
  sorry

end NUMINAMATH_GPT_math_problem_solution_l1239_123953


namespace NUMINAMATH_GPT_find_a_l1239_123990

theorem find_a (a : ℝ) (h : ∃ b : ℝ, (4:ℝ)*x^2 - (12:ℝ)*x + a = (2*x + b)^2) : a = 9 :=
sorry

end NUMINAMATH_GPT_find_a_l1239_123990


namespace NUMINAMATH_GPT_volume_of_inscribed_sphere_l1239_123907

theorem volume_of_inscribed_sphere (a : ℝ) (π : ℝ) (h : a = 6) : 
  (4 / 3 * π * (a / 2) ^ 3) = 36 * π :=
by
  sorry

end NUMINAMATH_GPT_volume_of_inscribed_sphere_l1239_123907


namespace NUMINAMATH_GPT_divisor_in_second_division_is_19_l1239_123989

theorem divisor_in_second_division_is_19 (n d : ℕ) (h1 : n % 25 = 4) (h2 : (n + 15) % d = 4) : d = 19 :=
sorry

end NUMINAMATH_GPT_divisor_in_second_division_is_19_l1239_123989


namespace NUMINAMATH_GPT_girls_more_than_boys_l1239_123983

theorem girls_more_than_boys : ∃ (b g x : ℕ), b = 3 * x ∧ g = 4 * x ∧ b + g = 35 ∧ g - b = 5 :=
by  -- We just define the theorem, no need for a proof, added "by sorry"
  sorry

end NUMINAMATH_GPT_girls_more_than_boys_l1239_123983


namespace NUMINAMATH_GPT_ratio_lt_one_l1239_123921

def product_sequence (k j : ℕ) := List.prod (List.range' k j)

theorem ratio_lt_one :
  let a := product_sequence 2020 4
  let b := product_sequence 2120 4
  a / b < 1 :=
by
  sorry

end NUMINAMATH_GPT_ratio_lt_one_l1239_123921


namespace NUMINAMATH_GPT_primes_solution_l1239_123994

theorem primes_solution (p q : ℕ) (m n : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hm : m ≥ 2) (hn : n ≥ 2) :
    p^n = q^m + 1 ∨ p^n = q^m - 1 → (p = 2 ∧ n = 3 ∧ q = 3 ∧ m = 2) :=
by
  sorry

end NUMINAMATH_GPT_primes_solution_l1239_123994


namespace NUMINAMATH_GPT_survey_min_people_l1239_123937

theorem survey_min_people (p : ℕ) : 
  (∃ p, ∀ k ∈ [18, 10, 5, 9], k ∣ p) → p = 90 :=
by sorry

end NUMINAMATH_GPT_survey_min_people_l1239_123937


namespace NUMINAMATH_GPT_find_number_l1239_123919

theorem find_number (X a b : ℕ) (hX : X = 10 * a + b) 
  (h1 : a * b = 24) (h2 : 10 * b + a = X + 18) : X = 46 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l1239_123919


namespace NUMINAMATH_GPT_percentage_mike_has_l1239_123901
-- Definitions and conditions
variables (phone_cost : ℝ) (additional_needed : ℝ)
def amount_mike_has := phone_cost - additional_needed

-- Main statement
theorem percentage_mike_has (phone_cost : ℝ) (additional_needed : ℝ) (h1 : phone_cost = 1300) (h2 : additional_needed = 780) : 
  (amount_mike_has phone_cost additional_needed) * 100 / phone_cost = 40 :=
by
  sorry

end NUMINAMATH_GPT_percentage_mike_has_l1239_123901


namespace NUMINAMATH_GPT_validColoringsCount_l1239_123942

-- Define the initial conditions
def isValidColoring (n : ℕ) (color : ℕ → ℕ) : Prop :=
  ∀ i ∈ Finset.range (n - 1), 
    (i % 2 = 1 → (color i = 1 ∨ color i = 3)) ∧
    color i ≠ color (i + 1)

noncomputable def countValidColorings : ℕ → ℕ
| 0     => 1
| 1     => 2
| (n+2) => 
    match n % 2 with
    | 0 => 2 * 3^(n/2)
    | _ => 4 * 3^((n-1)/2)

-- Main theorem
theorem validColoringsCount (n : ℕ) :
  (∀ color : ℕ → ℕ, isValidColoring n color) →
  (if n % 2 = 0 then countValidColorings n = 4 * 3^((n / 2) - 1) 
     else countValidColorings n = 2 * 3^(n / 2)) :=
by
  sorry

end NUMINAMATH_GPT_validColoringsCount_l1239_123942


namespace NUMINAMATH_GPT_volume_of_pyramid_SPQR_l1239_123944

variable (P Q R S : Type)
variable (SP SQ SR : ℝ)
variable (is_perpendicular_SP_SQ : SP * SQ = 0)
variable (is_perpendicular_SQ_SR : SQ * SR = 0)
variable (is_perpendicular_SR_SP : SR * SP = 0)
variable (SP_eq_9 : SP = 9)
variable (SQ_eq_8 : SQ = 8)
variable (SR_eq_7 : SR = 7)

theorem volume_of_pyramid_SPQR : 
  ∃ V : ℝ, V = 84 := by
  -- Conditions and assumption
  sorry

end NUMINAMATH_GPT_volume_of_pyramid_SPQR_l1239_123944


namespace NUMINAMATH_GPT_joan_balloons_l1239_123996

theorem joan_balloons (m t j : ℕ) (h1 : m = 41) (h2 : t = 81) : j = t - m → j = 40 :=
by
  intros h3
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_joan_balloons_l1239_123996


namespace NUMINAMATH_GPT_inverse_proposition_of_parallel_lines_l1239_123948

theorem inverse_proposition_of_parallel_lines 
  (P : Prop) (Q : Prop) 
  (h : P ↔ Q) : 
  (Q ↔ P) :=
by 
  sorry

end NUMINAMATH_GPT_inverse_proposition_of_parallel_lines_l1239_123948


namespace NUMINAMATH_GPT_games_draw_fraction_l1239_123949

-- Definitions from the conditions in the problems
def ben_win_fraction : ℚ := 4 / 9
def tom_win_fraction : ℚ := 1 / 3

-- The theorem we want to prove
theorem games_draw_fraction : 1 - (ben_win_fraction + (1 / 3)) = 2 / 9 := by
  sorry

end NUMINAMATH_GPT_games_draw_fraction_l1239_123949


namespace NUMINAMATH_GPT_range_of_x_when_m_eq_4_range_of_m_given_conditions_l1239_123977

-- Definitions of p and q
def p (x : ℝ) : Prop := x^2 - 7 * x + 10 < 0
def q (x m : ℝ) : Prop := x^2 - 4 * m * x + 3 * m^2 < 0

-- Question 1: Given m = 4 and conditions p ∧ q being true, prove the range of x is 4 < x < 5
theorem range_of_x_when_m_eq_4 (x m : ℝ) (h_m : m = 4) (h : p x ∧ q x m) : 4 < x ∧ x < 5 := 
by
  sorry

-- Question 2: Given conditions ⟪¬q ⟫is a sufficient but not necessary condition for ⟪¬p ⟫and constraints, prove the range of m is 5/3 ≤ m ≤ 2
theorem range_of_m_given_conditions (m : ℝ) (h_sufficient : ∀ (x : ℝ), ¬q x m → ¬p x) (h_constraints : m > 0) : 5 / 3 ≤ m ∧ m ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_x_when_m_eq_4_range_of_m_given_conditions_l1239_123977


namespace NUMINAMATH_GPT_Mandy_older_than_Jackson_l1239_123910

variable (M J A : ℕ)

-- Given conditions
variables (h1 : J = 20)
variables (h2 : A = (3 * J) / 4)
variables (h3 : (M + 10) + (J + 10) + (A + 10) = 95)

-- Prove that Mandy is 10 years older than Jackson
theorem Mandy_older_than_Jackson : M - J = 10 :=
by
  sorry

end NUMINAMATH_GPT_Mandy_older_than_Jackson_l1239_123910


namespace NUMINAMATH_GPT_output_y_for_x_eq_5_l1239_123938

def compute_y (x : Int) : Int :=
  if x > 0 then 3 * x + 1 else -2 * x + 3

theorem output_y_for_x_eq_5 : compute_y 5 = 16 := by
  sorry

end NUMINAMATH_GPT_output_y_for_x_eq_5_l1239_123938


namespace NUMINAMATH_GPT_value_expression_possible_values_l1239_123940

open Real

noncomputable def value_expression (a b : ℝ) : ℝ :=
  a^2 + 2 * a * b + b^2 + 2 * a^2 * b + 2 * a * b^2 + a^2 * b^2

theorem value_expression_possible_values (a b : ℝ)
  (h1 : (a / b) + (b / a) = 5 / 2)
  (h2 : a - b = 3 / 2) :
  value_expression a b = 0 ∨ value_expression a b = 81 :=
sorry

end NUMINAMATH_GPT_value_expression_possible_values_l1239_123940


namespace NUMINAMATH_GPT_largest_possible_a_l1239_123975

theorem largest_possible_a (a b c d : ℕ) (h1 : a < 3 * b) (h2 : b < 4 * c) (h3 : c < 5 * d) (h4 : d < 150) (hp : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) :
  a ≤ 8924 :=
sorry

end NUMINAMATH_GPT_largest_possible_a_l1239_123975


namespace NUMINAMATH_GPT_algebraic_expression_value_l1239_123904

variables {m n : ℝ}

theorem algebraic_expression_value (h : n = 3 - 5 * m) : 10 * m + 2 * n - 3 = 3 :=
by sorry

end NUMINAMATH_GPT_algebraic_expression_value_l1239_123904


namespace NUMINAMATH_GPT_parallelogram_base_length_l1239_123970

theorem parallelogram_base_length (Area Height : ℝ) (h1 : Area = 216) (h2 : Height = 18) : 
  Area / Height = 12 := 
by 
  sorry

end NUMINAMATH_GPT_parallelogram_base_length_l1239_123970


namespace NUMINAMATH_GPT_horizontal_distance_is_0_65_l1239_123925

def parabola (x : ℝ) : ℝ := 2 * x^2 - 3 * x - 4

-- Calculate the horizontal distance between two points on the parabola given their y-coordinates and prove it equals to 0.65
theorem horizontal_distance_is_0_65 :
  ∃ (x1 x2 : ℝ), 
    parabola x1 = 10 ∧ parabola x2 = 0 ∧ abs (x1 - x2) = 0.65 :=
sorry

end NUMINAMATH_GPT_horizontal_distance_is_0_65_l1239_123925


namespace NUMINAMATH_GPT_number_of_ways_to_choose_students_l1239_123902

theorem number_of_ways_to_choose_students :
  let female_students := 4
  let male_students := 3
  (female_students * male_students) = 12 :=
by
  sorry

end NUMINAMATH_GPT_number_of_ways_to_choose_students_l1239_123902


namespace NUMINAMATH_GPT_frog_jumps_further_l1239_123928

-- Given conditions
def grasshopper_jump : ℕ := 9 -- The grasshopper jumped 9 inches
def frog_jump : ℕ := 12 -- The frog jumped 12 inches

-- Proof statement
theorem frog_jumps_further : frog_jump - grasshopper_jump = 3 := by
  sorry

end NUMINAMATH_GPT_frog_jumps_further_l1239_123928


namespace NUMINAMATH_GPT_elder_person_age_l1239_123961

open Nat

variable (y e : ℕ)

-- Conditions
def age_difference := e = y + 16
def age_relation := e - 6 = 3 * (y - 6)

theorem elder_person_age
  (h1 : age_difference y e)
  (h2 : age_relation y e) :
  e = 30 :=
sorry

end NUMINAMATH_GPT_elder_person_age_l1239_123961


namespace NUMINAMATH_GPT_basketball_count_l1239_123955

theorem basketball_count (s b v : ℕ) 
  (h1 : s = b + 23) 
  (h2 : v = s - 18)
  (h3 : v = 40) : b = 35 :=
by sorry

end NUMINAMATH_GPT_basketball_count_l1239_123955


namespace NUMINAMATH_GPT_x_solves_quadratic_and_sum_is_75_l1239_123950

theorem x_solves_quadratic_and_sum_is_75
  (x a b : ℕ) (h : x^2 + 10 * x = 45) (hx_pos : 0 < x) (hx_form : x = Nat.sqrt a - b) 
  (ha_pos : 0 < a) (hb_pos : 0 < b)
  : a + b = 75 := 
sorry

end NUMINAMATH_GPT_x_solves_quadratic_and_sum_is_75_l1239_123950


namespace NUMINAMATH_GPT_Apollonian_Circle_Range_l1239_123916

def range_of_m := Set.Icc (Real.sqrt 5 / 2) (Real.sqrt 21 / 2)

theorem Apollonian_Circle_Range :
  ∃ P : ℝ × ℝ, ∃ m > 0, ((P.1 - 2) ^ 2 + (P.2 - m) ^ 2 = 1 / 4) ∧ 
            (Real.sqrt ((P.1 + 1) ^ 2 + P.2 ^ 2) = 2 * Real.sqrt ((P.1 - 2) ^ 2 + P.2 ^ 2)) →
            m ∈ range_of_m :=
  sorry

end NUMINAMATH_GPT_Apollonian_Circle_Range_l1239_123916


namespace NUMINAMATH_GPT_tan_theta_sub_pi_over_4_l1239_123965

open Real

theorem tan_theta_sub_pi_over_4 (θ : ℝ) (h1 : -π / 2 < θ ∧ θ < 0) 
  (h2 : sin (θ + π / 4) = 3 / 5) : tan (θ - π / 4) = -4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_tan_theta_sub_pi_over_4_l1239_123965


namespace NUMINAMATH_GPT_minimum_value_y_l1239_123906

noncomputable def y (x : ℝ) : ℝ := x + 1 / (x - 1)

theorem minimum_value_y (x : ℝ) (hx : x > 1) : ∃ A, (A = 3) ∧ (∀ y', y' = y x → y' ≥ A) := sorry

end NUMINAMATH_GPT_minimum_value_y_l1239_123906


namespace NUMINAMATH_GPT_certain_number_l1239_123939

theorem certain_number (x y : ℝ) (h1 : 0.65 * x = 0.20 * y) (h2 : x = 210) : y = 682.5 :=
by
  sorry

end NUMINAMATH_GPT_certain_number_l1239_123939


namespace NUMINAMATH_GPT_relationship_y1_y2_y3_l1239_123972

-- Define the function y = 3(x + 1)^2 - 8
def quadratic_fn (x : ℝ) : ℝ := 3 * (x + 1)^2 - 8

-- Define points A, B, and C on the graph of the quadratic function
def y1 := quadratic_fn 1
def y2 := quadratic_fn 2
def y3 := quadratic_fn (-2)

-- The goal is to prove the relationship y2 > y1 > y3
theorem relationship_y1_y2_y3 :
  y2 > y1 ∧ y1 > y3 :=
by sorry

end NUMINAMATH_GPT_relationship_y1_y2_y3_l1239_123972


namespace NUMINAMATH_GPT_quadratic_equation_standard_form_quadratic_equation_coefficients_l1239_123957

theorem quadratic_equation_standard_form : 
  ∀ (x : ℝ), (2 * x^2 - 1 = 6 * x) ↔ (2 * x^2 - 6 * x - 1 = 0) :=
by
  sorry

theorem quadratic_equation_coefficients : 
  ∃ (a b c : ℝ), (a = 2 ∧ b = -6 ∧ c = -1) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_equation_standard_form_quadratic_equation_coefficients_l1239_123957


namespace NUMINAMATH_GPT_seating_arrangement_7_people_l1239_123969

theorem seating_arrangement_7_people (n : Nat) (h1 : n = 7) :
  let m := n - 1
  (m.factorial / m) * 2 = 240 :=
by
  sorry

end NUMINAMATH_GPT_seating_arrangement_7_people_l1239_123969


namespace NUMINAMATH_GPT_inequality_relationship_l1239_123980

noncomputable def even_function_periodic_decreasing (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = f (-x)) ∧
  (∀ x, f (x + 2) = f x) ∧
  (∀ x1 x2, 0 ≤ x1 ∧ x1 < x2 ∧ x2 ≤ 1 → f x1 > f x2)

theorem inequality_relationship (f : ℝ → ℝ) (h : even_function_periodic_decreasing f) : 
  f (-1) < f (2.5) ∧ f (2.5) < f 0 :=
by 
  sorry

end NUMINAMATH_GPT_inequality_relationship_l1239_123980


namespace NUMINAMATH_GPT_dvd_player_movie_ratio_l1239_123968

theorem dvd_player_movie_ratio (M D : ℝ) (h1 : D = M + 63) (h2 : D = 81) : D / M = 4.5 :=
by
  sorry

end NUMINAMATH_GPT_dvd_player_movie_ratio_l1239_123968


namespace NUMINAMATH_GPT_other_asymptote_l1239_123912

-- Define the conditions
def C1 := ∀ x y, y = -2 * x
def C2 := ∀ x, x = -3

-- Formulate the problem
theorem other_asymptote :
  (∃ y m b, y = m * x + b ∧ m = 2 ∧ b = 12) :=
by
  sorry

end NUMINAMATH_GPT_other_asymptote_l1239_123912


namespace NUMINAMATH_GPT_sale_price_correct_l1239_123991

noncomputable def original_price : ℝ := 600.00
noncomputable def first_discount_factor : ℝ := 0.75
noncomputable def second_discount_factor : ℝ := 0.90
noncomputable def final_price : ℝ := original_price * first_discount_factor * second_discount_factor
noncomputable def expected_final_price : ℝ := 0.675 * original_price

theorem sale_price_correct : final_price = expected_final_price := sorry

end NUMINAMATH_GPT_sale_price_correct_l1239_123991


namespace NUMINAMATH_GPT_travel_cost_AB_l1239_123905

theorem travel_cost_AB
  (distance_AB : ℕ)
  (booking_fee : ℕ)
  (cost_per_km_flight : ℝ)
  (correct_total_cost : ℝ)
  (h1 : distance_AB = 4000)
  (h2 : booking_fee = 150)
  (h3 : cost_per_km_flight = 0.12) :
  correct_total_cost = 630 :=
by
  sorry

end NUMINAMATH_GPT_travel_cost_AB_l1239_123905


namespace NUMINAMATH_GPT_naomi_drives_to_parlor_l1239_123997

theorem naomi_drives_to_parlor (d v t t_back : ℝ)
  (ht : t = d / v)
  (ht_back : t_back = 2 * d / v)
  (h_total : 2 * (t + t_back) = 6) : 
  t = 1 :=
by sorry

end NUMINAMATH_GPT_naomi_drives_to_parlor_l1239_123997


namespace NUMINAMATH_GPT_inequality_proof_l1239_123930

variable {x : ℝ}
variable {n : ℕ}
variable {a : ℝ}

theorem inequality_proof (h1 : x > 0) (h2 : n > 0) (h3 : x + a / x^n ≥ n + 1) : a = n^n := 
sorry

end NUMINAMATH_GPT_inequality_proof_l1239_123930


namespace NUMINAMATH_GPT_maximum_value_of_omega_l1239_123945

variable (A ω : ℝ)

theorem maximum_value_of_omega (hA : 0 < A) (hω_pos : 0 < ω)
  (h1 : ω * (-π / 2) ≥ -π / 2) 
  (h2 : ω * (2 * π / 3) ≤ π / 2) :
  ω = 3 / 4 :=
sorry

end NUMINAMATH_GPT_maximum_value_of_omega_l1239_123945


namespace NUMINAMATH_GPT_min_value_ineq_l1239_123923

theorem min_value_ineq (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h_point_on_chord : ∃ x y : ℝ, x = 4 * a ∧ y = 2 * b ∧ (x + y = 2) ∧ (x^2 + y^2 = 4) ∧ ((x - 2)^2 + (y - 2)^2 = 4)) :
  1 / a + 2 / b ≥ 8 :=
by
  sorry

end NUMINAMATH_GPT_min_value_ineq_l1239_123923


namespace NUMINAMATH_GPT_part1_solution_set_part2_range_of_a_l1239_123931

-- Defining the function f(x) under given conditions
def f (x a : ℝ) : ℝ := |x - a^2| + |x - (2 * a + 1)|

-- Part 1: Determine the solution set for the inequality when a = 2
theorem part1_solution_set (x : ℝ) : (f x 2) ≥ 4 ↔ x ≤ 3/2 ∨ x ≥ 11/2 := by
  sorry

-- Part 2: Determine the range of values for a given the inequality
theorem part2_range_of_a (a : ℝ) : (∀ x : ℝ, f x a ≥ 4) ↔ a ≤ -1 ∨ a ≥ 3 := by
  sorry

end NUMINAMATH_GPT_part1_solution_set_part2_range_of_a_l1239_123931


namespace NUMINAMATH_GPT_compare_a_b_l1239_123917

theorem compare_a_b (a b : ℝ) (h1 : a = 2 * Real.sqrt 7) (h2 : b = 3 * Real.sqrt 5) : a < b :=
by {
  sorry -- We'll leave the proof as a placeholder.
}

end NUMINAMATH_GPT_compare_a_b_l1239_123917


namespace NUMINAMATH_GPT_toys_produced_each_day_l1239_123993

theorem toys_produced_each_day (weekly_production : ℕ) (days_worked : ℕ) (h₁ : weekly_production = 4340) (h₂ : days_worked = 2) : weekly_production / days_worked = 2170 :=
by {
  -- Proof can be filled in here
  sorry
}

end NUMINAMATH_GPT_toys_produced_each_day_l1239_123993


namespace NUMINAMATH_GPT_find_max_marks_l1239_123971

variable (M : ℕ) (P : ℕ)

theorem find_max_marks (h1 : M = 332) (h2 : P = 83) : 
  let Max_Marks := M / (P / 100)
  Max_Marks = 400 := 
by 
  sorry

end NUMINAMATH_GPT_find_max_marks_l1239_123971


namespace NUMINAMATH_GPT_magic_square_expression_l1239_123932

theorem magic_square_expression : 
  let a := 8
  let b := 6
  let c := 14
  let d := 10
  let e := 11
  let f := 5
  let g := 3
  a - b - c + d + e + f - g = 11 :=
by
  sorry

end NUMINAMATH_GPT_magic_square_expression_l1239_123932


namespace NUMINAMATH_GPT_find_tricksters_in_16_questions_l1239_123999

-- Definitions
def Inhabitant := {i : Nat // i < 65}
def isKnight (i : Inhabitant) : Prop := sorry  -- placeholder for actual definition
def isTrickster (i : Inhabitant) : Prop := sorry -- placeholder for actual definition

-- Conditions
def condition1 : ∀ i : Inhabitant, isKnight i ∨ isTrickster i := sorry
def condition2 : ∃ t1 t2 : Inhabitant, t1 ≠ t2 ∧ isTrickster t1 ∧ isTrickster t2 :=
  sorry
def condition3 : ∀ t1 t2 : Inhabitant, t1 ≠ t2 → ¬(isTrickster t1 ∧ isTrickster t2) → isKnight t1 ∧ isKnight t2 := 
  sorry 
def question (i : Inhabitant) (group : List Inhabitant) : Prop :=
  ∀ j ∈ group, isKnight j

-- Theorem statement
theorem find_tricksters_in_16_questions : ∃ (knight : Inhabitant) (knaves : (Inhabitant × Inhabitant)), 
  isKnight knight ∧ isTrickster knaves.fst ∧ isTrickster knaves.snd ∧ knaves.fst ≠ knaves.snd ∧
  (∀ questionsAsked ≤ 30, sorry) :=
  sorry

end NUMINAMATH_GPT_find_tricksters_in_16_questions_l1239_123999


namespace NUMINAMATH_GPT_find_x_six_l1239_123974

noncomputable def positive_real : Type := { x : ℝ // 0 < x }

theorem find_x_six (x : positive_real)
  (h : (1 - x.val ^ 3) ^ (1/3) + (1 + x.val ^ 3) ^ (1/3) = 1) :
  x.val ^ 6 = 28 / 27 := 
sorry

end NUMINAMATH_GPT_find_x_six_l1239_123974


namespace NUMINAMATH_GPT_max_sum_n_value_l1239_123908

open Nat

-- Definitions for the problem
def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  (n * (2 * a 0 + (n - 1) * (a 1 - a 0))) / 2

-- Statement of the theorem
theorem max_sum_n_value (a : ℕ → ℤ) (d : ℤ) (h_arith_seq : arithmetic_sequence a d) 
  (h_initial : a 0 > 0) (h_condition : 8 * a 4 = 13 * a 10) : 
  ∃ n, sum_of_first_n_terms a n = max (sum_of_first_n_terms a n) ∧ n = 20 :=
sorry

end NUMINAMATH_GPT_max_sum_n_value_l1239_123908


namespace NUMINAMATH_GPT_gcf_180_270_450_l1239_123958

theorem gcf_180_270_450 : Nat.gcd (Nat.gcd 180 270) 450 = 90 :=
by
  sorry

end NUMINAMATH_GPT_gcf_180_270_450_l1239_123958


namespace NUMINAMATH_GPT_tickets_to_buy_l1239_123941

theorem tickets_to_buy
  (ferris_wheel_cost : Float := 2.0)
  (roller_coaster_cost : Float := 7.0)
  (multiple_rides_discount : Float := 1.0)
  (newspaper_coupon : Float := 1.0) :
  (ferris_wheel_cost + roller_coaster_cost - multiple_rides_discount - newspaper_coupon = 7.0) :=
by
  sorry

end NUMINAMATH_GPT_tickets_to_buy_l1239_123941


namespace NUMINAMATH_GPT_value_of_f_sum_l1239_123903

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function (h_odd : ∀ x, f (-x) = -f x) : Prop
axiom period_9 (h_period : ∀ x, f (x + 9) = f x) : Prop
axiom f_one (h_f1 : f 1 = 5) : Prop

theorem value_of_f_sum (h_odd : ∀ x, f (-x) = -f x)
                       (h_period : ∀ x, f (x + 9) = f x)
                       (h_f1 : f 1 = 5) :
  f 2007 + f 2008 = 5 :=
sorry

end NUMINAMATH_GPT_value_of_f_sum_l1239_123903


namespace NUMINAMATH_GPT_proof_by_contradiction_l1239_123987

-- Definitions for the conditions
inductive ContradictionType
| known          -- ① Contradictory to what is known
| assumption     -- ② Contradictory to the assumption
| definitions    -- ③ Contradictory to definitions, theorems, axioms, laws
| facts          -- ④ Contradictory to facts

open ContradictionType

-- Proving that in proof by contradiction, a contradiction can be of type 1, 2, 3, or 4
theorem proof_by_contradiction :
  (∃ ct : ContradictionType, 
    ct = known ∨ 
    ct = assumption ∨ 
    ct = definitions ∨ 
    ct = facts) :=
by
  sorry

end NUMINAMATH_GPT_proof_by_contradiction_l1239_123987


namespace NUMINAMATH_GPT_prime_factors_of_difference_l1239_123922

theorem prime_factors_of_difference (A B : ℕ) (h_neq : A ≠ B) : 
  ∃ p, Nat.Prime p ∧ p ∣ (Nat.gcd (9 * A - 9 * B + 10) (9 * B - 9 * A - 10)) :=
by
  sorry

end NUMINAMATH_GPT_prime_factors_of_difference_l1239_123922


namespace NUMINAMATH_GPT_smallest_multiple_5_711_l1239_123943

theorem smallest_multiple_5_711 : ∃ n : ℕ, n = Nat.lcm 5 711 ∧ n = 3555 := 
by
  sorry

end NUMINAMATH_GPT_smallest_multiple_5_711_l1239_123943


namespace NUMINAMATH_GPT_max_m_for_factored_polynomial_l1239_123920

theorem max_m_for_factored_polynomial :
  ∃ m, (∀ A B : ℤ, (5 * x ^ 2 + m * x + 45 = (5 * x + A) * (x + B) → AB = 45) → 
    m = 226) :=
sorry

end NUMINAMATH_GPT_max_m_for_factored_polynomial_l1239_123920


namespace NUMINAMATH_GPT_red_marked_area_on_larger_sphere_l1239_123986

-- Define the conditions
def r1 : ℝ := 4 -- radius of the smaller sphere
def r2 : ℝ := 6 -- radius of the larger sphere
def A1 : ℝ := 37 -- area marked on the smaller sphere

-- State the proportional relationship as a Lean theorem
theorem red_marked_area_on_larger_sphere : 
  let A2 := A1 * (r2^2 / r1^2)
  A2 = 83.25 :=
by
  sorry

end NUMINAMATH_GPT_red_marked_area_on_larger_sphere_l1239_123986


namespace NUMINAMATH_GPT_average_age_of_family_l1239_123924

theorem average_age_of_family :
  let num_grandparents := 2
  let num_parents := 2
  let num_grandchildren := 3
  let avg_age_grandparents := 64
  let avg_age_parents := 39
  let avg_age_grandchildren := 6
  let total_age_grandparents := avg_age_grandparents * num_grandparents
  let total_age_parents := avg_age_parents * num_parents
  let total_age_grandchildren := avg_age_grandchildren * num_grandchildren
  let total_age_family := total_age_grandparents + total_age_parents + total_age_grandchildren
  let num_family_members := num_grandparents + num_parents + num_grandchildren
  let avg_age_family := total_age_family / num_family_members
  avg_age_family = 32 := 
  by 
  repeat { sorry }

end NUMINAMATH_GPT_average_age_of_family_l1239_123924


namespace NUMINAMATH_GPT_ratio_james_paid_l1239_123963

-- Define the parameters of the problem
def packs : ℕ := 4
def stickers_per_pack : ℕ := 30
def cost_per_sticker : ℚ := 0.10
def james_paid : ℚ := 6

-- Total number of stickers
def total_stickers : ℕ := packs * stickers_per_pack
-- Total cost of stickers
def total_cost : ℚ := total_stickers * cost_per_sticker

-- Theorem stating that the ratio of the amount James paid to the total cost of the stickers is 1:2
theorem ratio_james_paid : james_paid / total_cost = 1 / 2 :=
by 
  -- proof goes here
  sorry

end NUMINAMATH_GPT_ratio_james_paid_l1239_123963


namespace NUMINAMATH_GPT_time_to_write_all_rearrangements_in_hours_l1239_123962

/-- Michael's name length is 7 (number of unique letters) -/
def name_length : Nat := 7

/-- Michael can write 10 rearrangements per minute -/
def write_rate : Nat := 10

/-- Number of rearrangements of Michael's name -/
def num_rearrangements : Nat := (name_length.factorial)

theorem time_to_write_all_rearrangements_in_hours :
  (num_rearrangements / write_rate : ℚ) / 60 = 8.4 := by
  sorry

end NUMINAMATH_GPT_time_to_write_all_rearrangements_in_hours_l1239_123962


namespace NUMINAMATH_GPT_impossible_coins_l1239_123979

theorem impossible_coins (p1 p2 : ℝ) :
  ((1 - p1) * (1 - p2) = p1 * p2) →
  (p1 * (1 - p2) + p2 * (1 - p1) = p1 * p2) →
  false :=
by
  sorry

end NUMINAMATH_GPT_impossible_coins_l1239_123979


namespace NUMINAMATH_GPT_part_a_l1239_123929

theorem part_a (a b c : ℕ) (h : (a + b + c) % 6 = 0) : (a^5 + b^3 + c) % 6 = 0 := 
by sorry

end NUMINAMATH_GPT_part_a_l1239_123929


namespace NUMINAMATH_GPT_binary_arithmetic_l1239_123982

-- Define the binary numbers 11010_2, 11100_2, and 100_2
def x : ℕ := 0b11010 -- base 2 number 11010 in base 10 representation
def y : ℕ := 0b11100 -- base 2 number 11100 in base 10 representation
def d : ℕ := 0b100   -- base 2 number 100 in base 10 representation

-- Define the correct answer
def correct_answer : ℕ := 0b10101101 -- base 2 number 10101101 in base 10 representation

-- The proof problem statement
theorem binary_arithmetic : (x * y) / d = correct_answer := by
  sorry

end NUMINAMATH_GPT_binary_arithmetic_l1239_123982


namespace NUMINAMATH_GPT_probability_two_or_more_women_l1239_123981

-- Definitions based on the conditions
def men : ℕ := 8
def women : ℕ := 4
def total_people : ℕ := men + women
def chosen_people : ℕ := 4

-- Function to calculate the probability of a specific event
noncomputable def probability_event (event_count : ℕ) (total_count : ℕ) : ℚ :=
  event_count / total_count

-- Function to calculate the combination (binomial coefficient)
noncomputable def binom (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Probability calculations based on steps given in the solution:
noncomputable def prob_no_women : ℚ :=
  probability_event ((men - 0) * (men - 1) * (men - 2) * (men - 3)) (total_people * (total_people - 1) * (total_people - 2) * (total_people - 3))

noncomputable def prob_exactly_one_woman : ℚ :=
  probability_event (binom women 1 * binom men 3) (binom total_people chosen_people)

noncomputable def prob_fewer_than_two_women : ℚ :=
  prob_no_women + prob_exactly_one_woman

noncomputable def prob_at_least_two_women : ℚ :=
  1 - prob_fewer_than_two_women

-- The main theorem to be proved
theorem probability_two_or_more_women :
  prob_at_least_two_women = 67 / 165 :=
sorry

end NUMINAMATH_GPT_probability_two_or_more_women_l1239_123981


namespace NUMINAMATH_GPT_d_n_2_d_n_3_l1239_123964

def d (n k : ℕ) : ℕ :=
  if k = 0 then 1
  else if n = 1 then 0
  else (0:ℕ) -- Placeholder to demonstrate that we need a recurrence relation, not strictly necessary here for the statement.

theorem d_n_2 (n : ℕ) (hn : n ≥ 2) : 
  d n 2 = (n^2 - 3*n + 2) / 2 := 
by 
  sorry

theorem d_n_3 (n : ℕ) (hn : n ≥ 3) : 
  d n 3 = (n^3 - 7*n + 6) / 6 := 
by 
  sorry

end NUMINAMATH_GPT_d_n_2_d_n_3_l1239_123964


namespace NUMINAMATH_GPT_remainder_of_base12_2563_mod_17_l1239_123934

-- Define the base-12 number 2563 in decimal.
def base12_to_decimal : ℕ := 2 * 12^3 + 5 * 12^2 + 6 * 12^1 + 3 * 12^0

-- Define the number 17.
def divisor : ℕ := 17

-- Prove that the remainder when base12_to_decimal is divided by divisor is 1.
theorem remainder_of_base12_2563_mod_17 : base12_to_decimal % divisor = 1 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_base12_2563_mod_17_l1239_123934


namespace NUMINAMATH_GPT_pair_comparison_l1239_123998

theorem pair_comparison :
  (∀ (a b : ℤ), (a, b) = (-2^4, (-2)^4) → a ≠ b) ∧
  (∀ (a b : ℤ), (a, b) = (5^3, 3^5) → a ≠ b) ∧
  (∀ (a b : ℤ), (a, b) = (-(-3), -|-3|) → a ≠ b) ∧
  (∀ (a b : ℤ), (a, b) = ((-1)^2, (-1)^2008) → a = b) :=
by
  sorry

end NUMINAMATH_GPT_pair_comparison_l1239_123998


namespace NUMINAMATH_GPT_minimum_unit_cubes_l1239_123946

theorem minimum_unit_cubes (n : ℕ) (N : ℕ) : 
  (n ≥ 3) → (N = n^3) → ((n - 2)^3 > (1/2) * n^3) → 
  ∃ n : ℕ, N = n^3 ∧ (n - 2)^3 > (1/2) * n^3 ∧ N = 1000 :=
by
  intros
  sorry

end NUMINAMATH_GPT_minimum_unit_cubes_l1239_123946


namespace NUMINAMATH_GPT_problem_I4_1_l1239_123926

theorem problem_I4_1 (a : ℝ) : ((∃ y : ℝ, x + 2 * y + 3 = 0) ∧ (∃ y : ℝ, 4 * x - a * y + 5 = 0) ∧ 
  (∃ m1 m2 : ℝ, m1 = -(1 / 2) ∧ m2 = 4 / a ∧ m1 * m2 = -1)) → a = 2 :=
sorry

end NUMINAMATH_GPT_problem_I4_1_l1239_123926


namespace NUMINAMATH_GPT_piles_stones_l1239_123973

theorem piles_stones (a b c d : ℕ)
  (h₁ : a = 2011)
  (h₂ : b = 2010)
  (h₃ : c = 2009)
  (h₄ : d = 2008) :
  ∃ (k l m n : ℕ), (k, l, m, n) = (0, 0, 0, 2) ∧
  ((∃ x y z w : ℕ, k = x - y ∧ l = y - z ∧ m = z - w ∧ x + l + m + w = 0) ∨
   (∃ u : ℕ, k = a - u ∧ l = b - u ∧ m = c - u ∧ n = d - u)) :=
sorry

end NUMINAMATH_GPT_piles_stones_l1239_123973


namespace NUMINAMATH_GPT_solve_equation_l1239_123988

def euler_totient (n : ℕ) : ℕ := sorry  -- Placeholder, Euler's φ function definition
def sigma_function (n : ℕ) : ℕ := sorry  -- Placeholder, σ function definition

theorem solve_equation (x : ℕ) : euler_totient (sigma_function (2^x)) = 2^x → x = 1 := by
  sorry

end NUMINAMATH_GPT_solve_equation_l1239_123988


namespace NUMINAMATH_GPT_part1_part2_l1239_123954

noncomputable def f (x : ℝ) : ℝ := (2 * x) / (Real.log x)

theorem part1 : 
  (∀ x, 0 < x → x < 1 → (f x) < f (1)) ∧ 
  (∀ x, 1 < x → x < Real.exp 1 → (f x) < f (Real.exp 1)) :=
sorry

theorem part2 :
  ∃ k, k = 2 ∧ ∀ x, 0 < x → (f x) > (k / (Real.log x)) + 2 * Real.sqrt x :=
sorry

end NUMINAMATH_GPT_part1_part2_l1239_123954


namespace NUMINAMATH_GPT_fencers_count_l1239_123992

theorem fencers_count (n : ℕ) (h : n * (n - 1) = 72) : n = 9 :=
sorry

end NUMINAMATH_GPT_fencers_count_l1239_123992


namespace NUMINAMATH_GPT_marble_count_l1239_123966

theorem marble_count (a : ℕ) (h1 : a + 3 * a + 6 * a + 30 * a = 120) : a = 3 :=
  sorry

end NUMINAMATH_GPT_marble_count_l1239_123966


namespace NUMINAMATH_GPT_rotate_parabola_180_l1239_123951

theorem rotate_parabola_180 (x y : ℝ) : 
  (y = 2 * (x - 1)^2 + 2) → 
  (∃ x' y', x' = -x ∧ y' = -y ∧ y' = -2 * (x' + 1)^2 - 2) := 
sorry

end NUMINAMATH_GPT_rotate_parabola_180_l1239_123951


namespace NUMINAMATH_GPT_inverse_proportion_l1239_123911

variable {x y x1 x2 y1 y2 : ℝ}
variable {k : ℝ}

theorem inverse_proportion {h1 : x1 ≠ 0} {h2 : x2 ≠ 0} {h3 : y1 ≠ 0} {h4 : y2 ≠ 0}
  (h5 : (∃ k, ∀ (x y : ℝ), x * y = k))
  (h6 : x1 / x2 = 4 / 5) : 
  y1 / y2 = 5 / 4 :=
sorry

end NUMINAMATH_GPT_inverse_proportion_l1239_123911


namespace NUMINAMATH_GPT_negation_of_exists_l1239_123935

theorem negation_of_exists (x : ℝ) :
  ¬ (∃ x > 0, 2 * x + 3 ≤ 0) ↔ ∀ x > 0, 2 * x + 3 > 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_of_exists_l1239_123935


namespace NUMINAMATH_GPT_max_cross_section_area_l1239_123984

noncomputable def prism_cross_section_area : ℝ :=
  let z_axis_parallel := true
  let square_base := 8
  let plane := ∀ x y z, 3 * x - 5 * y + 2 * z = 20
  121.6

theorem max_cross_section_area :
  prism_cross_section_area = 121.6 :=
sorry

end NUMINAMATH_GPT_max_cross_section_area_l1239_123984


namespace NUMINAMATH_GPT_probability_of_diamond_ace_joker_l1239_123960

noncomputable def probability_event (total_cards : ℕ) (event_cards : ℕ) : ℚ :=
  event_cards / total_cards

noncomputable def probability_not_event (total_cards : ℕ) (event_cards : ℕ) : ℚ :=
  1 - probability_event total_cards event_cards

noncomputable def probability_none_event_two_trials (total_cards : ℕ) (event_cards : ℕ) : ℚ :=
  (probability_not_event total_cards event_cards) * (probability_not_event total_cards event_cards)

noncomputable def probability_at_least_one_event_two_trials (total_cards : ℕ) (event_cards : ℕ) : ℚ :=
  1 - probability_none_event_two_trials total_cards event_cards

theorem probability_of_diamond_ace_joker 
  (total_cards : ℕ := 54) (event_cards : ℕ := 18) :
  probability_at_least_one_event_two_trials total_cards event_cards = 5 / 9 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_diamond_ace_joker_l1239_123960


namespace NUMINAMATH_GPT_a_in_range_l1239_123978

noncomputable def kOM (t : ℝ) : ℝ := (Real.log t) / t
noncomputable def kON (a t : ℝ) : ℝ := (a + a * t - t^2) / t

theorem a_in_range (a : ℝ) : 
  (∀ t ∈ Set.Ici 1, 0 ≤ (1 - Real.log t + a) / t^2 + 1) →
  a ∈ Set.Ici (-2) := 
by
  sorry

end NUMINAMATH_GPT_a_in_range_l1239_123978


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1239_123915

theorem sufficient_but_not_necessary_condition (f : ℝ → ℝ) (h : ∀ x, f x = x⁻¹) :
  ∀ x, (x > 1 → f (x + 2) > f (2*x + 1)) ∧ (¬ (x > 1) → ¬ (f (x + 2) > f (2*x + 1))) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1239_123915


namespace NUMINAMATH_GPT_circle_positional_relationship_l1239_123909

noncomputable def r1 : ℝ := 2
noncomputable def r2 : ℝ := 3
noncomputable def d : ℝ := 5

theorem circle_positional_relationship :
  d = r1 + r2 → "externally tangent" = "externally tangent" := by
  intro h
  exact rfl

end NUMINAMATH_GPT_circle_positional_relationship_l1239_123909


namespace NUMINAMATH_GPT_ratio_of_red_to_total_simplified_l1239_123918

def number_of_red_haired_children := 9
def total_number_of_children := 48

theorem ratio_of_red_to_total_simplified:
  (number_of_red_haired_children: ℚ) / (total_number_of_children: ℚ) = (3 : ℚ) / (16 : ℚ) := 
by
  sorry

end NUMINAMATH_GPT_ratio_of_red_to_total_simplified_l1239_123918


namespace NUMINAMATH_GPT_area_of_cross_section_l1239_123952

noncomputable def area_cross_section (H α : ℝ) : ℝ :=
  let AC := 2 * H * Real.sqrt 3 * Real.tan (Real.pi / 2 - α)
  let MK := (H / 2) * Real.sqrt (1 + 16 * (Real.tan (Real.pi / 2 - α))^2)
  (1 / 2) * AC * MK

theorem area_of_cross_section (H α : ℝ) :
  area_cross_section H α = (H^2 * Real.sqrt 3 * Real.tan (Real.pi / 2 - α) / 2) * Real.sqrt (1 + 16 * (Real.tan (Real.pi / 2 - α))^2) :=
sorry

end NUMINAMATH_GPT_area_of_cross_section_l1239_123952


namespace NUMINAMATH_GPT_g_of_3_equals_5_l1239_123914

def g (x : ℝ) : ℝ := 2 * (x - 2) + 3

theorem g_of_3_equals_5 :
  g 3 = 5 :=
by
  sorry

end NUMINAMATH_GPT_g_of_3_equals_5_l1239_123914


namespace NUMINAMATH_GPT_smallest_possible_value_of_EF_minus_DE_l1239_123976

theorem smallest_possible_value_of_EF_minus_DE :
  ∃ (DE EF FD : ℤ), DE + EF + FD = 2010 ∧ DE < EF ∧ EF ≤ FD ∧ 1 = EF - DE ∧ DE > 0 ∧ EF > 0 ∧ FD > 0 ∧ 
  DE + EF > FD ∧ DE + FD > EF ∧ EF + FD > DE :=
by {
  sorry
}

end NUMINAMATH_GPT_smallest_possible_value_of_EF_minus_DE_l1239_123976


namespace NUMINAMATH_GPT_third_term_of_arithmetic_sequence_l1239_123959

variable (a : ℕ → ℤ)
variable (a1_eq_2 : a 1 = 2)
variable (a2_eq_8 : a 2 = 8)
variable (arithmetic_seq : ∀ n : ℕ, a n = a 1 + (n - 1) * (a 2 - a 1))

theorem third_term_of_arithmetic_sequence :
  a 3 = 14 :=
by
  sorry

end NUMINAMATH_GPT_third_term_of_arithmetic_sequence_l1239_123959


namespace NUMINAMATH_GPT_bert_total_stamps_l1239_123947

theorem bert_total_stamps (bought_stamps : ℕ) (half_stamps_before : ℕ) (total_stamps_after : ℕ) :
  (bought_stamps = 300) ∧ (half_stamps_before = bought_stamps / 2) → (total_stamps_after = half_stamps_before + bought_stamps) → (total_stamps_after = 450) :=
by
  sorry

end NUMINAMATH_GPT_bert_total_stamps_l1239_123947


namespace NUMINAMATH_GPT_initial_people_count_l1239_123927

theorem initial_people_count (left remaining total : ℕ) (h1 : left = 6) (h2 : remaining = 5) : total = 11 :=
  by
  sorry

end NUMINAMATH_GPT_initial_people_count_l1239_123927


namespace NUMINAMATH_GPT_find_x_for_given_y_l1239_123985

theorem find_x_for_given_y (x y : ℝ) (h_pos : 0 < x ∧ 0 < y) (h_initial : x = 2 ∧ y = 8) (h_inverse : (2 ^ 3) * 8 = 128) :
  y = 1728 → x = (1 / (13.5) ^ (1 / 3)) :=
by
  sorry

end NUMINAMATH_GPT_find_x_for_given_y_l1239_123985


namespace NUMINAMATH_GPT_part_1_part_2_l1239_123995

noncomputable def f (a x : ℝ) : ℝ := a - 1 / (1 + 2^x)

theorem part_1 (a : ℝ) (h1 : f a 1 + f a (-1) = 0) : a = 1 / 2 :=
by sorry

theorem part_2 : ∃ a : ℝ, ∀ x : ℝ, f a (-x) + f a x = 0 :=
by sorry

end NUMINAMATH_GPT_part_1_part_2_l1239_123995


namespace NUMINAMATH_GPT_simplify_expression_l1239_123936

theorem simplify_expression :
  (3 * Real.sqrt 10) / (Real.sqrt 5 + 2) = 15 * Real.sqrt 2 - 6 * Real.sqrt 10 := 
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1239_123936


namespace NUMINAMATH_GPT_automobile_travel_distance_l1239_123956

theorem automobile_travel_distance 
  (a r : ℝ) 
  (travel_rate : ℝ) (h1 : travel_rate = a / 6)
  (time_in_seconds : ℝ) (h2 : time_in_seconds = 180):
  (3 * time_in_seconds * travel_rate) * (1 / r) * (1 / 3) = 10 * a / r :=
by
  sorry

end NUMINAMATH_GPT_automobile_travel_distance_l1239_123956


namespace NUMINAMATH_GPT_jessica_milk_problem_l1239_123900

theorem jessica_milk_problem (gallons_owned : ℝ) (gallons_given : ℝ) : gallons_owned = 5 → gallons_given = 16 / 3 → gallons_owned - gallons_given = -(1 / 3) :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  -- sorry

end NUMINAMATH_GPT_jessica_milk_problem_l1239_123900


namespace NUMINAMATH_GPT_symmetric_points_x_axis_l1239_123913

theorem symmetric_points_x_axis (a b : ℤ) 
  (h1 : a - 1 = 2) (h2 : 5 = -(b - 1)) : (a + b) ^ 2023 = -1 := 
by
  -- The proof steps will go here.
  sorry

end NUMINAMATH_GPT_symmetric_points_x_axis_l1239_123913


namespace NUMINAMATH_GPT_butterfingers_count_l1239_123967

theorem butterfingers_count (total_candy_bars : ℕ) (snickers : ℕ) (mars_bars : ℕ) (h_total : total_candy_bars = 12) (h_snickers : snickers = 3) (h_mars : mars_bars = 2) : 
  ∃ (butterfingers : ℕ), butterfingers = 7 :=
by
  sorry

end NUMINAMATH_GPT_butterfingers_count_l1239_123967
