import Mathlib

namespace NUMINAMATH_GPT_vector_BC_calculation_l1149_114913

/--
If \(\overrightarrow{AB} = (3, 6)\) and \(\overrightarrow{AC} = (1, 2)\),
then \(\overrightarrow{BC} = (-2, -4)\).
-/
theorem vector_BC_calculation (AB AC BC : ℤ × ℤ) 
  (hAB : AB = (3, 6))
  (hAC : AC = (1, 2)) : 
  BC = (-2, -4) := 
by
  sorry

end NUMINAMATH_GPT_vector_BC_calculation_l1149_114913


namespace NUMINAMATH_GPT_find_x_l1149_114999

variable (A B : Set ℕ)
variable (x : ℕ)

theorem find_x (hA : A = {1, 3}) (hB : B = {2, x}) (hUnion : A ∪ B = {1, 2, 3, 4}) : x = 4 := by
  sorry

end NUMINAMATH_GPT_find_x_l1149_114999


namespace NUMINAMATH_GPT_westbound_speed_is_275_l1149_114953

-- Define the conditions for the problem at hand.
def east_speed : ℕ := 325
def separation_time : ℝ := 3.5
def total_distance : ℕ := 2100

-- Compute the known east-bound distance.
def east_distance : ℝ := east_speed * separation_time

-- Define the speed of the west-bound plane as an unknown variable.
variable (v : ℕ)

-- Compute the west-bound distance.
def west_distance := v * separation_time

-- The assertion that the sum of two distances equals the total distance.
def distance_equation := east_distance + (v * separation_time) = total_distance

-- Prove that the west-bound speed is 275 mph.
theorem westbound_speed_is_275 : v = 275 :=
by
  sorry

end NUMINAMATH_GPT_westbound_speed_is_275_l1149_114953


namespace NUMINAMATH_GPT_intersection_A_B_l1149_114977

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | 2 * x^2 - x - 1 < 0}
def B : Set ℝ := {x : ℝ | Real.log x / Real.log (1/2) < 3}

-- Define the intersection A ∩ B and state the theorem
theorem intersection_A_B : A ∩ B = {x : ℝ | 1/8 < x ∧ x < 1} := by
   sorry

end NUMINAMATH_GPT_intersection_A_B_l1149_114977


namespace NUMINAMATH_GPT_sum_of_squares_l1149_114992

theorem sum_of_squares (k₁ k₂ k₃ : ℝ)
  (h_sum : k₁ + k₂ + k₃ = 1) : k₁^2 + k₂^2 + k₃^2 ≥ 1/3 :=
by sorry

end NUMINAMATH_GPT_sum_of_squares_l1149_114992


namespace NUMINAMATH_GPT_sum_of_numbers_l1149_114951

theorem sum_of_numbers (x : ℕ) (first_num second_num third_num sum : ℕ) 
  (h1 : 5 * x = first_num) 
  (h2 : 3 * x = second_num)
  (h3 : 4 * x = third_num) 
  (h4 : second_num = 27)
  : first_num + second_num + third_num = 108 :=
by {
  sorry
}

end NUMINAMATH_GPT_sum_of_numbers_l1149_114951


namespace NUMINAMATH_GPT_acres_used_for_corn_l1149_114937

-- Define the conditions
def total_acres : ℝ := 5746
def ratio_beans : ℝ := 7.5
def ratio_wheat : ℝ := 3.2
def ratio_corn : ℝ := 5.6
def total_parts : ℝ := ratio_beans + ratio_wheat + ratio_corn

-- Define the statement to prove
theorem acres_used_for_corn : (total_acres / total_parts) * ratio_corn = 1975.46 :=
by
  -- Placeholder for the proof; to be completed separately
  sorry

end NUMINAMATH_GPT_acres_used_for_corn_l1149_114937


namespace NUMINAMATH_GPT_chess_team_selection_l1149_114902

theorem chess_team_selection:
  let boys := 10
  let girls := 12
  let team_size := 8     -- total team size
  let boys_selected := 5 -- number of boys to select
  let girls_selected := 3 -- number of girls to select
  ∃ (w : ℕ), 
  (w = Nat.choose boys boys_selected * Nat.choose girls girls_selected) ∧ 
  w = 55440 :=
by
  sorry

end NUMINAMATH_GPT_chess_team_selection_l1149_114902


namespace NUMINAMATH_GPT_spring_outing_students_l1149_114932

variable (x y : ℕ)

theorem spring_outing_students (hx : x % 10 = 0) (hy : y % 10 = 0) (h1 : x + y = 1008) (h2 : y - x = 133) :
  x = 437 ∧ y = 570 :=
by
  sorry

end NUMINAMATH_GPT_spring_outing_students_l1149_114932


namespace NUMINAMATH_GPT_odot_subtraction_l1149_114986

-- Define the new operation
def odot (a b : ℚ) : ℚ := (a^3) / (b^2)

-- State the theorem
theorem odot_subtraction :
  ((odot (odot 2 4) 6) - (odot 2 (odot 4 6)) = -81 / 32) :=
by
  sorry

end NUMINAMATH_GPT_odot_subtraction_l1149_114986


namespace NUMINAMATH_GPT_bears_on_each_shelf_l1149_114980

theorem bears_on_each_shelf 
    (initial_bears : ℕ) (shipment_bears : ℕ) (shelves : ℕ)
    (h1 : initial_bears = 4) (h2 : shipment_bears = 10) (h3 : shelves = 2) :
    (initial_bears + shipment_bears) / shelves = 7 := by
  sorry

end NUMINAMATH_GPT_bears_on_each_shelf_l1149_114980


namespace NUMINAMATH_GPT_sum_of_largest_and_smallest_l1149_114914

theorem sum_of_largest_and_smallest (a b c : ℕ) (h1 : a = 10) (h2 : b = 11) (h3 : c = 12) :
  a + c = 22 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_largest_and_smallest_l1149_114914


namespace NUMINAMATH_GPT_tim_initial_books_l1149_114910

def books_problem : Prop :=
  ∃ T : ℕ, 10 + T - 24 = 19 ∧ T = 33

theorem tim_initial_books : books_problem :=
  sorry

end NUMINAMATH_GPT_tim_initial_books_l1149_114910


namespace NUMINAMATH_GPT_tax_computation_l1149_114911

def income : ℕ := 56000
def first_portion_income : ℕ := 40000
def first_portion_rate : ℝ := 0.12
def remaining_income : ℕ := income - first_portion_income
def remaining_rate : ℝ := 0.20
def expected_tax : ℝ := 8000

theorem tax_computation :
  (first_portion_rate * first_portion_income) +
  (remaining_rate * remaining_income) = expected_tax := by
  sorry

end NUMINAMATH_GPT_tax_computation_l1149_114911


namespace NUMINAMATH_GPT_rainfall_difference_l1149_114982

theorem rainfall_difference :
  let day1 := 26
  let day2 := 34
  let day3 := day2 - 12
  let total_rainfall := day1 + day2 + day3
  let average_rainfall := 140
  (average_rainfall - total_rainfall = 58) :=
by
  sorry

end NUMINAMATH_GPT_rainfall_difference_l1149_114982


namespace NUMINAMATH_GPT_largest_divisor_of_n_given_n_squared_divisible_by_72_l1149_114924

theorem largest_divisor_of_n_given_n_squared_divisible_by_72 (n : ℕ) (h1 : 0 < n) (h2 : 72 ∣ n^2) :
  ∃ q, q = 12 ∧ q ∣ n :=
by
  sorry

end NUMINAMATH_GPT_largest_divisor_of_n_given_n_squared_divisible_by_72_l1149_114924


namespace NUMINAMATH_GPT_man_and_son_together_days_l1149_114931

noncomputable def man_days : ℝ := 7
noncomputable def son_days : ℝ := 5.25
noncomputable def combined_days : ℝ := man_days * son_days / (man_days + son_days)

theorem man_and_son_together_days :
  combined_days = 7 / 5 :=
by
  sorry

end NUMINAMATH_GPT_man_and_son_together_days_l1149_114931


namespace NUMINAMATH_GPT_union_M_N_inter_complement_M_N_union_complement_M_N_l1149_114900

open Set

variable (U : Set ℝ) (M : Set ℝ) (N : Set ℝ)

noncomputable def universal_set := U = univ

def set_M := M = {x : ℝ | x ≤ 3}
def set_N := N = {x : ℝ | x < 1}

theorem union_M_N (hU : universal_set U) (hM : set_M M) (hN : set_N N) :
    M ∪ N = {x : ℝ | x ≤ 3} :=
by sorry

theorem inter_complement_M_N (hU : universal_set U) (hM : set_M M) (hN : set_N N) :
    (U \ M) ∩ N = ∅ :=
by sorry

theorem union_complement_M_N (hU : universal_set U) (hM : set_M M) (hN : set_N N) :
    (U \ M) ∪ (U \ N) = {x : ℝ | x ≥ 1} :=
by sorry

end NUMINAMATH_GPT_union_M_N_inter_complement_M_N_union_complement_M_N_l1149_114900


namespace NUMINAMATH_GPT_min_cost_correct_l1149_114993

noncomputable def min_cost_to_feed_group : ℕ :=
  let main_courses := 50
  let salads := 30
  let soups := 15
  let price_salad := 200
  let price_soup_main := 350
  let price_salad_main := 350
  let price_all_three := 500
  17000

theorem min_cost_correct : min_cost_to_feed_group = 17000 :=
by
  sorry

end NUMINAMATH_GPT_min_cost_correct_l1149_114993


namespace NUMINAMATH_GPT_placemat_length_correct_l1149_114945

noncomputable def placemat_length (r : ℝ) : ℝ :=
  2 * r * Real.sin (Real.pi / 8)

theorem placemat_length_correct (r : ℝ) (n : ℕ) (w : ℝ) (y : ℝ) (h_r : r = 5)
  (h_n : n = 8) (h_w : w = 1)
  (h_y : y = placemat_length r) :
  y = 5 * Real.sqrt (2 - Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_GPT_placemat_length_correct_l1149_114945


namespace NUMINAMATH_GPT_inequalities_hold_l1149_114928

theorem inequalities_hold (a b c : ℝ) (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) :
  (b / a > c / a) ∧ (b - a) / c > 0 ∧ (a - c) / (a * c) < 0 :=
by 
  sorry

end NUMINAMATH_GPT_inequalities_hold_l1149_114928


namespace NUMINAMATH_GPT_minimum_routes_l1149_114995

theorem minimum_routes (a b c : ℕ) (h1 : a + b ≥ 14) (h2 : b + c ≥ 14) (h3 : c + a ≥ 14) :
  a + b + c ≥ 21 :=
by sorry

end NUMINAMATH_GPT_minimum_routes_l1149_114995


namespace NUMINAMATH_GPT_gcd_228_1995_l1149_114933

theorem gcd_228_1995 :
  Nat.gcd 228 1995 = 21 :=
sorry

end NUMINAMATH_GPT_gcd_228_1995_l1149_114933


namespace NUMINAMATH_GPT_find_a_l1149_114997

-- Define the function f given a parameter a
def f (x a : ℝ) : ℝ := x^3 - 3*x^2 + a

-- Condition: f(x+1) is an odd function
theorem find_a (a : ℝ) (h : ∀ x : ℝ, f (-(x+1)) a = -f (x+1) a) : a = 2 := 
sorry

end NUMINAMATH_GPT_find_a_l1149_114997


namespace NUMINAMATH_GPT_min_value_inverse_sum_l1149_114939

theorem min_value_inverse_sum (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + 2 * b = 2) :
  (1 / a + 2 / b) ≥ 9 / 2 :=
sorry

end NUMINAMATH_GPT_min_value_inverse_sum_l1149_114939


namespace NUMINAMATH_GPT_residue_n_mod_17_l1149_114989

noncomputable def satisfies_conditions (m n k : ℕ) : Prop :=
  m^2 + 1 = 2 * n^2 ∧ 2 * m^2 + 1 = 11 * k^2 

theorem residue_n_mod_17 (m n k : ℕ) (h : satisfies_conditions m n k) : n % 17 = 5 :=
  sorry

end NUMINAMATH_GPT_residue_n_mod_17_l1149_114989


namespace NUMINAMATH_GPT_residual_at_sample_point_l1149_114981

theorem residual_at_sample_point :
  ∀ (x y : ℝ), (8 * x - 70 = 10) → (x = 10) → (y = 13) → (13 - (8 * x - 70) = 3) :=
by
  intros x y h1 h2 h3
  sorry

end NUMINAMATH_GPT_residual_at_sample_point_l1149_114981


namespace NUMINAMATH_GPT_minimize_expression_l1149_114944

open Real

theorem minimize_expression (p q r : ℝ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) :
  4 * p^3 + 6 * q^3 + 24 * r^3 + 8 / (3 * p * q * r) ≥ 16 :=
sorry

end NUMINAMATH_GPT_minimize_expression_l1149_114944


namespace NUMINAMATH_GPT_tangent_line_to_circle_l1149_114973

theorem tangent_line_to_circle : 
  ∀ (ρ θ : ℝ), (ρ = 4 * Real.sin θ) → (∃ ρ θ : ℝ, ρ * Real.cos θ = 2) :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_to_circle_l1149_114973


namespace NUMINAMATH_GPT_right_triangle_set_D_l1149_114921

theorem right_triangle_set_D : (5^2 + 12^2 = 13^2) ∧ 
  ((3^2 + 3^2 ≠ 5^2) ∧ (6^2 + 8^2 ≠ 9^2) ∧ (4^2 + 5^2 ≠ 6^2)) :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_set_D_l1149_114921


namespace NUMINAMATH_GPT_train_speed_l1149_114949

theorem train_speed
  (length_m : ℝ)
  (time_s : ℝ)
  (h_length : length_m = 280.0224)
  (h_time : time_s = 25.2) :
  (length_m / 1000) / (time_s / 3600) = 40.0032 :=
by
  sorry

end NUMINAMATH_GPT_train_speed_l1149_114949


namespace NUMINAMATH_GPT_find_y_value_l1149_114930

-- Define the angles in Lean
def angle1 (y : ℕ) : ℕ := 6 * y
def angle2 (y : ℕ) : ℕ := 7 * y
def angle3 (y : ℕ) : ℕ := 3 * y
def angle4 (y : ℕ) : ℕ := 2 * y

-- The condition that the sum of the angles is 360
def angles_sum_to_360 (y : ℕ) : Prop :=
  angle1 y + angle2 y + angle3 y + angle4 y = 360

-- The proof problem statement
theorem find_y_value (y : ℕ) (h : angles_sum_to_360 y) : y = 20 :=
sorry

end NUMINAMATH_GPT_find_y_value_l1149_114930


namespace NUMINAMATH_GPT_mother_duck_multiple_of_first_two_groups_l1149_114936

variables (num_ducklings : ℕ) (snails_first_batch : ℕ) (snails_second_batch : ℕ)
          (total_snails : ℕ) (mother_duck_snails : ℕ)

-- Given conditions
def conditions : Prop :=
  num_ducklings = 8 ∧ 
  snails_first_batch = 3 * 5 ∧ 
  snails_second_batch = 3 * 9 ∧ 
  total_snails = 294 ∧ 
  total_snails = snails_first_batch + snails_second_batch + 2 * mother_duck_snails ∧ 
  mother_duck_snails > 0

-- Our goal is to prove that the mother duck finds 3 times the snails the first two groups of ducklings find
theorem mother_duck_multiple_of_first_two_groups (h : conditions num_ducklings snails_first_batch snails_second_batch total_snails mother_duck_snails) : 
  mother_duck_snails / (snails_first_batch + snails_second_batch) = 3 :=
by 
  sorry

end NUMINAMATH_GPT_mother_duck_multiple_of_first_two_groups_l1149_114936


namespace NUMINAMATH_GPT_minimum_value_of_quadratic_function_l1149_114963

def quadratic_function (x : ℝ) : ℝ :=
  x^2 - 8 * x + 15

theorem minimum_value_of_quadratic_function :
  ∃ x : ℝ, quadratic_function x = -1 ∧ ∀ y : ℝ, quadratic_function y ≥ -1 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_quadratic_function_l1149_114963


namespace NUMINAMATH_GPT_sqrt_eq_cubrt_l1149_114943

theorem sqrt_eq_cubrt (x : ℝ) (h : Real.sqrt x = x^(1/3)) : x = 0 ∨ x = 1 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_eq_cubrt_l1149_114943


namespace NUMINAMATH_GPT_instantaneous_velocity_at_t4_l1149_114964

def position (t : ℝ) : ℝ := t^2 - t + 2

theorem instantaneous_velocity_at_t4 : 
  (deriv position 4) = 7 := 
by
  sorry

end NUMINAMATH_GPT_instantaneous_velocity_at_t4_l1149_114964


namespace NUMINAMATH_GPT_find_real_numbers_a_b_l1149_114940

noncomputable def f (a b x : ℝ) : ℝ :=
  a * (Real.sin x * Real.cos x) - (Real.sqrt 3) * a * (Real.cos x) ^ 2 + Real.sqrt 3 / 2 * a + b

theorem find_real_numbers_a_b (a b : ℝ) (h1 : 0 < a)
    (h2 : ∀ x ∈ Set.Icc (0 : ℝ) (Real.pi / 2), -2 ≤ f a b x ∧ f a b x ≤ Real.sqrt 3)
    : a = 2 ∧ b = -2 + Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_find_real_numbers_a_b_l1149_114940


namespace NUMINAMATH_GPT_sqrt_of_expression_l1149_114927

theorem sqrt_of_expression : Real.sqrt (5^2 * 7^6) = 1715 := 
by
  sorry

end NUMINAMATH_GPT_sqrt_of_expression_l1149_114927


namespace NUMINAMATH_GPT_total_packs_l1149_114906

theorem total_packs (cards_bought : ℕ) (cards_per_pack : ℕ) (num_people : ℕ)
  (h1 : cards_bought = 540) (h2 : cards_per_pack = 20) (h3 : num_people = 4) :
  (cards_bought / cards_per_pack) * num_people = 108 :=
by
  sorry

end NUMINAMATH_GPT_total_packs_l1149_114906


namespace NUMINAMATH_GPT_average_rate_dan_trip_l1149_114959

/-- 
Given:
- Dan runs along a 4-mile stretch of river and then swims back along the same route.
- Dan runs at a rate of 10 miles per hour.
- Dan swims at a rate of 6 miles per hour.

Prove:
Dan's average rate for the entire trip is 0.125 miles per minute.
-/
theorem average_rate_dan_trip :
  let distance := 4 -- miles
  let run_rate := 10 -- miles per hour
  let swim_rate := 6 -- miles per hour
  let time_run_hours := distance / run_rate -- hours
  let time_swim_hours := distance / swim_rate -- hours
  let time_run_minutes := time_run_hours * 60 -- minutes
  let time_swim_minutes := time_swim_hours * 60 -- minutes
  let total_distance := distance + distance -- miles
  let total_time := time_run_minutes + time_swim_minutes -- minutes
  let average_rate := total_distance / total_time -- miles per minute
  average_rate = 0.125 :=
by sorry

end NUMINAMATH_GPT_average_rate_dan_trip_l1149_114959


namespace NUMINAMATH_GPT_divisible_by_6_l1149_114917

theorem divisible_by_6 (n : ℤ) : 6 ∣ (n^3 - n + 6) :=
by
  sorry

end NUMINAMATH_GPT_divisible_by_6_l1149_114917


namespace NUMINAMATH_GPT_binomial_60_3_l1149_114965

theorem binomial_60_3 : Nat.choose 60 3 = 34220 := by
  sorry

end NUMINAMATH_GPT_binomial_60_3_l1149_114965


namespace NUMINAMATH_GPT_julias_change_l1149_114970

theorem julias_change :
  let snickers := 2
  let mms := 3
  let cost_snickers := 1.5
  let cost_mms := 2 * cost_snickers
  let money_given := 2 * 10
  let total_cost := snickers * cost_snickers + mms * cost_mms
  let change := money_given - total_cost
  change = 8 :=
by
  sorry

end NUMINAMATH_GPT_julias_change_l1149_114970


namespace NUMINAMATH_GPT_binary_to_decimal_l1149_114929

theorem binary_to_decimal : 
  (1 * 2^0 + 1 * 2^1 + 0 * 2^2 + 0 * 2^3 + 1 * 2^4 + 1 * 2^5) = 51 := by
  sorry

end NUMINAMATH_GPT_binary_to_decimal_l1149_114929


namespace NUMINAMATH_GPT_product_pattern_l1149_114994

theorem product_pattern (m n : ℝ) : 
  m * n = ( ( m + n ) / 2 ) ^ 2 - ( ( m - n ) / 2 ) ^ 2 := 
by 
  sorry

end NUMINAMATH_GPT_product_pattern_l1149_114994


namespace NUMINAMATH_GPT_min_value_of_2gx_sq_minus_fx_l1149_114971

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x + b
noncomputable def g (a c : ℝ) (x : ℝ) : ℝ := a * x + c

theorem min_value_of_2gx_sq_minus_fx (a b c : ℝ) (h_a_nonzero : a ≠ 0)
  (h_min_fx : ∃ x : ℝ, 2 * (f a b x)^2 - g a c x = 7 / 2) :
  ∃ x : ℝ, 2 * (g a c x)^2 - f a b x = -15 / 4 :=
sorry

end NUMINAMATH_GPT_min_value_of_2gx_sq_minus_fx_l1149_114971


namespace NUMINAMATH_GPT_election_winning_percentage_l1149_114979

def total_votes (a b c : ℕ) : ℕ := a + b + c

def winning_percentage (votes_winning : ℕ) (total : ℕ) : ℚ :=
(votes_winning * 100 : ℚ) / total

theorem election_winning_percentage (a b c : ℕ) (h_votes : a = 6136 ∧ b = 7636 ∧ c = 11628) :
  winning_percentage c (total_votes a b c) = 45.78 := by
  sorry

end NUMINAMATH_GPT_election_winning_percentage_l1149_114979


namespace NUMINAMATH_GPT_tel_aviv_rain_probability_l1149_114975

def binom (n k : ℕ) : ℕ := Nat.choose n k

def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (binom n k : ℝ) * (p ^ k) * ((1 - p) ^ (n - k))

theorem tel_aviv_rain_probability :
  binomial_probability 6 4 0.5 = 0.234375 :=
by
  sorry

end NUMINAMATH_GPT_tel_aviv_rain_probability_l1149_114975


namespace NUMINAMATH_GPT_range_of_a_l1149_114934

theorem range_of_a (a : ℝ) : 
  (∀ x y : ℝ, x ≤ a → (x + y + 1 ≤ 2 * (x + 1) - 3 * (y + 1))) → a ≤ -2 :=
by 
  intros h
  sorry

end NUMINAMATH_GPT_range_of_a_l1149_114934


namespace NUMINAMATH_GPT_find_y_l1149_114947

theorem find_y (n x y : ℕ) 
    (h1 : (n + 200 + 300 + x) / 4 = 250)
    (h2 : (300 + 150 + n + x + y) / 5 = 200) :
    y = 50 := 
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_find_y_l1149_114947


namespace NUMINAMATH_GPT_single_digit_pairs_l1149_114903

theorem single_digit_pairs:
  ∃ x y: ℕ, x ≠ 1 ∧ x ≠ 9 ∧ y ≠ 1 ∧ y ≠ 9 ∧ x < 10 ∧ y < 10 ∧ 
  (x * y < 100 ∧ ((x * y) % 10 + (x * y) / 10 == x ∨ (x * y) % 10 + (x * y) / 10 == y))
  → (x, y) ∈ [(3, 4), (3, 7), (6, 4), (6, 7)] :=
by
  sorry

end NUMINAMATH_GPT_single_digit_pairs_l1149_114903


namespace NUMINAMATH_GPT_find_a_value_l1149_114957

noncomputable def find_a (a : ℝ) : Prop :=
  (a > 0) ∧ (1 / 3 = 2 / a)

theorem find_a_value (a : ℝ) (h : find_a a) : a = 6 :=
sorry

end NUMINAMATH_GPT_find_a_value_l1149_114957


namespace NUMINAMATH_GPT_infinite_geometric_series_first_term_l1149_114919

theorem infinite_geometric_series_first_term 
  (r : ℝ) 
  (S : ℝ) 
  (a : ℝ) 
  (h1 : r = -3/7) 
  (h2 : S = 18) 
  (h3 : S = a / (1 - r)) : 
  a = 180 / 7 := by
  -- omitted proof
  sorry

end NUMINAMATH_GPT_infinite_geometric_series_first_term_l1149_114919


namespace NUMINAMATH_GPT_parallel_lines_slope_equal_intercepts_lines_l1149_114908

theorem parallel_lines_slope (m : ℝ) :
  (∀ x y, (2 * x - y - 3 = 0 ∧ x - m * y + 1 - 3 * m = 0) → 2 = (1 / m)) → m = 1 / 2 :=
by
  intro h
  sorry

theorem equal_intercepts_lines (m : ℝ) :
  (m ≠ 0 → (∀ x y, (x - m * y + 1 - 3 * m = 0) → (1 - 3 * m) / m = 3 * m - 1)) →
  (m = -1 ∨ m = 1 / 3) →
  ∀ x y, (x - m * y + 1 - 3 * m = 0) →
  (x + y + 4 = 0 ∨ 3 * x - y = 0) :=
by
  intro h hm
  sorry

end NUMINAMATH_GPT_parallel_lines_slope_equal_intercepts_lines_l1149_114908


namespace NUMINAMATH_GPT_possible_galina_numbers_l1149_114922

def is_divisible_by (m n : ℕ) : Prop := n % m = 0

def conditions_for_galina_number (n : ℕ) : Prop :=
  let C1 := is_divisible_by 7 n
  let C2 := is_divisible_by 11 n
  let C3 := n < 13
  let C4 := is_divisible_by 77 n
  (C1 ∧ ¬C2 ∧ C3 ∧ ¬C4) ∨ (¬C1 ∧ C2 ∧ C3 ∧ ¬C4)

theorem possible_galina_numbers (n : ℕ) :
  conditions_for_galina_number n ↔ (n = 7 ∨ n = 11) :=
by
  -- Proof to be filled in
  sorry

end NUMINAMATH_GPT_possible_galina_numbers_l1149_114922


namespace NUMINAMATH_GPT_goshawk_nature_reserve_l1149_114988

-- Define the problem statement and conditions
def percent_hawks (H W K : ℝ) : Prop :=
  ∃ H W K : ℝ,
    -- Condition 1: 35% of the birds are neither hawks, paddyfield-warblers, nor kingfishers
    1 - (H + W + K) = 0.35 ∧
    -- Condition 2: 40% of the non-hawks are paddyfield-warblers
    W = 0.40 * (1 - H) ∧
    -- Condition 3: There are 25% as many kingfishers as paddyfield-warblers
    K = 0.25 * W ∧
    -- Given all conditions, calculate the percentage of hawks
    H = 0.65

theorem goshawk_nature_reserve :
  ∃ H W K : ℝ,
    1 - (H + W + K) = 0.35 ∧
    W = 0.40 * (1 - H) ∧
    K = 0.25 * W ∧
    H = 0.65 := by
    -- Proof is omitted
    sorry

end NUMINAMATH_GPT_goshawk_nature_reserve_l1149_114988


namespace NUMINAMATH_GPT_value_of_x_l1149_114955

-- Define the conditions extracted from problem (a)
def condition1 (x : ℝ) : Prop := x^2 - 1 = 0
def condition2 (x : ℝ) : Prop := x - 1 ≠ 0

-- The statement to be proved
theorem value_of_x : ∀ x : ℝ, condition1 x → condition2 x → x = -1 :=
by
  intros x h1 h2
  sorry

end NUMINAMATH_GPT_value_of_x_l1149_114955


namespace NUMINAMATH_GPT_monotonically_decreasing_range_l1149_114987

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 3 * x^2 - x + 1
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 6 * x - 1

theorem monotonically_decreasing_range (a : ℝ) :
  (∀ x : ℝ, f' a x ≤ 0) → a ≤ -3 := by
  sorry

end NUMINAMATH_GPT_monotonically_decreasing_range_l1149_114987


namespace NUMINAMATH_GPT_find_a_l1149_114976

theorem find_a (a : ℝ) (h : (3 * a + 2) + (a + 14) = 0) : a = -4 :=
sorry

end NUMINAMATH_GPT_find_a_l1149_114976


namespace NUMINAMATH_GPT_car_distance_l1149_114918

theorem car_distance 
  (speed : ℝ) 
  (time : ℝ) 
  (distance : ℝ) 
  (h_speed : speed = 160) 
  (h_time : time = 5) 
  (h_dist_formula : distance = speed * time) : 
  distance = 800 :=
by sorry

end NUMINAMATH_GPT_car_distance_l1149_114918


namespace NUMINAMATH_GPT_all_non_positive_l1149_114990

theorem all_non_positive (n : ℕ) (a : ℕ → ℤ) 
  (h₀ : a 0 = 0) 
  (hₙ : a n = 0) 
  (ineq : ∀ k, 1 ≤ k ∧ k ≤ n - 1 → a (k - 1) - 2 * a k + a (k + 1) ≥ 0) : ∀ k, a k ≤ 0 :=
by 
  sorry

end NUMINAMATH_GPT_all_non_positive_l1149_114990


namespace NUMINAMATH_GPT_pow_sub_nat_ge_seven_l1149_114958

open Nat

theorem pow_sub_nat_ge_seven
  (m n : ℕ) 
  (h1 : m > 1)
  (h2 : 2^(2 * m + 1) - n^2 ≥ 0) : 
  2^(2 * m + 1) - n^2 ≥ 7 :=
sorry

end NUMINAMATH_GPT_pow_sub_nat_ge_seven_l1149_114958


namespace NUMINAMATH_GPT_problem_statement_l1149_114905

theorem problem_statement (a b : ℝ) (h : a ≠ b) : (a - b) ^ 2 > 0 := sorry

end NUMINAMATH_GPT_problem_statement_l1149_114905


namespace NUMINAMATH_GPT_percentage_of_600_eq_half_of_900_l1149_114926

theorem percentage_of_600_eq_half_of_900 : 
  ∃ P : ℝ, (P / 100) * 600 = 0.5 * 900 ∧ P = 75 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_percentage_of_600_eq_half_of_900_l1149_114926


namespace NUMINAMATH_GPT_solution_set_ineq_l1149_114968

theorem solution_set_ineq (x : ℝ) :
  x * (2 * x^2 - 3 * x + 1) ≤ 0 ↔ (x ≤ 0 ∨ (1/2 ≤ x ∧ x ≤ 1)) :=
sorry

end NUMINAMATH_GPT_solution_set_ineq_l1149_114968


namespace NUMINAMATH_GPT_find_n_l1149_114978

theorem find_n (x : ℝ) (n : ℝ)
  (h1 : Real.log (Real.sin x) + Real.log (Real.cos x) = -2)
  (h2 : Real.log (Real.sin x + Real.cos x) = 1 / 2 * (Real.log n - 2)) :
  n = Real.exp 2 + 2 :=
by
  sorry

end NUMINAMATH_GPT_find_n_l1149_114978


namespace NUMINAMATH_GPT_stu_books_count_l1149_114907

noncomputable def elmo_books : ℕ := 24
noncomputable def laura_books : ℕ := elmo_books / 3
noncomputable def stu_books : ℕ := laura_books / 2

theorem stu_books_count :
  stu_books = 4 :=
by
  sorry

end NUMINAMATH_GPT_stu_books_count_l1149_114907


namespace NUMINAMATH_GPT_parallel_lines_iff_l1149_114961

theorem parallel_lines_iff (a : ℝ) :
  (∀ x y : ℝ, x - y - 1 = 0 → x + a * y - 2 = 0) ↔ (a = -1) :=
by
  sorry

end NUMINAMATH_GPT_parallel_lines_iff_l1149_114961


namespace NUMINAMATH_GPT_race_problem_l1149_114935

theorem race_problem (a_speed b_speed : ℕ) (A B : ℕ) (finish_dist : ℕ)
  (h1 : finish_dist = 3000)
  (h2 : A = finish_dist - 500)
  (h3 : B = finish_dist - 600)
  (h4 : A / a_speed = B / b_speed)
  (h5 : a_speed / b_speed = 25 / 24) :
  B - ((500 * b_speed) / a_speed) = 120 :=
by
  sorry

end NUMINAMATH_GPT_race_problem_l1149_114935


namespace NUMINAMATH_GPT_find_m_l1149_114985

theorem find_m (m : ℤ) (h₀ : 0 ≤ m) (h₁ : m < 31) (h₂ : 79453 % 31 = m) : m = 0 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l1149_114985


namespace NUMINAMATH_GPT_find_number_l1149_114938

theorem find_number (x : ℝ) (h : x - (3 / 5) * x = 64) : x = 160 :=
sorry

end NUMINAMATH_GPT_find_number_l1149_114938


namespace NUMINAMATH_GPT_solve_for_a_l1149_114956

theorem solve_for_a (a : ℚ) (h : a + a / 4 = 10 / 4) : a = 2 :=
sorry

end NUMINAMATH_GPT_solve_for_a_l1149_114956


namespace NUMINAMATH_GPT_meaningful_expression_range_l1149_114983

theorem meaningful_expression_range (x : ℝ) (h1 : 3 * x + 2 ≥ 0) (h2 : x ≠ 0) : 
  x ∈ Set.Ico (-2 / 3) 0 ∪ Set.Ioi 0 := 
  sorry

end NUMINAMATH_GPT_meaningful_expression_range_l1149_114983


namespace NUMINAMATH_GPT_part1_prob_dist_part1_expectation_part2_prob_A_wins_n_throws_l1149_114920

-- Definitions for part (1)
def P_X_1 : ℚ := 1 / 6
def P_X_2 : ℚ := 5 / 36
def P_X_3 : ℚ := 25 / 216
def P_X_4 : ℚ := 125 / 216
def E_X : ℚ := 671 / 216

theorem part1_prob_dist (X : ℚ) :
  (X = 1 → P_X_1 = 1 / 6) ∧
  (X = 2 → P_X_2 = 5 / 36) ∧
  (X = 3 → P_X_3 = 25 / 216) ∧
  (X = 4 → P_X_4 = 125 / 216) := 
by sorry

theorem part1_expectation :
  E_X = 671 / 216 :=
by sorry

-- Definition for part (2)
def P_A_wins_n_throws (n : ℕ) : ℚ := 1 / 6 * (5 / 6) ^ (2 * n - 2)

theorem part2_prob_A_wins_n_throws (n : ℕ) (hn : n ≥ 1) :
  P_A_wins_n_throws n = 1 / 6 * (5 / 6) ^ (2 * n - 2) :=
by sorry

end NUMINAMATH_GPT_part1_prob_dist_part1_expectation_part2_prob_A_wins_n_throws_l1149_114920


namespace NUMINAMATH_GPT_correct_equation_l1149_114952

theorem correct_equation (x : ℤ) : 232 + x = 3 * (146 - x) :=
sorry

end NUMINAMATH_GPT_correct_equation_l1149_114952


namespace NUMINAMATH_GPT_infinite_series_sum_l1149_114954

/-- The sum of the infinite series ∑ 1/(n(n+3)) for n from 1 to ∞ is 7/9. -/
theorem infinite_series_sum :
  ∑' n, (1 : ℝ) / (n * (n + 3)) = 7 / 9 :=
sorry

end NUMINAMATH_GPT_infinite_series_sum_l1149_114954


namespace NUMINAMATH_GPT_inverse_function_passes_through_point_a_l1149_114915

theorem inverse_function_passes_through_point_a
  (a : ℝ) (ha_pos : 0 < a) (ha_neq_one : a ≠ 1) :
  ∃ (A : ℝ × ℝ), A = (2, 3) ∧ (∀ x, (a^(x-3) + 1) = 2 ↔ x = 3) → (2 - 1)/(3-3) = 0 :=
by
  sorry

end NUMINAMATH_GPT_inverse_function_passes_through_point_a_l1149_114915


namespace NUMINAMATH_GPT_correct_option_is_C_l1149_114916

-- Definitions of the expressions given in the conditions
def optionA (a : ℝ) : ℝ := 3 * a^5 - a^5
def optionB (a : ℝ) : ℝ := a^2 + a^5
def optionC (a : ℝ) : ℝ := a^5 + a^5
def optionD (x y : ℝ) : ℝ := x^2 * y + x * y^2

-- The problem is to prove that optionC is correct and the others are not
theorem correct_option_is_C (a x y : ℝ) :
  (optionC a = 2 * a^5) ∧ 
  (optionA a ≠ 3) ∧ 
  (optionB a ≠ a^7) ∧ 
  (optionD x y ≠ 2 * (x ^ 3) * (y ^ 3)) :=
by
  sorry

end NUMINAMATH_GPT_correct_option_is_C_l1149_114916


namespace NUMINAMATH_GPT_sum_on_simple_interest_is_1400_l1149_114991

noncomputable def sum_placed_on_simple_interest : ℝ :=
  let P_c := 4000
  let r := 0.10
  let n := 1
  let t_c := 2
  let t_s := 3
  let A := P_c * (1 + r / n)^(n * t_c)
  let CI := A - P_c
  let SI := CI / 2
  100 * SI / (r * t_s)

theorem sum_on_simple_interest_is_1400 : sum_placed_on_simple_interest = 1400 := by
  sorry

end NUMINAMATH_GPT_sum_on_simple_interest_is_1400_l1149_114991


namespace NUMINAMATH_GPT_number_of_flower_sets_l1149_114998

theorem number_of_flower_sets (total_flowers : ℕ) (flowers_per_set : ℕ) (sets : ℕ) 
  (h1 : total_flowers = 270) 
  (h2 : flowers_per_set = 90) 
  (h3 : sets = total_flowers / flowers_per_set) : 
  sets = 3 := 
by 
  sorry

end NUMINAMATH_GPT_number_of_flower_sets_l1149_114998


namespace NUMINAMATH_GPT_nickels_left_l1149_114962

theorem nickels_left (n b : ℕ) (h₁ : n = 31) (h₂ : b = 20) : n - b = 11 :=
by
  sorry

end NUMINAMATH_GPT_nickels_left_l1149_114962


namespace NUMINAMATH_GPT_part1_part2_l1149_114967

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (Real.exp x / x) - Real.log x + x - a

theorem part1 (a : ℝ) : (∀ x > 0, f x a ≥ 0) ↔ a ≤ Real.exp 1 + 1 :=
  sorry

theorem part2 (a : ℝ) (x1 x2 : ℝ) (h1 : f x1 a = 0) (h2 : f x2 a = 0) : x1 * x2 < 1 :=
  sorry

end NUMINAMATH_GPT_part1_part2_l1149_114967


namespace NUMINAMATH_GPT_determine_C_cards_l1149_114972

-- Define the card numbers
def card_numbers : List ℕ := [1,2,3,4,5,6,7,8,9,10,11,12]

-- Define the card sum each person should have
def card_sum := 26

-- Define person's cards
def A_cards : List ℕ := [10, 12]
def B_cards : List ℕ := [6, 11]

-- Define sum constraints for A and B
def sum_A := A_cards.sum
def sum_B := B_cards.sum

-- Define C's complete set of numbers based on remaining cards and sum constraints
def remaining_cards := card_numbers.diff (A_cards ++ B_cards)
def sum_remaining := remaining_cards.sum

theorem determine_C_cards :
  (sum_A + (26 - sum_A)) = card_sum ∧
  (sum_B + (26 - sum_B)) = card_sum ∧
  (sum_remaining = card_sum) → 
  (remaining_cards = [8, 9]) :=
by
  sorry

end NUMINAMATH_GPT_determine_C_cards_l1149_114972


namespace NUMINAMATH_GPT_third_month_sale_l1149_114966

theorem third_month_sale (s1 s2 s4 s5 s6 avg_sale: ℕ) (h1: s1 = 5420) (h2: s2 = 5660) (h3: s4 = 6350) (h4: s5 = 6500) (h5: s6 = 8270) (h6: avg_sale = 6400) :
  ∃ s3: ℕ, s3 = 6200 :=
by
  sorry

end NUMINAMATH_GPT_third_month_sale_l1149_114966


namespace NUMINAMATH_GPT_greatest_divisor_l1149_114948

theorem greatest_divisor (d : ℕ) :
  (6215 % d = 23 ∧ 7373 % d = 29 ∧ 8927 % d = 35) → d = 36 :=
by
  sorry

end NUMINAMATH_GPT_greatest_divisor_l1149_114948


namespace NUMINAMATH_GPT_maximum_value_condition_l1149_114904

open Real

theorem maximum_value_condition {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (h1 : x + y = 16) (h2 : x = 2 * y) :
  (1 / x + 1 / y) = 9 / 32 :=
by
  sorry

end NUMINAMATH_GPT_maximum_value_condition_l1149_114904


namespace NUMINAMATH_GPT_solve_for_x_l1149_114941

variable (x : ℝ)

theorem solve_for_x (h : 0.05 * x + 0.12 * (30 + x) = 15.6) : x = 12 / 0.17 := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1149_114941


namespace NUMINAMATH_GPT_number_of_male_students_l1149_114996

variables (total_students sample_size female_sampled female_students male_students : ℕ)
variables (h_total : total_students = 1600)
variables (h_sample : sample_size = 200)
variables (h_female_sampled : female_sampled = 95)
variables (h_prob : (sample_size : ℚ) / total_students = (female_sampled : ℚ) / female_students)
variables (h_female_students : female_students = 760)

theorem number_of_male_students : male_students = total_students - female_students := by
  sorry

end NUMINAMATH_GPT_number_of_male_students_l1149_114996


namespace NUMINAMATH_GPT_like_terms_mn_l1149_114901

theorem like_terms_mn (m n : ℕ) (h1 : -2 * x^m * y^2 = 2 * x^3 * y^n) : m * n = 6 :=
by {
  -- Add the statements transforming the assumptions into intermediate steps
  sorry
}

end NUMINAMATH_GPT_like_terms_mn_l1149_114901


namespace NUMINAMATH_GPT_arithmetic_sequence_l1149_114912

noncomputable def M (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.sum (Finset.range n) (λ i => a (i + 1))) / n

theorem arithmetic_sequence (a : ℕ → ℝ) (C : ℝ)
  (h : ∀ {i j k : ℕ}, i ≠ j → j ≠ k → k ≠ i →
    (i - j) * M a k + (j - k) * M a i + (k - i) * M a j = C) :
  ∃ (d : ℝ), ∀ n : ℕ, a (n + 1) = a 1 + n * d :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_l1149_114912


namespace NUMINAMATH_GPT_find_rate_percent_l1149_114909

-- Define the conditions
def principal : ℝ := 1200
def time : ℝ := 4
def simple_interest : ℝ := 400

-- Define the rate that we need to prove
def rate : ℝ := 8.3333  -- approximately

-- Formalize the proof problem in Lean 4
theorem find_rate_percent
  (P : ℝ) (T : ℝ) (SI : ℝ) (R : ℝ)
  (hP : P = principal) (hT : T = time) (hSI : SI = simple_interest) :
  SI = (P * R * T) / 100 → R = rate :=
by
  intros h
  sorry

end NUMINAMATH_GPT_find_rate_percent_l1149_114909


namespace NUMINAMATH_GPT_product_xyz_l1149_114925

theorem product_xyz (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z)
  (h4 : x * y = 30 * (4:ℝ)^(1/3)) (h5 : x * z = 45 * (4:ℝ)^(1/3)) (h6 : y * z = 18 * (4:ℝ)^(1/3)) :
  x * y * z = 540 * Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_product_xyz_l1149_114925


namespace NUMINAMATH_GPT_population_meets_capacity_l1149_114950

-- Define the initial conditions and parameters
def initial_year : ℕ := 1998
def initial_population : ℕ := 100
def population_growth_rate : ℕ := 4  -- quadruples every 20 years
def years_per_growth_period : ℕ := 20
def land_area_hectares : ℕ := 15000
def hectares_per_person : ℕ := 2
def maximum_capacity : ℕ := land_area_hectares / hectares_per_person

-- Define the statement
theorem population_meets_capacity :
  ∃ (years_from_initial : ℕ), years_from_initial = 60 ∧
  initial_population * population_growth_rate ^ (years_from_initial / years_per_growth_period) ≥ maximum_capacity :=
by
  sorry

end NUMINAMATH_GPT_population_meets_capacity_l1149_114950


namespace NUMINAMATH_GPT_solve_for_f_8_l1149_114946

noncomputable def f (x : ℝ) : ℝ := (Real.logb 2 x)

theorem solve_for_f_8 {x : ℝ} (h : f (x^3) = Real.logb 2 x) : f 8 = 1 :=
by
sorry

end NUMINAMATH_GPT_solve_for_f_8_l1149_114946


namespace NUMINAMATH_GPT_problem1_problem2_l1149_114974

-- Definition of the function f(x)
def f (x a : ℝ) : ℝ := |2 * x - a| + |2 * x - 1|

-- 1st problem: Prove the solution set for f(x) ≤ 2 when a = -1 is { x | x = ± 1/2 }
theorem problem1 : (∀ x : ℝ, f x (-1) ≤ 2 ↔ x = 1/2 ∨ x = -1/2) :=
by sorry

-- 2nd problem: Prove the range of real number a is [0, 3]
theorem problem2 : (∃ a : ℝ, (∀ x ∈ Set.Icc (1/2:ℝ) 1, f x a ≤ |2 * x + 1| ) ↔ 0 ≤ a ∧ a ≤ 3) :=
by sorry

end NUMINAMATH_GPT_problem1_problem2_l1149_114974


namespace NUMINAMATH_GPT_probability_adjacent_A_before_B_l1149_114969

theorem probability_adjacent_A_before_B 
  (total_students : ℕ)
  (A B C D : ℚ)
  (hA : total_students = 8)
  (hB : B = 1/3) : 
  (∃ prob : ℚ, prob = 1/3) :=
by
  sorry

end NUMINAMATH_GPT_probability_adjacent_A_before_B_l1149_114969


namespace NUMINAMATH_GPT_woman_total_coins_l1149_114923

theorem woman_total_coins
  (num_each_coin : ℕ)
  (h : 1 * num_each_coin + 5 * num_each_coin + 10 * num_each_coin + 25 * num_each_coin + 100 * num_each_coin = 351)
  : 5 * num_each_coin = 15 :=
by
  sorry

end NUMINAMATH_GPT_woman_total_coins_l1149_114923


namespace NUMINAMATH_GPT_solve_x_l1149_114942

theorem solve_x (x y : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : 8 * x ^ 2 + 16 * x * y = x ^ 3 + 3 * x ^ 2 * y) (h₄ : y = 2 * x) : x = 40 / 7 :=
by
  sorry

end NUMINAMATH_GPT_solve_x_l1149_114942


namespace NUMINAMATH_GPT_negation_of_P_l1149_114984

theorem negation_of_P : ¬(∀ x : ℝ, x^2 + 1 ≥ 2 * x) ↔ ∃ x : ℝ, x^2 + 1 < 2 * x :=
by sorry

end NUMINAMATH_GPT_negation_of_P_l1149_114984


namespace NUMINAMATH_GPT_freezer_temp_correct_l1149_114960

variable (t_refrigeration : ℝ) (t_freezer : ℝ)

-- Given conditions
def refrigeration_temperature := t_refrigeration = 5
def freezer_temperature := t_freezer = -12

-- Goal: Prove that the freezer compartment's temperature is -12 degrees Celsius
theorem freezer_temp_correct : freezer_temperature t_freezer := by
  sorry

end NUMINAMATH_GPT_freezer_temp_correct_l1149_114960
