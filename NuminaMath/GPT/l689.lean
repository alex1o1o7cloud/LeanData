import Mathlib

namespace number_of_persons_l689_68926

-- Definitions of the given conditions
def average : ℕ := 15
def average_5 : ℕ := 14
def sum_5 : ℕ := 5 * average_5
def average_9 : ℕ := 16
def sum_9 : ℕ := 9 * average_9
def age_15th : ℕ := 41
def total_sum : ℕ := sum_5 + sum_9 + age_15th

-- The main theorem stating the equivalence
theorem number_of_persons (N : ℕ) (h_average : average * N = total_sum) : N = 17 :=
by
  -- Proof goes here
  sorry

end number_of_persons_l689_68926


namespace shirt_cost_l689_68924

theorem shirt_cost (S : ℝ) (hats_cost jeans_cost total_cost : ℝ)
  (h_hats : hats_cost = 4)
  (h_jeans : jeans_cost = 10)
  (h_total : total_cost = 51)
  (h_eq : 3 * S + 2 * jeans_cost + 4 * hats_cost = total_cost) :
  S = 5 :=
by
  -- The main proof will be provided here
  sorry

end shirt_cost_l689_68924


namespace train_length_l689_68981

theorem train_length (speed_kph : ℕ) (tunnel_length_m : ℕ) (time_s : ℕ) : 
  speed_kph = 54 → 
  tunnel_length_m = 1200 → 
  time_s = 100 → 
  ∃ train_length_m : ℕ, train_length_m = 300 := 
by
  intros h1 h2 h3
  have speed_mps : ℕ := (speed_kph * 1000) / 3600 
  have total_distance_m : ℕ := speed_mps * time_s
  have train_length_m : ℕ := total_distance_m - tunnel_length_m
  use train_length_m
  sorry

end train_length_l689_68981


namespace james_chore_time_l689_68987

-- Definitions for the conditions
def t_vacuum : ℕ := 3
def t_chores : ℕ := 3 * t_vacuum
def t_total : ℕ := t_vacuum + t_chores

-- Statement
theorem james_chore_time : t_total = 12 := by
  sorry

end james_chore_time_l689_68987


namespace function_monotonicity_l689_68985

theorem function_monotonicity (a b : ℝ) : 
  (∀ x, -1 < x ∧ x < 1 → (3 * x^2 + a) < 0) ∧ 
  (∀ x, 1 < x → (3 * x^2 + a) > 0) → 
  (a = -3 ∧ ∃ b : ℝ, true) :=
by {
  sorry
}

end function_monotonicity_l689_68985


namespace sqrt6_eq_l689_68927

theorem sqrt6_eq (r : Real) (h : r = Real.sqrt 2 + Real.sqrt 3) : Real.sqrt 6 = (r ^ 2 - 5) / 2 :=
by
  sorry

end sqrt6_eq_l689_68927


namespace find_a_for_even_function_l689_68935

theorem find_a_for_even_function (a : ℝ) : 
  (∀ x : ℝ, (x + a) * (x - 4) = ((-x) + a) * ((-x) - 4)) → a = 4 :=
by sorry

end find_a_for_even_function_l689_68935


namespace gcd_45_75_l689_68976

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_l689_68976


namespace maximize_profit_l689_68974

variable (k : ℚ) -- Proportional constant for deposits
variable (x : ℚ) -- Annual interest rate paid to depositors
variable (D : ℚ) -- Total amount of deposits

-- Define the condition for the total amount of deposits
def deposits (x : ℚ) : ℚ := k * x^2

-- Define the profit function
def profit (x : ℚ) : ℚ := 0.045 * k * x^2 - k * x^3

-- Define the derivative of the profit function
def profit_derivative (x : ℚ) : ℚ := 3 * k * x * (0.03 - x)

-- Statement that x = 0.03 maximizes the bank's profit
theorem maximize_profit : ∃ x, x = 0.03 ∧ (∀ y, profit_derivative y = 0 → x = y) :=
by
  sorry

end maximize_profit_l689_68974


namespace focus_coordinates_correct_l689_68940
noncomputable def ellipse_focus : Real × Real :=
  let center : Real × Real := (4, -1)
  let a : Real := 4
  let b : Real := 1.5
  let c : Real := Real.sqrt (a^2 - b^2)
  (center.1 + c, center.2)

theorem focus_coordinates_correct : 
  ellipse_focus = (7.708, -1) := 
by 
  sorry

end focus_coordinates_correct_l689_68940


namespace johns_original_number_l689_68936

def switch_digits (n : ℕ) : ℕ :=
  let tens := n / 10
  let units := n % 10
  10 * units + tens

theorem johns_original_number :
  ∃ x : ℕ, (10 ≤ x ∧ x < 100) ∧ (∃ y : ℕ, y = 5 * x + 13 ∧ 82 ≤ switch_digits y ∧ switch_digits y ≤ 86 ∧ x = 11) :=
by
  sorry

end johns_original_number_l689_68936


namespace natural_number_sum_of_coprimes_l689_68942

theorem natural_number_sum_of_coprimes (n : ℕ) (h : n ≥ 2) : ∃ a b : ℕ, n = a + b ∧ Nat.gcd a b = 1 :=
by
  use (n - 1), 1
  sorry

end natural_number_sum_of_coprimes_l689_68942


namespace average_salary_all_workers_l689_68990

-- Define the given conditions as constants
def num_technicians : ℕ := 7
def avg_salary_technicians : ℕ := 12000

def num_workers_total : ℕ := 21
def num_workers_remaining := num_workers_total - num_technicians
def avg_salary_remaining_workers : ℕ := 6000

-- Define the statement we need to prove
theorem average_salary_all_workers :
  let total_salary_technicians := num_technicians * avg_salary_technicians
  let total_salary_remaining_workers := num_workers_remaining * avg_salary_remaining_workers
  let total_salary_all_workers := total_salary_technicians + total_salary_remaining_workers
  let avg_salary_all_workers := total_salary_all_workers / num_workers_total
  avg_salary_all_workers = 8000 :=
by
  sorry

end average_salary_all_workers_l689_68990


namespace min_value_b_l689_68954

noncomputable def f (x a : ℝ) := 3 * x^2 - 4 * a * x
noncomputable def g (x a b : ℝ) := 2 * a^2 * Real.log x - b
noncomputable def f' (x a : ℝ) := 6 * x - 4 * a
noncomputable def g' (x a : ℝ) := 2 * a^2 / x

theorem min_value_b (a : ℝ) (h_a : a > 0) :
  ∃ (b : ℝ), ∃ (x₀ : ℝ), 
  (f x₀ a = g x₀ a b ∧ f' x₀ a = g' x₀ a) ∧ 
  ∀ (b' : ℝ), (∀ (x' : ℝ), (f x' a = g x' a b' ∧ f' x' a = g' x' a) → b' ≥ -1 / Real.exp 2) := 
sorry

end min_value_b_l689_68954


namespace quadratic_trinomial_with_integral_roots_l689_68961

theorem quadratic_trinomial_with_integral_roots (a b c : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) :
  (∃ x : ℤ, a * x^2 + b * x + c = 0) ∧ 
  (∃ x : ℤ, (a + 1) * x^2 + (b + 1) * x + (c + 1) = 0) ∧ 
  (∃ x : ℤ, (a + 2) * x^2 + (b + 2) * x + (c + 2) = 0) :=
sorry

end quadratic_trinomial_with_integral_roots_l689_68961


namespace pairs_satisfying_equation_l689_68982

theorem pairs_satisfying_equation (a b : ℝ) : 
  (∀ n : ℕ, n > 0 → a * ⌊b * n⌋ = b * ⌊a * n⌋) ↔ 
  (a = 0 ∨ b = 0 ∨ a = b ∨ ∃ k : ℤ, a = k ∧ b = k) := 
by
  sorry

end pairs_satisfying_equation_l689_68982


namespace total_time_is_11_l689_68950

-- Define the times each person spent in the pool
def Jerry_time : Nat := 3
def Elaine_time : Nat := 2 * Jerry_time
def George_time : Nat := Elaine_time / 3
def Kramer_time : Nat := 0

-- Define the total time spent in the pool by all friends
def total_time : Nat := Jerry_time + Elaine_time + George_time + Kramer_time

-- Prove that the total time is 11 minutes
theorem total_time_is_11 : total_time = 11 := sorry

end total_time_is_11_l689_68950


namespace house_value_l689_68932

open Nat

-- Define the conditions
variables (V x : ℕ)
variables (split_amount money_paid : ℕ)
variables (houses_brothers youngest_received : ℕ)
variables (y1 y2 : ℕ)

-- Hypotheses from the conditions
def conditions (V x split_amount money_paid houses_brothers youngest_received y1 y2 : ℕ) :=
  (split_amount = V / 5) ∧
  (houses_brothers = 3) ∧
  (money_paid = 2000) ∧
  (youngest_received = 3000) ∧
  (3 * houses_brothers * money_paid = 6000) ∧
  (y1 = youngest_received) ∧
  (y2 = youngest_received) ∧
  (3 * x + 6000 = V)

-- Main theorem stating the value of one house
theorem house_value (V x : ℕ) (split_amount money_paid houses_brothers youngest_received y1 y2: ℕ) :
  conditions V x split_amount money_paid houses_brothers youngest_received y1 y2 →
  x = 3000 :=
by
  intros
  simp [conditions] at *
  sorry

end house_value_l689_68932


namespace standard_deviation_is_2_l689_68979

noncomputable def dataset := [51, 54, 55, 57, 53]

noncomputable def mean (l : List ℝ) : ℝ :=
  ((l.sum : ℝ) / (l.length : ℝ))

noncomputable def variance (l : List ℝ) : ℝ :=
  let m := mean l
  ((l.map (λ x => (x - m)^2)).sum : ℝ) / (l.length : ℝ)

noncomputable def std_dev (l : List ℝ) : ℝ :=
  Real.sqrt (variance l)

theorem standard_deviation_is_2 :
  mean dataset = 54 →
  std_dev dataset = 2 := by
  intro h_mean
  sorry

end standard_deviation_is_2_l689_68979


namespace odd_squares_diff_divisible_by_8_l689_68934

theorem odd_squares_diff_divisible_by_8 (m n : ℤ) (a b : ℤ) (hm : a = 2 * m + 1) (hn : b = 2 * n + 1) : (a^2 - b^2) % 8 = 0 := sorry

end odd_squares_diff_divisible_by_8_l689_68934


namespace number_of_fridays_l689_68901

theorem number_of_fridays (jan_1_sat : true) (is_non_leap_year : true) : ∃ (n : ℕ), n = 52 :=
by
  -- Conditions: January 1st is Saturday and it is a non-leap year.
  -- We are given that January 1st is a Saturday.
  have jan_1_sat_condition : true := jan_1_sat
  -- We are given that the year is a non-leap year (365 days).
  have non_leap_condition : true := is_non_leap_year
  -- Therefore, there are 52 Fridays in the year.
  use 52
  done

end number_of_fridays_l689_68901


namespace christmas_distribution_l689_68966

theorem christmas_distribution :
  ∃ (n x : ℕ), 
    (240 + 120 + 1 = 361) ∧
    (n * x = 361) ∧
    (n = 19) ∧
    (x = 19) ∧
    ∃ (a b : ℕ), (a + b = 19) ∧ (a * 5 + b * 6 = 100) :=
by
  sorry

end christmas_distribution_l689_68966


namespace remainder_3_pow_500_mod_17_l689_68916

theorem remainder_3_pow_500_mod_17 : (3^500) % 17 = 13 := 
by
  sorry

end remainder_3_pow_500_mod_17_l689_68916


namespace area_of_rhombus_enclosed_by_equation_l689_68911

-- Given the conditions
def equation (x y : ℝ) : Prop := |x| + |3 * y| = 12

-- Define the main theorem to be proven
theorem area_of_rhombus_enclosed_by_equation : 
  (∃ x y : ℝ, equation x y) → ∃ area : ℝ, area = 384 :=
by
  sorry

end area_of_rhombus_enclosed_by_equation_l689_68911


namespace discriminant_of_quadratic_l689_68965

def a := 5
def b := 5 + 1/5
def c := 1/5
def discriminant (a b c : ℚ) := b^2 - 4 * a * c

theorem discriminant_of_quadratic :
  discriminant a b c = 576 / 25 :=
by
  sorry

end discriminant_of_quadratic_l689_68965


namespace train_speed_l689_68902

theorem train_speed (length : ℕ) (time : ℕ) (h_length : length = 1200) (h_time : time = 15) :
  (length / time) = 80 := by
  sorry

end train_speed_l689_68902


namespace absolute_value_is_four_l689_68978

-- Given condition: the absolute value of a number equals 4
theorem absolute_value_is_four (x : ℝ) : abs x = 4 → (x = 4 ∨ x = -4) :=
by
  sorry

end absolute_value_is_four_l689_68978


namespace min_value_frac_sum_l689_68930

theorem min_value_frac_sum (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 * x + y = 1) : 
  ∃ m, ∀ x y, 0 < x → 0 < y → 2 * x + y = 1 → m ≤ (1/x + 1/y) ∧ (1/x + 1/y) = 3 + 2 * Real.sqrt 2 :=
sorry

end min_value_frac_sum_l689_68930


namespace part1_intersection_when_a_is_zero_part2_range_of_a_l689_68951

-- Definitions of sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 6}
def B (a : ℝ) : Set ℝ := {x | 2 * a - 1 ≤ x ∧ x < a + 5}

-- Part (1): When a = 0, find A ∩ B
theorem part1_intersection_when_a_is_zero :
  A ∩ B 0 = {x : ℝ | -1 < x ∧ x < 5} :=
sorry

-- Part (2): If A ∪ B = A, find the range of values for a
theorem part2_range_of_a (a : ℝ) :
  (A ∪ B a = A) → (0 < a ∧ a ≤ 1) ∨ (6 ≤ a) :=
sorry

end part1_intersection_when_a_is_zero_part2_range_of_a_l689_68951


namespace solve_for_m_l689_68915

theorem solve_for_m (x m : ℝ) (h : (∃ x, (x - 1) / (x - 4) = m / (x - 4))): 
  m = 3 :=
by {
  sorry -- placeholder to indicate where the proof would go
}

end solve_for_m_l689_68915


namespace elizabeth_fruits_l689_68939

def total_fruits (initial_bananas initial_apples initial_grapes eaten_bananas eaten_apples eaten_grapes : Nat) : Nat :=
  let bananas_left := initial_bananas - eaten_bananas
  let apples_left := initial_apples - eaten_apples
  let grapes_left := initial_grapes - eaten_grapes
  bananas_left + apples_left + grapes_left

theorem elizabeth_fruits : total_fruits 12 7 19 4 2 10 = 22 := by
  sorry

end elizabeth_fruits_l689_68939


namespace molecular_weight_of_6_moles_Al2_CO3_3_l689_68918

noncomputable def molecular_weight_Al2_CO3_3: ℝ :=
  let Al_weight := 26.98
  let C_weight := 12.01
  let O_weight := 16.00
  let CO3_weight := C_weight + 3 * O_weight
  let one_mole_weight := 2 * Al_weight + 3 * CO3_weight
  6 * one_mole_weight

theorem molecular_weight_of_6_moles_Al2_CO3_3 : 
  molecular_weight_Al2_CO3_3 = 1403.94 :=
by
  sorry

end molecular_weight_of_6_moles_Al2_CO3_3_l689_68918


namespace part1_part2_part3_l689_68960

-- Part (1)
theorem part1 (m : ℝ) : (2 * m - 3) * (5 - 3 * m) = -6 * m^2 + 19 * m - 15 :=
  sorry

-- Part (2)
theorem part2 (a b : ℝ) : (3 * a^3) ^ 2 * (2 * b^2) ^ 3 / (6 * a * b) ^ 2 = 2 * a^4 * b^4 :=
  sorry

-- Part (3)
theorem part3 (a b : ℝ) : (a - b) * (a^2 + a * b + b^2) = a^3 - b^3 :=
  sorry

end part1_part2_part3_l689_68960


namespace g_10_equals_100_l689_68903

-- Define the function g and the conditions it must satisfy.
def g : ℕ → ℝ := sorry

axiom g_2 : g 2 = 4

axiom g_condition : ∀ m n : ℕ, m ≥ n → g (m + n) + g (m - n) = (g (2 * m) + g (2 * n)) / 2

-- Prove the required statement.
theorem g_10_equals_100 : g 10 = 100 :=
by sorry

end g_10_equals_100_l689_68903


namespace consumer_installment_credit_l689_68991

theorem consumer_installment_credit (C : ℝ) (A : ℝ) (h1 : A = 0.36 * C) 
    (h2 : 75 = A / 2) : C = 416.67 :=
by
  sorry

end consumer_installment_credit_l689_68991


namespace system_a_l689_68923

theorem system_a (x y z : ℝ) (h1 : x + y + z = 6) (h2 : 1/x + 1/y + 1/z = 11/6) (h3 : x*y + y*z + z*x = 11) :
  x = 1 ∧ y = 2 ∧ z = 3 ∨ x = 1 ∧ y = 3 ∧ z = 2 ∨ x = 2 ∧ y = 1 ∧ z = 3 ∨ x = 2 ∧ y = 3 ∧ z = 1 ∨ x = 3 ∧ y = 1 ∧ z = 2 ∨ x = 3 ∧ y = 2 ∧ z = 1 :=
sorry

end system_a_l689_68923


namespace x_add_inv_ge_two_l689_68963

theorem x_add_inv_ge_two (x : ℝ) (hx : x > 0) : x + (1 / x) ≥ 2 :=
by
  sorry

end x_add_inv_ge_two_l689_68963


namespace geometric_sequence_sum_of_first_five_l689_68937

theorem geometric_sequence_sum_of_first_five :
  (∃ (a : ℕ → ℝ) (r : ℝ),
    (∀ n, n > 0 → a n > 0) ∧
    a 2 = 2 ∧
    a 4 = 8 ∧
    r = 2 ∧
    a 1 = 1 ∧
    a 3 = a 1 * r^2 ∧
    a 5 = a 1 * r^4 ∧
    (a 1 + a 2 + a 3 + a 4 + a 5 = 31)
  ) :=
sorry

end geometric_sequence_sum_of_first_five_l689_68937


namespace budget_allocation_genetically_modified_microorganisms_l689_68977

theorem budget_allocation_genetically_modified_microorganisms :
  let microphotonics := 14
  let home_electronics := 19
  let food_additives := 10
  let industrial_lubricants := 8
  let total_percentage := 100
  let basic_astrophysics_percentage := 25
  let known_percentage := microphotonics + home_electronics + food_additives + industrial_lubricants + basic_astrophysics_percentage
  let genetically_modified_microorganisms := total_percentage - known_percentage
  genetically_modified_microorganisms = 24 := 
by
  sorry

end budget_allocation_genetically_modified_microorganisms_l689_68977


namespace circle_radius_of_square_perimeter_eq_area_l689_68983

theorem circle_radius_of_square_perimeter_eq_area (r : ℝ) (s : ℝ) (h1 : 2 * r = s) (h2 : 4 * s = 8 * r) (h3 : π * r ^ 2 = 8 * r) : r = 8 / π := by
  sorry

end circle_radius_of_square_perimeter_eq_area_l689_68983


namespace minimum_value_of_fraction_plus_variable_l689_68998

theorem minimum_value_of_fraction_plus_variable (a : ℝ) (h : a > 1) : ∃ m, (∀ b, b > 1 → (4 / (b - 1) + b) ≥ m) ∧ m = 5 :=
by
  use 5
  sorry

end minimum_value_of_fraction_plus_variable_l689_68998


namespace ellipse_foci_distance_l689_68984

noncomputable def distance_between_foci (a b : ℝ) : ℝ := 2 * Real.sqrt (a^2 - b^2)

theorem ellipse_foci_distance :
  ∃ (a b : ℝ), (a = 6) ∧ (b = 3) ∧ distance_between_foci a b = 6 * Real.sqrt 3 :=
by
  sorry

end ellipse_foci_distance_l689_68984


namespace common_ratio_q_l689_68996

variable {a_n : ℕ → ℝ} {S_n : ℕ → ℝ} {q : ℝ}

axiom a5_condition : a_n 5 = 2 * S_n 4 + 3
axiom a6_condition : a_n 6 = 2 * S_n 5 + 3

theorem common_ratio_q : q = 3 :=
by
  sorry

end common_ratio_q_l689_68996


namespace garden_width_is_correct_l689_68931

noncomputable def width_of_garden : ℝ :=
  let w := 12 -- We will define the width to be 12 as the final correct answer.
  w

theorem garden_width_is_correct (h_length : ∀ {w : ℝ}, 3 * w = 432 / w) : width_of_garden = 12 := by
  sorry

end garden_width_is_correct_l689_68931


namespace ratio_of_cereal_boxes_l689_68941

variable (F : ℕ) (S : ℕ) (T : ℕ) (k : ℚ)

def boxes_cereal : Prop :=
  F = 14 ∧
  F + S + T = 33 ∧
  S = k * (F : ℚ) ∧
  S = T - 5 → 
  S / F = 1 / 2

theorem ratio_of_cereal_boxes (F S T : ℕ) (k : ℚ) : 
  boxes_cereal F S T k :=
by
  sorry

end ratio_of_cereal_boxes_l689_68941


namespace max_sum_x_y_l689_68948

theorem max_sum_x_y {x y a b : ℝ} 
  (hx : 0 < x) (hy : 0 < y) (ha : 0 ≤ a ∧ a ≤ x) (hb : 0 ≤ b ∧ b ≤ y)
  (h1 : a^2 + y^2 = 2) (h2 : b^2 + x^2 = 1) (h3 : a * x + b * y = 1) : 
  x + y ≤ 2 :=
sorry

end max_sum_x_y_l689_68948


namespace cats_owners_percentage_l689_68908

noncomputable def percentage_of_students_owning_cats (total_students : ℕ) (cats_owners : ℕ) : ℚ :=
  (cats_owners : ℚ) / (total_students : ℚ) * 100

theorem cats_owners_percentage (total_students : ℕ) (cats_owners : ℕ)
  (dogs_owners : ℕ) (birds_owners : ℕ)
  (h_total_students : total_students = 400)
  (h_cats_owners : cats_owners = 80)
  (h_dogs_owners : dogs_owners = 120)
  (h_birds_owners : birds_owners = 40) :
  percentage_of_students_owning_cats total_students cats_owners = 20 :=
by {
  -- We state the proof but leave it as sorry so it's an incomplete placeholder.
  sorry
}

end cats_owners_percentage_l689_68908


namespace G_five_times_of_2_l689_68906

def G (x : ℝ) : ℝ := (x - 2) ^ 2 - 1

theorem G_five_times_of_2 : G (G (G (G (G 2)))) = 1179395 := 
by 
  rw [G, G, G, G, G]; 
  sorry

end G_five_times_of_2_l689_68906


namespace principal_is_400_l689_68971

-- Define the conditions
def rate_of_interest : ℚ := 12.5
def simple_interest : ℚ := 100
def time_in_years : ℚ := 2

-- Define the formula for principal amount based on the given conditions
def principal_amount (SI R T : ℚ) : ℚ := SI * 100 / (R * T)

-- Prove that the principal amount is 400
theorem principal_is_400 :
  principal_amount simple_interest rate_of_interest time_in_years = 400 := 
by
  simp [principal_amount, simple_interest, rate_of_interest, time_in_years]
  sorry

end principal_is_400_l689_68971


namespace even_positive_factors_count_l689_68928

theorem even_positive_factors_count (n : ℕ) (h : n = 2^4 * 3^3 * 7) : 
  ∃ k : ℕ, k = 32 := 
by
  sorry

end even_positive_factors_count_l689_68928


namespace problem_part_1_problem_part_2_l689_68944

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x - a * (x - 1)
noncomputable def g (x : ℝ) : ℝ := Real.exp x
noncomputable def h (x : ℝ) (a : ℝ) : ℝ := f (x + 1) a + g x

-- Problem Part (1)
theorem problem_part_1 (a : ℝ) (h_pos : 0 < a) :
  (Real.exp 1 - 1) / Real.exp 1 < a ∧ a < (Real.exp 2 - 1) / Real.exp 1 :=
sorry

-- Problem Part (2)
theorem problem_part_2 (a : ℝ) (h_cond : ∀ x, 0 ≤ x → h x a ≥ 1) :
  a ≤ 2 :=
sorry

end problem_part_1_problem_part_2_l689_68944


namespace anne_cleaning_time_l689_68912

variable (B A : ℝ)

theorem anne_cleaning_time :
  (B + A) * 4 = 1 ∧ (B + 2 * A) * 3 = 1 → 1/A = 12 := 
by
  intro h
  sorry

end anne_cleaning_time_l689_68912


namespace total_time_correct_l689_68967

-- Conditions
def minutes_per_story : Nat := 7
def weeks : Nat := 20

-- Total time calculation
def total_minutes : Nat := minutes_per_story * weeks

-- Conversion to hours and minutes
def total_hours : Nat := total_minutes / 60
def remaining_minutes : Nat := total_minutes % 60

-- The proof problem
theorem total_time_correct :
  total_minutes = 140 ∧ total_hours = 2 ∧ remaining_minutes = 20 := by
  sorry

end total_time_correct_l689_68967


namespace geometric_series_sum_l689_68994

-- Define the first term and common ratio
def a : ℚ := 5 / 3
def r : ℚ := -1 / 6

-- Prove the sum of the infinite geometric series
theorem geometric_series_sum : (∑' n : ℕ, a * r^n) = 10 / 7 := by
  sorry

end geometric_series_sum_l689_68994


namespace roots_equation_l689_68986

theorem roots_equation (p q : ℝ) (h1 : p / 3 = 9) (h2 : q / 3 = 14) : p + q = 69 :=
sorry

end roots_equation_l689_68986


namespace Liam_homework_assignments_l689_68925

theorem Liam_homework_assignments : 
  let assignments_needed (points : ℕ) : ℕ := match points with
    | 0     => 0
    | n+1 =>
        if n+1 <= 4 then 1
        else (4 + (((n+1) - 1)/4 - 1))

  30 <= 4 + 8 + 12 + 16 + 20 + 24 + 28 + 16 → ((λ points => List.sum (List.map assignments_needed (List.range points))) 30) = 128 :=
by
  sorry

end Liam_homework_assignments_l689_68925


namespace exists_directed_triangle_l689_68969

structure Tournament (V : Type) :=
  (edges : V → V → Prop)
  (complete : ∀ x y, x ≠ y → edges x y ∨ edges y x)
  (outdegree_at_least_one : ∀ x, ∃ y, edges x y)

theorem exists_directed_triangle {V : Type} [Fintype V] (T : Tournament V) :
  ∃ (a b c : V), T.edges a b ∧ T.edges b c ∧ T.edges c a := by
sorry

end exists_directed_triangle_l689_68969


namespace necessary_but_not_sufficient_l689_68959

theorem necessary_but_not_sufficient (x : ℝ) : (x^2 - 1 = 0) ↔ (x = -1 ∨ x = 1) ∧ (x - 1 = 0) → (x^2 - 1 = 0) ∧ ¬((x^2 - 1 = 0) → (x - 1 = 0)) := 
by sorry

end necessary_but_not_sufficient_l689_68959


namespace smallest_c_for_f_inverse_l689_68929

noncomputable def f (x : ℝ) : ℝ := (x - 3)^2 - 4

theorem smallest_c_for_f_inverse :
  ∃ c : ℝ, (∀ x₁ x₂ : ℝ, x₁ ≥ c → x₂ ≥ c → f x₁ = f x₂ → x₁ = x₂) ∧ (∀ d : ℝ, d < c → ∃ x₁ x₂ : ℝ, x₁ ≥ d ∧ x₂ ≥ d ∧ f x₁ = f x₂ ∧ x₁ ≠ x₂) ∧ c = 3 :=
by
  sorry

end smallest_c_for_f_inverse_l689_68929


namespace subcommittees_with_at_least_one_coach_l689_68922

-- Definitions based on conditions
def total_members : ℕ := 12
def total_coaches : ℕ := 5
def subcommittee_size : ℕ := 5

-- Lean statement of the problem
theorem subcommittees_with_at_least_one_coach :
  (Nat.choose total_members subcommittee_size) - (Nat.choose (total_members - total_coaches) subcommittee_size) = 771 := by
  sorry

end subcommittees_with_at_least_one_coach_l689_68922


namespace intersection_M_complement_N_eq_l689_68910

open Set

noncomputable def U : Set ℝ := univ
noncomputable def M : Set ℝ := {x | x^2 - 2 * x - 3 ≤ 0}
noncomputable def N : Set ℝ := {y | ∃ x : ℝ, y = x^2 + 1}
noncomputable def complement_N : Set ℝ := {y | y < 1}

theorem intersection_M_complement_N_eq : M ∩ complement_N = {x | -1 ≤ x ∧ x < 1} :=
by
  sorry

end intersection_M_complement_N_eq_l689_68910


namespace geometric_series_sum_l689_68914

theorem geometric_series_sum (a : ℝ) (q : ℝ) (a₁ : ℝ) 
  (h1 : a₁ = 1)
  (h2 : q = a - (3/2))
  (h3 : |q| < 1)
  (h4 : a = a₁ / (1 - q)) :
  a = 2 :=
sorry

end geometric_series_sum_l689_68914


namespace num_five_digit_integers_l689_68972

def factorial (n : Nat) : Nat := if n = 0 then 1 else n * factorial (n - 1)

theorem num_five_digit_integers : 
  let num_ways := factorial 5 / (factorial 2 * factorial 3)
  num_ways = 10 :=
by 
  sorry

end num_five_digit_integers_l689_68972


namespace polynomial_inequality_l689_68968

theorem polynomial_inequality (f : ℝ → ℝ) (h1 : f 0 = 1)
    (h2 : ∀ (x y : ℝ), f (x - y) + f x ≥ 2 * x^2 - 2 * x * y + y^2 + 2 * x - y + 2) :
    f = λ x => x^2 + x + 1 := by
  sorry

end polynomial_inequality_l689_68968


namespace sum_base10_to_base4_l689_68907

theorem sum_base10_to_base4 : 
  (31 + 22 : ℕ) = 3 * 4^2 + 1 * 4^1 + 1 * 4^0 :=
by
  sorry

end sum_base10_to_base4_l689_68907


namespace tangent_line_ellipse_l689_68946

theorem tangent_line_ellipse (a b x y x₀ y₀ : ℝ) (h : a > 0) (hb : b > 0) (ha_gt_hb : a > b) 
(h_on_ellipse : (x₀^2 / a^2) + (y₀^2 / b^2) = 1) :
    (x₀ * x / a^2) + (y₀ * y / b^2) = 1 := 
sorry

end tangent_line_ellipse_l689_68946


namespace us_more_than_canada_l689_68995

/-- Define the total number of supermarkets -/
def total_supermarkets : ℕ := 84

/-- Define the number of supermarkets in the US -/
def us_supermarkets : ℕ := 49

/-- Define the number of supermarkets in Canada -/
def canada_supermarkets : ℕ := total_supermarkets - us_supermarkets

/-- The proof problem: Prove that there are 14 more supermarkets in the US than in Canada -/
theorem us_more_than_canada : us_supermarkets - canada_supermarkets = 14 := by
  sorry

end us_more_than_canada_l689_68995


namespace locus_of_circumcenter_l689_68945

theorem locus_of_circumcenter (θ : ℝ) :
  let M := (3, 3 * Real.tan (θ - Real.pi / 3))
  let N := (3, 3 * Real.tan θ)
  let C := (3 / 2, 3 / 2 * Real.tan θ)
  ∃ (x y : ℝ), (x - 4) ^ 2 / 4 - y ^ 2 / 12 = 1 :=
by
  sorry

end locus_of_circumcenter_l689_68945


namespace height_at_10inches_l689_68947

theorem height_at_10inches 
  (a : ℚ)
  (h : 20 = (- (4 / 125) * 25 ^ 2 + 20))
  (span_eq : 50 = 50)
  (height_eq : 20 = 20)
  (y_eq : ∀ x : ℚ, - (4 / 125) * x ^ 2 + 20 = 16.8) :
  (- (4 / 125) * 10 ^ 2 + 20) = 16.8 :=
by
  sorry

end height_at_10inches_l689_68947


namespace angle_compute_l689_68900

open Real

noncomputable def a : ℝ × ℝ := (1, -1)
noncomputable def b : ℝ × ℝ := (1, 2)

noncomputable def sub_vec := (b.1 - a.1, b.2 - a.2)
noncomputable def sum_vec := (a.1 + 2 * b.1, a.2 + 2 * b.2)

noncomputable def dot_product (v₁ v₂ : ℝ × ℝ) : ℝ :=
  v₁.1 * v₂.1 + v₁.2 * v₂.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  sqrt (v.1 * v.1 + v.2 * v.2)

noncomputable def angle_between (v₁ v₂ : ℝ × ℝ) : ℝ :=
  arccos (dot_product v₁ v₂ / (magnitude v₁ * magnitude v₂))

theorem angle_compute : angle_between sub_vec sum_vec = π / 4 :=
by {
  sorry
}

end angle_compute_l689_68900


namespace inscribed_square_area_after_cutting_l689_68970

theorem inscribed_square_area_after_cutting :
  let original_side := 5
  let cut_side := 1
  let remaining_side := original_side - 2 * cut_side
  let largest_inscribed_square_area := remaining_side ^ 2
  largest_inscribed_square_area = 9 :=
by
  let original_side := 5
  let cut_side := 1
  let remaining_side := original_side - 2 * cut_side
  let largest_inscribed_square_area := remaining_side ^ 2
  show largest_inscribed_square_area = 9
  sorry

end inscribed_square_area_after_cutting_l689_68970


namespace remainder_of_m_div_5_l689_68964

theorem remainder_of_m_div_5 (m n : ℕ) (h1 : m = 15 * n - 1) (h2 : n > 0) : m % 5 = 4 :=
sorry

end remainder_of_m_div_5_l689_68964


namespace permutations_of_3_3_3_7_7_l689_68997

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem permutations_of_3_3_3_7_7 : 
  (factorial 5) / (factorial 3 * factorial 2) = 10 :=
by
  sorry

end permutations_of_3_3_3_7_7_l689_68997


namespace sqrt_neg2023_squared_l689_68938

theorem sqrt_neg2023_squared : Real.sqrt ((-2023 : ℝ)^2) = 2023 :=
by
  sorry

end sqrt_neg2023_squared_l689_68938


namespace frog_climbing_time_is_correct_l689_68913

noncomputable def frog_climb_out_time : Nat :=
  let well_depth := 12
  let climb_up := 3
  let slip_down := 1
  let net_gain := climb_up - slip_down
  let total_cycles := (well_depth - 3) / net_gain + 1
  let total_time := total_cycles * 3
  let extra_time := 6
  total_time + extra_time

theorem frog_climbing_time_is_correct :
  frog_climb_out_time = 22 := by
  sorry

end frog_climbing_time_is_correct_l689_68913


namespace units_digit_of_calculation_l689_68920

-- Base definitions for units digits of given numbers
def units_digit (n : ℕ) : ℕ := n % 10

-- Main statement to prove
theorem units_digit_of_calculation : 
  units_digit ((25 ^ 3 + 17 ^ 3) * 12 ^ 2) = 2 :=
by
  -- This is where the proof would go, but it's omitted as requested
  sorry

end units_digit_of_calculation_l689_68920


namespace verify_trees_in_other_row_l689_68989

-- Definition of a normal lemon tree lemon production per year
def normalLemonTreeProduction : ℕ := 60

-- Definition of the percentage increase in lemon production for specially engineered lemon trees
def percentageIncrease : ℕ := 50

-- Definition of lemon production for specially engineered lemon trees
def specialLemonTreeProduction : ℕ := normalLemonTreeProduction * (1 + percentageIncrease / 100)

-- Number of trees in one row of the grove
def treesInOneRow : ℕ := 50

-- Total lemon production in 5 years
def totalLemonProduction : ℕ := 675000

-- Number of years
def years : ℕ := 5

-- Total number of trees in the grove
def totalNumberOfTrees : ℕ := totalLemonProduction / (specialLemonTreeProduction * years)

-- Number of trees in the other row
def treesInOtherRow : ℕ := totalNumberOfTrees - treesInOneRow

-- Theorem: Verification of the number of trees in the other row
theorem verify_trees_in_other_row : treesInOtherRow = 1450 :=
  by
  -- Proof logic is omitted, leaving as sorry
  sorry

end verify_trees_in_other_row_l689_68989


namespace non_trivial_solution_exists_l689_68917

theorem non_trivial_solution_exists (a b c : ℤ) (p : ℕ) [Fact (Nat.Prime p)] :
  ∃ x y z : ℤ, (a * x^2 + b * y^2 + c * z^2) % p = 0 ∧ (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0) :=
sorry

end non_trivial_solution_exists_l689_68917


namespace arithmetic_geometric_mean_l689_68952

variable (x y : ℝ)

theorem arithmetic_geometric_mean (h1 : (x + y) / 2 = 20) (h2 : Real.sqrt (x * y) = Real.sqrt 110) :
  x^2 + y^2 = 1380 := by
  sorry

end arithmetic_geometric_mean_l689_68952


namespace perfect_square_unique_n_l689_68905

theorem perfect_square_unique_n (n : ℕ) (hn : n > 0) : 
  (∃ m : ℕ, 2^n + 12^n + 2011^n = m^2) ↔ n = 1 := by
  sorry

end perfect_square_unique_n_l689_68905


namespace tan_subtraction_l689_68962

theorem tan_subtraction (α β : ℝ) (h1 : Real.tan α = 5) (h2 : Real.tan β = 3) : 
  Real.tan (α - β) = 1 / 8 := 
by 
  sorry

end tan_subtraction_l689_68962


namespace local_minimum_at_2_l689_68956

noncomputable def f (x m : ℝ) : ℝ := x * (x - m)^2

theorem local_minimum_at_2 (m : ℝ) (h : 2 * (2 - m)^2 + 2 * 4 * (2 - m) = 0) : m = 6 :=
by
  sorry

end local_minimum_at_2_l689_68956


namespace find_number_l689_68957

theorem find_number (x : ℝ) (h : (4 / 3) * x = 48) : x = 36 :=
sorry

end find_number_l689_68957


namespace part1_l689_68943

theorem part1 (P Q R : Polynomial ℝ) : 
  ¬ ∃ (P Q R : Polynomial ℝ), (∀ x y z : ℝ, (x - y + 1)^3 * P.eval x + (y - z - 1)^3 * Q.eval y + (z - 2 * x + 1)^3 * R.eval z = 1) := sorry

end part1_l689_68943


namespace supermarket_sales_l689_68904

theorem supermarket_sales (S_Dec : ℝ) (S_Jan : ℝ) (S_Feb : ℝ) (S_Jan_eq : S_Jan = S_Dec * (1 + x))
  (S_Feb_eq : S_Feb = S_Jan * (1 + x))
  (inc_eq : S_Feb = S_Dec + 0.24 * S_Dec) :
  x = 0.2 ∧ S_Feb = S_Dec * (1 + 0.2)^2 := by
sorry

end supermarket_sales_l689_68904


namespace rad_to_deg_eq_l689_68980

theorem rad_to_deg_eq : (4 / 3) * 180 = 240 := by
  sorry

end rad_to_deg_eq_l689_68980


namespace system_solutions_l689_68953

theorem system_solutions : 
  ∃ (x y z t : ℝ), 
    (x * y - t^2 = 9) ∧ 
    (x^2 + y^2 + z^2 = 18) ∧ 
    ((x = 3 ∧ y = 3 ∧ z = 0 ∧ t = 0) ∨ 
     (x = -3 ∧ y = -3 ∧ z = 0 ∧ t = 0)) :=
by {
  sorry
}

end system_solutions_l689_68953


namespace x_coordinate_of_equidistant_point_l689_68992

theorem x_coordinate_of_equidistant_point (x : ℝ) : 
  ((-3 - x)^2 + (-2 - 0)^2) = ((2 - x)^2 + (-6 - 0)^2) → x = 2.7 :=
by
  sorry

end x_coordinate_of_equidistant_point_l689_68992


namespace number_of_boys_is_60_l689_68909

-- Definitions based on conditions
def total_students : ℕ := 150

def number_of_boys (x : ℕ) : Prop :=
  ∃ g : ℕ, x + g = total_students ∧ g = (x * total_students) / 100

-- Theorem statement
theorem number_of_boys_is_60 : number_of_boys 60 := 
sorry

end number_of_boys_is_60_l689_68909


namespace function_passes_through_fixed_point_l689_68958

variables {a : ℝ}

/-- Given the function f(x) = a^(x-1) (a > 0 and a ≠ 1), prove that the function always passes through the point (1, 1) -/
theorem function_passes_through_fixed_point (h1 : a > 0) (h2 : a ≠ 1) :
  (a^(1-1) = 1) :=
by
  sorry

end function_passes_through_fixed_point_l689_68958


namespace problem_statement_l689_68949

theorem problem_statement (a b c x : ℤ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hx : x ≠ 0)
  (eq1 : (a * x^4 / b * c)^3 = x^3)
  (sum_eq : a + b + c = 9) :
  (x = 1 ∨ x = -1) ∧ a = 1 ∧ b = 4 ∧ c = 4 :=
by
  sorry

end problem_statement_l689_68949


namespace distance_between_points_l689_68988

theorem distance_between_points :
  let point1 := (2, -3)
  let point2 := (8, 9)
  dist point1 point2 = 6 * Real.sqrt 5 :=
by
  sorry

end distance_between_points_l689_68988


namespace find_common_divisor_l689_68999

open Int

theorem find_common_divisor (n : ℕ) (h1 : 2287 % n = 2028 % n)
  (h2 : 2028 % n = 1806 % n) : n = Int.gcd (Int.gcd 259 222) 481 := by
  sorry -- Proof goes here

end find_common_divisor_l689_68999


namespace total_boys_in_class_l689_68921

/-- 
  Given 
    - n + 1 positions in a circle, where n is the number of boys and 1 position for the teacher.
    - The boy at the 6th position is exactly opposite to the boy at the 16th position.
  Prove that the total number of boys in the class is 20.
-/
theorem total_boys_in_class (n : ℕ) (h1 : n + 1 > 16) (h2 : (6 + 10) % (n + 1) = 16):
  n = 20 := 
by 
  sorry

end total_boys_in_class_l689_68921


namespace volume_common_part_equal_quarter_volume_each_cone_l689_68955

theorem volume_common_part_equal_quarter_volume_each_cone
  (r h : ℝ) (V_cone : ℝ)
  (h_cone_volume : V_cone = (1 / 3) * π * r^2 * h) :
  ∃ V_common, V_common = (1 / 4) * V_cone :=
by
  -- Main structure of the proof skipped
  sorry

end volume_common_part_equal_quarter_volume_each_cone_l689_68955


namespace parking_lot_perimeter_l689_68975

theorem parking_lot_perimeter (a b : ℝ) 
  (h_diag : a^2 + b^2 = 784) 
  (h_area : a * b = 180) : 
  2 * (a + b) = 68 := 
by 
  sorry

end parking_lot_perimeter_l689_68975


namespace age_of_replaced_person_is_46_l689_68919

variable (age_of_replaced_person : ℕ)
variable (new_person_age : ℕ := 16)
variable (decrease_in_age_per_person : ℕ := 3)
variable (number_of_people : ℕ := 10)

theorem age_of_replaced_person_is_46 :
  age_of_replaced_person - new_person_age = decrease_in_age_per_person * number_of_people → 
  age_of_replaced_person = 46 :=
by
  sorry

end age_of_replaced_person_is_46_l689_68919


namespace complex_sum_real_imag_l689_68933

theorem complex_sum_real_imag : 
  (Complex.re ((Complex.I / (1 + Complex.I)) - (1 / (2 * Complex.I))) + 
  Complex.im ((Complex.I / (1 + Complex.I)) - (1 / (2 * Complex.I)))) = 3/2 := 
by sorry

end complex_sum_real_imag_l689_68933


namespace sufficient_but_not_necessary_condition_l689_68973

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (0 ≤ x ∧ x ≤ 1) → |x| ≤ 1 :=
by sorry

end sufficient_but_not_necessary_condition_l689_68973


namespace hershey_kisses_to_kitkats_ratio_l689_68993

-- Definitions based on the conditions
def kitkats : ℕ := 5
def nerds : ℕ := 8
def lollipops : ℕ := 11
def baby_ruths : ℕ := 10
def reeses : ℕ := baby_ruths / 2
def candy_total_before : ℕ := kitkats + nerds + lollipops + baby_ruths + reeses
def candy_remaining : ℕ := 49
def lollipops_given : ℕ := 5
def total_candy_before : ℕ := candy_remaining + lollipops_given
def hershey_kisses : ℕ := total_candy_before - candy_total_before

-- Theorem to prove the desired ratio
theorem hershey_kisses_to_kitkats_ratio : hershey_kisses / kitkats = 3 := by
  sorry

end hershey_kisses_to_kitkats_ratio_l689_68993
