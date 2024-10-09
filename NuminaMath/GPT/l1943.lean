import Mathlib

namespace kylie_coins_left_l1943_194370

-- Definitions for each condition
def coins_from_piggy_bank : ℕ := 15
def coins_from_brother : ℕ := 13
def coins_from_father : ℕ := 8
def coins_given_to_friend : ℕ := 21

-- The total coins Kylie has initially
def initial_coins : ℕ := coins_from_piggy_bank + coins_from_brother
def total_coins_after_father : ℕ := initial_coins + coins_from_father
def coins_left : ℕ := total_coins_after_father - coins_given_to_friend

-- The theorem to prove the final number of coins left is 15
theorem kylie_coins_left : coins_left = 15 :=
by
  sorry -- Proof goes here

end kylie_coins_left_l1943_194370


namespace M_inter_N_l1943_194379

def M : Set ℝ := { y | y > 1 }
def N : Set ℝ := { x | 0 < x ∧ x < 2 }

theorem M_inter_N : M ∩ N = { z | 1 < z ∧ z < 2 } :=
by 
  sorry

end M_inter_N_l1943_194379


namespace remainder_division_l1943_194352

variable (P D K Q R R'_q R'_r : ℕ)

theorem remainder_division (h1 : P = Q * D + R) (h2 : R = R'_q * K + R'_r) (h3 : K < D) : 
  P % (D * K) = R'_r :=
sorry

end remainder_division_l1943_194352


namespace Vanya_original_number_l1943_194395

theorem Vanya_original_number (m n : ℕ) (hm : m ≤ 9) (hn : n ≤ 9) (h : (10 * n + m)^2 = 4 * (10 * m + n)) : 10 * m + n = 81 := by
  sorry

end Vanya_original_number_l1943_194395


namespace girls_more_than_boys_l1943_194318

theorem girls_more_than_boys : ∃ (b g x : ℕ), b = 3 * x ∧ g = 4 * x ∧ b + g = 35 ∧ g - b = 5 :=
by  -- We just define the theorem, no need for a proof, added "by sorry"
  sorry

end girls_more_than_boys_l1943_194318


namespace avg_annual_growth_rate_optimal_room_price_l1943_194349

-- Problem 1: Average Annual Growth Rate
theorem avg_annual_growth_rate (visitors_2021 visitors_2023 : ℝ) (years : ℕ) (visitors_2021_pos : 0 < visitors_2021) :
  visitors_2023 > visitors_2021 → visitors_2023 / visitors_2021 = 2.25 → 
  ∃ x : ℝ, (1 + x)^2 = 2.25 ∧ x = 0.5 :=
by sorry

-- Problem 2: Optimal Room Price for Desired Profit
theorem optimal_room_price (rooms : ℕ) (base_price cost_per_room desired_profit : ℝ)
  (rooms_pos : 0 < rooms) :
  base_price = 180 → cost_per_room = 20 → desired_profit = 9450 → 
  ∃ y : ℝ, (y - cost_per_room) * (rooms - (y - base_price) / 10) = desired_profit ∧ y = 230 :=
by sorry

end avg_annual_growth_rate_optimal_room_price_l1943_194349


namespace divisor_in_second_division_is_19_l1943_194330

theorem divisor_in_second_division_is_19 (n d : ℕ) (h1 : n % 25 = 4) (h2 : (n + 15) % d = 4) : d = 19 :=
sorry

end divisor_in_second_division_is_19_l1943_194330


namespace cost_price_percentage_l1943_194394

theorem cost_price_percentage (MP CP : ℝ) (h_discount : 0.75 * MP = CP * 1.171875) :
  ((CP / MP) * 100) = 64 :=
by
  sorry

end cost_price_percentage_l1943_194394


namespace find_m_when_z_is_real_l1943_194385

theorem find_m_when_z_is_real (m : ℝ) (h : (m ^ 2 + 2 * m - 15 = 0)) : m = 3 :=
sorry

end find_m_when_z_is_real_l1943_194385


namespace range_of_x_when_m_eq_4_range_of_m_given_conditions_l1943_194312

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

end range_of_x_when_m_eq_4_range_of_m_given_conditions_l1943_194312


namespace max_cross_section_area_l1943_194319

noncomputable def prism_cross_section_area : ℝ :=
  let z_axis_parallel := true
  let square_base := 8
  let plane := ∀ x y z, 3 * x - 5 * y + 2 * z = 20
  121.6

theorem max_cross_section_area :
  prism_cross_section_area = 121.6 :=
sorry

end max_cross_section_area_l1943_194319


namespace bert_total_stamps_l1943_194337

theorem bert_total_stamps (bought_stamps : ℕ) (half_stamps_before : ℕ) (total_stamps_after : ℕ) :
  (bought_stamps = 300) ∧ (half_stamps_before = bought_stamps / 2) → (total_stamps_after = half_stamps_before + bought_stamps) → (total_stamps_after = 450) :=
by
  sorry

end bert_total_stamps_l1943_194337


namespace validColoringsCount_l1943_194335

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

end validColoringsCount_l1943_194335


namespace prove_m_equals_9_given_split_l1943_194381

theorem prove_m_equals_9_given_split (m : ℕ) (h : 1 < m) (h1 : m^3 = 73) : m = 9 :=
sorry

end prove_m_equals_9_given_split_l1943_194381


namespace probability_two_or_more_women_l1943_194347

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

end probability_two_or_more_women_l1943_194347


namespace unique_divisor_of_2_pow_n_minus_1_l1943_194355

theorem unique_divisor_of_2_pow_n_minus_1 : ∀ (n : ℕ), n ≥ 1 → n ∣ (2^n - 1) → n = 1 := 
by
  intro n h1 h2
  sorry

end unique_divisor_of_2_pow_n_minus_1_l1943_194355


namespace find_max_marks_l1943_194325

variable (M : ℕ) (P : ℕ)

theorem find_max_marks (h1 : M = 332) (h2 : P = 83) : 
  let Max_Marks := M / (P / 100)
  Max_Marks = 400 := 
by 
  sorry

end find_max_marks_l1943_194325


namespace hcf_of_two_numbers_l1943_194354

noncomputable def find_hcf (x y : ℕ) (lcm_xy : ℕ) (prod_xy : ℕ) : ℕ :=
  prod_xy / lcm_xy

theorem hcf_of_two_numbers (x y : ℕ) (lcm_xy: ℕ) (prod_xy: ℕ) 
  (h_lcm: lcm x y = lcm_xy) (h_prod: x * y = prod_xy) :
  find_hcf x y lcm_xy prod_xy = 75 :=
by
  sorry

end hcf_of_two_numbers_l1943_194354


namespace other_function_value_at_20_l1943_194393

def linear_function (k b : ℝ) (x : ℝ) : ℝ :=
  k * x + b

theorem other_function_value_at_20
    (k1 k2 b1 b2 : ℝ)
    (h_intersect : linear_function k1 b1 2 = linear_function k2 b2 2)
    (h_diff_at_8 : abs (linear_function k1 b1 8 - linear_function k2 b2 8) = 8)
    (h_y1_at_20 : linear_function k1 b1 20 = 100) :
  linear_function k2 b2 20 = 76 ∨ linear_function k2 b2 20 = 124 :=
sorry

end other_function_value_at_20_l1943_194393


namespace subcommittee_ways_l1943_194356

theorem subcommittee_ways :
  ∃ (n : ℕ), n = Nat.choose 10 4 * Nat.choose 7 2 ∧ n = 4410 :=
by
  use 4410
  sorry

end subcommittee_ways_l1943_194356


namespace binary_arithmetic_l1943_194343

-- Define the binary numbers 11010_2, 11100_2, and 100_2
def x : ℕ := 0b11010 -- base 2 number 11010 in base 10 representation
def y : ℕ := 0b11100 -- base 2 number 11100 in base 10 representation
def d : ℕ := 0b100   -- base 2 number 100 in base 10 representation

-- Define the correct answer
def correct_answer : ℕ := 0b10101101 -- base 2 number 10101101 in base 10 representation

-- The proof problem statement
theorem binary_arithmetic : (x * y) / d = correct_answer := by
  sorry

end binary_arithmetic_l1943_194343


namespace maximum_value_of_omega_l1943_194344

variable (A ω : ℝ)

theorem maximum_value_of_omega (hA : 0 < A) (hω_pos : 0 < ω)
  (h1 : ω * (-π / 2) ≥ -π / 2) 
  (h2 : ω * (2 * π / 3) ≤ π / 2) :
  ω = 3 / 4 :=
sorry

end maximum_value_of_omega_l1943_194344


namespace add_and_multiply_l1943_194359

def num1 : ℝ := 0.0034
def num2 : ℝ := 0.125
def num3 : ℝ := 0.00678
def sum := num1 + num2 + num3

theorem add_and_multiply :
  (sum * 2) = 0.27036 := by
  sorry

end add_and_multiply_l1943_194359


namespace red_marked_area_on_larger_sphere_l1943_194308

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

end red_marked_area_on_larger_sphere_l1943_194308


namespace new_length_maintains_area_l1943_194357

noncomputable def new_length_for_doubled_width (A W : ℝ) : ℝ := A / (2 * W)

theorem new_length_maintains_area (A W : ℝ) (hA : A = 35.7) (hW : W = 3.8) :
  new_length_for_doubled_width A W = 4.69736842 :=
by
  rw [new_length_for_doubled_width, hA, hW]
  norm_num
  sorry

end new_length_maintains_area_l1943_194357


namespace find_b_l1943_194377

theorem find_b 
  (b : ℝ)
  (h_pos : 0 < b)
  (h_geom_sequence : ∃ r : ℝ, 10 * r = b ∧ b * r = 2 / 3) :
  b = 2 * Real.sqrt 15 / 3 :=
by
  sorry

end find_b_l1943_194377


namespace find_x_six_l1943_194338

noncomputable def positive_real : Type := { x : ℝ // 0 < x }

theorem find_x_six (x : positive_real)
  (h : (1 - x.val ^ 3) ^ (1/3) + (1 + x.val ^ 3) ^ (1/3) = 1) :
  x.val ^ 6 = 28 / 27 := 
sorry

end find_x_six_l1943_194338


namespace trisha_spent_on_eggs_l1943_194372

def totalSpent (meat chicken veggies eggs dogFood amountLeft initialAmount : ℕ) : ℕ :=
  initialAmount - (meat + chicken + veggies + dogFood + amountLeft)

theorem trisha_spent_on_eggs :
  ∀ (meat chicken veggies eggs dogFood amountLeft initialAmount : ℕ),
    meat = 17 →
    chicken = 22 →
    veggies = 43 →
    dogFood = 45 →
    amountLeft = 35 →
    initialAmount = 167 →
    totalSpent meat chicken veggies eggs dogFood amountLeft initialAmount = 5 :=
by
  intros meat chicken veggies eggs dogFood amountLeft initialAmount
  sorry

end trisha_spent_on_eggs_l1943_194372


namespace negation_of_exists_l1943_194329

theorem negation_of_exists (x : ℝ) :
  ¬ (∃ x > 0, 2 * x + 3 ≤ 0) ↔ ∀ x > 0, 2 * x + 3 > 0 :=
by
  sorry

end negation_of_exists_l1943_194329


namespace smallest_possible_value_of_EF_minus_DE_l1943_194315

theorem smallest_possible_value_of_EF_minus_DE :
  ∃ (DE EF FD : ℤ), DE + EF + FD = 2010 ∧ DE < EF ∧ EF ≤ FD ∧ 1 = EF - DE ∧ DE > 0 ∧ EF > 0 ∧ FD > 0 ∧ 
  DE + EF > FD ∧ DE + FD > EF ∧ EF + FD > DE :=
by {
  sorry
}

end smallest_possible_value_of_EF_minus_DE_l1943_194315


namespace gcf_180_270_450_l1943_194320

theorem gcf_180_270_450 : Nat.gcd (Nat.gcd 180 270) 450 = 90 :=
by
  sorry

end gcf_180_270_450_l1943_194320


namespace quadratic_equation_standard_form_quadratic_equation_coefficients_l1943_194300

theorem quadratic_equation_standard_form : 
  ∀ (x : ℝ), (2 * x^2 - 1 = 6 * x) ↔ (2 * x^2 - 6 * x - 1 = 0) :=
by
  sorry

theorem quadratic_equation_coefficients : 
  ∃ (a b c : ℝ), (a = 2 ∧ b = -6 ∧ c = -1) :=
by
  sorry

end quadratic_equation_standard_form_quadratic_equation_coefficients_l1943_194300


namespace fraction_is_one_fourth_l1943_194362

theorem fraction_is_one_fourth (f N : ℝ) 
  (h1 : (1/3) * f * N = 15) 
  (h2 : (3/10) * N = 54) : 
  f = 1/4 :=
by
  sorry

end fraction_is_one_fourth_l1943_194362


namespace time_to_write_all_rearrangements_in_hours_l1943_194342

/-- Michael's name length is 7 (number of unique letters) -/
def name_length : Nat := 7

/-- Michael can write 10 rearrangements per minute -/
def write_rate : Nat := 10

/-- Number of rearrangements of Michael's name -/
def num_rearrangements : Nat := (name_length.factorial)

theorem time_to_write_all_rearrangements_in_hours :
  (num_rearrangements / write_rate : ℚ) / 60 = 8.4 := by
  sorry

end time_to_write_all_rearrangements_in_hours_l1943_194342


namespace largest_possible_a_l1943_194314

theorem largest_possible_a (a b c d : ℕ) (h1 : a < 3 * b) (h2 : b < 4 * c) (h3 : c < 5 * d) (h4 : d < 150) (hp : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) :
  a ≤ 8924 :=
sorry

end largest_possible_a_l1943_194314


namespace sixth_oak_placement_l1943_194392

structure Point :=
  (x : ℝ)
  (y : ℝ)

def is_aligned (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

noncomputable def intersection_point (p1 p2 p3 p4 : Point) : Point := 
  let m1 := (p2.y - p1.y) / (p2.x - p1.x)
  let m2 := (p4.y - p3.y) / (p4.x - p3.x)
  let c1 := p1.y - (m1 * p1.x)
  let c2 := p3.y - (m2 * p3.x)
  let x := (c2 - c1) / (m1 - m2)
  let y := m1 * x + c1
  ⟨x, y⟩

theorem sixth_oak_placement 
  (A1 A2 A3 B1 B2 B3 : Point) 
  (hA : ¬ is_aligned A1 A2 A3)
  (hB : ¬ is_aligned B1 B2 B3) :
  ∃ P : Point, (∃ (C1 C2 : Point), C1 = A1 ∧ C2 = B1 ∧ is_aligned C1 C2 P) ∧ 
               (∃ (C3 C4 : Point), C3 = A2 ∧ C4 = B2 ∧ is_aligned C3 C4 P) := by
  sorry

end sixth_oak_placement_l1943_194392


namespace find_a_exactly_two_solutions_l1943_194390

theorem find_a_exactly_two_solutions :
  (∀ x y : ℝ, |x - 6 - y| + |x - 6 + y| = 12 ∧ (|x| - 6)^2 + (|y| - 8)^2 = a) ↔ (a = 4 ∨ a = 100) :=
sorry

end find_a_exactly_two_solutions_l1943_194390


namespace tan_theta_sub_pi_over_4_l1943_194310

open Real

theorem tan_theta_sub_pi_over_4 (θ : ℝ) (h1 : -π / 2 < θ ∧ θ < 0) 
  (h2 : sin (θ + π / 4) = 3 / 5) : tan (θ - π / 4) = -4 / 3 :=
by
  sorry

end tan_theta_sub_pi_over_4_l1943_194310


namespace remainder_of_base12_2563_mod_17_l1943_194328

-- Define the base-12 number 2563 in decimal.
def base12_to_decimal : ℕ := 2 * 12^3 + 5 * 12^2 + 6 * 12^1 + 3 * 12^0

-- Define the number 17.
def divisor : ℕ := 17

-- Prove that the remainder when base12_to_decimal is divided by divisor is 1.
theorem remainder_of_base12_2563_mod_17 : base12_to_decimal % divisor = 1 :=
by
  sorry

end remainder_of_base12_2563_mod_17_l1943_194328


namespace a_in_range_l1943_194336

noncomputable def kOM (t : ℝ) : ℝ := (Real.log t) / t
noncomputable def kON (a t : ℝ) : ℝ := (a + a * t - t^2) / t

theorem a_in_range (a : ℝ) : 
  (∀ t ∈ Set.Ici 1, 0 ≤ (1 - Real.log t + a) / t^2 + 1) →
  a ∈ Set.Ici (-2) := 
by
  sorry

end a_in_range_l1943_194336


namespace line_through_intersection_points_l1943_194383

noncomputable def circle1 := { p : ℝ × ℝ | p.1^2 + p.2^2 = 10 }
noncomputable def circle2 := { p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 3)^2 = 10 }

theorem line_through_intersection_points (p : ℝ × ℝ) (hp1 : p ∈ circle1) (hp2 : p ∈ circle2) :
  p.1 + 3 * p.2 - 5 = 0 :=
sorry

end line_through_intersection_points_l1943_194383


namespace d_n_2_d_n_3_l1943_194324

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

end d_n_2_d_n_3_l1943_194324


namespace math_problem_solution_l1943_194301

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

end math_problem_solution_l1943_194301


namespace elder_person_age_l1943_194341

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

end elder_person_age_l1943_194341


namespace notebooks_difference_l1943_194360

theorem notebooks_difference :
  ∀ (Jac_left Jac_Paula Jac_Mike Ger_not Jac_init : ℕ),
  Ger_not = 8 →
  Jac_left = 10 →
  Jac_Paula = 5 →
  Jac_Mike = 6 →
  Jac_init = Jac_left + Jac_Paula + Jac_Mike →
  Jac_init - Ger_not = 13 := 
by
  intros Jac_left Jac_Paula Jac_Mike Ger_not Jac_init
  intros Ger_not_8 Jac_left_10 Jac_Paula_5 Jac_Mike_6 Jac_init_def
  sorry

end notebooks_difference_l1943_194360


namespace actual_distance_traveled_l1943_194361

theorem actual_distance_traveled
  (t : ℕ)
  (H1 : 6 * t = 3 * t + 15) :
  3 * t = 15 :=
by
  exact sorry

end actual_distance_traveled_l1943_194361


namespace butterfingers_count_l1943_194304

theorem butterfingers_count (total_candy_bars : ℕ) (snickers : ℕ) (mars_bars : ℕ) (h_total : total_candy_bars = 12) (h_snickers : snickers = 3) (h_mars : mars_bars = 2) : 
  ∃ (butterfingers : ℕ), butterfingers = 7 :=
by
  sorry

end butterfingers_count_l1943_194304


namespace seating_arrangement_7_people_l1943_194322

theorem seating_arrangement_7_people (n : Nat) (h1 : n = 7) :
  let m := n - 1
  (m.factorial / m) * 2 = 240 :=
by
  sorry

end seating_arrangement_7_people_l1943_194322


namespace sequence_general_term_l1943_194351

theorem sequence_general_term (a : ℕ → ℕ) :
  (a 1 = 1 * 2) ∧ (a 2 = 2 * 3) ∧ (a 3 = 3 * 4) ∧ (a 4 = 4 * 5) ↔ 
    (∀ n, a n = n^2 + n) := sorry

end sequence_general_term_l1943_194351


namespace fencers_count_l1943_194331

theorem fencers_count (n : ℕ) (h : n * (n - 1) = 72) : n = 9 :=
sorry

end fencers_count_l1943_194331


namespace range_of_m_l1943_194389

theorem range_of_m (a m : ℝ) (h_a_neg : a < 0) (y1 y2 : ℝ)
  (hA : y1 = a * m^2 - 4 * a * m)
  (hB : y2 = 4 * a * m^2 - 8 * a * m)
  (hA_above : y1 > -3 * a)
  (hB_above : y2 > -3 * a)
  (hy1_gt_y2 : y1 > y2) :
  4 / 3 < m ∧ m < 3 / 2 :=
sorry

end range_of_m_l1943_194389


namespace side_length_of_square_l1943_194384

theorem side_length_of_square (P : ℕ) (h1 : P = 28) (h2 : P = 4 * s) : s = 7 :=
  by sorry

end side_length_of_square_l1943_194384


namespace mass_percentage_of_C_in_CCl4_l1943_194374

theorem mass_percentage_of_C_in_CCl4 :
  let mass_carbon : ℝ := 12.01
  let mass_chlorine : ℝ := 35.45
  let molar_mass_CCl4 : ℝ := mass_carbon + 4 * mass_chlorine
  let mass_percentage_C : ℝ := (mass_carbon / molar_mass_CCl4) * 100
  mass_percentage_C = 7.81 := 
by
  sorry

end mass_percentage_of_C_in_CCl4_l1943_194374


namespace find_a_l1943_194339

theorem find_a (a : ℝ) (h : ∃ b : ℝ, (4:ℝ)*x^2 - (12:ℝ)*x + a = (2*x + b)^2) : a = 9 :=
sorry

end find_a_l1943_194339


namespace part1_part2_l1943_194302

noncomputable def f (x : ℝ) : ℝ := (2 * x) / (Real.log x)

theorem part1 : 
  (∀ x, 0 < x → x < 1 → (f x) < f (1)) ∧ 
  (∀ x, 1 < x → x < Real.exp 1 → (f x) < f (Real.exp 1)) :=
sorry

theorem part2 :
  ∃ k, k = 2 ∧ ∀ x, 0 < x → (f x) > (k / (Real.log x)) + 2 * Real.sqrt x :=
sorry

end part1_part2_l1943_194302


namespace min_value_of_number_l1943_194369

theorem min_value_of_number (a b c d : ℕ) (h1 : 0 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ d) (h5 : d ≤ 9) (h6 : 1 ≤ d) : 
  a + b * 10 + c * 100 + d * 1000 = 1119 :=
by
  sorry

end min_value_of_number_l1943_194369


namespace dvd_player_movie_ratio_l1943_194305

theorem dvd_player_movie_ratio (M D : ℝ) (h1 : D = M + 63) (h2 : D = 81) : D / M = 4.5 :=
by
  sorry

end dvd_player_movie_ratio_l1943_194305


namespace chord_length_l1943_194368

theorem chord_length (r d: ℝ) (h1: r = 5) (h2: d = 4) : ∃ EF, EF = 6 := by
  sorry

end chord_length_l1943_194368


namespace impossible_coins_l1943_194306

theorem impossible_coins (p1 p2 : ℝ) :
  ((1 - p1) * (1 - p2) = p1 * p2) →
  (p1 * (1 - p2) + p2 * (1 - p1) = p1 * p2) →
  false :=
by
  sorry

end impossible_coins_l1943_194306


namespace find_sets_l1943_194387

variable (A X Y : Set ℕ) -- Mimicking sets of natural numbers for generality.

theorem find_sets (h1 : X ∪ Y = A) (h2 : X ∩ A = Y) : X = A ∧ Y = A := by
  -- This would need a proof, which shows that: X = A and Y = A
  sorry

end find_sets_l1943_194387


namespace counter_example_exists_l1943_194371

theorem counter_example_exists : 
  ∃ n : ℕ, n ≥ 2 ∧ ¬(∃ k : ℕ, (2 ^ 2 ^ n) % (2 ^ n - 1) = 4 ^ k) :=
  sorry

end counter_example_exists_l1943_194371


namespace age_difference_l1943_194378

variables (A B C : ℕ)

theorem age_difference (h : A + B = B + C + 12) : A - C = 12 :=
sorry

end age_difference_l1943_194378


namespace relationship_y1_y2_y3_l1943_194326

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

end relationship_y1_y2_y3_l1943_194326


namespace tickets_to_buy_l1943_194334

theorem tickets_to_buy
  (ferris_wheel_cost : Float := 2.0)
  (roller_coaster_cost : Float := 7.0)
  (multiple_rides_discount : Float := 1.0)
  (newspaper_coupon : Float := 1.0) :
  (ferris_wheel_cost + roller_coaster_cost - multiple_rides_discount - newspaper_coupon = 7.0) :=
by
  sorry

end tickets_to_buy_l1943_194334


namespace third_term_of_arithmetic_sequence_l1943_194321

variable (a : ℕ → ℤ)
variable (a1_eq_2 : a 1 = 2)
variable (a2_eq_8 : a 2 = 8)
variable (arithmetic_seq : ∀ n : ℕ, a n = a 1 + (n - 1) * (a 2 - a 1))

theorem third_term_of_arithmetic_sequence :
  a 3 = 14 :=
by
  sorry

end third_term_of_arithmetic_sequence_l1943_194321


namespace spherical_to_rectangular_coordinates_l1943_194397

theorem spherical_to_rectangular_coordinates :
  let ρ := 5
  let θ := Real.pi / 4
  let φ := Real.pi / 3
  let x := ρ * Real.sin φ * Real.cos θ
  let y := ρ * Real.sin φ * Real.sin θ
  let z := ρ * Real.cos φ
  x = (5 * Real.sqrt 6) / 4 ∧ y = (5 * Real.sqrt 6) / 4 ∧ z = 5 / 2
:= by
  let ρ := 5
  let θ := Real.pi / 4
  let φ := Real.pi / 3
  let x := ρ * Real.sin φ * Real.cos θ
  let y := ρ * Real.sin φ * Real.sin θ
  let z := ρ * Real.cos φ
  have hx : x = (5 * Real.sqrt 6) / 4 := sorry
  have hy : y = (5 * Real.sqrt 6) / 4 := sorry
  have hz : z = 5 / 2 := sorry
  exact ⟨hx, hy, hz⟩

end spherical_to_rectangular_coordinates_l1943_194397


namespace sale_price_correct_l1943_194340

noncomputable def original_price : ℝ := 600.00
noncomputable def first_discount_factor : ℝ := 0.75
noncomputable def second_discount_factor : ℝ := 0.90
noncomputable def final_price : ℝ := original_price * first_discount_factor * second_discount_factor
noncomputable def expected_final_price : ℝ := 0.675 * original_price

theorem sale_price_correct : final_price = expected_final_price := sorry

end sale_price_correct_l1943_194340


namespace pies_baked_l1943_194396

/-- Mrs. Hilt baked 16.0 pecan pies and 14.0 apple pies. She needs 5.0 times this amount.
    Prove that the total number of pies she has to bake is 150.0. -/
theorem pies_baked (pecan_pies : ℝ) (apple_pies : ℝ) (times : ℝ)
  (h1 : pecan_pies = 16.0) (h2 : apple_pies = 14.0) (h3 : times = 5.0) :
  times * (pecan_pies + apple_pies) = 150.0 := by
  sorry

end pies_baked_l1943_194396


namespace basketball_count_l1943_194311

theorem basketball_count (s b v : ℕ) 
  (h1 : s = b + 23) 
  (h2 : v = s - 18)
  (h3 : v = 40) : b = 35 :=
by sorry

end basketball_count_l1943_194311


namespace ratio_james_paid_l1943_194323

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

end ratio_james_paid_l1943_194323


namespace initial_lives_emily_l1943_194348

theorem initial_lives_emily (L : ℕ) (h1 : L - 25 + 24 = 41) : L = 42 :=
by
  sorry

end initial_lives_emily_l1943_194348


namespace fruit_prob_l1943_194388

variable (O A B S : ℕ) 

-- Define the conditions
variables (H1 : O + A + B + S = 32)
variables (H2 : O - 5 = 3)
variables (H3 : A - 3 = 7)
variables (H4 : S - 2 = 4)
variables (H5 : 3 + 7 + 4 + B = 20)

-- Define the proof problem
theorem fruit_prob :
  (O = 8) ∧ (A = 10) ∧ (B = 6) ∧ (S = 6) → (O + S) / (O + A + B + S) = 7 / 16 := 
by
  sorry

end fruit_prob_l1943_194388


namespace negation_of_proposition_l1943_194366

theorem negation_of_proposition (p : ∀ x : ℝ, -x^2 + 4 * x + 3 > 0) :
  (∃ x : ℝ, -x^2 + 4 * x + 3 ≤ 0) :=
sorry

end negation_of_proposition_l1943_194366


namespace proportion_of_adopted_kittens_l1943_194398

-- Define the relevant objects and conditions in Lean
def breeding_rabbits : ℕ := 10
def kittens_first_spring := 10 * breeding_rabbits -- 100 kittens
def kittens_second_spring : ℕ := 60
def adopted_first_spring (P : ℝ) := 100 * P
def returned_first_spring : ℕ := 5
def adopted_second_spring : ℕ := 4
def total_rabbits_in_house (P : ℝ) :=
  breeding_rabbits + (kittens_first_spring - adopted_first_spring P + returned_first_spring) +
  (kittens_second_spring - adopted_second_spring)

theorem proportion_of_adopted_kittens : ∃ (P : ℝ), total_rabbits_in_house P = 121 ∧ P = 0.5 :=
by
  use 0.5
  -- Proof part (with "sorry" to skip the detailed proof)
  sorry

end proportion_of_adopted_kittens_l1943_194398


namespace frustum_slant_height_l1943_194399

theorem frustum_slant_height
  (ratio_area : ℝ)
  (slant_height_removed : ℝ)
  (sf_ratio : ratio_area = 1/16)
  (shr : slant_height_removed = 3) :
  ∃ (slant_height_frustum : ℝ), slant_height_frustum = 9 :=
by
  sorry

end frustum_slant_height_l1943_194399


namespace value_expression_possible_values_l1943_194309

open Real

noncomputable def value_expression (a b : ℝ) : ℝ :=
  a^2 + 2 * a * b + b^2 + 2 * a^2 * b + 2 * a * b^2 + a^2 * b^2

theorem value_expression_possible_values (a b : ℝ)
  (h1 : (a / b) + (b / a) = 5 / 2)
  (h2 : a - b = 3 / 2) :
  value_expression a b = 0 ∨ value_expression a b = 81 :=
sorry

end value_expression_possible_values_l1943_194309


namespace jason_spent_on_shorts_l1943_194365

def total_spent : ℝ := 14.28
def jacket_spent : ℝ := 4.74
def shorts_spent : ℝ := total_spent - jacket_spent

theorem jason_spent_on_shorts :
  shorts_spent = 9.54 :=
by
  -- Placeholder for the proof. The statement is correct as it matches the given problem data.
  sorry

end jason_spent_on_shorts_l1943_194365


namespace solve_equation_l1943_194317

def euler_totient (n : ℕ) : ℕ := sorry  -- Placeholder, Euler's φ function definition
def sigma_function (n : ℕ) : ℕ := sorry  -- Placeholder, σ function definition

theorem solve_equation (x : ℕ) : euler_totient (sigma_function (2^x)) = 2^x → x = 1 := by
  sorry

end solve_equation_l1943_194317


namespace max_ratio_three_digit_l1943_194364

theorem max_ratio_three_digit (x a b c : ℕ) (h1 : 100 * a + 10 * b + c = x) (h2 : 1 ≤ a ∧ a ≤ 9)
  (h3 : 0 ≤ b ∧ b ≤ 9) (h4 : 0 ≤ c ∧ c ≤ 9) : 
  (x : ℚ) / (a + b + c) ≤ 100 := sorry

end max_ratio_three_digit_l1943_194364


namespace probability_of_choosing_A_on_second_day_l1943_194386

-- Definitions of the probabilities given in the problem conditions.
def p_first_day_A := 0.5
def p_first_day_B := 0.5
def p_second_day_A_given_first_day_A := 0.6
def p_second_day_A_given_first_day_B := 0.5

-- Define the problem to be proved in Lean 4
theorem probability_of_choosing_A_on_second_day :
  (p_first_day_A * p_second_day_A_given_first_day_A) +
  (p_first_day_B * p_second_day_A_given_first_day_B) = 0.55 :=
by
  sorry

end probability_of_choosing_A_on_second_day_l1943_194386


namespace volume_of_pyramid_SPQR_l1943_194327

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

end volume_of_pyramid_SPQR_l1943_194327


namespace minimum_unit_cubes_l1943_194303

theorem minimum_unit_cubes (n : ℕ) (N : ℕ) : 
  (n ≥ 3) → (N = n^3) → ((n - 2)^3 > (1/2) * n^3) → 
  ∃ n : ℕ, N = n^3 ∧ (n - 2)^3 > (1/2) * n^3 ∧ N = 1000 :=
by
  intros
  sorry

end minimum_unit_cubes_l1943_194303


namespace pencils_per_student_l1943_194367

theorem pencils_per_student (total_pencils : ℕ) (students : ℕ) (pencils_per_student : ℕ) 
  (h_total : total_pencils = 125) 
  (h_students : students = 25) 
  (h_div : pencils_per_student = total_pencils / students) : 
  pencils_per_student = 5 :=
by
  sorry

end pencils_per_student_l1943_194367


namespace proof_by_contradiction_l1943_194316

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

end proof_by_contradiction_l1943_194316


namespace problem_1_problem_2_l1943_194391

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | -2 < x ∧ x < 3}
def B : Set ℝ := {x | -3 ≤ x ∧ x ≤ 2}

-- 1. Prove that A ∩ B = {x | -2 < x ≤ 2}
theorem problem_1 : A ∩ B = {x | -2 < x ∧ x ≤ 2} :=
by
  sorry

-- 2. Prove that (complement U A) ∪ B = {x | x ≤ 2 ∨ x ≥ 3}
theorem problem_2 : (U \ A) ∪ B = {x | x ≤ 2 ∨ x ≥ 3} :=
by
  sorry

end problem_1_problem_2_l1943_194391


namespace cos_alpha_in_second_quadrant_l1943_194375

theorem cos_alpha_in_second_quadrant 
  (alpha : ℝ) 
  (h1 : π / 2 < alpha ∧ alpha < π)
  (h2 : ∀ x y : ℝ, 2 * x + (Real.tan alpha) * y + 1 = 0 → 8 / 3 = -(2 / (Real.tan alpha))) :
  Real.cos alpha = -4 / 5 :=
by
  sorry

end cos_alpha_in_second_quadrant_l1943_194375


namespace certain_number_l1943_194307

theorem certain_number (x y : ℝ) (h1 : 0.65 * x = 0.20 * y) (h2 : x = 210) : y = 682.5 :=
by
  sorry

end certain_number_l1943_194307


namespace total_doughnuts_l1943_194376

-- Definitions used in the conditions
def boxes : ℕ := 4
def doughnuts_per_box : ℕ := 12

theorem total_doughnuts : boxes * doughnuts_per_box = 48 :=
by
  sorry

end total_doughnuts_l1943_194376


namespace find_S_11_l1943_194363

variables (a : ℕ → ℤ)
variables (d : ℤ) (n : ℕ)

def is_arithmetic_sequence : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def sum_first_n_terms (n : ℕ) : ℤ :=
  (n * (a 1 + a n)) / 2

noncomputable def a_3 := a 3
noncomputable def a_6 := a 6
noncomputable def a_9 := a 9

theorem find_S_11
  (h1 : is_arithmetic_sequence a d)
  (h2 : a_3 + a_9 = 18 - a_6) :
  sum_first_n_terms a 11 = 66 :=
sorry

end find_S_11_l1943_194363


namespace area_of_rectangle_is_108_l1943_194373

theorem area_of_rectangle_is_108 (s w l : ℕ) (h₁ : s * s = 36) (h₂ : w = s) (h₃ : l = 3 * w) : w * l = 108 :=
by
  -- This is a placeholder for a detailed proof.
  sorry

end area_of_rectangle_is_108_l1943_194373


namespace piles_stones_l1943_194313

theorem piles_stones (a b c d : ℕ)
  (h₁ : a = 2011)
  (h₂ : b = 2010)
  (h₃ : c = 2009)
  (h₄ : d = 2008) :
  ∃ (k l m n : ℕ), (k, l, m, n) = (0, 0, 0, 2) ∧
  ((∃ x y z w : ℕ, k = x - y ∧ l = y - z ∧ m = z - w ∧ x + l + m + w = 0) ∨
   (∃ u : ℕ, k = a - u ∧ l = b - u ∧ m = c - u ∧ n = d - u)) :=
sorry

end piles_stones_l1943_194313


namespace geometric_progression_difference_l1943_194380

variable {n : ℕ}
variable {a : ℕ → ℝ} -- assuming the sequence is indexed by natural numbers
variable {a₁ : ℝ}
variable {r : ℝ} (hr : r = (1 + Real.sqrt 5) / 2)

def geometric_progression (a : ℕ → ℝ) (a₁ : ℝ) (r : ℝ) : Prop :=
  ∀ n, a n = a₁ * (r ^ n)

theorem geometric_progression_difference
  (a₁ : ℝ)
  (hr : r = (1 + Real.sqrt 5) / 2)
  (hg : geometric_progression a a₁ r) :
  ∀ n, n ≥ 2 → a n = a (n-1) - a (n-2) :=
by
  sorry

end geometric_progression_difference_l1943_194380


namespace total_population_is_3311_l1943_194358

-- Definitions based on the problem's conditions
def fewer_than_6000_inhabitants (L : ℕ) : Prop :=
  L < 6000

def more_girls_than_boys (girls boys : ℕ) : Prop :=
  girls = (11 * boys) / 10

def more_men_than_women (men women : ℕ) : Prop :=
  men = (23 * women) / 20

def more_children_than_adults (children adults : ℕ) : Prop :=
  children = (6 * adults) / 5

-- Prove that the total population is 3311 given the described conditions
theorem total_population_is_3311 {L n men women children boys girls : ℕ}
  (hc : more_children_than_adults children (n + men))
  (hm : more_men_than_women men n)
  (hg : more_girls_than_boys girls boys)
  (hL : L = n + men + boys + girls)
  (hL_lt : fewer_than_6000_inhabitants L) :
  L = 3311 :=
sorry

end total_population_is_3311_l1943_194358


namespace smallest_multiple_5_711_l1943_194333

theorem smallest_multiple_5_711 : ∃ n : ℕ, n = Nat.lcm 5 711 ∧ n = 3555 := 
by
  sorry

end smallest_multiple_5_711_l1943_194333


namespace area_of_cross_section_l1943_194346

noncomputable def area_cross_section (H α : ℝ) : ℝ :=
  let AC := 2 * H * Real.sqrt 3 * Real.tan (Real.pi / 2 - α)
  let MK := (H / 2) * Real.sqrt (1 + 16 * (Real.tan (Real.pi / 2 - α))^2)
  (1 / 2) * AC * MK

theorem area_of_cross_section (H α : ℝ) :
  area_cross_section H α = (H^2 * Real.sqrt 3 * Real.tan (Real.pi / 2 - α) / 2) * Real.sqrt (1 + 16 * (Real.tan (Real.pi / 2 - α))^2) :=
sorry

end area_of_cross_section_l1943_194346


namespace find_x_for_given_y_l1943_194345

theorem find_x_for_given_y (x y : ℝ) (h_pos : 0 < x ∧ 0 < y) (h_initial : x = 2 ∧ y = 8) (h_inverse : (2 ^ 3) * 8 = 128) :
  y = 1728 → x = (1 / (13.5) ^ (1 / 3)) :=
by
  sorry

end find_x_for_given_y_l1943_194345


namespace marble_count_l1943_194332

theorem marble_count (a : ℕ) (h1 : a + 3 * a + 6 * a + 30 * a = 120) : a = 3 :=
  sorry

end marble_count_l1943_194332


namespace model_tower_height_l1943_194353

-- Definitions based on conditions
def height_actual_tower : ℝ := 60
def volume_actual_tower : ℝ := 80000
def volume_model_tower : ℝ := 0.5

-- Theorem statement
theorem model_tower_height (h: ℝ) : h = 0.15 :=
by
  sorry

end model_tower_height_l1943_194353


namespace inequality_proof_l1943_194382

theorem inequality_proof (x y : ℝ) : 5 * x^2 + y^2 + 4 ≥ 4 * x + 4 * x * y :=
by
  sorry

end inequality_proof_l1943_194382


namespace number_of_5_dollar_bills_l1943_194350

def total_money : ℤ := 45
def value_of_each_bill : ℤ := 5

theorem number_of_5_dollar_bills : total_money / value_of_each_bill = 9 := by
  sorry

end number_of_5_dollar_bills_l1943_194350
