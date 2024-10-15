import Mathlib

namespace NUMINAMATH_GPT_relationship_x_y_l410_41045

theorem relationship_x_y (x y m : ℝ) (h1 : x + m = 4) (h2 : y - 5 = m) : x + y = 9 := 
by 
  sorry

end NUMINAMATH_GPT_relationship_x_y_l410_41045


namespace NUMINAMATH_GPT_base8_to_base10_l410_41042

theorem base8_to_base10 {a b : ℕ} (h1 : 3 * 64 + 7 * 8 + 4 = 252) (h2 : 252 = a * 10 + b) :
  (a + b : ℝ) / 20 = 0.35 :=
sorry

end NUMINAMATH_GPT_base8_to_base10_l410_41042


namespace NUMINAMATH_GPT_math_problem_l410_41043

variable (p q r : ℝ)

theorem math_problem
  (h1: p + q + r = 5)
  (h2: 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) :
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := by
  sorry

end NUMINAMATH_GPT_math_problem_l410_41043


namespace NUMINAMATH_GPT_total_money_is_twenty_l410_41064

-- Define Henry's initial money
def henry_initial_money : Nat := 5

-- Define the money Henry earned
def henry_earned_money : Nat := 2

-- Define Henry's total money
def henry_total_money : Nat := henry_initial_money + henry_earned_money

-- Define friend's money
def friend_money : Nat := 13

-- Define the total combined money
def total_combined_money : Nat := henry_total_money + friend_money

-- The main statement to prove
theorem total_money_is_twenty : total_combined_money = 20 := sorry

end NUMINAMATH_GPT_total_money_is_twenty_l410_41064


namespace NUMINAMATH_GPT_least_people_to_complete_job_on_time_l410_41028

theorem least_people_to_complete_job_on_time
  (total_duration : ℕ)
  (initial_days : ℕ)
  (initial_people : ℕ)
  (initial_work_done : ℚ)
  (efficiency_multiplier : ℚ)
  (remaining_work_fraction : ℚ)
  (remaining_days : ℕ)
  (resulting_people : ℕ)
  (work_rate_doubled : ℕ → ℚ → ℚ)
  (final_resulting_people : ℚ)
  : initial_work_done = 1/4 →
    efficiency_multiplier = 2 →
    remaining_work_fraction = 3/4 →
    total_duration = 40 →
    initial_days = 10 →
    initial_people = 12 →
    remaining_days = 20 →
    work_rate_doubled 12 2 = 24 →
    final_resulting_people = (1/2) →
    resulting_people = 6 :=
sorry

end NUMINAMATH_GPT_least_people_to_complete_job_on_time_l410_41028


namespace NUMINAMATH_GPT_count_even_thousands_digit_palindromes_l410_41072

-- Define the set of valid digits
def valid_A : Finset ℕ := {2, 4, 6, 8}
def valid_B : Finset ℕ := Finset.range 10

-- Define the condition of a four-digit palindrome ABBA where A is even and non-zero
def is_valid_palindrome (a b : ℕ) : Prop :=
  a ∈ valid_A ∧ b ∈ valid_B

-- The proof problem: Prove that the total number of valid palindromes ABBA is 40
theorem count_even_thousands_digit_palindromes :
  (valid_A.card) * (valid_B.card) = 40 :=
by
  -- Skipping the proof itself
  sorry

end NUMINAMATH_GPT_count_even_thousands_digit_palindromes_l410_41072


namespace NUMINAMATH_GPT_area_of_region_l410_41060

theorem area_of_region (x y : ℝ) (h : x^2 + y^2 + 6 * x - 8 * y - 5 = 0) : 
  ∃ (r : ℝ), (π * r^2 = 30 * π) :=
by -- Starting the proof, skipping the detailed steps
sorry -- Proof placeholder

end NUMINAMATH_GPT_area_of_region_l410_41060


namespace NUMINAMATH_GPT_find_erased_number_l410_41022

/-- Define the variables used in the conditions -/
def n : ℕ := 69
def erased_number_mean : ℚ := 35 + 7 / 17
def sequence_sum : ℕ := n * (n + 1) / 2

/-- State the condition for the erased number -/
noncomputable def erased_number (x : ℕ) : Prop :=
  (sequence_sum - x) / (n - 1) = erased_number_mean

/-- The main theorem stating that the erased number is 7 -/
theorem find_erased_number : ∃ x : ℕ, erased_number x ∧ x = 7 :=
by
  use 7
  unfold erased_number sequence_sum
  -- Sum of first 69 natural numbers is 69 * (69 + 1) / 2
  -- Hence,
  -- (69 * 70 / 2 - 7) / 68 = 35 + 7 / 17
  -- which simplifies to true under these conditions
  -- Detailed proof skipped here as per instructions
  sorry

end NUMINAMATH_GPT_find_erased_number_l410_41022


namespace NUMINAMATH_GPT_diamond_3_7_l410_41063

def star (a b : ℕ) : ℕ := a^2 + 2*a*b + b^2
def diamond (a b : ℕ) : ℕ := star a b - a * b

theorem diamond_3_7 : diamond 3 7 = 79 :=
by 
  sorry

end NUMINAMATH_GPT_diamond_3_7_l410_41063


namespace NUMINAMATH_GPT_find_analytical_expression_of_f_l410_41035

-- Given conditions: f(1/x) = 1/(x+1)
def f (x : ℝ) : ℝ := sorry

-- Domain statement (optional for additional clarity):
def domain (x : ℝ) := x ≠ 0 ∧ x ≠ -1

-- Proof obligation: Prove that f(x) = x / (x + 1)
theorem find_analytical_expression_of_f :
  ∀ x : ℝ, domain x → f x = x / (x + 1) := sorry

end NUMINAMATH_GPT_find_analytical_expression_of_f_l410_41035


namespace NUMINAMATH_GPT_smallest_possible_value_of_other_number_l410_41020

theorem smallest_possible_value_of_other_number (x n : ℕ) (h_pos : x > 0) 
  (h_gcd : Nat.gcd 72 n = x + 6) (h_lcm : Nat.lcm 72 n = x * (x + 6)) : n = 12 := by
  sorry

end NUMINAMATH_GPT_smallest_possible_value_of_other_number_l410_41020


namespace NUMINAMATH_GPT_power_inequality_l410_41046

open Nat

theorem power_inequality (a b : ℝ) (n : ℕ)
  (h1 : 0 < a) (h2 : 0 < b) (h3 : (1 / a) + (1 / b) = 1) :
  (a + b)^n - a^n - b^n ≥ 2^(2 * n) - 2^(n + 1) := 
  sorry

end NUMINAMATH_GPT_power_inequality_l410_41046


namespace NUMINAMATH_GPT_exists_common_plane_l410_41097

-- Definition of the triangular pyramids
structure Pyramid :=
(base_area : ℝ)
(height : ℝ)

-- Function to represent the area of the intersection produced by a horizontal plane at distance x from the table
noncomputable def sectional_area (P : Pyramid) (x : ℝ) : ℝ :=
  P.base_area * (1 - x / P.height) ^ 2

-- Given seven pyramids
variables {P1 P2 P3 P4 P5 P6 P7 : Pyramid}

-- For any three pyramids, there exists a horizontal plane that intersects them in triangles of equal area
axiom triple_intersection:
  ∀ (Pi Pj Pk : Pyramid), ∃ x : ℝ, x ≥ 0 ∧ x ≤ min (Pi.height) (min (Pj.height) (Pk.height)) ∧
    sectional_area Pi x = sectional_area Pj x ∧ sectional_area Pk x = sectional_area Pi x

-- Prove that there exists a plane that intersects all seven pyramids in triangles of equal area
theorem exists_common_plane :
  ∃ x : ℝ, x ≥ 0 ∧ x ≤ min P1.height (min P2.height (min P3.height (min P4.height (min P5.height (min P6.height P7.height))))) ∧
    sectional_area P1 x = sectional_area P2 x ∧
    sectional_area P2 x = sectional_area P3 x ∧
    sectional_area P3 x = sectional_area P4 x ∧
    sectional_area P4 x = sectional_area P5 x ∧
    sectional_area P5 x = sectional_area P6 x ∧
    sectional_area P6 x = sectional_area P7 x :=
sorry

end NUMINAMATH_GPT_exists_common_plane_l410_41097


namespace NUMINAMATH_GPT_M_intersection_P_l410_41066

namespace IntersectionProof

-- Defining the sets M and P with given conditions
def M : Set ℝ := {y | ∃ x : ℝ, y = 3 ^ x}
def P : Set ℝ := {y | y ≥ 1}

-- The theorem that corresponds to the problem statement
theorem M_intersection_P : (M ∩ P) = {y | y ≥ 1} :=
sorry

end IntersectionProof

end NUMINAMATH_GPT_M_intersection_P_l410_41066


namespace NUMINAMATH_GPT_estate_value_l410_41004

theorem estate_value (E : ℝ) (x : ℝ) (hx : 5 * x = 0.6 * E) (charity_share : ℝ)
  (hcharity : charity_share = 800) (hwife : 3 * x * 4 = 12 * x)
  (htotal : E = 17 * x + charity_share) : E = 1923 :=
by
  sorry

end NUMINAMATH_GPT_estate_value_l410_41004


namespace NUMINAMATH_GPT_students_per_bench_l410_41038

theorem students_per_bench (num_male num_benches : ℕ) (h₁ : num_male = 29) (h₂ : num_benches = 29) (h₃ : ∀ num_female, num_female = 4 * num_male) : 
  ((29 + 4 * 29) / 29) = 5 :=
by
  sorry

end NUMINAMATH_GPT_students_per_bench_l410_41038


namespace NUMINAMATH_GPT_original_number_l410_41089

theorem original_number (x : ℕ) (h : ∃ k, 14 * x = 112 * k) : x = 8 :=
sorry

end NUMINAMATH_GPT_original_number_l410_41089


namespace NUMINAMATH_GPT_circumcenter_rational_l410_41034

theorem circumcenter_rational {a1 b1 a2 b2 a3 b3 : ℚ} 
  (h1 : a1 ≠ a2 ∨ b1 ≠ b2) 
  (h2 : a1 ≠ a3 ∨ b1 ≠ b3) 
  (h3 : a2 ≠ a3 ∨ b2 ≠ b3) :
  ∃ (x y : ℚ), 
    (x - a1)^2 + (y - b1)^2 = (x - a2)^2 + (y - b2)^2 ∧
    (x - a1)^2 + (y - b1)^2 = (x - a3)^2 + (y - b3)^2 :=
sorry

end NUMINAMATH_GPT_circumcenter_rational_l410_41034


namespace NUMINAMATH_GPT_jason_age_at_end_of_2004_l410_41000

noncomputable def jason_age_in_1997 (y : ℚ) (g : ℚ) : Prop :=
  y = g / 3 

noncomputable def birth_years_sum (y : ℚ) (g : ℚ) : Prop :=
  (1997 - y) + (1997 - g) = 3852

theorem jason_age_at_end_of_2004
  (y g : ℚ)
  (h1 : jason_age_in_1997 y g)
  (h2 : birth_years_sum y g) :
  y + 7 = 42.5 :=
by
  sorry

end NUMINAMATH_GPT_jason_age_at_end_of_2004_l410_41000


namespace NUMINAMATH_GPT_tangential_circle_radius_l410_41068

theorem tangential_circle_radius (R r x : ℝ) (hR : R > r) (hx : x = 4 * R * r / (R + r)) :
  ∃ x, x = 4 * R * r / (R + r) := by
sorry

end NUMINAMATH_GPT_tangential_circle_radius_l410_41068


namespace NUMINAMATH_GPT_part1_coordinates_on_x_axis_part2_coordinates_parallel_y_axis_part3_distances_equal_second_quadrant_l410_41062

-- Part (1)
theorem part1_coordinates_on_x_axis (a : ℝ) (h : a + 5 = 0) : (2*a - 2, a + 5) = (-12, 0) :=
by sorry

-- Part (2)
theorem part2_coordinates_parallel_y_axis (a : ℝ) (h : 2*a - 2 = 4) : (2*a - 2, a + 5) = (4, 8) :=
by sorry

-- Part (3)
theorem part3_distances_equal_second_quadrant (a : ℝ) 
  (h1 : 2*a-2 < 0) (h2 : a+5 > 0) (h3 : abs (2*a - 2) = abs (a + 5)) : a^(2022 : ℕ) + 2022 = 2023 :=
by sorry

end NUMINAMATH_GPT_part1_coordinates_on_x_axis_part2_coordinates_parallel_y_axis_part3_distances_equal_second_quadrant_l410_41062


namespace NUMINAMATH_GPT_exists_integer_n_l410_41007

theorem exists_integer_n (k : ℕ) (hk : 0 < k) : 
  ∃ n : ℤ, (n + 1981^k)^(1/2 : ℝ) + (n : ℝ)^(1/2 : ℝ) = (1982^(1/2 : ℝ) + 1) ^ k :=
sorry

end NUMINAMATH_GPT_exists_integer_n_l410_41007


namespace NUMINAMATH_GPT_sin_405_eq_sqrt2_div2_l410_41005

theorem sin_405_eq_sqrt2_div2 : Real.sin (405 * Real.pi / 180) = Real.sqrt 2 / 2 := 
by
  sorry

end NUMINAMATH_GPT_sin_405_eq_sqrt2_div2_l410_41005


namespace NUMINAMATH_GPT_total_quarters_l410_41049

-- Definitions from conditions
def initial_quarters : ℕ := 49
def quarters_given_by_dad : ℕ := 25

-- Theorem to prove the total quarters is 74
theorem total_quarters : initial_quarters + quarters_given_by_dad = 74 :=
by sorry

end NUMINAMATH_GPT_total_quarters_l410_41049


namespace NUMINAMATH_GPT_world_expo_visitors_l410_41009

noncomputable def per_person_cost (x : ℕ) : ℕ :=
  if x <= 30 then 120 else max (120 - 2 * (x - 30)) 90

theorem world_expo_visitors (x : ℕ) (h_cost : x * per_person_cost x = 4000) : x = 40 :=
by
  sorry

end NUMINAMATH_GPT_world_expo_visitors_l410_41009


namespace NUMINAMATH_GPT_estimate_pi_simulation_l410_41091

theorem estimate_pi_simulation :
  let side := 2
  let radius := 1
  let total_seeds := 1000
  let seeds_in_circle := 778
  (π : ℝ) * radius^2 / side^2 = (seeds_in_circle : ℝ) / total_seeds → π = 3.112 :=
by
  intros
  sorry

end NUMINAMATH_GPT_estimate_pi_simulation_l410_41091


namespace NUMINAMATH_GPT_house_transaction_l410_41082

variable (initial_value : ℝ) (loss_rate : ℝ) (gain_rate : ℝ) (final_loss : ℝ)

theorem house_transaction
  (h_initial : initial_value = 12000)
  (h_loss : loss_rate = 0.15)
  (h_gain : gain_rate = 0.15)
  (h_final_loss : final_loss = 270) :
  let selling_price := initial_value * (1 - loss_rate)
  let buying_price := selling_price * (1 + gain_rate)
  (initial_value - buying_price) = final_loss :=
by
  simp only [h_initial, h_loss, h_gain, h_final_loss]
  sorry

end NUMINAMATH_GPT_house_transaction_l410_41082


namespace NUMINAMATH_GPT_tank_fill_time_l410_41018

theorem tank_fill_time (A_rate B_rate C_rate : ℝ) (hA : A_rate = 1/30) (hB : B_rate = 1/20) (hC : C_rate = -1/40) : 
  1 / (A_rate + B_rate + C_rate) = 120 / 7 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_tank_fill_time_l410_41018


namespace NUMINAMATH_GPT_sequence_sum_identity_l410_41048

theorem sequence_sum_identity 
  (a_n b_n : ℕ → ℕ) 
  (S_n T_n : ℕ → ℕ)
  (h1 : ∀ n, b_n n - a_n n = 2^n + 1)
  (h2 : ∀ n, S_n n + T_n n = 2^(n+1) + n^2 - 2) : 
  ∀ n, 2 * T_n n = n * (n - 1) :=
by sorry

end NUMINAMATH_GPT_sequence_sum_identity_l410_41048


namespace NUMINAMATH_GPT_ab_range_l410_41021

theorem ab_range (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + b + (1 / a) + (1 / b) = 5) :
  1 ≤ a + b ∧ a + b ≤ 4 :=
by
  sorry

end NUMINAMATH_GPT_ab_range_l410_41021


namespace NUMINAMATH_GPT_probability_below_8_l410_41087

theorem probability_below_8 (p10 p9 p8 : ℝ) (h1 : p10 = 0.20) (h2 : p9 = 0.30) (h3 : p8 = 0.10) : 
  1 - (p10 + p9 + p8) = 0.40 :=
by 
  rw [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_probability_below_8_l410_41087


namespace NUMINAMATH_GPT_line_segment_AB_length_l410_41084

noncomputable def length_AB (xA yA xB yB : ℝ) : ℝ :=
  Real.sqrt ((xA - xB)^2 + (yA - yB)^2)

theorem line_segment_AB_length :
  ∀ (xA yA xB yB : ℝ),
    (xA - yA = 0) →
    (xB + yB = 0) →
    (∃ k : ℝ, yA = k * (xA + 1) ∧ yB = k * (xB + 1)) →
    (-1 ≤ xA ∧ xA ≤ 0) →
    (xA + xB = 2 * k ∧ yA + yB = 2 * k) →
    length_AB xA yA xB yB = (4/3) * Real.sqrt 5 :=
by
  intros xA yA xB yB h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_line_segment_AB_length_l410_41084


namespace NUMINAMATH_GPT_sum_of_money_invested_l410_41006

noncomputable def principal_sum_of_money (R : ℝ) (T : ℝ) (CI_minus_SI : ℝ) : ℝ :=
  let SI := (625 * R * T / 100)
  let CI := 625 * ((1 + R / 100)^(T : ℝ) - 1)
  if (CI - SI = CI_minus_SI)
  then 625
  else 0

theorem sum_of_money_invested : 
  (principal_sum_of_money 4 2 1) = 625 :=
by
  unfold principal_sum_of_money
  sorry

end NUMINAMATH_GPT_sum_of_money_invested_l410_41006


namespace NUMINAMATH_GPT_miles_left_l410_41015

theorem miles_left (d_total d_covered d_left : ℕ) 
  (h₁ : d_total = 78) 
  (h₂ : d_covered = 32) 
  (h₃ : d_left = d_total - d_covered):
  d_left = 46 := 
by {
  sorry 
}

end NUMINAMATH_GPT_miles_left_l410_41015


namespace NUMINAMATH_GPT_molecular_weight_CaOH2_l410_41081

def atomic_weight_Ca : ℝ := 40.08
def atomic_weight_O : ℝ := 16.00
def atomic_weight_H : ℝ := 1.01

theorem molecular_weight_CaOH2 :
  (atomic_weight_Ca + 2 * atomic_weight_O + 2 * atomic_weight_H = 74.10) := 
by 
  sorry

end NUMINAMATH_GPT_molecular_weight_CaOH2_l410_41081


namespace NUMINAMATH_GPT_closed_polygonal_chain_exists_l410_41051

theorem closed_polygonal_chain_exists (n m : ℕ) : 
  ((n % 2 = 1 ∨ m % 2 = 1) ↔ 
   ∃ (length : ℕ), length = (n + 1) * (m + 1) ∧ length % 2 = 0) :=
by sorry

end NUMINAMATH_GPT_closed_polygonal_chain_exists_l410_41051


namespace NUMINAMATH_GPT_find_w_l410_41094

variable (p j t : ℝ) (w : ℝ)

-- Definitions based on conditions
def j_less_than_p : Prop := j = 0.75 * p
def j_less_than_t : Prop := j = 0.80 * t
def t_less_than_p : Prop := t = p * (1 - w / 100)

-- Objective: Prove that given these conditions, w = 6.25
theorem find_w (h1 : j_less_than_p p j) (h2 : j_less_than_t j t) (h3 : t_less_than_p t p w) : 
  w = 6.25 := 
by 
  sorry

end NUMINAMATH_GPT_find_w_l410_41094


namespace NUMINAMATH_GPT_triangle_properties_equivalence_l410_41086

-- Define the given properties for the two triangles
variables {A B C A' B' C' : Type}

-- Triangle side lengths and properties
def triangles_equal (b b' c c' : ℝ) : Prop :=
  (b = b') ∧ (c = c')

def equivalent_side_lengths (a a' b b' c c' : ℝ) : Prop :=
  a = a'

def equivalent_medians (ma ma' b b' c c' a a' : ℝ) : Prop :=
  ma = ma'

def equivalent_altitudes (ha ha' Δ Δ' a a' : ℝ) : Prop :=
  ha = ha'

def equivalent_angle_bisectors (ta ta' b b' c c' a a' : ℝ) : Prop :=
  ta = ta'

def equivalent_circumradii (R R' a a' b b' c c' : ℝ) : Prop :=
  R = R'

def equivalent_areas (Δ Δ' b b' c c' A A' : ℝ) : Prop :=
  Δ = Δ'

-- Main theorem statement
theorem triangle_properties_equivalence
  (b b' c c' a a' ma ma' ha ha' ta ta' R R' Δ Δ' : ℝ)
  (A A' : ℝ)
  (eq_b : b = b')
  (eq_c : c = c') :
  equivalent_side_lengths a a' b b' c c' ∧ 
  equivalent_medians ma ma' b b' c c' a a' ∧ 
  equivalent_altitudes ha ha' Δ Δ' a a' ∧ 
  equivalent_angle_bisectors ta ta' b b' c c' a a' ∧ 
  equivalent_circumradii R R' a a' b b' c c' ∧ 
  equivalent_areas Δ Δ' b b' c c' A A'
:= by
  sorry

end NUMINAMATH_GPT_triangle_properties_equivalence_l410_41086


namespace NUMINAMATH_GPT_prob_2_out_of_5_exactly_A_and_B_l410_41085

noncomputable def probability_exactly_A_and_B_selected (students : List String) : ℚ :=
  if students = ["A", "B", "C", "D", "E"] then 1 / 10 else 0

theorem prob_2_out_of_5_exactly_A_and_B :
  probability_exactly_A_and_B_selected ["A", "B", "C", "D", "E"] = 1 / 10 :=
by 
  sorry

end NUMINAMATH_GPT_prob_2_out_of_5_exactly_A_and_B_l410_41085


namespace NUMINAMATH_GPT_min_value_expression_l410_41069

theorem min_value_expression (a b c : ℝ) (h1 : 1 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ 5) :
  (a - 1)^2 + ((b / a) - 1)^2 + ((c / b) - 1)^2 + ((5 / c) - 1)^2 ≥ 20 - 8 * Real.sqrt 5 := 
by
  sorry

end NUMINAMATH_GPT_min_value_expression_l410_41069


namespace NUMINAMATH_GPT_solution_to_eq_l410_41092

def eq1 (x y z t : ℕ) : Prop := x * y - x * z + y * t = 182
def cond_numbers (n : ℕ) : Prop := n = 12 ∨ n = 14 ∨ n = 37 ∨ n = 65

theorem solution_to_eq 
  (x y z t : ℕ) 
  (hx : cond_numbers x) 
  (hy : cond_numbers y) 
  (hz : cond_numbers z) 
  (ht : cond_numbers t) 
  (h : eq1 x y z t) : 
  (x = 12 ∧ y = 37 ∧ z = 65 ∧ t = 14) ∨ 
  (x = 37 ∧ y = 12 ∧ z = 14 ∧ t = 65) := 
sorry

end NUMINAMATH_GPT_solution_to_eq_l410_41092


namespace NUMINAMATH_GPT_toothpicks_needed_l410_41056

-- Defining the number of rows in the large equilateral triangle.
def rows : ℕ := 10

-- Formula to compute the total number of smaller equilateral triangles.
def total_small_triangles (n : ℕ) : ℕ := n * (n + 1) / 2

-- Number of small triangles in this specific case.
def num_small_triangles : ℕ := total_small_triangles rows

-- Total toothpicks without sharing sides.
def total_sides_no_sharing (n : ℕ) : ℕ := 3 * num_small_triangles

-- Adjust for shared toothpicks internally.
def shared_toothpicks (n : ℕ) : ℕ := (total_sides_no_sharing n - 3 * rows) / 2 + 3 * rows

-- Total boundary toothpicks.
def boundary_toothpicks (n : ℕ) : ℕ := 3 * rows

-- Final total number of toothpicks required.
def total_toothpicks (n : ℕ) : ℕ := shared_toothpicks n + boundary_toothpicks n

-- The theorem to be proved
theorem toothpicks_needed : total_toothpicks rows = 98 :=
by
  -- You can complete the proof.
  sorry

end NUMINAMATH_GPT_toothpicks_needed_l410_41056


namespace NUMINAMATH_GPT_intersection_points_l410_41040

def f(x : ℝ) : ℝ := x^2 + 3*x + 2
def g(x : ℝ) : ℝ := 4*x^2 + 6*x + 2

theorem intersection_points : {p : ℝ × ℝ | ∃ x, f x = p.2 ∧ g x = p.2 ∧ p.1 = x} = { (0, 2), (-1, 0) } := 
by {
  sorry
}

end NUMINAMATH_GPT_intersection_points_l410_41040


namespace NUMINAMATH_GPT_range_of_a_l410_41061

theorem range_of_a (a : ℝ) : ¬ (∃ x : ℝ, a * x^2 - 3 * a * x + 9 ≤ 0) → a ∈ Set.Ico 0 4 := by
  sorry

end NUMINAMATH_GPT_range_of_a_l410_41061


namespace NUMINAMATH_GPT_students_standing_count_l410_41008

def students_seated : ℕ := 300
def teachers_seated : ℕ := 30
def total_attendees : ℕ := 355

theorem students_standing_count : total_attendees - (students_seated + teachers_seated) = 25 :=
by
  sorry

end NUMINAMATH_GPT_students_standing_count_l410_41008


namespace NUMINAMATH_GPT_complex_imaginary_part_l410_41001

theorem complex_imaginary_part : 
  Complex.im ((1 : ℂ) / (-2 + Complex.I) + (1 : ℂ) / (1 - 2 * Complex.I)) = 1/5 := 
  sorry

end NUMINAMATH_GPT_complex_imaginary_part_l410_41001


namespace NUMINAMATH_GPT_find_expression_l410_41053

theorem find_expression (E a : ℝ) 
  (h1 : (E + (3 * a - 8)) / 2 = 69) 
  (h2 : a = 26) : 
  E = 68 :=
sorry

end NUMINAMATH_GPT_find_expression_l410_41053


namespace NUMINAMATH_GPT_product_price_interval_l410_41047

def is_too_high (price guess : ℕ) : Prop := guess > price
def is_too_low  (price guess : ℕ) : Prop := guess < price

theorem product_price_interval 
    (price : ℕ)
    (h1 : is_too_high price 2000)
    (h2 : is_too_low price 1000)
    (h3 : is_too_high price 1500)
    (h4 : is_too_low price 1250)
    (h5 : is_too_low price 1375) :
    1375 < price ∧ price < 1500 :=
    sorry

end NUMINAMATH_GPT_product_price_interval_l410_41047


namespace NUMINAMATH_GPT_find_t_value_l410_41023

theorem find_t_value (x y z t : ℕ) (hx : x = 1) (hy : y = 2) (hz : z = 3) (hpos_x : 0 < x) (hpos_y : 0 < y) (hpos_z : 0 < z) :
  x + y + z + t = 10 → t = 4 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_find_t_value_l410_41023


namespace NUMINAMATH_GPT_determine_constants_l410_41075

theorem determine_constants :
  ∃ P Q R : ℚ, (∀ x : ℚ, x ≠ 1 → x ≠ 4 → x ≠ 6 → (x^2 - 4 * x + 8) / ((x - 1) * (x - 4) * (x - 6)) = P / (x - 1) + Q / (x - 4) + R / (x - 6)) ∧ 
  P = 1 / 3 ∧ Q = - 4 / 3 ∧ R = 2 :=
by
  -- Proof is left as a placeholder
  sorry

end NUMINAMATH_GPT_determine_constants_l410_41075


namespace NUMINAMATH_GPT_find_f_3_l410_41044

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation (x : ℝ) : f x + 3 * f (1 - x) = 4 * x ^ 2

theorem find_f_3 : f 3 = 3 / 2 := 
by
  sorry

end NUMINAMATH_GPT_find_f_3_l410_41044


namespace NUMINAMATH_GPT_thirtieth_triangular_number_sum_thirtieth_thirtyfirst_triangular_numbers_l410_41099

def triangular_number (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem thirtieth_triangular_number :
  triangular_number 30 = 465 :=
by
  sorry

theorem sum_thirtieth_thirtyfirst_triangular_numbers :
  triangular_number 30 + triangular_number 31 = 961 :=
by
  sorry

end NUMINAMATH_GPT_thirtieth_triangular_number_sum_thirtieth_thirtyfirst_triangular_numbers_l410_41099


namespace NUMINAMATH_GPT_margo_total_distance_l410_41083

theorem margo_total_distance (time_to_friend : ℝ) (time_back_home : ℝ) (average_rate : ℝ)
  (total_time_hours : ℝ) (total_miles : ℝ) :
  time_to_friend = 12 / 60 ∧
  time_back_home = 24 / 60 ∧
  total_time_hours = (12 / 60) + (24 / 60) ∧
  average_rate = 3 ∧
  total_miles = average_rate * total_time_hours →
  total_miles = 1.8 :=
by
  sorry

end NUMINAMATH_GPT_margo_total_distance_l410_41083


namespace NUMINAMATH_GPT_find_speed_B_l410_41037

def distance_to_location : ℝ := 12
def A_speed_is_1_2_times_B (speed_B speed_A : ℝ) : Prop := speed_A = 1.2 * speed_B
def A_arrives_1_6_hour_earlier (speed_B speed_A : ℝ) : Prop :=
  (distance_to_location / speed_B) - (distance_to_location / speed_A) = 1 / 6

theorem find_speed_B (speed_B : ℝ) (speed_A : ℝ) :
  A_speed_is_1_2_times_B speed_B speed_A →
  A_arrives_1_6_hour_earlier speed_B speed_A →
  speed_B = 12 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_find_speed_B_l410_41037


namespace NUMINAMATH_GPT_percent_of_juniors_involved_in_sports_l410_41026

theorem percent_of_juniors_involved_in_sports
  (total_students : ℕ)
  (percent_juniors : ℝ)
  (juniors_in_sports : ℕ)
  (h1 : total_students = 500)
  (h2 : percent_juniors = 0.40)
  (h3 : juniors_in_sports = 140) :
  (juniors_in_sports : ℝ) / (total_students * percent_juniors) * 100 = 70 := 
by
  -- By conditions h1, h2, h3:
  sorry

end NUMINAMATH_GPT_percent_of_juniors_involved_in_sports_l410_41026


namespace NUMINAMATH_GPT_cylinder_volume_l410_41073

theorem cylinder_volume (r h V: ℝ) (r_pos: r = 4) (lateral_area: 2 * 3.14 * r * h = 62.8) : 
    V = 125600 :=
by
  sorry

end NUMINAMATH_GPT_cylinder_volume_l410_41073


namespace NUMINAMATH_GPT_abc_sum_71_l410_41071

theorem abc_sum_71 (a b c : ℝ) (h₁ : ∀ x, (x ≤ -3 ∨ 23 ≤ x ∧ x < 27) ↔ ( (x - a) * (x - b) / (x - c) ≥ 0)) (h₂ : a < b) : 
  a + 2 * b + 3 * c = 71 :=
sorry

end NUMINAMATH_GPT_abc_sum_71_l410_41071


namespace NUMINAMATH_GPT_Donovan_Mitchell_current_average_l410_41098

theorem Donovan_Mitchell_current_average 
    (points_per_game_goal : ℕ) 
    (games_played : ℕ) 
    (total_games_goal : ℕ) 
    (average_needed_remaining_games : ℕ)
    (points_needed : ℕ) 
    (remaining_games : ℕ) 
    (x : ℕ) 
    (h₁ : games_played = 15) 
    (h₂ : total_games_goal = 20) 
    (h₃ : points_per_game_goal = 30) 
    (h₄ : remaining_games = total_games_goal - games_played)
    (h₅ : average_needed_remaining_games = 42) 
    (h₆ : points_needed = remaining_games * average_needed_remaining_games) 
    (h₇ : points_needed = 210)  
    (h₈ : points_per_game_goal * total_games_goal = 600) 
    (h₉ : games_played * x + points_needed = 600) : 
    x = 26 :=
by {
  sorry
}

end NUMINAMATH_GPT_Donovan_Mitchell_current_average_l410_41098


namespace NUMINAMATH_GPT_jessica_cut_roses_l410_41031

/-- There were 13 roses and 84 orchids in the vase. Jessica cut some more roses and 
orchids from her flower garden. There are now 91 orchids and 14 roses in the vase. 
How many roses did she cut? -/
theorem jessica_cut_roses :
  let initial_roses := 13
  let new_roses := 14
  ∃ cut_roses : ℕ, new_roses = initial_roses + cut_roses ∧ cut_roses = 1 :=
by
  sorry

end NUMINAMATH_GPT_jessica_cut_roses_l410_41031


namespace NUMINAMATH_GPT_quadrilateral_is_trapezoid_or_parallelogram_l410_41002

noncomputable def quadrilateral_property (s1 s2 s3 s4 : ℝ) : Prop :=
  (s1 + s2) * (s3 + s4) = (s1 + s4) * (s2 + s3)

theorem quadrilateral_is_trapezoid_or_parallelogram
  (s1 s2 s3 s4 : ℝ) (h : quadrilateral_property s1 s2 s3 s4) :
  (s1 = s3) ∨ (s2 = s4) ∨ -- Trapezoid conditions
  ∃ (p : ℝ), (p * s1 = s3 * (s1 + s4)) := -- Add necessary conditions to represent a parallelogram
sorry

end NUMINAMATH_GPT_quadrilateral_is_trapezoid_or_parallelogram_l410_41002


namespace NUMINAMATH_GPT_translate_down_three_units_l410_41077

def original_function (x : ℝ) : ℝ := 3 * x + 2

def translated_function (x : ℝ) : ℝ := 3 * x - 1

theorem translate_down_three_units :
  ∀ x : ℝ, translated_function x = original_function x - 3 :=
by
  intro x
  simp [original_function, translated_function]
  sorry

end NUMINAMATH_GPT_translate_down_three_units_l410_41077


namespace NUMINAMATH_GPT_find_integers_l410_41058

theorem find_integers (x y : ℕ) (d : ℕ) (x1 y1 : ℕ) 
  (hx1 : x = d * x1) (hy1 : y = d * y1)
  (hgcd : Nat.gcd x y = d)
  (hcoprime : Nat.gcd x1 y1 = 1)
  (h1 : x1 + y1 = 18)
  (h2 : d * x1 * y1 = 975) : 
  ∃ (x y : ℕ), (Nat.gcd x y > 0) ∧ (x / Nat.gcd x y + y / Nat.gcd x y = 18) ∧ (Nat.lcm x y = 975) :=
sorry

end NUMINAMATH_GPT_find_integers_l410_41058


namespace NUMINAMATH_GPT_fraction_of_remaining_paint_used_l410_41013

theorem fraction_of_remaining_paint_used (total_paint : ℕ) (first_week_fraction : ℚ) (total_used : ℕ) :
  total_paint = 360 ∧ first_week_fraction = 1/6 ∧ total_used = 120 →
  (total_used - first_week_fraction * total_paint) / (total_paint - first_week_fraction * total_paint) = 1/5 :=
  by
    sorry

end NUMINAMATH_GPT_fraction_of_remaining_paint_used_l410_41013


namespace NUMINAMATH_GPT_problem1_problem2_l410_41055

-- Define the propositions
def S (m : ℝ) : Prop := ∃ x : ℝ, m * x^2 + 2 * m * x + 2 - m = 0

def p (m : ℝ) : Prop := 0 < m ∧ m < 2

def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 2 * m * x + 1 > 0

-- Problem (1)
theorem problem1 (m : ℝ) (hS : S m) : m < 0 ∨ 1 ≤ m := sorry

-- Problem (2)
theorem problem2 (m : ℝ) (hpq : p m ∨ q m) (hnq : ¬ q m) : 1 ≤ m ∧ m < 2 := sorry

end NUMINAMATH_GPT_problem1_problem2_l410_41055


namespace NUMINAMATH_GPT_product_of_sequence_l410_41036

theorem product_of_sequence :
  (1 + 1 / 1) * (1 + 1 / 2) * (1 + 1 / 3) * (1 + 1 / 4) * (1 + 1 / 5) *
  (1 + 1 / 6) * (1 + 1 / 7) * (1 + 1 / 8) = 9 :=
by sorry

end NUMINAMATH_GPT_product_of_sequence_l410_41036


namespace NUMINAMATH_GPT_hannahs_adblock_not_block_l410_41090

theorem hannahs_adblock_not_block (x : ℝ) (h1 : 0.8 * x = 0.16) : x = 0.2 :=
by {
  sorry
}

end NUMINAMATH_GPT_hannahs_adblock_not_block_l410_41090


namespace NUMINAMATH_GPT_area_of_hexagon_l410_41080

-- Definitions of the angles and side lengths
def angle_A := 120
def angle_B := 120
def angle_C := 120
def angle_D := 150

def FA := 2
def AB := 2
def BC := 2
def CD := 3
def DE := 3
def EF := 3

-- Theorem statement for the area of hexagon ABCDEF
theorem area_of_hexagon : 
  (angle_A = 120 ∧ angle_B = 120 ∧ angle_C = 120 ∧ angle_D = 150 ∧
   FA = 2 ∧ AB = 2 ∧ BC = 2 ∧ CD = 3 ∧ DE = 3 ∧ EF = 3) →
  (∃ area : ℝ, area = 7.5 * Real.sqrt 3) :=
by
  sorry

end NUMINAMATH_GPT_area_of_hexagon_l410_41080


namespace NUMINAMATH_GPT_cost_prices_max_profit_find_m_l410_41057

-- Part 1
theorem cost_prices (x y: ℕ) (h1 : 40 * x + 30 * y = 5000) (h2 : 10 * x + 50 * y = 3800) : 
  x = 80 ∧ y = 60 :=
sorry

-- Part 2
theorem max_profit (a: ℕ) (h1 : 70 ≤ a ∧ a ≤ 75) : 
  (20 * a + 6000) ≤ 7500 :=
sorry

-- Part 3
theorem find_m (m : ℝ) (h1 : 4 < m ∧ m < 8) (h2 : (20 - 5 * m) * 70 + 6000 = 5720) : 
  m = 4.8 :=
sorry

end NUMINAMATH_GPT_cost_prices_max_profit_find_m_l410_41057


namespace NUMINAMATH_GPT_slope_angle_of_line_x_equal_one_l410_41095

noncomputable def slope_angle_of_vertical_line : ℝ := 90

theorem slope_angle_of_line_x_equal_one : slope_angle_of_vertical_line = 90 := by
  sorry

end NUMINAMATH_GPT_slope_angle_of_line_x_equal_one_l410_41095


namespace NUMINAMATH_GPT_parabola_through_point_l410_41016

theorem parabola_through_point (x y : ℝ) (hx : x = 2) (hy : y = 4) : 
  (∃ a : ℝ, y^2 = a * x ∧ a = 8) ∨ (∃ b : ℝ, x^2 = b * y ∧ b = 1) :=
sorry

end NUMINAMATH_GPT_parabola_through_point_l410_41016


namespace NUMINAMATH_GPT_calculate_expression_l410_41079

theorem calculate_expression :
  ((7 / 9) - (5 / 6) + (5 / 18)) * 18 = 4 :=
by
  -- proof to be filled in later.
  sorry

end NUMINAMATH_GPT_calculate_expression_l410_41079


namespace NUMINAMATH_GPT_minimum_value_of_expression_l410_41093

theorem minimum_value_of_expression {x : ℝ} (hx : x > 0) : (2 / x + x / 2) ≥ 2 :=
by sorry

end NUMINAMATH_GPT_minimum_value_of_expression_l410_41093


namespace NUMINAMATH_GPT_A_investment_l410_41010

-- Conditions as definitions
def B_investment := 72000
def C_investment := 81000
def C_profit := 36000
def Total_profit := 80000

-- Statement to prove
theorem A_investment : 
  ∃ (x : ℕ), x = 27000 ∧
  (C_profit / Total_profit = (9 : ℕ) / 20) ∧
  (C_investment / (x + B_investment + C_investment) = (9 : ℕ) / 20) :=
by sorry

end NUMINAMATH_GPT_A_investment_l410_41010


namespace NUMINAMATH_GPT_half_angle_quadrant_l410_41032

theorem half_angle_quadrant (k : ℤ) (α : ℝ) (h : 2 * k * π + π < α ∧ α < 2 * k * π + (3 * π / 2)) : 
  (∃ j : ℤ, j * π + (π / 2) < (α / 2) ∧ (α / 2) < j * π + (3 * π / 4)) :=
  by sorry

end NUMINAMATH_GPT_half_angle_quadrant_l410_41032


namespace NUMINAMATH_GPT_percentage_increase_of_base_l410_41024

theorem percentage_increase_of_base
  (h b : ℝ) -- Original height and base
  (h_new : ℝ) -- New height
  (b_new : ℝ) -- New base
  (A_original A_new : ℝ) -- Original and new areas
  (p : ℝ) -- Percentage increase in the base
  (h_new_def : h_new = 0.60 * h)
  (b_new_def : b_new = b * (1 + p / 100))
  (A_original_def : A_original = 0.5 * b * h)
  (A_new_def : A_new = 0.5 * b_new * h_new)
  (area_decrease : A_new = 0.84 * A_original) :
  p = 40 := by
  sorry

end NUMINAMATH_GPT_percentage_increase_of_base_l410_41024


namespace NUMINAMATH_GPT_problem_l410_41029

theorem problem (a b : ℝ)
  (h : ∀ x : ℝ, ax^2 + bx + 2 > 0 ↔ (x < -1/2 ∨ x > 1/3)) : 
  a + b = -14 :=
sorry

end NUMINAMATH_GPT_problem_l410_41029


namespace NUMINAMATH_GPT_min_possible_value_of_box_l410_41017

theorem min_possible_value_of_box
  (c d : ℤ)
  (distinct : c ≠ d)
  (h_cd : c * d = 29) :
  ∃ (box : ℤ), c^2 + d^2 = box ∧ box = 842 :=
by
  sorry

end NUMINAMATH_GPT_min_possible_value_of_box_l410_41017


namespace NUMINAMATH_GPT_no_such_convex_polyhedron_exists_l410_41014

-- Definitions of convex polyhedron and the properties related to its faces and vertices.
structure ConvexPolyhedron where
  V : ℕ  -- Number of vertices
  E : ℕ  -- Number of edges
  F : ℕ  -- Number of faces
  -- Additional properties and constraints can be added if necessary

-- Definition that captures the condition where each face has more than 5 sides.
def each_face_has_more_than_five_sides (P : ConvexPolyhedron) : Prop :=
  ∀ f, f > 5 -- Simplified assumption

-- Definition that captures the condition where more than five edges meet at each vertex.
def more_than_five_edges_meet_each_vertex (P : ConvexPolyhedron) : Prop :=
  ∀ v, v > 5 -- Simplified assumption

-- The statement to be proven
theorem no_such_convex_polyhedron_exists :
  ¬ ∃ (P : ConvexPolyhedron), (each_face_has_more_than_five_sides P) ∨ (more_than_five_edges_meet_each_vertex P) := by
  -- Proof of this theorem is omitted with "sorry"
  sorry

end NUMINAMATH_GPT_no_such_convex_polyhedron_exists_l410_41014


namespace NUMINAMATH_GPT_simple_interest_calculation_l410_41088

-- Define the known quantities
def principal : ℕ := 400
def rate_of_interest : ℕ := 15
def time : ℕ := 2

-- Define the formula for simple interest
def simple_interest (P R T : ℕ) : ℕ := (P * R * T) / 100

-- Statement to be proved
theorem simple_interest_calculation :
  simple_interest principal rate_of_interest time = 60 :=
by
  -- This space is used for the proof, We assume the user will complete it
  sorry

end NUMINAMATH_GPT_simple_interest_calculation_l410_41088


namespace NUMINAMATH_GPT_lisa_additional_marbles_l410_41050

theorem lisa_additional_marbles (n : ℕ) (f : ℕ) (m : ℕ) (current_marbles : ℕ) : 
  n = 12 ∧ f = n ∧ m = (n * (n + 1)) / 2 ∧ current_marbles = 34 → 
  m - current_marbles = 44 :=
by
  intros
  sorry

end NUMINAMATH_GPT_lisa_additional_marbles_l410_41050


namespace NUMINAMATH_GPT_positive_difference_of_two_numbers_l410_41041

theorem positive_difference_of_two_numbers
  (x y : ℝ)
  (h₁ : x + y = 10)
  (h₂ : x^2 - y^2 = 24) :
  |x - y| = 12 / 5 :=
sorry

end NUMINAMATH_GPT_positive_difference_of_two_numbers_l410_41041


namespace NUMINAMATH_GPT_original_percent_acid_l410_41065

open Real

variables (a w : ℝ)

theorem original_percent_acid 
  (h1 : (a + 2) / (a + w + 2) = 1 / 4)
  (h2 : (a + 2) / (a + w + 4) = 1 / 5) :
  a / (a + w) = 1 / 5 :=
sorry

end NUMINAMATH_GPT_original_percent_acid_l410_41065


namespace NUMINAMATH_GPT_price_difference_l410_41074

theorem price_difference (P : ℝ) :
  let new_price := 1.20 * P
  let discounted_price := 0.96 * P
  let difference := new_price - discounted_price
  difference = 0.24 * P := by
  let new_price := 1.20 * P
  let discounted_price := 0.96 * P
  let difference := new_price - discounted_price
  sorry

end NUMINAMATH_GPT_price_difference_l410_41074


namespace NUMINAMATH_GPT_rational_sum_of_cubes_l410_41067

theorem rational_sum_of_cubes (t : ℚ) : 
    ∃ (a b c : ℚ), t = (a^3 + b^3 + c^3) :=
by
  sorry

end NUMINAMATH_GPT_rational_sum_of_cubes_l410_41067


namespace NUMINAMATH_GPT_find_a_values_l410_41078

theorem find_a_values (a b x : ℝ) (h₁ : a ≠ b) (h₂ : a^3 - b^3 = 27 * x^3) (h₃ : a - b = 2 * x) :
  a = 3.041 * x ∨ a = -1.041 * x :=
by
  sorry

end NUMINAMATH_GPT_find_a_values_l410_41078


namespace NUMINAMATH_GPT_max_expression_value_l410_41096

theorem max_expression_value (a b c : ℝ) (hb : b > a) (ha : a > c) (hb_ne : b ≠ 0) :
  ∃ M, M = 27 ∧ (∀ a b c, b > a → a > c → b ≠ 0 → (∃ M, (2*a + 3*b)^2 + (b - c)^2 + (2*c - a)^2 ≤ M * b^2) → M ≤ 27) :=
  sorry

end NUMINAMATH_GPT_max_expression_value_l410_41096


namespace NUMINAMATH_GPT_mrs_heine_dogs_treats_l410_41027

theorem mrs_heine_dogs_treats (heart_biscuits_per_dog puppy_boots_per_dog total_items : ℕ)
  (h_biscuits : heart_biscuits_per_dog = 5)
  (h_boots : puppy_boots_per_dog = 1)
  (total : total_items = 12) :
  (total_items / (heart_biscuits_per_dog + puppy_boots_per_dog)) = 2 :=
by
  sorry

end NUMINAMATH_GPT_mrs_heine_dogs_treats_l410_41027


namespace NUMINAMATH_GPT_quadratic_equation_real_roots_l410_41003

theorem quadratic_equation_real_roots (k : ℝ) : 
  (∃ x : ℝ, k * x^2 - 6 * x + 9 = 0) ↔ (k ≤ 1 ∧ k ≠ 0) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_equation_real_roots_l410_41003


namespace NUMINAMATH_GPT_find_number_l410_41011

theorem find_number (x : ℚ) (h : 0.15 * 0.30 * 0.50 * x = 108) : x = 4800 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l410_41011


namespace NUMINAMATH_GPT_solution_system_of_equations_l410_41039

theorem solution_system_of_equations : 
  ∃ (x y : ℝ), (2 * x - y = 3 ∧ x + y = 3) ∧ (x = 2 ∧ y = 1) := 
by
  sorry

end NUMINAMATH_GPT_solution_system_of_equations_l410_41039


namespace NUMINAMATH_GPT_distance_behind_l410_41052

-- Given conditions
variables {A B E : ℝ} -- Speed of Anusha, Banu, and Esha
variables {Da Db De : ℝ} -- distances covered by Anusha, Banu, and Esha

axiom const_speeds : Da = 100 ∧ Db = 90 ∧ Db / Da = De / Db ∧ De = 90 * (Db / 100)

-- The proof to be established
theorem distance_behind (h : Da = 100 ∧ Db = 90 ∧ Db / Da = De / Db ∧ De = 90 * (Db / 100)) :
  100 - De = 19 :=
by sorry

end NUMINAMATH_GPT_distance_behind_l410_41052


namespace NUMINAMATH_GPT_find_square_side_length_l410_41070

open Nat

def original_square_side_length (s : ℕ) : Prop :=
  let length := s + 8
  let breadth := s + 4
  (2 * (length + breadth)) = 40 → s = 4

theorem find_square_side_length (s : ℕ) : original_square_side_length s := by
  sorry

end NUMINAMATH_GPT_find_square_side_length_l410_41070


namespace NUMINAMATH_GPT_exists_min_a_l410_41025

open Real

theorem exists_min_a (x y z : ℝ) : 
  (∃ x y z : ℝ, (sqrt (x - 1) + sqrt (y - 1) + sqrt (z - 1) = (11/2 - 1)) ∧ 
  (sqrt (x + 1) + sqrt (y + 1) + sqrt (z + 1) = (11/2 + 1))) :=
sorry

end NUMINAMATH_GPT_exists_min_a_l410_41025


namespace NUMINAMATH_GPT_acute_triangle_l410_41030

theorem acute_triangle (r R : ℝ) (h : R < r * (Real.sqrt 2 + 1)) : 
  ∃ (α β γ : ℝ), α + β + γ = π ∧ (0 < α) ∧ (0 < β) ∧ (0 < γ) ∧ (α < π / 2) ∧ (β < π / 2) ∧ (γ < π / 2) := 
sorry

end NUMINAMATH_GPT_acute_triangle_l410_41030


namespace NUMINAMATH_GPT_cousins_initial_money_l410_41012

theorem cousins_initial_money (x : ℕ) :
  let Carmela_initial := 7
  let num_cousins := 4
  let gift_each := 1
  Carmela_initial - num_cousins * gift_each = x + gift_each →
  x = 2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_cousins_initial_money_l410_41012


namespace NUMINAMATH_GPT_sum_x_y_z_eq_3_or_7_l410_41059

theorem sum_x_y_z_eq_3_or_7 (x y z : ℝ) (h1 : x + y / z = 2) (h2 : y + z / x = 2) (h3 : z + x / y = 2) : x + y + z = 3 ∨ x + y + z = 7 :=
by
  sorry

end NUMINAMATH_GPT_sum_x_y_z_eq_3_or_7_l410_41059


namespace NUMINAMATH_GPT_system_of_equations_abs_diff_l410_41033

theorem system_of_equations_abs_diff 
  (x y m n : ℝ) 
  (h₁ : 2 * x - y = m)
  (h₂ : x + m * y = n)
  (hx : x = 2)
  (hy : y = 1) : 
  |m - n| = 2 :=
by
  sorry

end NUMINAMATH_GPT_system_of_equations_abs_diff_l410_41033


namespace NUMINAMATH_GPT_solve_for_F_l410_41054

theorem solve_for_F (F C : ℝ) (h₁ : C = 4 / 7 * (F - 40)) (h₂ : C = 25) : F = 83.75 :=
sorry

end NUMINAMATH_GPT_solve_for_F_l410_41054


namespace NUMINAMATH_GPT_solve_for_x_l410_41019

theorem solve_for_x (x : ℝ) (h : 144 / 0.144 = x / 0.0144) : x = 14.4 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l410_41019


namespace NUMINAMATH_GPT_problem_inequality_l410_41076

variable {n : ℕ}
variable (S_n : Finset (Fin n)) (f : Finset (Fin n) → ℝ)

axiom pos_f : ∀ A : Finset (Fin n), 0 < f A
axiom cond_f : ∀ (A : Finset (Fin n)) (x y : Fin n), x ≠ y → f (A ∪ {x}) * f (A ∪ {y}) ≤ f (A ∪ {x, y}) * f A

theorem problem_inequality (A B : Finset (Fin n)) : f A * f B ≤ f (A ∪ B) * f (A ∩ B) := sorry

end NUMINAMATH_GPT_problem_inequality_l410_41076
