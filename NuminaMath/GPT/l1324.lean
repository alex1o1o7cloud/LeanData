import Mathlib

namespace max_soccer_balls_l1324_132437

theorem max_soccer_balls (bought_balls : ℕ) (total_cost : ℕ) (available_money : ℕ) (unit_cost : ℕ)
    (h1 : bought_balls = 6) (h2 : total_cost = 168) (h3 : available_money = 500)
    (h4 : unit_cost = total_cost / bought_balls) :
    (available_money / unit_cost) = 17 := 
by
  sorry

end max_soccer_balls_l1324_132437


namespace smoothie_ratios_l1324_132427

variable (initial_p initial_v m_p m_ratio_p_v: ℕ) (y_p y_v : ℕ)

-- Given conditions
theorem smoothie_ratios (h_initial_p : initial_p = 24) (h_initial_v : initial_v = 25) 
                        (h_m_p : m_p = 20) (h_m_ratio_p_v : m_ratio_p_v = 4)
                        (h_y_p : y_p = initial_p - m_p) (h_y_v : y_v = initial_v - m_p / m_ratio_p_v) :
  (y_p / gcd y_p y_v) = 1 ∧ (y_v / gcd y_p y_v) = 5 :=
by
  sorry

end smoothie_ratios_l1324_132427


namespace business_total_profit_l1324_132479

def total_profit (investmentB periodB profitB : ℝ) (investmentA periodA profitA : ℝ) (investmentC periodC profitC : ℝ) : ℝ :=
    (investmentA * periodA * profitA) + (investmentB * periodB * profitB) + (investmentC * periodC * profitC)

theorem business_total_profit 
    (investmentB periodB profitB : ℝ)
    (investmentA periodA profitA : ℝ)
    (investmentC periodC profitC : ℝ)
    (hA_inv : investmentA = 3 * investmentB)
    (hA_period : periodA = 2 * periodB)
    (hC_inv : investmentC = 2 * investmentB)
    (hC_period : periodC = periodB / 2)
    (hA_rate : profitA = 0.10)
    (hB_rate : profitB = 0.15)
    (hC_rate : profitC = 0.12)
    (hB_profit : investmentB * periodB * profitB = 4000) :
    total_profit investmentB periodB profitB investmentA periodA profitA investmentC periodC profitC = 23200 := 
sorry

end business_total_profit_l1324_132479


namespace vertex_of_parabola_l1324_132469

theorem vertex_of_parabola :
  ∃ (h k : ℝ), (∀ x : ℝ, -2 * (x - h) ^ 2 + k = -2 * (x - 2) ^ 2 - 5) ∧ h = 2 ∧ k = -5 :=
by
  sorry

end vertex_of_parabola_l1324_132469


namespace solutions_of_quadratic_l1324_132434

theorem solutions_of_quadratic (x : ℝ) : x^2 - x = 0 ↔ (x = 0 ∨ x = 1) :=
by
  sorry

end solutions_of_quadratic_l1324_132434


namespace shawn_red_pebbles_l1324_132451

variable (Total : ℕ)
variable (B : ℕ)
variable (Y : ℕ)
variable (P : ℕ)
variable (G : ℕ)

theorem shawn_red_pebbles (h1 : Total = 40)
                          (h2 : B = 13)
                          (h3 : B - Y = 7)
                          (h4 : P = Y)
                          (h5 : G = Y)
                          (h6 : 3 * Y + B = Total)
                          : Total - (B + P + Y + G) = 9 :=
by
 sorry

end shawn_red_pebbles_l1324_132451


namespace plane_through_point_and_line_l1324_132406

noncomputable def plane_equation (x y z : ℝ) : Prop :=
  12 * x + 67 * y + 23 * z - 26 = 0

theorem plane_through_point_and_line :
  ∃ (A B C D : ℤ), 
  (A > 0) ∧ (Int.gcd (abs A) (Int.gcd (abs B) (Int.gcd (abs C) (abs D))) = 1) ∧
  (plane_equation 1 4 (-6)) ∧  
  ∀ t : ℝ, (plane_equation (4 * t + 2)  (-t - 1) (5 * t + 3)) :=
sorry

end plane_through_point_and_line_l1324_132406


namespace correct_calculation_option_D_l1324_132468

theorem correct_calculation_option_D (a : ℝ) : (a ^ 3) ^ 2 = a ^ 6 :=
by sorry

end correct_calculation_option_D_l1324_132468


namespace quadratic_inequality_l1324_132454

theorem quadratic_inequality (a : ℝ) (h : 0 ≤ a ∧ a < 4) : ∀ x : ℝ, a * x^2 - a * x + 1 > 0 :=
by
  sorry

end quadratic_inequality_l1324_132454


namespace B_score_1_probability_correct_A_wins_without_tiebreaker_probability_correct_l1324_132430

def prob_A_solve : ℝ := 0.8
def prob_B_solve : ℝ := 0.75

-- Definitions for A and B scoring in rounds
def prob_B_score_1_point : ℝ := 
  prob_B_solve * (1 - prob_B_solve) + (1 - prob_B_solve) * prob_B_solve

-- Definitions for A winning without a tiebreaker
def prob_A_score_1_point : ℝ :=
  prob_A_solve * (1 - prob_A_solve) + (1 - prob_A_solve) * prob_A_solve

def prob_A_score_2_points : ℝ :=
  prob_A_solve * prob_A_solve

def prob_B_score_0_points : ℝ :=
  (1 - prob_B_solve) * (1 - prob_B_solve)

def prob_B_score_total : ℝ :=
  prob_B_score_1_point

def prob_A_wins_without_tiebreaker : ℝ :=
  prob_A_score_2_points * prob_B_score_1_point +
  prob_A_score_2_points * prob_B_score_0_points +
  prob_A_score_1_point * prob_B_score_0_points

theorem B_score_1_probability_correct :
  prob_B_score_1_point = 3 / 8 := 
by
  sorry

theorem A_wins_without_tiebreaker_probability_correct :
  prob_A_wins_without_tiebreaker = 3 / 10 := 
by 
  sorry

end B_score_1_probability_correct_A_wins_without_tiebreaker_probability_correct_l1324_132430


namespace cube_volume_l1324_132471

theorem cube_volume
  (s : ℝ) 
  (surface_area_eq : 6 * s^2 = 54) :
  s^3 = 27 := 
by 
  sorry

end cube_volume_l1324_132471


namespace hex_prism_paintings_l1324_132476

def num_paintings : ℕ :=
  -- The total number of distinct ways to paint a hex prism according to the conditions
  3 -- Two colors case: white-red, white-blue, red-blue
  + 6 -- Three colors with pattern 121213
  + 1 -- Three colors with identical opposite faces: 123123
  + 3 -- Three colors with non-identical opposite faces: 123213

theorem hex_prism_paintings : num_paintings = 13 := by
  sorry

end hex_prism_paintings_l1324_132476


namespace choir_members_l1324_132455

theorem choir_members (n : ℕ) : 
  (∃ k m : ℤ, n + 4 = 10 * k ∧ n + 5 = 11 * m) ∧ 200 < n ∧ n < 300 → n = 226 :=
by 
  sorry

end choir_members_l1324_132455


namespace first_number_positive_l1324_132403

-- Define the initial condition
def initial_pair : ℕ × ℕ := (1, 1)

-- Define the allowable transformations
def transform1 (x y : ℕ) : Prop :=
(x, y - 1) = initial_pair ∨ (x + y, y + 1) = initial_pair

def transform2 (x y : ℕ) : Prop :=
(x, x * y) = initial_pair ∨ (1 / x, y) = initial_pair

-- Define discriminant function
def discriminant (a b : ℕ) : ℤ := b ^ 2 - 4 * a

-- Define the invariants maintained by the transformations
def invariant (a b : ℕ) : Prop :=
discriminant a b < 0

-- Statement to be proven
theorem first_number_positive :
(∀ (a b : ℕ), invariant a b → a > 0) :=
by
  sorry

end first_number_positive_l1324_132403


namespace fraction_undefined_at_one_l1324_132446

theorem fraction_undefined_at_one (x : ℤ) (h : x = 1) : (x / (x - 1) = 1) := by
  have h : 1 / (1 - 1) = 1 := sorry
  sorry

end fraction_undefined_at_one_l1324_132446


namespace sum_of_smallest_natural_numbers_l1324_132407

-- Define the problem statement
def satisfies_eq (A B : ℕ) := 360 / (A^3 / B) = 5

-- Prove that there exist natural numbers A and B such that 
-- satisfies_eq A B is true, and their sum is 9
theorem sum_of_smallest_natural_numbers :
  ∃ (A B : ℕ), satisfies_eq A B ∧ A + B = 9 :=
by
  -- Sorry is used here to indicate the proof is not given
  sorry

end sum_of_smallest_natural_numbers_l1324_132407


namespace decrease_equation_l1324_132444

-- Define the initial and final average daily written homework time
def initial_time : ℝ := 100
def final_time : ℝ := 70

-- Define the rate of decrease factor
variable (x : ℝ)

-- Define the functional relationship between initial, final time and the decrease factor
def average_homework_time (x : ℝ) : ℝ :=
  initial_time * (1 - x)^2

-- The theorem to be proved statement
theorem decrease_equation :
  average_homework_time x = final_time ↔ 100 * (1 - x) ^ 2 = 70 :=
by
  -- Proof skipped
  sorry

end decrease_equation_l1324_132444


namespace two_times_koi_minus_X_is_64_l1324_132496

-- Definitions based on the conditions
def n : ℕ := 39
def X : ℕ := 14

-- Main proof statement
theorem two_times_koi_minus_X_is_64 : 2 * n - X = 64 :=
by
  sorry

end two_times_koi_minus_X_is_64_l1324_132496


namespace find_f_3_l1324_132413

def f (x : ℝ) (a b : ℝ) : ℝ := x^5 + a*x^3 + b*x - 8

theorem find_f_3 (a b : ℝ) (h : f (-3) a b = 10) : f 3 a b = -26 :=
by sorry

end find_f_3_l1324_132413


namespace value_of_a_l1324_132457

theorem value_of_a (a b : ℝ) (h1 : a + b = 12) (h2 : a * b = 20) (h3 : a > b) (h4 : a - b = 8) : a = 10 := 
by 
sorry

end value_of_a_l1324_132457


namespace number_added_is_10_l1324_132488

theorem number_added_is_10 (x y a : ℕ) (h1 : y = 40) 
  (h2 : x * 4 = 3 * y) 
  (h3 : (x + a) * 5 = 4 * (y + a)) : a = 10 := 
by
  sorry

end number_added_is_10_l1324_132488


namespace y_range_for_conditions_l1324_132443

theorem y_range_for_conditions (y : ℝ) (h1 : y < 0) (h2 : ⌈y⌉ * ⌊y⌋ = 72) : -9 ≤ y ∧ y < -8 :=
sorry

end y_range_for_conditions_l1324_132443


namespace unique_rectangle_l1324_132432

theorem unique_rectangle (a b : ℝ) (h : a < b) :
  ∃! (x y : ℝ), (x < y) ∧ (2 * (x + y) = a + b) ∧ (x * y = (a * b) / 4) := 
sorry

end unique_rectangle_l1324_132432


namespace find_y_given_x_zero_l1324_132416

theorem find_y_given_x_zero (t : ℝ) (y : ℝ) : 
  (3 - 2 * t = 0) → (y = 3 * t + 6) → y = 21 / 2 := 
by 
  sorry

end find_y_given_x_zero_l1324_132416


namespace distance_AO_min_distance_BM_l1324_132411

open Real

-- Definition of rectangular distance
def rectangular_distance (P Q : ℝ × ℝ) : ℝ :=
  abs (P.1 - Q.1) + abs (P.2 - Q.2)

-- Point A and O
def A : ℝ × ℝ := (-1, 3)
def O : ℝ × ℝ := (0, 0)

-- Point B
def B : ℝ × ℝ := (1, 0)

-- Line "x - y + 2 = 0"
def on_line (M : ℝ × ℝ) : Prop :=
  M.1 - M.2 + 2 = 0

-- Proof statement 1: distance from A to O is 4
theorem distance_AO : rectangular_distance A O = 4 := 
sorry

-- Proof statement 2: minimum distance from B to any point on the line is 3
theorem min_distance_BM (M : ℝ × ℝ) (h : on_line M) : rectangular_distance B M = 3 := 
sorry

end distance_AO_min_distance_BM_l1324_132411


namespace cos_double_angle_l1324_132423

theorem cos_double_angle (α : ℝ) (h : Real.cos (α + Real.pi / 2) = 1 / 3) :
  Real.cos (2 * α) = 7 / 9 :=
sorry

end cos_double_angle_l1324_132423


namespace nine_sided_polygon_diagonals_l1324_132429

def num_of_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem nine_sided_polygon_diagonals :
  num_of_diagonals 9 = 27 :=
by
  -- The formula for the number of diagonals in a polygon with n sides is:
  -- num_of_diagonals(n) = (n * (n - 3)) / 2
  
  -- For a nine-sided polygon:
  -- num_of_diagonals(9) = 9 * (9 - 3) / 2
  --                      = 9 * 6 / 2
  --                      = 54 / 2
  --                      = 27
  sorry

end nine_sided_polygon_diagonals_l1324_132429


namespace exponential_first_quadrant_l1324_132470

theorem exponential_first_quadrant (m : ℝ) : 
  (∀ x : ℝ, y = (1 / 2)^x + m → y ≤ 0) ↔ m ≤ -1 := 
by
  sorry

end exponential_first_quadrant_l1324_132470


namespace min_k_squared_floor_l1324_132400

open Nat

theorem min_k_squared_floor (n : ℕ) :
  (∀ k : ℕ, k >= 1 → k^2 + (n / k^2) ≥ 1991) ∧
  (∃ k : ℕ, k >= 1 ∧ k^2 + (n / k^2) < 1992) ↔
  1024 * 967 ≤ n ∧ n ≤ 1024 * 967 + 1023 := 
by
  sorry

end min_k_squared_floor_l1324_132400


namespace total_cost_after_discount_l1324_132475

noncomputable def mango_cost : ℝ := sorry
noncomputable def rice_cost : ℝ := sorry
noncomputable def flour_cost : ℝ := 21

theorem total_cost_after_discount :
  (10 * mango_cost = 24 * rice_cost) →
  (6 * flour_cost = 2 * rice_cost) →
  (flour_cost = 21) →
  (4 * mango_cost + 3 * rice_cost + 5 * flour_cost) * 0.9 = 808.92 :=
by
  intros h1 h2 h3
  -- sorry as placeholder for actual proof
  sorry

end total_cost_after_discount_l1324_132475


namespace find_x_in_coconut_grove_l1324_132473

theorem find_x_in_coconut_grove
  (x : ℕ)
  (h1 : (x + 2) * 30 + x * 120 + (x - 2) * 180 = 300 * x)
  (h2 : 3 * x ≠ 0) :
  x = 10 :=
by
  sorry

end find_x_in_coconut_grove_l1324_132473


namespace smallest_number_with_divisibility_condition_l1324_132445

theorem smallest_number_with_divisibility_condition :
  ∃ x : ℕ, (x + 7) % 24 = 0 ∧ (x + 7) % 36 = 0 ∧ (x + 7) % 50 = 0 ∧ (x + 7) % 56 = 0 ∧ (x + 7) % 81 = 0 ∧ x = 113393 :=
by {
  -- sorry is used to skip the proof.
  sorry
}

end smallest_number_with_divisibility_condition_l1324_132445


namespace combined_dog_years_difference_l1324_132489

theorem combined_dog_years_difference 
  (Max_age : ℕ) 
  (small_breed_rate medium_breed_rate large_breed_rate : ℕ) 
  (Max_turns_age : ℕ) 
  (small_breed_diff medium_breed_diff large_breed_diff combined_diff : ℕ) :
  Max_age = 3 →
  small_breed_rate = 5 →
  medium_breed_rate = 7 →
  large_breed_rate = 9 →
  Max_turns_age = 6 →
  small_breed_diff = small_breed_rate * Max_turns_age - Max_turns_age →
  medium_breed_diff = medium_breed_rate * Max_turns_age - Max_turns_age →
  large_breed_diff = large_breed_rate * Max_turns_age - Max_turns_age →
  combined_diff = small_breed_diff + medium_breed_diff + large_breed_diff →
  combined_diff = 108 :=
by
  intros
  sorry

end combined_dog_years_difference_l1324_132489


namespace isosceles_triangle_base_angle_l1324_132486

/-- In an isosceles triangle, if one angle is 110 degrees, then each base angle measures 35 degrees. -/
theorem isosceles_triangle_base_angle (α β γ : ℝ) (h1 : α + β + γ = 180)
  (h2 : α = β ∨ α = γ ∨ β = γ) (h3 : α = 110 ∨ β = 110 ∨ γ = 110) :
  β = 35 ∨ γ = 35 :=
sorry

end isosceles_triangle_base_angle_l1324_132486


namespace functional_equation_solution_l1324_132472

theorem functional_equation_solution {f : ℝ → ℝ} (h : ∀ x ≠ 1, (x - 1) * f (x + 1) - f x = x) :
    ∀ x, f x = 1 + 2 * x :=
by
  sorry

end functional_equation_solution_l1324_132472


namespace find_A_l1324_132461

theorem find_A : ∃ A : ℕ, 691 - (600 + A * 10 + 7) = 4 ∧ A = 8 := by
  sorry

end find_A_l1324_132461


namespace reach_one_from_45_reach_one_from_345_reach_one_from_any_nat_l1324_132422

theorem reach_one_from_45 : ∃ (n : ℕ), n = 1 :=
by
  -- Start from 45 and follow the given steps to reach 1.
  sorry

theorem reach_one_from_345 : ∃ (n : ℕ), n = 1 :=
by
  -- Start from 345 and follow the given steps to reach 1.
  sorry

theorem reach_one_from_any_nat (n : ℕ) (h : n ≠ 0) : ∃ (k : ℕ), k = 1 :=
by
  -- Prove that starting from any non-zero natural number, you can reach 1.
  sorry

end reach_one_from_45_reach_one_from_345_reach_one_from_any_nat_l1324_132422


namespace fractional_equation_solution_l1324_132402

theorem fractional_equation_solution (x : ℝ) (h : x = 7) : (3 / (x - 3)) - 1 = 1 / (3 - x) := by
  sorry

end fractional_equation_solution_l1324_132402


namespace evaluate_fraction_l1324_132417

theorem evaluate_fraction (a b c : ℝ) (h : a^3 - b^3 + c^3 ≠ 0) :
  (a^6 - b^6 + c^6) / (a^3 - b^3 + c^3) = a^3 + b^3 + c^3 :=
sorry

end evaluate_fraction_l1324_132417


namespace a_b_c_sum_l1324_132448

-- Definitions of the conditions
def f (x : ℝ) (a b c : ℝ) : ℝ := a * x^2 + b * x + c

theorem a_b_c_sum (a b c : ℝ) :
  (∀ x : ℝ, f (x + 4) a b c = 4 * x^2 + 9 * x + 5) ∧ (∀ x : ℝ, f x a b c = a * x^2 + b * x + c) →
  a + b + c = 14 :=
by
  intros h
  sorry

end a_b_c_sum_l1324_132448


namespace find_x_l1324_132466
-- Import all necessary libraries

-- Define the conditions
variables (x : ℝ) (log5x log6x log15x : ℝ)

-- Assume the edge lengths of the prism are logs with different bases
def edge_lengths (x : ℝ) (log5x log6x log15x : ℝ) : Prop :=
  log5x = Real.logb 5 x ∧ log6x = Real.logb 6 x ∧ log15x = Real.logb 15 x

-- Define the ratio of Surface Area to Volume
def ratio_SA_to_V (x : ℝ) (log5x log6x log15x : ℝ) : Prop :=
  let SA := 2 * (log5x * log6x + log5x * log15x + log6x * log15x)
  let V  := log5x * log6x * log15x
  SA / V = 10

-- Prove the value of x
theorem find_x (h1 : edge_lengths x log5x log6x log15x) (h2 : ratio_SA_to_V x log5x log6x log15x) :
  x = Real.rpow 450 (1/5) := 
sorry

end find_x_l1324_132466


namespace customer_difference_l1324_132431

theorem customer_difference (before after : ℕ) (h1 : before = 19) (h2 : after = 4) : before - after = 15 :=
by
  sorry

end customer_difference_l1324_132431


namespace nancy_tortilla_chips_l1324_132401

theorem nancy_tortilla_chips :
  ∀ (total_chips chips_brother chips_herself chips_sister : ℕ),
    total_chips = 22 →
    chips_brother = 7 →
    chips_herself = 10 →
    chips_sister = total_chips - chips_brother - chips_herself →
    chips_sister = 5 :=
by
  intros total_chips chips_brother chips_herself chips_sister
  intro h_total h_brother h_herself h_sister
  rw [h_total, h_brother, h_herself] at h_sister
  simp at h_sister
  assumption

end nancy_tortilla_chips_l1324_132401


namespace parabola_focus_condition_l1324_132458

theorem parabola_focus_condition (m : ℝ) : (∃ (x y : ℝ), x + y - 2 = 0 ∧ y = (1 / (4 * m))) → m = 1 / 8 :=
by
  sorry

end parabola_focus_condition_l1324_132458


namespace fraction_to_decimal_l1324_132428

theorem fraction_to_decimal : (31 : ℝ) / (2 * 5^6) = 0.000992 :=
by sorry

end fraction_to_decimal_l1324_132428


namespace sin_alpha_terminal_point_l1324_132447

theorem sin_alpha_terminal_point :
  let alpha := (2 * Real.cos (120 * (π / 180)), Real.sqrt 2 * Real.sin (225 * (π / 180)))
  α = -π / 4 →
  α.sin = - Real.sqrt 2 / 2
:=
by
  intro α_definition
  sorry

end sin_alpha_terminal_point_l1324_132447


namespace find_x_when_y_is_20_l1324_132449

variable (x y k : ℝ)

axiom constant_ratio : (5 * 4 - 6) / (5 + 20) = k

theorem find_x_when_y_is_20 (h : (5 * x - 6) / (y + 20) = k) (hy : y = 20) : x = 5.68 := by
  sorry

end find_x_when_y_is_20_l1324_132449


namespace find_value_of_c_l1324_132482

theorem find_value_of_c (c : ℝ) (h1 : c > 0) (h2 : c + ⌊c⌋ = 23.2) : c = 11.7 :=
sorry

end find_value_of_c_l1324_132482


namespace minimum_value_f_condition_f_geq_zero_l1324_132490

noncomputable def f (x a : ℝ) := Real.exp x - a * x - 1

theorem minimum_value_f (a : ℝ) (h : 0 < a) : 
  (∀ x : ℝ, f x a ≥ f (Real.log a) a) ∧ f (Real.log a) a = a - a * Real.log a - 1 :=
by 
  sorry

theorem condition_f_geq_zero (a : ℝ) (h : 0 < a) : 
  (∀ x : ℝ, f x a ≥ 0) ↔ a = 1 :=
by 
  sorry

end minimum_value_f_condition_f_geq_zero_l1324_132490


namespace complex_multiplication_l1324_132425

variable (i : ℂ)
axiom imag_unit : i^2 = -1

theorem complex_multiplication : (3 + i) * i = -1 + 3 * i :=
by
  sorry

end complex_multiplication_l1324_132425


namespace salary_percentage_l1324_132426

theorem salary_percentage (m n : ℝ) (P : ℝ) (h1 : m + n = 572) (h2 : n = 260) (h3 : m = (P / 100) * n) : P = 120 := 
by
  sorry

end salary_percentage_l1324_132426


namespace roots_of_quadratic_l1324_132414

theorem roots_of_quadratic (a b : ℝ) (h₁ : a + b = 2) (h₂ : a * b = -3) : a^2 + b^2 = 10 := 
by
  -- proof steps go here, but not required as per the instruction
  sorry

end roots_of_quadratic_l1324_132414


namespace average_and_difference_l1324_132474

theorem average_and_difference
  (x y : ℚ) 
  (h1 : (15 + 24 + x + y) / 4 = 20)
  (h2 : x - y = 6) :
  x = 23.5 ∧ y = 17.5 := by
  sorry

end average_and_difference_l1324_132474


namespace largest_n_for_divisibility_l1324_132404

theorem largest_n_for_divisibility :
  ∃ (n : ℕ), n = 5 ∧ 3^n ∣ (4^27000 - 82) ∧ ¬ 3^(n + 1) ∣ (4^27000 - 82) :=
by
  sorry

end largest_n_for_divisibility_l1324_132404


namespace complementary_angle_measure_l1324_132433

theorem complementary_angle_measure (x : ℝ) (h1 : 0 < x) (h2 : 4*x + x = 90) : 4*x = 72 :=
by
  sorry

end complementary_angle_measure_l1324_132433


namespace min_students_wearing_both_l1324_132498

theorem min_students_wearing_both (n : ℕ) (H1 : n % 3 = 0) (H2 : n % 6 = 0) (H3 : n = 6) :
  ∃ x : ℕ, x = 1 ∧ 
           (∃ b : ℕ, b = n / 3) ∧
           (∃ r : ℕ, r = 5 * n / 6) ∧
           6 = b + r - x :=
by sorry

end min_students_wearing_both_l1324_132498


namespace find_y_rotation_l1324_132421

def rotate_counterclockwise (A : Point) (B : Point) (θ : ℝ) : Point := sorry
def rotate_clockwise (A : Point) (B : Point) (θ : ℝ) : Point := sorry

variable {A B C : Point}
variable {y : ℝ}

theorem find_y_rotation
  (h1 : rotate_counterclockwise A B 450 = C)
  (h2 : rotate_clockwise A B y = C)
  (h3 : y < 360) :
  y = 270 :=
sorry

end find_y_rotation_l1324_132421


namespace quadratic_two_distinct_real_roots_l1324_132465

theorem quadratic_two_distinct_real_roots (m : ℝ) : 
  ∀ x : ℝ, x^2 + m * x - 2 = 0 → ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x - x₁) * (x - x₂) = 0 :=
by
  sorry

end quadratic_two_distinct_real_roots_l1324_132465


namespace find_finleys_age_l1324_132435

-- Definitions for given problem
def rogers_age (J A : ℕ) := (J + A) / 2
def alex_age (F : ℕ) := 3 * (F + 10) - 5

-- Given conditions
def jills_age : ℕ := 20
def in_15_years_age_difference (R J F : ℕ) := R + 15 - (J + 15) = F - 30
def rogers_age_twice_jill_plus_five (J : ℕ) := 2 * J + 5

-- Theorem stating the problem assertion
theorem find_finleys_age (F : ℕ) :
  rogers_age jills_age (alex_age F) = rogers_age_twice_jill_plus_five jills_age ∧ 
  in_15_years_age_difference (rogers_age jills_age (alex_age F)) jills_age F →
  F = 15 :=
by
  sorry

end find_finleys_age_l1324_132435


namespace train_crosses_pole_in_2point4_seconds_l1324_132419

noncomputable def time_to_cross (length : ℝ) (speed_kmh : ℝ) : ℝ :=
  length / (speed_kmh * (5/18))

theorem train_crosses_pole_in_2point4_seconds :
  time_to_cross 120 180 = 2.4 := by
  sorry

end train_crosses_pole_in_2point4_seconds_l1324_132419


namespace discount_per_person_correct_l1324_132459

noncomputable def price_per_person : ℕ := 147
noncomputable def total_people : ℕ := 2
noncomputable def total_cost_with_discount : ℕ := 266

theorem discount_per_person_correct :
  let total_cost_without_discount := price_per_person * total_people
  let total_discount := total_cost_without_discount - total_cost_with_discount
  let discount_per_person := total_discount / total_people
  discount_per_person = 14 := by
  sorry

end discount_per_person_correct_l1324_132459


namespace tunnel_depth_l1324_132492

theorem tunnel_depth (topWidth : ℝ) (bottomWidth : ℝ) (area : ℝ) (h : ℝ)
  (h1 : topWidth = 15)
  (h2 : bottomWidth = 5)
  (h3 : area = 400)
  (h4 : area = (1 / 2) * (topWidth + bottomWidth) * h) :
  h = 40 := 
sorry

end tunnel_depth_l1324_132492


namespace translated_parabola_expression_correct_l1324_132410

-- Definitions based on the conditions
def original_parabola (x : ℝ) : ℝ := x^2 - 1
def translated_parabola (x : ℝ) : ℝ := (x + 2)^2

-- The theorem to prove
theorem translated_parabola_expression_correct :
  ∀ x : ℝ, translated_parabola x = original_parabola (x + 2) + 1 :=
by
  sorry

end translated_parabola_expression_correct_l1324_132410


namespace dodgeballs_purchasable_l1324_132438

-- Definitions for the given conditions
def original_budget (B : ℝ) := B
def new_budget (B : ℝ) := 1.2 * B
def cost_per_dodgeball : ℝ := 5
def cost_per_softball : ℝ := 9
def softballs_purchased (B : ℝ) := 10

-- Theorem statement
theorem dodgeballs_purchasable {B : ℝ} (h : new_budget B = 90) : original_budget B / cost_per_dodgeball = 15 := 
by 
  sorry

end dodgeballs_purchasable_l1324_132438


namespace total_cost_of_dresses_l1324_132493

-- Define the costs of each dress
variables (patty_cost ida_cost jean_cost pauline_cost total_cost : ℕ)

-- Given conditions
axiom pauline_cost_is_30 : pauline_cost = 30
axiom jean_cost_is_10_less_than_pauline : jean_cost = pauline_cost - 10
axiom ida_cost_is_30_more_than_jean : ida_cost = jean_cost + 30
axiom patty_cost_is_10_more_than_ida : patty_cost = ida_cost + 10

-- Statement to prove total cost
theorem total_cost_of_dresses : total_cost = pauline_cost + jean_cost + ida_cost + patty_cost 
                                 → total_cost = 160 :=
by {
  -- Proof is left as an exercise
  sorry
}

end total_cost_of_dresses_l1324_132493


namespace expand_expression_l1324_132464

theorem expand_expression (x : ℝ) : (17 * x^2 + 20) * 3 * x^3 = 51 * x^5 + 60 * x^3 := 
by
  sorry

end expand_expression_l1324_132464


namespace arithmetic_sequence_sum_l1324_132487

variable (S : ℕ → ℕ)   -- S is a function that gives the sum of the first k*n terms

theorem arithmetic_sequence_sum
  (n : ℕ)
  (h1 : S n = 45)
  (h2 : S (2 * n) = 60) :
  S (3 * n) = 65 := sorry

end arithmetic_sequence_sum_l1324_132487


namespace max_m_n_sq_l1324_132494

theorem max_m_n_sq (m n : ℕ) (hm : 1 ≤ m ∧ m ≤ 1981) (hn : 1 ≤ n ∧ n ≤ 1981)
  (h : (n^2 - m * n - m^2)^2 = 1) : m^2 + n^2 ≤ 3524578 :=
sorry

end max_m_n_sq_l1324_132494


namespace solve_abc_l1324_132441

def f (x a b c : ℤ) : ℤ := x^3 + a*x^2 + b*x + c

theorem solve_abc (a b c : ℤ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a) 
  (h_fa : f a a b c = a^3) (h_fb : f b b a c = b^3) : 
  a = -2 ∧ b = 4 ∧ c = 16 := 
sorry

end solve_abc_l1324_132441


namespace g_value_at_100_l1324_132456

-- Given function g and its property
theorem g_value_at_100 (g : ℝ → ℝ) (h : ∀ x y : ℝ, 0 < x → 0 < y →
  x * g y - y * g x = g (x^2 / y)) : g 100 = 0 :=
sorry

end g_value_at_100_l1324_132456


namespace ratio_of_savings_to_earnings_l1324_132485

-- Definitions based on the given conditions
def earnings_washing_cars : ℤ := 20
def earnings_walking_dogs : ℤ := 40
def total_savings : ℤ := 150
def months : ℤ := 5

-- Statement to prove the ratio of savings per month to total earnings per month
theorem ratio_of_savings_to_earnings :
  (total_savings / months) = (earnings_washing_cars + earnings_walking_dogs) / 2 := by
  sorry

end ratio_of_savings_to_earnings_l1324_132485


namespace three_pow_gt_pow_three_for_n_ne_3_l1324_132480

theorem three_pow_gt_pow_three_for_n_ne_3 (n : ℕ) (h : n ≠ 3) : 3^n > n^3 :=
sorry

end three_pow_gt_pow_three_for_n_ne_3_l1324_132480


namespace coprime_integers_exist_l1324_132495

theorem coprime_integers_exist (a b c : ℚ) (t : ℤ) (h1 : a + b + c = t) (h2 : a^2 + b^2 + c^2 = t) (h3 : t ≥ 0) : 
  ∃ (u v : ℤ), Int.gcd u v = 1 ∧ abc = (u^2 : ℚ) / (v^3 : ℚ) :=
by sorry

end coprime_integers_exist_l1324_132495


namespace inverse_function_correct_l1324_132467

noncomputable def inverse_function (y : ℝ) : ℝ := (1 / 2) * y - (3 / 2)

theorem inverse_function_correct :
  ∀ x ∈ Set.Icc (0 : ℝ) (5 : ℝ), (inverse_function (2 * x + 3) = x) ∧ (0 ≤ 2 * x + 3) ∧ (2 * x + 3 ≤ 5) :=
by
  sorry

end inverse_function_correct_l1324_132467


namespace companion_sets_count_l1324_132477

def companion_set (A : Set ℝ) : Prop :=
  ∀ x ∈ A, (x ≠ 0) → (1 / x) ∈ A

def M : Set ℝ := { -1, 0, 1/2, 2, 3 }

theorem companion_sets_count : 
  ∃ S : Finset (Set ℝ), (∀ A ∈ S, companion_set A) ∧ (∀ A ∈ S, A ⊆ M) ∧ S.card = 3 := 
by
  sorry

end companion_sets_count_l1324_132477


namespace total_pages_in_book_l1324_132460

def pages_already_read : ℕ := 147
def pages_left_to_read : ℕ := 416

theorem total_pages_in_book : pages_already_read + pages_left_to_read = 563 := by
  sorry

end total_pages_in_book_l1324_132460


namespace simplify_expression_l1324_132409

theorem simplify_expression :
  (2 + Real.sqrt 3)^2 - Real.sqrt 18 * Real.sqrt (2 / 3) = 7 + 2 * Real.sqrt 3 :=
by
  sorry

end simplify_expression_l1324_132409


namespace clara_quarters_l1324_132463

theorem clara_quarters :
  ∃ q : ℕ, 8 < q ∧ q < 80 ∧ q % 3 = 1 ∧ q % 4 = 1 ∧ q % 5 = 1 ∧ q = 61 :=
by
  sorry

end clara_quarters_l1324_132463


namespace evaluate_expression_l1324_132452

theorem evaluate_expression : 6^3 - 4 * 6^2 + 4 * 6 - 1 = 95 :=
by
  sorry

end evaluate_expression_l1324_132452


namespace cube_volume_l1324_132405

theorem cube_volume (d : ℝ) (s : ℝ) (h : d = 3 * Real.sqrt 3) (h_s : s * Real.sqrt 3 = d) : s ^ 3 = 27 := by
  -- Assuming h: the formula for the given space diagonal
  -- Assuming h_s: the formula connecting side length and the space diagonal
  sorry

end cube_volume_l1324_132405


namespace combined_mean_of_scores_l1324_132442

theorem combined_mean_of_scores (f s : ℕ) (mean_1 mean_2 : ℕ) (ratio : f = (2 * s) / 3) 
  (hmean1 : mean_1 = 90) (hmean2 : mean_2 = 75) :
  (135 * s) / ((2 * s) / 3 + s) = 81 := 
by
  sorry

end combined_mean_of_scores_l1324_132442


namespace find_multiple_of_A_l1324_132499

def shares_division_problem (A B C : ℝ) (x : ℝ) : Prop :=
  C = 160 ∧
  x * A = 5 * B ∧
  x * A = 10 * C ∧
  A + B + C = 880

theorem find_multiple_of_A (A B C x : ℝ) (h : shares_division_problem A B C x) : x = 4 :=
by sorry

end find_multiple_of_A_l1324_132499


namespace problem_1_problem_2_l1324_132450

noncomputable def poly_expansion : (2 * x - 3) ^ 4 = a₀ + a₁ * x + a₂ * x ^ 2 + a₃ * x ^ 3 + a₄ * x ^ 4 := sorry

theorem problem_1 (poly_expansion : (2 * x - 3) ^ 4 = a₀ + a₁ * x + a₂ * x ^ 2 + a₃ * x ^ 3 + a₄ * x ^ 4) :
  a₁ + a₂ + a₃ + a₄ = -80 :=
sorry

theorem problem_2 (poly_expansion : (2 * x - 3) ^ 4 = a₀ + a₁ * x + a₂ * x ^ 2 + a₃ * x ^ 3 + a₄ * x ^ 4) :
  (a₀ + a₂ + a₄) ^ 2 - (a₁ + a₃) ^ 2 = 625 :=
sorry

end problem_1_problem_2_l1324_132450


namespace initial_books_correct_l1324_132408

def sold_books : ℕ := 78
def left_books : ℕ := 37
def initial_books : ℕ := sold_books + left_books

theorem initial_books_correct : initial_books = 115 := by
  sorry

end initial_books_correct_l1324_132408


namespace find_m_l1324_132483

-- Definition of vector
def vector (α : Type*) := α × α

-- Two vectors are collinear and have the same direction
def collinear_and_same_direction (a b : vector ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ a = (k * b.1, k * b.2)

-- The vectors a and b
def a (m : ℝ) : vector ℝ := (m, 1)
def b (m : ℝ) : vector ℝ := (4, m)

-- The theorem we want to prove
theorem find_m (m : ℝ) (h1 : collinear_and_same_direction (a m) (b m)) : m = 2 :=
  sorry

end find_m_l1324_132483


namespace find_ab_l1324_132412

theorem find_ab (a b : ℝ) (h₁ : a - b = 3) (h₂ : a^2 + b^2 = 29) : a * b = 10 :=
sorry

end find_ab_l1324_132412


namespace imaginary_unit_problem_l1324_132491

variable {a b : ℝ}

theorem imaginary_unit_problem (h : i * (a + i) = b + 2 * i) : a + b = 1 :=
sorry

end imaginary_unit_problem_l1324_132491


namespace kaleb_non_working_games_l1324_132415

theorem kaleb_non_working_games (total_games working_game_price earning : ℕ) (h1 : total_games = 10) (h2 : working_game_price = 6) (h3 : earning = 12) :
  total_games - (earning / working_game_price) = 8 :=
by
  sorry

end kaleb_non_working_games_l1324_132415


namespace polynomial_evaluation_l1324_132436

noncomputable def x : ℝ :=
  (3 + 3 * Real.sqrt 5) / 2

theorem polynomial_evaluation :
  (x^2 - 3 * x - 9 = 0) → (x^3 - 3 * x^2 - 9 * x + 7 = 7) :=
by
  intros h
  sorry

end polynomial_evaluation_l1324_132436


namespace function_through_point_l1324_132424

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := a^x

theorem function_through_point (a : ℝ) (x : ℝ) (hx : (2 : ℝ) = x) (h : f 2 a = 4) : f x 2 = 2^x :=
by sorry

end function_through_point_l1324_132424


namespace initial_amount_l1324_132439

theorem initial_amount (X : ℝ) (h : 0.7 * X = 3500) : X = 5000 :=
by
  sorry

end initial_amount_l1324_132439


namespace find_circle_center_l1324_132418

noncomputable def circle_center : (ℝ × ℝ) :=
  let x_center := 5
  let y_center := 4
  (x_center, y_center)

theorem find_circle_center (x y : ℝ) (h : x^2 - 10 * x + y^2 - 8 * y = 16) :
  circle_center = (5, 4) := by
  sorry

end find_circle_center_l1324_132418


namespace plant_initial_mass_l1324_132453

theorem plant_initial_mass (x : ℕ) :
  (27 * x + 52 = 133) → x = 3 :=
by
  intro h
  sorry

end plant_initial_mass_l1324_132453


namespace find_m_l1324_132462

def A := {x : ℝ | x^2 - 3 * x + 2 = 0}
def C (m : ℝ) := {x : ℝ | x^2 - m * x + 2 = 0}

theorem find_m (m : ℝ) (h : A ∩ C m = C m) : 
  m = 3 ∨ (-2 * Real.sqrt 2 < m ∧ m < 2 * Real.sqrt 2) :=
by sorry

end find_m_l1324_132462


namespace integral_of_x_squared_l1324_132481

-- Define the conditions
noncomputable def constant_term : ℝ := 3

-- Define the main theorem we want to prove
theorem integral_of_x_squared : ∫ (x : ℝ) in (1 : ℝ)..constant_term, x^2 = 26 / 3 := 
by 
  sorry

end integral_of_x_squared_l1324_132481


namespace aunt_may_milk_leftover_l1324_132484

noncomputable def milk_leftover : Real :=
let morning_milk := 5 * 13 + 4 * 0.5 + 10 * 0.25
let evening_milk := 5 * 14 + 4 * 0.6 + 10 * 0.2

let morning_spoiled := morning_milk * 0.1
let cheese_produced := morning_milk * 0.15
let remaining_morning_milk := morning_milk - morning_spoiled - cheese_produced
let ice_cream_sale := remaining_morning_milk * 0.7

let evening_spoiled := evening_milk * 0.05
let remaining_evening_milk := evening_milk - evening_spoiled
let cheese_shop_sale := remaining_evening_milk * 0.8

let leftover_previous_day := 15
let remaining_morning_after_sale := remaining_morning_milk - ice_cream_sale
let remaining_evening_after_sale := remaining_evening_milk - cheese_shop_sale

leftover_previous_day + remaining_morning_after_sale + remaining_evening_after_sale

theorem aunt_may_milk_leftover : 
  milk_leftover = 44.7735 := 
sorry

end aunt_may_milk_leftover_l1324_132484


namespace find_constants_l1324_132478

-- Definitions based on the given problem
def inequality_in_x (a : ℝ) (x : ℝ) : Prop :=
  a * x^2 - 3 * x + 2 > 0

def roots_eq (a : ℝ) (r1 r2 : ℝ) : Prop :=
  a * r1^2 - 3 * r1 + 2 = 0 ∧ a * r2^2 - 3 * r2 + 2 = 0

def solution_set (a b : ℝ) (x : ℝ) : Prop :=
  x < 1 ∨ x > b

-- Problem statement: given conditions find a and b
theorem find_constants (a b : ℝ) (h1 : 1 < b) (h2 : 0 < a) :
  roots_eq a 1 b ∧ solution_set a b 1 ∧ solution_set a b b :=
sorry

end find_constants_l1324_132478


namespace carton_weight_l1324_132440

theorem carton_weight :
  ∀ (x : ℝ),
  (12 * 4 + 16 * x = 96) → 
  x = 3 :=
by
  intros x h
  sorry

end carton_weight_l1324_132440


namespace max_pies_without_ingredients_l1324_132497

theorem max_pies_without_ingredients :
  let total_pies := 36
  let chocolate_pies := total_pies / 3
  let marshmallow_pies := total_pies / 4
  let cayenne_pies := total_pies / 2
  let soy_nuts_pies := total_pies / 8
  let max_ingredient_pies := max (max chocolate_pies marshmallow_pies) (max cayenne_pies soy_nuts_pies)
  total_pies - max_ingredient_pies = 18 :=
by
  sorry

end max_pies_without_ingredients_l1324_132497


namespace initial_investment_l1324_132420

theorem initial_investment (A r : ℝ) (n : ℕ) (P : ℝ) (hA : A = 630.25) (hr : r = 0.12) (hn : n = 5) :
  A = P * (1 + r) ^ n → P = 357.53 :=
by
  sorry

end initial_investment_l1324_132420
