import Mathlib

namespace find_f_2023_4_l1600_160088

noncomputable def f : ℕ → ℝ → ℝ
| 0, x => 2 * Real.sqrt x
| (n + 1), x => 4 / (2 - f n x)

theorem find_f_2023_4 : f 2023 4 = -2 := sorry

end find_f_2023_4_l1600_160088


namespace graph_through_origin_y_increases_with_x_not_pass_third_quadrant_l1600_160081

-- Part 1: Prove that if the graph passes through the origin, then m ≠ 2/3 and n = 1
theorem graph_through_origin {m n : ℝ} : 
  (3 * m - 2 ≠ 0) → (1 - n = 0) ↔ (m ≠ 2/3 ∧ n = 1) :=
by sorry

-- Part 2: Prove that if y increases as x increases, then m > 2/3 and n is any real number
theorem y_increases_with_x {m n : ℝ} : 
  (3 * m - 2 > 0) ↔ (m > 2/3 ∧ ∀ n : ℝ, True) :=
by sorry

-- Part 3: Prove that if the graph does not pass through the third quadrant, then m < 2/3 and n ≤ 1
theorem not_pass_third_quadrant {m n : ℝ} : 
  (3 * m - 2 < 0) ∧ (1 - n ≥ 0) ↔ (m < 2/3 ∧ n ≤ 1) :=
by sorry

end graph_through_origin_y_increases_with_x_not_pass_third_quadrant_l1600_160081


namespace range_of_a_l1600_160036

theorem range_of_a (a : ℝ) (h : ∀ x, x > a → 2 * x + 2 / (x - a) ≥ 5) : a ≥ 1 / 2 :=
sorry

end range_of_a_l1600_160036


namespace coupon_savings_inequalities_l1600_160053

variable {P : ℝ} (p : ℝ) (hP : P = 150 + p) (hp_pos : p > 0)
variable (ha : 0.15 * P > 30) (hb : 0.15 * P > 0.20 * p)
variable (cA_saving : ℝ := 0.15 * P)
variable (cB_saving : ℝ := 30)
variable (cC_saving : ℝ := 0.20 * p)

theorem coupon_savings_inequalities (h1 : 0.15 * P - 30 > 0) (h2 : 0.15 * P - 0.20 * (P - 150) > 0) :
  let x := 200
  let y := 600
  y - x = 400 :=
by
  sorry

end coupon_savings_inequalities_l1600_160053


namespace intersection_A_complementB_l1600_160037

open Set

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {2, 4, 6}
def B : Set ℕ := {1, 3, 5, 7}
def complementB := U \ B

theorem intersection_A_complementB :
  A ∩ complementB = {2, 4, 6} := 
by
  sorry

end intersection_A_complementB_l1600_160037


namespace distance_to_mothers_house_l1600_160095

theorem distance_to_mothers_house 
  (D : ℝ) 
  (h1 : (2 / 3) * D = 156.0) : 
  D = 234.0 := 
sorry

end distance_to_mothers_house_l1600_160095


namespace conference_center_capacity_l1600_160060

theorem conference_center_capacity (n_rooms : ℕ) (fraction_full : ℚ) (current_people : ℕ) (full_capacity : ℕ) (people_per_room : ℕ) 
  (h1 : n_rooms = 6) (h2 : fraction_full = 2/3) (h3 : current_people = 320) (h4 : current_people = fraction_full * full_capacity) 
  (h5 : people_per_room = full_capacity / n_rooms) : people_per_room = 80 :=
by
  -- The proof will go here.
  sorry

end conference_center_capacity_l1600_160060


namespace retirement_total_correct_l1600_160038

-- Definitions of the conditions
def hire_year : Nat := 1986
def hire_age : Nat := 30
def retirement_year : Nat := 2006

-- Calculation of age and years of employment at retirement
def employment_duration : Nat := retirement_year - hire_year
def age_at_retirement : Nat := hire_age + employment_duration

-- The required total of age and years of employment for retirement
def total_required_for_retirement : Nat := age_at_retirement + employment_duration

-- The theorem to be proven
theorem retirement_total_correct :
  total_required_for_retirement = 70 :=
  by 
  sorry

end retirement_total_correct_l1600_160038


namespace triangular_number_30_eq_465_perimeter_dots_30_eq_88_l1600_160027

-- Definition of the 30th triangular number
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

-- Definition of the perimeter dots for the triangular number
def perimeter_dots (n : ℕ) : ℕ := n + 2 * (n - 1)

-- Theorem to prove the 30th triangular number is 465
theorem triangular_number_30_eq_465 : triangular_number 30 = 465 := by
  sorry

-- Theorem to prove the perimeter dots for the 30th triangular number is 88
theorem perimeter_dots_30_eq_88 : perimeter_dots 30 = 88 := by
  sorry

end triangular_number_30_eq_465_perimeter_dots_30_eq_88_l1600_160027


namespace percent_less_l1600_160010

theorem percent_less (w u y z : ℝ) (P : ℝ) (hP : P = 0.40)
  (h1 : u = 0.60 * y)
  (h2 : z = 0.54 * y)
  (h3 : z = 1.50 * w) :
  w = (1 - P) * u := 
sorry

end percent_less_l1600_160010


namespace exponent_sum_equality_l1600_160072

theorem exponent_sum_equality {a : ℕ} (h1 : 2^12 + 1 = 17 * a) (h2: a = 2^8 + 2^7 + 2^6 + 2^5 + 2^0) : 
  ∃ a1 a2 a3 a4 a5 : ℕ, 
    a1 < a2 ∧ a2 < a3 ∧ a3 < a4 ∧ a4 < a5 ∧ 
    2^a1 + 2^a2 + 2^a3 + 2^a4 + 2^a5 = a ∧ 
    a1 = 0 ∧ a2 = 5 ∧ a3 = 6 ∧ a4 = 7 ∧ a5 = 8 ∧ 
    5 = 5 :=
by {
  sorry
}

end exponent_sum_equality_l1600_160072


namespace range_of_a_l1600_160059

theorem range_of_a (a : ℝ) (A : Set ℝ) (h : A = {x | a * x^2 - 3 * x + 1 = 0} ∧ ∃ (n : ℕ), 2 ^ n - 1 = 3) :
  a ∈ Set.Ioo (-(1:ℝ)/0) 0 ∪ Set.Ioo 0 (9 / 4) :=
sorry

end range_of_a_l1600_160059


namespace no_injective_function_satisfying_conditions_l1600_160044

open Real

theorem no_injective_function_satisfying_conditions :
  ¬ ∃ (f : ℝ → ℝ), (∀ x1 x2 : ℝ, x1 ≠ x2 → f x1 ≠ f x2)
  ∧ ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (∀ x : ℝ, f (x ^ 2) - (f (a * x + b)) ^ 2 ≥ 1 / 4) :=
by
  sorry

end no_injective_function_satisfying_conditions_l1600_160044


namespace equivalent_proof_problem_l1600_160023

lemma condition_1 (a b : ℝ) (h : b > 0 ∧ 0 > a) : (1 / a) < (1 / b) :=
sorry

lemma condition_2 (a b : ℝ) (h : 0 > a ∧ a > b) : (1 / b) > (1 / a) :=
sorry

lemma condition_4 (a b : ℝ) (h : a > b ∧ b > 0) : (1 / b) > (1 / a) :=
sorry

theorem equivalent_proof_problem (a b : ℝ) :
  (b > 0 ∧ 0 > a → (1 / a) < (1 / b)) ∧
  (0 > a ∧ a > b → (1 / b) > (1 / a)) ∧
  (a > b ∧ b > 0 → (1 / b) > (1 / a)) :=
by {
  exact ⟨condition_1 a b, condition_2 a b, condition_4 a b⟩
}

end equivalent_proof_problem_l1600_160023


namespace total_baseball_fans_l1600_160006

-- Conditions given
def ratio_YM (Y M : ℕ) : Prop := 2 * Y = 3 * M
def ratio_MR (M R : ℕ) : Prop := 4 * R = 5 * M
def M_value : ℕ := 88

-- Prove total number of baseball fans
theorem total_baseball_fans (Y M R : ℕ) (h1 : ratio_YM Y M) (h2 : ratio_MR M R) (hM : M = M_value) :
  Y + M + R = 330 :=
sorry

end total_baseball_fans_l1600_160006


namespace negation_of_proposition_l1600_160020

theorem negation_of_proposition :
  (¬ (∃ x : ℝ, x < 0 ∧ x^2 > 0)) ↔ (∀ x : ℝ, x < 0 → x^2 ≤ 0) :=
  sorry

end negation_of_proposition_l1600_160020


namespace exponent_calculation_l1600_160030

theorem exponent_calculation : (-1 : ℤ) ^ 53 + (2 : ℤ) ^ (5 ^ 3 - 2 ^ 3 + 3 ^ 2) = 2 ^ 126 - 1 :=
by 
  sorry

end exponent_calculation_l1600_160030


namespace calculate_abs_mul_l1600_160034

theorem calculate_abs_mul : |(-3 : ℤ)| * 2 = 6 := 
by 
  -- |(-3)| equals 3 and 3 * 2 equals 6.
  -- The "sorry" is used to complete the statement without proof.
  sorry

end calculate_abs_mul_l1600_160034


namespace zoe_total_earnings_l1600_160045

theorem zoe_total_earnings
  (weeks : ℕ → ℝ)
  (weekly_hours : ℕ → ℝ)
  (wage_per_hour : ℝ)
  (h1 : weekly_hours 3 = 28)
  (h2 : weekly_hours 2 = 18)
  (h3 : weeks 3 - weeks 2 = 64.40)
  (h_same_wage : ∀ n, weeks n = weekly_hours n * wage_per_hour) :
  weeks 3 + weeks 2 = 296.24 :=
sorry

end zoe_total_earnings_l1600_160045


namespace right_triangle_area_l1600_160089

open Real

theorem right_triangle_area
  (a : ℝ) 
  (h₁ : a > 0) 
  (h₂ : a < 24)
  (h₃ : 24^2 + a^2 = (48 - a)^2) : 
  1/2 * 24 * a = 216 :=
by
  -- This is just a statement, the proof is omitted
  sorry

end right_triangle_area_l1600_160089


namespace cost_price_A_min_cost_bshelves_l1600_160043

-- Define the cost price of type B bookshelf
def costB_bshelf : ℝ := 300

-- Define the cost price of type A bookshelf
def costA_bshelf : ℝ := 1.2 * costB_bshelf

-- Define the total number of bookshelves
def total_bshelves : ℕ := 60

-- Define the condition for type A and type B bookshelves count
def typeBshelves := λ (typeAshelves : ℕ) => total_bshelves - typeAshelves
def typeBshelves_constraints := λ (typeAshelves : ℕ) => total_bshelves - typeAshelves ≤ 2 * typeAshelves

-- Define the equation for the costs
noncomputable def total_cost (typeAshelves : ℕ) : ℝ :=
  360 * typeAshelves + 300 * (total_bshelves - typeAshelves)

-- Define the goal: cost price of type A bookshelf is 360 yuan
theorem cost_price_A : costA_bshelf = 360 :=
by 
  sorry

-- Define the goal: the school should buy 20 type A bookshelves and 40 type B bookshelves to minimize cost
theorem min_cost_bshelves : ∃ typeAshelves : ℕ, typeAshelves = 20 ∧ typeBshelves typeAshelves = 40 :=
by
  sorry

end cost_price_A_min_cost_bshelves_l1600_160043


namespace units_digit_of_product_of_seven_consecutive_l1600_160063

theorem units_digit_of_product_of_seven_consecutive (n : ℕ) : 
  ∃ d ∈ [n, n+1, n+2, n+3, n+4, n+5, n+6], d % 10 = 0 :=
by
  sorry

end units_digit_of_product_of_seven_consecutive_l1600_160063


namespace toll_for_18_wheel_truck_l1600_160070

theorem toll_for_18_wheel_truck : 
  let x := 5 
  let w := 15 
  let y := 2 
  let T := 3.50 + 0.50 * (x - 2) + 0.10 * w + y 
  T = 8.50 := 
by 
  -- let x := 5 
  -- let w := 15 
  -- let y := 2 
  -- let T := 3.50 + 0.50 * (x - 2) + 0.10 * w + y 
  -- Note: the let statements within the brackets above
  sorry

end toll_for_18_wheel_truck_l1600_160070


namespace sequence_bound_l1600_160002

theorem sequence_bound (a : ℕ → ℝ) (h_positive : ∀ n, a n > 0)
  (h_condition : ∀ n, (a n)^2 ≤ a n - a (n+1)) :
  ∀ n, a n < 1 / n :=
sorry

end sequence_bound_l1600_160002


namespace sum_of_roots_quadratic_eq_l1600_160014

variable (h : ℝ)
def quadratic_eq_roots (x : ℝ) : Prop := 6 * x^2 - 5 * h * x - 4 * h = 0

theorem sum_of_roots_quadratic_eq (x1 x2 : ℝ) (h : ℝ) 
  (h_roots : quadratic_eq_roots h x1 ∧ quadratic_eq_roots h x2) 
  (h_distinct : x1 ≠ x2) :
  x1 + x2 = 5 * h / 6 := by
sorry

end sum_of_roots_quadratic_eq_l1600_160014


namespace no_positive_divisor_of_2n2_square_l1600_160093

theorem no_positive_divisor_of_2n2_square (n : ℕ) (hn : n > 0) : 
  ∀ d : ℕ, d > 0 → d ∣ 2 * n ^ 2 → ¬∃ x : ℕ, x ^ 2 = d ^ 2 * n ^ 2 + d ^ 3 := 
by
  sorry

end no_positive_divisor_of_2n2_square_l1600_160093


namespace greatest_possible_multiple_of_4_l1600_160058

theorem greatest_possible_multiple_of_4 (x : ℕ) (h1 : x % 4 = 0) (h2 : x^2 < 400) : x ≤ 16 :=
by 
sorry

end greatest_possible_multiple_of_4_l1600_160058


namespace half_of_4_pow_2022_is_2_pow_4043_l1600_160013

theorem half_of_4_pow_2022_is_2_pow_4043 :
  (4 ^ 2022) / 2 = 2 ^ 4043 :=
by sorry

end half_of_4_pow_2022_is_2_pow_4043_l1600_160013


namespace area_of_triangle_l1600_160019

open Matrix

def a : Matrix (Fin 2) (Fin 1) ℤ := ![![4], ![-1]]
def b : Matrix (Fin 2) (Fin 1) ℤ := ![![3], ![5]]

theorem area_of_triangle : (abs (a 0 0 * b 1 0 - a 1 0 * b 0 0) : ℚ) / 2 = 23 / 2 :=
by
  -- To be proved (using :ℚ for the cast to rational for division)
  sorry

end area_of_triangle_l1600_160019


namespace range_u_of_given_condition_l1600_160032

theorem range_u_of_given_condition (x y : ℝ) (h : x^2 / 3 + y^2 = 1) :
  1 ≤ |2 * x + y - 4| + |3 - x - 2 * y| ∧ |2 * x + y - 4| + |3 - x - 2 * y| ≤ 13 := 
sorry

end range_u_of_given_condition_l1600_160032


namespace find_z_l1600_160000

theorem find_z (x y z : ℝ) (h1 : y = 3 * x - 5) (h2 : z = 3 * x + 3) (h3 : y = 1) : z = 9 := 
by
  sorry

end find_z_l1600_160000


namespace total_potatoes_sold_is_322kg_l1600_160090

-- Define the given conditions
def bags_morning := 29
def bags_afternoon := 17
def weight_per_bag := 7

-- The theorem to prove the total kilograms sold is 322kg
theorem total_potatoes_sold_is_322kg : (bags_morning + bags_afternoon) * weight_per_bag = 322 :=
by
  sorry -- Placeholder for the actual proof

end total_potatoes_sold_is_322kg_l1600_160090


namespace morgan_olivia_same_debt_l1600_160099

theorem morgan_olivia_same_debt (t : ℝ) : 
  (200 * (1 + 0.12 * t) = 300 * (1 + 0.04 * t)) → 
  t = 25 / 3 :=
by
  sorry

end morgan_olivia_same_debt_l1600_160099


namespace scientific_notation_of_000000301_l1600_160035

/--
Expressing a small number in scientific notation:
Prove that \(0.000000301\) can be written as \(3.01 \times 10^{-7}\).
-/
theorem scientific_notation_of_000000301 :
  0.000000301 = 3.01 * 10 ^ (-7) :=
sorry

end scientific_notation_of_000000301_l1600_160035


namespace find_S6_l1600_160074

variable (a_n : ℕ → ℝ) -- Assume a_n gives the nth term of an arithmetic sequence.
variable (S_n : ℕ → ℝ) -- Assume S_n gives the sum of the first n terms of the sequence.

-- Conditions:
axiom S_2_eq : S_n 2 = 2
axiom S_4_eq : S_n 4 = 10

-- Define what it means to find S_6
theorem find_S6 : S_n 6 = 18 :=
by
  sorry

end find_S6_l1600_160074


namespace unique_solution_to_function_equation_l1600_160005

theorem unique_solution_to_function_equation (f : ℕ → ℕ)
  (h1 : ∀ n : ℕ, f (2 * n) = 2 * f n)
  (h2 : ∀ n : ℕ, f (2 * n + 1) = 2 * f n + 1) :
  ∀ n : ℕ, f n = n :=
by
  sorry

end unique_solution_to_function_equation_l1600_160005


namespace cows_horses_ratio_l1600_160057

theorem cows_horses_ratio (cows horses : ℕ) (h : cows = 21) (ratio : cows / horses = 7 / 2) : horses = 6 :=
sorry

end cows_horses_ratio_l1600_160057


namespace calculation1_calculation2_calculation3_calculation4_l1600_160079

-- Define the problem and conditions
theorem calculation1 : 9.5 * 101 = 959.5 := 
by 
  sorry

theorem calculation2 : 12.5 * 8.8 = 110 := 
by 
  sorry

theorem calculation3 : 38.4 * 187 - 15.4 * 384 + 3.3 * 16 = 1320 := 
by 
  sorry

theorem calculation4 : 5.29 * 73 + 52.9 * 2.7 = 529 := 
by 
  sorry

end calculation1_calculation2_calculation3_calculation4_l1600_160079


namespace probability_A_not_losing_l1600_160091

theorem probability_A_not_losing (P_draw P_win : ℚ) (h1 : P_draw = 1/2) (h2 : P_win = 1/3) : 
  P_draw + P_win = 5/6 :=
by
  sorry

end probability_A_not_losing_l1600_160091


namespace average_rate_of_interest_l1600_160078

theorem average_rate_of_interest (total_investment : ℝ) (rate1 rate2 average_rate : ℝ) (amount1 amount2 : ℝ)
  (H1 : total_investment = 6000)
  (H2 : rate1 = 0.03)
  (H3 : rate2 = 0.07)
  (H4 : average_rate = 0.042)
  (H5 : amount1 + amount2 = total_investment)
  (H6 : rate1 * amount1 = rate2 * amount2) :
  (rate1 * amount1 + rate2 * amount2) / total_investment = average_rate := 
sorry

end average_rate_of_interest_l1600_160078


namespace min_sum_abc_l1600_160049

theorem min_sum_abc (a b c : ℕ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_abc : a * b * c = 1020) : a + b + c = 33 :=
sorry

end min_sum_abc_l1600_160049


namespace max_cos2_sinx_l1600_160041

noncomputable def cos2_sinx (x : ℝ) : ℝ := (Real.cos x) ^ 2 - Real.sin x

theorem max_cos2_sinx : ∃ x : ℝ, cos2_sinx x = 5 / 4 := 
by
  existsi (Real.arcsin (-1 / 2))
  rw [cos2_sinx]
  -- We need further steps to complete the proof
  sorry

end max_cos2_sinx_l1600_160041


namespace one_cow_one_bag_days_l1600_160018

-- Definitions based on conditions in a)
def cows : ℕ := 60
def bags : ℕ := 75
def days_total : ℕ := 45

-- Main statement for the proof problem
theorem one_cow_one_bag_days : 
  (cows : ℝ) * (bags : ℝ) / (days_total : ℝ) = 1 / 36 := 
by
  sorry   -- Proof placeholder

end one_cow_one_bag_days_l1600_160018


namespace eval_expr_l1600_160062

theorem eval_expr : 8^3 + 8^3 + 8^3 + 8^3 - 2^6 * 2^3 = 1536 := by
  -- Proof will go here
  sorry

end eval_expr_l1600_160062


namespace rainfall_second_week_l1600_160052

theorem rainfall_second_week (x : ℝ) 
  (h1 : x + 1.5 * x = 25) :
  1.5 * x = 15 :=
by
  sorry

end rainfall_second_week_l1600_160052


namespace number_of_diagonals_l1600_160096

-- Define the regular pentagonal prism and its properties
def regular_pentagonal_prism : Type := sorry

-- Define what constitutes a diagonal in this context
def is_diagonal (p : regular_pentagonal_prism) (v1 v2 : Nat) : Prop :=
  sorry -- We need to detail what counts as a diagonal based on the conditions

-- Hypothesis on the structure specifying that there are 5 vertices on the top and 5 on the bottom
axiom vertices_on_top_and_bottom (p : regular_pentagonal_prism) : sorry -- We need the precise formalization

-- The main theorem
theorem number_of_diagonals (p : regular_pentagonal_prism) : ∃ n, n = 10 :=
  sorry

end number_of_diagonals_l1600_160096


namespace sam_time_to_cover_distance_l1600_160017

/-- Define the total distance between points A and B as the sum of distances from A to C and C to B -/
def distance_A_to_C : ℕ := 600
def distance_C_to_B : ℕ := 400
def speed_sam : ℕ := 50
def distance_A_to_B : ℕ := distance_A_to_C + distance_C_to_B

theorem sam_time_to_cover_distance :
  let time := distance_A_to_B / speed_sam
  time = 20 := 
by
  sorry

end sam_time_to_cover_distance_l1600_160017


namespace unpainted_area_of_five_inch_board_l1600_160009

def width1 : ℝ := 5
def width2 : ℝ := 6
def angle : ℝ := 45

theorem unpainted_area_of_five_inch_board : 
  ∃ (area : ℝ), area = 30 :=
by
  sorry

end unpainted_area_of_five_inch_board_l1600_160009


namespace sin_690_degree_l1600_160098

theorem sin_690_degree : Real.sin (690 * Real.pi / 180) = -1/2 :=
by
  sorry

end sin_690_degree_l1600_160098


namespace repeatingDecimal_as_fraction_l1600_160077

def repeatingDecimal : ℚ := 0.136513513513

theorem repeatingDecimal_as_fraction : repeatingDecimal = 136377 / 999000 := 
by 
  sorry

end repeatingDecimal_as_fraction_l1600_160077


namespace number_of_pages_in_each_chapter_l1600_160054

variable (x : ℕ)  -- Variable for number of pages in each chapter

-- Definitions based on the problem conditions
def pages_read_before_4_o_clock := 10 * x
def pages_read_at_4_o_clock := 20
def pages_read_after_4_o_clock := 2 * x
def total_pages_read := pages_read_before_4_o_clock x + pages_read_at_4_o_clock + pages_read_after_4_o_clock x

-- The theorem statement
theorem number_of_pages_in_each_chapter (h : total_pages_read x = 500) : x = 40 :=
sorry

end number_of_pages_in_each_chapter_l1600_160054


namespace find_r_s_l1600_160083

noncomputable def parabola_line_intersection (x y m : ℝ) : Prop :=
  y = x^2 + 5*x ∧ y + 6 = m*(x - 10)

theorem find_r_s (r s m : ℝ) (Q : ℝ × ℝ)
  (hq : Q = (10, -6))
  (h_parabola : ∀ x, ∃ y, y = x^2 + 5*x)
  (h_line : ∀ x, ∃ y, y + 6 = m*(x - 10)) :
  parabola_line_intersection x y m → (r < m ∧ m < s) ∧ (r + s = 50) :=
sorry

end find_r_s_l1600_160083


namespace planar_graph_edge_bound_l1600_160050

structure Graph :=
  (V E : ℕ) -- vertices and edges

def planar_connected (G : Graph) : Prop := 
  sorry -- Planarity and connectivity conditions are complex to formalize

def num_faces (G : Graph) : ℕ :=
  sorry -- Number of faces based on V, E and planarity

theorem planar_graph_edge_bound (G : Graph) (h_planar : planar_connected G) 
  (euler : G.V - G.E + num_faces G = 2) 
  (face_bound : 2 * G.E ≥ 3 * num_faces G) : 
  G.E ≤ 3 * G.V - 6 :=
sorry

end planar_graph_edge_bound_l1600_160050


namespace factor_difference_of_squares_l1600_160007

theorem factor_difference_of_squares (y : ℝ) : 25 - 16 * y ^ 2 = (5 - 4 * y) * (5 + 4 * y) :=
by
  sorry

end factor_difference_of_squares_l1600_160007


namespace dividend_calculation_l1600_160069

theorem dividend_calculation :
  let divisor := 12
  let quotient := 909809
  let dividend := divisor * quotient
  dividend = 10917708 :=
by
  let divisor := 12
  let quotient := 909809
  let dividend := divisor * quotient
  show dividend = 10917708
  sorry

end dividend_calculation_l1600_160069


namespace auntie_em_can_park_l1600_160056

noncomputable def parking_probability : ℚ :=
  let total_ways := (Nat.choose 20 5)
  let unfavorables := (Nat.choose 14 5)
  let probability_cannot_park := (unfavorables : ℚ) / total_ways
  1 - probability_cannot_park

theorem auntie_em_can_park :
  parking_probability = 964 / 1107 :=
by
  sorry

end auntie_em_can_park_l1600_160056


namespace probability_of_quitters_from_10_member_tribe_is_correct_l1600_160042

noncomputable def probability_quitters_from_10_member_tribe : ℚ :=
  let total_contestants := 18
  let ten_member_tribe := 10
  let total_quitters := 2
  let comb (n k : ℕ) : ℕ := Nat.choose n k
  
  let total_combinations := comb total_contestants total_quitters
  let ten_tribe_combinations := comb ten_member_tribe total_quitters
  
  ten_tribe_combinations / total_combinations

theorem probability_of_quitters_from_10_member_tribe_is_correct :
  probability_quitters_from_10_member_tribe = 5 / 17 :=
  by
    sorry

end probability_of_quitters_from_10_member_tribe_is_correct_l1600_160042


namespace problem_statement_l1600_160071

-- Define a : ℝ such that (a + 1/a)^3 = 7
variables (a : ℝ) (h : (a + 1/a)^3 = 7)

-- Goal: Prove that a^4 + 1/a^4 = 1519/81
theorem problem_statement (a : ℝ) (h : (a + 1/a)^3 = 7) : a^4 + 1/a^4 = 1519 / 81 := 
sorry

end problem_statement_l1600_160071


namespace max_projection_area_tetrahedron_l1600_160085

-- Define the side length of the tetrahedron
variable (a : ℝ)

-- Define a theorem stating the maximum projection area of a tetrahedron
theorem max_projection_area_tetrahedron (h : a > 0) : 
  ∃ A, A = (a^2 / 2) :=
by
  -- Proof is omitted
  sorry

end max_projection_area_tetrahedron_l1600_160085


namespace jellybeans_problem_l1600_160033

theorem jellybeans_problem (n : ℕ) (h : n ≥ 100) (h_mod : n % 13 = 11) : n = 102 :=
sorry

end jellybeans_problem_l1600_160033


namespace circle_ratio_l1600_160047

theorem circle_ratio (R r a c : ℝ) (hR : 0 < R) (hr : 0 < r) (h_c_lt_a : 0 < c ∧ c < a) 
  (condition : π * R^2 = (a - c) * (π * R^2 - π * r^2)) :
  R / r = Real.sqrt ((a - c) / (c + 1 - a)) :=
by
  sorry

end circle_ratio_l1600_160047


namespace range_of_2a_plus_b_l1600_160087

theorem range_of_2a_plus_b (a b : ℝ) (ha : 1 < a ∧ a < 4) (hb : -2 < b ∧ b < 2) :
  0 < 2 * a + b ∧ 2 * a + b < 10 :=
sorry

end range_of_2a_plus_b_l1600_160087


namespace geometric_sequence_sum_l1600_160065

def geometric_sequence_props (a : ℕ → ℤ) (S : ℕ → ℤ) := 
  (∀ n, a n = a 1 * 2 ^ (n - 1)) ∧ 
  (∀ n, S n = (1 - 2^n) / (1 - 2)) ∧ 
  (a 5 - a 3 = 12) ∧ 
  (a 6 - a 4 = 24)

theorem geometric_sequence_sum (a : ℕ → ℤ) (S : ℕ → ℤ) 
  (h : geometric_sequence_props a S) :
  ∀ n, S n / a n = 2 - 2^(1 - n) :=
by
  sorry

end geometric_sequence_sum_l1600_160065


namespace integral_cos_square_div_one_plus_cos_minus_sin_squared_l1600_160067

theorem integral_cos_square_div_one_plus_cos_minus_sin_squared:
  ∫ x in (-2 * Real.pi / 3 : Real)..0, (Real.cos x)^2 / (1 + Real.cos x - Real.sin x)^2 = (Real.sqrt 3) / 2 - Real.log 2 := 
by
  sorry

end integral_cos_square_div_one_plus_cos_minus_sin_squared_l1600_160067


namespace focus_of_parabola_y_eq_8x2_l1600_160039

open Real

noncomputable def parabola_focus (a p : ℝ) : ℝ × ℝ :=
  (0, 1 / (4 * p))

theorem focus_of_parabola_y_eq_8x2 :
  parabola_focus 8 (1 / 16) = (0, 1 / 32) :=
by
  sorry

end focus_of_parabola_y_eq_8x2_l1600_160039


namespace isosceles_triangle_perimeter_l1600_160046

-- Define the sides of the isosceles triangle
def side1 : ℝ := 4
def side2 : ℝ := 8

-- Hypothesis: The perimeter of an isosceles triangle with the given sides
-- Given condition
def is_isosceles_triangle (a b c : ℝ) : Prop := (a = b ∨ b = c ∨ c = a) ∧ a + b > c ∧ b + c > a ∧ c + a > b

-- Theorem statement
theorem isosceles_triangle_perimeter (a b : ℝ) (h1 : a = side1 ∨ a = side2) (h2 : b = side1 ∨ b = side2) :
  ∃ p : ℝ, is_isosceles_triangle a b side2 ∧ p = a + b + side2 → p = 20 :=
sorry

end isosceles_triangle_perimeter_l1600_160046


namespace solve_3_pow_n_plus_55_eq_m_squared_l1600_160021

theorem solve_3_pow_n_plus_55_eq_m_squared :
  ∃ (n m : ℕ), 3^n + 55 = m^2 ∧ ((n = 2 ∧ m = 8) ∨ (n = 6 ∧ m = 28)) :=
by
  sorry

end solve_3_pow_n_plus_55_eq_m_squared_l1600_160021


namespace find_cost_of_baseball_l1600_160084

noncomputable def total_amount : ℝ := 20.52
noncomputable def cost_of_marbles : ℝ := 9.05
noncomputable def cost_of_football : ℝ := 4.95
noncomputable def cost_of_baseball : ℝ := total_amount - (cost_of_marbles + cost_of_football)

theorem find_cost_of_baseball : cost_of_baseball = 6.52 := sorry

end find_cost_of_baseball_l1600_160084


namespace find_expression_value_l1600_160092

theorem find_expression_value (x : ℝ) (h : x^2 - 3 * x - 1 = 0) : -3 * x^2 + 9 * x + 4 = 1 :=
by sorry

end find_expression_value_l1600_160092


namespace remainder_count_l1600_160068

theorem remainder_count (n : ℕ) (h : n > 5) : 
  ∃ l : List ℕ, l.length = 5 ∧ ∀ x ∈ l, x ∣ 42 ∧ x > 5 := 
sorry

end remainder_count_l1600_160068


namespace calculate_expression_l1600_160040

theorem calculate_expression :
  16 * (1/2) * 4 * (1/16) / 2 = 1 := 
by
  sorry

end calculate_expression_l1600_160040


namespace sum_of_edges_equals_74_l1600_160012

def V (pyramid : ℕ) : ℕ := pyramid

def E (pyramid : ℕ) : ℕ := 2 * (V pyramid - 1)

def sum_of_edges (pyramid1 pyramid2 pyramid3 : ℕ) : ℕ :=
  E pyramid1 + E pyramid2 + E pyramid3

theorem sum_of_edges_equals_74 (V₁ V₂ V₃ : ℕ) (h : V₁ + V₂ + V₃ = 40) :
  sum_of_edges V₁ V₂ V₃ = 74 :=
sorry

end sum_of_edges_equals_74_l1600_160012


namespace right_triangle_leg_square_l1600_160031

theorem right_triangle_leg_square (a c b : ℕ) (h1 : c = a + 1) (h2 : a^2 + b^2 = c^2) : b^2 = c + a :=
by
  sorry

end right_triangle_leg_square_l1600_160031


namespace polynomial_roots_fraction_sum_l1600_160055

theorem polynomial_roots_fraction_sum (a b c : ℝ) 
  (h1 : a + b + c = 12) 
  (h2 : ab + ac + bc = 20) 
  (h3 : abc = 3) : 
  (1 / a^2) + (1 / b^2) + (1 / c^2) = 328 / 9 := 
by 
  sorry

end polynomial_roots_fraction_sum_l1600_160055


namespace medians_formula_l1600_160011

noncomputable def ma (a b c : ℝ) : ℝ := (1 / 2) * ((2 * b^2 + 2 * c^2 - a^2) ^ (1 / 2))
noncomputable def mb (a b c : ℝ) : ℝ := (1 / 2) * ((2 * c^2 + 2 * a^2 - b^2) ^ (1 / 2))
noncomputable def mc (a b c : ℝ) : ℝ := (1 / 2) * ((2 * a^2 + 2 * b^2 - c^2) ^ (1 / 2))

theorem medians_formula (a b c : ℝ) :
  ma a b c = (1 / 2) * ((2 * b^2 + 2 * c^2 - a^2) ^ (1 / 2)) ∧
  mb a b c = (1 / 2) * ((2 * c^2 + 2 * a^2 - b^2) ^ (1 / 2)) ∧
  mc a b c = (1 / 2) * ((2 * a^2 + 2 * b^2 - c^2) ^ (1 / 2)) :=
by sorry

end medians_formula_l1600_160011


namespace max_distance_eq_of_l1_l1600_160080

noncomputable def equation_of_l1 (l1 l2 : ℝ → ℝ) (A B : ℝ × ℝ) : Prop :=
  A = (1, 3) ∧ B = (2, 4) ∧ -- Points A and B
  l1 A.1 = A.2 ∧ l2 B.1 = B.2 ∧ -- l1 passes through A and l2 passes through B
  (∀ (x : ℝ), l1 x - l2 x = 1) ∧ -- l1 and l2 are parallel (constant difference in y-values)
  (∃ (c : ℝ), ∀ (x : ℝ), l1 x = -x + c ∧ l2 x = -x + c + 1) -- distance maximized

theorem max_distance_eq_of_l1 : 
  ∃ (l1 l2 : ℝ → ℝ), equation_of_l1 l1 l2 (1, 3) (2, 4) ∧
  ∀ (x : ℝ), l1 x = -x + 4 := 
sorry

end max_distance_eq_of_l1_l1600_160080


namespace conditional_probability_B_given_A_l1600_160024

/-
Given a box containing 6 balls: 2 red, 2 yellow, and 2 blue.
One ball is drawn with replacement for 3 times.
Let event A be "the color of the ball drawn in the first draw is the same as the color of the ball drawn in the second draw".
Let event B be "the color of the balls drawn in all three draws is the same".
Prove that the conditional probability P(B|A) is 1/3.
-/
noncomputable def total_balls := 6
noncomputable def red_balls := 2
noncomputable def yellow_balls := 2
noncomputable def blue_balls := 2

noncomputable def event_A (n : ℕ) : ℕ := 
  3 * 2 * 2 * total_balls

noncomputable def event_AB (n : ℕ) : ℕ := 
  3 * 2 * 2 * 2

noncomputable def P_B_given_A : ℚ := 
  event_AB total_balls / event_A total_balls

theorem conditional_probability_B_given_A :
  P_B_given_A = 1 / 3 :=
by sorry

end conditional_probability_B_given_A_l1600_160024


namespace edward_spent_money_l1600_160064

-- Definitions based on the conditions
def books := 2
def cost_per_book := 3

-- Statement of the proof problem
theorem edward_spent_money : 
  (books * cost_per_book = 6) :=
by
  -- proof goes here
  sorry

end edward_spent_money_l1600_160064


namespace range_of_a_l1600_160073

noncomputable def f (x : ℝ) : ℝ := 2 * x + 1 / Real.exp x - Real.exp x

theorem range_of_a (a : ℝ) (h : f (a - 1) + f (2 * a ^ 2) ≤ 0) : 
  a ∈ Set.Iic (-1) ∪ Set.Ici (1 / 2) :=
sorry

end range_of_a_l1600_160073


namespace q_zero_iff_arithmetic_l1600_160015

-- Definitions of the terms and conditions
variables (A B q : ℝ) (hA : A ≠ 0)
def Sn (n : ℕ) : ℝ := A * n^2 + B * n + q
def arithmetic_sequence (an : ℕ → ℝ) : Prop := ∃ d a1, ∀ n, an n = a1 + n * d

-- The proof statement we need to show
theorem q_zero_iff_arithmetic (an : ℕ → ℝ) :
  (q = 0) ↔ (∃ a1 d, ∀ n, Sn A B 0 n = (d / 2) * n^2 + (a1 - d / 2) * n) :=
sorry

end q_zero_iff_arithmetic_l1600_160015


namespace system_is_inconsistent_l1600_160075

def system_of_equations (x1 x2 x3 : ℝ) : Prop :=
  (x1 + 4*x2 + 10*x3 = 1) ∧
  (0*x1 - 5*x2 - 13*x3 = -1.25) ∧
  (0*x1 + 0*x2 + 0*x3 = 1.25)

theorem system_is_inconsistent : 
  ∀ x1 x2 x3, ¬ system_of_equations x1 x2 x3 :=
by
  intro x1 x2 x3
  sorry

end system_is_inconsistent_l1600_160075


namespace smallest_whole_number_greater_than_sum_is_12_l1600_160003

-- Definitions of the mixed numbers as improper fractions
def a : ℚ := 5 / 3
def b : ℚ := 9 / 4
def c : ℚ := 27 / 8
def d : ℚ := 25 / 6

-- The target sum and the required proof statement
theorem smallest_whole_number_greater_than_sum_is_12 : 
  let sum := a + b + c + d
  let smallest_whole_number_greater_than_sum := Nat.ceil sum
  smallest_whole_number_greater_than_sum = 12 :=
by 
  sorry

end smallest_whole_number_greater_than_sum_is_12_l1600_160003


namespace books_left_over_l1600_160061

def total_books (box_count : ℕ) (books_per_box : ℤ) : ℤ :=
  box_count * books_per_box

theorem books_left_over
  (box_count : ℕ)
  (books_per_box : ℤ)
  (new_box_capacity : ℤ)
  (books_total : ℤ := total_books box_count books_per_box) :
  box_count = 1500 →
  books_per_box = 35 →
  new_box_capacity = 43 →
  books_total % new_box_capacity = 40 :=
by
  intros
  sorry

end books_left_over_l1600_160061


namespace rate_is_900_l1600_160016

noncomputable def rate_per_square_meter (L W : ℝ) (total_cost : ℝ) : ℝ :=
  total_cost / (L * W)

theorem rate_is_900 :
  rate_per_square_meter 5 4.75 21375 = 900 := by
  sorry

end rate_is_900_l1600_160016


namespace percentage_of_720_equals_356_point_4_l1600_160008

theorem percentage_of_720_equals_356_point_4 : 
  let part := 356.4
  let whole := 720
  (part / whole) * 100 = 49.5 :=
by
  sorry

end percentage_of_720_equals_356_point_4_l1600_160008


namespace number_of_nintendo_games_to_give_away_l1600_160097

-- Define the conditions
def initial_nintendo_games : ℕ := 20
def desired_nintendo_games_left : ℕ := 12

-- Define the proof problem as a Lean theorem
theorem number_of_nintendo_games_to_give_away :
  initial_nintendo_games - desired_nintendo_games_left = 8 :=
by
  sorry

end number_of_nintendo_games_to_give_away_l1600_160097


namespace intersection_eq_l1600_160082

def setM : Set ℝ := { x | x^2 - 2*x < 0 }
def setN : Set ℝ := { x | -1 ≤ x ∧ x ≤ 1 }

theorem intersection_eq : setM ∩ setN = { x | 0 < x ∧ x ≤ 1 } := sorry

end intersection_eq_l1600_160082


namespace average_class_size_l1600_160001

theorem average_class_size 
  (num_3_year_olds : ℕ) 
  (num_4_year_olds : ℕ) 
  (num_5_year_olds : ℕ) 
  (num_6_year_olds : ℕ) 
  (class_size_3_and_4 : num_3_year_olds = 13 ∧ num_4_year_olds = 20) 
  (class_size_5_and_6 : num_5_year_olds = 15 ∧ num_6_year_olds = 22) :
  (num_3_year_olds + num_4_year_olds + num_5_year_olds + num_6_year_olds) / 2 = 35 :=
by
  sorry

end average_class_size_l1600_160001


namespace minimum_value_of_function_l1600_160025

theorem minimum_value_of_function (x : ℝ) (hx : x > 5 / 4) : 
  ∃ y, y = 4 * x + 1 / (4 * x - 5) ∧ y = 7 :=
sorry

end minimum_value_of_function_l1600_160025


namespace emilia_cartons_total_l1600_160029

theorem emilia_cartons_total (strawberries blueberries supermarket : ℕ) (total_needed : ℕ)
  (h1 : strawberries = 2)
  (h2 : blueberries = 7)
  (h3 : supermarket = 33)
  (h4 : total_needed = strawberries + blueberries + supermarket) :
  total_needed = 42 :=
sorry

end emilia_cartons_total_l1600_160029


namespace first_digit_of_sum_l1600_160051

theorem first_digit_of_sum (n : ℕ) (a : ℕ) (hs : 9 * a = n)
  (h_sum : n = 43040102 - (10^7 * d - 10^7 * 4)) : 
  (10^7 * d - 10^7 * 4) / 10^7 = 8 :=
by
  sorry

end first_digit_of_sum_l1600_160051


namespace range_of_x_l1600_160094

theorem range_of_x (x y : ℝ) (h1 : 4 * x + y = 3) (h2 : -2 < y ∧ y ≤ 7) : 1 ≤ x ∧ x < 5 / 4 := 
  sorry

end range_of_x_l1600_160094


namespace triangle_pentagon_side_ratio_l1600_160026

theorem triangle_pentagon_side_ratio :
  let triangle_perimeter := 60
  let pentagon_perimeter := 60
  let triangle_side := triangle_perimeter / 3
  let pentagon_side := pentagon_perimeter / 5
  (triangle_side : ℕ) / (pentagon_side : ℕ) = 5 / 3 :=
by
  sorry

end triangle_pentagon_side_ratio_l1600_160026


namespace parallelogram_side_sum_l1600_160048

variable (x y : ℚ)

theorem parallelogram_side_sum :
  4 * x - 1 = 10 →
  5 * y + 3 = 12 →
  x + y = 91 / 20 :=
by
  intros h1 h2
  sorry

end parallelogram_side_sum_l1600_160048


namespace find_number_l1600_160022

theorem find_number (n : ℝ) (h : n / 0.06 = 16.666666666666668) : n = 1 :=
by
  sorry

end find_number_l1600_160022


namespace artifacts_per_wing_l1600_160028

theorem artifacts_per_wing
  (total_wings : ℕ)
  (num_paintings : ℕ)
  (num_artifacts : ℕ)
  (painting_wings : ℕ)
  (large_paintings_wings : ℕ)
  (small_paintings_wings : ℕ)
  (small_paintings_per_wing : ℕ)
  (artifact_wings : ℕ)
  (wings_division : total_wings = painting_wings + artifact_wings)
  (paintings_division : painting_wings = large_paintings_wings + small_paintings_wings)
  (num_large_paintings : large_paintings_wings = 2)
  (num_small_paintings : small_paintings_wings * small_paintings_per_wing = num_paintings - large_paintings_wings)
  (num_artifact_calc : num_artifacts = 8 * num_paintings)
  (artifact_wings_div : artifact_wings = total_wings - painting_wings)
  (artifact_calc : num_artifacts / artifact_wings = 66) :
  num_artifacts / artifact_wings = 66 := 
by
  sorry

end artifacts_per_wing_l1600_160028


namespace determinant_evaluation_l1600_160086

theorem determinant_evaluation (x z : ℝ) :
  (Matrix.det ![
    ![1, x, z],
    ![1, x + z, z],
    ![1, x, x + z]
  ]) = x * z - z * z := 
sorry

end determinant_evaluation_l1600_160086


namespace parabola_properties_l1600_160076

theorem parabola_properties (p : ℝ) (h : p > 0) (F : ℝ × ℝ) (l : ℝ → ℝ) (A B : ℝ × ℝ) (M : ℝ × ℝ)
  (hp : p = 4) 
  (hF : F = (p / 2, 0)) 
  (hA : A.2^2 = 2 * p * A.1) 
  (hB : B.2^2 = 2 * p * B.1) 
  (hM : M = ((A.1 + B.1) / 2, 2)) 
  (hl : ∀ x, l x = 2 * x - 4) 
  : (p = 4) ∧ (l 0 = -4) ∧ (A ≠ B) → 
    (p = 4) ∧ (l 0 = -4) ∧ (A ≠ B) ∧ (|A.1 - B.1| + |A.2 - B.2| = 10) :=
by 
  sorry

end parabola_properties_l1600_160076


namespace factorization_correct_l1600_160066

theorem factorization_correct (x : ℝ) : x^2 - 9 = (x + 3) * (x - 3) :=
by 
  sorry

end factorization_correct_l1600_160066


namespace simplify_expression_l1600_160004

theorem simplify_expression (x y z : ℝ) : - (x - (y - z)) = -x + y - z := by
  sorry

end simplify_expression_l1600_160004
