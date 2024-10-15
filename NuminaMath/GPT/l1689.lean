import Mathlib

namespace NUMINAMATH_GPT_average_price_of_tshirts_l1689_168957

theorem average_price_of_tshirts
  (A : ℝ)
  (total_cost_seven_remaining : ℝ := 7 * 505)
  (total_cost_three_returned : ℝ := 3 * 673)
  (total_cost_eight : ℝ := total_cost_seven_remaining + 673) -- since (1 t-shirt with price is included in the total)
  (total_cost_eight_eq : total_cost_eight = 8 * A) :
  A = 526 :=
by sorry

end NUMINAMATH_GPT_average_price_of_tshirts_l1689_168957


namespace NUMINAMATH_GPT_distance_between_ports_l1689_168904

theorem distance_between_ports (x : ℝ) (speed_ship : ℝ) (speed_water : ℝ) (time_difference : ℝ) 
  (speed_downstream := speed_ship + speed_water) 
  (speed_upstream := speed_ship - speed_water) 
  (time_downstream := x / speed_downstream) 
  (time_upstream := x / speed_upstream) 
  (h : time_downstream + time_difference = time_upstream) 
  (h_ship : speed_ship = 26)
  (h_water : speed_water = 2)
  (h_time : time_difference = 3) : x = 504 :=
by
  -- The proof is omitted 
  sorry

end NUMINAMATH_GPT_distance_between_ports_l1689_168904


namespace NUMINAMATH_GPT_distance_swim_against_current_l1689_168911

-- Definitions based on problem conditions
def swimmer_speed_still_water : ℝ := 4 -- km/h
def water_current_speed : ℝ := 1 -- km/h
def time_swimming_against_current : ℝ := 2 -- hours

-- Calculation of effective speed against the current
def effective_speed_against_current : ℝ :=
  swimmer_speed_still_water - water_current_speed

-- Proof statement
theorem distance_swim_against_current :
  effective_speed_against_current * time_swimming_against_current = 6 :=
by
  -- By substituting values from the problem,
  -- effective_speed_against_current * time_swimming_against_current = 3 * 2
  -- which equals 6.
  sorry

end NUMINAMATH_GPT_distance_swim_against_current_l1689_168911


namespace NUMINAMATH_GPT_sum_of_powers_2017_l1689_168991

theorem sum_of_powers_2017 (n : ℕ) (x : Fin n → ℤ) (h : ∀ i, x i = 0 ∨ x i = 1 ∨ x i = -1) (h_sum : (Finset.univ : Finset (Fin n)).sum x = 1000) :
  (Finset.univ : Finset (Fin n)).sum (λ i => (x i)^2017) = 1000 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_powers_2017_l1689_168991


namespace NUMINAMATH_GPT_conditional_probability_event_B_given_event_A_l1689_168965

-- Definitions of events A and B
def event_A := {outcomes | ∃ i j k, outcomes = (i, j, k) ∧ (i = 1 ∨ j = 1 ∨ k = 1)}
def event_B := {outcomes | ∃ i j k, outcomes = (i, j, k) ∧ (i + j + k = 1)}

-- Calculation of probabilities
def probability_AB := 3 / 8
def probability_A := 7 / 8

-- Prove conditional probability
theorem conditional_probability_event_B_given_event_A :
  (probability_AB / probability_A) = 3 / 7 :=
by
  sorry

end NUMINAMATH_GPT_conditional_probability_event_B_given_event_A_l1689_168965


namespace NUMINAMATH_GPT_age_ratio_proof_l1689_168964

-- Define the ages
def sonAge := 22
def manAge := sonAge + 24

-- Define the ratio computation statement
def ageRatioInTwoYears : ℚ := 
  let sonAgeInTwoYears := sonAge + 2
  let manAgeInTwoYears := manAge + 2
  manAgeInTwoYears / sonAgeInTwoYears

-- The theorem to prove
theorem age_ratio_proof : ageRatioInTwoYears = 2 :=
by
  sorry

end NUMINAMATH_GPT_age_ratio_proof_l1689_168964


namespace NUMINAMATH_GPT_ellipse_standard_equation_l1689_168926

theorem ellipse_standard_equation
  (F : ℝ × ℝ)
  (e : ℝ)
  (eq1 : F = (0, 1))
  (eq2 : e = 1 / 2) :
  ∃ (a b : ℝ), a = 2 ∧ b ^ 2 = 3 ∧ (∀ x y : ℝ, (y ^ 2 / 4) + (x ^ 2 / 3) = 1) :=
by
  sorry

end NUMINAMATH_GPT_ellipse_standard_equation_l1689_168926


namespace NUMINAMATH_GPT_integer_solutions_count_l1689_168948

theorem integer_solutions_count : 
  ∃ (S : Finset (ℤ × ℤ)), 
  (∀ x y, (x, y) ∈ S ↔ x^2 + x * y + 2 * y^2 = 29) ∧ 
  S.card = 4 := 
sorry

end NUMINAMATH_GPT_integer_solutions_count_l1689_168948


namespace NUMINAMATH_GPT_rice_mixture_ratio_l1689_168916

theorem rice_mixture_ratio (x y z : ℕ) (h : 16 * x + 24 * y + 30 * z = 18 * (x + y + z)) : 
  x = 9 * y + 18 * z :=
by
  sorry

end NUMINAMATH_GPT_rice_mixture_ratio_l1689_168916


namespace NUMINAMATH_GPT_sum_of_digits_is_13_l1689_168919

theorem sum_of_digits_is_13:
  ∀ (a b c d : ℕ),
  b + c = 10 ∧
  c + d = 1 ∧
  a + d = 2 →
  a + b + c + d = 13 :=
by {
  sorry
}

end NUMINAMATH_GPT_sum_of_digits_is_13_l1689_168919


namespace NUMINAMATH_GPT_car_r_speed_l1689_168932

variable (v : ℝ)

theorem car_r_speed (h1 : (300 / v - 2 = 300 / (v + 10))) : v = 30 := 
sorry

end NUMINAMATH_GPT_car_r_speed_l1689_168932


namespace NUMINAMATH_GPT_tangent_line_at_point_l1689_168995

noncomputable def tangent_line_eq (x y : ℝ) : Prop := x^3 - y = 0

theorem tangent_line_at_point :
  tangent_line_eq (-2) (-8) →
  ∃ (k : ℝ), (k = 12) ∧ (12 * x - y + 16 = 0) :=
sorry

end NUMINAMATH_GPT_tangent_line_at_point_l1689_168995


namespace NUMINAMATH_GPT_snake_body_length_l1689_168966

theorem snake_body_length (L : ℝ) (H : ℝ) (h1 : H = L / 10) (h2 : L = 10) : L - H = 9 :=
by
  sorry

end NUMINAMATH_GPT_snake_body_length_l1689_168966


namespace NUMINAMATH_GPT_problem1_problem2_l1689_168924

-- Problem 1: Proove that the given expression equals 1
theorem problem1 : (2021 * 2023) / (2022^2 - 1) = 1 :=
  by
  sorry

-- Problem 2: Proove that the given expression equals 45000
theorem problem2 : 2 * 101^2 + 2 * 101 * 98 + 2 * 49^2 = 45000 :=
  by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1689_168924


namespace NUMINAMATH_GPT_find_a_for_even_function_l1689_168967

theorem find_a_for_even_function :
  ∀ a : ℝ, (∀ x : ℝ, a * 3^x + 1 / 3^x = a * 3^(-x) + 1 / 3^(-x)) → a = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_a_for_even_function_l1689_168967


namespace NUMINAMATH_GPT_square_garden_tiles_l1689_168959

theorem square_garden_tiles (n : ℕ) (h : 2 * n - 1 = 25) : n^2 = 169 :=
by
  sorry

end NUMINAMATH_GPT_square_garden_tiles_l1689_168959


namespace NUMINAMATH_GPT_intersection_M_N_l1689_168930

-- Define set M and N
def M : Set ℝ := {x | x - 1 < 0}
def N : Set ℝ := {x | x^2 - 5 * x + 6 > 0}

-- Problem statement to show their intersection
theorem intersection_M_N :
  M ∩ N = {x | x < 1} := 
sorry

end NUMINAMATH_GPT_intersection_M_N_l1689_168930


namespace NUMINAMATH_GPT_sufficient_not_necessary_condition_l1689_168936

-- Definitions of propositions
def propA (x : ℝ) : Prop := (x - 1)^2 < 9
def propB (x a : ℝ) : Prop := (x + 2) * (x + a) < 0

-- Lean statement of the problem
theorem sufficient_not_necessary_condition (a : ℝ) : 
  (∀ x, propA x → propB x a) ∧ (∃ x, ¬ propA x ∧ propB x a) ↔ a < -4 :=
sorry

end NUMINAMATH_GPT_sufficient_not_necessary_condition_l1689_168936


namespace NUMINAMATH_GPT_total_cost_l1689_168915

def cost(M R F : ℝ) := 10 * M = 24 * R ∧ 6 * F = 2 * R ∧ F = 23

theorem total_cost (M R F : ℝ) (h : cost M R F) : 
  4 * M + 3 * R + 5 * F = 984.40 :=
by
  sorry

end NUMINAMATH_GPT_total_cost_l1689_168915


namespace NUMINAMATH_GPT_find_f_2012_l1689_168901

variable (f : ℕ → ℝ)

axiom f_one : f 1 = 3997
axiom recurrence : ∀ x, f x - f (x + 1) = 1

theorem find_f_2012 : f 2012 = 1986 :=
by
  -- Skipping proof
  sorry

end NUMINAMATH_GPT_find_f_2012_l1689_168901


namespace NUMINAMATH_GPT_min_value_of_fraction_l1689_168970

theorem min_value_of_fraction (m n : ℝ) (hm : 0 < m) (hn : 0 < n) (h : 2 * m + n = 1) : 
  (1 / m + 2 / n) = 8 := 
by 
  sorry

end NUMINAMATH_GPT_min_value_of_fraction_l1689_168970


namespace NUMINAMATH_GPT_solve_for_m_l1689_168953

theorem solve_for_m (a_0 a_1 a_2 a_3 a_4 a_5 m : ℝ)
  (h1 : (x : ℝ) → (x + m)^5 = a_0 + a_1 * (x+1) + a_2 * (x+1)^2 + a_3 * (x+1)^3 + a_4 * (x+1)^4 + a_5 * (x+1)^5)
  (h2 : a_0 + a_1 + a_2 + a_3 + a_4 + a_5 = 32) :
  m = 2 :=
sorry

end NUMINAMATH_GPT_solve_for_m_l1689_168953


namespace NUMINAMATH_GPT_MeatMarket_sales_l1689_168993

theorem MeatMarket_sales :
  let thursday_sales := 210
  let friday_sales := 2 * thursday_sales
  let saturday_sales := 130
  let sunday_sales := saturday_sales / 2
  let total_sales := thursday_sales + friday_sales + saturday_sales + sunday_sales
  let planned_sales := 500
  total_sales - planned_sales = 325 :=
by
  let thursday_sales := 210
  let friday_sales := 2 * thursday_sales
  let saturday_sales := 130
  let sunday_sales := saturday_sales / 2
  let total_sales := thursday_sales + friday_sales + saturday_sales + sunday_sales
  let planned_sales := 500
  show total_sales - planned_sales = 325
  sorry

end NUMINAMATH_GPT_MeatMarket_sales_l1689_168993


namespace NUMINAMATH_GPT_dreamCarCost_l1689_168952

-- Definitions based on given conditions
def monthlyEarnings : ℕ := 4000
def monthlySavings : ℕ := 500
def totalEarnings : ℕ := 360000

-- Theorem stating the desired result
theorem dreamCarCost :
  (totalEarnings / monthlyEarnings) * monthlySavings = 45000 :=
by
  sorry

end NUMINAMATH_GPT_dreamCarCost_l1689_168952


namespace NUMINAMATH_GPT_largest_angle_in_hexagon_l1689_168955

theorem largest_angle_in_hexagon :
  ∀ (x : ℝ), (2 * x + 3 * x + 3 * x + 4 * x + 4 * x + 5 * x = 720) →
  5 * x = 1200 / 7 :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_largest_angle_in_hexagon_l1689_168955


namespace NUMINAMATH_GPT_people_who_didnt_show_up_l1689_168913

-- Definitions based on the conditions
def invited_people : ℕ := 68
def people_per_table : ℕ := 3
def tables_needed : ℕ := 6

-- Theorem statement
theorem people_who_didnt_show_up : 
  (invited_people - tables_needed * people_per_table = 50) :=
by 
  sorry

end NUMINAMATH_GPT_people_who_didnt_show_up_l1689_168913


namespace NUMINAMATH_GPT_mo_tea_cups_l1689_168935

theorem mo_tea_cups (n t : ℤ) (h1 : 4 * n + 3 * t = 22) (h2 : 3 * t = 4 * n + 8) : t = 5 :=
by
  -- proof steps
  sorry

end NUMINAMATH_GPT_mo_tea_cups_l1689_168935


namespace NUMINAMATH_GPT_Tony_packs_of_pens_l1689_168946

theorem Tony_packs_of_pens (T : ℕ) 
  (Kendra_packs : ℕ := 4) 
  (pens_per_pack : ℕ := 3) 
  (Kendra_keep : ℕ := 2) 
  (Tony_keep : ℕ := 2)
  (friends_pens : ℕ := 14) 
  (total_pens_given : Kendra_packs * pens_per_pack - Kendra_keep + 3 * T - Tony_keep = friends_pens) :
  T = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_Tony_packs_of_pens_l1689_168946


namespace NUMINAMATH_GPT_distance_halfway_along_orbit_l1689_168923

variable {Zeta : Type}  -- Zeta is a type representing the planet
variable (distance_from_focus : Zeta → ℝ)  -- Function representing the distance from the sun (focus)

-- Conditions
variable (perigee_distance : ℝ := 3)
variable (apogee_distance : ℝ := 15)
variable (a : ℝ := (perigee_distance + apogee_distance) / 2)  -- semi-major axis

theorem distance_halfway_along_orbit (z : Zeta) (h1 : distance_from_focus z = perigee_distance) (h2 : distance_from_focus z = apogee_distance) :
  distance_from_focus z = a :=
sorry

end NUMINAMATH_GPT_distance_halfway_along_orbit_l1689_168923


namespace NUMINAMATH_GPT_Tony_slices_left_after_week_l1689_168945

-- Define the conditions and problem statement
def Tony_slices_per_day (days : ℕ) : ℕ := days * 2
def Tony_slices_on_Saturday : ℕ := 3 + 2
def Tony_slice_on_Sunday : ℕ := 1
def Total_slices_used (days : ℕ) : ℕ := Tony_slices_per_day days + Tony_slices_on_Saturday + Tony_slice_on_Sunday
def Initial_loaf : ℕ := 22
def Slices_left (days : ℕ) : ℕ := Initial_loaf - Total_slices_used days

-- Prove that Tony has 6 slices left after a week
theorem Tony_slices_left_after_week : Slices_left 5 = 6 := by
  sorry

end NUMINAMATH_GPT_Tony_slices_left_after_week_l1689_168945


namespace NUMINAMATH_GPT_probability_of_passing_through_correct_l1689_168954

def probability_of_passing_through (n k : ℕ) : ℚ :=
(2 * k * n - 2 * k^2 + 2 * k - 1) / n^2

theorem probability_of_passing_through_correct (n k : ℕ) (h1 : 1 ≤ k) (h2 : k ≤ n) :
  probability_of_passing_through n k = (2 * k * n - 2 * k^2 + 2 * k - 1) / n^2 := 
by
  sorry

end NUMINAMATH_GPT_probability_of_passing_through_correct_l1689_168954


namespace NUMINAMATH_GPT_infinite_series_sum_eq_1_div_432_l1689_168981

theorem infinite_series_sum_eq_1_div_432 :
  (∑' n : ℕ, (4 * (n + 1) + 1) / ((4 * (n + 1) - 1)^3 * (4 * (n + 1) + 3)^3)) = (1 / 432) :=
  sorry

end NUMINAMATH_GPT_infinite_series_sum_eq_1_div_432_l1689_168981


namespace NUMINAMATH_GPT_ratio_correct_l1689_168999

-- Definitions based on the problem conditions
def initial_cards_before_eating (X : ℤ) : ℤ := X
def cards_bought_new : ℤ := 4
def cards_left_after_eating : ℤ := 34

-- Definition of the number of cards eaten by the dog
def cards_eaten_by_dog (X : ℤ) : ℤ := X + cards_bought_new - cards_left_after_eating

-- Definition of the ratio of the number of cards eaten to the total number of cards before being eaten
def ratio_cards_eaten_to_total (X : ℤ) : ℚ := (cards_eaten_by_dog X : ℚ) / (X + cards_bought_new : ℚ)

-- Statement to prove
theorem ratio_correct (X : ℤ) : ratio_cards_eaten_to_total X = (X - 30) / (X + 4) := by
  sorry

end NUMINAMATH_GPT_ratio_correct_l1689_168999


namespace NUMINAMATH_GPT_period_of_f_l1689_168979

noncomputable def f : ℝ → ℝ := sorry

def functional_equation (f : ℝ → ℝ) := ∀ x y : ℝ, f (2 * x) + f (2 * y) = f (x + y) * f (x - y)

def f_pi_zero (f : ℝ → ℝ) := f (Real.pi) = 0

def f_not_identically_zero (f : ℝ → ℝ) := ∃ x : ℝ, f x ≠ 0

theorem period_of_f (f : ℝ → ℝ)
  (hf_eq : functional_equation f)
  (hf_pi_zero : f_pi_zero f)
  (hf_not_zero : f_not_identically_zero f) : 
  ∀ x : ℝ, f (x + 4 * Real.pi) = f x := sorry

end NUMINAMATH_GPT_period_of_f_l1689_168979


namespace NUMINAMATH_GPT_percentage_relationships_l1689_168975

variable (a b c d e f g : ℝ)

theorem percentage_relationships (h1 : d = 0.22 * b) (h2 : d = 0.35 * f)
                                 (h3 : e = 0.27 * a) (h4 : e = 0.60 * f)
                                 (h5 : c = 0.14 * a) (h6 : c = 0.40 * b)
                                 (h7 : d = 2 * c) (h8 : g = 3 * e):
    b = 0.7 * a ∧ f = 0.45 * a ∧ g = 0.81 * a :=
sorry

end NUMINAMATH_GPT_percentage_relationships_l1689_168975


namespace NUMINAMATH_GPT_problem_solution_l1689_168922

noncomputable def f (x : ℝ) : ℝ := (1 / 4) * (x + 1)^2

theorem problem_solution :
  (∀ x : ℝ, (0 < x ∧ x ≤ 5) → x ≤ f x ∧ f x ≤ 2 * |x - 1| + 1) →
  (f 1 = 4 * (1 / 4) + 1) →
  (∃ (t m : ℝ), m > 1 ∧ 
               (∀ x : ℝ, (1 ≤ x ∧ x ≤ m) → f t ≤ (1 / 4) * (x + t + 1)^2)) →
  (1 / 4 = 1 / 4) ∧ (m = 2) :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_problem_solution_l1689_168922


namespace NUMINAMATH_GPT_exp_7pi_over_2_eq_i_l1689_168927

theorem exp_7pi_over_2_eq_i : Complex.exp (7 * Real.pi * Complex.I / 2) = Complex.I :=
by
  sorry

end NUMINAMATH_GPT_exp_7pi_over_2_eq_i_l1689_168927


namespace NUMINAMATH_GPT_percentage_reduction_is_correct_l1689_168928

def percentage_reduction_alcohol_concentration (V_original V_added : ℚ) (C_original : ℚ) : ℚ :=
  let V_total := V_original + V_added
  let Amount_alcohol := V_original * C_original
  let C_new := Amount_alcohol / V_total
  ((C_original - C_new) / C_original) * 100

theorem percentage_reduction_is_correct :
  percentage_reduction_alcohol_concentration 12 28 0.20 = 70 := by
  sorry

end NUMINAMATH_GPT_percentage_reduction_is_correct_l1689_168928


namespace NUMINAMATH_GPT_carol_pennies_l1689_168908

variable (a c : ℕ)

theorem carol_pennies (h₁ : c + 2 = 4 * (a - 2)) (h₂ : c - 2 = 3 * (a + 2)) : c = 62 :=
by
  sorry

end NUMINAMATH_GPT_carol_pennies_l1689_168908


namespace NUMINAMATH_GPT_range_of_a_l1689_168983

section
variables (a : ℝ)
def p : Prop := ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0
def q : Prop := ∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0

theorem range_of_a (h : p a ∧ q a) : a = 1 ∨ a ≤ -2 :=
sorry
end

end NUMINAMATH_GPT_range_of_a_l1689_168983


namespace NUMINAMATH_GPT_polynomial_multiplication_l1689_168974

theorem polynomial_multiplication (x z : ℝ) :
  (3*x^5 - 7*z^3) * (9*x^10 + 21*x^5*z^3 + 49*z^6) = 27*x^15 - 343*z^9 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_multiplication_l1689_168974


namespace NUMINAMATH_GPT_original_price_of_sarees_l1689_168986

theorem original_price_of_sarees (P : ℝ) (h : 0.72 * P = 108) : P = 150 := 
by 
  sorry

end NUMINAMATH_GPT_original_price_of_sarees_l1689_168986


namespace NUMINAMATH_GPT_turtle_minimum_distance_l1689_168968

theorem turtle_minimum_distance 
  (constant_speed : ℝ)
  (turn_angle : ℝ)
  (total_time : ℕ) :
  constant_speed = 5 →
  turn_angle = 90 →
  total_time = 11 →
  ∃ (final_position : ℝ × ℝ), 
    (final_position = (5, 0) ∨ final_position = (-5, 0) ∨ final_position = (0, 5) ∨ final_position = (0, -5)) ∧
    dist final_position (0, 0) = 5 :=
by
  intros
  sorry

end NUMINAMATH_GPT_turtle_minimum_distance_l1689_168968


namespace NUMINAMATH_GPT_Daniel_correct_answers_l1689_168980

theorem Daniel_correct_answers
  (c w : ℕ)
  (h1 : c + w = 12)
  (h2 : 4 * c - 3 * w = 21) :
  c = 9 :=
sorry

end NUMINAMATH_GPT_Daniel_correct_answers_l1689_168980


namespace NUMINAMATH_GPT_Earl_rate_36_l1689_168969

theorem Earl_rate_36 (E : ℝ) (h1 : E + (2 / 3) * E = 60) : E = 36 :=
by {
  sorry
}

end NUMINAMATH_GPT_Earl_rate_36_l1689_168969


namespace NUMINAMATH_GPT_no_real_roots_iff_l1689_168938

noncomputable def f (x a : ℝ) : ℝ := x^2 + 2*x + a

theorem no_real_roots_iff (a : ℝ) : (∀ x : ℝ, f x a ≠ 0) → a > 1 :=
  by
    sorry

end NUMINAMATH_GPT_no_real_roots_iff_l1689_168938


namespace NUMINAMATH_GPT_speed_of_stream_l1689_168918

theorem speed_of_stream 
  (v : ℝ)
  (boat_speed : ℝ)
  (distance_downstream : ℝ)
  (distance_upstream : ℝ)
  (H1 : boat_speed = 12)
  (H2 : distance_downstream = 32)
  (H3 : distance_upstream = 16)
  (H4 : distance_downstream / (boat_speed + v) = distance_upstream / (boat_speed - v)) :
  v = 4 :=
by
  sorry

end NUMINAMATH_GPT_speed_of_stream_l1689_168918


namespace NUMINAMATH_GPT_gcd_m_n_15_lcm_m_n_45_l1689_168934

-- Let m and n be integers greater than 0, and 3m + 2n = 225.
variables (m n : ℕ) (h1 : m > 0) (h2 : n > 0) (h3 : 3 * m + 2 * n = 225)

-- First part: If the greatest common divisor of m and n is 15, then m + n = 105.
theorem gcd_m_n_15 (h4 : Int.gcd m n = 15) : m + n = 105 :=
sorry

-- Second part: If the least common multiple of m and n is 45, then m + n = 90.
theorem lcm_m_n_45 (h5 : Int.lcm m n = 45) : m + n = 90 :=
sorry

end NUMINAMATH_GPT_gcd_m_n_15_lcm_m_n_45_l1689_168934


namespace NUMINAMATH_GPT_inequality_system_solution_l1689_168984

theorem inequality_system_solution (a b : ℝ) (h : ∀ x : ℝ, x > -a → x > -b) : a ≥ b :=
by
  sorry

end NUMINAMATH_GPT_inequality_system_solution_l1689_168984


namespace NUMINAMATH_GPT_arithmetic_seq_middle_term_l1689_168937

theorem arithmetic_seq_middle_term (a1 a3 y : ℤ) (h1 : a1 = 3^2) (h2 : a3 = 3^4)
    (h3 : y = (a1 + a3) / 2) : y = 45 :=
by
  rw [h1, h2] at h3
  simp at h3
  exact h3

end NUMINAMATH_GPT_arithmetic_seq_middle_term_l1689_168937


namespace NUMINAMATH_GPT_factorize_cubic_expression_l1689_168941

variable (a : ℝ)

theorem factorize_cubic_expression : a^3 - a = a * (a + 1) * (a - 1) := 
sorry

end NUMINAMATH_GPT_factorize_cubic_expression_l1689_168941


namespace NUMINAMATH_GPT_sum_even_1_to_200_l1689_168925

open Nat

/-- The sum of all even numbers from 1 to 200 is 10100. --/
theorem sum_even_1_to_200 :
  let first_term := 2
  let last_term := 200
  let common_diff := 2
  let n := (last_term - first_term) / common_diff + 1
  let sum := n / 2 * (first_term + last_term)
  sum = 10100 :=
by
  let first_term := 2
  let last_term := 200
  let common_diff := 2
  let n := (last_term - first_term) / common_diff + 1
  let sum := n / 2 * (first_term + last_term)
  show sum = 10100
  sorry

end NUMINAMATH_GPT_sum_even_1_to_200_l1689_168925


namespace NUMINAMATH_GPT_discount_percentage_l1689_168976

theorem discount_percentage (CP SP SP_no_discount discount : ℝ)
  (h1 : SP = CP * (1 + 0.44))
  (h2 : SP_no_discount = CP * (1 + 0.50))
  (h3 : discount = SP_no_discount - SP) :
  (discount / SP_no_discount) * 100 = 4 :=
by
  sorry

end NUMINAMATH_GPT_discount_percentage_l1689_168976


namespace NUMINAMATH_GPT_company_stores_l1689_168912

theorem company_stores (total_uniforms : ℕ) (uniforms_per_store : ℕ) 
  (h1 : total_uniforms = 121) (h2 : uniforms_per_store = 4) : 
  total_uniforms / uniforms_per_store = 30 :=
by
  sorry

end NUMINAMATH_GPT_company_stores_l1689_168912


namespace NUMINAMATH_GPT_parallel_planes_l1689_168905

variables {Point Line Plane : Type}
variables (a : Line) (α β : Plane)

-- Conditions
def line_perpendicular_plane (l: Line) (p: Plane) : Prop := sorry
def planes_parallel (p₁ p₂: Plane) : Prop := sorry

-- Problem statement
theorem parallel_planes (h1: line_perpendicular_plane a α) 
                        (h2: line_perpendicular_plane a β) : 
                        planes_parallel α β :=
sorry

end NUMINAMATH_GPT_parallel_planes_l1689_168905


namespace NUMINAMATH_GPT_number_of_apartment_complexes_l1689_168994

theorem number_of_apartment_complexes (width_land length_land side_complex : ℕ)
    (h_width : width_land = 262) (h_length : length_land = 185) 
    (h_side : side_complex = 18) :
    width_land / side_complex * length_land / side_complex = 140 := by
  -- given conditions
  rw [h_width, h_length, h_side]
  -- apply calculation steps for clarity (not necessary for final theorem)
  -- calculate number of complexes along width
  have h1 : 262 / 18 = 14 := sorry
  -- calculate number of complexes along length
  have h2 : 185 / 18 = 10 := sorry
  -- final product calculation
  sorry

end NUMINAMATH_GPT_number_of_apartment_complexes_l1689_168994


namespace NUMINAMATH_GPT_abs_eq_three_system1_system2_l1689_168939

theorem abs_eq_three : ∀ x : ℝ, |x| = 3 ↔ x = 3 ∨ x = -3 := 
by sorry

theorem system1 : ∀ x y : ℝ, (y * (x - 1) = 0) ∧ (2 * x + 5 * y = 7) → 
(x = 7 / 2 ∧ y = 0) ∨ (x = 1 ∧ y = 1) := 
by sorry

theorem system2 : ∀ x y : ℝ, (x * y - 2 * x - y + 2 = 0) ∧ (x + 6 * y = 3) ∧ (3 * x + y = 8) → 
(x = 1 ∧ y = 5) ∨ (x = 2 ∧ y = 2) := 
by sorry

end NUMINAMATH_GPT_abs_eq_three_system1_system2_l1689_168939


namespace NUMINAMATH_GPT_mul_101_101_l1689_168933

theorem mul_101_101 : 101 * 101 = 10201 := 
by
  sorry

end NUMINAMATH_GPT_mul_101_101_l1689_168933


namespace NUMINAMATH_GPT_unshaded_squares_in_tenth_figure_l1689_168909

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a₁ + d * (n - 1)

theorem unshaded_squares_in_tenth_figure :
  arithmetic_sequence 8 4 10 = 44 :=
by
  sorry

end NUMINAMATH_GPT_unshaded_squares_in_tenth_figure_l1689_168909


namespace NUMINAMATH_GPT_arithmetic_sequence_n_l1689_168985

theorem arithmetic_sequence_n {a : ℕ → ℕ} (h1 : a 1 = 1) (h2 : ∀ n, a (n + 1) = a n + 3) :
  (∃ n : ℕ, a n = 2005) → (∃ n : ℕ, n = 669) :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_n_l1689_168985


namespace NUMINAMATH_GPT_small_poster_ratio_l1689_168988

theorem small_poster_ratio (total_posters : ℕ) (medium_posters large_posters small_posters : ℕ)
  (h1 : total_posters = 50)
  (h2 : medium_posters = 50 / 2)
  (h3 : large_posters = 5)
  (h4 : small_posters = total_posters - medium_posters - large_posters)
  (h5 : total_posters ≠ 0) :
  small_posters = 20 ∧ (small_posters : ℚ) / total_posters = 2 / 5 := 
sorry

end NUMINAMATH_GPT_small_poster_ratio_l1689_168988


namespace NUMINAMATH_GPT_ratio_of_x_y_l1689_168998

theorem ratio_of_x_y (x y : ℝ) (h₁ : 3 < (x - y) / (x + y)) (h₂ : (x - y) / (x + y) < 4) (h₃ : ∃ a b : ℤ, x = a * y / b ) (h₄ : x + y = 10) :
  x / y = -2 := sorry

end NUMINAMATH_GPT_ratio_of_x_y_l1689_168998


namespace NUMINAMATH_GPT_annual_average_growth_rate_l1689_168978

theorem annual_average_growth_rate (x : ℝ) (h : x > 0): 
  100 * (1 + x)^2 = 169 :=
sorry

end NUMINAMATH_GPT_annual_average_growth_rate_l1689_168978


namespace NUMINAMATH_GPT_competition_end_time_l1689_168944

def time_in_minutes := 24 * 60  -- Total minutes in 24 hours

def competition_start_time := 14 * 60 + 30  -- 2:30 p.m. in minutes from midnight

theorem competition_end_time :
  competition_start_time + 1440 = competition_start_time :=
by 
  sorry

end NUMINAMATH_GPT_competition_end_time_l1689_168944


namespace NUMINAMATH_GPT_quadratic_no_real_roots_l1689_168914

theorem quadratic_no_real_roots (c : ℝ) : (∀ x : ℝ, x^2 + 2 * x + c ≠ 0) → c > 1 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_no_real_roots_l1689_168914


namespace NUMINAMATH_GPT_sequence_fifth_term_l1689_168960

theorem sequence_fifth_term (a : ℕ → ℕ) (h₁ : a 1 = 1) (h₂ : a 2 = 2)
    (h₃ : ∀ n > 2, a n = a (n-1) + a (n-2)) : a 5 = 8 :=
sorry

end NUMINAMATH_GPT_sequence_fifth_term_l1689_168960


namespace NUMINAMATH_GPT_number_of_books_l1689_168947

theorem number_of_books (Maddie Luisa Amy Noah : ℕ)
  (H1 : Maddie = 15)
  (H2 : Luisa = 18)
  (H3 : Amy + Luisa = Maddie + 9)
  (H4 : Noah = Amy / 3)
  : Amy + Noah = 8 :=
sorry

end NUMINAMATH_GPT_number_of_books_l1689_168947


namespace NUMINAMATH_GPT_sum_of_three_numbers_l1689_168962

theorem sum_of_three_numbers :
  ∀ (a b c : ℕ), 
  a ≤ b ∧ b ≤ c → b = 10 →
  (a + b + c) / 3 = a + 20 →
  (a + b + c) / 3 = c - 30 →
  a + b + c = 60 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_three_numbers_l1689_168962


namespace NUMINAMATH_GPT_rosie_pies_l1689_168906

def number_of_pies (apples : ℕ) : ℕ := sorry

theorem rosie_pies (h : number_of_pies 9 = 2) : number_of_pies 27 = 6 :=
by sorry

end NUMINAMATH_GPT_rosie_pies_l1689_168906


namespace NUMINAMATH_GPT_find_value_of_m_and_n_l1689_168950

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := x^3 + 3*x^2 + m * x
noncomputable def g (x : ℝ) (n : ℝ) : ℝ := Real.log (x + 1) + n * x

theorem find_value_of_m_and_n (m n : ℝ) (h₀ : n > 0) 
  (h₁ : f (-1) m = -1) 
  (h₂ : ∀ x : ℝ, f x m = g x n → x = 0) :
  m + n = 5 := 
by 
  sorry

end NUMINAMATH_GPT_find_value_of_m_and_n_l1689_168950


namespace NUMINAMATH_GPT_minimum_students_l1689_168921

variables (b g : ℕ) -- Define variables for boys and girls

-- Define the conditions
def boys_passed : ℕ := (3 * b) / 4
def girls_passed : ℕ := (2 * g) / 3
def equal_passed := boys_passed b = girls_passed g

def total_students := b + g + 4

-- Statement to prove minimum students in the class
theorem minimum_students (h1 : equal_passed b g)
  (h2 : ∃ multiple_of_nine : ℕ, g = 9 * multiple_of_nine ∧ 3 * b = 4 * multiple_of_nine * 2) :
  total_students b g = 21 :=
sorry

end NUMINAMATH_GPT_minimum_students_l1689_168921


namespace NUMINAMATH_GPT_find_c_k_l1689_168977

noncomputable def common_difference (a : ℕ → ℕ) : ℕ := sorry
noncomputable def common_ratio (b : ℕ → ℕ) : ℕ := sorry
noncomputable def arith_seq (d : ℕ) (n : ℕ) : ℕ := 1 + (n - 1) * d
noncomputable def geom_seq (r : ℕ) (n : ℕ) : ℕ := r^(n - 1)
noncomputable def combined_seq (a : ℕ → ℕ) (b : ℕ → ℕ) (n : ℕ) : ℕ := a n + b n

variable (k : ℕ) (d : ℕ) (r : ℕ)

-- Conditions
axiom arith_condition : common_difference (arith_seq d) = d
axiom geom_condition : common_ratio (geom_seq r) = r
axiom combined_k_minus_1 : combined_seq (arith_seq d) (geom_seq r) (k - 1) = 50
axiom combined_k_plus_1 : combined_seq (arith_seq d) (geom_seq r) (k + 1) = 1500

-- Prove that c_k = 2406
theorem find_c_k : combined_seq (arith_seq d) (geom_seq r) k = 2406 := by
  sorry

end NUMINAMATH_GPT_find_c_k_l1689_168977


namespace NUMINAMATH_GPT_part_a_part_b_part_c_l1689_168929

/-- (a) Given that p = 33 and q = 216, show that the equation f(x) = 0 has 
three distinct integer solutions and the equation g(x) = 0 has two distinct integer solutions.
-/
theorem part_a (p q : ℕ) (h_p : p = 33) (h_q : q = 216) :
  (∃ x1 x2 x3 : ℤ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 ∧ (x1 * x2 * x3 = 216 ∧ x1 + x2 + x3 = 33 ∧ x1 = 0))
  ∧ (∃ y1 y2 : ℤ, y1 ≠ y2 ∧ (3 * y1 * y2 = 216 ∧ y1 + y1 = 22)) := sorry

/-- (b) Suppose that the equation f(x) = 0 has three distinct integer solutions 
and the equation g(x) = 0 has two distinct integer solutions. Prove the necessary conditions 
for p and q.
-/
theorem part_b (p q : ℕ) 
  (h_f : ∃ x1 x2 x3 : ℤ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 ∧ (x1 * x2 * x3 = q ∧ x1 + x2 + x3 = p))
  (h_g : ∃ y1 y2 : ℤ, y1 ≠ y2 ∧ (3 * y1 * y2 = q ∧ y1 + y1 = 2 * p)) :
  (∃ k : ℕ, p = 3 * k) ∧ (∃ l : ℕ, q = 9 * l) ∧ (∃ m n : ℕ, p^2 - 3 * q = m^2 ∧ p^2 - 4 * q = n^2) := sorry

/-- (c) Prove that there are infinitely many pairs of positive integers (p, q) for which:
1. The equation f(x) = 0 has three distinct integer solutions.
2. The equation g(x) = 0 has two distinct integer solutions.
3. The greatest common divisor of p and q is 3.
-/
theorem part_c :
  ∃ (p q : ℕ) (infinitely_many : ℕ → Prop),
  (∃ x1 x2 x3 : ℤ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 ∧ (x1 * x2 * x3 = q ∧ x1 + x2 + x3 = p))
  ∧ (∃ y1 y2 : ℤ, y1 ≠ y2 ∧ (3 * y1 * y2 = q ∧ y1 + y1 = 2 * p))
  ∧ ∃ k : ℕ, gcd p q = 3 ∧ infinitely_many k := sorry

end NUMINAMATH_GPT_part_a_part_b_part_c_l1689_168929


namespace NUMINAMATH_GPT_intersection_of_sets_l1689_168990

def setA : Set ℝ := {x | (x^2 - x - 2 < 0)}
def setB : Set ℝ := {y | ∃ x ≤ 0, y = 3^x}

theorem intersection_of_sets : (setA ∩ setB) = {z | 0 < z ∧ z ≤ 1} :=
sorry

end NUMINAMATH_GPT_intersection_of_sets_l1689_168990


namespace NUMINAMATH_GPT_students_passed_in_both_subjects_l1689_168982

theorem students_passed_in_both_subjects:
  ∀ (F_H F_E F_HE : ℝ), F_H = 0.30 → F_E = 0.42 → F_HE = 0.28 → (1 - (F_H + F_E - F_HE)) = 0.56 :=
by
  intros F_H F_E F_HE h1 h2 h3
  sorry

end NUMINAMATH_GPT_students_passed_in_both_subjects_l1689_168982


namespace NUMINAMATH_GPT_intersection_distance_zero_l1689_168956

noncomputable def A : Type := ℝ × ℝ

def P : A := (2, 0)

def line_intersects_parabola (x y : ℝ) : Prop :=
  y - 2 * x + 5 = 0 ∧ y^2 = 3 * x + 4

def distance (p1 p2 : A) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

theorem intersection_distance_zero :
  ∀ (A1 A2 : A),
  line_intersects_parabola A1.1 A1.2 ∧ line_intersects_parabola A2.1 A2.2 →
  (abs (distance A1 P - distance A2 P) = 0) :=
sorry

end NUMINAMATH_GPT_intersection_distance_zero_l1689_168956


namespace NUMINAMATH_GPT_smallest_delightful_integer_l1689_168963

-- Definition of "delightful" integer
def is_delightful (B : ℤ) : Prop :=
  ∃ (n : ℕ), (n > 0) ∧ ((n + 1) * (2 * B + n)) / 2 = 3050

-- Proving the smallest delightful integer
theorem smallest_delightful_integer : ∃ (B : ℤ), is_delightful B ∧ ∀ (B' : ℤ), is_delightful B' → B ≤ B' :=
  sorry

end NUMINAMATH_GPT_smallest_delightful_integer_l1689_168963


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_mod_l1689_168961

theorem arithmetic_sequence_sum_mod (a d l k S n : ℕ) 
  (h_seq_start : a = 3)
  (h_common_difference : d = 5)
  (h_last_term : l = 103)
  (h_sum_formula : S = (k * (3 + 103)) / 2)
  (h_term_count : k = 21)
  (h_mod_condition : 1113 % 17 = n)
  (h_range_condition : 0 ≤ n ∧ n < 17) : 
  n = 8 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_mod_l1689_168961


namespace NUMINAMATH_GPT_factorize_x3_minus_9x_factorize_a3b_minus_2a2b_plus_ab_l1689_168971

theorem factorize_x3_minus_9x (x : ℝ) : x^3 - 9 * x = x * (x + 3) * (x - 3) :=
sorry

theorem factorize_a3b_minus_2a2b_plus_ab (a b : ℝ) : a^3 * b - 2 * a^2 * b + a * b = a * b * (a - 1)^2 :=
sorry

end NUMINAMATH_GPT_factorize_x3_minus_9x_factorize_a3b_minus_2a2b_plus_ab_l1689_168971


namespace NUMINAMATH_GPT_exist_colored_points_r_gt_pi_div_sqrt3_exist_colored_points_r_gt_pi_div_2_l1689_168920

theorem exist_colored_points_r_gt_pi_div_sqrt3 (r : ℝ) (hr : r > π / Real.sqrt 3) 
    (coloring : ℝ × ℝ → Prop) : 
    (∀ p, 0 ≤ p.1 ∧ p.1^2 + p.2^2 < r^2 → coloring p ∨ ¬coloring p) → 
    ∃ A B : ℝ × ℝ, (0 ≤ A.1 ∧ 0 ≤ B.1 ∧ A.1^2 + A.2^2 < r^2 ∧ B.1^2 + B.2^2 < r^2 ∧ dist A B = π ∧ coloring A = coloring B) :=
sorry

theorem exist_colored_points_r_gt_pi_div_2 (r : ℝ) (hr : r > π / 2)
    (coloring : ℝ × ℝ → Prop) : 
    (∀ p, 0 ≤ p.1 ∧ p.1^2 + p.2^2 < r^2 → coloring p ∨ ¬coloring p) → 
    ∃ A B : ℝ × ℝ, (0 ≤ A.1 ∧ 0 ≤ B.1 ∧ A.1^2 + A.2^2 < r^2 ∧ B.1^2 + B.2^2 < r^2 ∧ dist A B = π ∧ coloring A = coloring B) :=
sorry

end NUMINAMATH_GPT_exist_colored_points_r_gt_pi_div_sqrt3_exist_colored_points_r_gt_pi_div_2_l1689_168920


namespace NUMINAMATH_GPT_determine_abc_l1689_168900

theorem determine_abc (a b c : ℕ) (h1 : a * b * c = 2^4 * 3^2 * 5^3) 
  (h2 : gcd a b = 15) (h3 : gcd a c = 5) (h4 : gcd b c = 20) : 
  a = 15 ∧ b = 60 ∧ c = 20 :=
by
  sorry

end NUMINAMATH_GPT_determine_abc_l1689_168900


namespace NUMINAMATH_GPT_ratio_of_almonds_to_walnuts_l1689_168951

theorem ratio_of_almonds_to_walnuts
  (A W : ℝ)
  (weight_almonds : ℝ)
  (total_weight : ℝ)
  (weight_walnuts : ℝ)
  (ratio : 2 * W = total_weight - weight_almonds)
  (given_almonds : weight_almonds = 107.14285714285714)
  (given_total_weight : total_weight = 150)
  (computed_weight_walnuts : weight_walnuts = 42.85714285714286)
  (proportion : A / (2 * W) = weight_almonds / weight_walnuts) :
  A / W = 5 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_almonds_to_walnuts_l1689_168951


namespace NUMINAMATH_GPT_number_of_students_l1689_168902

theorem number_of_students (B S : ℕ) 
  (h1 : S = 9 * B + 1) 
  (h2 : S = 10 * B - 10) : 
  S = 100 := 
by 
  { sorry }

end NUMINAMATH_GPT_number_of_students_l1689_168902


namespace NUMINAMATH_GPT_turtle_distance_in_six_minutes_l1689_168997

theorem turtle_distance_in_six_minutes 
  (observers : ℕ)
  (time_interval : ℕ)
  (distance_seen : ℕ)
  (total_time : ℕ)
  (total_distance : ℕ)
  (observation_per_minute : ∀ t ≤ total_time, ∃ n : ℕ, n ≤ observers ∧ (∃ interval : ℕ, interval ≤ time_interval ∧ distance_seen = 1)) :
  total_distance = 10 :=
sorry

end NUMINAMATH_GPT_turtle_distance_in_six_minutes_l1689_168997


namespace NUMINAMATH_GPT_compare_squares_l1689_168931

theorem compare_squares (a b c : ℝ) : a^2 + b^2 + c^2 ≥ ab + bc + ca :=
by
  sorry

end NUMINAMATH_GPT_compare_squares_l1689_168931


namespace NUMINAMATH_GPT_find_other_number_l1689_168940

theorem find_other_number (a b : ℕ) (h1 : Nat.gcd a b = 24) (h2 : Nat.lcm a b = 5040) (h3 : a = 240) : b = 504 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_other_number_l1689_168940


namespace NUMINAMATH_GPT_automobile_travel_distance_l1689_168907

variable (a r : ℝ)

theorem automobile_travel_distance (h : r ≠ 0) :
  (a / 4) * (240 / 1) * (1 / (3 * r)) = (20 * a) / r := 
by
  sorry

end NUMINAMATH_GPT_automobile_travel_distance_l1689_168907


namespace NUMINAMATH_GPT_cedar_vs_pine_height_cedar_vs_birch_height_l1689_168917

-- Define the heights as rational numbers
def pine_tree_height := 14 + 1/4
def birch_tree_height := 18 + 1/2
def cedar_tree_height := 20 + 5/8

-- Theorem to prove the height differences
theorem cedar_vs_pine_height :
  cedar_tree_height - pine_tree_height = 6 + 3/8 :=
by
  sorry

theorem cedar_vs_birch_height :
  cedar_tree_height - birch_tree_height = 2 + 1/8 :=
by
  sorry

end NUMINAMATH_GPT_cedar_vs_pine_height_cedar_vs_birch_height_l1689_168917


namespace NUMINAMATH_GPT_speed_of_man_in_still_water_l1689_168910

-- Define the conditions as given in step (a)
axiom conditions :
  ∃ (v_m v_s : ℝ),
    (40 / 5 = v_m + v_s) ∧
    (30 / 5 = v_m - v_s)

-- State the theorem that proves the speed of the man in still water
theorem speed_of_man_in_still_water : ∃ v_m : ℝ, v_m = 7 :=
by
  obtain ⟨v_m, v_s, h1, h2⟩ := conditions
  have h3 : v_m + v_s = 8 := by sorry
  have h4 : v_m - v_s = 6 := by sorry
  have h5 : 2 * v_m = 14 := by sorry
  exact ⟨7, by linarith⟩

end NUMINAMATH_GPT_speed_of_man_in_still_water_l1689_168910


namespace NUMINAMATH_GPT_total_population_after_births_l1689_168996

theorem total_population_after_births:
  let initial_population := 300000
  let immigrants := 50000
  let emigrants := 30000
  let pregnancies_fraction := 1 / 8
  let twins_fraction := 1 / 4
  let net_population := initial_population + immigrants - emigrants
  let pregnancies := net_population * pregnancies_fraction
  let twin_pregnancies := pregnancies * twins_fraction
  let twin_children := twin_pregnancies * 2
  let single_births := pregnancies - twin_pregnancies
  net_population + single_births + twin_children = 370000 := by
  sorry

end NUMINAMATH_GPT_total_population_after_births_l1689_168996


namespace NUMINAMATH_GPT_sample_capacity_l1689_168972

theorem sample_capacity (freq : ℕ) (freq_rate : ℚ) (H_freq : freq = 36) (H_freq_rate : freq_rate = 0.25) : 
  ∃ n : ℕ, n = 144 :=
by
  sorry

end NUMINAMATH_GPT_sample_capacity_l1689_168972


namespace NUMINAMATH_GPT_range_xf_ge_0_l1689_168992

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then -x - 2 else - (-x) - 2

theorem range_xf_ge_0 :
  { x : ℝ | x * f x ≥ 0 } = { x : ℝ | -2 ≤ x ∧ x ≤ 2 } :=
by
  sorry

end NUMINAMATH_GPT_range_xf_ge_0_l1689_168992


namespace NUMINAMATH_GPT_min_sum_of_squares_l1689_168973

theorem min_sum_of_squares (y1 y2 y3 : ℝ) (h1 : y1 > 0) (h2 : y2 > 0) (h3 : y3 > 0) (h4 : y1 + 3 * y2 + 4 * y3 = 72) : 
  y1^2 + y2^2 + y3^2 ≥ 2592 / 13 ∧ (∃ k, y1 = k ∧ y2 = 3 * k ∧ y3 = 4 * k ∧ k = 36 / 13) :=
sorry

end NUMINAMATH_GPT_min_sum_of_squares_l1689_168973


namespace NUMINAMATH_GPT_average_xyz_l1689_168989

theorem average_xyz (x y z : ℝ) 
  (h1 : 2003 * z - 4006 * x = 1002) 
  (h2 : 2003 * y + 6009 * x = 4004) : (x + y + z) / 3 = 5 / 6 :=
by
  sorry

end NUMINAMATH_GPT_average_xyz_l1689_168989


namespace NUMINAMATH_GPT_find_stream_speed_l1689_168903

variable (boat_speed dist_downstream dist_upstream : ℝ)
variable (stream_speed : ℝ)

noncomputable def speed_of_stream (boat_speed dist_downstream dist_upstream : ℝ) : ℝ :=
  let t_downstream := dist_downstream / (boat_speed + stream_speed)
  let t_upstream := dist_upstream / (boat_speed - stream_speed)
  if t_downstream = t_upstream then stream_speed else 0

theorem find_stream_speed
  (h : speed_of_stream 20 26 14 stream_speed = stream_speed) :
  stream_speed = 6 :=
sorry

end NUMINAMATH_GPT_find_stream_speed_l1689_168903


namespace NUMINAMATH_GPT_prisoners_freedom_guaranteed_l1689_168958

-- Definition of the problem strategy
def prisoners_strategy (n : ℕ) : Prop :=
  ∃ counter regular : ℕ → ℕ,
    (∀ i, i < n - 1 → regular i < 2) ∧ -- Each regular prisoner turns on the light only once
    (∃ count : ℕ, 
      counter count = 99 ∧  -- The counter counts to 99 based on the strategy
      (∀ k, k < 99 → (counter (k + 1) = counter k + 1))) -- Each turn off increases the count by one

-- The main proof statement that there is a strategy ensuring the prisoners' release
theorem prisoners_freedom_guaranteed : ∀ (n : ℕ), n = 100 →
  prisoners_strategy n :=
by {
  sorry -- The actual proof is omitted
}

end NUMINAMATH_GPT_prisoners_freedom_guaranteed_l1689_168958


namespace NUMINAMATH_GPT_possible_n_values_l1689_168949

theorem possible_n_values (x y z n : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
    x^3 + y^3 + z^3 = n * x^2 * y^2 * z^2 → n = 1 ∨ n = 3 :=
by 
  sorry

end NUMINAMATH_GPT_possible_n_values_l1689_168949


namespace NUMINAMATH_GPT_largest_common_value_under_800_l1689_168942

-- Let's define the problem conditions as arithmetic sequences
def sequence1 (a : ℤ) : Prop := ∃ n : ℤ, a = 4 + 5 * n
def sequence2 (a : ℤ) : Prop := ∃ m : ℤ, a = 7 + 8 * m

-- Now we state the theorem that the largest common value less than 800 is 799
theorem largest_common_value_under_800 : 
  ∃ a : ℤ, sequence1 a ∧ sequence2 a ∧ a < 800 ∧ ∀ b : ℤ, sequence1 b ∧ sequence2 b ∧ b < 800 → b ≤ a :=
sorry

end NUMINAMATH_GPT_largest_common_value_under_800_l1689_168942


namespace NUMINAMATH_GPT_area_triangle_COD_l1689_168987

noncomputable def area_of_triangle (t s : ℝ) : ℝ := 
  1 / 2 * abs (5 + 2 * s + 7 * t)

theorem area_triangle_COD (t s : ℝ) : 
  ∃ (C : ℝ × ℝ) (D : ℝ × ℝ), 
    C = (3 + 5 * t, 2 + 4 * t) ∧ 
    D = (2 + 5 * s, 3 + 4 * s) ∧ 
    area_of_triangle t s = 1 / 2 * abs (5 + 2 * s + 7 * t) :=
by
  sorry

end NUMINAMATH_GPT_area_triangle_COD_l1689_168987


namespace NUMINAMATH_GPT_equivalent_statements_l1689_168943

-- Definitions based on the problem
def is_not_negative (x : ℝ) : Prop := x >= 0
def is_not_positive (x : ℝ) : Prop := x <= 0
def is_positive (x : ℝ) : Prop := x > 0
def is_negative (x : ℝ) : Prop := x < 0

-- The main theorem statement
theorem equivalent_statements (x : ℝ) : 
  (is_not_negative x → is_not_positive (x^2)) ↔ (is_positive (x^2) → is_negative x) :=
by
  sorry

end NUMINAMATH_GPT_equivalent_statements_l1689_168943
