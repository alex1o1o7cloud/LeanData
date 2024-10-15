import Mathlib

namespace NUMINAMATH_GPT_fraction_equiv_l291_29179

theorem fraction_equiv (x y : ℝ) (h : x / 2 = y / 5) : x / y = 2 / 5 :=
by
  sorry

end NUMINAMATH_GPT_fraction_equiv_l291_29179


namespace NUMINAMATH_GPT_dist_between_centers_l291_29149

noncomputable def dist_centers_tangent_circles : ℝ :=
  let a₁ := 5 + 2 * Real.sqrt 2
  let a₂ := 5 - 2 * Real.sqrt 2
  Real.sqrt 2 * (a₁ - a₂)

theorem dist_between_centers :
  let a₁ := 5 + 2 * Real.sqrt 2
  let a₂ := 5 - 2 * Real.sqrt 2
  let C₁ := (a₁, a₁)
  let C₂ := (a₂, a₂)
  dist_centers_tangent_circles = 8 :=
by
  sorry

end NUMINAMATH_GPT_dist_between_centers_l291_29149


namespace NUMINAMATH_GPT_find_x_eq_3_plus_sqrt7_l291_29122

variable (x y : ℝ)
variable (h1 : x > y)
variable (h2 : x^2 * y^2 + x^2 + y^2 + 2 * x * y = 40)
variable (h3 : x * y + x + y = 8)

theorem find_x_eq_3_plus_sqrt7 (h1 : x > y) (h2 : x^2 * y^2 + x^2 + y^2 + 2 * x * y = 40) (h3 : x * y + x + y = 8) : 
  x = 3 + Real.sqrt 7 :=
sorry

end NUMINAMATH_GPT_find_x_eq_3_plus_sqrt7_l291_29122


namespace NUMINAMATH_GPT_joshua_miles_ratio_l291_29117

-- Definitions corresponding to conditions
def mitch_macarons : ℕ := 20
def joshua_extra : ℕ := 6
def total_kids : ℕ := 68
def macarons_per_kid : ℕ := 2

-- Variables for unspecified amounts
variable (M : ℕ) -- number of macarons Miles made

-- Calculations for Joshua and Renz's macarons based on given conditions
def joshua_macarons := mitch_macarons + joshua_extra
def renz_macarons := (3 * M) / 4 - 1

-- Total macarons calculation
def total_macarons := mitch_macarons + joshua_macarons + renz_macarons + M

-- Proof statement: Showing the ratio of number of macarons Joshua made to the number of macarons Miles made
theorem joshua_miles_ratio : (total_macarons = total_kids * macarons_per_kid) → (joshua_macarons : ℚ) / (M : ℚ) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_joshua_miles_ratio_l291_29117


namespace NUMINAMATH_GPT_max_points_of_intersection_l291_29107

open Set

def Point := ℝ × ℝ

structure Circle :=
(center : Point)
(radius : ℝ)

structure Line :=
(coeffs : ℝ × ℝ × ℝ) -- Assume line equation in the form Ax + By + C = 0

def max_intersection_points (circle : Circle) (lines : List Line) : ℕ :=
  let circle_line_intersect_count := 2
  let line_line_intersect_count := 1
  
  let number_of_lines := lines.length
  let pairwise_line_intersections := number_of_lines.choose 2
  
  let circle_and_lines_intersections := circle_line_intersect_count * number_of_lines
  let total_intersections := circle_and_lines_intersections + pairwise_line_intersections

  total_intersections

theorem max_points_of_intersection (c : Circle) (l1 l2 l3 : Line) :
  max_intersection_points c [l1, l2, l3] = 9 :=
by
  sorry

end NUMINAMATH_GPT_max_points_of_intersection_l291_29107


namespace NUMINAMATH_GPT_min_value_when_a_equals_1_range_of_a_for_f_geq_a_l291_29164

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 - 2 * a * x + 2

theorem min_value_when_a_equals_1 : 
  ∃ x, f x 1 = 1 :=
by
  sorry

theorem range_of_a_for_f_geq_a (a : ℝ) :
  (∀ x, x ≥ -1 → f x a ≥ a) ↔ (-3 ≤ a ∧ a ≤ 1) :=
by
  sorry

end NUMINAMATH_GPT_min_value_when_a_equals_1_range_of_a_for_f_geq_a_l291_29164


namespace NUMINAMATH_GPT_circumscribed_circle_radius_l291_29138

noncomputable def radius_of_circumscribed_circle 
  (a b c : ℝ) (A B C : ℝ) 
  (h1 : b = 2 * Real.sqrt 3) 
  (h2 : A + B + C = Real.pi) 
  (h3 : 2 * B = A + C) : ℝ :=
2

theorem circumscribed_circle_radius 
  {a b c A B C : ℝ} 
  (h1 : b = 2 * Real.sqrt 3) 
  (h2 : A + B + C = Real.pi) 
  (h3 : 2 * B = A + C) :
  radius_of_circumscribed_circle a b c A B C h1 h2 h3 = 2 :=
sorry

end NUMINAMATH_GPT_circumscribed_circle_radius_l291_29138


namespace NUMINAMATH_GPT_sum_squares_bound_l291_29159

theorem sum_squares_bound (a b c d : ℝ) (hpos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) (hsum : a + b + c + d = 4) : 
  ab + bc + cd + da ≤ 4 :=
sorry

end NUMINAMATH_GPT_sum_squares_bound_l291_29159


namespace NUMINAMATH_GPT_car_total_travel_time_l291_29113

-- Define the given conditions
def travel_time_ngapara_zipra : ℝ := 60
def travel_time_ningi_zipra : ℝ := 0.8 * travel_time_ngapara_zipra
def speed_limit_zone_fraction : ℝ := 0.25
def speed_reduction_factor : ℝ := 0.5
def travel_time_zipra_varnasi : ℝ := 0.75 * travel_time_ningi_zipra

-- Total adjusted travel time from Ningi to Zipra including speed limit delay
def adjusted_travel_time_ningi_zipra : ℝ :=
  let delayed_time := speed_limit_zone_fraction * travel_time_ningi_zipra * (2 - speed_reduction_factor)
  travel_time_ningi_zipra + delayed_time

-- Total travel time in the day
def total_travel_time : ℝ :=
  travel_time_ngapara_zipra + adjusted_travel_time_ningi_zipra + travel_time_zipra_varnasi

-- Proposition to prove
theorem car_total_travel_time : total_travel_time = 156 :=
by
  -- We skip the proof for now
  sorry

end NUMINAMATH_GPT_car_total_travel_time_l291_29113


namespace NUMINAMATH_GPT_find_p_q_r_sum_l291_29121

noncomputable def Q (p q r : ℝ) (v : ℂ) : Polynomial ℂ :=
  (Polynomial.C v + 2 * Polynomial.C Complex.I).comp Polynomial.X *
  (Polynomial.C v + 8 * Polynomial.C Complex.I).comp Polynomial.X *
  (Polynomial.C (3 * v - 5)).comp Polynomial.X

theorem find_p_q_r_sum (p q r : ℝ) (v : ℂ)
  (h_roots : ∃ v : ℂ, Polynomial.roots (Q p q r v) = {v + 2 * Complex.I, v + 8 * Complex.I, 3 * v - 5}) :
  (p + q + r) = -82 :=
by
  sorry

end NUMINAMATH_GPT_find_p_q_r_sum_l291_29121


namespace NUMINAMATH_GPT_equal_division_of_balls_l291_29166

def total_balls : ℕ := 10
def num_boxes : ℕ := 5
def balls_per_box : ℕ := total_balls / num_boxes

theorem equal_division_of_balls :
  balls_per_box = 2 :=
by
  sorry

end NUMINAMATH_GPT_equal_division_of_balls_l291_29166


namespace NUMINAMATH_GPT_kabulek_four_digits_l291_29160

def isKabulekNumber (N: ℕ) : Prop :=
  let a := N / 100
  let b := N % 100
  (a + b) ^ 2 = N

theorem kabulek_four_digits :
  {N : ℕ | 1000 ≤ N ∧ N < 10000 ∧ isKabulekNumber N} = {2025, 3025, 9801} :=
by sorry

end NUMINAMATH_GPT_kabulek_four_digits_l291_29160


namespace NUMINAMATH_GPT_print_pages_500_l291_29130

theorem print_pages_500 (cost_per_page cents total_dollars) : 
  cost_per_page = 3 → 
  total_dollars = 15 → 
  cents = 100 * total_dollars → 
  (cents / cost_per_page) = 500 :=
by 
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_print_pages_500_l291_29130


namespace NUMINAMATH_GPT_triangle_split_points_l291_29185

noncomputable def smallest_n_for_split (AB BC CA : ℕ) : ℕ := 
  if AB = 13 ∧ BC = 14 ∧ CA = 15 then 27 else sorry

theorem triangle_split_points (AB BC CA : ℕ) (h : AB = 13 ∧ BC = 14 ∧ CA = 15) :
  smallest_n_for_split AB BC CA = 27 :=
by
  cases h with | intro h1 h23 => sorry

-- Assertions for the explicit values provided in the conditions
example : smallest_n_for_split 13 14 15 = 27 :=
  triangle_split_points 13 14 15 ⟨rfl, rfl, rfl⟩

end NUMINAMATH_GPT_triangle_split_points_l291_29185


namespace NUMINAMATH_GPT_folding_hexagon_quadrilateral_folding_hexagon_pentagon_l291_29145

noncomputable def sum_of_interior_angles (n : ℕ) : ℝ :=
  (n - 2) * 180

theorem folding_hexagon_quadrilateral :
  (sum_of_interior_angles 4 = 360) :=
by
  sorry

theorem folding_hexagon_pentagon :
  (sum_of_interior_angles 5 = 540) :=
by
  sorry

end NUMINAMATH_GPT_folding_hexagon_quadrilateral_folding_hexagon_pentagon_l291_29145


namespace NUMINAMATH_GPT_problem_l291_29162

def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry

axiom odd_f : ∀ x, f x = -f (-x)
axiom periodic_g : ∀ x, g x = g (x + 2)
axiom f_at_neg1 : f (-1) = 3
axiom g_at_1 : g 1 = 3
axiom g_function : ∀ n : ℕ, g (2 * n * f 1) = n * f (f 1 + g (-1)) + 2

theorem problem : g (-6) + f 0 = 2 :=
by sorry

end NUMINAMATH_GPT_problem_l291_29162


namespace NUMINAMATH_GPT_domain_of_log_x_squared_sub_2x_l291_29147

theorem domain_of_log_x_squared_sub_2x (x : ℝ) : x^2 - 2 * x > 0 ↔ x < 0 ∨ x > 2 :=
by
  sorry

end NUMINAMATH_GPT_domain_of_log_x_squared_sub_2x_l291_29147


namespace NUMINAMATH_GPT_cloaks_always_short_l291_29152

-- Define the problem parameters
variables (Knights Cloaks : Type)
variables [Fintype Knights] [Fintype Cloaks]
variables (h_knights : Fintype.card Knights = 20) (h_cloaks : Fintype.card Cloaks = 20)

-- Assume every knight initially found their cloak too short
variable (too_short : Knights -> Prop)

-- Height order for knights
variable (height_order : LinearOrder Knights)
-- Length order for cloaks
variable (length_order : LinearOrder Cloaks)

-- Sorting function
noncomputable def sorted_cloaks (kn : Knights) : Cloaks := sorry

-- State that after redistribution, every knight's cloak is still too short
theorem cloaks_always_short : 
  ∀ (kn : Knights), too_short kn :=
by sorry

end NUMINAMATH_GPT_cloaks_always_short_l291_29152


namespace NUMINAMATH_GPT_sheila_monthly_savings_l291_29192

-- Define the conditions and the question in Lean
def initial_savings : ℕ := 3000
def family_contribution : ℕ := 7000
def years : ℕ := 4
def final_amount : ℕ := 23248

-- Function to calculate the monthly saving given the conditions
def monthly_savings (initial_savings family_contribution years final_amount : ℕ) : ℕ :=
  (final_amount - (initial_savings + family_contribution)) / (years * 12)

-- The theorem we need to prove in Lean
theorem sheila_monthly_savings :
  monthly_savings initial_savings family_contribution years final_amount = 276 :=
by
  sorry

end NUMINAMATH_GPT_sheila_monthly_savings_l291_29192


namespace NUMINAMATH_GPT_shepherd_initial_sheep_l291_29150

def sheep_pass_gate (sheep : ℕ) : ℕ :=
  sheep / 2 + 1

noncomputable def shepherd_sheep (initial_sheep : ℕ) : ℕ :=
  (sheep_pass_gate ∘ sheep_pass_gate ∘ sheep_pass_gate ∘ sheep_pass_gate ∘ sheep_pass_gate ∘ sheep_pass_gate) initial_sheep

theorem shepherd_initial_sheep (initial_sheep : ℕ) (h : shepherd_sheep initial_sheep = 2) :
  initial_sheep = 2 :=
sorry

end NUMINAMATH_GPT_shepherd_initial_sheep_l291_29150


namespace NUMINAMATH_GPT_gain_percent_of_cost_selling_relation_l291_29110

theorem gain_percent_of_cost_selling_relation (C S : ℕ) (h : 50 * C = 45 * S) : 
  (S > C) ∧ ((S - C) / C * 100 = 100 / 9) :=
by
  sorry

end NUMINAMATH_GPT_gain_percent_of_cost_selling_relation_l291_29110


namespace NUMINAMATH_GPT_positive_polynomial_l291_29115

theorem positive_polynomial (x : ℝ) : 3 * x ^ 2 - 6 * x + 3.5 > 0 := 
by sorry

end NUMINAMATH_GPT_positive_polynomial_l291_29115


namespace NUMINAMATH_GPT_num_positive_integers_which_make_polynomial_prime_l291_29104

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem num_positive_integers_which_make_polynomial_prime :
  (∃! n : ℕ, n > 0 ∧ is_prime (n^3 - 7 * n^2 + 18 * n - 10)) :=
sorry

end NUMINAMATH_GPT_num_positive_integers_which_make_polynomial_prime_l291_29104


namespace NUMINAMATH_GPT_least_months_for_tripling_debt_l291_29139

theorem least_months_for_tripling_debt (P : ℝ) (r : ℝ) (t : ℕ) : P = 1500 → r = 0.06 → (3 * P < P * (1 + r) ^ t) → t ≥ 20 :=
by
  intros hP hr hI
  rw [hP, hr] at hI
  norm_num at hI
  sorry

end NUMINAMATH_GPT_least_months_for_tripling_debt_l291_29139


namespace NUMINAMATH_GPT_factorize_expression_l291_29153

theorem factorize_expression (x y : ℝ) : 
  x^2 * (x + 1) - y * (x * y + x) = x * (x - y) * (x + y + 1) :=
by sorry

end NUMINAMATH_GPT_factorize_expression_l291_29153


namespace NUMINAMATH_GPT_total_number_of_coins_l291_29190

theorem total_number_of_coins {N B : ℕ} 
    (h1 : B - 2 = Nat.floor (N / 9))
    (h2 : N - 6 * (B - 3) = 3) 
    : N = 45 :=
by
  sorry

end NUMINAMATH_GPT_total_number_of_coins_l291_29190


namespace NUMINAMATH_GPT_no_fractions_satisfy_condition_l291_29157

theorem no_fractions_satisfy_condition :
  ∀ (x y : ℕ), 
    x > 0 → y > 0 → Nat.gcd x y = 1 →
    (1.2 : ℚ) * (x : ℚ) / (y : ℚ) = (x + 2 : ℚ) / (y + 2 : ℚ) →
    False :=
by
  intros x y hx hy hrel hcond
  sorry

end NUMINAMATH_GPT_no_fractions_satisfy_condition_l291_29157


namespace NUMINAMATH_GPT_problem_lean_statement_l291_29119

def P (x : ℝ) : ℝ := x^2 - 3*x - 9

theorem problem_lean_statement :
  let a := 61
  let b := 109
  let c := 621
  let d := 39
  let e := 20
  a + b + c + d + e = 850 := 
by
  sorry

end NUMINAMATH_GPT_problem_lean_statement_l291_29119


namespace NUMINAMATH_GPT_power_multiplication_equals_result_l291_29177

theorem power_multiplication_equals_result : 
  (-0.25)^11 * (-4)^12 = -4 := 
sorry

end NUMINAMATH_GPT_power_multiplication_equals_result_l291_29177


namespace NUMINAMATH_GPT_six_digit_pair_divisibility_l291_29161

theorem six_digit_pair_divisibility (a b : ℕ) (ha : 100000 ≤ a ∧ a < 1000000) (hb : 100000 ≤ b ∧ b < 1000000) :
  ((1000000 * a + b) % (a * b) = 0) ↔ (a = 166667 ∧ b = 333334) ∨ (a = 500001 ∧ b = 500001) :=
by sorry

end NUMINAMATH_GPT_six_digit_pair_divisibility_l291_29161


namespace NUMINAMATH_GPT_fifth_term_geometric_sequence_l291_29102

theorem fifth_term_geometric_sequence (x y : ℚ) (h1 : x = 2) (h2 : y = 1) :
    let a1 := x + y
    let a2 := x - y
    let a3 := x / y
    let a4 := x * y
    let r := (x - y)/(x + y)
    (a4 * r = (2 / 3)) :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_fifth_term_geometric_sequence_l291_29102


namespace NUMINAMATH_GPT_cos_5theta_l291_29137

theorem cos_5theta (theta : ℝ) (h : Real.cos theta = 2 / 5) : Real.cos (5 * theta) = 2762 / 3125 := 
sorry

end NUMINAMATH_GPT_cos_5theta_l291_29137


namespace NUMINAMATH_GPT_prove_area_and_sum_l291_29105

-- Define the coordinates of the vertices of the quadrilateral.
variables (a b : ℤ)

-- Define the non-computable requirements related to the problem.
noncomputable def problem_statement : Prop :=
  ∃ (a b : ℤ), a > 0 ∧ b > 0 ∧ a > b ∧ (4 * a * b = 32) ∧ (a + b = 5)

theorem prove_area_and_sum : problem_statement := 
sorry

end NUMINAMATH_GPT_prove_area_and_sum_l291_29105


namespace NUMINAMATH_GPT_weightlifter_one_hand_l291_29114

theorem weightlifter_one_hand (total_weight : ℕ) (h : total_weight = 20) (even_distribution : total_weight % 2 = 0) : total_weight / 2 = 10 :=
by
  sorry

end NUMINAMATH_GPT_weightlifter_one_hand_l291_29114


namespace NUMINAMATH_GPT_deduce_pi_from_cylinder_volume_l291_29186

theorem deduce_pi_from_cylinder_volume 
  (C h V : ℝ) 
  (Circumference : C = 20) 
  (Height : h = 11)
  (VolumeFormula : V = (1 / 12) * C^2 * h) : 
  pi = 3 :=
by 
  -- Carry out the proof
  sorry

end NUMINAMATH_GPT_deduce_pi_from_cylinder_volume_l291_29186


namespace NUMINAMATH_GPT_probability_three_fair_coins_l291_29103

noncomputable def probability_one_head_two_tails (n : ℕ) : ℚ :=
  if n = 3 then 3 / 8 else 0

theorem probability_three_fair_coins :
  probability_one_head_two_tails 3 = 3 / 8 :=
by
  sorry

end NUMINAMATH_GPT_probability_three_fair_coins_l291_29103


namespace NUMINAMATH_GPT_human_height_weight_correlated_l291_29163

-- Define the relationships as types
def taxiFareDistanceRelated : Prop := ∀ x y : ℕ, x = y → True
def houseSizePriceRelated : Prop := ∀ x y : ℕ, x = y → True
def humanHeightWeightCorrelated : Prop := ∃ k : ℕ, ∀ x y : ℕ, x / k = y
def ironBlockMassRelated : Prop := ∀ x y : ℕ, x = y → True

-- Main theorem statement
theorem human_height_weight_correlated : humanHeightWeightCorrelated :=
  sorry

end NUMINAMATH_GPT_human_height_weight_correlated_l291_29163


namespace NUMINAMATH_GPT_solve_for_s_l291_29188

theorem solve_for_s (r s : ℝ) (h1 : 1 < r) (h2 : r < s) (h3 : 1 / r + 1 / s = 3 / 4) (h4 : r * s = 8) : s = 4 :=
sorry

end NUMINAMATH_GPT_solve_for_s_l291_29188


namespace NUMINAMATH_GPT_find_fifth_day_income_l291_29127

-- Define the incomes for the first four days
def income_day1 := 45
def income_day2 := 50
def income_day3 := 60
def income_day4 := 65

-- Define the average income over five days
def average_income := 58

-- Expressing the question in terms of a function to determine the fifth day's income
theorem find_fifth_day_income : 
  ∃ (income_day5 : ℕ), 
    (income_day1 + income_day2 + income_day3 + income_day4 + income_day5) / 5 = average_income 
    ∧ income_day5 = 70 :=
sorry

end NUMINAMATH_GPT_find_fifth_day_income_l291_29127


namespace NUMINAMATH_GPT_perimeter_of_face_given_volume_l291_29197

-- Definitions based on conditions
def volume_of_cube (v : ℝ) := v = 512

def side_of_cube (s : ℝ) := s^3 = 512

def perimeter_of_face (p s : ℝ) := p = 4 * s

-- Lean 4 statement: prove that the perimeter of one face of the cube is 32 cm given the volume is 512 cm³.
theorem perimeter_of_face_given_volume :
  ∃ s : ℝ, volume_of_cube (s^3) ∧ perimeter_of_face 32 s :=
by sorry

end NUMINAMATH_GPT_perimeter_of_face_given_volume_l291_29197


namespace NUMINAMATH_GPT_temperature_difference_l291_29168

theorem temperature_difference (T_south T_north : ℝ) (h_south : T_south = 6) (h_north : T_north = -3) :
  T_south - T_north = 9 :=
by 
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_temperature_difference_l291_29168


namespace NUMINAMATH_GPT_at_least_one_less_than_equal_one_l291_29148

theorem at_least_one_less_than_equal_one
  (x y z : ℝ) 
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (hxyz : x + y + z = 3) : 
  x * (x + y - z) ≤ 1 ∨ y * (y + z - x) ≤ 1 ∨ z * (z + x - y) ≤ 1 := 
by 
  sorry

end NUMINAMATH_GPT_at_least_one_less_than_equal_one_l291_29148


namespace NUMINAMATH_GPT_average_rainfall_correct_l291_29176

-- Definitions based on given conditions
def total_rainfall : ℚ := 420 -- inches
def days_in_august : ℕ := 31
def hours_in_a_day : ℕ := 24

-- Defining total hours in August
def total_hours_in_august : ℕ := days_in_august * hours_in_a_day

-- The average rainfall in inches per hour
def average_rainfall_per_hour : ℚ := total_rainfall / total_hours_in_august

-- The statement to prove
theorem average_rainfall_correct :
  average_rainfall_per_hour = 420 / 744 :=
by
  sorry

end NUMINAMATH_GPT_average_rainfall_correct_l291_29176


namespace NUMINAMATH_GPT_range_of_m_l291_29155

open Real Set

def P (m : ℝ) := |m + 1| ≤ 2
def Q (m : ℝ) := ∃ x : ℝ, x^2 - m*x + 1 = 0 ∧ (m^2 - 4 ≥ 0)

theorem range_of_m (m : ℝ) :
  (¬¬ P m ∧ ¬ (P m ∧ Q m)) → -2 < m ∧ m ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l291_29155


namespace NUMINAMATH_GPT_sum_powers_div_5_iff_l291_29165

theorem sum_powers_div_5_iff (n : ℕ) (h : n > 0) : (1^n + 2^n + 3^n + 4^n) % 5 = 0 ↔ n % 4 ≠ 0 := 
sorry

end NUMINAMATH_GPT_sum_powers_div_5_iff_l291_29165


namespace NUMINAMATH_GPT_horse_running_time_l291_29132

def area_of_square_field : Real := 625
def speed_of_horse_around_field : Real := 25

theorem horse_running_time : (4 : Real) = 
  let side_length := Real.sqrt area_of_square_field
  let perimeter := 4 * side_length
  perimeter / speed_of_horse_around_field :=
by
  sorry

end NUMINAMATH_GPT_horse_running_time_l291_29132


namespace NUMINAMATH_GPT_radius_of_circle_through_points_l291_29142

theorem radius_of_circle_through_points : 
  ∃ (x : ℝ), 
  (dist (x, 0) (2, 5) = dist (x, 0) (3, 4)) →
  (∃ (r : ℝ), r = dist (x, 0) (2, 5) ∧ r = 5) :=
by
  sorry

end NUMINAMATH_GPT_radius_of_circle_through_points_l291_29142


namespace NUMINAMATH_GPT_leo_total_points_l291_29124

theorem leo_total_points (x y : ℕ) (h1 : x + y = 50) :
  0.4 * (x : ℝ) * 3 + 0.5 * (y : ℝ) * 2 = 0.2 * (x : ℝ) + 50 :=
by sorry

end NUMINAMATH_GPT_leo_total_points_l291_29124


namespace NUMINAMATH_GPT_binom_9_5_eq_126_l291_29195

theorem binom_9_5_eq_126 : Nat.choose 9 5 = 126 := by
  sorry

end NUMINAMATH_GPT_binom_9_5_eq_126_l291_29195


namespace NUMINAMATH_GPT_largest_result_among_expressions_l291_29128

def E1 : ℕ := 992 * 999 + 999
def E2 : ℕ := 993 * 998 + 998
def E3 : ℕ := 994 * 997 + 997
def E4 : ℕ := 995 * 996 + 996

theorem largest_result_among_expressions : E4 > E1 ∧ E4 > E2 ∧ E4 > E3 :=
by sorry

end NUMINAMATH_GPT_largest_result_among_expressions_l291_29128


namespace NUMINAMATH_GPT_exists_valid_circle_group_l291_29182

variable {P : Type}
variable (knows : P → P → Prop)

def knows_at_least_three (p : P) : Prop :=
  ∃ (p₁ p₂ p₃ : P), p₁ ≠ p ∧ p₂ ≠ p ∧ p₃ ≠ p ∧ knows p p₁ ∧ knows p p₂ ∧ knows p p₃

def valid_circle_group (G : List P) : Prop :=
  (2 < G.length) ∧ (G.length % 2 = 0) ∧ (∀ i, knows (G.nthLe i sorry) (G.nthLe ((i + 1) % G.length) sorry) ∧ knows (G.nthLe i sorry) (G.nthLe ((i - 1 + G.length) % G.length) sorry))

theorem exists_valid_circle_group (H : ∀ p : P, knows_at_least_three knows p) : 
  ∃ G : List P, valid_circle_group knows G := 
sorry

end NUMINAMATH_GPT_exists_valid_circle_group_l291_29182


namespace NUMINAMATH_GPT_total_people_can_ride_l291_29196

theorem total_people_can_ride (num_people_per_teacup : Nat) (num_teacups : Nat) (h1 : num_people_per_teacup = 9) (h2 : num_teacups = 7) : num_people_per_teacup * num_teacups = 63 := by
  sorry

end NUMINAMATH_GPT_total_people_can_ride_l291_29196


namespace NUMINAMATH_GPT_compute_expression_l291_29151

open Real

theorem compute_expression : 
  sqrt (1 / 4) * sqrt 16 - (sqrt (1 / 9))⁻¹ - sqrt 0 + sqrt (45 / 5) = 2 := 
by
  -- The proof details would go here, but they are omitted.
  sorry

end NUMINAMATH_GPT_compute_expression_l291_29151


namespace NUMINAMATH_GPT_ratio_sequences_l291_29194

-- Define positive integers n and k, with k >= n and k - n even.
variables {n k : ℕ} (h_k_ge_n : k ≥ n) (h_even : (k - n) % 2 = 0)

-- Define the sets S_N and S_M
def S_N (n k : ℕ) : ℕ := sorry -- Placeholder for the cardinality of S_N
def S_M (n k : ℕ) : ℕ := sorry -- Placeholder for the cardinality of S_M

-- Main theorem: N / M = 2^(k - n)
theorem ratio_sequences (h_k_ge_n : k ≥ n) (h_even : (k - n) % 2 = 0) :
  (S_N n k : ℝ) / (S_M n k : ℝ) = 2^(k - n) := sorry

end NUMINAMATH_GPT_ratio_sequences_l291_29194


namespace NUMINAMATH_GPT_solution_set_for_inequality_l291_29144

noncomputable def odd_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = -f (x)

noncomputable def decreasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
∀ ⦃x y⦄, x ∈ s → y ∈ s → x < y → f x > f y

theorem solution_set_for_inequality
  (f : ℝ → ℝ)
  (h_odd : odd_function f)
  (h_decreasing : decreasing_on f (Set.Iio 0))
  (h_f1 : f 1 = 0) :
  {x : ℝ | x^3 * f x > 0} = {x : ℝ | x > 1 ∨ x < -1} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_for_inequality_l291_29144


namespace NUMINAMATH_GPT_find_quotient_l291_29109

noncomputable def matrix_a : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![2, 3], ![4, 5]]

noncomputable def matrix_b (a b c d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![a, b], ![c, d]]

theorem find_quotient (a b c d : ℝ) (H1 : matrix_a * (matrix_b a b c d) = (matrix_b a b c d) * matrix_a)
  (H2 : 2*b ≠ 3*c) : ((a - d) / (c - 2*b)) = 3 / 2 :=
  sorry

end NUMINAMATH_GPT_find_quotient_l291_29109


namespace NUMINAMATH_GPT_smallest_feared_sequence_l291_29131

def is_feared (n : ℕ) : Prop :=
  -- This function checks if a number contains '13' as a contiguous substring.
  sorry

def is_fearless (n : ℕ) : Prop := ¬is_feared n

theorem smallest_feared_sequence : ∃ (n : ℕ) (a : ℕ), 0 < n ∧ a < 100 ∧ is_fearless n ∧ is_fearless (n + 10 * a) ∧ 
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ 9 → is_feared (n + k * a)) ∧ n = 1287 := 
by
  sorry

end NUMINAMATH_GPT_smallest_feared_sequence_l291_29131


namespace NUMINAMATH_GPT_rocky_first_round_knockouts_l291_29126

theorem rocky_first_round_knockouts
  (total_fights : ℕ)
  (knockout_percentage : ℝ)
  (first_round_knockout_percentage : ℝ)
  (h1 : total_fights = 190)
  (h2 : knockout_percentage = 0.50)
  (h3 : first_round_knockout_percentage = 0.20) :
  (total_fights * knockout_percentage * first_round_knockout_percentage = 19) := 
by
  sorry

end NUMINAMATH_GPT_rocky_first_round_knockouts_l291_29126


namespace NUMINAMATH_GPT_volume_of_alcohol_correct_l291_29172

noncomputable def radius := 3 / 2 -- radius of the tank
noncomputable def total_height := 9 -- total height of the tank
noncomputable def full_solution_height := total_height / 3 -- height of the liquid when the tank is one-third full
noncomputable def volume := Real.pi * radius^2 * full_solution_height -- volume of liquid in the tank
noncomputable def alcohol_ratio := 1 / 6 -- ratio of alcohol to the total solution
noncomputable def volume_of_alcohol := volume * alcohol_ratio -- volume of alcohol in the tank

theorem volume_of_alcohol_correct : volume_of_alcohol = (9 / 8) * Real.pi :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_volume_of_alcohol_correct_l291_29172


namespace NUMINAMATH_GPT_chiquita_height_l291_29193

theorem chiquita_height (C : ℝ) :
  (C + (C + 2) = 12) → (C = 5) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_chiquita_height_l291_29193


namespace NUMINAMATH_GPT_minimum_choir_members_l291_29189

theorem minimum_choir_members:
  ∃ n : ℕ, (n % 9 = 0) ∧ (n % 10 = 0) ∧ (n % 11 = 0) ∧ (∀ m : ℕ, (m % 9 = 0) ∧ (m % 10 = 0) ∧ (m % 11 = 0) → n ≤ m) → n = 990 :=
by
  sorry

end NUMINAMATH_GPT_minimum_choir_members_l291_29189


namespace NUMINAMATH_GPT_solve_equation_l291_29146

theorem solve_equation :
  let lhs := ((4 - 3.5 * (15/7 - 6/5)) / 0.16)
  let rhs := ((23/7 - (3/14) / (1/6)) / (3467/84 - 2449/60))
  lhs / 1 = rhs :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l291_29146


namespace NUMINAMATH_GPT_taxi_ride_cost_l291_29191

-- Lean statement
theorem taxi_ride_cost (base_fare : ℝ) (rate1 : ℝ) (rate1_miles : ℝ) (rate2 : ℝ) (total_miles : ℝ) 
  (h_base_fare : base_fare = 2.00)
  (h_rate1 : rate1 = 0.30)
  (h_rate1_miles : rate1_miles = 3)
  (h_rate2 : rate2 = 0.40)
  (h_total_miles : total_miles = 8) :
  let rate1_cost := rate1 * rate1_miles
  let rate2_cost := rate2 * (total_miles - rate1_miles)
  base_fare + rate1_cost + rate2_cost = 4.90 := by
  sorry

end NUMINAMATH_GPT_taxi_ride_cost_l291_29191


namespace NUMINAMATH_GPT_pandas_increase_l291_29167

theorem pandas_increase 
  (C P : ℕ) -- C: Number of cheetahs 5 years ago, P: Number of pandas 5 years ago
  (h_ratio_5_years_ago : C / P = 1 / 3)
  (h_cheetahs_increase : ∃ z : ℕ, z = 2)
  (h_ratio_now : ∃ k : ℕ, (C + k) / (P + x) = 1 / 3) :
  x = 6 :=
by
  sorry

end NUMINAMATH_GPT_pandas_increase_l291_29167


namespace NUMINAMATH_GPT_common_ratio_of_geometric_sequence_l291_29133

theorem common_ratio_of_geometric_sequence (a₁ : ℝ) (S : ℕ → ℝ) (q : ℝ) (h₁ : ∀ n, S (n + 1) = S n + a₁ * q ^ n) (h₂ : 2 * S n = S (n + 1) + S (n + 2)) :
  q = -2 :=
by
  sorry

end NUMINAMATH_GPT_common_ratio_of_geometric_sequence_l291_29133


namespace NUMINAMATH_GPT_buses_needed_for_trip_l291_29141

theorem buses_needed_for_trip :
  ∀ (total_students students_in_vans bus_capacity : ℕ),
  total_students = 500 →
  students_in_vans = 56 →
  bus_capacity = 45 →
  ⌈(total_students - students_in_vans : ℝ) / bus_capacity⌉ = 10 :=
by
  sorry

end NUMINAMATH_GPT_buses_needed_for_trip_l291_29141


namespace NUMINAMATH_GPT_line_equation_problem_l291_29175

theorem line_equation_problem
  (P : ℝ × ℝ)
  (h1 : (P.1 + P.2 - 2 = 0) ∧ (P.1 - P.2 + 4 = 0))
  (l : ℝ × ℝ → Prop)
  (h2 : ∀ A B : ℝ × ℝ, l A → l B → (∃ k, B.2 - A.2 = k * (B.1 - A.1)))
  (h3 : ∀ Q : ℝ × ℝ, l Q → (3 * Q.1 - 2 * Q.2 + 4 = 0)) :
  l P ↔ 3 * P.1 - 2 * P.2 + 9 = 0 := 
sorry

end NUMINAMATH_GPT_line_equation_problem_l291_29175


namespace NUMINAMATH_GPT_complement_intersection_l291_29116

open Set

variable (I : Set ℕ) (A B : Set ℕ)

-- Given the universal set and specific sets A and B
def universal_set : Set ℕ := {1,2,3,4,5}
def set_A : Set ℕ := {2,3,5}
def set_B : Set ℕ := {1,2}

-- To prove that the complement of B in I intersects A to be {3,5}
theorem complement_intersection :
  (universal_set \ set_B) ∩ set_A = {3,5} :=
sorry

end NUMINAMATH_GPT_complement_intersection_l291_29116


namespace NUMINAMATH_GPT_number_of_slices_left_l291_29181

-- Conditions
def total_slices : ℕ := 8
def slices_given_to_joe_and_darcy : ℕ := total_slices / 2
def slices_given_to_carl : ℕ := total_slices / 4

-- Question: How many slices were left?
def slices_left : ℕ := total_slices - (slices_given_to_joe_and_darcy + slices_given_to_carl)

-- Proof statement to demonstrate that slices_left == 2
theorem number_of_slices_left : slices_left = 2 := by
  sorry

end NUMINAMATH_GPT_number_of_slices_left_l291_29181


namespace NUMINAMATH_GPT_sum_of_three_squares_not_divisible_by_3_l291_29111

theorem sum_of_three_squares_not_divisible_by_3
    (N : ℕ) (n : ℕ) (a b c : ℤ) 
    (h1 : N = 9^n * (a^2 + b^2 + c^2))
    (h2 : ∃ (a1 b1 c1 : ℤ), a = 3 * a1 ∧ b = 3 * b1 ∧ c = 3 * c1) :
    ∃ (k m n : ℤ), N = k^2 + m^2 + n^2 ∧ (¬ (3 ∣ k ∧ 3 ∣ m ∧ 3 ∣ n)) :=
sorry

end NUMINAMATH_GPT_sum_of_three_squares_not_divisible_by_3_l291_29111


namespace NUMINAMATH_GPT_arithmetic_sequence_S7_eq_28_l291_29158

/--
Given the arithmetic sequence \( \{a_n\} \) and the sum of its first \( n \) terms is \( S_n \),
if \( a_3 + a_4 + a_5 = 12 \), then prove \( S_7 = 28 \).
-/
theorem arithmetic_sequence_S7_eq_28
  (a : ℕ → ℤ) -- Sequence a_n
  (S : ℕ → ℤ) -- Sum sequence S_n
  (h1 : a 3 + a 4 + a 5 = 12)
  (h2 : ∀ n, S n = n * (a 1 + a n) / 2) -- Sum formula
  : S 7 = 28 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_S7_eq_28_l291_29158


namespace NUMINAMATH_GPT_decreasing_function_condition_l291_29106

theorem decreasing_function_condition :
  (∀ (x1 x2 : ℝ), 0 < x1 → 0 < x2 → x1 ≠ x2 → (x1 - x2) * ((1 / x1 - x1) - (1 / x2 - x2)) < 0) :=
by
  -- Proof outline goes here
  sorry

end NUMINAMATH_GPT_decreasing_function_condition_l291_29106


namespace NUMINAMATH_GPT_trig_identity_l291_29173

theorem trig_identity (x : ℝ) (h : 2 * Real.cos x - 3 * Real.sin x = 4) : 
  2 * Real.sin x + 3 * Real.cos x = 1 ∨ 2 * Real.sin x + 3 * Real.cos x = 3 :=
sorry

end NUMINAMATH_GPT_trig_identity_l291_29173


namespace NUMINAMATH_GPT_part1_positive_root_part2_negative_solution_l291_29100

theorem part1_positive_root (x k : ℝ) (hx1 : x > 0)
  (h : 4 / (x + 1) + 3 / (x - 1) = k / (x^2 - 1)) : 
  k = 6 ∨ k = -8 := 
sorry

theorem part2_negative_solution (x k : ℝ) (hx2 : x < 0)
  (hx_ne1 : x ≠ 1) (hx_ne_neg1 : x ≠ -1)
  (h : 4 / (x + 1) + 3 / (x - 1) = k / (x^2 - 1)) : 
  k < -1 ∧ k ≠ -8 := 
sorry

end NUMINAMATH_GPT_part1_positive_root_part2_negative_solution_l291_29100


namespace NUMINAMATH_GPT_cost_per_can_of_tuna_l291_29101

theorem cost_per_can_of_tuna
  (num_cans : ℕ) -- condition 1
  (num_coupons : ℕ) -- condition 2
  (coupon_discount_cents : ℕ) -- condition 2 detail
  (amount_paid_dollars : ℚ) -- condition 3
  (change_received_dollars : ℚ) -- condition 3 detail
  (cost_per_can_cents: ℚ) : -- the quantity we want to prove
  num_cans = 9 →
  num_coupons = 5 →
  coupon_discount_cents = 25 →
  amount_paid_dollars = 20 →
  change_received_dollars = 5.5 →
  cost_per_can_cents = 175 :=
by
  intros hn hc hcd hap hcr
  sorry

end NUMINAMATH_GPT_cost_per_can_of_tuna_l291_29101


namespace NUMINAMATH_GPT_hole_depth_l291_29123

theorem hole_depth (height : ℝ) (half_depth : ℝ) (total_depth : ℝ) 
    (h_height : height = 90) 
    (h_half_depth : half_depth = total_depth / 2)
    (h_position : height + half_depth = total_depth - height) : 
    total_depth = 120 := 
by
    sorry

end NUMINAMATH_GPT_hole_depth_l291_29123


namespace NUMINAMATH_GPT_viewers_difference_l291_29170

theorem viewers_difference :
  let second_game := 80
  let first_game := second_game - 20
  let third_game := second_game + 15
  let fourth_game := third_game + (third_game / 10)
  let total_last_week := 350
  let total_this_week := first_game + second_game + third_game + fourth_game
  total_this_week - total_last_week = -10 := 
by
  sorry

end NUMINAMATH_GPT_viewers_difference_l291_29170


namespace NUMINAMATH_GPT_min_value_of_function_l291_29198

noncomputable def f (x : ℝ) : ℝ := 3 * x + 12 / x^2

theorem min_value_of_function :
  ∀ x > 0, f x ≥ 9 :=
by
  intro x hx_pos
  sorry

end NUMINAMATH_GPT_min_value_of_function_l291_29198


namespace NUMINAMATH_GPT_perpendicular_lines_l291_29169

theorem perpendicular_lines (a : ℝ) :
  (∃ (m₁ m₂ : ℝ), ((a + 1) * m₁ + a * m₂ = 0) ∧ 
                  (a * m₁ + 2 * m₂ = 1) ∧ 
                  m₁ * m₂ = -1) ↔ (a = 0 ∨ a = -3) := 
sorry

end NUMINAMATH_GPT_perpendicular_lines_l291_29169


namespace NUMINAMATH_GPT_solve_for_x_l291_29178

theorem solve_for_x 
  (a b : ℝ)
  (h1 : ∃ P : ℝ × ℝ, P = (3, -1) ∧ (P.2 = 3 + b) ∧ (P.2 = a * 3 + 2)) :
  (a - 1) * 3 = b - 2 :=
by sorry

end NUMINAMATH_GPT_solve_for_x_l291_29178


namespace NUMINAMATH_GPT_product_divisible_by_3_or_5_l291_29112

theorem product_divisible_by_3_or_5 {a b c d : ℕ} (h : Nat.lcm (Nat.lcm (Nat.lcm a b) c) d = a + b + c + d) :
  (a * b * c * d) % 3 = 0 ∨ (a * b * c * d) % 5 = 0 :=
by
  sorry

end NUMINAMATH_GPT_product_divisible_by_3_or_5_l291_29112


namespace NUMINAMATH_GPT_find_a7_l291_29184

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem find_a7 (a : ℕ → ℝ) (h_geom : geometric_sequence a)
  (h3 : a 3 = 1)
  (h_det : a 6 * a 8 - 8 * 8 = 0) :
  a 7 = 8 :=
sorry

end NUMINAMATH_GPT_find_a7_l291_29184


namespace NUMINAMATH_GPT_find_sum_lent_l291_29108

theorem find_sum_lent (r t : ℝ) (I : ℝ) (P : ℝ) (h1: r = 0.06) (h2 : t = 8) (h3 : I = P - 520) (h4: I = P * r * t) : P = 1000 := by
  sorry

end NUMINAMATH_GPT_find_sum_lent_l291_29108


namespace NUMINAMATH_GPT_longer_trip_due_to_red_lights_l291_29129

theorem longer_trip_due_to_red_lights :
  ∀ (num_lights : ℕ) (green_time first_route_base_time red_time_per_light second_route_time : ℕ),
  num_lights = 3 →
  first_route_base_time = 10 →
  red_time_per_light = 3 →
  second_route_time = 14 →
  (first_route_base_time + num_lights * red_time_per_light) - second_route_time = 5 :=
by
  intros num_lights green_time first_route_base_time red_time_per_light second_route_time
  sorry

end NUMINAMATH_GPT_longer_trip_due_to_red_lights_l291_29129


namespace NUMINAMATH_GPT_least_positive_a_exists_l291_29120

noncomputable def f (x a : ℤ) : ℤ := 5 * x ^ 13 + 13 * x ^ 5 + 9 * a * x

theorem least_positive_a_exists :
  ∃ a : ℕ, (∀ x : ℤ, 65 ∣ f x a) ∧ ∀ b : ℕ, (∀ x : ℤ, 65 ∣ f x b) → a ≤ b :=
sorry

end NUMINAMATH_GPT_least_positive_a_exists_l291_29120


namespace NUMINAMATH_GPT_sum_of_box_dimensions_l291_29156

theorem sum_of_box_dimensions (X Y Z : ℝ) (h1 : X * Y = 32) (h2 : X * Z = 50) (h3 : Y * Z = 80) :
  X + Y + Z = 25.5 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_GPT_sum_of_box_dimensions_l291_29156


namespace NUMINAMATH_GPT_no_negative_roots_of_polynomial_l291_29134

def polynomial (x : ℝ) := x^4 - 5 * x^3 - 4 * x^2 - 7 * x + 4

theorem no_negative_roots_of_polynomial :
  ¬ ∃ (x : ℝ), x < 0 ∧ polynomial x = 0 :=
by
  sorry

end NUMINAMATH_GPT_no_negative_roots_of_polynomial_l291_29134


namespace NUMINAMATH_GPT_proof_of_acdb_l291_29143

theorem proof_of_acdb
  (x a b c d : ℤ)
  (hx_eq : 7 * x - 8 * x = 20)
  (hx_form : (a + b * Real.sqrt c) / d = x)
  (hints : x = (4 + 2 * Real.sqrt 39) / 7)
  (int_cond : a = 4 ∧ b = 2 ∧ c = 39 ∧ d = 7) :
  a * c * d / b = 546 := by
sorry

end NUMINAMATH_GPT_proof_of_acdb_l291_29143


namespace NUMINAMATH_GPT_product_of_digits_l291_29154

theorem product_of_digits (A B : ℕ) (h1 : A + B = 13) (h2 : (10 * A + B) % 4 = 0) : A * B = 42 :=
by
  sorry

end NUMINAMATH_GPT_product_of_digits_l291_29154


namespace NUMINAMATH_GPT_polynomial_factor_l291_29199

def factorization_condition (p q : ℤ) : Prop :=
  ∃ r s : ℤ, 
    p = 4 * r ∧ 
    q = -3 * r + 4 * s ∧ 
    40 = 2 * r - 3 * s + 16 ∧ 
    -20 = s - 12

theorem polynomial_factor (p q : ℤ) (hpq : factorization_condition p q) : (p, q) = (0, -32) :=
by sorry

end NUMINAMATH_GPT_polynomial_factor_l291_29199


namespace NUMINAMATH_GPT_sqrt_sine_tan_domain_l291_29136

open Real

noncomputable def domain_sqrt_sine_tan : Set ℝ :=
  {x | ∃ (k : ℤ), (-π / 2 + 2 * k * π < x ∧ x < π / 2 + 2 * k * π) ∨ x = k * π}

theorem sqrt_sine_tan_domain (x : ℝ) :
  (sin x * tan x ≥ 0) ↔ x ∈ domain_sqrt_sine_tan :=
by
  sorry

end NUMINAMATH_GPT_sqrt_sine_tan_domain_l291_29136


namespace NUMINAMATH_GPT_find_a_8_l291_29140

variable {α : Type*} [LinearOrderedField α]
variables (a : ℕ → α) (n : ℕ)

-- Definition of an arithmetic sequence
def is_arithmetic_seq (a : ℕ → α) := ∃ d : α, ∀ n : ℕ, a (n + 1) = a n + d

-- Given condition
def given_condition (a : ℕ → α) : Prop :=
  a 4 + a 6 + a 8 + a 10 + a 12 = 120

-- Main theorem to prove
theorem find_a_8 (h_arith : is_arithmetic_seq a) (h_cond : given_condition a) : a 8 = 24 :=
  sorry

end NUMINAMATH_GPT_find_a_8_l291_29140


namespace NUMINAMATH_GPT_volume_small_pyramid_eq_27_60_l291_29125

noncomputable def volume_of_smaller_pyramid (base_edge : ℝ) (slant_edge : ℝ) (height_above_base : ℝ) : ℝ :=
  let total_height := Real.sqrt ((slant_edge ^ 2) - ((base_edge / (2 * Real.sqrt 2)) ^ 2))
  let smaller_pyramid_height := total_height - height_above_base
  let scale_factor := (smaller_pyramid_height / total_height)
  let new_base_edge := base_edge * scale_factor
  let new_base_area := (new_base_edge ^ 2) * 2
  (1 / 3) * new_base_area * smaller_pyramid_height

theorem volume_small_pyramid_eq_27_60 :
  volume_of_smaller_pyramid (10 * Real.sqrt 2) 12 4 = 27.6 :=
by
  sorry

end NUMINAMATH_GPT_volume_small_pyramid_eq_27_60_l291_29125


namespace NUMINAMATH_GPT_count_distinct_four_digit_numbers_ending_in_25_l291_29171

-- Define what it means for a number to be a four-digit number according to the conditions in (a).
def is_four_digit_number (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

-- Define what it means for a number to be divisible by 5 and end in 25 according to the conditions in (a).
def ends_in_25 (n : ℕ) : Prop := n % 100 = 25

-- Define the problem as a theorem in Lean
theorem count_distinct_four_digit_numbers_ending_in_25 : 
  ∃ (count : ℕ), count = 90 ∧ 
    (∀ n, is_four_digit_number n ∧ ends_in_25 n → n % 5 = 0) :=
by
  sorry

end NUMINAMATH_GPT_count_distinct_four_digit_numbers_ending_in_25_l291_29171


namespace NUMINAMATH_GPT_completing_square_result_l291_29180

theorem completing_square_result : ∀ x : ℝ, (x^2 - 4 * x - 1 = 0) → ((x - 2) ^ 2 = 5) :=
by
  intro x h
  sorry

end NUMINAMATH_GPT_completing_square_result_l291_29180


namespace NUMINAMATH_GPT_coeff_of_linear_term_l291_29135

def quadratic_eqn (x : ℝ) : ℝ := x^2 - 2*x + 3

theorem coeff_of_linear_term :
  ∀ (x : ℝ), (quadratic_eqn x = 0) → (∃ c_b : ℝ, quadratic_eqn x = x^2 + c_b * x + 3 ∧ c_b = -2) :=
by
  sorry

end NUMINAMATH_GPT_coeff_of_linear_term_l291_29135


namespace NUMINAMATH_GPT_container_capacity_l291_29118

theorem container_capacity 
  (C : ℝ)
  (h1 : 0.75 * C - 0.30 * C = 45) :
  C = 100 := by
  sorry

end NUMINAMATH_GPT_container_capacity_l291_29118


namespace NUMINAMATH_GPT_batsman_average_after_17th_inning_l291_29187

theorem batsman_average_after_17th_inning (A : ℝ) (h1 : 16 * A + 200 = 17 * (A + 10)) : 
  A + 10 = 40 := 
by
  sorry

end NUMINAMATH_GPT_batsman_average_after_17th_inning_l291_29187


namespace NUMINAMATH_GPT_largest_sum_of_two_largest_angles_of_EFGH_l291_29174

theorem largest_sum_of_two_largest_angles_of_EFGH (x d : ℝ) (y z : ℝ) :
  (∃ a b : ℝ, a + 2 * b = x + 70 ∧ a + b = 70 ∧ 2 * a + 3 * b = 180) ∧
  (2 * x + 3 * d = 180) ∧ (x = 30) ∧ (y = 70) ∧ (z = 100) ∧ (z + 70 = x + d) ∧
  x + d + x + 2 * d + x + 3 * d + x = 360 →
  max (70 + y) (70 + z) + max (y + 70) (z + 70) = 210 := 
sorry

end NUMINAMATH_GPT_largest_sum_of_two_largest_angles_of_EFGH_l291_29174


namespace NUMINAMATH_GPT_ratio_of_areas_l291_29183

noncomputable def side_length_WXYZ : ℝ := 16

noncomputable def WJ : ℝ := (3/4) * side_length_WXYZ
noncomputable def JX : ℝ := (1/4) * side_length_WXYZ

noncomputable def side_length_JKLM := 4 * Real.sqrt 2

noncomputable def area_JKLM := (side_length_JKLM)^2
noncomputable def area_WXYZ := (side_length_WXYZ)^2

theorem ratio_of_areas : area_JKLM / area_WXYZ = 1 / 8 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_areas_l291_29183
