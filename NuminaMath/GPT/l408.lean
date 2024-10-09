import Mathlib

namespace pascal_sum_of_squares_of_interior_l408_40877

theorem pascal_sum_of_squares_of_interior (eighth_row_interior : List ℕ) 
    (h : eighth_row_interior = [7, 21, 35, 35, 21, 7]) : 
    (eighth_row_interior.map (λ x => x * x)).sum = 3430 := 
by
  sorry

end pascal_sum_of_squares_of_interior_l408_40877


namespace investment_duration_p_l408_40862

-- Given the investments ratio, profits ratio, and time period for q,
-- proving the time period of p's investment is 7 months.
theorem investment_duration_p (T_p T_q : ℕ) 
  (investment_ratio : 7 * T_p = 5 * T_q) 
  (profit_ratio : 7 * T_p / T_q = 7 / 10)
  (T_q_eq : T_q = 14) : T_p = 7 :=
by
  sorry

end investment_duration_p_l408_40862


namespace diamond_cut_1_3_loss_diamond_max_loss_ratio_l408_40820

noncomputable def value (w : ℝ) : ℝ := 6000 * w^2

theorem diamond_cut_1_3_loss (a : ℝ) :
  (value a - (value (1/4 * a) + value (3/4 * a))) / value a = 0.375 :=
by sorry

theorem diamond_max_loss_ratio :
  ∀ (m n : ℝ), (m > 0) → (n > 0) → 
  (1 - (value (m/(m + n)) + value (n/(m + n))) ≤ 0.5) :=
by sorry

end diamond_cut_1_3_loss_diamond_max_loss_ratio_l408_40820


namespace problem_1_problem_2_l408_40822

theorem problem_1 (p x : ℝ) (h1 : |p| ≤ 2) (h2 : x^2 + p*x + 1 > 2*x + p) : x < -1 ∨ x > 3 :=
sorry

theorem problem_2 (p x : ℝ) (h1 : 2 ≤ x) (h2 : x ≤ 4) (h3 : x^2 + p*x + 1 > 2*x + p) : p > -1 :=
sorry

end problem_1_problem_2_l408_40822


namespace total_handshakes_l408_40842

def total_people := 40
def group_x_people := 25
def group_x_known_others := 5
def group_y_people := 15
def handshakes_between_x_y := group_x_people * group_y_people
def handshakes_within_x := 25 * (25 - 1 - 5) / 2
def handshakes_within_y := (15 * (15 - 1)) / 2

theorem total_handshakes 
    (h1 : total_people = 40)
    (h2 : group_x_people = 25)
    (h3 : group_x_known_others = 5)
    (h4 : group_y_people = 15) :
    handshakes_between_x_y + handshakes_within_x + handshakes_within_y = 717 := 
by
  sorry

end total_handshakes_l408_40842


namespace total_dinners_l408_40891

def monday_dinners := 40
def tuesday_dinners := monday_dinners + 40
def wednesday_dinners := tuesday_dinners / 2
def thursday_dinners := wednesday_dinners + 3

theorem total_dinners : monday_dinners + tuesday_dinners + wednesday_dinners + thursday_dinners = 203 := by
  sorry

end total_dinners_l408_40891


namespace number_is_100_l408_40894

theorem number_is_100 (x : ℝ) (h : 0.60 * (3 / 5) * x = 36) : x = 100 :=
by sorry

end number_is_100_l408_40894


namespace solve_functional_equation_l408_40839

theorem solve_functional_equation
  (f g h : ℝ → ℝ)
  (H : ∀ x y : ℝ, f x - g y = (x - y) * h (x + y)) :
  ∃ d c : ℝ, (∀ x, f x = d * x^2 + c) ∧ (∀ x, g x = d * x^2 + c) :=
sorry

end solve_functional_equation_l408_40839


namespace chessboard_property_exists_l408_40813

theorem chessboard_property_exists (n : ℕ) (x : Fin n → Fin n → ℝ) 
  (h : ∀ i j k : Fin n, x i j + x j k + x k i = 0) :
  ∃ (t : Fin n → ℝ), ∀ i j, x i j = t i - t j := 
sorry

end chessboard_property_exists_l408_40813


namespace min_value_expression_l408_40850

open Real

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 4) :
  ∃ z : ℝ, z = 16 / 7 ∧ ∀ u > 0, ∀ v > 0, u + v = 4 → ((u^2 / (u + 1)) + (v^2 / (v + 2))) ≥ z :=
by
  sorry

end min_value_expression_l408_40850


namespace find_jordana_and_james_age_l408_40878

variable (current_age_of_Jennifer : ℕ) (current_age_of_Jordana : ℕ) (current_age_of_James : ℕ)

-- Conditions
axiom jennifer_40_in_twenty_years : current_age_of_Jennifer + 20 = 40
axiom jordana_twice_jennifer_in_twenty_years : current_age_of_Jordana + 20 = 2 * (current_age_of_Jennifer + 20)
axiom james_ten_years_younger_in_twenty_years : current_age_of_James + 20 = 
  (current_age_of_Jennifer + 20) + (current_age_of_Jordana + 20) - 10

-- Prove that Jordana is currently 60 years old and James is currently 90 years old
theorem find_jordana_and_james_age : current_age_of_Jordana = 60 ∧ current_age_of_James = 90 :=
  sorry

end find_jordana_and_james_age_l408_40878


namespace fraction_zero_l408_40810

theorem fraction_zero (x : ℝ) (h₁ : 2 * x = 0) (h₂ : x + 2 ≠ 0) : (2 * x) / (x + 2) = 0 :=
by {
  sorry
}

end fraction_zero_l408_40810


namespace smallest_enclosing_sphere_radius_l408_40852

-- Define the conditions
def sphere_radius : ℝ := 2

-- Define the sphere center coordinates in each octant
def sphere_centers : List (ℝ × ℝ × ℝ) :=
  [ (2, 2, 2), (2, 2, -2), (2, -2, 2), (2, -2, -2),
    (-2, 2, 2), (-2, 2, -2), (-2, -2, 2), (-2, -2, -2) ]

-- Define the theorem statement
theorem smallest_enclosing_sphere_radius :
  (∃ (r : ℝ), r = 2 * Real.sqrt 3 + 2) :=
by
  -- conditions and proof will go here
  sorry

end smallest_enclosing_sphere_radius_l408_40852


namespace meryll_remaining_questions_l408_40828

variables (total_mc total_ps total_tf : ℕ)
variables (frac_mc frac_ps frac_tf : ℚ)

-- Conditions as Lean definitions:
def written_mc (total_mc : ℕ) (frac_mc : ℚ) := (frac_mc * total_mc).floor
def written_ps (total_ps : ℕ) (frac_ps : ℚ) := (frac_ps * total_ps).floor
def written_tf (total_tf : ℕ) (frac_tf : ℚ) := (frac_tf * total_tf).floor

def remaining_mc (total_mc : ℕ) (frac_mc : ℚ) := total_mc - written_mc total_mc frac_mc
def remaining_ps (total_ps : ℕ) (frac_ps : ℚ) := total_ps - written_ps total_ps frac_ps
def remaining_tf (total_tf : ℕ) (frac_tf : ℚ) := total_tf - written_tf total_tf frac_tf

def total_remaining (total_mc total_ps total_tf : ℕ) (frac_mc frac_ps frac_tf : ℚ) :=
  remaining_mc total_mc frac_mc + remaining_ps total_ps frac_ps + remaining_tf total_tf frac_tf

-- The statement to prove:
theorem meryll_remaining_questions :
  total_remaining 50 30 40 (5/8) (7/12) (2/5) = 56 :=
by
  sorry

end meryll_remaining_questions_l408_40828


namespace expand_product_l408_40817

theorem expand_product (x : ℝ) : (x^2 - 2*x + 2) * (x^2 + 2*x + 2) = x^4 + 4 :=
by
  sorry

end expand_product_l408_40817


namespace tetrahedron_volume_l408_40832

theorem tetrahedron_volume 
  (R S₁ S₂ S₃ S₄ : ℝ) : 
  V = R * (S₁ + S₂ + S₃ + S₄) :=
sorry

end tetrahedron_volume_l408_40832


namespace radius_of_circle_l408_40857

theorem radius_of_circle
  (d PQ QR : ℝ) (h1 : d = 15) (h2 : PQ = 7) (h3 : QR = 8) :
  ∃ r : ℝ, r = 2 * Real.sqrt 30 ∧ (PQ * (PQ + QR) = (d - r) * (d + r)) :=
by
  -- All necessary non-proof related statements
  sorry

end radius_of_circle_l408_40857


namespace find_number_l408_40847

-- Define the conditions
def satisfies_condition (x : ℝ) : Prop := x * 4 * 25 = 812

-- The main theorem stating that the number satisfying the condition is 8.12
theorem find_number (x : ℝ) (h : satisfies_condition x) : x = 8.12 :=
by
  sorry

end find_number_l408_40847


namespace faucet_fill_time_l408_40800

theorem faucet_fill_time (r : ℝ) (T1 T2 t : ℝ) (F1 F2 : ℕ) (h1 : T1 = 200) (h2 : t = 8) (h3 : F1 = 4) (h4 : F2 = 8) (h5 : T2 = 50) (h6 : r * F1 * t = T1) : 
(F2 * r) * t / (F1 * F2) = T2 -> by sorry := sorry

#check faucet_fill_time

end faucet_fill_time_l408_40800


namespace equality_conditions_l408_40892

theorem equality_conditions (a b c d : ℝ) :
  a + bcd = (a + b) * (a + c) * (a + d) ↔ a = 0 ∨ a^2 + a * (b + c + d) + bc + bd + cd = 1 :=
by
  sorry

end equality_conditions_l408_40892


namespace find_x_if_opposites_l408_40821

theorem find_x_if_opposites (x : ℝ) (h : 2 * (x - 3) = - 4 * (1 - x)) : x = -1 := 
by
  sorry

end find_x_if_opposites_l408_40821


namespace domain_of_sqrt_one_minus_ln_l408_40814

def domain (x : ℝ) : Prop := 0 < x ∧ x ≤ Real.exp 1

theorem domain_of_sqrt_one_minus_ln (x : ℝ) : (1 - Real.log x ≥ 0) ∧ (x > 0) ↔ domain x := by
sorry

end domain_of_sqrt_one_minus_ln_l408_40814


namespace gcd_of_ten_digit_same_five_digit_l408_40816

def ten_digit_same_five_digit (n : ℕ) : Prop :=
  n > 9999 ∧ n < 100000 ∧ ∃ k : ℕ, k = n * (10^10 + 10^5 + 1)

theorem gcd_of_ten_digit_same_five_digit :
  (∀ n : ℕ, ten_digit_same_five_digit n → ∃ d : ℕ, d = 10000100001 ∧ ∀ m : ℕ, m ∣ d) := 
sorry

end gcd_of_ten_digit_same_five_digit_l408_40816


namespace tan_A_in_right_triangle_l408_40802

theorem tan_A_in_right_triangle (AC : ℝ) (AB : ℝ) (BC : ℝ) (hAC : AC = Real.sqrt 20) (hAB : AB = 4) (h_right_triangle : AC^2 = AB^2 + BC^2) :
  Real.tan (Real.arcsin (AB / AC)) = 1 / 2 :=
by
  sorry

end tan_A_in_right_triangle_l408_40802


namespace problem_statement_l408_40869

noncomputable def necessary_but_not_sufficient_condition (x y : ℝ) (hx : x > 0) : Prop :=
  (x > |y| → x > y) ∧ ¬ (x > y → x > |y|)

theorem problem_statement
  (x y : ℝ)
  (hx : x > 0)
  : necessary_but_not_sufficient_condition x y hx :=
sorry

end problem_statement_l408_40869


namespace determine_n_l408_40883

theorem determine_n (n : ℕ) (h : 3^n = 3^2 * 9^4 * 81^3) : n = 22 := 
by
  sorry

end determine_n_l408_40883


namespace two_digit_sum_l408_40880

theorem two_digit_sum (x y n : ℕ) (hx : 10 ≤ x ∧ x < 100)
  (hy : 10 ≤ y ∧ y < 100) (h_rev : y = (x % 10) * 10 + x / 10)
  (h_diff_square : x^2 - y^2 = n^2) : x + y + n = 154 :=
sorry

end two_digit_sum_l408_40880


namespace AK_eq_CK_l408_40860

variable {α : Type*} [LinearOrder α] [LinearOrder ℝ]

variable (A B C L K : ℝ)
variable (triangle : ℝ)
variable (h₁ : AL = LB)
variable (h₂ : AK = CL)

--  Given that in triangle ABC,
--     AL is a bisector such that AL = LB,
--     and AK is on ray AL with AK = CL,
--     prove that AK = CK.
theorem AK_eq_CK (h₁ : AL = LB) (h₂ : AK = CL) : AK = CK := by
  sorry

end AK_eq_CK_l408_40860


namespace angle_of_inclination_l408_40849

-- The statement of the mathematically equivalent proof problem in Lean 4
theorem angle_of_inclination
  (k: ℝ)
  (α: ℝ)
  (line_eq: ∀ x, ∃ y, y = (k-1) * x + 2)
  (circle_eq: ∀ x y, x^2 + y^2 + k * x + 2 * y + k^2 = 0) :
  α = 3 * Real.pi / 4 :=
sorry -- Proof to be provided

end angle_of_inclination_l408_40849


namespace dan_helmet_craters_l408_40875

variable (D S : ℕ)
variable (h1 : D = S + 10)
variable (h2 : D + S + 15 = 75)

theorem dan_helmet_craters : D = 35 := by
  sorry

end dan_helmet_craters_l408_40875


namespace prove_inequality_l408_40871

variable (f : ℝ → ℝ)

-- Conditions
axiom condition : ∀ x y : ℝ, (f x + y) * (f y + x) > 0 → f x + y = f y + x

-- Proof of the desired statement
theorem prove_inequality : ∀ x y : ℝ, x > y → f x + y ≤ f y + x :=
by
  sorry

end prove_inequality_l408_40871


namespace probability_condition_l408_40899

namespace SharedPowerBank

def P (event : String) : ℚ :=
  match event with
  | "A" => 3 / 4
  | "B" => 1 / 2
  | _   => 0 -- Default case for any other event

def probability_greater_than_1000_given_greater_than_500 : ℚ :=
  P "B" / P "A"

theorem probability_condition :
  probability_greater_than_1000_given_greater_than_500 = 2 / 3 :=
by 
  sorry

end SharedPowerBank

end probability_condition_l408_40899


namespace area_excircle_gteq_four_times_area_l408_40896

-- Define the area function
def area (A B C : Point) : ℝ := sorry -- Area of triangle ABC (this will be implemented later)

-- Define the centers of the excircles (this needs precise definitions and setup)
def excircle_center (A B C : Point) : Point := sorry -- Centers of the excircles of triangle ABC (implementation would follow)

-- Define the area of the triangle formed by the excircle centers
def excircle_area (A B C : Point) : ℝ :=
  let O1 := excircle_center A B C
  let O2 := excircle_center B C A
  let O3 := excircle_center C A B
  area O1 O2 O3

-- Prove the main statement
theorem area_excircle_gteq_four_times_area (A B C : Point) :
  excircle_area A B C ≥ 4 * area A B C :=
by sorry

end area_excircle_gteq_four_times_area_l408_40896


namespace spending_on_games_l408_40844

-- Definitions converted from conditions
def totalAllowance := 48
def fractionClothes := 1 / 4
def fractionBooks := 1 / 3
def fractionSnacks := 1 / 6
def spentClothes := fractionClothes * totalAllowance
def spentBooks := fractionBooks * totalAllowance
def spentSnacks := fractionSnacks * totalAllowance
def spentGames := totalAllowance - (spentClothes + spentBooks + spentSnacks)

-- The theorem that needs to be proven
theorem spending_on_games : spentGames = 12 :=
by sorry

end spending_on_games_l408_40844


namespace blue_candy_count_l408_40838

def total_pieces : ℕ := 3409
def red_pieces : ℕ := 145
def blue_pieces : ℕ := total_pieces - red_pieces

theorem blue_candy_count :
  blue_pieces = 3264 := by
  sorry

end blue_candy_count_l408_40838


namespace xiao_yu_reading_days_l408_40818

-- Definition of Xiao Yu's reading problem
def number_of_pages_per_day := 15
def total_number_of_days := 24
def additional_pages_per_day := 3
def new_number_of_pages_per_day := number_of_pages_per_day + additional_pages_per_day
def total_pages := number_of_pages_per_day * total_number_of_days
def new_total_number_of_days := total_pages / new_number_of_pages_per_day

-- Theorem statement in Lean 4
theorem xiao_yu_reading_days : new_total_number_of_days = 20 :=
  sorry

end xiao_yu_reading_days_l408_40818


namespace part1_l408_40884

theorem part1 (a b c t m n : ℝ) (h1 : a > 0) (h2 : m = n) (h3 : t = (3 + (t + 1)) / 2) : t = 4 :=
sorry

end part1_l408_40884


namespace tom_found_dimes_l408_40848

theorem tom_found_dimes :
  let quarters := 10
  let nickels := 4
  let pennies := 200
  let total_value := 5
  let value_quarters := 0.25 * quarters
  let value_nickels := 0.05 * nickels
  let value_pennies := 0.01 * pennies
  let total_other := value_quarters + value_nickels + value_pennies
  let value_dimes := total_value - total_other
  value_dimes / 0.10 = 3 := sorry

end tom_found_dimes_l408_40848


namespace orchestra_members_l408_40835

theorem orchestra_members : ∃ (x : ℕ), (130 < x) ∧ (x < 260) ∧ (x % 6 = 1) ∧ (x % 5 = 2) ∧ (x % 7 = 3) ∧ (x = 241) :=
by
  sorry

end orchestra_members_l408_40835


namespace principal_amount_l408_40870

theorem principal_amount (P R T SI : ℝ) (hR : R = 4) (hT : T = 5) (hSI : SI = P - 2240) 
    (h_formula : SI = (P * R * T) / 100) : P = 2800 :=
by 
  sorry

end principal_amount_l408_40870


namespace no_solutions_m_m_plus_2_eq_n_n_plus_1_no_solutions_m_m_plus_3_eq_n_n_plus_1_solutions_exist_m_m_plus_k_eq_n_n_plus_1_l408_40837

-- Part (1): Prove that there do not exist positive integers m and n such that m(m+2) = n(n+1)
theorem no_solutions_m_m_plus_2_eq_n_n_plus_1 : ¬ ∃ (m n : ℕ), m > 0 ∧ n > 0 ∧ m * (m + 2) = n * (n + 1) :=
sorry

-- Part (2): Given k ≥ 3,
-- Case (a): Prove that for k=3, there do not exist positive integers m and n such that m(m+3) = n(n+1)
theorem no_solutions_m_m_plus_3_eq_n_n_plus_1 : ¬ ∃ (m n : ℕ), m > 0 ∧ n > 0 ∧ m * (m + 3) = n * (n + 1) :=
sorry

-- Case (b): Prove that for k ≥ 4, there exist positive integers m and n such that m(m+k) = n(n+1)
theorem solutions_exist_m_m_plus_k_eq_n_n_plus_1 (k : ℕ) (h : k ≥ 4) : 
  ∃ (m n : ℕ), m > 0 ∧ n > 0 ∧ m * (m + k) = n * (n + 1) :=
sorry

end no_solutions_m_m_plus_2_eq_n_n_plus_1_no_solutions_m_m_plus_3_eq_n_n_plus_1_solutions_exist_m_m_plus_k_eq_n_n_plus_1_l408_40837


namespace rope_costs_purchasing_plans_minimum_cost_l408_40890

theorem rope_costs (x y m : ℕ) :
  (10 * x + 5 * y = 175) →
  (15 * x + 10 * y = 300) →
  x = 10 ∧ y = 15 :=
sorry

theorem purchasing_plans (m : ℕ) :
  (10 * 10 + 15 * 15 = 300) →
  23 ≤ m ∧ m ≤ 25 :=
sorry

theorem minimum_cost (m : ℕ) :
  (23 ≤ m ∧ m ≤ 25) →
  m = 25 →
  10 * m + 15 * (45 - m) = 550 :=
sorry

end rope_costs_purchasing_plans_minimum_cost_l408_40890


namespace driving_time_ratio_l408_40874

theorem driving_time_ratio 
  (t : ℝ)
  (h : 30 * t + 60 * (2 * t) = 75) : 
  t / (2 * t) = 1 / 2 := 
by
  sorry

end driving_time_ratio_l408_40874


namespace total_bills_54_l408_40843

/-- A bank teller has some 5-dollar and 20-dollar bills in her cash drawer, 
and the total value of the bills is 780 dollars, with 20 5-dollar bills.
Show that the total number of bills is 54. -/
theorem total_bills_54 (value_total : ℕ) (num_5dollar : ℕ) (num_5dollar_value : ℕ) (num_20dollar : ℕ) :
    value_total = 780 ∧ num_5dollar = 20 ∧ num_5dollar_value = 5 ∧ num_20dollar * 20 + num_5dollar * num_5dollar_value = value_total
    → num_20dollar + num_5dollar = 54 :=
by
  sorry

end total_bills_54_l408_40843


namespace negation_equivalence_l408_40841

-- Define the angles in a triangle as three real numbers
def is_triangle (a b c : ℝ) : Prop :=
  a + b + c = 180 ∧ a > 0 ∧ b > 0 ∧ c > 0

-- Define the proposition
def at_least_one_angle_not_greater_than_60 (a b c : ℝ) : Prop :=
  a ≤ 60 ∨ b ≤ 60 ∨ c ≤ 60

-- Negate the proposition
def all_angles_greater_than_60 (a b c : ℝ) : Prop :=
  a > 60 ∧ b > 60 ∧ c > 60

-- Prove that the negation of the proposition is equivalent
theorem negation_equivalence (a b c : ℝ) (h_triangle : is_triangle a b c) :
  ¬ at_least_one_angle_not_greater_than_60 a b c ↔ all_angles_greater_than_60 a b c :=
by
  sorry

end negation_equivalence_l408_40841


namespace isosceles_trapezoid_diagonal_length_l408_40879

theorem isosceles_trapezoid_diagonal_length
  (AB CD : ℝ) (AD BC : ℝ) :
  AB = 15 →
  CD = 9 →
  AD = 12 →
  BC = 12 →
  (AC : ℝ) = Real.sqrt 279 :=
by
  intros hAB hCD hAD hBC
  sorry

end isosceles_trapezoid_diagonal_length_l408_40879


namespace vector_solution_l408_40811

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem vector_solution (a x : V) (h : 2 • x - 3 • (x - 2 • a) = 0) : x = 6 • a :=
by sorry

end vector_solution_l408_40811


namespace order_of_numbers_l408_40851

theorem order_of_numbers :
  2^30 < 10^10 ∧ 10^10 < 5^15 :=
by
  sorry

end order_of_numbers_l408_40851


namespace problem_statement_l408_40865

theorem problem_statement (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a * b + b * c + c * a = 1) :
  Real.sqrt (a + 1/a) + Real.sqrt (b + 1/b) + Real.sqrt (c + 1/c) ≥ 2 * (Real.sqrt a + Real.sqrt b + Real.sqrt c) :=
sorry

end problem_statement_l408_40865


namespace circle_diameter_from_area_l408_40859

theorem circle_diameter_from_area (A r d : ℝ) (hA : A = 64 * Real.pi) (h_area : A = Real.pi * r^2) : d = 16 :=
by
  sorry

end circle_diameter_from_area_l408_40859


namespace number_of_ordered_quadruples_l408_40888

theorem number_of_ordered_quadruples (x1 x2 x3 x4 : ℕ) (h1 : x1 > 0) (h2 : x2 > 0) (h3 : x3 > 0) (h4 : x4 > 0) (h_sum : x1 + x2 + x3 + x4 = 100) : 
  ∃ n : ℕ, n = 156849 := 
by 
  sorry

end number_of_ordered_quadruples_l408_40888


namespace fully_filled_boxes_l408_40829

theorem fully_filled_boxes (total_cards : ℕ) (cards_per_box : ℕ) (h1 : total_cards = 94) (h2 : cards_per_box = 8) : total_cards / cards_per_box = 11 :=
by {
  sorry
}

end fully_filled_boxes_l408_40829


namespace exponential_inequality_example_l408_40858

theorem exponential_inequality_example (a b : ℝ) (h : 1.5 > 0 ∧ 1.5 ≠ 1) (h2 : 2.3 < 3.2) : 1.5 ^ 2.3 < 1.5 ^ 3.2 :=
by 
  sorry

end exponential_inequality_example_l408_40858


namespace each_mouse_not_visit_with_every_other_once_l408_40834

theorem each_mouse_not_visit_with_every_other_once : 
    (∃ mice: Finset ℕ, mice.card = 24 ∧ (∀ f : ℕ → Finset ℕ, 
    (∀ n, (f n).card = 4) ∧ 
    (∀ i j, i ≠ j → (f i ∩ f j ≠ ∅) → (f i ∩ f j).card ≠ (mice.card - 1)))
    ) → false := 
by
  sorry

end each_mouse_not_visit_with_every_other_once_l408_40834


namespace exists_line_intersecting_circle_and_passing_origin_l408_40846

theorem exists_line_intersecting_circle_and_passing_origin :
  ∃ m : ℝ, (m = 1 ∨ m = -4) ∧ 
  ∃ (x y : ℝ), 
    ((x - 1) ^ 2 + (y + 2) ^ 2 = 9) ∧ 
    ((x - y + m = 0) ∧ 
     ∃ (x' y' : ℝ),
      ((x' - 1) ^ 2 + (y' + 2) ^ 2 = 9) ∧ 
      ((x' - y' + m = 0) ∧ ((x + x') / 2 = 0 ∧ (y + y') / 2 = 0))) :=
by 
  sorry

end exists_line_intersecting_circle_and_passing_origin_l408_40846


namespace eggs_per_meal_l408_40895

noncomputable def initial_eggs_from_store : ℕ := 12
noncomputable def additional_eggs_from_neighbor : ℕ := 12
noncomputable def eggs_used_for_cooking : ℕ := 2 + 4
noncomputable def remaining_eggs_after_cooking : ℕ := initial_eggs_from_store + additional_eggs_from_neighbor - eggs_used_for_cooking
noncomputable def eggs_given_to_aunt : ℕ := remaining_eggs_after_cooking / 2
noncomputable def remaining_eggs_after_giving_to_aunt : ℕ := remaining_eggs_after_cooking - eggs_given_to_aunt
noncomputable def planned_meals : ℕ := 3

theorem eggs_per_meal : remaining_eggs_after_giving_to_aunt / planned_meals = 3 := 
by 
  sorry

end eggs_per_meal_l408_40895


namespace equation_of_parabola_passing_through_points_l408_40823

noncomputable def parabola (x : ℝ) (b c : ℝ) : ℝ :=
  x^2 + b * x + c

theorem equation_of_parabola_passing_through_points :
  ∃ (b c : ℝ), 
    (parabola 0 b c = 5) ∧ (parabola 3 b c = 2) ∧
    (∀ x, parabola x b c = x^2 - 4 * x + 5) := 
by
  sorry

end equation_of_parabola_passing_through_points_l408_40823


namespace smallest_number_l408_40866

theorem smallest_number (n : ℕ) : 
  (∀ k ∈ [12, 16, 18, 21, 28, 35, 39], ∃ m : ℕ, (n - 3) = k * m) → 
  n = 65517 := by
  sorry

end smallest_number_l408_40866


namespace isabella_hair_length_l408_40854

-- Define the conditions and the question in Lean
def current_length : ℕ := 9
def length_cut_off : ℕ := 9

-- Main theorem statement
theorem isabella_hair_length 
  (current_length : ℕ) 
  (length_cut_off : ℕ) 
  (H1 : current_length = 9) 
  (H2 : length_cut_off = 9) : 
  current_length + length_cut_off = 18 :=
  sorry

end isabella_hair_length_l408_40854


namespace ammonium_iodide_requirement_l408_40836

theorem ammonium_iodide_requirement :
  ∀ (NH4I KOH NH3 KI H2O : ℕ),
  (NH4I + KOH = NH3 + KI + H2O) → 
  (NH4I = 3) →
  (KOH = 3) →
  (NH3 = 3) →
  (KI = 3) →
  (H2O = 3) →
  NH4I = 3 :=
by
  intros NH4I KOH NH3 KI H2O reaction_balanced NH4I_req KOH_req NH3_prod KI_prod H2O_prod
  exact NH4I_req

end ammonium_iodide_requirement_l408_40836


namespace sequence_a_n_term_l408_40809

theorem sequence_a_n_term :
  ∃ a : ℕ → ℕ, 
  a 1 = 1 ∧
  (∀ n : ℕ, a (n+1) = 2 * a n + 1) ∧
  a 10 = 1023 := by
  sorry

end sequence_a_n_term_l408_40809


namespace largest_possible_number_of_neither_l408_40864

theorem largest_possible_number_of_neither
  (writers : ℕ)
  (editors : ℕ)
  (attendees : ℕ)
  (x : ℕ)
  (N : ℕ)
  (h_writers : writers = 45)
  (h_editors_gt : editors > 38)
  (h_attendees : attendees = 90)
  (h_both : N = 2 * x)
  (h_equation : writers + editors - x + N = attendees) :
  N = 12 :=
by
  sorry

end largest_possible_number_of_neither_l408_40864


namespace mans_rate_is_19_l408_40831

-- Define the given conditions
def downstream_speed : ℝ := 25
def upstream_speed : ℝ := 13

-- Define the man's rate in still water and state the theorem
theorem mans_rate_is_19 : (downstream_speed + upstream_speed) / 2 = 19 := by
  -- Proof goes here
  sorry

end mans_rate_is_19_l408_40831


namespace enchilada_taco_cost_l408_40845

variables (e t : ℝ)

theorem enchilada_taco_cost 
  (h1 : 4 * e + 5 * t = 4.00) 
  (h2 : 5 * e + 3 * t = 3.80) 
  (h3 : 7 * e + 6 * t = 6.10) : 
  4 * e + 7 * t = 4.75 := 
sorry

end enchilada_taco_cost_l408_40845


namespace hummus_serving_amount_proof_l408_40824

/-- Given conditions: 
    one_can is the number of ounces of chickpeas in one can,
    total_cans is the number of cans Thomas buys,
    total_servings is the number of servings of hummus Thomas needs to make,
    to_produce_one_serving is the amount of chickpeas needed for one serving,
    we prove that to_produce_one_serving = 6.4 given the above conditions. -/
theorem hummus_serving_amount_proof 
  (one_can : ℕ) 
  (total_cans : ℕ) 
  (total_servings : ℕ) 
  (to_produce_one_serving : ℚ) 
  (h_one_can : one_can = 16) 
  (h_total_cans : total_cans = 8)
  (h_total_servings : total_servings = 20) 
  (h_total_ounces : total_cans * one_can = 128) : 
  to_produce_one_serving = 128 / 20 := 
by
  sorry

end hummus_serving_amount_proof_l408_40824


namespace number_of_color_copies_l408_40867

def charge_shop_X (n : ℕ) : ℝ := 1.20 * n
def charge_shop_Y (n : ℕ) : ℝ := 1.70 * n
def difference := 20

theorem number_of_color_copies (n : ℕ) (h : charge_shop_Y n = charge_shop_X n + difference) : n = 40 :=
by {
  sorry
}

end number_of_color_copies_l408_40867


namespace expression_equals_one_l408_40830

theorem expression_equals_one : 
  (Real.sqrt 6 / Real.sqrt 2) + abs (1 - Real.sqrt 3) - Real.sqrt 12 + (1 / 2)⁻¹ = 1 := 
by sorry

end expression_equals_one_l408_40830


namespace find_k_l408_40815

theorem find_k (k : ℚ) (h : ∃ k : ℚ, (3 * (4 - k) = 2 * (-5 - 3))): k = -4 / 3 := by
  sorry

end find_k_l408_40815


namespace relationship_p_q_l408_40863

noncomputable def log_a (a x : ℝ) := Real.log x / Real.log a

theorem relationship_p_q (x a p q : ℝ) (h1 : 0 ≤ x) (h2 : x < 1) (h3 : a > 0) (h4 : a ≠ 1)
  (hp : p = |log_a a (1 + x)|) (hq : q = |log_a a (1 - x)|) : p ≤ q :=
sorry

end relationship_p_q_l408_40863


namespace geometric_sequence_property_l408_40825

variables {a : ℕ → ℝ} {S : ℕ → ℝ}

noncomputable def a_n (n : ℕ) : ℝ := 2 * 3^(n - 1)
noncomputable def S_n (n : ℕ) : ℝ := 
  if n = 0 then 0
  else (2 * (1 - 3^n)) / (1 - 3)

theorem geometric_sequence_property 
  (h₁ : a 1 + a 2 + a 3 = 26)
  (h₂ : S 6 = 728)
  (h₃ : ∀ n, a n = a_n n)
  (h₄ : ∀ n, S n = S_n n) :
  ∀ n, S (n + 1) ^ 2 - S n * S (n + 2) = 4 * 3 ^ n :=
by sorry

end geometric_sequence_property_l408_40825


namespace arithmetic_sequence_minimization_l408_40806

theorem arithmetic_sequence_minimization (a b : ℕ) (h_range : 1 ≤ a ∧ b ≤ 17) (h_seq : a + b = 18) (h_min : ∀ x y, (1 ≤ x ∧ y ≤ 17 ∧ x + y = 18) → (1 / x + 25 / y) ≥ (1 / a + 25 / b)) : ∃ n : ℕ, n = 9 :=
by
  -- We'd usually follow by proving the conditions and defining the sequence correctly.
  -- Definitions and steps leading to finding n = 9 will be elaborated here.
  -- This placeholder is to satisfy the requirement only.
  sorry

end arithmetic_sequence_minimization_l408_40806


namespace total_bird_count_correct_l408_40893

-- Define initial counts
def initial_sparrows : ℕ := 89
def initial_pigeons : ℕ := 68
def initial_finches : ℕ := 74

-- Define additional birds
def additional_sparrows : ℕ := 42
def additional_pigeons : ℕ := 51
def additional_finches : ℕ := 27

-- Define total counts
def initial_total : ℕ := 231
def final_total : ℕ := 312

theorem total_bird_count_correct :
  initial_sparrows + initial_pigeons + initial_finches = initial_total ∧
  (initial_sparrows + additional_sparrows) + 
  (initial_pigeons + additional_pigeons) + 
  (initial_finches + additional_finches) = final_total := by
    sorry

end total_bird_count_correct_l408_40893


namespace cos_double_angle_identity_l408_40881

theorem cos_double_angle_identity (x : ℝ) (h : Real.sin (Real.pi / 2 + x) = 3 / 5) : 
  Real.cos (2 * x) = -7 / 25 :=
sorry

end cos_double_angle_identity_l408_40881


namespace exists_element_x_l408_40885

open Set

theorem exists_element_x (n : ℕ) (S : Finset (Fin n)) (A : Fin n → Finset (Fin n)) 
  (h_distinct : ∀ i j : Fin n, i ≠ j → A i ≠ A j) : 
  ∃ x ∈ S, ∀ i j : Fin n, i ≠ j → (A i \ {x}) ≠ (A j \ {x}) :=
sorry

end exists_element_x_l408_40885


namespace divides_both_numerator_and_denominator_l408_40801

theorem divides_both_numerator_and_denominator (x m : ℤ) :
  (x ∣ (5 * m + 6)) ∧ (x ∣ (8 * m + 7)) → (x = 1 ∨ x = -1 ∨ x = 13 ∨ x = -13) :=
by
  sorry

end divides_both_numerator_and_denominator_l408_40801


namespace factorizable_trinomial_l408_40819

theorem factorizable_trinomial (k : ℤ) : (∃ a b : ℤ, a + b = k ∧ a * b = 5) ↔ (k = 6 ∨ k = -6) :=
by
  sorry

end factorizable_trinomial_l408_40819


namespace find_m_l408_40853

open Set

def A (m: ℝ) := {x : ℝ | x^2 - m * x + m^2 - 19 = 0}

def B := {x : ℝ | x^2 - 5 * x + 6 = 0}

def C := ({2, -4} : Set ℝ)

theorem find_m (m : ℝ) (ha : A m ∩ B ≠ ∅) (hb : A m ∩ C = ∅) : m = -2 :=
  sorry

end find_m_l408_40853


namespace greatest_measure_length_l408_40886

theorem greatest_measure_length :
  let l1 := 18000
  let l2 := 50000
  let l3 := 1520
  ∃ d, d = Int.gcd (Int.gcd l1 l2) l3 ∧ d = 40 :=
by
  sorry

end greatest_measure_length_l408_40886


namespace wire_cut_problem_l408_40882

noncomputable def shorter_piece_length (total_length : ℝ) (ratio : ℝ) : ℝ :=
  let x := total_length / (1 + ratio)
  x

theorem wire_cut_problem (total_length : ℝ) (ratio : ℝ) (shorter_length : ℝ) :
  total_length = 35 → ratio = 5/2 → shorter_length = 10 → shorter_piece_length total_length ratio = shorter_length := by
  intros h1 h2 h3
  unfold shorter_piece_length
  rw [h1, h2, h3]
  sorry

end wire_cut_problem_l408_40882


namespace abe_family_total_yen_l408_40898

theorem abe_family_total_yen (yen_checking : ℕ) (yen_savings : ℕ) (h₁ : yen_checking = 6359) (h₂ : yen_savings = 3485) : yen_checking + yen_savings = 9844 :=
by
  sorry

end abe_family_total_yen_l408_40898


namespace calculate_expression_l408_40897

/-- Calculate the expression 2197 + 180 ÷ 60 × 3 - 197. -/
theorem calculate_expression : 2197 + (180 / 60) * 3 - 197 = 2009 := by
  sorry

end calculate_expression_l408_40897


namespace range_of_m_l408_40889
noncomputable def f (x : ℝ) : ℝ := ((x - 1) / (x + 1))^2

noncomputable def f_inv (x : ℝ) : ℝ := (1 + Real.sqrt x) / (1 - Real.sqrt x)

theorem range_of_m {x : ℝ} (m : ℝ) (h1 : 1 / 16 ≤ x) (h2 : x ≤ 1 / 4) 
  (h3 : ∀ (x : ℝ), (1 - Real.sqrt x) * f_inv x > m * (m - Real.sqrt x)): 
  -1 < m ∧ m < 5 / 4 :=
sorry

end range_of_m_l408_40889


namespace percentage_entree_cost_l408_40876

-- Conditions
def total_spent : ℝ := 50.0
def num_appetizers : ℝ := 2
def cost_per_appetizer : ℝ := 5.0
def total_appetizer_cost : ℝ := num_appetizers * cost_per_appetizer
def total_entree_cost : ℝ := total_spent - total_appetizer_cost

-- Proof Problem
theorem percentage_entree_cost :
  (total_entree_cost / total_spent) * 100 = 80 :=
sorry

end percentage_entree_cost_l408_40876


namespace full_time_and_year_l408_40803

variable (Total F Y N FY : ℕ)

theorem full_time_and_year (h1 : Total = 130)
                            (h2 : F = 80)
                            (h3 : Y = 100)
                            (h4 : N = 20)
                            (h5 : Total = FY + (F - FY) + (Y - FY) + N) :
    FY = 90 := 
sorry

end full_time_and_year_l408_40803


namespace parallelogram_base_length_l408_40861

theorem parallelogram_base_length (b : ℝ) (h : ℝ) (area : ℝ) 
  (h_eq_2b : h = 2 * b) (area_eq_98 : area = 98) 
  (area_def : area = b * h) : b = 7 :=
by
  sorry

end parallelogram_base_length_l408_40861


namespace percent_difference_l408_40856

def boys := 100
def girls := 125
def diff := girls - boys
def boys_less_than_girls_percent := (diff : ℚ) / girls  * 100
def girls_more_than_boys_percent := (diff : ℚ) / boys  * 100

theorem percent_difference :
  boys_less_than_girls_percent = 20 ∧ girls_more_than_boys_percent = 25 :=
by
  -- The proof here demonstrates the percentage calculations.
  sorry

end percent_difference_l408_40856


namespace shaded_fraction_l408_40826

noncomputable def fraction_shaded (l w : ℝ) : ℝ :=
  1 - (1 / 8)

theorem shaded_fraction (l w : ℝ) (hl : l > 0) (hw : w > 0) :
  fraction_shaded l w = 7 / 8 :=
by
  sorry

end shaded_fraction_l408_40826


namespace total_tickets_sold_l408_40873

-- Definitions of the conditions as given in the problem
def price_adult : ℕ := 7
def price_child : ℕ := 4
def total_revenue : ℕ := 5100
def child_tickets_sold : ℕ := 400

-- The main statement (theorem) to prove
theorem total_tickets_sold:
  ∃ (A C : ℕ), C = child_tickets_sold ∧ price_adult * A + price_child * C = total_revenue ∧ (A + C = 900) :=
by
  sorry

end total_tickets_sold_l408_40873


namespace volleyball_tournament_l408_40855

theorem volleyball_tournament (n : ℕ) (h : n * (n - 1) / 2 = 28) : n = 8 :=
sorry

end volleyball_tournament_l408_40855


namespace max_value_of_f_period_of_f_not_monotonically_increasing_incorrect_zeros_l408_40827

noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.cos x

theorem max_value_of_f : ∃ x, (f x) = 1/2 :=
sorry

theorem period_of_f : ∀ x, f (x + π) = f x :=
sorry

theorem not_monotonically_increasing : ¬ ∀ x y, 0 < x ∧ x < y ∧ y < π/2 → f x < f y :=
sorry

theorem incorrect_zeros : ∃ x y z, (0 ≤ x ∧ x < y ∧ y < z ∧ z ≤ π) ∧ (f x = 0 ∧ f y = 0 ∧ f z = 0) :=
sorry

end max_value_of_f_period_of_f_not_monotonically_increasing_incorrect_zeros_l408_40827


namespace least_pos_integer_with_8_factors_l408_40872

theorem least_pos_integer_with_8_factors : 
  ∃ k : ℕ, (k > 0 ∧ ((∃ m n p q : ℕ, k = p^m * q^n ∧ p ≠ q ∧ Prime p ∧ Prime q ∧ m + 1 = 4 ∧ n + 1 = 2) 
                     ∨ (∃ p : ℕ, k = p^7 ∧ Prime p)) ∧ 
            ∀ l : ℕ, (l > 0 ∧ ((∃ m n p q : ℕ, l = p^m * q^n ∧ p ≠ q ∧ Prime p ∧ Prime q ∧ m + 1 = 4 ∧ n + 1 = 2) 
                     ∨ (∃ p : ℕ, l = p^7 ∧ Prime p)) → k ≤ l)) ∧ k = 24 :=
sorry

end least_pos_integer_with_8_factors_l408_40872


namespace Mary_put_crayons_l408_40804

def initial_crayons : ℕ := 7
def final_crayons : ℕ := 10
def added_crayons (i f : ℕ) : ℕ := f - i

theorem Mary_put_crayons :
  added_crayons initial_crayons final_crayons = 3 := 
by
  sorry

end Mary_put_crayons_l408_40804


namespace increasing_geometric_progression_l408_40833

noncomputable def golden_ratio : ℝ := (Real.sqrt 5 + 1) / 2

theorem increasing_geometric_progression (a : ℝ) (ha : 0 < a)
  (h1 : ∃ b c q : ℝ, b = Int.floor a ∧ c = a - b ∧ a = b + c ∧ c = b * q ∧ a = c * q ∧ 1 < q) : 
  a = golden_ratio :=
sorry

end increasing_geometric_progression_l408_40833


namespace center_of_circle_l408_40807

theorem center_of_circle :
  ∀ (x y : ℝ), (x + 2)^2 + (y - 1)^2 = 1 → (x = -2 ∧ y = 1) :=
by
  intros x y hyp
  -- Here, we would perform the steps of comparing to the standard form and proving the center.
  sorry

end center_of_circle_l408_40807


namespace number_of_four_digit_numbers_l408_40887

/-- The number of four-digit numbers greater than 3999 where the product of the middle two digits exceeds 8 is 2400. -/
def four_digit_numbers_with_product_exceeding_8 : Nat :=
  let first_digit_options := 6  -- Choices are 4,5,6,7,8,9
  let last_digit_options := 10  -- Choices are 0,1,...,9
  let middle_digit_options := 40  -- Valid pairs counted from the solution
  first_digit_options * middle_digit_options * last_digit_options

theorem number_of_four_digit_numbers : four_digit_numbers_with_product_exceeding_8 = 2400 :=
by
  sorry

end number_of_four_digit_numbers_l408_40887


namespace integral_curves_l408_40808

theorem integral_curves (y x : ℝ) : 
  (∃ k : ℝ, (y - x) / (y + x) = k) → 
  (∃ c : ℝ, y = x * (c + 1) / (c - 1)) ∨ (y = 0) ∨ (y = x) ∨ (x = 0) :=
by
  sorry

end integral_curves_l408_40808


namespace evaluate_f_at_3_l408_40868

noncomputable def f : ℝ → ℝ := sorry

axiom h_odd : ∀ x : ℝ, f (-x) = -f x 
axiom h_periodic : ∀ x : ℝ, f x = f (x + 4)
axiom h_def : ∀ x : ℝ, 0 < x ∧ x < 2 → f x = 2 * x^2

theorem evaluate_f_at_3 : f 3 = -2 := by
  sorry

end evaluate_f_at_3_l408_40868


namespace determine_plane_by_trapezoid_legs_l408_40805

-- Defining basic objects
structure Point := (x : ℝ) (y : ℝ) (z : ℝ)
structure Line := (p1 : Point) (p2 : Point)
structure Plane := (l1 : Line) (l2 : Line)

-- Theorem statement for the problem
theorem determine_plane_by_trapezoid_legs (trapezoid_legs : Line) :
  ∃ (pl : Plane), ∀ (l1 l2 : Line), (l1 = trapezoid_legs) ∧ (l2 = trapezoid_legs) → (pl = Plane.mk l1 l2) :=
sorry

end determine_plane_by_trapezoid_legs_l408_40805


namespace negation_of_exists_l408_40840

theorem negation_of_exists:
  (¬ ∃ x₀ : ℝ, x₀^2 - x₀ + 1 < 0) ↔ (∀ x : ℝ, x^2 - x + 1 ≥ 0) := sorry

end negation_of_exists_l408_40840


namespace Leonard_is_11_l408_40812

def Leonard_age (L N J P T: ℕ) : Prop :=
  (L = N - 4) ∧
  (N = J / 2) ∧
  (P = 2 * L) ∧
  (T = P - 3) ∧
  (L + N + J + P + T = 75)

theorem Leonard_is_11 (L N J P T : ℕ) (h : Leonard_age L N J P T) : L = 11 :=
by {
  sorry
}

end Leonard_is_11_l408_40812
