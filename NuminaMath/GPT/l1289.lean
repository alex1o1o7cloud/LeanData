import Mathlib

namespace average_of_three_numbers_l1289_128937

theorem average_of_three_numbers
  (a b c : ℕ)
  (h1 : 2 * a + b + c = 130)
  (h2 : a + 2 * b + c = 138)
  (h3 : a + b + 2 * c = 152) :
  (a + b + c) / 3 = 35 :=
by
  sorry

end average_of_three_numbers_l1289_128937


namespace range_a_ff_a_eq_2_f_a_l1289_128943

noncomputable def f (x : ℝ) : ℝ :=
if x < 1 then 3 * x - 1 else 2 ^ x

theorem range_a_ff_a_eq_2_f_a :
  {a : ℝ | f (f a) = 2 ^ (f a)} = {a : ℝ | a ≥ 2/3} :=
sorry

end range_a_ff_a_eq_2_f_a_l1289_128943


namespace emissions_from_tap_water_l1289_128973

def carbon_dioxide_emission (x : ℕ) : ℕ := 9 / 10 * x  -- Note: using 9/10 instead of 0.9 to maintain integer type

theorem emissions_from_tap_water : carbon_dioxide_emission 10 = 9 :=
by
  sorry

end emissions_from_tap_water_l1289_128973


namespace intersection_A_B_l1289_128938

def A : Set ℕ := {0, 1, 2, 3}
def B : Set ℕ := {x | 0 < x ∧ x ≤ 2}

theorem intersection_A_B : A ∩ B = {1, 2} :=
by
  sorry

end intersection_A_B_l1289_128938


namespace log_sum_eq_l1289_128951

theorem log_sum_eq : ∀ (x y : ℝ), y = 2016 * x ∧ x^y = y^x → (Real.logb 2016 x + Real.logb 2016 y) = 2017 / 2015 :=
by
  intros x y h
  sorry

end log_sum_eq_l1289_128951


namespace intersection_eq_l1289_128931

open Set

variable (A B : Set ℝ)

def setA : A = {x | -3 < x ∧ x < 2} := sorry

def setB : B = {x | x^2 + 4*x - 5 ≤ 0} := sorry

theorem intersection_eq : A ∩ B = {x | -3 < x ∧ x ≤ 1} :=
sorry

end intersection_eq_l1289_128931


namespace expression_value_l1289_128949

noncomputable def expr := (1.90 * (1 / (1 - (3: ℝ)^(1/4)))) + (1 / (1 + (3: ℝ)^(1/4))) + (2 / (1 + (3: ℝ)^(1/2)))

theorem expression_value : expr = -2 := 
by
  sorry

end expression_value_l1289_128949


namespace total_wait_time_l1289_128992

def customs_wait : ℕ := 20
def quarantine_days : ℕ := 14
def hours_per_day : ℕ := 24

theorem total_wait_time :
  customs_wait + quarantine_days * hours_per_day = 356 := 
by
  sorry

end total_wait_time_l1289_128992


namespace line_through_two_points_l1289_128921

-- Define the points
def p1 : ℝ × ℝ := (1, 0)
def p2 : ℝ × ℝ := (0, -2)

-- Define the equation of the line passing through the points
def line_equation (x y : ℝ) : Prop :=
  2 * x - y - 2 = 0

-- The main theorem
theorem line_through_two_points : ∀ x y, p1 = (1, 0) ∧ p2 = (0, -2) → line_equation x y :=
  by sorry

end line_through_two_points_l1289_128921


namespace find_a_perpendicular_line_l1289_128905

theorem find_a_perpendicular_line (a : ℝ) : 
  (∀ x y : ℝ, (a * x + 3 * y + 1 = 0) → (2 * x + 2 * y - 3 = 0) → (-(a / 3) * (-1) = -1)) → 
  a = -3 :=
by
  sorry

end find_a_perpendicular_line_l1289_128905


namespace matrix_vector_product_l1289_128962

-- Definitions for matrix A and vector v
def A : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![-3, 4],
  ![2, -1]
]

def v : Fin 2 → ℤ := ![2, -2]

-- The theorem to prove
theorem matrix_vector_product :
  (A.mulVec v) = ![-14, 6] :=
by sorry

end matrix_vector_product_l1289_128962


namespace smallest_divisor_l1289_128929

-- Define the given number and the subtracting number
def original_num : ℕ := 378461
def subtract_num : ℕ := 5

-- Define the resulting number after subtraction
def resulting_num : ℕ := original_num - subtract_num

-- Theorem stating that 47307 is the smallest divisor greater than 5 of 378456
theorem smallest_divisor : ∃ d: ℕ, d > 5 ∧ d ∣ resulting_num ∧ ∀ x: ℕ, x > 5 → x ∣ resulting_num → d ≤ x := 
sorry

end smallest_divisor_l1289_128929


namespace work_completion_days_l1289_128953

theorem work_completion_days (A B : ℕ) (hB : B = 12) (work_together_days : ℕ) (work_together : work_together_days = 3) (work_alone_days : ℕ) (work_alone : work_alone_days = 3) : 
  (1 / A + 1 / B) * 3 + (1 / B) * 3 = 1 → A = 6 := 
by 
  intro h
  sorry

end work_completion_days_l1289_128953


namespace sum_first_five_terms_l1289_128961

noncomputable def geometric_sequence (a1 q : ℝ) (n : ℕ) : ℝ := a1 * q^(n-1)

noncomputable def sum_geometric_sequence (a1 q : ℝ) (n : ℕ) : ℝ := 
  if q = 1 then a1 * n else a1 * (1 - q^n) / (1 - q)

theorem sum_first_five_terms (a1 q : ℝ) 
  (h1 : geometric_sequence a1 q 2 * geometric_sequence a1 q 3 = 2 * a1)
  (h2 : (geometric_sequence a1 q 4 + 2 * geometric_sequence a1 q 7) / 2 = 5 / 4)
  : sum_geometric_sequence a1 q 5 = 31 :=
sorry

end sum_first_five_terms_l1289_128961


namespace Heath_current_age_l1289_128934

variable (H J : ℕ) -- Declare variables for Heath's and Jude's ages
variable (h1 : J = 2) -- Jude's current age is 2
variable (h2 : H + 5 = 3 * (J + 5)) -- In 5 years, Heath will be 3 times as old as Jude

theorem Heath_current_age : H = 16 :=
by
  -- Proof to be filled in later
  sorry

end Heath_current_age_l1289_128934


namespace subset_M_N_l1289_128995

def M : Set ℝ := {-1, 1}
def N : Set ℝ := { x | (1 / x < 2) }

theorem subset_M_N : M ⊆ N :=
by
  sorry -- Proof omitted as per the guidelines

end subset_M_N_l1289_128995


namespace grace_have_30_pastries_l1289_128926

theorem grace_have_30_pastries (F : ℕ) :
  (2 * (F + 8) + F + (F + 13) = 97) → (F + 13 = 30) :=
by
  sorry

end grace_have_30_pastries_l1289_128926


namespace gray_region_area_is_96pi_l1289_128927

noncomputable def smaller_circle_diameter : ℝ := 4

noncomputable def smaller_circle_radius : ℝ := smaller_circle_diameter / 2

noncomputable def larger_circle_radius : ℝ := 5 * smaller_circle_radius

noncomputable def area_of_larger_circle : ℝ := Real.pi * (larger_circle_radius ^ 2)

noncomputable def area_of_smaller_circle : ℝ := Real.pi * (smaller_circle_radius ^ 2)

noncomputable def area_of_gray_region : ℝ := area_of_larger_circle - area_of_smaller_circle

theorem gray_region_area_is_96pi : area_of_gray_region = 96 * Real.pi := by
  sorry

end gray_region_area_is_96pi_l1289_128927


namespace competition_order_l1289_128997

variable (A B C D : ℕ)

-- Conditions as given in the problem
axiom cond1 : B + D = 2 * A
axiom cond2 : A + C < B + D
axiom cond3 : A < B + C

-- The desired proof statement
theorem competition_order : D > B ∧ B > A ∧ A > C :=
by
  sorry

end competition_order_l1289_128997


namespace tetrahedron_coloring_l1289_128959

noncomputable def count_distinct_tetrahedron_colorings : ℕ :=
  sorry

theorem tetrahedron_coloring :
  count_distinct_tetrahedron_colorings = 6 :=
  sorry

end tetrahedron_coloring_l1289_128959


namespace lcm_18_27_l1289_128967

theorem lcm_18_27 : Nat.lcm 18 27 = 54 :=
by {
  sorry
}

end lcm_18_27_l1289_128967


namespace quadratic_real_solutions_l1289_128989

theorem quadratic_real_solutions (p : ℝ) : (∃ x : ℝ, x^2 + p = 0) ↔ p ≤ 0 :=
sorry

end quadratic_real_solutions_l1289_128989


namespace value_of_m_l1289_128907

theorem value_of_m (m x : ℝ) (h1 : mx + 1 = 2 * (m - x)) (h2 : |x + 2| = 0) : m = -|3 / 4| :=
by
  sorry

end value_of_m_l1289_128907


namespace pow_mod_eq_l1289_128971

theorem pow_mod_eq :
  11 ^ 2023 % 5 = 1 :=
by
  sorry

end pow_mod_eq_l1289_128971


namespace base_length_of_vessel_l1289_128969

def volume_of_cube (edge : ℝ) := edge^3

def volume_of_displaced_water (L width rise : ℝ) := L * width * rise

theorem base_length_of_vessel (edge width rise L : ℝ) 
  (h1 : edge = 15) (h2 : width = 15) (h3 : rise = 11.25) 
  (h4 : volume_of_displaced_water L width rise = volume_of_cube edge) : 
  L = 20 :=
by
  sorry

end base_length_of_vessel_l1289_128969


namespace geometric_sequence_a2_a6_l1289_128976

variable (a : ℕ → ℝ) (r : ℝ) (a1 : ℝ)
variable (a_geom_seq : ∀ n, a n = a1 * r^(n-1))
variable (h_a4 : a 4 = 4)

theorem geometric_sequence_a2_a6 : a 2 * a 6 = 16 :=
by
  -- Proof goes here
  sorry

end geometric_sequence_a2_a6_l1289_128976


namespace fractional_equation_solution_l1289_128902

theorem fractional_equation_solution (x : ℝ) (h : x ≠ 2) :
  (1 - x) / (2 - x) - 1 = (2 * x - 5) / (x - 2) → x = 3 :=
by 
  intro h_eq
  sorry

end fractional_equation_solution_l1289_128902


namespace isabella_total_haircut_length_l1289_128987

theorem isabella_total_haircut_length :
  (18 - 14) + (14 - 9) = 9 := 
sorry

end isabella_total_haircut_length_l1289_128987


namespace friends_total_earnings_l1289_128911

def Lauryn_earnings : ℝ := 2000
def Aurelia_fraction : ℝ := 0.7

def Aurelia_earnings : ℝ := Aurelia_fraction * Lauryn_earnings

def total_earnings : ℝ := Lauryn_earnings + Aurelia_earnings

theorem friends_total_earnings : total_earnings = 3400 := by
  -- We defined everything necessary here; the exact proof steps are omitted as per instructions.
  sorry

end friends_total_earnings_l1289_128911


namespace interest_rate_proven_l1289_128986

structure InvestmentProblem where
  P : ℝ  -- Principal amount
  A : ℝ  -- Accumulated amount
  n : ℕ  -- Number of times interest is compounded per year
  t : ℕ  -- Time in years
  rate : ℝ  -- Interest rate per annum (to be proven)

noncomputable def solve_interest_rate (ip : InvestmentProblem) : ℝ :=
  let half_yearly_rate := ip.rate / 2 / 100
  let amount_formula := ip.P * (1 + half_yearly_rate)^(ip.n * ip.t)
  half_yearly_rate

theorem interest_rate_proven :
  ∀ (P A : ℝ) (n t : ℕ), 
  P = 6000 → 
  A = 6615 → 
  n = 2 → 
  t = 1 → 
  solve_interest_rate {P := P, A := A, n := n, t := t, rate := 10.0952} = 10.0952 := 
by 
  intros
  rw [solve_interest_rate]
  sorry

end interest_rate_proven_l1289_128986


namespace toys_ratio_l1289_128958

theorem toys_ratio (k A M T : ℕ) (h1 : M = 6) (h2 : A = k * M) (h3 : A = T - 2) (h4 : A + M + T = 56):
  A / M = 4 :=
by
  sorry

end toys_ratio_l1289_128958


namespace Arianna_time_at_work_l1289_128946

theorem Arianna_time_at_work : 
  (24 - (5 + 13)) = 6 := 
by 
  sorry

end Arianna_time_at_work_l1289_128946


namespace sum_of_intersection_coordinates_l1289_128952

noncomputable def h : ℝ → ℝ := sorry

theorem sum_of_intersection_coordinates : 
  (∃ a b : ℝ, h a = h (a + 2) ∧ h 1 = 3 ∧ h (-1) = 3 ∧ a = -1 ∧ b = 3) → -1 + 3 = 2 :=
by
  intro h_assumptions
  sorry

end sum_of_intersection_coordinates_l1289_128952


namespace ratio_of_saute_times_l1289_128906

-- Definitions
def time_saute_onions : ℕ := 20
def time_saute_garlic_and_peppers : ℕ := 5
def time_knead_dough : ℕ := 30
def time_rest_dough : ℕ := 2 * time_knead_dough
def combined_knead_rest_time : ℕ := time_knead_dough + time_rest_dough
def time_assemble_calzones : ℕ := combined_knead_rest_time / 10
def total_time : ℕ := 124

-- Conditions
axiom saute_time_condition : time_saute_onions + time_saute_garlic_and_peppers + time_knead_dough + time_rest_dough + time_assemble_calzones = total_time

-- Question to be proved as a theorem
theorem ratio_of_saute_times :
  (time_saute_garlic_and_peppers : ℚ) / time_saute_onions = 1 / 4 :=
by
  -- proof goes here
  sorry

end ratio_of_saute_times_l1289_128906


namespace correct_diagram_l1289_128998

-- Definitions based on the conditions
def word : String := "KANGAROO"
def diagrams : List (String × Bool) :=
  [("Diagram A", False), ("Diagram B", False), ("Diagram C", False),
   ("Diagram D", False), ("Diagram E", True)]

-- Statement to prove that Diagram E correctly shows "KANGAROO"
theorem correct_diagram :
  ∃ d, (d.1 = "Diagram E") ∧ d.2 = True ∧ d ∈ diagrams :=
by
-- skipping the proof for now
sorry

end correct_diagram_l1289_128998


namespace soccer_league_equation_l1289_128977

noncomputable def equation_represents_soccer_league (x : ℕ) : Prop :=
  ∀ x : ℕ, (x * (x - 1)) / 2 = 50

theorem soccer_league_equation (x : ℕ) (h : equation_represents_soccer_league x) :
  (x * (x - 1)) / 2 = 50 :=
  by sorry

end soccer_league_equation_l1289_128977


namespace percentage_donated_l1289_128957

def income : ℝ := 1200000
def children_percentage : ℝ := 0.20
def wife_percentage : ℝ := 0.30
def remaining : ℝ := income - (children_percentage * 3 * income + wife_percentage * income)
def left_amount : ℝ := 60000
def donated : ℝ := remaining - left_amount

theorem percentage_donated : (donated / remaining) * 100 = 50 := by
  sorry

end percentage_donated_l1289_128957


namespace range_of_m_l1289_128944

theorem range_of_m (m : ℝ) : 
  (∀ x : ℤ, (x > 3 - m) ∧ (x ≤ 5) ↔ (1 ≤ x ∧ x ≤ 5)) →
  (2 < m ∧ m ≤ 3) := 
by
  sorry

end range_of_m_l1289_128944


namespace Buratino_math_problem_l1289_128999

theorem Buratino_math_problem (x : ℝ) : 4 * x + 15 = 15 * x + 4 → x = 1 :=
by
  intro h
  sorry

end Buratino_math_problem_l1289_128999


namespace find_solutions_in_positive_integers_l1289_128901

theorem find_solutions_in_positive_integers :
  ∃ a b c x y z : ℕ,
  a > 0 ∧ b > 0 ∧ c > 0 ∧ x > 0 ∧ y > 0 ∧ z > 0 ∧
  a + b + c = x * y * z ∧ x + y + z = a * b * c ∧
  ((a = 3 ∧ b = 2 ∧ c = 1 ∧ x = 3 ∧ y = 2 ∧ z = 1) ∨
   (a = 5 ∧ b = 2 ∧ c = 1 ∧ x = 3 ∧ y = 3 ∧ z = 1) ∨
   (a = 3 ∧ b = 3 ∧ c = 1 ∧ x = 5 ∧ y = 2 ∧ z = 1)) :=
sorry

end find_solutions_in_positive_integers_l1289_128901


namespace max_profit_is_45_6_l1289_128918

noncomputable def profit_A (x : ℝ) : ℝ := 5.06 * x - 0.15 * x^2
noncomputable def profit_B (x : ℝ) : ℝ := 2 * x

noncomputable def total_profit (x : ℝ) : ℝ :=
  profit_A x + profit_B (15 - x)

theorem max_profit_is_45_6 : 
  ∃ x, 0 ≤ x ∧ x ≤ 15 ∧ total_profit x = 45.6 :=
by
  sorry

end max_profit_is_45_6_l1289_128918


namespace greatest_3_digit_base_8_divisible_by_7_l1289_128923

open Nat

def is_3_digit_base_8 (n : ℕ) : Prop := n < 8^3

def is_divisible_by_7 (n : ℕ) : Prop := 7 ∣ n

theorem greatest_3_digit_base_8_divisible_by_7 :
  ∃ x : ℕ, is_3_digit_base_8 x ∧ is_divisible_by_7 x ∧ x = 7 * (8 * (8 * 7 + 7) + 7) :=
by
  sorry

end greatest_3_digit_base_8_divisible_by_7_l1289_128923


namespace ratio_y_share_to_total_l1289_128940

theorem ratio_y_share_to_total
  (total_profit : ℝ)
  (diff_share : ℝ)
  (h_total : total_profit = 800)
  (h_diff : diff_share = 160) :
  ∃ (a b : ℝ), (b / (a + b) = 2 / 5) ∧ (|a - b| = (a + b) / 5) :=
by
  sorry

end ratio_y_share_to_total_l1289_128940


namespace cara_between_pairs_l1289_128908

-- Definitions based on the conditions
def friends := 7 -- Cara has 7 friends
def fixed_neighbor : Prop := true -- Alex must always be one of the neighbors

-- Problem statement to be proven
theorem cara_between_pairs (h : fixed_neighbor): 
  ∃ n : ℕ, n = 6 ∧ (1 + (friends - 1)) = n := by
  sorry

end cara_between_pairs_l1289_128908


namespace Petya_can_verify_coins_l1289_128980

theorem Petya_can_verify_coins :
  ∃ (c₁ c₂ c₃ c₅ : ℕ), 
  (c₁ = 1 ∧ c₂ = 2 ∧ c₃ = 3 ∧ c₅ = 5) ∧
  (∃ (w : ℕ), w = 9) ∧
  (∃ (cond : ℕ → Prop), 
    cond 1 ∧ cond 2 ∧ cond 3 ∧ cond 5) := sorry

end Petya_can_verify_coins_l1289_128980


namespace min_elements_in_as_l1289_128941

noncomputable def min_elems_in_A_s (n : ℕ) (S : Finset ℝ) (hS : S.card = n) : ℕ :=
  if 2 ≤ n then 2 * n - 3 else 0

theorem min_elements_in_as (n : ℕ) (S : Finset ℝ) (hS : S.card = n) (hn: 2 ≤ n) :
  ∃ (A_s : Finset ℝ), A_s.card = min_elems_in_A_s n S hS := sorry

end min_elements_in_as_l1289_128941


namespace solve_system_l1289_128979

theorem solve_system (s t : ℚ) (h1 : 7 * s + 6 * t = 156) (h2 : s = t / 2 + 3) : s = 192 / 19 :=
sorry

end solve_system_l1289_128979


namespace max_value_of_expr_l1289_128983

noncomputable def max_expr_value (x : ℝ) : ℝ :=
  x^6 / (x^12 + 4*x^9 - 6*x^6 + 16*x^3 + 64)

theorem max_value_of_expr : ∀ x : ℝ, max_expr_value x ≤ 1/26 :=
by
  sorry

end max_value_of_expr_l1289_128983


namespace symmetric_point_m_eq_one_l1289_128991

theorem symmetric_point_m_eq_one (m : ℝ) (A B : ℝ × ℝ) 
  (hA : A = (-3, 2 * m - 1))
  (hB : B = (-3, -1))
  (symmetric : A.1 = B.1 ∧ A.2 = -B.2) : 
  m = 1 :=
by
  sorry

end symmetric_point_m_eq_one_l1289_128991


namespace not_possible_total_l1289_128933

-- Definitions
variables (d r : ℕ)

-- Theorem to prove that 58 cannot be expressed as 26d + 3r
theorem not_possible_total : ¬∃ (d r : ℕ), 26 * d + 3 * r = 58 :=
sorry

end not_possible_total_l1289_128933


namespace tom_read_chapters_l1289_128955

theorem tom_read_chapters (chapters pages: ℕ) (h1: pages = 8 * chapters) (h2: pages = 24):
  chapters = 3 :=
by
  sorry

end tom_read_chapters_l1289_128955


namespace solve_equation1_solve_equation2_solve_equation3_l1289_128935

-- For equation x^2 + 2x = 5
theorem solve_equation1 (x : ℝ) : x^2 + 2 * x = 5 ↔ (x = -1 + Real.sqrt 6) ∨ (x = -1 - Real.sqrt 6) :=
sorry

-- For equation x^2 - 2x - 1 = 0
theorem solve_equation2 (x : ℝ) : x^2 - 2 * x - 1 = 0 ↔ (x = 1 + Real.sqrt 2) ∨ (x = 1 - Real.sqrt 2) :=
sorry

-- For equation 2x^2 + 3x - 5 = 0
theorem solve_equation3 (x : ℝ) : 2 * x^2 + 3 * x - 5 = 0 ↔ (x = -5 / 2) ∨ (x = 1) :=
sorry

end solve_equation1_solve_equation2_solve_equation3_l1289_128935


namespace sin_cos_acute_angle_lt_one_l1289_128978

theorem sin_cos_acute_angle_lt_one (α β : ℝ) (a b c : ℝ) (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2) (h_triangle : a^2 + b^2 = c^2) (h_nonzero_c : c ≠ 0) :
  (a / c < 1) ∧ (b / c < 1) :=
by 
  sorry

end sin_cos_acute_angle_lt_one_l1289_128978


namespace total_capacity_is_correct_l1289_128963

-- Define small and large jars capacities
def small_jar_capacity : ℕ := 3
def large_jar_capacity : ℕ := 5

-- Define the total number of jars and the number of small jars
def total_jars : ℕ := 100
def small_jars : ℕ := 62

-- Define the number of large jars based on the total jars and small jars
def large_jars : ℕ := total_jars - small_jars

-- Calculate capacities
def small_jars_total_capacity : ℕ := small_jars * small_jar_capacity
def large_jars_total_capacity : ℕ := large_jars * large_jar_capacity

-- Define the total capacity
def total_capacity : ℕ := small_jars_total_capacity + large_jars_total_capacity

-- Prove that the total capacity is 376 liters
theorem total_capacity_is_correct : total_capacity = 376 := by
  sorry

end total_capacity_is_correct_l1289_128963


namespace exponent_neg_power_l1289_128985

theorem exponent_neg_power (a : ℝ) : -(a^3)^4 = -a^(3 * 4) := 
by
  sorry

end exponent_neg_power_l1289_128985


namespace rod_center_of_gravity_shift_l1289_128924

noncomputable def rod_shift (l : ℝ) (s : ℝ) : ℝ := 
  |(l / 2) - ((l - s) / 2)| 

theorem rod_center_of_gravity_shift : 
  rod_shift l 80 = 40 := by
  sorry

end rod_center_of_gravity_shift_l1289_128924


namespace least_integer_square_eq_12_more_than_three_times_l1289_128993

theorem least_integer_square_eq_12_more_than_three_times (x : ℤ) (h : x^2 = 3 * x + 12) : x = -3 :=
sorry

end least_integer_square_eq_12_more_than_three_times_l1289_128993


namespace isosceles_triangle_perimeter_l1289_128972

theorem isosceles_triangle_perimeter :
  ∃ P : ℕ, (P = 15 ∨ P = 18) ∧ ∀ (a b c : ℕ), (a = 7 ∨ b = 7 ∨ c = 7) ∧ (a = 4 ∨ b = 4 ∨ c = 4) → ((a = 7 ∨ a = 4) ∧ (b = 7 ∨ b = 4) ∧ (c = 7 ∨ c = 4)) ∧ P = a + b + c :=
by
  sorry

end isosceles_triangle_perimeter_l1289_128972


namespace sixth_number_is_eight_l1289_128968

/- 
  The conditions are:
  1. The sequence is an increasing list of consecutive integers.
  2. The 3rd and 4th numbers add up to 11.
  We need to prove that the 6th number is 8.
-/

theorem sixth_number_is_eight (n : ℕ) (h : n + (n + 1) = 11) : (n + 3) = 8 :=
by
  sorry

end sixth_number_is_eight_l1289_128968


namespace problem_statement_l1289_128947

noncomputable def tan_plus_alpha_half_pi (α : ℝ) : ℝ := -1 / (Real.tan α)

theorem problem_statement (α : ℝ) (h : tan_plus_alpha_half_pi α = -1 / 2) :
  (2 * Real.sin α + Real.cos α) / (Real.cos α - Real.sin α) = -5 := by
  sorry

end problem_statement_l1289_128947


namespace greater_number_is_18_l1289_128966

theorem greater_number_is_18 (x y : ℝ) 
  (h1 : x + y = 30) 
  (h2 : x - y = 6) 
  (h3 : y ≥ 10) : 
  x = 18 := 
by 
  sorry

end greater_number_is_18_l1289_128966


namespace exchange_5_rubles_l1289_128960

theorem exchange_5_rubles :
  ¬ ∃ n : ℕ, 1 * n + 2 * n + 3 * n + 5 * n = 500 :=
by 
  sorry

end exchange_5_rubles_l1289_128960


namespace red_marbles_more_than_yellow_l1289_128974

-- Define the conditions
def total_marbles : ℕ := 19
def yellow_marbles : ℕ := 5
def ratio_parts_blue : ℕ := 3
def ratio_parts_red : ℕ := 4

-- Prove the number of more red marbles than yellow marbles is equal to 3
theorem red_marbles_more_than_yellow :
  let remainder_marbles := total_marbles - yellow_marbles
  let total_parts := ratio_parts_blue + ratio_parts_red
  let marbles_per_part := remainder_marbles / total_parts
  let blue_marbles := ratio_parts_blue * marbles_per_part
  let red_marbles := ratio_parts_red * marbles_per_part
  red_marbles - yellow_marbles = 3 := by
  sorry

end red_marbles_more_than_yellow_l1289_128974


namespace find_min_a_l1289_128910

theorem find_min_a (a : ℕ) (h1 : (3150 * a) = x^2) (h2 : a > 0) :
  a = 14 := by
  sorry

end find_min_a_l1289_128910


namespace part1_part2_part3_l1289_128914

variable {α : Type} [LinearOrderedField α]

noncomputable def f (x : α) : α := sorry  -- as we won't define it explicitly, we use sorry

axiom f_conditions : ∀ (u v : α), - 1 ≤ u ∧ u ≤ 1 ∧ - 1 ≤ v ∧ v ≤ 1 → |f u - f v| ≤ |u - v|
axiom f_endpoints : f (-1 : α) = 0 ∧ f (1 : α) = 0

theorem part1 (x : α) (hx : -1 ≤ x ∧ x ≤ 1) : x - 1 ≤ f x ∧ f x ≤ 1 - x := by
  have hf : ∀ (u v : α), -1 ≤ u ∧ u ≤ 1 ∧ -1 ≤ v ∧ v ≤ 1 → |f u - f v| ≤ |u - v| := f_conditions
  sorry

theorem part2 (u v : α) (huv : -1 ≤ u ∧ u ≤ 1 ∧ -1 ≤ v ∧ v ≤ 1) : |f u - f v| ≤ 1 := by
  have hf : ∀ (u v : α), -1 ≤ u ∧ u ≤ 1 ∧ -1 ≤ v ∧ v ≤ 1 → |f u - f v| ≤ |u - v| := f_conditions
  sorry

theorem part3 : ¬ ∃ (f : α → α), (∀ (u v : α), - 1 ≤ u ∧ u ≤ 1 ∧ - 1 ≤ v ∧ v ≤ 1 → |f u - f v| ≤ |u - v| ∧ f (-1 : α) = 0 ∧ f (1 : α) = 0 ∧
  (∀ (x : α), - 1 ≤ x ∧ x ≤ 1 → f (- x) = - f x) ∧ -- odd function condition
  (∀ (u v : α), 0 ≤ u ∧ u ≤ 1/2 ∧ 0 ≤ v ∧ v ≤ 1/2 → |f u - f v| < |u - v|) ∧
  (∀ (u v : α), 1/2 ≤ u ∧ u ≤ 1 ∧ 1/2 ≤ v ∧ v ≤ 1 → |f u - f v| = |u - v|)) := by
  sorry

end part1_part2_part3_l1289_128914


namespace add_decimals_l1289_128942

theorem add_decimals :
  0.0935 + 0.007 + 0.2 = 0.3005 :=
by sorry

end add_decimals_l1289_128942


namespace sophia_book_pages_l1289_128939

theorem sophia_book_pages:
  ∃ (P : ℕ), (2 / 3 : ℚ) * P = (1 / 3 : ℚ) * P + 30 ∧ P = 90 :=
by
  sorry

end sophia_book_pages_l1289_128939


namespace solve_new_system_l1289_128912

theorem solve_new_system (a_1 b_1 a_2 b_2 c_1 c_2 x y : ℝ)
(h1 : a_1 * 2 - b_1 * (-1) = c_1)
(h2 : a_2 * 2 + b_2 * (-1) = c_2) :
  (x = -1) ∧ (y = 1) :=
by
  have hx : x + 3 = 2 := by sorry
  have hy : y - 2 = -1 := by sorry
  have hx_sol : x = -1 := by linarith
  have hy_sol : y = 1 := by linarith
  exact ⟨hx_sol, hy_sol⟩

end solve_new_system_l1289_128912


namespace storks_initial_count_l1289_128903

theorem storks_initial_count (S : ℕ) 
  (h1 : 6 = (S + 2) + 1) : S = 3 :=
sorry

end storks_initial_count_l1289_128903


namespace extreme_point_property_l1289_128913

variables (f : ℝ → ℝ) (a b x x₀ x₁ : ℝ) 

-- Define the function f
def func (x : ℝ) := x^3 - a * x - b

-- The main theorem
theorem extreme_point_property (h₀ : ∃ x₀, ∃ x₁, (x₀ ≠ 0) ∧ (x₀^2 = a / 3) ∧ (x₁ ≠ x₀) ∧ (func a b x₀ = func a b x₁)) :
  x₁ + 2 * x₀ = 0 :=
sorry

end extreme_point_property_l1289_128913


namespace geometric_sequence_problem_l1289_128948

variable (a_n : ℕ → ℝ)

def geometric_sequence (a₁ : ℝ) (q : ℝ) : ℕ → ℝ := λ n => a₁ * q^(n-1)

theorem geometric_sequence_problem (q a_1 : ℝ) (a_1_pos : a_1 = 9)
  (h : ∀ n, a_n n = geometric_sequence a_1 q n)
  (h5 : a_n 5 = a_n 3 * (a_n 4)^2) : 
  a_n 4 = 1/3 ∨ a_n 4 = -1/3 := by 
  sorry

end geometric_sequence_problem_l1289_128948


namespace sale_second_month_l1289_128900

def sale_first_month : ℝ := 5700
def sale_third_month : ℝ := 6855
def sale_fourth_month : ℝ := 3850
def sale_fifth_month : ℝ := 14045
def average_sale : ℝ := 7800

theorem sale_second_month : 
  ∃ x : ℝ, -- there exists a sale in the second month such that...
    (sale_first_month + x + sale_third_month + sale_fourth_month + sale_fifth_month) / 5 = average_sale
    ∧ x = 7550 := 
by
  sorry

end sale_second_month_l1289_128900


namespace problem_equiv_proof_l1289_128990

theorem problem_equiv_proof : ∀ (i : ℂ), i^2 = -1 → (1 + i^2017) / (1 - i) = i :=
by
  intro i h
  sorry

end problem_equiv_proof_l1289_128990


namespace John_used_16_bulbs_l1289_128982

variable (X : ℕ)

theorem John_used_16_bulbs
  (h1 : 40 - X = 2 * 12) :
  X = 16 := 
sorry

end John_used_16_bulbs_l1289_128982


namespace diagonal_length_of_octagon_l1289_128922

theorem diagonal_length_of_octagon 
  (r : ℝ) (s : ℝ) (has_symmetry_axes : ℕ) 
  (inscribed : r = 6) (side_length : s = 5) 
  (symmetry_condition : has_symmetry_axes = 4) : 
  ∃ (d : ℝ), d = 2 * Real.sqrt 40 := 
by 
  sorry

end diagonal_length_of_octagon_l1289_128922


namespace sum_of_first_9_primes_l1289_128928

theorem sum_of_first_9_primes : 
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19 + 23) = 100 := 
by
  sorry

end sum_of_first_9_primes_l1289_128928


namespace advertisement_length_l1289_128917

noncomputable def movie_length : ℕ := 90
noncomputable def replay_times : ℕ := 6
noncomputable def operation_time : ℕ := 660

theorem advertisement_length : ∃ A : ℕ, 90 * replay_times + 6 * A = operation_time ∧ A = 20 :=
by
  use 20
  sorry

end advertisement_length_l1289_128917


namespace sum_of_reciprocals_l1289_128964

theorem sum_of_reciprocals (x y : ℝ) (h : x + y = 6 * x * y) (hx : x ≠ 0) (hy : y ≠ 0) : 
  (1 / x) + (1 / y) = 2 := 
by
  sorry

end sum_of_reciprocals_l1289_128964


namespace problem_statement_l1289_128925

def f (x : ℝ) : ℝ := 5 * x + 2
def g (x : ℝ) : ℝ := 3 * x - 1

theorem problem_statement : g (f (g (f 1))) = 305 :=
by
  sorry

end problem_statement_l1289_128925


namespace integer_solution_system_eq_det_l1289_128970

theorem integer_solution_system_eq_det (a b c d : ℤ) 
  (h : ∀ m n : ℤ, ∃ x y : ℤ, a * x + b * y = m ∧ c * x + d * y = n) : 
  a * d - b * c = 1 ∨ a * d - b * c = -1 :=
by
  sorry

end integer_solution_system_eq_det_l1289_128970


namespace tangent_sum_l1289_128994

theorem tangent_sum (tan : ℝ → ℝ)
  (h1 : ∀ A B, tan (A + B) = (tan A + tan B) / (1 - tan A * tan B))
  (h2 : tan 60 = Real.sqrt 3) :
  tan 20 + tan 40 + Real.sqrt 3 * tan 20 * tan 40 = Real.sqrt 3 := 
by
  sorry

end tangent_sum_l1289_128994


namespace probability_compensation_l1289_128916

-- Define the probabilities of each vehicle getting into an accident
def p1 : ℚ := 1 / 20
def p2 : ℚ := 1 / 21

-- Define the probability of the complementary event
def comp_event : ℚ := (1 - p1) * (1 - p2)

-- Define the overall probability that at least one vehicle gets into an accident
def comp_unit : ℚ := 1 - comp_event

-- The theorem to be proved: the probability that the unit will receive compensation from this insurance within a year is 2 / 21
theorem probability_compensation : comp_unit = 2 / 21 :=
by
  -- giving the proof is not required
  sorry

end probability_compensation_l1289_128916


namespace range_of_a_l1289_128956

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, (x > 0) ∧ (π^x = (a + 1) / (2 - a))) → (1 / 2 < a ∧ a < 2) :=
by
  sorry

end range_of_a_l1289_128956


namespace circle1_correct_circle2_correct_l1289_128930

noncomputable def circle1_eq (x y : ℝ) : ℝ :=
  x^2 + y^2 + 4*x - 6*y - 12

noncomputable def circle2_eq (x y : ℝ) : ℝ :=
  36*x^2 + 36*y^2 - 24*x + 72*y + 31

theorem circle1_correct (x y : ℝ) :
  ((x + 2)^2 + (y - 3)^2 = 25) ↔ (circle1_eq x y = 0) :=
sorry

theorem circle2_correct (x y : ℝ) :
  (36 * ((x - 1/3)^2 + (y + 1)^2) = 9) ↔ (circle2_eq x y = 0) :=
sorry

end circle1_correct_circle2_correct_l1289_128930


namespace train_speed_correct_l1289_128975

def train_length : ℝ := 1500
def crossing_time : ℝ := 15
def correct_speed : ℝ := 100

theorem train_speed_correct : (train_length / crossing_time) = correct_speed := by 
  sorry

end train_speed_correct_l1289_128975


namespace fraction_of_employees_laid_off_l1289_128945

theorem fraction_of_employees_laid_off
    (total_employees : ℕ)
    (salary_per_employee : ℕ)
    (total_payment_after_layoffs : ℕ)
    (h1 : total_employees = 450)
    (h2 : salary_per_employee = 2000)
    (h3 : total_payment_after_layoffs = 600000) :
    (total_employees * salary_per_employee - total_payment_after_layoffs) / (total_employees * salary_per_employee) = 1 / 3 := 
by
    sorry

end fraction_of_employees_laid_off_l1289_128945


namespace capital_formula_minimum_m_l1289_128920

-- Define initial conditions
def initial_capital : ℕ := 50000  -- in thousand yuan
def annual_growth_rate : ℝ := 0.5
def submission_amount : ℕ := 10000  -- in thousand yuan

-- Define remaining capital after nth year
noncomputable def remaining_capital (n : ℕ) : ℝ :=
  4500 * (3 / 2)^(n - 1) + 2000  -- in thousand yuan

-- Prove the formula for a_n
theorem capital_formula (n : ℕ) : 
  remaining_capital n = 4500 * (3 / 2)^(n - 1) + 2000 := 
by
  sorry

-- Prove the minimum value of m for which a_m > 30000
theorem minimum_m (m : ℕ) : 
  remaining_capital m > 30000 ↔ m ≥ 6 := 
by
  sorry

end capital_formula_minimum_m_l1289_128920


namespace length_of_train_is_correct_l1289_128996

noncomputable def train_length (speed_km_hr : ℝ) (time_sec : ℝ) : ℝ :=
  let speed_m_s := (speed_km_hr * 1000) / 3600
  speed_m_s * time_sec

theorem length_of_train_is_correct (speed_km_hr : ℝ) (time_sec : ℝ) (expected_length : ℝ) :
  speed_km_hr = 60 → time_sec = 21 → expected_length = 350.07 →
  train_length speed_km_hr time_sec = expected_length :=
by
  intros h1 h2 h3
  simp [h1, h2, train_length]
  sorry

end length_of_train_is_correct_l1289_128996


namespace polynomial_value_l1289_128909

variable (x : ℝ)

theorem polynomial_value (h : x^2 - 2*x + 6 = 9) : 2*x^2 - 4*x + 6 = 12 :=
by
  sorry

end polynomial_value_l1289_128909


namespace no_adjacent_black_balls_l1289_128936

theorem no_adjacent_black_balls (m n : ℕ) (h : m > n) : 
  (m + 1).choose n = (m + 1).factorial / (n.factorial * (m + 1 - n).factorial) := by
  sorry

end no_adjacent_black_balls_l1289_128936


namespace abc_value_l1289_128988

noncomputable def find_abc (a b c : ℝ) (h1 : a * (b + c) = 152) (h2 : b * (c + a) = 162) (h3 : c * (a + b) = 170) : ℝ :=
  a * b * c

theorem abc_value (a b c : ℝ) (h1 : a * (b + c) = 152) (h2 : b * (c + a) = 162) (h3 : c * (a + b) = 170) :
  a * b * c = 720 :=
by
  -- We skip the proof by providing sorry.
  sorry

end abc_value_l1289_128988


namespace max_students_equal_division_l1289_128932

theorem max_students_equal_division (pens pencils : ℕ) (h_pens : pens = 640) (h_pencils : pencils = 520) : 
  Nat.gcd pens pencils = 40 :=
by
  rw [h_pens, h_pencils]
  have : Nat.gcd 640 520 = 40 := by norm_num
  exact this

end max_students_equal_division_l1289_128932


namespace find_rate_percent_l1289_128919

-- Given conditions as definitions
def SI : ℕ := 128
def P : ℕ := 800
def T : ℕ := 4

-- Define the formula for Simple Interest
def simple_interest (P R T : ℕ) : ℕ := (P * R * T) / 100

-- Define the rate percent we need to prove
def rate_percent : ℕ := 4

-- The theorem statement we need to prove
theorem find_rate_percent (h1 : simple_interest P rate_percent T = SI) : rate_percent = 4 := 
by sorry

end find_rate_percent_l1289_128919


namespace cost_price_of_product_l1289_128915

theorem cost_price_of_product (x y : ℝ)
  (h1 : 0.8 * y - x = 120)
  (h2 : 0.6 * y - x = -20) :
  x = 440 := sorry

end cost_price_of_product_l1289_128915


namespace books_remaining_correct_l1289_128981

-- Define the total number of books and the number of books read
def total_books : ℕ := 32
def books_read : ℕ := 17

-- Define the number of books remaining to be read
def books_remaining : ℕ := total_books - books_read

-- Prove that the number of books remaining to be read is 15
theorem books_remaining_correct : books_remaining = 15 := by
  sorry

end books_remaining_correct_l1289_128981


namespace find_b_l1289_128984

theorem find_b (A B C : ℝ) (a b c : ℝ)
  (h1 : Real.tan A = 1 / 3)
  (h2 : Real.tan B = 1 / 2)
  (h3 : a = 1)
  (h4 : A + B + C = π) -- This condition is added because angles in a triangle sum up to π.
  : b = Real.sqrt 2 :=
by
  sorry

end find_b_l1289_128984


namespace ways_to_divide_8_friends_l1289_128954

theorem ways_to_divide_8_friends : (4 : ℕ) ^ 8 = 65536 := by
  sorry

end ways_to_divide_8_friends_l1289_128954


namespace number_of_lucky_numbers_l1289_128950

-- Defining the concept of sequence with even number of digit 8
def is_lucky (seq : List ℕ) : Prop :=
  seq.count 8 % 2 = 0

-- Define S(n) recursive formula
noncomputable def S : ℕ → ℝ
| 0 => 0
| n+1 => 4 * (1 - (1 / (2 ^ (n+1))))

theorem number_of_lucky_numbers (n : ℕ) :
  ∀ (seq : List ℕ), (seq.length ≤ n) → is_lucky seq → S n = 4 * (1 - 1 / (2 ^ n)) :=
sorry

end number_of_lucky_numbers_l1289_128950


namespace robin_hair_length_l1289_128904

theorem robin_hair_length
  (l d g : ℕ)
  (h₁ : l = 16)
  (h₂ : d = 11)
  (h₃ : g = 12) :
  (l - d + g = 17) :=
by sorry

end robin_hair_length_l1289_128904


namespace garden_perimeter_l1289_128965

/-- Define the dimensions of the rectangle and triangle in the garden -/
def rectangle_length : ℕ := 10
def rectangle_width : ℕ := 4
def triangle_leg1 : ℕ := 3
def triangle_leg2 : ℕ := 4
def triangle_hypotenuse : ℕ := 5 -- calculated using Pythagorean theorem

/-- Prove that the total perimeter of the combined shape is 28 units -/
theorem garden_perimeter :
  let perimeter := 2 * rectangle_length + rectangle_width + triangle_leg1 + triangle_hypotenuse
  perimeter = 28 :=
by
  let perimeter := 2 * rectangle_length + rectangle_width + triangle_leg1 + triangle_hypotenuse
  have h : perimeter = 28 := sorry
  exact h

end garden_perimeter_l1289_128965
