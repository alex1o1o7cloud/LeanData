import Mathlib

namespace NUMINAMATH_GPT_geometric_sequence_ratio_l1519_151933

theorem geometric_sequence_ratio (a1 q : ℝ) (h : (a1 * (1 - q^3) / (1 - q)) / (a1 * (1 - q^2) / (1 - q)) = 3 / 2) :
  q = 1 ∨ q = -1 / 2 := by
  sorry

end NUMINAMATH_GPT_geometric_sequence_ratio_l1519_151933


namespace NUMINAMATH_GPT_scientific_notation_representation_l1519_151934

theorem scientific_notation_representation :
  1300000 = 1.3 * 10^6 :=
sorry

end NUMINAMATH_GPT_scientific_notation_representation_l1519_151934


namespace NUMINAMATH_GPT_unique_solution_exists_l1519_151908

theorem unique_solution_exists (k : ℝ) :
  (16 + 12 * k = 0) → ∃! x : ℝ, k * x^2 - 4 * x - 3 = 0 :=
by
  intro hk
  sorry

end NUMINAMATH_GPT_unique_solution_exists_l1519_151908


namespace NUMINAMATH_GPT_magic_island_red_parrots_l1519_151927

noncomputable def total_parrots : ℕ := 120

noncomputable def green_parrots : ℕ := (5 * total_parrots) / 8

noncomputable def non_green_parrots : ℕ := total_parrots - green_parrots

noncomputable def red_parrots : ℕ := non_green_parrots / 3

theorem magic_island_red_parrots : red_parrots = 15 :=
by
  sorry

end NUMINAMATH_GPT_magic_island_red_parrots_l1519_151927


namespace NUMINAMATH_GPT_lines_through_P_and_form_area_l1519_151983

-- Definition of the problem conditions
def passes_through_P (k b : ℝ) : Prop :=
  b = 2 - k

def forms_area_with_axes (k b : ℝ) : Prop :=
  b^2 = 8 * |k|

-- Theorem statement
theorem lines_through_P_and_form_area :
  ∃ (k1 k2 k3 b1 b2 b3 : ℝ),
    passes_through_P k1 b1 ∧ forms_area_with_axes k1 b1 ∧
    passes_through_P k2 b2 ∧ forms_area_with_axes k2 b2 ∧
    passes_through_P k3 b3 ∧ forms_area_with_axes k3 b3 ∧
    k1 ≠ k2 ∧ k2 ≠ k3 ∧ k1 ≠ k3 :=
sorry

end NUMINAMATH_GPT_lines_through_P_and_form_area_l1519_151983


namespace NUMINAMATH_GPT_fraction_of_4_is_8_l1519_151988

theorem fraction_of_4_is_8 (fraction : ℝ) (h : fraction * 4 = 8) : fraction = 8 := 
sorry

end NUMINAMATH_GPT_fraction_of_4_is_8_l1519_151988


namespace NUMINAMATH_GPT_financed_amount_correct_l1519_151915

-- Define the conditions
def monthly_payment : ℝ := 150.0
def years : ℝ := 5.0
def months_in_a_year : ℝ := 12.0

-- Define the total number of months
def total_months : ℝ := years * months_in_a_year

-- Define the amount financed
def total_financed : ℝ := monthly_payment * total_months

-- State the theorem
theorem financed_amount_correct :
  total_financed = 9000 :=
by
  -- Provide the proof here
  sorry

end NUMINAMATH_GPT_financed_amount_correct_l1519_151915


namespace NUMINAMATH_GPT_speed_calculation_l1519_151982

def distance := 600 -- in meters
def time := 2 -- in minutes

def distance_km := distance / 1000 -- converting meters to kilometers
def time_hr := time / 60 -- converting minutes to hours

theorem speed_calculation : (distance_km / time_hr = 18) :=
 by
  sorry

end NUMINAMATH_GPT_speed_calculation_l1519_151982


namespace NUMINAMATH_GPT_largest_4_digit_congruent_to_15_mod_25_l1519_151936

theorem largest_4_digit_congruent_to_15_mod_25 : 
  ∀ x : ℕ, (1000 ≤ x ∧ x < 10000 ∧ x % 25 = 15) → x = 9990 :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_largest_4_digit_congruent_to_15_mod_25_l1519_151936


namespace NUMINAMATH_GPT_only_one_correct_guess_l1519_151994

-- Define the contestants
inductive Contestant : Type
| person : ℕ → Contestant

def A_win_first (c: Contestant) : Prop :=
c = Contestant.person 4 ∨ c = Contestant.person 5

def B_not_win_first (c: Contestant) : Prop :=
c ≠ Contestant.person 3 

def C_win_first (c: Contestant) : Prop :=
c = Contestant.person 1 ∨ c = Contestant.person 2 ∨ c = Contestant.person 6

def D_not_win_first (c: Contestant) : Prop :=
c ≠ Contestant.person 4 ∧ c ≠ Contestant.person 5 ∧ c ≠ Contestant.person 6

-- The main theorem: Only one correct guess among A, B, C, and D
theorem only_one_correct_guess (win: Contestant) :
  (A_win_first win ↔ false) ∧ (B_not_win_first win ↔ false) ∧ (C_win_first win ↔ false) ∧ D_not_win_first win
:=
by
  sorry

end NUMINAMATH_GPT_only_one_correct_guess_l1519_151994


namespace NUMINAMATH_GPT_obtuse_triangle_l1519_151996

theorem obtuse_triangle (A B C M E : ℝ) (hM : M = (B + C) / 2) (hE : E > 0) 
(hcond : (B - E) ^ 2 + (C - E) ^ 2 >= 4 * (A - M) ^ 2): 
∃ α β γ, α > 90 ∧ β + γ < 90 ∧ α + β + γ = 180 :=
by
  sorry

end NUMINAMATH_GPT_obtuse_triangle_l1519_151996


namespace NUMINAMATH_GPT_polygon_number_of_sides_l1519_151909

theorem polygon_number_of_sides (P : ℝ) (L : ℝ) (n : ℕ) : P = 180 ∧ L = 15 ∧ n = P / L → n = 12 := by
  sorry

end NUMINAMATH_GPT_polygon_number_of_sides_l1519_151909


namespace NUMINAMATH_GPT_repeating_decimal_sum_in_lowest_terms_l1519_151925

noncomputable def repeating_decimal_to_fraction (s : String) : ℚ := sorry

theorem repeating_decimal_sum_in_lowest_terms :
  let x := repeating_decimal_to_fraction "0.2"
  let y := repeating_decimal_to_fraction "0.03"
  x + y = 25 / 99 := sorry

end NUMINAMATH_GPT_repeating_decimal_sum_in_lowest_terms_l1519_151925


namespace NUMINAMATH_GPT_simplify_expression_l1519_151989

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def c : ℝ := sorry

noncomputable def x : ℝ := (b / c) * (c / b)
noncomputable def y : ℝ := (a / c) * (c / a)
noncomputable def z : ℝ := (a / b) * (b / a)

theorem simplify_expression : x^2 + y^2 + z^2 + x^2 * y^2 * z^2 = 4 := 
by {
  sorry
}

end NUMINAMATH_GPT_simplify_expression_l1519_151989


namespace NUMINAMATH_GPT_math_problem_l1519_151973

theorem math_problem
  (numerator : ℕ := (Nat.factorial 10))
  (denominator : ℕ := (10 * 11 / 2)) :
  (numerator / denominator : ℚ) = 66069 + 1 / 11 := by
  sorry

end NUMINAMATH_GPT_math_problem_l1519_151973


namespace NUMINAMATH_GPT_find_b_amount_l1519_151937

theorem find_b_amount (A B : ℝ) (h1 : A + B = 100) (h2 : (3 / 10) * A = (1 / 5) * B) : B = 60 := 
by 
  sorry

end NUMINAMATH_GPT_find_b_amount_l1519_151937


namespace NUMINAMATH_GPT_CaitlinAge_l1519_151920

theorem CaitlinAge (age_AuntAnna : ℕ) (age_Brianna : ℕ) (age_Caitlin : ℕ)
  (h1 : age_AuntAnna = 42)
  (h2 : age_Brianna = age_AuntAnna / 2)
  (h3 : age_Caitlin = age_Brianna - 5) :
  age_Caitlin = 16 :=
by 
  sorry

end NUMINAMATH_GPT_CaitlinAge_l1519_151920


namespace NUMINAMATH_GPT_sum_of_decimals_l1519_151943

theorem sum_of_decimals : (5.47 + 4.96) = 10.43 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_decimals_l1519_151943


namespace NUMINAMATH_GPT_find_line_l_l1519_151913

theorem find_line_l :
  ∃ l : ℝ × ℝ → Prop,
    (∀ (B : ℝ × ℝ), (2 * B.1 + B.2 - 8 = 0) → 
      (∀ A : ℝ × ℝ, (A.1 = -B.1 ∧ A.2 = 2 * B.1 - 6 ) → 
        (A.1 - 3 * A.2 + 10 = 0) → 
          B.1 = 4 ∧ B.2 = 0 ∧ ∀ p : ℝ × ℝ, B.1 * p.1 + 4 * p.2 - 4 = 0)) := 
  sorry

end NUMINAMATH_GPT_find_line_l_l1519_151913


namespace NUMINAMATH_GPT_range_of_a_l1519_151974

theorem range_of_a (a : ℝ) (hx : ∀ x : ℝ, (a - 1) * x > a - 1 → x < 1) : a < 1 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1519_151974


namespace NUMINAMATH_GPT_tangent_product_l1519_151903

-- Declarations for circles, points of tangency, and radii
variables (R r : ℝ) -- radii of the circles
variables (A B C : ℝ) -- distances related to the tangents

-- Conditions: Two circles, a common internal tangent intersecting at points A and B, tangent at point C
axiom tangent_conditions : A * B = R * r

-- Problem statement: Prove that A * C * C * B = R * r
theorem tangent_product (R r A B C : ℝ) (h : A * B = R * r) : A * C * C * B = R * r :=
by
  sorry

end NUMINAMATH_GPT_tangent_product_l1519_151903


namespace NUMINAMATH_GPT_susan_betsy_ratio_l1519_151939

theorem susan_betsy_ratio (betsy_wins : ℕ) (helen_wins : ℕ) (susan_wins : ℕ) (total_wins : ℕ)
  (h1 : betsy_wins = 5)
  (h2 : helen_wins = 2 * betsy_wins)
  (h3 : betsy_wins + helen_wins + susan_wins = total_wins)
  (h4 : total_wins = 30) :
  susan_wins / betsy_wins = 3 := by
  sorry

end NUMINAMATH_GPT_susan_betsy_ratio_l1519_151939


namespace NUMINAMATH_GPT_college_girls_count_l1519_151912

/-- Given conditions:
 1. The ratio of the numbers of boys to girls is 8:5.
 2. The total number of students in the college is 416.
 
 Prove: The number of girls in the college is 160.
 -/
theorem college_girls_count (B G : ℕ) (h1 : B = (8 * G) / 5) (h2 : B + G = 416) : G = 160 :=
by
  sorry

end NUMINAMATH_GPT_college_girls_count_l1519_151912


namespace NUMINAMATH_GPT_school_bus_solution_l1519_151919

-- Define the capacities
def bus_capacity : Prop := 
  ∃ x y : ℕ, x + y = 75 ∧ 3 * x + 2 * y = 180 ∧ x = 30 ∧ y = 45

-- Define the rental problem
def rental_plans : Prop :=
  ∃ a : ℕ, 6 ≤ a ∧ a ≤ 8 ∧ 
  (30 * a + 45 * (25 - a) ≥ 1000) ∧ 
  (320 * a + 400 * (25 - a) ≤ 9550) ∧ 
  3 = 3

-- The main theorem combines the two aspects
theorem school_bus_solution: bus_capacity ∧ rental_plans := 
  sorry -- Proof omitted

end NUMINAMATH_GPT_school_bus_solution_l1519_151919


namespace NUMINAMATH_GPT_alice_savings_l1519_151978

variable (B : ℝ)

def savings (B : ℝ) : ℝ :=
  let first_month := 10
  let second_month := first_month + 30 + B
  let third_month := first_month + 30 + 30
  first_month + second_month + third_month

theorem alice_savings (B : ℝ) : savings B = 120 + B :=
by
  sorry

end NUMINAMATH_GPT_alice_savings_l1519_151978


namespace NUMINAMATH_GPT_sum_of_cubes_pattern_l1519_151949

theorem sum_of_cubes_pattern :
  (1^3 + 2^3 = 3^2) ->
  (1^3 + 2^3 + 3^3 = 6^2) ->
  (1^3 + 2^3 + 3^3 + 4^3 = 10^2) ->
  (1^3 + 2^3 + 3^3 + 4^3 + 5^3 + 6^3 = 21^2) :=
by
  intros h1 h2 h3
  -- Proof follows here
  sorry

end NUMINAMATH_GPT_sum_of_cubes_pattern_l1519_151949


namespace NUMINAMATH_GPT_greatest_four_digit_n_l1519_151967

theorem greatest_four_digit_n :
  ∃ (n : ℕ), (1000 ≤ n ∧ n ≤ 9999) ∧ (∃ m : ℕ, n + 1 = m^2) ∧ ¬(n! % (n * (n + 1) / 2) = 0) ∧ n = 9999 :=
by sorry

end NUMINAMATH_GPT_greatest_four_digit_n_l1519_151967


namespace NUMINAMATH_GPT_sum_of_base_radii_l1519_151972

theorem sum_of_base_radii (R : ℝ) (hR : R = 5) (a b c : ℝ) 
  (h_ratios : a = 1 ∧ b = 2 ∧ c = 3) 
  (r1 r2 r3 : ℝ) 
  (h_r1 : r1 = (a / (a + b + c)) * R)
  (h_r2 : r2 = (b / (a + b + c)) * R)
  (h_r3 : r3 = (c / (a + b + c)) * R) : 
  r1 + r2 + r3 = 5 := 
by
  subst hR
  simp [*, ←add_assoc, add_comm]
  sorry

end NUMINAMATH_GPT_sum_of_base_radii_l1519_151972


namespace NUMINAMATH_GPT_total_cost_of_fencing_l1519_151976

def diameter : ℝ := 28
def cost_per_meter : ℝ := 1.50
def pi_approx : ℝ := 3.14159

noncomputable def circumference : ℝ := pi_approx * diameter
noncomputable def total_cost : ℝ := circumference * cost_per_meter

theorem total_cost_of_fencing : total_cost = 131.94 :=
by
  sorry

end NUMINAMATH_GPT_total_cost_of_fencing_l1519_151976


namespace NUMINAMATH_GPT_quadrilateral_area_l1519_151950

theorem quadrilateral_area :
  let a1 := 9  -- adjacent side length
  let a2 := 6  -- other adjacent side length
  let d := 20  -- diagonal
  let θ1 := 35  -- first angle in degrees
  let θ2 := 110  -- second angle in degrees
  let sin35 := Real.sin (θ1 * Real.pi / 180)
  let sin110 := Real.sin (θ2 * Real.pi / 180)
  let area_triangle1 := (1/2 : ℝ) * a1 * d * sin35
  let area_triangle2 := (1/2 : ℝ) * a2 * d * sin110
  area_triangle1 + area_triangle2 = 108.006 := 
by
  let a1 := 9
  let a2 := 6
  let d := 20
  let θ1 := 35
  let θ2 := 110
  let sin35 := Real.sin (θ1 * Real.pi / 180)
  let sin110 := Real.sin (θ2 * Real.pi / 180)
  let area_triangle1 := (1/2 : ℝ) * a1 * d * sin35
  let area_triangle2 := (1/2 : ℝ) * a2 * d * sin110
  show area_triangle1 + area_triangle2 = 108.006
  sorry

end NUMINAMATH_GPT_quadrilateral_area_l1519_151950


namespace NUMINAMATH_GPT_det_matrix_example_l1519_151964

def det_2x2 (a b c d : ℤ) : ℤ := a * d - b * c

theorem det_matrix_example : det_2x2 4 5 2 3 = 2 :=
by
  sorry

end NUMINAMATH_GPT_det_matrix_example_l1519_151964


namespace NUMINAMATH_GPT_tangent_line_to_C1_and_C2_is_correct_l1519_151930

def C1 (x : ℝ) : ℝ := x ^ 2
def C2 (x : ℝ) : ℝ := -(x - 2) ^ 2
def l (x : ℝ) : ℝ := -2 * x + 3

theorem tangent_line_to_C1_and_C2_is_correct :
  (∃ x1 : ℝ, C1 x1 = l x1 ∧ deriv C1 x1 = deriv l x1) ∧
  (∃ x2 : ℝ, C2 x2 = l x2 ∧ deriv C2 x2 = deriv l x2) :=
sorry

end NUMINAMATH_GPT_tangent_line_to_C1_and_C2_is_correct_l1519_151930


namespace NUMINAMATH_GPT_celebration_women_count_l1519_151928

theorem celebration_women_count (num_men : ℕ) (num_pairs : ℕ) (pairs_per_man : ℕ) (pairs_per_woman : ℕ) 
  (hm : num_men = 15) (hpm : pairs_per_man = 4) (hwp : pairs_per_woman = 3) (total_pairs : num_pairs = num_men * pairs_per_man) : 
  num_pairs / pairs_per_woman = 20 :=
by
  sorry

end NUMINAMATH_GPT_celebration_women_count_l1519_151928


namespace NUMINAMATH_GPT_intersection_M_N_l1519_151985

-- Define sets M and N
def M := {x : ℝ | x^2 - 2*x ≤ 0}
def N := {x : ℝ | -2 < x ∧ x < 1}

-- The theorem stating the intersection of M and N equals [0, 1)
theorem intersection_M_N : M ∩ N = {x : ℝ | 0 ≤ x ∧ x < 1} := 
by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l1519_151985


namespace NUMINAMATH_GPT_isosceles_triangle_base_length_l1519_151997

open Real

noncomputable def average_distance_sun_earth : ℝ := 1.5 * 10^8 -- in kilometers
noncomputable def base_length_given_angle_one_second (legs_length : ℝ) : ℝ := 4.848 -- in millimeters when legs are 1 kilometer

theorem isosceles_triangle_base_length 
  (vertex_angle : ℝ) (legs_length : ℝ) 
  (h1 : vertex_angle = 1 / 3600) 
  (h2 : legs_length = average_distance_sun_earth) : 
  ∃ base_length: ℝ, base_length = 727.2 := 
by 
  sorry

end NUMINAMATH_GPT_isosceles_triangle_base_length_l1519_151997


namespace NUMINAMATH_GPT_fraction_of_students_l1519_151960

theorem fraction_of_students {G B T : ℕ} (h1 : B = 2 * G) (h2 : T = G + B) (h3 : (1 / 2) * (G : ℝ) = (x : ℝ) * (T : ℝ)) : x = (1 / 6) :=
by sorry

end NUMINAMATH_GPT_fraction_of_students_l1519_151960


namespace NUMINAMATH_GPT_willie_currency_exchange_l1519_151902

theorem willie_currency_exchange :
  let euro_amount := 70
  let pound_amount := 50
  let franc_amount := 30

  let euro_to_dollar := 1.2
  let pound_to_dollar := 1.5
  let franc_to_dollar := 1.1

  let airport_euro_rate := 5 / 7
  let airport_pound_rate := 3 / 4
  let airport_franc_rate := 9 / 10

  let flat_fee := 5

  let official_euro_dollars := euro_amount * euro_to_dollar
  let official_pound_dollars := pound_amount * pound_to_dollar
  let official_franc_dollars := franc_amount * franc_to_dollar

  let airport_euro_dollars := official_euro_dollars * airport_euro_rate
  let airport_pound_dollars := official_pound_dollars * airport_pound_rate
  let airport_franc_dollars := official_franc_dollars * airport_franc_rate

  let final_euro_dollars := airport_euro_dollars - flat_fee
  let final_pound_dollars := airport_pound_dollars - flat_fee
  let final_franc_dollars := airport_franc_dollars - flat_fee

  let total_dollars := final_euro_dollars + final_pound_dollars + final_franc_dollars

  total_dollars = 130.95 :=
by
  sorry

end NUMINAMATH_GPT_willie_currency_exchange_l1519_151902


namespace NUMINAMATH_GPT_staff_discount_l1519_151931

open Real

theorem staff_discount (d : ℝ) (h : d > 0) (final_price_eq : 0.14 * d = 0.35 * d * (1 - 0.6)) : 0.6 * 100 = 60 :=
by
  sorry

end NUMINAMATH_GPT_staff_discount_l1519_151931


namespace NUMINAMATH_GPT_fixed_point_on_line_AC_l1519_151926

-- Given definitions and conditions directly from a)
def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def line_through_P (x y : ℝ) : Prop := ∃ t : ℝ, x = t * y - 1
def reflection_across_x_axis (y : ℝ) : ℝ := -y

-- The final proof statement translating c)
theorem fixed_point_on_line_AC
  (A B C P : ℝ × ℝ)
  (hP : P = (-1, 0))
  (hA : parabola A.1 A.2)
  (hB : parabola B.1 B.2)
  (hAB : ∃ t : ℝ, line_through_P A.1 A.2 ∧ line_through_P B.1 B.2)
  (hRef : C = (B.1, reflection_across_x_axis B.2)) :
  ∃ x y : ℝ, (x, y) = (1, 0) ∧ line_through_P x y := 
sorry

end NUMINAMATH_GPT_fixed_point_on_line_AC_l1519_151926


namespace NUMINAMATH_GPT_candidate_percentage_l1519_151999

theorem candidate_percentage (P : ℕ) (total_votes : ℕ) (vote_diff : ℕ)
  (h1 : total_votes = 7000)
  (h2 : vote_diff = 2100)
  (h3 : (P * total_votes / 100) + (P * total_votes / 100) + vote_diff = total_votes) :
  P = 35 :=
by
  sorry

end NUMINAMATH_GPT_candidate_percentage_l1519_151999


namespace NUMINAMATH_GPT_neighbors_receive_equal_mangoes_l1519_151932

-- Definitions from conditions
def total_mangoes : ℕ := 560
def mangoes_sold : ℕ := total_mangoes / 2
def remaining_mangoes : ℕ := total_mangoes - mangoes_sold
def neighbors : ℕ := 8

-- The lean statement
theorem neighbors_receive_equal_mangoes :
  remaining_mangoes / neighbors = 35 :=
by
  -- This is where the proof would go, but we'll leave it with sorry for now.
  sorry

end NUMINAMATH_GPT_neighbors_receive_equal_mangoes_l1519_151932


namespace NUMINAMATH_GPT_min_value_problem_l1519_151987

noncomputable def minValueOfExpression (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x * y * z = 1) : ℝ :=
  (x + 2 * y) * (y + 2 * z) * (x * z + 1)

theorem min_value_problem (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x * y * z = 1) :
  minValueOfExpression x y z hx hy hz hxyz = 16 :=
  sorry

end NUMINAMATH_GPT_min_value_problem_l1519_151987


namespace NUMINAMATH_GPT_abs_sqrt3_minus_1_sub_2_cos30_eq_neg1_l1519_151951

theorem abs_sqrt3_minus_1_sub_2_cos30_eq_neg1 :
  |(Real.sqrt 3) - 1| - 2 * Real.cos (Real.pi / 6) = -1 := by
  sorry

end NUMINAMATH_GPT_abs_sqrt3_minus_1_sub_2_cos30_eq_neg1_l1519_151951


namespace NUMINAMATH_GPT_total_arrangements_l1519_151971

def count_arrangements : Nat :=
  let male_positions := 3
  let female_positions := 3
  let male_arrangements := Nat.factorial male_positions
  let female_arrangements := Nat.factorial (female_positions - 1)
  male_arrangements * female_arrangements / (male_positions - female_positions + 1)

theorem total_arrangements : count_arrangements = 36 := by
  sorry

end NUMINAMATH_GPT_total_arrangements_l1519_151971


namespace NUMINAMATH_GPT_perimeter_of_inner_polygon_le_outer_polygon_l1519_151940

-- Definitions of polygons (for simplicity considered as list of points or sides)
structure Polygon where
  sides : List ℝ  -- assuming sides lengths are given as list of real numbers
  convex : Prop   -- a property stating that the polygon is convex

-- Definition of the perimeter of a polygon
def perimeter (p : Polygon) : ℝ := p.sides.sum

-- Conditions from the problem
variable {P_in P_out : Polygon}
variable (h_convex_in : P_in.convex) (h_convex_out : P_out.convex)
variable (h_inside : ∀ s ∈ P_in.sides, s ∈ P_out.sides) -- simplifying the "inside" condition

-- The theorem statement
theorem perimeter_of_inner_polygon_le_outer_polygon :
  perimeter P_in ≤ perimeter P_out :=
by {
  sorry
}

end NUMINAMATH_GPT_perimeter_of_inner_polygon_le_outer_polygon_l1519_151940


namespace NUMINAMATH_GPT_volume_of_cube_l1519_151901

theorem volume_of_cube (a : ℕ) (h : a^3 - (a^3 - 4 * a) = 12) : a^3 = 27 :=
by 
  sorry

end NUMINAMATH_GPT_volume_of_cube_l1519_151901


namespace NUMINAMATH_GPT_fraction_result_l1519_151992

theorem fraction_result (a b c : ℝ) (h1 : a / 2 = b / 3) (h2 : b / 3 = c / 5) (h3 : a ≠ 0) (h4 : b ≠ 0) (h5 : c ≠ 0) :
  (a + b) / (c - a) = 5 / 3 :=
by
  sorry

end NUMINAMATH_GPT_fraction_result_l1519_151992


namespace NUMINAMATH_GPT_hamburger_price_l1519_151970

theorem hamburger_price (P : ℝ) 
    (h1 : 2 * 4 + 2 * 2 = 12) 
    (h2 : 12 * P + 4 * P = 50) : 
    P = 3.125 := 
by
  -- sorry added to skip the proof.
  sorry

end NUMINAMATH_GPT_hamburger_price_l1519_151970


namespace NUMINAMATH_GPT_only_prime_such_that_2p_plus_one_is_perfect_power_l1519_151998

theorem only_prime_such_that_2p_plus_one_is_perfect_power :
  ∃ (p : ℕ), p ≤ 1000 ∧ Prime p ∧ ∃ (m n : ℕ), n ≥ 2 ∧ 2 * p + 1 = m^n ∧ p = 13 :=
by
  sorry

end NUMINAMATH_GPT_only_prime_such_that_2p_plus_one_is_perfect_power_l1519_151998


namespace NUMINAMATH_GPT_product_mod_7_l1519_151922

theorem product_mod_7 (a b c: ℕ) (ha: a % 7 = 2) (hb: b % 7 = 3) (hc: c % 7 = 4) :
  (a * b * c) % 7 = 3 :=
by
  sorry

end NUMINAMATH_GPT_product_mod_7_l1519_151922


namespace NUMINAMATH_GPT_Toms_out_of_pocket_cost_l1519_151923

theorem Toms_out_of_pocket_cost (visit_cost cast_cost insurance_percent : ℝ) 
  (h1 : visit_cost = 300) 
  (h2 : cast_cost = 200) 
  (h3 : insurance_percent = 0.6) : 
  (visit_cost + cast_cost) - ((visit_cost + cast_cost) * insurance_percent) = 200 :=
by
  sorry

end NUMINAMATH_GPT_Toms_out_of_pocket_cost_l1519_151923


namespace NUMINAMATH_GPT_problem_condition_l1519_151981

variable {f : ℝ → ℝ}
variable {a b : ℝ}

noncomputable def fx_condition (f : ℝ → ℝ) :=
  ∀ x : ℝ, f x + x * (deriv f x) < 0

theorem problem_condition {f : ℝ → ℝ} {a b : ℝ} (h1 : fx_condition f) (h2 : a < b) :
  a * f a > b * f b :=
sorry

end NUMINAMATH_GPT_problem_condition_l1519_151981


namespace NUMINAMATH_GPT_find_y_l1519_151907

theorem find_y (y : ℝ) : 
  2 ≤ y / (3 * y - 4) ∧ y / (3 * y - 4) < 5 ↔ y ∈ Set.Ioc (10 / 7) (8 / 5) := 
sorry

end NUMINAMATH_GPT_find_y_l1519_151907


namespace NUMINAMATH_GPT_determine_m_values_l1519_151948

theorem determine_m_values (m : ℚ) :
  ((∃ x y : ℚ, x = -3 ∧ y = 0 ∧ (m^2 - 2 * m - 3) * x + (2 * m^2 + m - 1) * y = 2 * m - 6) ∨
  (∃ k : ℚ, k = -1 ∧ (m^2 - 2 * m - 3) + (2 * m^2 + m - 1) * k = 0)) →
  (m = -5/3 ∨ m = 4/3) :=
by
  sorry

end NUMINAMATH_GPT_determine_m_values_l1519_151948


namespace NUMINAMATH_GPT_percentage_loss_l1519_151956

theorem percentage_loss (CP SP : ℝ) (hCP : CP = 1400) (hSP : SP = 1148) : 
  (CP - SP) / CP * 100 = 18 := by 
  sorry

end NUMINAMATH_GPT_percentage_loss_l1519_151956


namespace NUMINAMATH_GPT_mean_problem_l1519_151938

theorem mean_problem : 
  (8 + 12 + 24) / 3 = (16 + z) / 2 → z = 40 / 3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_mean_problem_l1519_151938


namespace NUMINAMATH_GPT_binary_addition_to_decimal_l1519_151968

theorem binary_addition_to_decimal : (2^8 + 2^7 + 2^6 + 2^5 + 2^4 + 2^3 + 2^2 + 2^1 + 2^0)
                                     + (2^5 + 2^4 + 2^3 + 2^2) = 571 := by
  sorry

end NUMINAMATH_GPT_binary_addition_to_decimal_l1519_151968


namespace NUMINAMATH_GPT_abc_product_l1519_151946

theorem abc_product :
  ∃ (a b c P : ℕ), 
    b + c = 3 ∧ 
    c + a = 6 ∧ 
    a + b = 7 ∧ 
    P = a * b * c ∧ 
    P = 10 :=
by sorry

end NUMINAMATH_GPT_abc_product_l1519_151946


namespace NUMINAMATH_GPT_coin_tails_probability_l1519_151962

theorem coin_tails_probability (p : ℝ) (h : p = 0.5) (n : ℕ) (h_n : n = 3) :
  ∃ k : ℕ, k ≤ n ∧ (Nat.choose n k : ℝ) * p^k * (1 - p)^(n - k) = 0.375 :=
by
  sorry

end NUMINAMATH_GPT_coin_tails_probability_l1519_151962


namespace NUMINAMATH_GPT_inequality_proof_l1519_151991

theorem inequality_proof (a b c : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0) (h₃ : a * b * c = 1) :
  1 / (a^3 * (b + c)) + 1 / (b^3 * (c + a)) + 1 / (c^3 * (a + b)) ≥ 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1519_151991


namespace NUMINAMATH_GPT_rotated_line_eq_l1519_151984

theorem rotated_line_eq :
  ∀ (x y : ℝ), 
  (x - y + 4 = 0) ∨ (x - y - 4 = 0) ↔ 
  ∃ (x' y' : ℝ), (-x', -y') = (x, y) ∧ (x' - y' + 4 = 0) :=
by
  sorry

end NUMINAMATH_GPT_rotated_line_eq_l1519_151984


namespace NUMINAMATH_GPT_min_cost_for_boxes_l1519_151935

def box_volume (l w h : ℕ) : ℕ := l * w * h
def total_boxes_needed (total_volume box_volume : ℕ) : ℕ := (total_volume + box_volume - 1) / box_volume
def total_cost (num_boxes : ℕ) (cost_per_box : ℚ) : ℚ := num_boxes * cost_per_box

theorem min_cost_for_boxes : 
  let l := 20
  let w := 20
  let h := 15
  let cost_per_box := (7 : ℚ) / 10
  let total_volume := 3060000
  let volume_box := box_volume l w h
  let num_boxes_needed := total_boxes_needed total_volume volume_box
  (num_boxes_needed = 510) → 
  (total_cost num_boxes_needed cost_per_box = 357) :=
by
  intros
  sorry

end NUMINAMATH_GPT_min_cost_for_boxes_l1519_151935


namespace NUMINAMATH_GPT_system_linear_eq_sum_l1519_151975

theorem system_linear_eq_sum (x y : ℝ) (h₁ : 3 * x + 2 * y = 2) (h₂ : 2 * x + 3 * y = 8) : x + y = 2 :=
sorry

end NUMINAMATH_GPT_system_linear_eq_sum_l1519_151975


namespace NUMINAMATH_GPT_eraser_difference_l1519_151963

theorem eraser_difference
  (hanna_erasers rachel_erasers tanya_erasers tanya_red_erasers : ℕ)
  (h1 : hanna_erasers = 2 * rachel_erasers)
  (h2 : rachel_erasers = tanya_red_erasers)
  (h3 : tanya_erasers = 20)
  (h4 : tanya_red_erasers = tanya_erasers / 2)
  (h5 : hanna_erasers = 4) :
  rachel_erasers - (tanya_red_erasers / 2) = 5 :=
sorry

end NUMINAMATH_GPT_eraser_difference_l1519_151963


namespace NUMINAMATH_GPT_erin_walks_less_l1519_151952

variable (total_distance : ℕ)
variable (susan_distance : ℕ)

theorem erin_walks_less (h1 : total_distance = 15) (h2 : susan_distance = 9) :
  susan_distance - (total_distance - susan_distance) = 3 := by
  sorry

end NUMINAMATH_GPT_erin_walks_less_l1519_151952


namespace NUMINAMATH_GPT_necessary_and_sufficient_condition_l1519_151918

theorem necessary_and_sufficient_condition (a : ℝ) : (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≤ 0) ↔ (a ≥ 4) :=
by
  sorry

end NUMINAMATH_GPT_necessary_and_sufficient_condition_l1519_151918


namespace NUMINAMATH_GPT_perpendicular_lines_condition_l1519_151941

theorem perpendicular_lines_condition (a : ℝ) :
  (6 * a + 3 * 4 = 0) ↔ (a = -2) :=
sorry

end NUMINAMATH_GPT_perpendicular_lines_condition_l1519_151941


namespace NUMINAMATH_GPT_remainder_of_first_six_primes_sum_divided_by_seventh_prime_l1519_151969

theorem remainder_of_first_six_primes_sum_divided_by_seventh_prime : 
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
  sum_primes % p7 = 7 := by sorry

end NUMINAMATH_GPT_remainder_of_first_six_primes_sum_divided_by_seventh_prime_l1519_151969


namespace NUMINAMATH_GPT_Karen_baked_50_cookies_l1519_151947

def Karen_kept_cookies : ℕ := 10
def Karen_grandparents_cookies : ℕ := 8
def people_in_class : ℕ := 16
def cookies_per_person : ℕ := 2

theorem Karen_baked_50_cookies :
  Karen_kept_cookies + Karen_grandparents_cookies + (people_in_class * cookies_per_person) = 50 :=
by 
  sorry

end NUMINAMATH_GPT_Karen_baked_50_cookies_l1519_151947


namespace NUMINAMATH_GPT_exists_x_in_interval_iff_m_lt_3_l1519_151905

theorem exists_x_in_interval_iff_m_lt_3 (m : ℝ) :
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ 3 ∧ x^2 - 2 * x > m) ↔ m < 3 :=
by
  sorry

end NUMINAMATH_GPT_exists_x_in_interval_iff_m_lt_3_l1519_151905


namespace NUMINAMATH_GPT_find_y_l1519_151929

noncomputable def imaginary_unit : ℂ := Complex.I

noncomputable def z1 (y : ℝ) : ℂ := 3 + y * imaginary_unit

noncomputable def z2 : ℂ := 2 - imaginary_unit

theorem find_y (y : ℝ) (h : z1 y / z2 = 1 + imaginary_unit) : y = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_y_l1519_151929


namespace NUMINAMATH_GPT_parabola_directrix_l1519_151917

theorem parabola_directrix (x y : ℝ) (h : y = 4 * (x - 1)^2 + 3) : y = 11 / 4 :=
sorry

end NUMINAMATH_GPT_parabola_directrix_l1519_151917


namespace NUMINAMATH_GPT_vector_dot_product_result_l1519_151944

variable {α : Type*} [Field α]

structure Vector2 (α : Type*) :=
(x : α)
(y : α)

def vector_add (a b : Vector2 α) : Vector2 α :=
  ⟨a.x + b.x, a.y + b.y⟩

def vector_sub (a b : Vector2 α) : Vector2 α :=
  ⟨a.x - b.x, a.y - b.y⟩

def dot_product (a b : Vector2 α) : α :=
  a.x * b.x + a.y * b.y

variable (a b : Vector2 ℝ)

theorem vector_dot_product_result
  (h1 : vector_add a b = ⟨1, -3⟩)
  (h2 : vector_sub a b = ⟨3, 7⟩) :
  dot_product a b = -12 :=
by
  sorry

end NUMINAMATH_GPT_vector_dot_product_result_l1519_151944


namespace NUMINAMATH_GPT_problem_statement_l1519_151953

-- Defining the condition x^3 = 8
def condition1 (x : ℝ) : Prop := x^3 = 8

-- Defining the function f(x) = (x-1)(x+1)(x^2 + x + 1)
def f (x : ℝ) : ℝ := (x - 1) * (x + 1) * (x^2 + x + 1)

-- The theorem we want to prove: For any x satisfying the condition, the function value is 21
theorem problem_statement (x : ℝ) (h : condition1 x) : f x = 21 := 
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1519_151953


namespace NUMINAMATH_GPT_prob_B_independent_l1519_151900

-- Definitions based on the problem's conditions
def prob_A := 0.7
def prob_A_union_B := 0.94

-- With these definitions established, we need to state the theorem.
-- The theorem should express that the probability of B solving the problem independently (prob_B) is 0.8.
theorem prob_B_independent : 
    (∃ (prob_B: ℝ), prob_A = 0.7 ∧ prob_A_union_B = 0.94 ∧ prob_B = 0.8) :=
by
    sorry

end NUMINAMATH_GPT_prob_B_independent_l1519_151900


namespace NUMINAMATH_GPT_distance_from_point_A_l1519_151910

theorem distance_from_point_A :
  ∀ (A : ℝ) (area : ℝ) (white_area : ℝ) (black_area : ℝ), area = 18 →
  (black_area = 2 * white_area) →
  A = (12 * Real.sqrt 2) / 5 := by
  intros A area white_area black_area h1 h2
  sorry

end NUMINAMATH_GPT_distance_from_point_A_l1519_151910


namespace NUMINAMATH_GPT_car_miles_traveled_actual_miles_l1519_151986

noncomputable def count_skipped_numbers (n : ℕ) : ℕ :=
  let count_digit7 (x : ℕ) : Bool := x = 7
  -- Function to count the number of occurrences of digit 7 in each place value
  let rec count (x num_skipped : ℕ) : ℕ :=
    if x = 0 then num_skipped else
    let digit := x % 10
    let new_count := if count_digit7 digit then num_skipped + 1 else num_skipped
    count (x / 10) new_count
  count n 0

theorem car_miles_traveled (odometer_reading : ℕ) : ℕ :=
  let num_skipped := count_skipped_numbers 3008
  odometer_reading - num_skipped

theorem actual_miles {odometer_reading : ℕ} (h : odometer_reading = 3008) : car_miles_traveled odometer_reading = 2194 :=
by sorry

end NUMINAMATH_GPT_car_miles_traveled_actual_miles_l1519_151986


namespace NUMINAMATH_GPT_toby_friends_girls_count_l1519_151965

noncomputable def percentage_of_boys : ℚ := 55 / 100
noncomputable def boys_count : ℕ := 33
noncomputable def total_friends : ℚ := boys_count / percentage_of_boys
noncomputable def percentage_of_girls : ℚ := 1 - percentage_of_boys
noncomputable def girls_count : ℚ := percentage_of_girls * total_friends

theorem toby_friends_girls_count : girls_count = 27 := by
  sorry

end NUMINAMATH_GPT_toby_friends_girls_count_l1519_151965


namespace NUMINAMATH_GPT_cubic_box_dimension_l1519_151921

theorem cubic_box_dimension (a : ℤ) (h: 12 * a = 3 * (a^3)) : a = 2 :=
by
  sorry

end NUMINAMATH_GPT_cubic_box_dimension_l1519_151921


namespace NUMINAMATH_GPT_inequality_proof_l1519_151995

theorem inequality_proof (x y : ℝ) (h : x^8 + y^8 ≤ 2) : x^2 * y^2 + |x^2 - y^2| ≤ π / 2 :=
sorry

end NUMINAMATH_GPT_inequality_proof_l1519_151995


namespace NUMINAMATH_GPT_rectangle_area_l1519_151942

theorem rectangle_area (l w : ℝ) (h1 : l = 4 * w) (h2 : 2 * l + 2 * w = 200) : l * w = 1600 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_l1519_151942


namespace NUMINAMATH_GPT_arithmetic_sequence_a5_value_l1519_151990

theorem arithmetic_sequence_a5_value 
  (a : ℕ → ℝ) 
  (h1 : a 2 + a 4 = 16) 
  (h2 : a 1 = 1) : 
  a 5 = 15 := 
by 
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a5_value_l1519_151990


namespace NUMINAMATH_GPT_remainder_of_130_div_k_l1519_151906

theorem remainder_of_130_div_k (k a : ℕ) (hk : 90 = a * k^2 + 18) : 130 % k = 4 :=
sorry

end NUMINAMATH_GPT_remainder_of_130_div_k_l1519_151906


namespace NUMINAMATH_GPT_second_largest_geometric_sum_l1519_151959

theorem second_largest_geometric_sum {a r : ℕ} (h_sum: a + a * r + a * r^2 + a * r^3 = 1417) (h_geometric: 1 + r + r^2 + r^3 ∣ 1417) : (a * r^2 = 272) :=
sorry

end NUMINAMATH_GPT_second_largest_geometric_sum_l1519_151959


namespace NUMINAMATH_GPT_c_value_difference_l1519_151955

theorem c_value_difference (a b c : ℝ) 
  (h1 : a + b + c = 5) 
  (h2 : a^2 + b^2 + c^2 = 18) : 
  max c - min c = 34 / 3 :=
sorry

end NUMINAMATH_GPT_c_value_difference_l1519_151955


namespace NUMINAMATH_GPT_remainder_97_pow_50_mod_100_l1519_151924

theorem remainder_97_pow_50_mod_100 :
  (97 ^ 50) % 100 = 49 := 
by
  sorry

end NUMINAMATH_GPT_remainder_97_pow_50_mod_100_l1519_151924


namespace NUMINAMATH_GPT_car_length_l1519_151993

variables (L E C : ℕ)

theorem car_length (h1 : 150 * E = L + 150 * C) (h2 : 30 * E = L - 30 * C) : L = 113 * E :=
by
  sorry

end NUMINAMATH_GPT_car_length_l1519_151993


namespace NUMINAMATH_GPT_fraction_unspent_is_correct_l1519_151911

noncomputable def fraction_unspent (S : ℝ) : ℝ :=
  let after_tax := S - 0.15 * S
  let after_first_week := after_tax - 0.25 * after_tax
  let after_second_week := after_first_week - 0.3 * after_first_week
  let after_third_week := after_second_week - 0.2 * S
  let after_fourth_week := after_third_week - 0.1 * after_third_week
  after_fourth_week / S

theorem fraction_unspent_is_correct (S : ℝ) (hS : S > 0) : 
  fraction_unspent S = 0.221625 :=
by
  sorry

end NUMINAMATH_GPT_fraction_unspent_is_correct_l1519_151911


namespace NUMINAMATH_GPT_number_of_cows_l1519_151980

def each_cow_milk_per_day : ℕ := 1000
def total_milk_per_week : ℕ := 364000
def days_in_week : ℕ := 7

theorem number_of_cows : 
  (total_milk_per_week = 364000) →
  (each_cow_milk_per_day = 1000) →
  (days_in_week = 7) →
  (total_milk_per_week / (each_cow_milk_per_day * days_in_week)) = 52 :=
by
  sorry

end NUMINAMATH_GPT_number_of_cows_l1519_151980


namespace NUMINAMATH_GPT_root_in_interval_l1519_151979

noncomputable def f (x: ℝ) : ℝ := x^2 + (Real.log x) - 4

theorem root_in_interval : 
  (∃ ξ ∈ Set.Ioo 1 2, f ξ = 0) :=
by
  sorry

end NUMINAMATH_GPT_root_in_interval_l1519_151979


namespace NUMINAMATH_GPT_distance_covered_at_40_kmph_l1519_151916

theorem distance_covered_at_40_kmph
   (total_distance : ℝ)
   (speed1 : ℝ)
   (speed2 : ℝ)
   (total_time : ℝ)
   (part_distance1 : ℝ) :
   total_distance = 250 ∧
   speed1 = 40 ∧
   speed2 = 60 ∧
   total_time = 6 ∧
   (part_distance1 / speed1 + (total_distance - part_distance1) / speed2 = total_time) →
   part_distance1 = 220 :=
by sorry

end NUMINAMATH_GPT_distance_covered_at_40_kmph_l1519_151916


namespace NUMINAMATH_GPT_solve_for_sum_l1519_151961

theorem solve_for_sum (x y : ℝ) (h : x^2 + y^2 = 18 * x - 10 * y + 22) : x + y = 4 + 2 * Real.sqrt 42 :=
sorry

end NUMINAMATH_GPT_solve_for_sum_l1519_151961


namespace NUMINAMATH_GPT_negation_proposition_l1519_151914

theorem negation_proposition :
  (¬ ∀ x : ℝ, x^2 + 2 * x + 2 > 0) ↔ (∃ x : ℝ, x^2 + 2 * x + 2 ≤ 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_proposition_l1519_151914


namespace NUMINAMATH_GPT_triangle_height_l1519_151957

theorem triangle_height (area base height : ℝ) (h1 : area = 500) (h2 : base = 50) (h3 : area = (1 / 2) * base * height) : height = 20 :=
sorry

end NUMINAMATH_GPT_triangle_height_l1519_151957


namespace NUMINAMATH_GPT_percentage_cut_third_week_l1519_151977

noncomputable def initial_weight : ℝ := 300
noncomputable def first_week_percentage : ℝ := 0.30
noncomputable def second_week_percentage : ℝ := 0.30
noncomputable def final_weight : ℝ := 124.95

theorem percentage_cut_third_week :
  let remaining_after_first_week := initial_weight * (1 - first_week_percentage)
  let remaining_after_second_week := remaining_after_first_week * (1 - second_week_percentage)
  let cut_weight_third_week := remaining_after_second_week - final_weight
  let percentage_cut_third_week := (cut_weight_third_week / remaining_after_second_week) * 100
  percentage_cut_third_week = 15 :=
by
  sorry

end NUMINAMATH_GPT_percentage_cut_third_week_l1519_151977


namespace NUMINAMATH_GPT_percentage_students_passed_l1519_151966

theorem percentage_students_passed
    (total_students : ℕ)
    (students_failed : ℕ)
    (students_passed : ℕ)
    (percentage_passed : ℕ)
    (h1 : total_students = 840)
    (h2 : students_failed = 546)
    (h3 : students_passed = total_students - students_failed)
    (h4 : percentage_passed = (students_passed * 100) / total_students) :
    percentage_passed = 35 := by
  sorry

end NUMINAMATH_GPT_percentage_students_passed_l1519_151966


namespace NUMINAMATH_GPT_solve_for_x_l1519_151904

def f (x : ℝ) : ℝ := x^2 - 5 * x + 6

theorem solve_for_x :
  {x : ℝ | f (f x) = f x} = {0, 2, 3, 5} :=
by 
  sorry

end NUMINAMATH_GPT_solve_for_x_l1519_151904


namespace NUMINAMATH_GPT_dogs_not_eat_either_l1519_151954

-- Let's define the conditions
variables (total_dogs : ℕ) (dogs_like_carrots : ℕ) (dogs_like_chicken : ℕ) (dogs_like_both : ℕ)

-- Given conditions
def conditions : Prop :=
  total_dogs = 85 ∧
  dogs_like_carrots = 12 ∧
  dogs_like_chicken = 62 ∧
  dogs_like_both = 8

-- Problem to solve
theorem dogs_not_eat_either (h : conditions total_dogs dogs_like_carrots dogs_like_chicken dogs_like_both) :
  (total_dogs - (dogs_like_carrots - dogs_like_both + dogs_like_chicken - dogs_like_both + dogs_like_both)) = 19 :=
by {
  sorry 
}

end NUMINAMATH_GPT_dogs_not_eat_either_l1519_151954


namespace NUMINAMATH_GPT_proposition_D_l1519_151958

variable {A B C : Set α} (h1 : ∀ a (ha : a ∈ A), ∃ b ∈ B, a = b)
variable {A B C : Set α} (h2 : ∀ c (hc : c ∈ C), ∃ b ∈ B, b = c) 

theorem proposition_D (A B C : Set α) (h : A ∩ B = B ∪ C) : C ⊆ B :=
by 
  sorry

end NUMINAMATH_GPT_proposition_D_l1519_151958


namespace NUMINAMATH_GPT_fruit_seller_original_apples_l1519_151945

theorem fruit_seller_original_apples (x : ℝ) (h : 0.50 * x = 5000) : x = 10000 :=
sorry

end NUMINAMATH_GPT_fruit_seller_original_apples_l1519_151945
