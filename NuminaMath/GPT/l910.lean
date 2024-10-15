import Mathlib

namespace NUMINAMATH_GPT_ratio_of_additional_hours_james_danced_l910_91013

-- Definitions based on given conditions
def john_first_dance_time : ℕ := 3
def john_break_time : ℕ := 1
def john_second_dance_time : ℕ := 5
def combined_dancing_time_excluding_break : ℕ := 20

-- Calculations to be proved
def john_total_resting_dancing_time : ℕ :=
  john_first_dance_time + john_break_time + john_second_dance_time

def john_total_dancing_time : ℕ :=
  john_first_dance_time + john_second_dance_time

def james_dancing_time : ℕ :=
  combined_dancing_time_excluding_break - john_total_dancing_time

def additional_hours_james_danced : ℕ :=
  james_dancing_time - john_total_dancing_time

def desired_ratio : ℕ × ℕ :=
  (additional_hours_james_danced, john_total_resting_dancing_time)

-- Theorem to be proved according to the problem statement
theorem ratio_of_additional_hours_james_danced :
  desired_ratio = (4, 9) :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_ratio_of_additional_hours_james_danced_l910_91013


namespace NUMINAMATH_GPT_hamburger_varieties_l910_91032

-- Define the problem conditions as Lean definitions.
def condiments := 9  -- There are 9 condiments
def patty_choices := 3  -- Choices of 1, 2, or 3 patties

-- The goal is to prove that the number of different kinds of hamburgers is 1536.
theorem hamburger_varieties : (3 * 2^9) = 1536 := by
  sorry

end NUMINAMATH_GPT_hamburger_varieties_l910_91032


namespace NUMINAMATH_GPT_cos_identity_l910_91067

theorem cos_identity (α : ℝ) (h : Real.cos (π / 3 - α) = 3 / 5) : 
  Real.cos (2 * π / 3 + α) = -3 / 5 :=
by
  sorry

end NUMINAMATH_GPT_cos_identity_l910_91067


namespace NUMINAMATH_GPT_division_of_pow_of_16_by_8_eq_2_pow_4041_l910_91001

theorem division_of_pow_of_16_by_8_eq_2_pow_4041 :
  (16^1011) / 8 = 2^4041 :=
by
  -- Assume m = 16^1011
  let m := 16^1011
  -- Then expressing m in base 2
  have h_m_base2 : m = 2^4044 := by sorry
  -- Dividing m by 8
  have h_division : m / 8 = 2^4041 := by sorry
  -- Conclusion
  exact h_division

end NUMINAMATH_GPT_division_of_pow_of_16_by_8_eq_2_pow_4041_l910_91001


namespace NUMINAMATH_GPT_selling_price_l910_91080

theorem selling_price (cost_price profit_percentage selling_price : ℝ) (h1 : cost_price = 86.95652173913044)
  (h2 : profit_percentage = 0.15) : 
  selling_price = 100 :=
by
  sorry

end NUMINAMATH_GPT_selling_price_l910_91080


namespace NUMINAMATH_GPT_time_to_pass_telegraph_post_l910_91038

def conversion_factor_km_per_hour_to_m_per_sec := 1000 / 3600

noncomputable def train_length := 70
noncomputable def train_speed_kmph := 36

noncomputable def train_speed_m_per_sec := train_speed_kmph * conversion_factor_km_per_hour_to_m_per_sec

theorem time_to_pass_telegraph_post : (train_length / train_speed_m_per_sec) = 7 := by
  sorry

end NUMINAMATH_GPT_time_to_pass_telegraph_post_l910_91038


namespace NUMINAMATH_GPT_cost_of_soap_for_year_l910_91077

theorem cost_of_soap_for_year
  (months_per_bar cost_per_bar : ℕ)
  (months_in_year : ℕ)
  (h1 : months_per_bar = 2)
  (h2 : cost_per_bar = 8)
  (h3 : months_in_year = 12) :
  (months_in_year / months_per_bar) * cost_per_bar = 48 := by
  sorry

end NUMINAMATH_GPT_cost_of_soap_for_year_l910_91077


namespace NUMINAMATH_GPT_arithmetic_sequence__geometric_sequence__l910_91019

-- Part 1: Arithmetic Sequence
theorem arithmetic_sequence_
  (d : ℤ) (n : ℤ) (a_n : ℤ) (a_1 : ℤ) (S_n : ℤ)
  (h_d : d = 2) (h_n : n = 15) (h_a_n : a_n = -10)
  (h_a_1 : a_1 = -38) (h_S_n : S_n = -360) :
  a_n = a_1 + (n - 1) * d ∧ S_n = n * (a_1 + a_n) / 2 :=
by
  sorry

-- Part 2: Geometric Sequence
theorem geometric_sequence_
  (a_1 : ℝ) (q : ℝ) (S_10 : ℝ)
  (a_2 : ℝ) (a_3 : ℝ) (a_4 : ℝ)
  (h_a_2_3 : a_2 + a_3 = 6) (h_a_3_4 : a_3 + a_4 = 12)
  (h_a_1 : a_1 = 1) (h_q : q = 2) (h_S_10 : S_10 = 1023) :
  a_2 = a_1 * q ∧ a_3 = a_1 * q^2 ∧ a_4 = a_1 * q^3 ∧ S_10 = a_1 * (1 - q^10) / (1 - q) :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence__geometric_sequence__l910_91019


namespace NUMINAMATH_GPT_solve_system_of_equations_l910_91097

theorem solve_system_of_equations : 
  ∀ x y : ℝ, 
    (2 * x^2 - 3 * x * y + y^2 = 3) ∧ 
    (x^2 + 2 * x * y - 2 * y^2 = 6) 
    ↔ (x = 2 ∧ y = 1) ∨ (x = -2 ∧ y = -1) :=
by
  sorry

end NUMINAMATH_GPT_solve_system_of_equations_l910_91097


namespace NUMINAMATH_GPT_value_corresponds_l910_91047

-- Define the problem
def certain_number (x : ℝ) : Prop :=
  0.30 * x = 120

-- State the theorem to be proved
theorem value_corresponds (x : ℝ) (h : certain_number x) : 0.40 * x = 160 :=
by
  sorry

end NUMINAMATH_GPT_value_corresponds_l910_91047


namespace NUMINAMATH_GPT_find_pairs_l910_91030

theorem find_pairs (x y : Nat) (h : 1 + x + x^2 + x^3 + x^4 = y^2) : (x, y) = (0, 1) ∨ (x, y) = (3, 11) := by
  sorry

end NUMINAMATH_GPT_find_pairs_l910_91030


namespace NUMINAMATH_GPT_factorial_fraction_simplification_l910_91028

-- Define necessary factorial function
def fact : ℕ → ℕ
| 0       => 1
| (n + 1) => (n + 1) * fact n

-- Define the problem
theorem factorial_fraction_simplification :
  (4 * fact 6 + 20 * fact 5) / fact 7 = 22 / 21 := by
  sorry

end NUMINAMATH_GPT_factorial_fraction_simplification_l910_91028


namespace NUMINAMATH_GPT_area_percentage_decrease_l910_91044

theorem area_percentage_decrease {a b : ℝ} 
  (h1 : 2 * b = 0.1 * 4 * a) :
  ((b^2) / (a^2) * 100 = 4) :=
by
  sorry

end NUMINAMATH_GPT_area_percentage_decrease_l910_91044


namespace NUMINAMATH_GPT_bedroom_curtain_width_l910_91081

theorem bedroom_curtain_width
  (initial_fabric_area : ℕ)
  (living_room_curtain_area : ℕ)
  (fabric_left : ℕ)
  (bedroom_curtain_height : ℕ)
  (bedroom_curtain_area : ℕ)
  (bedroom_curtain_width : ℕ) :
  initial_fabric_area = 16 * 12 →
  living_room_curtain_area = 4 * 6 →
  fabric_left = 160 →
  bedroom_curtain_height = 4 →
  bedroom_curtain_area = 168 - 160 →
  bedroom_curtain_area = bedroom_curtain_width * bedroom_curtain_height →
  bedroom_curtain_width = 2 :=
by
  intros h1 h2 h3 h4 h5 h6
  -- Skipping the proof
  sorry

end NUMINAMATH_GPT_bedroom_curtain_width_l910_91081


namespace NUMINAMATH_GPT_certain_number_is_51_l910_91053

theorem certain_number_is_51 (G C : ℤ) 
  (h1 : G = 33) 
  (h2 : 3 * G = 2 * C - 3) : 
  C = 51 := 
by
  sorry

end NUMINAMATH_GPT_certain_number_is_51_l910_91053


namespace NUMINAMATH_GPT_average_age_l910_91082

theorem average_age (Devin_age Eden_age mom_age : ℕ)
  (h1 : Devin_age = 12)
  (h2 : Eden_age = 2 * Devin_age)
  (h3 : mom_age = 2 * Eden_age) :
  (Devin_age + Eden_age + mom_age) / 3 = 28 := by
  sorry

end NUMINAMATH_GPT_average_age_l910_91082


namespace NUMINAMATH_GPT_pilot_fish_speed_when_moved_away_l910_91022

/-- Conditions -/
def keanu_speed : ℕ := 20
def shark_new_speed (k : ℕ) : ℕ := 2 * k
def pilot_fish_increase_speed (k s_new : ℕ) : ℕ := k + (s_new - k) / 2

/-- The problem statement to prove -/
theorem pilot_fish_speed_when_moved_away (k : ℕ) (s_new : ℕ) (p_new : ℕ) 
  (h1 : k = 20) 
  (h2 : s_new = shark_new_speed k) 
  (h3 : p_new = pilot_fish_increase_speed k s_new) : 
  p_new = 30 :=
by
  rw [h1] at h2
  rw [h2, h1] at h3
  rw [h3]
  sorry

end NUMINAMATH_GPT_pilot_fish_speed_when_moved_away_l910_91022


namespace NUMINAMATH_GPT_which_calc_is_positive_l910_91026

theorem which_calc_is_positive :
  (-3 + 7 - 5 < 0) ∧
  ((1 - 2) * 3 < 0) ∧
  (-16 / (↑(-3)^2) < 0) ∧
  (-2^4 * (-6) > 0) :=
by
sorry

end NUMINAMATH_GPT_which_calc_is_positive_l910_91026


namespace NUMINAMATH_GPT_triangle_inequality_l910_91087

theorem triangle_inequality (ABC: Triangle) (M : Point) (a b c : ℝ)
  (h1 : a = BC) (h2 : b = CA) (h3 : c = AB) :
  (1 / a^2) + (1 / b^2) + (1 / c^2) ≥ 3 / (MA^2 + MB^2 + MC^2) := 
sorry

end NUMINAMATH_GPT_triangle_inequality_l910_91087


namespace NUMINAMATH_GPT_like_terms_mn_l910_91061

theorem like_terms_mn (m n : ℤ) 
  (H1 : m - 2 = 3) 
  (H2 : n + 2 = 1) : 
  m * n = -5 := 
by
  sorry

end NUMINAMATH_GPT_like_terms_mn_l910_91061


namespace NUMINAMATH_GPT_difference_of_two_numbers_l910_91063

theorem difference_of_two_numbers (a b : ℕ) 
(h1 : a + b = 17402) 
(h2 : ∃ k : ℕ, b = 10 * k) 
(h3 : ∃ k : ℕ, a + 9 * k = b) : 
10 * a - a = 14238 :=
by sorry

end NUMINAMATH_GPT_difference_of_two_numbers_l910_91063


namespace NUMINAMATH_GPT_inequality_solution_l910_91041

variable (a x : ℝ)

noncomputable def inequality_solutions :=
  if a = 0 then
    {x | x > 1}
  else if a > 1 then
    {x | (1 / a) < x ∧ x < 1}
  else if a = 1 then
    ∅
  else if 0 < a ∧ a < 1 then
    {x | 1 < x ∧ x < (1 / a)}
  else if a < 0 then
    {x | x < (1 / a) ∨ x > 1}
  else
    ∅

theorem inequality_solution (h : a ≠ 0) :
  if a = 0 then
    ∀ x, (a * x - 1) * (x - 1) < 0 → x > 1
  else if a > 1 then
    ∀ x, (a * x - 1) * (x - 1) < 0 ↔ ((1 / a) < x ∧ x < 1)
  else if a = 1 then
    ∀ x, ¬((a * x - 1) * (x - 1) < 0)
  else if 0 < a ∧ a < 1 then
    ∀ x, (a * x - 1) * (x - 1) < 0 ↔ (1 < x ∧ x < (1 / a))
  else if a < 0 then
    ∀ x, (a * x - 1) * (x - 1) < 0 ↔ (x < (1 / a) ∨ x > 1)
  else
    True := sorry

end NUMINAMATH_GPT_inequality_solution_l910_91041


namespace NUMINAMATH_GPT_average_rate_of_change_nonzero_l910_91069

-- Define the conditions related to the average rate of change.
variables {x0 : ℝ} {Δx : ℝ}

-- Define the statement to prove that in the definition of the average rate of change, Δx ≠ 0.
theorem average_rate_of_change_nonzero (h : Δx ≠ 0) : True :=
sorry  -- The proof is omitted as per instruction.

end NUMINAMATH_GPT_average_rate_of_change_nonzero_l910_91069


namespace NUMINAMATH_GPT_find_m_l910_91073

-- Define the conditions
def parabola_eq (m : ℝ) (x y : ℝ) : Prop := x^2 = m * y
def vertex_to_directrix_dist (d : ℝ) : Prop := d = 1 / 2

-- State the theorem
theorem find_m (m : ℝ) (x y d : ℝ) 
  (h1 : parabola_eq m x y) 
  (h2 : vertex_to_directrix_dist d) :
  m = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l910_91073


namespace NUMINAMATH_GPT_lines_through_three_distinct_points_l910_91099

theorem lines_through_three_distinct_points : 
  ∃ n : ℕ, n = 54 ∧ (∀ (i j k : ℕ), 1 ≤ i ∧ i ≤ 3 ∧ 1 ≤ j ∧ j ≤ 3 ∧ 1 ≤ k ∧ k ≤ 3 → 
  ∃ (a b c : ℤ), -- Direction vector (a, b, c)
  abs a ≤ 1 ∧ abs b ≤ 1 ∧ abs c ≤ 1 ∧
  ((i + a > 0 ∧ i + a ≤ 3) ∧ (j + b > 0 ∧ j + b ≤ 3) ∧ (k + c > 0 ∧ k + c ≤ 3) ∧
  (i + 2 * a > 0 ∧ i + 2 * a ≤ 3) ∧ (j + 2 * b > 0 ∧ j + 2 * b ≤ 3) ∧ (k + 2 * c > 0 ∧ k + 2 * c ≤ 3))) := 
sorry

end NUMINAMATH_GPT_lines_through_three_distinct_points_l910_91099


namespace NUMINAMATH_GPT_two_students_solve_all_problems_l910_91078

theorem two_students_solve_all_problems
    (students : Fin 15 → Fin 6 → Prop)
    (h : ∀ (p : Fin 6), (∃ (s1 s2 s3 s4 s5 s6 s7 s8 : Fin 15), 
          students s1 p ∧ students s2 p ∧ students s3 p ∧ students s4 p ∧ 
          students s5 p ∧ students s6 p ∧ students s7 p ∧ students s8 p)) :
    ∃ (s1 s2 : Fin 15), ∀ (p : Fin 6), students s1 p ∨ students s2 p := 
by
    sorry

end NUMINAMATH_GPT_two_students_solve_all_problems_l910_91078


namespace NUMINAMATH_GPT_maximum_value_F_l910_91093

noncomputable def f (x : Real) : Real := Real.sin x + Real.cos x

noncomputable def f' (x : Real) : Real := Real.cos x - Real.sin x

noncomputable def F (x : Real) : Real := f x * f' x + (f x) ^ 2

theorem maximum_value_F : ∃ x : Real, F x = 1 + Real.sqrt 2 :=
by
  -- The proof steps are to be added here.
  sorry

end NUMINAMATH_GPT_maximum_value_F_l910_91093


namespace NUMINAMATH_GPT_gcd_lcm_1365_910_l910_91017

theorem gcd_lcm_1365_910 :
  gcd 1365 910 = 455 ∧ lcm 1365 910 = 2730 :=
by
  sorry

end NUMINAMATH_GPT_gcd_lcm_1365_910_l910_91017


namespace NUMINAMATH_GPT_cube_volume_l910_91000

theorem cube_volume (a : ℕ) (h : a^3 - ((a - 2) * a * (a + 2)) = 16) : a^3 = 64 := by
  sorry

end NUMINAMATH_GPT_cube_volume_l910_91000


namespace NUMINAMATH_GPT_quadratic_increasing_l910_91023

noncomputable def quadratic (a b c x : ℝ) := a * x^2 + b * x + c

theorem quadratic_increasing (a b c : ℝ) 
  (h1 : quadratic a b c 0 = quadratic a b c 6)
  (h2 : quadratic a b c 0 < quadratic a b c 7) :
  ∀ x, x > 3 → ∀ y, y > 3 → x < y → quadratic a b c x < quadratic a b c y :=
sorry

end NUMINAMATH_GPT_quadratic_increasing_l910_91023


namespace NUMINAMATH_GPT_b_joined_after_a_l910_91085

def months_b_joined (a_investment : ℕ) (b_investment : ℕ) (profit_ratio : ℕ × ℕ) (total_months : ℕ) : ℕ :=
  let a_months := total_months
  let b_months := total_months - (b_investment / (3500 * profit_ratio.snd / profit_ratio.fst / b_investment))
  total_months - b_months

theorem b_joined_after_a (a_investment b_investment total_months : ℕ) (profit_ratio : ℕ × ℕ) (h_a_investment : a_investment = 3500)
   (h_b_investment : b_investment = 21000) (h_profit_ratio : profit_ratio = (2, 3)) : months_b_joined a_investment b_investment profit_ratio total_months = 9 := by
  sorry

end NUMINAMATH_GPT_b_joined_after_a_l910_91085


namespace NUMINAMATH_GPT_kindergarten_children_l910_91036

theorem kindergarten_children (x y z n : ℕ) 
  (h1 : 2 * x + 3 * y + 4 * z = n)
  (h2 : x + y + z = 26)
  : n = 24 := 
sorry

end NUMINAMATH_GPT_kindergarten_children_l910_91036


namespace NUMINAMATH_GPT_square_area_example_l910_91020

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

noncomputable def square_area (x1 y1 x2 y2 : ℝ) : ℝ :=
  (distance x1 y1 x2 y2)^2

theorem square_area_example : square_area 1 3 5 6 = 25 :=
by
  sorry

end NUMINAMATH_GPT_square_area_example_l910_91020


namespace NUMINAMATH_GPT_min_value_of_xy_l910_91034

theorem min_value_of_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_eq : 4 * x * y - x - 2 * y = 4) : 
  xy >= 2 :=
sorry

end NUMINAMATH_GPT_min_value_of_xy_l910_91034


namespace NUMINAMATH_GPT_AC_total_l910_91015

theorem AC_total (A B C : ℕ) (h1 : A + B + C = 600) (h2 : B + C = 450) (h3 : C = 100) : A + C = 250 := by
  sorry

end NUMINAMATH_GPT_AC_total_l910_91015


namespace NUMINAMATH_GPT_tangent_chord_equation_l910_91018

theorem tangent_chord_equation (x1 y1 x2 y2 : ℝ) :
  (x1^2 + y1^2 = 1) →
  (x2^2 + y2^2 = 1) →
  (2*x1 + 2*y1 + 1 = 0) →
  (2*x2 + 2*y2 + 1 = 0) →
  ∀ (x y : ℝ), 2*x + 2*y + 1 = 0 :=
by
  intros hx1 hy1 hx2 hy2 x y
  exact sorry

end NUMINAMATH_GPT_tangent_chord_equation_l910_91018


namespace NUMINAMATH_GPT_perfect_square_condition_l910_91043

noncomputable def isPerfectSquareQuadratic (m : ℤ) (x y : ℤ) :=
  ∃ (k : ℤ), (4 * x^2 + m * x * y + 25 * y^2) = k^2

theorem perfect_square_condition (m : ℤ) :
  (∀ x y : ℤ, isPerfectSquareQuadratic m x y) → (m = 20 ∨ m = -20) :=
by
  sorry

end NUMINAMATH_GPT_perfect_square_condition_l910_91043


namespace NUMINAMATH_GPT_parallelogram_proof_l910_91004

noncomputable def sin_angle_degrees (θ : ℝ) : ℝ := Real.sin (θ * Real.pi / 180)

theorem parallelogram_proof (x : ℝ) (A : ℝ) (r : ℝ) (side1 side2 : ℝ) (P : ℝ):
  (A = 972) → (r = 4 / 3) → (sin_angle_degrees 45 = Real.sqrt 2 / 2) →
  (side1 = 4 * x) → (side2 = 3 * x) →
  (A = side1 * (side2 * (Real.sqrt 2 / 2 / 3))) →
  x = 9 * 2^(3/4) →
  side1 = 36 * 2^(3/4) →
  side2 = 27 * 2^(3/4) →
  (P = 2 * (side1 + side2)) →
  (P = 126 * 2^(3/4)) :=
by
  intros
  sorry

end NUMINAMATH_GPT_parallelogram_proof_l910_91004


namespace NUMINAMATH_GPT_total_money_l910_91046

theorem total_money (A B C : ℕ) (h1 : A + C = 200) (h2 : B + C = 330) (h3 : C = 30) : 
  A + B + C = 500 :=
by
  sorry

end NUMINAMATH_GPT_total_money_l910_91046


namespace NUMINAMATH_GPT_wipes_per_pack_l910_91076

theorem wipes_per_pack (days : ℕ) (wipes_per_day : ℕ) (packs : ℕ) (total_wipes : ℕ) (n : ℕ)
    (h1 : days = 360)
    (h2 : wipes_per_day = 2)
    (h3 : packs = 6)
    (h4 : total_wipes = wipes_per_day * days)
    (h5 : total_wipes = n * packs) : 
    n = 120 := 
by 
  sorry

end NUMINAMATH_GPT_wipes_per_pack_l910_91076


namespace NUMINAMATH_GPT_reinforcement_size_l910_91039

theorem reinforcement_size (R : ℕ) : 
  2000 * 39 = (2000 + R) * 20 → R = 1900 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_reinforcement_size_l910_91039


namespace NUMINAMATH_GPT_range_of_m_l910_91074

variable {x m : ℝ}

def quadratic (x m : ℝ) : ℝ := x^2 + (m - 1) * x + (m^2 - 3 * m + 1)

def absolute_quadratic (x m : ℝ) : ℝ := abs (quadratic x m)

theorem range_of_m (h : ∀ x ∈ Set.Icc (-1 : ℝ) 0, absolute_quadratic x m ≥ absolute_quadratic (x - 1) m) :
  m = 1 ∨ m ≥ 3 :=
sorry

end NUMINAMATH_GPT_range_of_m_l910_91074


namespace NUMINAMATH_GPT_find_n_l910_91042

theorem find_n (n : ℤ) (h : (n + 1999) / 2 = -1) : n = -2001 := 
sorry

end NUMINAMATH_GPT_find_n_l910_91042


namespace NUMINAMATH_GPT_five_eight_sided_dice_not_all_same_l910_91094

noncomputable def probability_not_all_same : ℚ :=
  let total_outcomes := 8^5
  let same_number_outcomes := 8
  1 - (same_number_outcomes / total_outcomes)

theorem five_eight_sided_dice_not_all_same :
  probability_not_all_same = 4095 / 4096 :=
by
  sorry

end NUMINAMATH_GPT_five_eight_sided_dice_not_all_same_l910_91094


namespace NUMINAMATH_GPT_lap_length_l910_91075

theorem lap_length (I P : ℝ) (K : ℝ) 
  (h1 : 2 * I - 2 * P = 3 * K) 
  (h2 : 3 * I + 10 - 3 * P = 7 * K) : 
  K = 4 :=
by 
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_lap_length_l910_91075


namespace NUMINAMATH_GPT_mara_correct_answers_l910_91064

theorem mara_correct_answers :
  let math_total    := 30
  let science_total := 20
  let history_total := 50
  let math_percent  := 0.85
  let science_percent := 0.75
  let history_percent := 0.65
  let math_correct  := math_percent * math_total
  let science_correct := science_percent * science_total
  let history_correct := history_percent * history_total
  let total_correct := math_correct + science_correct + history_correct
  let total_problems := math_total + science_total + history_total
  let overall_percent := total_correct / total_problems
  overall_percent = 0.73 :=
by
  sorry

end NUMINAMATH_GPT_mara_correct_answers_l910_91064


namespace NUMINAMATH_GPT_original_inhabitants_l910_91056

theorem original_inhabitants (X : ℝ) 
  (h1 : 10 ≤ X) 
  (h2 : 0.9 * X * 0.75 + 0.225 * X * 0.15 = 5265) : 
  X = 7425 := 
sorry

end NUMINAMATH_GPT_original_inhabitants_l910_91056


namespace NUMINAMATH_GPT_divisible_by_five_l910_91071

theorem divisible_by_five (x y z : ℤ) (hxy : x ≠ y) (hyz : y ≠ z) (hzx : z ≠ x) :
  ∃ k : ℤ, (x-y)^5 + (y-z)^5 + (z-x)^5 = 5 * k * (y-z) * (z-x) * (x-y) :=
  sorry

end NUMINAMATH_GPT_divisible_by_five_l910_91071


namespace NUMINAMATH_GPT_solve_for_x_l910_91009

theorem solve_for_x (x : ℝ) (h1 : (3 - x) / (x + 2) + (3 * x - 9) / (3 - x) = 2) 
  (h2 : x ≠ -2) (h3 : x ≠ 3) : x = -1 / 6 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l910_91009


namespace NUMINAMATH_GPT_value_of_r_when_n_is_3_l910_91066

def r (s : ℕ) : ℕ := 4^s - 2 * s
def s (n : ℕ) : ℕ := 3^n + 2
def n : ℕ := 3

theorem value_of_r_when_n_is_3 : r (s n) = 4^29 - 58 :=
by
  sorry

end NUMINAMATH_GPT_value_of_r_when_n_is_3_l910_91066


namespace NUMINAMATH_GPT_jean_total_calories_l910_91010

-- Define the conditions
def pages_per_donut : ℕ := 2
def written_pages : ℕ := 12
def calories_per_donut : ℕ := 150

-- Define the question as a theorem
theorem jean_total_calories : (written_pages / pages_per_donut) * calories_per_donut = 900 := by
  sorry

end NUMINAMATH_GPT_jean_total_calories_l910_91010


namespace NUMINAMATH_GPT_total_votes_l910_91033

theorem total_votes (Ben_votes Matt_votes total_votes : ℕ)
  (h_ratio : 2 * Matt_votes = 3 * Ben_votes)
  (h_Ben_votes : Ben_votes = 24) :
  total_votes = Ben_votes + Matt_votes :=
sorry

end NUMINAMATH_GPT_total_votes_l910_91033


namespace NUMINAMATH_GPT_simplify_power_of_power_l910_91008

theorem simplify_power_of_power (a : ℝ) : (a^2)^3 = a^6 :=
by 
  sorry

end NUMINAMATH_GPT_simplify_power_of_power_l910_91008


namespace NUMINAMATH_GPT_ratio_red_to_black_l910_91091

theorem ratio_red_to_black (a b x : ℕ) (h1 : x + b = 3 * a) (h2 : x = 2 * b - 3 * a) :
  a / b = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_ratio_red_to_black_l910_91091


namespace NUMINAMATH_GPT_min_value_frac_l910_91007

theorem min_value_frac (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) :
  ∃ (x : ℝ), x = 16 ∧ (forall y, y = 9 / a + 1 / b → x ≤ y) :=
sorry

end NUMINAMATH_GPT_min_value_frac_l910_91007


namespace NUMINAMATH_GPT_rashmi_speed_second_day_l910_91065

noncomputable def rashmi_speed (distance speed1 time_late time_early : ℝ) : ℝ :=
  let time1 := distance / speed1
  let on_time := time1 - time_late / 60
  let time2 := on_time - time_early / 60
  distance / time2

theorem rashmi_speed_second_day :
  rashmi_speed 9.999999999999993 5 10 10 = 6 := by
  sorry

end NUMINAMATH_GPT_rashmi_speed_second_day_l910_91065


namespace NUMINAMATH_GPT_calculate_g_l910_91083

def g (a b c : ℚ) : ℚ := (2 * c + a) / (b - c)

theorem calculate_g : g 3 6 (-1) = 1 / 7 :=
by
    -- Proof is not included
    sorry

end NUMINAMATH_GPT_calculate_g_l910_91083


namespace NUMINAMATH_GPT_first_candidate_percentage_l910_91086

noncomputable
def passing_marks_approx : ℝ := 240

noncomputable
def total_marks (P : ℝ) : ℝ := (P + 30) / 0.45

noncomputable
def percentage_marks (T P : ℝ) : ℝ := ((P - 60) / T) * 100

theorem first_candidate_percentage :
  let P := passing_marks_approx
  let T := total_marks P
  percentage_marks T P = 30 :=
by
  sorry

end NUMINAMATH_GPT_first_candidate_percentage_l910_91086


namespace NUMINAMATH_GPT_bellas_score_l910_91037

-- Definitions from the problem conditions
def n : Nat := 17
def x : Nat := 75
def new_n : Nat := n + 1
def y : Nat := 76

-- Assertion that Bella's score is 93
theorem bellas_score : (new_n * y) - (n * x) = 93 :=
by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_bellas_score_l910_91037


namespace NUMINAMATH_GPT_negation_of_statement_l910_91027

theorem negation_of_statement (x : ℝ) :
  (¬ (x^2 = 1 → x = 1 ∨ x = -1)) ↔ (x^2 = 1 ∧ (x ≠ 1 ∧ x ≠ -1)) :=
sorry

end NUMINAMATH_GPT_negation_of_statement_l910_91027


namespace NUMINAMATH_GPT_sum_of_cubes_l910_91052

-- Definitions based on the conditions
variables (a b : ℝ)
variables (h1 : a + b = 2) (h2 : a * b = -3)

-- The Lean statement to prove the sum of their cubes is 26
theorem sum_of_cubes (a b : ℝ) (h1 : a + b = 2) (h2 : a * b = -3) : a^3 + b^3 = 26 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_cubes_l910_91052


namespace NUMINAMATH_GPT_second_smallest_sum_l910_91031

theorem second_smallest_sum (a b c d : ℕ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d)
                           (h4 : b ≠ c) (h5 : b ≠ d) (h6 : c ≠ d)
                           (h7 : a + b + c = 180) (h8 : a + c + d = 197)
                           (h9 : b + c + d = 208) (h10 : a + b + d = 222) :
  208 ≠ 180 ∧ 208 ≠ 197 ∧ 208 ≠ 222 := 
sorry

end NUMINAMATH_GPT_second_smallest_sum_l910_91031


namespace NUMINAMATH_GPT_afternoon_pear_sales_l910_91095

theorem afternoon_pear_sales (morning_sales afternoon_sales total_sales : ℕ)
  (h1 : afternoon_sales = 2 * morning_sales)
  (h2 : total_sales = morning_sales + afternoon_sales)
  (h3 : total_sales = 420) : 
  afternoon_sales = 280 :=
by {
  -- placeholders for the proof
  sorry 
}

end NUMINAMATH_GPT_afternoon_pear_sales_l910_91095


namespace NUMINAMATH_GPT_age_problem_l910_91060

theorem age_problem (c b a : ℕ) (h1 : b = 2 * c) (h2 : a = b + 2) (h3 : a + b + c = 47) : b = 18 :=
by
  sorry

end NUMINAMATH_GPT_age_problem_l910_91060


namespace NUMINAMATH_GPT_country_x_income_l910_91045

variable (income : ℝ)
variable (tax_paid : ℝ)
variable (income_first_40000_tax : ℝ := 40000 * 0.1)
variable (income_above_40000_tax_rate : ℝ := 0.2)
variable (total_tax_paid : ℝ := 8000)
variable (income_above_40000 : ℝ := (total_tax_paid - income_first_40000_tax) / income_above_40000_tax_rate)

theorem country_x_income : 
  income = 40000 + income_above_40000 → 
  total_tax_paid = tax_paid → 
  tax_paid = income_first_40000_tax + (income_above_40000 * income_above_40000_tax_rate) →
  income = 60000 :=
by sorry

end NUMINAMATH_GPT_country_x_income_l910_91045


namespace NUMINAMATH_GPT_purely_imaginary_a_eq_2_l910_91024

theorem purely_imaginary_a_eq_2 (a : ℝ) (h : (2 - a) / 2 = 0) : a = 2 :=
sorry

end NUMINAMATH_GPT_purely_imaginary_a_eq_2_l910_91024


namespace NUMINAMATH_GPT_find_value_of_a_l910_91054

theorem find_value_of_a (a : ℝ) (h : (3 + a + 10) / 3 = 5) : a = 2 := 
by {
  sorry
}

end NUMINAMATH_GPT_find_value_of_a_l910_91054


namespace NUMINAMATH_GPT_rectangular_plot_breadth_l910_91055

theorem rectangular_plot_breadth:
  ∀ (b l : ℝ), (l = b + 10) → (24 * b = l * b) → b = 14 :=
by
  intros b l hl hs
  sorry

end NUMINAMATH_GPT_rectangular_plot_breadth_l910_91055


namespace NUMINAMATH_GPT_sourdough_cost_eq_nine_l910_91049

noncomputable def cost_per_visit (white_bread_cost baguette_cost croissant_cost: ℕ) : ℕ :=
  2 * white_bread_cost + baguette_cost + croissant_cost

noncomputable def total_spent (weekly_cost num_weeks: ℕ) : ℕ :=
  weekly_cost * num_weeks

noncomputable def total_sourdough_spent (total_spent weekly_cost num_weeks: ℕ) : ℕ :=
  total_spent - weekly_cost * num_weeks

noncomputable def total_sourdough_per_week (total_sourdough_spent num_weeks: ℕ) : ℕ :=
  total_sourdough_spent / num_weeks

theorem sourdough_cost_eq_nine (white_bread_cost baguette_cost croissant_cost total_spent_over_4_weeks: ℕ)
  (h₁: white_bread_cost = 350) (h₂: baguette_cost = 150) (h₃: croissant_cost = 200) (h₄: total_spent_over_4_weeks = 7800) :
  total_sourdough_per_week (total_sourdough_spent total_spent_over_4_weeks (cost_per_visit white_bread_cost baguette_cost croissant_cost) 4) 4 = 900 :=
by 
  sorry

end NUMINAMATH_GPT_sourdough_cost_eq_nine_l910_91049


namespace NUMINAMATH_GPT_log10_cubic_solution_l910_91072

noncomputable def log10 (x: ℝ) : ℝ := Real.log x / Real.log 10

open Real

theorem log10_cubic_solution 
  (x : ℝ) 
  (hx1 : x < 1) 
  (hx2 : (log10 x)^3 - log10 (x^4) = 640) : 
  (log10 x)^4 - log10 (x^4) = 645 := 
by 
  sorry

end NUMINAMATH_GPT_log10_cubic_solution_l910_91072


namespace NUMINAMATH_GPT_geometric_sequence_seventh_term_l910_91005

theorem geometric_sequence_seventh_term (a r : ℝ) (ha : 0 < a) (hr : 0 < r) 
  (h4 : a * r^3 = 16) (h10 : a * r^9 = 2) : 
  a * r^6 = 2 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_seventh_term_l910_91005


namespace NUMINAMATH_GPT_sum_of_ages_l910_91092

variables (Matthew Rebecca Freddy: ℕ)
variables (H1: Matthew = Rebecca + 2)
variables (H2: Matthew = Freddy - 4)
variables (H3: Freddy = 15)

theorem sum_of_ages
  (H1: Matthew = Rebecca + 2)
  (H2: Matthew = Freddy - 4)
  (H3: Freddy = 15):
  Matthew + Rebecca + Freddy = 35 :=
  sorry

end NUMINAMATH_GPT_sum_of_ages_l910_91092


namespace NUMINAMATH_GPT_rachel_speed_painting_video_time_l910_91002

theorem rachel_speed_painting_video_time :
  let num_videos := 4
  let setup_time := 1
  let cleanup_time := 1
  let painting_time_per_video := 1
  let editing_time_per_video := 1.5
  (setup_time + cleanup_time + painting_time_per_video * num_videos + editing_time_per_video * num_videos) / num_videos = 3 :=
by
  sorry

end NUMINAMATH_GPT_rachel_speed_painting_video_time_l910_91002


namespace NUMINAMATH_GPT_total_messages_l910_91068

theorem total_messages (x : ℕ) (h : x * (x - 1) = 420) : x * (x - 1) = 420 :=
by
  sorry

end NUMINAMATH_GPT_total_messages_l910_91068


namespace NUMINAMATH_GPT_price_of_each_apple_l910_91098

-- Define the constants and conditions
def price_banana : ℝ := 0.60
def total_fruits : ℕ := 9
def total_cost : ℝ := 5.60

-- Declare the variables for number of apples and price of apples
variables (A : ℝ) (x y : ℕ)

-- Define the conditions in Lean
axiom h1 : x + y = total_fruits
axiom h2 : A * x + price_banana * y = total_cost

-- Prove that the price of each apple is $0.80
theorem price_of_each_apple : A = 0.80 :=
by sorry

end NUMINAMATH_GPT_price_of_each_apple_l910_91098


namespace NUMINAMATH_GPT_solution_y_eq_2_l910_91006

theorem solution_y_eq_2 (y : ℝ) (h_pos : y > 0) (h_eq : y^6 = 64) : y = 2 :=
sorry

end NUMINAMATH_GPT_solution_y_eq_2_l910_91006


namespace NUMINAMATH_GPT_betty_bracelets_l910_91070

theorem betty_bracelets : (140 / 14) = 10 := 
by
  norm_num

end NUMINAMATH_GPT_betty_bracelets_l910_91070


namespace NUMINAMATH_GPT_highest_power_of_3_dividing_N_is_1_l910_91016

-- Define the integer N as described in the problem
def N : ℕ := 313233515253

-- State the problem
theorem highest_power_of_3_dividing_N_is_1 : ∃ k : ℕ, (3^k ∣ N) ∧ ∀ m > 1, ¬ (3^m ∣ N) ∧ k = 1 :=
by
  -- Specific solution details and steps are not required here
  sorry

end NUMINAMATH_GPT_highest_power_of_3_dividing_N_is_1_l910_91016


namespace NUMINAMATH_GPT_proof_problem_l910_91088

theorem proof_problem (x y z : ℝ) (h₁ : x ≠ y) 
  (h₂ : (x^2 - y*z) / (x * (1 - y*z)) = (y^2 - x*z) / (y * (1 - x*z))) :
  x + y + z = 1/x + 1/y + 1/z :=
sorry

end NUMINAMATH_GPT_proof_problem_l910_91088


namespace NUMINAMATH_GPT_geometric_seq_increasing_condition_l910_91040

theorem geometric_seq_increasing_condition (q : ℝ) (a : ℕ → ℝ): 
  (∀ n : ℕ, a (n + 1) = q * a n) → (¬ (∀ a : ℕ → ℝ, (∀ n : ℕ, a (n + 1) = q * a n) → ∀ n m : ℕ, n < m → a n < a m) ∧ ¬ (¬ (∀ a : ℕ → ℝ, (∀ n : ℕ, a (n + 1) = q * a n) → ∀ n m : ℕ, n < m → a n < a m))) :=
sorry

end NUMINAMATH_GPT_geometric_seq_increasing_condition_l910_91040


namespace NUMINAMATH_GPT_min_value_expression_l910_91025

theorem min_value_expression : 
  ∃ (x y : ℝ), x^2 + 2 * x * y + 2 * y^2 + 3 * x - 5 * y = -8.5 := by
  sorry

end NUMINAMATH_GPT_min_value_expression_l910_91025


namespace NUMINAMATH_GPT_problem_solution_l910_91012

theorem problem_solution
  (a b : ℝ)
  (h1 : a * b = 2)
  (h2 : a - b = 3) :
  a^3 * b - 2 * a^2 * b^2 + a * b^3 = 18 :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l910_91012


namespace NUMINAMATH_GPT_inverse_sum_l910_91079

noncomputable def g (x : ℝ) : ℝ :=
if x < 15 then 2 * x + 4 else 3 * x - 1

theorem inverse_sum :
  g⁻¹ (10) + g⁻¹ (50) = 20 :=
sorry

end NUMINAMATH_GPT_inverse_sum_l910_91079


namespace NUMINAMATH_GPT_inequality_for_all_real_l910_91011

theorem inequality_for_all_real (a b c : ℝ) : 
  a^6 + b^6 + c^6 - 3 * a^2 * b^2 * c^2 ≥ 1/2 * (a - b)^2 * (b - c)^2 * (c - a)^2 :=
by 
  sorry

end NUMINAMATH_GPT_inequality_for_all_real_l910_91011


namespace NUMINAMATH_GPT_arithmetic_sequence_general_term_l910_91090

theorem arithmetic_sequence_general_term (S : ℕ → ℤ) (a : ℕ → ℤ) :
  (∀ n, S n = 3 * n^2 + 2 * n) →
  a 1 = S 1 ∧ (∀ n ≥ 2, a n = S n - S (n - 1)) →
  ∀ n, a n = 6 * n - 1 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_general_term_l910_91090


namespace NUMINAMATH_GPT_find_part_length_in_inches_find_part_length_in_feet_and_inches_l910_91057

def feetToInches (feet : ℕ) : ℕ := feet * 12

def totalLengthInInches (feet : ℕ) (inches : ℕ) : ℕ := feetToInches feet + inches

def partLengthInInches (totalLength : ℕ) (parts : ℕ) : ℕ := totalLength / parts

def inchesToFeetAndInches (inches : ℕ) : Nat × Nat := (inches / 12, inches % 12)

theorem find_part_length_in_inches (feet : ℕ) (inches : ℕ) (parts : ℕ)
    (h1 : feet = 10) (h2 : inches = 5) (h3 : parts = 5) :
    partLengthInInches (totalLengthInInches feet inches) parts = 25 := by
  sorry

theorem find_part_length_in_feet_and_inches (feet : ℕ) (inches : ℕ) (parts : ℕ)
    (h1 : feet = 10) (h2 : inches = 5) (h3 : parts = 5) :
    inchesToFeetAndInches (partLengthInInches (totalLengthInInches feet inches) parts) = (2, 1) := by
  sorry

end NUMINAMATH_GPT_find_part_length_in_inches_find_part_length_in_feet_and_inches_l910_91057


namespace NUMINAMATH_GPT_initial_amount_l910_91058

theorem initial_amount (spent_sweets friends_each left initial : ℝ) 
  (h1 : spent_sweets = 3.25) (h2 : friends_each = 2.20) (h3 : left = 2.45) :
  initial = spent_sweets + (friends_each * 2) + left :=
by
  sorry

end NUMINAMATH_GPT_initial_amount_l910_91058


namespace NUMINAMATH_GPT_initial_volume_of_solution_l910_91014

theorem initial_volume_of_solution (V : ℝ) (h0 : 0.10 * V = 0.08 * (V + 20)) : V = 80 :=
by
  sorry

end NUMINAMATH_GPT_initial_volume_of_solution_l910_91014


namespace NUMINAMATH_GPT_sequence_sum_l910_91029

theorem sequence_sum (a : ℕ → ℚ) (S : ℕ → ℚ) (h : ∀ n : ℕ, S n + a n = 2 * n + 1) :
  ∀ n : ℕ, a n = 2 - (1 / 2^n) :=
by
  sorry

end NUMINAMATH_GPT_sequence_sum_l910_91029


namespace NUMINAMATH_GPT_simplify_expression_l910_91062

theorem simplify_expression : 4 * (14 / 5) * (20 / -42) = -4 / 15 := 
by sorry

end NUMINAMATH_GPT_simplify_expression_l910_91062


namespace NUMINAMATH_GPT_find_smallest_x_l910_91048

theorem find_smallest_x :
  ∃ x : ℕ, x > 0 ∧
  (45 * x + 9) % 25 = 3 ∧
  (2 * x) % 5 = 8 ∧
  x = 20 :=
by
  sorry

end NUMINAMATH_GPT_find_smallest_x_l910_91048


namespace NUMINAMATH_GPT_evaluate_x_from_geometric_series_l910_91051

theorem evaluate_x_from_geometric_series (x : ℝ) (h : ∑' n : ℕ, x ^ n = 4) : x = 3 / 4 :=
sorry

end NUMINAMATH_GPT_evaluate_x_from_geometric_series_l910_91051


namespace NUMINAMATH_GPT_add_pure_acid_to_obtain_final_concentration_l910_91021

   variable (x : ℝ)

   def initial_solution_volume : ℝ := 60
   def initial_acid_concentration : ℝ := 0.10
   def final_acid_concentration : ℝ := 0.15

   axiom calculate_pure_acid (x : ℝ) :
     initial_acid_concentration * initial_solution_volume + x = final_acid_concentration * (initial_solution_volume + x)

   noncomputable def pure_acid_solution : ℝ := 3/0.85

   theorem add_pure_acid_to_obtain_final_concentration :
     x = pure_acid_solution := by
     sorry
   
end NUMINAMATH_GPT_add_pure_acid_to_obtain_final_concentration_l910_91021


namespace NUMINAMATH_GPT_area_square_EFGH_l910_91003

theorem area_square_EFGH (AB BE : ℝ) (h : BE = 2) (h2 : AB = 10) :
  ∃ s : ℝ, (s = 8 * Real.sqrt 6 - 2) ∧ s^2 = (8 * Real.sqrt 6 - 2)^2 := by
  sorry

end NUMINAMATH_GPT_area_square_EFGH_l910_91003


namespace NUMINAMATH_GPT_calculate_expression_l910_91050

theorem calculate_expression :
  -15 - 21 + 8 = -28 :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l910_91050


namespace NUMINAMATH_GPT_brian_tape_needed_l910_91089

-- Define lengths and number of each type of box
def long_side_15_30 := 32
def short_side_15_30 := 17
def num_15_30 := 5

def side_40_40 := 42
def num_40_40 := 2

def long_side_20_50 := 52
def short_side_20_50 := 22
def num_20_50 := 3

-- Calculate the total tape required
def total_tape : Nat :=
  (num_15_30 * (long_side_15_30 + 2 * short_side_15_30)) +
  (num_40_40 * (3 * side_40_40)) +
  (num_20_50 * (long_side_20_50 + 2 * short_side_20_50))

-- Proof statement
theorem brian_tape_needed : total_tape = 870 := by
  sorry

end NUMINAMATH_GPT_brian_tape_needed_l910_91089


namespace NUMINAMATH_GPT_train_travel_time_l910_91059

theorem train_travel_time 
  (speed : ℝ := 120) -- speed in kmph
  (distance : ℝ := 80) -- distance in km
  (minutes_in_hour : ℝ := 60) -- conversion factor
  : (distance / speed) * minutes_in_hour = 40 :=
by
  -- Sorry is used as a placeholder for the proof
  sorry

end NUMINAMATH_GPT_train_travel_time_l910_91059


namespace NUMINAMATH_GPT_geo_seq_product_l910_91084

theorem geo_seq_product (a : ℕ → ℝ) (r : ℝ) (h_pos : ∀ n, 0 < a n) 
  (h_geom : ∀ n, a (n + 1) = a n * r) (h_a1a9 : a 1 * a 9 = 16) :
  a 2 * a 5 * a 8 = 64 :=
sorry

end NUMINAMATH_GPT_geo_seq_product_l910_91084


namespace NUMINAMATH_GPT_shifted_graph_sum_l910_91035

noncomputable def f (x : ℝ) : ℝ := 3 * x ^ 2 - 2 * x + 5

def shift_right (f : ℝ → ℝ) (h : ℝ) (x : ℝ) : ℝ := f (x - h)
def shift_up (f : ℝ → ℝ) (k : ℝ) (x : ℝ) : ℝ := f x + k

noncomputable def g (x : ℝ) : ℝ := shift_up (shift_right f 7) 3 x

theorem shifted_graph_sum : (∃ (a b c : ℝ), g x = a * x ^ 2 + b * x + c ∧ (a + b + c = 128)) :=
by
  sorry

end NUMINAMATH_GPT_shifted_graph_sum_l910_91035


namespace NUMINAMATH_GPT_tiffany_max_points_l910_91096

section
  variables
  (initial_money : ℕ := 3)
  (cost_per_game : ℕ := 1)
  (rings_per_game : ℕ := 5)
  (points_red_bucket : ℕ := 2)
  (points_green_bucket : ℕ := 3)
  (points_miss : ℕ := 0)
  (games_played : ℕ := 2)
  (red_buckets : ℕ := 4)
  (green_buckets : ℕ := 5)
  (additional_games : ℕ := initial_money - games_played)
  (points_per_game_from_green_buckets : ℕ := rings_per_game * points_green_bucket)
  (total_points : ℕ := (red_buckets * points_red_bucket) + (green_buckets * points_green_bucket) + (additional_games * points_per_game_from_green_buckets))

  theorem tiffany_max_points : total_points = 38 := 
  sorry
end

end NUMINAMATH_GPT_tiffany_max_points_l910_91096
