import Mathlib

namespace NUMINAMATH_GPT_sum_of_two_numbers_l1875_187572

theorem sum_of_two_numbers :
  ∀ (A B : ℚ), (A - B = 8) → (1 / 4 * (A + B) = 6) → (A = 16) → (A + B = 24) :=
by
  intros A B h1 h2 h3
  sorry

end NUMINAMATH_GPT_sum_of_two_numbers_l1875_187572


namespace NUMINAMATH_GPT_power_equality_l1875_187576

theorem power_equality (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
sorry

end NUMINAMATH_GPT_power_equality_l1875_187576


namespace NUMINAMATH_GPT_probability_of_factor_less_than_ten_is_half_l1875_187587

-- Definitions for the factors and counts
def numFactors (n : ℕ) : ℕ :=
  let psa := 1;
  let psb := 2;
  let psc := 1;
  (psa + 1) * (psb + 1) * (psc + 1)

def factorsLessThanTen (n : ℕ) : List ℕ :=
  if n = 90 then [1, 2, 3, 5, 6, 9] else []

def probabilityLessThanTen (n : ℕ) : ℚ :=
  let totalFactors := numFactors n;
  let lessThanTenFactors := factorsLessThanTen n;
  let favorableOutcomes := lessThanTenFactors.length;
  favorableOutcomes / totalFactors

-- The proof statement
theorem probability_of_factor_less_than_ten_is_half :
  probabilityLessThanTen 90 = 1 / 2 := sorry

end NUMINAMATH_GPT_probability_of_factor_less_than_ten_is_half_l1875_187587


namespace NUMINAMATH_GPT_abc_product_le_two_l1875_187535

theorem abc_product_le_two (a b c : ℝ) (ha : 0 ≤ a) (ha2 : a ≤ 2) (hb : 0 ≤ b) (hb2 : b ≤ 2) (hc : 0 ≤ c) (hc2 : c ≤ 2) :
  (a - b) * (b - c) * (a - c) ≤ 2 :=
sorry

end NUMINAMATH_GPT_abc_product_le_two_l1875_187535


namespace NUMINAMATH_GPT_number_of_toys_l1875_187591

-- Definitions based on conditions
def selling_price : ℝ := 18900
def cost_price_per_toy : ℝ := 900
def gain_per_toy : ℝ := 3 * cost_price_per_toy

-- The number of toys sold
noncomputable def number_of_toys_sold (SP CP gain : ℝ) : ℝ :=
  (SP - gain) / CP

-- The theorem statement to prove
theorem number_of_toys (SP CP gain : ℝ) : number_of_toys_sold SP CP gain = 18 :=
by
  have h1: SP = 18900 := by sorry
  have h2: CP = 900 := by sorry
  have h3: gain = 3 * CP := by sorry
  -- Further steps to establish the proof
  sorry

end NUMINAMATH_GPT_number_of_toys_l1875_187591


namespace NUMINAMATH_GPT_basic_astrophysics_degrees_l1875_187528

-- Define the given percentages
def microphotonics_percentage : ℝ := 14
def home_electronics_percentage : ℝ := 24
def food_additives_percentage : ℝ := 10
def gmo_percentage : ℝ := 29
def industrial_lubricants_percentage : ℝ := 8
def total_circle_degrees : ℝ := 360

-- Define a proof problem to show that basic astrophysics research occupies 54 degrees in the circle
theorem basic_astrophysics_degrees :
  total_circle_degrees - (microphotonics_percentage + home_electronics_percentage + food_additives_percentage + gmo_percentage + industrial_lubricants_percentage) = 15 ∧
  0.15 * total_circle_degrees = 54 :=
by
  sorry

end NUMINAMATH_GPT_basic_astrophysics_degrees_l1875_187528


namespace NUMINAMATH_GPT_non_congruent_right_triangles_count_l1875_187593

def isRightTriangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

def areaEqualsFourTimesPerimeter (a b c : ℕ) : Prop :=
  a * b = 8 * (a + b + c)

theorem non_congruent_right_triangles_count :
  {n : ℕ // ∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ isRightTriangle a b c ∧ areaEqualsFourTimesPerimeter a b c ∧ n = 3} := sorry

end NUMINAMATH_GPT_non_congruent_right_triangles_count_l1875_187593


namespace NUMINAMATH_GPT_smallest_stable_triangle_side_length_l1875_187552

/-- The smallest possible side length that can appear in any stable triangle with side lengths that 
are multiples of 5, 80, and 112, respectively, is 20. -/
theorem smallest_stable_triangle_side_length {a b c : ℕ} 
  (hab : ∃ k₁, a = 5 * k₁) 
  (hbc : ∃ k₂, b = 80 * k₂) 
  (hac : ∃ k₃, c = 112 * k₃) 
  (abc_triangle : a + b > c ∧ a + c > b ∧ b + c > a) : 
  a = 20 ∨ b = 20 ∨ c = 20 :=
sorry

end NUMINAMATH_GPT_smallest_stable_triangle_side_length_l1875_187552


namespace NUMINAMATH_GPT_polygon_sides_l1875_187594

theorem polygon_sides (n : ℕ) : 
  ((n - 2) * 180 = 4 * 360) → n = 10 :=
by
  sorry

end NUMINAMATH_GPT_polygon_sides_l1875_187594


namespace NUMINAMATH_GPT_eval_expression_at_a_l1875_187558

theorem eval_expression_at_a (a : ℝ) (h : a = 1 / 2) : (2 * a⁻¹ + a⁻¹ / 2) / a = 10 :=
by
  sorry

end NUMINAMATH_GPT_eval_expression_at_a_l1875_187558


namespace NUMINAMATH_GPT_taxi_company_charges_l1875_187562

theorem taxi_company_charges
  (X : ℝ)  -- charge for the first 1/5 of a mile
  (C : ℝ)  -- charge for each additional 1/5 of a mile
  (total_charge : ℝ)  -- total charge for an 8-mile ride
  (remaining_distance_miles : ℝ)  -- remaining miles after the first 1/5 mile
  (remaining_increments : ℝ)  -- remaining 1/5 mile increments
  (charge_increments : ℝ)  -- total charge for remaining increments
  (X_val : X = 2.50)
  (C_val : C = 0.40)
  (total_charge_val : total_charge = 18.10)
  (remaining_distance_miles_val : remaining_distance_miles = 7.8)
  (remaining_increments_val : remaining_increments = remaining_distance_miles * 5)
  (charge_increments_val : charge_increments = remaining_increments * C)
  (proof_1: charge_increments = 15.60)
  (proof_2: total_charge - charge_increments = X) : X = 2.50 := 
by
  sorry

end NUMINAMATH_GPT_taxi_company_charges_l1875_187562


namespace NUMINAMATH_GPT_solve_linear_system_l1875_187509

theorem solve_linear_system :
  ∃ x y : ℚ, 7 * x = -10 - 3 * y ∧ 4 * x = 5 * y - 32 ∧ 
  x = -219 / 88 ∧ y = 97 / 22 :=
by
  sorry

end NUMINAMATH_GPT_solve_linear_system_l1875_187509


namespace NUMINAMATH_GPT_min_omega_l1875_187545

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + 1)
noncomputable def g (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * (x - 1) + 1)

def condition1 (ω : ℝ) : Prop := ω > 0
def condition2 (ω : ℝ) (x : ℝ) : Prop := g ω x = Real.sin (ω * x - ω + 1)
def condition3 (ω : ℝ) (k : ℤ) : Prop := ∃ k : ℤ, ω = 1 - k * Real.pi

theorem min_omega (ω : ℝ) (k : ℤ) (x : ℝ) : condition1 ω → condition2 ω x → condition3 ω k → ω = 1 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_min_omega_l1875_187545


namespace NUMINAMATH_GPT_right_square_pyramid_height_l1875_187543

theorem right_square_pyramid_height :
  ∀ (h x : ℝ),
    let topBaseSide := 3
    let bottomBaseSide := 6
    let lateralArea := 4 * (1/2) * (topBaseSide + bottomBaseSide) * x
    let baseAreasSum := topBaseSide^2 + bottomBaseSide^2
    lateralArea = baseAreasSum →
    x = 5/2 →
    h = 2 :=
by
  intros h x topBaseSide bottomBaseSide lateralArea baseAreasSum lateralEq baseEq
  sorry

end NUMINAMATH_GPT_right_square_pyramid_height_l1875_187543


namespace NUMINAMATH_GPT_probability_of_event_correct_l1875_187530

def within_interval (x : ℝ) : Prop :=
  0 ≤ x ∧ x ≤ Real.pi

def tan_in_range (x : ℝ) : Prop :=
  -1 ≤ Real.tan x ∧ Real.tan x ≤ Real.sqrt 3

def valid_subintervals (x : ℝ) : Prop :=
  within_interval x ∧ tan_in_range x

def interval_length (a b : ℝ) : ℝ :=
  b - a

noncomputable def probability_of_event : ℝ :=
  (interval_length 0 (Real.pi / 3) + interval_length (3 * Real.pi / 4) Real.pi) / Real.pi

theorem probability_of_event_correct :
  probability_of_event = 7 / 12 := sorry

end NUMINAMATH_GPT_probability_of_event_correct_l1875_187530


namespace NUMINAMATH_GPT_shortest_chord_line_intersect_circle_l1875_187554

-- Define the equation of the circle C
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 3 = 0

-- Define the fixed point
def fixed_point : ℝ × ℝ := (0, 1)

-- Define the center of the circle
def center : ℝ × ℝ := (1, 0)

-- Define the equation of the line l
def line_eq (x y : ℝ) : Prop := x - y + 1 = 0

-- The theorem that needs to be proven
theorem shortest_chord_line_intersect_circle :
  ∃ k : ℝ, ∀ x y : ℝ, (circle_eq x y ∧ y = k * x + 1) ↔ line_eq x y :=
by
  sorry

end NUMINAMATH_GPT_shortest_chord_line_intersect_circle_l1875_187554


namespace NUMINAMATH_GPT_probability_is_1_div_28_l1875_187531

noncomputable def probability_valid_combinations : ℚ :=
  let total_combinations := Nat.choose 8 3
  let valid_combinations := 2
  valid_combinations / total_combinations

theorem probability_is_1_div_28 :
  probability_valid_combinations = 1 / 28 := by
  sorry

end NUMINAMATH_GPT_probability_is_1_div_28_l1875_187531


namespace NUMINAMATH_GPT_ana_additional_payment_l1875_187570

theorem ana_additional_payment (A B L : ℝ) (h₁ : A < B) (h₂ : A < L) : 
  (A + (B + L - 2 * A) / 3 = ((A + B + L) / 3)) :=
by
  sorry

end NUMINAMATH_GPT_ana_additional_payment_l1875_187570


namespace NUMINAMATH_GPT_book_club_meeting_days_l1875_187592

theorem book_club_meeting_days :
  Nat.lcm (Nat.lcm 5 6) (Nat.lcm 8 (Nat.lcm 9 10)) = 360 := 
by sorry

end NUMINAMATH_GPT_book_club_meeting_days_l1875_187592


namespace NUMINAMATH_GPT_simple_interest_time_period_l1875_187581

theorem simple_interest_time_period 
  (P : ℝ) (R : ℝ := 4) (T : ℝ) (SI : ℝ := (2 / 5) * P) :
  SI = P * R * T / 100 → T = 10 :=
by {
  sorry
}

end NUMINAMATH_GPT_simple_interest_time_period_l1875_187581


namespace NUMINAMATH_GPT_solution_set_l1875_187557

noncomputable def f : ℝ → ℝ := sorry

axiom f_cond1 : ∀ x : ℝ, f x + deriv f x > 1
axiom f_cond2 : f 0 = 4

theorem solution_set (x : ℝ) : e^x * f x > e^x + 3 ↔ x > 0 :=
by sorry

end NUMINAMATH_GPT_solution_set_l1875_187557


namespace NUMINAMATH_GPT_enclosed_area_of_curve_l1875_187524

/-
  The closed curve in the figure is made up of 9 congruent circular arcs each of length \(\frac{\pi}{2}\),
  where each of the centers of the corresponding circles is among the vertices of a regular hexagon of side 3.
  We want to prove that the area enclosed by the curve is \(\frac{27\sqrt{3}}{2} + \frac{9\pi}{8}\).
-/

theorem enclosed_area_of_curve :
  let side_length := 3
  let arc_length := π / 2
  let num_arcs := 9
  let hexagon_area := (3 * Real.sqrt 3 / 2) * side_length^2
  let radius := 1 / 2
  let sector_area := (π * radius^2) / 4
  let total_sector_area := num_arcs * sector_area
  let enclosed_area := hexagon_area + total_sector_area
  enclosed_area = (27 * Real.sqrt 3) / 2 + (9 * π) / 8 :=
by
  sorry

end NUMINAMATH_GPT_enclosed_area_of_curve_l1875_187524


namespace NUMINAMATH_GPT_train_cross_post_time_proof_l1875_187546

noncomputable def train_cross_post_time (speed_kmh : ℝ) (length_m : ℝ) : ℝ :=
  let speed_ms := speed_kmh * 1000 / 3600
  length_m / speed_ms

theorem train_cross_post_time_proof : train_cross_post_time 40 190.0152 = 17.1 := by
  sorry

end NUMINAMATH_GPT_train_cross_post_time_proof_l1875_187546


namespace NUMINAMATH_GPT_first_division_percentage_l1875_187506

theorem first_division_percentage (total_students : ℕ) (second_division_percentage just_passed_students : ℕ) 
  (h1 : total_students = 300) (h2 : second_division_percentage = 54) (h3 : just_passed_students = 60) : 
  (100 - second_division_percentage - ((just_passed_students * 100) / total_students)) = 26 :=
by
  sorry

end NUMINAMATH_GPT_first_division_percentage_l1875_187506


namespace NUMINAMATH_GPT_negation_of_proposition_l1875_187541

theorem negation_of_proposition (p : Prop) : 
  (∀ x : ℝ, x ≥ 0 → x^2 - x + 1 ≥ 0) ↔ ¬(∃ x : ℝ, x ≥ 0 ∧ x^2 - x + 1 < 0) :=
by sorry

end NUMINAMATH_GPT_negation_of_proposition_l1875_187541


namespace NUMINAMATH_GPT_sin_ineq_l1875_187512

open Real

theorem sin_ineq (n : ℕ) (h : n > 0) : sin (π / (4 * n)) ≥ (sqrt 2) / (2 * n) :=
sorry

end NUMINAMATH_GPT_sin_ineq_l1875_187512


namespace NUMINAMATH_GPT_sufficient_not_necessary_condition_parallel_lines_l1875_187505

theorem sufficient_not_necessary_condition_parallel_lines :
  ∀ (a : ℝ), (a = 1/2 → (∀ x y : ℝ, x + 2*a*y = 1 ↔ (x - x + 1) ≠ 0) 
            ∧ ((∃ a', a' ≠ 1/2 ∧ (∀ x y : ℝ, x + 2*a'*y = 1 ↔ (x - x + 1) ≠ 0)) → (a ≠ 1/2))) :=
by
  intro a
  sorry

end NUMINAMATH_GPT_sufficient_not_necessary_condition_parallel_lines_l1875_187505


namespace NUMINAMATH_GPT_smallest_x_mod_7_one_sq_l1875_187500

theorem smallest_x_mod_7_one_sq (x : ℕ) (h : 1 < x) (hx : (x * x) % 7 = 1) : x = 6 :=
  sorry

end NUMINAMATH_GPT_smallest_x_mod_7_one_sq_l1875_187500


namespace NUMINAMATH_GPT_find_other_subject_given_conditions_l1875_187542

theorem find_other_subject_given_conditions :
  ∀ (P C M : ℕ),
  P = 65 →
  (P + C + M) / 3 = 85 →
  (P + M) / 2 = 90 →
  ∃ (S : ℕ), (P + S) / 2 = 70 ∧ S = C :=
by
  sorry

end NUMINAMATH_GPT_find_other_subject_given_conditions_l1875_187542


namespace NUMINAMATH_GPT_mrs_hilt_initial_money_l1875_187566

def initial_amount (pencil_cost candy_cost left_money : ℕ) := 
  pencil_cost + candy_cost + left_money

theorem mrs_hilt_initial_money :
  initial_amount 20 5 18 = 43 :=
by
  -- initial_amount 20 5 18 
  -- = 20 + 5 + 18
  -- = 25 + 18 
  -- = 43
  sorry

end NUMINAMATH_GPT_mrs_hilt_initial_money_l1875_187566


namespace NUMINAMATH_GPT_pistachio_shells_percentage_l1875_187518

theorem pistachio_shells_percentage (total_pistachios : ℕ) (opened_shelled_pistachios : ℕ) (P : ℝ) :
  total_pistachios = 80 →
  opened_shelled_pistachios = 57 →
  (0.75 : ℝ) * (P / 100) * (total_pistachios : ℝ) = (opened_shelled_pistachios : ℝ) →
  P = 95 :=
by
  intros h_total h_opened h_equation
  sorry

end NUMINAMATH_GPT_pistachio_shells_percentage_l1875_187518


namespace NUMINAMATH_GPT_sum_of_roots_l1875_187567

open Polynomial

noncomputable def f (a b : ℝ) : Polynomial ℝ := Polynomial.C b + Polynomial.C a * X + X^2
noncomputable def g (c d : ℝ) : Polynomial ℝ := Polynomial.C d + Polynomial.C c * X + X^2

theorem sum_of_roots (a b c d : ℝ)
  (h1 : eval 1 (f a b) = eval 2 (g c d))
  (h2 : eval 1 (g c d) = eval 2 (f a b))
  (hf_roots : ∃ r1 r2 : ℝ, (f a b).roots = {r1, r2})
  (hg_roots : ∃ s1 s2 : ℝ, (g c d).roots = {s1, s2}) :
  (-(a + c) = 6) :=
sorry

end NUMINAMATH_GPT_sum_of_roots_l1875_187567


namespace NUMINAMATH_GPT_triangle_area_eq_l1875_187533

/-- In a triangle ABC, given that A = arccos(7/8), BC = a, and the altitude from vertex A 
     is equal to the sum of the other two altitudes, show that the area of triangle ABC 
     is (a^2 * sqrt(15)) / 4. -/
theorem triangle_area_eq (a : ℝ) (angle_A : ℝ) (h_angle : angle_A = Real.arccos (7/8))
    (BC : ℝ) (h_BC : BC = a) (H : ∀ (AC AB altitude_A altitude_C altitude_B : ℝ),
    AC = X → AB = Y → 
    altitude_A = (altitude_C + altitude_B) → 
    ∃ (S : ℝ), 
    S = (1/2) * X * Y * Real.sin angle_A ∧ 
    altitude_A = (2 * S / X) + (2 * S / Y) 
    → (X * Y) = 4 * (a^2) 
    → S = ((a^2 * Real.sqrt 15) / 4)) :
S = (a^2 * Real.sqrt 15) / 4 := sorry

end NUMINAMATH_GPT_triangle_area_eq_l1875_187533


namespace NUMINAMATH_GPT_age_ratio_in_4_years_l1875_187515

variable {p k x : ℕ}

theorem age_ratio_in_4_years (h₁ : p - 8 = 2 * (k - 8)) (h₂ : p - 14 = 3 * (k - 14)) : x = 4 :=
by
  sorry

end NUMINAMATH_GPT_age_ratio_in_4_years_l1875_187515


namespace NUMINAMATH_GPT_find_n_for_geom_sum_l1875_187503

-- Define the first term and the common ratio
def first_term := 1
def common_ratio := 1 / 2

-- Define the sum function of the first n terms of the geometric sequence
def geom_sum (n : ℕ) : ℚ := first_term * (1 - (common_ratio)^n) / (1 - common_ratio)

-- Define the target sum
def target_sum := 31 / 16

-- State the theorem to prove
theorem find_n_for_geom_sum : ∃ n : ℕ, geom_sum n = target_sum := 
    by
    sorry

end NUMINAMATH_GPT_find_n_for_geom_sum_l1875_187503


namespace NUMINAMATH_GPT_find_g_minus_6_l1875_187501

-- Define the function g on integers
def g : ℤ → ℤ := sorry

-- Conditions given in the problem
axiom cond1 : g 1 - 1 > 0
axiom cond2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x
axiom cond3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3

-- The proof we want to make (assertion)
theorem find_g_minus_6 : g (-6) = 723 := 
sorry

end NUMINAMATH_GPT_find_g_minus_6_l1875_187501


namespace NUMINAMATH_GPT_find_number_l1875_187579

theorem find_number (x : ℕ) (hx : (x / 100) * 100 = 20) : x = 20 :=
sorry

end NUMINAMATH_GPT_find_number_l1875_187579


namespace NUMINAMATH_GPT_find_a_l1875_187596

theorem find_a (a : ℤ) (A B : Set ℤ) (hA : A = {0, 1}) (hB : B = {-1, 0, a + 3}) (h : A ⊆ B) : a = -2 := by
  sorry

end NUMINAMATH_GPT_find_a_l1875_187596


namespace NUMINAMATH_GPT_system_of_equations_solution_l1875_187517

theorem system_of_equations_solution (x y z u v : ℤ) 
  (h1 : x + y + z + u = 5)
  (h2 : y + z + u + v = 1)
  (h3 : z + u + v + x = 2)
  (h4 : u + v + x + y = 0)
  (h5 : v + x + y + z = 4) :
  v = -2 ∧ x = 2 ∧ y = 1 ∧ z = 3 ∧ u = -1 := 
by 
  sorry

end NUMINAMATH_GPT_system_of_equations_solution_l1875_187517


namespace NUMINAMATH_GPT_desk_chair_production_l1875_187555

theorem desk_chair_production (x : ℝ) (h₁ : x > 0) (h₂ : 540 / x - 540 / (x + 2) = 3) : 
  ∃ x, 540 / x - 540 / (x + 2) = 3 := 
by
  sorry

end NUMINAMATH_GPT_desk_chair_production_l1875_187555


namespace NUMINAMATH_GPT_role_of_scatter_plot_correct_l1875_187577

-- Definitions for problem context
def role_of_scatter_plot (role : String) : Prop :=
  role = "Roughly judging whether variables are linearly related"

-- Problem and conditions
theorem role_of_scatter_plot_correct :
  role_of_scatter_plot "Roughly judging whether variables are linearly related" :=
by 
  sorry

end NUMINAMATH_GPT_role_of_scatter_plot_correct_l1875_187577


namespace NUMINAMATH_GPT_non_upgraded_sensor_ratio_l1875_187547

theorem non_upgraded_sensor_ratio 
  (N U S : ℕ) 
  (units : ℕ := 24) 
  (fraction_upgraded : ℚ := 1 / 7) 
  (fraction_non_upgraded : ℚ := 6 / 7)
  (h1 : U / S = fraction_upgraded)
  (h2 : units * N = (fraction_non_upgraded * S)) : 
  N / U = 1 / 4 := 
by 
  sorry

end NUMINAMATH_GPT_non_upgraded_sensor_ratio_l1875_187547


namespace NUMINAMATH_GPT_bus_speed_l1875_187544

theorem bus_speed (t : ℝ) (d : ℝ) (h : t = 42 / 60) (d_eq : d = 35) : d / t = 50 :=
by
  -- Assume
  sorry

end NUMINAMATH_GPT_bus_speed_l1875_187544


namespace NUMINAMATH_GPT_line_through_point_equal_intercepts_locus_equidistant_lines_l1875_187586

theorem line_through_point_equal_intercepts (x y : ℝ) (hx : x = 1) (hy : y = 3) :
  (∃ k : ℝ, y = k * x ∧ k = 3) ∨ (∃ a : ℝ, x + y = a ∧ a = 4) :=
sorry

theorem locus_equidistant_lines (x y : ℝ) :
  ∀ (a b : ℝ), (2 * x + 3 * y - a = 0) ∧ (4 * x + 6 * y + b = 0) →
  ∀ b : ℝ, |b + 10| = |b - 8| → b = -9 → 
  4 * x + 6 * y - 9 = 0 :=
sorry

end NUMINAMATH_GPT_line_through_point_equal_intercepts_locus_equidistant_lines_l1875_187586


namespace NUMINAMATH_GPT_hexagon_interior_angles_l1875_187580

theorem hexagon_interior_angles
  (A B C D E F : ℝ)
  (hA : A = 90)
  (hB : B = 120)
  (hCD : C = D)
  (hE : E = 2 * C + 20)
  (hF : F = 60)
  (hsum : A + B + C + D + E + F = 720) :
  D = 107.5 := 
by
  -- formal proof required here
  sorry

end NUMINAMATH_GPT_hexagon_interior_angles_l1875_187580


namespace NUMINAMATH_GPT_apple_price_theorem_l1875_187578

-- Given conditions
def apple_counts : List Nat := [20, 40, 60, 80, 100, 120, 140]

-- Helper function to calculate revenue for a given apple count.
def revenue (apples : Nat) (price_per_batch : Nat) (price_per_leftover : Nat) (batch_size : Nat) : Nat :=
  (apples / batch_size) * price_per_batch + (apples % batch_size) * price_per_leftover

-- Theorem stating that the price per 7 apples is 1 cent and 3 cents per leftover apple ensures equal revenue.
theorem apple_price_theorem : 
  ∀ seller ∈ apple_counts, 
  revenue seller 1 3 7 = 20 :=
by
  intros seller h_seller
  -- Proof will follow here
  sorry

end NUMINAMATH_GPT_apple_price_theorem_l1875_187578


namespace NUMINAMATH_GPT_sum_of_coefficients_l1875_187571

/-- If (2x - 1)^4 = a₄x^4 + a₃x^3 + a₂x^2 + a₁x + a₀, then the sum of the coefficients a₀ + a₁ + a₂ + a₃ + a₄ is 1. -/
theorem sum_of_coefficients :
  ∃ a₄ a₃ a₂ a₁ a₀ : ℝ, (2 * x - 1) ^ 4 = a₄ * x ^ 4 + a₃ * x ^ 3 + a₂ * x ^ 2 + a₁ * x + a₀ → 
  a₀ + a₁ + a₂ + a₃ + a₄ = 1 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_l1875_187571


namespace NUMINAMATH_GPT_backyard_area_proof_l1875_187574

-- Condition: Walking the length of 40 times covers 1000 meters
def length_times_40_eq_1000 (L: ℝ) : Prop := 40 * L = 1000

-- Condition: Walking the perimeter 8 times covers 1000 meters
def perimeter_times_8_eq_1000 (P: ℝ) : Prop := 8 * P = 1000

-- Given the conditions, we need to find the Length and Width of the backyard
def is_backyard_dimensions (L W: ℝ) : Prop := 
  length_times_40_eq_1000 L ∧ 
  perimeter_times_8_eq_1000 (2 * (L + W))

-- We need to calculate the area
def backyard_area (L W: ℝ) : ℝ := L * W

-- The theorem to prove
theorem backyard_area_proof (L W: ℝ) 
  (h1: length_times_40_eq_1000 L) 
  (h2: perimeter_times_8_eq_1000 (2 * (L + W))) :
  backyard_area L W = 937.5 := 
  by 
    sorry

end NUMINAMATH_GPT_backyard_area_proof_l1875_187574


namespace NUMINAMATH_GPT_number_of_divisors_of_n_l1875_187513

theorem number_of_divisors_of_n :
  let n : ℕ := (7^3) * (11^2) * (13^4)
  ∃ d : ℕ, d = 60 ∧ ∀ m : ℕ, m ∣ n ↔ ∃ l₁ l₂ l₃ : ℕ, l₁ ≤ 3 ∧ l₂ ≤ 2 ∧ l₃ ≤ 4 ∧ m = 7^l₁ * 11^l₂ * 13^l₃ := 
by
  sorry

end NUMINAMATH_GPT_number_of_divisors_of_n_l1875_187513


namespace NUMINAMATH_GPT_sum_of_cubes_of_consecutive_numbers_divisible_by_9_l1875_187504

theorem sum_of_cubes_of_consecutive_numbers_divisible_by_9 (a : ℕ) (h : a > 1) : 
  9 ∣ ((a - 1)^3 + a^3 + (a + 1)^3) := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_cubes_of_consecutive_numbers_divisible_by_9_l1875_187504


namespace NUMINAMATH_GPT_units_digit_pow_prod_l1875_187582

theorem units_digit_pow_prod : 
  ((2 ^ 2023) * (5 ^ 2024) * (11 ^ 2025)) % 10 = 0 :=
by
  sorry

end NUMINAMATH_GPT_units_digit_pow_prod_l1875_187582


namespace NUMINAMATH_GPT_mean_of_smallest_and_largest_is_12_l1875_187514

-- Definition of the condition: the mean of five consecutive even numbers is 12.
def mean_of_five_consecutive_even_numbers_is_12 (n : ℤ) : Prop :=
  ((n - 4) + (n - 2) + n + (n + 2) + (n + 4)) / 5 = 12

-- Theorem stating that the mean of the smallest and largest of these numbers is 12.
theorem mean_of_smallest_and_largest_is_12 (n : ℤ) 
  (h : mean_of_five_consecutive_even_numbers_is_12 n) : 
  (8 + (16 : ℤ)) / (2 : ℤ) = 12 := 
by
  sorry

end NUMINAMATH_GPT_mean_of_smallest_and_largest_is_12_l1875_187514


namespace NUMINAMATH_GPT_average_percent_score_is_65_point_25_l1875_187550

theorem average_percent_score_is_65_point_25 :
  let percent_score : List (ℕ × ℕ) := [(95, 10), (85, 20), (75, 40), (65, 50), (55, 60), (45, 15), (35, 5)]
  let total_students : ℕ := 200
  let total_score : ℕ := percent_score.foldl (fun acc p => acc + p.1 * p.2) 0
  (total_score : ℚ) / (total_students : ℚ) = 65.25 := by
{
  sorry
}

end NUMINAMATH_GPT_average_percent_score_is_65_point_25_l1875_187550


namespace NUMINAMATH_GPT_perimeter_of_shaded_area_l1875_187568

theorem perimeter_of_shaded_area (AB AD : ℝ) (h1 : AB = 14) (h2 : AD = 12) : 
  2 * AB + 2 * AD = 52 := 
by
  sorry

end NUMINAMATH_GPT_perimeter_of_shaded_area_l1875_187568


namespace NUMINAMATH_GPT_number_of_shirts_in_first_batch_minimum_selling_price_per_shirt_l1875_187588

-- Define the conditions
def first_batch_cost : ℝ := 13200
def second_batch_cost : ℝ := 28800
def unit_price_difference : ℝ := 10
def discount_rate : ℝ := 0.8
def profit_margin : ℝ := 1.25
def last_batch_count : ℕ := 50

-- Define the theorem for the first part
theorem number_of_shirts_in_first_batch (x : ℕ) (h₁ : first_batch_cost / x + unit_price_difference = second_batch_cost / (2 * x)) : x = 120 :=
sorry

-- Define the theorem for the second part
theorem minimum_selling_price_per_shirt (x : ℕ) (y : ℝ)
  (h₁ : first_batch_cost / x + unit_price_difference = second_batch_cost / (2 * x))
  (h₂ : x = 120)
  (h₃ : (3 * x - last_batch_count) * y + last_batch_count * discount_rate * y ≥ (first_batch_cost + second_batch_cost) * profit_margin) : y ≥ 150 :=
sorry

end NUMINAMATH_GPT_number_of_shirts_in_first_batch_minimum_selling_price_per_shirt_l1875_187588


namespace NUMINAMATH_GPT_danny_reaches_steve_house_in_31_minutes_l1875_187583

theorem danny_reaches_steve_house_in_31_minutes:
  ∃ (t : ℝ), 2 * t - t = 15.5 * 2 ∧ t = 31 := sorry

end NUMINAMATH_GPT_danny_reaches_steve_house_in_31_minutes_l1875_187583


namespace NUMINAMATH_GPT_P_gt_Q_l1875_187536

theorem P_gt_Q (a : ℝ) : 
  let P := a^2 + 2*a
  let Q := 3*a - 1
  P > Q :=
by
  sorry

end NUMINAMATH_GPT_P_gt_Q_l1875_187536


namespace NUMINAMATH_GPT_barbara_shopping_l1875_187560

theorem barbara_shopping :
  let total_paid := 56
  let tuna_cost := 5 * 2
  let water_cost := 4 * 1.5
  let other_goods_cost := total_paid - tuna_cost - water_cost
  other_goods_cost = 40 :=
by
  sorry

end NUMINAMATH_GPT_barbara_shopping_l1875_187560


namespace NUMINAMATH_GPT_explicit_form_l1875_187540

-- Define the functional equation
def f (x : ℝ) : ℝ := sorry

-- Define the condition that f(x) satisfies
axiom functional_equation (x : ℝ) (h : x ≠ 0) : f x = 2 * f (1 / x) + 3 * x

-- State the theorem that we need to prove
theorem explicit_form (x : ℝ) (h : x ≠ 0) : f x = -x - (2 / x) :=
by
  sorry

end NUMINAMATH_GPT_explicit_form_l1875_187540


namespace NUMINAMATH_GPT_brady_earns_181_l1875_187519

def bradyEarnings (basic_count : ℕ) (gourmet_count : ℕ) (total_cards : ℕ) : ℕ :=
  let basic_earnings := basic_count * 70
  let gourmet_earnings := gourmet_count * 90
  let total_earnings := basic_earnings + gourmet_earnings
  let total_bonus := (total_cards / 100) * 10 + ((total_cards / 100) - 1) * 5
  total_earnings + total_bonus

theorem brady_earns_181 :
  bradyEarnings 120 80 200 = 181 :=
by 
  sorry

end NUMINAMATH_GPT_brady_earns_181_l1875_187519


namespace NUMINAMATH_GPT_sample_size_l1875_187507

variable (num_classes : ℕ) (papers_per_class : ℕ)

theorem sample_size (h_classes : num_classes = 8) (h_papers : papers_per_class = 12) : 
  num_classes * papers_per_class = 96 := 
by 
  sorry

end NUMINAMATH_GPT_sample_size_l1875_187507


namespace NUMINAMATH_GPT_number_of_k_solutions_l1875_187529

theorem number_of_k_solutions :
  ∃ (n : ℕ), n = 1006 ∧
  (∀ k, (∃ a b : ℕ+, (a ≠ b) ∧ (k * (a + b) = 2013 * Nat.lcm a b)) ↔ k ≤ n ∧ 0 < k) :=
by
  sorry

end NUMINAMATH_GPT_number_of_k_solutions_l1875_187529


namespace NUMINAMATH_GPT_sale_in_fifth_month_l1875_187584

def sales_month_1 := 6635
def sales_month_2 := 6927
def sales_month_3 := 6855
def sales_month_4 := 7230
def sales_month_6 := 4791
def target_average := 6500
def number_of_months := 6

def total_sales := sales_month_1 + sales_month_2 + sales_month_3 + sales_month_4 + sales_month_6

theorem sale_in_fifth_month :
  (target_average * number_of_months) - total_sales = 6562 :=
by
  sorry

end NUMINAMATH_GPT_sale_in_fifth_month_l1875_187584


namespace NUMINAMATH_GPT_amanda_pay_if_not_finished_l1875_187525

-- Define Amanda's hourly rate and daily work hours.
def amanda_hourly_rate : ℝ := 50
def amanda_daily_hours : ℝ := 10

-- Define the percentage of pay Jose will withhold.
def withholding_percentage : ℝ := 0.20

-- Define Amanda's total pay if she finishes the sales report.
def amanda_total_pay : ℝ := amanda_hourly_rate * amanda_daily_hours

-- Define the amount withheld if she does not finish the sales report.
def withheld_amount : ℝ := amanda_total_pay * withholding_percentage

-- Define the amount Amanda will receive if she does not finish the sales report.
def amanda_final_pay_not_finished : ℝ := amanda_total_pay - withheld_amount

-- The theorem to prove:
theorem amanda_pay_if_not_finished : amanda_final_pay_not_finished = 400 := by
  sorry

end NUMINAMATH_GPT_amanda_pay_if_not_finished_l1875_187525


namespace NUMINAMATH_GPT_Billy_weight_l1875_187599

variables (Billy Brad Carl Dave Edgar : ℝ)

-- Conditions
def conditions :=
  Carl = 145 ∧
  Dave = Carl + 8 ∧
  Brad = Dave / 2 ∧
  Billy = Brad + 9 ∧
  Edgar = 3 * Dave ∧
  Edgar = Billy + 20

-- The statement to prove
theorem Billy_weight (Billy Brad Carl Dave Edgar : ℝ) (h : conditions Billy Brad Carl Dave Edgar) : Billy = 85.5 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_Billy_weight_l1875_187599


namespace NUMINAMATH_GPT_coloring_triangles_l1875_187590

theorem coloring_triangles (n : ℕ) (k : ℕ) (h_n : n = 18) (h_k : k = 6) :
  (Nat.choose n k) = 18564 :=
by
  rw [h_n, h_k]
  sorry

end NUMINAMATH_GPT_coloring_triangles_l1875_187590


namespace NUMINAMATH_GPT_age_solution_l1875_187564

theorem age_solution (M S : ℕ) (h1 : M = S + 16) (h2 : M + 2 = 2 * (S + 2)) : S = 14 :=
by sorry

end NUMINAMATH_GPT_age_solution_l1875_187564


namespace NUMINAMATH_GPT_sum_of_f_l1875_187537

noncomputable def f (x : ℝ) : ℝ := 1 / (2^x + Real.sqrt 2)

theorem sum_of_f :
  f (-5) + f (-4) + f (-3) + f (-2) + f (-1) + f 0 + f 1 + f 2 + f 3 + f 4 + f 5 + f 6 = 3 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_f_l1875_187537


namespace NUMINAMATH_GPT_count_valid_n_l1875_187523

theorem count_valid_n (n : ℕ) (h₁ : (n % 2015) ≠ 0) :
  (n^3 + 3^n) % 5 = 0 :=
by
  sorry

end NUMINAMATH_GPT_count_valid_n_l1875_187523


namespace NUMINAMATH_GPT_product_of_constants_l1875_187565

theorem product_of_constants :
  ∃ M₁ M₂ : ℝ, 
    (∀ x : ℝ, x ≠ 2 ∧ x ≠ 3 → (45 * x - 82) / (x^2 - 5 * x + 6) = M₁ / (x - 2) + M₂ / (x - 3)) ∧ 
    M₁ * M₂ = -424 :=
by
  sorry

end NUMINAMATH_GPT_product_of_constants_l1875_187565


namespace NUMINAMATH_GPT_largest_value_is_E_l1875_187538

theorem largest_value_is_E :
  let A := 3 + 1 + 2 + 9
  let B := 3 * 1 + 2 + 9
  let C := 3 + 1 * 2 + 9
  let D := 3 + 1 + 2 * 9
  let E := 3 * 1 * 2 * 9
  E > A ∧ E > B ∧ E > C ∧ E > D := 
by
  let A := 3 + 1 + 2 + 9
  let B := 3 * 1 + 2 + 9
  let C := 3 + 1 * 2 + 9
  let D := 3 + 1 + 2 * 9
  let E := 3 * 1 * 2 * 9
  sorry

end NUMINAMATH_GPT_largest_value_is_E_l1875_187538


namespace NUMINAMATH_GPT_spadesuit_evaluation_l1875_187521

def spadesuit (x y : ℝ) : ℝ := (x + y) * (x - y)

theorem spadesuit_evaluation : spadesuit 3 (spadesuit 4 5) = -72 := by
  sorry

end NUMINAMATH_GPT_spadesuit_evaluation_l1875_187521


namespace NUMINAMATH_GPT_ellipse_k_values_l1875_187597

theorem ellipse_k_values (k : ℝ) :
  (∃ a b : ℝ, a = (k + 8) ∧ b = 9 ∧ 
  (b > a → (a * (1 - (1 / 2) ^ 2) = b - a) ∧ k = 4) ∧ 
  (a > b → (b * (1 - (1 / 2) ^ 2) = a - b) ∧ k = -5/4)) :=
sorry

end NUMINAMATH_GPT_ellipse_k_values_l1875_187597


namespace NUMINAMATH_GPT_find_certain_number_l1875_187534

theorem find_certain_number (d q r : ℕ) (HD : d = 37) (HQ : q = 23) (HR : r = 16) :
    ∃ n : ℕ, n = d * q + r ∧ n = 867 := by
  sorry

end NUMINAMATH_GPT_find_certain_number_l1875_187534


namespace NUMINAMATH_GPT_value_of_hash_l1875_187575

def hash (a b c d : ℝ) : ℝ := b^2 - 4 * a * c * d

theorem value_of_hash : hash 2 3 2 1 = -7 := by
  sorry

end NUMINAMATH_GPT_value_of_hash_l1875_187575


namespace NUMINAMATH_GPT_simplify_fraction_multiplication_l1875_187551

theorem simplify_fraction_multiplication :
  (15/35) * (28/45) * (75/28) = 5/7 :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_multiplication_l1875_187551


namespace NUMINAMATH_GPT_cubic_roots_identity_l1875_187569

noncomputable def roots_of_cubic (a b c : ℝ) : Prop :=
  (5 * a^3 - 2019 * a + 4029 = 0) ∧ 
  (5 * b^3 - 2019 * b + 4029 = 0) ∧ 
  (5 * c^3 - 2019 * c + 4029 = 0)

theorem cubic_roots_identity (a b c : ℝ) (h_roots : roots_of_cubic a b c) : 
  (a + b)^3 + (b + c)^3 + (c + a)^3 = 12087 / 5 :=
by 
  -- proof steps
  sorry

end NUMINAMATH_GPT_cubic_roots_identity_l1875_187569


namespace NUMINAMATH_GPT_CA_inter_B_l1875_187532

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {1, 3, 5}
def B : Set ℕ := {2, 5, 7}

theorem CA_inter_B :
  (U \ A) ∩ B = {2, 7} := by
  sorry

end NUMINAMATH_GPT_CA_inter_B_l1875_187532


namespace NUMINAMATH_GPT_bag_contains_twenty_cookies_l1875_187511

noncomputable def cookies_in_bag 
  (total_calories : ℕ) 
  (calories_per_cookie : ℕ)
  (bags_in_box : ℕ)
  : ℕ :=
  total_calories / (calories_per_cookie * bags_in_box)

theorem bag_contains_twenty_cookies 
  (H1 : total_calories = 1600) 
  (H2 : calories_per_cookie = 20) 
  (H3 : bags_in_box = 4)
  : cookies_in_bag total_calories calories_per_cookie bags_in_box = 20 := 
by
  have h1 : total_calories = 1600 := H1
  have h2 : calories_per_cookie = 20 := H2
  have h3 : bags_in_box = 4 := H3
  sorry

end NUMINAMATH_GPT_bag_contains_twenty_cookies_l1875_187511


namespace NUMINAMATH_GPT_find_y_l1875_187502

theorem find_y (y : ℝ) (h : 9 * y^3 = y * 81) : y = 3 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_find_y_l1875_187502


namespace NUMINAMATH_GPT_coefficient_of_c_l1875_187510

theorem coefficient_of_c (f c : ℝ) (h₁ : f = (9/5) * c + 32)
                         (h₂ : f + 25 = (9/5) * (c + 13.88888888888889) + 32) :
  (5/9) = (9/5) := sorry

end NUMINAMATH_GPT_coefficient_of_c_l1875_187510


namespace NUMINAMATH_GPT_proof_l1875_187516

noncomputable def problem_statement : Prop :=
  ( ( (Real.sqrt 1.21 * Real.sqrt 1.44) / (Real.sqrt 0.81 * Real.sqrt 0.64)
    + (Real.sqrt 1.0 * Real.sqrt 3.24) / (Real.sqrt 0.49 * Real.sqrt 2.25) ) ^ 3 
  = 44.6877470366 )

theorem proof : problem_statement := 
  by
  sorry

end NUMINAMATH_GPT_proof_l1875_187516


namespace NUMINAMATH_GPT_solve_years_later_twice_age_l1875_187527

-- Define the variables and the given conditions
def man_age (S: ℕ) := S + 25
def years_later_twice_age (S M: ℕ) (Y: ℕ) := (M + Y = 2 * (S + Y))

-- Given conditions
def present_age_son := 23
def present_age_man := man_age present_age_son

theorem solve_years_later_twice_age :
  ∃ Y, years_later_twice_age present_age_son present_age_man Y ∧ Y = 2 := by
  sorry

end NUMINAMATH_GPT_solve_years_later_twice_age_l1875_187527


namespace NUMINAMATH_GPT_complex_exponential_sum_l1875_187553

theorem complex_exponential_sum (γ δ : ℝ) 
  (h : Complex.exp (Complex.I * γ) + Complex.exp (Complex.I * δ) = -1 / 2 + 5 / 4 * Complex.I) :
  Complex.exp (-Complex.I * γ) + Complex.exp (-Complex.I * δ) = -1 / 2 - 5 / 4 * Complex.I :=
by
  sorry

end NUMINAMATH_GPT_complex_exponential_sum_l1875_187553


namespace NUMINAMATH_GPT_inequality_holds_for_all_real_numbers_l1875_187556

theorem inequality_holds_for_all_real_numbers (x : ℝ) : 3 * x - 5 ≤ 12 - 2 * x + x^2 :=
by sorry

end NUMINAMATH_GPT_inequality_holds_for_all_real_numbers_l1875_187556


namespace NUMINAMATH_GPT_Rob_has_three_dimes_l1875_187585

theorem Rob_has_three_dimes (quarters dimes nickels pennies : ℕ) 
                            (val_quarters val_nickels val_pennies : ℚ)
                            (total_amount : ℚ) :
  quarters = 7 →
  nickels = 5 →
  pennies = 12 →
  val_quarters = 0.25 →
  val_nickels = 0.05 →
  val_pennies = 0.01 →
  total_amount = 2.42 →
  (7 * 0.25 + 5 * 0.05 + 12 * 0.01 + dimes * 0.10 = total_amount) →
  dimes = 3 :=
by sorry

end NUMINAMATH_GPT_Rob_has_three_dimes_l1875_187585


namespace NUMINAMATH_GPT_James_will_take_7_weeks_l1875_187561

def pages_per_hour : ℕ := 5
def hours_per_day : ℕ := 4 - 1
def pages_per_day : ℕ := hours_per_day * pages_per_hour
def total_pages : ℕ := 735
def days_to_finish : ℕ := total_pages / pages_per_day
def weeks_to_finish : ℕ := days_to_finish / 7

theorem James_will_take_7_weeks :
  weeks_to_finish = 7 :=
by
  -- You can add the necessary proof steps here
  sorry

end NUMINAMATH_GPT_James_will_take_7_weeks_l1875_187561


namespace NUMINAMATH_GPT_find_x_l1875_187539

theorem find_x (x : ℝ) (h_pos : 0 < x) (h_eq : x * ⌊x⌋ = 48) : x = 8 :=
sorry

end NUMINAMATH_GPT_find_x_l1875_187539


namespace NUMINAMATH_GPT_second_bounce_distance_correct_l1875_187563

noncomputable def second_bounce_distance (R v g : ℝ) : ℝ := 2 * R - (2 * v / 3) * (Real.sqrt (R / g))

theorem second_bounce_distance_correct (R v g : ℝ) (hR : R > 0) (hv : v > 0) (hg : g > 0) :
  second_bounce_distance R v g = 2 * R - (2 * v / 3) * (Real.sqrt (R / g)) := 
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_second_bounce_distance_correct_l1875_187563


namespace NUMINAMATH_GPT_dividend_percentage_l1875_187522

theorem dividend_percentage (investment_amount market_value : ℝ) (interest_rate : ℝ) 
  (h1 : investment_amount = 44) (h2 : interest_rate = 12) (h3 : market_value = 33) : 
  ((interest_rate / 100) * investment_amount / market_value) * 100 = 16 := 
by
  sorry

end NUMINAMATH_GPT_dividend_percentage_l1875_187522


namespace NUMINAMATH_GPT_ratio_expression_x_2y_l1875_187548

theorem ratio_expression_x_2y :
  ∀ (x y : ℝ), x / (2 * y) = 27 → (7 * x + 6 * y) / (x - 2 * y) = 96 / 13 :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_ratio_expression_x_2y_l1875_187548


namespace NUMINAMATH_GPT_range_of_a_l1875_187589

noncomputable def f (a x : ℝ) : ℝ :=
  Real.exp x + x^2 + (3 * a + 2) * x

theorem range_of_a (a : ℝ) :
  (∃ x ∈ Set.Ioo (-1 : ℝ) 0, ∀ y ∈ Set.Ioo (-1 : ℝ) 0, f a x ≤ f a y) →
  a ∈ Set.Ioo (-1 : ℝ) (-1 / (3 * Real.exp 1)) :=
sorry

end NUMINAMATH_GPT_range_of_a_l1875_187589


namespace NUMINAMATH_GPT_value_of_b_l1875_187520

theorem value_of_b (a b : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 35 * 45 * b) : b = 105 :=
sorry

end NUMINAMATH_GPT_value_of_b_l1875_187520


namespace NUMINAMATH_GPT_ball_hits_ground_l1875_187598

theorem ball_hits_ground 
  (y : ℝ → ℝ) 
  (height_eq : ∀ t, y t = -3 * t^2 - 6 * t + 90) :
  ∃ t : ℝ, y t = 0 ∧ t = 5.00 :=
by
  sorry

end NUMINAMATH_GPT_ball_hits_ground_l1875_187598


namespace NUMINAMATH_GPT_lindy_total_distance_l1875_187549

-- Definitions derived from the conditions
def jack_speed : ℕ := 5
def christina_speed : ℕ := 7
def lindy_speed : ℕ := 12
def initial_distance : ℕ := 360

theorem lindy_total_distance :
  lindy_speed * (initial_distance / (jack_speed + christina_speed)) = 360 := by
  sorry

end NUMINAMATH_GPT_lindy_total_distance_l1875_187549


namespace NUMINAMATH_GPT_sequence_general_term_l1875_187559

theorem sequence_general_term :
  ∀ (a : ℕ → ℝ), a 1 = 2 ^ (5 / 2) ∧ 
  (∀ n, a (n+1) = 4 * (4 * a n) ^ (1/4)) →
  ∀ n, a n = 2 ^ (10 / 3 * (1 - 1 / 4 ^ n)) :=
by
  intros a h1 h_rec
  sorry

end NUMINAMATH_GPT_sequence_general_term_l1875_187559


namespace NUMINAMATH_GPT_martha_initial_crayons_l1875_187595

theorem martha_initial_crayons : ∃ (x : ℕ), (x / 2 + 20 = 29) ∧ x = 18 :=
by
  sorry

end NUMINAMATH_GPT_martha_initial_crayons_l1875_187595


namespace NUMINAMATH_GPT_cyclist_distance_l1875_187508

theorem cyclist_distance
  (v t d : ℝ)
  (h1 : d = v * t)
  (h2 : d = (v + 1) * (t - 0.5))
  (h3 : d = (v - 1) * (t + 1)) :
  d = 6 :=
by
  sorry

end NUMINAMATH_GPT_cyclist_distance_l1875_187508


namespace NUMINAMATH_GPT_smallest_number_of_sparrows_in_each_flock_l1875_187573

theorem smallest_number_of_sparrows_in_each_flock (P : ℕ) (H : 14 * P ≥ 182) : 
  ∃ S : ℕ, S = 14 ∧ S ∣ 182 ∧ (∃ P : ℕ, S ∣ (14 * P)) := 
by 
  sorry

end NUMINAMATH_GPT_smallest_number_of_sparrows_in_each_flock_l1875_187573


namespace NUMINAMATH_GPT_angle_A_measure_triangle_area_l1875_187526

variable {a b c : ℝ} 
variable {A B C : ℝ} 
variable (triangle : a^2 = b^2 + c^2 - 2 * b * c * (Real.cos A))

theorem angle_A_measure (h : (b - c)^2 = a^2 - b * c) : A = Real.pi / 3 :=
sorry

theorem triangle_area 
  (h1 : a = 3) 
  (h2 : Real.sin C = 2 * Real.sin B) 
  (h3 : A = Real.pi / 3) 
  (hb : b = Real.sqrt 3)
  (hc : c = 2 * Real.sqrt 3) : 
  (1/2) * b * c * Real.sin A = (3 * Real.sqrt 3) / 2 :=
sorry

end NUMINAMATH_GPT_angle_A_measure_triangle_area_l1875_187526
