import Mathlib

namespace find_value_of_a_l621_62176

theorem find_value_of_a (a : ℤ) (h : ∀ x : ℚ,  x^6 - 33 * x + 20 = (x^2 - x + a) * (x^4 + b * x^3 + c * x^2 + d * x + e)) :
  a = 4 := 
by 
  sorry

end find_value_of_a_l621_62176


namespace minimum_value_of_f_l621_62123

noncomputable def f (x : ℝ) : ℝ := x + 1 / (x - 2)

theorem minimum_value_of_f :
  (∀ x : ℝ, x > 2 → f x ≥ 4) ∧ (∃ x : ℝ, x > 2 ∧ f x = 4) :=
by {
  sorry
}

end minimum_value_of_f_l621_62123


namespace age_ratio_l621_62164

theorem age_ratio (B A : ℕ) (h1 : B = 4) (h2 : A - B = 12) :
  A / B = 4 :=
by
  sorry

end age_ratio_l621_62164


namespace smallest_possible_AC_l621_62169

-- Constants and assumptions
variables (AC CD : ℕ)
def BD_squared : ℕ := 68

-- Prime number constraint for CD
def is_prime (n : ℕ) : Prop := n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

-- Given facts
axiom eq_ab_ac (AB : ℕ) : AB = AC
axiom perp_bd_ac (BD AC : ℕ) : BD^2 = BD_squared
axiom int_ac_cd : AC = (CD^2 + BD_squared) / (2 * CD)

theorem smallest_possible_AC :
  ∃ AC : ℕ, (∃ CD : ℕ, is_prime CD ∧ CD < 10 ∧ AC = (CD^2 + BD_squared) / (2 * CD)) ∧ AC = 18 :=
by
  sorry

end smallest_possible_AC_l621_62169


namespace jeremy_is_40_l621_62153

-- Definitions for Jeremy (J), Sebastian (S), and Sophia (So)
def JeremyCurrentAge : ℕ := 40
def SebastianCurrentAge : ℕ := JeremyCurrentAge + 4
def SophiaCurrentAge : ℕ := 60 - 3

-- Assertion properties
axiom age_sum_in_3_years : (JeremyCurrentAge + 3) + (SebastianCurrentAge + 3) + (SophiaCurrentAge + 3) = 150
axiom sebastian_older_by_4 : SebastianCurrentAge = JeremyCurrentAge + 4
axiom sophia_age_in_3_years : SophiaCurrentAge + 3 = 60

-- The theorem to prove that Jeremy is currently 40 years old
theorem jeremy_is_40 : JeremyCurrentAge = 40 := by
  sorry

end jeremy_is_40_l621_62153


namespace erased_number_is_six_l621_62199

theorem erased_number_is_six (n x : ℕ) (h1 : (n * (n + 1)) / 2 - x = 45 * (n - 1) / 4):
  x = 6 :=
by
  sorry

end erased_number_is_six_l621_62199


namespace hyperbola_eccentricity_sqrt2_l621_62150

noncomputable def isHyperbolaPerpendicularAsymptotes (a b : ℝ) (ha : a > 0) (hb : b > 0) : Prop :=
  let asymptote1 := (1/a : ℝ)
  let asymptote2 := (-1/b : ℝ)
  asymptote1 * asymptote2 = -1

theorem hyperbola_eccentricity_sqrt2 (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  isHyperbolaPerpendicularAsymptotes a b ha hb →
  let e := Real.sqrt (1 + (b^2 / a^2))
  e = Real.sqrt 2 :=
by
  intro h
  sorry

end hyperbola_eccentricity_sqrt2_l621_62150


namespace first_term_of_arithmetic_sequence_l621_62137

theorem first_term_of_arithmetic_sequence (a : ℕ → ℚ) (S : ℕ → ℚ) (d : ℚ)
  (h1 : a 3 = 3) (h2 : S 9 - S 6 = 27)
  (h3 : ∀ n, a n = a 1 + (n - 1) * d)
  (h4 : ∀ n, S n = n * (a 1 + a n) / 2) : a 1 = 3 / 5 :=
by
  sorry

end first_term_of_arithmetic_sequence_l621_62137


namespace number_of_planting_methods_l621_62109

theorem number_of_planting_methods :
  let vegetables := ["cucumbers", "cabbages", "rape", "flat beans"]
  let plots := ["plot1", "plot2", "plot3"]
  (∀ v ∈ vegetables, v = "cucumbers") →
  (∃! n : ℕ, n = 18)
:= by
  sorry

end number_of_planting_methods_l621_62109


namespace circles_intersect_if_and_only_if_l621_62195

theorem circles_intersect_if_and_only_if (m : ℝ) :
  (∃ (x y : ℝ), x^2 + y^2 = m ∧ x^2 + y^2 + 6 * x - 8 * y - 11 = 0) ↔ (1 < m ∧ m < 121) :=
by
  sorry

end circles_intersect_if_and_only_if_l621_62195


namespace smallest_prime_divisor_of_sum_of_powers_l621_62157

theorem smallest_prime_divisor_of_sum_of_powers :
  ∃ p, Prime p ∧ p = Nat.gcd (3 ^ 25 + 11 ^ 19) 2 := by
  sorry

end smallest_prime_divisor_of_sum_of_powers_l621_62157


namespace find_a_and_mono_l621_62143

open Real

noncomputable def f (x : ℝ) (a : ℝ) := (a * 2^x + a - 2) / (2^x + 1)

theorem find_a_and_mono :
  (∀ x : ℝ, f x a + f (-x) a = 0) →
  a = 1 ∧ f 3 1 = 7 / 9 ∧ ∀ x1 x2 : ℝ, x1 < x2 → f x1 1 < f x2 1 :=
by
  sorry

end find_a_and_mono_l621_62143


namespace sequence_an_square_l621_62171

theorem sequence_an_square (a : ℕ → ℝ) (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, a (n + 1) > a n) 
  (h3 : ∀ n : ℕ, a (n + 1)^2 + a n^2 + 1 = 2 * (a (n + 1) * a n + a (n + 1) + a n)) :
  ∀ n : ℕ, a n = n^2 :=
by
  sorry

end sequence_an_square_l621_62171


namespace moles_NaOH_combined_with_HCl_l621_62121

-- Definitions for given conditions
def NaOH : Type := Unit
def HCl : Type := Unit
def NaCl : Type := Unit
def H2O : Type := Unit

def balanced_reaction (nHCl nNaOH nNaCl nH2O : ℕ) : Prop :=
  nHCl = nNaOH ∧ nNaOH = nNaCl ∧ nNaCl = nH2O

def mole_mass_H2O : ℕ := 18

-- Given: certain amount of NaOH combined with 1 mole of HCl
def initial_moles_HCl : ℕ := 1

-- Given: 18 grams of H2O formed
def grams_H2O : ℕ := 18

-- Molar mass of H2O is approximately 18 g/mol, so 18 grams is 1 mole
def moles_H2O : ℕ := grams_H2O / mole_mass_H2O

-- Prove that number of moles of NaOH combined with HCl is 1 mole
theorem moles_NaOH_combined_with_HCl : 
  balanced_reaction initial_moles_HCl 1 1 moles_H2O →
  moles_H2O = 1 →
  1 = 1 :=
by
  intros h1 h2
  sorry

end moles_NaOH_combined_with_HCl_l621_62121


namespace sin_eq_product_one_eighth_l621_62116

open Real

theorem sin_eq_product_one_eighth :
  (∀ (n k m : ℕ), 1 ≤ n → n ≤ 5 → 1 ≤ k → k ≤ 5 → 1 ≤ m → m ≤ 5 →
    sin (π * n / 12) * sin (π * k / 12) * sin (π * m / 12) = 1 / 8) ↔ (n = 2 ∧ k = 2 ∧ m = 2) := by
  sorry

end sin_eq_product_one_eighth_l621_62116


namespace negation_of_forall_log_gt_one_l621_62168

noncomputable def negation_of_p : Prop :=
∃ x : ℝ, Real.log x ≤ 1

theorem negation_of_forall_log_gt_one :
  (¬ (∀ x : ℝ, Real.log x > 1)) ↔ negation_of_p :=
by
  sorry

end negation_of_forall_log_gt_one_l621_62168


namespace rectangle_dimensions_l621_62160

-- Define the dimensions and properties of the rectangle
variables {a b : ℕ}

-- Theorem statement
theorem rectangle_dimensions 
  (h1 : b = a + 3)
  (h2 : 2 * a + 2 * b + a = a * b) : 
  (a = 3 ∧ b = 6) :=
by
  sorry

end rectangle_dimensions_l621_62160


namespace find_u_l621_62128

-- Definitions for given points lying on a straight line
def point := (ℝ × ℝ)

-- Points
def p1 : point := (2, 8)
def p2 : point := (6, 20)
def p3 : point := (10, 32)

-- Function to check if point is on the line derived from p1, p2, p3
def is_on_line (x y : ℝ) : Prop :=
  ∃ m b : ℝ, y = m * x + b ∧
  p1.2 = m * p1.1 + b ∧ 
  p2.2 = m * p2.1 + b ∧
  p3.2 = m * p3.1 + b

-- Statement to prove
theorem find_u (u : ℝ) (hu : is_on_line 50 u) : u = 152 :=
sorry

end find_u_l621_62128


namespace circle_numbers_contradiction_l621_62139

theorem circle_numbers_contradiction :
  ¬ ∃ (f : Fin 25 → Fin 25), ∀ i : Fin 25, 
  let a := f i
  let b := f ((i + 1) % 25)
  (b = a + 10 ∨ b = a - 10 ∨ ∃ k : Int, b = a * k) :=
by
  sorry

end circle_numbers_contradiction_l621_62139


namespace inequality_solution_range_l621_62112

theorem inequality_solution_range (a : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 5 → x ^ 2 + a * x + 4 < 0) ↔ a < -4 :=
by 
  sorry

end inequality_solution_range_l621_62112


namespace decreasing_interval_l621_62106

noncomputable def y (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 15 * x^2 + 36 * x - 24

def has_extremum_at (a : ℝ) (x_ext : ℝ) : Prop :=
  deriv (y a) x_ext = 0

theorem decreasing_interval (a : ℝ) (h_extremum_at : has_extremum_at a 3) :
  a = 2 → ∀ x, (2 < x ∧ x < 3) → deriv (y a) x < 0 :=
sorry

end decreasing_interval_l621_62106


namespace chocolate_chips_per_member_l621_62146

/-
Define the problem conditions:
-/
def family_members := 4
def batches_choc_chip := 3
def cookies_per_batch_choc_chip := 12
def chips_per_cookie_choc_chip := 2
def batches_double_choc_chip := 2
def cookies_per_batch_double_choc_chip := 10
def chips_per_cookie_double_choc_chip := 4

/-
State the theorem to be proved:
-/
theorem chocolate_chips_per_member : 
  let total_choc_chip_cookies := batches_choc_chip * cookies_per_batch_choc_chip
  let total_choc_chips_choc_chip := total_choc_chip_cookies * chips_per_cookie_choc_chip
  let total_double_choc_chip_cookies := batches_double_choc_chip * cookies_per_batch_double_choc_chip
  let total_choc_chips_double_choc_chip := total_double_choc_chip_cookies * chips_per_cookie_double_choc_chip
  let total_choc_chips := total_choc_chips_choc_chip + total_choc_chips_double_choc_chip
  let chips_per_member := total_choc_chips / family_members
  chips_per_member = 38 :=
by
  sorry

end chocolate_chips_per_member_l621_62146


namespace second_mechanic_hours_l621_62161

theorem second_mechanic_hours (x y : ℕ) (h1 : 45 * x + 85 * y = 1100) (h2 : x + y = 20) : y = 5 :=
by
  sorry

end second_mechanic_hours_l621_62161


namespace jellybean_problem_l621_62181

theorem jellybean_problem:
  ∀ (black green orange : ℕ),
  black = 8 →
  green = black + 2 →
  black + green + orange = 27 →
  green - orange = 1 :=
by
  intros black green orange h_black h_green h_total
  sorry

end jellybean_problem_l621_62181


namespace smallest_integer_greater_than_power_l621_62101

theorem smallest_integer_greater_than_power (sqrt3 sqrt2 : ℝ) (h1 : (sqrt3 + sqrt2)^6 = 485 + 198 * Real.sqrt 6)
(h2 : (sqrt3 - sqrt2)^6 = 485 - 198 * Real.sqrt 6)
(h3 : 0 < (sqrt3 - sqrt2)^6 ∧ (sqrt3 - sqrt2)^6 < 1) : 
  ⌈(sqrt3 + sqrt2)^6⌉ = 970 := 
sorry

end smallest_integer_greater_than_power_l621_62101


namespace find_a_parallel_l621_62172

-- Define the lines
def line1 (a : ℝ) (x : ℝ) (y : ℝ) : Prop :=
  (a + 1) * x + 2 * y = 2

def line2 (a : ℝ) (x : ℝ) (y : ℝ) : Prop :=
  x + a * y = 1

-- Define the parallel condition
def are_parallel (a : ℝ) : Prop :=
  ∀ x y : ℝ, line1 a x y → line2 a x y

-- The theorem stating our problem
theorem find_a_parallel (a : ℝ) : are_parallel a → a = -2 :=
by
  sorry

end find_a_parallel_l621_62172


namespace triangle_area_and_coordinates_l621_62100

noncomputable def positive_diff_of_coordinates (A B C R S : ℝ × ℝ) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  let (xr, yr) := R
  let (xs, ys) := S
  if xr = xs then abs (xr - (10 - (x3 - xr)))
  else 0 -- Should never be this case if conditions are properly followed

theorem triangle_area_and_coordinates
  (A B C R S : ℝ × ℝ)
  (h_A : A = (0, 10))
  (h_B : B = (4, 0))
  (h_C : C = (10, 0))
  (h_vertical : R.fst = S.fst)
  (h_intersect_AC : R.snd = -(R.fst - 10))
  (h_intersect_BC : S.snd = 0 ∧ S.fst = 10 - (C.fst - R.fst))
  (h_area : 1/2 * ((R.fst - C.fst) * (R.snd - C.snd)) = 15) :
  positive_diff_of_coordinates A B C R S = 2 * Real.sqrt 30 - 10 := sorry

end triangle_area_and_coordinates_l621_62100


namespace find_f_at_4_l621_62108

theorem find_f_at_4 (f : ℝ → ℝ) (h : ∀ x : ℝ, x ≠ 0 → 3 * f x - 2 * f (1 / x) = x) : 
  f 4 = 5 / 2 :=
sorry

end find_f_at_4_l621_62108


namespace smallest_integer_in_range_l621_62133

theorem smallest_integer_in_range :
  ∃ n : ℕ, 
  1 < n ∧ 
  n % 3 = 2 ∧ 
  n % 5 = 2 ∧ 
  n % 7 = 2 ∧ 
  90 < n ∧ n < 119 :=
sorry

end smallest_integer_in_range_l621_62133


namespace one_point_one_seven_three_billion_in_scientific_notation_l621_62138

theorem one_point_one_seven_three_billion_in_scientific_notation :
  (1.173 * 10^9 = 1.173 * 1000000000) :=
by
  sorry

end one_point_one_seven_three_billion_in_scientific_notation_l621_62138


namespace raised_bed_height_l621_62177

theorem raised_bed_height : 
  ∀ (total_planks : ℕ) (num_beds : ℕ) (planks_per_bed : ℕ) (height : ℚ),
  total_planks = 50 →
  num_beds = 10 →
  planks_per_bed = 4 * height →
  (total_planks = num_beds * planks_per_bed) →
  height = 5 / 4 :=
by
  intros total_planks num_beds planks_per_bed H
  intros h1 h2 h3 h4
  sorry

end raised_bed_height_l621_62177


namespace roots_poly_sum_cubed_eq_l621_62158

theorem roots_poly_sum_cubed_eq :
  ∀ (r s t : ℝ), (r + s + t = 0) 
  → (∀ x, 9 * x^3 + 2023 * x + 4047 = 0 → x = r ∨ x = s ∨ x = t) 
  → (r + s) ^ 3 + (s + t) ^ 3 + (t + r) ^ 3 = 1349 :=
by
  intros r s t h_sum h_roots
  sorry

end roots_poly_sum_cubed_eq_l621_62158


namespace min_value_f1_min_value_f1_achieved_max_value_f2_max_value_f2_achieved_l621_62107

-- Problem (Ⅰ)
theorem min_value_f1 (x : ℝ) (h : x > 0) : (12 / x + 3 * x) ≥ 12 :=
sorry

theorem min_value_f1_achieved : (12 / 2 + 3 * 2) = 12 :=
by norm_num

-- Problem (Ⅱ)
theorem max_value_f2 (x : ℝ) (h : 0 < x ∧ x < 1 / 3) : x * (1 - 3 * x) ≤ 1 / 12 :=
sorry

theorem max_value_f2_achieved : (1 / 6) * (1 - 3 * (1 / 6)) = 1 / 12 :=
by norm_num

end min_value_f1_min_value_f1_achieved_max_value_f2_max_value_f2_achieved_l621_62107


namespace line_AB_eq_x_plus_3y_zero_l621_62170

/-- 
Consider two circles defined by:
C1: x^2 + y^2 - 4x + 6y = 0
C2: x^2 + y^2 - 6x = 0

Prove that the equation of the line through the intersection points of these two circles (line AB)
is x + 3y = 0.
-/
theorem line_AB_eq_x_plus_3y_zero (x y : ℝ) :
  (x^2 + y^2 - 4 * x + 6 * y = 0) ∧ (x^2 + y^2 - 6 * x = 0) → (x + 3 * y = 0) :=
by
  sorry

end line_AB_eq_x_plus_3y_zero_l621_62170


namespace inequality_proof_l621_62130

theorem inequality_proof
  (a1 b1 c1 a2 b2 c2 : ℝ)
  (ha1 : 0 < a1) (hb1 : 0 < b1) (hc1 : 0 < c1)
  (ha2 : 0 < a2) (hb2 : 0 < b2) (hc2 : 0 < c2)
  (h1: b1^2 ≤ a1 * c1)
  (h2: b2^2 ≤ a2 * c2) :
  (a1 + a2 + 5) * (c1 + c2 + 2) > (b1 + b2 + 3)^2 :=
by
  sorry

end inequality_proof_l621_62130


namespace find_xyz_l621_62179

theorem find_xyz (x y z : ℝ) (h1 : x * (y + z) = 195) (h2 : y * (z + x) = 204) (h3 : z * (x + y) = 213) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x * y * z = 1029 := by
  sorry

end find_xyz_l621_62179


namespace cathy_wallet_left_money_l621_62129

noncomputable def amount_left_in_wallet (initial : ℝ) (dad_amount : ℝ) (book_cost : ℝ) (saving_percentage : ℝ) : ℝ :=
  let mom_amount := 2 * dad_amount
  let total_initial := initial + dad_amount + mom_amount
  let after_purchase := total_initial - book_cost
  let saved_amount := saving_percentage * after_purchase
  after_purchase - saved_amount

theorem cathy_wallet_left_money :
  amount_left_in_wallet 12 25 15 0.20 = 57.60 :=
by 
  sorry

end cathy_wallet_left_money_l621_62129


namespace average_cost_correct_l621_62104

-- Defining the conditions
def groups_of_4_oranges := 11
def cost_of_4_oranges_bundle := 15
def groups_of_7_oranges := 2
def cost_of_7_oranges_bundle := 25

-- Calculating the relevant quantities as per the conditions
def total_cost : ℕ := (groups_of_4_oranges * cost_of_4_oranges_bundle) + (groups_of_7_oranges * cost_of_7_oranges_bundle)
def total_oranges : ℕ := (groups_of_4_oranges * 4) + (groups_of_7_oranges * 7)
def average_cost_per_orange := (total_cost:ℚ) / (total_oranges:ℚ)

-- Proving the average cost per orange matches the correct answer
theorem average_cost_correct : average_cost_per_orange = 215 / 58 := by
  sorry

end average_cost_correct_l621_62104


namespace chords_intersect_probability_l621_62135

noncomputable def probability_chords_intersect (n m : ℕ) : ℚ :=
  if (n > 6 ∧ m = 2023) then
    1 / 72
  else
    0

theorem chords_intersect_probability :
  probability_chords_intersect 6 2023 = 1 / 72 :=
by
  sorry

end chords_intersect_probability_l621_62135


namespace edmonton_to_calgary_travel_time_l621_62105

theorem edmonton_to_calgary_travel_time :
  let distance_edmonton_red_deer := 220
  let distance_red_deer_calgary := 110
  let speed_to_red_deer := 100
  let detour_distance := 30
  let detour_time := (distance_edmonton_red_deer + detour_distance) / speed_to_red_deer
  let stop_time := 1
  let speed_to_calgary := 90
  let travel_time_to_calgary := distance_red_deer_calgary / speed_to_calgary
  detour_time + stop_time + travel_time_to_calgary = 4.72 := by
  sorry

end edmonton_to_calgary_travel_time_l621_62105


namespace average_minutes_run_per_day_l621_62193

-- Define the given averages for each grade
def sixth_grade_avg : ℕ := 10
def seventh_grade_avg : ℕ := 18
def eighth_grade_avg : ℕ := 12

-- Define the ratios of the number of students in each grade
def num_sixth_eq_three_times_num_seventh (num_seventh : ℕ) : ℕ := 3 * num_seventh
def num_eighth_eq_half_num_seventh (num_seventh : ℕ) : ℕ := num_seventh / 2

-- Average number of minutes run per day by all students
theorem average_minutes_run_per_day (num_seventh : ℕ) :
  (sixth_grade_avg * num_sixth_eq_three_times_num_seventh num_seventh +
   seventh_grade_avg * num_seventh +
   eighth_grade_avg * num_eighth_eq_half_num_seventh num_seventh) / 
  (num_sixth_eq_three_times_num_seventh num_seventh + 
   num_seventh + 
   num_eighth_eq_half_num_seventh num_seventh) = 12 := 
sorry

end average_minutes_run_per_day_l621_62193


namespace gcd_1855_1120_l621_62115

theorem gcd_1855_1120 : Int.gcd 1855 1120 = 35 :=
by
  sorry

end gcd_1855_1120_l621_62115


namespace max_area_triang_ABC_l621_62114

noncomputable def max_area_triang (a b c : ℝ) (M : ℝ) (BM : ℝ := 2) (AM : ℝ := c - b) : ℝ :=
if M = (b + c) / 2 then 2 * Real.sqrt 3 else 0

theorem max_area_triang_ABC (a b c : ℝ) (M : ℝ) (BM : ℝ := 2) (AM : ℝ := c - b) (M_midpoint : M = (b + c) / 2) :
  max_area_triang a b c M BM AM = 2 * Real.sqrt 3 :=
by
  sorry

end max_area_triang_ABC_l621_62114


namespace jose_share_of_profit_l621_62136

-- Definitions from problem conditions
def tom_investment : ℕ := 30000
def jose_investment : ℕ := 45000
def profit : ℕ := 27000
def months_total : ℕ := 12
def months_jose_investment : ℕ := 10

-- Derived calculations
def tom_month_investment := tom_investment * months_total
def jose_month_investment := jose_investment * months_jose_investment
def total_month_investment := tom_month_investment + jose_month_investment

-- Prove Jose's share of profit
theorem jose_share_of_profit : (jose_month_investment * profit) / total_month_investment = 15000 := by
  -- This is where the step-by-step proof would go
  sorry

end jose_share_of_profit_l621_62136


namespace hyperbola_equation_l621_62188

noncomputable def hyperbola_eqn : Prop :=
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (b = (1/2) * a) ∧ (a^2 + b^2 = 25) ∧ 
    (∀ x y, (x^2 / (a^2)) - (y^2 / (b^2)) = 1 ↔ (x^2 / 20) - (y^2 / 5) = 1)

theorem hyperbola_equation : hyperbola_eqn := 
  sorry

end hyperbola_equation_l621_62188


namespace option_d_correct_l621_62165

variable (a b : ℝ)

theorem option_d_correct : (-a^3)^4 = a^(12) := by sorry

end option_d_correct_l621_62165


namespace smallest_four_digit_integer_mod_8_eq_3_l621_62144

theorem smallest_four_digit_integer_mod_8_eq_3 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 8 = 3 ∧ n = 1003 := by
  -- Proof will be provided here
  sorry

end smallest_four_digit_integer_mod_8_eq_3_l621_62144


namespace lara_bag_total_chips_l621_62186

theorem lara_bag_total_chips (C : ℕ)
  (h1 : ∃ (b : ℕ), b = C / 6)
  (h2 : 34 + 16 + C / 6 = C) :
  C = 60 := by
  sorry

end lara_bag_total_chips_l621_62186


namespace horse_cow_difference_l621_62113

def initial_conditions (h c : ℕ) : Prop :=
  4 * c = h

def transaction (h c : ℕ) : Prop :=
  (h - 15) * 7 = (c + 15) * 13

def final_difference (h c : ℕ) : Prop := 
  h - 15 - (c + 15) = 30

theorem horse_cow_difference (h c : ℕ) (hc : initial_conditions h c) (ht : transaction h c) : final_difference h c :=
    by
      sorry

end horse_cow_difference_l621_62113


namespace complex_modulus_problem_l621_62151

open Complex

def modulus_of_z (z : ℂ) (h : (z - 2 * I) * (1 - I) = -2) : Prop :=
  abs z = Real.sqrt 2

theorem complex_modulus_problem (z : ℂ) (h : (z - 2 * I) * (1 - I) = -2) : 
  modulus_of_z z h :=
sorry

end complex_modulus_problem_l621_62151


namespace sum_of_consecutive_evens_l621_62173

/-- 
  Prove that the sum of five consecutive even integers 
  starting from 2n, with a common difference of 2, is 10n + 20.
-/
theorem sum_of_consecutive_evens (n : ℕ) :
  (2 * n) + (2 * n + 2) + (2 * n + 4) + (2 * n + 6) + (2 * n + 8) = 10 * n + 20 := 
by
  sorry

end sum_of_consecutive_evens_l621_62173


namespace root_division_simplification_l621_62103

theorem root_division_simplification (a : ℝ) (h1 : a = (7 : ℝ)^(1/4)) (h2 : a = (7 : ℝ)^(1/7)) :
  ((7 : ℝ)^(1/4) / (7 : ℝ)^(1/7)) = (7 : ℝ)^(3/28) :=
sorry

end root_division_simplification_l621_62103


namespace find_value_of_expression_l621_62124

theorem find_value_of_expression
  (a b c d : ℝ) (h₀ : a ≥ 0) (h₁ : b ≥ 0) (h₂ : c ≥ 0) (h₃ : d ≥ 0)
  (h₄ : a / (b + c + d) = b / (a + c + d))
  (h₅ : b / (a + c + d) = c / (a + b + d))
  (h₆ : c / (a + b + d) = d / (a + b + c))
  (h₇ : d / (a + b + c) = a / (b + c + d)) :
  (a + b) / (c + d) + (b + c) / (a + d) + (c + d) / (a + b) + (d + a) / (b + c) = 4 :=
by sorry

end find_value_of_expression_l621_62124


namespace find_scalars_l621_62110

noncomputable def N : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![3, 4], ![-2, 0]]

noncomputable def I : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![1, 0], ![0, 1]]

theorem find_scalars (r s : ℤ) (h_r : r = 3) (h_s : s = -8) :
    N * N = r • N + s • I :=
by
  rw [h_r, h_s]
  sorry

end find_scalars_l621_62110


namespace olivia_pieces_of_paper_l621_62111

theorem olivia_pieces_of_paper (initial_pieces : ℕ) (used_pieces : ℕ) (pieces_left : ℕ) 
  (h1 : initial_pieces = 81) (h2 : used_pieces = 56) : 
  pieces_left = 81 - 56 :=
by
  sorry

end olivia_pieces_of_paper_l621_62111


namespace inscribed_sphere_tetrahedron_volume_l621_62185

theorem inscribed_sphere_tetrahedron_volume
  (R : ℝ) (S1 S2 S3 S4 : ℝ) :
  ∃ V : ℝ, V = (1 / 3) * R * (S1 + S2 + S3 + S4) :=
sorry

end inscribed_sphere_tetrahedron_volume_l621_62185


namespace amount_leaked_during_repairs_l621_62117

theorem amount_leaked_during_repairs:
  let total_leaked := 6206
  let leaked_before_repairs := 2475
  total_leaked - leaked_before_repairs = 3731 :=
by
  sorry

end amount_leaked_during_repairs_l621_62117


namespace f_equals_one_l621_62167

-- Define the functions f, g, h with the given properties

def f : ℕ → ℕ := sorry
def g : ℕ → ℕ := sorry
def h : ℕ → ℕ := sorry

-- Condition 1: h is injective
axiom h_injective : ∀ {a b : ℕ}, h a = h b → a = b

-- Condition 2: g is surjective
axiom g_surjective : ∀ n : ℕ, ∃ m : ℕ, g m = n

-- Condition 3: Definition of f in terms of g and h
axiom f_def : ∀ n : ℕ, f n = g n - h n + 1

-- Prove that f(n) = 1 for all n ∈ ℕ
theorem f_equals_one : ∀ n : ℕ, f n = 1 := by
  sorry

end f_equals_one_l621_62167


namespace paper_boat_travel_time_l621_62152

-- Defining the conditions as constants
def distance_embankment : ℝ := 50
def speed_downstream : ℝ := 10
def speed_upstream : ℝ := 12.5

-- Definitions for the speeds of the boat and current
noncomputable def v_boat : ℝ := (speed_upstream + speed_downstream) / 2
noncomputable def v_current : ℝ := (speed_downstream - speed_upstream) / 2

-- Statement to prove the time taken for the paper boat
theorem paper_boat_travel_time :
  (distance_embankment / v_current) = 40 := by
  sorry

end paper_boat_travel_time_l621_62152


namespace largest_five_digit_integer_congruent_to_16_mod_25_l621_62187

theorem largest_five_digit_integer_congruent_to_16_mod_25 :
  ∃ x : ℤ, x % 25 = 16 ∧ x < 100000 ∧ ∀ y : ℤ, y % 25 = 16 → y < 100000 → y ≤ x :=
by
  sorry

end largest_five_digit_integer_congruent_to_16_mod_25_l621_62187


namespace milk_mixture_l621_62122

theorem milk_mixture:
  ∀ (x : ℝ), 0.40 * x + 1.6 = 0.20 * (x + 16) → x = 8 := 
by
  intro x
  sorry

end milk_mixture_l621_62122


namespace units_produced_today_l621_62163

theorem units_produced_today (n : ℕ) (X : ℕ) 
  (h1 : n = 9) 
  (h2 : (360 + X) / (n + 1) = 45) 
  (h3 : 40 * n = 360) : 
  X = 90 := 
sorry

end units_produced_today_l621_62163


namespace jackson_entertainment_expense_l621_62125

noncomputable def total_spent_on_entertainment_computer_game_original_price : ℝ :=
  66 / 0.85

noncomputable def movie_ticket_price_with_tax : ℝ :=
  12 * 1.10

noncomputable def total_movie_tickets_cost : ℝ :=
  3 * movie_ticket_price_with_tax

noncomputable def total_snacks_and_transportation_cost : ℝ :=
  7 + 5

noncomputable def total_spent : ℝ :=
  66 + total_movie_tickets_cost + total_snacks_and_transportation_cost

theorem jackson_entertainment_expense :
  total_spent = 117.60 :=
by
  sorry

end jackson_entertainment_expense_l621_62125


namespace sixth_graders_l621_62189

theorem sixth_graders (total_students sixth_graders seventh_graders : ℕ)
    (h1 : seventh_graders = 64)
    (h2 : 32 * total_students = 64 * 100)
    (h3 : sixth_graders * 100 = 38 * total_students) :
    sixth_graders = 76 := by
  sorry

end sixth_graders_l621_62189


namespace rectangle_side_length_along_hypotenuse_l621_62119

-- Define the right triangle with given sides
def triangle_PQR (PR PQ QR : ℝ) : Prop := 
  PR^2 + PQ^2 = QR^2

-- Condition: Right triangle PQR with PR = 9 and PQ = 12
def PQR : Prop := triangle_PQR 9 12 (Real.sqrt (9^2 + 12^2))

-- Define the property of the rectangle
def rectangle_condition (x : ℝ) (s : ℝ) : Prop := 
  (3 / (Real.sqrt (9^2 + 12^2))) = (x / 9) ∧ s = ((9 - x) * (Real.sqrt (9^2 + 12^2)) / 9)

-- Main theorem
theorem rectangle_side_length_along_hypotenuse : 
  PQR ∧ (∃ x, rectangle_condition x 12) → (∃ s, s = 12) :=
by
  intro h
  sorry

end rectangle_side_length_along_hypotenuse_l621_62119


namespace total_buyers_in_three_days_l621_62145

theorem total_buyers_in_three_days
  (D_minus_2 : ℕ)
  (D_minus_1 : ℕ)
  (D_0 : ℕ)
  (h1 : D_minus_2 = 50)
  (h2 : D_minus_1 = D_minus_2 / 2)
  (h3 : D_0 = D_minus_1 + 40) :
  D_minus_2 + D_minus_1 + D_0 = 140 :=
by
  sorry

end total_buyers_in_three_days_l621_62145


namespace propositions_true_false_l621_62132

theorem propositions_true_false :
  (∃ x : ℝ, x ^ 3 < 1) ∧ 
  ¬ (∃ x : ℚ, x ^ 2 = 2) ∧ 
  ¬ (∀ x : ℕ, x ^ 3 > x ^ 2) ∧ 
  (∀ x : ℝ, x ^ 2 + 1 > 0) :=
by
  sorry

end propositions_true_false_l621_62132


namespace radius_of_tangent_sphere_l621_62178

theorem radius_of_tangent_sphere (r1 r2 : ℝ) (h : r1 = 12 ∧ r2 = 3) :
  ∃ r : ℝ, (r = 6) :=
by
  sorry

end radius_of_tangent_sphere_l621_62178


namespace infinitely_many_primes_satisfying_condition_l621_62182

theorem infinitely_many_primes_satisfying_condition :
  ∀ k : Nat, ∃ p : Nat, Nat.Prime p ∧ ∃ n : Nat, n > 0 ∧ p ∣ (2014^(2^n) + 2014) := 
sorry

end infinitely_many_primes_satisfying_condition_l621_62182


namespace largest_number_is_89_l621_62131

theorem largest_number_is_89 (a b c d : ℕ) 
  (h1 : a + b + c = 180) 
  (h2 : a + b + d = 197) 
  (h3 : a + c + d = 208) 
  (h4 : b + c + d = 222) : 
  max a (max b (max c d)) = 89 := 
by sorry

end largest_number_is_89_l621_62131


namespace intersect_P_Q_l621_62155

open Set

def P : Set ℤ := { x | (x - 3) * (x - 6) ≤ 0 }
def Q : Set ℤ := { 5, 7 }

theorem intersect_P_Q : P ∩ Q = {5} :=
sorry

end intersect_P_Q_l621_62155


namespace ratio_of_members_l621_62196

theorem ratio_of_members (r p : ℕ) (h1 : 5 * r + 12 * p = 8 * (r + p)) : (r / p : ℚ) = 4 / 3 := by
  sorry -- This is a placeholder for the actual proof.

end ratio_of_members_l621_62196


namespace price_of_each_sundae_l621_62162

theorem price_of_each_sundae (A B : ℝ) (x y z : ℝ) (hx : 200 * x = 80) (hy : A = y) (hz : y = 0.40)
  (hxy : A - 80 = z) (hyz : 200 * z = B) : y = 0.60 :=
by
  sorry

end price_of_each_sundae_l621_62162


namespace faye_rows_l621_62102

theorem faye_rows (total_pencils : ℕ) (pencils_per_row : ℕ) (rows_created : ℕ) :
  total_pencils = 12 → pencils_per_row = 4 → rows_created = 3 := by
  sorry

end faye_rows_l621_62102


namespace jack_paid_20_l621_62159

-- Define the conditions
def numberOfSandwiches : Nat := 3
def costPerSandwich : Nat := 5
def changeReceived : Nat := 5

-- Define the total cost
def totalCost : Nat := numberOfSandwiches * costPerSandwich

-- Define the amount paid
def amountPaid : Nat := totalCost + changeReceived

-- Prove that the amount paid is 20
theorem jack_paid_20 : amountPaid = 20 := by
  -- You may assume the steps and calculations here, only providing the statement
  sorry

end jack_paid_20_l621_62159


namespace power_function_value_l621_62127

theorem power_function_value (f : ℝ → ℝ) (h : ∃ a : ℝ, ∀ x : ℝ, f x = x ^ a) (h₁ : f 4 = 1 / 2) :
  f (1 / 16) = 4 :=
sorry

end power_function_value_l621_62127


namespace arithmetic_sequence_8th_term_is_71_l621_62134

def arithmetic_sequence_8th_term (a d : ℤ) : ℤ := a + 7 * d

theorem arithmetic_sequence_8th_term_is_71 (a d : ℤ) 
  (h4 : a + 3 * d = 23) 
  (h6 : a + 5 * d = 47) : 
  arithmetic_sequence_8th_term a d = 71 :=
by
  sorry

end arithmetic_sequence_8th_term_is_71_l621_62134


namespace min_value_of_quadratic_l621_62140

theorem min_value_of_quadratic : ∃ x : ℝ, 7 * x^2 - 28 * x + 1702 = 1674 ∧ ∀ y : ℝ, 7 * y^2 - 28 * y + 1702 ≥ 1674 :=
by
  sorry

end min_value_of_quadratic_l621_62140


namespace max_value_is_zero_l621_62194

noncomputable def max_value (x y : ℝ) (h : 2 * (x^3 + y^3) = x^2 + y^2) : ℝ :=
  x^2 - y^2

theorem max_value_is_zero (x y : ℝ) (h : 2 * (x^3 + y^3) = x^2 + y^2) : max_value x y h = 0 :=
sorry

end max_value_is_zero_l621_62194


namespace one_number_greater_than_one_l621_62184

theorem one_number_greater_than_one
  (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_prod : a * b * c = 1)
  (h_sum : a + b + c > 1/a + 1/b + 1/c) :
  ((1 < a ∧ b ≤ 1 ∧ c ≤ 1) ∨ (a ≤ 1 ∧ 1 < b ∧ c ≤ 1) ∨ (a ≤ 1 ∧ b ≤ 1 ∧ 1 < c)) 
  ∧ (¬ ((1 < a ∧ 1 < b) ∨ (1 < b ∧ 1 < c) ∨ (1 < a ∧ 1 < c))) :=
sorry

end one_number_greater_than_one_l621_62184


namespace no_triangle_satisfies_sine_eq_l621_62197

theorem no_triangle_satisfies_sine_eq (A B C : ℝ) (a b c : ℝ) 
  (hA: 0 < A) (hB: 0 < B) (hC: 0 < C) 
  (hA_ineq: A < π) (hB_ineq: B < π) (hC_ineq: C < π) 
  (h_sum: A + B + C = π) 
  (sin_eq: Real.sin A + Real.sin B = Real.sin C)
  (h_tri_ineq: a + b > c ∧ a + c > b ∧ b + c > a) 
  (h_sines: a = 2 * (1) * Real.sin A ∧ b = 2 * (1) * Real.sin B ∧ c = 2 * (1) * Real.sin C) :
  False :=
sorry

end no_triangle_satisfies_sine_eq_l621_62197


namespace solve_for_multiplier_l621_62180

namespace SashaSoup
  
-- Variables representing the amounts of salt
variables (x y : ℝ)

-- Condition provided: amount of salt added today
def initial_salt := 2 * x
def additional_salt_today := 0.5 * y

-- Given relationship
axiom salt_relationship : x = 0.5 * y

-- The multiplier k to achieve the required amount of salt
def required_multiplier : ℝ := 1.5

-- Lean theorem statement
theorem solve_for_multiplier :
  (2 * x) * required_multiplier = x + y :=
by
  -- Mathematical proof goes here but since asked to skip proof we use sorry
  sorry

end SashaSoup

end solve_for_multiplier_l621_62180


namespace sequence_term_4th_l621_62192

theorem sequence_term_4th (a_n : ℕ → ℝ) (h : ∀ n, a_n n = 2 / (n^2 + n)) :
  ∃ n, a_n n = 1 / 10 ∧ n = 4 :=
by
  sorry

end sequence_term_4th_l621_62192


namespace scores_greater_than_18_l621_62154

noncomputable def olympiad_scores (scores : Fin 20 → ℕ) :=
∀ i j k : Fin 20, i ≠ j → i ≠ k → j ≠ k → scores i < scores j + scores k

theorem scores_greater_than_18 (scores : Fin 20 → ℕ) (h1 : ∀ i j, i < j → scores i < scores j)
  (h2 : olympiad_scores scores) : ∀ i, 18 < scores i :=
by
  intro i
  sorry

end scores_greater_than_18_l621_62154


namespace range_of_a_l621_62126

def A : Set ℝ := {x | x > 1}
def B (a : ℝ) : Set ℝ := {x | x < a}

theorem range_of_a (a : ℝ) (h : (A ∩ B a).Nonempty) : a > 1 :=
sorry

end range_of_a_l621_62126


namespace perpendicular_line_through_point_l621_62148

noncomputable def line_eq_perpendicular (x y : ℝ) : Prop :=
  3 * x - 6 * y = 9

noncomputable def slope_intercept_form (m b x y : ℝ) : Prop :=
  y = m * x + b

theorem perpendicular_line_through_point
  (x y : ℝ)
  (hx : x = 2)
  (hy : y = -3) :
  ∀ x y, line_eq_perpendicular x y →
  ∃ m b, slope_intercept_form m b x y ∧ m = -2 ∧ b = 1 :=
sorry

end perpendicular_line_through_point_l621_62148


namespace correct_option_d_l621_62118

variable (m t x1 x2 y1 y2 : ℝ)

theorem correct_option_d (h_m : m > 0)
  (h_y1 : y1 = m * x1^2 - 2 * m * x1 + 1)
  (h_y2 : y2 = m * x2^2 - 2 * m * x2 + 1)
  (h_x1 : t < x1 ∧ x1 < t + 1)
  (h_x2 : t + 2 < x2 ∧ x2 < t + 3)
  (h_t_geq1 : t ≥ 1) :
  y1 < y2 := sorry

end correct_option_d_l621_62118


namespace contrapositive_of_inequality_l621_62183

variable {a b c : ℝ}

theorem contrapositive_of_inequality (h : a + c ≤ b + c) : a ≤ b :=
sorry

end contrapositive_of_inequality_l621_62183


namespace geometric_sequence_sum_l621_62191

theorem geometric_sequence_sum (q : ℝ) (a : ℕ → ℝ)
  (h1 : a 1 = 1)
  (h_geometric : ∀ n, a (n + 1) = a n * q)
  (h2 : a 3 + a 5 = 6) :
  a 5 + a 7 + a 9 = 28 :=
  sorry

end geometric_sequence_sum_l621_62191


namespace symmetric_line_eq_l621_62166

/-- 
Given two circles O: x^2 + y^2 = 4 and C: x^2 + y^2 + 4x - 4y + 4 = 0, 
prove the equation of the line l such that the two circles are symmetric 
with respect to line l is x - y + 2 = 0.
-/
theorem symmetric_line_eq {x y : ℝ} :
  (∀ x y : ℝ, (x^2 + y^2 = 4) → (x^2 + y^2 + 4*x - 4*y + 4 = 0)) → (∀ x y : ℝ, (x - y + 2 = 0)) :=
  sorry

end symmetric_line_eq_l621_62166


namespace socks_thrown_away_l621_62190

theorem socks_thrown_away 
  (initial_socks new_socks current_socks : ℕ) 
  (h1 : initial_socks = 11) 
  (h2 : new_socks = 26) 
  (h3 : current_socks = 33) : 
  initial_socks + new_socks - current_socks = 4 :=
by {
  sorry
}

end socks_thrown_away_l621_62190


namespace positive_integer_pairs_divisibility_l621_62141

theorem positive_integer_pairs_divisibility (a b : ℕ) (h : a * b^2 + b + 7 ∣ a^2 * b + a + b) :
  (a = 11 ∧ b = 1) ∨ (a = 49 ∧ b = 1) ∨ ∃ k : ℕ, k > 0 ∧ a = 7 * k^2 ∧ b = 7 * k :=
sorry

end positive_integer_pairs_divisibility_l621_62141


namespace additional_charge_per_international_letter_l621_62174

-- Definitions based on conditions
def standard_postage_per_letter : ℕ := 108
def num_international_letters : ℕ := 2
def total_cost : ℕ := 460
def num_letters : ℕ := 4

-- Theorem stating the question
theorem additional_charge_per_international_letter :
  (total_cost - (num_letters * standard_postage_per_letter)) / num_international_letters = 14 :=
by
  sorry

end additional_charge_per_international_letter_l621_62174


namespace pentagon_sum_of_sides_and_vertices_eq_10_l621_62156

-- Define the number of sides of a pentagon
def number_of_sides : ℕ := 5

-- Define the number of vertices of a pentagon
def number_of_vertices : ℕ := 5

-- Define the sum of sides and vertices
def sum_of_sides_and_vertices : ℕ :=
  number_of_sides + number_of_vertices

-- The theorem to prove that the sum is 10
theorem pentagon_sum_of_sides_and_vertices_eq_10 : sum_of_sides_and_vertices = 10 :=
by
  sorry

end pentagon_sum_of_sides_and_vertices_eq_10_l621_62156


namespace max_value_condition_l621_62147

noncomputable def f (a x : ℝ) : ℝ :=
if h : 0 < x ∧ x ≤ a then
  Real.log x
else
  if x > a then
    a / x
  else
    0 -- This case should not happen given the domain conditions

theorem max_value_condition (a : ℝ) : 
  (∃ M, ∀ x > 0, x ≤ a → f a x ≤ M) ∧ (∀ x > a, f a x ≤ M) ↔ a ≥ Real.exp 1 :=
sorry

end max_value_condition_l621_62147


namespace can_form_triangle_l621_62149

-- Define the function to check for the triangle inequality
def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Problem statement: Prove that only the set (3, 4, 6) can form a triangle
theorem can_form_triangle :
  (¬ is_triangle 3 4 8) ∧
  (¬ is_triangle 5 6 11) ∧
  (¬ is_triangle 5 8 15) ∧
  (is_triangle 3 4 6) :=
by
  sorry

end can_form_triangle_l621_62149


namespace find_angle_A_area_bound_given_a_l621_62142

-- (1) Given the condition, prove that \(A = \frac{\pi}{3}\).
theorem find_angle_A
  {A B C : ℝ} {a b c : ℝ}
  (h1 : a / (Real.cos A) + b / (Real.cos B) + c / (Real.cos C) = (Real.sqrt 3) * c * (Real.sin B) / (Real.cos B * Real.cos C)) :
  A = Real.pi / 3 :=
sorry

-- (2) Given a = 4, prove the area S satisfies \(S \leq 4\sqrt{3}\).
theorem area_bound_given_a
  {A B C : ℝ} {a b c S : ℝ}
  (ha : a = 4)
  (hA : A = Real.pi / 3)
  (h1 : a / (Real.cos A) + b / (Real.cos B) + c / (Real.cos C) = (Real.sqrt 3) * c * (Real.sin B) / (Real.cos B * Real.cos C))
  (hS : S = 1 / 2 * b * c * Real.sin A) :
  S ≤ 4 * Real.sqrt 3 :=
sorry

end find_angle_A_area_bound_given_a_l621_62142


namespace simplify_expression_l621_62175

def a : ℚ := (3 / 4) * 60
def b : ℚ := (8 / 5) * 60
def c : ℚ := 63

theorem simplify_expression : a - b + c = 12 := by
  sorry

end simplify_expression_l621_62175


namespace find_r_l621_62198

-- Define the basic conditions based on the given problem.
def pr (r : ℕ) := 360 / 6
def p := pr 4 / 4
def cr (c r : ℕ) := 6 * c * r

-- Prove that r = 4 given the conditions.
theorem find_r (r : ℕ) : r = 4 :=
by
  sorry

end find_r_l621_62198


namespace maximum_area_of_triangle_OAB_l621_62120

noncomputable def maximum_area_triangle (a b : ℝ) : ℝ :=
  if 2 * a + b = 5 ∧ a > 0 ∧ b > 0 then (1 / 2) * a * b else 0

theorem maximum_area_of_triangle_OAB : 
  (∀ (a b : ℝ), 2 * a + b = 5 ∧ a > 0 ∧ b > 0 → (1 / 2) * a * b ≤ 25 / 16) :=
by
  sorry

end maximum_area_of_triangle_OAB_l621_62120
