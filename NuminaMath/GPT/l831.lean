import Mathlib

namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l831_83177

theorem arithmetic_sequence_sum 
  (a : ℕ → ℤ) 
  (h_arith : ∀ n : ℕ, a (n+1) - a n = a 1 - a 0) 
  (h_sum : a 2 + a 4 + a 6 = 12) : 
  (a 0 + a 1 + a 2 + a 3 + a 4 + a 5 + a 6 = 28) :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l831_83177


namespace NUMINAMATH_GPT_total_hamburgers_menu_l831_83111

def meat_patties_choices := 4
def condiment_combinations := 2 ^ 9

theorem total_hamburgers_menu :
  meat_patties_choices * condiment_combinations = 2048 :=
by
  sorry

end NUMINAMATH_GPT_total_hamburgers_menu_l831_83111


namespace NUMINAMATH_GPT_dave_trips_l831_83113

/-- Dave can only carry 9 trays at a time. -/
def trays_per_trip := 9

/-- Number of trays Dave has to pick up from one table. -/
def trays_from_table1 := 17

/-- Number of trays Dave has to pick up from another table. -/
def trays_from_table2 := 55

/-- Total number of trays Dave has to pick up. -/
def total_trays := trays_from_table1 + trays_from_table2

/-- The number of trips Dave will make. -/
def number_of_trips := total_trays / trays_per_trip

theorem dave_trips :
  number_of_trips = 8 :=
sorry

end NUMINAMATH_GPT_dave_trips_l831_83113


namespace NUMINAMATH_GPT_arithmetic_seq_sum_l831_83158

-- Definition of an arithmetic sequence using a common difference d
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d

-- Statement of the problem
theorem arithmetic_seq_sum (a : ℕ → ℝ) (d : ℝ) 
  (hs : arithmetic_sequence a d)
  (hmean : (a 3 + a 8) / 2 = 10) : 
  a 1 + a 10 = 20 :=
sorry

end NUMINAMATH_GPT_arithmetic_seq_sum_l831_83158


namespace NUMINAMATH_GPT_complement_intersection_l831_83105

open Set

universe u

def U : Set ℕ := {0, 1, 2, 3, 4}
def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {2, 3}

theorem complement_intersection :
  (U \ M) ∩ N = {3} :=
sorry

end NUMINAMATH_GPT_complement_intersection_l831_83105


namespace NUMINAMATH_GPT_find_range_of_m_l831_83191

noncomputable def range_of_m (m : ℝ) : Prop :=
  ((1 < m ∧ m ≤ 2) ∨ (3 ≤ m))

theorem find_range_of_m (m : ℝ) :
  (∃ x : ℝ, x^2 + m*x + 1 = 0 ∧ ∀ x1 x2 : ℝ, x1 ≠ x2 → x1 < 0 ∧ x2 < 0 ∧ x1^2 + m*x1 + 1 = 0 ∧ x2^2 + m*x2 + 1 = 0) ∨
  (¬ ∃ x : ℝ, 4 * x^2 + 4 * (m - 2) * x + 1 = 0 ∧ ∀ Δ, Δ < 0 ∧ Δ = 16 * (m^2 - 4 * m + 3)) ↔
  ¬((∃ x : ℝ, x^2 + m*x + 1 = 0 ∧ ∀ x1 x2 : ℝ, x1 ≠ x2 → x1 < 0 ∧ x2 < 0 ∧ x1^2 + m*x1 + 1 = 0 ∧ x2^2 + m*x2 + 1 = 0) ∧
  (¬ ∃ x : ℝ, 4 * x^2 + 4 * (m - 2) * x + 1 = 0 ∧ ∀ Δ, Δ < 0 ∧ Δ = 16 * (m^2 - 4 * m + 3))) →
  range_of_m m :=
sorry

end NUMINAMATH_GPT_find_range_of_m_l831_83191


namespace NUMINAMATH_GPT_intersecting_lines_l831_83142

variable (a b m : ℝ)

-- Conditions
def condition1 : Prop := 8 = -m + a
def condition2 : Prop := 8 = m + b

-- Statement to prove
theorem intersecting_lines : condition1 a m  → condition2 b m  → a + b = 16 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_intersecting_lines_l831_83142


namespace NUMINAMATH_GPT_sum_of_three_numbers_l831_83117

theorem sum_of_three_numbers
  (a b c : ℝ)
  (h1 : a^2 + b^2 + c^2 = 252)
  (h2 : ab + bc + ca = 116) :
  a + b + c = 22 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_three_numbers_l831_83117


namespace NUMINAMATH_GPT_jenny_mother_age_l831_83160

theorem jenny_mother_age:
  (∀ x : ℕ, (50 + x = 2 * (10 + x)) → (2010 + x = 2040)) :=
by
  sorry

end NUMINAMATH_GPT_jenny_mother_age_l831_83160


namespace NUMINAMATH_GPT_edmonton_to_calgary_travel_time_l831_83167

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

end NUMINAMATH_GPT_edmonton_to_calgary_travel_time_l831_83167


namespace NUMINAMATH_GPT_polynomial_coefficients_l831_83109

noncomputable def a : ℝ := 15
noncomputable def b : ℝ := -198
noncomputable def c : ℝ := 1

theorem polynomial_coefficients :
  (∀ x₁ x₂ x₃ : ℝ, 
    (x₁ + x₂ + x₃ = 0) ∧ 
    (x₁ * x₂ + x₂ * x₃ + x₃ * x₁ = -3) ∧ 
    (x₁ * x₂ * x₃ = -1) → 
    (a = 15) ∧ 
    (b = -198) ∧ 
    (c = 1)) := 
by sorry

end NUMINAMATH_GPT_polynomial_coefficients_l831_83109


namespace NUMINAMATH_GPT_percentage_increase_third_year_l831_83129

theorem percentage_increase_third_year
  (initial_price : ℝ)
  (price_2007 : ℝ := initial_price * (1 + 20 / 100))
  (price_2008 : ℝ := price_2007 * (1 - 25 / 100))
  (price_end_third_year : ℝ := initial_price * (108 / 100)) :
  ((price_end_third_year - price_2008) / price_2008) * 100 = 20 :=
by
  sorry

end NUMINAMATH_GPT_percentage_increase_third_year_l831_83129


namespace NUMINAMATH_GPT_probability_different_colors_is_correct_l831_83176

-- Definitions of chip counts
def blue_chips := 6
def red_chips := 5
def yellow_chips := 4
def green_chips := 3
def total_chips := blue_chips + red_chips + yellow_chips + green_chips

-- Definition of the probability calculation
def probability_different_colors := 
  ((blue_chips / total_chips) * ((red_chips + yellow_chips + green_chips) / total_chips)) +
  ((red_chips / total_chips) * ((blue_chips + yellow_chips + green_chips) / total_chips)) +
  ((yellow_chips / total_chips) * ((blue_chips + red_chips + green_chips) / total_chips)) +
  ((green_chips / total_chips) * ((blue_chips + red_chips + yellow_chips) / total_chips))

-- Given the problem conditions, we assert the correct answer
theorem probability_different_colors_is_correct :
  probability_different_colors = (119 / 162) := 
sorry

end NUMINAMATH_GPT_probability_different_colors_is_correct_l831_83176


namespace NUMINAMATH_GPT_meaning_of_probability_l831_83175

-- Definitions

def probability_of_winning (p : ℚ) : Prop :=
  p = 1 / 4

-- Theorem statement
theorem meaning_of_probability :
  probability_of_winning (1 / 4) →
  ∀ n : ℕ, (n ≠ 0) → (n / 4 * 4) = n :=
by
  -- Placeholder proof
  sorry

end NUMINAMATH_GPT_meaning_of_probability_l831_83175


namespace NUMINAMATH_GPT_find_multiple_of_numerator_l831_83197

theorem find_multiple_of_numerator
  (n d k : ℕ)
  (h1 : d = k * n - 1)
  (h2 : (n + 1) / (d + 1) = 3 / 5)
  (h3 : (n : ℚ) / d = 5 / 9) : k = 2 :=
sorry

end NUMINAMATH_GPT_find_multiple_of_numerator_l831_83197


namespace NUMINAMATH_GPT_problem_correct_l831_83116

def decimal_to_fraction_eq_80_5 : Prop :=
  ( (0.5 + 0.25 + 0.125) / (0.5 * 0.25 * 0.125) * ((7 / 18 * (9 / 2) + 1 / 6) / (13 + 1 / 3 - (15 / 4 * 16 / 5))) = 80.5 )

theorem problem_correct : decimal_to_fraction_eq_80_5 :=
  sorry

end NUMINAMATH_GPT_problem_correct_l831_83116


namespace NUMINAMATH_GPT_find_theta_l831_83110

open Real

theorem find_theta (theta : ℝ) : sin theta = -1/3 ∧ -π < theta ∧ theta < -π / 2 ↔ theta = -π - arcsin (-1 / 3) :=
by
  sorry

end NUMINAMATH_GPT_find_theta_l831_83110


namespace NUMINAMATH_GPT_fourth_even_integer_l831_83143

theorem fourth_even_integer (n : ℤ) (h : (n-2) + (n+2) = 92) : n + 4 = 50 := by
  -- This will skip the proof steps and assume the correct answer
  sorry

end NUMINAMATH_GPT_fourth_even_integer_l831_83143


namespace NUMINAMATH_GPT_two_class_students_l831_83174

-- Define the types of students and total sum variables
variables (H M E HM HE ME HME : ℕ)
variable (Total_Students : ℕ)

-- Given conditions
axiom condition1 : Total_Students = 68
axiom condition2 : H = 19
axiom condition3 : M = 14
axiom condition4 : E = 26
axiom condition5 : HME = 3

-- Inclusion-Exclusion principle formula application
def exactly_two_classes : Prop := 
  Total_Students = H + M + E - (HM + HE + ME) + HME

-- Theorem to prove the number of students registered for exactly two classes is 6
theorem two_class_students : H + M + E - 2 * HME + HME - (HM + HE + ME) = 6 := by
  sorry

end NUMINAMATH_GPT_two_class_students_l831_83174


namespace NUMINAMATH_GPT_trains_crossing_time_l831_83120

theorem trains_crossing_time (length : ℕ) (time1 time2 : ℕ) (h1 : length = 120) (h2 : time1 = 10) (h3 : time2 = 20) :
  (2 * length : ℚ) / (length / time1 + length / time2 : ℚ) = 13.33 :=
by
  sorry

end NUMINAMATH_GPT_trains_crossing_time_l831_83120


namespace NUMINAMATH_GPT_sam_dimes_now_l831_83121

-- Define the initial number of dimes Sam had
def initial_dimes : ℕ := 9

-- Define the number of dimes Sam gave away
def dimes_given : ℕ := 7

-- State the theorem: The number of dimes Sam has now is 2
theorem sam_dimes_now : initial_dimes - dimes_given = 2 := by
  sorry

end NUMINAMATH_GPT_sam_dimes_now_l831_83121


namespace NUMINAMATH_GPT_root_division_simplification_l831_83164

theorem root_division_simplification (a : ℝ) (h1 : a = (7 : ℝ)^(1/4)) (h2 : a = (7 : ℝ)^(1/7)) :
  ((7 : ℝ)^(1/4) / (7 : ℝ)^(1/7)) = (7 : ℝ)^(3/28) :=
sorry

end NUMINAMATH_GPT_root_division_simplification_l831_83164


namespace NUMINAMATH_GPT_number_minus_29_l831_83171

theorem number_minus_29 (x : ℕ) (h : x - 46 = 15) : x - 29 = 32 :=
sorry

end NUMINAMATH_GPT_number_minus_29_l831_83171


namespace NUMINAMATH_GPT_seq_general_form_l831_83178

theorem seq_general_form (p r : ℝ) (a : ℕ → ℝ)
  (hp : p > r)
  (hr : r > 0)
  (h_init : a 1 = r)
  (h_recurrence : ∀ n : ℕ, a (n+1) = p * a n + r^(n+1)) :
  ∀ n : ℕ, a n = r * (p^n - r^n) / (p - r) :=
by
  sorry

end NUMINAMATH_GPT_seq_general_form_l831_83178


namespace NUMINAMATH_GPT_ratio_boys_to_girls_l831_83102

theorem ratio_boys_to_girls (g b : ℕ) (h1 : g + b = 30) (h2 : b = g + 3) : 
  (b : ℚ) / g = 16 / 13 := 
by 
  sorry

end NUMINAMATH_GPT_ratio_boys_to_girls_l831_83102


namespace NUMINAMATH_GPT_volume_and_surface_area_implies_sum_of_edges_l831_83153

-- Define the problem conditions and prove the required statement
theorem volume_and_surface_area_implies_sum_of_edges :
  ∃ (a r : ℝ), 
    (a / r) * a * (a * r) = 216 ∧ 
    2 * ((a^2 / r) + a^2 * r + a^2) = 288 →
    4 * ((a / r) + a * r + a) = 96 :=
by
  sorry

end NUMINAMATH_GPT_volume_and_surface_area_implies_sum_of_edges_l831_83153


namespace NUMINAMATH_GPT_number_of_n_for_prime_l831_83114

theorem number_of_n_for_prime (n : ℕ) : (n > 0) → ∃! n, Nat.Prime (n * (n + 2)) :=
by 
  sorry

end NUMINAMATH_GPT_number_of_n_for_prime_l831_83114


namespace NUMINAMATH_GPT_find_m_of_odd_number_sequence_l831_83199

theorem find_m_of_odd_number_sequence : 
  ∃ m : ℕ, m > 1 ∧ (∃ a : ℕ, a = m * (m - 1) + 1 ∧ a = 2023) ↔ m = 45 :=
by
    sorry

end NUMINAMATH_GPT_find_m_of_odd_number_sequence_l831_83199


namespace NUMINAMATH_GPT_determine_s_l831_83118

def g (x s : ℝ) : ℝ := 3 * x^4 + 2 * x^3 - x^2 - 4 * x + s

theorem determine_s (s : ℝ) (h : g (-3) s = 0) : s = -192 :=
by
  sorry

end NUMINAMATH_GPT_determine_s_l831_83118


namespace NUMINAMATH_GPT_at_least_one_not_less_than_two_l831_83157

theorem at_least_one_not_less_than_two (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  a + (1 / b) ≥ 2 ∨ b + (1 / a) ≥ 2 :=
by
  sorry

end NUMINAMATH_GPT_at_least_one_not_less_than_two_l831_83157


namespace NUMINAMATH_GPT_actual_distance_traveled_l831_83151

theorem actual_distance_traveled 
  (D : ℝ)
  (h1 : ∃ (D : ℝ), D/12 = (D + 36)/20)
  : D = 54 :=
sorry

end NUMINAMATH_GPT_actual_distance_traveled_l831_83151


namespace NUMINAMATH_GPT_speed_of_second_car_l831_83141

theorem speed_of_second_car (s1 s2 s : ℕ) (v1 : ℝ) (h_s1 : s1 = 500) (h_s2 : s2 = 700) 
  (h_s : s = 100) (h_v1 : v1 = 10) : 
  (∃ v2 : ℝ, v2 = 12 ∨ v2 = 16) :=
by 
  sorry

end NUMINAMATH_GPT_speed_of_second_car_l831_83141


namespace NUMINAMATH_GPT_triangle_area_and_coordinates_l831_83150

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

end NUMINAMATH_GPT_triangle_area_and_coordinates_l831_83150


namespace NUMINAMATH_GPT_vectors_perpendicular_vector_combination_l831_83168

def vector_a : ℝ × ℝ := (2, -1)
def vector_b : ℝ × ℝ := (-3, 2)
def vector_c : ℝ × ℝ := (1, 1)

-- Auxiliary definition of vector addition
def vector_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 + v2.1, v1.2 + v2.2)

-- Auxiliary definition of dot product
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := (v1.1 * v2.1 + v1.2 * v2.2)

-- Auxiliary definition of scalar multiplication
def scalar_mul (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)

-- Proof that (vector_a + vector_b) is perpendicular to vector_c
theorem vectors_perpendicular : dot_product (vector_add vector_a vector_b) vector_c = 0 :=
by sorry

-- Proof that vector_c = 5 * vector_a + 3 * vector_b
theorem vector_combination : vector_c = vector_add (scalar_mul 5 vector_a) (scalar_mul 3 vector_b) :=
by sorry

end NUMINAMATH_GPT_vectors_perpendicular_vector_combination_l831_83168


namespace NUMINAMATH_GPT_faye_rows_l831_83163

theorem faye_rows (total_pencils : ℕ) (pencils_per_row : ℕ) (rows_created : ℕ) :
  total_pencils = 12 → pencils_per_row = 4 → rows_created = 3 := by
  sorry

end NUMINAMATH_GPT_faye_rows_l831_83163


namespace NUMINAMATH_GPT_curve_symmetry_l831_83172

-- Define the curve equation
def curve_eq (x y : ℝ) : Prop := x * y^2 - x^2 * y = -2

-- Define the symmetry condition about the line y = -x
def symmetry_about_y_equals_neg_x (x y : ℝ) : Prop :=
  curve_eq (-y) (-x)

-- Define the original curve equation
def original_curve (x y : ℝ) : Prop := curve_eq x y

-- Proof statement: The curve xy^2 - x^2y = -2 is symmetric about the line y = -x.
theorem curve_symmetry : ∀ (x y : ℝ), original_curve x y ↔ symmetry_about_y_equals_neg_x x y :=
by
  sorry

end NUMINAMATH_GPT_curve_symmetry_l831_83172


namespace NUMINAMATH_GPT_lcm_two_numbers_l831_83147

theorem lcm_two_numbers (a b : ℕ) (h1 : a * b = 17820) (h2 : Nat.gcd a b = 12) : Nat.lcm a b = 1485 := 
by
  sorry

end NUMINAMATH_GPT_lcm_two_numbers_l831_83147


namespace NUMINAMATH_GPT_fraction_division_l831_83107

theorem fraction_division : (3 / 4) / (2 / 5) = 15 / 8 := by
  sorry

end NUMINAMATH_GPT_fraction_division_l831_83107


namespace NUMINAMATH_GPT_intersection_eq_l831_83192

def set_M : Set ℝ := { x : ℝ | (x + 3) * (x - 2) < 0 }
def set_N : Set ℝ := { x : ℝ | 1 ≤ x ∧ x ≤ 3 }

theorem intersection_eq : set_M ∩ set_N = { x : ℝ | 1 ≤ x ∧ x < 2 } := by
  sorry

end NUMINAMATH_GPT_intersection_eq_l831_83192


namespace NUMINAMATH_GPT_sales_worth_l831_83140

variables (S : ℝ)
variables (old_scheme_remuneration new_scheme_remuneration : ℝ)

def old_scheme := 0.05 * S
def new_scheme := 1300 + 0.025 * (S - 4000)

theorem sales_worth :
  new_scheme S = old_scheme S + 600 →
  S = 24000 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_sales_worth_l831_83140


namespace NUMINAMATH_GPT_min_value_of_z_ineq_l831_83123

noncomputable def z (x y : ℝ) : ℝ := 2 * x + 4 * y

theorem min_value_of_z_ineq (k : ℝ) :
  (∃ x y : ℝ, (3 * x + y ≥ 0) ∧ (4 * x + 3 * y ≥ k) ∧ (z x y = -6)) ↔ k = 0 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_z_ineq_l831_83123


namespace NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l831_83195

-- Problem 1
theorem problem1 : (-3 : ℝ) ^ 2 + (1 / 2) ^ (-1 : ℝ) + (Real.pi - 3) ^ 0 = 12 :=
by
  sorry

-- Problem 2
theorem problem2 (x : ℝ) : (8 * x ^ 4 + 4 * x ^ 3 - x ^ 2) / (-2 * x) ^ 2 = 2 * x ^ 2 + x - 1 / 4 :=
by
  sorry

-- Problem 3
theorem problem3 (x : ℝ) : (2 * x + 1) ^ 2 - (4 * x + 1) * (x + 1) = -x :=
by
  sorry

-- Problem 4
theorem problem4 (x y : ℝ) : (x + 2 * y - 3) * (x - 2 * y + 3) = x ^ 2 - 4 * y ^ 2 + 12 * y - 9 :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l831_83195


namespace NUMINAMATH_GPT_farmer_has_11_goats_l831_83165

theorem farmer_has_11_goats
  (pigs cows goats : ℕ)
  (h1 : pigs = 2 * cows)
  (h2 : cows = goats + 4)
  (h3 : goats + cows + pigs = 56) :
  goats = 11 := by
  sorry

end NUMINAMATH_GPT_farmer_has_11_goats_l831_83165


namespace NUMINAMATH_GPT_bugs_meeting_time_l831_83127

/-- Two circles with radii 7 inches and 3 inches are tangent at a point P. 
Two bugs start crawling at the same time from point P, one along the larger circle 
at 4π inches per minute, and the other along the smaller circle at 3π inches per minute. 
Prove they will meet again after 14 minutes and determine how far each has traveled.

The bug on the larger circle will have traveled 28π inches.
The bug on the smaller circle will have traveled 42π inches.
-/
theorem bugs_meeting_time
  (r₁ r₂ : ℝ) (v₁ v₂ : ℝ)
  (h₁ : r₁ = 7) (h₂ : r₂ = 3) 
  (h₃ : v₁ = 4 * Real.pi) (h₄ : v₂ = 3 * Real.pi) :
  ∃ t d₁ d₂, t = 14 ∧ d₁ = 28 * Real.pi ∧ d₂ = 42 * Real.pi := by
  sorry

end NUMINAMATH_GPT_bugs_meeting_time_l831_83127


namespace NUMINAMATH_GPT_each_boy_makes_14_l831_83132

/-- Proof that each boy makes 14 dollars given the initial conditions and sales scheme. -/
theorem each_boy_makes_14 (victor_shrimp : ℕ)
                          (austin_shrimp : ℕ)
                          (brian_shrimp : ℕ)
                          (total_shrimp : ℕ)
                          (sets_sold : ℕ)
                          (total_earnings : ℕ)
                          (individual_earnings : ℕ)
                          (h1 : victor_shrimp = 26)
                          (h2 : austin_shrimp = victor_shrimp - 8)
                          (h3 : brian_shrimp = (victor_shrimp + austin_shrimp) / 2)
                          (h4 : total_shrimp = victor_shrimp + austin_shrimp + brian_shrimp)
                          (h5 : sets_sold = total_shrimp / 11)
                          (h6 : total_earnings = sets_sold * 7)
                          (h7 : individual_earnings = total_earnings / 3):
  individual_earnings = 14 := 
by
  sorry

end NUMINAMATH_GPT_each_boy_makes_14_l831_83132


namespace NUMINAMATH_GPT_zyka_expense_increase_l831_83180

theorem zyka_expense_increase (C_k C_c : ℝ) (h1 : 0.5 * C_k = 0.2 * C_c) : 
  (((1.2 * C_c) - C_c) / C_c) * 100 = 20 := by
  sorry

end NUMINAMATH_GPT_zyka_expense_increase_l831_83180


namespace NUMINAMATH_GPT_cannot_be_2009_l831_83183

theorem cannot_be_2009 (a b c : ℕ) (h : b * 1234^2 + c * 1234 + a = c * 1234^2 + a * 1234 + b) : (b * 1^2 + c * 1 + a ≠ 2009) :=
by
  sorry

end NUMINAMATH_GPT_cannot_be_2009_l831_83183


namespace NUMINAMATH_GPT_oakwood_team_count_l831_83124

theorem oakwood_team_count :
  let girls := 5
  let boys := 7
  let choose_3_girls := Nat.choose girls 3
  let choose_2_boys := Nat.choose boys 2
  choose_3_girls * choose_2_boys = 210 := by
sorry

end NUMINAMATH_GPT_oakwood_team_count_l831_83124


namespace NUMINAMATH_GPT_change_received_l831_83103

def cost_per_banana_cents : ℕ := 30
def cost_per_banana_dollars : ℝ := 0.30
def number_of_bananas : ℕ := 5
def total_paid_dollars : ℝ := 10.00

def total_cost (cost_per_banana_dollars : ℝ) (number_of_bananas : ℕ) : ℝ :=
  cost_per_banana_dollars * number_of_bananas

theorem change_received :
  total_paid_dollars - total_cost cost_per_banana_dollars number_of_bananas = 8.50 :=
by
  sorry

end NUMINAMATH_GPT_change_received_l831_83103


namespace NUMINAMATH_GPT_length_of_other_side_l831_83112

-- Defining the conditions
def roofs := 3
def sides_per_roof := 2
def length_of_one_side := 40 -- measured in feet
def shingles_per_square_foot := 8
def total_shingles := 38400

-- The proof statement
theorem length_of_other_side : 
    ∃ (L : ℕ), (total_shingles / shingles_per_square_foot / roofs / sides_per_roof = 40 * L) ∧ L = 20 :=
by
  sorry

end NUMINAMATH_GPT_length_of_other_side_l831_83112


namespace NUMINAMATH_GPT_no_solution_ineq_positive_exponents_l831_83104

theorem no_solution_ineq (m : ℝ) (h : m < 6) : ¬∃ x : ℝ, |x + 1| + |x - 5| ≤ m := 
sorry

theorem positive_exponents (a b : ℝ) (ha_pos : 0 < a) (hb_pos : 0 < b) (h_neq : a ≠ b) : a^a * b^b - a^b * b^a > 0 := 
sorry

end NUMINAMATH_GPT_no_solution_ineq_positive_exponents_l831_83104


namespace NUMINAMATH_GPT_is_inverse_g1_is_inverse_g2_l831_83188

noncomputable def f (x : ℝ) := 3 + 2*x - x^2

noncomputable def g1 (x : ℝ) := -1 + Real.sqrt (4 - x)
noncomputable def g2 (x : ℝ) := -1 - Real.sqrt (4 - x)

theorem is_inverse_g1 : ∀ x, f (g1 x) = x :=
by
  intro x
  sorry

theorem is_inverse_g2 : ∀ x, f (g2 x) = x :=
by
  intro x
  sorry

end NUMINAMATH_GPT_is_inverse_g1_is_inverse_g2_l831_83188


namespace NUMINAMATH_GPT_households_using_all_three_brands_correct_l831_83194

noncomputable def total_households : ℕ := 5000
noncomputable def non_users : ℕ := 1200
noncomputable def only_X : ℕ := 800
noncomputable def only_Y : ℕ := 600
noncomputable def only_Z : ℕ := 300

-- Let A be the number of households that used all three brands of soap
variable (A : ℕ)

-- For every household that used all three brands, 5 used only two brands and 10 used just one brand.
-- Number of households that used only two brands = 5 * A
-- Number of households that used only one brand = 10 * A

-- The equation for households that used just one brand:
def households_using_all_three_brands :=
10 * A = only_X + only_Y + only_Z

theorem households_using_all_three_brands_correct :
  (total_households - non_users = only_X + only_Y + only_Z + 5 * A + 10 * A) →
  (A = 170) := by
sorry

end NUMINAMATH_GPT_households_using_all_three_brands_correct_l831_83194


namespace NUMINAMATH_GPT_participants_neither_coffee_nor_tea_l831_83156

-- Define the total number of participants
def total_participants : ℕ := 30

-- Define the number of participants who drank coffee
def coffee_drinkers : ℕ := 15

-- Define the number of participants who drank tea
def tea_drinkers : ℕ := 18

-- Define the number of participants who drank both coffee and tea
def both_drinkers : ℕ := 8

-- The proof statement for the number of participants who drank neither coffee nor tea
theorem participants_neither_coffee_nor_tea :
  total_participants - (coffee_drinkers + tea_drinkers - both_drinkers) = 5 := by
  sorry

end NUMINAMATH_GPT_participants_neither_coffee_nor_tea_l831_83156


namespace NUMINAMATH_GPT_find_unknown_rate_l831_83106

def blankets_cost (num : ℕ) (rate : ℕ) (discount_tax : ℕ) (is_discount : Bool) : ℕ :=
  if is_discount then rate * (100 - discount_tax) / 100 * num
  else (rate * (100 + discount_tax) / 100) * num

def total_cost := blankets_cost 3 100 10 true +
                  blankets_cost 4 150 0 false +
                  blankets_cost 3 200 20 false

def avg_cost (total : ℕ) (num : ℕ) : ℕ :=
  total / num

theorem find_unknown_rate
  (unknown_rate : ℕ)
  (h1 : total_cost + 2 * unknown_rate = 1800)
  (h2 : avg_cost (total_cost + 2 * unknown_rate) 12 = 150) :
  unknown_rate = 105 :=
by
  sorry

end NUMINAMATH_GPT_find_unknown_rate_l831_83106


namespace NUMINAMATH_GPT_sara_height_correct_l831_83126

variable (Roy_height : ℕ)
variable (Joe_height : ℕ)
variable (Sara_height : ℕ)

def problem_conditions (Roy_height Joe_height Sara_height : ℕ) : Prop :=
  Roy_height = 36 ∧
  Joe_height = Roy_height + 3 ∧
  Sara_height = Joe_height + 6

theorem sara_height_correct (Roy_height Joe_height Sara_height : ℕ) :
  problem_conditions Roy_height Joe_height Sara_height → Sara_height = 45 := by
  sorry

end NUMINAMATH_GPT_sara_height_correct_l831_83126


namespace NUMINAMATH_GPT_cost_of_ice_cream_l831_83159

theorem cost_of_ice_cream (x : ℝ) (h1 : 10 * x = 40) : x = 4 :=
by sorry

end NUMINAMATH_GPT_cost_of_ice_cream_l831_83159


namespace NUMINAMATH_GPT_smallest_value_in_geometric_progression_l831_83152

open Real

theorem smallest_value_in_geometric_progression 
  (d : ℝ) : 
  (∀ a b c d : ℝ, 
    a = 5 ∧ b = 5 + d ∧ c = 5 + 2 * d ∧ d = 5 + 3 * d ∧ 
    ∀ a' b' c' d' : ℝ, 
      a' = 5 ∧ b' = 6 + d ∧ c' = 15 + 2 * d ∧ d' = 3 * d ∧ 
      (b' / a' = c' / b' ∧ c' / b' = d' / c')) → 
  (d = (-1 + 4 * sqrt 10) ∨ d = (-1 - 4 * sqrt 10)) → 
  (min (3 * (-1 + 4 * sqrt 10)) (3 * (-1 - 4 * sqrt 10)) = -3 - 12 * sqrt 10) :=
by
  intros ha hd
  sorry

end NUMINAMATH_GPT_smallest_value_in_geometric_progression_l831_83152


namespace NUMINAMATH_GPT_household_peak_consumption_l831_83187

theorem household_peak_consumption
  (p_orig p_peak p_offpeak : ℝ)
  (consumption : ℝ)
  (monthly_savings : ℝ)
  (x : ℝ)
  (h_orig : p_orig = 0.52)
  (h_peak : p_peak = 0.55)
  (h_offpeak : p_offpeak = 0.35)
  (h_consumption : consumption = 200)
  (h_savings : monthly_savings = 0.10) :
  (p_orig - p_peak) * x + (p_orig - p_offpeak) * (consumption - x) ≥ p_orig * consumption * monthly_savings → x ≤ 118 :=
sorry

end NUMINAMATH_GPT_household_peak_consumption_l831_83187


namespace NUMINAMATH_GPT_min_value_f1_min_value_f1_achieved_max_value_f2_max_value_f2_achieved_l831_83148

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

end NUMINAMATH_GPT_min_value_f1_min_value_f1_achieved_max_value_f2_max_value_f2_achieved_l831_83148


namespace NUMINAMATH_GPT_tradesman_gain_on_outlay_l831_83135

-- Define the percentage defrauded and the percentage gain in both buying and selling
def defraud_percent := 20
def original_value := 100
def buying_price := original_value * (1 - (defraud_percent / 100))
def selling_price := original_value * (1 + (defraud_percent / 100))
def gain := selling_price - buying_price
def gain_percent := (gain / buying_price) * 100

theorem tradesman_gain_on_outlay :
  gain_percent = 50 := 
sorry

end NUMINAMATH_GPT_tradesman_gain_on_outlay_l831_83135


namespace NUMINAMATH_GPT_aquarium_water_l831_83119

theorem aquarium_water (T1 T2 T3 T4 : ℕ) (g w : ℕ) (hT1 : T1 = 8) (hT2 : T2 = 8) (hT3 : T3 = 6) (hT4 : T4 = 6):
  (g = T1 + T2 + T3 + T4) → (w = g * 4) → w = 112 :=
by
  sorry

end NUMINAMATH_GPT_aquarium_water_l831_83119


namespace NUMINAMATH_GPT_cos_alpha_minus_pi_over_4_l831_83131

theorem cos_alpha_minus_pi_over_4 (α : ℝ) (hα1 : 0 < α) (hα2 : α < π / 2) (hα3 : Real.tan α = 2) :
  Real.cos (α - π / 4) = 3 * Real.sqrt 10 / 10 :=
sorry

end NUMINAMATH_GPT_cos_alpha_minus_pi_over_4_l831_83131


namespace NUMINAMATH_GPT_drums_per_day_l831_83122

theorem drums_per_day (total_drums : Nat) (days : Nat) (total_drums_eq : total_drums = 6264) (days_eq : days = 58) :
  total_drums / days = 108 :=
by
  sorry

end NUMINAMATH_GPT_drums_per_day_l831_83122


namespace NUMINAMATH_GPT_impossible_odd_sum_l831_83193

theorem impossible_odd_sum (n m : ℤ) (h1 : (n^3 + m^3) % 2 = 0) (h2 : (n^3 + m^3) % 4 = 0) : (n + m) % 2 = 0 :=
sorry

end NUMINAMATH_GPT_impossible_odd_sum_l831_83193


namespace NUMINAMATH_GPT_ratio_Lisa_Claire_l831_83130

-- Definitions
def Claire_photos : ℕ := 6
def Robert_photos : ℕ := Claire_photos + 12
def Lisa_photos : ℕ := Robert_photos

-- Theorem statement
theorem ratio_Lisa_Claire : (Lisa_photos : ℚ) / (Claire_photos : ℚ) = 3 / 1 :=
by
  sorry

end NUMINAMATH_GPT_ratio_Lisa_Claire_l831_83130


namespace NUMINAMATH_GPT_geometric_sequence_common_ratio_l831_83179

theorem geometric_sequence_common_ratio :
  ∀ (a : ℕ → ℝ) (q : ℝ),
  (∀ n, a (n + 1) = a n * q) →
  (∀ n m, n < m → a n < a m) →
  a 2 = 2 →
  a 4 - a 3 = 4 →
  q = 2 :=
by
  intros a q h_geo h_inc h_a2 h_a4_a3
  sorry

end NUMINAMATH_GPT_geometric_sequence_common_ratio_l831_83179


namespace NUMINAMATH_GPT_find_interest_rate_l831_83139

noncomputable def compound_interest_rate (A P : ℝ) (t n : ℕ) : ℝ := sorry

theorem find_interest_rate :
  compound_interest_rate 676 625 2 1 = 0.04 := 
sorry

end NUMINAMATH_GPT_find_interest_rate_l831_83139


namespace NUMINAMATH_GPT_max_marks_l831_83181

variable (M : ℝ)

-- Conditions
def needed_to_pass (M : ℝ) := 0.20 * M
def pradeep_marks := 390
def marks_short := 25
def total_marks_needed := pradeep_marks + marks_short

-- Theorem statement
theorem max_marks : needed_to_pass M = total_marks_needed → M = 2075 := by
  sorry

end NUMINAMATH_GPT_max_marks_l831_83181


namespace NUMINAMATH_GPT_corrected_mean_is_45_55_l831_83185

-- Define the initial conditions
def mean_of_100_observations (mean : ℝ) : Prop :=
  mean = 45

def incorrect_observation : ℝ := 32
def correct_observation : ℝ := 87

-- Define the calculation of the corrected mean
noncomputable def corrected_mean (incorrect_mean : ℝ) (incorrect_obs : ℝ) (correct_obs : ℝ) (n : ℕ) : ℝ :=
  let sum_original := incorrect_mean * n
  let difference := correct_obs - incorrect_obs
  (sum_original + difference) / n

-- Theorem: The corrected new mean is 45.55
theorem corrected_mean_is_45_55 : corrected_mean 45 32 87 100 = 45.55 :=
by
  sorry

end NUMINAMATH_GPT_corrected_mean_is_45_55_l831_83185


namespace NUMINAMATH_GPT_speed_of_stream_l831_83144

theorem speed_of_stream :
  ∃ (v : ℝ), (∀ (swim_speed : ℝ), swim_speed = 1.5 → 
    (∀ (time_upstream : ℝ) (time_downstream : ℝ), 
      time_upstream = 2 * time_downstream → 
      (1.5 + v) / (1.5 - v) = 2)) → v = 0.5 :=
sorry

end NUMINAMATH_GPT_speed_of_stream_l831_83144


namespace NUMINAMATH_GPT_points_satisfy_diamond_eq_l831_83133

noncomputable def diamond (a b : ℝ) : ℝ := a^3 * b - a * b^3

theorem points_satisfy_diamond_eq (x y : ℝ) :
  (diamond x y = diamond y x) ↔ ((x = 0) ∨ (y = 0) ∨ (x = y) ∨ (x = -y)) := 
by
  sorry

end NUMINAMATH_GPT_points_satisfy_diamond_eq_l831_83133


namespace NUMINAMATH_GPT_no_opposite_identical_numbers_l831_83198

open Finset

theorem no_opposite_identical_numbers : 
  ∀ (f g : Fin 20 → Fin 20), 
  (∀ i : Fin 20, ∃ j : Fin 20, f j = i ∧ g j = (i + j) % 20) → 
  ∃ k : ℤ, ∀ i : Fin 20, f (i + k) % 20 ≠ g i 
  := by
    sorry

end NUMINAMATH_GPT_no_opposite_identical_numbers_l831_83198


namespace NUMINAMATH_GPT_otimes_eq_abs_m_leq_m_l831_83184

noncomputable def otimes (x y : ℝ) : ℝ :=
if x ≤ y then x else y

theorem otimes_eq_abs_m_leq_m' :
  ∀ (m : ℝ), otimes (abs (m - 1)) m = abs (m - 1) → m ∈ Set.Ici (1 / 2) := 
by
  sorry

end NUMINAMATH_GPT_otimes_eq_abs_m_leq_m_l831_83184


namespace NUMINAMATH_GPT_triangle_ABC_properties_l831_83125

open Real

theorem triangle_ABC_properties
  (a b c : ℝ) 
  (A B C : ℝ) 
  (A_eq : A = π / 3) 
  (b_eq : b = sqrt 2) 
  (cond1 : b^2 + sqrt 2 * a * c = a^2 + c^2) 
  (cond2 : a * cos B = b * sin A) 
  (cond3 : sin B + cos B = sqrt 2) : 
  B = π / 4 ∧ (1 / 2) * a * b * sin (π - A - B) = (3 + sqrt 3) / 4 := 
by 
  sorry

end NUMINAMATH_GPT_triangle_ABC_properties_l831_83125


namespace NUMINAMATH_GPT_time_difference_l831_83173

-- Define the capacity of the tanks
def capacity : ℕ := 20

-- Define the inflow rates of tanks A and B in litres per hour
def inflow_rate_A : ℕ := 2
def inflow_rate_B : ℕ := 4

-- Define the times to fill tanks A and B
def time_A : ℕ := capacity / inflow_rate_A
def time_B : ℕ := capacity / inflow_rate_B

-- Proving the time difference between filling tanks A and B
theorem time_difference : (time_A - time_B) = 5 := by
  sorry

end NUMINAMATH_GPT_time_difference_l831_83173


namespace NUMINAMATH_GPT_digit_divisibility_by_7_l831_83138

theorem digit_divisibility_by_7 (d : ℕ) (h : d < 10) : (10000 + 100 * d + 10) % 7 = 0 ↔ d = 5 :=
by
  sorry

end NUMINAMATH_GPT_digit_divisibility_by_7_l831_83138


namespace NUMINAMATH_GPT_successfully_served_pizzas_l831_83161

-- Defining the conditions
def total_pizzas_served : ℕ := 9
def pizzas_returned : ℕ := 6

-- Stating the theorem
theorem successfully_served_pizzas :
  total_pizzas_served - pizzas_returned = 3 :=
by
  -- Since this is only the statement, the proof is omitted using sorry
  sorry

end NUMINAMATH_GPT_successfully_served_pizzas_l831_83161


namespace NUMINAMATH_GPT_find_a_for_parallel_lines_l831_83169

theorem find_a_for_parallel_lines (a : ℝ) :
  (∀ x y : ℝ, ax + 3 * y + 1 = 0 ↔ 2 * x + (a + 1) * y + 1 = 0) → a = -3 :=
by
  sorry

end NUMINAMATH_GPT_find_a_for_parallel_lines_l831_83169


namespace NUMINAMATH_GPT_repeatingDecimals_fraction_eq_l831_83145

noncomputable def repeatingDecimalsSum : ℚ :=
  let x : ℚ := 1 / 3
  let y : ℚ := 4 / 99
  let z : ℚ := 5 / 999
  x + y + z

theorem repeatingDecimals_fraction_eq : repeatingDecimalsSum = 42 / 111 :=
  sorry

end NUMINAMATH_GPT_repeatingDecimals_fraction_eq_l831_83145


namespace NUMINAMATH_GPT_intersection_line_eq_l831_83190

-- Definitions of the circles
def circle1 (x y : ℝ) := x^2 + y^2 - 4*x - 6 = 0
def circle2 (x y : ℝ) := x^2 + y^2 - 4*y - 6 = 0

-- The theorem stating that the equation of the line passing through their intersection points is x = y
theorem intersection_line_eq (x y : ℝ) :
  (circle1 x y → circle2 x y → x = y) := 
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_intersection_line_eq_l831_83190


namespace NUMINAMATH_GPT_intersection_domains_l831_83154

def domain_f : Set ℝ := {x : ℝ | x < 1}
def domain_g : Set ℝ := {x : ℝ | x > -1}

theorem intersection_domains : {x : ℝ | x < 1} ∩ {x : ℝ | x > -1} = {x : ℝ | -1 < x ∧ x < 1} := by
  sorry

end NUMINAMATH_GPT_intersection_domains_l831_83154


namespace NUMINAMATH_GPT_range_neg2a_plus_3_l831_83186

theorem range_neg2a_plus_3 (a : ℝ) (h : a < 1) : -2 * a + 3 > 1 :=
sorry

end NUMINAMATH_GPT_range_neg2a_plus_3_l831_83186


namespace NUMINAMATH_GPT_find_x_l831_83149

def a : ℝ × ℝ := (2, 3)
def b (x : ℝ) : ℝ × ℝ := (4, x)

theorem find_x (x : ℝ) (h : ∃k : ℝ, b x = (k * a.1, k * a.2)) : x = 6 := 
by 
  sorry

end NUMINAMATH_GPT_find_x_l831_83149


namespace NUMINAMATH_GPT_unique_valid_number_l831_83196

-- Define the form of the three-digit number.
def is_form_sixb5 (n : ℕ) : Prop :=
  ∃ b : ℕ, b < 10 ∧ n = 600 + 10 * b + 5

-- Define the condition for divisibility by 11.
def is_divisible_by_11 (n : ℕ) : Prop :=
  (n % 11 = 0)

-- Define the alternating sum property for our specific number format.
def alternating_sum_cond (b : ℕ) : Prop :=
  (11 - b) % 11 = 0

-- The final proposition to be proved.
theorem unique_valid_number : ∃ n, is_form_sixb5 n ∧ is_divisible_by_11 n ∧ n = 605 :=
by {
  sorry
}

end NUMINAMATH_GPT_unique_valid_number_l831_83196


namespace NUMINAMATH_GPT_intersection_distance_l831_83189

noncomputable def distance_between_intersections (l : ℝ → ℝ → Prop) (C : ℝ → ℝ → Prop) : Prop :=
  ∃ A B : ℝ × ℝ, 
    l A.1 A.2 ∧ C A.1 A.2 ∧ l B.1 B.2 ∧ C B.1 B.2 ∧ 
    dist A B = Real.sqrt 6

def line_l (x y : ℝ) : Prop :=
  x - y + 1 = 0

def curve_C (x y : ℝ) : Prop :=
  ∃ θ : ℝ, x = Real.sqrt 2 * Real.cos θ ∧ y = Real.sqrt 2 * Real.sin θ

theorem intersection_distance :
  distance_between_intersections line_l curve_C :=
sorry

end NUMINAMATH_GPT_intersection_distance_l831_83189


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_proof_l831_83134

theorem arithmetic_sequence_sum_proof
  (S : ℕ → ℕ) (a : ℕ → ℕ)
  (h1 : S 17 = 170)
  (h2 : a 2000 = 2001)
  (h3 : ∀ n, S n = n * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2)
  (h4 : ∀ n, a (n + 1) = a n + (a 2 - a 1)) :
  S 2008 = 2019044 :=
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_proof_l831_83134


namespace NUMINAMATH_GPT_compute_c_over_d_l831_83115

noncomputable def RootsResult (a b c d : ℝ) : Prop :=
  (3 * 4 + 4 * 5 + 5 * 3 = - c / a) ∧ (3 * 4 * 5 = - d / a)

theorem compute_c_over_d (a b c d : ℝ)
  (h1 : (a * 3 ^ 3 + b * 3 ^ 2 + c * 3 + d = 0))
  (h2 : (a * 4 ^ 3 + b * 4 ^ 2 + c * 4 + d = 0))
  (h3 : (a * 5 ^ 3 + b * 5 ^ 2 + c * 5 + d = 0)) 
  (hr : RootsResult a b c d) :
  c / d = 47 / 60 := 
by
  sorry

end NUMINAMATH_GPT_compute_c_over_d_l831_83115


namespace NUMINAMATH_GPT_max_elements_set_M_l831_83170

theorem max_elements_set_M (n : ℕ) (hn : n ≥ 2) (M : Finset (ℕ × ℕ))
  (hM : ∀ {i k}, (i, k) ∈ M → i < k → ∀ {m}, k < m → (k, m) ∉ M) :
  M.card ≤ n^2 / 4 :=
sorry

end NUMINAMATH_GPT_max_elements_set_M_l831_83170


namespace NUMINAMATH_GPT_decreasing_interval_l831_83162

noncomputable def y (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 15 * x^2 + 36 * x - 24

def has_extremum_at (a : ℝ) (x_ext : ℝ) : Prop :=
  deriv (y a) x_ext = 0

theorem decreasing_interval (a : ℝ) (h_extremum_at : has_extremum_at a 3) :
  a = 2 → ∀ x, (2 < x ∧ x < 3) → deriv (y a) x < 0 :=
sorry

end NUMINAMATH_GPT_decreasing_interval_l831_83162


namespace NUMINAMATH_GPT_find_positive_integer_solutions_l831_83136

def is_solution (x y : ℕ) : Prop :=
  4 * x^3 + 4 * x^2 * y - 15 * x * y^2 - 18 * y^3 - 12 * x^2 + 6 * x * y + 36 * y^2 + 5 * x - 10 * y = 0

theorem find_positive_integer_solutions :
  ∀ x y : ℕ, 0 < x ∧ 0 < y → (is_solution x y ↔ (x = 1 ∧ y = 1) ∨ (∃ y', y = y' ∧ x = 2 * y' ∧ 0 < y')) :=
by
  intros x y hxy
  sorry

end NUMINAMATH_GPT_find_positive_integer_solutions_l831_83136


namespace NUMINAMATH_GPT_average_cost_correct_l831_83166

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

end NUMINAMATH_GPT_average_cost_correct_l831_83166


namespace NUMINAMATH_GPT_log_sum_greater_than_two_l831_83155

variables {x y a m : ℝ}

theorem log_sum_greater_than_two
  (hx : 0 < x) (hxy : x < y) (hya : y < a) (ha1 : a < 1)
  (hm : m = Real.log x / Real.log a + Real.log y / Real.log a) : m > 2 :=
sorry

end NUMINAMATH_GPT_log_sum_greater_than_two_l831_83155


namespace NUMINAMATH_GPT_travel_time_at_constant_speed_l831_83128

theorem travel_time_at_constant_speed
  (distance : ℝ) (speed : ℝ) : 
  distance = 100 → speed = 20 → distance / speed = 5 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num

end NUMINAMATH_GPT_travel_time_at_constant_speed_l831_83128


namespace NUMINAMATH_GPT_smallest_integer_greater_than_power_l831_83146

theorem smallest_integer_greater_than_power (sqrt3 sqrt2 : ℝ) (h1 : (sqrt3 + sqrt2)^6 = 485 + 198 * Real.sqrt 6)
(h2 : (sqrt3 - sqrt2)^6 = 485 - 198 * Real.sqrt 6)
(h3 : 0 < (sqrt3 - sqrt2)^6 ∧ (sqrt3 - sqrt2)^6 < 1) : 
  ⌈(sqrt3 + sqrt2)^6⌉ = 970 := 
sorry

end NUMINAMATH_GPT_smallest_integer_greater_than_power_l831_83146


namespace NUMINAMATH_GPT_geometric_sequence_sum_l831_83182

theorem geometric_sequence_sum (a : Nat → ℝ) (q : ℝ) (h_geom : ∀ n, a (n + 1) = q * a n)
  (hq : q > 1) (h2011_root : 4 * a 2011 ^ 2 - 8 * a 2011 + 3 = 0)
  (h2012_root : 4 * a 2012 ^ 2 - 8 * a 2012 + 3 = 0) :
  a 2013 + a 2014 = 18 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l831_83182


namespace NUMINAMATH_GPT_smallest_n_square_average_l831_83137

theorem smallest_n_square_average (n : ℕ) (h : n > 1)
  (S : ℕ := (n * (n + 1) * (2 * n + 1)) / 6)
  (avg : ℕ := S / n) :
  (∃ k : ℕ, avg = k^2) → n = 337 := by
  sorry

end NUMINAMATH_GPT_smallest_n_square_average_l831_83137


namespace NUMINAMATH_GPT_smallest_checkered_rectangle_area_l831_83100

def even (n: ℕ) : Prop := n % 2 = 0

-- Both figure types are present and areas of these types are 1 and 2 respectively
def isValidPieceComposition (a b : ℕ) : Prop :=
  ∃ m n : ℕ, m > 0 ∧ n > 0 ∧ m * 1 + n * 2 = a * b

theorem smallest_checkered_rectangle_area :
  ∀ a b : ℕ, even a → even b → isValidPieceComposition a b → a * b ≥ 40 := 
by
  intro a b a_even b_even h_valid
  sorry

end NUMINAMATH_GPT_smallest_checkered_rectangle_area_l831_83100


namespace NUMINAMATH_GPT_increasing_intervals_l831_83108

noncomputable def f (x : ℝ) : ℝ := 4 * Real.sin (2 * x - Real.pi / 3)

theorem increasing_intervals :
  ∀ x : ℝ, x ∈ Set.Icc (-Real.pi) 0 →
    (f x > f (x - ε) ∧ f x < f (x + ε) ∧ x ∈ Set.Icc (-Real.pi) (-7 * Real.pi / 12) ∪ Set.Icc (-Real.pi / 12) 0) :=
sorry

end NUMINAMATH_GPT_increasing_intervals_l831_83108


namespace NUMINAMATH_GPT_arithmetic_seq_common_diff_l831_83101

theorem arithmetic_seq_common_diff
  (a : ℕ → ℝ) (d : ℝ)
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_d_nonzero : d ≠ 0)
  (h_a1 : a 1 = 1)
  (h_geomet : (a 3) ^ 2 = a 1 * a 13) :
  d = 2 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_seq_common_diff_l831_83101
