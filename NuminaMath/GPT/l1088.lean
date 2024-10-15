import Mathlib

namespace NUMINAMATH_GPT_A_beats_B_by_seconds_l1088_108809

theorem A_beats_B_by_seconds :
  ∀ (t_A : ℝ) (distance_A distance_B : ℝ),
  t_A = 156.67 →
  distance_A = 1000 →
  distance_B = 940 →
  (distance_A * t_A = 60 * (distance_A / t_A)) →
  t_A ≠ 0 →
  ((60 * t_A / distance_A) = 9.4002) :=
by
  intros t_A distance_A distance_B h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_A_beats_B_by_seconds_l1088_108809


namespace NUMINAMATH_GPT_cover_square_with_rectangles_l1088_108896

theorem cover_square_with_rectangles :
  ∃ (n : ℕ), 
    ∀ (a b : ℕ), 
      (a = 3) ∧ 
      (b = 4) ∧ 
      (n = (12 * 12) / (a * b)) ∧ 
      (144 = n * (a * b)) ∧ 
      (3 * 4 = a * b) 
  → 
    n = 12 :=
by
  sorry

end NUMINAMATH_GPT_cover_square_with_rectangles_l1088_108896


namespace NUMINAMATH_GPT_range_of_a_plus_b_l1088_108846

theorem range_of_a_plus_b (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : |Real.log a| = |Real.log b|) (h₄ : a ≠ b) :
  2 < a + b :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_plus_b_l1088_108846


namespace NUMINAMATH_GPT_Bob_walked_35_miles_l1088_108804

theorem Bob_walked_35_miles (distance : ℕ) 
  (Yolanda_rate Bob_rate : ℕ) (Bob_start_after : ℕ) (Yolanda_initial_walk : ℕ)
  (h1 : distance = 65) 
  (h2 : Yolanda_rate = 5) 
  (h3 : Bob_rate = 7) 
  (h4 : Bob_start_after = 1)
  (h5 : Yolanda_initial_walk = Yolanda_rate * Bob_start_after) :
  Bob_rate * (distance - Yolanda_initial_walk) / (Yolanda_rate + Bob_rate) = 35 := 
by 
  sorry

end NUMINAMATH_GPT_Bob_walked_35_miles_l1088_108804


namespace NUMINAMATH_GPT_new_ratio_after_adding_ten_l1088_108894

theorem new_ratio_after_adding_ten 
  (x : ℕ) 
  (h_ratio : 3 * x = 15) 
  (new_smaller : ℕ := x + 10) 
  (new_larger : ℕ := 15) 
  : new_smaller / new_larger = 1 :=
by sorry

end NUMINAMATH_GPT_new_ratio_after_adding_ten_l1088_108894


namespace NUMINAMATH_GPT_total_increase_area_l1088_108854

theorem total_increase_area (increase_broccoli increase_cauliflower increase_cabbage : ℕ)
    (area_broccoli area_cauliflower area_cabbage : ℝ)
    (h1 : increase_broccoli = 79)
    (h2 : increase_cauliflower = 25)
    (h3 : increase_cabbage = 50)
    (h4 : area_broccoli = 1)
    (h5 : area_cauliflower = 2)
    (h6 : area_cabbage = 1.5) :
    increase_broccoli * area_broccoli +
    increase_cauliflower * area_cauliflower +
    increase_cabbage * area_cabbage = 204 := 
by 
    sorry

end NUMINAMATH_GPT_total_increase_area_l1088_108854


namespace NUMINAMATH_GPT_Ryan_bike_time_l1088_108847

-- Definitions of the conditions
variables (B : ℕ)

-- Conditions
def bike_time := B
def bus_time := B + 10
def friend_time := B / 3
def commuting_time := bike_time B + 3 * bus_time B + friend_time B = 160

-- Goal to prove
theorem Ryan_bike_time : commuting_time B → B = 30 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_Ryan_bike_time_l1088_108847


namespace NUMINAMATH_GPT_total_candidates_l1088_108884

theorem total_candidates (T : ℝ) 
  (h1 : 0.45 * T = T * 0.45)
  (h2 : 0.38 * T = T * 0.38)
  (h3 : 0.22 * T = T * 0.22)
  (h4 : 0.12 * T = T * 0.12)
  (h5 : 0.09 * T = T * 0.09)
  (h6 : 0.10 * T = T * 0.10)
  (h7 : 0.05 * T = T * 0.05)
  (h_passed_english_alone : T - (0.45 * T - 0.12 * T - 0.10 * T + 0.05 * T) = 720) :
  T = 1000 :=
by
  sorry

end NUMINAMATH_GPT_total_candidates_l1088_108884


namespace NUMINAMATH_GPT_team_selection_l1088_108835

theorem team_selection (boys girls : ℕ) (choose_boys choose_girls : ℕ) 
  (boy_count girl_count : ℕ) (h1 : boy_count = 10) (h2 : girl_count = 12) 
  (h3 : choose_boys = 5) (h4 : choose_girls = 3) :
    (Nat.choose boy_count choose_boys) * (Nat.choose girl_count choose_girls) = 55440 :=
by
  rw [h1, h2, h3, h4]
  sorry

end NUMINAMATH_GPT_team_selection_l1088_108835


namespace NUMINAMATH_GPT_minimum_time_to_serve_tea_equals_9_l1088_108890

def boiling_water_time : Nat := 8
def washing_teapot_time : Nat := 1
def washing_teacups_time : Nat := 2
def fetching_tea_leaves_time : Nat := 2
def brewing_tea_time : Nat := 1

theorem minimum_time_to_serve_tea_equals_9 :
  boiling_water_time + brewing_tea_time = 9 := by
  sorry

end NUMINAMATH_GPT_minimum_time_to_serve_tea_equals_9_l1088_108890


namespace NUMINAMATH_GPT_sqrt_2700_minus_37_form_l1088_108853

theorem sqrt_2700_minus_37_form (a b : ℕ) (h1 : a > 0) (h2 : b > 0)
  (h3 : (Int.sqrt 2700 - 37) = Int.sqrt a - b ^ 3) : a + b = 13 :=
sorry

end NUMINAMATH_GPT_sqrt_2700_minus_37_form_l1088_108853


namespace NUMINAMATH_GPT_find_square_length_CD_l1088_108851

noncomputable def parabola (x : ℝ) : ℝ := 3 * x ^ 2 + 6 * x - 2

def is_midpoint (mid C D : (ℝ × ℝ)) : Prop :=
  mid.1 = (C.1 + D.1) / 2 ∧ mid.2 = (C.2 + D.2) / 2

theorem find_square_length_CD (C D : ℝ × ℝ)
  (hC : C.2 = parabola C.1)
  (hD : D.2 = parabola D.1)
  (h_mid : is_midpoint (0,0) C D) :
  (C.1 - D.1)^2 + (C.2 - D.2)^2 = 740 / 3 :=
sorry

end NUMINAMATH_GPT_find_square_length_CD_l1088_108851


namespace NUMINAMATH_GPT_find_x_l1088_108818

theorem find_x (x : ℤ) (h : 9873 + x = 13800) : x = 3927 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_x_l1088_108818


namespace NUMINAMATH_GPT_smallest_n_inequality_l1088_108819

theorem smallest_n_inequality : 
  ∃ (n : ℕ), (n > 0) ∧ ( ∀ m : ℕ, (m > 0) ∧ ( m < n ) → ¬( ( 1 : ℚ ) / m - ( 1 / ( m + 1 : ℚ ) ) < ( 1 / 15 ) ) ) ∧ ( ( 1 : ℚ ) / n - ( 1 / ( n + 1 : ℚ ) ) < ( 1 / 15 ) ) :=
sorry

end NUMINAMATH_GPT_smallest_n_inequality_l1088_108819


namespace NUMINAMATH_GPT_new_years_day_more_frequent_l1088_108828

-- Define conditions
def common_year_days : ℕ := 365
def leap_year_days : ℕ := 366
def century_is_leap_year (year : ℕ) : Prop := (year % 400 = 0)

-- Given: 23 October 1948 was a Saturday
def october_23_1948 : ℕ := 5 -- 5 corresponds to Saturday

-- Define the question proof statement
theorem new_years_day_more_frequent :
  (frequency_Sunday : ℕ) > (frequency_Monday : ℕ) :=
sorry

end NUMINAMATH_GPT_new_years_day_more_frequent_l1088_108828


namespace NUMINAMATH_GPT_solve_quadratic_equation_l1088_108830

theorem solve_quadratic_equation (x : ℝ) :
  (x^2 + 2 * x - 15 = 0) ↔ (x = 3 ∨ x = -5) :=
by
  sorry -- proof omitted

end NUMINAMATH_GPT_solve_quadratic_equation_l1088_108830


namespace NUMINAMATH_GPT_cafeteria_can_make_7_pies_l1088_108857

theorem cafeteria_can_make_7_pies (initial_apples handed_out apples_per_pie : ℕ)
  (h1 : initial_apples = 86)
  (h2 : handed_out = 30)
  (h3 : apples_per_pie = 8) :
  ((initial_apples - handed_out) / apples_per_pie) = 7 := 
by
  sorry

end NUMINAMATH_GPT_cafeteria_can_make_7_pies_l1088_108857


namespace NUMINAMATH_GPT_complete_job_days_l1088_108861

-- Variables and Conditions
variables (days_5_8 : ℕ) (days_1 : ℕ)

-- Assume that completing 5/8 of the job takes 10 days
def five_eighths_job_days := 10

-- Find days to complete one job at the same pace. 
-- This is the final statement we need to prove
theorem complete_job_days
  (h : 5 * days_1 = 8 * days_5_8) :
  days_1 = 16 := by
  -- Proof is omitted.
  sorry

end NUMINAMATH_GPT_complete_job_days_l1088_108861


namespace NUMINAMATH_GPT_quadratic_function_value_when_x_is_zero_l1088_108852

theorem quadratic_function_value_when_x_is_zero :
  (∃ h : ℝ, (∀ x : ℝ, x < -3 → (-(x + h)^2 < -(x + h + 1)^2)) ∧
            (∀ x : ℝ, x > -3 → (-(x + h)^2 > -(x + h - 1)^2)) ∧
            (y = -(0 + h)^2) → y = -9) := 
sorry

end NUMINAMATH_GPT_quadratic_function_value_when_x_is_zero_l1088_108852


namespace NUMINAMATH_GPT_sum_of_coefficients_l1088_108814

theorem sum_of_coefficients (x : ℝ) : 
  (1 - 2 * x) ^ 10 = 1 :=
sorry

end NUMINAMATH_GPT_sum_of_coefficients_l1088_108814


namespace NUMINAMATH_GPT_trig_identity_proof_l1088_108897

theorem trig_identity_proof 
  (α p q : ℝ)
  (hp : p ≠ 0) (hq : q ≠ 0)
  (tangent : Real.tan α = p / q) :
  Real.sin (2 * α) = 2 * p * q / (p^2 + q^2) ∧
  Real.cos (2 * α) = (q^2 - p^2) / (q^2 + p^2) ∧
  Real.tan (2 * α) = (2 * p * q) / (q^2 - p^2) :=
by
  sorry

end NUMINAMATH_GPT_trig_identity_proof_l1088_108897


namespace NUMINAMATH_GPT_count_twelfth_power_l1088_108845

-- Define the conditions under which a number must meet the criteria of being a square, a cube, and a fourth power
def is_square (n : ℕ) : Prop := ∃ m : ℕ, m^2 = n
def is_cube (n : ℕ) : Prop := ∃ m : ℕ, m^3 = n
def is_fourth_power (n : ℕ) : Prop := ∃ m : ℕ, m^4 = n

-- Define the main theorem, which proves the count of numbers less than 1000 meeting all criteria
theorem count_twelfth_power (h : ∀ n, is_square n → is_cube n → is_fourth_power n → n < 1000) :
  ∃! x : ℕ, x < 1000 ∧ ∃ k : ℕ, k^12 = x := 
sorry

end NUMINAMATH_GPT_count_twelfth_power_l1088_108845


namespace NUMINAMATH_GPT_cylinder_volume_l1088_108878

theorem cylinder_volume (r h : ℝ) (h_radius : r = 1) (h_height : h = 2) : (π * r^2 * h) = 2 * π :=
by
  sorry

end NUMINAMATH_GPT_cylinder_volume_l1088_108878


namespace NUMINAMATH_GPT_largest_n_S_n_positive_l1088_108806

-- We define the arithmetic sequence a_n.
def arith_seq (a_n : ℕ → ℝ) : Prop := 
  ∃ d : ℝ, ∀ n : ℕ, a_n (n + 1) = a_n n + d

-- Definitions for the conditions provided.
def first_term_positive (a_n : ℕ → ℝ) : Prop := 
  a_n 1 > 0

def term_sum_positive (a_n : ℕ → ℝ) : Prop :=
  a_n 2016 + a_n 2017 > 0

def term_product_negative (a_n : ℕ → ℝ) : Prop :=
  a_n 2016 * a_n 2017 < 0

-- Sum of the first n terms of an arithmetic sequence
noncomputable def sum_first_n_terms (a_n : ℕ → ℝ) (n : ℕ) : ℝ :=
  n * (a_n 1 + a_n n) / 2

-- Statement we want to prove in Lean 4.
theorem largest_n_S_n_positive (a_n : ℕ → ℝ) 
  (h_seq : arith_seq a_n) 
  (h1 : first_term_positive a_n) 
  (h2 : term_sum_positive a_n) 
  (h3 : term_product_negative a_n) : 
  ∀ n : ℕ, sum_first_n_terms a_n n > 0 → n ≤ 4032 := 
sorry

end NUMINAMATH_GPT_largest_n_S_n_positive_l1088_108806


namespace NUMINAMATH_GPT_geometric_sequence_l1088_108898

-- Define the set and its properties
variable (A : Set ℕ) (a : ℕ → ℕ) (n : ℕ)
variable (h1 : 1 ≤ a 1) 
variable (h2 : ∀ (i : ℕ), 1 ≤ i → i < n → a i < a (i + 1))
variable (h3 : n ≥ 5)
variable (h4 : ∀ (i j : ℕ), 1 ≤ i → i ≤ j → j ≤ n → (a i) * (a j) ∈ A ∨ (a i) / (a j) ∈ A)

-- Statement to prove that the sequence forms a geometric sequence
theorem geometric_sequence : 
  ∃ (c : ℕ), c > 1 ∧ ∀ (i : ℕ), 1 ≤ i → i ≤ n → a i = c^(i-1) := sorry

end NUMINAMATH_GPT_geometric_sequence_l1088_108898


namespace NUMINAMATH_GPT_value_of_a_squared_plus_b_squared_l1088_108871

theorem value_of_a_squared_plus_b_squared (a b : ℝ) 
  (h1 : (a + b) ^ 2 = 8) 
  (h2 : (a - b) ^ 2 = 12) : 
  a^2 + b^2 = 10 :=
sorry

end NUMINAMATH_GPT_value_of_a_squared_plus_b_squared_l1088_108871


namespace NUMINAMATH_GPT_intersection_A_B_eq_complement_union_eq_subset_condition_l1088_108808

open Set

noncomputable def A : Set ℝ := {x : ℝ | 1 ≤ x ∧ x ≤ 3}
noncomputable def B : Set ℝ := {x : ℝ | x > 3 / 2}
noncomputable def C (a : ℝ) : Set ℝ := {x : ℝ | 1 < x ∧ x < a}

theorem intersection_A_B_eq : A ∩ B = {x : ℝ | 3 / 2 < x ∧ x ≤ 3} :=
by sorry

theorem complement_union_eq : (univ \ B) ∪ A = {x : ℝ | x ≤ 3} :=
by sorry

theorem subset_condition (a : ℝ) : (C a ⊆ A) → (a ≤ 3) :=
by sorry

end NUMINAMATH_GPT_intersection_A_B_eq_complement_union_eq_subset_condition_l1088_108808


namespace NUMINAMATH_GPT_basketball_third_quarter_points_l1088_108868

noncomputable def teamA_points (a r : ℕ) : ℕ :=
a + a*r + a*r^2 + a*r^3

noncomputable def teamB_points (b d : ℕ) : ℕ :=
b + (b + d) + (b + 2*d) + (b + 3*d)

theorem basketball_third_quarter_points (a b d : ℕ) (r : ℕ) 
    (h1 : r > 1) (h2 : d > 0) (h3 : a * (r^4 - 1) / (r - 1) = 4 * b + 6 * d + 3)
    (h4 : a * (r^4 - 1) / (r - 1) ≤ 100) (h5 : 4 * b + 6 * d ≤ 100) :
    a * r^2 + b + 2 * d = 60 :=
sorry

end NUMINAMATH_GPT_basketball_third_quarter_points_l1088_108868


namespace NUMINAMATH_GPT_girls_more_than_boys_l1088_108800

-- Given conditions
def ratio_boys_girls : ℕ := 3
def ratio_girls_boys : ℕ := 4
def total_students : ℕ := 42

-- Theorem statement
theorem girls_more_than_boys : 
  let x := total_students / (ratio_boys_girls + ratio_girls_boys)
  let boys := ratio_boys_girls * x
  let girls := ratio_girls_boys * x
  girls - boys = 6 := by
  sorry

end NUMINAMATH_GPT_girls_more_than_boys_l1088_108800


namespace NUMINAMATH_GPT_problem_l1088_108875

theorem problem (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 1) :
  1 / (a ^ 3 * (b + c)) + 1 / (b ^ 3 * (c + a)) + 1 / (c ^ 3 * (a + b)) ≥ 3 / 2 :=
by sorry

end NUMINAMATH_GPT_problem_l1088_108875


namespace NUMINAMATH_GPT_faye_coloring_books_l1088_108805

theorem faye_coloring_books (initial_books : ℕ) (gave_away : ℕ) (bought_more : ℕ) (h1 : initial_books = 34) (h2 : gave_away = 3) (h3 : bought_more = 48) : 
  initial_books - gave_away + bought_more = 79 :=
by
  sorry

end NUMINAMATH_GPT_faye_coloring_books_l1088_108805


namespace NUMINAMATH_GPT_largest_consecutive_odd_number_sum_75_l1088_108872

theorem largest_consecutive_odd_number_sum_75 (a b c : ℤ) 
    (h1 : a + b + c = 75) 
    (h2 : b = a + 2) 
    (h3 : c = b + 2) : 
    c = 27 :=
by
  sorry

end NUMINAMATH_GPT_largest_consecutive_odd_number_sum_75_l1088_108872


namespace NUMINAMATH_GPT_ellipse_parabola_intersection_l1088_108839

theorem ellipse_parabola_intersection (a b k m : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c = Real.sqrt 2) (h4 : c^2 = a^2 - b^2)
    (h5 : (1 / 2) * 2 * a * 2 * b = 2 * Real.sqrt 3) (h6 : k ≠ 0) :
    (∃ (m: ℝ), (1 / 2) < m ∧ m < 2) :=
sorry

end NUMINAMATH_GPT_ellipse_parabola_intersection_l1088_108839


namespace NUMINAMATH_GPT_percentage_of_x_is_y_l1088_108895

theorem percentage_of_x_is_y (x y : ℝ) (h : 0.5 * (x - y) = 0.4 * (x + y)) : y = 0.1111 * x := 
sorry

end NUMINAMATH_GPT_percentage_of_x_is_y_l1088_108895


namespace NUMINAMATH_GPT_plane_equation_l1088_108862

variable (x y z : ℝ)

/-- Equation of the plane passing through points (0, 2, 3) and (2, 0, 3) and perpendicular to the plane 3x - y + 2z = 7 is 2x - 2y + z - 1 = 0. -/
theorem plane_equation :
  ∃ (A B C D : ℤ), A > 0 ∧ Int.gcd (Int.gcd A B) (Int.gcd C D) = 1 ∧ 
  (∀ (x y z : ℝ), (A * x + B * y + C * z + D = 0 ↔ 
  ((0, 2, 3) = (0, 2, 3) ∨ (2, 0, 3) = (2, 0, 3)) ∧ (3 * x - y + 2 * z = 7))) ∧
  A = 2 ∧ B = -2 ∧ C = 1 ∧ D = -1 :=
by
  sorry

end NUMINAMATH_GPT_plane_equation_l1088_108862


namespace NUMINAMATH_GPT_peanut_butter_sandwich_days_l1088_108850

theorem peanut_butter_sandwich_days 
  (H : ℕ)
  (total_days : ℕ)
  (probability_ham_and_cake : ℚ)
  (ham_probability : ℚ)
  (cake_probability : ℚ)
  (Ham_days : H = 3)
  (Total_days : total_days = 5)
  (Ham_probability_val : ham_probability = H / 5)
  (Cake_probability_val : cake_probability = 1 / 5)
  (Probability_condition : ham_probability * cake_probability = 0.12) :
  5 - H = 2 :=
by 
  sorry

end NUMINAMATH_GPT_peanut_butter_sandwich_days_l1088_108850


namespace NUMINAMATH_GPT_total_amount_correct_l1088_108856

namespace ProofExample

def initial_amount : ℝ := 3

def additional_amount : ℝ := 6.8

def total_amount (initial : ℝ) (additional : ℝ) : ℝ := initial + additional

theorem total_amount_correct : total_amount initial_amount additional_amount = 9.8 :=
by
  sorry

end ProofExample

end NUMINAMATH_GPT_total_amount_correct_l1088_108856


namespace NUMINAMATH_GPT_megatech_astrophysics_degrees_l1088_108891

theorem megatech_astrophysics_degrees :
  let microphotonics := 10
  let home_electronics := 24
  let food_additives := 15
  let gmo := 29
  let industrial_lubricants := 8
  let total_percentage := microphotonics + home_electronics + food_additives + gmo + industrial_lubricants
  let astrophysics_percentage := 100 - total_percentage
  let total_degrees := 360
  let astrophysics_degrees := (astrophysics_percentage / 100) * total_degrees
  astrophysics_degrees = 50.4 :=
by
  sorry

end NUMINAMATH_GPT_megatech_astrophysics_degrees_l1088_108891


namespace NUMINAMATH_GPT_water_supply_days_l1088_108886

theorem water_supply_days (C V : ℕ) 
  (h1: C = 75 * (V + 10))
  (h2: C = 60 * (V + 20)) : 
  (C / V) = 100 := 
sorry

end NUMINAMATH_GPT_water_supply_days_l1088_108886


namespace NUMINAMATH_GPT_range_of_x_l1088_108841

-- Define the condition: the expression under the square root must be non-negative
def condition (x : ℝ) : Prop := 3 + x ≥ 0

-- Define what we want to prove: the range of x such that the condition holds
theorem range_of_x (x : ℝ) : condition x ↔ x ≥ -3 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_range_of_x_l1088_108841


namespace NUMINAMATH_GPT_smallest_n_l1088_108893

theorem smallest_n (a b c n : ℕ) (h1 : n = 100 * a + 10 * b + c)
  (h2 : n = a + b + c + a * b + b * c + a * c + a * b * c)
  (h3 : n >= 100 ∧ n < 1000)
  (h4 : a ≥ 1 ∧ a ≤ 9)
  (h5 : b ≥ 0 ∧ b ≤ 9)
  (h6 : c ≥ 0 ∧ c ≤ 9) :
  n = 199 :=
sorry

end NUMINAMATH_GPT_smallest_n_l1088_108893


namespace NUMINAMATH_GPT_smallest_n_satisfying_conditions_l1088_108842

-- We need variables and statements
variables (n : ℕ)

-- Define the conditions
def condition1 : Prop := n % 6 = 4
def condition2 : Prop := n % 7 = 3
def condition3 : Prop := n > 20

-- The main theorem statement to be proved
theorem smallest_n_satisfying_conditions (h1 : condition1 n) (h2 : condition2 n) (h3 : condition3 n) : n = 52 :=
by 
  sorry

end NUMINAMATH_GPT_smallest_n_satisfying_conditions_l1088_108842


namespace NUMINAMATH_GPT_stamps_total_l1088_108899

theorem stamps_total (x y : ℕ) (hx : x = 34) (hy : y = x + 44) : x + y = 112 :=
by sorry

end NUMINAMATH_GPT_stamps_total_l1088_108899


namespace NUMINAMATH_GPT_percent_area_covered_by_hexagons_l1088_108801

theorem percent_area_covered_by_hexagons (a : ℝ) (h1 : 0 < a) :
  let large_square_area := 4 * a^2
  let hexagon_contribution := a^2 / 4
  (hexagon_contribution / large_square_area) * 100 = 25 := 
by
  sorry

end NUMINAMATH_GPT_percent_area_covered_by_hexagons_l1088_108801


namespace NUMINAMATH_GPT_fewer_cucumbers_than_potatoes_l1088_108859

theorem fewer_cucumbers_than_potatoes :
  ∃ C : ℕ, 237 + C + 2 * C = 768 ∧ 237 - C = 60 :=
by
  sorry

end NUMINAMATH_GPT_fewer_cucumbers_than_potatoes_l1088_108859


namespace NUMINAMATH_GPT_sin_600_eq_neg_sqrt3_div2_l1088_108807

theorem sin_600_eq_neg_sqrt3_div2 : Real.sin (600 * Real.pi / 180) = -Real.sqrt 3 / 2 :=
by sorry

end NUMINAMATH_GPT_sin_600_eq_neg_sqrt3_div2_l1088_108807


namespace NUMINAMATH_GPT_value_of_fraction_l1088_108879

variable (x y : ℝ)
variable (hx : x ≠ 0)
variable (hy : y ≠ 0)
variable (h : (4 * x + y) / (x - 4 * y) = -3)

theorem value_of_fraction : (x + 4 * y) / (4 * x - y) = 39 / 37 := by
  sorry

end NUMINAMATH_GPT_value_of_fraction_l1088_108879


namespace NUMINAMATH_GPT_periodic_length_le_T_l1088_108822

noncomputable def purely_periodic (a : ℚ) (T : ℕ) : Prop :=
∃ p : ℤ, a = p / (10^T - 1)

theorem periodic_length_le_T {a b : ℚ} {T : ℕ} 
  (ha : purely_periodic a T) 
  (hb : purely_periodic b T) 
  (hab_sum : purely_periodic (a + b) T)
  (hab_prod : purely_periodic (a * b) T) :
  ∃ Ta Tb : ℕ, Ta ≤ T ∧ Tb ≤ T ∧ purely_periodic a Ta ∧ purely_periodic b Tb := 
sorry

end NUMINAMATH_GPT_periodic_length_le_T_l1088_108822


namespace NUMINAMATH_GPT_number_of_valid_six_digit_house_numbers_l1088_108817

-- Define the set of two-digit primes less than 60
def two_digit_primes : List ℕ := [11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59]

-- Define a predicate checking if a number is a two-digit prime less than 60
def is_valid_prime (n : ℕ) : Prop :=
  n ∈ two_digit_primes

-- Define the function to count distinct valid primes forming ABCDEF
def count_valid_house_numbers : ℕ :=
  let primes_count := two_digit_primes.length
  primes_count * (primes_count - 1) * (primes_count - 2)

-- State the main theorem
theorem number_of_valid_six_digit_house_numbers : count_valid_house_numbers = 1716 := by
  -- Showing the count of valid house numbers forms 1716
  sorry

end NUMINAMATH_GPT_number_of_valid_six_digit_house_numbers_l1088_108817


namespace NUMINAMATH_GPT_evaluate_polynomial_given_condition_l1088_108824

theorem evaluate_polynomial_given_condition :
  ∀ x : ℝ, x > 0 → x^2 - 2 * x - 8 = 0 → (x^3 - 2 * x^2 - 8 * x + 4 = 4) := 
by
  intro x hx hcond
  sorry

end NUMINAMATH_GPT_evaluate_polynomial_given_condition_l1088_108824


namespace NUMINAMATH_GPT_factor_poly_find_abs_l1088_108860

theorem factor_poly_find_abs {
  p q : ℤ
} (h1 : 3 * (-2)^3 - p * (-2) + q = 0) 
  (h2 : 3 * (3)^3 - p * (3) + q = 0) :
  |3 * p - 2 * q| = 99 := sorry

end NUMINAMATH_GPT_factor_poly_find_abs_l1088_108860


namespace NUMINAMATH_GPT_area_triangle_ABC_l1088_108874

noncomputable def area_trapezoid (AB CD height : ℝ) : ℝ :=
  (AB + CD) * height / 2

noncomputable def area_triangle (base height : ℝ) : ℝ :=
  base * height / 2

variable (AB CD height area_ABCD : ℝ)
variables (h0 : CD = 3 * AB) (h1 : area_trapezoid AB CD height = 24)

theorem area_triangle_ABC : area_triangle AB height = 6 :=
by
  sorry

end NUMINAMATH_GPT_area_triangle_ABC_l1088_108874


namespace NUMINAMATH_GPT_find_z_l1088_108803

def z_value (i : ℂ) (z : ℂ) : Prop := z * (1 - (2 * i)) = 2 + (4 * i)

theorem find_z (i z : ℂ) (hi : i^2 = -1) (h : z_value i z) : z = - (2 / 5) + (8 / 5) * i := by
  sorry

end NUMINAMATH_GPT_find_z_l1088_108803


namespace NUMINAMATH_GPT_exists_nat_sol_x9_eq_2013y10_l1088_108864

theorem exists_nat_sol_x9_eq_2013y10 : ∃ (x y : ℕ), x^9 = 2013 * y^10 :=
by {
  -- Assume x and y are natural numbers, and prove that x^9 = 2013 y^10 has a solution
  sorry
}

end NUMINAMATH_GPT_exists_nat_sol_x9_eq_2013y10_l1088_108864


namespace NUMINAMATH_GPT_speed_limit_inequality_l1088_108873

theorem speed_limit_inequality (v : ℝ) : (v ≤ 40) :=
sorry

end NUMINAMATH_GPT_speed_limit_inequality_l1088_108873


namespace NUMINAMATH_GPT_larger_root_of_quadratic_eq_l1088_108869

theorem larger_root_of_quadratic_eq : 
  ∀ x : ℝ, x^2 - 13 * x + 36 = 0 → x = 9 ∨ x = 4 := by
  sorry

end NUMINAMATH_GPT_larger_root_of_quadratic_eq_l1088_108869


namespace NUMINAMATH_GPT_find_fraction_l1088_108892

def f (x : ℕ) : ℕ := 3 * x + 2
def g (x : ℕ) : ℕ := 2 * x - 3

theorem find_fraction : (f (g (f 2))) / (g (f (g 2))) = 41 / 7 := 
by 
  sorry

end NUMINAMATH_GPT_find_fraction_l1088_108892


namespace NUMINAMATH_GPT_number_of_pizzas_l1088_108823

-- Define the conditions
def slices_per_pizza := 8
def total_slices := 168

-- Define the statement we want to prove
theorem number_of_pizzas : total_slices / slices_per_pizza = 21 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_number_of_pizzas_l1088_108823


namespace NUMINAMATH_GPT_sum_of_squares_of_distances_l1088_108833

-- Definitions based on the conditions provided:
variables (A B C D X : Point)
variable (a : ℝ)
variable (h1 h2 h3 h4 : ℝ)

-- Conditions:
axiom square_side_length : a = 5
axiom area_ratios : (1/2 * a * h1) / (1/2 * a * h2) = 1 / 5 ∧ 
                    (1/2 * a * h2) / (1/2 * a * h3) = 5 / 9

-- Problem Statement to Prove:
theorem sum_of_squares_of_distances :
  h1^2 + h2^2 + h3^2 + h4^2 = 33 :=
sorry

end NUMINAMATH_GPT_sum_of_squares_of_distances_l1088_108833


namespace NUMINAMATH_GPT_trajectory_of_M_l1088_108831

variable (P : ℝ × ℝ) (A : ℝ × ℝ := (4, 0))
variable (M : ℝ × ℝ)

theorem trajectory_of_M (hP : P.1^2 + 4 * P.2^2 = 4) (hM : M = ((P.1 + 4) / 2, P.2 / 2)) :
  (M.1 - 2)^2 + 4 * M.2^2 = 1 :=
by
  sorry

end NUMINAMATH_GPT_trajectory_of_M_l1088_108831


namespace NUMINAMATH_GPT_contrapositive_proposition_l1088_108837

def proposition (x : ℝ) : Prop := x < 0 → x^2 > 0

theorem contrapositive_proposition :
  (∀ x : ℝ, proposition x) → (∀ x : ℝ, x^2 ≤ 0 → x ≥ 0) :=
by
  sorry

end NUMINAMATH_GPT_contrapositive_proposition_l1088_108837


namespace NUMINAMATH_GPT_johnny_worked_hours_l1088_108870

theorem johnny_worked_hours (total_earned hourly_wage hours_worked : ℝ) 
(h1 : total_earned = 16.5) (h2 : hourly_wage = 8.25) (h3 : total_earned / hourly_wage = hours_worked) : 
hours_worked = 2 := 
sorry

end NUMINAMATH_GPT_johnny_worked_hours_l1088_108870


namespace NUMINAMATH_GPT_triangle_perimeter_l1088_108855

-- Define the triangle with sides a, b, c
structure Triangle :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)

-- Define the predicate that checks if the triangle is isosceles
def isIsosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.a = t.c ∨ t.b = t.c

-- Define the predicate that calculates the perimeter of the triangle
def perimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

-- State the problem
theorem triangle_perimeter : 
  ∃ (t : Triangle), isIsosceles t ∧ (    (t.a = 6 ∧ t.b = 9 ∧ perimeter t = 24)
                                       ∨ (t.b = 6 ∧ t.a = 9 ∧ perimeter t = 24)
                                       ∨ (t.c = 6 ∧ t.a = 9 ∧ perimeter t = 21)
                                       ∨ (t.a = 6 ∧ t.c = 9 ∧ perimeter t = 21)
                                       ∨ (t.b = 6 ∧ t.c = 9 ∧ perimeter t = 24)
                                       ∨ (t.c = 6 ∧ t.b = 9 ∧ perimeter t = 21)
                                    ) :=
sorry

end NUMINAMATH_GPT_triangle_perimeter_l1088_108855


namespace NUMINAMATH_GPT_sqrt_221_between_15_and_16_l1088_108889

theorem sqrt_221_between_15_and_16 : 15 < Real.sqrt 221 ∧ Real.sqrt 221 < 16 := by
  sorry

end NUMINAMATH_GPT_sqrt_221_between_15_and_16_l1088_108889


namespace NUMINAMATH_GPT_square_B_perimeter_l1088_108883

theorem square_B_perimeter :
  ∀ (sideA sideB : ℝ), (4 * sideA = 24) → (sideB^2 = (sideA^2) / 4) → (4 * sideB = 12) :=
by
  sorry

end NUMINAMATH_GPT_square_B_perimeter_l1088_108883


namespace NUMINAMATH_GPT_smallest_Y_l1088_108840

-- Define the necessary conditions
def is_digits_0_1 (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, d = 0 ∨ d = 1

def is_divisible_by_15 (n : ℕ) : Prop :=
  n % 15 = 0

-- Define the main problem statement
theorem smallest_Y (S Y : ℕ) (hS_pos : S > 0) (hS_digits : is_digits_0_1 S) (hS_div_15 : is_divisible_by_15 S) (hY : Y = S / 15) :
  Y = 74 :=
sorry

end NUMINAMATH_GPT_smallest_Y_l1088_108840


namespace NUMINAMATH_GPT_fraction_transform_l1088_108802

theorem fraction_transform (x : ℝ) (h : (1/3) * x = 12) : (1/4) * x = 9 :=
by 
  sorry

end NUMINAMATH_GPT_fraction_transform_l1088_108802


namespace NUMINAMATH_GPT_range_of_function_l1088_108888

theorem range_of_function : 
  ∀ y : ℝ, 
  (∃ x : ℝ, y = x^2 + 1) ↔ (y ≥ 1) :=
by
  sorry

end NUMINAMATH_GPT_range_of_function_l1088_108888


namespace NUMINAMATH_GPT_smallest_positive_period_l1088_108863

noncomputable def f (A ω φ : ℝ) (x : ℝ) := A * Real.sin (ω * x + φ)

theorem smallest_positive_period 
  (A ω φ T : ℝ) 
  (hA : A > 0) 
  (hω : ω > 0)
  (h1 : f A ω φ (π / 2) = f A ω φ (2 * π / 3))
  (h2 : f A ω φ (π / 6) = -f A ω φ (π / 2))
  (h3 : ∀ x1 x2, (π / 6) ≤ x1 → x1 ≤ x2 → x2 ≤ (π / 2) → f A ω φ x1 ≤ f A ω φ x2) :
  T = π :=
sorry

end NUMINAMATH_GPT_smallest_positive_period_l1088_108863


namespace NUMINAMATH_GPT_find_a_for_odd_function_l1088_108815

theorem find_a_for_odd_function (f : ℝ → ℝ) (a : ℝ) (h₀ : ∀ x, f (-x) = -f x) (h₁ : ∀ x, x < 0 → f x = x^2 + a * x) (h₂ : f 3 = 6) : a = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_a_for_odd_function_l1088_108815


namespace NUMINAMATH_GPT_column_of_1000_is_C_l1088_108882

def column_of_integer (n : ℕ) : String :=
  ["A", "B", "C", "D", "E", "F", "E", "D", "C", "B"].get! ((n - 2) % 10)

theorem column_of_1000_is_C :
  column_of_integer 1000 = "C" :=
by
  sorry

end NUMINAMATH_GPT_column_of_1000_is_C_l1088_108882


namespace NUMINAMATH_GPT_son_age_l1088_108834

theorem son_age (M S : ℕ) (h1: M = S + 26) (h2: M + 2 = 2 * (S + 2)) : S = 24 :=
by
  sorry

end NUMINAMATH_GPT_son_age_l1088_108834


namespace NUMINAMATH_GPT_remainder_of_power_modulo_l1088_108820

theorem remainder_of_power_modulo (n : ℕ) (h : n = 2040) : 
  3^2040 % 5 = 1 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_power_modulo_l1088_108820


namespace NUMINAMATH_GPT_additional_rocks_needed_l1088_108880

-- Define the dimensions of the garden
def length (garden : Type) : ℕ := 15
def width (garden : Type) : ℕ := 10
def rock_cover (rock : Type) : ℕ := 1

-- Define the number of rocks Mrs. Hilt has
def rocks_possessed (mrs_hilt : Type) : ℕ := 64

-- Define the perimeter of the garden
def perimeter (garden : Type) : ℕ :=
  2 * (length garden + width garden)

-- Define the number of rocks required for the first layer
def rocks_first_layer (garden : Type) : ℕ :=
  perimeter garden

-- Define the number of rocks required for the second layer (only longer sides)
def rocks_second_layer (garden : Type) : ℕ :=
  2 * length garden

-- Define the total number of rocks needed
def total_rocks_needed (garden : Type) : ℕ :=
  rocks_first_layer garden + rocks_second_layer garden

-- Prove the number of additional rocks Mrs. Hilt needs
theorem additional_rocks_needed (garden : Type) (mrs_hilt : Type):
  total_rocks_needed garden - rocks_possessed mrs_hilt = 16 := by
  sorry

end NUMINAMATH_GPT_additional_rocks_needed_l1088_108880


namespace NUMINAMATH_GPT_bert_phone_price_l1088_108848

theorem bert_phone_price :
  ∃ x : ℕ, x * 8 = 144 := sorry

end NUMINAMATH_GPT_bert_phone_price_l1088_108848


namespace NUMINAMATH_GPT_neg_A_is_square_of_int_l1088_108877

theorem neg_A_is_square_of_int (x y z : ℤ) (A : ℤ) (h1 : A = x * y + y * z + z * x) 
  (h2 : A = (x + 1) * (y - 2) + (y - 2) * (z - 2) + (z - 2) * (x + 1)) : ∃ k : ℤ, -A = k^2 :=
by
  sorry

end NUMINAMATH_GPT_neg_A_is_square_of_int_l1088_108877


namespace NUMINAMATH_GPT_B_took_18_more_boxes_than_D_l1088_108813

noncomputable def A_boxes : ℕ := sorry
noncomputable def B_boxes : ℕ := A_boxes + 4
noncomputable def C_boxes : ℕ := sorry
noncomputable def D_boxes : ℕ := C_boxes + 8
noncomputable def A_owes_C : ℕ := 112
noncomputable def B_owes_D : ℕ := 72

theorem B_took_18_more_boxes_than_D : (B_boxes - D_boxes) = 18 :=
sorry

end NUMINAMATH_GPT_B_took_18_more_boxes_than_D_l1088_108813


namespace NUMINAMATH_GPT_cube_sum_l1088_108867

-- Definitions
variable (ω : ℂ) (h1 : ω^3 = 1) (h2 : ω^2 + ω + 1 = 0) -- nonreal root

-- Theorem statement
theorem cube_sum : (2 - ω + ω^2)^3 + (2 + ω - ω^2)^3 = -2 :=
by 
  sorry

end NUMINAMATH_GPT_cube_sum_l1088_108867


namespace NUMINAMATH_GPT_find_length_l1088_108812

variables (w h A l : ℕ)
variable (A_eq : A = 164)
variable (w_eq : w = 4)
variable (h_eq : h = 3)

theorem find_length : 2 * l * w + 2 * l * h + 2 * w * h = A → l = 10 :=
by
  intros H
  rw [w_eq, h_eq, A_eq] at H
  linarith

end NUMINAMATH_GPT_find_length_l1088_108812


namespace NUMINAMATH_GPT_number_of_solution_values_l1088_108832

theorem number_of_solution_values (c : ℕ) : 
  0 ≤ c ∧ c ≤ 2000 ↔ (∃ x : ℝ, 5 * (⌊x⌋ : ℝ) + 3 * (⌈x⌉ : ℝ) = c) →
  c = 251 := 
sorry

end NUMINAMATH_GPT_number_of_solution_values_l1088_108832


namespace NUMINAMATH_GPT_sum_of_possible_values_l1088_108829

theorem sum_of_possible_values (a b c d : ℝ) (h : (a - b) * (c - d) / ((b - c) * (d - a)) = 3 / 7) : 
  (a - c) * (b - d) / ((c - d) * (d - a)) = 1 :=
by
  -- Solution omitted
  sorry

end NUMINAMATH_GPT_sum_of_possible_values_l1088_108829


namespace NUMINAMATH_GPT_commission_percentage_l1088_108827

theorem commission_percentage (commission_earned total_sales : ℝ) (h₀ : commission_earned = 18) (h₁ : total_sales = 720) : 
  ((commission_earned / total_sales) * 100) = 2.5 := by {
  sorry
}

end NUMINAMATH_GPT_commission_percentage_l1088_108827


namespace NUMINAMATH_GPT_total_rainfall_2003_and_2004_l1088_108881

noncomputable def average_rainfall_2003 : ℝ := 45
noncomputable def months_in_year : ℕ := 12
noncomputable def percent_increase : ℝ := 0.05

theorem total_rainfall_2003_and_2004 :
  let rainfall_2004 := average_rainfall_2003 * (1 + percent_increase)
  let total_rainfall_2003 := average_rainfall_2003 * months_in_year
  let total_rainfall_2004 := rainfall_2004 * months_in_year
  total_rainfall_2003 = 540 ∧ total_rainfall_2004 = 567 := 
by 
  sorry

end NUMINAMATH_GPT_total_rainfall_2003_and_2004_l1088_108881


namespace NUMINAMATH_GPT_mrs_hilt_hot_dogs_l1088_108816

theorem mrs_hilt_hot_dogs (cost_per_hotdog total_cost : ℕ) (h1 : cost_per_hotdog = 50) (h2 : total_cost = 300) :
  total_cost / cost_per_hotdog = 6 := by
  sorry

end NUMINAMATH_GPT_mrs_hilt_hot_dogs_l1088_108816


namespace NUMINAMATH_GPT_average_weight_of_16_boys_is_50_25_l1088_108887

theorem average_weight_of_16_boys_is_50_25
  (W : ℝ)
  (h1 : 8 * 45.15 = 361.2)
  (h2 : 24 * 48.55 = 1165.2)
  (h3 : 16 * W + 361.2 = 1165.2) :
  W = 50.25 :=
sorry

end NUMINAMATH_GPT_average_weight_of_16_boys_is_50_25_l1088_108887


namespace NUMINAMATH_GPT_set_A_enum_l1088_108858

def A : Set ℤ := {z | ∃ x : ℕ, 6 / (x - 2) = z ∧ 6 % (x - 2) = 0}

theorem set_A_enum : A = {-3, -6, 6, 3, 2, 1} := by
  sorry

end NUMINAMATH_GPT_set_A_enum_l1088_108858


namespace NUMINAMATH_GPT_percentage_employees_6_years_or_more_is_26_l1088_108849

-- Define the units for different years of service
def units_less_than_2_years : ℕ := 4
def units_2_to_4_years : ℕ := 6
def units_4_to_6_years : ℕ := 7
def units_6_to_8_years : ℕ := 3
def units_8_to_10_years : ℕ := 2
def units_more_than_10_years : ℕ := 1

-- Define the total units
def total_units : ℕ :=
  units_less_than_2_years +
  units_2_to_4_years +
  units_4_to_6_years +
  units_6_to_8_years +
  units_8_to_10_years +
  units_more_than_10_years

-- Define the units representing employees with 6 years or more of service
def units_6_years_or_more : ℕ :=
  units_6_to_8_years +
  units_8_to_10_years +
  units_more_than_10_years

-- The goal is to prove that this percentage is 26%
theorem percentage_employees_6_years_or_more_is_26 :
  (units_6_years_or_more * 100) / total_units = 26 := by
  sorry

end NUMINAMATH_GPT_percentage_employees_6_years_or_more_is_26_l1088_108849


namespace NUMINAMATH_GPT_crease_length_l1088_108810

theorem crease_length 
  (AB AC : ℝ) (BC : ℝ) (BA' : ℝ) (A'C : ℝ)
  (h1 : AB = 10) (h2 : AC = 10) (h3 : BC = 8) (h4 : BA' = 3) (h5 : A'C = 5) :
  ∃ PQ : ℝ, PQ = (Real.sqrt 7393) / 15 := by
  sorry

end NUMINAMATH_GPT_crease_length_l1088_108810


namespace NUMINAMATH_GPT_factor_polynomial_l1088_108838

theorem factor_polynomial 
(a b c d : ℝ) :
  a^3 * (b^2 - d^2) + b^3 * (c^2 - a^2) + c^3 * (d^2 - b^2) + d^3 * (a^2 - c^2)
  = (a - b) * (b - c) * (c - d) * (d - a) * (a^2 + ab + ac + ad + b^2 + bc + bd + c^2 + cd + d^2) :=
sorry

end NUMINAMATH_GPT_factor_polynomial_l1088_108838


namespace NUMINAMATH_GPT_sum_of_cubics_l1088_108866

noncomputable def root_polynomial (x : ℝ) := 5 * x^3 + 2003 * x + 3005

theorem sum_of_cubics (a b c : ℝ)
  (h1 : root_polynomial a = 0)
  (h2 : root_polynomial b = 0)
  (h3 : root_polynomial c = 0) :
  (a + b)^3 + (b + c)^3 + (c + a)^3 = 1803 :=
sorry

end NUMINAMATH_GPT_sum_of_cubics_l1088_108866


namespace NUMINAMATH_GPT_solution_to_logarithmic_equation_l1088_108836

noncomputable def log_base (a b : ℝ) := Real.log b / Real.log a

def equation (x : ℝ) := log_base 2 x + 1 / log_base (x + 1) 2 = 1

theorem solution_to_logarithmic_equation :
  ∃ x > 0, equation x ∧ x = 1 :=
by
  sorry

end NUMINAMATH_GPT_solution_to_logarithmic_equation_l1088_108836


namespace NUMINAMATH_GPT_sin_60_eq_sqrt3_div_2_l1088_108825

theorem sin_60_eq_sqrt3_div_2 : Real.sin (Real.pi / 3) = Real.sqrt 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_sin_60_eq_sqrt3_div_2_l1088_108825


namespace NUMINAMATH_GPT_find_number_l1088_108876

theorem find_number (x : ℝ) (h : (x / 4) + 3 = 5) : x = 8 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l1088_108876


namespace NUMINAMATH_GPT_loss_record_l1088_108843

-- Conditions: a profit of 25 yuan is recorded as +25 yuan.
def profit_record (profit : Int) : Int :=
  profit

-- Statement we need to prove: A loss of 30 yuan is recorded as -30 yuan.
theorem loss_record : profit_record (-30) = -30 :=
by
  sorry

end NUMINAMATH_GPT_loss_record_l1088_108843


namespace NUMINAMATH_GPT_power_equivalence_l1088_108885

theorem power_equivalence (m n : ℕ) (hm_pos : 0 < m) (hn_pos : 0 < n) (x y : ℕ) 
  (hx : 2^m = x) (hy : 2^(2 * n) = y) : 4^(m + 2 * n) = x^2 * y^2 := 
by 
  sorry

end NUMINAMATH_GPT_power_equivalence_l1088_108885


namespace NUMINAMATH_GPT_unique_five_digit_integers_l1088_108826

-- Define the problem conditions
def digits := [2, 2, 3, 9, 9]
def total_spots := 5
def factorial (n : Nat) : Nat := if n = 0 then 1 else n * factorial (n - 1)

-- Compute the number of five-digit integers that can be formed
noncomputable def num_unique_permutations : Nat :=
  factorial total_spots / (factorial 2 * factorial 1 * factorial 2)

-- Proof statement
theorem unique_five_digit_integers : num_unique_permutations = 30 := by
  sorry

end NUMINAMATH_GPT_unique_five_digit_integers_l1088_108826


namespace NUMINAMATH_GPT_fraction_one_third_between_l1088_108821

theorem fraction_one_third_between (a b : ℚ) (h1 : a = 1/6) (h2 : b = 1/4) : (1/3 * (b - a) + a = 7/36) :=
by
  -- Conditions
  have ha : a = 1/6 := h1
  have hb : b = 1/4 := h2
  -- Start proof
  sorry

end NUMINAMATH_GPT_fraction_one_third_between_l1088_108821


namespace NUMINAMATH_GPT_abs_inequality_solution_set_l1088_108865

theorem abs_inequality_solution_set (x : ℝ) : -1 < x ∧ x < 1 ↔ |2*x - 1| - |x - 2| < 0 := by
  sorry

end NUMINAMATH_GPT_abs_inequality_solution_set_l1088_108865


namespace NUMINAMATH_GPT_max_sides_subdivision_13_max_sides_subdivision_1950_l1088_108844

-- Part (a)
theorem max_sides_subdivision_13 (n : ℕ) (h : n = 13) : 
  ∃ p : ℕ, p ≤ n ∧ p = 13 := 
sorry

-- Part (b)
theorem max_sides_subdivision_1950 (n : ℕ) (h : n = 1950) : 
  ∃ p : ℕ, p ≤ n ∧ p = 1950 := 
sorry

end NUMINAMATH_GPT_max_sides_subdivision_13_max_sides_subdivision_1950_l1088_108844


namespace NUMINAMATH_GPT_simplify_expression_l1088_108811

noncomputable def p (x a b c : ℝ) :=
  (x + 2 * a)^2 / ((a - b) * (a - c)) +
  (x + 2 * b)^2 / ((b - a) * (b - c)) +
  (x + 2 * c)^2 / ((c - a) * (c - b))

theorem simplify_expression (a b c x : ℝ) (h : a ≠ b ∧ a ≠ c ∧ b ≠ c) :
  p x a b c = 4 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1088_108811
