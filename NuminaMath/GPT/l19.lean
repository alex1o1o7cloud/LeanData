import Mathlib

namespace NUMINAMATH_GPT_shortest_distance_from_circle_to_line_l19_1903

theorem shortest_distance_from_circle_to_line :
  let circle := { p : ℝ × ℝ | (p.1 - 5)^2 + (p.2 - 3)^2 = 9 }
  let line := { p : ℝ × ℝ | 3 * p.1 + 4 * p.2 - 2 = 0 }
  ∀ (M : ℝ × ℝ), M ∈ circle → ∃ d : ℝ, d = 2 ∧ ∀ q ∈ line, dist M q = d := 
sorry

end NUMINAMATH_GPT_shortest_distance_from_circle_to_line_l19_1903


namespace NUMINAMATH_GPT_solve_for_n_l19_1929

def number_of_balls : ℕ := sorry

axiom A : number_of_balls = 2

theorem solve_for_n (n : ℕ) (h : (1 + 1 + n = number_of_balls) ∧ ((n : ℝ) / (1 + 1 + n) = 1 / 2)) : n = 2 :=
sorry

end NUMINAMATH_GPT_solve_for_n_l19_1929


namespace NUMINAMATH_GPT_gcd_115_161_l19_1923

theorem gcd_115_161 : Nat.gcd 115 161 = 23 := by
  sorry

end NUMINAMATH_GPT_gcd_115_161_l19_1923


namespace NUMINAMATH_GPT_determine_h_l19_1945

variable {R : Type*} [CommRing R]

def h_poly (x : R) : R := -8*x^4 + 2*x^3 + 4*x^2 - 6*x + 2

theorem determine_h (x : R) :
  (8*x^4 - 4*x^2 + 2 + h_poly x = 2*x^3 - 6*x + 4) ->
  h_poly x = -8*x^4 + 2*x^3 + 4*x^2 - 6*x + 2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_determine_h_l19_1945


namespace NUMINAMATH_GPT_range_of_a_l19_1944

noncomputable def f (x : ℝ) : ℝ := 6 / x - x

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 1 < x → x^2 + a * x - 6 > 0) ↔ 5 ≤ a :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l19_1944


namespace NUMINAMATH_GPT_students_count_l19_1925

theorem students_count :
  ∀ (sets marbles_per_set marbles_per_student total_students : ℕ),
    sets = 3 →
    marbles_per_set = 32 →
    marbles_per_student = 4 →
    total_students = (sets * marbles_per_set) / marbles_per_student →
    total_students = 24 :=
by
  intros sets marbles_per_set marbles_per_student total_students
  intros h_sets h_marbles_per_set h_marbles_per_student h_total_students
  rw [h_sets, h_marbles_per_set, h_marbles_per_student] at h_total_students
  exact h_total_students

end NUMINAMATH_GPT_students_count_l19_1925


namespace NUMINAMATH_GPT_ratio_of_albums_l19_1970

variable (M K B A : ℕ)
variable (s : ℕ)

-- Conditions
def adele_albums := (A = 30)
def bridget_albums := (B = A - 15)
def katrina_albums := (K = 6 * B)
def miriam_albums := (M = s * K)
def total_albums := (M + K + B + A = 585)

-- Proof statement
theorem ratio_of_albums (h1 : adele_albums A) (h2 : bridget_albums B A) (h3 : katrina_albums K B) 
(h4 : miriam_albums M s K) (h5 : total_albums M K B A) :
  s = 5 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_albums_l19_1970


namespace NUMINAMATH_GPT_derivative_at_pi_div_2_l19_1984

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin x - 4 * Real.cos x

theorem derivative_at_pi_div_2 : (deriv f (Real.pi / 2)) = 4 := 
by
  sorry

end NUMINAMATH_GPT_derivative_at_pi_div_2_l19_1984


namespace NUMINAMATH_GPT_find_f_prime_one_l19_1935

noncomputable def f (x : ℝ) : ℝ := 1 / (1 + x)
def f_condition (x : ℝ) : Prop := f (1 / x) = x / (1 + x)

theorem find_f_prime_one : f_condition 1 → deriv f 1 = -1 / 4 := by
  intro h
  sorry

end NUMINAMATH_GPT_find_f_prime_one_l19_1935


namespace NUMINAMATH_GPT_solution_eq_l19_1980

theorem solution_eq (a x : ℚ) :
  (2 * (x - 2 * (x - a / 4)) = 3 * x) ∧ ((x + a) / 9 - (1 - 3 * x) / 12 = 1) → 
  a = 65 / 11 ∧ x = 13 / 11 :=
by
  sorry

end NUMINAMATH_GPT_solution_eq_l19_1980


namespace NUMINAMATH_GPT_packs_of_yellow_balls_l19_1947

theorem packs_of_yellow_balls (Y : ℕ) : 
  3 * 19 + Y * 19 + 8 * 19 = 399 → Y = 10 :=
by sorry

end NUMINAMATH_GPT_packs_of_yellow_balls_l19_1947


namespace NUMINAMATH_GPT_find_salary_for_january_l19_1956

-- Definitions based on problem conditions
variables (J F M A May : ℝ)
variables (h1 : (J + F + M + A) / 4 = 8000)
variables (h2 : (F + M + A + May) / 4 = 8200)
variables (hMay : May = 6500)

-- Lean statement
theorem find_salary_for_january : J = 5700 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_salary_for_january_l19_1956


namespace NUMINAMATH_GPT_ned_pieces_left_l19_1902

def boxes_bought : ℝ := 14.0
def boxes_given : ℝ := 7.0
def pieces_per_box : ℝ := 6.0
def boxes_left (bought : ℝ) (given : ℝ) : ℝ := bought - given
def total_pieces (boxes : ℝ) (pieces_per_box : ℝ) : ℝ := boxes * pieces_per_box

theorem ned_pieces_left : total_pieces (boxes_left boxes_bought boxes_given) pieces_per_box = 42.0 := by
  sorry

end NUMINAMATH_GPT_ned_pieces_left_l19_1902


namespace NUMINAMATH_GPT_distance_between_towns_l19_1958

theorem distance_between_towns 
  (x : ℝ) 
  (h1 : x / 100 - x / 110 = 0.15) : 
  x = 165 := 
by 
  sorry

end NUMINAMATH_GPT_distance_between_towns_l19_1958


namespace NUMINAMATH_GPT_solve_basketball_points_l19_1993

noncomputable def y_points_other_members (x : ℕ) : ℕ :=
  let d_points := (1 / 3) * x
  let e_points := (3 / 8) * x
  let f_points := 18
  let total := x
  total - d_points - e_points - f_points

theorem solve_basketball_points (x : ℕ) (h1: x > 0) (h2: ∃ y ≤ 24, y = y_points_other_members x) :
  ∃ y, y = 21 :=
by
  sorry

end NUMINAMATH_GPT_solve_basketball_points_l19_1993


namespace NUMINAMATH_GPT_xy_exists_5n_l19_1939

theorem xy_exists_5n (n : ℕ) (hpos : 0 < n) :
  ∃ x y : ℤ, x^2 + y^2 = 5^n ∧ Int.gcd x 5 = 1 ∧ Int.gcd y 5 = 1 :=
sorry

end NUMINAMATH_GPT_xy_exists_5n_l19_1939


namespace NUMINAMATH_GPT_petya_correct_square_l19_1975

theorem petya_correct_square :
  ∃ x a b : ℕ, (1 ≤ x ∧ x ≤ 9) ∧
              (x^2 = 10 * a + b) ∧ 
              (2 * x = 10 * b + a) ∧
              (x^2 = 81) :=
by
  sorry

end NUMINAMATH_GPT_petya_correct_square_l19_1975


namespace NUMINAMATH_GPT_factorize_difference_of_squares_l19_1916

variable (x : ℝ)

theorem factorize_difference_of_squares :
  x^2 - 1 = (x + 1) * (x - 1) :=
sorry

end NUMINAMATH_GPT_factorize_difference_of_squares_l19_1916


namespace NUMINAMATH_GPT_born_in_1890_l19_1962

theorem born_in_1890 (x : ℕ) (h1 : x^2 - x - 2 = 1890) (h2 : x^2 < 1950) : x = 44 :=
by {
    sorry
}

end NUMINAMATH_GPT_born_in_1890_l19_1962


namespace NUMINAMATH_GPT_allen_change_l19_1934

-- Define the cost per box and the number of boxes
def cost_per_box : ℕ := 7
def num_boxes : ℕ := 5

-- Define the total cost including the tip
def total_cost := num_boxes * cost_per_box
def tip := total_cost / 7
def total_paid := total_cost + tip

-- Define the amount given to the delivery person
def amount_given : ℕ := 100

-- Define the change received
def change := amount_given - total_paid

-- The statement to prove
theorem allen_change : change = 60 :=
by
  -- sorry is used here to skip the proof, as per the instruction
  sorry

end NUMINAMATH_GPT_allen_change_l19_1934


namespace NUMINAMATH_GPT_contrapositive_of_real_roots_l19_1983

variable {a : ℝ}

theorem contrapositive_of_real_roots :
  (1 + 4 * a < 0) → (a < 0) := by
  sorry

end NUMINAMATH_GPT_contrapositive_of_real_roots_l19_1983


namespace NUMINAMATH_GPT_integer_solutions_for_xyz_l19_1936

theorem integer_solutions_for_xyz (x y z : ℤ) : 
  (x - y - 1)^3 + (y - z - 2)^3 + (z - x + 3)^3 = 18 ↔
  (x = y ∧ y = z) ∨
  (x = y - 1 ∧ y = z) ∨
  (x = y ∧ y = z + 5) ∨
  (x = y + 4 ∧ y = z + 5) ∨
  (x = y + 4 ∧ z = y) ∨
  (x = y - 1 ∧ z = y + 4) :=
by {
  sorry
}

end NUMINAMATH_GPT_integer_solutions_for_xyz_l19_1936


namespace NUMINAMATH_GPT_intersection_M_N_l19_1960

def M : Set ℝ := {x | x < 1}
def N : Set ℝ := {x | 0 < x ∧ x < 2}

theorem intersection_M_N : M ∩ N = {x | 0 < x ∧ x < 1} :=
by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l19_1960


namespace NUMINAMATH_GPT_championship_positions_l19_1957

def positions_valid : Prop :=
  ∃ (pos_A pos_B pos_D pos_E pos_V pos_G : ℕ),
  (pos_A = pos_B + 3) ∧
  (pos_D < pos_E ∧ pos_E < pos_B) ∧
  (pos_V < pos_G) ∧
  (pos_D = 1) ∧
  (pos_E = 2) ∧
  (pos_B = 3) ∧
  (pos_V = 4) ∧
  (pos_G = 5) ∧
  (pos_A = 6)

theorem championship_positions : positions_valid :=
by
  sorry

end NUMINAMATH_GPT_championship_positions_l19_1957


namespace NUMINAMATH_GPT_smallest_n_for_terminating_decimal_l19_1938

theorem smallest_n_for_terminating_decimal :
  ∃ n : ℕ, n > 0 ∧ (∀ m : ℕ, m > 0 ∧ (∀ p: ℕ, (p ∣ (m + 150) → (p = 2 ∨ p = 5)) → m ≥ n)) ∧
  (∃ two_five_factors : ℕ, (two_five_factors = 5 ^ 3 * 2 ^ 3) ∧ (n + 150 = two_five_factors)) := 
by 
   exact ⟨50, by sorry⟩

end NUMINAMATH_GPT_smallest_n_for_terminating_decimal_l19_1938


namespace NUMINAMATH_GPT_number_of_keepers_l19_1927

theorem number_of_keepers (hens goats camels : ℕ) (keepers feet heads : ℕ)
  (h_hens : hens = 50)
  (h_goats : goats = 45)
  (h_camels : camels = 8)
  (h_equation : (2 * hens + 4 * goats + 4 * camels + 2 * keepers) = (hens + goats + camels + keepers + 224))
  : keepers = 15 :=
by
sorry

end NUMINAMATH_GPT_number_of_keepers_l19_1927


namespace NUMINAMATH_GPT_percent_within_one_std_dev_l19_1937

theorem percent_within_one_std_dev (m d : ℝ) (dist : ℝ → ℝ)
  (symm : ∀ x, dist (m + x) = dist (m - x))
  (less_than_upper_bound : ∀ x, (x < (m + d)) → dist x < 0.92) :
  ∃ p : ℝ, p = 0.84 :=
by
  sorry

end NUMINAMATH_GPT_percent_within_one_std_dev_l19_1937


namespace NUMINAMATH_GPT_root_fraction_power_l19_1940

theorem root_fraction_power (a : ℝ) (ha : a = 5) : 
  (a^(1/3)) / (a^(1/5)) = a^(2/15) := by
  sorry

end NUMINAMATH_GPT_root_fraction_power_l19_1940


namespace NUMINAMATH_GPT_expression_equals_k_times_10_pow_1007_l19_1950

theorem expression_equals_k_times_10_pow_1007 :
  (3^1006 + 7^1007)^2 - (3^1006 - 7^1007)^2 = 588 * 10^1007 := by
  sorry

end NUMINAMATH_GPT_expression_equals_k_times_10_pow_1007_l19_1950


namespace NUMINAMATH_GPT_quadratic_ineq_solution_range_l19_1914

theorem quadratic_ineq_solution_range (a : ℝ) : 
  (∃ x : ℝ, 1 < x ∧ x < 4 ∧ 2*x^2 - 8*x - 4 - a > 0) ↔ a < -4 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_ineq_solution_range_l19_1914


namespace NUMINAMATH_GPT_find_number_l19_1968

variable (a : ℕ) (n : ℕ)

theorem find_number (h₁ : a = 105) (h₂ : a ^ 3 = 21 * n * 45 * 49) : n = 25 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l19_1968


namespace NUMINAMATH_GPT_geometric_sequence_a4_l19_1973

theorem geometric_sequence_a4 :
    ∀ (a : ℕ → ℝ) (n : ℕ), 
    a 1 = 2 → 
    (∀ n : ℕ, a (n + 1) = 3 * a n) → 
    a 4 = 54 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_a4_l19_1973


namespace NUMINAMATH_GPT_trig_expression_value_l19_1919

theorem trig_expression_value (x : ℝ) (h : Real.tan x = 1/2) :
  (2 * Real.sin x + 3 * Real.cos x) / (Real.cos x - Real.sin x) = 8 :=
by
  sorry

end NUMINAMATH_GPT_trig_expression_value_l19_1919


namespace NUMINAMATH_GPT_cost_price_of_article_l19_1999

theorem cost_price_of_article (x : ℝ) (h : 66 - x = x - 22) : x = 44 :=
sorry

end NUMINAMATH_GPT_cost_price_of_article_l19_1999


namespace NUMINAMATH_GPT_rice_field_sacks_l19_1969

theorem rice_field_sacks (x : ℝ)
  (h1 : ∀ x, x + 1.20 * x = 44) : x = 20 :=
sorry

end NUMINAMATH_GPT_rice_field_sacks_l19_1969


namespace NUMINAMATH_GPT_no_six_odd_numbers_sum_to_one_l19_1941

theorem no_six_odd_numbers_sum_to_one (a b c d e f : ℕ)
  (ha : a % 2 = 1) (hb : b % 2 = 1) (hc : c % 2 = 1) (hd : d % 2 = 1) (he : e % 2 = 1) (hf : f % 2 = 1)
  (h_diff : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ d ≠ e ∧ d ≠ f ∧ e ≠ f)
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f) :
  (1 / a : ℝ) + 1 / b + 1 / c + 1 / d + 1 / e + 1 / f ≠ 1 :=
by
  sorry

end NUMINAMATH_GPT_no_six_odd_numbers_sum_to_one_l19_1941


namespace NUMINAMATH_GPT_misha_scored_48_in_second_attempt_l19_1920

theorem misha_scored_48_in_second_attempt (P1 P2 P3 : ℕ)
  (h1 : P2 = 2 * P1)
  (h2 : P3 = (3 / 2) * P2)
  (h3 : 24 ≤ P1)
  (h4 : (3 / 2) * 2 * P1 = 72) : P2 = 48 :=
by sorry

end NUMINAMATH_GPT_misha_scored_48_in_second_attempt_l19_1920


namespace NUMINAMATH_GPT_option_c_is_always_odd_l19_1965

theorem option_c_is_always_odd (n : ℤ) : ∃ (q : ℤ), n^2 + n + 5 = 2*q + 1 := by
  sorry

end NUMINAMATH_GPT_option_c_is_always_odd_l19_1965


namespace NUMINAMATH_GPT_find_k_l19_1963

theorem find_k 
  (c : ℝ) (a₁ : ℝ) (S : ℕ → ℝ) (k : ℝ)
  (h1 : ∀ n, S (n+1) = c * S n) 
  (h2 : S 1 = 3 + k)
  (h3 : ∀ n, S n = 3^n + k) :
  k = -1 :=
sorry

end NUMINAMATH_GPT_find_k_l19_1963


namespace NUMINAMATH_GPT_distance_light_300_years_eq_l19_1995

-- Define the constant distance light travels in one year
def distance_light_year : ℕ := 9460800000000

-- Define the time period in years
def time_period : ℕ := 300

-- Define the expected distance light travels in 300 years in scientific notation
def expected_distance : ℝ := 28382 * 10^13

-- The theorem to prove
theorem distance_light_300_years_eq :
  (distance_light_year * time_period) = 2838200000000000 :=
by
  sorry

end NUMINAMATH_GPT_distance_light_300_years_eq_l19_1995


namespace NUMINAMATH_GPT_inv_proportion_through_point_l19_1986

theorem inv_proportion_through_point (m : ℝ) (x y : ℝ) (h1 : y = m / x) (h2 : x = 2) (h3 : y = -3) : m = -6 := by
  sorry

end NUMINAMATH_GPT_inv_proportion_through_point_l19_1986


namespace NUMINAMATH_GPT_triangle_square_ratio_l19_1948

theorem triangle_square_ratio (t s : ℝ) 
  (h1 : 3 * t = 15) 
  (h2 : 4 * s = 12) : 
  t / s = 5 / 3 :=
by 
  -- skipping the proof
  sorry

end NUMINAMATH_GPT_triangle_square_ratio_l19_1948


namespace NUMINAMATH_GPT_roots_of_polynomial_l19_1931

theorem roots_of_polynomial :
  (3 * (2 + Real.sqrt 3)^4 - 19 * (2 + Real.sqrt 3)^3 + 34 * (2 + Real.sqrt 3)^2 - 19 * (2 + Real.sqrt 3) + 3 = 0) ∧ 
  (3 * (2 - Real.sqrt 3)^4 - 19 * (2 - Real.sqrt 3)^3 + 34 * (2 - Real.sqrt 3)^2 - 19 * (2 - Real.sqrt 3) + 3 = 0) ∧
  (3 * ((7 + Real.sqrt 13) / 6)^4 - 19 * ((7 + Real.sqrt 13) / 6)^3 + 34 * ((7 + Real.sqrt 13) / 6)^2 - 19 * ((7 + Real.sqrt 13) / 6) + 3 = 0) ∧
  (3 * ((7 - Real.sqrt 13) / 6)^4 - 19 * ((7 - Real.sqrt 13) / 6)^3 + 34 * ((7 - Real.sqrt 13) / 6)^2 - 19 * ((7 - Real.sqrt 13) / 6) + 3 = 0) :=
by sorry

end NUMINAMATH_GPT_roots_of_polynomial_l19_1931


namespace NUMINAMATH_GPT_annulus_area_l19_1942

theorem annulus_area (B C RW : ℝ) (h1 : B > C)
  (h2 : B^2 - (C + 5)^2 = RW^2) : 
  π * RW^2 = π * (B^2 - (C + 5)^2) :=
by
  sorry

end NUMINAMATH_GPT_annulus_area_l19_1942


namespace NUMINAMATH_GPT_total_money_divided_l19_1926

theorem total_money_divided (A B C : ℝ) (hA : A = 280) (h1 : A = (2 / 3) * (B + C)) (h2 : B = (2 / 3) * (A + C)) :
  A + B + C = 700 := by
  sorry

end NUMINAMATH_GPT_total_money_divided_l19_1926


namespace NUMINAMATH_GPT_calculate_fg1_l19_1978

def f (x : ℝ) : ℝ := 5 - 2 * x
def g (x : ℝ) : ℝ := x^3 + 2

theorem calculate_fg1 : f (g 1) = -1 :=
by {
  sorry
}

end NUMINAMATH_GPT_calculate_fg1_l19_1978


namespace NUMINAMATH_GPT_tan_of_angle_in_fourth_quadrant_l19_1976

-- Define the angle α in the fourth quadrant in terms of its cosine value
variable (α : Real)
variable (h1 : α > 3 * Real.pi / 2 ∧ α < 2 * Real.pi) -- fourth quadrant condition
variable (h2 : Real.cos α = 4/5) -- given condition

-- Define the proof problem that tan α equals -3/4 given the conditions
theorem tan_of_angle_in_fourth_quadrant (α : Real) (h1 : α > 3 * Real.pi / 2 ∧ α < 2 * Real.pi) (h2 : Real.cos α = 4/5) : 
  Real.tan α = -3/4 :=
sorry

end NUMINAMATH_GPT_tan_of_angle_in_fourth_quadrant_l19_1976


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l19_1912

-- Define the conditions
def equilateral_triangle_side : ℕ := 15
def isosceles_triangle_side : ℕ := 15
def isosceles_triangle_base : ℕ := 10

-- Define the theorem to prove the perimeter of the isosceles triangle
theorem isosceles_triangle_perimeter : 
  (2 * isosceles_triangle_side + isosceles_triangle_base = 40) :=
by
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l19_1912


namespace NUMINAMATH_GPT_average_price_initial_l19_1998

noncomputable def total_cost_initial (P : ℕ) := 5 * P
noncomputable def total_cost_remaining := 3 * 12
noncomputable def total_cost_returned := 2 * 32

theorem average_price_initial (P : ℕ) : total_cost_initial P = total_cost_remaining + total_cost_returned → P = 20 := 
by
  sorry

end NUMINAMATH_GPT_average_price_initial_l19_1998


namespace NUMINAMATH_GPT_raccoon_carrots_hid_l19_1924

theorem raccoon_carrots_hid 
  (r : ℕ)
  (b : ℕ)
  (h1 : 5 * r = 8 * b)
  (h2 : b = r - 3) 
  : 5 * r = 40 :=
by
  sorry

end NUMINAMATH_GPT_raccoon_carrots_hid_l19_1924


namespace NUMINAMATH_GPT_abs_b_lt_abs_a_lt_2abs_b_l19_1979

variable {a b : ℝ}

theorem abs_b_lt_abs_a_lt_2abs_b (h : (6 * a + 9 * b) / (a + b) < (4 * a - b) / (a - b)) :
  |b| < |a| ∧ |a| < 2 * |b| :=
sorry

end NUMINAMATH_GPT_abs_b_lt_abs_a_lt_2abs_b_l19_1979


namespace NUMINAMATH_GPT_Vasya_distance_fraction_l19_1908

variable (a b c d s : ℝ)

theorem Vasya_distance_fraction :
  (a = b / 2) →
  (c = a + d) →
  (d = s / 10) →
  (a + b + c + d = s) →
  (b / s = 0.4) :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_Vasya_distance_fraction_l19_1908


namespace NUMINAMATH_GPT_problem_l19_1901

def A : Set ℝ := {x | x^2 + x - 6 > 0}
def B : Set ℝ := {x | 0 < x ∧ x < 6}
def C := (Aᶜ) ∩ B

theorem problem : C = {x | 0 < x ∧ x ≤ 2} :=
by
  sorry

end NUMINAMATH_GPT_problem_l19_1901


namespace NUMINAMATH_GPT_part_a_answer_part_b_answer_l19_1951

noncomputable def part_a_problem : Prop :=
  ∃! (x k : ℕ), x > 0 ∧ k > 0 ∧ 3^k - 1 = x^3

noncomputable def part_b_problem (n : ℕ) : Prop :=
  n > 1 ∧ n ≠ 3 → ∀ (x k : ℕ), ¬ (x > 0 ∧ k > 0 ∧ 3^k - 1 = x^n)

theorem part_a_answer : part_a_problem :=
  sorry

theorem part_b_answer (n : ℕ) : part_b_problem n :=
  sorry

end NUMINAMATH_GPT_part_a_answer_part_b_answer_l19_1951


namespace NUMINAMATH_GPT_fraction_irreducible_l19_1910

theorem fraction_irreducible (n : ℤ) : Int.gcd (21 * n + 4) (14 * n + 3) = 1 :=
by 
  sorry

end NUMINAMATH_GPT_fraction_irreducible_l19_1910


namespace NUMINAMATH_GPT_albert_number_solution_l19_1990

theorem albert_number_solution (A B C : ℝ) 
  (h1 : A = 2 * B + 1) 
  (h2 : B = 2 * C + 1) 
  (h3 : C = 2 * A + 2) : 
  A = -11 / 7 := 
by 
  sorry

end NUMINAMATH_GPT_albert_number_solution_l19_1990


namespace NUMINAMATH_GPT_curlers_total_l19_1974

theorem curlers_total (P B G : ℕ) (h1 : 4 * P = P + B + G) (h2 : B = 2 * P) (h3 : G = 4) : 
  4 * P = 16 := 
by sorry

end NUMINAMATH_GPT_curlers_total_l19_1974


namespace NUMINAMATH_GPT_ratio_problem_l19_1911

theorem ratio_problem
  (a b c d e : ℚ)
  (h1 : a / b = 5)
  (h2 : b / c = 1 / 4)
  (h3 : c / d = 7)
  (h4 : d / e = 2) :
  e / a = 2 / 35 := 
sorry

end NUMINAMATH_GPT_ratio_problem_l19_1911


namespace NUMINAMATH_GPT_intersection_vertices_of_regular_octagon_l19_1971

noncomputable def set_A (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | |p.1| + |p.2| = a ∧ a > 0}

def set_B : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | |p.1 * p.2| + 1 = |p.1| + |p.2|}

theorem intersection_vertices_of_regular_octagon (a : ℝ) :
  (∃ (p : ℝ × ℝ), p ∈ set_A a ∧ p ∈ set_B) ↔ (a = Real.sqrt 2 ∨ a = 2 + Real.sqrt 2) :=
  sorry

end NUMINAMATH_GPT_intersection_vertices_of_regular_octagon_l19_1971


namespace NUMINAMATH_GPT_surface_area_of_given_cube_l19_1949

-- Define the cube with its volume
def volume_of_cube : ℝ := 4913

-- Define the side length of the cube
def side_of_cube : ℝ := volume_of_cube^(1/3)

-- Define the surface area of the cube
def surface_area_of_cube (side : ℝ) : ℝ := 6 * (side^2)

-- Statement of the theorem
theorem surface_area_of_given_cube : 
  surface_area_of_cube side_of_cube = 1734 := 
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_surface_area_of_given_cube_l19_1949


namespace NUMINAMATH_GPT_difference_of_squares_l19_1932

theorem difference_of_squares (x y : ℝ) (h1 : x + y = 10) (h2 : x - y = 19) : x^2 - y^2 = 190 :=
by
  sorry

end NUMINAMATH_GPT_difference_of_squares_l19_1932


namespace NUMINAMATH_GPT_original_height_l19_1966

theorem original_height (total_travel : ℝ) (h : ℝ) (half: h/2 = (1/2 * h)): 
  (total_travel = h + 2 * (h / 2) + 2 * (h / 4)) → total_travel = 260 → h = 104 :=
by
  intro travel_eq
  intro travel_value
  sorry

end NUMINAMATH_GPT_original_height_l19_1966


namespace NUMINAMATH_GPT_incorrect_conclusion_l19_1922

theorem incorrect_conclusion (a b c : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : b > c) (h4 : c > 0) : ¬ (a / b > a / c) :=
sorry

end NUMINAMATH_GPT_incorrect_conclusion_l19_1922


namespace NUMINAMATH_GPT_find_A_and_B_l19_1994

theorem find_A_and_B : 
  ∃ A B : ℝ, 
    (A = 6.5 ∧ B = 0.5) ∧
    (∀ x : ℝ, (8 * x - 17) / ((3 * x + 5) * (x - 3)) = A / (3 * x + 5) + B / (x - 3)) :=
by
  sorry

end NUMINAMATH_GPT_find_A_and_B_l19_1994


namespace NUMINAMATH_GPT_equilateral_triangle_of_roots_of_unity_l19_1930

open Complex

/-- Given three distinct non-zero complex numbers z1, z2, z3 such that z1 * z2 = z3 ^ 2 and z2 * z3 = z1 ^ 2.
Prove that if z2 = z1 * alpha, then alpha is a cube root of unity and the points corresponding to z1, z2, z3
form an equilateral triangle in the complex plane -/
theorem equilateral_triangle_of_roots_of_unity {z1 z2 z3 : ℂ} (h1 : z1 ≠ 0) (h2 : z2 ≠ 0) (h3 : z3 ≠ 0)
  (h_distinct : z1 ≠ z2 ∧ z2 ≠ z3 ∧ z1 ≠ z3)
  (h1_2 : z1 * z2 = z3 ^ 2) (h2_3 : z2 * z3 = z1 ^ 2) (alpha : ℂ) (hz2 : z2 = z1 * alpha) :
  alpha^3 = 1 ∧ ∃ (w1 w2 w3 : ℂ), (w1 = z1) ∧ (w2 = z2) ∧ (w3 = z3) ∧ ((w1, w2, w3) = (z1, z1 * α, z3) 
  ∨ (w1, w2, w3) = (z3, z1, z1 * α) ∨ (w1, w2, w3) = (z1 * α, z3, z1)) 
  ∧ dist w1 w2 = dist w2 w3 ∧ dist w2 w3 = dist w3 w1 := sorry

end NUMINAMATH_GPT_equilateral_triangle_of_roots_of_unity_l19_1930


namespace NUMINAMATH_GPT_problem_statement_l19_1900

noncomputable def find_sum (x y : ℝ) : ℝ := x + y

theorem problem_statement (x y : ℝ)
  (hx : |x| + x + y = 12)
  (hy : x + |y| - y = 14) :
  find_sum x y = 22 / 5 :=
sorry

end NUMINAMATH_GPT_problem_statement_l19_1900


namespace NUMINAMATH_GPT_spending_spring_months_l19_1909

theorem spending_spring_months (spend_end_March spend_end_June : ℝ)
  (h1 : spend_end_March = 1) (h2 : spend_end_June = 4) :
  (spend_end_June - spend_end_March) = 3 :=
by
  rw [h1, h2]
  norm_num

end NUMINAMATH_GPT_spending_spring_months_l19_1909


namespace NUMINAMATH_GPT_inequality_solution_set_empty_range_l19_1955

theorem inequality_solution_set_empty_range (m : ℝ) :
  (∀ x : ℝ, mx^2 - mx - 1 < 0) ↔ -4 < m ∧ m ≤ 0 :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_set_empty_range_l19_1955


namespace NUMINAMATH_GPT_shaded_area_of_larger_circle_l19_1928

theorem shaded_area_of_larger_circle (R r : ℝ) (A_larger A_smaller : ℝ)
  (hR : R = 9)
  (hr : r = 4.5)
  (hA_larger : A_larger = Real.pi * R^2)
  (hA_smaller : A_smaller = 3 * Real.pi * r^2) :
  A_larger - A_smaller = 20.25 * Real.pi := by
  sorry

end NUMINAMATH_GPT_shaded_area_of_larger_circle_l19_1928


namespace NUMINAMATH_GPT_total_weight_of_beef_l19_1918

-- Define the conditions
def packages_weight := 4
def first_butcher_packages := 10
def second_butcher_packages := 7
def third_butcher_packages := 8

-- Define the total weight calculation
def total_weight := (first_butcher_packages * packages_weight) +
                    (second_butcher_packages * packages_weight) +
                    (third_butcher_packages * packages_weight)

-- The statement to prove
theorem total_weight_of_beef : total_weight = 100 := by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_total_weight_of_beef_l19_1918


namespace NUMINAMATH_GPT_power_sum_is_99_l19_1997

theorem power_sum_is_99 : 3^4 + (-3)^3 + (-3)^2 + (-3)^1 + 3^1 + 3^2 + 3^3 = 99 :=
by sorry

end NUMINAMATH_GPT_power_sum_is_99_l19_1997


namespace NUMINAMATH_GPT_AM_GM_inequality_l19_1981

theorem AM_GM_inequality (a b c d : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) : 
  (a / b + b / c + c / d + d / a) ≥ 4 := 
sorry

end NUMINAMATH_GPT_AM_GM_inequality_l19_1981


namespace NUMINAMATH_GPT_original_set_cardinality_l19_1987

-- Definitions based on conditions
def is_reversed_error (n : ℕ) : Prop :=
  ∃ (A B C : ℕ), 100 * A + 10 * B + C = n ∧ 100 * C + 10 * B + A = n + 198 ∧ C - A = 2

-- The theorem to prove
theorem original_set_cardinality : ∃ n : ℕ, is_reversed_error n ∧ n = 10 := by
  sorry

end NUMINAMATH_GPT_original_set_cardinality_l19_1987


namespace NUMINAMATH_GPT_poly_sum_of_squares_iff_nonneg_l19_1989

open Polynomial

variable {R : Type*} [Ring R] [OrderedRing R]

theorem poly_sum_of_squares_iff_nonneg (A : Polynomial ℝ) :
  (∃ P Q : Polynomial ℝ, A = P^2 + Q^2) ↔ ∀ x : ℝ, 0 ≤ A.eval x := sorry

end NUMINAMATH_GPT_poly_sum_of_squares_iff_nonneg_l19_1989


namespace NUMINAMATH_GPT_find_number_l19_1952

theorem find_number : ∃ x : ℝ, 3 * x - 1 = 2 * x ∧ x = 1 := sorry

end NUMINAMATH_GPT_find_number_l19_1952


namespace NUMINAMATH_GPT_length_of_water_fountain_l19_1991

theorem length_of_water_fountain :
  (∀ (L1 : ℕ), 20 * 14 = L1) ∧
  (35 * 3 = 21) →
  (20 * 14 = 56) := by
sorry

end NUMINAMATH_GPT_length_of_water_fountain_l19_1991


namespace NUMINAMATH_GPT_find_y_l19_1964

theorem find_y (x y : ℝ) (h1 : 2 * (x - y) = 12) (h2 : x + y = 14) : y = 4 := 
by
  sorry

end NUMINAMATH_GPT_find_y_l19_1964


namespace NUMINAMATH_GPT_distance_between_Q_and_R_l19_1972

noncomputable def distance_QR : Real :=
  let YZ := 9
  let XZ := 12
  let XY := 15
  
  -- assume QY = QX and tangent to YZ at Y, and RX = RY and tangent to XZ at X
  let QY := 12.5
  let QX := 12.5
  let RY := 12.5
  let RX := 12.5

  -- calculate and return the distance QR based on these assumptions
  (QX^2 + RY^2 - 2 * QX * RX * Real.cos 90)^(1/2)

theorem distance_between_Q_and_R (YZ XZ XY : ℝ) (QY QX RY RX : ℝ) (h1 : YZ = 9) (h2 : XZ = 12) (h3 : XY = 15)
  (h4 : QY = 12.5) (h5 : QX = 12.5) (h6 : RY = 12.5) (h7 : RX = 12.5) :
  distance_QR = 15 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_Q_and_R_l19_1972


namespace NUMINAMATH_GPT_base7_addition_problem_l19_1946

theorem base7_addition_problem
  (X Y : ℕ) :
  (5 * 7^1 + X * 7^0 + Y * 7^0 + 0 * 7^2 + 6 * 7^1 + 2 * 7^0) = (6 * 7^1 + 4 * 7^0 + X * 7^0 + 0 * 7^2) →
  X + 6 = 1 * 7 + 4 →
  Y + 2 = X →
  X + Y = 8 :=
by
  intro h1 h2 h3
  sorry

end NUMINAMATH_GPT_base7_addition_problem_l19_1946


namespace NUMINAMATH_GPT_friend_reading_time_l19_1943

theorem friend_reading_time (S : ℝ) (H1 : S > 0) (H2 : 3 = 2 * (3 / 2)) : 
  (1.5 / (5 * S)) = 0.3 :=
by 
  sorry

end NUMINAMATH_GPT_friend_reading_time_l19_1943


namespace NUMINAMATH_GPT_mechanical_moles_l19_1959

-- Define the conditions
def condition_one (x y : ℝ) : Prop :=
  x + y = 1 / 5

def condition_two (x y : ℝ) : Prop :=
  (1 / (3 * x)) + (2 / (3 * y)) = 10

-- Define the main theorem using the defined conditions
theorem mechanical_moles (x y : ℝ) (h1 : condition_one x y) (h2 : condition_two x y) :
  x = 1 / 30 ∧ y = 1 / 6 :=
  sorry

end NUMINAMATH_GPT_mechanical_moles_l19_1959


namespace NUMINAMATH_GPT_negative_fraction_comparison_l19_1967

theorem negative_fraction_comparison : (-3/4 : ℚ) > (-4/5 : ℚ) :=
sorry

end NUMINAMATH_GPT_negative_fraction_comparison_l19_1967


namespace NUMINAMATH_GPT_regular_polygon_exterior_angle_l19_1921

theorem regular_polygon_exterior_angle (n : ℕ) (h : 1 ≤ n) :
  (360 : ℝ) / (n : ℝ) = 60 → n = 6 :=
by
  intro h1
  sorry

end NUMINAMATH_GPT_regular_polygon_exterior_angle_l19_1921


namespace NUMINAMATH_GPT_fraction_product_l19_1913

theorem fraction_product : (2 / 9) * (5 / 11) = 10 / 99 := 
by
  sorry

end NUMINAMATH_GPT_fraction_product_l19_1913


namespace NUMINAMATH_GPT_elsa_final_marbles_l19_1988

def initial_marbles : ℕ := 40
def marbles_lost_at_breakfast : ℕ := 3
def marbles_given_to_susie : ℕ := 5
def marbles_bought_by_mom : ℕ := 12
def twice_marbles_given_back : ℕ := 2 * marbles_given_to_susie

theorem elsa_final_marbles :
    initial_marbles
    - marbles_lost_at_breakfast
    - marbles_given_to_susie
    + marbles_bought_by_mom
    + twice_marbles_given_back = 54 := 
by
    sorry

end NUMINAMATH_GPT_elsa_final_marbles_l19_1988


namespace NUMINAMATH_GPT_decimal_equivalent_of_fraction_l19_1933

theorem decimal_equivalent_of_fraction :
  (16 : ℚ) / 50 = 32 / 100 :=
by sorry

end NUMINAMATH_GPT_decimal_equivalent_of_fraction_l19_1933


namespace NUMINAMATH_GPT_rows_per_floor_l19_1917

theorem rows_per_floor
  (right_pos : ℕ) (left_pos : ℕ)
  (floors : ℕ) (total_cars : ℕ)
  (h_right : right_pos = 5) (h_left : left_pos = 4)
  (h_floors : floors = 10) (h_total : total_cars = 1600) :
  ∃ rows_per_floor : ℕ, rows_per_floor = 20 :=
by {
  sorry
}

end NUMINAMATH_GPT_rows_per_floor_l19_1917


namespace NUMINAMATH_GPT_marble_prism_weight_l19_1915

theorem marble_prism_weight :
  let height := 8
  let base_side := 2
  let density := 2700
  let volume := base_side * base_side * height
  volume * density = 86400 :=
by
  let height := 8
  let base_side := 2
  let density := 2700
  let volume := base_side * base_side * height
  sorry

end NUMINAMATH_GPT_marble_prism_weight_l19_1915


namespace NUMINAMATH_GPT_option_C_correct_l19_1982

theorem option_C_correct (a : ℤ) : (a = 3 → a = a + 1 → a = 4) :=
by {
  sorry
}

end NUMINAMATH_GPT_option_C_correct_l19_1982


namespace NUMINAMATH_GPT_range_of_a_l19_1992

variable (a x y : ℝ)

def proposition_p : Prop :=
  ∀ x : ℝ, a * x^2 + a * x + 1 > 0

def proposition_q : Prop :=
  (1 - a) * (a - 3) < 0

theorem range_of_a (h1 : proposition_p a) (h2 : proposition_q a) : 
  (0 ≤ a ∧ a < 1) ∨ (3 < a ∧ a < 4) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l19_1992


namespace NUMINAMATH_GPT_correlation_height_weight_l19_1904

def is_functional_relationship (pair: String) : Prop :=
  pair = "The area of a square and its side length" ∨
  pair = "The distance traveled by a vehicle moving at a constant speed and time"

def has_no_correlation (pair: String) : Prop :=
  pair = "A person's height and eyesight"

def is_correlation (pair: String) : Prop :=
  ¬ is_functional_relationship pair ∧ ¬ has_no_correlation pair

theorem correlation_height_weight :
  is_correlation "A person's height and weight" :=
by sorry

end NUMINAMATH_GPT_correlation_height_weight_l19_1904


namespace NUMINAMATH_GPT_simplify_evaluate_l19_1906

def f (x y : ℝ) : ℝ := 4 * x^2 * y - (6 * x * y - 3 * (4 * x - 2) - x^2 * y) + 1

theorem simplify_evaluate : f (-2) (1/2) = -13 := by
  sorry

end NUMINAMATH_GPT_simplify_evaluate_l19_1906


namespace NUMINAMATH_GPT_total_flags_l19_1985

theorem total_flags (x : ℕ) (hx1 : 4 * x + 20 > 8 * (x - 1)) (hx2 : 4 * x + 20 < 8 * x) : 4 * 6 + 20 = 44 :=
by sorry

end NUMINAMATH_GPT_total_flags_l19_1985


namespace NUMINAMATH_GPT_average_of_remaining_two_numbers_l19_1996

theorem average_of_remaining_two_numbers (a b c d e f : ℝ) 
  (h1 : (a + b + c + d + e + f) / 6 = 2.5)
  (h2 : (a + b) / 2 = 1.1)
  (h3 : (c + d) / 2 = 1.4) : 
  (e + f) / 2 = 5 :=
by
  sorry

end NUMINAMATH_GPT_average_of_remaining_two_numbers_l19_1996


namespace NUMINAMATH_GPT_intersection_of_M_and_N_l19_1961

def M : Set ℝ := { x | |x + 1| ≤ 1}

def N : Set ℝ := {-1, 0, 1}

theorem intersection_of_M_and_N : M ∩ N = {-1, 0} :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_M_and_N_l19_1961


namespace NUMINAMATH_GPT_santa_chocolate_candies_l19_1953

theorem santa_chocolate_candies (C M : ℕ) (h₁ : C + M = 2023) (h₂ : C = 3 * M / 4) : C = 867 :=
sorry

end NUMINAMATH_GPT_santa_chocolate_candies_l19_1953


namespace NUMINAMATH_GPT_no_solution_eq1_l19_1905

   theorem no_solution_eq1 : ¬ ∃ x, (3 - x) / (x - 4) - 1 / (4 - x) = 1 :=
   by
     sorry
   
end NUMINAMATH_GPT_no_solution_eq1_l19_1905


namespace NUMINAMATH_GPT_average_of_remaining_ten_numbers_l19_1977

theorem average_of_remaining_ten_numbers
  (avg_50 : ℝ)
  (n_50 : ℝ)
  (avg_40 : ℝ)
  (n_40 : ℝ)
  (sum_50 : n_50 * avg_50 = 3800)
  (sum_40 : n_40 * avg_40 = 3200)
  (n_10 : n_50 - n_40 = 10)
  : (3800 - 3200) / 10 = 60 :=
by
  sorry

end NUMINAMATH_GPT_average_of_remaining_ten_numbers_l19_1977


namespace NUMINAMATH_GPT_problem_statement_l19_1954

theorem problem_statement (m n : ℝ) (h1 : 1 + 27 = m) (h2 : 3 + 9 = n) : |m - n| = 16 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l19_1954


namespace NUMINAMATH_GPT_cars_on_river_road_l19_1907

theorem cars_on_river_road (B C : ℕ) (h1 : B = C - 40) (h2 : B * 3 = C) : C = 60 := 
sorry

end NUMINAMATH_GPT_cars_on_river_road_l19_1907
