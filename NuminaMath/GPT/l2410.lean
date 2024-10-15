import Mathlib

namespace NUMINAMATH_GPT_smallest_digit_to_correct_sum_l2410_241040

theorem smallest_digit_to_correct_sum :
  ∃ (d : ℕ), d = 3 ∧
  (3 ∈ [3, 5, 7]) ∧
  (371 + 569 + 784 + (d*100) = 1824) := sorry

end NUMINAMATH_GPT_smallest_digit_to_correct_sum_l2410_241040


namespace NUMINAMATH_GPT_second_point_x_coord_l2410_241078

open Function

variable (n : ℝ)

def line_eq (y : ℝ) : ℝ := 2 * y + 5

theorem second_point_x_coord (h₁ : ∀ (x y : ℝ), x = line_eq y → True) :
  ∃ m : ℝ, ∀ n : ℝ, m = 2 * n + 5 → (m + 1 = line_eq (n + 0.5)) :=
by
  sorry

end NUMINAMATH_GPT_second_point_x_coord_l2410_241078


namespace NUMINAMATH_GPT_nancy_weight_l2410_241089

theorem nancy_weight (w : ℕ) (h : (60 * w) / 100 = 54) : w = 90 :=
by
  sorry

end NUMINAMATH_GPT_nancy_weight_l2410_241089


namespace NUMINAMATH_GPT_union_of_A_and_B_l2410_241069

def A : Set Int := {-1, 1, 2, 4}
def B : Set Int := {-1, 0, 2}

theorem union_of_A_and_B :
  A ∪ B = {-1, 0, 1, 2, 4} :=
by
  sorry

end NUMINAMATH_GPT_union_of_A_and_B_l2410_241069


namespace NUMINAMATH_GPT_basic_astrophysics_degrees_l2410_241075

def percentages : List ℚ := [12, 22, 14, 27, 7, 5, 3, 4]

def total_budget_percentage : ℚ := 100

def degrees_in_circle : ℚ := 360

def remaining_percentage (lst : List ℚ) (total : ℚ) : ℚ :=
  total - lst.sum / 100  -- convert sum to percentage

def degrees_of_percentage (percent : ℚ) (circle_degrees : ℚ) : ℚ :=
  percent * (circle_degrees / total_budget_percentage) -- conversion rate per percentage point

theorem basic_astrophysics_degrees :
  degrees_of_percentage (remaining_percentage percentages total_budget_percentage) degrees_in_circle = 21.6 :=
by
  sorry

end NUMINAMATH_GPT_basic_astrophysics_degrees_l2410_241075


namespace NUMINAMATH_GPT_find_original_price_of_petrol_l2410_241077

open Real

noncomputable def original_price_of_petrol (P : ℝ) : Prop :=
  ∀ G : ℝ, 
  (G * P = 300) ∧ 
  ((G + 7) * 0.85 * P = 300) → 
  P = 7.56

-- Theorems should ideally be defined within certain scopes or namespaces
theorem find_original_price_of_petrol (P : ℝ) : original_price_of_petrol P :=
  sorry

end NUMINAMATH_GPT_find_original_price_of_petrol_l2410_241077


namespace NUMINAMATH_GPT_find_number_is_9_l2410_241073

noncomputable def number (y : ℕ) : ℕ := 3^(12 / y)

theorem find_number_is_9 (y : ℕ) (h_y : y = 6) (h_eq : (number y)^y = 3^12) : number y = 9 :=
by
  sorry

end NUMINAMATH_GPT_find_number_is_9_l2410_241073


namespace NUMINAMATH_GPT_line_eq_l2410_241038

theorem line_eq (x y : ℝ) :
  (∃ (x1 y1 x2 y2 : ℝ), x1 = 5 ∧ y1 = 0 ∧ x2 = 2 ∧ y2 = -5 ∧
    (y - y1) / (x - x1) = (y2 - y1) / (x2 - x1)) →
  5 * x - 3 * y - 25 = 0 :=
sorry

end NUMINAMATH_GPT_line_eq_l2410_241038


namespace NUMINAMATH_GPT_calc_expression_l2410_241028

theorem calc_expression :
  (8^5 / 8^3) * 3^6 = 46656 := by
  sorry

end NUMINAMATH_GPT_calc_expression_l2410_241028


namespace NUMINAMATH_GPT_range_of_a_outside_circle_l2410_241025

  variable (a : ℝ)

  def point_outside_circle (a : ℝ) : Prop :=
    let x := a
    let y := 2
    let distance_sqr := (x - a) ^ 2 + (y - 3 / 2) ^ 2
    let r_sqr := 1 / 4
    distance_sqr > r_sqr

  theorem range_of_a_outside_circle {a : ℝ} (h : point_outside_circle a) :
      2 < a ∧ a < 9 / 4 := sorry
  
end NUMINAMATH_GPT_range_of_a_outside_circle_l2410_241025


namespace NUMINAMATH_GPT_sum_of_ai_powers_l2410_241092

theorem sum_of_ai_powers :
  ∀ (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 : ℝ),
  (∀ x : ℝ, (1 + x) * (1 - 2 * x)^8 = 
            a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + 
            a_4 * x^4 + a_5 * x^5 + a_6 * x^6 + 
            a_7 * x^7 + a_8 * x^8 + a_9 * x^9) →
  a_1 * 2 + a_2 * 2^2 + a_3 * 2^3 + 
  a_4 * 2^4 + a_5 * 2^5 + a_6 * 2^6 + 
  a_7 * 2^7 + a_8 * 2^8 + a_9 * 2^9 = 3^9 - 1 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_ai_powers_l2410_241092


namespace NUMINAMATH_GPT_molecular_weight_Dinitrogen_pentoxide_l2410_241084

theorem molecular_weight_Dinitrogen_pentoxide :
  let atomic_weight_N := 14.01
  let atomic_weight_O := 16.00
  let molecular_formula := (2 * atomic_weight_N) + (5 * atomic_weight_O)
  molecular_formula = 108.02 :=
by
  sorry

end NUMINAMATH_GPT_molecular_weight_Dinitrogen_pentoxide_l2410_241084


namespace NUMINAMATH_GPT_g_g_g_g_3_l2410_241055

def g (x : ℕ) : ℕ :=
  if x % 2 = 0 then x / 2 else 5 * x + 3

theorem g_g_g_g_3 : g (g (g (g 3))) = 24 := by
  sorry

end NUMINAMATH_GPT_g_g_g_g_3_l2410_241055


namespace NUMINAMATH_GPT_midpoint_on_hyperbola_l2410_241043

theorem midpoint_on_hyperbola : ∃ A B : ℝ × ℝ, 
  (A.1 ^ 2 - A.2 ^ 2 / 9 = 1) ∧ (B.1 ^ 2 - B.2 ^ 2 / 9 = 1) ∧
  (∃ (M : ℝ × ℝ), M = (1/2 * (A.1 + B.1), 1/2 * (A.2 + B.2)) ∧ 
    (M = (-1, -4))) := 
sorry

end NUMINAMATH_GPT_midpoint_on_hyperbola_l2410_241043


namespace NUMINAMATH_GPT_equilateral_prism_lateral_edge_length_l2410_241057

theorem equilateral_prism_lateral_edge_length
  (base_side_length : ℝ)
  (h_base : base_side_length = 1)
  (perpendicular_diagonals : ∃ a b c : ℝ, a^2 + b^2 = c^2 ∧ a = base_side_length ∧ b = lateral_edge ∧ c = some_diagonal_length ∧ lateral_edge ≠ 0)
  : ∀ lateral_edge : ℝ, lateral_edge = (Real.sqrt 2) / 2 := sorry

end NUMINAMATH_GPT_equilateral_prism_lateral_edge_length_l2410_241057


namespace NUMINAMATH_GPT_hyperbola_equation_l2410_241096

theorem hyperbola_equation (a b : ℝ) (h₁ : a^2 + b^2 = 25) (h₂ : 2 * b / a = 1) : 
  a = 2 * Real.sqrt 5 ∧ b = Real.sqrt 5 ∧ (∀ x y : ℝ, x^2 / (a^2) - y^2 / (b^2) = 1 ↔ x^2 / 20 - y^2 / 5 = 1) :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_equation_l2410_241096


namespace NUMINAMATH_GPT_trapezoid_area_l2410_241017

theorem trapezoid_area (base1 base2 height : ℕ) (h_base1 : base1 = 9) (h_base2 : base2 = 11) (h_height : height = 3) :
  (1 / 2 : ℚ) * (base1 + base2 : ℕ) * height = 30 :=
by
  sorry

end NUMINAMATH_GPT_trapezoid_area_l2410_241017


namespace NUMINAMATH_GPT_octahedron_plane_intersection_l2410_241023

theorem octahedron_plane_intersection 
  (s : ℝ) 
  (a b c : ℕ) 
  (ha : Nat.Coprime a c) 
  (hb : ∀ p : ℕ, Prime p → p^2 ∣ b → False) 
  (hs : s = 2) 
  (hangle : ∀ θ, θ = 45 ∧ θ = 45) 
  (harea : ∃ A, A = (s^2 * Real.sqrt 3) / 2 ∧ A = a * Real.sqrt b / c): 
  a + b + c = 11 := 
by 
  sorry

end NUMINAMATH_GPT_octahedron_plane_intersection_l2410_241023


namespace NUMINAMATH_GPT_f_satisfies_conditions_l2410_241094

def g (n : Int) : Int :=
  if n >= 1 then 1 else 0

def f (n m : Int) : Int :=
  if m = 0 then n
  else n % m

theorem f_satisfies_conditions (n m : Int) : 
  (f 0 m = 0) ∧ 
  (f (n + 1) m = (1 - g m + g m * g (m - 1 - f n m)) * (1 + f n m)) := by
  sorry

end NUMINAMATH_GPT_f_satisfies_conditions_l2410_241094


namespace NUMINAMATH_GPT_find_C_l2410_241054

theorem find_C (A B C D E : ℕ) (h1 : A < 10) (h2 : B < 10) (h3 : C < 10) (h4 : D < 10) (h5 : E < 10) 
  (h : 4 * (10 * (10000 * A + 1000 * B + 100 * C + 10 * D + E) + 4) = 400000 + (10000 * A + 1000 * B + 100 * C + 10 * D + E)) : 
  C = 2 :=
sorry

end NUMINAMATH_GPT_find_C_l2410_241054


namespace NUMINAMATH_GPT_reciprocal_eq_self_l2410_241010

theorem reciprocal_eq_self (x : ℝ) : (1 / x = x) ↔ (x = 1 ∨ x = -1) :=
sorry

end NUMINAMATH_GPT_reciprocal_eq_self_l2410_241010


namespace NUMINAMATH_GPT_simplify_root_exponentiation_l2410_241035

theorem simplify_root_exponentiation : (7 ^ (1 / 3) : ℝ) ^ 6 = 49 := by
  sorry

end NUMINAMATH_GPT_simplify_root_exponentiation_l2410_241035


namespace NUMINAMATH_GPT_clock_angle_at_3_15_l2410_241053

-- Conditions
def full_circle_degrees : ℕ := 360
def hour_degree : ℕ := full_circle_degrees / 12
def minute_degree : ℕ := full_circle_degrees / 60
def minute_position (m : ℕ) : ℕ := m * minute_degree
def hour_position (h m : ℕ) : ℕ := h * hour_degree + m * (hour_degree / 60)

-- Theorem to prove
theorem clock_angle_at_3_15 : (|minute_position 15 - hour_position 3 15| : ℚ) = 7.5 := by
  sorry

end NUMINAMATH_GPT_clock_angle_at_3_15_l2410_241053


namespace NUMINAMATH_GPT_remainder_of_division_l2410_241014

theorem remainder_of_division (x r : ℕ) (h1 : 1620 - x = 1365) (h2 : 1620 = x * 6 + r) : r = 90 :=
sorry

end NUMINAMATH_GPT_remainder_of_division_l2410_241014


namespace NUMINAMATH_GPT_largest_percentage_increase_l2410_241000

def students_2003 := 80
def students_2004 := 88
def students_2005 := 94
def students_2006 := 106
def students_2007 := 130

theorem largest_percentage_increase :
  let incr_03_04 := (students_2004 - students_2003) / students_2003 * 100
  let incr_04_05 := (students_2005 - students_2004) / students_2004 * 100
  let incr_05_06 := (students_2006 - students_2005) / students_2005 * 100
  let incr_06_07 := (students_2007 - students_2006) / students_2006 * 100
  incr_06_07 > incr_03_04 ∧
  incr_06_07 > incr_04_05 ∧
  incr_06_07 > incr_05_06 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_largest_percentage_increase_l2410_241000


namespace NUMINAMATH_GPT_triangle_length_AX_l2410_241066

theorem triangle_length_AX (A B C X : Type*) (AB AC BC AX XB : ℝ)
  (hAB : AB = 70) (hAC : AC = 42) (hBC : BC = 56)
  (h_bisect : ∃ (k : ℝ), AX = 3 * k ∧ XB = 4 * k) :
  AX = 30 := 
by
  sorry

end NUMINAMATH_GPT_triangle_length_AX_l2410_241066


namespace NUMINAMATH_GPT_problem_ab_value_l2410_241064

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  if x ≥ 0 then 3 * x^2 - 4 * x else a * x^2 + b * x

theorem problem_ab_value (a b : ℝ) :
  (∀ x : ℝ, f x a b = f (-x) a b) → a * b = 12 :=
by
  intro h
  let f_eqn := h 1 -- Checking the function equality for x = 1
  sorry

end NUMINAMATH_GPT_problem_ab_value_l2410_241064


namespace NUMINAMATH_GPT_show_revenue_l2410_241098

theorem show_revenue (tickets_first_showing : ℕ) 
                     (tickets_second_showing : ℕ) 
                     (ticket_price : ℕ) :
                      tickets_first_showing = 200 →
                      tickets_second_showing = 3 * tickets_first_showing →
                      ticket_price = 25 →
                      (tickets_first_showing + tickets_second_showing) * ticket_price = 20000 :=
by
  intros h1 h2 h3
  have h4 : tickets_first_showing + tickets_second_showing = 800 := sorry -- Calculation step
  have h5 : (tickets_first_showing + tickets_second_showing) * ticket_price = 20000 := sorry -- Calculation step
  exact h5

end NUMINAMATH_GPT_show_revenue_l2410_241098


namespace NUMINAMATH_GPT_num_three_digit_integers_with_odd_factors_l2410_241045

theorem num_three_digit_integers_with_odd_factors : 
  ∃ n : ℕ, n = 22 ∧ ∀ k : ℕ, (10 ≤ k ∧ k ≤ 31) ↔ (100 ≤ k^2 ∧ k^2 ≤ 999) := 
sorry

end NUMINAMATH_GPT_num_three_digit_integers_with_odd_factors_l2410_241045


namespace NUMINAMATH_GPT_gcd_incorrect_l2410_241052

theorem gcd_incorrect (a b c : ℕ) (h : a * b * c = 3000) : gcd (gcd a b) c ≠ 15 := 
sorry

end NUMINAMATH_GPT_gcd_incorrect_l2410_241052


namespace NUMINAMATH_GPT_unit_prices_and_purchasing_schemes_l2410_241080

theorem unit_prices_and_purchasing_schemes :
  ∃ (x y : ℕ),
    (14 * x + 8 * y = 1600) ∧
    (3 * x = 4 * y) ∧
    (x = 80) ∧ 
    (y = 60) ∧
    ∃ (m : ℕ), 
      (m ≥ 29) ∧ 
      (m ≤ 30) ∧ 
      (80 * m + 60 * (50 - m) ≤ 3600) ∧
      (m = 29 ∨ m = 30) := 
sorry

end NUMINAMATH_GPT_unit_prices_and_purchasing_schemes_l2410_241080


namespace NUMINAMATH_GPT_find_f_pi_l2410_241065

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ :=
  Real.tan (ω * x + Real.pi / 3)

theorem find_f_pi (ω : ℝ) (h_positive : ω > 0) (h_period : Real.pi / ω = 3 * Real.pi) :
  f (ω := ω) Real.pi = -Real.sqrt 3 :=
by
  -- ω is given to be 1/3 by the condition h_period, substituting that 
  -- directly might be clearer for stating the problem accurately
  have h_omega : ω = 1 / 3 := by
    sorry
  rw [h_omega]
  sorry


end NUMINAMATH_GPT_find_f_pi_l2410_241065


namespace NUMINAMATH_GPT_smallest_real_number_l2410_241018

theorem smallest_real_number :
  ∃ (x : ℝ), x = -3 ∧ (∀ (y : ℝ), y = 0 ∨ y = (-1/3)^2 ∨ y = -((27:ℝ)^(1/3)) ∨ y = -2 → x ≤ y) := 
by 
  sorry

end NUMINAMATH_GPT_smallest_real_number_l2410_241018


namespace NUMINAMATH_GPT_geom_seq_a4_l2410_241046

theorem geom_seq_a4 (a : ℕ → ℝ) (r : ℝ)
  (h : ∀ n, a (n + 1) = a n * r)
  (h2 : a 3 = 9)
  (h3 : a 5 = 1) :
  a 4 = 3 ∨ a 4 = -3 :=
by {
  sorry
}

end NUMINAMATH_GPT_geom_seq_a4_l2410_241046


namespace NUMINAMATH_GPT_grid_divisible_by_rectangles_l2410_241020

theorem grid_divisible_by_rectangles (n : ℕ) :
  (∃ m : ℕ, n * n = 7 * m) ↔ (∃ k : ℕ, n = 7 * k ∧ k > 1) :=
by
  sorry

end NUMINAMATH_GPT_grid_divisible_by_rectangles_l2410_241020


namespace NUMINAMATH_GPT_log_arith_example_l2410_241027

noncomputable def log10 (x : ℝ) : ℝ := sorry -- Assume the definition of log base 10

theorem log_arith_example : log10 4 + 2 * log10 5 + 8^(2/3) = 6 := 
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_log_arith_example_l2410_241027


namespace NUMINAMATH_GPT_factorization_check_l2410_241086

theorem factorization_check 
  (A : 4 - x^2 + 3 * x ≠ (2 - x) * (2 + x) + 3)
  (B : -x^2 + 3 * x + 4 ≠ -(x + 4) * (x - 1))
  (D : x^2 * y - x * y + x^3 * y ≠ x * (x * y - y + x^2 * y)) :
  1 - 2 * x + x^2 = (1 - x) ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_factorization_check_l2410_241086


namespace NUMINAMATH_GPT_chess_team_girls_l2410_241007

theorem chess_team_girls (B G : ℕ) (h1 : B + G = 26) (h2 : (G / 2) + B = 16) : G = 20 := by
  sorry

end NUMINAMATH_GPT_chess_team_girls_l2410_241007


namespace NUMINAMATH_GPT_lcm_of_1_to_12_l2410_241034

noncomputable def lcm_1_to_12 : ℕ := 2^3 * 3^2 * 5 * 7 * 11

theorem lcm_of_1_to_12 : lcm_1_to_12 = 27720 := by
  sorry

end NUMINAMATH_GPT_lcm_of_1_to_12_l2410_241034


namespace NUMINAMATH_GPT_optimal_rental_plan_l2410_241013

theorem optimal_rental_plan (a b x y : ℕ)
  (h1 : 2 * a + b = 10)
  (h2 : a + 2 * b = 11)
  (h3 : 31 = 3 * x + 4 * y)
  (cost_a : ℕ := 100)
  (cost_b : ℕ := 120) :
  ∃ x y, 3 * x + 4 * y = 31 ∧ cost_a * x + cost_b * y = 940 := by
  sorry

end NUMINAMATH_GPT_optimal_rental_plan_l2410_241013


namespace NUMINAMATH_GPT_frequency_of_group5_l2410_241058

-- Define the total number of students and the frequencies of each group
def total_students : ℕ := 40
def freq_group1 : ℕ := 12
def freq_group2 : ℕ := 10
def freq_group3 : ℕ := 6
def freq_group4 : ℕ := 8

-- Define the frequency of the fifth group in terms of the above frequencies
def freq_group5 : ℕ := total_students - (freq_group1 + freq_group2 + freq_group3 + freq_group4)

-- The theorem to be proven
theorem frequency_of_group5 : freq_group5 = 4 := by
  -- Proof goes here, skipped with sorry
  sorry

end NUMINAMATH_GPT_frequency_of_group5_l2410_241058


namespace NUMINAMATH_GPT_infinite_non_congruent_right_triangles_l2410_241082

noncomputable def right_triangle_equal_perimeter_area : Prop :=
  ∃ (a b c : ℝ), (a > 0) ∧ (b > 0) ∧ (c > 0) ∧ (a^2 + b^2 = c^2) ∧ 
  (a + b + c = (1/2) * a * b)

theorem infinite_non_congruent_right_triangles :
  ∃ (k : ℕ), right_triangle_equal_perimeter_area :=
sorry

end NUMINAMATH_GPT_infinite_non_congruent_right_triangles_l2410_241082


namespace NUMINAMATH_GPT_problem_statement_l2410_241099

noncomputable def α : ℝ := 3 + Real.sqrt 8
noncomputable def β : ℝ := 3 - Real.sqrt 8
noncomputable def x : ℝ := α ^ 500
noncomputable def n : ℤ := Int.floor x
noncomputable def f : ℝ := x - n

theorem problem_statement : x * (1 - f) = 1 := 
by
  sorry

end NUMINAMATH_GPT_problem_statement_l2410_241099


namespace NUMINAMATH_GPT_fraction_decimal_representation_l2410_241060

noncomputable def fraction_as_term_dec : ℚ := 47 / (2^3 * 5^4)

theorem fraction_decimal_representation : fraction_as_term_dec = 0.0094 :=
by
  sorry

end NUMINAMATH_GPT_fraction_decimal_representation_l2410_241060


namespace NUMINAMATH_GPT_flower_combinations_l2410_241072

theorem flower_combinations (t l : ℕ) (h : 4 * t + 3 * l = 60) : 
  ∃ (t_values : Finset ℕ), (∀ x ∈ t_values, 0 ≤ x ∧ x ≤ 15 ∧ x % 3 = 0) ∧
  t_values.card = 6 :=
sorry

end NUMINAMATH_GPT_flower_combinations_l2410_241072


namespace NUMINAMATH_GPT_relationship_between_a_b_c_l2410_241041

noncomputable def a : ℝ := Real.log 3 / Real.log 2
noncomputable def b : ℝ := (1 / 3) ^ 2
noncomputable def c : ℝ := Real.log (1 / 30) / Real.log (1 / 3)

theorem relationship_between_a_b_c : c > a ∧ a > b := by
  sorry

end NUMINAMATH_GPT_relationship_between_a_b_c_l2410_241041


namespace NUMINAMATH_GPT_remainder_of_55_power_55_plus_55_l2410_241062

-- Define the problem statement using Lean

theorem remainder_of_55_power_55_plus_55 :
  (55 ^ 55 + 55) % 56 = 54 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_55_power_55_plus_55_l2410_241062


namespace NUMINAMATH_GPT_carnival_activity_order_l2410_241059

theorem carnival_activity_order :
  let dodgeball := 3 / 8
  let magic_show := 9 / 24
  let petting_zoo := 1 / 3
  let face_painting := 5 / 12
  let ordered_activities := ["Face Painting", "Dodgeball", "Magic Show", "Petting Zoo"]
  (face_painting > dodgeball) ∧ (dodgeball = magic_show) ∧ (magic_show > petting_zoo) →
  ordered_activities = ["Face Painting", "Dodgeball", "Magic Show", "Petting Zoo"] :=
by {
  sorry
}

end NUMINAMATH_GPT_carnival_activity_order_l2410_241059


namespace NUMINAMATH_GPT_geometric_sequence_fourth_term_l2410_241049

theorem geometric_sequence_fourth_term (a r T4 : ℝ)
  (h1 : a = 1024)
  (h2 : a * r^5 = 32)
  (h3 : T4 = a * r^3) :
  T4 = 128 :=
by {
  sorry
}

end NUMINAMATH_GPT_geometric_sequence_fourth_term_l2410_241049


namespace NUMINAMATH_GPT_Felix_can_lift_150_pounds_l2410_241005

theorem Felix_can_lift_150_pounds : ∀ (weightFelix weightBrother : ℝ),
  (weightBrother = 2 * weightFelix) →
  (3 * weightBrother = 600) →
  (Felix_can_lift = 1.5 * weightFelix) →
  Felix_can_lift = 150 :=
by
  intros weightFelix weightBrother h1 h2 h3
  sorry

end NUMINAMATH_GPT_Felix_can_lift_150_pounds_l2410_241005


namespace NUMINAMATH_GPT_bea_has_max_profit_l2410_241019

theorem bea_has_max_profit : 
  let price_bea := 25
  let price_dawn := 28
  let price_carla := 35
  let sold_bea := 10
  let sold_dawn := 8
  let sold_carla := 6
  let cost_bea := 10
  let cost_dawn := 12
  let cost_carla := 15
  let profit_bea := (price_bea * sold_bea) - (cost_bea * sold_bea)
  let profit_dawn := (price_dawn * sold_dawn) - (cost_dawn * sold_dawn)
  let profit_carla := (price_carla * sold_carla) - (cost_carla * sold_carla)
  profit_bea = 150 ∧ profit_dawn = 128 ∧ profit_carla = 120 ∧ ∀ p, p ∈ [profit_bea, profit_dawn, profit_carla] → p ≤ 150 :=
by
  sorry

end NUMINAMATH_GPT_bea_has_max_profit_l2410_241019


namespace NUMINAMATH_GPT_difference_between_twice_smaller_and_larger_is_three_l2410_241093

theorem difference_between_twice_smaller_and_larger_is_three
(S L x : ℕ) 
(h1 : L = 2 * S - x) 
(h2 : S + L = 39)
(h3 : S = 14) : 
2 * S - L = 3 := 
sorry

end NUMINAMATH_GPT_difference_between_twice_smaller_and_larger_is_three_l2410_241093


namespace NUMINAMATH_GPT_stuffed_animal_sales_l2410_241090

theorem stuffed_animal_sales (Q T J : ℕ) 
  (h1 : Q = 100 * T) 
  (h2 : J = T + 15) 
  (h3 : Q = 2000) : 
  Q - J = 1965 := 
by
  sorry

end NUMINAMATH_GPT_stuffed_animal_sales_l2410_241090


namespace NUMINAMATH_GPT_neon_signs_blink_together_l2410_241091

theorem neon_signs_blink_together (a b : ℕ) (ha : a = 9) (hb : b = 15) : Nat.lcm a b = 45 := by
  rw [ha, hb]
  have : Nat.lcm 9 15 = 45 := by sorry
  exact this

end NUMINAMATH_GPT_neon_signs_blink_together_l2410_241091


namespace NUMINAMATH_GPT_plywood_cut_difference_l2410_241087

theorem plywood_cut_difference :
  let original_width := 6
  let original_height := 9
  let total_area := original_width * original_height
  let num_pieces := 6
  let area_per_piece := total_area / num_pieces
  -- Let possible perimeters based on given conditions
  let max_perimeter := 20
  let min_perimeter := 15
  max_perimeter - min_perimeter = 5 :=
by
  sorry

end NUMINAMATH_GPT_plywood_cut_difference_l2410_241087


namespace NUMINAMATH_GPT_radius_of_circle_tangent_to_xaxis_l2410_241076

theorem radius_of_circle_tangent_to_xaxis
  (Ω : Set (ℝ × ℝ)) (Γ : Set (ℝ × ℝ))
  (hΓ : ∀ x y : ℝ, (x, y) ∈ Γ ↔ y^2 = 4 * x)
  (F : ℝ × ℝ) (hF : F = (1, 0))
  (hΩ_tangent : ∃ r : ℝ, ∀ x y : ℝ, (x - 1)^2 + (y - r)^2 = r^2 ∧ (1, 0) ∈ Ω)
  (hΩ_intersect : ∀ x y : ℝ, (x, y) ∈ Ω → (x, y) ∈ Γ → (x, y) = (1, 0)) :
  ∃ r : ℝ, r = 4 * Real.sqrt 3 / 9 :=
sorry

end NUMINAMATH_GPT_radius_of_circle_tangent_to_xaxis_l2410_241076


namespace NUMINAMATH_GPT_john_read_books_in_15_hours_l2410_241039

theorem john_read_books_in_15_hours (hreads_faster_ratio : ℝ) (brother_time : ℝ) (john_read_time : ℝ) : john_read_time = brother_time / hreads_faster_ratio → 3 * john_read_time = 15 :=
by
  intros H
  sorry

end NUMINAMATH_GPT_john_read_books_in_15_hours_l2410_241039


namespace NUMINAMATH_GPT_my_op_example_l2410_241006

def my_op (a b : Int) : Int := a^2 - abs b

theorem my_op_example : my_op (-2) (-1) = 3 := by
  sorry

end NUMINAMATH_GPT_my_op_example_l2410_241006


namespace NUMINAMATH_GPT_cos_double_angle_l2410_241048

theorem cos_double_angle (α : ℝ) (h : Real.tan α = -3) : Real.cos (2 * α) = -4 / 5 := sorry

end NUMINAMATH_GPT_cos_double_angle_l2410_241048


namespace NUMINAMATH_GPT_find_difference_l2410_241021

variables (x y : ℝ)

theorem find_difference (h1 : x * (y + 2) = 100) (h2 : y * (x + 2) = 60) : x - y = 20 :=
sorry

end NUMINAMATH_GPT_find_difference_l2410_241021


namespace NUMINAMATH_GPT_rectangle_area_is_180_l2410_241071

def area_of_square (side : ℕ) : ℕ := side * side
def length_of_rectangle (radius : ℕ) : ℕ := (2 * radius) / 5
def area_of_rectangle (length breadth : ℕ) : ℕ := length * breadth

theorem rectangle_area_is_180 :
  ∀ (side breadth : ℕ), 
    area_of_square side = 2025 → 
    breadth = 10 → 
    area_of_rectangle (length_of_rectangle side) breadth = 180 :=
by
  intros side breadth h_area h_breadth
  sorry

end NUMINAMATH_GPT_rectangle_area_is_180_l2410_241071


namespace NUMINAMATH_GPT_total_flowers_sold_l2410_241031

/-
Ginger owns a flower shop, where she sells roses, lilacs, and gardenias.
On Tuesday, she sold three times more roses than lilacs, and half as many gardenias as lilacs.
If she sold 10 lilacs, prove that the total number of flowers sold on Tuesday is 45.
-/

theorem total_flowers_sold
    (lilacs roses gardenias : ℕ)
    (h_lilacs : lilacs = 10)
    (h_roses : roses = 3 * lilacs)
    (h_gardenias : gardenias = lilacs / 2)
    (ht : lilacs + roses + gardenias = 45) :
    lilacs + roses + gardenias = 45 :=
by sorry

end NUMINAMATH_GPT_total_flowers_sold_l2410_241031


namespace NUMINAMATH_GPT_math_problem_l2410_241088

theorem math_problem 
  (a : Int) (b : Int) (c : Int)
  (h_a : a = -1)
  (h_b : b = 1)
  (h_c : c = 0) :
  a + c - b = -2 := 
by
  sorry

end NUMINAMATH_GPT_math_problem_l2410_241088


namespace NUMINAMATH_GPT_solve_for_a_l2410_241074

noncomputable def a := 3.6

theorem solve_for_a (h : 4 * ((a * 0.48 * 2.5) / (0.12 * 0.09 * 0.5)) = 3200.0000000000005) : 
    a = 3.6 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_a_l2410_241074


namespace NUMINAMATH_GPT_find_b_l2410_241056

variable {a b c : ℚ}

theorem find_b (h1 : a + b + c = 117) (h2 : a + 8 = 4 * c) (h3 : b - 10 = 4 * c) : b = 550 / 9 := by
  sorry

end NUMINAMATH_GPT_find_b_l2410_241056


namespace NUMINAMATH_GPT_find_slope_of_l_l2410_241044

noncomputable def parabola (x y : ℝ) := y ^ 2 = 4 * x

-- Definition of the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Definition of the point M
def M : ℝ × ℝ := (-1, 2)

-- Check if two vectors are perpendicular
def is_perpendicular (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 = 0

-- Proof problem statement
theorem find_slope_of_l (x1 x2 y1 y2 k : ℝ)
  (h1 : parabola x1 y1)
  (h2 : parabola x2 y2)
  (h3 : is_perpendicular (x1 + 1, y1 - 2) (x2 + 1, y2 - 2))
  (eq1 : y1 = k * (x1 - 1))
  (eq2 : y2 = k * (x2 - 1)) :
  k = 1 := by
  sorry

end NUMINAMATH_GPT_find_slope_of_l_l2410_241044


namespace NUMINAMATH_GPT_farmer_owned_land_l2410_241042

theorem farmer_owned_land (T : ℝ) (h : 0.10 * T = 720) : 0.80 * T = 5760 :=
by
  sorry

end NUMINAMATH_GPT_farmer_owned_land_l2410_241042


namespace NUMINAMATH_GPT_rectangle_width_l2410_241002

theorem rectangle_width (side_length square_len rect_len : ℝ) (h1 : side_length = 4) (h2 : rect_len = 4) (h3 : square_len = side_length * side_length) (h4 : square_len = rect_len * some_width) :
  some_width = 4 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_width_l2410_241002


namespace NUMINAMATH_GPT_coordinates_of_point_P_l2410_241032

theorem coordinates_of_point_P :
  ∀ (P : ℝ × ℝ), (P.1, P.2) = -1 ∧ (P.2 = -Real.sqrt 3) :=
by
  sorry

end NUMINAMATH_GPT_coordinates_of_point_P_l2410_241032


namespace NUMINAMATH_GPT_point_M_quadrant_l2410_241085

theorem point_M_quadrant (θ : ℝ) (h1 : π / 2 < θ) (h2 : θ < π) :
  (0 < Real.sin θ) ∧ (Real.cos θ < 0) :=
by
  sorry

end NUMINAMATH_GPT_point_M_quadrant_l2410_241085


namespace NUMINAMATH_GPT_max_value_of_seq_l2410_241097

theorem max_value_of_seq (S : ℕ → ℤ) (a : ℕ → ℤ) (hS : ∀ n, S n = -n^2 + 6 * n + 7)
  (h_a_def : ∀ n, a n = S n - S (n - 1)) : ∃ max_val, max_val = 12 ∧ ∀ n, a n ≤ max_val :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_seq_l2410_241097


namespace NUMINAMATH_GPT_watermelon_juice_percentage_l2410_241003

theorem watermelon_juice_percentage :
  ∀ (total_ounces orange_juice_percent grape_juice_ounces : ℕ), 
  orange_juice_percent = 25 →
  grape_juice_ounces = 70 →
  total_ounces = 200 →
  ((total_ounces - (orange_juice_percent * total_ounces / 100 + grape_juice_ounces)) / total_ounces) * 100 = 40 :=
by
  intros total_ounces orange_juice_percent grape_juice_ounces h1 h2 h3
  sorry

end NUMINAMATH_GPT_watermelon_juice_percentage_l2410_241003


namespace NUMINAMATH_GPT_maximum_utilization_rate_80_l2410_241051

noncomputable def maximum_utilization_rate (side_length : ℝ) (AF : ℝ) (BF : ℝ) : ℝ :=
  let area_square := side_length * side_length
  let length_rectangle := side_length
  let width_rectangle := AF / 2
  let area_rectangle := length_rectangle * width_rectangle
  (area_rectangle / area_square) * 100

theorem maximum_utilization_rate_80:
  maximum_utilization_rate 4 2 1 = 80 := by
  sorry

end NUMINAMATH_GPT_maximum_utilization_rate_80_l2410_241051


namespace NUMINAMATH_GPT_final_position_is_negative_one_total_revenue_is_118_yuan_l2410_241037

-- Define the distances
def distances : List Int := [9, -3, -6, 4, -8, 6, -3, -6, -4, 10]

-- Define the taxi price per kilometer
def price_per_km : Int := 2

-- Theorem to prove the final position of the taxi relative to Wu Zhong
theorem final_position_is_negative_one : 
  List.sum distances = -1 :=
by 
  sorry -- Proof omitted

-- Theorem to prove the total revenue for the afternoon
theorem total_revenue_is_118_yuan : 
  price_per_km * List.sum (List.map Int.natAbs distances) = 118 :=
by
  sorry -- Proof omitted

end NUMINAMATH_GPT_final_position_is_negative_one_total_revenue_is_118_yuan_l2410_241037


namespace NUMINAMATH_GPT_shortest_paths_ratio_l2410_241016

theorem shortest_paths_ratio (k n : ℕ) (h : k > 0):
  let paths_along_AB := Nat.choose (k * n + n - 1) (n - 1)
  let paths_along_AD := Nat.choose (k * n + n - 1) k * n - 1
  paths_along_AD = k * paths_along_AB :=
by sorry

end NUMINAMATH_GPT_shortest_paths_ratio_l2410_241016


namespace NUMINAMATH_GPT_max_value_f_l2410_241083

noncomputable def f (x : ℝ) : ℝ := (Real.sin x) ^ 2 + Real.sqrt 3 * Real.cos x - 3 / 4

theorem max_value_f :
  ∀ x ∈ Set.Icc (0 : ℝ) (Real.pi / 2), f x ≤ 1 :=
sorry

end NUMINAMATH_GPT_max_value_f_l2410_241083


namespace NUMINAMATH_GPT_total_price_of_books_l2410_241009

theorem total_price_of_books (total_books : ℕ) (math_books : ℕ) (cost_math_book : ℕ) (cost_history_book : ℕ) (remaining_books := total_books - math_books) (total_math_cost := math_books * cost_math_book) (total_history_cost := remaining_books * cost_history_book ) : total_books = 80 → math_books = 27 → cost_math_book = 4 → cost_history_book = 5 → total_math_cost + total_history_cost = 373 :=
by
  intros
  sorry

end NUMINAMATH_GPT_total_price_of_books_l2410_241009


namespace NUMINAMATH_GPT_pages_per_day_l2410_241001

theorem pages_per_day (total_pages : ℕ) (days : ℕ) (h1 : total_pages = 63) (h2 : days = 3) : total_pages / days = 21 :=
by
  sorry

end NUMINAMATH_GPT_pages_per_day_l2410_241001


namespace NUMINAMATH_GPT_simplify_polynomial_l2410_241068

noncomputable def f (r : ℝ) : ℝ := 2 * r^3 + r^2 + 4 * r - 3
noncomputable def g (r : ℝ) : ℝ := r^3 + r^2 + 6 * r - 8

theorem simplify_polynomial (r : ℝ) : f r - g r = r^3 - 2 * r + 5 := by
  sorry

end NUMINAMATH_GPT_simplify_polynomial_l2410_241068


namespace NUMINAMATH_GPT_product_of_values_of_x_l2410_241033

theorem product_of_values_of_x : 
  (∃ x : ℝ, |x^2 - 7| - 3 = -1) → 
  (∀ x1 x2 x3 x4 : ℝ, 
    (|x1^2 - 7| - 3 = -1) ∧
    (|x2^2 - 7| - 3 = -1) ∧
    (|x3^2 - 7| - 3 = -1) ∧
    (|x4^2 - 7| - 3 = -1) 
    → x1 * x2 * x3 * x4 = 45) :=
sorry

end NUMINAMATH_GPT_product_of_values_of_x_l2410_241033


namespace NUMINAMATH_GPT_point_P_on_x_axis_l2410_241050

noncomputable def point_on_x_axis (m : ℝ) : ℝ × ℝ := (4, m + 1)

theorem point_P_on_x_axis (m : ℝ) (h : point_on_x_axis m = (4, 0)) : m = -1 := 
by
  sorry

end NUMINAMATH_GPT_point_P_on_x_axis_l2410_241050


namespace NUMINAMATH_GPT_find_g2_l2410_241024

def odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x
def even_function (g : ℝ → ℝ) := ∀ x : ℝ, g (-x) = g x

theorem find_g2 {f g : ℝ → ℝ}
  (h1 : odd_function f)
  (h2 : even_function g)
  (h3 : ∀ x : ℝ, f x + g x = 2^x) :
  g 2 = 17 / 8 :=
sorry

end NUMINAMATH_GPT_find_g2_l2410_241024


namespace NUMINAMATH_GPT_smallest_value_of_a_l2410_241030

theorem smallest_value_of_a (a b c : ℤ) (h1 : a < b) (h2 : b < c) (h3 : 2 * b = a + c) (h4 : c^2 = a * b) : a = -4 :=
by
  sorry

end NUMINAMATH_GPT_smallest_value_of_a_l2410_241030


namespace NUMINAMATH_GPT_transfers_l2410_241004

variable (x : ℕ)
variable (gA gB gC : ℕ)

noncomputable def girls_in_A := x + 4
noncomputable def girls_in_B := x
noncomputable def girls_in_C := x - 1

variable (trans_A_to_B : ℕ)
variable (trans_B_to_C : ℕ)
variable (trans_C_to_A : ℕ)

axiom C_to_A_girls : trans_C_to_A = 2
axiom equal_girls : gA = x + 1 ∧ gB = x + 1 ∧ gC = x + 1

theorem transfers (hA : gA = girls_in_A - trans_A_to_B + trans_C_to_A)
                  (hB : gB = girls_in_B - trans_B_to_C + trans_A_to_B)
                  (hC : gC = girls_in_C - trans_C_to_A + trans_B_to_C) :
  trans_A_to_B = 5 ∧ trans_B_to_C = 4 :=
by
  sorry

end NUMINAMATH_GPT_transfers_l2410_241004


namespace NUMINAMATH_GPT_min_abs_val_sum_l2410_241081

noncomputable def abs_val_sum_min : ℝ := (4:ℝ)^(1/3)

theorem min_abs_val_sum (a b c : ℝ) (h : |(a - b) * (b - c) * (c - a)| = 1) :
  |a| + |b| + |c| >= abs_val_sum_min :=
sorry

end NUMINAMATH_GPT_min_abs_val_sum_l2410_241081


namespace NUMINAMATH_GPT_total_animal_legs_l2410_241095

theorem total_animal_legs (total_animals : ℕ) (sheep : ℕ) (chickens : ℕ) : 
  total_animals = 20 ∧ sheep = 10 ∧ chickens = 10 ∧ 
  2 * chickens + 4 * sheep = 60 :=
by 
  sorry

end NUMINAMATH_GPT_total_animal_legs_l2410_241095


namespace NUMINAMATH_GPT_chessboard_game_winner_l2410_241012

theorem chessboard_game_winner (m n : ℕ) (initial_position : ℕ × ℕ) :
  (m * n) % 2 = 0 → (∃ A_wins : Prop, A_wins) ∧ 
  (m * n) % 2 = 1 → (∃ B_wins : Prop, B_wins) :=
by
  sorry

end NUMINAMATH_GPT_chessboard_game_winner_l2410_241012


namespace NUMINAMATH_GPT_both_inequalities_equiv_l2410_241063

theorem both_inequalities_equiv (x : ℝ) : (x - 3)/(2 - x) ≥ 0 ↔ (3 - x)/(x - 2) ≥ 0 := by
  sorry

end NUMINAMATH_GPT_both_inequalities_equiv_l2410_241063


namespace NUMINAMATH_GPT_man_son_work_together_l2410_241026

theorem man_son_work_together (man_days : ℝ) (son_days : ℝ) (combined_days : ℝ) :
  man_days = 4 → son_days = 12 → (1 / man_days + 1 / son_days) = 1 / combined_days → combined_days = 3 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  sorry

end NUMINAMATH_GPT_man_son_work_together_l2410_241026


namespace NUMINAMATH_GPT_g_function_expression_l2410_241067

theorem g_function_expression (f g : ℝ → ℝ) (a : ℝ) (h1 : ∀ x : ℝ, f (-x) = -f x) (h2 : ∀ x : ℝ, g (-x) = g x) (h3 : ∀ x : ℝ, f x + g x = x^2 + a * x + 2 * a - 1) (h4 : f 1 = 2) :
  ∀ t : ℝ, g t = t^2 + 4 * t - 1 :=
by
  sorry

end NUMINAMATH_GPT_g_function_expression_l2410_241067


namespace NUMINAMATH_GPT_pentagon_largest_angle_l2410_241070

theorem pentagon_largest_angle
    (P Q : ℝ)
    (hP : P = 55)
    (hQ : Q = 120)
    (R S T : ℝ)
    (hR_eq_S : R = S)
    (hT : T = 2 * R + 20):
    R + S + T + P + Q = 540 → T = 192.5 :=
by
    sorry

end NUMINAMATH_GPT_pentagon_largest_angle_l2410_241070


namespace NUMINAMATH_GPT_max_eggs_l2410_241036

theorem max_eggs (x : ℕ) 
  (h1 : x < 200) 
  (h2 : x % 3 = 2) 
  (h3 : x % 4 = 3) 
  (h4 : x % 5 = 4) : 
  x = 179 := 
by
  sorry

end NUMINAMATH_GPT_max_eggs_l2410_241036


namespace NUMINAMATH_GPT_geometric_progression_product_l2410_241015

theorem geometric_progression_product (a : ℕ → ℝ) (r : ℝ) (a1 : ℝ)
  (h1 : a 3 = a1 * r^2)
  (h2 : a 10 = a1 * r^9)
  (h3 : a1 * r^2 + a1 * r^9 = 3)
  (h4 : a1^2 * r^11 = -5) :
  a 5 * a 8 = -5 :=
by
  sorry

end NUMINAMATH_GPT_geometric_progression_product_l2410_241015


namespace NUMINAMATH_GPT_initial_meals_is_70_l2410_241079

-- Define variables and conditions
variables (A : ℕ)
def initial_meals_for_adults := A

-- Given conditions
def condition_1 := true  -- Group of 55 adults and some children (not directly used in proving A)
def condition_2 := true  -- Either a certain number of adults or 90 children (implicitly used in equation)
def condition_3 := (A - 21) * (90 / A) = 63  -- 21 adults have their meal, remaining food serves 63 children

-- The proof statement
theorem initial_meals_is_70 (h : (A - 21) * (90 / A) = 63) : A = 70 :=
sorry

end NUMINAMATH_GPT_initial_meals_is_70_l2410_241079


namespace NUMINAMATH_GPT_remainder_is_4_over_3_l2410_241029

noncomputable def original_polynomial (z : ℝ) : ℝ := 3 * z ^ 3 - 4 * z ^ 2 - 14 * z + 3
noncomputable def divisor (z : ℝ) : ℝ := 3 * z + 5
noncomputable def quotient (z : ℝ) : ℝ := z ^ 2 - 3 * z + 1 / 3

theorem remainder_is_4_over_3 :
  ∃ r : ℝ, original_polynomial z = divisor z * quotient z + r ∧ r = 4 / 3 :=
sorry

end NUMINAMATH_GPT_remainder_is_4_over_3_l2410_241029


namespace NUMINAMATH_GPT_relationship_between_c_squared_and_ab_l2410_241011

theorem relationship_between_c_squared_and_ab (a b c : ℝ) 
  (h_pos_a : a > 0) 
  (h_pos_b : b > 0) 
  (h_pos_c : c > 0) 
  (h_c : c = (a + b) / 2) : 
  c^2 ≥ a * b := 
sorry

end NUMINAMATH_GPT_relationship_between_c_squared_and_ab_l2410_241011


namespace NUMINAMATH_GPT_meadow_to_campsite_distance_l2410_241022

variable (d1 d2 d_total d_meadow_to_campsite : ℝ)

theorem meadow_to_campsite_distance
  (h1 : d1 = 0.2)
  (h2 : d2 = 0.4)
  (h_total : d_total = 0.7)
  (h_before_meadow : d_before_meadow = d1 + d2)
  (h_distance : d_meadow_to_campsite = d_total - d_before_meadow) :
  d_meadow_to_campsite = 0.1 :=
by 
  sorry

end NUMINAMATH_GPT_meadow_to_campsite_distance_l2410_241022


namespace NUMINAMATH_GPT_cameron_speed_ratio_l2410_241008

variables (C Ch : ℝ)
-- Danielle's speed is three times Cameron's speed
def Danielle_speed := 3 * C
-- Danielle's travel time from Granville to Salisbury is 30 minutes
def Danielle_time := 30
-- Chase's travel time from Granville to Salisbury is 180 minutes
def Chase_time := 180

-- Prove the ratio of Cameron's speed to Chase's speed is 2
theorem cameron_speed_ratio :
  (Danielle_speed C / Ch) = (Chase_time / Danielle_time) → (C / Ch) = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_cameron_speed_ratio_l2410_241008


namespace NUMINAMATH_GPT_lottery_probability_correct_l2410_241061

def number_of_winnerballs_ways : ℕ := Nat.choose 50 6

def probability_megaBall : ℚ := 1 / 30

def probability_winnerBalls : ℚ := 1 / number_of_winnerballs_ways

def combined_probability : ℚ := probability_megaBall * probability_winnerBalls

theorem lottery_probability_correct : combined_probability = 1 / 476721000 := by
  sorry

end NUMINAMATH_GPT_lottery_probability_correct_l2410_241061


namespace NUMINAMATH_GPT_canoe_rental_cost_l2410_241047

theorem canoe_rental_cost :
  ∃ (C : ℕ) (K : ℕ), 
  (15 * K + C * (K + 4) = 288) ∧ 
  (3 * K + 12 = 12 * C) ∧ 
  (C = 14) :=
sorry

end NUMINAMATH_GPT_canoe_rental_cost_l2410_241047
