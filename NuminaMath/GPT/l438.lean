import Mathlib

namespace NUMINAMATH_GPT_not_mysterious_diff_consecutive_odd_l438_43867

/-- A mysterious number is defined as the difference of squares of two consecutive even numbers. --/
def is_mysterious (n : ℕ) : Prop :=
  ∃ k : ℕ, n = (2 * k + 2)^2 - (2 * k)^2

/-- The difference of the squares of two consecutive odd numbers. --/
def diff_squares_consecutive_odd (k : ℤ) : ℤ :=
  (2 * k + 1)^2 - (2 * k - 1)^2

/-- Prove that the difference of squares of two consecutive odd numbers is not a mysterious number. --/
theorem not_mysterious_diff_consecutive_odd (k : ℤ) : ¬ is_mysterious (Int.natAbs (diff_squares_consecutive_odd k)) :=
by
  sorry

end NUMINAMATH_GPT_not_mysterious_diff_consecutive_odd_l438_43867


namespace NUMINAMATH_GPT_age_ratio_l438_43804
open Nat

theorem age_ratio (B_c : ℕ) (h1 : B_c = 42) (h2 : ∀ A_c, A_c = B_c + 12) : (A_c + 10) / (B_c - 10) = 2 :=
by
  sorry

end NUMINAMATH_GPT_age_ratio_l438_43804


namespace NUMINAMATH_GPT_collinear_points_cube_l438_43832

-- Define a function that counts the sets of three collinear points in the described structure.
def count_collinear_points : Nat :=
  -- Placeholders for the points (vertices, edge midpoints, face centers, center of the cube) and the count logic
  -- The calculation logic will be implemented as the proof
  49

theorem collinear_points_cube : count_collinear_points = 49 :=
  sorry

end NUMINAMATH_GPT_collinear_points_cube_l438_43832


namespace NUMINAMATH_GPT_find_alpha_l438_43890

noncomputable def isGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * (a 2 / a 1)

-- Given that {a_n} is a geometric sequence,
-- a_1 and a_8 are roots of the equation
-- x^2 - 2x * sin(alpha) - √3 * sin(alpha) = 0,
-- and (a_1 + a_8)^2 = 2 * a_3 * a_6 + 6,
-- prove that alpha = π / 3.
theorem find_alpha :
  ∃ α : ℝ,
  (∀ (a : ℕ → ℝ), isGeometricSequence a ∧ 
  (∃ (a1 a8 : ℝ), 
    (a1 + a8)^2 = 2 * a 3 * a 6 + 6 ∧
    a1 + a8 = 2 * Real.sin α ∧
    a1 * a8 = - Real.sqrt 3 * Real.sin α)) →
  α = Real.pi / 3 :=
by 
  sorry

end NUMINAMATH_GPT_find_alpha_l438_43890


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l438_43843

theorem necessary_but_not_sufficient (a b : ℝ) : 
 (a > b) ↔ (a-1 > b+1) :=
by {
  sorry
}

end NUMINAMATH_GPT_necessary_but_not_sufficient_l438_43843


namespace NUMINAMATH_GPT_cow_count_16_l438_43842

theorem cow_count_16 (D C : ℕ) 
  (h1 : ∃ (L H : ℕ), L = 2 * D + 4 * C ∧ H = D + C ∧ L = 2 * H + 32) : C = 16 :=
by
  obtain ⟨L, H, ⟨hL, hH, hCond⟩⟩ := h1
  sorry

end NUMINAMATH_GPT_cow_count_16_l438_43842


namespace NUMINAMATH_GPT_min_value_l438_43838

theorem min_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 * x + y = 1) : 
  ∃ xy : ℝ, (xy = 9 ∧ (forall (u v : ℝ), (u > 0) → (v > 0) → 2 * u + v = 1 → (2 / u) + (1 / v) ≥ xy)) :=
by
  use 9
  sorry

end NUMINAMATH_GPT_min_value_l438_43838


namespace NUMINAMATH_GPT_determine_range_of_m_l438_43836

noncomputable def range_m (m : ℝ) (x : ℝ) : Prop :=
  ∀ x, (-x^2 + 7 * x + 8 < 0 → (x < 1 - 2 * |m| ∨ x > 1 + 2 * |m|)) ∧
       (∃ x, -x^2 + 7 * x + 8 ≥ 0)

theorem determine_range_of_m (m : ℝ) :
  (-1 ≤ m ∧ m ≤ 1) ↔
  (∀ x, (-x^2 + 7 * x + 8 < 0 → (x < 1 - 2 * |m| ∨ x > 1 + 2 * |m|)) ∧
         (∃ x, -x^2 + 7 * x + 8 ≥ 0)) :=
by
  sorry

end NUMINAMATH_GPT_determine_range_of_m_l438_43836


namespace NUMINAMATH_GPT_rachel_age_when_emily_half_her_age_l438_43896

theorem rachel_age_when_emily_half_her_age (emily_current_age rachel_current_age : ℕ) 
  (h1 : emily_current_age = 20) 
  (h2 : rachel_current_age = 24) 
  (age_difference : ℕ) 
  (h3 : rachel_current_age - emily_current_age = age_difference) 
  (emily_age_when_half : ℕ) 
  (rachel_age_when_half : ℕ) 
  (h4 : emily_age_when_half = rachel_age_when_half / 2)
  (h5 : rachel_age_when_half = emily_age_when_half + age_difference) :
  rachel_age_when_half = 8 :=
by
  sorry

end NUMINAMATH_GPT_rachel_age_when_emily_half_her_age_l438_43896


namespace NUMINAMATH_GPT_cos_alpha_plus_pi_six_l438_43866

theorem cos_alpha_plus_pi_six (α : ℝ) (hα_in_interval : 0 < α ∧ α < π / 2) (h_cos : Real.cos α = Real.sqrt 3 / 3) :
  Real.cos (α + π / 6) = (3 - Real.sqrt 6) / 6 := 
by
  sorry

end NUMINAMATH_GPT_cos_alpha_plus_pi_six_l438_43866


namespace NUMINAMATH_GPT_transformed_stats_l438_43880

noncomputable def mean (l : List ℝ) : ℝ :=
  l.sum / l.length

noncomputable def std_dev (l : List ℝ) : ℝ :=
  Real.sqrt ((l.map (λ x => (x - mean l)^2)).sum / l.length)

theorem transformed_stats (l : List ℝ) 
  (hmean : mean l = 10)
  (hstddev : std_dev l = 2) :
  mean (l.map (λ x => 2 * x - 1)) = 19 ∧ std_dev (l.map (λ x => 2 * x - 1)) = 4 := by
  sorry

end NUMINAMATH_GPT_transformed_stats_l438_43880


namespace NUMINAMATH_GPT_infinite_series_sum_eq_two_l438_43889

theorem infinite_series_sum_eq_two : 
  ∑' k : ℕ, (if k = 0 then 0 else (8^k / ((4^k - 3^k) * (4^(k + 1) - 3^(k + 1))))) = 2 :=
by
  sorry

end NUMINAMATH_GPT_infinite_series_sum_eq_two_l438_43889


namespace NUMINAMATH_GPT_sum_of_center_coordinates_l438_43854

def center_of_circle_sum (x y : ℝ) : Prop :=
  (x - 6)^2 + (y + 5)^2 = 101

theorem sum_of_center_coordinates : center_of_circle_sum x y → x + y = 1 :=
sorry

end NUMINAMATH_GPT_sum_of_center_coordinates_l438_43854


namespace NUMINAMATH_GPT_truck_distance_in_3_hours_l438_43822

theorem truck_distance_in_3_hours : 
  ∀ (speed_2miles_2_5minutes : ℝ) 
    (time_minutes : ℝ),
    (speed_2miles_2_5minutes = 2 / 2.5) →
    (time_minutes = 180) →
    (speed_2miles_2_5minutes * time_minutes = 144) :=
by
  intros
  sorry

end NUMINAMATH_GPT_truck_distance_in_3_hours_l438_43822


namespace NUMINAMATH_GPT_value_of_M_after_subtracting_10_percent_l438_43844

-- Define the given conditions and desired result formally in Lean 4
theorem value_of_M_after_subtracting_10_percent (M : ℝ) (h : 0.25 * M = 0.55 * 2500) :
  M - 0.10 * M = 4950 :=
by
  sorry

end NUMINAMATH_GPT_value_of_M_after_subtracting_10_percent_l438_43844


namespace NUMINAMATH_GPT_price_of_each_tomato_l438_43899

theorem price_of_each_tomato
  (customers_per_month : ℕ)
  (lettuce_per_customer : ℕ)
  (lettuce_price : ℕ)
  (tomatoes_per_customer : ℕ)
  (total_monthly_sales : ℕ)
  (total_lettuce_sales : ℕ)
  (total_tomato_sales : ℕ)
  (price_per_tomato : ℝ)
  (h1 : customers_per_month = 500)
  (h2 : lettuce_per_customer = 2)
  (h3 : lettuce_price = 1)
  (h4 : tomatoes_per_customer = 4)
  (h5 : total_monthly_sales = 2000)
  (h6 : total_lettuce_sales = customers_per_month * lettuce_per_customer * lettuce_price)
  (h7 : total_tomato_sales = total_monthly_sales - total_lettuce_sales)
  (h8 : total_lettuce_sales = 1000)
  (h9 : total_tomato_sales = 1000)
  (total_tomatoes_sold : ℕ := customers_per_month * tomatoes_per_customer)
  (h10 : total_tomatoes_sold = 2000) :
  price_per_tomato = 0.50 :=
by
  sorry

end NUMINAMATH_GPT_price_of_each_tomato_l438_43899


namespace NUMINAMATH_GPT_solve_equation_l438_43805

theorem solve_equation (x : ℝ) (h : x ≠ 1) : 
  1 / (x - 1) + 1 = 3 / (2 * x - 2) ↔ x = 3 / 2 := by
  sorry

end NUMINAMATH_GPT_solve_equation_l438_43805


namespace NUMINAMATH_GPT_distribution_y_value_l438_43821

theorem distribution_y_value :
  ∀ (x y : ℝ),
  (x + 0.1 + 0.3 + y = 1) →
  (7 * x + 8 * 0.1 + 9 * 0.3 + 10 * y = 8.9) →
  y = 0.4 :=
by
  intros x y h1 h2
  sorry

end NUMINAMATH_GPT_distribution_y_value_l438_43821


namespace NUMINAMATH_GPT_mean_properties_l438_43864

theorem mean_properties (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0)
  (arith_mean : (x + y + z) / 3 = 10)
  (geom_mean : (x * y * z) ^ (1 / 3) = 6)
  (harm_mean : 3 / (1/x + 1/y + 1/z) = 2.5) :
  x^2 + y^2 + z^2 = 540 := 
sorry

end NUMINAMATH_GPT_mean_properties_l438_43864


namespace NUMINAMATH_GPT_min_value_x2_y2_z2_l438_43828

theorem min_value_x2_y2_z2 (x y z : ℝ) (h : 2 * x + 3 * y + 4 * z = 11) : 
  x^2 + y^2 + z^2 ≥ 121 / 29 :=
sorry

end NUMINAMATH_GPT_min_value_x2_y2_z2_l438_43828


namespace NUMINAMATH_GPT_arithmetic_sequence_general_term_and_sum_l438_43892

theorem arithmetic_sequence_general_term_and_sum :
  (∃ (a₁ d : ℤ), a₁ + d = 14 ∧ a₁ + 4 * d = 5 ∧ ∀ n : ℤ, a_n = a₁ + (n - 1) * d ∧ (∀ N : ℤ, N ≥ 1 → S_N = N * ((2 * a₁ + (N - 1) * d) / 2) ∧ N = 10 → S_N = 35)) :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_general_term_and_sum_l438_43892


namespace NUMINAMATH_GPT_TeamC_fee_l438_43862

structure Team :=
(work_rate : ℚ)

def teamA : Team := ⟨1 / 36⟩
def teamB : Team := ⟨1 / 24⟩
def teamC : Team := ⟨1 / 18⟩

def total_fee : ℚ := 36000

def combined_work_rate_first_half (A B C : Team) : ℚ :=
(A.work_rate + B.work_rate + C.work_rate) * 1 / 2

def combined_work_rate_second_half (A C : Team) : ℚ :=
(A.work_rate + C.work_rate) * 1 / 2

def total_work_completed_by_TeamC (A B C : Team) : ℚ :=
C.work_rate * combined_work_rate_first_half A B C + C.work_rate * combined_work_rate_second_half A C

theorem TeamC_fee (A B C : Team) (total_fee : ℚ) :
  total_work_completed_by_TeamC A B C * total_fee = 20000 :=
by
  sorry

end NUMINAMATH_GPT_TeamC_fee_l438_43862


namespace NUMINAMATH_GPT_negation_of_forall_ge_implies_exists_lt_l438_43823

theorem negation_of_forall_ge_implies_exists_lt :
  ¬(∀ x : ℝ, x^2 + 1 ≥ 2 * x) ↔ ∃ x : ℝ, x^2 + 1 < 2 * x := by
  sorry

end NUMINAMATH_GPT_negation_of_forall_ge_implies_exists_lt_l438_43823


namespace NUMINAMATH_GPT_box_volume_l438_43865

theorem box_volume
  (l w h : ℝ)
  (h1 : l * w = 30)
  (h2 : w * h = 20)
  (h3 : l * h = 12)
  (h4 : l = h + 1) :
  l * w * h = 120 := 
sorry

end NUMINAMATH_GPT_box_volume_l438_43865


namespace NUMINAMATH_GPT_residue_calculation_l438_43839

theorem residue_calculation 
  (h1 : 182 ≡ 0 [MOD 14])
  (h2 : 182 * 12 ≡ 0 [MOD 14])
  (h3 : 15 * 7 ≡ 7 [MOD 14])
  (h4 : 3 ≡ 3 [MOD 14]) :
  (182 * 12 - 15 * 7 + 3) ≡ 10 [MOD 14] :=
sorry

end NUMINAMATH_GPT_residue_calculation_l438_43839


namespace NUMINAMATH_GPT_solve_for_z_l438_43813

theorem solve_for_z {x y z : ℝ} (h : (1 / x^2) - (1 / y^2) = 1 / z) :
  z = (x^2 * y^2) / (y^2 - x^2) :=
sorry

end NUMINAMATH_GPT_solve_for_z_l438_43813


namespace NUMINAMATH_GPT_ac_bd_leq_8_l438_43834

theorem ac_bd_leq_8 (a b c d : ℝ) (h1 : a^2 + b^2 = 4) (h2 : c^2 + d^2 = 16) : ac + bd ≤ 8 :=
sorry

end NUMINAMATH_GPT_ac_bd_leq_8_l438_43834


namespace NUMINAMATH_GPT_linear_function_quadrants_l438_43884

theorem linear_function_quadrants : 
  ∀ (x y : ℝ), y = -5 * x + 3 
  → ((x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0) ∨ (x > 0 ∧ y < 0)) :=
by 
  intro x y h
  sorry

end NUMINAMATH_GPT_linear_function_quadrants_l438_43884


namespace NUMINAMATH_GPT_probability_sum_even_is_five_over_eleven_l438_43870

noncomputable def probability_even_sum : ℚ :=
  let totalBalls := 12
  let totalWays := totalBalls * (totalBalls - 1)
  let evenBalls := 6
  let oddBalls := 6
  let evenWays := evenBalls * (evenBalls - 1)
  let oddWays := oddBalls * (oddBalls - 1)
  let totalEvenWays := evenWays + oddWays
  totalEvenWays / totalWays

theorem probability_sum_even_is_five_over_eleven : probability_even_sum = 5 / 11 := sorry

end NUMINAMATH_GPT_probability_sum_even_is_five_over_eleven_l438_43870


namespace NUMINAMATH_GPT_range_of_a_l438_43809

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, (a+2)/(x+1) = 1 ∧ x ≤ 0) ↔ (a ≤ -1 ∧ a ≠ -2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l438_43809


namespace NUMINAMATH_GPT_value_of_x0_l438_43888

noncomputable def f (x : ℝ) : ℝ := x^3

theorem value_of_x0 (x0 : ℝ) (h1 : f x0 = x0^3) (h2 : deriv f x0 = 3) :
  x0 = 1 ∨ x0 = -1 :=
by
  sorry

end NUMINAMATH_GPT_value_of_x0_l438_43888


namespace NUMINAMATH_GPT_fraction_q_over_p_l438_43883

noncomputable def proof_problem (p q : ℝ) : Prop :=
  ∃ k : ℝ, p = 9^k ∧ q = 12^k ∧ p + q = 16^k

theorem fraction_q_over_p (p q : ℝ) (h : proof_problem p q) : q / p = (1 + Real.sqrt 5) / 2 :=
sorry

end NUMINAMATH_GPT_fraction_q_over_p_l438_43883


namespace NUMINAMATH_GPT_four_cubic_feet_to_cubic_inches_l438_43895

theorem four_cubic_feet_to_cubic_inches (h : 1 = 12) : 4 * (12^3) = 6912 :=
by
  sorry

end NUMINAMATH_GPT_four_cubic_feet_to_cubic_inches_l438_43895


namespace NUMINAMATH_GPT_p_is_sufficient_but_not_necessary_for_q_l438_43808

variable (x : ℝ)

def p := x > 1
def q := x > 0

theorem p_is_sufficient_but_not_necessary_for_q : (p x → q x) ∧ ¬(q x → p x) := by
  sorry

end NUMINAMATH_GPT_p_is_sufficient_but_not_necessary_for_q_l438_43808


namespace NUMINAMATH_GPT_find_directrix_l438_43869

-- Define the parabola equation
def parabola_eq (x y : ℝ) : Prop := x^2 = 8 * y

-- State the problem to find the directrix of the given parabola
theorem find_directrix (x y : ℝ) (h : parabola_eq x y) : y = -2 :=
sorry

end NUMINAMATH_GPT_find_directrix_l438_43869


namespace NUMINAMATH_GPT_value_of_a_l438_43863

theorem value_of_a
  (a b : ℚ)
  (h1 : b / a = 4)
  (h2 : b = 18 - 6 * a) :
  a = 9 / 5 := by
  sorry

end NUMINAMATH_GPT_value_of_a_l438_43863


namespace NUMINAMATH_GPT_shirt_price_l438_43891

theorem shirt_price (T S : ℝ) (h1 : T + S = 80.34) (h2 : T = S - 7.43) : T = 36.455 :=
by 
sorry

end NUMINAMATH_GPT_shirt_price_l438_43891


namespace NUMINAMATH_GPT_student_score_l438_43835

theorem student_score (c w : ℕ) (h1 : c + w = 60) (h2 : 4 * c - w = 150) : c = 42 :=
by
-- Proof steps here, we skip by using sorry for now
sorry

end NUMINAMATH_GPT_student_score_l438_43835


namespace NUMINAMATH_GPT_no_uniformly_colored_rectangle_l438_43874

open Int

def point := (ℤ × ℤ)

def is_green (P : point) : Prop :=
  3 ∣ (P.1 + P.2)

def is_red (P : point) : Prop :=
  ¬ is_green P

def is_rectangle (A B C D : point) : Prop :=
  A.1 = D.1 ∧ B.1 = C.1 ∧ A.2 = B.2 ∧ D.2 = C.2

def rectangle_area (A B : point) : ℤ :=
  abs (B.1 - A.1) * abs (B.2 - A.2)

theorem no_uniformly_colored_rectangle :
  ∀ (A B C D : point) (k : ℕ), 
  is_rectangle A B C D →
  rectangle_area A C = 2^k →
  ¬ (is_green A ∧ is_green B ∧ is_green C ∧ is_green D) ∧
  ¬ (is_red A ∧ is_red B ∧ is_red C ∧ is_red D) :=
by sorry

end NUMINAMATH_GPT_no_uniformly_colored_rectangle_l438_43874


namespace NUMINAMATH_GPT_determine_k_l438_43837

def f(x : ℝ) : ℝ := 5 * x^2 - 3 * x + 8
def g(x k : ℝ) : ℝ := x^3 - k * x - 10

theorem determine_k : 
  (f (-5) - g (-5) k = -24) → k = 61 := 
by 
-- Begin the proof script here
sorry

end NUMINAMATH_GPT_determine_k_l438_43837


namespace NUMINAMATH_GPT_max_g_value_l438_43887

def g (n : ℕ) : ℕ :=
if h : n < 10 then 2 * n + 3 else g (n - 7)

theorem max_g_value : ∃ n, g n = 21 ∧ ∀ m, g m ≤ 21 :=
sorry

end NUMINAMATH_GPT_max_g_value_l438_43887


namespace NUMINAMATH_GPT_weight_of_b_l438_43841

theorem weight_of_b (A B C : ℝ)
  (h1 : A + B + C = 129)
  (h2 : A + B = 96)
  (h3 : B + C = 84) : B = 51 := 
by
  sorry

end NUMINAMATH_GPT_weight_of_b_l438_43841


namespace NUMINAMATH_GPT_proportion_of_boys_geq_35_percent_l438_43825

variables (a b c d n : ℕ)

axiom room_constraint : 2 * (b + d) ≥ n
axiom girl_constraint : 3 * a ≥ 8 * b

theorem proportion_of_boys_geq_35_percent : (3 * c + 4 * d : ℚ) / (3 * a + 4 * b + 3 * c + 4 * d : ℚ) ≥ 0.35 :=
by 
  sorry

end NUMINAMATH_GPT_proportion_of_boys_geq_35_percent_l438_43825


namespace NUMINAMATH_GPT_average_page_count_per_essay_l438_43824

-- Conditions
def numberOfStudents := 15
def pagesFirstFive := 5 * 2
def pagesNextFive := 5 * 3
def pagesLastFive := 5 * 1

-- Total pages
def totalPages := pagesFirstFive + pagesNextFive + pagesLastFive

-- Proof problem statement
theorem average_page_count_per_essay : totalPages / numberOfStudents = 2 := by
  sorry

end NUMINAMATH_GPT_average_page_count_per_essay_l438_43824


namespace NUMINAMATH_GPT_af_b_lt_bf_a_l438_43858

variable {f : ℝ → ℝ}
variable {a b : ℝ}

theorem af_b_lt_bf_a (h1 : ∀ x y, 0 < x → 0 < y → x < y → f x > f y)
                    (h2 : ∀ x, 0 < x → f x > 0)
                    (h3 : 0 < a)
                    (h4 : 0 < b)
                    (h5 : a < b) :
  a * f b < b * f a :=
sorry

end NUMINAMATH_GPT_af_b_lt_bf_a_l438_43858


namespace NUMINAMATH_GPT_min_value_reciprocal_sum_l438_43848

theorem min_value_reciprocal_sum (x y : ℝ) (hx_pos : x > 0) (hy_pos : y > 0) (h_sum : x + y = 1) : 
  ∃ z, z = 4 ∧ (∀ x y, x > 0 ∧ y > 0 ∧ x + y = 1 -> z ≤ (1/x + 1/y)) :=
sorry

end NUMINAMATH_GPT_min_value_reciprocal_sum_l438_43848


namespace NUMINAMATH_GPT_sin_sum_given_cos_tan_conditions_l438_43876

open Real

theorem sin_sum_given_cos_tan_conditions 
  (α β : ℝ)
  (h1 : cos α + cos β = 1 / 3)
  (h2 : tan (α + β) = 24 / 7)
  : sin α + sin β = 1 / 4 ∨ sin α + sin β = -4 / 9 := 
  sorry

end NUMINAMATH_GPT_sin_sum_given_cos_tan_conditions_l438_43876


namespace NUMINAMATH_GPT_servant_cash_received_l438_43801

theorem servant_cash_received (salary_cash : ℕ) (turban_value : ℕ) (months_worked : ℕ) (total_months : ℕ)
  (h_salary_cash : salary_cash = 90) (h_turban_value : turban_value = 70) (h_months_worked : months_worked = 9)
  (h_total_months : total_months = 12) : 
  salary_cash * months_worked / total_months + (turban_value * months_worked / total_months) - turban_value = 50 := by
sorry

end NUMINAMATH_GPT_servant_cash_received_l438_43801


namespace NUMINAMATH_GPT_pizza_toppings_l438_43846

theorem pizza_toppings (toppings : Finset String) (h : toppings.card = 8) :
  (toppings.card.choose 1 + toppings.card.choose 2 + toppings.card.choose 3) = 92 := by
  have ht : toppings.card = 8 := h
  sorry

end NUMINAMATH_GPT_pizza_toppings_l438_43846


namespace NUMINAMATH_GPT_time_to_see_slow_train_l438_43833

noncomputable def time_to_pass (length_fast_train length_slow_train relative_time_fast seconds_observed_by_slow : ℕ) : ℕ := 
  length_slow_train * seconds_observed_by_slow / length_fast_train

theorem time_to_see_slow_train :
  let length_fast_train := 150
  let length_slow_train := 200
  let seconds_observed_by_slow := 6
  let expected_time := 8
  time_to_pass length_fast_train length_slow_train length_fast_train seconds_observed_by_slow = expected_time :=
by sorry

end NUMINAMATH_GPT_time_to_see_slow_train_l438_43833


namespace NUMINAMATH_GPT_percentage_william_land_l438_43826

-- Definitions of the given conditions
def total_tax_collected : ℝ := 3840
def william_tax : ℝ := 480

-- Proof statement
theorem percentage_william_land :
  ((william_tax / total_tax_collected) * 100) = 12.5 :=
by
  sorry

end NUMINAMATH_GPT_percentage_william_land_l438_43826


namespace NUMINAMATH_GPT_min_value_is_neg_500000_l438_43827

noncomputable def min_expression_value (a b : ℝ) : ℝ :=
  let term1 := a + 1/b
  let term2 := b + 1/a
  (term1 * (term1 - 1000) + term2 * (term2 - 1000))

theorem min_value_is_neg_500000 (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  min_expression_value a b ≥ -500000 :=
sorry

end NUMINAMATH_GPT_min_value_is_neg_500000_l438_43827


namespace NUMINAMATH_GPT_general_formula_for_sequence_l438_43861

theorem general_formula_for_sequence :
  ∀ (a : ℕ → ℕ), (a 0 = 1) → (a 1 = 1) →
  (∀ n, 2 ≤ n → a n = 2 * a (n - 1) - a (n - 2)) →
  ∀ n, a n = (2^n - 1)^2 :=
by
  sorry

end NUMINAMATH_GPT_general_formula_for_sequence_l438_43861


namespace NUMINAMATH_GPT_abs_sum_less_than_two_l438_43898

theorem abs_sum_less_than_two (a b : ℝ) (ha : |a| < 1) (hb : |b| < 1) : |a + b| + |a - b| < 2 := 
sorry

end NUMINAMATH_GPT_abs_sum_less_than_two_l438_43898


namespace NUMINAMATH_GPT_smallest_possible_x_l438_43871

/-- Proof problem: When x is divided by 6, 7, and 8, remainders of 5, 6, and 7 (respectively) are obtained. 
We need to show that the smallest possible positive integer value of x is 167. -/
theorem smallest_possible_x (x : ℕ) (h1 : x % 6 = 5) (h2 : x % 7 = 6) (h3 : x % 8 = 7) : x = 167 :=
by 
  sorry

end NUMINAMATH_GPT_smallest_possible_x_l438_43871


namespace NUMINAMATH_GPT_max_a4_l438_43807

variable {a_n : ℕ → ℝ}

-- Assume a_n is a positive geometric sequence
def is_geometric_seq (a_n : ℕ → ℝ) : Prop :=
  ∃ r > 0, ∀ n, a_n (n + 1) = a_n n * r

-- Given conditions
def condition1 (a_n : ℕ → ℝ) : Prop := is_geometric_seq a_n
def condition2 (a_n : ℕ → ℝ) : Prop := a_n 3 + a_n 5 = 4

theorem max_a4 (a_n : ℕ → ℝ) (h1 : condition1 a_n) (h2 : condition2 a_n) :
    ∃ max_a4 : ℝ, max_a4 = 2 :=
  sorry

end NUMINAMATH_GPT_max_a4_l438_43807


namespace NUMINAMATH_GPT_largest_base5_three_digits_is_124_l438_43806

noncomputable def largest_base5_three_digits_to_base10 : ℕ :=
  4 * 5^2 + 4 * 5^1 + 4 * 5^0

theorem largest_base5_three_digits_is_124 :
  largest_base5_three_digits_to_base10 = 124 :=
by
  -- calculating 4 * 5^2 + 4 * 5^1 + 4 * 5^0 = 124
  sorry

end NUMINAMATH_GPT_largest_base5_three_digits_is_124_l438_43806


namespace NUMINAMATH_GPT_garden_area_l438_43860

variable (L W A : ℕ)
variable (H1 : 3000 = 50 * L)
variable (H2 : 3000 = 15 * (2*L + 2*W))

theorem garden_area : A = 2400 :=
by
  sorry

end NUMINAMATH_GPT_garden_area_l438_43860


namespace NUMINAMATH_GPT_simplify_fraction_l438_43877

theorem simplify_fraction : 
  (1:ℚ) / 462 + 17 / 42 = 94 / 231 := 
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l438_43877


namespace NUMINAMATH_GPT_square_area_increase_l438_43852

variable (a : ℕ)

theorem square_area_increase (a : ℕ) :
  (a + 6) ^ 2 - a ^ 2 = 12 * a + 36 :=
by
  sorry

end NUMINAMATH_GPT_square_area_increase_l438_43852


namespace NUMINAMATH_GPT_non_zero_real_x_solution_l438_43885

theorem non_zero_real_x_solution (x : ℝ) (hx : x ≠ 0) : (9 * x) ^ 18 = (18 * x) ^ 9 → x = 2 / 9 := by
  sorry

end NUMINAMATH_GPT_non_zero_real_x_solution_l438_43885


namespace NUMINAMATH_GPT_calculate_cakes_left_l438_43814

-- Define the conditions
def b_lunch : ℕ := 5
def s_dinner : ℕ := 6
def b_yesterday : ℕ := 3

-- Define the calculation of the total cakes baked and cakes left
def total_baked : ℕ := b_lunch + b_yesterday
def cakes_left : ℕ := total_baked - s_dinner

-- The theorem we want to prove
theorem calculate_cakes_left : cakes_left = 2 := 
by
  sorry

end NUMINAMATH_GPT_calculate_cakes_left_l438_43814


namespace NUMINAMATH_GPT_intersection_P_Q_eq_Q_l438_43882

def P : Set ℝ := { x | x < 2 }
def Q : Set ℝ := { x | x^2 ≤ 1 }

theorem intersection_P_Q_eq_Q : P ∩ Q = Q := 
sorry

end NUMINAMATH_GPT_intersection_P_Q_eq_Q_l438_43882


namespace NUMINAMATH_GPT_common_difference_of_arithmetic_sequence_l438_43803

theorem common_difference_of_arithmetic_sequence
  (a : ℕ → ℝ) (d : ℝ)
  (h1 : a 2 = 9)
  (h2 : a 5 = 33)
  (h_arith_seq : ∀ n : ℕ, a (n + 1) = a n + d) :
  d = 8 :=
sorry

end NUMINAMATH_GPT_common_difference_of_arithmetic_sequence_l438_43803


namespace NUMINAMATH_GPT_log_sin_decrease_interval_l438_43819

open Real

noncomputable def interval_of_decrease (x : ℝ) : Prop :=
  ∃ (k : ℤ), (k * π + π / 8 < x ∧ x ≤ k * π + 3 * π / 8)

theorem log_sin_decrease_interval (x : ℝ) :
  interval_of_decrease x ↔ ∃ (k : ℤ), (k * π + π / 8 < x ∧ x ≤ k * π + 3 * π / 8) :=
by
  sorry

end NUMINAMATH_GPT_log_sin_decrease_interval_l438_43819


namespace NUMINAMATH_GPT_Aiyanna_cookies_l438_43893

-- Define the conditions
def Alyssa_cookies : ℕ := 129
variable (x : ℕ)
def difference_condition : Prop := (Alyssa_cookies - x) = 11

-- The theorem to prove
theorem Aiyanna_cookies (x : ℕ) (h : difference_condition x) : x = 118 :=
by sorry

end NUMINAMATH_GPT_Aiyanna_cookies_l438_43893


namespace NUMINAMATH_GPT_ellipse_range_of_k_l438_43817

theorem ellipse_range_of_k (k : ℝ) :
  (∃ (eq : ((x y : ℝ) → (x ^ 2 / (3 + k) + y ^ 2 / (2 - k) = 1))),
  ((3 + k > 0) ∧ (2 - k > 0) ∧ (3 + k ≠ 2 - k))) ↔
  (k ∈ Set.Ioo (-3 : ℝ) ((-1) / 2) ∪ Set.Ioo ((-1) / 2) 2) :=
by sorry

end NUMINAMATH_GPT_ellipse_range_of_k_l438_43817


namespace NUMINAMATH_GPT_find_x_l438_43868

theorem find_x (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 80) : x = 26 :=
by 
  sorry

end NUMINAMATH_GPT_find_x_l438_43868


namespace NUMINAMATH_GPT_binom_difference_30_3_2_l438_43815

-- Define the binomial coefficient function.
def binom (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Theorem statement: binom(30, 3) - binom(30, 2) = 3625
theorem binom_difference_30_3_2 : binom 30 3 - binom 30 2 = 3625 := by
  sorry

end NUMINAMATH_GPT_binom_difference_30_3_2_l438_43815


namespace NUMINAMATH_GPT_sequence_non_zero_l438_43853

theorem sequence_non_zero :
  ∀ n : ℕ, ∃ a : ℕ → ℤ,
  (a 1 = 1) ∧
  (a 2 = 2) ∧
  (∀ n : ℕ, (a (n+1) % 2 = 1 ∧ a n % 2 = 1) → (a (n+2) = 5 * a (n+1) - 3 * a n)) ∧
  (∀ n : ℕ, (a (n+1) % 2 = 0 ∧ a n % 2 = 0) → (a (n+2) = a (n+1) - a n)) ∧
  (a n ≠ 0) :=
by
  sorry

end NUMINAMATH_GPT_sequence_non_zero_l438_43853


namespace NUMINAMATH_GPT_hancho_tape_length_l438_43802

noncomputable def tape_length (x : ℝ) : Prop :=
  (1 / 4) * (4 / 5) * x = 1.5

theorem hancho_tape_length : ∃ x : ℝ, tape_length x ∧ x = 7.5 :=
by sorry

end NUMINAMATH_GPT_hancho_tape_length_l438_43802


namespace NUMINAMATH_GPT_square_of_ratio_is_specified_value_l438_43840

theorem square_of_ratio_is_specified_value (a b c : ℝ) (h1 : c = Real.sqrt (a^2 + b^2)) (h2 : a / b = b / c) :
  (a / b)^2 = (-1 + Real.sqrt 5) / 2 :=
by
  sorry

end NUMINAMATH_GPT_square_of_ratio_is_specified_value_l438_43840


namespace NUMINAMATH_GPT_acai_juice_cost_l438_43829

noncomputable def cost_per_litre_juice (x : ℝ) : Prop :=
  let total_cost_cocktail := 1399.45 * 53.333333333333332
  let cost_mixed_fruit_juice := 32 * 262.85
  let cost_acai_juice := 21.333333333333332 * x
  total_cost_cocktail = cost_mixed_fruit_juice + cost_acai_juice

/-- The cost per litre of the açaí berry juice is $3105.00 given the specified conditions. -/
theorem acai_juice_cost : cost_per_litre_juice 3105.00 :=
  sorry

end NUMINAMATH_GPT_acai_juice_cost_l438_43829


namespace NUMINAMATH_GPT_correct_reference_l438_43859

variable (house : String) 
variable (beautiful_garden_in_front : Bool)
variable (I_like_this_house : Bool)
variable (enough_money_to_buy : Bool)

-- Statement: Given the conditions, prove that the correct word to fill in the blank is "it".
theorem correct_reference : I_like_this_house ∧ beautiful_garden_in_front ∧ ¬ enough_money_to_buy → "it" = "correct choice" :=
by
  sorry

end NUMINAMATH_GPT_correct_reference_l438_43859


namespace NUMINAMATH_GPT_apples_harvested_from_garden_l438_43818

def number_of_pies : ℕ := 10
def apples_per_pie : ℕ := 8
def apples_to_buy : ℕ := 30

def total_apples_needed : ℕ := number_of_pies * apples_per_pie

theorem apples_harvested_from_garden : total_apples_needed - apples_to_buy = 50 :=
by
  sorry

end NUMINAMATH_GPT_apples_harvested_from_garden_l438_43818


namespace NUMINAMATH_GPT_minimum_value_expression_l438_43857

theorem minimum_value_expression (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  4 * a ^ 3 + 8 * b ^ 3 + 27 * c ^ 3 + 64 * d ^ 3 + 2 / (a * b * c * d) ≥ 16 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_expression_l438_43857


namespace NUMINAMATH_GPT_sum_nonpositive_inequality_l438_43881

theorem sum_nonpositive_inequality (x : ℝ) : x + 5 ≤ 0 ↔ x + 5 ≤ 0 :=
by
  sorry

end NUMINAMATH_GPT_sum_nonpositive_inequality_l438_43881


namespace NUMINAMATH_GPT_find_y_l438_43851

theorem find_y (y : ℝ) (h : (y^2 - 11 * y + 24) / (y - 1) + (4 * y^2 + 20 * y - 25) / (4*y - 5) = 5) :
  y = 3 ∨ y = 4 :=
sorry

end NUMINAMATH_GPT_find_y_l438_43851


namespace NUMINAMATH_GPT_area_of_figure_eq_two_l438_43810

theorem area_of_figure_eq_two :
  ∫ x in (1 / Real.exp 1)..(Real.exp 1), 1 / x = 2 :=
by sorry

end NUMINAMATH_GPT_area_of_figure_eq_two_l438_43810


namespace NUMINAMATH_GPT_find_number_l438_43878

theorem find_number (x : ℝ) (h : 0.26 * x = 93.6) : x = 360 := sorry

end NUMINAMATH_GPT_find_number_l438_43878


namespace NUMINAMATH_GPT_total_fruits_l438_43875

theorem total_fruits (a b c : ℕ) (h1 : a = 5) (h2 : b = 6) (h3 : c = 4) : a + b + c = 15 := by
  sorry

end NUMINAMATH_GPT_total_fruits_l438_43875


namespace NUMINAMATH_GPT_digits_difference_l438_43897

-- Definitions based on conditions
variables (X Y : ℕ)

-- Condition: The difference between the original number and the interchanged number is 27
def difference_condition : Prop :=
  (10 * X + Y) - (10 * Y + X) = 27

-- Problem to prove: The difference between the two digits is 3
theorem digits_difference (h : difference_condition X Y) : X - Y = 3 :=
by sorry

end NUMINAMATH_GPT_digits_difference_l438_43897


namespace NUMINAMATH_GPT_projection_matrix_ordered_pair_l438_43800

theorem projection_matrix_ordered_pair (a c : ℚ)
  (P : Matrix (Fin 2) (Fin 2) ℚ) 
  (P := ![![a, 15 / 34], ![c, 25 / 34]]) :
  P * P = P ->
  (a, c) = (9 / 34, 15 / 34) :=
by
  sorry

end NUMINAMATH_GPT_projection_matrix_ordered_pair_l438_43800


namespace NUMINAMATH_GPT_locus_area_l438_43856

theorem locus_area (R : ℝ) (r : ℝ) (hR : R = 6 * Real.sqrt 7) (hr : r = Real.sqrt 7) :
    ∃ (L : ℝ), (L = 2 * Real.sqrt 42 ∧ L^2 * Real.pi = 168 * Real.pi) :=
by
  sorry

end NUMINAMATH_GPT_locus_area_l438_43856


namespace NUMINAMATH_GPT_child_ticket_cost_l438_43855

theorem child_ticket_cost 
    (total_people : ℕ) 
    (total_money_collected : ℤ) 
    (adult_ticket_price : ℤ) 
    (children_attended : ℕ) 
    (adults_count : ℕ) 
    (total_adult_cost : ℤ) 
    (total_child_cost : ℤ) 
    (c : ℤ)
    (total_people_eq : total_people = 22)
    (total_money_collected_eq : total_money_collected = 50)
    (adult_ticket_price_eq : adult_ticket_price = 8)
    (children_attended_eq : children_attended = 18)
    (adults_count_eq : adults_count = total_people - children_attended)
    (total_adult_cost_eq : total_adult_cost = adults_count * adult_ticket_price)
    (total_child_cost_eq : total_child_cost = children_attended * c)
    (money_collected_eq : total_money_collected = total_adult_cost + total_child_cost) 
  : c = 1 := 
  by
    sorry

end NUMINAMATH_GPT_child_ticket_cost_l438_43855


namespace NUMINAMATH_GPT_initial_volume_of_mixture_l438_43816

variable (V : ℝ)
variable (H1 : 0.2 * V + 12 = 0.25 * (V + 12))

theorem initial_volume_of_mixture (H : 0.2 * V + 12 = 0.25 * (V + 12)) : V = 180 := by
  sorry

end NUMINAMATH_GPT_initial_volume_of_mixture_l438_43816


namespace NUMINAMATH_GPT_inequality_proof_l438_43845

theorem inequality_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 1) :
  (1 / Real.sqrt (x + y)) + (1 / Real.sqrt (y + z)) + (1 / Real.sqrt (z + x)) ≤ 1 / Real.sqrt (2 * x * y * z) :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l438_43845


namespace NUMINAMATH_GPT_kanul_total_amount_l438_43812

theorem kanul_total_amount (T : ℝ) (h1 : 500 + 400 + 0.10 * T = T) : T = 1000 :=
  sorry

end NUMINAMATH_GPT_kanul_total_amount_l438_43812


namespace NUMINAMATH_GPT_quadratic_equation_problems_l438_43811

noncomputable def quadratic_has_real_roots (m : ℝ) : Prop :=
  let a := m
  let b := -(3 * m - 1)
  let c := 2 * m - 2
  let Δ := b ^ 2 - 4 * a * c
  Δ ≥ 0

noncomputable def valid_m_values (m : ℝ) : Prop :=
  let a := m
  let b := -(3 * m - 1)
  let c := 2 * m - 2
  let Δ := b ^ 2 - 4 * a * c
  1 = m ∨ -1 / 3 = m

theorem quadratic_equation_problems (m : ℝ) :
  quadratic_has_real_roots m ∧
  (∀ x1 x2 : ℝ, 
      (x1 ≠ x2) →
      x1 + x2 = -(3 * m - 1) / m →
      x1 * x2 = (2 * m - 2) / m →
      abs (x1 - x2) = 2 →
      valid_m_values m) :=
by 
  sorry

end NUMINAMATH_GPT_quadratic_equation_problems_l438_43811


namespace NUMINAMATH_GPT_darren_and_fergie_same_amount_in_days_l438_43886

theorem darren_and_fergie_same_amount_in_days : 
  ∀ (t : ℕ), (200 + 16 * t = 300 + 12 * t) → t = 25 := 
by sorry

end NUMINAMATH_GPT_darren_and_fergie_same_amount_in_days_l438_43886


namespace NUMINAMATH_GPT_Yihana_uphill_walking_time_l438_43873

theorem Yihana_uphill_walking_time :
  let t1 := 3
  let t2 := 2
  let t_total := t1 + t2
  t_total = 5 :=
by
  let t1 := 3
  let t2 := 2
  let t_total := t1 + t2
  show t_total = 5
  sorry

end NUMINAMATH_GPT_Yihana_uphill_walking_time_l438_43873


namespace NUMINAMATH_GPT_number_equation_l438_43831

-- Lean statement equivalent to the mathematical problem
theorem number_equation (x : ℝ) (h : 5 * x - 2 * x = 10) : 5 * x - 2 * x = 10 :=
by exact h

end NUMINAMATH_GPT_number_equation_l438_43831


namespace NUMINAMATH_GPT_find_other_number_l438_43847

/--
Given two numbers A and B, where:
    * The reciprocal of the HCF of A and B is \( \frac{1}{13} \).
    * The reciprocal of the LCM of A and B is \( \frac{1}{312} \).
    * A = 24
Prove that B = 169.
-/
theorem find_other_number 
  (A B : ℕ) 
  (h1 : A = 24)
  (h2 : (Nat.gcd A B) = 13)
  (h3 : (Nat.lcm A B) = 312) : 
  B = 169 := 
by 
  sorry

end NUMINAMATH_GPT_find_other_number_l438_43847


namespace NUMINAMATH_GPT_range_of_a_l438_43830

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x ≤ 2 then -x + 5 else a^x + 2 * a + 2

theorem range_of_a (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) :
  (∀ y ∈ Set.range (f a), y ≥ 3) ↔ (a ∈ Set.Ici (1/2) ∪ Set.Ioi 1) :=
sorry

end NUMINAMATH_GPT_range_of_a_l438_43830


namespace NUMINAMATH_GPT_min_value_f_when_a1_l438_43850

def f (x : ℝ) (a : ℝ) : ℝ := x^2 + |x - a|

theorem min_value_f_when_a1 : ∀ x : ℝ, f x 1 ≥ 3/4 :=
by sorry

end NUMINAMATH_GPT_min_value_f_when_a1_l438_43850


namespace NUMINAMATH_GPT_savannah_wrapped_gifts_with_second_roll_l438_43820

theorem savannah_wrapped_gifts_with_second_roll (total_gifts rolls_used roll_1_gifts roll_3_gifts roll_2_gifts : ℕ) 
  (h1 : total_gifts = 12) 
  (h2 : rolls_used = 3) 
  (h3 : roll_1_gifts = 3) 
  (h4 : roll_3_gifts = 4)
  (h5 : total_gifts - roll_1_gifts - roll_3_gifts = roll_2_gifts) :
  roll_2_gifts = 5 := 
by
  sorry

end NUMINAMATH_GPT_savannah_wrapped_gifts_with_second_roll_l438_43820


namespace NUMINAMATH_GPT_bob_password_probability_l438_43879

def num_non_negative_single_digits : ℕ := 10
def num_odd_single_digits : ℕ := 5
def num_even_positive_single_digits : ℕ := 4
def probability_first_digit_odd : ℚ := num_odd_single_digits / num_non_negative_single_digits
def probability_middle_letter : ℚ := 1
def probability_last_digit_even_positive : ℚ := num_even_positive_single_digits / num_non_negative_single_digits

theorem bob_password_probability :
  probability_first_digit_odd * probability_middle_letter * probability_last_digit_even_positive = 1 / 5 :=
by
  sorry

end NUMINAMATH_GPT_bob_password_probability_l438_43879


namespace NUMINAMATH_GPT_distance_greater_than_school_l438_43872

-- Let d1, d2, and d3 be the distances given as the conditions
def distance_orchard_to_house : ℕ := 800
def distance_house_to_pharmacy : ℕ := 1300
def distance_pharmacy_to_school : ℕ := 1700

-- The total distance from orchard to pharmacy via the house
def total_distance_orchard_to_pharmacy : ℕ :=
  distance_orchard_to_house + distance_house_to_pharmacy

-- The difference between the total distance from orchard to pharmacy and the distance from pharmacy to school
def distance_difference : ℕ :=
  total_distance_orchard_to_pharmacy - distance_pharmacy_to_school

-- The theorem to prove
theorem distance_greater_than_school :
  distance_difference = 400 := sorry

end NUMINAMATH_GPT_distance_greater_than_school_l438_43872


namespace NUMINAMATH_GPT_minimum_tan_theta_is_sqrt7_l438_43894

noncomputable def min_tan_theta (z : ℂ) : ℝ := (Complex.abs (Complex.im z) / Complex.abs (Complex.re z))

theorem minimum_tan_theta_is_sqrt7 {z : ℂ} 
  (hz_real : 0 ≤ Complex.re z)
  (hz_imag : 0 ≤ Complex.im z)
  (hz_condition : Complex.abs (z^2 + 2) ≤ Complex.abs z) :
  min_tan_theta z = Real.sqrt 7 := sorry

end NUMINAMATH_GPT_minimum_tan_theta_is_sqrt7_l438_43894


namespace NUMINAMATH_GPT_undefined_values_l438_43849

theorem undefined_values (b : ℝ) : (b^2 - 9 = 0) ↔ (b = -3 ∨ b = 3) := by
  sorry

end NUMINAMATH_GPT_undefined_values_l438_43849
