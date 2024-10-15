import Mathlib

namespace NUMINAMATH_GPT_extra_flowers_l2180_218071

-- Definitions from the conditions
def tulips : Nat := 57
def roses : Nat := 73
def daffodils : Nat := 45
def sunflowers : Nat := 35
def used_flowers : Nat := 181

-- Statement to prove
theorem extra_flowers : (tulips + roses + daffodils + sunflowers) - used_flowers = 29 := by
  sorry

end NUMINAMATH_GPT_extra_flowers_l2180_218071


namespace NUMINAMATH_GPT_find_angle_A_find_area_triangle_l2180_218097

-- Definitions for the triangle and the angles
def triangle (A B C : ℝ) : Prop :=
  0 < A ∧ A < Real.pi ∧ 0 < B ∧ B < Real.pi ∧ 0 < C ∧ C < Real.pi ∧ A + B + C = Real.pi

-- Given conditions
variables (a b c A B C : ℝ)
variables (hTriangle : triangle A B C)
variables (hEq : 2 * b * Real.cos A - Real.sqrt 3 * c * Real.cos A = Real.sqrt 3 * a * Real.cos C)
variables (hAngleB : B = Real.pi / 6)
variables (hMedianAM : Real.sqrt 7 = Real.sqrt (b^2 + (b / 2)^2 - 2 * b * (b / 2) * Real.cos (2 * Real.pi / 3)))

-- Proof statements
theorem find_angle_A : A = Real.pi / 6 :=
sorry

theorem find_area_triangle : (1/2) * b^2 * Real.sin C = Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_find_angle_A_find_area_triangle_l2180_218097


namespace NUMINAMATH_GPT_sugar_needed_287_163_l2180_218060

theorem sugar_needed_287_163 :
  let sugar_stored := 287
  let additional_sugar_needed := 163
  sugar_stored + additional_sugar_needed = 450 :=
by
  let sugar_stored := 287
  let additional_sugar_needed := 163
  sorry

end NUMINAMATH_GPT_sugar_needed_287_163_l2180_218060


namespace NUMINAMATH_GPT_intersection_points_of_line_l2180_218024

theorem intersection_points_of_line (x y : ℝ) :
  ((y = 2 * x - 1) → (y = 0 → x = 0.5)) ∧
  ((y = 2 * x - 1) → (x = 0 → y = -1)) :=
by sorry

end NUMINAMATH_GPT_intersection_points_of_line_l2180_218024


namespace NUMINAMATH_GPT_ab_sum_pow_eq_neg_one_l2180_218062

theorem ab_sum_pow_eq_neg_one (a b : ℝ) (h : |a - 3| + (b + 4)^2 = 0) : (a + b) ^ 2003 = -1 := 
by
  sorry

end NUMINAMATH_GPT_ab_sum_pow_eq_neg_one_l2180_218062


namespace NUMINAMATH_GPT_number_of_lines_l2180_218010

-- Define the point (0, 1)
def point : ℝ × ℝ := (0, 1)

-- Define the parabola y^2 = 4x
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- Define the condition that a line intersects a parabola at only one point
def line_intersects_parabola_at_one_point (m b x y : ℝ) : Prop :=
  y - (m * x + b) = 0 ∧ parabola x y

-- The proof problem: Prove there are 3 such lines
theorem number_of_lines : ∃ (n : ℕ), n = 3 ∧ (
  ∃ (m b : ℝ), line_intersects_parabola_at_one_point m b 0 1) :=
sorry

end NUMINAMATH_GPT_number_of_lines_l2180_218010


namespace NUMINAMATH_GPT_determine_n_l2180_218028

theorem determine_n (x a : ℝ) (n : ℕ)
  (h1 : (n.choose 3) * x^(n-3) * a^3 = 120)
  (h2 : (n.choose 4) * x^(n-4) * a^4 = 360)
  (h3 : (n.choose 5) * x^(n-5) * a^5 = 720) :
  n = 12 :=
sorry

end NUMINAMATH_GPT_determine_n_l2180_218028


namespace NUMINAMATH_GPT_HA_appears_at_least_once_l2180_218013

-- Define the set of letters to be arranged
def letters : List Char := ['A', 'A', 'A', 'H', 'H']

-- Define a function to count the number of ways to arrange letters such that "HA" appears at least once
def countHA(A : List Char) : Nat := sorry

-- The proof problem to establish that there are 9 such arrangements
theorem HA_appears_at_least_once : countHA letters = 9 :=
sorry

end NUMINAMATH_GPT_HA_appears_at_least_once_l2180_218013


namespace NUMINAMATH_GPT_waiter_tables_l2180_218063

theorem waiter_tables (total_customers : ℕ) (left_customers : ℕ) (people_per_table : ℕ) (remaining_customers : ℕ) (tables : ℕ) :
  total_customers = 62 →
  left_customers = 17 →
  people_per_table = 9 →
  remaining_customers = total_customers - left_customers →
  tables = remaining_customers / people_per_table →
  tables = 5 := by
  sorry

end NUMINAMATH_GPT_waiter_tables_l2180_218063


namespace NUMINAMATH_GPT_tax_collected_from_village_l2180_218026

-- Definitions according to the conditions in the problem
def MrWillamTax : ℝ := 500
def MrWillamPercentage : ℝ := 0.21701388888888893

-- The theorem to prove the total tax collected
theorem tax_collected_from_village : ∃ (total_collected : ℝ), MrWillamPercentage * total_collected = MrWillamTax ∧ total_collected = 2303.7037037037035 :=
sorry

end NUMINAMATH_GPT_tax_collected_from_village_l2180_218026


namespace NUMINAMATH_GPT_find_g_at_3_l2180_218093

theorem find_g_at_3 (g : ℝ → ℝ) (h : ∀ x : ℝ, g (3 * x - 2) = 4 * x + 1) : g 3 = 23 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_g_at_3_l2180_218093


namespace NUMINAMATH_GPT_grid_problem_l2180_218004

theorem grid_problem 
  (A B : ℕ) 
  (grid : (Fin 3) → (Fin 3) → ℕ)
  (h1 : ∀ i, grid 0 i ≠ grid 1 i)
  (h2 : ∀ i, grid 0 i ≠ grid 2 i)
  (h3 : ∀ i, grid 1 i ≠ grid 2 i)
  (h4 : ∀ i, (∃! x, grid x i = 1))
  (h5 : ∀ i, (∃! x, grid x i = 2))
  (h6 : ∀ i, (∃! x, grid x i = 3))
  (h7 : grid 1 2 = A)
  (h8 : grid 2 2 = B) : 
  A + B + 4 = 8 :=
by sorry

end NUMINAMATH_GPT_grid_problem_l2180_218004


namespace NUMINAMATH_GPT_tangent_line_to_circle_range_mn_l2180_218083

theorem tangent_line_to_circle_range_mn (m n : ℝ) 
  (h1 : (m + 1) * (m + 1) + (n + 1) * (n + 1) = 4) :
  (m + n ≤ 2 - 2 * Real.sqrt 2) ∨ (m + n ≥ 2 + 2 * Real.sqrt 2) :=
sorry

end NUMINAMATH_GPT_tangent_line_to_circle_range_mn_l2180_218083


namespace NUMINAMATH_GPT_sine_ratio_comparison_l2180_218053

theorem sine_ratio_comparison : (Real.sin (1 * Real.pi / 180) / Real.sin (2 * Real.pi / 180)) < (Real.sin (3 * Real.pi / 180) / Real.sin (4 * Real.pi / 180)) :=
sorry

end NUMINAMATH_GPT_sine_ratio_comparison_l2180_218053


namespace NUMINAMATH_GPT_value_of_c_minus_a_l2180_218080

variables (a b c : ℝ)

theorem value_of_c_minus_a (h1 : (a + b) / 2 = 45) (h2 : (b + c) / 2 = 60) : (c - a) = 30 :=
by
  have h3 : a + b = 90 := by sorry
  have h4 : b + c = 120 := by sorry
  -- now we have the required form of the problem statement
  -- c - a = 120 - 90
  sorry

end NUMINAMATH_GPT_value_of_c_minus_a_l2180_218080


namespace NUMINAMATH_GPT_find_salary_May_l2180_218086

-- Define the salaries for each month as variables
variables (J F M A May : ℝ)

-- Declare the conditions as hypotheses
def avg_salary_Jan_to_Apr := (J + F + M + A) / 4 = 8000
def avg_salary_Feb_to_May := (F + M + A + May) / 4 = 8100
def salary_Jan := J = 6100

-- The theorem stating the salary for the month of May
theorem find_salary_May (h1 : avg_salary_Jan_to_Apr J F M A) (h2 : avg_salary_Feb_to_May F M A May) (h3 : salary_Jan J) :
  May = 6500 :=
  sorry

end NUMINAMATH_GPT_find_salary_May_l2180_218086


namespace NUMINAMATH_GPT_triangle_area_l2180_218057

def is_isosceles (a b c : ℝ) : Prop :=
  (a = b ∨ a = c ∨ b = c)

def has_perimeter (a b c p : ℝ) : Prop :=
  a + b + c = p

def has_altitude (base side altitude : ℝ) : Prop :=
  (base / 2) ^ 2 + altitude ^ 2 = side ^ 2

def area_of_triangle (a base altitude : ℝ) : ℝ :=
  0.5 * base * altitude

theorem triangle_area (a b c : ℝ)
  (h_iso : is_isosceles a b c)
  (h_p : has_perimeter a b c 40)
  (h_alt : has_altitude (2 * a) b 12) :
  area_of_triangle a (2 * a) 12 = 76.8 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_l2180_218057


namespace NUMINAMATH_GPT_scientific_notation_41600_l2180_218065

theorem scientific_notation_41600 : (4.16 * 10^4) = 41600 := by
  sorry

end NUMINAMATH_GPT_scientific_notation_41600_l2180_218065


namespace NUMINAMATH_GPT_gold_coins_percent_l2180_218000

variable (total_objects beads papers coins silver_gold total_gold : ℝ)
variable (h1 : total_objects = 100)
variable (h2 : beads = 15)
variable (h3 : papers = 10)
variable (h4 : silver_gold = 30)
variable (h5 : total_gold = 52.5)

theorem gold_coins_percent : (total_objects - beads - papers) * (100 - silver_gold) / 100 = total_gold :=
by 
  -- Insert proof here
  sorry

end NUMINAMATH_GPT_gold_coins_percent_l2180_218000


namespace NUMINAMATH_GPT_prob_same_points_eq_prob_less_than_seven_eq_prob_greater_or_equal_eleven_eq_l2180_218061

def num_outcomes := 36

def same_points_events := 6
def less_than_seven_events := 15
def greater_than_or_equal_eleven_events := 3

def prob_same_points := (same_points_events : ℚ) / num_outcomes
def prob_less_than_seven := (less_than_seven_events : ℚ) / num_outcomes
def prob_greater_or_equal_eleven := (greater_than_or_equal_eleven_events : ℚ) / num_outcomes

theorem prob_same_points_eq : prob_same_points = 1 / 6 := by
  sorry

theorem prob_less_than_seven_eq : prob_less_than_seven = 5 / 12 := by
  sorry

theorem prob_greater_or_equal_eleven_eq : prob_greater_or_equal_eleven = 1 / 12 := by
  sorry

end NUMINAMATH_GPT_prob_same_points_eq_prob_less_than_seven_eq_prob_greater_or_equal_eleven_eq_l2180_218061


namespace NUMINAMATH_GPT_sum_of_series_l2180_218064

theorem sum_of_series (a b : ℝ) (h_pos : 0 < a ∧ 0 < b) (h_gt : a > b) :
  ∑' n, 1 / ( ((n - 1) * a + (n - 2) * b) * (n * a + (n - 1) * b)) = 1 / ((a + b) * b) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_series_l2180_218064


namespace NUMINAMATH_GPT_find_second_term_l2180_218041

theorem find_second_term 
  (a : ℕ → ℕ) 
  (S : ℕ → ℕ) 
  (h_sum : ∀ n, S n = n * (2 * n + 1))
  (h_S1 : S 1 = a 1) 
  (h_S2 : S 2 = a 1 + a 2) 
  (h_a1 : a 1 = 3) : 
  a 2 = 7 := 
sorry

end NUMINAMATH_GPT_find_second_term_l2180_218041


namespace NUMINAMATH_GPT_students_passing_in_sixth_year_l2180_218015

def numStudentsPassed (year : ℕ) : ℕ :=
 if year = 1 then 200 else 
 if year = 2 then 300 else 
 if year = 3 then 390 else 
 if year = 4 then 565 else 
 if year = 5 then 643 else 
 if year = 6 then 780 else 0

theorem students_passing_in_sixth_year : numStudentsPassed 6 = 780 := by
  sorry

end NUMINAMATH_GPT_students_passing_in_sixth_year_l2180_218015


namespace NUMINAMATH_GPT_evaluate_fractions_l2180_218048

theorem evaluate_fractions (a b c : ℝ) 
  (h : a / (30 - a) + b / (70 - b) + c / (55 - c) = 8) : 
  6 / (30 - a) + 14 / (70 - b) + 11 / (55 - c) = 2.2 := 
by
  sorry

end NUMINAMATH_GPT_evaluate_fractions_l2180_218048


namespace NUMINAMATH_GPT_bisect_segment_l2180_218076

variables {A B C D E P : Point}
variables {α β γ δ ε : Real} -- angles in degrees
variables {BD CE : Line}

-- Geometric predicates
def Angle (x y z : Point) : Real := sorry -- calculates the angle ∠xyz

def isMidpoint (M A B : Point) : Prop := sorry -- M is the midpoint of segment AB

-- Given Conditions
variables (h1 : convex_pentagon A B C D E)
          (h2 : Angle B A C = Angle C A D ∧ Angle C A D = Angle D A E)
          (h3 : Angle A B C = Angle A C D ∧ Angle A C D = Angle A D E)
          (h4 : intersects BD CE P)

-- Conclusion to be proved
theorem bisect_segment : isMidpoint P C D :=
by {
  sorry -- proof to be filled in
}

end NUMINAMATH_GPT_bisect_segment_l2180_218076


namespace NUMINAMATH_GPT_rice_yield_l2180_218027

theorem rice_yield (X : ℝ) (h1 : 0 ≤ X ∧ X ≤ 40) :
    0.75 * 400 * X + 0.25 * 800 * X + 500 * (40 - X) = 20000 := by
  sorry

end NUMINAMATH_GPT_rice_yield_l2180_218027


namespace NUMINAMATH_GPT_tangent_line_at_1_l2180_218002

noncomputable def f (x : ℝ) : ℝ := Real.log x - 3 * x

noncomputable def f' (x : ℝ) : ℝ := 1 / x - 3

theorem tangent_line_at_1 :
  let y := f 1
  let k := f' 1
  y = -3 ∧ k = -2 →
  ∀ (x y : ℝ), y = k * (x - 1) + f 1 ↔ 2 * x + y + 1 = 0 :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_at_1_l2180_218002


namespace NUMINAMATH_GPT_profit_percentage_l2180_218011

theorem profit_percentage (SP : ℝ) (h1 : SP > 0) (h2 : CP = 0.99 * SP) : (SP - CP) / CP * 100 = 1.01 :=
by
  sorry

end NUMINAMATH_GPT_profit_percentage_l2180_218011


namespace NUMINAMATH_GPT_original_number_divisible_l2180_218046

theorem original_number_divisible (N M R : ℕ) (n : ℕ) (hN : N = 1000 * M + R)
  (hDiff : (M - R) % n = 0) (hn : n = 7 ∨ n = 11 ∨ n = 13) : N % n = 0 :=
by
  sorry

end NUMINAMATH_GPT_original_number_divisible_l2180_218046


namespace NUMINAMATH_GPT_last_row_number_l2180_218066

/-
Given:
1. Each row forms an arithmetic sequence.
2. The common differences of the rows are:
   - 1st row: common difference = 1
   - 2nd row: common difference = 2
   - 3rd row: common difference = 4
   - ...
   - 2015th row: common difference = 2^2014
3. The nth row starts with \( (n+1) \times 2^{n-2} \).

Prove:
The number in the last row (2016th row) is \( 2017 \times 2^{2014} \).
-/
theorem last_row_number
  (common_diff : ℕ → ℕ)
  (h1 : common_diff 1 = 1)
  (h2 : common_diff 2 = 2)
  (h3 : common_diff 3 = 4)
  (h_general : ∀ n, common_diff n = 2^(n-1))
  (first_number_in_row : ℕ → ℕ)
  (first_number_in_row_def : ∀ n, first_number_in_row n = (n + 1) * 2^(n - 2)) :
  first_number_in_row 2016 = 2017 * 2^2014 := by
    sorry

end NUMINAMATH_GPT_last_row_number_l2180_218066


namespace NUMINAMATH_GPT_ambiguous_dates_in_year_l2180_218025

def is_ambiguous_date (m d : ℕ) : Prop :=
  1 ≤ m ∧ m ≤ 12 ∧ 1 ≤ d ∧ d ≤ 12 ∧ m ≠ d

theorem ambiguous_dates_in_year :
  ∃ n : ℕ, n = 132 ∧ (∀ m d : ℕ, is_ambiguous_date m d → n = 132) :=
sorry

end NUMINAMATH_GPT_ambiguous_dates_in_year_l2180_218025


namespace NUMINAMATH_GPT_cost_of_fencing_per_meter_l2180_218033

theorem cost_of_fencing_per_meter (length breadth : ℕ) (total_cost : ℚ) 
    (h_length : length = 61) 
    (h_rule : length = breadth + 22) 
    (h_total_cost : total_cost = 5300) :
    total_cost / (2 * length + 2 * breadth) = 26.5 := 
by 
  sorry

end NUMINAMATH_GPT_cost_of_fencing_per_meter_l2180_218033


namespace NUMINAMATH_GPT_determinant_expression_l2180_218016

theorem determinant_expression (a b c p q : ℝ) 
  (h_root : ∀ x : ℝ, (x = a ∨ x = b ∨ x = c) → (Polynomial.eval x (Polynomial.X ^ 3 - 3 * Polynomial.C p * Polynomial.X + 2 * Polynomial.C q) = 0)) :
  Matrix.det ![![2 + a, 1, 1], ![1, 2 + b, 1], ![1, 1, 2 + c]] = -3 * p - 2 * q + 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_determinant_expression_l2180_218016


namespace NUMINAMATH_GPT_probability_of_U_l2180_218019

def pinyin : List Char := ['S', 'H', 'U', 'X', 'U', 'E']
def total_letters : Nat := 6
def u_count : Nat := 2

theorem probability_of_U :
  ((u_count : ℚ) / (total_letters : ℚ)) = (1 / 3) :=
by
  sorry

end NUMINAMATH_GPT_probability_of_U_l2180_218019


namespace NUMINAMATH_GPT_find_x_l2180_218017

theorem find_x (x : ℝ) (h1 : ⌈x⌉ * x = 210) (h2 : ⌈x⌉ - 1 < x ∧ x ≤ ⌈x⌉) (h3 : 0 < x) : x = 14 :=
sorry

end NUMINAMATH_GPT_find_x_l2180_218017


namespace NUMINAMATH_GPT_ratio_X_N_l2180_218074

-- Given conditions as definitions
variables (P Q M N X : ℝ)
variables (hM : M = 0.40 * Q)
variables (hQ : Q = 0.30 * P)
variables (hN : N = 0.60 * P)
variables (hX : X = 0.25 * M)

-- Prove that X / N == 1 / 20
theorem ratio_X_N : X / N = 1 / 20 :=
by
  sorry

end NUMINAMATH_GPT_ratio_X_N_l2180_218074


namespace NUMINAMATH_GPT_find_M_N_l2180_218055

-- Define positive integers less than 10
def is_pos_int_lt_10 (x : ℕ) : Prop := x > 0 ∧ x < 10

-- Main theorem to prove M = 5 and N = 6 given the conditions
theorem find_M_N (M N : ℕ) (hM : is_pos_int_lt_10 M) (hN : is_pos_int_lt_10 N) 
  (h : 8 * (10 ^ 7) * M + 420852 * 9 = N * (10 ^ 7) * 9889788 * 11) : 
  M = 5 ∧ N = 6 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_M_N_l2180_218055


namespace NUMINAMATH_GPT_solution_set_of_inequality_l2180_218031

theorem solution_set_of_inequality (x : ℝ) : (1 / x ≤ x) ↔ (-1 ≤ x ∧ x < 0) ∨ (x ≥ 1) := sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l2180_218031


namespace NUMINAMATH_GPT_hypotenuse_length_l2180_218035

theorem hypotenuse_length (x y : ℝ) (V1 V2 : ℝ) 
  (h1 : V1 = 1350 * Real.pi) 
  (h2 : V2 = 2430 * Real.pi) 
  (h3 : (1/3) * Real.pi * y^2 * x = V1) 
  (h4 : (1/3) * Real.pi * x^2 * y = V2) 
  : Real.sqrt (x^2 + y^2) = Real.sqrt 954 :=
sorry

end NUMINAMATH_GPT_hypotenuse_length_l2180_218035


namespace NUMINAMATH_GPT_kittens_price_l2180_218047

theorem kittens_price (x : ℕ) 
  (h1 : 2 * x + 5 = 17) : x = 6 := by
  sorry

end NUMINAMATH_GPT_kittens_price_l2180_218047


namespace NUMINAMATH_GPT_find_four_numbers_l2180_218044

theorem find_four_numbers (a b c d : ℚ) :
  ((a + b = 1) ∧ (a + c = 5) ∧ 
   ((a + d = 8 ∧ b + c = 9) ∨ (a + d = 9 ∧ b + c = 8)) ) →
  ((a = -3/2 ∧ b = 5/2 ∧ c = 13/2 ∧ d = 19/2) ∨ 
   (a = -1 ∧ b = 2 ∧ c = 6 ∧ d = 10)) :=
  by
    sorry

end NUMINAMATH_GPT_find_four_numbers_l2180_218044


namespace NUMINAMATH_GPT_packets_for_dollars_l2180_218073

variable (P R C : ℕ)

theorem packets_for_dollars :
  let dimes := 10 * C
  let taxable_dimes := 9 * C
  ∃ x, x = taxable_dimes * P / R :=
sorry

end NUMINAMATH_GPT_packets_for_dollars_l2180_218073


namespace NUMINAMATH_GPT_blue_notes_per_red_note_l2180_218052

-- Given conditions
def total_red_notes : ℕ := 5 * 6
def additional_blue_notes : ℕ := 10
def total_notes : ℕ := 100
def total_blue_notes := total_notes - total_red_notes

-- Proposition that needs to be proved
theorem blue_notes_per_red_note (x : ℕ) : total_red_notes * x + additional_blue_notes = total_blue_notes → x = 2 := by
  intro h
  sorry

end NUMINAMATH_GPT_blue_notes_per_red_note_l2180_218052


namespace NUMINAMATH_GPT_bird_cost_l2180_218050

variable (scost bcost : ℕ)

theorem bird_cost (h1 : bcost = 2 * scost)
                  (h2 : (5 * bcost + 3 * scost) = (3 * bcost + 5 * scost) + 20) :
                  scost = 10 ∧ bcost = 20 :=
by {
  sorry
}

end NUMINAMATH_GPT_bird_cost_l2180_218050


namespace NUMINAMATH_GPT_system_of_equations_soln_l2180_218087

theorem system_of_equations_soln :
  {p : ℝ × ℝ | ∃ a : ℝ, (a * p.1 + p.2 = 2 * a + 3) ∧ (p.1 - a * p.2 = a + 4)} =
  {p : ℝ × ℝ | (p.1 - 3)^2 + (p.2 - 1)^2 = 5} \ {⟨2, -1⟩} :=
by
  sorry

end NUMINAMATH_GPT_system_of_equations_soln_l2180_218087


namespace NUMINAMATH_GPT_hyperbola_equation_l2180_218094

noncomputable def hyperbola : Prop :=
  ∃ (a b : ℝ), 
    (2 : ℝ) * a = (3 : ℝ) * b ∧
    ∀ (x y : ℝ), (4 * x^2 - 9 * y^2 = -32) → (x = 1) ∧ (y = 2)

theorem hyperbola_equation (a b : ℝ) :
  (2 * a = 3 * b) ∧ (∀ x y : ℝ, 4 * x^2 - 9 * y^2 = -32 → x = 1 ∧ y = 2) → 
  (9 / 32 * y^2 - x^2 / 8 = 1) :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_equation_l2180_218094


namespace NUMINAMATH_GPT_range_of_uv_sq_l2180_218092

theorem range_of_uv_sq (u v w : ℝ) (h₀ : 0 ≤ u) (h₁ : 0 ≤ v) (h₂ : 0 ≤ w) (h₃ : u + v + w = 2) :
  0 ≤ u^2 * v^2 + v^2 * w^2 + w^2 * u^2 ∧ u^2 * v^2 + v^2 * w^2 + w^2 * u^2 ≤ 1 :=
sorry

end NUMINAMATH_GPT_range_of_uv_sq_l2180_218092


namespace NUMINAMATH_GPT_probability_at_least_one_girl_l2180_218098

theorem probability_at_least_one_girl 
  (boys girls : ℕ) 
  (total : boys + girls = 7) 
  (combinations_total : ℕ := Nat.choose 7 2) 
  (combinations_boys : ℕ := Nat.choose 4 2) 
  (prob_no_girls : ℚ := combinations_boys / combinations_total) 
  (prob_at_least_one_girl : ℚ := 1 - prob_no_girls) :
  boys = 4 ∧ girls = 3 → prob_at_least_one_girl = 5 / 7 := 
by
  intro h
  cases h
  sorry

end NUMINAMATH_GPT_probability_at_least_one_girl_l2180_218098


namespace NUMINAMATH_GPT_solve_combination_eq_l2180_218058

theorem solve_combination_eq (x : ℕ) (h : x ≥ 3) : 
  (Nat.choose x 3 + Nat.choose x 2 = 12 * (x - 1)) ↔ (x = 9) := 
by
  sorry

end NUMINAMATH_GPT_solve_combination_eq_l2180_218058


namespace NUMINAMATH_GPT_find_a_b_l2180_218042

noncomputable def parabola_props (a b : ℝ) : Prop :=
a ≠ 0 ∧ 
∀ x : ℝ, a * x^2 + b * x - 4 = (1 / 2) * x^2 + x - 4

theorem find_a_b {a b : ℝ} (h1 : parabola_props a b) : 
a = 1 / 2 ∧ b = -1 :=
sorry

end NUMINAMATH_GPT_find_a_b_l2180_218042


namespace NUMINAMATH_GPT_beef_weight_before_processing_l2180_218068

-- Define the initial weight of the beef.
def W_initial := 1070.5882

-- Define the loss percentages.
def loss1 := 0.20
def loss2 := 0.15
def loss3 := 0.25

-- Define the final weight after all losses.
def W_final := 546.0

-- The main proof goal: show that W_initial results in W_final after considering the weight losses.
theorem beef_weight_before_processing (W_initial W_final : ℝ) (loss1 loss2 loss3 : ℝ) :
  W_final = (1 - loss3) * (1 - loss2) * (1 - loss1) * W_initial :=
by
  sorry

end NUMINAMATH_GPT_beef_weight_before_processing_l2180_218068


namespace NUMINAMATH_GPT_find_c_l2180_218029

theorem find_c
  (m b d c : ℝ)
  (h : m = b * d * c / (d + c)) :
  c = m * d / (b * d - m) :=
sorry

end NUMINAMATH_GPT_find_c_l2180_218029


namespace NUMINAMATH_GPT_perimeter_circumradius_ratio_neq_l2180_218077

-- Define the properties for the equilateral triangle
def Triangle (A K R P : ℝ) : Prop :=
  P = 3 * A ∧ K = A^2 * Real.sqrt 3 / 4 ∧ R = A * Real.sqrt 3 / 3

-- Define the properties for the square
def Square (b k r p : ℝ) : Prop :=
  p = 4 * b ∧ k = b^2 ∧ r = b * Real.sqrt 2 / 2

-- Main statement to prove
theorem perimeter_circumradius_ratio_neq 
  (A b K R P k r p : ℝ)
  (hT : Triangle A K R P) 
  (hS : Square b k r p) :
  P / p ≠ R / r := 
by
  rcases hT with ⟨hP, hK, hR⟩
  rcases hS with ⟨hp, hk, hr⟩
  sorry

end NUMINAMATH_GPT_perimeter_circumradius_ratio_neq_l2180_218077


namespace NUMINAMATH_GPT_tangent_line_curve_l2180_218020

theorem tangent_line_curve (a b : ℝ)
  (h1 : ∀ (x : ℝ), (x - (x^2 + a*x + b) + 1 = 0) ↔ (a = 1 ∧ b = 1))
  (h2 : ∀ (y : ℝ), (0, y) ∈ { p : ℝ × ℝ | p.2 = 0 ^ 2 + a * 0 + b }) :
  a = 1 ∧ b = 1 :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_curve_l2180_218020


namespace NUMINAMATH_GPT_geom_sum_3m_l2180_218081

variable (a_n : ℕ → ℝ)
variable (S : ℕ → ℝ)
variable (m : ℕ)

axiom geom_sum_m : S m = 10
axiom geom_sum_2m : S (2 * m) = 30

theorem geom_sum_3m : S (3 * m) = 70 :=
by
  sorry

end NUMINAMATH_GPT_geom_sum_3m_l2180_218081


namespace NUMINAMATH_GPT_max_product_l2180_218090

noncomputable def max_of_product (x y : ℝ) : ℝ := x * y

theorem max_product (x y : ℝ) (h1 : x ∈ Set.Ioi 0) (h2 : y ∈ Set.Ioi 0) (h3 : x + 4 * y = 1) :
  max_of_product x y ≤ 1 / 16 := sorry

end NUMINAMATH_GPT_max_product_l2180_218090


namespace NUMINAMATH_GPT_angle_terminal_side_equiv_l2180_218034

def angle_equiv_terminal_side (θ₁ θ₂ : ℝ) : Prop :=
  ∃ k : ℤ, θ₂ = θ₁ + 2 * k * Real.pi

theorem angle_terminal_side_equiv : angle_equiv_terminal_side (-Real.pi / 3) (5 * Real.pi / 3) :=
by
  sorry

end NUMINAMATH_GPT_angle_terminal_side_equiv_l2180_218034


namespace NUMINAMATH_GPT_snakes_in_pond_l2180_218014

theorem snakes_in_pond (S : ℕ) (alligators : ℕ := 10) (total_eyes : ℕ := 56) (alligator_eyes : ℕ := 2) (snake_eyes : ℕ := 2) :
  (alligators * alligator_eyes) + (S * snake_eyes) = total_eyes → S = 18 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_snakes_in_pond_l2180_218014


namespace NUMINAMATH_GPT_sqrt_combination_l2180_218096

theorem sqrt_combination : 
    ∃ (k : ℝ), (k * Real.sqrt 2 = Real.sqrt 8) ∧ 
    (¬(∃ (k : ℝ), (k * Real.sqrt 2 = Real.sqrt 3))) ∧ 
    (¬(∃ (k : ℝ), (k * Real.sqrt 2 = Real.sqrt 12))) ∧ 
    (¬(∃ (k : ℝ), (k * Real.sqrt 2 = Real.sqrt 0.2))) :=
by
  sorry

end NUMINAMATH_GPT_sqrt_combination_l2180_218096


namespace NUMINAMATH_GPT_pie_shop_revenue_l2180_218012

def costPerSlice : Int := 5
def slicesPerPie : Int := 4
def piesSold : Int := 9

theorem pie_shop_revenue : (costPerSlice * slicesPerPie * piesSold) = 180 := 
by
  sorry

end NUMINAMATH_GPT_pie_shop_revenue_l2180_218012


namespace NUMINAMATH_GPT_martin_class_number_l2180_218008

theorem martin_class_number (b : ℕ) (h1 : 100 < b) (h2 : b < 200) 
  (h3 : b % 3 = 2) (h4 : b % 4 = 1) (h5 : b % 5 = 1) : 
  b = 101 ∨ b = 161 := 
by
  sorry

end NUMINAMATH_GPT_martin_class_number_l2180_218008


namespace NUMINAMATH_GPT_fraction_received_A_correct_l2180_218032

def fraction_of_students_received_A := 0.7
def fraction_of_students_received_B := 0.2
def fraction_of_students_received_A_or_B := 0.9

theorem fraction_received_A_correct :
  fraction_of_students_received_A_or_B - fraction_of_students_received_B = fraction_of_students_received_A :=
by
  sorry

end NUMINAMATH_GPT_fraction_received_A_correct_l2180_218032


namespace NUMINAMATH_GPT_additional_charge_fraction_of_mile_l2180_218088

-- Conditions
def initial_fee : ℝ := 2.25
def additional_charge_per_mile_fraction : ℝ := 0.15
def total_charge (distance : ℝ) : ℝ := 2.25 + 0.15 * distance
def trip_distance : ℝ := 3.6
def total_cost : ℝ := 3.60

-- Question
theorem additional_charge_fraction_of_mile :
  ∃ f : ℝ, total_cost = initial_fee + additional_charge_per_mile_fraction * 3.6 ∧ f = 1 / 9 :=
by
  sorry

end NUMINAMATH_GPT_additional_charge_fraction_of_mile_l2180_218088


namespace NUMINAMATH_GPT_sallys_woodworking_llc_reimbursement_l2180_218005

/-
Conditions:
1. Remy paid $20,700 for 150 pieces of furniture.
2. The cost of a piece of furniture is $134.
-/
def reimbursement_amount (pieces_paid : ℕ) (total_paid : ℕ) (price_per_piece : ℕ) : ℕ :=
  total_paid - (pieces_paid * price_per_piece)

theorem sallys_woodworking_llc_reimbursement :
  reimbursement_amount 150 20700 134 = 600 :=
by 
  sorry

end NUMINAMATH_GPT_sallys_woodworking_llc_reimbursement_l2180_218005


namespace NUMINAMATH_GPT_not_possible_last_digit_l2180_218043

theorem not_possible_last_digit :
  ∀ (S : ℕ) (a : Fin 111 → ℕ),
  (∀ i, a i ≤ 500) →
  (∀ i j, i ≠ j → a i ≠ a j) →
  (∀ i, (a i) % 10 = (S - a i) % 10) →
  False :=
by
  intro S a h1 h2 h3
  sorry

end NUMINAMATH_GPT_not_possible_last_digit_l2180_218043


namespace NUMINAMATH_GPT_tom_father_time_saved_correct_l2180_218082

def tom_father_jog_time_saved : Prop :=
  let monday_speed := 6
  let tuesday_speed := 5
  let thursday_speed := 4
  let saturday_speed := 5
  let daily_distance := 3
  let hours_to_minutes := 60

  let monday_time := daily_distance / monday_speed
  let tuesday_time := daily_distance / tuesday_speed
  let thursday_time := daily_distance / thursday_speed
  let saturday_time := daily_distance / saturday_speed

  let total_time_original := monday_time + tuesday_time + thursday_time + saturday_time
  let always_5mph_time := 4 * (daily_distance / 5)
  let time_saved := total_time_original - always_5mph_time

  let time_saved_minutes := time_saved * hours_to_minutes

  time_saved_minutes = 3

theorem tom_father_time_saved_correct : tom_father_jog_time_saved := by
  sorry

end NUMINAMATH_GPT_tom_father_time_saved_correct_l2180_218082


namespace NUMINAMATH_GPT_sec_240_eq_neg2_l2180_218018

noncomputable def sec (θ : ℝ) : ℝ := 1 / Real.cos θ

theorem sec_240_eq_neg2 : sec 240 = -2 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_sec_240_eq_neg2_l2180_218018


namespace NUMINAMATH_GPT_time_saved_by_taking_route_B_l2180_218089

-- Defining the times for the routes A and B
def time_route_A_one_way : ℕ := 5
def time_route_B_one_way : ℕ := 2

-- The total round trip times
def time_route_A_round_trip : ℕ := 2 * time_route_A_one_way
def time_route_B_round_trip : ℕ := 2 * time_route_B_one_way

-- The statement to prove
theorem time_saved_by_taking_route_B :
  time_route_A_round_trip - time_route_B_round_trip = 6 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_time_saved_by_taking_route_B_l2180_218089


namespace NUMINAMATH_GPT_factor_x_squared_minus_144_l2180_218099

theorem factor_x_squared_minus_144 (x : ℝ) : x^2 - 144 = (x - 12) * (x + 12) :=
by
  sorry

end NUMINAMATH_GPT_factor_x_squared_minus_144_l2180_218099


namespace NUMINAMATH_GPT_genuine_product_probability_l2180_218079

-- Define the probabilities as constants
def P_second_grade := 0.03
def P_third_grade := 0.01

-- Define the total probability (outcome must be either genuine or substandard)
def P_substandard := P_second_grade + P_third_grade
def P_genuine := 1 - P_substandard

-- The statement to be proved
theorem genuine_product_probability :
  P_genuine = 0.96 :=
sorry

end NUMINAMATH_GPT_genuine_product_probability_l2180_218079


namespace NUMINAMATH_GPT_explicit_formula_solution_set_l2180_218070

noncomputable def f : ℝ → ℝ 
| x => if 0 < x ∧ x ≤ 4 then Real.log x / Real.log 2 else
       if -4 ≤ x ∧ x < 0 then Real.log (-x) / Real.log 2 else
       0

theorem explicit_formula (x : ℝ) :
  f x = if 0 < x ∧ x ≤ 4 then Real.log x / Real.log 2 else
        if -4 ≤ x ∧ x < 0 then Real.log (-x) / Real.log 2 else
        0 := 
by 
  sorry 

theorem solution_set (x : ℝ) : 
  (0 < x ∧ x < 1 ∨ -4 < x ∧ x < -1) ↔ x * f x < 0 := 
by
  sorry

end NUMINAMATH_GPT_explicit_formula_solution_set_l2180_218070


namespace NUMINAMATH_GPT_house_A_cost_l2180_218039

theorem house_A_cost (base_salary earnings commission_rate total_houses cost_A cost_B cost_C : ℝ)
  (H_base_salary : base_salary = 3000)
  (H_earnings : earnings = 8000)
  (H_commission_rate : commission_rate = 0.02)
  (H_cost_B : cost_B = 3 * cost_A)
  (H_cost_C : cost_C = 2 * cost_A - 110000)
  (H_total_commission : earnings - base_salary = 5000)
  (H_total_cost : 5000 / commission_rate = 250000)
  (H_total_houses : base_salary + commission_rate * (cost_A + cost_B + cost_C) = earnings) :
  cost_A = 60000 := sorry

end NUMINAMATH_GPT_house_A_cost_l2180_218039


namespace NUMINAMATH_GPT_no_such_number_exists_l2180_218075

def is_digit (d : ℕ) : Prop := d ≥ 0 ∧ d ≤ 9

/-- Define the number N as a sequence of digits a_n a_{n-1} ... a_0 -/
def number (a b : ℕ) (n : ℕ) : ℕ := a * 10^n + b

theorem no_such_number_exists :
  ¬ ∃ (N a_n b : ℕ) (n : ℕ), is_digit a_n ∧ a_n ≠ 0 ∧ b < 10^n ∧
    N = number a_n b n ∧
    b = N / 57 :=
sorry

end NUMINAMATH_GPT_no_such_number_exists_l2180_218075


namespace NUMINAMATH_GPT_integral_rational_term_expansion_l2180_218059

theorem integral_rational_term_expansion :
  ∫ x in 0.0..1.0, x ^ (1/6 : ℝ) = 6/7 := by
  sorry

end NUMINAMATH_GPT_integral_rational_term_expansion_l2180_218059


namespace NUMINAMATH_GPT_max_area_rect_l2180_218069

theorem max_area_rect (x y : ℝ) (h_perimeter : 2 * x + 2 * y = 40) : 
  x * y ≤ 100 :=
by
  sorry

end NUMINAMATH_GPT_max_area_rect_l2180_218069


namespace NUMINAMATH_GPT_max_sum_pyramid_l2180_218030

theorem max_sum_pyramid (F_pentagonal : ℕ) (F_rectangular : ℕ) (E_pentagonal : ℕ) (E_rectangular : ℕ) (V_pentagonal : ℕ) (V_rectangular : ℕ)
  (original_faces : ℕ) (original_edges : ℕ) (original_vertices : ℕ)
  (H1 : original_faces = 7)
  (H2 : original_edges = 15)
  (H3 : original_vertices = 10)
  (H4 : F_pentagonal = 11)
  (H5 : E_pentagonal = 20)
  (H6 : V_pentagonal = 11)
  (H7 : F_rectangular = 10)
  (H8 : E_rectangular = 19)
  (H9 : V_rectangular = 11) :
  max (F_pentagonal + E_pentagonal + V_pentagonal) (F_rectangular + E_rectangular + V_rectangular) = 42 :=
by
  sorry

end NUMINAMATH_GPT_max_sum_pyramid_l2180_218030


namespace NUMINAMATH_GPT_total_parking_spaces_l2180_218023

-- Definitions of conditions
def caravan_space : ℕ := 2
def number_of_caravans : ℕ := 3
def spaces_left : ℕ := 24

-- Proof statement
theorem total_parking_spaces :
  (number_of_caravans * caravan_space + spaces_left) = 30 :=
by
  sorry

end NUMINAMATH_GPT_total_parking_spaces_l2180_218023


namespace NUMINAMATH_GPT_Ryan_reads_more_l2180_218001

theorem Ryan_reads_more 
  (total_pages_Ryan : ℕ)
  (days_in_week : ℕ)
  (pages_per_book_brother : ℕ)
  (books_per_day_brother : ℕ)
  (total_pages_brother : ℕ)
  (Ryan_books : ℕ)
  (Ryan_weeks : ℕ)
  (Brother_weeks : ℕ)
  (days_in_week_def : days_in_week = 7)
  (total_pages_Ryan_def : total_pages_Ryan = 2100)
  (pages_per_book_brother_def : pages_per_book_brother = 200)
  (books_per_day_brother_def : books_per_day_brother = 1)
  (Ryan_weeks_def : Ryan_weeks = 1)
  (Brother_weeks_def : Brother_weeks = 1)
  (total_pages_brother_def : total_pages_brother = pages_per_book_brother * days_in_week)
  : ((total_pages_Ryan / days_in_week) - (total_pages_brother / days_in_week) = 100) :=
by
  -- We provide the proof steps
  sorry

end NUMINAMATH_GPT_Ryan_reads_more_l2180_218001


namespace NUMINAMATH_GPT_sarah_average_speed_l2180_218003

theorem sarah_average_speed :
  ∀ (total_distance race_time : ℕ) 
    (sadie_speed sadie_time ariana_speed ariana_time : ℕ)
    (distance_sarah speed_sarah time_sarah : ℚ),
  sadie_speed = 3 → 
  sadie_time = 2 → 
  ariana_speed = 6 → 
  ariana_time = 1 / 2 → 
  race_time = 9 / 2 → 
  total_distance = 17 →
  distance_sarah = total_distance - (sadie_speed * sadie_time + ariana_speed * ariana_time) →
  time_sarah = race_time - (sadie_time + ariana_time) →
  speed_sarah = distance_sarah / time_sarah →
  speed_sarah = 4 :=
by
  intros total_distance race_time sadie_speed sadie_time ariana_speed ariana_time distance_sarah speed_sarah time_sarah
  intros sadie_speed_eq sadie_time_eq ariana_speed_eq ariana_time_eq race_time_eq total_distance_eq distance_sarah_eq time_sarah_eq speed_sarah_eq
  sorry

end NUMINAMATH_GPT_sarah_average_speed_l2180_218003


namespace NUMINAMATH_GPT_min_value_f_l2180_218054

noncomputable def f (x : ℝ) : ℝ :=
  7 * (Real.sin x)^2 + 5 * (Real.cos x)^2 + 2 * Real.sin x

theorem min_value_f : ∃ x : ℝ, f x = 4.5 :=
  sorry

end NUMINAMATH_GPT_min_value_f_l2180_218054


namespace NUMINAMATH_GPT_problem1_problem2_l2180_218091

-- Problem 1: Prove that x = ±7/2 given 4x^2 - 49 = 0
theorem problem1 (x : ℝ) : 4 * x^2 - 49 = 0 → x = 7 / 2 ∨ x = -7 / 2 := 
by
  sorry

-- Problem 2: Prove that x = 2 given (x + 1)^3 - 27 = 0
theorem problem2 (x : ℝ) : (x + 1)^3 - 27 = 0 → x = 2 := 
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l2180_218091


namespace NUMINAMATH_GPT_seq_ratio_l2180_218022

noncomputable def arith_seq (a d : ℝ) (n : ℕ) : ℝ := a + n * d

theorem seq_ratio (a d : ℝ) (h₁ : d ≠ 0) (h₂ : (arith_seq a d 2)^2 = (arith_seq a d 0) * (arith_seq a d 8)) :
  (arith_seq a d 0 + arith_seq a d 2 + arith_seq a d 4) / (arith_seq a d 1 + arith_seq a d 3 + arith_seq a d 5) = 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_seq_ratio_l2180_218022


namespace NUMINAMATH_GPT_framed_painting_ratio_l2180_218078

-- Definitions and conditions
def painting_width : ℕ := 20
def painting_height : ℕ := 30
def frame_side_width (x : ℕ) : ℕ := x
def frame_top_bottom_width (x : ℕ) : ℕ := 3 * x

-- Overall dimensions of the framed painting
def framed_painting_width (x : ℕ) : ℕ := painting_width + 2 * frame_side_width x
def framed_painting_height (x : ℕ) : ℕ := painting_height + 2 * frame_top_bottom_width x

-- Area of the painting
def painting_area : ℕ := painting_width * painting_height

-- Area of the frame
def frame_area (x : ℕ) : ℕ := framed_painting_width x * framed_painting_height x - painting_area

-- Condition that frame area equals painting area
def frame_area_condition (x : ℕ) : Prop := frame_area x = painting_area

-- Theoretical ratio of smaller to larger dimension of the framed painting
def dimension_ratio (x : ℕ) : ℚ := (framed_painting_width x : ℚ) / (framed_painting_height x)

-- The mathematical problem to prove
theorem framed_painting_ratio : ∃ x : ℕ, frame_area_condition x ∧ dimension_ratio x = (4 : ℚ) / 7 :=
by
  sorry

end NUMINAMATH_GPT_framed_painting_ratio_l2180_218078


namespace NUMINAMATH_GPT_subset_relation_l2180_218049

def M : Set ℝ := {x | x < 9}
def N : Set ℝ := {x | x^2 < 9}

theorem subset_relation : N ⊆ M := by
  sorry

end NUMINAMATH_GPT_subset_relation_l2180_218049


namespace NUMINAMATH_GPT_bucket_capacity_l2180_218084

-- Given Conditions
variable (C : ℝ)
variable (h : (2 / 3) * C = 9)

-- Goal
theorem bucket_capacity : C = 13.5 := by
  sorry

end NUMINAMATH_GPT_bucket_capacity_l2180_218084


namespace NUMINAMATH_GPT_milk_needed_for_one_batch_l2180_218067

-- Define cost of one batch given amount of milk M
def cost_of_one_batch (M : ℝ) : ℝ := 1.5 * M + 6

-- Define cost of three batches
def cost_of_three_batches (M : ℝ) : ℝ := 3 * cost_of_one_batch M

theorem milk_needed_for_one_batch : ∃ M : ℝ, cost_of_three_batches M = 63 ∧ M = 10 :=
by
  sorry

end NUMINAMATH_GPT_milk_needed_for_one_batch_l2180_218067


namespace NUMINAMATH_GPT_sum_of_last_digits_l2180_218045

theorem sum_of_last_digits (num : Nat → Nat) (a b : Nat) :
  (∀ i, 1 ≤ i ∧ i < 2000 → (num i * 10 + num (i + 1)) % 17 = 0 ∨ (num i * 10 + num (i + 1)) % 23 = 0) →
  num 1 = 3 →
  (num 2000 = a ∨ num 2000 = b) →
  a = 2 →
  b = 5 →
  a + b = 7 :=
by 
  sorry

end NUMINAMATH_GPT_sum_of_last_digits_l2180_218045


namespace NUMINAMATH_GPT_problem_statement_l2180_218037

def scientific_notation (n: ℝ) (mantissa: ℝ) (exponent: ℤ) : Prop :=
  n = mantissa * 10 ^ exponent

theorem problem_statement : scientific_notation 320000 3.2 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_problem_statement_l2180_218037


namespace NUMINAMATH_GPT_hexagon_perimeter_l2180_218072

-- Defining the side lengths of the hexagon
def side_lengths : List ℕ := [7, 10, 8, 13, 11, 9]

-- Defining the perimeter calculation
def perimeter (sides : List ℕ) : ℕ := sides.sum

-- The main theorem stating the perimeter of the given hexagon
theorem hexagon_perimeter :
  perimeter side_lengths = 58 := by
  -- Skipping proof here
  sorry

end NUMINAMATH_GPT_hexagon_perimeter_l2180_218072


namespace NUMINAMATH_GPT_teairras_pants_count_l2180_218038

-- Definitions according to the given conditions
def total_shirts := 5
def plaid_shirts := 3
def purple_pants := 5
def neither_plaid_nor_purple := 21

-- The theorem we need to prove
theorem teairras_pants_count :
  ∃ (pants : ℕ), pants = (neither_plaid_nor_purple - (total_shirts - plaid_shirts)) + purple_pants ∧ pants = 24 :=
by
  sorry

end NUMINAMATH_GPT_teairras_pants_count_l2180_218038


namespace NUMINAMATH_GPT_max_sum_abc_divisible_by_13_l2180_218036

theorem max_sum_abc_divisible_by_13 :
  ∃ (A B C : ℕ), A ≤ 9 ∧ B ≤ 9 ∧ C ≤ 9 ∧ 13 ∣ (2000 + 100 * A + 10 * B + C) ∧ (A + B + C = 26) :=
by
  sorry

end NUMINAMATH_GPT_max_sum_abc_divisible_by_13_l2180_218036


namespace NUMINAMATH_GPT_total_students_is_46_l2180_218085

-- Define the constants for the problem
def students_in_history : ℕ := 19
def students_in_math : ℕ := 14
def students_in_english : ℕ := 26
def students_in_all_three : ℕ := 3
def students_in_exactly_two : ℕ := 7

-- The total number of students as per the inclusion-exclusion principle
def total_students : ℕ :=
  students_in_history + students_in_math + students_in_english
  - students_in_exactly_two - 2 * students_in_all_three + students_in_all_three

theorem total_students_is_46 : total_students = 46 :=
  sorry

end NUMINAMATH_GPT_total_students_is_46_l2180_218085


namespace NUMINAMATH_GPT_find_x_l2180_218040

noncomputable def satisfy_equation (x : ℝ) : Prop :=
  8 / (Real.sqrt (x - 10) - 10) +
  2 / (Real.sqrt (x - 10) - 5) +
  10 / (Real.sqrt (x - 10) + 5) +
  16 / (Real.sqrt (x - 10) + 10) = 0

theorem find_x : ∃ x : ℝ, satisfy_equation x ∧ x = 60 := sorry

end NUMINAMATH_GPT_find_x_l2180_218040


namespace NUMINAMATH_GPT_farey_sequence_problem_l2180_218007

theorem farey_sequence_problem (p q : ℕ) (hpq_pos : 0 < p ∧ 0 < q) 
  (h1 : 3 / 5 < p / q) (h2 : p / q < 5 / 8) 
  (h_minimal_q : ∀ p' q', 0 < p' ∧ 0 < q' → 3 / 5 < p' / q' → p' / q' < 5 / 8 → q' ≥ q) : 
  q - p = 5 := 
sorry

end NUMINAMATH_GPT_farey_sequence_problem_l2180_218007


namespace NUMINAMATH_GPT_mass_of_fourth_metal_l2180_218006

theorem mass_of_fourth_metal 
  (m1 m2 m3 m4 : ℝ)
  (total_mass : m1 + m2 + m3 + m4 = 20)
  (h1 : m1 = 1.5 * m2)
  (h2 : m2 = 3/4 * m3)
  (h3 : m3 = 5/6 * m4) :
  m4 = 20 * (48 / 163) :=
sorry

end NUMINAMATH_GPT_mass_of_fourth_metal_l2180_218006


namespace NUMINAMATH_GPT_students_in_fifth_and_sixth_classes_l2180_218021

theorem students_in_fifth_and_sixth_classes :
  let c1 := 20
  let c2 := 25
  let c3 := 25
  let c4 := c1 / 2
  let total_students := 136
  let total_first_four_classes := c1 + c2 + c3 + c4
  let c5_and_c6 := total_students - total_first_four_classes
  c5_and_c6 = 56 :=
by
  sorry

end NUMINAMATH_GPT_students_in_fifth_and_sixth_classes_l2180_218021


namespace NUMINAMATH_GPT_effect_on_revenue_l2180_218095

variables (P Q : ℝ)

def original_revenue : ℝ := P * Q
def new_price : ℝ := 1.60 * P
def new_quantity : ℝ := 0.80 * Q
def new_revenue : ℝ := new_price P * new_quantity Q

theorem effect_on_revenue (h1 : new_price P = 1.60 * P) (h2 : new_quantity Q = 0.80 * Q) :
  new_revenue P Q - original_revenue P Q = 0.28 * original_revenue P Q :=
by
  sorry

end NUMINAMATH_GPT_effect_on_revenue_l2180_218095


namespace NUMINAMATH_GPT_age_ratio_l2180_218009

theorem age_ratio (A B : ℕ) 
  (h1 : A = 39) 
  (h2 : B = 16) 
  (h3 : (A - 5) + (B - 5) = 45) 
  (h4 : A + 5 = 44) : A / B = 39 / 16 := 
by 
  sorry

end NUMINAMATH_GPT_age_ratio_l2180_218009


namespace NUMINAMATH_GPT_parity_of_expression_l2180_218051

theorem parity_of_expression (a b c : ℤ) (h : (a + b + c) % 2 = 1) : (a^2 + b^2 - c^2 + 2*a*b) % 2 = 1 :=
by
sorry

end NUMINAMATH_GPT_parity_of_expression_l2180_218051


namespace NUMINAMATH_GPT_smallest_N_divisible_by_p_l2180_218056

theorem smallest_N_divisible_by_p (p : ℕ) (hp : Nat.Prime p)
    (N1 : ℕ) (N2 : ℕ) :
  (∃ N1 N2, 
    (N1 % p = 0 ∧ ∀ n, 1 ≤ n ∧ n < p → N1 % n = 1) ∧
    (N2 % p = 0 ∧ ∀ n, 1 ≤ n ∧ n < p → N2 % n = n - 1)
  ) :=
sorry

end NUMINAMATH_GPT_smallest_N_divisible_by_p_l2180_218056
