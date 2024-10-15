import Mathlib

namespace NUMINAMATH_GPT_max_regions_1002_1000_l1108_110857

def regions_through_point (n : ℕ) : ℕ := (n * (n + 1)) / 2 + 1

def max_regions (a b : ℕ) : ℕ := 
  let rB := regions_through_point b
  let first_line_through_A := rB + b + 1
  let remaining_lines_through_A := (a - 1) * (b + 2)
  first_line_through_A + remaining_lines_through_A

theorem max_regions_1002_1000 : max_regions 1002 1000 = 1504503 := by
  sorry

end NUMINAMATH_GPT_max_regions_1002_1000_l1108_110857


namespace NUMINAMATH_GPT_find_k_l1108_110856

   theorem find_k (m n : ℝ) (k : ℝ) (hm : m > 0) (hn : n > 0)
     (h1 : k = Real.log m / Real.log 2)
     (h2 : k = Real.log n / (Real.log 4))
     (h3 : k = Real.log (4 * m + 3 * n) / (Real.log 8)) :
     k = 2 :=
   by
     sorry
   
end NUMINAMATH_GPT_find_k_l1108_110856


namespace NUMINAMATH_GPT_regular_polygon_sides_l1108_110870

theorem regular_polygon_sides (n : ℕ) (h : ∀ i < n, (interior_angle_i : ℝ) = 150) :
  (n = 12) :=
by
  sorry

end NUMINAMATH_GPT_regular_polygon_sides_l1108_110870


namespace NUMINAMATH_GPT_geometric_sequence_common_ratio_l1108_110813

theorem geometric_sequence_common_ratio (a_n : ℕ → ℝ) (q : ℝ) 
  (h1 : a_n 3 = a_n 2 * q) 
  (h2 : a_n 2 * q - 3 * a_n 2 = 2) 
  (h3 : 5 * a_n 4 = (12 * a_n 3 + 2 * a_n 5) / 2) : 
  q = 3 := 
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_common_ratio_l1108_110813


namespace NUMINAMATH_GPT_min_a_plus_b_l1108_110887

theorem min_a_plus_b (a b : ℝ) (h : a^2 + 2 * b^2 = 6) : a + b ≥ -3 :=
sorry

end NUMINAMATH_GPT_min_a_plus_b_l1108_110887


namespace NUMINAMATH_GPT_kirin_calculations_l1108_110859

theorem kirin_calculations (calculations_per_second : ℝ) (seconds : ℝ) (h1 : calculations_per_second = 10^10) (h2 : seconds = 2022) : 
    calculations_per_second * seconds = 2.022 * 10^13 := 
by
  sorry

end NUMINAMATH_GPT_kirin_calculations_l1108_110859


namespace NUMINAMATH_GPT_log_ratio_squared_eq_nine_l1108_110800

-- Given conditions
variable (x y : ℝ) 
variable (hx_pos : x > 0) 
variable (hy_pos : y > 0)
variable (hx_neq1 : x ≠ 1) 
variable (hy_neq1 : y ≠ 1)
variable (hlog : Real.log x / Real.log 3 = Real.log 81 / Real.log y)
variable (heq : x * y = 243)

-- Prove that (\log_3(\tfrac x y))^2 = 9
theorem log_ratio_squared_eq_nine (x y : ℝ) (hx_pos : x > 0) (hy_pos : y > 0)
  (hx_neq1 : x ≠ 1) (hy_neq1 : y ≠ 1) 
  (hlog : Real.log x / Real.log 3 = Real.log 81 / Real.log y) 
  (heq : x * y = 243) : 
  ((Real.log x - Real.log y) / Real.log 3) ^ 2 = 9 :=
sorry

end NUMINAMATH_GPT_log_ratio_squared_eq_nine_l1108_110800


namespace NUMINAMATH_GPT_two_digit_factors_of_3_18_minus_1_l1108_110810

theorem two_digit_factors_of_3_18_minus_1 : ∃ n : ℕ, n = 6 ∧ 
  ∀ x, x ∈ {y : ℕ | y ∣ 3^18 - 1 ∧ y > 9 ∧ y < 100} → 
  (x = 13 ∨ x = 26 ∨ x = 52 ∨ x = 14 ∨ x = 28 ∨ x = 91) :=
by
  use 6
  sorry

end NUMINAMATH_GPT_two_digit_factors_of_3_18_minus_1_l1108_110810


namespace NUMINAMATH_GPT_perimeter_of_resulting_figure_l1108_110828

def side_length := 100
def original_square_perimeter := 4 * side_length
def rectangle_width := side_length
def rectangle_height := side_length / 2
def number_of_longer_sides_of_rectangles_touching := 4

theorem perimeter_of_resulting_figure :
  let new_perimeter := 3 * side_length + number_of_longer_sides_of_rectangles_touching * rectangle_height
  new_perimeter = 500 :=
by
  sorry

end NUMINAMATH_GPT_perimeter_of_resulting_figure_l1108_110828


namespace NUMINAMATH_GPT_total_students_at_year_end_l1108_110885

def initial_students : ℝ := 10.0
def added_students : ℝ := 4.0
def new_students : ℝ := 42.0

theorem total_students_at_year_end : initial_students + added_students + new_students = 56.0 :=
by
  sorry

end NUMINAMATH_GPT_total_students_at_year_end_l1108_110885


namespace NUMINAMATH_GPT_union_of_A_and_B_l1108_110843

def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {1, 3, 5, 7}

theorem union_of_A_and_B :
  (A ∪ B) = {1, 2, 3, 4, 5, 7} := 
by
  sorry

end NUMINAMATH_GPT_union_of_A_and_B_l1108_110843


namespace NUMINAMATH_GPT_marked_price_of_jacket_l1108_110851

variable (x : ℝ) -- Define the variable x as a real number representing the marked price.

-- Define the conditions as a Lean theorem statement
theorem marked_price_of_jacket (cost price_sold profit : ℝ) (h1 : cost = 350) (h2 : price_sold = 0.8 * x) (h3 : profit = price_sold - cost) : 
  x = 550 :=
by
  -- We would solve the proof here using provided conditions
  sorry

end NUMINAMATH_GPT_marked_price_of_jacket_l1108_110851


namespace NUMINAMATH_GPT_seventh_term_geometric_sequence_l1108_110802

theorem seventh_term_geometric_sequence :
  ∃ (a₁ a₁₀ a₇ : ℕ) (r : ℕ),
    a₁ = 6 ∧ a₁₀ = 93312 ∧
    a₁₀ = a₁ * r^9 ∧
    a₇ = a₁ * r^6 ∧
    a₇ = 279936 :=
by
  sorry

end NUMINAMATH_GPT_seventh_term_geometric_sequence_l1108_110802


namespace NUMINAMATH_GPT_adults_had_meal_l1108_110847

theorem adults_had_meal (A : ℕ) (h1 : 70 ≥ A) (h2 : ((70 - A) * 9) = (72 * 7)) : A = 14 := 
by
  sorry

end NUMINAMATH_GPT_adults_had_meal_l1108_110847


namespace NUMINAMATH_GPT_simplify_expression_l1108_110888

theorem simplify_expression : 8 * (15 / 9) * (-45 / 40) = -1 :=
  by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1108_110888


namespace NUMINAMATH_GPT_arithmetic_sequence_common_diff_l1108_110817

theorem arithmetic_sequence_common_diff (d : ℝ) (a : ℕ → ℝ) 
  (h_first_term : a 0 = 24) 
  (h_arithmetic_sequence : ∀ n, a (n + 1) = a n + d)
  (h_ninth_term_nonneg : 24 + 8 * d ≥ 0) 
  (h_tenth_term_neg : 24 + 9 * d < 0) : 
  -3 ≤ d ∧ d < -8/3 :=
by 
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_common_diff_l1108_110817


namespace NUMINAMATH_GPT_Seulgi_second_round_need_l1108_110803

def Hohyeon_first_round := 23
def Hohyeon_second_round := 28
def Hyunjeong_first_round := 32
def Hyunjeong_second_round := 17
def Seulgi_first_round := 27

def Hohyeon_total := Hohyeon_first_round + Hohyeon_second_round
def Hyunjeong_total := Hyunjeong_first_round + Hyunjeong_second_round

def required_total_for_Seulgi := Hohyeon_total + 1

theorem Seulgi_second_round_need (Seulgi_second_round: ℕ) :
  Seulgi_first_round + Seulgi_second_round ≥ required_total_for_Seulgi → Seulgi_second_round ≥ 25 :=
by
  sorry

end NUMINAMATH_GPT_Seulgi_second_round_need_l1108_110803


namespace NUMINAMATH_GPT_line_no_intersect_parabola_range_l1108_110823

def parabola_eq (x : ℝ) : ℝ := x^2 + 4

def line_eq (m x : ℝ) : ℝ := m * (x - 10) + 6

theorem line_no_intersect_parabola_range (r s m : ℝ) :
  (m^2 - 40 * m + 8 = 0) →
  r < s →
  (∀ x, parabola_eq x ≠ line_eq m x) →
  r + s = 40 :=
by
  sorry

end NUMINAMATH_GPT_line_no_intersect_parabola_range_l1108_110823


namespace NUMINAMATH_GPT_calculate_myOp_l1108_110806

-- Define the operation
def myOp (x y : ℝ) : ℝ := x^3 - y

-- Given condition for h as a real number
variable (h : ℝ)

-- The theorem we need to prove
theorem calculate_myOp : myOp (2 * h) (myOp (2 * h) (2 * h)) = 2 * h := by
  sorry

end NUMINAMATH_GPT_calculate_myOp_l1108_110806


namespace NUMINAMATH_GPT_math_equivalence_l1108_110866

theorem math_equivalence (a b c : ℕ) (ha : 0 < a ∧ a < 12) (hb : 0 < b ∧ b < 12) (hc : 0 < c ∧ c < 12) (hbc : b + c = 12) :
  (12 * a + b) * (12 * a + c) = 144 * a * (a + 1) + b * c := 
by 
  sorry

end NUMINAMATH_GPT_math_equivalence_l1108_110866


namespace NUMINAMATH_GPT_Tori_current_height_l1108_110821

   -- Define the original height and the height she grew
   def Tori_original_height : Real := 4.4
   def Tori_growth : Real := 2.86

   -- Prove that Tori's current height is 7.26 feet
   theorem Tori_current_height : Tori_original_height + Tori_growth = 7.26 := by
     sorry
   
end NUMINAMATH_GPT_Tori_current_height_l1108_110821


namespace NUMINAMATH_GPT_whale_consumption_l1108_110882

-- Define the conditions
def first_hour_consumption (x : ℕ) := x
def second_hour_consumption (x : ℕ) := x + 3
def third_hour_consumption (x : ℕ) := x + 6
def fourth_hour_consumption (x : ℕ) := x + 9
def fifth_hour_consumption (x : ℕ) := x + 12
def sixth_hour_consumption (x : ℕ) := x + 15
def seventh_hour_consumption (x : ℕ) := x + 18
def eighth_hour_consumption (x : ℕ) := x + 21
def ninth_hour_consumption (x : ℕ) := x + 24

def total_consumed (x : ℕ) := 
  first_hour_consumption x + 
  second_hour_consumption x + 
  third_hour_consumption x + 
  fourth_hour_consumption x + 
  fifth_hour_consumption x + 
  sixth_hour_consumption x + 
  seventh_hour_consumption x + 
  eighth_hour_consumption x + 
  ninth_hour_consumption x

-- Prove that the total sum consumed equals 540
theorem whale_consumption : ∃ x : ℕ, total_consumed x = 540 ∧ sixth_hour_consumption x = 63 :=
by
  sorry

end NUMINAMATH_GPT_whale_consumption_l1108_110882


namespace NUMINAMATH_GPT_floor_plus_r_eq_10_3_implies_r_eq_5_3_l1108_110836

noncomputable def floor (x : ℝ) : ℤ := sorry -- Assuming the function exists

theorem floor_plus_r_eq_10_3_implies_r_eq_5_3 (r : ℝ) 
  (h : floor r + r = 10.3) : r = 5.3 :=
sorry

end NUMINAMATH_GPT_floor_plus_r_eq_10_3_implies_r_eq_5_3_l1108_110836


namespace NUMINAMATH_GPT_part_i_part_ii_part_iii_l1108_110873

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x

noncomputable def g (x : ℝ) : ℝ := f (x + Real.pi / 4) + f (x + 3 * Real.pi / 4)

theorem part_i : f (Real.pi / 2) = 1 :=
sorry

theorem part_ii : ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = 2 * Real.pi :=
sorry

theorem part_iii : ∃ x, g x = -2 :=
sorry

end NUMINAMATH_GPT_part_i_part_ii_part_iii_l1108_110873


namespace NUMINAMATH_GPT_interval_length_l1108_110874

theorem interval_length (c d : ℝ) (h : ∃ x : ℝ, c ≤ 3 * x + 4 ∧ 3 * x + 4 ≤ d)
  (length : (d - 4) / 3 - (c - 4) / 3 = 15) : d - c = 45 :=
by
  sorry

end NUMINAMATH_GPT_interval_length_l1108_110874


namespace NUMINAMATH_GPT_tetrahedron_cut_off_vertices_l1108_110898

theorem tetrahedron_cut_off_vertices :
  ∀ (V E : ℕ) (cut_effect : ℕ → ℕ),
    -- Initial conditions
    V = 4 → E = 6 →
    -- Effect of each cut (cutting one vertex introduces 3 new edges)
    (∀ v, v ≤ V → cut_effect v = 3 * v) →
    -- Prove the number of edges in the new figure
    (E + cut_effect V) = 18 :=
by
  intros V E cut_effect hV hE hcut
  sorry

end NUMINAMATH_GPT_tetrahedron_cut_off_vertices_l1108_110898


namespace NUMINAMATH_GPT_points_on_circle_l1108_110831

theorem points_on_circle (t : ℝ) (ht : t ≠ 0) :
  let x := (t + 1) / t ^ 2
  let y := (t - 1) / t ^ 2
  (x - 2)^2 + (y - 2)^2 = 4 :=
by
  let x := (t + 1) / t ^ 2
  let y := (t - 1) / t ^ 2
  sorry

end NUMINAMATH_GPT_points_on_circle_l1108_110831


namespace NUMINAMATH_GPT_time_to_write_numbers_in_minutes_l1108_110829

theorem time_to_write_numbers_in_minutes : 
  (1 * 5 + 2 * (99 - 10 + 1) + 3 * (105 - 100 + 1)) / 60 = 4 := 
  by
  -- Calculation steps would go here
  sorry

end NUMINAMATH_GPT_time_to_write_numbers_in_minutes_l1108_110829


namespace NUMINAMATH_GPT_triangle_angle_equality_l1108_110895

theorem triangle_angle_equality
  (α β γ α₁ β₁ γ₁ : ℝ)
  (hABC : α + β + γ = 180)
  (hA₁B₁C₁ : α₁ + β₁ + γ₁ = 180)
  (angle_relation : (α = α₁ ∨ α + α₁ = 180) ∧ (β = β₁ ∨ β + β₁ = 180) ∧ (γ = γ₁ ∨ γ + γ₁ = 180)) :
  α = α₁ ∧ β = β₁ ∧ γ = γ₁ :=
by {
  sorry
}

end NUMINAMATH_GPT_triangle_angle_equality_l1108_110895


namespace NUMINAMATH_GPT_range_of_m_l1108_110878

theorem range_of_m (m : ℝ) 
  (h : ∀ x : ℝ, 0 < x → m * x^2 + 2 * x + m ≤ 0) : m ≤ -1 :=
sorry

end NUMINAMATH_GPT_range_of_m_l1108_110878


namespace NUMINAMATH_GPT_find_winner_votes_l1108_110827

-- Define the conditions
variables (V : ℝ) (winner_votes second_votes : ℝ)
def election_conditions :=
  winner_votes = 0.468 * V ∧
  second_votes = 0.326 * V ∧
  winner_votes - second_votes = 752

-- State the theorem
theorem find_winner_votes (h : election_conditions V winner_votes second_votes) :
  winner_votes = 2479 :=
sorry

end NUMINAMATH_GPT_find_winner_votes_l1108_110827


namespace NUMINAMATH_GPT_gcd_g_x_l1108_110807

noncomputable def g (x : ℕ) : ℕ :=
  (3 * x + 5) * (7 * x + 2) * (13 * x + 7) * (2 * x + 10)

theorem gcd_g_x (x : ℕ) (h : x % 19845 = 0) : Nat.gcd (g x) x = 700 :=
  sorry

end NUMINAMATH_GPT_gcd_g_x_l1108_110807


namespace NUMINAMATH_GPT_jonah_added_yellow_raisins_l1108_110854

variable (y : ℝ)

theorem jonah_added_yellow_raisins (h : y + 0.4 = 0.7) : y = 0.3 := by
  sorry

end NUMINAMATH_GPT_jonah_added_yellow_raisins_l1108_110854


namespace NUMINAMATH_GPT_power_of_product_l1108_110845

variable (a b : ℝ) (m : ℕ)
theorem power_of_product (h : 0 < m) : (a * b)^m = a^m * b^m :=
sorry

end NUMINAMATH_GPT_power_of_product_l1108_110845


namespace NUMINAMATH_GPT_runners_meet_again_l1108_110853

-- Definitions based on the problem conditions
def track_length : ℝ := 500 
def speed_runner1 : ℝ := 4.4
def speed_runner2 : ℝ := 4.8
def speed_runner3 : ℝ := 5.0

-- The time at which runners meet again at the starting point
def time_when_runners_meet : ℝ := 2500

theorem runners_meet_again :
  ∀ t : ℝ, t = time_when_runners_meet → 
  (∀ n1 n2 n3 : ℤ, 
    ∃ k : ℤ, 
    speed_runner1 * t = n1 * track_length ∧ 
    speed_runner2 * t = n2 * track_length ∧ 
    speed_runner3 * t = n3 * track_length) :=
by 
  sorry

end NUMINAMATH_GPT_runners_meet_again_l1108_110853


namespace NUMINAMATH_GPT_homework_duration_decrease_l1108_110872

variable (a b x : ℝ)

theorem homework_duration_decrease (h: a * (1 - x)^2 = b) :
  a * (1 - x)^2 = b := 
by
  sorry

end NUMINAMATH_GPT_homework_duration_decrease_l1108_110872


namespace NUMINAMATH_GPT_odd_positive_93rd_l1108_110886

theorem odd_positive_93rd : 
  (2 * 93 - 1) = 185 := 
by sorry

end NUMINAMATH_GPT_odd_positive_93rd_l1108_110886


namespace NUMINAMATH_GPT_girls_divisible_by_nine_l1108_110850

def total_students (m c d u : ℕ) : ℕ := 1000 * m + 100 * c + 10 * d + u
def number_of_boys (m c d u : ℕ) : ℕ := m + c + d + u
def number_of_girls (m c d u : ℕ) : ℕ := total_students m c d u - number_of_boys m c d u 

theorem girls_divisible_by_nine (m c d u : ℕ) : 
  number_of_girls m c d u % 9 = 0 := 
by
    sorry

end NUMINAMATH_GPT_girls_divisible_by_nine_l1108_110850


namespace NUMINAMATH_GPT_tree_growth_period_l1108_110824

theorem tree_growth_period (initial height growth_rate : ℕ) (H4 final_height years : ℕ) 
  (h_init : initial_height = 4) 
  (h_growth_rate : growth_rate = 1) 
  (h_H4 : H4 = initial_height + 4 * growth_rate)
  (h_final_height : final_height = H4 + H4 / 4) 
  (h_years : years = (final_height - initial_height) / growth_rate) :
  years = 6 :=
by
  sorry

end NUMINAMATH_GPT_tree_growth_period_l1108_110824


namespace NUMINAMATH_GPT_value_of_x_squared_plus_y_squared_l1108_110839

theorem value_of_x_squared_plus_y_squared (x y : ℝ) (h1 : (x + y)^2 = 4) (h2 : x * y = -6) : x^2 + y^2 = 16 :=
by
  sorry

end NUMINAMATH_GPT_value_of_x_squared_plus_y_squared_l1108_110839


namespace NUMINAMATH_GPT_no_statement_implies_neg_p_or_q_l1108_110812

def statement1 (p q : Prop) : Prop := p ∨ q
def statement2 (p q : Prop) : Prop := p ∨ ¬ q
def statement3 (p q : Prop) : Prop := ¬ p ∨ q
def statement4 (p q : Prop) : Prop := ¬ p ∧ q
def neg_p_or_q (p q : Prop) : Prop := ¬ (p ∨ q)

theorem no_statement_implies_neg_p_or_q (p q : Prop) :
  ¬ (statement1 p q → neg_p_or_q p q) ∧
  ¬ (statement2 p q → neg_p_or_q p q) ∧
  ¬ (statement3 p q → neg_p_or_q p q) ∧
  ¬ (statement4 p q → neg_p_or_q p q)
:= by
  sorry

end NUMINAMATH_GPT_no_statement_implies_neg_p_or_q_l1108_110812


namespace NUMINAMATH_GPT_geometric_sequence_product_l1108_110811

variable {α : Type*} [LinearOrderedField α]

theorem geometric_sequence_product :
  ∀ (a r : α), (a^3 * r^6 = 3) → (a^3 * r^15 = 24) → (a^3 * r^24 = 192) :=
by
  intros a r h1 h2
  sorry

end NUMINAMATH_GPT_geometric_sequence_product_l1108_110811


namespace NUMINAMATH_GPT_time_per_lice_check_l1108_110849

-- Define the number of students in each grade
def kindergartners := 26
def first_graders := 19
def second_graders := 20
def third_graders := 25

-- Define the total number of students
def total_students := kindergartners + first_graders + second_graders + third_graders

-- Define the total time in minutes
def hours := 3
def minutes_per_hour := 60
def total_minutes := hours * minutes_per_hour

-- Define the correct answer for time per check
def time_per_check := total_minutes / total_students

-- Prove that the time for each check is 2 minutes
theorem time_per_lice_check : time_per_check = 2 := 
by
  sorry

end NUMINAMATH_GPT_time_per_lice_check_l1108_110849


namespace NUMINAMATH_GPT_find_b_l1108_110855

noncomputable def f (b x : ℝ) : ℝ :=
if x < 1 then 2 * x - b else 2 ^ x

theorem find_b (b : ℝ) (h : f b (f b (1 / 2)) = 4) : b = -1 :=
sorry

end NUMINAMATH_GPT_find_b_l1108_110855


namespace NUMINAMATH_GPT_find_x_l1108_110819

theorem find_x
  (a b x : ℝ)
  (h1 : a * (x + 2) + b * (x + 2) = 60)
  (h2 : a + b = 12) :
  x = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1108_110819


namespace NUMINAMATH_GPT_students_height_order_valid_after_rearrangement_l1108_110826
open List

variable {n : ℕ} -- number of students in each row
variable (a b : Fin n → ℝ) -- heights of students in each row

/-- Prove Gábor's observation remains valid after rearrangement: 
    each student in the back row is taller than the student in front of them.
    Given:
    - ∀ i, b i < a i (initial condition)
    - ∀ i < j, a i ≤ a j (rearrangement condition)
    Prove:
    - ∀ i, b i < a i (remains valid after rearrangement)
-/
theorem students_height_order_valid_after_rearrangement
  (h₁ : ∀ i : Fin n, b i < a i)
  (h₂ : ∀ (i j : Fin n), i < j → a i ≤ a j) :
  ∀ i : Fin n, b i < a i :=
by sorry

end NUMINAMATH_GPT_students_height_order_valid_after_rearrangement_l1108_110826


namespace NUMINAMATH_GPT_find_m_l1108_110897

theorem find_m (m : ℝ) (α : ℝ) (h_cos : Real.cos α = -3/5) (h_p : ((Real.cos α = m / (Real.sqrt (m^2 + 4^2)))) ∧ (Real.cos α < 0) ∧ (m < 0)) :

  m = -3 :=
by 
  sorry

end NUMINAMATH_GPT_find_m_l1108_110897


namespace NUMINAMATH_GPT_bmws_sold_l1108_110877

-- Definitions stated by the problem:
def total_cars : ℕ := 300
def percentage_mercedes : ℝ := 0.20
def percentage_toyota : ℝ := 0.25
def percentage_nissan : ℝ := 0.10
def percentage_bmws : ℝ := 1 - (percentage_mercedes + percentage_toyota + percentage_nissan)

-- Statement to prove:
theorem bmws_sold : (total_cars : ℝ) * percentage_bmws = 135 := by
  sorry

end NUMINAMATH_GPT_bmws_sold_l1108_110877


namespace NUMINAMATH_GPT_find_x_l1108_110808

theorem find_x (x : ℝ) : x - (502 / 100.4) = 5015 → x = 5020 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1108_110808


namespace NUMINAMATH_GPT_linear_function_no_second_quadrant_l1108_110891

theorem linear_function_no_second_quadrant (k : ℝ) :
  (∀ x : ℝ, (y : ℝ) → y = k * x - k + 3 → ¬(x < 0 ∧ y > 0)) ↔ k ≥ 3 :=
sorry

end NUMINAMATH_GPT_linear_function_no_second_quadrant_l1108_110891


namespace NUMINAMATH_GPT_work_rate_proof_l1108_110842

def combined_rate (a b c : ℚ) : ℚ := a + b + c

def inv (x : ℚ) : ℚ := 1 / x

theorem work_rate_proof (A B C : ℚ) (h₁ : A + B = 1/15) (h₂ : C = 1/10) :
  inv (combined_rate A B C) = 6 :=
by
  sorry

end NUMINAMATH_GPT_work_rate_proof_l1108_110842


namespace NUMINAMATH_GPT_count_valid_subsets_l1108_110892

open Set

theorem count_valid_subsets :
  ∀ (A : Set ℕ), (A ⊆ {1, 2, 3, 4, 5, 6, 7}) → 
  (∀ (a : ℕ), a ∈ A → (8 - a) ∈ A) → A ≠ ∅ → 
  ∃! (n : ℕ), n = 15 :=
  by
    sorry

end NUMINAMATH_GPT_count_valid_subsets_l1108_110892


namespace NUMINAMATH_GPT_part_I_part_II_l1108_110838

noncomputable def f_I (x : ℝ) : ℝ := abs (3*x - 1) + abs (x + 3)

theorem part_I :
  ∀ x : ℝ, f_I x ≥ 4 ↔ x ≤ 0 ∨ x ≥ 1/2 :=
by sorry

noncomputable def f_II (x b c : ℝ) : ℝ := abs (x - b) + abs (x + c)

theorem part_II :
  ∀ b c : ℝ, b > 0 → c > 0 → b + c = 1 → 
  (∀ x : ℝ, f_II x b c ≥ 1) → (1 / b + 1 / c = 4) :=
by sorry

end NUMINAMATH_GPT_part_I_part_II_l1108_110838


namespace NUMINAMATH_GPT_range_of_a_l1108_110858

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, (a-1)*x^2 + a*x + 1 ≥ 0) : a ≥ 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_range_of_a_l1108_110858


namespace NUMINAMATH_GPT_jana_walk_distance_l1108_110884

theorem jana_walk_distance :
  (1 / 20 * 15 : ℝ) = 0.8 :=
by sorry

end NUMINAMATH_GPT_jana_walk_distance_l1108_110884


namespace NUMINAMATH_GPT_solve_absolute_value_inequality_l1108_110867

theorem solve_absolute_value_inequality (x : ℝ) :
  3 ≤ |x + 3| ∧ |x + 3| ≤ 7 ↔ (-10 ≤ x ∧ x ≤ -6) ∨ (0 ≤ x ∧ x ≤ 4) :=
by
  sorry

end NUMINAMATH_GPT_solve_absolute_value_inequality_l1108_110867


namespace NUMINAMATH_GPT_optimal_route_l1108_110899

-- Define the probabilities of no traffic jam on each road segment.
def P_AC : ℚ := 9 / 10
def P_CD : ℚ := 14 / 15
def P_DB : ℚ := 5 / 6
def P_CF : ℚ := 9 / 10
def P_FB : ℚ := 15 / 16
def P_AE : ℚ := 9 / 10
def P_EF : ℚ := 9 / 10
def P_FB2 : ℚ := 19 / 20  -- Alias for repeated probability

-- Define the probability of encountering a traffic jam on a route
def prob_traffic_jam (p_no_jam : ℚ) : ℚ := 1 - p_no_jam

-- Define the probabilities of encountering a traffic jam along each route.
def P_ACDB_jam : ℚ := prob_traffic_jam (P_AC * P_CD * P_DB)
def P_ACFB_jam : ℚ := prob_traffic_jam (P_AC * P_CF * P_FB)
def P_AEFB_jam : ℚ := prob_traffic_jam (P_AE * P_EF * P_FB2)

-- State the theorem to prove the optimal route
theorem optimal_route : P_ACDB_jam < P_ACFB_jam ∧ P_ACDB_jam < P_AEFB_jam :=
by { sorry }

end NUMINAMATH_GPT_optimal_route_l1108_110899


namespace NUMINAMATH_GPT_need_to_work_24_hours_per_week_l1108_110893

-- Definitions
def original_hours_per_week := 20
def total_weeks := 12
def target_income := 3000

def missed_weeks := 2
def remaining_weeks := total_weeks - missed_weeks

-- Calculation
def new_hours_per_week := (original_hours_per_week * total_weeks) / remaining_weeks

-- Statement of the theorem
theorem need_to_work_24_hours_per_week : new_hours_per_week = 24 := 
by 
  -- Adding sorry to skip the proof, focusing on the statement.
  sorry

end NUMINAMATH_GPT_need_to_work_24_hours_per_week_l1108_110893


namespace NUMINAMATH_GPT_total_hair_cut_l1108_110841

-- Definitions from conditions
def first_cut : ℝ := 0.375
def second_cut : ℝ := 0.5

-- The theorem stating the math problem
theorem total_hair_cut : first_cut + second_cut = 0.875 := by
  sorry

end NUMINAMATH_GPT_total_hair_cut_l1108_110841


namespace NUMINAMATH_GPT_roots_condition_l1108_110889

theorem roots_condition (m : ℝ) (f : ℝ → ℝ) (x1 x2 : ℝ) (h_f : ∀ x, f x = x^2 + 2*(m - 1)*x - 5*m - 2) 
  (h_roots : ∃ x1 x2, x1 < 1 ∧ 1 < x2 ∧ f x1 = 0 ∧ f x2 = 0) : 
  m > 1 := 
by
  sorry

end NUMINAMATH_GPT_roots_condition_l1108_110889


namespace NUMINAMATH_GPT_intersection_P_compl_M_l1108_110832

-- Define universal set U
def U : Set ℤ := Set.univ

-- Define set M
def M : Set ℤ := {1, 2}

-- Define set P
def P : Set ℤ := {-2, -1, 0, 1, 2}

-- Define the complement of M in U
def M_compl : Set ℤ := { x | x ∉ M }

-- Define the intersection of P and the complement of M
def P_inter_M_compl : Set ℤ := P ∩ M_compl

-- The theorem we want to prove
theorem intersection_P_compl_M : P_inter_M_compl = {-2, -1, 0} := 
by {
  sorry
}

end NUMINAMATH_GPT_intersection_P_compl_M_l1108_110832


namespace NUMINAMATH_GPT_intersection_complement_eq_l1108_110833

-- Define the universal set U, and sets A and B
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {2, 3, 4}
def B : Set ℕ := {4, 5}

-- Define the complement of B in U
def complement_B_in_U : Set ℕ := { x ∈ U | x ∉ B }

-- The main theorem statement stating the required equality
theorem intersection_complement_eq : A ∩ complement_B_in_U = {2, 3} := by
  sorry

end NUMINAMATH_GPT_intersection_complement_eq_l1108_110833


namespace NUMINAMATH_GPT_find_m_plus_t_l1108_110869

-- Define the system of equations represented by the augmented matrix
def equation1 (m t : ℝ) : Prop := 3 * m - t = 22
def equation2 (t : ℝ) : Prop := t = 2

-- State the main theorem with the given conditions and the goal
theorem find_m_plus_t (m t : ℝ) (h1 : equation1 m t) (h2 : equation2 t) : m + t = 10 := 
by
  sorry

end NUMINAMATH_GPT_find_m_plus_t_l1108_110869


namespace NUMINAMATH_GPT_octopus_shoes_needed_l1108_110861

-- Defining the basic context: number of legs and current shod legs
def num_legs : ℕ := 8

-- Conditions based on the number of already shod legs for each member
def father_shod_legs : ℕ := num_legs / 2       -- Father-octopus has half of his legs shod
def mother_shod_legs : ℕ := 3                  -- Mother-octopus has 3 legs shod
def son_shod_legs : ℕ := 6                     -- Each son-octopus has 6 legs shod
def num_sons : ℕ := 2                          -- There are 2 sons

-- Calculate unshod legs for each 
def father_unshod_legs : ℕ := num_legs - father_shod_legs
def mother_unshod_legs : ℕ := num_legs - mother_shod_legs
def son_unshod_legs : ℕ := num_legs - son_shod_legs

-- Aggregate the total shoes needed based on unshod legs
def total_shoes_needed : ℕ :=
  father_unshod_legs + 
  mother_unshod_legs + 
  (son_unshod_legs * num_sons)

-- The theorem to prove
theorem octopus_shoes_needed : total_shoes_needed = 13 := 
  by 
    sorry

end NUMINAMATH_GPT_octopus_shoes_needed_l1108_110861


namespace NUMINAMATH_GPT_will_total_clothes_l1108_110894

theorem will_total_clothes (n1 n2 n3 : ℕ) (h1 : n1 = 32) (h2 : n2 = 9) (h3 : n3 = 3) : n1 + n2 * n3 = 59 := 
by
  sorry

end NUMINAMATH_GPT_will_total_clothes_l1108_110894


namespace NUMINAMATH_GPT_each_squirrel_needs_more_acorns_l1108_110818

noncomputable def acorns_needed : ℕ := 300
noncomputable def total_acorns_collected : ℕ := 4500
noncomputable def number_of_squirrels : ℕ := 20

theorem each_squirrel_needs_more_acorns : 
  (acorns_needed - total_acorns_collected / number_of_squirrels) = 75 :=
by
  sorry

end NUMINAMATH_GPT_each_squirrel_needs_more_acorns_l1108_110818


namespace NUMINAMATH_GPT_zoe_total_songs_l1108_110848

def total_songs (country_albums pop_albums songs_per_country_album songs_per_pop_album : ℕ) : ℕ :=
  country_albums * songs_per_country_album + pop_albums * songs_per_pop_album

theorem zoe_total_songs :
  total_songs 4 7 5 6 = 62 :=
by
  sorry

end NUMINAMATH_GPT_zoe_total_songs_l1108_110848


namespace NUMINAMATH_GPT_number_of_girls_l1108_110883

-- Define the number of boys and girls as natural numbers
variable (B G : ℕ)

-- First condition: The number of girls is 458 more than the number of boys
axiom h1 : G = B + 458

-- Second condition: The total number of pupils is 926
axiom h2 : G + B = 926

-- The theorem to be proved: The number of girls is 692
theorem number_of_girls : G = 692 := by
  sorry

end NUMINAMATH_GPT_number_of_girls_l1108_110883


namespace NUMINAMATH_GPT_emma_final_balance_correct_l1108_110820

def emma_initial_savings : ℕ := 230
def withdrawal_amount : ℕ := 60
def deposit_amount : ℕ := 2 * withdrawal_amount
def final_amount_in_account : ℕ := emma_initial_savings - withdrawal_amount + deposit_amount

theorem emma_final_balance_correct : final_amount_in_account = 290 := by
  sorry

end NUMINAMATH_GPT_emma_final_balance_correct_l1108_110820


namespace NUMINAMATH_GPT_delta_maximum_success_ratio_l1108_110852

theorem delta_maximum_success_ratio (x y z w : ℕ) (h1 : 0 < x ∧ x * 5 < y * 3)
    (h2 : 0 < z ∧ z * 5 < w * 3) (h3 : y + w = 600) :
    (x + z) / 600 ≤ 359 / 600 :=
by
  sorry

end NUMINAMATH_GPT_delta_maximum_success_ratio_l1108_110852


namespace NUMINAMATH_GPT_euler_totient_bound_l1108_110864

theorem euler_totient_bound (n k : ℕ) (h1 : 0 < n) (h2 : 0 < k) (h3 : (Nat.totient^[k]) n = 1) :
  n ≤ 3^k :=
sorry

end NUMINAMATH_GPT_euler_totient_bound_l1108_110864


namespace NUMINAMATH_GPT_solution_of_system_l1108_110837

theorem solution_of_system :
  ∃ x y : ℝ, (x^4 + y^4 = 17) ∧ (x + y = 3) ∧ ((x = 1 ∧ y = 2) ∨ (x = 2 ∧ y = 1)) :=
by
  sorry

end NUMINAMATH_GPT_solution_of_system_l1108_110837


namespace NUMINAMATH_GPT_solve_for_a_l1108_110860

open Complex

theorem solve_for_a (a : ℝ) (h : ∃ x : ℝ, (2 * Complex.I - (a * Complex.I) / (1 - Complex.I) = x)) : a = 4 := 
sorry

end NUMINAMATH_GPT_solve_for_a_l1108_110860


namespace NUMINAMATH_GPT_slices_per_pie_l1108_110834

variable (S : ℕ) -- Let S be the number of slices per pie

theorem slices_per_pie (h1 : 5 * S * 9 = 180) : S = 4 := by
  sorry

end NUMINAMATH_GPT_slices_per_pie_l1108_110834


namespace NUMINAMATH_GPT_infinite_perfect_squares_in_arithmetic_sequence_l1108_110881

theorem infinite_perfect_squares_in_arithmetic_sequence 
  (a d : ℕ) 
  (h_exists_perfect_square : ∃ (n₀ k : ℕ), a + n₀ * d = k^2) 
  : ∃ (S : ℕ → ℕ), (∀ n, ∃ t, S n = a + t * d ∧ ∃ k, S n = k^2) ∧ (∀ m n, S m = S n → m = n) :=
sorry

end NUMINAMATH_GPT_infinite_perfect_squares_in_arithmetic_sequence_l1108_110881


namespace NUMINAMATH_GPT_square_area_l1108_110814

theorem square_area (x : ℚ) (h : 3 * x - 12 = 15 - 2 * x) : (3 * (27 / 5) - 12)^2 = 441 / 25 :=
by
  sorry

end NUMINAMATH_GPT_square_area_l1108_110814


namespace NUMINAMATH_GPT_equation_has_two_solutions_l1108_110801

theorem equation_has_two_solutions (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : 
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a^x1 = x1^2 - 2*x1 - a ∧ a^x2 = x2^2 - 2*x2 - a :=
sorry

end NUMINAMATH_GPT_equation_has_two_solutions_l1108_110801


namespace NUMINAMATH_GPT_river_lengths_l1108_110844

theorem river_lengths (x : ℝ) (dnieper don : ℝ)
  (h1 : dnieper = (5 / (19 / 3)) * x)
  (h2 : don = (6.5 / 9.5) * x)
  (h3 : dnieper - don = 300) :
  x = 2850 ∧ dnieper = 2250 ∧ don = 1950 :=
by
  sorry

end NUMINAMATH_GPT_river_lengths_l1108_110844


namespace NUMINAMATH_GPT_candy_cost_l1108_110835

-- Definitions and assumptions from problem conditions
def cents_per_page := 1
def pages_per_book := 150
def books_read := 12
def leftover_cents := 300  -- $3 in cents

-- Total pages read
def total_pages_read := pages_per_book * books_read

-- Total earnings in cents
def total_cents_earned := total_pages_read * cents_per_page

-- Cost of the candy in cents
def candy_cost_cents := total_cents_earned - leftover_cents

-- Theorem statement
theorem candy_cost : candy_cost_cents = 1500 := 
  by 
    -- proof goes here
    sorry

end NUMINAMATH_GPT_candy_cost_l1108_110835


namespace NUMINAMATH_GPT_range_of_a_l1108_110846

def quadratic_function (a b c x : ℝ) : ℝ :=
  a * x^2 + b * x + c

noncomputable def roots (a : ℝ) : (ℝ × ℝ) :=
  (1, 3)

noncomputable def f_max (a : ℝ) :=
  -a

theorem range_of_a (a b c : ℝ) 
  (h1 : ∀ x, quadratic_function a b c x < 0 ↔ (x < 1 ∨ 3 < x))
  (h2 : f_max a < 2) : 
  -2 < a ∧ a < 0 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1108_110846


namespace NUMINAMATH_GPT_inequality_proof_l1108_110876

theorem inequality_proof 
  {a b c : ℝ}
  (ha : 0 ≤ a)
  (hb : 0 ≤ b)
  (hc : 0 ≤ c)
  (h1 : a^2 ≤ b^2 + c^2)
  (h2 : b^2 ≤ c^2 + a^2)
  (h3 : c^2 ≤ a^2 + b^2) :
  (a + b + c) * (a^2 + b^2 + c^2) * (a^3 + b^3 + c^3) ≥ 4 * (a^6 + b^6 + c^6) :=
sorry

end NUMINAMATH_GPT_inequality_proof_l1108_110876


namespace NUMINAMATH_GPT_correct_option_B_l1108_110816

theorem correct_option_B (a : ℤ) : (2 * a) ^ 3 = 8 * a ^ 3 :=
by
  sorry

end NUMINAMATH_GPT_correct_option_B_l1108_110816


namespace NUMINAMATH_GPT_mom_age_when_Jayson_born_l1108_110880

theorem mom_age_when_Jayson_born
  (Jayson_age : ℕ)
  (Dad_age : ℕ)
  (Mom_age : ℕ)
  (H1 : Jayson_age = 10)
  (H2 : Dad_age = 4 * Jayson_age)
  (H3 : Mom_age = Dad_age - 2) :
  Mom_age - Jayson_age = 28 := by
  sorry

end NUMINAMATH_GPT_mom_age_when_Jayson_born_l1108_110880


namespace NUMINAMATH_GPT_grey_area_of_first_grid_is_16_grey_area_of_second_grid_is_15_white_area_of_third_grid_is_5_l1108_110865

theorem grey_area_of_first_grid_is_16 (side_length : ℝ := 1) :
  let area_triangle (base height : ℝ) := 0.5 * base * height
  let area_rectangle (length width : ℝ) := length * width
  let grey_area := area_triangle 3 side_length 
                    + area_triangle 4 side_length 
                    + area_rectangle 6 side_length 
                    + area_triangle 2 side_length 
                    + area_triangle 2 side_length 
                    + area_rectangle 2 side_length 
                    + area_triangle 2 side_length 
                    + area_triangle 3 side_length
  grey_area = 16 := by
  sorry

theorem grey_area_of_second_grid_is_15 (side_length : ℝ := 1) :
  let area_triangle (base height : ℝ) := 0.5 * base * height
  let area_rectangle (length width : ℝ) := length * width
  let grey_area := area_triangle 4 side_length 
                    + area_rectangle 2 side_length
                    + area_triangle 6 side_length 
                    + area_rectangle 2 side_length 
                    + area_triangle 2 side_length 
                    + area_triangle 2 side_length 
                    + area_rectangle 4 side_length
  grey_area = 15 := by
  sorry

theorem white_area_of_third_grid_is_5 (total_rectangle_area dark_grey_area : ℝ) (grey_area1 grey_area2 : ℝ) :
    total_rectangle_area = 32 ∧ dark_grey_area = 4 ∧ grey_area1 = 16 ∧ grey_area2 = 15 →
    let total_grey_area_recounted := grey_area1 + grey_area2 - dark_grey_area
    let white_area := total_rectangle_area - total_grey_area_recounted
    white_area = 5 := by
  sorry

end NUMINAMATH_GPT_grey_area_of_first_grid_is_16_grey_area_of_second_grid_is_15_white_area_of_third_grid_is_5_l1108_110865


namespace NUMINAMATH_GPT_number_of_children_l1108_110805

-- Definitions based on conditions
def numDogs : ℕ := 2
def numCats : ℕ := 1
def numLegsTotal : ℕ := 22
def numLegsDog : ℕ := 4
def numLegsCat : ℕ := 4
def numLegsHuman : ℕ := 2

-- Main theorem proving the number of children
theorem number_of_children :
  let totalPetLegs := (numDogs * numLegsDog) + (numCats * numLegsCat)
  let totalLegsAccounted := totalPetLegs + numLegsHuman
  let numLegsRemaining := numLegsTotal - totalLegsAccounted
  let numChildren := numLegsRemaining / numLegsHuman
  numChildren = 4 :=
by
  let totalPetLegs := (numDogs * numLegsDog) + (numCats * numLegsCat)
  let totalLegsAccounted := totalPetLegs + numLegsHuman
  let numLegsRemaining := numLegsTotal - totalLegsAccounted
  let numChildren := numLegsRemaining / numLegsHuman
  exact sorry

end NUMINAMATH_GPT_number_of_children_l1108_110805


namespace NUMINAMATH_GPT_ratio_of_cream_l1108_110890

def initial_coffee := 18
def cup_capacity := 22
def Emily_drank := 3
def Emily_added_cream := 4
def Ethan_added_cream := 4
def Ethan_drank := 3

noncomputable def cream_in_Emily := Emily_added_cream

noncomputable def cream_remaining_in_Ethan :=
  Ethan_added_cream - (Ethan_added_cream * Ethan_drank / (initial_coffee + Ethan_added_cream))

noncomputable def resulting_ratio := cream_in_Emily / cream_remaining_in_Ethan

theorem ratio_of_cream :
  resulting_ratio = 200 / 173 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_cream_l1108_110890


namespace NUMINAMATH_GPT_least_n_satisfies_inequality_l1108_110863

theorem least_n_satisfies_inequality : ∃ n : ℕ, (1 : ℚ) / n - (1 : ℚ) / (n + 1) < 1 / 15 ∧ ∀ m : ℕ, (1 : ℚ) / m - (1 : ℚ) / (m + 1) < 1 / 15 -> ¬ (m < n) := 
sorry

end NUMINAMATH_GPT_least_n_satisfies_inequality_l1108_110863


namespace NUMINAMATH_GPT_sum_of_odd_integers_less_than_50_l1108_110896

def sumOddIntegersLessThan (n : Nat) : Nat :=
  List.sum (List.filter (λ x => x % 2 = 1) (List.range n))

theorem sum_of_odd_integers_less_than_50 : sumOddIntegersLessThan 50 = 625 :=
  by
    sorry

end NUMINAMATH_GPT_sum_of_odd_integers_less_than_50_l1108_110896


namespace NUMINAMATH_GPT_perpendicular_tangents_at_x0_l1108_110804

noncomputable def x0 := (36 : ℝ)^(1 / 3) / 6

theorem perpendicular_tangents_at_x0 :
  (∃ x0 : ℝ, (∃ f1 f2 : ℝ → ℝ,
    (∀ x, f1 x = x^2 - 1) ∧
    (∀ x, f2 x = 1 - x^3) ∧
    (2 * x0 * (-3 * x0^2) = -1)) ∧
    x0 = (36 : ℝ)^(1 / 3) / 6) := sorry

end NUMINAMATH_GPT_perpendicular_tangents_at_x0_l1108_110804


namespace NUMINAMATH_GPT_prob1_prob2_prob3_l1108_110879

def star (a b : ℤ) : ℤ :=
  if a = 0 then b^2
  else if b = 0 then a^2
  else if a > 0 ∧ b > 0 then a^2 + b^2
  else if a < 0 ∧ b < 0 then a^2 + b^2
  else -(a^2 + b^2)

theorem prob1 :
  star (-1) (-1) = 2 :=
sorry

theorem prob2 :
  star (-1) (star 0 (-2)) = -17 :=
sorry

theorem prob3 (m n : ℤ) :
  star (m-1) (n+2) = -2 → (m - n = 1 ∨ m - n = 5) :=
sorry

end NUMINAMATH_GPT_prob1_prob2_prob3_l1108_110879


namespace NUMINAMATH_GPT_incorrect_expressions_l1108_110809

theorem incorrect_expressions (x y : ℚ) (h : x / y = 2 / 5) :
    (x + 3 * y) / x ≠ 17 / 2 ∧ (x - y) / y ≠ 3 / 5 :=
by
  sorry

end NUMINAMATH_GPT_incorrect_expressions_l1108_110809


namespace NUMINAMATH_GPT_election_votes_l1108_110868

theorem election_votes (V : ℝ) (ha : 0.45 * V = 4860)
                       (hb : 0.30 * V = 3240)
                       (hc : 0.20 * V = 2160)
                       (hd : 0.05 * V = 540)
                       (hmaj : (0.45 - 0.30) * V = 1620) :
                       V = 10800 :=
by
  sorry

end NUMINAMATH_GPT_election_votes_l1108_110868


namespace NUMINAMATH_GPT_sum_of_consecutive_integers_l1108_110822

theorem sum_of_consecutive_integers {a b : ℤ} (h1 : a < b)
  (h2 : b = a + 1)
  (h3 : a < Real.sqrt 3)
  (h4 : Real.sqrt 3 < b) :
  a + b = 3 := 
sorry

end NUMINAMATH_GPT_sum_of_consecutive_integers_l1108_110822


namespace NUMINAMATH_GPT_tear_paper_l1108_110830

theorem tear_paper (n : ℕ) : 1 + 3 * n ≠ 2007 :=
by
  sorry

end NUMINAMATH_GPT_tear_paper_l1108_110830


namespace NUMINAMATH_GPT_divisible_by_42_l1108_110871

theorem divisible_by_42 (n : ℕ) : 42 ∣ (n^3 * (n^6 - 1)) :=
sorry

end NUMINAMATH_GPT_divisible_by_42_l1108_110871


namespace NUMINAMATH_GPT_minimize_quadratic_expression_l1108_110840

theorem minimize_quadratic_expression :
  ∃ x : ℝ, x = 3 ∧ ∀ y : ℝ, (y^2 - 6*y + 8) ≥ (x^2 - 6*x + 8) := by
sorry

end NUMINAMATH_GPT_minimize_quadratic_expression_l1108_110840


namespace NUMINAMATH_GPT_value_a7_l1108_110875

variables {a : ℕ → ℝ}

-- Condition 1: Arithmetic sequence where each term is non-zero
def arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

variable (h1 : arithmetic_sequence a)
-- Condition 2: 2a_3 - a_1^2 + 2a_11 = 0
variable (h2 : 2 * a 3 - (a 1)^2 + 2 * a 11 = 0)
-- Condition 3: a_3 + a_11 = 2a_7
variable (h3 : a 3 + a 11 = 2 * a 7)

theorem value_a7 : a 7 = 4 := by
  sorry

end NUMINAMATH_GPT_value_a7_l1108_110875


namespace NUMINAMATH_GPT_symmetric_points_origin_l1108_110825

theorem symmetric_points_origin (a b : ℝ) (h1 : a = -(-2)) (h2 : 1 = -b) : a + b = 1 :=
by
  sorry

end NUMINAMATH_GPT_symmetric_points_origin_l1108_110825


namespace NUMINAMATH_GPT_initial_volume_mixture_l1108_110815

theorem initial_volume_mixture (x : ℝ) :
  (4 * x) / (3 * x + 13) = 5 / 7 →
  13 * x = 65 →
  7 * x = 35 := 
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_initial_volume_mixture_l1108_110815


namespace NUMINAMATH_GPT_sum_of_digits_is_8_l1108_110862

theorem sum_of_digits_is_8 (d : ℤ) (h1 : d ≥ 0)
  (h2 : 8 * d / 5 - 80 = d) : (d / 100) + ((d % 100) / 10) + (d % 10) = 8 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_digits_is_8_l1108_110862
