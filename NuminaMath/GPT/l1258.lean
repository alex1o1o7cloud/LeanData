import Mathlib

namespace sum_of_digits_82_l1258_125812

def tens_digit (n : ℕ) : ℕ := n / 10
def units_digit (n : ℕ) : ℕ := n % 10
def sum_of_digits (n : ℕ) : ℕ := tens_digit n + units_digit n

theorem sum_of_digits_82 : sum_of_digits 82 = 10 := by
  sorry

end sum_of_digits_82_l1258_125812


namespace smallest_possible_average_l1258_125847

def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

def proper_digits (n : ℕ) : Prop :=
  ∀ (d : ℕ), d ∈ n.digits 10 → d = 0 ∨ d = 4 ∨ d = 8

theorem smallest_possible_average :
  ∃ n : ℕ, (n + 2) - n = 2 ∧ (sum_of_digits n + sum_of_digits (n + 2)) % 4 = 0 ∧ (∀ (d : ℕ), d ∈ n.digits 10 → d = 0 ∨ d = 4 ∨ d = 8) ∧ ∀ (d : ℕ), d ∈ (n + 2).digits 10 → d = 0 ∨ d = 4 ∨ d = 8 
  ∧ (n + (n + 2)) / 2 = 249 :=
sorry

end smallest_possible_average_l1258_125847


namespace f_at_2_l1258_125815

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  x^5 + a * x^3 + b * x

theorem f_at_2 (a b : ℝ) (h : f (-2) a b = 10) : f 2 a b = -10 :=
by 
  sorry

end f_at_2_l1258_125815


namespace total_balls_l1258_125844

def black_balls : ℕ := 8
def white_balls : ℕ := 6 * black_balls
theorem total_balls : white_balls + black_balls = 56 := 
by 
  sorry

end total_balls_l1258_125844


namespace boys_at_park_l1258_125867

theorem boys_at_park (girls parents groups people_per_group : ℕ) 
  (h_girls : girls = 14) 
  (h_parents : parents = 50)
  (h_groups : groups = 3) 
  (h_people_per_group : people_per_group = 25) : 
  (groups * people_per_group) - (girls + parents) = 11 := 
by 
  -- Not providing the proof, only the statement
  sorry

end boys_at_park_l1258_125867


namespace billy_reads_books_l1258_125813

theorem billy_reads_books :
  let hours_per_day := 8
  let days_per_weekend := 2
  let percent_playing_games := 0.75
  let percent_reading := 0.25
  let pages_per_hour := 60
  let pages_per_book := 80
  let total_free_time_per_weekend := hours_per_day * days_per_weekend
  let time_spent_playing := total_free_time_per_weekend * percent_playing_games
  let time_spent_reading := total_free_time_per_weekend * percent_reading
  let total_pages_read := time_spent_reading * pages_per_hour
  let books_read := total_pages_read / pages_per_book
  books_read = 3 := 
by
  -- proof would go here
  sorry

end billy_reads_books_l1258_125813


namespace pencils_per_row_l1258_125805

def total_pencils : ℕ := 32
def rows : ℕ := 4

theorem pencils_per_row : total_pencils / rows = 8 := by
  sorry

end pencils_per_row_l1258_125805


namespace tangent_position_is_six_oclock_l1258_125897

-- Define constants and initial conditions
def bigRadius : ℝ := 30
def smallRadius : ℝ := 15
def initialPosition := 12 -- 12 o'clock represented as initial tangent position
def initialArrowDirection := 0 -- upwards direction

-- Define that the small disk rolls counterclockwise around the clock face.
def rollsCCW := true

-- Define the destination position when the arrow next points upward.
def diskTangencyPosition (bR sR : ℝ) (initPos initDir : ℕ) (rolls : Bool) : ℕ :=
  if rolls then 6 else 12

theorem tangent_position_is_six_oclock :
  diskTangencyPosition bigRadius smallRadius initialPosition initialArrowDirection rollsCCW = 6 :=
sorry  -- the proof is omitted

end tangent_position_is_six_oclock_l1258_125897


namespace simplify_expression_l1258_125801

theorem simplify_expression (α : ℝ) (h_sin_ne_zero : Real.sin α ≠ 0) :
    (1 / Real.sin α + 1 / Real.tan α) * (1 - Real.cos α) = Real.sin α := 
sorry

end simplify_expression_l1258_125801


namespace visitors_not_ill_l1258_125836

theorem visitors_not_ill (total_visitors : ℕ) (ill_percentage : ℕ) (fall_ill : ℕ) : 
  total_visitors = 500 → 
  ill_percentage = 40 → 
  fall_ill = (ill_percentage * total_visitors) / 100 →
  total_visitors - fall_ill = 300 :=
by
  intros h1 h2 h3
  sorry

end visitors_not_ill_l1258_125836


namespace fraction_equality_l1258_125882

theorem fraction_equality (x y a b : ℝ) (hx : x / y = 3) (h : (2 * a - x) / (3 * b - y) = 3) : a / b = 9 / 2 :=
by
  sorry

end fraction_equality_l1258_125882


namespace seamless_assembly_with_equilateral_triangle_l1258_125880

theorem seamless_assembly_with_equilateral_triangle :
  ∃ (polygon : ℕ → ℝ) (angle_150 : ℝ),
    (polygon 4 = 90) ∧ (polygon 6 = 120) ∧ (polygon 8 = 135) ∧ (polygon 3 = 60) ∧ (angle_150 = 150) ∧
    (∃ (n₁ n₂ n₃ : ℕ), n₁ * 150 + n₂ * 150 + n₃ * 60 = 360) :=
by {
  -- The proof would involve checking the precise integer combination for seamless assembly
  sorry
}

end seamless_assembly_with_equilateral_triangle_l1258_125880


namespace find_puppy_weight_l1258_125811

noncomputable def weight_problem (a b c : ℕ) : Prop :=
  a + b + c = 36 ∧ a + c = 3 * b ∧ a + b = c + 6

theorem find_puppy_weight (a b c : ℕ) (h : weight_problem a b c) : a = 12 :=
sorry

end find_puppy_weight_l1258_125811


namespace first_pump_time_l1258_125819

-- Definitions for the conditions provided
def newer_model_rate := 1 / 6
def combined_rate := 1 / 3.6
def time_for_first_pump : ℝ := 9

-- The theorem to be proven
theorem first_pump_time (T : ℝ) (h1 : 1 / 6 + 1 / T = 1 / 3.6) : T = 9 :=
sorry

end first_pump_time_l1258_125819


namespace flight_duration_l1258_125870

theorem flight_duration (h m : ℕ) (H1 : 11 * 60 + 7 < 14 * 60 + 45) (H2 : 0 < m) (H3 : m < 60) :
  h + m = 41 := 
sorry

end flight_duration_l1258_125870


namespace ball_radius_and_surface_area_l1258_125879

theorem ball_radius_and_surface_area (d h : ℝ) (r : ℝ) :
  d = 12 ∧ h = 2 ∧ (6^2 + (r - h)^2 = r^2) → (r = 10 ∧ 4 * Real.pi * r^2 = 400 * Real.pi) := by
  sorry

end ball_radius_and_surface_area_l1258_125879


namespace max_value_of_symmetric_function_l1258_125883

def f (x a b : ℝ) : ℝ := (1 - x^2) * (x^2 + a * x + b)

theorem max_value_of_symmetric_function (a b : ℝ) (h_sym : ∀ x : ℝ, f x a b = f (-4 - x) a b) : 
  ∃ x : ℝ, (∀ y : ℝ, f y a b ≤ f x a b) ∧ f x a b = 16 :=
by
  sorry

end max_value_of_symmetric_function_l1258_125883


namespace every_positive_integer_has_good_multiple_l1258_125830

def is_good (n : ℕ) : Prop :=
  ∃ (D : Finset ℕ), (D.sum id = n) ∧ (1 ∈ D) ∧ (∀ d ∈ D, d ∣ n)

theorem every_positive_integer_has_good_multiple (n : ℕ) (hn : n > 0) : ∃ m : ℕ, (m % n = 0) ∧ is_good m :=
  sorry

end every_positive_integer_has_good_multiple_l1258_125830


namespace fractional_eq_a_range_l1258_125852

theorem fractional_eq_a_range (a : ℝ) :
  (∃ x : ℝ, (a / (x + 2) = 1 - 3 / (x + 2)) ∧ (x < 0)) ↔ (a < -1 ∧ a ≠ -3) := by
  sorry

end fractional_eq_a_range_l1258_125852


namespace geom_seq_sum_six_div_a4_minus_one_l1258_125803

theorem geom_seq_sum_six_div_a4_minus_one (a : ℕ → ℝ) (S : ℕ → ℝ) (r : ℝ) 
  (h1 : ∀ n, a (n + 1) = a 1 * r^n) 
  (h2 : a 1 = 1) 
  (h3 : a 2 * a 6 - 6 * a 4 - 16 = 0) :
  S 6 / (a 4 - 1) = 9 :=
sorry

end geom_seq_sum_six_div_a4_minus_one_l1258_125803


namespace cos_thm_l1258_125875

variable (θ : ℝ)

-- Conditions
def condition1 : Prop := 3 * Real.sin (2 * θ) = 4 * Real.tan θ
def condition2 : Prop := ∀ k : ℤ, θ ≠ k * Real.pi

-- Prove that cos 2θ = 1/3 given the conditions
theorem cos_thm (h1 : condition1 θ) (h2 : condition2 θ) : Real.cos (2 * θ) = 1 / 3 :=
by
  sorry

end cos_thm_l1258_125875


namespace find_cost_expensive_module_l1258_125878

-- Defining the conditions
def cost_cheaper_module : ℝ := 2.5
def total_modules : ℕ := 22
def num_cheaper_modules : ℕ := 21
def total_stock_value : ℝ := 62.5

-- The goal is to find the cost of the more expensive module 
def cost_expensive_module (cost_expensive_module : ℝ) : Prop :=
  num_cheaper_modules * cost_cheaper_module + cost_expensive_module = total_stock_value

-- The mathematically equivalent proof problem
theorem find_cost_expensive_module : cost_expensive_module 10 :=
by
  unfold cost_expensive_module
  norm_num
  sorry

end find_cost_expensive_module_l1258_125878


namespace table_length_is_77_l1258_125808

theorem table_length_is_77 :
  ∀ (x : ℕ), (∀ (sheets: ℕ), sheets = 72 → x = (5 + sheets)) → x = 77 :=
by {
  sorry
}

end table_length_is_77_l1258_125808


namespace range_alpha_sub_beta_l1258_125806

theorem range_alpha_sub_beta (α β : ℝ) (h₁ : -π/2 < α) (h₂ : α < β) (h₃ : β < π/2) : -π < α - β ∧ α - β < 0 := by
  sorry

end range_alpha_sub_beta_l1258_125806


namespace average_spring_headcount_average_fall_headcount_l1258_125817

namespace AverageHeadcount

def springHeadcounts := [10900, 10500, 10700, 11300]
def fallHeadcounts := [11700, 11500, 11600, 11300]

def averageHeadcount (counts : List ℕ) : ℕ :=
  counts.sum / counts.length

theorem average_spring_headcount :
  averageHeadcount springHeadcounts = 10850 := by
  sorry

theorem average_fall_headcount :
  averageHeadcount fallHeadcounts = 11525 := by
  sorry

end AverageHeadcount

end average_spring_headcount_average_fall_headcount_l1258_125817


namespace factor_polynomial_l1258_125841

theorem factor_polynomial (y : ℝ) : 
  y^6 - 64 = (y - 2) * (y + 2) * (y^2 + 2 * y + 4) * (y^2 - 2 * y + 4) :=
by
  sorry

end factor_polynomial_l1258_125841


namespace tan_difference_l1258_125810

variable (α β : ℝ)
variable (tan_α : ℝ := 3)
variable (tan_β : ℝ := 4 / 3)

theorem tan_difference (h₁ : Real.tan α = tan_α) (h₂ : Real.tan β = tan_β) : 
  Real.tan (α - β) = (tan_α - tan_β) / (1 + tan_α * tan_β) := by
  sorry

end tan_difference_l1258_125810


namespace set_union_inter_proof_l1258_125802

def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {1, 2, 3}

theorem set_union_inter_proof : A ∪ B = {0, 1, 2, 3} ∧ A ∩ B = {1, 2} := by
  sorry

end set_union_inter_proof_l1258_125802


namespace gold_bars_per_row_l1258_125850

theorem gold_bars_per_row 
  (total_worth : ℝ)
  (total_rows : ℕ)
  (value_per_bar : ℝ)
  (h_total_worth : total_worth = 1600000)
  (h_total_rows : total_rows = 4)
  (h_value_per_bar : value_per_bar = 40000) :
  total_worth / value_per_bar / total_rows = 10 :=
by
  sorry

end gold_bars_per_row_l1258_125850


namespace power_function_half_l1258_125854

theorem power_function_half (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = x ^ (1/2)) (hx : f 4 = 2) : 
  f (1/2) = (Real.sqrt 2) / 2 :=
by sorry

end power_function_half_l1258_125854


namespace sum_gcd_lcm_168_l1258_125885

def gcd_54_72 : ℕ := Nat.gcd 54 72

def lcm_50_15 : ℕ := Nat.lcm 50 15

def sum_gcd_lcm : ℕ := gcd_54_72 + lcm_50_15

theorem sum_gcd_lcm_168 : sum_gcd_lcm = 168 := by
  sorry

end sum_gcd_lcm_168_l1258_125885


namespace calculation_of_expression_l1258_125827

theorem calculation_of_expression
  (w x y z : ℕ)
  (h : 2^w * 3^x * 5^y * 7^z = 13230) :
  3 * w + 2 * x + 6 * y + 4 * z = 23 :=
sorry

end calculation_of_expression_l1258_125827


namespace four_ping_pong_four_shuttlecocks_cost_l1258_125839

theorem four_ping_pong_four_shuttlecocks_cost
  (x y : ℝ)
  (h1 : 3 * x + 2 * y = 15.5)
  (h2 : 2 * x + 3 * y = 17) :
  4 * x + 4 * y = 26 :=
sorry

end four_ping_pong_four_shuttlecocks_cost_l1258_125839


namespace estimate_greater_than_exact_l1258_125851

namespace NasreenRounding

variables (a b c d a' b' c' d' : ℕ)

-- Conditions: a, b, c, and d are large positive integers.
-- Definitions for rounding up and down
def round_up (n : ℕ) : ℕ := n + 1  -- Simplified model for rounding up
def round_down (n : ℕ) : ℕ := n - 1  -- Simplified model for rounding down

-- Conditions: a', b', c', and d' are the rounded values of a, b, c, and d respectively.
variable (h_round_a_up : a' = round_up a)
variable (h_round_b_down : b' = round_down b)
variable (h_round_c_down : c' = round_down c)
variable (h_round_d_down : d' = round_down d)

-- Question: Show that the estimate is greater than the original
theorem estimate_greater_than_exact :
  (a' / b' - c' * d') > (a / b - c * d) :=
sorry

end NasreenRounding

end estimate_greater_than_exact_l1258_125851


namespace rhombus_diagonals_l1258_125832

theorem rhombus_diagonals (p d_sum : ℝ) (h₁ : p = 100) (h₂ : d_sum = 62) :
  ∃ d₁ d₂ : ℝ, (d₁ + d₂ = d_sum) ∧ (d₁^2 + d₂^2 = (p/4)^2 * 4) ∧ ((d₁ = 48 ∧ d₂ = 14) ∨ (d₁ = 14 ∧ d₂ = 48)) :=
by
  sorry

end rhombus_diagonals_l1258_125832


namespace area_percent_of_smaller_rectangle_l1258_125824

-- Definitions of the main geometric elements and assumptions
def larger_rectangle (w h : ℝ) : Prop := (w > 0) ∧ (h > 0)
def radius_of_circle (w h r : ℝ) : Prop := r = Real.sqrt (w^2 + h^2)
def inscribed_smaller_rectangle (w h x y : ℝ) : Prop := 
  (0 < x) ∧ (x < 1) ∧ (0 < y) ∧ (y < 1) ∧
  ((h + 2 * y * h)^2 + (x * w)^2 = w^2 + h^2)

-- Prove the area percentage relationship
theorem area_percent_of_smaller_rectangle 
  (w h x y : ℝ) 
  (hw : w > 0) (hh : h > 0)
  (hcirc : radius_of_circle w h (Real.sqrt (w^2 + h^2)))
  (hsmall_rect : inscribed_smaller_rectangle w h x y) :
  (4 * x * y) / (4.0 * 1.0) * 100 = 8.33 := sorry

end area_percent_of_smaller_rectangle_l1258_125824


namespace totalBalls_l1258_125896

def jungkookBalls : Nat := 3
def yoongiBalls : Nat := 2

theorem totalBalls : jungkookBalls + yoongiBalls = 5 := by
  sorry

end totalBalls_l1258_125896


namespace population_of_village_l1258_125853

-- Define the given condition
def total_population (P : ℝ) : Prop :=
  0.4 * P = 23040

-- The theorem to prove that the total population is 57600
theorem population_of_village : ∃ P : ℝ, total_population P ∧ P = 57600 :=
by
  sorry

end population_of_village_l1258_125853


namespace max_area_of_triangle_l1258_125894

theorem max_area_of_triangle (AB BC AC : ℝ) (ratio : BC / AC = 3 / 5) (hAB : AB = 10) :
  ∃ A : ℝ, (A ≤ 260.52) :=
sorry

end max_area_of_triangle_l1258_125894


namespace sqrt_0_1681_eq_0_41_l1258_125859

theorem sqrt_0_1681_eq_0_41 (h : Real.sqrt 16.81 = 4.1) : Real.sqrt 0.1681 = 0.41 := by 
  sorry

end sqrt_0_1681_eq_0_41_l1258_125859


namespace two_students_cover_all_questions_l1258_125834

-- Define the main properties
variables (students : Finset ℕ) (questions : Finset ℕ)
variable (solves : ℕ → ℕ → Prop)

-- Assume the given conditions
axiom total_students : students.card = 8
axiom total_questions : questions.card = 8
axiom each_question_solved_by_min_5_students : ∀ q, q ∈ questions → 
(∃ student_set : Finset ℕ, student_set.card ≥ 5 ∧ ∀ s ∈ student_set, solves s q)

-- The theorem to be proven
theorem two_students_cover_all_questions :
  ∃ s1 s2 : ℕ, s1 ∈ students ∧ s2 ∈ students ∧ s1 ≠ s2 ∧ 
  ∀ q ∈ questions, solves s1 q ∨ solves s2 q :=
sorry -- proof to be written

end two_students_cover_all_questions_l1258_125834


namespace stratified_sampling_admin_staff_count_l1258_125886

theorem stratified_sampling_admin_staff_count
  (total_staff : ℕ)
  (admin_staff : ℕ)
  (sample_size : ℕ)
  (h_total : total_staff = 160)
  (h_admin : admin_staff = 32)
  (h_sample : sample_size = 20) :
  admin_staff * sample_size / total_staff = 4 :=
by
  sorry

end stratified_sampling_admin_staff_count_l1258_125886


namespace percentage_of_students_70_79_l1258_125821

-- Defining basic conditions
def students_in_range_90_100 := 5
def students_in_range_80_89 := 9
def students_in_range_70_79 := 7
def students_in_range_60_69 := 4
def students_below_60 := 3

-- Total number of students
def total_students := students_in_range_90_100 + students_in_range_80_89 + students_in_range_70_79 + students_in_range_60_69 + students_below_60

-- Percentage of students in the 70%-79% range
def percent_students_70_79 := (students_in_range_70_79 / total_students) * 100

theorem percentage_of_students_70_79 : percent_students_70_79 = 25 := by
  sorry

end percentage_of_students_70_79_l1258_125821


namespace sum_abc_l1258_125866

theorem sum_abc (A B C : ℕ) (hposA : 0 < A) (hposB : 0 < B) (hposC : 0 < C) (hgcd : Nat.gcd A (Nat.gcd B C) = 1)
  (hlog : A * Real.log 5 / Real.log 100 + B * Real.log 2 / Real.log 100 = C) : A + B + C = 5 :=
sorry

end sum_abc_l1258_125866


namespace total_blocks_fallen_l1258_125895

def stack_height (n : Nat) : Nat :=
  if n = 1 then 7
  else if n = 2 then 7 + 5
  else if n = 3 then 7 + 5 + 7
  else 0

def blocks_standing (n : Nat) : Nat :=
  if n = 1 then 0
  else if n = 2 then 2
  else if n = 3 then 3
  else 0

def blocks_fallen (n : Nat) : Nat :=
  stack_height n - blocks_standing n

theorem total_blocks_fallen : blocks_fallen 1 + blocks_fallen 2 + blocks_fallen 3 = 33 :=
  by
    sorry

end total_blocks_fallen_l1258_125895


namespace second_plan_minutes_included_l1258_125856

theorem second_plan_minutes_included 
  (monthly_fee1 : ℝ := 50) 
  (limit1 : ℝ := 500) 
  (cost_per_minute1 : ℝ := 0.35) 
  (monthly_fee2 : ℝ := 75) 
  (cost_per_minute2 : ℝ := 0.45) 
  (M : ℝ) 
  (usage : ℝ := 2500)
  (cost1 := monthly_fee1 + cost_per_minute1 * (usage - limit1))
  (cost2 := monthly_fee2 + cost_per_minute2 * (usage - M))
  (equal_costs : cost1 = cost2) : 
  M = 1000 := 
by
  sorry 

end second_plan_minutes_included_l1258_125856


namespace ratio_of_segments_intersecting_chords_l1258_125822

open Real

variables (EQ FQ HQ GQ : ℝ)

theorem ratio_of_segments_intersecting_chords 
  (h1 : EQ = 5) 
  (h2 : GQ = 7) 
  (h3 : EQ * FQ = GQ * HQ) : 
  FQ / HQ = 7 / 5 :=
by
  sorry

end ratio_of_segments_intersecting_chords_l1258_125822


namespace factorize_correct_l1258_125892
noncomputable def factorize_expression (a b : ℝ) : ℝ :=
  (a - b)^4 + (a + b)^4 + (a + b)^2 * (a - b)^2

theorem factorize_correct (a b : ℝ) :
  factorize_expression a b = (3 * a^2 + b^2) * (a^2 + 3 * b^2) :=
by
  sorry

end factorize_correct_l1258_125892


namespace part1_part2_l1258_125809

def A := {x : ℝ | 2 ≤ x ∧ x ≤ 7}
def B (m : ℝ) := {x : ℝ | -3 * m + 4 ≤ x ∧ x ≤ 2 * m - 1}

def p (m : ℝ) := ∀ x : ℝ, x ∈ A → x ∈ B m
def q (m : ℝ) := ∃ x : ℝ, x ∈ B m ∧ x ∈ A

theorem part1 (m : ℝ) : p m → m ≥ 4 := by
  sorry

theorem part2 (m : ℝ) : q m → m ≥ 3/2 := by
  sorry

end part1_part2_l1258_125809


namespace common_difference_arithmetic_sequence_l1258_125826

theorem common_difference_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ)
  (h1 : a 5 = 10) (h2 : a 12 = 31) : d = 3 :=
by
  sorry

end common_difference_arithmetic_sequence_l1258_125826


namespace value_of_expression_l1258_125858

theorem value_of_expression (b : ℚ) (h : b = 1/3) : (3 * b⁻¹ + (b⁻¹ / 3)) / b = 30 :=
by
  rw [h]
  sorry

end value_of_expression_l1258_125858


namespace find_p_l1258_125837

theorem find_p (p q : ℚ) (h1 : 5 * p + 6 * q = 10) (h2 : 6 * p + 5 * q = 17) : p = 52 / 11 :=
by
  sorry

end find_p_l1258_125837


namespace largest_n_l1258_125831

theorem largest_n : ∃ (n : ℕ), n < 1000 ∧ (∃ (m : ℕ), lcm m n = 3 * m * gcd m n) ∧ (∀ k, k < 1000 ∧ (∃ (m' : ℕ), lcm m' k = 3 * m' * gcd m' k) → k ≤ 972) := sorry

end largest_n_l1258_125831


namespace PQ_value_l1258_125825

theorem PQ_value (DE DF EF : ℕ) (CF : ℝ) (P Q : ℝ) 
  (h1 : DE = 996)
  (h2 : DF = 995)
  (h3 : EF = 994)
  (hCF :  CF = (995^2 - 4) / 1990)
  (hP : P = (1492.5 - EF))
  (hQ : Q = (s - DF)) :
  PQ = 1 ∧ m + n = 2 :=
by
  sorry

end PQ_value_l1258_125825


namespace Cindy_crayons_l1258_125857

variable (K : ℕ) -- Karen's crayons
variable (C : ℕ) -- Cindy's crayons

-- Given conditions
def Karen_has_639_crayons : Prop := K = 639
def Karen_has_135_more_crayons_than_Cindy : Prop := K = C + 135

-- The proof problem: showing Cindy's crayons
theorem Cindy_crayons (h1 : Karen_has_639_crayons K) (h2 : Karen_has_135_more_crayons_than_Cindy K C) : C = 504 :=
by
  sorry

end Cindy_crayons_l1258_125857


namespace find_k_l1258_125888

def f (a b c x : ℤ) : ℤ := a * x * x + b * x + c

theorem find_k : 
  ∃ k : ℤ, 
    ∃ a b c : ℤ, 
      f a b c 1 = 0 ∧
      60 < f a b c 6 ∧ f a b c 6 < 70 ∧
      120 < f a b c 9 ∧ f a b c 9 < 130 ∧
      10000 * k < f a b c 200 ∧ f a b c 200 < 10000 * (k + 1)
      ∧ k = 4 :=
by
  sorry

end find_k_l1258_125888


namespace range_of_a_l1258_125814

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 1 < x ∧ x < 2 → (x - 1) ^ 2 < Real.log x / Real.log a) → a ∈ Set.Ioc 1 2 :=
by
  sorry

end range_of_a_l1258_125814


namespace am_gm_inequality_l1258_125873

theorem am_gm_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (a + b) * (b + c) * (c + a) ≥ 8 * a * b * c :=
by 
  sorry

end am_gm_inequality_l1258_125873


namespace find_multiple_l1258_125823

-- Definitions of the divisor, original number, and remainders given in the problem conditions.
def D : ℕ := 367
def remainder₁ : ℕ := 241
def remainder₂ : ℕ := 115

-- Statement of the problem.
theorem find_multiple (N m k l : ℕ) :
  (N = k * D + remainder₁) →
  (m * N = l * D + remainder₂) →
  ∃ m, m > 0 ∧ 241 * m - 115 % 367 = 0 ∧ m = 2 :=
by
  sorry

end find_multiple_l1258_125823


namespace ben_eggs_remaining_l1258_125881

def initial_eggs : ℕ := 75

def ben_day1_morning : ℝ := 5
def ben_day1_afternoon : ℝ := 4.5
def alice_day1_morning : ℝ := 3.5
def alice_day1_evening : ℝ := 4

def ben_day2_morning : ℝ := 7
def ben_day2_evening : ℝ := 3
def alice_day2_morning : ℝ := 2
def alice_day2_afternoon : ℝ := 4.5
def alice_day2_evening : ℝ := 1.5

def ben_day3_morning : ℝ := 4
def ben_day3_afternoon : ℝ := 3.5
def alice_day3_evening : ℝ := 6.5

def total_eggs_eaten : ℝ :=
  (ben_day1_morning + ben_day1_afternoon + alice_day1_morning + alice_day1_evening) +
  (ben_day2_morning + ben_day2_evening + alice_day2_morning + alice_day2_afternoon + alice_day2_evening) +
  (ben_day3_morning + ben_day3_afternoon + alice_day3_evening)

def remaining_eggs : ℝ :=
  initial_eggs - total_eggs_eaten

theorem ben_eggs_remaining : remaining_eggs = 26 := by
  -- proof goes here
  sorry

end ben_eggs_remaining_l1258_125881


namespace victor_won_games_l1258_125848

theorem victor_won_games (V : ℕ) (ratio_victor_friend : 9 * 20 = 5 * V) : V = 36 :=
sorry

end victor_won_games_l1258_125848


namespace range_of_a3_l1258_125874

open Real

def convex_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, (a n + a (n + 2)) / 2 ≤ a (n + 1)

def sequence_condition (a : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
  ∀ n, 1 ≤ n → n < 10 → abs (a n - b n) ≤ 20

def b (n : ℕ) : ℝ := n^2 - 6 * n + 10

theorem range_of_a3 (a : ℕ → ℝ) :
  convex_sequence a →
  a 1 = 1 →
  a 10 = 28 →
  sequence_condition a b →
  7 ≤ a 3 ∧ a 3 ≤ 19 :=
sorry

end range_of_a3_l1258_125874


namespace ratio_of_segments_of_hypotenuse_l1258_125845

theorem ratio_of_segments_of_hypotenuse
  (a b c r s : ℝ) (h_right_triangle : a^2 + b^2 = c^2)
  (h_ratio : a / b = 2 / 5)
  (h_r : r = (a^2) / c) 
  (h_s : s = (b^2) / c) : 
  r / s = 4 / 25 := sorry

end ratio_of_segments_of_hypotenuse_l1258_125845


namespace bride_older_than_groom_l1258_125840

-- Define the ages of the bride and groom
variables (B G : ℕ)

-- Given conditions
def groom_age : Prop := G = 83
def total_age : Prop := B + G = 185

-- Theorem to prove how much older the bride is than the groom
theorem bride_older_than_groom (h1 : groom_age G) (h2 : total_age B G) : B - G = 19 :=
sorry

end bride_older_than_groom_l1258_125840


namespace fixed_point_of_line_l1258_125877

theorem fixed_point_of_line (m : ℝ) : 
  (m - 2) * (-3) - 8 + 3 * m + 2 = 0 :=
by
  sorry

end fixed_point_of_line_l1258_125877


namespace count_cubes_between_bounds_l1258_125804

theorem count_cubes_between_bounds : ∃ (n : ℕ), n = 42 ∧
  ∀ x, 2^9 + 1 ≤ x^3 ∧ x^3 ≤ 2^17 + 1 ↔ 9 ≤ x ∧ x ≤ 50 := 
sorry

end count_cubes_between_bounds_l1258_125804


namespace min_rows_for_students_l1258_125862

def min_rows (total_students seats_per_row max_students_per_school : ℕ) : ℕ :=
  total_students / seats_per_row + if total_students % seats_per_row == 0 then 0 else 1

theorem min_rows_for_students :
  ∀ (total_students seats_per_row max_students_per_school : ℕ),
  (total_students = 2016) →
  (seats_per_row = 168) →
  (max_students_per_school = 40) →
  min_rows total_students seats_per_row max_students_per_school = 15 :=
by
  intros total_students seats_per_row max_students_per_school h1 h2 h3
  -- We write down the proof outline to show that 15 is the required minimum
  sorry

end min_rows_for_students_l1258_125862


namespace exponent_multiplication_l1258_125884

theorem exponent_multiplication :
  (5^0.2 * 10^0.4 * 10^0.1 * 10^0.5 * 5^0.8) = 50 := by
  sorry

end exponent_multiplication_l1258_125884


namespace petya_vasya_meet_at_lantern_64_l1258_125833

-- Define the total number of lanterns and intervals
def total_lanterns : ℕ := 100
def total_intervals : ℕ := total_lanterns - 1

-- Define the positions of Petya and Vasya at a given time
def petya_initial : ℕ := 1
def vasya_initial : ℕ := 100
def petya_position : ℕ := 22
def vasya_position : ℕ := 88

-- Define the number of intervals covered by Petya and Vasya
def petya_intervals_covered : ℕ := petya_position - petya_initial
def vasya_intervals_covered : ℕ := vasya_initial - vasya_position

-- Define the combined intervals covered
def combined_intervals_covered : ℕ := petya_intervals_covered + vasya_intervals_covered

-- Define the interval after which Petya and Vasya will meet
def meeting_intervals : ℕ := total_intervals - combined_intervals_covered

-- Define the final meeting point according to Petya's travel
def meeting_lantern : ℕ := petya_initial + (meeting_intervals / 2)

theorem petya_vasya_meet_at_lantern_64 : meeting_lantern = 64 := by {
  -- Proof goes here
  sorry
}

end petya_vasya_meet_at_lantern_64_l1258_125833


namespace divides_14_pow_n_minus_27_for_all_natural_numbers_l1258_125871

theorem divides_14_pow_n_minus_27_for_all_natural_numbers :
  ∀ n : ℕ, 13 ∣ 14^n - 27 :=
by sorry

end divides_14_pow_n_minus_27_for_all_natural_numbers_l1258_125871


namespace sum_arithmetic_seq_nine_terms_l1258_125846

theorem sum_arithmetic_seq_nine_terms
  (a_n : ℕ → ℝ)
  (S_n : ℕ → ℝ)
  (k : ℝ)
  (h1 : ∀ n, a_n n = k * n + 4 - 5 * k)
  (h2 : ∀ n, S_n n = (n / 2) * (a_n 1 + a_n n))
  : S_n 9 = 36 :=
sorry

end sum_arithmetic_seq_nine_terms_l1258_125846


namespace complement_of_union_is_neg3_l1258_125843

open Set

variable (U A B : Set Int)

def complement_union (U A B : Set Int) : Set Int :=
  U \ (A ∪ B)

theorem complement_of_union_is_neg3 (U A B : Set Int) (hU : U = {-3, -2, -1, 0, 1, 2, 3, 4, 5, 6})
  (hA : A = {-1, 0, 1, 2, 3}) (hB : B = {-2, 3, 4, 5, 6}) :
  complement_union U A B = {-3} :=
by
  sorry

end complement_of_union_is_neg3_l1258_125843


namespace jason_books_is_21_l1258_125860

def keith_books : ℕ := 20
def total_books : ℕ := 41

theorem jason_books_is_21 (jason_books : ℕ) : 
  jason_books + keith_books = total_books → 
  jason_books = 21 := 
by 
  intro h
  sorry

end jason_books_is_21_l1258_125860


namespace correct_propositions_l1258_125890

noncomputable def f : ℝ → ℝ := sorry

def proposition1 : Prop :=
  ∀ x : ℝ, f (1 + 2 * x) = f (1 - 2 * x) → ∀ x : ℝ, f (2 - x) = f x

def proposition2 : Prop :=
  ∀ x : ℝ, f (x - 2) = f (2 - x)

def proposition3 : Prop :=
  (∀ x : ℝ, f x = f (-x)) ∧ (∀ x : ℝ, f (2 + x) = -f x) → ∀ x : ℝ, f x = f (4 - x)

def proposition4 : Prop :=
  (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x : ℝ, f x = f (-x - 2)) → ∀ x : ℝ, f (2 - x) = f x

theorem correct_propositions : proposition1 ∧ proposition2 ∧ proposition3 ∧ proposition4 :=
by sorry

end correct_propositions_l1258_125890


namespace gcd_547_323_l1258_125855

theorem gcd_547_323 : Nat.gcd 547 323 = 1 := 
by
  sorry

end gcd_547_323_l1258_125855


namespace difference_of_squares_l1258_125820

theorem difference_of_squares (x y : ℕ) (h1 : x + y = 60) (h2 : x - y = 16) : x^2 - y^2 = 960 :=
by
  sorry

end difference_of_squares_l1258_125820


namespace solution_set_of_inequality_l1258_125849

theorem solution_set_of_inequality :
  { x : ℝ | |x^2 - 3 * x| > 4 } = { x : ℝ | x < -1 ∨ x > 4 } :=
sorry

end solution_set_of_inequality_l1258_125849


namespace city_map_representation_l1258_125828

-- Given conditions
def scale (x : ℕ) : ℕ := x * 6
def cm_represents_km(cm : ℕ) : ℕ := scale cm
def fifteen_cm := 15
def ninety_km := 90

-- Given condition: 15 centimeters represents 90 kilometers
axiom representation : cm_represents_km fifteen_cm = ninety_km

-- Proof statement: A 20-centimeter length represents 120 kilometers
def twenty_cm := 20
def correct_answer := 120

theorem city_map_representation : cm_represents_km twenty_cm = correct_answer := by
  sorry

end city_map_representation_l1258_125828


namespace jason_bought_correct_dozens_l1258_125876

-- Given conditions
def cupcakes_per_cousin : Nat := 3
def cousins : Nat := 16
def cupcakes_per_dozen : Nat := 12

-- Calculated value
def total_cupcakes : Nat := cupcakes_per_cousin * cousins
def dozens_of_cupcakes_bought : Nat := total_cupcakes / cupcakes_per_dozen

-- Theorem statement
theorem jason_bought_correct_dozens : dozens_of_cupcakes_bought = 4 := by
  -- Proof omitted
  sorry

end jason_bought_correct_dozens_l1258_125876


namespace computer_price_difference_l1258_125864

-- Define the conditions as stated
def basic_computer_price := 1500
def total_price := 2500
def printer_price (P : ℕ) := basic_computer_price + P = total_price

def enhanced_computer_price (P E : ℕ) := P = (E + P) / 3

-- The theorem stating the proof problem
theorem computer_price_difference (P E : ℕ) 
  (h1 : printer_price P) 
  (h2 : enhanced_computer_price P E) : E - basic_computer_price = 500 :=
sorry

end computer_price_difference_l1258_125864


namespace infinitely_many_odd_n_composite_l1258_125887

theorem infinitely_many_odd_n_composite (n : ℕ) (h_odd : n % 2 = 1) : 
  ∃ (n : ℕ) (h_odd : n % 2 = 1), 
     ∀ k : ℕ, ∃ (m : ℕ) (h_odd_m : m % 2 = 1), 
     (∃ (d : ℕ), d ∣ (2^m + m) ∧ (1 < d ∧ d < 2^m + m))
:=
sorry

end infinitely_many_odd_n_composite_l1258_125887


namespace speed_of_train_l1258_125835

open Real

-- Define the conditions as given in the problem
def length_of_bridge : ℝ := 650
def length_of_train : ℝ := 200
def time_to_pass_bridge : ℝ := 17

-- Define the problem statement which needs to be proved
theorem speed_of_train : (length_of_bridge + length_of_train) / time_to_pass_bridge = 50 :=
by
  sorry

end speed_of_train_l1258_125835


namespace second_discount_percentage_l1258_125898

def normal_price : ℝ := 49.99
def first_discount : ℝ := 0.10
def final_price : ℝ := 36.0

theorem second_discount_percentage : 
  ∃ p : ℝ, (((normal_price - (first_discount * normal_price)) - final_price) / (normal_price - (first_discount * normal_price))) * 100 = p ∧ p = 20 :=
by
  sorry

end second_discount_percentage_l1258_125898


namespace trig_identity_proof_l1258_125872

theorem trig_identity_proof :
  (1 - 1 / (Real.cos (Real.pi / 6))) *
  (1 + 1 / (Real.sin (Real.pi / 3))) *
  (1 - 1 / (Real.sin (Real.pi / 6))) *
  (1 + 1 / (Real.cos (Real.pi / 3))) = 3 :=
by sorry

end trig_identity_proof_l1258_125872


namespace weight_of_B_l1258_125893

variable (W_A W_B W_C W_D : ℝ)

theorem weight_of_B (h1 : (W_A + W_B + W_C + W_D) / 4 = 60)
                    (h2 : (W_A + W_B) / 2 = 55)
                    (h3 : (W_B + W_C) / 2 = 50)
                    (h4 : (W_C + W_D) / 2 = 65) :
                    W_B = 50 :=
by sorry

end weight_of_B_l1258_125893


namespace math_club_team_selection_l1258_125889

noncomputable def choose (n k : ℕ) : ℕ :=
if h : k ≤ n then Nat.descFactorial n k / Nat.factorial k else 0

theorem math_club_team_selection :
  let boys := 10
  let girls := 12
  let team_size := 8
  let boys_selected := 4
  let girls_selected := 4
  choose boys boys_selected * choose girls girls_selected = 103950 := 
by simp [choose]; sorry

end math_club_team_selection_l1258_125889


namespace at_least_one_gt_one_l1258_125816

theorem at_least_one_gt_one (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y > 2) : (x > 1) ∨ (y > 1) :=
sorry

end at_least_one_gt_one_l1258_125816


namespace parallel_line_through_point_l1258_125891

theorem parallel_line_through_point (C : ℝ) :
  (∃ P : ℝ × ℝ, P.1 = 1 ∧ P.2 = 2) ∧ (∃ l : ℝ, ∀ x y : ℝ, 3 * x + y + l = 0) → 
  (3 * 1 + 2 + C = 0) → C = -5 :=
by
  sorry

end parallel_line_through_point_l1258_125891


namespace number_of_ways_to_form_team_l1258_125868

-- Defining the conditions
def total_employees : ℕ := 15
def num_men : ℕ := 10
def num_women : ℕ := 5
def team_size : ℕ := 6
def men_in_team : ℕ := 4
def women_in_team : ℕ := 2

-- Using binomial coefficient to represent combinations
noncomputable def choose (n k : ℕ) : ℕ := Nat.choose n k

-- The theorem to be proved
theorem number_of_ways_to_form_team :
  (choose num_men men_in_team) * (choose num_women women_in_team) = 
  choose 10 4 * choose 5 2 :=
by
  sorry

end number_of_ways_to_form_team_l1258_125868


namespace value_of_n_l1258_125818

theorem value_of_n (n : ℕ) (h1 : 0 < n) (h2 : n < Real.sqrt 65) (h3 : Real.sqrt 65 < n + 1) : n = 8 := 
sorry

end value_of_n_l1258_125818


namespace students_enthusiasts_both_l1258_125899

theorem students_enthusiasts_both {A B : Type} (class_size music_enthusiasts art_enthusiasts neither_enthusiasts enthusiasts_music_or_art : ℕ) 
(h_class_size : class_size = 50)
(h_music_enthusiasts : music_enthusiasts = 30) 
(h_art_enthusiasts : art_enthusiasts = 25)
(h_neither_enthusiasts : neither_enthusiasts = 4)
(h_enthusiasts_music_or_art : enthusiasts_music_or_art = class_size - neither_enthusiasts):
    (music_enthusiasts + art_enthusiasts - enthusiasts_music_or_art) = 9 := by
  sorry

end students_enthusiasts_both_l1258_125899


namespace symmetric_function_value_l1258_125865

theorem symmetric_function_value (f : ℝ → ℝ)
  (h : ∀ x, f (2^(x-2)) = x) : f 8 = 5 :=
sorry

end symmetric_function_value_l1258_125865


namespace oliver_shirts_not_washed_l1258_125863

theorem oliver_shirts_not_washed :
  let short_sleeve_shirts := 39
  let long_sleeve_shirts := 47
  let total_shirts := short_sleeve_shirts + long_sleeve_shirts
  let washed_shirts := 20
  let not_washed_shirts := total_shirts - washed_shirts
  not_washed_shirts = 66 := by
  sorry

end oliver_shirts_not_washed_l1258_125863


namespace arrangement_of_chairs_and_stools_l1258_125842

theorem arrangement_of_chairs_and_stools :
  (Nat.choose 10 3) = 120 :=
by
  -- Proof goes here
  sorry

end arrangement_of_chairs_and_stools_l1258_125842


namespace quadratic_inequality_solution_minimum_value_expression_l1258_125800

theorem quadratic_inequality_solution (a : ℝ) : (∀ x : ℝ, a * x^2 - 6 * x + 3 > 0) → a > 3 :=
sorry

theorem minimum_value_expression (a : ℝ) : (a > 3) → a + 9 / (a - 1) ≥ 7 ∧ (a + 9 / (a - 1) = 7 ↔ a = 4) :=
sorry

end quadratic_inequality_solution_minimum_value_expression_l1258_125800


namespace largest_possible_radius_tangent_circle_l1258_125861

theorem largest_possible_radius_tangent_circle :
  ∃ (r : ℝ), 0 < r ∧
    (∀ x y, (x - r)^2 + (y - r)^2 = r^2 → 
    ((x = 9 ∧ y = 2) → (r = 17))) :=
by
  sorry

end largest_possible_radius_tangent_circle_l1258_125861


namespace product_of_numbers_l1258_125838

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 60) (h2 : x - y = 10) : x * y = 875 :=
sorry

end product_of_numbers_l1258_125838


namespace ratio_water_duck_to_pig_l1258_125807

theorem ratio_water_duck_to_pig :
  let gallons_per_minute := 3
  let pumping_minutes := 25
  let total_gallons := gallons_per_minute * pumping_minutes
  let corn_rows := 4
  let plants_per_row := 15
  let gallons_per_corn_plant := 0.5
  let total_corn_plants := corn_rows * plants_per_row
  let total_corn_water := total_corn_plants * gallons_per_corn_plant
  let pig_count := 10
  let gallons_per_pig := 4
  let total_pig_water := pig_count * gallons_per_pig
  let duck_count := 20
  let total_duck_water := total_gallons - total_corn_water - total_pig_water
  let gallons_per_duck := total_duck_water / duck_count
  let ratio := gallons_per_duck / gallons_per_pig
  ratio = 1 / 16 := 
by
  sorry

end ratio_water_duck_to_pig_l1258_125807


namespace initial_students_count_eq_16_l1258_125829

variable (n T : ℕ)
variable (h1 : (T:ℝ) / n = 62.5)
variable (h2 : ((T - 70):ℝ) / (n - 1) = 62.0)

theorem initial_students_count_eq_16 :
  n = 16 :=
by
  sorry

end initial_students_count_eq_16_l1258_125829


namespace xy_value_x2_y2_value_l1258_125869

noncomputable def x : ℝ := Real.sqrt 7 + Real.sqrt 3
noncomputable def y : ℝ := Real.sqrt 7 - Real.sqrt 3

theorem xy_value : x * y = 4 := by
  -- proof goes here
  sorry

theorem x2_y2_value : x^2 + y^2 = 20 := by
  -- proof goes here
  sorry

end xy_value_x2_y2_value_l1258_125869
