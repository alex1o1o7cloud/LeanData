import Mathlib

namespace arithmetic_sequence_sum_l1138_113856

theorem arithmetic_sequence_sum (a : ℕ → ℚ) (S : ℕ → ℚ) (a_1 : ℚ) (d : ℚ) (m : ℕ) 
    (ha1 : a_1 = 2) 
    (ha2 : a 2 + a 8 = 24)
    (ham : 2 * a m = 24) 
    (h_sum : ∀ n, S n = (n * (2 * a_1 + (n - 1) * d)) / 2) 
    (h_an : ∀ n, a n = a_1 + (n - 1) * d) : 
    S (2 * m) = 265 / 2 :=
by
    sorry

end arithmetic_sequence_sum_l1138_113856


namespace students_not_opt_for_math_l1138_113857

theorem students_not_opt_for_math (total_students S E both_subjects M : ℕ) 
    (h1 : total_students = 40) 
    (h2 : S = 15) 
    (h3 : E = 2) 
    (h4 : both_subjects = 7) 
    (h5 : total_students - both_subjects = M + S - E) : M = 20 := 
  by
  sorry

end students_not_opt_for_math_l1138_113857


namespace find_a_c_area_A_90_area_B_90_l1138_113866

variable (a b c : ℝ)
variable (C : ℝ)

def triangle_condition1 := a + 1/a = 4 * Real.cos C
def triangle_condition2 := b = 1
def sin_C := Real.sin C = Real.sqrt 21 / 7

-- Proof problem for (1)
theorem find_a_c (h1 : triangle_condition1 a C) (h2 : triangle_condition2 b) (h3 : sin_C C) :
  (a = Real.sqrt 7 ∧ c = 2) ∨ (a = Real.sqrt 7 / 7 ∧ c = 2 * Real.sqrt 7 / 7) :=
sorry

-- Conditions for (2) when A=90°
def right_triangle_A := C = Real.pi / 2

-- Proof problem for (2) when A=90°
theorem area_A_90 (h1 : triangle_condition1 a C) (h2 : triangle_condition2 b) (h4 : right_triangle_A C) :
  ((a = Real.sqrt 3) → area = Real.sqrt 2 / 2) :=
sorry

-- Conditions for (2) when B=90°
def right_triangle_B := b = 1 ∧ C = Real.pi / 2

-- Proof problem for (2) when B=90°
theorem area_B_90 (h1 : triangle_condition1 a C) (h2 : triangle_condition2 b) (h5 : right_triangle_B b C) :
  ((a = Real.sqrt 3 / 3) → area = Real.sqrt 2 / 6) :=
sorry

end find_a_c_area_A_90_area_B_90_l1138_113866


namespace complex_unit_circle_sum_l1138_113899

theorem complex_unit_circle_sum :
  let z1 := (1 + Complex.I * Real.sqrt 3) / 2
  let z2 := (1 - Complex.I * Real.sqrt 3) / 2
  (z1 ^ 8 + z2 ^ 8 = -1) :=
by
  sorry

end complex_unit_circle_sum_l1138_113899


namespace intercept_sum_mod_7_l1138_113893

theorem intercept_sum_mod_7 :
  ∃ (x_0 y_0 : ℤ), (2 * x_0 ≡ 3 * y_0 + 1 [ZMOD 7]) ∧ (0 ≤ x_0) ∧ (x_0 < 7) ∧ (0 ≤ y_0) ∧ (y_0 < 7) ∧ (x_0 + y_0 = 6) :=
by
  sorry

end intercept_sum_mod_7_l1138_113893


namespace total_wet_surface_area_is_correct_l1138_113818

def cisternLength : ℝ := 8
def cisternWidth : ℝ := 4
def waterDepth : ℝ := 1.25

def bottomSurfaceArea : ℝ := cisternLength * cisternWidth
def longerSideSurfaceArea (depth : ℝ) : ℝ := depth * cisternLength * 2
def shorterSideSurfaceArea (depth : ℝ) : ℝ := depth * cisternWidth * 2

def totalWetSurfaceArea : ℝ :=
  bottomSurfaceArea + longerSideSurfaceArea waterDepth + shorterSideSurfaceArea waterDepth

theorem total_wet_surface_area_is_correct :
  totalWetSurfaceArea = 62 := by
  sorry

end total_wet_surface_area_is_correct_l1138_113818


namespace min_sqrt_diff_l1138_113840

theorem min_sqrt_diff (p : ℕ) (hp : Nat.Prime p) (hp_odd : p % 2 = 1) :
  ∃ x y : ℕ, x = (p - 1) / 2 ∧ y = (p + 1) / 2 ∧ x ≤ y ∧
    ∀ a b : ℕ, (a ≤ b) → (Real.sqrt (2 * p) - Real.sqrt a - Real.sqrt b ≥ 0) → 
      (Real.sqrt (2 * p) - Real.sqrt x - Real.sqrt y) ≤ (Real.sqrt (2 * p) - Real.sqrt a - Real.sqrt b) := 
by 
  -- Proof to be filled in
  sorry

end min_sqrt_diff_l1138_113840


namespace sequence_periodic_l1138_113883

theorem sequence_periodic (a : ℕ → ℚ) (h1 : a 1 = 4 / 5)
  (h2 : ∀ n, 0 ≤ a n ∧ a n ≤ 1 → 
    (a (n + 1) = if a n ≤ 1 / 2 then 2 * a n else 2 * a n - 1)) :
  a 2017 = 4 / 5 :=
sorry

end sequence_periodic_l1138_113883


namespace line_intercepts_and_slope_l1138_113817

theorem line_intercepts_and_slope :
  ∀ (x y : ℝ), (4 * x - 5 * y - 20 = 0) → 
  ∃ (x_intercept : ℝ) (y_intercept : ℝ) (slope : ℝ), 
    x_intercept = 5 ∧ y_intercept = -4 ∧ slope = 4 / 5 :=
by
  sorry

end line_intercepts_and_slope_l1138_113817


namespace auston_height_l1138_113834

noncomputable def auston_height_in_meters (height_in_inches : ℝ) : ℝ :=
  let height_in_cm := height_in_inches * 2.54
  height_in_cm / 100

theorem auston_height : auston_height_in_meters 65 = 1.65 :=
by
  sorry

end auston_height_l1138_113834


namespace white_sox_wins_l1138_113884

theorem white_sox_wins 
  (total_games : ℕ) 
  (games_won : ℕ) 
  (games_lost : ℕ)
  (win_loss_difference : ℤ) 
  (total_games_condition : total_games = 162) 
  (lost_games_condition : games_lost = 63) 
  (win_loss_diff_condition : (games_won : ℤ) - games_lost = win_loss_difference) 
  (win_loss_difference_value : win_loss_difference = 36) 
  : games_won = 99 :=
by
  sorry

end white_sox_wins_l1138_113884


namespace solve_quadratic_eqn_l1138_113804

theorem solve_quadratic_eqn : ∀ (x : ℝ), x^2 - 4 * x - 3 = 0 ↔ (x = 2 + Real.sqrt 7 ∨ x = 2 - Real.sqrt 7) :=
by
  sorry

end solve_quadratic_eqn_l1138_113804


namespace john_hourly_rate_with_bonus_l1138_113837

theorem john_hourly_rate_with_bonus:
  ∀ (daily_wage : ℝ) (work_hours : ℕ) (bonus : ℝ) (extra_hours : ℕ),
    daily_wage = 80 →
    work_hours = 8 →
    bonus = 20 →
    extra_hours = 2 →
    (daily_wage + bonus) / (work_hours + extra_hours) = 10 :=
by
  intros daily_wage work_hours bonus extra_hours
  intros h1 h2 h3 h4
  -- sorry: the proof is omitted
  sorry

end john_hourly_rate_with_bonus_l1138_113837


namespace books_sold_l1138_113848

theorem books_sold (initial_books remaining_books sold_books : ℕ):
  initial_books = 33 → 
  remaining_books = 7 → 
  sold_books = initial_books - remaining_books → 
  sold_books = 26 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end books_sold_l1138_113848


namespace distance_symmetric_line_eq_l1138_113854

noncomputable def distance_from_point_to_line : ℝ :=
  let x0 := 2
  let y0 := -1
  let A := 2
  let B := 3
  let C := 0
  (|A * x0 + B * y0 + C|) / (Real.sqrt (A^2 + B^2))

theorem distance_symmetric_line_eq : distance_from_point_to_line = 1 / (Real.sqrt 13) := by
  sorry

end distance_symmetric_line_eq_l1138_113854


namespace mobile_purchase_price_l1138_113876

theorem mobile_purchase_price (M : ℝ) 
  (P_grinder : ℝ := 15000)
  (L_grinder : ℝ := 0.05 * P_grinder)
  (SP_grinder : ℝ := P_grinder - L_grinder)
  (SP_mobile : ℝ := 1.1 * M)
  (P_overall : ℝ := P_grinder + M)
  (SP_overall : ℝ := SP_grinder + SP_mobile)
  (profit : ℝ := 50)
  (h : SP_overall = P_overall + profit) :
  M = 8000 :=
by 
  sorry

end mobile_purchase_price_l1138_113876


namespace quotient_when_divided_by_5_l1138_113833

theorem quotient_when_divided_by_5 (N : ℤ) (k : ℤ) (Q : ℤ) 
  (h1 : N = 5 * Q) 
  (h2 : N % 4 = 2) : 
  Q = 2 := 
sorry

end quotient_when_divided_by_5_l1138_113833


namespace inequality_solution_l1138_113827

theorem inequality_solution (x : ℝ) (h : x ≠ -5) : 
  (x^2 - 25) / (x + 5) < 0 ↔ x ∈ Set.union (Set.Iio (-5)) (Set.Ioo (-5) 5) := 
by
  sorry

end inequality_solution_l1138_113827


namespace interest_difference_l1138_113820

-- Conditions
def principal : ℕ := 350
def rate : ℕ := 4
def time : ℕ := 8

-- Question rewritten as a statement to prove
theorem interest_difference :
  let SI := (principal * rate * time) / 100 
  let difference := principal - SI
  difference = 238 := by
  sorry

end interest_difference_l1138_113820


namespace gasoline_price_percent_increase_l1138_113898

theorem gasoline_price_percent_increase 
  (highest_price : ℕ) (lowest_price : ℕ) 
  (h_highest : highest_price = 17) 
  (h_lowest : lowest_price = 10) : 
  (highest_price - lowest_price) * 100 / lowest_price = 70 := 
by 
  sorry

end gasoline_price_percent_increase_l1138_113898


namespace evaluate_expression_l1138_113872

theorem evaluate_expression : 
  let a := 3 
  let b := 2 
  (a^2 + b)^2 - (a^2 - b)^2 + 2*a*b = 78 := 
by
  let a := 3
  let b := 2
  sorry

end evaluate_expression_l1138_113872


namespace simplify_expr1_simplify_expr2_l1138_113821

variable {a b : ℝ}

theorem simplify_expr1 : 3 * a - (4 * b - 2 * a + 1) = 5 * a - 4 * b - 1 :=
by
  sorry

theorem simplify_expr2 : 2 * (5 * a - 3 * b) - 3 * (a ^ 2 - 2 * b) = 10 * a - 3 * a ^ 2 :=
by
  sorry

end simplify_expr1_simplify_expr2_l1138_113821


namespace solve_for_x_l1138_113888

theorem solve_for_x (x : ℝ) : 2 * x + 3 * x = 600 - (4 * x + 6 * x) → x = 40 :=
by
  intro h
  sorry

end solve_for_x_l1138_113888


namespace remainder_when_divided_by_6_l1138_113894

theorem remainder_when_divided_by_6 (n : ℕ) (h1 : Nat.Prime (n + 3)) (h2 : Nat.Prime (n + 7)) : n % 6 = 4 :=
  sorry

end remainder_when_divided_by_6_l1138_113894


namespace part1_part2_part3_l1138_113871

section CircleLine

-- Given: Circle C with equation x^2 + y^2 - 2x - 2y + 1 = 0
-- Tangent to line l intersecting the x-axis at A and the y-axis at B
variable (a b : ℝ) (ha : a > 2) (hb : b > 2)

-- Ⅰ. Prove that (a - 2)(b - 2) = 2
theorem part1 : (a - 2) * (b - 2) = 2 :=
sorry

-- Ⅱ. Find the equation of the trajectory of the midpoint of segment AB
theorem part2 (x y : ℝ) (hx : x > 1) (hy : y > 1) : (x - 1) * (y - 1) = 1 :=
sorry

-- Ⅲ. Find the minimum value of the area of triangle AOB
theorem part3 : ∃ (area : ℝ), area = 6 :=
sorry

end CircleLine

end part1_part2_part3_l1138_113871


namespace point_P_in_Quadrant_II_l1138_113859

noncomputable def α : ℝ := (5 * Real.pi) / 8

theorem point_P_in_Quadrant_II : (Real.sin α > 0) ∧ (Real.tan α < 0) := sorry

end point_P_in_Quadrant_II_l1138_113859


namespace minimize_sum_of_squares_l1138_113811

open Real

-- Assume x, y are positive real numbers and x + y = s
variables {x y s : ℝ}
variables (hx_pos : 0 < x) (hy_pos : 0 < y) (h_sum : x + y = s)

theorem minimize_sum_of_squares :
  (x = y) ∧ (2 * x * x = s * s / 2) → (x = s / 2 ∧ y = s / 2 ∧ x^2 + y^2 = s^2 / 2) :=
by
  sorry

end minimize_sum_of_squares_l1138_113811


namespace determine_c_l1138_113809

-- Define the points
def point1 : ℝ × ℝ := (-3, 1)
def point2 : ℝ × ℝ := (0, 4)

-- Define the direction vector calculation
def direction_vector : ℝ × ℝ := (point2.1 - point1.1, point2.2 - point1.2)

-- Define the target direction vector form
def target_direction_vector (c : ℝ) : ℝ × ℝ := (3, c)

-- Theorem stating that the calculated direction vector equals the target direction vector when c = 3
theorem determine_c : direction_vector = target_direction_vector 3 :=
by
  -- Proof omitted
  sorry

end determine_c_l1138_113809


namespace length_of_P1P2_segment_l1138_113847

theorem length_of_P1P2_segment (x : ℝ) (h₀ : 0 < x ∧ x < π / 2) (h₁ : 6 * Real.cos x = 9 * Real.tan x) :
  Real.sin x = 1 / 2 :=
by
  sorry

end length_of_P1P2_segment_l1138_113847


namespace mark_spending_l1138_113829

theorem mark_spending (initial_money : ℕ) (first_store_half : ℕ) (first_store_additional : ℕ) 
                      (second_store_third : ℕ) (remaining_money : ℕ) (total_spent : ℕ) : 
  initial_money = 180 ∧ 
  first_store_half = 90 ∧ 
  first_store_additional = 14 ∧ 
  total_spent = first_store_half + first_store_additional ∧
  remaining_money = initial_money - total_spent ∧
  second_store_third = 60 ∧ 
  remaining_money - second_store_third = 16 ∧ 
  initial_money - (total_spent + second_store_third + 16) = 0 → 
  remaining_money - second_store_third = 16 :=
by
  intro h
  sorry

end mark_spending_l1138_113829


namespace tangent_line_sin_at_pi_l1138_113858

theorem tangent_line_sin_at_pi :
  ∀ (f : ℝ → ℝ), 
    (∀ x, f x = Real.sin x) → ∀ x y, (x, y) = (Real.pi, 0) → 
    ∃ (m : ℝ) (b : ℝ), (∀ x, y = m * x + b) ∧ (m = -1) ∧ (b = Real.pi) :=
by
  sorry

end tangent_line_sin_at_pi_l1138_113858


namespace find_c_l1138_113826

/-- Seven unit squares are arranged in a row in the coordinate plane, 
with the lower left corner of the first square at the origin. 
A line extending from (c,0) to (4,4) divides the entire region 
into two regions of equal area. What is the value of c?
-/
theorem find_c (c : ℝ) (h : ∀ x y : ℝ, 0 ≤ x ∧ x ≤ 7 ∧ y = (4 / (4 - c)) * (x - c)) : c = 2.25 :=
sorry

end find_c_l1138_113826


namespace ondra_homework_problems_l1138_113890

theorem ondra_homework_problems (a b c d : ℤ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (-a) * (-b) ≠ -a - b ∧ 
  (-c) * (-d) = -182 * (1 / (-c - d)) →
  ((a = 2 ∧ b = 2) 
  ∨ (c = 1 ∧ d = 13) 
  ∨ (c = 13 ∧ d = 1)) :=
sorry

end ondra_homework_problems_l1138_113890


namespace two_digit_number_is_91_l1138_113864

/-- A positive two-digit number is odd and is a multiple of 13.
    The product of its digits is a perfect square.
    What is this two-digit number? -/
theorem two_digit_number_is_91 (M : ℕ) (h1 : M > 9) (h2 : M < 100) (h3 : M % 2 = 1) (h4 : M % 13 = 0) (h5 : ∃ n : ℕ, n * n = (M / 10) * (M % 10)) :
  M = 91 :=
sorry

end two_digit_number_is_91_l1138_113864


namespace exists_pow_two_sub_one_divisible_by_odd_l1138_113810

theorem exists_pow_two_sub_one_divisible_by_odd {a : ℕ} (h_odd : a % 2 = 1) 
  : ∃ b : ℕ, (2^b - 1) % a = 0 :=
sorry

end exists_pow_two_sub_one_divisible_by_odd_l1138_113810


namespace quadratic_real_equal_roots_l1138_113878

theorem quadratic_real_equal_roots (m : ℝ) :
  (3*x^2 + (2 - m)*x + 5 = 0 → (3 : ℕ) * x^2 + ((2 : ℕ) - m) * x + (5 : ℕ) = 0) →
  ∃ m₁ m₂ : ℝ, m₁ = 2 - 2 * Real.sqrt 15 ∧ m₂ = 2 + 2 * Real.sqrt 15 ∧ 
    (∀ x : ℝ, (3 * x^2 + (2 - m₁) * x + 5 = 0) ∧ (3 * x^2 + (2 - m₂) * x + 5 = 0)) :=
sorry

end quadratic_real_equal_roots_l1138_113878


namespace balloons_given_by_mom_l1138_113807

def num_balloons_initial : ℕ := 26
def num_balloons_total : ℕ := 60

theorem balloons_given_by_mom :
  (num_balloons_total - num_balloons_initial) = 34 := 
by
  sorry

end balloons_given_by_mom_l1138_113807


namespace part1_part2_l1138_113819

-- Define set A
def A : Set ℝ := {x | 3 < x ∧ x < 6}

-- Define set B
def B : Set ℝ := {x | 2 < x ∧ x < 9}

-- Define set complement in ℝ
def CR (S : Set ℝ) : Set ℝ := {x | ¬ (x ∈ S)}

-- First part of the problem
theorem part1 :
  (A ∩ B = {x | 3 < x ∧ x < 6}) ∧
  (CR A ∪ CR B = {x | x ≤ 3 ∨ x ≥ 6}) :=
sorry

-- Define set C depending on a
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < 2 * a - 1}

-- Second part of the problem
theorem part2 (a : ℝ) (h : B ∪ C a = B) :
  a ≤ 1 ∨ (2 ≤ a ∧ a ≤ 5) :=
sorry

end part1_part2_l1138_113819


namespace find_c_for_Q_l1138_113836

noncomputable def Q (c : ℚ) (x : ℚ) : ℚ := x^3 + 3*x^2 + c*x + 8

theorem find_c_for_Q (c : ℚ) : 
  (Q c 3 = 0) ↔ (c = -62 / 3) := by
  sorry

end find_c_for_Q_l1138_113836


namespace both_solve_correctly_l1138_113839

-- Define the probabilities of making an error for individuals A and B
variables (a b : ℝ)

-- Assuming a and b are probabilities, they must lie in the interval [0, 1]
axiom a_prob : 0 ≤ a ∧ a ≤ 1
axiom b_prob : 0 ≤ b ∧ b ≤ 1

-- Define the event that both individuals solve the problem correctly
theorem both_solve_correctly : (1 - a) * (1 - b) = (1 - a) * (1 - b) :=
by
  sorry

end both_solve_correctly_l1138_113839


namespace fraction_at_x_eq_4571_div_39_l1138_113874

def numerator (x : ℕ) : ℕ := x^6 - 16 * x^3 + x^2 + 64
def denominator (x : ℕ) : ℕ := x^3 - 8

theorem fraction_at_x_eq_4571_div_39 : numerator 5 / denominator 5 = 4571 / 39 :=
by
  sorry

end fraction_at_x_eq_4571_div_39_l1138_113874


namespace alexis_dresses_l1138_113832

-- Definitions based on the conditions
def isabella_total : ℕ := 13
def alexis_total : ℕ := 3 * isabella_total
def alexis_pants : ℕ := 21

-- Theorem statement
theorem alexis_dresses : alexis_total - alexis_pants = 18 := by
  sorry

end alexis_dresses_l1138_113832


namespace melissa_total_time_l1138_113853

-- Definitions based on the conditions in the problem
def time_replace_buckle : Nat := 5
def time_even_heel : Nat := 10
def time_fix_straps : Nat := 7
def time_reattach_soles : Nat := 12
def pairs_of_shoes : Nat := 8

-- Translation of the mathematically equivalent proof problem
theorem melissa_total_time : 
  (time_replace_buckle + time_even_heel + time_fix_straps + time_reattach_soles) * 16 = 544 :=
by
  sorry

end melissa_total_time_l1138_113853


namespace circle_center_radius_l1138_113881

theorem circle_center_radius :
  ∃ (h k r : ℝ), (∀ x y : ℝ, (x + 1)^2 + (y - 1)^2 = 4 → (x - h)^2 + (y - k)^2 = r^2) ∧
    h = -1 ∧ k = 1 ∧ r = 2 :=
by
  sorry

end circle_center_radius_l1138_113881


namespace find_k_value_l1138_113851

theorem find_k_value (Z K : ℤ) (h1 : 1000 < Z) (h2 : Z < 8000) (h3 : K > 2) (h4 : Z = K^3)
  (h5 : ∃ n : ℤ, Z = n^6) : K = 16 :=
sorry

end find_k_value_l1138_113851


namespace sheep_count_l1138_113869

theorem sheep_count {c s : ℕ} 
  (h1 : c + s = 20)
  (h2 : 2 * c + 4 * s = 60) : s = 10 :=
sorry

end sheep_count_l1138_113869


namespace area_of_sector_l1138_113852

noncomputable def circleAreaAboveXAxisAndRightOfLine : ℝ :=
  let radius := 10
  let area_of_circle := Real.pi * radius^2
  area_of_circle / 4

theorem area_of_sector :
  circleAreaAboveXAxisAndRightOfLine = 25 * Real.pi := sorry

end area_of_sector_l1138_113852


namespace hyperbola_eccentricity_l1138_113867

variable (a b c e : ℝ)
variable (a_pos : a > 0)
variable (b_pos : b > 0)
variable (hyperbola_eq : c = Real.sqrt (a^2 + b^2))
variable (y_B : ℝ)
variable (slope_eq : 3 = (y_B - 0) / (c - a))
variable (y_B_on_hyperbola : y_B = b^2 / a)

theorem hyperbola_eccentricity (h : a > 0) (h' : b > 0) (c_def : c = Real.sqrt (a^2 + b^2))
    (slope_cond : 3 = (y_B - 0) / (c - a)) (y_B_cond : y_B = b^2 / a) :
    e = 2 :=
sorry

end hyperbola_eccentricity_l1138_113867


namespace complement_union_l1138_113843

open Set

-- Definitions of the sets
def U : Set ℕ := {1, 2, 3, 4}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {2, 3}

-- Define the complement relative to U
def complement (A B : Set ℕ) : Set ℕ := { x ∈ B | x ∉ A }

-- The theorem we need to prove
theorem complement_union :
  complement (M ∪ N) U = {4} :=
by
  sorry

end complement_union_l1138_113843


namespace one_and_one_third_of_what_number_is_45_l1138_113805

theorem one_and_one_third_of_what_number_is_45 (x : ℚ) (h : (4 / 3) * x = 45) : x = 33.75 :=
by
  sorry

end one_and_one_third_of_what_number_is_45_l1138_113805


namespace soccer_tournament_solution_l1138_113862

-- Define the statement of the problem
theorem soccer_tournament_solution (k : ℕ) (n m : ℕ) (h1 : k ≥ 1) (h2 : n = (k+1)^2) (h3 : m = k*(k+1) / 2)
  (h4 : n > m) : 
  ∃ k : ℕ, n = (k + 1) ^ 2 ∧ m = k * (k + 1) / 2 ∧ k ≥ 1 := 
sorry

end soccer_tournament_solution_l1138_113862


namespace find_C_coordinates_l1138_113873

noncomputable def maximize_angle (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) (x : ℝ) : Prop :=
  ∀ C : ℝ × ℝ, C = (x, 0) → x = Real.sqrt (a * b)

theorem find_C_coordinates (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  maximize_angle a b ha hb hab (Real.sqrt (a * b)) :=
by  sorry

end find_C_coordinates_l1138_113873


namespace min_value_reciprocal_sum_l1138_113835

theorem min_value_reciprocal_sum (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_sum : a + b = 2) : 
  ∃ (m : ℝ), m = 2 ∧ ∀ c d : ℝ, 0 < c ∧ 0 < d ∧ c + d = 2 → (1/c + 1/d) ≥ m := 
sorry

end min_value_reciprocal_sum_l1138_113835


namespace sqrt_sqrt4_of_decimal_l1138_113803

theorem sqrt_sqrt4_of_decimal (h : 0.000625 = 625 / (10 ^ 6)) :
  Real.sqrt (Real.sqrt (Real.sqrt (Real.sqrt 625) / 1000)) = 0.4 :=
by
  sorry

end sqrt_sqrt4_of_decimal_l1138_113803


namespace count_common_divisors_l1138_113882

theorem count_common_divisors : 
  (Nat.divisors 60 ∩ Nat.divisors 90 ∩ Nat.divisors 30).card = 8 :=
by
  sorry

end count_common_divisors_l1138_113882


namespace positive_expression_l1138_113825

theorem positive_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  a^2 * (b + c) + a * (b^2 + c^2 - b * c) > 0 :=
by sorry

end positive_expression_l1138_113825


namespace john_expenditure_l1138_113891

theorem john_expenditure (X : ℝ) (h : (1/2) * X + (1/3) * X + (1/10) * X + 8 = X) : X = 120 :=
by
  sorry

end john_expenditure_l1138_113891


namespace smallest_n_for_divisibility_by_ten_million_l1138_113875

theorem smallest_n_for_divisibility_by_ten_million 
  (a₁ a₂ : ℝ) 
  (a₁_eq : a₁ = 5 / 6) 
  (a₂_eq : a₂ = 30) 
  (n : ℕ) 
  (T : ℕ → ℝ) 
  (T_def : ∀ (k : ℕ), T k = a₁ * (36 ^ (k - 1))) :
  (∃ n, T n = T 9 ∧ (∃ m : ℤ, T n = m * 10^7)) := 
sorry

end smallest_n_for_divisibility_by_ten_million_l1138_113875


namespace workshops_participation_l1138_113887

variable (x y z a b c d : ℕ)
variable (A B C : Finset ℕ)

theorem workshops_participation:
  (A.card = 15) →
  (B.card = 14) →
  (C.card = 11) →
  (25 = x + y + z + a + b + c + d) →
  (12 = a + b + c + d) →
  (A.card = x + a + c + d) →
  (B.card = y + a + b + d) →
  (C.card = z + b + c + d) →
  d = 0 :=
by
  intro hA hB hC hTotal hAtLeastTwo hAkA hBkA hCkA
  -- The proof will go here
  -- Parsing these inputs shall lead to establishing d = 0
  sorry

end workshops_participation_l1138_113887


namespace complement_union_l1138_113889

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {2, 4, 5}
def B : Set ℕ := {3, 4, 5}

theorem complement_union :
  ((U \ A) ∪ B) = {1, 3, 4, 5, 6} :=
by
  sorry

end complement_union_l1138_113889


namespace quadratic_root_k_eq_one_l1138_113812

theorem quadratic_root_k_eq_one
  (k : ℝ)
  (h₀ : (k + 3) ≠ 0)
  (h₁ : ∃ x : ℝ, (x = 0) ∧ ((k + 3) * x^2 + 5 * x + k^2 + 2 * k - 3 = 0)) :
  k = 1 :=
by
  sorry

end quadratic_root_k_eq_one_l1138_113812


namespace ticket_sales_amount_theater_collected_50_dollars_l1138_113815

variable (num_people total_people : ℕ) (cost_adult_entry cost_child_entry : ℕ) (num_children : ℕ)
variable (total_collected : ℕ)

theorem ticket_sales_amount
  (h1 : cost_adult_entry = 8)
  (h2 : cost_child_entry = 1)
  (h3 : total_people = 22)
  (h4 : num_children = 18)
  (h5 : num_people = total_people - num_children)
  : total_collected = (num_people * cost_adult_entry + num_children * cost_child_entry) := sorry

theorem theater_collected_50_dollars 
  (h1 : cost_adult_entry = 8)
  (h2 : cost_child_entry = 1)
  (h3 : total_people = 22)
  (h4 : num_children = 18)
  (h5 : total_collected = 50)
  : total_collected = 50 := sorry

end ticket_sales_amount_theater_collected_50_dollars_l1138_113815


namespace probability_largest_ball_is_six_l1138_113863

def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem probability_largest_ball_is_six : 
  (choose 6 4 : ℝ) / (choose 10 4 : ℝ) = (15 : ℝ) / (210 : ℝ) :=
by
  sorry

end probability_largest_ball_is_six_l1138_113863


namespace prove_b_minus_a_l1138_113828

noncomputable def point := (ℝ × ℝ)

def rotate90 (p : point) (c : point) : point :=
  let (x, y) := p
  let (h, k) := c
  (h - (y - k), k + (x - h))

def reflect_y_eq_x (p : point) : point :=
  let (x, y) := p
  (y, x)

def transformed_point (a b : ℝ) : point :=
  reflect_y_eq_x (rotate90 (a, b) (2, 6))

theorem prove_b_minus_a (a b : ℝ) (h1 : transformed_point a b = (-7, 4)) : b - a = 15 :=
by
  sorry

end prove_b_minus_a_l1138_113828


namespace maxwell_walking_speed_l1138_113823

-- Define Maxwell's walking speed
def Maxwell_speed (v : ℕ) : Prop :=
  ∀ t1 t2 : ℕ, t1 = 10 → t2 = 9 →
  ∀ d1 d2 : ℕ, d1 = 10 * v → d2 = 6 * t2 →
  ∀ d_total : ℕ, d_total = 94 →
  d1 + d2 = d_total

theorem maxwell_walking_speed : Maxwell_speed 4 :=
by
  sorry

end maxwell_walking_speed_l1138_113823


namespace range_of_m_l1138_113896

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, x^2 + m * x + 1 ≥ 0) ↔ (-2 ≤ m ∧ m ≤ 2) :=
by
  sorry

end range_of_m_l1138_113896


namespace part_1_part_2a_part_2b_l1138_113860

namespace InequalityProofs

-- Definitions extracted from the problem
def quadratic_function (m x : ℝ) : ℝ := m * x^2 + (1 - m) * x + m - 2

-- Lean statement for Part 1
theorem part_1 (m : ℝ) : (∀ x : ℝ, quadratic_function m x ≥ -2) ↔ m ∈ Set.Ici (1 / 3) :=
sorry

-- Lean statement for Part 2, breaking into separate theorems for different ranges of m
theorem part_2a (m : ℝ) (h : m < -1) :
  (∀ x : ℝ, quadratic_function m x < m - 1) → 
  (∀ x : ℝ, x ∈ (Set.Iic (-1 / m) ∪ Set.Ici 1)) :=
sorry

theorem part_2b (m : ℝ) (h : -1 < m ∧ m < 0) :
  (∀ x : ℝ, quadratic_function m x < m - 1) → 
  (∀ x : ℝ, x ∈ (Set.Iic 1 ∪ Set.Ici (-1 / m))) :=
sorry

end InequalityProofs

end part_1_part_2a_part_2b_l1138_113860


namespace shopkeeper_profit_percentage_l1138_113813

theorem shopkeeper_profit_percentage 
  (cost_price : ℝ := 100) 
  (loss_due_to_theft_percent : ℝ := 30) 
  (overall_loss_percent : ℝ := 23) 
  (remaining_goods_value : ℝ := 70) 
  (overall_loss_value : ℝ := 23) 
  (selling_price : ℝ := 77) 
  (profit_percentage : ℝ) 
  (h1 : remaining_goods_value = cost_price * (1 - loss_due_to_theft_percent / 100)) 
  (h2 : overall_loss_value = cost_price * (overall_loss_percent / 100)) 
  (h3 : selling_price = cost_price - overall_loss_value) 
  (h4 : remaining_goods_value + remaining_goods_value * profit_percentage / 100 = selling_price) :
  profit_percentage = 10 := 
by 
  sorry

end shopkeeper_profit_percentage_l1138_113813


namespace train_still_there_when_susan_arrives_l1138_113816

-- Define the conditions and primary question
def time_between_1_and_2 (t : ℝ) : Prop := 0 ≤ t ∧ t ≤ 60

def train_arrival := {t : ℝ // time_between_1_and_2 t}
def susan_arrival := {t : ℝ // time_between_1_and_2 t}

def train_present (train : train_arrival) (susan : susan_arrival) : Prop :=
  susan.val ≥ train.val ∧ susan.val ≤ (train.val + 30)

-- Define the probability calculation
noncomputable def probability_train_present : ℝ :=
  (30 * 30 + (30 * (60 - 30) * 2) / 2) / (60 * 60)

theorem train_still_there_when_susan_arrives :
  probability_train_present = 1 / 2 :=
sorry

end train_still_there_when_susan_arrives_l1138_113816


namespace axis_of_symmetry_l1138_113885

-- Define the given parabola
def parabola (x : ℝ) : ℝ := x^2 - 2 * x + 5

-- Define the statement that we need to prove
theorem axis_of_symmetry : (∃ (a : ℝ), ∀ x, parabola (x) = (x - a) ^ 2 + 4) ∧ 
                           (∃ (b : ℝ), b = 1) :=
by
  sorry

end axis_of_symmetry_l1138_113885


namespace max_value_of_f_min_value_of_a2_4b2_min_value_of_a2_4b2_equals_l1138_113824

noncomputable def f (x a b : ℝ) : ℝ := |x - a| - |x + 2 * b|

theorem max_value_of_f (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∀ x, f x a b ≤ a + 2 * b :=
by sorry

theorem min_value_of_a2_4b2 (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_max : a + 2 * b = 1) :
  a^2 + 4 * b^2 ≥ 1 / 2 :=
by sorry

theorem min_value_of_a2_4b2_equals (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_max : a + 2 * b = 1) :
  ∃ a b, a = 1 / 2 ∧ b = 1 / 4 ∧ (a^2 + 4 * b^2 = 1 / 2) :=
by sorry

end max_value_of_f_min_value_of_a2_4b2_min_value_of_a2_4b2_equals_l1138_113824


namespace fraction_value_l1138_113802

theorem fraction_value :
  (20 - 19 + 18 - 17 + 16 - 15 + 14 - 13 + 12 - 11 + 10 - 9 + 8 - 7 + 6 - 5 + 4 - 3 + 2 - 1) /
  (1 - 2 + 3 - 4 + 5 - 6 + 7 - 8 + 9 - 10 + 11 - 12 + 13 - 14 + 15 - 16 + 17 - 18 + 19 - 20) = -1 :=
by
  -- simplified proof omitted
  sorry

end fraction_value_l1138_113802


namespace find_a_l1138_113806

def point_of_tangency (x0 y0 a : ℝ) : Prop :=
  (x0 - y0 - 1 = 0) ∧ (y0 = a * x0^2) ∧ (2 * a * x0 = 1)

theorem find_a (x0 y0 a : ℝ) (h : point_of_tangency x0 y0 a) : a = 1/4 :=
by
  sorry

end find_a_l1138_113806


namespace sum_of_constants_eq_zero_l1138_113830

theorem sum_of_constants_eq_zero (A B C D E : ℝ) :
  (∀ (x : ℝ), (x + 1) / ((x + 2) * (x + 3) * (x + 4) * (x + 5) * (x + 6)) =
              A / (x + 2) + B / (x + 3) + C / (x + 4) + D / (x + 5) + E / (x + 6)) →
  A + B + C + D + E = 0 :=
by
  sorry

end sum_of_constants_eq_zero_l1138_113830


namespace find_c1_minus_c2_l1138_113895

theorem find_c1_minus_c2 (c1 c2 : ℝ) (h1 : 2 * 3 + 3 * 5 = c1) (h2 : 5 = c2) : c1 - c2 = 16 := by
  sorry

end find_c1_minus_c2_l1138_113895


namespace juice_m_smoothie_l1138_113831

/-- 
24 oz of juice p and 25 oz of juice v are mixed to make smoothies m and y. 
The ratio of p to v in smoothie m is 4 to 1 and that in y is 1 to 5. 
Prove that the amount of juice p in the smoothie m is 20 oz.
-/
theorem juice_m_smoothie (P_m P_y V_m V_y : ℕ)
  (h1 : P_m + P_y = 24)
  (h2 : V_m + V_y = 25)
  (h3 : 4 * V_m = P_m)
  (h4 : V_y = 5 * P_y) :
  P_m = 20 :=
sorry

end juice_m_smoothie_l1138_113831


namespace number_of_possible_values_for_a_l1138_113822

theorem number_of_possible_values_for_a 
  (a b c d : ℤ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d)
  (h5 : a > b) (h6 : b > c) (h7 : c > d)
  (h8 : a + b + c + d = 2004)
  (h9 : a^2 - b^2 - c^2 + d^2 = 1004) : 
  ∃ n : ℕ, n = 500 :=
  sorry

end number_of_possible_values_for_a_l1138_113822


namespace number_of_paths_l1138_113850

theorem number_of_paths (n m : ℕ) (h : m ≤ n) : 
  ∃ paths : ℕ, paths = Nat.choose n m := 
sorry

end number_of_paths_l1138_113850


namespace pencils_given_l1138_113801

theorem pencils_given (pencils_original pencils_left pencils_given : ℕ)
  (h1 : pencils_original = 142)
  (h2 : pencils_left = 111)
  (h3 : pencils_given = pencils_original - pencils_left) :
  pencils_given = 31 :=
by
  sorry

end pencils_given_l1138_113801


namespace initially_calculated_average_height_l1138_113841

theorem initially_calculated_average_height
    (A : ℕ)
    (initial_total_height : ℕ)
    (real_total_height : ℕ)
    (height_error : ℕ := 60)
    (num_boys : ℕ := 35)
    (actual_average_height : ℕ := 183)
    (initial_total_height_eq : initial_total_height = num_boys * A)
    (real_total_height_eq : real_total_height = num_boys * actual_average_height)
    (height_discrepancy : initial_total_height = real_total_height + height_error) :
    A = 181 :=
by
  sorry

end initially_calculated_average_height_l1138_113841


namespace jim_profit_percentage_l1138_113846

theorem jim_profit_percentage (S C : ℝ) (H1 : S = 670) (H2 : C = 536) :
  ((S - C) / C) * 100 = 25 :=
by
  sorry

end jim_profit_percentage_l1138_113846


namespace find_angle_B_in_right_triangle_l1138_113842

theorem find_angle_B_in_right_triangle (A B C : ℝ) (hC : C = 90) (hA : A = 35) :
  B = 55 :=
by
  -- Assuming A, B, and C represent the three angles of a triangle ABC
  -- where C = 90 degrees and A = 35 degrees, we need to prove B = 55 degrees.
  sorry

end find_angle_B_in_right_triangle_l1138_113842


namespace some_number_value_l1138_113855

theorem some_number_value (a : ℕ) (some_number : ℕ) (h_a : a = 105)
  (h_eq : a ^ 3 = some_number * 25 * 35 * 63) : some_number = 7 := by
  sorry

end some_number_value_l1138_113855


namespace cos_6theta_l1138_113897

theorem cos_6theta (θ : ℝ) (h : Real.cos θ = 1/4) : Real.cos (6 * θ) = -3224/4096 := 
by
  sorry

end cos_6theta_l1138_113897


namespace original_savings_l1138_113879

variable (A B : ℕ)

-- A's savings are 5 times that of B's savings
def cond1 : Prop := A = 5 * B

-- If A withdraws 60 yuan and B deposits 60 yuan, then B's savings will be twice that of A's savings
def cond2 : Prop := (B + 60) = 2 * (A - 60)

-- Prove the original savings of A and B
theorem original_savings (h1 : cond1 A B) (h2 : cond2 A B) : A = 100 ∧ B = 20 := by
  sorry

end original_savings_l1138_113879


namespace martha_saves_half_daily_allowance_l1138_113838

theorem martha_saves_half_daily_allowance {f : ℚ} (h₁ : 12 > 0) (h₂ : (6 : ℚ) * 12 * f + (3 : ℚ) = 39) : f = 1 / 2 :=
by
  sorry

end martha_saves_half_daily_allowance_l1138_113838


namespace sin_cos_product_l1138_113892

open Real

theorem sin_cos_product (θ : ℝ) (h : sin θ + cos θ = 3 / 4) : sin θ * cos θ = -7 / 32 := 
  by 
    sorry

end sin_cos_product_l1138_113892


namespace inequality_least_n_l1138_113880

theorem inequality_least_n (n : ℕ) (h : (1 : ℝ) / n - (1 : ℝ) / (n + 2) < 1 / 15) : n = 5 :=
sorry

end inequality_least_n_l1138_113880


namespace solve_for_x_l1138_113800

def f (x : ℝ) : ℝ := 3 * x - 5

theorem solve_for_x (x : ℝ) : 2 * f x - 10 = f (x - 2) ↔ x = 3 :=
by
  sorry

end solve_for_x_l1138_113800


namespace arithmetic_sqrt_of_nine_l1138_113861

-- Define the arithmetic square root function which only considers non-negative values
noncomputable def arithmetic_sqrt (x : ℝ) : ℝ :=
  if hx : x ≥ 0 then Real.sqrt x else 0

-- The theorem to prove: The arithmetic square root of 9 is 3.
theorem arithmetic_sqrt_of_nine : arithmetic_sqrt 9 = 3 :=
by
  sorry

end arithmetic_sqrt_of_nine_l1138_113861


namespace hannahs_son_cuts_three_strands_per_minute_l1138_113877

variable (x : ℕ)

theorem hannahs_son_cuts_three_strands_per_minute
  (total_strands : ℕ)
  (hannah_rate : ℕ)
  (total_time : ℕ)
  (total_strands_cut : ℕ := hannah_rate * total_time)
  (son_rate := (total_strands - total_strands_cut) / total_time)
  (hannah_rate := 8)
  (total_time := 2)
  (total_strands := 22) :
  son_rate = 3 := 
by
  sorry

end hannahs_son_cuts_three_strands_per_minute_l1138_113877


namespace carterHas152Cards_l1138_113868

-- Define the number of baseball cards Marcus has.
def marcusCards : Nat := 210

-- Define the number of baseball cards Carter has.
def carterCards : Nat := marcusCards - 58

-- Theorem to prove Carter's baseball cards total 152 given the conditions.
theorem carterHas152Cards (h1 : marcusCards = 210) (h2 : marcusCards = carterCards + 58) : carterCards = 152 :=
by
  -- Proof omitted for this exercise
  sorry

end carterHas152Cards_l1138_113868


namespace calculate_B_l1138_113849
open Real

theorem calculate_B 
  (A B : ℝ) 
  (a b : ℝ) 
  (hA : A = π / 6) 
  (ha : a = 1) 
  (hb : b = sqrt 3) 
  (h_sin_relation : sin B = (b * sin A) / a) : 
  (B = π / 3 ∨ B = 2 * π / 3) :=
sorry

end calculate_B_l1138_113849


namespace number_of_12_digit_numbers_with_consecutive_digits_same_l1138_113870

theorem number_of_12_digit_numbers_with_consecutive_digits_same : 
  let total := (2 : ℕ) ^ 12
  let excluded := 2
  total - excluded = 4094 :=
by
  let total := (2 : ℕ) ^ 12
  let excluded := 2
  have h : total = 4096 := by norm_num
  have h' : total - excluded = 4094 := by norm_num
  exact h'

end number_of_12_digit_numbers_with_consecutive_digits_same_l1138_113870


namespace find_smaller_number_l1138_113845

def smaller_number (x y : ℕ) : ℕ :=
  if x < y then x else y

theorem find_smaller_number (a b : ℕ) (h1 : a + b = 64) (h2 : a = b + 12) : smaller_number a b = 26 :=
by
  sorry

end find_smaller_number_l1138_113845


namespace unique_solution_of_system_of_equations_l1138_113808
open Set

variable {α : Type*} (A B X : Set α)

theorem unique_solution_of_system_of_equations :
  (X ∩ (A ∪ B) = X) ∧
  (A ∩ (B ∪ X) = A) ∧
  (B ∩ (A ∪ X) = B) ∧
  (X ∩ A ∩ B = ∅) →
  (X = (A \ B) ∪ (B \ A)) :=
by
  sorry

end unique_solution_of_system_of_equations_l1138_113808


namespace taxi_ride_cost_l1138_113844

-- Define the base fare
def base_fare : ℝ := 2.00

-- Define the cost per mile
def cost_per_mile : ℝ := 0.30

-- Define the distance traveled
def distance : ℝ := 8.00

-- Define the total cost function
def total_cost (base : ℝ) (per_mile : ℝ) (miles : ℝ) : ℝ :=
  base + (per_mile * miles)

-- The statement to prove: the total cost of an 8-mile taxi ride
theorem taxi_ride_cost : total_cost base_fare cost_per_mile distance = 4.40 :=
by
sorry

end taxi_ride_cost_l1138_113844


namespace expenditure_ratio_l1138_113886

variable (P1 P2 : Type)
variable (I1 I2 E1 E2 : ℝ)
variable (R_incomes : I1 / I2 = 5 / 4)
variable (S1 S2 : ℝ)
variable (S_equal : S1 = S2)
variable (I1_fixed : I1 = 4000)
variable (Savings : S1 = 1600)

theorem expenditure_ratio :
  (I1 - E1 = 1600) → 
  (I2 * 4 / 5 - E2 = 1600) →
  I2 = 3200 →
  E1 / E2 = 3 / 2 :=
by
  intro P1_savings P2_savings I2_calc
  -- proof steps go here
  sorry

end expenditure_ratio_l1138_113886


namespace find_d10_bills_l1138_113865

variable (V : Int) (d10 d20 : Int)

-- Given conditions
def spent_money (d10 d20 : Int) : Int := 10 * d10 + 20 * d20

axiom spent_amount : spent_money d10 d20 = 80
axiom more_20_bills : d20 = d10 + 1

-- Question to prove
theorem find_d10_bills : d10 = 2 :=
by {
  -- We mark the theorem to be proven
  sorry
}

end find_d10_bills_l1138_113865


namespace exists_trinomial_with_exponents_three_l1138_113814

theorem exists_trinomial_with_exponents_three (x y : ℝ) :
  ∃ (a b c : ℝ) (t1 t2 t3 : ℕ × ℕ), 
  t1.1 + t1.2 = 3 ∧ t2.1 + t2.2 = 3 ∧ t3.1 + t3.2 = 3 ∧
  (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) ∧
  (a * x ^ t1.1 * y ^ t1.2 + b * x ^ t2.1 * y ^ t2.2 + c * x ^ t3.1 * y ^ t3.2 ≠ 0) := sorry

end exists_trinomial_with_exponents_three_l1138_113814
