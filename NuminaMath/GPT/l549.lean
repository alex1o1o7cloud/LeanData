import Mathlib

namespace cube_surface_area_increase_l549_54935

theorem cube_surface_area_increase (s : ℝ) : 
    let initial_area := 6 * s^2
    let new_edge := 1.3 * s
    let new_area := 6 * (new_edge)^2
    let incr_area := new_area - initial_area
    let percentage_increase := (incr_area / initial_area) * 100
    percentage_increase = 69 :=
by
  let initial_area := 6 * s^2
  let new_edge := 1.3 * s
  let new_area := 6 * (new_edge)^2
  let incr_area := new_area - initial_area
  let percentage_increase := (incr_area / initial_area) * 100
  sorry

end cube_surface_area_increase_l549_54935


namespace original_average_l549_54902

theorem original_average (n : ℕ) (k : ℕ) (new_avg : ℝ) 
  (h1 : n = 35) 
  (h2 : k = 5) 
  (h3 : new_avg = 125) : 
  (new_avg / k) = 25 :=
by
  rw [h2, h3]
  simp
  sorry

end original_average_l549_54902


namespace area_of_region_l549_54920

def plane_region (x y : ℝ) : Prop := |x| ≤ 1 ∧ |y| ≤ 1

def inequality_holds (a b : ℝ) : Prop := ∀ x y : ℝ, plane_region x y → a * x - 2 * b * y ≤ 2

theorem area_of_region (a b : ℝ) (h : inequality_holds a b) : 
  (-2 ≤ a ∧ a ≤ 2) ∧ (-1 ≤ b ∧ b ≤ 1) ∧ (4 * 2 = 8) :=
sorry

end area_of_region_l549_54920


namespace smallest_integer_neither_prime_nor_square_no_prime_factor_less_than_60_l549_54964

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, 1 < m ∧ m < n → ¬(m ∣ n)
def is_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n
def has_prime_factor_less_than (n k : ℕ) : Prop := ∃ p : ℕ, p < k ∧ is_prime p ∧ p ∣ n

theorem smallest_integer_neither_prime_nor_square_no_prime_factor_less_than_60 :
  ∃ m : ℕ, 
    m = 4091 ∧ 
    ¬is_prime m ∧ 
    ¬is_square m ∧ 
    ¬has_prime_factor_less_than m 60 ∧ 
    (∀ n : ℕ, ¬is_prime n ∧ ¬is_square n ∧ ¬has_prime_factor_less_than n 60 → 4091 ≤ n) :=
by
  sorry

end smallest_integer_neither_prime_nor_square_no_prime_factor_less_than_60_l549_54964


namespace perpendicular_vectors_dot_product_zero_l549_54971

theorem perpendicular_vectors_dot_product_zero (m : ℝ) :
  let a := (1, 2)
  let b := (m + 1, -m)
  (a.1 * b.1 + a.2 * b.2 = 0) → m = 1 :=
by
  intros a b h_eq
  sorry

end perpendicular_vectors_dot_product_zero_l549_54971


namespace binom_12_9_eq_220_l549_54995

open Nat

theorem binom_12_9_eq_220 : Nat.choose 12 9 = 220 := by
  sorry

end binom_12_9_eq_220_l549_54995


namespace price_of_battery_l549_54939

def cost_of_tire : ℕ := 42
def cost_of_tires (num_tires : ℕ) : ℕ := num_tires * cost_of_tire
def total_cost : ℕ := 224
def num_tires : ℕ := 4
def cost_of_battery : ℕ := total_cost - cost_of_tires num_tires

theorem price_of_battery : cost_of_battery = 56 := by
  sorry

end price_of_battery_l549_54939


namespace polygon_perimeter_exposure_l549_54903

theorem polygon_perimeter_exposure:
  let triangle_sides := 3
  let square_sides := 4
  let pentagon_sides := 5
  let hexagon_sides := 6
  let heptagon_sides := 7
  let octagon_sides := 8
  let nonagon_sides := 9
  let exposure_triangle_nonagon := triangle_sides + nonagon_sides - 2
  let other_polygons_adjacency := 2 * 5
  let exposure_other_polygons := square_sides + pentagon_sides + hexagon_sides + heptagon_sides + octagon_sides - other_polygons_adjacency
  exposure_triangle_nonagon + exposure_other_polygons = 30 :=
by sorry

end polygon_perimeter_exposure_l549_54903


namespace chairs_bought_l549_54929

theorem chairs_bought (C : ℕ) (tables chairs total_time time_per_furniture : ℕ)
  (h1 : tables = 4)
  (h2 : time_per_furniture = 6)
  (h3 : total_time = 48)
  (h4 : total_time = time_per_furniture * (tables + chairs)) :
  C = 4 :=
by
  -- proof steps are omitted
  sorry

end chairs_bought_l549_54929


namespace num_of_3_digit_nums_with_one_even_digit_l549_54906

def is_even (n : Nat) : Bool :=
  n % 2 == 0

def count_3_digit_nums_with_exactly_one_even_digit : Nat :=
  let even_digits := [0, 2, 4, 6, 8]
  let odd_digits := [1, 3, 5, 7, 9]
  -- Case 1: A is even, B and C are odd
  let case1 := 4 * 5 * 5
  -- Case 2: B is even, A and C are odd
  let case2 := 5 * 5 * 5
  -- Case 3: C is even, A and B are odd
  let case3 := 5 * 5 * 5
  case1 + case2 + case3

theorem num_of_3_digit_nums_with_one_even_digit : count_3_digit_nums_with_exactly_one_even_digit = 350 := by
  sorry

end num_of_3_digit_nums_with_one_even_digit_l549_54906


namespace solution_of_inequality_l549_54980

-- Let us define the inequality and the solution set
def inequality (x : ℝ) := (x - 1)^2023 - 2^2023 * x^2023 ≤ x + 1
def solution_set (x : ℝ) := x ≥ -1

-- The theorem statement to prove that the solution set matches the inequality
theorem solution_of_inequality :
  {x : ℝ | inequality x} = {x : ℝ | solution_set x} := sorry

end solution_of_inequality_l549_54980


namespace parabola_equation_l549_54962

theorem parabola_equation (p : ℝ) (h : 2 * p = 8) :
  ∃ (a : ℝ), a = 8 ∧ (y^2 = a * x ∨ y^2 = -a * x) :=
by
  sorry

end parabola_equation_l549_54962


namespace pet_store_profit_is_205_l549_54942

def brandon_selling_price : ℤ := 100
def pet_store_selling_price : ℤ := 5 + 3 * brandon_selling_price
def pet_store_profit : ℤ := pet_store_selling_price - brandon_selling_price

theorem pet_store_profit_is_205 :
  pet_store_profit = 205 := by
  sorry

end pet_store_profit_is_205_l549_54942


namespace scenario1_winner_scenario2_winner_l549_54917

def optimal_play_winner1 (n : ℕ) (start_by_anna : Bool := true) : String :=
  if n % 6 = 0 then "Balázs"
  else "Anna"

def optimal_play_winner2 (n : ℕ) (start_by_anna : Bool := true) : String :=
  if n % 4 = 0 then "Balázs"
  else "Anna"

theorem scenario1_winner:
  optimal_play_winner1 39 true = "Balázs" :=
by 
  sorry

theorem scenario2_winner:
  optimal_play_winner2 39 true = "Anna" :=
by
  sorry

end scenario1_winner_scenario2_winner_l549_54917


namespace ruth_gave_janet_53_stickers_l549_54982

-- Definitions: Janet initially has 3 stickers, after receiving more from Ruth, she has 56 stickers in total.
def janet_initial : ℕ := 3
def janet_total : ℕ := 56

-- The statement to prove: Ruth gave Janet 53 stickers.
def stickers_from_ruth (initial: ℕ) (total: ℕ) : ℕ :=
  total - initial

theorem ruth_gave_janet_53_stickers : stickers_from_ruth janet_initial janet_total = 53 :=
by sorry

end ruth_gave_janet_53_stickers_l549_54982


namespace triangle_right_if_angle_difference_l549_54956

noncomputable def is_right_triangle (A B C : ℝ) : Prop := 
  A = 90

theorem triangle_right_if_angle_difference (A B C : ℝ) (h : A - B = C) (sum_angles : A + B + C = 180) :
  is_right_triangle A B C :=
  sorry

end triangle_right_if_angle_difference_l549_54956


namespace commute_times_variance_l549_54998

theorem commute_times_variance (x y : ℝ) :
  (x + y + 10 + 11 + 9) / 5 = 10 ∧
  ((x - 10)^2 + (y - 10)^2 + (10 - 10)^2 + (11 - 10)^2 + (9 - 10)^2) / 5 = 2 →
  |x - y| = 4 :=
by
  sorry

end commute_times_variance_l549_54998


namespace total_black_balls_l549_54912

-- Conditions
def number_of_white_balls (B : ℕ) : ℕ := 6 * B

def total_balls (B : ℕ) : ℕ := B + number_of_white_balls B

-- Theorem to prove
theorem total_black_balls (h : total_balls B = 56) : B = 8 :=
by
  sorry

end total_black_balls_l549_54912


namespace opposite_of_two_thirds_l549_54907

theorem opposite_of_two_thirds : - (2/3) = -2/3 :=
by
  sorry

end opposite_of_two_thirds_l549_54907


namespace rectangles_with_one_gray_cell_l549_54950

/- Definitions from conditions -/
def total_gray_cells : ℕ := 40
def blue_cells : ℕ := 36
def red_cells : ℕ := 4

/- The number of rectangles containing exactly one gray cell is the proof goal -/
theorem rectangles_with_one_gray_cell :
  (blue_cells * 4 + red_cells * 8) = 176 :=
sorry

end rectangles_with_one_gray_cell_l549_54950


namespace ratio_female_male_l549_54983

theorem ratio_female_male (f m : ℕ) 
  (h1 : (50 * f) / f = 50) 
  (h2 : (30 * m) / m = 30) 
  (h3 : (50 * f + 30 * m) / (f + m) = 35) : 
  f / m = 1 / 3 := 
by
  sorry

end ratio_female_male_l549_54983


namespace arithmetic_sequence_k_value_l549_54975

theorem arithmetic_sequence_k_value (a_1 d : ℕ) (h1 : a_1 = 1) (h2 : d = 2) (k : ℕ) (S : ℕ → ℕ) (h_sum : ∀ n, S n = n * (2 * a_1 + (n - 1) * d) / 2) (h_condition : S (k + 2) - S k = 24) : k = 5 :=
by {
  sorry
}

end arithmetic_sequence_k_value_l549_54975


namespace find_a_l549_54994

noncomputable def f (a x : ℝ) : ℝ := a * x^2 + 2 * a * x + 1

theorem find_a
  (a : ℝ)
  (h₁ : ∀ x ∈ Set.Icc (-3 : ℝ) (2 : ℝ), f a x ≤ 4)
  (h₂ : ∃ x ∈ Set.Icc (-3 : ℝ) (2 : ℝ), f a x = 4) :
  a = -3 ∨ a = 3 / 8 :=
by
  sorry

end find_a_l549_54994


namespace original_savings_l549_54904

-- Define original savings as a variable
variable (S : ℝ)

-- Define the condition that 1/4 of the savings equals 200
def tv_cost_condition : Prop := (1 / 4) * S = 200

-- State the theorem that if the condition is satisfied, then the original savings are 800
theorem original_savings (h : tv_cost_condition S) : S = 800 :=
by
  sorry

end original_savings_l549_54904


namespace minimum_n_minus_m_abs_l549_54941

theorem minimum_n_minus_m_abs (f g : ℝ → ℝ)
  (hf : ∀ x, f x = Real.exp x + 2 * x)
  (hg : ∀ x, g x = 4 * x)
  (m n : ℝ)
  (h_cond : f m = g n) :
  |n - m| = (1 / 2) - (1 / 2) * Real.log 2 := 
sorry

end minimum_n_minus_m_abs_l549_54941


namespace profit_calculation_l549_54923

-- Define conditions based on investments
def JohnInvestment := 700
def MikeInvestment := 300

-- Define the equality condition where John received $800 more than Mike
theorem profit_calculation (P : ℝ) 
  (h1 : (P / 6 + (7 / 10) * (2 * P / 3)) - (P / 6 + (3 / 10) * (2 * P / 3)) = 800) : 
  P = 3000 := 
sorry

end profit_calculation_l549_54923


namespace racket_price_l549_54966

theorem racket_price (cost_sneakers : ℕ) (cost_outfit : ℕ) (total_spent : ℕ) 
  (h_sneakers : cost_sneakers = 200) 
  (h_outfit : cost_outfit = 250) 
  (h_total : total_spent = 750) : 
  (total_spent - cost_sneakers - cost_outfit) = 300 :=
sorry

end racket_price_l549_54966


namespace mike_net_spending_l549_54954

-- Definitions for given conditions
def trumpet_cost : ℝ := 145.16
def song_book_revenue : ℝ := 5.84

-- Theorem stating the result
theorem mike_net_spending : trumpet_cost - song_book_revenue = 139.32 :=
by 
  sorry

end mike_net_spending_l549_54954


namespace greatest_possible_integer_l549_54948

theorem greatest_possible_integer (m : ℕ) (h1 : m < 150) (h2 : ∃ a : ℕ, m = 10 * a - 2) (h3 : ∃ b : ℕ, m = 9 * b - 4) : m = 68 := 
  by sorry

end greatest_possible_integer_l549_54948


namespace Jia_age_is_24_l549_54963

variable (Jia Yi Bing Ding : ℕ)

theorem Jia_age_is_24
  (h1 : (Jia + Yi + Bing) / 3 = (Jia + Yi + Bing + Ding) / 4 + 1)
  (h2 : (Jia + Yi) / 2 = (Jia + Yi + Bing) / 3 + 1)
  (h3 : Jia = Yi + 4)
  (h4 : Ding = 17) :
  Jia = 24 :=
by
  sorry

end Jia_age_is_24_l549_54963


namespace helen_chocolate_chip_cookies_l549_54952

theorem helen_chocolate_chip_cookies :
  let cookies_yesterday := 527
  let cookies_morning := 554
  cookies_yesterday + cookies_morning = 1081 :=
by
  let cookies_yesterday := 527
  let cookies_morning := 554
  show cookies_yesterday + cookies_morning = 1081
  -- The proof is omitted according to the provided instructions 
  sorry

end helen_chocolate_chip_cookies_l549_54952


namespace geometric_sequence_sum_l549_54943

-- Define the problem conditions and the result
theorem geometric_sequence_sum :
  ∃ (a : ℕ → ℝ), a 1 + a 2 = 16 ∧ a 3 + a 4 = 24 → a 7 + a 8 = 54 :=
by
  -- Preliminary steps and definitions to prove the theorem
  sorry

end geometric_sequence_sum_l549_54943


namespace log_sum_eq_five_l549_54993

variable {a : ℕ → ℝ}

-- Conditions from the problem
def geometric_seq (a : ℕ → ℝ) : Prop :=
∀ n, a (n + 1) = 3 * a n 

def sum_condition (a : ℕ → ℝ) : Prop :=
a 2 + a 4 + a 9 = 9

-- The mathematical statement to prove
theorem log_sum_eq_five (h1 : geometric_seq a) (h2 : sum_condition a) :
  Real.logb 3 (a 5 + a 7 + a 9) = 5 := 
sorry

end log_sum_eq_five_l549_54993


namespace increase_is_50_percent_l549_54938

theorem increase_is_50_percent (original new : ℕ) (h1 : original = 60) (h2 : new = 90) :
  ((new - original) * 100 / original) = 50 :=
by
  -- Proof can be filled here.
  sorry

end increase_is_50_percent_l549_54938


namespace perp_DM_PN_l549_54991

-- Definitions of the triangle and its elements
variables {A B C M N P D : Point}
variables (triangle_incircle_touch : ∀ (A B C : Point) (triangle : Triangle ABC),
  touches_incircle_at triangle B C M ∧ 
  touches_incircle_at triangle C A N ∧ 
  touches_incircle_at triangle A B P)
variables (point_D : lies_on_segment D N P)
variables {BD CD DP DN : ℝ}
variables (ratio_condition : DP / DN = BD / CD)

-- The theorem statement
theorem perp_DM_PN 
  (h1 : triangle_incircle_touch A B C) 
  (h2 : point_D)
  (h3 : ratio_condition) : 
  is_perpendicular D M P N := 
sorry

end perp_DM_PN_l549_54991


namespace fractional_part_exceeds_bound_l549_54973

noncomputable def x (a b : ℕ) : ℝ := Real.sqrt a + Real.sqrt b

theorem fractional_part_exceeds_bound
  (a b : ℕ)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hx_not_int : ¬ (∃ n : ℤ, x a b = n))
  (hx_lt : x a b < 1976) :
    x a b % 1 > 3.24e-11 :=
sorry

end fractional_part_exceeds_bound_l549_54973


namespace geometric_sequence_b_general_term_a_l549_54934

-- Definitions of sequences and given conditions
def a (n : ℕ) : ℕ := sorry -- The sequence a_n
def S (n : ℕ) : ℕ := sorry -- The sum of the first n terms S_n

axiom a1_condition : a 1 = 2
axiom recursion_formula (n : ℕ): S (n+1) = 4 * a n + 2

def b (n : ℕ) : ℕ := a (n+1) - 2 * a n -- Definition of b_n

-- Theorem 1: Prove that b_n is a geometric sequence
theorem geometric_sequence_b (n : ℕ) : ∃ q, ∀ m, b (m+1) = q * b m :=
  sorry

-- Theorem 2: Find the general term formula for a_n
theorem general_term_a (n : ℕ) : a n = n * 2^n :=
  sorry

end geometric_sequence_b_general_term_a_l549_54934


namespace task1_task2_task3_l549_54900

noncomputable def f (x a : ℝ) := x^2 - 4 * x + a + 3
noncomputable def g (x m : ℝ) := m * x + 5 - 2 * m

theorem task1 (a m : ℝ) (h₁ : a = -3) (h₂ : m = 0) :
  (∃ x : ℝ, f x a - g x m = 0) ↔ x = -1 ∨ x = 5 :=
sorry

theorem task2 (a : ℝ) :
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → f x a = 0) ↔ -8 ≤ a ∧ a ≤ 0 :=
sorry

theorem task3 (m : ℝ) :
  (∀ x₁ : ℝ, 1 ≤ x₁ ∧ x₁ ≤ 4 → ∃ x₂ : ℝ, 1 ≤ x₂ ∧ x₂ ≤ 4 ∧ f x₁ 0 = g x₂ m) ↔ m ≤ -3 ∨ 6 ≤ m :=
sorry

end task1_task2_task3_l549_54900


namespace geo_sequence_necessity_l549_54953

theorem geo_sequence_necessity (a1 a2 a3 a4 : ℝ) (h_non_zero: a1 ≠ 0 ∧ a2 ≠ 0 ∧ a3 ≠ 0 ∧ a4 ≠ 0) :
  (a1 * a4 = a2 * a3) → (∀ r : ℝ, (a2 = a1 * r) ∧ (a3 = a2 * r) ∧ (a4 = a3 * r)) → False :=
sorry

end geo_sequence_necessity_l549_54953


namespace find_t_l549_54901

-- Given a quadratic equation
def quadratic_eq (x : ℝ) := 4 * x ^ 2 - 16 * x - 200

-- Completing the square to find t
theorem find_t : ∃ q t : ℝ, (x : ℝ) → (quadratic_eq x = 0) → (x + q) ^ 2 = t ∧ t = 54 :=
by
  sorry

end find_t_l549_54901


namespace minimum_value_l549_54967

theorem minimum_value (x y z : ℝ) (h : x + y + z = 1) : 2 * x^2 + y^2 + 3 * z^2 ≥ 3 / 7 := by
  sorry

end minimum_value_l549_54967


namespace TylerWeightDifference_l549_54985

-- Define the problem conditions
def PeterWeight : ℕ := 65
def SamWeight : ℕ := 105
def TylerWeight := 2 * PeterWeight

-- State the theorem
theorem TylerWeightDifference : (TylerWeight - SamWeight = 25) :=
by
  -- proof goes here
  sorry

end TylerWeightDifference_l549_54985


namespace students_in_miss_evans_class_l549_54945

theorem students_in_miss_evans_class
  (total_contribution : ℕ)
  (class_funds : ℕ)
  (contribution_per_student : ℕ)
  (remaining_contribution : ℕ)
  (num_students : ℕ)
  (h1 : total_contribution = 90)
  (h2 : class_funds = 14)
  (h3 : contribution_per_student = 4)
  (h4 : remaining_contribution = total_contribution - class_funds)
  (h5 : num_students = remaining_contribution / contribution_per_student)
  : num_students = 19 :=
sorry

end students_in_miss_evans_class_l549_54945


namespace min_hypotenuse_l549_54924

theorem min_hypotenuse {a b : ℝ} (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : a + b = 10) :
  ∃ c : ℝ, c = Real.sqrt (a^2 + b^2) ∧ c ≥ 5 * Real.sqrt 2 :=
by
  sorry

end min_hypotenuse_l549_54924


namespace strictly_increasing_difference_l549_54911

variable {a b : ℝ}
variable {f g : ℝ → ℝ}

theorem strictly_increasing_difference
  (h_diff : ∀ x ∈ Set.Icc a b, DifferentiableAt ℝ f x ∧ DifferentiableAt ℝ g x)
  (h_eq : f a = g a)
  (h_diff_ineq : ∀ x ∈ Set.Ioo a b, (deriv f x : ℝ) > (deriv g x : ℝ)) :
  ∀ x ∈ Set.Ioo a b, f x > g x := by
  sorry

end strictly_increasing_difference_l549_54911


namespace cubic_polynomial_roots_l549_54908

variables (a b c : ℚ)

theorem cubic_polynomial_roots (a b c : ℚ) :
  (c = 0 → ∃ x y z : ℚ, (x = 0 ∧ y = 1 ∧ z = -2) ∧ (-a = x + y + z) ∧ (b = x*y + y*z + z*x) ∧ (-c = x*y*z)) ∧
  (c ≠ 0 → ∃ x y z : ℚ, (x = 1 ∧ y = -1 ∧ z = -1) ∧ (-a = x + y + z) ∧ (b = x*y + y*z + z*x) ∧ (-c = x*y*z)) :=
by
  sorry

end cubic_polynomial_roots_l549_54908


namespace exterior_angle_BAC_l549_54989

theorem exterior_angle_BAC (angle_octagon angle_rectangle : ℝ) (h_oct_135 : angle_octagon = 135) (h_rec_90 : angle_rectangle = 90) :
  360 - (angle_octagon + angle_rectangle) = 135 := 
by
  simp [h_oct_135, h_rec_90]
  sorry

end exterior_angle_BAC_l549_54989


namespace min_value_expression_l549_54937

theorem min_value_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  ∃ (m : ℝ), m = 3 / 2 ∧ ∀ t > 0, (2 * x / (x + 2 * y) + y / x) ≥ m :=
by
  use 3 / 2
  sorry

end min_value_expression_l549_54937


namespace focus_of_curve_is_4_0_l549_54986

noncomputable def is_focus (p : ℝ × ℝ) (curve : ℝ × ℝ → Prop) : Prop :=
  ∃ c : ℝ, ∀ x y : ℝ, curve (x, y) ↔ (y^2 = -16 * c * (x - 4))

def curve (p : ℝ × ℝ) : Prop := p.2^2 = -16 * p.1 + 64

theorem focus_of_curve_is_4_0 : is_focus (4, 0) curve :=
by
sorry

end focus_of_curve_is_4_0_l549_54986


namespace max_sin_a_given_sin_a_plus_b_l549_54997

theorem max_sin_a_given_sin_a_plus_b (a b : ℝ) (sin_add : Real.sin (a + b) = Real.sin a + Real.sin b) : 
  Real.sin a ≤ 1 := 
sorry

end max_sin_a_given_sin_a_plus_b_l549_54997


namespace hannah_total_spent_l549_54932

-- Definitions based on conditions
def sweatshirts_bought : ℕ := 3
def t_shirts_bought : ℕ := 2
def cost_per_sweatshirt : ℕ := 15
def cost_per_t_shirt : ℕ := 10

-- Definition of the theorem that needs to be proved
theorem hannah_total_spent : 
  (sweatshirts_bought * cost_per_sweatshirt + t_shirts_bought * cost_per_t_shirt) = 65 :=
by
  sorry

end hannah_total_spent_l549_54932


namespace find_a_values_l549_54918

theorem find_a_values (a x₁ x₂ : ℝ) (h1 : x^2 + a * x - 2 = 0)
                      (h2 : x₁ ≠ x₂)
                      (h3 : x₁^3 + 22 / x₂ = x₂^3 + 22 / x₁) :
                      a = 3 ∨ a = -3 :=
by
  sorry

end find_a_values_l549_54918


namespace lcm_one_to_twelve_l549_54926

theorem lcm_one_to_twelve : 
  Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 
  (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 (Nat.lcm 10 (Nat.lcm 11 12)))))))))) = 27720 := 
by sorry

end lcm_one_to_twelve_l549_54926


namespace inequality_solution_l549_54972

theorem inequality_solution (x : ℝ) (hx : x > 0) : 
  x + 1 / x ≥ 2 ∧ (x + 1 / x = 2 ↔ x = 1) := 
by
  sorry

end inequality_solution_l549_54972


namespace solve_first_equation_solve_second_equation_l549_54951

theorem solve_first_equation (x : ℝ) : (8 * x = -2 * (x + 5)) → (x = -1) :=
by
  intro h
  sorry

theorem solve_second_equation (x : ℝ) : ((x - 1) / 4 = (5 * x - 7) / 6 + 1) → (x = -1 / 7) :=
by
  intro h
  sorry

end solve_first_equation_solve_second_equation_l549_54951


namespace coordinates_of_P_l549_54930

open Real

theorem coordinates_of_P (P : ℝ × ℝ) (h1 : P.1 = 2 * cos (2 * π / 3)) (h2 : P.2 = 2 * sin (2 * π / 3)) :
  P = (-1, sqrt 3) :=
by
  sorry

end coordinates_of_P_l549_54930


namespace fixed_point_l549_54987

theorem fixed_point (a : ℝ) (h : a > 0 ∧ a ≠ 1) : (1, 4) ∈ {p : ℝ × ℝ | ∃ x, p = (x, a^(x-1) + 3)} :=
by
  sorry

end fixed_point_l549_54987


namespace Kendall_dimes_l549_54960

theorem Kendall_dimes (total_value : ℝ) (quarters : ℝ) (dimes : ℝ) (nickels : ℝ) 
  (num_quarters : ℕ) (num_nickels : ℕ) 
  (total_amount : total_value = 4)
  (quarter_amount : quarters = num_quarters * 0.25)
  (num_quarters_val : num_quarters = 10)
  (nickel_amount : nickels = num_nickels * 0.05) 
  (num_nickels_val : num_nickels = 6) :
  dimes = 12 := by
  sorry

end Kendall_dimes_l549_54960


namespace inequality_1_inequality_3_l549_54977

variable (a b : ℝ)
variable (hab : a > b ∧ b ≥ 2)

theorem inequality_1 (hab : a > b ∧ b ≥ 2) : b ^ 2 > 3 * b - a :=
by sorry

theorem inequality_3 (hab : a > b ∧ b ≥ 2) : a * b > a + b :=
by sorry

end inequality_1_inequality_3_l549_54977


namespace problem_1_problem_2_l549_54925

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 / 6 + 1 / x - a * Real.log x

theorem problem_1 (a : ℝ) (h : 0 < a) :
  (∀ x : ℝ, 0 < x ∧ x ≤ 3 → f x a ≤ f 3 a) → a ≥ 8 / 3 :=
sorry

theorem problem_2 (a : ℝ) (h1 : 0 < a) (x0 : ℝ) :
  (∃! t : ℝ, 0 < t ∧ f t a = 0) → Real.log x0 = (x0^3 + 6) / (2 * (x0^3 - 3)) :=
sorry

end problem_1_problem_2_l549_54925


namespace rabbit_carrots_l549_54981

theorem rabbit_carrots (h_r h_f x : ℕ) (H1 : 5 * h_r = x) (H2 : 6 * h_f = x) (H3 : h_r = h_f + 2) : x = 60 :=
by
  sorry

end rabbit_carrots_l549_54981


namespace adult_ticket_cost_is_19_l549_54958

variable (A : ℕ) -- the cost for an adult ticket
def child_ticket_cost : ℕ := 15
def total_receipts : ℕ := 7200
def total_attendance : ℕ := 400
def adults_attendance : ℕ := 280
def children_attendance : ℕ := 120

-- The equation representing the total receipts
theorem adult_ticket_cost_is_19 (h : total_receipts = 280 * A + 120 * child_ticket_cost) : A = 19 :=
  by sorry

end adult_ticket_cost_is_19_l549_54958


namespace symmetric_point_coordinates_l549_54921

-- Define the type for 3D points
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the symmetric point function with respect to the x-axis
def symmetricPointWithRespectToXAxis (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y, z := -p.z }

-- Define the specific point
def givenPoint : Point3D := { x := 2, y := 3, z := 4 }

-- State the theorem to be proven
theorem symmetric_point_coordinates : 
  symmetricPointWithRespectToXAxis givenPoint = { x := 2, y := -3, z := -4 } :=
by
  sorry

end symmetric_point_coordinates_l549_54921


namespace find_pq_l549_54996

theorem find_pq (p q : ℝ) (hp : 0 < p) (hq : 0 < q) (hline : ∀ x y : ℝ, px + qy = 24) 
  (harea : (1 / 2) * (24 / p) * (24 / q) = 48) : p * q = 12 :=
by
  sorry

end find_pq_l549_54996


namespace sum_of_consecutive_integers_l549_54916

theorem sum_of_consecutive_integers (x y z : ℤ) (h1 : y = x + 1) (h2 : z = y + 1) (h3 : z = 12) :
  x + y + z = 33 :=
sorry

end sum_of_consecutive_integers_l549_54916


namespace num_parallel_edge_pairs_correct_l549_54913

-- Define a rectangular prism with given dimensions
structure RectangularPrism where
  length : ℕ
  width : ℕ
  height : ℕ

-- Function to count the number of pairs of parallel edges
def num_parallel_edge_pairs (p : RectangularPrism) : ℕ :=
  4 * ((p.length + p.width + p.height) - 3)

-- Given conditions
def given_prism : RectangularPrism := { length := 4, width := 3, height := 2 }

-- Main theorem statement
theorem num_parallel_edge_pairs_correct :
  num_parallel_edge_pairs given_prism = 12 :=
by
  -- Skipping proof steps
  sorry

end num_parallel_edge_pairs_correct_l549_54913


namespace complex_inv_condition_l549_54949

theorem complex_inv_condition (i : ℂ) (h : i^2 = -1) : (i - 2 * i⁻¹)⁻¹ = -i / 3 :=
by
  sorry

end complex_inv_condition_l549_54949


namespace division_problem_l549_54961

theorem division_problem
  (R : ℕ) (D : ℕ) (Q : ℕ) (Div : ℕ)
  (hR : R = 5)
  (hD1 : D = 3 * Q)
  (hD2 : D = 3 * R + 3) :
  Div = D * Q + R :=
by
  have hR : R = 5 := hR
  have hD2 := hD2
  have hDQ := hD1
  -- Proof continues with steps leading to the final desired conclusion
  sorry

end division_problem_l549_54961


namespace balcony_more_than_orchestra_l549_54905

variables (O B : ℕ) (H1 : O + B = 380) (H2 : 12 * O + 8 * B = 3320)

theorem balcony_more_than_orchestra : B - O = 240 :=
by sorry

end balcony_more_than_orchestra_l549_54905


namespace necessary_and_sufficient_condition_l549_54965

variable (x a : ℝ)

-- Condition 1: For all x in [1, 2], x^2 - a ≥ 0
def condition1 (x a : ℝ) : Prop := 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0

-- Condition 2: There exists an x in ℝ such that x^2 + 2ax + 2 - a = 0
def condition2 (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0

-- Proof problem: The necessary and sufficient condition for p ∧ q is a ≤ -2 ∨ a = 1
theorem necessary_and_sufficient_condition (a : ℝ) :
  (∀ x, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0) ∧ (∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0) ↔ (a ≤ -2 ∨ a = 1) :=
sorry

end necessary_and_sufficient_condition_l549_54965


namespace feasible_stations_l549_54928

theorem feasible_stations (n : ℕ) (h: n > 0) 
  (pairings : ∀ (i j : ℕ), i ≠ j → i < n → j < n → ∃ k, (i+k) % n = j ∨ (j+k) % n = i) : n = 4 :=
sorry

end feasible_stations_l549_54928


namespace jacob_three_heads_probability_l549_54946

noncomputable section

def probability_three_heads_after_two_tails : ℚ := 1 / 96

theorem jacob_three_heads_probability :
  let p := (1 / 2) ^ 4 * (1 / 6)
  p = probability_three_heads_after_two_tails := by
sorry

end jacob_three_heads_probability_l549_54946


namespace Nancy_picked_l549_54992

def Alyssa_picked : ℕ := 42
def Total_picked : ℕ := 59

theorem Nancy_picked : Total_picked - Alyssa_picked = 17 := by
  sorry

end Nancy_picked_l549_54992


namespace simplify_cube_root_18_24_30_l549_54979

noncomputable def cube_root_simplification (a b c : ℕ) : ℕ :=
  let sum_cubes := a^3 + b^3 + c^3
  36

theorem simplify_cube_root_18_24_30 : 
  cube_root_simplification 18 24 30 = 36 :=
by {
  -- Proof steps would go here
  sorry
}

end simplify_cube_root_18_24_30_l549_54979


namespace all_visitors_can_buy_ticket_l549_54936

-- Define the coin types
inductive Coin
  | Three
  | Five

-- Define a function to calculate the total money from a list of coins
def totalMoney (coins : List Coin) : Int :=
  coins.foldr (fun c acc => acc + (match c with | Coin.Three => 3 | Coin.Five => 5)) 0

-- Define the initial state: each person has 22 tugriks in some combination of 3 and 5 tugrik coins
def initial_money := 22
def ticket_cost := 4

-- Each visitor and the cashier has 22 tugriks initially
axiom visitor_money_all_22 (n : Nat) : n ≤ 200 → totalMoney (List.replicate 2 Coin.Five ++ List.replicate 4 Coin.Three) = initial_money

-- We want to prove that all visitors can buy a ticket
theorem all_visitors_can_buy_ticket :
  ∀ n, n ≤ 200 → ∃ coins: List Coin, totalMoney coins = initial_money ∧ totalMoney coins ≥ ticket_cost := by
    sorry -- Proof goes here

end all_visitors_can_buy_ticket_l549_54936


namespace units_digit_3542_pow_876_l549_54947

theorem units_digit_3542_pow_876 : (3542 ^ 876) % 10 = 6 := by 
  sorry

end units_digit_3542_pow_876_l549_54947


namespace quadratic_root_property_l549_54974

theorem quadratic_root_property (m p : ℝ) 
  (h1 : (p^2 - 2 * p + m - 1 = 0)) 
  (h2 : (p^2 - 2 * p + 3) * (m + 4) = 7)
  (h3 : ∃ (r1 r2 : ℝ), r1 ≠ r2 ∧ r1^2 - 2 * r1 + m - 1 = 0 ∧ r2^2 - 2 * r2 + m - 1 = 0) : 
  m = -3 :=
by 
  sorry

end quadratic_root_property_l549_54974


namespace remainder_of_large_number_l549_54959

theorem remainder_of_large_number :
  (102938475610 % 12) = 10 :=
by
  have h1 : (102938475610 % 4) = 2 := sorry
  have h2 : (102938475610 % 3) = 1 := sorry
  sorry

end remainder_of_large_number_l549_54959


namespace non_poli_sci_gpa_below_or_eq_3_is_10_l549_54933

-- Definitions based on conditions
def total_applicants : ℕ := 40
def poli_sci_majors : ℕ := 15
def gpa_above_3 : ℕ := 20
def poli_sci_gpa_above_3 : ℕ := 5

-- Derived conditions from the problem
def poli_sci_gpa_below_or_eq_3 : ℕ := poli_sci_majors - poli_sci_gpa_above_3
def total_gpa_below_or_eq_3 : ℕ := total_applicants - gpa_above_3
def non_poli_sci_gpa_below_or_eq_3 : ℕ := total_gpa_below_or_eq_3 - poli_sci_gpa_below_or_eq_3

-- Statement to be proven
theorem non_poli_sci_gpa_below_or_eq_3_is_10 : non_poli_sci_gpa_below_or_eq_3 = 10 := by
  sorry

end non_poli_sci_gpa_below_or_eq_3_is_10_l549_54933


namespace population_net_increase_one_day_l549_54955

-- Define the given rates and constants
def birth_rate := 10 -- people per 2 seconds
def death_rate := 2 -- people per 2 seconds
def seconds_per_day := 24 * 60 * 60 -- seconds

-- Define the expected net population increase per second
def population_increase_per_sec := (birth_rate / 2) - (death_rate / 2)

-- Define the expected net population increase per day
def expected_population_increase_per_day := population_increase_per_sec * seconds_per_day

theorem population_net_increase_one_day :
  expected_population_increase_per_day = 345600 := by
  -- This will skip the proof implementation.
  sorry

end population_net_increase_one_day_l549_54955


namespace certain_number_sixth_powers_l549_54968

theorem certain_number_sixth_powers :
  ∃ N, (∀ n : ℕ, n < N → ∃ a : ℕ, n = a^6) ∧
       (∃ m ≤ N, (∀ n < m, ∃ k : ℕ, n = k^6) ∧ ¬ ∃ k : ℕ, m = k^6) :=
sorry

end certain_number_sixth_powers_l549_54968


namespace bad_games_count_l549_54944

/-- 
  Oliver bought a total of 11 video games, and 6 of them worked.
  Prove that the number of bad games he bought is 5.
-/
theorem bad_games_count (total_games : ℕ) (working_games : ℕ) (h1 : total_games = 11) (h2 : working_games = 6) : total_games - working_games = 5 :=
by
  sorry

end bad_games_count_l549_54944


namespace red_marbles_count_l549_54909

theorem red_marbles_count (R : ℕ) (h1 : 48 - R > 0) (h2 : ((48 - R) / 48 : ℚ) * ((48 - R) / 48) = 9 / 16) : R = 12 :=
sorry

end red_marbles_count_l549_54909


namespace stock_AB_increase_factor_l549_54976

-- Define the conditions as mathematical terms
def stock_A_initial := 300
def stock_B_initial := 300
def stock_C_initial := 300
def stock_C_final := stock_C_initial / 2
def total_final := 1350
def AB_combined_initial := stock_A_initial + stock_B_initial
def AB_combined_final := total_final - stock_C_final

-- The statement to prove that the factor by which stocks A and B increased in value is 2.
theorem stock_AB_increase_factor :
  AB_combined_final / AB_combined_initial = 2 :=
  by
    sorry

end stock_AB_increase_factor_l549_54976


namespace rope_length_loss_l549_54978

theorem rope_length_loss
  (stories_needed : ℕ)
  (feet_per_story : ℕ)
  (pieces_of_rope : ℕ)
  (feet_per_rope : ℕ)
  (total_feet_needed : ℕ)
  (total_feet_bought : ℕ)
  (percentage_lost : ℕ) :
  
  stories_needed = 6 →
  feet_per_story = 10 →
  pieces_of_rope = 4 →
  feet_per_rope = 20 →
  total_feet_needed = stories_needed * feet_per_story →
  total_feet_bought = pieces_of_rope * feet_per_rope →
  total_feet_needed <= total_feet_bought →
  percentage_lost = ((total_feet_bought - total_feet_needed) * 100) / total_feet_bought →
  percentage_lost = 25 :=
by
  intros h_stories h_feet_story h_pieces h_feet_rope h_total_needed h_total_bought h_needed_bought h_percentage
  sorry

end rope_length_loss_l549_54978


namespace volunteer_hours_per_year_l549_54957

def volunteer_sessions_per_month := 2
def hours_per_session := 3
def months_per_year := 12

theorem volunteer_hours_per_year : 
  (volunteer_sessions_per_month * months_per_year * hours_per_session) = 72 := 
by
  sorry

end volunteer_hours_per_year_l549_54957


namespace unique_two_digit_number_l549_54984

-- Definition of the problem in Lean
def is_valid_number (n : ℕ) : Prop :=
  n % 4 = 1 ∧ n % 17 = 1 ∧ 10 ≤ n ∧ n ≤ 99

theorem unique_two_digit_number :
  ∃! n : ℕ, is_valid_number n ∧ n = 69 :=
by
  sorry

end unique_two_digit_number_l549_54984


namespace total_painted_surface_area_l549_54915

-- Defining the conditions
def num_cubes := 19
def top_layer := 1
def middle_layer := 5
def bottom_layer := 13
def exposed_faces_top_layer := 5
def exposed_faces_middle_corner := 3
def exposed_faces_middle_center := 1
def exposed_faces_bottom_layer := 1

-- Question: How many square meters are painted?
theorem total_painted_surface_area : 
  let top_layer_area := top_layer * exposed_faces_top_layer
  let middle_layer_area := (4 * exposed_faces_middle_corner) + exposed_faces_middle_center
  let bottom_layer_area := bottom_layer * exposed_faces_bottom_layer
  top_layer_area + middle_layer_area + bottom_layer_area = 31 :=
by
  sorry

end total_painted_surface_area_l549_54915


namespace angle_B_range_l549_54940

theorem angle_B_range (A B C : ℝ) (h1 : A ≤ B) (h2 : B ≤ C) (h3 : A + B + C = 180) (h4 : 2 * B = 5 * A) :
  0 < B ∧ B ≤ 75 :=
by
  sorry

end angle_B_range_l549_54940


namespace water_hydrogen_oxygen_ratio_l549_54922

/-- In a mixture of water with a total mass of 171 grams, 
    where 19 grams are hydrogen, the ratio of hydrogen to oxygen by mass is 1:8. -/
theorem water_hydrogen_oxygen_ratio 
  (h_total_mass : ℝ) 
  (h_mass : ℝ) 
  (o_mass : ℝ) 
  (h_condition : h_total_mass = 171) 
  (h_hydrogen_mass : h_mass = 19) 
  (h_oxygen_mass : o_mass = h_total_mass - h_mass) :
  h_mass / o_mass = 1 / 8 := 
by
  sorry

end water_hydrogen_oxygen_ratio_l549_54922


namespace files_more_than_apps_l549_54988

def initial_apps : ℕ := 11
def initial_files : ℕ := 3
def remaining_apps : ℕ := 2
def remaining_files : ℕ := 24

theorem files_more_than_apps : remaining_files - remaining_apps = 22 :=
by
  sorry

end files_more_than_apps_l549_54988


namespace lunch_break_duration_l549_54990

/-- Define the total recess time as a sum of two 15-minute breaks and one 20-minute break. -/
def total_recess_time : ℕ := 15 + 15 + 20

/-- Define the total time spent outside of class. -/
def total_outside_class_time : ℕ := 80

/-- Prove that the lunch break is 30 minutes long. -/
theorem lunch_break_duration : total_outside_class_time - total_recess_time = 30 :=
by
  sorry

end lunch_break_duration_l549_54990


namespace simplify_expr_l549_54914

-- Define the condition on b
def condition (b : ℚ) : Prop :=
  b ≠ -1 / 2

-- Define the expression to be evaluated
def expression (b : ℚ) : ℚ :=
  1 - 1 / (1 + b / (1 + b))

-- Define the simplified form
def simplified_expr (b : ℚ) : ℚ :=
  b / (1 + 2 * b)

-- The theorem statement showing the equivalence
theorem simplify_expr (b : ℚ) (h : condition b) : expression b = simplified_expr b :=
by
  sorry

end simplify_expr_l549_54914


namespace avg_zits_per_kid_mr_jones_class_l549_54931

-- Define the conditions
def avg_zits_ms_swanson_class := 5
def num_kids_ms_swanson_class := 25
def num_kids_mr_jones_class := 32
def extra_zits_mr_jones_class := 67

-- Define the total number of zits in Ms. Swanson's class
def total_zits_ms_swanson_class := avg_zits_ms_swanson_class * num_kids_ms_swanson_class

-- Define the total number of zits in Mr. Jones' class
def total_zits_mr_jones_class := total_zits_ms_swanson_class + extra_zits_mr_jones_class

-- Define the problem statement to prove: the average number of zits per kid in Mr. Jones' class
theorem avg_zits_per_kid_mr_jones_class : 
  total_zits_mr_jones_class / num_kids_mr_jones_class = 6 := by
  sorry

end avg_zits_per_kid_mr_jones_class_l549_54931


namespace intersection_eq_union_eq_l549_54999

def A := { x : ℝ | x ≥ 2 }
def B := { x : ℝ | 1 < x ∧ x ≤ 4 }

theorem intersection_eq : A ∩ B = { x : ℝ | 2 ≤ x ∧ x ≤ 4 } :=
by sorry

theorem union_eq : A ∪ B = { x : ℝ | 1 < x } :=
by sorry

end intersection_eq_union_eq_l549_54999


namespace problem_7_sqrt_13_l549_54919

theorem problem_7_sqrt_13 : 
  let m := Int.floor (Real.sqrt 13)
  let n := 10 - Real.sqrt 13 - Int.floor (10 - Real.sqrt 13)
  m + n = 7 - Real.sqrt 13 :=
by
  sorry

end problem_7_sqrt_13_l549_54919


namespace cuboid_surface_area_l549_54970

/--
Given a cuboid with length 10 cm, breadth 8 cm, and height 6 cm, the surface area is 376 cm².
-/
theorem cuboid_surface_area 
  (length : ℝ) 
  (breadth : ℝ) 
  (height : ℝ) 
  (h_length : length = 10) 
  (h_breadth : breadth = 8) 
  (h_height : height = 6) : 
  2 * (length * height + length * breadth + breadth * height) = 376 := 
by 
  -- Replace these placeholders with the actual proof steps.
  sorry

end cuboid_surface_area_l549_54970


namespace inversely_proportional_rs_l549_54969

theorem inversely_proportional_rs (r s : ℝ) (k : ℝ) 
(h_invprop : r * s = k) 
(h1 : r = 40) (h2 : s = 5) 
(h3 : s = 8) : r = 25 := by
  sorry

end inversely_proportional_rs_l549_54969


namespace roots_real_roots_equal_l549_54927

noncomputable def discriminant (a : ℝ) : ℝ :=
  let b := 4 * a
  let c := 2 * a^2 - 1 + 3 * a
  b^2 - 4 * 1 * c

theorem roots_real (a : ℝ) : discriminant a ≥ 0 ↔ a ≤ 1/2 ∨ a ≥ 1 := sorry

theorem roots_equal (a : ℝ) : discriminant a = 0 ↔ a = 1 ∨ a = 1/2 := sorry

end roots_real_roots_equal_l549_54927


namespace intersection_S_T_eq_T_l549_54910

-- Definitions based on the given conditions
def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

-- The theorem statement to be proved
theorem intersection_S_T_eq_T : S ∩ T = T := 
sorry

end intersection_S_T_eq_T_l549_54910
