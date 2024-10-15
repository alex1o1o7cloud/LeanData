import Mathlib

namespace NUMINAMATH_GPT_range_of_m_l2200_220026

-- Define the two vectors a and b
def vector_a := (1, 2)
def vector_b (m : ℝ) := (m, 3 * m - 2)

-- Define the condition for non-collinearity
def non_collinear (m : ℝ) := ¬ (m / 1 = (3 * m - 2) / 2)

theorem range_of_m (m : ℝ) : non_collinear m ↔ m ≠ 2 :=
  sorry

end NUMINAMATH_GPT_range_of_m_l2200_220026


namespace NUMINAMATH_GPT_prove_value_of_expression_l2200_220040

theorem prove_value_of_expression (x y a b : ℝ)
    (h1 : x = 2) 
    (h2 : y = 1)
    (h3 : 2 * a + b = 5)
    (h4 : a + 2 * b = 1) : 
    3 - a - b = 1 := 
by
    -- Skipping proof
    sorry

end NUMINAMATH_GPT_prove_value_of_expression_l2200_220040


namespace NUMINAMATH_GPT_pyramid_section_rhombus_l2200_220079

structure Pyramid (A B C D : Type) := (point : Type)

def is_parallel (l1 l2 : ℝ) : Prop :=
  ∀ (m n : ℝ), m * l1 = n * l2

def is_parallelogram (K L M N : Type) : Prop :=
  sorry

def is_rhombus (K L M N : Type) : Prop :=
  sorry

noncomputable def side_length_rhombus (a b : ℝ) : ℝ :=
  (a * b) / (a + b)

/-- Prove that the section of pyramid ABCD with a plane parallel to edges AC and BD is a parallelogram,
and under certain conditions, this parallelogram is a rhombus. Find the side of this rhombus given AC = a and BD = b. -/
theorem pyramid_section_rhombus (A B C D K L M N : Type) (a b : ℝ) :
  is_parallel AC BD →
  is_parallelogram K L M N →
  is_rhombus K L M N →
  side_length_rhombus a b = (a * b) / (a + b) :=
by
  sorry

end NUMINAMATH_GPT_pyramid_section_rhombus_l2200_220079


namespace NUMINAMATH_GPT_num_tosses_l2200_220031

theorem num_tosses (n : ℕ) (h : (1 - (7 / 8 : ℝ)^n) = 0.111328125) : n = 7 :=
by
  sorry

end NUMINAMATH_GPT_num_tosses_l2200_220031


namespace NUMINAMATH_GPT_find_softball_players_l2200_220049

def total_players : ℕ := 51
def cricket_players : ℕ := 10
def hockey_players : ℕ := 12
def football_players : ℕ := 16

def softball_players : ℕ := total_players - (cricket_players + hockey_players + football_players)

theorem find_softball_players : softball_players = 13 := 
by {
  sorry
}

end NUMINAMATH_GPT_find_softball_players_l2200_220049


namespace NUMINAMATH_GPT_lowest_cost_per_ton_l2200_220004

-- Define the conditions given in the problem statement
variable (x : ℝ) (y : ℝ)

-- Define the annual production range
def production_range (x : ℝ) : Prop := x ≥ 150 ∧ x ≤ 250

-- Define the relationship between total annual production cost and annual production
def production_cost_relation (x y : ℝ) : Prop := y = (x^2 / 10) - 30 * x + 4000

-- State the main theorem: the annual production when the cost per ton is the lowest is 200 tons
theorem lowest_cost_per_ton (x : ℝ) (y : ℝ) (h1 : production_range x) (h2 : production_cost_relation x y) : x = 200 :=
sorry

end NUMINAMATH_GPT_lowest_cost_per_ton_l2200_220004


namespace NUMINAMATH_GPT_quadratic_real_solutions_l2200_220053

theorem quadratic_real_solutions (m : ℝ) :
  (∃ x : ℝ, (m - 3) * x^2 + 4 * x + 1 = 0) ↔ (m ≤ 7 ∧ m ≠ 3) := 
sorry

end NUMINAMATH_GPT_quadratic_real_solutions_l2200_220053


namespace NUMINAMATH_GPT_polynomial_factorization_l2200_220001

theorem polynomial_factorization :
  ∀ x : ℤ, x^15 + x^10 + x^5 + 1 = (x^2 + x + 1) * (x^13 - x^12 + x^10 - x^9 + x^7 - x^6 + x^4 - x^3 + x - 1) :=
by sorry

end NUMINAMATH_GPT_polynomial_factorization_l2200_220001


namespace NUMINAMATH_GPT_sum_of_cubes_l2200_220067

theorem sum_of_cubes (x y : ℂ) (h1 : x + y = 1) (h2 : x * y = 1) : x^3 + y^3 = -2 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_cubes_l2200_220067


namespace NUMINAMATH_GPT_mittens_in_each_box_l2200_220087

theorem mittens_in_each_box (boxes scarves_per_box total_clothing : ℕ) (h1 : boxes = 8) (h2 : scarves_per_box = 4) (h3 : total_clothing = 80) :
  ∃ (mittens_per_box : ℕ), mittens_per_box = 6 :=
by
  let total_scarves := boxes * scarves_per_box
  let total_mittens := total_clothing - total_scarves
  let mittens_per_box := total_mittens / boxes
  use mittens_per_box
  sorry

end NUMINAMATH_GPT_mittens_in_each_box_l2200_220087


namespace NUMINAMATH_GPT_acute_angle_inclination_range_l2200_220057

/-- 
For the line passing through points P(1-a, 1+a) and Q(3, 2a), 
prove that the range of the real number a such that the line has an acute angle of inclination is (-∞, 1) ∪ (1, 4).
-/
theorem acute_angle_inclination_range (a : ℝ) : 
  (a < 1 ∨ (1 < a ∧ a < 4)) ↔ (0 < (a - 1) / (4 - a)) :=
sorry

end NUMINAMATH_GPT_acute_angle_inclination_range_l2200_220057


namespace NUMINAMATH_GPT_additional_days_when_selling_5_goats_l2200_220044

variables (G D F X : ℕ)

def total_feed (num_goats days : ℕ) := G * num_goats * days

theorem additional_days_when_selling_5_goats
  (h1 : total_feed G 20 D = F)
  (h2 : total_feed G 15 (D + X) = F)
  (h3 : total_feed G 30 (D - 3) = F):
  X = 9 :=
by
  -- the exact proof is omitted and presented as 'sorry'
  sorry

end NUMINAMATH_GPT_additional_days_when_selling_5_goats_l2200_220044


namespace NUMINAMATH_GPT_inequality_transpose_l2200_220025

variable (a b : ℝ)

theorem inequality_transpose (h : a < b) (hab : b < 0) : (1 / a) > (1 / b) := by
  sorry

end NUMINAMATH_GPT_inequality_transpose_l2200_220025


namespace NUMINAMATH_GPT_sequence_B_is_arithmetic_l2200_220078

-- Definitions of the sequences
def S_n (n : ℕ) : ℕ := 2*n + 1

-- Theorem statement
theorem sequence_B_is_arithmetic : ∀ n : ℕ, S_n (n + 1) - S_n n = 2 :=
by
  intro n
  sorry

end NUMINAMATH_GPT_sequence_B_is_arithmetic_l2200_220078


namespace NUMINAMATH_GPT_max_angle_OAB_l2200_220007

/-- Let OA = a, OB = b, and OM = x on the right angle XOY, where a < b. 
    The value of x which maximizes the angle ∠AMB is sqrt(ab). -/
theorem max_angle_OAB (a b x : ℝ) (h : a < b) (h1 : x = Real.sqrt (a * b)) :
  x = Real.sqrt (a * b) :=
sorry

end NUMINAMATH_GPT_max_angle_OAB_l2200_220007


namespace NUMINAMATH_GPT_oprq_possible_figures_l2200_220034

theorem oprq_possible_figures (x1 y1 x2 y2 : ℝ) (h : (x1, y1) ≠ (x2, y2)) : 
  -- Define the points P, Q, and R
  let P := (x1, y1)
  let Q := (x2, y2)
  let R := (x1 - x2, y1 - y2)
  -- Proving the geometric possibilities
  (∃ k : ℝ, x1 = k * x2 ∧ y1 = k * y2) ∨
  -- When the points are collinear
  ((x1 + x2, y1 + y2) = (x1, y1)) :=
sorry

end NUMINAMATH_GPT_oprq_possible_figures_l2200_220034


namespace NUMINAMATH_GPT_smallest_sum_of_bases_l2200_220043

theorem smallest_sum_of_bases :
  ∃ (c d : ℕ), 8 * c + 9 = 9 * d + 8 ∧ c + d = 19 := 
by
  sorry

end NUMINAMATH_GPT_smallest_sum_of_bases_l2200_220043


namespace NUMINAMATH_GPT_annie_building_time_l2200_220002

theorem annie_building_time (b p : ℕ) (h1 : b = 3 * p - 5) (h2 : b + p = 67) : b = 49 :=
by
  sorry

end NUMINAMATH_GPT_annie_building_time_l2200_220002


namespace NUMINAMATH_GPT_prob_of_drawing_one_red_ball_distribution_of_X_l2200_220056

-- Definitions for conditions
def red_balls : ℕ := 2
def white_balls : ℕ := 3
def total_balls : ℕ := red_balls + white_balls
def balls_drawn : ℕ := 3

-- Combinations 
noncomputable def combination (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Probabilities
noncomputable def prob_ex_one_red_ball : ℚ :=
  (combination red_balls 1 * combination white_balls 2) / combination total_balls balls_drawn

noncomputable def prob_X_0 : ℚ := (combination white_balls 3) / combination total_balls balls_drawn
noncomputable def prob_X_1 : ℚ := prob_ex_one_red_ball
noncomputable def prob_X_2 : ℚ := (combination red_balls 2 * combination white_balls 1) / combination total_balls balls_drawn

-- Theorem statements
theorem prob_of_drawing_one_red_ball : prob_ex_one_red_ball = 3/5 := by
  sorry

theorem distribution_of_X : prob_X_0 = 1/10 ∧ prob_X_1 = 3/5 ∧ prob_X_2 = 3/10 := by
  sorry

end NUMINAMATH_GPT_prob_of_drawing_one_red_ball_distribution_of_X_l2200_220056


namespace NUMINAMATH_GPT_base_8_digits_sum_l2200_220088

theorem base_8_digits_sum
    (X Y Z : ℕ)
    (h1 : 1 ≤ X ∧ X < 8)
    (h2 : 1 ≤ Y ∧ Y < 8)
    (h3 : 1 ≤ Z ∧ Z < 8)
    (h4 : X ≠ Y)
    (h5 : Y ≠ Z)
    (h6 : Z ≠ X)
    (h7 : 8^2 * X + 8 * Y + Z + 8^2 * Y + 8 * Z + X + 8^2 * Z + 8 * X + Y = 8^3 * X + 8^2 * X + 8 * X) :
  Y + Z = 7 * X :=
by
  sorry

end NUMINAMATH_GPT_base_8_digits_sum_l2200_220088


namespace NUMINAMATH_GPT_workshop_employees_l2200_220063

theorem workshop_employees (x y : ℕ) 
  (H1 : (x + y) - ((1 / 2) * x + (1 / 3) * y + (1 / 3) * x + (1 / 2) * y) = 120)
  (H2 : (1 / 2) * x + (1 / 3) * y = (1 / 7) * ((1 / 3) * x + (1 / 2) * y) + (1 / 3) * x + (1 / 2) * y) : 
  x = 480 ∧ y = 240 := 
by
  sorry

end NUMINAMATH_GPT_workshop_employees_l2200_220063


namespace NUMINAMATH_GPT_fraction_of_loss_l2200_220010

theorem fraction_of_loss
  (SP CP : ℚ) (hSP : SP = 16) (hCP : CP = 17) :
  (CP - SP) / CP = 1 / 17 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_loss_l2200_220010


namespace NUMINAMATH_GPT_length_of_platform_l2200_220077

variable (L : ℕ)

theorem length_of_platform
  (train_length : ℕ)
  (time_cross_post : ℕ)
  (time_cross_platform : ℕ)
  (train_length_eq : train_length = 300)
  (time_cross_post_eq : time_cross_post = 18)
  (time_cross_platform_eq : time_cross_platform = 39)
  : L = 350 := sorry

end NUMINAMATH_GPT_length_of_platform_l2200_220077


namespace NUMINAMATH_GPT_division_neg4_by_2_l2200_220060

theorem division_neg4_by_2 : (-4) / 2 = -2 := sorry

end NUMINAMATH_GPT_division_neg4_by_2_l2200_220060


namespace NUMINAMATH_GPT_total_distance_fourth_fifth_days_l2200_220099

theorem total_distance_fourth_fifth_days (d : ℕ) (total_distance : ℕ) (n : ℕ) (q : ℚ) 
  (S_6 : d * (1 - q^6) / (1 - q) = 378) (ratio : q = 1/2) (n_six : n = 6) : 
  (d * q^3) + (d * q^4) = 36 :=
by 
  sorry

end NUMINAMATH_GPT_total_distance_fourth_fifth_days_l2200_220099


namespace NUMINAMATH_GPT_hall_length_width_difference_l2200_220041

theorem hall_length_width_difference (L W : ℝ) 
  (h1 : W = 1 / 2 * L) 
  (h2 : L * W = 128) : 
  L - W = 8 :=
by
  sorry

end NUMINAMATH_GPT_hall_length_width_difference_l2200_220041


namespace NUMINAMATH_GPT_batsman_average_increase_l2200_220003

theorem batsman_average_increase
  (A : ℕ)  -- Assume the initial average is a non-negative integer
  (h1 : 11 * A + 70 = 12 * (A + 3))  -- Condition derived from the problem
  : A + 3 = 37 := 
by {
  -- The actual proof would go here, but is replaced by sorry to skip the proof
  sorry
}

end NUMINAMATH_GPT_batsman_average_increase_l2200_220003


namespace NUMINAMATH_GPT_symmetry_y_axis_B_l2200_220069

def point_A : ℝ × ℝ := (-1, 2)

def symmetric_point (p : ℝ × ℝ) : ℝ × ℝ := (-(p.1), p.2)

theorem symmetry_y_axis_B :
  symmetric_point point_A = (1, 2) :=
by
  -- proof is omitted
  sorry

end NUMINAMATH_GPT_symmetry_y_axis_B_l2200_220069


namespace NUMINAMATH_GPT_point_in_third_quadrant_l2200_220058

theorem point_in_third_quadrant (a b : ℝ) (h1 : a < 0) (h2 : b > 0) : (-b < 0 ∧ a - 3 < 0) :=
by sorry

end NUMINAMATH_GPT_point_in_third_quadrant_l2200_220058


namespace NUMINAMATH_GPT_percentage_decrease_l2200_220030

variable (current_price original_price : ℝ)

theorem percentage_decrease (h1 : current_price = 760) (h2 : original_price = 1000) :
  (original_price - current_price) / original_price * 100 = 24 :=
by
  sorry

end NUMINAMATH_GPT_percentage_decrease_l2200_220030


namespace NUMINAMATH_GPT_divisibility_l2200_220096

theorem divisibility {n A B k : ℤ} (h_n : n = 1000 * B + A) (h_k : k = A - B) :
  (7 ∣ n ∨ 11 ∣ n ∨ 13 ∣ n) ↔ (7 ∣ k ∨ 11 ∣ k ∨ 13 ∣ k) :=
by
  sorry

end NUMINAMATH_GPT_divisibility_l2200_220096


namespace NUMINAMATH_GPT_leak_drain_time_l2200_220095

noncomputable def pump_rate : ℚ := 1/2
noncomputable def leak_empty_rate : ℚ := 1 / (1 / pump_rate - 5/11)

theorem leak_drain_time :
  let pump_rate := 1/2
  let combined_rate := 5/11
  let leak_rate := pump_rate - combined_rate
  1 / leak_rate = 22 :=
  by
    -- Definition of pump rate
    let pump_rate := 1/2
    -- Definition of combined rate
    let combined_rate := 5/11
    -- Definition of leak rate
    let leak_rate := pump_rate - combined_rate
    -- Calculate leak drain time
    show 1 / leak_rate = 22
    sorry

end NUMINAMATH_GPT_leak_drain_time_l2200_220095


namespace NUMINAMATH_GPT_area_of_inscribed_square_l2200_220066

theorem area_of_inscribed_square (XY YZ : ℝ) (hXY : XY = 18) (hYZ : YZ = 30) :
  ∃ (s : ℝ), s^2 = 540 :=
by
  sorry

end NUMINAMATH_GPT_area_of_inscribed_square_l2200_220066


namespace NUMINAMATH_GPT_arithmetic_sequence_problem_l2200_220073

variable (a : ℕ → ℤ) -- The arithmetic sequence as a function from natural numbers to integers
variable (S : ℕ → ℤ) -- Sum of the first n terms of the sequence

-- Conditions
variable (h1 : S 8 = 4 * a 3) -- Sum of the first 8 terms is 4 times the third term
variable (h2 : a 7 = -2)      -- The seventh term is -2

-- Proven Goal
theorem arithmetic_sequence_problem : a 9 = -6 := 
by sorry -- This is a placeholder for the proof

end NUMINAMATH_GPT_arithmetic_sequence_problem_l2200_220073


namespace NUMINAMATH_GPT_hyperbola_asymptotes_l2200_220035

open Real

noncomputable def hyperbola (x y m : ℝ) : Prop := (x^2 / 9) - (y^2 / m) = 1

noncomputable def on_line (x y : ℝ) : Prop := x + y = 5

theorem hyperbola_asymptotes (m : ℝ) (hm : 9 + m = 25) :
    (∃ x y : ℝ, hyperbola x y m ∧ on_line x y) →
    (∀ x : ℝ, on_line x ((4 / 3) * x) ∧ on_line x (-(4 / 3) * x)) :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_asymptotes_l2200_220035


namespace NUMINAMATH_GPT_effective_annual_rate_l2200_220083

theorem effective_annual_rate (i : ℚ) (n : ℕ) (h_i : i = 0.16) (h_n : n = 2) :
  (1 + i / n) ^ n - 1 = 0.1664 :=
by {
  sorry
}

end NUMINAMATH_GPT_effective_annual_rate_l2200_220083


namespace NUMINAMATH_GPT_p_sufficient_not_necessary_for_q_l2200_220000

def p (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 ≤ 2
def q (x y : ℝ) : Prop := y ≥ x - 1 ∧ y ≥ 1 - x ∧ y ≤ 1

theorem p_sufficient_not_necessary_for_q :
  (∀ x y : ℝ, q x y → p x y) ∧ ¬(∀ x y : ℝ, p x y → q x y) := by
  sorry

end NUMINAMATH_GPT_p_sufficient_not_necessary_for_q_l2200_220000


namespace NUMINAMATH_GPT_p_value_for_roots_l2200_220023

theorem p_value_for_roots (α β : ℝ) (h1 : 3 * α^2 + 5 * α + 2 = 0) (h2 : 3 * β^2 + 5 * β + 2 = 0)
  (hαβ : α + β = -5/3) (hαβ_prod : α * β = 2/3) : p = -49/9 :=
by
  sorry

end NUMINAMATH_GPT_p_value_for_roots_l2200_220023


namespace NUMINAMATH_GPT_myOperation_identity_l2200_220074

variable {R : Type*} [LinearOrderedField R]

def myOperation (a b : R) : R := (a - b) ^ 2

theorem myOperation_identity (x y : R) : myOperation ((x - y) ^ 2) ((y - x) ^ 2) = 0 := 
by 
  sorry

end NUMINAMATH_GPT_myOperation_identity_l2200_220074


namespace NUMINAMATH_GPT_area_of_isosceles_right_triangle_l2200_220062

def is_isosceles_right_triangle (X Y Z : Type*) : Prop :=
∃ (XY YZ XZ : ℝ), XY = 6.000000000000001 ∧ XY > YZ ∧ YZ = XZ ∧ XY = YZ * Real.sqrt 2

theorem area_of_isosceles_right_triangle
  {X Y Z : Type*}
  (h : is_isosceles_right_triangle X Y Z) :
  ∃ A : ℝ, A = 9.000000000000002 :=
by
  sorry

end NUMINAMATH_GPT_area_of_isosceles_right_triangle_l2200_220062


namespace NUMINAMATH_GPT_cos_alpha_condition_l2200_220093

theorem cos_alpha_condition (k : ℤ) (α : ℝ) :
  (α = 2 * k * Real.pi - Real.pi / 4 -> Real.cos α = Real.sqrt 2 / 2) ∧
  (Real.cos α = Real.sqrt 2 / 2 -> ∃ k : ℤ, α = 2 * k * Real.pi + Real.pi / 4 ∨ α = 2 * k * Real.pi - Real.pi / 4) :=
by
  sorry

end NUMINAMATH_GPT_cos_alpha_condition_l2200_220093


namespace NUMINAMATH_GPT_rationalization_correct_l2200_220085

noncomputable def rationalize_denominator : Prop :=
  (∃ (a b : ℝ), a = (12:ℝ).sqrt + (5:ℝ).sqrt ∧ b = (3:ℝ).sqrt + (5:ℝ).sqrt ∧
                (a / b) = (((15:ℝ).sqrt - 1) / 2))

theorem rationalization_correct : rationalize_denominator :=
by {
  sorry
}

end NUMINAMATH_GPT_rationalization_correct_l2200_220085


namespace NUMINAMATH_GPT_range_of_a_l2200_220052

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (a - 2)*x^2 + (2 * a - 1) * x + 6 > 0 ↔ x = 3 → true) ∧
  (∀ x : ℝ, (a - 2)*x^2 + (2 * a - 1) * x + 6 > 0 ↔ x = 5 → false) →
  1 < a ∧ a ≤ 7 / 5 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l2200_220052


namespace NUMINAMATH_GPT_shortest_distance_point_on_circle_to_line_l2200_220038

theorem shortest_distance_point_on_circle_to_line
  (P : ℝ × ℝ)
  (hP : (P.1 + 1)^2 + (P.2 - 2)^2 = 1) :
  ∃ (d : ℝ), d = 3 :=
sorry

end NUMINAMATH_GPT_shortest_distance_point_on_circle_to_line_l2200_220038


namespace NUMINAMATH_GPT_a_minus_b_eq_neg_9_or_neg_1_l2200_220059

theorem a_minus_b_eq_neg_9_or_neg_1 (a b : ℝ) (h₁ : |a| = 5) (h₂ : |b| = 4) (h₃ : a + b < 0) :
  a - b = -9 ∨ a - b = -1 :=
by
  sorry

end NUMINAMATH_GPT_a_minus_b_eq_neg_9_or_neg_1_l2200_220059


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l2200_220070

variable (a b x y : ℝ)

theorem sufficient_but_not_necessary_condition (ha : a > 0) (hb : b > 0) :
  ((x > a ∧ y > b) → (x + y > a + b ∧ x * y > a * b)) ∧
  ¬((x + y > a + b ∧ x * y > a * b) → (x > a ∧ y > b)) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l2200_220070


namespace NUMINAMATH_GPT_find_number_of_books_l2200_220055

-- Define the constants and equation based on the conditions
def price_paid_per_book : ℕ := 11
def price_sold_per_book : ℕ := 25
def total_difference : ℕ := 210

def books_equation (x : ℕ) : Prop :=
  (price_sold_per_book * x) - (price_paid_per_book * x) = total_difference

-- The theorem statement that needs to be proved
theorem find_number_of_books (x : ℕ) (h : books_equation x) : 
  x = 15 :=
sorry

end NUMINAMATH_GPT_find_number_of_books_l2200_220055


namespace NUMINAMATH_GPT_problem_statement_l2200_220016

def f (x : ℤ) : ℤ := 3*x + 4
def g (x : ℤ) : ℤ := 4*x - 3

theorem problem_statement : (f (g (f 2))) / (g (f (g 2))) = 115 / 73 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l2200_220016


namespace NUMINAMATH_GPT_marie_initial_erasers_l2200_220029

def erasers_problem : Prop :=
  ∃ initial_erasers : ℝ, initial_erasers + 42.0 = 137

theorem marie_initial_erasers : erasers_problem :=
  sorry

end NUMINAMATH_GPT_marie_initial_erasers_l2200_220029


namespace NUMINAMATH_GPT_probability_red_or_blue_marbles_l2200_220032

theorem probability_red_or_blue_marbles (red blue green total : ℕ) (h_red : red = 4) (h_blue : blue = 3) (h_green : green = 6) (h_total : total = red + blue + green) :
  (red + blue) / total = 7 / 13 :=
by
  sorry

end NUMINAMATH_GPT_probability_red_or_blue_marbles_l2200_220032


namespace NUMINAMATH_GPT_problem_l2200_220042

noncomputable def a (x : ℝ) : ℝ × ℝ := (5 * (Real.sqrt 3) * Real.cos x, Real.cos x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sin x, 2 * Real.cos x)
noncomputable def f (x : ℝ) : ℝ := 
  let dot_product := (a x).fst * (b x).fst + (a x).snd * (b x).snd
  let magnitude_square_b := (b x).fst ^ 2 + (b x).snd ^ 2
  dot_product + magnitude_square_b

theorem problem :
  (∀ x, f x = 5 * Real.sin (2 * x + Real.pi / 6) + 7 / 2) ∧
  (∃ T, T = Real.pi) ∧ 
  (∃ x, f x = 17 / 2) ∧ 
  (∃ x, f x = -3 / 2) ∧ 
  (∀ x ∈ Set.Icc 0 (Real.pi / 6), 0 ≤ x ∧ x ≤ Real.pi / 6) ∧
  (∀ x ∈ Set.Icc (2 * Real.pi / 3) Real.pi, (2 * Real.pi / 3) ≤ x ∧ x ≤ Real.pi)
:= by
  sorry

end NUMINAMATH_GPT_problem_l2200_220042


namespace NUMINAMATH_GPT_dishonest_dealer_weight_l2200_220028

noncomputable def dealer_weight_equiv (cost_price : ℝ) (selling_price : ℝ) (profit_percent : ℝ) : ℝ :=
  (1 - profit_percent / 100) * cost_price / selling_price

theorem dishonest_dealer_weight :
  dealer_weight_equiv 1 2 100 = 0.5 :=
by
  sorry

end NUMINAMATH_GPT_dishonest_dealer_weight_l2200_220028


namespace NUMINAMATH_GPT_exists_quadratic_sequence_l2200_220046

theorem exists_quadratic_sequence (b c : ℤ) : ∃ n : ℕ, ∃ (a : ℕ → ℤ), (a 0 = b) ∧ (a n = c) ∧ ∀ i : ℕ, 1 ≤ i → i ≤ n → |a i - a (i - 1)| = i ^ 2 := 
sorry

end NUMINAMATH_GPT_exists_quadratic_sequence_l2200_220046


namespace NUMINAMATH_GPT_workshop_workers_transfer_l2200_220097

theorem workshop_workers_transfer (w d t : ℕ) (h_w : 63 ≤ w) (h_d : d ≤ 31) 
(h_prod : 1994 = 31 * w + t * (t + 1) / 2) : 
(d = 28 ∧ t = 10) ∨ (d = 30 ∧ t = 21) := sorry

end NUMINAMATH_GPT_workshop_workers_transfer_l2200_220097


namespace NUMINAMATH_GPT_car_b_speed_l2200_220033

noncomputable def SpeedOfCarB (Speed_A Time_A Time_B d_ratio: ℝ) : ℝ :=
  let Distance_A := Speed_A * Time_A
  let Distance_B := Distance_A / d_ratio
  Distance_B / Time_B

theorem car_b_speed
  (Speed_A : ℝ) (Time_A : ℝ) (Time_B : ℝ) (d_ratio : ℝ)
  (h1 : Speed_A = 70) (h2 : Time_A = 10) (h3 : Time_B = 10) (h4 : d_ratio = 2) :
  SpeedOfCarB Speed_A Time_A Time_B d_ratio = 35 :=
by
  sorry

end NUMINAMATH_GPT_car_b_speed_l2200_220033


namespace NUMINAMATH_GPT_find_digits_l2200_220036

theorem find_digits (a b : ℕ) (h1 : (1 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9)) :
  (∃ (c : ℕ), 10000 * a + 6790 + b = 72 * c) ↔ (a = 3 ∧ b = 2) :=
by
  sorry

end NUMINAMATH_GPT_find_digits_l2200_220036


namespace NUMINAMATH_GPT_max_price_per_unit_l2200_220084

-- Define the conditions
def original_price : ℝ := 25
def original_sales_volume : ℕ := 80000
def price_increase_effect (t : ℝ) : ℝ := 2000 * (t - original_price)
def new_sales_volume (t : ℝ) : ℝ := 130 - 2 * t

-- Define the condition for revenue
def revenue_condition (t : ℝ) : Prop :=
  t * new_sales_volume t ≥ original_price * original_sales_volume

-- Statement to prove the maximum price per unit
theorem max_price_per_unit : ∀ t : ℝ, revenue_condition t → t ≤ 40 := sorry

end NUMINAMATH_GPT_max_price_per_unit_l2200_220084


namespace NUMINAMATH_GPT_total_cost_of_video_games_l2200_220098

theorem total_cost_of_video_games :
  let cost_football_game := 14.02
  let cost_strategy_game := 9.46
  let cost_batman_game := 12.04
  let total_cost := cost_football_game + cost_strategy_game + cost_batman_game
  total_cost = 35.52 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_total_cost_of_video_games_l2200_220098


namespace NUMINAMATH_GPT_carpet_width_l2200_220072

theorem carpet_width
  (carpet_percentage : ℝ)
  (living_room_area : ℝ)
  (carpet_length : ℝ) :
  carpet_percentage = 0.30 →
  living_room_area = 120 →
  carpet_length = 9 →
  carpet_percentage * living_room_area / carpet_length = 4 :=
by
  sorry

end NUMINAMATH_GPT_carpet_width_l2200_220072


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l2200_220064

theorem necessary_but_not_sufficient (x y : ℝ) : 
  (x < 0 ∨ y < 0) → x + y < 0 :=
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l2200_220064


namespace NUMINAMATH_GPT_crucian_carps_heavier_l2200_220071

-- Variables representing the weights
variables (K O L : ℝ)

-- Given conditions
axiom weight_6K_lt_5O : 6 * K < 5 * O
axiom weight_6K_gt_10L : 6 * K > 10 * L

-- The proof statement
theorem crucian_carps_heavier : 2 * K > 3 * L :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_crucian_carps_heavier_l2200_220071


namespace NUMINAMATH_GPT_determine_B_l2200_220054

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5})
variable (h1 : (A ∪ B)ᶜ = {1})
variable (h2 : A ∩ Bᶜ = {3})

theorem determine_B : B = {2, 4, 5} :=
by
  sorry

end NUMINAMATH_GPT_determine_B_l2200_220054


namespace NUMINAMATH_GPT_find_number_l2200_220008

theorem find_number (x : ℝ) :
  0.15 * x = 0.25 * 16 + 2 → x = 40 :=
by
  -- skipping the proof steps
  sorry

end NUMINAMATH_GPT_find_number_l2200_220008


namespace NUMINAMATH_GPT_polynomial_divisible_2520_l2200_220005

theorem polynomial_divisible_2520 (n : ℕ) : (n^7 - 14 * n^5 + 49 * n^3 - 36 * n) % 2520 = 0 := 
sorry

end NUMINAMATH_GPT_polynomial_divisible_2520_l2200_220005


namespace NUMINAMATH_GPT_problem_statement_l2200_220037

theorem problem_statement (x : ℝ) (h : x^2 + 3 * x - 1 = 0) : x^3 + 5 * x^2 + 5 * x + 18 = 20 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l2200_220037


namespace NUMINAMATH_GPT_new_class_mean_l2200_220090

theorem new_class_mean (n1 n2 : ℕ) (mean1 mean2 : ℝ) (h1 : n1 = 45) (h2 : n2 = 5) (h3 : mean1 = 0.85) (h4 : mean2 = 0.90) : 
(n1 + n2 = 50) → 
((n1 * mean1 + n2 * mean2) / (n1 + n2) = 0.855) := 
by
  intro total_students
  sorry

end NUMINAMATH_GPT_new_class_mean_l2200_220090


namespace NUMINAMATH_GPT_min_value_m_plus_2n_exists_min_value_l2200_220019

variable (n : ℝ) -- Declare n as a real number.

-- Define m in terms of n
def m (n : ℝ) : ℝ := n^2

-- State and prove that the minimum value of m + 2n is -1
theorem min_value_m_plus_2n : (m n + 2 * n) ≥ -1 :=
by sorry

-- Show there exists an n such that m + 2n = -1
theorem exists_min_value : ∃ n : ℝ, m n + 2 * n = -1 :=
by sorry

end NUMINAMATH_GPT_min_value_m_plus_2n_exists_min_value_l2200_220019


namespace NUMINAMATH_GPT_geometric_sequence_seventh_term_l2200_220009

noncomputable def a_7 (a₁ q : ℝ) : ℝ :=
  a₁ * q^6

theorem geometric_sequence_seventh_term :
  a_7 3 (Real.sqrt 2) = 24 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_seventh_term_l2200_220009


namespace NUMINAMATH_GPT_museum_paintings_discarded_l2200_220011

def initial_paintings : ℕ := 2500
def percentage_to_discard : ℝ := 0.35
def paintings_discarded : ℝ := initial_paintings * percentage_to_discard

theorem museum_paintings_discarded : paintings_discarded = 875 :=
by
  -- Lean automatically simplifies this using basic arithmetic rules
  sorry

end NUMINAMATH_GPT_museum_paintings_discarded_l2200_220011


namespace NUMINAMATH_GPT_carli_charlie_flute_ratio_l2200_220065

theorem carli_charlie_flute_ratio :
  let charlie_flutes := 1
  let charlie_horns := 2
  let charlie_harps := 1
  let carli_horns := charlie_horns / 2
  let total_instruments := 7
  ∃ (carli_flutes : ℕ), 
    (charlie_flutes + charlie_horns + charlie_harps + carli_flutes + carli_horns = total_instruments) ∧ 
    (carli_flutes / charlie_flutes = 2) :=
by
  sorry

end NUMINAMATH_GPT_carli_charlie_flute_ratio_l2200_220065


namespace NUMINAMATH_GPT_repeat_decimals_subtraction_l2200_220039

-- Define repeating decimal 0.4 repeating as a fraction
def repr_decimal_4 : ℚ := 4 / 9

-- Define repeating decimal 0.6 repeating as a fraction
def repr_decimal_6 : ℚ := 2 / 3

-- Theorem stating the equivalence of subtraction of these repeating decimals
theorem repeat_decimals_subtraction :
  repr_decimal_4 - repr_decimal_6 = -2 / 9 :=
sorry

end NUMINAMATH_GPT_repeat_decimals_subtraction_l2200_220039


namespace NUMINAMATH_GPT_chlorine_needed_l2200_220015

variable (Methane moles_HCl moles_Cl₂ : ℕ)

-- Given conditions
def reaction_started_with_one_mole_of_methane : Prop :=
  Methane = 1

def reaction_produces_two_moles_of_HCl : Prop :=
  moles_HCl = 2

-- Question to be proved
def number_of_moles_of_Chlorine_combined : Prop :=
  moles_Cl₂ = 2

theorem chlorine_needed
  (h1 : reaction_started_with_one_mole_of_methane Methane)
  (h2 : reaction_produces_two_moles_of_HCl moles_HCl)
  : number_of_moles_of_Chlorine_combined moles_Cl₂ :=
sorry

end NUMINAMATH_GPT_chlorine_needed_l2200_220015


namespace NUMINAMATH_GPT_probability_two_girls_l2200_220024

-- Define the conditions
def total_students := 8
def total_girls := 5
def total_boys := 3
def choose_two_from_n (n : ℕ) := n * (n - 1) / 2

-- Define the question as a statement that the probability equals 5/14
theorem probability_two_girls
    (h1 : choose_two_from_n total_students = 28)
    (h2 : choose_two_from_n total_girls = 10) :
    (choose_two_from_n total_girls : ℚ) / choose_two_from_n total_students = 5 / 14 :=
by
  sorry

end NUMINAMATH_GPT_probability_two_girls_l2200_220024


namespace NUMINAMATH_GPT_probability_non_edge_unit_square_l2200_220051

theorem probability_non_edge_unit_square : 
  let total_squares := 100
  let perimeter_squares := 36
  let non_perimeter_squares := total_squares - perimeter_squares
  let probability := (non_perimeter_squares : ℚ) / total_squares
  probability = 16 / 25 :=
by
  sorry

end NUMINAMATH_GPT_probability_non_edge_unit_square_l2200_220051


namespace NUMINAMATH_GPT_total_packages_l2200_220018

theorem total_packages (num_trucks : ℕ) (packages_per_truck : ℕ) (h1 : num_trucks = 7) (h2 : packages_per_truck = 70) : num_trucks * packages_per_truck = 490 := by
  sorry

end NUMINAMATH_GPT_total_packages_l2200_220018


namespace NUMINAMATH_GPT_calculate_fraction_value_l2200_220020

theorem calculate_fraction_value :
  1 + 1 / (1 + 1 / (1 + 1 / (1 + 2))) = 11 / 7 := 
  sorry

end NUMINAMATH_GPT_calculate_fraction_value_l2200_220020


namespace NUMINAMATH_GPT_pair_d_are_equal_l2200_220050

theorem pair_d_are_equal : -(2 ^ 3) = (-2) ^ 3 :=
by
  -- Detailed proof steps go here, but are omitted for this task.
  sorry

end NUMINAMATH_GPT_pair_d_are_equal_l2200_220050


namespace NUMINAMATH_GPT_how_many_tickets_left_l2200_220086

-- Define the conditions
def tickets_from_whack_a_mole : ℕ := 32
def tickets_from_skee_ball : ℕ := 25
def tickets_spent_on_hat : ℕ := 7

-- Define the total tickets won by Tom
def total_tickets : ℕ := tickets_from_whack_a_mole + tickets_from_skee_ball

-- State the theorem to be proved: how many tickets Tom has left
theorem how_many_tickets_left : total_tickets - tickets_spent_on_hat = 50 := by
  sorry

end NUMINAMATH_GPT_how_many_tickets_left_l2200_220086


namespace NUMINAMATH_GPT_asymptotes_of_hyperbola_l2200_220061

theorem asymptotes_of_hyperbola : 
  (∀ (x y : ℝ), (x^2 / 9) - (y^2 / 16) = 1 → y = (4 / 3) * x ∨ y = -(4 / 3) * x) :=
by
  intro x y h
  sorry

end NUMINAMATH_GPT_asymptotes_of_hyperbola_l2200_220061


namespace NUMINAMATH_GPT_star_vertex_angle_l2200_220006

-- Defining a function that calculates the star vertex angle for odd n-sided concave regular polygon
theorem star_vertex_angle (n : ℕ) (hn_odd : n % 2 = 1) (hn_gt3 : 3 < n) : 
  (180 - 360 / n) = (n - 2) * 180 / n := 
sorry

end NUMINAMATH_GPT_star_vertex_angle_l2200_220006


namespace NUMINAMATH_GPT_f_at_3_l2200_220094

noncomputable def f : ℝ → ℝ := sorry

lemma periodic (f : ℝ → ℝ) : ∀ x : ℝ, f (x + 4) = f x := sorry

lemma odd_function (f : ℝ → ℝ) : ∀ x : ℝ, f (-x) + f x = 0 := sorry

lemma given_interval (f : ℝ → ℝ) : ∀ x, 0 ≤ x ∧ x ≤ 2 → f x = (x - 1)^2 := sorry

theorem f_at_3 : f 3 = 0 := 
by
  sorry

end NUMINAMATH_GPT_f_at_3_l2200_220094


namespace NUMINAMATH_GPT_part_I_part_II_l2200_220014

-- Part (I)
theorem part_I (x a : ℝ) (h_a : a = 3) (h : abs (x - a) + abs (x + 5) ≥ 2 * abs (x + 5)) : x ≤ -1 := 
sorry

-- Part (II)
theorem part_II (a : ℝ) (h : ∀ x : ℝ, abs (x - a) + abs (x + 5) ≥ 6) : a ≥ 1 ∨ a ≤ -11 := 
sorry

end NUMINAMATH_GPT_part_I_part_II_l2200_220014


namespace NUMINAMATH_GPT_prime_divides_factorial_difference_l2200_220075

theorem prime_divides_factorial_difference (p : ℕ) (hp_prime : Nat.Prime p) (hp_ge_five : p ≥ 5) : 
  p^5 ∣ (Nat.factorial p - p) := by
  sorry

end NUMINAMATH_GPT_prime_divides_factorial_difference_l2200_220075


namespace NUMINAMATH_GPT_inequality_positive_reals_l2200_220045

theorem inequality_positive_reals (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (1 / a^2 + 1 / b^2 + 8 * a * b ≥ 8) ∧ (1 / a^2 + 1 / b^2 + 8 * a * b = 8 → a = b ∧ a = 1/2) :=
by
  sorry

end NUMINAMATH_GPT_inequality_positive_reals_l2200_220045


namespace NUMINAMATH_GPT_cars_meet_time_l2200_220082

theorem cars_meet_time (t : ℝ) (highway_length : ℝ) (speed_car1 : ℝ) (speed_car2 : ℝ)
  (h1 : highway_length = 105) (h2 : speed_car1 = 15) (h3 : speed_car2 = 20) :
  15 * t + 20 * t = 105 → t = 3 := by
  sorry

end NUMINAMATH_GPT_cars_meet_time_l2200_220082


namespace NUMINAMATH_GPT_buy_tshirts_l2200_220092

theorem buy_tshirts
  (P T : ℕ)
  (h1 : 3 * P + 6 * T = 1500)
  (h2 : P + 12 * T = 1500)
  (budget : ℕ)
  (budget_eq : budget = 800) :
  (budget / T) = 8 := by
  sorry

end NUMINAMATH_GPT_buy_tshirts_l2200_220092


namespace NUMINAMATH_GPT_find_value_of_a_l2200_220081

theorem find_value_of_a
  (a : ℝ)
  (h : (a + 3) * 2 * (-2 / 3) = -4) :
  a = -3 :=
sorry

end NUMINAMATH_GPT_find_value_of_a_l2200_220081


namespace NUMINAMATH_GPT_cost_price_percentage_l2200_220021

variables (CP MP SP : ℝ) (x : ℝ)

theorem cost_price_percentage (h1 : CP = (x / 100) * MP)
                             (h2 : SP = 0.5 * MP)
                             (h3 : SP = 2 * CP) :
                             x = 25 := by
  sorry

end NUMINAMATH_GPT_cost_price_percentage_l2200_220021


namespace NUMINAMATH_GPT_find_p_q_l2200_220068

theorem find_p_q (p q : ℤ)
  (h : (5 * d^2 - 4 * d + p) * (4 * d^2 + q * d - 5) = 20 * d^4 + 11 * d^3 - 45 * d^2 - 20 * d + 25) :
  p + q = 3 :=
sorry

end NUMINAMATH_GPT_find_p_q_l2200_220068


namespace NUMINAMATH_GPT_interest_calculation_l2200_220017

theorem interest_calculation :
  ∃ n : ℝ, 
  (1000 * 0.03 * n + 1400 * 0.05 * n = 350) →
  n = 3.5 := 
by 
  sorry

end NUMINAMATH_GPT_interest_calculation_l2200_220017


namespace NUMINAMATH_GPT_arithmetic_progression_impossible_geometric_progression_possible_l2200_220027

theorem arithmetic_progression_impossible (a b c : ℝ) (h1 : a = 2) (h2 : b = Real.sqrt 6) (h3 : c = 4.5) : 
  2 * b ≠ a + c :=
by {
    sorry
}

theorem geometric_progression_possible (a b c : ℝ) (h1 : a = 2) (h2 : b = Real.sqrt 6) (h3 : c = 4.5) : 
  ∃ r m : ℤ, (b / a)^r = (c / a)^m :=
by {
    sorry
}

end NUMINAMATH_GPT_arithmetic_progression_impossible_geometric_progression_possible_l2200_220027


namespace NUMINAMATH_GPT_smallest_integer_solution_l2200_220048

theorem smallest_integer_solution (x : ℤ) : 
  (10 * x * x - 40 * x + 36 = 0) → x = 2 :=
sorry

end NUMINAMATH_GPT_smallest_integer_solution_l2200_220048


namespace NUMINAMATH_GPT_batsman_average_after_17th_inning_l2200_220080

theorem batsman_average_after_17th_inning 
    (A : ℕ)  -- assuming A (the average before the 17th inning) is a natural number
    (h₁ : 16 * A + 85 = 17 * (A + 3)) : 
    A + 3 = 37 := by
  sorry

end NUMINAMATH_GPT_batsman_average_after_17th_inning_l2200_220080


namespace NUMINAMATH_GPT_napkin_coloring_l2200_220022

structure Napkin where
  top : ℝ
  bottom : ℝ
  left : ℝ
  right : ℝ

def intersects_vertically (n1 n2 : Napkin) : Prop :=
  n1.left ≤ n2.right ∧ n2.left ≤ n1.right

def intersects_horizontally (n1 n2 : Napkin) : Prop :=
  n1.bottom ≤ n2.top ∧ n2.bottom ≤ n1.top

def can_be_crossed_by_line (n1 n2 : Napkin) : Prop :=
  intersects_vertically n1 n2 ∨ intersects_horizontally n1 n2

theorem napkin_coloring
  (blue_napkins green_napkins : List Napkin)
  (h_cross : ∀ (b : Napkin) (g : Napkin), 
    b ∈ blue_napkins → g ∈ green_napkins → can_be_crossed_by_line b g) :
  ∃ (color : String) (h1 h2 : ℝ) (v : ℝ), 
    (color = "blue" ∧ ∀ b ∈ blue_napkins, (b.bottom ≤ h1 ∧ h1 ≤ b.top) ∨ (b.bottom ≤ h2 ∧ h2 ≤ b.top) ∨ (b.left ≤ v ∧ v ≤ b.right)) ∨
    (color = "green" ∧ ∀ g ∈ green_napkins, (g.bottom ≤ h1 ∧ h1 ≤ g.top) ∨ (g.bottom ≤ h2 ∧ h2 ≤ g.top) ∨ (g.left ≤ v ∧ v ≤ g.right)) :=
sorry

end NUMINAMATH_GPT_napkin_coloring_l2200_220022


namespace NUMINAMATH_GPT_john_alone_finishes_in_48_days_l2200_220013

theorem john_alone_finishes_in_48_days (J R : ℝ) (h1 : J + R = 1 / 24)
  (h2 : 16 * (J + R) = 16 / 24) (h3 : ∀ T : ℝ, J * T = 1 → T = 48) : 
  (J = 1 / 48) → (∀ T : ℝ, J * T = 1 → T = 48) :=
by
  intro hJohn
  sorry

end NUMINAMATH_GPT_john_alone_finishes_in_48_days_l2200_220013


namespace NUMINAMATH_GPT_solve_a_perpendicular_l2200_220047

theorem solve_a_perpendicular (a : ℝ) : 
  ((2 * a + 5) * (2 - a) + (a - 2) * (a + 3) = 0) ↔ (a = 2 ∨ a = -2) :=
by
  sorry

end NUMINAMATH_GPT_solve_a_perpendicular_l2200_220047


namespace NUMINAMATH_GPT_multiplier_of_reciprocal_l2200_220089

theorem multiplier_of_reciprocal (x m : ℝ) (h1 : x = 7) (h2 : x - 4 = m * (1 / x)) : m = 21 :=
by
  sorry

end NUMINAMATH_GPT_multiplier_of_reciprocal_l2200_220089


namespace NUMINAMATH_GPT_min_value_of_m_l2200_220091

def ellipse (x y : ℝ) := (y^2 / 16) + (x^2 / 9) = 1
def line (x y m : ℝ) := y = x + m
def shortest_distance (d : ℝ) := d = Real.sqrt 2

theorem min_value_of_m :
  ∃ (m : ℝ), (∀ (x y : ℝ), ellipse x y → ∃ d, shortest_distance d ∧ line x y m) 
  ∧ ∀ m', m' < m → ¬(∃ (x y : ℝ), ellipse x y ∧ ∃ d, shortest_distance d ∧ line x y m') :=
sorry

end NUMINAMATH_GPT_min_value_of_m_l2200_220091


namespace NUMINAMATH_GPT_arithmetic_sequence_properties_l2200_220012

theorem arithmetic_sequence_properties (a b c : ℝ) (h1 : ∃ d : ℝ, [2, a, b, c, 9] = [2, 2 + d, 2 + 2 * d, 2 + 3 * d, 2 + 4 * d]) : 
  c - a = 7 / 2 := 
by
  -- We assume the proof here
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_properties_l2200_220012


namespace NUMINAMATH_GPT_sum_of_primes_between_20_and_40_l2200_220076

theorem sum_of_primes_between_20_and_40 : 
  (23 + 29 + 31 + 37) = 120 := 
by
  -- Proof goes here
sorry

end NUMINAMATH_GPT_sum_of_primes_between_20_and_40_l2200_220076
