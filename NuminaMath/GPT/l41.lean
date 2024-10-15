import Mathlib

namespace NUMINAMATH_GPT_solve_eq1_solve_eq2_l41_4171

-- Define the problem for equation (1)
theorem solve_eq1 (x : Real) : (x - 1)^2 = 2 ↔ (x = 1 + Real.sqrt 2 ∨ x = 1 - Real.sqrt 2) :=
by 
  sorry

-- Define the problem for equation (2)
theorem solve_eq2 (x : Real) : x^2 - 6 * x - 7 = 0 ↔ (x = -1 ∨ x = 7) :=
by 
  sorry

end NUMINAMATH_GPT_solve_eq1_solve_eq2_l41_4171


namespace NUMINAMATH_GPT_sequence_item_l41_4138

theorem sequence_item (n : ℕ) (a_n : ℕ → Rat) (h : a_n n = 2 / (n^2 + n)) : a_n n = 1 / 15 → n = 5 := by
  sorry

end NUMINAMATH_GPT_sequence_item_l41_4138


namespace NUMINAMATH_GPT_pump_leak_drain_time_l41_4116

theorem pump_leak_drain_time {P L : ℝ} (hP : P = 0.25) (hPL : P - L = 0.05) : (1 / L) = 5 :=
by sorry

end NUMINAMATH_GPT_pump_leak_drain_time_l41_4116


namespace NUMINAMATH_GPT_a_plus_b_values_l41_4107

theorem a_plus_b_values (a b : ℝ) (h1 : abs a = 5) (h2 : abs b = 3) (h3 : abs (a - b) = b - a) : a + b = -2 ∨ a + b = -8 :=
sorry

end NUMINAMATH_GPT_a_plus_b_values_l41_4107


namespace NUMINAMATH_GPT_kate_hair_length_l41_4142

theorem kate_hair_length :
  ∀ (logans_hair : ℕ) (emilys_hair : ℕ) (kates_hair : ℕ),
  logans_hair = 20 →
  emilys_hair = logans_hair + 6 →
  kates_hair = emilys_hair / 2 →
  kates_hair = 13 :=
by
  intros logans_hair emilys_hair kates_hair
  sorry

end NUMINAMATH_GPT_kate_hair_length_l41_4142


namespace NUMINAMATH_GPT_wade_customers_l41_4192

theorem wade_customers (F : ℕ) (h1 : 2 * F + 6 * F + 72 = 296) : F = 28 := 
by 
  sorry

end NUMINAMATH_GPT_wade_customers_l41_4192


namespace NUMINAMATH_GPT_find_y_l41_4164

theorem find_y 
  (α : Real)
  (P : Real × Real)
  (P_coord : P = (-Real.sqrt 3, y))
  (sin_alpha : Real.sin α = Real.sqrt 13 / 13) :
  P.2 = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_y_l41_4164


namespace NUMINAMATH_GPT_flowers_in_each_basket_l41_4122

-- Definitions based on the conditions
def initial_flowers (d1 d2 : Nat) : Nat := d1 + d2
def grown_flowers (initial growth : Nat) : Nat := initial + growth
def remaining_flowers (grown dead : Nat) : Nat := grown - dead
def flowers_per_basket (remaining baskets : Nat) : Nat := remaining / baskets

-- Given conditions in Lean 4
theorem flowers_in_each_basket 
    (daughters_flowers : Nat) 
    (growth : Nat) 
    (dead : Nat) 
    (baskets : Nat) 
    (h_daughters : daughters_flowers = 5 + 5) 
    (h_growth : growth = 20) 
    (h_dead : dead = 10) 
    (h_baskets : baskets = 5) : 
    flowers_per_basket (remaining_flowers (grown_flowers (initial_flowers 5 5) growth) dead) baskets = 4 := 
sorry

end NUMINAMATH_GPT_flowers_in_each_basket_l41_4122


namespace NUMINAMATH_GPT_kim_boxes_on_tuesday_l41_4159

theorem kim_boxes_on_tuesday
  (sold_on_thursday : ℕ)
  (sold_on_wednesday : ℕ)
  (sold_on_tuesday : ℕ)
  (h1 : sold_on_thursday = 1200)
  (h2 : sold_on_wednesday = 2 * sold_on_thursday)
  (h3 : sold_on_tuesday = 2 * sold_on_wednesday) :
  sold_on_tuesday = 4800 :=
sorry

end NUMINAMATH_GPT_kim_boxes_on_tuesday_l41_4159


namespace NUMINAMATH_GPT_john_bought_notebooks_l41_4174

def pages_per_notebook : ℕ := 40
def pages_per_day : ℕ := 4
def total_days : ℕ := 50

theorem john_bought_notebooks : (pages_per_day * total_days) / pages_per_notebook = 5 :=
by
  sorry

end NUMINAMATH_GPT_john_bought_notebooks_l41_4174


namespace NUMINAMATH_GPT_leaves_count_l41_4126

theorem leaves_count {m n L : ℕ} (h1 : m + n = 10) (h2 : L = 5 * m + 2 * n) :
  ¬(L = 45 ∨ L = 39 ∨ L = 37 ∨ L = 31) :=
by
  sorry

end NUMINAMATH_GPT_leaves_count_l41_4126


namespace NUMINAMATH_GPT_axisymmetric_triangle_is_isosceles_l41_4144

-- Define a triangle and its properties
structure Triangle :=
  (a b c : ℝ) -- Triangle sides as real numbers
  (ha : a > 0)
  (hb : b > 0)
  (hc : c > 0)
  (h_triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b)

def is_axisymmetric (T : Triangle) : Prop :=
  -- Here define what it means for a triangle to be axisymmetric
  -- This is often represented as having at least two sides equal
  (T.a = T.b ∨ T.b = T.c ∨ T.c = T.a)

def is_isosceles (T : Triangle) : Prop :=
  -- Definition of an isosceles triangle
  (T.a = T.b ∨ T.b = T.c ∨ T.c = T.a)

-- The theorem to be proven
theorem axisymmetric_triangle_is_isosceles (T : Triangle) (h : is_axisymmetric T) : is_isosceles T :=
by {
  -- Proof would go here
  sorry
}

end NUMINAMATH_GPT_axisymmetric_triangle_is_isosceles_l41_4144


namespace NUMINAMATH_GPT_perfect_square_if_integer_l41_4167

theorem perfect_square_if_integer (n : ℤ) (k : ℤ) 
  (h : k = 2 + 2 * Int.sqrt (28 * n^2 + 1)) : ∃ m : ℤ, k = m^2 :=
by 
  sorry

end NUMINAMATH_GPT_perfect_square_if_integer_l41_4167


namespace NUMINAMATH_GPT_find_possible_values_of_n_l41_4168

theorem find_possible_values_of_n (n : ℕ) :
  (∃ k : ℕ, 1 ≤ k ∧ k ≤ n + 1 ∧ 
    (2*n*(2*n + 1))/2 - (n*k + (n*(n-1))/2) = 1615) ↔ (n = 34 ∨ n = 38) :=
by
  sorry

end NUMINAMATH_GPT_find_possible_values_of_n_l41_4168


namespace NUMINAMATH_GPT_sector_field_area_l41_4177

/-- Given a sector field with a circumference of 30 steps and a diameter of 16 steps, prove that its area is 120 square steps. --/
theorem sector_field_area (C : ℝ) (d : ℝ) (A : ℝ) : 
  C = 30 → d = 16 → A = 120 :=
by
  sorry

end NUMINAMATH_GPT_sector_field_area_l41_4177


namespace NUMINAMATH_GPT_smallest_d_for_inverse_domain_l41_4147

noncomputable def g (x : ℝ) : ℝ := 2 * (x + 1)^2 - 7

theorem smallest_d_for_inverse_domain : ∃ d : ℝ, (∀ x1 x2 : ℝ, x1 ≥ d → x2 ≥ d → g x1 = g x2 → x1 = x2) ∧ d = -1 :=
by
  use -1
  constructor
  · sorry
  · rfl

end NUMINAMATH_GPT_smallest_d_for_inverse_domain_l41_4147


namespace NUMINAMATH_GPT_rest_area_milepost_l41_4151

theorem rest_area_milepost 
  (milepost_fifth_exit : ℕ) 
  (milepost_fifteenth_exit : ℕ) 
  (rest_area_milepost : ℕ)
  (h1 : milepost_fifth_exit = 50)
  (h2 : milepost_fifteenth_exit = 350)
  (h3 : rest_area_milepost = (milepost_fifth_exit + (milepost_fifteenth_exit - milepost_fifth_exit) / 2)) :
  rest_area_milepost = 200 := 
by
  intros
  rw [h1, h2] at h3
  norm_num at h3
  exact h3

end NUMINAMATH_GPT_rest_area_milepost_l41_4151


namespace NUMINAMATH_GPT_fixed_points_and_zeros_no_fixed_points_range_b_l41_4176

def f (b c x : ℝ) : ℝ := x^2 + b * x + c

theorem fixed_points_and_zeros (b c : ℝ) (h1 : f b c (-3) = -3) (h2 : f b c 2 = 2) :
  ∃ x1 x2 : ℝ, f b c x1 = 0 ∧ f b c x2 = 0 ∧ x1 = -1 + Real.sqrt 7 ∧ x2 = -1 - Real.sqrt 7 :=
sorry

theorem no_fixed_points_range_b {b : ℝ} (h : ∀ x : ℝ, f b (b^2 / 4) x ≠ x) : 
  b > 1 / 3 ∨ b < -1 :=
sorry

end NUMINAMATH_GPT_fixed_points_and_zeros_no_fixed_points_range_b_l41_4176


namespace NUMINAMATH_GPT_average_increase_l41_4173

variable (A : ℕ) -- The batsman's average before the 17th inning
variable (runs_in_17th_inning : ℕ := 86) -- Runs made in the 17th inning
variable (new_average : ℕ := 38) -- The average after the 17th inning
variable (total_runs_16_innings : ℕ := 16 * A) -- Total runs after 16 innings
variable (total_runs_after_17_innings : ℕ := total_runs_16_innings + runs_in_17th_inning) -- Total runs after 17 innings
variable (total_runs_should_be : ℕ := 17 * new_average) -- Theoretical total runs after 17 innings

theorem average_increase :
  total_runs_after_17_innings = total_runs_should_be → (new_average - A) = 3 :=
by
  sorry

end NUMINAMATH_GPT_average_increase_l41_4173


namespace NUMINAMATH_GPT_eccentricity_range_of_hyperbola_l41_4172

open Real

noncomputable def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  (a > 0) ∧ (b > 0) ∧ (x^2 / a^2 - y^2 / b^2 = 1)

noncomputable def eccentricity_range :=
  ∀ (a b c : ℝ), 
    ∃ (e : ℝ),
      hyperbola a b (-c) 0 ∧ -- condition for point F
      (a + b > 0) ∧ -- additional conditions due to hyperbola properties
      (1 < e ∧ e < 2)
      
theorem eccentricity_range_of_hyperbola :
  eccentricity_range :=
by
  sorry

end NUMINAMATH_GPT_eccentricity_range_of_hyperbola_l41_4172


namespace NUMINAMATH_GPT_problem_l41_4178

theorem problem (n : ℝ) (h : (n - 2009)^2 + (2008 - n)^2 = 1) : (n - 2009) * (2008 - n) = 0 := 
by
  sorry

end NUMINAMATH_GPT_problem_l41_4178


namespace NUMINAMATH_GPT_binary_to_decimal_conversion_l41_4187

theorem binary_to_decimal_conversion : (1 * 2^5 + 1 * 2^4 + 0 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0) = 51 :=
by 
  sorry

end NUMINAMATH_GPT_binary_to_decimal_conversion_l41_4187


namespace NUMINAMATH_GPT_quadratic_inequality_l41_4179

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_inequality
  (a b c : ℝ)
  (h_pos : 0 < a)
  (h_roots : ∀ x : ℝ, a * x^2 + b * x + c ≠ 0)
  (x : ℝ) :
  f a b c x + f a b c (x - 1) - f a b c (x + 1) > -4 * a :=
  sorry

end NUMINAMATH_GPT_quadratic_inequality_l41_4179


namespace NUMINAMATH_GPT_find_x_intercept_l41_4148

variables (a x y : ℝ)
def l1 (a x y : ℝ) : Prop := (a + 2) * x + 3 * y = 5
def l2 (a x y : ℝ) : Prop := (a - 1) * x + 2 * y = 6
def are_parallel (a : ℝ) : Prop := (- (a + 2) / 3) = (- (a - 1) / 2)
def x_intercept_of_l1 (a x : ℝ) : Prop := l1 a x 0

theorem find_x_intercept (h : are_parallel a) : x_intercept_of_l1 7 (5 / 9) := 
sorry

end NUMINAMATH_GPT_find_x_intercept_l41_4148


namespace NUMINAMATH_GPT_derivative_at_one_l41_4115

noncomputable def f (x : ℝ) : ℝ := x^2 + Real.exp x

theorem derivative_at_one : deriv f 1 = 2 + Real.exp 1 := by
  sorry

end NUMINAMATH_GPT_derivative_at_one_l41_4115


namespace NUMINAMATH_GPT_problem_statement_l41_4139

theorem problem_statement (x y : ℝ) 
  (hA : A = (x + y) * (y - 3 * x))
  (hB : B = (x - y)^4 / (x - y)^2)
  (hCond : 2 * y + A = B - 6) :
  y = 2 * x^2 - 3 ∧ (y + 3)^2 - 2 * x * (x * y - 3) - 6 * x * (x + 1) = 0 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l41_4139


namespace NUMINAMATH_GPT_confidence_k_squared_l41_4188

-- Define the condition for 95% confidence relation between events A and B
def confidence_95 (A B : Prop) : Prop := 
  -- Placeholder for the actual definition, assume 95% confidence implies a specific condition
  True

-- Define the data value and critical value condition
def K_squared : ℝ := sorry  -- Placeholder for the actual K² value

theorem confidence_k_squared (A B : Prop) (h : confidence_95 A B) : K_squared > 3.841 := 
by
  sorry  -- Proof is not required, only the statement

end NUMINAMATH_GPT_confidence_k_squared_l41_4188


namespace NUMINAMATH_GPT_parabola_shifted_left_and_down_l41_4111

-- Define the original parabolic equation
def original_parabola (x : ℝ) : ℝ := 2 * x ^ 2 - 1

-- Define the transformed parabolic equation
def transformed_parabola (x : ℝ) : ℝ := 2 * (x + 1) ^ 2 - 3

-- Theorem statement
theorem parabola_shifted_left_and_down :
  ∀ x : ℝ, transformed_parabola x = 2 * (x + 1) ^ 2 - 3 :=
by 
  -- Proof Left as an exercise.
  sorry

end NUMINAMATH_GPT_parabola_shifted_left_and_down_l41_4111


namespace NUMINAMATH_GPT_transform_into_product_l41_4129

theorem transform_into_product : 447 * (Real.sin (75 * Real.pi / 180) + Real.sin (15 * Real.pi / 180)) = 447 * Real.sqrt 6 / 2 := by
  sorry

end NUMINAMATH_GPT_transform_into_product_l41_4129


namespace NUMINAMATH_GPT_egg_rolls_total_l41_4170

theorem egg_rolls_total (omar_egg_rolls karen_egg_rolls lily_egg_rolls : ℕ) :
  omar_egg_rolls = 219 → karen_egg_rolls = 229 → lily_egg_rolls = 275 → 
  omar_egg_rolls + karen_egg_rolls + lily_egg_rolls = 723 := 
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_egg_rolls_total_l41_4170


namespace NUMINAMATH_GPT_seating_arrangement_l41_4130

theorem seating_arrangement (M : ℕ) (h1 : 8 * M = 12 * M) : M = 3 :=
by
  sorry

end NUMINAMATH_GPT_seating_arrangement_l41_4130


namespace NUMINAMATH_GPT_largest_digit_to_correct_sum_l41_4156

theorem largest_digit_to_correct_sum :
  (725 + 864 + 991 = 2570) → (∃ (d : ℕ), d = 9 ∧ 
  (∃ (n1 : ℕ), n1 ∈ [702, 710, 711, 721, 715] ∧ 
  ∃ (n2 : ℕ), n2 ∈ [806, 805, 814, 854, 864] ∧ 
  ∃ (n3 : ℕ), n3 ∈ [918, 921, 931, 941, 981, 991] ∧ 
  n1 + n2 + n3 = n1 + n2 + n3 - 10))
    → d = 9 :=
by
  sorry

end NUMINAMATH_GPT_largest_digit_to_correct_sum_l41_4156


namespace NUMINAMATH_GPT_work_together_l41_4145

theorem work_together (W : ℝ) (Dx Dy : ℝ) (hx : Dx = 15) (hy : Dy = 30) : 
  (Dx * Dy) / (Dx + Dy) = 10 := 
by
  sorry

end NUMINAMATH_GPT_work_together_l41_4145


namespace NUMINAMATH_GPT_smallest_integer_to_make_multiple_of_five_l41_4182

/-- The smallest positive integer that can be added to 725 to make it a multiple of 5 is 5. -/
theorem smallest_integer_to_make_multiple_of_five : 
  ∃ k : ℕ, k > 0 ∧ (725 + k) % 5 = 0 ∧ ∀ m : ℕ, m > 0 ∧ (725 + m) % 5 = 0 → k ≤ m :=
sorry

end NUMINAMATH_GPT_smallest_integer_to_make_multiple_of_five_l41_4182


namespace NUMINAMATH_GPT_sandy_initial_payment_l41_4141

variable (P : ℝ) 

theorem sandy_initial_payment
  (h1 : (1.2 : ℝ) * (P + 200) = 1200) :
  P = 800 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_sandy_initial_payment_l41_4141


namespace NUMINAMATH_GPT_MarksScore_l41_4101

theorem MarksScore (h_highest : ℕ) (h_range : ℕ) (h_relation : h_highest - h_least = h_range) (h_mark_twice : Mark = 2 * h_least) : Mark = 46 :=
by
    let h_highest := 98
    let h_range := 75
    let h_least := h_highest - h_range
    let Mark := 2 * h_least
    have := h_relation
    have := h_mark_twice
    sorry

end NUMINAMATH_GPT_MarksScore_l41_4101


namespace NUMINAMATH_GPT_opposite_sides_range_l41_4186

theorem opposite_sides_range (a : ℝ) :
  (3 * (-3) - 2 * (-1) - a) * (3 * 4 - 2 * (-6) - a) < 0 ↔ -7 < a ∧ a < 24 :=
by
  simp
  sorry

end NUMINAMATH_GPT_opposite_sides_range_l41_4186


namespace NUMINAMATH_GPT_cylinder_ratio_l41_4183

theorem cylinder_ratio
  (V : ℝ) (R H : ℝ)
  (hV : V = 1000)
  (hVolume : π * R^2 * H = V) :
  H / R = 1 :=
by
  sorry

end NUMINAMATH_GPT_cylinder_ratio_l41_4183


namespace NUMINAMATH_GPT_pqrs_sum_l41_4180

noncomputable def distinct_real_numbers (p q r s : ℝ) : Prop :=
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s

theorem pqrs_sum (p q r s : ℝ) 
  (h1 : r + s = 12 * p)
  (h2 : r * s = -13 * q)
  (h3 : p + q = 12 * r)
  (h4 : p * q = -13 * s)
  (distinct : distinct_real_numbers p q r s) :
  p + q + r + s = -13 :=
  sorry

end NUMINAMATH_GPT_pqrs_sum_l41_4180


namespace NUMINAMATH_GPT_simplify_expr1_simplify_expr2_l41_4198

variable (a b t : ℝ)

theorem simplify_expr1 : 6 * a^2 - 2 * a * b - 2 * (3 * a^2 - (1 / 2) * a * b) = -a * b :=
by
  sorry

theorem simplify_expr2 : -(t^2 - t - 1) + (2 * t^2 - 3 * t + 1) = t^2 - 2 * t + 2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expr1_simplify_expr2_l41_4198


namespace NUMINAMATH_GPT_seeds_distributed_equally_l41_4163

theorem seeds_distributed_equally (S G n seeds_per_small_garden : ℕ) 
  (hS : S = 42) 
  (hG : G = 36) 
  (hn : n = 3) 
  (h_seeds : seeds_per_small_garden = (S - G) / n) : 
  seeds_per_small_garden = 2 := by
  rw [hS, hG, hn] at h_seeds
  simp at h_seeds
  exact h_seeds

end NUMINAMATH_GPT_seeds_distributed_equally_l41_4163


namespace NUMINAMATH_GPT_tiles_difference_eighth_sixth_l41_4146

-- Define the side length of the nth square
def side_length (n : ℕ) : ℕ := n

-- Define the number of tiles given the side length
def number_of_tiles (n : ℕ) : ℕ := n * n

-- State the theorem about the difference in tiles between the 8th and 6th squares
theorem tiles_difference_eighth_sixth :
  number_of_tiles (side_length 8) - number_of_tiles (side_length 6) = 28 :=
by
  -- skipping the proof
  sorry

end NUMINAMATH_GPT_tiles_difference_eighth_sixth_l41_4146


namespace NUMINAMATH_GPT_min_abs_sum_is_5_l41_4108

noncomputable def min_abs_sum (x : ℝ) : ℝ :=
  |x + 1| + |x + 3| + |x + 6|

theorem min_abs_sum_is_5 : ∃ x : ℝ, (∀ y : ℝ, min_abs_sum y ≥ min_abs_sum x) ∧ min_abs_sum x = 5 :=
by
  use -3
  sorry

end NUMINAMATH_GPT_min_abs_sum_is_5_l41_4108


namespace NUMINAMATH_GPT_find_Y_l41_4106

theorem find_Y 
  (a b c d X Y : ℕ)
  (h1 : a + b + c + d = 40)
  (h2 : X + Y + c + b = 40)
  (h3 : a + b + X = 30)
  (h4 : c + d + Y = 30)
  (h5 : X = 9) 
  : Y = 11 := 
by 
  sorry

end NUMINAMATH_GPT_find_Y_l41_4106


namespace NUMINAMATH_GPT_expression_not_equal_l41_4118

variable (a b c : ℝ)

theorem expression_not_equal :
  (a - (b - c)) ≠ (a - b - c) :=
by sorry

end NUMINAMATH_GPT_expression_not_equal_l41_4118


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l41_4190

def setA : Set ℝ := {x | abs (x - 1) < 2}

def setB : Set ℝ := {x | x^2 + x - 2 > 0}

theorem intersection_of_A_and_B :
  (setA ∩ setB) = {x | 1 < x ∧ x < 3} :=
sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l41_4190


namespace NUMINAMATH_GPT_other_endpoint_of_diameter_l41_4117

theorem other_endpoint_of_diameter (center endpoint : ℝ × ℝ) (hc : center = (1, 2)) (he : endpoint = (4, 6)) :
    ∃ other_endpoint : ℝ × ℝ, other_endpoint = (-2, -2) :=
by
  sorry

end NUMINAMATH_GPT_other_endpoint_of_diameter_l41_4117


namespace NUMINAMATH_GPT_yellow_curved_given_curved_l41_4114

variable (P_green : ℝ) (P_yellow : ℝ) (P_straight : ℝ) (P_curved : ℝ)
variable (P_red_given_straight : ℝ) 

-- Given conditions
variables (h1 : P_green = 3 / 4) 
          (h2 : P_yellow = 1 / 4) 
          (h3 : P_straight = 1 / 2) 
          (h4 : P_curved = 1 / 2)
          (h5 : P_red_given_straight = 1 / 3)

-- To be proven
theorem yellow_curved_given_curved : (P_yellow * P_curved) / P_curved = 1 / 4 :=
by
sorry

end NUMINAMATH_GPT_yellow_curved_given_curved_l41_4114


namespace NUMINAMATH_GPT_amount_left_for_gas_and_maintenance_l41_4165

def monthly_income : ℤ := 3200
def rent : ℤ := 1250
def utilities : ℤ := 150
def retirement_savings : ℤ := 400
def groceries_eating_out : ℤ := 300
def insurance : ℤ := 200
def miscellaneous : ℤ := 200
def car_payment : ℤ := 350

def total_expenses : ℤ :=
  rent + utilities + retirement_savings + groceries_eating_out + insurance + miscellaneous + car_payment

theorem amount_left_for_gas_and_maintenance : monthly_income - total_expenses = 350 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_amount_left_for_gas_and_maintenance_l41_4165


namespace NUMINAMATH_GPT_base_conversion_problem_l41_4185

theorem base_conversion_problem (b : ℕ) (h : b^2 + b + 3 = 34) : b = 6 :=
sorry

end NUMINAMATH_GPT_base_conversion_problem_l41_4185


namespace NUMINAMATH_GPT_minimum_value_l41_4135

open Real

theorem minimum_value (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x + y + z = 9) :
  (x ^ 2 + y ^ 2) / (x + y) + (x ^ 2 + z ^ 2) / (x + z) + (y ^ 2 + z ^ 2) / (y + z) ≥ 9 :=
by sorry

end NUMINAMATH_GPT_minimum_value_l41_4135


namespace NUMINAMATH_GPT_sum_of_smallest_and_largest_l41_4166

theorem sum_of_smallest_and_largest (n : ℕ) (h : Odd n) (b z : ℤ)
  (h_mean : z = b + n - 1 - 2 / (n : ℤ)) :
  ((b - 2) + (b + 2 * (n - 2))) = 2 * z - 4 + 4 / (n : ℤ) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_smallest_and_largest_l41_4166


namespace NUMINAMATH_GPT_least_sum_of_exponents_l41_4113

theorem least_sum_of_exponents (a b c : ℕ) (ha : 2^a ∣ 520) (hb : 2^b ∣ 520) (hc : 2^c ∣ 520) (h : a ≠ b ∧ b ≠ c ∧ a ≠ c) : 
  a + b + c = 12 :=
by
  sorry

end NUMINAMATH_GPT_least_sum_of_exponents_l41_4113


namespace NUMINAMATH_GPT_triangle_sum_l41_4112

def triangle (a b c : ℕ) : ℤ := a * b - c

theorem triangle_sum :
  triangle 2 3 5 + triangle 1 4 7 = -2 :=
by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_triangle_sum_l41_4112


namespace NUMINAMATH_GPT_part_I_part_II_l41_4196

noncomputable def f (x b c : ℝ) := x^2 + b*x + c

theorem part_I (x_1 x_2 b c : ℝ)
  (h1 : f x_1 b c = x_1) (h2 : f x_2 b c = x_2) (h3 : x_1 > 0) (h4 : x_2 - x_1 > 1) :
  b^2 > 2 * (b + 2 * c) :=
sorry

theorem part_II (x_1 x_2 b c t : ℝ)
  (h1 : f x_1 b c = x_1) (h2 : f x_2 b c = x_2) (h3 : x_1 > 0) (h4 : x_2 - x_1 > 1) (h5 : 0 < t ∧ t < x_1) :
  f t b c > x_1 :=
sorry

end NUMINAMATH_GPT_part_I_part_II_l41_4196


namespace NUMINAMATH_GPT_dvds_bought_online_l41_4160

theorem dvds_bought_online (total_dvds : ℕ) (store_dvds : ℕ) (online_dvds : ℕ) :
  total_dvds = 10 → store_dvds = 8 → online_dvds = total_dvds - store_dvds → online_dvds = 2 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_dvds_bought_online_l41_4160


namespace NUMINAMATH_GPT_profit_percentage_l41_4181

theorem profit_percentage (CP SP : ℝ) (hCP : CP = 500) (hSP : SP = 725) : 
  100 * (SP - CP) / CP = 45 :=
by
  sorry

end NUMINAMATH_GPT_profit_percentage_l41_4181


namespace NUMINAMATH_GPT_count_divisible_by_3_in_range_l41_4158

theorem count_divisible_by_3_in_range (a b : ℤ) :
  a = 252 → b = 549 → (∃ n : ℕ, (a ≤ 3 * n ∧ 3 * n ≤ b) ∧ (b - a) / 3 = (100 : ℝ)) :=
by
  intros ha hb
  have h1 : ∃ k : ℕ, a = 3 * k := by sorry
  have h2 : ∃ m : ℕ, b = 3 * m := by sorry
  sorry

end NUMINAMATH_GPT_count_divisible_by_3_in_range_l41_4158


namespace NUMINAMATH_GPT_youtube_more_than_tiktok_l41_4194

-- Definitions for followers in different social media platforms
def instagram_followers : ℕ := 240
def facebook_followers : ℕ := 500
def total_followers : ℕ := 3840

-- Number of followers on Twitter is half the sum of followers on Instagram and Facebook
def twitter_followers : ℕ := (instagram_followers + facebook_followers) / 2

-- Number of followers on TikTok is 3 times the followers on Twitter
def tiktok_followers : ℕ := 3 * twitter_followers

-- Calculate the number of followers on all social media except YouTube
def other_followers : ℕ := instagram_followers + facebook_followers + twitter_followers + tiktok_followers

-- Number of followers on YouTube
def youtube_followers : ℕ := total_followers - other_followers

-- Prove the number of followers on YouTube is greater than TikTok by a certain amount
theorem youtube_more_than_tiktok : youtube_followers - tiktok_followers = 510 := by
  -- Sorry is a placeholder for the proof
  sorry

end NUMINAMATH_GPT_youtube_more_than_tiktok_l41_4194


namespace NUMINAMATH_GPT_find_number_l41_4134

-- Define the conditions
variables (y : ℝ) (Some_number : ℝ) (x : ℝ)

-- State the given equation
def equation := 19 * (x + y) + Some_number = 19 * (-x + y) - 21

-- State the proposition to prove
theorem find_number (h : equation 1 y Some_number) : Some_number = -59 :=
sorry

end NUMINAMATH_GPT_find_number_l41_4134


namespace NUMINAMATH_GPT_problem_statement_l41_4131

def class_of_rem (k : ℕ) : Set ℤ := {n | ∃ m : ℤ, n = 4 * m + k}

theorem problem_statement : (2013 ∈ class_of_rem 1) ∧ 
                            (-2 ∈ class_of_rem 2) ∧ 
                            (∀ x : ℤ, x ∈ class_of_rem 0 ∨ x ∈ class_of_rem 1 ∨ x ∈ class_of_rem 2 ∨ x ∈ class_of_rem 3) ∧ 
                            (∀ a b : ℤ, (∃ k : ℕ, (a ∈ class_of_rem k ∧ b ∈ class_of_rem k)) ↔ (a - b) ∈ class_of_rem 0) :=
by
  -- each of the statements should hold true
  sorry

end NUMINAMATH_GPT_problem_statement_l41_4131


namespace NUMINAMATH_GPT_halfway_between_l41_4110

theorem halfway_between (a b : ℚ) (h1 : a = 1/12) (h2 : b = 1/15) : (a + b) / 2 = 3 / 40 := by
  -- proofs go here
  sorry

end NUMINAMATH_GPT_halfway_between_l41_4110


namespace NUMINAMATH_GPT_stock_value_order_l41_4195

-- Define the initial investment and yearly changes
def initialInvestment : Float := 100
def firstYearChangeA : Float := 1.30
def firstYearChangeB : Float := 0.70
def firstYearChangeG : Float := 1.10
def firstYearChangeD : Float := 1.00 -- unchanged

def secondYearChangeA : Float := 0.90
def secondYearChangeB : Float := 1.35
def secondYearChangeG : Float := 1.05
def secondYearChangeD : Float := 1.10

-- Calculate the final values after two years
def finalValueA : Float := initialInvestment * firstYearChangeA * secondYearChangeA
def finalValueB : Float := initialInvestment * firstYearChangeB * secondYearChangeB
def finalValueG : Float := initialInvestment * firstYearChangeG * secondYearChangeG
def finalValueD : Float := initialInvestment * firstYearChangeD * secondYearChangeD

-- Theorem statement - Prove that the final order of the values is B < D < G < A
theorem stock_value_order : finalValueB < finalValueD ∧ finalValueD < finalValueG ∧ finalValueG < finalValueA := by
  sorry

end NUMINAMATH_GPT_stock_value_order_l41_4195


namespace NUMINAMATH_GPT_infinite_series_sum_l41_4132

theorem infinite_series_sum :
  ∑' (n : ℕ), (1 / (1 + 3^n : ℝ) - 1 / (1 + 3^(n+1) : ℝ)) = 1/2 := 
sorry

end NUMINAMATH_GPT_infinite_series_sum_l41_4132


namespace NUMINAMATH_GPT_questions_left_blank_l41_4175

-- Definitions based on the conditions
def total_questions : Nat := 60
def word_problems : Nat := 20
def add_subtract_problems : Nat := 25
def algebra_problems : Nat := 10
def geometry_problems : Nat := 5
def total_time : Nat := 90

def time_per_word_problem : Nat := 2
def time_per_add_subtract_problem : Float := 1.5
def time_per_algebra_problem : Nat := 3
def time_per_geometry_problem : Nat := 4

def word_problems_answered : Nat := 15
def add_subtract_problems_answered : Nat := 22
def algebra_problems_answered : Nat := 8
def geometry_problems_answered : Nat := 3

-- The final goal is to prove that Steve left 12 questions blank
theorem questions_left_blank :
  total_questions - (word_problems_answered + add_subtract_problems_answered + algebra_problems_answered + geometry_problems_answered) = 12 :=
by
  sorry

end NUMINAMATH_GPT_questions_left_blank_l41_4175


namespace NUMINAMATH_GPT_cosine_of_3pi_over_2_l41_4100

theorem cosine_of_3pi_over_2 : Real.cos (3 * Real.pi / 2) = 0 := by
  sorry

end NUMINAMATH_GPT_cosine_of_3pi_over_2_l41_4100


namespace NUMINAMATH_GPT_width_of_field_l41_4128

noncomputable def field_width 
  (field_length : ℝ) 
  (rope_length : ℝ)
  (grazing_area : ℝ) : ℝ :=
if field_length > 2 * rope_length 
then rope_length
else grazing_area

theorem width_of_field 
  (field_length : ℝ := 45)
  (rope_length : ℝ := 22)
  (grazing_area : ℝ := 380.132711084365) : field_width field_length rope_length grazing_area = rope_length :=
by 
  sorry

end NUMINAMATH_GPT_width_of_field_l41_4128


namespace NUMINAMATH_GPT_find_integers_for_perfect_square_l41_4105

def is_perfect_square (n : ℤ) : Prop := ∃ m : ℤ, n = m * m

theorem find_integers_for_perfect_square :
  {x : ℤ | is_perfect_square (x^4 + x^3 + x^2 + x + 1)} = {-1, 0, 3} :=
by
  sorry

end NUMINAMATH_GPT_find_integers_for_perfect_square_l41_4105


namespace NUMINAMATH_GPT_stock_percent_change_l41_4153

variable (x : ℝ)

theorem stock_percent_change (h1 : ∀ x, 0.75 * x = x * 0.75)
                             (h2 : ∀ x, 1.05 * x = 0.75 * x + 0.3 * 0.75 * x):
    ((1.05 * x - x) / x) * 100 = 5 :=
by
  sorry

end NUMINAMATH_GPT_stock_percent_change_l41_4153


namespace NUMINAMATH_GPT_total_fence_poles_l41_4150

def num_poles_per_side : ℕ := 27
def sides_of_square : ℕ := 4
def corners_of_square : ℕ := 4

theorem total_fence_poles : 
  (num_poles_per_side * sides_of_square) - corners_of_square = 104 :=
  sorry

end NUMINAMATH_GPT_total_fence_poles_l41_4150


namespace NUMINAMATH_GPT_point_of_tangency_l41_4154

theorem point_of_tangency (x y : ℝ) (h : (y = x^3 + x - 2)) (slope : 4 = 3 * x^2 + 1) : (x, y) = (-1, -4) := 
sorry

end NUMINAMATH_GPT_point_of_tangency_l41_4154


namespace NUMINAMATH_GPT_quadratic_equation_has_root_l41_4161

theorem quadratic_equation_has_root (a b c : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : c ≠ 0) :
  ∃ (x : ℝ), (a * x^2 + 2 * b * x + c = 0) ∨
             (b * x^2 + 2 * c * x + a = 0) ∨
             (c * x^2 + 2 * a * x + b = 0) :=
sorry

end NUMINAMATH_GPT_quadratic_equation_has_root_l41_4161


namespace NUMINAMATH_GPT_percentage_proof_l41_4119

theorem percentage_proof (a : ℝ) (paise : ℝ) (x : ℝ) (h1: paise = 85) (h2: a = 170) : 
  (x/100) * a = paise ↔ x = 50 := 
by
  -- The setup includes:
  -- paise = 85
  -- a = 170
  -- We prove that x% of 170 equals 85 if and only if x = 50.
  sorry

end NUMINAMATH_GPT_percentage_proof_l41_4119


namespace NUMINAMATH_GPT_sandy_remaining_puppies_l41_4103

-- Definitions from the problem
def initial_puppies : ℕ := 8
def given_away_puppies : ℕ := 4

-- Theorem statement
theorem sandy_remaining_puppies : initial_puppies - given_away_puppies = 4 := by
  sorry

end NUMINAMATH_GPT_sandy_remaining_puppies_l41_4103


namespace NUMINAMATH_GPT_segments_have_common_point_l41_4184

-- Define the predicate that checks if two segments intersect
def segments_intersect (seg1 seg2 : ℝ × ℝ) : Prop :=
  let (a1, b1) := seg1
  let (a2, b2) := seg2
  max a1 a2 ≤ min b1 b2

-- Define the main theorem
theorem segments_have_common_point (segments : Fin 2019 → ℝ × ℝ)
  (h_intersect : ∀ (i j : Fin 2019), i ≠ j → segments_intersect (segments i) (segments j)) :
  ∃ p : ℝ, ∀ i : Fin 2019, (segments i).1 ≤ p ∧ p ≤ (segments i).2 :=
by
  sorry

end NUMINAMATH_GPT_segments_have_common_point_l41_4184


namespace NUMINAMATH_GPT_projectile_reaches_100_feet_l41_4162

noncomputable def projectile_height (t : ℝ) : ℝ :=
  -16 * t^2 + 80 * t

theorem projectile_reaches_100_feet :
  ∃ t : ℝ, t = 2.5 ∧ projectile_height t = 100 :=
by
  use 2.5
  sorry

end NUMINAMATH_GPT_projectile_reaches_100_feet_l41_4162


namespace NUMINAMATH_GPT_remainder_of_4521_l41_4191

theorem remainder_of_4521 (h1 : ∃ d : ℕ, d = 88)
  (h2 : 3815 % 88 = 31) : 4521 % 88 = 33 :=
sorry

end NUMINAMATH_GPT_remainder_of_4521_l41_4191


namespace NUMINAMATH_GPT_circles_intersect_l41_4197

def C1 (x y : ℝ) : Prop := x^2 + (y-1)^2 = 1
def C2 (x y a : ℝ) : Prop := (x-a)^2 + (y-1)^2 = 16

theorem circles_intersect (a : ℝ) (h : a > 0) :
  (∀ x y : ℝ, C1 x y → ∃ x' y' : ℝ, C2 x' y' a) ↔ 3 < a ∧ a < 4 :=
sorry

end NUMINAMATH_GPT_circles_intersect_l41_4197


namespace NUMINAMATH_GPT_points_distance_within_rectangle_l41_4169

theorem points_distance_within_rectangle :
  ∀ (points : Fin 6 → (ℝ × ℝ)), (∀ i, 0 ≤ (points i).1 ∧ (points i).1 ≤ 3 ∧ 0 ≤ (points i).2 ∧ (points i).2 ≤ 4) →
  ∃ (i j : Fin 6), i ≠ j ∧ dist (points i) (points j) ≤ Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_points_distance_within_rectangle_l41_4169


namespace NUMINAMATH_GPT_fill_cistern_time_l41_4136

theorem fill_cistern_time (R1 R2 R3 : ℝ) (H1 : R1 = 1/10) (H2 : R2 = 1/12) (H3 : R3 = 1/40) : 
  (1 / (R1 + R2 - R3)) = (120 / 19) :=
by
  sorry

end NUMINAMATH_GPT_fill_cistern_time_l41_4136


namespace NUMINAMATH_GPT_decompose_expression_l41_4127

-- Define the variables a and b as real numbers
variables (a b : ℝ)

-- State the theorem corresponding to the proof problem
theorem decompose_expression : 9 * a^2 * b - b = b * (3 * a + 1) * (3 * a - 1) :=
by
  sorry

end NUMINAMATH_GPT_decompose_expression_l41_4127


namespace NUMINAMATH_GPT_empty_seats_l41_4104

theorem empty_seats (total_seats : ℕ) (people_watching : ℕ) (h_total_seats : total_seats = 750) (h_people_watching : people_watching = 532) : 
  total_seats - people_watching = 218 :=
by
  sorry

end NUMINAMATH_GPT_empty_seats_l41_4104


namespace NUMINAMATH_GPT_average_mb_per_hour_of_music_l41_4125

/--
Given a digital music library:
- It contains 14 days of music.
- The first 7 days use 10,000 megabytes of disk space.
- The next 7 days use 14,000 megabytes of disk space.
- Each day has 24 hours.

Prove that the average megabytes per hour of music in this library is 71 megabytes.
-/
theorem average_mb_per_hour_of_music
  (days_total : ℕ) 
  (days_first : ℕ) 
  (days_second : ℕ) 
  (mb_first : ℕ) 
  (mb_second : ℕ) 
  (hours_per_day : ℕ) 
  (total_mb : ℕ) 
  (total_hours : ℕ) :
  days_total = 14 →
  days_first = 7 →
  days_second = 7 →
  mb_first = 10000 →
  mb_second = 14000 →
  hours_per_day = 24 →
  total_mb = mb_first + mb_second →
  total_hours = days_total * hours_per_day →
  total_mb / total_hours = 71 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end NUMINAMATH_GPT_average_mb_per_hour_of_music_l41_4125


namespace NUMINAMATH_GPT_range_of_a_l41_4189

-- Defining the function f
noncomputable def f (x a : ℝ) : ℝ :=
  (Real.exp x) * (2 * x - 1) - a * x + a

-- Main statement
theorem range_of_a (a : ℝ)
  (h1 : a < 1)
  (h2 : ∃ x0 x1 : ℤ, x0 ≠ x1 ∧ f x0 a ≤ 0 ∧ f x1 a ≤ 0) :
  (5 / (3 * Real.exp 2)) < a ∧ a ≤ (3 / (2 * Real.exp 1)) :=
sorry

end NUMINAMATH_GPT_range_of_a_l41_4189


namespace NUMINAMATH_GPT_cos_arcsin_eq_tan_arcsin_eq_l41_4120

open Real

theorem cos_arcsin_eq (h : arcsin (3 / 5) = θ) : cos (arcsin (3 / 5)) = 4 / 5 := by
  sorry

theorem tan_arcsin_eq (h : arcsin (3 / 5) = θ) : tan (arcsin (3 / 5)) = 3 / 4 := by
  sorry

end NUMINAMATH_GPT_cos_arcsin_eq_tan_arcsin_eq_l41_4120


namespace NUMINAMATH_GPT_calculate_final_amount_l41_4123

def initial_amount : ℝ := 7500
def first_year_rate : ℝ := 0.20
def second_year_rate : ℝ := 0.25

def first_year_interest (p : ℝ) (r : ℝ) : ℝ := p * r
def amount_after_first_year (p : ℝ) (i : ℝ) : ℝ := p + i

def second_year_interest (p : ℝ) (r : ℝ) : ℝ := p * r
def amount_after_second_year (p : ℝ) (i : ℝ) : ℝ := p + i

theorem calculate_final_amount :
  let initial : ℝ := initial_amount
  let interest1 : ℝ := first_year_interest initial first_year_rate
  let amount1 : ℝ := amount_after_first_year initial interest1
  let interest2 : ℝ := second_year_interest amount1 second_year_rate
  let final_amount : ℝ := amount_after_second_year amount1 interest2
  final_amount = 11250 := by
  sorry

end NUMINAMATH_GPT_calculate_final_amount_l41_4123


namespace NUMINAMATH_GPT_possible_values_of_polynomial_l41_4133

theorem possible_values_of_polynomial (x : ℝ) (h : x^2 - 7 * x + 12 < 0) : 
48 < x^2 + 7 * x + 12 ∧ x^2 + 7 * x + 12 < 64 :=
sorry

end NUMINAMATH_GPT_possible_values_of_polynomial_l41_4133


namespace NUMINAMATH_GPT_sum_a3_a4_a5_a6_l41_4102

theorem sum_a3_a4_a5_a6 (S : ℕ → ℕ) (h : ∀ n, S n = n^2 + 2 * n) : S 6 - S 2 = 40 :=
by
  sorry

end NUMINAMATH_GPT_sum_a3_a4_a5_a6_l41_4102


namespace NUMINAMATH_GPT_principal_amount_l41_4109

theorem principal_amount
  (SI : ℝ) (R : ℝ) (T : ℝ)
  (h1 : SI = 155) (h2 : R = 4.783950617283951) (h3 : T = 4) :
  SI * 100 / (R * T) = 810.13 := 
  by 
    -- proof omitted
    sorry

end NUMINAMATH_GPT_principal_amount_l41_4109


namespace NUMINAMATH_GPT_find_certain_number_l41_4152

theorem find_certain_number (x : ℝ) : 136 - 0.35 * x = 31 -> x = 300 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_certain_number_l41_4152


namespace NUMINAMATH_GPT_distance_formula_example_l41_4121

variable (x1 y1 x2 y2 : ℝ)

theorem distance_formula_example : dist (3, -1) (-4, 3) = Real.sqrt 65 :=
by
  let x1 := 3
  let y1 := -1
  let x2 := -4
  let y2 := 3
  sorry

end NUMINAMATH_GPT_distance_formula_example_l41_4121


namespace NUMINAMATH_GPT_min_restoration_time_l41_4199

/-- Prove the minimum time required to complete the restoration work of three handicrafts. -/

def shaping_time_A : Nat := 9
def shaping_time_B : Nat := 16
def shaping_time_C : Nat := 10

def painting_time_A : Nat := 15
def painting_time_B : Nat := 8
def painting_time_C : Nat := 14

theorem min_restoration_time : 
  (shaping_time_A + painting_time_A + painting_time_C + painting_time_B) = 46 := by
  sorry

end NUMINAMATH_GPT_min_restoration_time_l41_4199


namespace NUMINAMATH_GPT_prime_sum_eq_14_l41_4157

theorem prime_sum_eq_14 (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (h : q^5 - 2 * p^2 = 1) : p + q = 14 := 
sorry

end NUMINAMATH_GPT_prime_sum_eq_14_l41_4157


namespace NUMINAMATH_GPT_theta_in_first_quadrant_l41_4193

noncomputable def quadrant_of_theta (theta : ℝ) (h1 : Real.sin (Real.pi + theta) < 0) (h2 : Real.cos (Real.pi - theta) < 0) : ℕ :=
  if 0 < Real.sin theta ∧ 0 < Real.cos theta then 1 else sorry

theorem theta_in_first_quadrant (theta : ℝ) (h1 : Real.sin (Real.pi + theta) < 0) (h2 : Real.cos (Real.pi - theta) < 0) :
  quadrant_of_theta theta h1 h2 = 1 :=
by
  sorry

end NUMINAMATH_GPT_theta_in_first_quadrant_l41_4193


namespace NUMINAMATH_GPT_percentage_water_mixture_l41_4124

theorem percentage_water_mixture 
  (volume_A : ℝ) (volume_B : ℝ) (volume_C : ℝ)
  (ratio_A : ℝ := 5) (ratio_B : ℝ := 3) (ratio_C : ℝ := 2)
  (percentage_water_A : ℝ := 0.20) (percentage_water_B : ℝ := 0.35) (percentage_water_C : ℝ := 0.50) :
  (volume_A = ratio_A) → (volume_B = ratio_B) → (volume_C = ratio_C) → 
  ((percentage_water_A * volume_A + percentage_water_B * volume_B + percentage_water_C * volume_C) /
   (ratio_A + ratio_B + ratio_C)) * 100 = 30.5 := 
by 
  intros hA hB hC
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_percentage_water_mixture_l41_4124


namespace NUMINAMATH_GPT_flowers_per_vase_l41_4143

theorem flowers_per_vase (carnations roses vases total_flowers flowers_per_vase : ℕ)
  (h1 : carnations = 7)
  (h2 : roses = 47)
  (h3 : vases = 9)
  (h4 : total_flowers = carnations + roses)
  (h5 : flowers_per_vase = total_flowers / vases):
  flowers_per_vase = 6 := 
by {
  sorry
}

end NUMINAMATH_GPT_flowers_per_vase_l41_4143


namespace NUMINAMATH_GPT_min_value_of_quadratic_l41_4137

theorem min_value_of_quadratic (x : ℝ) : ∃ m : ℝ, (∀ x, x^2 + 10 * x ≥ m) ∧ m = -25 := by
  sorry

end NUMINAMATH_GPT_min_value_of_quadratic_l41_4137


namespace NUMINAMATH_GPT_gcd_of_45_75_90_l41_4140

def gcd_three_numbers (a b c : ℕ) : ℕ :=
  Nat.gcd (Nat.gcd a b) c

theorem gcd_of_45_75_90 : gcd_three_numbers 45 75 90 = 15 := by
  sorry

end NUMINAMATH_GPT_gcd_of_45_75_90_l41_4140


namespace NUMINAMATH_GPT_cos_value_proof_l41_4149

variable (α : Real)
variable (h1 : -Real.pi / 2 < α ∧ α < 0)
variable (h2 : Real.sin (α + Real.pi / 3) + Real.sin α = -(4 * Real.sqrt 3) / 5)

theorem cos_value_proof : Real.cos (α + 2 * Real.pi / 3) = 4 / 5 :=
by
  sorry

end NUMINAMATH_GPT_cos_value_proof_l41_4149


namespace NUMINAMATH_GPT_election_winner_votes_l41_4155

variable (V : ℝ) (winner_votes : ℝ) (winner_margin : ℝ)
variable (condition1 : V > 0)
variable (condition2 : winner_votes = 0.60 * V)
variable (condition3 : winner_margin = 240)

theorem election_winner_votes (h : winner_votes - 0.40 * V = winner_margin) : winner_votes = 720 := by
  sorry

end NUMINAMATH_GPT_election_winner_votes_l41_4155
