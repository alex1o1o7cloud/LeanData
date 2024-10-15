import Mathlib

namespace NUMINAMATH_GPT_find_angle_D_l590_59038

theorem find_angle_D (A B C D E F : ℝ) (hA : A = 50) (hB : B = 35) (hC : C = 40) 
  (triangle_sum1 : A + B + C + E + F = 180) (triangle_sum2 : D + E + F = 180) : 
  D = 125 :=
by
  -- Only adding a comment, proof omitted for the purpose of this task
  sorry

end NUMINAMATH_GPT_find_angle_D_l590_59038


namespace NUMINAMATH_GPT_cousins_arrangement_l590_59077

def number_of_arrangements (cousins rooms : ℕ) (min_empty_rooms : ℕ) : ℕ := sorry

theorem cousins_arrangement : number_of_arrangements 5 4 1 = 56 := 
by sorry

end NUMINAMATH_GPT_cousins_arrangement_l590_59077


namespace NUMINAMATH_GPT_correct_calculation_l590_59085

theorem correct_calculation (a b c d : ℤ) (h1 : a = -1) (h2 : b = -3) (h3 : c = 3) (h4 : d = -3) :
  a * b = c :=
by 
  rw [h1, h2]
  exact h3.symm

end NUMINAMATH_GPT_correct_calculation_l590_59085


namespace NUMINAMATH_GPT_peter_money_left_l590_59043

variable (soda_cost : ℝ) (money_brought : ℝ) (soda_ounces : ℝ)

theorem peter_money_left (h1 : soda_cost = 0.25) (h2 : money_brought = 2) (h3 : soda_ounces = 6) : 
    money_brought - soda_ounces * soda_cost = 0.50 := 
by 
  sorry

end NUMINAMATH_GPT_peter_money_left_l590_59043


namespace NUMINAMATH_GPT_parallel_lines_slope_l590_59026

theorem parallel_lines_slope (a : ℝ) (h : ∀ x y : ℝ, (x + a * y + 6 = 0) → ((a - 2) * x + 3 * y + 2 * a = 0)) : a = -1 :=
by
  sorry

end NUMINAMATH_GPT_parallel_lines_slope_l590_59026


namespace NUMINAMATH_GPT_apples_in_basket_l590_59041

-- Definitions based on conditions
def total_apples : ℕ := 138
def apples_per_box : ℕ := 18

-- Problem: prove the number of apples in the basket
theorem apples_in_basket : (total_apples % apples_per_box) = 12 :=
by 
  -- Skip the proof part by adding sorry
  sorry

end NUMINAMATH_GPT_apples_in_basket_l590_59041


namespace NUMINAMATH_GPT_markup_amount_l590_59062

def purchase_price : ℝ := 48
def overhead_percentage : ℝ := 0.35
def net_profit : ℝ := 18

def overhead : ℝ := purchase_price * overhead_percentage
def total_cost : ℝ := purchase_price + overhead
def selling_price : ℝ := total_cost + net_profit
def markup : ℝ := selling_price - purchase_price

theorem markup_amount : markup = 34.80 := by
  sorry

end NUMINAMATH_GPT_markup_amount_l590_59062


namespace NUMINAMATH_GPT_Bruno_wants_2_5_dozens_l590_59032

theorem Bruno_wants_2_5_dozens (total_pens : ℕ) (dozen_pens : ℕ) (h_total_pens : total_pens = 30) (h_dozen_pens : dozen_pens = 12) : (total_pens / dozen_pens : ℚ) = 2.5 :=
by 
  sorry

end NUMINAMATH_GPT_Bruno_wants_2_5_dozens_l590_59032


namespace NUMINAMATH_GPT_jacket_price_is_48_l590_59030

-- Definitions according to the conditions
def jacket_problem (P S D : ℝ) : Prop :=
  S = P + 0.40 * S ∧
  D = 0.80 * S ∧
  16 = D - P

-- Statement of the theorem
theorem jacket_price_is_48 :
  ∃ P S D, jacket_problem P S D ∧ P = 48 :=
by
  sorry

end NUMINAMATH_GPT_jacket_price_is_48_l590_59030


namespace NUMINAMATH_GPT_proof_problem_l590_59094

open Set Real

def M : Set ℝ := { x : ℝ | ∃ y : ℝ, y = log (1 - 2 / x) }
def N : Set ℝ := { x : ℝ | ∃ y : ℝ, y = sqrt (x - 1) }

theorem proof_problem : N ∩ (U \ M) = Icc 1 2 := by
  sorry

end NUMINAMATH_GPT_proof_problem_l590_59094


namespace NUMINAMATH_GPT_trigonometric_identity_l590_59067

theorem trigonometric_identity :
  (3 / (Real.sin (20 * Real.pi / 180))^2) - 
  (1 / (Real.cos (20 * Real.pi / 180))^2) + 
  64 * (Real.sin (20 * Real.pi / 180))^2 = 32 :=
by sorry

end NUMINAMATH_GPT_trigonometric_identity_l590_59067


namespace NUMINAMATH_GPT_bob_mother_twice_age_2040_l590_59015

theorem bob_mother_twice_age_2040 :
  ∀ (bob_age_2010 mother_age_2010 : ℕ), 
  bob_age_2010 = 10 ∧ mother_age_2010 = 50 →
  ∃ (x : ℕ), (mother_age_2010 + x = 2 * (bob_age_2010 + x)) ∧ (2010 + x = 2040) :=
by
  sorry

end NUMINAMATH_GPT_bob_mother_twice_age_2040_l590_59015


namespace NUMINAMATH_GPT_members_in_both_sets_l590_59002

def U : Nat := 193
def B : Nat := 41
def not_A_or_B : Nat := 59
def A : Nat := 116

theorem members_in_both_sets
  (h1 : 193 = U)
  (h2 : 41 = B)
  (h3 : 59 = not_A_or_B)
  (h4 : 116 = A) :
  (U - not_A_or_B) = A + B - 23 :=
by
  sorry

end NUMINAMATH_GPT_members_in_both_sets_l590_59002


namespace NUMINAMATH_GPT_min_value_of_a_l590_59044

theorem min_value_of_a 
  {f : ℕ → ℝ} 
  (h : ∀ x : ℕ, 0 < x → f x = (x^2 + a * x + 11) / (x + 1)) 
  (ineq : ∀ x : ℕ, 0 < x → f x ≥ 3) : a ≥ -8 / 3 :=
sorry

end NUMINAMATH_GPT_min_value_of_a_l590_59044


namespace NUMINAMATH_GPT_value_of_b_l590_59097

noncomputable def problem (a1 a2 a3 a4 a5 : ℤ) (b : ℤ) :=
  (a1 ≠ a2) ∧ (a1 ≠ a3) ∧ (a1 ≠ a4) ∧ (a1 ≠ a5) ∧
  (a2 ≠ a3) ∧ (a2 ≠ a4) ∧ (a2 ≠ a5) ∧
  (a3 ≠ a4) ∧ (a3 ≠ a5) ∧
  (a4 ≠ a5) ∧
  (a1 + a2 + a3 + a4 + a5 = 9) ∧
  ((b - a1) * (b - a2) * (b - a3) * (b - a4) * (b - a5) = 2009) ∧
  (∃ b : ℤ, b = 10)

theorem value_of_b (a1 a2 a3 a4 a5 : ℤ) (b : ℤ) :
  problem a1 a2 a3 a4 a5 b → b = 10 :=
  sorry

end NUMINAMATH_GPT_value_of_b_l590_59097


namespace NUMINAMATH_GPT_jill_spent_30_percent_on_food_l590_59028

variables (T F : ℝ)

theorem jill_spent_30_percent_on_food
  (h1 : 0.04 * T = 0.016 * T + 0.024 * T)
  (h2 : 0.40 + 0.30 + F = 1) :
  F = 0.30 :=
by 
  sorry

end NUMINAMATH_GPT_jill_spent_30_percent_on_food_l590_59028


namespace NUMINAMATH_GPT_find_coordinates_of_P_l590_59090

noncomputable def pointP_minimizes_dot_product : Prop :=
  let OA := (2, 2)
  let OB := (4, 1)
  let AP x := (x - 2, -2)
  let BP x := (x - 4, -1)
  let dot_product x := (AP x).1 * (BP x).1 + (AP x).2 * (BP x).2
  ∃ x, (dot_product x = (x - 3) ^ 2 + 1) ∧ (∀ y, dot_product y ≥ dot_product x) ∧ (x = 3)

theorem find_coordinates_of_P : pointP_minimizes_dot_product :=
  sorry

end NUMINAMATH_GPT_find_coordinates_of_P_l590_59090


namespace NUMINAMATH_GPT_geometric_sequence_ratio_l590_59070

theorem geometric_sequence_ratio (a1 : ℕ) (S : ℕ → ℕ) (r : ℤ) (h1 : r = -2) (h2 : ∀ n, S n = a1 * (1 - r ^ n) / (1 - r)) :
  S 4 / S 2 = 5 :=
by
  -- Placeholder for proof steps
  sorry

end NUMINAMATH_GPT_geometric_sequence_ratio_l590_59070


namespace NUMINAMATH_GPT_height_of_triangle_l590_59046

-- Define the dimensions of the rectangle
variable (l w : ℝ)

-- Assume the base of the triangle is equal to the length of the rectangle
-- We need to prove that the height of the triangle h = 2w

theorem height_of_triangle (h : ℝ) (hl_eq_length : l > 0) (hw_eq_width : w > 0) :
  (l * w) = (1 / 2) * l * h → h = 2 * w :=
by
  sorry

end NUMINAMATH_GPT_height_of_triangle_l590_59046


namespace NUMINAMATH_GPT_part1_part2_l590_59050

variable {R : Type} [LinearOrderedField R]

def f (x : R) : R := abs (x - 2) + 2
def g (m : R) (x : R) : R := m * abs x

theorem part1 (x : R) : f x > 5 ↔ x < -1 ∨ x > 5 := by
  sorry

theorem part2 (m : R) : (∀ x : R, f x ≥ g m x) → m ∈ Set.Iic (1 : R) := by
  sorry

end NUMINAMATH_GPT_part1_part2_l590_59050


namespace NUMINAMATH_GPT_f_strictly_increasing_on_l590_59056

-- Define the function
def f (x : ℝ) : ℝ := x^2 * (2 - x)

-- Define the derivative of the function
def f' (x : ℝ) : ℝ := -3 * x^2 + 4 * x

-- Define the property that the function is strictly increasing on an interval
def strictly_increasing_on (a b : ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

-- State the theorem
theorem f_strictly_increasing_on : strictly_increasing_on 0 (4/3) f :=
sorry

end NUMINAMATH_GPT_f_strictly_increasing_on_l590_59056


namespace NUMINAMATH_GPT_div_power_n_minus_one_l590_59045

theorem div_power_n_minus_one (n : ℕ) (hn : n > 0) (h : n ∣ (2^n - 1)) : n = 1 := by
  sorry

end NUMINAMATH_GPT_div_power_n_minus_one_l590_59045


namespace NUMINAMATH_GPT_sum_of_intersections_l590_59059

theorem sum_of_intersections :
  (∃ x1 y1 x2 y2 x3 y3 x4 y4, 
    y1 = (x1 - 1)^2 ∧ y2 = (x2 - 1)^2 ∧ y3 = (x3 - 1)^2 ∧ y4 = (x4 - 1)^2 ∧
    x1 - 2 = (y1 + 1)^2 ∧ x2 - 2 = (y2 + 1)^2 ∧ x3 - 2 = (y3 + 1)^2 ∧ x4 - 2 = (y4 + 1)^2 ∧
    (x1 + x2 + x3 + x4 + y1 + y2 + y3 + y4) = 2) :=
sorry

end NUMINAMATH_GPT_sum_of_intersections_l590_59059


namespace NUMINAMATH_GPT_roses_cut_l590_59064

def initial_roses : ℕ := 6
def new_roses : ℕ := 16

theorem roses_cut : new_roses - initial_roses = 10 := by
  sorry

end NUMINAMATH_GPT_roses_cut_l590_59064


namespace NUMINAMATH_GPT_percentage_of_loss_is_10_l590_59072

-- Definitions based on conditions
def cost_price : ℝ := 1800
def selling_price : ℝ := 1620
def loss : ℝ := cost_price - selling_price

-- The goal: prove the percentage of loss equals 10%
theorem percentage_of_loss_is_10 :
  (loss / cost_price) * 100 = 10 := by
  sorry

end NUMINAMATH_GPT_percentage_of_loss_is_10_l590_59072


namespace NUMINAMATH_GPT_max_triangle_side_l590_59074

-- Definitions of conditions
def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def has_perimeter (a b c : ℕ) (p : ℕ) : Prop :=
  a + b + c = p

def different_integers (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c

-- The main theorem to prove
theorem max_triangle_side (a b c : ℕ) (h_triangle : is_triangle a b c)
                         (h_perimeter : has_perimeter a b c 24)
                         (h_diff : different_integers a b c) :
  c ≤ 11 :=
sorry

end NUMINAMATH_GPT_max_triangle_side_l590_59074


namespace NUMINAMATH_GPT_trapezoid_dot_product_ad_bc_l590_59006

-- Define the trapezoid and its properties
variables (A B C D O : Type) (AB CD AO BO : ℝ)
variables (AD BC : ℝ)

-- Conditions from the problem
axiom AB_length : AB = 41
axiom CD_length : CD = 24
axiom diagonals_perpendicular : ∀ (v₁ v₂ : ℝ), (v₁ * v₂ = 0)

-- Using these conditions, prove that the dot product of the vectors AD and BC is 984
theorem trapezoid_dot_product_ad_bc : AD * BC = 984 :=
  sorry

end NUMINAMATH_GPT_trapezoid_dot_product_ad_bc_l590_59006


namespace NUMINAMATH_GPT_matrix_inverse_eq_scaling_l590_59092

variable (d k : ℚ)

def B : Matrix (Fin 3) (Fin 3) ℚ := ![
  ![1, 2, 3],
  ![4, 5, d],
  ![6, 7, 8]
]

theorem matrix_inverse_eq_scaling :
  (B d)⁻¹ = k • (B d) →
  d = 13/9 ∧ k = -329/52 :=
by
  sorry

end NUMINAMATH_GPT_matrix_inverse_eq_scaling_l590_59092


namespace NUMINAMATH_GPT_value_of_fraction_l590_59079

theorem value_of_fraction (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0)
  (h : (b / c = 2005) ∧ (c / b = 2005)) : (b + c) / (a + b) = 2005 :=
by
  sorry

end NUMINAMATH_GPT_value_of_fraction_l590_59079


namespace NUMINAMATH_GPT_pizza_slices_l590_59078

theorem pizza_slices (total_slices pepperoni_slices mushroom_slices : ℕ) 
  (h_total : total_slices = 24)
  (h_pepperoni : pepperoni_slices = 15)
  (h_mushrooms : mushroom_slices = 16)
  (h_at_least_one : total_slices = pepperoni_slices + mushroom_slices - both_slices)
  : both_slices = 7 :=
by
  have h1 : total_slices = 24 := h_total
  have h2 : pepperoni_slices = 15 := h_pepperoni
  have h3 : mushroom_slices = 16 := h_mushrooms
  have h4 : total_slices = 24 := by sorry
  sorry

end NUMINAMATH_GPT_pizza_slices_l590_59078


namespace NUMINAMATH_GPT_nina_homework_total_l590_59036

-- Definitions based on conditions
def ruby_math_homework : Nat := 6
def ruby_reading_homework : Nat := 2
def nina_math_homework : Nat := 4 * ruby_math_homework
def nina_reading_homework : Nat := 8 * ruby_reading_homework
def nina_total_homework : Nat := nina_math_homework + nina_reading_homework

-- The theorem to prove
theorem nina_homework_total : nina_total_homework = 40 := by
  sorry

end NUMINAMATH_GPT_nina_homework_total_l590_59036


namespace NUMINAMATH_GPT_maximum_distance_l590_59066

noncomputable def point_distance (x1 y1 x2 y2 : ℝ) : ℝ := 
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

def square_side_length := 2

def distance_condition (u v w : ℝ) : Prop := 
  u^2 + v^2 = 2 * w^2

theorem maximum_distance 
  (x y : ℝ) 
  (h1 : point_distance x y 0 0 = u) 
  (h2 : point_distance x y 2 0 = v) 
  (h3 : point_distance x y 2 2 = w)
  (h4 : distance_condition u v w) :
  ∃ (d : ℝ), d = point_distance x y 0 2 ∧ d = 2 * Real.sqrt 5 := sorry

end NUMINAMATH_GPT_maximum_distance_l590_59066


namespace NUMINAMATH_GPT_part1_part2_l590_59014

noncomputable def f (x : ℝ) : ℝ := |2 * x - 1| + |2 * x - 3|

theorem part1 (x : ℝ) (hx : f x ≤ 5) : x ∈ Set.Icc (-1/4 : ℝ) (9/4 : ℝ) := sorry

noncomputable def h (x a : ℝ) : ℝ := Real.log (f x + a)

theorem part2 (ha : ∀ x : ℝ, f x + a > 0) : a ∈ Set.Ioi (-2 : ℝ) := sorry

end NUMINAMATH_GPT_part1_part2_l590_59014


namespace NUMINAMATH_GPT_work_in_one_day_l590_59022

theorem work_in_one_day (A_days B_days : ℕ) (hA : A_days = 18) (hB : B_days = A_days / 2) :
  (1 / A_days + 1 / B_days) = 1 / 6 := 
by
  sorry

end NUMINAMATH_GPT_work_in_one_day_l590_59022


namespace NUMINAMATH_GPT_pills_first_day_l590_59061

theorem pills_first_day (P : ℕ) 
  (h1 : P + (P + 2) + (P + 4) + (P + 6) + (P + 8) + (P + 10) + (P + 12) = 49) : 
  P = 1 :=
by sorry

end NUMINAMATH_GPT_pills_first_day_l590_59061


namespace NUMINAMATH_GPT_sqrt_three_irrational_l590_59052

-- Define what it means for a number to be rational
def is_rational (x : ℝ) : Prop := ∃ (p q : ℤ), q ≠ 0 ∧ x = p / q

-- Define what it means for a number to be irrational
def is_irrational (x : ℝ) : Prop := ¬ is_rational x

-- State that sqrt(3) is irrational
theorem sqrt_three_irrational : is_irrational (Real.sqrt 3) :=
sorry

end NUMINAMATH_GPT_sqrt_three_irrational_l590_59052


namespace NUMINAMATH_GPT_middle_marble_radius_l590_59068

theorem middle_marble_radius (r_1 r_5 : ℝ) (h1 : r_1 = 8) (h5 : r_5 = 18) : 
  ∃ r_3 : ℝ, r_3 = 12 :=
by
  let r_3 := Real.sqrt (r_1 * r_5)
  have h : r_3 = 12 := sorry
  exact ⟨r_3, h⟩

end NUMINAMATH_GPT_middle_marble_radius_l590_59068


namespace NUMINAMATH_GPT_shirt_tie_combinations_l590_59088

noncomputable def shirts : ℕ := 8
noncomputable def ties : ℕ := 7
noncomputable def forbidden_combinations : ℕ := 2

theorem shirt_tie_combinations :
  shirts * ties - forbidden_combinations = 54 := by
  sorry

end NUMINAMATH_GPT_shirt_tie_combinations_l590_59088


namespace NUMINAMATH_GPT_problem_part1_problem_part2_l590_59013

-- Define the set A and the property it satisfies
variable (A : Set ℝ)
variable (H : ∀ a ∈ A, (1 + a) / (1 - a) ∈ A)

-- Suppose 2 is in A
theorem problem_part1 (h : 2 ∈ A) : A = {2, -3, -1 / 2, 1 / 3} :=
sorry

-- Prove the conjecture based on the elements of A found in part 1
theorem problem_part2 (h : 2 ∈ A) (hA : A = {2, -3, -1 / 2, 1 / 3}) :
  ¬ (0 ∈ A ∨ 1 ∈ A ∨ -1 ∈ A) ∧
  (2 * (-1 / 2) = -1 ∧ -3 * (1 / 3) = -1) :=
sorry

end NUMINAMATH_GPT_problem_part1_problem_part2_l590_59013


namespace NUMINAMATH_GPT_sufficient_condition_l590_59073

theorem sufficient_condition (a b c : ℤ) : (a = c + 1) → (b = a - 1) → a * (a - b) + b * (b - c) + c * (c - a) = 2 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_sufficient_condition_l590_59073


namespace NUMINAMATH_GPT_paco_salty_cookies_left_l590_59009

theorem paco_salty_cookies_left (initial_salty : ℕ) (eaten_salty : ℕ) : initial_salty = 26 ∧ eaten_salty = 9 → initial_salty - eaten_salty = 17 :=
by
  intro h
  cases h
  sorry


end NUMINAMATH_GPT_paco_salty_cookies_left_l590_59009


namespace NUMINAMATH_GPT_tyler_age_l590_59008

theorem tyler_age (T C : ℕ) (h1 : T = 3 * C + 1) (h2 : T + C = 21) : T = 16 :=
by
  sorry

end NUMINAMATH_GPT_tyler_age_l590_59008


namespace NUMINAMATH_GPT_find_digit_A_l590_59069

theorem find_digit_A (A : ℕ) (h1 : 0 ≤ A ∧ A ≤ 9) (h2 : (2 + A + 3 + A) % 9 = 0) : A = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_digit_A_l590_59069


namespace NUMINAMATH_GPT_vertex_angle_of_obtuse_isosceles_triangle_l590_59037

noncomputable def isosceles_obtuse_triangle (a b h : ℝ) (φ : ℝ) : Prop :=
  a^2 = 2 * b * h ∧
  b = 2 * a * Real.cos ((180 - φ) / 2) ∧
  h = a * Real.sin ((180 - φ) / 2) ∧
  90 < φ ∧ φ < 180

theorem vertex_angle_of_obtuse_isosceles_triangle (a b h : ℝ) (φ : ℝ) :
  isosceles_obtuse_triangle a b h φ → φ = 150 :=
by
  sorry

end NUMINAMATH_GPT_vertex_angle_of_obtuse_isosceles_triangle_l590_59037


namespace NUMINAMATH_GPT_suff_condition_not_necc_condition_l590_59055

variable (x : ℝ)

def A : Prop := 0 < x ∧ x < 5
def B : Prop := |x - 2| < 3

theorem suff_condition : A x → B x := by
  sorry

theorem not_necc_condition : B x → ¬ A x := by
  sorry

end NUMINAMATH_GPT_suff_condition_not_necc_condition_l590_59055


namespace NUMINAMATH_GPT_percentage_of_profits_to_revenues_l590_59027

theorem percentage_of_profits_to_revenues (R P : ℝ) (h1 : 0.7 * R = R - 0.3 * R) (h2 : 0.105 * R = 0.15 * (0.7 * R)) (h3 : 0.105 * R = 1.0499999999999999 * P) :
  (P / R) * 100 = 10 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_profits_to_revenues_l590_59027


namespace NUMINAMATH_GPT_simplify_expression_l590_59089

-- Define the given conditions
def pow_2_5 : ℕ := 32
def pow_4_4 : ℕ := 256
def pow_2_2 : ℕ := 4
def pow_neg_2_3 : ℤ := -8

-- State the theorem to prove
theorem simplify_expression : 
  (pow_2_5 + pow_4_4) * (pow_2_2 - pow_neg_2_3)^8 = 123876479488 := 
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l590_59089


namespace NUMINAMATH_GPT_square_feet_per_acre_l590_59021

theorem square_feet_per_acre 
  (pay_per_acre_per_month : ℕ) 
  (total_pay_per_month : ℕ) 
  (length : ℕ) 
  (width : ℕ) 
  (total_acres : ℕ) 
  (H1 : pay_per_acre_per_month = 30) 
  (H2 : total_pay_per_month = 300) 
  (H3 : length = 360) 
  (H4 : width = 1210) 
  (H5 : total_acres = 10) : 
  (length * width) / total_acres = 43560 :=
by 
  sorry

end NUMINAMATH_GPT_square_feet_per_acre_l590_59021


namespace NUMINAMATH_GPT_find_plane_equation_l590_59054

def point := ℝ × ℝ × ℝ

def plane_equation (A B C D : ℝ) (x y z : ℝ) : Prop :=
  A * x + B * y + C * z + D = 0

def points := (0, 3, -1) :: (4, 7, 1) :: (2, 5, 0) :: []

def correct_plane_equation : Prop :=
  ∃ A B C D : ℝ, plane_equation A B C D = fun x y z => A * x + B * y + C * z + D = 0 ∧ 
  (A, B, C, D) = (0, 1, -2, -5) ∧ ∀ x y z, (x, y, z) ∈ points → plane_equation A B C D x y z

theorem find_plane_equation : correct_plane_equation :=
sorry

end NUMINAMATH_GPT_find_plane_equation_l590_59054


namespace NUMINAMATH_GPT_find_a_of_binomial_square_l590_59011

theorem find_a_of_binomial_square (a : ℚ) :
  (∃ b : ℚ, (3 * (x : ℚ) + b)^2 = 9 * x^2 + 21 * x + a) ↔ a = 49 / 4 :=
by
  sorry

end NUMINAMATH_GPT_find_a_of_binomial_square_l590_59011


namespace NUMINAMATH_GPT_number_of_multiples_of_15_l590_59075

theorem number_of_multiples_of_15 (a b : ℕ) (h₁ : a = 15) (h₂ : b = 305) : 
  ∃ n : ℕ, n = 20 ∧ ∀ k, (1 ≤ k ∧ k ≤ n) → (15 * k) ≥ a ∧ (15 * k) ≤ b := by
  sorry

end NUMINAMATH_GPT_number_of_multiples_of_15_l590_59075


namespace NUMINAMATH_GPT_max_a_l590_59035

-- Define the conditions
def line_equation (m : ℚ) (x : ℤ) : ℚ := m * x + 3

def no_lattice_points (m : ℚ) : Prop :=
  ∀ (x : ℤ), 1 ≤ x ∧ x ≤ 50 → ¬ ∃ (y : ℤ), line_equation m x = y

def m_range (m a : ℚ) : Prop := (2 : ℚ) / 5 < m ∧ m < a

-- Define the problem statement
theorem max_a (a : ℚ) : (a = 22 / 51) ↔ (∃ m, no_lattice_points m ∧ m_range m a) :=
by 
  sorry

end NUMINAMATH_GPT_max_a_l590_59035


namespace NUMINAMATH_GPT_steak_amount_per_member_l590_59040

theorem steak_amount_per_member : 
  ∀ (num_members steaks_needed ounces_per_steak total_ounces each_amount : ℕ),
    num_members = 5 →
    steaks_needed = 4 →
    ounces_per_steak = 20 →
    total_ounces = steaks_needed * ounces_per_steak →
    each_amount = total_ounces / num_members →
    each_amount = 16 :=
by
  intros num_members steaks_needed ounces_per_steak total_ounces each_amount
  intro h_members h_steaks h_ounces_per_steak h_total_ounces h_each_amount
  sorry

end NUMINAMATH_GPT_steak_amount_per_member_l590_59040


namespace NUMINAMATH_GPT_range_of_independent_variable_l590_59003

theorem range_of_independent_variable (x : ℝ) : x + 3 ≥ 0 ↔ x ≥ -3 := by
  sorry

end NUMINAMATH_GPT_range_of_independent_variable_l590_59003


namespace NUMINAMATH_GPT_positive_integer_solution_of_inequality_l590_59005

theorem positive_integer_solution_of_inequality (x : ℕ) (h : 0 < x) : (3 * x - 1) / 2 + 1 ≥ 2 * x → x = 1 :=
by
  intros
  sorry

end NUMINAMATH_GPT_positive_integer_solution_of_inequality_l590_59005


namespace NUMINAMATH_GPT_spider_crawl_distance_l590_59001

theorem spider_crawl_distance :
  let a := -3
  let b := -8
  let c := 4
  let d := 7
  abs (b - a) + abs (c - b) + abs (d - c) = 20 :=
by
  let a := -3
  let b := -8
  let c := 4
  let d := 7
  sorry

end NUMINAMATH_GPT_spider_crawl_distance_l590_59001


namespace NUMINAMATH_GPT_find_m_when_circle_tangent_to_line_l590_59098

theorem find_m_when_circle_tangent_to_line 
    (m : ℝ)
    (circle_eq : (x y : ℝ) → (x - 1)^2 + (y - 1)^2 = 4 * m)
    (line_eq : (x y : ℝ) → x + y = 2 * m) :
    (m = 2 + Real.sqrt 3) ∨ (m = 2 - Real.sqrt 3) :=
sorry

end NUMINAMATH_GPT_find_m_when_circle_tangent_to_line_l590_59098


namespace NUMINAMATH_GPT_nominal_rate_of_interest_l590_59048

theorem nominal_rate_of_interest
  (EAR : ℝ)
  (n : ℕ)
  (h_EAR : EAR = 0.0609)
  (h_n : n = 2) :
  ∃ i : ℝ, (1 + i / n)^n - 1 = EAR ∧ i = 0.059 := 
by 
  sorry

end NUMINAMATH_GPT_nominal_rate_of_interest_l590_59048


namespace NUMINAMATH_GPT_find_a4_and_s5_l590_59083

def geometric_sequence (a : ℕ → ℚ) (q : ℚ) : Prop :=
  ∀ n, a (n + 1) = a n * q

variable (a : ℕ → ℚ) (q : ℚ)

axiom condition_1 : a 1 + a 3 = 10
axiom condition_2 : a 4 + a 6 = 1 / 4

theorem find_a4_and_s5 (h_geom : geometric_sequence a q) :
  a 4 = 1 ∧ (a 1 * (1 - q^5) / (1 - q)) = 31 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_a4_and_s5_l590_59083


namespace NUMINAMATH_GPT_range_of_k_l590_59081

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := x^(-k^2 + k + 2)

theorem range_of_k (k : ℝ) : (∃ k, (f 2 k < f 3 k)) ↔ (-1 < k) ∧ (k < 2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_k_l590_59081


namespace NUMINAMATH_GPT_simplify_expression_l590_59007

theorem simplify_expression (y : ℝ) : 
  4 * y + 9 * y ^ 2 + 8 - (3 - 4 * y - 9 * y ^ 2) = 18 * y ^ 2 + 8 * y + 5 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l590_59007


namespace NUMINAMATH_GPT_numberOfChromiumAtoms_l590_59023

noncomputable def molecularWeightOfCompound : ℕ := 296
noncomputable def atomicWeightOfPotassium : ℝ := 39.1
noncomputable def atomicWeightOfOxygen : ℝ := 16.0
noncomputable def atomicWeightOfChromium : ℝ := 52.0

def numberOfPotassiumAtoms : ℕ := 2
def numberOfOxygenAtoms : ℕ := 7

theorem numberOfChromiumAtoms
    (mw : ℕ := molecularWeightOfCompound)
    (awK : ℝ := atomicWeightOfPotassium)
    (awO : ℝ := atomicWeightOfOxygen)
    (awCr : ℝ := atomicWeightOfChromium)
    (numK : ℕ := numberOfPotassiumAtoms)
    (numO : ℕ := numberOfOxygenAtoms) :
  numK * awK + numO * awO + (mw - (numK * awK + numO * awO)) / awCr = 2 := 
by
  sorry

end NUMINAMATH_GPT_numberOfChromiumAtoms_l590_59023


namespace NUMINAMATH_GPT_relation_P_Q_l590_59060

def P : Set ℝ := {x | x ≠ 0}
def Q : Set ℝ := {x | x > 0}
def complement_P : Set ℝ := {0}

theorem relation_P_Q : Q ∩ complement_P = ∅ := 
by sorry

end NUMINAMATH_GPT_relation_P_Q_l590_59060


namespace NUMINAMATH_GPT_metallic_sheet_length_l590_59049

theorem metallic_sheet_length (w : ℝ) (s : ℝ) (v : ℝ) (L : ℝ) 
  (h_w : w = 38) 
  (h_s : s = 8) 
  (h_v : v = 5632) 
  (h_volume : (L - 2 * s) * (w - 2 * s) * s = v) : 
  L = 48 :=
by
  -- To complete the proof, follow the mathematical steps:
  -- (L - 2 * s) * (w - 2 * s) * s = v
  -- (L - 2 * 8) * (38 - 2 * 8) * 8 = 5632
  -- Simplify and solve for L
  sorry

end NUMINAMATH_GPT_metallic_sheet_length_l590_59049


namespace NUMINAMATH_GPT_cube_sum_from_square_l590_59058

noncomputable def a_plus_inv_a_squared_eq_5 (a : ℝ) : Prop :=
  (a + 1/a) ^ 2 = 5

theorem cube_sum_from_square (a : ℝ) (h : a_plus_inv_a_squared_eq_5 a) :
  a^3 + (1/a)^3 = 2 * Real.sqrt 5 ∨ a^3 + (1/a)^3 = -2 * Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_cube_sum_from_square_l590_59058


namespace NUMINAMATH_GPT_ratio_of_coefficients_l590_59095

theorem ratio_of_coefficients (x y c d : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hd : d ≠ 0)
  (H1 : 8 * x - 6 * y = c) (H2 : 12 * y - 18 * x = d) :
  c / d = -4 / 9 := 
by {
  sorry
}

end NUMINAMATH_GPT_ratio_of_coefficients_l590_59095


namespace NUMINAMATH_GPT_largest_integer_remainder_condition_l590_59071

theorem largest_integer_remainder_condition (number : ℤ) (h1 : number < 100) (h2 : number % 7 = 4) :
  number = 95 := sorry

end NUMINAMATH_GPT_largest_integer_remainder_condition_l590_59071


namespace NUMINAMATH_GPT_parallel_lines_slope_condition_l590_59024

theorem parallel_lines_slope_condition (m : ℝ) :
  (∀ x y : ℝ, 2 * x + (m + 1) * y + 4 = 0 → mx + 3 * y - 2 = 0) →
  (m = 2 ∨ m = -3) :=
by
  sorry

end NUMINAMATH_GPT_parallel_lines_slope_condition_l590_59024


namespace NUMINAMATH_GPT_find_k_for_quadratic_has_one_real_root_l590_59051

theorem find_k_for_quadratic_has_one_real_root (k : ℝ) : 
  (∃ x : ℝ, (3 * x - 4) * (x + 6) = -53 + k * x) ↔ (k = 14 + 2 * Real.sqrt 87 ∨ k = 14 - 2 * Real.sqrt 87) :=
sorry

end NUMINAMATH_GPT_find_k_for_quadratic_has_one_real_root_l590_59051


namespace NUMINAMATH_GPT_extremum_at_x_1_max_integer_k_l590_59031

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (x - 1) * Real.log x - (a + 1) * x

theorem extremum_at_x_1 (a : ℝ) : (∀ x : ℝ, 0 < x → ((Real.log x - 1 / x - a = 0) ↔ x = 1))
  → a = -1 ∧
  (∀ x : ℝ, 0 < x → (Real.log x - 1 / x + 1) < 0 → f x (-1) < f 1 (-1) ∧
  (Real.log x - 1 / x + 1) > 0 → f 1 (-1) < f x (-1)) :=
sorry

theorem max_integer_k (k : ℤ) :
  (∀ x : ℝ, 0 < x → (f x 1 > k))
  → k ≤ -4 :=
sorry

end NUMINAMATH_GPT_extremum_at_x_1_max_integer_k_l590_59031


namespace NUMINAMATH_GPT_smallest_D_for_inequality_l590_59096

theorem smallest_D_for_inequality :
  ∃ D : ℝ, (∀ x y z : ℝ, 2 * x^2 + 3 * y^2 + z^2 + 3 ≥ D * (x + y + z)) ∧ 
           D = -Real.sqrt (72 / 11) :=
by
  sorry

end NUMINAMATH_GPT_smallest_D_for_inequality_l590_59096


namespace NUMINAMATH_GPT_positive_partial_sum_existence_l590_59019

variable {n : ℕ}
variable {a : Fin n → ℝ}

theorem positive_partial_sum_existence (h : (Finset.univ.sum a) > 0) :
  ∃ i : Fin n, ∀ j : Fin n, i ≤ j → (Finset.Icc i j).sum a > 0 := by
  sorry

end NUMINAMATH_GPT_positive_partial_sum_existence_l590_59019


namespace NUMINAMATH_GPT_arrange_students_l590_59093

theorem arrange_students 
  (students : Fin 6 → Type) 
  (A B : Type) 
  (h1 : ∃ i j, students i = A ∧ students j = B ∧ (i = j + 1 ∨ j = i + 1)) : 
  (∃ (n : ℕ), n = 240) := 
sorry

end NUMINAMATH_GPT_arrange_students_l590_59093


namespace NUMINAMATH_GPT_sum_k_over_3_pow_k_eq_three_fourths_l590_59084

noncomputable def sum_k_over_3_pow_k : ℝ :=
  ∑' k : ℕ, (k + 1) / 3 ^ (k + 1)

theorem sum_k_over_3_pow_k_eq_three_fourths :
  sum_k_over_3_pow_k = 3 / 4 := sorry

end NUMINAMATH_GPT_sum_k_over_3_pow_k_eq_three_fourths_l590_59084


namespace NUMINAMATH_GPT_find_g7_l590_59029

namespace ProofProblem

variable (g : ℝ → ℝ)
variable (h1 : ∀ x y : ℝ, g (x + y) = g x + g y)
variable (h2 : g 6 = 8)

theorem find_g7 : g 7 = 28 / 3 := by
  sorry

end ProofProblem

end NUMINAMATH_GPT_find_g7_l590_59029


namespace NUMINAMATH_GPT_Polly_lunch_time_l590_59082

-- Define the conditions
def breakfast_time_per_day := 20
def total_days_in_week := 7
def dinner_time_4_days := 10
def remaining_days_in_week := 3
def remaining_dinner_time_per_day := 30
def total_cooking_time := 305

-- Define the total time Polly spends cooking breakfast in a week
def total_breakfast_time := breakfast_time_per_day * total_days_in_week

-- Define the total time Polly spends cooking dinner in a week
def total_dinner_time := (dinner_time_4_days * 4) + (remaining_dinner_time_per_day * remaining_days_in_week)

-- Define the time Polly spends cooking lunch in a week
def lunch_time := total_cooking_time - (total_breakfast_time + total_dinner_time)

-- The theorem to prove Polly's lunch time
theorem Polly_lunch_time : lunch_time = 35 :=
by
  sorry

end NUMINAMATH_GPT_Polly_lunch_time_l590_59082


namespace NUMINAMATH_GPT_Jovana_shells_l590_59063

theorem Jovana_shells (initial_shells : ℕ) (added_shells : ℕ) (total_shells : ℕ) :
  initial_shells = 5 → added_shells = 12 → total_shells = initial_shells + added_shells → total_shells = 17 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_Jovana_shells_l590_59063


namespace NUMINAMATH_GPT_square_of_cube_of_third_smallest_prime_l590_59039

-- Definition of the third smallest prime number
def third_smallest_prime : Nat := 5

-- Definition of the cube of a number
def cube (n : Nat) : Nat := n ^ 3

-- Definition of the square of a number
def square (n : Nat) : Nat := n ^ 2

-- Theorem stating that the square of the cube of the third smallest prime number is 15625
theorem square_of_cube_of_third_smallest_prime : 
  square (cube third_smallest_prime) = 15625 := by 
  sorry

end NUMINAMATH_GPT_square_of_cube_of_third_smallest_prime_l590_59039


namespace NUMINAMATH_GPT_smallest_possible_value_l590_59012

theorem smallest_possible_value (n : ℕ) (h1 : ∀ m, (Nat.lcm 60 m / Nat.gcd 60 m = 24) → m = n) (h2 : ∀ m, (m % 5 = 0) → m = n) : n = 160 :=
sorry

end NUMINAMATH_GPT_smallest_possible_value_l590_59012


namespace NUMINAMATH_GPT_proof_A_minus_2B_eq_11_l590_59016

theorem proof_A_minus_2B_eq_11 
  (a b : ℤ)
  (hA : ∀ a b, A = 3*b^2 - 2*a^2)
  (hB : ∀ a b, B = ab - 2*b^2 - a^2) 
  (ha : a = 2) 
  (hb : b = -1) : 
  (A - 2*B = 11) :=
by
  sorry

end NUMINAMATH_GPT_proof_A_minus_2B_eq_11_l590_59016


namespace NUMINAMATH_GPT_percentage_failed_in_Hindi_l590_59010

-- Define the percentage of students failed in English
def percentage_failed_in_English : ℝ := 56

-- Define the percentage of students failed in both Hindi and English
def percentage_failed_in_both : ℝ := 12

-- Define the percentage of students passed in both subjects
def percentage_passed_in_both : ℝ := 24

-- Define the total percentage of students
def percentage_total : ℝ := 100

-- Define what we need to prove
theorem percentage_failed_in_Hindi:
  ∃ (H : ℝ), H + percentage_failed_in_English - percentage_failed_in_both + percentage_passed_in_both = percentage_total ∧ H = 32 :=
  by 
    sorry

end NUMINAMATH_GPT_percentage_failed_in_Hindi_l590_59010


namespace NUMINAMATH_GPT_vec_c_is_linear_comb_of_a_b_l590_59076

structure Vec2 :=
  (x : ℝ)
  (y : ℝ)

def a := Vec2.mk 1 2
def b := Vec2.mk (-2) 3
def c := Vec2.mk 4 1

theorem vec_c_is_linear_comb_of_a_b : c = Vec2.mk (2 * a.x - b.x) (2 * a.y - b.y) :=
  by
    sorry

end NUMINAMATH_GPT_vec_c_is_linear_comb_of_a_b_l590_59076


namespace NUMINAMATH_GPT_three_distinct_solutions_no_solution_for_2009_l590_59033

-- Problem 1: Show that the equation has at least three distinct solutions if it has one
theorem three_distinct_solutions (n : ℕ) (hn : n > 0) :
  (∃ x y : ℤ, x^3 - 3*x*y^2 + y^3 = n) →
  (∃ (x1 y1 x2 y2 x3 y3 : ℤ), 
    x1^3 - 3*x1*y1^2 + y1^3 = n ∧ 
    x2^3 - 3*x2*y2^2 + y2^3 = n ∧ 
    x3^3 - 3*x3*y3^2 + y3^3 = n ∧ 
    (x1, y1) ≠ (x2, y2) ∧ 
    (x1, y1) ≠ (x3, y3) ∧ 
    (x2, y2) ≠ (x3, y3)) :=
sorry

-- Problem 2: Show that the equation has no solutions when n = 2009
theorem no_solution_for_2009 :
  ¬ ∃ x y : ℤ, x^3 - 3*x*y^2 + y^3 = 2009 :=
sorry

end NUMINAMATH_GPT_three_distinct_solutions_no_solution_for_2009_l590_59033


namespace NUMINAMATH_GPT_dog_adult_weight_l590_59099

theorem dog_adult_weight 
  (w7 : ℕ) (w7_eq : w7 = 6)
  (w9 : ℕ) (w9_eq : w9 = 2 * w7)
  (w3m : ℕ) (w3m_eq : w3m = 2 * w9)
  (w5m : ℕ) (w5m_eq : w5m = 2 * w3m)
  (w1y : ℕ) (w1y_eq : w1y = w5m + 30) :
  w1y = 78 := by
  -- Proof is not required, so we leave it with sorry.
  sorry

end NUMINAMATH_GPT_dog_adult_weight_l590_59099


namespace NUMINAMATH_GPT_total_seats_l590_59091

theorem total_seats (KA_pos : ℕ) (SL_pos : ℕ) (h1 : KA_pos = 10) (h2 : SL_pos = 29) (h3 : SL_pos = KA_pos + (KA_pos * 2 - 1) / 2):
  let total_positions := 2 * (SL_pos - KA_pos - 1) + 2
  total_positions = 38 :=
by
  sorry

end NUMINAMATH_GPT_total_seats_l590_59091


namespace NUMINAMATH_GPT_eval_gg3_l590_59057

def g (x : ℕ) : ℕ := 3 * x^2 + 3 * x - 2

theorem eval_gg3 : g (g 3) = 3568 :=
by 
  sorry

end NUMINAMATH_GPT_eval_gg3_l590_59057


namespace NUMINAMATH_GPT_xiaoying_final_score_l590_59000

def speech_competition_score (score_content score_expression score_demeanor : ℕ) 
                             (weight_content weight_expression weight_demeanor : ℝ) : ℝ :=
  score_content * weight_content + score_expression * weight_expression + score_demeanor * weight_demeanor

theorem xiaoying_final_score :
  speech_competition_score 86 90 80 0.5 0.4 0.1 = 87 :=
by 
  sorry

end NUMINAMATH_GPT_xiaoying_final_score_l590_59000


namespace NUMINAMATH_GPT_distance_points_3_12_and_10_0_l590_59017

theorem distance_points_3_12_and_10_0 : 
  Real.sqrt ((10 - 3)^2 + (0 - 12)^2) = Real.sqrt 193 := 
by
  sorry

end NUMINAMATH_GPT_distance_points_3_12_and_10_0_l590_59017


namespace NUMINAMATH_GPT_mia_days_not_worked_l590_59042

theorem mia_days_not_worked :
  ∃ (y : ℤ), (∃ (x : ℤ), 
  x + y = 30 ∧ 80 * x - 40 * y = 1600) ∧ y = 20 :=
by
  sorry

end NUMINAMATH_GPT_mia_days_not_worked_l590_59042


namespace NUMINAMATH_GPT_election_percentage_l590_59053

-- Define the total number of votes (V), winner's votes, and the vote difference
def total_votes (V : ℕ) : Prop := V = 1944 + (1944 - 288)

-- Define the percentage calculation from the problem
def percentage_of_votes (votes_received total_votes : ℕ) : ℕ := (votes_received * 100) / total_votes

-- State the core theorem to prove the winner received 54 percent of the total votes
theorem election_percentage (V : ℕ) (h : total_votes V) : percentage_of_votes 1944 V = 54 := by
  sorry

end NUMINAMATH_GPT_election_percentage_l590_59053


namespace NUMINAMATH_GPT_raj_snow_removal_volume_l590_59018

theorem raj_snow_removal_volume :
  let length := 30
  let width := 4
  let depth_layer1 := 0.5
  let depth_layer2 := 0.3
  let volume_layer1 := length * width * depth_layer1
  let volume_layer2 := length * width * depth_layer2
  let total_volume := volume_layer1 + volume_layer2
  total_volume = 96 := by
sorry

end NUMINAMATH_GPT_raj_snow_removal_volume_l590_59018


namespace NUMINAMATH_GPT_product_of_primes_l590_59087

theorem product_of_primes :
  (7 * 97 * 89) = 60431 :=
by
  sorry

end NUMINAMATH_GPT_product_of_primes_l590_59087


namespace NUMINAMATH_GPT_day_after_60_days_is_monday_l590_59086

theorem day_after_60_days_is_monday
    (birthday_is_thursday : ∃ d : ℕ, d % 7 = 0) :
    ∃ d : ℕ, (d + 60) % 7 = 4 :=
by
  -- Proof steps are omitted here
  sorry

end NUMINAMATH_GPT_day_after_60_days_is_monday_l590_59086


namespace NUMINAMATH_GPT_length_AC_eq_9_74_l590_59025

-- Define the cyclic quadrilateral and given constraints
noncomputable def quad (A B C D : Type) : Prop := sorry
def angle_BAC := 50
def angle_ADB := 60
def AD := 3
def BC := 9

-- Prove that length of AC is 9.74 given the above conditions
theorem length_AC_eq_9_74 
  (A B C D : Type)
  (h_quad : quad A B C D)
  (h_angle_BAC : angle_BAC = 50)
  (h_angle_ADB : angle_ADB = 60)
  (h_AD : AD = 3)
  (h_BC : BC = 9) :
  ∃ AC, AC = 9.74 :=
sorry

end NUMINAMATH_GPT_length_AC_eq_9_74_l590_59025


namespace NUMINAMATH_GPT_robin_total_distance_l590_59020

theorem robin_total_distance
  (d : ℕ)
  (d1 : ℕ)
  (h1 : d = 500)
  (h2 : d1 = 200)
  : 2 * d1 + d = 900 :=
by
  rewrite [h1, h2]
  rfl

end NUMINAMATH_GPT_robin_total_distance_l590_59020


namespace NUMINAMATH_GPT_lisa_hotdog_record_l590_59004

theorem lisa_hotdog_record
  (hotdogs_eaten : ℕ)
  (eaten_in_first_half : ℕ)
  (rate_per_minute : ℕ)
  (time_in_minutes : ℕ)
  (first_half_duration : ℕ)
  (remaining_time : ℕ) :
  eaten_in_first_half = 20 →
  rate_per_minute = 11 →
  first_half_duration = 5 →
  remaining_time = 5 →
  time_in_minutes = first_half_duration + remaining_time →
  hotdogs_eaten = eaten_in_first_half + rate_per_minute * remaining_time →
  hotdogs_eaten = 75 := by
  intros
  sorry

end NUMINAMATH_GPT_lisa_hotdog_record_l590_59004


namespace NUMINAMATH_GPT_car_speed_first_hour_l590_59047

theorem car_speed_first_hour (x : ℝ) (h1 : (x + 75) / 2 = 82.5) : x = 90 :=
sorry

end NUMINAMATH_GPT_car_speed_first_hour_l590_59047


namespace NUMINAMATH_GPT_opposite_of_3_is_neg3_l590_59080

theorem opposite_of_3_is_neg3 : forall (n : ℤ), n = 3 -> -n = -3 :=
by
  sorry

end NUMINAMATH_GPT_opposite_of_3_is_neg3_l590_59080


namespace NUMINAMATH_GPT_part1_part2_part3_l590_59065

noncomputable def A : Set ℝ := { x | x ≥ 1 ∨ x ≤ -3 }
noncomputable def B : Set ℝ := { x | -4 < x ∧ x < 0 }
noncomputable def C : Set ℝ := { x | x ≤ -4 ∨ x ≥ 0 }

theorem part1 : A ∩ B = { x | -4 < x ∧ x ≤ -3 } := 
by { sorry }

theorem part2 : A ∪ B = { x | x < 0 ∨ x ≥ 1 } := 
by { sorry }

theorem part3 : A ∪ C = { x | x ≤ -3 ∨ x ≥ 0 } := 
by { sorry }

end NUMINAMATH_GPT_part1_part2_part3_l590_59065


namespace NUMINAMATH_GPT_xyz_product_condition_l590_59034

theorem xyz_product_condition (x y z : ℝ) (h : x^2 + y^2 = x * y * (z + 1 / z)) : 
  x = y * z ∨ y = x * z :=
sorry

end NUMINAMATH_GPT_xyz_product_condition_l590_59034
