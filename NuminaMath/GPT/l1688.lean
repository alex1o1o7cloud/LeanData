import Mathlib

namespace NUMINAMATH_GPT_shirts_and_pants_neither_plaid_nor_purple_l1688_168861

variable (total_shirts total_pants plaid_shirts purple_pants : Nat)

def non_plaid_shirts (total_shirts plaid_shirts : Nat) : Nat := total_shirts - plaid_shirts
def non_purple_pants (total_pants purple_pants : Nat) : Nat := total_pants - purple_pants

theorem shirts_and_pants_neither_plaid_nor_purple :
  total_shirts = 5 → total_pants = 24 → plaid_shirts = 3 → purple_pants = 5 →
  non_plaid_shirts total_shirts plaid_shirts + non_purple_pants total_pants purple_pants = 21 :=
by
  intros
  -- Placeholder for proof to ensure the theorem builds correctly
  sorry

end NUMINAMATH_GPT_shirts_and_pants_neither_plaid_nor_purple_l1688_168861


namespace NUMINAMATH_GPT_geometric_mean_2_6_l1688_168859

theorem geometric_mean_2_6 : ∃ x : ℝ, x^2 = 2 * 6 ∧ (x = 2 * Real.sqrt 3 ∨ x = - (2 * Real.sqrt 3)) :=
by
  sorry

end NUMINAMATH_GPT_geometric_mean_2_6_l1688_168859


namespace NUMINAMATH_GPT_probability_correct_arrangement_l1688_168857

-- Definitions for conditions
def characters := {c : String | c = "医" ∨ c = "国"}

def valid_arrangements : Finset (List String) := 
    {["医", "医", "国"], ["医", "国", "医"], ["国", "医", "医"]}

def correct_arrangements : Finset (List String) := 
    {["医", "医", "国"], ["医", "国", "医"]}

-- Theorem statement
theorem probability_correct_arrangement :
  (correct_arrangements.card : ℚ) / (valid_arrangements.card : ℚ) = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_probability_correct_arrangement_l1688_168857


namespace NUMINAMATH_GPT_solve_equation_l1688_168832

theorem solve_equation (x y z : ℤ) (h : 19 * (x + y) + z = 19 * (-x + y) - 21) (hx : x = 1) : z = -59 := by
  sorry

end NUMINAMATH_GPT_solve_equation_l1688_168832


namespace NUMINAMATH_GPT_reflection_line_sum_l1688_168841

theorem reflection_line_sum (m b : ℝ) :
  (∀ (x y x' y' : ℝ), (x, y) = (2, 5) → (x', y') = (6, 1) →
  y' = m * x' + b ∧ y = m * x + b) → 
  m + b = 0 :=
sorry

end NUMINAMATH_GPT_reflection_line_sum_l1688_168841


namespace NUMINAMATH_GPT_calculate_expression_l1688_168821

theorem calculate_expression : 7 + 15 / 3 - 5 * 2 = 2 :=
by sorry

end NUMINAMATH_GPT_calculate_expression_l1688_168821


namespace NUMINAMATH_GPT_range_of_m_l1688_168800

open Set Real

noncomputable def f (x m : ℝ) : ℝ := abs (x^2 - 4 * x + 9 - 2 * m) + 2 * m

theorem range_of_m
  (h1 : ∀ x ∈ Icc (0 : ℝ) 4, f x m ≤ 9) : m ≤ 7 / 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1688_168800


namespace NUMINAMATH_GPT_find_d_l1688_168843

-- Definitions of the conditions
variables (r s t u d : ℤ)

-- Assume r, s, t, and u are positive integers
axiom r_pos : r > 0
axiom s_pos : s > 0
axiom t_pos : t > 0
axiom u_pos : u > 0

-- Given conditions
axiom h1 : r ^ 5 = s ^ 4
axiom h2 : t ^ 3 = u ^ 2
axiom h3 : t - r = 19
axiom h4 : d = u - s

-- Proof statement
theorem find_d : d = 757 :=
by sorry

end NUMINAMATH_GPT_find_d_l1688_168843


namespace NUMINAMATH_GPT_range_of_a_l1688_168892

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 2 * x^2 - 3 * a * x + 9 ≥ 0) ↔ (-2 * Real.sqrt 2 ≤ a ∧ a ≤ 2 * Real.sqrt 2) :=
sorry

end NUMINAMATH_GPT_range_of_a_l1688_168892


namespace NUMINAMATH_GPT_earnings_difference_l1688_168804

theorem earnings_difference :
  let oula_deliveries := 96
  let tona_deliveries := oula_deliveries * 3 / 4
  let area_A_fee := 100
  let area_B_fee := 125
  let area_C_fee := 150
  let oula_area_A_deliveries := 48
  let oula_area_B_deliveries := 32
  let oula_area_C_deliveries := 16
  let tona_area_A_deliveries := 27
  let tona_area_B_deliveries := 18
  let tona_area_C_deliveries := 9
  let oula_total_earnings := oula_area_A_deliveries * area_A_fee + oula_area_B_deliveries * area_B_fee + oula_area_C_deliveries * area_C_fee
  let tona_total_earnings := tona_area_A_deliveries * area_A_fee + tona_area_B_deliveries * area_B_fee + tona_area_C_deliveries * area_C_fee
  oula_total_earnings - tona_total_earnings = 4900 := by
sorry

end NUMINAMATH_GPT_earnings_difference_l1688_168804


namespace NUMINAMATH_GPT_ratio_one_six_to_five_eighths_l1688_168853

theorem ratio_one_six_to_five_eighths : (1 / 6) / (5 / 8) = 4 / 15 := by
  sorry

end NUMINAMATH_GPT_ratio_one_six_to_five_eighths_l1688_168853


namespace NUMINAMATH_GPT_initial_water_percentage_l1688_168886

variable (W : ℝ) -- Initial percentage of water in the milk

theorem initial_water_percentage 
  (final_water_content : ℝ := 2) 
  (pure_milk_added : ℝ := 15) 
  (initial_milk_volume : ℝ := 10)
  (final_mixture_volume : ℝ := initial_milk_volume + pure_milk_added)
  (water_equation : W / 100 * initial_milk_volume = final_water_content / 100 * final_mixture_volume) 
  : W = 5 :=
by
  sorry

end NUMINAMATH_GPT_initial_water_percentage_l1688_168886


namespace NUMINAMATH_GPT_tangent_line_at_point_l1688_168818

noncomputable def tangent_line_equation (x y : ℝ) : Prop :=
x + 4 * y - 3 = 0

theorem tangent_line_at_point (x y : ℝ) (h₁ : y = 1 / x^2) (h₂ : x = 2) (h₃ : y = 1/4) :
  tangent_line_equation x y :=
by 
  sorry

end NUMINAMATH_GPT_tangent_line_at_point_l1688_168818


namespace NUMINAMATH_GPT_axis_of_symmetry_l1688_168899

theorem axis_of_symmetry {a b c : ℝ} (h1 : (2 : ℝ) * (a * 2 + b) + c = 5) (h2 : (4 : ℝ) * (a * 4 + b) + c = 5) : 
  (2 + 4) / 2 = 3 := 
by 
  sorry

end NUMINAMATH_GPT_axis_of_symmetry_l1688_168899


namespace NUMINAMATH_GPT_sum_of_a_b_l1688_168825

theorem sum_of_a_b (a b : ℝ) (h1 : a - b = 1) (h2 : a^2 + b^2 = 25) : a + b = 7 ∨ a + b = -7 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_a_b_l1688_168825


namespace NUMINAMATH_GPT_perimeter_of_square_l1688_168808

-- Definitions based on problem conditions
def is_square_divided_into_four_congruent_rectangles (s : ℝ) (rect_perimeter : ℝ) : Prop :=
  rect_perimeter = 30 ∧ s > 0

-- Statement of the theorem to be proved
theorem perimeter_of_square (s : ℝ) (rect_perimeter : ℝ) (h : is_square_divided_into_four_congruent_rectangles s rect_perimeter) :
  4 * s = 48 :=
by sorry

end NUMINAMATH_GPT_perimeter_of_square_l1688_168808


namespace NUMINAMATH_GPT_circumscribed_circle_radius_l1688_168878

-- Definitions of side lengths
def a : ℕ := 5
def b : ℕ := 12

-- Defining the hypotenuse based on the Pythagorean theorem
def hypotenuse (a b : ℕ) : ℕ := Nat.sqrt (a * a + b * b)

-- Radius of the circumscribed circle of a right triangle
def radius (hypotenuse : ℕ) : ℕ := hypotenuse / 2

-- Theorem: The radius of the circumscribed circle of the right triangle is 13 / 2 = 6.5
theorem circumscribed_circle_radius : 
  radius (hypotenuse a b) = 13 / 2 :=
by
  sorry

end NUMINAMATH_GPT_circumscribed_circle_radius_l1688_168878


namespace NUMINAMATH_GPT_johnny_guitar_practice_l1688_168871

theorem johnny_guitar_practice :
  ∃ x : ℕ, (∃ d : ℕ, d = 20 ∧ ∀ n : ℕ, (n = x - d ∧ n = x / 2)) ∧ (x + 80 = 3 * x) :=
by
  sorry

end NUMINAMATH_GPT_johnny_guitar_practice_l1688_168871


namespace NUMINAMATH_GPT_range_of_m_l1688_168810

noncomputable def f (x : ℝ) : ℝ := x^2 - 2*x - 3

theorem range_of_m:
  ∀ m : ℝ, 
  (∀ x, 0 ≤ x ∧ x ≤ m → f x ≤ -3) ∧ 
  (∃ x, 0 ≤ x ∧ x ≤ m ∧ f x = -4) → 
  1 ≤ m ∧ m ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1688_168810


namespace NUMINAMATH_GPT_ordered_pairs_satisfy_conditions_l1688_168860

theorem ordered_pairs_satisfy_conditions :
  ∀ (a b : ℕ), 0 < a → 0 < b → (a^2 + b^2 + 25 = 15 * a * b) → Nat.Prime (a^2 + a * b + b^2) →
  (a = 1 ∧ b = 2) ∨ (a = 2 ∧ b = 1) :=
by
  intros a b ha hb h1 h2
  sorry

end NUMINAMATH_GPT_ordered_pairs_satisfy_conditions_l1688_168860


namespace NUMINAMATH_GPT_total_distance_correct_l1688_168850

noncomputable def total_distance_covered (rA rB rC : ℝ) (revA revB revC : ℕ) : ℝ :=
  let pi := Real.pi
  let circumference (r : ℝ) := 2 * pi * r
  let distance (r : ℝ) (rev : ℕ) := circumference r * rev
  distance rA revA + distance rB revB + distance rC revC

theorem total_distance_correct :
  total_distance_covered 22.4 35.7 55.9 600 450 375 = 316015.4 :=
by
  sorry

end NUMINAMATH_GPT_total_distance_correct_l1688_168850


namespace NUMINAMATH_GPT_polynomial_remainder_l1688_168837

theorem polynomial_remainder (x : ℤ) :
  let poly := x^5 + 3*x^3 + 1
  let divisor := (x + 1)^2
  let remainder := 5*x + 9
  ∃ q : ℤ, poly = divisor * q + remainder := by
  sorry

end NUMINAMATH_GPT_polynomial_remainder_l1688_168837


namespace NUMINAMATH_GPT_find_hyperbola_m_l1688_168898

theorem find_hyperbola_m (m : ℝ) (h : m > 0) : 
  (∀ x y : ℝ, (x^2 / m - y^2 / 3 = 1 → y = 1 / 2 * x)) → m = 12 :=
by
  intros
  sorry

end NUMINAMATH_GPT_find_hyperbola_m_l1688_168898


namespace NUMINAMATH_GPT_evaluate_x_l1688_168890

variable {R : Type*} [LinearOrderedField R]

theorem evaluate_x (m n k x : R) (hm : m ≠ 0) (hn : n ≠ 0) (h : m ≠ n) (h_eq : (x + m)^2 - (x + n)^2 = k * (m - n)^2) :
  x = ((k - 1) * (m + n) - 2 * k * n) / 2 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_x_l1688_168890


namespace NUMINAMATH_GPT_problem_1_problem_2_l1688_168864

open Set Real

noncomputable def A : Set ℝ := {x | x^2 - 3 * x - 18 ≤ 0}

noncomputable def B (m : ℝ) : Set ℝ := {x | m - 8 ≤ x ∧ x ≤ m + 4}

theorem problem_1 : (m = 3) → ((compl A) ∩ (B m) = {x | (-5 ≤ x ∧ x < -3) ∨ (6 < x ∧ x ≤ 7)}) :=
by
  sorry

theorem problem_2 : (A ∩ (B m) = A) → (2 ≤ m ∧ m ≤ 5) :=
by
  sorry

end NUMINAMATH_GPT_problem_1_problem_2_l1688_168864


namespace NUMINAMATH_GPT_right_triangle_side_length_l1688_168851

theorem right_triangle_side_length (c a b : ℕ) (hc : c = 13) (ha : a = 12) (hypotenuse_eq : c ^ 2 = a ^ 2 + b ^ 2) : b = 5 :=
sorry

end NUMINAMATH_GPT_right_triangle_side_length_l1688_168851


namespace NUMINAMATH_GPT_extreme_value_f_max_b_a_plus_1_l1688_168887

noncomputable def f (x : ℝ) := Real.exp x - x + (1/2)*x^2

noncomputable def g (x : ℝ) (a b : ℝ) := (1/2)*x^2 + a*x + b

theorem extreme_value_f :
  ∃ x, deriv f x = 0 ∧ f x = 3 / 2 :=
sorry

theorem max_b_a_plus_1 (a : ℝ) (b : ℝ) :
  (∀ x, f x ≥ g x a b) → b * (a+1) ≤ (a+1)^2 - (a+1)^2 * Real.log (a+1) :=
sorry

end NUMINAMATH_GPT_extreme_value_f_max_b_a_plus_1_l1688_168887


namespace NUMINAMATH_GPT_find_DG_l1688_168834

theorem find_DG (a b S k l DG BC : ℕ) (h1: S = 17 * (a + b)) (h2: S % a = 0) (h3: S % b = 0) (h4: a = S / k) (h5: b = S / l) (h6: BC = 17) (h7: (k - 17) * (l - 17) = 289) : DG = 306 :=
by
  sorry

end NUMINAMATH_GPT_find_DG_l1688_168834


namespace NUMINAMATH_GPT_sqrt_fraction_simplification_l1688_168884

theorem sqrt_fraction_simplification :
  (Real.sqrt ((25 / 49) - (16 / 81)) = (Real.sqrt 1241) / 63) := by
  sorry

end NUMINAMATH_GPT_sqrt_fraction_simplification_l1688_168884


namespace NUMINAMATH_GPT_jesses_room_length_l1688_168842

theorem jesses_room_length 
  (width : ℝ)
  (tile_area : ℝ)
  (num_tiles : ℕ)
  (total_area : ℝ := num_tiles * tile_area) 
  (room_length : ℝ := total_area / width)
  (hw : width = 12)
  (hta : tile_area = 4)
  (hnt : num_tiles = 6) :
  room_length = 2 :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_jesses_room_length_l1688_168842


namespace NUMINAMATH_GPT_blueberries_per_basket_l1688_168873

-- Definitions based on the conditions
def total_blueberries : ℕ := 200
def total_baskets : ℕ := 10

-- Statement to be proven
theorem blueberries_per_basket : total_blueberries / total_baskets = 20 := 
by
  sorry

end NUMINAMATH_GPT_blueberries_per_basket_l1688_168873


namespace NUMINAMATH_GPT_total_selling_price_correct_l1688_168833

-- Define the conditions
def metres_of_cloth : ℕ := 500
def loss_per_metre : ℕ := 5
def cost_price_per_metre : ℕ := 41
def selling_price_per_metre : ℕ := cost_price_per_metre - loss_per_metre
def expected_total_selling_price : ℕ := 18000

-- Define the theorem
theorem total_selling_price_correct : 
  selling_price_per_metre * metres_of_cloth = expected_total_selling_price := 
by
  sorry

end NUMINAMATH_GPT_total_selling_price_correct_l1688_168833


namespace NUMINAMATH_GPT_num_chairs_l1688_168812

variable (C : Nat)
variable (tables_sticks : Nat := 6 * 9)
variable (stools_sticks : Nat := 4 * 2)
variable (total_sticks_needed : Nat := 34 * 5)
variable (total_sticks_chairs : Nat := 6 * C)

theorem num_chairs (h : total_sticks_chairs + tables_sticks + stools_sticks = total_sticks_needed) : C = 18 := 
by sorry

end NUMINAMATH_GPT_num_chairs_l1688_168812


namespace NUMINAMATH_GPT_value_of_f_at_5_l1688_168819

def f (x : ℝ) : ℝ := 4 * x + 2

theorem value_of_f_at_5 : f 5 = 22 :=
by
  sorry

end NUMINAMATH_GPT_value_of_f_at_5_l1688_168819


namespace NUMINAMATH_GPT_triangle_area_correct_l1688_168829

def vector_2d (x y : ℝ) : ℝ × ℝ := (x, y)

def area_of_triangle (a b : ℝ × ℝ) : ℝ :=
  0.5 * abs (a.1 * b.2 - a.2 * b.1)

def a : ℝ × ℝ := vector_2d 3 2
def b : ℝ × ℝ := vector_2d 1 5

theorem triangle_area_correct : area_of_triangle a b = 6.5 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_correct_l1688_168829


namespace NUMINAMATH_GPT_probability_of_at_least_ten_heads_in_twelve_given_first_two_heads_l1688_168854

-- Define a fair coin
inductive Coin
| Heads
| Tails

def fair_coin : List Coin := [Coin.Heads, Coin.Tails]

-- Define a function to calculate the binomial coefficient
def binomial (n k : ℕ) : ℕ :=
  Nat.descFactorial n k / k.factorial

-- Define a function to calculate the probability of at least 8 heads in 10 flips
def prob_at_least_eight_heads_in_ten : ℚ :=
  (binomial 10 8 + binomial 10 9 + binomial 10 10) / (2 ^ 10)

-- Define our theorem statement
theorem probability_of_at_least_ten_heads_in_twelve_given_first_two_heads :
    (prob_at_least_eight_heads_in_ten = 7 / 128) :=
  by
    -- The proof steps can be written here later
    sorry

end NUMINAMATH_GPT_probability_of_at_least_ten_heads_in_twelve_given_first_two_heads_l1688_168854


namespace NUMINAMATH_GPT_large_circle_radius_l1688_168896

noncomputable def radius_of_large_circle : ℝ :=
  let r_small := 1
  let side_length := 2 * r_small
  let diagonal_length := Real.sqrt (side_length ^ 2 + side_length ^ 2)
  let radius_large := (diagonal_length / 2) + r_small
  radius_large + r_small

theorem large_circle_radius :
  radius_of_large_circle = Real.sqrt 2 + 2 :=
by
  sorry

end NUMINAMATH_GPT_large_circle_radius_l1688_168896


namespace NUMINAMATH_GPT_range_of_b_l1688_168816

theorem range_of_b :
  (∀ b, (∀ x : ℝ, x ≥ 1 → Real.log (2^x - b) ≥ 0) → b ≤ 1) :=
sorry

end NUMINAMATH_GPT_range_of_b_l1688_168816


namespace NUMINAMATH_GPT_area_of_shaded_region_l1688_168839

theorem area_of_shaded_region : 
  let side_length := 4
  let radius := side_length / 2 
  let area_of_square := side_length * side_length 
  let area_of_one_quarter_circle := (pi * radius * radius) / 4
  let total_area_of_quarter_circles := 4 * area_of_one_quarter_circle 
  let area_of_shaded_region := area_of_square - total_area_of_quarter_circles 
  area_of_shaded_region = 16 - 4 * pi :=
by
  let side_length := 4
  let radius := side_length / 2
  let area_of_square := side_length * side_length
  let area_of_one_quarter_circle := (pi * radius * radius) / 4
  let total_area_of_quarter_circles := 4 * area_of_one_quarter_circle
  let area_of_shaded_region := area_of_square - total_area_of_quarter_circles
  sorry

end NUMINAMATH_GPT_area_of_shaded_region_l1688_168839


namespace NUMINAMATH_GPT_inequality_property_l1688_168874

theorem inequality_property (a b : ℝ) (h : a > b) : -5 * a < -5 * b := sorry

end NUMINAMATH_GPT_inequality_property_l1688_168874


namespace NUMINAMATH_GPT_boys_and_girls_arrangement_l1688_168846

theorem boys_and_girls_arrangement : 
  ∃ (arrangements : ℕ), arrangements = 48 :=
  sorry

end NUMINAMATH_GPT_boys_and_girls_arrangement_l1688_168846


namespace NUMINAMATH_GPT_find_eccentricity_l1688_168868

variable (a b : ℝ) (ha : a > 0) (hb : b > 0)
variable (asymp_cond : b / a = 1 / 2)

theorem find_eccentricity : ∃ e : ℝ, e = Real.sqrt 5 / 2 :=
by
  let c := Real.sqrt ((a^2 + b^2) / 4)
  let e := c / a
  use e
  sorry

end NUMINAMATH_GPT_find_eccentricity_l1688_168868


namespace NUMINAMATH_GPT_true_propositions_count_l1688_168827

theorem true_propositions_count (b : ℤ) :
  (b = 3 → b^2 = 9) → 
  (∃! p : Prop, p = (b^2 ≠ 9 → b ≠ 3) ∨ p = (b ≠ 3 → b^2 ≠ 9) ∨ p = (b^2 = 9 → b = 3) ∧ (p = (b^2 ≠ 9 → b ≠ 3))) :=
sorry

end NUMINAMATH_GPT_true_propositions_count_l1688_168827


namespace NUMINAMATH_GPT_eight_in_C_l1688_168848

def C : Set ℕ := {x | 1 ≤ x ∧ x ≤ 10}

theorem eight_in_C : 8 ∈ C :=
by {
  sorry
}

end NUMINAMATH_GPT_eight_in_C_l1688_168848


namespace NUMINAMATH_GPT_positive_difference_perimeters_l1688_168866

def perimeter_rectangle (length : ℕ) (width : ℕ) : ℕ :=
  2 * (length + width)

def perimeter_cross_shape : ℕ := 
  let top_and_bottom := 3 + 3 -- top and bottom edges
  let left_and_right := 3 + 3 -- left and right edges
  let internal_subtraction := 4
  top_and_bottom + left_and_right - internal_subtraction

theorem positive_difference_perimeters :
  let length := 4
  let width := 3
  perimeter_rectangle length width - perimeter_cross_shape = 6 :=
by
  let length := 4
  let width := 3
  sorry

end NUMINAMATH_GPT_positive_difference_perimeters_l1688_168866


namespace NUMINAMATH_GPT_number_of_square_integers_l1688_168838

theorem number_of_square_integers (n : ℤ) (h1 : (0 ≤ n) ∧ (n < 30)) :
  (∃ (k : ℕ), n = 0 ∨ n = 15 ∨ n = 24 → ∃ (k : ℕ), n / (30 - n) = k^2) :=
by
  sorry

end NUMINAMATH_GPT_number_of_square_integers_l1688_168838


namespace NUMINAMATH_GPT_janous_inequality_l1688_168826

theorem janous_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x ≥ y + z) :
  (x + y) / z + (y + z) / x + (z + x) / y ≥ 7 :=
sorry

end NUMINAMATH_GPT_janous_inequality_l1688_168826


namespace NUMINAMATH_GPT_cos_six_arccos_two_fifths_l1688_168849

noncomputable def arccos (x : ℝ) : ℝ := Real.arccos x
noncomputable def cos (x : ℝ) : ℝ := Real.cos x

theorem cos_six_arccos_two_fifths : cos (6 * arccos (2 / 5)) = 12223 / 15625 := 
by
  sorry

end NUMINAMATH_GPT_cos_six_arccos_two_fifths_l1688_168849


namespace NUMINAMATH_GPT_value_of_expression_l1688_168894

variables {A B C : ℚ}

def conditions (A B C : ℚ) : Prop := A / B = 3 / 2 ∧ B / C = 2 / 5

theorem value_of_expression (h : conditions A B C) : 
  (4 * A + 3 * B) / (5 * C - 2 * A) = 18 / 19 :=
sorry

end NUMINAMATH_GPT_value_of_expression_l1688_168894


namespace NUMINAMATH_GPT_symmetric_point_l1688_168823

theorem symmetric_point (P Q : ℝ × ℝ)
  (l : ℝ → ℝ)
  (P_coords : P = (-1, 2))
  (l_eq : ∀ x, l x = x - 1) :
  Q = (3, -2) :=
by
  sorry

end NUMINAMATH_GPT_symmetric_point_l1688_168823


namespace NUMINAMATH_GPT_cosine_120_eq_negative_half_l1688_168803

theorem cosine_120_eq_negative_half :
  let θ := 120 * Real.pi / 180
  let point := (Real.cos θ, Real.sin θ)
  point.1 = -1 / 2 := by
  sorry

end NUMINAMATH_GPT_cosine_120_eq_negative_half_l1688_168803


namespace NUMINAMATH_GPT_contrapositive_proposition_l1688_168856

theorem contrapositive_proposition (α : ℝ) :
  (α = π / 4 → Real.tan α = 1) ↔ (Real.tan α ≠ 1 → α ≠ π / 4) :=
by
  sorry

end NUMINAMATH_GPT_contrapositive_proposition_l1688_168856


namespace NUMINAMATH_GPT_edward_money_l1688_168880

theorem edward_money (X : ℝ) (H1 : X - 130 - 0.25 * (X - 130) = 270) : X = 490 :=
by
  sorry

end NUMINAMATH_GPT_edward_money_l1688_168880


namespace NUMINAMATH_GPT_negation_of_forall_exp_gt_zero_l1688_168855

open Real

theorem negation_of_forall_exp_gt_zero : 
  (¬ (∀ x : ℝ, exp x > 0)) ↔ (∃ x : ℝ, exp x ≤ 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_forall_exp_gt_zero_l1688_168855


namespace NUMINAMATH_GPT_tee_shirts_with_60_feet_of_material_l1688_168863

def tee_shirts (f t : ℕ) : ℕ := t / f

theorem tee_shirts_with_60_feet_of_material :
  tee_shirts 4 60 = 15 :=
by
  sorry

end NUMINAMATH_GPT_tee_shirts_with_60_feet_of_material_l1688_168863


namespace NUMINAMATH_GPT_sum_repeating_decimals_as_fraction_l1688_168883

-- Definitions for repeating decimals
def rep2 : ℝ := 0.2222
def rep02 : ℝ := 0.0202
def rep0002 : ℝ := 0.00020002

-- Prove the sum of the repeating decimals is equal to the given fraction
theorem sum_repeating_decimals_as_fraction :
  rep2 + rep02 + rep0002 = (2224 / 9999 : ℝ) :=
sorry

end NUMINAMATH_GPT_sum_repeating_decimals_as_fraction_l1688_168883


namespace NUMINAMATH_GPT_atomic_weight_Oxygen_l1688_168888

theorem atomic_weight_Oxygen :
  ∀ (Ba_atomic_weight S_atomic_weight : ℝ),
    (Ba_atomic_weight = 137.33) →
    (S_atomic_weight = 32.07) →
    (Ba_atomic_weight + S_atomic_weight + 4 * 15.9 = 233) →
    15.9 = 233 - 137.33 - 32.07 / 4 := 
by
  intros Ba_atomic_weight S_atomic_weight hBa hS hm
  sorry

end NUMINAMATH_GPT_atomic_weight_Oxygen_l1688_168888


namespace NUMINAMATH_GPT_parabola_intersects_x_axis_two_points_l1688_168844

theorem parabola_intersects_x_axis_two_points (m : ℝ) : 
    ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ mx^2 + (m-3)*x - 1 = 0 :=
by
  sorry

end NUMINAMATH_GPT_parabola_intersects_x_axis_two_points_l1688_168844


namespace NUMINAMATH_GPT_value_of_b_l1688_168875

theorem value_of_b (b : ℝ) (h1 : 1/2 * (b / 3) * b = 6) (h2 : b ≥ 0) : b = 6 := sorry

end NUMINAMATH_GPT_value_of_b_l1688_168875


namespace NUMINAMATH_GPT_length_linear_function_alpha_increase_l1688_168897

variable (l : ℝ) (l₀ : ℝ) (t : ℝ) (α : ℝ)

theorem length_linear_function 
  (h_formula : l = l₀ * (1 + α * t)) : 
  ∃ (f : ℝ → ℝ), (∀ t, f t = l₀ + l₀ * α * t ∧ (l = f t)) :=
by {
  -- Proof would go here
  sorry
}

theorem alpha_increase 
  (h_formula : l = l₀ * (1 + α * t))
  (h_initial : t = 1) :
  α = (l - l₀) / l₀ :=
by {
  -- Proof would go here
  sorry
}

end NUMINAMATH_GPT_length_linear_function_alpha_increase_l1688_168897


namespace NUMINAMATH_GPT_probability_even_first_odd_second_l1688_168867

-- Definitions based on the conditions
def die_sides : Finset ℕ := {1, 2, 3, 4, 5, 6}
def even_numbers : Finset ℕ := {2, 4, 6}
def odd_numbers : Finset ℕ := {1, 3, 5}

-- Probability calculations
def prob_even := (even_numbers.card : ℚ) / (die_sides.card : ℚ)
def prob_odd := (odd_numbers.card : ℚ) / (die_sides.card : ℚ)

-- Proof statement
theorem probability_even_first_odd_second :
  prob_even * prob_odd = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_probability_even_first_odd_second_l1688_168867


namespace NUMINAMATH_GPT_third_angle_of_triangle_l1688_168820

theorem third_angle_of_triangle (a b : ℝ) (h₁ : a = 25) (h₂ : b = 70) : 180 - a - b = 85 := 
by
  sorry

end NUMINAMATH_GPT_third_angle_of_triangle_l1688_168820


namespace NUMINAMATH_GPT_evaluate_expression_l1688_168872

theorem evaluate_expression : 
  (4 * 6 / (12 * 16)) * (8 * 12 * 16 / (4 * 6 * 8)) = 1 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1688_168872


namespace NUMINAMATH_GPT_rita_needs_9_months_l1688_168817

def total_required_hours : ℕ := 4000
def backstroke_hours : ℕ := 100
def breaststroke_hours : ℕ := 40
def butterfly_hours : ℕ := 320
def monthly_practice_hours : ℕ := 400

def hours_already_completed : ℕ := backstroke_hours + breaststroke_hours + butterfly_hours
def remaining_hours : ℕ := total_required_hours - hours_already_completed
def months_needed : ℕ := (remaining_hours + monthly_practice_hours - 1) / monthly_practice_hours -- Ceiling division

theorem rita_needs_9_months :
  months_needed = 9 := by
  sorry

end NUMINAMATH_GPT_rita_needs_9_months_l1688_168817


namespace NUMINAMATH_GPT_sum_of_fraction_components_l1688_168852

def repeating_decimal_to_fraction (a b : ℕ) := a = 35 ∧ b = 99 ∧ Nat.gcd a b = 1

theorem sum_of_fraction_components :
  ∃ a b : ℕ, repeating_decimal_to_fraction a b ∧ a + b = 134 :=
by {
  sorry
}

end NUMINAMATH_GPT_sum_of_fraction_components_l1688_168852


namespace NUMINAMATH_GPT_monica_read_books_l1688_168835

theorem monica_read_books (x : ℕ) 
    (h1 : 2 * (2 * x) + 5 = 69) : 
    x = 16 :=
by 
  sorry

end NUMINAMATH_GPT_monica_read_books_l1688_168835


namespace NUMINAMATH_GPT_initial_average_marks_is_90_l1688_168885

def incorrect_average_marks (A : ℝ) : Prop :=
  let wrong_sum := 10 * A
  let correct_sum := 10 * 95
  wrong_sum + 50 = correct_sum

theorem initial_average_marks_is_90 : ∃ A : ℝ, incorrect_average_marks A ∧ A = 90 :=
by
  use 90
  unfold incorrect_average_marks
  simp
  sorry

end NUMINAMATH_GPT_initial_average_marks_is_90_l1688_168885


namespace NUMINAMATH_GPT_smaller_number_is_24_l1688_168870

theorem smaller_number_is_24 (x y : ℕ) (h1 : x + y = 44) (h2 : 5 * x = 6 * y) : x = 24 :=
by
  sorry

end NUMINAMATH_GPT_smaller_number_is_24_l1688_168870


namespace NUMINAMATH_GPT_find_a_l1688_168814

theorem find_a (a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) 
  (h_asymptote : ∀ x : ℝ, x = π/2 ∨ x = 3*π/2 ∨ x = -π/2 ∨ x = -3*π/2 → b*x = π/2 ∨ b*x = 3*π/2 ∨ b*x = -π/2 ∨ b*x = -3*π/2)
  (h_amplitude : ∀ x : ℝ, |a * (1 / Real.cos (b * x))| ≤ 3): 
  a = 3 := 
sorry

end NUMINAMATH_GPT_find_a_l1688_168814


namespace NUMINAMATH_GPT_pencil_case_solution_part1_pencil_case_solution_part2_1_pencil_case_solution_part2_2_l1688_168824

section pencil_case_problem

variables (x m : ℕ)

-- Part 1: The cost prices of each $A$ type and $B$ type pencil cases.
def cost_price_A (x : ℕ) : Prop := 
  (800 : ℝ) / x = (1000 : ℝ) / (x + 2)

-- Part 2.1: Maximum quantity of $B$ type pencil cases.
def max_quantity_B (m : ℕ) : Prop := 
  3 * m - 50 + m ≤ 910

-- Part 2.2: Number of different scenarios for purchasing the pencil cases.
def profit_condition (m : ℕ) : Prop := 
  4 * (3 * m - 50) + 5 * m > 3795

theorem pencil_case_solution_part1 (hA : cost_price_A x) : 
  x = 8 := 
sorry

theorem pencil_case_solution_part2_1 (hB : max_quantity_B m) : 
  m ≤ 240 := 
sorry

theorem pencil_case_solution_part2_2 (hB : max_quantity_B m) (hp : profit_condition m) : 
  236 ≤ m ∧ m ≤ 240 := 
sorry

end pencil_case_problem

end NUMINAMATH_GPT_pencil_case_solution_part1_pencil_case_solution_part2_1_pencil_case_solution_part2_2_l1688_168824


namespace NUMINAMATH_GPT_solve_system_of_equations_l1688_168879

theorem solve_system_of_equations (a b : ℝ) (h1 : a^2 ≠ 1) (h2 : b^2 ≠ 1) (h3 : a ≠ b) : 
  (∃ x y : ℝ, 
    (x - y) / (1 - x * y) = 2 * a / (1 + a^2) ∧ (x + y) / (1 + x * y) = 2 * b / (1 + b^2) ∧
    ((x = (a * b + 1) / (a + b) ∧ y = (a * b - 1) / (a - b)) ∨ 
     (x = (a + b) / (a * b + 1) ∧ y = (a - b) / (a * b - 1)))) :=
by
  sorry

end NUMINAMATH_GPT_solve_system_of_equations_l1688_168879


namespace NUMINAMATH_GPT_wastewater_volume_2013_l1688_168891

variable (x_2013 x_2014 : ℝ)
variable (condition1 : x_2014 = 38000)
variable (condition2 : x_2014 = 1.6 * x_2013)

theorem wastewater_volume_2013 : x_2013 = 23750 := by
  sorry

end NUMINAMATH_GPT_wastewater_volume_2013_l1688_168891


namespace NUMINAMATH_GPT_remainder_equality_l1688_168882

theorem remainder_equality 
  (Q Q' S S' E s s' : ℕ) 
  (Q_gt_Q' : Q > Q')
  (h1 : Q % E = S)
  (h2 : Q' % E = S')
  (h3 : (Q^2 * Q') % E = s)
  (h4 : (S^2 * S') % E = s') :
  s = s' :=
sorry

end NUMINAMATH_GPT_remainder_equality_l1688_168882


namespace NUMINAMATH_GPT_find_m_l1688_168869

open Real

/-- Define Circle C1 and C2 as having the given equations
and verify their internal tangency to find the possible m values -/
theorem find_m (m : ℝ) :
  (∃ (x y : ℝ), (x - m)^2 + (y + 2)^2 = 9) ∧ 
  (∃ (x y : ℝ), (x + 1)^2 + (y - m)^2 = 4) ∧ 
  (by exact (sqrt ((m + 1)^2 + (-2 - m)^2)) = 3 - 2) → 
  m = -2 ∨ m = -1 := 
sorry -- Proof is omitted

end NUMINAMATH_GPT_find_m_l1688_168869


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l1688_168836

open Set

def A : Set ℤ := {x | 1 ≤ x ∧ x ≤ 3}
def B : Set ℤ := {-1, 0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l1688_168836


namespace NUMINAMATH_GPT_ellipse_chord_through_focus_l1688_168807

theorem ellipse_chord_through_focus (x y : ℝ) (a b : ℝ := 6) (c : ℝ := 3 * Real.sqrt 3)
  (F : ℝ × ℝ := (3 * Real.sqrt 3, 0)) (AF BF : ℝ) :
  (x^2 / 36) + (y^2 / 9) = 1 ∧ ((x - 3 * Real.sqrt 3)^2 + y^2 = (3/2)^2) ∧
  (AF = 3 / 2) ∧ F.1 = 3 * Real.sqrt 3 ∧ F.2 = 0 →
  BF = 3 / 2 :=
sorry

end NUMINAMATH_GPT_ellipse_chord_through_focus_l1688_168807


namespace NUMINAMATH_GPT_average_customers_per_day_l1688_168895

-- Define the number of customers each day:
def customers_per_day : List ℕ := [10, 12, 15, 13, 18, 16, 11]

-- Define the total number of days in a week
def days_in_week : ℕ := 7

-- Define the theorem stating the average number of daily customers
theorem average_customers_per_day :
  (customers_per_day.sum : ℚ) / days_in_week = 13.57 :=
by
  sorry

end NUMINAMATH_GPT_average_customers_per_day_l1688_168895


namespace NUMINAMATH_GPT_provisions_remaining_days_l1688_168865

-- Definitions based on the conditions
def initial_men : ℕ := 1000
def initial_provisions_days : ℕ := 60
def days_elapsed : ℕ := 15
def reinforcement_men : ℕ := 1250

-- Mathematical computation for Lean
def total_provisions : ℕ := initial_men * initial_provisions_days
def provisions_left : ℕ := initial_men * (initial_provisions_days - days_elapsed)
def total_men_after_reinforcement : ℕ := initial_men + reinforcement_men

-- Statement to prove
theorem provisions_remaining_days : provisions_left / total_men_after_reinforcement = 20 :=
by
  -- The proof steps will be filled here, but for now, we use sorry to skip them.
  sorry

end NUMINAMATH_GPT_provisions_remaining_days_l1688_168865


namespace NUMINAMATH_GPT_solve_for_x_minus_y_l1688_168847

theorem solve_for_x_minus_y (x y : ℝ) 
  (h1 : 3 * x - 5 * y = 5)
  (h2 : x / (x + y) = 5 / 7) : 
  x - y = 3 := 
by 
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_solve_for_x_minus_y_l1688_168847


namespace NUMINAMATH_GPT_find_point_B_l1688_168830

def line_segment_parallel_to_x_axis (A B : (ℝ × ℝ)) : Prop :=
  A.snd = B.snd

def length_3 (A B : (ℝ × ℝ)) : Prop :=
  abs (A.fst - B.fst) = 3

theorem find_point_B (A B : (ℝ × ℝ))
  (h₁ : A = (3, 2))
  (h₂ : line_segment_parallel_to_x_axis A B)
  (h₃ : length_3 A B) :
  B = (0, 2) ∨ B = (6, 2) :=
sorry

end NUMINAMATH_GPT_find_point_B_l1688_168830


namespace NUMINAMATH_GPT_percentage_increase_l1688_168877

variable (m y : ℝ)

theorem percentage_increase (h : x = y + (m / 100) * y) : x = ((100 + m) / 100) * y := by
  sorry

end NUMINAMATH_GPT_percentage_increase_l1688_168877


namespace NUMINAMATH_GPT_min_area_triangle_l1688_168889

theorem min_area_triangle (m n : ℝ) (h1 : (1 : ℝ) / m + (2 : ℝ) / n = 1) (h2 : m > 0) (h3 : n > 0) :
  ∃ A B C : ℝ, 
  ((0 < A) ∧ (0 < B) ∧ ((1 : ℝ) / A + (2 : ℝ) / B = 1) ∧ (A * B = C) ∧ (2 / C = mn)) ∧ (C = 4) :=
by
  sorry

end NUMINAMATH_GPT_min_area_triangle_l1688_168889


namespace NUMINAMATH_GPT_remaining_value_subtract_70_percent_from_4500_l1688_168828

theorem remaining_value_subtract_70_percent_from_4500 (num : ℝ) 
  (h : 0.36 * num = 2376) : 4500 - 0.70 * num = -120 :=
by
  sorry

end NUMINAMATH_GPT_remaining_value_subtract_70_percent_from_4500_l1688_168828


namespace NUMINAMATH_GPT_number_of_female_fish_l1688_168802

-- Defining the constants given in the problem
def total_fish : ℕ := 45
def fraction_male : ℚ := 2 / 3

-- The statement we aim to prove in Lean
theorem number_of_female_fish : 
  (total_fish : ℚ) * (1 - fraction_male) = 15 :=
by
  sorry

end NUMINAMATH_GPT_number_of_female_fish_l1688_168802


namespace NUMINAMATH_GPT_arithmetic_first_term_l1688_168876

theorem arithmetic_first_term (a : ℕ) (d : ℕ) (T : ℕ → ℕ) (k : ℕ) :
  (∀ n : ℕ, T n = n * (2 * a + (n - 1) * d) / 2) →
  (∀ n : ℕ, T (4 * n) / T n = k) →
  d = 5 →
  k = 16 →
  a = 3 := 
by
  sorry

end NUMINAMATH_GPT_arithmetic_first_term_l1688_168876


namespace NUMINAMATH_GPT_x_intercept_of_line_is_7_over_2_l1688_168801

-- Definitions for the conditions
def point1 : ℝ × ℝ := (2, -3)
def point2 : ℝ × ℝ := (6, 5)

-- Define what it means to be the x-intercept of the line
def x_intercept_of_line (x : ℝ) : Prop :=
  ∃ m b : ℝ, (point1.snd) = m * (point1.fst) + b ∧ (point2.snd) = m * (point2.fst) + b ∧ 0 = m * x + b

-- The theorem stating the x-intercept
theorem x_intercept_of_line_is_7_over_2 : x_intercept_of_line (7 / 2) :=
sorry

end NUMINAMATH_GPT_x_intercept_of_line_is_7_over_2_l1688_168801


namespace NUMINAMATH_GPT_new_number_shifting_digits_l1688_168806

-- Definitions for the three digits
variables (h t u : ℕ)

-- The original three-digit number
def original_number : ℕ := 100 * h + 10 * t + u

-- The new number formed by placing the digits "12" after the three-digit number
def new_number : ℕ := original_number h t u * 100 + 12

-- The goal is to prove that this new number equals 10000h + 1000t + 100u + 12
theorem new_number_shifting_digits (h t u : ℕ) :
  new_number h t u = 10000 * h + 1000 * t + 100 * u + 12 := 
by
  sorry -- Proof to be filled in

end NUMINAMATH_GPT_new_number_shifting_digits_l1688_168806


namespace NUMINAMATH_GPT_arc_lengths_l1688_168893

-- Definitions for the given conditions
def circumference : ℝ := 80  -- Circumference of the circle

-- Angles in degrees
def angle_AOM : ℝ := 45
def angle_MOB : ℝ := 90

-- Radius of the circle using the formula C = 2 * π * r
noncomputable def radius : ℝ := circumference / (2 * Real.pi)

-- Calculate the arc lengths using the angles
noncomputable def arc_length_AM : ℝ := (angle_AOM / 360) * circumference
noncomputable def arc_length_MB : ℝ := (angle_MOB / 360) * circumference

-- The theorem stating the required lengths
theorem arc_lengths (h : circumference = 80 ∧ angle_AOM = 45 ∧ angle_MOB = 90) :
  arc_length_AM = 10 ∧ arc_length_MB = 20 :=
by
  sorry

end NUMINAMATH_GPT_arc_lengths_l1688_168893


namespace NUMINAMATH_GPT_compare_fractions_l1688_168862

theorem compare_fractions (x y : ℝ) (n : ℕ) (h1 : 0 < x) (h2 : x < 1) (h3 : 0 < y) (h4 : y < 1) (h5 : 0 < n) :
  (x^n / (1 - x^2) + y^n / (1 - y^2)) ≥ ((x^n + y^n) / (1 - x * y)) :=
by sorry

end NUMINAMATH_GPT_compare_fractions_l1688_168862


namespace NUMINAMATH_GPT_number_of_participants_l1688_168881

theorem number_of_participants (n : ℕ) (h : n * (n - 1) / 2 = 171) : n = 19 :=
by
  sorry

end NUMINAMATH_GPT_number_of_participants_l1688_168881


namespace NUMINAMATH_GPT_insurance_percentage_l1688_168831

noncomputable def total_pills_per_year : ℕ := 2 * 365

noncomputable def cost_per_pill : ℕ := 5

noncomputable def total_medication_cost_per_year : ℕ := total_pills_per_year * cost_per_pill

noncomputable def doctor_visits_per_year : ℕ := 2

noncomputable def cost_per_doctor_visit : ℕ := 400

noncomputable def total_doctor_cost_per_year : ℕ := doctor_visits_per_year * cost_per_doctor_visit

noncomputable def total_yearly_cost_without_insurance : ℕ := total_medication_cost_per_year + total_doctor_cost_per_year

noncomputable def total_payment_per_year : ℕ := 1530

noncomputable def insurance_coverage_per_year : ℕ := total_yearly_cost_without_insurance - total_payment_per_year

theorem insurance_percentage:
  (insurance_coverage_per_year * 100) / total_medication_cost_per_year = 80 :=
by sorry

end NUMINAMATH_GPT_insurance_percentage_l1688_168831


namespace NUMINAMATH_GPT_sum_of_values_of_n_l1688_168813

theorem sum_of_values_of_n (n₁ n₂ : ℚ) (h1 : 3 * n₁ - 8 = 5) (h2 : 3 * n₂ - 8 = -5) : n₁ + n₂ = 16 / 3 := 
by {
  -- Use the provided conditions to solve the problem
  sorry 
}

end NUMINAMATH_GPT_sum_of_values_of_n_l1688_168813


namespace NUMINAMATH_GPT_sales_tax_difference_l1688_168822

theorem sales_tax_difference:
  let original_price := 50 
  let discount_rate := 0.10 
  let sales_tax_rate_1 := 0.08
  let sales_tax_rate_2 := 0.075 
  let discounted_price := original_price * (1 - discount_rate) 
  let sales_tax_1 := discounted_price * sales_tax_rate_1 
  let sales_tax_2 := discounted_price * sales_tax_rate_2 
  sales_tax_1 - sales_tax_2 = 0.225 := by
  sorry

end NUMINAMATH_GPT_sales_tax_difference_l1688_168822


namespace NUMINAMATH_GPT_remainder_sum_mod_53_l1688_168815

theorem remainder_sum_mod_53 (a b c d : ℕ)
  (h1 : a % 53 = 31)
  (h2 : b % 53 = 45)
  (h3 : c % 53 = 17)
  (h4 : d % 53 = 6) :
  (a + b + c + d) % 53 = 46 := 
sorry

end NUMINAMATH_GPT_remainder_sum_mod_53_l1688_168815


namespace NUMINAMATH_GPT_side_length_square_l1688_168845

theorem side_length_square (s : ℝ) (h1 : ∃ (s : ℝ), (s > 0)) (h2 : 6 * s^2 = 3456) : s = 24 :=
sorry

end NUMINAMATH_GPT_side_length_square_l1688_168845


namespace NUMINAMATH_GPT_range_of_a_l1688_168811

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, ¬((a^2 - 4) * x^2 + (a + 2) * x - 1 ≥ 0)) → (-2 < a ∧ a ≤ 6/5) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1688_168811


namespace NUMINAMATH_GPT_probability_of_same_color_when_rolling_two_24_sided_dice_l1688_168805

-- Defining the conditions
def numSides : ℕ := 24
def purpleSides : ℕ := 5
def blueSides : ℕ := 8
def redSides : ℕ := 10
def goldSides : ℕ := 1

-- Required to use rational numbers for probabilities
def probability (eventSides : ℕ) (totalSides : ℕ) : ℚ := eventSides / totalSides

-- Main theorem statement
theorem probability_of_same_color_when_rolling_two_24_sided_dice :
  probability purpleSides numSides * probability purpleSides numSides +
  probability blueSides numSides * probability blueSides numSides +
  probability redSides numSides * probability redSides numSides +
  probability goldSides numSides * probability goldSides numSides =
  95 / 288 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_same_color_when_rolling_two_24_sided_dice_l1688_168805


namespace NUMINAMATH_GPT_num_parallelogram_even_l1688_168809

-- Define the conditions of the problem in Lean
def isosceles_right_triangle (base_length : ℕ) := 
  base_length = 2

def square (side_length : ℕ) := 
  side_length = 1

def parallelogram (sides_length : ℕ) (diagonals_length : ℕ) := 
  sides_length = 1 ∧ diagonals_length = 1

-- Main statement to prove
theorem num_parallelogram_even (num_triangles num_squares num_parallelograms : ℕ)
  (Htriangle : ∀ t, t < num_triangles → isosceles_right_triangle 2)
  (Hsquare : ∀ s, s < num_squares → square 1)
  (Hparallelogram : ∀ p, p < num_parallelograms → parallelogram 1 1) :
  num_parallelograms % 2 = 0 := 
sorry

end NUMINAMATH_GPT_num_parallelogram_even_l1688_168809


namespace NUMINAMATH_GPT_symmetric_function_exists_l1688_168840

-- Define the main sets A and B with given cardinalities
def A := { n : ℕ // n < 2011^2 }
def B := { n : ℕ // n < 2010 }

-- The main theorem to prove
theorem symmetric_function_exists :
  ∃ (f : A × A → B), 
  (∀ x y, f (x, y) = f (y, x)) ∧ 
  (∀ g : A → B, ∃ (a1 a2 : A), g a1 = f (a1, a2) ∧ g a2 = f (a1, a2) ∧ a1 ≠ a2) :=
sorry

end NUMINAMATH_GPT_symmetric_function_exists_l1688_168840


namespace NUMINAMATH_GPT_jogging_track_circumference_l1688_168858

/-- 
Given:
- Deepak's speed = 20 km/hr
- His wife's speed = 12 km/hr
- They meet for the first time in 32 minutes

Then:
The circumference of the jogging track is 17.0667 km.
-/
theorem jogging_track_circumference (deepak_speed : ℝ) (wife_speed : ℝ) (meet_time : ℝ)
  (h1 : deepak_speed = 20)
  (h2 : wife_speed = 12)
  (h3 : meet_time = (32 / 60) ) : 
  ∃ circumference : ℝ, circumference = 17.0667 :=
by
  sorry

end NUMINAMATH_GPT_jogging_track_circumference_l1688_168858
