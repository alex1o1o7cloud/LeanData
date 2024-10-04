import Mathlib

namespace sin_alpha_value_l266_266573

variable (α β : ℝ)
variable (h_cos_alpha_minus_beta : cos (α - β) = 3/5)
variable (h_sin_beta : sin β = -5/13)
variable (h_alpha_range : 0 < α ∧ α < π / 2)
variable (h_beta_range : -π / 2 < β ∧ β < 0)

theorem sin_alpha_value : sin α = 63 / 65 :=
by 
  -- This is where the proof would be written
  sorry

end sin_alpha_value_l266_266573


namespace train_speeds_l266_266349

theorem train_speeds (d t x : ℝ) (h1 : d = 450) (h2 : t = 5) (h3 : ∀ x : ℝ, 2 * x + 6 = 90) : 
  x = 42 ∧ x + 6 = 48 :=
by
  have H : 450 = (2 * 42 + 6) * 5 := by
    rw [h1, h2, h3]
  exact ⟨42, 48⟩
  sorry

end train_speeds_l266_266349


namespace find_circle_and_m_l266_266135

-- Definitions for given conditions
def on_line (x y : ℝ) : Prop := x - 2 * y = 0
def tangent_to_y_axis (x y : ℝ) : Prop := x = 0 ∧ y = 1
def circle_eq (x y r : ℝ) : ℝ := (x - 2) ^ 2 + (y - 1) ^ 2 - 4

-- The main theorem
theorem find_circle_and_m (a b r m : ℝ) :
  a = 2 * b →
  b = 1 →
  r = 2 →
  (tangent_to_y_axis 0 1) →
  (on_line a b) →
  (circle_eq a b r = 0) ∧ (|1 + m| = sqrt 2) →
  m = sqrt 2 - 1 ∨ m = -sqrt 2 - 1 :=
by {
  -- Circle equation proof
  sorry,
  -- Finding m proof
  sorry,
}

end find_circle_and_m_l266_266135


namespace third_shiny_on_fifth_draw_probability_l266_266442

/-- 
  Consider a box containing 5 shiny pennies and 5 dull pennies. 
  Each penny is drawn one by one at random and is not replaced.
  The goal is to compute the probability that it will take exactly 
  five draws for the third shiny penny to appear.
  We need to prove that this probability is equal to 5/14.
-/
theorem third_shiny_on_fifth_draw_probability :
  let total_shiny := 5
  let total_dull := 5
  let draws := 5
  let third_shiny_draw_position := 5
  probability_of_third_shiny_on_fifth_draw total_shiny total_dull draws third_shiny_draw_position = 5 / 14 :=
sorry

end third_shiny_on_fifth_draw_probability_l266_266442


namespace num_sixth_powers_below_1000_l266_266794

theorem num_sixth_powers_below_1000 : 
  {n : ℕ | n^6 < 1000 ∧ n > 0}.to_finset.card = 3 :=
by sorry

end num_sixth_powers_below_1000_l266_266794


namespace range_of_a_l266_266596

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, |x - 2| + |x + 2| ≤ a^2 - 3 * a) ↔ (a ≥ 4 ∨ a ≤ -1) :=
by
  sorry

end range_of_a_l266_266596


namespace nadine_chairs_l266_266247

theorem nadine_chairs (total_spent cost_table cost_chair : ℕ) (h_total : total_spent = 56) 
    (h_table : cost_table = 34) (h_chair : cost_chair = 11) : 
    (total_spent - cost_table) / cost_chair = 2 :=
by
  rw [h_total, h_table, h_chair]
  exact Nat.div_eq_of_eq_mul_left (by norm_num) (by norm_num) (by norm_num)

/- This theorem states that given the total amount spent, the cost of the table, 
   and the cost per chair, we conclude that the number of chairs Nadine bought is 2. 
   Here, we use the conditions given in the problem to establish our statement. -/

end nadine_chairs_l266_266247


namespace log_computation_l266_266577

noncomputable def log10 := Real.logBase 10

theorem log_computation (x : ℝ) (hx : x > 1) (h : (log10 x)^2 - log10 (x ^ 2) = 18) : 
  (log10 x)^4 - log10 (x ^ 4) = 472 + 76 * Real.sqrt 19 := by
  sorry

end log_computation_l266_266577


namespace nested_sqrt_eq_l266_266531

noncomputable def y := (sqrt(3 - sqrt(3 - sqrt(3 - sqrt(3 - ...)))))

theorem nested_sqrt_eq : (∃ y: ℝ, y = sqrt (3 - y) → y = (-1 + sqrt 13) / 2) :=
sorry

end nested_sqrt_eq_l266_266531


namespace rug_length_l266_266242

theorem rug_length (d : ℕ) (x y : ℕ) (h1 : x * x + y * y = d * d) (h2 : y / x = 2) (h3 : (x = 25 ∧ y = 50)) : 
  x = 25 := 
sorry

end rug_length_l266_266242


namespace polygon_sides_from_diagonals_l266_266864

theorem polygon_sides_from_diagonals (n : ℕ) (h : ↑((n * (n - 3)) / 2) = 14) : n = 7 :=
by
  sorry

end polygon_sides_from_diagonals_l266_266864


namespace min_cards_required_l266_266332

theorem min_cards_required 
  (card_vals : Set ℕ) 
  (has_1 : 1 ∈ card_vals) 
  (has_3 : 3 ∈ card_vals) 
  (has_5 : 5 ∈ card_vals) 
  (has_7 : 7 ∈ card_vals) 
  (has_9 : 9 ∈ card_vals)
  (cards_each : ∀ x ∈ card_vals, 30): 
  (∃ cards : ℕ → ℕ, (∀ n ∈ (Finset.range 201), ∃ (card_comb : Finset ℕ), card_comb.card ≤ 30 ∧ card_comb.sum cards = n) → (cards 1 + cards 3 + cards 5 + cards 7 + cards 9 ≥ 26)) := 
sorry

end min_cards_required_l266_266332


namespace number_of_individuals_left_at_zoo_l266_266456

theorem number_of_individuals_left_at_zoo 
  (students_class1 students_class2 students_left : ℕ)
  (initial_chaperones remaining_chaperones teachers : ℕ) :
  students_class1 = 10 ∧
  students_class2 = 10 ∧
  initial_chaperones = 5 ∧
  teachers = 2 ∧
  students_left = 10 ∧
  remaining_chaperones = initial_chaperones - 2 →
  (students_class1 + students_class2 - students_left) + remaining_chaperones + teachers = 15 :=
by
  sorry

end number_of_individuals_left_at_zoo_l266_266456


namespace both_square_and_cube_count_both_square_cube_less_than_1000_l266_266784

theorem both_square_and_cube (n : ℕ) (h_positive : n > 0) (h_bound : n < 1000) (h_square : ∃ k : ℕ, n = k^2) (h_cube : ∃ k : ℕ, n = k^3) :
  n ∈ {1, 64, 729} :=
sorry

theorem count_both_square_cube_less_than_1000 :
  {n : ℕ | n > 0 ∧ n < 1000 ∧ ∃ k : ℕ, n = k^2 ∧ ∃ k : ℕ, n = k^3}.to_finset.card = 3 :=
by
  have h_valid : ∀ n ∈ {1, 64, 729}, n > 0 ∧ n < 1000 ∧ (∃ k : ℕ, n = k^2) ∧ (∃ k : ℕ, n = k^3) := sorry
  have h_reverse : ∀ n, n > 0 ∧ n < 1000 ∧ (∃ k : ℕ, n = k^2) ∧ (∃ k : ℕ, n = k^3) → n ∈ {1, 64, 729} := sorry
  exact Finset.card_map _ {1, 64, 729}.to_finset
  sorry

end both_square_and_cube_count_both_square_cube_less_than_1000_l266_266784


namespace apples_and_oranges_l266_266010

theorem apples_and_oranges :
  ∃ x y : ℝ, 2 * x + 3 * y = 6 ∧ 4 * x + 7 * y = 13 ∧ (16 * x + 23 * y = 47) :=
by
  sorry

end apples_and_oranges_l266_266010


namespace bounded_g_of_f_l266_266231

theorem bounded_g_of_f
  (f g : ℝ → ℝ)
  (h1 : ∀ x y, f (x + y) + f (x - y) = 2 * f x * g y)
  (h2 : ∃ x, f x ≠ 0)
  (h3 : ∀ x, |f x| ≤ 1) :
  ∀ y, |g y| ≤ 1 := 
sorry

end bounded_g_of_f_l266_266231


namespace prob_yellow_is_3_over_5_required_red_balls_is_8_l266_266886

-- Defining the initial conditions
def total_balls : ℕ := 10
def red_balls : ℕ := 4
def yellow_balls : ℕ := 6

-- Part 1: Prove the probability of drawing a yellow ball is 3/5
theorem prob_yellow_is_3_over_5 :
  (yellow_balls : ℚ) / (total_balls : ℚ) = 3 / 5 := sorry

-- Part 2: Prove that adding 8 red balls makes the probability of drawing a red ball 2/3
theorem required_red_balls_is_8 (x : ℕ) :
  (red_balls + x : ℚ) / (total_balls + x : ℚ) = 2 / 3 → x = 8 := sorry

end prob_yellow_is_3_over_5_required_red_balls_is_8_l266_266886


namespace position_of_R_in_RHOMBUS_l266_266000

theorem position_of_R_in_RHOMBUS (total_spaces : ℕ) (word_length : ℕ) (empty_spaces : ℕ) (equal_sides : empty_spaces * 2 = total_spaces - word_length)
  (total_spaces = 31) (word_length = 7) : 12 + 1 = 13 :=
by
  sorry

end position_of_R_in_RHOMBUS_l266_266000


namespace weighted_average_yield_computation_overall_quote_computation_l266_266071

-- Conditions
variables (proportion_A proportion_B proportion_C : ℝ)
variables (investment : ℝ := 1000)
variables (yield_A : ℝ := 0.25) (quote_A : ℝ := 500)
variables (yield_B : ℝ := 0.12) (quote_B : ℝ := 300)
variables (yield_C : ℝ := 0.08) (quote_C : ℝ := 200)

-- Proportions should sum up to 1
axiom proportions_sum : proportion_A + proportion_B + proportion_C = 1

-- Weighted average yield and overall quote
noncomputable def weighted_average_yield : ℝ :=
  proportion_A * yield_A + proportion_B * yield_B + proportion_C * yield_C

noncomputable def overall_quote : ℝ := 
  proportion_A * quote_A + proportion_B * quote_B + proportion_C * quote_C

-- Proof statements
theorem weighted_average_yield_computation :
  proportions_sum → weighted_average_yield proportion_A proportion_B proportion_C = sorry :=
by sorry

theorem overall_quote_computation :
  proportions_sum → overall_quote proportion_A proportion_B proportion_C = $1000 :=
by sorry

end weighted_average_yield_computation_overall_quote_computation_l266_266071


namespace find_angle_XWY_l266_266206

noncomputable def triangle_angles (XYZ_angle YXZ_angle XZY_angle : ℝ) : Prop :=
  XYZ_angle + YXZ_angle + XZY_angle = 180

noncomputable def is_orthocenter (W X Y Z : Type*) (K L : Type*) : Prop :=
  ∠XWZ = 90 ∧ ∠YWX = 90

noncomputable def quadrilateral_angle_sum (XWYL_angle XKW_angle YLW_angle YXZ_angle : ℝ) : Prop :=
  XWYL_angle + XKW_angle + YLW_angle + YXZ_angle = 360

theorem find_angle_XWY 
  (XYZ_angle : ℝ)
  (XZY_angle : ℝ)
  (angle_XWYL : ℝ)
  (K L W X Y Z : Type*)
  (h_triangle : triangle_angles XYZ_angle (180 - XYZ_angle - XZY_angle) XZY_angle)
  (h_orthocenter : is_orthocenter W X Y Z K L)
  (h_quadrilateral : quadrilateral_angle_sum angle_XWYL 90 90 (180 - XYZ_angle - XZY_angle))
  (h_XYZ_angle : XYZ_angle = 30)
  (h_XZY_angle : XZY_angle = 80) :
  angle_XWYL = 110 :=
by
  sorry

end find_angle_XWY_l266_266206


namespace math_bonanza_2022_ir14_l266_266222

theorem math_bonanza_2022_ir14 :
  ∀ (A B C H H' M : ℝ) (AB AC BC : ℝ) 
    (mid_M : M = (B + C) / 2)
    (H' : H' = 2 * M - H),
    -- Given side lengths:
    AB = 6 →
    AC = 7 →
    BC = 8 →
    -- H is the orthocenter (conditionally given but needs to be defined within defining orthocenter properties)
    -- H' is the reflection of H across midpoint of BC
    let p := 132 in
    let q := 119 in
    p + q = 251 := 
sorry

end math_bonanza_2022_ir14_l266_266222


namespace coefficient_of_x2y6_l266_266900

theorem coefficient_of_x2y6 :
  let binom_coeff8 (r : ℕ) : ℕ := Nat.choose 8 r
  ((1 - y / x) * (x + y)^8).coeff (2, 6) = -28 :=
by
  sorry

end coefficient_of_x2y6_l266_266900


namespace smallest_divisible_fraction_l266_266095

theorem smallest_divisible_fraction :
  let nums := [8, 7, 15]
  let dens := [33, 22, 26]
  let lcm_nums := 120 -- LCM of the numerators
  let gcd_dens := 1   -- GCD of the denominators
  (forall f ∈ nums, lcm_nums % f = 0) ∧ (forall d ∈ dens, d % gcd_dens = 0) ->
  (lcm_nums : ℚ) / gcd_dens = 120 :=
by
  sorry

end smallest_divisible_fraction_l266_266095


namespace sam_has_45_nickels_l266_266996

def initial_nickels : ℕ := 29
def nickels_added_by_dad : ℕ := 24
def nickels_taken_by_dad : ℕ := 13
def additional_nickels_percent : ℝ := 0.20

def total_nickels (initial : ℕ) (added : ℕ) (taken : ℕ) (percent : ℝ) : ℕ :=
  initial + added - taken + nat.floor (percent * added)

theorem sam_has_45_nickels :
  total_nickels initial_nickels nickels_added_by_dad nickels_taken_by_dad additional_nickels_percent = 45 :=
by
  sorry

end sam_has_45_nickels_l266_266996


namespace value_of_X_is_one_l266_266518

-- Problem: Given the numbers 28 at the start of a row, 17 in the middle, and -15 in the same column as X,
-- we show the value of X must be 1 because the sequences are arithmetic.

theorem value_of_X_is_one (d : ℤ) (X : ℤ) :
  -- Conditions
  (17 - X = d) ∧ 
  (X - (-15) = d) ∧ 
  (d = 16) →
  -- Conclusion: X must be 1
  X = 1 :=
by 
  sorry

end value_of_X_is_one_l266_266518


namespace number_of_sixth_powers_lt_1000_l266_266691

theorem number_of_sixth_powers_lt_1000 : 
  {n : ℕ | ∃ k : ℕ, k^6 = n ∧ n < 1000}.to_finset.card = 3 :=
by
  sorry

end number_of_sixth_powers_lt_1000_l266_266691


namespace maximum_of_expression_l266_266547

theorem maximum_of_expression : 
  ∀ (a b c d : ℕ), 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  a ∈ {1, 2, 3, 4, 5, 6} ∧ b ∈ {1, 2, 3, 4, 5, 6} ∧ 
  c ∈ {1, 2, 3, 4, 5, 6} ∧ d ∈ {1, 2, 3, 4, 5, 6} →
  a / (b / (c * d)) ≤ 120 :=
by
  sorry

end maximum_of_expression_l266_266547


namespace count_paths_A_to_B_l266_266611

-- Define the points as constants
constant A B C D E F G : Type

-- Define the segments of the figure
constant segments : list (Type × Type) :=
  [(A, C), (C, B), (A, D), (D, C), (D, F), (F, C), 
   (D, E), (E, F), (A, F), (A, G), (G, F), (C, G), 
   (A, B), (A, E), (C, D), (A, D)]

-- Define a path with no revisits
structure Path (from to : Type) :=
  (points : list Type)
  (edges : list (Type × Type))
  (no_revisit : ∀ (p : Type), p ∈ points → points.count p = 1)
  (valid_edges : ∀ (e : Type × Type), e ∈ edges → e ∈ segments)

-- The statement to prove
theorem count_paths_A_to_B : 
  ∃ paths : list (Path A B), paths.length = 13 := 
sorry

end count_paths_A_to_B_l266_266611


namespace number_of_sixth_powers_lt_1000_l266_266692

theorem number_of_sixth_powers_lt_1000 : 
  {n : ℕ | ∃ k : ℕ, k^6 = n ∧ n < 1000}.to_finset.card = 3 :=
by
  sorry

end number_of_sixth_powers_lt_1000_l266_266692


namespace numPoorPeople_is_40_l266_266008

def numPoorPeople (totalMoney : ℝ) (x : ℝ) : ℝ := totalMoney / x

theorem numPoorPeople_is_40 (totalMoney : ℝ) (x : ℝ)
    (h1 : totalMoney = 120)
    (h2 : numPoorPeople totalMoney (x - 10) - numPoorPeople totalMoney x = 
          numPoorPeople totalMoney x - numPoorPeople totalMoney (x + 20)) :
  x = 40 :=
by
  rw [← h1] at *
  have h3 : 10 * x * (x + 20) = 20 * x * (x - 10), from sorry
  have h4 : 10 * (x*x + 20*x) = 20 * (x*x - 10*x), from sorry
  have h5 : 10x^2 + 200x = 20x^2 - 200x, from by rw h4 at h3; exact sorry
  have h6 : 0 = 10x^2 - 400x, from by rw [← sub_eq_zero] at h5; exact sorry
  have h7 : 0 = 10x(x - 40), from by rw [mul_sub] at h6; exact sorry
  cases eq_zero_or_eq_zero_of_mul_eq_zero h7
  case inl h8 => exact absurd h8 sorry
  case inr h9 => exact h9

end numPoorPeople_is_40_l266_266008


namespace count_squares_and_cubes_less_than_1000_l266_266725

theorem count_squares_and_cubes_less_than_1000 :
  {n : ℕ | 0 < n ∧ n < 1000 ∧ ∃ x, n = x ^ 6}.to_finset.card = 3 :=
by sorry

end count_squares_and_cubes_less_than_1000_l266_266725


namespace compute_9_times_one_seventh_pow_4_l266_266505

theorem compute_9_times_one_seventh_pow_4 :
  9 * (1 / 7) ^ 4 = 9 / 2401 :=
by
  -- The actual proof would go here.
  sorry

end compute_9_times_one_seventh_pow_4_l266_266505


namespace coeff_x2y6_in_expansion_l266_266903

theorem coeff_x2y6_in_expansion : 
  (coeff_of_term (x := 2) (y := 6)) ((1 - (y / x)) * (x + y) ^ 8) = -28 :=
sorry

end coeff_x2y6_in_expansion_l266_266903


namespace line_through_center_perpendicular_l266_266290

theorem line_through_center_perpendicular (x y : ℝ) :
  let C := (-1, 0)
  let circle_eq := x^2 + 2*x + y^2 = 0
  let perp_line := x + y = 0
  let line_eq := x - y + 1 = 0
  (C.1, C.2) = (-1, 0) →
  ∃ (m : ℝ), m = 1 ∧ line_eq = 0 :=
sorry

end line_through_center_perpendicular_l266_266290


namespace count_squares_and_cubes_l266_266653

theorem count_squares_and_cubes (bound : ℕ) (hk : bound = 1000) : 
  Nat.card {n : ℕ | n < bound ∧ (∃ k : ℕ, n = k ^ 6)} = 3 := 
by
  sorry

end count_squares_and_cubes_l266_266653


namespace price_after_reductions_l266_266422

theorem price_after_reductions (P : ℝ) : ((P * 0.85) * 0.90) = P * 0.765 :=
by sorry

end price_after_reductions_l266_266422


namespace modulus_of_complex_example_l266_266138

def complex_modulus (z : Complex) : ℝ :=
  Complex.abs z

theorem modulus_of_complex_example : complex_modulus (1/2 + 5/2 * Complex.i) = (Real.sqrt 26) / 2 := by
  sorry

end modulus_of_complex_example_l266_266138


namespace sum_of_consecutive_integers_l266_266364

theorem sum_of_consecutive_integers (n : ℕ) (h : n * (n + 1) / 2 ≤ 500) :
  n = 31 ∨ ((32 * (32 + 1) / 2) ≤ 500) := 
sorry

end sum_of_consecutive_integers_l266_266364


namespace distance_between_points_l266_266351

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem distance_between_points :
  distance (1, 2) (5, 6) = 4 * real.sqrt 2 :=
by
  sorry

end distance_between_points_l266_266351


namespace coefficient_x2_y6_in_expansion_l266_266928

theorem coefficient_x2_y6_in_expansion :
  ∀ (x y : ℝ), x ≠ 0 → 
    (let c8_6 := (nat.factorial 8) / ((nat.factorial 6) * (nat.factorial 2)),
         c8_5 := (nat.factorial 8) / ((nat.factorial 5) * (nat.factorial 3)) in
     c8_6 - c8_5 = -28) :=
by
  intros x y hx_ne_zero
  let c8_6 := (nat.factorial 8) / ((nat.factorial 6) * (nat.factorial 2))
  let c8_5 := (nat.factorial 8) / ((nat.factorial 5) * (nat.factorial 3))
  have h : c8_6 - c8_5 = -28 := sorry
  exact h

end coefficient_x2_y6_in_expansion_l266_266928


namespace degrees_to_radians_300_l266_266074

theorem degrees_to_radians_300:
  (300 * (Real.pi / 180) = 5 * Real.pi / 3) := 
by
  repeat { sorry }

end degrees_to_radians_300_l266_266074


namespace number_of_correct_statements_l266_266292

-- Definitions of statements
def statement1 : Prop := Linear regression analysis is the mathematical method of finding a straight line that fits close to these sample points from the sample points.
def statement2 : Prop := The scatter plot of sample points can be used to intuitively judge whether the relationship between two variables can be represented by a linear relationship.
def statement3 : Prop := Through the regression equation \(\widehat{y} = \widehat{b}x + \widehat{a}\), it is possible to estimate the values and trends of observed variables.
def statement4 : Prop := Since a linear regression equation can be obtained from any set of observed values, there is no need to perform a correlation test.

theorem number_of_correct_statements : (statement1 = true) → (statement2 = true) → (statement3 = true) → (statement4 = false) → (3 = 3) :=
by {
  intros h1 h2 h3 h4,
  have : 3 = 3 := rfl,
  exact this,
}

end number_of_correct_statements_l266_266292


namespace max_consecutive_integers_sum_lt_500_l266_266370

theorem max_consecutive_integers_sum_lt_500 
  (n : ℕ)
  (h₁ : ∑ k in Finset.range (n+1), k = n * (n + 1) / 2)
  (h₂ : n * (n + 1) / 2 < 500) 
  (h₃ : (n + 1) * (n + 2) / 2 > 500)
  : n = 31 := 
by sorry

end max_consecutive_integers_sum_lt_500_l266_266370


namespace find_other_diagonal_l266_266280

-- Define the necessary conditions and parameters
variables (d1 : ℝ) (d2 : ℝ) (area : ℝ)

-- Given conditions
def rhombus_area (d1 d2 : ℝ) : ℝ := (d1 * d2) / 2

theorem find_other_diagonal 
  (h_area : area = 80)
  (h_d1 : d1 = 16)
  (h_rhombus : rhombus_area d1 d2 = area) : 
  d2 = 10 :=
by
  sorry

end find_other_diagonal_l266_266280


namespace count_squares_and_cubes_less_than_1000_l266_266729

theorem count_squares_and_cubes_less_than_1000 :
  {n : ℕ | 0 < n ∧ n < 1000 ∧ ∃ x, n = x ^ 6}.to_finset.card = 3 :=
by sorry

end count_squares_and_cubes_less_than_1000_l266_266729


namespace num_sixth_powers_below_1000_l266_266797

theorem num_sixth_powers_below_1000 : 
  {n : ℕ | n^6 < 1000 ∧ n > 0}.to_finset.card = 3 :=
by sorry

end num_sixth_powers_below_1000_l266_266797


namespace count_sixth_powers_less_than_1000_l266_266755

theorem count_sixth_powers_less_than_1000 : 
  {n : ℕ | 0 < n ∧ n < 1000 ∧ (∃ k : ℕ, n = k^6)}.to_finset.card = 3 :=
by 
  sorry

end count_sixth_powers_less_than_1000_l266_266755


namespace percentage_increase_in_surface_area_l266_266867

variable (a : ℝ)

theorem percentage_increase_in_surface_area (ha : a > 0) :
  let original_surface_area := 6 * a^2
  let new_edge_length := 1.5 * a
  let new_surface_area := 6 * (new_edge_length)^2
  let area_increase := new_surface_area - original_surface_area
  let percentage_increase := (area_increase / original_surface_area) * 100
  percentage_increase = 125 := 
by 
  let original_surface_area := 6 * a^2
  let new_edge_length := 1.5 * a
  let new_surface_area := 6 * (new_edge_length)^2
  let area_increase := new_surface_area - original_surface_area
  let percentage_increase := (area_increase / original_surface_area) * 100
  sorry

end percentage_increase_in_surface_area_l266_266867


namespace line_dividing_equal_areas_l266_266438

/-- 
Given a line starting from (c, 0) to (4, 4) in a 3x3 grid of unit squares (with the lower left
corner of the grid at the origin), this theorem states that the line divides the grid into two
regions with equal area if c = 1.75.
-/
theorem line_dividing_equal_areas (c : ℝ) :
  let total_area := 9 in
  let half_area := total_area / 2 in
  let triangle_area (c : ℝ) := (8 - 2 * c) in
  (triangle_area c = half_area) ↔ (c = 1.75) :=
sorry

end line_dividing_equal_areas_l266_266438


namespace initial_capacity_of_drum_x_l266_266079

theorem initial_capacity_of_drum_x (C x : ℝ) (h_capacity_y : 2 * x = 2 * 0.75 * C) :
  x = 0.75 * C :=
sorry

end initial_capacity_of_drum_x_l266_266079


namespace unique_sequence_exists_and_bounded_l266_266120

theorem unique_sequence_exists_and_bounded (a : ℝ) (n : ℕ) :
  ∃! (x : ℕ → ℝ), -- There exists a unique sequence x : ℕ → ℝ
    (x 1 = x (n - 1)) ∧ -- x_1 = x_{n-1}
    (∀ i, 1 ≤ i ∧ i ≤ n → (1 / 2) * (x (i - 1) + x i) = x i + x i ^ 3 - a ^ 3) ∧ -- Condition for all 1 ≤ i ≤ n
    (∀ i, 0 ≤ i ∧ i ≤ n + 1 → |x i| ≤ |a|) -- Bounding condition for all 0 ≤ i ≤ n + 1
:= sorry

end unique_sequence_exists_and_bounded_l266_266120


namespace count_correct_propositions_l266_266568

def line_parallel_plane (a : Line) (M : Plane) : Prop := sorry
def line_perpendicular_plane (a : Line) (M : Plane) : Prop := sorry
def line_parallel_line (a b : Line) : Prop := sorry
def line_perpendicular_line (a b : Line) : Prop := sorry
def plane_perpendicular_plane (M N : Plane) : Prop := sorry

theorem count_correct_propositions 
  (a b c : Line) 
  (M N : Plane) 
  (h1 : ¬ (line_parallel_plane a M ∧ line_parallel_plane b M → line_parallel_line a b)) 
  (h2 : line_parallel_plane a M ∧ line_perpendicular_plane b M → line_perpendicular_line b a) 
  (h3 : ¬ ((line_parallel_plane a M ∧ line_perpendicular_plane b M ∧ line_perpendicular_line c a ∧ line_perpendicular_line c b) → line_perpendicular_plane c M))
  (h4 : line_perpendicular_plane a M ∧ line_parallel_plane a N → plane_perpendicular_plane M N) :
  (0 + 1 + 0 + 1) = 2 :=
sorry

end count_correct_propositions_l266_266568


namespace count_positive_integers_square_and_cube_lt_1000_l266_266759

theorem count_positive_integers_square_and_cube_lt_1000 : 
  {n : ℕ | n < 1000 ∧ ∃ k : ℕ, n = k^6}.to_finset.card = 3 :=
by
  sorry

end count_positive_integers_square_and_cube_lt_1000_l266_266759


namespace integral_triangle_area_l266_266042

theorem integral_triangle_area
  (a b c : ℕ)
  (h_perimeter : a + b + c = 12)
  (h_triangle_ineq1 : a + b > c)
  (h_triangle_ineq2 : b + c > a)
  (h_triangle_ineq3 : c + a > b)
  (h_smallest_triplet : {a, b, c} = {3, 4, 5}) :
  (1/2) * (a + b + c) * sqrt((1/2 * (a + b + c) - a) * (1/2 * (a + b + c) - b) * (1/2 * (a + b + c) - c)) = 6 := 
sorry

end integral_triangle_area_l266_266042


namespace count_squares_cubes_less_than_1000_l266_266808

theorem count_squares_cubes_less_than_1000 : 
  {n : ℕ | ∃ x : ℕ, n = x^6 ∧ n < 1000}.card = 3 :=
sorry

end count_squares_cubes_less_than_1000_l266_266808


namespace positional_relationship_l266_266128

-- Define vector types
def Vector3 := (ℝ × ℝ × ℝ)

-- Define dot product for vectors
def dot_product (v1 v2 : Vector3) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

-- Given the normal vector of the plane α
def mu : Vector3 := (2, 2, -1)

-- Given the direction vector of the line l
def a : Vector3 := (-3, 4, 2)

-- Prove the positional relationship between the line l and the plane α
theorem positional_relationship : dot_product mu a = 0 → 
  (l_parallel_to_or_in_plane : (true)) :=
by
  sorry

end positional_relationship_l266_266128


namespace find_other_endpoint_l266_266316

theorem find_other_endpoint (x_m y_m : ℤ) (x1 y1 : ℤ) 
(m_cond : x_m = (x1 + (-1)) / 2) (m_cond' : y_m = (y1 + (-4)) / 2) : 
(x_m, y_m) = (3, -1) ∧ (x1, y1) = (7, 2) → (-1, -4) = (-1, -4) :=
by
  sorry

end find_other_endpoint_l266_266316


namespace solve_equation_l266_266268

variable (x : ℝ)

def equation := (x / (2 * x - 3)) + (5 / (3 - 2 * x)) = 4
def condition := x ≠ 3 / 2

theorem solve_equation : equation x ∧ condition x → x = 1 :=
by
  sorry

end solve_equation_l266_266268


namespace problem_solution_l266_266957

noncomputable def T := {x : ℝ // x > 0}

def g (x : T) : ℝ :=
  sorry -- definition derived in solution

theorem problem_solution : 
  (∃ m t : ℝ, m * t = 3010 / 3) :=
by 
  let g : T → ℝ := λ x, 1003 + (1/x.1)
  have g_eq : ∀ x y : T, g x * g y = g ⟨x.1 * y.1, mul_pos x.2 y.2⟩ + 1003 * ((1/x.1) + (1/y.1) + 1002),
  { sorry }, -- Verification of function g as per given condition
  let m := 1, 
  let t := 3010/3, 
  use m, t,
  have h_mt : m * t = 3010/3 := 
  by 
    calc 
      m * t = 1 * (3010 / 3) : by rw mul_one
      ... = 3010 / 3 : by norm_num,
  exact h_mt

end problem_solution_l266_266957


namespace number_of_sixth_powers_less_than_1000_l266_266644

theorem number_of_sixth_powers_less_than_1000 : 
  {n : ℕ | n < 1000 ∧ ∃ (k : ℕ), n = k^6}.finite.card = 3 :=
by sorry

end number_of_sixth_powers_less_than_1000_l266_266644


namespace sale_price_relationship_l266_266529

/-- Elaine's Gift Shop increased the original prices of all items by 10% 
  and then offered a 30% discount on these new prices in a clearance sale 
  - proving the relationship between the final sale price and the original price of an item -/

theorem sale_price_relationship (p : ℝ) : 
  (0.7 * (1.1 * p) = 0.77 * p) :=
by 
  sorry

end sale_price_relationship_l266_266529


namespace jasons_shelves_l266_266212

theorem jasons_shelves (total_books : ℕ) (number_of_shelves : ℕ) (h_total_books : total_books = 315) (h_number_of_shelves : number_of_shelves = 7) : (total_books / number_of_shelves) = 45 := 
by
  sorry

end jasons_shelves_l266_266212


namespace jose_julia_completion_time_l266_266415

variable (J N L : ℝ)

theorem jose_julia_completion_time :
  J + N + L = 1/4 ∧
  J * (1/3) = 1/18 ∧
  N = 1/9 ∧
  L * (1/3) = 1/18 →
  1/J = 6 ∧ 1/L = 6 := sorry

end jose_julia_completion_time_l266_266415


namespace rational_root_count_l266_266070

theorem rational_root_count (b3 b2 b1 : ℤ) : 
  let p : RatlPoly := 12 * x^4 + b3 * x^3 + b2 * x^2 + b1 * x + 8 
  in count_unique_rational_roots(p) = 22 := sorry

end rational_root_count_l266_266070


namespace compound_interest_amount_l266_266868

variable (P r t SI : ℝ)

theorem compound_interest_amount :
  t = 4 → 
  r = 0.10 → 
  SI = 1200 → 
  ((SI = P * r * t) → 
  (let CI := P * (1 + r)^t - P in CI = 1392.3)) :=
by
  intros ht hr hsi h_simple_interest
  let P := 1200 / (0.4 : ℝ) -- From the step P = 1200 / 0.4
  let CI := P * (1 + r) ^ t - P
  sorry -- The rest of the proof is omitted

end compound_interest_amount_l266_266868


namespace simplify_expression_l266_266999

open Real

-- Define the given expression as a function of x
noncomputable def given_expression (x : ℝ) (hx : 0 < x) : ℝ :=
  sqrt (2 * (1 + sqrt (1 + ( (x^4 - 1) / (2 * x^2) )^2)))

-- Define the expected simplified expression
noncomputable def expected_expression (x : ℝ) (hx : 0 < x) : ℝ :=
  (x^2 + 1) / x

-- Proof statement to verify the simplification
theorem simplify_expression (x : ℝ) (hx : 0 < x) :
  given_expression x hx = expected_expression x hx :=
sorry

end simplify_expression_l266_266999


namespace count_squares_cubes_less_than_1000_l266_266818

theorem count_squares_cubes_less_than_1000 : 
  {n : ℕ | ∃ x : ℕ, n = x^6 ∧ n < 1000}.card = 3 :=
sorry

end count_squares_cubes_less_than_1000_l266_266818


namespace arithmetic_sequence_property_l266_266890

theorem arithmetic_sequence_property (a : ℕ → ℝ) (d a1 : ℝ) 
  [h_arith: ∀ (n : ℕ), a n = a1 + n * d]
  (h_sum : a 2 + a 4 + a 6 + a 8 + a 10 = 80) :
  a 7 - (1 / 2) * a 8 = 8 :=
by
  sorry

end arithmetic_sequence_property_l266_266890


namespace find_B_l266_266203

noncomputable def B2 (B : ℕ) : ℕ := 10 * B + 2
noncomputable def B7 (B : ℕ) : ℕ := 70 + B

theorem find_B (B : ℕ) : (B2 B) * (B7 B) = 6396 → B = 8 :=
by
  assume h : (B2 B) * (B7 B) = 6396
  sorry

end find_B_l266_266203


namespace max_value_expression_l266_266110

theorem max_value_expression (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 2 * a + 3 * b = 5) : 
  (∀ x y : ℝ, x = 2 * a + 2 → y = 3 * b + 1 → x * y ≤ 16) := by
  sorry

end max_value_expression_l266_266110


namespace number_of_squares_and_cubes_less_than_1000_l266_266712

noncomputable def count_squares_and_cubes : ℕ :=
  (List.range 1000).countp (λ n : ℕ, (∃ k : ℕ, k^6 = n))

theorem number_of_squares_and_cubes_less_than_1000 :
  count_squares_and_cubes = 3 :=
  by
    sorry

end number_of_squares_and_cubes_less_than_1000_l266_266712


namespace number_of_sixth_powers_lt_1000_l266_266701

-- Define the condition for being a sixth power and being less than 1000
def isSixthPower (n : ℕ) : Prop := ∃ k : ℕ, n = k^6

-- Define the main statement to be proved
theorem number_of_sixth_powers_lt_1000 : 
  {n : ℕ | n < 1000 ∧ isSixthPower n}.to_finset.card = 3 :=
by
  sorry

end number_of_sixth_powers_lt_1000_l266_266701


namespace max_consecutive_sum_less_500_l266_266358

theorem max_consecutive_sum_less_500 : 
  (argmax (λ n : ℕ, n * (n + 1) / 2 < 500) = 31) :=
by sorry

end max_consecutive_sum_less_500_l266_266358


namespace tan_2alpha_value_beta_value_l266_266553

variable (α β : ℝ)
variable (h1 : 0 < β ∧ β < α ∧ α < π / 2)
variable (h2 : Real.cos α = 1 / 7)
variable (h3 : Real.cos (α - β) = 13 / 14)

theorem tan_2alpha_value : Real.tan (2 * α) = - (8 * Real.sqrt 3 / 47) :=
by
  sorry

theorem beta_value : β = π / 3 :=
by
  sorry

end tan_2alpha_value_beta_value_l266_266553


namespace evaluate_expression_l266_266082

theorem evaluate_expression (x b : ℝ) (h : x = b + 4) : 2 * x - b + 5 = b + 13 := by
  sorry

end evaluate_expression_l266_266082


namespace g_sum_eq_14_l266_266232

noncomputable def g (x : ℝ) : ℝ := d * x^8 + e * x^6 - f * x^4 + g * x^2 + 5

theorem g_sum_eq_14
  (d e f g : ℝ)
  (h1 : g 10 = 7) :
  g 10 + g (-10) = 14 :=
by sorry

end g_sum_eq_14_l266_266232


namespace count_squares_and_cubes_less_than_1000_l266_266734

theorem count_squares_and_cubes_less_than_1000 :
  {n : ℕ | 0 < n ∧ n < 1000 ∧ ∃ x, n = x ^ 6}.to_finset.card = 3 :=
by sorry

end count_squares_and_cubes_less_than_1000_l266_266734


namespace both_square_and_cube_count_both_square_cube_less_than_1000_l266_266779

theorem both_square_and_cube (n : ℕ) (h_positive : n > 0) (h_bound : n < 1000) (h_square : ∃ k : ℕ, n = k^2) (h_cube : ∃ k : ℕ, n = k^3) :
  n ∈ {1, 64, 729} :=
sorry

theorem count_both_square_cube_less_than_1000 :
  {n : ℕ | n > 0 ∧ n < 1000 ∧ ∃ k : ℕ, n = k^2 ∧ ∃ k : ℕ, n = k^3}.to_finset.card = 3 :=
by
  have h_valid : ∀ n ∈ {1, 64, 729}, n > 0 ∧ n < 1000 ∧ (∃ k : ℕ, n = k^2) ∧ (∃ k : ℕ, n = k^3) := sorry
  have h_reverse : ∀ n, n > 0 ∧ n < 1000 ∧ (∃ k : ℕ, n = k^2) ∧ (∃ k : ℕ, n = k^3) → n ∈ {1, 64, 729} := sorry
  exact Finset.card_map _ {1, 64, 729}.to_finset
  sorry

end both_square_and_cube_count_both_square_cube_less_than_1000_l266_266779


namespace count_sixth_powers_less_than_1000_l266_266745

theorem count_sixth_powers_less_than_1000 : 
  {n : ℕ | 0 < n ∧ n < 1000 ∧ (∃ k : ℕ, n = k^6)}.to_finset.card = 3 :=
by 
  sorry

end count_sixth_powers_less_than_1000_l266_266745


namespace count_squares_and_cubes_less_than_1000_l266_266736

theorem count_squares_and_cubes_less_than_1000 :
  {n : ℕ | 0 < n ∧ n < 1000 ∧ ∃ x, n = x ^ 6}.to_finset.card = 3 :=
by sorry

end count_squares_and_cubes_less_than_1000_l266_266736


namespace min_value_of_function_l266_266578

theorem min_value_of_function (x : ℝ) (h : x > 2) : ∃ y, y = (x^2 - 4*x + 8) / (x - 2) ∧ (∀ z, z = (x^2 - 4*x + 8) / (x - 2) → y ≤ z) :=
sorry

end min_value_of_function_l266_266578


namespace number_of_squares_and_cubes_less_than_1000_l266_266715

noncomputable def count_squares_and_cubes : ℕ :=
  (List.range 1000).countp (λ n : ℕ, (∃ k : ℕ, k^6 = n))

theorem number_of_squares_and_cubes_less_than_1000 :
  count_squares_and_cubes = 3 :=
  by
    sorry

end number_of_squares_and_cubes_less_than_1000_l266_266715


namespace lacsap_problem_l266_266055

/-
We are given:
Emily (E) is one of the doctors,
Robert (R) is one of the nurses,
Without Emily, there are 5 doctors,
Without Robert, there are 3 nurses.

We need to prove that with d = 6 (number of doctors excluding Robert)
and n = 2 (number of nurses excluding Emily and Robert),
the product d * n = 12.
-/

noncomputable def doctors := {E, D1, D2, D3, D4, D5}
noncomputable def nurses := {R, N1, N2, N3}
def d := 6  -- number of doctors including Emily, excluding Robert.
def n := 2  -- number of nurses excluding both Emily and Robert.

theorem lacsap_problem : d * n = 12 :=
by
  sorry

end lacsap_problem_l266_266055


namespace number_of_outliers_l266_266586

def dataset := [4, 10, 22, 22, 26, 28, 28, 30, 37, 46]
def Q2 := 27
def Q1 := 22
def Q3 := 30

def IQR := Q3 - Q1
def lower_threshold := Q1 - 1.5 * IQR
def upper_threshold := Q3 + 1.5 * IQR

def is_outlier (x : ℕ) : Prop :=
  x < lower_threshold ∨ x > upper_threshold

def outliers := dataset.filter is_outlier

theorem number_of_outliers : outliers.length = 2 :=
by
  sorry

end number_of_outliers_l266_266586


namespace number_of_sixth_powers_lt_1000_l266_266690

theorem number_of_sixth_powers_lt_1000 : 
  {n : ℕ | ∃ k : ℕ, k^6 = n ∧ n < 1000}.to_finset.card = 3 :=
by
  sorry

end number_of_sixth_powers_lt_1000_l266_266690


namespace zoo_individuals_left_l266_266458

/-!
A fifth-grade class went on a field trip to the zoo, and their class of 10 students merged with another class with the same number of students.
5 parents offered to be a chaperone, and 2 teachers from both classes will be there too.
When the school day was over, 10 of the students left. Two of the chaperones, who were parents in that group, also left.
-/

theorem zoo_individuals_left (students_per_class chaperones teachers students_left chaperones_left : ℕ)
  (h1 : students_per_class = 10)
  (h2 : chaperones = 5)
  (h3 : teachers = 2)
  (h4 : students_left = 10)
  (h5 : chaperones_left = 2) : 
  let total_students := students_per_class * 2,
      total_initial := total_students + chaperones + teachers,
      total_remaining := total_initial - students_left - chaperones_left
  in
  total_remaining = 15 := by
  sorry

end zoo_individuals_left_l266_266458


namespace number_of_sixth_powers_less_than_1000_l266_266842

theorem number_of_sixth_powers_less_than_1000 : 
  {n : ℕ | ∃ m : ℕ, n = m^6 ∧ n < 1000}.to_finset.card = 3 := 
by {
  sorry
}

end number_of_sixth_powers_less_than_1000_l266_266842


namespace number_of_sixth_powers_less_than_1000_l266_266844

theorem number_of_sixth_powers_less_than_1000 : 
  {n : ℕ | ∃ m : ℕ, n = m^6 ∧ n < 1000}.to_finset.card = 3 := 
by {
  sorry
}

end number_of_sixth_powers_less_than_1000_l266_266844


namespace find_fx_l266_266229

-- Define the sine and cosine of alpha
variable (α : ℝ) (f : ℝ → ℝ)
hypothesis sin_alpha : Real.sin α = Real.sqrt 5 / 5

-- Define properties of the function f
hypothesis odd_fn : ∀ x, f (-x) = -f x
hypothesis periodic_fn : ∀ x, f (x + 2) = f x

-- Given specific value of f at -2/5
hypothesis f_neg_two_fifths : f (-2 / 5) = 3

-- Theorem statement
theorem find_fx : f (4 * Real.cos (2 * α)) = -3 := by
  sorry

end find_fx_l266_266229


namespace mysterious_division_l266_266982

theorem mysterious_division (d : ℕ) : (8 * d < 1000) ∧ (7 * d < 900) → d = 124 :=
by
  intro h
  sorry

end mysterious_division_l266_266982


namespace A_pos_l266_266527

def A : ℝ :=
(Σ i in finset.range 2016, (if (i % 4 = 0 ∨ i % 4 = 3) then 1 else -1) * (1 / (i + 1)))

theorem A_pos : A > 0 :=
sorry

end A_pos_l266_266527


namespace students_in_class_l266_266980

theorem students_in_class (g b : ℕ) (total_jellybeans leftover_jellybeans : ℕ)
  (h1 : total_jellybeans = 420)
  (h2 : leftover_jellybeans = 18)
  (h3 : b = g + 3)
  (h4 : g * g + b * b = total_jellybeans - leftover_jellybeans) :
  g + b = 29 :=
by {
  have h5 : 2 * g * g + 6 * g - 393 = 0,
  { 
    simp at h4,
    rw h3 at h4,
    linarith,
  },
  have h_g : g = 13,
  {
    sorry,
  },
  have h_b : b = 16,
  {
    rw h_g,
    linarith,
  },
  exact h_g + h_b
}

end students_in_class_l266_266980


namespace number_of_squares_and_cubes_less_than_1000_l266_266723

noncomputable def count_squares_and_cubes : ℕ :=
  (List.range 1000).countp (λ n : ℕ, (∃ k : ℕ, k^6 = n))

theorem number_of_squares_and_cubes_less_than_1000 :
  count_squares_and_cubes = 3 :=
  by
    sorry

end number_of_squares_and_cubes_less_than_1000_l266_266723


namespace relationship_among_abc_l266_266961

noncomputable def a : ℝ := 0.7 ^ 0.4
noncomputable def b : ℝ := 0.4 ^ 0.7
noncomputable def c : ℝ := 0.4 ^ 0.4

theorem relationship_among_abc : b < c ∧ c < a :=
by
  -- Proof needed here
  sorry

end relationship_among_abc_l266_266961


namespace count_sixth_powers_below_1000_l266_266831

-- Definition and necessary condition for perfect sixth power
def is_sixth_power (n : ℕ) : Prop := ∃ k : ℕ, n = k ^ 6

-- Main statement of the problem
theorem count_sixth_powers_below_1000 :
  {n : ℕ | n < 1000 ∧ is_sixth_power n}.to_finset.card = 3 := 
sorry

end count_sixth_powers_below_1000_l266_266831


namespace largest_n_l266_266538

def a_n (n : ℕ) (d_a : ℤ) : ℤ := 1 + (n-1) * d_a
def b_n (n : ℕ) (d_b : ℤ) : ℤ := 3 + (n-1) * d_b

theorem largest_n (d_a d_b : ℤ) (n : ℕ) :
  (a_n n d_a * b_n n d_b = 2304 ∧ a_n 1 d_a = 1 ∧ b_n 1 d_b = 3) 
  → n ≤ 20 := 
sorry

end largest_n_l266_266538


namespace democrat_voters_for_A_l266_266190

theorem democrat_voters_for_A (V : ℝ) (hV_pos : V > 0) :
  let dem := 0.60 * V in
  let rep := 0.40 * V in
  let dem_votes_for_A := (d : ℝ) * dem in
  let rep_votes_for_A := 0.20 * rep in
  let total_votes_for_A := 0.47 * V in
  dem_votes_for_A + rep_votes_for_A = total_votes_for_A →
  d = 0.65 :=
by
  sorry

end democrat_voters_for_A_l266_266190


namespace coefficient_x2y6_l266_266921

/-- In the expansion of (1 - y / x) * (x + y) ^ 8, the coefficient of x ^ 2 * y ^ 6 is -28. -/
theorem coefficient_x2y6 (x y : ℚ) :  
  coefficient ((1 - y / x) * (x + y)^8) (monomial (2, 6)) = -28 :=
sorry

end coefficient_x2y6_l266_266921


namespace base7_arithmetic_l266_266540

theorem base7_arithmetic : 
  let b1000 := 343  -- corresponding to 1000_7 in decimal
  let b666 := 342   -- corresponding to 666_7 in decimal
  let b1234 := 466  -- corresponding to 1234_7 in decimal
  let s := b1000 + b666  -- sum in decimal
  let s_base7 := 1421    -- sum back in base7 (1421 corresponds to 685 in decimal)
  let r_base7 := 254     -- result from subtraction in base7 (254 corresponds to 172 in decimal)
  (1000 * 7^0 + 0 * 7^1 + 0 * 7^2 + 1 * 7^3) + (6 * 7^0 + 6 * 7^1 + 6 * 7^2) - (4 * 7^0 + 3 * 7^1 + 2 * 7^2 + 1 * 7^3) = (4 * 7^0 + 5 * 7^1 + 2 * 7^2)
  :=
sorry

end base7_arithmetic_l266_266540


namespace distance_between_foci_l266_266299

-- Define the equation of the hyperbola.
def hyperbola_eq (x y : ℝ) : Prop := x * y = 4

-- The coordinates of foci for hyperbola of the form x*y = 4
def foci_1 : (ℝ × ℝ) := (2, 2)
def foci_2 : (ℝ × ℝ) := (-2, -2)

-- Define the Euclidean distance function.
def euclidean_distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Prove that the distance between the foci is 4√2.
theorem distance_between_foci : euclidean_distance foci_1 foci_2 = 4 * real.sqrt 2 := sorry

end distance_between_foci_l266_266299


namespace rectangular_prism_width_l266_266273

variables (w : ℝ)

theorem rectangular_prism_width (h : ℝ) (l : ℝ) (d : ℝ) (hyp_l : l = 5) (hyp_h : h = 7) (hyp_d : d = 15) :
  w = Real.sqrt 151 :=
by 
  -- Proof goes here
  sorry

end rectangular_prism_width_l266_266273


namespace speed_ratio_l266_266188

-- Define the speeds of A and B
variables (v_A v_B : ℝ)

-- Assume the conditions of the problem
axiom h1 : 200 / v_A = 400 / v_B

-- Prove the ratio of the speeds
theorem speed_ratio : v_A / v_B = 1 / 2 :=
by
  sorry

end speed_ratio_l266_266188


namespace coeff_x2y6_in_expansion_l266_266906

theorem coeff_x2y6_in_expansion : 
  (coeff_of_term (x := 2) (y := 6)) ((1 - (y / x)) * (x + y) ^ 8) = -28 :=
sorry

end coeff_x2y6_in_expansion_l266_266906


namespace find_a_range_find_gm_max_l266_266588

def f (x a : ℝ) : ℝ := (x^2 - a + 1) * Real.exp x

def has_two_distinct_extreme_points (a : ℝ) : Prop := 
  let disc := (2:ℝ)^2 - 4 * 1 * (1 - a)
  a > 0 ∧ disc > 0

def extreme_point_relation (m n : ℝ) : Prop := 
  m < n ∧ (|m + n| + 1 ≥ |m * n|)

theorem find_a_range (a : ℝ) (h1 : has_two_distinct_extreme_points a) (m n : ℝ) 
    (h2 : extreme_point_relation m n) : 
  0 < a ∧ a ≤ 4 := 
sorry

theorem find_gm_max (m : ℝ) (h1 : a = m^2 + 2*m + 1) (h2 : -3 ≤ m ∧ m < -1) 
    (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 2) : 
  let y := m * f x a
  let g (m : ℝ) := y
  g 0 = 0 := 
sorry

end find_a_range_find_gm_max_l266_266588


namespace part1_part2_l266_266590

def f (a b x : ℝ) : ℝ := a * Real.log x + (b * (x + 1)) / x

theorem part1 (a b : ℝ) : 
  f a b 1 = 2 ∧ (a - b = 0) :=
by
  sorry

theorem part2 (a b : ℝ) (k : ℝ) (h : a = 1) (h' : b = 1) :
  (∀ x > 1, f a b x > ((x - k) * Real.log x) / (x - 1)) ↔ k ∈ set.Ici (-1) :=
by
  sorry

end part1_part2_l266_266590


namespace coefficient_x2y6_l266_266923

/-- In the expansion of (1 - y / x) * (x + y) ^ 8, the coefficient of x ^ 2 * y ^ 6 is -28. -/
theorem coefficient_x2y6 (x y : ℚ) :  
  coefficient ((1 - y / x) * (x + y)^8) (monomial (2, 6)) = -28 :=
sorry

end coefficient_x2y6_l266_266923


namespace max_consecutive_sum_l266_266390

theorem max_consecutive_sum : ∃ (n : ℕ), (∀ m : ℕ, (m < n → (m * (m + 1)) / 2 < 500)) ∧ ¬((n * (n + 1)) / 2 < 500) :=
by {
  sorry
}

end max_consecutive_sum_l266_266390


namespace number_of_squares_and_cubes_less_than_1000_l266_266711

noncomputable def count_squares_and_cubes : ℕ :=
  (List.range 1000).countp (λ n : ℕ, (∃ k : ℕ, k^6 = n))

theorem number_of_squares_and_cubes_less_than_1000 :
  count_squares_and_cubes = 3 :=
  by
    sorry

end number_of_squares_and_cubes_less_than_1000_l266_266711


namespace regular_octagon_inscribed_circle_area_bounded_l266_266032

noncomputable def area_of_bounded_region (s : ℝ) (r : ℝ) (num_arcs : ℕ) : ℝ :=
  let area_octagon := 2 * (1 + real.sqrt 2) * s^2
  let angle := 45.0/360.0 * real.pi
  let sector_area := (angle / (2 * real.pi)) * r^2
  let triangle_area := real.sqrt 2 / 4 * s^2
  let reflected_arc_area := sector_area - triangle_area
  let total_reflected_arc_area := num_arcs * reflected_arc_area
  area_octagon - total_reflected_arc_area

theorem regular_octagon_inscribed_circle_area_bounded (s : ℝ) (r : ℝ) (num_arcs : ℕ) 
  (hs : s = 1) (hr : r = 2 + real.sqrt 2) (hn : num_arcs = 8) :
  area_of_bounded_region s r num_arcs = 4 + 2 * real.sqrt 2 - real.pi * (6 + 4 * real.sqrt 2) :=
begin
  -- proof skipped
  sorry
end

end regular_octagon_inscribed_circle_area_bounded_l266_266032


namespace num_sixth_powers_below_1000_l266_266793

theorem num_sixth_powers_below_1000 : 
  {n : ℕ | n^6 < 1000 ∧ n > 0}.to_finset.card = 3 :=
by sorry

end num_sixth_powers_below_1000_l266_266793


namespace sector_radius_l266_266279

theorem sector_radius (l : ℝ) (a : ℝ) (r : ℝ) (h1 : l = 2) (h2 : a = 4) (h3 : a = (1 / 2) * l * r) : r = 4 := by
  sorry

end sector_radius_l266_266279


namespace distance_between_foci_of_ellipse_l266_266339

-- Define the three given points
structure Point where
  x : ℝ
  y : ℝ

def p1 : Point := ⟨1, 3⟩
def p2 : Point := ⟨5, -1⟩
def p3 : Point := ⟨10, 3⟩

-- Define the statement that the distance between the foci of the ellipse they define is 2 * sqrt(4.25)
theorem distance_between_foci_of_ellipse : 
  ∃ (c : ℝ) (f : ℝ), f = 2 * Real.sqrt 4.25 ∧ 
  (∃ (ellipse : Point → Prop), ellipse p1 ∧ ellipse p2 ∧ ellipse p3) :=
sorry

end distance_between_foci_of_ellipse_l266_266339


namespace chord_perpendicular_intro_l266_266013

-- Definitions based on conditions:
def radius_K : ℝ := r
def radius_k : ℝ := r / 2
def y : ℝ := r / 8
def x0 : ℝ := 1 / 4 * r

theorem chord_perpendicular_intro 
  (r : ℝ)
  (radius_K := r)           -- Radius of the larger circle
  (radius_k := r / 2)       -- Radius of the smaller circle
  (y := r / 8)              -- Distance from the center of the larger circle to the chord
  (x0 := 1 / 4 * r)          -- Required distance pertaining to the chord conditions
  : 
  ∃ (chord : ℝ), 
    chord ∈ (set.range (λ r, 3 * x0)) ∧
    ∀ (x y : ℝ), 
      chord = radius_K * x 
      ∧ chord / 3 = (radius_K * (radius_k / (2 * radius_k - y)) ∧
          ∃ (x0: ℝ),
          x0 = 1 / 4 * radius_K  :=
begin
  obtain ⟨r, hr⟩ := by sorry,
  use (3/4) * r,
  exact ⟨r, hr, by sorry⟩
end

-- Noncomputable if necessary, depending on calculations for y and chord length

end chord_perpendicular_intro_l266_266013


namespace find_P_given_V_l266_266225

theorem find_P_given_V (h : ℕ) :
  (P = 3 * h * 6 + 5) → (41 = 3 * h * 6 + 5) → (P = 3 * h * 9 + 5) → (P = 59) :=
begin
  sorry
end

end find_P_given_V_l266_266225


namespace total_formula_portions_l266_266262

def puppies : ℕ := 7
def feedings_per_day : ℕ := 3
def days : ℕ := 5

theorem total_formula_portions : 
  (feedings_per_day * days * puppies = 105) := 
by
  sorry

end total_formula_portions_l266_266262


namespace minimum_value_of_omega_l266_266964

noncomputable theory

def f (ω : ℝ) (x : ℝ) : ℝ :=
  Real.cos (ω * x - π / 6)

theorem minimum_value_of_omega (ω : ℝ) :
  (ω > 0 ∧ ∀ x : ℝ, f ω x ≤ f ω (π / 4)) → ω = 2 / 3 :=
by
  sorry

end minimum_value_of_omega_l266_266964


namespace log_eq_neg_two_l266_266420

theorem log_eq_neg_two : 
  log (real.sndexp (log 8 (real.sndexp 1 1/2) - log 3 (real.sndexp 3 (1/2)))) (real.sqrt 3 / 3) = -2 := 
sorry

end log_eq_neg_two_l266_266420


namespace number_of_sixth_powers_lt_1000_l266_266702

-- Define the condition for being a sixth power and being less than 1000
def isSixthPower (n : ℕ) : Prop := ∃ k : ℕ, n = k^6

-- Define the main statement to be proved
theorem number_of_sixth_powers_lt_1000 : 
  {n : ℕ | n < 1000 ∧ isSixthPower n}.to_finset.card = 3 :=
by
  sorry

end number_of_sixth_powers_lt_1000_l266_266702


namespace c_investment_l266_266486

noncomputable def A : ℝ := 6300
noncomputable def B : ℝ := 4200
noncomputable def totalProfit : ℝ := 12200
noncomputable def AProfit : ℝ := 3660

theorem c_investment : ∃ C : ℝ, (6300 / (6300 + 4200 + C) = 3660 / 12200) ∧ C = 10490 := 
by {
  use 10490,
  rw (show A + B + 10490 = 6300 + 4200 + 10490, by { simp [A, B] }),
  use (A / (A + B + 10490) = AProfit / totalProfit),
  sorry
}

end c_investment_l266_266486


namespace max_consecutive_sum_less_500_l266_266357

theorem max_consecutive_sum_less_500 : 
  (argmax (λ n : ℕ, n * (n + 1) / 2 < 500) = 31) :=
by sorry

end max_consecutive_sum_less_500_l266_266357


namespace compute_fraction_power_l266_266508

theorem compute_fraction_power : 9 * (1 / 7)^4 = 9 / 2401 := by
  sorry

end compute_fraction_power_l266_266508


namespace count_squares_and_cubes_l266_266660

theorem count_squares_and_cubes (bound : ℕ) (hk : bound = 1000) : 
  Nat.card {n : ℕ | n < bound ∧ (∃ k : ℕ, n = k ^ 6)} = 3 := 
by
  sorry

end count_squares_and_cubes_l266_266660


namespace coefficient_x2_y6_in_expansion_l266_266930

theorem coefficient_x2_y6_in_expansion :
  ∀ (x y : ℝ), x ≠ 0 → 
    (let c8_6 := (nat.factorial 8) / ((nat.factorial 6) * (nat.factorial 2)),
         c8_5 := (nat.factorial 8) / ((nat.factorial 5) * (nat.factorial 3)) in
     c8_6 - c8_5 = -28) :=
by
  intros x y hx_ne_zero
  let c8_6 := (nat.factorial 8) / ((nat.factorial 6) * (nat.factorial 2))
  let c8_5 := (nat.factorial 8) / ((nat.factorial 5) * (nat.factorial 3))
  have h : c8_6 - c8_5 = -28 := sorry
  exact h

end coefficient_x2_y6_in_expansion_l266_266930


namespace locus_of_P_l266_266935

noncomputable def circle_center (O : Type*) [metric_space O] (C : set O) := 
∃ (r : ℝ) (o : O), C = metric.sphere o r

noncomputable def tangent_line_point (O : Type*) [metric_space O] (C : set O) (L : set O) := 
∃ (T : O), ∀ (P : O), P ∈ metric.tangent_lines T (circle_center C) ↔ P ∈ L

theorem locus_of_P {O : Type*} [metric_space O]
  (C : set O) (L : set O) (M : O) :
  (circle_center C ∧ tangent_line_point C L ∧ M ∈ L) →
  {P : O | (∃ Q R : O, Q ∈ L ∧ R ∈ L ∧ M = (Q + R) / 2 ∧ incircle_tangent_triangle P Q R C) } =
  {P : O | ∃ S T' : O, ∀ P ∈ ray_starting_from (T' : O) (direction_of S) \ {T'}} := 
sorry

end locus_of_P_l266_266935


namespace correct_statements_eq_two_l266_266490

def is_correct (i : ℕ) : Prop :=
  match i with
  | 1 => False  -- Statement ① is incorrect
  | 2 => True   -- Statement ② is correct
  | 3 => True   -- Statement ③ is correct
  | _ => False

def num_correct_statements : ℕ :=
  (if is_correct 1 then 1 else 0) +
  (if is_correct 2 then 1 else 0) +
  (if is_correct 3 then 1 else 0)

theorem correct_statements_eq_two : num_correct_statements = 2 :=
by
  -- Proof to be filled in
  sorry

end correct_statements_eq_two_l266_266490


namespace value_expression_eq_2_l266_266603

-- Definitions based on the conditions in the problem
variables {A B C N D P E T : Type} [has_angle A B C] [has_angle B C A]
variables (AB_midpoint : N = midpoint A B)
variables (angle_A_gt_angle_B : angle A > angle B)
variables (D_on_ray_AC : on_ray A C D)
variables (CD_equals_BC : segment C D = segment B C)
variables (P_on_ray_DN : on_ray D N P)
variables (same_side_BC : same_side P A B C)
variables (angle_PBC_equals_A : angle P B C = angle A)
variables (PC_intersects_AB_at_E : intersects (PC) (AB) E)
variables (DP_intersects_BC_at_T : intersects (DP) (BC) T)

-- The proof goal
theorem value_expression_eq_2 : 
  ∀ (BC TC EA EB : ℝ),
    BC / TC - EA / EB = 2 :=
sorry

end value_expression_eq_2_l266_266603


namespace number_of_sixth_powers_less_than_1000_l266_266841

theorem number_of_sixth_powers_less_than_1000 : 
  {n : ℕ | ∃ m : ℕ, n = m^6 ∧ n < 1000}.to_finset.card = 3 := 
by {
  sorry
}

end number_of_sixth_powers_less_than_1000_l266_266841


namespace vector_lines_proof_l266_266098

noncomputable def vector_lines {R : Type*} [IsROrC R] [NormedSpace R R] 
  (c r : R) (c_const : IsConst c) (v := CrossProduct.cross c r) : Prop :=
  ∃ k1 k2 t : ℝ, (dot c r = k1) ∧ (norm r ^ 2 = k2)

theorem vector_lines_proof {R : Type*} [IsROrC R] [NormedSpace R R] 
  (c r : R) (c_const : IsConst c) (v := CrossProduct.cross c r) : 
  ∃ t : ℝ, vector_lines c r c_const :=
begin
  unfold vector_lines,
  existsi [0, 0], -- dummy constants to satisfy the existence part
  sorry
end

end vector_lines_proof_l266_266098


namespace sum_of_consecutive_integers_l266_266361

theorem sum_of_consecutive_integers (n : ℕ) (h : n * (n + 1) / 2 ≤ 500) :
  n = 31 ∨ ((32 * (32 + 1) / 2) ≤ 500) := 
sorry

end sum_of_consecutive_integers_l266_266361


namespace factorable_polynomial_count_l266_266093

theorem factorable_polynomial_count :
  (∃ (n : ℕ), (1 ≤ n ∧ n ≤ 5000) ∧ (∃ (a b c : ℤ), (b = a + 1) ∧ (c = a * b) ∧ (a * c = n))) → 
  (finset.card ((finset.range 5001).filter 
  (λ n, ∃ (a b c : ℤ), (b = a + 1) ∧ (c = a * b) ∧ (a * c = n))) = 17) :=
by
  sorry

end factorable_polynomial_count_l266_266093


namespace problem_l266_266965

def f (n : ℕ) (x : ℝ) : ℝ :=
  (List.range n).map (λ k, Real.cos (x / 2^k)).prod

theorem problem (x : ℝ) (h : x = 8 * Real.pi / 3) : f 5 x = -Real.sqrt 3 / 32 :=
  by
  simp [f, h]
  sorry

end problem_l266_266965


namespace man_rate_in_still_water_l266_266467

-- Define the conditions as hypotheses
variables {R S : ℝ} -- R is the man's rate in still water, S is the speed of the stream

-- Given conditions
theorem man_rate_in_still_water (h1 : R + S = 16) (h2 : R - S = 4) : R = 10 :=
by
  simp at *
  sorry

end man_rate_in_still_water_l266_266467


namespace inlet_pipe_rate_l266_266464

-- Conditions definitions
def tank_capacity : ℕ := 4320
def leak_empty_time : ℕ := 6
def full_empty_time_with_inlet : ℕ := 8

-- Question translated into a theorem
theorem inlet_pipe_rate : 
  (tank_capacity / leak_empty_time) = 720 →
  (tank_capacity / full_empty_time_with_inlet) = 540 →
  ∀ R : ℕ, 
    R - 720 = 540 →
    (R / 60) = 21 :=
by
  intros h_leak h_net R h_R
  sorry

end inlet_pipe_rate_l266_266464


namespace intersection_point_l266_266205

variable (A B C : Vec3)

noncomputable def point_D (B C : Vec3) : Vec3 := (2 / 3) • C + (1 / 3) • B

noncomputable def point_E (A C : Vec3) : Vec3 := (2 / 3) • A + (1 / 3) • C

noncomputable def point_P (A B C : Vec3) : Vec3 := (1 / 3) • A + (1 / 2) • B + (1 / 6) • C

theorem intersection_point
   (A B C : Vec3)
   (D := point_D B C)
   (E := point_E A C) : 
   let P := (1 / 3) • A + (1 / 2) • B + (1 / 6) • C in
   ∃ (P : Vec3), ((P ∈ line (B, E)) ∧ (P ∈ line (A, D))) := 
begin
  sorry
end

end intersection_point_l266_266205


namespace min_value_a_l266_266156

theorem min_value_a (a : ℝ) : (∀ x : ℝ, a < x → 2 * x + 2 / (x - a) ≥ 7) → a ≥ 3 / 2 :=
by
  sorry

end min_value_a_l266_266156


namespace maximum_consecutive_positive_integers_sum_500_l266_266387

theorem maximum_consecutive_positive_integers_sum_500 : 
  ∃ n : ℕ, (n * (n + 1) / 2 < 500) ∧ (∀ m : ℕ, (m * (m + 1) / 2 < 500) → m ≤ n) :=
sorry

end maximum_consecutive_positive_integers_sum_500_l266_266387


namespace evaluate_expression_l266_266530

theorem evaluate_expression (a b : ℕ) (ha : a = 2) (hb : b = 3) : (a^b)^a - (b^a)^b = -665 :=
by
  rw [ha, hb]
  calc
    (2^3)^2 - (3^2)^3 = 8^2 - 9^3 : by rw [Nat.pow_pow 2 3 2, Nat.pow_pow 3 2 3]
                   ... = 64 - 729   : by rw [Nat.pow 8 2, Nat.pow 9 3]
                   ... = -665       : by norm_num

end evaluate_expression_l266_266530


namespace fixed_point_l266_266134

theorem fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  2 + a^(1-1) = 3 :=
by
  sorry

end fixed_point_l266_266134


namespace find_circle_radius_l266_266035

def diagonals := (d1 : ℝ) (d2 : ℝ)
def radius (d1 d2 : ℝ) : ℝ := 
  (d1 / 2) * (d1 / 2) + (d2 / 2) * (d2 / 2)

theorem find_circle_radius : 
  ∀ (d1 d2 : ℝ), d1 = 12 → d2 = 6 → radius d1 d2 = 7.5 := 
by
  intros d1 d2 h1 h2
  unfold radius
  rw [h1, h2]
  -- Here you would proceed with the steps given in the solution to prove the radius is as expected,
  -- but inserting the proof steps is not required.
  sorry

end find_circle_radius_l266_266035


namespace analytical_expression_monotonicity_solve_inequality_l266_266566

-- Definition of the function f given the properties
def f (x : ℝ) : ℝ := (x / (x ^ 2 + 1))

-- Condition: f is defined on (-1, 1)
def defined_on (x : ℝ) : Prop := -1 < x ∧ x < 1

-- Condition: f is odd
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Given condition: f(1/3) = 3/10
def value_at_point : Prop := f (1 / 3) = 3 / 10

-- Problem 1: Show the analytical expression is f(x) = x / (x^2 + 1)
theorem analytical_expression
  (h1 : value_at_point)
  (h2 : is_odd f)
  (h3 : ∀ x, defined_on x) :
  ∀ x, f x = x / (x ^ 2 + 1) := 
sorry

-- Problem 2: Prove the monotonicity of f(x) on (-1,1)
theorem monotonicity
  (h1 : ∀ x, defined_on x) :
  ∀ x1 x2 : ℝ, defined_on x1 → defined_on x2 → x1 < x2 → f x1 < f x2 := 
sorry

-- Problem 3: Solve the inequality f(2t) + f(3t - 1) < 0
theorem solve_inequality
  (h1 : value_at_point)
  (h2 : is_odd f)
  (h3 : ∀ x, defined_on x) :
  ∀ t : ℝ, defined_on (2 * t) → defined_on (3 * t - 1) → 0 < t ∧ t < 1 / 5 :=
sorry

end analytical_expression_monotonicity_solve_inequality_l266_266566


namespace proof_problem_l266_266875

theorem proof_problem (p q : Prop) (hnpq : ¬ (p ∧ q)) (hnp : ¬ p) : ¬ p :=
by
  exact hnp

end proof_problem_l266_266875


namespace new_roots_quadratic_l266_266271

variable {p q : ℝ}

theorem new_roots_quadratic :
  (∀ (r₁ r₂ : ℝ), r₁ + r₂ = -p ∧ r₁ * r₂ = q → 
  (x : ℝ) → x^2 + ((p^2 - 2 * q)^2 - 2 * q^2) * x + q^4 = 0) :=
by 
  intros r₁ r₂ h x
  have : r₁ + r₂ = -p := h.1
  have : r₁ * r₂ = q := h.2
  sorry

end new_roots_quadratic_l266_266271


namespace isosceles_triangle_vertex_angle_l266_266934

theorem isosceles_triangle_vertex_angle (A B C : Type) [is_Isosceles_triangle A B C] 
(angle_interior : ∃ angle : ℝ, angle = 80) : 
∀ vertex_angle : ℝ, vertex_angle = 80 ∨ vertex_angle = 20 :=
by
  sorry

end isosceles_triangle_vertex_angle_l266_266934


namespace find_vector_b_magnitude_l266_266132

variable {a b : ℝ^2}
variable (magnitude_a : ℝ) (dot_product_ab : ℝ) (angle : ℝ)

noncomputable def vector_b_magnitude
  (h1 : magnitude_a = 5)
  (h2 : dot_product_ab = 10)
  (h3 : angle = Real.pi / 3) : ℝ :=
  let ⟨a, b⟩ := (magnitude_a, dot_product_ab) in
  ⟨5, 10⟩

theorem find_vector_b_magnitude
  (h1 : |a| = 5)
  (h2 : a.dot b = 10)
  (h3 : ∃ (θ : ℝ), θ = Real.pi / 3) : |b| = 4 :=
by
  sorry

end find_vector_b_magnitude_l266_266132


namespace count_divisible_factorials_l266_266103

def S (n : ℕ) : ℕ := (n * (n + 1)) / 2 + 12

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

def is_divisible (a b : ℕ) : Prop := b ≠ 0 ∧ a % b = 0

theorem count_divisible_factorials : 
  (finset.filter (λ n : ℕ, is_divisible (factorial n) (S n)) ((finset.range 31).filter (λ x, x > 0))).card = 25 :=
by
  sorry

end count_divisible_factorials_l266_266103


namespace omega_range_l266_266119

theorem omega_range {ω : ℝ} 
  (hω : ω > 0)
  (a b : ℝ)
  (ha : π ≤ a) (hb : a < b) (hb' : b ≤ 2 * π)
  (h : sin (ω * a) + sin (ω * b) = 2) :
  (9 / 4 ≤ ω ∧ ω ≤ 5 / 2) ∨ (13 / 4 ≤ ω) :=
by sorry

end omega_range_l266_266119


namespace find_t_l266_266874

noncomputable def f (t : ℝ) (x : ℝ) : ℝ := 
  (2 * t * x^2 + (sqrt 2) * t * sin (x + π / 4) + x) / (2 * x^2 + cos x)

theorem find_t (a b t : ℝ) (h₀ : t ≠ 0) (h₁ : ∀ x, ∃ a b, is_max (f t x) a ∧ is_min (f t x) b)
  (h₂ : a + b = 2) : t = 1 := 
sorry

end find_t_l266_266874


namespace math_proof_problem_l266_266062

theorem math_proof_problem : 
  (325 - Real.sqrt 125) / 425 = 65 - 5 := 
by sorry

end math_proof_problem_l266_266062


namespace count_sixth_powers_below_1000_l266_266622

theorem count_sixth_powers_below_1000 : 
  {n : ℕ | (n < 1000) ∧ (∃ m : ℕ, n = m ^ 6)}.to_finset.card = 3 :=
by sorry

end count_sixth_powers_below_1000_l266_266622


namespace count_squares_and_cubes_less_than_1000_l266_266664

theorem count_squares_and_cubes_less_than_1000 : 
  {n : ℕ // n^6 < 1000}.card = 3 :=
by
  sorry

end count_squares_and_cubes_less_than_1000_l266_266664


namespace sum_of_solutions_eq_zero_l266_266096

noncomputable def f (x : ℝ) : ℝ := 3^(abs x) + 4 * abs x

theorem sum_of_solutions_eq_zero :
  {x : ℝ | f x = 20}.sum = 0 :=
sorry

end sum_of_solutions_eq_zero_l266_266096


namespace count_sixth_powers_below_1000_l266_266830

-- Definition and necessary condition for perfect sixth power
def is_sixth_power (n : ℕ) : Prop := ∃ k : ℕ, n = k ^ 6

-- Main statement of the problem
theorem count_sixth_powers_below_1000 :
  {n : ℕ | n < 1000 ∧ is_sixth_power n}.to_finset.card = 3 := 
sorry

end count_sixth_powers_below_1000_l266_266830


namespace number_of_sixth_powers_lt_1000_l266_266706

-- Define the condition for being a sixth power and being less than 1000
def isSixthPower (n : ℕ) : Prop := ∃ k : ℕ, n = k^6

-- Define the main statement to be proved
theorem number_of_sixth_powers_lt_1000 : 
  {n : ℕ | n < 1000 ∧ isSixthPower n}.to_finset.card = 3 :=
by
  sorry

end number_of_sixth_powers_lt_1000_l266_266706


namespace count_squares_cubes_less_than_1000_l266_266811

theorem count_squares_cubes_less_than_1000 : 
  {n : ℕ | ∃ x : ℕ, n = x^6 ∧ n < 1000}.card = 3 :=
sorry

end count_squares_cubes_less_than_1000_l266_266811


namespace sum_of_angles_eq_1080_degrees_l266_266323

theorem sum_of_angles_eq_1080_degrees :
  ∃ (φ : ℕ → ℝ) (n : ℕ),
    n = 6 ∧
    (∀ k, 1 ≤ k ∧ k ≤ n → 0 ≤ φ k ∧ φ k < 360) ∧
    (∃ z : ℂ,
      z ^ n = -1 / 2 - complex.I * real.sqrt 3 / 2 ∧
      ∀ k, φ k = (240 + 360 * k) / 6 ) →
    (∑ k in finset.range n, φ k) = 1080 :=
sorry

end sum_of_angles_eq_1080_degrees_l266_266323


namespace range_of_AB_l266_266133

-- Define the points and lengths involved
variables {A B P : ℝ^2} [is_unit_circle A] [is_unit_circle B] [is_unit_circle P]
variables (λ : ℝ) (AB BP BA : ℝ) (f : ℝ → ℝ)

-- Define the function f
def f (λ : ℝ) : ℝ := abs (BP - λ * BA)

-- Given conditions
axiom AB_is_chord : AB < 2
axiom P_moving_unit_circle : P ≠ A ∧ P ≠ B
axiom f_max : f λ ≥ 3/2

-- Prove the range of values for |AB|
theorem range_of_AB (AB : ℝ) : 0 < AB ∧ AB ≤ sqrt 3 := sorry

end range_of_AB_l266_266133


namespace prob_of_Xiao_Chen_given_Xiao_Li_l266_266416

def P (event: Type) := ℚ

variables (A B : Type)
variables (P_A : P A = 1/4) (P_B : P B = 1/3) (P_AB : P (A ∩ B) = 1/6)

theorem prob_of_Xiao_Chen_given_Xiao_Li :
  P (A ∩ B) / P A = 2/3 :=
by
  -- no proof needed
  sorry

end prob_of_Xiao_Chen_given_Xiao_Li_l266_266416


namespace find_a_value_l266_266014

theorem find_a_value :
  let center := (0.5, Real.sqrt 2)
  let line_dist (a : ℝ) := (abs (0.5 * a + Real.sqrt 2 - Real.sqrt 2)) / Real.sqrt (a^2 + 1)
  line_dist a = Real.sqrt 2 / 4 ↔ (a = 1 ∨ a = -1) :=
by
  sorry

end find_a_value_l266_266014


namespace initial_pencils_count_l266_266998

-- Define the conditions
def students : ℕ := 25
def pencils_per_student : ℕ := 5

-- Statement of the proof problem
theorem initial_pencils_count : students * pencils_per_student = 125 :=
by
  sorry

end initial_pencils_count_l266_266998


namespace tan_diff_l266_266108

variables {α β : ℝ}

theorem tan_diff (h1 : Real.tan α = -3/4) (h2 : Real.tan (Real.pi - β) = 1/2) :
  Real.tan (α - β) = -2/11 :=
by
  sorry

end tan_diff_l266_266108


namespace triangle_area_cut_off_by_line_l266_266537

-- Define the conditions
def diamond_region (x y : ℝ) : Prop := abs (x - 2) + abs (y - 3) ≤ 3
def cutting_line (x y : ℝ) : Prop := y = 2 * x + 2

-- Triangle area theorem statement
theorem triangle_area_cut_off_by_line : 
  (∃ x1 y1 x2 y2 x3 y3, diamond_region x1 y1 ∧ diamond_region x2 y2 ∧ diamond_region x3 y3 ∧ 
  cutting_line x1 y1 ∧ cutting_line x2 y2 ∧ ¬ cutting_line x3 y3 ∧ 
  colinear x1 y1 x2 y2 x3 y3 ∧ right_triangle x1 y1 x2 y2 x3 y3 ∧ 
  triangle_area x1 y1 x2 y2 x3 y3 = 3) := 
sorry

end triangle_area_cut_off_by_line_l266_266537


namespace coeff_x2y6_in_expansion_l266_266901

theorem coeff_x2y6_in_expansion : 
  (coeff_of_term (x := 2) (y := 6)) ((1 - (y / x)) * (x + y) ^ 8) = -28 :=
sorry

end coeff_x2y6_in_expansion_l266_266901


namespace count_sixth_powers_below_1000_l266_266835

-- Definition and necessary condition for perfect sixth power
def is_sixth_power (n : ℕ) : Prop := ∃ k : ℕ, n = k ^ 6

-- Main statement of the problem
theorem count_sixth_powers_below_1000 :
  {n : ℕ | n < 1000 ∧ is_sixth_power n}.to_finset.card = 3 := 
sorry

end count_sixth_powers_below_1000_l266_266835


namespace parallel_vectors_m_eq_neg3_l266_266607

theorem parallel_vectors_m_eq_neg3 {m : ℝ} :
  let a := (1, -2)
  let b := (1 + m, 1 - m)
  (a.1 * b.2 =  a.2 * b.1) → m = -3 :=
by 
  let a := (1, -2)
  let b := (1 + m, 1 - m)
  intro h
  sorry

end parallel_vectors_m_eq_neg3_l266_266607


namespace smallest_period_f_solution_set_f_l266_266589

noncomputable def f (x : ℝ) : ℝ :=
  2 * (sin x * cos x) + 2 * sqrt 3 * (cos x)^2 - sqrt 3

theorem smallest_period_f :
  (∃ T > 0, ∀ x, f (x + T) = f x) →
  (∀ T' > 0, (∃ T > 0, ∀ x, f(x + T) = f x) → T' >= T) →
  ∀ x, f (x + π) = f x :=
begin
  sorry
end

theorem solution_set_f :
  (∀ x ∈ Icc (0:ℝ) π, f x ≥ 1) →
  set_eq
    { x | x ∈ Icc (0:ℝ) π ∧ f x ≥ 1 }
    (Icc 0 (π / 4) ∪ Icc (11 * π / 12) π) :=
begin
  sorry
end

end smallest_period_f_solution_set_f_l266_266589


namespace distance_between_foci_of_hyperbola_l266_266309

theorem distance_between_foci_of_hyperbola (x y : ℝ) (h : x * y = 4) : 
  distance (2, 2) (-2, -2) = 8 :=
sorry

end distance_between_foci_of_hyperbola_l266_266309


namespace number_of_sixth_powers_lt_1000_l266_266682

theorem number_of_sixth_powers_lt_1000 : 
  {n : ℕ | ∃ k : ℕ, k^6 = n ∧ n < 1000}.to_finset.card = 3 :=
by
  sorry

end number_of_sixth_powers_lt_1000_l266_266682


namespace count_squares_and_cubes_less_than_1000_l266_266666

theorem count_squares_and_cubes_less_than_1000 : 
  {n : ℕ // n^6 < 1000}.card = 3 :=
by
  sorry

end count_squares_and_cubes_less_than_1000_l266_266666


namespace count_squares_and_cubes_less_than_1000_l266_266739

theorem count_squares_and_cubes_less_than_1000 :
  {n : ℕ | 0 < n ∧ n < 1000 ∧ ∃ x, n = x ^ 6}.to_finset.card = 3 :=
by sorry

end count_squares_and_cubes_less_than_1000_l266_266739


namespace coefficient_of_x2y6_l266_266894

theorem coefficient_of_x2y6 :
  let binom_coeff8 (r : ℕ) : ℕ := Nat.choose 8 r
  ((1 - y / x) * (x + y)^8).coeff (2, 6) = -28 :=
by
  sorry

end coefficient_of_x2y6_l266_266894


namespace SusanBooks_l266_266241

-- Definitions based on the conditions of the problem
def Lidia (S : ℕ) : ℕ := 4 * S
def TotalBooks (S : ℕ) : ℕ := S + Lidia S

-- The proof statement
theorem SusanBooks (S : ℕ) (h : TotalBooks S = 3000) : S = 600 :=
by
  sorry

end SusanBooks_l266_266241


namespace maximum_a_value_l266_266548

theorem maximum_a_value :
  ∀ (a : ℝ), (∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → -2022 ≤ (a + 1)*x^2 - (a + 1)*x + 2022 ∧ (a + 1)*x^2 - (a + 1)*x + 2022 ≤ 2022) →
  a ≤ 16175 := 
by {
  sorry
}

end maximum_a_value_l266_266548


namespace MrsHilt_money_left_l266_266245

theorem MrsHilt_money_left (initial_amount pencil_cost remaining_amount : ℕ) 
  (h_initial : initial_amount = 15) 
  (h_cost : pencil_cost = 11) 
  (h_remaining : remaining_amount = 4) : 
  initial_amount - pencil_cost = remaining_amount := 
by 
  sorry

end MrsHilt_money_left_l266_266245


namespace number_of_squares_and_cubes_less_than_1000_l266_266719

noncomputable def count_squares_and_cubes : ℕ :=
  (List.range 1000).countp (λ n : ℕ, (∃ k : ℕ, k^6 = n))

theorem number_of_squares_and_cubes_less_than_1000 :
  count_squares_and_cubes = 3 :=
  by
    sorry

end number_of_squares_and_cubes_less_than_1000_l266_266719


namespace number_of_sixth_powers_lt_1000_l266_266705

-- Define the condition for being a sixth power and being less than 1000
def isSixthPower (n : ℕ) : Prop := ∃ k : ℕ, n = k^6

-- Define the main statement to be proved
theorem number_of_sixth_powers_lt_1000 : 
  {n : ℕ | n < 1000 ∧ isSixthPower n}.to_finset.card = 3 :=
by
  sorry

end number_of_sixth_powers_lt_1000_l266_266705


namespace hamburgers_served_l266_266033

theorem hamburgers_served (total_hamburgers left_over_hamburgers: ℕ) (h1: total_hamburgers = 9) (h2: left_over_hamburgers = 6) : (total_hamburgers - left_over_hamburgers = 3) := 
by {
  -- Given conditions: total_hamburgers = 9 and left_over_hamburgers = 6
  -- We need to prove: total_hamburgers - left_over_hamburgers = 3
  rw [h1, h2],
  norm_num,
  sorry,
}

end hamburgers_served_l266_266033


namespace count_squares_and_cubes_l266_266658

theorem count_squares_and_cubes (bound : ℕ) (hk : bound = 1000) : 
  Nat.card {n : ℕ | n < bound ∧ (∃ k : ℕ, n = k ^ 6)} = 3 := 
by
  sorry

end count_squares_and_cubes_l266_266658


namespace largest_num_pencils_in_package_l266_266244

theorem largest_num_pencils_in_package (Ming_pencils Catherine_pencils : ℕ) 
  (Ming_pencils := 40) 
  (Catherine_pencils := 24) 
  (H : ∃ k, Ming_pencils = k * a ∧ Catherine_pencils = k * b) :
  gcd Ming_pencils Catherine_pencils = 8 :=
by
  sorry

end largest_num_pencils_in_package_l266_266244


namespace solve_the_system_l266_266269

def solve_system_eq (x y : ℝ) : Prop :=
  x^2 + y^2 = 1 ∧ 4 * x * y * (2 * y^2 - 1) = 1

def solutions : set (ℝ × ℝ) :=
  { (x, y) | (x = (sqrt(2 - sqrt 2) / 2) ∨ x = -(sqrt(2 - sqrt 2) / 2)
          ∨ x = (sqrt(2 + sqrt 2) / 2) ∨ x = -(sqrt(2 + sqrt 2) / 2)) ∧ 
           (y = (sqrt(2 + sqrt 2) / 2) ∨ y = -(sqrt(2 + sqrt 2) / 2)
          ∨ y = (sqrt(2 - sqrt 2) / 2) ∨ y = -(sqrt(2 - sqrt 2) / 2)) }

theorem solve_the_system : 
  ∀ (x y : ℝ), solve_system_eq x y ↔ (x, y) ∈ solutions := 
by sorry

end solve_the_system_l266_266269


namespace max_consecutive_integers_sum_lt_500_l266_266374

theorem max_consecutive_integers_sum_lt_500 
  (n : ℕ)
  (h₁ : ∑ k in Finset.range (n+1), k = n * (n + 1) / 2)
  (h₂ : n * (n + 1) / 2 < 500) 
  (h₃ : (n + 1) * (n + 2) / 2 > 500)
  : n = 31 := 
by sorry

end max_consecutive_integers_sum_lt_500_l266_266374


namespace factor_in_range_l266_266317

-- Define the given constants
def a : ℕ := 201212200619
def lower_bound : ℕ := 6000000000
def upper_bound : ℕ := 6500000000
def m : ℕ := 6490716149

-- The Lean proof statement
theorem factor_in_range :
  m ∣ a ∧ lower_bound < m ∧ m < upper_bound :=
by
  exact ⟨sorry, sorry, sorry⟩

end factor_in_range_l266_266317


namespace initial_speed_correct_total_time_late_adjustment_final_speed_correct_final_speed_correct_early_l266_266342

-- Define the problem conditions
def distance_travelled : ℕ := 112
def time_elapsed : ℕ := 2
def total_distance : ℕ := 280
def late_time : ℝ := 0.5
def early_time : ℝ := 0.5

-- Define the initial speed calculation
def initial_speed : ℕ := distance_travelled / time_elapsed

-- Proof statements concerning initial and final speeds
theorem initial_speed_correct : initial_speed = 56 := by
  sorry

-- Define the total time calculation with initial speed
def total_time_with_initial_speed : ℝ := total_distance / initial_speed

-- Proof of total time adjustment
theorem total_time_late_adjustment : total_time_with_initial_speed - late_time = 4.5 := by
  sorry

def remaining_distance : ℕ := total_distance - distance_travelled
def remaining_time_to_be_late : ℝ := 4.5 - time_elapsed

-- Calculate new speed to not be late
def new_speed_to_not_be_late : ℝ := remaining_distance / remaining_time_to_be_late

-- Verify the correct final speed for being on-time
theorem final_speed_correct : new_speed_to_not_be_late = 67.2 := by
  sorry

def remaining_time_to_be_early : ℝ := 4.5 - early_time - time_elapsed
def final_speed_to_be_early : ℝ := remaining_distance / remaining_time_to_be_early

-- Final correct speed for being early
theorem final_speed_correct_early : final_speed_to_be_early = 84 := by
  sorry

end initial_speed_correct_total_time_late_adjustment_final_speed_correct_final_speed_correct_early_l266_266342


namespace necessary_but_not_sufficient_condition_l266_266971

theorem necessary_but_not_sufficient_condition (x y : ℝ) :
  (x^2 + y^2 ≥ 9) → ¬((x > 3) ∧ (y ≥ 3)) :=
begin
  sorry
end

end necessary_but_not_sufficient_condition_l266_266971


namespace number_of_ways_to_arrange_C_and_D_next_to_each_other_l266_266876

theorem number_of_ways_to_arrange_C_and_D_next_to_each_other:
  ∃ n : ℕ, n = 48 ∧
    let people := ["A", "B", "C", "D", "E"] in
    ∀ (positions : list (list string)),
      (positions.length = 5 ∧
       ("C" :: "D" :: []) ∈ positions ∨
       ("D" :: "C" :: []) ∈ positions) →
      ∃! arrangement : list (list string),
        (arrangement.length = 48 ∧
         ∀ x ∈ arrangement, ((people.permutations).nodup ∧ ((∀ y ∈ x, y ∈ ["A", "B", ("C", "D"), "E"]) ∨
         (∀ z ∈ x, z ∈ ["A", "B", ("D", "C"), "E"])))) := sorry

end number_of_ways_to_arrange_C_and_D_next_to_each_other_l266_266876


namespace coefficient_x2_y6_l266_266910

theorem coefficient_x2_y6 : 
  let x y : ℚ
  in let expr := (1 - y / x) * (x + y)^8
  in polynomial.coeff (polynomial.expand x^2 y^6 expr) = -28 := 
by
  sorry

end coefficient_x2_y6_l266_266910


namespace sphere_plane_distance_l266_266585

theorem sphere_plane_distance (R r d : ℝ) (h_volume : (4/3) * Real.pi * R^3 = 36 * Real.pi) 
(diam_circle_eq : r = sqrt 5) 
(h_distance : d = sqrt (R^2 - r^2)) : d = 2 :=
sorry

end sphere_plane_distance_l266_266585


namespace correct_time_fraction_is_49_over_144_l266_266437

-- Defining the conditions
def incorrect_hours := [1, 2, 10, 11, 12]
def total_hours := 12
def correct_hour_fraction := (total_hours - incorrect_hours.length) / total_hours.toRat

def incorrect_minutes_1 := 15
def incorrect_minutes_2 := 15
def overlapping_incorrect_minutes := 5
def total_incorrect_minutes := incorrect_minutes_1 + incorrect_minutes_2 - overlapping_incorrect_minutes
def total_minutes := 60
def correct_minute_fraction := (total_minutes - total_incorrect_minutes) / total_minutes.toRat

-- The proof problem statement
theorem correct_time_fraction_is_49_over_144 :
  (correct_hour_fraction * correct_minute_fraction = (49/144 : ℚ)) :=
by
  sorry

end correct_time_fraction_is_49_over_144_l266_266437


namespace count_squares_and_cubes_l266_266649

theorem count_squares_and_cubes (bound : ℕ) (hk : bound = 1000) : 
  Nat.card {n : ℕ | n < bound ∧ (∃ k : ℕ, n = k ^ 6)} = 3 := 
by
  sorry

end count_squares_and_cubes_l266_266649


namespace find_angle_B_range_of_2a_add_c_l266_266940

-- Define the angles and sides of triangle ABC
variables {A B C a b c : ℝ}

-- Given conditions:
-- 1. Given condition relating cosine and sine
axiom given_condition : cos (2 * B) - cos (2 * A) = 2 * sin C * (sin A - sin C)
-- 2. Specific side length b
axiom side_length_b : b = sqrt 3

-- Part 1: Prove that B = π / 3
theorem find_angle_B (h₁ : cos (2 * B) - cos (2 * A) = 2 * sin C * (sin A - sin C))
: B = π / 3 :=
by {
  sorry,
}

-- Part 2: Determine the range of 2a + c
theorem range_of_2a_add_c (h₂ : b = sqrt 3) (h₃ : B = π / 3) :
  (2 * a + c) ∈ set.Icc (sqrt 3) (2 * sqrt 7) :=
by {
  sorry,
}

end find_angle_B_range_of_2a_add_c_l266_266940


namespace count_positive_integers_square_and_cube_lt_1000_l266_266757

theorem count_positive_integers_square_and_cube_lt_1000 : 
  {n : ℕ | n < 1000 ∧ ∃ k : ℕ, n = k^6}.to_finset.card = 3 :=
by
  sorry

end count_positive_integers_square_and_cube_lt_1000_l266_266757


namespace max_consecutive_sum_less_500_l266_266355

theorem max_consecutive_sum_less_500 : 
  (argmax (λ n : ℕ, n * (n + 1) / 2 < 500) = 31) :=
by sorry

end max_consecutive_sum_less_500_l266_266355


namespace calculate_unoccupied_volume_l266_266979

noncomputable def volume_unoccupied (V_container V_water_initial V_ice_cube V_total_ice V_water_final V_combined : ℝ) : ℝ :=
  V_container - V_combined

theorem calculate_unoccupied_volume :
  let V_container := 12 * 12 * 12,
      V_water_initial := (1 / 3) * V_container,
      V_ice_cube := 1.5 * 1.5 * 1.5,
      V_total_ice := 20 * V_ice_cube,
      V_water_final := V_water_initial,
      V_combined := V_water_final + V_total_ice
  in volume_unoccupied V_container V_water_initial V_ice_cube V_total_ice V_water_final V_combined = 1084.5 :=
by
  sorry

end calculate_unoccupied_volume_l266_266979


namespace count_positive_integers_square_and_cube_lt_1000_l266_266760

theorem count_positive_integers_square_and_cube_lt_1000 : 
  {n : ℕ | n < 1000 ∧ ∃ k : ℕ, n = k^6}.to_finset.card = 3 :=
by
  sorry

end count_positive_integers_square_and_cube_lt_1000_l266_266760


namespace distance_between_foci_l266_266298

-- Define the equation of the hyperbola.
def hyperbola_eq (x y : ℝ) : Prop := x * y = 4

-- The coordinates of foci for hyperbola of the form x*y = 4
def foci_1 : (ℝ × ℝ) := (2, 2)
def foci_2 : (ℝ × ℝ) := (-2, -2)

-- Define the Euclidean distance function.
def euclidean_distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Prove that the distance between the foci is 4√2.
theorem distance_between_foci : euclidean_distance foci_1 foci_2 = 4 * real.sqrt 2 := sorry

end distance_between_foci_l266_266298


namespace max_consecutive_integers_sum_l266_266402

theorem max_consecutive_integers_sum (n : ℕ) : 
  (∃ n, (n * (n + 1) / 2 < 500) ∧ (∀ m > n, m * (m + 1) / 2 ≥ 500)) :=
by {
  use 31,
  split,
  {
    norm_num,
  },
  {
    intros m hm,
    have h : m = 32 := by linarith,
    rw h,
    norm_num,
  },
  sorry
}

end max_consecutive_integers_sum_l266_266402


namespace total_card_units_traded_l266_266987

theorem total_card_units_traded :
  let padma_cards_round1 : Π (a b c : ℕ), ({ a := 50, b := 45, c := 30 } : CardCount)
  let padma_cards_round2 : Π (a b : ℕ), ({ a := padma_cards_round1.a - 5, b := padma_cards_round1.b - 12, c := padma_cards_round1.c + 20 } : CardCount)
  let robert_cards_round1 : Π (a b c : ℕ), ({ a := 60, b := 50, c := 40 } : CardCount)
  let robert_cards_round2 : Π (a b c : ℕ), ({ a := robert_cards_round1.a - 10, b := robert_cards_round1.b - 3, c := robert_cards_round1.c - 15, a := robert_cards_round1.a - 8, b := robert_cards_round1.b - 18 } : CardCount)
  let padma_cards_round3 : Π (a b c : ℕ), ({ a := padma_cards_round2.a + 8, b := padma_cards_round2.b + 18, c := padma_cards_round2.c + 15 } : CardCount)
  let total_card_units_b := 5 * 2 + 12 * 1.5 + 15 * 1.2 + 10 / 0.8
  let total_card_units_r := 10 * 1.5 + 3 * 2 + 15 * 1
  total_card_units_b + total_card_units_r = 94.75 :=
sorry

end total_card_units_traded_l266_266987


namespace coefficient_of_x2y6_l266_266896

theorem coefficient_of_x2y6 :
  let binom_coeff8 (r : ℕ) : ℕ := Nat.choose 8 r
  ((1 - y / x) * (x + y)^8).coeff (2, 6) = -28 :=
by
  sorry

end coefficient_of_x2y6_l266_266896


namespace river_trip_longer_than_lake_trip_l266_266503

theorem river_trip_longer_than_lake_trip (v w : ℝ) (h1 : v > w) : 
  (20 * v) / (v^2 - w^2) > 20 / v :=
by {
  sorry
}

end river_trip_longer_than_lake_trip_l266_266503


namespace range_of_a_l266_266601

noncomputable def setA : Set ℝ := {x | x^2 + 4 * x = 0}
noncomputable def setB (a : ℝ) : Set ℝ := {x | x^2 + a * x + a = 0}

theorem range_of_a :
  ∀ a : ℝ, (setA ∪ setB a) = setA ↔ 0 ≤ a ∧ a < 4 :=
by sorry

end range_of_a_l266_266601


namespace distance_between_foci_of_hyperbola_l266_266310

theorem distance_between_foci_of_hyperbola (x y : ℝ) (h : x * y = 4) : 
  distance (2, 2) (-2, -2) = 8 :=
sorry

end distance_between_foci_of_hyperbola_l266_266310


namespace number_of_sixth_powers_less_than_1000_l266_266641

theorem number_of_sixth_powers_less_than_1000 : 
  {n : ℕ | n < 1000 ∧ ∃ (k : ℕ), n = k^6}.finite.card = 3 :=
by sorry

end number_of_sixth_powers_less_than_1000_l266_266641


namespace gray_region_area_l266_266066

noncomputable def area_of_gray_region : ℝ :=
  let area_of_rectangle := 25
  let area_of_circle_C_sector := (9 * Real.pi) / 4
  let area_of_circle_D_sector := (25 * Real.pi) / 4
  area_of_rectangle - (area_of_circle_C_sector + area_of_circle_D_sector)

theorem gray_region_area (C_center D_center : ℝ × ℝ) (C_radius D_radius : ℝ) 
  (hC : C_center = (5, 5)) (hC_r : C_radius = 3)
  (hD : D_center = (10, 5)) (hD_r : D_radius = 5) :
  area_of_gray_region = 25 - 8.5 * Real.pi := 
by
  rw [← hC, ← hC_r, ← hD, ← hD_r]
  sorry

end gray_region_area_l266_266066


namespace num_sixth_powers_below_1000_l266_266804

theorem num_sixth_powers_below_1000 : 
  {n : ℕ | n^6 < 1000 ∧ n > 0}.to_finset.card = 3 :=
by sorry

end num_sixth_powers_below_1000_l266_266804


namespace count_squares_cubes_less_than_1000_l266_266820

theorem count_squares_cubes_less_than_1000 : 
  {n : ℕ | ∃ x : ℕ, n = x^6 ∧ n < 1000}.card = 3 :=
sorry

end count_squares_cubes_less_than_1000_l266_266820


namespace inequality_squares_l266_266109

theorem inequality_squares (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a^2 > b^2 :=
sorry

end inequality_squares_l266_266109


namespace max_consecutive_sum_l266_266391

theorem max_consecutive_sum : ∃ (n : ℕ), (∀ m : ℕ, (m < n → (m * (m + 1)) / 2 < 500)) ∧ ¬((n * (n + 1)) / 2 < 500) :=
by {
  sorry
}

end max_consecutive_sum_l266_266391


namespace number_of_sixth_powers_less_than_1000_l266_266636

theorem number_of_sixth_powers_less_than_1000 : 
  {n : ℕ | n < 1000 ∧ ∃ (k : ℕ), n = k^6}.finite.card = 3 :=
by sorry

end number_of_sixth_powers_less_than_1000_l266_266636


namespace cornbread_pieces_l266_266950

theorem cornbread_pieces (pan_length : ℕ) (pan_width : ℕ) (piece_length : ℕ) (piece_width : ℕ)
  (hl : pan_length = 20) (hw : pan_width = 18) (hp : piece_length = 2) (hq : piece_width = 2) :
  (pan_length * pan_width) / (piece_length * piece_width) = 90 :=
by
  sorry

end cornbread_pieces_l266_266950


namespace binomial_expansion_constant_term_l266_266871

theorem binomial_expansion_constant_term (x : ℝ) (n : ℕ) :
  (∃ c : ℝ, (3 * x^2 - (1 / (2 * x^3)))^5 = c ∧ c = 135 / 2) :=
by
  sorry

end binomial_expansion_constant_term_l266_266871


namespace cubic_polynomial_eval_at_5_l266_266449

theorem cubic_polynomial_eval_at_5 (p : ℚ[X])
  (h₁ : p.eval 1 = 1)
  (h₂ : p.eval 2 = 1 / 8)
  (h₃ : p.eval 3 = 1 / 27)
  (h₄ : p.eval 4 = 1 / 64) :
  p.eval 5 = 0 :=
sorry

end cubic_polynomial_eval_at_5_l266_266449


namespace noodles_left_l266_266075

theorem noodles_left (initial_noodles : ℝ) (given_noodles : ℝ) (left_noodles : ℝ) : 
  initial_noodles = 54.0 → given_noodles = 12.0 → left_noodles = initial_noodles - given_noodles → left_noodles = 42.0 :=
by
  intros h_initial h_given h_left
  rw [h_initial, h_given] at h_left
  exact h_left

end noodles_left_l266_266075


namespace count_sixth_powers_less_than_1000_l266_266750

theorem count_sixth_powers_less_than_1000 : 
  {n : ℕ | 0 < n ∧ n < 1000 ∧ (∃ k : ℕ, n = k^6)}.to_finset.card = 3 :=
by 
  sorry

end count_sixth_powers_less_than_1000_l266_266750


namespace general_equation_of_line_l266_266873

-- Define the points A and B
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (0, 1)

-- Define what it means for a line to pass through two points
def line_through_points (l : ℝ → ℝ → Prop) (A B : ℝ × ℝ) : Prop :=
  l A.1 A.2 ∧ l B.1 B.2

-- Define the equation of line l
def line_l (x y : ℝ) : Prop := x - 2 * y + 2 = 0

-- The theorem that needs to be proven
theorem general_equation_of_line : line_through_points line_l A B := 
by
  sorry

end general_equation_of_line_l266_266873


namespace exists_n_not_coprime_l266_266604

theorem exists_n_not_coprime (p q : ℕ) (h1 : Nat.gcd p q = 1) (h2 : q > p) (h3 : q - p > 1) :
  ∃ (n : ℕ), Nat.gcd (p + n) (q + n) ≠ 1 :=
by
  sorry

end exists_n_not_coprime_l266_266604


namespace fixed_point_proof_l266_266597

variable {p a b y₀ : ℝ}
variable (A B M M₁ M₂ : ℝ × ℝ)

def parabola (x y : ℝ) := y^2 = 2 * p * x

def point_on_parabola (M : ℝ × ℝ) := parabola M.1 M.2

def fixed_points_conditions (A B: ℝ × ℝ) :=
  A.2 ≠ 0 ∧ B.2 = 0 ∧ A.1 ≠ B.1 ∧ A.1 * B.2 ≠ 0 ∧ B.1 * B.1 ≠ 2 * p * A.1

def collinear (P Q R : ℝ × ℝ) :=
  (Q.2 - P.2) * (R.1 - P.1) = (R.2 - P.2) * (Q.1 - P.1)

def intersect_parabola (M : ℝ × ℝ) (δ : ℝ) :=
  point_on_parabola M → 
  ∃ M', collinear A M M' ∧ collinear B M M' ∧ point_on_parabola M'


theorem fixed_point_proof :
  ∀ (p a b : ℝ) (A B M : ℝ × ℝ),
    (parabola M.1 M.2) →
    fixed_points_conditions A B →
    A ≠ B →
    collinear A M M₁ →
    collinear B M M₂ →
    M₁ ≠ M₂ →
    ∃ (K : ℝ × ℝ),
    K = (a, (2 * p * a) / b) ∧ collinear M₁ M₂ K :=
sorry

end fixed_point_proof_l266_266597


namespace smallest_sum_exists_l266_266101

def consecutive_sum (n : ℕ) (a : ℕ) : ℕ :=
  (Finset.range n).sum (λ k, (a + k) * 10^(n-1-k))

def C_n (n : ℕ) (a b : ℕ) : ℕ :=
  (a + b) * (10^(2 * n) - 1) / 9

theorem smallest_sum_exists
  (n : ℕ) (a b : ℕ)
  (h₀ : n > 0)
  (h₁ : 0 < a) (h₂ : a < 10)
  (h₃ : 0 < b) (h₄ : b < 10) :
  ∃ n, C_n n a b = consecutive_sum n a * consecutive_sum n b ∧ a + b = 11 :=
begin
  sorry
end

end smallest_sum_exists_l266_266101


namespace largest_digit_divisible_by_4_l266_266939

theorem largest_digit_divisible_by_4 :
  ∃ (A : ℕ), A ≤ 9 ∧ (∃ n : ℕ, 100000 * 4 + 10000 * A + 67994 = n * 4) ∧ 
  (∀ B : ℕ, 0 ≤ B ∧ B ≤ 9 ∧ (∃ m : ℕ, 100000 * 4 + 10000 * B + 67994 = m * 4) → B ≤ A) :=
sorry

end largest_digit_divisible_by_4_l266_266939


namespace count_sixth_powers_less_than_1000_l266_266743

theorem count_sixth_powers_less_than_1000 : 
  {n : ℕ | 0 < n ∧ n < 1000 ∧ (∃ k : ℕ, n = k^6)}.to_finset.card = 3 :=
by 
  sorry

end count_sixth_powers_less_than_1000_l266_266743


namespace solve_for_x_l266_266267

theorem solve_for_x (x : ℝ) : 4^(x + 3) = 64^x → x = 3/2 :=
by
  sorry

end solve_for_x_l266_266267


namespace power_function_decreasing_l266_266158

theorem power_function_decreasing (m : ℝ) (h : m^2 - m - 1 = 1) :
  ∀ x : ℝ, (0 < x) → (m = 2) → (y = (m^2 - m - 1) * x^(m^2 - 2 * m - 3) → y = x^(-3)) :=
by 
  sorry

end power_function_decreasing_l266_266158


namespace minimum_daily_expense_l266_266448

-- Defining the context
variables (x y : ℕ)
def total_capacity (x y : ℕ) : ℕ := 24 * x + 30 * y
def cost (x y : ℕ) : ℕ := 320 * x + 504 * y

theorem minimum_daily_expense :
  (total_capacity x y ≥ 180) →
  (x ≤ 8) →
  (y ≤ 4) →
  cost x y = 2560 := sorry

end minimum_daily_expense_l266_266448


namespace probability_no_common_points_l266_266018

theorem probability_no_common_points :
  let outcomes : Finset (ℕ × ℕ) := Finset.product (Finset.range 1 7) (Finset.range 1 7)
  let event_condition (a b : ℕ) : Prop := a > b
  let favorable_outcomes : Finset (ℕ × ℕ) := outcomes.filter (λ p, event_condition p.1 p.2)
  let probability : ℚ := favorable_outcomes.card.to_rat / outcomes.card.to_rat
  in probability = 5 / 12 :=
by
  sorry

end probability_no_common_points_l266_266018


namespace probability_of_rain_l266_266315

theorem probability_of_rain {p : ℝ} (h : p = 0.95) :
  ∃ (q : ℝ), q = (1 - p) ∧ q < p :=
by
  sorry

end probability_of_rain_l266_266315


namespace coefficient_x2y6_l266_266917

/-- In the expansion of (1 - y / x) * (x + y) ^ 8, the coefficient of x ^ 2 * y ^ 6 is -28. -/
theorem coefficient_x2y6 (x y : ℚ) :  
  coefficient ((1 - y / x) * (x + y)^8) (monomial (2, 6)) = -28 :=
sorry

end coefficient_x2y6_l266_266917


namespace product_of_roots_l266_266094

theorem product_of_roots :
  ∀ a b c : ℚ, (a ≠ 0) → a = 24 → b = 60 → c = -600 → (c / a) = -25 :=
sorry

end product_of_roots_l266_266094


namespace count_positive_integers_square_and_cube_lt_1000_l266_266766

theorem count_positive_integers_square_and_cube_lt_1000 : 
  {n : ℕ | n < 1000 ∧ ∃ k : ℕ, n = k^6}.to_finset.card = 3 :=
by
  sorry

end count_positive_integers_square_and_cube_lt_1000_l266_266766


namespace find_L_l266_266489

-- Conditions definitions
def initial_marbles := 57
def marbles_won_second_game := 25
def final_marbles := 64

-- Definition of L
def L := initial_marbles - 18

theorem find_L (L : ℕ) (H1 : initial_marbles = 57) (H2 : marbles_won_second_game = 25) (H3 : final_marbles = 64) : 
(initial_marbles - L) + marbles_won_second_game = final_marbles -> 
L = 18 :=
by
  sorry

end find_L_l266_266489


namespace transformation_matrix_correct_l266_266092

def dilation_matrix (factor : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![factor, 0], ![0, factor]]

def rotation_matrix (angle : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![Real.cos angle, -Real.sin angle], ![Real.sin angle, Real.cos angle]]

def transform_matrix := rotation_matrix (Real.pi / 2) ⬝ dilation_matrix 2

theorem transformation_matrix_correct :
  transform_matrix = ![![0, -2], ![2, 0]] :=
by
  sorry

end transformation_matrix_correct_l266_266092


namespace faster_train_pass_time_correct_l266_266347

-- Definitions for the problem conditions
def length_of_train : ℝ := 62.5
def speed_faster_train_kmhr : ℝ := 46
def speed_slower_train_kmhr : ℝ := 36
def speed_kmhr_to_ms (speed_kmhr : ℝ) : ℝ := speed_kmhr * (1000 / 3600)

-- Converting speeds from km/hr to m/s
def speed_faster_train_ms : ℝ := speed_kmhr_to_ms speed_faster_train_kmhr
def speed_slower_train_ms : ℝ := speed_kmhr_to_ms speed_slower_train_kmhr
def relative_speed_ms : ℝ := speed_faster_train_ms - speed_slower_train_ms

-- Total distance to be covered by the faster train
def total_distance : ℝ := 2 * length_of_train

-- Time to pass
def time_to_pass : ℝ := total_distance / relative_speed_ms

theorem faster_train_pass_time_correct : time_to_pass = 45 := by
  sorry

end faster_train_pass_time_correct_l266_266347


namespace both_square_and_cube_count_both_square_cube_less_than_1000_l266_266775

theorem both_square_and_cube (n : ℕ) (h_positive : n > 0) (h_bound : n < 1000) (h_square : ∃ k : ℕ, n = k^2) (h_cube : ∃ k : ℕ, n = k^3) :
  n ∈ {1, 64, 729} :=
sorry

theorem count_both_square_cube_less_than_1000 :
  {n : ℕ | n > 0 ∧ n < 1000 ∧ ∃ k : ℕ, n = k^2 ∧ ∃ k : ℕ, n = k^3}.to_finset.card = 3 :=
by
  have h_valid : ∀ n ∈ {1, 64, 729}, n > 0 ∧ n < 1000 ∧ (∃ k : ℕ, n = k^2) ∧ (∃ k : ℕ, n = k^3) := sorry
  have h_reverse : ∀ n, n > 0 ∧ n < 1000 ∧ (∃ k : ℕ, n = k^2) ∧ (∃ k : ℕ, n = k^3) → n ∈ {1, 64, 729} := sorry
  exact Finset.card_map _ {1, 64, 729}.to_finset
  sorry

end both_square_and_cube_count_both_square_cube_less_than_1000_l266_266775


namespace difference_between_c_and_a_l266_266179

variables (a b c : ℝ)

theorem difference_between_c_and_a
  (h1 : (a + b) / 2 = 45)
  (h2 : (b + c) / 2 = 90) :
  c - a = 90 := by
  sorry

end difference_between_c_and_a_l266_266179


namespace number_of_sixth_powers_less_than_1000_l266_266845

theorem number_of_sixth_powers_less_than_1000 : 
  {n : ℕ | ∃ m : ℕ, n = m^6 ∧ n < 1000}.to_finset.card = 3 := 
by {
  sorry
}

end number_of_sixth_powers_less_than_1000_l266_266845


namespace count_sixth_powers_below_1000_l266_266833

-- Definition and necessary condition for perfect sixth power
def is_sixth_power (n : ℕ) : Prop := ∃ k : ℕ, n = k ^ 6

-- Main statement of the problem
theorem count_sixth_powers_below_1000 :
  {n : ℕ | n < 1000 ∧ is_sixth_power n}.to_finset.card = 3 := 
sorry

end count_sixth_powers_below_1000_l266_266833


namespace car_distance_l266_266011

noncomputable def distance_covered (S : ℝ) (T : ℝ) (new_speed : ℝ) : ℝ :=
  S * T

theorem car_distance (S : ℝ) (T : ℝ) (new_time : ℝ) (new_speed : ℝ)
  (h1 : T = 12)
  (h2 : new_time = (3/4) * T)
  (h3 : new_speed = 60)
  (h4 : distance_covered new_speed new_time = 540) :
    distance_covered S T = 540 :=
by
  sorry

end car_distance_l266_266011


namespace count_squares_and_cubes_less_than_1000_l266_266670

theorem count_squares_and_cubes_less_than_1000 : 
  {n : ℕ // n^6 < 1000}.card = 3 :=
by
  sorry

end count_squares_and_cubes_less_than_1000_l266_266670


namespace kangaroos_same_as_Kameron_after_40_days_l266_266946

def Kameron_kangaroos : ℕ := 100

def Bert_kangaroos_initial : ℕ := 20
def Bert_kangaroos_per_day : ℕ := 2

def Christina_kangaroos_initial : ℕ := 45
def Christina_kangaroos_per_day : ℕ := 3

def David_kangaroos_initial : ℕ := 10
def David_kangaroos_per_day : ℕ := 5

theorem kangaroos_same_as_Kameron_after_40_days :
  ∀ d : ℕ,
    (Bert_kangaroos_initial + d * Bert_kangaroos_per_day = Kameron_kangaroos) →
    (Christina_kangaroos_initial + d * Christina_kangaroos_per_day = Kameron_kangaroos) →
    (David_kangaroos_initial + d * David_kangaroos_per_day = Kameron_kangaroos) →
    d = 40 :=
by
  assume d,
  intros hBert hChristina hDavid,
  sorry

end kangaroos_same_as_Kameron_after_40_days_l266_266946


namespace graph_quadrants_implies_conditions_l266_266182

-- Define the function
def myFunction (a b x : ℝ) : ℝ := a^x - (b + 1)

-- Define the conditions
variables (a b : ℝ)
variables (h_a_pos : a > 0) (h_a_ne_one : a ≠ 1)
variables (graph_condition : ∀ x : ℝ, myFunction a b x ≠ 0)

-- Formulate the theorem to be proven
theorem graph_quadrants_implies_conditions : (∀ x : ℝ, myFunction a b x ≠ 0) → a > 1 ∧ b > 0 :=
begin
  sorry
end

end graph_quadrants_implies_conditions_l266_266182


namespace find_p_of_min_value_l266_266151

-- given conditions
def f (x : ℝ) (p : ℝ) : ℝ := x + p / (x - 1)
def is_min_value (f : ℝ → ℝ) (a : ℝ) (b : ℝ) (min_val : ℝ) : Prop :=
  ∀ (x : ℝ), a < x -> x < b -> f x ≥ min_val

-- the main problem statement
theorem find_p_of_min_value (p : ℝ) (h_pos : p > 0) 
  (h_min : is_min_value (λ x, f x p) 1 (real.top) 4) : 
  p = 9 / 4 := 
sorry

end find_p_of_min_value_l266_266151


namespace rectangular_prism_diagonals_l266_266031

theorem rectangular_prism_diagonals (h_edges : ∀ (prism : Type), has_edges prism ∧ (count_edges prism = 12))
                                   (h_vertices : ∀ (prism : Type), has_vertices prism ∧ (count_vertices prism = 8))
                                   (h_faces : ∀ (prism : Type), has_faces prism ∧ (count_faces prism = 6))
                                   (h_diagonal : ∀ (prism : Type) (v1 v2 : vertex prism), ¬adjacent v1 v2 → diagonal v1 v2) :
                                   count_diagonals rectangular_prism = 16 := 
sorry

end rectangular_prism_diagonals_l266_266031


namespace number_of_sixth_powers_lt_1000_l266_266681

theorem number_of_sixth_powers_lt_1000 : 
  {n : ℕ | ∃ k : ℕ, k^6 = n ∧ n < 1000}.to_finset.card = 3 :=
by
  sorry

end number_of_sixth_powers_lt_1000_l266_266681


namespace number_of_pupils_l266_266421

theorem number_of_pupils (n : ℕ) (h1 : 79 - 45 = 34)
  (h2 : 34 = 1 / 2 * n) : n = 68 :=
by
  sorry

end number_of_pupils_l266_266421


namespace count_positive_integers_square_and_cube_lt_1000_l266_266762

theorem count_positive_integers_square_and_cube_lt_1000 : 
  {n : ℕ | n < 1000 ∧ ∃ k : ℕ, n = k^6}.to_finset.card = 3 :=
by
  sorry

end count_positive_integers_square_and_cube_lt_1000_l266_266762


namespace sum_binom_identity_l266_266233

theorem sum_binom_identity (n : ℕ) (hpos : 0 < n) : 
  (∑ r in Finset.range (n + 1), 2^(r - 2 * n) * (Nat.choose (2 * n - r) n)) = 1 :=
by
  sorry

end sum_binom_identity_l266_266233


namespace coefficient_of_x2y6_l266_266895

theorem coefficient_of_x2y6 :
  let binom_coeff8 (r : ℕ) : ℕ := Nat.choose 8 r
  ((1 - y / x) * (x + y)^8).coeff (2, 6) = -28 :=
by
  sorry

end coefficient_of_x2y6_l266_266895


namespace count_squares_and_cubes_l266_266651

theorem count_squares_and_cubes (bound : ℕ) (hk : bound = 1000) : 
  Nat.card {n : ℕ | n < bound ∧ (∃ k : ℕ, n = k ^ 6)} = 3 := 
by
  sorry

end count_squares_and_cubes_l266_266651


namespace sn_integer_iff_n_equals_12_l266_266099

theorem sn_integer_iff_n_equals_12 (n : ℕ) (a : Fin n → ℝ) (h1 : 0 < n) 
  (h2 : ∀ i, 0 < a i) (h3 : ∑ i in Finset.range n, a i = 17) :
  (∑ i in Finset.range n, Real.sqrt ((2 * ↑i + 1)^2 + (a i)^2)) ∈ ℤ ↔ n = 12 :=
by sorry

end sn_integer_iff_n_equals_12_l266_266099


namespace number_of_sixth_powers_lt_1000_l266_266684

theorem number_of_sixth_powers_lt_1000 : 
  {n : ℕ | ∃ k : ℕ, k^6 = n ∧ n < 1000}.to_finset.card = 3 :=
by
  sorry

end number_of_sixth_powers_lt_1000_l266_266684


namespace period_of_transformed_function_l266_266591

theorem period_of_transformed_function :
  let y := λ x : ℝ, sqrt 3 * sin (2 * x) - 2 * cos x ^ 2
  let y' := λ x : ℝ, 2 * sin ((2 / 3) * x - π / 4) - 1
  ∃ T > 0, ∀ x : ℝ, y' (x + T) = y' x :=
by
  let y := λ x : ℝ, sqrt 3 * sin (2 * x) - 2 * cos x ^ 2
  let y' := λ x : ℝ, 2 * sin ((2 / 3) * x - π / 4) - 1
  exact ⟨3 * π, by norm_num⟩

end period_of_transformed_function_l266_266591


namespace lambda_range_l266_266274

def symmetric_point_property (f : ℝ → ℝ) (t m : ℝ) : Prop :=
  f (t - m) = f (t + m)

def f (x λ : ℝ) : ℝ := (x^2 + λ) / x

theorem lambda_range (λ : ℝ) :
  (∀ t ∈ Ioo (Real.sqrt 2) (Real.sqrt 6), ∃ m > 0, symmetric_point_property (f λ) t m) →
  0 < λ ∧ λ ≤ 2 :=
by
  sorry

end lambda_range_l266_266274


namespace number_of_sixth_powers_lt_1000_l266_266696

-- Define the condition for being a sixth power and being less than 1000
def isSixthPower (n : ℕ) : Prop := ∃ k : ℕ, n = k^6

-- Define the main statement to be proved
theorem number_of_sixth_powers_lt_1000 : 
  {n : ℕ | n < 1000 ∧ isSixthPower n}.to_finset.card = 3 :=
by
  sorry

end number_of_sixth_powers_lt_1000_l266_266696


namespace determinant_modified_matrix_l266_266956

open Real

variables (u v w : ℝ^3)
noncomputable def E : ℝ := u.dot_product (v.cross_product w)

theorem determinant_modified_matrix :
  let u' := 2 • u + v,
      v' := v + 2 • w,
      w' := 2 • w + u in
  det (λ i, ![u', v', w'].i) = 6 * E u v w :=
sorry

end determinant_modified_matrix_l266_266956


namespace greatest_m_l266_266353

theorem greatest_m (m : ℕ) : (∀ n : ℤ, odd n → n^2 * (1 + n^2 - n^4) ≡ 1 [MOD (2^m)]) → m ≤ 7 :=
  sorry

end greatest_m_l266_266353


namespace milk_butterfat_mixture_l266_266167

theorem milk_butterfat_mixture (x gallons_50 gall_10_perc final_gall mixture_perc: ℝ)
    (H1 : gall_10_perc = 24) 
    (H2 : mixture_perc = 0.20 * (x + gall_10_perc))
    (H3 : 0.50 * x + 0.10 * gall_10_perc = 0.20 * (x + gall_10_perc)) 
    (H4 : final_gall = 20) :
    x = 8 :=
sorry

end milk_butterfat_mixture_l266_266167


namespace penguins_count_l266_266461

theorem penguins_count : 
  ∃ P : ℕ, (P * 3 / 2 = 237) ∧ (let current := (6 * P + 129) in current = 1077) :=
sorry

end penguins_count_l266_266461


namespace sum_of_consecutive_integers_l266_266363

theorem sum_of_consecutive_integers (n : ℕ) (h : n * (n + 1) / 2 ≤ 500) :
  n = 31 ∨ ((32 * (32 + 1) / 2) ≤ 500) := 
sorry

end sum_of_consecutive_integers_l266_266363


namespace median_interval_l266_266040

-- Define the frequencies for the score intervals
def freq (score_min score_max : ℕ) (count : ℕ) : Prop := true

-- Define the count function for each interval
def score_counts : list (ℕ × ℕ × ℕ) := [
    (85, 89, 18),
    (80, 84, 15),
    (75, 79, 20),
    (70, 74, 25),
    (65, 69, 12),
    (60, 64, 10)
]

-- Define a function to calculate cumulative frequency
def cumulative_freq : list (ℕ × ℕ × ℕ) → list ℕ
| [] := []
| ((_, _, c)::tl) := c :: ((cumulative_freq tl).map (λ x => x + c))

-- Define the proof problem to show the interval containing the median
theorem median_interval (counts : list (ℕ × ℕ × ℕ)) (total_students : ℕ) : 
  let cumulative = cumulative_freq counts in
  let median_position := (total_students + 1) / 2 in
  ∃ (min max : ℕ) (count : ℕ), 
    (min, max, count) ∈ counts ∧ 
    (∀ cum_freq ∈ cumulative, cum_freq ≥ median_position) ∧ (∀ cum_freq ∈ cumulative.drop 1, cum_freq ≤ median_position + 1) :=
begin
  -- Sorry for now, actual proof to be done.
  sorry
end

#eval median_interval score_counts 100

end median_interval_l266_266040


namespace min_abs_m_l266_266937

noncomputable def quadratic_eq (x : ℂ) (z₁ z₂ m : ℂ) : ℂ := x^2 + z₁ * x + z₂ + m

theorem min_abs_m (z₁ z₂ m α β : ℂ) 
  (h_eq : quadratic_eq 0 z₁ z₂ m = 0)
  (h_roots : α + β = -z₁ ∧ α * β = z₂ + m)
  (h_condition : z₁^2 - 4 * z₂ = 16 + 20 * complex.I)
  (h_diff_roots : complex.abs (α - β) = 2 * real.sqrt 7) : 
  complex.abs m = 7 - sqrt 41 :=
begin
  sorry
end

end min_abs_m_l266_266937


namespace coefficient_of_x2y6_l266_266899

theorem coefficient_of_x2y6 :
  let binom_coeff8 (r : ℕ) : ℕ := Nat.choose 8 r
  ((1 - y / x) * (x + y)^8).coeff (2, 6) = -28 :=
by
  sorry

end coefficient_of_x2y6_l266_266899


namespace count_squares_cubes_less_than_1000_l266_266815

theorem count_squares_cubes_less_than_1000 : 
  {n : ℕ | ∃ x : ℕ, n = x^6 ∧ n < 1000}.card = 3 :=
sorry

end count_squares_cubes_less_than_1000_l266_266815


namespace ordered_pair_of_positive_integers_l266_266069

theorem ordered_pair_of_positive_integers :
  ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ (x^y + 4 = y^x) ∧ (3 * x^y = y^x + 10) ∧ (x = 7 ∧ y = 1) :=
by
  sorry

end ordered_pair_of_positive_integers_l266_266069


namespace functional_equation_solution_l266_266534

noncomputable def f : (ℝ → ℝ) := λ x, (x + 1 - 1/x - 1/(1-x)) / 2

theorem functional_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x, x ≠ 0 ∧ x ≠ 1 → f x + f (1/(1-x)) = x) 
  : ∀ x, x ≠ 0 ∧ x ≠ 1 → f x = (x + 1 - 1/x - 1/(1-x)) / 2 := 
by 
  sorry

end functional_equation_solution_l266_266534


namespace number_of_sixth_powers_less_than_1000_l266_266847

theorem number_of_sixth_powers_less_than_1000 : 
  {n : ℕ | ∃ m : ℕ, n = m^6 ∧ n < 1000}.to_finset.card = 3 := 
by {
  sorry
}

end number_of_sixth_powers_less_than_1000_l266_266847


namespace round_robin_odd_game_count_l266_266882

theorem round_robin_odd_game_count (n : ℕ) (h17 : n = 17) :
  ∃ p : ℕ, p < n ∧ (p % 2 = 0) :=
by {
  sorry
}

end round_robin_odd_game_count_l266_266882


namespace number_of_squares_and_cubes_less_than_1000_l266_266713

noncomputable def count_squares_and_cubes : ℕ :=
  (List.range 1000).countp (λ n : ℕ, (∃ k : ℕ, k^6 = n))

theorem number_of_squares_and_cubes_less_than_1000 :
  count_squares_and_cubes = 3 :=
  by
    sorry

end number_of_squares_and_cubes_less_than_1000_l266_266713


namespace coefficient_x2y6_l266_266919

/-- In the expansion of (1 - y / x) * (x + y) ^ 8, the coefficient of x ^ 2 * y ^ 6 is -28. -/
theorem coefficient_x2y6 (x y : ℚ) :  
  coefficient ((1 - y / x) * (x + y)^8) (monomial (2, 6)) = -28 :=
sorry

end coefficient_x2y6_l266_266919


namespace max_consecutive_integers_sum_l266_266399

theorem max_consecutive_integers_sum (n : ℕ) : 
  (∃ n, (n * (n + 1) / 2 < 500) ∧ (∀ m > n, m * (m + 1) / 2 ≥ 500)) :=
by {
  use 31,
  split,
  {
    norm_num,
  },
  {
    intros m hm,
    have h : m = 32 := by linarith,
    rw h,
    norm_num,
  },
  sorry
}

end max_consecutive_integers_sum_l266_266399


namespace number_of_sixth_powers_less_than_1000_l266_266852

theorem number_of_sixth_powers_less_than_1000 : 
  {n : ℕ | ∃ m : ℕ, n = m^6 ∧ n < 1000}.to_finset.card = 3 := 
by {
  sorry
}

end number_of_sixth_powers_less_than_1000_l266_266852


namespace max_consecutive_sum_leq_500_l266_266379

theorem max_consecutive_sum_leq_500 : ∃ n : ℕ, (∑ i in finset.range (n+1), i) ≤ 500 ∧ (∀ m : ℕ, m > n → (∑ i in finset.range (m+1), i) > 500) :=
sorry

end max_consecutive_sum_leq_500_l266_266379


namespace find_original_denominator_l266_266478

theorem find_original_denominator (d : ℕ) (h : (3 + 7) / (d + 7) = 1 / 3) : d = 23 :=
sorry

end find_original_denominator_l266_266478


namespace number_of_triangles_l266_266048

def triangle_count (a b c : ℕ) : Prop :=
  a ≤ b ∧ b ≤ c ∧ a + b + c ≤ 100 ∧ c - a ≤ 2 

theorem number_of_triangles : 
  (∃ n, n = 190 ∧ ∀ a b c : ℕ, triangle_count a b c → 
    [(a, b, c)] ∈ (list.range (a + b + c + 1)).product (list.range (a + b + c + 1)).product (list.range (a + b + c + 1)).nodup) :=
sorry

end number_of_triangles_l266_266048


namespace A_inter_B_eq_l266_266600

open Set

variable (α : Type) [Preorder α] [LinearOrder α]

def A : Set ℝ := { x | x > 1 }
def B : Set ℝ := { x | x^2 < 4 }
def A_inter_B : Set ℝ := { x | 1 < x ∧ x < 2 }

theorem A_inter_B_eq : A α ∩ B α = A_inter_B α :=
by
  sorry

end A_inter_B_eq_l266_266600


namespace total_amount_received_l266_266016

noncomputable def total_books_sold (B : ℕ) := (2 / 3 : ℝ) * B
noncomputable def total_amount (B : ℕ) := total_books_sold B * 2
def books_remaining (B : ℕ) := (1 / 3 : ℝ) * B

theorem total_amount_received : 
  (∀ (B : ℕ), books_remaining B = 36 → total_amount B = 144) :=
by
  intros B h
  sorry

end total_amount_received_l266_266016


namespace max_consecutive_integers_sum_l266_266396

theorem max_consecutive_integers_sum (n : ℕ) : 
  (∃ n, (n * (n + 1) / 2 < 500) ∧ (∀ m > n, m * (m + 1) / 2 ≥ 500)) :=
by {
  use 31,
  split,
  {
    norm_num,
  },
  {
    intros m hm,
    have h : m = 32 := by linarith,
    rw h,
    norm_num,
  },
  sorry
}

end max_consecutive_integers_sum_l266_266396


namespace count_squares_cubes_less_than_1000_l266_266814

theorem count_squares_cubes_less_than_1000 : 
  {n : ℕ | ∃ x : ℕ, n = x^6 ∧ n < 1000}.card = 3 :=
sorry

end count_squares_cubes_less_than_1000_l266_266814


namespace find_ratio_l266_266275

noncomputable def history_pages : ℕ := 160
def geography_pages : ℕ := history_pages + 70
def science_pages : ℕ := 2 * history_pages
def total_pages : ℕ := 905

def math_pages (R: ℚ) : ℕ := R * (history_pages + geography_pages)

theorem find_ratio (R : ℚ) (h : history_pages + geography_pages + math_pages R + science_pages = total_pages) :
  R = 1 / 2 :=
by
  sorry

end find_ratio_l266_266275


namespace num_sixth_powers_below_1000_l266_266799

theorem num_sixth_powers_below_1000 : 
  {n : ℕ | n^6 < 1000 ∧ n > 0}.to_finset.card = 3 :=
by sorry

end num_sixth_powers_below_1000_l266_266799


namespace count_sixth_powers_less_than_1000_l266_266754

theorem count_sixth_powers_less_than_1000 : 
  {n : ℕ | 0 < n ∧ n < 1000 ∧ (∃ k : ℕ, n = k^6)}.to_finset.card = 3 :=
by 
  sorry

end count_sixth_powers_less_than_1000_l266_266754


namespace num_sixth_powers_below_1000_l266_266790

theorem num_sixth_powers_below_1000 : 
  {n : ℕ | n^6 < 1000 ∧ n > 0}.to_finset.card = 3 :=
by sorry

end num_sixth_powers_below_1000_l266_266790


namespace neg_one_pow_2023_l266_266495

theorem neg_one_pow_2023 : (-1 : ℤ) ^ 2023 = -1 := 
by { 
  have h : even 2023 = false := by sorry, -- We would formally prove that 2023 is odd.
  sorry
}

end neg_one_pow_2023_l266_266495


namespace cos_omega_x_3_zeros_interval_l266_266144

theorem cos_omega_x_3_zeros_interval (ω : ℝ) (hω : ω > 0)
  (h3_zeros : ∃ a b c : ℝ, (0 ≤ a ∧ a ≤ 2 * Real.pi) ∧
    (0 ≤ b ∧ b ≤ 2 * Real.pi ∧ b ≠ a) ∧
    (0 ≤ c ∧ c ≤ 2 * Real.pi ∧ c ≠ a ∧ c ≠ b) ∧
    (∀ x : ℝ, (0 ≤ x ∧ x ≤ 2 * Real.pi) →
      (Real.cos (ω * x) - 1 = 0 ↔ x = a ∨ x = b ∨ x = c))) :
  2 ≤ ω ∧ ω < 3 :=
sorry

end cos_omega_x_3_zeros_interval_l266_266144


namespace company_profit_l266_266447

theorem company_profit (sales : ℝ) (profit_first : ℝ) (threshold : ℝ) (profit_rest : ℝ) :
  sales = 6000 ∧ profit_first = 0.06 ∧ threshold = 1000 ∧ profit_rest = 0.05 →
  let profit := threshold * profit_first + (sales - threshold) * profit_rest in
  profit = 310 :=
by 
  intros h
  sorry

end company_profit_l266_266447


namespace number_of_individuals_left_at_zoo_l266_266455

theorem number_of_individuals_left_at_zoo 
  (students_class1 students_class2 students_left : ℕ)
  (initial_chaperones remaining_chaperones teachers : ℕ) :
  students_class1 = 10 ∧
  students_class2 = 10 ∧
  initial_chaperones = 5 ∧
  teachers = 2 ∧
  students_left = 10 ∧
  remaining_chaperones = initial_chaperones - 2 →
  (students_class1 + students_class2 - students_left) + remaining_chaperones + teachers = 15 :=
by
  sorry

end number_of_individuals_left_at_zoo_l266_266455


namespace trapezoid_sides_and_height_l266_266883

def trapezoid_base_height (a h A: ℝ) :=
  (h = (2 * a + 3) / 2) ∧
  (A = a^2 + 3 * a + 9 / 4) ∧
  (A = 2 * a^2 - 7.75)

theorem trapezoid_sides_and_height :
  ∃ (a b h : ℝ), (b = a + 3) ∧
  trapezoid_base_height a h 7.75 ∧
  a = 5 ∧ b = 8 ∧ h = 6.5 :=
by
  sorry

end trapezoid_sides_and_height_l266_266883


namespace part1_part2_part3_l266_266115

variable {R : Type*} [OrderedCommGroup R]

-- Define the function and the given conditions
variable (f : R → R) 
variable (additivity : ∀ x y : R, f(x + y) = f(x) + f(y))
variable (positivity : ∀ x : R, 0 < x → 0 < f(x))

-- Prove f(0) = 0
theorem part1 : f(0) = 0 :=
sorry

-- Prove f(x) is an odd function
theorem part2 : ∀ x : R, f(-x) = -f(x) :=
sorry

-- Solve the inequality f(a - 4) + f(2a + 1) < 0
theorem part3 {a : R} : f(a - 4) + f(2a + 1) < 0 ↔ a < 1 :=
sorry

end part1_part2_part3_l266_266115


namespace counterexample_to_gt_implies_greater_l266_266993

theorem counterexample_to_gt_implies_greater :
  ∃ (a b c : ℝ), (a > b) ∧ ¬ (a * c > b * c) :=
by
  use 1, -1, 0
  split
  · sorry -- Proof that a > b
  · sorry -- Proof that ¬ (a * c > b * c)

end counterexample_to_gt_implies_greater_l266_266993


namespace percentage_cleared_all_sections_l266_266196

def total_candidates : ℝ := 1200
def cleared_none : ℝ := 0.05 * total_candidates
def cleared_one_section : ℝ := 0.25 * total_candidates
def cleared_four_sections : ℝ := 0.20 * total_candidates
def cleared_two_sections : ℝ := 0.245 * total_candidates
def cleared_three_sections : ℝ := 300

-- Let x be the percentage of candidates who cleared all sections
def cleared_all_sections (x: ℝ) : Prop :=
  let total_cleared := (cleared_none + 
                        cleared_one_section + 
                        cleared_four_sections + 
                        cleared_two_sections + 
                        cleared_three_sections + 
                        x * total_candidates / 100)
  total_cleared = total_candidates

theorem percentage_cleared_all_sections :
  ∃ x, cleared_all_sections x ∧ x = 0.5 :=
by
  sorry

end percentage_cleared_all_sections_l266_266196


namespace both_square_and_cube_count_both_square_cube_less_than_1000_l266_266786

theorem both_square_and_cube (n : ℕ) (h_positive : n > 0) (h_bound : n < 1000) (h_square : ∃ k : ℕ, n = k^2) (h_cube : ∃ k : ℕ, n = k^3) :
  n ∈ {1, 64, 729} :=
sorry

theorem count_both_square_cube_less_than_1000 :
  {n : ℕ | n > 0 ∧ n < 1000 ∧ ∃ k : ℕ, n = k^2 ∧ ∃ k : ℕ, n = k^3}.to_finset.card = 3 :=
by
  have h_valid : ∀ n ∈ {1, 64, 729}, n > 0 ∧ n < 1000 ∧ (∃ k : ℕ, n = k^2) ∧ (∃ k : ℕ, n = k^3) := sorry
  have h_reverse : ∀ n, n > 0 ∧ n < 1000 ∧ (∃ k : ℕ, n = k^2) ∧ (∃ k : ℕ, n = k^3) → n ∈ {1, 64, 729} := sorry
  exact Finset.card_map _ {1, 64, 729}.to_finset
  sorry

end both_square_and_cube_count_both_square_cube_less_than_1000_l266_266786


namespace sum_of_x_coordinates_l266_266072

noncomputable def g (x : ℝ) : ℝ :=
if x ≤ -3 then 0 else
if x ≤ -1 then 2 * (x + 3) - 4 else
if x ≤ 0 then -x - 1 else
if x ≤ 2 then 2 * x - 1 else
-x + 5

theorem sum_of_x_coordinates : ∑ p in [{-1.5}].to_finset, p = -1.5 :=
by
  sorry

end sum_of_x_coordinates_l266_266072


namespace concurrency_and_sum_xd_xe_xf_2xa_xb_xc_l266_266483

-- Definitions for points A, B, and C
variables {A B C D E F X : Type}
-- Triangle conditions
variables [h1 : ι Triangle (ABC)] (h2 : ∀ α ∈ angles A B C, α < 120)

-- Defining Equilateral Triangles (with vertices on the outside of triangle ABC)
def equilateral_triangle (a b c : Type) : Prop :=
  ∀ x y z : Type, dist x y = dist y z ∧ dist y z = dist z x ∧ dist z x = dist x y

variables [h3 : equilateral_triangle A B D]
variables [h4 : equilateral_triangle B C E]
variables [h5 : equilateral_triangle C A F]

-- Theorem statement
theorem concurrency_and_sum_xd_xe_xf_2xa_xb_xc :
  (∃ X, meets_at_point (line.from_points X D) (line.from_points X A) (line.from_points X E) (line.from_points X B) (line.from_points X F) (line.from_points X C))
  ∧ (distance(X, D) + distance(X, E) + distance(X, F) = 2 * (distance(X, A) + distance(X, B) + distance(X, C))) :=
sorry

end concurrency_and_sum_xd_xe_xf_2xa_xb_xc_l266_266483


namespace intersection_A_B_l266_266570

-- Define the sets A and B based on given conditions
def A : Set ℝ := {x | -2 < x ∧ x < 4}
def B : Set ℝ := {x | -1 < x}

-- The statement to prove
theorem intersection_A_B : (A ∩ B) = {x | -1 < x ∧ x < 4} :=
  sorry

end intersection_A_B_l266_266570


namespace sqrt_four_eq_two_l266_266499

theorem sqrt_four_eq_two : ∃ x : ℝ, x^2 = 4 ∧ x = 2 :=
by
  use 2
  split
  { norm_num }
  { refl }

end sqrt_four_eq_two_l266_266499


namespace quadratic_range_l266_266599

theorem quadratic_range (x y : ℝ) (h1 : y = -(x - 5) ^ 2 + 1) (h2 : 2 < x ∧ x < 6) :
  -8 < y ∧ y ≤ 1 := 
sorry

end quadratic_range_l266_266599


namespace find_d_l266_266974

-- Define AP terms as S_n = a + (n-1)d, sum of first 10 terms, and difference expression
def arithmetic_progression (S : ℕ → ℕ) (a d : ℕ) : Prop :=
  ∀ n, S n = a + (n - 1) * d

def sum_first_ten (S : ℕ → ℕ) : Prop :=
  (S 1) + (S 2) + (S 3) + (S 4) + (S 5) + (S 6) + (S 7) + (S 8) + (S 9) + (S 10) = 55

def difference_expression (S : ℕ → ℕ) (d : ℕ) : Prop :=
  (S 10 - S 8) + (S 9 - S 7) + (S 8 - S 6) + (S 7 - S 5) + (S 6 - S 4) +
  (S 5 - S 3) + (S 4 - S 2) + (S 3 - S 1) = d

theorem find_d : ∃ (d : ℕ) (S : ℕ → ℕ) (a : ℕ), 
  (∀ n, S n = a + (n - 1) * d) ∧ 
  (S 1) + (S 2) + (S 3) + (S 4) + (S 5) + (S 6) + (S 7) + (S 8) + (S 9) + (S 10) = 55 ∧
  (S 10 - S 8) + (S 9 - S 7) + (S 8 - S 6) + (S 7 - S 5) + (S 6 - S 4) +
  (S 5 - S 3) + (S 4 - S 2) + (S 3 - S 1) = 16 :=
by
  sorry  -- proof is not required

end find_d_l266_266974


namespace distance_between_foci_of_hyperbola_l266_266311

theorem distance_between_foci_of_hyperbola (x y : ℝ) (h : x * y = 4) : 
  distance (2, 2) (-2, -2) = 8 :=
sorry

end distance_between_foci_of_hyperbola_l266_266311


namespace math_problem_l266_266556

variable (a : ℝ) (m n : ℝ)

theorem math_problem
  (h1 : a^m = 3)
  (h2 : a^n = 2) :
  a^(2*m + 3*n) = 72 := 
  sorry

end math_problem_l266_266556


namespace count_squares_and_cubes_less_than_1000_l266_266661

theorem count_squares_and_cubes_less_than_1000 : 
  {n : ℕ // n^6 < 1000}.card = 3 :=
by
  sorry

end count_squares_and_cubes_less_than_1000_l266_266661


namespace probability_of_x_plus_y_leq_6_l266_266026

theorem probability_of_x_plus_y_leq_6 : 
  let S := { p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 4 ∧ 0 ≤ p.2 ∧ p.2 ≤ 5 } in
  let T := { p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 4 ∧ 0 ≤ p.2 ∧ p.2 ≤ 5 ∧ p.1 + p.2 ≤ 6 } in
  (Set.cardinal T / Set.cardinal S : ℝ) = 1 / 2 :=
by
  sorry

end probability_of_x_plus_y_leq_6_l266_266026


namespace f_zero_in_interval_l266_266953

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation : ∀ x : ℝ, f (2 * x^2 - 1) = 2 * x * f x
axiom continuity : continuous f

theorem f_zero_in_interval : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → f x = 0 :=
  sorry

end f_zero_in_interval_l266_266953


namespace both_square_and_cube_count_both_square_cube_less_than_1000_l266_266783

theorem both_square_and_cube (n : ℕ) (h_positive : n > 0) (h_bound : n < 1000) (h_square : ∃ k : ℕ, n = k^2) (h_cube : ∃ k : ℕ, n = k^3) :
  n ∈ {1, 64, 729} :=
sorry

theorem count_both_square_cube_less_than_1000 :
  {n : ℕ | n > 0 ∧ n < 1000 ∧ ∃ k : ℕ, n = k^2 ∧ ∃ k : ℕ, n = k^3}.to_finset.card = 3 :=
by
  have h_valid : ∀ n ∈ {1, 64, 729}, n > 0 ∧ n < 1000 ∧ (∃ k : ℕ, n = k^2) ∧ (∃ k : ℕ, n = k^3) := sorry
  have h_reverse : ∀ n, n > 0 ∧ n < 1000 ∧ (∃ k : ℕ, n = k^2) ∧ (∃ k : ℕ, n = k^3) → n ∈ {1, 64, 729} := sorry
  exact Finset.card_map _ {1, 64, 729}.to_finset
  sorry

end both_square_and_cube_count_both_square_cube_less_than_1000_l266_266783


namespace minimum_value_expression_l266_266235

open Real

theorem minimum_value_expression : ∃ x : ℝ, (x + 1) * (x + 2) * (x + 3) * (x + 4) + 2019 = 2018 := 
sorry

end minimum_value_expression_l266_266235


namespace beth_finishes_first_l266_266051

open Real

noncomputable def andy_lawn_area : ℝ := sorry
noncomputable def beth_lawn_area : ℝ := andy_lawn_area / 3
noncomputable def carlos_lawn_area : ℝ := andy_lawn_area / 4

noncomputable def andy_mowing_rate : ℝ := sorry
noncomputable def beth_mowing_rate : ℝ := andy_mowing_rate
noncomputable def carlos_mowing_rate : ℝ := andy_mowing_rate / 2

noncomputable def carlos_break : ℝ := 10

noncomputable def andy_mowing_time := andy_lawn_area / andy_mowing_rate
noncomputable def beth_mowing_time := beth_lawn_area / beth_mowing_rate
noncomputable def carlos_mowing_time := (carlos_lawn_area / carlos_mowing_rate) + carlos_break

theorem beth_finishes_first :
  beth_mowing_time < andy_mowing_time ∧ beth_mowing_time < carlos_mowing_time := by
  sorry

end beth_finishes_first_l266_266051


namespace maximum_consecutive_positive_integers_sum_500_l266_266386

theorem maximum_consecutive_positive_integers_sum_500 : 
  ∃ n : ℕ, (n * (n + 1) / 2 < 500) ∧ (∀ m : ℕ, (m * (m + 1) / 2 < 500) → m ≤ n) :=
sorry

end maximum_consecutive_positive_integers_sum_500_l266_266386


namespace both_square_and_cube_count_both_square_cube_less_than_1000_l266_266777

theorem both_square_and_cube (n : ℕ) (h_positive : n > 0) (h_bound : n < 1000) (h_square : ∃ k : ℕ, n = k^2) (h_cube : ∃ k : ℕ, n = k^3) :
  n ∈ {1, 64, 729} :=
sorry

theorem count_both_square_cube_less_than_1000 :
  {n : ℕ | n > 0 ∧ n < 1000 ∧ ∃ k : ℕ, n = k^2 ∧ ∃ k : ℕ, n = k^3}.to_finset.card = 3 :=
by
  have h_valid : ∀ n ∈ {1, 64, 729}, n > 0 ∧ n < 1000 ∧ (∃ k : ℕ, n = k^2) ∧ (∃ k : ℕ, n = k^3) := sorry
  have h_reverse : ∀ n, n > 0 ∧ n < 1000 ∧ (∃ k : ℕ, n = k^2) ∧ (∃ k : ℕ, n = k^3) → n ∈ {1, 64, 729} := sorry
  exact Finset.card_map _ {1, 64, 729}.to_finset
  sorry

end both_square_and_cube_count_both_square_cube_less_than_1000_l266_266777


namespace count_squares_and_cubes_less_than_1000_l266_266731

theorem count_squares_and_cubes_less_than_1000 :
  {n : ℕ | 0 < n ∧ n < 1000 ∧ ∃ x, n = x ^ 6}.to_finset.card = 3 :=
by sorry

end count_squares_and_cubes_less_than_1000_l266_266731


namespace sum_of_bases_l266_266887

theorem sum_of_bases (R1 R2 : ℕ)
  (h1 : ∀ F1 : ℚ, F1 = (4 * R1 + 8) / (R1 ^ 2 - 1) → F1 = (5 * R2 + 9) / (R2 ^ 2 - 1))
  (h2 : ∀ F2 : ℚ, F2 = (8 * R1 + 4) / (R1 ^ 2 - 1) → F2 = (9 * R2 + 5) / (R2 ^ 2 - 1)) :
  R1 + R2 = 24 :=
sorry

end sum_of_bases_l266_266887


namespace minimal_sum_of_digits_l266_266403

-- Define the function f as given in the problem
def f (n : ℕ) : ℕ := 17 * n^2 - 11 * n + 1

-- Define a function to calculate the sum of digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  (n.toString.data.map (λ c => c.toNat - '0'.toNat)).sum

-- State the theorem that the minimal sum of digits in the decimal representation of f(n) is 2
theorem minimal_sum_of_digits : (∀ n : ℕ, sum_of_digits (f n)) = 2 :=
begin
  sorry
end

end minimal_sum_of_digits_l266_266403


namespace octopus_leg_count_l266_266064

theorem octopus_leg_count :
  let num_initial_octopuses := 5
  let legs_per_normal_octopus := 8
  let num_removed_octopuses := 2
  let legs_first_mutant := 10
  let legs_second_mutant := 6
  let legs_third_mutant := 2 * legs_per_normal_octopus
  let num_initial_legs := num_initial_octopuses * legs_per_normal_octopus
  let num_removed_legs := num_removed_octopuses * legs_per_normal_octopus
  let num_mutant_legs := legs_first_mutant + legs_second_mutant + legs_third_mutant
  num_initial_legs - num_removed_legs + num_mutant_legs = 56 :=
by
  -- proof to be filled in later
  sorry

end octopus_leg_count_l266_266064


namespace four_digit_sum_seven_l266_266544

theorem four_digit_sum_seven : 
  (∃ (abcd : ℕ×ℕ×ℕ×ℕ), let (a, b, c, d) := abcd in 
    a ∈ finset.range 1 10 ∧ b ∈ finset.range 0 10 ∧ c ∈ finset.range 0 10 ∧ d ∈ finset.range 0 10 ∧ 
    a + b + c + d = 7) ↔ 
  (finset.card (finset.filter (λ abcd : ℕ×ℕ×ℕ×ℕ, 
    let (a, b, c, d) := abcd in 
      a ∈ finset.range 1 10 ∧ b ∈ finset.range 0 10 ∧ 
      c ∈ finset.range 0 10 ∧ d ∈ finset.range 0 10 ∧ 
      a + b + c + d = 7) 
    (finset.product (finset.range 10) (finset.product (finset.range 10) (finset.product (finset.range 10) (finset.range 10))) )) = 84) :=
sorry

end four_digit_sum_seven_l266_266544


namespace net_price_change_l266_266426

theorem net_price_change (P : ℝ) : 
  let decreased_price := P * (1 - 0.30)
  let increased_price := decreased_price * (1 + 0.20)
  increased_price - P = -0.16 * P :=
by
  -- The proof would go here. We just need the statement as per the prompt.
  sorry

end net_price_change_l266_266426


namespace ab_diff_eq_minus_seven_l266_266174

theorem ab_diff_eq_minus_seven (a b : ℝ) (h₁ : sqrt (a^2) = 3) (h₂ : sqrt b = 2) (h₃ : a * b < 0) : a - b = -7 :=
sorry

end ab_diff_eq_minus_seven_l266_266174


namespace necessary_and_sufficient_parallel_l266_266003

-- Define the equations of the lines
def line1 (m : ℝ) : ℝ × ℝ → Prop := λ p, (3 * m - 4) * p.1 + 4 * p.2 - 2 = 0
def line2 (m : ℝ) : ℝ × ℝ → Prop := λ p, m * p.1 + 2 * p.2 - 2 = 0

-- Define the slopes of the lines
def slope1 (m : ℝ) : ℝ := -(3 * m - 4) / 4
def slope2 (m : ℝ) : ℝ := -m / 2

-- Define the property of parallelism
def are_parallel (m : ℝ) : Prop := slope1 m = slope2 m

-- The proof statement that m = 4 is a necessary and sufficient condition for the lines to be parallel
theorem necessary_and_sufficient_parallel (m : ℝ) : (3 * m - 4) / 4 = m / 2 ↔ m = 4 := by
  sorry

end necessary_and_sufficient_parallel_l266_266003


namespace range_of_a_l266_266141

open Real

section

variables (a : ℝ)

def prop_p (a : ℝ) : Prop :=
  ∀ x : ℝ, x > 0 → x ≥ a * sqrt x - 1

def prop_q (a : ℝ) : Prop :=
  a > 0 ∧ a ≠ 1 ∧ ∃ m : ℝ, ∀ x : ℝ, (x^2 - a * x + 1) > 0 ∧ log a (x^2 - a * x + 1) ≥ m

theorem range_of_a : (∃ p : Prop, ∃ q : Prop, (prop_p a ∨ prop_q a) ∧ ¬ (prop_p a ∧ prop_q a) → (a = 2 ∨ a ≤ 1))
:= sorry

end

end range_of_a_l266_266141


namespace find_c_k_l266_266326

theorem find_c_k (a b : ℕ → ℕ) (c : ℕ → ℕ) (k : ℕ) (d r : ℕ) 
  (h1 : ∀ n, a n = 1 + (n-1)*d)
  (h2 : ∀ n, b n = r^(n-1))
  (h3 : ∀ n, c n = a n + b n)
  (h4 : c (k-1) = 80)
  (h5 : c (k+1) = 500) :
  c k = 167 := sorry

end find_c_k_l266_266326


namespace square_area_64_l266_266491

theorem square_area_64 (x : ℝ) (h1 : x = 8) (h2 : ∀ l w : ℝ, l = 2w → 2 * (l + w) = 20) (h3 : ∀ (a b : ℝ), a^2 = b → a = 8 → ∃ c : ℝ, c = b) : ∃ area : ℝ, area = 64 :=
by
  sorry

end square_area_64_l266_266491


namespace triangles_satisfying_eq_l266_266414

theorem triangles_satisfying_eq (a b c R : ℝ) (h_triangle : a ^ 2 + b ^ 2 + c ^ 2 = 8 * R ^ 2) :
  (exists A B C : ℝ, a = (dist A B) ∧ b = (dist B C) ∧ c = (dist C A) ∧ is_right_triangle A B C) :=
sorry

end triangles_satisfying_eq_l266_266414


namespace both_square_and_cube_count_both_square_cube_less_than_1000_l266_266782

theorem both_square_and_cube (n : ℕ) (h_positive : n > 0) (h_bound : n < 1000) (h_square : ∃ k : ℕ, n = k^2) (h_cube : ∃ k : ℕ, n = k^3) :
  n ∈ {1, 64, 729} :=
sorry

theorem count_both_square_cube_less_than_1000 :
  {n : ℕ | n > 0 ∧ n < 1000 ∧ ∃ k : ℕ, n = k^2 ∧ ∃ k : ℕ, n = k^3}.to_finset.card = 3 :=
by
  have h_valid : ∀ n ∈ {1, 64, 729}, n > 0 ∧ n < 1000 ∧ (∃ k : ℕ, n = k^2) ∧ (∃ k : ℕ, n = k^3) := sorry
  have h_reverse : ∀ n, n > 0 ∧ n < 1000 ∧ (∃ k : ℕ, n = k^2) ∧ (∃ k : ℕ, n = k^3) → n ∈ {1, 64, 729} := sorry
  exact Finset.card_map _ {1, 64, 729}.to_finset
  sorry

end both_square_and_cube_count_both_square_cube_less_than_1000_l266_266782


namespace count_squares_and_cubes_l266_266650

theorem count_squares_and_cubes (bound : ℕ) (hk : bound = 1000) : 
  Nat.card {n : ℕ | n < bound ∧ (∃ k : ℕ, n = k ^ 6)} = 3 := 
by
  sorry

end count_squares_and_cubes_l266_266650


namespace video_game_map_width_l266_266043

theorem video_game_map_width (volume length height : ℝ) (h1 : volume = 50)
                            (h2 : length = 5) (h3 : height = 2) :
  ∃ width : ℝ, volume = length * width * height ∧ width = 5 :=
by
  sorry

end video_game_map_width_l266_266043


namespace transform_sine_function_l266_266592

theorem transform_sine_function :
  (∀ x : ℝ, sin (x + π / 3) = sin (1 / 2 * x + π / 3)) :=
begin
  sorry
end

end transform_sine_function_l266_266592


namespace infinite_solutions_l266_266976

noncomputable def numFactorsOfTwo (m : ℕ) : ℕ :=
  if m = 0 then 0
  else m / 2 + numFactorsOfTwo (m / 2)

theorem infinite_solutions : ∃ᶠ m in at_top, (m : ℕ) - numFactorsOfTwo m = 1989 := 
begin
  sorry
end

end infinite_solutions_l266_266976


namespace graph_regions_count_l266_266170

theorem graph_regions_count : 
  ∀ x y : ℝ, (x^6 - x^5 + 3*x^4*y^2 + 10*x^3*y^2 + 3*x^2*y^4 - 5*x*y^4 + y^6 = 0) →
  (number of bounded regions by this graph = 5) :=
by
  sorry

end graph_regions_count_l266_266170


namespace draw_black_then_white_with_replacement_probability_y_minus_x_gt_2_without_replacement_l266_266189

-- Definitions for the problem conditions
def isBlackBall (n : ℕ) : Prop := n = 1 ∨ n = 2
def isWhiteBall (n : ℕ) : Prop := n = 3 ∨ n = 4 ∨ n = 5
def totalBalls : set ℕ := {1, 2, 3, 4, 5}

-- Part (1) Lean statement: Probability of drawing a black ball first and then a white ball with replacement
theorem draw_black_then_white_with_replacement :
  let PA := (2 / 5 : ℚ)
  let PB := (3 / 5 : ℚ)
  PA * PB = 6 / 25 :=
by
  sorry

-- Part (2) Lean statement: Probability that y - x > 2 when drawing two balls without replacement
theorem probability_y_minus_x_gt_2_without_replacement :
  let sample_space : set (ℕ × ℕ) := {(1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (4, 5)}
  let favorable_outcomes : set (ℕ × ℕ) := {(1, 4), (1, 5), (2, 5)}
  ((favorable_outcomes.card : ℚ) / sample_space.card) = 3 / 10 :=
by
  sorry

end draw_black_then_white_with_replacement_probability_y_minus_x_gt_2_without_replacement_l266_266189


namespace ratio_of_squares_l266_266477

-- Define lengths of the triangle sides
def a : ℝ := 5
def b : ℝ := 12
def c : ℝ := 13

-- Define side lengths of the squares inscribed in the given manner
def x : ℝ := 25 / 37
def y : ℝ := 65 / 17

-- Statement to prove the ratio of the side lengths of the squares equals 85 / 481
theorem ratio_of_squares : x / y = 85 / 481 := by 
  sorry

end ratio_of_squares_l266_266477


namespace class_size_l266_266417

theorem class_size :
  ∃ (N : ℕ), (20 ≤ N) ∧ (N ≤ 30) ∧ (∃ (x : ℕ), N = 3 * x + 1) ∧ (∃ (y : ℕ), N = 4 * y + 1) ∧ (N = 25) :=
by { sorry }

end class_size_l266_266417


namespace cassie_and_brian_meet_at_1111am_l266_266501

theorem cassie_and_brian_meet_at_1111am :
  ∃ t : ℕ, t = 11*60 + 11 ∧
    (∃ x : ℚ, x = 51/16 ∧ 
      14 * x + 18 * (x - 1) = 84) :=
sorry

end cassie_and_brian_meet_at_1111am_l266_266501


namespace number_of_sixth_powers_lt_1000_l266_266704

-- Define the condition for being a sixth power and being less than 1000
def isSixthPower (n : ℕ) : Prop := ∃ k : ℕ, n = k^6

-- Define the main statement to be proved
theorem number_of_sixth_powers_lt_1000 : 
  {n : ℕ | n < 1000 ∧ isSixthPower n}.to_finset.card = 3 :=
by
  sorry

end number_of_sixth_powers_lt_1000_l266_266704


namespace carter_picks_green_mms_25_percent_l266_266065

noncomputable def carter_jar_initial := {green := 35, red := 25, blue := 10, orange := 15}
noncomputable def carter_jar_after_carter := {green := carter_jar_initial.green - 20, red := carter_jar_initial.red - 8, blue := carter_jar_initial.blue, orange := carter_jar_initial.orange}
noncomputable def carter_jar_after_sister := {green := carter_jar_after_carter.green, red := carter_jar_after_carter.red / 2, blue := carter_jar_after_carter.blue - 5, orange := carter_jar_after_carter.orange, yellow := 14}
noncomputable def carter_jar_after_alex := {green := carter_jar_after_sister.green, red := carter_jar_after_sister.red, blue := carter_jar_after_sister.blue, orange := carter_jar_after_sister.orange - 7, yellow := carter_jar_after_sister.yellow - 3, purple := 8}
noncomputable def carter_jar_final := {green := carter_jar_after_alex.green, red := carter_jar_after_alex.red, blue := 0, orange := carter_jar_after_alex.orange, yellow := carter_jar_after_alex.yellow, purple := carter_jar_after_alex.purple, brown := 10}

def total_mms := carter_jar_final.green + carter_jar_final.red + carter_jar_final.orange + carter_jar_final.yellow + carter_jar_final.purple + carter_jar_final.brown 

def chance_green := carter_jar_final.green.to_rat / total_mms.to_rat

def percentage_chance_green := (chance_green * 100).to_nat -- Convert to natural number percentage

theorem carter_picks_green_mms_25_percent : percentage_chance_green = 25 := by 
  sorry

end carter_picks_green_mms_25_percent_l266_266065


namespace range_of_m_for_quadratic_eqn_with_real_roots_l266_266139

theorem range_of_m_for_quadratic_eqn_with_real_roots (m : ℝ) :
  (∀ (x : ℝ), (0 ≤ 2 * x^2 - m * x + 1)) ∧ 1 / 2 < m / 4 ∧ m / 4 < 4 ∧ m^2 - 8 > 0 →
  m ∈ set.Ioc (2 * real.sqrt 2) 3 :=
by
  sorry

end range_of_m_for_quadratic_eqn_with_real_roots_l266_266139


namespace quadratic_function_properties_l266_266293

theorem quadratic_function_properties :
  let f : ℝ → ℝ := λ x, x^2 - 3 * x - 4
  f(-2) = 6 ∧ f(0) = -4 ∧ f(1) = -6 →
  (∀ x, x > 1.5 → f(x) > f(1.5)) ∧ -- Conclusion ③
  (∃ a b c : ℝ, a > 0 ∧ b = -3 ∧ c = -4 ∧ f = λ x, a * x^2 + b * x + c) ∧ -- Conclusion ①
  ¬ (∀ a b c : ℝ, (f = λ x, a * x^2 + b * x + c) → x = 1) ∧ -- Negation of Conclusion ②
  ¬ (∃ x : ℝ, x > 4 ∧ (x^2 - 3 * x - 4) = 0) -- Negation of Conclusion ④
  :=
by {
  assume f_def,
  sorry
}

end quadratic_function_properties_l266_266293


namespace count_squares_cubes_less_than_1000_l266_266807

theorem count_squares_cubes_less_than_1000 : 
  {n : ℕ | ∃ x : ℕ, n = x^6 ∧ n < 1000}.card = 3 :=
sorry

end count_squares_cubes_less_than_1000_l266_266807


namespace multiply_polynomials_l266_266981

variable {x y z : ℝ}

theorem multiply_polynomials :
  (3 * x^4 - 4 * y^3 - 6 * z^2) * (9 * x^8 + 16 * y^6 + 36 * z^4 + 12 * x^4 * y^3 + 18 * x^4 * z^2 + 24 * y^3 * z^2)
  = 27 * x^12 - 64 * y^9 - 216 * z^6 - 216 * x^4 * y^3 * z^2 := by {
  sorry
}

end multiply_polynomials_l266_266981


namespace count_squares_and_cubes_less_than_1000_l266_266673

theorem count_squares_and_cubes_less_than_1000 : 
  {n : ℕ // n^6 < 1000}.card = 3 :=
by
  sorry

end count_squares_and_cubes_less_than_1000_l266_266673


namespace fractional_part_a_l266_266960

noncomputable def fractional_part (x : ℝ) : ℝ := x - x.floor

theorem fractional_part_a (a b : ℝ) (h₁ : a = (5 * Real.sqrt 2 + 7) ^ 2017)
  (h₂ : b = (5 * Real.sqrt 2 - 7) ^ 2017) (h₃ : 0 < b ∧ b < 1) :
  a * fractional_part a = 1 :=
sorry

end fractional_part_a_l266_266960


namespace age_ratio_in_2_years_is_2_1_l266_266468

-- Define the ages and conditions
def son_age (current_year : ℕ) : ℕ := 20
def man_age (current_year : ℕ) : ℕ := son_age current_year + 22

def son_age_in_2_years (current_year : ℕ) : ℕ := son_age current_year + 2
def man_age_in_2_years (current_year : ℕ) : ℕ := man_age current_year + 2

-- The theorem stating the ratio of the man's age to the son's age in two years is 2:1
theorem age_ratio_in_2_years_is_2_1 (current_year : ℕ) :
  man_age_in_2_years current_year = 2 * son_age_in_2_years current_year :=
by
  sorry

end age_ratio_in_2_years_is_2_1_l266_266468


namespace solution_l266_266970

noncomputable def problem_statement (n : ℕ) (hn : n > 1) (a : Fin n → ℝ) (x y : Fin n → ℝ) : Prop :=
  let P := ∏ i in Finset.univ, (X - C (a i))
  let Q := polynomial.derivative P
  (1 / n) * ∑ k in Finset.range n, x k ^ 2 > (1 / (n - 1)) * ∑ k in Finset.range (n - 1), y k ^ 2

theorem solution : ∀ (n : ℕ) (hn : n > 1) (a : Fin n → ℝ) (x y : Fin n → ℝ),
  (∀ k : Fin n, is_root (P n a) (x k)) →
  (∀ k : Fin (n-1), is_root (Q n a) (y k)) →
  (problem_statement n hn a x y) :=
sorry

end solution_l266_266970


namespace number_of_selections_l266_266510

-- Definitions
def grid_size : ℕ := 19
def selected_cells : ℕ := 99

-- Theorem Statement
theorem number_of_selections (n : ℕ) (h : n = 10) :
  let grid := (2 * n - 1) in
  let cells := grid * grid in
  ∃ (selections : ℕ), selections = 1000 ∧
  ∀ selected_positions : set (fin grid × fin grid),
    selected_positions.card = 99 →
    (∀ c₁ c₂ : fin grid × fin grid,
      c₁ ∈ selected_positions →
      c₂ ∈ selected_positions →
      (c₁ ≠ c₂ ∧ ¬ adjacent c₁ c₂)) :=
begin
  sorry
end

-- Definition for adjacency
def adjacent (c1 c2 : fin grid_size × fin grid_size) : Prop :=
  (abs (c1.1 - c2.1) ≤ 1 ∧ abs (c1.2 - c2.2) ≤ 1) ∧ c1 ≠ c2

end number_of_selections_l266_266510


namespace count_positive_integers_square_and_cube_lt_1000_l266_266765

theorem count_positive_integers_square_and_cube_lt_1000 : 
  {n : ℕ | n < 1000 ∧ ∃ k : ℕ, n = k^6}.to_finset.card = 3 :=
by
  sorry

end count_positive_integers_square_and_cube_lt_1000_l266_266765


namespace payment_to_y_l266_266428

theorem payment_to_y (X Y : ℝ) (h1 : X = 1.2 * Y) (h2 : X + Y = 580) : Y = 263.64 :=
by
  sorry

end payment_to_y_l266_266428


namespace four_p_minus_three_is_square_l266_266955

theorem four_p_minus_three_is_square
  (n : ℕ) (p : ℕ)
  (hn_pos : n > 1)
  (hp_prime : Prime p)
  (h1 : n ∣ (p - 1))
  (h2 : p ∣ (n^3 - 1)) : ∃ k : ℕ, 4 * p - 3 = k^2 := sorry

end four_p_minus_three_is_square_l266_266955


namespace minimum_length_ABC_l266_266201

def A := (-3, -4)
def C := (1.5, -2)
def B (xB : ℝ) := (xB, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def L (xB : ℝ) : ℝ :=
  distance A (B xB) + distance (B xB) C

theorem minimum_length_ABC : ∃ xB : ℝ, L xB = 7.5 :=
begin
  use 0,
  simp [L, A, B, C, distance],
  norm_num,
end

end minimum_length_ABC_l266_266201


namespace alice_has_ball_after_three_turns_is_1_l266_266045

noncomputable def probability_alice_has_ball_after_three_turns : ℝ :=
  let p_ab := 2/3   -- Probability Alice passes to Bob
  let p_aa := 1/3   -- Probability Alice keeps the ball
  let p_bc := 1/4   -- Probability Bob passes to Carol
  let p_ba := 3/4   -- Probability Bob passes to Alice
  let p_ca := 1/2   -- Probability Carol passes to Alice
  let p_cb := 1/2   -- Probability Carol passes to Bob in
  have pa_turn1_to_bob := p_ab, -- Probability Alice passes to Bob in turn 1
  have pb_turn2_to_carol := p_bc, -- Probability Bob passes to Carol in turn 2
  have pc_turn3_to_alice := p_ca, -- Probability Carol passes to Alice in turn 3
  have pb_turn2_to_alice := p_ba, -- Probability Bob passes to Alice in turn 2
  have pa_turn1_to_self := p_aa, -- Probability Alice keeps the ball in turn 1
  have total_probability := (p_ab * p_bc * p_ca) + (p_ab * p_ba) + p_aa
  total_probability

theorem alice_has_ball_after_three_turns_is_1 :
  probability_alice_has_ball_after_three_turns = 1 :=
sorry

end alice_has_ball_after_three_turns_is_1_l266_266045


namespace num_sixth_powers_below_1000_l266_266792

theorem num_sixth_powers_below_1000 : 
  {n : ℕ | n^6 < 1000 ∧ n > 0}.to_finset.card = 3 :=
by sorry

end num_sixth_powers_below_1000_l266_266792


namespace count_sixth_powers_less_than_1000_l266_266748

theorem count_sixth_powers_less_than_1000 : 
  {n : ℕ | 0 < n ∧ n < 1000 ∧ (∃ k : ℕ, n = k^6)}.to_finset.card = 3 :=
by 
  sorry

end count_sixth_powers_less_than_1000_l266_266748


namespace min_books_required_l266_266550

variable (Books : Type) (Person : Type)
variable (buys : Person → Finset Books)

theorem min_books_required (people : Finset Person)
    (h1 : people.card = 4)
    (h2 : ∀ p ∈ people, (buys p).card = 4)
    (h3 : ∀ p1 p2 : Person, p1 ≠ p2 → p1 ∈ people → p2 ∈ people → ((buys p1) ∩ (buys p2)).card = 2) :
    ∃ (S : Finset Books), (S.card = 7 ∧ ∀ p ∈ people, (buys p) ⊆ S) :=
sorry

end min_books_required_l266_266550


namespace min_value_expression_l266_266113

theorem min_value_expression (x y : ℝ) (h1 : x + y = 1) (h2 : y > 0) (h3 : x > 0) :
  ∃ (z : ℝ), z = (1 / (2 * x) + x / (y + 1)) ∧ z = 5 / 4 :=
sorry

end min_value_expression_l266_266113


namespace conditions_for_star_commute_l266_266521

-- Define the operation star
def star (a b : ℝ) : ℝ := a^3 * b^2 - a * b^3

-- Theorem stating the equivalence
theorem conditions_for_star_commute :
  ∀ (x y : ℝ), (star x y = star y x) ↔ (x = 0 ∨ y = 0 ∨ x = y ∨ x = -y) :=
sorry

end conditions_for_star_commute_l266_266521


namespace four_digit_sum_to_seven_l266_266545

theorem four_digit_sum_to_seven :
  (∃ a b c d : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9 ∧ a + b + c + d = 7) ↔ (84) := 
sorry

end four_digit_sum_to_seven_l266_266545


namespace alchemerion_age_problem_l266_266487

theorem alchemerion_age_problem 
  (A S F : ℕ)
  (h1 : A = 3 * S)
  (h2 : F = 2 * A + 40)
  (h3 : A = 360) :
  A + S + F = 1240 :=
by 
  sorry

end alchemerion_age_problem_l266_266487


namespace number_of_sixth_powers_less_than_1000_l266_266838

theorem number_of_sixth_powers_less_than_1000 : 
  {n : ℕ | ∃ m : ℕ, n = m^6 ∧ n < 1000}.to_finset.card = 3 := 
by {
  sorry
}

end number_of_sixth_powers_less_than_1000_l266_266838


namespace coefficient_x2_y6_in_expansion_l266_266931

theorem coefficient_x2_y6_in_expansion :
  ∀ (x y : ℝ), x ≠ 0 → 
    (let c8_6 := (nat.factorial 8) / ((nat.factorial 6) * (nat.factorial 2)),
         c8_5 := (nat.factorial 8) / ((nat.factorial 5) * (nat.factorial 3)) in
     c8_6 - c8_5 = -28) :=
by
  intros x y hx_ne_zero
  let c8_6 := (nat.factorial 8) / ((nat.factorial 6) * (nat.factorial 2))
  let c8_5 := (nat.factorial 8) / ((nat.factorial 5) * (nat.factorial 3))
  have h : c8_6 - c8_5 = -28 := sorry
  exact h

end coefficient_x2_y6_in_expansion_l266_266931


namespace number_of_sixth_powers_less_than_1000_l266_266631

theorem number_of_sixth_powers_less_than_1000 : 
  {n : ℕ | n < 1000 ∧ ∃ (k : ℕ), n = k^6}.finite.card = 3 :=
by sorry

end number_of_sixth_powers_less_than_1000_l266_266631


namespace count_squares_and_cubes_l266_266654

theorem count_squares_and_cubes (bound : ℕ) (hk : bound = 1000) : 
  Nat.card {n : ℕ | n < bound ∧ (∃ k : ℕ, n = k ^ 6)} = 3 := 
by
  sorry

end count_squares_and_cubes_l266_266654


namespace max_value_of_expression_l266_266104

noncomputable def max_expression (θ : Fin 2011 → ℝ) : ℝ :=
  ∑ i, (Real.sin (θ i)) ^ 2012 * (Real.cos (θ ((i + 1) % 2011))) ^ 2012

theorem max_value_of_expression (θ : Fin 2011 → ℝ) (h : θ 0 = θ 2011) :
  max_expression θ = 1005 :=
sorry

end max_value_of_expression_l266_266104


namespace simplify_fraction_l266_266265

theorem simplify_fraction (a b c d : ℝ) (h₁ : a = 3) (h₂ : b = -2) (h₃ : c = 1) (h₄ : d = 4) : 
    (a + b * complex.I) / (c + d * complex.I) = (-5 / 17) + (-14 / 17) * complex.I :=
by
    rw [h₁, h₂, h₃, h₄]
    sorry

end simplify_fraction_l266_266265


namespace commodity_price_return_to_initial_l266_266056

theorem commodity_price_return_to_initial (P : ℝ) (x : ℝ)
  (h_initial : P = 200)
  (h_january : P'₁ = P + 0.3 * P)
  (h_february : P'₂ = P'₁ - 0.1 * P'₁)
  (h_march : P'₃ = P'₂ + 0.2 * P'₂)
  (h_april : P'₄ = P'₃ - (x / 100) * P'₃) :
  (h_initial = h_april) → x = 29 :=

sorry

end commodity_price_return_to_initial_l266_266056


namespace max_consecutive_integers_sum_l266_266400

theorem max_consecutive_integers_sum (n : ℕ) : 
  (∃ n, (n * (n + 1) / 2 < 500) ∧ (∀ m > n, m * (m + 1) / 2 ≥ 500)) :=
by {
  use 31,
  split,
  {
    norm_num,
  },
  {
    intros m hm,
    have h : m = 32 := by linarith,
    rw h,
    norm_num,
  },
  sorry
}

end max_consecutive_integers_sum_l266_266400


namespace inequality_of_abc_l266_266952

variable (a b c : ℝ)

theorem inequality_of_abc (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c ≥ a * b * c) : 
  a^2 + b^2 + c^2 ≥ Real.sqrt 3 * a * b * c :=
sorry

end inequality_of_abc_l266_266952


namespace quadrilateral_problem_l266_266199

-- The problem condition as definitions and lemma statement. Sorry is used to skip the proof.
variables (A B C D E : Type)
variables (AB AC BD BC CD CE : ℕ)
variables (angle_BAD angle_ADC angle_ABD angle_BCD : angle)

def condition_1 : Prop := angle_BAD = angle_ADC
def condition_2 : Prop := angle_ABD = angle_BCD
def condition_3 : Prop := AB = 8
def condition_4 : Prop := BD = 10
def condition_5 : Prop := BC = 6
def condition_6 (CD : ℚ) (m n : ℕ) : Prop := (CD = m / n) ∧ (nat.coprime m n)

theorem quadrilateral_problem (m n : ℕ) : condition_1 ∧ condition_2 ∧ condition_3 ∧ condition_4 ∧ condition_5 ∧ condition_6 (64 / 5) m n → (m + n = 69) :=
by sorry

end quadrilateral_problem_l266_266199


namespace dishonest_dealer_weight_l266_266450

theorem dishonest_dealer_weight 
  (C : ℝ) -- cost price per kg
  (S : ℝ := 1.17370892018779344 * C) -- selling price per kg with 17.370892018779344% profit
  (W : ℝ) -- actual weight used per kg
  (h : S / W = C) -- dealer sells at cost price
  : W ≈ 0.8517 := by
  sorry

end dishonest_dealer_weight_l266_266450


namespace fifteenth_entry_l266_266100

def r_7 (n : ℕ) : ℕ := n % 7

theorem fifteenth_entry :
  (∃ l : List ℕ, is_sorted (l) (≤) ∧ (∀ n ∈ l, r_7 (3 * n) ≤ 3) ∧ l.length ≥ 15 ∧ l.get! 14 = 22) := by
  sorry

end fifteenth_entry_l266_266100


namespace number_of_sixth_powers_less_than_1000_l266_266839

theorem number_of_sixth_powers_less_than_1000 : 
  {n : ℕ | ∃ m : ℕ, n = m^6 ∧ n < 1000}.to_finset.card = 3 := 
by {
  sorry
}

end number_of_sixth_powers_less_than_1000_l266_266839


namespace count_squares_cubes_less_than_1000_l266_266810

theorem count_squares_cubes_less_than_1000 : 
  {n : ℕ | ∃ x : ℕ, n = x^6 ∧ n < 1000}.card = 3 :=
sorry

end count_squares_cubes_less_than_1000_l266_266810


namespace difference_between_rats_digging_distance_l266_266276

theorem difference_between_rats_digging_distance :
  let large_rat_distance (n : ℕ) := ∑ i in Finset.range n, 2^i
  let small_rat_distance (n : ℕ) := ∑ i in Finset.range n, (1 / 2^i)
  large_rat_distance 5 - small_rat_distance 5 = 29 + 1 / 16 :=
by {
  let large_rat_distance := λ n, ∑ i in Finset.range n, 2^i,
  let small_rat_distance := λ n, ∑ i in Finset.range n, (1 / 2^i),
  sorry
}

end difference_between_rats_digging_distance_l266_266276


namespace sport_flavoring_to_water_ratio_l266_266204

/-- The ratio by volume of flavoring to corn syrup to water in the 
standard formulation is 1:12:30. The sport formulation has a ratio 
of flavoring to corn syrup three times as great as in the standard formulation. 
A large bottle of the sport formulation contains 4 ounces of corn syrup and 
60 ounces of water. Prove that the ratio of the amount of flavoring to water 
in the sport formulation compared to the standard formulation is 1:2. -/
theorem sport_flavoring_to_water_ratio 
    (standard_flavoring : ℝ) 
    (standard_corn_syrup : ℝ) 
    (standard_water : ℝ) : 
  standard_flavoring = 1 → standard_corn_syrup = 12 → 
  standard_water = 30 → 
  ∃ sport_flavoring : ℝ, 
  ∃ sport_corn_syrup : ℝ, 
  ∃ sport_water : ℝ, 
  sport_corn_syrup = 4 ∧ 
  sport_water = 60 ∧ 
  (sport_flavoring / sport_water) = (standard_flavoring / standard_water) / 2 :=
by
  sorry

end sport_flavoring_to_water_ratio_l266_266204


namespace f_monotonically_increasing_in_interval_l266_266520

noncomputable def f (x : ℝ) : ℝ :=
  let a₁ := cos x - sin x
  let a₂ := sqrt 3
  let a₃ := cos (π / 2 + 2 * x)
  let a₄ := cos x + sin x
  let matrix := λ a₁ a₂ a₃ a₄, a₁ * a₄ - a₂ * a₃ -- defined determinant
  matrix a₁ a₂ a₃ a₄

theorem f_monotonically_increasing_in_interval (x : ℝ) :
  x ∈ Icc (-π/6) 0 → MonotoneOn (λ x, f x) (Icc (-π/6) 0) :=
by
  sorry

end f_monotonically_increasing_in_interval_l266_266520


namespace total_cost_is_58_l266_266451

-- Define the conditions
def cost_per_adult : Nat := 22
def cost_per_child : Nat := 7
def number_of_adults : Nat := 2
def number_of_children : Nat := 2

-- Define the theorem to prove the total cost
theorem total_cost_is_58 : number_of_adults * cost_per_adult + number_of_children * cost_per_child = 58 :=
by
  -- Steps of proof will go here
  sorry

end total_cost_is_58_l266_266451


namespace part1_part2_l266_266237

def f (x a : ℝ) : ℝ := |x + a| + |x - a^2|

theorem part1 (x : ℝ) : f x 1 ≥ 4 ↔ x ≤ -2 ∨ x ≥ 2 := sorry

theorem part2 (m : ℝ) : (∀ x : ℝ, ∃ a : ℝ, -1 < a ∧ a < 3 ∧ m < f x a) ↔ m < 12 := sorry

end part1_part2_l266_266237


namespace limit_sum_div_l266_266067

theorem limit_sum_div (h : ∀ n, ∑ k in Finset.range (n + 1), k = n * (n + 1) / 2) :
  (Real.seq_limit (fun n => (∑ k in Finset.range (n + 1), k) / (n^2 + 1)) (1 / 2)) :=
begin
  sorry
end

end limit_sum_div_l266_266067


namespace average_weight_20_boys_l266_266336

theorem average_weight_20_boys 
  (A : Real)
  (numBoys₁ numBoys₂ : ℕ)
  (weight₂ : Real)
  (avg_weight_class : Real)
  (h_numBoys₁ : numBoys₁ = 20)
  (h_numBoys₂ : numBoys₂ = 8)
  (h_weight₂ : weight₂ = 45.15)
  (h_avg_weight_class : avg_weight_class = 48.792857142857144)
  (h_total_boys : numBoys₁ + numBoys₂ = 28)
  (h_eq_weight : numBoys₁ * A + numBoys₂ * weight₂ = 28 * avg_weight_class) :
  A = 50.25 :=
  sorry

end average_weight_20_boys_l266_266336


namespace selection_methods_count_l266_266263

-- Define a function to compute combinations (n choose r)
def combination (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Problem statement
theorem selection_methods_count :
  combination 5 2 * combination 3 1 * combination 2 1 = 60 :=
by
  sorry

end selection_methods_count_l266_266263


namespace compute_fraction_power_l266_266507

theorem compute_fraction_power : 9 * (1 / 7)^4 = 9 / 2401 := by
  sorry

end compute_fraction_power_l266_266507


namespace calculate_fraction_l266_266327

theorem calculate_fraction : 
  ∃ f : ℝ, (14.500000000000002 ^ 2) * f = 126.15 ∧ f = 0.6 :=
by
  sorry

end calculate_fraction_l266_266327


namespace kelly_supplies_left_l266_266219

noncomputable def supplies_remaining 
  (num_students : ℕ) 
  (papers_per_student : ℕ) 
  (num_glue_bottles : ℕ) 
  (dropped_fraction : ℝ) 
  (additional_papers : ℕ) : ℕ :=
let total_papers := num_students * papers_per_student in
let initial_supplies := total_papers + num_glue_bottles in
let remaining_supplies := initial_supplies * dropped_fraction in
remaining_supplies + additional_papers

theorem kelly_supplies_left 
  (h1 : 8 = num_students) 
  (h2 : 3 = papers_per_student) 
  (h3 : 6 = num_glue_bottles) 
  (h4 : 0.5 = dropped_fraction) 
  (h5 : 5 = additional_papers) : 
  supplies_remaining 8 3 6 0.5 5 = 20 := 
by {
  unfold supplies_remaining,
  rw [h1, h2, h3, h4, h5],
  norm_num,
}

end kelly_supplies_left_l266_266219


namespace maximum_consecutive_positive_integers_sum_500_l266_266385

theorem maximum_consecutive_positive_integers_sum_500 : 
  ∃ n : ℕ, (n * (n + 1) / 2 < 500) ∧ (∀ m : ℕ, (m * (m + 1) / 2 < 500) → m ≤ n) :=
sorry

end maximum_consecutive_positive_integers_sum_500_l266_266385


namespace count_sixth_powers_below_1000_l266_266625

theorem count_sixth_powers_below_1000 : 
  {n : ℕ | (n < 1000) ∧ (∃ m : ℕ, n = m ^ 6)}.to_finset.card = 3 :=
by sorry

end count_sixth_powers_below_1000_l266_266625


namespace count_squares_and_cubes_less_than_1000_l266_266675

theorem count_squares_and_cubes_less_than_1000 : 
  {n : ℕ // n^6 < 1000}.card = 3 :=
by
  sorry

end count_squares_and_cubes_less_than_1000_l266_266675


namespace count_squares_and_cubes_less_than_1000_l266_266663

theorem count_squares_and_cubes_less_than_1000 : 
  {n : ℕ // n^6 < 1000}.card = 3 :=
by
  sorry

end count_squares_and_cubes_less_than_1000_l266_266663


namespace range_of_m_l266_266559

variable (x y m : ℝ)
variable (h1 : 0 < x)
variable (h2 : 0 < y)
variable (h3 : 2/x + 1/y = 1)
variable (h4 : ∀ x y : ℝ, x + 2*y > m^2 + 2*m)

theorem range_of_m (h1 : 0 < x) (h2 : 0 < y) (h3 : 2/x + 1/y = 1) (h4 : ∀ x y : ℝ, x + 2*y > m^2 + 2*m) : -4 < m ∧ m < 2 := 
sorry

end range_of_m_l266_266559


namespace polynomial_expansion_identity_l266_266106

theorem polynomial_expansion_identity
  (a a1 a3 a4 a5 : ℝ)
  (h : (a - x)^5 = a + a1 * x + 80 * x^2 + a3 * x^3 + a4 * x^4 + a5 * x^5) :
  a + a1 + 80 + a3 + a4 + a5 = 1 := 
sorry

end polynomial_expansion_identity_l266_266106


namespace range_of_m_l266_266160

open Real Set

def P (m : ℝ) := |m + 1| ≤ 2
def Q (m : ℝ) := ∃ x : ℝ, x^2 - m*x + 1 = 0 ∧ (m^2 - 4 ≥ 0)

theorem range_of_m (m : ℝ) :
  (¬¬ P m ∧ ¬ (P m ∧ Q m)) → -2 < m ∧ m ≤ 1 :=
by
  sorry

end range_of_m_l266_266160


namespace num_sixth_powers_below_1000_l266_266801

theorem num_sixth_powers_below_1000 : 
  {n : ℕ | n^6 < 1000 ∧ n > 0}.to_finset.card = 3 :=
by sorry

end num_sixth_powers_below_1000_l266_266801


namespace cannot_construct_square_can_construct_rectangle_can_construct_equilateral_triangle_l266_266124

def sum_segments (n : ℕ) : ℕ := (n * (n + 1)) / 2

theorem cannot_construct_square : sum_segments 99 % 4 ≠ 0 :=
by
  rw [sum_segments]
  have h : (99 * 100) / 2 = 4950 := rfl
  exact (Nat.mod_eq_of_lt (Nat.div_lt_self (99 * 100) 1_500)).mp ((Nat.one_lt_div_self Nat.zero_lt_bit0).mpr (Nat.div_pos h Nat.succ_pos')).symm

theorem can_construct_rectangle : ∃ l w, l * 2 + w * 2 = sum_segments 99 :=
by
  rw [sum_segments]
  have h : (99 * 100) / 2 = 4950 := rfl
  exact ⟨25 * 99, 99, by norm_num [mul_comm, ← h]⟩

theorem can_construct_equilateral_triangle : ∃ (a b c : ℕ), a = b ∧ b = c ∧ a + b + c = sum_segments 99 :=
by
  rw [sum_segments]
  have h : (99 * 100) / 2 = 4950 := rfl
  use [1650, 1650, 1650]
  exact ⟨rfl, rfl, by norm_num [← h]⟩

#align cannot_construct_square
#align can_construct_rectangle
#align can_construct_equilateral_triangle

end cannot_construct_square_can_construct_rectangle_can_construct_equilateral_triangle_l266_266124


namespace number_of_sixth_powers_less_than_1000_l266_266850

theorem number_of_sixth_powers_less_than_1000 : 
  {n : ℕ | ∃ m : ℕ, n = m^6 ∧ n < 1000}.to_finset.card = 3 := 
by {
  sorry
}

end number_of_sixth_powers_less_than_1000_l266_266850


namespace hyperbola_eccentricity_l266_266314

-- Define the hyperbola conditions hyperbola parameters a and b, foci F1 and F2
variables (a b c : ℝ) (h_a : a > 0) (h_b : b > 0) (h_c : c = sqrt(a^2 + b^2))

-- Define the line passing through F1 at an angle of 30 degrees and intersects the right branch of the hyperbola at M
variables (x y : ℝ) (M : ℝ × ℝ) (h_line : x = c) (h_angle : ∃ θ, θ = π / 6)

-- MF2 is perpendicular to the x-axis
variables (h_perpendicular : snd M = b^2 / a)

-- Eccentricity e of the hyperbola equals sqrt(3)
theorem hyperbola_eccentricity
  (h_hyperbola: ∀ x y, (x^2 / a^2) - (y^2 / b^2) = 1)
  (h_foci: F1 = (-c, 0) ∧ F2 = (c, 0))
  (h_intersect: M = (c, b^2 / a))
  (h_MF2_perpendicular: x = c) : 
  let e : ℝ := c / a in 
  e = sqrt(3) := sorry

end hyperbola_eccentricity_l266_266314


namespace count_positive_integers_square_and_cube_lt_1000_l266_266769

theorem count_positive_integers_square_and_cube_lt_1000 : 
  {n : ℕ | n < 1000 ∧ ∃ k : ℕ, n = k^6}.to_finset.card = 3 :=
by
  sorry

end count_positive_integers_square_and_cube_lt_1000_l266_266769


namespace number_of_sixth_powers_less_than_1000_l266_266638

theorem number_of_sixth_powers_less_than_1000 : 
  {n : ℕ | n < 1000 ∧ ∃ (k : ℕ), n = k^6}.finite.card = 3 :=
by sorry

end number_of_sixth_powers_less_than_1000_l266_266638


namespace noemie_and_tristan_sheep_count_unique_l266_266983

noncomputable def are_sheep_count_valid (a b : ℕ) : Prop :=
  let a_sq := a^2
  let b_sq := b^2
  a_sq > b_sq ∧
  97 ≤ a_sq + b_sq ∧ a_sq + b_sq ≤ 108 ∧
  a_sq ≥ 4 ∧ b_sq ≥ 4 ∧
  (a_sq + b_sq) % 2 = 1

theorem noemie_and_tristan_sheep_count_unique (a b : ℕ) :
  are_sheep_count_valid a b → a = 9 ∧ b = 4 :=
begin
  sorry
end

end noemie_and_tristan_sheep_count_unique_l266_266983


namespace circle_bisect_line_l266_266180

theorem circle_bisect_line (a : ℝ) :
  (∃ x y, (x - a) ^ 2 + (y + 1) ^ 2 = 3 ∧ 5 * x + 4 * y - a = 0) →
  a = 1 :=
by
  sorry

end circle_bisect_line_l266_266180


namespace find_transformation_matrix_l266_266091

variables {a b c d p q r s : ℝ}
def matrix := matrix (fin 2) (fin 2) ℝ

def N : matrix := ![![a, b], ![c, d]]
def M : matrix := ![![p, q], ![r, s]]
def result_matrix : matrix := ![![3 * a, 2 * b], ![3 * c, 2 * d]]

theorem find_transformation_matrix (H : M ⬝ N = result_matrix) :
  M = ![![3, 0], ![0, 2]] :=
sorry

end find_transformation_matrix_l266_266091


namespace correct_option_l266_266978

noncomputable def M : Set ℝ := {x | x > -2}

theorem correct_option : {0} ⊆ M := 
by 
  intros x hx
  simp at hx
  simp [M]
  show x > -2
  linarith

end correct_option_l266_266978


namespace count_sixth_powers_below_1000_l266_266826

-- Definition and necessary condition for perfect sixth power
def is_sixth_power (n : ℕ) : Prop := ∃ k : ℕ, n = k ^ 6

-- Main statement of the problem
theorem count_sixth_powers_below_1000 :
  {n : ℕ | n < 1000 ∧ is_sixth_power n}.to_finset.card = 3 := 
sorry

end count_sixth_powers_below_1000_l266_266826


namespace max_problems_missed_l266_266054

/-- At Summit High, to pass a biology test you must score at least 75%. 
    If there are 60 problems on the test, the maximum number of problems 
    that can be missed while still passing the test is 15. -/
theorem max_problems_missed (problems : ℕ) (pass_percentage : ℝ) (max_missed : ℕ) :
  problems = 60 → pass_percentage = 0.75 → max_missed = 15 :=
by
  intros h1 h2
  have h3 : 1 - pass_percentage = 0.25, by linarith
  have h4 : max_missed = 0.25 * problems, by rw [h1, h3]; norm_num
  assumption

end max_problems_missed_l266_266054


namespace max_non_neighboring_ten_digit_numbers_l266_266345

-- Definition of a ten-digit number: a list of 10 digits where the first digit is non-zero.
def ten_digit_number := { n : list ℕ // n.length = 10 ∧ (∀ (d : ℕ), d ∈ n → d ≥ 0 ∧ d < 10) ∧ n.head ≠ 0 }

-- Definition of neighboring ten-digit numbers
def neighboring (a b : ten_digit_number) : Prop :=
  a.val.length = 10 ∧ 
  b.val.length = 10 ∧ 
  list.length (list.filter (λ (d : ℕ × ℕ), d.fst ≠ d.snd) (list.zip a.val b.val)) = 1

-- Prove that the maximum number of ten-digit numbers where no two are neighboring is 9 * 10^8
theorem max_non_neighboring_ten_digit_numbers : 
  ∃ (s : finset ten_digit_number), 
    (∀ (a b : ten_digit_number), a ∈ s → b ∈ s → ¬ neighboring a b) ∧ s.card = 9 * 10^8 := 
sorry

end max_non_neighboring_ten_digit_numbers_l266_266345


namespace max_distance_on_unit_circle_l266_266163

theorem max_distance_on_unit_circle (alpha beta : ℝ) :
  let P := (Real.cos alpha, Real.sin alpha)
  let Q := (Real.cos beta, Real.sin beta)
  (max_dist := 2)
  |P Q| <= max_dist :=
by
  sorry

end max_distance_on_unit_circle_l266_266163


namespace max_consecutive_sum_l266_266392

theorem max_consecutive_sum : ∃ (n : ℕ), (∀ m : ℕ, (m < n → (m * (m + 1)) / 2 < 500)) ∧ ¬((n * (n + 1)) / 2 < 500) :=
by {
  sorry
}

end max_consecutive_sum_l266_266392


namespace intersection_points_of_parabolas_l266_266514

-- Define the parameters and conditions
def focus : Point := Point.mk 1 1

def a_values : Set Int := {-1, 0, 1}
def c_values : Set Int := {-2, -1, 0, 1, 2}

-- Define the condition for a directrix
inductive Directrix
| mk : Int -> Int -> Directrix
-- The form y = ax + c would be Directrix a c

-- Define the condition for a parabola with a focus and a directrix
structure Parabola :=
(focus : Point)
(directrix : Directrix)

def parabolas : Set Parabola :=
  { p | p.focus = focus ∧ (∃ a ∈ a_values, ∃ c ∈ c_values, p.directrix = Directrix.mk a c)}

-- Define the problem statement 
theorem intersection_points_of_parabolas : 
  ∃ n : Int, n = 360 ∧ ∀ p₁ p₂ ∈ parabolas, if p₁ ≠ p₂ then intersection_points p₁ p₂ = 2
  := by 
     sorry 

end intersection_points_of_parabolas_l266_266514


namespace construct_orthocenter_l266_266602

theorem construct_orthocenter (A B C : Point) (O M : Point) 
  (circumscribed_circle : Circle) 
  (h_radius : radius circumscribed_circle = 1)
  (h_circumcenter : is_circumcenter O A B C circumscribed_circle)
  (h_orthocenter : is_orthocenter M A B C)
  (compass : UnitRadiusCompass) :
  ∃ M' : Point, is_orthocenter M' A B C := 
sorry

end construct_orthocenter_l266_266602


namespace number_of_sixth_powers_lt_1000_l266_266700

-- Define the condition for being a sixth power and being less than 1000
def isSixthPower (n : ℕ) : Prop := ∃ k : ℕ, n = k^6

-- Define the main statement to be proved
theorem number_of_sixth_powers_lt_1000 : 
  {n : ℕ | n < 1000 ∧ isSixthPower n}.to_finset.card = 3 :=
by
  sorry

end number_of_sixth_powers_lt_1000_l266_266700


namespace sqrt_four_l266_266496

theorem sqrt_four : (∃ x : ℝ, x ^ 2 = 4) → (sqrt 4 = 2) :=
by
  intro h
  sorry

end sqrt_four_l266_266496


namespace rectangular_to_polar_l266_266519

theorem rectangular_to_polar (x y : ℝ) (h : x = 8 ∧ y = 2 * real.sqrt 6) :
  ∃ (r θ : ℝ), r = 2 * real.sqrt 22 ∧ 0 ≤ θ ∧ θ < 2 * real.pi ∧ real.tan θ = real.sqrt 6 / 4 :=
by
  sorry

end rectangular_to_polar_l266_266519


namespace find_positive_integers_n_l266_266085

theorem find_positive_integers_n (n : ℕ) (p : ℕ) (k : ℕ) 
  (h1 : n > 1) 
  (h2 : ∃ p k, Prime p ∧ k > 0 ∧ (finset.range(n).sum (λ x, (x+2)^2) = p^k)) :
  n = 2 ∨ n = 3 ∨ n = 4 ∨ n = 7 := by
  sorry

end find_positive_integers_n_l266_266085


namespace range_omega_for_three_zeros_l266_266146

theorem range_omega_for_three_zeros (ω : ℝ) (h : ω > 0)
  (h_three_zeros : ∃ a b c ∈ (Set.Icc (0 : ℝ) (2 * Real.pi)), a ≠ b ∧ b ≠ c ∧
  f a = 0 ∧ f b = 0 ∧ f c = 0 ∧
  ∀ x ∈ (Set.Icc (0 : ℝ) (2 * Real.pi)), f x = 0 → x ∈ {a, b, c}) :
  2 ≤ ω ∧ ω < 3 :=
begin
  let f := λ x, Real.cos (ω * x) - 1,
  sorry
end

end range_omega_for_three_zeros_l266_266146


namespace cos_theta_value_l266_266022

open Real

-- Define vectors v and w
def v : Fin 2 → ℝ := ![4, 5]
def w : Fin 2 → ℝ := ![2, 3]

-- Define the dot product function for 2D vectors
def dot_product (a b : Fin 2 → ℝ) : ℝ :=
  a 0 * b 0 + a 1 * b 1

-- Define the magnitude function for 2D vectors
def magnitude (a : Fin 2 → ℝ) : ℝ :=
  sqrt (a 0 * a 0 + a 1 * a 1)

-- Define θ as the acute angle between vectors v and w
def cos_theta : ℝ :=
  dot_product v w / (magnitude v * magnitude w)

-- The theorem stating the result
theorem cos_theta_value : cos_theta = 23 / sqrt 533 := by
  sorry

end cos_theta_value_l266_266022


namespace count_positive_integers_square_and_cube_lt_1000_l266_266764

theorem count_positive_integers_square_and_cube_lt_1000 : 
  {n : ℕ | n < 1000 ∧ ∃ k : ℕ, n = k^6}.to_finset.card = 3 :=
by
  sorry

end count_positive_integers_square_and_cube_lt_1000_l266_266764


namespace math_problem_l266_266541

theorem math_problem :  12 * 24 + 36 * 12 = 720 :=
by 
  calc
    12 * 24 + 36 * 12 = 12 * 24 + 12 * 36 : by rw mul_comm 36 12
    ...               = 12 * (24 + 36)      : by rw ← mul_add 12 24 36
    ...               = 12 * 60             : by norm_num
    ...               = 720                 : by norm_num

end math_problem_l266_266541


namespace cos_omega_x_3_zeros_interval_l266_266143

theorem cos_omega_x_3_zeros_interval (ω : ℝ) (hω : ω > 0)
  (h3_zeros : ∃ a b c : ℝ, (0 ≤ a ∧ a ≤ 2 * Real.pi) ∧
    (0 ≤ b ∧ b ≤ 2 * Real.pi ∧ b ≠ a) ∧
    (0 ≤ c ∧ c ≤ 2 * Real.pi ∧ c ≠ a ∧ c ≠ b) ∧
    (∀ x : ℝ, (0 ≤ x ∧ x ≤ 2 * Real.pi) →
      (Real.cos (ω * x) - 1 = 0 ↔ x = a ∨ x = b ∨ x = c))) :
  2 ≤ ω ∧ ω < 3 :=
sorry

end cos_omega_x_3_zeros_interval_l266_266143


namespace eight_digit_number_div_by_9_l266_266181

theorem eight_digit_number_div_by_9 (n : ℕ) (hn : 0 ≤ n ∧ n ≤ 9)
  (h : (8 + 5 + 4 + n + 5 + 2 + 6 + 8) % 9 = 0) : n = 7 :=
by
  sorry

end eight_digit_number_div_by_9_l266_266181


namespace coefficient_x2_y6_l266_266915

theorem coefficient_x2_y6 : 
  let x y : ℚ
  in let expr := (1 - y / x) * (x + y)^8
  in polynomial.coeff (polynomial.expand x^2 y^6 expr) = -28 := 
by
  sorry

end coefficient_x2_y6_l266_266915


namespace translate_set_exists_l266_266127

noncomputable theory

-- Definitions for the conditions
def distance_greater_than (A_i A_j : ℝ × ℝ) (d : ℝ) : Prop :=
  (A_i.1 - A_j.1)^2 + (A_i.2 - A_j.2)^2 > d^2

def area_less_than (M : set (ℝ × ℝ)) (a : ℝ) : Prop :=
  ∃ (f : ℝ × ℝ → ℝ) [measurable f], has_finite_measure (indicator_function M) ∧ ∫⁻ x, f x ∂volume < a

-- The main statement
theorem translate_set_exists (A : ℕ → (ℝ × ℝ)) (M : set (ℝ × ℝ))
  (h1 : ∀ i j, i ≠ j → distance_greater_than (A i) (A j) 2)
  (h2 : area_less_than M π) :
  ∃ (v : ℝ × ℝ), ‖v‖ < 1 ∧ ∀ i, (A i) ∉ (λ (p : ℝ × ℝ), (p.1 + v.1, p.2 + v.2)) '' M :=
sorry

end translate_set_exists_l266_266127


namespace vertex_farthest_from_origin_l266_266270

def dilation (p : ℝ × ℝ) (k : ℝ) : ℝ × ℝ := (k * p.1, k * p.2)

theorem vertex_farthest_from_origin :
  let center := (5, -5)
  let area := 16
  let top_side_horizontal := true
  let dilation_center := (0, 0)
  let scale_factor := 3
  let side_length := real.sqrt area
  let half_side := side_length / 2
  let p := (center.1 - half_side, center.2 + half_side)
  let q := (center.1 - half_side, center.2 - half_side)
  let r := (center.1 + half_side, center.2 - half_side)
  let s := (center.1 + half_side, center.2 + half_side)
  let p' := dilation p scale_factor
  let q' := dilation q scale_factor
  let r' := dilation r scale_factor
  let s' := dilation s scale_factor in
  r' = (21, -21) :=
by
  sorry

end vertex_farthest_from_origin_l266_266270


namespace max_consecutive_sum_l266_266389

theorem max_consecutive_sum : ∃ (n : ℕ), (∀ m : ℕ, (m < n → (m * (m + 1)) / 2 < 500)) ∧ ¬((n * (n + 1)) / 2 < 500) :=
by {
  sorry
}

end max_consecutive_sum_l266_266389


namespace number_of_sixth_powers_less_than_1000_l266_266637

theorem number_of_sixth_powers_less_than_1000 : 
  {n : ℕ | n < 1000 ∧ ∃ (k : ℕ), n = k^6}.finite.card = 3 :=
by sorry

end number_of_sixth_powers_less_than_1000_l266_266637


namespace foci_distance_l266_266302

def hyperbola (x y : ℝ) := x * y = 4

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

theorem foci_distance :
  distance (2, 2) (-2, -2) = 4 * real.sqrt 2 :=
by
  sorry

end foci_distance_l266_266302


namespace parallel_lines_iff_slope_eq_l266_266005

def line1 (m : ℝ) : ℝ × ℝ × ℝ := (3*m - 4, 4, -2)
def line2 (m : ℝ) : ℝ × ℝ × ℝ := (m, 2, -2)

def slope (a b c : ℝ) : ℝ := -a / b

theorem parallel_lines_iff_slope_eq (m : ℝ) :
  slope (3*m - 4) 4 (-2) = slope m 2 (-2) ↔ m = 4 := by
sorry

end parallel_lines_iff_slope_eq_l266_266005


namespace number_of_sixth_powers_less_than_1000_l266_266643

theorem number_of_sixth_powers_less_than_1000 : 
  {n : ℕ | n < 1000 ∧ ∃ (k : ℕ), n = k^6}.finite.card = 3 :=
by sorry

end number_of_sixth_powers_less_than_1000_l266_266643


namespace employees_excluding_manager_l266_266283

theorem employees_excluding_manager (E : ℕ) (avg_salary_employee : ℕ) (manager_salary : ℕ) (new_avg_salary : ℕ) (total_employees_with_manager : ℕ) :
  avg_salary_employee = 1800 →
  manager_salary = 4200 →
  new_avg_salary = avg_salary_employee + 150 →
  total_employees_with_manager = E + 1 →
  (1800 * E + 4200) / total_employees_with_manager = new_avg_salary →
  E = 15 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end employees_excluding_manager_l266_266283


namespace max_consecutive_sum_less_500_l266_266354

theorem max_consecutive_sum_less_500 : 
  (argmax (λ n : ℕ, n * (n + 1) / 2 < 500) = 31) :=
by sorry

end max_consecutive_sum_less_500_l266_266354


namespace number_of_possible_values_and_their_sum_l266_266228

theorem number_of_possible_values_and_their_sum:
  (∃ (f : ℝ → ℝ), (∀ x y : ℝ, f(x) * f(y) - f(x * y) = 3 * x + 2 * y) ∧
  let n := 2 in
  let s := -11 / 4 in
  n * s = -11 / 2) :=
begin
  sorry
end

end number_of_possible_values_and_their_sum_l266_266228


namespace both_square_and_cube_count_both_square_cube_less_than_1000_l266_266788

theorem both_square_and_cube (n : ℕ) (h_positive : n > 0) (h_bound : n < 1000) (h_square : ∃ k : ℕ, n = k^2) (h_cube : ∃ k : ℕ, n = k^3) :
  n ∈ {1, 64, 729} :=
sorry

theorem count_both_square_cube_less_than_1000 :
  {n : ℕ | n > 0 ∧ n < 1000 ∧ ∃ k : ℕ, n = k^2 ∧ ∃ k : ℕ, n = k^3}.to_finset.card = 3 :=
by
  have h_valid : ∀ n ∈ {1, 64, 729}, n > 0 ∧ n < 1000 ∧ (∃ k : ℕ, n = k^2) ∧ (∃ k : ℕ, n = k^3) := sorry
  have h_reverse : ∀ n, n > 0 ∧ n < 1000 ∧ (∃ k : ℕ, n = k^2) ∧ (∃ k : ℕ, n = k^3) → n ∈ {1, 64, 729} := sorry
  exact Finset.card_map _ {1, 64, 729}.to_finset
  sorry

end both_square_and_cube_count_both_square_cube_less_than_1000_l266_266788


namespace num_sixth_powers_below_1000_l266_266791

theorem num_sixth_powers_below_1000 : 
  {n : ℕ | n^6 < 1000 ∧ n > 0}.to_finset.card = 3 :=
by sorry

end num_sixth_powers_below_1000_l266_266791


namespace count_squares_and_cubes_less_than_1000_l266_266669

theorem count_squares_and_cubes_less_than_1000 : 
  {n : ℕ // n^6 < 1000}.card = 3 :=
by
  sorry

end count_squares_and_cubes_less_than_1000_l266_266669


namespace range_omega_for_three_zeros_l266_266145

theorem range_omega_for_three_zeros (ω : ℝ) (h : ω > 0)
  (h_three_zeros : ∃ a b c ∈ (Set.Icc (0 : ℝ) (2 * Real.pi)), a ≠ b ∧ b ≠ c ∧
  f a = 0 ∧ f b = 0 ∧ f c = 0 ∧
  ∀ x ∈ (Set.Icc (0 : ℝ) (2 * Real.pi)), f x = 0 → x ∈ {a, b, c}) :
  2 ≤ ω ∧ ω < 3 :=
begin
  let f := λ x, Real.cos (ω * x) - 1,
  sorry
end

end range_omega_for_three_zeros_l266_266145


namespace distance_between_A_and_B_l266_266936

noncomputable def distance_between_points_polar (rA θA rB θB : ℝ) :=
  (rA^2 + rB^2 - 2 * rA * rB * Real.cos (θB - θA))^.sqrt

theorem distance_between_A_and_B :
  distance_between_points_polar 1 (Real.pi / 6) 3 (5 * Real.pi / 6) = Real.sqrt 13 :=
by
  sorry

end distance_between_A_and_B_l266_266936


namespace even_number_divisible_by_8_l266_266176

theorem even_number_divisible_by_8 {n : ℤ} (h : ∃ k : ℤ, n = 2 * k) : 
  (n * (n^2 + 20)) % 8 = 0 ∧ 
  (n * (n^2 - 20)) % 8 = 0 ∧ 
  (n * (n^2 + 4)) % 8 = 0 ∧ 
  (n * (n^2 - 4)) % 8 = 0 :=
by
  sorry

end even_number_divisible_by_8_l266_266176


namespace walter_bus_time_l266_266350

noncomputable def walter_schedule : Prop :=
  let wake_up_time := 6  -- Walter gets up at 6:00 a.m.
  let leave_home_time := 7  -- Walter catches the school bus at 7:00 a.m.
  let arrival_home_time := 17  -- Walter arrives home at 5:00 p.m.
  let num_classes := 8  -- Walter has 8 classes
  let class_duration := 45  -- Each class lasts 45 minutes
  let lunch_duration := 40  -- Walter has 40 minutes for lunch
  let additional_activities_hours := 2.5  -- Walter has 2.5 hours of additional activities

  -- Total time calculation
  let total_away_hours := arrival_home_time - leave_home_time
  let total_away_minutes := total_away_hours * 60

  -- School-related activities calculation
  let total_class_minutes := num_classes * class_duration
  let total_additional_activities_minutes := additional_activities_hours * 60
  let total_school_activity_minutes := total_class_minutes + lunch_duration + total_additional_activities_minutes

  -- Time spent on the bus
  let bus_time := total_away_minutes - total_school_activity_minutes
  bus_time = 50

-- Statement to prove
theorem walter_bus_time : walter_schedule :=
  sorry

end walter_bus_time_l266_266350


namespace initial_balance_correct_l266_266220

variable (init_balance transferred_to_mom transferred_to_sister final_balance total_transferred : ℕ)

-- Define the conditions
def conditions : Prop :=
  transferred_to_mom = 60 ∧
  transferred_to_sister = transferred_to_mom / 2 ∧
  final_balance = 100 ∧
  total_transferred = transferred_to_mom + transferred_to_sister

-- Define the statement to be proved
theorem initial_balance_correct (h : conditions) : init_balance = final_transferred + final_balance :=
by
  sorry

end initial_balance_correct_l266_266220


namespace sqrt_four_eq_two_l266_266498

theorem sqrt_four_eq_two : ∃ x : ℝ, x^2 = 4 ∧ x = 2 :=
by
  use 2
  split
  { norm_num }
  { refl }

end sqrt_four_eq_two_l266_266498


namespace ratio_of_socks_l266_266216

variable (b : ℕ)            -- the number of pairs of blue socks
variable (x : ℝ)            -- the price of blue socks per pair

def original_cost : ℝ := 5 * 3 * x + b * x
def interchanged_cost : ℝ := b * 3 * x + 5 * x

theorem ratio_of_socks :
  (5 : ℝ) / b = 5 / 14 :=
by
  sorry

end ratio_of_socks_l266_266216


namespace kyoko_bought_balls_l266_266866

theorem kyoko_bought_balls (cost_per_ball : ℝ) (total_paid : ℝ) (number_of_balls : ℕ)
  (h1 : cost_per_ball = 1.54) 
  (h2 : total_paid = 4.62)
  (h3 : total_paid / cost_per_ball = number_of_balls) 
  : number_of_balls = 3 :=
by
  unfold number_of_balls
  rw [h1, h2]
  norm_num [h3]
  sorry

end kyoko_bought_balls_l266_266866


namespace coefficient_x2_y6_l266_266914

theorem coefficient_x2_y6 : 
  let x y : ℚ
  in let expr := (1 - y / x) * (x + y)^8
  in polynomial.coeff (polynomial.expand x^2 y^6 expr) = -28 := 
by
  sorry

end coefficient_x2_y6_l266_266914


namespace arithmetic_sequence_nth_term_l266_266129

variable {α : Type*} [AddCommGroup α]

noncomputable def a_n (a2 d : α) (n : ℕ) : α :=
a2 + (n - 2) * d

theorem arithmetic_sequence_nth_term (a2 d : α) (n : ℕ) (h1 : a2 = -4) (h2 : d = -3) (h3 : n = 8) :
  a_n a2 d n = -22 := by
  sorry

end arithmetic_sequence_nth_term_l266_266129


namespace pentagon_square_ratio_l266_266476

-- Define the variables and functions based on the conditions
def square_area (s_s : ℝ) : ℝ := s_s^2
def pentagon_area (s_p : ℝ) : ℝ := (5 * s_p^2 / 4) * ((Float.sqrt 5 + 1) / 2)

-- Main problem statement
theorem pentagon_square_ratio (s_s s_p : ℝ) (h1 : square_area s_s = pentagon_area s_p) : 
  s_p / s_s = Float.sqrt (8 / (5 * (Float.sqrt 5 + 1))) :=
by
  sorry

end pentagon_square_ratio_l266_266476


namespace range_of_a_l266_266152

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x^2 + a * x + b

def A (a : ℝ) (b : ℝ) : set ℝ := {x | f x a b ≤ 0}
def B (a : ℝ) (b : ℝ) : set ℝ := {x | f (f x a b) a b ≤ (5 / 4)}

theorem range_of_a (a : ℝ) (h : A a (5 / 4) = B a (5 / 4) ∧ A a (5 / 4) ≠ ∅) : 
  sqrt 5 ≤ a ∧ a ≤ 5 :=
begin
  sorry
end

end range_of_a_l266_266152


namespace calculate_f5_l266_266967

noncomputable def f_n (n : ℕ) (x : ℝ) : ℝ :=
  ∏ i in Finset.range n, Real.cos (x / 2 ^ i)

theorem calculate_f5 :
  f_n 5 (8 * Real.pi / 3) = - Real.sqrt 3 / 32 :=
by
  sorry

end calculate_f5_l266_266967


namespace count_sixth_powers_below_1000_l266_266836

-- Definition and necessary condition for perfect sixth power
def is_sixth_power (n : ℕ) : Prop := ∃ k : ℕ, n = k ^ 6

-- Main statement of the problem
theorem count_sixth_powers_below_1000 :
  {n : ℕ | n < 1000 ∧ is_sixth_power n}.to_finset.card = 3 := 
sorry

end count_sixth_powers_below_1000_l266_266836


namespace count_squares_and_cubes_less_than_1000_l266_266730

theorem count_squares_and_cubes_less_than_1000 :
  {n : ℕ | 0 < n ∧ n < 1000 ∧ ∃ x, n = x ^ 6}.to_finset.card = 3 :=
by sorry

end count_squares_and_cubes_less_than_1000_l266_266730


namespace count_squares_cubes_less_than_1000_l266_266813

theorem count_squares_cubes_less_than_1000 : 
  {n : ℕ | ∃ x : ℕ, n = x^6 ∧ n < 1000}.card = 3 :=
sorry

end count_squares_cubes_less_than_1000_l266_266813


namespace cornbread_pieces_l266_266948

theorem cornbread_pieces :
  let pan_length := 20
  let pan_width := 18
  let piece_length := 2
  let piece_width := 2
  let pan_area := pan_length * pan_width
  let piece_area := piece_length * piece_width
  let num_pieces := pan_area / piece_area
  num_pieces = 90 :=
by
  let pan_length := 20
  let pan_width := 18
  let piece_length := 2
  let piece_width := 2
  let pan_area := pan_length * pan_width
  let piece_area := piece_length * piece_width
  let num_pieces := pan_area / piece_area
  show num_pieces = 90
  from sorry

end cornbread_pieces_l266_266948


namespace power_equation_l266_266172

theorem power_equation (p : ℕ) : 81^6 = 3^p → p = 24 :=
by
  intro h
  have h1 : 81 = 3^4 := by norm_num
  rw [h1] at h
  rw [pow_mul] at h
  norm_num at h
  exact eq_of_pow_eq_pow _ h

end power_equation_l266_266172


namespace coeff_x2y6_in_expansion_l266_266908

theorem coeff_x2y6_in_expansion : 
  (coeff_of_term (x := 2) (y := 6)) ((1 - (y / x)) * (x + y) ^ 8) = -28 :=
sorry

end coeff_x2y6_in_expansion_l266_266908


namespace divisible_values_l266_266513

def is_digit (n : ℕ) : Prop :=
  n >= 0 ∧ n <= 9

def N (x y : ℕ) : ℕ :=
  30 * 10^7 + x * 10^6 + 7 * 10^4 + y * 10^3 + 3

def is_divisible_by_37 (n : ℕ) : Prop :=
  n % 37 = 0

theorem divisible_values :
  ∃ (x y : ℕ), is_digit x ∧ is_digit y ∧ is_divisible_by_37 (N x y) ∧ ((x, y) = (8, 1) ∨ (x, y) = (4, 4) ∨ (x, y) = (0, 7)) :=
by {
  sorry
}

end divisible_values_l266_266513


namespace eccentricity_range_l266_266330

theorem eccentricity_range (a b : ℝ) (e : ℝ) (h₀ : a > b) (h₁ : b > 0)
  (h₂ : ∀ P : ℝ × ℝ, P ∈ {P : ℝ × ℝ | ∃ (x y : ℝ), P = (x, y) ∧ (x^2 / a^2 + y^2 / b^2 = 1)} → ¬ obtuse F₁ P F₂) :
  0 < e ∧ e ≤ real.sqrt 2 / 2 := 
sorry

end eccentricity_range_l266_266330


namespace find_least_k_palindrome_l266_266539

def is_palindrome (n : ℕ) : Prop :=
  n.toString = n.toString.reverse

theorem find_least_k_palindrome : ∃ k : ℕ, k > 0 ∧ is_palindrome (k + 25973) ∧ k = 89 := 
by
  sorry

end find_least_k_palindrome_l266_266539


namespace percent_removed_correct_l266_266037

-- Define the dimensions of the original box
def box_length := 18
def box_width := 12
def box_height := 9

-- Define the side length of the cube removed from each corner
def side_length_cube := 2

-- Calculate the volume of the original box
def volume_box := box_length * box_width * box_height

-- Calculate the volume of one cube removed
def volume_cube := side_length_cube * side_length_cube * side_length_cube

-- Calculate the total volume of cubes removed (8 corners)
def volume_total_removed := 8 * volume_cube

-- Calculate the percent of the original volume that is removed
noncomputable def percent_removed := (volume_total_removed.toRat / volume_box.toRat) * 100

-- Theorem: The percent of the original volume removed is 3.29%
theorem percent_removed_correct : percent_removed ≈ 3.29 := sorry

end percent_removed_correct_l266_266037


namespace coefficient_of_x2y6_l266_266897

theorem coefficient_of_x2y6 :
  let binom_coeff8 (r : ℕ) : ℕ := Nat.choose 8 r
  ((1 - y / x) * (x + y)^8).coeff (2, 6) = -28 :=
by
  sorry

end coefficient_of_x2y6_l266_266897


namespace sum_odd_even_integers_l266_266407

theorem sum_odd_even_integers :
  let odd_terms_sum := (15 / 2) * (1 + 29)
  let even_terms_sum := (10 / 2) * (2 + 20)
  odd_terms_sum + even_terms_sum = 335 :=
by
  let odd_terms_sum := (15 / 2) * (1 + 29)
  let even_terms_sum := (10 / 2) * (2 + 20)
  show odd_terms_sum + even_terms_sum = 335
  sorry

end sum_odd_even_integers_l266_266407


namespace calculate_expression_value_l266_266063

/-- A theorem to calculate the value of the given expression. -/
theorem calculate_expression_value : 
  (-10) * (-1 / 2) - Real.sqrt 16 - (-1)^2022 + Real.cbrt (-8) = -2 := 
by
  sorry

end calculate_expression_value_l266_266063


namespace count_sixth_powers_below_1000_l266_266825

-- Definition and necessary condition for perfect sixth power
def is_sixth_power (n : ℕ) : Prop := ∃ k : ℕ, n = k ^ 6

-- Main statement of the problem
theorem count_sixth_powers_below_1000 :
  {n : ℕ | n < 1000 ∧ is_sixth_power n}.to_finset.card = 3 := 
sorry

end count_sixth_powers_below_1000_l266_266825


namespace average_marks_is_59_23_l266_266282

noncomputable def class1_average : ℝ := 50
def class1_students : ℕ := 25

noncomputable def class2_average : ℝ := 65
def class2_students : ℕ := 40

noncomputable def total_students : ℕ := class1_students + class2_students := by sorry

noncomputable def total_marks : ℝ :=
  class1_students * class1_average + class2_students * class2_average := by sorry

noncomputable def average_marks_all_students : ℝ :=
  total_marks / total_students := by sorry

theorem average_marks_is_59_23 : average_marks_all_students ≈ 59.23 := by sorry

end average_marks_is_59_23_l266_266282


namespace coefficient_x2y6_l266_266922

/-- In the expansion of (1 - y / x) * (x + y) ^ 8, the coefficient of x ^ 2 * y ^ 6 is -28. -/
theorem coefficient_x2y6 (x y : ℚ) :  
  coefficient ((1 - y / x) * (x + y)^8) (monomial (2, 6)) = -28 :=
sorry

end coefficient_x2y6_l266_266922


namespace infinitely_many_triples_exists_l266_266264

theorem infinitely_many_triples_exists :
  ∃ᶠ (x y z : ℝ) in filter.at_top, (x^2 + y = y^2 + z) ∧ 
                                    (y^2 + z = z^2 + x) ∧ 
                                    (z^2 + x = x^2 + y) ∧ 
                                    (x ≠ y ∧ y ≠ z ∧ z ≠ x) :=
sorry

end infinitely_many_triples_exists_l266_266264


namespace tweets_when_hungry_l266_266256

theorem tweets_when_hungry (H : ℕ) : 
  (18 * 20) + (H * 20) + (45 * 20) = 1340 → H = 4 := by
  sorry

end tweets_when_hungry_l266_266256


namespace min_value_sum_of_inverses_l266_266226

theorem min_value_sum_of_inverses (b : Fin 10 → ℝ) (h_pos : ∀ i, 0 < b i) (h_sum : (∑ i, b i) = 2) :
  (∑ i, 1 / (b i)) ≥ 50 := 
sorry

end min_value_sum_of_inverses_l266_266226


namespace part1_monotonic_interval_part2_range_k_l266_266240

noncomputable def f (x k : ℝ) : ℝ := x * Real.exp (k * x)

theorem part1_monotonic_interval (k : ℝ) (hk : k > 0) :
  (∀ x, x > - (1 / k) → (1 + k * x) * Real.exp(k * x) > 0) ∧
  (∀ x, x < - (1 / k) → (1 + k * x) * Real.exp(k * x) < 0) :=
sorry

theorem part2_range_k (k : ℝ) (hk1 : k ≠ 0)
  (h_mono_inc : ∀ x, -1 < x → x < 1 → (1 + k * x) * Real.exp(k * x) ≥ 0) :
  k ∈ Set.Icc (-1 : ℝ) 1 \ {0} :=
sorry

end part1_monotonic_interval_part2_range_k_l266_266240


namespace max_consecutive_integers_sum_l266_266401

theorem max_consecutive_integers_sum (n : ℕ) : 
  (∃ n, (n * (n + 1) / 2 < 500) ∧ (∀ m > n, m * (m + 1) / 2 ≥ 500)) :=
by {
  use 31,
  split,
  {
    norm_num,
  },
  {
    intros m hm,
    have h : m = 32 := by linarith,
    rw h,
    norm_num,
  },
  sorry
}

end max_consecutive_integers_sum_l266_266401


namespace frank_cut_rows_l266_266105

-- Define the conditions as assumptions
variables (columns rows people brownies_per_person total_brownies : ℕ)
variables (h1 : columns = 6) (h2 : people = 6) (h3 : brownies_per_person = 3)

-- Define the total number of brownies
def total_brownies := people * brownies_per_person

-- Prove the number of rows equals 3
theorem frank_cut_rows :
  total_brownies / columns = 3 :=
by
  rw [total_brownies, h1, h2, h3]
  simp
  sorry

end frank_cut_rows_l266_266105


namespace count_sixth_powers_less_than_1000_l266_266744

theorem count_sixth_powers_less_than_1000 : 
  {n : ℕ | 0 < n ∧ n < 1000 ∧ (∃ k : ℕ, n = k^6)}.to_finset.card = 3 :=
by 
  sorry

end count_sixth_powers_less_than_1000_l266_266744


namespace card_distribution_count_l266_266080

def card_set := {n : ℕ // 1 ≤ n ∧ n ≤ 20}

def elbert_cards (E : set card_set) := 
  E.card = 10

def yaiza_cards (Y : set card_set) := 
  Y.card = 10

def valid_distribution (E Y : set card_set) :=
  E ∪ Y = {n | 1 ≤ n ∧ n ≤ 20} ∧ disjoint E Y

def yaiza_lost_and_five_cards_played (E Y : set card_set) :=
  ∃ seq, seq.length = 5 ∧ seq.last ∈ E ∧ -- Yaiza lost
          ∀ i ∈ finset.range 5, seq.nth i ∈ E ∨ seq.nth i ∈ Y ∧ 
          (∀ j < i, seq.nth j < seq.nth i) -- Valid sequence of 5 cards played

theorem card_distribution_count :
  ∃ (E Y : set card_set), elbert_cards E ∧ yaiza_cards Y ∧ valid_distribution E Y ∧ 
  yaiza_lost_and_five_cards_played E Y ∧ 
  (cardinality {⟨E, Y⟩ | elbert_cards E ∧ yaiza_cards Y ∧ valid_distribution E Y ∧ 
    yaiza_lost_and_five_cards_played E Y} = 324) := 
sorry

end card_distribution_count_l266_266080


namespace moves_to_reach_3_in_5_moves_l266_266024

/--
Given a point that starts from the origin and moves 1 unit in either the positive or negative direction
in each of 5 moves, there are exactly 5 different movement schemes for the point to land on the number 3.
-/
theorem moves_to_reach_3_in_5_moves :
  let moves := {seq : list ℤ // seq.length = 5 ∧ (∀ x ∈ seq, x = 1 ∨ x = -1)}
  ∃ (m : moves), list.sum m.1 = 3 ∧ (list.filter (λ m, list.sum m.1 = 3) moves).length = 5 :=
sorry

end moves_to_reach_3_in_5_moves_l266_266024


namespace simplify_and_evaluate_l266_266266

/--
  Simplify and evaluate the expression:
  1 - (a - 2) / a ÷ (a^2 - 4) / (a^2 + a)
  where a = sqrt(3) - 2.

  The goal is to prove that this expression simplifies to sqrt(3) / 3.
-/
theorem simplify_and_evaluate :
  (1 - ((sqrt 3 - 2) - 2) / (sqrt 3 - 2) ÷ ((sqrt 3 - 2)^2 - 4) / ((sqrt 3 - 2)^2 + (sqrt 3 - 2))) = sqrt 3 / 3 :=
by
  sorry

end simplify_and_evaluate_l266_266266


namespace distance_of_lap_is_100_l266_266324

-- Definitions based on given conditions
variables (laps : ℕ) (minutes : ℕ) (earning_per_minute : ℝ) (award_per_hundred_meters : ℝ)
variables (total_earnings : ℝ) (total_hundred_meters : ℝ) (total_distance : ℝ) (distance_per_lap : ℝ)

-- Conditions
def conditions :=
  laps = 24 ∧
  minutes = 12 ∧
  earning_per_minute = 7 ∧
  award_per_hundred_meters = 3.5 ∧
  total_earnings = earning_per_minute * minutes ∧
  total_hundred_meters = total_earnings / award_per_hundred_meters ∧
  total_distance = total_hundred_meters * 100 ∧
  distance_per_lap = total_distance / laps

-- The main theorem to prove the distance per lap
theorem distance_of_lap_is_100 : distance_per_lap = 100 :=
by
  -- Assuming the conditions
  have h : conditions := sorry,
  sorry -- Proof is left as an exercise

end distance_of_lap_is_100_l266_266324


namespace certain_percentage_l266_266009

theorem certain_percentage (P : ℝ) : 
  0.15 * P * 0.50 * 4000 = 90 → P = 0.3 :=
by
  sorry

end certain_percentage_l266_266009


namespace count_divisibles_100_is_28_l266_266227

-- Define the conversion of numbers to base 8 as a sequence
def to_base_8 (n : ℕ) : List ℕ :=
  let rec digits (n : ℕ) : List ℕ :=
    if n = 0 then [] else digits (n / 8) ++ [n % 8]
  digits n

-- Defining the sequence b_k where b_k is 1 up to k in base 8
def b_seq (k : ℕ) : List (List ℕ) :=
  List.map to_base_8 (List.range (k + 1))

-- Checking the divisibility by 7
def is_divisible_by_7 (n : ℕ) : Prop :=
  n % 7 = 0

-- Counting values of b_k that are divisible by 7
def count_divisibles (n : ℕ) : ℕ :=
  (List.range n).count (λ k => is_divisible_by_7 (k * (k + 1) / 2))

theorem count_divisibles_100_is_28 : count_divisibles 100 = 28 :=
  by
    sorry

end count_divisibles_100_is_28_l266_266227


namespace evaluate_expression_l266_266083

theorem evaluate_expression : (7 - 3) ^ 2 + (7 ^ 2 - 3 ^ 2) = 56 := by
  sorry

end evaluate_expression_l266_266083


namespace max_consecutive_sum_leq_500_l266_266381

theorem max_consecutive_sum_leq_500 : ∃ n : ℕ, (∑ i in finset.range (n+1), i) ≤ 500 ∧ (∀ m : ℕ, m > n → (∑ i in finset.range (m+1), i) > 500) :=
sorry

end max_consecutive_sum_leq_500_l266_266381


namespace count_squares_and_cubes_l266_266655

theorem count_squares_and_cubes (bound : ℕ) (hk : bound = 1000) : 
  Nat.card {n : ℕ | n < bound ∧ (∃ k : ℕ, n = k ^ 6)} = 3 := 
by
  sorry

end count_squares_and_cubes_l266_266655


namespace part1_part2_l266_266112

def p (x a : ℝ) : Prop := x - a < 0
def q (x : ℝ) : Prop := x * x - 4 * x + 3 ≤ 0

theorem part1 (a : ℝ) (h : a = 2) (hpq : ∀ x : ℝ, p x a ∧ q x) :
  Set.Ico 1 (2 : ℝ) = {x : ℝ | p x a ∧ q x} :=
by {
  sorry
}

theorem part2 (hp : ∀ (x a : ℝ), p x a → ¬ q x) : {a : ℝ | ∀ x : ℝ, q x → p x a} = Set.Ioi 3 :=
by {
  sorry
}

end part1_part2_l266_266112


namespace coefficient_of_x2y6_l266_266893

theorem coefficient_of_x2y6 :
  let binom_coeff8 (r : ℕ) : ℕ := Nat.choose 8 r
  ((1 - y / x) * (x + y)^8).coeff (2, 6) = -28 :=
by
  sorry

end coefficient_of_x2y6_l266_266893


namespace number_of_sixth_powers_less_than_1000_l266_266849

theorem number_of_sixth_powers_less_than_1000 : 
  {n : ℕ | ∃ m : ℕ, n = m^6 ∧ n < 1000}.to_finset.card = 3 := 
by {
  sorry
}

end number_of_sixth_powers_less_than_1000_l266_266849


namespace count_sixth_powers_below_1000_l266_266824

-- Definition and necessary condition for perfect sixth power
def is_sixth_power (n : ℕ) : Prop := ∃ k : ℕ, n = k ^ 6

-- Main statement of the problem
theorem count_sixth_powers_below_1000 :
  {n : ℕ | n < 1000 ∧ is_sixth_power n}.to_finset.card = 3 := 
sorry

end count_sixth_powers_below_1000_l266_266824


namespace count_squares_and_cubes_less_than_1000_l266_266737

theorem count_squares_and_cubes_less_than_1000 :
  {n : ℕ | 0 < n ∧ n < 1000 ∧ ∃ x, n = x ^ 6}.to_finset.card = 3 :=
by sorry

end count_squares_and_cubes_less_than_1000_l266_266737


namespace maximum_consecutive_positive_integers_sum_500_l266_266384

theorem maximum_consecutive_positive_integers_sum_500 : 
  ∃ n : ℕ, (n * (n + 1) / 2 < 500) ∧ (∀ m : ℕ, (m * (m + 1) / 2 < 500) → m ≤ n) :=
sorry

end maximum_consecutive_positive_integers_sum_500_l266_266384


namespace inequality_relationship_l266_266130

variable (a b : ℝ)

theorem inequality_relationship
  (h1 : a < 0)
  (h2 : -1 < b ∧ b < 0) : a < a * b^2 ∧ a * b^2 < a * b :=
by
  sorry

end inequality_relationship_l266_266130


namespace find_sum_of_transformed_roots_l266_266322

-- Define the polynomial and its roots
def poly (x : ℝ) : ℝ := x^3 - x - 1

-- Assume r, s, t are the roots of the polynomial x^3 - x - 1 = 0
variables (r s t : ℝ)
hypothesis (h_r : poly r = 0)
hypothesis (h_s : poly s = 0)
hypothesis (h_t : poly t = 0)

-- Define the expression we need to prove
def expr : ℝ := (1 + r) / (1 - r) + (1 + s) / (1 - s) + (1 + t) / (1 - t)

-- The theorem that needs to be proved
theorem find_sum_of_transformed_roots : expr r s t = -7 := 
  sorry

end find_sum_of_transformed_roots_l266_266322


namespace solve_a_plus_n_l266_266077

theorem solve_a_plus_n (a n : ℕ) 
  (h : ((λ A : Matrix (Fin 3) (Fin 3) ℕ, A ^ n) (Matrix.of (fun i j => 
    match (i, j) with 
    | (0, 0) => 1
    | (0, 1) => 3
    | (0, 2) => a
    | (1, 1) => 1
    | (1, 2) => 5
    | (2, 2) => 1 
    | _ => 0))) ^ 2 = Matrix.of (λ i j => 
    match (i, j) with 
    | (0, 0) => 1
    | (0, 1) => 54
    | (0, 2) => 8008
    | (1, 1) => 1
    | (1, 2) => 90
    | (2, 2) => 1 
    | _ => 0)) : a + n = 291 :=
begin
  sorry
end

end solve_a_plus_n_l266_266077


namespace isosceles_triangle_properties_false_statements_l266_266413

variable (t : Type) [is_isosceles_triangle t]

def is_equangular (t : Type) : Prop := ∀ (a b c : ℝ), a = b = c
def have_at_least_two_equal_sides (t : Type) : Prop := ∃ (a b : ℝ), a = b ∧ t.has_side a ∧ t.has_side b
def is_regular_polygon (t : Type) : Prop := is_equangular t ∧ t.is_equilateral
def congruent_to_each_other (t : Type) : Prop := ∀ (t₁ t₂ : t), t₁ = t₂
def similar_to_each_other (t : Type) : Prop := ∀ (t₁ t₂ : t), t₁.is_similar t₂

theorem isosceles_triangle_properties_false_statements :
  ¬(is_equangular t) ∧
  (have_at_least_two_equal_sides t) ∧
  ¬(is_regular_polygon t) ∧
  ¬(congruent_to_each_other t) ∧
  ¬(similar_to_each_other t) :=
by
  sorry

end isosceles_triangle_properties_false_statements_l266_266413


namespace area_of_triangle_AMN_l266_266207

theorem area_of_triangle_AMN
  (ABC : Type) [triangle ABC]
  (A B C M N : ABC)
  (M_midpoint : is_midpoint M A C)
  (N_midpoint : is_midpoint N B M)
  (area_ABC : area ABC = 180) :
  area (triangle_of_points A M N) = 45 :=
by
  sorry

-- Definitions of is_midpoint and area would be necessary for this theorem, 
-- which in reality, would come from a geometry library. 

-- We use "is_midpoint" to assert that a point is the midpoint of two points.
-- We use "area" to find the area of a triangle.
-- "triangle_of_points" should be a function that forms a triangle from three points.

-- Since the full implementation and specifics are abstracted away, we use 'sorry' to skip the actual proof steps.

end area_of_triangle_AMN_l266_266207


namespace count_sixth_powers_below_1000_l266_266617

theorem count_sixth_powers_below_1000 : 
  {n : ℕ | (n < 1000) ∧ (∃ m : ℕ, n = m ^ 6)}.to_finset.card = 3 :=
by sorry

end count_sixth_powers_below_1000_l266_266617


namespace total_cost_is_9_43_l266_266944

def basketball_game_cost : ℝ := 5.20
def racing_game_cost : ℝ := 4.23
def total_cost : ℝ := basketball_game_cost + racing_game_cost

theorem total_cost_is_9_43 : total_cost = 9.43 := by
  sorry

end total_cost_is_9_43_l266_266944


namespace number_of_squares_and_cubes_less_than_1000_l266_266722

noncomputable def count_squares_and_cubes : ℕ :=
  (List.range 1000).countp (λ n : ℕ, (∃ k : ℕ, k^6 = n))

theorem number_of_squares_and_cubes_less_than_1000 :
  count_squares_and_cubes = 3 :=
  by
    sorry

end number_of_squares_and_cubes_less_than_1000_l266_266722


namespace number_of_sixth_powers_lt_1000_l266_266677

theorem number_of_sixth_powers_lt_1000 : 
  {n : ℕ | ∃ k : ℕ, k^6 = n ∧ n < 1000}.to_finset.card = 3 :=
by
  sorry

end number_of_sixth_powers_lt_1000_l266_266677


namespace valid_outfit_count_l266_266855

/--
I have 8 shirts and 8 hats, both coming in the colors tan, black, blue, gray, white, yellow, red, and green.
I have 6 pairs of pants available in tan, black, blue, gray, red, and green.
I refuse to wear an outfit where all three items are the same color or the pants and the hat are of the same color.

Prove that the number of valid outfits, consisting of one shirt, one hat, and one pair of pants, is 336.
-/
theorem valid_outfit_count :
  let shirts := 8
  let hats := 8
  let pants := 6
  let total_combinations := shirts * pants * hats
  let all_items_same_color := 6
  let pants_hat_same_color := 6 * shirts
  let valid_outfits := total_combinations - pants_hat_same_color in
  valid_outfits = 336 :=
by 
  let shirts := 8
  let hats := 8
  let pants := 6
  let total_combinations := shirts * pants * hats
  let all_items_same_color := 6
  let pants_hat_same_color := 6 * shirts
  let valid_outfits := total_combinations - pants_hat_same_color
  have h : valid_outfits = 336 := sorry
  exact h

end valid_outfit_count_l266_266855


namespace children_neither_happy_nor_sad_l266_266249

-- Define the total number of children
def total_children : ℕ := 60

-- Define the number of happy children
def happy_children : ℕ := 30

-- Define the number of sad children
def sad_children : ℕ := 10

-- Define the number of boys
def boys : ℕ := 18

-- Define the number of girls
def girls : ℕ := 42

-- Define the number of happy boys
def happy_boys : ℕ := 6

-- Define the number of sad girls
def sad_girls : ℕ := 4

theorem children_neither_happy_nor_sad :
  total_children - (happy_children + sad_children) = 20 :=
by
  -- Calculate the number of children who are either happy or sad
  have h1 : happy_children + sad_children = 40 :=
    by norm_num

  -- Calculate the number of children who are neither happy nor sad
  have h2 : total_children - (happy_children + sad_children) = 20 :=
    by norm_num

  exact h2

end children_neither_happy_nor_sad_l266_266249


namespace selling_price_eq_100_l266_266469

variable (CP SP : ℝ)

-- Conditions
def gain : ℝ := 20
def gain_percentage : ℝ := 0.25

-- The proof of the selling price
theorem selling_price_eq_100
  (h1 : gain = 20)
  (h2 : gain_percentage = 0.25)
  (h3 : gain = gain_percentage * CP)
  (h4 : SP = CP + gain) :
  SP = 100 := sorry

end selling_price_eq_100_l266_266469


namespace number_of_zeros_in_interval_range_of_m_f_minus_g_positive_l266_266594

-- Define the functions f and g
def f (x : ℝ) : ℝ := Real.exp x * Real.sin x - Real.cos x
def g (x : ℝ) : ℝ := x * Real.cos x - Real.sqrt 2 * Real.exp x

-- (1) Theorem: There is exactly one zero of f in (0, π/2)
theorem number_of_zeros_in_interval : set.countable {x : ℝ | f x = 0} ∧ 
  set.subset (fun x, 0 < x ∧ x < Real.pi / 2) {x : ℝ | f x = 0} :=
sorry

-- (2) Theorem: For all x1, x2 ∈ [0, π/2], f(x1) + g(x2) ≥ m implies m ≤ -1 - √2
theorem range_of_m (m : ℝ) :
  (∀ x1 x2 ∈ set.Icc 0 (Real.pi / 2), f x1 + g x2 ≥ m) → m ≤ -1 - Real.sqrt 2 :=
sorry

-- (3) Theorem: For x > -1, f(x) - g(x) > 0
theorem f_minus_g_positive (x : ℝ) (h : x > -1) : f x - g x > 0 :=
sorry

end number_of_zeros_in_interval_range_of_m_f_minus_g_positive_l266_266594


namespace connor_annual_income_l266_266881

theorem connor_annual_income (q : ℝ) (h₁ : q = 20) :
    let I := 46000 in
    let T := if I ≤ 35000 then 0.01 * q * I
             else if I ≤ 50000 then 0.01 * q * 35000 + 0.01 * (q + 3) * (I - 35000)
             else 0.01 * q * 35000 + 0.01 * (q + 3) * 15000 + 0.01 * (q + 5) * (I - 50000) in
    T = 0.01 * (q + 0.45) * I :=
by
  sorry

end connor_annual_income_l266_266881


namespace count_squares_cubes_less_than_1000_l266_266806

theorem count_squares_cubes_less_than_1000 : 
  {n : ℕ | ∃ x : ℕ, n = x^6 ∧ n < 1000}.card = 3 :=
sorry

end count_squares_cubes_less_than_1000_l266_266806


namespace count_squares_and_cubes_l266_266647

theorem count_squares_and_cubes (bound : ℕ) (hk : bound = 1000) : 
  Nat.card {n : ℕ | n < bound ∧ (∃ k : ℕ, n = k ^ 6)} = 3 := 
by
  sorry

end count_squares_and_cubes_l266_266647


namespace count_sixth_powers_less_than_1000_l266_266749

theorem count_sixth_powers_less_than_1000 : 
  {n : ℕ | 0 < n ∧ n < 1000 ∧ (∃ k : ℕ, n = k^6)}.to_finset.card = 3 :=
by 
  sorry

end count_sixth_powers_less_than_1000_l266_266749


namespace no_solution_inequality_l266_266183

theorem no_solution_inequality (a : ℝ) : (¬ ∃ x : ℝ, |x - 5| + |x + 3| < a) ↔ a ≤ 8 := 
sorry

end no_solution_inequality_l266_266183


namespace convex_m_gons_two_acute_angles_l266_266567

noncomputable def count_convex_m_gons_with_two_acute_angles (m n : ℕ) (P : Finset ℕ) : ℕ :=
  (2 * n + 1) * (Nat.choose (n + 1) (m - 1) + Nat.choose n (m - 1))

theorem convex_m_gons_two_acute_angles {m n : ℕ} {P : Finset ℕ}
  (hP : P.card = 2 * n + 1)
  (hmn : 4 < m ∧ m < n) :
  count_convex_m_gons_with_two_acute_angles m n P = 
  (2 * n + 1) * (Nat.choose (n + 1) (m - 1) + Nat.choose n (m - 1)) :=
sorry

end convex_m_gons_two_acute_angles_l266_266567


namespace part1_solution_part2_solution_l266_266149

section part1

def f (x : ℝ) : ℝ := (1 - x) / (2 * x + 2)

theorem part1_solution (x : ℝ) : f (2 ^ x) - 2 ^ (x + 1) + 2 > 0 ↔ x < 0 := by
  sorry

end part1

section part2

def f (x : ℝ) : ℝ := (1 - x) / (2 * x + 2)
def g (x : ℝ) := 2 ^ x + 2 ^ (-x) + 2

variable (x : ℝ) (k : ℝ)

theorem part2_solution (hx : x ≠ 0)
  (h : ∀ x ≠ 0, g (2 * x) + 3 ≥ k * (g x - 2)) : 
  k ≤ 7 / 2 := by
  sorry

end part2

end part1_solution_part2_solution_l266_266149


namespace most_reasonable_sampling_method_l266_266479

def significant_differences_among_stages : Prop := 
  ∀ (stages : List String) (students : stages → List Float), 
    -- Assume significant differences among stages
    ∀ s1 s2, s1 ≠ s2 → abs (avg (students s1) - avg (students s2)) > threshold

def minor_differences_within_stages : Prop := 
  ∀ (stages : List String) (students : stages → List Float) (gender : List String), 
    -- Assume minor differences within stages
    ∀ s g1 g2, g1 ≠ g2 → abs (avg (students (filter_gender s g1)) - avg (students (filter_gender s g2))) < threshold

theorem most_reasonable_sampling_method :
  significant_differences_among_stages →
  minor_differences_within_stages →
  stratified_sampling_by_educational_stage = most_reasonable_sampling
:= by
  intros h1 h2
  sorry

end most_reasonable_sampling_method_l266_266479


namespace relativistic_sum_power_13_l266_266171

def relativistic_sum (x y : ℝ) : ℝ := (x + y) / (1 + x * y)

lemma relativistic_sum_comm (x y : ℝ) : relativistic_sum x y = relativistic_sum y x :=
  sorry

lemma relativistic_sum_assoc (x y z : ℝ) : relativistic_sum (relativistic_sum x y) z = relativistic_sum x (relativistic_sum y z) :=
  sorry

def v : ℝ := (Real.root 7 17 - 1) / (Real.root 7 17 + 1)

theorem relativistic_sum_power_13 :
  relativistic_sum (relativistic_sum (relativistic_sum (relativistic_sum (relativistic_sum (relativistic_sum 
  (relativistic_sum (relativistic_sum (relativistic_sum (relativistic_sum (relativistic_sum 
  (relativistic_sum v v) v) v) v) v) v) v) v) v) v) v) v = (17^(13/7) - 1) / (17^(13/7) + 1) :=
  sorry

end relativistic_sum_power_13_l266_266171


namespace count_squares_and_cubes_less_than_1000_l266_266728

theorem count_squares_and_cubes_less_than_1000 :
  {n : ℕ | 0 < n ∧ n < 1000 ∧ ∃ x, n = x ^ 6}.to_finset.card = 3 :=
by sorry

end count_squares_and_cubes_less_than_1000_l266_266728


namespace find_t_exists_l266_266533

noncomputable def find_t : Real :=
  30 * (1 + Real.sqrt 5)

theorem find_t_exists
  {P : Fin 120 → (ℝ × ℝ)}
  (h : ∀ i, (P i).1 ∈ (Set.Icc 0 1) ∨ (P i).2 ∈ (Set.Icc 0 1)) :
  ∃ Q : (ℝ × ℝ), 
  ( (Q.1 ∈ (Set.Icc 0 1) ∨ Q.2 ∈ (Set.Icc 0 1))
    ∧ ∑ i, Real.dist (P i) Q = find_t := 
begin
  sorry
end

end find_t_exists_l266_266533


namespace max_consecutive_sum_l266_266395

theorem max_consecutive_sum : ∃ (n : ℕ), (∀ m : ℕ, (m < n → (m * (m + 1)) / 2 < 500)) ∧ ¬((n * (n + 1)) / 2 < 500) :=
by {
  sorry
}

end max_consecutive_sum_l266_266395


namespace number_of_sixth_powers_lt_1000_l266_266708

-- Define the condition for being a sixth power and being less than 1000
def isSixthPower (n : ℕ) : Prop := ∃ k : ℕ, n = k^6

-- Define the main statement to be proved
theorem number_of_sixth_powers_lt_1000 : 
  {n : ℕ | n < 1000 ∧ isSixthPower n}.to_finset.card = 3 :=
by
  sorry

end number_of_sixth_powers_lt_1000_l266_266708


namespace f_2_eq_4_l266_266859

def f (n : ℕ) : ℕ := (List.range (n + 1)).sum + (List.range n).sum

theorem f_2_eq_4 : f 2 = 4 := by
  sorry

end f_2_eq_4_l266_266859


namespace max_consecutive_integers_sum_lt_500_l266_266372

theorem max_consecutive_integers_sum_lt_500 
  (n : ℕ)
  (h₁ : ∑ k in Finset.range (n+1), k = n * (n + 1) / 2)
  (h₂ : n * (n + 1) / 2 < 500) 
  (h₃ : (n + 1) * (n + 2) / 2 > 500)
  : n = 31 := 
by sorry

end max_consecutive_integers_sum_lt_500_l266_266372


namespace count_sixth_powers_less_than_1000_l266_266756

theorem count_sixth_powers_less_than_1000 : 
  {n : ℕ | 0 < n ∧ n < 1000 ∧ (∃ k : ℕ, n = k^6)}.to_finset.card = 3 :=
by 
  sorry

end count_sixth_powers_less_than_1000_l266_266756


namespace both_square_and_cube_count_both_square_cube_less_than_1000_l266_266781

theorem both_square_and_cube (n : ℕ) (h_positive : n > 0) (h_bound : n < 1000) (h_square : ∃ k : ℕ, n = k^2) (h_cube : ∃ k : ℕ, n = k^3) :
  n ∈ {1, 64, 729} :=
sorry

theorem count_both_square_cube_less_than_1000 :
  {n : ℕ | n > 0 ∧ n < 1000 ∧ ∃ k : ℕ, n = k^2 ∧ ∃ k : ℕ, n = k^3}.to_finset.card = 3 :=
by
  have h_valid : ∀ n ∈ {1, 64, 729}, n > 0 ∧ n < 1000 ∧ (∃ k : ℕ, n = k^2) ∧ (∃ k : ℕ, n = k^3) := sorry
  have h_reverse : ∀ n, n > 0 ∧ n < 1000 ∧ (∃ k : ℕ, n = k^2) ∧ (∃ k : ℕ, n = k^3) → n ∈ {1, 64, 729} := sorry
  exact Finset.card_map _ {1, 64, 729}.to_finset
  sorry

end both_square_and_cube_count_both_square_cube_less_than_1000_l266_266781


namespace total_area_of_WIN_sectors_l266_266015

theorem total_area_of_WIN_sectors (r : ℝ) (A_total : ℝ) (Prob_WIN : ℝ) (A_WIN : ℝ) : 
  r = 15 → 
  A_total = π * r^2 → 
  Prob_WIN = 3/7 → 
  A_WIN = Prob_WIN * A_total → 
  A_WIN = 3/7 * 225 * π :=
by {
  intros;
  sorry
}

end total_area_of_WIN_sectors_l266_266015


namespace foci_distance_l266_266301

def hyperbola (x y : ℝ) := x * y = 4

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

theorem foci_distance :
  distance (2, 2) (-2, -2) = 4 * real.sqrt 2 :=
by
  sorry

end foci_distance_l266_266301


namespace vector_v_satisfies_conditions_l266_266958

noncomputable def vector_a : ℝ × ℝ × ℝ :=
  (2, 1, -1)

noncomputable def vector_b : ℝ × ℝ × ℝ :=
  (1, -2, 3)

noncomputable def vector_v : ℝ × ℝ × ℝ :=
  (7 / 6, -2 / 3, 7 / 6)

theorem vector_v_satisfies_conditions : 
  (2 * cross_product vector_v vector_a = cross_product vector_b vector_a) ∧ 
  (3 * cross_product vector_v vector_b = cross_product vector_a vector_b) :=
  sorry

end vector_v_satisfies_conditions_l266_266958


namespace number_of_sixth_powers_lt_1000_l266_266685

theorem number_of_sixth_powers_lt_1000 : 
  {n : ℕ | ∃ k : ℕ, k^6 = n ∧ n < 1000}.to_finset.card = 3 :=
by
  sorry

end number_of_sixth_powers_lt_1000_l266_266685


namespace units_digit_a_2017_eq_2_l266_266161

noncomputable def sequence_a (n : ℕ) : ℝ :=
  (real.sqrt 2 + 1)^n - (real.sqrt 2 - 1)^n

def units_digit (x : ℝ) : ℕ :=
  (⌊x⌋ : ℕ) % 10

theorem units_digit_a_2017_eq_2 :
  units_digit (sequence_a 2017) = 2 :=
  sorry

end units_digit_a_2017_eq_2_l266_266161


namespace area_when_other_side_shortened_l266_266208

def original_width := 5
def original_length := 8
def target_area := 24
def shortened_amount := 2

theorem area_when_other_side_shortened :
  (original_width - shortened_amount) * original_length = target_area →
  original_width * (original_length - shortened_amount) = 30 :=
by
  intros h
  sorry

end area_when_other_side_shortened_l266_266208


namespace count_squares_and_cubes_l266_266659

theorem count_squares_and_cubes (bound : ℕ) (hk : bound = 1000) : 
  Nat.card {n : ℕ | n < bound ∧ (∃ k : ℕ, n = k ^ 6)} = 3 := 
by
  sorry

end count_squares_and_cubes_l266_266659


namespace count_sixth_powers_below_1000_l266_266628

theorem count_sixth_powers_below_1000 : 
  {n : ℕ | (n < 1000) ∧ (∃ m : ℕ, n = m ^ 6)}.to_finset.card = 3 :=
by sorry

end count_sixth_powers_below_1000_l266_266628


namespace necessary_and_sufficient_parallel_l266_266002

-- Define the equations of the lines
def line1 (m : ℝ) : ℝ × ℝ → Prop := λ p, (3 * m - 4) * p.1 + 4 * p.2 - 2 = 0
def line2 (m : ℝ) : ℝ × ℝ → Prop := λ p, m * p.1 + 2 * p.2 - 2 = 0

-- Define the slopes of the lines
def slope1 (m : ℝ) : ℝ := -(3 * m - 4) / 4
def slope2 (m : ℝ) : ℝ := -m / 2

-- Define the property of parallelism
def are_parallel (m : ℝ) : Prop := slope1 m = slope2 m

-- The proof statement that m = 4 is a necessary and sufficient condition for the lines to be parallel
theorem necessary_and_sufficient_parallel (m : ℝ) : (3 * m - 4) / 4 = m / 2 ↔ m = 4 := by
  sorry

end necessary_and_sufficient_parallel_l266_266002


namespace rent_cost_l266_266210

-- Definitions based on conditions
def daily_supplies_cost : ℕ := 12
def price_per_pancake : ℕ := 2
def pancakes_sold_per_day : ℕ := 21

-- Proving the daily rent cost
theorem rent_cost (total_sales : ℕ) (rent : ℕ) :
  total_sales = pancakes_sold_per_day * price_per_pancake →
  rent = total_sales - daily_supplies_cost →
  rent = 30 :=
by
  intro h_total_sales h_rent
  sorry

end rent_cost_l266_266210


namespace minimum_m_value_l266_266975

theorem minimum_m_value (m n : ℕ) (h_pos_m : 0 < m) (h_pos_n : 0 < n) (h_eq : 24 * m = n^4) : m = 54 := sorry

end minimum_m_value_l266_266975


namespace transformed_mean_variance_l266_266564

variable {n : ℕ}
variable {x : Fin n → ℝ}

noncomputable def mean (data : Fin n → ℝ) : ℝ :=
  (∑ i, data i) / n

noncomputable def variance (data : Fin n → ℝ) (mean : ℝ) : ℝ :=
  (∑ i, (data i - mean) ^ 2) / n

theorem transformed_mean_variance (hmean : mean x = 2) (hvariance : variance x 2 = 3) :
  mean (fun i => 2 * x i + 5) = 9 ∧ variance (fun i => 2 * x i + 5) 9 = 12 := by
  sorry

end transformed_mean_variance_l266_266564


namespace find_tangent_equal_l266_266089

theorem find_tangent_equal (n : ℤ) (hn : -90 < n ∧ n < 90) (htan : Real.tan (n * Real.pi / 180) = Real.tan (75 * Real.pi / 180)) : n = 75 :=
sorry

end find_tangent_equal_l266_266089


namespace rahul_task_days_l266_266994

theorem rahul_task_days (R : ℕ) (h1 : ∀ x : ℤ, x > 0 → 1 / R + 1 / 84 = 1 / 35) : R = 70 := 
by
  -- placeholder for the proof
  sorry

end rahul_task_days_l266_266994


namespace compute_fraction_power_l266_266509

theorem compute_fraction_power : 9 * (1 / 7)^4 = 9 / 2401 := by
  sorry

end compute_fraction_power_l266_266509


namespace count_squares_and_cubes_less_than_1000_l266_266733

theorem count_squares_and_cubes_less_than_1000 :
  {n : ℕ | 0 < n ∧ n < 1000 ∧ ∃ x, n = x ^ 6}.to_finset.card = 3 :=
by sorry

end count_squares_and_cubes_less_than_1000_l266_266733


namespace num_values_g_100_eq_0_l266_266230

noncomputable def g_0 (x : ℝ) : ℝ := x + |x - 150| - |x + 150|

noncomputable def g_n : ℕ → ℝ → ℝ
| 0     x := g_0 x
| (n+1) x := |g_n n x| - 2

theorem num_values_g_100_eq_0 : 
  (finset.range 100).filter (fun x => g_n 100 x = 0).card = 79 :=
sorry

end num_values_g_100_eq_0_l266_266230


namespace bus_driver_earnings_l266_266444

def regular_rate : ℝ := 18
def max_regular_hours : ℝ := 40
def overtime_multiplier : ℝ := 1.75
def total_hours_worked : ℝ := 48.12698412698413
def total_compensation : ℝ := 976 -- Correct answer

theorem bus_driver_earnings :
  let regular_hours := min total_hours_worked max_regular_hours,
      overtime_hours := max (total_hours_worked - max_regular_hours) 0,
      regular_pay := regular_rate * regular_hours,
      overtime_rate := regular_rate * overtime_multiplier,
      overtime_pay := overtime_rate * overtime_hours,
      total_earnings := regular_pay + overtime_pay
  in total_earnings = total_compensation :=
by
  sorry

end bus_driver_earnings_l266_266444


namespace distance_midpoint_KN_to_LM_l266_266258

theorem distance_midpoint_KN_to_LM :
  let K := (0 : ℝ, 0 : ℝ),
      L := (4 : ℝ, 0 : ℝ),
      M := (4 : ℝ, 10 : ℝ),
      N := (0 : ℝ, 12 : ℝ),
      midKN := ((0 + 0) / 2, (0 + 12) / 2),
      LM := set.univ.prod {p : ℝ × ℝ | p.1 = 4}
  in dist (midKN.1, midKN.2) (4, midKN.2) = 6.87 :=
  sorry

end distance_midpoint_KN_to_LM_l266_266258


namespace count_sixth_powers_below_1000_l266_266829

-- Definition and necessary condition for perfect sixth power
def is_sixth_power (n : ℕ) : Prop := ∃ k : ℕ, n = k ^ 6

-- Main statement of the problem
theorem count_sixth_powers_below_1000 :
  {n : ℕ | n < 1000 ∧ is_sixth_power n}.to_finset.card = 3 := 
sorry

end count_sixth_powers_below_1000_l266_266829


namespace count_squares_cubes_less_than_1000_l266_266816

theorem count_squares_cubes_less_than_1000 : 
  {n : ℕ | ∃ x : ℕ, n = x^6 ∧ n < 1000}.card = 3 :=
sorry

end count_squares_cubes_less_than_1000_l266_266816


namespace count_squares_and_cubes_l266_266656

theorem count_squares_and_cubes (bound : ℕ) (hk : bound = 1000) : 
  Nat.card {n : ℕ | n < bound ∧ (∃ k : ℕ, n = k ^ 6)} = 3 := 
by
  sorry

end count_squares_and_cubes_l266_266656


namespace nonneg_difference_roots_l266_266404

theorem nonneg_difference_roots : 
  ∀ (a b c d : ℝ), a = 1 ∧ b = 34 ∧ c = 225 ∧ d = -49 → 
  (let Δ := b^2 - 4*a*(c-d) in
  let root1 := (-b + Real.sqrt Δ) / (2*a) in
  let root2 := (-b - Real.sqrt Δ) / (2*a) in
  abs (root1 - root2) = 6) :=
begin
  intros a b c d h,
  cases h with ha h1,
  cases h1 with hb h2,
  cases h2 with hc hd,
  have h_delta : Δ = hb^2 - 4 * ha * (hc + -hd) := by rw [ha, hb, hc, hd]; simp,
  let Δ := hb^2 - 4 * ha * (hc + -hd),
  let root1 := (-hb + Real.sqrt Δ) / (2 * ha),
  let root2 := (-hb - Real.sqrt Δ) / (2 * ha),
  have h_roots : abs (root1 - root2) = 6 := by rw [ha, hb, hc, hd]; sorry, -- Finish from here
  exact h_roots,
end

end nonneg_difference_roots_l266_266404


namespace sum_of_consecutive_integers_l266_266365

theorem sum_of_consecutive_integers (n : ℕ) (h : n * (n + 1) / 2 ≤ 500) :
  n = 31 ∨ ((32 * (32 + 1) / 2) ≤ 500) := 
sorry

end sum_of_consecutive_integers_l266_266365


namespace side_length_is_50_point_5_l266_266462

-- Define the area of the square plot
def area_of_square_plot : ℝ := 2550.25

-- Define the side length of the square plot
def side_length_of_square_plot (A : ℝ) : ℝ := Real.sqrt A

-- The main statement to be proven
theorem side_length_is_50_point_5 
  (h : area_of_square_plot = 2550.25) : 
  side_length_of_square_plot area_of_square_plot = 50.5 := 
sorry

end side_length_is_50_point_5_l266_266462


namespace total_surface_area_of_new_solid_l266_266473

-- Define the heights of the pieces using the given conditions
def height_A := 1 / 4
def height_B := 1 / 5
def height_C := 1 / 6
def height_D := 1 / 7
def height_E := 1 / 8
def height_F := 1 - (height_A + height_B + height_C + height_D + height_E)

-- Assembling the pieces back in reverse order (F to A), encapsulate the total surface area calculation
theorem total_surface_area_of_new_solid : 
  (2 * (1 : ℝ)) + (2 * (1 * 1 : ℝ)) + (2 * (1 * 1 : ℝ)) = 6 :=
by
  sorry

end total_surface_area_of_new_solid_l266_266473


namespace A_tensor_A_correct_l266_266608

def A := {0, 2, 3} : Set ℕ
def A_tensor_A := {x : ℕ | ∃ a b : ℕ, a ∈ A ∧ b ∈ A ∧ x = a + b}

theorem A_tensor_A_correct : A_tensor_A = {0, 2, 3, 4, 5, 6} := by
  sorry

end A_tensor_A_correct_l266_266608


namespace number_of_sixth_powers_lt_1000_l266_266678

theorem number_of_sixth_powers_lt_1000 : 
  {n : ℕ | ∃ k : ℕ, k^6 = n ∧ n < 1000}.to_finset.card = 3 :=
by
  sorry

end number_of_sixth_powers_lt_1000_l266_266678


namespace both_square_and_cube_count_both_square_cube_less_than_1000_l266_266776

theorem both_square_and_cube (n : ℕ) (h_positive : n > 0) (h_bound : n < 1000) (h_square : ∃ k : ℕ, n = k^2) (h_cube : ∃ k : ℕ, n = k^3) :
  n ∈ {1, 64, 729} :=
sorry

theorem count_both_square_cube_less_than_1000 :
  {n : ℕ | n > 0 ∧ n < 1000 ∧ ∃ k : ℕ, n = k^2 ∧ ∃ k : ℕ, n = k^3}.to_finset.card = 3 :=
by
  have h_valid : ∀ n ∈ {1, 64, 729}, n > 0 ∧ n < 1000 ∧ (∃ k : ℕ, n = k^2) ∧ (∃ k : ℕ, n = k^3) := sorry
  have h_reverse : ∀ n, n > 0 ∧ n < 1000 ∧ (∃ k : ℕ, n = k^2) ∧ (∃ k : ℕ, n = k^3) → n ∈ {1, 64, 729} := sorry
  exact Finset.card_map _ {1, 64, 729}.to_finset
  sorry

end both_square_and_cube_count_both_square_cube_less_than_1000_l266_266776


namespace distance_between_foci_l266_266305

-- Let the hyperbola be defined by the equation xy = 4.
def hyperbola (x y : ℝ) : Prop := x * y = 4

-- Prove that the distance between the foci of this hyperbola is 8.
theorem distance_between_foci : ∀ (x y : ℝ), hyperbola x y → ∃ d, d = 8 :=
by {
    sorry
}

end distance_between_foci_l266_266305


namespace count_valid_n_l266_266516

theorem count_valid_n (n : ℕ) (h₁ : (n % 2015) ≠ 0) :
  (n^3 + 3^n) % 5 = 0 :=
by
  sorry

end count_valid_n_l266_266516


namespace distance_between_foci_of_hyperbola_l266_266308

theorem distance_between_foci_of_hyperbola (x y : ℝ) (h : x * y = 4) : 
  distance (2, 2) (-2, -2) = 8 :=
sorry

end distance_between_foci_of_hyperbola_l266_266308


namespace num_sixth_powers_below_1000_l266_266796

theorem num_sixth_powers_below_1000 : 
  {n : ℕ | n^6 < 1000 ∧ n > 0}.to_finset.card = 3 :=
by sorry

end num_sixth_powers_below_1000_l266_266796


namespace count_squares_and_cubes_less_than_1000_l266_266726

theorem count_squares_and_cubes_less_than_1000 :
  {n : ℕ | 0 < n ∧ n < 1000 ∧ ∃ x, n = x ^ 6}.to_finset.card = 3 :=
by sorry

end count_squares_and_cubes_less_than_1000_l266_266726


namespace count_sixth_powers_below_1000_l266_266614

theorem count_sixth_powers_below_1000 : 
  {n : ℕ | (n < 1000) ∧ (∃ m : ℕ, n = m ^ 6)}.to_finset.card = 3 :=
by sorry

end count_sixth_powers_below_1000_l266_266614


namespace go_team_probability_l266_266340

noncomputable def combination (n k : ℕ) : ℝ :=
Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem go_team_probability :
  let C₁₀₅ := combination 10 5 in
  let C₈₄ := combination 8 4 in
  C₁₀₅ ≠ 0 → (C₈₄ / C₁₀₅) = 35 / 126 := by
  sorry

end go_team_probability_l266_266340


namespace count_positive_integers_square_and_cube_lt_1000_l266_266770

theorem count_positive_integers_square_and_cube_lt_1000 : 
  {n : ℕ | n < 1000 ∧ ∃ k : ℕ, n = k^6}.to_finset.card = 3 :=
by
  sorry

end count_positive_integers_square_and_cube_lt_1000_l266_266770


namespace megan_markers_l266_266433

theorem megan_markers (initial_markers : ℕ) (new_markers : ℕ) (total_markers : ℕ) :
  initial_markers = 217 →
  new_markers = 109 →
  total_markers = 326 →
  initial_markers + new_markers = 326 :=
by
  sorry

end megan_markers_l266_266433


namespace investment_problem_l266_266945

variable (x : ℝ)

def invested_bank_a (x : ℝ) : Prop :=
  let y := 1200 - x in
  (x * 1.0816 + y * 1.1236 = 1300.50)

theorem investment_problem (x = 1138.57) : invested_bank_a x :=
  sorry

end investment_problem_l266_266945


namespace number_of_sixth_powers_less_than_1000_l266_266840

theorem number_of_sixth_powers_less_than_1000 : 
  {n : ℕ | ∃ m : ℕ, n = m^6 ∧ n < 1000}.to_finset.card = 3 := 
by {
  sorry
}

end number_of_sixth_powers_less_than_1000_l266_266840


namespace find_d_l266_266565

noncomputable def arithmetic_sequence (a : ℕ → ℝ) (a₁ d : ℝ) : Prop :=
∀ n, a n = a₁ + d * (n - 1)

theorem find_d
  (a : ℕ → ℝ)
  (a₁ d : ℝ)
  (h_arith : arithmetic_sequence a a₁ d)
  (h₁ : a₁ = 1)
  (h_geom_mean : a 2 ^ 2 = a 1 * a 4)
  (h_d_neq_zero : d ≠ 0):
  d = 1 :=
sorry

end find_d_l266_266565


namespace sum_of_consecutive_integers_l266_266367

theorem sum_of_consecutive_integers (n : ℕ) (h : n * (n + 1) / 2 ≤ 500) :
  n = 31 ∨ ((32 * (32 + 1) / 2) ≤ 500) := 
sorry

end sum_of_consecutive_integers_l266_266367


namespace count_squares_cubes_less_than_1000_l266_266809

theorem count_squares_cubes_less_than_1000 : 
  {n : ℕ | ∃ x : ℕ, n = x^6 ∧ n < 1000}.card = 3 :=
sorry

end count_squares_cubes_less_than_1000_l266_266809


namespace foci_distance_l266_266300

def hyperbola (x y : ℝ) := x * y = 4

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

theorem foci_distance :
  distance (2, 2) (-2, -2) = 4 * real.sqrt 2 :=
by
  sorry

end foci_distance_l266_266300


namespace distinct_integers_sum_l266_266107

theorem distinct_integers_sum {a b c d : ℤ} (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) 
  (h_product : a * b * c * d = 25) : a + b + c + d = 0 := by
  sorry

end distinct_integers_sum_l266_266107


namespace number_of_zeros_in_interval_l266_266460

noncomputable def f : ℝ → ℝ := sorry

variable (T : ℝ) (hf_odd : ∀ x : ℝ, f (-x) = -f x) (hf_periodic : ∀ x : ℝ, f x = f (x + T))

theorem number_of_zeros_in_interval : ∃ n : ℕ, n = 5 ∧
  ∃ z : fin n → ℝ, ∀ i : fin n, z i ∈ set.Icc (-T) T ∧ f (z i) = 0 :=
sorry

end number_of_zeros_in_interval_l266_266460


namespace count_positive_integers_square_and_cube_lt_1000_l266_266758

theorem count_positive_integers_square_and_cube_lt_1000 : 
  {n : ℕ | n < 1000 ∧ ∃ k : ℕ, n = k^6}.to_finset.card = 3 :=
by
  sorry

end count_positive_integers_square_and_cube_lt_1000_l266_266758


namespace number_of_sixth_powers_lt_1000_l266_266688

theorem number_of_sixth_powers_lt_1000 : 
  {n : ℕ | ∃ k : ℕ, k^6 = n ∧ n < 1000}.to_finset.card = 3 :=
by
  sorry

end number_of_sixth_powers_lt_1000_l266_266688


namespace distance_between_foci_l266_266297

-- Define the equation of the hyperbola.
def hyperbola_eq (x y : ℝ) : Prop := x * y = 4

-- The coordinates of foci for hyperbola of the form x*y = 4
def foci_1 : (ℝ × ℝ) := (2, 2)
def foci_2 : (ℝ × ℝ) := (-2, -2)

-- Define the Euclidean distance function.
def euclidean_distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Prove that the distance between the foci is 4√2.
theorem distance_between_foci : euclidean_distance foci_1 foci_2 = 4 * real.sqrt 2 := sorry

end distance_between_foci_l266_266297


namespace germination_percentage_l266_266542

theorem germination_percentage :
  ∀ (seeds_plot1 seeds_plot2 germination_rate1 germination_rate2 : ℝ),
    seeds_plot1 = 300 →
    seeds_plot2 = 200 →
    germination_rate1 = 0.30 →
    germination_rate2 = 0.35 →
    ((germination_rate1 * seeds_plot1 + germination_rate2 * seeds_plot2) / (seeds_plot1 + seeds_plot2)) * 100 = 32 :=
by
  intros seeds_plot1 seeds_plot2 germination_rate1 germination_rate2 h1 h2 h3 h4
  sorry

end germination_percentage_l266_266542


namespace number_of_sixth_powers_lt_1000_l266_266687

theorem number_of_sixth_powers_lt_1000 : 
  {n : ℕ | ∃ k : ℕ, k^6 = n ∧ n < 1000}.to_finset.card = 3 :=
by
  sorry

end number_of_sixth_powers_lt_1000_l266_266687


namespace number_of_sixth_powers_less_than_1000_l266_266846

theorem number_of_sixth_powers_less_than_1000 : 
  {n : ℕ | ∃ m : ℕ, n = m^6 ∧ n < 1000}.to_finset.card = 3 := 
by {
  sorry
}

end number_of_sixth_powers_less_than_1000_l266_266846


namespace eccentricity_of_ellipse_l266_266572

noncomputable def ellipse_eccentricity {a b : ℝ} (h1 : a > b) (h2 : b > 0) : ℝ :=
  let c := (sqrt (a^2 - b^2))
  c / a

theorem eccentricity_of_ellipse (a b : ℝ) 
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : |((b / (a * sqrt (a^2 / b^2 - 1))) / (b / (sqrt (a^2 - b^2)))) + 
         (b / (a * sqrt (a^2 / b^2 - 1))) / (-b / (sqrt (a^2 - b^2))))| = 1) :
  ellipse_eccentricity h1 h2 = sqrt 3 / 2 := 
sorry

end eccentricity_of_ellipse_l266_266572


namespace projection_computation_l266_266136

open Real

theorem projection_computation :
  let u := (3, 6 : ℝ × ℝ) in
  let v1 := (3, -4 : ℝ × ℝ) in
  let v2 := (3, 1 : ℝ × ℝ) in
  let proj (x y : ℝ × ℝ) : ℝ × ℝ := 
    let dot (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2 in
    let scalar := dot x y / dot y y in
    (scalar * y.1, scalar * y.2) in 
  proj (proj v1 u) v2 = (-3/2, -1/2 : ℝ × ℝ) ∧ proj v1 u = (-1, -2 : ℝ × ℝ) := sorry

end projection_computation_l266_266136


namespace train_passing_time_l266_266346

/-- Two trains of equal length are running on parallel lines in the same direction. 
    The faster train passes the slower train.
    - Speed of faster train: 44 km/hr
    - Speed of slower train: 36 km/hr
    - Length of each train: 40 meters
  Prove that the time it takes for the faster train to pass the slower train is 36 seconds.
-/
theorem train_passing_time
  (speed_faster_train_kmh : ℝ) (speed_slower_train_kmh : ℝ) (train_length_m : ℝ)
  (h_speed_faster : speed_faster_train_kmh = 44)
  (h_speed_slower : speed_slower_train_kmh = 36)
  (h_length : train_length_m = 40) :
  let relative_speed_kmh := speed_faster_train_kmh - speed_slower_train_kmh,
      relative_speed_ms := (relative_speed_kmh * 5 / 18 : ℝ),
      total_distance_m := train_length_m * 2 in
  (total_distance_m / relative_speed_ms) = 36 :=
by {
  sorry
}

end train_passing_time_l266_266346


namespace geom_seq_ratio_l266_266191

variable {a : ℕ → ℝ}
variable {q : ℝ}
variable {a1 : ℝ}
variable (pq : ∀ n, 0 < a n)
variable (h1 : a 2 + a 6 = 10)
variable (h2 : a 3 * a 5 = 16)
variable (geom_seq : ∀ n, a n = a1 * q^(n - 1))

theorem geom_seq_ratio :
  (q = Real.sqrt 2) ∨ (q = Real.sqrt 2 / 2) :=
by
  sorry

end geom_seq_ratio_l266_266191


namespace coefficient_x2_y6_in_expansion_l266_266929

theorem coefficient_x2_y6_in_expansion :
  ∀ (x y : ℝ), x ≠ 0 → 
    (let c8_6 := (nat.factorial 8) / ((nat.factorial 6) * (nat.factorial 2)),
         c8_5 := (nat.factorial 8) / ((nat.factorial 5) * (nat.factorial 3)) in
     c8_6 - c8_5 = -28) :=
by
  intros x y hx_ne_zero
  let c8_6 := (nat.factorial 8) / ((nat.factorial 6) * (nat.factorial 2))
  let c8_5 := (nat.factorial 8) / ((nat.factorial 5) * (nat.factorial 3))
  have h : c8_6 - c8_5 = -28 := sorry
  exact h

end coefficient_x2_y6_in_expansion_l266_266929


namespace lateral_surface_area_measure_l266_266285

-- Define the basic conditions
variables (a : ℝ)
variables (side_length_a : a > 0)

-- Define the lateral surface area computation from the conditions
def lateral_surface_area_prism
  (a : ℝ)
  (side_length_a : a > 0)
  : ℝ :=
  (a^2 * Real.sqrt 3 * (2 + Real.sqrt 13)) / 3

-- Statement of the problem to be proven
theorem lateral_surface_area_measure
  (a : ℝ)
  (side_length_a : a > 0) 
  (inclination_angle : ℝ)
  (ha : inclination_angle = 60) :
  lateral_surface_area_prism a side_length_a = (a^2 * Real.sqrt 3 * (2 + Real.sqrt 13)) / 3 :=
sorry

end lateral_surface_area_measure_l266_266285


namespace grass_area_calculation_l266_266446

noncomputable def remaining_grass_area_proof
  (r : ℝ) (path_width : ℝ) (square_side : ℝ)
  (circle_area : ℝ) (square_area : ℝ) 
  (sector_area : ℝ) (triangle_area : ℝ) (segment_area : ℝ)
  : ℝ :=
  let remaining := circle_area - (segment_area + square_area)
  in remaining

theorem grass_area_calculation
  : remaining_grass_area_proof 10 4 6 (100 * Real.pi) 36 (21.79 * Real.pi) 49 (21.79 * Real.pi - 49) = 78.21 * Real.pi + 13 :=
by
  simp
  sorry

end grass_area_calculation_l266_266446


namespace num_sixth_powers_below_1000_l266_266795

theorem num_sixth_powers_below_1000 : 
  {n : ℕ | n^6 < 1000 ∧ n > 0}.to_finset.card = 3 :=
by sorry

end num_sixth_powers_below_1000_l266_266795


namespace constant_function_solution_l266_266523

theorem constant_function_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, 2 * f x = f (x + y) + f (x + 2 * y)) :
  ∃ c : ℝ, ∀ x : ℝ, f x = c := by
  sorry

end constant_function_solution_l266_266523


namespace no_valid_configuration_of_points_l266_266432

noncomputable def Point := (ℝ × ℝ)
noncomputable def Distance (p1 p2: Point) := ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2).sqrt

def closest_point (p: Point) (points: List Point) : Point :=
  points.minBy (λ q => Distance p q)

/-- 
  There are no configurations of 10 red points, 10 blue points, and 10 green points on a plane 
  such that for each red point, the closest point is blue; for each blue point, the closest point
  is green; and for each green point, the closest point is red.
-/
theorem no_valid_configuration_of_points :
  ¬ ∃ red_points blue_points green_points: List Point,
    red_points.length = 10 ∧
    blue_points.length = 10 ∧
    green_points.length = 10 ∧
    (∀ A ∈ red_points, closest_point A (blue_points ++ green_points ++ red_points) ∈ blue_points) ∧
    (∀ B ∈ blue_points, closest_point B (blue_points ++ green_points ++ red_points) ∈ green_points) ∧
    (∀ C ∈ green_points, closest_point C (blue_points ++ green_points ++ red_points) ∈ red_points) :=
sorry

end no_valid_configuration_of_points_l266_266432


namespace total_area_combined_l266_266034

-- Definitions based on conditions
def diagonal1 : ℝ := 30
def diagonal2 : ℝ := 18
def base : ℝ := 9
def height : ℝ := 8

-- Helper functions to compute areas
def area_rhombus (d1 d2 : ℝ) : ℝ := (d1 * d2) / 2
def area_triangle (b h : ℝ) : ℝ := (1 / 2) * b * h

-- The proof problem statement
theorem total_area_combined (d1 d2 b h : ℝ) :
  area_rhombus d1 d2 + area_triangle b h = 306 :=
by
  have h1 : area_rhombus d1 d2 = 270 := by
    unfold area_rhombus
    norm_num
  have h2 : area_triangle b h = 36 := by
    unfold area_triangle
    norm_num
  rw [h1, h2]
  norm_num

end total_area_combined_l266_266034


namespace count_squares_cubes_less_than_1000_l266_266812

theorem count_squares_cubes_less_than_1000 : 
  {n : ℕ | ∃ x : ℕ, n = x^6 ∧ n < 1000}.card = 3 :=
sorry

end count_squares_cubes_less_than_1000_l266_266812


namespace upper_square_side_length_twice_l266_266030

-- Define the polyhedron with given faces
structure Polyhedron :=
(faces : List (Σ (n : ℕ), (fin n → EuclideanGeometry.Point ℝ)))

-- Define a function to check the specific type of polyhedron described
def is_given_polyhedron (P : Polyhedron) : Prop :=
  (P.faces.filter (λ f => f.1 = 5)).length = 4 ∧  -- Four pentagons
  (P.faces.filter (λ f => f.1 = 3)).length = 4 ∧  -- Four triangles
  (P.faces.filter (λ f => f.1 = 4)).length = 2    -- Two squares

-- Define side length relation
def side_length_relation (P : Polyhedron) (s₁ s₂ : ℝ) : Prop :=
  ∀ (face₁ face₂ : Subtype (λ f => f.1 = 4)), face₁ ∈ P.faces ∧ face₂ ∈ P.faces →
  (face₁ = s₁ ∧ face₂ = 2 * s₁) 

-- State the theorem
theorem upper_square_side_length_twice (P : Polyhedron) (s₁ s₂ : ℝ) :
  is_given_polyhedron P →
  s₁ = 1 →
  side_length_relation P s₁ s₂ →
  s₂ = √2 * s₁ :=
by 
  sorry

end upper_square_side_length_twice_l266_266030


namespace coefficient_x2_y6_l266_266912

theorem coefficient_x2_y6 : 
  let x y : ℚ
  in let expr := (1 - y / x) * (x + y)^8
  in polynomial.coeff (polynomial.expand x^2 y^6 expr) = -28 := 
by
  sorry

end coefficient_x2_y6_l266_266912


namespace quadratic_reciprocal_sum_l266_266972

theorem quadratic_reciprocal_sum :
  ∃ (x1 x2 : ℝ), (x1^2 - 5 * x1 + 4 = 0) ∧ (x2^2 - 5 * x2 + 4 = 0) ∧ (x1 ≠ x2) ∧ (x1 + x2 = 5) ∧ (x1 * x2 = 4) ∧ (1 / x1 + 1 / x2 = 5 / 4) :=
sorry

end quadratic_reciprocal_sum_l266_266972


namespace sum_series_eq_51_l266_266114

-- Define x_i as i / 101
def x (i : ℕ) : ℚ := i / 101

-- Define the series S to be proved
def S : ℚ := ∑ i in Finset.range 102, x i ^ 3 / (3 * x i ^ 2 - 3 * x i + 1)

theorem sum_series_eq_51 : S = 51 := by
  sorry

end sum_series_eq_51_l266_266114


namespace range_of_f_range_of_a_l266_266147

-- Definition of f(x)
def f (a b : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 - a * x^2 + 3 * x + b

-- Range problem (I)
theorem range_of_f (a : ℝ) (b : ℝ) : 
  (a = 2) ∧ (b = 0) → 
  ∀ x ∈ set.Icc 0 3, f a b x ∈ set.Icc 0 (4/3) := 
sorry

-- Range problem (II)
theorem range_of_a (b : ℝ) (a : ℝ) : 
  (∀ x : ℝ, |f a b x| - (2/3) == 0 → finite {x}) → 
  -2 ≤ a ∧ a ≤ 2 := 
sorry

end range_of_f_range_of_a_l266_266147


namespace expected_distinct_points_l266_266959

open MeasureTheory
open Asymptotics

def bernoulli_rv {Ω : Type*} (p : ℝ) [measurable_space Ω] [probability_space Ω] : Ω → bool :=
λ ω, if ω ≤ p then true else false

noncomputable def Xi (Ω : Type*) (p : ℝ) [measurable_space Ω] [probability_space Ω] (i : ℕ) : Ω → ℤ :=
λ ω, if bernoulli_rv p ω then 1 else -1

noncomputable def Sn (Ω : Type*) (p : ℝ) [measurable_space Ω] [probability_space Ω] (i : ℕ) : (ℕ → ℤ) :=
λ n, ∑ k in finset.range (n + 1), Xi Ω p k

noncomputable def distinct_points {Ω : Type*} (N : ℕ) (p : ℝ) [measurable_space Ω] [probability_space Ω] : ℕ :=
(finset.range (N + 1)).image (λ n, Sn Ω p n).card

theorem expected_distinct_points (p : ℝ) : 
  p = 1/2 → 
  (λ N, (expectation (λ ω, distinct_points N p))) ~ 
  (λ N, sqrt (2 * N / π)) :=
by sorry

end expected_distinct_points_l266_266959


namespace count_squares_and_cubes_l266_266646

theorem count_squares_and_cubes (bound : ℕ) (hk : bound = 1000) : 
  Nat.card {n : ℕ | n < bound ∧ (∃ k : ℕ, n = k ^ 6)} = 3 := 
by
  sorry

end count_squares_and_cubes_l266_266646


namespace number_of_sixth_powers_lt_1000_l266_266697

-- Define the condition for being a sixth power and being less than 1000
def isSixthPower (n : ℕ) : Prop := ∃ k : ℕ, n = k^6

-- Define the main statement to be proved
theorem number_of_sixth_powers_lt_1000 : 
  {n : ℕ | n < 1000 ∧ isSixthPower n}.to_finset.card = 3 :=
by
  sorry

end number_of_sixth_powers_lt_1000_l266_266697


namespace probability_greater_than_30_l266_266552

def S : set ℕ := {1, 2, 3, 4}

-- Define a predicate that checks if a number is a two-digit number formed by two different elements of the set S
def is_valid_two_digit (n : ℕ) : Prop :=
  ∃ a b : ℕ, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ n = 10 * a + b

-- Define a predicate that checks if a number is greater than 30
def greater_than_30 (n : ℕ) : Prop := n > 30

-- Define the set of all possible two-digit numbers from the set S
def possible_numbers : set ℕ := {n | is_valid_two_digit n}

-- Define the set of two-digit numbers that are greater than 30
def favorable_numbers : set ℕ := {n | is_valid_two_digit n ∧ greater_than_30 n}

-- Prove that the probability of forming a two-digit number greater than 30 is 1/2
theorem probability_greater_than_30 : 
  ∃ (total_cases favorable_cases : ℕ), 
  total_cases = 12 ∧ 
  favorable_cases = 6 ∧ 
  ((favorable_cases : ℚ) / total_cases) = (1 / 2) :=
sorry

end probability_greater_than_30_l266_266552


namespace number_of_pairs_of_different_genres_l266_266856

theorem number_of_pairs_of_different_genres :
  let mystery := 4
  let fantasy := 3
  let biographies := 3
  (mystery * fantasy) + (mystery * biographies) + (fantasy * biographies) = 33 := 
by
  let mystery := 4
  let fantasy := 3
  let biographies := 3
  have h1 : mystery * fantasy = 4 * 3 := rfl
  have h2 : mystery * biographies = 4 * 3 := rfl
  have h3 : fantasy * biographies = 3 * 3 := rfl
  show (mystery * fantasy) + (mystery * biographies) + (fantasy * biographies) = 33 from by
    rw [h1, h2, h3]
    norm_num
  sorry

end number_of_pairs_of_different_genres_l266_266856


namespace max_consecutive_integers_sum_l266_266397

theorem max_consecutive_integers_sum (n : ℕ) : 
  (∃ n, (n * (n + 1) / 2 < 500) ∧ (∀ m > n, m * (m + 1) / 2 ≥ 500)) :=
by {
  use 31,
  split,
  {
    norm_num,
  },
  {
    intros m hm,
    have h : m = 32 := by linarith,
    rw h,
    norm_num,
  },
  sorry
}

end max_consecutive_integers_sum_l266_266397


namespace area_of_rhombus_correct_l266_266038

noncomputable def height_of_equilateral_triangle (s : ℝ) : ℝ := (s * Real.sqrt 3) / 2

-- Problem Definitions
def side_length_square : ℝ := 4
def side_length_triangle : ℝ := side_length_square -- Triangles are equilateral and sides are equal to square sides
def height_triangle : ℝ := height_of_equilateral_triangle side_length_triangle

-- Intermediate Calculations
def vertical_overlap : ℝ := 2 * height_triangle - side_length_square

-- Rhombus Diagonals
def rhombus_diagonal1 : ℝ := side_length_square
def rhombus_diagonal2 : ℝ := vertical_overlap

-- Expected Area Calculation
def area_of_rhombus : ℝ := (rhombus_diagonal1 * rhombus_diagonal2) / 2
def expected_area : ℝ := 8 * Real.sqrt 3 - 8

-- Theorem to Prove
theorem area_of_rhombus_correct : area_of_rhombus = expected_area := by
  sorry

end area_of_rhombus_correct_l266_266038


namespace volume_of_cube_l266_266288

noncomputable def cost_face_A : ℕ := 12
noncomputable def cost_face_B : ℕ := 13
noncomputable def cost_face_C : ℕ := 14
noncomputable def cost_face_D : ℕ := 15
noncomputable def cost_face_E : ℕ := 16
noncomputable def cost_face_F : ℕ := 17
noncomputable def total_cost_paise : ℕ := 51234

theorem volume_of_cube (a : ℝ) (h : 6 * a ^ 2 = total_cost_paise / 87) : 
  a^3 = 589 * real.sqrt 589 :=
sorry

end volume_of_cube_l266_266288


namespace number_of_lines_passing_through_M_and_tangent_to_parabola_l266_266465

-- Definitions for the conditions
def M := (2, 4)
def parabola (x y : ℝ) := y^2 = 8 * x

-- The statement of the proof problem
theorem number_of_lines_passing_through_M_and_tangent_to_parabola : 
  ∃ (lines : ℕ), lines = 2 ∧ 
  (∀ (l : ℝ × ℝ → ℝ), 
    (l M.1 M.2 = 0) → 
    ∃ x y, parabola x y ∧ l x y = 0 ∧ ∀ z, l z.1 z.2 = 0 → (x, y) = (z.1, z.2)) := 
sorry

end number_of_lines_passing_through_M_and_tangent_to_parabola_l266_266465


namespace find_n_l266_266344

theorem find_n (n : ℕ) :
  Int.lcm n 16 = 52 ∧ Nat.gcd n 16 = 8 → n = 26 :=
by
  sorry

end find_n_l266_266344


namespace number_of_sixth_powers_less_than_1000_l266_266633

theorem number_of_sixth_powers_less_than_1000 : 
  {n : ℕ | n < 1000 ∧ ∃ (k : ℕ), n = k^6}.finite.card = 3 :=
by sorry

end number_of_sixth_powers_less_than_1000_l266_266633


namespace CD_eq_AG_l266_266331

open Real

variables (A B C D F G : ℝ)

-- Conditions derived from the given problem
axiom h1 : A = 1 * (π / 6) -- Point A at 1 o'clock (30 degrees)
axiom h2 : B = 5 * (π / 6) -- Point B at 5 o'clock (150 degrees)
axiom h3 : C = 8 * (π / 6) -- Point C at 8 o'clock (240 degrees)

axiom h4 : D is the foot of the altitude from A to BC
axiom h5 : F is the foot of the altitude from C to AB
axiom h6 : G is the projection of F onto AC

theorem CD_eq_AG : dist C D = dist A G := sorry

end CD_eq_AG_l266_266331


namespace count_sixth_powers_below_1000_l266_266823

-- Definition and necessary condition for perfect sixth power
def is_sixth_power (n : ℕ) : Prop := ∃ k : ℕ, n = k ^ 6

-- Main statement of the problem
theorem count_sixth_powers_below_1000 :
  {n : ℕ | n < 1000 ∧ is_sixth_power n}.to_finset.card = 3 := 
sorry

end count_sixth_powers_below_1000_l266_266823


namespace triangle_tan_inequality_l266_266257

theorem triangle_tan_inequality 
  {A B C : ℝ} 
  (h1 : π / 2 ≠ A) 
  (h2 : A ≥ B) 
  (h3 : B ≥ C) : 
  |Real.tan A| ≥ Real.tan B ∧ Real.tan B ≥ Real.tan C := 
  by
    sorry

end triangle_tan_inequality_l266_266257


namespace find_possible_values_for_P_l266_266238

theorem find_possible_values_for_P (x y P : ℕ) (h1 : x < y) :
  P = (x^3 - y) / (1 + x * y) → (P = 0 ∨ P ≥ 2) :=
by
  sorry

end find_possible_values_for_P_l266_266238


namespace distance_between_foci_l266_266304

-- Let the hyperbola be defined by the equation xy = 4.
def hyperbola (x y : ℝ) : Prop := x * y = 4

-- Prove that the distance between the foci of this hyperbola is 8.
theorem distance_between_foci : ∀ (x y : ℝ), hyperbola x y → ∃ d, d = 8 :=
by {
    sorry
}

end distance_between_foci_l266_266304


namespace max_value_range_a_l266_266142

theorem max_value_range_a (a : ℝ) :
  (∀ x ∈ Icc (-2 : ℝ) 3, 
    (if x ≤ 0 then (2 * x^3 + 3 * x^2 + 1) else (Real.exp (a * x))) ≤ 2) ↔ 
  a ∈ Iic ((1 / 3) * Real.log 2) :=
by
  sorry

end max_value_range_a_l266_266142


namespace count_squares_and_cubes_less_than_1000_l266_266676

theorem count_squares_and_cubes_less_than_1000 : 
  {n : ℕ // n^6 < 1000}.card = 3 :=
by
  sorry

end count_squares_and_cubes_less_than_1000_l266_266676


namespace range_of_a_l266_266574

section
variables {a x : ℝ}

def p : Prop := ∀ x > 0, log a (x + 1) ≤ log a (0 + 1)
def q : Prop := (2 * a - 3)^2 - 4 > 0

theorem range_of_a (h1 : a > 0) (h2 : a ≠ 1) (h3 : p ∨ q) (h4 : ¬(p ∧ q)) :
  (1 / 2 ≤ a ∧ a < 1) ∨ (5 / 2 < a) :=
begin
  sorry
end
end

end range_of_a_l266_266574


namespace max_area_triangle_min_area_quadrilateral_l266_266118

open Real

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := (x^2) / 2 + y^2 = 1

-- Define points A and B
def points_on_line_through_ellipse (l : ℝ → ℝ) (A B : ℝ × ℝ) : Prop :=
  ellipse A.1 A.2 ∧ ellipse B.1 B.2 ∧ l A.1 = A.2 ∧ l B.1 = B.2

-- Define the set of lines L
def lines_making_max_triangle_area (l : (ℝ → ℝ)) : Prop :=
  ∃ A B : ℝ × ℝ, points_on_line_through_ellipse l A B ∧ 
  ((1/2) * abs (A.1 * B.2 - A.2 * B.1) = (sqrt 2) / 2)

-- Problem 1: Maximum area of the triangle AOB
theorem max_area_triangle : ∀ (l : ℝ → ℝ), (∃ A B : ℝ × ℝ, points_on_line_through_ellipse l A B) →
  (∃ l, lines_making_max_triangle_area l) :=
sorry

-- Problem 2: Minimum area of quadrilateral formed by lines in L
theorem min_area_quadrilateral (k1 k2 k3 k4 : ℝ) (hk : k1 + k2 + k3 + k4 = 0) :
  ∃ (l1 l2 l3 l4 : ℝ → ℝ),
    lines_making_max_triangle_area l1 ∧
    lines_making_max_triangle_area l2 ∧
    lines_making_max_triangle_area l3 ∧
    lines_making_max_triangle_area l4 ∧
    ((∀ x, l1 x = x * k1 + sqrt(k1^2 + 1/2)) ∨ (l1 x = x * k1 - sqrt(k1^2 + 1/2))) ∧
    ((∀ x, l2 x = x * k2 + sqrt(k2^2 + 1/2)) ∨ (l2 x = x * k2 - sqrt(k2^2 + 1/2))) ∧
    ((∀ x, l3 x = x * k3 + sqrt(k3^2 + 1/2)) ∨ (l3 x = x * k3 - sqrt(k3^2 + 1/2))) ∧
    ((∀ x, l4 x = x * k4 + sqrt(k4^2 + 1/2)) ∨ (l4 x = x * k4 - sqrt(k4^2 + 1/2))) ∧
    ((abs (k1) > 0 ∧ abs (k2) > 0 ∧ abs (k3) > 0 ∧ abs (k4) > 0)) :=
sorry

end max_area_triangle_min_area_quadrilateral_l266_266118


namespace express_a_in_terms_of_b_l266_266175

noncomputable def a : ℝ := Real.log 1250 / Real.log 6
noncomputable def b : ℝ := Real.log 50 / Real.log 3

theorem express_a_in_terms_of_b : a = (b + 0.6826) / 1.2619 :=
by
  sorry

end express_a_in_terms_of_b_l266_266175


namespace degree_of_product_l266_266525

noncomputable def p1 : ℤ[ℤ] := X^5

noncomputable def p2 : ℤ[ℤ] := X + 1 / X

noncomputable def p3 : ℤ[ℤ] := 1 + 3 / X + 4 / (X^2) + 5 / (X^3)

theorem degree_of_product :
  (p1 * p2 * p3).degree = 6 := by
  sorry

end degree_of_product_l266_266525


namespace count_sixth_powers_below_1000_l266_266827

-- Definition and necessary condition for perfect sixth power
def is_sixth_power (n : ℕ) : Prop := ∃ k : ℕ, n = k ^ 6

-- Main statement of the problem
theorem count_sixth_powers_below_1000 :
  {n : ℕ | n < 1000 ∧ is_sixth_power n}.to_finset.card = 3 := 
sorry

end count_sixth_powers_below_1000_l266_266827


namespace cornbread_pieces_l266_266949

theorem cornbread_pieces (pan_length : ℕ) (pan_width : ℕ) (piece_length : ℕ) (piece_width : ℕ)
  (hl : pan_length = 20) (hw : pan_width = 18) (hp : piece_length = 2) (hq : piece_width = 2) :
  (pan_length * pan_width) / (piece_length * piece_width) = 90 :=
by
  sorry

end cornbread_pieces_l266_266949


namespace zoo_individuals_left_l266_266457

/-!
A fifth-grade class went on a field trip to the zoo, and their class of 10 students merged with another class with the same number of students.
5 parents offered to be a chaperone, and 2 teachers from both classes will be there too.
When the school day was over, 10 of the students left. Two of the chaperones, who were parents in that group, also left.
-/

theorem zoo_individuals_left (students_per_class chaperones teachers students_left chaperones_left : ℕ)
  (h1 : students_per_class = 10)
  (h2 : chaperones = 5)
  (h3 : teachers = 2)
  (h4 : students_left = 10)
  (h5 : chaperones_left = 2) : 
  let total_students := students_per_class * 2,
      total_initial := total_students + chaperones + teachers,
      total_remaining := total_initial - students_left - chaperones_left
  in
  total_remaining = 15 := by
  sorry

end zoo_individuals_left_l266_266457


namespace find_p_l266_266861

theorem find_p 
  (a : ℝ) (p : ℕ) 
  (h1 : 12345 * 6789 = a * 10^p)
  (h2 : 1 ≤ a) (h3 : a < 10) (h4 : 0 < p) 
  : p = 7 := 
sorry

end find_p_l266_266861


namespace coefficient_x2_y6_l266_266916

theorem coefficient_x2_y6 : 
  let x y : ℚ
  in let expr := (1 - y / x) * (x + y)^8
  in polynomial.coeff (polynomial.expand x^2 y^6 expr) = -28 := 
by
  sorry

end coefficient_x2_y6_l266_266916


namespace coefficient_x2_y6_in_expansion_l266_266932

theorem coefficient_x2_y6_in_expansion :
  ∀ (x y : ℝ), x ≠ 0 → 
    (let c8_6 := (nat.factorial 8) / ((nat.factorial 6) * (nat.factorial 2)),
         c8_5 := (nat.factorial 8) / ((nat.factorial 5) * (nat.factorial 3)) in
     c8_6 - c8_5 = -28) :=
by
  intros x y hx_ne_zero
  let c8_6 := (nat.factorial 8) / ((nat.factorial 6) * (nat.factorial 2))
  let c8_5 := (nat.factorial 8) / ((nat.factorial 5) * (nat.factorial 3))
  have h : c8_6 - c8_5 = -28 := sorry
  exact h

end coefficient_x2_y6_in_expansion_l266_266932


namespace lenny_has_39_left_l266_266221

/-- Define the initial amount Lenny has -/
def initial_amount : ℕ := 84

/-- Define the amount Lenny spent on video games -/
def spent_on_video_games : ℕ := 24

/-- Define the amount Lenny spent at the grocery store -/
def spent_on_groceries : ℕ := 21

/-- Define the total amount Lenny spent -/
def total_spent : ℕ := spent_on_video_games + spent_on_groceries

/-- Calculate the amount Lenny has left -/
def amount_left (initial amount_spent : ℕ) : ℕ :=
  initial - amount_spent

/-- The statement of our mathematical equivalent proof problem
  Prove that Lenny has $39 left given the initial amount,
  and the amounts spent on video games and groceries.
-/
theorem lenny_has_39_left :
  amount_left initial_amount total_spent = 39 :=
by
  -- Leave the proof as 'sorry' for now
  sorry

end lenny_has_39_left_l266_266221


namespace car_interval_length_l266_266284

theorem car_interval_length (S1 T : ℝ) (interval_length : ℝ) 
  (h1 : S1 = 39) 
  (h2 : (fun (n : ℕ) => S1 - 3 * n) 4 = 27)
  (h3 : 3.6 = 27 * T) 
  (h4 : interval_length = T * 60) :
  interval_length = 8 :=
by
  sorry

end car_interval_length_l266_266284


namespace max_consecutive_integers_sum_lt_500_l266_266369

theorem max_consecutive_integers_sum_lt_500 
  (n : ℕ)
  (h₁ : ∑ k in Finset.range (n+1), k = n * (n + 1) / 2)
  (h₂ : n * (n + 1) / 2 < 500) 
  (h₃ : (n + 1) * (n + 2) / 2 > 500)
  : n = 31 := 
by sorry

end max_consecutive_integers_sum_lt_500_l266_266369


namespace coefficient_x2y6_l266_266918

/-- In the expansion of (1 - y / x) * (x + y) ^ 8, the coefficient of x ^ 2 * y ^ 6 is -28. -/
theorem coefficient_x2y6 (x y : ℚ) :  
  coefficient ((1 - y / x) * (x + y)^8) (monomial (2, 6)) = -28 :=
sorry

end coefficient_x2y6_l266_266918


namespace count_sixth_powers_below_1000_l266_266822

-- Definition and necessary condition for perfect sixth power
def is_sixth_power (n : ℕ) : Prop := ∃ k : ℕ, n = k ^ 6

-- Main statement of the problem
theorem count_sixth_powers_below_1000 :
  {n : ℕ | n < 1000 ∧ is_sixth_power n}.to_finset.card = 3 := 
sorry

end count_sixth_powers_below_1000_l266_266822


namespace eval_abs_expression_l266_266532

theorem eval_abs_expression (x : ℝ) (h : x ≤ -3) : |2 - |2 + x|| = 4 + x := 
by
  sorry

end eval_abs_expression_l266_266532


namespace probability_between_r_and_s_l266_266255

noncomputable def pq := 1 -- Let's assume the length of PQ is 1 unit for simplicity.

-- Conditions
def ps : ℝ := pq / 4
def qr : ℝ := pq / 8

-- Length of RS
def rs : ℝ := pq - ps - qr

-- Probability of selecting a point between R and S on PQ
theorem probability_between_r_and_s : rs / pq = 5 / 8 := 
by 
  sorry

end probability_between_r_and_s_l266_266255


namespace six_good_implies_seven_good_l266_266193

-- Define 6-good
def six_good (society : Type) (knows : society → society → Prop) : Prop :=
  ∀ (six_people : Finset society), six_people.card = 6 →
  ∃ (p : (six_people : set society) → six_people → Prop) (circ : list society),
    circ.perm six_people.to_list ∧ list.cycle (equiv_fun_on Finset.to_set p) circ

-- Define 7-good
def seven_good (society : Type) (knows : society → society → Prop) : Prop :=
  ∀ (seven_people : Finset society), seven_people.card = 7 →
  ∃ (p : (seven_people : set society) → seven_people → Prop) (circ : list society),
    circ.perm seven_people.to_list ∧ list.cycle (equiv_fun_on Finset.to_set p) circ

-- Theorem statement
theorem six_good_implies_seven_good (society : Type) (knows : society → society → Prop) :
  six_good society knows → seven_good society knows := 
by
  sorry

end six_good_implies_seven_good_l266_266193


namespace number_of_sixth_powers_lt_1000_l266_266693

-- Define the condition for being a sixth power and being less than 1000
def isSixthPower (n : ℕ) : Prop := ∃ k : ℕ, n = k^6

-- Define the main statement to be proved
theorem number_of_sixth_powers_lt_1000 : 
  {n : ℕ | n < 1000 ∧ isSixthPower n}.to_finset.card = 3 :=
by
  sorry

end number_of_sixth_powers_lt_1000_l266_266693


namespace count_positive_integers_square_and_cube_lt_1000_l266_266767

theorem count_positive_integers_square_and_cube_lt_1000 : 
  {n : ℕ | n < 1000 ∧ ∃ k : ℕ, n = k^6}.to_finset.card = 3 :=
by
  sorry

end count_positive_integers_square_and_cube_lt_1000_l266_266767


namespace count_sixth_powers_below_1000_l266_266627

theorem count_sixth_powers_below_1000 : 
  {n : ℕ | (n < 1000) ∧ (∃ m : ℕ, n = m ^ 6)}.to_finset.card = 3 :=
by sorry

end count_sixth_powers_below_1000_l266_266627


namespace angle_CAD_30_l266_266049

theorem angle_CAD_30
  (A B C G F : Type)
  (BC_eq_BG : BC = BG)
  (ABC_equilateral : ∀ {X Y Z : Type}, internal_angle X Y Z = 60)
  (BCFG_rectangle : is_rectangle B C F G)
  (angle_GBC_eq_80 : internal_angle G B C = 80) :
  internal_angle C A D = 30 := by
  sorry

end angle_CAD_30_l266_266049


namespace solution_set_g_lt_6_range_of_a_given_opposite_values_l266_266155

open Real

def f (a x : ℝ) : ℝ := 3 * |x - a| + |3 * x + 1|
def g (x : ℝ) : ℝ := |4 * x - 1| - |x + 2|

theorem solution_set_g_lt_6 : {x : ℝ | g x < 6} = {x : ℝ | -7/5 < x ∧ x < 3} :=
by
  sorry

theorem range_of_a_given_opposite_values :
  (∃ (x1 x2 : ℝ), f a x1 = -g x2) → -13/12 ≤ a ∧ a ≤ 5/12 :=
by
  sorry

end solution_set_g_lt_6_range_of_a_given_opposite_values_l266_266155


namespace count_positive_integers_square_and_cube_lt_1000_l266_266763

theorem count_positive_integers_square_and_cube_lt_1000 : 
  {n : ℕ | n < 1000 ∧ ∃ k : ℕ, n = k^6}.to_finset.card = 3 :=
by
  sorry

end count_positive_integers_square_and_cube_lt_1000_l266_266763


namespace card_arrangement_possible_l266_266334

theorem card_arrangement_possible (n : ℕ) 
  (card_numbers : Finset (Fin n)) 
  (each_number_twice : ∀ k : Fin n, 2 = (card_numbers.filter (λ x, x = k)).card) : 
  ∃ arrangement : List (Fin n), (card_numbers = arrangement.to_finset) ∧ (Set.Univ ⊆ arrangement.to_set) :=
by
  sorry

end card_arrangement_possible_l266_266334


namespace no_intersection_ln_ax_two_intersections_ln_ax_l266_266123

/-- Let f(x) = ln x and g(x) = a*x with a ∈ ℝ.
If the graphs of y = f(x) and y = g(x) do not intersect, then a in the range of (1/e, +∞) -/
theorem no_intersection_ln_ax {a : ℝ} (h : ∀ x : ℝ, x > 0 → ln x ≠ a * x) :
  a ∈ Set.Ioi (1 / Real.exp 1) :=
sorry

/-- Let f(x) = ln x and g(x) = a*x with a ∈ ℝ.
If there exist two real numbers x1 and x2 such that x1 ≠ x2 and both functions intersect, 
then x1*x2 > e^2. -/
theorem two_intersections_ln_ax {a x1 x2 : ℝ} (h1 : 0 < x1) (h2 : 0 < x2) (h3 : x1 ≠ x2)
    (h4 : ln x1 = a * x1) (h5 : ln x2 = a * x2) : 
  x1 * x2 > Real.exp 2 := 
sorry

end no_intersection_ln_ax_two_intersections_ln_ax_l266_266123


namespace count_positive_integers_square_and_cube_lt_1000_l266_266772

theorem count_positive_integers_square_and_cube_lt_1000 : 
  {n : ℕ | n < 1000 ∧ ∃ k : ℕ, n = k^6}.to_finset.card = 3 :=
by
  sorry

end count_positive_integers_square_and_cube_lt_1000_l266_266772


namespace coefficient_x2y6_l266_266924

/-- In the expansion of (1 - y / x) * (x + y) ^ 8, the coefficient of x ^ 2 * y ^ 6 is -28. -/
theorem coefficient_x2y6 (x y : ℚ) :  
  coefficient ((1 - y / x) * (x + y)^8) (monomial (2, 6)) = -28 :=
sorry

end coefficient_x2y6_l266_266924


namespace reaction_entropy_increases_l266_266580

theorem reaction_entropy_increases :
  ∀ (ΔH : ℝ) (ΔS : ℝ) (T : ℝ),
  ΔH = 2171 * 1000 ∧ ΔS = 635.5 ∧ T = 298 →
  ΔS > 0 :=
begin
  intros ΔH ΔS T h,
  cases h with h1 h2,
  cases h2 with h3 h4,
  rw h3,
  norm_num,
  linarith
end

end reaction_entropy_increases_l266_266580


namespace tetrahedron_surface_area_ratio_l266_266185

-- Define the problem details in Lean 4 statement
theorem tetrahedron_surface_area_ratio (S₁ S₂ : ℝ) (S₁_is_surface_area : S₁ = 4 * (sqrt 3) * a^2) (S₂_is_surface_area : S₂ = (π * a^2) / 6)
  (a > 0) : S₁ / S₂ = 6 * (sqrt 3) / π :=
by 
  sorry

end tetrahedron_surface_area_ratio_l266_266185


namespace max_x_minus_y_isosceles_l266_266050

theorem max_x_minus_y_isosceles (x y : ℝ) (hx : x ≠ 50) (hy : y ≠ 50) 
  (h_iso1 : x = y ∨ 50 = y) (h_iso2 : x = y ∨ 50 = x)
  (h_triangle : 50 + x + y = 180) : 
  max (x - y) (y - x) = 30 :=
sorry

end max_x_minus_y_isosceles_l266_266050


namespace max_value_f_l266_266558

theorem max_value_f (x y : ℝ) (h : x^2 + y^2 = 25) : 
  ∃ M, (∀ x y, x^2 + y^2 = 25 → f(x, y) ≤ M) ∧ M = 6 * Real.sqrt 10 :=
sorry

noncomputable def f (x y : ℝ) : ℝ :=
  Real.sqrt (8 * y - 6 * x + 50) + Real.sqrt (8 * y + 6 * x + 50)

end max_value_f_l266_266558


namespace foci_distance_l266_266303

def hyperbola (x y : ℝ) := x * y = 4

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

theorem foci_distance :
  distance (2, 2) (-2, -2) = 4 * real.sqrt 2 :=
by
  sorry

end foci_distance_l266_266303


namespace coefficient_of_x2y6_l266_266898

theorem coefficient_of_x2y6 :
  let binom_coeff8 (r : ℕ) : ℕ := Nat.choose 8 r
  ((1 - y / x) * (x + y)^8).coeff (2, 6) = -28 :=
by
  sorry

end coefficient_of_x2y6_l266_266898


namespace statement_a_statement_b_statement_c_statement_d_l266_266412

open_locale big_operators -- to use summations comfortably

namespace CombinatorialProofs

variables {n m : ℕ}

-- Define the binomial coefficient
def binom : ℕ → ℕ → ℕ
| n 0 := 1
| 0 k := 0
| (n+1) (k+1) := binom n k + binom n (k+1)

-- Define the permutation coefficient
def perm : ℕ → ℕ → ℕ
| n 0     := 1
| 0 k     := 0
| (n + 1) (k + 1) := (n + 1) * perm n k

-- Statement A: C(20, 3) = C(20, 17)
theorem statement_a : binom 20 3 = binom 20 17 := sorry

-- Statement B: Disprove: C(20, 1) + C(20, 2) + ... + C(20, 20) = 2^20
theorem statement_b : ¬((∑ i in finset.range 20 \ {0}, binom 20 i.succ) = 2^20) := sorry

-- Statement C: Disprove: A(n, m-1) = n! / (n-m-1)!
theorem statement_c : ¬(perm n (m-1) = nat.factorial n / nat.factorial (n-m-1)) := sorry

-- Statement D: (n+1) * A(n, m) = A(n+1, m+1)
theorem statement_d : (n + 1) * perm n m = perm (n + 1) (m + 1) := sorry

end CombinatorialProofs

end statement_a_statement_b_statement_c_statement_d_l266_266412


namespace largest_trailing_zeros_l266_266059

def count_trailing_zeros (n : Nat) : Nat :=
  if n = 0 then 0
  else Nat.min (Nat.factorial (n / 10)) (Nat.factorial (n / 5))

theorem largest_trailing_zeros :
  (count_trailing_zeros (4^3 * 5^6 * 6^5) > count_trailing_zeros (2^5 * 3^4 * 5^6)) ∧
  (count_trailing_zeros (4^3 * 5^6 * 6^5) > count_trailing_zeros (2^4 * 3^4 * 5^5)) ∧
  (count_trailing_zeros (4^3 * 5^6 * 6^5) > count_trailing_zeros (4^2 * 5^4 * 6^3)) :=
  sorry

end largest_trailing_zeros_l266_266059


namespace maximum_distance_line_l266_266291

noncomputable def line_equation (A B : ℝ × ℝ) : ℝ × ℝ → Prop :=
λ (X : ℝ × ℝ), (B.1 - A.1) * (X.2 - A.2) + (A.2 - B.2) * (X.1 - A.1) = 0

theorem maximum_distance_line (A B : ℝ × ℝ) (hA : A = (2, 1)) (hB : B = (1, 3)) : 
  line_equation A B (x, y) ↔ x - 2 * y = 0 := 
by
  sorry

end maximum_distance_line_l266_266291


namespace count_sixth_powers_below_1000_l266_266618

theorem count_sixth_powers_below_1000 : 
  {n : ℕ | (n < 1000) ∧ (∃ m : ℕ, n = m ^ 6)}.to_finset.card = 3 :=
by sorry

end count_sixth_powers_below_1000_l266_266618


namespace coplanar_k_values_l266_266253

noncomputable def coplanar_lines (k : ℝ) : Prop :=
  let v1 : ℝ × ℝ × ℝ := (2, 3, -k)
  let v2 : ℝ × ℝ × ℝ := (k, -1, 2)
  let p1 : ℝ × ℝ × ℝ := (3, 2, 2)
  let p2 : ℝ × ℝ × ℝ := (1, 5, 6)
  let ℓ1 : ℝ × ℝ × ℝ := (v1.1 * p1.1 + v1.2 * p1.2 + v1.3 * p1.3, v1.1 * p1.1 + v1.2 * p1.2 + v1.3 * p1.3, v1.1 * p1.1 + v1.2 * p1.2 + v1.3 * p1.3)
  let ℓ2 : ℝ × ℝ × ℝ := (v2.1 * p2.1 + v2.2 * p2.2 + v2.3 * p2.3, v2.1 * p2.1 + v2.2 * p2.2 + v2.3 * p2.3, v2.1 * p2.1 + v2.2 * p2.2 + v2.3 * p2.3)
  let d : ℝ × ℝ × ℝ := (p2.1 - p1.1, p2.2 - p1.2, p2.3 - p1.3)
  let determinant := d.1*v1.2*v2.3 + d.2*v1.3*v2.1 + d.3*v1.1*v2.2 - d.3*v1.2*v2.1 - d.2*v1.1*v2.3 - d.1*v1.3*v2.2
  determinant = 0

theorem coplanar_k_values (k : ℝ) : coplanar_lines k ↔ (k = -5 - 3 * real.sqrt 3 ∨ k = -5 + 3 * real.sqrt 3) :=
by
  sorry

end coplanar_k_values_l266_266253


namespace transformed_stats_l266_266581

variables (x : ℕ → ℝ) (y : ℕ → ℝ)

def average (n : ℕ) (f : ℕ → ℝ) := ∑ i in finset.range n, f i / n
def variance (n : ℕ) (f : ℕ → ℝ) := ∑ i in finset.range n, (f i - average n f) ^ 2 / n

theorem transformed_stats :
  (average 2022 x = 3) →
  (variance 2022 x = 56) →
  (∀ i, y i = 2 * x i + 3) →
  (average 2022 y = 9 ∧ variance 2022 y = 224) :=
by
  intros h_avg h_var h_trans
  sorry

end transformed_stats_l266_266581


namespace range_of_a_l266_266555

theorem range_of_a
  (a : ℝ)
  (h : ∃ x : ℝ, x > 0 ∧ (a * exp x + 3 = 0)) :
  -3 < a ∧ a < 0 :=
sorry

end range_of_a_l266_266555


namespace count_sixth_powers_less_than_1000_l266_266741

theorem count_sixth_powers_less_than_1000 : 
  {n : ℕ | 0 < n ∧ n < 1000 ∧ (∃ k : ℕ, n = k^6)}.to_finset.card = 3 :=
by 
  sorry

end count_sixth_powers_less_than_1000_l266_266741


namespace total_commuting_time_correct_l266_266209

def regular_walking_time : ℝ := 2
def regular_biking_time : ℝ := 1
def faster_pace_reduction : ℝ := 0.25
def longer_route_addition : ℝ := 0.5
def busy_road_addition : ℝ := 0.25
def rain_increase_factor : ℝ := 0.2
def rest_stop_walking : ℝ := 10 / 60
def rest_stop_biking : ℝ := (2 * 5) / 60
def total_commuting_time : ℝ :=
  let monday := ((regular_walking_time - faster_pace_reduction) * (1 + rain_increase_factor) + rest_stop_walking) * 2
  let tuesday := (regular_biking_time + rest_stop_biking) * 2
  let wednesday := (regular_walking_time + longer_route_addition + rest_stop_walking) * 2
  let thursday := ((regular_walking_time - faster_pace_reduction) * (1 + rain_increase_factor) + rest_stop_walking) * 2
  let friday := (regular_biking_time + busy_road_addition + rest_stop_biking) * 2
  monday + tuesday + wednesday + thursday + friday

theorem total_commuting_time_correct : total_commuting_time = 19.566 := by
  -- Proof goes here.
  sorry

end total_commuting_time_correct_l266_266209


namespace speed_of_second_car_l266_266341

theorem speed_of_second_car
  (t : ℝ)
  (distance_apart : ℝ)
  (speed_first_car : ℝ)
  (speed_second_car : ℝ)
  (h_total_distance : distance_apart = t * speed_first_car + t * speed_second_car)
  (h_time : t = 2.5)
  (h_distance_apart : distance_apart = 310)
  (h_speed_first_car : speed_first_car = 60) :
  speed_second_car = 64 := by
  sorry

end speed_of_second_car_l266_266341


namespace probability_of_x_plus_y_leq_6_l266_266025

theorem probability_of_x_plus_y_leq_6 : 
  let S := { p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 4 ∧ 0 ≤ p.2 ∧ p.2 ≤ 5 } in
  let T := { p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 4 ∧ 0 ≤ p.2 ∧ p.2 ≤ 5 ∧ p.1 + p.2 ≤ 6 } in
  (Set.cardinal T / Set.cardinal S : ℝ) = 1 / 2 :=
by
  sorry

end probability_of_x_plus_y_leq_6_l266_266025


namespace super_ball_distance_l266_266436

noncomputable def total_distance (n : ℕ) (h₀ : ℚ) (r : ℚ) : ℚ :=
let descents := List.range n |>.map (fun i => h₀ * r ^ i)
let ascents := List.range n |>.map (fun i => h₀ * r ^ (i+1))
descents.sum + ascents.sum

theorem super_ball_distance : total_distance 4 20 (2/3) ≈ 80 := sorry

end super_ball_distance_l266_266436


namespace earliest_time_100_degrees_l266_266187

def temperature (t : ℝ) : ℝ := -t^2 + 15 * t + 40

theorem earliest_time_100_degrees :
  ∃ t : ℝ, temperature t = 100 ∧ (∀ t' : ℝ, temperature t' = 100 → t' ≥ t) :=
by
  sorry

end earliest_time_100_degrees_l266_266187


namespace football_practice_hours_l266_266020

theorem football_practice_hours (practice_hours_per_day : ℕ) (days_per_week : ℕ) (missed_days_due_to_rain : ℕ) 
  (practice_hours_per_day_eq_six : practice_hours_per_day = 6)
  (days_per_week_eq_seven : days_per_week = 7)
  (missed_days_due_to_rain_eq_one : missed_days_due_to_rain = 1) : 
  practice_hours_per_day * (days_per_week - missed_days_due_to_rain) = 36 := 
by
  -- proof goes here
  sorry

end football_practice_hours_l266_266020


namespace count_squares_and_cubes_less_than_1000_l266_266735

theorem count_squares_and_cubes_less_than_1000 :
  {n : ℕ | 0 < n ∧ n < 1000 ∧ ∃ x, n = x ^ 6}.to_finset.card = 3 :=
by sorry

end count_squares_and_cubes_less_than_1000_l266_266735


namespace point_on_circle_distances_condition_l266_266973

theorem point_on_circle_distances_condition {R l : ℝ} (R_pos : 0 < R) (l_pos : l ≥ 0) :
  let f := λ x : ℝ, x^2 - 2*(l + R)*x + l^2 in
  (∃ M : ℝ × ℝ, ((M.1)^2 + (M.2)^2 = R^2) ∧ ((M.1 + √(M.1*(2*R - M.1))) = l))
  ↔ (l ≤ R*(Real.sqrt 2 + 1)) ∧
     (if l < 2*R then ∃ x : ℝ, f(x) = 0 ∧ x ≤ l
      else if l = 2*R then ∃ x1 x2 : ℝ, f(x1) = 0 ∧ f(x2) = 0
      else if l > 2*R ∧ l ≤ R*(Real.sqrt 2 + 1) then ∃ x1 x2 : ℝ, f(x1) = 0 ∧ f(x2) = 0 ∧ x1 < l ∧ x2 < l
      else false) := sorry

end point_on_circle_distances_condition_l266_266973


namespace count_squares_and_cubes_less_than_1000_l266_266727

theorem count_squares_and_cubes_less_than_1000 :
  {n : ℕ | 0 < n ∧ n < 1000 ∧ ∃ x, n = x ^ 6}.to_finset.card = 3 :=
by sorry

end count_squares_and_cubes_less_than_1000_l266_266727


namespace pirate_coins_total_l266_266989

theorem pirate_coins_total : ∀ x : ℕ, (1 + 2 + 3 + ... + x = 3 * x) → 4 * x = 20 := by
  sorry

end pirate_coins_total_l266_266989


namespace count_squares_and_cubes_l266_266645

theorem count_squares_and_cubes (bound : ℕ) (hk : bound = 1000) : 
  Nat.card {n : ℕ | n < bound ∧ (∃ k : ℕ, n = k ^ 6)} = 3 := 
by
  sorry

end count_squares_and_cubes_l266_266645


namespace sqrt5_parts_sum_parts_sqrt7_custom_op_plus_a_l266_266260

-- Define the integer and decimal parts of a root
def integer_part (x : ℝ) : ℤ := ⌊x⌋
def decimal_part (x : ℝ) : ℝ := x - ⌊x⌋

-- Define the custom operation ※
def custom_op (a b : ℝ) : ℝ := |a - b|

-- Problems
theorem sqrt5_parts :
  integer_part (Real.sqrt 5) = 2 ∧ decimal_part (Real.sqrt 5) = Real.sqrt 5 - 2 :=
by sorry

theorem sum_parts_sqrt7 (m n : ℝ) (h1 : m = Real.sqrt 7 - 2) (h2 : n = 2) :
  m + n - Real.sqrt 7 = 0 :=
by sorry

theorem custom_op_plus_a (a : ℝ) (b : ℝ) (h1 : a = Real.sqrt 3) (h2 : b = 3) :
  custom_op a b + a = 3 :=
by sorry

end sqrt5_parts_sum_parts_sqrt7_custom_op_plus_a_l266_266260


namespace count_sixth_powers_below_1000_l266_266834

-- Definition and necessary condition for perfect sixth power
def is_sixth_power (n : ℕ) : Prop := ∃ k : ℕ, n = k ^ 6

-- Main statement of the problem
theorem count_sixth_powers_below_1000 :
  {n : ℕ | n < 1000 ∧ is_sixth_power n}.to_finset.card = 3 := 
sorry

end count_sixth_powers_below_1000_l266_266834


namespace train_speed_computed_l266_266481

noncomputable def train_speed_in_kmh (train_length : ℝ) (platform_length : ℝ) (time_in_seconds : ℝ) : ℝ :=
  let total_distance := train_length + platform_length
  let speed_mps := total_distance / time_in_seconds
  speed_mps * 3.6

theorem train_speed_computed :
  train_speed_in_kmh 250 50.024 15 = 72.006 := by
  sorry

end train_speed_computed_l266_266481


namespace sequence_1995_eq_142_l266_266102

-- The least prime not dividing x
def p (x : ℕ) : ℕ := if x = 1 then 2 else
  List.find! (fun p => p.prime ∧ ¬ (p ∣ x)) [2, 3, 5, 7, 11, 13, 17, 19, 23, 29] -- This list can be extended as needed

-- The product of all primes less than p(x)
def q (x : ℕ) : ℕ := if p x = 2 then 1 else
  List.prod (List.filter (fun p => p.prime ∧ p < p x) [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]) -- Same as above

-- The sequence definition
def x (n : ℕ) : ℕ :=
  Nat.recOn n 1 (fun n x_n => x_n * p(x_n) / q(x_n))

-- The theorem to prove
theorem sequence_1995_eq_142 : ∀ n, x n = 1995 ↔ n = 142 :=
by
  intros n
  sorry

end sequence_1995_eq_142_l266_266102


namespace coefficient_x2y6_l266_266920

/-- In the expansion of (1 - y / x) * (x + y) ^ 8, the coefficient of x ^ 2 * y ^ 6 is -28. -/
theorem coefficient_x2y6 (x y : ℚ) :  
  coefficient ((1 - y / x) * (x + y)^8) (monomial (2, 6)) = -28 :=
sorry

end coefficient_x2y6_l266_266920


namespace num_sixth_powers_below_1000_l266_266803

theorem num_sixth_powers_below_1000 : 
  {n : ℕ | n^6 < 1000 ∧ n > 0}.to_finset.card = 3 :=
by sorry

end num_sixth_powers_below_1000_l266_266803


namespace minimum_erasures_correct_l266_266261

open Nat List

-- define a function that checks if a number represented as a list of digits is a palindrome
def is_palindrome (l : List ℕ) : Prop :=
  l = l.reverse

-- the given problem statement
def given_number := [1, 2, 3, 2, 3, 3, 1, 4]

-- function to find the minimum erasures to make a list a palindrome
noncomputable def min_erasures_to_palindrome (l : List ℕ) : ℕ :=
  sorry -- function implementation skipped

-- the main theorem statement
theorem minimum_erasures_correct : min_erasures_to_palindrome given_number = 3 :=
  sorry

end minimum_erasures_correct_l266_266261


namespace mass_of_15_moles_is_9996_9_l266_266060

/-- Calculation of the molar mass of potassium aluminum sulfate dodecahydrate -/
def KAl_SO4_2_12H2O_molar_mass : ℝ :=
  let K := 39.10
  let Al := 26.98
  let S := 32.07
  let O := 16.00
  let H := 1.01
  K + Al + 2 * S + (8 + 24) * O + 24 * H

/-- Mass calculation for 15 moles of potassium aluminum sulfate dodecahydrate -/
def mass_of_15_moles_KAl_SO4_2_12H2O : ℝ :=
  15 * KAl_SO4_2_12H2O_molar_mass

/-- Proof statement that the mass of 15 moles of potassium aluminum sulfate dodecahydrate is 9996.9 grams -/
theorem mass_of_15_moles_is_9996_9 : mass_of_15_moles_KAl_SO4_2_12H2O = 9996.9 := by
  -- assume KAl_SO4_2_12H2O_molar_mass = 666.46 (from the problem solution steps)
  sorry

end mass_of_15_moles_is_9996_9_l266_266060


namespace george_paints_room_l266_266198

theorem george_paints_room (n k: ℕ) (h_n: n = 10) (h_k: k = 3) :
  nat.choose n k = 120 :=
by {
  rw [h_n, h_k],
  calc nat.choose 10 3 = 120 : by norm_num
}

end george_paints_room_l266_266198


namespace fuel_remaining_l266_266579

-- Definitions given in the conditions of the original problem
def initial_fuel : ℕ := 48
def fuel_consumption_rate : ℕ := 8

-- Lean 4 statement of the mathematical proof problem
theorem fuel_remaining (x : ℕ) : 
  ∃ y : ℕ, y = initial_fuel - fuel_consumption_rate * x :=
sorry

end fuel_remaining_l266_266579


namespace number_of_sixth_powers_less_than_1000_l266_266837

theorem number_of_sixth_powers_less_than_1000 : 
  {n : ℕ | ∃ m : ℕ, n = m^6 ∧ n < 1000}.to_finset.card = 3 := 
by {
  sorry
}

end number_of_sixth_powers_less_than_1000_l266_266837


namespace smallest_d_l266_266425

theorem smallest_d (d : ℕ) (h_pos : 0 < d) (h_square : ∃ k : ℕ, 3150 * d = k^2) : d = 14 :=
sorry

end smallest_d_l266_266425


namespace train_length_approx_110_l266_266482

noncomputable def speedKmph := 60 -- Speed of the train in km/h
noncomputable def timeSec := 16.7986561075114 -- Time in seconds to cross the bridge
noncomputable def lengthBridge := 170 -- Length of the bridge in meters

noncomputable def speedMps : ℝ := speedKmph * 1000 / 3600 -- Convert speed from km/h to m/s
noncomputable def totalDistance : ℝ := speedMps * timeSec -- Distance covered by train while crossing the bridge
noncomputable def lengthTrain : ℝ := totalDistance - lengthBridge -- Length of the train

theorem train_length_approx_110 :
  abs (lengthTrain - 110) < 1e-7 := by
  sorry

end train_length_approx_110_l266_266482


namespace original_price_l266_266321

variables (p : ℝ)

theorem original_price (h : 1.1132 * p = 4) : p = 3.593 := 
by
sory

end original_price_l266_266321


namespace distance_to_other_focus_of_ellipse_l266_266178

noncomputable def ellipse_param (a b : ℝ) (x y : ℝ) := (x^2 / a^2) + (y^2 / b^2) = 1

def is_focus_distance (a distF1 distF2 : ℝ) : Prop :=
  ∀ P₁ P₂ : ℝ, distF1 + distF2 = 2 * a

theorem distance_to_other_focus_of_ellipse (x y : ℝ) (distF1 : ℝ) :
  ellipse_param 4 5 x y ∧ distF1 = 6 → is_focus_distance 5 distF1 4 :=
by
  simp [ellipse_param, is_focus_distance]
  sorry

end distance_to_other_focus_of_ellipse_l266_266178


namespace find_a_l266_266535
open Real

theorem find_a (a : ℝ) (k : ℤ) :
  (∃ x1 y1 x2 y2 : ℝ,
    (x1^2 + y1^2 = 10 * (x1 * cos a + y1 * sin a) ∧
     x2^2 + y2^2 = 10 * (x2 * sin (3 * a) + y2 * cos (3 * a)) ∧
     (x2 - x1)^2 + (y2 - y1)^2 = 64)) ↔
  (∃ k : ℤ, a = π / 8 + k * π / 2) :=
sorry

end find_a_l266_266535


namespace problem1_slope_problem2_monotonicity_l266_266007

-- Problem 1: Slope of the tangent line at x = 1 for a = 2
theorem problem1_slope (x : ℝ) : (∀ x, f(x) = 2 * x + ln x) → (deriv (λ x : ℝ, 2 * x + ln x) 1 = 3) := 
begin
  -- The statement asserts that the derivative of f(x) at x = 1 is 3
  sorry
end

-- Problem 2: Monotonicity intervals for f(x)
theorem problem2_monotonicity (a x : ℝ) (h_pos : x > 0) :
  (∀ x, f(x) = a * x + ln x) →
  (if a ≥ 0 then (∀ x > 0, deriv f x > 0) else (∀ x ∈ (0, -1/a) ∪ (-1/a, +∞), 
  (deriv f x > 0 → x ∈ (0, -1/a)) ∧ (deriv f x < 0 → x ∈ (-1/a, +∞)))) :=
sorry

-- Problem 3: Range of values for a under the given condition
lemma problem3_range_a (a : ℝ) :
  (∀ x (x1 ∈ Ioi 0), ∃ x2 ∈ Icc 0 1, f x1 < g x2) → (a < -1 / exp 1) :=
sorry

-- Definitions of functions f and g
def f (x : ℝ) : ℝ := a * x + ln x
def g (x : ℝ) : ℝ := x^2 - 2 * x + 2

end problem1_slope_problem2_monotonicity_l266_266007


namespace Bessie_ways_to_travel_5_5_l266_266058

def is_valid_point (p : ℕ × ℕ) : Prop :=
  p ≠ (1, 1) ∧ p ≠ (2, 3) ∧ p ≠ (3, 2)

def ways_to_reach (n m : ℕ) (p : ℕ × ℕ → Prop) : ℕ :=
  if (n, m) = (0, 0) then 1
  else if p (n, m) then
    (if n > 0 then ways_to_reach (n-1) m p else 0) +
    (if m > 0 then ways_to_reach n (m-1) p else 0)
  else 0

theorem Bessie_ways_to_travel_5_5 : ways_to_reach 5 5 is_valid_point = 32 :=
  sorry

end Bessie_ways_to_travel_5_5_l266_266058


namespace solution_set_condition_l266_266184

theorem solution_set_condition {a : ℝ} : 
  (∀ x : ℝ, (x > a ∧ x ≥ 3) ↔ (x ≥ 3)) → a < 3 := 
by 
  intros h
  sorry

end solution_set_condition_l266_266184


namespace marked_points_inside_2k_gon_l266_266880

def convex_hull {α : Type*} [linear_ordered_field α] (s : set (α × α)) : set (α × α) := sorry

theorem marked_points_inside_2k_gon :
  Π (k : ℕ) (polygon : list (ℝ × ℝ)) (marked_points : list (ℝ × ℝ)),
    100 = polygon.length → 
    2 ≤ k → k ≤ 50 → 
    ∃ (selected_vertices : list (ℝ × ℝ)), 
      2 * k = selected_vertices.length ∧ 
      ∀ (p ∈ marked_points), p ∈ convex_hull (set.of_list selected_vertices) :=
by
  intros k polygon marked_points polygon_length h2_le_k h_k_le_50
  sorry

end marked_points_inside_2k_gon_l266_266880


namespace sum_of_digits_of_M_l266_266224

open Nat

-- Definition of M being divisible by every positive integer less than 9
def is_divisible_by_all_less_than_9 (n : Nat) : Prop :=
  ∀ m, m < 9 ∧ m > 0 → m ∣ n

-- Definition of the second smallest multiple condition for M
def second_smallest_multiple (n : Nat) : Prop :=
  ∃ k1 k2, k1 > 0 ∧ k2 > 0 ∧ (lcm 1 2 3 4 5 6 7 8) * k1 = n ∧ (lcm 1 2 3 4 5 6 7 8) * k2 = 2 * n

-- Combined condition for M
def M (n : Nat) : Prop :=
  is_divisible_by_all_less_than_9 n ∧ second_smallest_multiple n

-- Statement: Sum of the digits of M
theorem sum_of_digits_of_M : ∃ m, M m ∧ digitSum m = 15 := sorry

end sum_of_digits_of_M_l266_266224


namespace number_of_sixth_powers_lt_1000_l266_266698

-- Define the condition for being a sixth power and being less than 1000
def isSixthPower (n : ℕ) : Prop := ∃ k : ℕ, n = k^6

-- Define the main statement to be proved
theorem number_of_sixth_powers_lt_1000 : 
  {n : ℕ | n < 1000 ∧ isSixthPower n}.to_finset.card = 3 :=
by
  sorry

end number_of_sixth_powers_lt_1000_l266_266698


namespace num_sixth_powers_below_1000_l266_266802

theorem num_sixth_powers_below_1000 : 
  {n : ℕ | n^6 < 1000 ∧ n > 0}.to_finset.card = 3 :=
by sorry

end num_sixth_powers_below_1000_l266_266802


namespace bicycle_discount_l266_266441

theorem bicycle_discount (P : ℝ) (hP : P > 0) :
  let new_price := 0.56 * P in
  let reduction := (P - new_price) / P * 100 in
  reduction = 44 := 
by
  -- Proof content goes here.
  sorry

end bicycle_discount_l266_266441


namespace first_more_than_200_paperclips_day_l266_266502

-- Definitions based on the conditions:
def paperclips_on_day (k : ℕ) : ℕ :=
  3 * 2^k

-- The theorem stating the solution:
theorem first_more_than_200_paperclips_day :
  ∃ k : ℕ, paperclips_on_day k > 200 ∧ k = 7 :=
by
  use 7
  sorry

end first_more_than_200_paperclips_day_l266_266502


namespace max_consecutive_sum_leq_500_l266_266377

theorem max_consecutive_sum_leq_500 : ∃ n : ℕ, (∑ i in finset.range (n+1), i) ≤ 500 ∧ (∀ m : ℕ, m > n → (∑ i in finset.range (m+1), i) > 500) :=
sorry

end max_consecutive_sum_leq_500_l266_266377


namespace slope_of_line_through_origin_and_A_l266_266865

theorem slope_of_line_through_origin_and_A :
  ∀ (x1 y1 x2 y2 : ℝ), (x1 = 0) → (y1 = 0) → (x2 = -2) → (y2 = -2) →
  (y2 - y1) / (x2 - x1) = 1 :=
by intros; sorry

end slope_of_line_through_origin_and_A_l266_266865


namespace count_sixth_powers_below_1000_l266_266832

-- Definition and necessary condition for perfect sixth power
def is_sixth_power (n : ℕ) : Prop := ∃ k : ℕ, n = k ^ 6

-- Main statement of the problem
theorem count_sixth_powers_below_1000 :
  {n : ℕ | n < 1000 ∧ is_sixth_power n}.to_finset.card = 3 := 
sorry

end count_sixth_powers_below_1000_l266_266832


namespace find_f_2017_l266_266575

def f : ℝ → ℝ
| x := if x < 0 then logBase 2 (-x) else f (x - 5)

theorem find_f_2017 : f 2017 = logBase 2 3 :=
by
  sorry

end find_f_2017_l266_266575


namespace angle_of_expression_l266_266492

noncomputable def cis (θ : ℝ) : ℂ :=
  complex.exp (complex.I * θ * real.pi / 180)

theorem angle_of_expression :
  ∃ r θ : ℝ,
    (60 ≤ θ ∧ θ < 360) ∧
    (∑ k in finset.range 10, cis (60 + k * 8)) = r * cis θ ∧
    θ = 96 :=
sorry

end angle_of_expression_l266_266492


namespace certain_event_l266_266410

theorem certain_event :
  ∃ event, event = "Drawing an arbitrary triangle in a workbook where the sum of the interior angles is 180°" :=
by
  use ("Drawing an arbitrary triangle in a workbook where the sum of the interior angles is 180°")
  sorry

end certain_event_l266_266410


namespace sales_and_profit_expression_determine_x_for_daily_profit_1200_feasibility_of_2000_daily_profit_l266_266445

variable (c : ℝ) (s : ℝ) (q : ℝ) (x : ℝ)

-- Conditions:
def cost_price := c
def original_selling_price := s
def original_sales := q
def price_reduction (x : ℝ) := x
def new_sales_per_day (q : ℝ) (x : ℝ) := q + 2 * x
def profit_per_piece (s : ℝ) (c : ℝ) (x : ℝ) := (s - x) - c

-- Question 1:
theorem sales_and_profit_expression (x : ℝ) : 
  new_sales_per_day q x = 20 + 2 * x ∧ profit_per_piece s c x = 40 - x :=
by
  sorry

-- Question 2 (solve for x):
def daily_profit (q : ℝ) (s : ℝ) (c : ℝ) (x : ℝ) := new_sales_per_day q x * profit_per_piece s c x

theorem determine_x_for_daily_profit_1200 (x : ℝ) : 
  daily_profit q s c x = 1200 → (x = 20 ∨ x = 10) :=
by
  sorry

-- Question 3 (determine feasibility):
theorem feasibility_of_2000_daily_profit : ¬ ∃ x, daily_profit q s c x = 2000 :=
by
  sorry

-- Parameters for initial problem setup
example : c = 80 ∧ s = 120 ∧ q = 20 :=
by
  split; exact dec_trivial

end sales_and_profit_expression_determine_x_for_daily_profit_1200_feasibility_of_2000_daily_profit_l266_266445


namespace graph_of_abs_g_l266_266522

def g (x : ℝ) : ℝ :=
  if -4 ≤ x ∧ x ≤ -1 then -1 - x / 2
  else if -1 < x ∧ x ≤ 3 then Real.sqrt (9 - (x - 3) ^ 2) - 3
  else if 3 < x ∧ x ≤ 5 then 1.5 * (x - 3)
  else 0

noncomputable def abs_g (x : ℝ) : ℝ := |g x|

theorem graph_of_abs_g :
  ∀ (x y : ℝ),
    ((-4 ≤ x ∧ x ≤ -1) → y = 1 + x / 2) ∨
    ((-1 < x ∧ x ≤ 3) → y = 3 - Real.sqrt (9 - (x - 3) ^ 2)) ∨
    ((3 < x ∧ x ≤ 5) → y = 1.5 * (x - 3)) →
    y = |g x| := by
sorry

end graph_of_abs_g_l266_266522


namespace relationship_between_a_b_c_l266_266111

def f (x : ℝ) : ℝ := (Real.log x + 1) / x

def a : ℝ := f 4
def b : ℝ := 2 / Real.exp 1
def c : ℝ := f Real.pi

theorem relationship_between_a_b_c : a < c ∧ c < b := 
by
  calc
    a = (Real.log 4 + 1) / 4 : by rfl
    c = (Real.log Real.pi + 1) / Real.pi : by rfl
    b = 2 / Real.exp 1 : by rfl
  sorry

end relationship_between_a_b_c_l266_266111


namespace ferris_wheel_large_seats_undetermined_l266_266277

theorem ferris_wheel_large_seats_undetermined
  (small_seats : ℕ)
  (large_seats : ℕ)
  (people_per_small_seat : ℕ)
  (people_per_large_seat : ℕ)
  (total_people_on_small_seats : ℕ)
  (total_small_seats : ℕ)
  (total_people : ℕ)
  (h1 : small_seats = 2)
  (h2 : large_seats >= 0)
  (h3 : people_per_small_seat = 14)
  (h4 : people_per_large_seat = 54)
  (h5 : total_people_on_small_seats = 28)
  (h6 : total_small_seats = total_people_on_small_seats / people_per_small_seat) :
  total_small_seats = small_seats :=
begin
  sorry,
end

end ferris_wheel_large_seats_undetermined_l266_266277


namespace number_of_squares_and_cubes_less_than_1000_l266_266720

noncomputable def count_squares_and_cubes : ℕ :=
  (List.range 1000).countp (λ n : ℕ, (∃ k : ℕ, k^6 = n))

theorem number_of_squares_and_cubes_less_than_1000 :
  count_squares_and_cubes = 3 :=
  by
    sorry

end number_of_squares_and_cubes_less_than_1000_l266_266720


namespace num_divisors_18060_l266_266612

def sum_of_numbers (a b : ℕ) : ℕ := a + b

def num_divisors (n : ℕ) : ℕ :=
  ∏ p in (n.factors.erase_dup), (n.factors.count p + 1)

theorem num_divisors_18060 : num_divisors (sum_of_numbers 9240 8820) = 48 := 
by
  sorry

end num_divisors_18060_l266_266612


namespace line_equation_l266_266088

theorem line_equation (x y : ℝ) : 
  (3 * x + y = 0) ∧ (x + y - 2 = 0) ∧ 
  ∃ m : ℝ, -2 = -(1 / m) ∧ 
  (∃ b : ℝ, (y = m * x + b) ∧ (3 = m * (-1) + b)) ∧ 
  x - 2 * y + 7 = 0 :=
sorry

end line_equation_l266_266088


namespace valid_triples_condition_l266_266086

theorem valid_triples_condition (x y z : ℕ) (h_cond : x ∈ {1, 2, 3, 4, 5, 6} ∧ y ∈ {1, 2, 3, 4, 5, 6} ∧ z ∈ {2, 4, 6}) :
  (100 * x + 10 * y + z = 2 * (49 * x + 7 * y + z)) → (x, y, z) ∈ {(3,1,2), (5,2,2), (4,1,4), (6,2,4), (5,1,6)} :=
begin
  sorry
end

end valid_triples_condition_l266_266086


namespace angle_between_points_on_sphere_l266_266073

def Point := (Real × Real)

def EarthRadius : Real -- The Earth's radius, R.
axiom R : Real

def P : Point := (0, 100) -- Point P at (0° latitude, 100° E longitude)
def Q : Point := (30, -100) -- Point Q at (30° N latitude, 100° W longitude)
def C : Point := (0, 0) -- Center of the Earth

noncomputable def angleCPQ (P Q C: Point) : Real := 
  let (latP, lonP) := P
  let (latQ, lonQ) := Q
  let lonDifference := abs (lonP - lonQ)
  if lonDifference > 180 then 360 - lonDifference else lonDifference
    
theorem angle_between_points_on_sphere :
  angleCPQ P Q C = 160 := 
begin
  -- The proof is not required as per instructions
  sorry
end

end angle_between_points_on_sphere_l266_266073


namespace calc_molecular_weight_l266_266061

/-- Atomic weights in g/mol -/
def atomic_weight (e : String) : Float :=
  match e with
  | "Ca"   => 40.08
  | "O"    => 16.00
  | "H"    => 1.01
  | "Al"   => 26.98
  | "S"    => 32.07
  | "K"    => 39.10
  | "N"    => 14.01
  | _      => 0.0

/-- Molecular weight calculation for specific compounds -/
def molecular_weight (compound : String) : Float :=
  match compound with
  | "Ca(OH)2"     => atomic_weight "Ca" + 2 * atomic_weight "O" + 2 * atomic_weight "H"
  | "Al2(SO4)3"   => 2 * atomic_weight "Al" + 3 * (atomic_weight "S" + 4 * atomic_weight "O")
  | "KNO3"        => atomic_weight "K" + atomic_weight "N" + 3 * atomic_weight "O"
  | _             => 0.0

/-- Given moles of different compounds, calculate the total molecular weight -/
def total_molecular_weight (moles : List (String × Float)) : Float :=
  moles.foldl (fun acc (compound, n) => acc + n * molecular_weight compound) 0.0

/-- The given problem -/
theorem calc_molecular_weight :
  total_molecular_weight [("Ca(OH)2", 4), ("Al2(SO4)3", 2), ("KNO3", 3)] = 1284.07 :=
by
  sorry

end calc_molecular_weight_l266_266061


namespace sum_of_an_plus_an1_l266_266938

def sequence (a : ℕ → ℝ) :=
  ∀ n : ℕ, a (n + 1) - a n = 2

def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (finset.range n).sum (λ i, a i)

theorem sum_of_an_plus_an1 (a : ℕ → ℝ) (S : ℕ → ℝ) :
  sequence a →
  S 10 = 50 →
  (finset.range 10).sum (λ i, a i + a (i + 1)) = 120 :=
by
  intros h_seq h_S10
  sorry

end sum_of_an_plus_an1_l266_266938


namespace compute_9_times_one_seventh_pow_4_l266_266504

theorem compute_9_times_one_seventh_pow_4 :
  9 * (1 / 7) ^ 4 = 9 / 2401 :=
by
  -- The actual proof would go here.
  sorry

end compute_9_times_one_seventh_pow_4_l266_266504


namespace goods_train_speed_l266_266470

noncomputable def passenger_train_speed := 64 -- in km/h
noncomputable def passing_time := 18 -- in seconds
noncomputable def goods_train_length := 420 -- in meters
noncomputable def relative_speed_kmh := 84 -- in km/h (derived from solution)

theorem goods_train_speed :
  (∃ V_g, relative_speed_kmh = V_g + passenger_train_speed) →
  (goods_train_length / (passing_time / 3600): ℝ) = relative_speed_kmh →
  V_g = 20 :=
by
  intro h1 h2
  sorry

end goods_train_speed_l266_266470


namespace num_sixth_powers_below_1000_l266_266789

theorem num_sixth_powers_below_1000 : 
  {n : ℕ | n^6 < 1000 ∧ n > 0}.to_finset.card = 3 :=
by sorry

end num_sixth_powers_below_1000_l266_266789


namespace maximum_value_quad_function_l266_266598

theorem maximum_value_quad_function (x : ℝ) : 
  let y := -2 * (x + 1) ^ 2 + 3 in 
  ∃ M : ℝ, (∀ x : ℝ, y ≤ M) ∧ (∃ x₀ : ℝ, y = M) := 
sorry

end maximum_value_quad_function_l266_266598


namespace series_sum_eq_l266_266097

noncomputable def series_sum (n : ℕ) : ℚ :=
  (∑ i in Finset.range n, 1 / ((i+2)^2 - 1))

theorem series_sum_eq (n : ℕ) (h : 0 < n) :
  series_sum n = 3/4 - 1/2 * (1/(n+1) + 1/(n+2)) :=
sorry

end series_sum_eq_l266_266097


namespace range_of_f_f_x_l266_266148

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ := if x < 0 then x^2 else -x^2 + 2 * x

-- Define the statement to be proven
theorem range_of_f_f_x (x : ℝ) : (f (f x) ≥ 9) ↔ (x ≤ -sqrt 3 ∨ x ≥ sqrt 3) := by
  sorry

end range_of_f_f_x_l266_266148


namespace number_of_squares_and_cubes_less_than_1000_l266_266710

noncomputable def count_squares_and_cubes : ℕ :=
  (List.range 1000).countp (λ n : ℕ, (∃ k : ℕ, k^6 = n))

theorem number_of_squares_and_cubes_less_than_1000 :
  count_squares_and_cubes = 3 :=
  by
    sorry

end number_of_squares_and_cubes_less_than_1000_l266_266710


namespace cylinder_height_in_pyramid_l266_266044

theorem cylinder_height_in_pyramid (a : Real) : 
  ∃ (r : Real), 2 * r = 2 * a / (Real.sqrt 2 + 2) :=
begin
  use a / (Real.sqrt 2 + 2),
  field_simp [mul_comm a 2, mul_assoc],
  rw [mul_div_cancel'],
  apply ne_of_gt,
  exact add_pos (Real.sqrt_pos.mpr zero_lt_two) zero_lt_two,
  exact (ne_of_gt zero_lt_two),
end

end cylinder_height_in_pyramid_l266_266044


namespace math_proof_problem_l266_266857

noncomputable def a : ℝ := 2 ^ 0.3
noncomputable def b : ℝ := Real.sin 2
noncomputable def c : ℝ := Real.log 0.3 / Real.log 2

theorem math_proof_problem : c < b ∧ b < a :=
by
  sorry

end math_proof_problem_l266_266857


namespace parallel_lines_iff_slope_eq_l266_266004

def line1 (m : ℝ) : ℝ × ℝ × ℝ := (3*m - 4, 4, -2)
def line2 (m : ℝ) : ℝ × ℝ × ℝ := (m, 2, -2)

def slope (a b c : ℝ) : ℝ := -a / b

theorem parallel_lines_iff_slope_eq (m : ℝ) :
  slope (3*m - 4) 4 (-2) = slope m 2 (-2) ↔ m = 4 := by
sorry

end parallel_lines_iff_slope_eq_l266_266004


namespace license_plate_palindrome_l266_266243

-- Define the components of the problem

def prob_digit_palindrome : ℚ := 1 / 100
def prob_letter_palindrome : ℚ := 13 / 8884

def prob_no_digit_palindrome : ℚ := 1 - prob_digit_palindrome
def prob_no_letter_palindrome : ℚ := 1 - prob_letter_palindrome
def prob_no_palindrome : ℚ := prob_no_digit_palindrome * prob_no_letter_palindrome

def prob_at_least_one_palindrome : ℚ := 1 - prob_no_palindrome

def reduced_fraction (pq : ℚ) : ℚ :=
  let num := pq.num.natAbs
  let den := pq.denom
  let g := Nat.gcd num den
  ⟨num / g, den / g⟩

def m : ℕ := (reduced_fraction prob_at_least_one_palindrome).num.natAbs
def n : ℕ := (reduced_fraction prob_at_least_one_palindrome).denom

def m_plus_n : ℕ := m + n

theorem license_plate_palindrome : m_plus_n = 897071 :=
by
  sorry

end license_plate_palindrome_l266_266243


namespace count_squares_and_cubes_less_than_1000_l266_266665

theorem count_squares_and_cubes_less_than_1000 : 
  {n : ℕ // n^6 < 1000}.card = 3 :=
by
  sorry

end count_squares_and_cubes_less_than_1000_l266_266665


namespace max_value_3a_plus_b_l266_266872

theorem max_value_3a_plus_b (a b : ℝ) :
  (∀ x : ℝ, x ∈ set.Icc 1 2 → abs (a * x^2 + b * x + a) ≤ x) →
  (3 * a + b ≤ 3) :=
by
  sorry

end max_value_3a_plus_b_l266_266872


namespace arithmetic_sequence_general_term_min_lambda_for_Tn_l266_266472

theorem arithmetic_sequence_general_term
    (h1 : ∃ d ≠ 0, S_3 = 9)
    (h2 : ∃a1 a2 a5, (a_1 (a_1 + 4 * d) = (a_1 + d)^2 ∧ a_1 ≠ 0 ∧ d ≠ 0)) :
    ∃ (a : ℕ → ℝ), ∀ n, a n = 2 * n - 1 :=
begin
  sorry
end

theorem min_lambda_for_Tn
    (h1 : ∀ (n : ℕ+), T_n = (1 / 2) * (1 - 1 / (2n + 1))) :
    ∃ λ, (∀ n ∈ ℕ+, T_n ≤ λ * a_(n + 1)) ∧ λ = 1 / 9 :=
begin
  sorry
end

end arithmetic_sequence_general_term_min_lambda_for_Tn_l266_266472


namespace least_points_tenth_game_l266_266197

theorem least_points_tenth_game 
  (points_6_to_9 : ℕ := 18 + 25 + 15 + 22)
  (total_games : ℕ := 10)
  (target_avg : ℕ := 20) 
  (total_points : ℕ := target_avg * total_games + 1) : 
  ∃ points_1_to_6 : ℕ, 
  points_1_to_6 ≤ 108 → 
  let required_tenth_game_points := total_points - points_6_to_9 - points_1_to_6
  in required_tenth_game_points = 13 :=
by
  sorry

end least_points_tenth_game_l266_266197


namespace coefficient_x2_y6_l266_266909

theorem coefficient_x2_y6 : 
  let x y : ℚ
  in let expr := (1 - y / x) * (x + y)^8
  in polynomial.coeff (polynomial.expand x^2 y^6 expr) = -28 := 
by
  sorry

end coefficient_x2_y6_l266_266909


namespace max_consecutive_sum_leq_500_l266_266375

theorem max_consecutive_sum_leq_500 : ∃ n : ℕ, (∑ i in finset.range (n+1), i) ≤ 500 ∧ (∀ m : ℕ, m > n → (∑ i in finset.range (m+1), i) > 500) :=
sorry

end max_consecutive_sum_leq_500_l266_266375


namespace sin_double_angle_solution_correct_l266_266571

theorem sin_double_angle_solution_correct (θ : ℝ) (h : 2^(-5/2 + 3 * cos θ) + 1 = 2^(1/2 + cos θ)) : 
  sin (2 * θ) = (5 * real.sqrt 11) / 18 := 
  sorry

end sin_double_angle_solution_correct_l266_266571


namespace ratio_Pat_Mark_l266_266988

-- Definitions inferred from the conditions
def total_hours : ℕ := 135
def Kate_hours (K : ℕ) : ℕ := K
def Pat_hours (K : ℕ) : ℕ := 2 * K
def Mark_hours (K : ℕ) : ℕ := K + 75

-- The main statement
theorem ratio_Pat_Mark (K : ℕ) (h : Kate_hours K + Pat_hours K + Mark_hours K = total_hours) :
  (Pat_hours K) / (Mark_hours K) = 1 / 3 := by
  sorry

end ratio_Pat_Mark_l266_266988


namespace count_squares_and_cubes_less_than_1000_l266_266667

theorem count_squares_and_cubes_less_than_1000 : 
  {n : ℕ // n^6 < 1000}.card = 3 :=
by
  sorry

end count_squares_and_cubes_less_than_1000_l266_266667


namespace find_length_of_second_movie_l266_266610

noncomputable def length_of_second_movie := 1.5

theorem find_length_of_second_movie
  (total_free_time : ℝ)
  (first_movie_duration : ℝ)
  (words_read : ℝ)
  (reading_rate : ℝ) : 
  first_movie_duration = 3.5 → 
  total_free_time = 8 → 
  words_read = 1800 → 
  reading_rate = 10 → 
  length_of_second_movie = 1.5 := 
by
  intros h1 h2 h3 h4
  -- Here should be the proof steps, which are abstracted away.
  sorry

end find_length_of_second_movie_l266_266610


namespace total_number_of_swallows_l266_266337

-- Definitions for the European and American swallows.
variables (E A : ℕ) -- number of European and American swallows

-- Conditions
def twice_as_many_A_as_E : Prop := A = 2 * E
def american_swallow_capacity : Prop := 5 * A
def european_swallow_capacity : Prop := 10 * E
def max_combined_weight_capacity (E A : ℕ) : Prop := 5 * A + 10 * E = 600

-- Statement
theorem total_number_of_swallows (h1 : twice_as_many_A_as_E E A) 
                                  (h2 : max_combined_weight_capacity E A) : 
  E + A = 90 :=
begin
  -- The proof would go here, but is not required for this task.
  sorry,
end

end total_number_of_swallows_l266_266337


namespace angle_C_triangle_area_l266_266884

noncomputable def sin : ℝ → ℝ := sorry -- assuming noncomputable definition for sin

theorem angle_C (a b c : ℝ) (A B C : ℝ)
(h_acute : A < 90 ∧ B < 90 ∧ C < 90)
(h_sides : a = opposite to angle A ∧ b = opposite to angle B ∧ c = opposite to angle C)
(h_relation : sqrt 3 * b = 2 * c * sin B) :
  C = 60 :=
sorry

theorem triangle_area (a b c : ℝ) (A B C : ℝ)
(h_acute : A < 90 ∧ B < 90 ∧ C < 90)
(h_sides : a = opposite to angle A ∧ b = opposite to angle B ∧ c = opposite to angle C)
(h_value_c : c = sqrt 6)
(h_sum_a_b : a + b = 3)
(h_angle_C : C = 60) :
  area a b C = sqrt 3 / 4 :=
sorry

end angle_C_triangle_area_l266_266884


namespace distance_between_foci_l266_266296

-- Define the equation of the hyperbola.
def hyperbola_eq (x y : ℝ) : Prop := x * y = 4

-- The coordinates of foci for hyperbola of the form x*y = 4
def foci_1 : (ℝ × ℝ) := (2, 2)
def foci_2 : (ℝ × ℝ) := (-2, -2)

-- Define the Euclidean distance function.
def euclidean_distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Prove that the distance between the foci is 4√2.
theorem distance_between_foci : euclidean_distance foci_1 foci_2 = 4 * real.sqrt 2 := sorry

end distance_between_foci_l266_266296


namespace maximum_ab_l266_266150

noncomputable def f (x a : ℝ) := real.exp x - a * x + a

theorem maximum_ab (a b : ℝ) (h : ∀ x : ℝ, f x a ≥ b) :
  a = real.exp (3 / 2) → b = 2 * a - a * real.log a → a * b = (real.exp 3) / 2 :=
by
  sorry

end maximum_ab_l266_266150


namespace hyperbola_asymptotes_l266_266595

theorem hyperbola_asymptotes (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a = √2 * b) :
  (∀ x y : ℝ, y = √2 * x ∨ y = -√2 * x) :=
sorry

end hyperbola_asymptotes_l266_266595


namespace problem_f_l266_266557

noncomputable def f (α : ℝ) : ℝ :=
  (cos (π / 2 + α) * cos (2 * π - α) * sin (-α + 3 / 2 * π)) / (sin (-π - α) * sin (3 / 2 * π + α))

theorem problem_f (α : ℝ) (hα1 : 3 * π / 2 < α ∧ α < 2 * π)
  (hα2 : cos (α - 3 * π / 2) = 1 / 5) :
  f α = 2 * real.sqrt 6 / 5 := 
sorry

end problem_f_l266_266557


namespace collections_of_9_letters_l266_266166

theorem collections_of_9_letters :
  (nat.choose 34 9) = (nat.choose 34 (9:ℕ)) :=
sorry

end collections_of_9_letters_l266_266166


namespace bell_rings_by_geography_l266_266985

def class_schedule := ["Maths", "History", "Geography", "Science", "Music"]
def bell_rings_for_class (class_name : String) : Bool :=
  class_name ∈ class_schedule ∧ ¬ class_name = "History"

theorem bell_rings_by_geography :
  count (λ class_name, bell_rings_for_class class_name) class_schedule ≤ 3 :=
begin
  -- The proof would normally be carried out here
  sorry
end

end bell_rings_by_geography_l266_266985


namespace maximum_consecutive_positive_integers_sum_500_l266_266382

theorem maximum_consecutive_positive_integers_sum_500 : 
  ∃ n : ℕ, (n * (n + 1) / 2 < 500) ∧ (∀ m : ℕ, (m * (m + 1) / 2 < 500) → m ≤ n) :=
sorry

end maximum_consecutive_positive_integers_sum_500_l266_266382


namespace both_square_and_cube_count_both_square_cube_less_than_1000_l266_266780

theorem both_square_and_cube (n : ℕ) (h_positive : n > 0) (h_bound : n < 1000) (h_square : ∃ k : ℕ, n = k^2) (h_cube : ∃ k : ℕ, n = k^3) :
  n ∈ {1, 64, 729} :=
sorry

theorem count_both_square_cube_less_than_1000 :
  {n : ℕ | n > 0 ∧ n < 1000 ∧ ∃ k : ℕ, n = k^2 ∧ ∃ k : ℕ, n = k^3}.to_finset.card = 3 :=
by
  have h_valid : ∀ n ∈ {1, 64, 729}, n > 0 ∧ n < 1000 ∧ (∃ k : ℕ, n = k^2) ∧ (∃ k : ℕ, n = k^3) := sorry
  have h_reverse : ∀ n, n > 0 ∧ n < 1000 ∧ (∃ k : ℕ, n = k^2) ∧ (∃ k : ℕ, n = k^3) → n ∈ {1, 64, 729} := sorry
  exact Finset.card_map _ {1, 64, 729}.to_finset
  sorry

end both_square_and_cube_count_both_square_cube_less_than_1000_l266_266780


namespace slope_of_line_is_neg_one_l266_266569

theorem slope_of_line_is_neg_one (y : ℝ) (h : (y - 5) / (5 - (-3)) = -1) : y = -3 :=
by
  sorry

end slope_of_line_is_neg_one_l266_266569


namespace range_of_f_x1_x2_l266_266153

theorem range_of_f_x1_x2 (a x1 x2 : ℝ) (h1 : 2 * (Real.exp 1 + Real.exp (-1)) < a)
  (h2 : a < 20 / 3) (hx_ext : f '' Set.Ioc (0:ℝ) a ∶ Set.Pair x1 x2)
  (hx_le : x1 < x2) :
  e^2 - (1/e^2) - 4 < f(x1) - f(x2) < 80/9 - 4* log 3 :=
by
  sorry

noncomputable def f (x : ℝ) := x^2 - a*x + 2 * log x

end range_of_f_x1_x2_l266_266153


namespace remainder_of_3_pow_244_mod_5_l266_266406

theorem remainder_of_3_pow_244_mod_5 : (3^244) % 5 = 1 := by
  sorry

end remainder_of_3_pow_244_mod_5_l266_266406


namespace count_squares_cubes_less_than_1000_l266_266817

theorem count_squares_cubes_less_than_1000 : 
  {n : ℕ | ∃ x : ℕ, n = x^6 ∧ n < 1000}.card = 3 :=
sorry

end count_squares_cubes_less_than_1000_l266_266817


namespace distance_between_foci_l266_266306

-- Let the hyperbola be defined by the equation xy = 4.
def hyperbola (x y : ℝ) : Prop := x * y = 4

-- Prove that the distance between the foci of this hyperbola is 8.
theorem distance_between_foci : ∀ (x y : ℝ), hyperbola x y → ∃ d, d = 8 :=
by {
    sorry
}

end distance_between_foci_l266_266306


namespace both_square_and_cube_count_both_square_cube_less_than_1000_l266_266773

theorem both_square_and_cube (n : ℕ) (h_positive : n > 0) (h_bound : n < 1000) (h_square : ∃ k : ℕ, n = k^2) (h_cube : ∃ k : ℕ, n = k^3) :
  n ∈ {1, 64, 729} :=
sorry

theorem count_both_square_cube_less_than_1000 :
  {n : ℕ | n > 0 ∧ n < 1000 ∧ ∃ k : ℕ, n = k^2 ∧ ∃ k : ℕ, n = k^3}.to_finset.card = 3 :=
by
  have h_valid : ∀ n ∈ {1, 64, 729}, n > 0 ∧ n < 1000 ∧ (∃ k : ℕ, n = k^2) ∧ (∃ k : ℕ, n = k^3) := sorry
  have h_reverse : ∀ n, n > 0 ∧ n < 1000 ∧ (∃ k : ℕ, n = k^2) ∧ (∃ k : ℕ, n = k^3) → n ∈ {1, 64, 729} := sorry
  exact Finset.card_map _ {1, 64, 729}.to_finset
  sorry

end both_square_and_cube_count_both_square_cube_less_than_1000_l266_266773


namespace find_x_l266_266858

theorem find_x (b x : ℝ) (hb : b > 1) (hx : x > 0)
  (h : (3 * x)^(Real.log 3 / Real.log b) - (5 * x)^(Real.log 5 / Real.log b) = 0) :
  x = 1 / 5^(Real.log b / Real.log (5 / 3)) :=
by
  sorry

end find_x_l266_266858


namespace number_of_sixth_powers_lt_1000_l266_266695

-- Define the condition for being a sixth power and being less than 1000
def isSixthPower (n : ℕ) : Prop := ∃ k : ℕ, n = k^6

-- Define the main statement to be proved
theorem number_of_sixth_powers_lt_1000 : 
  {n : ℕ | n < 1000 ∧ isSixthPower n}.to_finset.card = 3 :=
by
  sorry

end number_of_sixth_powers_lt_1000_l266_266695


namespace hiker_speed_third_day_l266_266021

-- Define the conditions
def first_day_distance : ℕ := 18
def first_day_speed : ℕ := 3
def second_day_distance : ℕ :=
  let first_day_hours := first_day_distance / first_day_speed
  let second_day_hours := first_day_hours - 1
  let second_day_speed := first_day_speed + 1
  second_day_hours * second_day_speed
def total_distance : ℕ := 53
def third_day_hours : ℕ := 3

-- Define the speed on the third day based on given conditions
def speed_on_third_day : ℕ :=
  let third_day_distance := total_distance - first_day_distance - second_day_distance
  third_day_distance / third_day_hours

-- The theorem we need to prove
theorem hiker_speed_third_day : speed_on_third_day = 5 := by
  sorry

end hiker_speed_third_day_l266_266021


namespace count_sixth_powers_below_1000_l266_266613

theorem count_sixth_powers_below_1000 : 
  {n : ℕ | (n < 1000) ∧ (∃ m : ℕ, n = m ^ 6)}.to_finset.card = 3 :=
by sorry

end count_sixth_powers_below_1000_l266_266613


namespace distinct_flags_count_l266_266019

theorem distinct_flags_count : 
  let colors := [red, white, blue, green, yellow],
      strips := 4
  in  ∃ c1 c2 c3 c4 : colors, 
         (c1 ≠ c2) ∧ 
         (c2 ≠ c3) ∧ 
         (c3 ≠ c4) ∧ 
         (fintype.card colors = 5) → (4 * 4 * 4 * 5 = 320) := sorry

end distinct_flags_count_l266_266019


namespace equal_dice_probability_l266_266435

noncomputable def probability_equal_one_and_two_digit (dice : Fin 5 → Fin 20) : ℚ :=
  let one_digit_prob := 9 / 20
  let two_digit_prob := 11 / 20
  let comb_factor : ℚ := Nat.choose 5 2
  let individual_prob : ℚ := (one_digit_prob ^ 2) * (two_digit_prob ^ 3)
  comb_factor * individual_prob

theorem equal_dice_probability :
  probability_equal_one_and_two_digit = 539055 / 1600000 :=
by
  sorry

end equal_dice_probability_l266_266435


namespace coeff_x2y6_in_expansion_l266_266902

theorem coeff_x2y6_in_expansion : 
  (coeff_of_term (x := 2) (y := 6)) ((1 - (y / x)) * (x + y) ^ 8) = -28 :=
sorry

end coeff_x2y6_in_expansion_l266_266902


namespace power_equation_l266_266173

theorem power_equation (p : ℕ) : 81^6 = 3^p → p = 24 :=
by
  intro h
  have h1 : 81 = 3^4 := by norm_num
  rw [h1] at h
  rw [pow_mul] at h
  norm_num at h
  exact eq_of_pow_eq_pow _ h

end power_equation_l266_266173


namespace daps_equivalent_to_66_dips_l266_266272

def daps := ℝ
def dops := ℝ
def dips := ℝ

def equivalent_daps (d : daps) (dop : dops) := 5 * d = 4 * dop
def equivalent_dops (dop : dops) (dip : dips) := 3 * dop = 11 * dip

theorem daps_equivalent_to_66_dips (d66 : dips) (dap : daps) (dop: dops)
  (h1 : equivalent_daps dap dop)
  (h2 : equivalent_dops dop d66) :
  d66 = 66 → dap = 22.5 :=
by
  -- Proof goes here
  sorry

end daps_equivalent_to_66_dips_l266_266272


namespace sum_of_squares_of_medians_correct_l266_266563

-- Define the side lengths of the triangle
def AB : ℝ := 13
def BC : ℝ := 15
def CA : ℝ := 14

-- Define the formula for the length of a median connecting a vertex to the midpoint of the opposite side
def median_length (a b c : ℝ) : ℝ :=
  (1 / 2) * Real.sqrt (2 * b^2 + 2 * c^2 - a^2)

-- Calculate the lengths of the medians
def m_a : ℝ := median_length BC CA AB
def m_b : ℝ := median_length CA AB BC
def m_c : ℝ := median_length AB BC CA

-- Calculate the squares of the medians
def m_a_sq : ℝ := m_a^2
def m_b_sq : ℝ := m_b^2
def m_c_sq : ℝ := m_c^2

-- Calculate the sum of the squares of the medians
def sum_of_squares_of_medians : ℝ := m_a_sq + m_b_sq + m_c_sq

-- Prove that the sum of the squares of the medians is 442.5
theorem sum_of_squares_of_medians_correct : sum_of_squares_of_medians = 442.5 :=
by
  sorry

end sum_of_squares_of_medians_correct_l266_266563


namespace count_6x6_arrays_l266_266165

-- Define the total number of configurations
def count_6x6_configs : ℕ := 
  578600

-- Prove the number of arrays with given properties
theorem count_6x6_arrays (n : ℕ) (A : matrix (fin n) (fin n) ℤ) (h1 : n = 6) 
  (h_entries : ∀ i j, A i j = 1 ∨ A i j = -1) 
  (h_row_sum : ∀ i, ∑ j, A i j = 0)
  (h_col_sum : ∀ j, ∑ i, A i j = 0) : 
  (fintype.card {A : matrix (fin 6) (fin 6) ℤ | 
    (∀ i j, A i j = 1 ∨ A i j = -1 ) ∧  (∀ i, ∑ j, A i j = 0) ∧ (∀ j, ∑ i, A i j = 0) }) = count_6x6_configs :=
sorry

end count_6x6_arrays_l266_266165


namespace total_cost_is_9_43_l266_266943

def basketball_game_cost : ℝ := 5.20
def racing_game_cost : ℝ := 4.23
def total_cost : ℝ := basketball_game_cost + racing_game_cost

theorem total_cost_is_9_43 : total_cost = 9.43 := by
  sorry

end total_cost_is_9_43_l266_266943


namespace probability_sum_greater_than_15_over_16_l266_266576

theorem probability_sum_greater_than_15_over_16 :
  (∀ x : ℝ, f(x) * g'(x) < f'(x) * g(x)) → 
  (∀ x : ℝ, f(x) = a^x * g(x)) → 
  (f(1) / g(1) + f(-1) / g(-1) = 5 / 2) → 
  (∃ (k : Fin 11) (n : ℕ), 
    ∑ i in range k.val, f(i) / g(i) > 15 / 16 → (k.val = 5)) := sorry

end probability_sum_greater_than_15_over_16_l266_266576


namespace sum_of_digits_l266_266200

theorem sum_of_digits (P Q R : ℕ) (hP : P < 10) (hQ : Q < 10) (hR : R < 10)
 (h_sum : P * 1000 + Q * 100 + Q * 10 + R = 2009) : P + Q + R = 10 :=
by
  -- The proof is omitted
  sorry

end sum_of_digits_l266_266200


namespace count_squares_cubes_less_than_1000_l266_266805

theorem count_squares_cubes_less_than_1000 : 
  {n : ℕ | ∃ x : ℕ, n = x^6 ∧ n < 1000}.card = 3 :=
sorry

end count_squares_cubes_less_than_1000_l266_266805


namespace opposite_pairs_l266_266047

theorem opposite_pairs : 
  (   ( ∃ x y: ℤ, x = (-3)^2 ∧ y = -3^2 ∧ x = 9 ∧ y = -9 ∧ x = -y ) 
   ∧ ¬( ∃ x y: ℤ, x = (-3)^2 ∧ y = 3^2 ∧ x = y )
   ∧ ¬( ∃ x y: ℤ, x = (-2)^3 ∧ y = -2^3 ∧ x = y )
   ∧ ¬( ∃ x y: ℤ, x = | -2 |^3 ∧ y = | -2^3 | ∧ x = y )
  ) :=
by{
    sorry
}

end opposite_pairs_l266_266047


namespace coefficient_x2_y6_in_expansion_l266_266926

theorem coefficient_x2_y6_in_expansion :
  ∀ (x y : ℝ), x ≠ 0 → 
    (let c8_6 := (nat.factorial 8) / ((nat.factorial 6) * (nat.factorial 2)),
         c8_5 := (nat.factorial 8) / ((nat.factorial 5) * (nat.factorial 3)) in
     c8_6 - c8_5 = -28) :=
by
  intros x y hx_ne_zero
  let c8_6 := (nat.factorial 8) / ((nat.factorial 6) * (nat.factorial 2))
  let c8_5 := (nat.factorial 8) / ((nat.factorial 5) * (nat.factorial 3))
  have h : c8_6 - c8_5 = -28 := sorry
  exact h

end coefficient_x2_y6_in_expansion_l266_266926


namespace inequality_solution_l266_266977

theorem inequality_solution (x : ℝ) (hx : x > 0) : 
  x + 1 / x ≥ 2 ∧ (x + 1 / x = 2 ↔ x = 1) := 
by
  sorry

end inequality_solution_l266_266977


namespace airplane_throw_competition_l266_266052
noncomputable theory

variables (a b h v m : ℝ)

theorem airplane_throw_competition :
  a + b + h + v + m = 41 ∧
  a = m + 0.9 ∧
  v = a + 0.6 ∧
  a + v + m = 24 ∧
  h > a ∧ h > b ∧ h > v ∧ h > m
  → (a, b, h, v, m) = (8.1, 8, 9, 8.7, 7.2) :=
begin
  sorry,
end

end airplane_throw_competition_l266_266052


namespace count_sixth_powers_less_than_1000_l266_266747

theorem count_sixth_powers_less_than_1000 : 
  {n : ℕ | 0 < n ∧ n < 1000 ∧ (∃ k : ℕ, n = k^6)}.to_finset.card = 3 :=
by 
  sorry

end count_sixth_powers_less_than_1000_l266_266747


namespace checkerboard_corners_sum_l266_266440

theorem checkerboard_corners_sum : 
  let checkerboard := (List.range 81).map (λ n, n + 2),
      top_left := checkerboard.head!,
      top_right := checkerboard.nth! 8,
      bottom_left := checkerboard.nth! 72,
      bottom_right := checkerboard.nth! 80
  in
    top_left + top_right + bottom_left + bottom_right = 168 := by
  sorry

end checkerboard_corners_sum_l266_266440


namespace count_squares_and_cubes_less_than_1000_l266_266732

theorem count_squares_and_cubes_less_than_1000 :
  {n : ℕ | 0 < n ∧ n < 1000 ∧ ∃ x, n = x ^ 6}.to_finset.card = 3 :=
by sorry

end count_squares_and_cubes_less_than_1000_l266_266732


namespace four_digit_sum_to_seven_l266_266546

theorem four_digit_sum_to_seven :
  (∃ a b c d : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9 ∧ a + b + c + d = 7) ↔ (84) := 
sorry

end four_digit_sum_to_seven_l266_266546


namespace coefficient_x2_y6_in_expansion_l266_266925

theorem coefficient_x2_y6_in_expansion :
  ∀ (x y : ℝ), x ≠ 0 → 
    (let c8_6 := (nat.factorial 8) / ((nat.factorial 6) * (nat.factorial 2)),
         c8_5 := (nat.factorial 8) / ((nat.factorial 5) * (nat.factorial 3)) in
     c8_6 - c8_5 = -28) :=
by
  intros x y hx_ne_zero
  let c8_6 := (nat.factorial 8) / ((nat.factorial 6) * (nat.factorial 2))
  let c8_5 := (nat.factorial 8) / ((nat.factorial 5) * (nat.factorial 3))
  have h : c8_6 - c8_5 = -28 := sorry
  exact h

end coefficient_x2_y6_in_expansion_l266_266925


namespace number_of_sixth_powers_less_than_1000_l266_266640

theorem number_of_sixth_powers_less_than_1000 : 
  {n : ℕ | n < 1000 ∧ ∃ (k : ℕ), n = k^6}.finite.card = 3 :=
by sorry

end number_of_sixth_powers_less_than_1000_l266_266640


namespace number_of_sixth_powers_lt_1000_l266_266703

-- Define the condition for being a sixth power and being less than 1000
def isSixthPower (n : ℕ) : Prop := ∃ k : ℕ, n = k^6

-- Define the main statement to be proved
theorem number_of_sixth_powers_lt_1000 : 
  {n : ℕ | n < 1000 ∧ isSixthPower n}.to_finset.card = 3 :=
by
  sorry

end number_of_sixth_powers_lt_1000_l266_266703


namespace conference_games_scheduled_l266_266278

theorem conference_games_scheduled
  (divisions : ℕ)
  (teams_per_division : ℕ)
  (intra_games_per_pair : ℕ)
  (inter_games_per_pair : ℕ)
  (h_div : divisions = 3)
  (h_teams : teams_per_division = 4)
  (h_intra : intra_games_per_pair = 3)
  (h_inter : inter_games_per_pair = 2) :
  let intra_division_games := (teams_per_division * (teams_per_division - 1) / 2) * intra_games_per_pair
  let intra_division_total := intra_division_games * divisions
  let inter_division_games := teams_per_division * (teams_per_division * (divisions - 1)) * inter_games_per_pair
  let inter_division_total := inter_division_games * divisions / 2
  let total_games := intra_division_total + inter_division_total
  total_games = 150 :=
by
  sorry

end conference_games_scheduled_l266_266278


namespace count_sixth_powers_below_1000_l266_266620

theorem count_sixth_powers_below_1000 : 
  {n : ℕ | (n < 1000) ∧ (∃ m : ℕ, n = m ^ 6)}.to_finset.card = 3 :=
by sorry

end count_sixth_powers_below_1000_l266_266620


namespace jo_kate_sum_difference_l266_266213

theorem jo_kate_sum_difference :
  let jo_sum := (120 * (120 + 1)) / 2
  let kate_sum := 24 * ((0 * 2) + (5 * 3))
  abs (jo_sum - kate_sum) = 6900 :=
by
  let jo_sum := (120 * (120 + 1)) / 2
  let kate_sum := 24 * ((0 * 2) + (5 * 3))
  show abs (jo_sum - kate_sum) = 6900
  sorry

end jo_kate_sum_difference_l266_266213


namespace find_line_equation_l266_266087

-- Definitions of the given points
def F := (2, 0)  -- Example point as per typical problem of intersection
def B := (1, 1)  -- Example point on the ellipse
def M := (x / (1 + 3 * k ^ 2), -12 * k ^ 2 / (1 + 3 * k ^ 2)) -- Definition based on the centroid condition

-- The ellipse equation condition
def ellipse (x y : ℝ) : Prop := (x^2 / 6) + (y^2 / 2) = 1

-- Centroid condition
def centroid (O : (ℝ × ℝ)) : Prop :=
  O = (0, 0) ∧ ∃ (A B M : (ℝ × ℝ)), 
    let (x1, y1) := A in
    let (x2, y2) := B in
    let (xm, ym) := M in
    (0, 0) = ((x1 + x2 + xm) / 3, (y1 + y2 + ym) / 3)

noncomputable def k := sqrt(5) / 5

-- The statement to prove
theorem find_line_equation (c a b : ℝ) (F B M : (ℝ × ℝ))
  (hF : F = (2, 0))
  (hB : B = (1, 1)) 
  (hM : M = (2 / (1 + 3 * k^2), -12 * k^2 / (1 + 3 * k^2))) 
  (hO : centroid (0, 0)) :
  y = ± sqrt(5) / 5 * x - 2 := sorry

end find_line_equation_l266_266087


namespace max_consecutive_integers_sum_lt_500_l266_266371

theorem max_consecutive_integers_sum_lt_500 
  (n : ℕ)
  (h₁ : ∑ k in Finset.range (n+1), k = n * (n + 1) / 2)
  (h₂ : n * (n + 1) / 2 < 500) 
  (h₃ : (n + 1) * (n + 2) / 2 > 500)
  : n = 31 := 
by sorry

end max_consecutive_integers_sum_lt_500_l266_266371


namespace infinite_coprime_sequence_exists_l266_266528

/-- The sequence a is defined such that
  a_1 = 1,
  a_2 = 7,
  a_{n+1} = (3 * a_n)! + 1.
-/
def sequence : ℕ → ℕ
| 1 := 1
| 2 := 7
| (n + 1) := (3 * sequence n)! + 1

/-- There exists an infinite increasing sequence of natural numbers
such that the sum of any two distinct terms is coprime with
the sum of any three distinct terms -/
theorem infinite_coprime_sequence_exists :
  ∃ a : ℕ → ℕ,
  (∀ n, a n < a (n + 1)) ∧
  (∀ i j p q r : ℕ, i ≠ j → i ≠ p → i ≠ q → i ≠ r →
                     j ≠ p → j ≠ q → j ≠ r →
                     p ≠ q → p ≠ r → q ≠ r →
  gcd (a i + a j) (a p + a q + a r) = 1) :=
by
  use sequence
  sorry

end infinite_coprime_sequence_exists_l266_266528


namespace aquariums_have_13_saltwater_animals_l266_266215

theorem aquariums_have_13_saltwater_animals:
  ∀ x : ℕ, 26 * x = 52 → (∀ n : ℕ, n = 26 → (n * x = 52 ∧ x % 2 = 1 ∧ x > 1)) → x = 13 :=
by
  sorry

end aquariums_have_13_saltwater_animals_l266_266215


namespace transaction_gain_per_year_l266_266474

theorem transaction_gain_per_year
  (principal : ℝ) (borrow_rate : ℝ) (lend_rate : ℝ) (time : ℕ)
  (principal_eq : principal = 5000)
  (borrow_rate_eq : borrow_rate = 0.04)
  (lend_rate_eq : lend_rate = 0.06)
  (time_eq : time = 2) :
  (principal * lend_rate * time - principal * borrow_rate * time) / time = 100 := by
  sorry

end transaction_gain_per_year_l266_266474


namespace number_of_sixth_powers_lt_1000_l266_266707

-- Define the condition for being a sixth power and being less than 1000
def isSixthPower (n : ℕ) : Prop := ∃ k : ℕ, n = k^6

-- Define the main statement to be proved
theorem number_of_sixth_powers_lt_1000 : 
  {n : ℕ | n < 1000 ∧ isSixthPower n}.to_finset.card = 3 :=
by
  sorry

end number_of_sixth_powers_lt_1000_l266_266707


namespace count_quadratic_polynomials_l266_266169

theorem count_quadratic_polynomials : 
  ∃ (s : Finset (ℤ × ℤ × ℤ)), s.card = 12 ∧
  ∀ (a b c : ℤ), (a, b, c) ∈ s ↔ 
    (∀ x ∈ set.Icc (0 : ℝ) 1, 0 ≤ a * x^2 + b * x + c ∧ a * x^2 + b * x + c ≤ 1) :=
sorry

end count_quadratic_polynomials_l266_266169


namespace perpendicular_EF_AB_minimize_area_l266_266466

-- Definitions based on conditions
def parabola (x y : ℝ) : Prop := y^2 = 2 * x
def focus : ℝ × ℝ := (1 / 2, 0)
def line_through_focus (n y : ℝ) : ℝ := n * y + 1 / 2

noncomputable def point_A (n : ℝ) : ℝ × ℝ :=
  let y := n + real.sqrt (n^2 + 1) in (n * y + 1 / 2, y)

noncomputable def point_B (n : ℝ) : ℝ × ℝ :=
  let y := n - real.sqrt (n^2 + 1) in (n * y + 1 / 2, y)

noncomputable def point_E (n : ℝ) : ℝ × ℝ := 
  ((point_A n).fst + (point_B n).fst / 2, n)

-- Proof Problem 1 (show EF ⊥ AB)
theorem perpendicular_EF_AB (n : ℝ) :
  let A := point_A n in
  let B := point_B n in
  let E := point_E n in
  ∃ kEF kAB : ℝ, kEF * kAB = -1 := sorry

-- Proof Problem 2 (minimize area of triangle ABE when λ in [1/3, 1/2])
theorem minimize_area (λ : ℝ) (hλ : λ ∈ set.Icc (1 / 3) (1 / 2)) :
  ∃ S_min : ℝ, S_min = (real.sqrt (1 / 8) + 1)^3 / 2 := sorry

end perpendicular_EF_AB_minimize_area_l266_266466


namespace total_hours_worked_l266_266211

-- Definitions based on conditions
def hourly_rate_after_school : ℝ := 4.0
def hourly_rate_saturday : ℝ := 6.0
def weekly_earnings : ℝ := 88.0
def saturday_hours_worked : ℝ := 8.0
def saturday_earnings := saturday_hours_worked * hourly_rate_saturday
def after_school_earnings := weekly_earnings - saturday_earnings
def after_school_hours_worked := after_school_earnings / hourly_rate_after_school

-- Proving the total hours worked in a week
theorem total_hours_worked :
  after_school_hours_worked + saturday_hours_worked = 18 := by
  sorry

end total_hours_worked_l266_266211


namespace coeff_x2y6_in_expansion_l266_266904

theorem coeff_x2y6_in_expansion : 
  (coeff_of_term (x := 2) (y := 6)) ((1 - (y / x)) * (x + y) ^ 8) = -28 :=
sorry

end coeff_x2y6_in_expansion_l266_266904


namespace arithmetic_identity_l266_266511

theorem arithmetic_identity : 72 * 989 - 12 * 989 = 59340 := by
  sorry

end arithmetic_identity_l266_266511


namespace unique_line_intercept_l266_266889

noncomputable def is_positive_integer (n : ℕ) : Prop := n > 0
noncomputable def is_prime (n : ℕ) : Prop := n = 2 ∨ (n > 2 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0)

theorem unique_line_intercept (a b : ℕ) :
  ((is_positive_integer a) ∧ (is_prime b) ∧ (6 * b + 5 * a = a * b)) ↔ (a = 11 ∧ b = 11) :=
by
  sorry

end unique_line_intercept_l266_266889


namespace max_consecutive_integers_sum_lt_500_l266_266368

theorem max_consecutive_integers_sum_lt_500 
  (n : ℕ)
  (h₁ : ∑ k in Finset.range (n+1), k = n * (n + 1) / 2)
  (h₂ : n * (n + 1) / 2 < 500) 
  (h₃ : (n + 1) * (n + 2) / 2 > 500)
  : n = 31 := 
by sorry

end max_consecutive_integers_sum_lt_500_l266_266368


namespace range_of_k_l266_266154

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.log x + x
noncomputable def g (x : ℝ) : ℝ := (1/2) * k * x^2 - x + 3
noncomputable def g' (x : ℝ) : ℝ := k * x - 1

theorem range_of_k (k : ℝ) :
  (x₀ : ℝ) (hx₀ : x₀ > 0), f x₀ = g' x₀ → k ≥ 2 :=
sorry

end range_of_k_l266_266154


namespace money_left_l266_266942

theorem money_left (initial amount spent remaining : ℝ) 
  (h1 : initial = 100.00) 
  (h2 : spent = 15.00) 
  (h3 : remaining = initial - spent) :
  remaining = 85.00 :=
by
  rw [h1, h2] at h3
  exact h3

end money_left_l266_266942


namespace count_sixth_powers_below_1000_l266_266616

theorem count_sixth_powers_below_1000 : 
  {n : ℕ | (n < 1000) ∧ (∃ m : ℕ, n = m ^ 6)}.to_finset.card = 3 :=
by sorry

end count_sixth_powers_below_1000_l266_266616


namespace total_students_in_school_l266_266192

theorem total_students_in_school : 
  let C1 := 32
  let C2 := C1 - 2
  let C3 := C2 - 2
  let C4 := C3 - 2
  let C5 := C4 - 2
  C1 + C2 + C3 + C4 + C5 = 140 :=
by
  let C1 := 32
  let C2 := C1 - 2
  let C3 := C2 - 2
  let C4 := C3 - 2
  let C5 := C4 - 2
  sorry

end total_students_in_school_l266_266192


namespace whole_number_between_interval_l266_266485

theorem whole_number_between_interval (M : ℕ) (h : 9 < M / 4 ∧ M / 4 < 10) : M ∈ {37, 38, 39} :=
sorry

end whole_number_between_interval_l266_266485


namespace primes_in_sequence_l266_266524

theorem primes_in_sequence :
  let seq (n : ℕ) := 47 * (10 ^ (2 * (n + 1)) - 1) / 99 in
  (finset.filter (λ n, nat.prime (seq n)) (finset.range 100)).card = 1 :=
by sorry

end primes_in_sequence_l266_266524


namespace both_square_and_cube_count_both_square_cube_less_than_1000_l266_266785

theorem both_square_and_cube (n : ℕ) (h_positive : n > 0) (h_bound : n < 1000) (h_square : ∃ k : ℕ, n = k^2) (h_cube : ∃ k : ℕ, n = k^3) :
  n ∈ {1, 64, 729} :=
sorry

theorem count_both_square_cube_less_than_1000 :
  {n : ℕ | n > 0 ∧ n < 1000 ∧ ∃ k : ℕ, n = k^2 ∧ ∃ k : ℕ, n = k^3}.to_finset.card = 3 :=
by
  have h_valid : ∀ n ∈ {1, 64, 729}, n > 0 ∧ n < 1000 ∧ (∃ k : ℕ, n = k^2) ∧ (∃ k : ℕ, n = k^3) := sorry
  have h_reverse : ∀ n, n > 0 ∧ n < 1000 ∧ (∃ k : ℕ, n = k^2) ∧ (∃ k : ℕ, n = k^3) → n ∈ {1, 64, 729} := sorry
  exact Finset.card_map _ {1, 64, 729}.to_finset
  sorry

end both_square_and_cube_count_both_square_cube_less_than_1000_l266_266785


namespace add_water_and_alcohol_l266_266439

open Classical

theorem add_water_and_alcohol :
  ∀ (x : ℝ),
    x = 4.5 →
    0.05 * 40 + 5.5 = 7.5 → 
    ∃ (final_sol : ℝ), 
      final_sol = 40 + 5.5 + x ∧ 
      7.5 / final_sol = 0.15 := 
by
  intros x hx h1
  use 40 + 5.5 + x
  split
  · rw [hx]
  · sorry

end add_water_and_alcohol_l266_266439


namespace reflected_area_double_l266_266254

variable {A B C D O : Type} 
variables (coords : A → B × B) (area_OABC : B)
variable (quad_area : B) 
variable (O_reflections : B → B) 

-- Conditions
-- 1. Point O inside a convex quadrilateral ABCD with area S
def is_point_inside (x : B) (quad : A) : Prop := sorry

def convex_quadrilateral (quad : A) : Prop := sorry

def area_of_quadrilateral (quad : A) : B := quad_area

-- 2. Midpoints of the sides of the quadrilateral
def midpoint (p1 p2 : B) : B := sorry

def reflect (p midpoint : B) : B := sorry

-- 3. Reflect O about the midpoints
def M1 := midpoint (coords A) (coords B)
def M2 := midpoint (coords B) (coords C)
def M3 := midpoint (coords C) (coords D)
def M4 := midpoint (coords D) (coords A)

def E1 := reflect coords(O) M1
def F1 := reflect coords(O) M2
def G1 := reflect coords(O) M3
def H1 := reflect coords(O) M4

-- Proof Problem
theorem reflected_area_double (h1 : convex_quadrilateral A)
    (h2 : ∀ p ∈ [A, B, C, D], is_point_inside O p)
    (h3 : S = area_of_quadrilateral [A, B, C, D]) :
  area_of_quadrilateral [E1, F1, G1, H1] = 2 * S :=
sorry

end reflected_area_double_l266_266254


namespace number_of_sixth_powers_less_than_1000_l266_266851

theorem number_of_sixth_powers_less_than_1000 : 
  {n : ℕ | ∃ m : ℕ, n = m^6 ∧ n < 1000}.to_finset.card = 3 := 
by {
  sorry
}

end number_of_sixth_powers_less_than_1000_l266_266851


namespace num_integers_lowest_terms_l266_266430

theorem num_integers_lowest_terms : 
  (∃ count : ℕ, count = (finset.filter (λ n, ∀ m : ℕ, 
  ((n^2 - 9).gcd (n^2 - 7) = 1)) (finset.Icc 10 100)).card = 46) := 
sorry

end num_integers_lowest_terms_l266_266430


namespace valid_rearrangement_count_l266_266853

open List

noncomputable def valid_rearrangements : Finset (List Char) :=
  {permutation |  
    permutation ~ ['a', 'b', 'c', 'd'] ∧
    ∀ i < (permutation.length - 1),
      let x := permutation.nth_le i ‹_›
      let y := permutation.nth_le (i + 1) (by linarith)
      in
      (x, y) ≠ ('a', 'b') ∧ (x, y) ≠ ('b', 'a') ∧
      (x, y) ≠ ('b', 'c') ∧ (x, y) ≠ ('c', 'b') ∧
      (x, y) ≠ ('c', 'd') ∧ (x, y) ≠ ('d', 'c')
  }.toFinset

theorem valid_rearrangement_count : valid_rearrangements.card = 4 := sorry

end valid_rearrangement_count_l266_266853


namespace geometric_sequence_sum_l266_266933

variable (a : ℕ → ℝ)

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r, ∀ n, a (n + 1) = a n * r

theorem geometric_sequence_sum (h1 : geometric_sequence a)
  (h2 : a 1 > 0)
  (h3 : a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 36) :
  a 3 + a 5 = 6 :=
sorry

end geometric_sequence_sum_l266_266933


namespace number_of_squares_and_cubes_less_than_1000_l266_266721

noncomputable def count_squares_and_cubes : ℕ :=
  (List.range 1000).countp (λ n : ℕ, (∃ k : ℕ, k^6 = n))

theorem number_of_squares_and_cubes_less_than_1000 :
  count_squares_and_cubes = 3 :=
  by
    sorry

end number_of_squares_and_cubes_less_than_1000_l266_266721


namespace central_angle_minor_arc_l266_266286

theorem central_angle_minor_arc (x y : ℝ) :
  (x^2 + y^2 = 4) → (sqrt 3 * x + y = 2 * sqrt 3) → ∃ θ, θ = π / 3 :=
by
  intro h_circle h_line
  use π / 3
  sorry

end central_angle_minor_arc_l266_266286


namespace geom_seq_l266_266202

theorem geom_seq
  (q : ℕ) (n : ℕ) (a : ℕ → ℕ)
  (h1 : ∀ i, a (i + 1) = a 1 * q ^ i)
  (h2 : a 1 * a 2 * a 3 * ... * a 30 = 2 ^ 30)
  (h3 : q = 2) :
  (a 3) * (a 6) * (a 9) * ... * (a 30) = 2 ^ 20 :=
sorry

end geom_seq_l266_266202


namespace maximum_consecutive_positive_integers_sum_500_l266_266388

theorem maximum_consecutive_positive_integers_sum_500 : 
  ∃ n : ℕ, (n * (n + 1) / 2 < 500) ∧ (∀ m : ℕ, (m * (m + 1) / 2 < 500) → m ≤ n) :=
sorry

end maximum_consecutive_positive_integers_sum_500_l266_266388


namespace complex_number_solution_l266_266239

theorem complex_number_solution :
  ∃ a b : ℝ, ∀ z : ℂ, 
  z = ((1 + Complex.i)^2 + 3 * (1 - Complex.i)) / (2 + Complex.i) ∧ 
  (z^2 + a * z + b = 1 + Complex.i) →
  (a = -3 ∧ b = 4) :=
by
  sorry

end complex_number_solution_l266_266239


namespace student_did_not_pass_then_scored_less_than_90_l266_266246

variable {Student : Type} (P Q : Student → Prop)

theorem student_did_not_pass_then_scored_less_than_90 
  (h : ∀ s : Student, Q s → P s) : 
  ∀ s : Student, ¬ Q s → ¬ P s :=
by
  intro s
  intros h1 h2
  exact h s h2 h1
  sorry

end student_did_not_pass_then_scored_less_than_90_l266_266246


namespace number_of_sixth_powers_lt_1000_l266_266689

theorem number_of_sixth_powers_lt_1000 : 
  {n : ℕ | ∃ k : ℕ, k^6 = n ∧ n < 1000}.to_finset.card = 3 :=
by
  sorry

end number_of_sixth_powers_lt_1000_l266_266689


namespace inequality_on_f_l266_266116

theorem inequality_on_f {f : ℕ+ → ℝ} 
  (h : ∀ (m n : ℕ+), |f m + f n - f (m + n)| ≤ m * n) :
  ∀ (n : ℕ+), |f n - ∑ k in Finset.range(n) + 1, (f k) / k| ≤ (n * (n - 1)) / 4 :=
by
  sorry

end inequality_on_f_l266_266116


namespace sqrt_fraction_evaluation_l266_266494

theorem sqrt_fraction_evaluation :
  (Real.sqrt ((2 / 25) + (1 / 49) - (1 / 100)) = 3 / 10) :=
by sorry

end sqrt_fraction_evaluation_l266_266494


namespace minimum_m_value_l266_266954

-- Define the conditions and the main theorem to prove
open Set

theorem minimum_m_value (n : ℕ) (A B : Set ℤ) (h₁ : n ≥ 5) (h₂ : |A| = n) (h₃ : |B| = _ )
  (h₄ : A ⊆ B) (h₅ : ∀ {x y : ℤ}, x ≠ y → x ∈ B → y ∈ B → (x + y ∈ B ↔ x ∈ A ∧ y ∈ A)) : 
  m ≥ 3*n - 3 :=

sorry

end minimum_m_value_l266_266954


namespace count_squares_and_cubes_l266_266657

theorem count_squares_and_cubes (bound : ℕ) (hk : bound = 1000) : 
  Nat.card {n : ℕ | n < bound ∧ (∃ k : ℕ, n = k ^ 6)} = 3 := 
by
  sorry

end count_squares_and_cubes_l266_266657


namespace count_sixth_powers_below_1000_l266_266615

theorem count_sixth_powers_below_1000 : 
  {n : ℕ | (n < 1000) ∧ (∃ m : ℕ, n = m ^ 6)}.to_finset.card = 3 :=
by sorry

end count_sixth_powers_below_1000_l266_266615


namespace cos_double_angle_l266_266554

theorem cos_double_angle (θ : ℝ) (h : Real.cos θ = 1 / 3) : Real.cos (2 * θ) = -7 / 9 :=
by
  sorry

end cos_double_angle_l266_266554


namespace count_squares_and_cubes_less_than_1000_l266_266740

theorem count_squares_and_cubes_less_than_1000 :
  {n : ℕ | 0 < n ∧ n < 1000 ∧ ∃ x, n = x ^ 6}.to_finset.card = 3 :=
by sorry

end count_squares_and_cubes_less_than_1000_l266_266740


namespace number_of_sixth_powers_lt_1000_l266_266680

theorem number_of_sixth_powers_lt_1000 : 
  {n : ℕ | ∃ k : ℕ, k^6 = n ∧ n < 1000}.to_finset.card = 3 :=
by
  sorry

end number_of_sixth_powers_lt_1000_l266_266680


namespace convex_hull_perimeter_le_sum_perimeters_l266_266126

theorem convex_hull_perimeter_le_sum_perimeters 
  (polygons : list (set (ℝ × ℝ)))
  (hconvex : ∀ p ∈ polygons, convex p)
  (hno_line : ∀ l : ℝ → ℝ → Prop, ∃ p ∈ polygons, ∃ q ∈ polygons, l p ∧ l q) :
  ∃ (hull : set (ℝ × ℝ)), convex hull ∧
    (is_convex_hull_of polygons hull) ∧
    (perimeter hull ≤ polygons.sum (λ p, perimeter p)) :=
by
  sorry

end convex_hull_perimeter_le_sum_perimeters_l266_266126


namespace max_consecutive_sum_less_500_l266_266356

theorem max_consecutive_sum_less_500 : 
  (argmax (λ n : ℕ, n * (n + 1) / 2 < 500) = 31) :=
by sorry

end max_consecutive_sum_less_500_l266_266356


namespace max_consecutive_sum_l266_266393

theorem max_consecutive_sum : ∃ (n : ℕ), (∀ m : ℕ, (m < n → (m * (m + 1)) / 2 < 500)) ∧ ¬((n * (n + 1)) / 2 < 500) :=
by {
  sorry
}

end max_consecutive_sum_l266_266393


namespace max_tan_angle_D1MD_l266_266338

-- Geometric setting definition
structure Cube (α : Type*) :=
(A B C D A1 B1 C1 D1 : α)

-- Variables and theorem setup
variables {α : Type*} [InnerProductSpace ℝ α]
variables (cube : Cube α) (M : α)
variables (a : ℝ) -- side length of the cube

-- Assume point M is on the line A1C1 of the base A1B1C1D1
def M_on_A1C1 (cube : Cube α) (M : α) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ M = t • (cube.C1 - cube.A1) + cube.A1

-- The main theorem to state
theorem max_tan_angle_D1MD (hM : M_on_A1C1 cube M) :
  ∃ θ : ℝ,
  θ = Real.atan (√2) :=
sorry

end max_tan_angle_D1MD_l266_266338


namespace four_digit_sum_seven_l266_266543

theorem four_digit_sum_seven : 
  (∃ (abcd : ℕ×ℕ×ℕ×ℕ), let (a, b, c, d) := abcd in 
    a ∈ finset.range 1 10 ∧ b ∈ finset.range 0 10 ∧ c ∈ finset.range 0 10 ∧ d ∈ finset.range 0 10 ∧ 
    a + b + c + d = 7) ↔ 
  (finset.card (finset.filter (λ abcd : ℕ×ℕ×ℕ×ℕ, 
    let (a, b, c, d) := abcd in 
      a ∈ finset.range 1 10 ∧ b ∈ finset.range 0 10 ∧ 
      c ∈ finset.range 0 10 ∧ d ∈ finset.range 0 10 ∧ 
      a + b + c + d = 7) 
    (finset.product (finset.range 10) (finset.product (finset.range 10) (finset.product (finset.range 10) (finset.range 10))) )) = 84) :=
sorry

end four_digit_sum_seven_l266_266543


namespace sqrt_expr_evaluation_l266_266084

theorem sqrt_expr_evaluation :
  (Real.sqrt (5 + 4 * Real.sqrt 3) - Real.sqrt (5 - 4 * Real.sqrt 3)) = 2 * Real.sqrt 2 :=
  sorry

end sqrt_expr_evaluation_l266_266084


namespace total_stones_is_60_l266_266997

-- Define the number of stones in each heap
variables (x1 x2 x3 x4 x5 : ℕ)

-- Conditions translated to Lean:
def condition1 := x5 = 6 * x3
def condition2 := x2 = 2 * (x3 + x5)
def condition3 := (x1 = x5 / 3) ∧ (x1 = x4 - 10)
def condition4 := x4 = x2 / 2

-- Theorem to prove the total number of stones in the five piles
theorem total_stones_is_60 (h1 : condition1) (h2 : condition2) (h3 : condition3) (h4 : condition4) :
  x1 + x2 + x3 + x4 + x5 = 60 :=
sorry

end total_stones_is_60_l266_266997


namespace tiles_needed_for_room_l266_266036

def wall_length : ℝ := 8.16 -- Length of the wall in meters
def wall_width : ℝ := 4.32 -- Width of the adjacent wall in meters
def recess_width : ℝ := 1.24 -- Width of the recess in meters
def recess_length : ℝ := 2.0 -- Length of the recess in meters
def protrusion_width : ℝ := 0.48 -- Width of the protrusion in meters
def protrusion_length : ℝ := 0.96 -- Length of the protrusion in meters

noncomputable def total_floor_area : ℝ :=
  (wall_length * wall_width) + (recess_width * recess_length) + (protrusion_width * protrusion_length)

def tile_side : ℝ := 0.48 -- Length of one side of the tile in meters

noncomputable def tile_area : ℝ := tile_side * tile_side -- Area of one tile in square meters

noncomputable def num_tiles (total_area : ℝ) (tile_area : ℝ) : ℝ :=
  (total_area / tile_area).ceil -- Calculate the least number of tiles needed

theorem tiles_needed_for_room : num_tiles total_floor_area tile_area = 166 :=
by
  -- The proof would go here, with the required calculations showing that the number of tiles is indeed 166
  sorry

end tiles_needed_for_room_l266_266036


namespace task_completion_deadline_l266_266480

theorem task_completion_deadline :
  ∃ x : ℝ, (∀ A_duration B_duration : ℝ,
    A_duration = x ∧ B_duration = x + 3 →
    2/A_duration + 2/B_duration + (x - 2)/B_duration = 1) → x = 6 :=
begin
  sorry
end

end task_completion_deadline_l266_266480


namespace Liza_initial_balance_l266_266250

theorem Liza_initial_balance
  (W: Nat)   -- Liza's initial balance on Tuesday
  (rent: Nat := 450)
  (deposit: Nat := 1500)
  (electricity: Nat := 117)
  (internet: Nat := 100)
  (phone: Nat := 70)
  (final_balance: Nat := 1563) 
  (balance_eq: W - rent + deposit - electricity - internet - phone = final_balance) 
  : W = 800 :=
sorry

end Liza_initial_balance_l266_266250


namespace number_of_sixth_powers_less_than_1000_l266_266632

theorem number_of_sixth_powers_less_than_1000 : 
  {n : ℕ | n < 1000 ∧ ∃ (k : ℕ), n = k^6}.finite.card = 3 :=
by sorry

end number_of_sixth_powers_less_than_1000_l266_266632


namespace num_sixth_powers_below_1000_l266_266800

theorem num_sixth_powers_below_1000 : 
  {n : ℕ | n^6 < 1000 ∧ n > 0}.to_finset.card = 3 :=
by sorry

end num_sixth_powers_below_1000_l266_266800


namespace find_larger_number_l266_266424

theorem find_larger_number (L S : ℕ) 
  (h1 : L - S = 1365)
  (h2 : L = 8 * S + 15) : L = 1557 := 
sorry

end find_larger_number_l266_266424


namespace count_squares_and_cubes_less_than_1000_l266_266672

theorem count_squares_and_cubes_less_than_1000 : 
  {n : ℕ // n^6 < 1000}.card = 3 :=
by
  sorry

end count_squares_and_cubes_less_than_1000_l266_266672


namespace find_j_value_l266_266969

def g (x : ℝ) : ℝ := (Real.tan (x / 3)) - (Real.tan x)

theorem find_j_value :
  (∃ j : ℝ, ∀ x : ℝ, g(x) = Real.sin (j * x) / (Real.sin (x / 3) * Real.sin x)) ∧ 
  (j = -2/3) :=
by
  sorry

end find_j_value_l266_266969


namespace union_A_B_comp_U_A_inter_B_range_of_a_l266_266125

namespace ProofProblem

def A : Set ℝ := { x | 2 ≤ x ∧ x ≤ 8 }
def B : Set ℝ := { x | 1 < x ∧ x < 6 }
def C (a : ℝ) : Set ℝ := { x | x > a }
def U : Set ℝ := Set.univ

theorem union_A_B : A ∪ B = { x | 1 < x ∧ x ≤ 8 } := by
  sorry

theorem comp_U_A_inter_B : (U \ A) ∩ B = { x | 1 < x ∧ x < 2 } := by
  sorry

theorem range_of_a (a : ℝ) : (A ∩ C a).Nonempty → a < 8 := by
  sorry

end ProofProblem

end union_A_B_comp_U_A_inter_B_range_of_a_l266_266125


namespace side_length_of_square_l266_266888

open Real

theorem side_length_of_square
  (x : ℝ)
  (h1 : x > 8) -- side length greater than 8 units from RU
  (h2 : x * x = (x - 8) * (x - 8) + (x - 1) * (x - 1)) -- Pythagorean theorem for the point P
  : x = 13 :=
begin
  sorry
end

end side_length_of_square_l266_266888


namespace decompose_polynomial_l266_266320

theorem decompose_polynomial (n : ℕ) :
  (∀ (a : ℕ → ℕ → ℕ), (∀ k, a k n ∈ finset.range (n + 1) ∧ bijective (λ k, a k n)) →
  1976 * (finset.sum (finset.range n) (λ k, k + 1))) ∃ t, (1976 * n = t * n * (n + 1) / 2) →
  t = 2 * 1976 / (n + 1) ∧ ∃ (m : ℕ), n + 1 = 2 ^ m  * (n div 2^m) :=
  n = 7 ∨ n = 103 ∨ n = 1975 :=
begin
  sorry
end

end decompose_polynomial_l266_266320


namespace triangle_isosceles_l266_266131

theorem triangle_isosceles (a b c : ℝ) (h1 : a^2 + b * c = b^2 + a * c) (h2 : a + b > c)
  (h3 : a + c > b) (h4 : b + c > a) : a = b :=
begin
  sorry
end

end triangle_isosceles_l266_266131


namespace problem1_correct_problem2_correct_problem3_correct_problem4_correct_l266_266500

-- Problem (1)
def problem1 : Real :=
  (-23) - (-58) + (-17)

theorem problem1_correct : problem1 = 18 := by
  calc
    problem1 = (-23) - (-58) + (-17) : by rfl
        ... = -23 + 58 - 17        : by norm_num
        ... = -40 + 58            : by norm_num
        ... = 18                  : by norm_num

-- Problem (2)
def problem2 : Real :=
  (-8) / (-1.1) * 0.125

theorem problem2_correct : problem2 = 9 / 10 := by
  calc
    problem2 = (-8) / (-1.1) * 0.125 : by rfl
        ... = 8 / (10/9) * 0.125     : by norm_num
        ... = 8 * 9 / 10 * 0.125     : by norm_num
        ... = 9 / 10                 : by norm_num

-- Problem (3)
def problem3 : Real :=
  ((- 1 / 3) - (1 / 4) + (1 / 15)) * (-60)

theorem problem3_correct : problem3 = 31 := by
  calc
    problem3 = ((- 1 / 3) - (1 / 4) + (1 / 15)) * (-60) : by rfl
        ... = (- 1 / 3) * (-60) - (1 / 4) * (-60) + (1 / 15) * (-60) : by norm_num
        ... = 20 + 15 - 4                                          : by norm_num
        ... = 31                                                   : by norm_num

-- Problem (4)
def problem4 : Real :=
  (-1)^2 * abs(-1/4) + (-1/2)^3 / (-1)^2023

theorem problem4_correct : problem4 = -1 / 8 := by
  calc
    problem4 = (-1)^2 * abs(-1/4) + (-1/2)^3 / (-1)^2023 : by rfl
        ... = 1 * 1 / 4 + (-1/8) * (-1)                  : by norm_num
        ... = 1 / 4 + 1 / 8                              : by norm_num
        ... = -1 / 8                                     : by norm_num

end problem1_correct_problem2_correct_problem3_correct_problem4_correct_l266_266500


namespace number_of_sixth_powers_less_than_1000_l266_266634

theorem number_of_sixth_powers_less_than_1000 : 
  {n : ℕ | n < 1000 ∧ ∃ (k : ℕ), n = k^6}.finite.card = 3 :=
by sorry

end number_of_sixth_powers_less_than_1000_l266_266634


namespace number_of_possible_values_of_a_l266_266991

noncomputable def problem_statement : Prop :=
  ∃ (a b c d : ℕ),
  a > b ∧ b > c ∧ c > d ∧
  a + b + c + d = 2040 ∧
  a^2 - b^2 + c^2 - d^2 = 2040 ∧
  508 ∈ {a | ∃ b c d, a > b ∧ b > c ∧ c > d ∧ a + b + c + d = 2040 ∧ a^2 - b^2 + c^2 - d^2 = 2040}

theorem number_of_possible_values_of_a : problem_statement :=
  sorry

end number_of_possible_values_of_a_l266_266991


namespace log5_square_simplification_l266_266408

noncomputable def log5 (x : ℝ) : ℝ := Real.log x / Real.log 5

theorem log5_square_simplification : (log5 (7 * log5 25))^2 = (log5 14)^2 :=
by
  sorry

end log5_square_simplification_l266_266408


namespace percentage_of_total_spent_on_other_items_l266_266252

-- Definitions for the given problem conditions

def total_amount (a : ℝ) := a
def spent_on_clothing (a : ℝ) := 0.50 * a
def spent_on_food (a : ℝ) := 0.10 * a
def spent_on_other_items (a x_clothing x_food : ℝ) := a - x_clothing - x_food
def tax_on_clothing (x_clothing : ℝ) := 0.04 * x_clothing
def tax_on_food := 0
def tax_on_other_items (x_other_items : ℝ) := 0.08 * x_other_items
def total_tax (a : ℝ) := 0.052 * a

-- The theorem we need to prove
theorem percentage_of_total_spent_on_other_items (a x_clothing x_food x_other_items : ℝ)
    (h1 : x_clothing = spent_on_clothing a)
    (h2 : x_food = spent_on_food a)
    (h3 : x_other_items = spent_on_other_items a x_clothing x_food)
    (h4 : tax_on_clothing x_clothing + tax_on_food + tax_on_other_items x_other_items = total_tax a) :
    0.40 * a = x_other_items :=
sorry

end percentage_of_total_spent_on_other_items_l266_266252


namespace arithmetic_sequence_a10_l266_266885

theorem arithmetic_sequence_a10 (a : ℕ → ℤ) 
  (d : ℤ) (h1 : ∀ n, a (n + 1) = a n + d) 
  (h2 : d = 4) 
  (h3 : (a 3 + 2) / 2 = real.sqrt (2 * (a 1 + a 2 + a 3))) :
  a 10 = 38 := 
  sorry

end arithmetic_sequence_a10_l266_266885


namespace rt_triangle_case1_rt_triangle_case2_rt_triangle_case3_l266_266121

-- Case 1
theorem rt_triangle_case1
  (a : ℝ) (b : ℝ) (c : ℝ) (A B C : ℝ)
  (h : A = 30) (h_bc : B + C = 90) (h_ac : A + C = 90)
  (ha : a = 4) (hb : b = 4 * Real.sqrt 3) (hc : c = 8)
  : b = 4 * Real.sqrt 3 ∧ c = 8 := by
  sorry

-- Case 2
theorem rt_triangle_case2
  (a : ℝ) (b : ℝ) (c : ℝ) (A B C : ℝ)
  (h : B = 60) (h_bc : B + C = 90) (h_ac : A + C = 90)
  (ha : a = Real.sqrt 3 - 1) (hb : b = 3 - Real.sqrt 3) 
  (ha_b: A = 30)
  (h_c: c = 2 * Real.sqrt 3 - 2)
  : B = 60 ∧ A = 30 ∧ c = 2 * Real.sqrt 3 - 2 := by
  sorry

-- Case 3
theorem rt_triangle_case3
  (a : ℝ) (b : ℝ) (c : ℝ) (A B C : ℝ)
  (h : A = 60) (h_bc : B + C = 90) (h_ac : A + C = 90)
  (hc : c = 2 + Real.sqrt 3)
  (ha : a = Real.sqrt 3 + 3/2) 
  (hb: b = (2 + Real.sqrt 3) / 2)
  : a = Real.sqrt 3 + 3/2 ∧ b = (2 + Real.sqrt 3) / 2 := by
  sorry

end rt_triangle_case1_rt_triangle_case2_rt_triangle_case3_l266_266121


namespace number_of_squares_and_cubes_less_than_1000_l266_266714

noncomputable def count_squares_and_cubes : ℕ :=
  (List.range 1000).countp (λ n : ℕ, (∃ k : ℕ, k^6 = n))

theorem number_of_squares_and_cubes_less_than_1000 :
  count_squares_and_cubes = 3 :=
  by
    sorry

end number_of_squares_and_cubes_less_than_1000_l266_266714


namespace total_price_is_correct_l266_266453

-- Define the cost of an adult ticket
def cost_adult : ℕ := 22

-- Define the cost of a children ticket
def cost_child : ℕ := 7

-- Define the number of adults in the family
def num_adults : ℕ := 2

-- Define the number of children in the family
def num_children : ℕ := 2

-- Define the total price the family will pay
def total_price : ℕ := cost_adult * num_adults + cost_child * num_children

-- The proof to check the total price
theorem total_price_is_correct : total_price = 58 :=
by 
  -- Here we would solve the proof
  sorry

end total_price_is_correct_l266_266453


namespace complex_parts_nonzero_l266_266560

noncomputable def complex_expr : ℂ := (1 - I)^(10) + (1 + I)^(10)

theorem complex_parts_nonzero (a b : ℝ) (h : a + b * I = complex_expr) : a ≠ 0 ∧ b ≠ 0 :=
sorry

end complex_parts_nonzero_l266_266560


namespace problem_statement_l266_266860

-- Define the variables
variables (S T Tie : ℝ)

-- Define the given conditions
def condition1 : Prop := 6 * S + 4 * T + 2 * Tie = 80
def condition2 : Prop := 5 * S + 3 * T + 2 * Tie = 110

-- Define the question to be proved
def target : Prop := 4 * S + 2 * T + 2 * Tie = 50

-- Lean theorem statement
theorem problem_statement (h1 : condition1 S T Tie) (h2 : condition2 S T Tie) : target S T Tie :=
  sorry

end problem_statement_l266_266860


namespace david_biology_marks_l266_266076

theorem david_biology_marks
  (english math physics chemistry avg_marks num_subjects : ℕ)
  (h_english : english = 86)
  (h_math : math = 85)
  (h_physics : physics = 82)
  (h_chemistry : chemistry = 87)
  (h_avg_marks : avg_marks = 85)
  (h_num_subjects : num_subjects = 5) :
  ∃ (biology : ℕ), biology = 85 :=
by
  -- Total marks for all subjects
  let total_marks_for_all_subjects := avg_marks * num_subjects
  -- Total marks in English, Mathematics, Physics, and Chemistry
  let total_marks_in_other_subjects := english + math + physics + chemistry
  -- Marks in Biology
  let biology := total_marks_for_all_subjects - total_marks_in_other_subjects
  existsi biology
  sorry

end david_biology_marks_l266_266076


namespace actual_price_of_good_l266_266423

theorem actual_price_of_good (P : ℝ) (h : 0.684 * P = 6600) : P = 9649.12 :=
sorry

end actual_price_of_good_l266_266423


namespace maximum_consecutive_positive_integers_sum_500_l266_266383

theorem maximum_consecutive_positive_integers_sum_500 : 
  ∃ n : ℕ, (n * (n + 1) / 2 < 500) ∧ (∀ m : ℕ, (m * (m + 1) / 2 < 500) → m ≤ n) :=
sorry

end maximum_consecutive_positive_integers_sum_500_l266_266383


namespace marble_probability_difference_l266_266333

theorem marble_probability_difference :
  let r := 1200
  let b := 800
  let total := r + b
  let P_s := (r * (r - 1) / 2 + b * (b - 1) / 2) / (total * (total - 1) / 2)
  let P_d := (r * b) / (total * (total - 1) / 2)
  abs (P_s - P_d) = 7900 / 199900 :=
by
  -- Definitions
  let r := 1200
  let b := 800
  let total := r + b
  let P_s := (r * (r - 1) / 2 + b * (b - 1) / 2) / (total * (total - 1) / 2)
  let P_d := (r * b) / (total * (total - 1) / 2)

  -- To show:
  -- abs (P_s - P_d) = 7900 / 199900
  have h1 : abs (P_s - P_d) = abs ((1039000 / 1999000) - (960000 / 1999000)),
  {
    sorry
  },
  have h2 : 1039000 - 960000 = 79000,
  {
    sorry
  },
  have h3 : abs (79000 / 1999000) = 79000 / 1999000,
  {
    sorry
  },
  have h4 : 79000 / 1999000 = 7900 / 199900,
  {
    sorry
  },
  rw [h1, h2, h3, h4],
  exact nat_abs.symm,
  done

end marble_probability_difference_l266_266333


namespace joyce_apples_l266_266218

/-- Joyce starts with some apples. She gives 52 apples to Larry and ends up with 23 apples. 
    Prove that Joyce initially had 75 apples. -/
theorem joyce_apples (initial_apples given_apples final_apples : ℕ) 
  (h1 : given_apples = 52) 
  (h2 : final_apples = 23) 
  (h3 : initial_apples = given_apples + final_apples) : 
  initial_apples = 75 := 
by 
  sorry

end joyce_apples_l266_266218


namespace number_of_real_roots_of_equation_l266_266318

def f (x : ℝ) : ℝ := (x^2 - x + 1)^3 / (x^2 * (x - 1)^2)

theorem number_of_real_roots_of_equation :
  (∃ (n : ℕ), n = 6 ∧ ∀ x : ℝ, f(x) = f(π) → x = π ∨ x = 1/π ∨ x = 1 - π ∨ x = 1 / (1 - π) ∨ x =  1 - 1 / π ∨ x = π / (π - 1)) := sorry

end number_of_real_roots_of_equation_l266_266318


namespace count_sixth_powers_below_1000_l266_266623

theorem count_sixth_powers_below_1000 : 
  {n : ℕ | (n < 1000) ∧ (∃ m : ℕ, n = m ^ 6)}.to_finset.card = 3 :=
by sorry

end count_sixth_powers_below_1000_l266_266623


namespace count_squares_and_cubes_less_than_1000_l266_266738

theorem count_squares_and_cubes_less_than_1000 :
  {n : ℕ | 0 < n ∧ n < 1000 ∧ ∃ x, n = x ^ 6}.to_finset.card = 3 :=
by sorry

end count_squares_and_cubes_less_than_1000_l266_266738


namespace fixed_monthly_charge_l266_266081

-- Given conditions
variable (F C_J : ℕ)
axiom january_bill : F + C_J = 46
axiom february_bill : F + 2 * C_J = 76

-- Proof problem
theorem fixed_monthly_charge : F = 16 :=
by
  sorry

end fixed_monthly_charge_l266_266081


namespace max_consecutive_sum_leq_500_l266_266380

theorem max_consecutive_sum_leq_500 : ∃ n : ℕ, (∑ i in finset.range (n+1), i) ≤ 500 ∧ (∀ m : ℕ, m > n → (∑ i in finset.range (m+1), i) > 500) :=
sorry

end max_consecutive_sum_leq_500_l266_266380


namespace distance_between_foci_l266_266307

-- Let the hyperbola be defined by the equation xy = 4.
def hyperbola (x y : ℝ) : Prop := x * y = 4

-- Prove that the distance between the foci of this hyperbola is 8.
theorem distance_between_foci : ∀ (x y : ℝ), hyperbola x y → ∃ d, d = 8 :=
by {
    sorry
}

end distance_between_foci_l266_266307


namespace num_triangles_in_grid_l266_266854

theorem num_triangles_in_grid : 
  let count_small := 1 + 2 + 3 + 4 in
  let count_four := 3 + 2 + 1 in
  let count_nine := 1 in
  let count_ten := 1 in
  count_small + count_four + count_nine + count_ten = 18 :=
by
  sorry

end num_triangles_in_grid_l266_266854


namespace equal_x_l266_266177

theorem equal_x (x y : ℝ) (h : x / (x + 2) = (y^2 + 3 * y - 2) / (y^2 + 3 * y + 1)) :
  x = (2 * y^2 + 6 * y - 4) / 3 :=
sorry

end equal_x_l266_266177


namespace equation_of_ellipse_AN_BM_constant_l266_266122

noncomputable def a := 2
noncomputable def b := 1
noncomputable def e := (Real.sqrt 3) / 2
noncomputable def c := Real.sqrt 3

def ellipse (x y : ℝ) : Prop := (x^2 / 4) + y^2 = 1

theorem equation_of_ellipse :
  ellipse a b
:=
by
  sorry

theorem AN_BM_constant (x0 y0 : ℝ) (hx : x0^2 + 4 * y0^2 = 4) :
  let AN := 2 + x0 / (y0 - 1)
  let BM := 1 + 2 * y0 / (x0 - 2)
  abs (AN * BM) = 4
:=
by
  sorry

end equation_of_ellipse_AN_BM_constant_l266_266122


namespace max_profit_l266_266012

def fixed_cost : ℝ := 2.5 * 10^6

def additional_cost (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 80 then (1 / 3) * x^2 + 10 * x
  else if x ≥ 80 then 51 * x + 10^4 / x - 1450
  else 0
  
def selling_price_per_unit : ℝ := 0.05 * 10^6

def revenue (x : ℝ) : ℝ := 50 * x

def profit (x : ℝ) : ℝ :=
  revenue x - additional_cost x - fixed_cost / 10^6  -- divide fixed cost by 10^6 to convert to the same unit (million yuan)

theorem max_profit : ∃ x, x = 100 ∧ profit x = 1000 := by
  sorry

end max_profit_l266_266012


namespace scenario_1_is_linear_scenario_2_is_quadratic_scenario_3_is_inverse_correct_choice_is_B_l266_266294

noncomputable def scenario_1 (x : ℝ) : ℝ := 5 * (10 - x)

noncomputable def scenario_2 (x : ℝ) : ℝ := (30 + x) * (20 + x)

noncomputable def scenario_3 (x : ℝ) : ℝ := 1000 / x

theorem scenario_1_is_linear : ∃ m b, scenario_1 = λ x, m * x + b := 
by {
  use [-5, 50],
  ext,
  simp [scenario_1],
  exact eq_comm.mp rfl
}

theorem scenario_2_is_quadratic : ∃ a b c, scenario_2 = λ x, a * x^2 + b * x + c := 
by {
  use [1, 50, 600],
  ext,
  simp [scenario_2],
  exact eq_comm.mp rfl
}

theorem scenario_3_is_inverse : ∃ k, scenario_3 = λ x, k / x := 
by {
  use 1000,
  ext,
  simp [scenario_3],
  exact eq_comm.mp rfl
}

theorem correct_choice_is_B :
  scenario_1_is_linear ∧ scenario_2_is_quadratic ∧ scenario_3_is_inverse :=
by {
  split,
  { exact scenario_1_is_linear },
  split,
  { exact scenario_2_is_quadratic },
  { exact scenario_3_is_inverse }
}

end scenario_1_is_linear_scenario_2_is_quadratic_scenario_3_is_inverse_correct_choice_is_B_l266_266294


namespace compute_9_times_one_seventh_pow_4_l266_266506

theorem compute_9_times_one_seventh_pow_4 :
  9 * (1 / 7) ^ 4 = 9 / 2401 :=
by
  -- The actual proof would go here.
  sorry

end compute_9_times_one_seventh_pow_4_l266_266506


namespace number_of_squares_and_cubes_less_than_1000_l266_266716

noncomputable def count_squares_and_cubes : ℕ :=
  (List.range 1000).countp (λ n : ℕ, (∃ k : ℕ, k^6 = n))

theorem number_of_squares_and_cubes_less_than_1000 :
  count_squares_and_cubes = 3 :=
  by
    sorry

end number_of_squares_and_cubes_less_than_1000_l266_266716


namespace number_of_sixth_powers_lt_1000_l266_266679

theorem number_of_sixth_powers_lt_1000 : 
  {n : ℕ | ∃ k : ℕ, k^6 = n ∧ n < 1000}.to_finset.card = 3 :=
by
  sorry

end number_of_sixth_powers_lt_1000_l266_266679


namespace find_integral_l266_266287

-- Define the given expansion condition
def expansion_condition (a : ℝ) : Prop :=
  (a^5 * (Real.sqrt 3) = -Real.sqrt 3)

-- Define the integral condition
def integral_condition (a : ℝ) : ℝ :=
  ∫ x in -2..a, x^2

-- Theorem statement
theorem find_integral :
  ∃ a : ℝ, expansion_condition a ∧ integral_condition a = 7 / 3 :=
by
  sorry

end find_integral_l266_266287


namespace zoe_strawberries_count_l266_266419

-- Definitions and conditions extracted from the problem
def strawberries_calories (strawberries : ℕ) : ℕ := strawberries * 4
def yogurt_calories (ounces : ℕ) : ℕ := ounces * 17
def total_calories (strawberries : ℕ) (ounces : ℕ) : ℕ := strawberries_calories(strawberries) + yogurt_calories(ounces)

-- The problem states Zoe ate 150 calories in total, where 6 ounces of yogurt were consumed
theorem zoe_strawberries_count (strawberries : ℕ) (hyp1 : yogurt_calories 6 = 102) (hyp2 : total_calories strawberries 6 = 150) : strawberries = 12 :=
  sorry

end zoe_strawberries_count_l266_266419


namespace S_on_circumcircle_ABC_l266_266223

-- Define a triangle and its circumcircle
variables {A B C D S : Type} 

-- Assume points and properties stated in the problem
axiom angle_bisector (h : ∀ (A B C D S : Type), angle_bisector (∠BAC) [A, D]) -- angle bisector of ∠BAC intersects BC at D
axiom circumcircle (c : ∀ (A B D S : Type), tangent (circumcircle ([A, B, D])) B S) -- tangent at B to the circumcircle of ABD intersects AD at S

-- Define our main statement
theorem S_on_circumcircle_ABC (h1 : ∀ (A B C D S : Type), angle_bisector (∠BAC) [A, D]) 
(h2 : ∀ (A B D S : Type), tangent (circumcircle ([A, B, D])) B S) :
  S ∈ circumcircle ([A, B, C]) :=
sorry

end S_on_circumcircle_ABC_l266_266223


namespace equation_infinite_solutions_l266_266549

theorem equation_infinite_solutions (k : ℝ) : k = -7.5 -> ∀ x : ℝ, 4 * (3 * x - k) = 3 * (4 * x + 10) :=
by
  intro hk
  rw hk
  intro x
  calc
    4 * (3 * x - (-7.5)) = 4 * (3 * x + 7.5)               : by rw sub_neg_eq_add
    ... = 12 * x + 30                                      : by ring
    ... = 3 * (4 * x + 10)                                 : by ring
  sorry

end equation_infinite_solutions_l266_266549


namespace placemat_length_correct_l266_266475

noncomputable def placemat_length (r : ℝ) : ℝ :=
  2 * r * Real.sin (Real.pi / 8)

theorem placemat_length_correct (r : ℝ) (n : ℕ) (w : ℝ) (y : ℝ) (h_r : r = 5)
  (h_n : n = 8) (h_w : w = 1)
  (h_y : y = placemat_length r) :
  y = 5 * Real.sqrt (2 - Real.sqrt 2) :=
by
  sorry

end placemat_length_correct_l266_266475


namespace parallel_lines_l266_266561

open EuclideanGeometry

variables {A B C D E P Q R S P' Q' R' S' : Point}

-- Given a convex quadrilateral with diagonals intersecting perpendicularly
axiom quadrilateral_convex (ABCD : ConvexQuadrilateral A B C D) 
(AC : Line A C) (BD : Line B D) (E : Point) (H_intersect : AC ∩ BD = {E})
(H_perp : IsPerpendicular AC BD)

-- Perpendicular feet from E to the sides
axiom feet (H_perp_EP : IsPerpendicular (Line E P) (Line A B)) 
(H_perp_EQ : IsPerpendicular (Line E Q) (Line B C)) 
(H_perp_ER : IsPerpendicular (Line E R) (Line C D)) 
(H_perp_ES : IsPerpendicular (Line E S) (Line D A)) 

-- The perpendiculars meet the sides 
axiom meeting (H_meet_PP' : Line E P = Line P' D) 
(H_meet_QQ' : Line E Q = Line Q' A) 
(H_meet_RR' : Line E R = Line R' B)
(H_meet_SS' : Line E S = Line S' C)

-- Problem statement
theorem parallel_lines :
  Parallel (Line S' R') (Line Q' P') ∧ Parallel (Line Q' P') (Line A C) ∧
  Parallel (Line Q' R') (Line S' P') ∧ Parallel (Line S' P') (Line B D) :=
sorry

end parallel_lines_l266_266561


namespace calculate_f5_l266_266968

noncomputable def f_n (n : ℕ) (x : ℝ) : ℝ :=
  ∏ i in Finset.range n, Real.cos (x / 2 ^ i)

theorem calculate_f5 :
  f_n 5 (8 * Real.pi / 3) = - Real.sqrt 3 / 32 :=
by
  sorry

end calculate_f5_l266_266968


namespace sequence_inequality_l266_266325

variable {a : Nat → ℝ}

noncomputable def sequence_conditions :=
  ∀ k : ℕ, k ≥ 1 → (a k - 2 * a (k + 1) + a (k + 2) ≥ 0) ∧ (0 ≤ a k) ∧ (∑ i in Finset.range k, a i ≤ 1)

theorem sequence_inequality (h : sequence_conditions a) (k : ℕ) (hk : k ≥ 1) : 0 ≤ (a k - a (k + 1)) ∧ (a k - a (k + 1)) < 2 / (k^2) :=
sorry

end sequence_inequality_l266_266325


namespace only_abundant_number_l266_266411

-- Define what it means for a number to be abundant
def is_abundant (n : ℕ) : Prop :=
  (∑ d in (Nat.divisors n).erase n, d) > n

-- List of given options
def options : List ℕ := [8, 10, 14, 18, 22]

-- Prove that 18 is the only abundant number among the given options
theorem only_abundant_number : ∃! n ∈ options, is_abundant n :=
  sorry

end only_abundant_number_l266_266411


namespace find_k_l266_266962

noncomputable def f (k a : ℝ) : Polynomial ℝ :=
  Polynomial.monicCubic (x - (k + 2)) (x - (k + 6)) (x - a)

noncomputable def g (k b : ℝ) : Polynomial ℝ :=
  Polynomial.monicCubic (x - (k + 4)) (x - (k + 8)) (x - b)

theorem find_k (k : ℝ) (a b : ℝ)
    (h_f : f k a - g k b = Polynomial.C (x + k)) : k = 7 :=
sorry

end find_k_l266_266962


namespace angle_measure_proof_l266_266892

noncomputable def angle_RPQ_measure (x : ℝ) : ℝ :=
  128.57

theorem angle_measure_proof (x : ℝ) (PQ PR RS QP SQR : Point) 
  (hP_on_RS : P ∈ [RS]) 
  (hQP_bisects_SQR : angle_bisector QP (∠(S, Q, R)))
  (hPQ_eq_PR : PQ = PR) 
  (angle_RSQ : angle (R ↔ S ↔ Q).measure = 4 * x) 
  (angle_RPQ : angle (R ↔ P ↔ Q).measure = 5 * x) :
  angle (R ↔ P ↔ Q).measure = angle_RPQ_measure x :=
by
  -- Proof omitted intentionally
  sorry

end angle_measure_proof_l266_266892


namespace proof_problem_l266_266029

noncomputable def problem_conditions : Prop :=
  ∃ (M : ℝ × ℝ) (O : ℝ → ℝ → ℝ) (l : ℝ → ℝ → ℝ) (E : ℝ → ℝ → ℝ) (p : ℝ) (k₁ k₂ m : ℝ),
    M = (sqrt 3, -1) ∧
    O = λ x y, x^2 + y^2 - 4 ∧
    ∀ (tangent_line : ℝ → ℝ → ℝ), 
      tangent_line = λ x y, y + 1 - sqrt 3 * (x - sqrt 3) ∧
      l = λ x y, x + sqrt 3 * y - 3 * sqrt 3 ∧ 
      ∀ F : ℝ × ℝ, 
        F = (0, 3) ∧ 
        p > 0 ∧ 
        E = λ x y, x^2 - 2*p*y ∧ 
        2 * p = 6 ∧
        ∀ A B C D : ℝ × ℝ,
          (A.1 = 12*k₁ ∧ B.1 * A.1 = -24) ∧
          (C.1 = 12 * (- A.1) ∧ D.1 = 12 * (- B.1)) ∧
          A.1 * C.1 = A.2 * D.2 = -12 ∧
          (0, 1) ∈ AC ∧ 
          (0, 1) ∈ BD ∧
          AC = (λ x, x) ∧ BD = (λ x, x) ∧

theorem proof_problem :
  problem_conditions →
  (∃ (l : ℝ → ℝ → ℝ) (E : ℝ → ℝ → ℝ) (k₁ k₂ : ℝ), 
    l = λ x y, x + sqrt 3 * y - 3 * sqrt 3 ∧
    E = λ x y, x^2 - 12 * y ∧
    k₁ / k₂ = 2) :=
by
  sorry

end proof_problem_l266_266029


namespace number_of_squares_and_cubes_less_than_1000_l266_266717

noncomputable def count_squares_and_cubes : ℕ :=
  (List.range 1000).countp (λ n : ℕ, (∃ k : ℕ, k^6 = n))

theorem number_of_squares_and_cubes_less_than_1000 :
  count_squares_and_cubes = 3 :=
  by
    sorry

end number_of_squares_and_cubes_less_than_1000_l266_266717


namespace slower_train_speed_l266_266348

/-- Two trains of equal length 25 meters each are running on parallel lines in the same direction.
    The faster train is running at 46 km/hr and passes the slower train in 18 seconds.
    This theorem finds the speed of the slower train. -/
theorem slower_train_speed
  (speed_faster : ℝ := 46)
  (length_train : ℝ := 25)
  (time_to_pass : ℝ := 18)
  (relative_speed_conversion : ℝ := 5 / 18)
  (total_distance : ℝ := 50) :
  ∃ (v : ℝ), 
    let relative_speed := (speed_faster - v) * relative_speed_conversion in
    relative_speed = total_distance / time_to_pass ∧ v = 36 := 
by
  let v := 36
  let relative_speed := (speed_faster - v) * relative_speed_conversion
  have h1: relative_speed = total_distance / time_to_pass :=
    calc 
      relative_speed 
        = (46 - 36) * (5 / 18) : by sorry
    ... = 10 * (5 / 18) : by sorry
    ... = 50 / 18 : by sorry
  exact ⟨v, h1, rfl⟩

end slower_train_speed_l266_266348


namespace problem_l266_266966

def f (n : ℕ) (x : ℝ) : ℝ :=
  (List.range n).map (λ k, Real.cos (x / 2^k)).prod

theorem problem (x : ℝ) (h : x = 8 * Real.pi / 3) : f 5 x = -Real.sqrt 3 / 32 :=
  by
  simp [f, h]
  sorry

end problem_l266_266966


namespace g_1986_l266_266863

def g : ℕ → ℤ := sorry

axiom g_def : ∀ n : ℕ, g n ≥ 0
axiom g_one : g 1 = 3
axiom g_func_eq : ∀ (a b : ℕ), g (a + b) = g a + g b - 3 * g (a * b)

theorem g_1986 : g 1986 = 0 :=
by
  sorry

end g_1986_l266_266863


namespace count_sixth_powers_below_1000_l266_266828

-- Definition and necessary condition for perfect sixth power
def is_sixth_power (n : ℕ) : Prop := ∃ k : ℕ, n = k ^ 6

-- Main statement of the problem
theorem count_sixth_powers_below_1000 :
  {n : ℕ | n < 1000 ∧ is_sixth_power n}.to_finset.card = 3 := 
sorry

end count_sixth_powers_below_1000_l266_266828


namespace final_value_of_x_l266_266186

noncomputable def initial_x : ℝ := 52 * 1.2
noncomputable def decreased_x : ℝ := initial_x * 0.9
noncomputable def final_x : ℝ := decreased_x * 1.15

theorem final_value_of_x : final_x = 64.584 := by
  sorry

end final_value_of_x_l266_266186


namespace number_of_sixth_powers_lt_1000_l266_266683

theorem number_of_sixth_powers_lt_1000 : 
  {n : ℕ | ∃ k : ℕ, k^6 = n ∧ n < 1000}.to_finset.card = 3 :=
by
  sorry

end number_of_sixth_powers_lt_1000_l266_266683


namespace count_squares_and_cubes_less_than_1000_l266_266668

theorem count_squares_and_cubes_less_than_1000 : 
  {n : ℕ // n^6 < 1000}.card = 3 :=
by
  sorry

end count_squares_and_cubes_less_than_1000_l266_266668


namespace cube_edge_length_l266_266017

/-- The dimensions of the base of the vessel are 20 cm * 14 cm, and the rise in water level is 
12.053571428571429 cm. Prove that the edge length of the cube that causes this rise is 15 cm. -/
theorem cube_edge_length (length width h_rise s : ℝ) (h_length : length = 20) (h_width : width = 14)
  (h_h_rise : h_rise = 12.053571428571429) (h_s_cube : s^3 = length * width * h_rise) :
  s = 15 :=
by {
  sorry,
}

end cube_edge_length_l266_266017


namespace student_percentage_l266_266039

-- Define the percentages for each subject and the overall percentage
variables (P : ℕ) -- Percentage in the first subject
variables (second_percentage : ℕ) -- Percentage in the second subject
variables (third_percentage : ℕ) -- Percentage in the third subject
variables (overall_percentage : ℕ) -- Overall percentage

-- Conditions: given percentages for second and third subjects, and the overall percentage
axiom second_subject_percentage : second_percentage = 70
axiom third_subject_percentage : third_percentage = 90
axiom overall_percentage_value : overall_percentage = 70

-- Lean statement to prove P = 50 given the conditions
theorem student_percentage (h : (P + second_percentage + third_percentage) / 3 = overall_percentage) : P = 50 :=
by
  simp [second_subject_percentage, third_subject_percentage, overall_percentage_value] at h
  sorry

end student_percentage_l266_266039


namespace negation_of_proposition_l266_266159

theorem negation_of_proposition :
  (∀ x : ℝ, 2^x + x^2 > 0) → (∃ x0 : ℝ, 2^x0 + x0^2 ≤ 0) :=
sorry

end negation_of_proposition_l266_266159


namespace max_difference_negative_one_l266_266001

-- Define the type for vertex labels
def vertex_labels := Fin 100 → ℝ

-- Define the condition that sums the squares to 1
def sum_of_squares_condition (x : vertex_labels) : Prop :=
  (Finset.univ.sum (λ i, (x i) ^ 2) = 1)

-- Define the sum of products for red and blue segments
def sum_of_products_red (x : vertex_labels) : ℝ :=
  Finset.univ.sum (λ ⟨i, _⟩, Finset.univ.sum (λ ⟨j, _⟩,
    if ((j - i).natAbs % 2 = 0) then x i * x j else 0))

def sum_of_products_blue (x : vertex_labels) : ℝ :=
  Finset.univ.sum (λ ⟨i, _⟩, Finset.univ.sum (λ ⟨j, _⟩,
    if ((j - i).natAbs % 2 = 1) then x i * x j else 0))

-- Define the maximum difference R - B
def max_difference (x : vertex_labels) : ℝ :=
  sum_of_products_red x - sum_of_products_blue x

-- State the theorem
theorem max_difference_negative_one (x : vertex_labels) (h : sum_of_squares_condition x) : 
  max_difference x = -1 :=
  sorry

end max_difference_negative_one_l266_266001


namespace count_sixth_powers_below_1000_l266_266619

theorem count_sixth_powers_below_1000 : 
  {n : ℕ | (n < 1000) ∧ (∃ m : ℕ, n = m ^ 6)}.to_finset.card = 3 :=
by sorry

end count_sixth_powers_below_1000_l266_266619


namespace number_of_sixth_powers_lt_1000_l266_266686

theorem number_of_sixth_powers_lt_1000 : 
  {n : ℕ | ∃ k : ℕ, k^6 = n ∧ n < 1000}.to_finset.card = 3 :=
by
  sorry

end number_of_sixth_powers_lt_1000_l266_266686


namespace triangle_area_ratio_l266_266427

structure Point := (x : ℝ) (y : ℝ)

def A := Point.mk 2 0
def B := Point.mk 8 12
def C := Point.mk 14 0

def X := Point.mk 6 0
def Y := Point.mk 8 4
def Z := Point.mk 10 0

def base (P Q : Point) : ℝ :=
  abs (P.x - Q.x)

def height (P : Point) : ℝ :=
  P.y

def area (P Q R : Point) : ℝ :=
  0.5 * (base P R) * (height Q)

theorem triangle_area_ratio : 
  (area X Y Z) / (area A B C) = 1 / 9 :=
by
  sorry

end triangle_area_ratio_l266_266427


namespace number_of_sixth_powers_less_than_1000_l266_266843

theorem number_of_sixth_powers_less_than_1000 : 
  {n : ℕ | ∃ m : ℕ, n = m^6 ∧ n < 1000}.to_finset.card = 3 := 
by {
  sorry
}

end number_of_sixth_powers_less_than_1000_l266_266843


namespace div_three_S_2n_l266_266431

-- Define the number of distinct paths S(n) under the given constraints
def S : ℕ → ℕ
| 0     := 1
| 1     := 2
| (n+2) := 3 * S (n + 1) + (∑ k in range (n + 1), S k * S (n - k))

theorem div_three_S_2n (n : ℕ) (hn : n > 0) : 3 ∣ S (2 * n) :=
sorry

end div_three_S_2n_l266_266431


namespace number_of_sixth_powers_less_than_1000_l266_266642

theorem number_of_sixth_powers_less_than_1000 : 
  {n : ℕ | n < 1000 ∧ ∃ (k : ℕ), n = k^6}.finite.card = 3 :=
by sorry

end number_of_sixth_powers_less_than_1000_l266_266642


namespace probability_triangle_ABC_isosceles_right_triangle_probability_l266_266515

open Real

noncomputable def area {A B C P : ℝ × ℝ} : ℝ :=
  let h := (dist(P, line B C) : ℝ)
  1 / 2 * 4 * h

theorem probability_triangle_ABC (A B C P : ℝ × ℝ)
 (hABC: area (0, 0) (4, 0) (0, 4) = 8 )
 (hPBC: area P B C < 1 / 3 * area (0, 0) (4, 0) (0, 4))
 (hA_ne_B : (0, 0) ≠ (4, 0)) (hA_ne_C : (0, 0) ≠ (0, 4)) 
 (hB_ne_C : (4, 0) ≠ (0, 4)) : ℝ :=
  1 / 3

-- Variables and definitions for triangle vertices
variable (A : ℝ × ℝ := (0, 0))
variable (B : ℝ × ℝ := (4, 0))
variable (C : ℝ × ℝ := (0, 4))

-- Main theorem statement
theorem isosceles_right_triangle_probability :
  ∀ P : ℝ × ℝ, (0 ≤ P.fst ) → (P.fst ≤ 4 → (0 ≤ P.snd) → (P.snd ≤ 4)) → 
  dist(P, line B C) < 4 / 3 → 2 * (dist(P, line B C)) < 8 / 3 :=
begin
  sorry
end

end probability_triangle_ABC_isosceles_right_triangle_probability_l266_266515


namespace xiaoming_speed_correct_l266_266418

-- Defining the conditions
def circular_track_length : ℕ := 600
def time_between_encounters : ℕ := 50
def xiaohong_speed (x : ℕ) : ℕ := x
def xiaoming_speed (x : ℕ) : ℕ := x + 2

-- The combined speed when they will meet running in opposite directions
def combined_speed (x : ℕ) : ℕ := xiaohong_speed x + xiaoming_speed x

-- The proof statement
theorem xiaoming_speed_correct (x : ℕ) (h: combined_speed x * time_between_encounters = circular_track_length) :
  xiaoming_speed x = 7 :=
begin
  -- define the hypothesis
  have h1 : combined_speed x = 12, from sorry, -- here the solver would show solving 2x + 2 = 12
  -- substitute into xiaoming_speed definition
  subst h1,
  -- show x + 2 = 7
  sorry
end

end xiaoming_speed_correct_l266_266418


namespace arithmetic_sequence_sum_10_terms_l266_266195

variable {α : Type*}

def arithmetic_sequence_sum (a₁ a₁₀ : α) [Add α] [Mul α] [Div α] [Coe α ℝ] : Prop :=
  a₁ + a₁₀ = (12 : α)

noncomputable def sum_of_first_n_terms (n : ℕ) (a₁ a₁₀ : α) [Add α] [Mul α] [Div α] [Coe α ℝ] : α :=
  (n / 2) * (a₁ + a₁₀)

theorem arithmetic_sequence_sum_10_terms :
  ∀ (a₁ a₁₀ : ℕ),
  arithmetic_sequence_sum a₁ a₁₀ →
  sum_of_first_n_terms 10 a₁ a₁₀ = 60 :=
by
  intros a₁ a₁₀ h
  sorry

end arithmetic_sequence_sum_10_terms_l266_266195


namespace solve_xy_l266_266434

variable (x y : ℝ)

-- Given conditions
def condition1 : Prop := y = (2 / 3) * x
def condition2 : Prop := 0.4 * x = (1 / 3) * y + 110

-- Statement we want to prove
theorem solve_xy (h1 : condition1 x y) (h2 : condition2 x y) : x = 618.75 ∧ y = 412.5 :=
  by sorry

end solve_xy_l266_266434


namespace both_square_and_cube_count_both_square_cube_less_than_1000_l266_266774

theorem both_square_and_cube (n : ℕ) (h_positive : n > 0) (h_bound : n < 1000) (h_square : ∃ k : ℕ, n = k^2) (h_cube : ∃ k : ℕ, n = k^3) :
  n ∈ {1, 64, 729} :=
sorry

theorem count_both_square_cube_less_than_1000 :
  {n : ℕ | n > 0 ∧ n < 1000 ∧ ∃ k : ℕ, n = k^2 ∧ ∃ k : ℕ, n = k^3}.to_finset.card = 3 :=
by
  have h_valid : ∀ n ∈ {1, 64, 729}, n > 0 ∧ n < 1000 ∧ (∃ k : ℕ, n = k^2) ∧ (∃ k : ℕ, n = k^3) := sorry
  have h_reverse : ∀ n, n > 0 ∧ n < 1000 ∧ (∃ k : ℕ, n = k^2) ∧ (∃ k : ℕ, n = k^3) → n ∈ {1, 64, 729} := sorry
  exact Finset.card_map _ {1, 64, 729}.to_finset
  sorry

end both_square_and_cube_count_both_square_cube_less_than_1000_l266_266774


namespace number_of_sixth_powers_less_than_1000_l266_266639

theorem number_of_sixth_powers_less_than_1000 : 
  {n : ℕ | n < 1000 ∧ ∃ (k : ℕ), n = k^6}.finite.card = 3 :=
by sorry

end number_of_sixth_powers_less_than_1000_l266_266639


namespace problem1_problem2_l266_266593

section
variables {x m : ℝ}

def f (x : ℝ) : ℝ := |x - 3| - 2
def g (x : ℝ) : ℝ := -|x + 1| + 4

-- Problem 1: Prove the range of 'x' given that f(x) ≤ 1.
theorem problem1 (x : ℝ) : f(x) ≤ 1 ↔ 0 ≤ x ∧ x ≤ 6 := sorry

-- Problem 2: Prove the range of 'm' given the solution to f(x) - g(x) ≥ m + 1 is ℝ.
theorem problem2 (∀ x : ℝ, f(x) - g(x) ≥ m + 1) : m ≤ -3 := sorry

end

end problem1_problem2_l266_266593


namespace max_gcd_consecutive_terms_l266_266526

-- Define the sequence b_n = n! + 2n
def b (n : ℕ) : ℕ := nat.factorial n + 2 * n

-- Define the gcd of two consecutive terms of the sequence
def gcd_consecutive_terms (n : ℕ) : ℕ := nat.gcd (b n) (b (n + 1))

-- The theorem stating the maximum possible value of the gcd of two consecutive terms
theorem max_gcd_consecutive_terms : ∃ n, gcd_consecutive_terms n = 3 :=
begin
  -- We claim that the maximum gcd value is 3 for some n
  sorry
end

end max_gcd_consecutive_terms_l266_266526


namespace total_cost_is_58_l266_266452

-- Define the conditions
def cost_per_adult : Nat := 22
def cost_per_child : Nat := 7
def number_of_adults : Nat := 2
def number_of_children : Nat := 2

-- Define the theorem to prove the total cost
theorem total_cost_is_58 : number_of_adults * cost_per_adult + number_of_children * cost_per_child = 58 :=
by
  -- Steps of proof will go here
  sorry

end total_cost_is_58_l266_266452


namespace number_of_sixth_powers_less_than_1000_l266_266635

theorem number_of_sixth_powers_less_than_1000 : 
  {n : ℕ | n < 1000 ∧ ∃ (k : ℕ), n = k^6}.finite.card = 3 :=
by sorry

end number_of_sixth_powers_less_than_1000_l266_266635


namespace max_consecutive_sum_leq_500_l266_266378

theorem max_consecutive_sum_leq_500 : ∃ n : ℕ, (∑ i in finset.range (n+1), i) ≤ 500 ∧ (∀ m : ℕ, m > n → (∑ i in finset.range (m+1), i) > 500) :=
sorry

end max_consecutive_sum_leq_500_l266_266378


namespace tangent_line_to_curve_ln_x_l266_266409

theorem tangent_line_to_curve_ln_x (a : ℝ) :
  (∃ x₀ : ℝ, y = x + a ∧ y = ln x ∧ (∂/(∂x) ln x = 1/x₀)) → a = -1 :=
by
  sorry

end tangent_line_to_curve_ln_x_l266_266409


namespace number_of_sixth_powers_less_than_1000_l266_266848

theorem number_of_sixth_powers_less_than_1000 : 
  {n : ℕ | ∃ m : ℕ, n = m^6 ∧ n < 1000}.to_finset.card = 3 := 
by {
  sorry
}

end number_of_sixth_powers_less_than_1000_l266_266848


namespace sum_of_consecutive_integers_l266_266366

theorem sum_of_consecutive_integers (n : ℕ) (h : n * (n + 1) / 2 ≤ 500) :
  n = 31 ∨ ((32 * (32 + 1) / 2) ≤ 500) := 
sorry

end sum_of_consecutive_integers_l266_266366


namespace count_sixth_powers_less_than_1000_l266_266752

theorem count_sixth_powers_less_than_1000 : 
  {n : ℕ | 0 < n ∧ n < 1000 ∧ (∃ k : ℕ, n = k^6)}.to_finset.card = 3 :=
by 
  sorry

end count_sixth_powers_less_than_1000_l266_266752


namespace count_sixth_powers_less_than_1000_l266_266751

theorem count_sixth_powers_less_than_1000 : 
  {n : ℕ | 0 < n ∧ n < 1000 ∧ (∃ k : ℕ, n = k^6)}.to_finset.card = 3 :=
by 
  sorry

end count_sixth_powers_less_than_1000_l266_266751


namespace last_number_crossed_out_l266_266248

theorem last_number_crossed_out (n : ℕ) (circ : List ℕ) (h1 : circ = List.range (n+1)) 
  (h2 : n = 2016) 
  (h3 : ∀ (k : ℕ), k > 0 ∧ k ≤ n → circ[(k mod n)] = k + 1) :
  circ.last = 2015 := 
sorry

end last_number_crossed_out_l266_266248


namespace result_of_expression_l266_266493

def calculate_expression : ℝ :=
  15.380 * (3.15 + 0.014 + 0.458) / 4.25

theorem result_of_expression :
  Float.round (calculate_expression * 1000) / 1000 = 13.112 :=
by
  sorry

end result_of_expression_l266_266493


namespace count_positive_integers_square_and_cube_lt_1000_l266_266768

theorem count_positive_integers_square_and_cube_lt_1000 : 
  {n : ℕ | n < 1000 ∧ ∃ k : ℕ, n = k^6}.to_finset.card = 3 :=
by
  sorry

end count_positive_integers_square_and_cube_lt_1000_l266_266768


namespace max_consecutive_sum_l266_266394

theorem max_consecutive_sum : ∃ (n : ℕ), (∀ m : ℕ, (m < n → (m * (m + 1)) / 2 < 500)) ∧ ¬((n * (n + 1)) / 2 < 500) :=
by {
  sorry
}

end max_consecutive_sum_l266_266394


namespace rowing_speed_in_still_water_l266_266023

noncomputable def speedInStillWater (distance_m : ℝ) (time_s : ℝ) (speed_current : ℝ) : ℝ :=
  let distance_km := distance_m / 1000
  let time_h := time_s / 3600
  let speed_downstream := distance_km / time_h
  speed_downstream - speed_current

theorem rowing_speed_in_still_water :
  speedInStillWater 45.5 9.099272058235341 8.5 = 9.5 :=
by
  sorry

end rowing_speed_in_still_water_l266_266023


namespace count_squares_and_cubes_l266_266648

theorem count_squares_and_cubes (bound : ℕ) (hk : bound = 1000) : 
  Nat.card {n : ℕ | n < bound ∧ (∃ k : ℕ, n = k ^ 6)} = 3 := 
by
  sorry

end count_squares_and_cubes_l266_266648


namespace count_sixth_powers_below_1000_l266_266621

theorem count_sixth_powers_below_1000 : 
  {n : ℕ | (n < 1000) ∧ (∃ m : ℕ, n = m ^ 6)}.to_finset.card = 3 :=
by sorry

end count_sixth_powers_below_1000_l266_266621


namespace vector_parallel_l266_266606

theorem vector_parallel (m : ℝ) 
  (h1 : (2, 1) = (2 : ℝ, 1 : ℝ)) 
  (h2 : ∃ k : ℝ, (m, 2) = k • (2, 1)) : 
  3 • (2, 1 : ℝ × ℝ) + 2 • (m, 2) = (14, 7) := 
by 
  sorry

end vector_parallel_l266_266606


namespace trapezium_area_calculation_l266_266536

structure Trapezium where
  a : ℝ -- Length of one parallel side
  b : ℝ -- Length of the other parallel side
  h : ℝ -- Distance between the parallel sides

def TrapeziumArea (tz : Trapezium) : ℝ :=
  (1 / 2) * (tz.a + tz.b) * tz.h

theorem trapezium_area_calculation (a b h : ℝ) (h_a : a = 20) (h_b : b = 18) (h_h : h = 25) :
  TrapeziumArea ⟨a, b, h⟩ = 475 :=
by
  sorry

end trapezium_area_calculation_l266_266536


namespace max_consecutive_sum_less_500_l266_266360

theorem max_consecutive_sum_less_500 : 
  (argmax (λ n : ℕ, n * (n + 1) / 2 < 500) = 31) :=
by sorry

end max_consecutive_sum_less_500_l266_266360


namespace number_of_squares_and_cubes_less_than_1000_l266_266709

noncomputable def count_squares_and_cubes : ℕ :=
  (List.range 1000).countp (λ n : ℕ, (∃ k : ℕ, k^6 = n))

theorem number_of_squares_and_cubes_less_than_1000 :
  count_squares_and_cubes = 3 :=
  by
    sorry

end number_of_squares_and_cubes_less_than_1000_l266_266709


namespace fruitseller_apples_l266_266459

theorem fruitseller_apples (x : ℝ) (sold_percent remaining_apples : ℝ) 
  (h_sold : sold_percent = 0.80) 
  (h_remaining : remaining_apples = 500) 
  (h_equation : (1 - sold_percent) * x = remaining_apples) : 
  x = 2500 := 
by 
  sorry

end fruitseller_apples_l266_266459


namespace count_squares_cubes_less_than_1000_l266_266819

theorem count_squares_cubes_less_than_1000 : 
  {n : ℕ | ∃ x : ℕ, n = x^6 ∧ n < 1000}.card = 3 :=
sorry

end count_squares_cubes_less_than_1000_l266_266819


namespace number_of_sixth_powers_lt_1000_l266_266694

-- Define the condition for being a sixth power and being less than 1000
def isSixthPower (n : ℕ) : Prop := ∃ k : ℕ, n = k^6

-- Define the main statement to be proved
theorem number_of_sixth_powers_lt_1000 : 
  {n : ℕ | n < 1000 ∧ isSixthPower n}.to_finset.card = 3 :=
by
  sorry

end number_of_sixth_powers_lt_1000_l266_266694


namespace intersection_of_lines_l266_266090

theorem intersection_of_lines :
  let x := 5 / 3
  let y := 7 / 3
  10 * x - 5 * y = 5 ∧ 8 * x + 2 * y = 18 :=
by
  let x := 5 / 3
  let y := 7 / 3
  split
  · calc
      10 * x - 5 * y = 10 * (5 / 3) - 5 * (7 / 3) : by rfl
       ... = (50 / 3) - (35 / 3) : by norm_num
       ... = 15 / 3 : by norm_num
       ... = 5 : by norm_num
  · calc
      8 * x + 2 * y = 8 * (5 / 3) + 2 * (7 / 3) : by rfl
       ... = (40 / 3) + (14 / 3) : by norm_num
       ... = 54 / 3 : by norm_num
       ... = 18 : by norm_num

end intersection_of_lines_l266_266090


namespace number_of_squares_and_cubes_less_than_1000_l266_266718

noncomputable def count_squares_and_cubes : ℕ :=
  (List.range 1000).countp (λ n : ℕ, (∃ k : ℕ, k^6 = n))

theorem number_of_squares_and_cubes_less_than_1000 :
  count_squares_and_cubes = 3 :=
  by
    sorry

end number_of_squares_and_cubes_less_than_1000_l266_266718


namespace max_X_in_grid_l266_266878

def is_valid_placement (grid : List (List Bool)) : Prop :=
  ∀ i j, (i ≤ 2 ∨ ¬(grid.get! i j ∧ grid.get! (i+1) j ∧ grid.get! (i+2) j)) ∧
         (j ≤ 2 ∨ ¬(grid.get! i j ∧ grid.get! i (j+1) ∧ grid.get! i (j+2))) ∧
         (i ≤ 2 ∨ j ≤ 2 ∨ ¬(grid.get! i j ∧ grid.get! (i+1) (j+1) ∧ grid.get! (i+2) (j+2))) ∧
         (i ≤ 2 ∨ j ≥ 2 ∨ ¬(grid.get! i j ∧ grid.get! (i+1) (j-1) ∧ grid.get! (i+2) (j-2)))

theorem max_X_in_grid : ∃ grid : List (List Bool), 
  length grid = 5 ∧ 
  (∀ row, length row = 5) ∧ 
  is_valid_placement grid ∧ 
  (grid.foldl (λ total row, total + row.foldl (λ acc cell, if cell then acc + 1 else acc) 0) 0 = 14) :=
sorry

end max_X_in_grid_l266_266878


namespace product_of_numbers_l266_266343

variable {x y : ℝ}

theorem product_of_numbers (h1 : x - y = 1 * k) (h2 : x + y = 8 * k) (h3 : x * y = 40 * k) : 
  x * y = 6400 / 63 := by
  sorry

end product_of_numbers_l266_266343


namespace actual_distance_l266_266251

theorem actual_distance (scale : ℕ) (map_distance : ℕ) 
 (h_scale : scale = 1000000) (h_map_distance : map_distance = 12) : 
  (map_distance * scale) / 100000 = 120 :=
by 
  rw [h_scale, h_map_distance]
  sorry

end actual_distance_l266_266251


namespace arithmetic_sequence_sum_l266_266583

theorem arithmetic_sequence_sum (a : ℕ → ℕ) (S : ℕ → ℕ) (d : ℤ) (n : ℕ)
  (h₁ : ∀ n : ℕ, a n = a 1 + (n - 1) * d)
  (h₂ : ∀ n : ℕ, S n = n * (a 1 + a n) / 2)
  (h₃ : 3 * a 5 - a 1 = 10) :
  S 13 = 117 := 
sorry

end arithmetic_sequence_sum_l266_266583


namespace prob_of_target_hit_l266_266162

noncomputable def probability_target_hit : ℚ :=
  let pA := (1 : ℚ) / 2
  let pB := (1 : ℚ) / 3
  let pC := (1 : ℚ) / 4
  let pA' := 1 - pA
  let pB' := 1 - pB
  let pC' := 1 - pC
  let pNoneHit := pA' * pB' * pC'
  1 - pNoneHit

-- Statement to be proved
theorem prob_of_target_hit : probability_target_hit = 3 / 4 :=
  sorry

end prob_of_target_hit_l266_266162


namespace sine_transformation_correct_l266_266312

def transform_sine_function (x : ℝ) : ℝ :=
  sin (2 * x - 2 * (π / 3))

theorem sine_transformation_correct :
  ∀ x, transform_sine_function (x - π / 3) = sin (2 * x - 2 * (π / 3)) :=
sorry

end sine_transformation_correct_l266_266312


namespace coefficient_x2_y6_in_expansion_l266_266927

theorem coefficient_x2_y6_in_expansion :
  ∀ (x y : ℝ), x ≠ 0 → 
    (let c8_6 := (nat.factorial 8) / ((nat.factorial 6) * (nat.factorial 2)),
         c8_5 := (nat.factorial 8) / ((nat.factorial 5) * (nat.factorial 3)) in
     c8_6 - c8_5 = -28) :=
by
  intros x y hx_ne_zero
  let c8_6 := (nat.factorial 8) / ((nat.factorial 6) * (nat.factorial 2))
  let c8_5 := (nat.factorial 8) / ((nat.factorial 5) * (nat.factorial 3))
  have h : c8_6 - c8_5 = -28 := sorry
  exact h

end coefficient_x2_y6_in_expansion_l266_266927


namespace number_of_sixth_powers_less_than_1000_l266_266629

theorem number_of_sixth_powers_less_than_1000 : 
  {n : ℕ | n < 1000 ∧ ∃ (k : ℕ), n = k^6}.finite.card = 3 :=
by sorry

end number_of_sixth_powers_less_than_1000_l266_266629


namespace number_of_squares_and_cubes_less_than_1000_l266_266724

noncomputable def count_squares_and_cubes : ℕ :=
  (List.range 1000).countp (λ n : ℕ, (∃ k : ℕ, k^6 = n))

theorem number_of_squares_and_cubes_less_than_1000 :
  count_squares_and_cubes = 3 :=
  by
    sorry

end number_of_squares_and_cubes_less_than_1000_l266_266724


namespace John_gets_new_EpiPen_every_6_months_l266_266214

variable (EpiPen_cost : ℝ)
variable (insurance_coverage : ℝ)
variable (yearly_payment : ℝ)

def months_to_new_EpiPen (EpiPen_cost : ℝ) (insurance_coverage : ℝ) (yearly_payment : ℝ) : ℝ :=
  let johns_share := EpiPen_cost * (1 - insurance_coverage)
  let EpiPens_per_year := yearly_payment / johns_share
  12 / EpiPens_per_year

theorem John_gets_new_EpiPen_every_6_months :
  months_to_new_EpiPen 500 0.75 250 = 6 :=
by
  sorry

end John_gets_new_EpiPen_every_6_months_l266_266214


namespace age_problem_l266_266877

-- Defining the conditions and the proof problem
variables (B A : ℕ) -- B and A are natural numbers

-- Given conditions
def B_age : ℕ := 38
def A_age (B : ℕ) : ℕ := B + 8
def age_in_10_years (A : ℕ) : ℕ := A + 10
def years_ago (B : ℕ) (X : ℕ) : ℕ := B - X

-- Lean statement of the problem
theorem age_problem (X : ℕ) (hB : B = B_age) (hA : A = A_age B):
  age_in_10_years A = 2 * (years_ago B X) → X = 10 :=
by
  sorry

end age_problem_l266_266877


namespace coeff_x2y6_in_expansion_l266_266907

theorem coeff_x2y6_in_expansion : 
  (coeff_of_term (x := 2) (y := 6)) ((1 - (y / x)) * (x + y) ^ 8) = -28 :=
sorry

end coeff_x2y6_in_expansion_l266_266907


namespace no_real_solution_of_fraction_eq_l266_266869

theorem no_real_solution_of_fraction_eq (m : ℝ) :
  (∀ x : ℝ, (x - 1) / (x + 4) ≠ m / (x + 4)) → m = -5 :=
sorry

end no_real_solution_of_fraction_eq_l266_266869


namespace sum_zero_implies_product_terms_nonpositive_l266_266941

theorem sum_zero_implies_product_terms_nonpositive (a b c : ℝ) (h : a + b + c = 0) : 
  a * b + a * c + b * c ≤ 0 := 
by 
  sorry

end sum_zero_implies_product_terms_nonpositive_l266_266941


namespace joshua_crates_l266_266217

def joshua_packs (b : ℕ) (not_packed : ℕ) (b_per_crate : ℕ) : ℕ :=
  (b - not_packed) / b_per_crate

theorem joshua_crates : joshua_packs 130 10 12 = 10 := by
  sorry

end joshua_crates_l266_266217


namespace greatest_integer_b_l266_266352

theorem greatest_integer_b (b : ℤ) :
  (∀ x : ℝ, x^2 + (b : ℝ) * x + 5 ≠ -25) → b ≤ 10 :=
by
  intro
  sorry

end greatest_integer_b_l266_266352


namespace g_of_5_l266_266862

noncomputable def g (x : ℝ) : ℝ := -2 / x

theorem g_of_5 (x : ℝ) : g (g (g (g (g x)))) = -2 / x :=
by
  sorry

end g_of_5_l266_266862


namespace average_difference_l266_266484

def student_count : ℕ := 200
def professor_count : ℕ := 4
def class_sizes : List ℕ := [100, 50, 30, 20]

noncomputable def average_for_professor : ℕ :=
(class_sizes.sum * 1) / professor_count

noncomputable def average_for_student : ℝ :=
∑ c in class_sizes, (c * (c / student_count))

theorem average_difference :
  average_for_professor - average_for_student = -19 := by
  sorry

end average_difference_l266_266484


namespace cornbread_pieces_l266_266947

theorem cornbread_pieces :
  let pan_length := 20
  let pan_width := 18
  let piece_length := 2
  let piece_width := 2
  let pan_area := pan_length * pan_width
  let piece_area := piece_length * piece_width
  let num_pieces := pan_area / piece_area
  num_pieces = 90 :=
by
  let pan_length := 20
  let pan_width := 18
  let piece_length := 2
  let piece_width := 2
  let pan_area := pan_length * pan_width
  let piece_area := piece_length * piece_width
  let num_pieces := pan_area / piece_area
  show num_pieces = 90
  from sorry

end cornbread_pieces_l266_266947


namespace both_square_and_cube_count_both_square_cube_less_than_1000_l266_266787

theorem both_square_and_cube (n : ℕ) (h_positive : n > 0) (h_bound : n < 1000) (h_square : ∃ k : ℕ, n = k^2) (h_cube : ∃ k : ℕ, n = k^3) :
  n ∈ {1, 64, 729} :=
sorry

theorem count_both_square_cube_less_than_1000 :
  {n : ℕ | n > 0 ∧ n < 1000 ∧ ∃ k : ℕ, n = k^2 ∧ ∃ k : ℕ, n = k^3}.to_finset.card = 3 :=
by
  have h_valid : ∀ n ∈ {1, 64, 729}, n > 0 ∧ n < 1000 ∧ (∃ k : ℕ, n = k^2) ∧ (∃ k : ℕ, n = k^3) := sorry
  have h_reverse : ∀ n, n > 0 ∧ n < 1000 ∧ (∃ k : ℕ, n = k^2) ∧ (∃ k : ℕ, n = k^3) → n ∈ {1, 64, 729} := sorry
  exact Finset.card_map _ {1, 64, 729}.to_finset
  sorry

end both_square_and_cube_count_both_square_cube_less_than_1000_l266_266787


namespace coefficient_x2_y6_l266_266911

theorem coefficient_x2_y6 : 
  let x y : ℚ
  in let expr := (1 - y / x) * (x + y)^8
  in polynomial.coeff (polynomial.expand x^2 y^6 expr) = -28 := 
by
  sorry

end coefficient_x2_y6_l266_266911


namespace sqrt_four_l266_266497

theorem sqrt_four : (∃ x : ℝ, x ^ 2 = 4) → (sqrt 4 = 2) :=
by
  intro h
  sorry

end sqrt_four_l266_266497


namespace number_of_sixth_powers_less_than_1000_l266_266630

theorem number_of_sixth_powers_less_than_1000 : 
  {n : ℕ | n < 1000 ∧ ∃ (k : ℕ), n = k^6}.finite.card = 3 :=
by sorry

end number_of_sixth_powers_less_than_1000_l266_266630


namespace positive_difference_values_l266_266236

def g (n : ℝ) : ℝ :=
  if n < 0 then n^2 + 4 * n + 3
  else 3 * n - 15

theorem positive_difference_values (b1 b2 : ℝ)
  (h1 : b1 < 0) (h2 : ¬ (b2 < 0))
  (eq1 : g -3 + g 3 + g b1 = 10)
  (eq2 : g -3 + g 3 + g b2 = 10) :
  |b1 - b2| = 19.26 := sorry

end positive_difference_values_l266_266236


namespace correct_propositions_are_2_and_3_l266_266046

-- Definitions and problem statement
def proposition_1 : Prop :=
  ∃ q : ℚ, ∃ i : ℤ, ∀ x : ℤ, x ∈ ℚ → i ∈ ℚ → i ≠ q

def proposition_2 : Prop :=
  let R2_1 := 0.976;
  let R2_2 := 0.776;
  let R2_3 := 0.076;
  let R2_4 := 0.351;
  R2_1 > R2_2 ∧ R2_1 > R2_3 ∧ R2_1 > R2_4

def proposition_3 : Prop :=
  ∀ a b c : ℝ, a < 0 ∧ b < 0 ∧ c < 0 →
  a + 1 / b > -2 ∨ b + 1 / c > -2 ∨ c + 1 / a > -2

def proposition_4 : Prop :=
  ∃ x : ℝ, (x = 5 ∨ x = real.sqrt 7) ∧
            (6^2 + 8^2 = 10^2) ∧
            (6 / 3 = 8 / 4 = 10 / x)

noncomputable def correct_propositions : list ℕ :=
  [2, 3]

-- Theorem to prove
theorem correct_propositions_are_2_and_3:
  (proposition_2 ∧ proposition_3) ∧
  ¬proposition_1 ∧
  ¬proposition_4 :=
by
  split;
  sorry -- Proofs of propositions 2 and 3
  split;
  sorry -- Proofs of ¬proposition_1 and ¬proposition_4

end correct_propositions_are_2_and_3_l266_266046


namespace seven_pointed_star_angle_l266_266057

theorem seven_pointed_star_angle (ϕ : ℝ) (hϕ : ϕ = (π / 7)) :
  ∃ a : ℝ, a = (5 * π / 7) :=
by
  use (5 * π / 7)
  sorry

end seven_pointed_star_angle_l266_266057


namespace initial_average_is_100_l266_266281

-- Definitions based on the conditions from step a)
def students : ℕ := 10
def wrong_mark : ℕ := 90
def correct_mark : ℕ := 10
def correct_average : ℝ := 92

-- Initial average marks before correcting the error
def initial_average_marks (A : ℝ) : Prop :=
  10 * A = (students * correct_average) + (wrong_mark - correct_mark)

theorem initial_average_is_100 :
  ∃ A : ℝ, initial_average_marks A ∧ A = 100 :=
by {
  -- We are defining the placeholder for the actual proof.
  sorry
}

end initial_average_is_100_l266_266281


namespace count_squares_and_cubes_less_than_1000_l266_266674

theorem count_squares_and_cubes_less_than_1000 : 
  {n : ℕ // n^6 < 1000}.card = 3 :=
by
  sorry

end count_squares_and_cubes_less_than_1000_l266_266674


namespace expansion_terms_count_eq_45_l266_266068

theorem expansion_terms_count_eq_45 : 
  (∑ i j k : ℕ, if i + j + k = 8 then 1 else 0) = 45 :=
begin
  sorry
end

end expansion_terms_count_eq_45_l266_266068


namespace compute_100p_plus_q_l266_266234

theorem compute_100p_plus_q
  (p q : ℝ)
  (h1 : ∀ x : ℝ, (x + p) * (x + q) * (x + 15) = 0 → 
                  x ≠ -4 → x ≠ -15 → x ≠ -p → x ≠ -q)
  (h2 : ∀ x : ℝ, (x + 2 * p) * (x + 4) * (x + 9) = 0 → 
                  x ≠ -q → x ≠ -15 → (x = -4 ∨ x = -9))
  : 100 * p + q = -191 := 
sorry

end compute_100p_plus_q_l266_266234


namespace find_y_l266_266891

theorem find_y (PQ_parallel_RS : ∀ P Q R S : Point, parallel PQ RS)
  (PRT_straight : ∀ P R T : Point, collinear P R T)
  (angle_QPR : angle Q P R = 85)
  (angle_PRS : angle P R S = 125)
  (angle_PRC : angle P R C = 110)
  (angle_PQS : angle P Q S = y) : 
  y = 30 := by
  sorry

end find_y_l266_266891


namespace coefficient_x2_y6_l266_266913

theorem coefficient_x2_y6 : 
  let x y : ℚ
  in let expr := (1 - y / x) * (x + y)^8
  in polynomial.coeff (polynomial.expand x^2 y^6 expr) = -28 := 
by
  sorry

end coefficient_x2_y6_l266_266913


namespace probability_x_plus_y_le_6_is_correct_l266_266027

noncomputable def probability_x_plus_y_le_6 : ℚ :=
  let rect_area := 4 * 5 in
  let trapezoid_area := (1 / 2) * (5 + 2) * 4 in
  trapezoid_area / rect_area

theorem probability_x_plus_y_le_6_is_correct :
  probability_x_plus_y_le_6 = 7 / 10 :=
by
  sorry

end probability_x_plus_y_le_6_is_correct_l266_266027


namespace hyperbola_eq_and_mn_length_l266_266289

theorem hyperbola_eq_and_mn_length
  (a b : ℝ) 
  (h_a_pos : a > 0) 
  (h_b_pos : b > 0) 
  (asymptote_eq : ∀ x, y = (sqrt 3 / 3) * x) 
  (chord_length : ℝ := 6) 
  (h_chord_length : ∀ a b, (2 * b^2) / a = 6)
  : (y^2 - (x^2 / 3) = 1) ∧ 
    (M N : ℝ) (line_eq : ∀ x, y = x - 2) (h_line_intersects : ∀ x y, 2x^2 - 12x + 9 = 0) → 
    (|MN| = 6) := 
by 
  sorry

end hyperbola_eq_and_mn_length_l266_266289


namespace probability_green_cube_l266_266443

/-- A box contains 36 pink, 18 blue, 9 green, 6 red, and 3 purple cubes that are identical in size.
    Prove that the probability that a randomly selected cube is green is 1/8. -/
theorem probability_green_cube :
  let pink_cubes := 36
  let blue_cubes := 18
  let green_cubes := 9
  let red_cubes := 6
  let purple_cubes := 3
  let total_cubes := pink_cubes + blue_cubes + green_cubes + red_cubes + purple_cubes
  let probability := (green_cubes : ℚ) / total_cubes
  probability = 1 / 8 := 
by
  sorry

end probability_green_cube_l266_266443


namespace segment_divides_triangle_into_two_equal_parts_if_passes_through_circumcenter_l266_266551

variables {A B C D E F O : Type} [EuclideanGeometry A B C D E F O]

-- Definition of the triangle and the noted points.
variables (A B C : Point) (D : Foot A B C) (E : Foot D A) (F : Foot D C) (O : Circumcenter A B C)
variables (EF : Line E F)

-- Condition that the line EF passes through the circumcenter O
variables (EF_passes_O : O ∈ EF)

-- The main statement to be proven
theorem segment_divides_triangle_into_two_equal_parts_if_passes_through_circumcenter :
  divides_area_equally A B C EF :=
sorry

end segment_divides_triangle_into_two_equal_parts_if_passes_through_circumcenter_l266_266551


namespace max_consecutive_integers_sum_l266_266398

theorem max_consecutive_integers_sum (n : ℕ) : 
  (∃ n, (n * (n + 1) / 2 < 500) ∧ (∀ m > n, m * (m + 1) / 2 ≥ 500)) :=
by {
  use 31,
  split,
  {
    norm_num,
  },
  {
    intros m hm,
    have h : m = 32 := by linarith,
    rw h,
    norm_num,
  },
  sorry
}

end max_consecutive_integers_sum_l266_266398


namespace sequence_properties_l266_266137

noncomputable def aₙ (n : ℕ) : ℝ :=
  if n = 1 then 1/3 else -(3 * Sn n * Sn (n-1))

noncomputable def Sn (n : ℕ) : ℝ :=
  if n = 1 then aₙ 1 else (λ k, ∑ i in range k, aₙ (i+1)) n

theorem sequence_properties {n : ℕ} (hne : n ≥ 2) :
  (aₙ n + 3 * Sn n * Sn (n - 1) = 0)
  ∧ (∀ n ≥ 1, Sn n = 1 / (3 * n))
  ∧ (∀ n ≥ 2, (∑ k in range n, (1 / Sn k) - (1 / Sn (k-1))) = 3 * positive integer sum)
  ∧ (∀ k, ∃ r, Sn (3 ^ k) / Sn (3 ^ (k-1)) = r) :=
sorry

end sequence_properties_l266_266137


namespace wrapping_paper_area_correct_l266_266512

-- Define the length, width, and height of the box
variables (l w h : ℝ)

-- Define the function to calculate the area of the wrapping paper
def wrapping_paper_area (l w h : ℝ) : ℝ := 2 * (l + w + h) ^ 2

-- Statement problem that we need to prove
theorem wrapping_paper_area_correct :
  wrapping_paper_area l w h = 2 * (l + w + h) ^ 2 := 
sorry

end wrapping_paper_area_correct_l266_266512


namespace count_sixth_powers_below_1000_l266_266626

theorem count_sixth_powers_below_1000 : 
  {n : ℕ | (n < 1000) ∧ (∃ m : ℕ, n = m ^ 6)}.to_finset.card = 3 :=
by sorry

end count_sixth_powers_below_1000_l266_266626


namespace sum_of_squares_of_sines_60_degree_arithmetic_l266_266609

theorem sum_of_squares_of_sines_60_degree_arithmetic (α : ℝ) : 
  (sin α)^2 + (sin (α + 60 * (π / 180)))^2 + (sin (α + 120 * (π / 180)))^2 = 3 / 2 :=
by sorry

end sum_of_squares_of_sines_60_degree_arithmetic_l266_266609


namespace min_scarves_for_correct_circle_reduce_scarves_from_more_than_12_l266_266295

noncomputable def isCorrect (n : ℕ) (h : fin n → bool) : Prop :=
  ∀ i, ¬ h i → (h ((i + 1) % n) ∨ h ((i - 1 + n) % n))

theorem min_scarves_for_correct_circle : ∃ h : fin 25 → bool, (∑ i, if h i then 1 else 0) = 9 ∧ isCorrect 25 h :=
  sorry

theorem reduce_scarves_from_more_than_12 : ∀ h : fin 25 → bool, (∑ i, if h i then 1 else 0) > 12 → ∃ h' : fin 25 → bool, (∑ i, if h' i then 1 else 0) ≤ 12 ∧ isCorrect 25 h' :=
  sorry

end min_scarves_for_correct_circle_reduce_scarves_from_more_than_12_l266_266295


namespace allie_and_betty_product_l266_266488

def g (n : ℕ) : ℕ :=
  if n % 2 = 0 ∧ n % 3 = 0 then 6
  else if n % 3 = 0 then 3
  else if n % 2 = 0 then 2
  else 0

def Allie_rolls : List ℕ := [6, 3, 2, 4]
def Betty_rolls : List ℕ := [5, 2, 3, 6]

def total_points (rolls : List ℕ) : ℕ :=
  List.foldr (λ roll acc, g roll + acc) 0 rolls

def allie_points : ℕ := total_points Allie_rolls
def betty_points : ℕ := total_points Betty_rolls

theorem allie_and_betty_product :
  allie_points * betty_points = 143 := by
  sorry

end allie_and_betty_product_l266_266488


namespace number_of_non_possible_d_l266_266319

-- Define constants and conditions from the problem
constant k : ℕ := 12
constant a : ℕ := 4020

-- Conditions
def perimeter_condition (t s : ℕ) : Prop := 3 * t = 4 * s + a
def length_condition (t s d : ℕ) : Prop := t = abs (s - k) + d
def positive_perimeter (s : ℕ) : Prop := s > 0

-- Theorem statement
theorem number_of_non_possible_d : ∀ (d : ℕ), 
  (∀ t s, positive_perimeter s → perimeter_condition t s → length_condition t s d → d > 1352) → 
  (1 ≤ d ∧ d ≤ 1352) → 
  false := 
sorry

end number_of_non_possible_d_l266_266319


namespace quadrilateral_EFGH_area_l266_266053

theorem quadrilateral_EFGH_area (ABCD : square) (area_ABCD : area ABCD = 1)
  (M : point) (M_midpoint : midpoint M (side CD ABCD))
  (E F : point) (BE_EF_FC : BE = EF ∧ EF = FC)
  (H G : point) (intersect_AE_DF_BM : intersects H (line AE (point A E)) (line BM (point B M)) ∧
                intersects G (line DF (point D F)) (line BM (point B M))) :
  area (quadrilateral E F G H) = 23 / 210 := 
sorry

end quadrilateral_EFGH_area_l266_266053


namespace num_sixth_powers_below_1000_l266_266798

theorem num_sixth_powers_below_1000 : 
  {n : ℕ | n^6 < 1000 ∧ n > 0}.to_finset.card = 3 :=
by sorry

end num_sixth_powers_below_1000_l266_266798


namespace minimize_cosine_sum_l266_266605

variables {A B P Q M : Point}
variables {a b : ℝ}
variables [Real.number ℝ]
variables (PQ_lines : ∃ (P Q : Point), line P Q)

theorem minimize_cosine_sum
  (hPQ : line P Q)
  (hab_pos : a > 0 ∧ b > 0) :
  (∃ (M : Point) (hM : M ∈ PQ_lines),
    ∀ (X : Point),
      (X ∈ PQ_lines →
        b * dist A M + a * dist B M ≤ b * dist A X + a * dist B X) ↔
        ∀θ (θ1 : θ = angle A M P) (θ2 : θ' = angle B M Q),
        cos θ / cos θ' = a / b) := sorry

end minimize_cosine_sum_l266_266605


namespace algorithm_output_is_10_l266_266517

variable (I : Type) -- Declaring variable I
variable (S : ℕ) -- Declaring variable S

theorem algorithm_output_is_10 :  
  let S := 1 in
  let S := S + 1 in
  let S := S + 3 in
  let S := S + 5 in
  S = 10 :=
by
  -- Proof goes here
  sorry

end algorithm_output_is_10_l266_266517


namespace perpendicular_condition_l266_266117

variable {α : Type*} [linear_ordered_field α] 

-- Definitions of the problem context
variables (l a : set α) (α : set (set α))

-- Conditions
def line_a_subset_alpha := a ⊆ α
def line_l_perpendicular_alpha := ∀ b ∈ α, l ∩ b = ∅
def line_l_perpendicular_a := l ∩ a = ∅

-- Theorem statement
theorem perpendicular_condition (h₁ : line_a_subset_alpha a α)
  (h₂ : line_l_perpendicular_alpha l α) :
  line_l_perpendicular_a l a ∧ ¬ (line_l_perpendicular_a l a → line_l_perpendicular_alpha l α) :=
sorry

end perpendicular_condition_l266_266117


namespace hyperbola_equation_l266_266562

theorem hyperbola_equation
  (a b : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : ∃ x y, (x - 2)^2 + y^2 = 4 ∧ (x + 2)^2 + y^2 = 4)
  (h_asymptote : ∀ x y, y = sqrt 3 * x ∨ y = -sqrt 3 * x) :
  ∃ a b, x^2 - (y^2 / 3) = 1 :=
by {
  sorry
}

end hyperbola_equation_l266_266562


namespace total_price_is_correct_l266_266454

-- Define the cost of an adult ticket
def cost_adult : ℕ := 22

-- Define the cost of a children ticket
def cost_child : ℕ := 7

-- Define the number of adults in the family
def num_adults : ℕ := 2

-- Define the number of children in the family
def num_children : ℕ := 2

-- Define the total price the family will pay
def total_price : ℕ := cost_adult * num_adults + cost_child * num_children

-- The proof to check the total price
theorem total_price_is_correct : total_price = 58 :=
by 
  -- Here we would solve the proof
  sorry

end total_price_is_correct_l266_266454


namespace sin_alpha_equals_neg_half_l266_266584

theorem sin_alpha_equals_neg_half 
  (α : Real) 
  (h : ∃ α : Real, ∀ (x y : Real), (x = - (Real.sqrt 3) / 2) ∧ (y = -1 / 2) ∧ (cos α, sin α) = (x, y)
  ) : sin α = -1 / 2 :=
sorry

end sin_alpha_equals_neg_half_l266_266584


namespace count_positive_integers_square_and_cube_lt_1000_l266_266761

theorem count_positive_integers_square_and_cube_lt_1000 : 
  {n : ℕ | n < 1000 ∧ ∃ k : ℕ, n = k^6}.to_finset.card = 3 :=
by
  sorry

end count_positive_integers_square_and_cube_lt_1000_l266_266761


namespace number_of_correct_propositions_l266_266140

theorem number_of_correct_propositions :
  (∀ a b : ℝ, a < b → ∫ x in b..a, (1:ℝ) = b - a) → 
  (∫ x in (-1:ℝ)..0, x^2 = ∫ x in (0:ℝ)..1, x^2) → 
  (2 * ∫ x in (0:ℝ)..Real.pi, sin x = 4) → 
  (1 = 1) :=
by
  sorry

end number_of_correct_propositions_l266_266140


namespace count_sixth_powers_below_1000_l266_266624

theorem count_sixth_powers_below_1000 : 
  {n : ℕ | (n < 1000) ∧ (∃ m : ℕ, n = m ^ 6)}.to_finset.card = 3 :=
by sorry

end count_sixth_powers_below_1000_l266_266624


namespace count_sixth_powers_below_1000_l266_266821

-- Definition and necessary condition for perfect sixth power
def is_sixth_power (n : ℕ) : Prop := ∃ k : ℕ, n = k ^ 6

-- Main statement of the problem
theorem count_sixth_powers_below_1000 :
  {n : ℕ | n < 1000 ∧ is_sixth_power n}.to_finset.card = 3 := 
sorry

end count_sixth_powers_below_1000_l266_266821


namespace largest_of_six_consecutive_sum_2070_is_347_l266_266328

theorem largest_of_six_consecutive_sum_2070_is_347 (n : ℕ) :
  n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) = 2070 → n + 5 = 347 :=
by
  intro h
  sorry

end largest_of_six_consecutive_sum_2070_is_347_l266_266328


namespace number_of_solutions_l266_266168

theorem number_of_solutions (x : ℕ) : (8 < -2 * x + 17) → (x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4) :=
by sorry

end number_of_solutions_l266_266168


namespace parallel_lines_assumption_line2_does_not_pass_first_quadrant_l266_266157

section
variables (a b : ℝ)

def line1 (x y: ℝ) : Prop := a*x - 3*y + 1 = 0
def line2 (x y: ℝ) : Prop := x - b*y + 2 = 0

-- Prove line1 is parallel to line2 implies ab = 3
theorem parallel_lines_assumption (hx : ∃ (x y : ℝ), line1 x y) (hy : ∃ (x y : ℝ), line2 x y): 
  (∀ x1 y1 x2 y2, line1 x1 y1 ∧ line2 x2 y2 → a/3 = 1/b) → a * b = 3 :=
begin
  sorry
end

-- Prove when b < 0, line2 does not pass through the first quadrant
theorem line2_does_not_pass_first_quadrant {x y : ℝ} (hb : b < 0) 
  (hx : line2 x y) : x ≤ 0 ∨ y ≤ 0 := 
begin
  sorry
end

end

end parallel_lines_assumption_line2_does_not_pass_first_quadrant_l266_266157


namespace probability_x_plus_y_le_6_is_correct_l266_266028

noncomputable def probability_x_plus_y_le_6 : ℚ :=
  let rect_area := 4 * 5 in
  let trapezoid_area := (1 / 2) * (5 + 2) * 4 in
  trapezoid_area / rect_area

theorem probability_x_plus_y_le_6_is_correct :
  probability_x_plus_y_le_6 = 7 / 10 :=
by
  sorry

end probability_x_plus_y_le_6_is_correct_l266_266028


namespace DJ_eq_DL_l266_266951

variable (A B C D E F O J K L : Type)
variable [MetricSpace A] [MetricSpace B] [MetricSpace C]
variable [MetricSpace D] [MetricSpace E] [MetricSpace F]
variable [MetricSpace O] [MetricSpace J] [MetricSpace K] [MetricSpace L]

-- Convex hexagon ABCDEF tangent to a circle with center O
variable (hex_tangent : convex_hexagon_tangent_to_circle A B C D E F O)

-- Circumcircle of triangle ACE is concentric with circle ω
variable (concentric : circumcircle_concentric_with_circle A C E O)

-- Definitions of J, K, and L
variable (J_foot : foot_of_perpendicular B C D J)
variable (K_intersect : perpendicular_intersection B D F E O K)
variable (L_foot : foot_of_perpendicular K D E L)

-- Theorem DJ = DL
theorem DJ_eq_DL : distance D J = distance D L := sorry

end DJ_eq_DL_l266_266951


namespace max_consecutive_sum_less_500_l266_266359

theorem max_consecutive_sum_less_500 : 
  (argmax (λ n : ℕ, n * (n + 1) / 2 < 500) = 31) :=
by sorry

end max_consecutive_sum_less_500_l266_266359


namespace count_sixth_powers_less_than_1000_l266_266746

theorem count_sixth_powers_less_than_1000 : 
  {n : ℕ | 0 < n ∧ n < 1000 ∧ (∃ k : ℕ, n = k^6)}.to_finset.card = 3 :=
by 
  sorry

end count_sixth_powers_less_than_1000_l266_266746


namespace count_squares_and_cubes_l266_266652

theorem count_squares_and_cubes (bound : ℕ) (hk : bound = 1000) : 
  Nat.card {n : ℕ | n < bound ∧ (∃ k : ℕ, n = k ^ 6)} = 3 := 
by
  sorry

end count_squares_and_cubes_l266_266652


namespace problem_1_problem_2_l266_266587

-- Definition of the function f(x)
def f (x : ℝ) := sqrt 3 * sin (x + π / 4)

-- Problem 1: Proving intervals where f(x) is monotonically increasing
theorem problem_1 :
  ∀ k : ℤ, 
    ∀ x : ℝ, 
      x ∈ set.Icc (-(3 * π / 4) + 2 * k * π) (π / 4 + 2 * k * π) → 
      deriv f x ≥ 0 :=
sorry

-- Problem 2: Proving maximum value of given trigonometric expression
theorem problem_2 :
  ∀ (A B C : ℝ),
    A + B + C = π → 
    f(B) = sqrt 3 → 
    ∃ x : ℝ, 
       sqrt 2 * cos A + cos C = 1 :=
sorry

end problem_1_problem_2_l266_266587


namespace solution_set_of_inequality_l266_266582

variable (f : ℝ → ℝ)

-- Define the conditions
def isEvenFunction : Prop := ∀ x : ℝ, f x = f (-x)
def derivativeCondition : Prop := ∀ x : ℝ, (x > 0) → f' x > 2 * x
def functionValueAt2 : Prop := f 2 = 4

-- Define the inequality
def inequalityHolds : ℝ → Prop :=
  λ x, x * f (x - 1) + 2 * x^2 > x^3 + x

-- Define the solution set
def solutionSet (x : ℝ) : Prop :=
  (x > -1 ∧ x < 0) ∨ (x > 3)

-- The statement to prove
theorem solution_set_of_inequality :
  isEvenFunction f →
  derivativeCondition f →
  functionValueAt2 f →
  ∀ x : ℝ, inequalityHolds f x ↔ solutionSet x :=
by sorry

end solution_set_of_inequality_l266_266582


namespace possible_sampling_interval_l266_266329

-- Definitions based on conditions
def total_population_size : ℕ := 102
def omitted_individuals : ℕ := 2
def sample_size : ℕ := total_population_size - omitted_individuals

-- The proof problem
theorem possible_sampling_interval : ∃ (n : ℕ), n = 10 ∧ (sample_size % n = 0) :=
by {
  have h1 : sample_size = 100 := by simp [total_population_size, omitted_individuals],
  existsi 10,
  split,
  exact rfl,
  rw h1,
  norm_num,
  sorry
}

end possible_sampling_interval_l266_266329


namespace count_sixth_powers_less_than_1000_l266_266753

theorem count_sixth_powers_less_than_1000 : 
  {n : ℕ | 0 < n ∧ n < 1000 ∧ (∃ k : ℕ, n = k^6)}.to_finset.card = 3 :=
by 
  sorry

end count_sixth_powers_less_than_1000_l266_266753


namespace angle_between_KO_QR_is_90_degrees_l266_266990

-- Definitions for the points and their properties
variables (A B C O Q R P S K : Type)
variables [IsTriangle ABC] -- Representing triangle condition
variable [IsCenter O ABC]   -- O is the center of circumcircle
variable [LiesOn Q A B]     -- Q lies on segment AB
variable [LiesOn R B C]     -- R lies on segment BC
variable [Intersects QR (Circumcircle ABR) P] -- QR intersects circumcircle ABR at P
variable [Intersects QR (Circumcircle BCQ) S] -- QR intersects circumcircle BCQ at S
variable [Intersection AP CS K]               -- AP and CS intersect at K

-- Equivalent proof problem
theorem angle_between_KO_QR_is_90_degrees :
  angle (line KO) (line QR) = 90 := 
sorry

end angle_between_KO_QR_is_90_degrees_l266_266990


namespace range_of_a_l266_266963

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f' : ℝ → ℝ := sorry

axiom differentiable_f (x : ℝ) : has_deriv_at f (f' x) x
axiom condition_f' (x : ℝ) : f' x < x
axiom condition_f_a (a : ℝ) : f (1 - a) - f a ≤ 1 / 2 - a

theorem range_of_a (a : ℝ) : a ≤ 1 / 2 :=
begin
  sorry
end

end range_of_a_l266_266963


namespace coeff_x2y6_in_expansion_l266_266905

theorem coeff_x2y6_in_expansion : 
  (coeff_of_term (x := 2) (y := 6)) ((1 - (y / x)) * (x + y) ^ 8) = -28 :=
sorry

end coeff_x2y6_in_expansion_l266_266905


namespace count_valid_colorings_l266_266194

def point := (ℕ × ℕ × ℕ)
def E : set point := { p | ∀ i ∈ {0, 1, 2}, p.1 ≤ 1982 ∧ p.2 ≤ 1982 ∧ p.3 ≤ 1982 }

def is_coloring_valid (coloring : point → Prop) : Prop :=
  ∀ p1 p2 p3 p4, (p1.1 = p2.1 ∧ p3.1 = p4.1 ∧ p1.2 = p3.2 ∧ p2.2 = p4.2 ∧ p1.3 = p1.3 ∧ p3.3 = p4.3) →
  ((if coloring p1 then 0 else 1) + (if coloring p2 then 0 else 1) + 
   (if coloring p3 then 0 else 1) + (if coloring p4 then 0 else 1)) % 4 = 0

theorem count_valid_colorings :
  ∃ n : ℕ, n = 2^5947 ∧ ∀ coloring, is_coloring_valid coloring → n := sorry

end count_valid_colorings_l266_266194


namespace max_consecutive_integers_sum_lt_500_l266_266373

theorem max_consecutive_integers_sum_lt_500 
  (n : ℕ)
  (h₁ : ∑ k in Finset.range (n+1), k = n * (n + 1) / 2)
  (h₂ : n * (n + 1) / 2 < 500) 
  (h₃ : (n + 1) * (n + 2) / 2 > 500)
  : n = 31 := 
by sorry

end max_consecutive_integers_sum_lt_500_l266_266373


namespace man_speed_still_water_l266_266471

theorem man_speed_still_water (v_m v_s : ℝ) :
  (v_m + v_s = 14) ∧ (v_m - v_s = 6) → v_m = 10 :=
by
  intro h
  cases h with h1 h2
  sorry

end man_speed_still_water_l266_266471


namespace distance_midpoint_KN_to_LM_l266_266259

theorem distance_midpoint_KN_to_LM :
  let K := (0 : ℝ, 0 : ℝ),
      L := (4 : ℝ, 0 : ℝ),
      M := (4 : ℝ, 10 : ℝ),
      N := (0 : ℝ, 12 : ℝ),
      midKN := ((0 + 0) / 2, (0 + 12) / 2),
      LM := set.univ.prod {p : ℝ × ℝ | p.1 = 4}
  in dist (midKN.1, midKN.2) (4, midKN.2) = 6.87 :=
  sorry

end distance_midpoint_KN_to_LM_l266_266259


namespace smallest_time_four_horses_meet_l266_266335

open Nat

def horse_lap_times : List ℕ := List.range 12 |>.map (λ k => k + 1)

theorem smallest_time_four_horses_meet (T : ℕ) :
  (∃ hs : Finset ℕ, hs.card = 4 ∧ ∀ h ∈ hs, h ∈ Finset.univ ∧ h + 1 ∣ T) ∧
  (∀ h ∉ hs, h + 1 = T) :=
  T = 12 :=
begin
  sorry
end

end smallest_time_four_horses_meet_l266_266335


namespace find_n_for_perfect_square_l266_266078

theorem find_n_for_perfect_square (n : ℕ) : n = 12 → ∃ k : ℕ, k * k = 2^8 + 2^{11} + 2^n :=
by {
  intro h,
  rw [h],
  use 80,
  norm_num,
  exact trivial,
}

end find_n_for_perfect_square_l266_266078


namespace hyperbola_foci_y_axis_condition_l266_266006

theorem hyperbola_foci_y_axis_condition (m n : ℝ) (h : m * n < 0) : 
  (mx^2 + ny^2 = 1) →
  (m < 0 ∧ n > 0) :=
sorry

end hyperbola_foci_y_axis_condition_l266_266006


namespace count_sixth_powers_less_than_1000_l266_266742

theorem count_sixth_powers_less_than_1000 : 
  {n : ℕ | 0 < n ∧ n < 1000 ∧ (∃ k : ℕ, n = k^6)}.to_finset.card = 3 :=
by 
  sorry

end count_sixth_powers_less_than_1000_l266_266742


namespace determine_x_l266_266164

def vector (α : Type) := α × α

variable {α : Type}
variables (a b : vector ℝ) (x : ℝ)

def vec_a : vector ℝ := (1, 1)
def vec_b : vector ℝ := (2, x)

def add_vectors (u v : vector ℝ) : vector ℝ :=
(u.1 + v.1, u.2 + v.2)

def scalar_mul_vector (k : ℝ) (u : vector ℝ) : vector ℝ :=
(k * u.1, k * u.2)

def is_parallel (u v : vector ℝ) : Prop :=
∃ k : ℝ, u = scalar_mul_vector k v

theorem determine_x 
  (h_parallel : is_parallel (add_vectors vec_a vec_b) (add_vectors (scalar_mul_vector 4 vec_b) (scalar_mul_vector (-2) vec_a))) :
  x = 2 :=
sorry

end determine_x_l266_266164


namespace min_vertical_segment_length_l266_266313

noncomputable def min_segment_length : ℝ := Real.Inf (Set.Range (λ x : ℝ, abs (x + abs x^2 + 2 * x - 1)))

theorem min_vertical_segment_length :
  min_segment_length = 3 / 4 :=
sorry

end min_vertical_segment_length_l266_266313


namespace count_squares_and_cubes_less_than_1000_l266_266662

theorem count_squares_and_cubes_less_than_1000 : 
  {n : ℕ // n^6 < 1000}.card = 3 :=
by
  sorry

end count_squares_and_cubes_less_than_1000_l266_266662


namespace number_of_sixth_powers_lt_1000_l266_266699

-- Define the condition for being a sixth power and being less than 1000
def isSixthPower (n : ℕ) : Prop := ∃ k : ℕ, n = k^6

-- Define the main statement to be proved
theorem number_of_sixth_powers_lt_1000 : 
  {n : ℕ | n < 1000 ∧ isSixthPower n}.to_finset.card = 3 :=
by
  sorry

end number_of_sixth_powers_lt_1000_l266_266699


namespace minimum_diagonal_pairs_l266_266041

theorem minimum_diagonal_pairs {n : ℕ} (h : n = 5784) :
  let total_cells := (n * (n + 1)) / 2 in
  ∃ m, m = 2892 ∧
    ∀ cell_pairs,
    (∀ pair ∈ cell_pairs, adjacent pair) ∧
    (∀ pair ∈ cell_pairs, in_same_row pair → false) →
    count_diagonal_pairs cell_pairs ≥ m :=
by
  sorry

end minimum_diagonal_pairs_l266_266041


namespace center_of_circle_with_conditions_l266_266870

theorem center_of_circle_with_conditions (α : ℝ) 
  (h1 : ∀ x y : ℝ, x^2 * Real.cos α - y^2 * Real.sin α + 2 = 0 → Real.cos α * Real.sin α > 0) :
  let center := (-Real.cos α, Real.sin α) in
  (center.1 <= 0 ∧ center.2 >= 0) ∨ (center.1 <= 0 ∧ center.2 <= 0) :=
by
  sorry

end center_of_circle_with_conditions_l266_266870


namespace clerical_percentage_l266_266984

theorem clerical_percentage (e : ℕ) (f : ℚ) (r : ℚ) (h_e : e = 3600) (h_f : f = 1/6) (h_r : r = 1/4) :
  ((f * e - r * (f * e)) / (e - r * (f * e))) * 100 ≈ 13.04 :=
by
  sorry

end clerical_percentage_l266_266984


namespace ratio_boys_to_girls_l266_266879

-- Define the given conditions
def G : ℕ := 300
def T : ℕ := 780

-- State the proposition to be proven
theorem ratio_boys_to_girls (B : ℕ) (h : B + G = T) : B / G = 8 / 5 :=
by
  -- Proof placeholder
  sorry

end ratio_boys_to_girls_l266_266879


namespace net_profit_calculation_l266_266986

-- Definitions of given conditions
def revenue_tough_week : ℕ := 800
def revenue_good_week : ℕ := 2 * revenue_tough_week
def discount_tough_week : ℝ := 0.10
def tax_rate : ℝ := 0.05
def transportation_cost_per_week : ℕ := 50
def good_weeks : ℕ := 5
def tough_weeks : ℕ := 3

-- Definitions from the conditions
def net_revenue_tough_week : ℕ := revenue_tough_week - (revenue_tough_week * discount_tough_week).to_nat
def revenue_good_weeks : ℕ := good_weeks * revenue_good_week
def revenue_tough_weeks : ℕ := tough_weeks * net_revenue_tough_week
def total_revenue : ℕ := revenue_good_weeks + revenue_tough_weeks

def total_tax_good_weeks : ℕ := (revenue_good_weeks * tax_rate).to_nat
def total_tax_tough_weeks : ℕ := (revenue_tough_weeks * tax_rate).to_nat
def total_tax : ℕ := total_tax_good_weeks + total_tax_tough_weeks

def total_transportation_costs : ℕ := (good_weeks + tough_weeks) * transportation_cost_per_week

noncomputable def net_profit : ℕ := total_revenue - total_tax - total_transportation_costs

-- The goal to prove
theorem net_profit_calculation : net_profit = 9252 := by
  sorry

end net_profit_calculation_l266_266986


namespace sum_of_consecutive_integers_l266_266362

theorem sum_of_consecutive_integers (n : ℕ) (h : n * (n + 1) / 2 ≤ 500) :
  n = 31 ∨ ((32 * (32 + 1) / 2) ≤ 500) := 
sorry

end sum_of_consecutive_integers_l266_266362


namespace both_square_and_cube_count_both_square_cube_less_than_1000_l266_266778

theorem both_square_and_cube (n : ℕ) (h_positive : n > 0) (h_bound : n < 1000) (h_square : ∃ k : ℕ, n = k^2) (h_cube : ∃ k : ℕ, n = k^3) :
  n ∈ {1, 64, 729} :=
sorry

theorem count_both_square_cube_less_than_1000 :
  {n : ℕ | n > 0 ∧ n < 1000 ∧ ∃ k : ℕ, n = k^2 ∧ ∃ k : ℕ, n = k^3}.to_finset.card = 3 :=
by
  have h_valid : ∀ n ∈ {1, 64, 729}, n > 0 ∧ n < 1000 ∧ (∃ k : ℕ, n = k^2) ∧ (∃ k : ℕ, n = k^3) := sorry
  have h_reverse : ∀ n, n > 0 ∧ n < 1000 ∧ (∃ k : ℕ, n = k^2) ∧ (∃ k : ℕ, n = k^3) → n ∈ {1, 64, 729} := sorry
  exact Finset.card_map _ {1, 64, 729}.to_finset
  sorry

end both_square_and_cube_count_both_square_cube_less_than_1000_l266_266778


namespace probability_absolute_difference_l266_266995

noncomputable def fair_coin_flip : MeasureSpace ℝ := sorry
noncomputable def uniform_random_variable : MeasureSpace ℝ := sorry

variables (x y : ℝ) (hx : x ∈ set.Icc 0 1) (hy : y ∈ set.Icc 0 1)

theorem probability_absolute_difference :
  (MeasureTheory.Measure.prob (λ x y, |x - y| > 1/3) 
    [fair_coin_flip, uniform_random_variable, uniform_random_variable, fair_coin_flip]) = 5/9 :=
sorry

end probability_absolute_difference_l266_266995


namespace max_consecutive_sum_leq_500_l266_266376

theorem max_consecutive_sum_leq_500 : ∃ n : ℕ, (∑ i in finset.range (n+1), i) ≤ 500 ∧ (∀ m : ℕ, m > n → (∑ i in finset.range (m+1), i) > 500) :=
sorry

end max_consecutive_sum_leq_500_l266_266376


namespace count_squares_and_cubes_less_than_1000_l266_266671

theorem count_squares_and_cubes_less_than_1000 : 
  {n : ℕ // n^6 < 1000}.card = 3 :=
by
  sorry

end count_squares_and_cubes_less_than_1000_l266_266671


namespace count_positive_integers_square_and_cube_lt_1000_l266_266771

theorem count_positive_integers_square_and_cube_lt_1000 : 
  {n : ℕ | n < 1000 ∧ ∃ k : ℕ, n = k^6}.to_finset.card = 3 :=
by
  sorry

end count_positive_integers_square_and_cube_lt_1000_l266_266771


namespace principal_amount_l266_266405

theorem principal_amount 
  (R : ℝ) (SI : ℝ) (T : ℝ) (hR : R = 0.0833333333333334) (hSI : SI = 400) (hT : T = 4) : 
  let P := SI / (R * T) in P = 1200 :=
by
  let P := SI / (R * T)
  have hP : P = 1200 := sorry
  exact hP

end principal_amount_l266_266405


namespace unique_path_connected_implies_tree_l266_266992

variables (V : Type) [Fintype V] [DecidableEq V]
variables (G : SimpleGraph V)

/-- A tree is a connected graph with no cycles -/
def is_tree (G : SimpleGraph V) : Prop :=
  G.connected ∧ ∀ (x : V) (p : G.walk x x), p.length = 0

/-- A graph where any two nodes are connected by exactly one unique path -/
def unique_path_connected (G : SimpleGraph V) : Prop :=
  ∀ x y : V, ∃! p : G.walk x y, G.unique_walks_from x y p

theorem unique_path_connected_implies_tree (h : unique_path_connected G) : is_tree G :=
sorry

end unique_path_connected_implies_tree_l266_266992


namespace equivalent_single_discount_l266_266463

def original_price : ℝ := 50  -- Original price of the jacket
def first_discount : ℝ := 0.25  -- First discount rate
def second_discount : ℝ := 0.20  -- Second discount rate

theorem equivalent_single_discount :
  ∃ x : ℝ, (original_price * (1 - first_discount) * (1 - second_discount) = original_price * (1 - x) ∧ x = 0.40) :=
begin
  -- Proof details would go here
  sorry
end

end equivalent_single_discount_l266_266463


namespace mitigate_bank_profit_loss_l266_266429

-- We define the terminology and conditions described in the problem
def bank_suboptimal_cashback_behavior (customer_behavior: ℕ → ℕ) : Prop :=
  ∀ (i : ℕ), (customer_behavior i) = category_specific_cashback i ∧ is_financially_savvy customer_behavior

def category_specific_cashback (i : ℕ) : ℕ := -- categorical cashback logic
  sorry 

def is_financially_savvy (customer_behavior: ℕ → ℕ) : Prop :=
  ∃ (cards : List ℕ), ∀ (i : ℕ), customer_behavior i = max_cashback_for_category i cards

def max_cashback_for_category (i : ℕ) (cards : List ℕ) : ℕ := 
  -- Define the logic for max cashback possible using multiple cards
  sorry

def dynamic_cashback_rates (total_cashback: ℕ) : ℕ := 
  -- Logic for dynamic cashback rates decreasing as total cashback increases
  sorry

def categorical_caps_rotating (i : ℕ) (period: ℕ) : ℕ := 
  -- Logic for rotating cashback caps for categories
  sorry

-- Lean statement that banks can avoid problems due to financially savvy customer behavior
theorem mitigate_bank_profit_loss (customer_behavior: ℕ → ℕ)
  (H : bank_suboptimal_cashback_behavior customer_behavior)
  : (∃ f : ℕ → ℕ, f = dynamic_cashback_rates) ∨ (∃ g : ℕ → ℕ × ℕ, g = categorical_caps_rotating) :=
  sorry

end mitigate_bank_profit_loss_l266_266429
